//! Per-view appearance compensation for Brush training.
//!
//! Two independently toggleable models, both applied to the *rendered* image
//! before the photometric loss so the splats themselves learn canonical
//! (appearance-free) colors:
//!
//! - [`bilagrid`]: per-view 3D bilateral grids ("Bilateral Guided Radiance
//!   Field Processing", `BilaRF`). Each training view owns a `[12, L, H, W]`
//!   grid of 3x4 affine color transforms, sliced per pixel by screen position
//!   and grayscale guidance. The slicing behavior follows gsplat's Apache-2.0
//!   reference (`grid_sample` with aligned corners and border padding).
//!
//! - [`ppisp`]: physically-plausible ISP compensation (NVIDIA PPISP). Models
//!   per-frame exposure + color homography and per-camera vignetting + CRF
//!   with a handful of parameters each, applied per pixel by a fused kernel.
//!   The parameter regularisation is plain tensor ops (the params are tiny).
//!
//! Both follow the `brush-loss` pattern: a backend trait implemented for the
//! raw `CubeCL` backend and the Fusion backend, plus custom Burn autodiff ops
//! so gradients flow to both the appearance params and the rendered image.

pub mod bilagrid;
mod bilagrid_kernels;
pub mod ppisp;
mod ppisp_kernels;
mod ppisp_math;
pub mod train_state;

use burn::backend::wgpu::WgpuRuntime;
use burn::tensor::{DType, Shape};
use burn_cubecl::{CubeRuntime, fusion::FusionCubeRuntime, tensor::CubeTensor};
use burn_fusion::{
    FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

pub(crate) use brush_cube::{AtomicAddF32, CasAtomicAdd, HfAtomicAdd};

pub(crate) fn alloc_zeros<R: CubeRuntime>(
    template: &CubeTensor<R>,
    shape: Shape,
    dtype: DType,
) -> CubeTensor<R> {
    burn_cubecl::ops::numeric::zeros_client::<R>(
        template.client.clone(),
        template.device.clone(),
        shape,
        dtype,
    )
}

/// Wraps a closure as a fusion `Operation` (same pattern as `brush-loss`).
struct ClosureOp<F> {
    desc: CustomOpIr,
    op: F,
}

impl<F> std::fmt::Debug for ClosureOp<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClosureOp({:?})", self.desc)
    }
}

impl<F> Operation<FusionCubeRuntime<WgpuRuntime>> for ClosureOp<F>
where
    F: Fn(&CustomOpIr, &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>)
        + Send
        + Sync
        + 'static,
{
    fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        (self.op)(&self.desc, h);
    }
}

pub(crate) type FusionTensor = burn_fusion::FusionTensor<FusionCubeRuntime<WgpuRuntime>>;

/// Register a custom op with `N` inputs and `M` outputs on the Fusion
/// stream. Generalises `brush-loss`'s single-output helper: each output is
/// described by `(shape, dtype)`; `op` runs against the inner backend when
/// fusion executes the queued op.
pub(crate) fn dispatch_custom<const N: usize, const M: usize, F>(
    name: &'static str,
    inputs: [FusionTensor; N],
    outputs: [(Shape, DType); M],
    op: F,
) -> [FusionTensor; M]
where
    F: Fn(&CustomOpIr, &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>)
        + Send
        + Sync
        + 'static,
{
    let client = inputs[0].client.clone();
    let outs =
        outputs.map(|(shape, dtype)| TensorIr::uninit(client.create_empty_handle(), shape, dtype));
    let stream = StreamId::current();
    let desc = CustomOpIr::new(name, &inputs.map(|t| t.into_ir()), &outs);
    let wrapped = ClosureOp {
        desc: desc.clone(),
        op,
    };
    client
        .register(stream, OperationIr::Custom(desc), wrapped)
        .outputs()
}

/// Resolve a possibly-fused float tensor into a contiguous `CubeTensor`.
pub(crate) fn contiguous<R: CubeRuntime>(t: CubeTensor<R>) -> CubeTensor<R> {
    burn_cubecl::kernel::into_contiguous(t)
}

pub use bilagrid::{BilagridModel, bilagrid_apply, bilagrid_tv_loss};
pub use ppisp::{PpispModel, PpispStages, ppisp_apply};
pub use train_state::{ActiveAppearance, AppearanceTrainState};

/// Static configuration for the appearance models.
#[derive(Debug, Clone)]
pub struct AppearanceConfig {
    /// Per-view affine bilateral grids.
    pub bilagrid: bool,
    /// Full per-frame PPISP.
    pub ppisp: bool,
    /// Grid dims `(x, y, guidance)` — spatial width/height and the
    /// grayscale guidance dimension.
    pub bilagrid_dims: (usize, usize, usize),
    /// Weight of the grid total-variation regulariser.
    pub bilagrid_tv_weight: f32,
    /// Grid learning rate (warmup + exponential decay applied).
    pub bilagrid_lr: f64,
    /// Adam betas for the sparse per-view grid updates.
    pub bilagrid_betas: (f64, f64),
    /// PPISP learning rate (warmup + exponential decay applied).
    pub ppisp_lr: f64,
    /// Scale on all PPISP regularisation terms.
    pub ppisp_reg_scale: f32,
}

impl Default for AppearanceConfig {
    fn default() -> Self {
        Self {
            bilagrid: false,
            ppisp: false,
            bilagrid_dims: (16, 16, 8),
            bilagrid_tv_weight: 10.0,
            bilagrid_lr: 2e-3,
            bilagrid_betas: (0.9, 0.999),
            ppisp_lr: 2e-3,
            ppisp_reg_scale: 1.0,
        }
    }
}

/// Warmup + exponential-decay LR schedule used for both appearance models:
/// linear warmup
/// from `start_factor * base` over `warmup_steps`, then exponential decay
/// toward `final_factor * base` at `decay_steps`.
pub fn warmup_exp_lr(
    step: u32,
    base: f64,
    warmup_steps: u32,
    start_factor: f64,
    final_factor: f64,
    decay_steps: u32,
) -> f64 {
    if step < warmup_steps {
        let t = (step as f64 + 1.0) / warmup_steps as f64;
        base * (start_factor + (1.0 - start_factor) * t)
    } else {
        let decay_step = (step - warmup_steps) as f64;
        base * final_factor.powf(decay_step / decay_steps.max(1) as f64)
    }
}
