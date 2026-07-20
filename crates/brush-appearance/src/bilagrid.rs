//! Host side of the per-view bilateral grids: backend trait + kernel
//! launches, Fusion dispatch, Burn autodiff ops and the `BilagridModel`
//! module holding the learned grids.

use brush_cube::{MainBackend, MainBackendBase};
use brush_render::burn_glue::{
    AutodiffMain, unwrap_ad_wgpu_float, wrap_ad_wgpu_float, wrap_wgpu_float,
};
use burn::{
    backend::{
        Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::FloatTensor,
    },
    module::{Module, Param},
    tensor::{DType, Device, Tensor, s},
};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use burn_fusion::Fusion;

use crate::bilagrid_kernels as kernels;
use crate::{CasAtomicAdd, HfAtomicAdd, alloc_zeros, contiguous, dispatch_custom};

/// Backend hooks for the bilateral-grid kernels.
pub trait BilagridOps<B: Backend> {
    /// Slice the `view_idx`-th grid of `grids` `[N, 12, L, H, W]` by `rgb`
    /// `[h, w, 3|4]`, returning the transformed image (alpha untouched).
    fn bilagrid_slice_fwd(
        grids: FloatTensor<B>,
        rgb: FloatTensor<B>,
        view_idx: usize,
    ) -> FloatTensor<B>;

    /// Returns `(dL/dgrids, dL/drgb)`. The grid gradient is full-size with
    /// only the active view's slice populated.
    fn bilagrid_slice_bwd(
        grids: FloatTensor<B>,
        rgb: FloatTensor<B>,
        v_out: FloatTensor<B>,
        view_idx: usize,
    ) -> (FloatTensor<B>, FloatTensor<B>);
}

/// `(n, channels, l, h, w)` of a `[N, C, L, H, W]` grid tensor.
pub(crate) fn grid_dims5<R: CubeRuntime>(grids: &CubeTensor<R>) -> (u32, u32, u32, u32, u32) {
    let dims = grids.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 5, "grids must be [N, C, L, H, W]");
    // Dims of 1 would divide by zero in the interpolation / TV normalisation.
    assert!(
        dims[2] >= 2 && dims[3] >= 2 && dims[4] >= 2,
        "bilateral grid dims must each be >= 2 (got {}x{}x{})",
        dims[4],
        dims[3],
        dims[2],
    );
    (
        dims[0] as u32,
        dims[1] as u32,
        dims[2] as u32,
        dims[3] as u32,
        dims[4] as u32,
    )
}

/// Affine-grid dims: asserts the 12-coefficient payload.
fn grid_dims<R: CubeRuntime>(grids: &CubeTensor<R>) -> (u32, u32, u32, u32) {
    let (n, c, gl, gh, gw) = grid_dims5(grids);
    assert_eq!(c, 12, "affine grids must be [N, 12, L, H, W]");
    (n, gl, gh, gw)
}

fn img_dims<R: CubeRuntime>(rgb: &CubeTensor<R>) -> (u32, u32, u32) {
    let dims = rgb.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "rgb must be [h, w, c]");
    let ch = dims[2] as u32;
    assert!(ch == 3 || ch == 4, "rgb must have 3 or 4 channels");
    (dims[0] as u32, dims[1] as u32, ch)
}

fn launch_slice_fwd<R: CubeRuntime>(
    grids: CubeTensor<R>,
    rgb: CubeTensor<R>,
    view_idx: usize,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};

    let grids = contiguous(grids);
    let rgb = contiguous(rgb);
    let (n, gl, gh, gw) = grid_dims(&grids);
    let (h, w, ch) = img_dims(&rgb);
    assert!((view_idx as u32) < n, "view index out of range");

    let out = alloc_zeros(&rgb, rgb.shape(), DType::F32);
    let client = rgb.client.clone();
    kernels::bilagrid_slice_fwd_kernel::launch::<R>(
        &client,
        CubeCount::Static((h * w).div_ceil(kernels::BLOCK_SIZE), 1, 1),
        CubeDim::new_1d(kernels::BLOCK_SIZE),
        grids.into_tensor_arg(),
        rgb.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        gl,
        gh,
        gw,
        h,
        w,
        view_idx as u32 * 12 * gl * gh * gw,
        ch,
        ch == 4,
    );
    out
}

fn launch_slice_bwd<R: CubeRuntime>(
    grids: CubeTensor<R>,
    rgb: CubeTensor<R>,
    v_out: CubeTensor<R>,
    view_idx: usize,
) -> (CubeTensor<R>, CubeTensor<R>) {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};

    let grids = contiguous(grids);
    let rgb = contiguous(rgb);
    let v_out = contiguous(v_out);
    let (n, gl, gh, gw) = grid_dims(&grids);
    let (h, w, ch) = img_dims(&rgb);
    assert!((view_idx as u32) < n, "view index out of range");

    let grad_grids = alloc_zeros(&grids, grids.shape(), DType::F32);
    let grad_rgb = alloc_zeros(&rgb, rgb.shape(), DType::F32);
    let client = rgb.client.clone();

    let cube_count = CubeCount::Static((h * w).div_ceil(kernels::BLOCK_SIZE), 1, 1);
    let cube_dim = CubeDim::new_1d(kernels::BLOCK_SIZE);
    let grid_offset = view_idx as u32 * 12 * gl * gh * gw;
    if brush_cube::supports_float_atomics::<R>(&client) {
        kernels::bilagrid_slice_bwd_kernel::launch::<HfAtomicAdd, R>(
            &client,
            cube_count,
            cube_dim,
            grids.into_tensor_arg(),
            rgb.into_tensor_arg(),
            v_out.into_tensor_arg(),
            grad_grids.clone().into_tensor_arg(),
            grad_rgb.clone().into_tensor_arg(),
            gl,
            gh,
            gw,
            h,
            w,
            grid_offset,
            ch,
            ch == 4,
        );
    } else {
        kernels::bilagrid_slice_bwd_kernel::launch::<CasAtomicAdd, R>(
            &client,
            cube_count,
            cube_dim,
            grids.into_tensor_arg(),
            rgb.into_tensor_arg(),
            v_out.into_tensor_arg(),
            grad_grids.clone().into_tensor_arg(),
            grad_rgb.clone().into_tensor_arg(),
            gl,
            gh,
            gw,
            h,
            w,
            grid_offset,
            ch,
            ch == 4,
        );
    }
    #[allow(clippy::tuple_array_conversions)]
    (grad_grids, grad_rgb)
}

impl BilagridOps<Self> for MainBackendBase {
    fn bilagrid_slice_fwd(
        grids: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        view_idx: usize,
    ) -> FloatTensor<Self> {
        launch_slice_fwd(grids, rgb, view_idx)
    }

    fn bilagrid_slice_bwd(
        grids: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        v_out: FloatTensor<Self>,
        view_idx: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        launch_slice_bwd(grids, rgb, v_out, view_idx)
    }
}

impl BilagridOps<Self> for Fusion<MainBackendBase> {
    fn bilagrid_slice_fwd(
        grids: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        view_idx: usize,
    ) -> FloatTensor<Self> {
        let shape = rgb.shape();
        let [out] = dispatch_custom(
            "bilagrid_slice_fwd",
            [grids, rgb],
            [(shape, DType::F32)],
            move |desc, h| {
                let ([grids, rgb], [out]) = desc.as_fixed();
                let res = MainBackendBase::bilagrid_slice_fwd(
                    h.get_float_tensor::<MainBackendBase>(grids),
                    h.get_float_tensor::<MainBackendBase>(rgb),
                    view_idx,
                );
                h.register_float_tensor::<MainBackendBase>(&out.id, res);
            },
        );
        out
    }

    fn bilagrid_slice_bwd(
        grids: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        v_out: FloatTensor<Self>,
        view_idx: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let grids_shape = grids.shape();
        let rgb_shape = rgb.shape();
        let [grad_grids, grad_rgb] = dispatch_custom(
            "bilagrid_slice_bwd",
            [grids, rgb, v_out],
            [(grids_shape, DType::F32), (rgb_shape, DType::F32)],
            move |desc, h| {
                let ([grids, rgb, v_out], [grad_grids, grad_rgb]) = desc.as_fixed();
                let (gg, gr) = MainBackendBase::bilagrid_slice_bwd(
                    h.get_float_tensor::<MainBackendBase>(grids),
                    h.get_float_tensor::<MainBackendBase>(rgb),
                    h.get_float_tensor::<MainBackendBase>(v_out),
                    view_idx,
                );
                h.register_float_tensor::<MainBackendBase>(&grad_grids.id, gg);
                h.register_float_tensor::<MainBackendBase>(&grad_rgb.id, gr);
            },
        );
        #[allow(clippy::tuple_array_conversions)]
        (grad_grids, grad_rgb)
    }
}

// ---------------------------------------------------------------------------
// Autodiff ops
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct BilagridSliceBackward;

#[derive(Debug, Clone)]
struct BilagridSliceState<B: Backend> {
    grids: FloatTensor<B>,
    rgb: FloatTensor<B>,
    view_idx: usize,
}

impl<B: Backend + BilagridOps<B>> Backward<B, 2> for BilagridSliceBackward {
    type State = BilagridSliceState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let v_out = grads.consume::<B>(&ops.node);
        let [grids_parent, rgb_parent] = ops.parents;
        let (grad_grids, grad_rgb) =
            B::bilagrid_slice_bwd(state.grids, state.rgb, v_out, state.view_idx);
        if let Some(node) = grids_parent {
            grads.register::<B>(node.id, grad_grids);
        }
        if let Some(node) = rgb_parent {
            grads.register::<B>(node.id, grad_rgb);
        }
    }
}

/// Apply the `view_idx`-th bilateral grid to a rendered `[H, W, 3|4]` image.
/// Differentiable w.r.t. both `grids` and `rgb`; the alpha channel (if any)
/// passes through untouched. Inputs must be on an autodiff-enabled Wgpu
/// device.
pub fn bilagrid_apply(grids: Tensor<5>, rgb: Tensor<3>, view_idx: usize) -> Tensor<3> {
    let grids_ad = unwrap_ad_wgpu_float(grids);
    let rgb_ad = unwrap_ad_wgpu_float(rgb);

    let prep = BilagridSliceBackward
        .prepare::<NoCheckpointing>([grids_ad.node.clone(), rgb_ad.node.clone()])
        .compute_bound()
        .stateful();

    let grids_p = grids_ad.primitive;
    let rgb_p = rgb_ad.primitive;
    let out = <MainBackend as BilagridOps<MainBackend>>::bilagrid_slice_fwd(
        grids_p.clone(),
        rgb_p.clone(),
        view_idx,
    );

    let out_ad: FloatTensor<AutodiffMain> = match prep {
        OpsKind::Tracked(prep) => prep.finish(
            BilagridSliceState {
                grids: grids_p,
                rgb: rgb_p,
                view_idx,
            },
            out,
        ),
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    wrap_ad_wgpu_float::<3>(out_ad)
}

/// Forward-only version of [`bilagrid_apply`] for non-autodiff tensors
/// (e.g. eval-time visualisation).
pub fn bilagrid_apply_inner(grids: Tensor<5>, rgb: Tensor<3>, view_idx: usize) -> Tensor<3> {
    use brush_render::burn_glue::unwrap_wgpu_float;
    let grids_p = unwrap_wgpu_float(grids);
    let rgb_p = unwrap_wgpu_float(rgb);
    let out =
        <MainBackend as BilagridOps<MainBackend>>::bilagrid_slice_fwd(grids_p, rgb_p, view_idx);
    wrap_wgpu_float::<3>(out)
}

/// Total-variation regulariser over all views' grids. Returns a `[1]`
/// tensor; differentiable w.r.t. `grids`.
pub fn bilagrid_tv_loss(grids: Tensor<5>) -> Tensor<1> {
    let [_, _, guidance, grid_y, grid_x] = grids.dims();
    let dx = grids.clone().slice(s![.., .., .., .., 1..grid_x])
        - grids.clone().slice(s![.., .., .., .., 0..grid_x - 1]);
    let dy = grids.clone().slice(s![.., .., .., 1..grid_y, ..])
        - grids.clone().slice(s![.., .., .., 0..grid_y - 1, ..]);
    let dz = grids.clone().slice(s![.., .., 1..guidance, .., ..])
        - grids.slice(s![.., .., 0..guidance - 1, .., ..]);
    dx.powi_scalar(2).mean() + dy.powi_scalar(2).mean() + dz.powi_scalar(2).mean()
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// One 3D bilateral grid per training view, initialised to the identity
/// affine transform.
#[derive(Module, Debug)]
pub struct BilagridModel {
    /// `[num_views, 12, guidance, grid_y, grid_x]`.
    pub grids: Param<Tensor<5>>,
}

impl BilagridModel {
    pub fn new(
        num_views: usize,
        grid_x: usize,
        grid_y: usize,
        grid_guidance: usize,
        device: &Device,
    ) -> Self {
        // Identity 3x4 affine: coefficient channels 0, 5, 10 (the diagonal
        // of the 3x3 part) are 1, everything else 0.
        let mut grids = Tensor::zeros([num_views, 12, grid_guidance, grid_y, grid_x], device);
        let ones = Tensor::ones([num_views, 1, grid_guidance, grid_y, grid_x], device);
        for c in [0, 5, 10] {
            grids = grids.slice_assign(s![.., c..c + 1], ones.clone());
        }
        Self {
            grids: Param::from_tensor(grids),
        }
    }

    /// Apply this view's grid to a rendered `[H, W, 3|4]` image.
    pub fn apply(&self, img: Tensor<3>, view_idx: usize) -> Tensor<3> {
        bilagrid_apply(self.grids.val(), img, view_idx)
    }

    /// Total-variation regulariser (unweighted).
    pub fn tv_loss(&self) -> Tensor<1> {
        bilagrid_tv_loss(self.grids.val())
    }
}
