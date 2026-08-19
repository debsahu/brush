#![allow(clippy::match_wildcard_for_single_variants)]

use crate::burn_glue::{
    AutodiffMain, lift_to_autodiff, unwrap_ad_wgpu_float, wrap_ad_wgpu_float, wrap_wgpu_float,
    wrap_wgpu_int,
};
use crate::{
    SplatOps,
    camera::Camera,
    gaussian_splats::{Rasterizer, SplatRenderMode, Splats, fold_min_scale},
    sh::sh_coeffs_for_degree,
    shaders::helpers::ProjectUniforms,
};
use brush_cube::{MainBackend, MainBackendBase};
use burn::{
    backend::{
        AutodiffBackend, Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        ops::FloatTensorOps,
        tensor::{FloatTensor, IntTensor},
        wgpu::WgpuRuntime,
    },
    module::Param,
    tensor::{DType, FloatDType, Gradients as TensorGradients, Int, Shape, Tensor},
};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use glam::Vec3;

fn training_rasterizer() -> Rasterizer {
    use std::sync::OnceLock;

    static SELECTED: OnceLock<Rasterizer> = OnceLock::new();
    *SELECTED.get_or_init(|| {
        if crate::native_msl::fine_raster_tiles_requested() {
            tracing::info!("16x8 training raster tiles enabled");
            Rasterizer::Candidate
        } else {
            Rasterizer::Legacy
        }
    })
}

/// Intermediate gradients from the rasterize backward pass.
///
/// Sparse buffer of shape `[num_visible, COMPACT_GRAD_LANES]`, indexed by
/// `compact_gid`. Lanes: 0..=1 xy, 2..=4 conic, 5..=7 rgb, 8 alpha,
/// 9 refine-weight, 10 expected-depth, 11..=14 PGSR plane-aux values
/// (see `bwd::kernels::rasterize_backwards`).
#[derive(Debug, Clone)]
pub(crate) struct RasterizeGrads<B: Backend> {
    pub v_combined: FloatTensor<B>,
}

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone)]
pub(crate) struct SplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Projection gradients when SH coefficient materialization is deferred to
/// the optimizer. The other model gradients remain dense and unchanged.
#[derive(Debug, Clone)]
pub(crate) struct DeferredSplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Concrete backward kernels behind [`SplatOps::render`].
///
/// Deliberately not `: SplatOps`. This is the set of backends that have
/// backward kernels, which is smaller than the set you can render on:
/// `AutodiffMain` and the generated `Dispatch` impl `SplatOps` but have no
/// meaningful `rasterize_bwd`, since these run on concrete tensors and are
/// called from the `Backward` impl on the inner backend.
pub(crate) trait SplatBwdOps: Backend {
    /// Backward pass for rasterization.
    /// Returns sparse `v_combined` [`num_visible`, `COMPACT_GRAD_LANES`] indexed by `compact_gid`.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        plane_aux: Option<FloatTensor<Self>>,
        global_from_compact_gid: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
        render_depth: bool,
        render_plane: bool,
    ) -> RasterizeGrads<Self>;

    /// Specialized raster backward which may omit the refinement-only
    /// statistic. Backends that do not override this retain the compatible
    /// full-gradient behavior.
    #[allow(clippy::fn_params_excessive_bools)]
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd_with_refine_weight(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        plane_aux: Option<FloatTensor<Self>>,
        global_from_compact_gid: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
        _compute_refine_weight: bool,
        render_depth: bool,
        render_plane: bool,
    ) -> RasterizeGrads<Self> {
        Self::rasterize_bwd(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            plane_aux,
            global_from_compact_gid,
            background,
            img_size,
            v_output,
            smooth_cutoff,
            render_depth,
            render_plane,
        )
    }

    /// Backward pass for projection.
    /// Reads sparse `v_combined` [`num_visible`, 9], writes dense outputs (scatter in kernel).
    /// `sh_coeffs` is the original (input) SH coefficient tensor — needed
    /// so the kernel can backprop `v_color` through the SH basis to the
    /// view direction and then to the mean.
    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self>;

    /// Projection backward without allocating or writing a dense SH
    /// coefficient gradient. Used only by the private training bridge after
    /// optimizer compatibility has been checked.
    #[allow(clippy::too_many_arguments)]
    fn project_bwd_deferred_sh(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> DeferredSplatGrads<Self> {
        // Preserve compatibility for external backend implementations. The
        // built-in Wgpu paths override this to avoid the dense coefficient
        // allocation; other backends may compute and discard it safely.
        let grads = Self::project_bwd(
            transforms,
            sh_coeffs,
            raw_opac,
            global_from_compact_gid,
            project_uniforms,
            render_mode,
            v_combined,
        );
        DeferredSplatGrads {
            v_transforms: grads.v_transforms,
            v_raw_opac: grads.v_raw_opac,
            v_refine_weight: grads.v_refine_weight,
        }
    }
}

/// Raster inputs saved by this crate's forward pass.
///
/// The fields and constructor stay private to this module so downstream safe
/// callers cannot claim the invariants required by the unchecked native-MSL
/// launch. Backend implementations can only consume a value produced by the
/// private autodiff bridge below.
pub(crate) struct ForwardRasterBackward<B: Backend> {
    out_img: FloatTensor<B>,
    projected_splats: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,
    plane_aux: Option<FloatTensor<B>>,
    global_from_compact_gid: IntTensor<B>,
    background: Vec3,
    img_size: glam::UVec2,
    v_output: FloatTensor<B>,
    rasterizer: Rasterizer,
    smooth_cutoff: bool,
    compute_refine_weight: bool,
    render_depth: bool,
    render_plane: bool,
}

impl<B: Backend> ForwardRasterBackward<B> {
    #[allow(clippy::fn_params_excessive_bools)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        out_img: FloatTensor<B>,
        projected_splats: FloatTensor<B>,
        compact_gid_from_isect: IntTensor<B>,
        tile_offsets: IntTensor<B>,
        plane_aux: Option<FloatTensor<B>>,
        global_from_compact_gid: IntTensor<B>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<B>,
        rasterizer: Rasterizer,
        smooth_cutoff: bool,
        compute_refine_weight: bool,
        render_depth: bool,
        render_plane: bool,
    ) -> Self {
        Self {
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            plane_aux,
            global_from_compact_gid,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
            render_depth,
            render_plane,
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        FloatTensor<B>,
        FloatTensor<B>,
        IntTensor<B>,
        IntTensor<B>,
        Option<FloatTensor<B>>,
        IntTensor<B>,
        Vec3,
        glam::UVec2,
        FloatTensor<B>,
        Rasterizer,
        bool,
        bool,
        bool,
        bool,
    ) {
        (
            self.out_img,
            self.projected_splats,
            self.compact_gid_from_isect,
            self.tile_offsets,
            self.plane_aux,
            self.global_from_compact_gid,
            self.background,
            self.img_size,
            self.v_output,
            self.rasterizer,
            self.smooth_cutoff,
            self.compute_refine_weight,
            self.render_depth,
            self.render_plane,
        )
    }
}

/// Crate-private bridge for backward work that relies on forward-produced
/// tensor relationships. This is deliberately separate from [`SplatBwdOps`],
/// whose safe public methods always retain kernel bounds checks.
pub(crate) trait InternalSplatBwdOps: SplatBwdOps {
    fn rasterize_bwd_from_forward(input: ForwardRasterBackward<Self>) -> RasterizeGrads<Self>;
}

/// State saved during forward pass for backward computation.
#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    sh_coeffs: FloatTensor<B>,
    raw_opacity: FloatTensor<B>,

    projected_splats: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
    global_from_compact_gid: IntTensor<B>,

    out_img: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,

    /// PGSR plane-auxiliary per-splat channels `[N, PLANE_AUX_LANES]`, on the
    /// inner backend. `None` unless the rasterization mode composites them.
    plane_aux: Option<FloatTensor<B>>,

    render_mode: SplatRenderMode,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
    rasterization_mode: crate::gaussian_splats::RasterizationMode,
    background: Vec3,
    img_size: glam::UVec2,
}

#[derive(Debug)]
struct RenderBackwards;

/// Parents of the hand-rolled render op, in `prepare` order:
/// `transforms, refine_weight, sh_coeffs, raw_opacities, deferred_sh, plane_aux`.
/// The sixth (plane-aux) parent is what makes the PGSR plane channels a real
/// differentiable input rather than a read-only side channel.
const NUM_BWD_ARGS: usize = 6;

// Implement gradient registration when rendering backwards.
impl<B: Backend + InternalSplatBwdOps> Backward<B, NUM_BWD_ARGS> for RenderBackwards {
    type State = GaussianBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, NUM_BWD_ARGS>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_gaussians backwards").entered();

        let state = ops.state;
        let v_output = grads.consume::<B>(&ops.node);

        // Register gradients for parent nodes (This code is already skipped entirely
        // if no parent nodes require gradients).
        let [
            transforms_parent,
            refine_weight,
            coeffs_parent,
            raw_opacity_parent,
            deferred_sh_parent,
            plane_aux_parent,
        ] = ops.parents;
        let compute_refine_weight = refine_weight.is_some();
        let render_plane = state.rasterization_mode.render_plane();
        let global_from_compact_gid = state.global_from_compact_gid.clone();
        let num_visible = state.project_uniforms.num_visible as usize;
        let total_splats = state.project_uniforms.total_splats as usize;
        let device = state.transforms.device();

        let rasterize_grads = B::rasterize_bwd_from_forward(ForwardRasterBackward::new(
            state.out_img,
            state.projected_splats,
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.plane_aux,
            global_from_compact_gid.clone(),
            state.background,
            state.img_size,
            v_output,
            state.rasterizer,
            state.pass.smooth_cutoff(),
            compute_refine_weight,
            state.rasterization_mode.render_depth(),
            render_plane,
        ));

        // Scatter the compact plane-aux VALUE gradients back to the dense
        // `[N, PLANE_AUX_LANES]` parent. `global_from_compact_gid` is injective
        // over `0..num_visible` (one row per visible splat), so the index-add is
        // a plain scatter. Lane bounds come from the shared constants — never
        // restated — per the stride-drift history in `rasterize_backwards`.
        if let Some(node) = plane_aux_parent {
            let lanes = crate::kernels::helpers::PLANE_AUX_LANES as usize;
            let start = crate::bwd::kernels::rasterize_backwards::PLANE_GRAD_LANE_START as usize;
            let zeros = B::float_zeros([total_splats, lanes].into(), &device, FloatDType::F32);
            let v_aux = if render_plane && num_visible > 0 {
                let compact = B::float_slice(
                    rasterize_grads.v_combined.clone(),
                    &[(0..num_visible).into(), (start..start + lanes).into()],
                );
                let indices = B::int_slice(global_from_compact_gid, &[(0..num_visible).into()]);
                B::float_select_add(zeros, 0, indices, compact)
            } else {
                zeros
            };
            grads.register::<B>(node.id, v_aux);
        }

        if let Some(deferred_sh_parent) = deferred_sh_parent {
            let compact_sh_grads = rasterize_grads.v_combined.clone();
            let splat_grads = B::project_bwd_deferred_sh(
                state.transforms,
                state.sh_coeffs,
                state.raw_opacity,
                state.global_from_compact_gid,
                state.project_uniforms,
                state.render_mode,
                rasterize_grads.v_combined,
            );

            if let Some(node) = transforms_parent {
                grads.register::<B>(node.id, splat_grads.v_transforms);
            }
            if let Some(node) = refine_weight {
                grads.register::<B>(node.id, splat_grads.v_refine_weight);
            }
            if let Some(node) = raw_opacity_parent {
                grads.register::<B>(node.id, splat_grads.v_raw_opac);
            }
            grads.register::<B>(deferred_sh_parent.id, compact_sh_grads);
        } else {
            let splat_grads = B::project_bwd(
                state.transforms,
                state.sh_coeffs,
                state.raw_opacity,
                state.global_from_compact_gid,
                state.project_uniforms,
                state.render_mode,
                rasterize_grads.v_combined,
            );

            if let Some(node) = transforms_parent {
                grads.register::<B>(node.id, splat_grads.v_transforms);
            }
            if let Some(node) = refine_weight {
                grads.register::<B>(node.id, splat_grads.v_refine_weight);
            }
            if let Some(node) = coeffs_parent {
                grads.register::<B>(node.id, splat_grads.v_coeffs);
            }
            if let Some(node) = raw_opacity_parent {
                grads.register::<B>(node.id, splat_grads.v_raw_opac);
            }
        }
    }
}

/// Sparse SH-gradient payload extracted after the render backward pass.
/// Tensor clones in this payload are handle-only and preserve the exact
/// forward-time projection inputs.
#[doc(hidden)]
pub struct DeferredShGrad {
    pub compact_grads: Tensor<2>,
    pub render_transforms: Tensor<2>,
    pub global_from_compact_gid: Tensor<1, Int>,
    pub project_uniforms: ProjectUniforms,
}

/// Autodiff holder used to route compact raster gradients across Burn's
/// backward boundary without assigning them to the dense SH parameter.
#[doc(hidden)]
pub struct DeferredShGradHandle {
    holder: Tensor<2>,
    render_transforms: Tensor<2>,
    global_from_compact_gid: Tensor<1, Int>,
    project_uniforms: ProjectUniforms,
}

impl DeferredShGradHandle {
    pub fn take(self, grads: &mut TensorGradients) -> Option<DeferredShGrad> {
        Some(DeferredShGrad {
            compact_grads: self.holder.grad_remove(grads)?,
            render_transforms: self.render_transforms,
            global_from_compact_gid: self.global_from_compact_gid,
            project_uniforms: self.project_uniforms,
        })
    }
}

pub struct SplatOutputDiff {
    /// Rendered image, on the autodiff graph (this is what the loss backprops through).
    pub img: Tensor<3>,
    pub num_visible: u32,
    /// Per-splat visibility aux — on the **inner** backend (no gradients).
    pub visible: Tensor<1>,
    /// Per-splat max screen radius aux — on the **inner** backend (no gradients).
    pub max_radius: Tensor<1>,
    pub refine_weight_holder: Tensor<1>,
}

/// Private cross-crate training protocol. Keeping this separate preserves the
/// public [`SplatOutputDiff`] layout for downstream struct construction and
/// exhaustive destructuring.
#[doc(hidden)]
pub struct TrainingSplatOutputDiff {
    pub img: Tensor<3>,
    pub num_visible: u32,
    pub visible: Tensor<1>,
    pub max_radius: Tensor<1>,
    pub refine_weight_holder: Tensor<1>,
    pub deferred_sh_grad: Option<DeferredShGradHandle>,
}

impl TrainingSplatOutputDiff {
    fn into_public(self) -> SplatOutputDiff {
        SplatOutputDiff {
            img: self.img,
            num_visible: self.num_visible,
            visible: self.visible,
            max_radius: self.max_radius,
            refine_weight_holder: self.refine_weight_holder,
        }
    }
}

/// Equivalent to `Module::train()` for [`Splats`], routing through
/// [`lift_to_autodiff`] so the autodiff `checkpointing` field is set. Use this
/// instead of `splats.train()` until upstream burn-dispatch fixes `from_inner`.
pub fn lift_splats_to_autodiff(splats: Splats) -> Splats {
    let mip = splats.render_mip;
    let min_scale = splats.min_scale.clone();
    let (transforms_id, transforms, _) = splats.transforms.consume();
    let (sh_coeffs_id, sh_coeffs, _) = splats.sh_coeffs.consume();
    let (raw_opacity_id, raw_opacity, _) = splats.raw_opacities.consume();
    Splats {
        transforms: Param::initialized(transforms_id, lift_to_autodiff(transforms).require_grad()),
        sh_coeffs: Param::initialized(sh_coeffs_id, lift_to_autodiff(sh_coeffs).require_grad()),
        raw_opacities: Param::initialized(
            raw_opacity_id,
            lift_to_autodiff(raw_opacity).require_grad(),
        ),
        render_mip: mip,
        // Keep the frozen floor on the inner backend. `#[module(skip)]` fields
        // aren't converted by `.valid()`, so lifting it here would leave an
        // autodiff `f` on an inner module after eval-strip and mix backends in
        // `scales()`/`opacities()`. The bwd render lifts a temporary copy.
        min_scale,
    }
}

/// Render splats on a differentiable device.
///
/// Panics if the device is not autodiff-enabled.
pub async fn render_splats(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> SplatOutputDiff {
    render_splats_with_refine_weight(splats, camera, img_size, background, true).await
}

/// Render splats on a differentiable device, optionally tracking the
/// refinement-only screen-space gradient statistic.
///
/// Model gradients and render auxiliaries are unchanged when
/// `compute_refine_weight` is false.
pub async fn render_splats_with_refine_weight(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    compute_refine_weight: bool,
) -> SplatOutputDiff {
    render_splats_with_pass_and_refine_weight(
        splats,
        camera,
        img_size,
        background,
        crate::gaussian_splats::RasterPass::Backward,
        Rasterizer::Legacy,
        compute_refine_weight,
        crate::gaussian_splats::RasterizationMode::Rgba,
        false,
        None,
    )
    .await
    .into_public()
}

/// Training-only render entry point that can route compact SH gradients to a
/// compatible optimizer. When `defer_sh_grad` is honored, backward does not
/// populate `splats.sh_coeffs.grad()`; the caller must extract and consume
/// `deferred_sh_grad` with that optimizer. Unsupported builds ignore the
/// request and preserve the dense coefficient gradient.
#[doc(hidden)]
pub async fn render_splats_for_training(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    compute_refine_weight: bool,
    rasterization_mode: crate::gaussian_splats::RasterizationMode,
    defer_sh_grad: bool,
) -> TrainingSplatOutputDiff {
    render_splats_for_training_with_plane_aux(
        splats,
        camera,
        img_size,
        background,
        compute_refine_weight,
        rasterization_mode,
        defer_sh_grad,
        None,
    )
    .await
}

/// Training render with the PGSR plane-auxiliary channels (approach B).
///
/// `plane_aux` is the on-tape `[N, 4]` output of
/// `brush_train::train::plane_features` — camera-frame normal + signed offset —
/// and must be `Some` exactly when `rasterization_mode` is
/// [`crate::gaussian_splats::RasterizationMode::RgbaDepthPlane`]. The channels
/// are composited by the MAIN rasterizer and appear in `img` at channels
/// `5..=8` (after rgba and centre depth); the alpha channel `3` is the coverage
/// term `plane_depth_from_features` expects as its fifth input.
///
/// Backward contract (plan section 4.5, row 3): unlike the centre-depth channel,
/// the plane channels' error DOES reach the blending weights — opacity, conic
/// and screen-space position — by design. See `plane_channel_bwd` in
/// `bwd::kernels::rasterize_backwards`.
///
/// [`render_splats_for_training`] is this with `plane_aux = None`, i.e.
/// byte-identical to the pre-PGSR behaviour.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub async fn render_splats_for_training_with_plane_aux(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    compute_refine_weight: bool,
    rasterization_mode: crate::gaussian_splats::RasterizationMode,
    defer_sh_grad: bool,
    plane_aux: Option<Tensor<2>>,
) -> TrainingSplatOutputDiff {
    let defer_sh_grad = defer_sh_grad
        && cfg!(all(
            feature = "native-msl",
            target_os = "macos",
            target_arch = "aarch64",
            not(target_family = "wasm")
        ));
    render_splats_with_pass_and_refine_weight(
        splats,
        camera,
        img_size,
        background,
        crate::gaussian_splats::RasterPass::Backward,
        training_rasterizer(),
        compute_refine_weight,
        rasterization_mode,
        defer_sh_grad,
        plane_aux,
    )
    .await
}

/// Like [`render_splats`] but lets the caller pick the
/// [`crate::gaussian_splats::RasterPass`]. Used by the finite-diff
/// test suite to enable the C^1 smooth-cutoff surrogate; production code
/// should use [`render_splats`].
pub async fn render_splats_with_pass(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    rasterization_mode: crate::gaussian_splats::RasterizationMode,
) -> SplatOutputDiff {
    render_splats_with_pass_and_rasterizer(
        splats,
        camera,
        img_size,
        background,
        pass,
        Rasterizer::Legacy,
        rasterization_mode,
    )
    .await
}

/// Selector-aware differentiable render entry point for internal rasterizer
/// parity tests. Product code should use [`render_splats_with_pass`].
#[doc(hidden)]
pub async fn render_splats_with_pass_and_rasterizer(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
    rasterization_mode: crate::gaussian_splats::RasterizationMode,
) -> SplatOutputDiff {
    render_splats_with_pass_and_refine_weight(
        splats,
        camera,
        img_size,
        background,
        pass,
        rasterizer,
        true,
        rasterization_mode,
        false,
        None,
    )
    .await
    .into_public()
}

/// Like [`render_splats_with_pass`] but also supplies the PGSR
/// plane-auxiliary channels (approach B).
///
/// Exists for the gradient test suite, which needs a caller-chosen
/// [`crate::gaussian_splats::RasterPass`] (the C^1 smooth-cutoff surrogate), a
/// caller-chosen [`Rasterizer`] (so the 16x8 native-MSL training tile layout is
/// covered for the plane lanes too — that is where lane bugs have hidden before)
/// and the plane input. Production training goes through
/// [`render_splats_for_training_with_plane_aux`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub async fn render_splats_with_pass_and_plane_aux(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
    rasterization_mode: crate::gaussian_splats::RasterizationMode,
    plane_aux: Option<Tensor<2>>,
) -> SplatOutputDiff {
    render_splats_with_pass_and_refine_weight(
        splats,
        camera,
        img_size,
        background,
        pass,
        rasterizer,
        true,
        rasterization_mode,
        false,
        plane_aux,
    )
    .await
    .into_public()
}

async fn render_splats_with_pass_and_refine_weight(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
    compute_refine_weight: bool,
    rasterization_mode: crate::gaussian_splats::RasterizationMode,
    defer_sh_grad: bool,
    plane_aux: Option<Tensor<2>>,
) -> TrainingSplatOutputDiff {
    splats.clone().validate_values().await;

    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "brush_render::bwd::render_splats requires an autodiff-enabled device"
    );

    let refine_weight_holder = Tensor::<1>::zeros([1], &device);
    let refine_weight_holder = if compute_refine_weight {
        refine_weight_holder.require_grad()
    } else {
        refine_weight_holder
    };
    let deferred_sh_holder = Tensor::<2>::zeros([1, 1], &device);
    let deferred_sh_holder = if defer_sh_grad {
        deferred_sh_holder.require_grad()
    } else {
        deferred_sh_holder
    };

    let render_plane = rasterization_mode.render_plane();
    assert!(
        !render_plane || plane_aux.is_some(),
        "RasterizationMode::RgbaDepthPlane requires the [N, PLANE_AUX_LANES] plane features"
    );
    // Non-grad anchor when the plane channels are off, mirroring the
    // deferred-SH anchor above: `RenderBackwards` has a fixed parent set, so the
    // sixth slot is always filled and simply stays untracked.
    let plane_aux = plane_aux.unwrap_or_else(|| {
        Tensor::<2>::zeros([1, crate::kernels::helpers::PLANE_AUX_LANES_USIZE], &device)
    });

    // Fold the 3D-filter floor into scales/opacity for the render. `min_scale`
    // lives on the inner backend; `fold_min_scale` lifts it onto the autodiff
    // graph to match the param values.
    let (transforms_val, raw_opac_val) = match &splats.min_scale {
        Some(f) => fold_min_scale(
            splats.transforms.val(),
            splats.raw_opacities.val(),
            f.clone(),
        ),
        None => (splats.transforms.val(), splats.raw_opacities.val()),
    };

    let transforms_ad = unwrap_ad_wgpu_float(transforms_val);
    let sh_coeffs_ad = unwrap_ad_wgpu_float(splats.sh_coeffs.val());
    let raw_opac_ad = unwrap_ad_wgpu_float(raw_opac_val);
    let refine_weight_ad = unwrap_ad_wgpu_float(refine_weight_holder.clone());
    let deferred_sh_ad = unwrap_ad_wgpu_float(deferred_sh_holder.clone());
    let plane_aux_ad = unwrap_ad_wgpu_float(plane_aux);

    let prep_nodes = RenderBackwards
        .prepare::<NoCheckpointing>([
            transforms_ad.node.clone(),
            refine_weight_ad.node.clone(),
            sh_coeffs_ad.node.clone(),
            raw_opac_ad.node.clone(),
            deferred_sh_ad.node.clone(),
            plane_aux_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    let transforms_inner: FloatTensor<MainBackend> = transforms_ad.primitive.clone();
    let sh_inner: FloatTensor<MainBackend> = sh_coeffs_ad.primitive;
    let raw_opac_inner: FloatTensor<MainBackend> = raw_opac_ad.primitive.clone();
    let plane_aux_inner: Option<FloatTensor<MainBackend>> =
        render_plane.then(|| plane_aux_ad.primitive.clone());

    assert!(
        pass.bwd_info(),
        "render_splats_with_pass requires a Backward variant"
    );
    // Hand-rolled autodiff bridge: render on the inner backend through the
    // rasterizer selector (so the 16x8 `Candidate` tile layout survives — the
    // generic `SplatOps::render` path only ever uses `Rasterizer::Legacy`),
    // then wire the backward manually via `RenderBackwards`.
    let output = crate::burn_glue::render_fusion_with_plane_aux(
        camera,
        img_size,
        transforms_inner.clone(),
        sh_inner.clone(),
        raw_opac_inner.clone(),
        plane_aux_inner.clone(),
        render_mode,
        rasterization_mode,
        background,
        pass,
        rasterizer,
    )
    .await;

    output.clone().validate().await;

    let num_visible = output.aux.num_visible;
    let visible_inner = output.aux.visible.clone();
    let max_radius_inner = output.aux.max_radius.clone();
    let deferred_sh_grad = defer_sh_grad.then(|| DeferredShGradHandle {
        holder: deferred_sh_holder,
        render_transforms: wrap_wgpu_float(transforms_inner.clone()),
        global_from_compact_gid: wrap_wgpu_int(output.global_from_compact_gid.clone()),
        project_uniforms: output.project_uniforms,
    });

    let img_ad: FloatTensor<AutodiffMain> = match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = GaussianBackwardState {
                transforms: transforms_inner,
                sh_coeffs: sh_inner,
                raw_opacity: raw_opac_inner,
                out_img: output.out_img.clone(),
                projected_splats: output.projected_splats,
                project_uniforms: output.project_uniforms,
                tile_offsets: output.aux.tile_offsets.clone(),
                compact_gid_from_isect: output.compact_gid_from_isect,
                plane_aux: plane_aux_inner,
                render_mode,
                pass,
                rasterizer,
                rasterization_mode,
                global_from_compact_gid: output.global_from_compact_gid,
                background,
                img_size,
            };
            prep.finish(state, output.out_img)
        }
        OpsKind::UnTracked(prep) => prep.finish(output.out_img),
    };

    TrainingSplatOutputDiff {
        img: wrap_ad_wgpu_float(img_ad),
        num_visible,
        // `visible` / `max_radius` are render aux — they only feed refine
        // bookkeeping and never get a backward. Hand them back on the inner
        // backend directly so callers don't have to strip autodiff off them.
        visible: wrap_wgpu_float(visible_inner),
        max_radius: wrap_wgpu_float(max_radius_inner),
        refine_weight_holder,
        deferred_sh_grad,
    }
}

/// Autodiff arm of the `#[backend_extension(Wgpu, Autodiff)]` `SplatOps`
/// trait. The generated `Dispatch` impl routes autodiff tensors here.
///
/// Production training goes through `render_splats_with_pass_and_refine_weight`
/// (hand-rolled so the rasterizer selector and deferred-SH path survive); this
/// impl exists to satisfy the extension macro and to service any generic
/// autodiff render. It threads a non-`require_grad` deferred-SH anchor so the
/// 5-input `RenderBackwards` has its full parent set while always taking the
/// dense-coefficient branch.
impl SplatOps for AutodiffMain {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        refine_weight: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        rasterization_mode: crate::gaussian_splats::RasterizationMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> crate::RenderOutput<Self> {
        // Non-grad deferred-SH anchor: the generic dispatch never defers SH, so
        // this parent stays untracked and `RenderBackwards` takes the dense
        // branch. Kept only to fill the fixed 5-input parent set.
        let device = transforms.device();
        let deferred_sh = Self::float_zeros(Shape::new([1, 1]), &device, FloatDType::F32);
        // Likewise a non-grad anchor for the plane-aux parent: the generic
        // dispatch never composites plane channels (it cannot — there is nowhere
        // in `SplatOps::render` to pass them), so this stays untracked.
        let plane_aux = Self::float_zeros(
            Shape::new([1, crate::kernels::helpers::PLANE_AUX_LANES_USIZE]),
            &device,
            FloatDType::F32,
        );
        assert!(
            !rasterization_mode.render_plane(),
            "plane-mode rendering must go through bwd::render_splats_for_training_with_plane_aux"
        );

        let prep_nodes = RenderBackwards
            .prepare::<NoCheckpointing>([
                transforms.node.clone(),
                refine_weight.node.clone(),
                sh_coeffs.node.clone(),
                raw_opacities.node.clone(),
                deferred_sh.node.clone(),
                plane_aux.node.clone(),
            ])
            .compute_bound()
            .stateful();

        let transforms_inner: FloatTensor<MainBackend> = transforms.primitive.clone();
        let sh_inner: FloatTensor<MainBackend> = sh_coeffs.primitive;
        let raw_opac_inner: FloatTensor<MainBackend> = raw_opacities.primitive.clone();

        let output = <MainBackend as SplatOps>::render(
            camera,
            img_size,
            transforms_inner.clone(),
            sh_inner.clone(),
            raw_opac_inner.clone(),
            refine_weight.primitive,
            render_mode,
            rasterization_mode,
            background,
            pass,
        )
        .await;

        output.clone().validate().await;

        let img_ad: FloatTensor<Self> = match prep_nodes {
            OpsKind::Tracked(prep) => {
                let state = GaussianBackwardState {
                    transforms: transforms_inner,
                    sh_coeffs: sh_inner,
                    raw_opacity: raw_opac_inner,
                    out_img: output.out_img.clone(),
                    projected_splats: output.projected_splats.clone(),
                    project_uniforms: output.project_uniforms,
                    tile_offsets: output.aux.tile_offsets.clone(),
                    compact_gid_from_isect: output.compact_gid_from_isect.clone(),
                    plane_aux: None,
                    render_mode,
                    pass,
                    rasterizer: Rasterizer::Legacy,
                    rasterization_mode,
                    global_from_compact_gid: output.global_from_compact_gid.clone(),
                    background,
                    img_size,
                };
                prep.finish(state, output.out_img)
            }
            OpsKind::UnTracked(prep) => prep.finish(output.out_img),
        };

        // Lift the remaining float aux onto the autodiff graph. None of these
        // carry a backward — they only feed refine bookkeeping — but the
        // extension trait's output is uniformly `RenderOutput<Self>`, so they
        // ride along as untracked autodiff tensors. Int tensors share the
        // inner backend's primitive and pass through unchanged.
        let lift = <Self as AutodiffBackend>::from_inner;

        crate::RenderOutput {
            out_img: img_ad,
            aux: crate::RenderAuxInner {
                num_visible: output.aux.num_visible,
                num_intersections: output.aux.num_intersections,
                visible: lift(output.aux.visible),
                max_radius: lift(output.aux.max_radius),
                tile_offsets: output.aux.tile_offsets,
                img_size: output.aux.img_size,
            },
            projected_splats: lift(output.projected_splats),
            compact_gid_from_isect: output.compact_gid_from_isect,
            project_uniforms: output.project_uniforms,
            global_from_compact_gid: output.global_from_compact_gid,
        }
    }
}

#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_arguments)]
fn rasterize_bwd_fusion(
    out_img: FloatTensor<Fusion<MainBackendBase>>,
    projected_splats: FloatTensor<Fusion<MainBackendBase>>,
    compact_gid_from_isect: IntTensor<Fusion<MainBackendBase>>,
    tile_offsets: IntTensor<Fusion<MainBackendBase>>,
    plane_aux: Option<FloatTensor<Fusion<MainBackendBase>>>,
    global_from_compact_gid: IntTensor<Fusion<MainBackendBase>>,
    background: Vec3,
    img_size: glam::UVec2,
    v_output: FloatTensor<Fusion<MainBackendBase>>,
    rasterizer: Rasterizer,
    smooth_cutoff: bool,
    compute_refine_weight: bool,
    render_depth: bool,
    render_plane: bool,
    trusted_forward: bool,
) -> RasterizeGrads<Fusion<MainBackendBase>> {
    #[derive(Debug)]
    struct CustomOp {
        desc: CustomOpIr,
        background: Vec3,
        img_size: glam::UVec2,
        rasterizer: Rasterizer,
        smooth_cutoff: bool,
        compute_refine_weight: bool,
        render_depth: bool,
        render_plane: bool,
        trusted_forward: bool,
    }

    impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
        fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
            let (inputs, outputs) = self.desc.as_fixed();

            // Order must match `input_tensors` below. `plane_aux` is always a
            // real binding (a 1-element dummy when the plane mode is off) so the
            // input arity stays fixed.
            let [
                v_output,
                out_img,
                projected_splats,
                compact_gid_from_isect,
                tile_offsets,
                plane_aux,
                global_from_compact_gid,
            ] = inputs;

            let [v_combined] = outputs;

            let plane_aux = self
                .render_plane
                .then(|| h.get_float_tensor::<MainBackendBase>(plane_aux));

            let grads = if self.trusted_forward {
                <MainBackendBase as InternalSplatBwdOps>::rasterize_bwd_from_forward(
                    ForwardRasterBackward::new(
                        h.get_float_tensor::<MainBackendBase>(out_img),
                        h.get_float_tensor::<MainBackendBase>(projected_splats),
                        h.get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                        h.get_int_tensor::<MainBackendBase>(tile_offsets),
                        plane_aux,
                        h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                        self.background,
                        self.img_size,
                        h.get_float_tensor::<MainBackendBase>(v_output),
                        self.rasterizer,
                        self.smooth_cutoff,
                        self.compute_refine_weight,
                        self.render_depth,
                        self.render_plane,
                    ),
                )
            } else {
                <MainBackendBase as SplatBwdOps>::rasterize_bwd_with_refine_weight(
                    h.get_float_tensor::<MainBackendBase>(out_img),
                    h.get_float_tensor::<MainBackendBase>(projected_splats),
                    h.get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                    h.get_int_tensor::<MainBackendBase>(tile_offsets),
                    plane_aux,
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    self.background,
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(v_output),
                    self.smooth_cutoff,
                    self.compute_refine_weight,
                    self.render_depth,
                    self.render_plane,
                )
            };

            h.register_float_tensor::<MainBackendBase>(&v_combined.id, grads.v_combined);
        }
    }

    // projected_splats is [num_visible, PROJECTED_LANES], so shape[0] gives num_visible.
    let num_visible_val = projected_splats.shape()[0] as u32;

    let client = v_output.client.clone();
    let num_visible = (num_visible_val as usize).max(1);
    // Always a real binding: CubeCL takes every tensor argument even when a
    // comptime flag removes all its accesses, so a 1-element dummy keeps the
    // custom-op input arity fixed instead of forking the descriptor.
    let plane_aux = plane_aux.unwrap_or_else(|| {
        <Fusion<MainBackendBase> as FloatTensorOps<Fusion<MainBackendBase>>>::float_zeros(
            [1].into(),
            &v_output.device(),
            FloatDType::F32,
        )
    });
    let input_tensors = [
        v_output,
        out_img,
        projected_splats,
        compact_gid_from_isect,
        tile_offsets,
        plane_aux,
        global_from_compact_gid,
    ];
    // Sparse [num_visible, COMPACT_GRAD_LANES] indexed by compact_gid. Lane 10
    // is the expected-depth gradient (zero/unused when render_depth is false);
    // lanes 11..=14 carry the PGSR plane-aux value gradients.
    // Stride owned by bwd::kernels::rasterize_backwards::COMPACT_GRAD_LANES.
    let v_combined_out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([
            num_visible,
            crate::bwd::kernels::rasterize_backwards::COMPACT_GRAD_LANES as usize,
        ]),
        DType::F32,
    );
    let desc = CustomOpIr::new(
        "rasterize_bwd",
        &input_tensors.map(|tensor| tensor.into_ir()),
        &[v_combined_out],
    );
    let [v_combined] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            CustomOp {
                desc,
                background,
                img_size,
                rasterizer,
                smooth_cutoff,
                compute_refine_weight,
                render_depth,
                render_plane,
                trusted_forward,
            },
        )
        .outputs();

    RasterizeGrads { v_combined }
}

impl SplatBwdOps for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        plane_aux: Option<FloatTensor<Self>>,
        global_from_compact_gid: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
        render_depth: bool,
        render_plane: bool,
    ) -> RasterizeGrads<Self> {
        rasterize_bwd_fusion(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            plane_aux,
            global_from_compact_gid,
            background,
            img_size,
            v_output,
            Rasterizer::Legacy,
            smooth_cutoff,
            true,
            render_depth,
            render_plane,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd_with_refine_weight(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        plane_aux: Option<FloatTensor<Self>>,
        global_from_compact_gid: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
        compute_refine_weight: bool,
        render_depth: bool,
        render_plane: bool,
    ) -> RasterizeGrads<Self> {
        rasterize_bwd_fusion(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            plane_aux,
            global_from_compact_gid,
            background,
            img_size,
            v_output,
            Rasterizer::Legacy,
            smooth_cutoff,
            compute_refine_weight,
            render_depth,
            render_plane,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        // The screen-area regulariser only acts in the backward kernel, so we
        // stamp the weight onto the uniforms here rather than in the forward.
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            render_mode: SplatRenderMode,
            project_uniforms: ProjectUniforms,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [
                    transforms,
                    sh_coeffs,
                    raw_opac,
                    global_from_compact_gid,
                    v_combined_in,
                ] = inputs;

                let [v_transforms, v_coeffs, v_raw_opac, v_refine_weight] = outputs;

                let grads = <MainBackendBase as SplatBwdOps>::project_bwd(
                    h.get_float_tensor::<MainBackendBase>(transforms),
                    h.get_float_tensor::<MainBackendBase>(sh_coeffs),
                    h.get_float_tensor::<MainBackendBase>(raw_opac),
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    self.project_uniforms,
                    self.render_mode,
                    h.get_float_tensor::<MainBackendBase>(v_combined_in),
                );

                h.register_float_tensor::<MainBackendBase>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<MainBackendBase>(&v_coeffs.id, grads.v_coeffs);
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<MainBackendBase>(
                    &v_refine_weight.id,
                    grads.v_refine_weight,
                );
            }
        }

        let client = transforms.client.clone();
        let num_points = transforms.shape[0];
        let coeffs = sh_coeffs_for_degree(project_uniforms.sh_degree) as usize;

        let input_tensors = [
            transforms,
            sh_coeffs,
            raw_opac,
            global_from_compact_gid,
            v_combined,
        ];

        let outputs = {
            let v_transforms_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points, 10]),
                DType::F32,
            );
            let v_coeffs_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points, coeffs, 3]),
                DType::F32,
            );
            let v_raw_opac_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points]),
                DType::F32,
            );
            let v_refine_weight_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points]),
                DType::F32,
            );

            let stream = StreamId::current();
            let desc = CustomOpIr::new(
                "project_bwd",
                &input_tensors.map(|t| t.into_ir()),
                &[
                    v_transforms_out,
                    v_coeffs_out,
                    v_raw_opac_out,
                    v_refine_weight_out,
                ],
            );

            client
                .register(
                    stream,
                    OperationIr::Custom(desc.clone()),
                    CustomOp {
                        desc,
                        render_mode,
                        project_uniforms,
                    },
                )
                .outputs()
        };

        let [v_transforms, v_coeffs, v_raw_opac, v_refine_weight] = outputs;

        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd_deferred_sh(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> DeferredSplatGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            render_mode: SplatRenderMode,
            project_uniforms: ProjectUniforms,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();
                let [
                    transforms,
                    sh_coeffs,
                    raw_opac,
                    global_from_compact_gid,
                    v_combined,
                ] = inputs;
                let [v_transforms, v_raw_opac, v_refine_weight] = outputs;

                let grads = <MainBackendBase as SplatBwdOps>::project_bwd_deferred_sh(
                    h.get_float_tensor::<MainBackendBase>(transforms),
                    h.get_float_tensor::<MainBackendBase>(sh_coeffs),
                    h.get_float_tensor::<MainBackendBase>(raw_opac),
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    self.project_uniforms,
                    self.render_mode,
                    h.get_float_tensor::<MainBackendBase>(v_combined),
                );

                h.register_float_tensor::<MainBackendBase>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<MainBackendBase>(
                    &v_refine_weight.id,
                    grads.v_refine_weight,
                );
            }
        }

        let client = transforms.client.clone();
        let num_points = transforms.shape[0];
        let input_tensors = [
            transforms,
            sh_coeffs,
            raw_opac,
            global_from_compact_gid,
            v_combined,
        ];
        let v_transforms = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 10]),
            DType::F32,
        );
        let v_raw_opac = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points]),
            DType::F32,
        );
        let v_refine_weight = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points]),
            DType::F32,
        );
        let desc = CustomOpIr::new(
            "project_bwd_deferred_sh",
            &input_tensors.map(|tensor| tensor.into_ir()),
            &[v_transforms, v_raw_opac, v_refine_weight],
        );
        let [v_transforms, v_raw_opac, v_refine_weight] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                CustomOp {
                    desc,
                    render_mode,
                    project_uniforms,
                },
            )
            .outputs();

        DeferredSplatGrads {
            v_transforms,
            v_raw_opac,
            v_refine_weight,
        }
    }
}

impl InternalSplatBwdOps for Fusion<MainBackendBase> {
    fn rasterize_bwd_from_forward(input: ForwardRasterBackward<Self>) -> RasterizeGrads<Self> {
        let (
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            plane_aux,
            global_from_compact_gid,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
            render_depth,
            render_plane,
        ) = input.into_parts();
        rasterize_bwd_fusion(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            plane_aux,
            global_from_compact_gid,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
            render_depth,
            render_plane,
            true,
        )
    }
}
