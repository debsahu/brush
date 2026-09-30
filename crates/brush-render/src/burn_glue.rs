#![allow(clippy::match_wildcard_for_single_variants)]

use crate::fusion::bind;
use brush_cube::{MainBackend, MainBackendBase};
use burn::backend::{
    Autodiff, BackendTensor, DispatchAutodiffContext, DispatchTensor, DispatchTensorKind,
    GradientCheckpointingStrategy,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{Int, Tensor};
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::Fusion;
use glam::Vec3;

use crate::gaussian_splats::RasterizationMode;
use crate::{
    RenderAuxInner, SplatOps, SplatRasterizerOps, backend_kind,
    camera::Camera,
    gaussian_splats::{Rasterizer, SplatRenderMode},
    render_aux::RenderOutput,
};

/// Inner autodiff backend (same as `Autodiff<burn::backend::Wgpu>`).
/// Used as the primitive backend for autodiff `Tensor<D>` operations.
pub type AutodiffMain = Autodiff<MainBackend>;

// ---------------------------------------------------------------------------
// `Tensor<D>` <-> backend-level primitive bridges.
//
// `Tensor<D>` is pinned to burn's `Dispatch` backend and brush only runs on
// the cubecl backend, so every helper here assumes a `DispatchTensorKind::Cube`
// (optionally wrapped in `Autodiff`) and panics otherwise. Upstream brush has
// dropped most of these now that its backward runs on generated extension
// impls; this fork keeps them because its hand-rolled training bridges (the
// plane-aux / deferred-SH render, the feature backward, LPIPS, PPISP, TIDI)
// still move primitives across the dispatch boundary directly.
// ---------------------------------------------------------------------------

/// Extract the inner fusion float tensor from a non-autodiff `Tensor<D>`.
pub fn unwrap_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.float(),
        other => panic!(
            "expected Wgpu tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner fusion int tensor from a non-autodiff `Tensor<D, Int>`.
pub fn unwrap_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_wgpu_float`]: wraps a fusion float tensor as a
/// user-facing `Tensor<D>`.
pub fn wrap_wgpu_float<const D: usize>(t: FloatTensor<MainBackend>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: backend_kind!(BackendTensor::Float(t)),
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

/// Inverse of [`unwrap_wgpu_int`]: wraps a fusion int tensor as a
/// user-facing `Tensor<D, Int>`.
pub fn wrap_wgpu_int<const D: usize>(t: IntTensor<MainBackend>) -> Tensor<D, Int> {
    Tensor::from_dispatch(DispatchTensor {
        kind: backend_kind!(BackendTensor::Int(t)),
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

/// Extract the inner `AutodiffTensor<MainBackend>` from a `Tensor<D>` on an
/// autodiff-enabled device. Panics on any other shape.
pub fn unwrap_ad_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<AutodiffMain> {
    let prim: DispatchTensor = t.into_dispatch();
    match prim.kind {
        DispatchTensorKind::Autodiff(inner) => match *inner {
            backend_kind!(BackendTensor::Autodiff(t)) => t,
            other => panic!(
                "autodiff inner kind is not Wgpu: {:?}",
                std::mem::discriminant(&other)
            ),
        },
        other => panic!(
            "expected autodiff-enabled tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner `IntTensor` regardless of whether the tensor is wrapped
/// in an autodiff device — ints are never autodiff-tracked.
pub fn unwrap_ad_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    match kind {
        backend_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_ad_wgpu_float`]: wraps an autodiff tensor as a
/// user-facing `Tensor<D>` on the autodiff device.
pub fn wrap_ad_wgpu_float<const D: usize>(t: FloatTensor<AutodiffMain>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: DispatchTensorKind::Autodiff(Box::new(backend_kind!(BackendTensor::Autodiff(t)))),
        autodiff: DispatchAutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
    })
}

/// Strip the autodiff wrapping from a `Tensor<D>`.
///
/// Operates directly on the `DispatchTensor` kind so it works both for an
/// autodiff input (unwrap one level) and an already-inner input (passthrough),
/// always landing with autodiff disabled.
pub fn detach_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    // Hand-rolled render/backward bridges store the concrete autodiff tensor
    // inside the backend variant. Strip that layer too; merely removing
    // Dispatch's outer bridge leaves `BackendTensor::Autodiff` behind and a
    // subsequent inner custom op panics when it requests a float primitive.
    let kind = match kind {
        backend_kind!(BackendTensor::Autodiff(t)) => {
            backend_kind!(BackendTensor::Float(t.into_primitive()))
        }
        other => other,
    };
    Tensor::from_dispatch(DispatchTensor {
        kind,
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

/// Lift a non-autodiff `Tensor<D>` into the autodiff graph as a constant.
/// A no-op if `t` is already autodiff.
///
/// Upstream burn fixed the `from_inner` bridge this fork used to hand-roll
/// around, so this is now burn's own `Tensor::autodiff`.
pub fn lift_to_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    t.autodiff()
}

fn is_autodiff<const D: usize>(t: &Tensor<D>) -> bool {
    matches!(
        t.clone().into_dispatch().kind,
        DispatchTensorKind::Autodiff(_)
    )
}

/// Put `t` on the same autodiff/inner backend variant as `reference`. Brush
/// keeps some frozen tensors (e.g. the 3D-filter floor) on the inner backend
/// but folds them against params that may be lifted to autodiff; this aligns
/// both operands so dispatch ops don't trip a cross-backend assertion.
pub(crate) fn match_backend<const D: usize, const DR: usize>(
    t: Tensor<D>,
    reference: &Tensor<DR>,
) -> Tensor<D> {
    if is_autodiff(reference) {
        lift_to_autodiff(t)
    } else {
        detach_autodiff(t)
    }
}

/// Like [`detach_autodiff`] for `Tensor<D, Int>`.
pub fn detach_autodiff_int<const D: usize>(t: Tensor<D, Int>) -> Tensor<D, Int> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    Tensor::from_dispatch(DispatchTensor {
        kind,
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

/// Resolve pending fusion operations and return the underlying tensor.
pub fn resolve_to_cube_float<const D: usize>(tensor: Tensor<D>) -> CubeTensor {
    let fusion = unwrap_wgpu_float(tensor);
    let client = fusion.client.clone();
    client.resolve_tensor_float::<MainBackendBase>(fusion)
}

impl SplatOps for Fusion<MainBackendBase> {
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        // Ignored in the forward: the refine statistic and the SH second
        // moment are produced by the backward. Present only as differentiable
        // trait inputs.
        _refine_weight: FloatTensor<Self>,
        _coeffs_grad_sq: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        raster_mode: RasterizationMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> RenderOutput<Self> {
        <Self as SplatRasterizerOps>::render_with_rasterizer(
            camera,
            img_size,
            transforms,
            sh_coeffs,
            raw_opacities,
            min_scale,
            has_min_scale,
            render_mode,
            raster_mode,
            background,
            pass,
            Rasterizer::Legacy,
        )
        .await
    }
}

impl SplatRasterizerOps for Fusion<MainBackendBase> {
    async fn render_with_rasterizer(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        render_mode: SplatRenderMode,
        raster_mode: RasterizationMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
        rasterizer: Rasterizer,
    ) -> RenderOutput<Self> {
        render_fusion_with_plane_aux(
            camera,
            img_size,
            transforms,
            sh_coeffs,
            raw_opacities,
            min_scale,
            has_min_scale,
            None,
            render_mode,
            raster_mode,
            background,
            pass,
            rasterizer,
        )
        .await
    }
}

/// Fusion-layer render with the optional PGSR plane-auxiliary input.
///
/// Body of [`SplatRasterizerOps::render_with_rasterizer`] for
/// `Fusion<MainBackendBase>`; the trait method delegates with `plane_aux = None`
/// so pre-existing callers are byte-identical. See
/// [`crate::render::render_base_with_plane_aux`] for why this is a free function
/// rather than a widened trait method.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn render_fusion_with_plane_aux(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<Fusion<MainBackendBase>>,
    sh_coeffs: FloatTensor<Fusion<MainBackendBase>>,
    raw_opacities: FloatTensor<Fusion<MainBackendBase>>,
    min_scale: FloatTensor<Fusion<MainBackendBase>>,
    has_min_scale: bool,
    plane_aux: Option<FloatTensor<Fusion<MainBackendBase>>>,
    render_mode: SplatRenderMode,
    raster_mode: RasterizationMode,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    rasterizer: Rasterizer,
) -> RenderOutput<Fusion<MainBackendBase>> {
    let client = transforms.client.clone();

    // Resolve fusion inputs to MainBackendBase tensors. This drains any
    // pending fusion operations into a concrete buffer.
    let base_transforms = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(transforms);
    let base_sh_coeffs = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(sh_coeffs);
    let base_raw_opac = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(raw_opacities);
    let base_min_scale = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(min_scale);
    let base_plane_aux =
        plane_aux.map(|t| client.clone().resolve_tensor_float::<MainBackendBase>(t));

    // Run the full pipeline on MainBackendBase. `render_with_rasterizer`
    // takes no `refine_weight`: the refine statistic is produced by the
    // rasterize backward, not the forward.
    let out = crate::render::render_base_with_plane_aux(
        camera,
        img_size,
        base_transforms,
        base_sh_coeffs,
        base_raw_opac,
        base_min_scale,
        has_min_scale,
        base_plane_aux,
        render_mode,
        raster_mode,
        background,
        pass,
        rasterizer,
    )
    .await;

    let RenderOutput {
        out_img,
        aux,
        projected_splats,
        compact_gid_from_isect,
        project_uniforms,
        global_from_compact_gid,
        compact_from_global,
    } = out;
    let RenderAuxInner {
        num_visible,
        num_intersections,
        visible,
        max_radius,
        opacities,
        tile_offsets,
        img_size,
    } = aux;

    // Seed the finished tensors back into the fusion stream (upstream #553).
    let bind = |t| bind(&client, t);

    RenderOutput {
        out_img: bind(out_img),
        aux: RenderAuxInner {
            num_visible,
            num_intersections,
            visible: bind(visible),
            max_radius: bind(max_radius),
            opacities: bind(opacities),
            tile_offsets: bind(tile_offsets),
            img_size,
        },
        projected_splats: bind(projected_splats),
        compact_gid_from_isect: bind(compact_gid_from_isect),
        project_uniforms,
        global_from_compact_gid: bind(global_from_compact_gid),
        compact_from_global: bind(compact_from_global),
    }
}
