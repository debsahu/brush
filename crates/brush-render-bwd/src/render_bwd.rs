use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor};
use brush_render::gaussian_splats::{Rasterizer, SplatRenderMode};
use brush_render::kernels::types::RasterizeUniformsLaunch;
use brush_render::sh::sh_coeffs_for_degree;
use burn::backend::TensorMetadata;
use burn::backend::ops::{FloatTensorOps, IntTensorOps};
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::tensor::{DType, FloatDType, IntDType};
use burn_cubecl::cubecl::CubeCount;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::cubecl::features::{AtomicUsage, Plane};
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, Type};
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use glam::{Vec3, uvec2};

use crate::burn_glue::{
    DeferredSplatGrads, ForwardRasterBackward, InternalSplatBwdOps, RasterizeGrads, SplatBwdOps,
    SplatGrads,
};
use crate::kernels;
use brush_render::shaders::helpers::ProjectUniforms;

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn use_unchecked_raster_bwd() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = brush_render::native_msl::option_requested(
            brush_render::native_msl::UNCHECKED_RASTER_BWD_ENV,
        );
        if enabled {
            tracing::warn!(
                "experimental unchecked native-MSL raster backward requested; devices without native float atomics retain bounds checks"
            );
        }
        enabled
    })
}

#[cfg(not(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
)))]
fn use_unchecked_raster_bwd() -> bool {
    false
}

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn use_coalesced_sh_grad() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        brush_render::native_msl::option_requested(brush_render::native_msl::COALESCED_SH_GRAD_ENV)
    })
}

#[cfg(not(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
)))]
fn use_coalesced_sh_grad() -> bool {
    false
}

fn should_launch_unchecked(
    hard_floats: bool,
    unchecked_requested: bool,
    trusted_forward: bool,
) -> bool {
    hard_floats && unchecked_requested && trusted_forward
}

#[allow(clippy::too_many_arguments)]
fn rasterize_bwd_impl(
    out_img: FloatTensor<MainBackendBase>,
    projected_splats: FloatTensor<MainBackendBase>,
    compact_gid_from_isect: IntTensor<MainBackendBase>,
    tile_offsets: IntTensor<MainBackendBase>,
    background: Vec3,
    img_size: glam::UVec2,
    v_output: FloatTensor<MainBackendBase>,
    rasterizer: Rasterizer,
    smooth_cutoff: bool,
    compute_refine_weight: bool,
    trusted_forward: bool,
) -> RasterizeGrads<MainBackendBase> {
    let _span = tracing::trace_span!("rasterize_bwd").entered();

    let v_output = into_contiguous(v_output);
    let device = out_img.device.clone();
    let num_visible = projected_splats.shape()[0].max(1);
    let client = projected_splats.client.clone();

    // Sparse [num_visible, 10] indexed by compact_gid.
    let v_combined =
        MainBackendBase::float_zeros([num_visible, 10].into(), &device, FloatDType::F32);

    let tile_width = rasterizer.tile_width();
    let tile_height = rasterizer.tile_height();
    let tile_bounds = uvec2(
        img_size.x.div_ceil(tile_width),
        img_size.y.div_ceil(tile_height),
    );
    let tile_offset_shape = tile_offsets.shape();
    assert_eq!(tile_offset_shape.rank(), 3, "tile offsets must be rank 3");
    assert_eq!(
        tile_offset_shape[0], tile_bounds.y as usize,
        "tile-offset height must match the selected rasterizer"
    );
    assert_eq!(
        tile_offset_shape[1], tile_bounds.x as usize,
        "tile-offset width must match the selected rasterizer"
    );
    assert_eq!(
        tile_offset_shape[2], 2,
        "tile offsets must store one start/end pair per tile"
    );

    let hard_floats = client
        .properties()
        .atomic_type_usage(Type::atomic(Type::scalar(ElemType::Float(FloatKind::F32))))
        .contains(AtomicUsage::Add);

    let cube_count = CubeCount::Static(tile_bounds.x, tile_bounds.y, 1);
    let cube_dim = CubeDim::new_1d(kernels::rasterize_backwards::SPLAT_BATCH);
    let uniforms = RasterizeUniformsLaunch::new(
        tile_bounds.x,
        img_size.x,
        img_size.y,
        background.x,
        background.y,
        background.z,
    );
    let unchecked_requested = trusted_forward && use_unchecked_raster_bwd();
    let use_unchecked = should_launch_unchecked(hard_floats, unchecked_requested, trusted_forward);

    tracing::trace_span!("RasterizeBackwards").in_scope(|| {
        use kernels::rasterize_backwards::{CasAtomicAdd, HfAtomicAdd, rasterize_backwards_kernel};
        if use_unchecked {
            // SAFETY: `trusted_forward` is only reachable through the opaque
            // `ForwardRasterBackward` produced by this crate's private autodiff bridge. The
            // selected rasterizer's tile-offset shape is asserted above, and the forward pass
            // guarantees ordered offsets bounded by the intersection buffer, valid compact IDs,
            // fixed projected rows, and exact contiguous image/output sizes. Image and partial
            // batch accesses remain guarded in the kernel. This path uses native float atomics,
            // not the CAS retry loop.
            unsafe {
                rasterize_backwards_kernel::launch_unchecked::<HfAtomicAdd, WgpuRuntime>(
                    &client,
                    cube_count,
                    cube_dim,
                    compact_gid_from_isect.into_tensor_arg(),
                    tile_offsets.into_tensor_arg(),
                    projected_splats.into_tensor_arg(),
                    out_img.into_tensor_arg(),
                    v_output.into_tensor_arg(),
                    v_combined.clone().into_tensor_arg(),
                    uniforms,
                    smooth_cutoff,
                    compute_refine_weight,
                    tile_width,
                    tile_height,
                );
            }
        } else if hard_floats {
            rasterize_backwards_kernel::launch::<HfAtomicAdd, WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                compact_gid_from_isect.into_tensor_arg(),
                tile_offsets.into_tensor_arg(),
                projected_splats.into_tensor_arg(),
                out_img.into_tensor_arg(),
                v_output.into_tensor_arg(),
                v_combined.clone().into_tensor_arg(),
                uniforms,
                smooth_cutoff,
                compute_refine_weight,
                tile_width,
                tile_height,
            );
        } else {
            // Keep bounds checks for the CAS fallback: its weak-CAS retry loop does not meet
            // CubeCL's unchecked-launch termination contract on every target.
            rasterize_backwards_kernel::launch::<CasAtomicAdd, WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                compact_gid_from_isect.into_tensor_arg(),
                tile_offsets.into_tensor_arg(),
                projected_splats.into_tensor_arg(),
                out_img.into_tensor_arg(),
                v_output.into_tensor_arg(),
                v_combined.clone().into_tensor_arg(),
                uniforms,
                smooth_cutoff,
                compute_refine_weight,
                tile_width,
                tile_height,
            );
        }
    });

    RasterizeGrads { v_combined }
}

impl SplatBwdOps for MainBackendBase {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
    ) -> RasterizeGrads<Self> {
        rasterize_bwd_impl(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            Rasterizer::Legacy,
            smooth_cutoff,
            true,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd_with_refine_weight(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
        compute_refine_weight: bool,
    ) -> RasterizeGrads<Self> {
        rasterize_bwd_impl(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            Rasterizer::Legacy,
            smooth_cutoff,
            compute_refine_weight,
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
        let _span = tracing::trace_span!("project_bwd").entered();

        // The screen-area regulariser only acts in this backward kernel, so we
        // stamp the weight onto the uniforms here rather than in the forward.
        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opac = into_contiguous(raw_opac);

        let device = transforms.device.clone();
        let num_points = transforms.shape()[0];
        let client = transforms.client.clone();

        let use_materialized_sh_grad = use_coalesced_sh_grad()
            && num_points > 0
            && project_uniforms.total_splats as usize == num_points
            && project_uniforms.sh_degree <= 4
            && {
                let features = client.features();
                let properties = client.properties();
                features.plane.contains(Plane::Ops)
                    && features.plane.contains(Plane::NonUniformControlFlow)
                    && properties.hardware.plane_size_min
                        == kernels::sh_grad_materialize::PLANE_SIZE
                    && properties.hardware.plane_size_max
                        == kernels::sh_grad_materialize::PLANE_SIZE
                    && properties.hardware.max_units_per_cube
                        >= kernels::sh_grad_materialize::WG_SIZE
                    && properties.hardware.max_cube_dim.0 >= kernels::sh_grad_materialize::WG_SIZE
            };
        // Dense outputs, the kernel scatters compact→global internally.
        let v_transforms = Self::float_zeros([num_points, 10].into(), &device, FloatDType::F32);
        let coeff_shape = [
            num_points,
            sh_coeffs_for_degree(project_uniforms.sh_degree) as usize,
            3,
        ];
        let v_coeffs = if use_materialized_sh_grad {
            create_tensor(coeff_shape, &device, DType::F32)
        } else {
            Self::float_zeros(coeff_shape.into(), &device, FloatDType::F32)
        };
        let v_raw_opac = Self::float_zeros([num_points].into(), &device, FloatDType::F32);
        let v_refine_weight = Self::float_zeros([num_points].into(), &device, FloatDType::F32);

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        let num_visible = project_uniforms.num_visible;

        let uniforms = project_uniforms.to_launch_object();
        let sh_grad_inputs = use_materialized_sh_grad.then(|| {
            (
                transforms.clone(),
                global_from_compact_gid.clone(),
                v_combined.clone(),
                Self::int_zeros([num_points].into(), &device, IntDType::U32),
            )
        });

        tracing::trace_span!("ProjectBackwards").in_scope(|| {
            kernels::project_backwards::project_backwards_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, kernels::project_backwards::WG_SIZE),
                CubeDim::new_1d(kernels::project_backwards::WG_SIZE),
                transforms.into_tensor_arg(),
                sh_coeffs.into_tensor_arg(),
                raw_opac.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_combined.into_tensor_arg(),
                v_transforms.clone().into_tensor_arg(),
                v_coeffs.clone().into_tensor_arg(),
                v_raw_opac.clone().into_tensor_arg(),
                v_refine_weight.clone().into_tensor_arg(),
                uniforms,
                mip_splat,
                project_uniforms.sh_degree,
                project_uniforms.camera_model,
                use_materialized_sh_grad,
            );
        });

        if let Some((
            transforms_for_sh_grad,
            global_from_compact_for_sh_grad,
            v_combined_for_sh_grad,
            compact_plus_one_from_global,
        )) = sh_grad_inputs
        {
            if num_visible > 0 {
                tracing::trace_span!("BuildCompactShMap").in_scope(|| {
                    kernels::sh_grad_materialize::build_compact_sh_map_kernel::launch::<
                        WgpuRuntime,
                    >(
                        &client,
                        calc_cube_count_1d(
                            num_visible,
                            kernels::sh_grad_materialize::WG_SIZE,
                        ),
                        CubeDim::new_1d(kernels::sh_grad_materialize::WG_SIZE),
                        global_from_compact_for_sh_grad.into_tensor_arg(),
                        v_combined_for_sh_grad.clone().into_tensor_arg(),
                        compact_plus_one_from_global.clone().into_tensor_arg(),
                        project_uniforms.to_launch_object(),
                    );
                });
            }
            tracing::trace_span!("MaterializeShGrad").in_scope(|| {
                // SAFETY: the gate above proves total_splats == num_points,
                // degree <= 4, and a fixed 32-lane plane. Every active plane
                // therefore owns one in-bounds global row; compact+1 is either
                // the zero sentinel or indexes the compact [num_visible, 10]
                // gradient, and the three lane stores cover the entire SH row.
                unsafe {
                    kernels::sh_grad_materialize::materialize_sh_grad_kernel::launch_unchecked::<
                        WgpuRuntime,
                    >(
                        &client,
                        calc_cube_count_1d(
                            project_uniforms.total_splats,
                            kernels::sh_grad_materialize::SPLATS_PER_WG,
                        ),
                        CubeDim::new_1d(kernels::sh_grad_materialize::WG_SIZE),
                        transforms_for_sh_grad.into_tensor_arg(),
                        compact_plus_one_from_global.into_tensor_arg(),
                        v_combined_for_sh_grad.into_tensor_arg(),
                        v_coeffs.clone().into_tensor_arg(),
                        project_uniforms.to_launch_object(),
                        project_uniforms.sh_degree,
                    );
                }
            });
        }

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
        let _span = tracing::trace_span!("project_bwd_deferred_sh").entered();
        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opac = into_contiguous(raw_opac);
        let device = transforms.device.clone();
        let num_points = transforms.shape()[0];
        let client = transforms.client.clone();

        let v_transforms = Self::float_zeros([num_points, 10].into(), &device, FloatDType::F32);
        // ProjectBackwards takes the coefficient output as a storage binding,
        // but the final comptime flag removes every access on this path.
        let unused_v_coeffs = Self::float_zeros([1].into(), &device, FloatDType::F32);
        let v_raw_opac = Self::float_zeros([num_points].into(), &device, FloatDType::F32);
        let v_refine_weight = Self::float_zeros([num_points].into(), &device, FloatDType::F32);

        tracing::trace_span!("ProjectBackwardsDeferredSh").in_scope(|| {
            kernels::project_backwards::project_backwards_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(
                    project_uniforms.num_visible,
                    kernels::project_backwards::WG_SIZE,
                ),
                CubeDim::new_1d(kernels::project_backwards::WG_SIZE),
                transforms.into_tensor_arg(),
                sh_coeffs.into_tensor_arg(),
                raw_opac.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_combined.into_tensor_arg(),
                v_transforms.clone().into_tensor_arg(),
                unused_v_coeffs.into_tensor_arg(),
                v_raw_opac.clone().into_tensor_arg(),
                v_refine_weight.clone().into_tensor_arg(),
                project_uniforms.to_launch_object(),
                matches!(render_mode, SplatRenderMode::Mip),
                project_uniforms.sh_degree,
                project_uniforms.camera_model,
                true,
            );
        });

        DeferredSplatGrads {
            v_transforms,
            v_raw_opac,
            v_refine_weight,
        }
    }
}

impl InternalSplatBwdOps for MainBackendBase {
    fn rasterize_bwd_from_forward(input: ForwardRasterBackward<Self>) -> RasterizeGrads<Self> {
        let (
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
        ) = input.into_parts();
        rasterize_bwd_impl(
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            background,
            img_size,
            v_output,
            rasterizer,
            smooth_cutoff,
            compute_refine_weight,
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::should_launch_unchecked;

    #[test]
    fn unchecked_launch_requires_forward_provenance_and_device_support() {
        assert!(!should_launch_unchecked(true, true, false));
        assert!(!should_launch_unchecked(true, false, true));
        assert!(!should_launch_unchecked(false, true, true));
        assert!(should_launch_unchecked(true, true, true));
    }
}
