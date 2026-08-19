use crate::camera::calculate_jacobian_clamp_limits;
use crate::{
    RenderAuxInner, SplatOps, SplatRasterizerOps,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::{RasterPass, Rasterizer, SplatRenderMode},
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    kernels,
    render_aux::RenderOutput,
    sh::sh_degree_from_coeffs,
    shaders,
};
use brush_cube::create_tensor;
use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::backend::TensorMetadata;
use burn::backend::ops::TransactionPrimitive;
use burn::backend::ops::{FloatTensorOps, IntTensorOps, TransactionOps};
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, FloatDType, IntDType};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use glam::{Vec3, uvec2};
use kernels::types::RasterizeUniformsLaunch;
use std::f32::consts::PI;

#[doc(hidden)]
pub fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    calc_tile_bounds_for_dims(
        img_size,
        shaders::helpers::TILE_WIDTH,
        shaders::helpers::TILE_WIDTH,
    )
}

fn calc_tile_bounds_for_dims(
    img_size: glam::UVec2,
    tile_width: u32,
    tile_height: u32,
) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(tile_width),
        img_size.y.div_ceil(tile_height),
    )
}

impl SplatOps for MainBackendBase {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        _refine_weight: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        raster_mode: crate::gaussian_splats::RasterizationMode,
        background: Vec3,
        pass: RasterPass,
    ) -> RenderOutput<Self> {
        <Self as SplatRasterizerOps>::render_with_rasterizer(
            camera,
            img_size,
            transforms,
            sh_coeffs,
            raw_opacities,
            render_mode,
            raster_mode,
            background,
            pass,
            Rasterizer::Legacy,
        )
        .await
    }
}

impl SplatRasterizerOps for MainBackendBase {
    #[allow(clippy::too_many_arguments)]
    async fn render_with_rasterizer(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        raster_mode: crate::gaussian_splats::RasterizationMode,
        background: Vec3,
        pass: RasterPass,
        rasterizer: Rasterizer,
    ) -> RenderOutput<Self> {
        render_base_with_plane_aux(
            camera,
            img_size,
            transforms,
            sh_coeffs,
            raw_opacities,
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

/// Full forward pipeline, with the optional PGSR plane-auxiliary input.
///
/// This is the body of [`SplatRasterizerOps::render_with_rasterizer`]; the trait
/// method delegates here with `plane_aux = None`, so every pre-existing caller
/// is byte-identical. Kept as a free function rather than widening the trait
/// because `SplatOps` / `SplatRasterizerOps` are `#[backend_extension]` traits
/// whose `Dispatch` impls are macro-generated; the plane path is only ever
/// reached through the hand-rolled autodiff bridge in `bwd::burn_glue`, so it
/// does not need to exist on the generic backend API.
///
/// `plane_aux` is `[total_splats, PLANE_AUX_LANES]` indexed by GLOBAL gid, and
/// must be `Some` exactly when the effective plane mode is active (see the
/// `render_plane` assertion below). PGSR (Chen et al. 2024, arXiv:2406.06521).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn render_base_with_plane_aux(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<MainBackendBase>,
    sh_coeffs: FloatTensor<MainBackendBase>,
    raw_opacities: FloatTensor<MainBackendBase>,
    plane_aux: Option<FloatTensor<MainBackendBase>>,
    render_mode: SplatRenderMode,
    raster_mode: crate::gaussian_splats::RasterizationMode,
    background: Vec3,
    pass: RasterPass,
    rasterizer: Rasterizer,
) -> RenderOutput<MainBackendBase> {
    assert!(
        img_size[0] > 0 && img_size[1] > 0,
        "Can't render images with 0 size."
    );
    let bwd_info = pass.bwd_info();
    let smooth_cutoff = pass.smooth_cutoff();
    let tile_width = rasterizer.tile_width();
    let tile_height = rasterizer.tile_height();
    let tile_size = rasterizer.tile_size();
    let render_depth = raster_mode.render_depth() && bwd_info;
    // Plane compositing only makes sense on a backward-enabled pass: the
    // per-splat lookup goes through the `bwd_info`-only `load_gid` staging, and
    // the whole point of the channel is to be differentiated. Same gating idiom
    // as `render_depth` above.
    let render_plane = raster_mode.render_plane() && bwd_info;

    let transforms = into_contiguous(transforms);
    let sh_coeffs = into_contiguous(sh_coeffs);
    let raw_opacities = into_contiguous(raw_opacities);
    assert!(
        !render_plane || plane_aux.is_some(),
        "RasterizationMode::RgbaDepthPlane requires a [N, PLANE_AUX_LANES] plane_aux input"
    );
    let plane_aux = plane_aux.map(into_contiguous);

    let dim_check = DimCheck::new()
        .check_dims("transforms", &transforms, &["D".into(), 10.into()])
        .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
        .check_dims("raw_opacities", &raw_opacities, &["D".into()]);
    if let Some(plane_aux) = plane_aux.as_ref() {
        // MUST chain into the SAME `DimCheck`. `DimCheck::bound` is
        // per-instance (dim_check.rs), so a fresh `DimCheck::new()` binds `"D"`
        // to plane_aux's OWN row count and never compares it to the splat
        // count — an `[N - 1, 4]` input would validate clean and then read out
        // of bounds at `plane_aux[global_gid * PLANE_AUX_LANES + 3]` in both
        // the forward and backward rasterizers. That is silently-zero garbage
        // (a plausible-looking plane) under the bounds-checked wgpu launch and
        // undefined behaviour on the unchecked native-MSL backward launch.
        let shape = plane_aux.shape();
        // `check_dims` zips the shape against the bound list, so a rank-1 input
        // would skip the lane bound entirely. Pin the rank explicitly rather
        // than relying on that zip.
        assert_eq!(
            shape.rank(),
            2,
            "plane_aux must be rank 2 [total_splats, PLANE_AUX_LANES], got rank {}",
            shape.rank()
        );
        assert_eq!(
            shape[0],
            transforms.shape()[0],
            "plane_aux row count must equal the splat count",
        );
        assert_eq!(
            shape[1],
            kernels::helpers::PLANE_AUX_LANES_USIZE,
            "plane_aux must have PLANE_AUX_LANES columns",
        );
        dim_check.check_dims(
            "plane_aux",
            plane_aux,
            &[
                "D".into(),
                (kernels::helpers::PLANE_AUX_LANES_USIZE as u32).into(),
            ],
        );
    }

    let total_splats = transforms.shape()[0] as u32;
    let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape()[1] as u32);
    let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

    let half_max_render_fov =
        ((camera.fov_x as f32).hypot(camera.fov_y as f32) * 1.05).min(2.0 * PI - 1e-6) * 0.5;
    let pinhole_params = camera.build_pinhole_params(img_size);

    let mut project_uniforms = shaders::helpers::ProjectUniforms {
        viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
        camera_model: camera.camera_model,
        half_max_render_fov,
        pinhole_params,
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        img_size: img_size.into(),
        tile_bounds: calc_tile_bounds_for_dims(img_size, tile_width, tile_height).into(),
        sh_degree,
        total_splats,
        num_visible: 0, // num_visible — not yet known.
        jacobian_clamp_limits: calculate_jacobian_clamp_limits(
            img_size,
            pinhole_params,
            camera.camera_model,
        ),
    };

    let device = transforms.device.clone();
    let client = transforms.client.clone();

    // CubeCL binds every tensor argument even when a comptime flag removes all
    // of its accesses, so the plane slot always needs SOME buffer.
    //
    // `create_tensor` rather than `float_zeros`: the placeholder is never read,
    // so zero-filling it costs a whole extra GPU dispatch per render — on the
    // viewer's per-frame path — to initialise four bytes nothing looks at. What
    // remains is a 4-byte suballocation from CubeCL's memory pool, with no
    // driver allocation and no dispatch.
    //
    // Re-binding an already-bound tensor (e.g. `projected_splats`) instead would
    // be free, but it does NOT work: CubeCL binds every tensor as
    // `STORAGE_READ_WRITE` regardless of the kernel declaring `&Tensor`, so wgpu
    // rejects the second binding with "conflicting usages ... exclusive usage".
    // Measured, not assumed — it fails `brush-bench-test`'s render tests.
    let plane_aux_arg = plane_aux
        .clone()
        .unwrap_or_else(|| create_tensor([1], &device, DType::F32));

    // No projection buffer can be sliced or dispatched for an empty
    // scene. Rasterizing zero tile ranges still gives both output modes
    // their normal contracts: RGBA f32 for backward-enabled renders and
    // packed RGBA8 bits for forward-only renders.
    if total_splats == 0 {
        let tile_bounds: glam::UVec2 = project_uniforms.tile_bounds.into();
        let num_tiles = tile_bounds.x * tile_bounds.y;
        let tile_offsets = MainBackendBase::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            &device,
            IntDType::U32,
        );
        let projected_splats = MainBackendBase::float_zeros(
            [1, kernels::helpers::PROJECTED_LANES_USIZE].into(),
            &device,
            FloatDType::F32,
        );
        let compact_gid_from_isect = MainBackendBase::int_zeros([1].into(), &device, IntDType::U32);
        let global_from_compact_gid =
            MainBackendBase::int_zeros([1].into(), &device, IntDType::U32);

        // Reuse the already-resolved empty input allocation for true
        // zero-length aux tensors. Calling `float_zeros([0])` would itself
        // attempt a zero-sized fill dispatch on some CubeCL backends.
        let max_radius = MainBackendBase::float_reshape(transforms.clone(), [0].into());
        let visible = if bwd_info {
            MainBackendBase::float_reshape(transforms.clone(), [0].into())
        } else {
            MainBackendBase::float_zeros([1].into(), &device, FloatDType::F32)
        };

        let out_dim = if bwd_info {
            kernels::helpers::raster_out_channels(render_depth, render_plane) as usize
        } else {
            1
        };
        let out_img = create_tensor(
            [img_size.y as usize, img_size.x as usize, out_dim],
            &device,
            DType::F32,
        );
        let (out_packed_arg, out_f32_arg) = if bwd_info {
            (create_tensor([1], &device, DType::U32), out_img.clone())
        } else {
            (out_img.clone(), create_tensor([1], &device, DType::F32))
        };

        tracing::trace_span!("RasterizeEmpty").in_scope(|| {
            let uniforms = RasterizeUniformsLaunch::new(
                project_uniforms.tile_bounds[0],
                project_uniforms.img_size[0],
                project_uniforms.img_size[1],
                background.x,
                background.y,
                background.z,
            );
            kernels::rasterize::rasterize_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_tiles * tile_size, tile_size),
                CubeDim::new_1d(tile_size),
                compact_gid_from_isect.clone().into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                plane_aux_arg.clone().into_tensor_arg(),
                out_packed_arg.into_tensor_arg(),
                out_f32_arg.into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                visible.clone().into_tensor_arg(),
                uniforms,
                // Precomputed divisor for the tiles-per-row split in the
                // per-pixel index math (matches the populated launch below).
                project_uniforms.tile_bounds[0],
                bwd_info,
                smooth_cutoff,
                tile_width,
                tile_height,
                render_depth,
                render_plane,
            );
        });

        #[cfg(feature = "raster-census")]
        if bwd_info && let Some(request) = crate::raster_census::take_request() {
            let offsets = vec![0; num_tiles as usize * 2];
            let report = crate::raster_census::analyze(&crate::raster_census::RasterCensusInput {
                request,
                img_size,
                tile_bounds,
                tile_width,
                tile_height,
                num_visible: 0,
                num_intersections: 0,
                smooth_cutoff,
                pre_offsets: &offsets,
                post_offsets: &offsets,
                compact_gid_from_isect: &[],
                projected_splats: &[],
            })
            .expect("empty raster census analysis failed");
            crate::raster_census::emit(&report);
        }

        return RenderOutput {
            out_img,
            aux: RenderAuxInner {
                num_visible: 0,
                num_intersections: 0,
                visible,
                max_radius,
                tile_offsets,
                img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            project_uniforms,
            global_from_compact_gid,
        };
    }

    let (
        global_from_presort_gid,
        depths,
        intersect_counts,
        max_radius,
        num_visible_buf,
        num_intersections_buf,
    ) = {
        let project_uniforms: &shaders::helpers::ProjectUniforms = &project_uniforms;
        let _span = tracing::trace_span!("ProjectSplats").entered();

        let total_splats = project_uniforms.total_splats as usize;
        let num_visible_buf = MainBackendBase::int_zeros([1].into(), &device, IntDType::U32);
        let num_intersections_buf = MainBackendBase::int_zeros([1].into(), &device, IntDType::U32);
        let intersect_counts =
            MainBackendBase::int_zeros([total_splats].into(), &device, IntDType::U32);
        let max_radius =
            MainBackendBase::float_zeros([total_splats].into(), &device, FloatDType::F32);

        let global_from_presort_gid = create_tensor([total_splats], &device, DType::U32);
        let depths = create_tensor([total_splats], &device, DType::F32);

        let uniforms = project_uniforms.to_launch_object();

        kernels::project_forward::project_forward_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(
                project_uniforms.total_splats,
                kernels::project_forward::WG_SIZE,
            ),
            CubeDim::new_1d(kernels::project_forward::WG_SIZE),
            transforms.clone().into_tensor_arg(),
            raw_opacities.clone().into_tensor_arg(),
            global_from_presort_gid.clone().into_tensor_arg(),
            depths.clone().into_tensor_arg(),
            num_visible_buf.clone().into_tensor_arg(),
            intersect_counts.clone().into_tensor_arg(),
            num_intersections_buf.clone().into_tensor_arg(),
            max_radius.clone().into_tensor_arg(),
            uniforms,
            mip_splat,
            camera.camera_model,
            tile_width,
            tile_height,
        );
        (
            global_from_presort_gid,
            depths,
            intersect_counts,
            max_radius,
            num_visible_buf,
            num_intersections_buf,
        )
    };

    // Read both atomic counts in one transaction BEFORE the sort.
    let (num_visible, num_intersections) = if total_splats == 0 {
        (0, 0)
    } else {
        let tp = TransactionPrimitive::<MainBackendBase>::new(
            vec![],
            vec![],
            vec![num_visible_buf, num_intersections_buf],
            vec![],
        );
        let data = <MainBackendBase as TransactionOps<MainBackendBase>>::tr_execute(tp)
            .await
            .expect("Failed to read counts");
        let num_visible = data.read_ints[0]
            .clone()
            .into_vec::<u32>()
            .expect("num_visible")[0];
        let num_intersections = data.read_ints[1]
            .clone()
            .into_vec::<u32>()
            .expect("num_intersections")[0];
        (num_visible, num_intersections)
    };

    project_uniforms.num_visible = num_visible;

    let mip_splat = matches!(render_mode, SplatRenderMode::Mip);
    let img_size: glam::UVec2 = project_uniforms.img_size.into();
    let tile_bounds: glam::UVec2 = project_uniforms.tile_bounds.into();
    let num_visible_sz = (num_visible as usize).max(1);

    let global_from_compact_gid = {
        let depths = MainBackendBase::float_slice(depths, &[(0..num_visible_sz).into()]);
        let global_from_presort_gid =
            MainBackendBase::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);
        let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort")
            .in_scope(|| radix_argsort(depths, global_from_presort_gid, 32));
        global_from_compact_gid
    };
    let compact_counts =
        MainBackendBase::int_gather(0, intersect_counts, global_from_compact_gid.clone());
    let cum_tiles_hit =
        tracing::trace_span!("PrefixSumGaussHits").in_scope(|| prefix_sum(compact_counts));
    let projected_splats = create_tensor(
        [num_visible_sz, kernels::helpers::PROJECTED_LANES_USIZE],
        &device,
        DType::F32,
    );
    tracing::trace_span!("ProjectVisible").in_scope(|| {
        let uniforms = project_uniforms.to_launch_object();
        kernels::project_visible::project_visible_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_visible, kernels::project_visible::WG_SIZE),
            CubeDim::new_1d(kernels::project_visible::WG_SIZE),
            transforms.into_tensor_arg(),
            sh_coeffs.into_tensor_arg(),
            raw_opacities.into_tensor_arg(),
            global_from_compact_gid.clone().into_tensor_arg(),
            projected_splats.clone().into_tensor_arg(),
            uniforms,
            mip_splat,
            sh_degree,
            camera.camera_model,
        );
    });
    let num_tiles = tile_bounds.x * tile_bounds.y;
    let buffer_size = (num_intersections as usize).max(1);
    let tile_id_from_isect = create_tensor([buffer_size], &device, DType::U32);
    let compact_gid_from_isect = create_tensor([buffer_size], &device, DType::U32);
    tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
        kernels::map_gaussians::map_gaussians_to_intersect_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_visible, kernels::map_gaussians::WG_SIZE),
            CubeDim::new_1d(kernels::map_gaussians::WG_SIZE),
            projected_splats.clone().into_tensor_arg(),
            cum_tiles_hit.clone().into_tensor_arg(),
            tile_id_from_isect.clone().into_tensor_arg(),
            compact_gid_from_isect.clone().into_tensor_arg(),
            project_uniforms.tile_bounds[0],
            project_uniforms.tile_bounds[1],
            num_visible,
            tile_width,
            tile_height,
        );
    });
    let bits = u32::BITS - num_tiles.leading_zeros();
    let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
        .in_scope(|| radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits));
    let cube_dim = CubeDim::new_1d(256);
    let tile_offsets = MainBackendBase::int_zeros(
        [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
        &device,
        IntDType::U32,
    );
    tracing::trace_span!("GetTileOffsets").in_scope(|| {
        get_tile_offsets::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_intersections, cube_dim.x * CHECKS_PER_ITER),
            cube_dim,
            num_intersections,
            num_tiles,
            tile_id_from_isect.into_tensor_arg(),
            tile_offsets.clone().into_tensor_arg(),
        );
    });
    #[cfg(feature = "raster-census")]
    let raster_census_capture = if bwd_info {
        if let Some(request) = crate::raster_census::take_request() {
            let tp = TransactionPrimitive::<MainBackendBase>::new(
                vec![],
                vec![],
                vec![tile_offsets.clone()],
                vec![],
            );
            let data = <MainBackendBase as TransactionOps<MainBackendBase>>::tr_execute(tp)
                .await
                .expect("failed to read pre-raster tile offsets for raster census");
            let pre_offsets = data.read_ints[0]
                .clone()
                .into_vec::<u32>()
                .expect("raster census tile offsets must be u32");
            Some((request, pre_offsets))
        } else {
            None
        }
    } else {
        None
    };
    let out_dim = if bwd_info {
        kernels::helpers::raster_out_channels(render_depth, render_plane) as usize
    } else {
        1
    };
    let out_img = create_tensor(
        [img_size.y as usize, img_size.x as usize, out_dim],
        &device,
        DType::F32,
    );
    let (out_packed_arg, out_f32_arg) = if bwd_info {
        (create_tensor([1], &device, DType::U32), out_img.clone())
    } else {
        (out_img.clone(), create_tensor([1], &device, DType::F32))
    };
    let total_splats = project_uniforms.total_splats as usize;
    let visible = if bwd_info {
        MainBackendBase::float_zeros([total_splats].into(), &device, FloatDType::F32)
    } else {
        // Zero-init the dummy — `create_tensor` doesn't initialise, and
        // validate() may read this tensor to check its invariants.
        // Using `float_zeros` makes that read a well-defined no-op.
        MainBackendBase::float_zeros([1].into(), &device, FloatDType::F32)
    };
    tracing::trace_span!("Rasterize").in_scope(|| {
        let uniforms = RasterizeUniformsLaunch::new(
            project_uniforms.tile_bounds[0],
            project_uniforms.img_size[0],
            project_uniforms.img_size[1],
            background.x,
            background.y,
            background.z,
        );
        kernels::rasterize::rasterize_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_tiles * tile_size, tile_size),
            CubeDim::new_1d(tile_size),
            compact_gid_from_isect.clone().into_tensor_arg(),
            tile_offsets.clone().into_tensor_arg(),
            projected_splats.clone().into_tensor_arg(),
            plane_aux_arg.clone().into_tensor_arg(),
            out_packed_arg.into_tensor_arg(),
            out_f32_arg.into_tensor_arg(),
            global_from_compact_gid.clone().into_tensor_arg(),
            visible.clone().into_tensor_arg(),
            uniforms,
            // Precomputed divisor for the tiles-per-row split in the
            // per-pixel index math.
            project_uniforms.tile_bounds[0],
            bwd_info,
            smooth_cutoff,
            tile_width,
            tile_height,
            render_depth,
            render_plane,
        );
    });
    #[cfg(feature = "raster-census")]
    if let Some((request, pre_offsets)) = raster_census_capture {
        let tp = TransactionPrimitive::<MainBackendBase>::new(
            vec![projected_splats.clone()],
            vec![],
            vec![compact_gid_from_isect.clone(), tile_offsets.clone()],
            vec![],
        );
        let data = <MainBackendBase as TransactionOps<MainBackendBase>>::tr_execute(tp)
            .await
            .expect("failed to read raster census inputs");
        let projected_splats_host = data.read_floats[0]
            .clone()
            .into_vec::<f32>()
            .expect("raster census projected splats must be f32");
        let compact_gid_from_isect_host = data.read_ints[0]
            .clone()
            .into_vec::<u32>()
            .expect("raster census compact IDs must be u32");
        let post_offsets = data.read_ints[1]
            .clone()
            .into_vec::<u32>()
            .expect("raster census tile offsets must be u32");
        let report = crate::raster_census::analyze(&crate::raster_census::RasterCensusInput {
            request,
            img_size,
            tile_bounds,
            tile_width,
            tile_height,
            num_visible,
            num_intersections,
            smooth_cutoff,
            pre_offsets: &pre_offsets,
            post_offsets: &post_offsets,
            compact_gid_from_isect: &compact_gid_from_isect_host,
            projected_splats: &projected_splats_host,
        })
        .expect("raster census analysis failed");
        crate::raster_census::emit(&report);
    }
    RenderOutput {
        out_img,
        aux: RenderAuxInner {
            num_visible,
            num_intersections,
            visible,
            max_radius,
            tile_offsets,
            img_size,
        },
        projected_splats,
        compact_gid_from_isect,
        project_uniforms,
        global_from_compact_gid,
    }
}
