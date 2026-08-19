//! Independent CPU reference for the forward raster stage.
//!
//! The oracle deliberately walks every globally depth-sorted projected splat
//! for every pixel. It does not consume the GPU tile lists, so a candidate
//! rasterizer cannot make a missing tile candidate disappear from both the
//! implementation and its reference.

use crate::{
    camera::Camera,
    gaussian_splats::{RasterPass, Rasterizer, SplatRenderMode},
    kernels::camera_model::CameraModel,
};
use brush_cube::{MainBackendBase, Runtime};
use burn::{
    backend::{
        TensorMetadata,
        ops::{FloatTensorOps, IntTensorOps},
    },
    tensor::{DType, TensorData},
};
use burn_wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use glam::{UVec2, Vec3};
use wasm_bindgen_test::wasm_bindgen_test;

// Must match the kernel's projected-splat stride. The buffer carries 10 lanes:
//   0:xy_x 1:xy_y 2:conic_x 3:conic_y 4:conic_z 5:color_a
//   6:color_r 7:color_g 8:color_b 9:depth
// (see crate::kernels::helpers::PROJECTED_LANES). Sourced from that constant so
// the oracle stride can never silently drift from the kernel layout again: a
// hardcoded 9 here, stale after the depth lane was added (helpers.rs, gsplat
// depth-loss merge), is exactly what misaligned this oracle's per-splat reads.
// The RGBA forward composite below uses lanes 0..=8 only; the depth lane (9) is
// carried in the buffer but not blended into the color output.
const PROJECTED_LANES: usize = crate::kernels::helpers::PROJECTED_LANES as usize;
/// Lanes per splat in the PGSR plane-auxiliary buffer. Sourced from the same
/// shared constant the kernels use — see the note above; this file was itself a
/// casualty of a re-literalized stride once already.
const PLANE_AUX_LANES: usize = crate::kernels::helpers::PLANE_AUX_LANES_USIZE;
const ALPHA_CUTOFF_MID: f32 = 1.0 / 255.0;
const ALPHA_CUTOFF_BAND: f32 = 1.0e-3;

/// Channels the backward-enabled rendered image carries for a given mode.
/// Derived, never restated — same constant the forward kernel, the backward's
/// `pix_state`, and the host `out_dim` allocation all use.
fn out_channels(render_depth: bool, render_plane: bool) -> usize {
    crate::kernels::helpers::raster_out_channels(render_depth, render_plane) as usize
}

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn cutoff_weight(alpha: f32, smooth_cutoff: bool) -> f32 {
    if !smooth_cutoff {
        return if alpha >= ALPHA_CUTOFF_MID { 1.0 } else { 0.0 };
    }

    let low = ALPHA_CUTOFF_MID - 0.5 * ALPHA_CUTOFF_BAND;
    let t = ((alpha - low) / ALPHA_CUTOFF_BAND).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rasterize globally depth-sorted projected splats without any spatial bins.
fn rasterize_reference(
    projected: &[f32],
    img_size: UVec2,
    background: Vec3,
    smooth_cutoff: bool,
) -> Vec<f32> {
    rasterize_reference_full(projected, None, false, img_size, background, smooth_cutoff)
}

/// As [`rasterize_reference`], but also compositing the centre-depth lane and
/// the four PGSR plane-auxiliary lanes.
///
/// `plane_compact` is `[num_splats * PLANE_AUX_LANES]` in the SAME order as
/// `projected` (i.e. compact / depth-sorted), so the caller does the
/// compact -> global permutation the kernel does with
/// `global_from_compact_gid`. Keeping the permutation on the caller's side is
/// deliberate: the oracle then reproduces the composite from first principles
/// and an indexing bug in the kernel's own lookup cannot hide in both.
///
/// Depth and the plane lanes have NO background term (matching the kernel) and
/// are raw composited sums with no alpha division.
fn rasterize_reference_full(
    projected: &[f32],
    plane_compact: Option<&[f32]>,
    render_depth: bool,
    img_size: UVec2,
    background: Vec3,
    smooth_cutoff: bool,
) -> Vec<f32> {
    assert_eq!(projected.len() % PROJECTED_LANES, 0);
    let num_splats = projected.len() / PROJECTED_LANES;
    if let Some(plane) = plane_compact {
        assert_eq!(plane.len(), num_splats * PLANE_AUX_LANES);
    }

    let chans = out_channels(render_depth, plane_compact.is_some());
    let plane_off = crate::kernels::helpers::plane_channel_offset(render_depth) as usize;
    let mut image = vec![0.0; img_size.x as usize * img_size.y as usize * chans];
    for y in 0..img_size.y {
        for x in 0..img_size.x {
            let pixel_x = x as f32 + 0.5;
            let pixel_y = y as f32 + 0.5;
            let mut transmittance = 1.0f32;
            let mut rgb = Vec3::ZERO;
            let mut depth = 0.0f32;
            let mut plane = [0.0f32; PLANE_AUX_LANES];

            for (index, splat) in projected.chunks_exact(PROJECTED_LANES).enumerate() {
                let dx = pixel_x - splat[0];
                let dy = pixel_y - splat[1];
                let sigma = 0.5 * (splat[2] * dx * dx + splat[4] * dy * dy) + splat[3] * dx * dy;
                let alpha = (splat[5] * (-sigma).exp()).min(0.999);
                let weight = cutoff_weight(alpha, smooth_cutoff);

                if sigma >= 0.0 && weight > 0.0 {
                    let effective_alpha = alpha * weight;
                    let next_transmittance = transmittance * (1.0 - effective_alpha);
                    // Match the GPU kernel: the splat that crosses the early-out
                    // threshold is not accumulated.
                    if next_transmittance <= 1.0e-4 {
                        break;
                    }

                    let visibility = effective_alpha * transmittance;
                    rgb.x += splat[6].max(0.0) * visibility;
                    rgb.y += splat[7].max(0.0) * visibility;
                    rgb.z += splat[8].max(0.0) * visibility;
                    if render_depth {
                        depth += splat[9] * visibility;
                    }
                    if let Some(aux) = plane_compact {
                        for (lane, value) in plane.iter_mut().enumerate() {
                            *value += aux[index * PLANE_AUX_LANES + lane] * visibility;
                        }
                    }
                    transmittance = next_transmittance;
                }
            }

            rgb += background * transmittance;
            let base = (y as usize * img_size.x as usize + x as usize) * chans;
            image[base] = rgb.x;
            image[base + 1] = rgb.y;
            image[base + 2] = rgb.z;
            image[base + 3] = 1.0 - transmittance;
            if render_depth {
                image[base + 4] = depth;
            }
            if plane_compact.is_some() {
                image[base + plane_off..base + plane_off + PLANE_AUX_LANES].copy_from_slice(&plane);
            }
        }
    }
    image
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{label}[{index}]: actual={actual:e}, expected={expected:e}, error={error:e}, tolerance={tolerance:e}",
        );
    }
}

#[test]
fn oracle_composites_one_splat_and_clamps_negative_color() {
    // Trailing lane is the unused depth channel (see PROJECTED_LANES).
    let projected = [0.5, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.5, -1.0, 0.0];
    let image = rasterize_reference(&projected, UVec2::ONE, Vec3::new(0.2, 0.4, 0.6), false);
    assert_close("one splat", &image, &[0.6, 0.45, 0.3, 0.5], 1.0e-6);
}

#[test]
fn oracle_does_not_accumulate_the_early_out_splat() {
    // Trailing lane per splat is the unused depth channel (see PROJECTED_LANES).
    let projected = [
        0.5, 0.5, 1.0, 0.0, 1.0, 0.999, 1.0, 0.0, 0.0, 0.0, // red
        0.5, 0.5, 1.0, 0.0, 1.0, 0.999, 0.0, 1.0, 0.0, 0.0, // green
    ];
    let image = rasterize_reference(&projected, UVec2::ONE, Vec3::ZERO, false);
    assert_close("early out", &image, &[0.999, 0.0, 0.0, 0.999], 1.0e-6);
}

#[test]
fn oracle_matches_hard_and_smooth_cutoff_definitions() {
    // Trailing lane is the unused depth channel (see PROJECTED_LANES).
    let projected = [
        0.5,
        0.5,
        1.0,
        0.0,
        1.0,
        ALPHA_CUTOFF_MID,
        1.0,
        0.0,
        0.0,
        0.0,
    ];
    let hard = rasterize_reference(&projected, UVec2::ONE, Vec3::ZERO, false);
    let smooth = rasterize_reference(&projected, UVec2::ONE, Vec3::ZERO, true);
    assert_close(
        "hard cutoff",
        &hard,
        &[ALPHA_CUTOFF_MID, 0.0, 0.0, ALPHA_CUTOFF_MID],
        1.0e-7,
    );
    assert_close(
        "smooth cutoff",
        &smooth,
        &[ALPHA_CUTOFF_MID * 0.5, 0.0, 0.0, ALPHA_CUTOFF_MID * 0.5],
        1.0e-7,
    );
}

fn cube_tensor<const D: usize>(
    device: &WgpuDevice,
    shape: [usize; D],
    data: &[f32],
) -> CubeTensor<WgpuRuntime> {
    let client = WgpuRuntime::client(device);
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        burn::tensor::Shape::new(shape),
        handle,
        DType::F32,
    )
}

async fn read_f32(tensor: CubeTensor<WgpuRuntime>) -> Vec<f32> {
    let data: TensorData = MainBackendBase::float_into_data(tensor)
        .await
        .expect("readback");
    data.as_slice::<f32>().expect("f32 tensor").to_vec()
}

/// Camera shared by every oracle scene.
fn oracle_camera() -> Camera {
    Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

/// Flat host-side buffers for the shared oracle scene.
struct OracleScene {
    transforms: Vec<f32>,
    sh: Vec<f32>,
    raw_opacity: Vec<f32>,
}

fn oracle_scene() -> OracleScene {
    let means = [
        [0.00, 0.00, 0.00],
        [0.08, -0.04, 0.07],
        [-0.12, 0.06, 0.14],
        [0.16, 0.10, 0.21],
        [-0.04, -0.14, 0.28],
        [0.02, 0.03, 0.35],
    ];
    let log_scales = [
        [-0.65, -1.00, -0.85],
        [-0.80, -0.60, -0.95],
        [-0.55, -0.90, -0.70],
        [-1.00, -0.55, -0.80],
        [-0.70, -0.75, -0.60],
        [-0.60, -0.65, -0.90],
    ];
    let rotations = [
        [1.0, 0.0, 0.0, 0.0],
        [0.92, 0.12, 0.24, 0.08],
        [0.81, -0.16, 0.31, 0.10],
        [0.74, 0.28, -0.12, 0.20],
        [0.88, -0.20, 0.14, 0.09],
        [0.79, 0.18, 0.22, -0.11],
    ];
    let mut transforms = Vec::with_capacity(means.len() * 10);
    for ((mean, rotation), scale) in means.iter().zip(rotations).zip(log_scales) {
        transforms.extend_from_slice(mean);
        transforms.extend_from_slice(&rotation);
        transforms.extend_from_slice(&scale);
    }
    let sh = [
        0.8, 0.2, -0.5, 0.1, 0.9, 0.3, -0.3, 0.4, 1.0, 0.7, -0.2, 0.5, 0.2, 0.6, 0.8, 0.9, 0.3, 0.1,
    ];
    let raw_opacity = [3.6, 3.1, 2.8, 3.4, 3.0, 3.8];

    OracleScene {
        transforms,
        sh: sh.to_vec(),
        raw_opacity: raw_opacity.to_vec(),
    }
}

async fn render_test_scene(
    rasterizer: Rasterizer,
    pass: RasterPass,
    img_size: UVec2,
) -> (Vec<f32>, Vec<f32>, u32, [usize; 3]) {
    let device = brush_cube::test_helpers::test_device().await;
    let scene = oracle_scene();
    let num_splats = scene.raw_opacity.len();

    let output = <MainBackendBase as crate::SplatRasterizerOps>::render_with_rasterizer(
        &oracle_camera(),
        img_size,
        cube_tensor(&device, [num_splats, 10], &scene.transforms),
        cube_tensor(&device, [num_splats, 1, 3], &scene.sh),
        cube_tensor(&device, [num_splats], &scene.raw_opacity),
        SplatRenderMode::Default,
        crate::gaussian_splats::RasterizationMode::Rgba,
        Vec3::new(0.13, 0.07, 0.19),
        pass,
        rasterizer,
    )
    .await;
    output.clone().validate().await;

    let num_visible = output.aux.num_visible;
    let tile_offsets_shape = output.aux.tile_offsets.shape();
    let tile_offsets_shape = [
        tile_offsets_shape[0],
        tile_offsets_shape[1],
        tile_offsets_shape[2],
    ];
    let image = read_f32(output.out_img).await;
    let mut projected = read_f32(output.projected_splats).await;
    projected.truncate(num_visible as usize * PROJECTED_LANES);
    (image, projected, num_visible, tile_offsets_shape)
}

fn expected_tile_offsets_shape(img_size: UVec2, rasterizer: Rasterizer) -> [usize; 3] {
    let tile_width = rasterizer.tile_width();
    let tile_height = rasterizer.tile_height();
    [
        img_size.y.div_ceil(tile_height) as usize,
        img_size.x.div_ceil(tile_width) as usize,
        2,
    ]
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn both_selectors_match_the_independent_cpu_oracle() {
    let img_size = glam::uvec2(23, 19);
    let background = Vec3::new(0.13, 0.07, 0.19);

    for pass in [RasterPass::Backward, RasterPass::BackwardSmoothCutoff] {
        let (legacy_image, legacy_projected, legacy_visible, legacy_tile_shape) =
            render_test_scene(Rasterizer::Legacy, pass, img_size).await;
        let (candidate_image, candidate_projected, candidate_visible, candidate_tile_shape) =
            render_test_scene(Rasterizer::Candidate, pass, img_size).await;

        assert_eq!(candidate_visible, legacy_visible, "{pass:?} visible count");
        assert_eq!(
            legacy_tile_shape,
            expected_tile_offsets_shape(img_size, Rasterizer::Legacy),
            "{pass:?} legacy tile offsets",
        );
        assert_eq!(
            candidate_tile_shape,
            expected_tile_offsets_shape(img_size, Rasterizer::Candidate),
            "{pass:?} candidate tile offsets",
        );
        assert_close(
            &format!("{pass:?} selector image"),
            &candidate_image,
            &legacy_image,
            1.0e-6,
        );
        assert_close(
            &format!("{pass:?} projected splats"),
            &candidate_projected,
            &legacy_projected,
            1.0e-6,
        );

        let legacy_reference = rasterize_reference(
            &legacy_projected,
            img_size,
            background,
            pass.smooth_cutoff(),
        );
        let candidate_reference = rasterize_reference(
            &candidate_projected,
            img_size,
            background,
            pass.smooth_cutoff(),
        );
        assert_close(
            &format!("{pass:?} legacy oracle"),
            &legacy_image,
            &legacy_reference,
            2.0e-5,
        );
        assert_close(
            &format!("{pass:?} candidate oracle"),
            &candidate_image,
            &candidate_reference,
            2.0e-5,
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn selectors_match_across_tile_boundaries() {
    let background = Vec3::new(0.13, 0.07, 0.19);
    let pass = RasterPass::Backward;

    // Below, exactly on, and beyond the 16x8 candidate boundaries. The final
    // two cases cross the candidate and legacy grids in different ways.
    for (img_size, distinct_tile_grids) in [
        (glam::uvec2(15, 7), false),
        (glam::uvec2(16, 8), false),
        (glam::uvec2(17, 9), true),
        (glam::uvec2(9, 17), true),
    ] {
        let (legacy_image, legacy_projected, legacy_visible, legacy_tile_shape) =
            render_test_scene(Rasterizer::Legacy, pass, img_size).await;
        let (candidate_image, candidate_projected, candidate_visible, candidate_tile_shape) =
            render_test_scene(Rasterizer::Candidate, pass, img_size).await;

        assert_eq!(
            candidate_visible, legacy_visible,
            "{img_size} visible count"
        );
        assert_eq!(
            legacy_tile_shape,
            expected_tile_offsets_shape(img_size, Rasterizer::Legacy),
            "{img_size} legacy tile offsets",
        );
        assert_eq!(
            candidate_tile_shape,
            expected_tile_offsets_shape(img_size, Rasterizer::Candidate),
            "{img_size} candidate tile offsets",
        );
        assert_eq!(
            candidate_tile_shape != legacy_tile_shape,
            distinct_tile_grids,
            "{img_size} tile-grid distinction",
        );
        assert_close(
            &format!("{img_size} selector image"),
            &candidate_image,
            &legacy_image,
            1.0e-6,
        );
        assert_close(
            &format!("{img_size} projected splats"),
            &candidate_projected,
            &legacy_projected,
            1.0e-6,
        );

        let reference = rasterize_reference(&legacy_projected, img_size, background, false);
        assert_close(
            &format!("{img_size} legacy oracle"),
            &legacy_image,
            &reference,
            2.0e-5,
        );
        assert_close(
            &format!("{img_size} candidate oracle"),
            &candidate_image,
            &reference,
            2.0e-5,
        );
    }
}

async fn read_u32(tensor: CubeTensor<WgpuRuntime>) -> Vec<u32> {
    let data: TensorData = MainBackendBase::int_into_data(tensor)
        .await
        .expect("readback");
    data.as_slice::<u32>().expect("u32 tensor").to_vec()
}

/// Deterministic, plausible-looking plane parameters: a unit-ish camera-frame
/// normal plus a signed offset. The oracle only cares that the four lanes are
/// distinct per splat and per lane, which is what makes a stride or
/// compact->global indexing slip visible rather than self-cancelling.
fn plane_aux_values(num_splats: usize) -> Vec<f32> {
    (0..num_splats)
        .flat_map(|i| {
            let t = i as f32;
            let n = glam::vec3(0.13 * t - 0.4, 0.21 - 0.07 * t, 1.0).normalize();
            [n.x, n.y, n.z, 2.5 + 0.37 * t]
        })
        .collect()
}

/// Test 11 of the PGSR plane-render plan: the independent CPU oracle must
/// reproduce the main kernel's plane-channel composite.
///
/// This exists because adding lanes to the rendered image is exactly the change
/// that has silently broken strides in this crate before (fork commits
/// `a79f41b9`, `5e477544`, `ae2ec651` — and `5e477544` was this very file). The
/// oracle walks every depth-sorted splat per pixel and never consumes the GPU
/// tile lists, so a tile-binning or stride bug cannot disappear from both sides.
/// Both rasterizer selectors are checked, so the 16x8 native-MSL training tile
/// layout is covered too.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn plane_channels_match_the_independent_cpu_oracle() {
    let img_size = glam::uvec2(23, 19);
    let background = Vec3::new(0.13, 0.07, 0.19);

    for rasterizer in [Rasterizer::Legacy, Rasterizer::Candidate] {
        for pass in [RasterPass::Backward, RasterPass::BackwardSmoothCutoff] {
            let (image, projected, plane_compact, chans) =
                render_plane_test_scene(rasterizer, pass, img_size).await;

            assert_eq!(
                chans,
                out_channels(true, true),
                "{rasterizer:?}/{pass:?} plane render must emit rgba + depth + 4 plane channels",
            );

            let reference = rasterize_reference_full(
                &projected,
                Some(&plane_compact),
                true,
                img_size,
                background,
                pass.smooth_cutoff(),
            );
            assert_close(
                &format!("{rasterizer:?}/{pass:?} plane oracle"),
                &image,
                &reference,
                2.0e-5,
            );
        }
    }
}

/// Renders the shared oracle scene with the PGSR plane channels on, and returns
/// `(image, projected, plane_aux_in_compact_order, channels_per_pixel)`.
async fn render_plane_test_scene(
    rasterizer: Rasterizer,
    pass: RasterPass,
    img_size: UVec2,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let device = brush_cube::test_helpers::test_device().await;
    let scene = oracle_scene();
    let num_splats = scene.raw_opacity.len();
    let plane_values = plane_aux_values(num_splats);

    let output = crate::render::render_base_with_plane_aux(
        &oracle_camera(),
        img_size,
        cube_tensor(&device, [num_splats, 10], &scene.transforms),
        cube_tensor(&device, [num_splats, 1, 3], &scene.sh),
        cube_tensor(&device, [num_splats], &scene.raw_opacity),
        Some(cube_tensor(
            &device,
            [num_splats, PLANE_AUX_LANES],
            &plane_values,
        )),
        SplatRenderMode::Default,
        crate::gaussian_splats::RasterizationMode::RgbaDepthPlane,
        Vec3::new(0.13, 0.07, 0.19),
        pass,
        rasterizer,
    )
    .await;
    output.clone().validate().await;

    let num_visible = output.aux.num_visible as usize;
    let chans = output.out_img.shape()[2];
    let image = read_f32(output.out_img).await;
    let mut projected = read_f32(output.projected_splats).await;
    projected.truncate(num_visible * PROJECTED_LANES);
    let global_from_compact = read_u32(output.global_from_compact_gid).await;

    // Permute the global-gid-indexed plane buffer into compact (depth-sorted)
    // order, which is the order `projected` is in.
    let mut plane_compact = vec![0.0f32; num_visible * PLANE_AUX_LANES];
    for compact in 0..num_visible {
        let global = global_from_compact[compact] as usize;
        plane_compact[compact * PLANE_AUX_LANES..(compact + 1) * PLANE_AUX_LANES].copy_from_slice(
            &plane_values[global * PLANE_AUX_LANES..(global + 1) * PLANE_AUX_LANES],
        );
    }

    (image, projected, plane_compact, chans)
}
