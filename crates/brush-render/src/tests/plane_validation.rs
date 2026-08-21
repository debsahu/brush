//! Shape validation for the PGSR plane-auxiliary input.
//!
//! The plane tensor is indexed by GLOBAL gid (`plane_aux[gid * PLANE_AUX_LANES + k]`)
//! in both `kernels::rasterize` and `bwd::kernels::rasterize_backwards`, so its
//! row count MUST equal the splat count. Nothing downstream re-derives it: a
//! short buffer reads past the end, which is silently-zero garbage under the
//! bounds-checked wgpu launch (a plausible-looking plane that trains) and
//! undefined behaviour on the unchecked native-MSL backward launch.
//!
//! The validation lives in the same `DimCheck` instance as the splat tensors,
//! because `DimCheck::bound` is per-instance: a separate `DimCheck::new()` binds
//! `"D"` to the plane tensor's own first dimension and compares it to nothing.

use crate::{
    camera::Camera,
    gaussian_splats::{RasterPass, RasterizationMode, Rasterizer, SplatRenderMode},
    kernels::{camera_model::CameraModel, helpers::PLANE_AUX_LANES_USIZE},
};
use burn::tensor::DType;
use burn_wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use glam::Vec3;

fn cube_tensor<const D: usize>(
    device: &WgpuDevice,
    shape: [usize; D],
    data: &[f32],
) -> CubeTensor<WgpuRuntime> {
    use brush_cube::Runtime;
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

/// Renders `n_splats` splats with a plane tensor of `plane_rows` rows.
async fn render_with_plane_rows(plane_rows: usize) {
    let device = brush_cube::test_helpers::test_device().await;
    let n = 4usize;
    let camera = Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );

    let mut transforms = Vec::new();
    for i in 0..n {
        let t = i as f32;
        transforms.extend_from_slice(&[0.1 * t, -0.05 * t, 0.02 * t]);
        transforms.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        transforms.extend_from_slice(&[-1.2, -1.5, -1.8]);
    }
    let sh: Vec<f32> = (0..n * 3).map(|i| 0.4 + 0.01 * i as f32).collect();
    let raw_opacity = vec![2.0f32; n];
    let plane: Vec<f32> = (0..plane_rows * PLANE_AUX_LANES_USIZE)
        .map(|i| 0.1 + 0.01 * i as f32)
        .collect();

    let _ = crate::render::render_base_with_plane_aux(
        &camera,
        glam::uvec2(16, 16),
        cube_tensor(&device, [n, 10], &transforms),
        cube_tensor(&device, [n, 1, 3], &sh),
        cube_tensor(&device, [n], &raw_opacity),
        Some(cube_tensor(
            &device,
            [plane_rows, PLANE_AUX_LANES_USIZE],
            &plane,
        )),
        SplatRenderMode::Default,
        RasterizationMode::RgbaDepthPlane,
        Vec3::ZERO,
        RasterPass::Backward,
        Rasterizer::Legacy,
    )
    .await;
}

#[tokio::test]
async fn plane_aux_with_matching_rows_is_accepted() {
    render_with_plane_rows(4).await;
}

/// A short plane buffer must be rejected loudly, not read out of bounds.
#[tokio::test]
#[should_panic(expected = "plane_aux row count must equal the splat count")]
async fn plane_aux_with_too_few_rows_panics() {
    render_with_plane_rows(3).await;
}

/// ...and a long one too: an over-long buffer is a caller/geometry mismatch just
/// as much as a short one, and silently ignoring the tail hides it.
#[tokio::test]
#[should_panic(expected = "plane_aux row count must equal the splat count")]
async fn plane_aux_with_too_many_rows_panics() {
    render_with_plane_rows(5).await;
}
