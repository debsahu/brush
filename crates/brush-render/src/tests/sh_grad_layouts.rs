//! The two SH-gradient layouts of the projection backward must agree.
//!
//! Upstream's layout writes one compact row per visible splat (plus a zero row
//! that culled splats gather), expanded to `[total_splats, coeffs, 3]` by a
//! gather through `compact_from_global`. This fork's coalesced native-MSL
//! materializer writes the dense tensor directly. The trainer registers
//! whichever one ran, so the two must be the same gradient; this compares them
//! on identical inputs, with the 3D-filter floor on so the in-kernel floor chain
//! is exercised too.

use crate::{
    bwd::render_bwd::{materialize_supported, project_bwd_impl},
    camera::Camera,
    gaussian_splats::{RasterPass, RasterizationMode, Rasterizer, SplatRenderMode},
    kernels::camera_model::CameraModel,
};
use brush_cube::{CubeDevice, MainBackendBase};
use burn::backend::{
    TensorMetadata,
    ops::{FloatTensorOps, IntTensorOps},
};
use burn::tensor::{DType, TensorData};
use burn_wgpu::CubeTensor;
use glam::Vec3;

const SH_DEGREE: u32 = 2;
const COEFFS: usize = 9;

fn cube_tensor<const D: usize>(device: &CubeDevice, shape: [usize; D], data: &[f32]) -> CubeTensor {
    let client = device.client();
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        burn::tensor::Shape::new(shape),
        handle,
        DType::F32,
    )
}

async fn read_f32(tensor: CubeTensor) -> Vec<f32> {
    let data: TensorData = MainBackendBase::float_into_data(tensor)
        .await
        .expect("readback");
    data.as_slice::<f32>().expect("f32 tensor").to_vec()
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        let tol = 1e-6 + 1e-5 * e.abs().max(a.abs());
        assert!(
            (e - a).abs() <= tol,
            "{label}[{i}]: compact+gather {e} != dense {a}"
        );
    }
}

#[tokio::test]
async fn compact_and_materialized_sh_gradients_agree() {
    let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
    let camera = Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );

    // Twelve splats: nine inside the frustum, three far off to the side so
    // the dense result carries real zero rows for culled splats.
    let n = 12usize;
    let mut transforms = Vec::with_capacity(n * 10);
    for i in 0..n {
        let t = i as f32;
        let x = if i >= 9 { 50.0 + t } else { 0.05 * (t - 4.0) };
        transforms.extend_from_slice(&[x, 0.03 * (t - 4.0), 0.04 * t]);
        transforms.extend_from_slice(&[0.9, 0.1 * (t % 3.0), -0.05 * t, 0.2]);
        transforms.extend_from_slice(&[-1.4 - 0.02 * t, -1.2, -1.6 + 0.01 * t]);
    }
    let sh: Vec<f32> = (0..n * COEFFS * 3)
        .map(|i| 0.3 + 0.05 * ((i * 7 % 11) as f32 - 5.0))
        .collect();
    let raw_opacity: Vec<f32> = (0..n).map(|i| 0.5 + 0.1 * i as f32).collect();
    // A floor in the regime where it actually inflates the smallest axes.
    let floor: Vec<f32> = (0..n).map(|i| 0.12 + 0.01 * i as f32).collect();

    let img_size = glam::uvec2(32, 32);
    let output = crate::render::render_base_with_plane_aux(
        &camera,
        img_size,
        cube_tensor(&device, [n, 10], &transforms),
        cube_tensor(&device, [n, COEFFS, 3], &sh),
        cube_tensor(&device, [n], &raw_opacity),
        cube_tensor(&device, [n], &floor),
        true,
        None,
        SplatRenderMode::Default,
        RasterizationMode::Rgba,
        Vec3::ZERO,
        RasterPass::Backward,
        Rasterizer::Legacy,
    )
    .await;
    let num_visible = output.aux.num_visible as usize;
    assert!(
        (2..n).contains(&num_visible),
        "fixture must leave some splats visible and some culled, got {num_visible}/{n}"
    );
    let uniforms = output.project_uniforms;
    assert!(
        materialize_supported(&device, n, &uniforms),
        "the coalesced materializer must be runnable here, or this test compares nothing"
    );

    // Raster gradients for every visible splat. The last visible row gets an
    // all-zero colour gradient so both paths' "no SH gradient" branch runs.
    let lanes = crate::bwd::COMPACT_GRAD_LANES as usize;
    let rgb = crate::bwd::RGB_LANE;
    let mut v_combined: Vec<f32> = (0..num_visible * lanes)
        .map(|i| 0.01 * ((i * 13 % 17) as f32 - 8.0))
        .collect();
    for c in 0..3 {
        v_combined[(num_visible - 1) * lanes + rgb + c] = 0.0;
    }

    let run = |materialize: bool| {
        project_bwd_impl(
            cube_tensor(&device, [n, 10], &transforms),
            cube_tensor(&device, [n, COEFFS, 3], &sh),
            cube_tensor(&device, [n], &raw_opacity),
            cube_tensor(&device, [n], &floor),
            true,
            output.global_from_compact_gid.clone(),
            uniforms,
            SplatRenderMode::Default,
            cube_tensor(&device, [num_visible, lanes], &v_combined),
            materialize,
        )
    };
    let compact = run(false);
    let dense = run(true);
    assert!(!compact.coeffs_dense && dense.coeffs_dense);
    assert_eq!(uniforms.sh_degree, SH_DEGREE);

    let gathered = MainBackendBase::float_select(
        compact.v_coeffs.clone(),
        0,
        output.compact_from_global.clone(),
    );
    assert_eq!(gathered.shape().dims::<3>(), [n, COEFFS, 3]);
    assert_eq!(dense.v_coeffs.shape().dims::<3>(), [n, COEFFS, 3]);
    let gathered = read_f32(gathered).await;
    let materialized = read_f32(dense.v_coeffs).await;

    // Discriminating power: a vacuous all-zero comparison would pass anything.
    let row = COEFFS * 3;
    let nonzero_rows = materialized
        .chunks(row)
        .filter(|r| r.iter().any(|v| *v != 0.0))
        .count();
    let zero_rows = n - nonzero_rows;
    assert!(
        nonzero_rows >= 2 && zero_rows >= 3,
        "fixture must produce both populated and zero SH rows ({nonzero_rows} / {zero_rows})"
    );
    assert_close("v_coeffs", &gathered, &materialized);

    // Everything else is compact in both layouts and must match exactly.
    for (label, a, b) in [
        ("v_transforms", compact.v_transforms, dense.v_transforms),
        ("v_raw_opac", compact.v_raw_opac, dense.v_raw_opac),
        (
            "v_refine_weight",
            compact.v_refine_weight,
            dense.v_refine_weight,
        ),
    ] {
        let a = read_f32(a).await;
        let b = read_f32(b).await;
        assert_eq!(a, b, "{label} must not depend on the SH layout");
    }

    // The zero row culled splats gather must really be zero.
    let compact_from_global: Vec<u32> = MainBackendBase::int_into_data(output.compact_from_global)
        .await
        .expect("readback")
        .as_slice::<u32>()
        .expect("u32")
        .to_vec();
    assert_eq!(
        compact_from_global.iter().filter(|v| **v == 0).count(),
        n - num_visible,
        "every culled splat must point at the zero row"
    );
}
