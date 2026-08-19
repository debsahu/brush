//! Replay of the independent alpha-compositing VJP golden vectors through the
//! real raster backward kernel.
//!
//! Source of truth: `docs/superpowers/specs/2026-08-19-alpha-vjp-derivation.md`
//! and its executable counterpart `analyze/vjp_reference/` in the outer
//! `slam` repo (numpy, float64, finite-difference validated, 83/83 checks).
//! `vjp_golden_vectors.json` beside this file is a verbatim copy of that
//! generator's output — regenerate it there, never edit it here.
//!
//! **Why this test exists.** The shipped backward is a *deliberately modified*
//! gradient: the centre-depth channel's alpha term is dropped while RGB and the
//! four PGSR plane channels keep theirs (plan section 4.5). A modified gradient
//! is not the derivative of any scalar function, so finite differences cannot
//! validate it as a whole. The identified top failure mode is that the plane
//! `λ = 1` term is never actually written: approach B then silently degenerates
//! into approach A, the forward stays bit-identical, every gradient stays finite
//! and nonzero through the value path, `plane_forward_parity_a_vs_b` still
//! passes, and the ablation concludes "B ≈ A" for the wrong reason. Only a
//! VALUE comparison catches that, which is what this is.
//!
//! It drives `SplatBwdOps::rasterize_bwd` directly with a hand-built projected
//! buffer, so no projection, SH evaluation or tile binning sits between the
//! golden numbers and the kernel — the comparison is at exactly the level the
//! derivation defines.
//!
//! Only the `__mixed_contract` cases are replayable. The `__all_live` siblings
//! describe a configuration this kernel cannot produce (depth in the weight
//! path); they exist in the reference to prove the drop is a clean subtraction.

use crate::{
    bwd::burn_glue::SplatBwdOps,
    kernels::helpers::{PLANE_AUX_LANES_USIZE, PROJECTED_LANES_USIZE},
};
use brush_cube::{MainBackendBase, Runtime};
use burn::{
    backend::ops::FloatTensorOps,
    tensor::{DType, TensorData},
};
use burn_wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use glam::{UVec2, Vec3};
use serde_json::Value;
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// Golden channel order is `[r, g, b, depth, n_x, n_y, n_z, offset]`; the
/// rendered image is `[r, g, b, alpha, depth, n_x, n_y, n_z, offset]`.
const GOLDEN_DEPTH: usize = 3;
const GOLDEN_PLANE: usize = 4;

fn cube_tensor_f32<const D: usize>(
    device: &WgpuDevice,
    shape: [usize; D],
    data: &[f32],
) -> CubeTensor<WgpuRuntime> {
    let expect: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        expect,
        "shape {shape:?} vs {} values",
        data.len()
    );
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

fn cube_tensor_u32<const D: usize>(
    device: &WgpuDevice,
    shape: [usize; D],
    data: &[u32],
) -> CubeTensor<WgpuRuntime> {
    let client = WgpuRuntime::client(device);
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        burn::tensor::Shape::new(shape),
        handle,
        DType::U32,
    )
}

async fn read_f32(tensor: CubeTensor<WgpuRuntime>) -> Vec<f32> {
    let data: TensorData = MainBackendBase::float_into_data(tensor)
        .await
        .expect("readback");
    data.as_slice::<f32>().expect("f32 tensor").to_vec()
}

fn nums(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect()
}

/// Relative tolerance for f32-kernel vs float64-reference.
///
/// The reference is float64 and the kernel is f32, so ~1e-7 relative is the
/// floor. `1e-4` leaves room for the `three_splat_alpha_near_zero` case, whose
/// `ra` and smoothstep derivative amplify f32 rounding, without leaving room
/// for a missing or misplaced term: every failure mode in the derivation's
/// checklist changes a value by a factor, not by a few ulps.
const REL_TOL: f64 = 1.0e-4;
const ABS_FLOOR: f64 = 1.0e-6;

fn check(case: &str, quantity: &str, splat: usize, actual: f64, expected: f64) -> Option<String> {
    let tol = ABS_FLOOR + REL_TOL * expected.abs().max(actual.abs());
    if (actual - expected).abs() <= tol {
        return None;
    }
    Some(format!(
        "{case} / {quantity}[splat {splat}]: kernel {actual:.9e} vs reference {expected:.9e} \
         (|Δ| = {:.3e} > tol {tol:.3e})",
        (actual - expected).abs()
    ))
}

/// Every `__mixed_contract` golden case, replayed through `rasterize_bwd`.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn raster_backward_matches_the_independent_vjp_reference() {
    let device = brush_cube::test_helpers::test_device().await;
    let golden: Value = serde_json::from_str(include_str!("vjp_golden_vectors.json"))
        .expect("golden vectors must parse");

    let background = {
        let ch = golden["_channels"].as_array().expect("channels");
        Vec3::new(
            ch[0]["background"].as_f64().expect("bg r") as f32,
            ch[1]["background"].as_f64().expect("bg g") as f32,
            ch[2]["background"].as_f64().expect("bg b") as f32,
        )
    };

    let mut failures: Vec<String> = Vec::new();
    let mut checked_cases = 0usize;

    for case in golden["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        if !name.ends_with("__mixed_contract") {
            continue;
        }
        checked_cases += 1;

        let splats = case["input"]["splats_front_to_back"]
            .as_array()
            .expect("splats");
        let n = splats.len();
        let v_out = nums(&case["input"]["v_out"]);
        let v_alpha_out = case["input"]["v_alpha_out"].as_f64().expect("v_alpha_out");
        let out = nums(&case["forward"]["out"]);
        let alpha_out = case["forward"]["alpha_out"].as_f64().expect("alpha_out");

        // One pixel, so its centre is (0.5, 0.5) and `delta_xy` is exactly the
        // kernel's `splat.xy - pixel_coord`.
        let img_size = UVec2::ONE;
        let mut projected = vec![0.0f32; n * PROJECTED_LANES_USIZE];
        let mut plane = vec![0.0f32; n * PLANE_AUX_LANES_USIZE];
        for (i, splat) in splats.iter().enumerate() {
            let delta = nums(&splat["delta_xy"]);
            let conic = nums(&splat["conic"]);
            let values = nums(&splat["values"]);
            let base = i * PROJECTED_LANES_USIZE;
            projected[base] = 0.5 + delta[0] as f32;
            projected[base + 1] = 0.5 + delta[1] as f32;
            projected[base + 2] = conic[0] as f32;
            projected[base + 3] = conic[1] as f32;
            projected[base + 4] = conic[2] as f32;
            projected[base + 5] = splat["opacity"].as_f64().expect("opacity") as f32;
            projected[base + 6] = values[0] as f32;
            projected[base + 7] = values[1] as f32;
            projected[base + 8] = values[2] as f32;
            projected[base + 9] = values[GOLDEN_DEPTH] as f32;
            for lane in 0..PLANE_AUX_LANES_USIZE {
                plane[i * PLANE_AUX_LANES_USIZE + lane] = values[GOLDEN_PLANE + lane] as f32;
            }
        }

        // Rendered image and cotangent, in the kernel's channel order.
        let out_img: Vec<f32> = [
            out[0],
            out[1],
            out[2],
            alpha_out,
            out[GOLDEN_DEPTH],
            out[GOLDEN_PLANE],
            out[GOLDEN_PLANE + 1],
            out[GOLDEN_PLANE + 2],
            out[GOLDEN_PLANE + 3],
        ]
        .iter()
        .map(|v| *v as f32)
        .collect();
        let v_output: Vec<f32> = [
            v_out[0],
            v_out[1],
            v_out[2],
            v_alpha_out,
            v_out[GOLDEN_DEPTH],
            v_out[GOLDEN_PLANE],
            v_out[GOLDEN_PLANE + 1],
            v_out[GOLDEN_PLANE + 2],
            v_out[GOLDEN_PLANE + 3],
        ]
        .iter()
        .map(|v| *v as f32)
        .collect();

        // Front-to-back order IS compact order here, and the single tile spans
        // every splat.
        let ids: Vec<u32> = (0..n as u32).collect();
        let grads = <MainBackendBase as SplatBwdOps>::rasterize_bwd(
            cube_tensor_f32(&device, [1, 1, 9], &out_img),
            cube_tensor_f32(&device, [n, PROJECTED_LANES_USIZE], &projected),
            cube_tensor_u32(&device, [n], &ids),
            cube_tensor_u32(&device, [1, 1, 2], &[0, n as u32]),
            Some(cube_tensor_f32(&device, [n, PLANE_AUX_LANES_USIZE], &plane)),
            cube_tensor_u32(&device, [n], &ids),
            background,
            img_size,
            cube_tensor_f32(&device, [1, 1, 9], &v_output),
            // The golden vectors are generated with the C^1 smoothstep, and the
            // three-splat case deliberately puts one alpha INSIDE the band, so
            // the `(w + alpha * w')` chain is exercised rather than collapsing
            // to 1.
            true,
            true,
            true,
        );

        let lanes = crate::bwd::COMPACT_GRAD_LANES as usize;
        let plane_start = crate::bwd::PLANE_GRAD_LANE_START as usize;
        let combined = read_f32(grads.v_combined).await;
        assert_eq!(combined.len(), n * lanes, "{name}: v_combined width");
        assert!(
            combined.iter().all(|v| v.is_finite()),
            "{name}: non-finite gradient lane"
        );

        let expected = &case["expected"];
        let v_opacity = nums(&expected["v_opacity"]);
        let v_conic = expected["v_conic"].as_array().expect("v_conic");
        let v_means2d = expected["v_means2d"].as_array().expect("v_means2d");
        let v_values = expected["v_values"].as_array().expect("v_values");

        for i in 0..n {
            let row = &combined[i * lanes..(i + 1) * lanes];
            let m = nums(&v_means2d[i]);
            let c = nums(&v_conic[i]);
            let v = nums(&v_values[i]);

            let mut push = |q: &str, actual: f32, expected: f64| {
                if let Some(msg) = check(name, q, i, f64::from(actual), expected) {
                    failures.push(msg);
                }
            };
            push("v_means2d.x", row[0], m[0]);
            push("v_means2d.y", row[1], m[1]);
            push("v_conic.c00", row[2], c[0]);
            push("v_conic.c01", row[3], c[1]);
            push("v_conic.c11", row[4], c[2]);
            push("v_value.r", row[5], v[0]);
            push("v_value.g", row[6], v[1]);
            push("v_value.b", row[7], v[2]);
            // Lane 8 is the gradient w.r.t. the PROJECTED opacity (`color_a`),
            // which is the reference's `v_opacity`. Lane 9 is the refine-weight
            // statistic and has no counterpart in the reference.
            push("v_opacity", row[8], v_opacity[i]);
            push("v_value.depth", row[10], v[GOLDEN_DEPTH]);
            for lane in 0..PLANE_AUX_LANES_USIZE {
                push(
                    &format!("v_value.plane{lane}"),
                    row[plane_start + lane],
                    v[GOLDEN_PLANE + lane],
                );
            }
        }
    }

    assert_eq!(
        checked_cases, 3,
        "expected the three __mixed_contract golden cases; the JSON copy may be stale"
    );
    assert!(
        failures.is_empty(),
        "raster backward disagrees with the independent VJP reference:\n  {}",
        failures.join("\n  "),
    );
}
