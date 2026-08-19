//! WS-A test 8: finite-difference validation of the PGSR **plane-aux** gradient
//! path (plan §5 item 8, templated on `dig_features.rs`'s
//! `feature_gradients_match_finite_diff`).
//!
//! What is under test is the whole approach-A chain:
//!
//! ```text
//! transforms -> plane_features -> feature rasterizer -> plane_depth_from_features -> loss
//! ```
//!
//! and specifically that its analytic VJP into the **means** and the
//! **quaternions** matches central differences of the same function.
//!
//! # Why the rasterizer is fed UNPERTURBED geometry
//!
//! This is the load-bearing subtlety, not a shortcut. The feature rasterizer's
//! backward (`bwd/features_bwd.rs`) registers exactly ONE parent — the feature
//! values — so the compositing weights are **constants** by construction. That
//! is approach A's defining property (§4.5 row 2): geometry gradients arrive
//! through feature VALUES only. The function the tape differentiates is
//! therefore
//!
//! ```text
//! L(θ) = loss(composite(w̄, plane_features(θ)))      with w̄ held fixed
//! ```
//!
//! so the finite-difference reference has to hold `w̄` fixed too — hence the
//! base transforms go to `render_splat_features` while the perturbed ones go to
//! `plane_features`. Perturbing both would difference `L(θ, w(θ))`, whose total
//! derivative includes the weight path that A deliberately drops, and the test
//! would "fail" by measuring the approximation rather than the implementation.
//! That weight path is exactly what approach B (`--depth-source plane-fused`)
//! adds and what ablation arm 4 vs arm 5 is designed to price. Its own
//! finite-diff test (plan item 9) perturbs both, on purpose.
//!
//! PGSR: Chen et al. 2024, arXiv:2406.06521.

#![allow(clippy::missing_assert_message)]

use brush_loss::plane_depth_from_features;
use brush_render::bwd::render_splat_features;
use brush_render::{
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_train::train::plane_features;
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, Vec3};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const IMG: glam::UVec2 = glam::uvec2(48, 48);
/// Same thresholds the trainer pins at its call site.
const MIN_ALPHA: f32 = 0.5;
const MIN_DENOM: f32 = 0.05;
const MIN_DEPTH: f32 = 1e-3;
const MAX_DEPTH: f32 = 1e4;

fn test_camera() -> Camera {
    Camera::new(
        Vec3::new(0.2, -0.3, -5.0),
        Quat::from_euler(glam::EulerRot::XYZ, 0.10, -0.18, 0.05),
        0.7,
        0.7,
        glam::vec2(0.5, 0.5),
        Pinhole,
    )
}

/// A tilted slab whose per-splat log-scales are **well separated**.
///
/// This is the scene constraint specific to plane features. `splat_normals`
/// selects the thinnest axis with a detached `argmin`; two axes of nearly equal
/// scale sit on the discontinuity where that selection flips, and a central
/// difference that straddles it differences two different functions. `-1.2 /
/// -1.6 / -3.0` keeps every pair more than 0.4 apart in log space, which no
/// perturbation used here comes close to crossing.
fn slab(device: &Device) -> Splats {
    let q = Quat::from_rotation_y(0.5);
    let e1 = q * Vec3::new(1.0, 0.0, 0.0);
    let e2 = q * Vec3::new(0.0, 1.0, 0.0);

    let mut means = vec![];
    let n_side = 7;
    for iy in 0..n_side {
        for ix in 0..n_side {
            let f = |i: i32| (i as f32 / (n_side - 1) as f32) * 2.0 - 1.0;
            let p = e1 * f(ix) + e2 * f(iy);
            means.extend_from_slice(&[p.x, p.y, p.z]);
        }
    }
    let n = means.len() / 3;
    let rotations: Vec<f32> = (0..n).flat_map(|_| [q.w, q.x, q.y, q.z]).collect();
    let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.2, -1.6, -3.0]).collect();
    let sh: Vec<f32> = (0..n).flat_map(|_| [0.5, 0.5, 0.5]).collect();
    let opac: Vec<f32> = vec![4.0; n];

    Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh,
        opac,
        SplatRenderMode::Default,
        device,
    )
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn plane_aux_gradients_match_finite_diff() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = slab(&device);
    let camera = test_camera();
    let n = splats.num_splats() as usize;

    let base_transforms = splats.transforms.val();
    let base_opac = splats.raw_opacities.val();
    let base_vals: Vec<f32> = base_transforms
        .clone()
        .into_data_async()
        .await
        .expect("transform readback")
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(base_vals.len(), n * 10);

    let focal = camera.focal(IMG);
    let center = camera.center(IMG);

    // A FIXED depth target, 10% nearer than the unperturbed prediction, shared by
    // every evaluation below. Differencing a residual rather than the raw depth
    // is what makes this test numerically viable: `Σ z²` over a covered slab is
    // O(1e4), so a central difference of it loses ~4 significant digits to f32
    // cancellation before the derivative is even visible. `Σ (z − 0.9 z₀)²` is
    // two orders smaller for the same derivative, which buys back the precision.
    // It is also the shape of the real supervision — the trainer's depth loss is
    // a residual against a GT depth map, not a magnitude.
    let gt = {
        let feats = plane_features(base_transforms.clone(), &camera);
        let feat_img = render_splat_features(
            base_transforms.clone(),
            base_opac.clone(),
            feats,
            &camera,
            IMG,
            SplatRenderMode::Default,
        )
        .await;
        let (depth, _n, _v) = plane_depth_from_features(
            feat_img, focal.x, focal.y, center.x, center.y, MIN_ALPHA, MIN_DENOM, MIN_DEPTH,
            MAX_DEPTH,
        );
        depth.detach().mul_scalar(0.9)
    };

    let plane_loss = |vals: Vec<f32>| {
        let device = device.clone();
        let base_transforms = base_transforms.clone();
        let base_opac = base_opac.clone();
        let camera = camera.clone();
        let gt = gt.clone();
        async move {
            let theta: Tensor<2> =
                Tensor::from_data(TensorData::new(vals, [n, 10]), &device).require_grad();
            let feats = plane_features(theta.clone(), &camera);
            // Base geometry into the rasterizer: see the module comment.
            let feat_img = render_splat_features(
                base_transforms,
                base_opac,
                feats,
                &camera,
                IMG,
                SplatRenderMode::Default,
            )
            .await;
            let (depth, _normal, valid) = plane_depth_from_features(
                feat_img, focal.x, focal.y, center.x, center.y, MIN_ALPHA, MIN_DENOM, MIN_DEPTH,
                MAX_DEPTH,
            );
            let resid = (depth - gt) * valid;
            let loss = (resid.clone() * resid).sum();
            (loss, theta)
        }
    };

    let (loss, theta) = plane_loss(base_vals.clone()).await;
    let loss0: f32 = loss.clone().into_scalar_async().await.expect("readback");
    assert!(
        loss0 > 1.0,
        "the slab must actually be covered for this test to mean anything, loss {loss0}"
    );
    println!("plane-aux finite diff: base loss {loss0}");
    let grads = loss.backward();
    let analytic: Vec<f32> = theta
        .grad(&grads)
        .expect("plane depth must reach the transforms")
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .unwrap();

    // Scales must be EXACTLY untouched — the detached argmin (see `slab`).
    for i in 0..n {
        for c in 7..10 {
            assert_eq!(
                analytic[i * 10 + c],
                0.0,
                "log-scale {c} of splat {i} received a plane gradient; the \
                 thinnest-axis argmin is supposed to be detached"
            );
        }
    }

    // Probe means (channels 0..3) and quaternions (channels 3..7) on splats
    // spread across the slab. `eps` is small against the 0.4 log-scale margin and
    // against the plane's extent, but large against f32 noise in the loss.
    let eps = 1e-2f32;
    let mut probes = vec![];
    for splat in [0usize, 9, 24, 40] {
        for c in 0..7 {
            probes.push(splat * 10 + c);
        }
    }

    let mut checked = 0usize;
    let mut worst_rel = 0.0f32;
    for idx in probes {
        let mut plus = base_vals.clone();
        plus[idx] += eps;
        let mut minus = base_vals.clone();
        minus[idx] -= eps;
        let (lp, _) = plane_loss(plus).await;
        let (lm, _) = plane_loss(minus).await;
        let lp: f32 = lp.into_scalar_async().await.expect("readback");
        let lm: f32 = lm.into_scalar_async().await.expect("readback");
        let fd = (lp - lm) / (2.0 * eps);
        let an = analytic[idx];

        // Entries whose true derivative is ~0 (e.g. sliding a splat centre
        // TANGENT to its own plane leaves the offset unchanged) carry no
        // information and are dominated by differencing noise. Skip them
        // explicitly rather than hiding them under a denominator floor.
        if fd.abs() < 1e-1 && an.abs() < 1e-1 {
            continue;
        }
        checked += 1;
        let denom = fd.abs().max(an.abs());
        let rel = ((fd - an) / denom).abs();
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < 5e-2,
            "plane-aux gradient mismatch at transform entry {idx} \
             (splat {}, channel {}): finite-diff {fd} vs analytic {an}, relative {rel}",
            idx / 10,
            idx % 10
        );
    }

    assert!(
        checked >= 12,
        "only {checked} entries carried a usable derivative; the probe set is \
         not exercising the plane path"
    );
    println!("plane-aux finite diff: {checked} entries checked, worst relative error {worst_rel}");
}
