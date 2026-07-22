//! Edge-guidance (Canny) growth/replacement weighting — MRNF port, delta #4
//! (Phase 4).
//!
//! MRNF biases densification toward high-frequency image edges. LFS
//! (`training/strategies/mrnf.cpp:1440-1470`, `edge_rasterizer.cpp`) does this in
//! three stages, per sampled training view:
//!   1. A fused-Canny NMS edge map of the GT image
//!      (`kernels/image_kernels.cu:46`, itself adapted from spirulae-splat).
//!   2. An *alpha-blended* edge rasterization that accumulates, per gaussian,
//!      `sum_pixels (T·α)·edge(pixel)` — the gaussian's rendering contribution to
//!      pixels that lie on edges (`edge_compute/.../kernels_forward.cuh:386`).
//!   3. Positive-median normalization of both the edge map and the per-gaussian
//!      score, accumulated across a refine window, then
//!      `factor = score·0.25 + 1.0` multiplying the growth/replacement weights
//!      (`edge_guidance_factor`, mrnf.cpp:1456).
//!
//! # Fallback status (see design Part D, Phase 4)
//!
//! Stage 2 in LFS is a full tiled, front-to-back, alpha-compositing forward
//! rasterizer that writes a per-gaussian scalar. Porting that as a first-class
//! CubeCL kernel is the "genuinely new render-like subsystem" the design flags as
//! the only large piece of the MRNF port. This module instead lands a **burn-op
//! projection fallback**: it projects each gaussian's *center* to a pixel and
//! samples the edge map there, weighting by the gaussian's opacity as a proxy for
//! its rendering contribution. That captures the intent (grow more where edges
//! are) and is fully unit-testable via a projection-correctness test, but it is
//! an approximation of the alpha-blended accumulation and is:
//!   - **pinhole only** (equirect/fisheye return a zero score);
//!   - **center-sample, nearest-pixel** (no per-pixel alpha compositing, no
//!     bilinear tap, no NMS on the Canny map);
//!   - **opacity-weighted**, not transmittance·alpha weighted.
//!
//! The public surface (config flags, `RefineRecord` accumulator, growth/
//! replacement wiring) is the real scaffold; swapping `project_edge_scores` for a
//! CubeCL alpha-blended edge rasterizer is the remaining work to reach parity.

use burn::tensor::{Tensor, module::conv2d, ops::ConvOptions, s};
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;

// spirulae-splat 5x5 Gaussian blur (matches LFS `SPIRULAE_BLUR_5x5`,
// image_kernels.cu:21).
#[rustfmt::skip]
const SPIRULAE_BLUR_5X5: [f32; 25] = [
    2.0 / 159.0,  4.0 / 159.0,  5.0 / 159.0,  4.0 / 159.0, 2.0 / 159.0,
    4.0 / 159.0,  9.0 / 159.0, 12.0 / 159.0,  9.0 / 159.0, 4.0 / 159.0,
    5.0 / 159.0, 12.0 / 159.0, 15.0 / 159.0, 12.0 / 159.0, 5.0 / 159.0,
    4.0 / 159.0,  9.0 / 159.0, 12.0 / 159.0,  9.0 / 159.0, 4.0 / 159.0,
    2.0 / 159.0,  4.0 / 159.0,  5.0 / 159.0,  4.0 / 159.0, 2.0 / 159.0,
];

// spirulae-splat 3x3 gradient (LFS `SPIRULAE_CANNY_3x3`, image_kernels.cu:28).
#[rustfmt::skip]
const SOBEL_X: [f32; 9] = [
    -1.0, 0.0, 1.0,
    -2.0, 0.0, 2.0,
    -1.0, 0.0, 1.0,
];
// Transpose of SOBEL_X — LFS derives the second gradient by index-swapping the
// same 3x3 (`conv_weight_2 = SPIRULAE_CANNY_3x3[(cx+1)*3 + (cy+1)]`).
#[rustfmt::skip]
const SOBEL_Y: [f32; 9] = [
    -1.0, -2.0, -1.0,
     0.0,  0.0,  0.0,
     1.0,  2.0,  1.0,
];

/// Gradient-magnitude edge map of a `[H, W, 3]` RGB image (values in ~[0, 1]).
///
/// Mirrors the front of LFS's fused Canny (`launch_fused_canny_edge_filter_chw`):
/// luminance conversion, 5x5 Gaussian blur, 3x3 gradients, magnitude. It does
/// **not** apply the non-maximum-suppression thinning step (a per-pixel gather
/// along the gradient direction); the downstream positive-median normalization
/// makes the guidance robust to the resulting thicker edges, and NMS only sharpens
/// the bias rather than changing its sign. Runs on whichever backend `rgb` lives
/// on (call with an inner/detached tensor — this is non-differentiable bookkeeping).
pub(crate) fn canny_edge_map(rgb: Tensor<3>) -> Tensor<2> {
    let [h, w, _c] = rgb.dims();
    let device = rgb.device();

    let r = rgb.clone().slice(s![.., .., 0..1]);
    let g = rgb.clone().slice(s![.., .., 1..2]);
    let b = rgb.slice(s![.., .., 2..3]);
    // Rec.601 luma, matching LFS `0.299R + 0.587G + 0.114B`.
    let gray = r.mul_scalar(0.299) + g.mul_scalar(0.587) + b.mul_scalar(0.114);
    let gray = gray.reshape([1, 1, h as i32, w as i32]);

    let blur_w = Tensor::<1>::from_floats(SPIRULAE_BLUR_5X5, &device).reshape([1, 1, 5, 5]);
    let blurred = conv2d(gray, blur_w, None, ConvOptions::new([1, 1], [2, 2], [1, 1], 1));

    let sx = Tensor::<1>::from_floats(SOBEL_X, &device).reshape([1, 1, 3, 3]);
    let sy = Tensor::<1>::from_floats(SOBEL_Y, &device).reshape([1, 1, 3, 3]);
    let gx = conv2d(blurred.clone(), sx, None, ConvOptions::new([1, 1], [1, 1], [1, 1], 1));
    let gy = conv2d(blurred, sy, None, ConvOptions::new([1, 1], [1, 1], [1, 1], 1));

    let mag = (gx.clone() * gx + gy.clone() * gy).sqrt();
    mag.reshape([h as i32, w as i32])
}

/// Per-gaussian edge score for one view: project each gaussian center to a pixel
/// and sample `edge_map` there, weighted by `opacities`.
///
/// `means` is `[N, 3]` world-space, `opacities` is `[N]` (activated), `edge_map`
/// is `[H, W]`. Returns `[N]`. Gaussians behind the camera or off-screen score 0.
/// **Pinhole cameras only** — any other model returns all-zeros (documented
/// fallback limitation; the aerial pipeline renders pinhole/cube frames).
pub(crate) fn project_edge_scores(
    means: Tensor<2>,
    opacities: Tensor<1>,
    edge_map: Tensor<2>,
    camera: &Camera,
    img_size: glam::UVec2,
) -> Tensor<1> {
    let n = means.dims()[0];
    let device = means.device();
    let [height, width] = edge_map.dims();

    if n == 0 || !matches!(camera.camera_model, CameraModel::Pinhole) {
        return Tensor::zeros([n], &device);
    }

    // World -> camera (glam Affine3A: p_cam = matrix3 * p_world + translation;
    // Mat3A columns are the basis axes, so row i is (x_axis[i], y_axis[i], z_axis[i])).
    let wl = camera.world_to_local();
    let m = wl.matrix3;
    let t = wl.translation;
    let r0 = Tensor::<1>::from_floats([m.x_axis.x, m.y_axis.x, m.z_axis.x], &device).reshape([1, 3]);
    let r1 = Tensor::<1>::from_floats([m.x_axis.y, m.y_axis.y, m.z_axis.y], &device).reshape([1, 3]);
    let r2 = Tensor::<1>::from_floats([m.x_axis.z, m.y_axis.z, m.z_axis.z], &device).reshape([1, 3]);

    let px = (means.clone() * r0).sum_dim(1).add_scalar(t.x); // [N, 1]
    let py = (means.clone() * r1).sum_dim(1).add_scalar(t.y);
    let pz = (means * r2).sum_dim(1).add_scalar(t.z);

    let focal = camera.focal(img_size);
    let center = camera.center(img_size);
    let near = 0.01f32;

    let inv_z = pz.clone().clamp_min(1e-6).recip();
    let u = (px * inv_z.clone())
        .mul_scalar(focal.x)
        .add_scalar(center.x); // [N, 1]
    let v = (py * inv_z).mul_scalar(focal.y).add_scalar(center.y);

    // Valid = in front of the near plane AND inside the image rectangle.
    let in_front = pz.greater_elem(near);
    let u_ok = u
        .clone()
        .greater_equal_elem(0.0)
        .bool_and(u.clone().lower_elem(width as f32));
    let v_ok = v
        .clone()
        .greater_equal_elem(0.0)
        .bool_and(v.clone().lower_elem(height as f32));
    let valid = in_front.bool_and(u_ok).bool_and(v_ok).float(); // [N, 1]

    // Nearest-pixel gather (floor). Clamp keeps flat indices in range; invalid
    // rows are zeroed by `valid` after the gather.
    let col = u.clamp(0.0, (width - 1) as f32).floor().int(); // [N, 1] Int
    let row = v.clamp(0.0, (height - 1) as f32).floor().int();
    let flat = (row.mul_scalar(width as i32) + col).reshape([n]); // [N]

    let edge_flat = edge_map.reshape([(height * width) as i32]);
    let sampled = edge_flat.select(0, flat).reshape([n, 1]); // [N, 1]

    let score = sampled * opacities.reshape([n, 1]) * valid;
    score.reshape([n])
}

/// Divide every entry by the median of the strictly-positive entries (clamped to
/// `>= 1e-9`), after zeroing NaNs. Mirrors LFS
/// `normalize_by_positive_median_inplace` (mrnf.cpp:345); uses the upper median
/// `sorted[len/2]` to match. All-nonpositive input is zeroed.
pub(crate) fn normalize_by_positive_median(values: &mut [f32]) {
    for x in values.iter_mut() {
        if x.is_nan() {
            *x = 0.0;
        }
    }
    let mut positive: Vec<f32> = values.iter().copied().filter(|x| *x > 0.0).collect();
    if positive.is_empty() {
        for x in values.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    positive.sort_by(f32::total_cmp);
    let median = positive[positive.len() / 2].max(1e-9);
    for x in values.iter_mut() {
        *x /= median;
    }
}

/// Turn accumulated per-gaussian mean edge scores into a multiplicative growth/
/// replacement weight: `normalize_by_positive_median(scores) * weight + 1.0`
/// (LFS `edge_guidance_factor`, weight defaults to `MRNF_EDGE_SCORE_WEIGHT = 0.25`).
/// A gaussian with a median edge score gets ~`1 + weight`; edge-free gaussians
/// stay at 1.0, so the factor only ever *biases* sampling, never zeros it.
pub(crate) fn edge_guidance_factor(mut mean_scores: Vec<f32>, weight: f32) -> Vec<f32> {
    normalize_by_positive_median(&mut mean_scores);
    for v in mean_scores.iter_mut() {
        *v = *v * weight + 1.0;
    }
    mean_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn positive_median_normalizes_and_zeros_negatives() {
        // positives {1,2,3,4} -> upper median = sorted[2] = 3.
        let mut v = vec![1.0, 2.0, 3.0, 4.0, -5.0, f32::NAN];
        normalize_by_positive_median(&mut v);
        assert!((v[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
        assert!((v[3] - 4.0 / 3.0).abs() < 1e-6);
        assert!((v[4] - (-5.0 / 3.0)).abs() < 1e-6); // scaled, not clamped
        assert_eq!(v[5], 0.0); // NaN -> 0
    }

    #[test]
    fn edge_factor_is_one_plus_weight_at_median() {
        let f = edge_guidance_factor(vec![1.0, 2.0, 3.0], 0.25);
        // median of {1,2,3} = 2 -> normalized {0.5,1.0,1.5} -> *0.25 + 1.
        assert!((f[0] - 1.125).abs() < 1e-6);
        assert!((f[1] - 1.25).abs() < 1e-6);
        assert!((f[2] - 1.375).abs() < 1e-6);
    }

    /// Projection-correctness gate (design Part E.8: edge score is detached
    /// bookkeeping, so verify projection against a reference, NOT finite-diff).
    ///
    /// A 90-deg pinhole at the origin looking down +Z has focal = (dim/2) and
    /// principal point at the image center. An edge map whose value equals the
    /// flat pixel index lets us assert each gaussian samples exactly the pixel its
    /// center projects to, scaled by opacity, and that a behind-camera gaussian
    /// scores zero.
    #[tokio::test]
    async fn project_edge_scores_hits_the_projected_pixel() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (100usize, 100usize);
        let img_size = glam::uvec2(w as u32, h as u32);

        // fov 90 deg -> focal = (dim/2)/tan(45) = 50; center_uv 0.5 -> (50, 50).
        let camera = Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        );

        // A: (0,0,5)   -> u=50,  v=50  -> pixel (50,50) flat 5050
        // B: (1,0,5)   -> u=60,  v=50  -> pixel (60,50) flat 5060
        // C: (0,0,-5)  -> behind camera -> score 0
        let means = Tensor::<1>::from_floats(
            [0.0, 0.0, 5.0, 1.0, 0.0, 5.0, 0.0, 0.0, -5.0],
            &device,
        )
        .reshape([3, 3]);
        let opac = Tensor::<1>::from_floats([0.5, 0.5, 0.5], &device);

        // edge_map[r][c] = flat index (r*w + c).
        let iota: Vec<f32> = (0..(h * w)).map(|i| i as f32).collect();
        let edge_map =
            Tensor::<1>::from_data(TensorData::new(iota, [h * w]), &device).reshape([h as i32, w as i32]);

        let score = project_edge_scores(means, opac, edge_map, &camera, img_size);
        let score: Vec<f32> = score
            .into_data_async()
            .await
            .expect("readback")
            .into_vec()
            .expect("f32");

        assert!((score[0] - 5050.0 * 0.5).abs() < 1e-2, "A: {}", score[0]);
        assert!((score[1] - 5060.0 * 0.5).abs() < 1e-2, "B: {}", score[1]);
        assert_eq!(score[2], 0.0, "C behind camera must score 0");
    }
}
