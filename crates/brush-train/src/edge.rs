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
//!   3. Positive-median normalization of the per-gaussian score, accumulated
//!      across a refine window, then `factor = score·0.25 + 1.0` multiplying the
//!      growth/replacement weights (`edge_guidance_factor`, mrnf.cpp:1456).
//!
//! # Implementation: alpha-blended parity via the `feat_dim=1` feature backward
//!
//! Stage 2's per-gaussian `Σ_p T_i(p)·α_i(p)·edge(p)` is computed here by
//! *reusing the fork's `DiG` feature forward/backward at `feat_dim=1`* rather
//! than a new `CubeCL` kernel. The feature forward composites
//! `feat_img[p][0] = Σ_i T_i(p)·α_i(p)·feat_i[0]` (raw `Σ T·α`, no
//! alpha-normalization, no background — `rasterize_features.rs`), so for
//! `L = Σ_p edge(p)·feat_img[p][0]` the feature gradient is
//! `∂L/∂feat_i[0] = Σ_p edge(p)·T_i(p)·α_i(p) = score_i`, term-for-term equal to
//! LFS's `atomicAdd(accum_weights[g], (transmittance·alpha)·edge)`. There are NO
//! scaling constants to divide out (the score flows through the feature channel,
//! so no `SH_C0`, no luma weight, and none of the color-path VJP gates —
//! negative-DC zeroing / alpha-saturation skip). The blend cutoffs (max alpha
//! 0.999, min alpha 1/255, transmittance terminate 1e-4, `sigma<0` skip, pixel
//! center +0.5) are identical between the feature forward and LFS.
//!
//! Honest scope: the 2D conic (hence the exact T/α values) comes from Brush's
//! own project/EWA (cov2d dilation, mip), NOT LFS's `dilation=0.3`, so numeric
//! score *values* do not byte-match LFS — correctly, because guidance must steer
//! Brush's own densification geometry. The parity that matters (the accumulation
//! formula, blend cutoffs, and zero scaling) holds exactly. Camera coverage is
//! whatever `render_splat_features` supports (Pinhole + the three distortion
//! models), strictly wider than LFS's pinhole-only edge rasterizer.
//!
//! Documented escape hatches (redundant on M4 Max unified memory, out of scope
//! here; the first two levers if a profile flags the feature-image allocation or
//! the per-view readbacks):
//!   - a device-side `normalize_by_positive_median` via `Tensor::sort` + `+inf`
//!     padding, for a discrete GPU where host readback is a real copy;
//!   - a dedicated forward-only `rasterize_edge_score` `CubeCL` kernel (score-only
//!     scatter, no feature image, no fwd/bwd pair).

use brush_render::burn_glue::{detach_autodiff, lift_to_autodiff};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats, fold_min_scale};
use brush_render_bwd::render_splat_features;
use burn::tensor::{Tensor, module::conv2d, ops::ConvOptions, s};

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

/// Shift a `[H, W]` tensor along dim 0 (rows) by `oy ∈ {-1, 0, 1}`, with
/// clamp-to-edge borders: `out[y] = mag[clamp(y + oy, 0, H-1)]`.
fn shift_row(t: Tensor<2>, oy: i32, h: usize) -> Tensor<2> {
    match oy {
        0 => t,
        1 => {
            let body = t.clone().slice(s![1..h, ..]);
            let last = t.slice(s![h - 1..h, ..]);
            Tensor::cat(vec![body, last], 0)
        }
        -1 => {
            let first = t.clone().slice(s![0..1, ..]);
            let body = t.slice(s![0..h - 1, ..]);
            Tensor::cat(vec![first, body], 0)
        }
        _ => unreachable!("shift_row only handles -1, 0, 1"),
    }
}

/// Shift a `[H, W]` tensor along dim 1 (columns) by `ox ∈ {-1, 0, 1}`, with
/// clamp-to-edge borders: `out[.., x] = mag[.., clamp(x + ox, 0, W-1)]`.
fn shift_col(t: Tensor<2>, ox: i32, w: usize) -> Tensor<2> {
    match ox {
        0 => t,
        1 => {
            let body = t.clone().slice(s![.., 1..w]);
            let last = t.slice(s![.., w - 1..w]);
            Tensor::cat(vec![body, last], 1)
        }
        -1 => {
            let first = t.clone().slice(s![.., 0..1]);
            let body = t.slice(s![.., 0..w - 1]);
            Tensor::cat(vec![first, body], 1)
        }
        _ => unreachable!("shift_col only handles -1, 0, 1"),
    }
}

/// Single-pass directional non-maximum suppression, matching LFS
/// `fused_canny_edge_filter` (image_kernels.cu:135-145).
///
/// Quantizes the gradient direction to the nearest of the 8 neighbors
/// (`dx = clamp(round(gx/mag), -1, 1)`, likewise `dy`; `round(x) = floor(x+0.5)`;
/// a zero-magnitude pixel gets `dx = dy = 0`), then zeroes any pixel whose
/// magnitude is below either its forward `(dx, dy)` or backward `(-dx, -dy)`
/// neighbor along that direction. Output is the *continuous* suppressed
/// magnitude — no hysteresis / double threshold, exactly LFS single-pass NMS.
///
/// Implemented with 9 slice/pad shifts + masked select (Design-0's lower-risk
/// path vs. a computed-index gather). Borders use clamp-to-edge; LFS uses a
/// shared-memory halo, so the 1px frame edge differs marginally — immaterial
/// after the downstream positive-median normalization.
fn directional_nms(mag: Tensor<2>, gx: Tensor<2>, gy: Tensor<2>) -> Tensor<2> {
    let [h, w] = mag.dims();
    let device = mag.device();

    // `round(gx/mag)` clamped to {-1,0,1}. clamp_min guards mag==0: there gx==gy==0
    // (mag = hypot(gx,gy)), so 0·(1/1e-12) = 0 → round → 0 → dx=dy=0 naturally.
    let inv_mag = mag.clone().clamp_min(1e-12).recip();
    let dx = (gx * inv_mag.clone())
        .add_scalar(0.5)
        .floor()
        .clamp(-1.0, 1.0);
    let dy = (gy * inv_mag).add_scalar(0.5).floor().clamp(-1.0, 1.0);

    // fwd[p] = mag at p's forward neighbor (dx,dy); bwd[p] = mag at (-dx,-dy).
    // Exactly one (ox,oy) matches each pixel's quantized direction, so the masked
    // sum selects that single neighbor.
    let mut fwd = Tensor::<2>::zeros([h, w], &device);
    let mut bwd = Tensor::<2>::zeros([h, w], &device);
    for oy in -1..=1i32 {
        for ox in -1..=1i32 {
            let neighbor = shift_col(shift_row(mag.clone(), oy, h), ox, w);
            let mask_fwd = dx
                .clone()
                .equal_elem(ox as f32)
                .bool_and(dy.clone().equal_elem(oy as f32))
                .float();
            fwd = fwd + neighbor.clone() * mask_fwd;
            let mask_bwd = dx
                .clone()
                .equal_elem((-ox) as f32)
                .bool_and(dy.clone().equal_elem((-oy) as f32))
                .float();
            bwd = bwd + neighbor * mask_bwd;
        }
    }

    // Suppress where mag < fwd OR mag < bwd (keep = mag >= fwd AND mag >= bwd).
    let keep = mag
        .clone()
        .greater_equal(fwd)
        .bool_and(mag.clone().greater_equal(bwd))
        .float();
    mag * keep
}

/// Fused-Canny NMS edge map of a `[H, W, 3]` RGB image (values in ~[0, 1]).
///
/// Mirrors LFS's `launch_fused_canny_edge_filter_chw`: luminance conversion, 5x5
/// Gaussian blur, 3x3 gradients, magnitude, and single-pass directional NMS
/// ([`directional_nms`]). Runs on whichever backend `rgb` lives on (call with an
/// inner/detached tensor — this is non-differentiable bookkeeping).
///
/// NOTE: this does NOT positive-median normalize the edge map (LFS step (a)).
/// That normalize is a provable no-op given the per-gaussian score normalize
/// (LFS step (b)) is always applied downstream: `normalize_by_positive_median`
/// is scale-equivariant, so scaling the edge map by `1/m` scales every score by
/// `1/m`, which step (b) cancels exactly. See `train::accumulate_edge_sample`.
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
    let blurred = conv2d(
        gray,
        blur_w,
        None,
        ConvOptions::new([1, 1], [2, 2], [1, 1], 1),
    );

    let sx = Tensor::<1>::from_floats(SOBEL_X, &device).reshape([1, 1, 3, 3]);
    let sy = Tensor::<1>::from_floats(SOBEL_Y, &device).reshape([1, 1, 3, 3]);
    let gx = conv2d(
        blurred.clone(),
        sx,
        None,
        ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
    )
    .reshape([h as i32, w as i32]);
    let gy = conv2d(
        blurred,
        sy,
        None,
        ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
    )
    .reshape([h as i32, w as i32]);

    let mag = (gx.clone() * gx.clone() + gy.clone() * gy.clone()).sqrt();
    directional_nms(mag, gx, gy)
}

/// Per-gaussian edge score for one view: `score_i = Σ_p T_i(p)·α_i(p)·edge(p)`,
/// the alpha-blended rendering contribution of gaussian `i` to edge pixels.
///
/// Computed by the `feat_dim=1` feature backward (see the module docstring): a
/// unit feature per gaussian is rasterized to `[H, W, 2]` (composited feature +
/// alpha), and the gradient of `Σ_p edge(p)·feat_img[p][0]` w.r.t. the features
/// is exactly the per-gaussian score. `edge_map` is `[H, W]`, already NMS'd and
/// intentionally NOT median-normalized. Rendered at FULL `img_size` (not a
/// reduced `DiG` `feat_size`). Returns `[N]` on the inner backend. `N == 0` →
/// empty.
///
/// Camera coverage is `render_splat_features`'s: Pinhole plus the
/// `KannalaBrandt4` / `RadialTangential8` / `ThinPrismFisheye` distortion models
/// (Brush has no Equirectangular model, so 360 stays on the cube /
/// pinhole-undistort path).
pub(crate) async fn project_edge_scores(
    splats: &Splats,
    edge_map: Tensor<2>,
    camera: &Camera,
    img_size: glam::UVec2,
) -> Tensor<1> {
    let n = splats.num_splats() as usize;
    let device = splats.transforms.val().device();
    if n == 0 {
        return Tensor::zeros([0], &device);
    }

    // Fold the Mip-Splatting 3D-filter floor into geometry, exactly as the RGB /
    // DiG render paths do (train.rs feature call site). `render_splat_features`
    // detaches geometry internally (`to_inner_float`), so only `features` carries
    // the gradient we read back.
    let (transforms, raw_opac) = match &splats.min_scale {
        Some(f) => fold_min_scale(
            splats.transforms.val(),
            splats.raw_opacities.val(),
            f.clone(),
        ),
        None => (splats.transforms.val(), splats.raw_opacities.val()),
    };
    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    // feat_dim = 1, unit features marked require_grad. The feature value is
    // irrelevant (the forward is linear in features); ones is a convenient leaf.
    // `detach_autodiff` forces an inner-kind tensor so `lift_to_autodiff` re-roots
    // it as a fresh autodiff leaf regardless of the backend `splats` arrived on.
    let ones = detach_autodiff(Tensor::<2>::ones([n, 1], &device));
    let feats = lift_to_autodiff(ones).require_grad();

    let feat_img = render_splat_features(
        transforms,
        raw_opac,
        feats.clone(),
        camera,
        img_size,
        render_mode,
    )
    .await;

    let [h, w, _] = feat_img.dims();
    let feat0 = feat_img
        .slice(s![.., .., 0..1])
        .reshape([h as i32, w as i32]);

    // `edge_map` is a no-grad constant; lift it onto the same autodiff graph so
    // the elementwise product stays on one backend.
    let edge_ad = lift_to_autodiff(edge_map);
    let loss = (feat0 * edge_ad).sum();
    let grads = loss.backward();
    feats
        .grad(&grads)
        .expect("feature leaf must receive a gradient")
        .reshape([n as i32])
}

/// Per-gaussian coverage-weighted MEAN of `map` over each gaussian's footprint:
///
/// ```text
/// score_g = (Σ_p T_g(p)·α_g(p)·map(p)) / (Σ_p T_g(p)·α_g(p))
/// ```
///
/// i.e. LFS's error-weighted row (numerator, `Σ T·α·map`, fastgs
/// `kernels_backward.cuh:564`) DIVIDED by LFS's coverage/weight row (denominator,
/// `Σ T·α`, the `densification_weight` at `kernels_backward.cuh:563` /
/// gsplat `RasterizeToPixelsFromWorld3DGSBwd.cu:352`). LFS accumulates BOTH rows
/// but thresholds only the raw numerator (`mrnf.cpp:601-605,726`); see
/// [`crate::error_map`] for WHY the port divides where LFS does not (the raw sum
/// scales with footprint pixel-count and does not transfer LFS's `τ` across the
/// port's 8K-derived render resolution — the ratio is footprint- and
/// resolution-invariant, on the `map` scale).
///
/// BOTH rows come from a SINGLE `feat_dim=2` feature backward: channel 0 is
/// weighted by `map`, channel 1 by the constant `1.0`, so one forward+backward
/// yields `[Σ T·α·map, Σ T·α]` per gaussian. Returns the per-gaussian ratio
/// `[N]` on the inner backend (`0` where a gaussian has ~no coverage; `N == 0`
/// → `[0]`), finite and nonnegative. The caller applies the per-view
/// positive-median re-anchor that puts the threshold on a stable scale.
pub(crate) async fn project_coverage_weighted_mean(
    splats: &Splats,
    map: Tensor<2>,
    camera: &Camera,
    img_size: glam::UVec2,
) -> Tensor<1> {
    let n = splats.num_splats() as usize;
    let device = splats.transforms.val().device();
    if n == 0 {
        return Tensor::zeros([0], &device);
    }

    // Same Mip-Splatting 3D-filter floor fold + render mode as the RGB / DiG
    // paths and `project_edge_scores`.
    let (transforms, raw_opac) = match &splats.min_scale {
        Some(f) => fold_min_scale(
            splats.transforms.val(),
            splats.raw_opacities.val(),
            f.clone(),
        ),
        None => (splats.transforms.val(), splats.raw_opacities.val()),
    };
    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    // feat_dim = 2 unit features: channel 0 collects `Σ T·α·map`, channel 1
    // collects `Σ T·α`. `render_splat_features` supports arbitrary feat_dim (the
    // DiG DINO path renders dozens of channels), so both rows fall out of one
    // forward + one backward — half the render cost of two `feat_dim=1` passes.
    let ones = detach_autodiff(Tensor::<2>::ones([n, 2], &device));
    let feats = lift_to_autodiff(ones).require_grad();

    let feat_img = render_splat_features(
        transforms,
        raw_opac,
        feats.clone(),
        camera,
        img_size,
        render_mode,
    )
    .await;

    let [h, w, _] = feat_img.dims();
    let feat0 = feat_img
        .clone()
        .slice(s![.., .., 0..1])
        .reshape([h as i32, w as i32]);
    let feat1 = feat_img
        .slice(s![.., .., 1..2])
        .reshape([h as i32, w as i32]);

    let map_ad = lift_to_autodiff(map);
    // loss = Σ_p map(p)·feat0(p) + Σ_p 1·feat1(p); ∂/∂feat[g][0] = Σ_p map·T_g·α_g
    // (row 1), ∂/∂feat[g][1] = Σ_p T_g·α_g (row 0).
    let loss = (feat0 * map_ad).sum() + feat1.sum();
    let grads = loss.backward();
    let rows = feats
        .grad(&grads)
        .expect("feature leaf must receive a gradient"); // [N, 2]
    // A handful of degenerate/newborn gaussians (right after a split) can yield a
    // non-finite feature gradient — the RGB path tolerates this via fastgs
    // `clamp_grad`, but this isolated backward does not. Sanitize NaN→0 and tame
    // ±inf BEFORE any reduction: a single NaN in the scene-mean sum below would
    // otherwise poison EVERY gaussian's score (observed: iter 400 → all-NaN,
    // threshold count 0, growth stalled). A sanitized-to-0 gaussian falls below τ and
    // is simply not selected — the correct outcome for a degenerate splat.
    let sanitize = |t: Tensor<1>| t.clone().mask_fill(t.is_nan(), 0.0).clamp(-1e12, 1e12);
    let row1 = sanitize(rows.clone().slice(s![.., 0..1]).reshape([n as i32]));
    let row0 = sanitize(rows.slice(s![.., 1..2]).reshape([n as i32]));

    // Per-gaussian coverage-weighted mean error; guard the near-zero-coverage
    // denominator. A gaussian with ~no visible contribution has row0 ≈ row1 ≈ 0,
    // so the ratio is ~0 and the `vis_count > 0` gate excludes it regardless.
    // Final NaN→0 + bound so nothing non-finite can reach the caller (the
    // per-view positive-median re-anchor and the window-MAX accumulator both
    // assume finite input). The caller ([`crate::train::accumulate_error_sample`])
    // positive-median normalizes this per view — anchoring the threshold — since
    // the raw coverage-weighted mean of a mean-normalized `ê` is not O(1) under
    // the map's heavy right-skew + masking (see that fn's defect-2 note).
    let per_g = row1 / row0.clamp_min(1e-8);
    per_g.clone().mask_fill(per_g.is_nan(), 0.0).clamp(0.0, 1e6)
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
    for v in &mut mean_scores {
        *v = *v * weight + 1.0;
    }
    mean_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;
    use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
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

    /// Directional NMS thins a ridge to its crest along the gradient direction.
    /// Field: every row `mag = [1,2,3,2,1]`, `gx = mag` (so `dx = round(gx/mag) =
    /// 1`), `gy = 0` (`dy = 0`). By the LFS rule each pixel compares against its
    /// left/right neighbors, leaving only the crest column: `[0,0,3,0,0]`.
    #[tokio::test]
    async fn directional_nms_thins_ridge() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (h, w) = (5usize, 5usize);
        let row = [1.0f32, 2.0, 3.0, 2.0, 1.0];
        let mut data = Vec::with_capacity(h * w);
        for _ in 0..h {
            data.extend_from_slice(&row);
        }
        let mag = Tensor::<1>::from_data(TensorData::new(data.clone(), [h * w]), &device)
            .reshape([h as i32, w as i32]);
        // gx = mag (positive) -> dx = round(gx/mag) = 1 everywhere; gy = 0.
        let gx = mag.clone();
        let gy = Tensor::<2>::zeros([h, w], &device);

        let out = directional_nms(mag, gx, gy);
        let out: Vec<f32> = out
            .into_data_async()
            .await
            .expect("readback")
            .into_vec()
            .expect("f32");

        for r in 0..h {
            let base = r * w;
            assert!(
                (out[base + 2] - 3.0).abs() < 1e-5,
                "crest kept: {}",
                out[base + 2]
            );
            for &c in &[0usize, 1, 3, 4] {
                assert!(
                    out[base + c].abs() < 1e-5,
                    "off-ridge suppressed at {c}: {}",
                    out[base + c]
                );
            }
        }
    }

    // --- GPU parity helpers -------------------------------------------------

    /// A pinhole looking down +Z from the origin: focal = dim/2, principal point
    /// at the image center (fov 90 deg, center_uv 0.5).
    fn origin_pinhole() -> Camera {
        Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    /// Build a `Splats` from per-gaussian means / log-scales / raw opacities,
    /// identity rotations, degree-0 (white) SH.
    fn make_splats(
        means: &[[f32; 3]],
        log_scales: &[[f32; 3]],
        raw_opac: &[f32],
        device: &burn::tensor::Device,
    ) -> Splats {
        let n = means.len();
        let means_flat: Vec<f32> = means.iter().flatten().copied().collect();
        let scales_flat: Vec<f32> = log_scales.iter().flatten().copied().collect();
        let means_t = Tensor::<1>::from_data(TensorData::new(means_flat, [n * 3]), device)
            .reshape([n as i32, 3]);
        let scales_t = Tensor::<1>::from_data(TensorData::new(scales_flat, [n * 3]), device)
            .reshape([n as i32, 3]);
        let quats: Tensor<2> = Tensor::<1>::from_floats(glam::Quat::IDENTITY.to_array(), device)
            .unsqueeze_dim(0)
            .repeat_dim(0, n);
        let sh = Tensor::<3>::ones([n, 1, 3], device);
        let opac = Tensor::<1>::from_data(TensorData::new(raw_opac.to_vec(), [n]), device);
        Splats::from_tensor_data(means_t, quats, scales_t, sh, opac, SplatRenderMode::Default)
    }

    /// `edge_map[r][c] = r*w + c` (flat pixel index), as `[H, W]`.
    fn iota_edge_map(h: usize, w: usize, device: &burn::tensor::Device) -> Tensor<2> {
        let iota: Vec<f32> = (0..(h * w)).map(|i| i as f32).collect();
        Tensor::<1>::from_data(TensorData::new(iota, [h * w]), device).reshape([h as i32, w as i32])
    }

    /// Edge map that is 1.0 in a centered `k×k` window, 0 elsewhere.
    fn central_window_edge_map(
        h: usize,
        w: usize,
        k: usize,
        device: &burn::tensor::Device,
    ) -> Tensor<2> {
        let mut data = vec![0.0f32; h * w];
        let r0 = (h - k) / 2;
        let c0 = (w - k) / 2;
        for r in r0..r0 + k {
            for c in c0..c0 + k {
                data[r * w + c] = 1.0;
            }
        }
        Tensor::<1>::from_data(TensorData::new(data, [h * w]), device).reshape([h as i32, w as i32])
    }

    /// A very narrow-FOV pinhole (fov ≈ 2.9°) at the origin looking down +Z.
    /// The tight FOV keeps the affine EWA projection accurate and lets a
    /// moderately-scaled on-axis gaussian cover the frame with near-uniform
    /// alpha.
    fn narrow_pinhole() -> Camera {
        Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            0.05,
            0.05,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    async fn read1(t: Tensor<1>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("readback")
            .into_vec()
            .expect("f32")
    }

    /// Independent forward-kernel oracle: render this splat set with unit
    /// features (`feat_dim=1`) and return the scalar `Σ_p edge(p)·feat_img[p][0]`.
    /// For a single gaussian (or spatially disjoint gaussians), `feat_img[p][0] =
    /// Σ_i T_i·α_i` with `T=1` over each gaussian's own pixels, so this equals
    /// that gaussian's `Σ_p α(p)·edge(p)` — the reference the backward score must
    /// reproduce. Uses the forward feature kernel, a genuinely different code
    /// path from `project_edge_scores`'s backward.
    async fn edge_coverage_sum(
        splats: &Splats,
        edge: &Tensor<2>,
        camera: &Camera,
        img_size: glam::UVec2,
    ) -> f32 {
        let n = splats.num_splats() as usize;
        let device = splats.transforms.val().device();
        let ones = detach_autodiff(Tensor::<2>::ones([n, 1], &device));
        let feats = lift_to_autodiff(ones).require_grad();
        let feat_img = render_splat_features(
            splats.transforms.val(),
            splats.raw_opacities.val(),
            feats,
            camera,
            img_size,
            SplatRenderMode::Default,
        )
        .await;
        let [h, w, _] = feat_img.dims();
        let feat0 = detach_autodiff(
            feat_img
                .slice(s![.., .., 0..1])
                .reshape([h as i32, w as i32]),
        );
        let s = (feat0 * edge.clone()).sum();
        read1(s.reshape([1])).await[0]
    }

    fn approx(a: f32, b: f32, rel: f32, abs: f32) -> bool {
        (a - b).abs() <= rel * b.abs().max(1.0) + abs
    }

    /// Primary correctness gate: the feat_dim=1 backward score matches, per
    /// gaussian, the independent forward-kernel `Σ_p T·α·edge` reference. Two
    /// spatially disjoint gaussians (so each has `T=1` over its own footprint),
    /// each compared against its single-gaussian forward render. Also pins the
    /// "raw `Σ T·α`, no alpha-normalization, no background" feature-compositor
    /// semantics by asserting the feature channel equals the alpha channel under
    /// unit features (guards a future Brush bump that adds normalization/bg).
    #[tokio::test]
    async fn edge_score_matches_alpha_blend_reference() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (64usize, 64usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = origin_pinhole();

        // Two gaussians well separated on screen (u≈16 and u≈48), small footprint
        // so they don't overlap: T=1 over each.
        let means = [[-2.5f32, 0.0, 5.0], [2.5, 0.0, 5.0]];
        let log_scales = [[-1.8f32, -1.8, -1.8], [-1.8, -1.8, -1.8]];
        let raw_opac = [2.5f32, 2.5];
        let splats = make_splats(&means, &log_scales, &raw_opac, &device);
        let edge = iota_edge_map(h, w, &device);

        let score =
            read1(project_edge_scores(&splats, edge.clone(), &camera, img_size).await).await;
        assert_eq!(score.len(), 2);

        for g in 0..2 {
            let single = make_splats(
                std::slice::from_ref(&means[g]),
                std::slice::from_ref(&log_scales[g]),
                std::slice::from_ref(&raw_opac[g]),
                &device,
            );
            let reference = edge_coverage_sum(&single, &edge, &camera, img_size).await;
            assert!(
                reference > 0.0,
                "g{g} reference must be nonzero: {reference}"
            );
            assert!(
                approx(score[g], reference, 5e-3, 1e-2),
                "g{g}: backward score {} vs forward reference {}",
                score[g],
                reference
            );
        }

        // Pin: with unit features the composited feature channel is the raw
        // Σ T·α, which equals the accumulated alpha channel — no normalization,
        // no background term.
        let n = splats.num_splats() as usize;
        let ones = detach_autodiff(Tensor::<2>::ones([n, 1], &device));
        let feats = lift_to_autodiff(ones).require_grad();
        let feat_img = render_splat_features(
            splats.transforms.val(),
            splats.raw_opacities.val(),
            feats,
            &camera,
            img_size,
            SplatRenderMode::Default,
        )
        .await;
        let feat0 = detach_autodiff(feat_img.clone().slice(s![.., .., 0..1]));
        let alpha = detach_autodiff(feat_img.slice(s![.., .., 1..2]));
        let diff = (feat0 - alpha).abs().max();
        let diff = read1(diff.reshape([1])).await[0];
        assert!(
            diff < 1e-5,
            "feature channel must equal alpha channel: {diff}"
        );
    }

    /// Occlusion / transmittance (`T < 1`): a front opaque gaussian shadows a rear
    /// one at the same screen position. The front's score is unaffected by what is
    /// behind it; the rear's score is attenuated by the front's transmittance.
    #[tokio::test]
    async fn edge_score_occlusion_reduces_rear() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (64usize, 64usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = origin_pinhole();

        // Front: near, large, very opaque. Rear: same screen center, behind.
        let front = [0.0f32, 0.0, 3.0];
        let rear = [0.0f32, 0.0, 6.0];
        let front_scale = [-0.5f32, -0.5, -0.5];
        let rear_scale = [-0.5f32, -0.5, -0.5];
        let edge = iota_edge_map(h, w, &device);

        let both = make_splats(
            &[front, rear],
            &[front_scale, rear_scale],
            &[6.0, 3.0],
            &device,
        );
        let front_only = make_splats(&[front], &[front_scale], &[6.0], &device);
        let rear_only = make_splats(&[rear], &[rear_scale], &[3.0], &device);

        let s_both = read1(project_edge_scores(&both, edge.clone(), &camera, img_size).await).await;
        let s_front =
            read1(project_edge_scores(&front_only, edge.clone(), &camera, img_size).await).await;
        let s_rear =
            read1(project_edge_scores(&rear_only, edge.clone(), &camera, img_size).await).await;

        // Front unaffected by the rear behind it (T_front == 1 either way).
        assert!(
            approx(s_both[0], s_front[0], 5e-3, 1e-2),
            "front score changed by occluded rear: {} vs {}",
            s_both[0],
            s_front[0]
        );
        // Rear attenuated by the opaque front (strictly less than rear-alone).
        assert!(
            s_rear[0] > 0.0,
            "rear-alone must be positive: {}",
            s_rear[0]
        );
        assert!(
            s_both[1] < s_rear[0] * 0.99,
            "rear score not attenuated: both={} alone={}",
            s_both[1],
            s_rear[0]
        );
    }

    /// Camera coverage: a non-pinhole (KannalaBrandt4 fisheye) camera yields
    /// finite, nonzero scores — the old pinhole-only restriction is gone.
    #[tokio::test]
    async fn edge_score_nonzero_for_fisheye() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (48usize, 48usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = Camera::new(
            glam::vec3(0.0, 0.0, -3.0),
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::KannalaBrandt4(KannalaBrandt4Params::default()),
        );
        let means = [[0.0f32, 0.0, 0.0], [0.1, -0.05, 0.2]];
        let log_scales = [[-0.8f32, -0.8, -0.8], [-0.9, -0.9, -0.9]];
        let splats = make_splats(&means, &log_scales, &[3.0, 3.0], &device);
        let edge = iota_edge_map(h, w, &device);

        let score = read1(project_edge_scores(&splats, edge, &camera, img_size).await).await;
        assert!(
            score.iter().all(|s| s.is_finite()),
            "scores must be finite: {score:?}"
        );
        assert!(
            score.iter().sum::<f32>() > 0.0,
            "fisheye scores all zero: {score:?}"
        );
    }

    /// Scale-equivariance regression (guards the dropped LFS step (a)): scaling
    /// the edge map by an arbitrary constant scales every raw score by the same
    /// constant, which the per-gaussian positive-median normalize (step (b))
    /// cancels exactly. So post-step-(b) scores must be identical with and
    /// without the edge-map scale — proving the omitted edge-map normalize was a
    /// true no-op.
    #[tokio::test]
    async fn edge_score_scale_equivariant() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (64usize, 64usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = origin_pinhole();

        let means = [[-1.5f32, 0.5, 5.0], [1.5, -0.5, 5.0], [0.0, 0.0, 4.0]];
        let log_scales = [
            [-1.6f32, -1.6, -1.6],
            [-1.6, -1.6, -1.6],
            [-1.6, -1.6, -1.6],
        ];
        let splats = make_splats(&means, &log_scales, &[2.5, 2.5, 2.5], &device);
        let edge = iota_edge_map(h, w, &device);

        let s1 = read1(project_edge_scores(&splats, edge.clone(), &camera, img_size).await).await;
        let s2 =
            read1(project_edge_scores(&splats, edge.mul_scalar(0.37), &camera, img_size).await)
                .await;

        let mut n1 = s1.clone();
        let mut n2 = s2.clone();
        normalize_by_positive_median(&mut n1);
        normalize_by_positive_median(&mut n2);
        for (a, b) in n1.iter().zip(&n2) {
            assert!(
                (a - b).abs() < 1e-4,
                "scale-equivariance broken: {a} vs {b}"
            );
        }
    }

    /// Genuinely independent correctness gate on the score's ABSOLUTE
    /// magnitude, hand-derived from first principles rather than from another
    /// call into `render_splat_features` (which would only re-confirm autodiff
    /// linearity). A single on-axis isotropic gaussian with `raw_opacity = 0`
    /// has peak alpha `sigmoid(0) * filter_comp = 0.5` (filter_comp == 1.0 in
    /// Default / non-mip mode; verified in `project_forward.rs`). It is the
    /// only gaussian, so `T == 1` in front of it, and its wide projected
    /// footprint (≈130 px 2D std) makes alpha ≈ 0.5 across the tiny central
    /// `k×k` edge window (falloff over ±4 px is < 0.1 %). Hence
    /// `score = Σ_window T·α·edge = 0.5 · k²`, a literal number no kernel bug
    /// (wrong opacity activation, dropped transmittance base case, wrongly scaled
    /// Σ T·α) can satisfy silently. Loose tol absorbs sub-pixel centering and
    /// the residual window falloff.
    #[tokio::test]
    async fn edge_score_absolute_value_from_first_principles() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (32usize, 32usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = narrow_pinhole();

        // On-axis; 3σ (= 3·0.8 = 2.4) < depth (4) so it stays in front of the
        // near plane, yet the narrow FOV projects it to a footprint far wider
        // than the 8×8 window.
        let means = [[0.0f32, 0.0, 4.0]];
        let log_scales = [[-0.223_143_5f32, -0.223_143_5, -0.223_143_5]]; // ln(0.8)
        let raw_opac = [0.0f32]; // sigmoid(0) = 0.5
        let splats = make_splats(&means, &log_scales, &raw_opac, &device);

        let k = 8usize;
        let edge = central_window_edge_map(h, w, k, &device);
        let expected = 0.5 * (k * k) as f32; // 0.5 per window pixel, T == 1

        let score = read1(project_edge_scores(&splats, edge, &camera, img_size).await).await;
        assert_eq!(score.len(), 1);
        assert!(score[0].is_finite(), "score must be finite: {}", score[0]);
        assert!(
            (score[0] - expected).abs() <= 0.08 * expected,
            "score {} deviates from first-principles reference {expected}",
            score[0]
        );
    }

    /// Coverage regression (behind-camera + off-frustum culling). A gaussian
    /// behind the near plane and one laterally outside the frustum are both
    /// terminated in projection, so each must yield a FINITE score of exactly
    /// 0.0 — never a projection-singularity NaN (which `normalize_by_positive_
    /// median` would silently mask) and never a spurious nonzero (which would
    /// bias densification undetected). A third in-frame gaussian confirms the
    /// zeros are real culling, not a global render failure.
    #[tokio::test]
    async fn edge_score_zero_for_behind_and_offscreen() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (64usize, 64usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = origin_pinhole(); // at origin looking +Z

        // g0 behind the camera (z < 0); g1 far to the side (projects off-frame);
        // g2 in front, on-axis (visible control).
        let means = [[0.0f32, 0.0, -5.0], [100.0, 0.0, 5.0], [0.0, 0.0, 5.0]];
        let log_scales = [
            [-1.8f32, -1.8, -1.8],
            [-1.8, -1.8, -1.8],
            [-1.8, -1.8, -1.8],
        ];
        let splats = make_splats(&means, &log_scales, &[2.5, 2.5, 2.5], &device);
        let edge = iota_edge_map(h, w, &device);

        let score = read1(project_edge_scores(&splats, edge, &camera, img_size).await).await;
        assert_eq!(score.len(), 3);
        assert!(
            score.iter().all(|s| s.is_finite()),
            "scores must be finite: {score:?}"
        );
        assert_eq!(
            score[0], 0.0,
            "behind-camera gaussian must score exactly 0: {}",
            score[0]
        );
        assert_eq!(
            score[1], 0.0,
            "off-frustum gaussian must score exactly 0: {}",
            score[1]
        );
        assert!(
            score[2] > 0.0,
            "in-frame control must score > 0: {}",
            score[2]
        );
    }

    #[test]
    fn normalize_by_positive_median_zeros_all_nonpositive() {
        // All-zero input: no positives -> zeroed.
        let mut z = vec![0.0f32, 0.0, 0.0];
        normalize_by_positive_median(&mut z);
        assert_eq!(z, [0.0, 0.0, 0.0]);
        // All-negative input: no positives -> zeroed (not scaled).
        let mut n = vec![-1.0f32, -2.0, -3.0];
        normalize_by_positive_median(&mut n);
        assert_eq!(n, [0.0, 0.0, 0.0]);
        // edge_guidance_factor on all-zero scores is the neutral 1.0 everywhere.
        let f = edge_guidance_factor(vec![0.0, 0.0, 0.0], 0.25);
        assert_eq!(f, [1.0, 1.0, 1.0]);
    }

    /// Blank-view path (e.g. a sky-only frame with an all-zero edge map):
    /// every per-gaussian score is exactly 0.0 and finite, exercising
    /// `normalize_by_positive_median`'s empty-positive zeroing branch and
    /// `edge_guidance_factor`'s neutral 1.0 output — the factor must never go
    /// inf/NaN and poison densify for the window.
    #[tokio::test]
    async fn edge_score_zero_edge_map_is_neutral() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (48usize, 48usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = origin_pinhole();

        let means = [[-1.5f32, 0.5, 5.0], [1.5, -0.5, 5.0], [0.0, 0.0, 4.0]];
        let log_scales = [
            [-1.6f32, -1.6, -1.6],
            [-1.6, -1.6, -1.6],
            [-1.6, -1.6, -1.6],
        ];
        let splats = make_splats(&means, &log_scales, &[2.5, 2.5, 2.5], &device);
        let zero_edge = Tensor::<2>::zeros([h, w], &device);

        let score = read1(project_edge_scores(&splats, zero_edge, &camera, img_size).await).await;
        assert_eq!(score.len(), 3);
        assert!(
            score.iter().all(|s| s.is_finite()),
            "scores must be finite: {score:?}"
        );
        assert!(
            score.iter().all(|s| *s == 0.0),
            "zero edge map must give all-zero scores: {score:?}"
        );

        let factor = edge_guidance_factor(score, 0.25);
        assert!(
            factor.iter().all(|f| (f - 1.0).abs() < 1e-6),
            "neutral factor expected for a blank view: {factor:?}"
        );
    }

    /// DEFECT-1 regression: a projected per-gaussian score is rooted in an
    /// ISOLATED autodiff feature-backward graph, so it comes back autodiff-KIND.
    /// `gather_error` must DETACH it to the inner backend at the store; otherwise
    /// `error_score_max_or_zeros()` returns an autodiff tensor and multiplying it
    /// against a genuinely-inner tensor (as the growth path does,
    /// `above_threshold.float() * growth_base`) crosses backends and panics
    /// ("tensors are not on the same backend"). This test stores a real projected
    /// score and multiplies the RAW stored value against an inner tensor — the
    /// exact leak. Pre-fix (no detach in `gather_error`) it panics; post-fix it
    /// yields finite output.
    #[tokio::test]
    async fn projected_score_survives_gather_error_and_inner_multiply() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (64usize, 64usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = origin_pinhole();

        // A centered gaussian that covers the frame, so it has real error mass.
        let means = [[0.0f32, 0.0, 5.0]];
        let log_scales = [[-0.5f32, -0.5, -0.5]];
        let splats = make_splats(&means, &log_scales, &[3.0], &device);
        let n = splats.num_splats() as usize;
        // A nonuniform error map so the score is nonzero and view-dependent.
        let map = central_window_edge_map(h, w, 24, &device);

        // The score comes back autodiff-KIND (feature-backward bridge).
        let score = project_coverage_weighted_mean(&splats, map, &camera, img_size).await;

        // Store it, then pull the RAW stored score back out (NOT through the
        // median-normalize path, which would launder the backend via a host
        // rebuild) and multiply against a genuinely-inner tensor.
        let mut record = crate::stats::RefineRecord::new(n as u32, &device);
        record.gather_error(score);
        let stored = record.error_score_max_or_zeros();
        let inner_ones = Tensor::<1>::ones([n], &device);
        let weights = read1((stored * inner_ones).reshape([n as i32])).await;
        assert!(
            weights.iter().all(|v| v.is_finite()),
            "growth weights must be finite (no backend leak): {weights:?}"
        );
    }

    /// DEFECT-2 semantics: `project_coverage_weighted_mean` returns the
    /// coverage-weighted MEAN of the map over each gaussian's footprint, so a
    /// CONSTANT map of value `k` yields `k` for EVERY gaussian regardless of its
    /// footprint size (`(Σ T·α·k)/(Σ T·α) = k`) — the footprint- and
    /// resolution-invariance the raw pixel-SUM lacked (the sum would give a big
    /// gaussian a far larger score than a small one). Two disjoint gaussians of
    /// very different scale must both land on `k`.
    #[tokio::test]
    async fn coverage_weighted_mean_is_footprint_invariant() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (w, h) = (64usize, 64usize);
        let img_size = glam::uvec2(w as u32, h as u32);
        let camera = origin_pinhole();

        // Disjoint on screen (u≈16, u≈48); very different scales/footprints.
        let means = [[-2.5f32, 0.0, 5.0], [2.5, 0.0, 5.0]];
        let log_scales = [[-0.6f32, -0.6, -0.6], [-2.0, -2.0, -2.0]];
        let splats = make_splats(&means, &log_scales, &[3.0, 3.0], &device);

        // Constant map k = 2.0 everywhere.
        let k = 2.0f32;
        let map = Tensor::<2>::ones([h, w], &device).mul_scalar(k);

        let score =
            read1(project_coverage_weighted_mean(&splats, map, &camera, img_size).await).await;
        assert_eq!(score.len(), 2);
        for (g, s) in score.iter().enumerate() {
            assert!(
                (s - k).abs() < 5e-3,
                "g{g}: coverage-weighted mean of constant {k} must be {k}, got {s}"
            );
        }
    }
}
