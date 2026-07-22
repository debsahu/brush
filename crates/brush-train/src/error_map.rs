//! Error-map growth signal (MRNF `use_error_map` port).
//!
//! LFS's `use_error_map` densification swaps the growth criterion from the
//! screen-space position-gradient norm to an *error-weighted* signal: per
//! gaussian `g`, per view `v`,
//!
//! ```text
//! s_g^v = Σ_{p in footprint(g,v)} T_g(p) · α_g(p) · ê_v(p)
//! ê_v(p) = e_v(p) / mean_p(e_v)                       (MRNF map-mean → 1.0)
//! e_v(p) = max(0, 1 − (1/C) Σ_c SSIM_c(p)) · mask(p)  (clean nonneg D-SSIM)
//! ```
//!
//! and the window signal is `S_g = max_v s_g^v` (window-MAX). Growth admits
//! `S_g > τ_err (=0.003) AND vis_count_g > 0`. See LFS
//! `trainer.cpp:3480-3567` (error map, incl. `tile_error_map.div_(map_mean)` at
//! 3563-3566 — LFS DOES mean-normalize `ê`, verified), `ssim.cu:2319-2336`
//! (`ssim_to_error_map`), `kernels_backward.cuh:559-565` (per-pixel
//! `Σ T·α·error` into row 1, `Σ T·α` into row 0 at 563), `mrnf.cpp:600-616,726-736`
//! (window-MAX of row 1 + threshold; row 0 used only for the `vis>0` gate).
//!
//! # Defect-2 fix (2026-07-22): coverage normalization, `τ_err = 1.0`
//!
//! LFS thresholds the RAW pixel-sum `s_g = Σ_p T·α·ê` at `τ_err = 0.003`. But
//! 0.003 is the scale of the GRADIENT-mode row 1 — a per-gaussian per-view
//! SCALAR, the mean2d gradient norm (`kernels_backward.cuh:335`) — not the
//! pixel-SUMMED error (`:564`). On a pixel-sum, `s_g` scales with the gaussian's
//! footprint pixel-count; at the port's 8K-derived cube render size a background
//! gaussian's footprint is 10^5–10^6 px, so `s_g` reached ~1.16e6 and 0.003
//! admitted 99.99% of gaussians (a no-op floor — in LFS the real pressure there
//! is the weighted sample at `mrnf.cpp:790`, not the threshold). To make the
//! THRESHOLD select (the port's design goal), the port divides `s_g` by the
//! coverage sum `Σ_p T·α` — LFS's own row 0, `densification_weight`
//! (`kernels_backward.cuh:563` / gsplat `RasterizeToPixelsFromWorld3DGSBwd.cu:352`)
//! — yielding the coverage-weighted MEAN error per gaussian, footprint- and
//! resolution-INVARIANT (in [`crate::edge::project_coverage_weighted_mean`],
//! both rows from one `feat_dim=2` backward). That mean is then per-view
//! POSITIVE-MEDIAN normalized in [`crate::train`] (median → 1.0, mirroring the
//! edge path) so the natural anchor is `τ_err = 1.0` — worse than the per-view
//! median. (A scene-mean anchor was tried and rejected: it explodes on a
//! near-converged view and poisons the window-MAX.)
//!
//! # Implementation path (Design 1, fallback path A) — and why
//!
//! The design spec's PRIMARY path (B) accumulates `Σ_p T·α·ê` as an `[N]`
//! side-output written by atomic-add *inside the wgpu RGB rasterize backward*.
//! That is not reachable in Brush without a genuine architectural
//! contradiction: the RGB backward runs inside burn's autodiff `.backward()`
//! via a `ForwardRasterBackward` node captured at FORWARD time, but the error
//! map `ê` depends on the SSIM of the rendered image, which does not exist
//! until AFTER the forward pass. burn's autodiff exposes no API to inject a
//! post-forward tensor into an already-captured backward node (the
//! `refine_weight_holder` leaf is `require_grad()`-ed at forward time, before
//! the render output — hence before `ê` — is available). The spec anticipates
//! exactly this and sanctions the fallback: "if the atomic-backward path (B)
//! hits an unforeseen Metal wall, we drop to Design 0 wholesale … nothing is
//! lost." Design 0 / path A is what the fork's edge guidance already does.
//!
//! So the per-gaussian `s_g^v = Σ_p T·α·ê(p)` is computed by the SAME proven
//! `feat_dim=1` feature backward the edge path uses
//! ([`crate::edge::project_edge_scores`]): `∂(Σ_p ê(p)·feat0(p))/∂feat_g =
//! Σ_p ê(p)·T_g(p)·α_g(p)`, term-for-term equal to LFS's atomic accumulation,
//! at Brush's own render geometry. Only the input map changes: a mean-
//! normalized D-SSIM error map (this module) instead of a Canny edge map.
//!
//! The D-SSIM error map itself is computed here in pure burn ops (separable
//! 11-tap Gaussian, σ=1.5, standard C1/C2), mirroring how
//! [`crate::edge::canny_edge_map`] computes the edge map with `conv2d` rather
//! than editing the fragile `brush-loss` SSIM `CubeCL` kernel (which also carries
//! the training backward's saved-partials machinery — a bit-identical-when-off
//! risk we avoid entirely by decoupling). Same definition as LFS
//! `ssim_to_error_map`, testable in isolation.

use burn::tensor::{Tensor, module::conv2d, ops::ConvOptions, s};

// SSIM constants, identical to `brush-loss` (`lib.rs:120-121`) and the standard
// (0.01·L)², (0.03·L)² on L = 1.0.
const C1: f32 = 0.01 * 0.01;
const C2: f32 = 0.03 * 0.03;

/// 11-tap Gaussian weights at σ = 1.5, normalized to sum 1. Byte-identical to
/// `brush-loss::kernels::gauss_taps` (`lib.rs:84-99`), so the standalone error
/// map matches the loss's SSIM window.
fn gauss_taps() -> [f32; 11] {
    let sigma = 1.5_f32;
    let mut w = [0.0_f32; 11];
    let mut sum = 0.0_f32;
    for (i, wi) in w.iter_mut().enumerate() {
        let x = i as f32 - 5.0;
        *wi = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += *wi;
    }
    for wi in &mut w {
        *wi /= sum;
    }
    w
}

/// Separable 11×11 Gaussian blur of a `[N, 1, H, W]` batch (padding 5 = SAME).
fn gaussian_blur(x: Tensor<4>, taps_h: &Tensor<4>, taps_v: &Tensor<4>) -> Tensor<4> {
    // Horizontal (1×11) then vertical (11×1); depthwise over the batch via
    // groups = 1 with a single-channel kernel applied to a C = 1 batch.
    let x = conv2d(
        x,
        taps_h.clone(),
        None,
        ConvOptions::new([1, 1], [0, 5], [1, 1], 1),
    );
    conv2d(
        x,
        taps_v.clone(),
        None,
        ConvOptions::new([1, 1], [5, 0], [1, 1], 1),
    )
}

/// Clean, nonnegative D-SSIM error map of a predicted vs. GT `[H, W, 3]` image
/// (values ~[0, 1]): `e(p) = max(0, 1 − (1/3) Σ_c SSIM_c(p))`, exactly LFS
/// `ssim_to_error_map` (channel-MEAN of SSIM, then `1 − ·`, clamped ≥ 0). Uses
/// the same 11-tap σ=1.5 Gaussian window and C1/C2 as the training loss's SSIM.
/// Runs on whichever (inner) backend the inputs live on. NOT masked and NOT
/// mean-normalized here — the caller applies the optional mask and the
/// map-mean normalize (see [`mean_normalize`]).
pub(crate) fn ssim_error_map(pred: Tensor<3>, gt: Tensor<3>) -> Tensor<2> {
    let [h, w, _c] = pred.dims();
    let device = pred.device();

    let taps = gauss_taps();
    // Horizontal kernel [1,1,1,11] and vertical [1,1,11,1].
    let taps_h = Tensor::<1>::from_floats(taps, &device).reshape([1, 1, 1, 11]);
    let taps_v = Tensor::<1>::from_floats(taps, &device).reshape([1, 1, 11, 1]);

    // [H,W,3] -> [3,1,H,W] (one channel per batch row).
    let to_batch = |t: Tensor<3>| t.permute([2, 0, 1]).reshape([3, 1, h as i32, w as i32]);
    let p = to_batch(pred);
    let g = to_batch(gt);
    let pp = p.clone() * p.clone();
    let gg = g.clone() * g.clone();
    let pg = p.clone() * g.clone();

    // Blur all five quantities in one stacked pass: [15,1,H,W].
    let stacked = Tensor::cat(vec![p, g, pp, gg, pg], 0);
    let blurred = gaussian_blur(stacked, &taps_h, &taps_v);

    let mu1 = blurred.clone().slice(s![0..3, .., .., ..]);
    let mu2 = blurred.clone().slice(s![3..6, .., .., ..]);
    let b_pp = blurred.clone().slice(s![6..9, .., .., ..]);
    let b_gg = blurred.clone().slice(s![9..12, .., .., ..]);
    let b_pg = blurred.slice(s![12..15, .., .., ..]);

    let mu1_sq = mu1.clone() * mu1.clone();
    let mu2_sq = mu2.clone() * mu2.clone();
    let mu1_mu2 = mu1 * mu2;
    // Variances clamped ≥ 0 exactly as the loss kernel (`F::max(zero, …)`).
    let sigma1_sq = (b_pp - mu1_sq.clone()).clamp_min(0.0);
    let sigma2_sq = (b_gg - mu2_sq.clone()).clamp_min(0.0);
    let sigma12 = b_pg - mu1_mu2.clone();

    let a = (mu1_sq + mu2_sq).add_scalar(C1);
    let b = (sigma1_sq + sigma2_sq).add_scalar(C2);
    let c_top = mu1_mu2.mul_scalar(2.0).add_scalar(C1);
    let d_top = sigma12.mul_scalar(2.0).add_scalar(C2);

    // SSIM per channel, clamped to [-1, 1] like the loss kernel.
    let ssim = ((c_top * d_top) / (a * b)).clamp(-1.0, 1.0);
    // Channel-MEAN over the 3 batch rows -> [H, W]; then max(0, 1 - meanSSIM).
    let mean_ssim = ssim
        .reshape([3, h as i32, w as i32])
        .mean_dim(0)
        .reshape([h as i32, w as i32]);
    mean_ssim.neg().add_scalar(1.0).clamp_min(0.0)
}

/// MRNF map-mean normalization: divide by the (clamp-guarded) mean so the map's
/// mean becomes ~1.0, anchoring the per-gaussian score onto LFS's native scale
/// (so `τ_err` = 0.003 transfers verbatim). Mirrors `trainer.cpp:3561-3566`
/// (`tile_error_map.div_(map_mean)`); `clamp_min(1e-8)` guards the near-
/// converged all-zero-error map (risk §7.5).
pub(crate) fn mean_normalize(e: Tensor<2>) -> Tensor<2> {
    let mean = e.clone().mean().clamp_min(1e-8);
    e / mean.reshape([1, 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    async fn read2(t: Tensor<2>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("readback")
            .into_vec()
            .expect("f32")
    }

    fn const_img(h: usize, w: usize, v: f32, device: &burn::tensor::Device) -> Tensor<3> {
        Tensor::<3>::ones([h, w, 3], device).mul_scalar(v)
    }

    /// T4 (nonnegativity / perfect-match): identical pred == gt ⇒ SSIM = 1 ⇒
    /// error 0 everywhere, and NEVER negative. Guards the coverage-coupled
    /// `−w` offset leak that raw `loss_map` reuse would introduce.
    #[tokio::test]
    async fn ssim_error_zero_and_nonneg_for_identical() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (h, w) = (32usize, 32usize);
        // A textured image (not flat) so SSIM is exercised on real variance.
        let iota: Vec<f32> = (0..(h * w * 3)).map(|i| (i % 7) as f32 / 7.0).collect();
        let img = Tensor::<1>::from_data(TensorData::new(iota, [h * w * 3]), &device)
            .reshape([h as i32, w as i32, 3]);
        let e = read2(ssim_error_map(img.clone(), img)).await;
        assert!(e.iter().all(|v| v.is_finite()), "error must be finite");
        assert!(e.iter().all(|v| *v >= 0.0), "error must be nonnegative");
        // Interior (away from the zero-padded border) must be ~0 for identical
        // images. Check the central pixel.
        let center = e[(h / 2) * w + w / 2];
        assert!(center < 1e-3, "identical-image error must be ~0: {center}");
    }

    /// A mismatched pred vs. gt yields strictly positive error somewhere.
    #[tokio::test]
    async fn ssim_error_positive_for_mismatch() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (h, w) = (32usize, 32usize);
        let pred = const_img(h, w, 0.2, &device);
        // gt with a bright central block => local structural mismatch.
        let mut data = vec![0.2f32; h * w * 3];
        for r in 8..24 {
            for c in 8..24 {
                for ch in 0..3 {
                    data[(r * w + c) * 3 + ch] = 0.9;
                }
            }
        }
        let gt = Tensor::<1>::from_data(TensorData::new(data, [h * w * 3]), &device)
            .reshape([h as i32, w as i32, 3]);
        let e = read2(ssim_error_map(pred, gt)).await;
        assert!(
            e.iter().copied().fold(0.0f32, f32::max) > 0.05,
            "mismatch must produce positive error"
        );
    }

    /// T-numeric (definitional parity): for uniform pred = a, gt = b the D-SSIM
    /// error collapses to the closed form `(a − b)^2 / (a^2 + b^2 + C1)` at any
    /// interior pixel. Both variances and the covariance vanish on a constant
    /// image, so the contrast/structure ratio `d_top / b` is `C2 / C2 = 1` and
    /// `SSIM = (2ab + C1) / (a^2 + b^2 + C1)`, hence `1 − SSIM` as above. This
    /// pins the C1 placement and the luminance/contrast/structure grouping
    /// against a first-principles value — the property tests (identical ⇒ 0,
    /// mismatch > 0, normalize ⇒ mean 1) all still pass under a swapped
    /// C1<->C2 or a regrouped term, so only a numeric anchor catches those.
    #[tokio::test]
    async fn ssim_error_matches_closed_form_uniform() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let (h, w) = (32usize, 32usize);
        for &(a, b) in &[(0.2f32, 0.9f32), (0.1f32, 0.7f32), (0.5f32, 0.5f32)] {
            let pred = const_img(h, w, a, &device);
            let gt = const_img(h, w, b, &device);
            let e = read2(ssim_error_map(pred, gt)).await;
            // Central pixel is >= HALO (5) from every border on a 32x32 image,
            // so the 11-tap Gaussian sees full support (mu = a exactly).
            let center = e[(h / 2) * w + w / 2];
            let expected = (a - b).powi(2) / (a * a + b * b + C1);
            assert!(
                (center - expected).abs() < 1e-3,
                "closed-form mismatch for a={a}, b={b}: got {center}, want {expected}"
            );
        }
    }

    /// T3 (normalization): map-mean normalize sends the mean to ~1.0 and scales
    /// linearly (a 2× input map still normalizes to mean 1.0), so a downstream
    /// `Σ T·α·ê` lands on the same scale regardless of the raw map magnitude.
    #[tokio::test]
    async fn mean_normalize_sets_mean_to_one() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let h = 8usize;
        let w = 8usize;
        // Base map with mean 2.0.
        let base = Tensor::<2>::ones([h, w], &device).mul_scalar(2.0);
        let n1 = read2(mean_normalize(base.clone())).await;
        let mean1: f32 = n1.iter().sum::<f32>() / n1.len() as f32;
        assert!(
            (mean1 - 1.0).abs() < 1e-5,
            "normalized mean must be 1.0: {mean1}"
        );

        // Scaling the input by any constant leaves the normalized map identical.
        let n2 = read2(mean_normalize(base.mul_scalar(3.7))).await;
        for (a, b) in n1.iter().zip(&n2) {
            assert!(
                (a - b).abs() < 1e-5,
                "normalize not scale-invariant: {a} vs {b}"
            );
        }
    }

    /// T3b (map-mean guard): an all-zero error map does not divide-by-zero; the
    /// `clamp_min(1e-8)` keeps the output finite (and ~0), so a fully-converged
    /// view can't poison the window with inf/NaN.
    #[tokio::test]
    async fn mean_normalize_all_zero_is_finite() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let z = Tensor::<2>::zeros([4, 4], &device);
        let n = read2(mean_normalize(z)).await;
        assert!(
            n.iter().all(|v| v.is_finite()),
            "all-zero map must stay finite"
        );
    }
}
