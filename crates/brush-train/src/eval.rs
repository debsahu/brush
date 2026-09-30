#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::view_to_packed_data;
use brush_loss::{ImageLossConfig, image_loss_eval, psnr_from_mse};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, RenderAux, TextureMode, render_splats};
use burn::tensor::{Device, Int, Tensor, s};
use glam::Vec3;
use image::DynamicImage;

pub struct EvalSample {
    pub gt_img: DynamicImage,
    pub rendered: Tensor<3>,
    /// Scored over EVERY pixel, masked ones included. Under
    /// `AlphaMode::Masked` the alpha channel is a mask (sky, transients), so
    /// these count masked-out regions as reconstruction error against a black
    /// background. That makes them pessimistic and NOT comparable to the
    /// Stage 4 >= 24 dB gate — but every number recorded before 2026-08-10 is
    /// this one, so it is kept under its original name rather than silently
    /// redefined.
    pub psnr: Tensor<1>,
    pub ssim: Tensor<1>,
    /// Scored over the unmasked pixels only. Use these to gate.
    ///
    /// Equal to the unmasked pair when there is no mask.
    pub psnr_masked: Tensor<1>,
    pub ssim_masked: Tensor<1>,
    /// Mean alpha, i.e. the fraction of the frame carrying real content. 1.0
    /// when unmasked. Reported so a reader can see how far apart the two
    /// figures should be rather than guessing.
    pub valid_frac: f32,
    pub render_aux: RenderAux,
}

pub async fn eval_stats(
    splats: Splats,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &Device,
    correction: Option<&(dyn Fn(Tensor<3>) -> Tensor<3> + Sync)>,
) -> Result<EvalSample> {
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    // Render on reference black background.
    let (img, render_aux) =
        render_splats(splats, gt_cam, res, Vec3::ZERO, None, TextureMode::Float).await;
    let render_rgb = img.slice(s![.., .., 0..3]);

    // Apply the learned per-view appearance correction when scoring a
    // training view (`--train-on-eval`): without it, scores on
    // appearance-varying datasets mostly measure the splat <-> average
    // appearance offset rather than reconstruction quality.
    let render_rgb = match correction {
        Some(f) => f(render_rgb),
        None => render_rgb,
    };

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (render_rgb * 255.0).round() / 255.0;

    let EvalScores {
        psnr,
        ssim,
        psnr_masked,
        ssim_masked,
        valid_frac,
    } = score_eval(&render_rgb, &gt_img, alpha_mode, device);

    Ok(EvalSample {
        gt_img,
        psnr,
        ssim,
        psnr_masked,
        ssim_masked,
        valid_frac,
        rendered: render_rgb,
        render_aux,
    })
}

/// The five numbers on the Stage 4 eval line, computed from an already-rendered,
/// already-8-bit-rounded `[H, W, 3]` image against the ground truth.
///
/// Split out of [`eval_stats`] (which calls it) so the masking and
/// `valid_frac` normalisation can be tested on a hand-built image pair without
/// a splat render in the loop.
pub struct EvalScores {
    pub psnr: Tensor<1>,
    pub ssim: Tensor<1>,
    pub psnr_masked: Tensor<1>,
    pub ssim_masked: Tensor<1>,
    pub valid_frac: f32,
}

pub fn score_eval(
    render_rgb: &Tensor<3>,
    gt_img: &DynamicImage,
    alpha_mode: AlphaMode,
    device: &Device,
) -> EvalScores {
    let (gt_packed_data, _has_alpha) = view_to_packed_data(gt_img.clone(), alpha_mode);
    let gt_packed: Tensor<2, Int> = Tensor::from_data(gt_packed_data, device);

    let cfg = |l1, ssim, mask| ImageLossConfig {
        l1_weight: l1,
        ssim_weight: ssim,
        composite_bg: None,
        mask,
    };
    // MSE = mean(L1^2) since |a - b|^2 == (a - b)^2.
    let mse = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(1.0, 0.0, false))
        .powi_scalar(2)
        .mean();
    let psnr = psnr_from_mse(mse);
    let ssim = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(0.0, 1.0, false)).mean();

    // Masked scoring. `mask: true` multiplies each loss-map pixel by `gt.a`,
    // which zeroes the masked pixels' contribution to the NUMERATOR — but the
    // `.mean()` below still divides by every pixel. Flipping the flag alone
    // therefore understates MSE by exactly `valid_frac` and OVERSTATES PSNR by
    // `-10*log10(valid_frac)`: about +2.3 dB on a 41%-masked frame, which is
    // larger than most deltas we rank runs on. Dividing by `valid_frac`
    // restores the true mean over the pixels that actually contributed.
    //
    // `valid_frac` is mean(a), the exact normaliser for a weighted mean. Note
    // the L1 map is multiplied by `a` and THEN squared, so the MSE numerator
    // carries `a^2`; our masks are binary (0 or 255) so `a^2 == a` and the two
    // coincide. Fractional alpha would need mean(a^2) here instead.
    let valid_frac: f32 = if alpha_mode == AlphaMode::Masked && gt_img.color().has_alpha() {
        let rgba = gt_img.to_rgba8();
        let total = f64::from(rgba.width()) * f64::from(rgba.height());
        let sum: f64 = rgba.pixels().map(|p| f64::from(p.0[3]) / 255.0).sum();
        // Clamped so an entirely-masked frame yields 0 rather than a division
        // by zero that would surface as an infinite PSNR.
        ((sum / total) as f32).max(1e-6)
    } else {
        1.0
    };

    let (psnr_masked, ssim_masked) = if valid_frac >= 1.0 {
        (psnr.clone(), ssim.clone())
    } else {
        let mse_m = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(1.0, 0.0, true))
            .powi_scalar(2)
            .mean()
            .div_scalar(valid_frac);
        let psnr_m = psnr_from_mse(mse_m);
        // SSIM is windowed, so a window straddling a mask boundary still mixes
        // masked and unmasked pixels no matter how this is normalised. Treat
        // the masked SSIM as indicative, not exact.
        let ssim_m = image_loss_eval(render_rgb.clone(), gt_packed, cfg(0.0, 1.0, true))
            .mean()
            .div_scalar(valid_frac);
        (psnr_m, ssim_m)
    };

    EvalScores {
        psnr,
        ssim,
        psnr_masked,
        ssim_masked,
        valid_frac,
    }
}

impl EvalSample {
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        use image::Rgb32FImage;
        log::info!("Saving eval image to disk.");
        let img = self.rendered.clone();
        let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = img.clone().into_data_async().await?.try_into_vec::<f32>()?;
        let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
            .expect("Failed to create image from tensor")
            .into();
        let img: image::DynamicImage = img.into_rgb8().into();
        let parent = path.parent().expect("Eval must have a filename");
        tokio::fs::create_dir_all(parent).await?;
        log::info!("Saving eval view to {path:?}");
        img.save(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Stage 4 eval-line scoring on hand-built image pairs.
    //!
    //! Fixture: an 8x8 ground truth, every channel 128. The render is 128 + `A`
    //! on the unmasked (alpha 255) pixels and 128 + `B` on the masked (alpha 0)
    //! ones, with `A != B`, and only 24 of 64 pixels unmasked (coverage 0.375).
    //! The three candidate masked-PSNR rules then give three separated answers,
    //! each asserted apart below:
    //!
    //! ```text
    //!   correct   : MSE = a^2                     (a = A/255)
    //!   un-normed : MSE = 0.375 * a^2             -> +4.26 dB too high
    //!   unmasked  : MSE = (24 a^2 + 40 b^2) / 64
    //! ```
    use super::*;
    use burn::tensor::TensorData;
    use image::{Rgb, RgbImage, Rgba, RgbaImage};

    const W: u32 = 8;
    const H: u32 = 8;
    const GT: u8 = 128;
    /// Unmasked columns: `x < VALID_COLS`, i.e. 24 of 64 pixels.
    const VALID_COLS: u32 = 3;
    const COVERAGE: f32 = (VALID_COLS * H) as f32 / (W * H) as f32;

    async fn device() -> Device {
        Device::from(brush_cube::test_helpers::test_device().await)
    }

    /// Ground truth with the left `VALID_COLS` columns unmasked.
    fn gt_masked() -> DynamicImage {
        RgbaImage::from_fn(W, H, |x, _| {
            Rgba([GT, GT, GT, if x < VALID_COLS { 255 } else { 0 }])
        })
        .into()
    }

    /// Render: `GT + a` on unmasked pixels, `GT + b` on masked ones, all
    /// channels, exactly on the 8-bit grid (as `eval_stats` rounds it).
    fn render(device: &Device, a: u8, b: u8) -> Tensor<3> {
        let mut v = Vec::with_capacity((W * H * 3) as usize);
        for _y in 0..H {
            for x in 0..W {
                let d = if x < VALID_COLS { a } else { b };
                let c = f32::from(GT + d) / 255.0;
                v.extend_from_slice(&[c, c, c]);
            }
        }
        Tensor::from_data(TensorData::new(v, [H as usize, W as usize, 3]), device)
    }

    async fn scalar(t: Tensor<1>) -> f32 {
        t.into_data_async()
            .await
            .expect("readback")
            .try_to_vec::<f32>()
            .expect("f32")[0]
    }

    fn psnr_of(mse: f64) -> f32 {
        (-10.0 * mse.log10()) as f32
    }

    const TOL_DB: f32 = 1e-3;

    #[tokio::test]
    async fn masked_psnr_is_normalised_by_coverage() {
        let device = device().await;
        let (a, b) = (16u8, 48u8);
        let s = score_eval(
            &render(&device, a, b),
            &gt_masked(),
            AlphaMode::Masked,
            &device,
        );

        let (a2, b2) = (
            (f64::from(a) / 255.0).powi(2),
            (f64::from(b) / 255.0).powi(2),
        );
        let cov = f64::from(COVERAGE);
        let want_masked = psnr_of(a2);
        let want_naive = psnr_of(cov * a2);
        let want_unmasked = psnr_of(cov * a2 + (1.0 - cov) * b2);

        // The fixture must separate the correct rule from both wrong ones by
        // far more than the tolerance, or the assertions below prove nothing.
        assert!((want_naive - want_masked).abs() > 4.0);
        assert!((want_unmasked - want_masked).abs() > 4.0);

        assert!((s.valid_frac - COVERAGE).abs() < 1e-6, "{}", s.valid_frac);
        let pm = scalar(s.psnr_masked).await;
        let pu = scalar(s.psnr).await;
        assert!(
            (pm - want_masked).abs() < TOL_DB,
            "masked PSNR {pm}, want {want_masked} (un-normalised rule gives {want_naive})"
        );
        assert!(
            (pu - want_unmasked).abs() < TOL_DB,
            "unmasked PSNR {pu}, want {want_unmasked}"
        );
    }

    #[tokio::test]
    async fn error_only_in_masked_pixels_leaves_masked_psnr_at_the_cap() {
        let device = device().await;
        let b = 40u8;
        let s = score_eval(
            &render(&device, 0, b),
            &gt_masked(),
            AlphaMode::Masked,
            &device,
        );
        let pm = scalar(s.psnr_masked).await;
        let pu = scalar(s.psnr).await;
        // `psnr_from_mse` clamps MSE at 1e-10, i.e. a 100 dB ceiling.
        assert!(
            (pm - 100.0).abs() < 1e-2,
            "zero error on every unmasked pixel must hit the PSNR cap, got {pm}"
        );
        let cov = f64::from(COVERAGE);
        let want_unmasked = psnr_of((1.0 - cov) * (f64::from(b) / 255.0).powi(2));
        assert!(
            (pu - want_unmasked).abs() < TOL_DB,
            "unmasked PSNR must count the masked-out error: {pu} vs {want_unmasked}"
        );
        assert!(pu < 30.0, "unmasked PSNR must drop, got {pu}");
    }

    #[tokio::test]
    async fn full_coverage_scores_masked_equal_to_unmasked() {
        let device = device().await;
        let (a, b) = (16u8, 48u8);
        let r = render(&device, a, b);

        // Fully-opaque RGBA under Masked, and plain RGB: both have coverage 1.
        let opaque: DynamicImage = RgbaImage::from_pixel(W, H, Rgba([GT, GT, GT, 255])).into();
        let rgb: DynamicImage = RgbImage::from_pixel(W, H, Rgb([GT, GT, GT])).into();
        let cov = f64::from(COVERAGE);
        let want = psnr_of(
            cov * (f64::from(a) / 255.0).powi(2) + (1.0 - cov) * (f64::from(b) / 255.0).powi(2),
        );

        for (name, gt) in [("opaque rgba", opaque), ("rgb", rgb)] {
            let s = score_eval(&r, &gt, AlphaMode::Masked, &device);
            assert_eq!(s.valid_frac, 1.0, "{name}");
            let pm = scalar(s.psnr_masked).await;
            let pu = scalar(s.psnr).await;
            let sm = scalar(s.ssim_masked).await;
            let su = scalar(s.ssim).await;
            assert!((pu - want).abs() < TOL_DB, "{name}: {pu} vs {want}");
            assert!(
                (pm - pu).abs() < 1e-6,
                "{name}: masked {pm} != unmasked {pu}"
            );
            assert!((sm - su).abs() < 1e-6, "{name}: ssim {sm} != {su}");
        }
    }

    /// Coverage is only a mask under `AlphaMode::Masked`; any other mode
    /// reports 1.0 and scores masked == unmasked.
    #[tokio::test]
    async fn coverage_is_one_outside_masked_mode() {
        let device = device().await;
        let s = score_eval(
            &render(&device, 16, 48),
            &gt_masked(),
            AlphaMode::Transparent,
            &device,
        );
        assert_eq!(s.valid_frac, 1.0);
        let pm = scalar(s.psnr_masked).await;
        let pu = scalar(s.psnr).await;
        assert!((pm - pu).abs() < 1e-6, "{pm} vs {pu}");
    }
}
