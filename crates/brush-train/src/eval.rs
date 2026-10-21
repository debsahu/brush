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
    /// The rendered alpha channel (`[H, W]`, 0 = nothing drawn, 1 = opaque).
    /// Not touched by the appearance correction or the 8-bit rounding that the
    /// RGB gets. Only read back when a per-view record is requested.
    pub rendered_alpha: Tensor<2>,
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
    let render_rgb = img.clone().slice(s![.., .., 0..3]);
    let [rh, rw, _] = img.dims();
    let rendered_alpha: Tensor<2> = img.slice(s![.., .., 3..4]).reshape([rh, rw]);

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
        rendered_alpha,
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

/// Which pixels of `gt` count as "kept" by the masked scores: every pixel,
/// unless the eval is masked AND the image carries alpha (the exact condition
/// under which [`score_eval`] computes a coverage below 1), in which case a
/// pixel is kept when its alpha byte is >= 128.
///
/// The threshold is a choice, not a fact about the loss: the loss weights each
/// pixel by its fractional alpha, and a resize leaves fractional alpha on mask
/// edges. 128 is "more kept than dropped".
fn kept_pixels(gt: &DynamicImage, alpha_mode: AlphaMode) -> Option<Vec<bool>> {
    (alpha_mode == AlphaMode::Masked && gt.color().has_alpha())
        .then(|| gt.to_rgba8().pixels().map(|p| p.0[3] >= 128).collect())
}

/// `(mean, fraction below 0.5)` of the RENDERED alpha over the KEPT pixels
/// only (see [`kept_pixels`]). Pixels the mask drops are ignored: a hole in
/// the render inside the mask is what this exists to expose, and an empty
/// patch of sky that the mask already excludes is not one. NaN for both when
/// no pixel is kept (a fully-masked view), which serialises as `null` rather
/// than as a plausible-looking 0.
pub fn rendered_alpha_stats(
    rendered_alpha: &[f32],
    gt: &DynamicImage,
    alpha_mode: AlphaMode,
) -> (f32, f32) {
    assert_eq!(
        rendered_alpha.len(),
        (gt.width() * gt.height()) as usize,
        "rendered alpha and ground truth must have the same pixel count"
    );
    let kept = kept_pixels(gt, alpha_mode);
    let (mut sum, mut holes, mut n) = (0.0_f64, 0_u64, 0_u64);
    for (i, &a) in rendered_alpha.iter().enumerate() {
        if kept.as_ref().is_some_and(|k| !k[i]) {
            continue;
        }
        sum += f64::from(a);
        holes += u64::from(a < 0.5);
        n += 1;
    }
    if n == 0 {
        return (f32::NAN, f32::NAN);
    }
    ((sum / n as f64) as f32, (holes as f64 / n as f64) as f32)
}

/// One eval view's numbers, exactly the per-view quantities whose means the
/// `eval iter N: ...` line prints (see [`eval_summary_line`]), plus the
/// render-hole statistics that the line does not carry.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalViewRecord {
    pub iter: u32,
    /// The view's image path as the dataset loader recorded it.
    pub image: String,
    pub psnr_masked: f32,
    pub ssim_masked: f32,
    /// The legacy all-pixel score (`unmasked (legacy)` on the line).
    pub psnr_unmasked: f32,
    pub ssim_unmasked: f32,
    /// `valid_frac`: mean ground-truth alpha, 1.0 when unmasked.
    pub coverage: f32,
    /// Mean rendered alpha over kept pixels.
    pub alpha_mean: f32,
    /// Fraction of kept pixels whose rendered alpha is < 0.5.
    pub alpha_lt_half: f32,
}

/// The `eval iter N: ...` log line, formed from the per-view records: each
/// figure is the plain f32 mean of that field over the views (so a mean of the
/// per-view JSONL reproduces it), NOT the PSNR of a pooled MSE.
pub fn eval_summary_line(iter: u32, records: &[EvalViewRecord]) -> String {
    let n = records.len() as f32;
    let mean = |f: fn(&EvalViewRecord) -> f32| records.iter().map(f).sum::<f32>() / n;
    let (psnr_masked, ssim_masked) = (mean(|r| r.psnr_masked), mean(|r| r.ssim_masked));
    let (psnr, ssim) = (mean(|r| r.psnr_unmasked), mean(|r| r.ssim_unmasked));
    let coverage = mean(|r| r.coverage);
    format!(
        "eval iter {iter}: masked PSNR {psnr_masked:.3} SSIM {ssim_masked:.4}          | unmasked (legacy) PSNR {psnr:.3} SSIM {ssim:.4}          | mean coverage {:.1}%",
        coverage * 100.0
    )
}

impl EvalSample {
    /// Read this view's scores and rendered alpha back to the host and build
    /// its [`EvalViewRecord`]. `alpha_mode` is the view's own, as for
    /// [`eval_stats`].
    pub async fn view_record(
        &self,
        iter: u32,
        image: &str,
        alpha_mode: AlphaMode,
    ) -> Result<EvalViewRecord> {
        let alpha = self
            .rendered_alpha
            .clone()
            .into_data_async()
            .await?
            .try_into_vec::<f32>()?;
        let (alpha_mean, alpha_lt_half) = rendered_alpha_stats(&alpha, &self.gt_img, alpha_mode);
        Ok(EvalViewRecord {
            iter,
            image: image.to_owned(),
            psnr_masked: self.psnr_masked.clone().into_scalar_async::<f32>().await?,
            ssim_masked: self.ssim_masked.clone().into_scalar_async::<f32>().await?,
            psnr_unmasked: self.psnr.clone().into_scalar_async::<f32>().await?,
            ssim_unmasked: self.ssim.clone().into_scalar_async::<f32>().await?,
            coverage: self.valid_frac,
            alpha_mean,
            alpha_lt_half,
        })
    }
}

/// The ground truth exactly as the loss sees it, as two 8-bit images: the
/// colour (premultiplied by alpha under `Transparent`, as `view_to_packed_data`
/// does, untouched under `Masked`) and the mask weight (the alpha channel under
/// `Masked`+alpha, all 255 otherwise). Saved beside the renders so an
/// independent scorer needs neither the dataset nor Brush's resize filter.
pub fn gt_rgb_and_mask(
    gt: &DynamicImage,
    alpha_mode: AlphaMode,
) -> (image::RgbImage, image::GrayImage) {
    let (packed, _) = view_to_packed_data(gt.clone(), alpha_mode);
    let packed = packed.try_into_vec::<i32>().expect("packed gt is i32");
    let masked = alpha_mode == AlphaMode::Masked && gt.color().has_alpha();
    let (w, h) = (gt.width(), gt.height());
    let mut rgb = Vec::with_capacity(packed.len() * 3);
    let mut mask = Vec::with_capacity(packed.len());
    for p in packed {
        let [r, g, b, a] = p.to_le_bytes();
        rgb.extend_from_slice(&[r, g, b]);
        mask.push(if masked { a } else { 255 });
    }
    (
        image::RgbImage::from_raw(w, h, rgb).expect("rgb buffer size"),
        image::GrayImage::from_raw(w, h, mask).expect("mask buffer size"),
    )
}

/// The 8-bit RGB image that [`EvalSample::save_to_disk`] writes for a render.
pub async fn rendered_to_rgb8(rendered: &Tensor<3>) -> anyhow::Result<image::RgbImage> {
    use image::Rgb32FImage;
    let [h, w, _] = [rendered.dims()[0], rendered.dims()[1], rendered.dims()[2]];
    let data = rendered
        .clone()
        .into_data_async()
        .await?
        .try_into_vec::<f32>()?;
    let img: DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
        .expect("Failed to create image from tensor")
        .into();
    Ok(img.into_rgb8())
}

impl EvalSample {
    /// Write the render as an 8-bit RGB PNG. The render is on the 8-bit grid
    /// already (see [`eval_stats`]), so this is lossless with respect to the
    /// tensor that was scored.
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        log::info!("Saving eval image to disk.");
        let img: image::DynamicImage = rendered_to_rgb8(&self.rendered).await?.into();
        let parent = path.parent().expect("Eval must have a filename");
        tokio::fs::create_dir_all(parent).await?;
        log::info!("Saving eval view to {path:?}");
        img.save(path)?;
        Ok(())
    }

    /// Write the ground truth as the loss saw it ([`gt_rgb_and_mask`]): an
    /// 8-bit RGB PNG at `gt_path` and the 8-bit mask weight at `mask_path`.
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_gt_to_disk(
        &self,
        gt_path: &Path,
        mask_path: &Path,
        alpha_mode: AlphaMode,
    ) -> anyhow::Result<()> {
        let (rgb, mask) = gt_rgb_and_mask(&self.gt_img, alpha_mode);
        for path in [gt_path, mask_path] {
            let parent = path.parent().expect("Eval must have a filename");
            tokio::fs::create_dir_all(parent).await?;
        }
        rgb.save(gt_path)?;
        mask.save(mask_path)?;
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

    // ---- per-view eval record -------------------------------------------

    /// A `RenderAux` of zeros: `view_record` never reads it, but an
    /// `EvalSample` cannot be built without one.
    fn dummy_aux(device: &Device) -> RenderAux {
        RenderAux {
            num_visible: 0,
            num_intersections: 0,
            visible: Tensor::zeros([1], device),
            max_radius: Tensor::zeros([1], device),
            opacities: Tensor::zeros([1], device),
            tile_offsets: Tensor::zeros([1, 1, 2], device),
            img_size: glam::uvec2(W, H),
        }
    }

    /// Rendered alpha on the 8x8 grid. Kept columns (`x < 3`, 24 pixels) hold
    /// 1.0, 0.5 and (0.25 on even rows, 1.0 on odd rows); the masked columns
    /// hold 0.0, i.e. "holes" that a mean over ALL pixels would count.
    ///
    /// ```text
    ///   kept : sum = 8*1.5 + 4*0.25 + 4*1.0 = 17      mean 17/24      holes 4/24
    ///   all  : mean 17/64 = 0.2656                    holes (4+40)/64 = 0.6875
    /// ```
    fn alpha_fixture() -> Vec<f32> {
        let mut v = Vec::new();
        for y in 0..H {
            for x in 0..W {
                v.push(match x {
                    0 => 1.0,
                    1 => 0.5,
                    2 if y % 2 == 0 => 0.25,
                    2 => 1.0,
                    _ => 0.0,
                });
            }
        }
        v
    }

    #[test]
    fn rendered_alpha_stats_count_kept_pixels_only() {
        let (mean, holes) = rendered_alpha_stats(&alpha_fixture(), &gt_masked(), AlphaMode::Masked);
        assert!((mean - 17.0 / 24.0).abs() < 1e-6, "mean {mean}");
        // 0.5 is NOT a hole (strictly less than), 0.25 is.
        assert!((holes - 4.0 / 24.0).abs() < 1e-6, "holes {holes}");

        // The same render read without a mask counts the masked columns, and
        // the answers differ by far more than the tolerance above.
        let (all_mean, all_holes) =
            rendered_alpha_stats(&alpha_fixture(), &gt_masked(), AlphaMode::Transparent);
        assert!((all_mean - 17.0 / 64.0).abs() < 1e-6, "{all_mean}");
        assert!((all_holes - 44.0 / 64.0).abs() < 1e-6, "{all_holes}");
    }

    #[test]
    fn rendered_alpha_stats_threshold_and_empty_cases() {
        // alpha 128 is kept, 127 is not: a 2x1 image, rendered alpha 0.1 / 0.9.
        let gt: DynamicImage =
            RgbaImage::from_fn(2, 1, |x, _| Rgba([0, 0, 0, if x == 0 { 128 } else { 127 }])).into();
        let (mean, holes) = rendered_alpha_stats(&[0.1, 0.9], &gt, AlphaMode::Masked);
        assert!(
            (mean - 0.1).abs() < 1e-6,
            "only the alpha-128 pixel is kept: {mean}"
        );
        assert_eq!(holes, 1.0);

        // Nothing kept -> undefined, not a plausible 0.
        let none: DynamicImage = RgbaImage::from_pixel(2, 1, Rgba([0, 0, 0, 0])).into();
        let (mean, holes) = rendered_alpha_stats(&[1.0, 1.0], &none, AlphaMode::Masked);
        assert!(mean.is_nan() && holes.is_nan());
    }

    /// The whole record, through `EvalSample::view_record`, against values
    /// derived by hand from the fixture rather than from the code under test.
    #[tokio::test]
    async fn view_record_fields_match_hand_derivation() {
        let device = device().await;
        let (a, b) = (16u8, 48u8);
        let rendered = render(&device, a, b);
        let gt = gt_masked();
        let s = score_eval(&rendered, &gt, AlphaMode::Masked, &device);
        let sample = EvalSample {
            gt_img: gt,
            rendered,
            psnr: s.psnr,
            ssim: s.ssim,
            psnr_masked: s.psnr_masked,
            ssim_masked: s.ssim_masked,
            valid_frac: s.valid_frac,
            rendered_alpha: Tensor::from_data(
                TensorData::new(alpha_fixture(), [H as usize, W as usize]),
                &device,
            ),
            render_aux: dummy_aux(&device),
        };

        let r = sample
            .view_record(7500, "images/frame_0003_left.png", AlphaMode::Masked)
            .await
            .expect("record");

        let (a2, b2) = (
            (f64::from(a) / 255.0).powi(2),
            (f64::from(b) / 255.0).powi(2),
        );
        let cov = f64::from(COVERAGE);
        assert_eq!(r.iter, 7500);
        assert_eq!(r.image, "images/frame_0003_left.png");
        assert!(
            (r.psnr_masked - psnr_of(a2)).abs() < TOL_DB,
            "{}",
            r.psnr_masked
        );
        assert!(
            (r.psnr_unmasked - psnr_of(cov * a2 + (1.0 - cov) * b2)).abs() < TOL_DB,
            "{}",
            r.psnr_unmasked
        );
        // Separation: the two PSNRs and a coverage-free masked PSNR are far apart.
        assert!((r.psnr_masked - r.psnr_unmasked).abs() > 4.0);
        assert!((r.coverage - COVERAGE).abs() < 1e-6, "{}", r.coverage);
        assert!(
            (r.alpha_mean - 17.0 / 24.0).abs() < 1e-6,
            "{}",
            r.alpha_mean
        );
        assert!(
            (r.alpha_lt_half - 4.0 / 24.0).abs() < 1e-6,
            "{}",
            r.alpha_lt_half
        );
        assert!(r.ssim_masked.is_finite() && r.ssim_unmasked.is_finite());
    }

    fn record(psnr_masked: f32, psnr_unmasked: f32, coverage: f32) -> EvalViewRecord {
        EvalViewRecord {
            iter: 1,
            image: "x.png".to_owned(),
            psnr_masked,
            ssim_masked: 0.5,
            psnr_unmasked,
            ssim_unmasked: 0.25,
            coverage,
            alpha_mean: 1.0,
            alpha_lt_half: 0.0,
        }
    }

    /// The printed line is the plain mean of the per-view values. 30 and 40 dB
    /// average to 35, whereas the PSNR of the pooled MSE would be ~33.0, so the
    /// fixture separates "mean of dB" from "dB of mean".
    #[test]
    fn summary_line_is_the_mean_of_per_view_records() {
        let records = [record(30.0, 20.0, 0.25), record(40.0, 22.0, 0.75)];
        assert_eq!(
            eval_summary_line(2500, &records),
            "eval iter 2500: masked PSNR 35.000 SSIM 0.5000          | unmasked (legacy) PSNR 21.000 SSIM 0.2500          | mean coverage 50.0%"
        );
        let pooled = -10.0 * ((10f64.powf(-3.0) + 10f64.powf(-4.0)) / 2.0).log10();
        assert!(
            (pooled - 35.0).abs() > 1.5,
            "fixture must separate: {pooled}"
        );
    }

    // ---- what --eval-save-to-disk writes ---------------------------------

    #[test]
    fn gt_images_are_the_loss_view_of_the_ground_truth() {
        let rgba: DynamicImage = RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgba([200, 100, 50, 255])
            } else {
                Rgba([200, 100, 50, 128])
            }
        })
        .into();

        let (rgb, mask) = gt_rgb_and_mask(&rgba, AlphaMode::Masked);
        assert_eq!(rgb.get_pixel(1, 0).0, [200, 100, 50], "mask keeps colour");
        assert_eq!(mask.as_raw(), &[255, 128]);

        // Transparent premultiplies (128 -> (c*128+127)/255) and has no mask.
        let (rgb, mask) = gt_rgb_and_mask(&rgba, AlphaMode::Transparent);
        assert_eq!(rgb.get_pixel(0, 0).0, [200, 100, 50]);
        assert_eq!(rgb.get_pixel(1, 0).0, [100, 50, 25]);
        assert_eq!(mask.as_raw(), &[255, 255]);

        // No alpha channel at all: mask is all 255 even under Masked, and the
        // file is still written so a scorer never has to guess.
        let plain: DynamicImage = RgbImage::from_pixel(2, 1, Rgb([9, 8, 7])).into();
        let (rgb, mask) = gt_rgb_and_mask(&plain, AlphaMode::Masked);
        assert_eq!(rgb.get_pixel(1, 0).0, [9, 8, 7]);
        assert_eq!(mask.as_raw(), &[255, 255]);
    }

    /// An independent scorer, given ONLY what `--eval-save-to-disk` writes (the
    /// 8-bit render, the 8-bit GT and the 8-bit mask), recomputes the masked
    /// PSNR that `score_eval` printed. Fixture: textured GT and render, and an
    /// edge column of FRACTIONAL alpha (128), because the loss squares
    /// `a * |d|` but normalises by `mean(a)` and a scorer that gets that wrong
    /// only shows it off the binary-mask case.
    #[tokio::test]
    async fn saved_images_reproduce_the_printed_masked_psnr() {
        let device = device().await;
        let tex = |x: u32, y: u32, c: u32| ((x * 17 + y * 5 + c * 40) % 256) as u8;
        let gt: DynamicImage = RgbaImage::from_fn(W, H, |x, y| {
            Rgba([
                tex(x, y, 0),
                tex(x, y, 1),
                tex(x, y, 2),
                match x {
                    0..=2 => 255,
                    3 => 128,
                    _ => 0,
                },
            ])
        })
        .into();
        let mut v = Vec::new();
        for y in 0..H {
            for x in 0..W {
                for c in 0..3 {
                    // Small error where the mask keeps the pixel, large where it drops it.
                    let amp = if x < 4 { 3 } else { 23 };
                    let noisy =
                        i32::from(tex(x, y, c)) + ((x + 2 * y + c) % amp) as i32 - amp as i32 / 2;
                    v.push(noisy.clamp(0, 255) as f32 / 255.0);
                }
            }
        }
        let rendered: Tensor<3> =
            Tensor::from_data(TensorData::new(v, [H as usize, W as usize, 3]), &device);
        let s = score_eval(&rendered, &gt, AlphaMode::Masked, &device);
        let printed = scalar(s.psnr_masked).await;
        let printed_unmasked = scalar(s.psnr).await;

        // ---- the scorer: saved files only ----
        let render_png = rendered_to_rgb8(&rendered).await.expect("render image");
        let (gt_png, mask_png) = gt_rgb_and_mask(&gt, AlphaMode::Masked);
        let n = f64::from(W * H);
        let (mut num, mut sum_a) = (0.0_f64, 0.0_f64);
        for (x, y, m) in mask_png.enumerate_pixels() {
            let a = f64::from(m.0[0]) / 255.0;
            sum_a += a;
            for c in 0..3 {
                let d = f64::from(render_png.get_pixel(x, y).0[c]) / 255.0
                    - f64::from(gt_png.get_pixel(x, y).0[c]) / 255.0;
                num += (a * d) * (a * d);
            }
        }
        let mse = (num / (3.0 * n) / (sum_a / n)).max(1e-10);
        let recomputed = (-10.0 * mse.log10()) as f32;

        assert!(
            (recomputed - printed).abs() < 0.01,
            "scorer {recomputed} dB vs printed {printed} dB"
        );
        // The fixture must separate the right formula from the wrong ones.
        assert!((printed - printed_unmasked).abs() > 2.0);
        let naive = (-10.0 * (num / (3.0 * n)).max(1e-10).log10()) as f32;
        assert!((naive - printed).abs() > 2.0, "no-coverage rule: {naive}");
    }

    /// Through the real renderer: the channel `eval_stats` stores as
    /// `rendered_alpha` is the alpha of the SAME pixels as the RGB (a
    /// premultiplied composite over the black background), not depth or an
    /// off-by-one channel. One opaque-ish splat in front of the camera, and a
    /// mask that keeps only the middle columns.
    #[tokio::test]
    async fn eval_stats_stores_the_rendered_alpha_channel() {
        use brush_render::gaussian_splats::{SplatRenderMode, Splats};
        use brush_render::kernels::camera_model::CameraModel;

        let device = device().await;
        // Isotropic 0.5 m splat at the origin, DC colour 1.0 on every channel
        // (rgb = 0.5 + 0.2820948 * dc), logit-opacity 3.0 (sigmoid ~0.953).
        let splats = Splats::from_raw(
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5f32.ln(); 3],
            vec![1.0, 1.0, 1.0],
            vec![3.0],
            SplatRenderMode::Default,
            &device,
        );
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, -5.0),
            glam::Quat::IDENTITY,
            0.8,
            0.8,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        );
        const N: u32 = 32;
        let gt: DynamicImage = RgbaImage::from_fn(N, N, |x, _| {
            Rgba([128, 128, 128, if (13..19).contains(&x) { 255 } else { 0 }])
        })
        .into();

        let sample = eval_stats(splats, &camera, gt, AlphaMode::Masked, &device, None)
            .await
            .expect("eval");

        let alpha = sample
            .rendered_alpha
            .clone()
            .into_data_async()
            .await
            .expect("alpha readback")
            .try_into_vec::<f32>()
            .expect("f32");
        let rgb = sample
            .rendered
            .clone()
            .into_data_async()
            .await
            .expect("rgb readback")
            .try_into_vec::<f32>()
            .expect("f32");
        assert_eq!(alpha.len(), (N * N) as usize);
        let at = |x: u32, y: u32| alpha[(y * N + x) as usize];
        assert!(
            at(16, 16) > 0.9 && at(16, 16) < 1.0,
            "centre alpha {}",
            at(16, 16)
        );
        assert!(at(0, 0) < 0.05, "corner alpha {}", at(0, 0));
        // Premultiplied over black: rgb = colour * alpha. Colour is 0.5 + 0.282.
        let colour = 0.5 + 0.282_094_8;
        let centre_rgb = rgb[((16 * N + 16) * 3) as usize];
        assert!(
            (centre_rgb - colour * at(16, 16)).abs() < 0.02,
            "rgb {centre_rgb} is not colour*alpha = {}",
            colour * at(16, 16)
        );

        // Kept columns 13..19 only; computed here straight from the readback.
        let kept: Vec<f32> = (0..N)
            .flat_map(|y| (13..19).map(move |x| (x, y)))
            .map(|(x, y)| at(x, y))
            .collect();
        let want_mean = kept.iter().sum::<f32>() / kept.len() as f32;
        let all_mean = alpha.iter().sum::<f32>() / alpha.len() as f32;
        assert!(
            (want_mean - all_mean).abs() > 0.1,
            "fixture must separate kept from all"
        );
        let r = sample
            .view_record(1, "a.png", AlphaMode::Masked)
            .await
            .expect("record");
        assert!(
            (r.alpha_mean - want_mean).abs() < 1e-5,
            "{} vs {want_mean}",
            r.alpha_mean
        );
        assert!((r.coverage - 6.0 / 32.0).abs() < 1e-6);
    }
}
