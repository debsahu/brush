#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::view_to_packed_data;
use brush_loss::{ImageLossConfig, image_loss_eval};
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

    let (gt_packed_data, _has_alpha) = view_to_packed_data(gt_img.clone(), alpha_mode);
    let gt_packed: Tensor<2, Int> = Tensor::from_data(gt_packed_data, device);

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
    let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
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
        let psnr_m = mse_m.recip().log() * 10.0 / std::f32::consts::LN_10;
        // SSIM is windowed, so a window straddling a mask boundary still mixes
        // masked and unmasked pixels no matter how this is normalised. Treat
        // the masked SSIM as indicative, not exact.
        let ssim_m = image_loss_eval(render_rgb.clone(), gt_packed, cfg(0.0, 1.0, true))
            .mean()
            .div_scalar(valid_frac);
        (psnr_m, ssim_m)
    };

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

impl EvalSample {
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        use image::Rgb32FImage;
        log::info!("Saving eval image to disk.");
        let img = self.rendered.clone();
        let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = img.clone().into_data_async().await?.into_vec::<f32>()?;
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
