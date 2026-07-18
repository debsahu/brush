//! Post-hoc quality evaluation for an exported Brush PLY checkpoint.
//!
//! This binary only renders held-out dataset views. It never trains, refines,
//! or mutates the checkpoint, so comparing two runs does not depend on fresh
//! optimizer state or benchmark warmup.

#[cfg(not(target_family = "wasm"))]
mod native {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use anyhow::{Context, Result, bail};
    use brush_dataset::{config::LoadDatasetConfig, load_dataset, scene::view_to_sample_image};
    use brush_render::{AlphaMode, gaussian_splats::SplatRenderMode};
    use brush_serde::load_splat_from_ply;
    use brush_train::eval::eval_stats;
    use brush_vfs::BrushVfs;
    use burn::{
        prelude::Device,
        tensor::{Tensor, TensorData},
    };
    use clap::Parser;
    use lpips::{LPIPS_MIN_IMAGE_SIZE, LpipsModel, load_vgg_lpips};
    use serde_json::json;

    #[derive(Debug, Parser)]
    #[command(about = "Evaluate an exported Brush PLY on held-out dataset views")]
    struct Args {
        /// Dataset directory containing cameras and source images.
        #[arg(long)]
        dataset: PathBuf,

        /// Exported Brush PLY checkpoint to evaluate.
        #[arg(long)]
        ply: PathBuf,

        /// Long-edge image resolution used for evaluation.
        #[arg(long, default_value_t = 1920)]
        max_resolution: u32,

        /// Request holding out every nth image after deterministic dataset
        /// ordering. A dataset-defined validation/test set takes precedence.
        #[arg(long, default_value_t = 20)]
        eval_split_every: usize,

        /// Apply one explicit alpha interpretation to every evaluation view.
        #[arg(long, value_enum)]
        alpha_mode: AlphaMode,

        /// Optional directory for rendered evaluation PNG images.
        #[arg(long)]
        save_dir: Option<PathBuf>,

        /// Also report full-frame VGG LPIPS at the evaluation resolution (lower
        /// is better). GT alpha is composited onto black before inference.
        #[arg(long, default_value_t = false)]
        lpips: bool,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct ViewMetric {
        index: usize,
        image: String,
        width: u32,
        height: u32,
        psnr: f64,
        ssim: f64,
        lpips_vgg: Option<f64>,
    }

    fn validate_args(args: &Args) -> Result<()> {
        if args.max_resolution == 0 {
            bail!("--max-resolution must be at least 1");
        }
        if args.eval_split_every < 2 {
            bail!("--eval-split-every must be at least 2");
        }
        Ok(())
    }

    fn averages(metrics: &[ViewMetric]) -> Option<(f64, f64, Option<f64>)> {
        if metrics.is_empty() {
            return None;
        }
        let (psnr, ssim) = metrics.iter().fold((0.0, 0.0), |(psnr, ssim), metric| {
            (psnr + metric.psnr, ssim + metric.ssim)
        });
        let count = metrics.len() as f64;
        let lpips_vgg = metrics
            .iter()
            .try_fold(0.0, |sum, metric| metric.lpips_vgg.map(|value| sum + value))
            .map(|sum| sum / count);
        Some((psnr / count, ssim / count, lpips_vgg))
    }

    fn validate_metric_values(psnr: f64, ssim: f64, lpips_vgg: Option<f64>) -> Result<()> {
        if psnr.is_nan() || psnr == f64::NEG_INFINITY {
            bail!("PSNR is not a valid finite value or positive infinity");
        }
        if !ssim.is_finite() {
            bail!("SSIM is not finite");
        }
        if lpips_vgg.is_some_and(|value| !value.is_finite()) {
            bail!("VGG LPIPS is not finite");
        }
        Ok(())
    }

    fn metric_value_json(value: f64) -> serde_json::Value {
        if value == f64::INFINITY {
            json!("inf")
        } else if value == f64::NEG_INFINITY {
            json!("-inf")
        } else if value.is_nan() {
            json!("nan")
        } else {
            json!(value)
        }
    }

    fn alpha_mode_name(mode: AlphaMode) -> &'static str {
        match mode {
            AlphaMode::Masked => "masked",
            AlphaMode::Transparent => "transparent",
        }
    }

    fn metric_json(metric: &ViewMetric) -> serde_json::Value {
        json!({
            "index": metric.index,
            "image": metric.image,
            "width": metric.width,
            "height": metric.height,
            "psnr": metric_value_json(metric.psnr),
            "ssim": metric_value_json(metric.ssim),
            "lpips_vgg": metric.lpips_vgg.map(metric_value_json),
        })
    }

    fn render_file_name(index: usize, image_name: &str) -> String {
        let stem = Path::new(image_name)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .map_or_else(|| format!("view_{index:04}"), ToOwned::to_owned);
        format!("{index:04}_{stem}.png")
    }

    async fn load_checkpoint(
        path: &Path,
        device: &Device,
    ) -> Result<brush_render::gaussian_splats::Splats> {
        let vfs = BrushVfs::from_path(path)
            .await
            .with_context(|| format!("failed to mount checkpoint {}", path.display()))?;
        let ply_path = vfs
            .file_paths()
            .next()
            .context("checkpoint VFS contained no files")?;
        let reader = vfs
            .reader_at_path(&ply_path)
            .await
            .context("failed to open checkpoint")?;
        let message = load_splat_from_ply(reader, None)
            .await
            .context("failed to parse checkpoint PLY")?;
        let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
        Ok(message.data.into_splats(device, mode))
    }

    #[tokio::main(flavor = "current_thread")]
    pub(super) async fn run() -> Result<()> {
        let args = Args::parse();
        validate_args(&args)?;

        let device = Device::from(brush_process::burn_init_setup().await);
        let checkpoint = load_checkpoint(&args.ply, &device).await?;
        let splat_count = checkpoint.num_splats();
        let lpips_model = args.lpips.then(|| load_vgg_lpips(&device));

        let dataset_vfs = Arc::new(
            BrushVfs::from_path(&args.dataset)
                .await
                .with_context(|| format!("failed to mount dataset {}", args.dataset.display()))?,
        );
        let load_config = LoadDatasetConfig {
            max_frames: None,
            max_resolution: args.max_resolution,
            eval_split_every: Some(args.eval_split_every),
            subsample_frames: None,
            subsample_points: None,
            alpha_mode: Some(args.alpha_mode),
            train_on_eval: false,
            // Evaluation loads each held-out view exactly once; retain the
            // conventional native budget for loader/config parity.
            max_scene_batch_cache_size: 6 * 1024 * 1024 * 1024,
        };
        let loaded = load_dataset(dataset_vfs, &load_config)
            .await
            .context("failed to load dataset")?;
        for warning in &loaded.warnings {
            println!("BRUSH_EVAL_WARNING {}", json!({ "message": warning }));
        }
        let warnings = loaded.warnings;
        let eval_scene = loaded
            .dataset
            .eval
            .context("dataset produced no held-out evaluation views")?;
        if eval_scene.views.is_empty() {
            bail!("dataset produced no held-out evaluation views");
        }

        if let Some(save_dir) = &args.save_dir {
            tokio::fs::create_dir_all(save_dir)
                .await
                .with_context(|| format!("failed to create {}", save_dir.display()))?;
        }

        let mut metrics = Vec::with_capacity(eval_scene.views.len());
        for (index, view) in eval_scene.views.iter().enumerate() {
            let image_name = view.image.img_name();
            let gt_image = view
                .image
                .load()
                .await
                .with_context(|| format!("failed to load evaluation image {image_name}"))?;
            let (width, height) = (gt_image.width(), gt_image.height());
            let sample = eval_stats(
                checkpoint.clone(),
                &view.camera,
                gt_image,
                view.image.alpha_mode(),
                &device,
                None,
            )
            .await
            .with_context(|| format!("failed to evaluate {image_name}"))?;

            let psnr = sample
                .psnr
                .clone()
                .into_scalar_async::<f32>()
                .await
                .with_context(|| format!("failed to read PSNR for {image_name}"))?
                as f64;
            let ssim = sample
                .ssim
                .clone()
                .into_scalar_async::<f32>()
                .await
                .with_context(|| format!("failed to read SSIM for {image_name}"))?
                as f64;
            let lpips_vgg = calculate_lpips(&sample, lpips_model.as_ref(), &image_name).await?;
            validate_metric_values(psnr, ssim, lpips_vgg)
                .with_context(|| format!("invalid metrics for {image_name}"))?;

            if let Some(save_dir) = &args.save_dir {
                let path = save_dir.join(render_file_name(index, &image_name));
                sample
                    .save_to_disk(&path)
                    .await
                    .with_context(|| format!("failed to save render {}", path.display()))?;
            }

            let metric = ViewMetric {
                index,
                image: image_name,
                width,
                height,
                psnr,
                ssim,
                lpips_vgg,
            };
            println!("BRUSH_EVAL_VIEW {}", metric_json(&metric));
            metrics.push(metric);
        }

        let (avg_psnr, avg_ssim, avg_lpips_vgg) =
            averages(&metrics).context("dataset produced no held-out evaluation metrics")?;
        let compiler = if cfg!(all(feature = "native-msl", target_os = "macos")) {
            "native-msl"
        } else {
            "wgsl"
        };
        let per_view = metrics.iter().map(metric_json).collect::<Vec<_>>();
        println!(
            "BRUSH_EVAL_RESULT {}",
            json!({
                "compiler": compiler,
                "dataset": args.dataset.display().to_string(),
                "ply": args.ply.display().to_string(),
                "splats": splat_count,
                "views": metrics.len(),
                "max_resolution": args.max_resolution,
                "requested_eval_split_every": args.eval_split_every,
                "eval_selection": "dataset-loader-held-out",
                "alpha_mode": alpha_mode_name(args.alpha_mode),
                "avg_psnr": metric_value_json(avg_psnr),
                "avg_ssim": metric_value_json(avg_ssim),
                "avg_lpips_vgg": avg_lpips_vgg.map(metric_value_json),
                "lpips_variant": args.lpips.then_some("vgg"),
                "lpips_policy": args.lpips.then_some("full-frame-at-eval-resolution"),
                "lpips_alpha_mode": args.lpips.then_some("gt-black-composited"),
                "warnings": warnings,
                "per_view": per_view,
            })
        );

        Ok(())
    }

    async fn calculate_lpips(
        sample: &brush_train::eval::EvalSample,
        model: Option<&LpipsModel>,
        image_name: &str,
    ) -> Result<Option<f64>> {
        let Some(model) = model else {
            return Ok(None);
        };
        let [height, width, channels] = sample.rendered.dims();
        validate_lpips_view(width, height, channels, image_name)?;

        // Rendered RGB already uses a black background. Composite GT alpha onto
        // black to avoid scoring arbitrary hidden RGB in masked source images.
        // Unlike historical bake-off scripts, this metric intentionally stays
        // at the caller's evaluation resolution and does not mask the render.
        let target = lpips_target_image(&sample.gt_img);
        let (target_width, target_height) = target.dimensions();
        if target_width as usize != width || target_height as usize != height {
            bail!(
                "VGG LPIPS render/target dimensions differ for {image_name}: {width}x{height} versus {target_width}x{target_height}"
            );
        }
        let target = Tensor::<4>::from_data(
            TensorData::new(
                target.into_vec(),
                [1, target_height as usize, target_width as usize, 3],
            ),
            &sample.rendered.device(),
        );

        let value = model
            // `eval_stats` preserves out-of-gamut float values for its legacy
            // metrics, while LPIPS is defined on RGB normalized to [0, 1].
            .lpips(
                sample.rendered.clone().clamp(0.0, 1.0).unsqueeze_dim(0),
                target,
            )
            .into_scalar_async::<f32>()
            .await
            .with_context(|| format!("failed to read VGG LPIPS for {image_name}"))?;
        Ok(Some(value as f64))
    }

    fn validate_lpips_view(
        width: usize,
        height: usize,
        channels: usize,
        image_name: &str,
    ) -> Result<()> {
        if height < LPIPS_MIN_IMAGE_SIZE || width < LPIPS_MIN_IMAGE_SIZE {
            bail!(
                "VGG LPIPS requires images of at least {LPIPS_MIN_IMAGE_SIZE}x{LPIPS_MIN_IMAGE_SIZE}; {image_name} is {width}x{height}"
            );
        }
        if channels != 3 {
            bail!("VGG LPIPS requires RGB renders; {image_name} has {channels} channels");
        }
        Ok(())
    }

    fn lpips_target_image(gt_img: &image::DynamicImage) -> image::Rgb32FImage {
        view_to_sample_image(gt_img.clone(), AlphaMode::Transparent).to_rgb32f()
    }

    #[cfg(test)]
    mod tests {
        use super::{
            Args, ViewMetric, averages, lpips_target_image, metric_value_json, render_file_name,
            validate_args, validate_lpips_view, validate_metric_values,
        };
        use brush_render::AlphaMode;
        use clap::Parser;
        use image::{DynamicImage, RgbaImage};
        use std::path::PathBuf;

        fn args() -> Args {
            Args::try_parse_from([
                "brush-eval-checkpoint",
                "--dataset",
                "dataset",
                "--ply",
                "checkpoint.ply",
                "--alpha-mode",
                "masked",
            ])
            .expect("valid args")
        }

        #[test]
        fn args_use_reproducible_defaults() {
            let args = args();
            assert_eq!(args.dataset, PathBuf::from("dataset"));
            assert_eq!(args.ply, PathBuf::from("checkpoint.ply"));
            assert_eq!(args.max_resolution, 1920);
            assert_eq!(args.eval_split_every, 20);
            assert_eq!(args.alpha_mode, AlphaMode::Masked);
            assert_eq!(args.save_dir, None);
            assert!(!args.lpips);
            validate_args(&args).expect("defaults are valid");
        }

        #[test]
        fn alpha_mode_is_explicitly_required() {
            assert!(
                Args::try_parse_from([
                    "brush-eval-checkpoint",
                    "--dataset",
                    "dataset",
                    "--ply",
                    "checkpoint.ply",
                ])
                .is_err()
            );
        }

        #[test]
        fn validation_rejects_empty_resolution_and_degenerate_split() {
            let mut args = args();
            args.max_resolution = 0;
            assert_eq!(
                validate_args(&args).unwrap_err().to_string(),
                "--max-resolution must be at least 1"
            );

            args.max_resolution = 1920;
            args.eval_split_every = 1;
            assert_eq!(
                validate_args(&args).unwrap_err().to_string(),
                "--eval-split-every must be at least 2"
            );
        }

        #[test]
        fn averages_weight_each_view_equally() {
            let metrics = [
                ViewMetric {
                    index: 0,
                    image: "a.jpg".to_owned(),
                    width: 1,
                    height: 1,
                    psnr: 20.0,
                    ssim: 0.8,
                    lpips_vgg: Some(0.4),
                },
                ViewMetric {
                    index: 1,
                    image: "b.jpg".to_owned(),
                    width: 2,
                    height: 2,
                    psnr: 30.0,
                    ssim: 1.0,
                    lpips_vgg: Some(0.2),
                },
            ];
            let (psnr, ssim, lpips_vgg) = averages(&metrics).expect("non-empty metrics");
            assert_eq!(psnr, 25.0);
            assert!((ssim - 0.9).abs() < 1e-12);
            assert!((lpips_vgg.expect("enabled LPIPS") - 0.3).abs() < 1e-12);
            assert_eq!(averages(&[]), None);

            let metrics_without_lpips = metrics.map(|metric| ViewMetric {
                lpips_vgg: None,
                ..metric
            });
            assert_eq!(averages(&metrics_without_lpips), Some((25.0, 0.9, None)));
        }

        #[test]
        fn machine_output_preserves_positive_infinite_psnr() {
            assert_eq!(metric_value_json(f64::INFINITY), serde_json::json!("inf"));
            validate_metric_values(f64::INFINITY, 1.0, Some(0.0)).expect("perfect PSNR is valid");
            assert!(validate_metric_values(f64::NAN, 1.0, None).is_err());
            assert!(validate_metric_values(20.0, f64::NAN, None).is_err());
            assert!(validate_metric_values(20.0, 1.0, Some(f64::NAN)).is_err());
        }

        #[test]
        fn render_names_are_unique_and_keep_the_source_stem() {
            assert_eq!(
                render_file_name(3, "nested name.jpg"),
                "0003_nested name.png"
            );
            assert_eq!(render_file_name(4, ""), "0004_view_0004.png");
        }

        #[test]
        fn lpips_validates_shape_and_black_composites_alpha() {
            validate_lpips_view(400, 300, 3, "opaque.jpg").expect("valid RGB shape");
            assert!(validate_lpips_view(15, 16, 3, "tiny.jpg").is_err());

            let rgba =
                RgbaImage::from_raw(1, 1, vec![200, 100, 50, 128]).expect("valid RGBA image");
            let target = lpips_target_image(&DynamicImage::ImageRgba8(rgba));
            let pixel = target.get_pixel(0, 0).0;
            assert!((pixel[0] - 100.0 / 255.0).abs() < 1e-6);
            assert!((pixel[1] - 50.0 / 255.0).abs() < 1e-6);
            assert!((pixel[2] - 25.0 / 255.0).abs() < 1e-6);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn main() -> anyhow::Result<()> {
    native::run()
}

#[cfg(target_family = "wasm")]
fn main() {
    eprintln!("brush-eval-checkpoint is only available on native targets");
}
