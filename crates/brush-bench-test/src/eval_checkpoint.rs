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
    use brush_dataset::{config::LoadDatasetConfig, load_dataset};
    use brush_render::{AlphaMode, gaussian_splats::SplatRenderMode};
    use brush_serde::load_splat_from_ply;
    use brush_train::eval::eval_stats;
    use brush_vfs::BrushVfs;
    use burn::prelude::Device;
    use clap::Parser;
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
    }

    #[derive(Debug, Clone, PartialEq)]
    struct ViewMetric {
        index: usize,
        image: String,
        width: u32,
        height: u32,
        psnr: f64,
        ssim: f64,
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

    fn averages(metrics: &[ViewMetric]) -> Option<(f64, f64)> {
        if metrics.is_empty() {
            return None;
        }
        let (psnr, ssim) = metrics.iter().fold((0.0, 0.0), |(psnr, ssim), metric| {
            (psnr + metric.psnr, ssim + metric.ssim)
        });
        let count = metrics.len() as f64;
        Some((psnr / count, ssim / count))
    }

    fn validate_metric_values(psnr: f64, ssim: f64) -> Result<()> {
        if psnr.is_nan() || psnr == f64::NEG_INFINITY {
            bail!("PSNR is not a valid finite value or positive infinity");
        }
        if !ssim.is_finite() {
            bail!("SSIM is not finite");
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
            estimate_metric_scale: false,
            // Evaluation loads each held-out view exactly once; retain the
            // conventional native budget for loader/config parity.
            max_scene_batch_cache_size: 6 * 1024 * 1024 * 1024,
            features_dir_name: "dino_features".to_owned(),
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
            validate_metric_values(psnr, ssim)
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
            };
            println!("BRUSH_EVAL_VIEW {}", metric_json(&metric));
            metrics.push(metric);
        }

        let (avg_psnr, avg_ssim) =
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
                "warnings": warnings,
                "per_view": per_view,
            })
        );

        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::{
            Args, ViewMetric, averages, metric_value_json, render_file_name, validate_args,
            validate_metric_values,
        };
        use brush_render::AlphaMode;
        use clap::Parser;
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
                },
                ViewMetric {
                    index: 1,
                    image: "b.jpg".to_owned(),
                    width: 2,
                    height: 2,
                    psnr: 30.0,
                    ssim: 1.0,
                },
            ];
            assert_eq!(averages(&metrics), Some((25.0, 0.9)));
            assert_eq!(averages(&[]), None);
        }

        #[test]
        fn machine_output_preserves_positive_infinite_psnr() {
            assert_eq!(metric_value_json(f64::INFINITY), serde_json::json!("inf"));
            validate_metric_values(f64::INFINITY, 1.0).expect("perfect PSNR is valid");
            assert!(validate_metric_values(f64::NAN, 1.0).is_err());
            assert!(validate_metric_values(20.0, f64::NAN).is_err());
        }

        #[test]
        fn render_names_are_unique_and_keep_the_source_stem() {
            assert_eq!(
                render_file_name(3, "nested name.jpg"),
                "0003_nested name.png"
            );
            assert_eq!(render_file_name(4, ""), "0004_view_0004.png");
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
