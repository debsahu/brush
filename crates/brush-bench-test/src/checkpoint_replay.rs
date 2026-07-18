//! Steady-state training replay using an exported splat checkpoint and real views.
//!
//! This is a standalone binary, rather than a Divan target, so the normal
//! benchmark suite never depends on external datasets. It restores splat
//! parameters from PLY but deliberately starts fresh optimizer state; after
//! warmup, the tensor shapes and GPU kernel workload match a resumed model.

#[cfg(not(target_family = "wasm"))]
mod native {
    use std::{path::PathBuf, sync::Arc, time::Instant};

    use anyhow::{Context, Result, bail};
    use brush_dataset::{
        config::LoadDatasetConfig,
        load_dataset,
        scene::{SceneBatch, sample_to_packed_data, view_to_sample_image},
    };
    use brush_render::{AlphaMode, gaussian_splats::SplatRenderMode};
    use brush_render_bwd::burn_glue::lift_splats_to_autodiff;
    use brush_serde::load_splat_from_ply;
    use brush_train::{
        WASSERSTEIN_MIN_IMAGE_SIZE,
        config::TrainConfig,
        train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
    };
    use brush_vfs::BrushVfs;
    use burn::{module::AutodiffModule, prelude::Device, tensor::Tensor};
    use clap::Parser;

    #[derive(Debug, Parser)]
    #[command(about = "Replay steady-state training from a real PLY checkpoint")]
    struct Args {
        /// Dataset directory containing cameras and source images.
        #[arg(long)]
        dataset: PathBuf,

        /// Exported Brush PLY checkpoint to replay.
        #[arg(long)]
        ply: PathBuf,

        /// Long-edge image resolution used while loading views.
        #[arg(long, default_value_t = 1920)]
        max_resolution: u32,

        /// Number of evenly spaced training views to cycle through.
        #[arg(long, default_value_t = 4)]
        views: usize,

        /// Optional dataset evaluation split, matching Brush's training CLI.
        #[arg(long)]
        eval_split_every: Option<usize>,

        /// How alpha from images or masks participates in the loss.
        #[arg(long, value_enum)]
        alpha_mode: Option<AlphaMode>,

        /// Untimed steps used to compile pipelines and initialize optimizer state.
        #[arg(long, default_value_t = 4)]
        warmup_steps: u32,

        /// Training steps between GPU synchronizations in each timed sample.
        #[arg(long, default_value_t = 4)]
        steps_per_sample: u32,

        /// Number of synchronized timing samples.
        #[arg(long, default_value_t = 20)]
        samples: usize,

        /// Seed for GPU-side training noise.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Global training iteration represented by the first replay step.
        #[arg(long, default_value_t = 0)]
        start_iter: u32,

        /// WD-R objective scale. Zero benchmarks the normal RGB objective.
        #[arg(long, default_value_t = 0.0)]
        wd_r_gamma: f32,

        /// Global iteration at which WD-R replaces the warm-up RGB objective.
        #[arg(long, default_value_t = 3000)]
        wd_r_warmup_iters: u32,

        /// Skip the refinement-only raster gradient statistic for late-phase A/B timing.
        #[arg(long)]
        skip_refine_weight: bool,

        /// Analyze this many deterministic tiles per selected view during the first untimed
        /// warmup cycle. Requires the `raster-census` Cargo feature and invalidates warmup timing.
        #[cfg(feature = "raster-census")]
        #[arg(long, value_name = "TILES")]
        raster_census_tiles: Option<usize>,
    }

    fn validate_args(args: &Args) -> Result<()> {
        if args.max_resolution == 0 {
            bail!("--max-resolution must be at least 1");
        }
        if args.views == 0 {
            bail!("--views must be at least 1");
        }
        if args.warmup_steps == 0 {
            bail!("--warmup-steps must be at least 1");
        }
        if args.steps_per_sample == 0 {
            bail!("--steps-per-sample must be at least 1");
        }
        if args.samples == 0 {
            bail!("--samples must be at least 1");
        }
        if args.eval_split_every == Some(0) {
            bail!("--eval-split-every must be at least 1");
        }
        if !args.wd_r_gamma.is_finite() || args.wd_r_gamma < 0.0 {
            bail!("--wd-r-gamma must be finite and non-negative");
        }
        if args.wd_r_gamma > 0.0 && args.alpha_mode == Some(AlphaMode::Masked) {
            bail!("WD-R does not support --alpha-mode masked; use transparent compositing");
        }
        if args.wd_r_gamma > 0.0 && args.start_iter < args.wd_r_warmup_iters {
            bail!(
                "WD-R is not active at --start-iter {}; use a value at or above --wd-r-warmup-iters {} for steady-state replay",
                args.start_iter,
                args.wd_r_warmup_iters
            );
        }
        let sample_count = u32::try_from(args.samples).context("--samples is too large")?;
        let timed_steps = args
            .steps_per_sample
            .checked_mul(sample_count)
            .context("timed replay step count overflows u32")?;
        let replay_steps = args
            .warmup_steps
            .checked_add(timed_steps)
            .context("total replay step count overflows u32")?;
        args.start_iter
            .checked_add(replay_steps)
            .context("global replay iteration overflows u32")?;
        #[cfg(feature = "raster-census")]
        if args.raster_census_tiles == Some(0) {
            bail!("--raster-census-tiles must be at least 1");
        }
        Ok(())
    }

    async fn load_batches(args: &Args) -> Result<(Vec<SceneBatch>, Vec<String>)> {
        let vfs = Arc::new(
            BrushVfs::from_path(&args.dataset)
                .await
                .with_context(|| format!("failed to mount dataset {}", args.dataset.display()))?,
        );
        let load_config = LoadDatasetConfig {
            max_frames: None,
            max_resolution: args.max_resolution,
            eval_split_every: args.eval_split_every,
            subsample_frames: None,
            subsample_points: None,
            alpha_mode: args.alpha_mode,
            train_on_eval: false,
            // The replay owns its few decoded views directly, so the scene-loader
            // cache is unused. Keep the conventional value for config parity.
            max_scene_batch_cache_size: 6 * 1024 * 1024 * 1024,
        };
        let loaded = load_dataset(vfs, &load_config)
            .await
            .context("failed to load dataset")?;
        let scene = loaded.dataset.train;
        let view_count = args.views.min(scene.views.len());
        let mut batches = Vec::with_capacity(view_count);
        let mut view_labels = Vec::with_capacity(view_count);

        for slot in 0..view_count {
            let index = slot * scene.views.len() / view_count;
            let view = &scene.views[index];
            let image = view.image.load().await.with_context(|| {
                format!(
                    "failed to load replay view {index} of {}",
                    scene.views.len()
                )
            })?;
            let sample = view_to_sample_image(image, view.image.alpha_mode());
            let (img_packed, has_alpha) = sample_to_packed_data(sample);
            let batch = SceneBatch {
                img_packed,
                has_alpha,
                alpha_mode: view.image.alpha_mode(),
                camera: view.camera,
                view_index: index,
            };
            let [height, width] = batch.img_size();
            view_labels.push(format!(
                "{}:{width}x{height}:{:?}",
                view.image.img_name(),
                view.image.alpha_mode()
            ));
            batches.push(batch);
        }

        Ok((batches, view_labels))
    }

    async fn load_checkpoint(
        args: &Args,
        device: &Device,
    ) -> Result<brush_render::gaussian_splats::Splats> {
        let vfs = BrushVfs::from_path(&args.ply)
            .await
            .with_context(|| format!("failed to mount checkpoint {}", args.ply.display()))?;
        let path = vfs
            .file_paths()
            .next()
            .context("checkpoint VFS contained no files")?;
        let reader = vfs
            .reader_at_path(&path)
            .await
            .context("failed to open checkpoint")?;
        let message = load_splat_from_ply(reader, None)
            .await
            .context("failed to parse checkpoint PLY")?;
        let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
        Ok(message.data.into_splats(device, mode))
    }

    async fn run_steps(
        trainer: &mut SplatTrainer,
        splats: &mut Option<brush_render::gaussian_splats::Splats>,
        batches: &[SceneBatch],
        step_offset: u32,
        start_iter: u32,
        steps: u32,
        compute_refine_weight: bool,
    ) -> Option<Tensor<1>> {
        let mut last_loss = None;
        for step in 0..steps {
            let replay_step = step_offset
                .checked_add(step)
                .expect("validated replay step range");
            let global_iter = start_iter
                .checked_add(replay_step)
                .expect("validated global replay iteration range");
            let batch = batches[replay_step as usize % batches.len()].clone();
            let current = splats.take().expect("replay always restores splats");
            let differentiable = lift_splats_to_autodiff(current);
            let (updated, stats) = trainer
                .step_with_refine_weight_at(
                    global_iter,
                    batch,
                    differentiable,
                    compute_refine_weight,
                )
                .await;
            *splats = Some(updated.valid());
            last_loss = Some(stats.loss);
        }
        last_loss
    }

    fn percentile(sorted: &[f64], fraction: f64) -> f64 {
        let index = ((sorted.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        sorted[index]
    }

    fn median(sorted: &[f64]) -> f64 {
        let middle = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    #[tokio::main(flavor = "current_thread")]
    pub(super) async fn run() -> Result<()> {
        let args = Args::parse();
        validate_args(&args)?;

        // Match the CLI's command batching and GPU memory allocator, then mirror
        // its inner/lift/valid splat lifecycle in `run_steps`.
        let device = Device::from(brush_process::burn_init_setup().await);
        device.seed(args.seed);

        let (batches, view_labels) = load_batches(&args).await?;
        if args.wd_r_gamma > 0.0 {
            if batches
                .iter()
                .any(|batch| batch.alpha_mode == AlphaMode::Masked)
            {
                bail!(
                    "WD-R resolved a masked-alpha replay view; pass --alpha-mode transparent to composite masks"
                );
            }
            for (batch, label) in batches.iter().zip(&view_labels) {
                let [height, width] = batch.img_size();
                if height < WASSERSTEIN_MIN_IMAGE_SIZE || width < WASSERSTEIN_MIN_IMAGE_SIZE {
                    bail!(
                        "WD-R requires replay views of at least {WASSERSTEIN_MIN_IMAGE_SIZE}x{WASSERSTEIN_MIN_IMAGE_SIZE}; {label} is {width}x{height}"
                    );
                }
            }
        }
        if !(args.warmup_steps as usize).is_multiple_of(batches.len()) {
            bail!(
                "--warmup-steps must be a multiple of the {} selected views",
                batches.len()
            );
        }
        if !(args.steps_per_sample as usize).is_multiple_of(batches.len()) {
            bail!(
                "--steps-per-sample must be a multiple of the {} selected views",
                batches.len()
            );
        }
        let checkpoint = load_checkpoint(&args, &device).await?;
        let splat_count = checkpoint.num_splats();
        let bounds = get_splat_bounds(checkpoint.clone(), BOUND_PERCENTILE).await;
        let mut splats = Some(checkpoint);
        let config = TrainConfig {
            background_noise_strength: 0.0,
            wd_r_gamma: args.wd_r_gamma,
            wd_r_warmup_iters: args.wd_r_warmup_iters,
            ..TrainConfig::default()
        };
        let mut trainer = SplatTrainer::try_new(&config, &device, bounds)
            .context("invalid replay training configuration")?;

        let compute_refine_weight = !args.skip_refine_weight;
        #[cfg(feature = "raster-census")]
        if let Some(sample_tiles) = args.raster_census_tiles {
            brush_render::raster_census::request(batches.len(), sample_tiles)
                .map_err(anyhow::Error::msg)?;
        }
        let _ = run_steps(
            &mut trainer,
            &mut splats,
            &batches,
            0,
            args.start_iter,
            args.warmup_steps,
            compute_refine_weight,
        )
        .await;
        device.sync().context("failed to synchronize warmup")?;

        let mut sample_ms_per_step = Vec::with_capacity(args.samples);
        let mut final_loss = None;
        let mut step_offset = args.warmup_steps;
        for _ in 0..args.samples {
            let start = Instant::now();
            final_loss = run_steps(
                &mut trainer,
                &mut splats,
                &batches,
                step_offset,
                args.start_iter,
                args.steps_per_sample,
                compute_refine_weight,
            )
            .await;
            device.sync().context("failed to synchronize sample")?;
            sample_ms_per_step
                .push(start.elapsed().as_secs_f64() * 1000.0 / f64::from(args.steps_per_sample));
            step_offset += args.steps_per_sample;
        }

        let final_loss = final_loss
            .context("replay produced no final loss")?
            .into_scalar_async::<f32>()
            .await
            .context("failed to read final loss")?;
        let measured_steps = args
            .steps_per_sample
            .checked_mul(u32::try_from(args.samples).expect("validated sample count"))
            .expect("validated measured step count");
        let global_end_iter = args
            .start_iter
            .checked_add(args.warmup_steps)
            .and_then(|iter| iter.checked_add(measured_steps))
            .expect("validated global replay iteration range");

        let raw_samples = sample_ms_per_step
            .iter()
            .map(|sample| format!("{sample:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        sample_ms_per_step.sort_by(f64::total_cmp);
        let median = median(&sample_ms_per_step);
        let p95 = percentile(&sample_ms_per_step, 0.95);
        let mean = sample_ms_per_step.iter().sum::<f64>() / sample_ms_per_step.len() as f64;
        let min = sample_ms_per_step[0];
        let max = sample_ms_per_step[sample_ms_per_step.len() - 1];
        let compiler = if cfg!(all(feature = "native-msl", target_os = "macos")) {
            "native-msl"
        } else {
            "wgsl"
        };
        let preset_requested = brush_render::native_msl::preset_requested();
        let unchecked_raster_requested = brush_render::native_msl::option_requested(
            brush_render::native_msl::UNCHECKED_RASTER_BWD_ENV,
        );
        let fused_sh_adam_requested =
            brush_render::native_msl::option_requested(brush_render::native_msl::FUSED_SH_ADAM_ENV);
        let coalesced_sh_grad_requested = brush_render::native_msl::option_requested(
            brush_render::native_msl::COALESCED_SH_GRAD_ENV,
        );
        let saved_loss_partials_requested = brush_render::native_msl::option_requested(
            brush_render::native_msl::SAVED_LOSS_PARTIALS_ENV,
        );
        let sparse_sh_adam_requested = brush_render::native_msl::option_requested(
            brush_render::native_msl::SPARSE_SH_ADAM_ENV,
        );
        let fine_raster_tiles_requested = brush_render::native_msl::fine_raster_tiles_requested();

        println!("checkpoint: {}", args.ply.display());
        println!("dataset: {}", args.dataset.display());
        println!("splats: {splat_count}");
        println!("views: {} ({})", batches.len(), view_labels.join(", "));
        println!("refinement weight: {compute_refine_weight}");
        println!(
            "objective: {} | WD-R gamma: {} | WD-R warm-up: {} | global range: {}..{}",
            if args.wd_r_gamma > 0.0 {
                "wd-r"
            } else {
                "baseline"
            },
            args.wd_r_gamma,
            args.wd_r_warmup_iters,
            args.start_iter,
            global_end_iter,
        );
        println!(
            "compiler: {compiler} | native MSL preset requested: {preset_requested} | unchecked raster requested: {unchecked_raster_requested} | fused SH Adam requested: {fused_sh_adam_requested} | coalesced SH grad requested: {coalesced_sh_grad_requested} | saved loss partials requested: {saved_loss_partials_requested} | sparse SH Adam requested: {sparse_sh_adam_requested} | fine raster tiles requested: {fine_raster_tiles_requested} | seed: {}",
            args.seed
        );
        println!(
            "samples: {} x {} steps ({} warmup)",
            args.samples, args.steps_per_sample, args.warmup_steps
        );
        println!(
            "median {median:.3} ms/step | p95 {p95:.3} | mean {mean:.3} | min {min:.3} | max {max:.3} | {:.2} steps/s",
            1000.0 / median
        );
        println!("final loss: {final_loss:.9}");
        println!(
            "BRUSH_REPLAY_RESULT compiler={compiler} native_msl_preset_requested={preset_requested} unchecked_raster_requested={unchecked_raster_requested} fused_sh_adam_requested={fused_sh_adam_requested} coalesced_sh_grad_requested={coalesced_sh_grad_requested} saved_loss_partials_requested={saved_loss_partials_requested} sparse_sh_adam_requested={sparse_sh_adam_requested} fine_raster_tiles_requested={fine_raster_tiles_requested} compute_refine_weight={compute_refine_weight} loss_mode={} seed={} global_iter_start={} timed_global_iter_start={} global_iter_end_exclusive={global_end_iter} wd_r_gamma={} wd_r_warmup_iters={} splats={splat_count} views={} view_set={} samples={} steps_per_sample={} warmup_steps={} median_ms={median:.6} p95_ms={p95:.6} mean_ms={mean:.6} min_ms={min:.6} max_ms={max:.6} steps_per_s={:.6} final_loss={final_loss:.9}",
            if args.wd_r_gamma > 0.0 {
                "wd-r"
            } else {
                "baseline"
            },
            args.seed,
            args.start_iter,
            args.start_iter + args.warmup_steps,
            args.wd_r_gamma,
            args.wd_r_warmup_iters,
            batches.len(),
            view_labels.join(","),
            args.samples,
            args.steps_per_sample,
            args.warmup_steps,
            1000.0 / median
        );
        println!(
            "BRUSH_REPLAY_SAMPLES compute_refine_weight={compute_refine_weight} ms_per_step={raw_samples}"
        );

        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::{Args, median, percentile, validate_args};
        use clap::Parser;

        #[test]
        fn percentile_uses_nearest_rank() {
            let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
            assert_eq!(percentile(&samples, 0.5), 3.0);
            assert_eq!(percentile(&samples, 0.95), 5.0);
        }

        #[test]
        fn median_averages_the_middle_pair() {
            assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
            assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        }

        #[test]
        fn wd_r_replay_requires_active_global_iteration_and_transparent_masks() {
            let inactive = Args::try_parse_from([
                "replay",
                "--dataset",
                "dataset",
                "--ply",
                "checkpoint.ply",
                "--wd-r-gamma",
                "0.028",
            ])
            .expect("valid syntax");
            assert!(
                validate_args(&inactive)
                    .expect_err("inactive WD-R must be rejected")
                    .to_string()
                    .contains("--start-iter")
            );

            let masked = Args::try_parse_from([
                "replay",
                "--dataset",
                "dataset",
                "--ply",
                "checkpoint.ply",
                "--wd-r-gamma",
                "0.028",
                "--start-iter",
                "15000",
                "--alpha-mode",
                "masked",
            ])
            .expect("valid syntax");
            assert!(
                validate_args(&masked)
                    .expect_err("masked WD-R must be rejected")
                    .to_string()
                    .contains("transparent")
            );

            let transparent = Args::try_parse_from([
                "replay",
                "--dataset",
                "dataset",
                "--ply",
                "checkpoint.ply",
                "--wd-r-gamma",
                "0.028",
                "--start-iter",
                "15000",
                "--alpha-mode",
                "transparent",
            ])
            .expect("valid syntax");
            validate_args(&transparent).expect("active transparent WD-R must be valid");
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn main() -> anyhow::Result<()> {
    native::run()
}

#[cfg(target_family = "wasm")]
fn main() {
    eprintln!("brush-checkpoint-replay is only available on native targets");
}
