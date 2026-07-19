#![recursion_limit = "256"]
#![cfg(not(target_family = "wasm"))]

use brush_async::Actor;
use brush_process::DataSource;
use brush_process::RunningProcess;
use brush_process::config::TrainStreamConfig;
use brush_process::create_process;
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;

use clap::{
    CommandFactory, Error, FromArgMatches, Parser, builder::ArgPredicate, error::ErrorKind,
    parser::ValueSource,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::trace_span;

#[derive(Parser)]
#[command(
    author,
    version,
    arg_required_else_help = false,
    about = "Brush - universal splats"
)]
pub struct Cli {
    /// Source to load from (path or URL).
    #[arg(value_name = "PATH_OR_URL")]
    pub source: Option<DataSource>,

    #[arg(
        long,
        default_value = "true",
        default_value_if("source", ArgPredicate::IsPresent, "false"),
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,
}

/// Parsed CLI plus the training fields explicitly supplied by the user. This
/// keeps occurrence metadata without changing the public shape of [`Cli`].
pub struct ParsedCli {
    cli: Cli,
    explicit_train_fields: Vec<String>,
}

impl std::ops::Deref for ParsedCli {
    type Target = Cli;

    fn deref(&self) -> &Self::Target {
        &self.cli
    }
}

impl Cli {
    pub fn parse_with_explicit_fields() -> ParsedCli {
        Self::try_parse_from_with_explicit_fields(std::env::args_os())
            .unwrap_or_else(|error| error.exit())
    }

    pub fn try_parse_from_with_explicit_fields<I, T>(args: I) -> Result<ParsedCli, Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let command = Self::command();
        let train_fields: Vec<(String, String)> = command
            .get_arguments()
            .filter_map(|arg| {
                let id = arg.get_id().as_str();
                if matches!(id, "source" | "with_viewer") {
                    return None;
                }
                arg.get_long().map(|long| (id.to_owned(), long.to_owned()))
            })
            .collect();
        let mut matches = command.try_get_matches_from(args)?;
        let explicit_train_fields = train_fields
            .into_iter()
            .filter_map(|(id, long)| {
                (matches.value_source(&id) == Some(ValueSource::CommandLine)).then_some(long)
            })
            .collect();
        let cli = Self::from_arg_matches_mut(&mut matches)?;
        Ok(ParsedCli {
            cli,
            explicit_train_fields,
        })
    }

    pub fn validate(self) -> Result<Self, Error> {
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}

impl ParsedCli {
    pub fn validate(self) -> Result<Self, Error> {
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}

/// Build the training process described by `args`, or `None` if no source was
/// given. Shared by the standalone CLI binary and brush-app's headless path.
pub fn build_process(args: &Cli) -> Option<RunningProcess> {
    build_process_with_fields(args, &[])
}

/// Build a process from occurrence-aware parsed arguments.
pub fn build_parsed_process(args: &ParsedCli) -> Option<RunningProcess> {
    build_process_with_fields(&args.cli, &args.explicit_train_fields)
}

fn build_process_with_fields(
    args: &Cli,
    explicit_train_fields: &[String],
) -> Option<RunningProcess> {
    let source = args.source.clone()?;
    let cli_config = args.train_stream.clone();
    let explicit_train_fields = explicit_train_fields.to_vec();
    Some(create_process(source, async move |init| {
        Some(if explicit_train_fields.is_empty() {
            brush_process::args_file::merge_configs(&init, &cli_config)
        } else {
            brush_process::args_file::merge_explicit_cli_fields(
                &init,
                &cli_config,
                &explicit_train_fields,
            )
        })
    }))
}

/// Initialize the backend, then drive `process` to completion on the CLI UI.
pub async fn run_headless(
    process: RunningProcess,
    train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    brush_process::burn_init_setup().await;
    run_cli_ui(process, train_stream_config).await
}

/// Run the CLI: pin the trainer stream to a dedicated [`Actor`] thread,
/// drive the indicatif UI on the main task.
pub async fn run_cli_ui(
    mut process: RunningProcess,
    mut train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    // Pump the trainer stream from a dedicated Actor thread; the
    // indicatif UI loop below consumes its output on the main task.
    let (tx, mut messages) = mpsc::unbounded_channel();
    let trainer = Actor::new("cli-trainer");
    trainer
        .run(move || async move {
            while let Some(msg) = process.stream.next().await {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        })
        .detach();

    // Hold the actor for the lifetime of the UI loop; dropping it
    // would kill the pump.
    let _trainer = trainer;

    // Initialize the logger with indicatif integration to prevent
    // progress bars from clobbering log output.
    let sp = {
        let mut builder = env_logger::builder();
        builder.target(env_logger::Target::Stdout);
        let logger = builder.build();
        let level = logger.filter();
        let multi = MultiProgress::new();

        LogWrapper::new(multi.clone(), logger)
            .try_init()
            .expect("Failed to initialize logger");
        log::set_max_level(level);

        multi
    };

    let main_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indacitif config")
            .tick_strings(&[
                "🖌️      ",
                "█🖌️     ",
                "▓█🖌️    ",
                "░▓█🖌️   ",
                "•░▓█🖌️  ",
                "·•░▓█🖌️ ",
                " ·•░▓🖌️ ",
                "  ·•░🖌️ ",
                "   ·•🖌️ ",
                "    ·🖌️ ",
                "     🖌️ ",
                "    🖌️ █",
                "   🖌️ █▓",
                "  🖌️ █▓░",
                " 🖌️ █▓░•",
                "🖌️ █▓░•·",
                "🖌️ ▓░•· ",
                "🖌️ ░•·  ",
                "🖌️ •·   ",
                "🖌️ ·    ",
                "🖌️      ",
            ]),
    );

    let stats_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indicatif config")
            .tick_strings(&["ℹ️", "ℹ️"]),
    );

    let train_progress = {
        let tc = &train_stream_config.train_config;
        let bar = ProgressBar::new(tc.total_iters() as u64)
        .with_style(
            ProgressStyle::with_template(
                "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({per_sec}, {eta} remaining)",
            )
            .expect("Invalid indicatif config").progress_chars("◍○○"),
        )
        .with_message("Steps");
        sp.add(bar)
    };

    let main_spinner = sp.add(main_spinner);
    main_spinner.enable_steady_tick(Duration::from_millis(120));

    let eval_spinner = sp.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .expect("Invalid indicatif config")
                .tick_strings(&["✅", "✅"]),
        ),
    );

    eval_spinner.set_message("waiting for dataset...");

    let stats_spinner = sp.add(stats_spinner);
    stats_spinner.set_message("Starting up");
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        let _ =
            sp.println("ℹ️  running in debug mode, compile with --release for best performance");
    }

    #[allow(unused_mut)]
    let mut duration = Duration::from_secs(0);

    while let Some(msg) = messages.recv().await {
        let _span = trace_span!("CLI UI").entered();

        let msg = match msg {
            Ok(msg) => msg,
            Err(error) => {
                // Don't print the error here. It'll bubble up and be printed as output.
                let _ = sp.println("❌ Encountered an error");
                return Err(error);
            }
        };

        match msg {
            ProcessMessage::NewProcess => {
                main_spinner.set_message("Starting process...");
            }
            ProcessMessage::StartLoading { name, training, .. } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    let _ = sp.println("❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)");
                    break;
                }
                main_spinner.set_message(format!("Loading {name}..."));
            }
            ProcessMessage::SplatsUpdated { .. } => {}
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainConfig { config } => {
                    train_progress.set_length(config.train_config.total_iters() as u64);
                    train_stream_config = *config;
                }
                TrainMessage::Dataset { dataset } => {
                    let train_views = dataset.train.views.len();
                    let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                    log::info!(
                        "Loaded dataset with {train_views} training, {eval_views} eval views",
                    );
                    main_spinner.set_message(format!(
                        "Loading dataset with {train_views} training, {eval_views} eval views",
                    ));
                    if eval_views > 0 {
                        eval_spinner.set_message(format!(
                            "evaluating {} views every {} steps",
                            eval_views, train_stream_config.process_config.eval_every,
                        ));
                    } else {
                        eval_spinner.finish_and_clear();
                    }
                }
                TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    lod_progress,
                    ..
                } => {
                    if let Some((lod, total_lods)) = lod_progress {
                        main_spinner.set_message(format!("LOD {lod}/{total_lods}"));
                    } else {
                        main_spinner.set_message("Training");
                    }
                    train_progress.set_position(iter as u64);
                    duration = total_elapsed;
                }
                TrainMessage::RefineStep {
                    cur_splat_count,
                    iter,
                    ..
                } => {
                    stats_spinner.set_message(format!("Current splat count {cur_splat_count}"));
                    log::info!("Refine iter {iter}, {cur_splat_count} splats.");
                }
                TrainMessage::EvalResult {
                    iter,
                    avg_psnr,
                    avg_ssim,
                } => {
                    log::info!("Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}");

                    eval_spinner.set_message(format!(
                        "Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}"
                    ));
                }
                TrainMessage::DoneTraining => {}
            },
            ProcessMessage::DoneLoading => {
                log::info!("Completed loading.");
                main_spinner.set_message("Completed loading");
                stats_spinner.set_message("Completed loading");
            }
            ProcessMessage::Warning { error } => {
                log::warn!("{error}");
                sp.println(format!("⚠️: {error}"))?;
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    let duration_secs = Duration::from_secs(duration.as_secs());
    let _ = sp.println(format!(
        "Training took {}",
        humantime::format_duration(duration_secs)
    ));

    log::info!(
        "Done training! Took {:?}.",
        humantime::format_duration(duration_secs)
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_explicit_default_valued_training_option() {
        let cli = Cli::try_parse_from_with_explicit_fields([
            "brush",
            "dataset",
            "--total-train-iters",
            "30000",
        ])
        .expect("CLI should parse");

        assert_eq!(
            cli.explicit_train_fields,
            vec!["total-train-iters".to_owned()]
        );
    }

    #[test]
    fn does_not_treat_top_level_options_as_training_overrides() {
        let cli = Cli::try_parse_from_with_explicit_fields(["brush", "dataset", "--with-viewer"])
            .expect("CLI should parse");

        assert!(cli.explicit_train_fields.is_empty());
    }

    #[test]
    fn parsed_explicit_default_overrides_saved_value_end_to_end() {
        let parsed = Cli::try_parse_from_with_explicit_fields([
            "brush",
            "dataset",
            "--total-train-iters",
            "30000",
        ])
        .expect("CLI should parse");
        let mut initial = TrainStreamConfig::default();
        initial.train_config.total_train_iters = 5000;

        let merged = brush_process::args_file::merge_explicit_cli_fields(
            &initial,
            &parsed.train_stream,
            &parsed.explicit_train_fields,
        );

        assert_eq!(merged.train_config.total_train_iters, 30_000);
    }

    #[test]
    fn parsed_explicit_false_disables_saved_boolean() {
        let parsed =
            Cli::try_parse_from_with_explicit_fields(["brush", "dataset", "--train-on-eval=false"])
                .expect("CLI should parse");
        let mut initial = TrainStreamConfig::default();
        initial.load_config.train_on_eval = true;

        let merged = brush_process::args_file::merge_explicit_cli_fields(
            &initial,
            &parsed.train_stream,
            &parsed.explicit_train_fields,
        );

        assert!(!merged.load_config.train_on_eval);
    }

    #[test]
    fn optional_boolean_does_not_consume_following_source() {
        let bare = Cli::try_parse_from_with_explicit_fields([
            "brush",
            "--train-on-eval",
            "dataset",
            "--with-viewer",
        ])
        .expect("bare boolean must leave the positional source untouched");
        assert!(bare.train_stream.load_config.train_on_eval);
        assert!(bare.source.is_some());

        let explicit_false = Cli::try_parse_from_with_explicit_fields([
            "brush",
            "--train-on-eval=false",
            "dataset",
            "--with-viewer",
        ])
        .expect("equals-form boolean must leave the positional source untouched");
        assert!(!explicit_false.train_stream.load_config.train_on_eval);
        assert!(explicit_false.source.is_some());
    }
}
