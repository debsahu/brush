use clap::{Args, Parser};
use serde::{Deserialize, Serialize};

#[derive(Clone, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ProcessConfig {
    /// Random seed.
    #[arg(long, help_heading = "Process options", default_value = "42")]
    pub seed: u64,
    /// Iteration to resume from
    #[arg(long, help_heading = "Process options", default_value = "0")]
    pub start_iter: u32,
    /// Eval every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "1000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub eval_every: u32,
    /// Save the rendered eval images to disk. Uses export-path for the file location.
    #[arg(
        long,
        help_heading = "Process options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = false
    )]
    pub eval_save_to_disk: bool,
    /// Export every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "5000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub export_every: u32,
    /// Location to put exported files. Supports {dataset} interpolation for the dataset
    /// folder name. Path is relative to the dataset's parent directory (or CWD if unavailable).
    /// Use "./{dataset}/" to export inside the dataset folder.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "./{dataset}_exports/"
    )]
    pub export_path: String,
    /// Filename of exported ply file
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "export_{iter}.ply"
    )]
    pub export_name: String,
}

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainStreamConfig {
    #[clap(flatten)]
    #[serde(flatten)]
    pub train_config: brush_train::config::TrainConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub model_config: brush_dataset::config::ModelConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub load_config: brush_dataset::config::LoadDatasetConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub process_config: ProcessConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub rerun_config: brush_rerun::RerunConfig,
}

impl Default for TrainStreamConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

impl TrainStreamConfig {
    /// Validate values that can also be supplied programmatically by the GUI
    /// and C API, bypassing Clap's value parsers.
    pub fn validate(&self) -> Result<(), String> {
        self.load_config.validate()?;
        self.train_config.validate()?;

        for (name, value) in [
            ("refine-every", self.train_config.refine_every),
            ("lod-refine-steps", self.train_config.lod_refine_steps),
            ("eval-every", self.process_config.eval_every),
            ("export-every", self.process_config.export_every),
            (
                "rerun-log-train-stats-every",
                self.rerun_config.rerun_log_train_stats_every,
            ),
            (
                "rerun-log-distribution-every",
                self.rerun_config.rerun_log_distribution_every,
            ),
        ] {
            if value == 0 {
                return Err(format!("{name} must be greater than zero"));
            }
        }
        if self.rerun_config.rerun_log_splats_every == Some(0) {
            return Err("rerun-log-splats-every must be greater than zero".to_owned());
        }

        for (name, value) in [
            ("lod-decimation-keep", self.train_config.lod_decimation_keep),
            ("lod-image-scale", self.train_config.lod_image_scale),
        ] {
            if !(1..=100).contains(&value) {
                return Err(format!("{name} must be between 1 and 100"));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn rejects_programmatic_zero_cadences() {
        let mut config = TrainStreamConfig::default();
        config.process_config.eval_every = 0;
        assert!(config.validate().is_err());

        config.process_config.eval_every = 1;
        config.train_config.lod_refine_steps = 0;
        assert!(config.validate().is_err());

        config.train_config.lod_refine_steps = 1;
        config.rerun_config.rerun_log_splats_every = Some(0);
        assert!(config.validate().is_err());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn rejects_programmatic_invalid_mean_schedule() {
        let mut config = TrainStreamConfig::default();
        config.train_config.total_train_iters = 0;
        assert!(config.validate().is_err());

        config.train_config.total_train_iters = 1;
        config.train_config.lr_mean_end = config.train_config.lr_mean * 2.0;
        assert!(config.validate().is_err());
    }
}
