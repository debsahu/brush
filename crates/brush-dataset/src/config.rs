use brush_render::AlphaMode;
use clap::Args;
use serde::{Deserialize, Serialize};

/// Default Cache budget for packed scene batches. 6 GB on native; less on
/// wasm since the whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "6GiB";
#[cfg(target_family = "wasm")]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "2GiB";

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ModelConfig {
    /// SH degree of splats.
    #[arg(
        long,
        help_heading = "Model Options",
        default_value = "3",
        value_parser = clap::value_parser!(u32).range(0..=4)
    )]
    pub sh_degree: u32,
}

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct LoadDatasetConfig {
    /// Max nr. of frames of dataset to load
    #[arg(
        long,
        help_heading = "Dataset Options",
        value_parser = parse_positive_usize
    )]
    pub max_frames: Option<usize>,
    /// Max resolution of images to load.
    #[arg(long, help_heading = "Dataset Options", default_value = "1920")]
    pub max_resolution: u32,
    /// Create an eval dataset by selecting every nth image
    #[arg(
        long,
        help_heading = "Dataset Options",
        value_parser = parse_positive_usize
    )]
    pub eval_split_every: Option<usize>,
    /// Keep the eval views in the training set instead of holding them out.
    /// Useful with appearance compensation, where held-out views have no
    /// learned per-view corrections and eval scores mostly measure the
    /// splat <-> average-appearance drift.
    #[arg(
        long,
        help_heading = "Dataset Options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = false
    )]
    #[serde(default)]
    pub train_on_eval: bool,
    /// Load only every nth frame
    #[arg(
        long,
        help_heading = "Dataset Options",
        value_parser = parse_positive_u32
    )]
    pub subsample_frames: Option<u32>,
    /// Load only every nth point from the initial sfm data
    #[arg(
        long,
        help_heading = "Dataset Options",
        value_parser = parse_positive_u32
    )]
    pub subsample_points: Option<u32>,
    /// Whether to interpret an alpha channel (or masks) as transparency or masking.
    #[arg(long, help_heading = "Dataset Options")]
    pub alpha_mode: Option<AlphaMode>,
    /// Max size of the cache for frames of the dataset, larger values usually improve performance for large datasets at the cost of more memory usage, can be e.g. 6G, 6000M, 6000MiB, 6000MB
    #[arg(long, help_heading = "Dataset Options", default_value = DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE, value_parser = parse_size)]
    pub max_scene_batch_cache_size: u64,
    /// Name of the folder containing per-view feature maps (`<image_stem>.npy`)
    #[arg(
        long,
        help_heading = "Dataset Options",
        default_value = "dino_features"
    )]
    pub features_dir_name: String,
}

impl LoadDatasetConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_resolution == 0 {
            return Err("max-resolution must be greater than zero".to_owned());
        }
        if self.max_frames == Some(0) {
            return Err("max-frames must be greater than zero".to_owned());
        }
        match self.eval_split_every {
            Some(0) => return Err("eval-split-every must be greater than zero".to_owned()),
            Some(1) if !self.train_on_eval => {
                return Err(
                    "eval-split-every must be at least 2 unless train-on-eval is enabled"
                        .to_owned(),
                );
            }
            _ => {}
        }
        for (name, value) in [
            ("subsample-frames", self.subsample_frames.map(u64::from)),
            ("subsample-points", self.subsample_points.map(u64::from)),
        ] {
            if value == Some(0) {
                return Err(format!("{name} must be greater than zero"));
            }
        }
        Ok(())
    }
}

fn parse_size(s: &str) -> Result<u64, parse_size::Error> {
    parse_size::parse_size(s)
}

fn parse_positive_usize(s: &str) -> Result<usize, String> {
    s.parse::<std::num::NonZeroUsize>()
        .map(Into::into)
        .map_err(|error| error.to_string())
}

fn parse_positive_u32(s: &str) -> Result<u32, String> {
    s.parse::<std::num::NonZeroU32>()
        .map(Into::into)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[derive(Debug, Parser)]
    struct TestCli {
        #[command(flatten)]
        dataset: LoadDatasetConfig,
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn rejects_zero_dataset_cadences() {
        for option in ["--max-frames", "--subsample-frames", "--subsample-points"] {
            let result = TestCli::try_parse_from(["test", option, "0"]);
            assert!(result.is_err(), "{option} accepted zero");
        }
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn rejects_eval_split_that_would_empty_training_set() {
        let result = TestCli::try_parse_from(["test", "--eval-split-every", "0"]);
        assert!(result.is_err(), "zero eval split interval was accepted");

        let config = TestCli::parse_from(["test", "--eval-split-every", "1"]);
        assert!(config.dataset.validate().is_err());
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn accepts_eval_every_view_when_eval_views_remain_in_training() {
        let config =
            TestCli::try_parse_from(["test", "--eval-split-every", "1", "--train-on-eval"])
                .expect("all-view eval should parse when eval views remain in training");

        assert!(config.dataset.validate().is_ok());
        assert_eq!(config.dataset.eval_split_every, Some(1));
        assert!(config.dataset.train_on_eval);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn accepts_positive_dataset_cadences() {
        let cli = TestCli::try_parse_from([
            "test",
            "--eval-split-every",
            "2",
            "--subsample-frames",
            "2",
            "--subsample-points",
            "3",
        ])
        .expect("positive cadence values should parse");

        assert_eq!(cli.dataset.eval_split_every, Some(2));
        assert_eq!(cli.dataset.subsample_frames, Some(2));
        assert_eq!(cli.dataset.subsample_points, Some(3));
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn rejects_programmatic_zero_dataset_values() {
        let mut config = TestCli::parse_from(["test"]).dataset;
        config.subsample_frames = Some(0);
        assert!(config.validate().is_err());

        config.subsample_frames = None;
        config.max_resolution = 0;
        assert!(config.validate().is_err());

        config.max_resolution = 1;
        config.eval_split_every = Some(0);
        assert!(config.validate().is_err());

        config.eval_split_every = Some(1);
        assert!(config.validate().is_err());

        config.train_on_eval = true;
        assert!(config.validate().is_ok());

        config.eval_split_every = None;
        config.max_frames = Some(0);
        assert!(config.validate().is_err());
    }
}
