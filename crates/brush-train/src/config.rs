use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "30000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub total_train_iters: u32,

    #[arg(long, help_heading = "Training options")]
    pub render_mode: Option<SplatRenderMode>,

    /// Start learning rate for the mean parameters.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "2e-5",
        value_parser = parse_learning_rate
    )]
    pub lr_mean: f64,

    /// End learning rate for the mean parameters.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "2e-7",
        value_parser = parse_learning_rate
    )]
    pub lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[arg(long, help_heading = "Training options", default_value = "50.0")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "10.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_rotation: f64,

    /// Max nr. of splats. This is only an upper bound, the actual final number of splats is NOT determined by this.
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(
        long,
        help_heading = "Refine options",
        default_value = "200",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[arg(long, help_heading = "Refine options", default_value = "0.0025")]
    pub growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[arg(long, help_heading = "Refine options", default_value = "0.25")]
    pub growth_select_fraction: f32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Split any splat whose max screen-space extent exceeds this fraction of
    /// the image dimension, shrinking the children so they land at (at most)
    /// this size on screen. 0 disables.
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
    pub split_at_screen_size: f32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Factor of the opacity decay.
    #[arg(long, help_heading = "Training options", default_value = "0.004")]
    pub opac_decay: f32,

    /// Factor of the per-refine scale decay (MRNF port, delta #1). Mirrors
    /// opacity decay but shrinks the log-scales: `scale *= 1 - scale_decay *
    /// t_shrink`, strongest early in a phase and fading to zero at its end.
    /// 0 disables (matching upstream Brush behaviour).
    #[arg(long, help_heading = "Refine options", default_value = "0.002")]
    pub scale_decay: f32,

    /// Prune genuinely degenerate splats whose smallest scale axis falls below
    /// `1e-10` (MRNF delta #3). OFF by default: Brush deliberately keeps thin
    /// "pancake" surface splats, and the Mip-Splatting min-scale floor already
    /// keeps rendered scales above this, so this only bites raw-degenerate
    /// splats. Enable only if an A/B shows it helps without softening surfaces.
    #[arg(long, help_heading = "Refine options", default_value = "false")]
    pub min_scale_prune: bool,

    /// Smallest-scale-axis threshold for the optional min-scale degenerate
    /// prune (only used when `--min-scale-prune` is set). Matches MRNF's
    /// `MRNF_LOG_MIN_SCALE_THRESHOLD = log(1e-10)` (mrnf.cpp:72), expressed here
    /// as the linear scale so it compares against the effective (floored)
    /// scales.
    #[arg(long, help_heading = "Refine options", default_value = "1e-10")]
    pub min_scale_prune_threshold: f32,

    /// Opacity below which a splat is pruned, and the clamp applied to split
    /// children's opacity. Mirrors MRNF's `min_opacity = 1/255`
    /// (parameters.cpp:249, prune threshold `logit(1/255)` at mrnf.cpp:71).
    #[arg(long, help_heading = "Refine options", default_value_t = 1.0f32 / 255.0)]
    pub min_opacity: f32,

    /// World-extent multiplier for the out-of-bounds prune: a splat is culled if
    /// any scale axis, or its distance from the robust scene center, exceeds this
    /// factor times the scene's largest robust half-extent. Mirrors MRNF's
    /// `max_allowed = max_extent * 100` (mrnf.cpp:644). This is the sky-floater
    /// killer; lower it to cull closer to the scene box.
    #[arg(long, help_heading = "Refine options", default_value = "100.0")]
    pub prune_extent_factor: f32,

    /// Percentile for the robust per-axis AABB recomputed each refine (drives the
    /// out-of-bounds prune). Mirrors MRNF's `bounds_percentile = 0.8`
    /// (parameters.hpp:182). Note: this governs the per-refine bounds recompute;
    /// the one-time initial bounds use the module default.
    #[arg(long, help_heading = "Refine options", default_value = "0.8")]
    pub bounds_percentile: f32,

    /// Long-Axis-Split (LAS) longest-axis factor (MRNF delta #2): the split
    /// halves the longest scale axis and offsets the two children apart by this
    /// fraction of its world extent. Mirrors MRNF's fixed `0.5`
    /// (densification_kernels.cu:669-771). For oversized splats the effective
    /// longest-axis shrink is further capped by `--split-at-screen-size`.
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
    pub split_long_axis_scale: f32,

    /// Long-Axis-Split (LAS) shrink applied to the two non-longest scale axes of
    /// both split children. Mirrors MRNF's fixed `0.85`
    /// (densification_kernels.cu:669-771).
    #[arg(long, help_heading = "Refine options", default_value = "0.85")]
    pub split_other_axis_scale: f32,

    /// Long-Axis-Split (LAS) opacity multiplier applied to both split children:
    /// `sigmoid(raw) *= split_opacity_scale`. Mirrors MRNF's revised-opacity
    /// `inverse_sigmoid(sigmoid(opacity) * 0.6)` (densification_kernels.cu:722).
    /// NOT mass-conserving; set to 1.0 for a mass-conserving-ish split A/B.
    #[arg(long, help_heading = "Refine options", default_value = "0.6")]
    pub split_opacity_scale: f32,

    /// Edge-guidance densification (MRNF port, delta #4). When set, a Canny edge
    /// map of each sampled GT view is projected onto the gaussians and the
    /// accumulated per-gaussian edge score biases growth + dead-slot replacement
    /// toward high-frequency image edges (LFS `use_edge_map`, mrnf_defaults). OFF
    /// by default; this is the highest-effort MRNF lever, only worth enabling if
    /// Phases 1-3 leave a floater/detail gap.
    ///
    /// NOTE: the current implementation is a burn-op projection fallback
    /// (pinhole-only, center-sample, opacity-weighted), NOT the full alpha-blended
    /// edge rasterizer — see `crate::edge`.
    #[arg(long, help_heading = "Refine options", default_value = "false")]
    #[serde(default)]
    pub use_edge_map: bool,

    /// Strength of the edge-guidance factor: the normalized per-gaussian edge
    /// score is scaled by this before the `+ 1.0` that turns it into a
    /// multiplicative sampling weight (LFS `MRNF_EDGE_SCORE_WEIGHT = 0.25`,
    /// mrnf.cpp:68). Only used when `--use-edge-map` is set.
    #[arg(long, help_heading = "Refine options", default_value = "0.25")]
    #[serde(default = "default_edge_score_weight")]
    pub edge_score_weight: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,

    /// Enable `DiG` DINO feature training. Requires per-view feature maps
    /// extracted with `scripts/extract_dino_features.py` (see
    /// `--features-dir-name`).
    #[arg(long, help_heading = "Training options", default_value = "false")]
    pub dino: bool,

    /// Weight of the `DiG` DINO feature MSE loss.
    #[arg(long, help_heading = "Training options", default_value = "1.0")]
    pub dino_loss_weight: f32,

    /// Per-gaussian stored feature dimension for `DiG` training.
    #[arg(long, help_heading = "Training options", default_value = "64")]
    pub dino_feature_dim: u32,

    /// Upscale of the rendered feature image vs. the GT feature-map
    /// resolution (the reference's `dino_rescale_factor`).
    #[arg(long, help_heading = "Training options", default_value = "5")]
    pub dino_rescale_factor: u32,

    /// Start learning rate for the `DiG` features and decoder MLP.
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    pub dino_lr: f64,

    /// Final learning rate for the `DiG` features and decoder MLP
    /// (exponential decay over 6000 steps, then held).
    #[arg(long, help_heading = "Training options", default_value = "1e-3")]
    pub dino_lr_end: f64,

    /// Weight of the 3-nearest-neighbor feature-variance regularizer
    /// (enabled after step 1000; 0 disables).
    #[arg(long, help_heading = "Training options", default_value = "0.01")]
    pub dino_nn_reg_weight: f32,

    /// Weight of l1 loss on depth (disparity-space)
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    pub depth_loss_weight: f32,

    /// Base background color (R,G,B) used during training.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0,0,0",
        value_delimiter = ',',
        num_args = 3
    )]
    pub background_color: Vec<f32>,

    /// Strength of random noise added to the background color each step.
    /// Noise is uniform in [-strength, +strength], clamped to [0, 1].
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub background_noise_strength: f32,

    /// Number of LOD levels to generate after initial training (0 = disabled).
    #[arg(long, help_heading = "LOD options", default_value = "0")]
    pub lod_levels: u32,

    /// Number of refinement training steps per LOD level.
    #[arg(
        long,
        help_heading = "LOD options",
        default_value = "5000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub lod_refine_steps: u32,

    /// Percentage of gaussians to keep at each LOD level (1-100).
    #[arg(
        long,
        help_heading = "LOD options",
        default_value = "50",
        value_parser = clap::value_parser!(u32).range(1..=100)
    )]
    pub lod_decimation_keep: u32,

    /// Percentage to scale source images at each LOD level (1-100).
    #[arg(
        long,
        help_heading = "LOD options",
        default_value = "50",
        value_parser = clap::value_parser!(u32).range(1..=100)
    )]
    pub lod_image_scale: u32,

    /// Scene scale used for random splat initialization.
    /// When no init is provided, splats are randomly placed
    /// inside camera frustums up to this depth. By default this is
    /// estimated from the camera spacing (with a 1m minimum).
    #[arg(long, help_heading = "Training options")]
    pub random_init_scene_scale: Option<f32>,

    /// Enable per-view affine bilateral grids (BilaRF-style). Mutually exclusive
    /// with PPISP.
    #[arg(
        long,
        help_heading = "Appearance options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = false,
        conflicts_with = "ppisp"
    )]
    #[serde(default)]
    pub bilateral_grid: bool,

    /// Bilateral grid dimensions as `x,y,guidance`.
    #[arg(
        long,
        help_heading = "Appearance options",
        default_value = "16,16,8",
        value_delimiter = ',',
        num_args = 3,
        value_parser = clap::value_parser!(u32).range(2..)
    )]
    #[serde(default = "default_bilagrid_dims")]
    pub bilagrid_dims: Vec<u32>,

    /// Weight of the bilateral grid's total-variation regularizer.
    #[arg(long, help_heading = "Appearance options", default_value = "10.0")]
    #[serde(default = "default_bilagrid_tv_weight")]
    pub bilagrid_tv_weight: f32,

    /// Learning rate for the bilateral grids.
    #[arg(long, help_heading = "Appearance options", default_value = "2e-3")]
    #[serde(default = "default_bilagrid_lr")]
    pub bilagrid_lr: f64,

    /// Adam betas for the per-view grid updates as `b1,b2`. The sparse
    /// updates are dense-Adam equivalent (moments decay over the gap
    /// between a view's visits), so the horizons are in global steps and
    /// the defaults match the reference implementations.
    #[arg(
        long,
        help_heading = "Appearance options",
        default_value = "0.9,0.999",
        value_delimiter = ',',
        num_args = 2
    )]
    #[serde(default = "default_bilagrid_betas")]
    pub bilagrid_betas: Vec<f64>,

    /// Enable PPISP appearance compensation: per-frame exposure + color
    /// homography and per-camera vignetting + tone curve (physically
    /// plausible ISP model), applied to the render before the loss. Mutually
    /// exclusive with the bilateral grid.
    #[arg(
        long,
        help_heading = "Appearance options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = false,
        conflicts_with = "bilateral_grid"
    )]
    #[serde(default)]
    pub ppisp: bool,

    /// Learning rate for the PPISP parameters.
    #[arg(long, help_heading = "Appearance options", default_value = "2e-3")]
    #[serde(default = "default_ppisp_lr")]
    pub ppisp_lr: f64,

    /// Scale on all PPISP parameter-regularization terms.
    #[arg(long, help_heading = "Appearance options", default_value = "1.0")]
    #[serde(default = "default_ppisp_reg_scale")]
    pub ppisp_reg_scale: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

impl TrainConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.total_train_iters == 0 {
            return Err("total-train-iters must be greater than zero".to_owned());
        }
        for (name, value) in [("lr-mean", self.lr_mean), ("lr-mean-end", self.lr_mean_end)] {
            if !valid_learning_rate(value) {
                return Err(format!("{name} must be finite and in the range (0, 1]"));
            }
        }
        if self.lr_mean_end > self.lr_mean {
            return Err("lr-mean-end must not exceed lr-mean".to_owned());
        }
        if self
            .lod_levels
            .checked_mul(self.lod_refine_steps)
            .and_then(|lod_iters| self.total_train_iters.checked_add(lod_iters))
            .is_none()
        {
            return Err("total training and LOD iterations exceed u32::MAX".to_owned());
        }
        Ok(())
    }

    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }

    pub fn appearance_enabled(&self) -> bool {
        self.bilateral_grid || self.ppisp
    }
}

fn parse_learning_rate(value: &str) -> Result<f64, String> {
    let value = value.parse::<f64>().map_err(|error| error.to_string())?;
    if valid_learning_rate(value) {
        Ok(value)
    } else {
        Err("learning rate must be finite and in the range (0, 1]".to_owned())
    }
}

fn valid_learning_rate(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value <= 1.0
}

fn default_bilagrid_dims() -> Vec<u32> {
    vec![16, 16, 8]
}

fn default_bilagrid_tv_weight() -> f32 {
    10.0
}

fn default_bilagrid_lr() -> f64 {
    2e-3
}

fn default_bilagrid_betas() -> Vec<f64> {
    vec![0.9, 0.999]
}

fn default_ppisp_lr() -> f64 {
    2e-3
}

fn default_ppisp_reg_scale() -> f32 {
    1.0
}

fn default_edge_score_weight() -> f32 {
    0.25
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_rejects_stacked_appearance_models() {
        let error = TrainConfig::try_parse_from(["brush", "--bilateral-grid", "--ppisp"])
            .err()
            .expect("stacked appearance flags must conflict");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn cli_rejects_invalid_lod_ranges() {
        for args in [
            ["brush", "--lod-refine-steps", "0"],
            ["brush", "--lod-decimation-keep", "0"],
            ["brush", "--lod-decimation-keep", "101"],
            ["brush", "--lod-image-scale", "0"],
            ["brush", "--lod-image-scale", "101"],
        ] {
            assert!(
                TrainConfig::try_parse_from(args).is_err(),
                "accepted invalid LOD option: {args:?}"
            );
        }
    }

    #[test]
    fn cli_rejects_invalid_mean_schedule_values() {
        for args in [
            ["brush", "--total-train-iters", "0"],
            ["brush", "--lr-mean", "0"],
            ["brush", "--lr-mean", "2"],
            ["brush", "--lr-mean-end", "0"],
        ] {
            assert!(
                TrainConfig::try_parse_from(args).is_err(),
                "accepted invalid mean schedule option: {args:?}"
            );
        }
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn validation_rejects_invalid_programmatic_mean_schedule() {
        let mut config = TrainConfig::default();
        config.total_train_iters = 0;
        assert!(config.validate().is_err());

        config.total_train_iters = 1;
        config.lr_mean_end = config.lr_mean * 2.0;
        assert!(config.validate().is_err());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn validation_rejects_total_iteration_overflow() {
        let mut config = TrainConfig::default();
        config.total_train_iters = u32::MAX;
        config.lod_levels = 1;
        config.lod_refine_steps = 1;

        assert_eq!(
            config.validate(),
            Err("total training and LOD iterations exceed u32::MAX".to_owned())
        );
    }
}
