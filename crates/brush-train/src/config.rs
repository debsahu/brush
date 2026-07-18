use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_train_iters: u32,

    #[arg(long, help_heading = "Training options")]
    pub render_mode: Option<SplatRenderMode>,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-5")]
    pub lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-7")]
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

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    /// Weight of the optional additive LPIPS loss.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0.0",
        conflicts_with = "wd_r_gamma"
    )]
    pub lpips_loss_weight: f32,

    /// Global scale gamma for the WD-R perceptual objective. Zero disables WD-R.
    ///
    /// After the warm-up, RGB reconstruction becomes
    /// `gamma * (WD + (1 / 0.09) * original_rgb_loss)`. Alpha and appearance
    /// regularizers remain outside this scale.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0.0",
        value_parser = parse_non_negative_f32,
        conflicts_with = "lpips_loss_weight"
    )]
    #[serde(default)]
    pub wd_r_gamma: f32,

    /// Global training iteration at which WD-R replaces the warm-up RGB loss.
    #[arg(long, help_heading = "Training options", default_value = "3000")]
    #[serde(default = "default_wd_r_warmup_iters")]
    pub wd_r_warmup_iters: u32,

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
    #[arg(long, help_heading = "LOD options", default_value = "5000")]
    pub lod_refine_steps: u32,

    /// Percentage of gaussians to keep at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_decimation_keep: u32,

    /// Percentage to scale source images at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
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
        default_value = "false",
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
        default_value = "false",
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
    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }

    pub fn appearance_enabled(&self) -> bool {
        self.bilateral_grid || self.ppisp
    }

    pub fn wd_r_enabled_at(&self, global_iter: u32) -> bool {
        self.wd_r_gamma > 0.0 && global_iter >= self.wd_r_warmup_iters
    }

    /// Validate invariants that clap cannot enforce for serde, UI, or library callers.
    pub fn validate(&self) -> Result<(), String> {
        if !self.wd_r_gamma.is_finite() || self.wd_r_gamma < 0.0 {
            return Err("WD-R gamma must be finite and non-negative".to_owned());
        }
        if self.lpips_loss_weight > 0.0 && self.wd_r_gamma > 0.0 {
            return Err("LPIPS and WD-R cannot be enabled together".to_owned());
        }
        Ok(())
    }
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

fn default_wd_r_warmup_iters() -> u32 {
    3000
}

fn parse_non_negative_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|error| format!("invalid floating-point value: {error}"))?;
    if parsed.is_finite() && parsed >= 0.0 {
        Ok(parsed)
    } else {
        Err("value must be finite and non-negative".to_owned())
    }
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
    fn cli_rejects_lpips_with_wd_r() {
        let error = TrainConfig::try_parse_from([
            "brush",
            "--lpips-loss-weight",
            "0.1",
            "--wd-r-gamma",
            "0.028",
        ])
        .err()
        .expect("LPIPS and WD-R must not be stacked implicitly");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn wd_r_defaults_off_with_paper_warmup() {
        let config = TrainConfig::default();
        assert_eq!(config.wd_r_gamma, 0.0);
        assert_eq!(config.wd_r_warmup_iters, 3000);
    }

    #[test]
    fn wd_r_switches_on_at_global_warmup_boundary() {
        let config = TrainConfig {
            wd_r_gamma: 0.028,
            wd_r_warmup_iters: 3000,
            ..TrainConfig::default()
        };
        assert!(!config.wd_r_enabled_at(2999));
        assert!(config.wd_r_enabled_at(3000));
    }

    #[test]
    fn cli_rejects_invalid_wd_r_gamma() {
        for argument in ["--wd-r-gamma=-0.1", "--wd-r-gamma=NaN", "--wd-r-gamma=inf"] {
            let error = TrainConfig::try_parse_from(["brush", argument])
                .err()
                .expect("invalid WD-R gamma must be rejected");
            assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
        }
    }

    #[test]
    fn validation_rejects_non_cli_perceptual_conflicts() {
        let config = TrainConfig {
            lpips_loss_weight: 0.1,
            wd_r_gamma: 0.028,
            ..TrainConfig::default()
        };

        assert_eq!(
            config.validate().expect_err("stacked modes must fail"),
            "LPIPS and WD-R cannot be enabled together"
        );
    }
}
