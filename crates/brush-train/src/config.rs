// Re-exported so `brush_train::config::DepthLossSpace` resolves for CLI/GUI
// consumers, the same way `DepthSource` and `DepthWeightDecay` below do. The
// enum itself lives beside the loss it selects (`brush-loss`).
pub use brush_loss::{DepthLossSpace, DepthUncovered};
use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

/// Decay shape for the depth-loss annealing schedule (`--depth-weight-decay`).
#[derive(Default, Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DepthWeightDecay {
    #[default]
    Linear,
    Cosine,
}

/// Where the trainer's supervised depth comes from (`--depth-source`).
///
/// PGSR (Chen et al. 2024, arXiv:2406.06521) observes that the alpha-composited
/// camera-z of splat MEANS is a centre-biased surface estimate: a depth loss
/// against it constrains only where along the ray a gaussian sits, never its
/// orientation, so gaussians satisfy it while forming a thick shell of tilted
/// ellipsoids around the true surface. The plane sources composite per-splat
/// tangent-plane parameters instead and intersect the pixel ray with the
/// composited plane, which is unbiased.
///
/// The three variants have deliberately DIFFERENT backward semantics; see the
/// contract table at the dispatch site in `train.rs` before reading an ablation
/// result. `Center` is the default and is byte-identical to the pre-PGSR
/// trainer.
#[derive(Default, Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DepthSource {
    /// Alpha-composited camera-z of splat means (previous behaviour).
    #[default]
    Center,
    /// PGSR plane-intersection depth via the auxiliary feature pass (approach A).
    /// Geometry gradients arrive exclusively through feature VALUES; the
    /// compositing weights are constants, so depth error cannot reach opacity.
    PlaneAux,
    /// **EXPERIMENTAL — measured HARMFUL in this trainer; do not use for delivery.**
    ///
    /// PGSR plane-intersection depth via the main rasterize kernel (approach B).
    /// Blending-weight gradients are LIVE for the plane channels, so depth error
    /// does reach opacity — by design, and the one thing `PlaneAux` cannot
    /// express.
    ///
    /// Measured on two scenes (`docs/superpowers/specs/2026-08-20-pgsr-ablation-synthesis.md`
    /// §3.3): opacity p50 falls 28% on `ARKitScenes` 48018538 (0.0722 → 0.0519)
    /// and 34% on `playroom_0812` (0.2118 → 0.1395), monotonically and with none
    /// of the cap-bound recovery every other arm shows; playroom lands at
    /// **23.918 dB, under our 24 dB delivery gate**. The apparent mechanism is
    /// "fade rather than rotate": with the alpha VJP open, fading a splat out is
    /// a cheaper descent direction for the plane-depth term than rotating it.
    ///
    /// **The technique itself is NOT known to be broken.** The `gauss-surf`
    /// reference trainer (Pablo Vela, Apache-2.0) carries blending-weight
    /// gradients on its default path too, and when run on our priors, our seed
    /// and our cameras it holds opacity p50 at **0.9934** — so the collapse is a
    /// property of THIS trainer, not of the formulation
    /// (`work/arkitscenes_48018538/reference/README.md` §8). Cause under
    /// investigation; the leading suspects are our disparity-space depth loss
    /// (a 1/d² gradient makes near-camera fading cheap in a way a metric-L1 loss
    /// does not) and our densifier. Kept selectable precisely so those
    /// experiments can run.
    PlaneFused,
}

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

    /// MRNF-gated noise injection (MRNF port, R2). Like the generic Brush
    /// per-step noise, LFS injects mean-noise EVERY training step from
    /// `post_backward` (mrnf.cpp:617); the difference this flag makes is the
    /// GATING, not the frequency. When set, the low-opacity mean-noise
    /// perturbation is gated on VALID robust bounds (LFS `_bounds_valid`) and
    /// on the ACCUMULATED per-refine-window visibility count
    /// (`RefineRecord::vis_weight`, LFS `_vis_count > 0`) instead of the
    /// single-step `visible` mask, mirroring LFS `MRNF::inject_noise` /
    /// `launch_mrnf_noise_injection` (mrnf.cpp:1085, `mrnf_kernels.cu:41`).
    /// Replaces Brush's generic per-step noise. ON by default (LFS
    /// `mrnf_defaults` parity); disable per-run with
    /// `--mrnf-noise-injection=false`.
    #[arg(
        long,
        help_heading = "Refine options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = true
    )]
    #[serde(default = "default_true")]
    pub mrnf_noise_injection: bool,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "10.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Start learning rate for the scale parameters. Default 7e-3 (LFS
    /// `scaling_lr` in `mrnf_defaults`); together with the 5e-3 `lr_scale_end`
    /// this activates the LFS 7e-3 -> 5e-3 scale-LR exponential decay by
    /// default. Set `--lr-scale` == `--lr-scale-end` to disable the schedule.
    #[arg(long, help_heading = "Training options", default_value = "7e-3")]
    pub lr_scale: f64,

    /// End learning rate for the scale parameters (MRNF LR schedule, R1).
    /// Independent exponential decay `lr_scale` -> `lr_scale_end` over
    /// `total_train_iters`, mirroring LFS `scaling_lr_end` +
    /// `compute_decay_gamma` (mrnf.cpp:425) and the per-step
    /// `_scale_lr_current *= _scale_lr_gamma` (mrnf.cpp:1360). Default 5e-3
    /// (LFS `scaling_lr_end`); with the 7e-3 `lr_scale` start the LFS scale-LR
    /// decay is ON by default. Set equal to `lr_scale` to make it a no-op.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale_end: f64,

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
    /// Increase this to make splats grow more aggressively. Default 0.07 (LFS
    /// `grow_fraction` in `mrnf_defaults`); comparable now that the error-map
    /// growth signal is default-on.
    #[arg(long, help_heading = "Refine options", default_value = "0.07")]
    pub growth_select_fraction: f32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Iteration after which the refine step is skipped ENTIRELY: no prune, no
    /// dead-splat replacement, no oversized force-split. Topology is frozen and
    /// the remaining iterations are pure photometric optimisation of a fixed
    /// splat set. Mirrors LFS MRNF's `stop_refine` (mrnf.cpp `is_refining()`
    /// requires `iter < stop_refine`, default 28500), which our port previously
    /// had no equivalent for: `--growth-stop-iter` only stops NET growth, while
    /// prune + 1:1 multinomial replacement keep churning the population every
    /// `--refine-every` steps. At a saturated `--max-splats` cap that churn has
    /// no sink for elongated splats, so spindle fraction rises monotonically.
    /// 0 disables (previous behaviour). Set equal to `--growth-stop-iter` for
    /// LFS-like semantics.
    ///
    /// MEASURED OUTCOME (0726hickorywood, 60k, 5M cap, SH2, three arms identical
    /// but for this flag): this flag LOST to leaving refine alone. Control
    /// PSNR 14.377 / SSIM 0.7884 / needle frac 0.1400; with this flag at 30k,
    /// PSNR 13.978 / SSIM 0.7823 / needle frac 0.1618. It raised median opacity
    /// 0.21 -> 0.23, but opacity moved OPPOSITE to PSNR/SSIM, so do not treat it
    /// as a quality proxy. Kept for LFS parity and further study; NOT recommended.
    #[arg(long, help_heading = "Refine options", default_value = "0")]
    pub stop_refine_iter: u32,

    /// Iteration after which pruned splats are NO LONGER backfilled by
    /// multinomial replacement, while prune itself keeps running. Growth is
    /// governed separately by `--growth-stop-iter`.
    ///
    /// Motivation (measured on 0726hickorywood, 2026-08-05): the two halves of
    /// refine pull in opposite directions once the `--max-splats` cap is
    /// saturated. Replacement re-splits to hold the count, and MRNF's LAS split
    /// gives BOTH children 0.6x the parent opacity, so the population drifts
    /// translucent (median opacity fell to 0.08 on a 10M-cap run). Prune, on the
    /// other hand, is the only sink for over-stretched splats, via its
    /// `scale_max > extent * --prune-extent-factor` term. Freezing all of refine
    /// with `--stop-refine-iter` therefore fixes the opacity dilution but
    /// REMOVES the spindle sink: measured elongation slope rose from
    /// +0.0029/1k iters to +0.0047/1k after the freeze.
    ///
    /// This flag separates them: keep the prune sink, drop the dilution. Splat
    /// count decays gently from the cap instead of churning at it.
    /// 0 disables (previous behaviour).
    ///
    /// MEASURED OUTCOME (same three-arm test): also LOST to the control --
    /// PSNR 13.649 / SSIM 0.7779 / needle frac 0.1606, the worst PSNR of the
    /// three. The premise that prune is the needle sink did not hold: needle
    /// fraction was ~equal to the full-freeze arm (0.1606 vs 0.1618) despite
    /// prune running throughout. Its one real effect is a 21% smaller asset
    /// (3.94M vs 5.00M splats) at ~equal opacity, so it is a size lever, not a
    /// quality lever. NOT recommended for quality.
    #[arg(long, help_heading = "Refine options", default_value = "0")]
    pub stop_replace_iter: u32,

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
    /// Default 0.002 (LFS `scale_decay` in `mrnf_defaults`); pass
    /// `--scale-decay=0` to disable (upstream Brush behaviour).
    #[arg(long, help_heading = "Refine options", default_value = "0.002")]
    pub scale_decay: f32,

    /// Prune genuinely degenerate splats whose smallest scale axis falls below
    /// `1e-10` (MRNF delta #3). ON by default (LFS `min_scale_prune` in
    /// `mrnf_defaults`): the Mip-Splatting min-scale floor already keeps
    /// rendered scales above this, so this only bites raw-degenerate splats.
    /// Disable per-run with `--min-scale-prune=false` (e.g. to keep thin
    /// "pancake" surface splats for an A/B).
    #[arg(
        long,
        help_heading = "Refine options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = true
    )]
    #[serde(default = "default_true")]
    pub min_scale_prune: bool,

    /// Smallest-scale-axis threshold for the optional min-scale degenerate
    /// prune (only used when `--min-scale-prune` is set). Matches MRNF's
    /// `MRNF_LOG_MIN_SCALE_THRESHOLD = log(1e-10)` (mrnf.cpp:72), expressed here
    /// as the linear scale so it compares against the effective (floored)
    /// scales.
    #[arg(long, help_heading = "Refine options", default_value = "1e-10")]
    pub min_scale_prune_threshold: f32,

    /// Prune splats whose raw quaternion has collapsed toward zero (squared
    /// norm < 1e-8), i.e. a degenerate rotation that renders as garbage.
    /// Mirrors MRNF's `compute_near_zero_rotation_mask` (mrnf.cpp:667;
    /// `pruning_kernels.cu:64` `mag_sq = q.q < 1e-8`). ON by default (LFS
    /// `near_zero_rotation_prune` in `mrnf_defaults`): a healthy quaternion has
    /// norm ~1 so this only bites already-collapsed splats. Disable per-run with
    /// `--near-zero-rotation-prune=false`.
    #[arg(
        long,
        help_heading = "Refine options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = true
    )]
    #[serde(default = "default_true")]
    pub near_zero_rotation_prune: bool,

    /// Use an L2 radial distance from the robust scene center for the
    /// out-of-bounds prune instead of the per-axis (L-inf / Chebyshev) test.
    /// NOTE: MRNF is NOT radial — its out-of-bounds cull is L-inf:
    /// `dist_from_center = (means - center).abs().max(1)` then
    /// `dist_from_center > max_allowed` (mrnf.cpp:663-669). So Brush's DEFAULT
    /// per-axis test already matches MRNF; this flag is a STRICTER divergence
    /// experiment, not MRNF parity. OFF by default: L2 >= L-inf so this prunes
    /// a superset of the per-axis (MRNF) test, changing default behaviour,
    /// hence flag-gated.
    #[arg(long, help_heading = "Refine options", default_value = "false")]
    pub radial_bounds_prune: bool,

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
    /// `inverse_sigmoid(sigmoid(opacity) * 0.6)` (`densification_kernels.cu:722`).
    /// NOT mass-conserving; set to 1.0 for a mass-conserving-ish split A/B.
    #[arg(long, help_heading = "Refine options", default_value = "0.6")]
    pub split_opacity_scale: f32,

    /// Edge-guidance densification (MRNF port, delta #4). When set, a Canny edge
    /// map of each sampled GT view is projected onto the gaussians and the
    /// accumulated per-gaussian edge score biases growth + dead-slot replacement
    /// toward high-frequency image edges (LFS `use_edge_map`, `mrnf_defaults`). ON
    /// by default (LFS `mrnf_defaults` parity); disable per-run with
    /// `--use-edge-map=false` (it is the highest-effort MRNF lever).
    ///
    /// The per-gaussian score is the alpha-blended `Σ_p T·α·edge` (LFS parity),
    /// computed by the `feat_dim=1` feature backward — see `crate::edge`. Works for
    /// every camera model the renderer supports (pinhole + distortion models).
    #[arg(
        long,
        help_heading = "Refine options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = true
    )]
    #[serde(default = "default_true")]
    pub use_edge_map: bool,

    /// Strength of the edge-guidance factor: the normalized per-gaussian edge
    /// score is scaled by this before the `+ 1.0` that turns it into a
    /// multiplicative sampling weight (LFS `MRNF_EDGE_SCORE_WEIGHT = 0.25`,
    /// mrnf.cpp:68). Only used when `--use-edge-map` is set.
    #[arg(long, help_heading = "Refine options", default_value = "0.25")]
    #[serde(default = "default_edge_score_weight")]
    pub edge_score_weight: f32,

    /// Error-map densification (MRNF `use_error_map` port). When set, the growth
    /// signal switches from the screen-space position-gradient norm to LFS's
    /// error-weighted signal: a mean-normalized D-SSIM error map of each sampled
    /// view is projected onto the gaussians as the coverage-weighted mean error
    /// `(Σ_p T·α·ê)/(Σ_p T·α)`, window-MAX accumulated, and thresholded (LFS
    /// `use_error_map`, mrnf.cpp:726, coverage-normalized — see the defect-2 note
    /// on `error_map_growth_threshold`). ON by default (LFS `use_error_map` in
    /// `mrnf_defaults`); disable per-run with `--error-map-densification=false`,
    /// which reverts to bit-identical upstream (gradient-driven) growth. This
    /// path costs ~+80% step time (the extra feature backward) and that cost is
    /// intentionally default-on per operator. When BOTH this and `--use-edge-map`
    /// are set, error is the base growth signal and edge is a multiplicative bias
    /// within the error-thresholded set (LFS semantics).
    ///
    /// The per-gaussian score comes from a `feat_dim=2` feature backward (error
    /// and coverage rows in one pass), then per-view positive-median normalized
    /// (see `crate::edge` / `crate::error_map`), so it works for every camera
    /// model the renderer supports.
    #[arg(
        long,
        help_heading = "Refine options",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = true
    )]
    #[serde(default = "default_true")]
    pub error_map_densification: bool,

    /// Growth threshold for the error-map signal (`τ_err`). Governs ONLY the
    /// `--error-map-densification` path; the gradient path uses
    /// `--growth-grad-threshold` (a different scale, do not conflate). The
    /// per-gaussian score is the coverage-weighted MEAN error over a gaussian's
    /// footprint, `(Σ T·α·ê)/(Σ T·α)`, then per-view POSITIVE-MEDIAN normalized
    /// (median → 1.0, like the edge path) so it lands on a stable scale
    /// (defect-2 fix, 2026-07-22). So the default 1.0 admits gaussians
    /// reconstructing WORSE than the per-view median. NOTE this is NOT LFS's raw-sum `τ_err = 0.003`
    /// (mrnf.cpp:726) — that 0.003 is the gradient-mode scalar scale and on the
    /// pixel-summed error degenerates to a no-op floor at the port's render
    /// resolution; see `train::accumulate_error_sample` for the full derivation.
    /// `--growth-select-fraction` still layers on selection pressure.
    #[arg(long, help_heading = "Refine options", default_value = "1.0")]
    #[serde(default = "default_error_map_growth_threshold")]
    pub error_map_growth_threshold: f32,

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

    /// Start learning rate for the `DiG` DECODER MLP.
    ///
    /// Note this used to drive the per-gaussian feature table as well, which
    /// made `DiG` train nothing usable: the table is ~N rows touched only when
    /// a gaussian is visible, while the decoder is shared and touched every
    /// pixel of every step, so one value cannot serve both. At 1e-2 the decoder
    /// trains and the table never leaves its N(0,1) init (measured top-1/4/16
    /// explained variance 0.0223/0.0829/0.2880 vs 0.0156/0.0625/0.25 for
    /// isotropic noise, unchanged at 6.5x more gradient per gaussian); at 1.0
    /// the table forms structure but the decoder collapses to zero output.
    /// The table now has its own `--dino-feature-lr`.
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    pub dino_lr: f64,

    /// Final learning rate for the `DiG` decoder MLP
    /// (exponential decay over 6000 steps, then held).
    #[arg(long, help_heading = "Training options", default_value = "1e-3")]
    pub dino_lr_end: f64,

    /// Start learning rate for the per-gaussian `DiG` feature table.
    ///
    /// Much larger than the decoder LR on purpose: each row receives gradient
    /// only on steps where its gaussian is visible, so its effective update
    /// rate is orders of magnitude lower than the shared decoder's.
    #[arg(long, help_heading = "Training options", default_value = "3e-1")]
    pub dino_feature_lr: f64,

    /// Final learning rate for the per-gaussian `DiG` feature table
    /// (same exponential decay horizon as the decoder).
    #[arg(long, help_heading = "Training options", default_value = "3e-2")]
    pub dino_feature_lr_end: f64,

    /// Weight of the 3-nearest-neighbor feature-variance regularizer
    /// (enabled after step 1000; 0 disables).
    #[arg(long, help_heading = "Training options", default_value = "0.01")]
    pub dino_nn_reg_weight: f32,

    /// Weight of l1 loss on depth. The residual is disparity-space by default;
    /// see `--depth-loss-space` to switch it to metric, which also changes how
    /// `--normalize-metric-weights` treats THIS weight.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    pub depth_loss_weight: f32,

    /// Space the depth residual is measured in.
    ///
    /// `disparity` (default, byte-identical) scores `|1/pred - 1/gt|`;
    /// `metric` scores `|pred - gt|` in the prior's own units. Masking,
    /// denominator and the optional gradient-aware pixel weight are identical
    /// either way — only the residual changes.
    ///
    /// The two differ in how the per-pixel gradient scales with range:
    /// disparity's is `1/d²` (a near splat gets orders of magnitude more depth
    /// gradient than a far one), metric's is a constant `1`. That matters most
    /// with `--depth-source plane-fused`, where the blending-weight gradients
    /// are live and the optimizer can answer a large near-field depth gradient
    /// by FADING the splat rather than rotating it — the opacity collapse
    /// measured on both ablation scenes. The `gauss-surf` PGSR reference
    /// (rerun-io/examples-monorepo, Apache-2.0, by Pablo Vela) runs fused with
    /// a metric L1 at weight `3.2 / scene_scale` and does not see that collapse.
    ///
    /// **Scale-normalization flips with this flag.** Under `disparity` the
    /// depth weight is excluded from `--normalize-metric-weights` (dividing a
    /// `1/m` residual's weight by a length moves it the wrong way by `s²`);
    /// under `metric` the residual is in metres, so the depth weight joins the
    /// normalized set and is divided by the scene scale, reproducing the
    /// reference's `3.2 / scene_scale` from a scene-independent `3.2`.
    #[arg(long, help_heading = "Training options", default_value = "disparity")]
    #[serde(default)]
    pub depth_loss_space: DepthLossSpace,

    /// What the depth loss does with pixels the render does not COVER.
    ///
    /// The center depth source composites `accum / α.clamp_min(1e-10)`, which
    /// is exactly 0 where nothing was rendered. Such a pixel can still have a
    /// valid GT depth, and the legacy mask is GT-only — so it scores a
    /// full-magnitude residual (`|0 − 1/D_gt|`: 2.0 m⁻¹ against a 0.5 m prior)
    /// and is counted in the mean. That is a floor in the REPORTED loss plus a
    /// dilution of everything else in it, proportional to the uncovered
    /// fraction of the frame.
    ///
    /// * `count` — default, byte-identical: uncovered pixels stay in both sums.
    /// * `exclude-numerator` — they leave the numerator only. In the default
    ///   disparity space this is GRADIENT-IDENTICAL to `count` (the removed
    ///   terms sit behind a `mask_fill` whose VJP zeroes them), so it corrects
    ///   the reported number and nothing else. **Not** gradient-identical under
    ///   `--depth-loss-space metric`, which has no `pred <= 0` guard and so
    ///   carries a live `∓1` there — see `DepthUncovered`'s docs.
    /// * `exclude` — `LichtFeld Studio` semantics (`depth_loss.cu`'s
    ///   `pixel_active` skips inactive pixels from every sum): they leave the
    ///   numerator AND the denominator. This rescales every surviving pixel's
    ///   gradient by `N_gt-valid / N_(covered ∧ gt-valid)`, so it is a real
    ///   change in effective depth-supervision weight, largest where coverage
    ///   is lowest (early training, frame edges).
    ///
    /// The two exclude modes are separate flag values on purpose: it is what
    /// lets a metric movement under `exclude` be attributed to the denominator
    /// rescale rather than to the numerator change.
    ///
    /// No effect with `--depth-source plane-aux`/`plane-fused`: those already
    /// zero the GT at plane-invalid pixels, so uncovered pixels have left
    /// through the `gt > 0` mask before this one is consulted.
    #[arg(long, help_heading = "Training options", default_value = "count")]
    #[serde(default)]
    pub depth_uncovered: DepthUncovered,

    /// Global iter at which the depth-loss weight starts decaying. Full
    /// `--depth-loss-weight` before it. Only meaningful with
    /// `--depth-weight-end-iter` > 0.
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default = "default_depth_weight_start_iter")]
    pub depth_weight_start_iter: u32,

    /// Global iter at which the decay finishes; weight is
    /// `--depth-weight-end` from here on. 0 = annealing OFF (constant
    /// weight, previous behaviour, byte-identical).
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default = "default_depth_weight_end_iter")]
    pub depth_weight_end_iter: u32,

    /// Final depth-loss weight after --depth-weight-end-iter.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default = "default_depth_weight_end")]
    pub depth_weight_end: f32,

    /// Decay shape between start and end iters.
    #[arg(long, help_heading = "Training options", default_value = "linear")]
    #[serde(default)]
    pub depth_weight_decay: DepthWeightDecay,

    /// DN-Splatter-style gradient-aware depth weighting: per-pixel weight
    /// `w = exp(-|grad I| / sigma)` from the GT RGB image multiplies the depth-loss
    /// map — full weight on textureless regions, down-weighted on image edges.
    /// Composes multiplicatively with the annealed scalar weight.
    #[arg(long, help_heading = "Training options", default_value = "false")]
    #[serde(default)]
    pub depth_grad_aware: bool,

    /// Sigma for the gradient-aware weight, in `[0, 1]` RGB intensity units of the
    /// channel-mean forward-difference gradient. Smaller = harsher edge
    /// down-weighting. Ignored unless --depth-grad-aware.
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    #[serde(default = "default_depth_grad_sigma")]
    pub depth_grad_sigma: f32,

    /// Weight of the l1 loss between the rendered per-gaussian normal image and
    /// an external normal prior (`normal/<stem>.tiff`, 3-channel float32,
    /// camera-frame `OpenCV`-convention unit normals, `(0,0,0)` = invalid).
    /// Needs prior data; inert without it. DN-Splatter's normal loss /
    /// `PlanarGS` `L_rn`; `PlanarGS` ratio suggests ~0.2. 0 disables.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub normal_loss_weight: f32,

    /// Weight of the depth/normal consistency term: `1 - dot` between normals
    /// derived from the RENDERED depth and the rendered per-gaussian normals
    /// (`PlanarGS` `L_dn`). Needs no prior data at all. Setting it forces the
    /// depth render channel on. Pinhole cameras only — skipped with a warning
    /// on fisheye models. `PlanarGS` ratio suggests ~0.05. 0 disables.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub depth_normal_weight: f32,

    /// Iteration at which `--depth-normal-weight` switches on. Before it the
    /// consistency term contributes nothing AND its render work is skipped.
    ///
    /// 2DGS gates exactly this term (`lambda_normal = opt.lambda_normal if
    /// iteration > 7000 else 0.0`, i.e. 7k of 30k, ~23% of the run) while
    /// letting densification continue to 15k — so the gate is NOT about waiting
    /// for topology to settle, it is about not enforcing self-consistency on
    /// geometry that has barely formed. Counts GLOBAL iterations, so a resumed
    /// run does not restart the countdown. 0 = never gate (previous behaviour).
    ///
    /// Counting globally also means the gate does NOT re-close at an LOD
    /// transition, even though the trainer is re-seeded there. That is
    /// deliberate — by then the geometry it was waiting for exists.
    ///
    /// Note the asymmetry: `--flatten-loss-weight` deliberately has NO such
    /// gate, because DN-Splatter runs the identical scale term ungated at 1.0
    /// from step 0.
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default)]
    pub depth_normal_start_iter: u32,

    // ------------------------------------------------------------------
    // PGSR plane-render config surface (WS-L). Every field below defaults to
    // an OFF sentinel that leaves the trainer byte-identical to its pre-change
    // behaviour; the `playroom_0812` 15k baseline (opacity p50 0.2132, scale
    // p50 0.0082, on-seed 92.2%, dark splats 1.79%) must stay reproducible
    // with defaults.
    // ------------------------------------------------------------------
    /// Source of the supervised depth map.
    ///
    /// `center` (default) is the alpha-composited camera-z of splat means —
    /// previous behaviour, byte-identical. `plane-aux` and `plane-fused` use
    /// PGSR ray-plane depth (arXiv:2406.06521) and differ from each other only
    /// in their BACKWARD contract, not their forward values; see `DepthSource`.
    ///
    /// `plane-aux` is SCENE-DEPENDENT, and its measured benefit depends on what
    /// it is stacked on — quote each number with its comparison:
    ///
    ///   * `playroom_0812`, ALONE vs baseline: **−3.8° thin-axis, +0.36 dB**.
    ///   * `playroom_0812`, ON TOP OF `--flatten-loss-weight 1.0`: **−0.8°,
    ///     +0.03 dB** — most of what it buys alone, flatten has already bought.
    ///   * `ARKitScenes` 48018538, ALONE: −0.3°, i.e. null.
    ///   * `ARKitScenes` 48018538, ON TOP OF flatten: **+0.7° WORSE.**
    ///
    /// So it is not a free addition to the flatten recipe: it competes for the
    /// same smallest-axis degree of freedom. Discriminator, ~7k iterations:
    /// run flatten-alone and flatten+plane-aux side by side and keep plane-aux
    /// only if on-seed recovers and thin-axis does not worsen.
    ///
    /// `plane-fused` is **EXPERIMENTAL and measured harmful in this trainer** —
    /// it collapses opacity p50 by 28%/34% on those two scenes and puts playroom
    /// under the 24 dB gate. The cause is under investigation and is believed to
    /// be ours, not the technique's (the reference trainer's weight-path-live
    /// renderer does not collapse on the same data). Full numbers and the
    /// pending experiments: `docs/superpowers/specs/2026-08-20-pgsr-ablation-synthesis.md`
    /// §3.3 and §5.2.
    #[arg(long, help_heading = "Training options", default_value = "center")]
    #[serde(default)]
    pub depth_source: DepthSource,

    /// Global iteration at which the normal-term ramp starts. **0 = OFF**
    /// (full weight from step 0, previous behaviour, byte-identical).
    ///
    /// When set, `--normal-loss-weight` and `--depth-normal-weight` are both
    /// multiplied by a ramp that is 0 before this iteration and climbs linearly
    /// to 1 over `--normal-ramp-iters`. A ramp value of exactly 0 also SKIPS
    /// the normal render pass entirely (same philosophy as
    /// `--depth-normal-start-iter`), so the gate costs nothing rather than
    /// rendering work that gets multiplied by zero.
    ///
    /// Counts GLOBAL iterations, like every other schedule knob in this fork,
    /// so a resumed run does not restart the countdown and an LOD transition
    /// does not re-close the gate.
    ///
    /// Recipe, from the `gauss-surf` PGSR trainer (rerun-io/examples-monorepo,
    /// Apache-2.0, by Pablo Vela), which starts at step 1,400 and ramps over 875
    /// of its 7,000 — i.e. 20% of the run, ramping over 12.5%. These are its
    /// implementation constants, not values either paper states:
    ///   * 15k ablation run: `--normal-ramp-start-iter 3000 --normal-ramp-iters 1875`
    ///   * 30k default run (`--total-train-iters` defaults to 30000):
    ///     `--normal-ramp-start-iter 6000 --normal-ramp-iters 3750`
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default)]
    pub normal_ramp_start_iter: u32,

    /// Length of the normal-term ramp in global iterations, after
    /// `--normal-ramp-start-iter`. 0 = hard step to full weight at the start
    /// iteration. Inert unless `--normal-ramp-start-iter` is nonzero (setting
    /// it alone is rejected by `validate()` rather than silently ignored).
    ///
    /// Units: iterations. See `--normal-ramp-start-iter` for the recipe.
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default)]
    pub normal_ramp_iters: u32,

    /// Final value of `--depth-normal-weight` after the late consistency bump.
    /// **0.0 = OFF** (constant weight, previous behaviour, byte-identical).
    ///
    /// The `gauss-surf` PGSR trainer (rerun-io/examples-monorepo, Apache-2.0, by
    /// Pablo Vela) bumps its consistency weight 0.50 -> 0.55 late in training,
    /// from step 5,500 over 500 of its 7,000 — 78.6% of the run, over 7.1%.
    /// Another implementation constant with no counterpart in the papers.
    /// Units: same dimensionless weight as `--depth-normal-weight`.
    ///
    /// Recipe: at 15k, `--depth-normal-weight-end-start-iter 11800
    /// --depth-normal-weight-end-ramp-iters 1050`; at the 30k default,
    /// `23600` / `2100`.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub depth_normal_weight_end: f32,

    /// Global iteration at which the late consistency bump starts ramping.
    /// Units: iterations. Inert unless `--depth-normal-weight-end` > 0.
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default)]
    pub depth_normal_weight_end_start_iter: u32,

    /// Length of the late consistency bump ramp, in global iterations. 0 = hard
    /// step at `--depth-normal-weight-end-start-iter`. Inert unless
    /// `--depth-normal-weight-end` > 0.
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default)]
    pub depth_normal_weight_end_ramp_iters: u32,

    /// Per-pixel normal contradiction gate, in DEGREES. **0 = OFF** (previous
    /// behaviour, byte-identical).
    ///
    /// `NeuRIS`-style (arXiv:2206.13597): a prior-normal pixel whose angle to the
    /// RENDERED normal exceeds this threshold is dropped from
    /// `--normal-loss-weight`'s mask for that step. Both operands are detached
    /// — it is a mask, not a gradient path. The mean is taken over the GATED
    /// valid count, so surviving pixels keep full per-pixel magnitude instead of
    /// the whole term silently annealing as the gate tightens.
    ///
    /// This is the per-pixel refinement of the whole-frame median-cosine check
    /// `ingest/splatcam/normals_moge.py` already performs at prior-generation
    /// time: the frame check catches sign-flipped priors, the gate drops
    /// locally-contradicted pixels (transients, reflections, `MoGe` failures)
    /// inside otherwise-good frames.
    ///
    /// Units: degrees, in `[0, 180]`. Reference value: 30.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub normal_gate_degrees: f32,

    /// Global iteration at which `--normal-gate-degrees` arms. 0 = armed from
    /// step 0 whenever the gate is on. Units: iterations. Inert unless
    /// `--normal-gate-degrees` > 0.
    ///
    /// Recipe: the `gauss-surf` PGSR trainer (rerun-io/examples-monorepo,
    /// Apache-2.0, by Pablo Vela) arms its 30 degree gate at step 2,625 of
    /// 7,000 — 37.5% of the run, which is ~5600 at 15k and ~11250 at the 30k
    /// default. The 30 degrees is `NeuRIS`'s; the arming step is `gauss-surf`'s.
    #[arg(long, help_heading = "Training options", default_value = "0")]
    #[serde(default)]
    pub normal_gate_start_iter: u32,

    /// Divide the METRIC-DIMENSIONED loss weights by the scene scale, so a
    /// single recipe transfers between scenes of different physical size.
    /// **Default off = byte-identical.**
    ///
    /// The scale is captured ONCE from the training camera poses at trainer
    /// construction and never updated — a live, refine-updated value would make
    /// the effective weights drift mid-run and confound the ramp schedules.
    ///
    /// Which weights this touches, and why, is spelled out at the consumption
    /// site in `train.rs`; the short version is that `--depth-loss-weight` is
    /// deliberately NOT in the list because our depth loss is disparity-space.
    #[arg(long, help_heading = "Training options", default_value = "false")]
    #[serde(default)]
    pub normalize_metric_weights: bool,

    /// Keep training after a non-finite (NaN / inf) total loss instead of
    /// aborting the run.
    ///
    /// **This default is a DELIBERATE BEHAVIOUR CHANGE.** Before this flag
    /// existed there was no check at all: a single NaN loss step wrote NaN
    /// gradients through the optimizer into the parameters, and the run
    /// continued for hours producing garbage that only surfaced at export. The
    /// trainer now aborts instead, matching the reference trainer, which raises
    /// `FloatingPointError` on any non-finite loss term rather than warning.
    ///
    /// So a run that previously "succeeded" on a poisoned scene will now STOP.
    /// That is the point: the output of such a run was never usable. Pass this
    /// flag only to reproduce or debug the poisoning itself — the resulting
    /// splats are not a deliverable.
    ///
    /// The check is not free (it reads a scalar back from the GPU, which
    /// synchronises), so it does not run every step. See
    /// `NONFINITE_LOSS_CHECK_STEPS` in `train.rs` for the cadence and the
    /// blast-radius argument for the gaps between checks.
    #[arg(long, help_heading = "Training options", default_value = "false")]
    #[serde(default)]
    pub allow_nonfinite_loss: bool,

    /// Weight of total-variation smoothness on the rendered normal image
    /// (DN-Splatter's `L_smooth`). Needs no prior data.
    ///
    /// DN-Splatter weights this **0.5**, five times its normal data term (0.1),
    /// making it the largest weight in their normal group. On a textureless wall
    /// the per-pixel normal field can be noisy while still matching the prior on
    /// average; the data term cannot see that and this can. Since textureless
    /// walls are the reason these priors exist, this is the load-bearing one.
    /// 0 disables.
    ///
    /// CAVEAT on 0.5: DN-Splatter's RELEASED CODE never applies its
    /// `normal_lambda`, so the runs behind their tables used smoothness:data at
    /// 1:1, not 5:1. Our TV is also a pooled per-element mean where theirs is
    /// `mean(|h|) + mean(|w|)` (~2x) and the paper normalises per-pixel (~3x), so
    /// the same number does not mean the same strength. Order of magnitude to
    /// sweep, not a validated setting.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub normal_smooth_weight: f32,

    /// Weight of the flattening term: the population mean of each gaussian's
    /// smallest activated scale, on the RAW (pre-3D-filter) scales. A soft
    /// 2DGS-style pressure toward surface-aligned gaussians — nothing is
    /// collapsed or re-parametrized. `PlanarGS` `L_s`; `PlanarGS` ratio suggests
    /// ~1.0. 0 disables.
    ///
    /// **RECOMMENDED CORE SETTING: `1.0`.** The default stays 0 only so that
    /// existing runs and recorded baselines do not change under anyone's feet —
    /// it is NOT a recommendation. Across a 20-arm `ARKitScenes` matrix and a
    /// 9-arm `playroom_0812` matrix this is the **only** ingredient that
    /// improves splat orientation on both scenes
    /// (`docs/superpowers/specs/2026-08-20-pgsr-ablation-synthesis.md` §1, §3.1):
    ///
    ///   * thin-axis median (angle between a splat's smallest axis and the local
    ///     surface normal): **46.61° → 37.68° (−8.9°)** on `ARKitScenes` 48018538,
    ///     **39.65° → 25.37° (−14.3°)** on `playroom_0812`;
    ///   * splats within 15° of their surface: 9.8% → 16.5% and 15.0% → 30.5%;
    ///   * PSNR is essentially unchanged (−0.09 dB / +0.15 dB) — a PSNR-gated
    ///     sweep cannot see this term at all, in either direction.
    ///
    /// Mechanism, visible in the min-axis median: 7.2 mm → 0.75 mm (`ARKitScenes`),
    /// 3.7 mm → 0.16 mm (playroom). It trades centre accuracy for orientation, so
    /// on-seed@1cm drops 1–3 pp; that is the term working, not a regression.
    ///
    /// Costs and interactions worth knowing before turning it on:
    ///   * −30% it/min (`ARKitScenes`) / −15% (playroom at matched splat count);
    ///   * do NOT combine with `--normalize-metric-weights` on a metric scene —
    ///     it divides this weight by the scene scale and measurably weakens it
    ///     (+1.4° / +1.1°);
    ///   * it drives log-scales down, which walks splats into the region where
    ///     the Mip-Splatting 3D-filter fold used to NaN their gradients. That
    ///     bug is fixed in this branch, but Stage 5 `--filter-nan` stays
    ///     mandatory regardless (SOG codebook poisoning);
    ///   * at weights above 1.0 the first thing to break is not the trainer but
    ///     SOG's fixed 8-bit quaternion precision (§4 "Costs").
    ///
    /// A cap that BINDS is part of the recipe: opacity only recovers once the
    /// population pins at `--max-splats`.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub flatten_loss_weight: f32,

    /// Weight of the scale-explosion regularizer (Stipple `L_scale-regularizer`,
    /// arXiv:2608.00931): mean `s²` over ACTIVATED scales above
    /// `--scale-reg-threshold`, zero at or below it. A differentiable brake on
    /// MRNF's runaway "fog" gaussians in unconstrained regions (sky-smear) —
    /// prevents the blow-up our prune-side guards only remove after the fact.
    /// 0 disables.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub scale_reg_weight: f32,

    /// Activated-scale threshold above which `--scale-reg-weight` penalizes `s²`.
    /// World/scale units; set above the surface population's p99 so only the
    /// exploded tail is gated. Inert unless `--scale-reg-weight > 0`.
    #[arg(long, help_heading = "Training options", default_value = "3.0")]
    #[serde(default = "default_scale_reg_threshold")]
    pub scale_reg_threshold: f32,

    /// Weight of the anti-needle isotropy regularizer (Stipple `L_anti-needle`,
    /// arXiv:2608.00931): mean `exp(log s_max − log s_min)` per gaussian. Pulls
    /// anisotropic splats toward isotropic covariance. 0 disables.
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub anti_needle_weight: f32,

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

    // ------------------------------------------------------------------
    // TIDI-GS floater / haze suppression (arXiv 2601.09291). See
    // `research/indoor-360-haze-removal.md` and `crate::tidi`. Multi-signal +
    // isolation pruning for the translucent equilibrium floaters that grow on
    // textureless indoor walls (which opacity thresholds structurally cannot
    // remove). The WHOLE family is default-OFF: with BOTH `--tidi-prune` and
    // `--tidi-depth-prune` unset the trainer never allocates TIDI state, so MRNF
    // / depth-loss / PPISP / normal-prior runs take the identical path they
    // always did. `--tidi-depth-prune` (defined after the photometric knobs) is
    // a SEPARATE, standalone path that runs even without `--tidi-prune`.
    // ------------------------------------------------------------------
    /// Master switch for TIDI-GS multi-signal + isolation floater pruning.
    #[arg(long, help_heading = "TIDI options", default_value = "false")]
    #[serde(default)]
    pub tidi_prune: bool,

    /// Global iter at which TIDI pruning may begin (paper: a warmup so the scene
    /// has formed). Signals accumulate from the first refine regardless; only
    /// the prune is gated.
    #[arg(long, help_heading = "TIDI options", default_value = "500")]
    #[serde(default = "default_tidi_prune_start_iter")]
    pub tidi_prune_start_iter: u32,

    /// Minimum global-iter gap between TIDI cleanup passes (paper: every 400
    /// steps). Effective cadence is rounded up to the next refine cycle, since
    /// TIDI runs inside the refine hook.
    #[arg(long, help_heading = "TIDI options", default_value = "400")]
    #[serde(default = "default_tidi_prune_every")]
    pub tidi_prune_every: u32,

    /// Per-Gaussian warmup: a splat younger than this many steps (since birth /
    /// last split) is never a prune candidate, protecting fresh detail.
    #[arg(long, help_heading = "TIDI options", default_value = "500")]
    #[serde(default = "default_tidi_warmup_steps")]
    pub tidi_warmup_steps: u32,

    /// Visibility signal: candidate iff the number of refine WINDOWS in which
    /// this gaussian was ever visible is <= this (paper `τ_vis` = 2.0). NOTE: the
    /// unit is refine windows, not steps -- see `TidiState::accumulate_window`,
    /// which collapses each window to a 0/1 "seen" indicator so this threshold is
    /// reachable (raw per-step counts scale with training length).
    #[arg(long, help_heading = "TIDI options", default_value = "2.0")]
    #[serde(default = "default_tidi_vis_threshold")]
    pub tidi_vis_threshold: f32,

    /// Opacity signal: candidate iff opacity <= this (paper `τ_α` = 0.04).
    #[arg(long, help_heading = "TIDI options", default_value = "0.04")]
    #[serde(default = "default_tidi_opacity_threshold")]
    pub tidi_opacity_threshold: f32,

    /// Learned-importance signal: candidate iff `sigmoid(omega_i)` <= this
    /// (paper `τ_ω` = 0.35).
    #[arg(long, help_heading = "TIDI options", default_value = "0.35")]
    #[serde(default = "default_tidi_importance_threshold")]
    pub tidi_importance_threshold: f32,

    /// Position-gradient EMA signal: candidate iff EMA <= this (paper
    /// `τ_grad` = 5e-4).
    #[arg(long, help_heading = "TIDI options", default_value = "5e-4")]
    #[serde(default = "default_tidi_grad_threshold")]
    pub tidi_grad_threshold: f32,

    /// EMA decay for the per-refine position-gradient signal (paper β = 0.99).
    /// NOTE: advances once per refine window, not per step (Brush only exposes a
    /// window position-gradient signal).
    #[arg(long, help_heading = "TIDI options", default_value = "0.99")]
    #[serde(default = "default_tidi_grad_ema_beta")]
    pub tidi_grad_ema_beta: f32,

    /// L1 sparsity weight on sigmoid(omega). 0 keeps omega alive on the
    /// photometric gradient only (importance never falls -> effectively a
    /// 3-signal gate); >0 lets persistently idle Gaussians decay into the pool.
    #[arg(long, help_heading = "TIDI options", default_value = "0.01")]
    #[serde(default = "default_tidi_importance_reg")]
    pub tidi_importance_reg: f32,

    /// Adam LR for the omega importance leaf parameter.
    #[arg(long, help_heading = "TIDI options", default_value = "0.05")]
    #[serde(default = "default_tidi_importance_lr")]
    pub tidi_importance_lr: f64,

    /// SH high-frequency-energy detail-guard quantile: exempt a candidate whose
    /// ||`f_rest`|| is at/above this quantile of the STABLE set (specular/detail).
    /// 0 disables. ADAPTIVE: the flag sets the quantile, not a fixed threshold.
    #[arg(long, help_heading = "TIDI options", default_value = "0.95")]
    #[serde(default = "default_tidi_guard_sh_quantile")]
    pub tidi_guard_sh_quantile: f32,

    /// Thinness detail-guard quantile: exempt a candidate whose smallest scale
    /// axis is at/below this quantile of the STABLE set (thin structure). 0
    /// disables. ADAPTIVE.
    #[arg(long, help_heading = "TIDI options", default_value = "0.10")]
    #[serde(default = "default_tidi_guard_thin_quantile")]
    pub tidi_guard_thin_quantile: f32,

    /// Anisotropy detail-guard quantile: exempt a candidate whose scale ratio
    /// s3/s1 is at/above this quantile of the STABLE set (an elongated sheet /
    /// needle the thinness guard misses when s1 is not small). 0 disables.
    /// ADAPTIVE.
    #[arg(long, help_heading = "TIDI options", default_value = "0.95")]
    #[serde(default = "default_tidi_guard_aniso_quantile")]
    pub tidi_guard_aniso_quantile: f32,

    /// Local colour-variance detail-guard quantile (needs an extra candidate
    /// k-NN pass; off by default for cost). Thresholded against the CANDIDATE
    /// distribution rather than the stable set. 0 disables (default).
    #[arg(long, help_heading = "TIDI options", default_value = "0.0")]
    #[serde(default = "default_tidi_guard_color_var_quantile")]
    pub tidi_guard_color_var_quantile: f32,

    /// k for the isolation k-NN (paper k = 16).
    #[arg(long, help_heading = "TIDI options", default_value = "16")]
    #[serde(default = "default_tidi_knn_k")]
    pub tidi_knn_k: u32,

    /// Isolation local cap: prune at most this fraction of a spatial cell's
    /// candidates per cycle (paper 1.0%).
    #[arg(long, help_heading = "TIDI options", default_value = "0.01")]
    #[serde(default = "default_tidi_local_cap_frac")]
    pub tidi_local_cap_frac: f32,

    /// Isolation global cap: prune at most this fraction of ALL Gaussians per
    /// cycle (paper 0.2%).
    #[arg(long, help_heading = "TIDI options", default_value = "0.002")]
    #[serde(default = "default_tidi_global_cap_frac")]
    pub tidi_global_cap_frac: f32,

    // ------------------------------------------------------------------
    // TIDI depth / LiDAR-residual prune (a SEPARATE, standalone path from the
    // four photometric signals above -- deliberately NOT AND-gated with
    // opacity/omega/grad). The four photometric signals structurally CANNOT
    // remove indoor equilibrium wall-haze: the haze floaters are photometrically
    // VALID (stuck at a cancelling-error equilibrium), so they pass
    // opacity/omega/grad and are never candidates. The one signal that CAN
    // distinguish haze from real surface is GEOMETRY: haze floats in empty space
    // in FRONT of the measured LiDAR/depth surface; real geometry sits AT the
    // surface. This path reuses the exact per-frame depth `--depth-loss-weight`
    // consumes (`depth/<stem>.tiff`, 0 = no return) as a hard prune
    // discriminator. Default OFF and byte-inert unless enabled.
    // ------------------------------------------------------------------
    /// Master switch for the depth/LiDAR-residual prune path. Independent of
    /// `--tidi-prune`'s photometric path (both live under the same TIDI state);
    /// if this is set but `--tidi-prune` is not, ONLY the depth path runs. Inert
    /// (no state allocated, no per-step cost) when both are unset. Opt-in
    /// (never auto-on) because it DELETES splats on a depth signal whose quality
    /// varies: LiDAR-projected depth is trustworthy, a mono estimator (e.g. DA3)
    /// on featureless walls is sparse/unreliable, and the code cannot tell them
    /// apart -- so the operator opts into "use depth when present" deliberately.
    #[arg(long, help_heading = "TIDI options", default_value = "false")]
    #[serde(default)]
    pub tidi_depth_prune: bool,

    /// Depth-residual margin: a Gaussian counts as "floating" only when its
    /// camera-space z is more than this in FRONT of the measured depth at its
    /// projected pixel. Units are the DEPTH map's units -- metres for
    /// LiDAR/metric depth (the default 0.05 = 5 cm); an `SfM` dataset's depth may
    /// be non-metric, so scale this accordingly.
    #[arg(long, help_heading = "TIDI options", default_value = "0.05")]
    #[serde(default = "default_tidi_depth_margin")]
    pub tidi_depth_margin: f32,

    /// Depth path: prune when a Gaussian floats in at least this fraction of the
    /// views that carry a valid depth return behind it (default 0.5 = 50%).
    #[arg(long, help_heading = "TIDI options", default_value = "0.5")]
    #[serde(default = "default_tidi_depth_float_frac")]
    pub tidi_depth_float_frac: f32,

    /// Depth path SAFETY gate: never prune a Gaussian unless at least this many
    /// views have a real depth return behind it. This is what keeps the depth
    /// path from touching unscanned regions (no `LiDAR` return -> exempt).
    #[arg(long, help_heading = "TIDI options", default_value = "4")]
    #[serde(default = "default_tidi_depth_min_valid_views")]
    pub tidi_depth_min_valid_views: u32,

    /// Depth path per-cycle global cap: prune at most this fraction of ALL
    /// Gaussians via the depth path per cleanup. Looser than the photometric
    /// `--tidi-global-cap-frac` (0.002) because the LiDAR-gated depth signal is
    /// trustworthy.
    #[arg(long, help_heading = "TIDI options", default_value = "0.02")]
    #[serde(default = "default_tidi_depth_cap_frac")]
    pub tidi_depth_cap_frac: f32,

    // ------------------------------------------------------------------
    // Depth-coupled opacity regularizer -- the SMOOTH, differentiable
    // alternative to the hard `--tidi-depth-prune`. Instead of deleting a
    // floater in one step (which orphans its load-bearing colour and leaves a
    // black halo), this adds a per-step loss whose ONLY gradient path is the
    // Gaussian's activated opacity, fading off-surface splats out SMOOTHLY so
    // the optimizer redistributes their colour into on-surface splats BEFORE
    // they vanish. Gated on a VIEW-INDEPENDENT 3D test: a Gaussian is penalized
    // when its centre is FAR from the seed/LiDAR point cloud, looked up in a
    // static distance-to-cloud grid built once from the seed cloud. No per-frame
    // depth and no camera projection (unlike the old per-view z-buffer residual).
    // Independent of `--depth-loss-weight` and of the TIDI prune state; needs no
    // persistent accumulators. Default OFF and byte-inert (no lookup, no loss
    // term) when the weight is 0.
    // ------------------------------------------------------------------
    /// Depth-coupled opacity-regularizer weight (lambda). 0 = OFF (inert).
    /// >0 adds `lambda * mean_i(p_i * sigmoid(opacity_i))` to the loss,
    /// > where `p_i` is a DETACHED smooth ramp that is ~1 for a Gaussian whose
    /// > centre is FAR (> margin) from the nearest seed/LiDAR cloud point and ~0
    /// > for one on/near the cloud. The gradient reaches ONLY the opacity leaf,
    /// > so far-from-cloud splats fade smoothly rather than being hard-deleted.
    #[arg(long, help_heading = "TIDI options", default_value = "0.0")]
    #[serde(default = "default_depth_opacity_reg_weight")]
    pub depth_opacity_reg_weight: f32,

    /// Depth-opacity-reg margin: a Gaussian is penalized once its centre is more
    /// than this far from the nearest seed/LiDAR cloud point in 3D. Units are
    /// SCENE 3D-distance units (metres for a LiDAR-metric scene; possibly
    /// non-metric for `SfM`). Set it a few times the cloud's nearest-neighbour
    /// spacing so on-surface splats stay safe -- e.g. spacing ~0.024 -> margin
    /// ~0.1-0.2. (Was a per-view depth residual; it is now a 3D distance.)
    #[arg(long, help_heading = "TIDI options", default_value = "0.15")]
    #[serde(default = "default_depth_opacity_reg_margin")]
    pub depth_opacity_reg_margin: f32,

    /// Depth-opacity-reg softness: the width of the sigmoid ramp (in the same 3D
    /// distance units as margin) over which the penalty climbs from ~0 to ~1 as a
    /// Gaussian moves farther from the cloud. Smaller = sharper on/off boundary.
    /// Keep it `< margin`: the ramp is centred at `d = margin` (p = 0.5), so a
    /// smaller softness keeps the penalty near 0 for on-surface splats (`d ~ 0`)
    /// and stops it fading correctly-reconstructed walls. At the default 0.05 vs
    /// margin 0.15, `p(d=0) ≈ 0.047`.
    #[arg(long, help_heading = "TIDI options", default_value = "0.05")]
    #[serde(default = "default_depth_opacity_reg_softness")]
    pub depth_opacity_reg_softness: f32,

    /// Global iteration before which the depth-coupled opacity regularizer is
    /// inert (skipped, no cost). The densifier backfills opacity-faded regions,
    /// so firing the reg before densification stops (`--growth-stop-iter`,
    /// default 15000) fights that backfill loop; start after it (e.g. 15000).
    /// Default 0 = active from the first step (behaviour unchanged).
    #[arg(long, help_heading = "TIDI options", default_value = "0")]
    #[serde(default = "default_depth_opacity_reg_start_iter")]
    pub depth_opacity_reg_start_iter: u32,

    /// PLANE-GATE (FIX 1): augment the distance-to-cloud opacity field with
    /// distance-to-nearest-PLANE, so a wall splat sitting BETWEEN sparse cloud
    /// points (inside the wall's extent) reads on-surface instead of being
    /// penalised as a floater, while a mid-air splat far from every plane is still
    /// caught. Planes are extracted by RANSAC from the seed/LiDAR cloud (NOT a
    /// VLM) once at init. Only meaningful alongside `--depth-opacity-reg-weight`
    /// (it changes what that regularizer's field stores). `false` = the exact
    /// point-only field (byte-identical to the pre-plane behaviour); no RANSAC
    /// runs unless this or `--plane-coplanarity-weight` is set.
    #[arg(long, help_heading = "TIDI options", default_value = "false")]
    #[serde(default)]
    pub plane_gate: bool,

    /// CO-PLANARITY (FIX 2): weight of the plane geometry constraint. 0 = OFF
    /// (inert, no RANSAC on its own). >0 adds, for every Gaussian assigned to a
    /// RANSAC plane, `weight * mean[(n·mu − d)² + variance-along-n]`, which pulls
    /// the centre onto the plane AND flattens it against the plane. Unlike the
    /// opacity gate this carries a real gradient on POSITION and SCALE, so it
    /// removes the geometric ambiguity on featureless walls directly. Riskier than
    /// the gate (it moves geometry). Independent of `--plane-gate`.
    ///
    /// **UNITS — this is why the old suggested 0.05 was inert.** The term is a
    /// squared distance: `(n·mu − d)²` plus a variance, so it carries **metres²**
    /// on a metric scene. `--flatten-loss-weight` carries linear **metres**. At
    /// the scales these terms operate on those are not comparable magnitudes — a
    /// splat 1 cm off its plane contributes 1e-4 here against 1e-2 there, so a
    /// weight tuned by analogy with flatten is two orders of magnitude short.
    ///
    /// **Measured working value on a metric indoor scene: 20.** A 1k-iteration
    /// sweep on `ARKitScenes` 48018538 at w = 0.05 / 2 / 20 gives thin-axis medians
    /// 54.90° / 52.65° / 45.58° — monotone in the weight, with 0.05 sitting at
    /// the inert end (its PSNR is 0.028 dB from the no-coplanarity control, and
    /// the whole 0.05 → 20 sweep spans 0.046 dB, so PSNR cannot adjudicate this).
    /// At 7k, w = 20 with a cloud-derived assignment band is the best fork-native
    /// configuration measured on that scene: thin-axis 33.30° against arm 6's
    /// 36.84°, within-15° 17.5% → 24.8%, opacity p50 UP (0.068 → 0.073 — the only
    /// orientation lever that raises it), PSNR −0.09 dB, at −40% it/min.
    ///
    /// **The value scales with the scene's units**, because the term does: a
    /// scene whose units are 10× larger needs ~100× less weight for the same
    /// pressure. 20 is calibrated for metres. Re-derive rather than copy on a
    /// non-metric `SfM` scene, or use `--normalize-metric-weights` — but note that
    /// flag also divides `--flatten-loss-weight`, which is a measured dilution on
    /// a metric scene.
    ///
    /// Single-scene evidence, and it has not been tried on `playroom_0812`;
    /// `docs/superpowers/specs/2026-08-20-pgsr-ablation-synthesis.md` §3.1, §4, §6.
    #[arg(long, help_heading = "TIDI options", default_value = "0.0")]
    #[serde(default = "default_plane_coplanarity_weight")]
    pub plane_coplanarity_weight: f32,

    /// CO-PLANARITY assignment band: a Gaussian is assigned to a plane only when
    /// its perpendicular distance is below this (in scene 3D-distance units) AND
    /// it projects inside the plane's bounded extent.
    ///
    /// `<= 0` (the default) derives the band from the seed cloud's MEASURED
    /// nearest-neighbour spacing: **2.75× spacing**, the same quantity the
    /// RANSAC inlier band is built from. The resolved value is printed in the
    /// `Plane priors:` line at startup — read it rather than assuming.
    ///
    /// THIS IS A DISTANCE THAT MUST SCALE WITH THE CLOUD, and getting it wrong
    /// is silent. It previously defaulted to `--depth-opacity-reg-margin`
    /// (0.15 m), a different feature's knob in absolute units: on our 7.3 mm
    /// `ARKitScenes` seed that is a 20× band which assigns **68% of all splats**
    /// to the eight room planes and flattens furniture onto the walls. The
    /// derived 0.02 m band assigns 34% and keeps the gain — thin-axis 33.30°
    /// with on-seed@1cm held at 61.7%, against 31.71° with on-seed@1cm
    /// collapsing to 48.1% at the old default.
    ///
    /// Rule if you set it by hand: ≈2.5–3× the seed cloud's NN spacing. Check
    /// membership with `work/arkitscenes_48018538/tools/ransac_bands.py`; over
    /// ~40% of splats assigned means the band is too wide.
    ///
    /// Evidence is single-scene (`ARKitScenes` 48018538);
    /// `docs/superpowers/specs/2026-08-20-pgsr-ablation-synthesis.md` §3.1, §6.
    #[arg(long, help_heading = "TIDI options", default_value = "-1.0")]
    #[serde(default = "default_plane_coplanarity_assign_dist")]
    pub plane_coplanarity_assign_dist: f32,

    // ------------------------------------------------------------------
    // Hard cloud-distance prune (`--cloud-prune`). The VALIDATED floater
    // remover: an in-training HARD prune that DELETES any Gaussian whose centre
    // is FAR (in 3D) from the seed/LiDAR point cloud -- i.e. floating in empty
    // space. View-INDEPENDENT (no camera, no z-buffer, so no see-through leak),
    // it looks the Gaussian's centre up in a static distance-to-cloud grid built
    // ONCE from the seed cloud and prunes when d > `--cloud-prune-dist`. Unlike
    // `--tidi-depth-prune` (per-view projected depth, which reads far points
    // through gaps in a sparse cloud and prunes SURFACE splats), this is the
    // honest floater signal: the cloud IS the measured surface. Because it
    // deletes DURING training (inside the refine cycle), the surface heals and
    // colour redistributes -- no black halo, unlike a post-hoc opacity filter.
    // Its grid is ALWAYS point-only (never plane-augmented, even under
    // `--plane-gate`): a plane would shield wall-perpendicular floaters. Default
    // OFF and byte-inert (grid never built, no lookup) when unset. Pair with
    // `--stop-replace-iter` + `--growth-stop-iter` (same iter as
    // `--cloud-prune-start-iter`) so the prune NET-REDUCES rather than backfills.
    // ------------------------------------------------------------------
    /// Master switch for the hard distance-to-cloud floater prune.
    #[arg(long, help_heading = "TIDI options", default_value = "false")]
    #[serde(default)]
    pub cloud_prune: bool,

    /// Distance threshold: a Gaussian whose centre is farther than this (in scene
    /// 3D-distance units) from the nearest seed/LiDAR cloud point is pruned as a
    /// floater. Default 0.19 ~= 8x a ~0.024 cloud spacing. Larger = only very
    /// isolated Gaussians go; smaller = more aggressive. NOTE: the grid's
    /// conservative half-voxel bias (vox = dist/3) means the EFFECTIVE cut sits a
    /// bit ABOVE the nominal value, erring toward keeping splats.
    #[arg(long, help_heading = "TIDI options", default_value = "0.19")]
    #[serde(default = "default_cloud_prune_dist")]
    pub cloud_prune_dist: f32,

    /// Global iter before which the cloud-prune is inert (no grid lookup, no
    /// prune). Default 0 = prune from the first refine. Pair with
    /// `--stop-replace-iter` (stop dead-slot backfill) and `--growth-stop-iter`
    /// (stop densification) at the SAME iter so the prune net-reduces the splat
    /// count instead of being immediately backfilled.
    #[arg(long, help_heading = "TIDI options", default_value = "0")]
    #[serde(default = "default_cloud_prune_start_iter")]
    pub cloud_prune_start_iter: u32,
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
        if self.depth_weight_end_iter != 0
            && self.depth_weight_end_iter <= self.depth_weight_start_iter
        {
            return Err(
                "depth-weight-end-iter must be greater than depth-weight-start-iter".to_owned(),
            );
        }
        if self.depth_weight_end < 0.0 {
            return Err("depth-weight-end must not be negative".to_owned());
        }
        if self.depth_grad_aware && self.depth_grad_sigma <= 0.0 {
            return Err(
                "depth-grad-sigma must be positive when depth-grad-aware is set".to_owned(),
            );
        }
        // `--normal-ramp-start-iter 0` means the ramp is OFF, so a ramp LENGTH
        // with no start would be silently inert. Reject rather than ignore:
        // a config that quietly does nothing is the failure mode this whole
        // default-inertness discipline exists to make impossible.
        if self.normal_ramp_iters != 0 && self.normal_ramp_start_iter == 0 {
            return Err(
                "normal-ramp-iters requires a nonzero normal-ramp-start-iter (0 = ramp OFF)"
                    .to_owned(),
            );
        }
        if self.depth_normal_weight_end < 0.0 {
            return Err("depth-normal-weight-end must not be negative".to_owned());
        }
        if !(self.normal_gate_degrees.is_finite()
            && (0.0..=180.0).contains(&self.normal_gate_degrees))
        {
            return Err("normal-gate-degrees must be finite and in [0, 180]".to_owned());
        }
        Ok(())
    }

    /// Effective depth-loss weight at `global_iter`. `end_iter == 0` disables
    /// annealing entirely (returns `depth_loss_weight` unchanged).
    pub fn depth_weight_at(&self, global_iter: u32) -> f32 {
        let w0 = self.depth_loss_weight;
        if self.depth_weight_end_iter == 0 {
            return w0;
        }
        let s = self.depth_weight_start_iter;
        let e = self.depth_weight_end_iter;
        if global_iter <= s {
            return w0;
        }
        let w1 = self.depth_weight_end;
        if global_iter >= e {
            return w1;
        }
        let t = (global_iter - s) as f32 / (e - s) as f32;
        match self.depth_weight_decay {
            DepthWeightDecay::Linear => w0 + (w1 - w0) * t,
            DepthWeightDecay::Cosine => {
                w1 + (w0 - w1) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
            }
        }
    }

    /// Multiplier applied to BOTH `--normal-loss-weight` and the effective
    /// `--depth-normal-weight` at `global_iter`.
    ///
    /// `normal_ramp_start_iter == 0` disables the ramp entirely and returns an
    /// exact `1.0`, which is what makes the default byte-identical. Counts
    /// global iterations.
    pub fn normal_ramp_at(&self, global_iter: u32) -> f32 {
        if self.normal_ramp_start_iter == 0 {
            return 1.0;
        }
        linear_ramp_weight(
            global_iter,
            self.normal_ramp_start_iter,
            self.normal_ramp_iters,
        )
    }

    /// Effective `--depth-normal-weight` at `global_iter`, including the late
    /// consistency bump. `depth_normal_weight_end == 0.0` disables the bump and
    /// returns `depth_normal_weight` unchanged.
    ///
    /// Does NOT include `normal_ramp_at`; the two multiply at the call site.
    pub fn depth_normal_weight_at(&self, global_iter: u32) -> f32 {
        let w0 = self.depth_normal_weight;
        if self.depth_normal_weight_end <= 0.0 {
            return w0;
        }
        let t = linear_ramp_weight(
            global_iter,
            self.depth_normal_weight_end_start_iter,
            self.depth_normal_weight_end_ramp_iters,
        );
        w0 + (self.depth_normal_weight_end - w0) * t
    }

    /// Cosine threshold for the per-pixel normal contradiction gate at
    /// `global_iter`, or `None` when the gate is off or not yet armed.
    ///
    /// `None` is the pre-gate code path exactly; `normal_loss` takes this
    /// straight through.
    pub fn normal_gate_cos_at(&self, global_iter: u32) -> Option<f32> {
        (self.normal_gate_degrees > 0.0 && global_iter >= self.normal_gate_start_iter)
            .then(|| self.normal_gate_degrees.to_radians().cos())
    }

    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }

    pub fn appearance_enabled(&self) -> bool {
        self.bilateral_grid || self.ppisp
    }
}

/// Linear ramp primitive: `0` before `start`, then
/// `min(1, (step - start + 1) / ramp)`.
///
/// Taken from the `gauss-surf` PGSR trainer (rerun-io/examples-monorepo,
/// Apache-2.0, by Pablo Vela), NOT from the PGSR paper — neither PGSR
/// (arXiv:2406.06521) nor `NeuRIS` (arXiv:2206.13597) specifies a ramp shape at
/// all, let alone this one. Credited here so a reader who wants to question the
/// formula knows where to go and look.
///
/// **The `+ 1` is load-bearing and deliberate.** The ramp is NONZERO at the
/// start step (it is `1/ramp`, not `0`) and saturates at `start + ramp - 1`,
/// which is why `gauss-surf`'s saturation for `start=1400, ramp=875` is
/// step 2274 rather than 2275. An off-by-one here is silent: the run trains
/// fine, one step out of phase, and never exactly reproduces the reference.
///
/// `ramp == 0` degenerates to a hard step to `1.0` at `start`.
pub fn linear_ramp_weight(step: u32, start: u32, ramp: u32) -> f32 {
    if step < start {
        return 0.0;
    }
    if ramp == 0 {
        return 1.0;
    }
    // `step >= start`, so this cannot underflow.
    let progressed = (step - start) as f32 + 1.0;
    (progressed / ramp as f32).min(1.0)
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

// TIDI-GS serde defaults, kept in sync with the clap `default_value`s above so a
// config that omits these fields deserializes to the same values the CLI uses.
fn default_tidi_prune_start_iter() -> u32 {
    500
}
fn default_tidi_prune_every() -> u32 {
    400
}
fn default_tidi_warmup_steps() -> u32 {
    500
}
fn default_tidi_vis_threshold() -> f32 {
    2.0
}
fn default_tidi_opacity_threshold() -> f32 {
    0.04
}
fn default_tidi_importance_threshold() -> f32 {
    0.35
}
fn default_tidi_grad_threshold() -> f32 {
    5e-4
}
fn default_tidi_grad_ema_beta() -> f32 {
    0.99
}
fn default_tidi_importance_reg() -> f32 {
    0.01
}
fn default_tidi_importance_lr() -> f64 {
    0.05
}
fn default_tidi_guard_sh_quantile() -> f32 {
    0.95
}
fn default_tidi_guard_thin_quantile() -> f32 {
    0.10
}

fn default_scale_reg_threshold() -> f32 {
    3.0
}
fn default_tidi_guard_aniso_quantile() -> f32 {
    0.95
}
fn default_tidi_guard_color_var_quantile() -> f32 {
    0.0
}
fn default_tidi_knn_k() -> u32 {
    16
}
fn default_tidi_local_cap_frac() -> f32 {
    0.01
}
fn default_tidi_depth_margin() -> f32 {
    0.05
}
fn default_tidi_depth_float_frac() -> f32 {
    0.5
}
fn default_tidi_depth_min_valid_views() -> u32 {
    4
}
fn default_tidi_depth_cap_frac() -> f32 {
    0.02
}
fn default_depth_opacity_reg_weight() -> f32 {
    0.0
}
fn default_depth_opacity_reg_margin() -> f32 {
    0.15
}
fn default_depth_opacity_reg_softness() -> f32 {
    0.05
}
fn default_depth_opacity_reg_start_iter() -> u32 {
    0
}
fn default_plane_coplanarity_weight() -> f32 {
    0.0
}
fn default_plane_coplanarity_assign_dist() -> f32 {
    -1.0
}
fn default_cloud_prune_dist() -> f32 {
    0.19
}
fn default_cloud_prune_start_iter() -> u32 {
    0
}
fn default_depth_weight_start_iter() -> u32 {
    0
}
fn default_depth_weight_end_iter() -> u32 {
    0
}
fn default_depth_weight_end() -> f32 {
    0.0
}
fn default_depth_grad_sigma() -> f32 {
    0.1
}
fn default_tidi_global_cap_frac() -> f32 {
    0.002
}

/// Serde default for MRNF flags that are ON by default (LFS `mrnf_defaults`
/// parity). Keeps deserialization of configs that omit these fields in sync
/// with the clap `default_value_t = true`.
fn default_true() -> bool {
    true
}

fn default_error_map_growth_threshold() -> f32 {
    // Scene-average anchor on the coverage-weighted mean-error scale (mean-
    // normalized ê has scene mean 1.0); selects gaussians reconstructing worse
    // than average. See `TrainConfig::error_map_growth_threshold` (defect-2 fix).
    1.0
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

    /// cloud-prune TEST (b): default-OFF / byte-inert. A command line that does
    /// not mention it leaves `cloud_prune` false (the gate the trainer keys on to
    /// skip building the grid and to skip the refine union), with the documented
    /// threshold + start-iter defaults. `--cloud-prune` flips it on without
    /// disturbing the defaults.
    #[test]
    fn cloud_prune_defaults_off_and_parses() {
        let def = TrainConfig::default();
        assert!(
            !def.cloud_prune,
            "cloud-prune must default OFF (byte-inert)"
        );
        assert_eq!(def.cloud_prune_dist, 0.19);
        assert_eq!(def.cloud_prune_start_iter, 0);

        // An unrelated flag must not switch it on.
        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert!(!other.cloud_prune);

        // The flag turns it on and the tuning knobs parse.
        let on = TrainConfig::try_parse_from([
            "brush",
            "--cloud-prune",
            "--cloud-prune-dist",
            "0.25",
            "--cloud-prune-start-iter",
            "12000",
        ])
        .expect("cloud-prune flags must parse");
        assert!(on.cloud_prune);
        assert_eq!(on.cloud_prune_dist, 0.25);
        assert_eq!(on.cloud_prune_start_iter, 12000);
    }

    /// Inertness contract for the geometry-prior flags: a command line that
    /// does not mention them must leave all three at exactly 0.0, which is what
    /// the `use_*` gates in the train loop key on. Losing this default silently
    /// changes every existing run.
    #[test]
    fn geometry_prior_weights_default_to_zero_and_parse() {
        let def = TrainConfig::default();
        assert_eq!(def.normal_loss_weight, 0.0);
        assert_eq!(def.depth_normal_weight, 0.0);
        assert_eq!(def.flatten_loss_weight, 0.0);
        assert_eq!(def.normal_smooth_weight, 0.0);
        // The depth half is untouched by this change.
        assert_eq!(def.depth_loss_weight, 0.0);

        // 0 means "never gate", so the consistency term behaves exactly as it
        // did before the gate existed. A nonzero default here would silently
        // DISABLE the term for the first N iterations of every run that sets
        // --depth-normal-weight, which is the opposite failure to the weights
        // above and just as invisible.
        assert_eq!(def.depth_normal_start_iter, 0);

        // Unrelated flags must not switch them on.
        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert_eq!(other.normal_loss_weight, 0.0);
        assert_eq!(other.depth_normal_weight, 0.0);
        assert_eq!(other.flatten_loss_weight, 0.0);
        assert_eq!(other.normal_smooth_weight, 0.0);
        assert_eq!(other.depth_normal_start_iter, 0);

        let on = TrainConfig::try_parse_from([
            "brush",
            "--normal-loss-weight",
            "0.2",
            "--depth-normal-weight",
            "0.05",
            "--flatten-loss-weight",
            "1.0",
            "--normal-smooth-weight",
            "0.5",
            "--depth-normal-start-iter",
            "7000",
        ])
        .expect("geometry-prior flags must parse");
        assert!((on.normal_loss_weight - 0.2).abs() < 1e-9);
        assert!((on.depth_normal_weight - 0.05).abs() < 1e-9);
        assert!((on.flatten_loss_weight - 1.0).abs() < 1e-9);
        assert!((on.normal_smooth_weight - 0.5).abs() < 1e-9);
        assert_eq!(on.depth_normal_start_iter, 7000);

        // There is deliberately NO --flatten-start-iter: DN-Splatter runs the
        // identical scale term ungated at 1.0 from step 0, so a gate would be
        // inventing a schedule no reference implementation uses.
        assert!(TrainConfig::try_parse_from(["brush", "--flatten-start-iter", "7000"]).is_err());
    }

    #[test]
    fn error_map_flags_default_on_and_parse() {
        // Default (LFS `mrnf_defaults` parity): error-map densification ON;
        // τ_err = 1.0 scene-average anchor on the coverage-weighted mean-error
        // scale (defect-2 fix) — tau is unchanged from the LFS-parity flip.
        let def = TrainConfig::default();
        assert!(def.error_map_densification);
        assert!((def.error_map_growth_threshold - 1.0).abs() < 1e-9);
        // The gradient threshold is a SEPARATE knob (different scale), untouched.
        assert!((def.growth_grad_threshold - 0.0025).abs() < 1e-9);

        let on = TrainConfig::try_parse_from([
            "brush",
            "--error-map-densification",
            "--error-map-growth-threshold",
            "0.01",
        ])
        .expect("error-map flags must parse");
        assert!(on.error_map_densification);
        assert!((on.error_map_growth_threshold - 0.01).abs() < 1e-9);

        // Off-switch: default-true MRNF flags must be disable-able per-run via
        // the `--flag=false` value form (require_equals idiom).
        let off = TrainConfig::try_parse_from(["brush", "--error-map-densification=false"])
            .expect("error-map disable form must parse");
        assert!(!off.error_map_densification);
    }

    #[test]
    fn mrnf_lfs_parity_defaults() {
        // Operator decision "default-on MRNF should match LFS": these are the
        // LFS `mrnf_defaults` this fork now ships by default.
        let def = TrainConfig::default();
        assert!(def.mrnf_noise_injection);
        assert!(def.use_edge_map);
        assert!(def.error_map_densification);
        assert!(def.min_scale_prune);
        assert!(def.near_zero_rotation_prune);
        assert!((def.scale_decay - 0.002).abs() < 1e-9);
        assert!((def.growth_select_fraction - 0.07).abs() < 1e-9);
        assert!((def.lr_scale - 7e-3).abs() < 1e-12);
        assert!((def.lr_scale_end - 5e-3).abs() < 1e-12);
        // radial_bounds_prune stays OFF: default matches MRNF's L-inf cull; the
        // flag is a stricter divergence experiment, not parity.
        assert!(!def.radial_bounds_prune);
    }

    #[test]
    fn mrnf_default_on_flags_have_off_switch() {
        // Every flag flipped to default-true MUST remain disable-able from the
        // CLI via the `--flag=false` value form.
        let off = TrainConfig::try_parse_from([
            "brush",
            "--mrnf-noise-injection=false",
            "--use-edge-map=false",
            "--error-map-densification=false",
            "--min-scale-prune=false",
            "--near-zero-rotation-prune=false",
        ])
        .expect("MRNF default-on flags must accept the =false disable form");
        assert!(!off.mrnf_noise_injection);
        assert!(!off.use_edge_map);
        assert!(!off.error_map_densification);
        assert!(!off.min_scale_prune);
        assert!(!off.near_zero_rotation_prune);

        // Bare flag form (as the 5M/aerial recipes pass them) still enables.
        let on = TrainConfig::try_parse_from([
            "brush",
            "--mrnf-noise-injection",
            "--use-edge-map",
            "--error-map-densification",
            "--min-scale-prune",
            "--near-zero-rotation-prune",
        ])
        .expect("bare MRNF flags must still parse to true");
        assert!(on.mrnf_noise_injection);
        assert!(on.use_edge_map);
        assert!(on.error_map_densification);
        assert!(on.min_scale_prune);
        assert!(on.near_zero_rotation_prune);
    }

    /// Inertness contract for the TIDI-GS family: a command line that does not
    /// mention `--tidi-prune` leaves the master switch OFF (so the trainer never
    /// allocates TIDI state and the render/refine paths are byte-identical), and
    /// the paper's Table II constants are the shipped defaults.
    #[test]
    fn tidi_flags_default_off_and_match_paper() {
        let def = TrainConfig::default();
        assert!(
            !def.tidi_prune,
            "TIDI must be off unless explicitly enabled"
        );
        // paper Table II constants
        assert!((def.tidi_vis_threshold - 2.0).abs() < 1e-9);
        assert!((def.tidi_opacity_threshold - 0.04).abs() < 1e-9);
        assert!((def.tidi_importance_threshold - 0.35).abs() < 1e-9);
        assert!((def.tidi_grad_threshold - 5e-4).abs() < 1e-12);
        assert!((def.tidi_grad_ema_beta - 0.99).abs() < 1e-9);
        assert_eq!(def.tidi_knn_k, 16);
        assert!((def.tidi_local_cap_frac - 0.01).abs() < 1e-9);
        assert!((def.tidi_global_cap_frac - 0.002).abs() < 1e-9);

        // Depth-prune path: also OFF by default, with the documented depth
        // constants. Byte-inertness of the whole TIDI family keys on BOTH master
        // switches being false.
        assert!(
            !def.tidi_depth_prune,
            "depth-prune must be off unless explicitly enabled"
        );
        assert!((def.tidi_depth_margin - 0.05).abs() < 1e-9);
        assert!((def.tidi_depth_float_frac - 0.5).abs() < 1e-9);
        assert_eq!(def.tidi_depth_min_valid_views, 4);
        assert!((def.tidi_depth_cap_frac - 0.02).abs() < 1e-9);

        // Depth-coupled opacity regularizer: OFF (weight 0 = inert, no loss
        // term / no projection) by default, with the documented ramp constants.
        assert!(
            (def.depth_opacity_reg_weight - 0.0).abs() < 1e-9,
            "depth-opacity-reg must be off (weight 0) by default"
        );
        // margin / softness are now in SCENE 3D-distance units (distance-to-cloud
        // gate), defaulting to 0.15 / 0.05.
        assert!((def.depth_opacity_reg_margin - 0.15).abs() < 1e-9);
        // softness MUST default to < margin so the ramp reaches ~0 by the surface
        // (d=0); 0.05 vs margin 0.15 gives p(0) ~ 0.047.
        assert!((def.depth_opacity_reg_softness - 0.05).abs() < 1e-9);
        assert!(
            def.depth_opacity_reg_softness < def.depth_opacity_reg_margin,
            "softness must stay below margin or the ramp fades on-surface splats"
        );
        // The start-iter gate defaults to 0 (active from step 0 = unchanged).
        assert_eq!(def.depth_opacity_reg_start_iter, 0);

        // Plane priors (FIX 1 + FIX 2): both OFF by default and byte-inert. With
        // --plane-gate false AND --plane-coplanarity-weight 0, no RANSAC runs and
        // the run is identical to the pre-plane branch.
        assert!(!def.plane_gate, "plane-gate must default off");
        assert!(
            (def.plane_coplanarity_weight - 0.0).abs() < 1e-9,
            "co-planarity must default off (weight 0)"
        );
        // assign-dist sentinel <= 0 means "derive the band from the seed cloud's
        // measured NN spacing" (tidi::resolve_coplanarity_assign_dist). It used
        // to mean "fall back to --depth-opacity-reg-margin", which was a
        // different feature's knob in absolute units — see that function.
        assert!(def.plane_coplanarity_assign_dist <= 0.0);

        // Unrelated flags must not switch it on.
        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert!(!other.tidi_prune);
        assert!(!other.tidi_depth_prune);
        assert!((other.depth_opacity_reg_weight - 0.0).abs() < 1e-9);

        // Bare `--tidi-prune` (a presence flag) enables it. There is deliberately
        // no `--tidi-prune=false` form: the master switch is SetTrue so that the
        // common `brush --tidi-prune` (no value) works; absence is the off state.
        let on = TrainConfig::try_parse_from(["brush", "--tidi-prune"])
            .expect("--tidi-prune must parse");
        assert!(on.tidi_prune);
        // Enabling the photometric path must NOT enable the depth path.
        assert!(!on.tidi_depth_prune);

        // The depth path is independent: `--tidi-depth-prune` on its own enables
        // ONLY the depth path (photometric stays off).
        let depth_on = TrainConfig::try_parse_from(["brush", "--tidi-depth-prune"])
            .expect("--tidi-depth-prune must parse");
        assert!(depth_on.tidi_depth_prune);
        assert!(
            !depth_on.tidi_prune,
            "depth path must not imply photometric"
        );

        // The depth-coupled opacity regularizer is a value flag, independent of
        // both prune switches: setting its weight enables it alone.
        let opacreg = TrainConfig::try_parse_from([
            "brush",
            "--depth-opacity-reg-weight",
            "0.5",
            "--depth-opacity-reg-margin",
            "0.15",
        ])
        .expect("--depth-opacity-reg-weight must parse");
        assert!((opacreg.depth_opacity_reg_weight - 0.5).abs() < 1e-9);
        assert!((opacreg.depth_opacity_reg_margin - 0.15).abs() < 1e-9);
        assert!(
            !opacreg.tidi_prune,
            "opacity-reg must not imply photometric"
        );
        assert!(
            !opacreg.tidi_depth_prune,
            "opacity-reg must not imply depth-prune"
        );

        // The start-iter gate parses and is honoured (used to defer the term
        // until densification stops).
        let gated = TrainConfig::try_parse_from([
            "brush",
            "--depth-opacity-reg-weight",
            "0.1",
            "--depth-opacity-reg-start-iter",
            "15000",
        ])
        .expect("--depth-opacity-reg-start-iter must parse");
        assert_eq!(gated.depth_opacity_reg_start_iter, 15000);

        // Plane flags parse independently and do not imply any other switch.
        // 20.0 is the measured working weight on a metric scene, not an
        // arbitrary literal: the term is metres² where flatten is metres, so the
        // magnitude here is the point (see the flag doc).
        let plane = TrainConfig::try_parse_from([
            "brush",
            "--plane-gate",
            "--plane-coplanarity-weight",
            "20.0",
        ])
        .expect("--plane-gate / --plane-coplanarity-weight must parse");
        assert!(plane.plane_gate);
        assert!((plane.plane_coplanarity_weight - 20.0).abs() < 1e-9);
        assert!(!plane.tidi_prune, "plane flags must not imply photometric");
        assert!(
            (plane.depth_opacity_reg_weight - 0.0).abs() < 1e-9,
            "plane flags must not imply the opacity regularizer"
        );
        // --plane-gate is a presence flag; absence leaves it off.
        let no_plane = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert!(!no_plane.plane_gate);
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

    /// `--depth-loss-space`: default-inert and round-trips.
    ///
    /// The default MUST be `Disparity`. It is not a style preference: every
    /// recorded figure in the fork's ablations — the playroom 15k baseline, the
    /// `ARKitScenes` matrix, the `center` step-0 identity hash — was measured
    /// with a disparity residual, and flipping the default would silently
    /// reinterpret `--depth-loss-weight` by a factor of `d²` in every existing
    /// recipe.
    #[test]
    fn depth_loss_space_defaults_to_disparity_and_parses() {
        assert_eq!(
            TrainConfig::default().depth_loss_space,
            DepthLossSpace::Disparity
        );
        assert_eq!(DepthLossSpace::default(), DepthLossSpace::Disparity);

        // An unrelated flag must not disturb it — including one that touches
        // the same weight the space reinterprets.
        let other = TrainConfig::try_parse_from(["brush", "--depth-loss-weight", "1.2"])
            .expect("unrelated flags must parse");
        assert_eq!(other.depth_loss_space, DepthLossSpace::Disparity);

        for (arg, want) in [
            ("disparity", DepthLossSpace::Disparity),
            ("metric", DepthLossSpace::Metric),
        ] {
            let cfg = TrainConfig::try_parse_from(["brush", "--depth-loss-space", arg])
                .expect("--depth-loss-space must parse");
            assert_eq!(cfg.depth_loss_space, want, "--depth-loss-space {arg}");
            // serde round-trip, in the kebab-case spelling the CLI uses.
            let json = serde_json::to_string(&cfg.depth_loss_space).expect("serialize");
            assert_eq!(json, format!("\"{arg}\""));
            let back: DepthLossSpace = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, want);
        }

        // Serde default: a config file written before this flag existed still
        // loads, and loads as disparity. Built by round-tripping a default
        // config with the key REMOVED, which is what an older file looks like —
        // `TrainConfig` is not fully serde-defaulted, so a bare `{}` would fail
        // on unrelated required fields and prove nothing.
        let mut value =
            serde_json::to_value(TrainConfig::default()).expect("config must serialize");
        let removed = value
            .as_object_mut()
            .expect("config serializes to a JSON object")
            .remove("depth-loss-space");
        assert!(
            removed.is_some(),
            "the field must serialize under this key, or the removal below is a no-op \
             and this test would pass vacuously"
        );
        let from_json: TrainConfig =
            serde_json::from_value(value).expect("a config without the key must deserialize");
        assert_eq!(from_json.depth_loss_space, DepthLossSpace::Disparity);
    }

    /// **`--depth-uncovered` default-inertness + round-trip** (plan §5, T6).
    ///
    /// `count` must stay the default for the same reason `disparity` does: every
    /// recorded ablation figure — the playroom 15k baseline, the `ARKitScenes`
    /// matrix, the `center` step-0 identity hash — was measured with uncovered
    /// pixels counted in both sums, and any other default would silently move
    /// the reported depth loss (and, under `exclude`, the effective depth weight)
    /// in every existing recipe.
    #[test]
    fn depth_uncovered_defaults_to_count_and_parses() {
        assert_eq!(
            TrainConfig::default().depth_uncovered,
            DepthUncovered::Count
        );
        assert_eq!(DepthUncovered::default(), DepthUncovered::Count);

        // An unrelated flag must not disturb it — including the two it composes
        // with most closely.
        let other = TrainConfig::try_parse_from([
            "brush",
            "--depth-loss-weight",
            "1.2",
            "--depth-loss-space",
            "metric",
        ])
        .expect("unrelated flags must parse");
        assert_eq!(other.depth_uncovered, DepthUncovered::Count);

        for (arg, want) in [
            ("count", DepthUncovered::Count),
            ("exclude-numerator", DepthUncovered::ExcludeNumerator),
            ("exclude", DepthUncovered::Exclude),
        ] {
            let cfg = TrainConfig::try_parse_from(["brush", "--depth-uncovered", arg])
                .expect("--depth-uncovered must parse");
            assert_eq!(cfg.depth_uncovered, want, "--depth-uncovered {arg}");
            // serde round-trip, in the kebab-case spelling the CLI uses.
            let json = serde_json::to_string(&cfg.depth_uncovered).expect("serialize");
            assert_eq!(json, format!("\"{arg}\""));
            let back: DepthUncovered = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, want);
        }

        // Serde default: a config file written before this flag existed still
        // loads, and loads as `count`. Same construction as the
        // `--depth-loss-space` test above — round-trip a default config with the
        // key REMOVED, since a bare `{}` would fail on unrelated required fields
        // and prove nothing.
        let mut value =
            serde_json::to_value(TrainConfig::default()).expect("config must serialize");
        let removed = value
            .as_object_mut()
            .expect("config serializes to a JSON object")
            .remove("depth-uncovered");
        assert!(
            removed.is_some(),
            "the field must serialize under this key, or the removal below is a no-op \
             and this test would pass vacuously"
        );
        let from_json: TrainConfig =
            serde_json::from_value(value).expect("a config without the key must deserialize");
        assert_eq!(from_json.depth_uncovered, DepthUncovered::Count);
    }

    /// Depth-anneal + grad-aware flags: default-inert (byte-identical to the
    /// pre-change behaviour), unrelated flags leave them, all six flags
    /// round-trip on the CLI, and `validate()` rejects the two invalid combos.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn depth_anneal_and_grad_aware_default_noop_and_parse() {
        let def = TrainConfig::default();
        assert_eq!(def.depth_weight_start_iter, 0);
        assert_eq!(def.depth_weight_end_iter, 0);
        assert_eq!(def.depth_weight_end, 0.0);
        assert_eq!(def.depth_weight_decay, DepthWeightDecay::Linear);
        assert!(!def.depth_grad_aware);
        assert!((def.depth_grad_sigma - 0.1).abs() < 1e-9);

        // An unrelated flag must not disturb any of them.
        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert_eq!(other.depth_weight_start_iter, 0);
        assert_eq!(other.depth_weight_end_iter, 0);
        assert_eq!(other.depth_weight_end, 0.0);
        assert_eq!(other.depth_weight_decay, DepthWeightDecay::Linear);
        assert!(!other.depth_grad_aware);
        assert!((other.depth_grad_sigma - 0.1).abs() < 1e-9);

        // All six flags parse and round-trip.
        let on = TrainConfig::try_parse_from([
            "brush",
            "--depth-loss-weight",
            "1.0",
            "--depth-weight-start-iter",
            "100",
            "--depth-weight-end-iter",
            "300",
            "--depth-weight-end",
            "0.25",
            "--depth-weight-decay",
            "cosine",
            "--depth-grad-aware",
            "--depth-grad-sigma",
            "0.2",
        ])
        .expect("depth-anneal / grad-aware flags must parse");
        assert_eq!(on.depth_weight_start_iter, 100);
        assert_eq!(on.depth_weight_end_iter, 300);
        assert!((on.depth_weight_end - 0.25).abs() < 1e-9);
        assert_eq!(on.depth_weight_decay, DepthWeightDecay::Cosine);
        assert!(on.depth_grad_aware);
        assert!((on.depth_grad_sigma - 0.2).abs() < 1e-9);

        // validate(): end-iter <= start-iter (nonzero) is rejected.
        let mut bad = TrainConfig::default();
        bad.depth_weight_start_iter = 300;
        bad.depth_weight_end_iter = 300;
        assert!(bad.validate().is_err());

        // validate(): sigma <= 0 with grad-aware on is rejected.
        let mut bad_sigma = TrainConfig::default();
        bad_sigma.depth_grad_aware = true;
        bad_sigma.depth_grad_sigma = 0.0;
        assert!(bad_sigma.validate().is_err());
    }

    /// `depth_weight_at` schedule pins. With defaults it is the byte-identity
    /// constant `depth_loss_weight`; with a schedule set it interpolates.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn depth_weight_schedule_pins() {
        // Default cfg: annealing OFF, exact byte-identity to depth_loss_weight.
        let def = TrainConfig::default();
        for i in [0u32, 7000, u32::MAX] {
            assert_eq!(def.depth_weight_at(i), def.depth_loss_weight);
        }

        // Linear schedule w0=1.0 over [100, 300] down to 0.0.
        let mut lin = TrainConfig::default();
        lin.depth_loss_weight = 1.0;
        lin.depth_weight_start_iter = 100;
        lin.depth_weight_end_iter = 300;
        lin.depth_weight_end = 0.0;
        lin.depth_weight_decay = DepthWeightDecay::Linear;
        assert!((lin.depth_weight_at(0) - 1.0).abs() < 1e-6);
        assert!((lin.depth_weight_at(100) - 1.0).abs() < 1e-6);
        assert!((lin.depth_weight_at(200) - 0.5).abs() < 1e-6);
        assert!((lin.depth_weight_at(300) - 0.0).abs() < 1e-6);
        assert!((lin.depth_weight_at(10_000) - 0.0).abs() < 1e-6);

        // Cosine schedule, same endpoints.
        let mut cos = lin.clone();
        cos.depth_weight_decay = DepthWeightDecay::Cosine;
        assert!((cos.depth_weight_at(200) - 0.5).abs() < 1e-6);
        // t = 0.25 -> 0.5 * (1 + cos(pi/4)) = 0.853553...
        assert!((cos.depth_weight_at(150) - 0.853_553_4).abs() < 1e-6);

        // Nonzero end weight pins the endpoints for both shapes.
        let mut lin_nz = lin;
        lin_nz.depth_weight_end = 0.25;
        assert!((lin_nz.depth_weight_at(100) - 1.0).abs() < 1e-6);
        assert!((lin_nz.depth_weight_at(300) - 0.25).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // PGSR plane-render config surface (WS-L). Every one of these pins the
    // DEFAULT to a value that leaves the trainer byte-identical to its
    // pre-change behaviour; the parse half proves the flag is reachable.
    // ------------------------------------------------------------------

    /// `--depth-source` defaults to `center` (the alpha-composited camera-z of
    /// splat means, i.e. exactly what the trainer did before the plane paths
    /// existed) and round-trips all three values.
    #[test]
    fn depth_source_defaults_to_center_and_parses() {
        let def = TrainConfig::default();
        assert_eq!(def.depth_source, DepthSource::Center);

        // An unrelated flag must not disturb it.
        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert_eq!(other.depth_source, DepthSource::Center);

        for (arg, want) in [
            ("center", DepthSource::Center),
            ("plane-aux", DepthSource::PlaneAux),
            ("plane-fused", DepthSource::PlaneFused),
        ] {
            let cfg = TrainConfig::try_parse_from(["brush", "--depth-source", arg])
                .unwrap_or_else(|e| panic!("--depth-source {arg} must parse: {e}"));
            assert_eq!(cfg.depth_source, want);
        }

        assert!(TrainConfig::try_parse_from(["brush", "--depth-source", "plane"]).is_err());
    }

    /// The raw ramp primitive, pinned on its own because of the `+ 1`. The
    /// reference's ramp is `min(1, (step - start + 1) / ramp)`, which is
    /// NONZERO at the start step. An off-by-one here is silent: the run still
    /// trains, it just supervises on a schedule that is one step out of phase
    /// and never exactly reproduces the reference.
    #[test]
    fn linear_ramp_weight_is_nonzero_at_the_start_step() {
        // Before the start: hard zero.
        assert_eq!(linear_ramp_weight(1399, 1400, 875), 0.0);
        // AT the start: 1/875, not 0.
        assert!((linear_ramp_weight(1400, 1400, 875) - 1.0 / 875.0).abs() < 1e-9);
        // Saturates at start + ramp - 1 = 2274 (the reference's stated value).
        assert!((linear_ramp_weight(2273, 1400, 875) - 874.0 / 875.0).abs() < 1e-6);
        assert_eq!(linear_ramp_weight(2274, 1400, 875), 1.0);
        assert_eq!(linear_ramp_weight(u32::MAX, 1400, 875), 1.0);
        // ramp == 0 is a hard step at `start`.
        assert_eq!(linear_ramp_weight(99, 100, 0), 0.0);
        assert_eq!(linear_ramp_weight(100, 100, 0), 1.0);
    }

    /// L1 ramp: default-inert, unrelated flags leave it, all five flags parse,
    /// and `validate()` rejects the two nonsense combos.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn normal_ramp_defaults_noop_and_parse() {
        let def = TrainConfig::default();
        assert_eq!(def.normal_ramp_start_iter, 0);
        assert_eq!(def.normal_ramp_iters, 0);
        assert_eq!(def.depth_normal_weight_end, 0.0);
        assert_eq!(def.depth_normal_weight_end_start_iter, 0);
        assert_eq!(def.depth_normal_weight_end_ramp_iters, 0);

        // Defaults: the ramp is EXACTLY 1.0 everywhere, so the two normal
        // weights are multiplied by an exact identity.
        for i in [0u32, 15_000, 30_000, u32::MAX] {
            assert_eq!(def.normal_ramp_at(i), 1.0);
        }

        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert_eq!(other.normal_ramp_start_iter, 0);
        assert_eq!(other.normal_ramp_iters, 0);
        assert_eq!(other.depth_normal_weight_end, 0.0);
        for i in [0u32, 15_000, u32::MAX] {
            assert_eq!(other.normal_ramp_at(i), 1.0);
        }

        let on = TrainConfig::try_parse_from([
            "brush",
            "--normal-ramp-start-iter",
            "3000",
            "--normal-ramp-iters",
            "1875",
            "--depth-normal-weight-end",
            "0.055",
            "--depth-normal-weight-end-start-iter",
            "11800",
            "--depth-normal-weight-end-ramp-iters",
            "1050",
        ])
        .expect("normal-ramp flags must parse");
        assert_eq!(on.normal_ramp_start_iter, 3000);
        assert_eq!(on.normal_ramp_iters, 1875);
        assert!((on.depth_normal_weight_end - 0.055).abs() < 1e-9);
        assert_eq!(on.depth_normal_weight_end_start_iter, 11800);
        assert_eq!(on.depth_normal_weight_end_ramp_iters, 1050);

        // validate(): a ramp length with no start is the silent-no-op trap.
        let mut orphan = TrainConfig::default();
        orphan.normal_ramp_iters = 1875;
        assert!(orphan.validate().is_err());

        // validate(): a negative end weight is rejected.
        let mut neg = TrainConfig::default();
        neg.depth_normal_weight_end = -0.1;
        assert!(neg.validate().is_err());
    }

    /// `normal_ramp_at` schedule pins at the documented 15k recipe
    /// (`--normal-ramp-start-iter 3000 --normal-ramp-iters 1875`).
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn normal_ramp_schedule_pins() {
        let mut cfg = TrainConfig::default();
        cfg.normal_ramp_start_iter = 3000;
        cfg.normal_ramp_iters = 1875;

        assert_eq!(cfg.normal_ramp_at(0), 0.0);
        assert_eq!(cfg.normal_ramp_at(2999), 0.0);
        // Nonzero AT the start step: (3000 - 3000 + 1) / 1875.
        assert!((cfg.normal_ramp_at(3000) - 1.0 / 1875.0).abs() < 1e-9);
        // Midpoint: (3937 - 3000 + 1) / 1875 = 938/1875.
        assert!((cfg.normal_ramp_at(3937) - 938.0 / 1875.0).abs() < 1e-6);
        // Saturates at start + ramp - 1.
        assert_eq!(cfg.normal_ramp_at(3000 + 1875 - 1), 1.0);
        assert_eq!(cfg.normal_ramp_at(15_000), 1.0);

        // ramp 0 = hard step at start.
        let mut step = TrainConfig::default();
        step.normal_ramp_start_iter = 3000;
        assert_eq!(step.normal_ramp_at(2999), 0.0);
        assert_eq!(step.normal_ramp_at(3000), 1.0);
    }

    /// `depth_normal_weight_at`: identity at defaults, and the reference's
    /// late consistency bump (0.50 -> 0.55 from 5500 over 500) when armed.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn depth_normal_weight_schedule_pins() {
        let def = TrainConfig::default();
        for i in [0u32, 15_000, u32::MAX] {
            assert_eq!(def.depth_normal_weight_at(i), def.depth_normal_weight);
        }

        // Even with a base weight set, end == 0.0 means OFF -> constant.
        let mut base_only = TrainConfig::default();
        base_only.depth_normal_weight = 0.05;
        for i in [0u32, 5_500, u32::MAX] {
            assert_eq!(base_only.depth_normal_weight_at(i), 0.05);
        }

        // The reference bump, verbatim.
        let mut bump = TrainConfig::default();
        bump.depth_normal_weight = 0.50;
        bump.depth_normal_weight_end = 0.55;
        bump.depth_normal_weight_end_start_iter = 5500;
        bump.depth_normal_weight_end_ramp_iters = 500;
        assert!((bump.depth_normal_weight_at(5499) - 0.50).abs() < 1e-6);
        // (5500 - 5500 + 1)/500 = 0.002 -> 0.50 + 0.05*0.002
        assert!((bump.depth_normal_weight_at(5500) - (0.50 + 0.05 * 0.002)).abs() < 1e-6);
        // Saturates at 5999 per the reference.
        assert!((bump.depth_normal_weight_at(5999) - 0.55).abs() < 1e-6);
        assert!((bump.depth_normal_weight_at(15_000) - 0.55).abs() < 1e-6);
    }

    /// L2 gate: default-inert and parses; `validate()` bounds the angle.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn normal_gate_defaults_noop_and_parse() {
        let def = TrainConfig::default();
        assert_eq!(def.normal_gate_degrees, 0.0);
        assert_eq!(def.normal_gate_start_iter, 0);
        // 0 degrees = OFF -> no cosine threshold is ever produced.
        for i in [0u32, 15_000, u32::MAX] {
            assert_eq!(def.normal_gate_cos_at(i), None);
        }

        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert_eq!(other.normal_gate_degrees, 0.0);
        assert_eq!(other.normal_gate_start_iter, 0);

        let on = TrainConfig::try_parse_from([
            "brush",
            "--normal-gate-degrees",
            "30",
            "--normal-gate-start-iter",
            "5600",
        ])
        .expect("normal-gate flags must parse");
        assert!((on.normal_gate_degrees - 30.0).abs() < 1e-9);
        assert_eq!(on.normal_gate_start_iter, 5600);
        assert_eq!(on.normal_gate_cos_at(5599), None);
        let cos30 = on
            .normal_gate_cos_at(5600)
            .expect("gate is armed from its start iter");
        assert!((cos30 - 30.0_f32.to_radians().cos()).abs() < 1e-6);

        // §10d item 8: the boundary itself, stated in the unit the flag is
        // written in. A 30-DEGREE gate must admit 29 degrees and reject 31.
        // The comparison above is self-referential (it recomputes the same
        // expression), so it cannot catch a caller-visible unit slip on its
        // own; this can. A `cos(30)` that forgot `to_radians()` is 0.154, which
        // both of these cosines clear, so the second assertion is the one that
        // fires. 0.0086 separates cos(29) from cos(31) — about 7e4 f32 epsilons,
        // so no rounding argument can move a normal across this line.
        assert!(
            29.0_f32.to_radians().cos() >= cos30,
            "a 30-degree gate must admit a 29-degree disagreement"
        );
        assert!(
            31.0_f32.to_radians().cos() < cos30,
            "a 30-degree gate must reject a 31-degree disagreement"
        );

        // validate(): out-of-range angles are rejected.
        for bad_deg in [-1.0, 180.5, f32::NAN] {
            let mut bad = TrainConfig::default();
            bad.normal_gate_degrees = bad_deg;
            assert!(
                bad.validate().is_err(),
                "accepted normal-gate-degrees {bad_deg}"
            );
        }
    }

    /// L3 metric-weight normalization: default OFF and parses.
    #[test]
    fn normalize_metric_weights_defaults_off_and_parse() {
        let def = TrainConfig::default();
        assert!(!def.normalize_metric_weights);

        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert!(!other.normalize_metric_weights);

        let on = TrainConfig::try_parse_from(["brush", "--normalize-metric-weights"])
            .expect("--normalize-metric-weights must parse");
        assert!(on.normalize_metric_weights);
    }

    /// `--allow-nonfinite-loss` defaults OFF, i.e. a non-finite loss ABORTS.
    ///
    /// Note this is the one flag in this file whose inert default is not the
    /// pre-change behaviour: before the guard existed, a NaN loss was silently
    /// trained through. The default pinned here is the new, strict behaviour;
    /// the flag exists to restore the old one deliberately rather than by
    /// accident. See the field's doc comment.
    #[test]
    fn allow_nonfinite_loss_defaults_off_and_parse() {
        let def = TrainConfig::default();
        assert!(
            !def.allow_nonfinite_loss,
            "a non-finite loss must abort by default"
        );

        let other = TrainConfig::try_parse_from(["brush", "--total-train-iters", "100"])
            .expect("unrelated flags must parse");
        assert!(!other.allow_nonfinite_loss);

        let on = TrainConfig::try_parse_from(["brush", "--allow-nonfinite-loss"])
            .expect("--allow-nonfinite-loss must parse");
        assert!(on.allow_nonfinite_loss);

        // Serde default: a config file written before this flag existed still
        // loads, and loads as the strict setting. Built by round-tripping a
        // default config with the key REMOVED, which is exactly what an older
        // file looks like — `TrainConfig` is not fully serde-defaulted, so a
        // bare `{}` would fail on unrelated required fields and prove nothing.
        let mut value =
            serde_json::to_value(TrainConfig::default()).expect("config must serialize");
        let removed = value
            .as_object_mut()
            .expect("config serializes to a JSON object")
            .remove("allow-nonfinite-loss");
        assert!(
            removed.is_some(),
            "the field must serialize under this key, or the removal below is a no-op \
             and this test would pass vacuously"
        );
        let from_json: TrainConfig =
            serde_json::from_value(value).expect("a config without the key must deserialize");
        assert!(!from_json.allow_nonfinite_loss);
    }
}
