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

    /// Weight of l1 loss on depth (disparity-space)
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    pub depth_loss_weight: f32,

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
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    #[serde(default)]
    pub flatten_loss_weight: f32,

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
    /// this gaussian was ever visible is <= this (paper τ_vis = 2.0). NOTE: the
    /// unit is refine windows, not steps -- see `TidiState::accumulate_window`,
    /// which collapses each window to a 0/1 "seen" indicator so this threshold is
    /// reachable (raw per-step counts scale with training length).
    #[arg(long, help_heading = "TIDI options", default_value = "2.0")]
    #[serde(default = "default_tidi_vis_threshold")]
    pub tidi_vis_threshold: f32,

    /// Opacity signal: candidate iff opacity <= this (paper τ_α = 0.04).
    #[arg(long, help_heading = "TIDI options", default_value = "0.04")]
    #[serde(default = "default_tidi_opacity_threshold")]
    pub tidi_opacity_threshold: f32,

    /// Learned-importance signal: candidate iff sigmoid(omega_i) <= this
    /// (paper τ_ω = 0.35).
    #[arg(long, help_heading = "TIDI options", default_value = "0.35")]
    #[serde(default = "default_tidi_importance_threshold")]
    pub tidi_importance_threshold: f32,

    /// Position-gradient EMA signal: candidate iff EMA <= this (paper
    /// τ_grad = 5e-4).
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
    /// ||f_rest|| is at/above this quantile of the STABLE set (specular/detail).
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
    /// LiDAR/metric depth (the default 0.05 = 5 cm); an SfM dataset's depth may
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
    /// path from touching unscanned regions (no LiDAR return -> exempt).
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
    // they vanish. Reuses the exact per-frame depth + pinhole projection the
    // depth-prune uses. Independent of `--depth-loss-weight` and of the TIDI
    // prune state; needs no persistent accumulators. Default OFF and byte-inert
    // (no projection, no loss term) when the weight is 0.
    // ------------------------------------------------------------------
    /// Depth-coupled opacity-regularizer weight (lambda). 0 = OFF (inert).
    /// >0 adds `lambda * mean_over_valid(p_i * sigmoid(opacity_i))` to the loss,
    /// where `p_i` is a DETACHED smooth ramp that is ~1 for a Gaussian floating
    /// well in front of a valid LiDAR/depth return and 0 on/behind the surface
    /// (or where no return exists). The gradient reaches ONLY the opacity leaf,
    /// so floating splats fade smoothly rather than being hard-deleted.
    #[arg(long, help_heading = "TIDI options", default_value = "0.0")]
    #[serde(default = "default_depth_opacity_reg_weight")]
    pub depth_opacity_reg_weight: f32,

    /// Depth-opacity-reg margin: a Gaussian must sit more than this in FRONT of
    /// the measured depth before its opacity starts being penalized. Units are
    /// the DEPTH map's units -- metres for LiDAR/metric depth (default 0.05 =
    /// 5 cm); an SfM dataset's depth may be non-metric, so scale accordingly.
    #[arg(long, help_heading = "TIDI options", default_value = "0.05")]
    #[serde(default = "default_depth_opacity_reg_margin")]
    pub depth_opacity_reg_margin: f32,

    /// Depth-opacity-reg softness: the width of the sigmoid ramp (in depth
    /// units) over which the penalty weight climbs from ~0 to ~1 as a Gaussian
    /// moves further in front of the surface. Smaller = sharper on/off boundary.
    #[arg(long, help_heading = "TIDI options", default_value = "0.1")]
    #[serde(default = "default_depth_opacity_reg_softness")]
    pub depth_opacity_reg_softness: f32,
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
    0.05
}
fn default_depth_opacity_reg_softness() -> f32 {
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
        assert!((def.depth_opacity_reg_margin - 0.05).abs() < 1e-9);
        assert!((def.depth_opacity_reg_softness - 0.1).abs() < 1e-9);

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
