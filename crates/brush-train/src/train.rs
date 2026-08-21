use crate::{
    adam_scaled::{AdamScaled, AdamState},
    config::{DepthSource, TrainConfig},
    dig::{self, DigTrainState},
    edge, error_map,
    min_scale::compute_min_scale,
    msg::{RefineStats, TrainStepStats},
    multinomial::multinomial_sample,
    quat_vec::quaternion_vec_multiply,
    splat_init::bounds_from_pos,
    stats::RefineRecord,
    tidi::{
        CloudDistanceGrid, PlaneSet, TidiPruneParams, TidiState, extract_planes_from_cloud,
        opacity_reg_active, plane_coplanarity_loss,
    },
};
use brush_appearance::{AppearanceConfig, AppearanceTrainState};
use brush_dataset::scene::SceneBatch;
use brush_loss::{
    ImageLossConfig, depth_loss, depth_normal_loss, image_loss, normal_loss, normal_smooth_loss,
    normals_from_depth, plane_depth_from_features, rgb_grad_weight,
};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterizationMode, Splats, fold_min_scale};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{AlphaMode, bounding_box::BoundingBox, sh::sh_coeffs_for_degree};
// bwd crate dissolved into brush-render/src/bwd/ (upstream #517).
use brush_render::bwd::{
    DeferredShGrad, render_splat_features, render_splats_for_training_with_plane_aux,
};
// The `plane_aux = None` convenience wrapper. The trainer itself always
// goes through the `_with_plane_aux` entry point (it is the same call for
// `None`); the test modules below use the short form.
#[cfg(test)]
use brush_render::bwd::render_splats_for_training;
use brush_render::kernels::helpers::{PLANE_AUX_LANES_USIZE, plane_channel_offset};
use burn::{
    module::{AutodiffModule, Param, ParamId},
    optim::GradientsParams,
    tensor::{
        Bool, Device, Distribution, Gradients, IndexingUpdateOp, Int, Tensor, TensorData,
        activation::sigmoid, s,
    },
};

use hashbrown::HashSet;
use rand::SeedableRng;
use tracing::{Instrument, trace_span};

/// Exponential learning-rate schedule: `lr(n) = initial_lr · gamma^n`, advanced
/// one step per `step()`. A behaviour-exact local replica of burn's
/// `ExponentialLrScheduler` (whose `Config::init()` returns a `ModuleLrScheduler`
/// under burn 0.22, no longer the raw scalar scheduler we need). Seeding
/// `previous_lr = initial_lr / gamma` makes the first `step()` return
/// `initial_lr`, exactly as burn's `build()` does.
#[derive(Clone, Copy, Debug)]
struct ExpLrScheduler {
    previous_lr: f64,
    gamma: f64,
}

impl ExpLrScheduler {
    fn new(initial_lr: f64, gamma: f64) -> Self {
        // burn's ExponentialLrSchedulerConfig::build() rejects these ranges; the
        // local replica must keep the same fail-fast guarantee (a mis-set
        // lr_*_end > lr_* yields gamma > 1, i.e. a GROWING lr, silently).
        assert!(
            initial_lr > 0.0,
            "ExpLrScheduler initial_lr must be > 0, got {initial_lr}"
        );
        assert!(
            gamma > 0.0 && gamma <= 1.0,
            "ExpLrScheduler gamma must be in (0, 1], got {gamma}"
        );
        Self {
            previous_lr: initial_lr / gamma,
            gamma,
        }
    }

    fn step(&mut self) -> f64 {
        self.previous_lr *= self.gamma;
        self.previous_lr
    }
}

/// Default robust-AABB percentile, used for the one-time initial/LOD bounds by
/// external callers. The per-refine bounds recompute inside the trainer uses the
/// configurable `TrainConfig::bounds_percentile` (default matches this).
pub const BOUND_PERCENTILE: f32 = 0.8;

/// Mip-Splatting 3D-filter strength. This is intentionally fixed: changing it
/// alters the learned/exported representation rather than just training speed.
const MIN_SCALE_FACTOR: f32 = 0.1;

/// Target number of GT views sampled per refine window for edge guidance
/// (MRNF port, delta #4; LFS `MRNF_EDGE_MIN_VIEW_SAMPLES = 10`, mrnf.cpp:69).
/// The trainer samples every `refine_every / this` steps so a full window
/// contributes roughly this many views to the per-gaussian edge accumulator.
const EDGE_MIN_VIEW_SAMPLES: u32 = 10;

/// Surviving-pixel fraction below which the normal contradiction gate
/// (`--normal-gate-degrees`) is suspected of over-masking rather than doing its
/// job (plan §4.7).
///
/// What this detects: the gate is meant to drop LOCALLY contradicted pixels —
/// transients, reflections, isolated prior-model failures — inside a frame whose
/// prior is broadly correct. If it is instead discarding most of the frame, the
/// prior and the render disagree systematically, and the normal loss is being
/// silently starved of supervision rather than cleaned up.
///
/// **Nothing else in the pipeline can see this state**, which is what makes the
/// guard load-bearing rather than decorative. Verified by reading
/// `ingest/splatcam/normals_moge.py:120-166` directly:
///
/// * That check runs at prior-GENERATION time and compares `MoGe`'s normals
///   against normals DIFFERENTIATED FROM DEPTH — never against the trained
///   renderer's output. It cannot observe prior-vs-geometry disagreement at all.
/// * It reduces to a median across per-frame medians, so it is doubly robust to
///   outliers by construction and insensitive to any single bad frame.
/// * It only aborts when that median is ANTI-correlated past `-min_cos`; a
///   near-zero median passes. And it writes every `.tiff` before evaluating,
///   so "it refuses to write a bad prior" is not accurate either.
///
/// So a miscalibrated threshold — or a gate armed before the renderer's normals
/// are plausible — sails through that check untouched. And it is invisible in
/// the trainer too: the masked-mean denominator is the GATED count, so the loss
/// magnitude stays perfectly normal while the supervision behind it collapses.
/// There is no loss-curve signal. This log is the only signal.
const NORMAL_GATE_LOW_FRACTION: f32 = 0.20;

/// Consecutive low samples before the over-masking warning fires.
///
/// "Sustained" per plan §4.7: a single low frame is ordinary (a close-up, a
/// transient filling the view) and must not warn. A run of them is the
/// systematic case. One good sample resets the counter.
///
/// Known transient (measured 2026-08-19, and the reason the run must be
/// consecutive rather than cumulative): EARLY in training the splats have not
/// yet covered the frame, so prior pixels with nothing rendered behind them
/// score `cos ≈ 0` and the gate drops them. On a synthetic 500-step run with a
/// correct prior this bottomed out at 12% around step 100, warned once, then
/// climbed to 26% by step 300 and the counter reset — exactly the intended
/// behaviour. It does not arise under the documented recipe at all, because the
/// reference arms the gate at ~37.5% of the run (`--normal-gate-start-iter`), by
/// which time the geometry exists. A run that arms the gate at step 0 should
/// expect one early warning and read a LATER sustained run as the real signal.
const NORMAL_GATE_LOW_SAMPLES_TO_WARN: u32 = 3;

/// Target number of contradiction-gate diagnostic samples per refine window.
///
/// The sampling stride is derived from `refine_every` (same idiom as
/// `EDGE_MIN_VIEW_SAMPLES`) rather than being a fresh absolute constant, so the
/// diagnostic's readback rides a cadence the training loop already runs periodic
/// device work on instead of introducing a new stall rhythm of its own. On every
/// other step the diagnostic costs nothing at all — no tensor is built and
/// nothing is read back.
const NORMAL_GATE_SAMPLES_PER_WINDOW: u32 = 2;

/// Steps from the start of training for which the TOTAL LOSS is checked for
/// finiteness on EVERY iteration.
///
/// # Why a cadence at all
///
/// Reading the loss scalar forces a GPU readback, which synchronises. The
/// trainer deliberately avoids that on the hot path — it is why the rerun and
/// JSONL loggers gate their own reads. An unconditional per-step check would
/// hand every run a permanent stall to protect against a rare event.
///
/// # Why EVERY step early, and only sampled later
///
/// Explosions happen early: the learning rate is at its highest, the scene is
/// least settled, and densification is at its most aggressive. Several hundred
/// readbacks at the very start cost a fraction of a second against a run that
/// would otherwise spend hours training on NaN and produce nothing.
///
/// # Blast radius between checks
///
/// After the early window the check rides the refine cadence, and that is not
/// only a cost argument — it is a containment one. While refinement is active,
/// the non-finite parameter prune inside `refine_for_phase` runs on exactly
/// those iterations, so any parameter that NaNs between two checks is swept at
/// the very step the next check happens. The unbounded window is therefore only
/// the one AFTER refinement stops, which is precisely the window
/// `prune_non_finite_splats` was added to cover from the eval cadence.
const NONFINITE_LOSS_CHECK_STEPS: u32 = 250;

/// The three per-parameter Adam states of a [`Splats`] module, owned directly
/// so the trainer can update LR scaling every step and surgically edit the
/// momentum tensors during refine — all GPU-side, no record round-trips. burn
/// 0.22 dropped the `SimpleOptimizer`/`OptimizerAdaptor` machinery our old
/// `AdamScaled` rode; this hand-rolled owner replaces it (upstream #517 design).
pub(crate) struct SplatOptim {
    adam: AdamScaled,
    transforms: AdamState<2>,
    sh_coeffs: AdamState<3>,
    opacities: AdamState<1>,
}

/// Step one parameter: pull its gradient, run Adam on the inner
/// (autodiff-free) tensor, and re-wrap tracking. Parameters without a
/// gradient this step are left untouched.
fn step_param<const D: usize>(
    adam: &AdamScaled,
    lr: f64,
    param: Param<Tensor<D>>,
    state: &mut AdamState<D>,
    grads: &mut Gradients,
) -> Param<Tensor<D>> {
    param.map(|t| {
        let Some(grad) = t.grad_remove(grads) else {
            return t;
        };
        let stepped = adam.step(lr, t.inner(), &grad, state);
        Tensor::from_inner(stepped).require_grad()
    })
}

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn sparse_sh_adam_requested() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = brush_render::native_msl::option_requested(
            brush_render::native_msl::SPARSE_SH_ADAM_ENV,
        );
        if enabled {
            tracing::warn!("experimental sparse native-MSL SH Adam enabled");
        }
        enabled
    })
}

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn can_defer_sh_grad(optimizer: &SplatOptim, splats: &Splats) -> bool {
    if !sparse_sh_adam_requested()
        || cfg!(feature = "debug-validation")
        || !splats.sh_coeffs.val().is_require_grad()
        || splats.sh_coeffs.val().is_distributed()
    {
        return false;
    }
    use brush_render::burn_glue::detach_autodiff;
    let param = detach_autodiff(splats.sh_coeffs.val());
    if !crate::sh_adam::sparse_sh_adam_supported(&param) {
        return false;
    }
    // AdamScaled has no gradient clipping, so the only gate is state
    // compatibility (which is false until sh_coeffs has taken its first step and
    // its momentum is populated).
    AdamScaled::sparse_sh_compatible(&param, &optimizer.sh_coeffs)
}

#[cfg(not(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
)))]
fn can_defer_sh_grad(_optimizer: &SplatOptim, _splats: &Splats) -> bool {
    false
}

pub struct SplatTrainer {
    config: TrainConfig,
    sched_mean: ExpLrScheduler,
    sched_scale: ExpLrScheduler,
    refine_record: Option<RefineRecord>,
    optim: Option<SplatOptim>,
    /// Optional per-view appearance compensation (bilateral grid / PPISP).
    /// Lives on the inner backend between steps, like the splats.
    appearance: Option<AppearanceTrainState>,
    ssim_enabled: bool,
    bounds: BoundingBox,
    step_count: u32,
    max_sh_degree: u32,
    rng: rand::rngs::StdRng,
    /// Run seed, kept so the one-time RANSAC plane extraction is deterministic.
    seed: u64,
    /// Per-train-view (world center, focal in px at native res) for the
    /// Mip-Splatting 3D filter. Empty disables it. The floor itself lives on
    /// the splats (recomputed at each refine), not here.
    view_cams: Vec<(glam::Vec3, f32)>,
    /// `DiG` feature-training state; created lazily on the first batch that
    /// carries feature maps.
    dig: Option<DigTrainState>,
    /// TIDI-GS floater-suppression state (learned importance `ω` + persistent
    /// visibility / gradient-EMA / birth-iter accumulators). Created lazily on
    /// the first step when `--tidi-prune` is set; `None` (and inert) otherwise.
    tidi: Option<TidiState>,
    /// Static distance-to-cloud grid for the depth-coupled opacity regularizer
    /// (`--depth-opacity-reg-weight`). Built ONCE from the seed point cloud at
    /// training start (see `train_stream`), carried across LOD boundaries, and
    /// `None` (inert) when the regularizer is off or there is no seed cloud.
    opacity_reg_grid: Option<CloudDistanceGrid>,
    /// RANSAC planes extracted ONCE from the seed cloud (shared infra for the
    /// plane-gated distance field, FIX 1, and the co-planarity constraint, FIX
    /// 2). `None` unless `--plane-gate` or `--plane-coplanarity-weight` is set.
    /// Cheap + device-independent, so it carries across LOD boundaries verbatim.
    plane_set: Option<PlaneSet>,
    /// Static distance-to-cloud grid for the hard cloud-distance prune
    /// (`--cloud-prune`). Built ONCE from the seed cloud at training start,
    /// ALWAYS point-only (planes = `None`) — the prune must not use the
    /// plane-augmented distance (a plane shields wall-perpendicular floaters), so
    /// this is a SEPARATE grid from `opacity_reg_grid` even when `--plane-gate` is
    /// on. Carried across LOD boundaries; `None` (inert) when `--cloud-prune` is
    /// off or there is no seed cloud.
    cloud_prune_grid: Option<CloudDistanceGrid>,
    /// Scene scale captured ONCE from the training camera poses, for
    /// `--normalize-metric-weights` (see `scene_scale_from_cameras`).
    ///
    /// Deliberately not the live, refine-updated `bounds` — a moving scale would
    /// make the effective metric weights drift mid-run and confound the ramp
    /// schedules, so an ablation could not attribute a result to either.
    /// `set_init_scene_scale` is one-shot for the same reason: LOD phases
    /// re-supply cameras, and the scale must not move there either.
    init_scene_scale: Option<f32>,
    /// Consecutive contradiction-gate samples whose surviving fraction was below
    /// `NORMAL_GATE_LOW_FRACTION`. Drives the "sustained" half of the
    /// over-masking warning; reset by any healthy sample. Never touched unless
    /// `--normal-gate-degrees` is set.
    normal_gate_low_samples: u32,
    #[cfg(not(target_family = "wasm"))]
    lpips: Option<lpips::LpipsModel>,
}

fn inv_sigmoid(x: Tensor<1>) -> Tensor<1> {
    (x.clone() / (1.0f32 - x)).log()
}

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn step_sh_coeffs(
    optimizer: &mut SplatOptim,
    mut splats: Splats,
    grads: &mut Gradients,
    deferred: Option<DeferredShGrad>,
    learning_rate: f64,
) -> Splats {
    let Some(deferred) = deferred else {
        splats.sh_coeffs = step_param(
            &optimizer.adam,
            learning_rate,
            splats.sh_coeffs,
            &mut optimizer.sh_coeffs,
            grads,
        );
        return splats;
    };

    // Positive evidence that the sparse fused path really runs when
    // `BRUSH_NATIVE_MSL_SPARSE_SH_ADAM=1` (rather than silently falling back to
    // the dense `step_param` branch above). `tracing::warn!` is NOT visible
    // under brush-cli — it wires `env_logger` only, with no tracing-log bridge —
    // so this uses `log::info!`, which is.
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SPARSE_STEPS: AtomicU64 = AtomicU64::new(0);
        let n = SPARSE_STEPS.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n.is_multiple_of(100) {
            log::info!("sparse native-MSL SH Adam: step_sparse_sh executed {n} time(s)");
        }
    }

    use brush_render::burn_glue::{detach_autodiff, lift_to_autodiff};
    let param = detach_autodiff(splats.sh_coeffs.val());
    // Take the SH state out by value for the consuming `step_sparse_sh`, then
    // write the returned state back.
    let state = std::mem::replace(&mut optimizer.sh_coeffs, AdamState::new(None, false));
    assert!(
        AdamScaled::sparse_sh_compatible(&param, &state),
        "deferred SH optimizer state changed after render preflight"
    );
    assert!(
        crate::sh_adam::sparse_sh_adam_supported(&param),
        "deferred SH device support changed after render preflight"
    );

    let (param, new_state) = optimizer.adam.step_sparse_sh(
        learning_rate,
        param,
        deferred.render_transforms,
        deferred.global_from_compact_gid,
        detach_autodiff(deferred.compact_grads),
        deferred.project_uniforms,
        state,
    );
    optimizer.sh_coeffs = new_state;
    // `param` is the freshly-stepped tensor from `step_sparse_sh`, and it is
    // ALREADY on the inner (non-autodiff) backend: it entered as
    // `detach_autodiff(..)` and `step_sparse_sh` never lifts it. Re-lift it into
    // the graph with `lift_to_autodiff(..).require_grad()`, which is what this
    // call site did before the burn-0.22 port (commit 3178759e) and what every
    // other inner->autodiff parameter boundary in brush-train / brush-appearance
    // uses (see `edge.rs`, `tidi.rs`, `brush-appearance/train_state.rs`).
    //
    // The port replaced this with `Tensor::from_inner(param.inner())` on the
    // false premise that `lift_to_autodiff` had become `pub(crate)` — it is
    // `pub` (brush-render/src/burn_glue.rs). That rewrite added one `.inner()`
    // too many: `.inner()` on an already-inner Dispatch tensor panics with
    // "Requires autodiff tensor." (burn-dispatch backend.rs:584), which is why
    // BRUSH_NATIVE_MSL_SPARSE_SH_ADAM=1 died immediately. `lift_to_autodiff` is
    // also the correct helper rather than `Tensor::from_inner` here because it
    // lifts at the concrete-Wgpu autodiff level and sets `checkpointing`
    // explicitly (see its doc comment).
    splats.sh_coeffs = splats
        .sh_coeffs
        .map(|_| lift_to_autodiff(param).require_grad());
    splats
}

#[cfg(not(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
)))]
fn step_sh_coeffs(
    optimizer: &mut SplatOptim,
    mut splats: Splats,
    grads: &mut Gradients,
    deferred: Option<DeferredShGrad>,
    learning_rate: f64,
) -> Splats {
    debug_assert!(
        deferred.is_none(),
        "non-native builds must never request deferred SH gradients"
    );
    drop(deferred);
    splats.sh_coeffs = step_param(
        &optimizer.adam,
        learning_rate,
        splats.sh_coeffs,
        &mut optimizer.sh_coeffs,
        grads,
    );
    splats
}

/// Rodrigues rotation taking `up` (assumed unit-length) onto `+Z`.
///
/// Degenerate case: when `up × Z` is below the epsilon, `up` is already
/// (anti)parallel to `+Z`. Parallel yields the identity; antiparallel is a 180°
/// turn about an arbitrary axis perpendicular to `up`, chosen deterministically
/// so the result does not depend on floating-point luck. (In the antiparallel
/// case the choice of perpendicular axis does change the resulting X/Y
/// coordinates, so a deterministic pick is what makes `scene_scale_from_cameras`
/// reproducible.)
fn rotation_up_to_z(up: glam::Vec3) -> glam::Mat3 {
    let z = glam::Vec3::Z;
    let axis = up.cross(z);
    let axis_len = axis.length();
    if axis_len < 1e-6 {
        return if up.dot(z) >= 0.0 {
            glam::Mat3::IDENTITY
        } else {
            let helper = if up.x.abs() < 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            let perp = up.cross(helper).normalize();
            glam::Mat3::from_axis_angle(perp, std::f32::consts::PI)
        };
    }
    let angle = up.dot(z).clamp(-1.0, 1.0).acos();
    glam::Mat3::from_axis_angle(axis / axis_len, angle)
}

/// Mean world-frame "up" direction of a camera set, unit-length.
///
/// **This is the one convention-sensitive line of `scene_scale_from_cameras`,
/// which is why it is a separate, directly testable function.** The `gauss-surf`
/// PGSR trainer's poses are OpenGL, where camera `+Y` is UP, so it averages the
/// c2w `+Y` column as-is. Ours are `OpenCV`, where camera `+Y` points DOWN, so the
/// world-frame up axis is `−(R_c2w · Ŷ)`. Getting the sign wrong yields a
/// rotation that differs by a 180° turn about a horizontal axis, which silently
/// changes the derived scale on any scene whose cameras are not symmetric about
/// that axis.
///
/// Falls back to `+Z` when the camera up axes cancel exactly (a synthetic
/// back-to-back pair), which skips the reorientation rather than normalizing
/// numerical noise.
pub fn mean_camera_up(cameras: &[Camera]) -> glam::Vec3 {
    if cameras.is_empty() {
        return glam::Vec3::Z;
    }
    let sum = cameras.iter().fold(glam::Vec3::ZERO, |acc, cam| {
        acc - (cam.rotation * glam::Vec3::Y)
    });
    if sum.length() < 1e-6 {
        glam::Vec3::Z
    } else {
        sum.normalize()
    }
}

/// Scene scale in world (metric, for a metric capture) units, computed once from
/// the training camera poses.
///
/// This is `scene_scale` from the `gauss-surf` PGSR trainer
/// (rerun-io/examples-monorepo, Apache-2.0, by Pablo Vela), measured from its
/// `train_gsplat/cache.py` — an implementation detail, not something the PGSR
/// paper (arXiv:2406.06521) defines. Credited explicitly because the weights
/// below are divided by this quantity, so anyone questioning a ratio needs to
/// be able to find what the ratio was calibrated against.
///
/// It is deliberately NOT `BoundingBox::median_size()` or
/// `splat_init::estimate_scene_scale()` — neither has these semantics, and
/// `gauss-surf`'s constant ratios only transfer if the scale they divide is the
/// same quantity. The five steps:
///
/// 1. `translation = mean(camera origins)`
/// 2. `up = normalize(mean(world-frame camera up axes))`
/// 3. Rodrigues rotation `R` taking `up` onto `+Z`
/// 4. `oriented = R · (origin − translation)` for every camera
/// 5. `scene_scale = max(|component|)` over every oriented origin
///
/// **Convention note.** `gauss-surf`'s poses are OpenGL, where camera `+Y` is
/// UP, so it averages c2w column 1 directly. Ours are `OpenCV`, where camera `+Y`
/// is DOWN, so the world-frame up axis is `−(R_c2w · Ŷ)`. The sign matters: a
/// flipped `up` produces a rotation that differs by 180°, which changes the
/// oriented coordinates and hence the maximum. `scene_scale_from_camera_ring`
/// pins this.
///
/// Returns `None` for an empty camera list or a non-finite/zero result; callers
/// fall back to 1.0 (i.e. to unnormalized weights) rather than poisoning the
/// loss.
pub fn scene_scale_from_cameras(cameras: &[Camera]) -> Option<f32> {
    if cameras.is_empty() {
        return None;
    }

    let n = cameras.len() as f32;
    let translation = cameras
        .iter()
        .fold(glam::Vec3::ZERO, |acc, cam| acc + cam.position)
        / n;

    let up = mean_camera_up(cameras);
    let rot = rotation_up_to_z(up);

    let scale = cameras
        .iter()
        .map(|cam| {
            let oriented = rot * (cam.position - translation);
            oriented.x.abs().max(oriented.y.abs()).max(oriented.z.abs())
        })
        .fold(0.0f32, f32::max);

    (scale.is_finite() && scale > 0.0).then_some(scale)
}

pub async fn get_splat_bounds(splats: Splats, percentile: f32) -> BoundingBox {
    let means: Vec<f32> = splats
        .means()
        .into_data_async()
        .await
        .expect("Failed to fetch splat data")
        .to_vec()
        .expect("Failed to get means");
    bounds_from_pos(percentile, &means)
}

impl SplatTrainer {
    #[allow(unused_variables)]
    pub fn new(config: &TrainConfig, device: &Device, bounds: BoundingBox) -> Self {
        Self::new_seeded(config, device, bounds, 42)
    }

    #[allow(unused_variables)]
    pub fn new_seeded(
        config: &TrainConfig,
        device: &Device,
        bounds: BoundingBox,
        seed: u64,
    ) -> Self {
        let decay =
            (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_train_iters as f64);
        let lr_mean = ExpLrScheduler::new(config.lr_mean, decay);

        // MRNF LR schedule (R1): independent exponential decay for the log-scale
        // parameters, mirroring LFS `_scale_lr_gamma` (mrnf.cpp:425) and the
        // per-step `_scale_lr_current *= _scale_lr_gamma` (mrnf.cpp:1360). Guarded
        // like LFS `compute_decay_gamma` (start/end > 0). ON by default now
        // (LFS `mrnf_defaults` parity): lr_scale 7e-3 -> lr_scale_end 5e-3 gives
        // gamma < 1.0. Set `--lr-scale-end` == `--lr-scale` to make gamma == 1.0.
        let scale_decay = if config.lr_scale > 0.0 && config.lr_scale_end > 0.0 {
            (config.lr_scale_end / config.lr_scale)
                .powf(1.0 / config.total_train_iters.max(1) as f64)
        } else {
            1.0
        };
        let lr_scale = ExpLrScheduler::new(config.lr_scale, scale_decay);

        let ssim_enabled = config.ssim_weight > 0.0;

        // Growth is gated on the global iter. LOD phases run past
        // total_train_iters but their refines should never grow — clamp
        // here so growth_stop is never effectively past end-of-training.
        let mut config = config.clone();
        config.growth_stop_iter = config.growth_stop_iter.min(config.total_train_iters);

        #[cfg(not(target_family = "wasm"))]
        let lpips = (config.lpips_loss_weight > 0.0).then(|| lpips::load_vgg_lpips(device));

        Self {
            config,
            sched_mean: lr_mean,
            sched_scale: lr_scale,
            optim: None,
            appearance: None,
            refine_record: None,
            ssim_enabled,
            bounds,
            step_count: 0,
            max_sh_degree: 0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            seed,
            view_cams: Vec::new(),
            dig: None,
            tidi: None,
            opacity_reg_grid: None,
            plane_set: None,
            cloud_prune_grid: None,
            init_scene_scale: None,
            normal_gate_low_samples: 0,
            #[cfg(not(target_family = "wasm"))]
            lpips,
        }
    }

    /// Supply per-train-view (world center, focal-px at native res) for the
    /// Mip-Splatting 3D filter.
    pub fn set_view_cams(&mut self, view_cams: Vec<(glam::Vec3, f32)>) {
        self.view_cams = view_cams;
    }

    /// Capture the fixed scene scale for `--normalize-metric-weights` from the
    /// full set of training camera poses.
    ///
    /// **One-shot on purpose**: later calls (LOD phases re-supply cameras) are
    /// ignored, so the effective metric weights never move mid-run. A no-op
    /// unless `--normalize-metric-weights` is set, so a default run allocates
    /// and logs nothing.
    pub fn set_init_scene_scale(&mut self, cameras: &[Camera]) {
        if !self.config.normalize_metric_weights || self.init_scene_scale.is_some() {
            return;
        }
        // `log::`, not `tracing::` — brush-cli wires `env_logger` with no
        // tracing-log bridge, so a `tracing::info!` here would be invisible in
        // exactly the runs that need to record which scale was used (same note
        // as the sparse SH Adam site above).
        match scene_scale_from_cameras(cameras) {
            Some(scale) => {
                log::info!(
                    "normalize-metric-weights: scene scale {scale} from {} camera poses \
                     (flatten weight /= scale, scale-reg weight /= scale^2, \
                     scale-reg threshold *= scale)",
                    cameras.len()
                );
                self.init_scene_scale = Some(scale);
            }
            None => {
                log::warn!(
                    "normalize-metric-weights: could not derive a scene scale from {} camera \
                     poses; falling back to 1.0 (weights stay unnormalized)",
                    cameras.len()
                );
            }
        }
    }

    /// Divisor applied to the METRIC-dimensioned loss weights. `1.0` (exact
    /// identity) unless `--normalize-metric-weights` is on AND a usable scale
    /// was captured.
    fn metric_weight_scale(&self) -> f32 {
        if !self.config.normalize_metric_weights {
            return 1.0;
        }
        self.init_scene_scale
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(1.0)
    }

    /// Attach the Mip-Splatting scale floor for the trainer's active camera
    /// resolution. Replaces any existing floor without baking it; callers
    /// that change splat count must drop or select the old floor first.
    pub fn apply_min_scale_floor(&self, splats: Splats) -> Splats {
        let means = splats.means();
        match compute_min_scale(&means, &self.view_cams, MIN_SCALE_FACTOR) {
            Some(floor) => splats.with_min_scale(floor),
            None => splats,
        }
    }

    /// Set up per-view appearance compensation (bilateral grid or PPISP,
    /// gated on the train config). `camera_indices` maps each training view
    /// to a physical-camera group for PPISP's per-camera params; same length
    /// and order as the scene's view list.
    pub fn init_appearance(
        &mut self,
        camera_indices: Vec<u32>,
        start_iter: u32,
        device: &Device,
    ) -> anyhow::Result<()> {
        if !self.config.appearance_enabled() {
            self.appearance = None;
            return Ok(());
        }
        anyhow::ensure!(
            start_iter == 0,
            "appearance parameters are not stored in PLY checkpoints; resume with --start-iter is unsupported when --bilateral-grid or --ppisp is enabled"
        );
        let [grid_x, grid_y, guidance] = self.config.bilagrid_dims.as_slice() else {
            anyhow::bail!("bilagrid-dims must contain exactly `x,y,guidance`");
        };
        let [beta1, beta2] = self.config.bilagrid_betas.as_slice() else {
            anyhow::bail!("bilagrid-betas must contain exactly `b1,b2`");
        };
        let config = AppearanceConfig {
            bilagrid: self.config.bilateral_grid,
            bilagrid_dims: (*grid_x as usize, *grid_y as usize, *guidance as usize),
            bilagrid_tv_weight: self.config.bilagrid_tv_weight,
            bilagrid_lr: self.config.bilagrid_lr,
            bilagrid_betas: (*beta1, *beta2),
            ppisp: self.config.ppisp,
            ppisp_lr: self.config.ppisp_lr,
            ppisp_reg_scale: self.config.ppisp_reg_scale,
        };
        self.appearance =
            AppearanceTrainState::new(config, camera_indices, self.config.total_iters(), device)
                .map_err(anyhow::Error::msg)?;
        Ok(())
    }

    /// Whether appearance compensation is active.
    pub fn has_appearance(&self) -> bool {
        self.appearance.is_some()
    }

    /// Move appearance parameters and optimizer state into a replacement
    /// trainer (used at LOD boundaries).
    pub fn take_appearance(&mut self) -> Option<AppearanceTrainState> {
        self.appearance.take()
    }

    pub fn set_appearance(&mut self, appearance: Option<AppearanceTrainState>) {
        self.appearance = appearance;
    }

    /// Build the static distance-to-cloud grid for the depth-coupled opacity
    /// regularizer from the seed point cloud (`cloud_means` = the seed splats'
    /// centres — the measured surface the run is seeded from). Called ONCE at
    /// training start when `--depth-opacity-reg-weight > 0`; the grid is
    /// view-independent, so it never needs rebuilding as training proceeds.
    /// A no-op (leaves the grid `None`, regularizer inert) when the cloud is
    /// empty — e.g. a random-init run with no seed points.
    pub async fn init_opacity_reg_grid(&mut self, cloud_means: Tensor<2>, device: &Device) {
        self.init_plane_priors(cloud_means, device).await;
    }

    /// Build the seed-cloud priors ONCE at training start: the distance-to-cloud
    /// grid (when `--depth-opacity-reg-weight > 0`) and the RANSAC planes (when
    /// `--plane-gate` or `--plane-coplanarity-weight` is set). When `--plane-gate`
    /// is on, the extracted planes are baked into the grid's field (FIX 1). The
    /// planes are also stored on the trainer for the co-planarity constraint (FIX
    /// 2). All view-independent, so nothing here is rebuilt as training proceeds.
    pub async fn init_plane_priors(&mut self, cloud_means: Tensor<2>, device: &Device) {
        let margin = self.config.depth_opacity_reg_margin;
        let softness = self.config.depth_opacity_reg_softness;
        let need_planes = self.config.plane_gate || self.config.plane_coplanarity_weight > 0.0;

        // RANSAC planes (shared by both features), extracted once from the cloud.
        let planes = if need_planes {
            extract_planes_from_cloud(cloud_means.clone(), self.seed).await
        } else {
            None
        };

        // Hard cloud-distance prune (`--cloud-prune`): its OWN point-only grid,
        // sized to the prune threshold. ALWAYS point-only (planes = None) even
        // under --plane-gate — the prune must NOT read the plane-augmented
        // distance (a plane shields wall-perpendicular floaters). Built here from
        // a CLONE so the opacity-reg build below can still consume `cloud_means`.
        if self.config.cloud_prune {
            let dist = self.config.cloud_prune_dist.max(1e-6);
            // Softness sizes only the field's accurate/truncation REACH past
            // `dist` (not the stored on-surface distances); mirror the opacity-reg
            // proportion (softness = margin/3) so the field is accurate out to
            // ~1.7·dist and truncates at ~3·dist — both > dist, so a floater at
            // ANY distance beyond `dist` reads a value > dist. vox = dist/3 keeps
            // the on-surface quantisation a small fraction of the threshold.
            let softness = dist / 3.0;
            self.cloud_prune_grid =
                CloudDistanceGrid::build(cloud_means.clone(), dist, softness, None, device).await;
        }

        // Distance-to-cloud grid, plane-augmented only when --plane-gate is set.
        if self.config.depth_opacity_reg_weight > 0.0 {
            let plane_slice = if self.config.plane_gate {
                planes.as_ref().map(|p| p.planes.as_slice())
            } else {
                None
            };
            self.opacity_reg_grid =
                CloudDistanceGrid::build(cloud_means, margin, softness, plane_slice, device).await;
        }
        self.plane_set = planes;
    }

    /// Whether the opacity-reg cloud grid is built (for logging).
    pub fn has_opacity_reg_grid(&self) -> bool {
        self.opacity_reg_grid.is_some()
    }

    /// Summary of the extracted planes (for logging): count + inlier fractions,
    /// or `None` when plane priors are off / no planar structure was found.
    pub fn plane_summary(&self) -> Option<String> {
        self.plane_set.as_ref().map(|ps| {
            let fracs: Vec<String> = ps
                .planes
                .iter()
                .map(|p| format!("{:.1}%", p.inlier_frac * 100.0))
                .collect();
            // The co-planarity assignment band is the knob that decides how much
            // of the scene gets flattened onto these planes, and its default is
            // derived rather than literal — so it has to appear in the log or
            // nobody can tell an over-flattened run from a tuned one after the
            // fact. Only printed when the term is actually on.
            let coplanarity = if self.config.plane_coplanarity_weight > 0.0 {
                let assign = crate::tidi::resolve_coplanarity_assign_dist(
                    self.config.plane_coplanarity_assign_dist,
                    ps.spacing,
                );
                let source = if self.config.plane_coplanarity_assign_dist > 0.0 {
                    "explicit"
                } else {
                    "default, from spacing"
                };
                format!(
                    ", coplanarity w {} assign-dist {:.4} ({source})",
                    self.config.plane_coplanarity_weight, assign
                )
            } else {
                String::new()
            };
            format!(
                "{} planes (spacing {:.4}, band {:.4}){}, inliers: [{}]",
                ps.planes.len(),
                ps.spacing,
                ps.threshold,
                coplanarity,
                fracs.join(", ")
            )
        })
    }

    /// Move the RANSAC plane set out, to carry across an LOD boundary (the cloud
    /// never changes, so the planes are reused verbatim rather than recomputed).
    pub fn take_plane_set(&mut self) -> Option<PlaneSet> {
        self.plane_set.take()
    }

    pub fn set_plane_set(&mut self, planes: Option<PlaneSet>) {
        self.plane_set = planes;
    }

    /// Move the (static) opacity-reg grid out, to carry across an LOD boundary
    /// where the trainer is rebuilt. The cloud never changes, so the grid is
    /// reused verbatim rather than recomputed.
    pub fn take_opacity_reg_grid(&mut self) -> Option<CloudDistanceGrid> {
        self.opacity_reg_grid.take()
    }

    pub fn set_opacity_reg_grid(&mut self, grid: Option<CloudDistanceGrid>) {
        self.opacity_reg_grid = grid;
    }

    /// Whether the cloud-prune point-only grid is built (for logging).
    pub fn has_cloud_prune_grid(&self) -> bool {
        self.cloud_prune_grid.is_some()
    }

    /// Move the (static, point-only) cloud-prune grid out, to carry across an LOD
    /// boundary where the trainer is rebuilt. The seed cloud never changes, so the
    /// grid is reused verbatim rather than recomputed.
    pub fn take_cloud_prune_grid(&mut self) -> Option<CloudDistanceGrid> {
        self.cloud_prune_grid.take()
    }

    pub fn set_cloud_prune_grid(&mut self, grid: Option<CloudDistanceGrid>) {
        self.cloud_prune_grid = grid;
    }

    /// Magnitude summary of the learned appearance parameters (`None` when
    /// appearance compensation is disabled).
    pub async fn appearance_stats(&self) -> Option<String> {
        match &self.appearance {
            Some(state) => state.stats().await,
            None => None,
        }
    }

    /// Snapshot the `DiG` features + decoder for export, if feature
    /// training is active.
    pub async fn dig_export(&self) -> Option<dig::DigExport> {
        match &self.dig {
            Some(d) => Some(d.module.export().await),
            None => None,
        }
    }

    /// Forward-only appearance correction for an eval render of *training*
    /// view `view_idx` (`--train-on-eval`). `img` is `[H, W, 3|4]` on the
    /// inner backend; returns it unchanged when appearance is disabled.
    pub fn appearance_eval_correction(&self, img: Tensor<3>, view_idx: usize) -> Tensor<3> {
        match &self.appearance {
            Some(state) => state.apply_eval(img, view_idx),
            None => img,
        }
    }

    /// A viewer-friendly recoloring of `splats` by their current `DiG`
    /// features, if feature training is active: decode each gaussian's
    /// feature through the MLP and map the first three output channels to
    /// RGB. The decoder targets the dataset's PCA space, whose channels
    /// are variance-ordered, so channels 0..3 are already the top PCA
    /// components — no extra projection needed. `splats` must be on the
    /// inner (non-autodiff) backend, as between training steps.
    pub fn dig_view_splats(&self, splats: &Splats) -> Option<Splats> {
        let dig = self.dig.as_ref()?;
        if splats.num_splats() as usize != dig.module.features.dims()[0] {
            // Mid-refine mismatch; skip this preview tick.
            return None;
        }
        let module = dig.module.valid();
        let decoded = module.decode(module.features.val());
        let rgb = decoded.slice(s![.., 0..3]);
        // Robust per-channel normalization: mean ± 2σ → [0, 1].
        let mean = rgb.clone().mean_dim(0);
        let std = rgb.clone().var(0).sqrt().clamp_min(1e-6);
        let color = ((rgb - mean) / (std * 4.0) + 0.5).clamp(0.0, 1.0);
        let sh = ((color - 0.5) / brush_render::kernels::sh::SH_C0).unsqueeze_dim(1);

        Some(Splats {
            transforms: Param::initialized(ParamId::new(), splats.transforms.val()),
            sh_coeffs: Param::initialized(ParamId::new(), sh),
            raw_opacities: Param::initialized(ParamId::new(), splats.raw_opacities.val()),
            render_mip: splats.render_mip,
            min_scale: splats.min_scale.clone(),
        })
    }

    pub async fn step(&mut self, batch: SceneBatch, splats: Splats) -> (Splats, TrainStepStats) {
        // `step_count` is this trainer's own count, which equals the global
        // iteration for a run that starts at 0. The stream path passes the true
        // global iteration instead, so `--depth-normal-start-iter` behaves
        // correctly on a resume.
        let iter = self.step_count;
        self.step_with_refine_weight(batch, splats, true, iter)
            .await
    }

    /// Whether the refinement-only gradient statistic is still consumed by
    /// high-gradient densification at `global_iter`.
    pub fn refinement_weight_needed(&self, global_iter: u32) -> bool {
        global_iter < self.config.growth_stop_iter
    }

    /// Steps between edge-guidance samples: `refine_every / EDGE_MIN_VIEW_SAMPLES`
    /// so a refine window contributes ~`EDGE_MIN_VIEW_SAMPLES` views.
    fn edge_sample_stride(&self) -> u32 {
        (self.config.refine_every / EDGE_MIN_VIEW_SAMPLES).max(1)
    }

    /// Steps between contradiction-gate diagnostic samples, derived from the
    /// refine window like `edge_sample_stride`.
    fn normal_gate_sample_stride(&self) -> u32 {
        (self.config.refine_every / NORMAL_GATE_SAMPLES_PER_WINDOW).max(1)
    }

    /// Whether this step should measure the contradiction gate's surviving
    /// fraction (plan §4.7).
    ///
    /// **False for every step of a default run.** With `--normal-gate-degrees`
    /// unset `normal_gate_cos_at` is `None`, so the caller builds no diagnostic
    /// tensor and performs no readback — the diagnostic is as inert as the gate
    /// it observes. Counts GLOBAL iterations, like the gate's own arming step.
    fn should_sample_normal_gate(&self, global_iter: u32) -> bool {
        self.config.normal_gate_cos_at(global_iter).is_some()
            && global_iter.is_multiple_of(self.normal_gate_sample_stride())
    }

    /// Record one contradiction-gate sample: log the surviving fraction, and
    /// warn when it has been low for `NORMAL_GATE_LOW_SAMPLES_TO_WARN`
    /// consecutive samples. Returns whether the warning fired, for tests.
    ///
    /// `surviving` / `valid` are the two counts from
    /// `brush_loss::normal_gate_counts`. A frame with `valid == 0` carried no
    /// usable prior, which says nothing about the gate: it is neither logged nor
    /// counted toward the sustained-low run, and it does not reset it either.
    ///
    /// `log::`, not `tracing::` — brush-cli wires `env_logger` with no
    /// tracing-log bridge, so a `tracing::warn!` here would be invisible in
    /// exactly the headless runs this guard exists for.
    fn record_normal_gate_sample(&mut self, global_iter: u32, surviving: f32, valid: f32) -> bool {
        if !(valid.is_finite() && valid > 0.0) || !surviving.is_finite() {
            return false;
        }
        let fraction = surviving / valid;
        log::info!(
            "normal gate: iter {global_iter} kept {:.1}% of prior pixels ({} of {})",
            fraction * 100.0,
            surviving as u64,
            valid as u64
        );
        if fraction < NORMAL_GATE_LOW_FRACTION {
            self.normal_gate_low_samples += 1;
            if self.normal_gate_low_samples >= NORMAL_GATE_LOW_SAMPLES_TO_WARN {
                log::warn!(
                    "normal gate: surviving fraction has been under {:.0}% for {} consecutive \
                     samples (now {:.1}%). The gate is masking most of the prior rather than \
                     cleaning it up, so --normal-loss-weight is being starved. Check the \
                     normal prior's sign/frame convention for this capture, or widen \
                     --normal-gate-degrees.",
                    NORMAL_GATE_LOW_FRACTION * 100.0,
                    self.normal_gate_low_samples,
                    fraction * 100.0
                );
                return true;
            }
        } else {
            self.normal_gate_low_samples = 0;
        }
        false
    }

    /// Accumulate this step's GT-view edge score into the refine record (MRNF
    /// port, delta #4). No-op unless `--use-edge-map` is set and this step lands
    /// on the sampling stride. `splats` are the render-time (pre-optimizer-step)
    /// splats; `camera`/`gt_packed` are this step's view. Pure non-differentiable
    /// bookkeeping: the alpha-blended score comes from an isolated feature
    /// forward+backward rooted at a throwaway unit-feature leaf, dropped after the
    /// gradient read — it never entangles the training photometric graph. See
    /// `crate::edge::project_edge_scores` for the T·α·edge parity.
    async fn accumulate_edge_sample(
        &mut self,
        splats: &Splats,
        camera: &Camera,
        gt_packed: &Tensor<2, Int>,
        composite_bg: Option<glam::Vec3>,
        img_size: glam::UVec2,
    ) {
        if !self.config.use_edge_map || !self.step_count.is_multiple_of(self.edge_sample_stride()) {
            return;
        }
        // GT RGB `[H, W, 3]` on the inner backend (same unpack the LPIPS path uses).
        let gt_rgb = brush_loss::unpack_gt_rgb(gt_packed.clone(), composite_bg);
        // Canny + directional NMS. Intentionally NOT median-normalized here: LFS
        // step (a) (per-view edge-MAP normalize) is a provable no-op given the
        // per-gaussian score normalize (step (b)) below is always applied —
        // `normalize_by_positive_median` is scale-equivariant, so an edge-map
        // scale cancels in the quotient. Do not restore it.
        let edge_map = edge::canny_edge_map(gt_rgb);
        let valid = splats.valid();
        let n = valid.num_splats() as usize;
        let device = splats.device().inner();
        let score = edge::project_edge_scores(&valid, edge_map, camera, img_size).await;

        // LFS step (b): positive-median normalize this view's per-gaussian score
        // before accumulation, so a high-contrast view can't dominate the window.
        let mut score_host: Vec<f32> = score
            .into_data_async()
            .await
            .expect("edge score readback")
            .into_vec()
            .expect("f32 edge score");
        edge::normalize_by_positive_median(&mut score_host);
        let score = Tensor::<1>::from_data(TensorData::new(score_host, [n]), &device);

        // INVARIANT: the splat set is constant within a refine window — this runs
        // on the render-time splats, `gather_edge` only sums, and `RefineRecord`
        // is recreated fresh right after each prune. So there is no mid-window
        // splat creation/freeze, which is why LFS's `zero_frozen_scores`
        // (mrnf.cpp:566) has no analogue here (its frozen set is empty in Brush's
        // model). If mid-window splat birth/freezing is ever introduced, restore
        // that guard so newborn splats can't self-reinforce their own scores.
        if let Some(record) = self.refine_record.as_mut() {
            record.gather_edge(score);
        }
    }

    /// Accumulate this step's error-map growth score into the refine record
    /// (MRNF `use_error_map` port). No-op unless `--error-map-densification` is
    /// set. `pred_rgb` is the post-appearance predicted RGB `[H, W, 3]` on the
    /// inner backend (the same image the SSIM loss sees); `gt_packed` is this
    /// view's GT; `masked` is true in `AlphaMode::Masked` (mask densification).
    ///
    /// Reproduces LFS's per-view error signal term-for-term:
    ///   1. `ê = mean_normalize(max(0, 1 − meanSSIM(pred, gt)) · mask(p))` — a
    ///      clean, nonnegative, map-mean-normalized D-SSIM error map
    ///      (`crate::error_map`), with the SAME `· gt.a` masking the photometric
    ///      loss applies in `AlphaMode::Masked` (brush-loss `mask` multiply).
    ///   2. `s_g = (Σ_p T_g·α_g·ê) / (Σ_p T_g·α_g)` — the coverage-weighted MEAN
    ///      of `ê` over the gaussian's footprint, via a `feat_dim=2` feature
    ///      backward (`crate::edge::project_coverage_weighted_mean`).
    ///   3. window-MAX accumulate (`gather_error`), NOT sum/mean.
    ///
    /// UNLIKE the edge path, this runs on EVERY step, not on the edge sampling
    /// stride: LFS folds row1 (the error map) into `_refine_weight_max` on every
    /// training view (`kernels_backward.cuh` + mrnf.cpp:602-605), whereas the edge
    /// guidance is deliberately sub-sampled (`MRNF_EDGE_MIN_VIEW_SAMPLES = 10`).
    /// Sharing the edge stride would score the error map on only ~10 of the
    /// ~`refine_every` views/window, so any gaussian whose reconstruction error
    /// peaks in an unsampled view would be under-scored (or, if never sampled,
    /// scored 0 and silently excluded from error-driven growth). The cost is one
    /// extra feature render+backward per step while the flag is on; this is
    /// intrinsic to the path-A fallback (LFS gets row1 for free inside its RGB
    /// backward, which Brush's autodiff cannot piggyback — see `error_map` doc).
    ///
    /// SCALE (defect-2 fix, 2026-07-22). LFS thresholds the RAW pixel-sum
    /// `Σ_p T·α·ê` at `τ_err = 0.003` (`mrnf.cpp:601-605,726`), but that 0.003 is
    /// the scale of the gradient-mode row 1 — a per-gaussian per-view SCALAR
    /// (the mean2d gradient norm, `kernels_backward.cuh:335`) — NOT the
    /// pixel-SUMMED error (`kernels_backward.cuh:564`). On a pixel-sum, `Σ T·α`
    /// scales with the gaussian's footprint (10^5–10^6 px at the port's
    /// 8K-derived render size), so the raw score reached ~1.16e6 and 0.003
    /// admitted 99.99% of gaussians — a no-op floor (in LFS-land the real
    /// pressure there is the weighted sample, `mrnf.cpp:790`, not the threshold).
    /// To make the THRESHOLD itself select — the port's stated design goal — the
    /// per-gaussian score divides the error sum by the coverage sum (LFS's own
    /// row 0, `densification_weight`, `kernels_backward.cuh:563`), recovering the
    /// coverage-weighted MEAN error per gaussian, footprint- and
    /// resolution-INVARIANT. That mean is still not O(1) (the mean-normalized `ê`
    /// is heavily right-skewed and, in masked mode, masked-out zeros deflate the
    /// covered-region mean, so ~all unmasked gaussians exceed the map's spatial
    /// mean 1.0). So — exactly as the edge path does — each view's per-gaussian
    /// scores are then POSITIVE-MEDIAN normalized (median → 1.0), which is robust
    /// to that skew and to low-error views (a scene-mean anchor instead explodes
    /// on a near-converged view and poisons the window-MAX). On that scale the
    /// natural anchor is `τ_err = 1.0` — a gaussian reconstructing worse than the
    /// per-view median — not 0.003. Pure non-differentiable bookkeeping: the
    /// score comes from an isolated feature forward+backward that never touches
    /// the photometric graph.
    async fn accumulate_error_sample(
        &mut self,
        splats: &Splats,
        camera: &Camera,
        pred_rgb: Tensor<3>,
        gt_packed: &Tensor<2, Int>,
        composite_bg: Option<glam::Vec3>,
        masked: bool,
        img_size: glam::UVec2,
    ) {
        if !self.config.error_map_densification {
            return;
        }
        // GT RGB `[H, W, 3]` on the inner backend (same unpack the loss uses for
        // its SSIM). `pred_rgb` already arrives detached on the inner backend.
        let gt_rgb = brush_loss::unpack_gt_rgb(gt_packed.clone(), composite_bg);
        // Clean D-SSIM error map.
        let error = error_map::ssim_error_map(pred_rgb, gt_rgb);
        // Masked-mode parity: the photometric loss multiplies its loss-map by
        // `gt.a` in `AlphaMode::Masked` (brush-loss kernel `mask` multiply), so
        // masked-out pixels contribute ZERO loss. Mirror it here BEFORE the
        // map-mean normalize — otherwise a masked region (e.g. sky, `gt.a = 0`)
        // with a bright raw GT vs a near-black pred yields a HIGH D-SSIM error
        // and the growth signal would densify gaussians INTO the masked region,
        // inverting the very mask that exists to suppress it. `gt.a` is bits
        // 24..31 of the packed `[r8 g8 b8 a8]` u32; `>> 24 & 0xff` recovers the
        // byte regardless of the i32 arithmetic-shift sign extension.
        let error = if masked {
            let alpha = gt_packed
                .clone()
                .bitwise_right_shift_scalar(24)
                .bitwise_and_scalar(0xff)
                .float()
                .div_scalar(255.0);
            error * alpha
        } else {
            error
        };
        // MRNF map-mean normalize to ~1.0 so the projected score lands on LFS's
        // native scale.
        let error = error_map::mean_normalize(error);

        let valid = splats.valid();
        let n = valid.num_splats() as usize;
        if n == 0 {
            return;
        }
        // Coverage-weighted MEAN error per gaussian: `(Σ_p T·α·ê)/(Σ_p T·α)`,
        // via a single feat_dim=2 feature backward (numerator = LFS row 1,
        // denominator = LFS row 0 / `densification_weight`). The division makes
        // the score footprint- and resolution-invariant, on the `ê` scale — see
        // `accumulate_error_sample` doc and `crate::error_map` for why the port
        // divides where LFS sums.
        let score = edge::project_coverage_weighted_mean(&valid, error, camera, img_size).await;

        // INVARIANT (same as the edge path): the splat set is constant within a
        // refine window, and `RefineRecord` is recreated fresh after each prune,
        // so the window-MAX starts from zero every window (no cross-window leak).
        // The raw coverage-weighted mean is accumulated by window-MAX here;
        // positive-median normalization is applied ONCE at refine over the final
        // window-MAX (`RefineRecord::error_scores_median_normalized`), NOT
        // per-view — a per-view median normalize is defeated by the window-MAX
        // (the max over ~`refine_every` views of a median-1.0 quantity lands well
        // above 1.0 for ~every gaussian; smoke: still 99.6%). Normalizing the
        // final MAX distribution anchors its median at 1.0 so `τ_err = 1.0`
        // selects the worse-than-median half.
        if let Some(record) = self.refine_record.as_mut() {
            record.gather_error(score);
        }
    }

    /// Run one training step, optionally omitting the refinement-only raster
    /// gradient statistic. Model gradients, visibility, and screen-radius
    /// bookkeeping are always preserved.
    pub async fn step_with_refine_weight(
        &mut self,
        batch: SceneBatch,
        splats: Splats,
        compute_refine_weight: bool,
        global_iter: u32,
    ) -> (Splats, TrainStepStats) {
        let mut splats = splats;

        // Track max SH degree from the first splats we see.
        if self.step_count == 0 {
            self.max_sh_degree = splats.sh_degree();
        }
        self.step_count += 1;

        let [img_h, img_w] = batch.img_size();
        let camera = batch.camera;

        let device = splats.device();

        // TIDI-GS: allocate the floater-suppression state on the first step when
        // EITHER the photometric (`--tidi-prune`) OR the depth
        // (`--tidi-depth-prune`) path is enabled. `ω` is created on the autodiff
        // device (it is a leaf that gets a photometric gradient each step, unused
        // by the depth path); the counts are kept in lockstep with the splats
        // through `keep`/`split` at refine. With NEITHER flag set the state is
        // never allocated, so the whole family stays byte-inert.
        if (self.config.tidi_prune || self.config.tidi_depth_prune) && self.tidi.is_none() {
            self.tidi = Some(TidiState::new(splats.num_splats(), global_iter, &device));
        }
        let has_alpha = batch.has_alpha;
        // GT lives on the GPU as packed `[H, W]` u32 (RGBA u8). All mixing
        // (bg compositing, alpha matching, mask) is folded into the loss
        // kernels; no f32 GT image is ever materialised here.
        // GT is pure data — never differentiated. Build it on the inner
        // backend so it doesn't inherit the autodiff device's residual
        // checkpointing flag (the LPIPS `unpack_gt_rgb` path, via
        // `unwrap_wgpu_int`, expects a clean Wgpu tensor).
        let gt_packed: Tensor<2, Int> =
            Tensor::from_data(batch.img_packed, &device.clone().inner());
        let img_size = glam::uvec2(img_w as u32, img_h as u32);
        let base = &self.config.background_color;
        let base_bg = glam::Vec3::new(base[0], base[1], base[2]);
        let background = sample_background_color(
            base_bg,
            self.config.background_noise_strength,
            &mut self.rng,
        );

        let median_scale = self.bounds.median_size();
        // The first optimizer step stays dense so Adam can initialize its
        // moments. Later steps defer only after the existing state and device
        // have passed every sparse-path compatibility check.
        let defer_sh_grad = self
            .optim
            .as_ref()
            .is_some_and(|optimizer| can_defer_sh_grad(optimizer, &splats));

        // Lift the active view's appearance params onto the autodiff graph
        // for this step.
        let active_appearance = self
            .appearance
            .as_mut()
            .map(|state| state.begin_step(batch.view_index));

        // Contradiction-gate diagnostic sample for this step, if one was taken.
        // Declared out here so the loss block only ever borrows `self`
        // immutably; the sustained-low bookkeeping runs after the block.
        let mut normal_gate_sample: Option<(f32, f32)> = None;

        let (mut grads, visible, num_visible, loss_inner, deferred_sh_grad) = {
            // The splats already carry their 3D-filter floor (set at refine);
            // the render path folds it in. Optimizer/refine work on raw params.
            //
            // TIDI-GS learned-importance gate: when enabled, the render sees each
            // opacity multiplied by `σ(ω)` (paper signal (c)). This is what gives
            // `ω` a photometric gradient — a Gaussian that matters to the image
            // is pushed to `σ(ω)→1`, while the L1 sparsity term lets idle ones
            // decay toward the candidate pool. The gate maps the raw opacity
            // through `logit(σ(raw)·σ(ω))`, so grad still flows to the ORIGINAL
            // opacity leaf (queried by id below); `ω` is exported nowhere. The
            // gate starts near-identity (`σ(ω)≈0.998`), so enabling TIDI barely
            // perturbs the scene, and it is skipped entirely when `--tidi-prune`
            // is off.
            let render_input = match self.tidi.as_ref() {
                Some(tidi) if self.config.tidi_prune => {
                    let mut ri = splats.clone();
                    ri.raw_opacities = ri.raw_opacities.map(|ro| tidi.gate_opacity(ro));
                    ri
                }
                _ => splats.clone(),
            };
            let eff_depth_weight = self.config.depth_weight_at(global_iter);
            let use_depth = batch.depth.is_some() && eff_depth_weight > 0.0;
            // Geometry-prior terms (all inert at their 0.0 defaults, gated
            // exactly like `use_depth`, so a run that does not pass the flags
            // takes the same path it always did).
            //
            // The normal-term ramp (`--normal-ramp-start-iter`) multiplies BOTH
            // normal weights. An exact 0 additionally DROPS `use_prior_normal` /
            // `use_dn`, so the normal render pass is skipped rather than
            // rendered and multiplied by zero — the same philosophy as
            // `--depth-normal-start-iter` below. At the default it is an exact
            // 1.0, so this is byte-identical.
            let normal_ramp = self.config.normal_ramp_at(global_iter);
            let use_prior_normal =
                self.config.normal_loss_weight > 0.0 && batch.normal.is_some() && normal_ramp > 0.0;
            // 2DGS gates this term at 7k of 30k. Gating here rather than just
            // zeroing the weight also skips the depth channel and the normal
            // render pass entirely before the start iteration, so the gate costs
            // nothing instead of rendering work that gets multiplied by zero.
            let dn_started = self.config.depth_normal_start_iter == 0
                || global_iter >= self.config.depth_normal_start_iter;
            // Gate on the EFFECTIVE weight (including the late consistency bump)
            // rather than the base one, so a run that ramps a zero base up to a
            // nonzero end still activates. With the bump off this is exactly
            // `depth_normal_weight`, i.e. byte-identical.
            let eff_dn_weight = self.config.depth_normal_weight_at(global_iter);
            let use_dn = eff_dn_weight > 0.0 && dn_started && normal_ramp > 0.0;
            let use_smooth = self.config.normal_smooth_weight > 0.0;
            let use_normal_render = use_prior_normal || use_dn || use_smooth;
            let use_flatten = self.config.flatten_loss_weight > 0.0;
            // The depth/normal consistency term reads the rendered depth, so it
            // needs the depth channel even without any gt depth map.
            let has_depth_channel = use_depth || use_dn;

            // ---- PGSR depth-source gating, hoisted above the render ----
            //
            // `center` and `plane-aux` leave the MAIN rasterization alone, so
            // for them this block is pure boolean arithmetic and the render
            // below is the call it always was. `plane-fused` (approach B)
            // composites its plane channels IN the main kernel, so the mode and
            // the aux tensor have to be decided before the render runs — which
            // is the only reason these three lines live up here rather than
            // beside the consumers.
            //
            // `normals_from_depth` and the ray-plane grid are both pinhole-only
            // (our fisheye split path is KB4; interior cube faces are Pinhole).
            // Same warn-and-skip contract the consistency term already had.
            let plane_selected = !matches!(self.config.depth_source, DepthSource::Center);
            let is_pinhole = matches!(camera.camera_model, CameraModel::Pinhole);
            let depth_consumer = use_depth || use_dn;
            if plane_selected && depth_consumer && !is_pinhole {
                warn_plane_depth_needs_pinhole();
            }
            let use_plane_depth = plane_selected && depth_consumer && is_pinhole;
            // Approach B. Falls back to nothing: when the plane depth is not
            // usable at all (non-pinhole, or no depth consumer) `plane-fused`
            // takes exactly the path `center` takes, as `plane-aux` does.
            let use_fused =
                use_plane_depth && matches!(self.config.depth_source, DepthSource::PlaneFused);

            let raster_mode = if use_fused {
                RasterizationMode::RgbaDepthPlane
            } else if has_depth_channel {
                RasterizationMode::RgbaAndDepth
            } else {
                RasterizationMode::Rgba
            };

            // The SAME on-tape `[N, 4]` construction approach A rasterizes in a
            // separate feature pass — handed to the main kernel instead. That is
            // deliberate: it makes the A/B delta exactly {weight-path gradients,
            // single-pass fusion} and nothing else, with no duplicated
            // smallest-axis/quaternion math in kernel code.
            //
            // Built from `splats` (not `render_input`) and folded exactly as the
            // aux path folds it, so the two approaches composite identical
            // per-splat values and any forward disagreement is a real bug.
            //
            // PGSR (Chen et al. 2024, arXiv:2406.06521), plane parameterization.
            let plane_aux = use_fused.then(|| {
                let t_fold = match &splats.min_scale {
                    Some(f) => {
                        fold_min_scale(
                            splats.transforms.val(),
                            splats.raw_opacities.val(),
                            f.clone(),
                        )
                        .0
                    }
                    None => splats.transforms.val(),
                };
                plane_features(t_fold, &camera)
            });

            // `render_splats_for_training(..)` IS this with `plane_aux = None`
            // (a direct delegation), so `center` and `plane-aux` are unchanged.
            let diff_out = render_splats_for_training_with_plane_aux(
                render_input,
                &camera,
                img_size,
                background,
                compute_refine_weight,
                raster_mode,
                defer_sh_grad,
                plane_aux,
            )
            .instrument(trace_span!("Forward"))
            .await;

            // The selected per-view appearance correction happens on the
            // rendered image before any loss term sees it, so the splats
            // themselves learn appearance-free colors. Alpha passes through
            // untouched.
            //
            // Appearance correction (PPISP / bilateral grid) models only the
            // color/alpha response and its ISP kernel asserts a 3- or 4-channel
            // input. When a depth-consuming term is active (depth loss, or the
            // depth/normal consistency term) the render carries an extra
            // depth channel (index 4) that is geometry, not color: it must
            // bypass the correction. Split the RGBA channels off, correct
            // those, then re-attach the untouched depth so `pred_image` keeps
            // its [H, W, 5] layout for the depth-loss term below.
            //
            // Under `plane-fused` the same applies to the four plane channels
            // that follow the depth one — also geometry, also not colour. The
            // upper bound is `raster_mode.bwd_out_channels()` rather than a
            // literal so the two cases stay one expression; at `RgbaAndDepth`
            // it is 5, i.e. the `4..5` slice this always did.
            let pred_image = match &active_appearance {
                Some(active) if has_depth_channel => {
                    let geom_end = raster_mode.bwd_out_channels();
                    let rgba = diff_out.img.clone().slice(s![.., .., 0..4]);
                    let depth = diff_out.img.slice(s![.., .., 4..geom_end]);
                    Tensor::cat(vec![active.apply(rgba), depth], 2)
                }
                Some(active) => active.apply(diff_out.img),
                None => diff_out.img,
            };
            let refine_weight_holder = diff_out.refine_weight_holder;
            let deferred_sh_grad = diff_out.deferred_sh_grad;
            let visible = diff_out.visible;
            let max_radius = diff_out.max_radius;

            // RGB loss is `(1 - w) * L1 + (-w) * SSIM` per pixel. Bg
            // compositing always runs in the kernel; for synthesised opaque
            // alpha or zero bg it's a no-op. Mask multiplies the loss-map
            // by `gt.a`; for synthesised opaque alpha that's a no-op too.
            // Alpha matching needs a real alpha source (synthesised
            // a = 1 would pull predicted alpha to fully opaque); we feed
            // `pred` with 4 channels and the kernel's `c == 3` workgroup
            // emits `|pred.a - gt.a|` into the alpha channel.
            let masked_alpha = batch.alpha_mode == AlphaMode::Masked;
            let (l1_w, ssim_w) = if self.ssim_enabled {
                (1.0 - self.config.ssim_weight, -self.config.ssim_weight)
            } else {
                (1.0, 0.0)
            };
            let do_alpha_match = has_alpha && !masked_alpha && self.config.match_alpha_weight > 0.0;
            // Only composite when there's a real alpha channel and a non-zero
            // bg to mix in; the kernel skips the per-pixel `(1-a)*bg` math
            // entirely when this is None.
            let composite_bg = (has_alpha && background != glam::Vec3::ZERO).then_some(background);
            let cfg = ImageLossConfig {
                l1_weight: l1_w,
                ssim_weight: ssim_w,
                composite_bg,
                mask: masked_alpha,
            };
            let pred_for_loss = if do_alpha_match {
                pred_image.clone().slice(s![.., .., 0..4])
            } else {
                pred_image.clone().slice(s![.., .., 0..3])
            };
            let loss_map = image_loss(pred_for_loss, gt_packed.clone(), cfg);

            // `loss` is only reassigned by the LPIPS path below, which is
            // compiled out on wasm — so `mut` is unused there.
            #[cfg_attr(target_family = "wasm", allow(unused_mut))]
            let mut loss = if do_alpha_match {
                let rgb = loss_map.clone().slice(s![.., .., 0..3]).mean();
                let alpha = loss_map.slice(s![.., .., 3..4]).mean();
                rgb + alpha * self.config.match_alpha_weight
            } else {
                loss_map.mean()
            };

            // LPIPS still needs an f32 RGB tensor for VGG. Materialising it
            // here costs ~99 MB at 4K, only when LPIPS is enabled.
            #[cfg(not(target_family = "wasm"))]
            if let Some(lpips) = &self.lpips {
                let gt_rgb = brush_loss::unpack_gt_rgb(gt_packed.clone(), composite_bg);
                let gt_rgb_diff: Tensor<3> = Tensor::from_inner(gt_rgb);
                loss = loss
                    + lpips.lpips(
                        pred_image.clone().slice(s![.., .., 0..3]).unsqueeze_dim(0),
                        gt_rgb_diff.unsqueeze_dim(0),
                    ) * self.config.lpips_loss_weight;
            }

            // Appearance regularisers (bilagrid TV, PPISP param priors).
            if let Some(active) = &active_appearance
                && let Some(reg) = active.reg_loss()
            {
                loss = loss + reg;
            }

            // TIDI-GS: L1 sparsity on the learned importance `σ(ω)`. Balances
            // against the photometric gradient the opacity gate feeds `ω`, so
            // contributing Gaussians settle at `σ(ω)→1` and persistently idle
            // ones decay toward `τ_ω`. Weight 0 leaves `ω` on the photometric
            // gradient only (a 3-signal gate); inert when `--tidi-prune` is off.
            if self.config.tidi_prune
                && self.config.tidi_importance_reg > 0.0
                && let Some(tidi) = self.tidi.as_ref()
            {
                loss = loss + tidi.sparsity_loss() * self.config.tidi_importance_reg;
            }

            // DiG: DINO feature MSE on a rendered feature image (geometry
            // detached, matching the reference), plus a neighbor feature-
            // variance regularizer after warmup.
            if self.config.dino
                && self.config.dino_loss_weight > 0.0
                && let Some((feat_data, feat_c)) = &batch.features
            {
                let feature_dim = self.config.dino_feature_dim as usize;
                let dig = self.dig.get_or_insert_with(|| {
                    DigTrainState::new(splats.num_splats(), feature_dim, *feat_c, &device)
                });
                let gt_dims = feat_data.shape.clone();
                let (gt_h, gt_w) = (gt_dims[0], gt_dims[1]);
                let rescale = self.config.dino_rescale_factor as usize;
                let feat_size = glam::uvec2((gt_w * rescale) as u32, (gt_h * rescale) as u32);
                // Render with the same 3D-filter-folded geometry as the RGB
                // pass; `render_splat_features` detaches it internally.
                let (t_fold, o_fold) = match &splats.min_scale {
                    Some(f) => fold_min_scale(
                        splats.transforms.val(),
                        splats.raw_opacities.val(),
                        f.clone(),
                    ),
                    None => (splats.transforms.val(), splats.raw_opacities.val()),
                };
                let render_mode = if splats.render_mip {
                    brush_render::gaussian_splats::SplatRenderMode::Mip
                } else {
                    brush_render::gaussian_splats::SplatRenderMode::Default
                };
                let feat_img = render_splat_features(
                    t_fold,
                    o_fold,
                    dig.module.features.val(),
                    &camera,
                    feat_size,
                    render_mode,
                )
                .instrument(trace_span!("Feature forward"))
                .await;
                let [fh, fw, _] = feat_img.dims();
                let alpha = feat_img
                    .clone()
                    .slice(s![.., .., feature_dim..feature_dim + 1])
                    .detach();
                let raw = feat_img.slice(s![.., .., 0..feature_dim]);
                let normed = raw / alpha.clamp_min(1e-10);
                let decoded = dig
                    .module
                    .decode(normed.reshape([-1, feature_dim as i32]))
                    .reshape([fh as i32, fw as i32, *feat_c as i32]);

                // Bilinear-upsample the GT feature map to the rendered size
                // (the reference resizes GT up to the rendered resolution).
                let gt: Tensor<3> = Tensor::from_data(feat_data.clone(), &device);
                let gt = gt.permute([2, 0, 1]).unsqueeze::<4>();
                let gt = burn::tensor::module::interpolate(
                    gt,
                    [fh, fw],
                    burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    ),
                );
                let gt = gt.squeeze_dim::<3>(0).permute([1, 2, 0]);

                let dino_loss = (decoded - gt).powi_scalar(2).mean();
                loss = loss + dino_loss * self.config.dino_loss_weight;

                if self.step_count > dig::NN_REG_START_STEP && self.config.dino_nn_reg_weight > 0.0
                {
                    let means = splats.valid().means();
                    let inds = dig.neighbor_indices(&means, &device).await;
                    let n = inds.dims()[0];
                    let nn_feats = dig
                        .module
                        .features
                        .val()
                        .select(0, inds.reshape([(n * dig::NN_K) as i32]))
                        .reshape([n as i32, dig::NN_K as i32, feature_dim as i32]);
                    loss = loss + nn_feats.var(1).sum() * self.config.dino_nn_reg_weight;
                }
            }

            // ---- Geometry feature render (normals, and optionally PGSR planes) ----
            //
            // ONE feature rasterization serves every geometry term. It is
            // hoisted above the depth loss because, with a plane depth source
            // selected, the depth loss consumes its output. Only the RENDER
            // moves; every `loss = loss + ...` accumulation below stays exactly
            // where it was, so the summation order — and therefore the f32
            // result — is unchanged.
            //
            // What gets rendered depends on `--depth-source`:
            //
            //   center (DEFAULT)  3 channels = `splat_normals`, WORLD frame,
            //                     rotated to the camera frame per-pixel after
            //                     compositing. Byte-identical to the previous
            //                     code: the ops below are the same ops in the
            //                     same order.
            //   plane-aux         4 channels = `plane_features` — camera-frame
            //                     unit normal (0..3) + signed plane offset (3).
            //                     Same rasterization COUNT as `center` whenever
            //                     a normal term was already on; one extra pass
            //                     when depth is the only consumer.
            //   plane-fused       the same 4 channels, but composited by the
            //                     MAIN kernel (channels 5..=8, alongside rgba
            //                     and centre depth). NO feature rasterization
            //                     at all — the render above already carries
            //                     them, so this block only re-slices it.
            //
            // Why the plane path needs no per-pixel rotation: `plane_features`
            // rotates each splat's normal into the camera frame BEFORE
            // compositing. A rotation is linear and orthonormal, so
            // `Σwᵢ(R·nᵢ) = R·(Σwᵢnᵢ)` and `normalize(R·v) = R·normalize(v)` —
            // the two orders agree analytically and differ only in f32 rounding.
            // The `center` branch keeps the old order verbatim because that
            // rounding is exactly what the byte-identity gate pins.
            //
            // PGSR (Chen et al. 2024, arXiv:2406.06521), unbiased depth
            // rendering; construction shared with approach B via
            // `plane_features`.
            // `plane_selected` / `use_plane_depth` / `use_fused` and the
            // pinhole warning are decided above the render (approach B needs
            // them to pick the rasterization mode).

            // `(n_cam, normal_alpha)` — the rendered camera-frame normal image
            // and its DETACHED alpha, shared by every normal term below.
            let mut normal_render: Option<(Tensor<3>, Tensor<3>)> = None;
            // `(depth, valid)` — PGSR plane-intersection depth and its validity
            // mask. `None` means the consumers fall back to centre depth.
            let mut plane_depth: Option<(Tensor<2>, Tensor<2>)> = None;

            if use_normal_render || use_plane_depth {
                let render_mode = if splats.render_mip {
                    brush_render::gaussian_splats::SplatRenderMode::Mip
                } else {
                    brush_render::gaussian_splats::SplatRenderMode::Default
                };
                // `plane-fused` re-slices the main render and rasterizes nothing
                // here, so it needs no folded copy; `center` and `plane-aux`
                // both run a feature pass and do. Selecting `render_mode` first
                // is host-side only — no tensor op moved past another, so the
                // `center` op sequence is unchanged.
                let folded = (!use_fused).then(|| match &splats.min_scale {
                    Some(f) => fold_min_scale(
                        splats.transforms.val(),
                        splats.raw_opacities.val(),
                        f.clone(),
                    ),
                    None => (splats.transforms.val(), splats.raw_opacities.val()),
                });

                if use_plane_depth {
                    // Either way the result is the [H, W, 5] `n_sum(3) +
                    // offset_sum(1) + alpha(1)` that `plane_depth_from_features`
                    // contracts for — approaches A and B differ in WHICH kernel
                    // composited it, not in what it means.
                    let feat_img = if use_fused {
                        // Approach B: the main rasterizer already composited the
                        // plane lanes, with the blending-weight gradient path
                        // LIVE (plan section 4.5 row 3). Channel 3 is that
                        // render's coverage alpha, which is the same Σw the
                        // feature pass reports in its trailing channel.
                        //
                        // Offsets come from the shared const fns: re-literalizing
                        // the stride is precisely what broke three call sites
                        // when the depth lane was added.
                        let plane_c = plane_channel_offset(raster_mode.render_depth()) as usize;
                        Tensor::cat(
                            vec![
                                pred_image.clone().slice(s![
                                    ..,
                                    ..,
                                    plane_c..plane_c + PLANE_AUX_LANES_USIZE
                                ]),
                                pred_image.clone().slice(s![.., .., 3..4]),
                            ],
                            2,
                        )
                    } else {
                        // Approach A: a separate feature rasterization of the
                        // same on-tape [N, 4] values.
                        let (t_fold, o_fold) =
                            folded.expect("the non-fused paths always fold min-scale");
                        let feats = plane_features(t_fold.clone(), &camera);
                        render_splat_features(t_fold, o_fold, feats, &camera, img_size, render_mode)
                            .instrument(trace_span!("Plane feature forward"))
                            .await
                    };

                    let normal_alpha = feat_img.clone().slice(s![.., .., 4..5]).detach();

                    let focal = camera.focal(img_size);
                    let center = camera.center(img_size);
                    let (depth, _plane_normal, valid) = plane_depth_from_features(
                        feat_img.clone(),
                        focal.x,
                        focal.y,
                        center.x,
                        center.y,
                        PLANE_MIN_ALPHA,
                        PLANE_MIN_DENOM,
                        PLANE_MIN_DEPTH,
                        PLANE_MAX_DEPTH,
                    );

                    // The returned `_plane_normal` is `normalize(n_sum)`; the
                    // `n_cam` built here is `normalize(n_sum / α)`. Alpha is a
                    // positive scalar per pixel, so the two are the same
                    // direction — we keep `n_cam` so the prior-normal, TV
                    // smoothness and consistency terms all read ONE normal
                    // image, produced by the same expression the `center` path
                    // produces it with. The pixels `_plane_normal` would zero
                    // (grazing / out-of-range) are already dropped downstream:
                    // plane depth is 0 there, so `normals_from_depth` emits
                    // `(0,0,0)` and `depth_normal_loss`'s length gate rejects it.
                    let n_cam = normal_alpha_normalize(
                        feat_img.slice(s![.., .., 0..3]),
                        normal_alpha.clone(),
                    );

                    // Free diagnostic: v1 deliberately keeps the centre depth
                    // channel rendered (§4.2), so the centre-vs-plane residual
                    // costs nothing but a periodic readback. WS-1 measured
                    // centre depth ~2% biased against plane depth on a tilted
                    // slab (≈10 cm at 5 m), so a residual near zero means the
                    // plane path is not actually engaging.
                    if has_depth_channel && global_iter.is_multiple_of(PLANE_RESIDUAL_LOG_EVERY) {
                        let centre = (pred_image.clone().slice(s![.., .., 4..5])
                            / pred_image.clone().slice(s![.., .., 3..4]).clamp_min(1e-10))
                        .reshape([img_h, img_w]);
                        log_plane_vs_centre_residual(
                            depth.clone(),
                            centre,
                            valid.clone(),
                            global_iter,
                        )
                        .await;
                    }

                    normal_render = Some((n_cam, normal_alpha));
                    plane_depth = Some((depth, valid));
                } else {
                    // ---- UNCHANGED `center` path. Do not reorder. ----
                    let (t_fold, o_fold) =
                        folded.expect("the non-fused paths always fold min-scale");
                    let normals = splat_normals(t_fold.clone(), camera.position);
                    let normal_img = render_splat_features(
                        t_fold,
                        o_fold,
                        normals,
                        &camera,
                        img_size,
                        render_mode,
                    )
                    .instrument(trace_span!("Normal forward"))
                    .await;

                    // Same detached alpha normalization as the DiG and depth
                    // paths: the normal terms must not be able to lower their
                    // error by changing transparency.
                    let normal_alpha = normal_img.clone().slice(s![.., .., 3..4]).detach();
                    let n_world =
                        normal_img.slice(s![.., .., 0..3]) / normal_alpha.clone().clamp_min(1e-10);
                    let n_len = n_world
                        .clone()
                        .powi_scalar(2)
                        .sum_dim(2)
                        .sqrt()
                        .clamp_min(1e-6);
                    let n_world = n_world / n_len;

                    // World -> camera. Right-multiplying row vectors by Rᵀ is
                    // the same as left-multiplying column vectors by R.
                    let r_t = world_to_cam_rot_t(&camera, &device);
                    let n_cam = n_world
                        .reshape([(img_h * img_w) as i32, 3])
                        .matmul(r_t)
                        .reshape([img_h, img_w, 3]);

                    normal_render = Some((n_cam, normal_alpha));
                }
            }

            // Depth Disparity L1 loss on rendered expected depth
            if use_depth && let Some(depth_data) = &batch.depth {
                let gt_depth: Tensor<2> = Tensor::from_data(depth_data.clone(), &device);
                // ---- DEPTH SOURCE DISPATCH — backward contracts (§4.5) ----
                //
                // | source      | blending weights | geometry grads via     | opacity reachable |
                // |-------------|------------------|------------------------|-------------------|
                // | center      | detached in-kernel (dropped dot_rgb) + detached α denominator
                // |             |                  | lane 10 -> means z     | NO                |
                // | plane-aux   | CONSTANTS (feature pass; features_bwd.rs tracks
                // |             | only the feature VALUES)
                // |             |                  | feature values -> means (via the plane
                // |             |                  | offset) and quats (via the normal)
                // |             |                  |                        | NO                |
                // | plane-fused | LIVE for the plane lanes (WS-B)          | YES, by design    |
                //
                // SCALES get NO gradient from either plane channel: the
                // thinnest-axis `argmin` inside `splat_normals` is detached, so
                // the normal — and hence the plane — is a function of the
                // quaternion and the axis CHOICE only. That is the same
                // situation today's normal loss is in, and it is deliberate: the
                // choice is a permutation, and differentiating through it means
                // differentiating a discontinuity. `--flatten-loss-weight`
                // remains the scale-side pressure. Do NOT "fix" this by
                // un-detaching the argmin.
                // PGSR plane-intersection depth (arXiv:2406.06521). Invalid
                // pixels come back as exactly 0, which `depth_loss` reads as
                // "no prediction" and scores as a FULL-magnitude disparity
                // error against the GT, still counted in the denominator.
                // Zeroing the GT there instead drops them from the numerator AND
                // the denominator, which is what "no supervision here" has to
                // mean. No alpha division: it cancels between the ray-plane
                // numerator and denominator (see `plane_depth_from_features`).
                let (expected_depth, gt_depth) = if let Some((depth, valid)) = &plane_depth {
                    (depth.clone(), gt_depth * valid.clone())
                } else {
                    let accumulated_depth = pred_image.clone().slice(s![.., .., 4..5]);
                    // Detach the alpha denominator so depth loss cannot lower its
                    // error by changing transparency. A differentiable denominator
                    // lets depth error flow into opacity. This closes one of two
                    // coupling routes; the other lives in the rasterize backward,
                    // where the depth-channel gradient feeds the alpha term (see
                    // rasterize_backwards.rs, the dropped dot_rgb depth term).
                    // Together they detach the blending weights from depth, so depth
                    // supervision moves gaussian positions only. LFS does the same
                    // (detach_depth_weights), as does DN-Splatter. This mirrors the
                    // DINO feature normalization above.
                    let alpha = pred_image.clone().slice(s![.., .., 3..4]).detach();
                    (
                        (accumulated_depth / alpha.clamp_min(1e-10)).reshape([img_h, img_w]),
                        gt_depth,
                    )
                };
                // DN-Splatter gradient-aware weighting: build a per-pixel weight
                // from the GT RGB image (same source `image_loss` consumes),
                // lifted onto the AD graph as a constant like the LPIPS path, so
                // it grows no tape. Composes multiplicatively with the annealed
                // scalar weight outside. Only materialises the f32 GT when on.
                let pixel_weight = if self.config.depth_grad_aware {
                    let gt_rgb: Tensor<3> = Tensor::from_inner(brush_loss::unpack_gt_rgb(
                        gt_packed.clone(),
                        composite_bg,
                    ));
                    Some(rgb_grad_weight(gt_rgb, self.config.depth_grad_sigma))
                } else {
                    None
                };
                loss = loss + depth_loss(expected_depth, gt_depth, pixel_weight) * eff_depth_weight;
            }

            // Depth-coupled opacity regularizer (3D distance-to-cloud gate). For
            // every Gaussian whose centre sits FAR (> margin) from the seed/LiDAR
            // point cloud in 3D, add a per-step penalty whose ONLY gradient path
            // is the activated opacity, so floaters in empty space fade out
            // smoothly (their colour redistributes into on-surface splats) instead
            // of being hard-deleted (which orphans that colour and leaves a black
            // halo). Positions/field detached. VIEW-INDEPENDENT — no per-frame
            // depth or camera projection, just the static cloud grid (built once
            // from the seed cloud in `train_stream`); this replaces the old
            // per-view z-buffer residual, which leaked background through
            // foreground gaps and marked surface splats as floating. Gated on
            // `--depth-opacity-reg-start-iter` so densification can finish
            // backfilling opacity-faded regions first. Inert (no lookup, no term)
            // when the weight is 0 or before the start iter.
            if opacity_reg_active(
                global_iter,
                self.config.depth_opacity_reg_start_iter,
                self.config.depth_opacity_reg_weight,
            ) {
                match &self.opacity_reg_grid {
                    Some(grid) => {
                        let term = crate::tidi::depth_opacity_reg_loss(
                            splats.raw_opacities.val(),
                            splats.means(),
                            grid,
                            self.config.depth_opacity_reg_margin,
                            self.config.depth_opacity_reg_softness,
                        );
                        loss = loss + term * self.config.depth_opacity_reg_weight;
                    }
                    None => warn_depth_opacity_reg_no_cloud(),
                }
            }

            // Co-planarity constraint (FIX 2, `--plane-coplanarity-weight`): for
            // every Gaussian assigned to a RANSAC seed-cloud plane, pull its centre
            // onto the plane and flatten it against the plane. Unlike the opacity
            // gate above this is a real GEOMETRY gradient (position + scale +
            // rotation), so it directly removes the photometric rank deficiency on
            // featureless walls. Assignment is detached; only means/scales/rotations
            // carry gradient. Inert (no term) when the weight is 0 or no planes were
            // extracted.
            if self.config.plane_coplanarity_weight > 0.0
                && let Some(planes) = &self.plane_set
            {
                let assign = crate::tidi::resolve_coplanarity_assign_dist(
                    self.config.plane_coplanarity_assign_dist,
                    planes.spacing,
                );
                if let Some(term) = plane_coplanarity_loss(
                    splats.means(),
                    splats.rotations(),
                    splats.scales(),
                    planes,
                    assign,
                    &device,
                ) {
                    loss = loss + term * self.config.plane_coplanarity_weight;
                }
            }

            // Geometry priors: the normal half of DN-Splatter / PlanarGS.
            //
            // The rendered normal image reuses `render_splat_features`, the same
            // vehicle the DiG path uses: it detaches geometry internally and
            // back-props into the FEATURE values, so feeding it per-splat
            // normals derived from the quaternions makes this loss rotate
            // gaussians. No new kernel is involved.
            if use_normal_render {
                // The render block above runs whenever `use_normal_render ||
                // use_plane_depth`, so this is unreachable — `expect` rather
                // than an `if let` because a silently-skipped normal term is a
                // regression that no test failure would announce: the run just
                // trains without the supervision it was configured for.
                let (n_cam, normal_alpha) = normal_render
                    .as_ref()
                    .expect("use_normal_render implies the feature render ran");
                let n_cam = n_cam.clone();
                let normal_alpha = normal_alpha.clone();

                if use_prior_normal && let Some(normal_data) = &batch.normal {
                    let gt_normal: Tensor<3> = Tensor::from_data(normal_data.clone(), &device);
                    // NeuRIS per-pixel contradiction gate (arXiv:2206.13597);
                    // the 30 degree value and its arming step come from the
                    // `gauss-surf` PGSR trainer (rerun-io/examples-monorepo,
                    // Apache-2.0, by Pablo Vela). `None` at the default is
                    // literally the pre-gate code path.
                    let gate_cos = self.config.normal_gate_cos_at(global_iter);

                    // Over-masking diagnostic (plan §4.7). Built ONLY on the
                    // sampling stride and ONLY when the gate is on, so a default
                    // run constructs no tensor and reads nothing back. The
                    // readback is a real device sync, which is why it rides the
                    // refine-derived cadence rather than firing every step; the
                    // condition it detects is by definition not a single-step
                    // event. Counts are stashed here and acted on after this
                    // block, so the loss path keeps its immutable borrow of
                    // `self`.
                    if let Some(gate_cos) = gate_cos
                        && self.should_sample_normal_gate(global_iter)
                    {
                        let counts = brush_loss::normal_gate_counts(
                            n_cam.clone(),
                            gt_normal.clone(),
                            gate_cos,
                        );
                        if let Ok(data) = counts.inner().into_data_async().await
                            && let Ok(v) = data.to_vec::<f32>()
                            && v.len() == 2
                        {
                            normal_gate_sample = Some((v[0], v[1]));
                        }
                    }

                    loss = loss
                        + normal_loss(n_cam.clone(), gt_normal, gate_cos)
                            * (self.config.normal_loss_weight * normal_ramp);
                }

                // TV smoothness on the rendered normal image. Needs no prior
                // data and no depth channel, so it is deliberately NOT gated on
                // an iteration: DN-Splatter runs its normal terms ungated.
                if use_smooth {
                    loss = loss
                        + normal_smooth_loss(n_cam.clone(), normal_alpha.clone())
                            * self.config.normal_smooth_weight;
                }

                if use_dn {
                    // Unprojection is pinhole-only for now; our fisheye-split
                    // path is KB4, interior cube faces are Pinhole. Skip with a
                    // warning rather than silently supervising with wrong math.
                    if is_pinhole {
                        // Same dispatch as the depth loss above, and for the
                        // same reason: PGSR's consistency term compares the
                        // rendered plane normal against normals differentiated
                        // from the PLANE depth, not from the centre depth. No
                        // extra masking is needed — invalid plane pixels are
                        // exactly 0, `normals_from_depth` requires all three
                        // contributing depths to be positive, and
                        // `depth_normal_loss` then drops the `(0,0,0)` it emits.
                        let expected_depth = if let Some((depth, _valid)) = &plane_depth {
                            depth.clone()
                        } else {
                            let accumulated_depth = pred_image.clone().slice(s![.., .., 4..5]);
                            let alpha = pred_image.clone().slice(s![.., .., 3..4]).detach();
                            (accumulated_depth / alpha.clamp_min(1e-10)).reshape([img_h, img_w])
                        };
                        let focal = camera.focal(img_size);
                        let center = camera.center(img_size);
                        let n_from_depth = normals_from_depth(
                            expected_depth,
                            focal.x,
                            focal.y,
                            center.x,
                            center.y,
                        );
                        loss = loss
                            + depth_normal_loss(n_from_depth, n_cam, normal_alpha)
                                * (eff_dn_weight * normal_ramp);
                    } else {
                        warn_depth_normal_needs_pinhole();
                    }
                }
            }

            // Flattening pressure (PlanarGS `L_s`): the mean smallest activated
            // scale, on the RAW pre-3D-filter scales. The Mip filter floors the
            // RENDERED thinness; the penalty deliberately acts on the learned
            // scale so exports keep the thin axis and we do not fight the
            // anti-aliasing floor. MRNF's prune keys on `scale_max`, so there is
            // no interaction with it.
            //
            // ---- `--normalize-metric-weights` (default off = exact 1.0) ----
            //
            // These two terms are the only ones in the loss whose VALUE carries
            // physical units, so they are the only ones whose weight has to be
            // divided by the scene scale for a recipe to transfer between scenes
            // of different physical size:
            //
            //   * flatten  = mean(min activated scale)      -> metres    -> / s
            //   * scale-reg = mean(s² above a threshold)    -> metres²   -> / s²
            //     and its THRESHOLD is itself a length      -> metres    -> × s
            //
            // **`--depth-loss-weight` is deliberately NOT in this list, and must
            // not be "fixed" to match the reference.** The `gauss-surf` PGSR
            // trainer (rerun-io/examples-monorepo, Apache-2.0, by Pablo Vela)
            // divides its depth weight by the scene scale because its depth loss
            // is a metric L1 in metres. Ours is DISPARITY-space (`1/m`,
            // `brush-loss/src/lib.rs` `depth_loss`), so its residual scales as
            // `1/s`, not `s` — dividing by `s` would move the effective weight
            // the WRONG WAY by a factor of `s²`. The dimensionless weights
            // (`--normal-loss-weight`, `--depth-normal-weight`,
            // `--normal-smooth-weight`, `--anti-needle-weight`) need no
            // normalization for the same reason, and get none.
            let metric_scale = self.metric_weight_scale();
            if use_flatten {
                let scales = splats.transforms.val().slice(s![.., 7..10]).exp();
                loss = loss
                    + scales.min_dim(1).mean() * (self.config.flatten_loss_weight / metric_scale);
            }

            // Scale-explosion + anti-needle regularizers (Stipple, arXiv:2608.00931).
            // Differentiable PREVENTION of the MRNF scale blow-up that our
            // prune-side guards only remove after the fact. Both act on the RAW
            // pre-3D-filter scales, like the flatten term above. Default-off.
            if self.config.scale_reg_weight > 0.0 {
                let scales = splats.transforms.val().slice(s![.., 7..10]).exp();
                loss = loss
                    + crate::tidi::scale_reg_loss(
                        scales,
                        self.config.scale_reg_threshold * metric_scale,
                    ) * (self.config.scale_reg_weight / (metric_scale * metric_scale));
            }
            if self.config.anti_needle_weight > 0.0 {
                let log_scales = splats.transforms.val().slice(s![.., 7..10]);
                loss = loss
                    + crate::tidi::anti_needle_loss(log_scales) * self.config.anti_needle_weight;
            }

            // Strip the autodiff graph off the loss so consumers can read the
            // scalar later without keeping the backward pass alive.
            let loss_inner = loss.clone().inner();

            // ---- Total-loss finiteness guard ----
            //
            // Deliberately BEFORE the backward and the optimizer step: a NaN
            // loss produces NaN gradients, which the optimizer then writes into
            // every parameter it touches. Checking here means the run aborts
            // with the parameters still clean, so whatever the caller does next
            // (checkpoint, inspect) sees the last good state rather than a
            // scene that has already been overwritten with NaN.
            //
            // Cadence and its rationale: `NONFINITE_LOSS_CHECK_STEPS`.
            if self.should_check_loss_finite(global_iter) {
                let loss_value: f32 = loss_inner
                    .clone()
                    .into_scalar_async()
                    .await
                    .expect("total-loss readback for the finiteness guard");
                if !loss_value.is_finite() {
                    self.report_nonfinite_loss(loss_value, &splats, global_iter)
                        .await;
                }
            }

            let mut grads = splats.bwd_validate(loss).await;

            let deferred_sh_grad = deferred_sh_grad.map(|handle| {
                handle
                    .take(&mut grads)
                    .expect("deferred SH gradient holder was not populated")
            });

            trace_span!("Housekeeping").in_scope(|| {
                // Refine state accumulates on the inner (non-autodiff) device
                // so we can mix it with `.inner()`-stripped gradients/aux
                // without crossing backends. `detach_autodiff` also clears
                // the residual `checkpointing` flag that bare `.inner()`
                // leaves behind (see `brush_render::burn_glue`).
                use brush_render::burn_glue::detach_autodiff;
                let device = splats.device().inner();
                let record = self
                    .refine_record
                    .get_or_insert_with(|| RefineRecord::new(splats.num_splats(), &device));
                // `visible` / `max_radius` already arrive on the inner backend;
                // only a freshly-extracted `refine_weight` gradient needs the
                // autodiff stripped off. Once growth stops, it is no longer
                // consumed, but visibility and screen size still feed pruning
                // and oversized-splat splitting.
                if compute_refine_weight {
                    let refine_weight = refine_weight_holder
                        .grad_remove(&mut grads)
                        .expect("XY gradients need to be calculated.");
                    record.gather_stats(
                        detach_autodiff(refine_weight),
                        visible.clone(),
                        max_radius,
                    );
                } else {
                    record.gather_aux_stats(visible.clone(), max_radius);
                }
            });

            // Edge-guidance accumulation (MRNF port, delta #4). Uses the
            // render-time splats + this view's GT and camera; gated on
            // `--use-edge-map` and the sampling stride inside the method.
            self.accumulate_edge_sample(&splats, &camera, &gt_packed, composite_bg, img_size)
                .await;

            // Error-map growth accumulation (MRNF `use_error_map` port). Uses
            // the post-appearance predicted RGB (the same image the SSIM loss
            // sees) detached onto the inner backend, plus this view's GT; gated
            // on `--error-map-densification` inside. Runs every step (LFS folds
            // the error map into `_refine_weight_max` on every view). `masked_alpha`
            // applies the same `gt.a` mask the photometric loss uses.
            let pred_rgb = brush_render::burn_glue::detach_autodiff(pred_image.clone().slice(s![
                ..,
                ..,
                0..3
            ]));
            self.accumulate_error_sample(
                &splats,
                &camera,
                pred_rgb,
                &gt_packed,
                composite_bg,
                masked_alpha,
                img_size,
            )
            .await;

            // TIDI depth / LiDAR-residual prune: fold THIS view's depth residual
            // into the persistent float/valid counters (`TidiState::accumulate_depth`).
            // Deliberately independent of `--depth-loss-weight` — it only reuses
            // the same per-frame depth tensor the depth loss consumes, not the
            // loss. No-op when the batch carries no depth (nerfstudio
            // `depth: None`); a run that set `--tidi-depth-prune` with no depth at
            // all warns once at prune time and stays inert.
            if self.config.tidi_depth_prune
                && let Some(depth_data) = &batch.depth
            {
                let margin = self.config.tidi_depth_margin;
                if let Some(tidi) = self.tidi.as_mut() {
                    tidi.accumulate_depth(
                        splats.means(),
                        depth_data.clone(),
                        &camera,
                        img_size,
                        margin,
                    );
                }
            }

            (
                grads,
                visible,
                diff_out.num_visible,
                loss_inner,
                deferred_sh_grad,
            )
        };

        // Contradiction-gate over-masking diagnostic (plan §4.7). `None` on
        // every step of a default run, and on every non-sampling step even when
        // the gate is on.
        if let Some((surviving, valid)) = normal_gate_sample {
            self.record_normal_gate_sample(global_iter, surviving, valid);
        }

        // The optimizer strips autodiff before stepping, so optimizer state
        // (scaling, momentum) lives on the inner device.
        let opt_device = device.clone().inner();
        let optimizer =
            self.optim.get_or_insert_with(|| {
                let sh_degree = splats.sh_degree();
                let num_coeffs = sh_coeffs_for_degree(sh_degree) as usize;

                // DC (band 0) uses full LR; bands 1+ are scaled down.
                let mut scales = vec![1.0f32; num_coeffs];
                let rest_scale = 1.0 / self.config.lr_coeffs_sh_scale;
                for s in &mut scales[1..] {
                    *s = rest_scale;
                }
                let sh_lr_scales = Tensor::<1>::from_floats(scales.as_slice(), &opt_device)
                    .reshape([1, num_coeffs as i32, 1]);

                SplatOptim {
                    adam: AdamScaled::new(1e-15),
                    transforms: AdamState::new(None, false),
                    sh_coeffs: AdamState::new(Some(sh_lr_scales), true),
                    opacities: AdamState::new(None, false),
                }
            });

        let lr_mean = self.sched_mean.step() * median_scale as f64;
        // MRNF LR schedule (R1): step the independent scale-LR schedule in
        // lock-step with the mean schedule (LFS mrnf.cpp:1360).
        let lr_scale = self.sched_scale.step();

        // Update per-component LR scaling for the transforms param.
        // transforms layout: means(3) + rotations(4) + log_scales(3)
        // We use base_lr=1.0 and encode actual LRs in the scaling tensor. Adam
        // momentum persists across steps in `optimizer.transforms.momentum`; only
        // the scaling tensor is swapped here to follow the LR schedules.
        {
            let lr_values: [f32; 10] = [
                lr_mean as f32,
                lr_mean as f32,
                lr_mean as f32,
                self.config.lr_rotation as f32,
                self.config.lr_rotation as f32,
                self.config.lr_rotation as f32,
                self.config.lr_rotation as f32,
                lr_scale as f32,
                lr_scale as f32,
                lr_scale as f32,
            ];
            optimizer.transforms.scaling =
                Some(Tensor::<1>::from_floats(lr_values.as_slice(), &opt_device).reshape([1, 10]));
        }

        splats = trace_span!("Optimizer step").in_scope(|| {
            splats.transforms = trace_span!("Transforms step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    1.0,
                    splats.transforms,
                    &mut optimizer.transforms,
                    &mut grads,
                )
            });
            splats = trace_span!("SH Coeffs step").in_scope(|| {
                step_sh_coeffs(
                    optimizer,
                    splats,
                    &mut grads,
                    deferred_sh_grad,
                    self.config.lr_coeffs_dc,
                )
            });
            splats.raw_opacities = trace_span!("Opacity step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    self.config.lr_opac,
                    splats.raw_opacities,
                    &mut optimizer.opacities,
                    &mut grads,
                )
            });
            splats
        });

        // Appearance optimizer step: the active view's bilateral grid gets a
        // sparse Adam update and the PPISP params a dense one, each on its
        // own warmup + exp-decay LR schedule.
        if let (Some(state), Some(active)) = (self.appearance.as_mut(), active_appearance) {
            trace_span!("Appearance step").in_scope(|| {
                state.end_step(active, &mut grads);
            });
        }

        if let Some(dig) = &mut self.dig {
            trace_span!("DiG step").in_scope(|| {
                let lr = dig::dig_lr(
                    self.step_count,
                    self.config.dino_lr,
                    self.config.dino_lr_end,
                );
                // Two parameter groups with opposite update characteristics, so
                // they get separate schedules. `lr` above is the decoder's.
                let feat_lr = dig::dig_lr(
                    self.step_count,
                    self.config.dino_feature_lr,
                    self.config.dino_feature_lr_end,
                );
                let module = dig.module.clone();
                let grad_feat =
                    GradientsParams::from_params(&mut grads, &module, &[module.features.id]);
                let module = dig.optim.step(feat_lr, module, grad_feat);
                let grad_mlp =
                    GradientsParams::from_params(&mut grads, &module, &module.mlp_param_ids());
                dig.module = dig.optim.step(lr, module, grad_mlp);
            });
        }

        // TIDI-GS: one Adam step on the learned importance `ω` from the same
        // backward pass (photometric gate + L1 sparsity). `ω` participates in
        // every step's graph via the opacity gate ONLY on the photometric path
        // (`--tidi-prune`), so its grad is present exactly then. Gate the step on
        // `tidi_prune`: on a depth-only run (`--tidi-depth-prune` alone) the state
        // exists but `ω` never entered the graph, so there is nothing to step.
        if self.config.tidi_prune
            && let Some(tidi) = &mut self.tidi
        {
            trace_span!("TIDI importance step").in_scope(|| {
                tidi.optimize(self.config.tidi_importance_lr, &mut grads);
            });
        }

        // Add random noise to the means of low-opacity gaussians. Only do this
        // in the growth phase, otherwise let the splats settle in without
        // noise — not much point exploring regions anymore. The noise gate is
        // non-differentiable bookkeeping: read opacity from the valid (inner)
        // splats so the sigmoid never lands on the autodiff graph, and the
        // visibility gate is already inner — so nothing here builds a node that
        // won't get a backward pass.
        //
        // MRNF port (R2): LFS injects mean-noise EVERY step from `post_backward`
        // (mrnf.cpp:617), the same frequency as this generic block. When
        // `--mrnf-noise-injection` is set we change the GATING (not the location
        // or frequency): the per-gaussian gate becomes the ACCUMULATED
        // per-refine-window visibility (`RefineRecord::vis_weight > 0`, LFS
        // `_vis_count > 0`) instead of the single-step `visible` mask, plus a
        // bounds-valid gate (LFS `_bounds_valid`) that skips injection until the
        // robust median extent is finite and positive.
        if self.config.mrnf_noise_injection {
            // Bounds-valid gate (LFS `_bounds_valid`): skip entirely until the
            // robust per-axis bounds have a finite, positive median extent.
            if median_scale.is_finite() && median_scale > 0.0 {
                let num_splats = splats.num_splats() as usize;
                let inv_opac: Tensor<1> = 1.0 - splats.valid().opacities();
                // Accumulated per-refine-window visibility gate (LFS
                // `_vis_count > 0`). `RefineRecord::vis_mask()` already lives on
                // the inner device. Fall back to the single-step `visible`
                // tensor when the record is missing or its length no longer
                // matches the current splat count (first steps of a window after
                // a count change) — the record is recreated post-refine so
                // mid-window alignment normally holds.
                let vis_gate = match self.refine_record.as_ref() {
                    Some(record) if record.vis_weight.dims()[0] == num_splats => {
                        record.vis_mask().float()
                    }
                    _ => visible,
                };
                let noise_weight =
                    (inv_opac.powi_scalar(150.0).clamp(0.0, 1.0) * vis_gate).unsqueeze_dim(1);
                // `samples` is pure data — keep it on the inner device so it can
                // multiply with the inner `noise_weight` without crossing backends.
                let samples = Tensor::random(
                    [num_splats, 3],
                    Distribution::Normal(0.0, 1.0),
                    &splats.device().inner(),
                );
                // Scale by THIS step's stepped mean LR (already median-size-
                // folded, as LFS's optimizer Means LR); mean_lr already decays
                // over time.
                let noise_weight_means =
                    noise_weight * (lr_mean as f32 * self.config.mean_noise_weight);

                splats.transforms = splats.transforms.map(|t| {
                    // Clamp travel to one robust median box (LFS clamps per-dim
                    // noise to +/- median_size).
                    let noise_m = (samples * noise_weight_means).clamp(-median_scale, median_scale);
                    let inner = t.inner();
                    let noised_means = inner.clone().slice(s![.., 0..3]) + noise_m;
                    let out = inner.slice_assign(s![.., 0..3], noised_means);
                    Tensor::from_inner(out).require_grad()
                });
            }
        } else {
            let inv_opac: Tensor<1> = 1.0 - splats.valid().opacities();
            let noise_weight = inv_opac.powi_scalar(150.0).clamp(0.0, 1.0) * visible;
            let noise_weight = noise_weight.unsqueeze_dim(1);
            // `samples` is pure data — keep it on the inner device so it can
            // multiply with the `.inner()`-stripped `noise_weight` without
            // crossing backends.
            let samples = Tensor::random(
                [splats.num_splats() as usize, 3],
                Distribution::Normal(0.0, 1.0),
                &splats.device().inner(),
            );

            // Could scale by train time, but, the mean_lr already decays over time.
            let noise_weight_means =
                noise_weight * (lr_mean as f32 * self.config.mean_noise_weight);

            // Add noise to the means portion (cols 0..3), and optionally scales
            // (cols 7..10) and rotations (cols 3..7).
            splats.transforms = splats.transforms.map(|t| {
                // Only allow noised gaussians to travel at most the entire extent of the current bounds.
                let noise_m = (samples * noise_weight_means).clamp(-median_scale, median_scale);
                let inner = t.inner();
                // slice + slice_assign with a clone of inner avoids holding two
                // refs across slice_assign — `inner` is consumed by slice_assign
                // and the resulting buffer is the only writer.
                let noised_means = inner.clone().slice(s![.., 0..3]) + noise_m;
                let out = inner.slice_assign(s![.., 0..3], noised_means);
                Tensor::from_inner(out).require_grad()
            });
        } // end per-step noise block

        let stats = TrainStepStats {
            num_visible,
            lr_mean,
            lr_rotation: self.config.lr_rotation,
            lr_scale: self.config.lr_scale,
            lr_coeffs: self.config.lr_coeffs_dc,
            lr_opac: self.config.lr_opac,
            loss: loss_inner,
        };

        (splats, stats)
    }

    /// Whether this step's total loss should be read back and checked.
    ///
    /// See [`NONFINITE_LOSS_CHECK_STEPS`] for why this is sampled rather than
    /// unconditional, and why the sampled cadence is the refine cadence.
    fn should_check_loss_finite(&self, global_iter: u32) -> bool {
        if self.step_count <= NONFINITE_LOSS_CHECK_STEPS {
            return true;
        }
        // The refine cadence is a step that ALREADY synchronises (the refine
        // prune reads its own counts back), so the marginal cost here is one
        // extra scalar rather than a new stall.
        global_iter.is_multiple_of(self.config.refine_every.max(1))
    }

    /// Report — and by default ABORT on — a non-finite total loss.
    ///
    /// The reference trainer raises `FloatingPointError` on any non-finite loss
    /// term rather than warning, and that is the right posture: a step whose
    /// loss is NaN writes NaN gradients into the parameters, and every
    /// subsequent step trains on garbage. A run that continues past this point
    /// has no usable output, so continuing only wastes GPU hours and, worse,
    /// produces a file that LOOKS like a deliverable.
    ///
    /// `--allow-nonfinite-loss` restores the old continue-anyway behaviour for
    /// debugging the poisoning itself.
    async fn report_nonfinite_loss(&self, loss_value: f32, splats: &Splats, global_iter: u32) {
        // This path is already failing, so the extra readbacks below are free in
        // any sense that matters. WHICH parameter group is non-finite is the
        // only cheap diagnostic available — the per-TERM loss breakdown is not,
        // because the terms are accumulated into one scalar (`loss = loss + ..`)
        // and are not retained separately. Splat-side counts still separate the
        // two cases that matter: parameters already poisoned before this step
        // (counts > 0) versus a loss that went non-finite on clean parameters
        // (counts == 0, i.e. look at the batch and the loss terms instead).
        let counts = non_finite_splat_masks(splats).counts().await;
        log::error!(
            "NON-FINITE TOTAL LOSS ({loss_value}) at iter {global_iter}. {}",
            counts.report(global_iter, "loss guard")
        );

        assert!(
            self.config.allow_nonfinite_loss,
            "non-finite total loss ({loss_value}) at iter {global_iter}: aborting before the \
             backward pass writes NaN gradients into the parameters. Splat parameters \
             non-finite at this point: {} of {} [transforms {} | sh {} | opacity {}]. \
             Training past this point produces an export that cannot be used — one \
             non-finite value poisons an entire SOG codebook downstream. Pass \
             --allow-nonfinite-loss to continue anyway (for debugging the cause only; \
             the result is not a deliverable).",
            counts.any, counts.total, counts.transforms, counts.sh, counts.opacities
        );

        log::error!(
            "--allow-nonfinite-loss is set: continuing to train on a non-finite loss. \
             The resulting splats are NOT a deliverable."
        );
    }

    /// Prune non-finite splats OUTSIDE the refine cadence.
    ///
    /// # The gap this closes
    ///
    /// The non-finite prune inside `refine_for_phase` only runs on refine
    /// steps, so any splat that goes NaN/inf AFTER the last refinement survives
    /// all the way to export. Measured on `ARKitScenes`: **14-25 non-finite
    /// splats in every exported ply**. That number was only ever visible
    /// because `analyze/splatstats/` counts them after the fact; the trainer
    /// should own it, which is what the log line here does.
    ///
    /// Downstream this is not cosmetic — one non-finite value poisons an entire
    /// SOG codebook at Stage 7, which is a whole-scene failure, not a
    /// three-splat one.
    ///
    /// # Cost, and where to call it
    ///
    /// Counting synchronises with the GPU, so this is for cadences that already
    /// sync (eval) or paths where a sync is free relative to what follows
    /// (immediately before an export writes a file).
    ///
    /// # Byte-identity
    ///
    /// When nothing is non-finite this returns EARLY — before taking the
    /// optimizer, before touching the refine record, before `prune_points`.
    /// A clean run therefore executes exactly the sequence it did before this
    /// method existed, and only the (value-free) count readback is added.
    pub async fn prune_non_finite_splats(
        &mut self,
        iter: u32,
        splats: Splats,
        site: &str,
    ) -> (Splats, u32) {
        let masks = non_finite_splat_masks(&splats);
        let counts = masks.counts().await;
        if counts.any == 0 {
            return (splats, 0);
        }
        log::warn!("{}", counts.report(iter, site));

        // The optimizer is created lazily on the first step. Without it there
        // is no moment state to keep in lockstep, but `prune_points` needs one
        // to reindex; nothing has trained yet either, so leave it alone.
        let Some(mut optim) = self.optim.take() else {
            log::warn!(
                "non-finite splats found at iter {iter} ({site}) before the optimizer exists; \
                 leaving them in place"
            );
            return (splats, 0);
        };

        // `refine_record` is None only right after a refine consumed it; `step`
        // recreates it on the next iteration regardless, so a fresh zeroed
        // record loses nothing that was going to survive anyway.
        let device = splats.device();
        let refiner = self
            .refine_record
            .take()
            .unwrap_or_else(|| RefineRecord::new(splats.num_splats(), &device));

        let (splats, refiner, pruned) = prune_points(
            splats,
            &mut optim,
            refiner,
            masks.any(),
            self.dig.as_mut(),
            self.tidi.as_mut(),
        )
        .await;
        self.optim = Some(optim);
        self.refine_record = Some(refiner);

        log::warn!(
            "pruned {pruned} non-finite splats at iter {iter} ({site}); {} remain",
            splats.num_splats()
        );
        (splats, pruned)
    }

    pub async fn refine(&mut self, iter: u32, splats: Splats) -> (Splats, RefineStats) {
        self.refine_for_phase(iter, iter, self.config.total_train_iters, splats)
            .await
    }

    /// Refine using a global iteration for densification gates and a separate
    /// phase-local iteration for schedules that restart in each LOD phase.
    pub async fn refine_for_phase(
        &mut self,
        global_iter: u32,
        phase_iter: u32,
        phase_total: u32,
        splats: Splats,
    ) -> (Splats, RefineStats) {
        // Keep the floor auxiliary while prune decisions are made so effective
        // scales/opacities remain visible. It is cleared immediately before
        // canonical parameters change and replaced after positions/count are
        // final; baking here would accumulate the filter at every refinement
        // and leave Adam moments inconsistent with the rewritten parameters.
        let device = splats.device();

        let refiner = self
            .refine_record
            .take()
            .expect("Can only refine if refine stats are initialized");

        // Track how many splats are visually large (the "big-low-α" failure
        // mode). `max_screen_size` is the larger 2D ellipse extent as a
        // fraction of the image dim; area is approximated by its square.
        if log::log_enabled!(log::Level::Debug) {
            let ss_data = refiner
                .max_screen_size
                .clone()
                .into_data_async()
                .await
                .expect("Failed to read screen size")
                .into_vec::<f32>()
                .expect("Failed to read screen size vec");
            let mut sorted: Vec<f32> = ss_data.iter().copied().filter(|v| v.is_finite()).collect();
            if !sorted.is_empty() {
                sorted.sort_by(|a, b| a.total_cmp(b));
                let n = sorted.len();
                let pct = |p: f32| sorted[((p * (n - 1) as f32) as usize).min(n - 1)];
                let n_total = n as f64;
                let n_gt_025 = sorted.iter().filter(|v| **v > 0.25).count();
                let n_gt_010 = sorted.iter().filter(|v| **v > 0.10).count();
                let n_gt_005 = sorted.iter().filter(|v| **v > 0.05).count();
                let n_area_gt_005 = sorted.iter().filter(|v| (*v * *v) > 0.05).count();
                let n_area_gt_010 = sorted.iter().filter(|v| (*v * *v) > 0.10).count();
                log::debug!(
                    "screen_size iter={} n={} max_dim p50={:.4} p95={:.4} p99={:.4} max={:.4} frac>0.05={:.4} frac>0.10={:.4} frac>0.25={:.4} frac_area>0.05={:.4} frac_area>0.10={:.4}",
                    global_iter,
                    n,
                    pct(0.5),
                    pct(0.95),
                    pct(0.99),
                    pct(1.0),
                    n_gt_005 as f64 / n_total,
                    n_gt_010 as f64 / n_total,
                    n_gt_025 as f64 / n_total,
                    n_area_gt_005 as f64 / n_total,
                    n_area_gt_010 as f64 / n_total,
                );
            }
        }

        let max_allowed_bounds = self.bounds.extent.max_element() * self.config.prune_extent_factor;

        // If not refining, update splat to step with gradients applied.
        // Prune dead splats. This ALWAYS happen even if we're not "refining" anymore.
        let mut optim = self
            .optim
            .take()
            .expect("Can only refine after optimizer is initialized");
        let alpha_mask = splats.opacities().lower_elem(self.config.min_opacity);
        let scales = splats.scales();

        // Note: we do NOT cull on a minimum scale. A genuinely flat splat
        // (a thin "pancake" representing a surface) legitimately has a tiny
        // smallest axis, so there's no correct min-scale threshold — the
        // non-finite check below still removes actually-degenerate splats.
        let scale_big = scales
            .clone()
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Remove splats that are way out of bounds.
        let center = self.bounds.center;
        let bound_center =
            Tensor::<1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        // Out-of-bounds prune. Default: per-axis (L-inf / Chebyshev) test — a
        // splat is culled if ANY axis' |distance from center| exceeds the
        // bound. This per-axis default is what MRNF actually does: LFS computes
        // `dist_from_center = (means - center).abs().max(1)` (L-inf) then culls
        // `dist_from_center > max_allowed` (mrnf.cpp:663-669) — it is NOT
        // radial. `--radial-bounds-prune` switches to the L2 radial distance
        // instead; since L2 >= L-inf it prunes a superset, so it is a STRICTER
        // divergence experiment, not MRNF parity.
        let bound_mask = if self.config.radial_bounds_prune {
            let diff = splats.means() - bound_center;
            let radial = diff.powi_scalar(2).sum_dim(1).sqrt().squeeze_dim(1);
            radial.greater_elem(max_allowed_bounds)
        } else {
            let splat_dists = (splats.means() - bound_center).abs();
            splat_dists
                .greater_elem(max_allowed_bounds)
                .any_dim(1)
                .squeeze_dim(1)
        };

        // Prune parameters that are NaN/inf. ONE definition of "non-finite
        // splat" lives in `non_finite_splat_masks`, shared by this prune, the
        // out-of-refine sweep (`prune_non_finite_splats`) and the loss guard's
        // diagnostic — three callers that must agree on what they are counting.
        let non_finite = non_finite_splat_masks(&splats);
        let non_finite_mask = non_finite.any();
        let non_finite_counts = non_finite.counts().await;
        let num_pruned_non_finite = non_finite_counts.any;
        if num_pruned_non_finite > 0 {
            log::info!("{}", non_finite_counts.report(global_iter, "refine"));
        }

        let prune_mask = alpha_mask
            .bool_or(scale_big)
            .bool_or(bound_mask)
            .bool_or(non_finite_mask);

        // Optional min-scale degenerate prune (MRNF port, delta #3). MRNF culls
        // splats whose smallest log-scale axis drops below log(1e-10) (see
        // mrnf.cpp:668, MRNF_LOG_MIN_SCALE_THRESHOLD). ON by default (LFS
        // `mrnf_defaults` parity); disable with `--min-scale-prune=false` to
        // keep thin "pancake" surface splats (see note above). Tests the RAW log-scales
        // (`log_scales().exp()`), NOT `scales()` which folds in the
        // Mip-Splatting min-scale floor; that floor keeps folded scales above
        // the threshold so the prune would never fire. Testing raw scales
        // reproduces LFS's raw-scale cull (mrnf.cpp:668).
        let prune_mask = if self.config.min_scale_prune {
            let scale_small = splats
                .log_scales()
                .exp()
                .lower_elem(self.config.min_scale_prune_threshold)
                .any_dim(1)
                .squeeze_dim(1);
            prune_mask.bool_or(scale_small)
        } else {
            prune_mask
        };

        // Near-zero-rotation prune (MRNF port). Cull splats whose RAW
        // quaternion has collapsed toward zero (squared norm < 1e-8), a
        // degenerate rotation. Mirrors compute_near_zero_rotation_mask
        // (mrnf.cpp:667; pruning_kernels.cu:64 `mag_sq = q.q < 1e-8`). Uses the
        // raw quaternion (`splats.rotations()` = transforms[.., 3..7]),
        // matching LFS's `rotation_raw()`. ON by default (LFS `mrnf_defaults`
        // parity); disable with `--near-zero-rotation-prune=false`.
        let prune_mask = if self.config.near_zero_rotation_prune {
            let quat_norm_sq = splats.rotations().powi_scalar(2).sum_dim(1).squeeze_dim(1);
            let near_zero_rot = quat_norm_sq.lower_elem(1e-8f32);
            prune_mask.bool_or(near_zero_rot)
        } else {
            prune_mask
        };

        // TIDI-GS floater pruning (opt-in). Every refine window folds this
        // window's visibility + position-gradient into the persistent
        // accumulators; on a cleanup cycle (≈ every `--tidi-prune-every` steps,
        // past the start iter) the isolation-selected floater set is unioned
        // into `prune_mask`. Built here as ONE combined mask before the single
        // `prune_points` call, on the pre-prune tensor snapshot, so there is no
        // double-prune / index invalidation. The four-signal candidate rule is
        // an AND (see `tidi::select_prune_indices`) and is deliberately NOT
        // folded into the OR-based geometric culls above — only its final
        // isolation-selected output joins the union.
        // Enter when EITHER path is enabled. `accumulate_window` folds the
        // photometric (visibility + position-gradient) signals and only runs on
        // the photometric path; the depth path's `float`/`valid` counters are
        // folded per-step in `step_with_refine_weight`, not here. The prune
        // cadence + start-iter gate are shared, and `select_prune_mask` internally
        // runs whichever candidate path(s) `tidi_params` marked active.
        let prune_mask = if self.config.tidi_prune || self.config.tidi_depth_prune {
            let params = tidi_params(&self.config);
            if let Some(tidi) = self.tidi.as_mut() {
                if self.config.tidi_prune {
                    tidi.accumulate_window(
                        refiner.vis_weight.clone(),
                        refiner.refine_weight_norm.clone(),
                        self.config.tidi_grad_ema_beta,
                    );
                }
                if tidi.should_prune(
                    global_iter,
                    self.config.tidi_prune_start_iter,
                    self.config.tidi_prune_every,
                ) {
                    match tidi
                        .select_prune_mask(
                            &params,
                            global_iter,
                            splats.opacities(),
                            splats.means(),
                            splats.sh_coeffs.val(),
                            splats.scales(),
                            &device,
                        )
                        .await
                    {
                        Some(tidi_mask) => prune_mask.bool_or(tidi_mask),
                        None => prune_mask,
                    }
                } else {
                    prune_mask
                }
            } else {
                prune_mask
            }
        } else {
            prune_mask
        };

        // Hard cloud-distance prune (`--cloud-prune`): union any Gaussian whose
        // LIVE centre is farther than `--cloud-prune-dist` from the nearest
        // seed/LiDAR cloud point (a floater in empty space) into the prune mask.
        // View-INDEPENDENT: it gathers the CURRENT means against the STATIC
        // point-only distance grid built once at init (no camera, no z-buffer,
        // no see-through), so a Gaussian that drifted off the surface since the
        // last refine is caught this cycle. Independent of the TIDI paths above;
        // built here on the same pre-prune snapshot so the single `prune_points`
        // call reindexes everything at once. Inert (no lookup) when the flag is
        // off, before the start iter, or when no seed cloud built a grid.
        let prune_mask =
            if self.config.cloud_prune && global_iter >= self.config.cloud_prune_start_iter {
                match &self.cloud_prune_grid {
                    Some(grid) => {
                        // Gather each LIVE mean's distance-to-cloud (out-of-grid /
                        // non-finite already forced to +inf), read it to the host,
                        // and build the far mask the IDENTICAL way `select_prune_mask`
                        // builds its prune output: a host 0/1 float uploaded with
                        // `from_data(.., device)` on `device` (= `splats.device()`,
                        // the same device every other prune mask here is built on)
                        // then `greater_elem(0.5)`. That makes `far` the same Bool
                        // kind as `prune_mask`, so `bool_or` cannot trip a
                        // Bool(U32)/Bool(Native) TypeMismatch — the far mask is
                        // constructed by the exact proven idiom, not a new one. The
                        // `[N]` distance readback is the same order as the readbacks
                        // the TIDI prune already does each cleanup.
                        let dists: Vec<f32> = grid
                            .gather_prune_distances(splats.means())
                            .into_data_async()
                            .await
                            .expect("cloud-prune distance readback")
                            .into_vec()
                            .expect("f32");
                        let cp = self.config.cloud_prune_dist;
                        let n = dists.len();
                        let far_f: Vec<f32> = dists
                            .iter()
                            .map(|&d| if d > cp { 1.0 } else { 0.0 })
                            .collect();
                        let far = Tensor::<1>::from_data(TensorData::new(far_f, [n]), &device)
                            .greater_elem(0.5);
                        prune_mask.bool_or(far)
                    }
                    None => prune_mask,
                }
            } else {
                prune_mask
            };

        let (mut splats, refiner, pruned_count) = prune_points(
            splats,
            &mut optim,
            refiner,
            prune_mask,
            self.dig.as_mut(),
            self.tidi.as_mut(),
        )
        .await;

        // Edge-guidance factor (MRNF port, delta #4), aligned to the post-prune
        // splat order. Multiplies into both the dead-slot replacement and the
        // high-gradient growth sampling weights below so densification is biased
        // toward high-frequency image edges. `None` when edge guidance is off or
        // no views were sampled this window.
        let edge_factor = if self.config.use_edge_map {
            refiner
                .edge_factor_host(self.config.edge_score_weight)
                .await
        } else {
            None
        };

        let mut split_inds = HashSet::new();

        // Replace dead gaussians so the pruned budget is reused -- unless
        // `--stop-replace-iter` has disabled backfill, in which case prune keeps
        // culling (notably the over-stretched splats caught by the max-scale
        // term) but the count is allowed to decay rather than being held at cap
        // by opacity-diluting splits.
        let replace_stopped =
            self.config.stop_replace_iter > 0 && global_iter >= self.config.stop_replace_iter;
        if pruned_count > 0 && !replace_stopped {
            // Replacement weighting. By default opacity × visibility. With
            // `replace_by_gradient > 0`, interpolate toward the gradient-
            // weighted distribution (where error actually lives).
            let vis_f = refiner.vis_mask().float();
            let resampled_weights = splats.opacities() * vis_f.clone();
            let mut resampled_weights = resampled_weights
                .into_data_async()
                .await
                .expect("Failed to get weights")
                .into_vec::<f32>()
                .expect("Failed to read weights");
            // Bias replacement toward edge gaussians (MRNF delta #4).
            if let Some(factor) = &edge_factor {
                for (w, f) in resampled_weights.iter_mut().zip(factor.iter()) {
                    *w *= f;
                }
            }
            let resampled_inds =
                multinomial_sample(&mut self.rng, &resampled_weights, pruned_count);
            split_inds.extend(resampled_inds);
        }

        // Force-split splats that are too big on screen (every refine). Rather
        // than killing them (the old `kill_at_screen_size`), we split them and
        // shrink the children down to `split_at_screen_size` on screen — see
        // `refine_splats`. Capped by the remaining `max_splats` budget.
        let pre_oversized = split_inds.len();
        if self.config.split_at_screen_size > 0.0 {
            let oversized = refiner.above_screen_size(self.config.split_at_screen_size);
            let oversized_inds = oversized.argwhere_async().await;
            if oversized_inds.dims()[0] > 0 {
                let oversized_inds = oversized_inds
                    .squeeze_dim::<1>(1)
                    .into_data_async()
                    .await
                    .expect("Failed to get oversized indices")
                    .into_vec::<i32>()
                    .expect("Failed to read oversized indices");
                let mut budget = self
                    .config
                    .max_splats
                    .saturating_sub(splats.num_splats() + split_inds.len() as u32);
                for ind in oversized_inds {
                    if budget == 0 {
                        break;
                    }
                    if split_inds.insert(ind) {
                        budget -= 1;
                    }
                }
            }
        }
        let num_split_oversized = (split_inds.len() - pre_oversized) as u32;

        let pre_high_grad = split_inds.len();
        if global_iter < self.config.growth_stop_iter {
            // Growth signal selection (MRNF `use_error_map` port). When error-map
            // densification is on, the growth candidate set + sampling weight come
            // from the window-MAX error score `Σ T·α·ê` thresholded at τ_err
            // (LFS `refine_candidates = _refine_weight_max > 0.003 && _vis_count>0`,
            // mrnf.cpp:726) — the error signal REPLACES the gradient as the base
            // signal. Off, this is exactly upstream's gradient-norm gate.
            let (above_threshold, growth_base) = if self.config.error_map_densification {
                (
                    refiner
                        .error_above_threshold(self.config.error_map_growth_threshold)
                        .await,
                    refiner.error_scores_median_normalized().await,
                )
            } else {
                (
                    refiner.above_threshold(self.config.growth_grad_threshold),
                    refiner.refine_weight_norm.clone(),
                )
            };

            let threshold_count = above_threshold
                .clone()
                .int()
                .sum()
                .into_scalar_async::<i32>()
                .await
                .expect("Failed to get threshold") as u32;

            let grow_count =
                (threshold_count as f32 * self.config.growth_select_fraction).round() as u32;

            let sample_high_grad = grow_count.saturating_sub(pruned_count);

            // Saturating — cur_splats can exceed max_splats if the scene
            // was loaded above cap, and the u32 underflow would request
            // ~4B new splats.
            let cur_splats = splats.num_splats() + split_inds.len() as u32;
            let headroom = self.config.max_splats.saturating_sub(cur_splats);
            let grow_count = sample_high_grad.min(headroom);

            // If still growing, sample from indices which are over the threshold.
            if grow_count > 0 {
                // Base sampling weight: gradient-norm, or the error score when
                // error-map densification is on (both masked to the thresholded
                // set). Edge guidance, when ALSO on, is a MULTIPLICATIVE bias
                // layered on top — it only reweights WITHIN the thresholded set,
                // never adds a gaussian the base signal missed (LFS: error is the
                // base growth signal, edge is `factor = score·w + 1.0`).
                let weights = above_threshold.float() * growth_base;
                let mut weights = weights
                    .into_data_async()
                    .await
                    .expect("Failed to get weights")
                    .into_vec::<f32>()
                    .expect("Failed to read weights");
                // Bias growth toward edge gaussians (MRNF delta #4).
                if let Some(factor) = &edge_factor {
                    for (w, f) in weights.iter_mut().zip(factor.iter()) {
                        *w *= f;
                    }
                }
                let growth_inds = multinomial_sample(&mut self.rng, &weights, grow_count);
                split_inds.extend(growth_inds);
            }
        }

        let num_split_high_grad = (split_inds.len() - pre_high_grad) as u32;
        let refine_count = split_inds.len();
        // Per-splat max on-screen extent, used by `refine_splats` to cap the
        // split shrink so oversized splats' children land at `split_at_screen_size`.
        let screen_sizes = refiner.max_screen_size.clone();
        splats = self.refine_splats(
            &device,
            optim,
            splats,
            split_inds,
            screen_sizes,
            global_iter,
            phase_iter,
            phase_total,
        );
        if let Some(dig) = &mut self.dig {
            dig.invalidate_neighbors();
        }

        // Update current bounds based on the splats.
        self.bounds = get_splat_bounds(splats.clone(), self.config.bounds_percentile).await;
        // Recompute the per-splat 3D-filter floor against the new positions/
        // count and attach it. Refine must always leave the floor attached:
        // otherwise the late-training and LOD tails can shrink below it.
        // `splats` is already on the inner backend here, so `means()` is too.
        splats = self.apply_min_scale_floor(splats);

        let splat_count = splats.num_splats();

        (
            splats,
            RefineStats {
                num_added: refine_count as u32,
                num_split_oversized,
                num_split_high_grad,
                num_pruned: pruned_count,
                num_pruned_non_finite,
                total_splats: splat_count,
            },
        )
    }

    fn refine_splats(
        &mut self,
        device: &Device,
        mut optim: SplatOptim,
        mut splats: Splats,
        split_inds: HashSet<i32>,
        screen_sizes: Tensor<1>,
        global_iter: u32,
        phase_iter: u32,
        phase_total: u32,
    ) -> Splats {
        let refine_count = split_inds.len();

        // From this point on we mutate canonical parameters and may change
        // cardinality. The old floor is camera-derived auxiliary state; drop
        // it without folding it into parameters, then recompute it at the end.
        splats.min_scale = None;

        if refine_count > 0 {
            let refine_inds = Tensor::from_data(
                TensorData::new(split_inds.into_iter().collect::<Vec<_>>(), [refine_count]),
                device,
            );

            let cur_transforms = splats.transforms.val().select(0, refine_inds.clone());
            let cur_means = cur_transforms.clone().slice(s![.., 0..3]);
            let cur_rots_raw = cur_transforms.clone().slice(s![.., 3..7]);
            let magnitudes = Tensor::clamp_min(
                Tensor::sum_dim(cur_rots_raw.clone().powi_scalar(2), 1).sqrt(),
                1e-32,
            );
            let cur_rots = cur_rots_raw / magnitudes;
            let cur_log_scale = cur_transforms.slice(s![.., 7..10]);
            let cur_sh_coeffs = splats.sh_coeffs.val().select(0, refine_inds.clone());
            let cur_raw_opac = splats.raw_opacities.val().select(0, refine_inds.clone());

            let cur_scales = cur_log_scale.clone().exp();

            // Long-Axis-Split (LAS) child geometry — MRNF port, delta #2.
            // Mirrors LFS `long_axis_split_gaussians_inplace_kernel`
            // (densification_kernels.cu:669-771): split the single longest scale
            // axis, halving it and pushing the two children apart along that axis
            // by half its world extent (centroid-preserving), shrink the other
            // two axes to 0.85x, and cut opacity to 0.6x. Replaces Brush's
            // covariance 1/√2 split. The append + Adam-moment-zeroing +
            // require_grad re-lift scaffold below is unchanged (see E.1).

            // Post-split child (and parent) opacity: sigmoid(raw) * 0.6, back to
            // logit — LFS `inverse_sigmoid(sigmoid(opacity) * 0.6)`. NOT
            // mass-conserving like the old 1-inv^(1/√2) rule; intentional per
            // MRNF (watch for a small brightness step at refines, E.2).
            let new_opac: Tensor<1> =
                sigmoid(cur_raw_opac.clone()).mul_scalar(self.config.split_opacity_scale);
            let new_raw_opac =
                inv_sigmoid(new_opac.clamp(self.config.min_opacity, 1.0 - self.config.min_opacity));

            // One-hot [refine_count, 3] selecting each splat's longest log-scale
            // axis. exp is monotone so argmax over log-scale == argmax over scale;
            // argmax picks a single index, matching LFS `get_max_value_index`
            // (avoids the double-offset a tie in an `equal` mask could cause).
            let ls_device = cur_log_scale.device();
            let longest_idx: Tensor<2, Int> = cur_log_scale.clone().argmax(1);
            let long_onehot: Tensor<2> = Tensor::zeros([refine_count, 3], &ls_device).scatter(
                1,
                longest_idx,
                Tensor::ones([refine_count, 1], &ls_device),
                IndexingUpdateOp::Add,
            );

            // Longest-axis shrink multiplier. LFS uses a fixed 0.5; we keep that
            // for normally-sized splats but let `split_at_screen_size` shrink the
            // longest axis harder for oversized ones so their children land at (at
            // most) the on-screen cap — Brush's oversize benefit, retained per the
            // design's recommendation for equirect. min(0.5, split_at_screen_size
            // / screen); reduces to LFS's fixed 0.5 when the cap is not binding or
            // is disabled (split_at_screen_size <= 0).
            let m_long: Tensor<2> = if self.config.split_at_screen_size > 0.0 {
                screen_sizes
                    .select(0, refine_inds.clone())
                    .unsqueeze_dim(1)
                    .clamp_min(1e-6)
                    .recip()
                    .mul_scalar(self.config.split_at_screen_size)
                    .clamp_max(self.config.split_long_axis_scale)
            } else {
                Tensor::zeros([refine_count, 1], &ls_device)
                    .add_scalar(self.config.split_long_axis_scale)
            };

            // Offset each child by half the longest axis' world extent, along that
            // axis in world space: the local vector `e_L * (0.5 * exp(scale[L]))`
            // (non-zero only on axis L) rotated by the parent quaternion. Matches
            // LFS `global_offset = R[:,L] * (exp(scale[L]) * 0.5)`. Offset
            // magnitude stays at LFS's fixed 0.5*extent; only the scale shrink
            // follows m_long.
            let offset_local = long_onehot.clone() * cur_scales * self.config.split_long_axis_scale;
            let samples = quaternion_vec_multiply(cur_rots.clone(), offset_local);

            // New log-scales: shrink all three axes to 0.85x, then override the
            // longest axis to `log_scale[L] + ln(m_long)` (== +ln(0.5) in the
            // default case). LFS: new_scale[L] = scale[L]+ln(0.5),
            // new_scale[other] = scale[other]+ln(0.85).
            let base_log_scales = cur_log_scale
                .clone()
                .add_scalar(self.config.split_other_axis_scale.ln());
            let long_log_scales = cur_log_scale.clone() + m_long.log();
            let new_log_scales =
                base_log_scales * (-long_onehot.clone() + 1.0) + long_log_scales * long_onehot;
            // LAS keeps the parent rotation for both children.
            let child_rots = cur_rots;

            // Scatter into transforms: build a [refine_count, 10] update tensor
            // with means offset in cols 0..3 and log_scales difference in cols 7..10
            let refine_inds_10 = refine_inds.clone().unsqueeze_dim(1).repeat_dim(1, 10);
            let scale_difference = new_log_scales.clone() - cur_log_scale;

            splats.transforms = splats.transforms.map(|t| {
                let dev = t.device();
                let mut update = Tensor::zeros([refine_count, 10], &dev);
                // Place -samples in means columns (0..3)
                update = update.slice_assign(s![.., 0..3], -samples.clone());
                // Place scale difference in log_scales columns (7..10)
                update = update.slice_assign(s![.., 7..10], scale_difference.clone());
                t.scatter(0, refine_inds_10.clone(), update, IndexingUpdateOp::Add)
            });
            splats.raw_opacities = splats.raw_opacities.map(|m| {
                let difference = new_raw_opac.clone() - cur_raw_opac.clone();
                m.scatter(0, refine_inds.clone(), difference, IndexingUpdateOp::Add)
            });

            // Child sits at parent_mean + samples (parent moves to
            // parent_mean - samples) — anti-correlated, centroid-preserving.
            // Build new transforms row: means(3) + rotations(4) + log_scales(3)
            let new_transforms =
                Tensor::cat(vec![cur_means + samples, child_rots, new_log_scales], 1);

            // Optimizer state lives on the inner (non-autodiff) device.
            let opt_device = device.clone().inner();
            let refine_inds_opt = refine_inds.clone().to_device(&opt_device);

            // DiG features split alongside the splats — the remap details
            // (copy parents, zero Adam moments) live on `DigTrainState`.
            if let Some(dig) = &mut self.dig {
                dig.split(&refine_inds, &refine_inds_opt, &opt_device);
            }

            // TIDI state splits alongside. Children are APPENDED (matching the
            // `cat` order below) and get a FRESH state (importance ≈ 1, zero
            // visibility / grad-EMA, birth = this global iter) so new detail is
            // never inherited into the candidate pool and is protected by the
            // per-Gaussian warmup — see `TidiState::split`.
            if let Some(tidi) = &mut self.tidi {
                tidi.split(
                    refine_count,
                    &refine_inds_opt,
                    &opt_device,
                    &device.clone().inner(),
                    global_iter,
                );
            }

            // Both halves of a split start with zero Adam moments.
            //
            // Burn's scatter bridge
            // only implements Add, so we add the negated parent value to zero
            // it out instead of using Assign.
            splats = map_splats_and_opt(
                splats,
                &mut optim,
                |x| Tensor::cat(vec![x, new_transforms], 0),
                |x| Tensor::cat(vec![x, cur_sh_coeffs], 0),
                |x| Tensor::cat(vec![x, new_raw_opac], 0),
                |x: Tensor<2>| {
                    let d1 = x.dims()[1];
                    let neg_parent = -x.clone().select(0, refine_inds_opt.clone());
                    let inds: Tensor<2, Int> =
                        refine_inds_opt.clone().unsqueeze_dim(1).repeat_dim(1, d1);
                    let x = x.scatter(0, inds, neg_parent, IndexingUpdateOp::Add);
                    Tensor::cat(vec![x, Tensor::zeros([refine_count, d1], &opt_device)], 0)
                },
                |x: Tensor<3>| {
                    let [_, d1, d2] = x.dims();
                    let neg_parent = -x.clone().select(0, refine_inds_opt.clone());
                    let inds_2: Tensor<2, Int> =
                        refine_inds_opt.clone().unsqueeze_dim(1).repeat_dim(1, d1);
                    let inds: Tensor<3, Int> = inds_2.unsqueeze_dim(2).repeat_dim(2, d2);
                    let x = x.scatter(0, inds, neg_parent, IndexingUpdateOp::Add);
                    Tensor::cat(
                        vec![x, Tensor::zeros([refine_count, d1, d2], &opt_device)],
                        0,
                    )
                },
                |x: Tensor<1>| {
                    let neg_parent = -x.clone().select(0, refine_inds_opt.clone());
                    let x = x.scatter(
                        0,
                        refine_inds_opt.clone(),
                        neg_parent,
                        IndexingUpdateOp::Add,
                    );
                    Tensor::cat(vec![x, Tensor::zeros([refine_count], &opt_device)], 0)
                },
            );
        }

        let train_t = (phase_iter as f32 / phase_total.max(1) as f32).clamp(0.0, 1.0);
        let t_shrink_strength = 1.0 - train_t;
        let minus_opac = self.config.opac_decay * t_shrink_strength;

        // Lower opacity slowly over time.
        splats.raw_opacities = splats.raw_opacities.map(|f| {
            let new_opac = sigmoid(f) - minus_opac;
            inv_sigmoid(new_opac.clamp(1e-12, 1.0 - 1e-12))
        });

        // Shrink scales slowly over time (MRNF port, delta #1). MRNF decays
        // both opacity and scale every refine; Brush only had the opacity half.
        // Mirrors `mrnf_decay_kernel` (mrnf_kernels.cu:127-131):
        //   scale = exp(log_scale) * (1 - scale_decay * t_shrink)
        //   log_scale = log(max(scale, 1e-12))
        // Uses the SAME phase-local `t_shrink_strength` as opacity decay (not
        // MRNF's global iter/iterations) to keep LOD-aware training intact, and
        // is applied to the SAME rows — all live splats after append — so freshly
        // split children get exactly one decay. This writes the RAW log-scales
        // (transforms cols 7..10) BEFORE the caller recomputes the Mip-Splatting
        // min-scale floor (`apply_min_scale_floor`), so the floor and decay do
        // not fight. Off (scale_decay == 0) reproduces upstream behaviour.
        if self.config.scale_decay > 0.0 {
            let scale_factor = 1.0 - self.config.scale_decay * t_shrink_strength;
            splats.transforms = splats.transforms.map(|t| {
                let log_scales = t.clone().slice(s![.., 7..10]);
                let new_log_scales = log_scales
                    .exp()
                    .mul_scalar(scale_factor)
                    .clamp_min(1e-12)
                    .log();
                t.slice_assign(s![.., 7..10], new_log_scales)
            });
        }

        self.optim = Some(optim);
        splats
    }
}

fn map_splats_and_opt(
    mut splats: Splats,
    optim: &mut SplatOptim,
    map_transforms: impl FnOnce(Tensor<2>) -> Tensor<2>,
    map_sh_coeffs: impl FnOnce(Tensor<3>) -> Tensor<3>,
    map_opac: impl FnOnce(Tensor<1>) -> Tensor<1>,

    map_opt_transforms: impl Fn(Tensor<2>) -> Tensor<2>,
    map_opt_sh_coeffs: impl Fn(Tensor<3>) -> Tensor<3>,
    map_opt_opac: impl Fn(Tensor<1>) -> Tensor<1>,
) -> Splats {
    splats.transforms = splats.transforms.map(map_transforms);
    optim.transforms.map_momentum(map_opt_transforms);
    splats.sh_coeffs = splats.sh_coeffs.map(map_sh_coeffs);
    optim.sh_coeffs.map_momentum(map_opt_sh_coeffs);
    splats.raw_opacities = splats.raw_opacities.map(map_opac);
    optim.opacities.map_momentum(map_opt_opac);
    splats
}

/// Snapshot the TIDI thresholds/caps from the training config into the
/// config-free params the pure selection logic consumes.
fn tidi_params(config: &TrainConfig) -> TidiPruneParams {
    TidiPruneParams {
        photometric: config.tidi_prune,
        vis_threshold: config.tidi_vis_threshold,
        opacity_threshold: config.tidi_opacity_threshold,
        importance_threshold: config.tidi_importance_threshold,
        grad_threshold: config.tidi_grad_threshold,
        warmup_steps: config.tidi_warmup_steps as i32,
        guard_sh_quantile: config.tidi_guard_sh_quantile,
        guard_thin_quantile: config.tidi_guard_thin_quantile,
        guard_aniso_quantile: config.tidi_guard_aniso_quantile,
        guard_color_var_quantile: config.tidi_guard_color_var_quantile,
        knn_k: config.tidi_knn_k as usize,
        local_cap_frac: config.tidi_local_cap_frac,
        global_cap_frac: config.tidi_global_cap_frac,
        depth_prune: config.tidi_depth_prune,
        depth_float_frac: config.tidi_depth_float_frac,
        depth_min_valid_views: config.tidi_depth_min_valid_views as f32,
        depth_cap_frac: config.tidi_depth_cap_frac,
    }
}

/// Per-parameter-group "this splat has a non-finite value" masks.
///
/// Split by group rather than pre-`or`ed because WHICH parameter went bad is
/// the only cheap diagnostic available when a run poisons itself, and it is
/// what a future root-cause pass will start from. The combined mask is
/// [`Self::any`].
struct NonFiniteSplatMasks {
    transforms: Tensor<1, Bool>,
    sh: Tensor<1, Bool>,
    opacities: Tensor<1, Bool>,
}

/// Counts behind [`NonFiniteSplatMasks`]. `any` is the number of splats with a
/// non-finite value ANYWHERE, so it is <= the sum of the three groups (one
/// splat can be bad in more than one).
struct NonFiniteSplatCounts {
    transforms: u32,
    sh: u32,
    opacities: u32,
    any: u32,
    total: u32,
}

impl NonFiniteSplatCounts {
    /// One-line report, shared by every site that prunes or detects. Kept
    /// identical across sites so the numbers are greppable across a whole run.
    fn report(&self, iter: u32, site: &str) -> String {
        format!(
            "non-finite splats at iter {iter} ({site}): {} of {}              [transforms {} | sh {} | opacity {}]",
            self.any, self.total, self.transforms, self.sh, self.opacities
        )
    }
}

impl NonFiniteSplatMasks {
    fn any(&self) -> Tensor<1, Bool> {
        self.transforms
            .clone()
            .bool_or(self.sh.clone())
            .bool_or(self.opacities.clone())
    }

    /// Reads all four counts back from the GPU. This SYNCHRONISES — callers are
    /// responsible for only doing it on a cadence that already syncs, or on a
    /// path that is already failing.
    async fn counts(&self) -> NonFiniteSplatCounts {
        async fn count(mask: Tensor<1, Bool>) -> u32 {
            mask.int()
                .sum()
                .into_scalar_async::<i32>()
                .await
                .expect("Failed to count non-finite splats") as u32
        }
        let total = self.transforms.dims()[0] as u32;
        NonFiniteSplatCounts {
            transforms: count(self.transforms.clone()).await,
            sh: count(self.sh.clone()).await,
            opacities: count(self.opacities.clone()).await,
            any: count(self.any()).await,
            total,
        }
    }
}

/// The single definition of "this splat is non-finite", used by the refine
/// prune, the out-of-refine sweep and the loss guard's diagnostic.
fn non_finite_splat_masks(splats: &Splats) -> NonFiniteSplatMasks {
    fn row_non_finite(t: &Tensor<2>) -> Tensor<1, Bool> {
        t.clone().is_finite().bool_not().any_dim(1).squeeze_dim(1)
    }
    NonFiniteSplatMasks {
        transforms: row_non_finite(&splats.transforms.val()),
        sh: row_non_finite(&splats.sh_coeffs.val().flatten(1, 2)),
        opacities: row_non_finite(&splats.raw_opacities.val().unsqueeze_dim(1)),
    }
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
async fn prune_points(
    mut splats: Splats,
    optim: &mut SplatOptim,
    mut refiner: RefineRecord,
    prune: Tensor<1, Bool>,
    dig: Option<&mut DigTrainState>,
    tidi: Option<&mut TidiState>,
) -> (Splats, RefineRecord, u32) {
    assert_eq!(
        prune.dims()[0] as u32,
        splats.num_splats(),
        "Prune mask must have same number of elements as splats"
    );

    let prune_count = prune.dims()[0];
    if prune_count == 0 {
        return (splats, refiner, 0);
    }

    let valid_inds = prune.bool_not().argwhere_async().await;

    if valid_inds.dims()[0] == 0 {
        log::warn!("Trying to create empty splat!");
        return (splats, refiner, 0);
    }

    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0] as u32;
    if new_points < start_splats {
        let valid_inds = valid_inds.squeeze_dim(1);
        // Splat params + optimizer state share the autodiff device, but the
        // refiner runs on the inner device — give `keep()` an inner copy.
        use brush_render::burn_glue::detach_autodiff_int;
        let inner_valid_inds = detach_autodiff_int(valid_inds.clone().inner());
        if let Some(floor) = splats.min_scale.take() {
            splats.min_scale = Some(floor.select(0, inner_valid_inds.clone()));
        }
        splats = map_splats_and_opt(
            splats,
            optim,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
        );
        if let Some(dig) = dig {
            dig.keep(&valid_inds);
        }
        // Reindex the TIDI accumulators + `ω` in lockstep. `ω`/its Adam state
        // ride the autodiff `valid_inds`; the inner accumulators use the inner
        // copy — same split the DiG `keep` / `RefineRecord::keep` above use.
        if let Some(tidi) = tidi {
            tidi.keep(&valid_inds, &inner_valid_inds);
        }
        refiner = refiner.keep(inner_valid_inds);
    }
    (splats, refiner, start_splats - new_points)
}

/// Sample a background color: base + uniform noise in [-strength, +strength], clamped to [0, 1].
fn sample_background_color<R: rand::Rng + ?Sized>(
    base: glam::Vec3,
    strength: f32,
    rng: &mut R,
) -> glam::Vec3 {
    if strength <= 0.0 {
        return base.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
    }
    use rand::RngExt as _;
    let noise = glam::Vec3::new(
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
    );
    (base + noise).clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}

/// Warn exactly once that `--depth-normal-weight` is being skipped because the
/// camera is not a pinhole. Once, because it would otherwise fire every step of
/// every view.
fn warn_depth_normal_needs_pinhole() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "--depth-normal-weight is set but this camera is not Pinhole; the \
             depth/normal consistency term is skipped for non-pinhole views \
             (unprojection for fisheye models is not implemented yet)."
        );
    });
}

/// Warn exactly once that a plane depth source was requested on a non-pinhole
/// camera. Both the ray-plane grid and `normals_from_depth` assume a pinhole
/// unprojection, so the step falls back to the centre depth channel rather than
/// supervising with wrong math. Once, because it would otherwise fire every step
/// of every view.
fn warn_plane_depth_needs_pinhole() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "--depth-source selects PGSR plane depth but this camera is not \
             Pinhole; falling back to the centre-expected depth channel for \
             non-pinhole views (ray-plane intersection for fisheye models is not \
             implemented)."
        );
    });
}

/// Warn exactly once that `--depth-opacity-reg-weight` is set but no distance-to-
/// cloud grid was built (the run has no seed point cloud — e.g. a random-init run
/// with no COLMAP/LiDAR points). The regularizer then no-ops for the run; it never
/// panics. The grid is a dataset-level property, so this fires at most once.
fn warn_depth_opacity_reg_no_cloud() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "--depth-opacity-reg-weight is set but no seed point cloud was available \
             to build the distance-to-cloud grid; the depth-coupled opacity \
             regularizer is inert for this run. Seed the run from a point cloud \
             (COLMAP points3D / LiDAR ply) to enable it."
        );
    });
}

/// ---- PGSR plane-depth validity thresholds (approach A call site) ----
///
/// Deliberately NOT config-exposed in v1 (plan §4.1): they are PGSR-paper-typical
/// and the sweep, if one is needed, belongs to the ablation rather than the
/// operator surface.
///
/// A pixel needs real coverage before its composited plane means anything. 0.5
/// is the same coverage threshold `depth_normal_loss` already uses, so the depth
/// and consistency terms agree about which pixels exist.
const PLANE_MIN_ALPHA: f32 = 0.5;
/// `|n_sum · ray|` below this is a plane seen edge-on: the intersection runs off
/// to infinity and its gradient with it. Reject rather than clamp.
///
/// **The test is against the ALPHA-WEIGHTED composited sum, not the geometric
/// `n̂ · ray`.** `n_sum` is `Σ wᵢ nᵢ`, so at coverage α the quantity compared here
/// is `α · (n̂ · ray)` and the effective GEOMETRIC cutoff is `min_denom / α` —
/// tighter as coverage falls (at α = 0.7, ≈ 0.071). This is the one place alpha
/// does not cancel: it divides out of the quotient `offset_sum / (n_sum · ray)`
/// exactly, but not out of this validity comparison. Defensible — a thinly
/// covered pixel is a worse plane estimate as well as a more grazing one, and
/// `PLANE_MIN_ALPHA` only bounds α from below, it does not make α equal 1 — but
/// it is a coupling of two thresholds and it is not obvious from the call, so it
/// is written down. Anyone building a test fixture at a chosen denominator must
/// tilt the geometric normal to `denom/α`; `plane_depth_grazing_tests` does.
///
/// **0.05, where the reference uses 1e-4 — a choice, not a transcription slip.**
/// `gauss-surf` computes in float64, where the quotient is still accurate to
/// ~2e-7 at 1.01x a 1e-4 cutoff. We are f32 end to end, and near-grazing is
/// ill-conditioned: `n_sum · ray` cancels from O(1) down to `O(min_denom)`, so its
/// ~`eps·‖ray‖` absolute rounding becomes a `eps·‖ray‖/|denom|` RELATIVE error in
/// the depth. At 1e-4 that is ~2e-3 relative — a plane pixel accurate to two
/// digits, fed straight into the depth loss. At 0.05 it is ~4e-6, which is the
/// accuracy `plane_depth_grazing_tests` measures and pins. The 500x gap is the
/// price of f32; do not "restore parity" with the reference by lowering it.
const PLANE_MIN_DENOM: f32 = 0.05;
/// `min_depth > 0` is what rejects a plane BEHIND the camera — an `|z| < max`
/// test alone would accept `z = -3`.
const PLANE_MIN_DEPTH: f32 = 1e-3;
/// Far cut for whatever survives `PLANE_MIN_DENOM`. Generous on purpose: `SfM`
/// scenes are not metric, so this cannot be a physical distance. With
/// `min_denom = 0.05` the intersection is already bounded by `20·|offset|`, so
/// this only catches the residue.
const PLANE_MAX_DEPTH: f32 = 1e4;
/// Interval (in global iterations) for the centre-vs-plane residual readback.
/// Sampled rather than per-step because it forces a GPU sync; zero work at all
/// when the depth source is `center`.
const PLANE_RESIDUAL_LOG_EVERY: u32 = 500;

/// `normalize(features / α)` for a `[H, W, 3]` composited normal image —
/// the plane path's copy of the `center` path's alpha-normalize-then-unit-norm
/// sequence, with the same two clamps (`1e-10` on the alpha divide, `1e-6` on
/// the length).
///
/// Deliberately a separate function rather than a shared one the `center` path
/// also calls: the `center` sequence is pinned byte-identical, and hoisting it
/// into a shared helper is exactly the kind of "harmless" refactor that would
/// have to be re-proven. `center_normalize_matches_plane_helper` pins that the
/// two agree.
fn normal_alpha_normalize(features: Tensor<3>, alpha: Tensor<3>) -> Tensor<3> {
    let n = features / alpha.clamp_min(1e-10);
    let len = n.clone().powi_scalar(2).sum_dim(2).sqrt().clamp_min(1e-6);
    n / len
}

/// Log the mean relative disagreement between PGSR plane depth and the
/// centre-expected depth channel, over the pixels where the plane is valid.
///
/// v1 keeps the centre depth channel rendered even when a plane source is active
/// (plan §4.2), so this residual is free apart from the readback. It is a real
/// signal, not noise: WS-1 measured centre depth ~2% biased against plane depth
/// on a tilted slab (mean 1.9e-2, max 2.3e-2 relative — ≈10 cm at 5 m). A
/// residual that sits at ~0 means the plane path is not actually engaging (e.g.
/// nearly everything invalid), which no loss curve would show.
async fn log_plane_vs_centre_residual(
    plane: Tensor<2>,
    centre: Tensor<2>,
    valid: Tensor<2>,
    global_iter: u32,
) {
    let count = valid.clone().sum();
    // `centre` can be 0 on uncovered pixels; clamp before the divide so the
    // masked-out entries cannot produce a non-finite that the mask then
    // multiplies to NaN (the 0·∞ lesson).
    let rel = ((plane - centre.clone()).abs() / centre.abs().clamp_min(1e-6)) * valid;
    let stats: Vec<f32> = Tensor::cat(vec![rel.sum().reshape([1]), count.reshape([1])], 0)
        .into_data_async()
        .await
        .expect("plane residual readback")
        .into_vec()
        .expect("f32 plane residual stats");
    let (sum, n) = (stats[0], stats[1]);
    if n > 0.0 {
        log::info!(
            "iter {global_iter}: plane-vs-centre depth residual {:.4} relative over {n} valid px",
            sum / n
        );
    } else {
        log::warn!(
            "iter {global_iter}: PGSR plane depth is valid at ZERO pixels — the depth \
             and consistency terms are unsupervised this step. Check --depth-source, \
             the camera model, and scene scale against PLANE_MIN_DEPTH/PLANE_MAX_DEPTH."
        );
    }
}

/// Per-splat world-space surface normal: the gaussian's thinnest local axis,
/// rotated into world space and oriented toward the camera. `[N, 10]` ->
/// `[N, 3]` unit vectors.
///
/// Two deliberately DETACHED discrete choices, both standard in the
/// DN-Splatter / `PlanarGS` family:
/// - which axis is thinnest (`argmin` over the log-scales), so the normal
///   does not try to differentiate a permutation;
/// - the camera-facing sign flip, so the loss cannot "fix" a wrong normal by
///   toggling the sign instead of rotating the gaussian.
///
/// What remains live is the quaternion, so a normal loss rotates gaussians.
/// The scales are read but not differentiated through.
fn splat_normals(transforms: Tensor<2>, cam_pos: glam::Vec3) -> Tensor<2> {
    let n = transforms.dims()[0];
    let device = transforms.device();

    let means = transforms.clone().slice(s![.., 0..3]);
    let quats = transforms.clone().slice(s![.., 3..7]);
    let log_scales = transforms.slice(s![.., 7..10]);

    // One-hot over the thinnest axis. `exp` is monotone so argmin over the log
    // scales is argmin over the scales; `argmin` picks a single index, which
    // avoids the double-count an `equal`-mask tie would cause.
    let min_idx: Tensor<2, Int> = log_scales.detach().argmin(1);
    let axis: Tensor<2> = Tensor::zeros([n, 3], &device).scatter(
        1,
        min_idx,
        Tensor::ones([n, 1], &device),
        IndexingUpdateOp::Add,
    );

    // Rotating the local axis by the (normalized) quaternion is exactly the
    // corresponding column of the rotation matrix, computed differentiably.
    let q_len = quats
        .clone()
        .powi_scalar(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12);
    let unit_quats = quats / q_len;
    let normal = quaternion_vec_multiply(unit_quats, axis);
    let n_len = normal
        .clone()
        .powi_scalar(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12);
    let normal = normal / n_len;

    // Face the camera: we want `n · (mean - cam) < 0`. `sign()` would emit 0 on
    // an exactly perpendicular splat and annihilate its normal, so build the
    // ±1 selector from a comparison instead.
    let to_splat = (means
        - Tensor::<1>::from_floats([cam_pos.x, cam_pos.y, cam_pos.z], &device).reshape([1, 3]))
    .detach();
    let facing = (to_splat * normal.clone().detach()).sum_dim(1);
    let sign = facing
        .lower_elem(0.0)
        .float()
        .mul_scalar(2.0)
        .sub_scalar(1.0);

    normal * sign
}

/// The world→camera rotation as a `[3, 3]` tensor laid out so that a `[.., 3]`
/// stack of ROW vectors can be rotated with a single `matmul`.
///
/// `glam`'s `Mat3::x_axis` is the first COLUMN, so reading the axes out in this
/// order builds `Rᵀ` in row-major; right-multiplying row vectors by `Rᵀ` is the
/// same as left-multiplying column vectors by `R`. Same idiom (and same layout)
/// as the normal-render block in `step()`.
pub fn world_to_cam_rot_t(cam: &Camera, device: &Device) -> Tensor<2> {
    let rot = cam.world_to_local().matrix3;
    Tensor::<1>::from_floats(
        [
            rot.x_axis.x,
            rot.x_axis.y,
            rot.x_axis.z,
            rot.y_axis.x,
            rot.y_axis.y,
            rot.y_axis.z,
            rot.z_axis.x,
            rot.z_axis.y,
            rot.z_axis.z,
        ],
        device,
    )
    .reshape([3, 3])
}

/// Per-splat camera-frame tangent-PLANE parameters, `[N, 10]` -> `[N, 4]`:
/// channels `0..3` are the camera-frame unit normal `n_cam`, channel `3` is the
/// signed plane offset `d`, defined so every point `p` on the splat's tangent
/// plane satisfies `n_cam · p = d` in the `OpenCV` camera frame.
///
/// PGSR (Chen et al. 2024, arXiv:2406.06521), plane parameterization. Rendering
/// these four channels through the feature rasterizer and intersecting each
/// camera ray with the composited plane
/// ([`brush_loss::plane_depth_from_features`]) gives PGSR's unbiased surface
/// depth, in place of the alpha-composited camera-`z` of the splat CENTRES that
/// `project_visible.rs:86` emits.
///
/// # Gradient contract (this is the whole point of the function)
///
/// The normal is `splat_normals` verbatim — including its two deliberately
/// DETACHED discrete choices (thinnest-axis `argmin`, camera-facing sign) — then
/// rotated into the camera frame, which is a constant orthonormal map. So
/// channels `0..3` carry exactly today's normal-render gradient: live into the
/// quaternions, nothing into the means or the scales.
///
/// Channel 3 is where this DIFFERS from `splat_normals`, and the difference is
/// load-bearing. `splat_normals` uses the splat position only to pick a facing
/// sign and therefore detaches it outright. Here
///
/// ```text
/// d = n_world · (mean − cam_pos)
/// ```
///
/// and the MEAN ENTERS DIFFERENTIABLY. That is the gradient path from plane
/// depth back to gaussian positions, and it is the only one the feature pass can
/// express (the feature rasterizer treats geometry as constant and back-props
/// into feature VALUES only). Detaching the mean here would leave a function
/// that still renders a perfectly correct plane-depth map and still trains —
/// just with no position supervision at all from it, i.e. silently inert rather
/// than visibly broken. `plane_features_offset_moves_means` pins it.
///
/// The camera-facing sign stays detached (it flips both `n` and `d` together, so
/// the plane is unchanged and only the parameterization's sign convention moves).
///
/// # Why world frame for the offset
///
/// `d = n_cam · mean_cam` by definition, but a rotation preserves dot products
/// and `mean_cam = R(mean − cam_pos)`, `n_cam = R·n_world`, so
/// `n_cam · mean_cam = n_world · (mean − cam_pos)`. Computing it in world frame
/// uses tensors already in hand and needs no `matmul` on the means.
///
/// `transforms` is expected to be the min-scale-FOLDED transforms
/// (`fold_min_scale`), matching what the current normal render feeds
/// `splat_normals`.
pub fn plane_features(transforms: Tensor<2>, cam: &Camera) -> Tensor<2> {
    let device = transforms.device();

    let means = transforms.clone().slice(s![.., 0..3]);
    let n_world = splat_normals(transforms, cam.position);

    // Offset: LIVE in both `n_world` and `means`. See the gradient contract.
    let cam_pos =
        Tensor::<1>::from_floats([cam.position.x, cam.position.y, cam.position.z], &device)
            .reshape([1, 3]);
    let offset = (n_world.clone() * (means - cam_pos)).sum_dim(1);

    let n_cam = n_world.matmul(world_to_cam_rot_t(cam, &device));

    Tensor::cat(vec![n_cam, offset], 1)
}

#[cfg(test)]
mod seeded_rng_tests {
    use super::*;

    #[test]
    fn seeded_background_noise_is_repeatable() {
        let mut first = rand::rngs::StdRng::seed_from_u64(123);
        let mut second = rand::rngs::StdRng::seed_from_u64(123);
        let base = glam::Vec3::splat(0.5);

        assert_eq!(
            sample_background_color(base, 0.25, &mut first),
            sample_background_color(base, 0.25, &mut second)
        );
    }
}

#[cfg(test)]
mod depth_loss_grad_tests {
    use super::*;
    use brush_render::gaussian_splats::SplatRenderMode;
    use brush_render::kernels::camera_model::CameraModel;

    /// A depth-only loss must move gaussian positions and leave opacity untouched.
    ///
    /// The rendered depth is the alpha-normalized expected depth
    /// `accum(ch4) / alpha(ch3)`. Two routes let depth error reach opacity. The
    /// alpha denominator is differentiable, and the depth-loss term detaches it.
    /// The rasterize backward also folds the depth-channel gradient into the
    /// alpha gradient, and `rasterize_backwards.rs` drops that term. With both
    /// routes closed the depth blending weights are detached, so depth
    /// supervision moves the per-splat depth values only. This matches the
    /// `detach_depth_weights` behavior; see the kernel comment for the citation.
    /// The test renders the differentiable depth path, applies a depth-only
    /// loss, and asserts the raw opacities get no gradient while the positions do.
    #[tokio::test]
    async fn depth_loss_does_not_touch_opacity() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

        // A handful of near-opaque gaussians spread in depth in front of a
        // camera that looks down +z.
        let means = vec![
            0.0, 0.0, 0.0, //
            0.3, 0.0, 0.5, //
            -0.3, 0.2, 1.0, //
            0.1, -0.2, 1.5, //
        ];
        let n = means.len() / 3;
        let rotations: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
        let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.0, -1.0, -1.0]).collect();
        let sh: Vec<f32> = (0..n).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        // High raw opacity so the depth channel carries real weight.
        let opac: Vec<f32> = vec![4.0; n];

        let splats = Splats::from_raw(
            means,
            rotations,
            log_scales,
            sh,
            opac,
            SplatRenderMode::Default,
            &device,
        );

        let camera = Camera::new(
            glam::vec3(0.0, 0.0, -5.0),
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        );
        let img_size = glam::uvec2(48, 48);

        let out = render_splats_for_training(
            splats.clone(),
            &camera,
            img_size,
            glam::Vec3::ZERO,
            false,
            RasterizationMode::RgbaAndDepth,
            false,
        )
        .await;

        let [img_h, img_w, _] = out.img.dims();
        let accumulated_depth = out.img.clone().slice(s![.., .., 4..5]);
        // Same detached denominator as the training depth-loss term.
        let alpha = out.img.clone().slice(s![.., .., 3..4]).detach();
        let expected_depth = (accumulated_depth / alpha.clamp_min(1e-10)).reshape([img_h, img_w]);

        // A positive constant target, so the disparity error and its gradient are
        // nonzero wherever a gaussian was rendered.
        let gt_depth = Tensor::<2>::ones([img_h, img_w], &device) * 3.0;
        let loss = depth_loss(expected_depth, gt_depth, None);

        let grads = splats.bwd_validate(loss).await;

        // Positions live in transforms columns 0..3 and must receive gradient.
        let transforms_grad = splats
            .transforms
            .grad(&grads)
            .expect("depth loss must reach gaussian positions");
        let means_grad_absmax = transforms_grad
            .slice(s![.., 0..3])
            .abs()
            .max()
            .into_data_async()
            .await
            .expect("means grad readback")
            .to_vec::<f32>()
            .expect("f32 means grad")[0];
        assert!(
            means_grad_absmax > 1e-8,
            "expected a nonzero position gradient, got {means_grad_absmax}"
        );

        // Opacity must receive no gradient. burn either prunes the leaf (None) or
        // returns an all-zero gradient. A nonzero one means depth error can still
        // push opacity, which is the regression this test guards.
        if let Some(opac_grad) = splats.raw_opacities.grad(&grads) {
            let opac_grad_absmax = opac_grad
                .abs()
                .max()
                .into_data_async()
                .await
                .expect("opacity grad readback")
                .to_vec::<f32>()
                .expect("f32 opacity grad")[0];
            assert!(
                opac_grad_absmax < 1e-8,
                "depth loss must not push opacity, got {opac_grad_absmax}"
            );
        }
    }
}

#[cfg(test)]
mod normal_prior_grad_tests {
    use super::*;
    use brush_render::gaussian_splats::SplatRenderMode;
    use brush_render::kernels::camera_model::CameraModel;

    const IMG: glam::UVec2 = glam::uvec2(48, 48);

    /// A slab of overlapping gaussians filling the middle of the frame, all
    /// tilted by the same rotation about +Y so their surface normal disagrees
    /// with the (flat) rendered depth. Thinnest axis is local +Z.
    fn tilted_plane_splats(device: &Device, tilt: f32) -> Splats {
        let mut means = vec![];
        let n_side = 7;
        for iy in 0..n_side {
            for ix in 0..n_side {
                let f = |i: i32| (i as f32 / (n_side - 1) as f32) * 2.0 - 1.0;
                means.extend_from_slice(&[f(ix), f(iy), 0.0]);
            }
        }
        let n = means.len() / 3;
        let q = glam::Quat::from_rotation_y(tilt);
        let rotations: Vec<f32> = (0..n).flat_map(|_| [q.w, q.x, q.y, q.z]).collect();
        // Thinnest axis is z, so `splat_normals` picks the local +Z column.
        let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.6, -1.6, -2.5]).collect();
        let sh: Vec<f32> = (0..n).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        let opac: Vec<f32> = vec![4.0; n];

        Splats::from_raw(
            means,
            rotations,
            log_scales,
            sh,
            opac,
            SplatRenderMode::Default,
            device,
        )
    }

    fn test_camera() -> Camera {
        Camera::new(
            glam::vec3(0.0, 0.0, -5.0),
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    async fn absmax(t: Tensor<2>) -> f32 {
        t.abs()
            .max()
            .into_data_async()
            .await
            .expect("grad readback")
            .to_vec::<f32>()
            .expect("f32 grad")[0]
    }

    async fn opacity_absmax(splats: &Splats, grads: &Gradients) -> f32 {
        match splats.raw_opacities.grad(grads) {
            None => 0.0,
            Some(g) => g
                .abs()
                .max()
                .into_data_async()
                .await
                .expect("opacity grad readback")
                .to_vec::<f32>()
                .expect("f32 opacity grad")[0],
        }
    }

    /// Render the per-gaussian normal image the training loop builds, in the
    /// camera frame. Mirrors the `use_normal_render` block of `step()`.
    async fn render_camera_normals(splats: &Splats, camera: &Camera) -> Tensor<3> {
        let device = splats.device();
        let transforms = splats.transforms.val();
        let normals = splat_normals(transforms.clone(), camera.position);
        let img = render_splat_features(
            transforms,
            splats.raw_opacities.val(),
            normals,
            camera,
            IMG,
            SplatRenderMode::Default,
        )
        .await;

        let alpha = img.clone().slice(s![.., .., 3..4]).detach();
        let n_world = img.slice(s![.., .., 0..3]) / alpha.clamp_min(1e-10);
        let n_len = n_world
            .clone()
            .powi_scalar(2)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-6);
        let n_world = n_world / n_len;

        let rot = camera.world_to_local().matrix3;
        let r_t: Tensor<2> = Tensor::<1>::from_floats(
            [
                rot.x_axis.x,
                rot.x_axis.y,
                rot.x_axis.z,
                rot.y_axis.x,
                rot.y_axis.y,
                rot.y_axis.z,
                rot.z_axis.x,
                rot.z_axis.y,
                rot.z_axis.z,
            ],
            &device,
        )
        .reshape([3, 3]);

        let [h, w, _] = n_world.dims();
        n_world
            .reshape([(h * w) as i32, 3])
            .matmul(r_t)
            .reshape([h, w, 3])
    }

    /// The prior-normal loss must rotate gaussians and nothing else.
    ///
    /// `render_splat_features` detaches geometry internally and back-props into
    /// the feature VALUES, and `splat_normals` detaches both discrete choices
    /// (thinnest axis, camera-facing sign). So the only live path from this loss
    /// back into the model is the quaternion, transforms columns 3..7 — not
    /// means, not scales, not opacity. That is the contract this test guards.
    /// **The over-masking warning, observed firing on real render output.**
    ///
    /// A diagnostic that has never been seen to fire is not yet a diagnostic, so
    /// this drives the whole §4.7 path end to end: render normals, build a prior
    /// from them, feed the real `normal_gate_counts`, and hand the counts to the
    /// real `record_normal_gate_sample`.
    ///
    /// The contradicted prior here is a SIGN FLIP, chosen only because it is the
    /// cleanest way to manufacture total disagreement — the counts it produces
    /// (0 survivors of a nonzero valid count) are the same shape a
    /// miscalibrated threshold produces, which is the case that actually
    /// motivates the guard. `ingest/splatcam/normals_moge.py`'s check would
    /// likely have flagged this particular prior upstream; it could not have
    /// flagged the miscalibrated one, because it never compares the prior
    /// against the renderer at all (see `NORMAL_GATE_LOW_FRACTION`).
    #[tokio::test]
    #[allow(clippy::field_reassign_with_default)]
    async fn normal_gate_warning_fires_on_a_contradicted_prior() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = tilted_plane_splats(&device, 0.0);
        let camera = test_camera();

        let n_cam = render_camera_normals(&splats, &camera).await;
        let rendered: Vec<f32> = n_cam
            .clone()
            .into_data_async()
            .await
            .expect("normal readback")
            .to_vec()
            .expect("f32 normals");

        // Build the prior FROM the render: exact agreement on covered pixels,
        // `(0,0,0)` (no prior) where nothing rendered. That isolates the gate —
        // any masking observed below is the gate's doing, not missing coverage.
        let mut agreeing = vec![0.0f32; rendered.len()];
        let mut covered = 0usize;
        for (dst, src) in agreeing.chunks_exact_mut(3).zip(rendered.chunks_exact(3)) {
            let len = (src[0] * src[0] + src[1] * src[1] + src[2] * src[2]).sqrt();
            if len > 0.5 {
                dst.copy_from_slice(src);
                covered += 1;
            }
        }
        assert!(
            covered > 100,
            "test scene must actually render normals, covered = {covered}"
        );

        let dims = [IMG.y as usize, IMG.x as usize, 3];
        let cos30 = 30.0_f32.to_radians().cos();

        let gt_ok = Tensor::<3>::from_data(TensorData::new(agreeing.clone(), dims), &device);
        let ok: Vec<f32> = brush_loss::normal_gate_counts(n_cam.clone(), gt_ok, cos30)
            .inner()
            .into_data_async()
            .await
            .expect("counts readback")
            .to_vec()
            .expect("f32 counts");
        assert!((ok[1] - covered as f32).abs() < 1e-3, "valid = {}", ok[1]);
        assert!(
            (ok[0] - ok[1]).abs() < 1e-3,
            "an exactly-agreeing prior must survive the gate entirely: {} of {}",
            ok[0],
            ok[1]
        );

        // Now the sign-flipped prior: same valid pixels, all contradicted.
        let flipped: Vec<f32> = agreeing.iter().map(|v| -v).collect();
        let gt_bad = Tensor::<3>::from_data(TensorData::new(flipped, dims), &device);
        let bad: Vec<f32> = brush_loss::normal_gate_counts(n_cam, gt_bad, cos30)
            .inner()
            .into_data_async()
            .await
            .expect("counts readback")
            .to_vec()
            .expect("f32 counts");
        assert!((bad[1] - covered as f32).abs() < 1e-3, "valid = {}", bad[1]);
        assert_eq!(bad[0], 0.0, "a sign-flipped prior must survive nothing");

        // Feed both through the real bookkeeping. The healthy counts never warn;
        // the contradicted ones warn once the run is sustained.
        let mut cfg = TrainConfig::default();
        cfg.normal_gate_degrees = 30.0;
        let mut trainer = SplatTrainer::new(
            &cfg,
            &Default::default(),
            BoundingBox::from_min_max(glam::Vec3::ZERO, glam::Vec3::ONE),
        );
        for i in 0..5 {
            assert!(
                !trainer.record_normal_gate_sample(i, ok[0], ok[1]),
                "a healthy prior must never warn"
            );
        }
        let mut fired = None;
        for i in 0..NORMAL_GATE_LOW_SAMPLES_TO_WARN {
            if trainer.record_normal_gate_sample(100 + i, bad[0], bad[1]) {
                fired = Some(i + 1);
                break;
            }
        }
        assert_eq!(
            fired,
            Some(NORMAL_GATE_LOW_SAMPLES_TO_WARN),
            "the over-masking warning must fire on the {NORMAL_GATE_LOW_SAMPLES_TO_WARN}th \
             consecutive contradicted sample"
        );
    }

    #[tokio::test]
    async fn normal_loss_moves_rotations_only() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = tilted_plane_splats(&device, 0.5);
        let camera = test_camera();

        let n_cam = render_camera_normals(&splats, &camera).await;

        // Fronto-parallel prior everywhere: disagrees with the tilted splats, so
        // the error and its gradient are nonzero.
        let mut gt = vec![0.0f32; (IMG.y * IMG.x) as usize * 3];
        for px in gt.chunks_exact_mut(3) {
            px[2] = -1.0;
        }
        let gt = Tensor::<3>::from_data(
            TensorData::new(gt, [IMG.y as usize, IMG.x as usize, 3]),
            &device,
        );

        let loss = normal_loss(n_cam, gt, None);
        let loss_val = loss
            .clone()
            .into_data_async()
            .await
            .expect("loss readback")
            .to_vec::<f32>()
            .expect("f32 loss")[0];
        assert!(
            loss_val > 1e-6 && loss_val.is_finite(),
            "expected a real normal loss, got {loss_val}"
        );

        let grads = splats.bwd_validate(loss).await;
        let transforms_grad = splats
            .transforms
            .grad(&grads)
            .expect("normal loss must reach the transforms");

        let rot_grad = absmax(transforms_grad.clone().slice(s![.., 3..7])).await;
        assert!(
            rot_grad > 1e-8,
            "expected a nonzero rotation gradient, got {rot_grad}"
        );

        let mean_grad = absmax(transforms_grad.clone().slice(s![.., 0..3])).await;
        assert!(
            mean_grad < 1e-8,
            "prior-normal loss must not move means, got {mean_grad}"
        );

        let scale_grad = absmax(transforms_grad.slice(s![.., 7..10])).await;
        assert!(
            scale_grad < 1e-8,
            "prior-normal loss must not move scales, got {scale_grad}"
        );

        let opac_grad = opacity_absmax(&splats, &grads).await;
        assert!(
            opac_grad < 1e-8,
            "prior-normal loss must not push opacity, got {opac_grad}"
        );
    }

    /// The flatten term is a pressure on scales alone.
    #[tokio::test]
    async fn flatten_loss_touches_scales_only() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = tilted_plane_splats(&device, 0.0);

        let scales = splats.transforms.val().slice(s![.., 7..10]).exp();
        let loss = scales.min_dim(1).mean();

        let grads = splats.bwd_validate(loss).await;
        let transforms_grad = splats
            .transforms
            .grad(&grads)
            .expect("flatten loss must reach the transforms");

        let scale_grad = absmax(transforms_grad.clone().slice(s![.., 7..10])).await;
        assert!(
            scale_grad > 1e-8,
            "expected a nonzero scale gradient, got {scale_grad}"
        );

        let other_grad = absmax(transforms_grad.slice(s![.., 0..7])).await;
        assert!(
            other_grad < 1e-8,
            "flatten loss must not move means or rotations, got {other_grad}"
        );

        let opac_grad = opacity_absmax(&splats, &grads).await;
        assert!(
            opac_grad < 1e-8,
            "flatten loss must not push opacity, got {opac_grad}"
        );
    }

    /// Depth/normal consistency: finite loss on a real render, and a live
    /// gradient path back into the rotations.
    #[tokio::test]
    async fn depth_normal_consistency_has_grad() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = tilted_plane_splats(&device, 0.5);
        let camera = test_camera();

        let out = render_splats_for_training(
            splats.clone(),
            &camera,
            IMG,
            glam::Vec3::ZERO,
            false,
            RasterizationMode::RgbaAndDepth,
            false,
        )
        .await;

        let [h, w, _] = out.img.dims();
        let alpha = out.img.clone().slice(s![.., .., 3..4]).detach();
        let expected_depth = (out.img.clone().slice(s![.., .., 4..5])
            / alpha.clone().clamp_min(1e-10))
        .reshape([h, w]);

        let focal = camera.focal(IMG);
        let center = camera.center(IMG);
        let n_from_depth = normals_from_depth(expected_depth, focal.x, focal.y, center.x, center.y);

        let n_cam = render_camera_normals(&splats, &camera).await;
        let loss = depth_normal_loss(n_from_depth, n_cam, alpha);

        let loss_val = loss
            .clone()
            .into_data_async()
            .await
            .expect("loss readback")
            .to_vec::<f32>()
            .expect("f32 loss")[0];
        assert!(
            loss_val.is_finite() && loss_val > 1e-6,
            "expected a real consistency loss on a tilted plane, got {loss_val}"
        );

        let grads = splats.bwd_validate(loss).await;
        let transforms_grad = splats
            .transforms
            .grad(&grads)
            .expect("consistency loss must reach the transforms");
        let rot_grad = absmax(transforms_grad.slice(s![.., 3..7])).await;
        assert!(
            rot_grad > 1e-8,
            "expected a nonzero rotation gradient, got {rot_grad}"
        );
    }

    /// A fronto-parallel slab must report near-zero disagreement: this is the
    /// sign/orientation check for the whole chain (splat normal -> rendered
    /// feature -> camera frame -> depth-derived normal).
    #[tokio::test]
    async fn flat_slab_agrees_with_its_own_depth() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = tilted_plane_splats(&device, 0.0);
        let camera = test_camera();

        let out = render_splats_for_training(
            splats.clone(),
            &camera,
            IMG,
            glam::Vec3::ZERO,
            false,
            RasterizationMode::RgbaAndDepth,
            false,
        )
        .await;
        let [h, w, _] = out.img.dims();
        let alpha = out.img.clone().slice(s![.., .., 3..4]).detach();
        let expected_depth = (out.img.clone().slice(s![.., .., 4..5])
            / alpha.clone().clamp_min(1e-10))
        .reshape([h, w]);

        let focal = camera.focal(IMG);
        let center = camera.center(IMG);
        let n_from_depth = normals_from_depth(expected_depth, focal.x, focal.y, center.x, center.y);
        let n_cam = render_camera_normals(&splats, &camera).await;

        let loss = depth_normal_loss(n_from_depth, n_cam, alpha)
            .into_data_async()
            .await
            .expect("loss readback")
            .to_vec::<f32>()
            .expect("f32 loss")[0];
        assert!(
            loss < 0.05,
            "a flat slab must agree with its own depth, got {loss}"
        );
    }
}

/// Regression guard for the inner/autodiff bridge at the end of the native-MSL
/// sparse SH Adam step (`step_sh_coeffs`'s deferred branch).
///
/// `step_sparse_sh` is fed `detach_autodiff(splats.sh_coeffs.val())` and never
/// lifts it, so the parameter it returns is ALREADY on the inner (non-autodiff)
/// backend. Re-wrapping that with `Tensor::from_inner(param.inner())` calls
/// `.inner()` on an already-inner Dispatch tensor, which panics
/// "Requires autodiff tensor." (burn-dispatch `backend.rs:584`) on the very
/// first sparse step. The correct bridge is `lift_to_autodiff(param)`, which
/// accepts either kind and sets `checkpointing` explicitly.
///
/// This module is compiled only where that branch exists, and the test drives
/// the real production sequence (dense warm-up step, then a deferred render) so
/// it cannot pass by silently falling through to the dense path — it asserts
/// that the deferred payload really was produced and that the dense SH gradient
/// really was withheld.
#[cfg(test)]
#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
mod sparse_sh_adam_autodiff_bridge_tests {
    use super::*;
    use brush_render::gaussian_splats::SplatRenderMode;

    const IMG: glam::UVec2 = glam::uvec2(48, 48);
    const LR: f64 = 1e-3;

    fn test_camera() -> Camera {
        Camera::new(
            glam::vec3(0.0, 0.0, -5.0),
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    /// A handful of near-opaque gaussians in front of the camera, carrying four
    /// SH coefficients per channel (degree 1) so the sparse kernel runs one of
    /// its real `coeffs` cases rather than the degenerate single-coefficient one.
    fn test_splats(device: &Device) -> Splats {
        let means = vec![
            0.0, 0.0, 0.0, //
            0.3, 0.0, 0.5, //
            -0.3, 0.2, 1.0, //
            0.1, -0.2, 1.5, //
        ];
        let n = means.len() / 3;
        let rotations: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
        let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.0, -1.0, -1.0]).collect();
        // 4 coefficients x 3 channels per splat = SH degree 1.
        let sh: Vec<f32> = (0..n)
            .flat_map(|i| {
                let base = 0.4 + i as f32 * 0.05;
                [
                    base, base, base, // dc
                    0.02, -0.01, 0.03, //
                    -0.02, 0.015, 0.01, //
                    0.01, 0.02, -0.015, //
                ]
            })
            .collect();
        let opac: Vec<f32> = vec![4.0; n];

        Splats::from_raw(
            means,
            rotations,
            log_scales,
            sh,
            opac,
            SplatRenderMode::Default,
            device,
        )
    }

    /// The optimizer the trainer builds for these splats: SH gets a per-degree
    /// LR scaling of shape `[1, coeffs, 1]` and `reduce_moment_2`, which is what
    /// `sparse_sh_compatible` requires.
    fn test_optim(splats: &Splats, device: &Device) -> SplatOptim {
        let num_coeffs = splats.sh_coeffs.val().dims()[1] as i32;
        let scales: Vec<f32> = (0..num_coeffs)
            .map(|c| if c == 0 { 1.0 } else { 0.05 })
            .collect();
        let sh_lr_scales = Tensor::<1>::from_floats(scales.as_slice(), &device.clone().inner())
            .reshape([1, num_coeffs, 1]);
        SplatOptim {
            adam: AdamScaled::new(1e-15),
            transforms: AdamState::new(None, false),
            sh_coeffs: AdamState::new(Some(sh_lr_scales), true),
            opacities: AdamState::new(None, false),
        }
    }

    /// Render + backward once. Returns the gradients and, when `defer` is set,
    /// the sparse SH payload the trainer would hand to `step_sh_coeffs`.
    async fn render_backward(
        splats: &Splats,
        defer: bool,
    ) -> (Gradients, Option<DeferredShGrad>, bool) {
        let out = render_splats_for_training(
            splats.clone(),
            &test_camera(),
            IMG,
            glam::Vec3::ZERO,
            false,
            RasterizationMode::Rgba,
            defer,
        )
        .await;
        let handle = out.deferred_sh_grad;
        let handle_present = handle.is_some();
        // Any loss with a real gradient on every channel.
        let loss = out
            .img
            .clone()
            .slice(s![.., .., 0..3])
            .powf_scalar(2.0)
            .sum();
        let mut grads = splats.bwd_validate(loss).await;
        let deferred = handle.map(|h| {
            h.take(&mut grads)
                .expect("deferred SH gradient holder was not populated")
        });
        (grads, deferred, handle_present)
    }

    #[tokio::test]
    async fn sparse_sh_step_returns_a_differentiable_parameter() {
        let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let mut splats = test_splats(&device);
        let mut optim = test_optim(&splats, &device);

        // The sparse kernel needs plane ops / a fixed plane size. The module is
        // already gated to native-MSL macOS arm64, so an unsupported device here
        // is a real problem, not a reason to quietly skip: a test that never
        // enters the sparse branch would be worse than no test at all.
        {
            use brush_render::burn_glue::detach_autodiff;
            let param = detach_autodiff(splats.sh_coeffs.val());
            assert!(
                crate::sh_adam::sparse_sh_adam_supported(&param),
                "this device cannot run the sparse SH Adam kernel, so the \
                 regression under test is not being exercised"
            );
        }

        // --- Step 1: dense warm-up ---------------------------------------
        // Production never defers on the first step: `can_defer_sh_grad` is
        // false until Adam's moments exist. Take that step here so the sparse
        // preconditions become true the same way they do in a real run.
        let (mut grads, deferred, handle_present) = render_backward(&splats, false).await;
        assert!(
            !handle_present,
            "a non-deferred render must not produce a sparse SH payload"
        );
        assert!(
            splats.sh_coeffs.grad(&grads).is_some(),
            "the dense warm-up step needs a dense SH gradient"
        );
        splats = step_sh_coeffs(&mut optim, splats, &mut grads, deferred, LR);

        // --- Step 2: the deferred/sparse step ----------------------------
        {
            use brush_render::burn_glue::detach_autodiff;
            let param = detach_autodiff(splats.sh_coeffs.val());
            assert!(
                AdamScaled::sparse_sh_compatible(&param, &optim.sh_coeffs),
                "the warm-up step should have populated Adam's SH moments"
            );
        }

        let (mut grads, deferred, handle_present) = render_backward(&splats, true).await;
        assert!(
            handle_present,
            "the deferred render must hand back a sparse SH payload"
        );
        let deferred = deferred.expect("deferred SH payload");
        // Contract of the deferred path: backward withholds the dense SH
        // gradient, so `step_sh_coeffs` MUST consume the sparse payload. If this
        // ever became `Some`, the test could pass through the dense branch
        // without touching the code under test.
        assert!(
            splats.sh_coeffs.grad(&grads).is_none(),
            "a deferred render must not also populate the dense SH gradient"
        );

        // The regression itself: this call panicked with
        // "Requires autodiff tensor." while the bridge was
        // `Tensor::from_inner(param.inner())`.
        splats = step_sh_coeffs(&mut optim, splats, &mut grads, Some(deferred), LR);

        // And the stepped parameter must be a real autodiff leaf again, not an
        // inner tensor smuggled back into the module — otherwise the next
        // backward would silently stop producing SH gradients.
        assert!(
            splats.sh_coeffs.val().is_require_grad(),
            "the stepped SH parameter must still require grad"
        );
        let (grads, _, _) = render_backward(&splats, false).await;
        assert!(
            splats.sh_coeffs.grad(&grads).is_some(),
            "the stepped SH parameter must still receive gradients"
        );
    }
}

/// WS-L config-surface tests: the scene-scale helper
/// (`--normalize-metric-weights`) and the contradiction-gate diagnostic's
/// decision logic. CPU-only — pure `glam` arithmetic on camera poses and pure
/// bookkeeping; no device and no tensors.
///
/// **Merge note.** This module is appended at the end of `train.rs`, as is
/// WS-1's `plane_feature_tests`. Git's line-level heuristic interleaves the two
/// into invalid Rust, so integration treats each `#[cfg(test)] mod` as ONE
/// opaque block. Keep this module self-contained (it takes nothing from outside
/// `super::*`) and do not append unrelated items after it.
#[cfg(test)]
mod scene_scale_tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;

    /// Build an upright `OpenCV` camera at `pos` looking at `target`, with the
    /// given world up direction.
    ///
    /// `OpenCV` camera frame: `+X` right, `+Y` DOWN, `+Z` forward. So the c2w
    /// columns are `[right, -up, forward]` and `mean_camera_up` must negate
    /// column 1 to recover `up`.
    fn cam_at(pos: glam::Vec3, target: glam::Vec3, up: glam::Vec3) -> Camera {
        let forward = (target - pos).normalize();
        let down = -up.normalize();
        let right = down.cross(forward).normalize();
        // Re-orthogonalize so a non-perpendicular (pos, target, up) triple still
        // yields a proper rotation.
        let down = forward.cross(right).normalize();
        let rotation =
            glam::Quat::from_mat3(&glam::Mat3::from_cols(right, down, forward)).normalize();
        Camera::new(
            pos,
            rotation,
            0.8,
            0.8,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    /// A ring of `n` cameras of radius `r`, centred on `center`, orbiting about
    /// `up` and all looking inward at `center`.
    fn camera_ring(n: usize, r: f32, center: glam::Vec3, up: glam::Vec3) -> Vec<Camera> {
        let up = up.normalize();
        // Two orthonormal in-plane axes.
        let a = if up.x.abs() < 0.9 {
            up.cross(glam::Vec3::X).normalize()
        } else {
            up.cross(glam::Vec3::Y).normalize()
        };
        let b = up.cross(a).normalize();
        (0..n)
            .map(|k| {
                let theta = std::f32::consts::TAU * k as f32 / n as f32;
                let pos = center + (a * theta.cos() + b * theta.sin()) * r;
                cam_at(pos, center, up)
            })
            .collect()
    }

    /// The `OpenCV` column choice, pinned on its own. This is the single line the
    /// port's convention hangs on: the reference reads c2w column 1 directly
    /// (OpenGL, `+Y` up), we must negate it (`OpenCV`, `+Y` down).
    #[test]
    fn mean_camera_up_negates_the_opencv_down_column() {
        for up in [
            glam::Vec3::Z,
            glam::Vec3::Y,
            -glam::Vec3::X,
            glam::vec3(0.0, 1.0, 1.0).normalize(),
        ] {
            let cams = camera_ring(6, 2.0, glam::vec3(1.0, -2.0, 3.0), up);
            let got = mean_camera_up(&cams);
            assert!(
                (got - up).length() < 1e-5,
                "mean_camera_up = {got:?}, want {up:?} \
                 (a result of {:?} would mean the OpenCV sign flip was missed)",
                -up
            );
        }
    }

    /// Empty input, and a back-to-back pair whose up axes cancel: no panic, no
    /// NaN, deterministic fallback.
    #[test]
    fn mean_camera_up_degenerate_cases() {
        assert_eq!(mean_camera_up(&[]), glam::Vec3::Z);
        let a = cam_at(glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Z);
        let b = cam_at(glam::Vec3::ZERO, glam::Vec3::X, -glam::Vec3::Z);
        assert_eq!(mean_camera_up(&[a, b]), glam::Vec3::Z);
    }

    /// `rotation_up_to_z` really lands `up` on `+Z`, including both degenerate
    /// (anti)parallel cases.
    #[test]
    fn rotation_up_to_z_lands_on_z() {
        for up in [
            glam::Vec3::Z,
            -glam::Vec3::Z,
            glam::Vec3::Y,
            glam::Vec3::X,
            glam::vec3(0.3, -0.5, 0.8).normalize(),
            glam::vec3(1.0, 1.0, -1.0).normalize(),
        ] {
            let r = rotation_up_to_z(up);
            let landed = r * up;
            assert!(
                (landed - glam::Vec3::Z).length() < 1e-5,
                "up {up:?} landed on {landed:?}"
            );
            // Proper rotation: determinant +1.
            assert!((r.determinant() - 1.0).abs() < 1e-5);
        }
    }

    /// The whole pipeline on a synthetic ring whose answer is known by
    /// construction.
    ///
    /// A ring of 8 cameras of radius `R` about `up`, translated anywhere: the
    /// mean origin is the ring centre, so centring puts the ring at the origin;
    /// the mean up is `up`, so the Rodrigues step lays the ring flat in the
    /// world XY plane; and with 8 evenly spaced cameras two of them land exactly
    /// on an in-plane axis, so the largest absolute coordinate is exactly `R`.
    #[test]
    fn scene_scale_from_camera_ring() {
        // Z-up ring, offset far from the origin: centring must remove the offset
        // entirely, so the answer is the radius, not the distance to the origin.
        let cams = camera_ring(8, 2.0, glam::vec3(0.0, 0.0, 5.0), glam::Vec3::Z);
        let scale = scene_scale_from_cameras(&cams).expect("ring has a scale");
        assert!((scale - 2.0).abs() < 1e-4, "z-up ring scale = {scale}");

        // Y-up ring (the COLMAP/SuperSplat-style frame): identical answer, but
        // now the Rodrigues step does real work rotating +Y onto +Z.
        let cams = camera_ring(8, 3.0, glam::vec3(-4.0, 9.0, 2.0), glam::Vec3::Y);
        let scale = scene_scale_from_cameras(&cams).expect("ring has a scale");
        assert!((scale - 3.0).abs() < 1e-4, "y-up ring scale = {scale}");

        // A tilted up axis is still just a ring: the reorientation flattens it.
        let up = glam::vec3(0.0, 1.0, 1.0).normalize();
        let cams = camera_ring(8, 1.5, glam::vec3(2.0, 2.0, 2.0), up);
        let scale = scene_scale_from_cameras(&cams).expect("ring has a scale");
        assert!((scale - 1.5).abs() < 1e-4, "tilted ring scale = {scale}");

        // The scale is a RADIUS, not a diameter, and it scales linearly.
        let cams = camera_ring(8, 20.0, glam::Vec3::ZERO, glam::Vec3::Z);
        let scale = scene_scale_from_cameras(&cams).expect("ring has a scale");
        assert!((scale - 20.0).abs() < 1e-3, "scale = {scale}");

        // Empty input yields None (callers fall back to 1.0, i.e. unnormalized).
        assert_eq!(scene_scale_from_cameras(&[]), None);

        // A single camera is its own mean, so every centred origin is zero and
        // there is no usable scale.
        let one = camera_ring(1, 2.0, glam::Vec3::ZERO, glam::Vec3::Z);
        assert_eq!(scene_scale_from_cameras(&one), None);
    }

    /// The gate diagnostic must be completely inert at the default.
    ///
    /// With `--normal-gate-degrees` unset there is no gate to observe, so the
    /// trainer must never take a sample — which is what keeps the readback (a
    /// real device sync) out of every default run.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn normal_gate_diagnostic_is_inert_by_default() {
        let device = Default::default();
        let bounds = BoundingBox::from_min_max(glam::Vec3::ZERO, glam::Vec3::ONE);

        let def = SplatTrainer::new(&TrainConfig::default(), &device, bounds);
        for i in [0u32, 1, 99, 100, 200, 5_000, 15_000, 30_000] {
            assert!(
                !def.should_sample_normal_gate(i),
                "default config sampled the gate at iter {i}"
            );
        }

        // Gate on: sampling happens, on the refine-derived stride only.
        let mut on = TrainConfig::default();
        on.normal_gate_degrees = 30.0;
        let stride = on.refine_every / NORMAL_GATE_SAMPLES_PER_WINDOW;
        assert!(stride > 1, "test assumes a nontrivial stride, got {stride}");
        let trainer = SplatTrainer::new(&on, &device, bounds);
        assert!(trainer.should_sample_normal_gate(0));
        assert!(trainer.should_sample_normal_gate(stride));
        assert!(trainer.should_sample_normal_gate(stride * 3));
        assert!(!trainer.should_sample_normal_gate(stride + 1));
        assert!(!trainer.should_sample_normal_gate(stride - 1));

        // Gate armed later: no sampling before its start iter, even on-stride.
        let mut late = on;
        late.normal_gate_start_iter = stride * 4;
        let trainer = SplatTrainer::new(&late, &device, bounds);
        assert!(!trainer.should_sample_normal_gate(stride * 2));
        assert!(trainer.should_sample_normal_gate(stride * 4));
    }

    /// The sustained-low warning: fires only after a RUN of low samples, resets
    /// on a healthy one, and ignores frames that carried no prior at all.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn normal_gate_warns_only_when_sustained() {
        let device = Default::default();
        let bounds = BoundingBox::from_min_max(glam::Vec3::ZERO, glam::Vec3::ONE);
        let mut cfg = TrainConfig::default();
        cfg.normal_gate_degrees = 30.0;
        let mut t = SplatTrainer::new(&cfg, &device, bounds);

        // A healthy fraction never warns, however often it is sampled.
        for i in 0..10 {
            assert!(!t.record_normal_gate_sample(i, 900.0, 1000.0));
        }
        assert_eq!(t.normal_gate_low_samples, 0);

        // Low samples accumulate; the warning fires on the Nth, not the first.
        assert!(!t.record_normal_gate_sample(100, 50.0, 1000.0));
        assert!(!t.record_normal_gate_sample(200, 50.0, 1000.0));
        assert!(
            t.record_normal_gate_sample(300, 50.0, 1000.0),
            "warning must fire after {NORMAL_GATE_LOW_SAMPLES_TO_WARN} consecutive low samples"
        );
        // It keeps firing while the condition persists — it is a sustained state.
        assert!(t.record_normal_gate_sample(400, 50.0, 1000.0));

        // One healthy sample resets the run.
        assert!(!t.record_normal_gate_sample(500, 900.0, 1000.0));
        assert_eq!(t.normal_gate_low_samples, 0);
        assert!(!t.record_normal_gate_sample(600, 50.0, 1000.0));

        // Exactly at the threshold is NOT low (the comparison is strict).
        let mut edge = SplatTrainer::new(&cfg, &device, bounds);
        for i in 0..5 {
            assert!(!edge.record_normal_gate_sample(i, NORMAL_GATE_LOW_FRACTION * 1000.0, 1000.0));
        }

        // A frame with no usable prior says nothing about the gate: it must not
        // count toward the run, and must not reset it either.
        let mut empty = SplatTrainer::new(&cfg, &device, bounds);
        assert!(!empty.record_normal_gate_sample(0, 50.0, 1000.0));
        assert!(!empty.record_normal_gate_sample(1, 0.0, 0.0));
        assert_eq!(
            empty.normal_gate_low_samples, 1,
            "an empty-prior frame must neither advance nor reset the run"
        );
        assert!(!empty.record_normal_gate_sample(2, 50.0, 1000.0));
        assert!(empty.record_normal_gate_sample(3, 50.0, 1000.0));

        // Non-finite counts are ignored rather than propagated into the run.
        let mut nan = SplatTrainer::new(&cfg, &device, bounds);
        assert!(!nan.record_normal_gate_sample(0, f32::NAN, 1000.0));
        assert!(!nan.record_normal_gate_sample(1, 50.0, f32::NAN));
        assert_eq!(nan.normal_gate_low_samples, 0);
    }

    /// `metric_weight_scale()` is an exact 1.0 unless the flag is on AND a scale
    /// was captured — the default-inertness guarantee for L3, at the consumption
    /// site rather than in config.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn metric_weight_scale_is_exactly_one_by_default() {
        let device = Default::default();
        let bounds = BoundingBox::from_min_max(glam::Vec3::ZERO, glam::Vec3::ONE);
        let cams = camera_ring(8, 2.0, glam::Vec3::ZERO, glam::Vec3::Z);

        // Flag off: the setter is a no-op and the divisor is an exact identity.
        let mut off = SplatTrainer::new(&TrainConfig::default(), &device, bounds);
        off.set_init_scene_scale(&cams);
        assert_eq!(off.init_scene_scale, None);
        assert_eq!(off.metric_weight_scale(), 1.0);

        // Flag on: captured once, and later calls with a DIFFERENT camera set
        // (the LOD re-supply) must not move it.
        let mut cfg = TrainConfig::default();
        cfg.normalize_metric_weights = true;
        let mut on = SplatTrainer::new(&cfg, &device, bounds);
        on.set_init_scene_scale(&cams);
        assert!((on.metric_weight_scale() - 2.0).abs() < 1e-4);
        let bigger = camera_ring(8, 50.0, glam::Vec3::ZERO, glam::Vec3::Z);
        on.set_init_scene_scale(&bigger);
        assert!(
            (on.metric_weight_scale() - 2.0).abs() < 1e-4,
            "set_init_scene_scale must be one-shot, got {}",
            on.metric_weight_scale()
        );

        // Flag on but no usable scale: fall back to 1.0 rather than poisoning
        // the loss with a zero or NaN divisor.
        let mut none = SplatTrainer::new(&cfg, &device, bounds);
        none.set_init_scene_scale(&[]);
        assert_eq!(none.metric_weight_scale(), 1.0);
    }

    /// **§10d item 9.** A common world translation cannot change the scene
    /// scale.
    ///
    /// Ported from the reference's `test_scene_scale_is_translation_invariant`,
    /// including its `(17, −11, 5)` offset and its `4 · eps` budget. The
    /// existing `scene_scale_from_camera_ring` centres its rings away from the
    /// origin, which exercises the same code — but it re-derives the expected
    /// answer from the radius, so a centring bug that scaled the offset instead
    /// of removing it could in principle be absorbed by the 1e-4 comparison.
    /// This states the invariance directly: two DIFFERENT inputs, one number.
    ///
    /// The tolerance is a real budget, not a formality. `translation` is a mean
    /// of the origins, so the subtraction `origin − translation` cancels the
    /// offset to within its own rounding: with coordinates ~17 and a result
    /// ~2.7, a few eps of `17.0` is the whole error term.
    #[test]
    fn scene_scale_is_translation_invariant() {
        let offset = glam::vec3(17.0, -11.0, 5.0);

        for (n, r, center, up) in [
            (8usize, 2.0f32, glam::Vec3::ZERO, glam::Vec3::Z),
            (8, 3.0, glam::vec3(-4.0, 9.0, 2.0), glam::Vec3::Y),
            (
                5,
                1.5,
                glam::vec3(2.0, 2.0, 2.0),
                glam::vec3(0.0, 1.0, 1.0).normalize(),
            ),
        ] {
            let here =
                scene_scale_from_cameras(&camera_ring(n, r, center, up)).expect("ring has a scale");
            let there = scene_scale_from_cameras(&camera_ring(n, r, center + offset, up))
                .expect("translated ring has a scale");
            assert!(
                (here - there).abs() <= 4.0 * f32::EPSILON * here.max(1.0),
                "scene scale moved under a pure translation: {here} vs {there}"
            );
        }
    }

    /// **§10d item 9, second half.** The reference's three-pose fixture, whose
    /// answer is exactly `8/3`.
    ///
    /// From `test_training_cameras_keep_metric_centers_and_reproduce_applied_scale`.
    /// Camera centres `(0,0,0)`, `(2,0,0)`, `(0,4,0)` with a common up axis:
    /// the mean is `(2/3, 4/3, 0)`, so the centred origins are `(−2/3, −4/3, 0)`,
    /// `(4/3, −4/3, 0)` and `(−2/3, 8/3, 0)`, and the largest absolute
    /// coordinate of any of them is `8/3`. Reorienting up onto `+Z` permutes
    /// which axis holds it but cannot change the maximum.
    ///
    /// Worth its own test alongside the ring fixtures because a ring is
    /// SYMMETRIC: its answer is the radius under almost any plausible
    /// mis-definition of "scale" (RMS, mean distance, half the extent, the
    /// largest coordinate). This asymmetric triple separates them — an RMS would
    /// give 1.63, a mean distance 1.80, half the bounding extent 2.0.
    ///
    /// The reference's poses are `OpenGL` (c2w column 1 IS up); ours are
    /// `OpenCV` (column 1 is DOWN), so the fixture's rotation is a 180-degree
    /// turn about `+X` — which is what makes our `mean_camera_up` recover `+Y`
    /// from the same geometry. That difference is pinned on its own in
    /// `mean_camera_up_negates_the_opencv_down_column`.
    #[test]
    fn scene_scale_matches_the_reference_three_pose_fixture() {
        let flip_x = glam::Quat::from_rotation_x(std::f32::consts::PI);
        let cams: Vec<Camera> = [
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(2.0, 0.0, 0.0),
            glam::vec3(0.0, 4.0, 0.0),
        ]
        .into_iter()
        .map(|pos| {
            Camera::new(
                pos,
                flip_x,
                0.8,
                0.8,
                glam::vec2(0.5, 0.5),
                CameraModel::Pinhole,
            )
        })
        .collect();

        // Sanity: these poses really are +Y-up under our OpenCV reading.
        let up = mean_camera_up(&cams);
        assert!(
            (up - glam::Vec3::Y).length() < 1e-6,
            "fixture up = {up:?}, want +Y"
        );

        // The reference asserts this at 2 eps ABSOLUTE, in float32, on its own
        // parser path. Ours goes through a Rodrigues rotation of the centred
        // origins, so the budget is stated relative and measured rather than
        // copied: worst observed deviation is 0 eps (the rotation for a +Y up
        // axis is an exact axis permutation), and 8 eps leaves headroom for a
        // tilted-up variant without becoming meaningless — it is still five
        // orders of magnitude tighter than the 1e-4 the ring fixtures use.
        let want = 8.0f32 / 3.0;
        let scale = scene_scale_from_cameras(&cams).expect("three poses have a scale");
        assert!(
            (scale - want).abs() <= 8.0 * f32::EPSILON * want,
            "scene scale = {scale}, want 8/3 = {want} (the reference's pinned value); \
             an RMS would give 1.63, a mean distance 1.80, half the extent 2.0"
        );
    }
}

/// WS-1 pins for the shared PGSR plane math: the gradient contract of
/// [`plane_features`], and an end-to-end check that a real rasterized slab's
/// composited plane features intersect back to the slab.
///
/// The analytic ray-plane unit tests live in brush-loss
/// (`plane_depth_tests`); this module is where `plane_features` and the feature
/// rasterizer are both reachable, so it is the only place the full chain
/// splat quaternion -> plane parameters -> compositing -> ray intersection can
/// be closed. It is the plane-path sibling of
/// `normal_prior_grad_tests::flat_slab_agrees_with_its_own_depth`.
#[cfg(test)]
mod plane_feature_tests {
    use super::*;
    use brush_loss::plane_depth_from_features;
    use brush_render::gaussian_splats::SplatRenderMode;
    use brush_render::kernels::camera_model::CameraModel;

    const IMG: glam::UVec2 = glam::uvec2(48, 48);

    /// Camera at `-5` on the optical axis, identity rotation, so the
    /// world→camera rotation is the identity and every sign in the chain is
    /// readable by hand.
    fn test_camera() -> Camera {
        Camera::new(
            glam::vec3(0.0, 0.0, -5.0),
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    /// A slab of overlapping gaussians whose CENTRES ALL LIE ON one plane
    /// through the world origin, tilted by `tilt` about `+Y`, each gaussian
    /// rotated to match so its thinnest axis is the plane normal.
    ///
    /// The "centres lie on the plane" part is what makes the end-to-end test
    /// exact rather than approximate: every splat then has the SAME plane offset
    /// `d`, so the composited `Σwᵢdᵢ / Σwᵢnᵢ·ray` reduces to `d/(n·ray)` no
    /// matter what per-pixel coverage weights the rasterizer produces. A slab
    /// whose centres merely scatter near the plane would make the expected depth
    /// depend on the weights, which are not knowable outside the kernel.
    ///
    /// This is the difference from `normal_prior_grad_tests::tilted_plane_splats`,
    /// which rotates the gaussians but leaves the centres in the `z = 0` plane —
    /// fine for a normal test, useless for a depth one.
    fn planar_slab(device: &Device, tilt: f32) -> Splats {
        let q = glam::Quat::from_rotation_y(tilt);
        // Thinnest axis is local +Z, so the plane's in-plane spans are the
        // rotated local X and Y axes.
        let e1 = q * glam::vec3(1.0, 0.0, 0.0);
        let e2 = q * glam::vec3(0.0, 1.0, 0.0);

        let mut means = vec![];
        let n_side = 9;
        for iy in 0..n_side {
            for ix in 0..n_side {
                let f = |i: i32| (i as f32 / (n_side - 1) as f32) * 2.0 - 1.0;
                let p = e1 * f(ix) + e2 * f(iy);
                means.extend_from_slice(&[p.x, p.y, p.z]);
            }
        }
        let n = means.len() / 3;
        let rotations: Vec<f32> = (0..n).flat_map(|_| [q.w, q.x, q.y, q.z]).collect();
        let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.6, -1.6, -2.5]).collect();
        let sh: Vec<f32> = (0..n).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        let opac: Vec<f32> = vec![4.0; n];

        Splats::from_raw(
            means,
            rotations,
            log_scales,
            sh,
            opac,
            SplatRenderMode::Default,
            device,
        )
    }

    async fn absmax(t: Tensor<2>) -> f32 {
        t.abs()
            .max()
            .into_data_async()
            .await
            .expect("grad readback")
            .to_vec::<f32>()
            .expect("f32 grad")[0]
    }

    async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .to_vec::<f32>()
            .expect("f32 tensor")
    }

    /// The offset channel is the ONLY path from plane supervision back to
    /// gaussian positions, and its gradient is not merely nonzero — it is
    /// exactly the world normal.
    ///
    /// `d = n_world · (mean − cam_pos)`, so `∂(Σd)/∂mean = n_world` (the
    /// camera-facing sign is detached, so it contributes nothing extra). Pinning
    /// the VALUE, not just "greater than zero", is what makes this test catch the
    /// failure the plan warns about: detaching the mean here leaves a function
    /// that still renders a correct plane-depth map and still trains — it just
    /// silently supervises no positions at all.
    #[tokio::test]
    async fn plane_features_offset_moves_means() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = planar_slab(&device, 0.5);
        let camera = test_camera();

        let feats = plane_features(splats.transforms.val(), &camera);
        assert_eq!(feats.dims(), [splats.num_splats() as usize, 4]);

        let loss = feats.slice(s![.., 3..4]).sum();
        let grads = splats.bwd_validate(loss).await;
        let transforms_grad = splats
            .transforms
            .grad(&grads)
            .expect("the plane offset must reach the transforms");

        let mean_grad = read(transforms_grad.clone().slice(s![.., 0..3])).await;
        let want = read(splat_normals(splats.transforms.val(), camera.position)).await;
        assert_eq!(mean_grad.len(), want.len());
        let worst = mean_grad
            .iter()
            .zip(want.iter())
            .map(|(g, n)| (g - n).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "d(offset)/d(mean) must equal the world normal, worst component error {worst}"
        );
        // Guard against the assertion above being satisfied by an all-zero
        // gradient matching an all-zero normal.
        let magnitude = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            magnitude > 0.5,
            "the fixture's normals must be nonzero, got {magnitude}"
        );

        // The offset also depends on the quaternion through `n_world`, so that
        // path must be live too.
        let rot_grad = absmax(transforms_grad.clone().slice(s![.., 3..7])).await;
        assert!(
            rot_grad > 1e-8,
            "the plane offset must also reach the rotations, got {rot_grad}"
        );

        // The thinnest-axis choice is a detached `argmin`, so scales get nothing
        // — the flatten term stays the scale-side pressure. Same contract as the
        // existing normal loss.
        let scale_grad = absmax(transforms_grad.slice(s![.., 7..10])).await;
        assert!(
            scale_grad < 1e-8,
            "the detached argmin must leave scales alone, got {scale_grad}"
        );

        let opac_grad = match splats.raw_opacities.grad(&grads) {
            None => 0.0,
            Some(g) => absmax(g.unsqueeze_dim(1)).await,
        };
        assert!(
            opac_grad < 1e-8,
            "plane features must not push opacity, got {opac_grad}"
        );
    }

    /// The normal channels carry the same contract `splat_normals` always had:
    /// rotations only. Rotating into the camera frame is a constant orthonormal
    /// map and must not open a new path.
    #[tokio::test]
    async fn plane_features_normal_moves_rotations_only() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = planar_slab(&device, 0.5);
        let camera = test_camera();

        let feats = plane_features(splats.transforms.val(), &camera);
        // A weighted sum, so the three channels cannot cancel each other out.
        let w: Tensor<2> = Tensor::<1>::from_floats([0.3f32, -0.7, 1.1], &device).reshape([1, 3]);
        let loss = (feats.slice(s![.., 0..3]) * w).sum();

        let grads = splats.bwd_validate(loss).await;
        let transforms_grad = splats
            .transforms
            .grad(&grads)
            .expect("the plane normal must reach the transforms");

        let rot_grad = absmax(transforms_grad.clone().slice(s![.., 3..7])).await;
        assert!(
            rot_grad > 1e-8,
            "expected a nonzero rotation gradient, got {rot_grad}"
        );
        let mean_grad = absmax(transforms_grad.clone().slice(s![.., 0..3])).await;
        assert!(
            mean_grad < 1e-8,
            "the plane NORMAL must not move means, got {mean_grad}"
        );
        let scale_grad = absmax(transforms_grad.slice(s![.., 7..10])).await;
        assert!(
            scale_grad < 1e-8,
            "the plane normal must not move scales, got {scale_grad}"
        );
    }

    /// End-to-end: rasterize `plane_features` through the feature pass, feed the
    /// `[H, W, 5]` result to `plane_depth_from_features`, and check the recovered
    /// depth against the slab's closed-form ray-plane depth.
    ///
    /// Run on a fronto-parallel slab AND a slab tilted 0.5 rad about `+Y`. The
    /// tilted arm is the one with teeth: a fronto-parallel plane has
    /// `n_cam = (0, 0, −1)`, so the intersection collapses to `depth = −d` and
    /// the ray grid, the intrinsics and the world→camera rotation all drop out.
    /// Under the tilt the depth varies ~1.5x across the frame and every one of
    /// those is live.
    ///
    /// Closed form for a plane through the world origin with camera-frame unit
    /// normal `n` and offset `d = n · (0 − cam_pos)`:
    /// `z(u, v) = d / (n · ray(u, v))`.
    #[tokio::test]
    async fn plane_depth_matches_a_rendered_slab() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let camera = test_camera();
        let focal = camera.focal(IMG);
        let center = camera.center(IMG);
        let (w, h) = (IMG.x as usize, IMG.y as usize);

        for tilt in [0.0f32, 0.5] {
            let splats = planar_slab(&device, tilt);
            let transforms = splats.transforms.val();

            let feats = plane_features(transforms.clone(), &camera);
            let feat_img = render_splat_features(
                transforms,
                splats.raw_opacities.val(),
                feats,
                &camera,
                IMG,
                SplatRenderMode::Default,
            )
            .await;
            assert_eq!(feat_img.dims(), [h, w, 5]);

            let (depth, normal, valid) = plane_depth_from_features(
                feat_img, focal.x, focal.y, center.x, center.y, 0.5, 0.05, 0.05, 100.0,
            );
            let depth = read(depth).await;
            let normal = read(normal).await;
            let valid = read(valid).await;

            // Expected plane, derived independently of `plane_features`: the
            // thinnest axis is local +Z, rotated by the tilt; `splat_normals`
            // flips it to face the camera; the camera rotation is the identity.
            let n = {
                let v = glam::Quat::from_rotation_y(tilt) * glam::vec3(0.0, 0.0, 1.0);
                // Camera at -5z, plane through the origin: `n · (0 − cam)` is
                // `+5·v.z`, which is positive, so the facing rule flips `v`.
                -v
            };
            let d = n.dot(glam::Vec3::ZERO - camera.position);

            let mut covered = 0usize;
            let mut worst = 0.0f32;
            for py in 0..h {
                for px in 0..w {
                    let i = py * w + px;
                    if valid[i] == 0.0 {
                        continue;
                    }
                    covered += 1;

                    let ray = glam::vec3(
                        (px as f32 + 0.5 - center.x) / focal.x,
                        (py as f32 + 0.5 - center.y) / focal.y,
                        1.0,
                    );
                    let want = d / n.dot(ray);
                    worst = worst.max((depth[i] - want).abs() / want);

                    for c in 0..3 {
                        assert!(
                            (normal[i * 3 + c] - n[c]).abs() < 1e-5,
                            "tilt {tilt}: normal[{c}] at ({px},{py}) = {}, want {}",
                            normal[i * 3 + c],
                            n[c]
                        );
                    }
                }
            }

            // The slab spans ±1 in the plane at depth ~5 with a 0.7 rad FOV, so
            // it covers the middle of the frame, not all of it.
            assert!(
                covered > 400,
                "tilt {tilt}: expected the slab to cover a real region, got {covered} valid pixels"
            );
            assert!(
                worst < 1e-5,
                "tilt {tilt}: worst relative depth error {worst} against the closed-form plane"
            );
        }
    }

    /// **§10d item 2.** `q` and `−q` are the same rotation, so they must give
    /// the same plane features.
    ///
    /// Ported from the reference's
    /// `test_gaussian_plane_features_face_along_camera_ray`
    /// (`gauss-surf`, Apache-2.0, Pablo Vela), which uses this exact fixture —
    /// two gaussians on the optical axis at z = 2 and z = 3, thin along local
    /// +Z, quaternions `+[1,0,0,0]` and `−[1,0,0,0]`.
    ///
    /// **The reference pins `n = (0, 0, +1)` and offsets `(2, 3)`; we produce
    /// `n = (0, 0, −1)` and `(−2, −3)`, and that is correct, not a port bug.**
    /// `splat_normals` turns the normal to FACE the camera (`n·(mean − cam) < 0`
    /// — see its sign block), the reference points it away. Every consumer on
    /// our side agrees with our choice: `normals_from_depth` emits `n_z ≤ 0` by
    /// construction, and the offset follows the normal's sign because
    /// `d = n·(mean − cam)`. The depth `d/(n·ray)` is invariant to the pair
    /// flipping together, which is why the two conventions produce identical
    /// depth maps. Pinned by value here so that a future half-flip — normal
    /// negated without the offset, or the other way round — cannot pass.
    ///
    /// Bit-identity is the right assertion for the sign half, not an
    /// approximation: every entry of a rotation matrix built from a quaternion
    /// is a sum of PRODUCTS OF TWO quaternion components, so negating all four
    /// leaves each product unchanged exactly under IEEE-754, and the
    /// normalisation divides by the same length. There is no reordering and no
    /// atomic accumulation anywhere in this path.
    #[tokio::test]
    async fn plane_features_are_invariant_to_quaternion_sign() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

        // Camera at the origin, identity rotation: the world frame IS the
        // camera frame, so every number below is readable by hand.
        let camera = Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        );

        let means = vec![0.0, 0.0, 2.0, 0.0, 0.0, 3.0];
        // Thinnest axis is local +Z, so the plane normal is the quaternion's
        // third rotation column.
        let log_scales: Vec<f32> = (0..2).flat_map(|_| [1.0, 1.0, -2.0]).collect();
        let sh: Vec<f32> = (0..2).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        let opac = vec![4.0; 2];

        let features_for = |quats: Vec<f32>| {
            let splats = Splats::from_raw(
                means.clone(),
                quats,
                log_scales.clone(),
                sh.clone(),
                opac.clone(),
                SplatRenderMode::Default,
                &device,
            );
            plane_features(splats.transforms.val(), &camera)
        };

        let positive = read(features_for(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])).await;
        let negative = read(features_for(vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0])).await;

        assert_eq!(
            positive, negative,
            "q and -q are the same rotation and must give bit-identical plane features"
        );

        // The values themselves, so this cannot pass by both sides being wrong
        // in the same way.
        assert_eq!(
            positive,
            vec![0.0, 0.0, -1.0, -2.0, 0.0, 0.0, -1.0, -3.0],
            "camera-facing normal (0,0,-1) with offsets (-2,-3); the reference's \
             away-facing convention is the exact negation of this pair"
        );
    }
}

/// WS-A pins for the `--depth-source` consumer wiring: the backward contract of
/// the plane-aux depth path (§4.5 row 2), and the two bit-identity claims the
/// `center` default rests on.
///
/// The plane MATH is pinned in `plane_feature_tests` and in brush-loss'
/// `plane_depth_tests`. What is pinned HERE is the wiring the trainer does
/// around it: which gradients a depth loss on plane depth is allowed to reach,
/// and that selecting `center` still runs the pre-change op sequence.
#[cfg(test)]
mod plane_aux_consumer_tests {
    use super::*;
    use brush_loss::plane_depth_from_features;
    use brush_render::gaussian_splats::SplatRenderMode;
    use brush_render::kernels::camera_model::CameraModel;

    const IMG: glam::UVec2 = glam::uvec2(48, 48);

    /// Last-bit budget, in ULPs, for two expression trees that apply the SAME
    /// operations to the SAME inputs and are only expected to agree to the
    /// arithmetic's own precision.
    ///
    /// Derived, not tuned. Two textually identical sequences are not one
    /// program on a GPU: the shader compiler may contract a multiply-add into an
    /// FMA in one and not the other, and may reassociate the 3-term sum inside
    /// `sum_dim`. Each rounding costs at most half an ULP and there are a
    /// handful of them in the divide/square/sum/sqrt/divide chain, so 4 ULP
    /// bounds the honest disagreement with room to spare. Independently the same
    /// number `brush-bench-test/tests/center_source_identity.rs` reached for the
    /// same reason (`SAME_PATH_MARGIN`).
    ///
    /// **Do not replace this with `assert_eq!`.** That claim was made here once
    /// and measured false — see `center_normalize_matches_plane_helper`.
    const SAME_PATH_ULPS: f32 = 4.0;

    /// A camera that is NOT axis-aligned with the world, so the world→camera
    /// rotation is a real rotation. `plane_feature_tests` deliberately uses an
    /// identity-rotation camera to keep its hand-derived signs readable; here the
    /// rotation is the thing under test, so an identity would pass vacuously.
    fn tilted_camera() -> Camera {
        Camera::new(
            glam::vec3(0.3, -0.4, -5.0),
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.12, -0.21, 0.07),
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    /// A tilted slab of gaussians with WELL-SEPARATED log scales per splat.
    ///
    /// The separation is not cosmetic. `splat_normals` picks the thinnest axis
    /// with a detached `argmin`, so two axes of nearly equal scale put the splat
    /// on a discontinuity: an infinitesimal parameter change flips which axis is
    /// "the" normal and the plane jumps. Every gradient assertion below (and the
    /// finite-difference test in brush-bench-test) requires the scene to sit away
    /// from that boundary.
    fn slab(device: &Device) -> Splats {
        let q = glam::Quat::from_rotation_y(0.5);
        let e1 = q * glam::vec3(1.0, 0.0, 0.0);
        let e2 = q * glam::vec3(0.0, 1.0, 0.0);

        let mut means = vec![];
        let n_side = 7;
        for iy in 0..n_side {
            for ix in 0..n_side {
                let f = |i: i32| (i as f32 / (n_side - 1) as f32) * 2.0 - 1.0;
                let p = e1 * f(ix) + e2 * f(iy);
                means.extend_from_slice(&[p.x, p.y, p.z]);
            }
        }
        let n = means.len() / 3;
        let rotations: Vec<f32> = (0..n).flat_map(|_| [q.w, q.x, q.y, q.z]).collect();
        // -1.2 / -1.6 / -3.0: every pair is > 0.4 apart in log space, so the
        // argmin is unambiguous and stays that way under perturbation.
        let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.2, -1.6, -3.0]).collect();
        let sh: Vec<f32> = (0..n).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        let opac: Vec<f32> = vec![4.0; n];

        Splats::from_raw(
            means,
            rotations,
            log_scales,
            sh,
            opac,
            SplatRenderMode::Default,
            device,
        )
    }

    async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .to_vec::<f32>()
            .expect("f32 tensor")
    }

    async fn absmax<const D: usize>(t: Tensor<D>) -> f32 {
        read(t.abs().max()).await[0]
    }

    /// Render the plane feature pass exactly the way `step()` does, and take the
    /// production depth loss on the result.
    ///
    /// Deliberately the SAME call sequence and the SAME thresholds as the
    /// trainer, including zeroing the GT at invalid pixels — a pin written
    /// against a simplified stand-in would not be pinning the shipped path.
    async fn plane_depth_loss(splats: &Splats, camera: &Camera) -> Tensor<1> {
        let transforms = splats.transforms.val();
        let feats = plane_features(transforms.clone(), camera);
        let feat_img = render_splat_features(
            transforms,
            splats.raw_opacities.val(),
            feats,
            camera,
            IMG,
            SplatRenderMode::Default,
        )
        .await;

        let focal = camera.focal(IMG);
        let center = camera.center(IMG);
        let (depth, _normal, valid) = plane_depth_from_features(
            feat_img,
            focal.x,
            focal.y,
            center.x,
            center.y,
            PLANE_MIN_ALPHA,
            PLANE_MIN_DENOM,
            PLANE_MIN_DEPTH,
            PLANE_MAX_DEPTH,
        );

        // A GT that is 10% nearer than the prediction everywhere, so every valid
        // pixel carries a real, same-signed disparity error. `* valid` is the
        // trainer's own masking of unsupervised pixels.
        let gt = depth.clone().detach().mul_scalar(0.9) * valid;
        depth_loss(depth, gt, None)
    }

    /// §4.5 row 2: with `plane-aux`, depth error must NOT be able to reach
    /// opacity.
    ///
    /// Approach A gets this for free rather than by construction — the feature
    /// rasterizer's backward tracks the feature VALUES only (`features_bwd.rs`
    /// registers a single parent), so the compositing weights are constants and
    /// there is no alpha VJP to leak through. That is exactly the property
    /// approach B (`plane-fused`) deliberately gives up, which is why the two
    /// arms of the ablation are comparable only if this holds here.
    #[tokio::test]
    async fn plane_aux_depth_does_not_touch_opacity() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = slab(&device);
        let camera = tilted_camera();

        let loss = plane_depth_loss(&splats, &camera).await;
        // Guard against a vacuous pass: a zero loss would satisfy every
        // assertion below without exercising anything.
        assert!(
            read(loss.clone()).await[0] > 1e-6,
            "the plane depth loss must be nonzero for this pin to mean anything"
        );
        let grads = splats.bwd_validate(loss).await;

        match splats.raw_opacities.grad(&grads) {
            None => {}
            Some(g) => {
                let worst = absmax(g).await;
                assert_eq!(
                    worst, 0.0,
                    "plane-aux depth error reached opacity (max |grad| {worst}); \
                     that is the plane-FUSED contract, not this one"
                );
            }
        }
    }

    /// §4.5 row 2, the positive half: geometry gradients arrive through the
    /// feature VALUES — means via the plane offset, quaternions via the normal.
    ///
    /// Scales are asserted EXACTLY zero. That is not an oversight to fix later:
    /// `splat_normals` detaches the thinnest-axis `argmin`, so the plane is a
    /// function of the quaternion and of a discrete axis CHOICE, and
    /// differentiating the choice means differentiating a permutation.
    /// `--flatten-loss-weight` is the scale-side pressure in this design.
    #[tokio::test]
    async fn plane_aux_depth_moves_means_and_quats() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = slab(&device);
        let camera = tilted_camera();

        let loss = plane_depth_loss(&splats, &camera).await;
        let grads = splats.bwd_validate(loss).await;
        let g = splats
            .transforms
            .grad(&grads)
            .expect("plane depth must reach the transforms at all");

        let means = absmax(g.clone().slice(s![.., 0..3])).await;
        let quats = absmax(g.clone().slice(s![.., 3..7])).await;
        let scales = absmax(g.slice(s![.., 7..10])).await;

        assert!(
            means > 1e-8,
            "plane depth must move gaussian MEANS (via the offset channel), got {means}"
        );
        assert!(
            quats > 1e-8,
            "plane depth must move gaussian QUATERNIONS (via the normal channels), got {quats}"
        );
        assert_eq!(
            scales, 0.0,
            "the thinnest-axis argmin is detached, so scales must get EXACTLY no \
             gradient from the plane path; got {scales}"
        );
    }

    /// Byte-identity pin 1: the `center` branch now builds its world→camera
    /// rotation with [`world_to_cam_rot_t`] instead of the inline `glam` unroll
    /// it used before. That is the ONLY edit inside the default path, so this
    /// test asserts the two constructions agree BIT for bit — not to a
    /// tolerance, because a tolerance would not pin byte-identity.
    #[tokio::test]
    async fn world_to_cam_rot_t_is_the_inline_construction() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let camera = tilted_camera();

        // Verbatim copy of the pre-change inline construction.
        let rot = camera.world_to_local().matrix3;
        let legacy: Tensor<2> = Tensor::<1>::from_floats(
            [
                rot.x_axis.x,
                rot.x_axis.y,
                rot.x_axis.z,
                rot.y_axis.x,
                rot.y_axis.y,
                rot.y_axis.z,
                rot.z_axis.x,
                rot.z_axis.y,
                rot.z_axis.z,
            ],
            &device,
        )
        .reshape([3, 3]);

        let got = read(world_to_cam_rot_t(&camera, &device)).await;
        let want = read(legacy).await;
        assert_eq!(
            got, want,
            "world_to_cam_rot_t must reproduce the inline construction exactly"
        );
    }

    /// Byte-identity pin 2, and the claim §4.3 makes about the plane path's
    /// normal channels: compositing WORLD normals and rotating the image (the
    /// `center` order) agrees with compositing CAMERA-frame normals directly
    /// (the `plane-aux` order, which `plane_features` enables by rotating
    /// per-splat before the rasterizer sees them).
    ///
    /// `Σwᵢ(R·nᵢ) = R·(Σwᵢnᵢ)` and `normalize(R·v) = R·normalize(v)` for an
    /// orthonormal `R`, so they agree analytically; in f32 they agree to
    /// rounding. That is exactly why `step()` keeps BOTH orders instead of
    /// unifying them: the `center` order is the one the byte-identity gate pins,
    /// and swapping it for the (equally correct) plane order would perturb the
    /// recorded `playroom_0812` baseline in the last bits.
    #[tokio::test]
    async fn plane_normal_channels_match_the_center_normal_render() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = slab(&device);
        let camera = tilted_camera();
        let transforms = splats.transforms.val();
        let opac = splats.raw_opacities.val();
        let (h, w) = (IMG.y as usize, IMG.x as usize);

        // --- `center` order: world normals -> composite -> /a -> unit -> R ---
        let normal_img = render_splat_features(
            transforms.clone(),
            opac.clone(),
            splat_normals(transforms.clone(), camera.position),
            &camera,
            IMG,
            SplatRenderMode::Default,
        )
        .await;
        let a = normal_img.clone().slice(s![.., .., 3..4]).detach();
        let n_world = normal_img.slice(s![.., .., 0..3]) / a.clone().clamp_min(1e-10);
        let n_len = n_world
            .clone()
            .powi_scalar(2)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-6);
        let n_world = n_world / n_len;
        let center_n = (n_world.reshape([(h * w) as i32, 3]))
            .matmul(world_to_cam_rot_t(&camera, &device))
            .reshape([h, w, 3]);

        // --- `plane-aux` order: camera normals -> composite -> /a -> unit ---
        let feat_img = render_splat_features(
            transforms.clone(),
            opac,
            plane_features(transforms, &camera),
            &camera,
            IMG,
            SplatRenderMode::Default,
        )
        .await;
        let plane_a = feat_img.clone().slice(s![.., .., 4..5]).detach();
        let plane_n = normal_alpha_normalize(feat_img.slice(s![.., .., 0..3]), plane_a.clone());

        // Compare only where the pixel is actually covered: the uncovered
        // background is `0/1e-10` in both, i.e. a normalized zero vector whose
        // direction is meaningless and whose agreement would prove nothing.
        let cover = read(plane_a.reshape([h, w])).await;
        let got = read(plane_n).await;
        let want = read(center_n).await;

        let mut covered = 0usize;
        let mut worst = 0.0f32;
        for i in 0..h * w {
            if cover[i] < PLANE_MIN_ALPHA {
                continue;
            }
            covered += 1;
            for c in 0..3 {
                worst = worst.max((got[i * 3 + c] - want[i * 3 + c]).abs());
            }
        }
        assert!(
            covered > 400,
            "expected the slab to cover a real region, got {covered} covered pixels"
        );
        assert!(
            worst < 1e-5,
            "compositing order changed the rendered normal by {worst}; the two \
             orders are the same linear map and must agree to f32 rounding"
        );
    }
    /// The duplication justification on [`normal_alpha_normalize`], made
    /// enforceable.
    ///
    /// The `center` branch of `step()` writes the alpha-normalize-then-unit-norm
    /// sequence INLINE, and the plane branch calls the helper. That is deliberate
    /// duplication — the `center` sequence is what the byte-identity gate pins, so
    /// folding both onto one shared helper is a refactor that would have to be
    /// re-proven rather than assumed. The cost of the duplication is that the two
    /// copies can drift.
    ///
    /// This pins that they have not. The comparison covers EVERY pixel,
    /// background included, rather than filtering to the covered region.
    ///
    /// # Why this is a MARGIN and not `assert_eq!` — corrected 2026-08-20
    ///
    /// It was written as `assert_eq!` on the raw f32s, on the reasoning that the
    /// same ops in the same order must be bit-equal. **That reasoning is wrong
    /// on a GPU, and it was measured wrong here**: on the M4 Max this failed 9
    /// runs out of 9, with **340 of 6912 elements differing by exactly one ULP**
    /// (1.19e-7 at component magnitudes of 0.48-0.88), on both the default
    /// backend and `native-msl`. Two textually identical expression trees are
    /// not one program: the shader compiler is free to contract a multiply-add
    /// into an FMA in one and not the other, and to reassociate, and the
    /// autotuner's kernel choice differs between the two dispatch shapes.
    ///
    /// It is the fifth member of the class commit `08f60b6f` converted for four
    /// other assertions in this port, and this one was missed because — and this
    /// is the part worth recording — **it evidently PASSED on the integrator's
    /// build** (405 tests green, 60/60 stabilization runs). So the divergence is
    /// **build-environment-dependent**, which is worse than flaky: a machine
    /// that happens to contract the same way sees nothing at all, and the test
    /// silently means something different on every box it runs on.
    ///
    /// Hence the margin is derived from FIRST PRINCIPLES, not from this
    /// machine's measurement. The two sequences apply the same four operations
    /// (divide, square, sum-of-3, sqrt, divide) to the same inputs, so each
    /// output can differ by at most the accumulated last-bit error of that
    /// chain: a handful of roundings, each at most half an ULP, plus one
    /// FMA-contraction difference per multiply-add. [`SAME_PATH_ULPS`] = 4 is
    /// that budget rounded up — the same constant `center_source_identity.rs`
    /// arrived at independently for the same reason. It is NOT tuned to the
    /// observed 1 ULP; if it were, this test would only be meaningful here.
    ///
    /// The thing a margin gives up is the ability to see a sub-ULP drift, and
    /// there is no such thing: any REAL divergence between these two sequences —
    /// a reordered normalize, a different clamp target, a wrong slice — moves
    /// components by 1e-2 or more, i.e. five orders of magnitude above this
    /// budget. The mutation record below is the evidence for that claim: the
    /// reordered-normalize mutation lands at 8.4e6 ULP.
    ///
    /// # What this does and does NOT pin — verified by mutation, not assumed
    ///
    /// Checked by deliberately breaking the helper and re-running:
    ///
    /// - Reordering the sequence (normalize before the alpha divide instead of
    ///   after) **fails** the test. That is the drift worth guarding, and it is
    ///   guarded. Re-measured 2026-08-20 under the ULP margin: it still fails,
    ///   at **8.4e6 ULP** (worst element -119.76 vs -0.479, 2162 of 6912
    ///   elements differing) against a 4-ULP budget — six orders of magnitude
    ///   clear, so the margin costs this test nothing.
    /// - Changing a clamp CONSTANT (`1e-10` -> `1e-9`) **passes**, under the
    ///   margin exactly as it did under `assert_eq!` (re-measured 2026-08-20).
    ///   That is not a
    ///   weakness to fix by tightening the test; it is a property of the
    ///   expression. Wherever alpha is ~0 the composited feature numerator is ~0
    ///   too, so the quotient is 0 for any clamp value — the clamps exist to keep
    ///   `0/0` from being `NaN`, not to shape the output. No test on rendered data
    ///   can distinguish their values, so do not read this test as pinning them.
    ///
    /// Stating that explicitly because the comment this test exists to justify
    /// once cited a pin that had never been written; a pin whose reach is
    /// overstated is the same failure one step further along.
    #[tokio::test]
    async fn center_normalize_matches_plane_helper() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let splats = slab(&device);
        let camera = tilted_camera();
        let transforms = splats.transforms.val();
        let (h, w) = (IMG.y as usize, IMG.x as usize);

        // A real composited `[H, W, 4]`: 3 normal channels + alpha, with genuine
        // covered and uncovered regions rather than synthetic values.
        let img = render_splat_features(
            transforms.clone(),
            splats.raw_opacities.val(),
            splat_normals(transforms, camera.position),
            &camera,
            IMG,
            SplatRenderMode::Default,
        )
        .await;
        assert_eq!(img.dims(), [h, w, 4]);

        let alpha = img.clone().slice(s![.., .., 3..4]).detach();

        // --- Verbatim copy of the `center` branch's inline sequence. ---
        let inline = {
            let n = img.clone().slice(s![.., .., 0..3]) / alpha.clone().clamp_min(1e-10);
            let len = n.clone().powi_scalar(2).sum_dim(2).sqrt().clamp_min(1e-6);
            n / len
        };

        // --- The helper the plane branch calls. ---
        let helper = normal_alpha_normalize(img.slice(s![.., .., 0..3]), alpha.clone());

        let got = read(helper).await;
        let want = read(inline).await;
        assert_eq!(got.len(), want.len());

        // Worst deviation, in ULPs at the larger of the two magnitudes. `ulp()`
        // floors at `f32::MIN_POSITIVE` so a pair of exact zeros — most of the
        // background — yields a finite, non-zero epsilon rather than 0/0.
        let ulp = |x: f32| f32::EPSILON * x.abs().max(f32::MIN_POSITIVE);
        let (worst_ulps, worst_at) = got
            .iter()
            .zip(want.iter())
            .enumerate()
            .map(|(i, (a, b))| ((a - b).abs() / ulp(a.abs().max(b.abs())), i))
            .fold((0.0f32, 0usize), |acc, x| if x.0 > acc.0 { x } else { acc });
        let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();

        assert!(
            worst_ulps <= SAME_PATH_ULPS,
            "normal_alpha_normalize has drifted from the center branch's inline \
             sequence: worst deviation {worst_ulps:.2} ULP at element {worst_at} \
             ({} vs {}), over the {SAME_PATH_ULPS}-ULP same-path budget, with \
             {differing} of {} elements differing at all. A REAL drift (a \
             reordered normalize, a different clamp target, a wrong slice) lands \
             ~1e5 ULP out — see the mutation record on this test — so anything \
             in this range is a genuine change to the expression, not FMA \
             contraction.",
            got[worst_at],
            want[worst_at],
            got.len()
        );

        // Guard against a vacuous pass: if the render produced nothing, two
        // all-zero images would compare equal and prove nothing. Require both a
        // real covered region (unit-length normals) and real background.
        let cover = read(alpha.reshape([h, w])).await;
        let covered = cover.iter().filter(|a| **a >= PLANE_MIN_ALPHA).count();
        let background = cover.iter().filter(|a| **a < 1e-6).count();
        assert!(
            covered > 400,
            "expected a real covered region, got {covered} covered pixels"
        );
        assert!(
            background > 100,
            "expected real uncovered background (where the clamps bite), got              {background} pixels"
        );
    }
}

/// WS-F pins for the two NaN-containment guards: the total-loss finiteness
/// check (gap 1) and the out-of-refine non-finite splat sweep (gap 2).
///
/// Both are guards, so the only test that means anything is one that makes them
/// FIRE. A guard whose trigger path is never executed is indistinguishable from
/// a guard that was accidentally wired to a condition that can never be true —
/// which is the same class of defect as a dispatch that silently selects the
/// default path. So every pin here poisons real state and observes the error
/// path, rather than asserting that clean input stays clean.
///
/// **Merge note.** This module is appended at the end of `train.rs`, as are
/// `scene_scale_tests` and `plane_feature_tests`. Git's line-level 3-way
/// heuristic interleaves fragments of such modules into invalid Rust that still
/// looks like a plausible merge, so integration must treat each `#[cfg(test)]
/// mod` as ONE opaque block and concatenate whole modules. Keep this module
/// self-contained (it takes nothing from outside `super::*`) and do not append
/// unrelated items after it.
#[cfg(test)]
mod nonfinite_guard_tests {
    use super::*;
    use brush_render::gaussian_splats::SplatRenderMode;
    use brush_render::kernels::camera_model::CameraModel;

    const IMG: glam::UVec2 = glam::uvec2(32, 32);
    const N_SIDE: usize = 4;

    fn guard_camera() -> Camera {
        Camera::new(
            glam::vec3(0.0, 0.0, -5.0),
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    /// A small fronto-parallel slab at `z = 0`, entirely inside the frame.
    ///
    /// With `poison`, splat 0's SH DC term is `NaN`. SH is chosen deliberately
    /// over the mean: a `NaN` mean projects to a `NaN` screen position and the
    /// visibility test rejects it (every float comparison against `NaN` is
    /// false), so the splat is CULLED and the loss stays finite — the poison
    /// would never reach the thing under test. A `NaN` colour on a splat that
    /// still rasterizes normally does reach it.
    fn guard_splats(device: &Device, poison: bool) -> Splats {
        let mut means = Vec::new();
        for iy in 0..N_SIDE {
            for ix in 0..N_SIDE {
                let f = |i: usize| (i as f32 / (N_SIDE - 1) as f32) * 2.0 - 1.0;
                means.extend_from_slice(&[f(ix), f(iy), 0.0]);
            }
        }
        let n = means.len() / 3;
        let rotations: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
        let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.2, -1.6, -2.5]).collect();
        let mut sh: Vec<f32> = (0..n).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        if poison {
            sh[0] = f32::NAN;
        }
        let opac: Vec<f32> = vec![4.0; n];
        Splats::from_raw(
            means,
            rotations,
            log_scales,
            sh,
            opac,
            SplatRenderMode::Default,
            device,
        )
    }

    fn guard_batch() -> SceneBatch {
        let (h, w) = (IMG.y as usize, IMG.x as usize);
        let img_packed = TensorData::new(
            (0..h * w)
                .map(|i| {
                    // Deterministic opaque RGBA. Wrapping on purpose: this is a
                    // hash, not arithmetic, and a plain multiply overflows.
                    let rgb = (i as u32).wrapping_mul(2_654_435_761) & 0x00ff_ffff;
                    (rgb | 0xff00_0000) as i32
                })
                .collect::<Vec<i32>>(),
            [h, w],
        );
        SceneBatch {
            img_packed,
            has_alpha: false,
            alpha_mode: AlphaMode::Transparent,
            features: None,
            depth: None,
            normal: None,
            camera: guard_camera(),
            view_index: 0,
        }
    }

    fn guard_trainer(allow_nonfinite_loss: bool, device: &Device) -> SplatTrainer {
        let config = TrainConfig {
            allow_nonfinite_loss,
            ..Default::default()
        };
        SplatTrainer::new(
            &config,
            device,
            BoundingBox::from_min_max(glam::Vec3::splat(-2.0), glam::Vec3::splat(2.0)),
        )
    }

    /// The cadence: every step early, then aligned to the refine cadence.
    ///
    /// Pins BOTH halves. Only asserting the early window would accept a guard
    /// that stops checking forever afterwards; only asserting the late stride
    /// would accept one that skips the explosion-prone opening.
    #[test]
    fn loss_check_cadence_is_early_then_refine_aligned() {
        let device = Default::default();
        let mut trainer = guard_trainer(false, &device);
        let every = trainer.config.refine_every;
        assert!(every > 1, "test assumes a nontrivial refine cadence");

        // Early window: every step, whatever the iteration number.
        for step in [1u32, 2, 7, NONFINITE_LOSS_CHECK_STEPS] {
            trainer.step_count = step;
            assert!(
                trainer.should_check_loss_finite(step),
                "step {step} is inside the early window and must be checked"
            );
            // Even on an iteration that is NOT on the refine stride.
            assert!(trainer.should_check_loss_finite(every * 3 + 1));
        }

        // After it: only on the refine stride.
        trainer.step_count = NONFINITE_LOSS_CHECK_STEPS + 1;
        assert!(trainer.should_check_loss_finite(every * 4));
        assert!(!trainer.should_check_loss_finite(every * 4 + 1));
        assert!(!trainer.should_check_loss_finite(every * 4 - 1));
    }

    /// The escape hatch is off by default, so a non-finite loss ABORTS.
    #[tokio::test]
    #[should_panic(expected = "non-finite total loss")]
    async fn nonfinite_loss_aborts_by_default() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let trainer = guard_trainer(false, &device);
        let splats = guard_splats(&device, false);
        trainer.report_nonfinite_loss(f32::NAN, &splats, 42).await;
    }

    /// `--allow-nonfinite-loss` restores the old continue-anyway behaviour.
    #[tokio::test]
    async fn nonfinite_loss_escape_hatch_allows_continuing() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let trainer = guard_trainer(true, &device);
        let splats = guard_splats(&device, false);
        // Must return rather than panic. Both non-finite kinds, since `inf` and
        // `NaN` reach the check by different arithmetic.
        trainer.report_nonfinite_loss(f32::NAN, &splats, 7).await;
        trainer
            .report_nonfinite_loss(f32::INFINITY, &splats, 8)
            .await;
    }

    /// **The guard must be WIRED INTO `step`, not merely defined.**
    ///
    /// The sibling failure to a dispatch that never selects its branch: a
    /// perfectly correct `report_nonfinite_loss` is worth nothing if
    /// `should_check_loss_finite` is never consulted. So poison the accumulated
    /// total loss for real, run a real `step`, and require the abort.
    ///
    /// # Why the poison is a weight and not a splat
    ///
    /// The obvious injection — a `NaN` splat parameter — does NOT work, and
    /// finding that out is worth more than the test itself. Measured here: a
    /// splat whose SH DC term is `NaN` is pruned by
    /// `prune_non_finite_splats` (the sibling test proves that) yet leaves the
    /// step-0 loss perfectly FINITE. The rasterizer never lets the poison
    /// through: `NaN` colours are clamped away and `NaN` geometry fails the
    /// visibility and alpha-cutoff comparisons (every float compare against
    /// `NaN` is false), so the splat is simply skipped.
    ///
    /// That is exactly why gap 2 exists as a separate problem from gap 1: a
    /// non-finite splat is INVISIBLE to the loss, so no loss-side guard will
    /// ever catch it, and it survives all the way to the exported ply. The two
    /// guards are not redundant — neither one subsumes the other.
    ///
    /// So this pin injects on the side the loss guard actually watches: a
    /// non-finite weight makes the accumulated total non-finite, which is the
    /// shape a real numerical blow-up takes.
    #[tokio::test]
    #[should_panic(expected = "non-finite total loss")]
    async fn nonfinite_loss_guard_is_live_in_step() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let config = TrainConfig {
            anti_needle_weight: f32::INFINITY,
            ..Default::default()
        };
        let mut trainer = SplatTrainer::new(
            &config,
            &device,
            BoundingBox::from_min_max(glam::Vec3::splat(-2.0), glam::Vec3::splat(2.0)),
        );
        let _ = trainer
            .step(guard_batch(), guard_splats(&device, false))
            .await;
    }

    /// A `NaN` splat parameter does NOT reach the rendered image.
    ///
    /// Pinned as its own fact because the whole justification for gap 2's
    /// separate sweep rests on it: the loss guard cannot cover non-finite
    /// splats, because they never reach the loss. If a future rasterizer change
    /// DID start propagating splat `NaN`s into the image, this test fails and
    /// tells the reader that the containment argument needs revisiting — rather
    /// than the sweep quietly looking redundant.
    ///
    /// Stated on the RENDER rather than on a full `step`, deliberately. A step
    /// runs the backward, and `bwd_validate` asserts on `NaN` gradients
    /// whenever `brush-render`'s `debug-validation` is on — which a workspace
    /// build turns on for every crate through feature unification, even though
    /// `cargo test -p brush-train` alone does not. A step-level version of this
    /// test therefore passes standalone and fails under `cargo test
    /// --workspace`. The render is also simply where the mechanism lives.
    #[tokio::test]
    async fn a_nonfinite_splat_never_reaches_the_image() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let poisoned = guard_splats(&device, true);
        let counts = non_finite_splat_masks(&poisoned).counts().await;
        assert_eq!(counts.any, 1, "fixture must actually be poisoned");
        assert_eq!(
            counts.sh, 1,
            "the poison must be in the SH, as the fixture intends"
        );

        // A debug-validation build panics on the non-finite INPUT before the
        // renderer can demonstrate anything about its output, so there is
        // nothing here to observe. Skip rather than assert something weaker:
        // the claim this test makes is about the DEFAULT build, which is the
        // one that ships and the one gap 2 was measured on.
        if brush_render::validation::HARD_FAILS_ON_NON_FINITE {
            return;
        }

        let out = render_splats_for_training(
            poisoned,
            &guard_camera(),
            IMG,
            glam::Vec3::ZERO,
            false,
            RasterizationMode::Rgba,
            false,
        )
        .await;
        let img = out
            .img
            .into_data_async()
            .await
            .expect("image readback")
            .into_vec::<f32>()
            .expect("image as f32");
        let bad = img.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            bad,
            0,
            "{bad} of {} rendered values are non-finite. A non-finite splat now DOES \
             reach the image, so the loss guard partly covers gap 2 and the sweep's \
             rationale needs updating.",
            img.len()
        );
    }

    /// The complementary half: a clean step must NOT trip the guard.
    ///
    /// Without this, a guard hard-wired to `true` would pass the test above and
    /// break every real run.
    #[tokio::test]
    async fn clean_step_does_not_trip_the_guard() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let mut trainer = guard_trainer(false, &device);
        let splats = guard_splats(&device, false);
        let (_next, stats) = trainer.step(guard_batch(), splats).await;
        let loss: f32 = stats.loss.into_scalar_async().await.expect("loss readback");
        assert!(loss.is_finite(), "a clean scene produced a non-finite loss");
    }

    /// The out-of-refine sweep removes non-finite splats, and is INERT when
    /// there are none.
    ///
    /// The inert half is the byte-identity argument in test form: on a clean
    /// scene the sweep must report zero and return the same population, having
    /// touched neither the optimizer nor the refine record.
    #[tokio::test]
    async fn prune_non_finite_splats_removes_them_and_is_inert_when_clean() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let mut trainer = guard_trainer(false, &device);

        // One clean step so the optimizer and refine record exist, which is the
        // state the sweep has to keep in lockstep when it prunes.
        let (stepped, _stats) = trainer
            .step(guard_batch(), guard_splats(&device, false))
            .await;
        let n = stepped.num_splats();

        // Inert on a clean population.
        let (clean, pruned) = trainer
            .prune_non_finite_splats(100, stepped, "unit test (clean)")
            .await;
        assert_eq!(pruned, 0, "a clean scene must prune nothing");
        assert_eq!(
            clean.num_splats(),
            n,
            "the inert path must not change the population"
        );

        // And it fires on a poisoned one. Same count as the stepped population,
        // so the optimizer state the sweep reindexes lines up.
        let poisoned = guard_splats(&device, true);
        assert_eq!(
            poisoned.num_splats(),
            n,
            "test fixture must match the stepped splat count"
        );
        let (swept, pruned) = trainer
            .prune_non_finite_splats(101, poisoned, "unit test (poisoned)")
            .await;
        assert_eq!(pruned, 1, "the single poisoned splat must be pruned");
        assert_eq!(swept.num_splats(), n - 1);

        // And what survives is actually finite — the point of the exercise.
        let counts = non_finite_splat_masks(&swept).counts().await;
        assert_eq!(
            counts.any, 0,
            "the sweep left {} non-finite splats behind",
            counts.any
        );
    }
}
