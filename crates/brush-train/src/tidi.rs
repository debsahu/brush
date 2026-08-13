//! TIDI-GS floater / haze suppression (arXiv 2601.09291, "Training-time
//! Isolation-and-Detail-preserving pruning").
//!
//! Port of the SOTA indoor-360 haze remover into our Brush fork. Motivation and
//! the paper math are written up in `research/indoor-360-haze-removal.md`. The
//! one fact that drives the whole design: a wall floater settles into an
//! *opacity equilibrium* where its blended colour cancels the background, so its
//! opacity gradient vanishes and it is trapped in a spurious minimum. Opacity
//! thresholds and any post-hoc filter therefore CANNOT remove it — it has to be
//! caught at training time by a combination of signals, none of which is opacity
//! alone.
//!
//! The pass runs inside the existing refine cycle
//! ([`crate::train::SplatTrainer::refine_for_phase`]) and produces ONE extra
//! prune set that is unioned into the standard prune mask before the single
//! `prune_points` call. It never adds a loss to the RGB path except the optional
//! L1 sparsity on the learned importance `ω` (which is itself a real leaf
//! parameter, so signal 3 is alive on the photometric gradient). The whole
//! family is gated behind `--tidi-prune` and is completely inert (state never
//! even allocated) when the flag is off, so MRNF / depth-loss / PPISP /
//! normal-priors runs take the identical code path they always did.
//!
//! Design decisions where the paper and the Brush internals had to be
//! reconciled are called out inline with `NOTE:`.

use brush_render::burn_glue::{detach_autodiff, lift_to_autodiff};
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use burn::{
    Tensor,
    module::{Module, Param, ParamId},
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Bool, Device, Gradients, Int, TensorData, activation::sigmoid},
};
use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;

use crate::adam_scaled::{AdamScaled, AdamScaledConfig};

/// Raw `ω` init. `σ(ω)` starts at ≈0.998 so the learned-importance opacity gate
/// begins as near-identity (a ~0.2% opacity nudge when TIDI is first enabled)
/// and every fresh/split Gaussian starts far above `τ_ω` — i.e. "importance
/// ≈ 1.0" as the paper specifies, which also protects new detail from being an
/// immediate candidate until the sparsity term has had time to pull it down.
pub const OMEGA_INIT: f32 = 6.0;

/// The learnable per-Gaussian importance `ω_i`, kept as its own tiny module so
/// it gets a real Adam optimizer (moments, `to_record`/`load_record`
/// round-trip) exactly like the DiG feature table, and is kept in lockstep with
/// the splats through refine. It is training-time only — never exported, so the
/// ply / viewer / FFI surface is untouched.
#[derive(Module, Debug)]
pub struct ImportanceModule {
    /// `[N]` raw importance logits; `σ(ω)` is the importance in `[0, 1]`.
    pub omega: Param<Tensor<1>>,
}

impl ImportanceModule {
    fn new(num_splats: u32, device: &Device) -> Self {
        let omega = Tensor::<1>::ones([num_splats as usize], device).mul_scalar(OMEGA_INIT);
        Self {
            omega: Param::initialized(ParamId::new(), omega.require_grad()),
        }
    }
}

type ImportanceOptimizer = OptimizerAdaptor<AdamScaled, ImportanceModule>;

fn create_importance_optimizer() -> ImportanceOptimizer {
    AdamScaledConfig::new().with_epsilon(1e-15).init()
}

/// Trainer-owned TIDI state. Everything here PERSISTS across refine cycles — it
/// is deliberately NOT stored in [`crate::stats::RefineRecord`], which is
/// window-scoped and rebuilt after every refine (that would reset the
/// accumulators the paper needs to grow over training).
///
/// `omega` lives on the autodiff backend (it is a leaf that gets a photometric
/// gradient every step); the three accumulators are pure data on the inner
/// backend, matching the `RefineRecord` tensors they are fed from.
///
/// GAP-5 (LOD): "persists across cycles" holds WITHIN a training phase. Under
/// `--lod-levels > 0` the process rebuilds `SplatTrainer` at each LOD boundary
/// (see `train_stream`), and — unlike `appearance`, which is carried via
/// `set_appearance` — TIDI state is NOT carried, so `vis_accum` / `grad_ema` /
/// `omega` reset and the per-Gaussian warmup restarts on the decimated splats.
/// This fails safe (no crash; TIDI simply re-warms and re-learns per phase), but
/// a floater surviving decimation loses its accumulated history. Carrying it
/// across the boundary would need the LOD decimation to expose surviving indices
/// for a `keep`-style reindex; not done here.
pub struct TidiState {
    importance: ImportanceModule,
    optim: ImportanceOptimizer,
    /// `[N]` cumulative count of refine windows in which the Gaussian was
    /// visible (paper signal (a), "multi-view visibility count").
    vis_accum: Tensor<1>,
    /// `[N]` EMA of the per-window position-gradient signal (paper signal (b)).
    grad_ema: Tensor<1>,
    /// `[N]` global iteration at which each Gaussian was born / last split.
    /// Feeds the per-Gaussian warmup so fresh detail is never pruned early.
    birth_iter: Tensor<1, Int>,
    /// Global iter of the last cleanup pass; gates the ~400-step cadence.
    last_prune_iter: Option<u32>,
    /// `[N]` cumulative count of training VIEWS in which this Gaussian projected
    /// in FRONT of a valid measured depth return by more than the margin (i.e.
    /// "floated in empty space between the camera and the surface"). Depth-prune
    /// path only; stays zero (and inert) unless `--tidi-depth-prune` is on.
    float_accum: Tensor<1>,
    /// `[N]` cumulative count of training VIEWS in which a VALID depth return
    /// existed at this Gaussian's projected pixel (regardless of float). The
    /// denominator for `float_frac`, and the safety gate: a Gaussian in an
    /// unscanned region (few valid returns behind it) is never depth-pruned.
    valid_accum: Tensor<1>,
}

/// The four per-Gaussian signals, read back to the host once per cleanup for the
/// candidate rule and the adaptive guard quantiles.
struct HostSignals {
    n: usize,
    vis: Vec<f32>,
    grad_ema: Vec<f32>,
    opacity: Vec<f32>,
    sigma_w: Vec<f32>,
    /// Non-DC SH energy `‖f_rest‖₂` per Gaussian (guard 1).
    sh_hf: Vec<f32>,
    /// Smallest scale axis `s₁` per Gaussian (thinness guard).
    min_scale: Vec<f32>,
    /// Anisotropy `s₃/s₁` (largest / smallest scale axis) per Gaussian
    /// (anisotropy guard — protects elongated sheet / needle structures).
    aniso: Vec<f32>,
    /// Positions `[N*3]` for the isolation k-NN.
    pos: Vec<f32>,
    /// DC colour `[N*3]` for the optional local-colour-variance guard.
    dc_color: Vec<f32>,
    /// Age in steps since birth/last split, for the per-Gaussian warmup.
    age: Vec<i32>,
    /// Depth-prune path (`float_accum` / `valid_accum`): per-Gaussian counts of
    /// views it floated in front of a surface, and views a valid return existed
    /// behind it. All-zero unless `--tidi-depth-prune` accumulated.
    float_accum: Vec<f32>,
    valid_accum: Vec<f32>,
}

impl TidiState {
    /// Allocate for `num_splats`. `device` is the autodiff device (for `ω`); the
    /// accumulators are placed on its inner counterpart.
    pub fn new(num_splats: u32, cur_iter: u32, device: &Device) -> Self {
        // `ω` lives on the autodiff `device`; the accumulators are pure data fed
        // from the inner-backend `RefineRecord`, so they go on its inner device.
        let inner = device.clone().inner();
        let n = num_splats as usize;
        Self {
            importance: ImportanceModule::new(num_splats, device),
            optim: create_importance_optimizer(),
            vis_accum: Tensor::<1>::zeros([n], &inner),
            grad_ema: Tensor::<1>::zeros([n], &inner),
            birth_iter: Tensor::<1, Int>::full([n], cur_iter as i32, &inner),
            last_prune_iter: None,
            float_accum: Tensor::<1>::zeros([n], &inner),
            valid_accum: Tensor::<1>::zeros([n], &inner),
        }
    }

    /// `σ(ω)` on the autodiff graph — used to build the opacity gate that gives
    /// `ω` its photometric gradient (see [`Self::gate_opacity`]).
    pub fn sigma_omega(&self) -> Tensor<1> {
        sigmoid(self.importance.omega.val())
    }

    /// Gate a render-input opacity by the learned importance:
    /// `α' = α · σ(ω)`, in raw (pre-sigmoid) logit space so the render path's
    /// sigmoid + 3D-filter fold see the gated value. Differentiable w.r.t. BOTH
    /// the opacity leaf and `ω`, so the photometric loss trains importance up
    /// for Gaussians that matter and (with the L1 term) lets idle ones decay —
    /// paper signal (c). `raw_opac` and the returned tensor are on the autodiff
    /// backend.
    pub fn gate_opacity(&self, raw_opac: Tensor<1>) -> Tensor<1> {
        let alpha = sigmoid(raw_opac);
        let gated = (alpha * self.sigma_omega()).clamp(1e-6, 1.0 - 1e-6);
        // logit(gated)
        gated.clone().div(gated.neg().add_scalar(1.0)).log()
    }

    /// L1 sparsity penalty `mean(σ(ω))`, scaled by the caller's weight and added
    /// to the training loss. Pulls every importance toward zero; the photometric
    /// gate pushes contributing Gaussians back up, so the two balance at the
    /// `τ_ω` boundary. Weight 0 leaves `ω` on the photometric gradient only
    /// (importance never falls → the gate reduces to the other three signals).
    pub fn sparsity_loss(&self) -> Tensor<1> {
        self.sigma_omega().mean()
    }

    /// Pull `ω`'s gradient out of this step's backward pass and take one Adam
    /// step, mirroring the DiG feature step (clone the module out, extract the
    /// grad by id, step, write back). `ω` gets its signal from the photometric
    /// (and depth) loss via the opacity gate plus the L1 sparsity term.
    pub fn optimize(&mut self, lr: f64, grads: &mut Gradients) {
        let module = self.importance.clone();
        let grad = GradientsParams::from_params(grads, &module, &[module.omega.id]);
        self.importance = self.optim.step(lr, module, grad);
    }

    /// True on cleanup cycles: at/after `start_iter` and at least `every` steps
    /// since the previous pass. Signals still accumulate every refine window
    /// regardless (see [`Self::accumulate_window`]); only the prune is throttled.
    pub fn should_prune(&self, cur_iter: u32, start_iter: u32, every: u32) -> bool {
        if cur_iter < start_iter {
            return false;
        }
        match self.last_prune_iter {
            None => true,
            Some(last) => cur_iter.saturating_sub(last) >= every.max(1),
        }
    }

    pub fn mark_pruned(&mut self, cur_iter: u32) {
        self.last_prune_iter = Some(cur_iter);
    }

    /// Fold one refine window's stats into the persistent accumulators. `vis` and
    /// `grad` are the `RefineRecord` window quantities (visibility count and the
    /// window position-gradient signal), both inner and aligned to the current
    /// (pre-prune) splats.
    ///
    /// NOTE (visibility scale): `RefineRecord::vis_weight` is `+= visible` on
    /// EVERY training step (one view/step), so it already scales with the number
    /// of steps in a window (tens-to-hundreds). Summing that raw across windows
    /// would put `vis_accum` in the hundreds-to-thousands, and `τ_vis = 2.0`
    /// (`fail_vis` = `vis ≤ 2`) would be unreachable for any persistently
    /// rendered gaussian — signal (a) would never fire. So we collapse each
    /// window to a 0/1 "seen at least once this window" indicator and sum THAT:
    /// `vis_accum` is "number of refine windows in which this gaussian was ever
    /// visible", and `τ_vis` is on that window scale.
    ///
    /// NOTE (gradient EMA cadence): the paper's β=0.99 EMA is defined per
    /// training step over `‖∇_x L‖₂`. Brush only materialises a per-refine-window
    /// position-gradient signal (`RefineRecord::refine_weight_norm`, the same
    /// quantity the growth gate thresholds), so the EMA here advances once per
    /// refine window, not per step. β is exposed so the operator can compensate
    /// for the coarser cadence.
    pub fn accumulate_window(&mut self, vis: Tensor<1>, grad: Tensor<1>, beta: f32) {
        let seen = vis.greater_elem(0.0).float();
        self.vis_accum = self.vis_accum.clone() + seen;
        self.grad_ema = self.grad_ema.clone().mul_scalar(beta) + grad.mul_scalar(1.0 - beta);
    }

    /// Fold ONE training view's LiDAR/depth residual into the persistent
    /// `float_accum` / `valid_accum` counters (the depth-prune path's signal).
    /// Runs per step whenever `--tidi-depth-prune` is on and the batch carries a
    /// depth map; NOT gated on `--depth-loss-weight` (this path is independent of
    /// the depth *loss*, it only reuses the same per-frame depth tensor).
    ///
    /// For every Gaussian, project its centre `mu_i` into this view, read the
    /// ground-truth depth `Z̃` at the projected pixel, and compare against the
    /// Gaussian's own camera-space depth `z`:
    ///   * `valid` when `Z̃` is a real return (`> 0`, finite) AND the projection
    ///     lands in-frame in front of the camera (`z > 0`) → `valid_accum += 1`;
    ///   * `floating` when additionally `z - Z̃ < -margin`, i.e. the Gaussian sits
    ///     more than `margin` in FRONT of the measured surface → `float_accum +=
    ///     1`. `margin` is in the depth map's own units (metres for LiDAR/metric
    ///     depth; possibly non-metric for SfM depth).
    ///
    /// Everything is one GPU tensor pass on the inner backend (no host readback,
    /// no autodiff graph) so an enabled run pays a handful of elementwise ops per
    /// step and a disabled run pays nothing (the whole call is gated out).
    ///
    /// PROJECTION: the world→camera transform is Brush's own
    /// [`Camera::world_to_local`]; the pixel projection is the pinhole model
    /// (matching [`brush_render::kernels::camera_model::pinhole::project_pinhole`]:
    /// `u = fx·x/z + cx`, `v = fy·y/z + cy`). Non-pinhole camera models
    /// (fisheye / radial-tangential) are not implemented on this host-tensor path
    /// — for them the call warns once and no-ops, exactly like the depth/normal
    /// consistency term (`warn_depth_normal_needs_pinhole`).
    pub fn accumulate_depth(
        &mut self,
        means: Tensor<2>,
        gt_depth: TensorData,
        camera: &Camera,
        img_size: glam::UVec2,
        margin: f32,
    ) {
        // Only pinhole projection is implemented here; other models unproject
        // differently (see the depth/normal term's identical restriction).
        if !matches!(camera.camera_model, CameraModel::Pinhole) {
            warn_depth_prune_non_pinhole();
            return;
        }

        // Everything runs on the accumulators' (inner) device. `detach_autodiff`
        // drops the means to the INNER backend (no graph retained) and places
        // them on that device; the shared projection then builds gt/R/t there
        // too, so the whole prune-path projection stays inner-kind — matching the
        // inner accumulators it feeds.
        let device = self.valid_accum.device();
        let means = detach_autodiff(means).to_device(&device);
        let Some((residual, valid)) =
            project_depth_residual(means, gt_depth, camera, img_size, &device)
        else {
            return;
        };
        // Floating: in front of the surface by more than the margin.
        let floating = valid.clone().bool_and(residual.lower_elem(-margin));

        self.valid_accum = self.valid_accum.clone() + valid.float();
        self.float_accum = self.float_accum.clone() + floating.float();
    }

    /// Reindex through a prune. `valid_inds` (autodiff) reindexes `ω` and its
    /// Adam state; `inner_valid_inds` reindexes the inner accumulators. Called
    /// from `prune_points` beside the DiG `keep`, so the tables can never
    /// silently desync from the splats.
    pub fn keep(&mut self, valid_inds: &Tensor<1, Int>, inner_valid_inds: &Tensor<1, Int>) {
        self.importance.omega = self
            .importance
            .omega
            .clone()
            .map(|x| x.select(0, valid_inds.clone()));
        let mut record = self.optim.to_record();
        if record.contains_key(&self.importance.omega.id) {
            crate::train::map_opt(self.importance.omega.id, &mut record, &|x: Tensor<1>| {
                x.select(0, valid_inds.clone())
            });
            self.optim = create_importance_optimizer().load_record(record);
        }
        self.vis_accum = self.vis_accum.clone().select(0, inner_valid_inds.clone());
        self.grad_ema = self.grad_ema.clone().select(0, inner_valid_inds.clone());
        self.birth_iter = self.birth_iter.clone().select(0, inner_valid_inds.clone());
        // Depth-prune accumulators ride the identical inner reindex.
        self.float_accum = self.float_accum.clone().select(0, inner_valid_inds.clone());
        self.valid_accum = self.valid_accum.clone().select(0, inner_valid_inds.clone());
    }

    /// Reindex through a split. Children are APPENDED (matching `refine_splats`'
    /// `cat` order) and, per the paper, get a FRESH state rather than inheriting
    /// the parent's: `ω = OMEGA_INIT` (protected, importance ≈ 1), zero
    /// visibility, zero grad-EMA, and `birth_iter = cur_iter` so the warmup
    /// starts over. The parent rows are left untouched (unlike DiG, which zeroes
    /// both halves — a parent's importance history stays valid through a split).
    pub fn split(
        &mut self,
        refine_count: usize,
        refine_inds_opt: &Tensor<1, Int>,
        opt_device: &Device,
        inner_device: &Device,
        cur_iter: u32,
    ) {
        let device = self.importance.omega.device();
        let fresh = Tensor::<1>::ones([refine_count], &device).mul_scalar(OMEGA_INIT);
        self.importance.omega = self
            .importance
            .omega
            .clone()
            .map(|x| Tensor::cat(vec![x, fresh.clone()], 0));
        let mut record = self.optim.to_record();
        if record.contains_key(&self.importance.omega.id) {
            let opt_device = opt_device.clone();
            // Parent moments untouched; children start at zero.
            let _ = refine_inds_opt; // symmetry with DiG's signature; not needed here.
            crate::train::map_opt(self.importance.omega.id, &mut record, &move |x: Tensor<
                1,
            >| {
                Tensor::cat(vec![x, Tensor::<1>::zeros([refine_count], &opt_device)], 0)
            });
            self.optim = create_importance_optimizer().load_record(record);
        }
        let zeros = Tensor::<1>::zeros([refine_count], inner_device);
        self.vis_accum = Tensor::cat(vec![self.vis_accum.clone(), zeros.clone()], 0);
        self.grad_ema = Tensor::cat(vec![self.grad_ema.clone(), zeros.clone()], 0);
        let births = Tensor::<1, Int>::full([refine_count], cur_iter as i32, inner_device);
        self.birth_iter = Tensor::cat(vec![self.birth_iter.clone(), births], 0);
        // Children start with FRESH depth counters (zero float / zero valid): a
        // split child cannot reach `min_valid_views` until it has actually been
        // observed behind a surface again, which is exactly the per-Gaussian
        // protection the photometric path gets from `birth_iter` + warmup.
        self.float_accum = Tensor::cat(vec![self.float_accum.clone(), zeros.clone()], 0);
        self.valid_accum = Tensor::cat(vec![self.valid_accum.clone(), zeros], 0);
    }

    /// Read every signal to the host for the cleanup decision. Runs only on
    /// cleanup cycles (≈ every 400 steps), so the readback cost is amortised.
    /// `opacity`, `sh_coeffs` (`[N, C, 3]`) and `scales` (`[N, 3]`) come from the
    /// current pre-prune splats.
    async fn read_signals(
        &self,
        cur_iter: u32,
        opacity: Tensor<1>,
        means: Tensor<2>,
        sh_coeffs: Tensor<3>,
        scales: Tensor<2>,
    ) -> HostSignals {
        let n = self.vis_accum.dims()[0];
        // σ(ω) with the grad detached — the mask needs a plain inner value.
        let sigma_w = detach_autodiff(self.sigma_omega());

        // Non-DC SH energy ‖f_rest‖₂ = sqrt(Σ over the SH bands ≥ 1 of coeff²).
        let [_, c, _] = sh_coeffs.dims();
        let sh_hf: Tensor<1> = if c > 1 {
            sh_coeffs
                .clone()
                .slice(burn::tensor::s![.., 1..c, ..])
                .powi_scalar(2)
                .sum_dim(2)
                .sum_dim(1)
                .squeeze_dim::<2>(2)
                .squeeze_dim::<1>(1)
                .sqrt()
        } else {
            Tensor::<1>::zeros([n], &opacity.device())
        };
        // DC colour (SH band 0) for the optional local-colour-variance guard.
        let dc_color = sh_coeffs
            .slice(burn::tensor::s![.., 0..1, ..])
            .squeeze_dim::<2>(1);

        async fn host1(t: Tensor<1>) -> Vec<f32> {
            t.into_data_async()
                .await
                .expect("tidi readback")
                .into_vec()
                .expect("f32")
        }
        async fn host2(t: Tensor<2>) -> Vec<f32> {
            t.into_data_async()
                .await
                .expect("tidi readback")
                .into_vec()
                .expect("f32")
        }

        // Read birth iters and turn them into an age (in steps) on the host, so
        // no Int-tensor arithmetic is needed.
        let birth: Vec<i32> = self
            .birth_iter
            .clone()
            .into_data_async()
            .await
            .expect("tidi birth readback")
            .into_vec()
            .expect("i32");
        let age: Vec<i32> = birth.iter().map(|&b| cur_iter as i32 - b).collect();

        // Per-axis scales `[N*3]` read to the host once; the thinness guard wants
        // the smallest axis `s₁` and the anisotropy guard wants `s₃/s₁`. NOTE:
        // Brush scales are XYZ axes, NOT rank-ordered, so min/max over the three
        // (not a fixed column) is what gives `s₁` / `s₃`.
        let scales_host = host2(scales).await;
        let mut min_scale = vec![0.0f32; n];
        let mut aniso = vec![1.0f32; n];
        for i in 0..n {
            let s = [
                scales_host[i * 3],
                scales_host[i * 3 + 1],
                scales_host[i * 3 + 2],
            ];
            let mn = s.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            min_scale[i] = mn;
            aniso[i] = if mn > 1e-20 { mx / mn } else { f32::INFINITY };
        }

        HostSignals {
            n,
            vis: host1(self.vis_accum.clone()).await,
            grad_ema: host1(self.grad_ema.clone()).await,
            opacity: host1(opacity).await,
            sigma_w: host1(sigma_w).await,
            sh_hf: host1(sh_hf).await,
            min_scale,
            aniso,
            pos: host2(means).await,
            dc_color: host2(dc_color).await,
            age,
            float_accum: host1(self.float_accum.clone()).await,
            valid_accum: host1(self.valid_accum.clone()).await,
        }
    }

    /// Compute the TIDI prune set over the current pre-prune splats and return it
    /// as a boolean `[N]` mask (inner) ready to be unioned into the standard
    /// prune mask. `None` means "nothing to prune this cycle". Marks the cleanup
    /// cycle as done.
    #[allow(clippy::too_many_arguments)]
    pub async fn select_prune_mask(
        &mut self,
        cfg: &TidiPruneParams,
        cur_iter: u32,
        opacity: Tensor<1>,
        means: Tensor<2>,
        sh_coeffs: Tensor<3>,
        scales: Tensor<2>,
        device: &Device,
    ) -> Option<Tensor<1, Bool>> {
        self.mark_pruned(cur_iter);
        let s = self
            .read_signals(cur_iter, opacity, means, sh_coeffs, scales)
            .await;
        // Depth-prune was requested but no view has ever produced a valid depth
        // return: the dataset carries no per-frame depth (nerfstudio `depth:
        // None`) or every map was empty. Warn ONCE and let the depth path no-op
        // (all `valid_accum` are 0, so no depth candidate can form) — never panic.
        if cfg.depth_prune && s.valid_accum.iter().all(|&v| v <= 0.0) {
            warn_depth_prune_no_depth();
        }
        let prune_idx = select_prune_indices(cfg, &s);
        if prune_idx.is_empty() {
            log::debug!("tidi iter={cur_iter}: 0 pruned (candidates gated out)");
            return None;
        }
        log::debug!(
            "tidi iter={cur_iter}: pruning {} floaters (of {} gaussians)",
            prune_idx.len(),
            s.n
        );
        // Scatter the selected indices into a dense mask; return it as Bool
        // (via a float 0/1 tensor) so it can be OR-ed into the standard prune
        // mask on the same backend.
        let mut mask = vec![0.0f32; s.n];
        for &i in &prune_idx {
            mask[i as usize] = 1.0;
        }
        let t = Tensor::<1>::from_data(TensorData::new(mask, [s.n]), device);
        Some(t.greater_elem(0.5))
    }
}

/// Pinhole projection + GT-depth lookup for the hard `accumulate_depth` prune
/// counter (`--tidi-depth-prune`). The smooth opacity regularizer no longer uses
/// this: it moved to a VIEW-INDEPENDENT 3D distance-to-cloud gate
/// ([`CloudDistanceGrid`] + [`depth_opacity_reg_loss`]), which needs no per-view
/// projection at all. This helper stays for the prune path, which still counts
/// per-view "floated in front of the measured surface" events.
///
/// The BACKEND is the caller's responsibility: the prune path detaches onto the
/// INNER backend (its accumulators live there), so the caller must grad-stop
/// `means` and place it on `device`'s backend BEFORE calling — via
/// `detach_autodiff(..).to_device(inner)`. This helper builds `gt` / R / t on
/// `device` and does the projection on `means` directly, so every tensor stays
/// on `means`'s backend end to end.
///
/// Projects each Gaussian centre into the view, reads the GT depth `Z̃` at the
/// projected pixel, and returns:
///   * `residual` `[N]` — the signed camera-space depth residual `r_i = z_i - Z̃`
///     (< -margin means the Gaussian floats in FRONT of the measured surface);
///   * `valid`    `[N]` — true where the projection is in-frame, in front of the
///     camera (`z > 0`), and lands on a finite positive depth return.
/// Callers guard pinhole themselves (each owns its warning); this returns `None`
/// only for an empty depth map.
///
/// PROJECTION: the world→camera transform is Brush's own [`Camera::world_to_local`];
/// the pixel projection is the pinhole model (matching
/// [`brush_render::kernels::camera_model::pinhole::project_pinhole`]:
/// `u = fx·x/z + cx`, `v = fy·y/z + cy`). The end-to-end projection + residual is
/// verified on-device in `accumulate_depth_counts_front_surface_and_unscanned`.
fn project_depth_residual(
    means: Tensor<2>,
    gt_depth: TensorData,
    camera: &Camera,
    img_size: glam::UVec2,
    device: &Device,
) -> Option<(Tensor<1>, Tensor<1, Bool>)> {
    // `means` arrives already grad-stopped and on `device`'s backend (see the
    // doc above). Build gt + R + t on the SAME `device` so the whole projection
    // stays on that one backend.
    let gt = Tensor::<2>::from_data(gt_depth, device);
    let [h, w] = gt.dims();
    if h == 0 || w == 0 {
        return None;
    }

    // World → camera: `mean_cam = R·mu + t`, R/t from Brush's own camera.
    // Building the [3,3] tensor from `matrix3.to_cols_array()` (glam is
    // column-major) yields exactly Rᵀ in row-major, so `means @ Rᵀ` gives the
    // rotated points as row vectors.
    let w2c = camera.world_to_local();
    let cols = w2c.matrix3.to_cols_array();
    let r_t = Tensor::<2>::from_data(TensorData::new(cols.to_vec(), [3, 3]), device);
    let t = w2c.translation;
    let trans =
        Tensor::<1>::from_data(TensorData::new(vec![t.x, t.y, t.z], [3]), device).reshape([1, 3]);
    let mean_cam = means.matmul(r_t) + trans; // [N, 3]

    let focal = camera.focal(img_size);
    let center = camera.center(img_size);
    let x = mean_cam.clone().slice(burn::tensor::s![.., 0..1]);
    let y = mean_cam.clone().slice(burn::tensor::s![.., 1..2]);
    let z = mean_cam.slice(burn::tensor::s![.., 2..3]);
    let inv_z = z.clone().recip();
    // project_pinhole: u = fx·x/z + cx, v = fy·y/z + cy.
    let u = (x * inv_z.clone())
        .mul_scalar(focal.x)
        .add_scalar(center.x)
        .squeeze_dim::<1>(1); // [N]
    let v = (y * inv_z)
        .mul_scalar(focal.y)
        .add_scalar(center.y)
        .squeeze_dim::<1>(1);
    let z = z.squeeze_dim::<1>(1); // [N] camera-space depth
    let ur = u.round();
    let vr = v.round();

    // In-frame: z > 0 and the rounded pixel lands inside [0, W-1]×[0, H-1].
    // (`>= 0` is written as `!(<0)` since only strict comparators are used
    // elsewhere in this module; NaN u/v from a behind-camera splat compare
    // false here and are excluded — and masked out by `in_front` regardless.)
    let in_front = z.clone().greater_elem(0.0);
    let in_x = ur
        .clone()
        .lower_elem(0.0)
        .bool_not()
        .bool_and(ur.clone().greater_elem((w - 1) as f32).bool_not());
    let in_y = vr
        .clone()
        .lower_elem(0.0)
        .bool_not()
        .bool_and(vr.clone().greater_elem((h - 1) as f32).bool_not());
    // Defensive finite gate on u/v: a `0 · ∞` (mean at x=0 AND z=0) or any other
    // degenerate projection yields NaN u/v, and `NaN < 0` / `NaN > w-1` are both
    // false, so `in_x`/`in_y` would spuriously read TRUE. Excluding non-finite
    // u/v keeps such a row OUT of `valid`, which is what both callers rely on for
    // NaN-safety (the loss path's penalty substitution and the prune path's
    // bool_and both assume a NaN row is never valid).
    let uv_finite = ur.clone().is_finite().bool_and(vr.clone().is_finite());
    let in_frame = in_front.bool_and(in_x).bool_and(in_y).bool_and(uv_finite);

    // Flat pixel index (row-major, row = v = y, col = u = x). Clamp to a valid
    // range so the gather is always in-bounds even for the masked-out (NaN /
    // behind-camera) rows we discard afterwards.
    let uc = ur.clamp(0.0, (w - 1) as f32);
    let vc = vr.clamp(0.0, (h - 1) as f32);
    let flat = (vc.mul_scalar(w as f32) + uc)
        .clamp(0.0, (h * w - 1) as f32)
        .int(); // [N]
    let gt_flat = gt.reshape([(h * w) as i32]);
    // `select` on a 1-D source with a 1-D index is a per-element gather.
    let z_tilde = gt_flat.select(0, flat); // [N]

    // Valid return: a real, finite, positive depth AND an in-frame pixel.
    let ret_valid = z_tilde
        .clone()
        .greater_elem(0.0)
        .bool_and(z_tilde.clone().is_finite());
    let valid = in_frame.bool_and(ret_valid);
    let residual = z - z_tilde; // z - Z̃ ; < -margin means "in front"
    Some((residual, valid))
}

// ---- 3D distance-to-cloud opacity regularizer --------------------------------
//
// FIX (2026-08): the depth-coupled opacity regularizer was gated on a PER-VIEW
// z-buffer residual (project each Gaussian, compare its depth to the projected
// LiDAR z-buffer). That z-buffer is sparse + dilated, so background leaks through
// foreground gaps ("see-through") and marks on-surface splats as floating — no
// margin fixes it, and it fades real surface.
//
// The gate is now VIEW-INDEPENDENT: a Gaussian is a floater when its centre is
// FAR from the seed/LiDAR point cloud in 3D. The cloud IS the measured surface,
// so distance-to-nearest-cloud-point is the honest floater signal, with no camera
// and no z-buffer to leak through. It is a per-step O(1) grid lookup against a
// static distance field built ONCE from the cloud (see [`CloudDistanceGrid`]).
//
// Note the SIGN flip vs the old residual version: now FAR-from-cloud is penalized
// (`d > margin`); before, IN-FRONT-of-surface was penalized (`r < -margin`).

/// Voxels per `margin` along an axis for the distance field. Finer (larger)
/// keeps the on-surface quantisation error a small fraction of `margin`, so
/// correctly-reconstructed surface splats read `d ~ 0` (penalty ~ 0); coarser
/// blows the grid up cubically. 3 keeps the worst in-voxel error ~0.29·margin.
const VOX_PER_MARGIN: f32 = 3.0;
/// The field is computed accurately out to `margin + REACH_SOFT·softness`; every
/// voxel farther than that reads `max_dist` (below), where the ramp is already
/// saturated so the exact distance is irrelevant. Bounds the per-voxel search.
const REACH_SOFT: f32 = 2.0;
/// Far-field / truncation distance, in `softness` units past `margin`. A voxel
/// with no nearby cloud point stores `margin + FAR_SPAN·softness`, whose
/// `p = σ(FAR_SPAN) ≈ 1`, so genuinely empty space is fully penalized.
const FAR_SPAN: f32 = 6.0;
/// Hard cap on total voxels (~4 bytes each): `vox` is grown until the padded
/// grid fits. 24e6 ≈ 96 MB. Raise on a big-VRAM box for finer far scenes.
const MAX_VOXELS: usize = 24_000_000;
/// Half the voxel space-diagonal (√3/2). The stored field is the voxel-CENTRE to
/// nearest-point distance, but the queried Gaussian can sit up to this·vox from
/// its voxel centre; subtracting it makes the stored value a conservative LOWER
/// bound on the true Gaussian→cloud distance, so a surface splat is never
/// spuriously penalized by quantisation.
const HALF_DIAG: f32 = 0.866_025_4;

// ---- LiDAR plane priors (RANSAC on the seed cloud) ---------------------------
//
// The LiDAR version of PlanarGS. The seed/LiDAR cloud is SPARSE, so a genuine
// wall splat that sits BETWEEN cloud points reads "far from every cloud point"
// on the point-only distance field and is wrongly penalized (FIX 1 / see-through
// holes). A PLANE fit to the wall's inliers INTERPOLATES those gaps: any splat
// within the plane's bounded extent and near the plane is on-surface, even where
// no individual cloud point is nearby. A mid-air floater is far from EVERY plane,
// so it is still cleanly caught.
//
// Detection is geometric, NOT a VLM: iterative RANSAC on the point cloud. Runs
// ONCE on the host at init (over the ~1M seed points), so it is CPU + rayon; the
// planes it finds feed both FIX 1 (the augmented distance field) and FIX 2 (the
// co-planarity geometry constraint).

/// Iterations of the RANSAC minimal-sample loop per extracted plane. Each
/// iteration samples 3 points, fits a plane, and counts inliers; the best over
/// all iterations is refined and kept. 1000 is ample for a wall that occupies a
/// meaningful fraction of the cloud (the hit probability per sample is high).
const RANSAC_ITERS_PER_PLANE: usize = 1000;
/// Max planes to extract. Walls + floor + ceiling of a room is ~6; 8 leaves slack.
const RANSAC_MAX_PLANES: usize = 8;
/// Keep an extracted plane only if its inliers are at least this fraction of the
/// (original) cloud. Below it, the "plane" is noise/detail, so extraction stops.
const RANSAC_MIN_INLIER_FRAC: f32 = 0.02;
/// RANSAC inlier band = this multiple of the estimated nearest-neighbour spacing.
/// Wide enough that a slightly noisy wall is one plane, tight enough that two
/// parallel walls a few spacings apart are not merged.
const RANSAC_THRESH_SPACING_MULT: f32 = 2.5;

/// One extracted plane. The plane is `n · x = d` with `n` a unit normal; the
/// bounded extent is stored as an in-plane orthonormal basis `(u, v)` plus the
/// axis-aligned bounds of the inliers' `(u, v)` projections, so membership is a
/// cheap "does the query project inside `[u_min,u_max] × [v_min,v_max]`" test.
/// This keeps the plane finite (the actual wall), not the infinite plane.
#[derive(Clone, Debug)]
pub struct Plane {
    /// Unit normal `n`.
    pub normal: [f32; 3],
    /// Offset `d` in `n · x = d`.
    pub offset: f32,
    /// In-plane orthonormal basis vector `u`.
    pub u_axis: [f32; 3],
    /// In-plane orthonormal basis vector `v` (`= n × u`).
    pub v_axis: [f32; 3],
    /// Inlier `u`-projection bounds.
    pub u_min: f32,
    pub u_max: f32,
    /// Inlier `v`-projection bounds.
    pub v_min: f32,
    pub v_max: f32,
    /// Inlier count as a fraction of the whole cloud (for logging).
    pub inlier_frac: f32,
}

/// The planes extracted from a seed cloud, stored on the trainer so BOTH the
/// augmented distance field (FIX 1) and the co-planarity constraint (FIX 2) read
/// the same geometry. Cheap + device-independent (≤ 8 planes), so it carries
/// across an LOD boundary verbatim like the distance grid.
#[derive(Clone, Debug)]
pub struct PlaneSet {
    pub planes: Vec<Plane>,
    /// Estimated nearest-neighbour spacing of the cloud (logging / provenance).
    pub spacing: f32,
    /// RANSAC inlier band actually used (`= spacing · RANSAC_THRESH_SPACING_MULT`).
    pub threshold: f32,
}

/// Estimate the cloud's nearest-neighbour spacing from a random sample. Buckets
/// points into a hash grid sized to ≈1 point per cell (cell = mean inter-point
/// spacing from the bbox volume), then for a sample of points finds the nearest
/// OTHER point within a 5×5×5 cell neighbourhood and returns the MEDIAN of those
/// nearest distances. `None` when there are < 2 finite points.
fn estimate_nn_spacing(pos: &[f32]) -> Option<f32> {
    let n = pos.len() / 3;
    if n < 2 {
        return None;
    }
    let p = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    let mut finite = 0usize;
    for i in 0..n {
        let q = p(i);
        if q.iter().all(|v| v.is_finite()) {
            finite += 1;
            for d in 0..3 {
                mn[d] = mn[d].min(q[d]);
                mx[d] = mx[d].max(q[d]);
            }
        }
    }
    if finite < 2 || !mn.iter().all(|v| v.is_finite()) {
        return None;
    }
    let vol = (0..3).map(|d| (mx[d] - mn[d]).max(1e-9)).product::<f32>();
    // Mean inter-point spacing estimate: (volume / n)^(1/3). Guarded > 0.
    let cell = (vol / finite as f32).cbrt().max(1e-9);
    let key = |q: [f32; 3]| -> (i64, i64, i64) {
        (
            ((q[0] - mn[0]) / cell) as i64,
            ((q[1] - mn[1]) / cell) as i64,
            ((q[2] - mn[2]) / cell) as i64,
        )
    };
    let mut grid: hashbrown::HashMap<(i64, i64, i64), Vec<u32>> = hashbrown::HashMap::new();
    for i in 0..n {
        let q = p(i);
        if q.iter().all(|v| v.is_finite()) {
            grid.entry(key(q)).or_default().push(i as u32);
        }
    }
    // Deterministic stride sample of up to ~3000 points.
    let sample_target = 3000usize.min(finite);
    let stride = (finite / sample_target).max(1);
    let mut dists: Vec<f32> = Vec::new();
    let mut taken = 0usize;
    for i in (0..n).step_by(stride) {
        let q = p(i);
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        let (cx, cy, cz) = key(q);
        let mut best = f32::INFINITY;
        for dx in -2..=2i64 {
            for dy in -2..=2i64 {
                for dz in -2..=2i64 {
                    if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in ids {
                            if j as usize == i {
                                continue;
                            }
                            let r = p(j as usize);
                            let d2 = (0..3).map(|d| (q[d] - r[d]).powi(2)).sum::<f32>();
                            if d2 < best {
                                best = d2;
                            }
                        }
                    }
                }
            }
        }
        if best.is_finite() {
            dists.push(best.sqrt());
        }
        taken += 1;
        if taken >= sample_target {
            break;
        }
    }
    if dists.is_empty() {
        // Sparse relative to the grid: fall back to the mean spacing estimate.
        return Some(cell);
    }
    dists.sort_by(|a, b| a.total_cmp(b));
    Some(dists[dists.len() / 2].max(1e-9))
}

/// Symmetric-3×3 eigendecomposition by cyclic Jacobi rotations (no external
/// linear-algebra dependency). Returns the three eigenvalues and the matching
/// eigenvectors as columns of `vecs`. Used to refit a plane's normal as the
/// smallest-eigenvalue eigenvector (the least-squares total-least-squares fit)
/// of the inliers' covariance, which is far more stable than the minimal-sample
/// normal. Converges in a handful of sweeps for a 3×3.
fn symmetric_eig3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[0.0f64; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..50 {
        // Largest off-diagonal magnitude.
        let (mut p, mut q, mut off) = (0usize, 1usize, 0.0f64);
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > off {
                    off = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-12 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        // Jacobi rotation angle. Annihilating a_pq requires
        // tan(2θ) = 2·apq / (aqq − app), i.e. θ = ½·atan2(2·apq, aqq − app).
        // NOTE: the x-argument is `aqq − app`, NOT `app − aqq`; the flipped sign
        // gives a θ that does NOT zero a_pq, yet the block below hardcodes it to
        // 0, corrupting the matrix (it stops being similar to the original) and
        // returning wrong eigenvectors for any non-axis-aligned covariance — i.e.
        // every real (tilted) wall.
        let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (s, c) = theta.sin_cos();
        // Symmetric Jacobi rotation A' = JᵀAJ, all updates from ORIGINAL values
        // (temporaries), so the 2×2 pivot block is never double-applied.
        // Off-block rows/cols k ∉ {p,q}: keep symmetry a[k][p] == a[p][k].
        for k in 0..3 {
            if k == p || k == q {
                continue;
            }
            let akp = a[k][p];
            let akq = a[k][q];
            let new_kp = c * akp - s * akq;
            let new_kq = s * akp + c * akq;
            a[k][p] = new_kp;
            a[p][k] = new_kp;
            a[k][q] = new_kq;
            a[q][k] = new_kq;
        }
        // 2×2 pivot block (from originals); a[p][q] is annihilated.
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        // Accumulate the eigenvectors (columns p, q of V).
        for k in 0..3 {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Build an in-plane orthonormal basis `(u, v)` for a unit normal `n`, choosing
/// the reference axis least parallel to `n` so `u` is well-conditioned.
fn plane_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let nv = glam::Vec3::from(n);
    let refv = if n[0].abs() < 0.9 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    };
    let u = nv.cross(refv).normalize_or_zero();
    let u = if u.length_squared() < 1e-12 {
        // n parallel to both fallbacks is impossible, but stay safe.
        glam::Vec3::Z
    } else {
        u
    };
    let v = nv.cross(u).normalize_or_zero();
    (u.to_array(), v.to_array())
}

/// Refit a plane to a set of inlier points by total least squares: the normal is
/// the smallest-eigenvalue eigenvector of the (mean-centred) covariance, and the
/// offset passes through the centroid. Returns `(normal, offset)`.
fn refit_plane(pos: &[f32], inliers: &[u32]) -> Option<([f32; 3], f32)> {
    if inliers.len() < 3 {
        return None;
    }
    let inv = 1.0 / inliers.len() as f64;
    let mut c = [0.0f64; 3];
    for &i in inliers {
        for d in 0..3 {
            c[d] += pos[i as usize * 3 + d] as f64;
        }
    }
    for d in &mut c {
        *d *= inv;
    }
    let mut cov = [[0.0f64; 3]; 3];
    for &i in inliers {
        let dx = [
            pos[i as usize * 3] as f64 - c[0],
            pos[i as usize * 3 + 1] as f64 - c[1],
            pos[i as usize * 3 + 2] as f64 - c[2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] += dx[a] * dx[b];
            }
        }
    }
    let (vals, vecs) = symmetric_eig3(cov);
    // Smallest-eigenvalue eigenvector = plane normal.
    let mut mi = 0usize;
    for i in 1..3 {
        if vals[i] < vals[mi] {
            mi = i;
        }
    }
    let n = glam::vec3(vecs[0][mi] as f32, vecs[1][mi] as f32, vecs[2][mi] as f32);
    let n = n.normalize_or_zero();
    if n.length_squared() < 1e-12 {
        return None;
    }
    let d = n.dot(glam::vec3(c[0] as f32, c[1] as f32, c[2] as f32));
    Some((n.to_array(), d))
}

/// Extract up to [`RANSAC_MAX_PLANES`] planes from a flat `[N*3]` cloud by
/// iterative RANSAC: find the largest plane, refit + record it with its bounded
/// extent, remove its inliers, repeat until the next plane is below the inlier
/// floor. Pure host logic (deterministic given `seed`), unit-testable without a
/// device. `None`/empty when the cloud is too small or planar structure is absent.
///
/// Inlier counting inside the minimal-sample loop is parallel (rayon); the outer
/// loop is sequential (each plane depends on the previous removal). One-time init
/// cost on ~1M points is a few seconds on the build box.
fn extract_planes(pos: &[f32], seed: u64) -> PlaneSet {
    let n = pos.len() / 3;
    let spacing = estimate_nn_spacing(pos).unwrap_or(1e-3);
    let threshold = (spacing * RANSAC_THRESH_SPACING_MULT).max(1e-6);
    let mut planes = Vec::new();
    if n < 3 {
        return PlaneSet {
            planes,
            spacing,
            threshold,
        };
    }
    let min_inliers = ((RANSAC_MIN_INLIER_FRAC * n as f32).ceil() as usize).max(3);

    // Remaining (not-yet-assigned) point indices; RANSAC operates on these.
    let mut remaining: Vec<u32> = (0..n as u32).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for _plane_i in 0..RANSAC_MAX_PLANES {
        if remaining.len() < min_inliers {
            break;
        }
        let dist = |nrm: glam::Vec3, off: f32, idx: u32| -> f32 {
            let q = glam::vec3(
                pos[idx as usize * 3],
                pos[idx as usize * 3 + 1],
                pos[idx as usize * 3 + 2],
            );
            (nrm.dot(q) - off).abs()
        };

        // Minimal-sample loop: sample 3 remaining points, fit, count inliers.
        // Parallelised across iterations; each returns (inlier_count, plane).
        let samples: Vec<(u32, u32, u32)> = (0..RANSAC_ITERS_PER_PLANE)
            .map(|_| {
                let a = rng.random_range(0..remaining.len());
                let b = rng.random_range(0..remaining.len());
                let c = rng.random_range(0..remaining.len());
                (remaining[a], remaining[b], remaining[c])
            })
            .collect();
        let best = samples
            .par_iter()
            .filter_map(|&(ia, ib, ic)| {
                if ia == ib || ib == ic || ia == ic {
                    return None;
                }
                let pa = glam::vec3(
                    pos[ia as usize * 3],
                    pos[ia as usize * 3 + 1],
                    pos[ia as usize * 3 + 2],
                );
                let pb = glam::vec3(
                    pos[ib as usize * 3],
                    pos[ib as usize * 3 + 1],
                    pos[ib as usize * 3 + 2],
                );
                let pc = glam::vec3(
                    pos[ic as usize * 3],
                    pos[ic as usize * 3 + 1],
                    pos[ic as usize * 3 + 2],
                );
                let nrm = (pb - pa).cross(pc - pa);
                if nrm.length_squared() < 1e-20 {
                    return None; // collinear sample
                }
                let nrm = nrm.normalize();
                let off = nrm.dot(pa);
                let count = remaining
                    .iter()
                    .filter(|&&idx| dist(nrm, off, idx) < threshold)
                    .count();
                Some((count, nrm.to_array(), off))
            })
            .max_by_key(|(count, _, _)| *count);

        let Some((count, nrm0, off0)) = best else {
            break;
        };
        if count < min_inliers {
            break;
        }

        // Collect the minimal-sample inliers, refit the plane to them (TLS), then
        // recollect inliers against the refined plane for a stable extent.
        let nrm0 = glam::Vec3::from(nrm0);
        let inliers0: Vec<u32> = remaining
            .iter()
            .copied()
            .filter(|&idx| dist(nrm0, off0, idx) < threshold)
            .collect();
        let (nrm, off) = refit_plane(pos, &inliers0).unwrap_or((nrm0.to_array(), off0));
        let nrm = glam::Vec3::from(nrm);
        let inliers: Vec<u32> = remaining
            .iter()
            .copied()
            .filter(|&idx| dist(nrm, off, idx) < threshold)
            .collect();
        if inliers.len() < min_inliers {
            break;
        }

        // Bounded extent: project inliers onto the in-plane basis, take min/max.
        let (u_axis, v_axis) = plane_basis(nrm.to_array());
        let uv = glam::Vec3::from(u_axis);
        let vv = glam::Vec3::from(v_axis);
        let (mut u_min, mut u_max, mut v_min, mut v_max) = (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        );
        for &idx in &inliers {
            let q = glam::vec3(
                pos[idx as usize * 3],
                pos[idx as usize * 3 + 1],
                pos[idx as usize * 3 + 2],
            );
            let uu = uv.dot(q);
            let vvv = vv.dot(q);
            u_min = u_min.min(uu);
            u_max = u_max.max(uu);
            v_min = v_min.min(vvv);
            v_max = v_max.max(vvv);
        }

        let inlier_frac = inliers.len() as f32 / n as f32;
        planes.push(Plane {
            normal: nrm.to_array(),
            offset: off,
            u_axis,
            v_axis,
            u_min,
            u_max,
            v_min,
            v_max,
            inlier_frac,
        });

        // Remove this plane's inliers and continue on the rest.
        let inlier_set: hashbrown::HashSet<u32> = inliers.into_iter().collect();
        remaining.retain(|idx| !inlier_set.contains(idx));
    }

    PlaneSet {
        planes,
        spacing,
        threshold,
    }
}

/// Read a seed cloud to the host and extract its planes by RANSAC (FIX 1 + FIX
/// 2's shared infra). Runs ONCE at init when `--plane-gate` or
/// `--plane-coplanarity-weight` is on. `None` for an empty cloud; an all-non-
/// planar cloud returns an empty `PlaneSet` (both features then no-op).
pub async fn extract_planes_from_cloud(cloud_means: Tensor<2>, seed: u64) -> Option<PlaneSet> {
    if cloud_means.dims()[0] == 0 {
        return None;
    }
    let host: Vec<f32> = detach_autodiff(cloud_means)
        .into_data_async()
        .await
        .ok()?
        .into_vec()
        .ok()?;
    Some(extract_planes(&host, seed))
}

/// Conservative distance from a voxel CENTRE to the nearest bounded plane, or
/// `+inf` if the centre projects outside every plane's extent. Mirrors the
/// point-distance path's `HALF_DIAG · vox` bias so a splat sitting on a plane
/// reads ~0 after the same quantisation correction. Used ONLY when `--plane-gate`
/// augments the distance field.
fn nearest_plane_distance(center: [f32; 3], planes: &[Plane], vox: f32) -> f32 {
    let c = glam::Vec3::from(center);
    let mut best = f32::INFINITY;
    for pl in planes {
        let nrm = glam::Vec3::from(pl.normal);
        let uu = glam::Vec3::from(pl.u_axis).dot(c);
        let vv = glam::Vec3::from(pl.v_axis).dot(c);
        if uu < pl.u_min || uu > pl.u_max || vv < pl.v_min || vv > pl.v_max {
            continue; // outside the bounded wall
        }
        let d = (nrm.dot(c) - pl.offset).abs();
        if d < best {
            best = d;
        }
    }
    if best.is_finite() {
        (best - HALF_DIAG * vox).max(0.0)
    } else {
        best
    }
}

/// Host-side product of [`build_distance_field`]: a dense truncated distance
/// field plus the geometry needed to map a world point to a flat voxel index.
struct DistanceFieldData {
    /// `[nx*ny*nz]` truncated world-space distance, flat `(i·ny + j)·nz + k`.
    field: Vec<f32>,
    origin: [f32; 3],
    vox: f32,
    dims: [usize; 3],
}

/// Build the truncated distance field on the host from a flat `[N*3]` cloud.
/// Pure (no device) so it is unit-testable. `None` when the cloud has no finite
/// point. The field stores, per voxel, a conservative distance from the voxel to
/// the nearest cloud point, clamped to `max_dist`; voxels with no cloud point
/// within the search reach keep `max_dist`.
///
/// COST (one-time, at training start): buckets the cloud by voxel, dilates the
/// occupied set by the search radius `r = ceil(reach/vox)` into a candidate set,
/// then computes each candidate's nearest-point distance in parallel (each
/// candidate is written exactly once, so there is no write race). Empty space is
/// never visited. On a house-scale LiDAR cloud this is seconds on the build box.
///
/// PLANE AUGMENTATION (FIX 1, `--plane-gate`): when `planes` is `Some`, the stored
/// value is `min(distance-to-nearest-cloud-point, distance-to-nearest-bounded-plane)`.
/// This INTERPOLATES the sparse cloud: a voxel between cloud points but inside a
/// wall's extent reads ~0 (on-surface) instead of `max_dist`, so genuine wall
/// splats in the gaps are no longer penalised, while a mid-air voxel (far from
/// every cloud point AND outside every plane's extent) still reads `max_dist`.
/// The plane pass sweeps ALL voxels (parallel, one write each), because a
/// plane-covered voxel need not be near any occupied voxel — that is the whole
/// point of the fill. `None` reproduces the exact point-only field (byte-inert).
fn build_distance_field(
    pos: &[f32],
    margin: f32,
    softness: f32,
    planes: Option<&[Plane]>,
) -> Option<DistanceFieldData> {
    let n = pos.len() / 3;
    if n == 0 {
        return None;
    }
    let soft = softness.max(1e-6);
    let margin = margin.max(0.0);
    let reach = margin + REACH_SOFT * soft; // accurate-distance radius
    let max_dist = margin + FAR_SPAN * soft; // far / truncation value

    // Finite bounding box over the cloud.
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        let p = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
        if p.iter().all(|v| v.is_finite()) {
            for d in 0..3 {
                mn[d] = mn[d].min(p[d]);
                mx[d] = mx[d].max(p[d]);
            }
        }
    }
    if !mn.iter().all(|v| v.is_finite()) {
        return None; // no finite point in the cloud
    }

    // Pad so a shell of `reach` around the cloud has real voxels.
    let pad = reach;
    let mut vox = (margin / VOX_PER_MARGIN).max(1e-6);
    let dims_for = |vox: f32| -> [usize; 3] {
        let mut d = [0usize; 3];
        for a in 0..3 {
            let ext = (mx[a] - mn[a]) + 2.0 * pad;
            d[a] = ((ext / vox).ceil() as usize).max(1) + 1;
        }
        d
    };
    let mut dims = dims_for(vox);
    // Grow the voxel until the grid fits the memory cap (big scenes get coarser).
    while dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]) > MAX_VOXELS {
        vox *= 1.5;
        dims = dims_for(vox);
    }
    let origin = [mn[0] - pad, mn[1] - pad, mn[2] - pad];
    let [nx, ny, nz] = dims;
    let num_vox = nx * ny * nz;
    let flat = |i: usize, j: usize, k: usize| -> usize { (i * ny + j) * nz + k };
    let vkey = |p: [f32; 3]| -> [i64; 3] {
        [
            ((p[0] - origin[0]) / vox).floor() as i64,
            ((p[1] - origin[1]) / vox).floor() as i64,
            ((p[2] - origin[2]) / vox).floor() as i64,
        ]
    };

    // Point buckets keyed by voxel.
    let mut buckets: hashbrown::HashMap<[i64; 3], Vec<u32>> = hashbrown::HashMap::new();
    for i in 0..n {
        let p = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
        if p.iter().all(|v| v.is_finite()) {
            buckets.entry(vkey(p)).or_default().push(i as u32);
        }
    }

    // Candidate voxels = occupied voxels dilated by the search radius (in voxels).
    let r = ((reach / vox).ceil() as i64).max(1);
    let mut cand: hashbrown::HashSet<[i64; 3]> = hashbrown::HashSet::new();
    for key in buckets.keys() {
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    let c = [key[0] + dx, key[1] + dy, key[2] + dz];
                    if c[0] >= 0
                        && c[1] >= 0
                        && c[2] >= 0
                        && (c[0] as usize) < nx
                        && (c[1] as usize) < ny
                        && (c[2] as usize) < nz
                    {
                        cand.insert(c);
                    }
                }
            }
        }
    }

    // Per-candidate nearest-point distance (parallel; each written once → race
    // free). Searching ±r from a candidate that is itself within r of an occupied
    // voxel reaches every point within `reach`, so distances up to `reach` are
    // exact; farther candidates saturate to `max_dist`.
    let cand: Vec<[i64; 3]> = cand.into_iter().collect();
    let updates: Vec<(usize, f32)> = cand
        .par_iter()
        .map(|&c| {
            let center = [
                origin[0] + (c[0] as f32 + 0.5) * vox,
                origin[1] + (c[1] as f32 + 0.5) * vox,
                origin[2] + (c[2] as f32 + 0.5) * vox,
            ];
            let mut best2 = f32::INFINITY;
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if let Some(ids) = buckets.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            for &pi in ids {
                                let pi = pi as usize;
                                let d2 = (center[0] - pos[pi * 3]).powi(2)
                                    + (center[1] - pos[pi * 3 + 1]).powi(2)
                                    + (center[2] - pos[pi * 3 + 2]).powi(2);
                                if d2 < best2 {
                                    best2 = d2;
                                }
                            }
                        }
                    }
                }
            }
            // Conservative bias (subtract half the voxel diagonal), clamped to
            // [0, max_dist]: report the SMALLEST distance the queried Gaussian
            // could have given voxel quantisation, so a surface splat reads ~0.
            let dist = (best2.sqrt() - HALF_DIAG * vox).clamp(0.0, max_dist);
            (flat(c[0] as usize, c[1] as usize, c[2] as usize), dist)
        })
        .collect();

    let mut field = vec![max_dist; num_vox];
    for (idx, dist) in updates {
        field[idx] = dist;
    }

    // FIX 1: augment with distance-to-nearest-bounded-plane (min with the
    // point-distance already stored). Sweeps every voxel in parallel because a
    // plane-covered voxel in a sparse gap need not be a point-distance candidate.
    if let Some(planes) = planes
        && !planes.is_empty()
    {
        field.par_iter_mut().enumerate().for_each(|(idx, cell)| {
            let i = idx / (ny * nz);
            let rem = idx % (ny * nz);
            let j = rem / nz;
            let k = rem % nz;
            let center = [
                origin[0] + (i as f32 + 0.5) * vox,
                origin[1] + (j as f32 + 0.5) * vox,
                origin[2] + (k as f32 + 0.5) * vox,
            ];
            let pd = nearest_plane_distance(center, planes, vox).min(max_dist);
            if pd < *cell {
                *cell = pd;
            }
        });
    }

    Some(DistanceFieldData {
        field,
        origin,
        vox,
        dims,
    })
}

/// A static, view-independent distance-to-cloud lookup, built ONCE at training
/// start from the seed/LiDAR point cloud (the measured surface the run is seeded
/// from). Holds a dense truncated distance field as a device tensor plus the
/// geometry to map a Gaussian centre to a flat voxel index; the per-step query is
/// a single gather. The cloud does not move, so this is never rebuilt.
pub struct CloudDistanceGrid {
    /// `[nx*ny*nz]` truncated world-distance field, on the trainer device. The
    /// per-voxel value already saturates at the far/truncation distance, so the
    /// query needs only the field + geometry, not the scalar `max_dist`.
    field: Tensor<1>,
    origin: [f32; 3],
    vox: f32,
    dims: [usize; 3],
}

impl CloudDistanceGrid {
    /// Read the cloud to the host, build the distance field, upload it to
    /// `device`. `None` when the cloud has no finite point (e.g. a random-init
    /// run with no seed) — the caller then warns once and no-ops the reg.
    ///
    /// `planes` (FIX 1, `--plane-gate`) augments the stored field with
    /// distance-to-nearest-bounded-plane; pass `None` for the exact point-only
    /// field (identical to the pre-plane behaviour).
    pub async fn build(
        cloud_means: Tensor<2>,
        margin: f32,
        softness: f32,
        planes: Option<&[Plane]>,
        device: &Device,
    ) -> Option<Self> {
        if cloud_means.dims()[0] == 0 {
            return None;
        }
        // `detach_autodiff` is a passthrough on an already-inner tensor and drops
        // the graph on an autodiff one; either way we just want the raw values.
        let host: Vec<f32> = detach_autodiff(cloud_means)
            .into_data_async()
            .await
            .ok()?
            .into_vec()
            .ok()?;
        let data = build_distance_field(&host, margin, softness, planes)?;
        let num = data.field.len();
        let field = Tensor::<1>::from_data(TensorData::new(data.field, [num]), device);
        Some(Self {
            field,
            origin: data.origin,
            vox: data.vox,
            dims: data.dims,
        })
    }

    /// Gather each Gaussian centre's 3D distance to the nearest CLOUD POINT from
    /// the static field, for the hard cloud-distance prune (`--cloud-prune`).
    /// Returns the per-Gaussian DISTANCES (`Tensor<1>` f32) — NOT a Bool. The
    /// caller thresholds `d > --cloud-prune-dist` to a mask AT THE UNION SITE
    /// using the SAME `from_data(.., device).greater_elem(..)` idiom every other
    /// `refine_for_phase` prune mask resolves to, so the mask is the identical
    /// Bool kind as `prune_mask` and `bool_or` cannot trip a Bool(U32)/Bool(Native)
    /// mismatch. Out-of-grid and non-finite rows are already forced to `+inf`
    /// here, so a plain `> dist` comparison prunes them.
    ///
    /// POINT-ONLY: the caller MUST have built this grid with `planes = None`. The
    /// prune wants distance-to-nearest-cloud-POINT, NOT the plane-augmented
    /// distance the opacity regularizer can use under `--plane-gate` — a plane
    /// INTERPOLATES the sparse cloud and would read a wall-perpendicular floater
    /// (far from every cloud point but near the wall plane) as on-surface,
    /// shielding exactly the floaters this prune must delete. `cloud_prune_grid`
    /// is therefore a SEPARATE, always-point-only grid from `opacity_reg_grid`.
    ///
    /// LIVE means: this gathers the CURRENT `means` passed in against the STATIC
    /// distance field (distance-to-static-cloud). The cloud never moves, so the
    /// field is fixed; gathering live means each refine cycle is both correct and
    /// cheap — a Gaussian that DRIFTED off the surface since the last cycle reads
    /// its new, larger distance and is caught now. No rebuild is needed.
    ///
    /// Pure inner-backend O(1) gather. Reuses the a8e88f47 INTEGER flat-index
    /// (never f32: a grid can exceed 2^24 voxels, and an f32 `i·(ny·nz)+…` would
    /// round a large index onto an ADJACENT voxel). `means` are detached (the
    /// prune is a boolean decision, so no gradient is needed). A divergent
    /// (NaN/±inf) centre is forced to `+inf` so it always reads as a floater
    /// (it is separately caught by the trainer's non-finite prune too).
    pub fn gather_prune_distances(&self, means: Tensor<2>) -> Tensor<1> {
        // PRUNE path = NO gradient, so force BOTH the field and the means onto the
        // SAME INNER backend/device before any op. `detach_autodiff` drops an
        // autodiff-kind tensor to the inner backend (passthrough if already
        // inner). This matters because the field may have been built on an
        // autodiff-MARKED device at init (the grid `build` uses whatever device
        // train_stream passes), while the prune means arrive inner-kind (splats
        // are `.valid()` in `refine_for_phase`). Gathering an inner index against
        // an autodiff-kind field — or subtracting an inner `origin` from an
        // autodiff `means` — trips the cross-backend panic that bit the opacity
        // regularizer. Aligning here on the field's INNER device fixes it. Do NOT
        // `lift_to_autodiff`: this is a boolean mask, not a loss term.
        let field = detach_autodiff(self.field.clone());
        let device = field.device();
        let means = detach_autodiff(means).to_device(&device);
        let [nx, ny, nz] = self.dims;
        let num_vox = nx * ny * nz;

        // Voxel coord = floor((mu - origin) / vox), clamped per axis into the grid.
        let origin = Tensor::<2>::from_data(
            TensorData::new(vec![self.origin[0], self.origin[1], self.origin[2]], [1, 3]),
            &device,
        );
        let coord = means.sub(origin).div_scalar(self.vox.max(1e-6)).floor();

        // A diverged (NaN/±inf) mean gives a non-finite coord on at least one
        // axis. Flag such rows BEFORE the clamp turns ±inf into a finite in-range
        // index, so they are forced to `+inf` (far → prune) below rather than
        // silently reading whatever voxel a garbage index lands on. Mirrors
        // `depth_opacity_reg_loss`'s BUG-2 guard.
        let nonfinite = coord
            .clone()
            .is_finite()
            .float()
            .sum_dim(1) // [N, 1]: 3.0 iff all three axes are finite
            .greater_elem(2.5)
            .bool_not(); // [N, 1] Bool, true = divergent row

        let cxf = coord.clone().slice(burn::tensor::s![.., 0..1]);
        let cyf = coord.clone().slice(burn::tensor::s![.., 1..2]);
        let czf = coord.slice(burn::tensor::s![.., 2..3]);

        // Explicit UNCLAMPED in-grid mask (mirrors `project_depth_residual`'s
        // `in_frame`): a centre whose voxel coord falls outside `[0, dim-1]` on ANY
        // axis is OUTSIDE the padded grid → genuinely far from the cloud → forced
        // to `+inf` (pruned), NOT read from the clamped boundary voxel. The clamp
        // alone was a hard-delete hazard on a LARGE-extent cloud with a small
        // `--cloud-prune-dist`: the `vox *= 1.5` coarsening loop (MAX_VOXELS cap)
        // can push `vox` large enough that the boundary voxel's stored distance
        // drops below `dist`, so a wildly out-of-grid floater aligned with real
        // cloud on the other two axes would read as near-cloud and be spared. The
        // unclamped test removes that vox-size dependency entirely — out-of-grid is
        // always a floater. Strict comparators only (module style): `x >= lo` is
        // `!(x < lo)`, `x <= hi` is `!(x > hi)`. (NaN rows read false here but are
        // already covered by `nonfinite`.)
        let in_axis = |c: Tensor<2>, hi: f32| -> Tensor<2, Bool> {
            c.clone()
                .lower_elem(0.0)
                .bool_not()
                .bool_and(c.greater_elem(hi).bool_not())
        };
        let in_grid = in_axis(cxf.clone(), (nx - 1) as f32)
            .bool_and(in_axis(cyf.clone(), (ny - 1) as f32))
            .bool_and(in_axis(czf.clone(), (nz - 1) as f32)); // [N,1]
        // Force `+inf` on divergent OR out-of-grid rows.
        let force_far = nonfinite.clone().bool_or(in_grid.bool_not()); // [N,1]

        let cx = cxf
            .clamp(0.0, (nx - 1) as f32)
            .mask_fill(nonfinite.clone(), 0.0)
            .int();
        let cy = cyf
            .clamp(0.0, (ny - 1) as f32)
            .mask_fill(nonfinite.clone(), 0.0)
            .int();
        let cz = czf
            .clamp(0.0, (nz - 1) as f32)
            .mask_fill(nonfinite.clone(), 0.0)
            .int();

        // Flat index `(i·ny + j)·nz + k` in INTEGER arithmetic (a8e88f47), exactly
        // matching the host `usize` index in `build_distance_field`.
        let flat = cx
            .mul_scalar((ny * nz) as i64)
            .add(cy.mul_scalar(nz as i64))
            .add(cz)
            .clamp(0i64, (num_vox - 1) as i64)
            .squeeze_dim::<1>(1); // [N] Int

        // Distance-to-nearest-cloud-point, gathered from the static field. Both
        // `field` (detached above) and `flat` are inner-kind on `device`.
        let d = field.select(0, flat); // [N]
        // Force divergent + out-of-grid rows to `+inf` (always > any threshold →
        // pruned); the clamped index for those rows is valid but discarded here.
        d.mask_fill(force_far.squeeze_dim::<1>(1), f32::INFINITY)
    }
}

/// Whether the depth-coupled opacity regularizer contributes a term this step:
/// its weight is positive AND the global iteration has reached `start_iter`.
/// Pulled out so the start-iter gate is unit-testable without a GPU or trainer.
/// Before `start_iter` the term is skipped entirely (no projection, no cost), so
/// densification can finish backfilling opacity-faded regions first.
pub fn opacity_reg_active(global_iter: u32, start_iter: u32, weight: f32) -> bool {
    weight > 0.0 && global_iter >= start_iter
}

/// Depth-coupled opacity regularizer — the SMOOTH, differentiable floater fade,
/// now gated on 3D distance-to-cloud (see the module note above). Instead of
/// deleting a floating Gaussian (which orphans its load-bearing colour and leaves
/// a black halo), this adds a per-step loss whose ONLY gradient path is the
/// activated opacity, so the optimizer fades far-from-cloud Gaussians out SMOOTHLY
/// and their colour redistributes into on-surface Gaussians before they vanish.
///
/// For each Gaussian centre `mu_i` it gathers the DETACHED distance-to-cloud
/// `d_i` from the static grid, then `p_i = σ((d_i - margin) / softness)`: ~0 for
/// `d_i ≤ margin` (on / near the surface), 0.5 at `d_i = margin`, ~1 for
/// `d_i ≫ margin` (a floater in empty space). The term is
/// `λ · mean_i(p_i · σ(raw_opacity_i))`.
///
/// BACKEND: everything runs on the opacity leaf's autodiff backend, built on
/// `raw_opacity.device()`. `means.detach()` stops the position gradient while
/// STAYING on the autodiff backend (unlike `detach_autodiff`, which drops to the
/// inner backend and then panics when multiplied against the autodiff opacity),
/// and the static field is `lift_to_autodiff`-ed into the graph as a CONSTANT.
/// So `d_i` / `p_i` are autodiff-kind constants (no grad), `alpha` is autodiff
/// WITH grad, and `p · alpha` is a clean autodiff×autodiff product.
///
/// `raw_opacity` is the LIVE opacity leaf (`splats.raw_opacities.val()`), so the
/// only gradient is `∂L/∂raw_opacity_i = λ · p_i · σ'(raw_opacity_i)` — it drives
/// far-from-cloud Gaussians' opacity toward 0 and touches nothing else. The mean
/// is over ALL Gaussians: on-surface ones carry `p_i ~ 0`, so they sit in the
/// denominator but contribute ~0 to the numerator.
pub fn depth_opacity_reg_loss(
    raw_opacity: Tensor<1>,
    means: Tensor<2>,
    grid: &CloudDistanceGrid,
    margin: f32,
    softness: f32,
) -> Tensor<1> {
    let device = raw_opacity.device();
    // Grad-stopped positions, STAYING on the autodiff backend (see the doc above).
    let means = means.detach();
    let [nx, ny, nz] = grid.dims;
    let num_vox = nx * ny * nz;

    // Voxel coord = floor((mu - origin) / vox), clamped per axis into the grid.
    let origin = Tensor::<2>::from_data(
        TensorData::new(vec![grid.origin[0], grid.origin[1], grid.origin[2]], [1, 3]),
        &device,
    );
    let coord = means.sub(origin).div_scalar(grid.vox.max(1e-6)).floor();

    // BUG-2 guard: a diverged (NaN/±inf) mean gives a non-finite coord on at least
    // one axis. Do NOT route such a row through a real voxel index — the far value
    // of any particular voxel (e.g. index 0) is an implementation detail, so a
    // divergent Gaussian could silently read as on-surface. Instead flag the row
    // here (before the clamp turns ±inf into a finite in-range index) and FORCE
    // `p = 1` for it below, so divergent Gaussians are always fully penalized.
    let nonfinite = coord
        .clone()
        .is_finite()
        .float()
        .sum_dim(1) // [N, 1]: 3.0 iff all three axes are finite
        .greater_elem(2.5)
        .bool_not(); // [N, 1] Bool, true = divergent row

    // Per-axis voxel coord, clamped into the grid; divergent rows are zeroed so
    // the Int cast is well defined (their `p` is overridden regardless).
    let cx = coord
        .clone()
        .slice(burn::tensor::s![.., 0..1])
        .clamp(0.0, (nx - 1) as f32)
        .mask_fill(nonfinite.clone(), 0.0)
        .int();
    let cy = coord
        .clone()
        .slice(burn::tensor::s![.., 1..2])
        .clamp(0.0, (ny - 1) as f32)
        .mask_fill(nonfinite.clone(), 0.0)
        .int();
    let cz = coord
        .slice(burn::tensor::s![.., 2..3])
        .clamp(0.0, (nz - 1) as f32)
        .mask_fill(nonfinite.clone(), 0.0)
        .int();

    // BUG-1 fix: build the flat index `(i·ny + j)·nz + k` in INTEGER arithmetic,
    // matching the exact host `usize` index used in `build_distance_field`. f32 is
    // exact only to 2^24, but a grid can ship up to MAX_VOXELS (> 2^24) voxels, so
    // an f32 `ci·(ny·nz)+…` would round ~half the Gaussians onto an ADJACENT voxel
    // and gather the wrong distance. Int mul/add is exact (the largest term is
    // < num_vox ≤ MAX_VOXELS < 2^31). Clamp defensively, then gather.
    let flat = cx
        .mul_scalar((ny * nz) as i64)
        .add(cy.mul_scalar(nz as i64))
        .add(cz)
        .clamp(0i64, (num_vox - 1) as i64)
        .squeeze_dim::<1>(1); // [N] Int

    // Detached per-Gaussian distance-to-cloud, gathered from the static field.
    // Lift the field into the autodiff graph as a CONSTANT so it shares the
    // opacity leaf's backend for the multiply, with no gradient path of its own.
    let d = lift_to_autodiff(grid.field.clone()).select(0, flat);

    // p_i ramps UP with distance from the cloud; detached (no gradient), so the
    // ONLY graph-connected factor below is the activated opacity of the live leaf.
    let p = sigmoid(d.sub_scalar(margin).div_scalar(softness.max(1e-6)));
    // BUG-2: force full penalty on divergent (non-finite-mean) rows.
    let p = p.mask_fill(nonfinite.squeeze_dim::<1>(1), 1.0);
    let alpha = sigmoid(raw_opacity);
    (p * alpha).mean()
}

// ---- FIX 2: co-planarity geometry constraint ---------------------------------
//
// A STRONGER, separate opt-in (`--plane-coplanarity-weight`). Unlike the opacity
// gate (which only fades far-from-cloud splats via an opacity gradient), this is a
// real GEOMETRY constraint with a gradient on POSITION and SCALE (and, as a
// byproduct of the projected-variance form, on rotation): for each Gaussian
// ASSIGNED to a plane it (a) pulls the centre onto the plane and (b) flattens the
// Gaussian against the plane. This directly removes the photometric rank
// deficiency on a featureless wall.
//
// ASSIGNMENT is detached and recomputed each step: a Gaussian is assigned to the
// nearest plane whose perpendicular distance is `< assign_dist` AND whose bounded
// extent contains the Gaussian's projection. The plane params `(n, d)` are
// detached constants; only `means` / `scales` / `rotations` carry gradient.

/// Co-planarity loss (FIX 2). `means` `[N,3]`, `rotations` `[N,4]` (w,x,y,z), and
/// `scales` `[N,3]` (world std-devs) are the LIVE autodiff leaves. Returns `None`
/// when there are no planes (so the caller adds nothing). The term is
///   `mean over assigned i of [ (n_i·mu_i − d_i)² + Σ_k (s_ik · w_ik)² ]`
/// where `w_i = R_iᵀ n_i` is the plane normal expressed in the Gaussian's local
/// frame, so `Σ_k (s_ik w_ik)²` is exactly the Gaussian's variance ALONG the
/// plane normal — driving it to zero flattens the Gaussian onto the plane.
///
/// BACKEND: everything is built on `means.device()` (autodiff). The plane
/// constants are `from_data` on that device (autodiff-kind constants, no grad),
/// and assignment uses `means.detach()` (stays autodiff-kind, no position grad
/// through the mask) — mirroring the opacity gate's discipline. The live product
/// `(proj − d)²` and the projected variance keep `means` / `scales` / `rotations`
/// on the graph, so the gradient reaches geometry without a backend cross-kind
/// panic.
pub fn plane_coplanarity_loss(
    means: Tensor<2>,
    rotations: Tensor<2>,
    scales: Tensor<2>,
    planes: &PlaneSet,
    assign_dist: f32,
    device: &Device,
) -> Option<Tensor<1>> {
    let p = planes.planes.len();
    if p == 0 {
        return None;
    }
    let n = means.dims()[0];
    if n == 0 {
        return None;
    }

    // Plane constants as device tensors (autodiff-kind, no grad): normals [P,3],
    // offsets [P], and the in-plane basis + bounds for the extent test.
    let mut normals = Vec::with_capacity(p * 3);
    let mut offsets = Vec::with_capacity(p);
    let mut u_ax = Vec::with_capacity(p * 3);
    let mut v_ax = Vec::with_capacity(p * 3);
    let mut u_lo = Vec::with_capacity(p);
    let mut u_hi = Vec::with_capacity(p);
    let mut v_lo = Vec::with_capacity(p);
    let mut v_hi = Vec::with_capacity(p);
    for pl in &planes.planes {
        normals.extend_from_slice(&pl.normal);
        offsets.push(pl.offset);
        u_ax.extend_from_slice(&pl.u_axis);
        v_ax.extend_from_slice(&pl.v_axis);
        u_lo.push(pl.u_min);
        u_hi.push(pl.u_max);
        v_lo.push(pl.v_min);
        v_hi.push(pl.v_max);
    }
    let normals_t = Tensor::<2>::from_data(TensorData::new(normals, [p, 3]), device);
    let offsets_t = Tensor::<1>::from_data(TensorData::new(offsets.clone(), [p]), device);
    let u_ax_t = Tensor::<2>::from_data(TensorData::new(u_ax, [p, 3]), device);
    let v_ax_t = Tensor::<2>::from_data(TensorData::new(v_ax, [p, 3]), device);

    // -- Detached assignment. For each plane, the per-Gaussian perpendicular
    // distance, masked to +LARGE outside the assign band or the bounded extent.
    // Stacked to [N, P]; the argmin/min over P give the nearest valid plane.
    const LARGE: f32 = 1e30;
    let md = means.clone().detach(); // [N,3], autodiff-kind, no grad
    let mut cand_cols: Vec<Tensor<2>> = Vec::with_capacity(p);
    for pi in 0..p {
        let nrm = normals_t.clone().slice(burn::tensor::s![pi..pi + 1, ..]); // [1,3]
        let off = offsets.get(pi).copied().unwrap_or(0.0);
        // signed = md·n − d ; perp = |signed|  → [N,1]
        let signed = (md.clone() * nrm.clone()).sum_dim(1).sub_scalar(off);
        let perp = signed.abs();
        // In-plane projections and the extent test.
        let uu = (md.clone() * u_ax_t.clone().slice(burn::tensor::s![pi..pi + 1, ..])).sum_dim(1); // [N,1]
        let vv = (md.clone() * v_ax_t.clone().slice(burn::tensor::s![pi..pi + 1, ..])).sum_dim(1);
        // Bounded-extent test with strict comparators only (matching this
        // module's style): `x >= lo` is `!(x < lo)`, `x <= hi` is `!(x > hi)`.
        let ge_ulo = uu.clone().lower_elem(u_lo[pi]).bool_not();
        let le_uhi = uu.greater_elem(u_hi[pi]).bool_not();
        let ge_vlo = vv.clone().lower_elem(v_lo[pi]).bool_not();
        let le_vhi = vv.greater_elem(v_hi[pi]).bool_not();
        let inside = ge_ulo.bool_and(le_uhi).bool_and(ge_vlo).bool_and(le_vhi);
        let near = perp.clone().lower_elem(assign_dist);
        let valid = inside.bool_and(near);
        // Distance where valid, +LARGE elsewhere.
        let col = perp.mask_fill(valid.bool_not(), LARGE);
        cand_cols.push(col);
    }
    let cand = Tensor::cat(cand_cols, 1); // [N, P]
    let best_dist = cand.clone().min_dim(1); // [N,1]
    let best_plane = cand.argmin(1); // [N,1] Int
    // Assigned iff some plane was within band+extent (min < LARGE).
    let assigned = best_dist.lower_elem(LARGE * 0.5); // [N,1] Bool
    let assigned_f = assigned.clone().float(); // [N,1]
    let best_idx = best_plane.squeeze_dim::<1>(1); // [N] Int

    // Per-Gaussian assigned plane params (detached constants).
    let n_i = normals_t.select(0, best_idx.clone()); // [N,3]
    let d_i = offsets_t.select(0, best_idx); // [N]

    // NaN-guard (the same class the sibling `depth_opacity_reg_loss` fixed once).
    // A divergent (NaN/±inf) mean, scale, or quaternion would make a row's term
    // non-finite. Assignment on the RAW detached means already EXCLUDES such a row
    // (`perp` compares false → `assigned_f1[i] = 0`), but masking the term only in
    // the FORWARD pass is not enough: the row's derivative is still NaN, and
    // `0 · NaN = NaN` poisons the BACKWARD too — and unlike the opacity gate the
    // means here are a LIVE leaf, so that NaN reaches means/scales/rotations for
    // EVERY Gaussian. The fix sanitizes the LIVE inputs BEFORE the differentiable
    // ops: `mask_fill` replaces a non-finite entry with a constant 0 (grad 0 there,
    // identity elsewhere), so finite rows are unaffected and a divergent row
    // contributes a finite 0 that assignment then zeroes out anyway.
    let means = means
        .clone()
        .mask_fill(means.clone().is_finite().bool_not(), 0.0); // [N,3]
    let scales = scales
        .clone()
        .mask_fill(scales.clone().is_finite().bool_not(), 0.0); // [N,3]
    let rotations = rotations
        .clone()
        .mask_fill(rotations.clone().is_finite().bool_not(), 0.0); // [N,4]

    // -- Position pull: (n_i·mu_i − d_i)² with LIVE (sanitized) means.
    let proj = (means * n_i.clone()).sum_dim(1).squeeze_dim::<1>(1); // [N]
    let pos_term = (proj - d_i).powi_scalar(2); // [N]

    // -- Flatten: variance of the Gaussian along the plane normal. Express the
    // (unit) normal in the Gaussian's local frame via the conjugate of the
    // normalised quaternion (Rᵀ n), then Σ_k (s_k · w_k)² = nᵀ R S² Rᵀ n. A
    // sanitized (zeroed) quaternion has norm 0 → clamp_min keeps the divide finite
    // → w_local = 0 → flatten 0 for that (already-excluded) row.
    let qnorm = rotations
        .clone()
        .powi_scalar(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12); // [N,1]
    let qunit = rotations.div(qnorm); // [N,4] unit
    // Conjugate = (w, −x, −y, −z): flips the vector part.
    let sign = Tensor::<2>::from_data(
        TensorData::new(vec![1.0f32, -1.0, -1.0, -1.0], [1, 4]),
        device,
    );
    let qconj = qunit * sign; // [N,4]
    let w_local = crate::quat_vec::quaternion_vec_multiply(qconj, n_i); // [N,3] = Rᵀ n
    let flatten_term = (scales * w_local)
        .powi_scalar(2)
        .sum_dim(1)
        .squeeze_dim::<1>(1); // [N]

    // -- Masked mean over the assigned Gaussians (all terms finite by now).
    let assigned_f1 = assigned_f.squeeze_dim::<1>(1); // [N]
    let per = (pos_term + flatten_term) * assigned_f1.clone(); // [N]
    let count = assigned_f1.sum().clamp_min(1.0); // scalar-as-[1]
    Some(per.sum() / count)
}

/// Thresholds + caps for the cleanup pass, snapshotted from `TrainConfig` so the
/// pure selection logic below has no dependency on the config crate.
pub struct TidiPruneParams {
    /// Whether the four-signal PHOTOMETRIC candidate path is active
    /// (`--tidi-prune`). When off, no Gaussian becomes a photometric candidate,
    /// so only the depth path (if on) can contribute prunes.
    pub photometric: bool,
    pub vis_threshold: f32,
    pub opacity_threshold: f32,
    pub importance_threshold: f32,
    pub grad_threshold: f32,
    pub warmup_steps: i32,
    pub guard_sh_quantile: f32,
    pub guard_thin_quantile: f32,
    pub guard_aniso_quantile: f32,
    pub guard_color_var_quantile: f32,
    pub knn_k: usize,
    pub local_cap_frac: f32,
    pub global_cap_frac: f32,
    /// Whether the SEPARATE depth/LiDAR-residual candidate path is active
    /// (`--tidi-depth-prune`).
    pub depth_prune: bool,
    /// Prune a Gaussian when it floated in front of the surface in at least this
    /// fraction of the views that had a valid return behind it.
    pub depth_float_frac: f32,
    /// Safety gate: minimum number of views with a valid return behind a
    /// Gaussian before the depth path may prune it (unscanned regions exempt).
    pub depth_min_valid_views: f32,
    /// Per-cycle global cap for the depth path, as a fraction of ALL Gaussians.
    /// Looser than `global_cap_frac` because the LiDAR-gated signal is trusted.
    pub depth_cap_frac: f32,
}

/// Host-side quantile of a slice (already the filtered subset). `q` in `[0,1]`;
/// returns `None` for an empty input.
fn quantile(mut v: Vec<f32>, q: f32) -> Option<f32> {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let idx = ((q.clamp(0.0, 1.0) * (v.len() - 1) as f32).round() as usize).min(v.len() - 1);
    Some(v[idx])
}

/// The pure TIDI selection. Two INDEPENDENT candidate paths, unioned at the end:
///   * PHOTOMETRIC (`--tidi-prune`): the four-signal AND (vis ∧ opacity ∧ ω ∧
///     grad) → detail guards → isolation k-NN with the local + global caps.
///   * DEPTH (`--tidi-depth-prune`): a Gaussian that floats in front of the
///     measured LiDAR/depth surface in ≥ `depth_float_frac` of the views that
///     had a valid return behind it (≥ `depth_min_valid_views` of them) → the
///     SAME detail guards → its own looser global cap.
/// The depth path is deliberately NOT AND-gated with the photometric signals:
/// equilibrium wall-haze is photometrically valid, so it would never survive the
/// four-signal AND, yet it is exactly what the depth residual catches. Split out
/// so it is unit testable without any GPU state.
fn select_prune_indices(cfg: &TidiPruneParams, s: &HostSignals) -> Vec<u32> {
    let n = s.n;
    if n == 0 {
        return Vec::new();
    }

    // -- Photometric candidate: FAILS ALL FOUR signal thresholds AND is past its
    // per-Gaussian warmup. This is an AND, NOT the OR that Brush's five geometric
    // culls use — a floater is only a candidate when every signal agrees it is
    // idle. (paper §III-B, Table II). Gated on `cfg.photometric` so a
    // depth-only run (`--tidi-depth-prune` without `--tidi-prune`) forms no
    // photometric candidates.
    let mut candidate = vec![false; n];
    if cfg.photometric {
        for i in 0..n {
            let past_warmup = s.age[i] >= cfg.warmup_steps;
            let fail_vis = s.vis[i] <= cfg.vis_threshold;
            let fail_alpha = s.opacity[i] <= cfg.opacity_threshold;
            let fail_omega = s.sigma_w[i] <= cfg.importance_threshold;
            let fail_grad = s.grad_ema[i] <= cfg.grad_threshold;
            candidate[i] = past_warmup && fail_vis && fail_alpha && fail_omega && fail_grad;
        }
    }

    // -- Depth candidate (standalone): enough views saw a valid return behind
    // this Gaussian AND it floated in front of that surface in a high enough
    // fraction of them. The `min_valid_views` gate is the safety net — a
    // Gaussian in an unscanned region (too few valid returns) is NEVER a depth
    // candidate. A split resets these counters, so a fresh child cannot reach
    // `min_valid_views` until it has been re-observed, giving the same
    // protection the photometric path gets from warmup.
    let mut depth_candidate = vec![false; n];
    if cfg.depth_prune {
        for i in 0..n {
            let valid = s.valid_accum[i];
            if valid >= cfg.depth_min_valid_views {
                let float_frac = s.float_accum[i] / valid.max(1.0);
                depth_candidate[i] = float_frac >= cfg.depth_float_frac;
            }
        }
    }

    // Combined candidate set: the STABLE (guard-reference) distribution and the
    // guards themselves apply to BOTH paths, so a thin/specular structure that
    // reads as floating near a depth discontinuity is still protected.
    let any_candidate: Vec<bool> = (0..n).map(|i| candidate[i] || depth_candidate[i]).collect();

    // -- Adaptive detail guards. Thresholds are a quantile of the STABLE
    // (non-candidate) distribution, recomputed each cycle — the paper gives no
    // fixed numbers, so the flags set the quantile, not the value. A candidate
    // is exempted (kept) if it passes ANY guard (OR): unusually high non-DC SH
    // energy (specular / view-dependent detail), an unusually thin smallest axis
    // (a thin structure), unusually high anisotropy `s₃/s₁` (an elongated sheet /
    // needle), or high local colour variance.
    let stable_of = |vals: &[f32]| -> Vec<f32> {
        (0..n)
            .filter(|&i| !any_candidate[i])
            .map(|i| vals[i])
            .collect()
    };

    // Guard 1: SH high-frequency energy, exempt at/above the high quantile.
    let tau_h = (cfg.guard_sh_quantile > 0.0)
        .then(|| quantile(stable_of(&s.sh_hf), cfg.guard_sh_quantile))
        .flatten();
    // Guard 2 (thinness): exempt at/below the low quantile of s₁ (thin sliver).
    let tau_s = (cfg.guard_thin_quantile > 0.0)
        .then(|| quantile(stable_of(&s.min_scale), cfg.guard_thin_quantile))
        .flatten();
    // Guard 3 (anisotropy): exempt at/above the high quantile of s₃/s₁ (an
    // elongated structure the thinness guard misses when s₁ itself is not small,
    // e.g. a wide, flat sheet). GAP-4 fix.
    let tau_a = (cfg.guard_aniso_quantile > 0.0)
        .then(|| quantile(stable_of(&s.aniso), cfg.guard_aniso_quantile))
        .flatten();

    // Guard 2 (optional, default off): local colour variance among a candidate's
    // k-NN. NOTE: deriving τ_V from the stable set would need a full-set k-NN;
    // to avoid that cost this computes V only for candidates and thresholds
    // against the CANDIDATE distribution — a documented approximation, hence
    // off by default.
    let color_var: Option<(Vec<f32>, f32)> = if cfg.guard_color_var_quantile > 0.0 {
        let cand_idx: Vec<u32> = (0..n as u32)
            .filter(|&i| any_candidate[i as usize])
            .collect();
        let neigh = knn_neighbor_indices(&s.pos, &cand_idx, cfg.knn_k);
        let mut v = vec![0.0f32; cand_idx.len()];
        for (row, nbrs) in neigh.chunks(cfg.knn_k).enumerate() {
            v[row] = dc_color_variance(&s.dc_color, cand_idx[row], nbrs);
        }
        let tau_v = quantile(v.clone(), cfg.guard_color_var_quantile);
        tau_v.map(|t| {
            // Re-expand candidate-local V back to a per-candidate lookup.
            let mut per_cand = std::collections::HashMap::new();
            for (row, &ci) in cand_idx.iter().enumerate() {
                per_cand.insert(ci, v[row]);
            }
            let dense: Vec<f32> = (0..n as u32)
                .map(|i| *per_cand.get(&i).unwrap_or(&0.0))
                .collect();
            (dense, t)
        })
    } else {
        None
    };

    // Apply the guards (a candidate survives iff it is NOT exempt), then keep
    // the two paths' survivors SEPARATE so each gets its own cap. A Gaussian
    // flagged by both paths lands in both lists; the final union dedups it.
    let exempt = |i: usize| -> bool {
        let exempt_sh = tau_h.is_some_and(|t| s.sh_hf[i] >= t);
        let exempt_thin = tau_s.is_some_and(|t| s.min_scale[i] <= t);
        let exempt_aniso = tau_a.is_some_and(|t| s.aniso[i] >= t);
        let exempt_cv = color_var.as_ref().is_some_and(|(dense, t)| dense[i] >= *t);
        exempt_sh || exempt_thin || exempt_aniso || exempt_cv
    };
    let photo_prune: Vec<u32> = (0..n)
        .filter(|&i| candidate[i] && !exempt(i))
        .map(|i| i as u32)
        .collect();
    let depth_prune: Vec<u32> = (0..n)
        .filter(|&i| depth_candidate[i] && !exempt(i))
        .map(|i| i as u32)
        .collect();

    // -- PHOTOMETRIC isolation pruning: score survivors by mean distance to
    // their k=16 nearest neighbours over the FULL point set. LARGE distance =
    // isolated = floater. Two caps, DIFFERENT denominators, both applied (min):
    // at most `local_cap_frac` of a spatial cell's candidates, and at most
    // `global_cap_frac` of ALL Gaussians, per cycle.
    let photo_selected = if photo_prune.is_empty() {
        Vec::new()
    } else {
        isolation_select(
            &s.pos,
            &photo_prune,
            n,
            cfg.knn_k,
            cfg.local_cap_frac,
            cfg.global_cap_frac,
        )
    };

    // -- DEPTH cap: this path is trusted (LiDAR-gated) and does NOT need spatial
    // isolation — a haze splat in front of a measured surface is the signal,
    // whether or not it is isolated. Cap by its own looser `depth_cap_frac` of
    // ALL Gaussians, keeping the most-floating (highest `float_frac`) first.
    let depth_selected = depth_cap_select(&depth_prune, s, n, cfg.depth_cap_frac);

    // Union (dedup) the two paths.
    if depth_selected.is_empty() {
        return photo_selected;
    }
    let mut seen = vec![false; n];
    let mut out = Vec::with_capacity(photo_selected.len() + depth_selected.len());
    for i in photo_selected.into_iter().chain(depth_selected) {
        if !seen[i as usize] {
            seen[i as usize] = true;
            out.push(i);
        }
    }
    out
}

/// Depth-path cap: keep the most-floating survivors up to `cap_frac` of ALL
/// Gaussians. Trusted (LiDAR-gated) so no isolation scoring — ranked purely by
/// `float_frac = float_accum / max(valid_accum, 1)` so the most-confidently
/// floating Gaussians are pruned first when the cap bites.
fn depth_cap_select(
    depth_prune: &[u32],
    s: &HostSignals,
    n_total: usize,
    cap_frac: f32,
) -> Vec<u32> {
    if depth_prune.is_empty() || cap_frac <= 0.0 {
        return Vec::new();
    }
    let cap = (n_total as f32 * cap_frac).floor() as usize;
    if cap == 0 {
        return Vec::new();
    }
    let mut ranked: Vec<(u32, f32)> = depth_prune
        .iter()
        .map(|&i| {
            let valid = s.valid_accum[i as usize].max(1.0);
            (i, s.float_accum[i as usize] / valid)
        })
        .collect();
    // Most-floating first; the cap keeps the head.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(cap);
    ranked.into_iter().map(|(i, _)| i).collect()
}

/// Warn exactly once that `--tidi-depth-prune` is on but the active camera is
/// not a pinhole, so the depth accumulation is skipped for those views. Once,
/// because it would otherwise fire every step of every non-pinhole view.
fn warn_depth_prune_non_pinhole() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "--tidi-depth-prune is set but this camera is not Pinhole; the depth \
             residual accumulation is skipped for non-pinhole views (the pinhole \
             projection is the only model implemented on this path, matching the \
             --depth-normal-weight restriction). The depth prune will be inert \
             unless pinhole views are present."
        );
    });
}

/// Warn exactly once that `--tidi-depth-prune` is on but NO view ever produced a
/// valid depth return (dataset has no per-frame depth, or every map was empty).
/// The depth path then no-ops (no candidate can form); it never panics.
fn warn_depth_prune_no_depth() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "--tidi-depth-prune is set but no per-frame depth was found (no view \
             produced a valid depth return). The depth-residual prune path is \
             inert for this run; load depth maps (depth/<stem>.tiff) to enable it."
        );
    });
}

/// Uniform-grid bucketing shared by the isolation scorer and the colour-variance
/// guard. Returns the grid, the per-axis minimum, and the cell size.
fn build_grid(
    pos: &[f32],
    n: usize,
) -> (hashbrown::HashMap<(i64, i64, i64), Vec<u32>>, [f32; 3], f32) {
    let p = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        let q = p(i);
        for d in 0..3 {
            if q[d].is_finite() {
                mn[d] = mn[d].min(q[d]);
                mx[d] = mx[d].max(q[d]);
            }
        }
    }
    let extent = (0..3)
        .map(|d| (mx[d] - mn[d]).max(1e-6))
        .fold(0.0, f32::max);
    // ~8 points per cell so a 3×3×3 scan usually clears k=16.
    let cells_per_axis = ((n as f32 / 8.0).cbrt().ceil() as i64).max(1);
    let cell = extent / cells_per_axis as f32;
    let mut grid: hashbrown::HashMap<(i64, i64, i64), Vec<u32>> = hashbrown::HashMap::new();
    for i in 0..n {
        let q = p(i);
        if q.iter().all(|v| v.is_finite()) {
            grid.entry(cell_key(q, mn, cell))
                .or_default()
                .push(i as u32);
        }
    }
    (grid, mn, cell)
}

fn cell_key(q: [f32; 3], mn: [f32; 3], cell: f32) -> (i64, i64, i64) {
    (
        ((q[0] - mn[0]) / cell) as i64,
        ((q[1] - mn[1]) / cell) as i64,
        ((q[2] - mn[2]) / cell) as i64,
    )
}

/// Mean distance from each query point to its `k` nearest neighbours over the
/// FULL point set (the query's own index is skipped). Grid-accelerated, mirrors
/// the DiG `grid_knn` neighbourhood scan but returns distances rather than
/// indices.
fn knn_mean_dist(pos: &[f32], query: &[u32], k: usize) -> Vec<f32> {
    let n = pos.len() / 3;
    let (grid, mn, cell) = build_grid(pos, n);
    let p = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    query
        .par_iter()
        .map(|&qi| {
            let q = p(qi as usize);
            let mut best: Vec<f32> = Vec::with_capacity(64);
            for radius in 1..=3i64 {
                let (cx, cy, cz) = cell_key(q, mn, cell);
                for dx in -radius..=radius {
                    for dy in -radius..=radius {
                        for dz in -radius..=radius {
                            if radius > 1
                                && dx.abs() < radius
                                && dy.abs() < radius
                                && dz.abs() < radius
                            {
                                continue;
                            }
                            if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &j in ids {
                                    if j == qi {
                                        continue;
                                    }
                                    let r = p(j as usize);
                                    best.push((0..3).map(|d| (q[d] - r[d]).powi(2)).sum::<f32>());
                                }
                            }
                        }
                    }
                }
                if best.len() >= k {
                    break;
                }
            }
            if best.is_empty() {
                return f32::INFINITY; // fully isolated → maximally prunable.
            }
            let take = k.min(best.len());
            // Partition by SQUARED distance to isolate the k nearest (sqrt is
            // monotone, so the k-smallest set is identical).
            best.select_nth_unstable_by(take - 1, |a, b| a.total_cmp(b));
            // Isolation score = MEAN of the k neighbour DISTANCES (spec + the
            // docstring above), i.e. `mean(d)`. NOT `sqrt(mean(d²))` = RMS, which
            // is ≥ mean and reads a 15-close-1-far cluster as spuriously
            // isolated. So take sqrt of each squared distance BEFORE averaging.
            best[..take].iter().map(|d2| d2.sqrt()).sum::<f32>() / take as f32
        })
        .collect()
}

/// k-NN neighbour indices for each query (used by the colour-variance guard).
fn knn_neighbor_indices(pos: &[f32], query: &[u32], k: usize) -> Vec<i64> {
    let n = pos.len() / 3;
    let (grid, mn, cell) = build_grid(pos, n);
    let p = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    let mut out = vec![-1i64; query.len() * k];
    out.par_chunks_mut(k)
        .zip(query.par_iter())
        .for_each(|(slot, &qi)| {
            let q = p(qi as usize);
            let mut best: Vec<(f32, u32)> = Vec::with_capacity(64);
            for radius in 1..=3i64 {
                let (cx, cy, cz) = cell_key(q, mn, cell);
                for dx in -radius..=radius {
                    for dy in -radius..=radius {
                        for dz in -radius..=radius {
                            if radius > 1
                                && dx.abs() < radius
                                && dy.abs() < radius
                                && dz.abs() < radius
                            {
                                continue;
                            }
                            if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &j in ids {
                                    if j == qi {
                                        continue;
                                    }
                                    let r = p(j as usize);
                                    let d2 = (0..3).map(|d| (q[d] - r[d]).powi(2)).sum::<f32>();
                                    best.push((d2, j));
                                }
                            }
                        }
                    }
                }
                if best.len() >= k {
                    break;
                }
            }
            let take = k.min(best.len());
            if best.len() > take && take > 0 {
                best.select_nth_unstable_by(take - 1, |a, b| a.0.total_cmp(&b.0));
            }
            for (slot_i, item) in slot.iter_mut().enumerate() {
                *item = best.get(slot_i).map_or(-1, |&(_, j)| i64::from(j));
            }
        });
    out
}

/// Variance of DC colour across a query and its neighbours, meaned over RGB.
fn dc_color_variance(dc: &[f32], qi: u32, nbrs: &[i64]) -> f32 {
    let mut total = 0.0f32;
    for ch in 0..3 {
        let mut vals: Vec<f32> = vec![dc[qi as usize * 3 + ch]];
        for &j in nbrs {
            if j >= 0 {
                vals.push(dc[j as usize * 3 + ch]);
            }
        }
        let m = vals.iter().sum::<f32>() / vals.len() as f32;
        total += vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / vals.len() as f32;
    }
    total / 3.0
}

/// Isolation selection with the two caps. Returns the subset of `c_prune` to
/// actually prune. `n_total` is the whole scene's Gaussian count.
fn isolation_select(
    pos: &[f32],
    c_prune: &[u32],
    n_total: usize,
    k: usize,
    local_frac: f32,
    global_frac: f32,
) -> Vec<u32> {
    let dists = knn_mean_dist(pos, c_prune, k);
    let (_, mn, cell) = build_grid(pos, n_total);
    let p = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];

    // Group candidate rows by spatial cell, carrying (index, isolation distance).
    let mut by_cell: hashbrown::HashMap<(i64, i64, i64), Vec<(u32, f32)>> =
        hashbrown::HashMap::new();
    for (row, &gi) in c_prune.iter().enumerate() {
        let key = cell_key(p(gi as usize), mn, cell);
        by_cell.entry(key).or_default().push((gi, dists[row]));
    }

    // Local cap: per cell, keep only the most-isolated `local_frac`. NOTE:
    // `build_grid` sizes cells to ≈8 points, so a cell holds only a handful of
    // candidates. A plain `floor(count * 0.01)` would be 0 for any cell with
    // < 100 candidates — i.e. essentially always, and WORST for the paper's lone
    // isolated floater (lowest per-cell count). So round UP and allow at least
    // one prune from a non-empty cell; the global cap below is the real budget.
    let mut pool: Vec<(u32, f32)> = Vec::new();
    for (_, mut members) in by_cell {
        members.sort_by(|a, b| b.1.total_cmp(&a.1)); // isolation descending
        let cap = if members.is_empty() {
            0
        } else {
            ((members.len() as f32 * local_frac).ceil() as usize)
                .max(1)
                .min(members.len())
        };
        pool.extend(members.into_iter().take(cap));
    }

    // Global cap: keep only the most-isolated `global_frac` of the WHOLE scene.
    let global_cap = (n_total as f32 * global_frac).floor() as usize;
    pool.sort_by(|a, b| b.1.total_cmp(&a.1));
    pool.truncate(global_cap);
    pool.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params() -> TidiPruneParams {
        TidiPruneParams {
            photometric: true,
            vis_threshold: 2.0,
            opacity_threshold: 0.04,
            importance_threshold: 0.35,
            grad_threshold: 5e-4,
            warmup_steps: 500,
            guard_sh_quantile: 0.95,
            guard_thin_quantile: 0.10,
            guard_aniso_quantile: 0.95,
            guard_color_var_quantile: 0.0,
            knn_k: 16,
            local_cap_frac: 1.0,
            global_cap_frac: 1.0,
            depth_prune: false,
            depth_float_frac: 0.5,
            depth_min_valid_views: 4.0,
            depth_cap_frac: 1.0,
        }
    }

    /// Build N stable points on a tight grid plus one isolated floater that
    /// fails all four signals; only the floater should be selected.
    fn scene_with_floater() -> HostSignals {
        let mut pos = Vec::new();
        let mut vis = Vec::new();
        let mut grad = Vec::new();
        let mut opac = Vec::new();
        let mut sw = Vec::new();
        let mut sh = Vec::new();
        let mut ms = Vec::new();
        let mut an = Vec::new();
        let mut dc = Vec::new();
        let mut age = Vec::new();
        // 27 stable points, well observed, opaque, important, still moving, and
        // moderately elongated.
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
                    vis.push(50.0);
                    grad.push(1e-2);
                    opac.push(0.9);
                    sw.push(0.99);
                    sh.push(0.1);
                    ms.push(0.05);
                    an.push(2.0);
                    dc.extend_from_slice(&[0.5, 0.5, 0.5]);
                    age.push(5000);
                }
            }
        }
        // One isolated floater far away, failing all four signals. Its smallest
        // scale axis is LARGER than the stable set (a blobby floater, not a thin
        // structure) and it is near-isotropic (aniso below the stable set), so
        // neither the thinness nor the anisotropy guard exempts it.
        pos.extend_from_slice(&[100.0, 100.0, 100.0]);
        vis.push(1.0);
        grad.push(1e-5);
        opac.push(0.02);
        sw.push(0.1);
        sh.push(0.01);
        ms.push(0.10);
        an.push(1.0);
        dc.extend_from_slice(&[0.5, 0.5, 0.5]);
        age.push(5000);
        let n = vis.len();
        HostSignals {
            n,
            vis,
            grad_ema: grad,
            opacity: opac,
            sigma_w: sw,
            sh_hf: sh,
            min_scale: ms,
            aniso: an,
            pos,
            dc_color: dc,
            age,
            // Depth path off in these photometric-path fixtures.
            float_accum: vec![0.0; n],
            valid_accum: vec![0.0; n],
        }
    }

    #[test]
    fn isolated_floater_is_selected() {
        let s = scene_with_floater();
        let idx = select_prune_indices(&base_params(), &s);
        assert_eq!(idx, vec![(s.n - 1) as u32], "only the isolated floater");
    }

    #[test]
    fn candidate_rule_is_and_not_or() {
        // A point that fails only THREE signals (still important) must not be a
        // candidate — proves the four are AND-ed.
        let mut s = scene_with_floater();
        let f = s.n - 1;
        s.sigma_w[f] = 0.99; // importance high → passes signal 3
        let idx = select_prune_indices(&base_params(), &s);
        assert!(idx.is_empty(), "high-importance floater is not a candidate");
    }

    #[test]
    fn warmup_protects_new_gaussians() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        s.age[f] = 100; // younger than the 500-step warmup
        let idx = select_prune_indices(&base_params(), &s);
        assert!(idx.is_empty(), "a fresh floater is protected by warmup");
    }

    #[test]
    fn sh_energy_guard_exempts_specular_detail() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        // Make the floater's non-DC SH energy the highest in the scene, above
        // the stable 0.95 quantile → detail guard keeps it.
        s.sh_hf[f] = 100.0;
        let idx = select_prune_indices(&base_params(), &s);
        assert!(idx.is_empty(), "high SH-energy candidate is guarded");
    }

    #[test]
    fn global_cap_limits_prune_count() {
        // Many isolated floaters, but a 0-fraction global cap prunes none.
        let mut s = scene_with_floater();
        let mut p = base_params();
        p.global_cap_frac = 0.0;
        let idx = select_prune_indices(&p, &s);
        assert!(idx.is_empty(), "zero global cap prunes nothing");
        // Sanity: with a full cap the floater returns.
        p.global_cap_frac = 1.0;
        let _ = &mut s;
        assert_eq!(select_prune_indices(&p, &s).len(), 1);
    }

    /// BUG-1 regression: at the REAL default `--tidi-local-cap-frac = 0.01`, a
    /// lone isolated floater (the paper's headline case, and the lowest per-cell
    /// count) must still be selected. The pre-fix `floor(1 * 0.01) = 0` dropped
    /// it. Uses a full global cap so this isolates the local-cap path (the
    /// global floor only bites sub-500-gaussian scenes, never real ones).
    #[test]
    fn default_local_cap_still_selects_lone_floater() {
        let s = scene_with_floater();
        let mut p = base_params();
        p.local_cap_frac = 0.01; // production default
        p.global_cap_frac = 1.0;
        let idx = select_prune_indices(&p, &s);
        assert_eq!(
            idx,
            vec![(s.n - 1) as u32],
            "default local cap must not floor a lone floater to zero"
        );
    }

    /// GAP-4: the anisotropy guard exempts an elongated (sheet / needle)
    /// candidate whose s3/s1 is above the stable set's high quantile, even when
    /// its smallest axis is not thin (so the thinness guard would miss it).
    #[test]
    fn anisotropy_guard_exempts_elongated_structure() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        s.aniso[f] = 100.0; // far above the stable 0.95 quantile (2.0)
        let idx = select_prune_indices(&base_params(), &s);
        assert!(idx.is_empty(), "highly anisotropic candidate is guarded");
    }

    /// BUG-3 regression: the isolation score is the MEAN of the k neighbour
    /// distances, not the RMS. Two neighbours at distances 0 and 4 must score
    /// mean = 2.0, not RMS = sqrt((0+16)/2) = 2.83.
    #[test]
    fn isolation_score_is_mean_not_rms() {
        // Query at origin; neighbours at distance 0 (coincident) and 4.
        let pos = vec![
            0.0, 0.0, 0.0, // query (index 0)
            0.0, 0.0, 0.0, // neighbour at distance 0
            4.0, 0.0, 0.0, // neighbour at distance 4
        ];
        let d = knn_mean_dist(&pos, &[0u32], 2);
        assert!(
            (d[0] - 2.0).abs() < 1e-5,
            "expected mean(0,4)=2.0, got {} (RMS would be ~2.83)",
            d[0]
        );
    }

    // BUG-2 (windowed visibility) is exercised at the tensor level by
    // `windowed_visibility_counts_windows_not_steps` below, which runs the real
    // `accumulate_window` on a device rather than hand-injecting `vis` values.

    // ---- Depth / LiDAR-residual prune path (pure selection) ----------------
    //
    // These build a HostSignals directly on the accumulator counts
    // (`float_accum` / `valid_accum`), the way `accumulate_depth` would have left
    // them, and check the depth candidate rule + safety gate. The geometry →
    // counts mapping itself (including "unscanned even if in front → never a
    // valid, so never a float") is checked on-device in
    // `accumulate_depth_counts_front_surface_and_unscanned`.

    /// A DEPTH-ONLY run (photometric path off): a Gaussian that floated in front
    /// of a valid surface in enough views is selected purely by the depth path.
    #[test]
    fn depth_path_selects_floating_gaussian() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        // The floater has been seen behind a real return in 10 views and floated
        // in 8 of them (float_frac 0.8 >= 0.5), well past min_valid_views = 4.
        s.valid_accum[f] = 10.0;
        s.float_accum[f] = 8.0;
        let mut p = base_params();
        p.photometric = false; // depth path stands alone
        p.depth_prune = true;
        let idx = select_prune_indices(&p, &s);
        assert_eq!(
            idx,
            vec![f as u32],
            "a Gaussian floating in front of the surface must be depth-pruned"
        );
    }

    /// A Gaussian sitting AT the surface (valid returns behind it, but it rarely
    /// floats) is below `depth_float_frac` and is exempt.
    #[test]
    fn depth_path_exempts_surface_gaussian() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        s.valid_accum[f] = 10.0;
        s.float_accum[f] = 1.0; // float_frac 0.1 < 0.5
        let mut p = base_params();
        p.photometric = false;
        p.depth_prune = true;
        assert!(
            select_prune_indices(&p, &s).is_empty(),
            "a Gaussian at the measured surface must not be depth-pruned"
        );
    }

    /// SAFETY GATE: a Gaussian in an unscanned region (too few valid returns
    /// behind it) is exempt even when it floats in every view it was seen —
    /// `valid_accum < min_valid_views` short-circuits the candidate rule.
    #[test]
    fn depth_path_exempts_unscanned_region() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        // Only 2 valid returns (< min_valid_views = 4), floated in both.
        s.valid_accum[f] = 2.0;
        s.float_accum[f] = 2.0; // float_frac 1.0, but valid too low
        let mut p = base_params();
        p.photometric = false;
        p.depth_prune = true;
        assert!(
            select_prune_indices(&p, &s).is_empty(),
            "an under-observed (unscanned) Gaussian must never be depth-pruned"
        );
    }

    /// The depth path honours the SAME detail guards as the photometric path: a
    /// floating candidate with unusually high SH energy (specular detail near a
    /// depth discontinuity) is exempted.
    #[test]
    fn depth_path_respects_detail_guards() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        s.valid_accum[f] = 10.0;
        s.float_accum[f] = 10.0;
        s.sh_hf[f] = 100.0; // above the stable 0.95 quantile
        let mut p = base_params();
        p.photometric = false;
        p.depth_prune = true;
        assert!(
            select_prune_indices(&p, &s).is_empty(),
            "a high-SH-energy floating candidate is guarded on the depth path too"
        );
    }

    /// The depth path has its OWN cap: a zero `depth_cap_frac` prunes nothing
    /// even with valid floating candidates.
    #[test]
    fn depth_cap_frac_zero_prunes_nothing() {
        let mut s = scene_with_floater();
        let f = s.n - 1;
        s.valid_accum[f] = 10.0;
        s.float_accum[f] = 10.0;
        let mut p = base_params();
        p.photometric = false;
        p.depth_prune = true;
        p.depth_cap_frac = 0.0;
        assert!(
            select_prune_indices(&p, &s).is_empty(),
            "a zero depth cap prunes nothing"
        );
    }

    /// The two paths UNION: with both on, a photometric floater and a distinct
    /// depth floater are both pruned (dedup handles overlap).
    #[test]
    fn photometric_and_depth_paths_union() {
        // Two isolated floaters: index n-1 (the existing photometric one) and a
        // NEW point we tag depth-floating. Build on scene_with_floater and make a
        // stable interior point into a depth-only candidate.
        let mut s = scene_with_floater();
        let photo = (s.n - 1) as u32;
        let depth_only = 0usize; // a stable grid point, NOT a photometric candidate
        s.valid_accum[depth_only] = 10.0;
        s.float_accum[depth_only] = 10.0;
        let mut p = base_params();
        p.depth_prune = true; // photometric also on (base default)
        // Disable the detail guards: `depth_only` is a copy of a stable grid
        // point, so its SH/thinness values sit exactly at the stable quantiles
        // and the guards would (correctly) exempt it. This test isolates the
        // union of the two candidate PATHS, not the guard behaviour (covered by
        // `depth_path_respects_detail_guards`).
        p.guard_sh_quantile = 0.0;
        p.guard_thin_quantile = 0.0;
        p.guard_aniso_quantile = 0.0;
        let mut idx = select_prune_indices(&p, &s);
        idx.sort_unstable();
        assert!(idx.contains(&photo), "photometric floater pruned");
        assert!(
            idx.contains(&(depth_only as u32)),
            "depth floater pruned via the standalone path"
        );
    }

    // ---- Plane priors (RANSAC + FIX 1 augmentation), pure host --------------

    /// TEST (a): RANSAC recovers a synthetic plane's normal and offset. A dense
    /// z=0 grid dominates the cloud (plus a few off-plane outliers); the top plane
    /// must have a ~±z normal, ~0 offset, and hold the bulk of the points.
    #[test]
    fn ransac_finds_synthetic_plane() {
        let mut pos = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                pos.extend_from_slice(&[i as f32 * 0.1, j as f32 * 0.1, 0.0]);
            }
        }
        // Off-plane outliers (z = 1) well outside the inlier band.
        for k in 0..10 {
            pos.extend_from_slice(&[k as f32 * 0.1, 0.5, 1.0]);
        }
        let ps = extract_planes(&pos, 42);
        assert!(!ps.planes.is_empty(), "RANSAC must find the dominant plane");
        let p = &ps.planes[0];
        assert!(
            p.normal[2].abs() > 0.99,
            "normal should be ~±z, got {:?}",
            p.normal
        );
        assert!(
            p.normal[0].abs() < 0.05 && p.normal[1].abs() < 0.05,
            "normal should be axis-aligned, got {:?}",
            p.normal
        );
        assert!(
            p.offset.abs() < 0.05,
            "plane passes through z=0, offset ~0, got {}",
            p.offset
        );
        assert!(
            p.inlier_frac > 0.5,
            "the grid plane dominates the cloud, got frac {}",
            p.inlier_frac
        );
        // Extent covers the grid span [0, 1.9] on both in-plane axes.
        assert!(p.u_max - p.u_min > 1.5 && p.v_max - p.v_min > 1.5);
    }

    /// TEST (b): plane-gate fills the gap between SPARSE cloud points. Four wall
    /// corners 2 units apart leave the on-wall centre (1,1,0) far from every point
    /// (beyond the search reach), so the point-only field marks it a floater
    /// (`max_dist`) — the exact see-through hole. With the plane, the same voxel
    /// reads ~0 (on-surface). The `nearest_plane_distance` helper is checked
    /// directly for the two other cases: outside the bounded extent → no plane
    /// distance; well off the plane → a large distance (still caught).
    #[test]
    fn plane_gate_fills_gaps_between_sparse_points() {
        let margin = 0.15f32;
        let softness = 0.05f32;
        let pos = vec![
            0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            2.0, 2.0, 0.0,
        ];
        let plane = Plane {
            normal: [0.0, 0.0, 1.0],
            offset: 0.0,
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, 1.0, 0.0],
            u_min: 0.0,
            u_max: 2.0,
            v_min: 0.0,
            v_max: 2.0,
            inlier_frac: 1.0,
        };
        let max_dist = margin + FAR_SPAN * softness.max(1e-6);

        // Point-only: the between-points on-wall voxel is a hole (max_dist).
        let none = build_distance_field(&pos, margin, softness, None).expect("field");
        let [nx, ny, nz] = none.dims;
        let flat = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
        let vidx = |p: f32, o: f32| (((p - o) / none.vox).floor() as usize);
        let (gi, gj, gk) = (
            vidx(1.0, none.origin[0]),
            vidx(1.0, none.origin[1]),
            vidx(0.0, none.origin[2]),
        );
        let _ = nx;
        assert!(
            (none.field[flat(gi, gj, gk)] - max_dist).abs() < 1e-6,
            "point-only: the on-wall gap voxel is a hole (max_dist), got {}",
            none.field[flat(gi, gj, gk)]
        );

        // Plane-gated: same geometry (planes don't change the bbox), gap filled.
        let aug =
            build_distance_field(&pos, margin, softness, Some(&[plane.clone()])).expect("field");
        assert_eq!(
            aug.dims, none.dims,
            "planes must not change the grid geometry"
        );
        assert!(
            aug.field[flat(gi, gj, gk)] < 0.05,
            "plane-gate fills the gap: on-wall voxel reads ~0, got {}",
            aug.field[flat(gi, gj, gk)]
        );

        // Helper: inside-extent near voxel ~0; outside-extent → +inf; off-plane →
        // ~1.0 (still a floater).
        let on = nearest_plane_distance([1.0, 1.0, 0.0], std::slice::from_ref(&plane), 0.05);
        assert!(on < 0.05, "on-wall centre ~0, got {on}");
        let outside = nearest_plane_distance([5.0, 5.0, 0.0], std::slice::from_ref(&plane), 0.05);
        assert!(
            !outside.is_finite(),
            "a point outside the wall extent has no plane distance, got {outside}"
        );
        let above = nearest_plane_distance([1.0, 1.0, 1.0], std::slice::from_ref(&plane), 0.05);
        assert!(
            above > 0.9,
            "a point 1 unit off the plane is not filled, got {above}"
        );
    }

    /// The Jacobi symmetric-3×3 eigensolver on a NON-diagonal matrix — the case
    /// `ransac_finds_synthetic_plane` structurally cannot reach (its axis-aligned
    /// grid gives a diagonal covariance, so the `off < 1e-12` check fires on
    /// iteration 0 and the rotation code never runs). This exercises the rotation:
    /// each returned eigenpair must satisfy `A·v = λ·v` (a wrong rotation angle
    /// corrupts `A` and breaks this), and the smallest eigenvalue must be the true
    /// ~1.697 (the sign-flipped angle converged to a stable-but-wrong ~2.807).
    #[test]
    fn symmetric_eig3_diagonalizes_non_diagonal() {
        let a = [[2.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 9.0]];
        let (vals, vecs) = symmetric_eig3(a);
        for j in 0..3 {
            let v = [vecs[0][j], vecs[1][j], vecs[2][j]];
            let av = [
                a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
                a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
                a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
            ];
            let resid: f64 = (0..3)
                .map(|k| (av[k] - vals[j] * v[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(resid < 1e-6, "eigenpair {j}: A·v = λ·v residual {resid}");
            let vnorm: f64 = (0..3).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
            assert!(
                (vnorm - 1.0).abs() < 1e-6,
                "eigenvector {j} not unit: {vnorm}"
            );
        }
        let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            (mn - 1.697).abs() < 0.01,
            "smallest eigenvalue must be ~1.697 (sign bug gives ~2.807), got {mn}"
        );
    }

    /// BUG-1 regression: `refit_plane` on a deliberately TILTED (non-axis-aligned)
    /// planar patch must recover the ground-truth normal. The tilted grid gives a
    /// non-diagonal covariance, so the Jacobi rotation actually runs; the flipped
    /// angle returned a normal with cosine-similarity ~0.14 to truth (essentially
    /// the wrong direction), corrupting both FIX 1's field and FIX 2's pull.
    #[test]
    fn refit_plane_recovers_tilted_normal() {
        let n_gt = glam::Vec3::new(0.3, 0.4, 0.8).normalize();
        let (u, v) = plane_basis(n_gt.to_array());
        let (u, v) = (glam::Vec3::from(u), glam::Vec3::from(v));
        let mut pos = Vec::new();
        let mut inliers: Vec<u32> = Vec::new();
        for i in -5..=5 {
            for j in -5..=5 {
                let p = u * (i as f32 * 0.1) + v * (j as f32 * 0.1);
                pos.extend_from_slice(&[p.x, p.y, p.z]);
                inliers.push(inliers.len() as u32);
            }
        }
        let (n, d) = refit_plane(&pos, &inliers).expect("refit a tilted plane");
        let cos = glam::Vec3::from(n).dot(n_gt).abs();
        assert!(
            cos > 0.999,
            "tilted-plane normal cos-sim {cos} to ground truth (the sign bug gives ~0.14)"
        );
        assert!(d.abs() < 1e-4, "plane through origin → offset ~0, got {d}");
    }
}

#[cfg(test)]
mod device_tests {
    // Mirrors the sibling `tests` module and train.rs's test module: the glob
    // brings in tidi's `Tensor` / `TensorData` / `Device` imports plus `TidiState`.
    use super::*;

    /// BUG-2 regression: `accumulate_window` must count the NUMBER OF WINDOWS a
    /// gaussian was seen (a 0/1-per-window indicator summed), NOT the raw
    /// per-step visibility count. Two windows with per-step counts [10, 0, 3]
    /// then [5, 0, 0] must give vis_accum [2, 0, 1] (windows seen), not [15, 0,
    /// 3] (steps seen) — otherwise `fail_vis` (<= τ_vis = 2) is unreachable for
    /// any persistently rendered gaussian. Runs the real tensor path on a test
    /// device (GPU harness on the build box).
    #[tokio::test]
    async fn windowed_visibility_counts_windows_not_steps() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let mut tidi = TidiState::new(3, 0, &device);
        let inner = device.clone().inner();
        let zero_grad = Tensor::<1>::zeros([3], &inner);

        tidi.accumulate_window(
            Tensor::<1>::from_data(TensorData::new(vec![10.0f32, 0.0, 3.0], [3]), &inner),
            zero_grad.clone(),
            0.99,
        );
        tidi.accumulate_window(
            Tensor::<1>::from_data(TensorData::new(vec![5.0f32, 0.0, 0.0], [3]), &inner),
            zero_grad,
            0.99,
        );

        let got: Vec<f32> = tidi
            .vis_accum
            .clone()
            .into_data_async()
            .await
            .expect("vis_accum readback")
            .into_vec()
            .expect("f32");
        assert_eq!(
            got,
            vec![2.0, 0.0, 1.0],
            "vis_accum must count windows-seen, not steps-seen"
        );
    }

    /// The real geometry → counts mapping in `accumulate_depth`, covering the
    /// three spec cases in ONE view against a 4×4 pinhole camera at the origin
    /// looking down +z (identity world→cam, so camera-space z == world z):
    ///   * G0 in FRONT of a valid surface  → valid += 1, float += 1;
    ///   * G1 AT the surface                → valid += 1, float += 0;
    ///   * G2 over an INVALID (unscanned)   → valid += 0, float += 0 (never a
    ///     float, even though it is geometrically in front of the camera).
    #[tokio::test]
    async fn accumulate_depth_counts_front_surface_and_unscanned() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let inner = device.clone().inner();
        let mut tidi = TidiState::new(3, 0, &device);

        // 4×4 pinhole, 90° fov → fx = fy = 2, cx = cy = 2 (center_uv 0.5).
        let img = glam::UVec2::new(4, 4);
        let camera = Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        );

        // Camera-space (== world here) positions, all at z = 5:
        //   G0 (0,0,5)    → pixel (u=2, v=2)
        //   G1 (2.5,0,5)  → pixel (u=3, v=2)
        //   G2 (-2.5,0,5) → pixel (u=1, v=2)
        let means = Tensor::<2>::from_data(
            TensorData::new(
                vec![0.0f32, 0.0, 5.0, 2.5, 0.0, 5.0, -2.5, 0.0, 5.0],
                [3, 3],
            ),
            &inner,
        );

        // Depth map [H=4, W=4], row-major (row = v, col = u), 0 = invalid:
        //   (2,2) = 10  (surface 5 behind G0's z=5 → G0 floats)
        //   (2,3) = 5   (surface at G1's z=5      → G1 at surface)
        //   (2,1) = 0   (no return                → G2 unscanned)
        let mut depth = vec![0.0f32; 16];
        depth[2 * 4 + 2] = 10.0;
        depth[2 * 4 + 3] = 5.0;
        depth[2 * 4 + 1] = 0.0;
        let gt = TensorData::new(depth, [4, 4]);

        tidi.accumulate_depth(means, gt, &camera, img, 0.05);

        let valid: Vec<f32> = tidi
            .valid_accum
            .clone()
            .into_data_async()
            .await
            .expect("valid_accum readback")
            .into_vec()
            .expect("f32");
        let floating: Vec<f32> = tidi
            .float_accum
            .clone()
            .into_data_async()
            .await
            .expect("float_accum readback")
            .into_vec()
            .expect("f32");

        assert_eq!(
            valid,
            vec![1.0, 1.0, 0.0],
            "valid: front, surface, unscanned"
        );
        assert_eq!(
            floating,
            vec![1.0, 0.0, 0.0],
            "float: only the Gaussian in front of a valid surface"
        );
    }

    /// FIX 1 — the 3D distance-to-cloud gate routes an opacity gradient to a
    /// Gaussian FAR from the cloud and leaves an on-surface one ~0. Build a small
    /// planar cloud patch near the origin (plus a distant background point), then
    /// probe two opacity rows. Proves (1) the activated opacity is the live leaf
    /// the term differentiates, (2) the field/positions are detached (the loss is
    /// finite and only opacity moves), and (3) the SIGN is right: far = penalized.
    #[tokio::test]
    async fn opacity_reg_3d_gate_grads_far_not_on_surface() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let margin = 0.15f32;
        let softness = 0.05f32;

        // Cloud: a 3×3 patch of points in the z = 0 plane (spacing 0.05) = the
        // measured surface, plus one distant background point at (2, 2, 2).
        let mut cloud = Vec::new();
        for i in -1..=1 {
            for j in -1..=1 {
                cloud.extend_from_slice(&[i as f32 * 0.05, j as f32 * 0.05, 0.0]);
            }
        }
        cloud.extend_from_slice(&[2.0, 2.0, 2.0]);
        let m = cloud.len() / 3;
        let cloud = Tensor::<2>::from_data(TensorData::new(cloud, [m, 3]), &device);
        let grid = CloudDistanceGrid::build(cloud, margin, softness, None, &device)
            .await
            .expect("a non-empty cloud builds a grid");

        // G0 ON the surface (coincident with a cloud point) → d ~ 0 → p ~ 0.
        // G1 FAR in empty space (1,1,1), ~1.7 from either cluster → d = max_dist → p ~ 1.
        let means = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], [2, 3]),
            &device,
        );
        // Opacity leaf; σ'(0) = 0.25, a clean nonzero derivative on every row so a
        // zero gradient can only come from a zero penalty weight.
        let raw = lift_to_autodiff(Tensor::<1>::from_data(
            TensorData::new(vec![0.0f32, 0.0], [2]),
            &device,
        ))
        .require_grad();

        let loss = depth_opacity_reg_loss(raw.clone(), means, &grid, margin, softness);
        let grads = loss.backward();
        let g: Vec<f32> = raw
            .grad(&grads)
            .expect("the opacity leaf must receive a gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");

        // Far-from-cloud Gaussian: penalized, so a clearly POSITIVE gradient
        // (descent lowers its opacity). This is the whole point of the term.
        assert!(
            g[1] > 1e-4,
            "far-from-cloud Gaussian must get a positive opacity gradient, got {}",
            g[1]
        );
        // On the surface: penalty weight ~0 → negligible gradient vs the floater
        // (also proves positions are detached — only the penalty scales it).
        assert!(
            g[0].abs() < g[1] * 0.2,
            "on-surface Gaussian gradient must be ~0 vs the floater, got {} (floater {})",
            g[0],
            g[1]
        );
    }

    /// FIX 1 — immunity to the SEE-THROUGH case the per-pixel z-buffer failed on.
    /// A surface Gaussian sitting just in front of a cloud point stays SAFE even
    /// though a far background point exists "behind" it — which a naive projected
    /// depth would have let leak through a foreground gap, flagging the surface
    /// splat as floating. The 3D gate has no camera and no z-buffer, so only
    /// proximity to the cloud matters: the background point cannot mark it.
    #[tokio::test]
    async fn opacity_reg_3d_gate_immune_to_see_through() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let margin = 0.15f32;
        let softness = 0.05f32;

        // One foreground surface point at the origin and a far background point
        // 3 units behind it (the "see-through" background).
        let cloud = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 3.0], [2, 3]),
            &device,
        );
        let grid = CloudDistanceGrid::build(cloud, margin, softness, None, &device)
            .await
            .expect("a non-empty cloud builds a grid");

        // The surface Gaussian sits 0.02 in front of the foreground point (well
        // within margin), with the background point 3 units further along +z.
        let means =
            Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 0.0, 0.02], [1, 3]), &device);
        let raw = lift_to_autodiff(Tensor::<1>::from_data(
            TensorData::new(vec![0.0f32], [1]),
            &device,
        ))
        .require_grad();

        let loss = depth_opacity_reg_loss(raw.clone(), means, &grid, margin, softness);
        let grads = loss.backward();
        let g: Vec<f32> = raw
            .grad(&grads)
            .expect("the opacity leaf must receive a gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");

        // A floater at this opacity would get ~0.25; the surface splat's grad must
        // stay small (~p·0.25 with p ≈ 0.05), i.e. it is NOT treated as floating.
        assert!(
            g[0] < 0.05,
            "a surface Gaussian near a cloud point must stay safe (small opacity \
             gradient) despite a far background point, got {}",
            g[0]
        );
    }

    /// FIX 2 — the start-iter gate: no term before `start_iter` (and none when the
    /// weight is 0); it fires at/after it. Pure logic, so no device needed.
    #[test]
    fn opacity_reg_start_iter_gates_term() {
        // Weight 0 → never active, regardless of iter.
        assert!(!opacity_reg_active(20_000, 15_000, 0.0));
        // Before start_iter → inactive (no cost, lets densification finish).
        assert!(!opacity_reg_active(14_999, 15_000, 0.1));
        // At and after start_iter → active.
        assert!(opacity_reg_active(15_000, 15_000, 0.1));
        assert!(opacity_reg_active(30_000, 15_000, 0.1));
        // Default start 0 → active from the first step (unchanged behaviour).
        assert!(opacity_reg_active(0, 0, 0.1));
    }

    /// The host distance-field build: an on-cloud voxel reads ~0 (conservative
    /// bias), a voxel in genuinely-empty space saturates to `max_dist`, and the
    /// field is dense over the padded grid. Pure host logic, no device.
    #[test]
    fn build_distance_field_on_cloud_zero_far_saturated() {
        let margin = 0.15f32;
        let softness = 0.05f32;
        // Two points 2 units apart: the gap between them is genuinely empty.
        let data = build_distance_field(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0], margin, softness, None)
            .expect("a finite point builds a field");
        let [nx, ny, nz] = data.dims;
        let flat = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
        let vidx = |p: f32, o: f32| (((p - o) / data.vox).floor() as usize);

        // The voxel holding a cloud point reads ~0.
        let (i, j, k) = (
            vidx(0.0, data.origin[0]),
            vidx(0.0, data.origin[1]),
            vidx(0.0, data.origin[2]),
        );
        assert!(
            data.field[flat(i, j, k)] < 0.02,
            "on-cloud voxel must read ~0, got {}",
            data.field[flat(i, j, k)]
        );
        // A voxel in the empty gap (x ≈ 1) is beyond the search reach → max_dist.
        let expected_max = margin + FAR_SPAN * softness.max(1e-6);
        let (gi, gj, gk) = (
            vidx(1.0, data.origin[0]),
            vidx(0.0, data.origin[1]),
            vidx(0.0, data.origin[2]),
        );
        assert!(
            (data.field[flat(gi, gj, gk)] - expected_max).abs() < 1e-6,
            "an empty-space voxel must read the truncation max_dist ({expected_max}), got {}",
            data.field[flat(gi, gj, gk)]
        );
        assert_eq!(
            data.field.len(),
            nx * ny * nz,
            "field is dense over the grid"
        );
    }

    /// BUG-1 regression: the flat voxel index MUST be computed in integer
    /// arithmetic. A grid can ship up to MAX_VOXELS (> 2^24) voxels, and f32 is
    /// exact only to 2^24, so an f32 `i·(ny·nz)+…` rounds a large index onto an
    /// adjacent voxel — the device gather would then read the wrong distance for
    /// ~half the Gaussians. This encodes the contract the loss now relies on: the
    /// integer formula is exact where the f32 formula collides, at an index the
    /// production cap can actually reach.
    #[test]
    fn flat_index_stays_exact_beyond_f32_mantissa() {
        // A realistic ~100 m corridor at margin 0.15 (vox 0.05) reaches ~20 M
        // voxels — under the 24 M cap, so it ships without growing `vox`.
        let (ny, nz) = (300usize, 300usize); // ny*nz = 90_000
        let (i, j, k) = (233usize, 7usize, 3usize); // i*ny*nz = 20_970_000 > 2^24
        let exact = (i as i64) * (ny as i64) * (nz as i64) + (j as i64) * (nz as i64) + k as i64;
        assert!(
            exact > (1i64 << 24) && (exact as usize) < MAX_VOXELS,
            "test index must be in the lossy zone yet under the cap"
        );
        // The f32 computation the OLD code used collides with a different index.
        let as_f32 = (i as f32) * (ny * nz) as f32 + (j as f32) * (nz as f32) + k as f32;
        assert_ne!(
            as_f32 as i64, exact,
            "f32 index must round above 2^24 (this is the bug the int fix avoids)"
        );
    }

    /// BUG-2 regression: a Gaussian with a divergent (NaN) centre must be FULLY
    /// penalized (large opacity gradient), never silently spared by whatever
    /// voxel a garbage index happens to land on. The non-finite-row mask forces
    /// `p = 1` for it.
    #[tokio::test]
    async fn opacity_reg_3d_gate_penalizes_divergent_mean() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let margin = 0.15f32;
        let softness = 0.05f32;

        // A small surface patch so the grid is well-formed.
        let mut cloud = Vec::new();
        for i in -1..=1 {
            for j in -1..=1 {
                cloud.extend_from_slice(&[i as f32 * 0.05, j as f32 * 0.05, 0.0]);
            }
        }
        let m = cloud.len() / 3;
        let cloud = Tensor::<2>::from_data(TensorData::new(cloud, [m, 3]), &device);
        let grid = CloudDistanceGrid::build(cloud, margin, softness, None, &device)
            .await
            .expect("a non-empty cloud builds a grid");

        // G0 on the surface (safe); G1 with a NaN centre (divergent → must be
        // penalized regardless of which voxel the zeroed index resolves to).
        let means = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, f32::NAN, f32::NAN, f32::NAN], [2, 3]),
            &device,
        );
        let raw = lift_to_autodiff(Tensor::<1>::from_data(
            TensorData::new(vec![0.0f32, 0.0], [2]),
            &device,
        ))
        .require_grad();

        let loss = depth_opacity_reg_loss(raw.clone(), means, &grid, margin, softness);
        let grads = loss.backward();
        let g: Vec<f32> = raw
            .grad(&grads)
            .expect("the opacity leaf must receive a gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");

        // The loss must stay finite (no NaN poison) and the divergent row must be
        // penalized (p forced to 1 → the largest opacity gradient here).
        assert!(
            g.iter().all(|v| v.is_finite()),
            "a NaN-centre Gaussian must not poison the loss/gradients, got {g:?}"
        );
        assert!(
            g[1] > 1e-4,
            "a divergent (NaN-centre) Gaussian must be fully penalized, got {}",
            g[1]
        );
    }

    /// TEST (c): the co-planarity constraint (FIX 2) routes a real GEOMETRY
    /// gradient. Build one bounded plane at z=0; probe an off-plane Gaussian
    /// (0,0,0.3) inside the assign band and an on-plane Gaussian (0.5,0.5,0). The
    /// off-plane one must get a POSITIVE z position-gradient (descent pulls it onto
    /// the plane) and a nonzero SCALE gradient on its normal-aligned axis (the
    /// flatten term); the on-plane one's position gradient must be ~0. This also
    /// proves the term reaches means AND scales without a backend cross-kind panic.
    #[tokio::test]
    async fn coplanarity_pulls_off_plane_gaussian_and_spares_on_plane() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let plane = Plane {
            normal: [0.0, 0.0, 1.0],
            offset: 0.0,
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, 1.0, 0.0],
            u_min: -2.0,
            u_max: 2.0,
            v_min: -2.0,
            v_max: 2.0,
            inlier_frac: 1.0,
        };
        let planes = PlaneSet {
            planes: vec![plane],
            spacing: 0.05,
            threshold: 0.1,
        };

        // G0 off-plane (z=0.3, within assign_dist=1.0); G1 on-plane.
        let means = lift_to_autodiff(Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.3, 0.5, 0.5, 0.0], [2, 3]),
            &device,
        ))
        .require_grad();
        // Identity quaternions (w,x,y,z) = (1,0,0,0): local frame == world.
        let rots = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [2, 4]),
            &device,
        );
        let scales = lift_to_autodiff(Tensor::<2>::from_data(
            TensorData::new(vec![0.1f32, 0.1, 0.1, 0.1, 0.1, 0.1], [2, 3]),
            &device,
        ))
        .require_grad();

        let term =
            plane_coplanarity_loss(means.clone(), rots, scales.clone(), &planes, 1.0, &device)
                .expect("a non-empty plane set with assigned gaussians yields a term");
        let grads = term.backward();

        let gm: Vec<f32> = means
            .grad(&grads)
            .expect("means must receive a position gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");
        // G0 z-gradient positive (pull onto the plane); G1 z-gradient ~0.
        assert!(
            gm[2] > 1e-3,
            "off-plane Gaussian must be pulled toward the plane (+z grad), got {}",
            gm[2]
        );
        assert!(
            gm[5].abs() < 1e-4,
            "on-plane Gaussian must have ~0 position gradient, got {}",
            gm[5]
        );

        let gs: Vec<f32> = scales
            .grad(&grads)
            .expect("scales must receive a flatten gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");
        // The flatten term penalizes the normal-aligned (z) scale axis of the
        // assigned Gaussian — a nonzero, POSITIVE gradient there (shrinks it).
        assert!(
            gs[2] > 1e-4,
            "the flatten term must reach the normal-aligned scale axis, got {}",
            gs[2]
        );
    }

    /// TEST (d, part): an empty plane set makes the co-planarity term inert
    /// (returns `None`, so the caller adds nothing) — the byte-inert path when no
    /// planar structure is found. The config-level test covers the flags defaulting
    /// off; this covers the loss guard.
    #[tokio::test]
    async fn coplanarity_empty_planeset_is_none() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let means =
            Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 0.0, 0.0], [1, 3]), &device);
        let rots = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0], [1, 4]),
            &device,
        );
        let scales =
            Tensor::<2>::from_data(TensorData::new(vec![0.1f32, 0.1, 0.1], [1, 3]), &device);
        let empty = PlaneSet {
            planes: vec![],
            spacing: 0.05,
            threshold: 0.1,
        };
        assert!(
            plane_coplanarity_loss(means, rots, scales, &empty, 0.15, &device).is_none(),
            "an empty plane set must produce no term"
        );
    }

    /// BUG-2 regression: a Gaussian with a divergent (NaN) centre must NOT poison
    /// the co-planarity loss or its backward. Assignment already excludes the NaN
    /// row, but `0 · NaN = NaN` in BOTH forward and backward would have NaN-ed the
    /// gradient for EVERY Gaussian (means is a live leaf here). The input
    /// sanitization must keep the loss and all geometry gradients finite while the
    /// on-plane Gaussian is still pulled normally.
    #[tokio::test]
    async fn coplanarity_nan_mean_does_not_poison_loss() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let plane = Plane {
            normal: [0.0, 0.0, 1.0],
            offset: 0.0,
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, 1.0, 0.0],
            u_min: -2.0,
            u_max: 2.0,
            v_min: -2.0,
            v_max: 2.0,
            inlier_frac: 1.0,
        };
        let planes = PlaneSet {
            planes: vec![plane],
            spacing: 0.05,
            threshold: 0.1,
        };

        // G0 on-plane (assigned, pulled); G1 with a NaN centre (excluded — must
        // not poison the loss/gradients).
        let means = lift_to_autodiff(Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.3, f32::NAN, f32::NAN, f32::NAN], [2, 3]),
            &device,
        ))
        .require_grad();
        let rots = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [2, 4]),
            &device,
        );
        let scales = lift_to_autodiff(Tensor::<2>::from_data(
            TensorData::new(vec![0.1f32, 0.1, 0.1, 0.1, 0.1, 0.1], [2, 3]),
            &device,
        ))
        .require_grad();

        let term =
            plane_coplanarity_loss(means.clone(), rots, scales.clone(), &planes, 1.0, &device)
                .expect("a non-empty plane set with an assigned gaussian yields a term");
        // The loss value itself must be finite.
        let lv: Vec<f32> = term
            .clone()
            .into_data_async()
            .await
            .expect("loss readback")
            .into_vec()
            .expect("f32");
        assert!(
            lv.iter().all(|v| v.is_finite()),
            "a NaN-centre Gaussian must not poison the loss value, got {lv:?}"
        );

        let grads = term.backward();
        let gm: Vec<f32> = means
            .grad(&grads)
            .expect("means gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");
        let gs: Vec<f32> = scales
            .grad(&grads)
            .expect("scales gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");
        assert!(
            gm.iter().all(|v| v.is_finite()),
            "means gradients must stay finite despite the NaN centre, got {gm:?}"
        );
        assert!(
            gs.iter().all(|v| v.is_finite()),
            "scales gradients must stay finite despite the NaN centre, got {gs:?}"
        );
        // The on-plane Gaussian is still pulled (finite, positive +z position grad).
        assert!(
            gm[2] > 1e-3,
            "the finite on-plane Gaussian must still be pulled, got {}",
            gm[2]
        );
    }

    /// cloud-prune TEST (a): the hard distance-to-cloud prune marks a Gaussian
    /// FAR from the cloud and spares an on-surface one. Build a small planar cloud
    /// patch at the origin, then read `gather_prune_distances`: an on-surface
    /// centre reads `d ≤ dist` (kept) and a mid-air centre `d > dist` (pruned).
    /// The DISTANCES are read directly (f32) — the union-site threshold is what
    /// turns them into a `prune_mask`-kind Bool, so the test verifies the actual
    /// signal without depending on any Bool representation.
    #[tokio::test]
    async fn cloud_prune_far_mask_marks_far_spares_on_surface() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let dist = 0.19f32;
        let softness = dist / 3.0; // the trainer's sizing (see init_plane_priors)

        // Cloud: a 3×3 patch in the z = 0 plane (spacing 0.05) = the measured surface.
        let mut cloud = Vec::new();
        for i in -1..=1 {
            for j in -1..=1 {
                cloud.extend_from_slice(&[i as f32 * 0.05, j as f32 * 0.05, 0.0]);
            }
        }
        let m = cloud.len() / 3;
        let cloud = Tensor::<2>::from_data(TensorData::new(cloud, [m, 3]), &device);
        // POINT-ONLY grid (planes = None), exactly as the cloud-prune builds it.
        let grid = CloudDistanceGrid::build(cloud, dist, softness, None, &device)
            .await
            .expect("a non-empty cloud builds a grid");

        // G0 ON the surface (coincident with a cloud point); G1 FAR at (1,1,1).
        let means = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], [2, 3]),
            &device,
        );
        let dists: Vec<f32> = grid
            .gather_prune_distances(means)
            .into_data_async()
            .await
            .expect("distance readback")
            .into_vec()
            .expect("f32");
        assert!(
            dists[0] <= dist,
            "an on-surface Gaussian must read d ≤ dist (not pruned), got {}",
            dists[0]
        );
        assert!(
            dists[1] > dist,
            "a far-from-cloud Gaussian must read d > dist (pruned), got {}",
            dists[1]
        );
    }

    /// cloud-prune TEST (c): the gather uses LIVE means. Against ONE static grid,
    /// the SAME Gaussian reads as on-surface at one position and as a floater
    /// after it MOVES away — proving the prune gathers the current mean each cycle
    /// (positions drift during training), not a stale snapshot. Also covers the
    /// divergent-centre row (NaN → always far → pruned).
    #[tokio::test]
    async fn cloud_prune_gathers_live_means_after_move() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let dist = 0.19f32;
        let softness = dist / 3.0;

        let mut cloud = Vec::new();
        for i in -1..=1 {
            for j in -1..=1 {
                cloud.extend_from_slice(&[i as f32 * 0.05, j as f32 * 0.05, 0.0]);
            }
        }
        let m = cloud.len() / 3;
        let cloud = Tensor::<2>::from_data(TensorData::new(cloud, [m, 3]), &device);
        let grid = CloudDistanceGrid::build(cloud, dist, softness, None, &device)
            .await
            .expect("a non-empty cloud builds a grid");

        // Cycle 1: the Gaussian sits on the surface → d ≤ dist → not pruned.
        let near = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 0.0, 0.0], [1, 3]), &device);
        let d1: Vec<f32> = grid
            .gather_prune_distances(near)
            .into_data_async()
            .await
            .expect("distance readback")
            .into_vec()
            .expect("f32");
        assert!(
            d1[0] <= dist,
            "before moving, the on-surface Gaussian reads d ≤ dist, got {}",
            d1[0]
        );

        // Cycle 2: the SAME Gaussian has drifted far off the surface → d > dist now,
        // against the identical (static) grid. This is the live-means contract.
        let moved =
            Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 0.0, 1.5], [1, 3]), &device);
        let d2: Vec<f32> = grid
            .gather_prune_distances(moved)
            .into_data_async()
            .await
            .expect("distance readback")
            .into_vec()
            .expect("f32");
        assert!(
            d2[0] > dist,
            "after drifting off the surface the SAME Gaussian reads d > dist (live means), got {}",
            d2[0]
        );

        // A divergent (NaN) centre is forced to +inf → always > dist → pruned.
        let nan = Tensor::<2>::from_data(
            TensorData::new(vec![f32::NAN, f32::NAN, f32::NAN], [1, 3]),
            &device,
        );
        let dn: Vec<f32> = grid
            .gather_prune_distances(nan)
            .into_data_async()
            .await
            .expect("distance readback")
            .into_vec()
            .expect("f32");
        assert!(
            dn[0] > dist,
            "a divergent (NaN) centre must read d > dist (forced +inf), got {}",
            dn[0]
        );
    }

    /// cloud-prune out-of-AABB: a floater far OUTSIDE the cloud's bounding box
    /// must read a LARGE distance and be pruned, not wrap onto a low-distance
    /// boundary voxel and be spared. The grid pads the cloud bbox by `reach`
    /// (= margin + 2·softness) on every side and CLAMPS the voxel coord into
    /// range, so an out-of-box centre lands on a boundary voxel that is at least
    /// `pad = reach` from any cloud point. For the cloud-prune sizing (margin =
    /// dist, softness = dist/3) `pad = 1.67·dist > dist`, so the clamped read is
    /// always > dist. Probe centres far out along each axis (and a diagonal).
    #[tokio::test]
    async fn cloud_prune_out_of_aabb_reads_far_and_prunes() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let dist = 0.19f32;
        let softness = dist / 3.0;

        // A small planar patch at the origin = the whole cloud bbox.
        let mut cloud = Vec::new();
        for i in -1..=1 {
            for j in -1..=1 {
                cloud.extend_from_slice(&[i as f32 * 0.05, j as f32 * 0.05, 0.0]);
            }
        }
        let m = cloud.len() / 3;
        let cloud = Tensor::<2>::from_data(TensorData::new(cloud, [m, 3]), &device);
        let grid = CloudDistanceGrid::build(cloud, dist, softness, None, &device)
            .await
            .expect("a non-empty cloud builds a grid");

        // Centres FAR outside the bbox along +x, -y, +z, and a big diagonal. Each
        // shares two in-range axes with a real cloud column, so a per-axis clamp
        // that dropped the out-of-range axis's distance would spuriously spare it.
        let means = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    50.0f32, 0.0, 0.0, // +x, y/z aligned with a cloud column
                    0.0, -50.0, 0.0, // -y
                    0.0, 0.0, 50.0, // +z
                    100.0, 100.0, 100.0, // far diagonal
                ],
                [4, 3],
            ),
            &device,
        );
        let dists: Vec<f32> = grid
            .gather_prune_distances(means)
            .into_data_async()
            .await
            .expect("distance readback")
            .into_vec()
            .expect("f32");
        assert!(
            dists.iter().all(|&d| d > dist),
            "every out-of-AABB centre must read far (> dist) and be pruned, got {dists:?}"
        );
    }

    /// cloud-prune coarse-grid regression: on a LARGE-extent cloud with a small
    /// `--cloud-prune-dist`, the `vox *= 1.5` coarsening loop (MAX_VOXELS cap)
    /// pushes `vox` past ~1.82·dist, at which point the clamped boundary voxel's
    /// stored distance drops BELOW `dist`. Without the unclamped in-grid mask, a
    /// wildly out-of-grid floater aligned with real cloud on two axes would read
    /// that small boundary distance and be spuriously SPARED (a hard-delete miss).
    /// The in-grid mask forces every out-of-grid centre to `+inf`, removing the
    /// vox-size dependency. Here `dist = 0.01` over a ~5-unit cloud forces the
    /// coarsening, and the far floater must still prune while an on-cloud point is
    /// spared.
    #[tokio::test]
    async fn cloud_prune_out_of_grid_prunes_on_coarsened_grid() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let dist = 0.01f32;
        let softness = dist / 3.0;

        // Two points 5 units apart: a large bbox that forces the fine grid over the
        // 24M-voxel cap, so `build_distance_field` coarsens `vox` well past dist.
        let cloud = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 5.0, 5.0, 5.0], [2, 3]),
            &device,
        );
        let grid = CloudDistanceGrid::build(cloud, dist, softness, None, &device)
            .await
            .expect("a non-empty cloud builds a grid");

        // Confirm we ARE in the coarsened regime the fix targets (vox > 1.82·dist),
        // so the clamp-only path would have read a boundary distance < dist.
        assert!(
            grid.vox > 1.82 * dist,
            "test must exercise the coarsened regime (vox {} vs 1.82·dist {})",
            grid.vox,
            1.82 * dist
        );

        // G0 far OUT of the grid (x = 1000, y/z aligned with the (0,0,0) cloud
        // point); G1 ON a cloud point. Out-of-grid must prune; on-cloud must not.
        let means = Tensor::<2>::from_data(
            TensorData::new(vec![1000.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 3]),
            &device,
        );
        let dists: Vec<f32> = grid
            .gather_prune_distances(means)
            .into_data_async()
            .await
            .expect("distance readback")
            .into_vec()
            .expect("f32");
        assert!(
            dists[0] > dist,
            "an out-of-grid floater must read d > dist even when vox is coarsened past dist, got {}",
            dists[0]
        );
        assert!(
            dists[1] <= dist,
            "an on-cloud Gaussian must read d ≤ dist (spared), got {}",
            dists[1]
        );
    }
}
