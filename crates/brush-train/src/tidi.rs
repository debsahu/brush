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

use brush_render::burn_glue::{detach_autodiff, match_backend};
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use burn::{
    Tensor,
    module::{Module, Param, ParamId},
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Bool, Device, Gradients, Int, TensorData, activation::sigmoid},
};
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

        // Everything runs on the accumulators' (inner) device; the shared
        // projection detaches the means off the autodiff graph and moves them
        // there, so no graph is retained and the elementwise math never touches a
        // mismatched backend.
        let device = self.valid_accum.device();
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

/// Shared pinhole projection + GT-depth lookup for BOTH depth-driven paths: the
/// hard `accumulate_depth` prune counter and the smooth `depth_opacity_reg_loss`
/// regularizer. Both need the identical geometry, so the projection lives here
/// once.
///
/// Detaches `means` off the autodiff graph, moves everything to `device`,
/// projects each Gaussian centre into the view, reads the GT depth `Z̃` at the
/// projected pixel, and returns:
///   * `residual` `[N]` — the signed camera-space depth residual `r_i = z_i - Z̃`
///     (< -margin means the Gaussian floats in FRONT of the measured surface);
///   * `valid`    `[N]` — true where the projection is in-frame, in front of the
///     camera (`z > 0`), and lands on a finite positive depth return.
/// The returned tensors are DETACHED (no autodiff node): the depth-residual math
/// never needs a gradient through position. Callers guard pinhole themselves
/// (each owns its warning); this returns `None` only for an empty depth map.
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
    // Detach the means off the autodiff graph and move them onto `device` so no
    // graph is retained and the elementwise math never touches a mismatched
    // backend.
    let means = detach_autodiff(means).to_device(device);

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
    let in_frame = in_front.bool_and(in_x).bool_and(in_y);

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

/// Depth-coupled opacity regularizer — the SMOOTH, differentiable alternative to
/// the hard `accumulate_depth`/depth-prune path. Instead of deleting a floating
/// Gaussian (which orphans its load-bearing colour and leaves a black halo),
/// this adds a per-step loss whose ONLY gradient path is the activated opacity,
/// so the optimizer fades off-surface Gaussians out SMOOTHLY and their colour
/// redistributes into on-surface Gaussians before they vanish.
///
/// For every Gaussian projecting in front of a valid measured depth by more than
/// `margin`, a DETACHED penalty weight
/// `p_i = σ((-r_i - margin) / softness)` (a smooth ramp: ~0 on/behind the
/// surface, → 1 as the Gaussian floats further in front), gated to 0 where the
/// return is invalid, multiplies the ACTIVATED opacity `σ(raw_opacity_i)`. The
/// term is `λ · mean_over_valid(p_i · σ(raw_opacity_i))`.
///
/// `raw_opacity` is the LIVE opacity leaf (`splats.raw_opacities.val()`), so it
/// stays in the autodiff graph; `p_i` and the projection are detached. The only
/// gradient is therefore
/// `∂L/∂raw_opacity_i = λ · p_i · σ'(raw_opacity_i)`, which drives floating
/// Gaussians' opacity toward 0 and touches nothing else.
///
/// Returns `None` (add no term) for an empty depth map, so the caller can skip
/// cleanly. Pinhole is assumed — the caller guards non-pinhole + warns, mirroring
/// the depth-prune and depth/normal terms. `softness` is floored to a tiny
/// positive value so a `0` never divides.
pub fn depth_opacity_reg_loss(
    raw_opacity: Tensor<1>,
    means: Tensor<2>,
    gt_depth: TensorData,
    camera: &Camera,
    img_size: glam::UVec2,
    margin: f32,
    softness: f32,
) -> Option<Tensor<1>> {
    // Project on the opacity leaf's physical device. `project_depth_residual`
    // detaches the means onto the INNER backend (the identical code the prune
    // path runs), so `residual` / `valid` come back inner-kind and carry no
    // gradient.
    let device = raw_opacity.device();
    let (residual, valid) = project_depth_residual(means, gt_depth, camera, img_size, &device)?;

    let valid_f = valid.float();
    // Mean over the valid set. `p_i` is ~0 for valid-but-on-surface Gaussians, so
    // they contribute ~0 to the numerator while still counting in the
    // denominator — a genuine mean over the constrained Gaussians. Floored at 1
    // so an all-invalid view (no valid returns) yields a 0 term, never a NaN.
    let denom = valid_f.clone().sum().clamp_min(1.0);

    // Detached smooth penalty weight: σ((-r - margin) / softness), 0 where the
    // return is invalid. `p_i` inherits the detached projection (no gradient) and
    // is still on the INNER backend at this point.
    let soft = softness.max(1e-8);
    let p = sigmoid(residual.neg().sub_scalar(margin).div_scalar(soft)) * valid_f;

    // Bridge the detached penalty + denominator UP onto the opacity leaf's
    // (autodiff) backend so they can combine with the live activated opacity
    // without a cross-backend panic (project_depth_residual left them inner-kind
    // for the prune path). `match_backend` lifts them as NO-GRAD constants when
    // the reference (raw_opacity) is autodiff, so the ONLY gradient path is
    // through `sigmoid(raw_opacity)` — mirroring how Brush folds the frozen
    // 3D-filter floor against an autodiff param (see `match_backend`).
    let p = match_backend(p, &raw_opacity);
    let denom = match_backend(denom, &raw_opacity);

    // The ONLY graph-connected factor: the activated opacity of the live leaf.
    let alpha = sigmoid(raw_opacity);
    Some((p * alpha).sum() / denom)
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

    /// The depth-coupled opacity regularizer routes a gradient to the opacity
    /// leaf ONLY for a Gaussian floating in front of a valid depth return, and
    /// leaves the leaf's other rows (a Gaussian at/behind the surface, and one
    /// over an unscanned pixel) with ~0 / exactly-0 gradient. Same 4×4 pinhole
    /// at the origin looking down +z as the accumulate test (identity
    /// world→cam, so camera z == world z). Proves (1) the activated opacity is
    /// the live leaf the term differentiates, (2) positions/`p_i` are detached
    /// (no other row moves), and (3) the invalid-return gate zeroes the penalty.
    #[tokio::test]
    async fn depth_opacity_reg_grads_only_floating_gaussian() {
        use brush_render::burn_glue::lift_to_autodiff;
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

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
        //   G0 (0,0,5)    → pixel (u=2, v=2)  floats FAR in front of the surface
        //   G1 (2.5,0,5)  → pixel (u=3, v=2)  sits at/behind the surface
        //   G2 (-2.5,0,5) → pixel (u=1, v=2)  projects over an UNSCANNED pixel
        let means = Tensor::<2>::from_data(
            TensorData::new(
                vec![0.0f32, 0.0, 5.0, 2.5, 0.0, 5.0, -2.5, 0.0, 5.0],
                [3, 3],
            ),
            &device,
        );

        // Depth map [H=4, W=4], row-major (row = v, col = u), 0 = invalid:
        //   (2,2) = 10  surface 5 BEHIND G0 (residual -5 ≪ -margin → floats, p≈1)
        //   (2,3) = 4   surface 1 IN FRONT of G1 (residual +1 → p≈0, not floating)
        //   (2,1) = 0   no return over G2 (valid mask 0 → p gated to exactly 0)
        let mut depth = vec![0.0f32; 16];
        depth[2 * 4 + 2] = 10.0;
        depth[2 * 4 + 3] = 4.0;
        depth[2 * 4 + 1] = 0.0;
        let gt = TensorData::new(depth, [4, 4]);

        // Opacity leaf; σ'(0) = 0.25, a clean nonzero derivative on every row so
        // a zero gradient can only come from a zero penalty weight.
        let raw = lift_to_autodiff(Tensor::<1>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0], [3]),
            &device,
        ))
        .require_grad();

        let loss = depth_opacity_reg_loss(raw.clone(), means, gt, &camera, img, 0.05, 0.1)
            .expect("valid returns exist, so a term is produced");
        let grads = loss.backward();
        let g: Vec<f32> = raw
            .grad(&grads)
            .expect("the opacity leaf must receive a gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec()
            .expect("f32");

        // Floating Gaussian: penalized, so a clearly POSITIVE gradient (descent
        // lowers its opacity). This is the whole point of the term.
        assert!(
            g[0] > 1e-4,
            "floating Gaussian must get a positive opacity gradient, got {}",
            g[0]
        );
        // At/behind the surface: penalty weight ≈ 0 → negligible gradient vs the
        // floater (proves positions are detached — only the penalty scales it).
        assert!(
            g[1].abs() < g[0] * 1e-2,
            "at/behind-surface Gaussian gradient must be ~0 vs the floater, got {} (floater {})",
            g[1],
            g[0]
        );
        // Unscanned pixel: valid mask 0 → penalty gated to exactly 0 → no grad.
        assert_eq!(
            g[2], 0.0,
            "a Gaussian over an unscanned pixel must get exactly zero opacity gradient"
        );
    }
}
