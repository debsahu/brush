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

use brush_render::burn_glue::detach_autodiff;
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
    /// Smallest scale axis `s₁` per Gaussian (guard 3, thinness).
    min_scale: Vec<f32>,
    /// Positions `[N*3]` for the isolation k-NN.
    pos: Vec<f32>,
    /// DC colour `[N*3]` for the optional local-colour-variance guard.
    dc_color: Vec<f32>,
    /// Age in steps since birth/last split, for the per-Gaussian warmup.
    age: Vec<i32>,
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
    /// NOTE: the paper's β=0.99 EMA is defined per training step over
    /// `‖∇_x L‖₂`. Brush only materialises a per-refine-window position-gradient
    /// signal (`RefineRecord::refine_weight_norm`, the same quantity the growth
    /// gate thresholds), so the EMA here advances once per refine window, not
    /// per step. β is exposed so the operator can compensate for the coarser
    /// cadence.
    pub fn accumulate_window(&mut self, vis: Tensor<1>, grad: Tensor<1>, beta: f32) {
        self.vis_accum = self.vis_accum.clone() + vis;
        self.grad_ema = self.grad_ema.clone().mul_scalar(beta) + grad.mul_scalar(1.0 - beta);
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
        self.grad_ema = Tensor::cat(vec![self.grad_ema.clone(), zeros], 0);
        let births = Tensor::<1, Int>::full([refine_count], cur_iter as i32, inner_device);
        self.birth_iter = Tensor::cat(vec![self.birth_iter.clone(), births], 0);
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
        // s₁ = smallest scale axis. NOTE: Brush scales are XYZ axes, NOT
        // rank-ordered, so a min-reduction (not a fixed column) is what gives the
        // smallest axis; likewise max/min for anisotropy if a caller wants it.
        let min_scale = scales.clone().min_dim(1).squeeze_dim::<1>(1);
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

        HostSignals {
            n,
            vis: host1(self.vis_accum.clone()).await,
            grad_ema: host1(self.grad_ema.clone()).await,
            opacity: host1(opacity).await,
            sigma_w: host1(sigma_w).await,
            sh_hf: host1(sh_hf).await,
            min_scale: host1(min_scale).await,
            pos: host2(means).await,
            dc_color: host2(dc_color).await,
            age,
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

/// Thresholds + caps for the cleanup pass, snapshotted from `TrainConfig` so the
/// pure selection logic below has no dependency on the config crate.
pub struct TidiPruneParams {
    pub vis_threshold: f32,
    pub opacity_threshold: f32,
    pub importance_threshold: f32,
    pub grad_threshold: f32,
    pub warmup_steps: i32,
    pub guard_sh_quantile: f32,
    pub guard_thin_quantile: f32,
    pub guard_color_var_quantile: f32,
    pub knn_k: usize,
    pub local_cap_frac: f32,
    pub global_cap_frac: f32,
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

/// The pure TIDI selection: candidate rule (AND of four) → detail guards (set
/// difference) → isolation k-NN with the two caps. Split out so it is unit
/// testable without any GPU state.
fn select_prune_indices(cfg: &TidiPruneParams, s: &HostSignals) -> Vec<u32> {
    let n = s.n;
    if n == 0 {
        return Vec::new();
    }

    // -- C_base: a Gaussian is a candidate iff it FAILS ALL FOUR signal
    // thresholds AND is past its per-Gaussian warmup. This is an AND, NOT the
    // OR that Brush's five geometric culls use — a floater is only a candidate
    // when every signal agrees it is idle. (paper §III-B, Table II)
    let mut candidate = vec![false; n];
    for i in 0..n {
        let past_warmup = s.age[i] >= cfg.warmup_steps;
        let fail_vis = s.vis[i] <= cfg.vis_threshold;
        let fail_alpha = s.opacity[i] <= cfg.opacity_threshold;
        let fail_omega = s.sigma_w[i] <= cfg.importance_threshold;
        let fail_grad = s.grad_ema[i] <= cfg.grad_threshold;
        candidate[i] = past_warmup && fail_vis && fail_alpha && fail_omega && fail_grad;
    }

    // -- Adaptive detail guards. Thresholds are a quantile of the STABLE
    // (non-candidate) distribution, recomputed each cycle — the paper gives no
    // fixed numbers, so the flags set the quantile, not the value. A candidate
    // is exempted (kept) if it passes ANY guard (OR): unusually high non-DC SH
    // energy (specular / view-dependent detail), an unusually thin smallest axis
    // (a thin structure), or high local colour variance.
    let stable_of =
        |vals: &[f32]| -> Vec<f32> { (0..n).filter(|&i| !candidate[i]).map(|i| vals[i]).collect() };

    // Guard 1: SH high-frequency energy, exempt at/above the high quantile.
    let tau_h = (cfg.guard_sh_quantile > 0.0)
        .then(|| quantile(stable_of(&s.sh_hf), cfg.guard_sh_quantile))
        .flatten();
    // Guard 3: thinness, exempt at/below the low quantile of s₁.
    let tau_s = (cfg.guard_thin_quantile > 0.0)
        .then(|| quantile(stable_of(&s.min_scale), cfg.guard_thin_quantile))
        .flatten();

    // Guard 2 (optional, default off): local colour variance among a candidate's
    // k-NN. NOTE: deriving τ_V from the stable set would need a full-set k-NN;
    // to avoid that cost this computes V only for candidates and thresholds
    // against the CANDIDATE distribution — a documented approximation, hence
    // off by default.
    let color_var: Option<(Vec<f32>, f32)> = if cfg.guard_color_var_quantile > 0.0 {
        let cand_idx: Vec<u32> = (0..n as u32).filter(|&i| candidate[i as usize]).collect();
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

    // C_prune = C_base \ M_detail.
    let mut c_prune: Vec<u32> = Vec::new();
    for i in 0..n {
        if !candidate[i] {
            continue;
        }
        let exempt_sh = tau_h.is_some_and(|t| s.sh_hf[i] >= t);
        let exempt_thin = tau_s.is_some_and(|t| s.min_scale[i] <= t);
        let exempt_cv = color_var.as_ref().is_some_and(|(dense, t)| dense[i] >= *t);
        if !(exempt_sh || exempt_thin || exempt_cv) {
            c_prune.push(i as u32);
        }
    }
    if c_prune.is_empty() {
        return Vec::new();
    }

    // -- Isolation pruning: score survivors by mean distance to their k=16
    // nearest neighbours over the FULL point set. LARGE distance = isolated =
    // floater. Two caps, DIFFERENT denominators, both applied (min): at most
    // `local_cap_frac` of a spatial cell's candidates, and at most
    // `global_cap_frac` of ALL Gaussians, per cycle.
    isolation_select(
        &s.pos,
        &c_prune,
        n,
        cfg.knn_k,
        cfg.local_cap_frac,
        cfg.global_cap_frac,
    )
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
            best.select_nth_unstable_by(take - 1, |a, b| a.total_cmp(b));
            let mean_sq = best[..take].iter().sum::<f32>() / take as f32;
            mean_sq.sqrt()
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

    // Local cap: per cell, keep only the most-isolated `local_frac` (floor).
    let mut pool: Vec<(u32, f32)> = Vec::new();
    for (_, mut members) in by_cell {
        members.sort_by(|a, b| b.1.total_cmp(&a.1)); // isolation descending
        let cap = (members.len() as f32 * local_frac).floor() as usize;
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
            vis_threshold: 2.0,
            opacity_threshold: 0.04,
            importance_threshold: 0.35,
            grad_threshold: 5e-4,
            warmup_steps: 500,
            guard_sh_quantile: 0.95,
            guard_thin_quantile: 0.10,
            guard_color_var_quantile: 0.0,
            knn_k: 16,
            local_cap_frac: 1.0,
            global_cap_frac: 1.0,
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
        let mut dc = Vec::new();
        let mut age = Vec::new();
        // 27 stable points, well observed, opaque, important, still moving.
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
                    dc.extend_from_slice(&[0.5, 0.5, 0.5]);
                    age.push(5000);
                }
            }
        }
        // One isolated floater far away, failing all four signals. Its smallest
        // scale axis is LARGER than the stable set (a blobby floater, not a thin
        // structure), so the thinness guard does not exempt it.
        pos.extend_from_slice(&[100.0, 100.0, 100.0]);
        vis.push(1.0);
        grad.push(1e-5);
        opac.push(0.02);
        sw.push(0.1);
        sh.push(0.01);
        ms.push(0.10);
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
            pos,
            dc_color: dc,
            age,
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
}
