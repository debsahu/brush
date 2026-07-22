use brush_render::burn_glue::detach_autodiff;
use burn::{
    prelude::Int,
    tensor::{Bool, Device, Tensor, TensorData},
};
use tracing::trace_span;

pub(crate) struct RefineRecord {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    pub refine_weight_norm: Tensor<1>,
    pub vis_weight: Tensor<1>,
    pub max_screen_size: Tensor<1>,
    // Edge-guidance accumulator (MRNF port, delta #4): sum over the refine
    // window's sampled views of the per-gaussian projected edge score, plus the
    // sample count. `None`/0 until the first sample; only populated when
    // `--use-edge-map` is set. Lives on the inner device like the others.
    edge_score_sum: Option<Tensor<1>>,
    edge_sample_count: u32,
    // Error-map growth accumulator (MRNF `use_error_map` port): the window-MAX
    // over sampled views of the per-gaussian coverage-weighted MEAN error
    // `(Σ_p T·α·ê)/(Σ_p T·α)`, per-view positive-median normalized (LFS
    // `_refine_weight_max`, mrnf.cpp:602-605, but coverage-normalized — see
    // `train::accumulate_error_sample` defect-2 note). Distinct from `edge_score_sum`
    // (SUM/mean, a bias factor): this is MAX-accumulated and REPLACES the
    // gradient as the growth signal when `--error-map-densification` is set.
    // `None` until the first sample; reset every window because the whole
    // `RefineRecord` is recreated fresh after each refine (LFS resets
    // `_refine_weight_max` at refine, mrnf.cpp:709/712).
    error_score_max: Option<Tensor<1>>,
}

impl RefineRecord {
    pub(crate) fn new(num_points: u32, device: &Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<1>::zeros([num_points as usize], device),
            vis_weight: Tensor::<1>::zeros([num_points as usize], device),
            max_screen_size: Tensor::<1>::zeros([num_points as usize], device),
            edge_score_sum: None,
            edge_sample_count: 0,
            error_score_max: None,
        }
    }

    pub(crate) fn above_threshold(&self, threshold: f32) -> Tensor<1, Bool> {
        self.refine_weight_norm
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    /// The raw window-MAX error score as a dense `[N]` tensor (zeros if no error
    /// views were sampled): the per-gaussian coverage-weighted MEAN error,
    /// window-MAX'd over the refine window. NOT yet normalized — feed through
    /// [`Self::error_scores_median_normalized`] for the thresholded/sampling
    /// signal.
    pub(crate) fn error_score_max_or_zeros(&self) -> Tensor<1> {
        match &self.error_score_max {
            Some(score) => score.clone(),
            None => Tensor::<1>::zeros(self.refine_weight_norm.dims(), &self.device()),
        }
    }

    /// The growth SAMPLING/threshold signal: the window-MAX coverage-weighted
    /// mean error (`error_score_max_or_zeros`) POSITIVE-MEDIAN normalized so its
    /// median is 1.0 (defect-2 fix). Normalization is applied ONCE here, over the
    /// final MAX distribution — a per-view normalize is defeated by the window-MAX
    /// (see `train::accumulate_error_sample`). On this scale `τ_err = 1.0` selects
    /// the worse-than-median half. Reads back once per refine (host median), like
    /// the edge path; also zeroes any NaN.
    pub(crate) async fn error_scores_median_normalized(&self) -> Tensor<1> {
        let raw = self.error_score_max_or_zeros();
        let n = raw.dims()[0];
        let device = raw.device();
        let mut host: Vec<f32> = raw
            .into_data_async()
            .await
            .expect("error score readback")
            .into_vec()
            .expect("f32 error score");
        crate::edge::normalize_by_positive_median(&mut host);
        Tensor::<1>::from_data(TensorData::new(host, [n]), &device)
    }

    /// Error-map growth gate (MRNF `use_error_map`): `normalized_score > threshold
    /// AND vis_count > 0` (LFS `refine_candidates`, mrnf.cpp:726-727), where the
    /// score is median-normalized ([`Self::error_scores_median_normalized`]). An
    /// all-zero (no-error-views) window normalizes to zeros, so nothing is
    /// admitted — safe (grows nothing from the error signal, no gradient
    /// fallback).
    pub(crate) async fn error_above_threshold(&self, threshold: f32) -> Tensor<1, Bool> {
        self.error_scores_median_normalized()
            .await
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    fn device(&self) -> Device {
        self.refine_weight_norm.device()
    }

    /// Visible splats whose max 2D screen-space extent (as a fraction of the
    /// image dim) exceeds `threshold` — i.e. the "too big on screen" outliers.
    pub(crate) fn above_screen_size(&self, threshold: f32) -> Tensor<1, Bool> {
        self.max_screen_size
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    pub(crate) fn gather_stats(
        &mut self,
        refine_weight: Tensor<1>,
        visible: Tensor<1>,
        screen_radius: Tensor<1>,
    ) {
        let _span = trace_span!("Gather stats").entered();
        self.refine_weight_norm = refine_weight.max_pair(self.refine_weight_norm.clone());
        self.vis_weight = self.vis_weight.clone() + visible;
        self.max_screen_size = screen_radius.max_pair(self.max_screen_size.clone());
    }

    pub(crate) fn gather_aux_stats(&mut self, visible: Tensor<1>, screen_radius: Tensor<1>) {
        let _span = trace_span!("Gather stats").entered();
        self.vis_weight = self.vis_weight.clone() + visible;
        self.max_screen_size = screen_radius.max_pair(self.max_screen_size.clone());
    }

    pub(crate) fn vis_mask(&self) -> Tensor<1, Bool> {
        self.vis_weight.clone().greater_elem(0.0)
    }

    /// Accumulate one sampled view's per-gaussian edge score. `score` is `[N]`,
    /// aligned to the current (constant within a refine window) splat count.
    pub(crate) fn gather_edge(&mut self, score: Tensor<1>) {
        let _span = trace_span!("Gather edge").entered();
        self.edge_score_sum = Some(match self.edge_score_sum.take() {
            Some(sum) => sum + score,
            None => score,
        });
        self.edge_sample_count += 1;
    }

    /// Accumulate one sampled view's per-gaussian error score by window-MAX
    /// (MRNF `use_error_map`, LFS `launch_elementwise_max_inplace`,
    /// mrnf.cpp:602-605). `score` is `[N]`, aligned to the current (constant
    /// within a refine window) splat count. MAX — not SUM — so the window
    /// signal is the single worst view's contribution per gaussian, exactly
    /// mirroring LFS's `_refine_weight_max`.
    pub(crate) fn gather_error(&mut self, score: Tensor<1>) {
        let _span = trace_span!("Gather error").entered();
        // DEFECT-1 FIX: the score is produced by `project_coverage_weighted_mean`,
        // which roots an ISOLATED autodiff feature-backward graph, so its
        // `feats.grad(..)` result comes back autodiff-KIND (a `BackendTensor::
        // Autodiff` bridge tensor). The rest of `RefineRecord` (`vis_weight`,
        // `refine_weight_norm`) lives on the INNER Wgpu backend, and the growth
        // path multiplies this against inner tensors (`above_threshold.float() *
        // growth_base`, train.rs). Storing the autodiff tensor unmodified crossed
        // backends and panicked at first refine ("tensors are not on the same
        // backend"). Detach to inner at the STORE — mirroring the
        // `detach_autodiff(refine_weight)` that keeps `refine_weight_norm` inner.
        // `detach_autodiff` (not `.inner()`) is the checked passthrough: it
        // unwraps the autodiff bridge AND is a no-op on an already-inner tensor,
        // whereas `.inner()` panics on the latter.
        let score = detach_autodiff(score);
        self.error_score_max = Some(match self.error_score_max.take() {
            Some(prev) => score.max_pair(prev),
            None => score,
        });
    }

    /// Per-gaussian edge-guidance factor as a host vector, aligned to the current
    /// splat order (call AFTER `keep`/prune so it matches the post-prune weights).
    /// `None` when no edge samples were gathered. Reads back once per refine — the
    /// weights it multiplies into are already host vectors, so no extra roundtrip.
    pub(crate) async fn edge_factor_host(&self, weight: f32) -> Option<Vec<f32>> {
        if self.edge_sample_count == 0 {
            return None;
        }
        let sum = self.edge_score_sum.clone()?;
        let mean = sum
            .mul_scalar(1.0 / self.edge_sample_count as f32)
            .into_data_async()
            .await
            .ok()?
            .into_vec::<f32>()
            .ok()?;
        Some(crate::edge::edge_guidance_factor(mean, weight))
    }

    pub(crate) fn keep(self, indices: Tensor<1, Int>) -> Self {
        Self {
            refine_weight_norm: self.refine_weight_norm.select(0, indices.clone()),
            vis_weight: self.vis_weight.clone().select(0, indices.clone()),
            edge_score_sum: self.edge_score_sum.map(|s| s.select(0, indices.clone())),
            edge_sample_count: self.edge_sample_count,
            error_score_max: self.error_score_max.map(|s| s.select(0, indices.clone())),
            max_screen_size: self.max_screen_size.select(0, indices),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    async fn values(tensor: Tensor<1>) -> Vec<f32> {
        tensor
            .into_data_async()
            .await
            .expect("readback")
            .into_vec::<f32>()
            .expect("f32 tensor")
    }

    #[tokio::test]
    async fn aux_stats_leave_refine_weight_unchanged() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut record = RefineRecord::new(3, &device);
        record.gather_stats(
            Tensor::from_floats([1.0, 2.0, 3.0], &device),
            Tensor::from_floats([1.0, 0.0, 2.0], &device),
            Tensor::from_floats([0.1, 0.2, 0.3], &device),
        );
        record.gather_aux_stats(
            Tensor::from_floats([0.5, 1.0, 0.0], &device),
            Tensor::from_floats([0.2, 0.1, 0.4], &device),
        );

        assert_eq!(values(record.refine_weight_norm).await, [1.0, 2.0, 3.0]);
        assert_eq!(values(record.vis_weight).await, [1.5, 1.0, 2.0]);
        assert_eq!(values(record.max_screen_size).await, [0.2, 0.2, 0.4]);
    }

    /// Edge accumulation across a splat-count change: `gather_edge` sums two
    /// views elementwise (constant-N invariant), `keep` reindexes the
    /// accumulator through a prune, and `edge_factor_host` averages by the
    /// sample count and normalizes — the factors must track exactly the kept
    /// gaussians in their post-prune order.
    #[tokio::test]
    async fn edge_accumulation_survives_prune_and_averages_by_sample_count() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut record = RefineRecord::new(4, &device);

        // Two sampled views; elementwise `sum + score` requires N == 4 both times.
        record.gather_edge(Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device));
        record.gather_edge(Tensor::from_floats([3.0, 4.0, 5.0, 6.0], &device));
        // sum = [4, 6, 8, 10]; edge_sample_count = 2.

        // Prune gaussian 1: keep indices [0, 2, 3].
        let keep_idx: Tensor<1, Int> =
            Tensor::<1, Int>::from_data(TensorData::new(vec![0i32, 2, 3], [3]), &device);
        let record = record.keep(keep_idx);

        // kept sum = [4, 8, 10]; mean = sum/2 = [2, 4, 5];
        // positive-median = 4 (upper median of {2,4,5});
        // normalized = [0.5, 1.0, 1.25]; *0.25 + 1 = [1.125, 1.25, 1.3125].
        let factors = record
            .edge_factor_host(0.25)
            .await
            .expect("edge factors present after 2 samples");
        assert_eq!(factors.len(), 3, "factors track the 3 kept gaussians");
        let expected = [1.125f32, 1.25, 1.3125];
        for (got, want) in factors.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "factor {got} vs {want}");
        }
    }

    /// No edge samples gathered -> `edge_factor_host` yields `None` (guidance
    /// is a no-op, weights untouched).
    #[tokio::test]
    async fn edge_factor_host_none_without_samples() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let record = RefineRecord::new(3, &device);
        assert!(record.edge_factor_host(0.25).await.is_none());
    }

    async fn bools(t: Tensor<1, Bool>) -> Vec<bool> {
        // Read through int: some wgpu backends store Bool as native (not U32),
        // which a direct `into_vec::<bool>` rejects. `.int()` normalizes it.
        t.int()
            .into_data_async()
            .await
            .expect("readback")
            .into_vec::<i32>()
            .expect("i32 tensor")
            .into_iter()
            .map(|v| v != 0)
            .collect()
    }

    /// T5 (window-MAX semantics): `gather_error` folds views by per-element MAX,
    /// NOT sum/mean. Two views [1,3,2] then [2,1,4] must give [2,3,4] (element
    /// max), distinguishing this from the edge SUM path ([3,4,6]).
    #[tokio::test]
    async fn error_score_accumulates_by_window_max() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut record = RefineRecord::new(3, &device);
        record.gather_error(Tensor::from_floats([1.0, 3.0, 2.0], &device));
        record.gather_error(Tensor::from_floats([2.0, 1.0, 4.0], &device));
        assert_eq!(
            values(record.error_score_max_or_zeros()).await,
            [2.0, 3.0, 4.0]
        );
    }

    /// T5b (window-boundary reset): a fresh `RefineRecord` (which is what the
    /// trainer builds after every refine, since the old one is `take`n) carries
    /// no error score — so the window-MAX cannot leak across refine windows.
    #[tokio::test]
    async fn fresh_record_has_no_error_score() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let record = RefineRecord::new(3, &device);
        // Zeros (not the prior window's max), so `error_above_threshold` admits
        // nothing until this window accumulates its own views.
        assert_eq!(
            values(record.error_score_max_or_zeros()).await,
            [0.0, 0.0, 0.0]
        );
        assert_eq!(
            bools(record.error_above_threshold(1.0).await).await,
            [false, false, false]
        );
    }

    /// T7 (threshold gate): the candidate set is exactly `median_normalized_score
    /// > τ AND vis_count > 0`. Positive-median normalization sends the (upper)
    /// median to 1.0, so at `τ = 1.0` only strictly-above-median gaussians are
    /// candidates — AND only if visible. Scores [1,2,3,4,5] → upper median 3 →
    /// normalized [⅓,⅔,1,4⁄3,5⁄3]; g3,g4 exceed 1.0, but g3 is invisible, so only
    /// g4 is admitted (below-median g0,g1 and at-median g2 excluded).
    #[tokio::test]
    async fn error_above_threshold_is_score_and_visibility() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut record = RefineRecord::new(5, &device);
        // Visibility: all but g3 visible.
        record.gather_aux_stats(
            Tensor::from_floats([1.0, 1.0, 1.0, 0.0, 1.0], &device),
            Tensor::from_floats([0.0, 0.0, 0.0, 0.0, 0.0], &device),
        );
        record.gather_error(Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device));
        assert_eq!(
            bools(record.error_above_threshold(1.0).await).await,
            // g0,g1 below median; g2 at median (=1.0, not >); g3 above but
            // invisible; g4 above and visible.
            [false, false, false, false, true],
        );
    }

    /// T8 (replace-vs-bias): the growth SAMPLING weight is `above_threshold ·
    /// median_normalized_score`, and a subsequent edge-factor multiply is a bias
    /// WITHIN the thresholded set — a gaussian below τ_err (mask 0) stays weight 0
    /// no matter how large its edge factor, so edge can never GROW a gaussian the
    /// error signal did not admit (req 7). Mirrors the train.rs growth path
    /// (`above_threshold.float() * growth_base`, then `*= edge_factor`).
    #[tokio::test]
    async fn edge_factor_never_grows_subthreshold_error_gaussian() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut record = RefineRecord::new(4, &device);
        record.gather_aux_stats(
            Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
            Tensor::from_floats([0.0, 0.0, 0.0, 0.0], &device),
        );
        // [1,2,3,9] → upper median 3 → normalized [⅓,⅔,1,3]; only g3 clears τ=1.0.
        record.gather_error(Tensor::from_floats([1.0, 2.0, 3.0, 9.0], &device));
        let above = record.error_above_threshold(1.0).await;
        let base = above.float() * record.error_scores_median_normalized().await;
        // A huge edge factor on the sub-threshold g0 must not resurrect it.
        let edge_factor = [100.0f32, 1.0, 1.0, 1.0];
        let mut weights = values(base).await;
        for (w, f) in weights.iter_mut().zip(&edge_factor) {
            *w *= f;
        }
        assert_eq!(weights[0], 0.0, "sub-threshold gaussian must stay weight 0");
        assert!(weights[3] > 0.0, "admitted gaussian keeps weight");
    }

    /// T-prune (desync guard): `keep` reindexes `error_score_max` through a
    /// prune via `select(0, indices)` identically to the other accumulators, so
    /// scores stay aligned to gaussian indices after the first prune.
    #[tokio::test]
    async fn error_score_reindexes_through_prune() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut record = RefineRecord::new(4, &device);
        record.gather_error(Tensor::from_floats([10.0, 20.0, 30.0, 40.0], &device));
        // Prune gaussian 1: keep [0, 2, 3].
        let keep_idx: Tensor<1, Int> =
            Tensor::<1, Int>::from_data(TensorData::new(vec![0i32, 2, 3], [3]), &device);
        let record = record.keep(keep_idx);
        assert_eq!(
            values(record.error_score_max_or_zeros()).await,
            [10.0, 30.0, 40.0]
        );
    }
}
