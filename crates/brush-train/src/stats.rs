use burn::{
    prelude::Int,
    tensor::{Bool, Device, Tensor},
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
}

impl RefineRecord {
    pub(crate) fn new(num_points: u32, device: &Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<1>::zeros([num_points as usize], device),
            vis_weight: Tensor::<1>::zeros([num_points as usize], device),
            max_screen_size: Tensor::<1>::zeros([num_points as usize], device),
            edge_score_sum: None,
            edge_sample_count: 0,
        }
    }

    pub(crate) fn above_threshold(&self, threshold: f32) -> Tensor<1, Bool> {
        self.refine_weight_norm
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
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
            max_screen_size: self.max_screen_size.select(0, indices),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
