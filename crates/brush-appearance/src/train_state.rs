//! Training state and optimizer updates for appearance compensation.
//!
//! Only the active view's bilateral-grid slice is attached to the autodiff
//! graph. The stored grid and its Adam moments stay on the inner backend, so
//! per-step work does not grow with the number of input views. PPISP tensors
//! are small enough to update densely.

use burn::tensor::{Device, Gradients, Tensor, s};

use crate::bilagrid::bilagrid_apply_inner;
use crate::ppisp::{PpispModel, ppisp_apply_inner};
use crate::{AppearanceConfig, BilagridModel, bilagrid_apply, bilagrid_tv_loss};
use brush_render::burn_glue::{detach_autodiff, lift_to_autodiff};

const ADAM_EPS: f64 = 1e-15;
const BILAGRID_LR_WARMUP: u32 = 1000;
const PPISP_LR_WARMUP: u32 = 500;
const LR_START_FACTOR: f64 = 0.01;
const LR_FINAL_FACTOR: f64 = 0.01;

#[cfg(not(target_family = "wasm"))]
const MAX_GRID_STATE_BYTES: u64 = 1024 * 1024 * 1024;
#[cfg(target_family = "wasm")]
const MAX_GRID_STATE_BYTES: u64 = 256 * 1024 * 1024;

fn adam_update<const D: usize>(
    param: Tensor<D>,
    grad: Tensor<D>,
    m1: Tensor<D>,
    m2: Tensor<D>,
    t: i32,
    lr: f64,
    betas: (f64, f64),
) -> (Tensor<D>, Tensor<D>, Tensor<D>) {
    let (b1, b2) = betas;
    let m1 = m1 * b1 + grad.clone() * (1.0 - b1);
    let m2 = m2 * b2 + grad.powi_scalar(2) * (1.0 - b2);
    let m1_hat = m1.clone() / (1.0 - b1.powi(t));
    let m2_hat = m2.clone() / (1.0 - b2.powi(t));
    let param = param - m1_hat / (m2_hat.sqrt() + ADAM_EPS) * lr;
    (param, m1, m2)
}

fn scheduled_lr(step: u32, base: f64, warmup: u32, total_iters: u32) -> f64 {
    crate::warmup_exp_lr(
        step,
        base,
        warmup,
        LR_START_FACTOR,
        LR_FINAL_FACTOR,
        total_iters,
    )
}

struct GridTrainState {
    grids: Option<Tensor<5>>,
    m1: Option<Tensor<5>>,
    m2: Option<Tensor<5>>,
    /// Global step of each view's most recent non-zero-gradient update.
    last_step: Vec<i64>,
    betas: (f64, f64),
}

impl GridTrainState {
    fn new(
        num_views: usize,
        dims: (usize, usize, usize),
        betas: (f64, f64),
        device: &Device,
    ) -> Self {
        let (gx, gy, guidance) = dims;
        let grids = BilagridModel::new(num_views, gx, gy, guidance, device)
            .grids
            .into_value();
        let zeros = || Tensor::zeros(grids.dims(), device);
        Self {
            m1: Some(zeros()),
            m2: Some(zeros()),
            grids: Some(grids),
            last_step: vec![-1; num_views],
            betas,
        }
    }

    fn grids_ref(&self) -> &Tensor<5> {
        self.grids
            .as_ref()
            .expect("grid storage is present between optimizer operations")
    }

    fn view_grid(&self, view_idx: usize) -> Tensor<5> {
        self.grids_ref().clone().slice(s![view_idx..view_idx + 1])
    }

    /// Apply the zero-gradient Adam updates that occurred since this view was
    /// last sampled. This must happen before rendering the current step: the
    /// current gradient must be evaluated at the caught-up parameters.
    fn prepare_view(&mut self, view_idx: usize, global_step: u32, base_lr: f64, total_iters: u32) {
        let last = self.last_step[view_idx];
        if last < 0 || i64::from(global_step) <= last + 1 {
            return;
        }

        let skipped =
            i32::try_from(i64::from(global_step) - last - 1).expect("appearance step gap fits i32");
        let old_t = i32::try_from(last + 1).expect("appearance step fits i32");
        let (b1, b2) = self.betas;

        // With epsilon negligible, all zero-gradient tail updates share the
        // same element-wise m/sqrt(v) direction. Consolidate them into one
        // tensor operation while retaining each historical step's LR and bias
        // correction. This differs from dense Adam only at epsilon scale.
        let mut tail = 0.0f64;
        for k in 1..=skipped {
            let decay = b1.powi(k) / b2.powi(k).sqrt();
            if decay < 1e-14 {
                break;
            }
            let t = old_t + k;
            let step = u32::try_from(last + i64::from(k)).expect("appearance step is non-negative");
            let lr = scheduled_lr(step, base_lr, BILAGRID_LR_WARMUP, total_iters);
            let bias = (1.0 - b2.powi(t)).sqrt() / (1.0 - b1.powi(t));
            tail += lr * decay * bias;
        }

        let range = s![view_idx..view_idx + 1];
        let grids = self.grids.take().expect("grid storage present");
        let m1 = self.m1.take().expect("first moments present");
        let m2 = self.m2.take().expect("second moments present");
        let mut grid = grids.clone().slice(range);
        let m1_view = m1.clone().slice(range);
        let m2_view = m2.clone().slice(range);

        if tail > 0.0 {
            grid = grid - m1_view.clone() / (m2_view.clone().sqrt() + ADAM_EPS) * tail;
        }
        let m1_view = m1_view * b1.powi(skipped);
        let m2_view = m2_view * b2.powi(skipped);

        self.grids = Some(grids.slice_assign(range, grid));
        self.m1 = Some(m1.slice_assign(range, m1_view));
        self.m2 = Some(m2.slice_assign(range, m2_view));
    }

    fn step(&mut self, view_idx: usize, grad: Tensor<5>, lr: f64, global_step: u32) {
        let range = s![view_idx..view_idx + 1];
        let grids = self.grids.take().expect("grid storage present");
        let m1 = self.m1.take().expect("first moments present");
        let m2 = self.m2.take().expect("second moments present");
        let t = i32::try_from(global_step + 1).expect("appearance step fits i32");

        let (grid, m1_view, m2_view) = adam_update(
            grids.clone().slice(range),
            grad,
            m1.clone().slice(range),
            m2.clone().slice(range),
            t,
            lr,
            self.betas,
        );

        self.grids = Some(grids.slice_assign(range, grid));
        self.m1 = Some(m1.slice_assign(range, m1_view));
        self.m2 = Some(m2.slice_assign(range, m2_view));
        self.last_step[view_idx] = i64::from(global_step);
    }
}

struct PpispTrainState {
    model: PpispModel,
    m_exposure: (Tensor<1>, Tensor<1>),
    m_vignetting: (Tensor<3>, Tensor<3>),
    m_color: (Tensor<2>, Tensor<2>),
    m_crf: (Tensor<3>, Tensor<3>),
}

impl PpispTrainState {
    fn new(
        num_cameras: usize,
        num_views: usize,
        camera_indices: Vec<u32>,
        device: &Device,
    ) -> Self {
        let model = PpispModel::new(num_cameras, num_views, camera_indices, device);
        let zeros_like1 = |t: &Tensor<1>| {
            (
                Tensor::zeros(t.dims(), device),
                Tensor::zeros(t.dims(), device),
            )
        };
        let zeros_like2 = |t: &Tensor<2>| {
            (
                Tensor::zeros(t.dims(), device),
                Tensor::zeros(t.dims(), device),
            )
        };
        let zeros_like3 = |t: &Tensor<3>| {
            (
                Tensor::zeros(t.dims(), device),
                Tensor::zeros(t.dims(), device),
            )
        };
        Self {
            m_exposure: zeros_like1(&model.exposure.val()),
            m_vignetting: zeros_like3(&model.vignetting.val()),
            m_color: zeros_like2(&model.color.val()),
            m_crf: zeros_like3(&model.crf.val()),
            model,
        }
    }

    fn lifted(&self) -> PpispModel {
        use burn::module::Param;
        PpispModel {
            exposure: Param::from_tensor(
                lift_to_autodiff(self.model.exposure.val()).require_grad(),
            ),
            vignetting: Param::from_tensor(
                lift_to_autodiff(self.model.vignetting.val()).require_grad(),
            ),
            color: Param::from_tensor(lift_to_autodiff(self.model.color.val()).require_grad()),
            crf: Param::from_tensor(lift_to_autodiff(self.model.crf.val()).require_grad()),
            camera_indices: self.model.camera_indices.clone(),
        }
    }

    fn step(&mut self, lifted: &PpispModel, grads: &mut Gradients, t: i32, lr: f64) {
        use burn::module::Param;

        fn update<const D: usize>(
            param: &mut Param<Tensor<D>>,
            moments: &mut (Tensor<D>, Tensor<D>),
            lifted: &Param<Tensor<D>>,
            grads: &mut Gradients,
            t: i32,
            lr: f64,
        ) {
            let Some(grad) = lifted.val().grad_remove(grads) else {
                return;
            };
            let (param_value, m1, m2) = adam_update(
                param.val(),
                detach_autodiff(grad),
                moments.0.clone(),
                moments.1.clone(),
                t,
                lr,
                (0.9, 0.999),
            );
            *moments = (m1, m2);
            *param = Param::from_tensor(param_value);
        }

        update(
            &mut self.model.exposure,
            &mut self.m_exposure,
            &lifted.exposure,
            grads,
            t,
            lr,
        );
        update(
            &mut self.model.vignetting,
            &mut self.m_vignetting,
            &lifted.vignetting,
            grads,
            t,
            lr,
        );
        update(
            &mut self.model.color,
            &mut self.m_color,
            &lifted.color,
            grads,
            t,
            lr,
        );
        update(
            &mut self.model.crf,
            &mut self.m_crf,
            &lifted.crf,
            grads,
            t,
            lr,
        );
    }
}

pub struct AppearanceTrainState {
    config: AppearanceConfig,
    total_iters: u32,
    step: u32,
    grid: Option<GridTrainState>,
    ppisp: Option<PpispTrainState>,
}

pub struct ActiveAppearance {
    view_idx: usize,
    view_grid: Option<Tensor<5>>,
    ppisp: Option<PpispModel>,
    tv_weight: f32,
    reg_scale: f32,
}

impl AppearanceTrainState {
    pub fn new(
        config: AppearanceConfig,
        camera_indices: Vec<u32>,
        total_iters: u32,
        device: &Device,
    ) -> Result<Option<Self>, String> {
        if !config.bilagrid && !config.ppisp {
            return Ok(None);
        }
        if camera_indices.is_empty() {
            return Err("appearance compensation requires at least one training view".to_owned());
        }
        if total_iters == 0 {
            return Err(
                "appearance compensation requires at least one training iteration".to_owned(),
            );
        }
        validate_config(&config, camera_indices.len())?;

        let device = device.clone().inner();
        let num_views = camera_indices.len();
        let num_cameras = camera_indices.iter().copied().max().unwrap_or(0) as usize + 1;
        let grid = config.bilagrid.then(|| {
            GridTrainState::new(
                num_views,
                config.bilagrid_dims,
                config.bilagrid_betas,
                &device,
            )
        });
        let ppisp = config
            .ppisp
            .then(|| PpispTrainState::new(num_cameras, num_views, camera_indices, &device));

        Ok(Some(Self {
            config,
            total_iters,
            step: 0,
            grid,
            ppisp,
        }))
    }

    pub fn begin_step(&mut self, view_idx: usize) -> ActiveAppearance {
        if let Some(grid) = self.grid.as_mut() {
            grid.prepare_view(
                view_idx,
                self.step,
                self.config.bilagrid_lr,
                self.total_iters,
            );
        }
        ActiveAppearance {
            view_idx,
            view_grid: self
                .grid
                .as_ref()
                .map(|grid| lift_to_autodiff(grid.view_grid(view_idx)).require_grad()),
            ppisp: self.ppisp.as_ref().map(PpispTrainState::lifted),
            tv_weight: self.config.bilagrid_tv_weight,
            reg_scale: self.config.ppisp_reg_scale,
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn end_step(&mut self, active: ActiveAppearance, grads: &mut Gradients) {
        if let (Some(state), Some(view_grid)) = (self.grid.as_mut(), &active.view_grid)
            && let Some(grad) = view_grid.clone().grad_remove(grads)
        {
            let lr = scheduled_lr(
                self.step,
                self.config.bilagrid_lr,
                BILAGRID_LR_WARMUP,
                self.total_iters,
            );
            state.step(active.view_idx, detach_autodiff(grad), lr, self.step);
        }
        if let (Some(state), Some(lifted)) = (self.ppisp.as_mut(), &active.ppisp) {
            let lr = scheduled_lr(
                self.step,
                self.config.ppisp_lr,
                PPISP_LR_WARMUP,
                self.total_iters,
            );
            let t = i32::try_from(self.step + 1).expect("appearance step fits i32");
            state.step(lifted, grads, t, lr);
        }
        self.step = self.step.saturating_add(1);
    }

    pub async fn stats(&self) -> Option<String> {
        let read = |tensor: Tensor<1>| async move {
            tensor
                .into_scalar_async::<f32>()
                .await
                .expect("appearance stats readback")
        };
        let mut parts = Vec::new();
        if let Some(grid) = &self.grid {
            let values = grid.grids_ref().clone();
            let lo = read(values.clone().min()).await;
            let hi = read(values.max()).await;
            parts.push(format!("grid range [{lo:.3}, {hi:.3}]"));
        }
        if let Some(ppisp) = &self.ppisp {
            let exposure = ppisp.model.exposure.val();
            let lo = read(exposure.clone().min()).await;
            let hi = read(exposure.max()).await;
            let vignetting = read(ppisp.model.vignetting.val().abs().max()).await;
            parts.push(format!(
                "exposure [{lo:+.3}, {hi:+.3}] stops, vignetting |max| {vignetting:.3}"
            ));
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }

    pub fn apply_eval(&self, img: Tensor<3>, view_idx: usize) -> Tensor<3> {
        let img = match &self.ppisp {
            Some(state) => {
                let model = &state.model;
                let camera_idx = model.camera_indices[view_idx] as usize;
                ppisp_apply_inner(
                    model.exposure.val(),
                    model.vignetting.val(),
                    model.color.val(),
                    model.crf.val(),
                    img,
                    camera_idx,
                    view_idx,
                    crate::PpispStages::ALL,
                )
            }
            None => img,
        };
        match &self.grid {
            Some(grid) => bilagrid_apply_inner(grid.grids_ref().clone(), img, view_idx),
            None => img,
        }
    }
}

impl ActiveAppearance {
    pub fn apply(&self, img: Tensor<3>) -> Tensor<3> {
        let img = match &self.ppisp {
            Some(ppisp) => ppisp.apply(img, self.view_idx),
            None => img,
        };
        match &self.view_grid {
            Some(grid) => bilagrid_apply(grid.clone(), img, 0),
            None => img,
        }
    }

    pub fn reg_loss(&self) -> Option<Tensor<1>> {
        let mut loss: Option<Tensor<1>> = None;
        if let Some(grid) = &self.view_grid
            && self.tv_weight > 0.0
        {
            loss = Some(bilagrid_tv_loss(grid.clone()) * self.tv_weight);
        }
        if let Some(ppisp) = &self.ppisp
            && self.reg_scale > 0.0
        {
            let regularization = ppisp.reg_loss() * self.reg_scale;
            loss = Some(match loss {
                Some(grid_loss) => grid_loss + regularization,
                None => regularization,
            });
        }
        loss
    }
}

fn validate_config(config: &AppearanceConfig, num_views: usize) -> Result<(), String> {
    let (gx, gy, guidance) = config.bilagrid_dims;
    if config.bilagrid && [gx, gy, guidance].iter().any(|dim| *dim < 2) {
        return Err(format!(
            "bilagrid-dims must each be at least 2 (got {gx},{gy},{guidance})"
        ));
    }
    let (beta1, beta2) = config.bilagrid_betas;
    if config.bilagrid
        && (!beta1.is_finite()
            || !beta2.is_finite()
            || !(0.0..1.0).contains(&beta1)
            || !(0.0..1.0).contains(&beta2))
    {
        return Err("bilagrid-betas must be finite values in [0, 1)".to_owned());
    }
    if config.bilagrid && (!config.bilagrid_lr.is_finite() || config.bilagrid_lr <= 0.0) {
        return Err("bilagrid-lr must be finite and greater than zero".to_owned());
    }
    if !config.bilagrid_tv_weight.is_finite() || config.bilagrid_tv_weight < 0.0 {
        return Err("bilagrid-tv-weight must be finite and non-negative".to_owned());
    }
    if config.ppisp && (!config.ppisp_lr.is_finite() || config.ppisp_lr <= 0.0) {
        return Err("ppisp-lr must be finite and greater than zero".to_owned());
    }
    if !config.ppisp_reg_scale.is_finite() || config.ppisp_reg_scale < 0.0 {
        return Err("ppisp-reg-scale must be finite and non-negative".to_owned());
    }

    if config.bilagrid {
        let elements = u64::try_from(num_views)
            .ok()
            .and_then(|n| n.checked_mul(12))
            .and_then(|n| n.checked_mul(gx as u64))
            .and_then(|n| n.checked_mul(gy as u64))
            .and_then(|n| n.checked_mul(guidance as u64))
            .ok_or_else(|| "bilateral-grid dimensions overflow the addressable size".to_owned())?;
        // Parameters plus first and second Adam moments, all f32.
        let bytes = elements
            .checked_mul(3 * size_of::<f32>() as u64)
            .ok_or_else(|| "bilateral-grid state size overflows u64".to_owned())?;
        if bytes > MAX_GRID_STATE_BYTES {
            return Err(format!(
                "bilateral-grid state would allocate {:.1} MiB, above the {:.0} MiB safety limit; reduce views or --bilagrid-dims",
                bytes as f64 / (1024.0 * 1024.0),
                MAX_GRID_STATE_BYTES as f64 / (1024.0 * 1024.0),
            ));
        }
    }
    Ok(())
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lazy_grid_adam_matches_dense_adam_with_scheduled_lr() {
        let device = Device::from(brush_cube::test_helpers::test_device().await);
        let betas = (0.9f64, 0.999f64);
        let mut state = GridTrainState::new(3, (2, 2, 2), betas, &device);
        let base_lr = 2e-3;
        let total_iters = 500;
        let gap = 40u32;
        let visits = 10;

        let (mut param, mut m1, mut m2) = (1.0f64, 0.0f64, 0.0f64);
        for step in 0..=gap * (visits - 1) {
            let gradient = if step % gap == 0 {
                let value = 1.0 + 0.5 * f64::sin(f64::from(step));
                state.prepare_view(0, step, base_lr, total_iters);
                state.step(
                    0,
                    Tensor::<5>::full([1, 12, 2, 2, 2], value as f32, &device),
                    scheduled_lr(step, base_lr, BILAGRID_LR_WARMUP, total_iters),
                    step,
                );
                value
            } else {
                0.0
            };
            m1 = betas.0 * m1 + (1.0 - betas.0) * gradient;
            m2 = betas.1 * m2 + (1.0 - betas.1) * gradient * gradient;
            let t = f64::from(step + 1);
            let m1_hat = m1 / (1.0 - betas.0.powf(t));
            let m2_hat = m2 / (1.0 - betas.1.powf(t));
            let lr = scheduled_lr(step, base_lr, BILAGRID_LR_WARMUP, total_iters);
            param -= lr * m1_hat / (m2_hat.sqrt() + ADAM_EPS);
        }

        let got = f64::from(
            state
                .view_grid(0)
                .into_data_async()
                .await
                .expect("readback")
                .to_vec::<f32>()
                .expect("f32 data")[0],
        );
        let relative_error = (got - param).abs() / param.abs().max(1e-12);
        assert!(
            relative_error < 0.02,
            "lazy update {got:.6} vs dense Adam {param:.6} (relative error {relative_error:.3})"
        );
    }

    #[test]
    fn rejects_oversized_grid_state() {
        let config = AppearanceConfig {
            bilagrid: true,
            bilagrid_dims: (usize::MAX, 16, 8),
            ..AppearanceConfig::default()
        };
        assert!(validate_config(&config, 2).is_err());
    }
}
