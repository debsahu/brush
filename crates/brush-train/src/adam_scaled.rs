use burn::tensor::{ElementConversion, Tensor};

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
use brush_render::shaders::helpers::ProjectUniforms;
#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
use burn::tensor::Int;

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn use_fused_sh_adam() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled =
            brush_render::native_msl::option_requested(brush_render::native_msl::FUSED_SH_ADAM_ENV);
        if enabled {
            tracing::warn!("experimental native-MSL fused SH Adam enabled");
        }
        enabled
    })
}

/// Adam with per-parameter second-moment reduction (via [`AdamState::reduce_moment_2`])
/// and per-component learning-rate scaling (via [`AdamState::scaling`]).
///
/// Hand-rolled rather than built on burn's `Adam`, which has neither the
/// per-component LR scaling nor the second-moment reduction below. The
/// trainer also edits momentum tensors in place during refine (split/prune)
/// and swaps the transforms LR scaling every step; burn's low-level
/// `Optimizer` trait does now hand back live `Tensor` state rather than only
/// serialized records, so that part is no longer a blocker — the two missing
/// behaviours are what keep this separate.
#[derive(Clone)]
pub(crate) struct AdamScaled {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

/// Per-parameter momentum state. When `reduce_moment_2` is set on the owning
/// [`AdamState`], `moment_2` has size 1 in trailing dims; [`AdamState::map_momentum`]
/// callers must stay shape-agnostic along those.
#[derive(Clone)]
pub(crate) struct MomentumState<const D: usize> {
    pub moment_1: Tensor<D>,
    pub moment_2: Tensor<D>,
    pub time: usize,
}

/// Per-parameter optimizer state.
#[derive(Clone)]
pub(crate) struct AdamState<const D: usize> {
    pub momentum: Option<MomentumState<D>>,
    /// Per-component learning rate scaling (e.g. different LR for means vs
    /// rotations vs scales within the transforms tensor).
    pub scaling: Option<Tensor<D>>,
    /// When true, the second moment is reduced to a scalar per row. Set by the
    /// caller when initializing state for parameters where per-element variance
    /// is not needed.
    pub reduce_moment_2: bool,
}

impl<const D: usize> AdamState<D> {
    pub fn new(scaling: Option<Tensor<D>>, reduce_moment_2: bool) -> Self {
        Self {
            momentum: None,
            scaling,
            reduce_moment_2,
        }
    }

    /// Apply `map_fn` to both momentum tensors (no-op before the first step).
    /// `map_fn` must be shape-agnostic along trailing dims since `moment_2`
    /// may have size-1 trailing dims under `reduce_moment_2`.
    pub fn map_momentum(&mut self, map_fn: impl Fn(Tensor<D>) -> Tensor<D>) {
        self.momentum = self.momentum.take().map(|mut moment| {
            moment.moment_1 = map_fn(moment.moment_1);
            moment.moment_2 = map_fn(moment.moment_2);
            moment
        });
    }
}

// -----------------------------------------------------------------------------
// Native-MSL fused / sparse SH Adam (our addition, kept on top of upstream's
// hand-rolled AdamScaled). The dense fused path is folded into `step` below; the
// sparse deferred path (`step_sparse_sh`) is driven from the trainer when the
// render backward hands back compact SH gradients instead of a dense tensor.
// -----------------------------------------------------------------------------
#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
impl AdamScaled {
    fn sh_adam_config(&self, lr: f64, next_time: usize) -> crate::sh_adam::ShAdamConfig {
        let time = next_time as i32;
        crate::sh_adam::ShAdamConfig {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            bias_correction_1: 1.0 - self.beta_1.powi(time),
            bias_correction_2: 1.0 - self.beta_2.powi(time),
            epsilon: self.epsilon,
            learning_rate: lr as f32,
        }
    }

    /// Pure shape/state predicate for the sparse SH path — it reads nothing
    /// off the optimizer, so it is an associated function rather than a method.
    pub(crate) fn sparse_sh_compatible(param: &Tensor<3>, state: &AdamState<3>) -> bool {
        if !state.reduce_moment_2 {
            return false;
        }
        let [num_splats, coeffs, channels] = param.dims();
        let Some(momentum) = state.momentum.as_ref() else {
            return false;
        };
        let Some(scaling) = state.scaling.as_ref() else {
            return false;
        };
        channels == 3
            && num_splats > 0
            && matches!(coeffs, 1 | 4 | 9 | 16 | 25)
            && momentum.moment_1.dims() == [num_splats, coeffs, 3]
            && momentum.moment_2.dims() == [num_splats, 1, 1]
            && scaling.dims() == [1, coeffs, 1]
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_sparse_sh(
        &self,
        lr: f64,
        param: Tensor<3>,
        render_transforms: Tensor<2>,
        global_from_compact_gid: Tensor<1, Int>,
        compact_grads: Tensor<2>,
        project_uniforms: ProjectUniforms,
        state: AdamState<3>,
    ) -> (Tensor<3>, AdamState<3>) {
        assert!(
            Self::sparse_sh_compatible(&param, &state),
            "sparse SH Adam requires preflighted parameter and optimizer state"
        );
        let momentum = state
            .momentum
            .expect("sparse SH Adam momentum was preflighted");
        let scaling = state
            .scaling
            .expect("sparse SH Adam scaling was preflighted");
        let next_time = momentum.time + 1;
        let config = self.sh_adam_config(lr, next_time);
        let (param, moment_1, moment_2) = crate::sh_adam::sparse_sh_adam(
            param,
            render_transforms,
            global_from_compact_gid,
            compact_grads,
            momentum.moment_1,
            momentum.moment_2,
            scaling.clone(),
            project_uniforms,
            config,
        );
        (
            param,
            AdamState {
                momentum: Some(MomentumState {
                    moment_1,
                    moment_2,
                    time: next_time,
                }),
                scaling: Some(scaling),
                reduce_moment_2: true,
            },
        )
    }
}

impl AdamScaled {
    pub fn new(epsilon: f32) -> Self {
        Self {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon,
        }
    }

    /// One Adam step for a single parameter. `tensor` and `grad` live on the
    /// inner (non-autodiff) backend; `state` is updated in place.
    pub fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        state: &mut AdamState<D>,
    ) -> Tensor<D> {
        // Native-MSL dense fused SH Adam fast path: a single kernel does the
        // moment update + parameter step for the [N, coeffs, 3] SH table. Falls
        // through to the generic path when disabled/unsupported/shape-mismatched.
        #[cfg(all(
            feature = "native-msl",
            target_os = "macos",
            target_arch = "aarch64",
            not(target_family = "wasm")
        ))]
        if use_fused_sh_adam() && state.reduce_moment_2 && D == 3 {
            let shape = tensor.dims();
            let coeffs = shape[1];
            let mut reduced_shape = [1usize; D];
            reduced_shape[0] = shape[0];
            let mut scaling_shape = [1usize; D];
            scaling_shape[1] = coeffs;
            let ready = shape[0] > 0
                && shape[2] == 3
                && matches!(coeffs, 1 | 4 | 9 | 16 | 25)
                && state.momentum.as_ref().is_some_and(|m| {
                    m.moment_1.dims() == shape && m.moment_2.dims() == reduced_shape
                })
                && state
                    .scaling
                    .as_ref()
                    .is_some_and(|s| s.dims() == scaling_shape)
                && crate::sh_adam::fused_sh_adam_supported(&tensor);
            if ready {
                let momentum = state
                    .momentum
                    .as_ref()
                    .expect("fused SH Adam momentum was preflighted");
                let scaling = state
                    .scaling
                    .as_ref()
                    .expect("fused SH Adam scaling was preflighted");
                let next_time = momentum.time + 1;
                let config = self.sh_adam_config(lr, next_time);
                let (out, moment_1, moment_2) = crate::sh_adam::sh_adam(
                    tensor.reshape([shape[0], coeffs, 3]),
                    grad.clone().reshape([shape[0], coeffs, 3]),
                    momentum.moment_1.clone().reshape([shape[0], coeffs, 3]),
                    momentum.moment_2.clone().reshape([shape[0], 1, 1]),
                    scaling.clone().reshape([1, coeffs, 1]),
                    config,
                );
                // `momentum`/`scaling` borrows end above; safe to mutate `state`.
                let new_momentum = MomentumState {
                    moment_1: moment_1.reshape(shape),
                    moment_2: moment_2.reshape(reduced_shape),
                    time: next_time,
                };
                state.momentum = Some(new_momentum);
                return out.reshape(shape);
            }
        }

        let (grad, momentum) = self.transform(grad, state.momentum.take(), state.reduce_moment_2);
        state.momentum = Some(momentum);

        let delta = if let Some(scale) = &state.scaling {
            grad * (scale.clone() * lr).unsqueeze()
        } else {
            grad * lr
        };
        tensor - delta
    }

    fn transform<const D: usize>(
        &self,
        grad: &Tensor<D>,
        momentum_state: Option<MomentumState<D>>,
        reduce_moment_2: bool,
    ) -> (Tensor<D>, MomentumState<D>) {
        let grad_sq = grad.clone().powi_scalar(2);
        let grad_sq_for_moment = if reduce_moment_2 && D > 1 {
            mean_trailing_dims(grad_sq)
        } else {
            grad_sq
        };

        let state = if let Some(mut state) = momentum_state {
            let factor = 1.0 - self.beta_1;
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor));

            let factor = 1.0 - self.beta_2;
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad_sq_for_moment.mul_scalar(factor));

            state.time += 1;
            state
        } else {
            let factor = 1.0 - self.beta_1;
            let moment_1 = grad.clone().mul_scalar(factor);

            let factor = 1.0 - self.beta_2;
            let moment_2 = grad_sq_for_moment.mul_scalar(factor);

            MomentumState {
                moment_1,
                moment_2,
                time: 1,
            }
        };

        let time = (state.time as i32).elem();
        let moment_1_corrected = state
            .moment_1
            .clone()
            .div_scalar(1f32 - self.beta_1.powi(time));
        let moment_2_corrected = state
            .moment_2
            .clone()
            .div_scalar(1f32 - self.beta_2.powi(time));
        // moment_2_corrected broadcasts when it has reduced trailing dims
        let grad = moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));
        (grad, state)
    }
}

/// Reduce to a single mean per row by averaging across all trailing dims (1..D).
/// Result has size 1 in each trailing dim so it broadcasts back to the full shape.
fn mean_trailing_dims<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    debug_assert!(D > 1, "mean_trailing_dims requires D > 1");
    let shape = t.dims();
    let n = shape[0];
    let trailing_count: usize = shape[1..].iter().product();

    // Single flatten + sum avoids one kernel launch per trailing dim.
    let flat: Tensor<2> = t.flatten(1, D - 1);
    let reduced: Tensor<2> = flat.sum_dim(1) / trailing_count as f32;

    let mut target = [1usize; D];
    target[0] = n;
    reduced.reshape(target)
}
