use burn::{
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::LearningRate,
    optim::{
        SimpleOptimizer,
        adaptor::OptimizerAdaptor,
        decay::{WeightDecay, WeightDecayConfig},
    },
    record::Record,
    tensor::{Device, ElementConversion, Tensor},
};

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

/// Adam with per-parameter second-moment reduction (via [`AdamState::reduce_moment_2`]).
#[derive(Clone)]
pub(crate) struct AdamScaled {
    momentum: AdaptiveMomentum,
    weight_decay: Option<WeightDecay>,
}

#[derive(Config, Debug)]
pub(crate) struct AdamScaledConfig {
    #[config(default = 0.9)]
    beta_1: f32,
    #[config(default = 0.999)]
    beta_2: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    weight_decay: Option<WeightDecayConfig>,
    grad_clipping: Option<GradientClippingConfig>,
}

#[derive(Clone)]
struct AdaptiveMomentum {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

/// Per-parameter momentum state. When `reduce_moment_2` is set on the owning
/// [`AdamState`], `moment_2` has size 1 in trailing dims; `map_opt` callers
/// must stay shape-agnostic along those.
#[derive(Record, Clone)]
pub(crate) struct MomentumState<const D: usize> {
    pub moment_1: Tensor<D>,
    pub moment_2: Tensor<D>,
    pub time: usize,
}

impl<const D: usize> MomentumState<D> {
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(self, device: &Device) -> Self {
        Self {
            moment_1: self.moment_1.to_device(device),
            moment_2: self.moment_2.to_device(device),
            time: self.time,
        }
    }
}

/// Per-parameter optimizer state.
#[derive(Record, Clone)]
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

impl AdamScaledConfig {
    pub(crate) fn init<M: AutodiffModule>(&self) -> OptimizerAdaptor<AdamScaled, M> {
        let optim = AdamScaled {
            momentum: AdaptiveMomentum {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                epsilon: self.epsilon,
            },
            weight_decay: self.weight_decay.as_ref().map(WeightDecay::new),
        };
        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
impl AdamScaled {
    fn sh_adam_config(&self, lr: LearningRate, next_time: usize) -> crate::sh_adam::ShAdamConfig {
        let time = next_time as i32;
        crate::sh_adam::ShAdamConfig {
            beta_1: self.momentum.beta_1,
            beta_2: self.momentum.beta_2,
            bias_correction_1: 1.0 - self.momentum.beta_1.powi(time),
            bias_correction_2: 1.0 - self.momentum.beta_2.powi(time),
            epsilon: self.momentum.epsilon,
            learning_rate: lr as f32,
        }
    }

    pub(crate) fn sparse_sh_compatible(&self, param: &Tensor<3>, state: &AdamState<3>) -> bool {
        if self.weight_decay.is_some() || !state.reduce_moment_2 {
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
        lr: LearningRate,
        param: Tensor<3>,
        render_transforms: Tensor<2>,
        global_from_compact_gid: Tensor<1, Int>,
        compact_grads: Tensor<2>,
        project_uniforms: ProjectUniforms,
        state: AdamState<3>,
    ) -> (Tensor<3>, AdamState<3>) {
        assert!(
            self.sparse_sh_compatible(&param, &state),
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

impl SimpleOptimizer for AdamScaled {
    type State<const D: usize> = AdamState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        mut grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let mut state_momentum = None;
        let mut scaling = None;
        let reduce = state.as_ref().is_some_and(|s| s.reduce_moment_2);

        if let Some(state) = state {
            state_momentum = state.momentum;
            scaling = state.scaling;
        }

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        #[cfg(all(
            feature = "native-msl",
            target_os = "macos",
            target_arch = "aarch64",
            not(target_family = "wasm")
        ))]
        if use_fused_sh_adam()
            && reduce
            && D == 3
            && let (Some(momentum), Some(scaling)) = (state_momentum.as_ref(), scaling.as_ref())
        {
            let shape = tensor.dims();
            let coeffs = shape[1];
            let mut reduced_shape = [1usize; D];
            reduced_shape[0] = shape[0];
            let mut scaling_shape = [1usize; D];
            scaling_shape[1] = coeffs;
            let shapes_match = shape[0] > 0
                && shape[2] == 3
                && matches!(coeffs, 1 | 4 | 9 | 16 | 25)
                && momentum.moment_1.dims() == shape
                && momentum.moment_2.dims() == reduced_shape
                && scaling.dims() == scaling_shape;
            if shapes_match && crate::sh_adam::fused_sh_adam_supported(&tensor) {
                let next_time = momentum.time + 1;
                let config = self.sh_adam_config(lr, next_time);
                let (tensor, moment_1, moment_2) = crate::sh_adam::sh_adam(
                    tensor.reshape([shape[0], coeffs, 3]),
                    grad.reshape([shape[0], coeffs, 3]),
                    momentum.moment_1.clone().reshape([shape[0], coeffs, 3]),
                    momentum.moment_2.clone().reshape([shape[0], 1, 1]),
                    scaling.clone().reshape([1, coeffs, 1]),
                    config,
                );
                return (
                    tensor.reshape(shape),
                    Some(AdamState {
                        momentum: Some(MomentumState {
                            moment_1: moment_1.reshape(shape),
                            moment_2: moment_2.reshape(reduced_shape),
                            time: next_time,
                        }),
                        scaling: Some(scaling.clone()),
                        reduce_moment_2: true,
                    }),
                );
            }
        }

        let (grad, state_momentum) = self.momentum.transform(&grad, state_momentum, reduce);

        let state = AdamState {
            momentum: Some(state_momentum),
            scaling: scaling.clone(),
            reduce_moment_2: reduce,
        };

        let delta = if let Some(scale) = scaling {
            grad * (scale * lr).unsqueeze()
        } else {
            grad * lr
        };

        (tensor - delta, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.map(|m| m.to_device(device));
        state
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

impl AdaptiveMomentum {
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
