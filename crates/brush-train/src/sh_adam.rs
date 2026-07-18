//! Native-MSL fused Adam update for spherical-harmonic coefficients.
//!
//! Each Apple 32-lane SIMD group owns one splat. It reduces the full SH
//! gradient row to the scalar second moment, then updates the full first
//! moment and parameter row without materialising intermediate tensors.

use brush_cube::{MainBackend as Wgpu, MainBackendBase, calc_cube_count_1d};
use brush_render::shaders::helpers::ProjectUniforms;
use burn::{
    Tensor,
    backend::{
        Backend, Dispatch, DispatchTensor, DispatchTensorKind, ExtensionType, TensorMetadata,
        ops::IntTensorOps,
        tensor::{FloatTensor, IntTensor},
        wgpu::{AutoCompiler, WgpuRuntime},
    },
    tensor::{DType, Int, IntDType, Shape},
};
use burn_cubecl::{
    CubeRuntime,
    cubecl::{Runtime, features::Plane},
    fusion::FusionCubeRuntime,
    kernel::into_contiguous,
    tensor::CubeTensor,
};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

const PLANE_SIZE: u32 = 32;
const WORKGROUP_SIZE: u32 = 256;
const SPLATS_PER_WORKGROUP: u32 = WORKGROUP_SIZE / PLANE_SIZE;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShAdamConfig {
    pub beta_1: f32,
    pub beta_2: f32,
    pub bias_correction_1: f32,
    pub bias_correction_2: f32,
    pub epsilon: f32,
    pub learning_rate: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SparseShRenderConfig {
    camera_position: [f32; 3],
    sh_degree: u32,
    total_splats: u32,
    num_visible: u32,
}

impl From<ProjectUniforms> for SparseShRenderConfig {
    fn from(value: ProjectUniforms) -> Self {
        Self {
            camera_position: [
                value.camera_position[0],
                value.camera_position[1],
                value.camera_position[2],
            ],
            sh_degree: value.sh_degree,
            total_splats: value.total_splats,
            num_visible: value.num_visible,
        }
    }
}

#[derive(Debug, Clone, ExtensionType)]
pub(crate) struct ShAdamOutput<B: Backend> {
    pub param: FloatTensor<B>,
    pub moment_1: FloatTensor<B>,
    pub moment_2: FloatTensor<B>,
}

#[burn::backend::backend_extension(Wgpu)]
pub(crate) trait ShAdamOps: Backend {
    fn sh_adam(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self>;

    #[allow(clippy::too_many_arguments)]
    fn sparse_sh_adam(
        param: FloatTensor<Self>,
        render_transforms: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        compact_grads: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        render: SparseShRenderConfig,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self>;
}

pub(crate) fn sh_adam(
    param: Tensor<3>,
    grad: Tensor<3>,
    moment_1: Tensor<3>,
    moment_2: Tensor<3>,
    scaling: Tensor<3>,
    config: ShAdamConfig,
) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
    let output = <Dispatch as ShAdamOps>::sh_adam(
        param.into_dispatch(),
        grad.into_dispatch(),
        moment_1.into_dispatch(),
        moment_2.into_dispatch(),
        scaling.into_dispatch(),
        config,
    );
    (
        Tensor::from_dispatch(output.param),
        Tensor::from_dispatch(output.moment_1),
        Tensor::from_dispatch(output.moment_2),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_sh_adam(
    param: Tensor<3>,
    render_transforms: Tensor<2>,
    global_from_compact_gid: Tensor<1, Int>,
    compact_grads: Tensor<2>,
    moment_1: Tensor<3>,
    moment_2: Tensor<3>,
    scaling: Tensor<3>,
    project_uniforms: ProjectUniforms,
    config: ShAdamConfig,
) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
    fn require_inner(name: &str, tensor: &DispatchTensor) {
        assert!(
            !matches!(&tensor.kind, DispatchTensorKind::Autodiff(_)),
            "sparse SH Adam {name} must be on the inner backend"
        );
    }
    let param = param.into_dispatch();
    let render_transforms = render_transforms.into_dispatch();
    let global_from_compact_gid = global_from_compact_gid.into_dispatch();
    let compact_grads = compact_grads.into_dispatch();
    let moment_1 = moment_1.into_dispatch();
    let moment_2 = moment_2.into_dispatch();
    let scaling = scaling.into_dispatch();
    for (name, tensor) in [
        ("parameter", &param),
        ("render transforms", &render_transforms),
        ("compact gradients", &compact_grads),
        ("moment_1", &moment_1),
        ("moment_2", &moment_2),
        ("scaling", &scaling),
    ] {
        require_inner(name, tensor);
    }
    require_inner("compact-to-global map", &global_from_compact_gid);
    let output = <Dispatch as ShAdamOps>::sparse_sh_adam(
        param,
        render_transforms,
        global_from_compact_gid,
        compact_grads,
        moment_1,
        moment_2,
        scaling,
        project_uniforms.into(),
        config,
    );
    (
        Tensor::from_dispatch(output.param),
        Tensor::from_dispatch(output.moment_1),
        Tensor::from_dispatch(output.moment_2),
    )
}

fn sh_adam_device_supported<const D: usize>(
    param: &Tensor<D>,
    require_non_uniform_control_flow: bool,
) -> bool {
    if u32::try_from(param.shape().num_elements()).is_err() || param.dtype() != DType::F32 {
        return false;
    }
    // Query the adapter that owns this tensor. Brush normally uses the default
    // device, but callers can construct splats on another Wgpu adapter; a
    // process-global capability cache could approve an SH path for the wrong
    // device.
    let param = brush_render::burn_glue::unwrap_wgpu_float(param.clone());
    let client = WgpuRuntime::<AutoCompiler>::client(param.client.device());
    let properties = client.properties();
    let features = client.features();
    features.plane.contains(Plane::Ops)
        && (!require_non_uniform_control_flow
            || features.plane.contains(Plane::NonUniformControlFlow))
        && properties.hardware.plane_size_min == PLANE_SIZE
        && properties.hardware.plane_size_max == PLANE_SIZE
        && properties.hardware.max_units_per_cube >= WORKGROUP_SIZE
        && properties.hardware.max_cube_dim.0 >= WORKGROUP_SIZE
}

pub(crate) fn fused_sh_adam_supported<const D: usize>(param: &Tensor<D>) -> bool {
    sh_adam_device_supported(param, false)
}

pub(crate) fn sparse_sh_adam_supported(param: &Tensor<3>) -> bool {
    let [num_splats, coeffs, channels] = param.dims();
    num_splats > 0
        && channels == 3
        && matches!(coeffs, 1 | 4 | 9 | 16 | 25)
        && sh_adam_device_supported(param, true)
}

mod kernel {
    use brush_render::kernels::sh::{num_sh_coeffs, sh_basis, sh_color_component};
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::cube;
    use burn_cubecl::cubecl::prelude::*;

    use super::{PLANE_SIZE, SPLATS_PER_WORKGROUP};

    #[allow(clippy::too_many_arguments)]
    #[cube]
    fn update_element(
        param: &Tensor<f32>,
        moment_1: &Tensor<f32>,
        scaling: &Tensor<f32>,
        out_param: &mut Tensor<f32>,
        out_moment_1: &mut Tensor<f32>,
        row_base: u32,
        element: u32,
        grad: f32,
        update_factor: f32,
        beta_1: f32,
    ) {
        let index = (row_base + element) as usize;
        let new_moment_1 = moment_1[index] * beta_1 + grad * (1.0f32 - beta_1);
        let coeff_scale = scaling[(element / 3u32) as usize];
        out_moment_1[index] = new_moment_1;
        out_param[index] = param[index] - new_moment_1 * update_factor * coeff_scale;
    }

    /// Invert the compact-to-global map only for rows with a non-zero color
    /// gradient. A zero sentinel means the global row follows exact
    /// zero-gradient Adam semantics.
    #[cube(launch)]
    pub fn build_compact_sh_map_kernel(
        global_from_compact_gid: &Tensor<u32>,
        compact_grads: &Tensor<f32>,
        compact_plus_one_from_global: &mut Tensor<u32>,
        num_visible: u32,
    ) {
        let compact_gid = ABSOLUTE_POS as u32;
        if compact_gid >= num_visible {
            terminate!();
        }
        let grad_base = (compact_gid * 10u32) as usize;
        let r = compact_grads[grad_base + 5];
        let g = compact_grads[grad_base + 6];
        let b = compact_grads[grad_base + 7];
        if r != 0.0f32 || g != 0.0f32 || b != 0.0f32 {
            let global_gid = global_from_compact_gid[compact_gid as usize];
            compact_plus_one_from_global[global_gid as usize] = compact_gid + 1u32;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cube(launch, launch_unchecked)]
    pub fn sparse_sh_adam_kernel(
        param: &Tensor<f32>,
        render_transforms: &Tensor<f32>,
        compact_plus_one_from_global: &Tensor<u32>,
        compact_grads: &Tensor<f32>,
        moment_1: &Tensor<f32>,
        moment_2: &Tensor<f32>,
        scaling: &Tensor<f32>,
        out_param: &mut Tensor<f32>,
        out_moment_1: &mut Tensor<f32>,
        out_moment_2: &mut Tensor<f32>,
        num_splats: u32,
        camera_x: f32,
        camera_y: f32,
        camera_z: f32,
        beta_1: f32,
        beta_2: f32,
        bias_correction_1: f32,
        bias_correction_2: f32,
        epsilon: f32,
        learning_rate: f32,
        #[comptime] sh_degree: u32,
    ) {
        let global_gid = CUBE_POS as u32 * SPLATS_PER_WORKGROUP + PLANE_POS;
        let lane = UNIT_POS_PLANE;
        let active = global_gid < num_splats;
        let row_len = comptime![num_sh_coeffs(sh_degree) * 3u32];
        let row_base = global_gid * row_len;
        let index_0 = lane;
        let index_1 = lane + PLANE_SIZE;
        let index_2 = lane + 2u32 * PLANE_SIZE;

        let mut compact_plus_one = 0u32;
        if active && lane == 0u32 {
            compact_plus_one = compact_plus_one_from_global[global_gid as usize];
        }
        compact_plus_one = plane_broadcast(compact_plus_one, 0u32);
        let has_grad = compact_plus_one > 0u32;

        let mut grad_0 = 0.0f32;
        let mut grad_1 = 0.0f32;
        let mut grad_2 = 0.0f32;
        if active && has_grad {
            let compact_gid = compact_plus_one - 1u32;
            let transform_base = (global_gid * 10u32) as usize;
            let grad_base = (compact_gid * 10u32) as usize;
            let mut field = 0.0f32;
            if lane == 0u32 {
                field = render_transforms[transform_base];
            } else if lane == 1u32 {
                field = render_transforms[transform_base + 1];
            } else if lane == 2u32 {
                field = render_transforms[transform_base + 2];
            } else if lane == 3u32 {
                field = compact_grads[grad_base + 5];
            } else if lane == 4u32 {
                field = compact_grads[grad_base + 6];
            } else if lane == 5u32 {
                field = compact_grads[grad_base + 7];
            }
            let mean_x = plane_broadcast(field, 0u32);
            let mean_y = plane_broadcast(field, 1u32);
            let mean_z = plane_broadcast(field, 2u32);
            let v_color_r = plane_broadcast(field, 3u32);
            let v_color_g = plane_broadcast(field, 4u32);
            let v_color_b = plane_broadcast(field, 5u32);

            let dx = mean_x - camera_x;
            let dy = mean_y - camera_y;
            let dz = mean_z - camera_z;
            let inv_len = 1.0f32 / f32::sqrt(dx * dx + dy * dy + dz * dz);
            let view_x = dx * inv_len;
            let view_y = dy * inv_len;
            let view_z = dz * inv_len;
            let basis = sh_basis(lane, sh_degree, view_x, view_y, view_z);
            // Every lane must participate in each shuffle; only the result
            // assignment is row-length guarded. Diverging around a plane
            // intrinsic would leave valid degree-3/4 tail elements undefined.
            let basis_0 = plane_shuffle(basis, index_0 / 3u32);
            let basis_1 = plane_shuffle(basis, index_1 / 3u32);
            let basis_2 = plane_shuffle(basis, index_2 / 3u32);

            if index_0 < row_len {
                grad_0 = basis_0 * sh_color_component(index_0, v_color_r, v_color_g, v_color_b);
            }
            if index_1 < row_len {
                grad_1 = basis_1 * sh_color_component(index_1, v_color_r, v_color_g, v_color_b);
            }
            if index_2 < row_len {
                grad_2 = basis_2 * sh_color_component(index_2, v_color_r, v_color_g, v_color_b);
            }
        }

        let sum_sq = plane_sum(grad_0 * grad_0 + grad_1 * grad_1 + grad_2 * grad_2);
        let mean_sq = sum_sq / row_len as f32;
        let mut update_factor = 0.0f32;
        if active && lane == 0u32 {
            let new_moment_2 = moment_2[global_gid as usize] * beta_2 + mean_sq * (1.0f32 - beta_2);
            out_moment_2[global_gid as usize] = new_moment_2;
            update_factor = learning_rate
                / bias_correction_1
                / (f32::sqrt(new_moment_2 / bias_correction_2) + epsilon);
        }
        update_factor = plane_broadcast(update_factor, 0u32);

        if active && index_0 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_0,
                grad_0,
                update_factor,
                beta_1,
            );
        }
        if active && index_1 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_1,
                grad_1,
                update_factor,
                beta_1,
            );
        }
        if active && index_2 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_2,
                grad_2,
                update_factor,
                beta_1,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cube(launch, launch_unchecked)]
    pub fn sh_adam_kernel(
        param: &Tensor<f32>,
        grad: &Tensor<f32>,
        moment_1: &Tensor<f32>,
        moment_2: &Tensor<f32>,
        scaling: &Tensor<f32>,
        out_param: &mut Tensor<f32>,
        out_moment_1: &mut Tensor<f32>,
        out_moment_2: &mut Tensor<f32>,
        num_splats: u32,
        beta_1: f32,
        beta_2: f32,
        bias_correction_1: f32,
        bias_correction_2: f32,
        epsilon: f32,
        learning_rate: f32,
        #[comptime] row_len: u32,
    ) {
        let splat_id = CUBE_POS as u32 * SPLATS_PER_WORKGROUP + PLANE_POS;
        let active = splat_id < num_splats;
        let lane = UNIT_POS_PLANE;
        let row_base = splat_id * row_len;

        // Degree 4 has 75 values per splat, so each lane owns at most three.
        // Keep them in registers across the plane reduction to avoid rereading
        // the 222 MB gradient tensor during the update phase.
        let index_0 = lane;
        let index_1 = lane + PLANE_SIZE;
        let index_2 = lane + 2u32 * PLANE_SIZE;
        let mut grad_0 = 0.0f32;
        let mut grad_1 = 0.0f32;
        let mut grad_2 = 0.0f32;
        if active && index_0 < row_len {
            grad_0 = grad[(row_base + index_0) as usize];
        }
        if active && index_1 < row_len {
            grad_1 = grad[(row_base + index_1) as usize];
        }
        if active && index_2 < row_len {
            grad_2 = grad[(row_base + index_2) as usize];
        }

        let sum_sq = plane_sum(grad_0 * grad_0 + grad_1 * grad_1 + grad_2 * grad_2);
        let mean_sq = sum_sq / row_len as f32;

        let mut update_factor = 0.0f32;
        if active && lane == 0u32 {
            let new_moment_2 = moment_2[splat_id as usize] * beta_2 + mean_sq * (1.0f32 - beta_2);
            out_moment_2[splat_id as usize] = new_moment_2;
            update_factor = learning_rate
                / bias_correction_1
                / (f32::sqrt(new_moment_2 / bias_correction_2) + epsilon);
        }
        update_factor = plane_broadcast(update_factor, 0u32);

        if active && index_0 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_0,
                grad_0,
                update_factor,
                beta_1,
            );
        }
        if active && index_1 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_1,
                grad_1,
                update_factor,
                beta_1,
            );
        }
        if active && index_2 < row_len {
            update_element(
                param,
                moment_1,
                scaling,
                out_param,
                out_moment_1,
                row_base,
                index_2,
                grad_2,
                update_factor,
                beta_1,
            );
        }
    }
}

fn empty_like<R: CubeRuntime>(template: &CubeTensor<R>) -> CubeTensor<R> {
    let shape = Shape::from(template.shape().as_slice().to_vec());
    let buffer = template
        .client
        .empty(shape.num_elements() * template.dtype.size());
    CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        shape,
        buffer,
        template.dtype,
    )
}

impl ShAdamOps for MainBackendBase {
    fn sh_adam(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self> {
        param.assert_is_on_same_device(&grad);
        param.assert_is_on_same_device(&moment_1);
        param.assert_is_on_same_device(&moment_2);
        param.assert_is_on_same_device(&scaling);

        let param = into_contiguous(param);
        let grad = into_contiguous(grad);
        let moment_1 = into_contiguous(moment_1);
        let moment_2 = into_contiguous(moment_2);
        let scaling = into_contiguous(scaling);

        for (name, tensor) in [
            ("parameter", &param),
            ("gradient", &grad),
            ("moment_1", &moment_1),
            ("moment_2", &moment_2),
            ("scaling", &scaling),
        ] {
            assert_eq!(tensor.dtype, DType::F32, "fused SH Adam {name} must be f32");
        }

        let shape = param.shape();
        let dims = shape.as_slice();
        assert_eq!(dims.len(), 3, "fused SH Adam expects [N, C, 3]");
        assert_eq!(dims[2], 3, "fused SH Adam expects RGB coefficients");
        assert_eq!(grad.shape(), shape, "gradient shape must match parameter");
        assert_eq!(
            moment_1.shape(),
            shape,
            "moment_1 shape must match parameter"
        );
        assert_eq!(
            moment_2.shape().as_slice(),
            &[dims[0], 1, 1],
            "moment_2 must be scalar per splat"
        );
        assert_eq!(
            scaling.shape().as_slice(),
            &[1, dims[1], 1],
            "scaling must be one value per SH coefficient"
        );
        assert!(dims[0] > 0, "fused SH Adam requires at least one splat");
        assert!(dims[1] <= 25, "fused SH Adam supports degrees 0 through 4");
        assert!(
            u32::try_from(shape.num_elements()).is_ok(),
            "fused SH Adam flattened indices must fit in u32"
        );

        let properties = param.client.properties();
        assert!(
            param.client.features().plane.contains(Plane::Ops),
            "fused SH Adam requires plane operations"
        );
        assert_eq!(
            properties.hardware.plane_size_min, PLANE_SIZE,
            "fused SH Adam requires 32-lane planes"
        );
        assert_eq!(
            properties.hardware.plane_size_max, PLANE_SIZE,
            "fused SH Adam requires a fixed plane size"
        );
        assert!(
            properties.hardware.max_units_per_cube >= WORKGROUP_SIZE,
            "fused SH Adam requires 256 units per workgroup"
        );
        assert!(
            properties.hardware.max_cube_dim.0 >= WORKGROUP_SIZE,
            "fused SH Adam requires a 256-wide workgroup"
        );

        let num_splats = u32::try_from(dims[0]).expect("splat count exceeds u32");
        let row_len = u32::try_from(dims[1] * dims[2]).expect("SH row exceeds u32");
        let workgroups = calc_cube_count_1d(num_splats, SPLATS_PER_WORKGROUP);
        let out_param = empty_like(&param);
        let out_moment_1 = empty_like(&moment_1);
        let out_moment_2 = empty_like(&moment_2);
        let client = param.client.clone();

        // SAFETY: all inputs are contiguous and shape-checked above. Every
        // active plane owns one row, its row indices are below N*C*3, its
        // scaling index is below C, and lane zero alone writes moment_2[N].
        unsafe {
            kernel::sh_adam_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                workgroups,
                burn_cubecl::cubecl::CubeDim::new_1d(WORKGROUP_SIZE),
                param.into_tensor_arg(),
                grad.into_tensor_arg(),
                moment_1.into_tensor_arg(),
                moment_2.into_tensor_arg(),
                scaling.into_tensor_arg(),
                out_param.clone().into_tensor_arg(),
                out_moment_1.clone().into_tensor_arg(),
                out_moment_2.clone().into_tensor_arg(),
                num_splats,
                config.beta_1,
                config.beta_2,
                config.bias_correction_1,
                config.bias_correction_2,
                config.epsilon,
                config.learning_rate,
                row_len,
            );
        }

        ShAdamOutput {
            param: out_param,
            moment_1: out_moment_1,
            moment_2: out_moment_2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_sh_adam(
        param: FloatTensor<Self>,
        render_transforms: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        compact_grads: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        render: SparseShRenderConfig,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self> {
        param.assert_is_on_same_device(&render_transforms);
        param.assert_is_on_same_device(&global_from_compact_gid);
        param.assert_is_on_same_device(&compact_grads);
        param.assert_is_on_same_device(&moment_1);
        param.assert_is_on_same_device(&moment_2);
        param.assert_is_on_same_device(&scaling);

        let param = into_contiguous(param);
        let render_transforms = into_contiguous(render_transforms);
        let global_from_compact_gid = into_contiguous(global_from_compact_gid);
        let compact_grads = into_contiguous(compact_grads);
        let moment_1 = into_contiguous(moment_1);
        let moment_2 = into_contiguous(moment_2);
        let scaling = into_contiguous(scaling);

        for (name, tensor) in [
            ("parameter", &param),
            ("render transforms", &render_transforms),
            ("compact gradients", &compact_grads),
            ("moment_1", &moment_1),
            ("moment_2", &moment_2),
            ("scaling", &scaling),
        ] {
            assert_eq!(
                tensor.dtype,
                DType::F32,
                "sparse SH Adam {name} must be f32"
            );
        }
        assert_eq!(
            global_from_compact_gid.dtype,
            DType::U32,
            "sparse SH Adam compact map must be u32"
        );

        let shape = param.shape();
        let dims = shape.as_slice();
        assert_eq!(dims.len(), 3, "sparse SH Adam expects [N, C, 3]");
        assert_eq!(dims[2], 3, "sparse SH Adam expects RGB coefficients");
        assert_eq!(
            render_transforms.shape().as_slice(),
            &[dims[0], 10],
            "render transforms must be [N, 10]"
        );
        assert_eq!(
            moment_1.shape(),
            shape,
            "moment_1 shape must match parameter"
        );
        assert_eq!(
            moment_2.shape().as_slice(),
            &[dims[0], 1, 1],
            "moment_2 must be scalar per splat"
        );
        assert_eq!(
            scaling.shape().as_slice(),
            &[1, dims[1], 1],
            "scaling must be one value per SH coefficient"
        );
        assert_eq!(
            compact_grads.shape().as_slice().get(1),
            Some(&10),
            "compact gradients must have stride 10"
        );
        assert_eq!(
            render.total_splats as usize, dims[0],
            "render splat count must match the SH parameter"
        );
        assert_eq!(
            (render.sh_degree as usize + 1).pow(2),
            dims[1],
            "render SH degree must match the coefficient count"
        );
        assert!(render.sh_degree <= 4, "sparse SH Adam supports degree <= 4");
        assert!(dims[0] > 0, "sparse SH Adam requires at least one splat");
        assert!(
            global_from_compact_gid.shape()[0] >= render.num_visible.max(1) as usize,
            "compact-to-global map is too small"
        );
        assert!(
            compact_grads.shape()[0] >= render.num_visible.max(1) as usize,
            "compact gradient buffer is too small"
        );
        assert!(
            u32::try_from(shape.num_elements()).is_ok(),
            "sparse SH Adam flattened indices must fit in u32"
        );

        let properties = param.client.properties();
        let features = param.client.features();
        assert!(
            features.plane.contains(Plane::Ops),
            "sparse SH Adam requires plane operations"
        );
        assert!(
            features.plane.contains(Plane::NonUniformControlFlow),
            "sparse SH Adam requires non-uniform plane control flow"
        );
        assert_eq!(
            properties.hardware.plane_size_min, PLANE_SIZE,
            "sparse SH Adam requires 32-lane planes"
        );
        assert_eq!(
            properties.hardware.plane_size_max, PLANE_SIZE,
            "sparse SH Adam requires a fixed plane size"
        );
        assert!(
            properties.hardware.max_units_per_cube >= WORKGROUP_SIZE,
            "sparse SH Adam requires 256 units per workgroup"
        );
        assert!(
            properties.hardware.max_cube_dim.0 >= WORKGROUP_SIZE,
            "sparse SH Adam requires a 256-wide workgroup"
        );

        let num_splats = u32::try_from(dims[0]).expect("splat count exceeds u32");
        let compact_plus_one_from_global =
            Self::int_zeros(Shape::new([dims[0]]), &param.device, IntDType::U32);
        let client = param.client.clone();
        if render.num_visible > 0 {
            tracing::trace_span!("BuildSparseShMap").in_scope(|| {
                kernel::build_compact_sh_map_kernel::launch::<WgpuRuntime>(
                    &client,
                    calc_cube_count_1d(render.num_visible, WORKGROUP_SIZE),
                    burn_cubecl::cubecl::CubeDim::new_1d(WORKGROUP_SIZE),
                    global_from_compact_gid.into_tensor_arg(),
                    compact_grads.clone().into_tensor_arg(),
                    compact_plus_one_from_global.clone().into_tensor_arg(),
                    render.num_visible,
                );
            });
        }

        let out_param = empty_like(&param);
        let out_moment_1 = empty_like(&moment_1);
        let out_moment_2 = empty_like(&moment_2);
        tracing::trace_span!("SparseShAdam").in_scope(|| {
            // SAFETY: all tensors are contiguous and shape-checked. Each
            // 32-lane plane owns one global row, the inverse map is either a
            // zero sentinel or an in-bounds compact row, and the lane stores
            // cover the complete degree-0..4 SH row exactly once.
            unsafe {
                kernel::sparse_sh_adam_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    calc_cube_count_1d(num_splats, SPLATS_PER_WORKGROUP),
                    burn_cubecl::cubecl::CubeDim::new_1d(WORKGROUP_SIZE),
                    param.into_tensor_arg(),
                    render_transforms.into_tensor_arg(),
                    compact_plus_one_from_global.into_tensor_arg(),
                    compact_grads.into_tensor_arg(),
                    moment_1.into_tensor_arg(),
                    moment_2.into_tensor_arg(),
                    scaling.into_tensor_arg(),
                    out_param.clone().into_tensor_arg(),
                    out_moment_1.clone().into_tensor_arg(),
                    out_moment_2.clone().into_tensor_arg(),
                    num_splats,
                    render.camera_position[0],
                    render.camera_position[1],
                    render.camera_position[2],
                    config.beta_1,
                    config.beta_2,
                    config.bias_correction_1,
                    config.bias_correction_2,
                    config.epsilon,
                    config.learning_rate,
                    render.sh_degree,
                );
            }
        });

        ShAdamOutput {
            param: out_param,
            moment_1: out_moment_1,
            moment_2: out_moment_2,
        }
    }
}

#[derive(Debug)]
struct ShAdamFusionOp {
    desc: CustomOpIr,
    config: ShAdamConfig,
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for ShAdamFusionOp {
    fn execute(&self, handles: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        let ([param, grad, moment_1, moment_2, scaling], [out_param, out_moment_1, out_moment_2]) =
            self.desc.as_fixed();
        let output = <MainBackendBase as ShAdamOps>::sh_adam(
            handles.get_float_tensor::<MainBackendBase>(param),
            handles.get_float_tensor::<MainBackendBase>(grad),
            handles.get_float_tensor::<MainBackendBase>(moment_1),
            handles.get_float_tensor::<MainBackendBase>(moment_2),
            handles.get_float_tensor::<MainBackendBase>(scaling),
            self.config,
        );
        handles.register_float_tensor::<MainBackendBase>(&out_param.id, output.param);
        handles.register_float_tensor::<MainBackendBase>(&out_moment_1.id, output.moment_1);
        handles.register_float_tensor::<MainBackendBase>(&out_moment_2.id, output.moment_2);
    }
}

#[derive(Debug)]
struct SparseShAdamFusionOp {
    desc: CustomOpIr,
    render: SparseShRenderConfig,
    config: ShAdamConfig,
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for SparseShAdamFusionOp {
    fn execute(&self, handles: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        let (
            [
                param,
                render_transforms,
                global_from_compact_gid,
                compact_grads,
                moment_1,
                moment_2,
                scaling,
            ],
            [out_param, out_moment_1, out_moment_2],
        ) = self.desc.as_fixed();
        let output = <MainBackendBase as ShAdamOps>::sparse_sh_adam(
            handles.get_float_tensor::<MainBackendBase>(param),
            handles.get_float_tensor::<MainBackendBase>(render_transforms),
            handles.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
            handles.get_float_tensor::<MainBackendBase>(compact_grads),
            handles.get_float_tensor::<MainBackendBase>(moment_1),
            handles.get_float_tensor::<MainBackendBase>(moment_2),
            handles.get_float_tensor::<MainBackendBase>(scaling),
            self.render,
            self.config,
        );
        handles.register_float_tensor::<MainBackendBase>(&out_param.id, output.param);
        handles.register_float_tensor::<MainBackendBase>(&out_moment_1.id, output.moment_1);
        handles.register_float_tensor::<MainBackendBase>(&out_moment_2.id, output.moment_2);
    }
}

impl ShAdamOps for Fusion<MainBackendBase> {
    fn sh_adam(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self> {
        let client = param.client.clone();
        let out_param = TensorIr::uninit(client.create_empty_handle(), param.shape(), DType::F32);
        let out_moment_1 =
            TensorIr::uninit(client.create_empty_handle(), moment_1.shape(), DType::F32);
        let out_moment_2 =
            TensorIr::uninit(client.create_empty_handle(), moment_2.shape(), DType::F32);
        let desc = CustomOpIr::new(
            "fused_sh_adam",
            &[
                param.into_ir(),
                grad.into_ir(),
                moment_1.into_ir(),
                moment_2.into_ir(),
                scaling.into_ir(),
            ],
            &[out_param, out_moment_1, out_moment_2],
        );
        let operation = ShAdamFusionOp {
            desc: desc.clone(),
            config,
        };
        let [param, moment_1, moment_2] = client
            .register(StreamId::current(), OperationIr::Custom(desc), operation)
            .outputs();
        ShAdamOutput {
            param,
            moment_1,
            moment_2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_sh_adam(
        param: FloatTensor<Self>,
        render_transforms: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        compact_grads: FloatTensor<Self>,
        moment_1: FloatTensor<Self>,
        moment_2: FloatTensor<Self>,
        scaling: FloatTensor<Self>,
        render: SparseShRenderConfig,
        config: ShAdamConfig,
    ) -> ShAdamOutput<Self> {
        let client = param.client.clone();
        let out_param = TensorIr::uninit(client.create_empty_handle(), param.shape(), DType::F32);
        let out_moment_1 =
            TensorIr::uninit(client.create_empty_handle(), moment_1.shape(), DType::F32);
        let out_moment_2 =
            TensorIr::uninit(client.create_empty_handle(), moment_2.shape(), DType::F32);
        let desc = CustomOpIr::new(
            "sparse_sh_adam",
            &[
                param.into_ir(),
                render_transforms.into_ir(),
                global_from_compact_gid.into_ir(),
                compact_grads.into_ir(),
                moment_1.into_ir(),
                moment_2.into_ir(),
                scaling.into_ir(),
            ],
            &[out_param, out_moment_1, out_moment_2],
        );
        let operation = SparseShAdamFusionOp {
            desc: desc.clone(),
            render,
            config,
        };
        let [param, moment_1, moment_2] = client
            .register(StreamId::current(), OperationIr::Custom(desc), operation)
            .outputs();
        ShAdamOutput {
            param,
            moment_1,
            moment_2,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;

    fn reference_update(
        param: Tensor<3>,
        grad: Tensor<3>,
        moment_1: Tensor<3>,
        moment_2: Tensor<3>,
        scaling: Tensor<3>,
        config: ShAdamConfig,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let [num_splats, coeffs, channels] = param.dims();
        let row_len = coeffs * channels;
        let new_moment_1 =
            moment_1.mul_scalar(config.beta_1) + grad.clone().mul_scalar(1.0 - config.beta_1);
        let grad_sq_flat: Tensor<2> = grad.powi_scalar(2).flatten(1, 2);
        let mean_grad_sq = grad_sq_flat
            .sum_dim(1)
            .div_scalar(row_len as f32)
            .reshape([num_splats, 1, 1]);
        let new_moment_2 =
            moment_2.mul_scalar(config.beta_2) + mean_grad_sq.mul_scalar(1.0 - config.beta_2);
        let normalized = new_moment_1
            .clone()
            .div_scalar(config.bias_correction_1)
            .div(
                new_moment_2
                    .clone()
                    .div_scalar(config.bias_correction_2)
                    .sqrt()
                    .add_scalar(config.epsilon),
            );
        let out_param = param - normalized * scaling.mul_scalar(config.learning_rate);
        (out_param, new_moment_1, new_moment_2)
    }

    fn patterned_values(len: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let centered = (index * multiplier + 3) % modulus;
                (centered as f32 - modulus as f32 * 0.5) * scale
            })
            .collect()
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], relative: f32, absolute: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = absolute + relative * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}[{index}]: actual={actual:e}, expected={expected:e}, diff={:e}, tolerance={tolerance:e}",
                (actual - expected).abs(),
            );
        }
    }

    async fn compare_case(num_splats: usize, coeffs: usize, time: i32) {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let num_values = num_splats * coeffs * 3;
        let param_values = patterned_values(num_values, 17, 101, 0.002);
        let mut grad_values = patterned_values(num_values, 13, 67, 0.0007);
        let moment_1_values = patterned_values(num_values, 19, 79, 0.0002);
        let moment_2_values: Vec<f32> = (0..num_splats)
            .map(|row| 0.0001 + (row % 11) as f32 * 0.00003)
            .collect();
        let scaling_values: Vec<f32> = (0..coeffs)
            .map(|coeff| {
                if coeff == 0 {
                    1.0
                } else {
                    0.035 + (coeff % 7) as f32 * 0.017
                }
            })
            .collect();

        // Exercise zero-gradient rows with non-zero momentum. They must still
        // receive the decaying first-moment update.
        for row in (0..num_splats).step_by(4) {
            grad_values[row * coeffs * 3..(row + 1) * coeffs * 3].fill(0.0);
        }

        let config = ShAdamConfig {
            beta_1: 0.87,
            beta_2: 0.996,
            bias_correction_1: 1.0 - 0.87f32.powi(time),
            bias_correction_2: 1.0 - 0.996f32.powi(time),
            epsilon: 1e-15,
            learning_rate: 0.0023,
        };
        let param = Tensor::from_data(
            TensorData::new(param_values.clone(), [num_splats, coeffs, 3]),
            &device,
        );
        let grad = Tensor::from_data(
            TensorData::new(grad_values, [num_splats, coeffs, 3]),
            &device,
        );
        let moment_1 = Tensor::from_data(
            TensorData::new(moment_1_values, [num_splats, coeffs, 3]),
            &device,
        );
        let moment_2 = Tensor::from_data(
            TensorData::new(moment_2_values, [num_splats, 1, 1]),
            &device,
        );
        let scaling = Tensor::from_data(TensorData::new(scaling_values, [1, coeffs, 1]), &device);

        let expected = reference_update(
            param.clone(),
            grad.clone(),
            moment_1.clone(),
            moment_2.clone(),
            scaling.clone(),
            config,
        );
        let actual = sh_adam(param, grad, moment_1, moment_2, scaling, config);
        let expected_param: Vec<f32> = expected
            .0
            .into_data_async()
            .await
            .expect("reference parameter readback")
            .to_vec()
            .expect("reference parameter type");
        let expected_moment_1: Vec<f32> = expected
            .1
            .into_data_async()
            .await
            .expect("reference moment_1 readback")
            .to_vec()
            .expect("reference moment_1 type");
        let expected_moment_2: Vec<f32> = expected
            .2
            .into_data_async()
            .await
            .expect("reference moment_2 readback")
            .to_vec()
            .expect("reference moment_2 type");
        let actual_param: Vec<f32> = actual
            .0
            .into_data_async()
            .await
            .expect("fused parameter readback")
            .to_vec()
            .expect("fused parameter type");
        let actual_moment_1: Vec<f32> = actual
            .1
            .into_data_async()
            .await
            .expect("fused moment_1 readback")
            .to_vec()
            .expect("fused moment_1 type");
        let actual_moment_2: Vec<f32> = actual
            .2
            .into_data_async()
            .await
            .expect("fused moment_2 readback")
            .to_vec()
            .expect("fused moment_2 type");

        let case = format!("N={num_splats}, C={coeffs}, time={time}");
        assert_close(
            &format!("{case} parameter"),
            &actual_param,
            &expected_param,
            2e-5,
            2e-7,
        );
        assert_close(
            &format!("{case} moment_1"),
            &actual_moment_1,
            &expected_moment_1,
            2e-5,
            2e-7,
        );
        assert_close(
            &format!("{case} moment_2"),
            &actual_moment_2,
            &expected_moment_2,
            5e-5,
            1e-12,
        );
        let actual_delta: Vec<f32> = param_values
            .iter()
            .zip(&actual_param)
            .map(|(before, after)| before - after)
            .collect();
        let expected_delta: Vec<f32> = param_values
            .iter()
            .zip(&expected_param)
            .map(|(before, after)| before - after)
            .collect();
        assert_close(
            &format!("{case} update delta"),
            &actual_delta,
            &expected_delta,
            2e-5,
            1e-8,
        );
    }

    #[tokio::test]
    async fn fused_matches_generic_for_all_sh_degrees_and_times() {
        for coeffs in [1, 4, 9, 16, 25] {
            for time in [2, 200, 15_000] {
                compare_case(9, coeffs, time).await;
            }
        }
    }

    #[tokio::test]
    async fn fused_updates_partial_workgroups() {
        for num_splats in [1, 7, 8, 9, 15, 16, 17] {
            compare_case(num_splats, 16, 37).await;
        }
    }

    fn host_sh_basis(index: usize, degree: usize, x: f32, y: f32, z: f32) -> f32 {
        if index == 0 {
            return brush_render::kernels::sh::SH_C0;
        }
        if degree >= 1 {
            let f0a = 0.488_602_5f32;
            match index {
                1 => return -f0a * y,
                2 => return f0a * z,
                3 => return -f0a * x,
                _ => {}
            }
        }
        if degree >= 2 {
            let z2 = z * z;
            let f0b = -1.092_548_5f32 * z;
            let f1a = 0.546_274_24f32;
            let fc1 = x * x - y * y;
            let fs1 = 2.0f32 * x * y;
            match index {
                4 => return f1a * fs1,
                5 => return f0b * y,
                6 => return 0.946_174_7f32 * z2 - 0.315_391_57f32,
                7 => return f0b * x,
                8 => return f1a * fc1,
                _ => {}
            }
        }
        if degree >= 3 {
            let z2 = z * z;
            let f0c = -2.285_229f32 * z2 + 0.457_045_8f32;
            let f1b = 1.445_305_7f32 * z;
            let f2a = -0.590_043_6f32;
            let fc1 = x * x - y * y;
            let fs1 = 2.0f32 * x * y;
            let fc2 = x * fc1 - y * fs1;
            let fs2 = x * fs1 + y * fc1;
            match index {
                9 => return f2a * fs2,
                10 => return f1b * fs1,
                11 => return f0c * y,
                12 => return z * (1.865_881_7f32 * z2 - 1.119_529f32),
                13 => return f0c * x,
                14 => return f1b * fc1,
                15 => return f2a * fc2,
                _ => {}
            }
        }
        if degree >= 4 {
            let z2 = z * z;
            let f0d = z * (-4.683_326f32 * z2 + 2.007_139_6f32);
            let f1c = 3.311_611_4f32 * z2 - 0.473_087_35f32;
            let f2b = -1.770_130_8f32 * z;
            let f3a = 0.625_835_8f32;
            let fc1 = x * x - y * y;
            let fs1 = 2.0f32 * x * y;
            let fc2 = x * fc1 - y * fs1;
            let fs2 = x * fs1 + y * fc1;
            let fc3 = x * fc2 - y * fs2;
            let fs3 = x * fs2 + y * fc2;
            let p_sh6 = 0.946_174_7f32 * z2 - 0.315_391_57f32;
            let p_sh12 = z * (1.865_881_7f32 * z2 - 1.119_529f32);
            match index {
                16 => return f3a * fs3,
                17 => return f2b * fs2,
                18 => return f1c * fs1,
                19 => return f0d * y,
                20 => return 1.984_313_5f32 * z * p_sh12 - 1.006_230_6f32 * p_sh6,
                21 => return f0d * x,
                22 => return f1c * fc1,
                23 => return f2b * fc2,
                24 => return f3a * fc3,
                _ => {}
            }
        }
        0.0
    }

    async fn compare_sparse_case(coeffs: usize, zero_visible: bool) {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let num_splats = 9usize;
        let degree = coeffs.isqrt() - 1;
        let num_values = num_splats * coeffs * 3;
        let param_values = patterned_values(num_values, 17, 101, 0.002);
        let moment_1_values = patterned_values(num_values, 19, 79, 0.0002);
        let moment_2_values: Vec<f32> = (0..num_splats)
            .map(|row| 0.0001 + (row % 11) as f32 * 0.00003)
            .collect();
        let scaling_values: Vec<f32> = (0..coeffs)
            .map(|coeff| {
                if coeff == 0 {
                    1.0
                } else {
                    0.04 + coeff as f32 * 0.003
                }
            })
            .collect();
        let camera = [0.2f32, -0.4, 0.1];
        let mut transform_values = vec![0.0f32; num_splats * 10];
        for row in 0..num_splats {
            transform_values[row * 10] = 0.8 + row as f32 * 0.13;
            transform_values[row * 10 + 1] = -0.15 + row as f32 * 0.07;
            transform_values[row * 10 + 2] = 1.1 - row as f32 * 0.04;
        }

        let visible_globals: Vec<u32> = if zero_visible {
            vec![0]
        } else {
            vec![1, 4, 7, 8]
        };
        let num_visible = if zero_visible {
            0
        } else {
            visible_globals.len()
        };
        let mut compact_values = vec![0.0f32; visible_globals.len() * 10];
        if !zero_visible {
            for compact in 0..visible_globals.len() {
                compact_values[compact * 10 + 5] = 0.003 * (compact as f32 + 1.0);
                compact_values[compact * 10 + 6] = -0.002 * (compact as f32 + 0.5);
                compact_values[compact * 10 + 7] = 0.0015 * (compact as f32 + 0.25);
            }
            // A visible row with exactly zero color gradient must follow the
            // same momentum-decay path as a non-visible row.
            compact_values[2 * 10 + 5..2 * 10 + 8].fill(0.0);
        }

        let mut dense_grad = vec![0.0f32; num_values];
        for (compact, &global) in visible_globals.iter().take(num_visible).enumerate() {
            let color = [
                compact_values[compact * 10 + 5],
                compact_values[compact * 10 + 6],
                compact_values[compact * 10 + 7],
            ];
            if color == [0.0; 3] {
                continue;
            }
            let base = global as usize * 10;
            let dx = transform_values[base] - camera[0];
            let dy = transform_values[base + 1] - camera[1];
            let dz = transform_values[base + 2] - camera[2];
            let inv_len = 1.0 / (dx * dx + dy * dy + dz * dz).sqrt();
            for coeff in 0..coeffs {
                let basis = host_sh_basis(coeff, degree, dx * inv_len, dy * inv_len, dz * inv_len);
                for channel in 0..3 {
                    dense_grad[(global as usize * coeffs + coeff) * 3 + channel] =
                        basis * color[channel];
                }
            }
        }

        let config = ShAdamConfig {
            beta_1: 0.87,
            beta_2: 0.996,
            bias_correction_1: 1.0 - 0.87f32.powi(200),
            bias_correction_2: 1.0 - 0.996f32.powi(200),
            epsilon: 1e-15,
            learning_rate: 0.0023,
        };
        let param = Tensor::from_data(
            TensorData::new(param_values, [num_splats, coeffs, 3]),
            &device,
        );
        let grad = Tensor::from_data(
            TensorData::new(dense_grad, [num_splats, coeffs, 3]),
            &device,
        );
        let transforms: Tensor<2> =
            Tensor::from_data(TensorData::new(transform_values, [num_splats, 10]), &device);
        let compact_map: Tensor<1, Int> = Tensor::<1, Int>::from_data(
            TensorData::new(visible_globals, [num_visible.max(1)]),
            &device,
        )
        .cast(IntDType::U32);
        let compact: Tensor<2> = Tensor::from_data(
            TensorData::new(compact_values, [num_visible.max(1), 10]),
            &device,
        );
        let moment_1 = Tensor::from_data(
            TensorData::new(moment_1_values, [num_splats, coeffs, 3]),
            &device,
        );
        let moment_2 = Tensor::from_data(
            TensorData::new(moment_2_values, [num_splats, 1, 1]),
            &device,
        );
        let scaling = Tensor::from_data(TensorData::new(scaling_values, [1, coeffs, 1]), &device);

        let expected = sh_adam(
            param.clone(),
            grad,
            moment_1.clone(),
            moment_2.clone(),
            scaling.clone(),
            config,
        );
        let output = <Dispatch as ShAdamOps>::sparse_sh_adam(
            param.into_dispatch(),
            transforms.into_dispatch(),
            compact_map.into_dispatch(),
            compact.into_dispatch(),
            moment_1.into_dispatch(),
            moment_2.into_dispatch(),
            scaling.into_dispatch(),
            SparseShRenderConfig {
                camera_position: camera,
                sh_degree: degree as u32,
                total_splats: num_splats as u32,
                num_visible: num_visible as u32,
            },
            config,
        );
        let actual = (
            Tensor::<3>::from_dispatch(output.param),
            Tensor::<3>::from_dispatch(output.moment_1),
            Tensor::<3>::from_dispatch(output.moment_2),
        );

        for (label, actual, expected) in [
            ("parameter", actual.0, expected.0),
            ("moment_1", actual.1, expected.1),
            ("moment_2", actual.2, expected.2),
        ] {
            let actual: Vec<f32> = actual
                .into_data_async()
                .await
                .expect("sparse output readback")
                .to_vec()
                .expect("sparse output type");
            let expected: Vec<f32> = expected
                .into_data_async()
                .await
                .expect("dense output readback")
                .to_vec()
                .expect("dense output type");
            let case = format!("sparse degree {degree}, zero_visible={zero_visible}, {label}");
            // The reference SH row is evaluated on the CPU, where FMA
            // contraction can differ slightly from MSL. Adam itself is held
            // to a sub-micro absolute tolerance here.
            assert_close(&case, &actual, &expected, 1e-4, 1e-6);
        }
    }

    #[tokio::test]
    async fn sparse_matches_dense_for_all_sh_degrees_and_zero_rows() {
        for coeffs in [1, 4, 9, 16, 25] {
            compare_sparse_case(coeffs, false).await;
        }
        compare_sparse_case(25, true).await;
    }

    #[tokio::test]
    #[ignore = "large native-Metal launch geometry soak"]
    async fn fused_crosses_2d_dispatch_boundary() {
        compare_case(524_281, 1, 15_000).await;
    }
}
