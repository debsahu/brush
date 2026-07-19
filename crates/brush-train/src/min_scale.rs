//! One-pass GPU evaluation of the Mip-Splatting world-space scale floor.

use brush_cube::{MainBackend as Wgpu, MainBackendBase, calc_cube_count_1d};
use burn::{
    Tensor as BurnTensor,
    backend::{Backend, Dispatch, TensorMetadata, tensor::FloatTensor},
    tensor::{DType, Shape, TensorData},
};
use burn_cubecl::{
    cubecl,
    cubecl::{CubeDim, cube, prelude::*},
    fusion::FusionCubeRuntime,
    kernel::into_contiguous,
    tensor::CubeTensor,
};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::{AutoCompiler, WgpuRuntime};

const WORKGROUP_SIZE: u32 = 256;

#[cube(launch)]
fn min_scale_kernel(
    means: &Tensor<f32>,
    cameras: &Tensor<f32>,
    output: &mut Tensor<f32>,
    num_splats: u32,
    num_cameras: u32,
    factor_sqrt: f32,
) {
    let splat = ABSOLUTE_POS as u32;
    if splat >= num_splats {
        terminate!();
    }

    let mean_base = (splat * 3u32) as usize;
    let mx = means[mean_base];
    let my = means[mean_base + 1];
    let mz = means[mean_base + 2];

    let dx0 = mx - cameras[0];
    let dy0 = my - cameras[1];
    let dz0 = mz - cameras[2];
    let focal0 = f32::max(cameras[3], 1e-6f32);
    let mut min_ratio = f32::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0) / focal0;

    for camera in 1u32..num_cameras {
        let base = (camera * 4u32) as usize;
        let dx = mx - cameras[base];
        let dy = my - cameras[base + 1];
        let dz = mz - cameras[base + 2];
        let focal = f32::max(cameras[base + 3], 1e-6f32);
        let ratio = f32::sqrt(dx * dx + dy * dy + dz * dz) / focal;
        min_ratio = f32::min(min_ratio, ratio);
    }

    output[splat as usize] = min_ratio * factor_sqrt;
}

#[burn::backend::backend_extension(Wgpu)]
trait MinScaleOps: Backend {
    fn min_scale(
        means: FloatTensor<Self>,
        cameras: FloatTensor<Self>,
        factor_sqrt: f32,
    ) -> FloatTensor<Self>;
}

/// Evaluate `sqrt(factor) * min_v(distance(mean, camera_v) / focal_v)` for
/// every splat in one GPU launch. The camera tensor is tiny and the kernel
/// streams it directly; no `[splats, views]` intermediate is materialised.
pub(super) fn compute_min_scale(
    means: &BurnTensor<2>,
    view_cams: &[(glam::Vec3, f32)],
    factor: f32,
) -> Option<BurnTensor<1>> {
    if factor <= 0.0 || view_cams.is_empty() || means.dims()[0] == 0 {
        return None;
    }
    let means = brush_render::burn_glue::detach_autodiff(means.clone());

    let mut camera_data = Vec::with_capacity(view_cams.len() * 4);
    for (center, focal) in view_cams {
        camera_data.extend_from_slice(&[center.x, center.y, center.z, focal.max(1e-6)]);
    }
    let cameras = BurnTensor::<2>::from_data(
        TensorData::new(camera_data, [view_cams.len(), 4]),
        &means.device(),
    );
    let output = <Dispatch as MinScaleOps>::min_scale(
        means.into_dispatch(),
        cameras.into_dispatch(),
        factor.sqrt(),
    );
    Some(BurnTensor::from_dispatch(output))
}

fn empty_output(
    template: &CubeTensor<WgpuRuntime<AutoCompiler>>,
    len: usize,
) -> CubeTensor<WgpuRuntime<AutoCompiler>> {
    let shape = Shape::new([len]);
    let handle = template
        .client
        .empty(shape.num_elements() * DType::F32.size());
    CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        shape,
        handle,
        DType::F32,
    )
}

impl MinScaleOps for MainBackendBase {
    fn min_scale(
        means: FloatTensor<Self>,
        cameras: FloatTensor<Self>,
        factor_sqrt: f32,
    ) -> FloatTensor<Self> {
        means.assert_is_on_same_device(&cameras);
        let means = into_contiguous(means);
        let cameras = into_contiguous(cameras);
        let means_shape = means.shape();
        let camera_shape = cameras.shape();
        let means_dims = means_shape.as_slice();
        let camera_dims = camera_shape.as_slice();
        assert_eq!(means_dims.len(), 2, "min-scale means must be [N, 3]");
        assert_eq!(means_dims[1], 3, "min-scale means must be XYZ");
        assert_eq!(camera_dims.len(), 2, "min-scale cameras must be [V, 4]");
        assert_eq!(camera_dims[1], 4, "min-scale cameras must be XYZ+focal");
        assert!(camera_dims[0] > 0, "min-scale requires at least one camera");

        let num_splats = u32::try_from(means_dims[0]).expect("splat count exceeds u32");
        let num_cameras = u32::try_from(camera_dims[0]).expect("camera count exceeds u32");
        let output = empty_output(&means, means_dims[0]);
        let client = means.client.clone();
        min_scale_kernel::launch::<WgpuRuntime<AutoCompiler>>(
            &client,
            calc_cube_count_1d(num_splats, WORKGROUP_SIZE),
            CubeDim::new_1d(WORKGROUP_SIZE),
            means.into_tensor_arg(),
            cameras.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            num_splats,
            num_cameras,
            factor_sqrt,
        );
        output
    }
}

#[derive(Debug)]
struct MinScaleFusionOp {
    desc: CustomOpIr,
    factor_sqrt: f32,
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for MinScaleFusionOp {
    fn execute(&self, handles: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        let ([means, cameras], [output]) = self.desc.as_fixed();
        let result = <MainBackendBase as MinScaleOps>::min_scale(
            handles.get_float_tensor::<MainBackendBase>(means),
            handles.get_float_tensor::<MainBackendBase>(cameras),
            self.factor_sqrt,
        );
        handles.register_float_tensor::<MainBackendBase>(&output.id, result);
    }
}

impl MinScaleOps for Fusion<MainBackendBase> {
    fn min_scale(
        means: FloatTensor<Self>,
        cameras: FloatTensor<Self>,
        factor_sqrt: f32,
    ) -> FloatTensor<Self> {
        let client = means.client.clone();
        let [num_splats, _] = means.shape().dims();
        let output = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_splats]),
            DType::F32,
        );
        let desc = CustomOpIr::new(
            "compute_min_scale",
            &[means.into_ir(), cameras.into_ir()],
            &[output],
        );
        let operation = MinScaleFusionOp {
            desc: desc.clone(),
            factor_sqrt,
        };
        let [output] = client
            .register(StreamId::current(), OperationIr::Custom(desc), operation)
            .outputs();
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::gaussian_splats::{SplatRenderMode, Splats};

    #[tokio::test]
    async fn gpu_min_scale_matches_reference() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let means = BurnTensor::<2>::from_data(
            TensorData::new(vec![0.0, 0.0, 0.0, 3.0, 4.0, 0.0, -2.0, 0.0, 0.0], [3, 3]),
            &device,
        );
        let cameras = [
            (glam::Vec3::new(0.0, 0.0, 10.0), 100.0),
            (glam::Vec3::new(1.0, 0.0, 0.0), 20.0),
        ];
        let actual: Vec<f32> = compute_min_scale(&means, &cameras, 0.1)
            .expect("scale floor")
            .into_data_async()
            .await
            .expect("readback")
            .to_vec()
            .expect("f32 output");
        let expected: Vec<f32> = means
            .into_data_async()
            .await
            .expect("means readback")
            .to_vec::<f32>()
            .expect("f32 means")
            .chunks_exact(3)
            .map(|xyz| {
                cameras
                    .iter()
                    .map(|(center, focal)| {
                        glam::Vec3::from_slice(xyz).distance(*center) / focal.max(1e-6)
                    })
                    .fold(f32::INFINITY, f32::min)
                    * 0.1f32.sqrt()
            })
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[tokio::test]
    async fn gpu_min_scale_accepts_autodiff_means() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let means =
            BurnTensor::<2>::from_data(TensorData::new(vec![0.0, 0.0, 0.0], [1, 3]), &device);
        let floor: Vec<f32> =
            compute_min_scale(&means, &[(glam::Vec3::new(0.0, 0.0, 10.0), 100.0)], 0.1)
                .expect("scale floor")
                .into_data_async()
                .await
                .expect("readback")
                .to_vec()
                .expect("f32 output");

        assert_eq!(floor.len(), 1);
        assert!((floor[0] - 0.1 * 0.1f32.sqrt()).abs() < 1e-6);
    }

    #[tokio::test]
    async fn refreshing_same_floor_does_not_accumulate_filter() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let splats = Splats::from_raw(
            vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0; 6],
            vec![0.0; 6],
            vec![0.0; 2],
            SplatRenderMode::Mip,
            &device,
        );
        let cameras = [(glam::Vec3::new(0.0, 0.0, 10.0), 100.0)];

        let floor = compute_min_scale(&splats.means(), &cameras, 0.1).expect("scale floor");
        let splats = splats.with_min_scale(floor);
        let first_scales: Vec<f32> = splats
            .scales()
            .into_data_async()
            .await
            .expect("first scale readback")
            .to_vec()
            .expect("f32 scales");
        let first_opacities: Vec<f32> = splats
            .opacities()
            .into_data_async()
            .await
            .expect("first opacity readback")
            .to_vec()
            .expect("f32 opacities");

        let floor = compute_min_scale(&splats.means(), &cameras, 0.1).expect("refreshed floor");
        let refreshed = splats.with_min_scale(floor);
        let refreshed_scales: Vec<f32> = refreshed
            .scales()
            .into_data_async()
            .await
            .expect("refreshed scale readback")
            .to_vec()
            .expect("f32 scales");
        let refreshed_opacities: Vec<f32> = refreshed
            .opacities()
            .into_data_async()
            .await
            .expect("refreshed opacity readback")
            .to_vec()
            .expect("f32 opacities");

        assert_eq!(first_scales, refreshed_scales);
        assert_eq!(first_opacities, refreshed_opacities);
    }
}
