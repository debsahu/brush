//! Correctness tests for the appearance kernels.

use brush_appearance::bilagrid::{BilagridModel, bilagrid_apply, bilagrid_tv_loss};
use brush_appearance::ppisp::{PpispModel, PpispStages, ppisp_apply};
use burn::tensor::{Device, Tensor};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

async fn ad_device() -> Device {
    Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

fn pattern(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
            let hash = (word >> 22) ^ word;
            lo + (hi - lo) * (hash as f32 / u32::MAX as f32)
        })
        .collect()
}

async fn read<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("f32 data")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let (index, error) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    assert!(
        error <= tolerance,
        "{label}: max error {error} at {index}; got {}, expected {}",
        actual[index],
        expected[index]
    );
}

#[allow(clippy::too_many_arguments)]
fn bilagrid_cpu(
    grid: &[f32],
    rgb: &[f32],
    grid_l: usize,
    grid_h: usize,
    grid_w: usize,
    image_h: usize,
    image_w: usize,
) -> Vec<f32> {
    const LUMA: [f32; 3] = [0.299, 0.587, 0.114];
    let mut output = vec![0.0; image_h * image_w * 3];
    let cells = grid_l * grid_h * grid_w;
    for pixel_y in 0..image_h {
        for pixel_x in 0..image_w {
            let base = (pixel_y * image_w + pixel_x) * 3;
            let input = [rgb[base], rgb[base + 1], rgb[base + 2], 1.0];
            let x = pixel_x as f32 * (grid_w - 1) as f32 / (image_w - 1).max(1) as f32;
            let y = pixel_y as f32 * (grid_h - 1) as f32 / (image_h - 1).max(1) as f32;
            let z = ((LUMA[0] * input[0] + LUMA[1] * input[1] + LUMA[2] * input[2])
                * (grid_l - 1) as f32)
                .clamp(0.0, (grid_l - 1) as f32);
            let (x0, y0, z0) = (x.floor() as usize, y.floor() as usize, z.floor() as usize);
            let (x1, y1, z1) = (
                (x0 + 1).min(grid_w - 1),
                (y0 + 1).min(grid_h - 1),
                (z0 + 1).min(grid_l - 1),
            );
            let (tx, ty, tz) = (x - x.floor(), y - y.floor(), z - z.floor());
            for row in 0..3 {
                for column in 0..4 {
                    let coefficient = row * 4 + column;
                    let at = |zz: usize, yy: usize, xx: usize| {
                        grid[coefficient * cells + (zz * grid_h + yy) * grid_w + xx]
                    };
                    let z0_value = (at(z0, y0, x0) * (1.0 - tx) + at(z0, y0, x1) * tx) * (1.0 - ty)
                        + (at(z0, y1, x0) * (1.0 - tx) + at(z0, y1, x1) * tx) * ty;
                    let z1_value = (at(z1, y0, x0) * (1.0 - tx) + at(z1, y0, x1) * tx) * (1.0 - ty)
                        + (at(z1, y1, x0) * (1.0 - tx) + at(z1, y1, x1) * tx) * ty;
                    output[base + row] += (z0_value * (1.0 - tz) + z1_value * tz) * input[column];
                }
            }
        }
    }
    output
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilateral_grid_identity_is_noop() {
    let device = ad_device().await;
    let (height, width) = (17, 23);
    let model = BilagridModel::new(3, 8, 8, 4, &device);
    let rgb_data = pattern(height * width * 3, 7, 0.0, 1.0);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([height, width, 3]);

    let output = read(model.apply(rgb, 1)).await;
    assert_close(&output, &rgb_data, 1e-5, "identity bilateral grid");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilateral_grid_matches_cpu_reference() {
    let device = ad_device().await;
    let (height, width) = (15, 21);
    let (grid_x, grid_y, grid_l) = (6, 5, 4);
    let (views, view) = (2, 1);
    let grids_data = pattern(views * 12 * grid_l * grid_y * grid_x, 3, -0.6, 1.2);
    let rgb_data = pattern(height * width * 3, 11, 0.0, 1.0);
    let grids = Tensor::<1>::from_floats(grids_data.as_slice(), &device)
        .reshape([views, 12, grid_l, grid_y, grid_x]);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([height, width, 3]);

    let output = read(bilagrid_apply(grids, rgb, view)).await;
    let view_elements = 12 * grid_l * grid_y * grid_x;
    let expected = bilagrid_cpu(
        &grids_data[view * view_elements..(view + 1) * view_elements],
        &rgb_data,
        grid_l,
        grid_y,
        grid_x,
        height,
        width,
    );
    assert_close(&output, &expected, 1e-4, "bilateral grid CPU reference");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilateral_grid_preserves_alpha() {
    let device = ad_device().await;
    let (height, width) = (9, 13);
    let model = BilagridModel::new(1, 4, 4, 2, &device);
    let rgba_data = pattern(height * width * 4, 5, 0.0, 1.0);
    let rgba = Tensor::<1>::from_floats(rgba_data.as_slice(), &device).reshape([height, width, 4]);

    let output = read(model.apply(rgba, 0)).await;
    for pixel in 0..height * width {
        assert_eq!(output[pixel * 4 + 3], rgba_data[pixel * 4 + 3]);
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilateral_grid_gradients_match_finite_differences() {
    let device = ad_device().await;
    let (height, width) = (7, 9);
    let (grid_x, grid_y, grid_l) = (4, 4, 3);
    let grids_data = pattern(12 * grid_l * grid_y * grid_x, 17, -0.4, 1.1);
    let rgb_data = pattern(height * width * 3, 19, 0.05, 0.95);
    let weights_data = pattern(height * width * 3, 23, -1.0, 1.0);
    let make_grid = |data: &[f32]| {
        Tensor::<1>::from_floats(data, &device).reshape([1, 12, grid_l, grid_y, grid_x])
    };
    let make_rgb =
        |data: &[f32]| Tensor::<1>::from_floats(data, &device).reshape([height, width, 3]);
    let weights = make_rgb(&weights_data);
    let grids = make_grid(&grids_data).require_grad();
    let rgb = make_rgb(&rgb_data).require_grad();
    let grads = (bilagrid_apply(grids.clone(), rgb.clone(), 0) * weights.clone())
        .sum()
        .backward();
    let grid_grad = read(grids.grad(&grads).expect("grid gradient")).await;
    let rgb_grad = read(rgb.grad(&grads).expect("RGB gradient")).await;
    let loss = |grid: &[f32], rgb: &[f32]| {
        (bilagrid_apply(make_grid(grid), make_rgb(rgb), 0) * weights.clone()).sum()
    };
    let epsilon = 2e-3;

    for index in [7usize, 101, 12 * grid_l * grid_y * grid_x - 5] {
        let mut plus = grids_data.clone();
        let mut minus = grids_data.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference = (loss(&plus, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("plus loss")
            - loss(&minus, &rgb_data)
                .into_scalar_async::<f32>()
                .await
                .expect("minus loss"))
            / (2.0 * epsilon);
        assert!((grid_grad[index] - finite_difference).abs() < 2e-2);
    }
    for index in [4usize, 50, 151] {
        let mut plus = rgb_data.clone();
        let mut minus = rgb_data.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference = (loss(&grids_data, &plus)
            .into_scalar_async::<f32>()
            .await
            .expect("plus loss")
            - loss(&grids_data, &minus)
                .into_scalar_async::<f32>()
                .await
                .expect("minus loss"))
            / (2.0 * epsilon);
        assert!((rgb_grad[index] - finite_difference).abs() < 2e-2);
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilateral_grid_tv_has_correct_gradient() {
    let device = ad_device().await;
    let shape = [2, 12, 3, 3, 4];
    let data = pattern(shape.iter().product(), 29, -0.8, 0.8);
    let make = |values: &[f32]| Tensor::<1>::from_floats(values, &device).reshape(shape);
    let grids = make(&data).require_grad();
    let loss = bilagrid_tv_loss(grids.clone());
    let grads = loss.backward();
    let gradient = read(grids.grad(&grads).expect("TV gradient")).await;
    let epsilon = 1e-2;
    for index in [0usize, 13, data.len() - 1] {
        let mut plus = data.clone();
        let mut minus = data.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference = (bilagrid_tv_loss(make(&plus))
            .into_scalar_async::<f32>()
            .await
            .expect("plus TV")
            - bilagrid_tv_loss(make(&minus))
                .into_scalar_async::<f32>()
                .await
                .expect("minus TV"))
            / (2.0 * epsilon);
        assert!((gradient[index] - finite_difference).abs() < 2e-2);
    }
}

fn ppisp_inputs(
    device: &Device,
    exposure: f32,
    rgb: &[f32],
    height: usize,
    width: usize,
) -> (Tensor<1>, Tensor<3>, Tensor<2>, Tensor<3>, Tensor<3>) {
    (
        Tensor::from_floats([exposure], device),
        Tensor::zeros([1, 3, 5], device),
        Tensor::zeros([1, 8], device),
        Tensor::zeros([1, 3, 4], device),
        Tensor::<1>::from_floats(rgb, device).reshape([height, width, 3]),
    )
}

const FRAME_ONLY: PpispStages = PpispStages {
    frame: true,
    vignetting: false,
    crf: false,
};

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_exposure_matches_log2_gain() {
    let device = ad_device().await;
    let (height, width) = (5, 7);
    let rgb_data = pattern(height * width * 3, 41, 0.05, 0.8);
    let exposure = 0.75;
    let (exp, vig, color, crf, rgb) = ppisp_inputs(&device, exposure, &rgb_data, height, width);
    let output = read(ppisp_apply(exp, vig, color, crf, rgb, 0, 0, FRAME_ONLY)).await;
    let expected: Vec<_> = rgb_data
        .iter()
        .map(|value| value * 2.0f32.powf(exposure))
        .collect();
    assert_close(&output, &expected, 2e-4, "PPISP exposure");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_gradients_match_finite_differences() {
    let device = ad_device().await;
    let (height, width) = (5, 7);
    let rgb_data = pattern(height * width * 3, 43, 0.05, 0.8);
    let weights_data = pattern(height * width * 3, 47, -1.0, 1.0);
    let (exp, vig, color, crf, rgb) = ppisp_inputs(&device, 0.2, &rgb_data, height, width);
    let exp = exp.require_grad();
    let rgb = rgb.require_grad();
    let weights =
        Tensor::<1>::from_floats(weights_data.as_slice(), &device).reshape([height, width, 3]);
    let grads = (ppisp_apply(exp.clone(), vig, color, crf, rgb.clone(), 0, 0, FRAME_ONLY)
        * weights.clone())
    .sum()
    .backward();
    let exposure_grad = read(exp.grad(&grads).expect("exposure gradient")).await[0];
    let rgb_grad = read(rgb.grad(&grads).expect("RGB gradient")).await;
    let loss = |exposure: f32, rgb_values: &[f32]| {
        let (exp, vig, color, crf, rgb) =
            ppisp_inputs(&device, exposure, rgb_values, height, width);
        (ppisp_apply(exp, vig, color, crf, rgb, 0, 0, FRAME_ONLY) * weights.clone()).sum()
    };
    let epsilon = 1e-3;
    let exposure_fd = (loss(0.2 + epsilon, &rgb_data)
        .into_scalar_async::<f32>()
        .await
        .expect("plus loss")
        - loss(0.2 - epsilon, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("minus loss"))
        / (2.0 * epsilon);
    assert!((exposure_grad - exposure_fd).abs() < 2e-2);

    let index = 17;
    let mut plus = rgb_data.clone();
    let mut minus = rgb_data.clone();
    plus[index] += epsilon;
    minus[index] -= epsilon;
    let rgb_fd = (loss(0.2, &plus)
        .into_scalar_async::<f32>()
        .await
        .expect("plus loss")
        - loss(0.2, &minus)
            .into_scalar_async::<f32>()
            .await
            .expect("minus loss"))
        / (2.0 * epsilon);
    assert!((rgb_grad[index] - rgb_fd).abs() < 2e-2);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_model_identity_and_regularization() {
    let device = ad_device().await;
    let model = PpispModel::new(2, 3, vec![0, 1, 1], &device);
    let rgb_data = pattern(9 * 11 * 3, 53, 0.05, 0.95);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([9, 11, 3]);
    let output = read(model.apply(rgb, 2)).await;
    assert_close(&output, &rgb_data, 2e-4, "PPISP identity");
    assert_eq!(model.camera_indices, vec![0, 1, 1]);
    let regularization = model
        .reg_loss()
        .into_scalar_async::<f32>()
        .await
        .expect("regularization readback");
    assert!(regularization.abs() < 1e-6);
}
