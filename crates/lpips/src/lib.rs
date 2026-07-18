#![recursion_limit = "256"]

use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::tensor::Device;
use burn::tensor::activation::relu;
use burn::tensor::module::{avg_pool2d, conv2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::s;
use burn::{config::Config, module::Module, tensor::Tensor};

/// Spatial scale used by the paper's Wasserstein Distortion loss.
pub const WASSERSTEIN_SIGMA: f32 = 4.0;

/// Minimum input height and width required by the five-slice LPIPS VGG path.
pub const LPIPS_MIN_IMAGE_SIZE: usize = 16;

/// Minimum input height and width required by the paper's three-scale VGG path.
pub const WASSERSTEIN_MIN_IMAGE_SIZE: usize = 61;

const WASSERSTEIN_LOG2_SIGMA: f32 = 2.0;
const WASSERSTEIN_NUM_LEVELS: usize = 5;
const WASSERSTEIN_NUM_SCALES: usize = 3;

// Wasserstein Distortion is translated from balle-lab/wasserstein-distortion,
// commit 32e3da1b22b2a42c3f8a1cd6d1909732b47b284f, licensed Apache-2.0.
// Copyright 2025 Yueyu Hu and Jona Balle. The binomial low-pass filter,
// multi-level local statistics, VGG feature selection, average pooling, and
// ImageNet normalization follow that implementation.

fn lowpass_kernel(device: &Device) -> Tensor<4> {
    Tensor::<1>::from_floats(
        [
            0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
        ],
        device,
    )
    .reshape([1, 1, 3, 3])
}

fn lowpass_2d(input: Tensor<4>, stride: usize, kernel: &Tensor<4>) -> Tensor<4> {
    let channels = input.dims()[1];
    let kernel = kernel.clone().repeat_dim(0, channels);
    conv2d(
        input,
        kernel,
        None,
        ConvOptions::new([stride, stride], [1, 1], [1, 1], channels),
    )
}

fn wasserstein_distortion_feature(
    features_a: Tensor<4>,
    features_b: Tensor<4>,
    log2_sigma: f32,
    kernel: &Tensor<4>,
) -> Tensor<1> {
    // `sigma_map` starts in [0, 2] and the nonnegative low-pass kernel cannot
    // increase it. At pyramid level `n`, the triangular weight is therefore
    // identically zero once `n > ceil(log2_sigma)`. The reference always runs
    // five levels, but those extra terms and all of their gradients are zero.
    let active_levels = (log2_sigma.ceil() as usize).min(WASSERSTEIN_NUM_LEVELS);
    wasserstein_distortion_feature_with_num_levels(
        features_a,
        features_b,
        log2_sigma,
        kernel,
        active_levels,
    )
}

fn wasserstein_distortion_feature_with_num_levels(
    features_a: Tensor<4>,
    features_b: Tensor<4>,
    log2_sigma: f32,
    kernel: &Tensor<4>,
    num_levels: usize,
) -> Tensor<1> {
    assert!(
        num_levels <= WASSERSTEIN_NUM_LEVELS,
        "Wasserstein feature pyramid supports at most {WASSERSTEIN_NUM_LEVELS} levels"
    );
    assert_eq!(
        features_a.dims(),
        features_b.dims(),
        "Wasserstein feature maps must have the same shape"
    );

    let [batch, _channels, height, width] = features_a.dims();
    let device = features_a.device();
    let mut sigma_map = Tensor::full([batch, 1, height, width], log2_sigma, &device);
    let mut current_a = features_a;
    let mut current_b = features_b;
    let mut squared_a = current_a.clone().powi_scalar(2);
    let mut squared_b = current_b.clone().powi_scalar(2);

    let pixel_distance = (current_a.clone() - current_b.clone()).powi_scalar(2);
    let pixel_weight = (1.0f32 - sigma_map.clone().abs()).clamp_min(0.0);
    let mut distance = (pixel_weight * pixel_distance).mean();

    for level in 0..num_levels {
        let mean_a = lowpass_2d(current_a, 1, kernel);
        let mean_b = lowpass_2d(current_b, 1, kernel);
        let second_moment_a = lowpass_2d(squared_a, 1, kernel);
        let second_moment_b = lowpass_2d(squared_b, 1, kernel);

        let std_a = (second_moment_a.clone() - mean_a.clone().powi_scalar(2))
            .clamp_min(1e-8)
            .sqrt();
        let std_b = (second_moment_b.clone() - mean_b.clone().powi_scalar(2))
            .clamp_min(1e-8)
            .sqrt();
        let statistic_distance =
            (mean_a.clone() - mean_b.clone()).powi_scalar(2) + (std_a - std_b).powi_scalar(2);

        let pyramid_level = (level + 1) as f32;
        let weight = (1.0f32 - (sigma_map.clone() - pyramid_level).abs()).clamp_min(0.0);
        distance = distance + (weight * statistic_distance).mean();

        // Match the reference's `m[..., ::2, ::2]` / `p[..., ::2, ::2]`
        // statistic pyramid and its stride-2 low-pass of the sigma map.
        current_a = mean_a.slice(s![.., .., ..;2, ..;2]);
        current_b = mean_b.slice(s![.., .., ..;2, ..;2]);
        squared_a = second_moment_a.slice(s![.., .., ..;2, ..;2]);
        squared_b = second_moment_b.slice(s![.., .., ..;2, ..;2]);
        sigma_map = lowpass_2d(sigma_map, 2, kernel);
    }

    distance
}

fn effective_log2_sigma(
    source_height: usize,
    source_width: usize,
    feature_height: usize,
    feature_width: usize,
) -> f32 {
    let log_ratio_h = (source_height as f32 / feature_height as f32).log2();
    let log_ratio_w = (source_width as f32 / feature_width as f32).log2();
    (WASSERSTEIN_LOG2_SIGMA - (log_ratio_h + log_ratio_w) * 0.5).max(0.0)
}

/// Residual layer block configuration.
#[derive(Config, Debug)]
struct VggBlockConfig {
    num_blocks: usize,
    in_channels: usize,
    out_channels: usize,
}

impl VggBlockConfig {
    /// Initialize a new `LayerBlock` module.
    fn init(&self, device: &Device) -> VggBlock {
        let convs = (0..self.num_blocks)
            .map(|b| {
                let in_channels = if b == 0 {
                    self.in_channels
                } else {
                    self.out_channels
                };

                // conv3x3
                let conv = Conv2dConfig::new([in_channels, self.out_channels], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .with_bias(true);
                conv.init(device)
            })
            .collect();

        VggBlock { convs }
    }
}

#[derive(Module, Debug)]
struct VggBlock {
    convs: Vec<Conv2d>,
}

impl VggBlock {
    pub(crate) fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let mut cur = input;
        for conv in &self.convs {
            cur = relu(conv.forward(cur));
        }
        cur
    }
}

#[derive(Module, Debug)]
pub struct LpipsModel {
    blocks: Vec<VggBlock>,
    heads: Vec<Conv2d>,
    max_pool: MaxPool2d,
}

fn norm_vec(vec: Tensor<4>) -> Tensor<4> {
    let norm_factor = vec.clone().powi_scalar(2).sum_dim(1).sqrt();
    vec / (norm_factor + 1e-10)
}

impl LpipsModel {
    /// Calculate LPIPS for NHWC RGB images normalized to `[0, 1]`.
    pub fn lpips(&self, imgs_a: Tensor<4>, imgs_b: Tensor<4>) -> Tensor<1> {
        let dims_a = imgs_a.dims();
        let dims_b = imgs_b.dims();
        assert_eq!(dims_a, dims_b, "LPIPS inputs must have the same shape");
        let [batch, height, width, channels] = dims_a;
        assert_eq!(
            batch, 1,
            "LPIPS currently evaluates one image pair at a time"
        );
        assert_eq!(channels, 3, "LPIPS expects RGB images");
        assert!(
            height >= LPIPS_MIN_IMAGE_SIZE && width >= LPIPS_MIN_IMAGE_SIZE,
            "LPIPS requires images of at least {LPIPS_MIN_IMAGE_SIZE}x{LPIPS_MIN_IMAGE_SIZE}, got {height}x{width}"
        );

        let device = imgs_a.device();

        // Convert NHWC to NCHW and to [-1, 1].
        let imgs_a = imgs_a.permute([0, 3, 1, 2]) * 2.0 - 1.0;
        let imgs_b = imgs_b.permute([0, 3, 1, 2]) * 2.0 - 1.0;

        let shift =
            Tensor::<1>::from_floats([-0.030, -0.088, -0.188], &device).reshape([1, 3, 1, 1]);
        let scale = Tensor::<1>::from_floats([0.458, 0.448, 0.450], &device).reshape([1, 3, 1, 1]);

        let mut imgs_a = (imgs_a - shift.clone()) / scale.clone();
        let mut imgs_b = (imgs_b - shift) / scale;

        let mut loss = Tensor::<1>::zeros([1], &device);
        for (i, (block, head)) in self.blocks.iter().zip(&self.heads).enumerate() {
            // TODO: concatenating first might be faster.
            if i != 0 {
                imgs_a = self.max_pool.forward(imgs_a);
                imgs_b = self.max_pool.forward(imgs_b);
            }

            // Process each part through the block
            imgs_a = block.forward(imgs_a);
            imgs_b = block.forward(imgs_b);

            let normed_a = norm_vec(imgs_a.clone());
            let normed_b = norm_vec(imgs_b.clone());

            let diff = (normed_a - normed_b).powi_scalar(2);
            let class = head.forward(diff);
            // Add spatial mean.
            loss = loss + class.mean_dim(2).mean_dim(3).reshape([1]);
        }
        loss
    }

    /// Calculate VGG-16 Wasserstein Distortion for NHWC RGB images normalized
    /// to `[0, 1]`.
    ///
    /// This uses the paper's fixed `sigma = 4`, three image scales, five VGG
    /// feature slices per scale, and five local-statistics pyramid levels.
    /// Inputs need to be at least 61 pixels on each side so every VGG slice is
    /// defined at the coarsest image scale.
    pub fn wasserstein_distance(&self, imgs_a: Tensor<4>, imgs_b: Tensor<4>) -> Tensor<1> {
        self.wasserstein_distance_with_num_scales(imgs_a, imgs_b, WASSERSTEIN_NUM_SCALES)
    }

    /// Calculate VGG-16 Wasserstein Distortion with a caller-selected number
    /// of image scales. The reference and the paper use three scales.
    pub fn wasserstein_distance_with_num_scales(
        &self,
        imgs_a: Tensor<4>,
        imgs_b: Tensor<4>,
        num_scales: usize,
    ) -> Tensor<1> {
        let dims_a = imgs_a.dims();
        let dims_b = imgs_b.dims();
        assert_eq!(
            dims_a, dims_b,
            "Wasserstein Distance inputs must have the same shape"
        );
        let [batch, source_height, source_width, channels] = dims_a;
        assert!(batch > 0, "Wasserstein Distance expects a non-empty batch");
        assert_eq!(channels, 3, "Wasserstein Distance expects RGB images");
        if num_scales == WASSERSTEIN_NUM_SCALES {
            assert!(
                source_height >= WASSERSTEIN_MIN_IMAGE_SIZE
                    && source_width >= WASSERSTEIN_MIN_IMAGE_SIZE,
                "Wasserstein Distance requires images of at least {WASSERSTEIN_MIN_IMAGE_SIZE}x{WASSERSTEIN_MIN_IMAGE_SIZE} for three scales, got {source_height}x{source_width}"
            );
        }

        let device = imgs_a.device();
        let mean = Tensor::<1>::from_floats([0.485, 0.456, 0.406], &device).reshape([1, 3, 1, 1]);
        let std = Tensor::<1>::from_floats([0.229, 0.224, 0.225], &device).reshape([1, 3, 1, 1]);
        let mut current_a = (imgs_a.permute([0, 3, 1, 2]) - mean.clone()) / std.clone();
        let mut current_b = (imgs_b.permute([0, 3, 1, 2]) - mean) / std;

        let kernel = lowpass_kernel(&device);

        let log2_sigma =
            effective_log2_sigma(source_height, source_width, source_height, source_width);
        let mut distance = wasserstein_distortion_feature(
            current_a.clone(),
            current_b.clone(),
            log2_sigma,
            &kernel,
        );

        for scale in 0..num_scales {
            let [_, _, height, width] = current_a.dims();
            assert!(
                height >= 16 && width >= 16,
                "Wasserstein Distance image scale {scale} is too small for all VGG slices: {height}x{width}"
            );

            let mut features_a = current_a.clone();
            let mut features_b = current_b.clone();
            for (index, block) in self.blocks.iter().enumerate() {
                if index != 0 {
                    features_a = avg_pool2d(features_a, [2, 2], [2, 2], [0, 0], true, false);
                    features_b = avg_pool2d(features_b, [2, 2], [2, 2], [0, 0], true, false);
                }

                // Keep the target path separate. Batching it with the tracked
                // prediction would make autodiff retain and traverse both.
                features_a = block.forward(features_a);
                features_b = block.forward(features_b);
                let [_, _, feature_height, feature_width] = features_a.dims();
                let log2_sigma = effective_log2_sigma(
                    source_height,
                    source_width,
                    feature_height,
                    feature_width,
                );
                distance = distance
                    + wasserstein_distortion_feature(
                        features_a.clone(),
                        features_b.clone(),
                        log2_sigma,
                        &kernel,
                    );
            }

            if scale + 1 < num_scales {
                current_a = lowpass_2d(current_a, 2, &kernel);
                current_b = lowpass_2d(current_b, 2, &kernel);
            }
        }

        distance
    }
}

impl LpipsModel {
    pub fn new(device: &Device) -> Self {
        // Could have different variations here but just doing VGG for now.
        let blocks = [
            (2, 3, 64),
            (2, 64, 128),
            (3, 128, 256),
            (3, 256, 512),
            (3, 512, 512),
        ]
        .iter()
        .map(|&(num_blocks, in_channels, out_channels)| {
            VggBlockConfig::new(num_blocks, in_channels, out_channels).init(device)
        })
        .collect();

        let heads = [64, 128, 256, 512, 512]
            .iter()
            .map(|&channels| {
                Conv2dConfig::new([channels, 1], [1, 1])
                    .with_stride([1, 1])
                    .with_bias(false)
                    .init(device)
            })
            .collect();

        Self {
            blocks,
            heads,
            max_pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }
}

pub fn load_vgg_lpips(device: &Device) -> LpipsModel {
    use burn::record::{BinBytesRecorder, HalfPrecisionSettings, Recorder};
    let model = LpipsModel::new(device);

    #[allow(clippy::large_include_file)]
    let bytes = include_bytes!("../burn_mapped.bin");

    model
        .load_record(
            BinBytesRecorder::<HalfPrecisionSettings, &[u8]>::default()
                .load(bytes, device)
                .expect("Should decode state successfully"),
        )
        .no_grad()
}

#[cfg(test)]
mod tests {
    use super::{
        WASSERSTEIN_NUM_LEVELS, load_vgg_lpips, lowpass_2d, lowpass_kernel,
        wasserstein_distortion_feature, wasserstein_distortion_feature_with_num_levels,
    };
    use burn::tensor::{Device, Tensor, TensorData};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    static APPLE_PNG: &[u8] = include_bytes!("../apple.png");
    static PEAR_PNG: &[u8] = include_bytes!("../pear.png");

    fn image_to_tensor(device: &Device, img: &image::DynamicImage) -> Tensor<4> {
        let rgb_img = img.to_rgb32f();
        let (w, h) = rgb_img.dimensions();
        let data = TensorData::new(rgb_img.into_vec(), [1, h as usize, w as usize, 3]);
        Tensor::from_data(data, device)
    }

    async fn read_scalar(t: Tensor<1>) -> f32 {
        t.into_scalar_async::<f32>().await.expect("readback")
    }

    fn pattern_nchw(device: &Device, size: usize, multiplier: usize, offset: usize) -> Tensor<4> {
        let values = (0..3 * size * size)
            .map(|index| ((index * multiplier + offset) % 256) as f32 / 255.0)
            .collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(values, [1, 3, size, size]), device)
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_structural_properties() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let image1 = image::load_from_memory(APPLE_PNG).expect("Failed to load apple.png");
        let image2 = image::load_from_memory(PEAR_PNG).expect("Failed to load pear.png");
        let apple = image_to_tensor(&device, &image1);
        let pear = image_to_tensor(&device, &image2);
        let model = load_vgg_lpips(&device);

        // Identity: LPIPS(x, x) == 0.
        let identity = read_scalar(model.lpips(apple.clone(), apple.clone())).await;
        assert!(identity.abs() < 1e-5, "LPIPS(apple, apple) = {identity}");

        // Symmetry: LPIPS(a, b) == LPIPS(b, a).
        let ab = read_scalar(model.lpips(apple.clone(), pear.clone())).await;
        let ba = read_scalar(model.lpips(pear, apple)).await;
        assert!((ab - ba).abs() < 1e-5, "asymmetric: ab = {ab}, ba = {ba}");
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_matches_pytorch_reference() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let image1 = image::load_from_memory(APPLE_PNG).expect("Failed to load apple.png");
        let image2 = image::load_from_memory(PEAR_PNG).expect("Failed to load pear.png");
        let apple = image_to_tensor(&device, &image1);
        let pear = image_to_tensor(&device, &image2);
        let model = load_vgg_lpips(&device);
        let score = read_scalar(model.lpips(apple, pear)).await;
        assert!(
            (score - 0.657_102).abs() < 1e-4,
            "LPIPS(apple, pear) = {score}, PyTorch reference 0.6571019887924194",
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn wasserstein_lowpass_matches_balle_lab_reference() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let input = Tensor::from_data(
            TensorData::new((0..9).map(|value| value as f32).collect(), [1, 1, 3, 3]),
            &device,
        );
        let kernel = lowpass_kernel(&device);

        let stride_one = lowpass_2d(input.clone(), 1, &kernel)
            .into_data_async()
            .await
            .expect("stride-one readback")
            .to_vec::<f32>()
            .expect("f32 lowpass");
        let stride_two = lowpass_2d(input, 2, &kernel)
            .into_data_async()
            .await
            .expect("stride-two readback")
            .to_vec::<f32>()
            .expect("f32 lowpass");

        assert_eq!(stride_one, [0.75, 1.5, 1.5, 2.5, 4.0, 3.5, 3.0, 4.5, 3.75]);
        assert_eq!(stride_two, [0.75, 1.5, 3.0, 3.75]);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn wasserstein_feature_matches_balle_lab_reference() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let values_a = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let values_b = (0..4)
            .flat_map(|row| (0..4).rev().map(move |column| (row * 4 + column) as f32))
            .collect::<Vec<_>>();
        let a = Tensor::from_data(TensorData::new(values_a, [1, 1, 4, 4]), &device);
        let b = Tensor::from_data(TensorData::new(values_b, [1, 1, 4, 4]), &device);
        let kernel = lowpass_kernel(&device);

        let identity = read_scalar(wasserstein_distortion_feature(
            a.clone(),
            a.clone(),
            2.0,
            &kernel,
        ))
        .await;
        let ab = read_scalar(wasserstein_distortion_feature(
            a.clone(),
            b.clone(),
            2.0,
            &kernel,
        ))
        .await;
        let ba = read_scalar(wasserstein_distortion_feature(b, a, 2.0, &kernel)).await;

        assert!(identity.abs() < 1e-6, "WD feature identity = {identity}");
        assert!(ab >= 0.0, "WD feature distance is negative: {ab}");
        assert!(
            (ab - ba).abs() < 1e-6,
            "WD feature is asymmetric: {ab} vs {ba}"
        );
        assert!(
            (ab - 0.053_318_12).abs() < 1e-5,
            "WD feature distance = {ab}, balle-lab reference 0.053318120539188385"
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn wasserstein_level_pruning_matches_five_level_reference() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let values_a = (0..2 * 9 * 7)
            .map(|index| ((index * 37) % 101) as f32 / 100.0)
            .collect::<Vec<_>>();
        let values_b = (0..2 * 9 * 7)
            .map(|index| ((index * 73 + 19) % 103) as f32 / 102.0)
            .collect::<Vec<_>>();
        let a = Tensor::from_data(TensorData::new(values_a.clone(), [1, 2, 9, 7]), &device);
        let b = Tensor::from_data(TensorData::new(values_b.clone(), [1, 2, 9, 7]), &device);
        let kernel = lowpass_kernel(&device);

        for sigma in [0.0, 0.5, 1.0, 1.5, 2.0] {
            let pruned = read_scalar(wasserstein_distortion_feature(
                a.clone(),
                b.clone(),
                sigma,
                &kernel,
            ))
            .await;
            let full = read_scalar(wasserstein_distortion_feature_with_num_levels(
                a.clone(),
                b.clone(),
                sigma,
                &kernel,
                WASSERSTEIN_NUM_LEVELS,
            ))
            .await;
            assert!(
                (pruned - full).abs() < 1e-6,
                "sigma {sigma}: pruned WD {pruned} != five-level WD {full}"
            );
        }

        let autodiff_device = device.autodiff();
        let pruned_input = Tensor::from_data(
            TensorData::new(values_a.clone(), [1, 2, 9, 7]),
            &autodiff_device,
        )
        .require_grad();
        let full_input =
            Tensor::from_data(TensorData::new(values_a, [1, 2, 9, 7]), &autodiff_device)
                .require_grad();
        let target = Tensor::from_data(TensorData::new(values_b, [1, 2, 9, 7]), &autodiff_device);
        let kernel = lowpass_kernel(&autodiff_device);
        let pruned_loss =
            wasserstein_distortion_feature(pruned_input.clone(), target.clone(), 1.5, &kernel);
        let full_loss = wasserstein_distortion_feature_with_num_levels(
            full_input.clone(),
            target,
            1.5,
            &kernel,
            WASSERSTEIN_NUM_LEVELS,
        );
        let pruned_grads = pruned_loss.backward();
        let full_grads = full_loss.backward();
        let pruned_gradient = pruned_input
            .grad(&pruned_grads)
            .expect("pruned feature gradient")
            .into_data_async()
            .await
            .expect("pruned gradient readback")
            .to_vec::<f32>()
            .expect("f32 pruned gradient");
        let full_gradient = full_input
            .grad(&full_grads)
            .expect("five-level feature gradient")
            .into_data_async()
            .await
            .expect("five-level gradient readback")
            .to_vec::<f32>()
            .expect("f32 five-level gradient");
        for (index, (pruned, full)) in pruned_gradient.iter().zip(full_gradient).enumerate() {
            assert!(
                (pruned - full).abs() < 1e-6,
                "gradient {index}: pruned {pruned} != five-level {full}"
            );
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn wasserstein_distance_matches_balle_lab_reference() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let a = pattern_nchw(&device, 64, 37, 0).permute([0, 2, 3, 1]);
        let b = pattern_nchw(&device, 64, 73, 19).permute([0, 2, 3, 1]);
        let model = load_vgg_lpips(&device);

        let score = read_scalar(model.wasserstein_distance(a, b)).await;
        assert!(score.is_finite(), "WD score is not finite: {score}");
        assert!(score >= 0.0, "WD score is negative: {score}");
        assert!(
            (score - 10.540_909).abs() < 2e-3,
            "WD score = {score}, balle-lab reference 10.540908813476562"
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn wasserstein_distance_has_finite_input_gradient_and_frozen_vgg() {
        let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let prediction = pattern_nchw(&device, 32, 37, 0)
            .permute([0, 2, 3, 1])
            .require_grad();
        let target = pattern_nchw(&device, 32, 73, 19).permute([0, 2, 3, 1]);
        let model = load_vgg_lpips(&device);

        let loss = model.wasserstein_distance_with_num_scales(prediction.clone(), target, 1);
        let grads = loss.backward();
        let gradient = prediction
            .grad(&grads)
            .expect("WD must preserve prediction gradients")
            .into_data_async()
            .await
            .expect("gradient readback")
            .to_vec::<f32>()
            .expect("f32 gradient");

        assert!(
            gradient.iter().all(|value| value.is_finite()),
            "WD input gradient contains a non-finite value"
        );
        assert!(
            gradient.iter().any(|value| value.abs() > 1e-10),
            "WD input gradient is identically zero"
        );
        assert!(
            model.blocks[0].convs[0].weight.val().grad(&grads).is_none(),
            "WD must not compute VGG parameter gradients"
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    #[allow(clippy::excessive_precision)] // Preserve the pinned reference gradient verbatim.
    async fn wasserstein_feature_gradient_matches_balle_lab_reference() {
        let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let values_a = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let values_b = (0..4)
            .flat_map(|row| (0..4).rev().map(move |column| (row * 4 + column) as f32))
            .collect::<Vec<_>>();
        let a = Tensor::from_data(TensorData::new(values_a, [1, 1, 4, 4]), &device).require_grad();
        let b = Tensor::from_data(TensorData::new(values_b, [1, 1, 4, 4]), &device);
        let kernel = lowpass_kernel(&device);

        let loss = wasserstein_distortion_feature(a.clone(), b, 2.0, &kernel);
        let gradients = loss.backward();
        let actual = a
            .grad(&gradients)
            .expect("feature gradient")
            .into_data_async()
            .await
            .expect("feature-gradient readback")
            .to_vec::<f32>()
            .expect("f32 feature gradient");
        let expected = [
            -0.0016647941,
            -0.0016847791,
            -0.0010306847,
            -0.0006650249,
            -0.0069068656,
            -0.0053582182,
            -0.0029921830,
            -0.0016966688,
            -0.0141097195,
            -0.0104192067,
            -0.0057679983,
            -0.0031354714,
            -0.0099821398,
            -0.0071761874,
            -0.0038898878,
            -0.0020706800,
        ];

        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "feature gradient {index}: {actual} != {expected}"
            );
        }
    }
}
