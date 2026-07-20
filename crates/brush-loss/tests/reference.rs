//! Smoke + invariant tests for the loss kernels.
//!
//! GT lives as `[H, W]` u32 packing `[r g b a]` u8. We feed deterministic u8
//! data through `image_loss` and check structural properties (`SSIM(x, x) ≈ 1`,
//! output range, backward produces finite gradients). Bit-exact reference
//! matching is covered by the integration training tests in `brush-bench-test`.

use brush_loss::{ImageLossConfig, image_loss};
use burn::tensor::{Device, Int, Tensor, TensorData};
use glam::Vec3;
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn pack_rgba(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|p| {
            u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16 | u32::from(p[3]) << 24
        })
        .collect()
}

/// Deterministic u8 pattern (avoids RNG so the test is reproducible across
/// machines). Returns `H*W*4` RGBA bytes.
fn make_pattern(h: usize, w: usize, scale: u32, offset: u32) -> Vec<u8> {
    (0..h * w * 4)
        .map(|i| ((i as u32 * scale + offset) % 251) as u8)
        .collect()
}

fn pred_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<3> {
    let rgb: Vec<f32> = bytes
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]].map(|b| b as f32 / 255.0))
        .collect();
    Tensor::<1>::from_floats(rgb.as_slice(), device).reshape([h, w, 3])
}

fn gt_packed_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<2, Int> {
    // Bit-reinterpret the u32 packing as i32 so the dispatch int_from_data
    // path doesn't reject magnitudes > i32::MAX.
    let packed: Vec<i32> = pack_rgba(bytes).into_iter().map(|x| x as i32).collect();
    Tensor::from_data(TensorData::new(packed, [h, w]), device)
}

fn ssim_only_cfg() -> ImageLossConfig {
    ImageLossConfig {
        l1_weight: 0.0,
        ssim_weight: 1.0,
        composite_bg: None,
        mask: false,
    }
}

fn vjp_inputs(h: usize, w: usize, channels: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let mut pred = Vec::with_capacity(h * w * channels);
    let mut gt = Vec::with_capacity(h * w * 4);
    for pixel in 0..h * w {
        // Keep each prediction away from its effective GT so +/- epsilon does
        // not cross the L1 kink during the finite-difference checks.
        let offset = (pixel % 17) as f32 * 0.002;
        pred.extend_from_slice(&[0.75 + offset, 0.65 + offset, 0.10 + offset]);
        if channels == 4 {
            pred.push(0.05 + offset);
        }
        gt.extend_from_slice(&[
            (30 + (pixel * 7) % 31) as u8,
            (80 + (pixel * 11) % 31) as u8,
            (130 + (pixel * 13) % 31) as u8,
            (100 + (pixel * 17) % 131) as u8,
        ]);
    }
    let chain = (0..h * w * channels)
        .map(|i| {
            let magnitude = 0.25 + ((i * 19 + 7) % 31) as f32 / 31.0;
            if i % 2 == 0 { magnitude } else { -magnitude }
        })
        .collect();
    (pred, gt, chain)
}

async fn dot_loss(
    pred_data: &[f32],
    gt_bytes: &[u8],
    chain: &[f32],
    shape: (usize, usize, usize),
    cfg: ImageLossConfig,
    device: &Device,
) -> f64 {
    let (h, w, channels) = shape;
    let pred = Tensor::<1>::from_floats(pred_data, device).reshape([h, w, channels]);
    let gt = gt_packed_from_bytes(gt_bytes, h, w, device);
    let map: Vec<f32> = image_loss(pred, gt, cfg)
        .into_data_async()
        .await
        .expect("loss-map readback")
        .to_vec()
        .expect("loss-map data");
    map.iter()
        .zip(chain)
        .map(|(&value, &weight)| f64::from(value) * f64::from(weight))
        .sum()
}

async fn analytical_vjp(
    pred_data: &[f32],
    gt_bytes: &[u8],
    chain: &[f32],
    shape: (usize, usize, usize),
    cfg: ImageLossConfig,
    device: &Device,
) -> Vec<f32> {
    let (h, w, channels) = shape;
    let pred = Tensor::<1>::from_floats(pred_data, device)
        .reshape([h, w, channels])
        .require_grad();
    let gt = gt_packed_from_bytes(gt_bytes, h, w, device);
    let dl_dmap = Tensor::<1>::from_floats(chain, device).reshape([h, w, channels]);
    let grads = (image_loss(pred.clone(), gt, cfg) * dl_dmap)
        .sum()
        .backward();
    pred.grad(&grads)
        .expect("prediction gradient")
        .into_data_async()
        .await
        .expect("gradient readback")
        .to_vec()
        .expect("gradient data")
}

async fn assert_l1_forward_matches_cpu(
    pred_data: &[f32],
    gt_bytes: &[u8],
    shape: (usize, usize, usize),
    cfg: ImageLossConfig,
    device: &Device,
) {
    let (h, w, channels) = shape;
    let pred = Tensor::<1>::from_floats(pred_data, device).reshape([h, w, channels]);
    let gt = gt_packed_from_bytes(gt_bytes, h, w, device);
    let actual: Vec<f32> = image_loss(pred, gt, cfg)
        .into_data_async()
        .await
        .expect("L1 map readback")
        .to_vec()
        .expect("L1 map data");
    let background = cfg.composite_bg.unwrap_or(Vec3::ZERO);
    let background = [background.x, background.y, background.z];

    for pixel in 0..h * w {
        let alpha = f32::from(gt_bytes[pixel * 4 + 3]) / 255.0;
        for channel in 0..channels {
            let gt = if channel == 3 {
                alpha
            } else {
                let base = f32::from(gt_bytes[pixel * 4 + channel]) / 255.0;
                match cfg.composite_bg {
                    Some(_) => base + (1.0 - alpha) * background[channel],
                    None => base,
                }
            };
            let weight = if channel == 3 { 1.0 } else { cfg.l1_weight };
            let mask = if cfg.mask { alpha } else { 1.0 };
            let index = pixel * channels + channel;
            let expected = weight * (pred_data[index] - gt).abs() * mask;
            assert!(
                (actual[index] - expected).abs() < 2e-6,
                "pixel={pixel} channel={channel}: actual={}, expected={expected}",
                actual[index]
            );
        }
    }
}

async fn check_vjp_case(
    shape: (usize, usize, usize),
    cfg: ImageLossConfig,
    probes: &[(usize, usize, usize)],
    device: &Device,
) {
    // The loss map is f32. A relatively wide epsilon keeps its per-pixel
    // roundoff from dominating the SSIM-only VJP while staying far from the
    // deliberately avoided L1 kinks in `vjp_inputs`.
    const EPSILON: f32 = 1e-2;

    let (h, w, channels) = shape;
    let (base, gt, chain) = vjp_inputs(h, w, channels);
    let analytical = analytical_vjp(&base, &gt, &chain, shape, cfg, device).await;
    for &(y, x, channel) in probes {
        let index = (y * w + x) * channels + channel;
        let mut plus = base.clone();
        plus[index] += EPSILON;
        let mut minus = base.clone();
        minus[index] -= EPSILON;
        let numerical = ((dot_loss(&plus, &gt, &chain, shape, cfg, device).await
            - dot_loss(&minus, &gt, &chain, shape, cfg, device).await)
            / (2.0 * f64::from(EPSILON))) as f32;
        let actual = analytical[index];
        let scale = numerical.abs().max(actual.abs()).max(1e-6);
        let tolerance = 1e-3 + 0.01 * scale;
        assert!(
            (numerical - actual).abs() <= tolerance,
            "{h}x{w}x{channels} [{y},{x},{channel}]: numerical={numerical}, analytical={actual}, tolerance={tolerance}"
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_identical_inputs_is_one() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes = make_pattern(h, w, 11, 13);
    let pred = pred_from_bytes(&bytes, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = image_loss(pred, gt, ssim_only_cfg());
    let mean: f32 = map
        .into_data_async()
        .await
        .expect("readback")
        .iter::<f32>()
        .sum::<f32>()
        / (h * w * 3) as f32;
    // Identical inputs SSIM saturates at 1; allow a sub-ULP roundoff.
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "SSIM(x, x) should be 1, got {mean}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_in_clamp_range() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes_a = make_pattern(h, w, 7, 19);
    let bytes_b = make_pattern(h, w, 13, 7);
    let pred = pred_from_bytes(&bytes_a, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let data: Vec<f32> = image_loss(pred, gt, ssim_only_cfg())
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("vec");
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (-1.0..=1.0).contains(&min) && (-1.0..=1.0).contains(&max),
        "SSIM out of [-1, 1]: min={min} max={max}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn image_loss_backward_runs() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (32, 48);
    let bytes_a = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let pred = pred_from_bytes(&bytes_a, h, w, &device).require_grad();
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let map = image_loss(
        pred.clone(),
        gt,
        ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: None,
            mask: false,
        },
    );
    let grads = map.mean().backward();
    let grad = pred.grad(&grads).expect("pred should have a gradient");
    let data: Vec<f32> = grad
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("vec");
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs > 0.0,
        "backward should produce non-zero gradients, got all zeros"
    );
    assert!(
        data.iter().all(|v| v.is_finite()),
        "gradients should be finite"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn image_loss_direct_vjp_matches_finite_difference() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cfg = |composite_bg, mask| ImageLossConfig {
        l1_weight: 0.8,
        ssim_weight: -0.2,
        composite_bg,
        mask,
    };
    let background = Some(Vec3::new(0.05, 0.10, 0.15));
    let ssim_only_composited = ImageLossConfig {
        l1_weight: 0.0,
        ssim_weight: 1.0,
        composite_bg: background,
        mask: false,
    };

    check_vjp_case(
        (1, 1, 3),
        cfg(None, false),
        &[(0, 0, 0), (0, 0, 2)],
        &device,
    )
    .await;
    check_vjp_case(
        (3, 7, 4),
        cfg(background, true),
        &[(0, 0, 0), (1, 3, 3), (2, 6, 2)],
        &device,
    )
    .await;
    check_vjp_case(
        (17, 19, 3),
        ssim_only_composited,
        &[(0, 0, 0), (15, 16, 1), (16, 18, 2)],
        &device,
    )
    .await;
    check_vjp_case(
        (33, 35, 3),
        cfg(None, true),
        &[(16, 16, 0), (32, 34, 2)],
        &device,
    )
    .await;
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn l1_only_specialization_matches_finite_difference() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let l1_only = ImageLossConfig {
        l1_weight: 0.73,
        ssim_weight: 0.0,
        composite_bg: Some(Vec3::new(0.07, 0.11, 0.19)),
        mask: true,
    };
    check_vjp_case(
        (7, 9, 4),
        l1_only,
        &[(0, 0, 0), (3, 4, 1), (6, 8, 2), (2, 5, 3)],
        &device,
    )
    .await;
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn l1_only_forward_matches_cpu_oracle() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let gt = [
        10, 100, 200, 0, 250, 50, 25, 64, 30, 90, 150, 192, 240, 180, 60, 255,
    ];
    let rgb = [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.7, 0.3, 0.05, 0.4, 0.6, 0.95];
    assert_l1_forward_matches_cpu(
        &rgb,
        &gt,
        (2, 2, 3),
        ImageLossConfig {
            l1_weight: 0.73,
            ssim_weight: 0.0,
            composite_bg: None,
            mask: false,
        },
        &device,
    )
    .await;
    assert_l1_forward_matches_cpu(
        &rgb,
        &gt,
        (2, 2, 3),
        ImageLossConfig {
            l1_weight: 0.41,
            ssim_weight: 0.0,
            composite_bg: Some(Vec3::new(0.1, 0.25, 0.8)),
            mask: true,
        },
        &device,
    )
    .await;

    let rgba = [
        0.9, 0.1, 0.4, 0.8, 0.2, 0.8, 0.5, 0.1, 0.7, 0.3, 0.05, 0.9, 0.4, 0.6, 0.95, 0.2,
    ];
    assert_l1_forward_matches_cpu(
        &rgba,
        &gt,
        (2, 2, 4),
        ImageLossConfig {
            l1_weight: 0.19,
            ssim_weight: 0.0,
            composite_bg: None,
            mask: false,
        },
        &device,
    )
    .await;
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn alpha_match_via_4ch_pred() {
    // Feeding 4-channel `pred` makes the kernel emit `|pred.a - gt.a|`
    // into the alpha channel of the loss map.
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (16, 24);
    let bytes = make_pattern(h, w, 17, 5);
    let rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
        .reshape([h, w, 4])
        .require_grad();
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = image_loss(
        pred,
        gt,
        ImageLossConfig {
            l1_weight: 1.0,
            ssim_weight: 0.0,
            composite_bg: None,
            mask: false,
        },
    );
    let _grads = map.mean().backward();
}
