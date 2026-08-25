//! Dispatch-tiling regression tests for the PPISP kernels.
//!
//! A flat `h*w/BLOCK_SIZE` workgroup count exceeds wgpu's 65535-per-dimension
//! dispatch limit above a 2896px square face: a 2816px cube face needs 61,952
//! workgroups (94.5% of the cap) and a 3840px one needs 115,200. Over the cap
//! the device-runner thread dies and every tensor afterwards is garbage, which
//! surfaces downstream as a non-finite loss rather than as a dispatch error.
//!
//! `SUB_LIMIT` exercises the 1D path (unchanged), `OVER_LIMIT` the 2D-tiled
//! one. Both print an exact bitwise checksum so the two code paths can be
//! diffed across a code change.

use brush_appearance::ppisp::{PpispStages, ppisp_apply};
use burn::tensor::{Device, Tensor};

/// 61,952 workgroups — fits the 65535 cap, so this dispatch stays 1D.
const SUB_LIMIT: usize = 2816;
/// 115,200 workgroups — over the cap, so this dispatch must be tiled.
const OVER_LIMIT: usize = 3840;

const FRAME_ONLY: PpispStages = PpispStages {
    frame: true,
    vignetting: false,
    crf: false,
};

async fn ad_device() -> Device {
    Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

/// Cheap deterministic per-pixel values (PCG-ish, matches `reference.rs`).
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

/// FNV-1a over the raw f32 bits: equal checksums mean bit-identical buffers.
fn checksum(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

async fn read<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("f32 data")
}

/// Forward + backward at `size x size`, checked against an exact CPU reference
/// and reported as a bitwise checksum.
async fn run_case(label: &str, size: usize) {
    let device = ad_device().await;
    let count = size * size * 3;
    let rgb_data = pattern(count, 41, 0.05, 0.8);
    let weight_data = pattern(count, 47, -1.0, 1.0);
    let exposure_value = 0.75f32;

    let exposure = Tensor::from_floats([exposure_value], &device).require_grad();
    let vignetting = Tensor::zeros([1, 3, 5], &device);
    let color = Tensor::zeros([1, 8], &device);
    let crf = Tensor::zeros([1, 3, 4], &device);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device)
        .reshape([size, size, 3])
        .require_grad();
    let weights =
        Tensor::<1>::from_floats(weight_data.as_slice(), &device).reshape([size, size, 3]);

    let out = ppisp_apply(
        exposure.clone(),
        vignetting,
        color,
        crf,
        rgb.clone(),
        0,
        0,
        FRAME_ONLY,
    );
    let grads = (out.clone() * weights.clone()).sum().backward();
    let out_values = read(out).await;
    let rgb_grad = read(rgb.grad(&grads).expect("rgb gradient")).await;
    let exposure_grad = read(exposure.grad(&grads).expect("exposure gradient")).await[0];

    // Every pixel must be visited exactly once: `out` and `rgb` share the flat
    // index, so a mis-tiled index leaves holes (still zero from `alloc_zeros`)
    // or double-writes, and both show up here.
    let gain = (exposure_value * std::f32::consts::LN_2).exp();
    // 2e-4, the same tolerance `reference.rs` uses for PPISP: the GPU's
    // `exp()` and Rust's differ by a few f32 ulp on the shared gain factor
    // (measured 3.4e-6 on the forward, 5.0e-5 through the gradient). An
    // indexing fault is O(0.1) — a pixel either is never written (and stays
    // 0.0) or takes some other pixel's value — so this cannot mask one.
    let tolerance = 2e-4;
    let mut worst = 0.0f32;
    for (index, (got, source)) in out_values.iter().zip(&rgb_data).enumerate() {
        let error = (got - source * gain).abs();
        assert!(
            error <= tolerance,
            "{label}: forward mismatch at {index}: got {got}, want {}",
            source * gain
        );
        worst = worst.max(error);
        assert!(
            got.is_finite(),
            "{label}: non-finite forward value at {index}"
        );
    }

    // dL/drgb is the incoming weight scaled by the exposure gain.
    let mut worst_grad = 0.0f32;
    for (index, (got, weight)) in rgb_grad.iter().zip(&weight_data).enumerate() {
        let error = (got - weight * gain).abs();
        assert!(
            error <= tolerance,
            "{label}: rgb-grad mismatch at {index}: got {got}, want {}",
            weight * gain
        );
        worst_grad = worst_grad.max(error);
    }

    // dL/dexposure comes out of the per-cube `partials` buffer, which has one
    // row per *untiled* cube — the extra cubes a 2D tiling adds must not write
    // past the end, and must not be missing from the sum either.
    let reference: f64 = rgb_data
        .iter()
        .zip(&weight_data)
        .map(|(source, weight)| f64::from(*source) * f64::from(gain) * f64::from(*weight))
        .sum::<f64>()
        * f64::from(std::f32::consts::LN_2);
    let relative = ((f64::from(exposure_grad) - reference) / reference).abs();
    assert!(
        relative < 1e-3,
        "{label}: exposure grad {exposure_grad} vs reference {reference} (rel {relative})"
    );

    println!(
        "TILING {label} size={size} cubes={} fwd_chk={:016x} rgbgrad_chk={:016x} \
         exp_grad_bits={:08x} exp_grad={exposure_grad:.9e} max_fwd_err={worst:e} \
         max_grad_err={worst_grad:e} exp_grad_rel={relative:e}",
        (size * size).div_ceil(128),
        checksum(&out_values),
        checksum(&rgb_grad),
        exposure_grad.to_bits(),
    );
}

/// One test, both cases in sequence: each case peaks around 1.4 GB of image
/// buffers, and `cargo test` would otherwise run two of them concurrently.
#[tokio::test]
async fn ppisp_dispatch_tiling_spans_the_workgroup_limit() {
    run_case("sub_limit", SUB_LIMIT).await;
    run_case("over_limit", OVER_LIMIT).await;
}
