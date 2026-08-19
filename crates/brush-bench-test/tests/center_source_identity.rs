//! WS-A byte-identity harness for `--depth-source center` (the default).
//!
//! # What this is for, and why it is a printed number rather than an assertion
//!
//! Plan §5 asks for a "defaults-only replay of the playroom recipe (same seed)
//! before/after the merge; loss logs must be identical". **That gate is not
//! achievable as written against this trainer, and that is a property of the
//! trainer, not of any workstream's change.** Measured on this Mac 2026-08-19:
//! the SAME `brush-cli` binary, run twice on `playroom_0812` with `--seed 7`,
//! `--max-resolution 1920`, 100 iterations, exports two plys of the same splat
//! count in which **99.09% of the 117,970,264 floats differ, worst component by
//! 0.61**. That is view-order divergence, not last-bit drift — the four
//! dataloader threads race for batch slots, so a full-run replay cannot be a
//! bit-comparison for anyone.
//!
//! So the gate is enforced one level down, where determinism does hold: a fixed
//! camera, a fixed batch, a fixed splat set, and `SplatTrainer::step` driven
//! directly. This test prints the resulting loss as raw IEEE-754 bits. Run it on
//! the base commit and on the change; the printed lines must match character for
//! character.
//!
//! It is deliberately NOT a golden-constant assertion: the value depends on the
//! GPU, the driver and the autotuner's kernel choices, so a hardcoded expectation
//! would be a cross-machine tripwire rather than a regression gate. The
//! comparison is between two builds on ONE machine, which is exactly what
//! byte-identity means here.
//!
//! The configuration below deliberately turns on every term WS-A's edit touches:
//! the depth loss (hoisted feature render, dispatched depth source), the
//! depth/normal consistency term (second depth consumer), and the TV normal
//! smoothness (so the normal render is live and its alpha is consumed).

#![allow(clippy::missing_assert_message)]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::tensor::{Device, TensorData};
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};

const SEED: u64 = 20_260_819;
const IMG: glam::UVec2 = glam::uvec2(48, 48);

fn test_camera() -> Camera {
    Camera::new(
        Vec3::new(0.2, -0.3, -5.0),
        Quat::from_euler(glam::EulerRot::XYZ, 0.10, -0.18, 0.05),
        0.7,
        0.7,
        glam::vec2(0.5, 0.5),
        Pinhole,
    )
}

fn test_splats(device: &Device) -> Splats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let count = 64;
    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-1.2..1.2),
                rng.random_range(-1.2..1.2),
                rng.random_range(-0.6..0.6),
            ]
        })
        .collect();
    // Well-separated log scales: the thinnest-axis argmin must not sit on a tie.
    let log_scales: Vec<f32> = (0..count).flat_map(|_| [-1.2, -1.6, -3.0]).collect();
    let rotations: Vec<f32> = (0..count)
        .flat_map(|_| {
            let q = Quat::from_euler(
                glam::EulerRot::XYZ,
                rng.random_range(-0.5..0.5),
                rng.random_range(-0.5..0.5),
                rng.random_range(-0.5..0.5),
            );
            [q.w, q.x, q.y, q.z]
        })
        .collect();
    let sh: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
            ]
        })
        .collect();
    let opac: Vec<f32> = (0..count).map(|_| 2.0).collect();
    Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh,
        opac,
        SplatRenderMode::Default,
        device,
    )
}

#[tokio::test]
async fn center_depth_source_step_loss_bits() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut splats = test_splats(&device);

    let config = TrainConfig {
        // `depth_source` is left at its Default (== Center) ON PURPOSE: naming
        // it here would pin the field rather than the default.
        depth_loss_weight: 1.0,
        depth_normal_weight: 0.05,
        normal_smooth_weight: 0.1,
        ..Default::default()
    };
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );

    let (h, w) = (IMG.y as usize, IMG.x as usize);
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED ^ 0x5eed);
    let img_packed = TensorData::new(
        (0..h * w)
            .map(|_| rng.random_range(0..0x00ff_ffffi32) | 0xff00_0000u32 as i32)
            .collect::<Vec<i32>>(),
        [h, w],
    );
    // A depth map in front of the camera with a tilt, plus a band of invalid
    // (0) pixels so the masked-denominator path is exercised too.
    let depth: Vec<f32> = (0..h * w)
        .map(|i| {
            let (y, x) = (i / w, i % w);
            if y % 11 == 0 {
                0.0
            } else {
                4.2 + 0.01 * x as f32 + 0.004 * y as f32
            }
        })
        .collect();

    let batch = SceneBatch {
        img_packed,
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        features: None,
        depth: Some(TensorData::new(depth, [h, w])),
        normal: None,
        camera: test_camera(),
        view_index: 0,
    };

    // --- The gate: the FIRST step's loss, from a fresh trainer. ---
    //
    // Only step 0 is a bit-comparison. Measured here 2026-08-19: repeated runs
    // of this same binary agree exactly on step 0 but drift by ~1 ULP on steps 1
    // and 2 (`0x3ebdceae` vs `0x3ebdcead`). That is the rasterize backward's
    // atomic gradient accumulation — the order threads add into a gradient
    // buffer is not fixed, so the optimizer's state diverges in the last bit and
    // every later forward inherits it. Nothing to do with `--depth-source`; it is
    // why the full-run replay in §5 cannot be a bit-comparison either.
    let mut first: Option<u32> = None;
    for rep in 0..3 {
        let mut fresh = SplatTrainer::new(
            &config,
            &device,
            BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
        );
        let (_next, stats) = fresh.step(batch.clone(), test_splats(&device)).await;
        let loss: f32 = stats.loss.into_scalar_async().await.expect("loss readback");
        assert!(loss.is_finite(), "rep {rep} produced a non-finite loss");
        let bits = loss.to_bits();
        println!("CENTER_IDENTITY step0 rep={rep} loss_bits=0x{bits:08x} loss={loss}");
        match first {
            None => first = Some(bits),
            Some(want) => assert_eq!(
                bits, want,
                "the step-0 forward loss must be bit-reproducible on one machine;                  without that there is no byte-identity gate to run at all"
            ),
        }
    }

    // Informational: the continued sequence. Compare across builds only to ~1
    // ULP, for the reason above.
    for step in 0..3 {
        let (next, stats) = trainer.step(batch.clone(), splats).await;
        splats = next;
        let loss: f32 = stats.loss.into_scalar_async().await.expect("loss readback");
        assert!(loss.is_finite(), "step {step} produced a non-finite loss");
        println!(
            "CENTER_IDENTITY seq step={step} loss_bits=0x{:08x}",
            loss.to_bits()
        );
    }
}
