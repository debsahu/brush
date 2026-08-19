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
    kernels::camera_model::CameraModel,
    kernels::camera_model::CameraModel::Pinhole,
    kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params,
};
use brush_train::{
    config::{DepthSource, TrainConfig},
    train::SplatTrainer,
};
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

/// The same camera with a `KannalaBrandt4` fisheye model instead of a pinhole.
///
/// Both the ray-plane grid and `normals_from_depth` assume a pinhole
/// unprojection, so a plane depth source must WARN AND FALL BACK here rather
/// than supervise with wrong math.
fn fisheye_camera() -> Camera {
    Camera::new(
        Vec3::new(0.2, -0.3, -5.0),
        Quat::from_euler(glam::EulerRot::XYZ, 0.10, -0.18, 0.05),
        0.7,
        0.7,
        glam::vec2(0.5, 0.5),
        CameraModel::KannalaBrandt4(KannalaBrandt4Params {
            k1: 0.05,
            k2: -0.01,
            k3: 0.002,
            k4: 0.0,
        }),
    )
}

/// A fixed batch: deterministic RGB, and a tilted depth map in front of the
/// camera with every 11th row invalid so the masked-denominator path is live.
fn make_batch(camera: Camera) -> SceneBatch {
    let (h, w) = (IMG.y as usize, IMG.x as usize);
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED ^ 0x5eed);
    let img_packed = TensorData::new(
        (0..h * w)
            .map(|_| rng.random_range(0..0x00ff_ffffi32) | 0xff00_0000u32 as i32)
            .collect::<Vec<i32>>(),
        [h, w],
    );
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

    SceneBatch {
        img_packed,
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        features: None,
        depth: Some(TensorData::new(depth, [h, w])),
        normal: None,
        camera,
        view_index: 0,
    }
}

/// One `SplatTrainer::step` from a fresh trainer and a fresh splat set, as raw
/// loss bits. Fresh because only step 0 is bit-reproducible (see the module
/// comment).
async fn step0_loss_bits(depth_source: DepthSource, camera: Camera, device: &Device) -> u32 {
    step0_loss(depth_source, camera, device).await.to_bits()
}

/// The shared config every dispatch pin drives: a depth loss, the depth/normal
/// consistency term, and TV normal smoothness, so every consumer of the
/// `--depth-source` dispatch is live.
fn dispatch_config(depth_source: DepthSource) -> TrainConfig {
    TrainConfig {
        depth_source,
        depth_loss_weight: 1.0,
        depth_normal_weight: 0.05,
        normal_smooth_weight: 0.1,
        ..Default::default()
    }
}

/// The step-0 loss as an `f32`. `step0_loss_bits` is this, reinterpreted — the
/// bit form is for the equality/inequality pins, the float form for the
/// tolerance pin (approaches A and B run DIFFERENT kernels over the same math,
/// so they are expected to agree closely, never bitwise).
async fn step0_loss(depth_source: DepthSource, camera: Camera, device: &Device) -> f32 {
    let config = dispatch_config(depth_source);
    let mut trainer = SplatTrainer::new(
        &config,
        device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );
    let (_next, stats) = trainer.step(make_batch(camera), test_splats(device)).await;
    let loss: f32 = stats.loss.into_scalar_async().await.expect("loss readback");
    assert!(loss.is_finite(), "step 0 produced a non-finite loss");
    loss
}

/// The RAW opacities after exactly ONE optimizer step under `depth_source`.
///
/// This is the step-level probe for the §4.5 backward contract: the forward is
/// (deliberately) almost the same for A and B, so only the post-step WEIGHTS can
/// show whether plane error reached opacity.
async fn step0_raw_opacities(
    depth_source: DepthSource,
    camera: Camera,
    device: &Device,
) -> Vec<f32> {
    let config = dispatch_config(depth_source);
    let mut trainer = SplatTrainer::new(
        &config,
        device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );
    let (next, _stats) = trainer.step(make_batch(camera), test_splats(device)).await;
    let opac = next
        .raw_opacities
        .val()
        .into_data_async()
        .await
        .expect("raw opacity readback")
        .into_vec::<f32>()
        .expect("raw opacity as f32");
    assert!(
        opac.iter().all(|v| v.is_finite()),
        "a stepped raw opacity was non-finite"
    );
    opac
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

    let batch = make_batch(test_camera());

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

/// **The `--depth-source` dispatch must be observable in `step()`'s output.**
///
/// # The failure mode this guards
///
/// Every other WS-A test drives `plane_features` → `render_splat_features` →
/// `plane_depth_from_features` DIRECTLY. None of them touches the conjunct that
/// decides whether `step()` actually takes that path:
///
/// ```text
/// use_plane_depth = plane_selected && (use_depth || use_dn) && is_pinhole
/// ```
///
/// Invert any one of those three and `--depth-source plane-aux` silently trains
/// exactly like `center`. Nothing announces it: the plane math is still correct
/// and still unit-tested, the loss curve looks normal, the run completes. The
/// damage lands in the ablation, where **arm 4 silently degenerates into arm 0**
/// and the recorded conclusion becomes "plane-aux ≈ baseline" for the wrong
/// reason — the same shape as the verifier's top-ranked hazard for approach B
/// (B degenerating into A), one layer up.
///
/// The centre-vs-plane residual diagnostic only executes INSIDE the
/// `use_plane_depth` branch, so it is evidence the path is live — but it is a
/// log line, not a gate. This is the gate.
///
/// # Both directions
///
/// Asserting only "the losses differ" pins the positive half. A test suite that
/// stopped there would accept an implementation that took the plane path
/// unconditionally, ignoring `is_pinhole` — which is worse than not dispatching
/// at all, because the ray-plane grid assumes a pinhole unprojection and would
/// supervise fisheye views with wrong math. So the fisheye arm asserts the
/// complementary thing: on a non-pinhole camera, `plane-aux` must be **bit-equal**
/// to `center`, because it is required to warn and fall back.
#[tokio::test]
async fn plane_aux_dispatch_is_live_in_step() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

    // --- Pinhole: the plane path must engage and change the number. ---
    let center = step0_loss_bits(DepthSource::Center, test_camera(), &device).await;
    let plane = step0_loss_bits(DepthSource::PlaneAux, test_camera(), &device).await;
    println!("DISPATCH pinhole  center=0x{center:08x} plane-aux=0x{plane:08x}");
    assert_ne!(
        plane, center,
        "--depth-source plane-aux produced a bit-identical step loss to center on \
         a PINHOLE camera with a depth loss active: the plane path is not being \
         taken, and ablation arm 4 would silently be a rerun of arm 0"
    );

    // Guard the guard: `center` must be reproducible, or `assert_ne!` above
    // would pass on run-to-run noise rather than on the dispatch.
    let center_again = step0_loss_bits(DepthSource::Center, test_camera(), &device).await;
    assert_eq!(
        center, center_again,
        "the center step-0 loss is not reproducible, so the inequality above \
         cannot be attributed to the depth source"
    );

    // --- Fisheye: the plane path must NOT engage, and must fall back exactly. ---
    let fish_center = step0_loss_bits(DepthSource::Center, fisheye_camera(), &device).await;
    let fish_plane = step0_loss_bits(DepthSource::PlaneAux, fisheye_camera(), &device).await;
    println!("DISPATCH fisheye  center=0x{fish_center:08x} plane-aux=0x{fish_plane:08x}");
    assert_eq!(
        fish_plane, fish_center,
        "--depth-source plane-aux changed the step loss on a NON-PINHOLE camera; \
         it is required to warn and fall back to centre depth there, because the \
         ray-plane grid assumes a pinhole unprojection"
    );

    // And the fisheye arm must not have passed by both sides being degenerate.
    assert_ne!(
        fish_center, center,
        "the fisheye and pinhole cameras produced identical losses, so the \
         fallback assertion above proves nothing"
    );
}

/// **The `--depth-source plane-fused` dispatch must be observable in `step()`'s
/// output — and observable in the WEIGHTS, not just the loss.**
///
/// The sibling of `plane_aux_dispatch_is_live_in_step`, for approach B. It needs
/// a different shape of evidence, because A and B are *designed* to agree in
/// their forward values: both composite the same on-tape `plane_features` and
/// intersect the same ray, so a forward-only test cannot tell "B is wired" from
/// "B silently ran A". The plan's §4.5 table says where they part company —
/// **only in the backward** — so the discriminating assertion has to be taken
/// after an optimizer step.
///
/// Three pins, in increasing strength:
///
/// 1. `plane-fused` ≠ `center` at step 0. The dispatch selects plane math at
///    all. Guarded by the same reproducibility check the aux pin uses, so the
///    inequality cannot be satisfied by run-to-run noise.
/// 2. `plane-fused` ≈ `plane-aux` at step 0, to a tolerance rather than
///    bitwise. This pins that fused renders THE PLANE FORWARD, not something
///    else: a kernel that composited garbage into the plane lanes would still
///    pass pin 1. It is deliberately not `assert_eq!` — the main rasterizer and
///    the feature rasterizer are different kernels reducing in different orders,
///    and demanding bit-equality of them would be demanding the wrong thing.
/// 3. **The one that matters.** One optimizer step under each, from the same
///    fixed batch and the same fixed splats, must leave DIFFERENT raw
///    opacities. Approach A routes geometry gradients exclusively through the
///    feature VALUES, so plane error structurally cannot reach opacity; approach
///    B folds the plane channels into the alpha VJP, so it can and must. If the
///    fused kernel had dropped that term — the verifier's top-ranked
///    degeneration, and one that is INVISIBLE to forward parity — pins 1 and 2
///    would both still pass and only this one would fail.
#[tokio::test]
async fn plane_fused_dispatch_is_live_in_step() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

    // --- Pin 1: fused engages the plane path at all. ---
    //
    // Stated as a MARGIN against the measured run-to-run jitter rather than as
    // `assert_ne!` plus an exact-reproducibility guard. Measured on this Mac
    // 2026-08-19: the step-0 `center` loss is reproducible in *most* processes
    // but flips its last bit (0x3ec517d1 <-> 0x3ec517d2) in roughly one run in
    // eight once several measurements are taken in one process — the same
    // atomic-accumulation nondeterminism the module comment describes for steps
    // 1+, which reaches step 0 too. An equality guard on that is a coin flip; a
    // margin is both immune to it and a strictly stronger claim, since it
    // demands the dispatch move the loss by far more than the noise floor.
    let center = step0_loss(DepthSource::Center, test_camera(), &device).await;
    let center_again = step0_loss(DepthSource::Center, test_camera(), &device).await;
    let fused = step0_loss(DepthSource::PlaneFused, test_camera(), &device).await;
    let aux = step0_loss(DepthSource::PlaneAux, test_camera(), &device).await;
    println!(
        "FUSED_DISPATCH step0 center=0x{:08x} ({center})  plane-aux=0x{:08x} ({aux})  \
         plane-fused=0x{:08x} ({fused})",
        center.to_bits(),
        aux.to_bits(),
        fused.to_bits()
    );

    // Noise floor: the observed jitter, but never below one ULP at this
    // magnitude, so a lucky pair of identical reads cannot make the margin
    // trivially satisfiable.
    let jitter = (center - center_again)
        .abs()
        .max(f32::EPSILON * center.abs());
    let separation = (fused - center).abs();
    println!(
        "FUSED_DISPATCH center jitter={jitter:e}  |fused-center|={separation:e}  \
         ratio={:e}",
        separation / jitter
    );
    assert!(
        separation > 1.0e3 * jitter,
        "--depth-source plane-fused moved the step-0 loss by only {separation:e} \
         against a center-path noise floor of {jitter:e} on a PINHOLE camera with \
         a depth loss active. The plane path is not being taken, and ablation arm 5 \
         would silently be a rerun of arm 0."
    );

    // --- Pin 2: fused renders the PLANE forward, agreeing with aux. ---
    let denom = aux.abs().max(1e-6);
    let rel = (fused - aux).abs() / denom;
    println!("FUSED_DISPATCH forward-parity |fused-aux|/|aux| = {rel:e}");
    assert!(
        rel < 1e-3,
        "plane-fused and plane-aux disagree on the step-0 loss by {rel:e} relative \
         (aux={aux}, fused={fused}). They composite the SAME plane features and \
         intersect the SAME ray, so they must agree to compositing noise. A larger \
         disagreement is a bug in one of the two kernels, NOT a tolerance to widen."
    );

    // --- Pin 3: the backward contract, read off the weights. ---
    let aux_opac = step0_raw_opacities(DepthSource::PlaneAux, test_camera(), &device).await;
    let fused_opac = step0_raw_opacities(DepthSource::PlaneFused, test_camera(), &device).await;
    assert_eq!(
        aux_opac.len(),
        fused_opac.len(),
        "the two paths stepped different splat counts"
    );

    // Reproducibility first: without this, "the opacities differ" could be
    // nondeterminism in the rasterize backward's atomic accumulation rather than
    // the alpha VJP term under test.
    let aux_opac_again = step0_raw_opacities(DepthSource::PlaneAux, test_camera(), &device).await;
    assert_eq!(
        aux_opac, aux_opac_again,
        "one plane-aux step is not reproducible, so the divergence below cannot be \
         attributed to the fused kernel's alpha VJP"
    );

    let diffs = aux_opac
        .iter()
        .zip(&fused_opac)
        .filter(|(a, f)| a.to_bits() != f.to_bits())
        .count();
    let worst = aux_opac
        .iter()
        .zip(&fused_opac)
        .map(|(a, f)| (a - f).abs())
        .fold(0.0f32, f32::max);
    println!(
        "FUSED_DISPATCH raw-opacity divergence: {diffs}/{} splats differ, worst |Δ| = {worst:e}",
        aux_opac.len()
    );
    assert!(
        diffs > 0,
        "one optimizer step under plane-fused left raw opacities BIT-IDENTICAL to \
         plane-aux across all {} splats. Approach B's whole reason to exist is that \
         plane error reaches the blending weights (plan section 4.5, row 3); approach \
         A structurally cannot express that. Identical opacities mean the plane \
         channels' alpha-VJP term is not being accumulated, i.e. B has degenerated \
         into A with an extra render mode — which forward parity cannot see.",
        aux_opac.len()
    );
}
