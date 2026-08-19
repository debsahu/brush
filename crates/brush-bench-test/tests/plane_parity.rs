//! PGSR plane-render A-vs-B parity and the approach-B backward contract.
//!
//! Covers tests 10 and the section-4.5 row-3 contract pin of
//! `docs/superpowers/plans/2026-08-19-brush-pgsr-plane-render.md`.
//!
//! PGSR: Chen et al. 2024, arXiv:2406.06521 (plane parameterization and
//! ray-plane surface depth).
//!
//! Both approaches composite the SAME per-splat `plane_features()` values in the
//! SAME depth-sorted order with the SAME alpha weights, so their FORWARD outputs
//! must agree to f32 rounding. Their BACKWARDS deliberately do not: A routes
//! geometry gradients through feature values only, B additionally folds the
//! plane channels into the alpha VJP.

#![allow(clippy::missing_assert_message)]

use brush_render::bwd::{render_splat_features, render_splats_with_pass_and_plane_aux};
use brush_render::gaussian_splats::{
    RasterPass, RasterizationMode, Rasterizer, SplatRenderMode, Splats,
};
use brush_render::{camera::Camera, kernels::camera_model::CameraModel};
use brush_train::train::plane_features;
use burn::tensor::{Tensor, s};
use glam::Vec3;

/// Parity must be measured on the HARD cutoff: the feature rasterizer has no
/// smooth-cutoff variant, so `BackwardSmoothCutoff` would compare two different
/// compositing functions and the disagreement would be about the cutoff, not
/// about the plane lanes.
const PASS: RasterPass = RasterPass::Backward;

/// Elementwise tolerance for A-vs-B agreement.
///
/// Both sides are f32 alpha compositing over the same splats in the same order.
/// The ONLY licensed source of disagreement is fused-multiply-add / reassociation
/// in the Metal backend, which is order-1e-7 relative here.
///
/// **A larger disagreement is a bug, never a tolerance to widen.** If this fails,
/// the two paths are compositing different things — a lane/stride slip, a
/// different visibility order, or a different cutoff — and the fix is in the
/// kernel. If it fails ONLY under `--features native-msl`, that is a finding
/// about the MSL preset kernels and should be escalated as such.
const PARITY_TOL: f32 = 1e-5;

const IMG: glam::UVec2 = glam::uvec2(48, 40);

// Channel indices into the `RgbaDepthPlane` image, derived from the same
// `const fn` the kernel uses rather than restated as literals.
use brush_render::kernels::helpers::{
    PLANE_AUX_LANES_USIZE, plane_channel_offset, raster_out_channels,
};
/// Coverage alpha: the last rgba channel, and the fifth input
/// `plane_depth_from_features` expects.
const ALPHA_CH: usize = raster_out_channels(false, false) as usize - 1;
/// Alpha-composited camera-z of the splat centres.
const DEPTH_CH: usize = raster_out_channels(false, false) as usize;
/// The four PGSR plane channels, `PLANE_LO..PLANE_HI`.
const PLANE_LO: usize = plane_channel_offset(true) as usize;
const PLANE_HI: usize = PLANE_LO + PLANE_AUX_LANES_USIZE;
const PLANE_CHANS: usize = raster_out_channels(true, true) as usize;

fn test_camera() -> Camera {
    Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

/// Six splats spread in depth, comfortably inside the frame, opacity well above
/// the 1/255 cutoff.
///
/// **Log-scales are well separated per splat** (~0.6 apart). `plane_features`
/// picks the thin axis with a DETACHED `argmin`, so two near-equal scales put
/// the scene on a discontinuity: an arbitrarily small perturbation swaps the
/// chosen axis and flips the plane normal. That breaks finite differences and
/// makes A-vs-B parity depend on two independent argmin evaluations agreeing.
fn test_splats(device: &burn::tensor::Device) -> Splats {
    let means = vec![
        0.20, -0.10, 0.00, //
        -0.30, 0.40, 0.20, //
        0.10, 0.30, -0.30, //
        -0.20, -0.20, 0.10, //
        0.35, 0.05, 0.40, //
        -0.05, -0.35, -0.15, //
    ];
    let rots = vec![
        0.90, 0.10, 0.05, 0.03, //
        0.70, 0.20, 0.30, 0.10, //
        0.50, 0.40, 0.30, 0.20, //
        0.80, 0.10, 0.10, 0.20, //
        0.60, -0.30, 0.15, 0.25, //
        0.75, 0.05, -0.25, 0.15, //
    ];
    let log_scales = vec![
        -1.0, -1.6, -2.2, //
        -1.7, -1.1, -2.3, //
        -2.4, -1.2, -1.8, //
        -1.1, -2.5, -1.7, //
        -1.9, -2.6, -1.3, //
        -2.7, -1.4, -2.0, //
    ];
    let sh_dc = vec![
        0.45, 0.55, 0.50, //
        0.60, 0.40, 0.30, //
        0.35, 0.50, 0.65, //
        0.50, 0.45, 0.55, //
        0.40, 0.60, 0.35, //
        0.55, 0.35, 0.45, //
    ];
    let raw_opac = vec![2.5, 2.0, 2.2, 2.4, 2.1, 2.3];
    Splats::from_raw(
        means,
        rots,
        log_scales,
        sh_dc,
        raw_opac,
        SplatRenderMode::Default,
        device,
    )
}

async fn read_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .expect("vec")
}

fn assert_parity(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{label}[{i}] non-finite");
        let err = (a - b).abs();
        if err > worst {
            worst = err;
            worst_at = i;
        }
    }
    assert!(
        worst <= PARITY_TOL,
        "{label}: worst |Δ| = {worst:e} at index {worst_at} (A={}, B={}), tolerance {PARITY_TOL:e}. \
         This is a BUG, not a tolerance to widen — see the PARITY_TOL doc comment.",
        expected[worst_at],
        actual[worst_at],
    );
}

/// Test 10: the main kernel's four composited plane channels must match the
/// feature pass's, and so must the depth derived from them.
#[tokio::test]
async fn plane_forward_parity_a_vs_b() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = test_camera();
    let splats = test_splats(&device);
    let transforms = splats.transforms.val();
    let feats = plane_features(transforms.clone(), &camera);

    // Approach A: the [H, W, 5] feature pass (4 plane channels + alpha).
    let a_img = render_splat_features(
        transforms,
        splats.raw_opacities.val(),
        feats.clone(),
        &camera,
        IMG,
        SplatRenderMode::Default,
    )
    .await;

    // Approach B: the main rasterizer, [H, W, 9] = rgba + centre depth + plane.
    let b = render_splats_with_pass_and_plane_aux(
        splats,
        &camera,
        IMG,
        Vec3::ZERO,
        PASS,
        Rasterizer::Legacy,
        RasterizationMode::RgbaDepthPlane,
        Some(feats),
    )
    .await;
    let b_img = b.img;

    let (h, w) = (IMG.y as usize, IMG.x as usize);
    assert_eq!(a_img.dims(), [h, w, 5]);
    assert_eq!(b_img.dims(), [h, w, PLANE_CHANS]);

    // Channel layout contract: plane lanes follow rgba + centre depth, and the
    // alpha the plane reduction needs is the ordinary rgba alpha at channel 3.
    let b_feat = Tensor::cat(
        vec![
            b_img.clone().slice(s![.., .., PLANE_LO..PLANE_HI]),
            b_img.clone().slice(s![.., .., ALPHA_CH..ALPHA_CH + 1]),
        ],
        2,
    );

    let a_plane =
        read_vec(
            a_img
                .clone()
                .slice(s![.., .., 0..raster_out_channels(false, false) as usize]),
        )
        .await;
    let b_plane = read_vec(b_img.slice(s![.., .., PLANE_LO..PLANE_HI])).await;
    assert_parity("composited plane channels", &b_plane, &a_plane);

    let a_alpha = read_vec(a_img.clone().slice(s![.., .., DEPTH_CH..DEPTH_CH + 1])).await;
    let b_alpha = read_vec(b_feat.clone().slice(s![.., .., DEPTH_CH..DEPTH_CH + 1])).await;
    assert_parity("coverage alpha", &b_alpha, &a_alpha);
    assert!(
        a_alpha.iter().any(|&v| v > 0.5),
        "degenerate scene: nothing is covered, parity would be vacuous"
    );

    // Derived ray-plane depth, compared on the intersection of the two valid
    // masks (the masks themselves must agree, since the inputs do).
    let focal = camera.focal(IMG);
    let center = camera.center(IMG);
    let thresholds = (0.5f32, 0.05f32, 0.05f32, 100.0f32);
    let (a_depth, _, a_valid) = brush_loss::plane_depth_from_features(
        a_img,
        focal.x,
        focal.y,
        center.x,
        center.y,
        thresholds.0,
        thresholds.1,
        thresholds.2,
        thresholds.3,
    );
    let (b_depth, _, b_valid) = brush_loss::plane_depth_from_features(
        b_feat,
        focal.x,
        focal.y,
        center.x,
        center.y,
        thresholds.0,
        thresholds.1,
        thresholds.2,
        thresholds.3,
    );

    let a_valid = read_vec(a_valid).await;
    let b_valid = read_vec(b_valid).await;
    assert_eq!(a_valid, b_valid, "valid masks must agree exactly");
    let covered = a_valid.iter().filter(|&&v| v > 0.5).count();
    assert!(
        covered > 100,
        "only {covered} valid pixels — scene is too sparse for a meaningful depth parity check"
    );

    let a_depth = read_vec(a_depth).await;
    let b_depth = read_vec(b_depth).await;
    let (a_masked, b_masked): (Vec<f32>, Vec<f32>) = a_valid
        .iter()
        .zip(a_depth.iter().zip(b_depth.iter()))
        .filter(|(v, _)| **v > 0.5)
        .map(|(_, (a, b))| (*a, *b))
        .unzip();
    assert_parity("derived plane depth", &b_masked, &a_masked);
}

/// Contract row 3 of plan section 4.5: `plane_fused_depth_reaches_opacity`.
///
/// **The opacity gradient here is nonzero BY DESIGN.** Approach B folds the
/// plane channels' `(value − residual)·v_channel` term into the alpha VJP, so
/// plane (geometry) error is attributable to the blending weights — opacity,
/// conic and screen-space position — exactly as RGB error is. That is the whole
/// difference between B and the feature-pass approach A, and it is what the
/// ablation (arm 4 vs arm 5) adjudicates.
///
/// This is DELIBERATELY the opposite of `depth_loss_does_not_touch_opacity`,
/// which pins that the CENTRE-depth channel's analogous term stays dropped.
/// A reviewer who "fixes" this test to assert a zero opacity gradient has
/// misread the design and has silently converted B into A.
///
/// **This assertion is nonzero-only and is not sufficient on its own** — if the
/// plane alpha term were never written, the gradient would still be nonzero
/// through the value path and this test would pass. The VALUE checks that close
/// that gap live in brush-render:
/// `tests::vjp_golden::raster_backward_matches_the_independent_vjp_reference`
/// (against the float64 reference in `analyze/vjp_reference/`) and
/// `tests::raster_bwd_twin::raster_backward_matches_the_autodiff_twin` (against
/// a Burn-autodiff reimplementation of the same contract). This test is the
/// end-to-end sibling: it is the only one of the three that runs through the
/// real projection, tiling and autodiff bridge.
#[tokio::test]
async fn plane_fused_depth_reaches_opacity() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = test_camera();
    let splats = test_splats(&device);
    let feats = plane_features(splats.transforms.val(), &camera);

    let out = render_splats_with_pass_and_plane_aux(
        splats.clone(),
        &camera,
        IMG,
        Vec3::ZERO,
        PASS,
        Rasterizer::Legacy,
        RasterizationMode::RgbaDepthPlane,
        Some(feats),
    )
    .await;

    // Loss on the PLANE channels only — no RGB term — so any opacity gradient
    // observed can only have come through the plane lanes.
    let loss = out.img.slice(s![.., .., PLANE_LO..PLANE_HI]).mean();
    let grads = loss.backward();

    let opac = read_vec(splats.raw_opacities.grad(&grads).expect("opacity grad")).await;
    let transforms = read_vec(splats.transforms.grad(&grads).expect("transforms grad")).await;

    assert!(opac.iter().all(|v| v.is_finite()));
    assert!(transforms.iter().all(|v| v.is_finite()));
    let max_opac = opac.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        max_opac > 1e-6,
        "approach B must let plane error reach opacity; max |d loss / d raw_opacity| = {max_opac:e}"
    );
    let max_transform = transforms.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        max_transform > 1e-6,
        "plane error must also reach geometry; max |d loss / d transforms| = {max_transform:e}"
    );
}

/// Contract row 1 must SURVIVE the plane extension: the CENTRE-depth channel's
/// alpha term stays dropped even in plane mode.
///
/// The two `dot` contributions sit two lines apart in the same function, and a
/// plane-lane edit that generalised the accumulator over "all aux channels"
/// would silently resurrect the depth term. The term is linear in `v_out`, so a
/// cotangent that is one-hot on the centre-depth channel must give **bit-zero**
/// opacity gradient — not "small". The plane-mode sibling of brush-train's
/// `depth_loss_does_not_touch_opacity`, which only covers `RgbaAndDepth`.
#[tokio::test]
async fn plane_mode_keeps_centre_depth_detached_from_opacity() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = test_camera();
    let splats = test_splats(&device);
    let feats = plane_features(splats.transforms.val(), &camera);

    let out = render_splats_with_pass_and_plane_aux(
        splats.clone(),
        &camera,
        IMG,
        Vec3::ZERO,
        PASS,
        Rasterizer::Legacy,
        RasterizationMode::RgbaDepthPlane,
        Some(feats),
    )
    .await;

    // One-hot on channel 4 (centre depth) only.
    let grads = out
        .img
        .slice(s![.., .., DEPTH_CH..DEPTH_CH + 1])
        .mean()
        .backward();
    let opac = read_vec(splats.raw_opacities.grad(&grads).expect("opacity grad")).await;
    let transforms = read_vec(splats.transforms.grad(&grads).expect("transforms grad")).await;

    assert!(
        opac.iter().all(|v| *v == 0.0),
        "centre-depth error must not reach opacity even in plane mode; got {opac:?}"
    );
    // ...but it must still reach positions through the depth VALUE lane, or the
    // test would pass on a render that produced no depth gradient at all.
    assert!(
        transforms.iter().any(|v| v.abs() > 1e-6),
        "centre-depth error must still move gaussian positions"
    );
}

/// Default-inertness: with the plane mode unselected, the rendered image and
/// every model gradient are unchanged by the existence of the plane path.
///
/// Checked against BOTH pre-existing modes: `Rgba` (4 channels) and
/// `RgbaAndDepth` (5). The second matters because the plane lanes are appended
/// after the depth lane, so a stale channel stride would land on depth first.
#[tokio::test]
async fn plane_mode_unselected_is_inert() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = test_camera();

    let rgba_splats = test_splats(&device);
    let rgba = render_splats_with_pass_and_plane_aux(
        rgba_splats.clone(),
        &camera,
        IMG,
        Vec3::ZERO,
        PASS,
        Rasterizer::Legacy,
        RasterizationMode::Rgba,
        None,
    )
    .await;
    let rgba_img = read_vec(rgba.img.clone()).await;
    let rgba_grads = rgba.img.mean().backward();
    let rgba_transforms = read_vec(rgba_splats.transforms.grad(&rgba_grads).unwrap()).await;
    let rgba_opac = read_vec(rgba_splats.raw_opacities.grad(&rgba_grads).unwrap()).await;

    // Same render through the plane-capable code path; the rgba channels and
    // their gradients must be bit-for-bit unaffected by the extra lanes.
    let plane_splats = test_splats(&device);
    let feats = plane_features(plane_splats.transforms.val(), &camera);
    let plane = render_splats_with_pass_and_plane_aux(
        plane_splats.clone(),
        &camera,
        IMG,
        Vec3::ZERO,
        PASS,
        Rasterizer::Legacy,
        RasterizationMode::RgbaDepthPlane,
        Some(feats),
    )
    .await;
    let plane_rgba = read_vec(plane.img.clone().slice(s![
        ..,
        ..,
        0..raster_out_channels(false, false) as usize
    ]))
    .await;
    let plane_grads = plane
        .img
        .slice(s![.., .., 0..raster_out_channels(false, false) as usize])
        .mean()
        .backward();
    let plane_transforms = read_vec(plane_splats.transforms.grad(&plane_grads).unwrap()).await;
    let plane_opac = read_vec(plane_splats.raw_opacities.grad(&plane_grads).unwrap()).await;

    assert_parity("rgba forward", &plane_rgba, &rgba_img);
    assert_parity("rgba transforms grad", &plane_transforms, &rgba_transforms);
    assert_parity("rgba opacity grad", &plane_opac, &rgba_opac);

    // The 5-channel mode: rgba + centre depth must be untouched by the plane
    // extension, forward and backward.
    let depth_splats = test_splats(&device);
    let depth = render_splats_with_pass_and_plane_aux(
        depth_splats.clone(),
        &camera,
        IMG,
        Vec3::ZERO,
        PASS,
        Rasterizer::Legacy,
        RasterizationMode::RgbaAndDepth,
        None,
    )
    .await;
    let depth_img =
        read_vec(
            depth
                .img
                .clone()
                .slice(s![.., .., 0..raster_out_channels(true, false) as usize]),
        )
        .await;
    let depth_grads = depth
        .img
        .slice(s![.., .., 0..raster_out_channels(true, false) as usize])
        .mean()
        .backward();
    let depth_transforms = read_vec(depth_splats.transforms.grad(&depth_grads).unwrap()).await;

    let plane_depth_splats = test_splats(&device);
    let plane_depth_feats = plane_features(plane_depth_splats.transforms.val(), &camera);
    let plane_depth = render_splats_with_pass_and_plane_aux(
        plane_depth_splats.clone(),
        &camera,
        IMG,
        Vec3::ZERO,
        PASS,
        Rasterizer::Legacy,
        RasterizationMode::RgbaDepthPlane,
        Some(plane_depth_feats),
    )
    .await;
    let plane_depth_img = read_vec(plane_depth.img.clone().slice(s![
        ..,
        ..,
        0..raster_out_channels(true, false) as usize
    ]))
    .await;
    let plane_depth_grads = plane_depth
        .img
        .slice(s![.., .., 0..raster_out_channels(true, false) as usize])
        .mean()
        .backward();
    let plane_depth_transforms = read_vec(
        plane_depth_splats
            .transforms
            .grad(&plane_depth_grads)
            .unwrap(),
    )
    .await;

    assert_parity("rgba+depth forward", &plane_depth_img, &depth_img);
    assert_parity(
        "rgba+depth transforms grad",
        &plane_depth_transforms,
        &depth_transforms,
    );
}

/// The 16x8 "candidate" training tile layout (native-MSL `FINE_RASTER_TILES`)
/// must produce the same plane channels and the same gradients as the 16x16
/// legacy layout.
///
/// This is the tile-geometry half of the stride blast radius: the forward
/// composites the plane lanes inside a tile batch and the backward replays them
/// through a `pix_state` whose stride grew by four, both parameterised on
/// `tile_width`/`tile_height`. A layout-dependent slip would show up here and
/// nowhere else, because the production training rasterizer is exactly this
/// selector whenever the native-MSL preset is on.
#[tokio::test]
async fn plane_candidate_selector_matches_legacy() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = test_camera();
    // Deliberately not a multiple of either tile size, so the two grids differ.
    let img_size = glam::uvec2(35, 29);
    let background = Vec3::new(0.11, 0.07, 0.19);

    let mut results = Vec::new();
    for rasterizer in [Rasterizer::Legacy, Rasterizer::Candidate] {
        let splats = test_splats(&device);
        let feats = plane_features(splats.transforms.val(), &camera);
        let out = render_splats_with_pass_and_plane_aux(
            splats.clone(),
            &camera,
            img_size,
            background,
            PASS,
            rasterizer,
            RasterizationMode::RgbaDepthPlane,
            Some(feats),
        )
        .await;
        let img = read_vec(out.img.clone()).await;
        let grads = out
            .img
            .slice(s![.., .., PLANE_LO..PLANE_HI])
            .mean()
            .backward();
        let transforms = read_vec(splats.transforms.grad(&grads).expect("transforms grad")).await;
        let opac = read_vec(splats.raw_opacities.grad(&grads).expect("opacity grad")).await;
        results.push((img, transforms, opac));
    }

    let (legacy, candidate) = (&results[0], &results[1]);
    assert_parity("plane image", &candidate.0, &legacy.0);
    assert_parity("plane transforms grad", &candidate.1, &legacy.1);
    assert_parity("plane opacity grad", &candidate.2, &legacy.2);
    assert!(
        candidate.2.iter().any(|v| v.abs() > 1e-6),
        "candidate-layout opacity gradient is all zero — the comparison would be vacuous"
    );
}
