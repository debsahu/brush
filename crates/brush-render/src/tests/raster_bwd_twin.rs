//! Autodiff twin for the raster backward — the strategy gsplat uses to validate
//! its own hand-written CUDA backward against a pure-PyTorch reimplementation
//! (`gsplat/cuda/_torch_impl.py`).
//!
//! Here the twin is written in plain Burn tensor ops, so **Burn's autodiff
//! generates its backward** and nothing about it shares code with
//! `rasterize_backwards.rs`. The twin reproduces the kernel's contract exactly,
//! including the two deliberate departures from the true derivative:
//!
//! * the CENTRE-DEPTH channel is composited with a **detached** blending weight,
//!   which is precisely what the kernel's dropped `dot_rgb` depth term achieves;
//! * the RGB and the four PGSR plane channels are composited with the weight
//!   LIVE.
//!
//! That is what makes this test able to catch the *symmetric* failure pair that
//! the derivation's checklist (C6) calls the most expensive in the port —
//! depth's alpha term resurrected, or the plane terms never written — neither of
//! which changes any forward value, and both of which leave every gradient
//! finite and nonzero.
//!
//! Boundary handling follows the same discipline: rather than loosening the
//! tolerance to absorb the pipeline's genuine discontinuities (the `alpha`
//! saturation clamp, the `T <= 1e-4` early-out, the `sigma >= 0` / `w_cut > 0`
//! predicates), the scenes are built to stay off them and a pre-pass asserts so.
//! One splat is deliberately parked far below the cutoff, which gives the
//! gradient-sparsity assertion something real to check.

use super::raster_oracle::rasterize_reference_full;
use crate::{
    bwd::{
        ALPHA_LANE, CONIC_LANE, DEPTH_LANE, PLANE_GRAD_LANE_START, REFINE_LANE, RGB_LANE, XY_LANE,
        burn_glue::SplatBwdOps,
    },
    kernels::helpers::{
        ALPHA_CUTOFF_BAND, ALPHA_CUTOFF_MID, PLANE_AUX_LANES_USIZE, PROJECTED_LANES_USIZE,
    },
};
use brush_cube::{MainBackendBase, Runtime};
use burn::{
    backend::ops::FloatTensorOps,
    tensor::{DType, Tensor, TensorData, s},
};
use burn_wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use glam::{UVec2, Vec3};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const IMG_W: usize = 5;
const IMG_H: usize = 4;
const PIXELS: usize = IMG_W * IMG_H;
/// Full plane-mode channel count: rgba (4) + centre depth (1) + plane (4).
/// Derived from the shared `const fn` so a lane addition cannot leave this
/// twin silently comparing the wrong stride — the exact drift that broke
/// three call sites when the depth lane landed.
const CHANS: usize = crate::kernels::helpers::raster_out_channels(true, true) as usize;
/// Channels that are alpha-composited from a per-splat value: rgb (3), centre
/// depth (1), plane (4). The remaining output channel is the coverage alpha,
/// which is `1 - T` rather than a composited value.
const VALUE_CHANS: usize = CHANS - 1;
const BG: Vec3 = Vec3::new(0.13, 0.17, 0.29);

/// One splat's projected state, as the rasterizer sees it.
#[derive(Clone, Copy)]
struct TwinSplat {
    xy: [f32; 2],
    conic: [f32; 3],
    /// Projected opacity (`color_a`), already post-sigmoid.
    opacity: f32,
    rgb: [f32; 3],
    depth: f32,
    plane: [f32; PLANE_AUX_LANES_USIZE],
}

/// The three opaque splats' opacities are a parameter so the same geometry can
/// be run FAINT and OPAQUE. That is the discriminator for the suffix-buffer
/// read/write ordering bug (checklist C1): reading `pix_state` after updating it
/// instead of before mis-scales the alpha VJP by `alpha/(1 - alpha)`, which is
/// ~5% at `alpha = 0.05` — small enough for a loose tolerance to absorb — and
/// nearly an order of magnitude at `alpha = 0.9`. Agreement must hold at BOTH.
///
/// Front-to-back list. Deliberate properties, all asserted by
/// `assert_off_the_discontinuities`:
///
/// * splat 0-2 have alpha comfortably above the smoothstep band, so `w = 1`;
/// * splat 3 sits INSIDE the band on some pixels, exercising `w' != 0` (the
///   `(w + alpha*w')` chain the hard-cutoff production path never sees);
/// * splat 4 is parked far off-screen: below the cutoff on every pixel, so its
///   entire gradient row must be exactly zero on both sides;
/// * every alpha stays under the `0.999` clamp and the running transmittance
///   stays above the `1e-4` early-out, so no discontinuity is straddled;
/// * every rgb is strictly positive, so the forward's `max(v, 0)` is inert and
///   the kernel's pre-clamp sign gate cannot differ from the twin's.
fn twin_scene(opacities: [f32; 3]) -> Vec<TwinSplat> {
    vec![
        TwinSplat {
            xy: [2.1, 1.8],
            conic: [0.30, 0.05, 0.26],
            opacity: opacities[0],
            rgb: [0.80, 0.40, 0.25],
            depth: 3.0,
            plane: [0.10, -0.20, 0.97, 3.1],
        },
        TwinSplat {
            xy: [3.0, 2.4],
            conic: [0.22, -0.04, 0.31],
            opacity: opacities[1],
            rgb: [0.15, 0.70, 0.35],
            depth: 4.2,
            plane: [-0.30, 0.15, 0.94, 4.4],
        },
        TwinSplat {
            xy: [1.4, 2.9],
            conic: [0.27, 0.03, 0.20],
            opacity: opacities[2],
            rgb: [0.35, 0.25, 0.80],
            depth: 5.5,
            plane: [0.22, 0.35, 0.91, 5.6],
        },
        TwinSplat {
            // Weak and broad: lands inside the C^1 cutoff band over the frame.
            xy: [2.5, 2.0],
            conic: [0.02, 0.0, 0.02],
            opacity: 0.0041,
            rgb: [0.60, 0.55, 0.45],
            depth: 6.7,
            plane: [-0.12, -0.44, 0.89, 6.8],
        },
        TwinSplat {
            // Far outside the frame: contributes nowhere.
            xy: [80.0, 80.0],
            conic: [0.9, 0.0, 0.9],
            opacity: 0.9,
            rgb: [0.5, 0.5, 0.5],
            depth: 9.0,
            plane: [0.0, 0.0, 1.0, 9.0],
        },
    ]
}

fn smoothstep_cutoff(alpha: f32) -> f32 {
    let low = ALPHA_CUTOFF_MID - 0.5 * ALPHA_CUTOFF_BAND;
    let t = ((alpha - low) / ALPHA_CUTOFF_BAND).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Host pre-pass. Confirms the scene stays off every pipeline discontinuity and
/// returns, per splat, whether it contributes to any pixel at all.
fn assert_off_the_discontinuities(scene: &[TwinSplat]) -> Vec<bool> {
    let mut contributes = vec![false; scene.len()];
    let mut saw_band = false;
    for py in 0..IMG_H {
        for px in 0..IMG_W {
            let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
            let mut t_acc = 1.0f32;
            for (i, sp) in scene.iter().enumerate() {
                let dx = sp.xy[0] - cx;
                let dy = sp.xy[1] - cy;
                let sigma =
                    0.5 * (sp.conic[0] * dx * dx + sp.conic[2] * dy * dy) + sp.conic[1] * dx * dy;
                assert!(sigma >= 0.0, "conic must be PSD (splat {i})");
                let raw = sp.opacity * (-sigma).exp();
                assert!(
                    raw < 0.99,
                    "splat {i} alpha {raw} is near the 0.999 saturation clamp"
                );
                let w = smoothstep_cutoff(raw);
                // Off the *edges* of the smoothstep band: either fully out
                // (w == 0), fully in (w == 1), or strictly inside it.
                let low = ALPHA_CUTOFF_MID - 0.5 * ALPHA_CUTOFF_BAND;
                let high = ALPHA_CUTOFF_MID + 0.5 * ALPHA_CUTOFF_BAND;
                assert!(
                    (raw - low).abs() > 1.0e-5 && (raw - high).abs() > 1.0e-5,
                    "splat {i} alpha {raw} sits on a band edge"
                );
                if w > 0.0 && w < 1.0 {
                    saw_band = true;
                }
                if w > 0.0 {
                    contributes[i] = true;
                    t_acc *= 1.0 - raw * w;
                }
            }
            assert!(
                t_acc > 1.0e-3,
                "pixel ({px},{py}) transmittance {t_acc} is close to the 1e-4 early-out"
            );
        }
    }
    assert!(
        saw_band,
        "scene never enters the C^1 cutoff band, so `w'` is never exercised"
    );
    assert!(
        !contributes[scene.len() - 1],
        "the off-screen splat must contribute nowhere, or the sparsity check is vacuous"
    );
    contributes
}

/// Per-pixel cotangent, distinct in every slot so a transposed lane cannot
/// cancel out.
fn cotangent() -> Vec<f32> {
    (0..PIXELS * CHANS)
        .map(|i| {
            let t = i as f32;
            0.35 + 0.11 * (t * 0.7).sin() - 0.05 * (t % 7.0)
        })
        .collect()
}

fn flat_projected(scene: &[TwinSplat]) -> (Vec<f32>, Vec<f32>) {
    let mut projected = vec![0.0f32; scene.len() * PROJECTED_LANES_USIZE];
    let mut plane = vec![0.0f32; scene.len() * PLANE_AUX_LANES_USIZE];
    for (i, sp) in scene.iter().enumerate() {
        let b = i * PROJECTED_LANES_USIZE;
        projected[b] = sp.xy[0];
        projected[b + 1] = sp.xy[1];
        projected[b + 2] = sp.conic[0];
        projected[b + 3] = sp.conic[1];
        projected[b + 4] = sp.conic[2];
        projected[b + 5] = sp.opacity;
        projected[b + 6] = sp.rgb[0];
        projected[b + 7] = sp.rgb[1];
        projected[b + 8] = sp.rgb[2];
        projected[b + 9] = sp.depth;
        plane[i * PLANE_AUX_LANES_USIZE..(i + 1) * PLANE_AUX_LANES_USIZE]
            .copy_from_slice(&sp.plane);
    }
    (projected, plane)
}

/// Gradients the twin produces, laid out in the kernel's `v_combined` lane
/// order so the comparison is index-for-index.
struct TwinGrads {
    /// `[num_splats][COMPACT_GRAD_LANES]`, refine lane (9) left at zero.
    rows: Vec<Vec<f32>>,
    forward: Vec<f32>,
}

/// Front-to-back alpha compositing in plain Burn ops, with the kernel's
/// contract. Returns the leaf gradients arranged by `v_combined` lane.
async fn run_twin(scene: &[TwinSplat], v_out: &[f32]) -> TwinGrads {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let n = scene.len();

    // Leaves, one tensor per projected field.
    let leaf = |values: Vec<f32>| -> Tensor<1> {
        Tensor::<1>::from_floats(values.as_slice(), &device).require_grad()
    };
    let xy_x = leaf(scene.iter().map(|s| s.xy[0]).collect());
    let xy_y = leaf(scene.iter().map(|s| s.xy[1]).collect());
    let c00 = leaf(scene.iter().map(|s| s.conic[0]).collect());
    let c01 = leaf(scene.iter().map(|s| s.conic[1]).collect());
    let c11 = leaf(scene.iter().map(|s| s.conic[2]).collect());
    let opac = leaf(scene.iter().map(|s| s.opacity).collect());
    let chan: Vec<Tensor<1>> = (0..VALUE_CHANS)
        .map(|c| {
            leaf(
                scene
                    .iter()
                    .map(|s| match c {
                        0..=2 => s.rgb[c],
                        3 => s.depth,
                        _ => s.plane[c - 4],
                    })
                    .collect(),
            )
        })
        .collect();

    // Pixel centres, flattened row-major exactly like `pix_id`.
    let mut px = Vec::with_capacity(PIXELS);
    let mut py = Vec::with_capacity(PIXELS);
    for y in 0..IMG_H {
        for x in 0..IMG_W {
            px.push(x as f32 + 0.5);
            py.push(y as f32 + 0.5);
        }
    }
    let px = Tensor::<1>::from_floats(px.as_slice(), &device);
    let py = Tensor::<1>::from_floats(py.as_slice(), &device);

    let mut t_acc = Tensor::<1>::ones([PIXELS], &device);
    // Value accumulators: rgb (3) + depth (1) + plane (4). Alpha is 1 - T.
    let mut acc: Vec<Tensor<1>> = (0..VALUE_CHANS)
        .map(|_| Tensor::<1>::zeros([PIXELS], &device))
        .collect();

    let low = ALPHA_CUTOFF_MID - 0.5 * ALPHA_CUTOFF_BAND;
    for i in 0..n {
        let pick = |t: &Tensor<1>| t.clone().slice(s![i..i + 1]);
        let dx = pick(&xy_x) - px.clone();
        let dy = pick(&xy_y) - py.clone();
        let sigma = (pick(&c00) * dx.clone() * dx.clone() + pick(&c11) * dy.clone() * dy.clone())
            .mul_scalar(0.5)
            + pick(&c01) * dx * dy;
        // `min(0.999, ...)` is inert on this scene (asserted host-side), so it
        // is omitted rather than introducing a clamp whose subgradient at the
        // cap would differ from the kernel's explicit suppression.
        let alpha = pick(&opac) * sigma.neg().exp();
        let t = alpha.clone().sub_scalar(low).div_scalar(ALPHA_CUTOFF_BAND);
        let t = t.clamp(0.0, 1.0);
        let w_cut = t.clone() * t.clone() * (t.mul_scalar(-2.0).add_scalar(3.0));
        let alpha_eff = alpha * w_cut;
        let vis = alpha_eff.clone() * t_acc.clone();

        for (c, a) in acc.iter_mut().enumerate() {
            // Channel 3 is the CENTRE DEPTH: its blending weight is DETACHED,
            // which is exactly what dropping the depth term from `dot_rgb`
            // achieves. Every other channel keeps the weight live.
            let weight = if c == 3 {
                vis.clone().detach()
            } else {
                vis.clone()
            };
            *a = a.clone() + pick(&chan[c]) * weight;
        }
        t_acc = t_acc * alpha_eff.neg().add_scalar(1.0);
    }

    // Background rides on RGB only, and the output alpha is 1 - T.
    let bg = [BG.x, BG.y, BG.z];
    let mut out: Vec<Tensor<1>> = Vec::with_capacity(CHANS);
    for c in 0..3 {
        out.push(acc[c].clone() + t_acc.clone().mul_scalar(bg[c]));
    }
    out.push(t_acc.clone().neg().add_scalar(1.0));
    for a in acc.iter().skip(3) {
        out.push(a.clone());
    }

    // Loss linear in the output, so its gradient IS the VJP for `v_out`.
    let mut loss = Tensor::<1>::zeros([1], &device);
    let mut forward = vec![0.0f32; PIXELS * CHANS];
    for (c, o) in out.iter().enumerate() {
        let cot: Vec<f32> = (0..PIXELS).map(|p| v_out[p * CHANS + c]).collect();
        let cot = Tensor::<1>::from_floats(cot.as_slice(), &device);
        loss = loss + (o.clone() * cot).sum();
        let values = o
            .clone()
            .into_data_async()
            .await
            .expect("forward readback")
            .into_vec::<f32>()
            .expect("vec");
        for (p, v) in values.into_iter().enumerate() {
            forward[p * CHANS + c] = v;
        }
    }
    let grads = loss.backward();

    let read = |t: &Tensor<1>| -> Option<Tensor<1>> { t.grad(&grads) };
    let lanes = crate::bwd::COMPACT_GRAD_LANES as usize;
    let plane_start = PLANE_GRAD_LANE_START;
    let mut rows = vec![vec![0.0f32; lanes]; n];

    let mut place = |lane: usize, values: Vec<f32>| {
        for (i, v) in values.into_iter().enumerate() {
            rows[i][lane] = v;
        }
    };
    async fn to_vec(t: Option<Tensor<1>>) -> Vec<f32> {
        t.expect("leaf gradient")
            .into_data_async()
            .await
            .expect("grad readback")
            .into_vec::<f32>()
            .expect("vec")
    }
    place(XY_LANE, to_vec(read(&xy_x)).await);
    place(XY_LANE + 1, to_vec(read(&xy_y)).await);
    place(CONIC_LANE, to_vec(read(&c00)).await);
    place(CONIC_LANE + 1, to_vec(read(&c01)).await);
    place(CONIC_LANE + 2, to_vec(read(&c11)).await);
    place(ALPHA_LANE, to_vec(read(&opac)).await);
    for c in 0..VALUE_CHANS {
        let lane = match c {
            0..=2 => RGB_LANE + c,
            3 => DEPTH_LANE,
            _ => plane_start + (c - 4),
        };
        place(lane, to_vec(read(&chan[c])).await);
    }

    TwinGrads { rows, forward }
}

fn cube_f32<const D: usize>(
    device: &WgpuDevice,
    shape: [usize; D],
    data: &[f32],
) -> CubeTensor<WgpuRuntime> {
    let client = WgpuRuntime::client(device);
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        burn::tensor::Shape::new(shape),
        handle,
        DType::F32,
    )
}

fn cube_u32<const D: usize>(
    device: &WgpuDevice,
    shape: [usize; D],
    data: &[u32],
) -> CubeTensor<WgpuRuntime> {
    let client = WgpuRuntime::client(device);
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        burn::tensor::Shape::new(shape),
        handle,
        DType::U32,
    )
}

async fn read_f32(tensor: CubeTensor<WgpuRuntime>) -> Vec<f32> {
    let data: TensorData = MainBackendBase::float_into_data(tensor)
        .await
        .expect("readback");
    data.as_slice::<f32>().expect("f32 tensor").to_vec()
}

/// Kernel backward vs the autodiff twin, lane for lane, on a faint scene and an
/// opaque one. See `twin_scene` for why both.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn raster_backward_matches_the_autodiff_twin() {
    for opacities in [[0.05, 0.04, 0.06], [0.90, 0.85, 0.80]] {
        check_twin_agreement(opacities).await;
    }
}

async fn check_twin_agreement(opacities: [f32; 3]) {
    let scene = twin_scene(opacities);
    let contributes = assert_off_the_discontinuities(&scene);
    let n = scene.len();
    let v_out = cotangent();

    let twin = run_twin(&scene, &v_out).await;

    // Independent CPU forward, so the image the kernel's backward replays from
    // is not produced by either the kernel or the twin.
    let (projected, plane) = flat_projected(&scene);
    let oracle = rasterize_reference_full(
        &projected,
        Some(&plane),
        true,
        UVec2::new(IMG_W as u32, IMG_H as u32),
        BG,
        true,
    );
    assert_eq!(oracle.len(), PIXELS * CHANS);
    for (i, (&a, &b)) in twin.forward.iter().zip(&oracle).enumerate() {
        assert!(
            (a - b).abs() <= 2.0e-6 + 2.0e-5 * b.abs(),
            "twin forward disagrees with the CPU oracle at {i}: {a:e} vs {b:e}"
        );
    }

    let device = brush_cube::test_helpers::test_device().await;
    let ids: Vec<u32> = (0..n as u32).collect();
    let grads = <MainBackendBase as SplatBwdOps>::rasterize_bwd(
        cube_f32(&device, [IMG_H, IMG_W, CHANS], &oracle),
        cube_f32(&device, [n, PROJECTED_LANES_USIZE], &projected),
        cube_u32(&device, [n], &ids),
        cube_u32(&device, [1, 1, 2], &[0, n as u32]),
        Some(cube_f32(&device, [n, PLANE_AUX_LANES_USIZE], &plane)),
        cube_u32(&device, [n], &ids),
        BG,
        UVec2::new(IMG_W as u32, IMG_H as u32),
        cube_f32(&device, [IMG_H, IMG_W, CHANS], &v_out),
        true,
        true,
        true,
    );

    let lanes = crate::bwd::COMPACT_GRAD_LANES as usize;
    let combined = read_f32(grads.v_combined).await;
    assert_eq!(combined.len(), n * lanes);

    // The refine-weight lane is a training statistic with no twin counterpart.
    let lane_name = |lane: usize| -> String {
        match lane {
            l if l < CONIC_LANE => format!("v_means2d[{}]", l - XY_LANE),
            l if l < RGB_LANE => format!("v_conic[{}]", l - CONIC_LANE),
            l if l < ALPHA_LANE => format!("v_rgb[{}]", l - RGB_LANE),
            ALPHA_LANE => "v_opacity".to_owned(),
            DEPTH_LANE => "v_depth".to_owned(),
            _ => format!("v_plane[{}]", lane - PLANE_GRAD_LANE_START),
        }
    };

    let mut failures = Vec::new();
    let mut kernel_nonzero = 0usize;
    let mut twin_nonzero = 0usize;
    for i in 0..n {
        for lane in 0..lanes {
            if lane == REFINE_LANE {
                continue;
            }
            let k = combined[i * lanes + lane];
            let t = twin.rows[i][lane];
            if k != 0.0 {
                kernel_nonzero += 1;
            }
            if t != 0.0 {
                twin_nonzero += 1;
            }
            let tol = 1.0e-6 + 2.0e-4 * t.abs().max(k.abs());
            if (k - t).abs() > tol {
                failures.push(format!(
                    "opacities {opacities:?} splat {i} {}: kernel {k:e} vs twin {t:e} \
                     (|Δ| = {:e} > tol {tol:e})",
                    lane_name(lane),
                    (k - t).abs()
                ));
            }
        }
    }

    // Gradient-sparsity check. A transposed or off-by-one lane keeps values of
    // plausible magnitude and slips past every "finite and nonzero" assertion;
    // it does NOT keep the zero structure, because the non-contributing splat's
    // whole row must be zero on both sides.
    assert_eq!(
        kernel_nonzero, twin_nonzero,
        "gradient sparsity differs: kernel has {kernel_nonzero} nonzero lanes, twin {twin_nonzero}"
    );
    for (i, contributes) in contributes.iter().enumerate() {
        if *contributes {
            continue;
        }
        for lane in 0..lanes {
            assert_eq!(
                combined[i * lanes + lane],
                0.0,
                "splat {i} never contributed, so lane {lane} must be exactly zero"
            );
        }
    }

    assert!(
        failures.is_empty(),
        "raster backward disagrees with the autodiff twin:\n  {}",
        failures.join("\n  "),
    );
}

/// Lane-level default-inertness, in both directions.
///
/// * A `RgbaAndDepth` backward (`render_plane = false`) must leave the four
///   plane lanes **exactly** zero — the widened `v_combined` buffer is allocated
///   unconditionally, so "nobody wrote it" has to mean bit-zero, not "small".
/// * With a plane-mode backward whose plane cotangent is zero, every pre-PGSR
///   lane must come out **bit-identical** to the `RgbaAndDepth` run. The plane
///   term is linear in `v_out`, so a zero cotangent must contribute exactly
///   nothing; anything else means a term is being added that does not depend on
///   the plane cotangent at all.
///
/// The complementary direction — a plane-mode render leaves NONE of those lanes
/// zero — is covered by the sparsity assertion in
/// `raster_backward_matches_the_autodiff_twin`.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn plane_lanes_are_inert_without_plane_mode() {
    let scene = twin_scene([0.62, 0.48, 0.55]);
    assert_off_the_discontinuities(&scene);
    let n = scene.len();
    let (projected, plane) = flat_projected(&scene);
    let device = brush_cube::test_helpers::test_device().await;
    let img = UVec2::new(IMG_W as u32, IMG_H as u32);
    let ids: Vec<u32> = (0..n as u32).collect();

    let full_cot = cotangent();
    // rgba + centre depth, no plane lanes. Derived, never literal.
    const DEPTH_CHANS: usize = crate::kernels::helpers::raster_out_channels(true, false) as usize;

    // rgba + centre depth only.
    let depth_out = rasterize_reference_full(&projected, None, true, img, BG, true);
    assert_eq!(depth_out.len(), PIXELS * DEPTH_CHANS);
    let mut depth_cot = Vec::with_capacity(PIXELS * DEPTH_CHANS);
    let mut plane_cot = Vec::with_capacity(PIXELS * CHANS);
    for p in 0..PIXELS {
        for c in 0..CHANS {
            let v = full_cot[p * CHANS + c];
            if c < DEPTH_CHANS {
                depth_cot.push(v);
                plane_cot.push(v);
            } else {
                plane_cot.push(0.0);
            }
        }
    }
    let depth_grads = <MainBackendBase as SplatBwdOps>::rasterize_bwd(
        cube_f32(&device, [IMG_H, IMG_W, DEPTH_CHANS], &depth_out),
        cube_f32(&device, [n, PROJECTED_LANES_USIZE], &projected),
        cube_u32(&device, [n], &ids),
        cube_u32(&device, [1, 1, 2], &[0, n as u32]),
        None,
        cube_u32(&device, [n], &ids),
        BG,
        img,
        cube_f32(&device, [IMG_H, IMG_W, DEPTH_CHANS], &depth_cot),
        true,
        true,
        false,
    );
    let depth_lanes = read_f32(depth_grads.v_combined).await;

    // Plane mode, same cotangent on the first five channels and ZERO on the
    // four plane channels.
    let plane_out = rasterize_reference_full(&projected, Some(&plane), true, img, BG, true);
    let plane_grads = <MainBackendBase as SplatBwdOps>::rasterize_bwd(
        cube_f32(&device, [IMG_H, IMG_W, CHANS], &plane_out),
        cube_f32(&device, [n, PROJECTED_LANES_USIZE], &projected),
        cube_u32(&device, [n], &ids),
        cube_u32(&device, [1, 1, 2], &[0, n as u32]),
        Some(cube_f32(&device, [n, PLANE_AUX_LANES_USIZE], &plane)),
        cube_u32(&device, [n], &ids),
        BG,
        img,
        cube_f32(&device, [IMG_H, IMG_W, CHANS], &plane_cot),
        true,
        true,
        true,
    );
    let plane_lanes = read_f32(plane_grads.v_combined).await;

    let lanes = crate::bwd::COMPACT_GRAD_LANES as usize;
    let plane_start = PLANE_GRAD_LANE_START;
    for i in 0..n {
        for lane in plane_start..lanes {
            assert_eq!(
                depth_lanes[i * lanes + lane],
                0.0,
                "RgbaAndDepth must leave plane lane {lane} bit-zero (splat {i})"
            );
            assert_eq!(
                plane_lanes[i * lanes + lane],
                0.0,
                "a zero plane cotangent must give a bit-zero plane value grad (splat {i})"
            );
        }
        for lane in 0..plane_start {
            assert_eq!(
                plane_lanes[i * lanes + lane],
                depth_lanes[i * lanes + lane],
                "lane {lane} of splat {i} moved when the plane channels carried no cotangent"
            );
        }
    }
}
