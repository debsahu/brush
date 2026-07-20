//! Image-loss kernels for Brush.
//!
//! GT lives on the GPU as a `Tensor<u32>` of shape `[H, W]`, where each u32
//! packs `[r8, g8, b8, a8]` (LSB → MSB). Conversion to f32 happens inside
//! the kernels via shift-and-divide-by-255. No f32 GT image is ever
//! materialised on the autograd tape.
//!
//! Public surface:
//! - [`image_loss`]: per-pixel `l1_w * |pred - gt_eff| + ssim_w * ssim(pred, gt_eff)`,
//!   with optional background-compositing of GT (`gt_eff = gt + (1 - gt.a) * bg`)
//!   and optional mask multiplication (`out = out * gt.a`) folded into the kernel.
//! - [`image_loss_eval`]: forward-only loss map for non-differentiable backends.
//!
//! Backward normally recomputes SSIM partials inline. Apple Silicon native-MSL
//! builds can opt into saving the same f32 partials on the autograd tape to
//! trade memory for a faster backward pass.

use brush_cube::{MainBackend, MainBackendBase};
use brush_render::burn_glue::{
    AutodiffMain, unwrap_ad_wgpu_float, unwrap_ad_wgpu_int, unwrap_wgpu_float, unwrap_wgpu_int,
    wrap_ad_wgpu_float, wrap_wgpu_float,
};
use burn::{
    backend::{
        Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
        wgpu::WgpuRuntime,
    },
    tensor::{DType, Int, Shape, Tensor},
};
use burn_cubecl::{
    CubeRuntime, fusion::FusionCubeRuntime, kernel::into_contiguous, tensor::CubeTensor,
};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use glam::Vec3;

#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn use_saved_loss_partials() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = brush_render::native_msl::option_requested(
            brush_render::native_msl::SAVED_LOSS_PARTIALS_ENV,
        );
        if enabled {
            tracing::warn!("experimental native-MSL saved loss partials enabled");
        }
        enabled
    })
}

#[cfg(not(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
)))]
fn use_saved_loss_partials() -> bool {
    false
}

mod kernels {
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::cube;
    use burn_cubecl::cubecl::frontend::CompilationArg;
    use burn_cubecl::cubecl::frontend::IndexMutExpand;
    use burn_cubecl::cubecl::prelude::*;

    /// 11-tap Gaussian weights at sigma = 1.5, normalised to sum to 1.
    /// Called from `comptime!` so it runs once per kernel build, baking each
    /// weight as an f32 literal into the generated kernel.
    fn gauss_taps() -> [f32; 11] {
        let sigma = 1.5_f32;
        let mut w = [0.0_f32; 11];
        let mut sum = 0.0;
        for (i, w) in w.iter_mut().enumerate() {
            let x = i as f32 - 5.0;
            *w = (-x * x / (2.0 * sigma * sigma)).exp();
            sum += *w;
        }
        for w in &mut w {
            *w /= sum;
        }
        w
    }

    pub const BLOCK_X: u32 = 16;
    pub const BLOCK_Y: u32 = 16;
    const HALO: u32 = 5;
    const SHARED_X: u32 = BLOCK_X + 2 * HALO; // 26
    const SHARED_Y: u32 = BLOCK_Y + 2 * HALO; // 26
    pub const BWD_TILE_SMALL: u32 = 8;
    pub const BWD_TILE_LARGE: u32 = 16;

    const fn backward_shared_elements(tile: u32) -> usize {
        let shared = tile + 2 * HALO;
        let extended = tile + 4 * HALO;
        (extended * extended * 2 + extended * shared * 5) as usize
    }

    /// Shared-memory footprint of the fast 16x16 f32 specialization.
    pub const BWD_LARGE_SHARED_BYTES: usize =
        backward_shared_elements(BWD_TILE_LARGE) * size_of::<f32>();

    const C1: f32 = 0.01 * 0.01;
    const C2: f32 = 0.03 * 0.03;
    const INV_255: f32 = 1.0 / 255.0;

    /// Read `pred[c, y, x]` returning zero for out-of-bounds. The
    /// `if/else` form generated a non-uniform branch that Naga's MSL
    /// backend tracked into the post-load `workgroupBarrier()`; we use
    /// `select` to keep control flow uniform. The read always executes —
    /// for OOB threads `(y, x) = (0, 0)` (see `coords`), so the index
    /// `c * h * w + 0` is always in-bounds.
    #[cube]
    fn read_pred<F: Float>(
        pred: &Tensor<F>,
        c: u32,
        y: u32,
        x: u32,
        oob: bool,
        h: u32,
        w: u32,
    ) -> F {
        let v = pred[(c * h * w + y * w + x) as usize];
        select(oob, F::cast_from(0.0_f32), v)
    }

    /// Read one `[r8 g8 b8 a8]`-packed pixel from `gt_packed`. Returns the
    /// requested colour byte and the alpha byte, both in `[0, 1]`. The alpha
    /// is always returned so it's available for compositing or masking when
    /// those flags are on. As with `read_pred`, the body runs unconditionally
    /// and `oob` is folded in via `select` so we don't emit a non-uniform
    /// branch before a workgroup barrier.
    #[cube]
    fn read_gt<F: Float>(
        gt_packed: &Tensor<u32>,
        c: u32,
        y: u32,
        x: u32,
        oob: bool,
        w: u32,
    ) -> (F, F) {
        let val = gt_packed[(y * w + x) as usize];
        let byte_c = f32::cast_from((val >> (c * 8u32)) & 0xffu32);
        let byte_a = f32::cast_from((val >> 24u32) & 0xffu32);
        let zero = F::cast_from(0.0_f32);
        let gt_c = F::cast_from(byte_c * INV_255);
        let gt_a = F::cast_from(byte_a * INV_255);
        (select(oob, zero, gt_c), select(oob, zero, gt_a))
    }

    /// Map a tile-local position offset by `halo` to global image coords.
    #[cube]
    fn coords(
        tile_y0: u32,
        tile_x0: u32,
        local_y: u32,
        local_x: u32,
        #[comptime] halo: u32,
        h: u32,
        w: u32,
    ) -> (u32, u32, bool) {
        let total_y = tile_y0 + local_y;
        let total_x = tile_x0 + local_x;
        let oob_under = total_y < halo || total_x < halo;
        let zero = u32::cast_from(0u32);
        let gy = select(oob_under, zero, total_y - halo);
        let gx = select(oob_under, zero, total_x - halo);
        (gy, gx, oob_under || gy >= h || gx >= w)
    }

    #[cube]
    fn gw<F: Float>(#[comptime] i: u32) -> F {
        F::new(comptime![gauss_taps()[i as usize]])
    }

    #[cube]
    fn ssim_partials<F: Float>(mu1: F, mu2: F, a: F, b: F, c_top: F, d_top: F) -> (F, F, F) {
        let zero = F::cast_from(0.0_f32);
        let one = F::cast_from(1.0_f32);
        let two = F::cast_from(2.0_f32);
        let inv_ab = one / (a * b);
        let cd = c_top * d_top * inv_ab;
        let clamped = cd < F::cast_from(-1.0_f32) || cd > one;
        let dmu1 = if clamped {
            zero
        } else {
            two * mu2 * inv_ab * (d_top - c_top) - two * mu1 * cd * (one / a - one / b)
        };
        let dsigma1 = if clamped { zero } else { -cd / b };
        let dsigma12 = if clamped { zero } else { two * c_top * inv_ab };
        (dmu1, dsigma1, dsigma12)
    }

    /// Read one saved SSIM partial, returning zero for image padding. The
    /// cache is `SoA` `[partial=3, rgb=3, H, W]`, flattened as `[9, H, W]`.
    #[cube]
    fn read_saved_partial<F: Float>(
        partials: &Tensor<F>,
        partial: u32,
        c: u32,
        y: u32,
        x: u32,
        oob: bool,
        h: u32,
        w: u32,
    ) -> F {
        let idx = (((partial * 3u32 + c) * h + y) * w + x) as usize;
        select(oob, F::cast_from(0.0_f32), partials[idx])
    }

    /// L1-only training/eval specialization. Keeping this separate from the
    /// SSIM kernel guarantees zero SSIM weight does not allocate shared tiles
    /// or execute either 11-tap blur.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch)]
    pub fn image_l1_forward_kernel<F: Float>(
        pred: &Tensor<F>,
        gt_packed: &Tensor<u32>,
        loss_map: &mut Tensor<F>,
        h: u32,
        w: u32,
        l1_weight: f32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
        #[comptime] mask: bool,
    ) {
        let c = CUBE_POS_Z;
        let pix_y = CUBE_POS_Y * BLOCK_Y + UNIT_POS_Y;
        let pix_x = CUBE_POS_X * BLOCK_X + UNIT_POS_X;
        if pix_x >= w || pix_y >= h {
            terminate!();
        }

        let idx = (c * h * w + pix_y * w + pix_x) as usize;
        let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
        let gt_eff = if c == 3u32 {
            gt_a
        } else if composite {
            let bg_c = if c == 0u32 {
                F::cast_from(bg_r)
            } else if c == 1u32 {
                F::cast_from(bg_g)
            } else {
                F::cast_from(bg_b)
            };
            gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
        } else {
            gt_c
        };
        let weight = if c == 3u32 {
            F::cast_from(1.0_f32)
        } else {
            F::cast_from(l1_weight)
        };
        let mut loss = weight * F::abs(pred[idx] - gt_eff);
        if mask {
            loss = loss * gt_a;
        }
        loss_map[idx] = loss;
    }

    /// VJP matching [`image_l1_forward_kernel()`], with one independent thread
    /// per output element and no shared memory.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch)]
    pub fn image_l1_backward_kernel<F: Float>(
        pred: &Tensor<F>,
        gt_packed: &Tensor<u32>,
        dl_dmap: &Tensor<F>,
        dl_dpred: &mut Tensor<F>,
        h: u32,
        w: u32,
        l1_weight: f32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
        #[comptime] mask: bool,
    ) {
        let c = CUBE_POS_Z;
        let pix_y = CUBE_POS_Y * BLOCK_Y + UNIT_POS_Y;
        let pix_x = CUBE_POS_X * BLOCK_X + UNIT_POS_X;
        if pix_x >= w || pix_y >= h {
            terminate!();
        }

        let idx = (c * h * w + pix_y * w + pix_x) as usize;
        let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
        let gt_eff = if c == 3u32 {
            gt_a
        } else if composite {
            let bg_c = if c == 0u32 {
                F::cast_from(bg_r)
            } else if c == 1u32 {
                F::cast_from(bg_g)
            } else {
                F::cast_from(bg_b)
            };
            gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
        } else {
            gt_c
        };
        let diff = pred[idx] - gt_eff;
        let zero = F::cast_from(0.0_f32);
        let sign = if diff > zero {
            F::cast_from(1.0_f32)
        } else if diff < zero {
            F::cast_from(-1.0_f32)
        } else {
            zero
        };
        let weight = if c == 3u32 {
            F::cast_from(1.0_f32)
        } else {
            F::cast_from(l1_weight)
        };
        let mut chain = dl_dmap[idx];
        if mask {
            chain = chain * gt_a;
        }
        dl_dpred[idx] = weight * sign * chain;
    }

    /// Forward: produce the L1 + SSIM loss map. When dispatched with `C = 4`,
    /// the workgroup at `c == 3` produces `|pred.a - gt.a|` into the alpha
    /// channel of the loss map — folding the previously-separate alpha-match
    /// kernel into the same launch.
    ///
    /// Comptime flags:
    /// - `composite`: apply `gt + (1 - gt.a) * bg` to the gt sample. Set when
    ///   the source has real alpha and `bg != 0`; opaque/synthesised alpha or
    ///   zero bg make the math a no-op so callers gate it off to skip the work.
    /// - `mask`: multiply the loss-map output by `gt.a` per pixel.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch)]
    pub fn image_loss_forward_kernel<F: Float>(
        pred: &Tensor<F>,
        gt_packed: &Tensor<u32>,
        loss_map: &mut Tensor<F>,
        saved_partials: ComptimeOption<&mut Tensor<F>>,
        h: u32,
        w: u32,
        l1_weight: f32,
        ssim_weight: f32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
        #[comptime] mask: bool,
    ) {
        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * BLOCK_Y;
        let tile_x0 = CUBE_POS_X * BLOCK_X;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;

        // Alpha-match channel: simple per-pixel `|pred - gt.a|`, no blur.
        if c == 3u32 {
            if pix_x < w && pix_y < h {
                let idx = (3u32 * h * w + pix_y * w + pix_x) as usize;
                let (_, gt_a) = read_gt::<F>(gt_packed, 0u32, pix_y, pix_x, false, w);
                let mut v = F::abs(pred[idx] - gt_a);
                if mask {
                    v = v * gt_a;
                }
                loss_map[idx] = v;
            }
            terminate!();
        }

        // Tile + halo of (pred, gt_eff_c) interleaved as 2 floats. This
        // 16x16 forward layout uses about 13.4 KiB of shared memory, below the
        // WebGPU downlevel limit. gt_a was previously carried here too; the
        // mask=true path now re-reads it at the centre.
        let mut s_tile = Shared::new_slice((SHARED_Y * SHARED_X * 2) as usize);
        let mut x_conv = Shared::new_slice((SHARED_Y * BLOCK_X * 5) as usize);

        let bg_c = if composite {
            if c == 0u32 {
                F::cast_from(bg_r)
            } else if c == 1u32 {
                F::cast_from(bg_g)
            } else {
                F::cast_from(bg_b)
            }
        } else {
            F::cast_from(0.0_f32)
        };

        let thread_rank = UNIT_POS_Y * BLOCK_X + UNIT_POS_X;
        let threads = BLOCK_X * BLOCK_Y;
        let tile_size = SHARED_Y * SHARED_X;
        #[unroll]
        for s in 0u32..3u32 {
            let tid = s * threads + thread_rank;
            if tid < tile_size {
                let local_y = tid / SHARED_X;
                let local_x = tid % SHARED_X;
                let (gy, gx, oob) = coords(tile_y0, tile_x0, local_y, local_x, HALO, h, w);
                let pv = read_pred::<F>(pred, c, gy, gx, oob, h, w);
                let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                let gt_eff = if composite {
                    gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
                } else {
                    gt_c
                };
                let base = ((local_y * SHARED_X + local_x) * 2u32) as usize;
                s_tile[base] = pv;
                s_tile[base + 1] = gt_eff;
            }
        }
        sync_cube();

        // Horizontal 11-tap blur over (pred, gt_eff_c) -> 5 sums per pixel.
        let lx = UNIT_POS_X + HALO;
        #[unroll]
        for pass in 0u32..2u32 {
            let ly = UNIT_POS_Y + pass * BLOCK_Y;
            if ly < SHARED_Y {
                let mut sum_x = F::cast_from(0.0_f32);
                let mut sum_x2 = F::cast_from(0.0_f32);
                let mut sum_y = F::cast_from(0.0_f32);
                let mut sum_y2 = F::cast_from(0.0_f32);
                let mut sum_xy = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let il = (ly * SHARED_X + (lx - d)) as usize;
                    let ir = (ly * SHARED_X + (lx + d)) as usize;
                    let xl = s_tile[il * 2];
                    let yl = s_tile[il * 2 + 1];
                    let xr = s_tile[ir * 2];
                    let yr = s_tile[ir * 2 + 1];
                    sum_x += (xl + xr) * w_d;
                    sum_x2 += (xl * xl + xr * xr) * w_d;
                    sum_y += (yl + yr) * w_d;
                    sum_y2 += (yl * yl + yr * yr) * w_d;
                    sum_xy += (xl * yl + xr * yr) * w_d;
                }
                let ic = (ly * SHARED_X + lx) as usize;
                let xc = s_tile[ic * 2];
                let yc = s_tile[ic * 2 + 1];
                let wc = gw::<F>(5u32);
                sum_x += xc * wc;
                sum_x2 += xc * xc * wc;
                sum_y += yc * wc;
                sum_y2 += yc * yc * wc;
                sum_xy += xc * yc * wc;
                let base = ((ly * BLOCK_X + UNIT_POS_X) * 5) as usize;
                x_conv[base] = sum_x;
                x_conv[base + 1] = sum_x2;
                x_conv[base + 2] = sum_y;
                x_conv[base + 3] = sum_y2;
                x_conv[base + 4] = sum_xy;
            }
        }
        sync_cube();

        // Vertical 11-tap blur, then derive SSIM and emit L1 + SSIM loss.
        let ly = UNIT_POS_Y + HALO;
        let lx = UNIT_POS_X;
        let mut out0 = F::cast_from(0.0_f32);
        let mut out1 = F::cast_from(0.0_f32);
        let mut out2 = F::cast_from(0.0_f32);
        let mut out3 = F::cast_from(0.0_f32);
        let mut out4 = F::cast_from(0.0_f32);
        #[unroll]
        for d in 1u32..6u32 {
            let w_d = gw::<F>(comptime![5u32 - d]);
            let bt = (((ly - d) * BLOCK_X + lx) * 5) as usize;
            let bb = (((ly + d) * BLOCK_X + lx) * 5) as usize;
            out0 += (x_conv[bt] + x_conv[bb]) * w_d;
            out1 += (x_conv[bt + 1] + x_conv[bb + 1]) * w_d;
            out2 += (x_conv[bt + 2] + x_conv[bb + 2]) * w_d;
            out3 += (x_conv[bt + 3] + x_conv[bb + 3]) * w_d;
            out4 += (x_conv[bt + 4] + x_conv[bb + 4]) * w_d;
        }
        let bc = ((ly * BLOCK_X + lx) * 5) as usize;
        let wc = gw::<F>(5u32);
        out0 += x_conv[bc] * wc;
        out1 += x_conv[bc + 1] * wc;
        out2 += x_conv[bc + 2] * wc;
        out3 += x_conv[bc + 3] * wc;
        out4 += x_conv[bc + 4] * wc;

        if pix_x < w && pix_y < h {
            let zero = F::cast_from(0.0_f32);
            let two = F::cast_from(2.0_f32);
            let mu1 = out0;
            let mu2 = out2;
            let mu1_sq = mu1 * mu1;
            let mu2_sq = mu2 * mu2;
            let sigma1_sq = F::max(zero, out1 - mu1_sq);
            let sigma2_sq = F::max(zero, out3 - mu2_sq);
            let sigma12 = out4 - mu1 * mu2;
            let a = mu1_sq + mu2_sq + F::new(C1);
            let b = sigma1_sq + sigma2_sq + F::new(C2);
            let c_top = two * mu1 * mu2 + F::new(C1);
            let d_top = two * sigma12 + F::new(C2);
            let raw = (c_top * d_top) / (a * b);
            let val = clamp(raw, F::cast_from(-1.0_f32), F::cast_from(1.0_f32));

            let centre = ((UNIT_POS_Y + HALO) * SHARED_X + (UNIT_POS_X + HALO)) as usize;
            let p1 = s_tile[centre * 2];
            let p2 = s_tile[centre * 2 + 1];
            let l1 = F::abs(p1 - p2);
            let mut loss_v = F::cast_from(l1_weight) * l1 + F::cast_from(ssim_weight) * val;
            if mask {
                let (_, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
                loss_v = loss_v * gt_a;
            }
            let global_idx = (c * h * w + pix_y * w + pix_x) as usize;
            loss_map[global_idx] = loss_v;
            #[comptime]
            if let ComptimeOption::Some(saved_partials) = saved_partials {
                let (dmu1, dsigma1, dsigma12) = ssim_partials::<F>(mu1, mu2, a, b, c_top, d_top);
                let pixel_idx = (pix_y * w + pix_x) as usize;
                saved_partials[((c) * h * w) as usize + pixel_idx] = dmu1;
                saved_partials[((3u32 + c) * h * w) as usize + pixel_idx] = dsigma1;
                saved_partials[((6u32 + c) * h * w) as usize + pixel_idx] = dsigma12;
            }
        }
    }

    /// Backward: either recompute SSIM partials inline or consume the optional
    /// `[9, H, W]` f32 partial tensor saved by the training forward.
    ///
    /// Each `sync_cube` boundary frees a scratch role, so the four logical
    /// arrays alias into two physical buffers. The host selects a 16x16 tile
    /// (29,088 bytes shared memory) when device limits allow it and otherwise
    /// falls back to 8x8 (16,352 bytes). `tile` is comptime, so both choices
    /// compile to dedicated kernels with no shader-side branch.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch)]
    pub fn image_loss_backward_kernel<F: Float>(
        pred: &Tensor<F>,
        gt_packed: &Tensor<u32>,
        dl_dmap: &Tensor<F>,
        saved_partials: ComptimeOption<&Tensor<F>>,
        dl_dpred: &mut Tensor<F>,
        h: u32,
        w: u32,
        l1_weight: f32,
        ssim_weight: f32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
        #[comptime] mask: bool,
        #[comptime] tile: u32,
    ) {
        let shared = comptime![tile + 2u32 * HALO];
        let extended = comptime![tile + 4u32 * HALO];
        let threads = comptime![tile * tile];
        let load_iters = comptime![(extended * extended).div_ceil(threads)];
        let hblur_iters = comptime![(extended * shared).div_ceil(threads)];
        let partial_iters = comptime![(shared * shared).div_ceil(threads)];
        let inner_h_passes = comptime![shared.div_ceil(tile)];
        let saved = comptime![saved_partials.is_some()];

        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * tile;
        let tile_x0 = CUBE_POS_X * tile;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;

        // Alpha-match channel: simple sign-of-diff. No SSIM machinery.
        if c == 3u32 {
            if pix_x < w && pix_y < h {
                let idx = (3u32 * h * w + pix_y * w + pix_x) as usize;
                let (_, gt_a) = read_gt::<F>(gt_packed, 0u32, pix_y, pix_x, false, w);
                let diff = pred[idx] - gt_a;
                let zero = F::cast_from(0.0_f32);
                let sign = if diff > zero {
                    F::cast_from(1.0_f32)
                } else if diff < zero {
                    F::cast_from(-1.0_f32)
                } else {
                    zero
                };
                let mut chain = dl_dmap[idx];
                if mask {
                    chain = chain * gt_a;
                }
                dl_dpred[idx] = sign * chain;
            }
            terminate!();
        }

        // In recompute mode buf_a/b hold the image tile and first blur before
        // being reused for chain*partials and the second blur. The saved mode
        // compiles to the smaller allocations only (13,104 bytes at 16x16).
        let mut buf_a = Shared::new_slice(comptime![if saved {
            (shared * shared * 3u32) as usize
        } else {
            (extended * extended * 2u32) as usize
        }]);
        let mut buf_b = Shared::new_slice(comptime![if saved {
            (shared * tile * 3u32) as usize
        } else {
            (extended * shared * 5u32) as usize
        }]);

        let bg_c = if composite {
            if c == 0u32 {
                F::cast_from(bg_r)
            } else if c == 1u32 {
                F::cast_from(bg_g)
            } else {
                F::cast_from(bg_b)
            }
        } else {
            F::cast_from(0.0_f32)
        };

        let thread_rank = UNIT_POS_Y * tile + UNIT_POS_X;

        #[comptime]
        match saved_partials {
            ComptimeOption::None => {
                // Load pred and effective-gt with halo of 2*HALO into buf_a.
                let ext_size = extended * extended;
                #[unroll]
                for s in 0u32..load_iters {
                    let tid = s * threads + thread_rank;
                    if tid < ext_size {
                        let local_y = tid / extended;
                        let local_x = tid % extended;
                        let (gy, gx, oob) =
                            coords(tile_y0, tile_x0, local_y, local_x, 2u32 * HALO, h, w);
                        let pv = read_pred::<F>(pred, c, gy, gx, oob, h, w);
                        let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                        let gt_eff = if composite {
                            gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
                        } else {
                            gt_c
                        };
                        let base = ((local_y * extended + local_x) * 2u32) as usize;
                        buf_a[base] = pv;
                        buf_a[base + 1] = gt_eff;
                    }
                }
                sync_cube();

                // Horizontal blur over the extended tile.
                let horiz_size = extended * shared;
                #[unroll]
                for s in 0u32..hblur_iters {
                    let tid = s * threads + thread_rank;
                    if tid < horiz_size {
                        let row_y = tid / shared;
                        let col_x = tid % shared;
                        let center = col_x + HALO;
                        let mut sum_x = F::cast_from(0.0_f32);
                        let mut sum_x2 = F::cast_from(0.0_f32);
                        let mut sum_y = F::cast_from(0.0_f32);
                        let mut sum_y2 = F::cast_from(0.0_f32);
                        let mut sum_xy = F::cast_from(0.0_f32);
                        #[unroll]
                        for d in 1u32..6u32 {
                            let w_d = gw::<F>(comptime![5u32 - d]);
                            let il = ((row_y * extended + (center - d)) * 2u32) as usize;
                            let ir = ((row_y * extended + (center + d)) * 2u32) as usize;
                            let xl = buf_a[il];
                            let yl = buf_a[il + 1];
                            let xr = buf_a[ir];
                            let yr = buf_a[ir + 1];
                            sum_x += (xl + xr) * w_d;
                            sum_x2 += (xl * xl + xr * xr) * w_d;
                            sum_y += (yl + yr) * w_d;
                            sum_y2 += (yl * yl + yr * yr) * w_d;
                            sum_xy += (xl * yl + xr * yr) * w_d;
                        }
                        let ic = ((row_y * extended + center) * 2u32) as usize;
                        let xc = buf_a[ic];
                        let yc = buf_a[ic + 1];
                        let wc = gw::<F>(5u32);
                        sum_x += xc * wc;
                        sum_x2 += xc * xc * wc;
                        sum_y += yc * wc;
                        sum_y2 += yc * yc * wc;
                        sum_xy += xc * yc * wc;
                        let base = ((row_y * shared + col_x) * 5u32) as usize;
                        buf_b[base] = sum_x;
                        buf_b[base + 1] = sum_x2;
                        buf_b[base + 2] = sum_y;
                        buf_b[base + 3] = sum_y2;
                        buf_b[base + 4] = sum_xy;
                    }
                }
                sync_cube();

                // Vertical blur, derive SSIM partials, multiply by chain * (mask if any).
                // Reuses buf_a (image tile is dead) for chain*partials.
                let partial_size = shared * shared;
                #[unroll]
                for s in 0u32..partial_iters {
                    let tid = s * threads + thread_rank;
                    if tid < partial_size {
                        let part_y = tid / shared;
                        let part_x = tid % shared;
                        let center = part_y + HALO;

                        let mut out0 = F::cast_from(0.0_f32);
                        let mut out1 = F::cast_from(0.0_f32);
                        let mut out2 = F::cast_from(0.0_f32);
                        let mut out3 = F::cast_from(0.0_f32);
                        let mut out4 = F::cast_from(0.0_f32);
                        #[unroll]
                        for d in 1u32..6u32 {
                            let w_d = gw::<F>(comptime![5u32 - d]);
                            let bt = (((center - d) * shared + part_x) * 5u32) as usize;
                            let bb = (((center + d) * shared + part_x) * 5u32) as usize;
                            out0 += (buf_b[bt] + buf_b[bb]) * w_d;
                            out1 += (buf_b[bt + 1] + buf_b[bb + 1]) * w_d;
                            out2 += (buf_b[bt + 2] + buf_b[bb + 2]) * w_d;
                            out3 += (buf_b[bt + 3] + buf_b[bb + 3]) * w_d;
                            out4 += (buf_b[bt + 4] + buf_b[bb + 4]) * w_d;
                        }
                        let bc = ((center * shared + part_x) * 5u32) as usize;
                        let wc = gw::<F>(5u32);
                        out0 += buf_b[bc] * wc;
                        out1 += buf_b[bc + 1] * wc;
                        out2 += buf_b[bc + 2] * wc;
                        out3 += buf_b[bc + 3] * wc;
                        out4 += buf_b[bc + 4] * wc;

                        let zero = F::cast_from(0.0_f32);
                        let two = F::cast_from(2.0_f32);
                        let mu1 = out0;
                        let mu2 = out2;
                        let mu1_sq = mu1 * mu1;
                        let mu2_sq = mu2 * mu2;
                        let sigma1_sq = F::max(zero, out1 - mu1_sq);
                        let sigma2_sq = F::max(zero, out3 - mu2_sq);
                        let sigma12 = out4 - mu1 * mu2;
                        let a = mu1_sq + mu2_sq + F::new(C1);
                        let b = sigma1_sq + sigma2_sq + F::new(C2);
                        let c_top = two * mu1 * mu2 + F::new(C1);
                        let d_top = two * sigma12 + F::new(C2);
                        let (dmu1, dsigma1, dsigma12) =
                            ssim_partials::<F>(mu1, mu2, a, b, c_top, d_top);

                        let (gy, gx, oob) = coords(tile_y0, tile_x0, part_y, part_x, HALO, h, w);
                        let mut chain = read_pred::<F>(dl_dmap, c, gy, gx, oob, h, w);
                        if mask {
                            let (_unused, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                            chain = chain * gt_a;
                        }

                        let base = ((part_y * shared + part_x) * 3u32) as usize;
                        buf_a[base] = dmu1 * chain;
                        buf_a[base + 1] = dsigma1 * chain;
                        buf_a[base + 2] = dsigma12 * chain;
                    }
                }
                sync_cube();
            }
            ComptimeOption::Some(saved_partials) => {
                // Load saved partials with one halo, fold in the arbitrary
                // upstream chain and optional alpha mask, then join the common
                // second-blur/finalization path below.
                let partial_size = shared * shared;
                #[unroll]
                for s in 0u32..partial_iters {
                    let tid = s * threads + thread_rank;
                    if tid < partial_size {
                        let part_y = tid / shared;
                        let part_x = tid % shared;
                        let (gy, gx, oob) = coords(tile_y0, tile_x0, part_y, part_x, HALO, h, w);
                        let mut chain = read_pred::<F>(dl_dmap, c, gy, gx, oob, h, w);
                        if mask {
                            let (_unused, gt_a) = read_gt::<F>(gt_packed, c, gy, gx, oob, w);
                            chain = chain * gt_a;
                        }
                        let base = ((part_y * shared + part_x) * 3u32) as usize;
                        buf_a[base] =
                            read_saved_partial::<F>(saved_partials, 0u32, c, gy, gx, oob, h, w)
                                * chain;
                        buf_a[base + 1] =
                            read_saved_partial::<F>(saved_partials, 1u32, c, gy, gx, oob, h, w)
                                * chain;
                        buf_a[base + 2] =
                            read_saved_partial::<F>(saved_partials, 2u32, c, gy, gx, oob, h, w)
                                * chain;
                    }
                }
                sync_cube();
            }
        }

        // Second horizontal blur over chain * partials.
        // Reuses buf_b (1st-blur sums are dead) for the inner-blur output.
        let lx_b = UNIT_POS_X + HALO;
        #[unroll]
        for pass in 0u32..inner_h_passes {
            let ly_b = UNIT_POS_Y + pass * tile;
            if ly_b < shared {
                let mut a0 = F::cast_from(0.0_f32);
                let mut a1 = F::cast_from(0.0_f32);
                let mut a2 = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let il = ((ly_b * shared + (lx_b - d)) * 3u32) as usize;
                    let ir = ((ly_b * shared + (lx_b + d)) * 3u32) as usize;
                    a0 += (buf_a[il] + buf_a[ir]) * w_d;
                    a1 += (buf_a[il + 1] + buf_a[ir + 1]) * w_d;
                    a2 += (buf_a[il + 2] + buf_a[ir + 2]) * w_d;
                }
                let ic = ((ly_b * shared + lx_b) * 3u32) as usize;
                let wc = gw::<F>(5u32);
                a0 += buf_a[ic] * wc;
                a1 += buf_a[ic + 1] * wc;
                a2 += buf_a[ic + 2] * wc;
                let base = ((ly_b * tile + UNIT_POS_X) * 3u32) as usize;
                buf_b[base] = a0;
                buf_b[base + 1] = a1;
                buf_b[base + 2] = a2;
            }
        }
        sync_cube();

        // Second vertical blur + L1 sign + write.
        if pix_x < w && pix_y < h {
            let ly = UNIT_POS_Y + HALO;
            let lx = UNIT_POS_X;
            let mut s0 = F::cast_from(0.0_f32);
            let mut s1 = F::cast_from(0.0_f32);
            let mut s2 = F::cast_from(0.0_f32);
            #[unroll]
            for d in 1u32..6u32 {
                let w_d = gw::<F>(comptime![5u32 - d]);
                let bt = (((ly - d) * tile + lx) * 3u32) as usize;
                let bb = (((ly + d) * tile + lx) * 3u32) as usize;
                s0 += (buf_b[bt] + buf_b[bb]) * w_d;
                s1 += (buf_b[bt + 1] + buf_b[bb + 1]) * w_d;
                s2 += (buf_b[bt + 2] + buf_b[bb + 2]) * w_d;
            }
            let bc = ((ly * tile + lx) * 3u32) as usize;
            let wc = gw::<F>(5u32);
            s0 += buf_b[bc] * wc;
            s1 += buf_b[bc + 1] * wc;
            s2 += buf_b[bc + 2] * wc;

            let pix_idx = (c * h * w + pix_y * w + pix_x) as usize;
            let p1 = pred[pix_idx];
            let (gt_c, gt_a) = read_gt::<F>(gt_packed, c, pix_y, pix_x, false, w);
            let gt_eff = if composite {
                gt_c + (F::cast_from(1.0_f32) - gt_a) * bg_c
            } else {
                gt_c
            };
            let ssim_grad = s0 + (F::cast_from(2.0_f32) * p1) * s1 + gt_eff * s2;
            let diff = p1 - gt_eff;
            let zero = F::cast_from(0.0_f32);
            let l1_sign = if diff > zero {
                F::cast_from(1.0_f32)
            } else if diff < zero {
                F::cast_from(-1.0_f32)
            } else {
                zero
            };
            let mut chain_centre = dl_dmap[pix_idx];
            if mask {
                chain_centre = chain_centre * gt_a;
            }
            dl_dpred[pix_idx] = F::cast_from(ssim_weight) * ssim_grad
                + F::cast_from(l1_weight) * l1_sign * chain_centre;
        }
    }

    /// Decode `gt_packed` to `[H, W, 3]` f32 RGB. Comptime `composite` gates
    /// the `gt + (1 - gt.a) * bg` math; callers pass false when the source
    /// has no real alpha or when `bg == 0`. Used by the LPIPS path.
    #[cube(launch)]
    pub fn unpack_gt_rgb_kernel<F: Float>(
        gt_packed: &Tensor<u32>,
        out: &mut Tensor<F>,
        h: u32,
        w: u32,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
        #[comptime] composite: bool,
    ) {
        let pix_y = CUBE_POS_Y * BLOCK_Y + UNIT_POS_Y;
        let pix_x = CUBE_POS_X * BLOCK_X + UNIT_POS_X;
        if pix_x >= w || pix_y >= h {
            terminate!();
        }
        let val = gt_packed[(pix_y * w + pix_x) as usize];
        let mut r = f32::cast_from(val & 0xffu32) * INV_255;
        let mut g = f32::cast_from((val >> 8u32) & 0xffu32) * INV_255;
        let mut b = f32::cast_from((val >> 16u32) & 0xffu32) * INV_255;
        if composite {
            let inv_a = 1.0_f32 - f32::cast_from(val >> 24u32) * INV_255;
            r += inv_a * bg_r;
            g += inv_a * bg_g;
            b += inv_a * bg_b;
        }
        let base = ((pix_y * w + pix_x) * 3u32) as usize;
        out[base] = F::cast_from(r);
        out[base + 1] = F::cast_from(g);
        out[base + 2] = F::cast_from(b);
    }
}

/// Image-loss configuration.
///
/// `composite_bg = Some(bg)` folds `gt + (1 - gt.a) * bg` into the kernel
/// before comparing against `pred`. `None` skips the math entirely — set it
/// when GT has no real alpha (synthesised `a = 1` makes the term zero) or
/// when `bg == 0`, since the kernel pays for the always-on math otherwise.
#[derive(Debug, Clone, Copy)]
pub struct ImageLossConfig {
    pub l1_weight: f32,
    pub ssim_weight: f32,
    pub composite_bg: Option<Vec3>,
    /// If true, multiply each loss-map pixel by `gt.a`.
    pub mask: bool,
}

/// Training-only result that keeps the three RGB SSIM partials needed by
/// backward in one planar `[9, H, W]` f32 tensor.
#[derive(Debug, Clone)]
struct ImageLossForwardSaved<B: Backend> {
    map: FloatTensor<B>,
    partials: FloatTensor<B>,
}

/// Backend hooks for the loss kernels. When `pred` has 4 channels, the
/// `c == 3` workgroup of `image_loss_*` runs the alpha-match path
/// (`|pred.a - gt.a|`) instead of SSIM + L1 — folding the previously-separate
/// alpha-match kernel into the same launch.
pub trait LossOps<B: Backend> {
    fn image_loss_forward(
        pred: FloatTensor<B>,
        gt_packed: IntTensor<B>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<B>;

    fn image_loss_backward(
        pred: FloatTensor<B>,
        gt_packed: IntTensor<B>,
        dl_dmap: FloatTensor<B>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<B>;

    fn unpack_gt_rgb(gt_packed: IntTensor<B>, composite_bg: Option<Vec3>) -> FloatTensor<B>;
}

/// Internal companion operations for the opt-in native-MSL tape. Keeping
/// these separate avoids exposing an experimental implementation detail in
/// the public backend extension trait.
trait SavedLossOps<B: Backend> {
    fn image_loss_forward_saved(
        pred: FloatTensor<B>,
        gt_packed: IntTensor<B>,
        cfg: ImageLossConfig,
    ) -> ImageLossForwardSaved<B>;

    fn image_loss_backward_saved(
        pred: FloatTensor<B>,
        gt_packed: IntTensor<B>,
        dl_dmap: FloatTensor<B>,
        partials: FloatTensor<B>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<B>;
}

fn alloc_zeros<R: CubeRuntime>(template: &CubeTensor<R>) -> CubeTensor<R> {
    burn_cubecl::ops::numeric::zeros_client::<R>(
        template.client.clone(),
        template.device.clone(),
        Shape::from(template.shape().as_slice().to_vec()),
        template.dtype,
    )
}

fn alloc_empty<R: CubeRuntime>(
    template: &CubeTensor<R>,
    shape: Shape,
    dtype: DType,
) -> CubeTensor<R> {
    let handle = template.client.empty(shape.num_elements() * dtype.size());
    CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        shape,
        handle,
        dtype,
    )
}

/// Wraps a closure as a fusion `Operation`. Lets each fusion-side method on
/// `LossOps` skip its own `struct CustomOp` + `impl Operation` boilerplate;
/// the closure captures whatever extra config it needs.
struct ClosureOp<F> {
    desc: CustomOpIr,
    op: F,
}

impl<F> std::fmt::Debug for ClosureOp<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClosureOp({:?})", self.desc)
    }
}

impl<F> Operation<FusionCubeRuntime<WgpuRuntime>> for ClosureOp<F>
where
    F: Fn(&CustomOpIr, &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>)
        + Send
        + Sync
        + 'static,
{
    fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        (self.op)(&self.desc, h);
    }
}

/// Register a custom op on the Fusion stream. Each input/output is a fusion
/// `FusionTensor` (Float and Int both lower to the same primitive on this
/// backend), and `op` is the closure that runs against the inner backend
/// when fusion eventually executes the queued op.
fn dispatch_custom<const N: usize, F>(
    name: &'static str,
    inputs: [burn_fusion::FusionTensor<FusionCubeRuntime<WgpuRuntime>>; N],
    out_shape: Shape,
    out_dtype: DType,
    op: F,
) -> burn_fusion::FusionTensor<FusionCubeRuntime<WgpuRuntime>>
where
    F: Fn(&CustomOpIr, &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>)
        + Send
        + Sync
        + 'static,
{
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), out_shape, out_dtype);
    let stream = StreamId::current();
    let desc = CustomOpIr::new(name, &inputs.map(|t| t.into_ir()), &[out]);
    let wrapped = ClosureOp {
        desc: desc.clone(),
        op,
    };
    let [out] = client
        .register(stream, OperationIr::Custom(desc), wrapped)
        .outputs();
    out
}

fn cube_count_3d(c: u32, h: u32, w: u32) -> burn_cubecl::cubecl::prelude::CubeCount {
    use burn_cubecl::cubecl::prelude::CubeCount;
    CubeCount::Static(
        w.div_ceil(kernels::BLOCK_X),
        h.div_ceil(kernels::BLOCK_Y),
        c,
    )
}

fn cube_count_3d_bwd(c: u32, h: u32, w: u32, tile: u32) -> burn_cubecl::cubecl::prelude::CubeCount {
    use burn_cubecl::cubecl::prelude::CubeCount;
    CubeCount::Static(w.div_ceil(tile), h.div_ceil(tile), c)
}

fn select_backward_tile(
    max_shared_memory_size: usize,
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
) -> u32 {
    if max_shared_memory_size >= kernels::BWD_LARGE_SHARED_BYTES
        && max_units_per_cube >= kernels::BWD_TILE_LARGE * kernels::BWD_TILE_LARGE
        && max_cube_dim.0 >= kernels::BWD_TILE_LARGE
        && max_cube_dim.1 >= kernels::BWD_TILE_LARGE
    {
        kernels::BWD_TILE_LARGE
    } else {
        kernels::BWD_TILE_SMALL
    }
}

fn launch_image_forward<R: CubeRuntime>(
    pred: CubeTensor<R>,
    gt_packed: CubeTensor<R>,
    cfg: ImageLossConfig,
) -> CubeTensor<R> {
    launch_image_forward_impl(pred, gt_packed, cfg, false).0
}

fn launch_image_forward_saved<R: CubeRuntime>(
    pred: CubeTensor<R>,
    gt_packed: CubeTensor<R>,
    cfg: ImageLossConfig,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let (map, partials) = launch_image_forward_impl(pred, gt_packed, cfg, true);
    (
        map,
        partials.expect("saved loss forward must allocate partials"),
    )
}

fn launch_image_forward_impl<R: CubeRuntime>(
    pred: CubeTensor<R>,
    gt_packed: CubeTensor<R>,
    cfg: ImageLossConfig,
    save_partials: bool,
) -> (CubeTensor<R>, Option<CubeTensor<R>>) {
    use burn_cubecl::cubecl::prelude::CubeDim;

    let pred = into_contiguous(pred);
    let gt_packed = into_contiguous(gt_packed);
    let dims = pred.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "image_loss expects [C, H, W] pred");
    let (c, h, w) = (dims[0] as u32, dims[1] as u32, dims[2] as u32);
    assert!(matches!(c, 3 | 4), "image loss expects RGB or RGBA pred");
    let gt_dims = gt_packed.shape().as_slice().to_vec();
    assert_eq!(gt_dims.len(), 2, "image_loss expects [H, W] gt_packed");
    assert_eq!(
        gt_dims[0] as u32, h,
        "gt_packed height must match pred height"
    );
    assert_eq!(
        gt_dims[1] as u32, w,
        "gt_packed width must match pred width"
    );

    let composite = cfg.composite_bg.is_some();
    let bg = cfg.composite_bg.unwrap_or(Vec3::ZERO);
    let map = alloc_zeros(&pred);
    if cfg.ssim_weight == 0.0 && !save_partials {
        let client = pred.client.clone();
        kernels::image_l1_forward_kernel::launch::<f32, R>(
            &client,
            cube_count_3d(c, h, w),
            CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
            pred.into_tensor_arg(),
            gt_packed.into_tensor_arg(),
            map.clone().into_tensor_arg(),
            h,
            w,
            cfg.l1_weight,
            bg.x,
            bg.y,
            bg.z,
            composite,
            cfg.mask,
        );
        return (map, None);
    }
    let partials = if save_partials {
        assert!(
            matches!(c, 3 | 4),
            "saved loss partials require RGB or RGBA pred, got {c} channels"
        );
        Some(alloc_empty(
            &pred,
            Shape::new([9, h as usize, w as usize]),
            DType::F32,
        ))
    } else {
        None
    };
    let client = pred.client.clone();
    kernels::image_loss_forward_kernel::launch::<f32, R>(
        &client,
        cube_count_3d(c, h, w),
        CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
        pred.into_tensor_arg(),
        gt_packed.into_tensor_arg(),
        map.clone().into_tensor_arg(),
        partials
            .clone()
            .map(|partials| partials.into_tensor_arg())
            .into(),
        h,
        w,
        cfg.l1_weight,
        cfg.ssim_weight,
        bg.x,
        bg.y,
        bg.z,
        composite,
        cfg.mask,
    );
    (map, partials)
}

fn launch_image_backward<R: CubeRuntime>(
    pred: CubeTensor<R>,
    gt_packed: CubeTensor<R>,
    dl_dmap: CubeTensor<R>,
    cfg: ImageLossConfig,
) -> CubeTensor<R> {
    launch_image_backward_with_tile(pred, gt_packed, dl_dmap, cfg, None)
}

fn launch_image_backward_with_tile<R: CubeRuntime>(
    pred: CubeTensor<R>,
    gt_packed: CubeTensor<R>,
    dl_dmap: CubeTensor<R>,
    cfg: ImageLossConfig,
    tile_override: Option<u32>,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::CubeDim;

    let pred = into_contiguous(pred);
    let gt_packed = into_contiguous(gt_packed);
    let dl_dmap = into_contiguous(dl_dmap);
    let dims = pred.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "image_loss_backward expects [C, H, W] pred");
    let (c, h, w) = (dims[0] as u32, dims[1] as u32, dims[2] as u32);
    assert!(matches!(c, 3 | 4), "image loss expects RGB or RGBA pred");

    let composite = cfg.composite_bg.is_some();
    let bg = cfg.composite_bg.unwrap_or(Vec3::ZERO);
    let dl_dpred = alloc_zeros(&pred);
    let client = pred.client.clone();
    if cfg.ssim_weight == 0.0 {
        kernels::image_l1_backward_kernel::launch::<f32, R>(
            &client,
            cube_count_3d(c, h, w),
            CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
            pred.into_tensor_arg(),
            gt_packed.into_tensor_arg(),
            dl_dmap.into_tensor_arg(),
            dl_dpred.clone().into_tensor_arg(),
            h,
            w,
            cfg.l1_weight,
            bg.x,
            bg.y,
            bg.z,
            composite,
            cfg.mask,
        );
        return dl_dpred;
    }
    let hardware = &client.properties().hardware;
    let tile = tile_override.unwrap_or_else(|| {
        select_backward_tile(
            hardware.max_shared_memory_size,
            hardware.max_units_per_cube,
            hardware.max_cube_dim,
        )
    });
    debug_assert!(
        matches!(tile, kernels::BWD_TILE_SMALL | kernels::BWD_TILE_LARGE),
        "backward loss tile must be 8 or 16, got {tile}"
    );

    kernels::image_loss_backward_kernel::launch::<f32, R>(
        &client,
        cube_count_3d_bwd(c, h, w, tile),
        CubeDim::new_2d(tile, tile),
        pred.into_tensor_arg(),
        gt_packed.into_tensor_arg(),
        dl_dmap.into_tensor_arg(),
        None.into(),
        dl_dpred.clone().into_tensor_arg(),
        h,
        w,
        cfg.l1_weight,
        cfg.ssim_weight,
        bg.x,
        bg.y,
        bg.z,
        composite,
        cfg.mask,
        tile,
    );
    dl_dpred
}

fn launch_image_backward_saved<R: CubeRuntime>(
    pred: CubeTensor<R>,
    gt_packed: CubeTensor<R>,
    dl_dmap: CubeTensor<R>,
    partials: CubeTensor<R>,
    cfg: ImageLossConfig,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::CubeDim;

    let pred = into_contiguous(pred);
    let gt_packed = into_contiguous(gt_packed);
    let dl_dmap = into_contiguous(dl_dmap);
    let partials = into_contiguous(partials);
    let dims = pred.shape().as_slice().to_vec();
    assert_eq!(
        dims.len(),
        3,
        "image_loss_backward_saved expects [C, H, W] pred"
    );
    let (c, h, w) = (dims[0] as u32, dims[1] as u32, dims[2] as u32);
    assert!(
        matches!(c, 3 | 4),
        "saved loss partials require RGB or RGBA pred, got {c} channels"
    );
    assert_eq!(
        partials.shape().as_slice(),
        [9, h as usize, w as usize],
        "saved loss partial shape must be [9, H, W]"
    );

    let composite = cfg.composite_bg.is_some();
    let bg = cfg.composite_bg.unwrap_or(Vec3::ZERO);
    let dl_dpred = alloc_zeros(&pred);
    let client = pred.client.clone();
    let hardware = &client.properties().hardware;
    let tile = select_backward_tile(
        hardware.max_shared_memory_size,
        hardware.max_units_per_cube,
        hardware.max_cube_dim,
    );

    kernels::image_loss_backward_kernel::launch::<f32, R>(
        &client,
        cube_count_3d_bwd(c, h, w, tile),
        CubeDim::new_2d(tile, tile),
        pred.into_tensor_arg(),
        gt_packed.into_tensor_arg(),
        dl_dmap.into_tensor_arg(),
        Some(partials.into_tensor_arg()).into(),
        dl_dpred.clone().into_tensor_arg(),
        h,
        w,
        cfg.l1_weight,
        cfg.ssim_weight,
        bg.x,
        bg.y,
        bg.z,
        composite,
        cfg.mask,
        tile,
    );
    dl_dpred
}

fn launch_unpack_gt_rgb<R: CubeRuntime>(
    gt_packed: CubeTensor<R>,
    composite_bg: Option<Vec3>,
) -> CubeTensor<R> {
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};

    let gt_packed = into_contiguous(gt_packed);
    let dims = gt_packed.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 2, "unpack_gt_rgb expects [H, W] gt_packed");
    let (h, w) = (dims[0] as u32, dims[1] as u32);
    let composite = composite_bg.is_some();
    let bg = composite_bg.unwrap_or(Vec3::ZERO);

    let client = gt_packed.client.clone();
    let out = burn_cubecl::ops::numeric::zeros_client::<R>(
        client.clone(),
        gt_packed.device.clone(),
        Shape::new([h as usize, w as usize, 3]),
        DType::F32,
    );
    let cube_count = CubeCount::Static(
        w.div_ceil(kernels::BLOCK_X),
        h.div_ceil(kernels::BLOCK_Y),
        1,
    );
    kernels::unpack_gt_rgb_kernel::launch::<f32, R>(
        &client,
        cube_count,
        CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
        gt_packed.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        h,
        w,
        bg.x,
        bg.y,
        bg.z,
        composite,
    );
    out
}

impl LossOps<Self> for MainBackendBase {
    fn image_loss_forward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        launch_image_forward(pred, gt_packed, cfg)
    }

    fn image_loss_backward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        launch_image_backward(pred, gt_packed, dl_dmap, cfg)
    }

    fn unpack_gt_rgb(gt_packed: IntTensor<Self>, composite_bg: Option<Vec3>) -> FloatTensor<Self> {
        launch_unpack_gt_rgb(gt_packed, composite_bg)
    }
}

impl SavedLossOps<Self> for MainBackendBase {
    fn image_loss_forward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
    ) -> ImageLossForwardSaved<Self> {
        let (map, partials) = launch_image_forward_saved(pred, gt_packed, cfg);
        ImageLossForwardSaved { map, partials }
    }

    fn image_loss_backward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        partials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        launch_image_backward_saved(pred, gt_packed, dl_dmap, partials, cfg)
    }
}

impl LossOps<Self> for Fusion<MainBackendBase> {
    fn image_loss_forward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        let shape = pred.shape();
        dispatch_custom(
            "image_loss_forward",
            [pred, gt_packed],
            shape,
            DType::F32,
            move |desc, h| {
                let ([pred, gt_packed], [map]) = desc.as_fixed();
                let out = MainBackendBase::image_loss_forward(
                    h.get_float_tensor::<MainBackendBase>(pred),
                    h.get_int_tensor::<MainBackendBase>(gt_packed),
                    cfg,
                );
                h.register_float_tensor::<MainBackendBase>(&map.id, out);
            },
        )
    }

    fn image_loss_backward(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        let shape = pred.shape();
        dispatch_custom(
            "image_loss_backward",
            [pred, gt_packed, dl_dmap],
            shape,
            DType::F32,
            move |desc, h| {
                let ([pred, gt_packed, dl_dmap], [dl_dpred]) = desc.as_fixed();
                let out = MainBackendBase::image_loss_backward(
                    h.get_float_tensor::<MainBackendBase>(pred),
                    h.get_int_tensor::<MainBackendBase>(gt_packed),
                    h.get_float_tensor::<MainBackendBase>(dl_dmap),
                    cfg,
                );
                h.register_float_tensor::<MainBackendBase>(&dl_dpred.id, out);
            },
        )
    }

    fn unpack_gt_rgb(gt_packed: IntTensor<Self>, composite_bg: Option<Vec3>) -> FloatTensor<Self> {
        let [gh, gw] = gt_packed.shape().dims();
        dispatch_custom(
            "unpack_gt_rgb",
            [gt_packed],
            Shape::new([gh, gw, 3]),
            DType::F32,
            move |desc, h| {
                let ([gt_packed], [out]) = desc.as_fixed();
                let res = MainBackendBase::unpack_gt_rgb(
                    h.get_int_tensor::<MainBackendBase>(gt_packed),
                    composite_bg,
                );
                h.register_float_tensor::<MainBackendBase>(&out.id, res);
            },
        )
    }
}

impl SavedLossOps<Self> for Fusion<MainBackendBase> {
    fn image_loss_forward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        cfg: ImageLossConfig,
    ) -> ImageLossForwardSaved<Self> {
        let map_shape = pred.shape();
        let [_, h, w] = map_shape.dims();
        let partials_shape = Shape::new([9, h, w]);
        let client = pred.client.clone();
        let map_out = TensorIr::uninit(client.create_empty_handle(), map_shape, DType::F32);
        let partials_out =
            TensorIr::uninit(client.create_empty_handle(), partials_shape, DType::F32);
        let inputs = [pred, gt_packed];
        let desc = CustomOpIr::new(
            "image_loss_forward_saved",
            &inputs.map(|tensor| tensor.into_ir()),
            &[map_out, partials_out],
        );
        let wrapped = ClosureOp {
            desc: desc.clone(),
            op: move |desc: &CustomOpIr,
                      handles: &mut HandleContainer<
                FusionHandle<FusionCubeRuntime<WgpuRuntime>>,
            >| {
                let ([pred, gt_packed], [map, partials]) = desc.as_fixed();
                let out =
                    <MainBackendBase as SavedLossOps<MainBackendBase>>::image_loss_forward_saved(
                        handles.get_float_tensor::<MainBackendBase>(pred),
                        handles.get_int_tensor::<MainBackendBase>(gt_packed),
                        cfg,
                    );
                handles.register_float_tensor::<MainBackendBase>(&map.id, out.map);
                handles.register_float_tensor::<MainBackendBase>(&partials.id, out.partials);
            },
        };
        let [map, partials] = client
            .register(StreamId::current(), OperationIr::Custom(desc), wrapped)
            .outputs();
        ImageLossForwardSaved { map, partials }
    }

    fn image_loss_backward_saved(
        pred: FloatTensor<Self>,
        gt_packed: IntTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        partials: FloatTensor<Self>,
        cfg: ImageLossConfig,
    ) -> FloatTensor<Self> {
        let shape = pred.shape();
        dispatch_custom(
            "image_loss_backward_saved",
            [pred, gt_packed, dl_dmap, partials],
            shape,
            DType::F32,
            move |desc, h| {
                let ([pred, gt_packed, dl_dmap, partials], [dl_dpred]) = desc.as_fixed();
                let out =
                    <MainBackendBase as SavedLossOps<MainBackendBase>>::image_loss_backward_saved(
                        h.get_float_tensor::<MainBackendBase>(pred),
                        h.get_int_tensor::<MainBackendBase>(gt_packed),
                        h.get_float_tensor::<MainBackendBase>(dl_dmap),
                        h.get_float_tensor::<MainBackendBase>(partials),
                        cfg,
                    );
                h.register_float_tensor::<MainBackendBase>(&dl_dpred.id, out);
            },
        )
    }
}

#[derive(Debug)]
struct ImageLossBackward;

#[derive(Debug, Clone)]
struct ImageLossState<B: Backend> {
    pred: FloatTensor<B>,
    gt_packed: IntTensor<B>,
    saved_partials: Option<FloatTensor<B>>,
    cfg: ImageLossConfig,
}

impl<B: Backend + LossOps<B> + SavedLossOps<B>> Backward<B, 1> for ImageLossBackward {
    type State = ImageLossState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let dl_dmap = grads.consume::<B>(&ops.node);
        let [pred_parent] = ops.parents;
        let dl_dpred = if let Some(partials) = state.saved_partials {
            <B as SavedLossOps<B>>::image_loss_backward_saved(
                state.pred,
                state.gt_packed,
                dl_dmap,
                partials,
                state.cfg,
            )
        } else {
            B::image_loss_backward(state.pred, state.gt_packed, dl_dmap, state.cfg)
        };
        if let Some(node) = pred_parent {
            grads.register::<B>(node.id, dl_dpred);
        }
    }
}

/// L1 + SSIM image loss with optional bg-compositing and masking, all folded
/// into a single fused kernel. Pass `pred` with 4 channels (RGBA) to also
/// emit `|pred.a - gt.a|` into the alpha channel of the loss map; pass 3
/// (RGB) to skip the alpha-match work entirely.
///
/// `pred` must be on an autodiff-enabled Wgpu device.
pub fn image_loss(pred: Tensor<3>, gt_packed: Tensor<2, Int>, cfg: ImageLossConfig) -> Tensor<3> {
    let pred_chw = pred.permute([2, 0, 1]);
    let pred_ad = unwrap_ad_wgpu_float(pred_chw);
    let gt_p = unwrap_ad_wgpu_int(gt_packed);

    let prep = ImageLossBackward
        .prepare::<NoCheckpointing>([pred_ad.node.clone()])
        .compute_bound()
        .stateful();

    let pred_p = pred_ad.primitive;
    let map_ad: FloatTensor<AutodiffMain> = match prep {
        OpsKind::Tracked(prep) if use_saved_loss_partials() && cfg.ssim_weight != 0.0 => {
            let out = <MainBackend as SavedLossOps<MainBackend>>::image_loss_forward_saved(
                pred_p.clone(),
                gt_p.clone(),
                cfg,
            );
            prep.finish(
                ImageLossState {
                    pred: pred_p,
                    gt_packed: gt_p,
                    saved_partials: Some(out.partials),
                    cfg,
                },
                out.map,
            )
        }
        OpsKind::Tracked(prep) => {
            let map = <MainBackend as LossOps<MainBackend>>::image_loss_forward(
                pred_p.clone(),
                gt_p.clone(),
                cfg,
            );
            prep.finish(
                ImageLossState {
                    pred: pred_p,
                    gt_packed: gt_p,
                    saved_partials: None,
                    cfg,
                },
                map,
            )
        }
        OpsKind::UnTracked(prep) => {
            let map = <MainBackend as LossOps<MainBackend>>::image_loss_forward(pred_p, gt_p, cfg);
            prep.finish(map)
        }
    };
    wrap_ad_wgpu_float::<3>(map_ad).permute([1, 2, 0])
}

/// Forward-only loss map for non-differentiable backends. Same kernel as
/// the training forward; eval picks `cfg` to compute SSIM, L1, or whatever
/// combination it needs (e.g. MSE = `l1_eval(...).powi(2).mean()`).
pub fn image_loss_eval(
    pred: Tensor<3>,
    gt_packed: Tensor<2, Int>,
    cfg: ImageLossConfig,
) -> Tensor<3> {
    let pred_chw = pred.permute([2, 0, 1]);
    let pred_p = unwrap_wgpu_float(pred_chw);
    let gt_p = unwrap_wgpu_int(gt_packed);
    let map = <MainBackend as LossOps<MainBackend>>::image_loss_forward(pred_p, gt_p, cfg);
    wrap_wgpu_float::<3>(map).permute([1, 2, 0])
}

/// L1 depth loss in disparity (inverse-depth) space
pub fn depth_loss(pred_depth: Tensor<2>, gt_depth: Tensor<2>) -> Tensor<1> {
    let pred_invalid = pred_depth.clone().lower_equal_elem(0.0);
    let disp_pred = pred_depth.recip().mask_fill(pred_invalid, 0.0);

    let gt_valid = gt_depth.clone().greater_elem(0.0);
    let gt_invalid = gt_depth.clone().lower_equal_elem(0.0);
    let disp_gt = gt_depth.recip().mask_fill(gt_invalid, 0.0);

    let valid = gt_valid.float();
    let abs_err = (disp_pred - disp_gt).abs() * valid.clone();

    abs_err.sum() / valid.sum().clamp_min(1.0)
}

/// Decode `gt_packed` back to a `[H, W, 3]` f32 RGB tensor. `composite_bg =
/// Some(bg)` folds in `gt + (1 - gt.a) * bg`; `None` skips that math.
/// Materialising f32 GT defeats the whole point of the packed format, so
/// this is reserved for the LPIPS path which feeds f32 RGB into a VGG
/// forward and has no kernel-fused alternative today.
pub fn unpack_gt_rgb(gt_packed: Tensor<2, Int>, composite_bg: Option<Vec3>) -> Tensor<3> {
    let gt_p = unwrap_wgpu_int(gt_packed);
    let out = <MainBackend as LossOps<MainBackend>>::unpack_gt_rgb(gt_p, composite_bg);
    wrap_wgpu_float(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_tile_selection_respects_device_limits() {
        let large = kernels::BWD_TILE_LARGE;
        let generous_dims = (large, large, 1);
        let generous_units = large * large;

        assert_eq!(kernels::BWD_LARGE_SHARED_BYTES, 29_088);
        assert_eq!(
            select_backward_tile(29_087, generous_units, generous_dims),
            kernels::BWD_TILE_SMALL
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units, generous_dims),
            large
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units - 1, generous_dims),
            kernels::BWD_TILE_SMALL
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units, (large - 1, large, 1)),
            kernels::BWD_TILE_SMALL
        );
        assert_eq!(
            select_backward_tile(29_088, generous_units, (large, large - 1, 1)),
            kernels::BWD_TILE_SMALL
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn backward_tile_specializations_match() {
        use brush_cube::{CubeTensor, create_tensor_from_slice};
        use burn::{backend::wgpu::WgpuDevice, tensor::Shape};

        fn shaped_f32(data: &[f32], shape: Shape, device: &WgpuDevice) -> CubeTensor<WgpuRuntime> {
            let flat = create_tensor_from_slice(data, device, DType::F32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        fn shaped_i32(data: &[i32], shape: Shape, device: &WgpuDevice) -> CubeTensor<WgpuRuntime> {
            let flat = create_tensor_from_slice(data, device, DType::I32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        let device = brush_cube::test_helpers::test_device().await;
        let (c, h, w) = (4usize, 17usize, 19usize);
        let pred: Vec<f32> = (0..c * h * w)
            .map(|i| 0.1 + ((i * 17 + 3) % 71) as f32 / 100.0)
            .collect();
        let chain: Vec<f32> = (0..c * h * w)
            .map(|i| {
                let value = 0.2 + ((i * 13 + 5) % 37) as f32 / 50.0;
                if i % 2 == 0 { value } else { -value }
            })
            .collect();
        let gt: Vec<i32> = (0..h * w)
            .map(|i| {
                let r = (30 + (i * 7) % 101) as u32;
                let g = (70 + (i * 11) % 101) as u32;
                let b = (110 + (i * 13) % 101) as u32;
                let a = (100 + (i * 17) % 131) as u32;
                (r | g << 8 | b << 16 | a << 24) as i32
            })
            .collect();
        let cfg = ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: Some(Vec3::new(0.05, 0.1, 0.15)),
            mask: true,
        };

        let make_pred = || shaped_f32(&pred, Shape::new([c, h, w]), &device);
        let make_gt = || shaped_i32(&gt, Shape::new([h, w]), &device);
        let make_chain = || shaped_f32(&chain, Shape::new([c, h, w]), &device);
        let small_pred = make_pred();
        let selected_tile = {
            let hardware = &small_pred.client.properties().hardware;
            select_backward_tile(
                hardware.max_shared_memory_size,
                hardware.max_units_per_cube,
                hardware.max_cube_dim,
            )
        };
        let small = launch_image_backward_with_tile(
            small_pred,
            make_gt(),
            make_chain(),
            cfg,
            Some(kernels::BWD_TILE_SMALL),
        );
        let small: Vec<f32> = burn_cubecl::ops::into_data_sync(small)
            .to_vec()
            .expect("small-tile gradient data");
        assert!(
            small.iter().all(|value| value.is_finite()),
            "small-tile gradients must be finite"
        );
        if selected_tile != kernels::BWD_TILE_LARGE {
            return;
        }

        let large = launch_image_backward_with_tile(
            make_pred(),
            make_gt(),
            make_chain(),
            cfg,
            Some(kernels::BWD_TILE_LARGE),
        );
        let large: Vec<f32> = burn_cubecl::ops::into_data_sync(large)
            .to_vec()
            .expect("large-tile gradient data");

        for (index, (&small, &large)) in small.iter().zip(&large).enumerate() {
            let tolerance = 5e-5 + 5e-5 * small.abs().max(large.abs());
            assert!(
                (small - large).abs() <= tolerance,
                "tile gradients differ at {index}: 8x8={small}, 16x16={large}, tolerance={tolerance}"
            );
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn saved_partials_match_recomputed_forward_and_vjp() {
        use brush_cube::{CubeTensor, create_tensor_from_slice};
        use burn::{backend::wgpu::WgpuDevice, tensor::Shape};

        fn shaped_f32(data: &[f32], shape: Shape, device: &WgpuDevice) -> CubeTensor<WgpuRuntime> {
            let flat = create_tensor_from_slice(data, device, DType::F32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        fn shaped_i32(data: &[i32], shape: Shape, device: &WgpuDevice) -> CubeTensor<WgpuRuntime> {
            let flat = create_tensor_from_slice(data, device, DType::I32);
            CubeTensor::new_contiguous(flat.client, flat.device, shape, flat.handle, flat.dtype)
        }

        fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
            assert_eq!(expected.len(), actual.len(), "{label} length");
            for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 5e-5 + 5e-5 * expected.abs().max(actual.abs());
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{label} differs at {index}: expected={expected}, actual={actual}, tolerance={tolerance}"
                );
            }
        }

        let device = brush_cube::test_helpers::test_device().await;
        let (c, h, w) = (4usize, 17usize, 19usize);
        let pred_data: Vec<f32> = (0..c * h * w)
            .map(|i| 0.05 + ((i * 17 + 3) % 83) as f32 / 100.0)
            .collect();
        let chain_data: Vec<f32> = (0..c * h * w)
            .map(|i| {
                let value = 0.15 + ((i * 13 + 5) % 41) as f32 / 50.0;
                if i % 2 == 0 { value } else { -value }
            })
            .collect();
        let gt_data: Vec<i32> = (0..h * w)
            .map(|i| {
                let r = (20 + (i * 7) % 151) as u32;
                let g = (50 + (i * 11) % 151) as u32;
                let b = (90 + (i * 13) % 151) as u32;
                let alpha_values = [0u32, 1, 127, 254, 255];
                let a = alpha_values[i % alpha_values.len()];
                (r | g << 8 | b << 16 | a << 24) as i32
            })
            .collect();
        let cfg = ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: Some(Vec3::new(0.05, 0.1, 0.15)),
            mask: true,
        };
        let make_pred = || shaped_f32(&pred_data, Shape::new([c, h, w]), &device);
        let make_gt = || shaped_i32(&gt_data, Shape::new([h, w]), &device);
        let make_chain = || shaped_f32(&chain_data, Shape::new([c, h, w]), &device);

        let control_map = launch_image_forward(make_pred(), make_gt(), cfg);
        let (saved_map, partials) = launch_image_forward_saved(make_pred(), make_gt(), cfg);
        let control_grad = launch_image_backward(make_pred(), make_gt(), make_chain(), cfg);
        let saved_grad =
            launch_image_backward_saved(make_pred(), make_gt(), make_chain(), partials, cfg);

        let control_map: Vec<f32> = burn_cubecl::ops::into_data_sync(control_map)
            .to_vec()
            .expect("control map data");
        let saved_map: Vec<f32> = burn_cubecl::ops::into_data_sync(saved_map)
            .to_vec()
            .expect("saved map data");
        let control_grad: Vec<f32> = burn_cubecl::ops::into_data_sync(control_grad)
            .to_vec()
            .expect("control gradient data");
        let saved_grad: Vec<f32> = burn_cubecl::ops::into_data_sync(saved_grad)
            .to_vec()
            .expect("saved gradient data");

        assert_close("forward map", &control_map, &saved_map);
        assert_close("prediction VJP", &control_grad, &saved_grad);
        let alpha_offset = 3 * h * w;
        assert_eq!(
            &control_grad[alpha_offset..],
            &saved_grad[alpha_offset..],
            "alpha VJP must remain bit-identical"
        );
    }
}
