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
    tensor::{DType, Int, Shape, Tensor, s},
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

/// Space in which the depth residual is measured (`--depth-loss-space`).
///
/// Both variants are an L1 masked mean over `gt > 0` with the same denominator
/// and the same non-finite discipline; they differ only in the residual.
///
/// ```text
///   Disparity : |1/pred - 1/gt|   d/d(pred) = ∓1/pred²   (default, previous behaviour)
///   Metric    : |pred - gt|       d/d(pred) = ±1
/// ```
///
/// # Why this is a knob and not a constant
///
/// The disparity gradient scales as `1/d²`, so a splat one metre from the
/// camera receives a hundred times the depth gradient of one ten metres away.
/// With the plane-fused depth source (`--depth-source plane-fused`), whose
/// blending-weight gradients are live by design, that near-field pressure has a
/// second outlet: *fading a splat out* lowers the depth term more cheaply than
/// *rotating it*, which is the opacity collapse measured on both ablation
/// scenes (plan §10e/§10f: opacity p50 −28% / −34%). A metric L1 never creates
/// that pressure — its per-pixel gradient magnitude is a constant `1`,
/// independent of range — and it is what the `gauss-surf` PGSR reference
/// (rerun-io/examples-monorepo, Apache-2.0, by Pablo Vela) trains with, at
/// `3.2 / scene_scale`, while running fused. So `Metric` is the arm that
/// separates "the technique is harmful" from "our loss made it harmful".
///
/// # Scale-normalization semantics FLIP with this flag
///
/// `--normalize-metric-weights` divides the metric-dimensioned loss weights by
/// the scene scale. The depth weight is deliberately EXCLUDED from that set
/// under `Disparity` — the residual is in `1/m`, so dividing by `s` moves the
/// effective weight the wrong way by `s²` (plan §4.8). Under `Metric` the
/// residual is in metres like flatten's, so the depth weight JOINS the
/// normalized set and is divided by `s`, matching the reference's
/// `3.2 / scene_scale`. See `TrainConfig::depth_loss_space` and the divisor at
/// the consumption site in `brush-train`'s `train.rs`.
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
    clap::ValueEnum,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum DepthLossSpace {
    /// `|1/pred - 1/gt|`. Previous behaviour, byte-identical.
    #[default]
    Disparity,
    /// `|pred - gt|`, in the GT's own units (metres for our depth priors).
    Metric,
}

/// How pixels the render does not COVER are handled by [`depth_loss`]
/// (`--depth-uncovered`).
///
/// The center depth source composites `accumulated_depth / α.clamp_min(1e-10)`,
/// which is **exactly 0** where nothing was rendered. Such a pixel may still
/// have a perfectly valid GT depth behind it, and the legacy mask is GT-only —
/// so it enters the numerator as a full-magnitude residual (`|0 − 1/D_gt|`,
/// i.e. 2.0 m⁻¹ for a 0.5 m prior) and is counted in the denominator. That is
/// an unreducible floor in the *reported* loss plus a dilution of everything
/// else in it, proportional to the uncovered fraction of the frame.
///
/// ```text
///   Count            numerator: kept      denominator: kept    (default, legacy)
///   ExcludeNumerator numerator: dropped   denominator: kept
///   Exclude          numerator: dropped   denominator: dropped (LFS semantics)
/// ```
///
/// # Why three modes and not two
///
/// `Exclude` is the semantically right answer — a pixel with no prediction
/// carries no information about the geometry, so it belongs in neither sum, and
/// it is what `LichtFeld Studio`'s `depth_loss.cu` does (its `pixel_active`
/// predicate skips inactive pixels from *every* sum). But it does not merely
/// clean up the report: dropping the pixel from the DENOMINATOR rescales every
/// surviving pixel's gradient by a per-frame factor
/// `N_gt-valid / N_(covered ∧ gt-valid)`, which is a real change to the
/// effective depth-supervision weight and lands hardest exactly where coverage
/// is lowest (early training, frame edges).
///
/// `ExcludeNumerator` is the same coverage test with the denominator left
/// alone. In the DEFAULT disparity space it is **gradient-identical** to
/// `Count` (see the caveat below), so it changes the reported number and
/// nothing else. It exists so that any metric movement observed under
/// `Exclude` has exactly one candidate cause — the denominator rescale — rather
/// than two.
///
/// # The gradient-identity claim is SPACE-DEPENDENT — read this before using it
///
/// Under [`DepthLossSpace::Disparity`] the excluded terms are constants in
/// every lane that carries a finite gradient: an uncovered pixel has
/// `pred <= 0`, so `disp_pred` comes out of
/// `recip().mask_fill(pred_invalid, 0.0)` whose VJP zeroes that lane, and the
/// residual `|0 − 1/D_gt|` contributes to the value but not to the derivative
/// of any covered pixel. Verified by
/// `exclude_numerator_preserves_every_finite_disparity_gradient`.
///
/// **With one measured correction to the plan** (2026-08-22): plan §2.1 states
/// that such a pixel "carries no gradient". It actually carries a `NaN` —
/// `mask_fill` zeroes the gradient arriving at `recip`'s OUTPUT, but `recip`'s
/// own backward is `-grad · (1/pred)²`, which at `pred == 0` is `0 · ∞`. It is
/// a LATENT defect rather than a live one, and `Count` keeps it unchanged for
/// byte-identity: an uncovered pixel is by definition one no gaussian
/// contributed to, so the rasterize backward has nothing to scatter it into and
/// the `NaN` dies at the image boundary of the graph — measured, not assumed,
/// by `brush-train`'s `depth_loss_does_not_touch_opacity`, which renders 4
/// splats on a 48x48 frame against a dense GT and passes `bwd_validate`. Both
/// exclude modes replace it with an honest `0` as a side effect of substituting
/// the prediction before the arithmetic.
///
/// Under [`DepthLossSpace::Metric`] it is **NOT**. There is no `pred <= 0`
/// guard there by design (a non-positive prediction is a legitimate finite
/// residual, not a singularity), so an uncovered pixel scores `|0 − D_gt|` with
/// a live `∓1` gradient — and since the center path's prediction is
/// `accumulated_depth / α.clamp_min(1e-10)`, the chain rule multiplies that by
/// up to `1e10` at a pixel with no coverage. So in metric space
/// `ExcludeNumerator` is a genuine gradient change (a large one, and in the
/// safe direction), not a reporting-only change. This was found while
/// implementing plan §5 and is recorded there; the plan's flat
/// "gradient-identical" claim holds for the default space only.
/// Pinned by `exclude_numerator_changes_metric_space_gradients`.
///
/// # Coverage test
///
/// `!(pred > 0)`, spelled as the COMPLEMENT of the positivity test rather than
/// as `pred <= 0`, for exactly the reason the `gt_invalid = !gt_valid` comment
/// inside [`depth_loss`] gives: every comparison against `NaN` is false, so the
/// two spellings differ on precisely one value and only the complement form
/// substitutes a `NaN` prediction instead of letting it ride into the
/// arithmetic. Byte-identical for all-finite predictions.
///
/// The stricter LFS-style test is `alpha > 1e-3`, which additionally drops
/// barely-covered pixels whose normalised depth is noise-dominated; it needs
/// the (detached) alpha plumbed down to this function and is deliberately NOT
/// taken here — plan §5.2 defers it to the near-field-floor work, which is
/// where barely-covered pixels would first misbehave.
///
/// # Interaction with the plane depth sources
///
/// None, by construction. Both plane paths already multiply their GT by the
/// plane-validity mask at the dispatch site in `brush-train`'s `train.rs`, so a
/// plane-invalid pixel has `gt == 0` and leaves through `gt_valid` before this
/// mask is consulted. The composition is a no-op there in all three modes,
/// pinned by `pre_masked_plane_style_gt_is_mode_invariant`.
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
    clap::ValueEnum,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum DepthUncovered {
    /// Uncovered pixels stay in BOTH sums. Previous behaviour, byte-identical.
    #[default]
    Count,
    /// Uncovered pixels leave the numerator only. In disparity space this
    /// preserves every finite gradient bit for bit (and replaces the uncovered
    /// lanes' latent `NaN` with 0), so it is a reporting-only change there.
    ExcludeNumerator,
    /// Uncovered pixels leave the numerator AND the denominator (LFS
    /// semantics). Rescales surviving gradients by `N_gt / N_covered`.
    Exclude,
}

/// L1 depth loss, in disparity (inverse-depth) space by default or in metric
/// space under [`DepthLossSpace::Metric`]. Masked to `gt > 0`; the mean is over
/// the UNWEIGHTED valid count so an optional `pixel_weight` modulates the error
/// map without moving the denominator.
///
/// `uncovered` selects what happens to pixels the render does not cover (the
/// center depth source composites those as exactly 0) — see [`DepthUncovered`].
/// The default [`DepthUncovered::Count`] constructs no extra op at all and is
/// byte-identical to the pre-flag function.
pub fn depth_loss(
    pred_depth: Tensor<2>,
    gt_depth: Tensor<2>,
    pixel_weight: Option<Tensor<2>>,
    space: DepthLossSpace,
    uncovered: DepthUncovered,
) -> Tensor<1> {
    let gt_valid = gt_depth.clone().greater_elem(0.0);
    // `!(gt > 0)`, NOT `gt <= 0` — and the difference is exactly one value.
    // Every comparison against `NaN` is false, so a `NaN` GT is neither
    // `> 0` (not supervised) nor `<= 0` (not substituted): under the old
    // spelling it fell through BOTH tests, reached the arithmetic intact and
    // met `* valid` as `NaN · 0 = NaN`, poisoning the frame — the same
    // `0 · ∞` shape the discipline below exists to prevent, entered from the
    // GT side instead of the prediction side. Complementing the validity mask
    // closes it by construction: whatever is not supervised is substituted.
    //
    // **Byte-identical for all-finite GT** (for which `!(x > 0)` and `x <= 0`
    // agree elementwise), so the recorded identity hashes are unaffected. This
    // repairs the DISPARITY path too, which had the same hole and no test that
    // poisoned a GT pixel to find it — the pre-existing poison test perturbs
    // the prediction and the per-pixel weight only.
    //
    // Note `+inf` is deliberately NOT covered here: `inf > 0` is true, so an
    // infinite GT depth is a SUPERVISED pixel under this function's stated
    // contract, not a masked-out one. Rejecting it belongs to the loader.
    let gt_invalid = gt_valid.clone().bool_not();

    // ---- COVERAGE (`--depth-uncovered`) ----------------------------------
    //
    // A pixel the render does not cover carries no information about the
    // geometry, but the mask above is GT-only, so under `Count` it scores a
    // full-magnitude residual against the prior and is counted in the mean.
    // See [`DepthUncovered`] for the full argument, including why
    // `ExcludeNumerator` preserves every finite gradient in DISPARITY space but
    // not in metric space, and why the uncovered lanes' `Count`-mode VJP is a
    // `NaN` rather than the 0 the plan predicted.
    //
    // Spelled `!(pred > 0)` and not `pred <= 0` for the same one-value reason
    // as `gt_invalid` above: `NaN` fails both comparisons, so only the
    // complement form substitutes it. Byte-identical for finite predictions.
    //
    // `Count` takes the `None` arm and constructs NO tensor op, so the default
    // path is the pre-flag graph node for node.
    let uncovered_mask = match uncovered {
        DepthUncovered::Count => None,
        DepthUncovered::ExcludeNumerator | DepthUncovered::Exclude => {
            Some(pred_depth.clone().greater_elem(0.0).bool_not())
        }
    };

    // The substitution mask for the NUMERATOR. Everything downstream that used
    // to substitute on `gt_invalid` now substitutes on this instead, which is
    // the same object under `Count`.
    //
    // Folding coverage in HERE rather than adding a second `mask_fill` later is
    // deliberate and load-bearing for the non-finite discipline below: the
    // disparity arm's `disp_gt = gt.recip()` would otherwise be substituted on
    // `gt_invalid` only, so an uncovered-but-GT-valid pixel would keep a live
    // `1/D_gt` in the numerator while its `disp_pred` had been zeroed — a
    // residual of `-1/D_gt`, i.e. exactly the term we set out to remove.
    let num_invalid = match &uncovered_mask {
        None => gt_invalid,
        Some(u) => gt_invalid.bool_or(u.clone()),
    };
    let num_invalid_w = num_invalid.clone();

    // NON-FINITE DISCIPLINE — this `mask_fill` must stay BEFORE the arithmetic.
    // A masked-out pixel has to contribute exactly nothing, and `x * 0.0` does
    // not deliver that when `x` is `inf` or `NaN`: `0 · ∞ = NaN`, and autodiff
    // reproduces the same product in the VJP even if the forward is repaired
    // afterwards. Substituting the value up front is the only spelling that
    // works, and it is the same idiom `plane_depth_from_features` uses for the
    // identical reason. Note `NaN` in particular defeats the `pred <= 0` guard
    // below all on its own, because every comparison against `NaN` is false.
    //
    // **Byte-identical for all-finite inputs**, which is what makes it safe
    // against the recorded identity hashes: at a masked-out pixel the old code
    // computed some finite `|disp_pred - 0|` and multiplied it by `valid == 0`,
    // giving `0`; the new code computes `|0 - 0| * 0`, also `0`. Valid pixels
    // are not touched at all (`mask_fill` is a select, so their bits pass
    // through unchanged), and its VJP is an elementwise `grad * !mask` — no
    // reduction, hence no reassociation that could move a last bit.
    let pred_depth = pred_depth.mask_fill(num_invalid.clone(), 0.0);

    // The two spaces share every line above and below this match; only the
    // residual differs. Keeping them in ONE function rather than two is
    // deliberate — the masking, the denominator and the non-finite discipline
    // are the parts that are easy to get subtly different, and a sibling
    // function would have to re-derive all three.
    let residual = match space {
        DepthLossSpace::Disparity => {
            let pred_invalid = pred_depth.clone().lower_equal_elem(0.0);
            let disp_pred = pred_depth.recip().mask_fill(pred_invalid, 0.0);

            let disp_gt = gt_depth.recip().mask_fill(num_invalid, 0.0);

            disp_pred - disp_gt
        }
        DepthLossSpace::Metric => {
            // Same NON-FINITE DISCIPLINE as the `pred_depth` line above, and it
            // is load-bearing on THIS side too: `pred` was already substituted,
            // but a `NaN`/`±inf` sitting in the GT outside the mask would ride
            // straight through the subtraction (`0 - NaN = NaN`) and then meet
            // `* 0.0` in the multiply below, which is exactly the `0 · ∞` shape
            // the discipline exists to avoid. The disparity arm gets this for
            // free because it already substitutes on the `recip()` result.
            //
            // No `pred <= 0` guard here: in metric space a negative or zero
            // prediction is a legitimate, finite residual rather than a
            // singularity, and the plane depth sources already zero the GT at
            // their invalid pixels (see the dispatch site in `train.rs`), so
            // those pixels leave through the mask rather than through a guard.
            pred_depth - gt_depth.mask_fill(num_invalid, 0.0)
        }
    };

    // The DENOMINATOR. `Count` and `ExcludeNumerator` share the legacy GT-only
    // count; only `Exclude` narrows it, and it narrows it to exactly the
    // complement of the numerator mask — i.e. `gt_valid ∧ covered` — so the two
    // sums are over the same pixel set, which is what makes the result an
    // honest mean rather than a mean with a hole in it.
    let valid = match uncovered {
        DepthUncovered::Count | DepthUncovered::ExcludeNumerator => gt_valid.float(),
        DepthUncovered::Exclude => num_invalid_w.clone().bool_not().float(),
    };
    let abs_err = residual.abs() * valid.clone();

    // DN-Splatter semantics: per-pixel modulation of the error map; the
    // denominator stays the UNweighted valid count, so w == 1 (and None) is
    // byte-identical to the old fn.
    let abs_err = match pixel_weight {
        // Same discipline for the weight: it is caller-supplied (today
        // `rgb_grad_weight`, always finite), but "the caller is careful" is not
        // a property the type system carries, and `abs_err` is already 0 here,
        // so a non-finite weight outside the mask would re-poison a term that
        // was just made safe. Byte-identical for finite weights, and the
        // `None` arm — the default path — constructs no op at all.
        Some(w) => abs_err * w.mask_fill(num_invalid_w, 0.0),
        None => abs_err,
    };

    abs_err.sum() / valid.sum().clamp_min(1.0)
}

/// Per-pixel depth-loss weight `exp(-|grad I| / sigma)` from a GT RGB image
/// `[H, W, 3]` in [0, 1]. |grad I| is the channel-mean L1 forward difference,
/// `(sum_c |I[y,x+1]-I[y,x]| + sum_c |I[y+1,x]-I[y,x]|) / 3`. The forward-
/// difference border (last row/col) reads gradient 0, i.e. full weight.
pub fn rgb_grad_weight(gt_rgb: Tensor<3>, sigma: f32) -> Tensor<2> {
    let [h, w, _] = gt_rgb.dims();
    let device = gt_rgb.device();
    if h < 2 || w < 2 {
        return Tensor::ones([h, w], &device);
    }

    let dx = (gt_rgb.clone().slice(s![.., 1..w, ..]) - gt_rgb.clone().slice(s![.., 0..w - 1, ..]))
        .abs()
        .sum_dim(2)
        .div_scalar(3.0); // [H, W-1, 1]
    let dy = (gt_rgb.clone().slice(s![1..h, .., ..]) - gt_rgb.slice(s![0..h - 1, .., ..]))
        .abs()
        .sum_dim(2)
        .div_scalar(3.0); // [H-1, W, 1]

    let gx = Tensor::<2>::zeros([h, w], &device)
        .slice_assign(s![.., 0..w - 1], dx.reshape([h as i32, (w - 1) as i32]));
    let gy = Tensor::<2>::zeros([h, w], &device)
        .slice_assign(s![0..h - 1, ..], dy.reshape([(h - 1) as i32, w as i32]));

    (gx + gy).mul_scalar(-1.0 / sigma).exp()
}

/// L1 loss between a rendered camera-frame normal image and an external normal
/// prior, both `[H, W, 3]`.
///
/// Validity follows the prior: a pixel counts when `|gt| > 0.5`, i.e. the writer
/// stored a unit normal there. `(0, 0, 0)` marks "no prior" and is skipped, the
/// same contract `depth_loss` uses for `gt <= 0`. The mean is over valid
/// pixels × 3 channels, with the denominator clamped so an all-invalid frame
/// yields 0 rather than NaN.
///
/// L1 (not `1 - cos`) matches DN-Splatter's default normal loss.
///
/// `gate_cos` is the optional NeuRIS-style per-pixel contradiction gate
/// (arXiv:2206.13597), as a COSINE threshold (the caller converts from degrees;
/// `TrainConfig::normal_gate_cos_at` does this). `None` is the pre-gate code
/// path exactly — no extra tensor op is constructed at all.
///
/// When set, a pixel additionally requires
/// `dot(normalize(pred.detach()), normalize(gt.detach())) >= gate_cos`. **Both
/// operands are detached**: this is a mask on which pixels are supervised, not a
/// second gradient path into the rendered normals. Locally contradicted pixels
/// — transients, reflections, prior-model failures inside an otherwise-good
/// frame — drop out.
///
/// The denominator is the GATED valid count, so surviving pixels keep full
/// per-pixel magnitude. An unrenormalized denominator would silently anneal the
/// entire term as the gate tightens, conflating gate strength with the weight
/// schedule. An empty mask still yields a differentiable exact 0 via the
/// `clamp_min(1.0)`, matching the reference's `sum / max(count, 1)` discipline.
pub fn normal_loss(
    pred_normal: Tensor<3>,
    gt_normal: Tensor<3>,
    gate_cos: Option<f32>,
) -> Tensor<1> {
    let mut valid = normal_prior_valid_mask(gt_normal.clone());

    if let Some(gate_cos) = gate_cos {
        valid = valid * normal_gate_mask(pred_normal.clone(), gt_normal.clone(), gate_cos);
    }

    // NON-FINITE DISCIPLINE: substitute before the subtract, never rely on the
    // `* valid` to erase a non-finite value — see the block comment in
    // `depth_loss` for why `0 · ∞` makes the multiply-only spelling wrong, and
    // why this is byte-identical on all-finite inputs. Both operands are
    // sanitised: a `NaN` in the PRIOR fails `|gt| > 0.5` (so the pixel is
    // already invalid) but would still poison `pred - gt`.
    let invalid3 = valid.clone().lower_elem(0.5).repeat_dim(2, 3);
    let pred_normal = pred_normal.mask_fill(invalid3.clone(), 0.0);
    let gt_normal = gt_normal.mask_fill(invalid3, 0.0);

    let abs_err = (pred_normal - gt_normal).abs().sum_dim(2) * valid.clone();

    abs_err.sum() / valid.sum().mul_scalar(3.0).clamp_min(1.0)
}

/// Prior-validity mask, `[H, W, 1]`: 1.0 where the prior writer stored a unit
/// normal, 0.0 at the `(0, 0, 0)` "no prior" sentinel.
fn normal_prior_valid_mask(gt_normal: Tensor<3>) -> Tensor<3> {
    gt_normal
        .powi_scalar(2)
        .sum_dim(2)
        .sqrt()
        .greater_elem(0.5)
        .float()
}

/// The contradiction-gate mask, `[H, W, 1]`: 1.0 where the rendered and prior
/// normals agree to within `gate_cos`.
///
/// **Detached on both sides.** This decides WHICH pixels are supervised; it is
/// never a second gradient path into the rendered normals. Lengths are clamped
/// away from zero because invalid `(0, 0, 0)` prior pixels are still present
/// here — they are removed by the validity mask regardless, and the clamp keeps
/// the division finite so no NaN can reach the multiply (the 0·∞ lesson from
/// `normals_from_depth`).
fn normal_gate_mask(pred_normal: Tensor<3>, gt_normal: Tensor<3>, gate_cos: f32) -> Tensor<3> {
    let pred_d = pred_normal.detach();
    let gt_d = gt_normal.detach();
    let pred_len = pred_d
        .clone()
        .powi_scalar(2)
        .sum_dim(2)
        .sqrt()
        .clamp_min(1e-6);
    let gt_len = gt_d
        .clone()
        .powi_scalar(2)
        .sum_dim(2)
        .sqrt()
        .clamp_min(1e-6);
    let cos = (pred_d * gt_d).sum_dim(2) / (pred_len * gt_len);
    cos.greater_equal_elem(gate_cos).float()
}

/// Diagnostic counts for the contradiction gate, as a 2-element tensor
/// `[surviving, valid]`.
///
/// `valid` is the number of pixels carrying a usable prior; `surviving` is how
/// many of those the gate kept. **Two counts rather than a ready-made fraction**
/// so the caller can tell "the gate masked almost everything" (the failure this
/// exists to surface) apart from "this frame had almost no prior to begin with"
/// (which says nothing about the gate). A single ratio collapses those into the
/// same number.
///
/// Fully detached and off the autodiff tape: purely observational. Only
/// meaningful when the gate is on, and only built on the steps the trainer
/// actually samples — see `SplatTrainer::should_sample_normal_gate`.
pub fn normal_gate_counts(
    pred_normal: Tensor<3>,
    gt_normal: Tensor<3>,
    gate_cos: f32,
) -> Tensor<1> {
    let valid = normal_prior_valid_mask(gt_normal.clone());
    let surviving = valid.clone() * normal_gate_mask(pred_normal, gt_normal, gate_cos);
    Tensor::cat(vec![surviving.sum(), valid.sum()], 0).detach()
}

/// Surface normals derived from a depth map by unprojecting to camera-frame
/// points and taking finite differences, `[H, W]` -> `[H, W, 3]`.
///
/// `P(u, v) = z * ((u - cx) / fx, (v - cy) / fy, 1)` in the `OpenCV` camera frame
/// (+X right, +Y down, +Z forward); the normal is
/// `normalize(dP/dv × dP/du)`, whose sign is camera-facing (`n.z <= 0`) for any
/// depth graph — no data-dependent flip is needed. Forward differences, so the
/// LAST row and column are invalid and emit `(0, 0, 0)`, as does any pixel whose
/// three contributing depths are not all positive.
///
/// Differentiable through `depth`; the intrinsics are constants.
pub fn normals_from_depth(depth: Tensor<2>, fx: f32, fy: f32, cx: f32, cy: f32) -> Tensor<3> {
    let [h, w] = depth.dims();
    let device = depth.device();

    if h < 2 || w < 2 {
        return Tensor::zeros([h, w, 3], &device);
    }

    // Pixel-centre grids. Built host-side (no `arange` in this burn rev) and
    // broadcast against the depth map.
    let us: Vec<f32> = (0..w).map(|u| (u as f32 - cx) / fx).collect();
    let vs: Vec<f32> = (0..h).map(|v| (v as f32 - cy) / fy).collect();
    let a_u: Tensor<2> = Tensor::<1>::from_floats(us.as_slice(), &device).reshape([1, w]);
    let b_v: Tensor<2> = Tensor::<1>::from_floats(vs.as_slice(), &device).reshape([h, 1]);

    let px = depth.clone() * a_u;
    let py = depth.clone() * b_v;
    let pz = depth.clone();
    let p: Tensor<3> = Tensor::cat(
        vec![
            px.reshape([h, w, 1]),
            py.reshape([h, w, 1]),
            pz.reshape([h, w, 1]),
        ],
        2,
    );

    let base = p.clone().slice(s![0..h - 1, 0..w - 1, ..]);
    let du = p.clone().slice(s![0..h - 1, 1..w, ..]) - base.clone();
    let dv = p.slice(s![1..h, 0..w - 1, ..]) - base;

    let comp = |t: &Tensor<3>, c: usize| t.clone().slice(s![.., .., c..c + 1]);
    let (dux, duy, duz) = (comp(&du, 0), comp(&du, 1), comp(&du, 2));
    let (dvx, dvy, dvz) = (comp(&dv, 0), comp(&dv, 1), comp(&dv, 2));

    // dP/dv × dP/du: already camera-facing for a depth graph (a fronto-parallel
    // plane gives exactly (0, 0, -1)).
    let cx_ = dvy.clone() * duz.clone() - dvz.clone() * duy.clone();
    let cy_ = dvz * dux.clone() - dvx.clone() * duz;
    let cz = dvx * duy - dvy * dux;
    let cross: Tensor<3> = Tensor::cat(vec![cx_, cy_, cz], 2);

    // Safe norm: clamp the squared length off zero BEFORE the sqrt. A
    // degenerate (all-zero) cross product — every background pixel, where the
    // render's depth is exactly 0 — has sum_sq == 0, and sqrt has an infinite
    // local derivative there. Autodiff then evaluates 0 * inf = NaN for the
    // (masked, zero-weight) gradient and poisons the whole map, which flows
    // through the render depth backward into gradient_transforms. Clamping
    // sum_sq keeps the sqrt derivative finite; valid pixels (sum_sq >> floor)
    // pass through clamp_min unchanged, so their normal and gradient stay exact.
    let len = cross
        .clone()
        .powi_scalar(2)
        .sum_dim(2)
        .clamp_min(1e-24)
        .sqrt();
    let normal = cross / len.clone().clamp_min(1e-12);

    // A degenerate (zero-length) cross product carries no orientation.
    let finite = len.greater_elem(1e-12).float();
    // All three contributing depths must be real measurements.
    let d_pos = depth.greater_elem(0.0).float().reshape([h, w, 1]);
    let valid = d_pos.clone().slice(s![0..h - 1, 0..w - 1, ..])
        * d_pos.clone().slice(s![0..h - 1, 1..w, ..])
        * d_pos.slice(s![1..h, 0..w - 1, ..])
        * finite;

    let interior = normal * valid;

    Tensor::zeros([h, w, 3], &device).slice_assign(s![0..h - 1, 0..w - 1, ..], interior)
}

/// Per-pixel **unbiased surface depth** by intersecting each camera ray with the
/// alpha-composited tangent plane of the gaussians covering that pixel.
///
/// PGSR (Chen et al. 2024, arXiv:2406.06521), "unbiased depth rendering". The
/// bias this removes: brush's existing depth channel composites the camera-`z`
/// of each gaussian's **centre** (`project_visible.rs:86`), so the supervised
/// surface sits at the centres of a shell of ellipsoids rather than on the
/// surface those ellipsoids tile. Compositing each splat's tangent-plane
/// parameters instead and intersecting the ray with the composited plane puts
/// the depth on the plane, where the surface actually is.
///
/// `feat_img` is `[H, W, 5]` as emitted by the feature rasterizer:
/// channels `0..3` = `Σ wᵢ · n_camᵢ`, channel `3` = `Σ wᵢ · dᵢ`, channel `4` =
/// `α = Σ wᵢ`; the per-splat `(n_cam, d)` pairs come from
/// `brush_train::train::plane_features`, where `d` is defined so that every
/// point `p` on splat `i`'s tangent plane satisfies `n_camᵢ · p = dᵢ` in the
/// `OpenCV` camera frame.
///
/// # No alpha division
///
/// The plane is intersected as
/// `z = offset_sum / (n_sum · ray)`, using the **raw composited sums**. Alpha
/// cancels exactly between numerator and denominator — dividing both by `α`
/// would give the identical quotient — so, unlike the centre-depth path
/// (train.rs:1201-1203), there is no alpha normalization here and therefore no
/// detach decision to make about it. Alpha is still read, but only as a
/// coverage **mask** (`α ≥ min_alpha`), and it is detached for that use: a
/// validity test is not a gradient path.
///
/// # Ray convention — deliberately NOT the one `normals_from_depth` uses
///
/// The ray through integer pixel `(u, v)` is
/// `((u + 0.5 − cx) / fx, (v + 0.5 − cy) / fy, 1)`. The `+ 0.5` is the pixel
/// **centre**, which is the convention the rasterizer itself uses
/// (`rasterize.rs`: `pixel_coord_x = pix_x + 0.5`, compared against
/// `fx·x/z + cx` from `project_pinhole`). [`normals_from_depth`] above omits
/// it; that is harmless there because it is a *finite-difference* estimator and
/// a constant shift of the sample grid cancels to first order, but a ray-plane
/// intersection is a *direct evaluation* and does not get that cancellation —
/// a half-pixel error here is a real depth error of order
/// `0.5·|n_x|/(fx·(n·ray))` relative, which blows up at grazing incidence. Do
/// not "unify" the two by dropping the `0.5`.
///
/// # Validity and `NaN` discipline
///
/// A pixel is valid when all of: `α ≥ min_alpha`; `|n_sum · ray| ≥ min_denom`
/// (a plane seen edge-on has no well-defined ray intersection); and
/// `min_depth ≤ z ≤ max_depth`. Invalid pixels emit exactly `0` in both the
/// depth and the normal, and carry no gradient.
///
/// Every input channel is `mask_fill`-sanitised to `0` before any division or
/// `sqrt`, and the denominator is `mask_fill`-replaced by `1.0` wherever it
/// fails `min_denom`, so no non-finite value is ever produced in the forward
/// pass. This matters more than it looks: this fork has already paid once for a
/// `0 · ∞ = NaN` in the *backward* of a masked-out pixel (see the safe-norm
/// comment in [`normals_from_depth`]), and multiplying a non-finite value by a
/// zero mask reproduces it exactly. Masking with `mask_fill` (which replaces the
/// value and blocks the gradient) rather than with a multiply is what keeps that
/// closed.
///
/// # Returns
///
/// `(depth [H, W], normal [H, W, 3], valid [H, W])`, where `normal` is the
/// normalized `n_sum` (the composited camera-frame plane normal, for the
/// depth/normal consistency term when plane depth is the active depth source)
/// and `valid` is a `1.0`/`0.0` float mask.
///
/// Thresholds are parameters, not config: v1 pins PGSR-paper-typical constants
/// at the call site.
pub fn plane_depth_from_features(
    feat_img: Tensor<3>,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    min_alpha: f32,
    min_denom: f32,
    min_depth: f32,
    max_depth: f32,
) -> (Tensor<2>, Tensor<3>, Tensor<2>) {
    let [h, w, c] = feat_img.dims();
    assert_eq!(
        c, 5,
        "plane feature image must be [H, W, 5] (n_sum(3) + offset_sum(1) + alpha(1)), got {c} channels"
    );
    let device = feat_img.device();

    let chan =
        |t: Tensor<3>, i: usize| -> Tensor<2> { t.slice(s![.., .., i..i + 1]).reshape([h, w]) };
    let c0 = chan(feat_img.clone(), 0);
    let c1 = chan(feat_img.clone(), 1);
    let c2 = chan(feat_img.clone(), 2);
    let c3 = chan(feat_img.clone(), 3);
    let c4 = chan(feat_img, 4);

    // Sanitise EVERY channel up front, on the JOINT finite mask: one non-finite
    // channel makes the whole pixel meaningless, so zero the pixel rather than
    // repairing it channel-by-channel (a `NaN` in `n_x` alone would otherwise
    // decay into a perfectly plausible axis-aligned plane and be reported valid).
    // The pixel is then also excluded from `valid` below.
    //
    // Doing this BEFORE any division or sqrt is what keeps the backward clean: a
    // non-finite value that survives into an op and is masked out afterwards
    // reappears as `0 · ∞ = NaN` in that op's VJP and poisons the whole map.
    // `mask_fill` substitutes the value (so the forward never sees it) and
    // zeroes the gradient there (so the backward never multiplies by it).
    let all_finite = c0
        .clone()
        .is_finite()
        .bool_and(c1.clone().is_finite())
        .bool_and(c2.clone().is_finite())
        .bool_and(c3.clone().is_finite())
        .bool_and(c4.clone().is_finite());
    let non_finite = all_finite.clone().bool_not();

    let nx = c0.mask_fill(non_finite.clone(), 0.0);
    let ny = c1.mask_fill(non_finite.clone(), 0.0);
    let nz = c2.mask_fill(non_finite.clone(), 0.0);
    let offset = c3.mask_fill(non_finite.clone(), 0.0);
    // Alpha is a coverage MASK, never a gradient path.
    let alpha = c4.mask_fill(non_finite, 0.0).detach();

    // Pixel-CENTRE ray grid, built host-side (no `arange` in this burn rev) and
    // broadcast against the feature planes. See the ray-convention note above
    // for why the `+ 0.5` is here and not in `normals_from_depth`.
    let us: Vec<f32> = (0..w).map(|u| (u as f32 + 0.5 - cx) / fx).collect();
    let vs: Vec<f32> = (0..h).map(|v| (v as f32 + 0.5 - cy) / fy).collect();
    let a_u: Tensor<2> = Tensor::<1>::from_floats(us.as_slice(), &device).reshape([1, w]);
    let b_v: Tensor<2> = Tensor::<1>::from_floats(vs.as_slice(), &device).reshape([h, 1]);

    // n_sum · ray, with ray_z == 1 by construction.
    let denom = nx.clone() * a_u + ny.clone() * b_v + nz.clone();

    // A near-zero denominator is an edge-on plane: the intersection is
    // unbounded, so reject rather than clamp. The replacement value 1.0 is
    // arbitrary — it only has to be far from zero, since the quotient is
    // discarded at exactly these pixels.
    let denom_ok = denom.clone().abs().lower_elem(min_denom).bool_not();
    let safe_denom = denom.mask_fill(denom_ok.clone().bool_not(), 1.0);

    let depth_raw = offset / safe_denom;

    // Physical plausibility. After the sanitisation above nothing here is
    // non-finite, but both comparisons are false for a `NaN` anyway, so this
    // also fails safe. Note `min_depth > 0` is what rejects a plane BEHIND the
    // camera; an `|z| < max_depth` test would happily accept it.
    let in_range = depth_raw
        .clone()
        .lower_elem(min_depth)
        .bool_not()
        .bool_and(depth_raw.clone().greater_elem(max_depth).bool_not());

    let valid_mask = alpha
        .lower_elem(min_alpha)
        .bool_not()
        .bool_and(all_finite)
        .bool_and(denom_ok)
        .bool_and(in_range);
    let invalid = valid_mask.clone().bool_not();

    let depth = depth_raw.mask_fill(invalid.clone(), 0.0);

    // Safe norm, same discipline as `normals_from_depth`: clamp the SQUARED
    // length off zero before the sqrt, so the sqrt's derivative stays finite on
    // the (masked, zero-weight) background pixels.
    let n_sum: Tensor<3> = Tensor::cat(
        vec![
            nx.reshape([h, w, 1]),
            ny.reshape([h, w, 1]),
            nz.reshape([h, w, 1]),
        ],
        2,
    );
    let len = n_sum
        .clone()
        .powi_scalar(2)
        .sum_dim(2)
        .clamp_min(1e-24)
        .sqrt();
    let normal = n_sum / len.clamp_min(1e-12);
    let invalid3 = invalid.reshape([h, w, 1]).repeat_dim(2, 3);
    let normal = normal.mask_fill(invalid3, 0.0);

    (depth, normal, valid_mask.float())
}

/// Depth/normal consistency: `1 - dot` between normals derived from the
/// rendered depth and the rendered per-gaussian normals (`PlanarGS` `L_dn`).
///
/// `alpha` is `[H, W, 1]` and is expected to arrive already detached — the
/// consistency term must not be able to lower its error by changing
/// transparency, exactly like the depth loss's detached denominator. Pixels
/// with `alpha <= 0.5`, or with either normal invalid, are skipped.
pub fn depth_normal_loss(
    normal_from_depth: Tensor<3>,
    normal_rendered: Tensor<3>,
    alpha: Tensor<3>,
) -> Tensor<1> {
    let covered = alpha.greater_elem(0.5).float();
    let len_d = normal_from_depth.clone().powi_scalar(2).sum_dim(2).sqrt();
    let len_r = normal_rendered.clone().powi_scalar(2).sum_dim(2).sqrt();
    let valid = covered * len_d.greater_elem(0.5).float() * len_r.greater_elem(0.5).float();

    // NON-FINITE DISCIPLINE: substitute before the dot product. Same reasoning
    // and same byte-identity argument as `depth_loss` — an uncovered pixel holds
    // `raw / alpha.clamp_min(1e-10)`, which is exactly where a non-finite value
    // would come from if one ever did.
    let invalid3 = valid.clone().lower_elem(0.5).repeat_dim(2, 3);
    let normal_from_depth = normal_from_depth.mask_fill(invalid3.clone(), 0.0);
    let normal_rendered = normal_rendered.mask_fill(invalid3, 0.0);

    let dot = (normal_from_depth * normal_rendered).sum_dim(2);
    let err = dot.neg().add_scalar(1.0) * valid.clone();

    err.sum() / valid.sum().clamp_min(1.0)
}

/// Total-variation smoothness on a rendered camera-frame normal image,
/// `[H, W, 3]` (DN-Splatter's `L_smooth`).
///
/// `Σ |N[i+1,j] - N[i,j]| + |N[i,j+1] - N[i,j]|`, meaned over the differences
/// actually counted.
///
/// Why this exists at all: DN-Splatter weights this **0.5**, five times its
/// normal data term (0.1), making it the largest weight in their normal group.
/// On a low-texture surface the per-pixel normal field can be noisy while still
/// matching the prior *on average* — the data term cannot see that, this can.
/// Since textureless walls are the whole reason we added normal priors, dropping
/// the smoothness term would have left the most load-bearing piece out.
///
/// Deliberate deviation from DN-Splatter's plain TV: a difference counts only
/// when BOTH pixels are covered (`alpha > 0.5`) and carry a valid normal. Plain
/// TV also penalises the step across a silhouette, where the neighbour is the
/// `(0, 0, 0)` of an uncovered pixel rather than a surface measurement, and
/// smoothing that boundary is exactly backwards. Same validity contract as
/// `depth_normal_loss`.
///
/// The masking is load-bearing, not cosmetic: the caller builds this image as
/// `normal_img / alpha.clamp_min(1e-10)` and then unit-normalises it, so an
/// uncovered pixel holds amplified numerical noise pointing in an arbitrary
/// direction — not a benign background colour. Plain TV would push that garbage
/// into every silhouette-adjacent covered pixel. Note the mask drops only
/// covered↔uncovered differences; covered↔covered ones still count right up to
/// the edge, so smoothing survives where the noise actually is.
///
/// The `|n| > 0.5` validity check is near-vacuous for that caller (the input is
/// already unit-length wherever alpha is high) and is kept for contract parity
/// with `depth_normal_loss`, where it does real work on depth-derived normals.
///
/// `alpha` is `[H, W, 1]` and is expected to arrive already detached, so the term
/// cannot lower its error by changing transparency.
pub fn normal_smooth_loss(normal: Tensor<3>, alpha: Tensor<3>) -> Tensor<1> {
    let [h, w, _] = normal.dims();
    let device = normal.device();

    if h < 2 || w < 2 {
        return Tensor::zeros([1], &device);
    }

    let covered = alpha.greater_elem(0.5).float();
    let len = normal.clone().powi_scalar(2).sum_dim(2).sqrt();
    let valid = covered * len.greater_elem(0.5).float();

    // NON-FINITE DISCIPLINE, the fourth member of the family — see the block
    // comment in `depth_loss`. This one matters MORE than the others, not less:
    // the caller builds `normal` as `normal_img / alpha.clamp_min(1e-10)`, and
    // the doc comment above says in as many words that an uncovered pixel
    // therefore holds amplified numerical noise. A difference that touches such
    // a pixel is zeroed by `v_row`/`v_col`, which is exactly the `0 · ∞`
    // situation. Byte-identical for finite input: a difference with an invalid
    // endpoint is multiplied by 0 either way, and covered↔covered differences
    // never see the substitution.
    let invalid3 = valid.clone().lower_elem(0.5).repeat_dim(2, 3);
    let normal = normal.mask_fill(invalid3, 0.0);

    // Row differences: N[i+1, j] - N[i, j].
    let d_row = (normal.clone().slice(s![1..h, .., ..])
        - normal.clone().slice(s![0..h - 1, .., ..]))
    .abs()
    .sum_dim(2);
    let v_row = valid.clone().slice(s![1..h, .., ..]) * valid.clone().slice(s![0..h - 1, .., ..]);

    // Column differences: N[i, j+1] - N[i, j].
    let d_col = (normal.clone().slice(s![.., 1..w, ..]) - normal.slice(s![.., 0..w - 1, ..]))
        .abs()
        .sum_dim(2);
    let v_col = valid.clone().slice(s![.., 1..w, ..]) * valid.slice(s![.., 0..w - 1, ..]);

    let err = (d_row * v_row.clone()).sum() + (d_col * v_col.clone()).sum();

    // × 3 because `sum_dim(2)` already folded the three channels into each
    // counted difference, matching `normal_loss`'s denominator.
    err / (v_row.sum() + v_col.sum()).mul_scalar(3.0).clamp_min(1.0)
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

#[cfg(all(test, not(target_family = "wasm")))]
mod normal_loss_tests {
    use super::*;
    use burn::tensor::TensorData;

    const W: usize = 8;
    const H: usize = 6;
    const FX: f32 = 120.0;
    const FY: f32 = 110.0;
    const CX: f32 = W as f32 / 2.0;
    const CY: f32 = H as f32 / 2.0;

    async fn device() -> burn::tensor::Device {
        brush_cube::test_helpers::test_device().await.into()
    }

    async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .to_vec::<f32>()
            .expect("f32 tensor")
    }

    /// Interior pixels of a constant-depth map must give exactly (0, 0, -1),
    /// and the forward-difference border must be marked invalid.
    #[tokio::test]
    async fn fronto_parallel_plane_gives_minus_z() {
        let device = device().await;
        let depth = Tensor::<2>::ones([H, W], &device) * 2.0;
        let normals = read(normals_from_depth(depth, FX, FY, CX, CY)).await;

        for v in 0..H {
            for u in 0..W {
                let i = (v * W + u) * 3;
                let n = [normals[i], normals[i + 1], normals[i + 2]];
                if v + 1 == H || u + 1 == W {
                    assert_eq!(n, [0.0, 0.0, 0.0], "border ({u},{v}) must be invalid");
                } else {
                    assert!(n[0].abs() < 1e-5, "nx at ({u},{v}) = {}", n[0]);
                    assert!(n[1].abs() < 1e-5, "ny at ({u},{v}) = {}", n[1]);
                    assert!((n[2] + 1.0).abs() < 1e-5, "nz at ({u},{v}) = {}", n[2]);
                }
            }
        }
    }

    /// A plane `z = z0 + a * x_cam` has the closed-form camera-frame normal
    /// `(a, 0, -1) / sqrt(1 + a^2)`. The finite-difference estimator must
    /// recover it (exactly, up to float error: the plane is linear in the
    /// unprojected coordinates the differences are taken in).
    #[tokio::test]
    async fn slanted_plane_matches_the_closed_form_normal() {
        let device = device().await;
        let (z0, a) = (3.0f32, 0.35f32);

        let mut depth = vec![0.0f32; H * W];
        for v in 0..H {
            for u in 0..W {
                let a_u = (u as f32 - CX) / FX;
                depth[v * W + u] = z0 / (1.0 - a * a_u);
            }
        }
        let depth = Tensor::<2>::from_data(TensorData::new(depth, [H, W]), &device);
        let normals = read(normals_from_depth(depth, FX, FY, CX, CY)).await;

        let inv = 1.0 / (1.0 + a * a).sqrt();
        let want = [a * inv, 0.0, -inv];
        for v in 0..H - 1 {
            for u in 0..W - 1 {
                let i = (v * W + u) * 3;
                for c in 0..3 {
                    assert!(
                        (normals[i + c] - want[c]).abs() < 1e-4,
                        "n[{c}] at ({u},{v}) = {}, want {}",
                        normals[i + c],
                        want[c]
                    );
                }
            }
        }
    }

    /// The same check on a plane tilted in BOTH axes, which is what actually
    /// pins the estimator's handedness.
    ///
    /// `slanted_plane_matches_the_closed_form_normal` above tilts only in x, so
    /// its expected normal is `(a, 0, -1)`: `ny` is identically zero. That blind
    /// spot is bigger than it looks. Because the plane is flat in y, scaling the
    /// `dP/dv` difference by any constant leaves the normalized cross product
    /// unchanged — so `fy` can be wrong, or swapped with `fx`, and every
    /// assertion still passes. A fronto-parallel plane is worse again, giving
    /// `(0, 0, -1)` under nearly any convention and testing only the z sign.
    ///
    /// `z = z0 + a*x_cam + b*y_cam` has normal `(a, b, -1)/sqrt(1 + a^2 + b^2)`.
    /// Depth is built from the plane's geometry — substituting `x_cam = z*a_u`
    /// and `y_cam = z*b_v` and solving for z — not from the cross-product
    /// formula under test, so this is an independent check rather than a
    /// restatement. `a` and `b` differ in magnitude and sign, and `FX != FY`
    /// with `W != H`, so an axis swap, an intrinsic swap, or a sign flip each
    /// move the answer.
    #[tokio::test]
    async fn doubly_tilted_plane_pins_both_axes_and_intrinsics() {
        let device = device().await;
        let (z0, a, b) = (3.0f32, 0.35f32, -0.6f32);

        let mut depth = vec![0.0f32; H * W];
        for v in 0..H {
            for u in 0..W {
                let a_u = (u as f32 - CX) / FX;
                let b_v = (v as f32 - CY) / FY;
                depth[v * W + u] = z0 / (1.0 - a * a_u - b * b_v);
            }
        }
        let depth = Tensor::<2>::from_data(TensorData::new(depth, [H, W]), &device);
        let normals = read(normals_from_depth(depth, FX, FY, CX, CY)).await;

        let inv = 1.0 / (1.0 + a * a + b * b).sqrt();
        let want = [a * inv, b * inv, -inv];
        for v in 0..H - 1 {
            for u in 0..W - 1 {
                let i = (v * W + u) * 3;
                for c in 0..3 {
                    assert!(
                        (normals[i + c] - want[c]).abs() < 1e-4,
                        "n[{c}] at ({u},{v}) = {}, want {}",
                        normals[i + c],
                        want[c]
                    );
                }
            }
        }
    }

    /// Non-positive depths carry no geometry, so anything touching them is
    /// invalid.
    #[tokio::test]
    async fn non_positive_depth_is_invalid() {
        let device = device().await;
        let mut depth = vec![2.0f32; H * W];
        depth[2 * W + 3] = 0.0;
        let depth = Tensor::<2>::from_data(TensorData::new(depth, [H, W]), &device);
        let normals = read(normals_from_depth(depth, FX, FY, CX, CY)).await;

        // The hole itself and the two pixels whose forward differences read it.
        for (u, v) in [(3usize, 2usize), (2, 2), (3, 1)] {
            let i = (v * W + u) * 3;
            assert_eq!(
                [normals[i], normals[i + 1], normals[i + 2]],
                [0.0, 0.0, 0.0],
                "({u},{v}) must be invalid"
            );
        }
        // A pixel far from the hole is unaffected.
        let i = 0;
        assert!((normals[i + 2] + 1.0).abs() < 1e-5);
    }

    /// `normal_loss` averages |pred - gt| over valid pixels x 3 channels, and
    /// skips `(0,0,0)` prior pixels entirely.
    #[tokio::test]
    async fn normal_loss_masks_invalid_prior_pixels() {
        let device = device().await;
        // 2 pixels: one valid prior (0,0,-1), one invalid (0,0,0).
        let gt = Tensor::<3>::from_data(
            TensorData::new(vec![0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [1, 2, 3]),
            &device,
        );
        // Predictions differ on BOTH pixels; only the first may count.
        let pred = Tensor::<3>::from_data(
            TensorData::new(vec![0.0, 0.3, -1.0, 5.0, 5.0, 5.0], [1, 2, 3]),
            &device,
        );

        let loss = read(normal_loss(pred, gt, None)).await[0];
        // |0.3| spread over 1 valid pixel * 3 channels.
        assert!((loss - 0.1).abs() < 1e-6, "loss = {loss}");
    }

    /// An all-invalid prior yields 0, not NaN.
    ///
    /// The inputs are built with `from_data`, not `Tensor::zeros`/`ones`, on
    /// purpose. Production always calls `normal_loss(n_cam, gt_normal)` with a
    /// rendered `n_cam` and `gt_normal = Tensor::from_data(normal_data)` — both
    /// real, uploaded device tensors, and an all-invalid prior is simply a
    /// loaded `(0,0,0)` tensor. A lazy `Tensor::zeros`/`ones` graph is a
    /// different beast: when every input to this masked-reduction chain is a
    /// compile-time fill constant, burn-fusion 0.22-pre.2 constant-folds the
    /// whole fused block until its fallback op list drains to zero, then indexes
    /// that empty list from a stale ordering and panics on the shared-device
    /// runner thread (nondeterministically, whichever folded block loses the
    /// race that run). That fold path is unreachable from any real caller, so
    /// this test feeds the same all-zero values the way production does, which
    /// exercises the true code path and is robust. The asserted value is
    /// unchanged: an all-invalid prior must score exactly 0.
    #[tokio::test]
    async fn normal_loss_is_zero_with_no_valid_prior() {
        let device = device().await;
        let gt = Tensor::<3>::from_data(TensorData::new(vec![0.0f32; 6], [1, 2, 3]), &device);
        let pred = Tensor::<3>::from_data(TensorData::new(vec![1.0f32; 6], [1, 2, 3]), &device);
        let loss = read(normal_loss(pred, gt, None)).await[0];
        assert_eq!(loss, 0.0);
    }

    /// The NeuRIS-style per-pixel contradiction gate (`gate_cos`).
    ///
    /// Pins the three things that can silently go wrong: `None` must be the
    /// literal pre-gate code path, a gate wide enough to admit everything must
    /// equal the ungated result, and — the load-bearing one — the denominator
    /// must be the GATED valid count, so a surviving pixel keeps its full
    /// per-pixel magnitude instead of the whole term annealing as the gate
    /// tightens.
    #[tokio::test]
    async fn normal_loss_gate_none_matches_old() {
        let device = device().await;

        // Four pixels, all with a valid (unit-length) prior pointing at the
        // camera. Two predictions agree closely with the prior, two are
        // opposed and must be gated out at 30 degrees.
        //
        // Priors: all (0, 0, -1).
        let gt = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, -1.0, // agreeing pixel A
                    0.0, 0.0, -1.0, // agreeing pixel B
                    0.0, 0.0, -1.0, // contradicted pixel C
                    0.0, 0.0, -1.0, // contradicted pixel D
                ],
                [1, 4, 3],
            ),
            &device,
        );
        // A and B: exactly (0,0,-1) offset by 0.3 in one channel -> cos to the
        // prior is 1/sqrt(1+0.09) = 0.9578 -> 16.7 degrees, INSIDE a 30 degree
        // gate. C and D: (0,0,+1), exactly opposed -> cos = -1, gated out.
        let pred = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.3, -1.0, // A
                    0.0, 0.3, -1.0, // B
                    0.0, 0.0, 1.0, // C
                    0.0, 0.0, 1.0, // D
                ],
                [1, 4, 3],
            ),
            &device,
        );

        // Ungated: |0.3| twice from A/B plus |2.0| twice from C/D, over
        // 4 valid pixels * 3 channels.
        let ungated = read(normal_loss(pred.clone(), gt.clone(), None)).await[0];
        let want_ungated = (0.3 + 0.3 + 2.0 + 2.0) / 12.0;
        assert!(
            (ungated - want_ungated).abs() < 1e-6,
            "ungated = {ungated}, want {want_ungated}"
        );

        // A gate at 180 degrees (cos = -1) admits every pixel, so it must be
        // numerically identical to `None`.
        let wide = read(normal_loss(pred.clone(), gt.clone(), Some(-1.0))).await[0];
        assert!(
            (wide - ungated).abs() < 1e-6,
            "180-degree gate = {wide}, ungated = {ungated}"
        );

        // A 30 degree gate keeps exactly A and B. Denominator is the GATED
        // count: 2 pixels * 3 channels, NOT the ungated 4 * 3.
        let cos30 = 30.0_f32.to_radians().cos();
        let gated = read(normal_loss(pred.clone(), gt.clone(), Some(cos30))).await[0];
        let want_gated = (0.3 + 0.3) / 6.0;
        assert!(
            (gated - want_gated).abs() < 1e-6,
            "30-degree gate = {gated}, want {want_gated} (denominator must be the gated count)"
        );

        // An unrenormalized denominator would have given (0.3+0.3)/12 = 0.05.
        assert!(
            (gated - 0.05).abs() > 1e-3,
            "the gate must not silently anneal the whole term"
        );

        // A gate that survives nothing yields a differentiable exact 0, not
        // NaN: the `clamp_min(1.0)` denominator, matching the reference's
        // `sum / max(count, 1)` discipline.
        let empty = read(normal_loss(pred, gt, Some(0.999_999))).await[0];
        assert_eq!(empty, 0.0);
    }

    /// `normal_gate_counts` is the diagnostic behind the "gate is over-masking"
    /// warning, so it has to be right about both halves of the ratio.
    ///
    /// The two-count return is what lets the trainer distinguish "the gate
    /// masked almost everything" from "this frame barely had a prior" — a
    /// pre-divided fraction cannot express the difference, and the second case
    /// must never trip the warning.
    #[tokio::test]
    async fn normal_gate_counts_report_survivors_and_valid_pixels() {
        let device = device().await;
        let cos30 = 30.0_f32.to_radians().cos();

        // Four pixels: two priors agreeing with the prediction to ~16.7 deg,
        // two exactly opposed, all four priors valid.
        let gt = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, -1.0, //
                    0.0, 0.0, -1.0, //
                    0.0, 0.0, -1.0, //
                    0.0, 0.0, -1.0, //
                ],
                [1, 4, 3],
            ),
            &device,
        );
        let pred = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.3, -1.0, //
                    0.0, 0.3, -1.0, //
                    0.0, 0.0, 1.0, //
                    0.0, 0.0, 1.0, //
                ],
                [1, 4, 3],
            ),
            &device,
        );

        let counts = read(normal_gate_counts(pred.clone(), gt.clone(), cos30)).await;
        assert_eq!(counts.len(), 2, "counts must be [surviving, valid]");
        assert!((counts[0] - 2.0).abs() < 1e-6, "surviving = {}", counts[0]);
        assert!((counts[1] - 4.0).abs() < 1e-6, "valid = {}", counts[1]);

        // A gate wide enough to admit everything: every valid pixel survives.
        let wide = read(normal_gate_counts(pred.clone(), gt.clone(), -1.0)).await;
        assert!((wide[0] - 4.0).abs() < 1e-6, "surviving = {}", wide[0]);
        assert!((wide[1] - 4.0).abs() < 1e-6, "valid = {}", wide[1]);

        // A gate nothing survives: this is the shape the warning keys on —
        // zero survivors out of a NONZERO valid count.
        let none = read(normal_gate_counts(pred, gt, 0.999_999)).await;
        assert_eq!(none[0], 0.0);
        assert!((none[1] - 4.0).abs() < 1e-6, "valid = {}", none[1]);
    }

    /// A frame with no usable prior reports `valid == 0`, NOT a zero fraction.
    ///
    /// This is the case that must never trip the over-masking warning: the gate
    /// discarded nothing, there was simply nothing to discard. Collapsing to a
    /// ratio here would divide by zero or report 0%, either of which reads as a
    /// broken gate.
    #[tokio::test]
    async fn normal_gate_counts_separate_empty_prior_from_full_masking() {
        let device = device().await;
        let gt = Tensor::<3>::from_data(TensorData::new(vec![0.0f32; 12], [1, 4, 3]), &device);
        let pred = Tensor::<3>::from_data(TensorData::new(vec![1.0f32; 12], [1, 4, 3]), &device);

        let counts = read(normal_gate_counts(pred, gt, 30.0_f32.to_radians().cos())).await;
        assert_eq!(counts[0], 0.0, "no valid prior means no survivors");
        assert_eq!(counts[1], 0.0, "valid count must be 0, not clamped to 1");
    }

    /// `depth_normal_loss` is 0 when the two normal fields agree, 2 when they
    /// are opposed, and ignores uncovered pixels.
    #[tokio::test]
    async fn depth_normal_loss_scores_agreement_under_the_alpha_mask() {
        let device = device().await;
        let n_d = Tensor::<3>::from_data(
            TensorData::new(vec![0.0, 0.0, -1.0, 0.0, 0.0, -1.0], [1, 2, 3]),
            &device,
        );
        let n_r = Tensor::<3>::from_data(
            TensorData::new(vec![0.0, 0.0, -1.0, 0.0, 0.0, 1.0], [1, 2, 3]),
            &device,
        );

        // Both pixels covered: mean of (1 - 1) and (1 - (-1)) = 1.0.
        let alpha = Tensor::<3>::ones([1, 2, 1], &device);
        let both = read(depth_normal_loss(n_d.clone(), n_r.clone(), alpha)).await[0];
        assert!((both - 1.0).abs() < 1e-6, "both = {both}");

        // Only the agreeing pixel covered: 0.
        let alpha = Tensor::<3>::from_data(TensorData::new(vec![1.0, 0.0], [1, 2, 1]), &device);
        let first = read(depth_normal_loss(n_d.clone(), n_r.clone(), alpha)).await[0];
        assert!(first.abs() < 1e-6, "first = {first}");

        // Nothing covered: 0, not NaN.
        let alpha = Tensor::<3>::zeros([1, 2, 1], &device);
        let none = read(depth_normal_loss(n_d, n_r, alpha)).await[0];
        assert_eq!(none, 0.0);
    }

    /// Hand-computed TV on a 2x3 grid of unit normals, all covered.
    /// Row diffs (1x3): col1 = |(0,0,1)-(0,1,0)| = 2, others 0. count 3.
    /// Col diffs (2x2): row0: 0, 2; row1: 2, 2. count 4.
    /// err = 8, denom = (3+4)*3 = 21.
    #[tokio::test]
    async fn normal_smooth_loss_matches_hand_computed_tv() {
        let device = device().await;
        #[rustfmt::skip]
        let n = Tensor::<3>::from_data(
            TensorData::new(vec![
                0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,   0.0, 1.0, 0.0,   1.0, 0.0, 0.0,
            ], [2, 3, 3]),
            &device,
        );
        let alpha = Tensor::<3>::ones([2, 3, 1], &device);
        let loss = read(normal_smooth_loss(n, alpha)).await[0];
        assert!((loss - 8.0 / 21.0).abs() < 1e-6, "loss = {loss}");
    }

    /// Uncovering pixel (1,1) must drop every difference that touches it:
    /// row count 3->2 (err 0), col count 4->2 (err 2). loss = 2/(4*3).
    #[tokio::test]
    async fn normal_smooth_loss_drops_diffs_touching_uncovered_pixels() {
        let device = device().await;
        #[rustfmt::skip]
        let n = Tensor::<3>::from_data(
            TensorData::new(vec![
                0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,   0.0, 1.0, 0.0,   1.0, 0.0, 0.0,
            ], [2, 3, 3]),
            &device,
        );
        let alpha = Tensor::<3>::from_data(
            TensorData::new(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0], [2, 3, 1]),
            &device,
        );
        let loss = read(normal_smooth_loss(n, alpha)).await[0];
        assert!((loss - 2.0 / 12.0).abs() < 1e-6, "loss = {loss}");
    }

    /// Zero-length (invalid) normals are skipped even when covered, and an
    /// all-invalid frame yields 0, not NaN. A 1-pixel-tall frame returns 0.
    ///
    /// Every input is built with `from_data`, matching the real caller
    /// (`normal_smooth_loss(n_cam, normal_alpha)`, both rendered device
    /// tensors) and side-stepping the burn-fusion 0.22-pre.2 fold panic that a
    /// lazy `Tensor::zeros`/`ones` graph provokes — see the note on
    /// `normal_loss_is_zero_with_no_valid_prior`. The asserted values (2/6 and
    /// 0) are unchanged.
    #[tokio::test]
    async fn normal_smooth_loss_edge_cases() {
        let device = device().await;
        // Covered but zero-length normal at (0,1) of a 2x2 grid.
        #[rustfmt::skip]
        let n = Tensor::<3>::from_data(
            TensorData::new(vec![
                0.0, 0.0, 1.0,   0.0, 0.0, 0.0,
                0.0, 0.0, 1.0,   1.0, 0.0, 0.0,
            ], [2, 2, 3]),
            &device,
        );
        let alpha = Tensor::<3>::from_data(TensorData::new(vec![1.0f32; 4], [2, 2, 1]), &device);
        // Valid diffs: row col0 (0), col row1 |(0,0,1)-(1,0,0)|=2. counts: row 1, col 1.
        let loss = read(normal_smooth_loss(n, alpha)).await[0];
        assert!((loss - 2.0 / 6.0).abs() < 1e-6, "loss = {loss}");

        // Nothing covered: 0, not NaN.
        let n2 = Tensor::<3>::from_data(TensorData::new(vec![1.0f32; 12], [2, 2, 3]), &device);
        let a2 = Tensor::<3>::from_data(TensorData::new(vec![0.0f32; 4], [2, 2, 1]), &device);
        assert_eq!(read(normal_smooth_loss(n2, a2)).await[0], 0.0);

        // Degenerate frame: too small for any difference.
        let n3 = Tensor::<3>::from_data(TensorData::new(vec![1.0f32; 15], [1, 5, 3]), &device);
        let a3 = Tensor::<3>::from_data(TensorData::new(vec![1.0f32; 5], [1, 5, 1]), &device);
        assert_eq!(read(normal_smooth_loss(n3, a3)).await[0], 0.0);
    }

    /// `depth_loss(.., None)` must equal `depth_loss(.., Some(ones))` exactly:
    /// a unit per-pixel weight is the byte-identity of the pre-change fn, even
    /// with some `gt <= 0` (invalid) pixels in the map.
    #[tokio::test]
    async fn depth_loss_none_matches_unit_weight() {
        let device = device().await;
        let pred = Tensor::<2>::from_data(
            TensorData::new(vec![1.0, 2.0, 4.0, 0.5, 3.0, 1.5], [2, 3]),
            &device,
        );
        // Two of the six pixels have gt <= 0 (invalid, skipped).
        let gt = Tensor::<2>::from_data(
            TensorData::new(vec![2.0, 0.0, 3.0, 1.0, -1.0, 2.0], [2, 3]),
            &device,
        );
        let ones = Tensor::<2>::ones([2, 3], &device);

        let none = read(depth_loss(
            pred.clone(),
            gt.clone(),
            None,
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        ))
        .await[0];
        let unit = read(depth_loss(
            pred,
            gt,
            Some(ones),
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        ))
        .await[0];
        assert_eq!(none, unit);
    }

    /// **The `--depth-loss-space` value pin.** Hand-computed on a two-pixel map
    /// so the two spaces cannot be confused for one another by a refactor.
    ///
    /// One valid pixel (`pred = 2`, `gt = 1`) and one invalid (`gt = 0`), so
    /// the denominator is 1 and the loss IS the single residual:
    ///
    /// ```text
    ///   disparity : |1/2 - 1/1| = 0.5
    ///   metric    : |2   -   1| = 1.0
    /// ```
    ///
    /// The factor of exactly 2 between them is the point: at `pred = 2` a
    /// disparity residual is a quarter of a metric one per unit of error, and
    /// the numbers here would coincide if either arm silently fell through to
    /// the other. The masked-out pixel carries a large prediction (`50.0`)
    /// against `gt = 0`, which the metric arm would score as an enormous error
    /// if its mask were dropped — so this also pins that both arms share the
    /// SAME validity rule, not just the same denominator.
    #[tokio::test]
    async fn depth_loss_metric_and_disparity_are_pinned_and_distinct() {
        let device = device().await;
        let pred = Tensor::<2>::from_data(TensorData::new(vec![2.0, 50.0], [1, 2]), &device);
        let gt = Tensor::<2>::from_data(TensorData::new(vec![1.0, 0.0], [1, 2]), &device);

        let disparity = read(depth_loss(
            pred.clone(),
            gt.clone(),
            None,
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        ))
        .await[0];
        let metric = read(depth_loss(
            pred,
            gt,
            None,
            DepthLossSpace::Metric,
            DepthUncovered::Count,
        ))
        .await[0];

        assert!(
            (disparity - 0.5).abs() < 1e-6,
            "disparity loss {disparity}, want 0.5"
        );
        assert!(
            (metric - 1.0).abs() < 1e-6,
            "metric loss {metric}, want 1.0"
        );
    }

    /// A vertical step edge (cols 0-2 = 0.0, cols 3-5 = 1.0) gives weight
    /// `exp(-1/sigma)` only where a forward difference crosses the step
    /// (column 2), and 1.0 everywhere else including the last row/col.
    #[tokio::test]
    async fn rgb_grad_weight_synthetic_edge() {
        let device = device().await;
        let (h, w) = (4usize, 6usize);
        let mut data = vec![0.0f32; h * w * 3];
        for y in 0..h {
            for x in 3..w {
                for c in 0..3 {
                    data[(y * w + x) * 3 + c] = 1.0;
                }
            }
        }
        let img = Tensor::<3>::from_data(TensorData::new(data, [h, w, 3]), &device);
        let weight = read(rgb_grad_weight(img, 0.5)).await;

        let edge = (-2.0f32).exp(); // exp(-|1|/0.5)
        for y in 0..h {
            for x in 0..w {
                let got = weight[y * w + x];
                let expected = if x == 2 { edge } else { 1.0 };
                assert!(
                    (got - expected).abs() < 1e-6,
                    "weight at ({x},{y}) = {got}, expected {expected}"
                );
            }
        }
    }

    /// Mirror of `rgb_grad_weight_synthetic_edge` on the VERTICAL axis: a
    /// horizontal step edge (rows 0-2 = 0.0, rows 3-5 = 1.0) gives weight
    /// `exp(-1/sigma)` only where a row forward difference crosses the step
    /// (row 2), and 1.0 everywhere else including the last row/col. Exercises
    /// the `gy` `slice_assign` zero-pad direction the x-only fixtures never hit.
    #[tokio::test]
    async fn rgb_grad_weight_synthetic_edge_horizontal() {
        let device = device().await;
        let (h, w) = (6usize, 4usize);
        let mut data = vec![0.0f32; h * w * 3];
        for y in 3..h {
            for x in 0..w {
                for c in 0..3 {
                    data[(y * w + x) * 3 + c] = 1.0;
                }
            }
        }
        let img = Tensor::<3>::from_data(TensorData::new(data, [h, w, 3]), &device);
        let weight = read(rgb_grad_weight(img, 0.5)).await;

        let edge = (-2.0f32).exp(); // exp(-|1|/0.5)
        for y in 0..h {
            for x in 0..w {
                let got = weight[y * w + x];
                let expected = if y == 2 { edge } else { 1.0 };
                assert!(
                    (got - expected).abs() < 1e-6,
                    "weight at ({x},{y}) = {got}, expected {expected}"
                );
            }
        }
    }

    /// Gradient-aware weighting down-weights edges: a constant disparity-error
    /// field modulated by the step-image weight equals the hand-computed
    /// `sum(w)/N * err` and is strictly below the unweighted loss.
    #[tokio::test]
    async fn grad_aware_depth_loss_downweights_edges() {
        let device = device().await;
        let (h, w) = (4usize, 6usize);

        // Constant per-pixel disparity error: pred disparity 1/1 = 1, gt
        // disparity 1/2 = 0.5, so |err| = 0.5 at every valid pixel.
        let pred = Tensor::<2>::ones([h, w], &device);
        let gt = Tensor::<2>::ones([h, w], &device) * 2.0;
        let err = 0.5f32;
        let n = (h * w) as f32;

        // The step image + its weight (edge column 2 = exp(-2), else 1.0).
        let mut data = vec![0.0f32; h * w * 3];
        for y in 0..h {
            for x in 3..w {
                for c in 0..3 {
                    data[(y * w + x) * 3 + c] = 1.0;
                }
            }
        }
        let img = Tensor::<3>::from_data(TensorData::new(data, [h, w, 3]), &device);
        let weight = rgb_grad_weight(img, 0.5);
        let w_vals = read(weight.clone()).await;
        let sum_w: f32 = w_vals.iter().sum();

        let weighted = read(depth_loss(
            pred.clone(),
            gt.clone(),
            Some(weight),
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        ))
        .await[0];
        let unweighted = read(depth_loss(
            pred,
            gt,
            None,
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        ))
        .await[0];

        // Denominator is the unweighted valid count N; err is constant.
        assert!(
            (weighted - sum_w / n * err).abs() < 1e-6,
            "weighted = {weighted}, expected {}",
            sum_w / n * err
        );
        assert!(
            weighted < unweighted,
            "edge weighting must reduce the loss: {weighted} !< {unweighted}"
        );
    }
}

/// Tests 4 and 5 of the PGSR plane-render plan: the ray-plane depth math.
///
/// These build the composited feature image ANALYTICALLY rather than by
/// rendering. That is deliberate and is the stronger check for this function:
/// a real render can only ever produce one particular coverage weight per pixel,
/// whereas the whole "no alpha division" claim is a statement about what happens
/// when the weight VARIES. Building `(w·n, w·d, w)` by hand with a per-pixel `w`
/// lets `alpha_cancels_out_of_the_quotient` pin exactly that. The end-to-end
/// check against a real rasterized slab lives in brush-train
/// (`plane_depth_matches_a_rendered_slab`), where `plane_features` is available.
#[cfg(all(test, not(target_family = "wasm")))]
mod plane_depth_tests {
    use super::*;
    use brush_render::burn_glue::lift_to_autodiff;
    use burn::tensor::TensorData;

    // A deliberately SHORT focal on a wide-ish image: the rays then span
    // roughly ±0.75 in x and ±0.61 in y, so a tilted plane's depth varies by
    // more than 4x across the frame. With the long focals the `normal_loss_tests`
    // module uses, every ray is within 3% of the optical axis and a tilted
    // plane is numerically indistinguishable from a fronto-parallel one — which
    // is exactly the bug class this test exists to catch.
    const W: usize = 16;
    const H: usize = 12;
    const FX: f32 = 10.0;
    const FY: f32 = 9.0;
    const CX: f32 = W as f32 / 2.0;
    const CY: f32 = H as f32 / 2.0;

    const MIN_ALPHA: f32 = 0.1;
    const MIN_DENOM: f32 = 0.05;
    const MIN_DEPTH: f32 = 0.05;
    const MAX_DEPTH: f32 = 100.0;

    async fn device() -> burn::tensor::Device {
        brush_cube::test_helpers::test_device().await.into()
    }

    async fn autodiff_device() -> burn::tensor::Device {
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff()
    }

    async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .to_vec::<f32>()
            .expect("f32 tensor")
    }

    /// Pixel-CENTRE ray, matching the rasterizer (`pix + 0.5` against
    /// `fx·x/z + cx`). Written out here independently of the implementation so
    /// a convention change in one has to be made in the other too.
    fn ray(u: usize, v: usize) -> (f32, f32) {
        ((u as f32 + 0.5 - CX) / FX, (v as f32 + 0.5 - CY) / FY)
    }

    /// Camera-facing unit normal and offset of the plane `z = z0 + a·x + b·y`,
    /// in the `OpenCV` camera frame.
    ///
    /// `n·p = (a·x + b·y − z)/L`, which on the plane is `−z0/L`, so
    /// `n = (a, b, −1)/L` and `d = −z0/L` with `L = sqrt(1 + a² + b²)`. The
    /// normal points back toward the camera (`n_z < 0`) and the offset of a
    /// visible plane is negative, which is the sign convention `splat_normals`
    /// produces and `normals_from_depth` agrees with.
    fn plane_params(z0: f32, a: f32, b: f32) -> ([f32; 3], f32) {
        let l = (1.0 + a * a + b * b).sqrt();
        ([a / l, b / l, -1.0 / l], -z0 / l)
    }

    /// Closed-form depth of that plane along the pixel ray, derived from the
    /// plane's GEOMETRY (substitute `x = z·rᵤ`, `y = z·r_v` into
    /// `z = z0 + a·x + b·y` and solve) rather than from the `d/(n·ray)`
    /// formula under test — same construction the existing
    /// `doubly_tilted_plane_pins_both_axes_and_intrinsics` uses.
    fn plane_depth(z0: f32, a: f32, b: f32, u: usize, v: usize) -> f32 {
        let (ru, rv) = ray(u, v);
        z0 / (1.0 - a * ru - b * rv)
    }

    /// Build `[H, W, 5]` = `(w·n, w·d, w)` for a single plane covering the whole
    /// frame with per-pixel coverage weight `w`.
    fn feature_image(n: [f32; 3], d: f32, weight: &dyn Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut data = vec![0.0f32; H * W * 5];
        for v in 0..H {
            for u in 0..W {
                let w = weight(u, v);
                let i = (v * W + u) * 5;
                data[i] = w * n[0];
                data[i + 1] = w * n[1];
                data[i + 2] = w * n[2];
                data[i + 3] = w * d;
                data[i + 4] = w;
            }
        }
        data
    }

    /// A varying, always-covered weight. Deliberately not constant: see the
    /// module comment.
    fn varying_weight(u: usize, v: usize) -> f32 {
        0.3 + 0.65 * (((u * 3 + v * 5) % 7) as f32 / 6.0)
    }

    async fn run(data: Vec<f32>, device: &burn::tensor::Device) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let feat = Tensor::<3>::from_data(TensorData::new(data, [H, W, 5]), device);
        let (depth, normal, valid) = plane_depth_from_features(
            feat, FX, FY, CX, CY, MIN_ALPHA, MIN_DENOM, MIN_DEPTH, MAX_DEPTH,
        );
        (read(depth).await, read(normal).await, read(valid).await)
    }

    /// **Test 4.** The composited plane features of a single flat splat must
    /// yield exactly the closed-form ray-plane depth, on a fronto-parallel slab
    /// AND on a doubly-tilted one.
    ///
    /// The tilted case is the load-bearing one. A fronto-parallel plane has
    /// `n = (0, 0, −1)`, so `n·ray` collapses to `−1` regardless of the ray grid
    /// — the intrinsics, the pixel-centre offset, and both transverse normal
    /// components are all multiplied by zero and the test passes under almost
    /// any convention. Tilting in BOTH axes, with `a` and `b` differing in
    /// magnitude and sign and `FX != FY` with `W != H`, makes an axis swap, an
    /// intrinsic swap, a sign flip, or a half-pixel grid offset each move the
    /// answer well outside 1e-4.
    #[tokio::test]
    async fn plane_depth_flat_slab_exact() {
        let device = device().await;

        for (z0, a, b) in [(3.0f32, 0.0f32, 0.0f32), (3.0, 0.35, -0.6)] {
            let (n, d) = plane_params(z0, a, b);
            let (depth, normal, valid) = run(feature_image(n, d, &varying_weight), &device).await;

            for v in 0..H {
                for u in 0..W {
                    let i = v * W + u;
                    assert_eq!(
                        valid[i], 1.0,
                        "({u},{v}) must be valid for z0={z0} a={a} b={b}"
                    );

                    let want = plane_depth(z0, a, b, u, v);
                    assert!(
                        (depth[i] - want).abs() < 1e-4,
                        "depth at ({u},{v}) for z0={z0} a={a} b={b} = {}, want {want}",
                        depth[i]
                    );

                    for c in 0..3 {
                        assert!(
                            (normal[i * 3 + c] - n[c]).abs() < 1e-5,
                            "normal[{c}] at ({u},{v}) = {}, want {}",
                            normal[i * 3 + c],
                            n[c]
                        );
                    }
                }
            }

            // The tilted arm must actually be tilted: if the depth were flat
            // across the frame the assertions above would be vacuous for the
            // ray grid.
            if a != 0.0 || b != 0.0 {
                let (lo, hi) = depth
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), &z| (lo.min(z), hi.max(z)));
                assert!(
                    hi / lo > 3.0,
                    "the tilted fixture must span a wide depth range, got {lo}..{hi}"
                );
            }
        }
    }

    /// The "no alpha division" contract, stated as a test: alpha cancels
    /// between numerator and denominator, so scaling every channel of a pixel by
    /// its coverage weight must leave the depth bit-for-bit alone.
    #[tokio::test]
    async fn alpha_cancels_out_of_the_quotient() {
        let device = device().await;
        let (n, d) = plane_params(3.0, 0.35, -0.6);

        let (varying, _, _) = run(feature_image(n, d, &varying_weight), &device).await;
        let (unit, _, _) = run(feature_image(n, d, &|_, _| 1.0), &device).await;

        for (i, (a, b)) in varying.iter().zip(unit.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "pixel {i}: depth changed with coverage weight, {a} vs {b} — \
                 alpha must cancel out of offset_sum / (n_sum · ray)"
            );
        }
    }

    /// **Test 5.** Every rejection route emits exactly 0 in both outputs, and —
    /// the part that actually matters — the BACKWARD stays finite.
    ///
    /// This is the `0 · ∞ = NaN` regression class the fork already paid for once
    /// (the safe-norm comment in `normals_from_depth`). A masked pixel whose raw
    /// value is `inf` or `NaN` poisons the gradient of the WHOLE map if the mask
    /// is applied as a multiply, because autodiff then evaluates `0 · ∞`. The
    /// forward assertions below would pass under that bug; only the gradient
    /// check catches it, so assert on the backward, not just the forward.
    #[tokio::test]
    async fn plane_depth_invalid_pixels_are_zero_and_nan_free() {
        let device = autodiff_device().await;
        let (n, d) = plane_params(3.0, 0.35, -0.6);
        let mut data = feature_image(n, d, &varying_weight);

        let set = |data: &mut Vec<f32>, u: usize, v: usize, px: [f32; 5]| {
            let i = (v * W + u) * 5;
            data[i..i + 5].copy_from_slice(&px);
        };

        // (0, 0): zero alpha — an uncovered background pixel. Everything is 0,
        // so the denominator is 0 too; both routes must reject it.
        set(&mut data, 0, 0, [0.0; 5]);
        // (1, 0): covered, but the plane is seen exactly edge-on. The ray at
        // (1, 0) is (-0.65, -0.611, 1); a normal orthogonal to it gives
        // denom == 0 while alpha is healthy, isolating the min_denom route.
        let (r1u, r1v) = ray(1, 0);
        let graze = {
            // Any vector orthogonal to the ray: (ray x e_x), normalized.
            let (gx, gy, gz) = (0.0, 1.0, -r1v);
            let l = (gx * gx + gy * gy + gz * gz).sqrt();
            let _ = r1u;
            [gx / l, gy / l, gz / l]
        };
        set(&mut data, 1, 0, [graze[0], graze[1], graze[2], -3.0, 1.0]);
        // (2, 0): depth far beyond max_depth (tiny but supra-threshold denom).
        set(&mut data, 2, 0, [0.0, 0.0, -0.06, -1000.0, 1.0]);
        // (3, 0): depth below min_depth — the plane is behind / on the lens.
        set(&mut data, 3, 0, [0.0, 0.0, -1.0, -0.001, 1.0]);
        // (4, 0): NEGATIVE depth — a plane behind the camera. `min_depth > 0`
        // must reject it; a bare `|z| < max` test would not.
        set(&mut data, 4, 0, [0.0, 0.0, -1.0, 3.0, 1.0]);
        // (5, 0) and (6, 0): non-finite channels arriving from upstream.
        set(&mut data, 5, 0, [f32::NAN, 0.0, -1.0, -3.0, 1.0]);
        set(&mut data, 6, 0, [0.0, 0.0, -1.0, f32::INFINITY, 1.0]);

        let invalid: Vec<(usize, usize)> = (0..7).map(|u| (u, 0)).collect();

        let feat = lift_to_autodiff(Tensor::<3>::from_data(
            TensorData::new(data, [H, W, 5]),
            &device,
        ))
        .require_grad();

        let (depth, normal, valid) = plane_depth_from_features(
            feat.clone(),
            FX,
            FY,
            CX,
            CY,
            MIN_ALPHA,
            MIN_DENOM,
            MIN_DEPTH,
            MAX_DEPTH,
        );

        let depth_v = read(depth.clone()).await;
        let normal_v = read(normal.clone()).await;
        let valid_v = read(valid).await;

        for &(u, v) in &invalid {
            let i = v * W + u;
            assert_eq!(valid_v[i], 0.0, "({u},{v}) must be marked invalid");
            assert_eq!(depth_v[i], 0.0, "({u},{v}) depth must be exactly 0");
            for c in 0..3 {
                assert_eq!(
                    normal_v[i * 3 + c],
                    0.0,
                    "({u},{v}) normal[{c}] must be exactly 0"
                );
            }
        }

        // Untouched pixels must be unaffected by their poisoned neighbours: this
        // function is per-pixel, so a NaN that "spreads" means it leaked through
        // a reduction it should not have.
        for v in 0..H {
            for u in 0..W {
                if invalid.contains(&(u, v)) {
                    continue;
                }
                let i = v * W + u;
                assert_eq!(valid_v[i], 1.0, "({u},{v}) must still be valid");
                let want = plane_depth(3.0, 0.35, -0.6, u, v);
                assert!(
                    (depth_v[i] - want).abs() < 1e-4,
                    "depth at ({u},{v}) = {}, want {want}",
                    depth_v[i]
                );
            }
        }

        // Backward. A loss over BOTH outputs, so neither the division nor the
        // safe-norm can hide.
        let loss = depth.sum() + normal.sum();
        let grads = loss.backward();
        let g = read(
            feat.grad(&grads)
                .expect("the feature image must receive a gradient"),
        )
        .await;

        assert!(
            g.iter().all(|x| x.is_finite()),
            "plane-depth backward produced {} non-finite gradient entries out of {}",
            g.iter().filter(|x| !x.is_finite()).count(),
            g.len()
        );
        // The gradient must be zero exactly on the rejected pixels (a mask that
        // blocks the value but leaks the gradient is the other half of the bug).
        for &(u, v) in &invalid {
            let i = (v * W + u) * 5;
            for c in 0..5 {
                assert_eq!(
                    g[i + c],
                    0.0,
                    "({u},{v}) channel {c} must carry no gradient, got {}",
                    g[i + c]
                );
            }
        }
        // ...and nonzero somewhere on the valid pixels, so the test cannot pass
        // by the whole map being dead.
        let live: f32 = g[W * 5..].iter().map(|x| x.abs()).sum();
        assert!(
            live > 1e-6,
            "valid pixels must carry a real gradient, got {live}"
        );
    }
}

/// Near-grazing accuracy and the exact validity cutoff of the ray-plane solve,
/// ported from the reference suite (`gauss-surf`, Apache-2.0, Pablo Vela:
/// `tests/test_gaussurf_geometry.py`,
/// `test_plane_depth_matches_exact_world_ray_plane_intersection` and
/// `test_plane_depth_uses_the_documented_degenerate_cutoff`).
///
/// `plane_depth_tests` above pins the well-conditioned case at a flat `1e-4`.
/// That tolerance is far LOOSER than the arithmetic actually delivers there
/// (measured worst error on the tilted slab: 4.8e-7), so a regression that
/// degraded accuracy by two orders of magnitude would still pass it. These
/// tests close that gap from the other end: they drive the denominator down to
/// the validity cutoff, where the quotient is worst-conditioned, and compare
/// against an f64 host-side oracle under a tolerance derived from the
/// conditioning rather than picked round.
///
/// **Why the reference's flat `2e-7` absolute tolerance does not port.** The
/// reference computes the whole chain in float64; we are f32 end to end, and
/// `depth = offset / (n·ray)` near-grazing is genuinely ill-conditioned: `n·ray`
/// is an O(1) sum that cancels down to O(`min_denom`), so its ~`eps·‖ray‖`
/// absolute rounding error becomes a `eps·‖ray‖/|denom|` RELATIVE error in the
/// quotient. At our production `min_denom = 0.05` that is ~4e-6 relative, i.e.
/// about 30x the raw f32 epsilon and ~200x the reference's absolute figure at a
/// depth of 1 m. Asserting 2e-7 here would not be a stricter test, it would be a
/// test of float64 that we cannot run. The conditioning-scaled bound below is
/// the strongest statement the arithmetic supports, and it is much tighter than
/// a flat `1e-4` everywhere except right at the cutoff.
#[cfg(all(test, not(target_family = "wasm")))]
mod plane_depth_grazing_tests {
    use super::*;
    use burn::tensor::TensorData;

    const W: usize = 8;
    const H: usize = 8;
    const FX: f32 = 3.0;
    const FY: f32 = 2.5;
    const CX: f32 = 4.0;
    const CY: f32 = 4.0;

    // The PRODUCTION thresholds (train.rs `PLANE_MIN_*`), not the softer ones
    // `plane_depth_tests` uses: the point of this module is the behaviour AT the
    // real cutoff, so mirroring the real constant is load-bearing. If train.rs
    // changes `PLANE_MIN_DENOM`, this module is meant to be updated with it.
    const MIN_ALPHA: f32 = 0.5;
    const MIN_DENOM: f32 = 0.05;
    const MIN_DEPTH: f32 = 1e-3;
    const MAX_DEPTH: f32 = 1e4;

    /// How many f32 epsilons of conditioning-scaled slack the assertions allow.
    /// Measured, not guessed: the worst of the 64 cases lands at 0.32 (see
    /// `plane_depth_near_grazing_matches_a_float64_oracle`), so 2.0 is ~6x
    /// headroom. Do not raise it to make a failure go away — the whole point of
    /// the conditioning scaling is that the budget stays constant while the
    /// conditioning varies by 40x across the fixture.
    const TOL_EPS_MULTIPLE: f64 = 2.0;

    async fn device() -> burn::tensor::Device {
        brush_cube::test_helpers::test_device().await.into()
    }

    async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .to_vec::<f32>()
            .expect("f32 tensor")
    }

    /// `(fx, fy, cx, cy)`, bundled so the runner keeps a readable arity.
    type Intrinsics = (f32, f32, f32, f32);

    async fn run(
        data: Vec<f32>,
        (h, w): (usize, usize),
        (fx, fy, cx, cy): Intrinsics,
        device: &burn::tensor::Device,
    ) -> (Vec<f32>, Vec<f32>) {
        let feat = Tensor::<3>::from_data(TensorData::new(data, [h, w, 5]), device);
        let (depth, _normal, valid) = plane_depth_from_features(
            feat, fx, fy, cx, cy, MIN_ALPHA, MIN_DENOM, MIN_DEPTH, MAX_DEPTH,
        );
        (read(depth).await, read(valid).await)
    }

    /// A tiny deterministic PRNG. The reference randomises with `hypothesis`;
    /// we have no such dependency and would not want a differently-shrunk
    /// counterexample on every CI run anyway — a fixed seed makes a failure
    /// reproducible from the test name alone.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f64 {
            // Knuth/MMIX constants.
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn norm3(v: [f64; 3]) -> [f64; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    }

    fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// **§10d item 1.** 64 randomised plane/ray constructions, each with a
    /// prescribed `n·ray`, compared against an f64 oracle.
    ///
    /// Structure ported from the reference's 64-example `hypothesis` property
    /// test. Two deliberate deviations, both stated so a reader can judge them:
    ///
    /// 1. **Camera-frame only.** The reference randomises a world rotation and
    ///    translation because its `gaussian_plane_features` runs inside the
    ///    property. `plane_depth_from_features` never sees the world frame — it
    ///    consumes `n_cam` and `d` — so a random rotation would test nothing
    ///    here. The world→camera half is pinned separately, and by VALUE, in
    ///    brush-train (`plane_feature_tests`).
    /// 2. **One pixel per case, 64 pixels in one image.** Every case gets a
    ///    different ray because the ray grid varies across the frame, which is
    ///    the same coverage the reference buys with a per-case principal-point
    ///    offset — and it costs one dispatch instead of 64.
    ///
    /// The first eight pixels pin the near-cutoff multipliers the reference
    /// samples explicitly (±1.01x, ±1.1x, ±2x the cutoff, plus ±1.5x); the rest
    /// spread over the well-conditioned range. Each carries a random coverage
    /// weight, so the "no alpha division" cancellation is exercised here too.
    ///
    /// # Mutation-checked, 2026-08-20
    ///
    /// Injecting a 1e-5 RELATIVE depth error (`depth_raw * (1.0 + 1e-5)`) into
    /// `plane_depth_from_features`:
    ///
    /// - `plane_depth_flat_slab_exact`, tolerance 1e-4: **passes**. The
    ///   regression is invisible to it.
    /// - this test: **fails** at 24.5 conditioning-epsilons against a budget of
    ///   2.0, i.e. 12x over.
    ///
    /// That pair IS §10d item 1: the flat tolerance would miss a real accuracy
    /// regression, and this bound would not.
    #[tokio::test]
    async fn plane_depth_near_grazing_matches_a_float64_oracle() {
        let device = device().await;

        // Multiples of MIN_DENOM. 1.01 is the reference's tightest case: just
        // inside the cutoff, where the quotient is worst-conditioned.
        const PINNED: [f64; 8] = [1.01, -1.01, 1.1, -1.1, 1.5, -1.5, 2.0, -2.0];

        let mut rng = Lcg(0x5eed_1234_abcd_0001);
        let mut data = vec![0.0f32; H * W * 5];
        // Per-case bookkeeping for the assertions: (oracle depth, |ray|, denom).
        let mut cases: Vec<(f64, f64, f64)> = Vec::with_capacity(H * W);

        for v in 0..H {
            for u in 0..W {
                let k = v * W + u;

                // The ray this pixel's centre subtends, in f64.
                let ru = (u as f64 + 0.5 - f64::from(CX)) / f64::from(FX);
                let rv = (v as f64 + 0.5 - f64::from(CY)) / f64::from(FY);
                let ray = [ru, rv, 1.0];
                let ray_len = (ru * ru + rv * rv + 1.0).sqrt();
                let ray_unit = [ru / ray_len, rv / ray_len, 1.0 / ray_len];

                // An orthonormal pair spanning the plane perpendicular to the
                // ray, so the normal's transverse part can point anywhere.
                let p0 = norm3([1.0, 0.0, -ru]);
                let p1 = cross3(ray_unit, p0);
                let theta = rng.range(0.0, std::f64::consts::TAU);
                let (st, ct) = theta.sin_cos();
                let perp = [
                    p0[0] * ct + p1[0] * st,
                    p0[1] * ct + p1[1] * st,
                    p0[2] * ct + p1[2] * st,
                ];

                // A varying coverage weight. NOTE, and this is not obvious:
                // `min_denom` is compared against the ALPHA-WEIGHTED composited
                // sum, not against the geometric `n·ray` — the weight cancels
                // out of the quotient but NOT out of the validity test. So the
                // geometric normal is tilted to `denom/w` in order to land the
                // STORED denominator on the value this case means to pin. (An
                // earlier draft set the geometric denominator instead and every
                // near-cutoff case was rejected at `w ~ 0.7`, which is how the
                // coupling surfaced.)
                let w = rng.range(0.55, 0.95);

                // Prescribe the stored `n_sum · ray` exactly: with |n| = 1, the
                // component of n along the unit ray is (denom/w)/|ray|.
                let denom = if k < PINNED.len() {
                    PINNED[k] * f64::from(MIN_DENOM)
                } else {
                    // Capped at 0.5 (not the reference's 0.95) so `denom/w`
                    // stays inside the unit sphere for every weight above.
                    let mag = rng.range(2.0 * f64::from(MIN_DENOM), 0.5);
                    if rng.unit() < 0.5 { -mag } else { mag }
                };
                let c = denom / w / ray_len;
                assert!(c.abs() <= 1.0, "case {k}: unrealisable tilt {c}");
                let s = (1.0 - c * c).sqrt();
                let n = [
                    c * ray_unit[0] + s * perp[0],
                    c * ray_unit[1] + s * perp[1],
                    c * ray_unit[2] + s * perp[2],
                ];

                // The oracle: put the plane through a point at a known depth
                // along this very ray, so `offset = n·(ray·z)`.
                let depth = rng.range(0.25, 20.0);
                let offset = dot3(n, ray) * depth;

                // f64 self-consistency of the construction itself — this is the
                // reference's `assert_close(means @ n, plane_offset)` step. It
                // catches a typo in the basis math that would otherwise be
                // absorbed silently into the f32 tolerance below.
                let stored_denom = dot3(n, ray) * w;
                assert!(
                    (stored_denom - denom).abs() < 1e-12,
                    "case {k}: constructed stored n_sum·ray = {stored_denom}, want {denom}"
                );
                let oracle = offset / dot3(n, ray);
                assert!(
                    (oracle - depth).abs() < 1e-9 * depth,
                    "case {k}: f64 oracle {oracle} disagrees with the construction depth {depth}"
                );

                let i = k * 5;
                data[i] = (w * n[0]) as f32;
                data[i + 1] = (w * n[1]) as f32;
                data[i + 2] = (w * n[2]) as f32;
                data[i + 3] = (w * offset) as f32;
                data[i + 4] = w as f32;

                cases.push((oracle, ray_len, denom));
            }
        }

        let (depth, valid) = run(data, (H, W), (FX, FY, CX, CY), &device).await;

        // The measured error, expressed in units of the predicted conditioning
        // bound. Reported at the end so the headroom is visible in the failure
        // message rather than folded into a pass/fail.
        let mut worst_ratio = 0.0f64;
        let mut worst_case = (0usize, 0.0f64, 0.0f64);

        for (k, &(oracle, ray_len, denom)) in cases.iter().enumerate() {
            assert_eq!(
                valid[k], 1.0,
                "case {k} (denom {denom}, depth {oracle}) must be valid: \
                 every generated denominator is strictly above the cutoff"
            );

            let err = (f64::from(depth[k]) - oracle).abs();
            // Predicted relative error: `eps` from rounding the offset channel,
            // plus `eps·‖ray‖/|denom|` from the cancellation in `n·ray`.
            let conditioning = 1.0 + ray_len / denom.abs();
            let unit_bound = f64::from(f32::EPSILON) * conditioning * oracle.abs();
            let ratio = err / unit_bound;
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst_case = (k, err, oracle);
            }
        }

        assert!(
            worst_ratio < TOL_EPS_MULTIPLE,
            "worst error was {worst_ratio:.2} conditioning-epsilons \
             (case {}, absolute error {:.3e} on an oracle depth of {:.4}); \
             the budget is {TOL_EPS_MULTIPLE}. A ratio that grew without the \
             conditioning changing means the SOLVE regressed, not the float type.",
            worst_case.0,
            worst_case.1,
            worst_case.2,
        );

        // Guard the guard: if the measured worst case ever drops far below the
        // budget, the budget is stale and should be tightened rather than left
        // as dead slack. Measured 2026-08-20 on the M4 Max: worst ratio 0.32 —
        // an absolute error of 2.70e-6 on an oracle depth of 9.98, i.e. 37x
        // INSIDE the flat 1e-4 that `plane_depth_flat_slab_exact` allows, which
        // is the gap §10d item 1 exists to close.
        assert!(
            worst_ratio > 0.02,
            "worst ratio {worst_ratio:.4} is implausibly small — the fixture \
             probably degenerated (e.g. every denominator landed well away from \
             the cutoff) and stopped testing the grazing regime"
        );
    }

    /// **§10d item 1, second half.** Both sides of the validity cutoff, on the
    /// reference's own construction.
    ///
    /// Ported from `test_plane_depth_uses_the_documented_degenerate_cutoff`:
    /// `fx = fy = 1`, `cx = cy = 0`, so pixel `(u, 0)` subtends the ray
    /// `(u + 0.5, 0.5, 1)`. Choosing `n = (1, 0, denom − (u + 0.5))` makes
    /// `n·ray` equal `denom` and `offset = 2·denom` makes the depth exactly 2 —
    /// the same trick the reference uses, so a divergence between the two
    /// suites is a real disagreement and not a fixture difference.
    ///
    /// Half the cutoff must be rejected; twice it must be accepted AND give the
    /// right depth. Rejecting both would pass a naive "grazing rays are invalid"
    /// test while making the whole plane path dead.
    #[tokio::test]
    async fn plane_depth_pins_both_sides_of_the_denominator_cutoff() {
        let device = device().await;

        let below = MIN_DENOM / 2.0;
        let above = MIN_DENOM * 2.0;
        let data = vec![
            // (0, 0): ray (0.5, 0.5, 1), n·ray = below -> rejected.
            1.0,
            0.0,
            below - 0.5,
            2.0 * below,
            1.0,
            // (1, 0): ray (1.5, 0.5, 1), n·ray = above -> accepted, depth 2.
            1.0,
            0.0,
            above - 1.5,
            2.0 * above,
            1.0,
        ];

        let (depth, valid) = run(data, (1, 2), (1.0, 1.0, 0.0, 0.0), &device).await;

        assert_eq!(
            valid[0], 0.0,
            "|denom| = {below} is half the {MIN_DENOM} cutoff and must be rejected"
        );
        assert_eq!(depth[0], 0.0, "a rejected pixel must emit exactly 0");
        assert_eq!(
            valid[1], 1.0,
            "|denom| = {above} is twice the {MIN_DENOM} cutoff and must be accepted"
        );
        // `n_z` is formed as `above - 1.5` and the kernel re-adds 1.5, so the
        // recovered denominator carries one ulp of 1.5 (~1.2e-7) of absolute
        // error, i.e. ~1.2e-6 relative at |denom| = 0.1. 1e-4 on a depth of 2 is
        // ~40x that; it is a bound on the FIXTURE's arithmetic, not a licence
        // for the solve to be sloppy — the tight statement lives in the
        // grazing property test above.
        assert!(
            (depth[1] - 2.0).abs() < 1e-4,
            "depth just above the cutoff = {}, want 2.0",
            depth[1]
        );
    }

    /// The cutoff comparison is INCLUSIVE: `|denom| == min_denom` is valid.
    ///
    /// `denom_ok` is written as `!(|denom| < min_denom)`, so the boundary itself
    /// passes. Flipping it to `>` would silently shave the outermost grazing
    /// pixels off every plane in every frame — a change no forward-parity test
    /// would notice. Pinned on the one construction where f32 can hit the
    /// boundary EXACTLY: `cx = cy = 0.5, fx = fy = 1` makes pixel (0, 0)'s ray
    /// exactly `(0, 0, 1)`, so `denom = 0·0 + 0·0 + n_z` reproduces `n_z`
    /// bit-for-bit with no rounding to argue about.
    #[tokio::test]
    async fn plane_depth_cutoff_comparison_is_inclusive() {
        let device = device().await;

        let data = vec![0.0, 0.0, MIN_DENOM, 2.0 * MIN_DENOM, 1.0];
        let (depth, valid) = run(data, (1, 1), (1.0, 1.0, 0.5, 0.5), &device).await;

        assert_eq!(
            valid[0], 1.0,
            "|denom| exactly at the {MIN_DENOM} cutoff must be VALID \
             (the test is `!(|denom| < min_denom)`)"
        );
        // 2·MIN_DENOM / MIN_DENOM: doubling is exact in binary floating point,
        // so this quotient is exactly 2 with no tolerance needed.
        assert_eq!(depth[0], 2.0);
    }

    /// **§10d item 3.** A plane BEHIND the camera is invalid, emits exactly 0,
    /// and stays finite — on the reference's literal construction.
    ///
    /// The reference writes it as `n = (0, 0, 1)`, `offset = −2`. Our sign
    /// convention is the opposite (`splat_normals` faces the normal AT the
    /// camera, so a visible plane has `n_z < 0` and a negative offset), and
    /// `plane_depth_invalid_pixels_are_zero_and_nan_free` above already covers
    /// OUR spelling of it at pixel (4, 0): `n = (0, 0, −1)`, `offset = +3`.
    /// Both give a negative quotient; this test pins the reference's spelling
    /// too, so the two suites cannot silently disagree about which sign pair
    /// means "behind me".
    ///
    /// `min_depth > 0` is the whole mechanism: an `|z| < max_depth` range test
    /// would happily accept z = −2, and every downstream consumer would then
    /// supervise a mirrored surface.
    #[tokio::test]
    async fn plane_depth_rejects_the_reference_behind_camera_plane() {
        let device = device().await;

        // fx = fy = 1, cx = cy = 0.5 -> the single pixel's ray is (0, 0, 1),
        // so denom = n_z = +1 and depth = offset = -2.
        for (n_z, offset, label) in [
            (1.0f32, -2.0f32, "reference convention (n = +z, d < 0)"),
            (-1.0, 2.0, "our convention (n = -z, d > 0)"),
        ] {
            let data = vec![0.0, 0.0, n_z, offset, 1.0];
            let (depth, valid) = run(data, (1, 1), (1.0, 1.0, 0.5, 0.5), &device).await;

            assert_eq!(
                valid[0], 0.0,
                "{label}: a plane behind the camera is invalid"
            );
            assert_eq!(depth[0], 0.0, "{label}: depth must be exactly 0");
            assert!(depth[0].is_finite(), "{label}: depth must be finite");
        }
    }
}

/// Masked-mean semantics and the contradiction gate, ported from the reference
/// suite (`gauss-surf`, Apache-2.0, Pablo Vela: `tests/test_gaussurf_losses.py`).
///
/// Everything here is about pixels that are supposed to contribute NOTHING —
/// the reference pins that as a bit-identity property rather than an
/// approximate one, and it is worth having because "outside the mask" is
/// exactly where garbage lives: uncovered background, priors the writer marked
/// absent, and the amplified noise of `raw / alpha.clamp_min(1e-10)`.
#[cfg(all(test, not(target_family = "wasm")))]
mod masked_mean_and_gate_tests {
    use super::*;
    use brush_render::burn_glue::lift_to_autodiff;
    use burn::tensor::TensorData;

    async fn device() -> burn::tensor::Device {
        brush_cube::test_helpers::test_device().await.into()
    }

    async fn autodiff_device() -> burn::tensor::Device {
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff()
    }

    async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .to_vec::<f32>()
            .expect("f32 tensor")
    }

    fn img3(data: Vec<f32>, device: &burn::tensor::Device) -> Tensor<3> {
        let n = data.len() / 3;
        Tensor::<3>::from_data(TensorData::new(data, [1, n, 3]), device)
    }

    fn img1(data: Vec<f32>, device: &burn::tensor::Device) -> Tensor<3> {
        let n = data.len();
        Tensor::<3>::from_data(TensorData::new(data, [1, n, 1]), device)
    }

    /// An explicitly-shaped `[H, W, C]` image, for the smoothness term, which
    /// needs a real 2-D neighbourhood rather than the `[1, N, C]` strip the
    /// other helpers build.
    fn grid3(data: Vec<f32>, shape: [usize; 3], device: &burn::tensor::Device) -> Tensor<3> {
        Tensor::<3>::from_data(TensorData::new(data, shape), device)
    }

    fn img2(data: Vec<f32>, device: &burn::tensor::Device) -> Tensor<2> {
        let n = data.len();
        Tensor::<2>::from_data(TensorData::new(data, [1, n]), device)
    }

    /// **The hypothesis, in test form** (plan §10f, §11.2).
    ///
    /// The measured plane-fused opacity collapse is attributed to the disparity
    /// gradient's `1/d²` scaling: a near splat receives orders of magnitude
    /// more depth gradient than a far one, and with the blending-weight
    /// gradients live, fading it is a cheaper descent direction than rotating
    /// it. Metric L1 cannot create that pressure, because its per-pixel
    /// gradient magnitude is a constant.
    ///
    /// Two valid pixels at very different ranges (`pred = 2 m` and
    /// `pred = 20 m`, each 10% beyond its GT so both residuals are
    /// same-signed), denominator 2:
    ///
    /// ```text
    ///   metric    : d/d(pred) = sign(pred - gt) / 2      = +0.5   at BOTH pixels
    ///   disparity : d/d(pred) = sign(...) / (2 · pred²)  = +0.125 and +0.00125
    /// ```
    ///
    /// The SIGNS agree, and that is worth stating because an earlier draft of
    /// this test asserted they would not: the disparity chain flips twice
    /// (over-estimating depth UNDER-estimates disparity, and `d(1/d)/dd` is
    /// itself negative), so both losses push an over-estimated depth down. The
    /// two losses never disagree about direction.
    ///
    /// What they disagree about is MAGNITUDE, and the assertion is not merely
    /// "the numbers differ": it is that the metric gradient is
    /// RANGE-INDEPENDENT (both pixels bit-identical) while the disparity
    /// gradient spans the 100x that `1/d²` predicts across a 10x range. The
    /// ranges are 2 m and 20 m rather than 1 m and 10 m so that no cell of the
    /// 2x2 coincides numerically with another — at 1 m the disparity and metric
    /// gradients are both exactly 0.5, and a test in which two of the four
    /// values happen to be equal is a test that a fallthrough could pass.
    #[tokio::test]
    async fn depth_loss_gradient_is_range_independent_only_in_metric_space() {
        let device = autodiff_device().await;
        // pred is 10% beyond gt at both pixels, at ranges 2 m and 20 m.
        let gt = || img2(vec![2.0 / 1.1, 20.0 / 1.1], &device);

        let pred_m = lift_to_autodiff(img2(vec![2.0, 20.0], &device)).require_grad();
        let loss = depth_loss(
            pred_m.clone(),
            gt(),
            None,
            DepthLossSpace::Metric,
            DepthUncovered::Count,
        );
        let grads = loss.backward();
        let g_metric = read(pred_m.grad(&grads).expect("metric depth gradient")).await;

        let pred_d = lift_to_autodiff(img2(vec![2.0, 20.0], &device)).require_grad();
        let loss = depth_loss(
            pred_d.clone(),
            gt(),
            None,
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        );
        let grads = loss.backward();
        let g_disp = read(pred_d.grad(&grads).expect("disparity depth gradient")).await;

        // Metric: +1/N at both pixels, N = 2 valid pixels.
        for (i, g) in g_metric.iter().enumerate() {
            assert!(
                (g - 0.5).abs() < 1e-5,
                "metric gradient at pixel {i} is {g}, want +0.5 (constant magnitude, \
                 independent of range)"
            );
        }

        // Disparity: +1/(N·pred²) -> 0.125 at 2 m, 0.00125 at 20 m.
        assert!(
            (g_disp[0] - 0.125).abs() < 1e-5,
            "disparity gradient at 2 m is {}, want +0.125",
            g_disp[0]
        );
        assert!(
            (g_disp[1] - 0.00125).abs() < 1e-5,
            "disparity gradient at 20 m is {}, want +0.00125",
            g_disp[1]
        );

        // The headline: 100x spread across a 10x range change, vs none.
        let metric_ratio = (g_metric[0] / g_metric[1]).abs();
        let disp_ratio = (g_disp[0] / g_disp[1]).abs();
        assert!(
            (metric_ratio - 1.0).abs() < 1e-4,
            "metric near/far gradient ratio {metric_ratio}, want 1.0"
        );
        assert!(
            (disp_ratio - 100.0).abs() < 1e-2,
            "disparity near/far gradient ratio {disp_ratio}, want 100 (1/d^2 over a 10x range)"
        );
    }

    /// **§10d item 4.** A pixel outside a loss's validity mask has NO numerical
    /// effect on that loss — not "a negligible one", none: the results must be
    /// bit-identical.
    ///
    /// Ported from `test_masked_mean_is_bit_identical_after_outside_mask_perturbations`.
    /// This is the property that makes a masked mean a mean over the mask rather
    /// than a mean over the frame that happens to be dominated by the mask, and
    /// it is the one that catches a denominator counting the wrong pixels: if
    /// the divisor were the pixel count instead of the valid count, the value
    /// would still be stable here, but a masked-out pixel leaking into the
    /// NUMERATOR (a missing `* valid`, a `+` where a `*` belongs, an off-by-one
    /// slice) changes the result the moment that pixel stops being ordinary.
    ///
    /// ±1e6 is deliberately far outside the range any of these signals can
    /// legitimately take, so a leak cannot hide inside a tolerance. The
    /// non-finite half of the reference's contract is a separate test.
    ///
    /// # Mutation-checked, 2026-08-20
    ///
    /// Replacing `* valid` with `* valid.add_scalar(1e-9)` in `normal_loss` —
    /// a leak nine orders of magnitude below anything a tolerance-based test
    /// would notice — **fails** this test on the `normal_loss` arm. The ±1e6
    /// perturbation is what converts that invisible floor into a visible one.
    #[tokio::test]
    async fn losses_ignore_finite_perturbations_outside_the_mask() {
        let device = device().await;
        const WILD: f32 = 1.0e6;

        // --- normal_loss: validity comes from the PRIOR, so the prediction at
        // an absent-prior pixel is what gets perturbed.
        let gt_n = img3(
            vec![
                0.0, 0.0, -1.0, // valid prior
                0.0, 0.0, -1.0, // valid prior
                0.0, 0.0, 0.0, // (0,0,0) = "no prior here"
                0.0, 0.0, 0.0, // ditto
            ],
            &device,
        );
        let tame = vec![
            0.0, 0.2, -1.0, //
            0.1, 0.0, -1.0, //
            0.0, 0.0, -1.0, //
            0.3, -0.4, -1.0, //
        ];
        let mut wild = tame.clone();
        wild[6..9].copy_from_slice(&[WILD, -WILD, WILD]);
        wild[9..12].copy_from_slice(&[-WILD, WILD, -WILD]);

        let base = read(normal_loss(img3(tame, &device), gt_n.clone(), None)).await[0];
        let perturbed = read(normal_loss(img3(wild, &device), gt_n, None)).await[0];
        assert_eq!(
            base, perturbed,
            "normal_loss changed when pixels with NO prior were perturbed"
        );
        assert!(base > 0.0, "the fixture must produce a nonzero loss");

        // --- depth_loss: validity is `gt > 0`.
        let gt_d = img2(vec![2.0, 3.0, 0.0, 0.0], &device);
        let tame = vec![1.9, 3.2, 1.0, 5.0];
        let wild = vec![1.9, 3.2, WILD, 1.0 / WILD];

        let base = read(depth_loss(
            img2(tame, &device),
            gt_d.clone(),
            None,
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        ))
        .await[0];
        let perturbed = read(depth_loss(
            img2(wild, &device),
            gt_d,
            None,
            DepthLossSpace::Disparity,
            DepthUncovered::Count,
        ))
        .await[0];
        assert_eq!(
            base, perturbed,
            "depth_loss changed when pixels with no GT depth were perturbed"
        );
        assert!(base > 0.0, "the fixture must produce a nonzero loss");

        // --- depth_normal_loss: validity is `alpha > 0.5`, so the perturbation
        // goes into the two uncovered pixels' normals.
        let alpha = img1(vec![1.0, 1.0, 0.0, 0.0], &device);
        let n_from_depth = img3(
            vec![
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
            ],
            &device,
        );
        let tame = vec![
            0.0, 0.0, -1.0, //
            1.0, 0.0, 0.0, //
            0.0, 0.0, -1.0, //
            0.0, 1.0, 0.0, //
        ];
        let mut wild = tame.clone();
        wild[6..12].copy_from_slice(&[WILD, -WILD, WILD, -WILD, WILD, -WILD]);

        let base = read(depth_normal_loss(
            n_from_depth.clone(),
            img3(tame, &device),
            alpha.clone(),
        ))
        .await[0];
        let perturbed = read(depth_normal_loss(n_from_depth, img3(wild, &device), alpha)).await[0];
        assert_eq!(
            base, perturbed,
            "depth_normal_loss changed when UNCOVERED pixels were perturbed"
        );
        assert!(base > 0.0, "the fixture must produce a nonzero loss");
    }

    /// **§10d item 4, second half.** A NON-FINITE value outside the mask must
    /// contribute exact zero, forward and backward.
    ///
    /// Ported from `test_masked_out_nonfinite_values_do_not_poison_loss_or_gradients`.
    /// This is the contract that forbids implementing a masked mean as
    /// `value * mask`: `0.0 * inf` is `NaN`, so one non-finite value in a region
    /// the loss is supposed to ignore takes down the whole frame — and autodiff
    /// reproduces the same `0 · ∞` in the VJP even when the forward is repaired
    /// afterwards. The fix is to substitute the value BEFORE the arithmetic,
    /// which is the idiom `plane_depth_from_features` already used and which the
    /// three masked-mean losses now use too (see the `NON-FINITE DISCIPLINE`
    /// comments at each site).
    ///
    /// 48 (loss x poison) combinations: `normal_loss`, `depth_loss`
    /// (disparity bare, disparity with a poisoned per-pixel weight, metric with
    /// a poisoned PREDICTION) x 3 poisons, `depth_normal_loss` and
    /// `normal_smooth_loss` x 3, plus a poisoned GT in both spaces x the 2
    /// poisons that fall outside the `gt > 0` mask x the 3 `--depth-uncovered`
    /// modes, plus a poisoned PREDICTION x 3 poisons x 2 spaces x 3 modes.
    ///
    /// The 22 -> 48 growth is the `--depth-uncovered` sweep (plan §5.4). It is
    /// not padding: the coverage mask is derived FROM the prediction, so a
    /// poisoned prediction is an input to the mask that is supposed to contain
    /// it, and `Exclude` additionally moves the DENOMINATOR, which is a second
    /// place a `NaN` can reach the result.
    ///
    /// The metric and poisoned-GT combinations were added with
    /// `--depth-loss-space` (WS-M). The metric arm has neither of the
    /// accidental protections the disparity arm enjoys — no `recip()` to map
    /// `+inf` to 0, no `pred <= 0` guard to catch `-inf` — so its `mask_fill`s
    /// are the whole defence. **The poisoned-GT half found a live hole in the
    /// shared masking** that predates this change and affected the disparity
    /// path equally: see the `gt_invalid = !gt_valid` comment in `depth_loss`.
    ///
    /// # History — this test was red when it was written, and that was the point
    ///
    /// As first measured on 2026-08-20, **8 of the original 9 (loss × poison)
    /// combinations failed**, because every masked mean ended in `err * valid`:
    ///
    /// ```text
    ///   normal_loss       [inf/-inf/NaN]: forward = NaN, want 0.06666667
    ///   depth_normal_loss [inf/-inf/NaN]: forward = NaN, want 1
    ///   depth_loss        [NaN]         : forward = NaN, want 0.026315808
    ///   depth_loss        [NaN]         : backward = [-0.27700832, NaN]
    /// ```
    ///
    /// The single survivor was `depth_loss` under ±inf, and only by accident of
    /// its shape: `recip()` maps `+inf` to 0 and the `pred <= 0` guard catches
    /// `-inf` before the multiply. `NaN` defeated both, because every comparison
    /// against `NaN` is false and `mask_fill` therefore never fired.
    ///
    /// Note WHERE the damage showed, since it is the part that makes this worth
    /// a contract rather than a shrug: for `normal_loss` the backward stayed
    /// finite while the forward was `NaN` (`d|x|/dx · valid` is an honest 0), so
    /// the poison travelled as a `NaN` SCALAR LOSS. Summed into the total it
    /// takes the gradient of every OTHER term in the step with it — a worse
    /// failure than a local one, and an easy one to misattribute.
    ///
    /// Not reachable from a healthy render today (the rendered normal is
    /// `normal_img / alpha.clamp_min(1e-10)` re-normalised and the rendered
    /// depth is `accum / alpha.clamp_min(1e-10)`; neither division reaches `inf`
    /// at rasterizer magnitudes), so this is a robustness contract rather than a
    /// live-bug regression test — which is exactly why it needs a test and not a
    /// runtime symptom.
    ///
    /// The other two members of the family were then swept in rather than filed:
    /// `normal_smooth_loss` (whose own doc comment already says an uncovered
    /// pixel holds amplified noise from `raw / alpha.clamp_min(1e-10)`, so it
    /// was the most exposed of the five) and `depth_loss`'s optional per-pixel
    /// weight. Both were vulnerable to ±inf as well as `NaN`.
    ///
    /// # Mutation-checked, 2026-08-20
    ///
    /// - Reverting the three original `mask_fill` calls returns all 8 original
    ///   failures verbatim.
    /// - Reverting only the two later ones fails 6 of 15 — every
    ///   `normal_smooth_loss` and `depth_loss[weighted]` combination — which is
    ///   what makes those two hunks load-bearing rather than defensive.
    #[tokio::test]
    async fn losses_ignore_nonfinite_values_outside_the_mask() {
        let device = autodiff_device().await;
        let mut violations: Vec<String> = Vec::new();

        for poison in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut check = |what: &str, got: f32, want: f32, grads: Vec<f32>| {
                // `partial_cmp`, not `>= 1e-6`: the value under test is very
                // often `NaN`, and every ordinary comparison against `NaN` is
                // false — a `>=` test would silently report the exact failure
                // this test exists to find as a pass.
                let close =
                    (got - want).abs().partial_cmp(&1e-6) == Some(core::cmp::Ordering::Less);
                if !close {
                    violations.push(format!("{what} [{poison}]: forward = {got}, want {want}"));
                }
                if !grads.iter().all(|x| x.is_finite()) {
                    violations.push(format!("{what} [{poison}]: backward = {grads:?}"));
                }
            };

            // --- normal_loss, on a prior-absent pixel.
            let gt = img3(
                vec![
                    0.0, 0.0, -1.0, // valid prior
                    0.0, 0.0, 0.0, // no prior here
                ],
                &device,
            );
            let pred =
                lift_to_autodiff(img3(vec![0.0, 0.2, -1.0, poison, poison, poison], &device))
                    .require_grad();
            let loss = normal_loss(pred.clone(), gt, None);
            let value = read(loss.clone()).await[0];
            let grads = loss.backward();
            let g = read(pred.grad(&grads).expect("prediction gradient")).await;
            check("normal_loss", value, 0.2 / 3.0, g);

            // --- depth_loss, on a pixel with no GT depth.
            let gt_d = img2(vec![2.0, 0.0], &device);
            let pred_d = lift_to_autodiff(img2(vec![1.9, poison], &device)).require_grad();
            let loss = depth_loss(
                pred_d.clone(),
                gt_d,
                None,
                DepthLossSpace::Disparity,
                DepthUncovered::Count,
            );
            let value = read(loss.clone()).await[0];
            let grads = loss.backward();
            let g = read(pred_d.grad(&grads).expect("depth gradient")).await;
            check("depth_loss", value, (1.0f32 / 1.9 - 1.0 / 2.0).abs(), g);

            // --- depth_loss in METRIC space, same pixel. The disparity arm
            // survives ±inf by accident of its shape (`recip()` maps `+inf` to
            // 0 and the `pred <= 0` guard catches `-inf`); the metric arm has
            // neither, so its ONLY protection is the pair of `mask_fill`s, and
            // a poisoned GT reaches the subtraction directly. That makes these
            // two combinations the load-bearing ones for the metric branch.
            let gt_d = img2(vec![2.0, 0.0], &device);
            let pred_d = lift_to_autodiff(img2(vec![1.9, poison], &device)).require_grad();
            let loss = depth_loss(
                pred_d.clone(),
                gt_d,
                None,
                DepthLossSpace::Metric,
                DepthUncovered::Count,
            );
            let value = read(loss.clone()).await[0];
            let grads = loss.backward();
            let g = read(pred_d.grad(&grads).expect("metric depth gradient")).await;
            check("depth_loss[metric]", value, (1.9f32 - 2.0).abs(), g);

            // --- and with the GT ITSELF poisoned. This is the case the
            // pre-existing test never covered — it perturbs predictions and
            // weights only — and it is how the `gt_invalid = !gt_valid`
            // spelling was found to be load-bearing: under the old
            // `gt <= 0`, a `NaN` GT satisfied NEITHER comparison (every
            // comparison against `NaN` is false), so it was neither supervised
            // nor substituted, rode into the arithmetic and met `* valid` as
            // `NaN · 0`. Both spaces were affected; the metric arm is simply
            // where it was noticed.
            //
            // `+inf` is EXCLUDED, and not as a convenience: `inf > 0` is true,
            // so an infinite GT depth is a SUPERVISED pixel under this
            // function's contract, and demanding it be ignored would be
            // asserting a different contract than the one the code states.
            // `NaN` and `-inf` are both genuinely outside the mask.
            //
            // **Swept across all three `--depth-uncovered` modes** (plan §5.4).
            // The poisoned pixel here is COVERED (`pred = 1.0 > 0`) but
            // GT-invalid, so it leaves through `gt_valid` in every mode and the
            // expected value is mode-independent — which is the point: the new
            // coverage mask must not create a second, weaker path for a
            // poisoned GT to survive, and `Exclude`'s narrowed denominator must
            // still count exactly the one supervised pixel.
            if poison != f32::INFINITY {
                for space in [DepthLossSpace::Disparity, DepthLossSpace::Metric] {
                    for uncovered in [
                        DepthUncovered::Count,
                        DepthUncovered::ExcludeNumerator,
                        DepthUncovered::Exclude,
                    ] {
                        let gt_d = img2(vec![2.0, poison], &device);
                        let pred_d = lift_to_autodiff(img2(vec![1.9, 1.0], &device)).require_grad();
                        let loss = depth_loss(pred_d.clone(), gt_d, None, space, uncovered);
                        let value = read(loss.clone()).await[0];
                        let grads = loss.backward();
                        let g = read(pred_d.grad(&grads).expect("depth gradient")).await;
                        let want = match space {
                            DepthLossSpace::Disparity => (1.0f32 / 1.9 - 1.0 / 2.0).abs(),
                            DepthLossSpace::Metric => (1.9f32 - 2.0).abs(),
                        };
                        check(
                            &format!("depth_loss[{space:?}, {uncovered:?}, poisoned gt]"),
                            value,
                            want,
                            g,
                        );
                    }
                }
            }

            // --- POISONED PREDICTION at a pixel with no GT depth, swept across
            // all three `--depth-uncovered` modes (plan §5.4, the other half).
            //
            // Note what makes this non-trivial rather than a re-run of the two
            // `depth_loss` checks above: the coverage mask is derived FROM the
            // prediction, so a poisoned prediction is an input to the new mask
            // itself. `!(pred > 0)` is false for `+inf` (so `+inf` is "covered"
            // and must be caught by the GT mask, as before) and true for `NaN`
            // and `-inf`. All three modes must survive all three poisons in
            // both spaces regardless of which mask does the catching.
            for space in [DepthLossSpace::Disparity, DepthLossSpace::Metric] {
                for uncovered in [
                    DepthUncovered::Count,
                    DepthUncovered::ExcludeNumerator,
                    DepthUncovered::Exclude,
                ] {
                    let gt_d = img2(vec![2.0, 0.0], &device);
                    let pred_d = lift_to_autodiff(img2(vec![1.9, poison], &device)).require_grad();
                    let loss = depth_loss(pred_d.clone(), gt_d, None, space, uncovered);
                    let value = read(loss.clone()).await[0];
                    let grads = loss.backward();
                    let g = read(pred_d.grad(&grads).expect("depth gradient")).await;
                    let want = match space {
                        DepthLossSpace::Disparity => (1.0f32 / 1.9 - 1.0 / 2.0).abs(),
                        DepthLossSpace::Metric => (1.9f32 - 2.0).abs(),
                    };
                    check(
                        &format!("depth_loss[{space:?}, {uncovered:?}, poisoned pred]"),
                        value,
                        want,
                        g,
                    );
                }
            }

            // --- depth_normal_loss, on an UNCOVERED pixel.
            let alpha = img1(vec![1.0, 0.0], &device);
            let n_d = img3(vec![0.0, 0.0, -1.0, 0.0, 0.0, -1.0], &device);
            let n_r = lift_to_autodiff(img3(vec![1.0, 0.0, 0.0, poison, poison, poison], &device))
                .require_grad();
            let loss = depth_normal_loss(n_d, n_r.clone(), alpha);
            let value = read(loss.clone()).await[0];
            let grads = loss.backward();
            let g = read(n_r.grad(&grads).expect("rendered-normal gradient")).await;
            check("depth_normal_loss", value, 1.0, g);

            // --- normal_smooth_loss, on an UNCOVERED pixel. A 2x2 (its row and
            // column differences both need two rows AND two columns, or the
            // function early-returns a constant and the test proves nothing):
            // three covered pixels that agree, and one uncovered + poisoned. So
            // there are two covered-covered differences to score, both 0, and
            // two that touch the poisoned pixel and must be dropped.
            let alpha = grid3(vec![1.0, 1.0, 1.0, 0.0], [2, 2, 1], &device);
            let n_s = lift_to_autodiff(grid3(
                vec![
                    0.0, 0.0, -1.0, //
                    0.0, 0.0, -1.0, //
                    0.0, 0.0, -1.0, //
                    poison, poison, poison,
                ],
                [2, 2, 3],
                &device,
            ))
            .require_grad();
            let loss = normal_smooth_loss(n_s.clone(), alpha);
            let value = read(loss.clone()).await[0];
            let grads = loss.backward();
            let g = read(n_s.grad(&grads).expect("smoothness gradient")).await;
            check("normal_smooth_loss", value, 0.0, g);

            // --- depth_loss's optional per-pixel weight, poisoned outside the
            // mask. `abs_err` is already 0 there, so this is purely about the
            // weight multiply.
            let gt_d = img2(vec![2.0, 0.0], &device);
            let pred_d = lift_to_autodiff(img2(vec![1.9, 1.0], &device)).require_grad();
            let w = img2(vec![1.0, poison], &device);
            let loss = depth_loss(
                pred_d.clone(),
                gt_d,
                Some(w),
                DepthLossSpace::Disparity,
                DepthUncovered::Count,
            );
            let value = read(loss.clone()).await[0];
            let grads = loss.backward();
            let g = read(pred_d.grad(&grads).expect("weighted depth gradient")).await;
            check(
                "depth_loss[weighted]",
                value,
                (1.0f32 / 1.9 - 1.0 / 2.0).abs(),
                g,
            );
        }

        assert!(
            violations.is_empty(),
            "a non-finite value OUTSIDE the validity mask changed the loss or its \
             gradient in {} of 48 (loss x poison) combinations:\n  {}",
            violations.len(),
            violations.join("\n  ")
        );
    }

    /// **§10d item 5.** The contradiction gate decides WHICH pixels are
    /// supervised; it must never become a second gradient path into the
    /// rendered normals.
    ///
    /// Pinned without reaching into the implementation: gating a pixel out by
    /// CONTRADICTION and gating the same pixel out by marking its prior ABSENT
    /// must produce bit-identical gradients. The two routes share only the
    /// surviving set and the denominator; if the gate carried gradient — a
    /// softened threshold, a forgotten `detach()` on either operand — the gated
    /// run would pick up extra terms at exactly the contradicted pixels and the
    /// two would diverge.
    ///
    /// The paired assertion that the contradicted pixels receive EXACTLY zero
    /// gradient is what stops the test passing by both paths being equally
    /// wrong.
    ///
    /// # Mutation-checked, 2026-08-20 — including one mutation it does NOT catch
    ///
    /// - Replacing the hard threshold with a SOFT one
    ///   (`sigmoid(5·(cos − gate_cos))`) and dropping both `detach()` calls:
    ///   **fails**, on exactly this assertion.
    /// - Dropping both `detach()` calls and NOTHING else: **passes**.
    ///
    /// The second result is worth stating rather than hiding. The `detach()`
    /// calls in `normal_gate_mask` are documentary, not load-bearing: the
    /// gradient is already killed by `greater_equal_elem(..).float()`, a
    /// boolean comparison with no derivative, so removing the detaches changes
    /// no number anywhere and no behavioural test can see it. What this test
    /// pins is the PROPERTY (the gate is not on the tape), which is what
    /// actually matters and which the soft-gate mutation confirms it holds. Do
    /// not read it as pinning the `detach()` calls themselves.
    #[tokio::test]
    async fn normal_gate_carries_no_gradient() {
        let device = autodiff_device().await;
        let cos30 = 30.0_f32.to_radians().cos();

        // Pixels 0,1 agree with the prior to ~16.7 deg; pixels 2,3 are exactly
        // opposed, so a 30 deg gate drops them.
        let pred_data = vec![
            0.0, 0.3, -1.0, //
            0.0, 0.3, -1.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, //
        ];
        let all_valid = vec![
            0.0, 0.0, -1.0, //
            0.0, 0.0, -1.0, //
            0.0, 0.0, -1.0, //
            0.0, 0.0, -1.0, //
        ];
        // The same frame with the contradicted pixels' priors marked absent.
        let prior_absent = vec![
            0.0, 0.0, -1.0, //
            0.0, 0.0, -1.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
        ];

        async fn grad_of(
            pred_data: &[f32],
            gt_data: Vec<f32>,
            gate: Option<f32>,
            device: &burn::tensor::Device,
        ) -> Vec<f32> {
            let pred = lift_to_autodiff(img3(pred_data.to_vec(), device)).require_grad();
            let gt = img3(gt_data, device);
            let loss = normal_loss(pred.clone(), gt, gate);
            let grads = loss.backward();
            read(
                pred.grad(&grads)
                    .expect("the prediction must receive a gradient"),
            )
            .await
        }

        let gated = grad_of(&pred_data, all_valid, Some(cos30), &device).await;
        let prior_masked = grad_of(&pred_data, prior_absent, None, &device).await;

        assert_eq!(
            gated, prior_masked,
            "gating by contradiction and gating by an absent prior must produce \
             identical gradients — a difference means the gate is on the tape"
        );
        for (i, g) in gated.iter().enumerate().skip(6) {
            assert_eq!(
                *g, 0.0,
                "channel {i} of a gated-out pixel must carry exactly no gradient"
            );
        }
        let live: f32 = gated[..6].iter().map(|g| g.abs()).sum();
        assert!(
            live > 0.0,
            "the surviving pixels must carry a real gradient, else the equality \
             above is satisfied by an all-zero map"
        );
    }

    /// **§10d item 6.** The gate reduces the PRIOR term's pixel count without
    /// touching the depth/normal consistency mask.
    ///
    /// The two terms are meant to disagree about a contradicted pixel on
    /// purpose: the prior term stops trusting the external normal there, while
    /// the consistency term — which compares two of OUR OWN renders and needs no
    /// prior at all — must keep supervising it. Wiring the gate into both would
    /// quietly delete supervision from precisely the pixels where the geometry
    /// is least settled.
    ///
    /// Pinned by value rather than by signature: the fixture is built so that
    /// the consistency loss reads 1.0 with all four pixels counted and would
    /// read 0.0 if the gate's surviving set had leaked into it.
    #[tokio::test]
    async fn normal_gate_does_not_touch_the_depth_consistency_mask() {
        let device = device().await;
        let cos30 = 30.0_f32.to_radians().cos();

        // Depth-derived normals: all facing the camera.
        let n_from_depth = img3(
            vec![
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
            ],
            &device,
        );
        // Rendered normals: pixels 0,1 agree; pixels 2,3 are opposed.
        let rendered = vec![
            0.0, 0.0, -1.0, //
            0.0, 0.0, -1.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, //
        ];
        let prior = img3(
            vec![
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
            ],
            &device,
        );
        let alpha = img1(vec![1.0, 1.0, 1.0, 1.0], &device);

        // The gate really does bite on this frame: 2 of 4 survive.
        let counts = read(normal_gate_counts(
            img3(rendered.clone(), &device),
            prior.clone(),
            cos30,
        ))
        .await;
        assert_eq!((counts[0], counts[1]), (2.0, 4.0));

        // ...and the prior term's denominator follows it: |0| twice over
        // 2 pixels x 3 channels.
        let gated = read(normal_loss(
            img3(rendered.clone(), &device),
            prior,
            Some(cos30),
        ))
        .await[0];
        assert_eq!(gated, 0.0, "the surviving pixels agree exactly");

        // The consistency term sees all four: (0 + 0 + 2 + 2) / 4 == 1.0.
        // If the gate had reached it, the contradicted pair would be gone and
        // the answer would be 0/2 == 0.
        let consistency = read(depth_normal_loss(
            n_from_depth,
            img3(rendered, &device),
            alpha,
        ))
        .await[0];
        assert!(
            (consistency - 1.0).abs() < 1e-6,
            "depth/normal consistency = {consistency}, want 1.0 — a value of 0 \
             would mean the contradiction gate leaked into the consistency mask"
        );
    }

    /// **§10d item 7.** A gate that survives nothing yields an exact,
    /// differentiable zero — and the BACKWARD still runs.
    ///
    /// `normal_loss_gate_none_matches_old` already pins the forward zero. The
    /// backward is the half that catches the real failure: `sum / count` with an
    /// unclamped zero count is `0/0 = NaN`, whose gradient is NaN everywhere and
    /// which would then poison the total loss for every other term in the step.
    /// Ported from `test_empty_masks_produce_finite_differentiable_zero_losses`.
    #[tokio::test]
    async fn normal_gate_empty_mask_yields_a_differentiable_zero() {
        let device = autodiff_device().await;

        let pred = lift_to_autodiff(img3(
            vec![
                0.0, 0.3, -1.0, //
                0.0, 0.0, 1.0, //
            ],
            &device,
        ))
        .require_grad();
        let gt = img3(
            vec![
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
            ],
            &device,
        );

        // A gate so tight nothing can pass it.
        let loss = normal_loss(pred.clone(), gt, Some(0.999_999));
        let value = read(loss.clone()).await[0];
        assert_eq!(value, 0.0, "an empty gate must give exactly 0, not NaN");
        assert!(value.is_finite());

        let grads = loss.backward();
        let g = read(
            pred.grad(&grads)
                .expect("backward must still run on an empty mask"),
        )
        .await;
        assert!(
            g.iter().all(|x| x.is_finite()),
            "empty-mask backward produced non-finite gradients: {g:?}"
        );
        assert!(
            g.iter().all(|x| *x == 0.0),
            "nothing survived the gate, so nothing may be supervised: {g:?}"
        );
    }

    /// **§10d item 8.** The 30-degree gate admits 29 degrees and rejects 31.
    ///
    /// Neither suite pinned this. Both would pass today with the comparison
    /// inverted only if the fixture happened to straddle nothing — and, more to
    /// the point, with the threshold off by a small epsilon, or interpreted in
    /// the wrong unit somewhere along the way. cos(29 deg) = 0.87462 and
    /// cos(31 deg) = 0.85717 sit 0.0086 either side of cos(30 deg) = 0.86603,
    /// which is ~7e4 f32 epsilons: nothing about float error can move a pixel
    /// across this boundary, so a failure here is a semantic bug every time.
    #[tokio::test]
    async fn normal_gate_boundary_admits_29_and_rejects_31_degrees() {
        let device = device().await;
        let cos30 = 30.0_f32.to_radians().cos();

        // Prior straight at the camera; predictions tilted by exactly 29 and 31
        // degrees away from it, so the cosine to the prior IS cos(29) / cos(31).
        let gt = img3(
            vec![
                0.0, 0.0, -1.0, //
                0.0, 0.0, -1.0, //
            ],
            &device,
        );
        let tilt = |deg: f32| {
            let r = deg.to_radians();
            [r.sin(), 0.0, -r.cos()]
        };
        let a = tilt(29.0);
        let b = tilt(31.0);
        let pred = img3(vec![a[0], a[1], a[2], b[0], b[1], b[2]], &device);

        let counts = read(normal_gate_counts(pred.clone(), gt.clone(), cos30)).await;
        assert_eq!(
            (counts[0], counts[1]),
            (1.0, 2.0),
            "exactly the 29-degree pixel may survive a 30-degree gate"
        );

        // And the loss agrees: only the 29-degree pixel's L1 is counted, over a
        // denominator of 1 pixel x 3 channels.
        let gated = read(normal_loss(pred, gt, Some(cos30))).await[0];
        let want = (a[0].abs() + a[1].abs() + (a[2] + 1.0).abs()) / 3.0;
        assert!(
            (gated - want).abs() < 1e-6,
            "gated loss = {gated}, want {want} (only the 29-degree pixel, \
             denominator = 1 pixel x 3 channels)"
        );
    }
}

/// **`--depth-uncovered` — plan §5.4.** Coverage handling in [`depth_loss`].
///
/// The three modes are pinned by hand on frames small enough to compute in the
/// head, and the two claims that make the split worth having are tested
/// separately: that `ExcludeNumerator` moves the reported number by exactly the
/// analytically predicted mass and nothing else, and that `Exclude` additionally
/// rescales by the coverage factor.
///
/// The one place the plan's wording had to be corrected is recorded in
/// [`DepthUncovered`]'s docs and pinned by
/// `exclude_numerator_changes_metric_space_gradients`: "gradient-identical"
/// holds in the DEFAULT disparity space only.
#[cfg(all(test, not(target_family = "wasm")))]
mod depth_uncovered_tests {
    use super::*;
    use brush_render::burn_glue::lift_to_autodiff;
    use burn::tensor::TensorData;

    async fn device() -> burn::tensor::Device {
        brush_cube::test_helpers::test_device().await.into()
    }

    async fn autodiff_device() -> burn::tensor::Device {
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff()
    }

    async fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .to_vec::<f32>()
            .expect("f32 tensor")
    }

    fn img2(data: Vec<f32>, device: &burn::tensor::Device) -> Tensor<2> {
        let n = data.len();
        Tensor::<2>::from_data(TensorData::new(data, [1, n]), device)
    }

    const MODES: [DepthUncovered; 3] = [
        DepthUncovered::Count,
        DepthUncovered::ExcludeNumerator,
        DepthUncovered::Exclude,
    ];

    /// **The value pin.** A four-pixel frame with valid GT everywhere and the
    /// render covering exactly half of it — the fixture plan §5.4 item 1 asks
    /// for. Every number below is an exact binary fraction, so these are
    /// equalities in spirit even though they are written with a tolerance.
    ///
    /// ```text
    ///   pred = [1.0, 2.0, 0.0, 0.0]      (the last two: nothing rendered)
    ///   gt   = [2.0, 4.0, 0.5, 1.0]
    ///
    ///   disparity residuals   covered: |1/1 - 1/2| = 0.5 , |1/2 - 1/4| = 0.25
    ///                       uncovered: |0 - 1/0.5| = 2.0 , |0 - 1/1  | = 1.0
    ///
    ///   count             = (0.5 + 0.25 + 2.0 + 1.0) / 4 = 0.9375
    ///   exclude-numerator = (0.5 + 0.25            ) / 4 = 0.1875
    ///   exclude           = (0.5 + 0.25            ) / 2 = 0.375
    /// ```
    ///
    /// Note the shape of the defect this pins: the uncovered pixels contribute
    /// **four times** the covered ones' error here, and the closer the prior
    /// the worse it gets (a 0.5 m surface scores 2.0 m⁻¹), so the reported
    /// depth loss on a partly-covered frame is dominated by exactly the pixels
    /// that can teach the optimiser nothing.
    #[tokio::test]
    async fn uncovered_modes_are_pinned_on_a_half_covered_frame() {
        let device = device().await;
        let pred = || img2(vec![1.0, 2.0, 0.0, 0.0], &device);
        let gt = || img2(vec![2.0, 4.0, 0.5, 1.0], &device);

        for (space, want) in [
            (DepthLossSpace::Disparity, [0.9375f32, 0.1875, 0.375]),
            // metric residuals: covered |1-2| = 1, |2-4| = 2; uncovered
            // |0-0.5| = 0.5, |0-1| = 1.  count = 4.5/4, excl-num = 3/4,
            // exclude = 3/2.
            (DepthLossSpace::Metric, [1.125f32, 0.75, 1.5]),
        ] {
            for (mode, want) in MODES.into_iter().zip(want) {
                let got = read(depth_loss(pred(), gt(), None, space, mode)).await[0];
                assert!(
                    (got - want).abs() < 1e-6,
                    "depth_loss[{space:?}, {mode:?}] = {got}, want {want}"
                );
            }
        }
    }

    /// **The analytic prediction from plan §5.3**, stated as its own assertion
    /// rather than left implicit in the pinned values above: the drop from
    /// `count` to `exclude-numerator` is exactly the excluded mass
    /// `Σ_excluded |1/D_gt| / N_gt-valid`, and `exclude` is then
    /// `exclude-numerator` scaled by the coverage factor
    /// `N_gt-valid / N_covered`.
    ///
    /// A 3-of-5 covered frame, so the coverage factor is `5/3` and cannot be
    /// confused with the `4/2 = 2` of the fixture above.
    #[tokio::test]
    async fn the_reported_drop_and_the_rescale_match_their_closed_forms() {
        let device = device().await;
        let pred = || img2(vec![1.0, 2.0, 4.0, 0.0, 0.0], &device);
        let gt = || img2(vec![1.0, 1.0, 1.0, 1.0, 1.0], &device);

        let space = DepthLossSpace::Disparity;
        let count = read(depth_loss(pred(), gt(), None, space, DepthUncovered::Count)).await[0];
        let excl_num = read(depth_loss(
            pred(),
            gt(),
            None,
            space,
            DepthUncovered::ExcludeNumerator,
        ))
        .await[0];
        let excl = read(depth_loss(
            pred(),
            gt(),
            None,
            space,
            DepthUncovered::Exclude,
        ))
        .await[0];

        // Excluded mass: two pixels at |0 - 1/1| = 1, over the GT-valid count 5.
        let predicted_drop = (1.0f32 + 1.0) / 5.0;
        assert!(
            ((count - excl_num) - predicted_drop).abs() < 1e-6,
            "reported drop {} != predicted {predicted_drop}",
            count - excl_num
        );

        // Coverage rescale: 5 GT-valid pixels, 3 of them covered.
        let factor = 5.0f32 / 3.0;
        assert!(
            (excl - excl_num * factor).abs() < 1e-6,
            "exclude {excl} != exclude-numerator {excl_num} x {factor}"
        );
    }

    /// **The separation the two-step split exists for** (plan §5.2/§8): in the
    /// DEFAULT disparity space, `exclude-numerator` changes the reported loss
    /// and leaves every FINITE gradient alone, bit for bit.
    ///
    /// Why the covered lanes are bit-identical rather than approximately so:
    /// nothing about them changes. The numerator sum loses terms, but the
    /// gradient never passed through those terms, and no reduction the surviving
    /// lanes participate in is re-associated.
    ///
    /// # The uncovered lanes are the exception, and the plan's wording is wrong about them
    ///
    /// Plan §2.1 says an uncovered-but-GT-valid pixel "carries **no gradient**
    /// (`mask_fill`'s VJP zeroes that pixel)". **Measured 2026-08-22: it carries
    /// a `NaN`.** `mask_fill` zeroes the gradient arriving at `recip`'s OUTPUT,
    /// but `recip`'s own backward is `-grad · (1/pred)²`, and at `pred == 0`
    /// that is `0 · ∞ = NaN` — the same `0 · ∞` shape the file's non-finite
    /// discipline exists to prevent, entered through a derivative instead of
    /// through a value. It is disparity-specific: the metric arm has no
    /// `recip()` and its uncovered lanes carry an honest finite `∓1/N`
    /// (`exclude_numerator_changes_metric_space_gradients`).
    ///
    /// Why this has never shown up as a training failure, and why it is still
    /// only a latent defect rather than a live one: an uncovered pixel is by
    /// definition one no gaussian contributed to, so the rasterize backward has
    /// nothing to scatter that pixel's gradient into and the `NaN` dies at the
    /// image boundary of the graph. `brush-train`'s
    /// `depth_loss_does_not_touch_opacity` exercises exactly this configuration
    /// — 4 splats on a 48x48 frame, so most of it is uncovered, against a dense
    /// `gt = 3.0` — and passes `bwd_validate`, which rejects any non-finite
    /// parameter gradient. So the containment is measured, not assumed.
    ///
    /// It is therefore left alone: `count` stays byte-identical, `NaN` included
    /// (T6), and this test pins the difference instead of papering over it. Both
    /// exclude modes replace the `NaN` with an honest `0` as a side effect of
    /// substituting the prediction before the arithmetic — which removes the
    /// reliance on that containment, and is one more reason to prefer them.
    #[tokio::test]
    async fn exclude_numerator_preserves_every_finite_disparity_gradient() {
        let device = autodiff_device().await;
        let gt = || img2(vec![2.0, 4.0, 0.5, 1.0, 3.0, 0.25], &device);
        let vals = vec![1.0, 2.0, 0.0, 0.0, 2.5, 0.3];
        // Lanes 2 and 3 are the uncovered ones (`pred == 0`, GT still valid).
        let uncovered_lanes = [2usize, 3];

        let mut values = Vec::new();
        let mut grads = Vec::new();
        for mode in [DepthUncovered::Count, DepthUncovered::ExcludeNumerator] {
            let pred = lift_to_autodiff(img2(vals.clone(), &device)).require_grad();
            let loss = depth_loss(pred.clone(), gt(), None, DepthLossSpace::Disparity, mode);
            values.push(read(loss.clone()).await[0]);
            let g = loss.backward();
            grads.push(read(pred.grad(&g).expect("depth gradient")).await);
        }

        for i in 0..vals.len() {
            if uncovered_lanes.contains(&i) {
                continue;
            }
            assert_eq!(
                grads[0][i], grads[1][i],
                "exclude-numerator moved the COVERED disparity gradient at lane                  {i}: count {:?} vs exclude-numerator {:?}",
                grads[0], grads[1]
            );
            assert!(
                grads[0][i].is_finite(),
                "covered lane {i} must carry a finite gradient: {:?}",
                grads[0]
            );
        }

        // The documented exception, pinned in BOTH directions so neither half
        // can rot silently: `count` emits the `NaN`, `exclude-numerator` emits 0.
        for i in uncovered_lanes {
            assert!(
                grads[0][i].is_nan(),
                "count-mode disparity gradient at uncovered lane {i} = {},                  expected NaN (see this test's doc comment — if this now reads                  0.0, something upstream fixed `recip`'s 0 · ∞ backward and the                  doc comment above is stale)",
                grads[0][i]
            );
            assert_eq!(
                grads[1][i], 0.0,
                "exclude-numerator must replace the uncovered lane {i} NaN with 0"
            );
        }

        assert!(
            values[0] > values[1],
            "the fixture must actually exclude something: count {} !> \
             exclude-numerator {}",
            values[0],
            values[1]
        );
        assert!(
            values.iter().all(|v| v.is_finite()),
            "both reported VALUES stay finite (only the VJP is affected): {values:?}"
        );
    }

    /// **The correction to the plan, pinned.** Plan §5.2 calls
    /// `exclude-numerator` "gradient-identical to `count`" without qualifying
    /// the space. It is not, under `--depth-loss-space metric`, and the
    /// difference is large rather than marginal.
    ///
    /// The metric arm deliberately has no `pred <= 0` guard — a non-positive
    /// prediction is a legitimate finite residual there, not a singularity — so
    /// an uncovered pixel scores `|0 - D_gt|` with a LIVE `∓1` gradient. Under
    /// `count` that gradient is real supervision pushing an uncovered pixel's
    /// prediction toward the prior; and since the center path's prediction is
    /// `accumulated_depth / α.clamp_min(1e-10)`, the chain rule multiplies it
    /// by up to `1e10` where there is no coverage. Excluding the pixel removes
    /// that entirely.
    ///
    /// So in metric space the two exclude modes are BOTH gradient-affecting.
    /// That does not weaken the attribution argument for the default space, but
    /// it does mean the two-step sequencing of plan §8 buys nothing under
    /// `--depth-loss-space metric` — the numerator step is already a gradient
    /// change there — and any metric-space arm must be read accordingly.
    #[tokio::test]
    async fn exclude_numerator_changes_metric_space_gradients() {
        let device = autodiff_device().await;
        let gt = || img2(vec![2.0, 4.0, 0.5, 1.0], &device);
        let vals = vec![1.0, 2.0, 0.0, 0.0];

        let mut grads = Vec::new();
        for mode in [DepthUncovered::Count, DepthUncovered::ExcludeNumerator] {
            let pred = lift_to_autodiff(img2(vals.clone(), &device)).require_grad();
            let loss = depth_loss(pred.clone(), gt(), None, DepthLossSpace::Metric, mode);
            let g = loss.backward();
            grads.push(read(pred.grad(&g).expect("metric depth gradient")).await);
        }

        // Covered lanes are untouched...
        assert_eq!(
            grads[0][0..2],
            grads[1][0..2],
            "covered lanes must not move: {:?} vs {:?}",
            grads[0],
            grads[1]
        );
        // ...and the uncovered lanes carry a live -1/N under `count` (N = 4
        // GT-valid pixels, residual 0 - gt < 0 so d|r|/dpred = -1) and exactly
        // 0 once excluded.
        for i in 2..4 {
            assert!(
                (grads[0][i] - (-0.25)).abs() < 1e-6,
                "count-mode metric gradient at uncovered lane {i} = {}, want -0.25",
                grads[0][i]
            );
            assert_eq!(
                grads[1][i], 0.0,
                "exclude-numerator must zero the uncovered lane {i}"
            );
        }
    }

    /// **Trap T10 — the plane paths must be untouched.** Both plane depth
    /// sources multiply their GT by the plane-validity mask at the dispatch site
    /// (`train.rs`, `gt_depth * valid`), so a pixel with no plane intersection
    /// arrives here with `gt == 0` and leaves through `gt_valid` before the
    /// coverage mask is consulted.
    ///
    /// This reproduces that pre-masking exactly — GT zeroed wherever the plane
    /// depth came back as the invalid marker `0` — and requires all three modes
    /// to agree in both spaces. If a future change relocated the `gt * valid`
    /// multiply, or made this mask fire on the plane paths, this is what would
    /// catch it.
    #[tokio::test]
    async fn pre_masked_plane_style_gt_is_mode_invariant() {
        let device = device().await;
        // Plane depth: pixels 2 and 3 had no valid intersection -> exactly 0,
        // and the dispatch site has already zeroed their GT to match.
        let pred = || img2(vec![1.0, 2.0, 0.0, 0.0, 3.0], &device);
        let gt = || img2(vec![2.0, 4.0, 0.0, 0.0, 1.5], &device);

        for space in [DepthLossSpace::Disparity, DepthLossSpace::Metric] {
            let mut seen = Vec::new();
            for mode in MODES {
                seen.push(read(depth_loss(pred(), gt(), None, space, mode)).await[0]);
            }
            assert_eq!(
                seen[0], seen[1],
                "{space:?}: exclude-numerator changed a pre-masked plane frame"
            );
            assert_eq!(
                seen[0], seen[2],
                "{space:?}: exclude changed a pre-masked plane frame"
            );
            assert!(
                seen[0] > 0.0,
                "{space:?}: the fixture must produce a nonzero loss"
            );
        }
    }

    /// A frame the render covers COMPLETELY must be mode-invariant too — the
    /// coverage mask has to be inert when there is nothing to exclude, which is
    /// what makes `count` the safe default rather than merely the legacy one.
    /// Includes an invalid-GT pixel so the two masks are exercised together.
    #[tokio::test]
    async fn fully_covered_frames_are_mode_invariant() {
        let device = device().await;
        let pred = || img2(vec![1.0, 2.0, 4.0, 7.0], &device);
        let gt = || img2(vec![2.0, 4.0, 0.0, 1.0], &device);

        for space in [DepthLossSpace::Disparity, DepthLossSpace::Metric] {
            let mut seen = Vec::new();
            for mode in MODES {
                seen.push(read(depth_loss(pred(), gt(), None, space, mode)).await[0]);
            }
            assert_eq!(
                seen[0], seen[1],
                "{space:?}: excl-num moved a covered frame"
            );
            assert_eq!(seen[0], seen[2], "{space:?}: exclude moved a covered frame");
        }
    }

    /// An all-uncovered frame must yield exactly 0 under `exclude` rather than
    /// `NaN`: the denominator's `clamp_min(1.0)` is what stands between an
    /// empty coverage set and a divide by zero, and `exclude` is the first mode
    /// that can empty it while GT pixels still exist. (`count` cannot: its
    /// denominator is the GT count.)
    #[tokio::test]
    async fn exclude_survives_a_frame_with_no_coverage_at_all() {
        let device = device().await;
        let pred = || img2(vec![0.0, 0.0, 0.0], &device);
        let gt = || img2(vec![2.0, 4.0, 1.0], &device);

        for space in [DepthLossSpace::Disparity, DepthLossSpace::Metric] {
            let got = read(depth_loss(
                pred(),
                gt(),
                None,
                space,
                DepthUncovered::Exclude,
            ))
            .await[0];
            assert_eq!(got, 0.0, "{space:?}: all-uncovered frame must score 0");
        }
    }

    /// The optional per-pixel weight composes with the coverage mask the same
    /// way it composes with the GT mask: it modulates the numerator and never
    /// the denominator. Under `exclude` the denominator is the COVERED count,
    /// so a unit weight must still reproduce the `None` path exactly.
    #[tokio::test]
    async fn pixel_weight_still_leaves_the_denominator_alone_under_exclude() {
        let device = device().await;
        let pred = || img2(vec![1.0, 2.0, 0.0, 0.0], &device);
        let gt = || img2(vec![2.0, 4.0, 0.5, 1.0], &device);
        let ones = || Tensor::<2>::ones([1, 4], &device);

        for mode in MODES {
            let none = read(depth_loss(
                pred(),
                gt(),
                None,
                DepthLossSpace::Disparity,
                mode,
            ))
            .await[0];
            let unit = read(depth_loss(
                pred(),
                gt(),
                Some(ones()),
                DepthLossSpace::Disparity,
                mode,
            ))
            .await[0];
            assert_eq!(none, unit, "{mode:?}: unit weight must be the identity");
        }
    }
}
