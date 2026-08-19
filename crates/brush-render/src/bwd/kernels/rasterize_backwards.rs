//! Per-splat backward rasterizer.
//!
//! One workgroup per tile, each thread owns one splat from the current
//! batch. Pixel state lives in shared memory and is walked in
//! forward-replay order via diagonal scheduling: at iteration `i`, thread
//! `T` is responsible for `(splat=T, pixel=i-T)`. Each thread accumulates
//! the full gradient for its splat in registers and emits a single atomic
//! add per gradient component per batch.
//!
//! The atomic accumulation is parametrised by the [`AtomicAddF32`] trait:
//! `HfAtomicAdd` (native `Atomic<f32>::fetch_add`) when the device
//! supports it, `CasAtomicAdd` (`Atomic<u32>` + CAS over the bit pattern)
//! otherwise. The host picks the impl based on `AtomicUsage::Add`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use crate::kernels::helpers::{
    ALPHA_CUTOFF_MID, PLANE_AUX_LANES, PLANE_AUX_LANES_USIZE, alpha_cutoff_weight,
    alpha_cutoff_weight_deriv, plane_channel_offset, raster_out_channels, read_projected_splat,
};
use crate::kernels::types::{RasterizeUniforms, Splat, Sym2};

// SPLAT_BATCH = 32 = one Apple-Silicon SIMD group, so the per-iter
// sync_cube collapses to a SIMD-lockstep no-op on hardware.
pub const SPLAT_BATCH: u32 = 32;

// Lane indices within one `v_combined` row. This block is the single source of
// truth for the layout: `PLANE_GRAD_LANE_START` and `COMPACT_GRAD_LANES` are
// DERIVED from the last non-plane lane rather than written out, so adding a new
// non-plane lane moves the plane block instead of silently overlapping it. A
// hand-written `PLANE_GRAD_LANE_START = 11` encoded the pre-PGSR lane count and
// would have compiled fine while aliasing the plane values.
//
/// Screen-space xy occupies `XY_LANE ..= XY_LANE + 1`.
pub const XY_LANE: usize = 0;
/// The symmetric 2D conic occupies `CONIC_LANE ..= CONIC_LANE + 2`.
pub const CONIC_LANE: usize = XY_LANE + 2;
/// Per-splat rgb occupies `RGB_LANE ..= RGB_LANE + 2`.
pub const RGB_LANE: usize = CONIC_LANE + 3;
/// Gradient w.r.t. the PROJECTED opacity (`Splat::color_a`), not the raw logit.
pub const ALPHA_LANE: usize = RGB_LANE + 3;
/// Refinement-only screen-space statistic; written only when
/// `compute_refine_weight`.
pub const REFINE_LANE: usize = ALPHA_LANE + 1;
/// Alpha-composited camera-z of the splat CENTRE; written only when
/// `render_depth`.
pub const DEPTH_LANE: usize = REFINE_LANE + 1;

/// First lane of the four PGSR plane-auxiliary VALUE gradients in
/// `v_combined`. Everything below it is the pre-PGSR layout, unchanged.
pub const PLANE_GRAD_LANE_START: usize = DEPTH_LANE + 1;

/// Stride of the compact per-splat backward-gradient buffer (`v_combined`),
/// indexed by `compact_gid`. The 15 lanes are:
///   0..=1 screen-space xy, 2..=4 conic, 5..=7 rgb, 8 alpha,
///   9 refine-weight, 10 expected-depth, 11..=14 PGSR plane-aux values.
///
/// This is the single source of truth for that stride. Every kernel that
/// indexes `v_combined` (this kernel, `project_backwards`, and the coalesced
/// `sh_grad_materialize`) and the host buffer allocation in `render_bwd` MUST
/// derive their stride from this constant. The depth lane (10) was appended
/// after the coalesced materializer was written against a stride of 10, and a
/// hard-coded `* 10` there silently wrongly indexed every `compact_gid >= 1` — a
/// shared constant makes that class of drift impossible. The four plane lanes
/// (11..=14) were appended the same way; the same rule applies, and the sparse
/// SH-Adam consumer in brush-train plus the finite-diff lane assertions in
/// brush-bench-test both derive from this constant rather than restating it.
pub const COMPACT_GRAD_LANES: u32 = (PLANE_GRAD_LANE_START + PLANE_AUX_LANES_USIZE) as u32;

/// Per-splat gradient accumulator for the rasterize backward.
#[derive(CubeType, Copy, Clone)]
pub struct SplatGrad {
    pub xy_x: f32,
    pub xy_y: f32,
    pub conic_x: f32,
    pub conic_y: f32,
    pub conic_z: f32,
    pub rgb_r: f32,
    pub rgb_g: f32,
    pub rgb_b: f32,
    pub alpha: f32,
    pub refine: f32,
    pub depth: f32,
    /// PGSR plane-auxiliary VALUE gradients (`d loss / d plane_aux[k]`).
    pub plane_0: f32,
    pub plane_1: f32,
    pub plane_2: f32,
    pub plane_3: f32,
}

/// One splat's four PGSR plane-auxiliary values, staged per backward batch.
#[derive(CubeType, CubeTypeMut, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct PlaneAux {
    pub v0: f32,
    pub v1: f32,
    pub v2: f32,
    pub v3: f32,
}

#[cube]
fn zero_plane_aux() -> PlaneAux {
    PlaneAux {
        v0: 0.0f32,
        v1: 0.0f32,
        v2: 0.0f32,
        v3: 0.0f32,
    }
}

/// Backward of one alpha-composited PGSR plane channel at one splat/pixel.
///
/// Returns `(alpha-VJP contribution, value-gradient contribution, updated
/// suffix state)`.
///
/// The first component is the load-bearing difference between this ("fused",
/// approach B) path and the feature-pass path (approach A). For a
/// front-to-back composite `P = Σ_i w_i a_i` with `w_i = α_i T_i` and
/// `T_{i+1} = T_i (1 − α_i)`, the suffix `S_i = Σ_{j≥i} w_j a_j` gives
///
/// ```text
/// ∂P/∂α_i = T_i a_i − (S_i − α_i T_i a_i)/(1 − α_i) = (T_i a_i − S_i)/(1 − α_i)
/// ```
///
/// which is exactly the `(state_w * a − state_p) * ra` form the RGB channels
/// already use. Folding it into the alpha VJP is what lets plane (geometry)
/// error reach opacity, conic and means2d.
///
/// This DELIBERATELY differs from the centre-depth channel two lines away,
/// which drops its analogous term so depth error cannot move blending weights.
/// That asymmetry is the design (plan section 4.5, contract rows 1 and 3), not
/// an oversight: PGSR's whole claim is that a plane-intersection depth is an
/// unbiased surface, so its error SHOULD be attributable to opacity/shape.
/// Do not "fix" this to match the centre-depth precedent.
#[cube]
fn plane_channel_bwd(
    state_p: f32,
    v_o_p: f32,
    aux_v: f32,
    state_w: f32,
    vis: f32,
    ra: f32,
) -> (f32, f32, f32) {
    (
        (state_w * aux_v - state_p) * v_o_p * ra,
        vis * v_o_p,
        state_p - vis * aux_v,
    )
}

#[cube]
fn zero_grad() -> SplatGrad {
    SplatGrad {
        xy_x: 0.0f32,
        xy_y: 0.0f32,
        conic_x: 0.0f32,
        conic_y: 0.0f32,
        conic_z: 0.0f32,
        rgb_r: 0.0f32,
        rgb_g: 0.0f32,
        rgb_b: 0.0f32,
        alpha: 0.0f32,
        refine: 0.0f32,
        depth: 0.0f32,
        plane_0: 0.0f32,
        plane_1: 0.0f32,
        plane_2: 0.0f32,
        plane_3: 0.0f32,
    }
}

// f32-atomic-add abstraction lives in `brush-cube` (shared with the
// appearance-grid backward); re-exported here for the host launch code.
pub use brush_cube::{AtomicAddF32, CasAtomicAdd, HfAtomicAdd};

#[allow(clippy::fn_params_excessive_bools)]
#[cube(launch, launch_unchecked)]
pub fn rasterize_backwards_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    // `[N, PLANE_AUX_LANES]` PGSR plane parameters, GLOBAL-gid indexed (the
    // forward reads the same buffer the same way). 1-element dummy when
    // `render_plane` is false.
    plane_aux: &Tensor<f32>,
    // Compact -> global gid map, needed only to address `plane_aux`.
    global_from_compact_gid: &Tensor<u32>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    v_splats: &mut Tensor<Atomic<A::Storage>>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
    #[comptime] compute_refine_weight: bool,
    #[comptime] tile_width: u32,
    #[comptime] tile_height: u32,
    #[comptime] render_depth: bool,
    #[comptime] render_plane: bool,
) {
    let tile_size = comptime![tile_width * tile_height];
    let (tile_id, tile_origin_x, tile_origin_y) = tile_origin(u.tile_bw, tile_width, tile_height);
    // Only `pix_state` lives in shared memory — it gets read-modify-
    // written each iteration (alpha decay) so threads need to see each
    // other's writes. The other per-pixel inputs (`v_output`, the alpha
    // pre-roll) are read-only post-init and L1-cached, so we re-derive
    // them inline in the inner loop. Smaller shared footprint → more
    // workgroup occupancy on Apple.
    let pix_stride = comptime![raster_out_channels(render_depth, render_plane)];
    let mut pix_state = Shared::new_slice((tile_size * pix_stride) as usize);
    load_pixel_state(
        output,
        u,
        tile_origin_x,
        tile_origin_y,
        &mut pix_state,
        tile_width,
        tile_height,
        render_depth,
        render_plane,
    );
    let (range_lo, range_hi) = load_range(tile_offsets, tile_id);
    let num_splats_in_tile = range_hi - range_lo;
    let rounds = (num_splats_in_tile + SPLAT_BATCH - 1u32) / SPLAT_BATCH;

    let mut batch_idx = 0u32;
    while batch_idx < rounds {
        let (compact_gid, splat, splat_active, aux) = load_splat_for_batch(
            compact_gid_from_isect,
            projected,
            plane_aux,
            global_from_compact_gid,
            range_lo,
            num_splats_in_tile,
            batch_idx,
            render_plane,
        );
        let grad = accumulate_grads_for_batch(
            splat,
            aux,
            splat_active,
            tile_origin_x,
            tile_origin_y,
            num_splats_in_tile,
            batch_idx,
            &mut pix_state,
            output,
            v_output,
            u,
            smooth_cutoff,
            compute_refine_weight,
            tile_width,
            tile_height,
            render_depth,
            render_plane,
        );
        if splat_active {
            let base = (compact_gid * COMPACT_GRAD_LANES) as usize;
            A::add(&v_splats[base + XY_LANE], grad.xy_x);
            A::add(&v_splats[base + XY_LANE + 1], grad.xy_y);
            A::add(&v_splats[base + CONIC_LANE], grad.conic_x);
            A::add(&v_splats[base + CONIC_LANE + 1], grad.conic_y);
            A::add(&v_splats[base + CONIC_LANE + 2], grad.conic_z);
            A::add(&v_splats[base + RGB_LANE], grad.rgb_r);
            A::add(&v_splats[base + RGB_LANE + 1], grad.rgb_g);
            A::add(&v_splats[base + RGB_LANE + 2], grad.rgb_b);
            A::add(&v_splats[base + ALPHA_LANE], grad.alpha);
            if comptime![compute_refine_weight] {
                A::add(&v_splats[base + REFINE_LANE], grad.refine);
            }
            if comptime![render_depth] {
                A::add(&v_splats[base + DEPTH_LANE], grad.depth);
            }
            if comptime![render_plane] {
                let p = base + PLANE_GRAD_LANE_START;
                A::add(&v_splats[p], grad.plane_0);
                A::add(&v_splats[p + 1], grad.plane_1);
                A::add(&v_splats[p + 2], grad.plane_2);
                A::add(&v_splats[p + 3], grad.plane_3);
            }
        }
        batch_idx += 1u32;
    }
}

#[cube]
fn tile_origin(
    tile_bw: u32,
    #[comptime] tile_width: u32,
    #[comptime] tile_height: u32,
) -> (u32, u32, u32) {
    let tile_id = CUBE_POS as u32;
    let tile_origin_x = (tile_id % tile_bw) * tile_width;
    let tile_origin_y = (tile_id / tile_bw) * tile_height;
    (tile_id, tile_origin_x, tile_origin_y)
}

#[cube]
fn load_range(tile_offsets: &Tensor<u32>, tile_id: u32) -> (u32, u32) {
    let mut range_buf = Shared::new_slice(2usize);
    if UNIT_POS == 0u32 {
        range_buf[0] = tile_offsets[(tile_id * 2u32) as usize];
        range_buf[1] = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
    }
    // Uniform-marked loads so loop bounds derived from these don't trip
    // WebGPU's "barrier in non-uniform control flow" check.
    (
        workgroup_uniform_load(&range_buf[0]),
        workgroup_uniform_load(&range_buf[1]),
    )
}

/// Seed `pix_state` with the post-rasterise RGB minus the bg pre-roll
/// (so subtracting visited splats walks back to zero) and `T=1`. Pixels
/// outside the image area get all-zero state — the inner loop's
/// `state_w > 1.0e-4` guard then skips them.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_pixel_state(
    output: &Tensor<f32>,
    u: RasterizeUniforms,
    tile_origin_x: u32,
    tile_origin_y: u32,
    pix_state: &mut Shared<[f32]>,
    #[comptime] tile_width: u32,
    #[comptime] tile_height: u32,
    #[comptime] render_depth: bool,
    #[comptime] render_plane: bool,
) {
    let tile_size = comptime![tile_width * tile_height];
    // Channels in the rendered image / per-pixel state stride. The depth
    // numerator (if present) sits at offset 4, after rgba; the four PGSR plane
    // sums follow it.
    let out_chans = comptime![raster_out_channels(render_depth, render_plane)];
    let plane_off = comptime![plane_channel_offset(render_depth) as usize];
    let pixels_per_load = (tile_size + SPLAT_BATCH - 1u32) / SPLAT_BATCH;
    let mut p = 0u32;
    while p < pixels_per_load {
        let pix_rank = UNIT_POS + p * SPLAT_BATCH;
        if pix_rank < tile_size {
            let pix_x = tile_origin_x + pix_rank % tile_width;
            let pix_y = tile_origin_y + pix_rank / tile_width;
            let inside = pix_x < u.img_w && pix_y < u.img_h;
            let s = (pix_rank * out_chans) as usize;
            if inside {
                let pix_id = pix_x + pix_y * u.img_w;
                let base = (pix_id * out_chans) as usize;
                let final_r = output[base];
                let final_g = output[base + 1];
                let final_b = output[base + 2];
                let final_a = output[base + 3];
                let t_final = 1.0f32 - final_a;
                pix_state[s] = final_r - t_final * u.bg_r;
                pix_state[s + 1] = final_g - t_final * u.bg_g;
                pix_state[s + 2] = final_b - t_final * u.bg_b;
                pix_state[s + 3] = 1.0f32;
                if comptime![render_depth] {
                    // Depth numerator has no background term, so the seed is
                    // just the accumulated sum_i w_i z_i.
                    pix_state[s + 4] = output[base + 4];
                }
                if comptime![render_plane] {
                    // Same as depth: raw composited sums, no background term.
                    pix_state[s + plane_off] = output[base + plane_off];
                    pix_state[s + plane_off + 1] = output[base + plane_off + 1];
                    pix_state[s + plane_off + 2] = output[base + plane_off + 2];
                    pix_state[s + plane_off + 3] = output[base + plane_off + 3];
                }
            } else {
                pix_state[s] = 0.0f32;
                pix_state[s + 1] = 0.0f32;
                pix_state[s + 2] = 0.0f32;
                pix_state[s + 3] = 0.0f32;
                if comptime![render_depth] {
                    pix_state[s + 4] = 0.0f32;
                }
                if comptime![render_plane] {
                    pix_state[s + plane_off] = 0.0f32;
                    pix_state[s + plane_off + 1] = 0.0f32;
                    pix_state[s + plane_off + 2] = 0.0f32;
                    pix_state[s + plane_off + 3] = 0.0f32;
                }
            }
        }
        p += 1u32;
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn load_splat_for_batch(
    compact_gid_from_isect: &Tensor<u32>,
    projected: &Tensor<f32>,
    plane_aux: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    range_lo: u32,
    num_splats_in_tile: u32,
    batch_idx: u32,
    #[comptime] render_plane: bool,
) -> (u32, Splat, bool, PlaneAux) {
    let splat_offset = batch_idx * SPLAT_BATCH + UNIT_POS;
    let mut compact_gid = 0u32;
    let mut splat = Splat::zero();
    let mut splat_active = false;
    let mut aux = zero_plane_aux();
    if splat_offset < num_splats_in_tile {
        compact_gid = compact_gid_from_isect[(range_lo + splat_offset) as usize];
        splat = read_projected_splat(projected, compact_gid);
        splat_active = true;
        if comptime![render_plane] {
            // Same addressing as the forward: GLOBAL gid, straight from global
            // memory. Read once per batch, reused across the whole pixel walk.
            let ab = (global_from_compact_gid[compact_gid as usize] * PLANE_AUX_LANES) as usize;
            aux = PlaneAux {
                v0: plane_aux[ab],
                v1: plane_aux[ab + 1],
                v2: plane_aux[ab + 2],
                v3: plane_aux[ab + 3],
            };
        }
    }
    (compact_gid, splat, splat_active, aux)
}

#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_arguments)]
#[cube]
fn accumulate_grads_for_batch(
    splat: Splat,
    aux: PlaneAux,
    splat_active: bool,
    tile_origin_x: u32,
    tile_origin_y: u32,
    num_splats_in_tile: u32,
    batch_idx: u32,
    pix_state: &mut Shared<[f32]>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
    #[comptime] compute_refine_weight: bool,
    #[comptime] tile_width: u32,
    #[comptime] tile_height: u32,
    #[comptime] render_depth: bool,
    #[comptime] render_plane: bool,
) -> SplatGrad {
    let tile_size = comptime![tile_width * tile_height];
    let out_chans = comptime![raster_out_channels(render_depth, render_plane)];
    let plane_off = comptime![plane_channel_offset(render_depth) as usize];
    let conic = Sym2 {
        c00: splat.conic_x,
        c01: splat.conic_y,
        c11: splat.conic_z,
    };
    let clamped_r = max(splat.color_r, 0.0f32);
    let clamped_g = max(splat.color_g, 0.0f32);
    let clamped_b = max(splat.color_b, 0.0f32);

    let num_splats_this_batch = min(SPLAT_BATCH, num_splats_in_tile - batch_idx * SPLAT_BATCH);
    let total_iters = num_splats_this_batch + tile_size - 1u32;

    let mut grad = zero_grad();

    let mut i = 0u32;
    while i < total_iters {
        let active_iter = splat_active && i >= UNIT_POS && (i - UNIT_POS) < tile_size;

        if active_iter {
            let pixel_rank = i - UNIT_POS;
            let s = (pixel_rank * out_chans) as usize;
            let state_x = pix_state[s];
            let state_y = pix_state[s + 1];
            let state_z = pix_state[s + 2];
            let state_w = pix_state[s + 3];

            if state_w > 1.0e-4f32 {
                let pix_x = tile_origin_x + pixel_rank % tile_width;
                let pix_y = tile_origin_y + pixel_rank / tile_width;
                let pixel_coord_x = pix_x as f32 + 0.5f32;
                let pixel_coord_y = pix_y as f32 + 0.5f32;
                let dx = splat.xy_x - pixel_coord_x;
                let dy = splat.xy_y - pixel_coord_y;
                let sigma =
                    0.5f32 * (conic.c00 * dx * dx + conic.c11 * dy * dy) + conic.c01 * dx * dy;
                let gaussian = f32::exp(-sigma);
                let alpha = min(0.999f32, splat.color_a * gaussian);

                let w_cut = if comptime![smooth_cutoff] {
                    alpha_cutoff_weight(alpha)
                } else {
                    select(alpha >= ALPHA_CUTOFF_MID, 1.0f32, 0.0f32)
                };
                if sigma >= 0.0f32 && w_cut > 0.0f32 {
                    let alpha_eff = alpha * w_cut;
                    let next_t = state_w * (1.0f32 - alpha_eff);
                    if next_t <= 1.0e-4f32 {
                        pix_state[s + 3] = 0.0f32;
                    } else {
                        let vis = alpha_eff * state_w;
                        // Re-derive v_out and inv_final_a from `v_output` /
                        // `output` directly. These reads hit the global
                        // tensor each iter rather than shared memory, but
                        // they're L1-cached and only touched on the
                        // not-fully-transparent path. Trades a few global
                        // loads for ~5 KiB of shared memory back, which
                        // recovers an Apple-GPU occupancy slot.
                        let pix_id = pix_x + pix_y * u.img_w;
                        let pix_base = (pix_id * out_chans) as usize;
                        let v_o_x = v_output[pix_base];
                        let v_o_y = v_output[pix_base + 1];
                        let v_o_z = v_output[pix_base + 2];
                        let v_a = v_output[pix_base + 3];
                        let final_a = output[pix_base + 3];
                        let t_final = 1.0f32 - final_a;
                        let v_o_w =
                            (v_a - (u.bg_r * v_o_x + u.bg_g * v_o_y + u.bg_b * v_o_z)) * t_final;
                        // Gate the rgb VJP on the original (pre-clamp) sign:
                        // negative raw values clamp to zero and contribute
                        // no gradient.
                        grad.rgb_r += select(splat.color_r >= 0.0f32, vis * v_o_x, 0.0f32);
                        grad.rgb_g += select(splat.color_g >= 0.0f32, vis * v_o_y, 0.0f32);
                        grad.rgb_b += select(splat.color_b >= 0.0f32, vis * v_o_z, 0.0f32);

                        let ra = 1.0f32 / (1.0f32 - alpha_eff);
                        // Depth no longer contributes to this dot accumulator
                        // (see the render_depth block below). The PGSR plane
                        // channels DO — that is approach B's defining property,
                        // so this accumulator is mutable again.
                        let mut dot_rgb = ((state_w * clamped_r - state_x) * v_o_x
                            + (state_w * clamped_g - state_y) * v_o_y
                            + (state_w * clamped_b - state_z) * v_o_z)
                            * ra;
                        let new_remain_x = state_x - vis * clamped_r;
                        let new_remain_y = state_y - vis * clamped_g;
                        let new_remain_z = state_z - vis * clamped_b;
                        if comptime![render_depth] {
                            let v_o_d = v_output[pix_base + 4];
                            let state_d = pix_state[s + 4];
                            // Depth supervises gaussian positions only. Route the
                            // depth-channel gradient to the per-splat depth value
                            // (grad.depth), but do NOT fold it into the alpha VJP.
                            // The term
                            //   dot_rgb += (state_w * splat.depth - state_d) * v_o_d * ra
                            // is dropped on purpose so depth loss cannot lower its
                            // error by changing blending weights (opacity/shape)
                            // instead of moving. This detaches the depth blending
                            // weights, matching LFS detach_depth_weights
                            // (kernels_backward.cuh:529). The paired denominator
                            // detach lives in brush-train train.rs. The state update
                            // below stays for the front-to-back depth bookkeeping.
                            grad.depth += vis * v_o_d;
                            pix_state[s + 4] = state_d - vis * splat.depth;
                        }
                        if comptime![render_plane] {
                            // PGSR plane channels: the alpha term IS folded in.
                            // See `plane_channel_bwd` for the derivation and for
                            // why this deliberately diverges from the depth
                            // block directly above.
                            let (d0, g0, n0) = plane_channel_bwd(
                                pix_state[s + plane_off],
                                v_output[pix_base + plane_off],
                                aux.v0,
                                state_w,
                                vis,
                                ra,
                            );
                            let (d1, g1, n1) = plane_channel_bwd(
                                pix_state[s + plane_off + 1],
                                v_output[pix_base + plane_off + 1],
                                aux.v1,
                                state_w,
                                vis,
                                ra,
                            );
                            let (d2, g2, n2) = plane_channel_bwd(
                                pix_state[s + plane_off + 2],
                                v_output[pix_base + plane_off + 2],
                                aux.v2,
                                state_w,
                                vis,
                                ra,
                            );
                            let (d3, g3, n3) = plane_channel_bwd(
                                pix_state[s + plane_off + 3],
                                v_output[pix_base + plane_off + 3],
                                aux.v3,
                                state_w,
                                vis,
                                ra,
                            );
                            dot_rgb += d0 + d1 + d2 + d3;
                            grad.plane_0 += g0;
                            grad.plane_1 += g1;
                            grad.plane_2 += g2;
                            grad.plane_3 += g3;
                            pix_state[s + plane_off] = n0;
                            pix_state[s + plane_off + 1] = n1;
                            pix_state[s + plane_off + 2] = n2;
                            pix_state[s + plane_off + 3] = n3;
                        }
                        // Chain through the cutoff. Hard step (production):
                        // w' = 0 and w == 1 in-branch, so the factor is 1.
                        let v_alpha_eff = dot_rgb + v_o_w * ra;
                        let dw_dalpha = if comptime![smooth_cutoff] {
                            alpha_cutoff_weight_deriv(alpha)
                        } else {
                            0.0f32 * alpha
                        };
                        let v_alpha = v_alpha_eff * (w_cut + alpha * dw_dalpha);
                        let v_sigma = -alpha * v_alpha;
                        let vxy_x = v_sigma * (conic.c00 * dx + conic.c01 * dy);
                        let vxy_y = v_sigma * (conic.c01 * dx + conic.c11 * dy);

                        // Suppress the alpha-saturated gradient term — at the
                        // cap the alpha derivative discontinuously flattens.
                        if splat.color_a * gaussian <= 0.999f32 {
                            grad.conic_x += 0.5f32 * v_sigma * dx * dx;
                            grad.conic_y += v_sigma * dx * dy;
                            grad.conic_z += 0.5f32 * v_sigma * dy * dy;
                            grad.xy_x += vxy_x;
                            grad.xy_y += vxy_y;
                            grad.alpha += v_alpha * gaussian;
                            if comptime![compute_refine_weight] {
                                let img_size_x = u.img_w as f32;
                                let img_size_y = u.img_h as f32;
                                let len = f32::sqrt(
                                    vxy_x * img_size_x * vxy_x * img_size_x
                                        + vxy_y * img_size_y * vxy_y * img_size_y,
                                );
                                grad.refine += len / max(final_a, 1.0e-5f32);
                            }
                        }

                        pix_state[s] = new_remain_x;
                        pix_state[s + 1] = new_remain_y;
                        pix_state[s + 2] = new_remain_z;
                        pix_state[s + 3] = next_t;
                    }
                }
            }
        }

        sync_cube();
        i += 1u32;
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::{
        ALPHA_LANE, COMPACT_GRAD_LANES, CONIC_LANE, DEPTH_LANE, PLANE_GRAD_LANE_START, REFINE_LANE,
        RGB_LANE, XY_LANE,
    };
    use crate::kernels::helpers::PLANE_AUX_LANES_USIZE;

    /// PINS the current lane layout. The constants above DERIVE from one
    /// another, which is what stops a new lane from silently overlapping the
    /// plane block — but a derivation alone would let the layout shift under
    /// consumers that legitimately encode it out-of-band: the independent VJP
    /// golden vectors, the derivation doc, and any recorded gradient dump.
    /// Changing these numbers is therefore a deliberate act, and this test is
    /// where you acknowledge it.
    #[test]
    fn lane_layout_is_pinned_and_the_plane_block_comes_last() {
        assert_eq!(XY_LANE, 0);
        assert_eq!(CONIC_LANE, 2);
        assert_eq!(RGB_LANE, 5);
        assert_eq!(ALPHA_LANE, 8);
        assert_eq!(REFINE_LANE, 9);
        assert_eq!(DEPTH_LANE, 10);
        assert_eq!(PLANE_GRAD_LANE_START, 11);
        assert_eq!(COMPACT_GRAD_LANES, 15);

        // The structural invariant the derivation exists to preserve: the plane
        // block starts after EVERY non-plane lane and the stride covers it.
        for lane in [
            XY_LANE + 1,
            CONIC_LANE + 2,
            RGB_LANE + 2,
            ALPHA_LANE,
            REFINE_LANE,
            DEPTH_LANE,
        ] {
            assert!(
                lane < PLANE_GRAD_LANE_START,
                "lane {lane} overlaps the plane block at {PLANE_GRAD_LANE_START}"
            );
        }
        assert_eq!(
            COMPACT_GRAD_LANES as usize,
            PLANE_GRAD_LANE_START + PLANE_AUX_LANES_USIZE
        );
    }
}
