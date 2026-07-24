//! Opt-in CPU workload census for the tile-based raster pipeline.
//!
//! This module is compiled only with the `raster-census` feature. It reads the
//! existing projected splats, intersection list, and tile offsets around the
//! unchanged raster launch, then analyzes them on the CPU. The readbacks are
//! intentionally synchronous and must never be used for performance timing.

use std::sync::Mutex;

use glam::UVec2;
use serde::Serialize;

use crate::kernels::helpers::{ALPHA_CUTOFF_BAND, ALPHA_CUTOFF_MID, PROJECTED_LANES_USIZE};

#[derive(Debug)]
struct PendingCensus {
    remaining: usize,
    sample_tiles: usize,
    next_sequence: usize,
}

static PENDING_CENSUS: Mutex<Option<PendingCensus>> = Mutex::new(None);

/// Request a census for the next `render_count` backward-enabled renders.
///
/// The checkpoint replay uses this for its first untimed pass over the selected
/// views. Only one request batch may be active at a time.
pub fn request(render_count: usize, sample_tiles: usize) -> Result<(), String> {
    if render_count == 0 {
        return Err("raster census requires at least one render".into());
    }
    if sample_tiles == 0 {
        return Err("raster census requires at least one sampled tile".into());
    }
    let mut pending = PENDING_CENSUS
        .lock()
        .map_err(|_poisoned| "raster census request state is poisoned".to_owned())?;
    if pending.is_some() {
        return Err("a raster census request is already active".into());
    }
    *pending = Some(PendingCensus {
        remaining: render_count,
        sample_tiles,
        next_sequence: 0,
    });
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RasterCensusRequest {
    sequence: usize,
    sample_tiles: usize,
}

pub(crate) fn take_request() -> Option<RasterCensusRequest> {
    let mut pending = PENDING_CENSUS
        .lock()
        .expect("raster census request state is poisoned");
    let request = {
        let state = pending.as_mut()?;
        let request = RasterCensusRequest {
            sequence: state.next_sequence,
            sample_tiles: state.sample_tiles,
        };
        state.next_sequence += 1;
        state.remaining -= 1;
        request
    };
    if pending.as_ref().is_some_and(|state| state.remaining == 0) {
        *pending = None;
    }
    Some(request)
}

#[derive(Clone, Debug, Serialize)]
pub struct DistributionSummary {
    pub count: usize,
    pub zeros: usize,
    pub mean: f64,
    pub p50: u32,
    pub p90: u32,
    pub p99: u32,
    pub max: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct AtomicFanInSummary {
    #[serde(flatten)]
    pub distribution: DistributionSummary,
    pub one: usize,
    pub multiple: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SampledRasterWorkload {
    pub requested_tiles: usize,
    pub sampled_tiles: usize,
    pub sampled_intersections: u64,
    pub potential_pairs: u64,
    pub evaluated_pairs: u64,
    pub early_skipped_pairs: u64,
    pub sigma_rejected_pairs: u64,
    pub cutoff_rejected_pairs: u64,
    pub early_terminated_pairs: u64,
    pub composited_pairs: u64,
    pub zero_contribution_intersections: u64,
    pub gpu_post_intersections: u64,
    pub cpu_post_intersections: u64,
    pub range_end_mismatches: usize,
    pub max_range_end_difference: u32,
    pub evaluated_fraction: f64,
    pub rejection_rate: f64,
    pub cutoff_rejection_rate: f64,
    pub zero_contribution_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RasterCensusReport {
    pub sequence: usize,
    pub image_width: u32,
    pub image_height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub visible_splats: u32,
    pub reserved_intersections: u32,
    pub valid_pre_intersections: u64,
    pub sentinel_intersections: u64,
    pub post_intersections: u64,
    pub occlusion_pruned_intersections: u64,
    pub pre_tile_occupancy: DistributionSummary,
    pub post_tile_occupancy: DistributionSummary,
    pub atomic_fan_in: AtomicFanInSummary,
    pub logical_atomic_writes_without_refine: u64,
    pub logical_atomic_writes_with_refine: u64,
    pub sampled: SampledRasterWorkload,
}

pub(crate) struct RasterCensusInput<'a> {
    pub request: RasterCensusRequest,
    pub img_size: UVec2,
    pub tile_bounds: UVec2,
    pub tile_width: u32,
    pub tile_height: u32,
    pub num_visible: u32,
    pub num_intersections: u32,
    pub smooth_cutoff: bool,
    pub pre_offsets: &'a [u32],
    pub post_offsets: &'a [u32],
    pub compact_gid_from_isect: &'a [u32],
    pub projected_splats: &'a [f32],
}

fn percentile(sorted: &[u32], fraction: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn summarize(values: &[u32]) -> DistributionSummary {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let sum = values.iter().map(|&value| u64::from(value)).sum::<u64>();
    DistributionSummary {
        count: values.len(),
        zeros: values.iter().filter(|&&value| value == 0).count(),
        mean: if values.is_empty() {
            0.0
        } else {
            sum as f64 / values.len() as f64
        },
        p50: percentile(&sorted, 0.50),
        p90: percentile(&sorted, 0.90),
        p99: percentile(&sorted, 0.99),
        max: sorted.last().copied().unwrap_or(0),
    }
}

fn rate(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mix_id(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn select_tiles(num_tiles: usize, requested: usize) -> Vec<usize> {
    let mut keyed = (0..num_tiles)
        .map(|tile_id| (mix_id(tile_id as u64), tile_id))
        .collect::<Vec<_>>();
    keyed.sort_unstable();
    keyed.truncate(requested.min(num_tiles));
    let mut selected = keyed
        .into_iter()
        .map(|(_, tile_id)| tile_id)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

fn cutoff_weight(alpha: f32, smooth_cutoff: bool) -> f32 {
    if !smooth_cutoff {
        return if alpha >= ALPHA_CUTOFF_MID { 1.0 } else { 0.0 };
    }

    let low = ALPHA_CUTOFF_MID - 0.5 * ALPHA_CUTOFF_BAND;
    let t = ((alpha - low) / ALPHA_CUTOFF_BAND).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn validate_offsets(
    label: &str,
    offsets: &[u32],
    num_tiles: usize,
    max_intersections: u32,
) -> Result<(), String> {
    if offsets.len() < num_tiles * 2 {
        return Err(format!(
            "{label} tile offsets contain {} values, expected at least {}",
            offsets.len(),
            num_tiles * 2
        ));
    }
    for tile_id in 0..num_tiles {
        let lo = offsets[tile_id * 2];
        let hi = offsets[tile_id * 2 + 1];
        if lo > hi || hi > max_intersections {
            return Err(format!(
                "{label} tile {tile_id} has invalid range {lo}..{hi} for {max_intersections} intersections"
            ));
        }
    }
    Ok(())
}

pub(crate) fn analyze(input: &RasterCensusInput<'_>) -> Result<RasterCensusReport, String> {
    if input.tile_width == 0 || input.tile_height == 0 {
        return Err("raster census tile dimensions must be at least 1".to_owned());
    }
    let num_tiles = (input.tile_bounds.x as usize)
        .checked_mul(input.tile_bounds.y as usize)
        .ok_or_else(|| "tile count overflow".to_owned())?;
    validate_offsets(
        "pre-raster",
        input.pre_offsets,
        num_tiles,
        input.num_intersections,
    )?;
    validate_offsets(
        "post-raster",
        input.post_offsets,
        num_tiles,
        input.num_intersections,
    )?;

    let expected_projected = input.num_visible as usize * PROJECTED_LANES_USIZE;
    if input.projected_splats.len() < expected_projected {
        return Err(format!(
            "projected splat readback contains {} values, expected at least {expected_projected}",
            input.projected_splats.len()
        ));
    }
    if input.compact_gid_from_isect.len() < input.num_intersections as usize {
        return Err(format!(
            "intersection readback contains {} values, expected at least {}",
            input.compact_gid_from_isect.len(),
            input.num_intersections
        ));
    }

    let mut pre_occupancy = Vec::with_capacity(num_tiles);
    let mut post_occupancy = Vec::with_capacity(num_tiles);
    let mut fan_in = vec![0u32; input.num_visible as usize];

    for tile_id in 0..num_tiles {
        let pre_lo = input.pre_offsets[tile_id * 2];
        let pre_hi = input.pre_offsets[tile_id * 2 + 1];
        let post_lo = input.post_offsets[tile_id * 2];
        let post_hi = input.post_offsets[tile_id * 2 + 1];
        if post_lo != pre_lo || post_hi < post_lo || post_hi > pre_hi {
            return Err(format!(
                "tile {tile_id} changed from {pre_lo}..{pre_hi} to invalid post range {post_lo}..{post_hi}"
            ));
        }
        pre_occupancy.push(pre_hi - pre_lo);
        post_occupancy.push(post_hi - post_lo);

        for isect_id in post_lo..post_hi {
            let compact_gid = input.compact_gid_from_isect[isect_id as usize];
            let Some(count) = fan_in.get_mut(compact_gid as usize) else {
                return Err(format!(
                    "intersection {isect_id} references compact splat {compact_gid}, but only {} are visible",
                    input.num_visible
                ));
            };
            *count = count
                .checked_add(1)
                .ok_or_else(|| format!("atomic fan-in overflow for compact splat {compact_gid}"))?;
        }
    }

    let valid_pre_intersections = pre_occupancy.iter().map(|&v| u64::from(v)).sum::<u64>();
    let post_intersections = post_occupancy
        .iter()
        .map(|&value| u64::from(value))
        .sum::<u64>();
    if valid_pre_intersections > u64::from(input.num_intersections) {
        return Err(format!(
            "valid tile ranges contain {valid_pre_intersections} intersections, exceeding the reserved {}",
            input.num_intersections
        ));
    }

    let selected_tiles = select_tiles(num_tiles, input.request.sample_tiles);
    let mut sampled = SampledRasterWorkload {
        requested_tiles: input.request.sample_tiles,
        sampled_tiles: selected_tiles.len(),
        ..Default::default()
    };

    for tile_id in selected_tiles {
        let pre_lo = input.pre_offsets[tile_id * 2];
        let pre_hi = input.pre_offsets[tile_id * 2 + 1];
        let post_lo = input.post_offsets[tile_id * 2];
        let post_hi = input.post_offsets[tile_id * 2 + 1];
        let entry_count = (pre_hi - pre_lo) as usize;
        let mut contributed = vec![false; entry_count];
        let tile_x = tile_id as u32 % input.tile_bounds.x;
        let tile_y = tile_id as u32 / input.tile_bounds.x;
        let min_x = tile_x * input.tile_width;
        let min_y = tile_y * input.tile_height;
        let max_x = (min_x + input.tile_width).min(input.img_size.x);
        let max_y = (min_y + input.tile_height).min(input.img_size.y);
        let valid_pixels = u64::from(max_x - min_x) * u64::from(max_y - min_y);

        sampled.sampled_intersections += entry_count as u64;
        sampled.potential_pairs += entry_count as u64 * valid_pixels;
        sampled.gpu_post_intersections += u64::from(post_hi - post_lo);

        let mut cpu_post_hi = pre_lo;
        for pixel_y in min_y..max_y {
            for pixel_x in min_x..max_x {
                let pixel_x = pixel_x as f32 + 0.5;
                let pixel_y = pixel_y as f32 + 0.5;
                let mut transmittance = 1.0f32;

                for (local_index, isect_id) in (pre_lo..pre_hi).enumerate() {
                    sampled.evaluated_pairs += 1;
                    let compact_gid = input.compact_gid_from_isect[isect_id as usize] as usize;
                    if compact_gid >= input.num_visible as usize {
                        return Err(format!(
                            "intersection {isect_id} references compact splat {compact_gid}, but only {} are visible",
                            input.num_visible
                        ));
                    }
                    let base = compact_gid * PROJECTED_LANES_USIZE;
                    let splat = &input.projected_splats[base..base + PROJECTED_LANES_USIZE];
                    let dx = pixel_x - splat[0];
                    let dy = pixel_y - splat[1];
                    let sigma =
                        0.5 * (splat[2] * dx * dx + splat[4] * dy * dy) + splat[3] * dx * dy;
                    let alpha = (splat[5] * (-sigma).exp()).min(0.999);

                    if sigma.is_nan() || sigma < 0.0 {
                        sampled.sigma_rejected_pairs += 1;
                        continue;
                    }
                    let weight = cutoff_weight(alpha, input.smooth_cutoff);
                    if weight.is_nan() || weight <= 0.0 {
                        sampled.cutoff_rejected_pairs += 1;
                        continue;
                    }

                    let next_transmittance = transmittance * (1.0 - alpha * weight);
                    if next_transmittance <= 1.0e-4 {
                        sampled.early_terminated_pairs += 1;
                        break;
                    }

                    sampled.composited_pairs += 1;
                    contributed[local_index] = true;
                    cpu_post_hi = cpu_post_hi.max(isect_id + 1);
                    transmittance = next_transmittance;
                }
            }
        }

        sampled.zero_contribution_intersections +=
            contributed.iter().filter(|&&hit| !hit).count() as u64;
        sampled.cpu_post_intersections += u64::from(cpu_post_hi - pre_lo);
        if cpu_post_hi != post_hi {
            sampled.range_end_mismatches += 1;
            sampled.max_range_end_difference = sampled
                .max_range_end_difference
                .max(cpu_post_hi.abs_diff(post_hi));
        }
    }

    sampled.early_skipped_pairs = sampled
        .potential_pairs
        .checked_sub(sampled.evaluated_pairs)
        .ok_or_else(|| "sampled evaluated pairs exceeded potential pairs".to_owned())?;
    let classified = sampled.sigma_rejected_pairs
        + sampled.cutoff_rejected_pairs
        + sampled.early_terminated_pairs
        + sampled.composited_pairs;
    if classified != sampled.evaluated_pairs {
        return Err(format!(
            "classified {classified} sampled pairs, but evaluated {}",
            sampled.evaluated_pairs
        ));
    }
    sampled.evaluated_fraction = rate(sampled.evaluated_pairs, sampled.potential_pairs);
    sampled.rejection_rate = rate(
        sampled.sigma_rejected_pairs + sampled.cutoff_rejected_pairs,
        sampled.evaluated_pairs,
    );
    sampled.cutoff_rejection_rate = rate(sampled.cutoff_rejected_pairs, sampled.evaluated_pairs);
    sampled.zero_contribution_rate = rate(
        sampled.zero_contribution_intersections,
        sampled.sampled_intersections,
    );

    let fan_in_distribution = summarize(&fan_in);
    let atomic_fan_in = AtomicFanInSummary {
        one: fan_in.iter().filter(|&&value| value == 1).count(),
        multiple: fan_in.iter().filter(|&&value| value > 1).count(),
        distribution: fan_in_distribution,
    };

    Ok(RasterCensusReport {
        sequence: input.request.sequence,
        image_width: input.img_size.x,
        image_height: input.img_size.y,
        tile_width: input.tile_width,
        tile_height: input.tile_height,
        tiles_x: input.tile_bounds.x,
        tiles_y: input.tile_bounds.y,
        visible_splats: input.num_visible,
        reserved_intersections: input.num_intersections,
        valid_pre_intersections,
        sentinel_intersections: u64::from(input.num_intersections) - valid_pre_intersections,
        post_intersections,
        occlusion_pruned_intersections: valid_pre_intersections - post_intersections,
        pre_tile_occupancy: summarize(&pre_occupancy),
        post_tile_occupancy: summarize(&post_occupancy),
        atomic_fan_in,
        logical_atomic_writes_without_refine: post_intersections * 9,
        logical_atomic_writes_with_refine: post_intersections * 10,
        sampled,
    })
}

pub(crate) fn emit(report: &RasterCensusReport) {
    println!(
        "BRUSH_RASTER_CENSUS {}",
        serde_json::to_string(report).expect("raster census report must serialize")
    );
}

#[cfg(test)]
mod tests {
    use super::{RasterCensusInput, RasterCensusRequest, analyze, request, take_request};
    use crate::kernels::helpers::PROJECTED_LANES_USIZE;

    const LOW_ALPHA: f32 = 1.0e-4;

    // Each hand-authored projected splat must carry the full PROJECTED_LANES
    // layout that `analyze` reads back: lanes 0..=8 are xy/conic/color and lane
    // 9 is the (composite-unused) depth channel. Tying the return type to
    // PROJECTED_LANES_USIZE makes the array literal fail to compile if a lane is
    // ever added or removed, so this can't silently desync from the kernel the
    // way the old fixed `[f32; 9]` did after the depth lane landed.
    fn splat(alpha: f32) -> [f32; PROJECTED_LANES_USIZE] {
        [0.5, 0.5, 1.0, 0.0, 1.0, alpha, 1.0, 1.0, 1.0, 0.0]
    }

    #[test]
    fn request_batch_is_ordered_and_reusable() {
        request(2, 17).expect("first request");
        assert!(request(1, 1).is_err(), "overlapping request must fail");

        let first = take_request().expect("first report");
        let second = take_request().expect("second report");
        assert_eq!(first.sequence, 0);
        assert_eq!(second.sequence, 1);
        assert_eq!(first.sample_tiles, 17);
        assert_eq!(second.sample_tiles, 17);
        assert!(take_request().is_none());

        request(1, 3).expect("request state resets after the batch");
        let next = take_request().expect("next report");
        assert_eq!(next.sequence, 0);
        assert_eq!(next.sample_tiles, 3);
        assert!(take_request().is_none());
    }

    #[test]
    fn reports_occupancy_fan_in_and_zero_contribution_entries() {
        let projected = [splat(0.5), splat(LOW_ALPHA)].concat();
        let compact = [0, 1, 0];
        let pre = [0, 2, 2, 3];
        let post = [0, 1, 2, 3];
        let report = analyze(&RasterCensusInput {
            request: RasterCensusRequest {
                sequence: 3,
                sample_tiles: 2,
            },
            img_size: glam::uvec2(32, 16),
            tile_bounds: glam::uvec2(2, 1),
            tile_width: 16,
            tile_height: 16,
            num_visible: 2,
            num_intersections: 3,
            smooth_cutoff: false,
            pre_offsets: &pre,
            post_offsets: &post,
            compact_gid_from_isect: &compact,
            projected_splats: &projected,
        })
        .expect("valid census");

        assert_eq!(report.sequence, 3);
        assert_eq!(report.valid_pre_intersections, 3);
        assert_eq!(report.post_intersections, 2);
        assert_eq!(report.occlusion_pruned_intersections, 1);
        assert_eq!(report.pre_tile_occupancy.p50, 1);
        assert_eq!(report.pre_tile_occupancy.max, 2);
        assert_eq!(report.atomic_fan_in.distribution.max, 2);
        assert_eq!(report.atomic_fan_in.distribution.zeros, 1);
        assert_eq!(report.logical_atomic_writes_without_refine, 18);
        assert_eq!(report.logical_atomic_writes_with_refine, 20);
        // The low-alpha splat contributes nowhere in tile 0, and the splat
        // centered in tile 0 has no support in tile 1.
        assert_eq!(report.sampled.zero_contribution_intersections, 2);
    }

    #[test]
    fn sampled_replay_classifies_cutoff_composite_and_termination() {
        let projected = [splat(LOW_ALPHA), splat(0.999), splat(0.999)].concat();
        let compact = [0, 1, 2];
        let offsets = [0, 3];
        let post = [0, 2];
        let report = analyze(&RasterCensusInput {
            request: RasterCensusRequest {
                sequence: 0,
                sample_tiles: 1,
            },
            img_size: glam::uvec2(1, 1),
            tile_bounds: glam::uvec2(1, 1),
            tile_width: 16,
            tile_height: 16,
            num_visible: 3,
            num_intersections: 3,
            smooth_cutoff: false,
            pre_offsets: &offsets,
            post_offsets: &post,
            compact_gid_from_isect: &compact,
            projected_splats: &projected,
        })
        .expect("valid census");

        assert_eq!(report.sampled.potential_pairs, 3);
        assert_eq!(report.sampled.evaluated_pairs, 3);
        assert_eq!(report.sampled.cutoff_rejected_pairs, 1);
        assert_eq!(report.sampled.composited_pairs, 1);
        assert_eq!(report.sampled.early_terminated_pairs, 1);
        assert_eq!(report.sampled.early_skipped_pairs, 0);
        assert_eq!(report.sampled.zero_contribution_intersections, 2);
        assert_eq!(report.sampled.cpu_post_intersections, 2);
        assert_eq!(report.sampled.range_end_mismatches, 0);
    }

    #[test]
    fn sampled_replay_uses_rectangular_tile_height() {
        // Trailing lane is the (composite-unused) depth channel; see `splat`.
        let projected: [f32; PROJECTED_LANES_USIZE] =
            [0.5, 12.5, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0];
        let compact = [0];
        let offsets = [0, 0, 0, 1];
        let report = analyze(&RasterCensusInput {
            request: RasterCensusRequest {
                sequence: 0,
                sample_tiles: 2,
            },
            img_size: glam::uvec2(16, 16),
            tile_bounds: glam::uvec2(1, 2),
            tile_width: 16,
            tile_height: 8,
            num_visible: 1,
            num_intersections: 1,
            smooth_cutoff: false,
            pre_offsets: &offsets,
            post_offsets: &offsets,
            compact_gid_from_isect: &compact,
            projected_splats: &projected,
        })
        .expect("valid rectangular census");

        assert_eq!(report.tile_width, 16);
        assert_eq!(report.tile_height, 8);
        assert_eq!(report.sampled.sampled_tiles, 2);
        assert_eq!(report.sampled.potential_pairs, 128);
    }
}
