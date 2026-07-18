# Egg raster workload census

Date: 2026-07-17

## Conclusion

The real egg workload strongly supports a finer spatial raster candidate. At
1440x1920, **75.1-79.0%** of evaluated pixel/splat pairs at 1.16M splats and
**78.6-81.1%** at 2.49M splats fail the alpha cutoff. At 2400x3200 and 2.49M
splats, the cutoff-rejection range remains **65.4-69.8%**.

The first candidate should therefore test 8x8 raster tiles while preserving
the current global depth sort and stable per-tile order. It must remain opt-in:
finer tiles can increase intersection storage and per-splat backward atomic
fan-in even when they reduce wasted pixel work.

Follow-up: 8x8 passed the speed gate but exceeded the 1.16M memory limit. The
subsequent 16x8 candidate passed both measured gates and is the retained
experimental geometry. See
[`egg-raster-fine-tile-results.md`](egg-raster-fine-tile-results.md).

## Provenance

- Dataset: egg capture (local dataset; not included in this repository)
- Branch baseline: `8d5a9d15` (`perf/macos-training-optimizations`)
- Raster selector/oracle foundation: `21530cca`
- Hardware: Apple M4 Pro, 16 GPU cores, 24 GiB unified memory
- Compiler/preset: native MSL with `BRUSH_NATIVE_MSL_PRESET=1`
- Views: `IMG_0003`, `IMG_0077`, `IMG_0152`, and `IMG_0227`
- Alpha mode: masked
- Sample: 256 deterministically selected 16x16 tiles per view

The census uses synchronous readbacks around the unchanged raster launch. Its
reported replay timing is intentionally not a benchmark.

## Results

| Checkpoint | Resolution | Cutoff rejection | Zero-contribution intersections | Evaluated pair fraction |
|---|---:|---:|---:|---:|
| 1.16M splats | 1440x1920 | 75.1-79.0% | 51.1-58.5% | 37.2-43.8% |
| 2.49M splats | 1440x1920 | 78.6-81.1% | 14.7-31.3% | 69.4-88.7% |
| 2.49M splats | 2400x3200 | 65.4-69.8% | 16.6-33.5% | 66.8-84.5% |

`Evaluated pair fraction` is the share of potential 16x16 tile-list pairs
reached before per-pixel transmittance early-out. `Zero-contribution
intersections` is the share of sampled pre-raster tile entries that contribute
to no pixel.

At 1.16M splats and 1440x1920, forward early-out shortens the valid tile lists
from 5.54-5.75M entries to 2.86-3.28M. The remaining backward fan-in averages
2.59-2.85 tiles per visible splat and produces 28.6-32.8M logical gradient
atomics when refinement weight is enabled.

At 2.49M splats and 1440x1920, the post-raster lists contain 4.50-5.61M
entries. Mean backward fan-in is 2.73-3.35 tiles per visible splat, and the
logical gradient-atomic count is 45.0-56.1M per view. At 2400x3200, mean fan-in
rises to 4.48-5.57 and the logical atomic count reaches 73.7-92.4M.

The CPU replay reproduced the GPU-shortened range end for every sampled tile
in all 12 view/resolution combinations: zero mismatches and zero index
difference. Sentinel padding was also negligible, at 1-10 entries per view.
This validates the census as a reliable guide for the candidate design.

## Candidate gate

The initial 8x8 specialization should keep 16x16 as the default and fallback.
Before a real-dataset quality bake-off, require:

1. Exact stable front-to-back ordering and forward/auxiliary parity.
2. Raw raster-gradient and finite-difference parity.
3. ABBA replay at 1.16M and 2.49M splats at both tested resolutions.
4. At least 5% whole-step improvement.
5. No more than 25% peak unified-memory growth.

Record intersection count, tile occupancy, fan-in, sort time, raster-forward
time, raster-backward time, and peak memory. If intersection or atomic growth
erases the fine-tile gain, retain 16x16 parent lists and investigate stable
8x8 child masks/lists inside each parent instead of globally expanding the
intersection array.
