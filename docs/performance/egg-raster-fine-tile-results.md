# Egg raster fine-tile candidates

- Date: 2026-07-17
- Machine: M4 Pro
- Dataset: egg capture (local dataset; not included in this repository)
- Compiler: native MSL
- Short replays: four fixed masked training views; 20 samples x 4 synchronized
  steps after 4 warmup steps, except the 2400x3200 stress pairs (5 x 4)
- Quality bake-off: 285 training views and 16 held-out views

## Verdict

The first 8x8 candidate was consistently faster, but it failed the provisional
memory gate at the 1.16M checkpoint. A 16x8 specialization retains the speedup,
avoids the 8x8 intersection explosion, and passes the measured whole-step and
memory gates at every tested scale. It also passes the six-run frozen 15k
quality bake-off and a separate 30k stability soak.

The 16x8 training path is therefore included in the Apple Silicon native-MSL
preset:

```sh
BRUSH_NATIVE_MSL_PRESET=1 ./target/release/brush
```

The standalone `BRUSH_NATIVE_MSL_FINE_RASTER_TILES=1` option remains available.
An explicit `BRUSH_NATIVE_MSL_FINE_RASTER_TILES=0` restores Legacy 16x16
training while retaining the rest of the preset. Product rendering entry points
continue to use 16x16. Fine tiles alone use checked raster backward; the full
preset also requests the host-validated unchecked raster-backward path.

## Gate

The promotion gate was:

- at least 5% lower whole-step latency;
- no more than 25% growth in `/usr/bin/time -l` peak memory footprint;
- selector parity for forward output, geometry-invariant auxiliaries, expected
  tile-offset shape, and gradients;
- no practical quality regression or structured artifact across three balanced
  15k runs per selector;
- a successful 30k run with growth stopped at 15k, leaving 15k steady-state
  iterations at high density.

The 16x8 candidate clears every gate. The first of the three 15k run-number
comparisons is 4.992% faster, fractionally below the standalone 5% target, but
the 6.54% mean, 7.43% median, other two pairs, and all short-replay workloads
clear it.

## 8x8 result: rejected on 1.16M footprint

| Workload | 16x16 | 8x8 | Latency change | Peak footprint change |
|---|---:|---:|---:|---:|
| 1.16M, algorithm-only ABBA | 88.512 ms | 81.786 ms | -7.60% | +30.38% |
| 1.16M, deployment ABBA | 70.082 ms | 64.679 ms | -7.71% | +30.47% (separate repeat) |
| 2.49M, 1440x1920 | 121.437 ms | 105.465 ms | -13.15% | +13.72% |

For the 1.16M algorithm-only ABBA, average peak footprint rose from 3.645 GiB
to 4.753 GiB. The stable full-preset repeat independently measured +30.47%.
That exceeds the 25% limit even though the larger checkpoint stayed within it.

## 16x8 result: selected candidate

| Workload | 16x16 | 16x8 | Latency change | Throughput change | Peak footprint change |
|---|---:|---:|---:|---:|---:|
| 1.16M, checked algorithm-only ABBA | 75.462 ms | 68.258 ms | **-9.55%** | +10.55% | -23.65% |
| 1.16M, full-preset deployment ABBA | 71.730 ms | 65.055 ms | **-9.31%** | +10.26% | -23.65% |
| 2.49M, 1440x1920 interleaved pair | 127.991 ms | 109.682 ms | **-14.30%** | +16.69% | -7.35% |
| 2.49M, 2400x3200, mean of two pairs | 218.741 ms | 204.009 ms | **-6.73%** | +7.22% | +3.74% |

At 1.16M, the two deployment pairs were individually 9.32% and 9.29% faster.
At 2400x3200, absolute time moved substantially with concurrent system load,
but the two adjacent A/B pairs remained 6.28% and 7.12% faster. The
interleaved ratios, rather than cross-phase absolute times, are the useful
comparison.

`maximum resident set size` stayed nearly unchanged between selectors. The
table uses the separate macOS `peak memory footprint` field as the end-to-end
memory guard; it should not be interpreted as GPU allocation telemetry alone.

## Intersection census

The census uses the same four 1.16M/1440x1920 views and reports exact GPU range
ends. All sampled CPU replays matched the shortened GPU range ends.

| Geometry | Reserved intersections | Relative | Post-raster intersections | Relative |
|---|---:|---:|---:|---:|
| 16x16 | 5.658M | 1.000x | 3.077M | 1.000x |
| 8x8 | 12.866M | 2.274x | 6.042M | 1.963x |
| 16x8 | 8.552M | 1.512x | 4.338M | 1.410x |

The rectangular grid is the useful middle point: it halves per-tile pixel work
while adding about 51% reserved and 41% surviving intersections, rather than
the 8x8 candidate's 127% and 96% increases. Its 128-thread Morton mapping is an
exact 16-column by 8-row cover.

## Correctness coverage

- Independent CPU raster oracle for hard and smooth cutoff.
- Selector parity for output image and projected splats.
- Boundary grids below, on, and above the 16x8 dimensions.
- Hard- and smooth-cutoff parity for visibility, radius, transform gradients,
  SH gradients, and opacity gradients.
- Candidate backward exercised through both checked and native-MSL unchecked
  launches.
- Rectangular census fixture verifies the 8-pixel y stride.

This is strong kernel-level and short-replay evidence, not a replacement for a
full training-quality comparison. Numerical differences from changed atomic
grouping remain within the existing test tolerances; checkpoint-replay final
losses differ at the normal low-order nondeterministic level.

## Frozen 15k quality bake-off

The full training gate used one frozen binary for both rasterizers so the tile
selector was the only implementation difference:

- commit: `7e9a7afa1035cd5613d3565b0399b55551a34541`;
- native-MSL training binary SHA-256:
  `fde18a84d219512306afb2de2822e2ad723a314d335f6a7cc8ca44ade90fdab9`;
- frozen evaluator commit: `ebf28189eea376a97ad586f082b47c9b9da251b1`;
- evaluator SHA-256:
  `c5876ac391624de8f4f2934550d1863ef835b464266a5b91ba78c179dbbdeb94`.

Each run trained from COLMAP initialization for 15,000 iterations with seed 42,
growth through iteration 15,000, maximum resolution 1920, masked alpha, and
every 20th registered image held out. Egg supplied 285 training and 16 held-out
1440x1920 views. Three stochastic runs per rasterizer were balanced in the
order 16x8, Legacy, Legacy, 16x8, 16x8, Legacy. Both variants used the full
native-MSL preset; the pre-promotion binary additionally selected 16x8 only for
the candidate runs.

| Variant | Run | Wall time | Splats | PLY bytes | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Legacy | 01 | 904.85 s | 1,154,547 | 272,474,709 | 7.645929 | 0.385249 |
| Legacy | 02 | 788.99 s | 1,166,206 | 275,226,233 | 7.746936 | 0.395022 |
| Legacy | 03 | 789.79 s | 1,162,652 | 274,387,489 | 7.654452 | 0.385141 |
| 16x8 | 01 | 859.68 s | 1,163,970 | 274,698,537 | 7.663639 | 0.384354 |
| 16x8 | 02 | 730.42 s | 1,157,874 | 273,259,881 | 7.654840 | 0.386158 |
| 16x8 | 03 | 731.11 s | 1,165,564 | 275,074,721 | 7.648749 | 0.385727 |

Mean wall time fell from 827.877 to 773.737 seconds, a **6.54% reduction** and
1.070x throughput. Median wall time fell from 789.79 to 731.11 seconds, a
**7.43% reduction** and 1.080x throughput. All three run-number comparisons
favored 16x8 by 4.992%, 7.423%, and 7.430%. Mean splat count and PLY size both
increased by 0.115%; the byte difference is exactly explained by the additional
splats rather than a checkpoint-format change.

The same frozen evaluator rendered all 16 held-out views from every checkpoint
with native MSL, masked alpha, and no warnings. Values below are the mean and
sample standard deviation across the three run means:

| Metric | Legacy | 16x8 | 16x8 - Legacy |
|---|---:|---:|---:|
| Canonical PSNR | 7.682439 +/- 0.056018 dB | 7.655743 +/- 0.007486 dB | -0.026696 dB |
| Canonical SSIM | 0.388470 +/- 0.005674 | 0.385413 +/- 0.000942 | -0.003057 |
| Foreground PSNR | 25.223762 +/- 0.031757 dB | 25.252996 +/- 0.005954 dB | **+0.029234 dB** |
| Black-composited VGG LPIPS | 0.01913527 +/- 0.00004174 | 0.01917050 +/- 0.00010329 | +0.00003523 |
| Mean background leakage | 0.0317022 +/- 0.0120599 | 0.0251841 +/- 0.0005882 | -20.56% |
| RMS background leakage | 0.1079598 +/- 0.0335631 | 0.0910367 +/- 0.0024957 | -15.68% |

No practical object-quality regression was detected. Foreground PSNR improves
slightly; the LPIPS change is 0.184% and smaller than observed run variation.
Per-view foreground deltas range from -0.189 to +0.206 dB, with 10 of 16 views
favoring 16x8, and matched contact sheets show no new structured artifact.

Brush's canonical masked-mode metrics retain source RGB outside the alpha mask
while renders are composited against black, so bright out-of-mask leakage can
accidentally improve the full-frame scores. Legacy run 02 has the highest
canonical scores while also having the lowest foreground PSNR and much higher
leakage. Robust run medians show foreground PSNR +0.015957 dB, LPIPS
+0.000009113, essentially neutral mean leakage (+0.688%), and RMS leakage
-0.949%. The large mean leakage improvement should therefore not be read as a
general quality claim.

This is a practical parity result on one dataset, one M4 Pro, and three
stochastic final checkpoints per variant, not a formal equivalence proof. The
raw protocol, logs, hashes, renders, analysis, and montages are retained locally
under `target/quality-bakeoff/egg-15k-legacy-vs-fine-16x8-7e9a7afa`. That path
is gitignored benchmark storage and is not durable across `cargo clean`.

## 30k stability soak

The final promotion gate ran the same frozen candidate binary for 30,000
iterations with growth stopped at 15,000. This exercised the 16x8 path for a
further 15,000 high-density iterations after the normal growth phase.

- process exit: success, with no warning, error, NaN, panic, validation, or
  device-loss record in the training log;
- wall time: 1,721.47 seconds (trainer-reported time: 1,714 seconds);
- final splats: 870,287;
- PLY size: 205,389,348 bytes;
- PLY SHA-256:
  `30c1087b1c507785fd327b787a25bbd7c6b94c8fb7dfb21f554f2e9b2e8861cf`;
- built-in held-out evaluation: 16 views, PSNR 7.652655, SSIM 0.377652;
- frozen post-hoc evaluation: the same 16 views, native MSL, masked alpha,
  PSNR 7.652655, SSIM 0.377652, and an empty warning list.

Both the built-in and independent evaluator produced all expected renders.
This is a stability gate rather than a matched 30k Legacy quality comparison;
the balanced six-run 15k bake-off above provides the selector comparison.

## Reproduction

Build:

```sh
cargo build --release -p brush-bench-test \
  --bin brush-checkpoint-replay --features native-msl
```

Algorithm-only comparison (set fine tiles to `0` and `1` in separate A/B/B/A
processes):

```sh
EGG_DATASET=/path/to/egg
BRUSH_NATIVE_MSL_PRESET=1 \
BRUSH_NATIVE_MSL_UNCHECKED_RASTER_BWD=0 \
BRUSH_NATIVE_MSL_FINE_RASTER_TILES=0 \
./target/release/brush-checkpoint-replay \
  --dataset "$EGG_DATASET" \
  --ply target/bench-checkpoints/egg-quality-15k-exact/egg_15000.ply \
  --max-resolution 1920 --views 4 --eval-split-every 20 \
  --alpha-mode masked --warmup-steps 4 --steps-per-sample 4 --samples 20
```

Deployment comparison uses the same command without the explicit unchecked
override, so both geometries receive the full preset. Census runs add the
`raster-census` feature and `--raster-census-tiles 256`; census readbacks are
excluded from timing runs.

## Follow-up

The Apple Silicon native-MSL preset gate is complete. Repeating the quality
protocol on additional datasets and devices would broaden external validity but
is not required for this scoped promotion. The next material performance work
should target the algorithmic raster design rather than another isolated
micro-fusion.
