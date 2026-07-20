# Egg 15k upstream-versus-optimized bake-off

Date: 2026-07-17

## Conclusion

**Pass.** The fully optimized macOS preset reduces complete 15k wall time by
29.0% on average (1.41x throughput) without a measurable object-quality
regression on this dataset. The optimized and upstream checkpoints have
effectively identical foreground PSNR and black-composited LPIPS. The
optimized runs also have less out-of-mask background leakage.

This is the frozen baseline before an algorithmic raster redesign. A new
raster path should remain opt-in and be compared against the current optimized
build using the same protocol.

## Frozen provenance

- Dataset: egg capture (local dataset; not included in this repository)
- Upstream commit: `3b80985709e2ec04fd6c8622a40e36473647a8e0`
- Optimized commit: `ebf28189eea376a97ad586f082b47c9b9da251b1`
- Upstream binary SHA-256: `eea3a3e30b0889c5070629941a80736798a43339035737684d0d84ca78504f31`
- Optimized binary SHA-256: `330d74cdae577020818da49bc5a2ff7d4d22dc87fc937c0387e02bfebdb9ce23`
- Common evaluator SHA-256: `c5876ac391624de8f4f2934550d1863ef835b464266a5b91ba78c179dbbdeb94`
- Hardware: Apple M4 Pro, 16 GPU cores, 24 GiB unified memory
- Split: 285 training views, 16 held-out views
- Order: optimized, upstream, upstream, optimized, optimized, upstream

Every run used 15,000 iterations, seed 42, max resolution 1920,
`--eval-split-every 20`, explicit masked alpha mode, growth through iteration
15,000, and otherwise identical production defaults. Training contains
unseeded sampling, so three runs per variant were required.

## Performance

| Variant | Wall time, mean | Wall time, median | Range | Relative throughput |
|---|---:|---:|---:|---:|
| Upstream | 1198.95 s | 1179.83 s | 1161.34-1255.67 s | 1.00x |
| Optimized | 851.47 s | 857.47 s | 821.71-875.23 s | 1.41x |

- Mean wall-time reduction: **28.98%**.
- Median wall-time reduction: **27.32%**.
- Run-number comparisons: **24.64-31.71% faster**.
- Upstream run 1 overlapped a CPU-only evaluator build for about 53 seconds;
  it did not initialize the GPU. The other comparisons remain uncontended and
  show the same material speedup.

| Variant | Run | Wall time | Splats | Built-in PSNR | Built-in SSIM |
|---|---:|---:|---:|---:|---:|
| Upstream | 1 | 1179.83 s | 1,159,754 | 7.792214 | 0.400739 |
| Upstream | 2 | 1161.34 s | 1,153,905 | 7.670518 | 0.387223 |
| Upstream | 3 | 1255.67 s | 1,153,682 | 7.644256 | 0.385524 |
| Optimized | 1 | 821.71 s | 1,163,815 | 7.669554 | 0.387982 |
| Optimized | 2 | 875.23 s | 1,156,569 | 7.633237 | 0.383196 |
| Optimized | 3 | 857.47 s | 1,157,473 | 7.644230 | 0.386196 |

The optimized checkpoints average 0.303% more splats and 0.303% larger PLYs,
which is negligible compared with the training-time gain.

## Quality

The same frozen native-MSL evaluator rendered every exported PLY. Each log was
validated against the expected resolved checkpoint path, PLY vertex count,
dataset, split, alpha mode, compiler, per-view aggregate, and 16-view set.

Values below are means across the three run-level means; `+/-` is the sample
standard deviation across runs.

| Metric | Upstream | Optimized | Optimized - upstream |
|---|---:|---:|---:|
| Canonical Brush PSNR | 7.702330 +/- 0.078942 dB | 7.649007 +/- 0.018624 dB | -0.053323 dB |
| Canonical Brush SSIM | 0.391164 +/- 0.008337 | 0.385791 +/- 0.002418 | -0.005373 |
| Foreground PSNR | 25.225972 +/- 0.100294 dB | 25.217716 +/- 0.033223 dB | **-0.008257 dB** |
| Black-composited VGG LPIPS at long edge 512 | 0.0192477 +/- 0.0003174 | 0.0193654 +/- 0.0001391 | **+0.0001177** |
| Background leakage, mean RGB | 0.032854 +/- 0.014016 | 0.024761 +/- 0.001769 | **-0.008093** |
| Background leakage, RMS RGB | 0.107952 +/- 0.035375 | 0.088321 +/- 0.003275 | **-0.019631** |

Lower LPIPS and leakage are better. The foreground PSNR delta is effectively
zero. The LPIPS delta is 0.000118, or 0.61% of the upstream value, and is
smaller than the observed run-to-run variation. Optimized mean leakage is
24.6% lower and RMS leakage is 18.2% lower. Using run medians to reduce the
effect of the upstream outlier still favors optimized by 4.8% and 2.9%,
respectively.

## Why the canonical metric looks worse

Brush's masked-mode built-in PSNR/SSIM retains the source RGB outside the
object mask while the evaluator renders against black. It can therefore reward
bright out-of-mask leakage.

Upstream run 1 illustrates the problem: it has the best canonical PSNR
(7.7922), but the worst upstream foreground PSNR (25.1115), by far the highest
background leakage (0.0490 mean RGB), and visible bright halos. That one run
drives most of the canonical mean gap. The object-aware measures do not show a
systematic optimized regression.

Across view-level variant means, optimized foreground PSNR ranges from
-0.360 dB to +0.458 dB relative to upstream, with 9 views lower and 7 higher.
The visual differences are concentrated around out-of-mask halos and object
edges; the contact sheets show no systematic loss of object texture or
geometry.

## Raster-redesign gate

Use the current optimized build as the implementation baseline for the raster
redesign, not upstream. Preserve the existing six checkpoints and analysis as
the historical upstream gate. For a redesign candidate:

1. Keep the new path opt-in until it passes gradients and this real-dataset
   protocol.
2. Run at least three 15k repeats against the frozen current optimized build.
3. Judge foreground PSNR, black-composited LPIPS, leakage, per-view deltas, and
   matched renders; do not use canonical masked-mode PSNR alone.
4. Treat a persistent object-quality shift, a new structured visual artifact,
   or an unexplained model-size change as a blocker even if throughput improves.

With only three stochastic runs per variant, this bake-off establishes strong
practical parity but not a narrow formal statistical-equivalence bound.
