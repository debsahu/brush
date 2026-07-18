# WD-R macOS checkpoint-replay pilot

- Date: 2026-07-18
- Base commit: `a3d4ad2bd63bbe497d71629bd2b3ab39f133bbd8`
- Branch: `codex/wd-r-perceptual-loss`
- Machine: Apple M4 Pro, 16-core GPU, 24 GiB unified memory
- OS: macOS 27.0 (26A5378n)
- Compiler: native MSL with the Apple Silicon preset

## Verdict

The exact WD-R objective is functional on the production inner/autodiff device
lifecycle, but it is not suitable as the default macOS training objective yet.
At a 600x800 training resolution it reduced throughput by 23.6x and increased
the measured peak memory footprint by 4.53 GiB. A 300x400 configuration is much
more practical, but still ran 7.49x slower than the normal objective.

WD-R therefore remains native-only, experimental, and disabled by default.
The next gate should be a matched quality pilot at 400 px on opaque Tanks &
Temples data. It should not be promoted until perceptual quality, splat count,
and end-to-end training time justify the cost.

## Paper alignment

The implementation follows *Drop-In Perceptual Optimization for 3D Gaussian
Splatting*:

- fixed `sigma = 4`;
- three image scales and five VGG16 feature slices;
- `gamma * (WD + (1 / 0.09) * original_loss)` after a 3,000-iteration warm-up;
- unchanged renderer, optimizer, densification, and pruning logic.

The paper calibrates `gamma` per dataset because it changes the positional
gradient magnitudes that drive adaptive densification. Its WD-R values are
0.025 for Deep Blending and indoor Mip-NeRF 360, 0.028 for outdoor Mip-NeRF
360, 0.032 for Tanks & Temples, and 0.025 for BungeeNeRF. The replay's 0.028 is
only a runtime workload selector; it is not a proposed Egg quality setting.

Appendix A.1 reports a 4.5x overhead for the authors' unoptimized A100 path and
about 2.8x after caching ground-truth VGG/statistics and pruning zero-weight
pyramid levels. Brush now implements the exact zero-weight pruning. It does not
cache ground-truth features: a full cache is a poor initial fit for
unified-memory Macs because the multi-scale feature/statistics tensors are
large and training datasets commonly contain hundreds of views.

## Exact optimization retained

The reference always evaluates five local-statistics levels per feature map.
For fixed `sigma = 4`, effective `log2(sigma)` is clamped to `[0, 2]`, and the
nonnegative low-pass filter cannot increase it. Every level above
`ceil(log2(sigma))` therefore has an identically zero triangular weight.

Bounding the loop to those active levels preserves the pinned reference value
and gradients. A dedicated regression also compares the pruned path against
the full five-level path for sigma values 0, 0.5, 1, 1.5, and 2 on odd-sized
feature maps.

On the one-view 600x800 smoke workload, this reduced the timed WD-R step from
2,761.829 ms to 858.479 ms and the peak footprint from 8.37 GiB to 6.93 GiB.
An alternative that batched prediction and target tensors was rejected: Burn
then retained the target half on the prediction tape and peak footprint rose to
12.77 GiB.

## Replay assets and protocol

- Dataset: `/Users/markus/Downloads/GS_DATASETS/egg`
- Checkpoint: `target/bench-checkpoints/egg-quality-15k-exact/egg_15000.ply`
- Checkpoint SHA-256:
  `1f578e9e4a1f236aa70b102b17b0b11deebbcf43bbda49c14eda8276496a3caf`
- Checkpoint size: 273,420,125 bytes
- Splats: 1,158,553
- Views: four fixed 300x400 or 600x800 views
- Timing: five samples of four queued steps after four untimed warm-up steps
- Global iteration range: `15000..15024`
- Refinement-only gradient statistic: disabled, matching late training
- Alpha: transparent compositing for both arms

Egg normally uses masked-alpha training. WD-R intentionally rejects masked
feature-space semantics, so this replay is valid for runtime and memory only;
its losses are not comparable to the historical masked Egg quality runs.

The measured command shape was:

```sh
BRUSH_NATIVE_MSL_PRESET=1 \
BRUSH_NATIVE_MSL_SAVED_LOSS_PARTIALS=0 \
/usr/bin/time -l ./target/release/brush-checkpoint-replay \
  --dataset /Users/markus/Downloads/GS_DATASETS/egg \
  --ply target/bench-checkpoints/egg-quality-15k-exact/egg_15000.ply \
  --max-resolution 800 \
  --views 4 \
  --eval-split-every 20 \
  --alpha-mode transparent \
  --start-iter 15000 \
  --wd-r-warmup-iters 3000 \
  --wd-r-gamma 0.028 \
  --warmup-steps 4 \
  --steps-per-sample 4 \
  --samples 5 \
  --seed 42 \
  --skip-refine-weight
```

Baseline arms used the same command with `--wd-r-gamma 0`. Baseline values in
the table are the mean of the surrounding A and A repeat medians and peak
footprints.

## Results

| Training view | Baseline median | WD-R median | Slowdown | Baseline peak | WD-R peak | Peak delta |
|---|---:|---:|---:|---:|---:|---:|
| 600x800 | 35.088 ms | 828.343 ms | 23.608x | 2.531 GiB | 7.058 GiB | +4.527 GiB |
| 300x400 | 31.440 ms | 235.563 ms | 7.493x | 2.337 GiB | 3.393 GiB | +1.055 GiB |

At 600x800, the two baseline medians were 35.088729 and 35.087490 ms;
WD-R's five sample values ranged from 825.585 to 833.788 ms. At 300x400,
the baseline medians were 31.529677 and 31.349562 ms; WD-R ranged from
234.244 to 236.301 ms. The A-repeat stability makes drift an implausible
explanation for the measured gap.

`maximum resident set size` moved much less than `peak memory footprint`
because Metal allocations use unified memory. As in the other macOS replay
gates, the peak-footprint field is the relevant end-to-end guard, not standalone
GPU allocation telemetry.

## Quality-gate follow-up

The follow-up [400 px Tanks & Temples quality pilot](wd-r-tandt-400-quality-pilot.md)
held both arms to 800,000 splats and added opt-in full-frame VGG LPIPS to the
checkpoint evaluator. WD-R improved LPIPS by 16.37% on average and on every one
of the 16 held-out views in the preliminary pair. After the final input-contract
fix, the reversed-order pair improved LPIPS by 15.11% on all 16 views, gained
0.280 dB PSNR, lost 0.00692 SSIM, took 3.99x the end-to-end wall time, and
increased peak footprint by 0.846 GiB.

That passes the exploratory quality gate, but not the production gate. The
result is one nondeterministic training pair on one scene, uses a short 6k
schedule, and does not yet include DISTS.
