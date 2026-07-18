# WD-R Tanks & Temples 400 px quality pilot

- Date: 2026-07-18
- Base commit: `a3d4ad2bd63bbe497d71629bd2b3ab39f133bbd8`
- Branch: `codex/wd-r-perceptual-loss`
- Machine: Apple M4 Pro, 16-core GPU, 24 GiB unified memory
- OS: macOS 27.0 (26A5378n)
- Compiler: native MSL with the Apple Silicon preset

## Verdict

WD-R passes this exploratory quality gate at a fixed 800,000-splat budget. On
16 held-out views, full-frame VGG LPIPS improved from 0.17220 to 0.14619
(`-15.11%`), with a lower score on every view. Average PSNR improved by
0.280 dB while SSIM fell by 0.00692. Representative render inspection showed
less broad color haze and better hillside structure without an obvious new
structured artifact.

The cost is still too high for a default macOS objective. End-to-end wall time
rose from 170.85 to 681.72 seconds (`3.99x`), and the WD-active half of the run
was approximately `6.08x` slower. Peak memory footprint increased by 0.846 GiB.

Keep WD-R native-only, experimental, and disabled by default. The result is
promising enough for repeatability and full-schedule work, but it is not a
production-quality claim: this is one short training pair on one scene,
refinement is not bitwise deterministic, and DISTS is not implemented.

## Protocol

The dataset is the opaque Tanks & Temples `train` scene at
`/Users/markus/Downloads/GS_DATASETS/tandt_db/tandt/train`:

- 301 registered 980x545 JPEG images and 182,686 COLMAP initialization points;
- long edge 400, producing 400x222 training and evaluation images;
- deterministic filename ordering with `--eval-split-every 20`;
- 285 training views and 16 held-out views;
- seed 42 and background noise disabled;
- 6,000 iterations, including a common 3,000-iteration RGB-loss warm-up;
- growth allowed through iteration 6,000, capped at 800,000 splats;
- WD-R `gamma = 0.032`, the paper's Tanks & Temples value;
- transparent alpha mode, which is equivalent to opaque RGB for this dataset.

The final-code pair ran WD-R first and baseline second to reverse the execution
order of the preliminary pair. Training clamps only the perceptual VGG input to
`[0, 1]`; the original RGB loss continues to consume the raw float render.

Both arms used the same command shape; only `--wd-r-gamma` and the export path
changed:

```sh
BRUSH_NATIVE_MSL_PRESET=1 \
BRUSH_NATIVE_MSL_SAVED_LOSS_PARTIALS=0 \
RUST_LOG=info \
/usr/bin/time -l ./target/release/brush-cli \
  /Users/markus/Downloads/GS_DATASETS/tandt_db/tandt/train \
  --max-resolution 400 \
  --eval-split-every 20 \
  --alpha-mode transparent \
  --total-train-iters 6000 \
  --growth-stop-iter 6000 \
  --max-splats 800000 \
  --wd-r-warmup-iters 3000 \
  --wd-r-gamma 0.032 \
  --background-noise-strength 0 \
  --seed 42 \
  --eval-every 1000 \
  --eval-save-to-disk \
  --export-every 3000 \
  --export-name 'model_{iter}.ply' \
  --export-path /path/to/wd-r
```

The baseline used `--wd-r-gamma 0`. Both final PLY files contain exactly
800,000 vertices and are 188,801,619 bytes.

## Why capacity was fixed

An initial run allowed up to two million splats. At iteration 3,401, after only
401 WD-active steps, WD-R had reached 1,140,838 splats versus the baseline's
905,260 (`+26.0%`). The run was stopped at the predefined 25% representation
divergence threshold.

This confirms the paper's warning that `gamma` changes positional gradients
and therefore adaptive densification. Comparing those final models would mix
loss quality with representation size. The 800,000-splat cap isolates the
objective at equal checkpoint capacity; it does not describe WD-R's natural
uncapped model size.

## Held-out result

The standalone checkpoint evaluator loaded each final PLY independently and
reported equal-view averages:

| Metric | Baseline | WD-R | WD-R delta | Per-view wins |
|---|---:|---:|---:|---:|
| VGG LPIPS (lower) | 0.172204325 | 0.146192102 | **-0.026012223 (-15.11%)** | **16/16** |
| PSNR (higher) | 20.304248 dB | 20.584352 dB | +0.280104 dB | 10/16 |
| SSIM (higher) | 0.840090588 | 0.833172522 | -0.006918065 | 5/16 |

LPIPS deltas ranged from -0.042585 to -0.013339, with a median of -0.024650.
The unanimous direction is a strong descriptive signal. PSNR's median delta
was +0.281 dB, and SSIM's median delta was -0.00470.

The independently evaluated iteration-3,000 checkpoints, immediately before
WD-R activation, already differed slightly because refinement sampling is not
bitwise deterministic: LPIPS was 0.208158 for baseline and 0.207155 for the
future WD-R arm (`-0.48%`). From iteration 3,000 to 6,000, baseline LPIPS then
improved by 17.27% while the WD-R arm improved by 29.43%. Most of the final
gap therefore accumulated after the objective switched, rather than during the
common warm-up.

A preliminary pair, run before adding the final perceptual-input clamp, showed
the same direction: LPIPS improved by 16.37% and WD-R won all 16 views. It is
useful corroboration, but it is not combined with the final-code pair as a
replicate because the optimized objective changed.

LPIPS is opt-in in `brush-eval-checkpoint`. This pilot uses VGG LPIPS on the
full 400x222 frame, clamps the render to `[0, 1]`, and composites ground-truth
alpha onto black. That policy is explicit in the JSON output and is not
comparable to the older Egg policy that masks and resizes images to 512 px.

## Runtime and memory

| Measurement | Baseline | WD-R | Ratio or delta |
|---|---:|---:|---:|
| Trainer duration | 167 s | 678 s | 4.06x |
| End-to-end wall time | 170.85 s | 681.72 s | 3.99x |
| Approximate iterations 3,000-6,000 | 101 s | 614 s | 6.08x |
| Maximum resident set size | 1.502 GiB | 1.355 GiB | -0.147 GiB |
| Peak memory footprint | 2.710 GiB | 3.556 GiB | +0.846 GiB |

The peak-footprint field is the more relevant macOS unified-memory guard. The
short common warm-up and evaluation/export overhead make whole-run slowdown
smaller than the steady-state 7.49x result in the checkpoint-replay pilot.

## Reproduction and evidence

The final evaluator command for each arm was:

```sh
BRUSH_NATIVE_MSL_PRESET=1 ./target/release/brush-eval-checkpoint \
  --dataset /Users/markus/Downloads/GS_DATASETS/tandt_db/tandt/train \
  --ply /path/to/model_6000.ply \
  --max-resolution 400 \
  --eval-split-every 20 \
  --alpha-mode transparent \
  --lpips
```

Generated logs and renders are under
`target/wd-r-quality-pilot/tandt-train-400-6k-budget-800k-repeat-final`. They
are local benchmark artifacts and are not source-controlled.

## Promotion gates

Before considering WD-R a production preset:

1. Seed refinement sampling or run at least three balanced training pairs to
   establish repeatability.
2. Add a separately validated DISTS evaluator and confirm that its direction
   agrees with LPIPS and visual review.
3. Run a full 30k comparison, controlling representation capacity explicitly
   or tuning `gamma` so uncapped output sizes are comparable.
4. Recheck peak footprint at the intended deployment resolution and scene
   scale.
5. Profile ground-truth feature caching only after quality promotion; its
   unified-memory cost may outweigh the runtime gain on smaller Macs.
