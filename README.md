# Brush

<video src=https://github.com/user-attachments/assets/5756967a-846c-44cf-bde9-3ca4c86f1a4d>A video showing various Brush features and scenes</video>

<p align="center">
  <i>
    Massive thanks to <a href="https://www.youtube.com/@gradeeterna">@GradeEterna</a> for the beautiful scenes
  </i>
</p>

Brush is a 3D reconstruction engine using [Gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). It works on a wide range of systems: **macOS/windows/linux**, **AMD/Nvidia/Intel** cards, **Android**, and in a **browser**. To achieve this, it uses WebGPU compatible tech and the [Burn](https://github.com/tracel-ai/burn) machine learning framework.

Machine learning for real time rendering has tons of potential, but most ML tools don't work well with it: Rendering requires realtime interactivity, usually involve dynamic shapes & computations, don't run on most platforms, and it can be cumbersome to ship apps with large CUDA deps. Brush on the other hand produces simple dependency free binaries, runs on nearly all devices, without any setup.

[**Try the web demo** <img src="https://cdn-icons-png.flaticon.com/256/888/888846.png" alt="chrome logo" width="24"/>
](https://arthurbrussee.github.io/brush-demo)
_NOTE: Only works on Chrome and Edge. Firefox and Safari are hopefully supported soon)_

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/TbxJST2BbC)](https://discord.gg/TbxJST2BbC)

# Features

## Training

Brush takes in COLMAP data or datasets in the Nerfstudio format. Training is fully supported natively, on mobile, and in a browser. While training you can interact with the scene and see the training dynamics live, and compare the current rendering to input views as the training progresses.

It also supports masking images:
- Images with transparency. This will force the final splat to match the transparency of the input.
- A folder of images called 'masks'. This ignores parts of the image that are masked out.

### Appearance compensation

For captures with varying exposure, white balance, or lens vignetting between images, Brush can learn per-view photometric corrections during training so the variation isn't baked into the splats. The corrections only apply while training — exported splats keep canonical colors and render unmodified.

- `--bilateral-grid` learns a per-view affine color grid using [gsplat's](https://github.com/nerfstudio-project/gsplat) Apache-2.0 bilateral-grid semantics.
- `--ppisp` enables the full [NVIDIA PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/) model: per-frame exposure and color plus per-camera vignetting and tone curves.

Choose one appearance model per training run. The two flags are mutually exclusive because stacking the models introduces redundant, poorly identified corrections.

Tunables: `--bilagrid-dims x,y,guidance`, `--bilagrid-tv-weight`, `--bilagrid-lr`, `--bilagrid-betas b1,b2`, `--ppisp-lr`, `--ppisp-reg-scale`.

By default evaluation compares the raw, uncorrected render against ground truth — on appearance-varying captures that mostly measures the offset between the splats and the average appearance. Pass `--train-on-eval` to keep eval views in the training set; eval then applies each view's learned correction, which is the more meaningful comparison for these models.

Appearance parameters are training-only and are not stored in PLY checkpoints or used for novel-view rendering. Resuming at a non-zero iteration with appearance compensation is rejected to avoid silently resetting them.

## Viewer
Brush also works well as a splat viewer, including on the web. It can load .ply & .compressed.ply files. You can stream in data from a URL (for a web app, simply append `?url=`).

Brush also can load .zip of splat files to display them as an animation, or a special ply that includes delta frames (see [cat-4D](https://cat-4d.github.io/) and [Cap4D](https://felixtaubner.github.io/cap4d/)!).

## CLI
Brush can be used as a CLI. Run `brush --help` to get an overview. Every CLI command can work with `--with-viewer` which also opens the UI, for easy debugging.

## Rerun

https://github.com/user-attachments/assets/f679fec0-935d-4dd2-87e1-c301db9cdc2c

While training, additional data can be visualized with the excellent [rerun](https://rerun.io/). To install rerun on your machine, please follow their [instructions](https://rerun.io/docs/getting-started/installing-viewer). Open the ./brush_blueprint.rbl in the viewer for best results.

## Building Brush
First install rust 1.88+. You can run tests with `cargo test --all`. Brush uses the wonderful [rerun](https://rerun.io/) for additional visualizations while training, run `cargo install rerun-cli` if you want to use it.

### Windows/macOS/Linux
Use `cargo run --release` from the workspace root to make an optimized build. Use `cargo run` to run a debug build.

On macOS, native Metal Shading Language code generation is opt-in. WGSL remains the default:

```sh
cargo run --release --features native-msl
```

The same feature is available on `brush-cli` and `brush-c`. On non-Metal backends it continues to use WGSL. The compiler choice applies to the whole binary, so compare WGSL and MSL with separate builds.

On Apple Silicon, one runtime preset requests all five retained native-MSL
training optimizations. Compile native MSL into the binary once, then enable the
preset when launching it:

```sh
cargo build --release --features native-msl
BRUSH_NATIVE_MSL_PRESET=1 ./target/release/brush
```

The preset is equivalent to setting these individual options to `1`:

- `BRUSH_NATIVE_MSL_UNCHECKED_RASTER_BWD`
- `BRUSH_NATIVE_MSL_FUSED_SH_ADAM`
- `BRUSH_NATIVE_MSL_COALESCED_SH_GRAD`
- `BRUSH_NATIVE_MSL_SAVED_LOSS_PARTIALS`
- `BRUSH_NATIVE_MSL_SPARSE_SH_ADAM`

Each option remains subject to its compile-time, tensor-shape, and device
capability checks; unsupported cases retain the existing implementation. An
explicit per-option value overrides the preset, which is useful for isolation
or memory-constrained runs:

```sh
BRUSH_NATIVE_MSL_PRESET=1 \
BRUSH_NATIVE_MSL_SAVED_LOSS_PARTIALS=0 \
./target/release/brush
```

Only `1` and case-insensitive `true` enable a switch. `0`, case-insensitive
`false`, or an unrecognized explicit value disable it. The preset and all
individual options are off by default, and have no effect unless the required
native-MSL build and platform gates are present.

Native-MSL builds also expose an experimental, off-by-default raster-backward path without generated buffer bounds checks. It relies on the renderer's tile/range invariants and requires native float atomics (otherwise it falls back to the checked path), so use it for controlled benchmarking and soaks rather than production builds:

```sh
BRUSH_NATIVE_MSL_UNCHECKED_RASTER_BWD=1 cargo run --release --features native-msl
```

An experimental fused update for the spherical-harmonic Adam state is also
available on Apple Silicon native-MSL builds. It preserves the existing
per-coefficient learning-rate scaling and reduced second-moment state, and
falls back to the generic optimizer for unsupported tensor shapes or devices:

```sh
BRUSH_NATIVE_MSL_FUSED_SH_ADAM=1 cargo run --release --features native-msl
```

An experimental Apple Silicon native-MSL path can also coalesce dense
spherical-harmonic gradient materialization. This path preserves exact zero
rows for splats that do not contribute to the sampled view, so optimizer
momentum decay and the dense gradient contract remain unchanged. It falls back
to the existing path when the required 32-lane SIMD-group support is
unavailable:

```sh
BRUSH_NATIVE_MSL_COALESCED_SH_GRAD=1 cargo run --release --features native-msl
```

The steady-state Apple Silicon path can instead keep spherical-harmonic
gradients sparse and fuse their reconstruction directly into the reduced Adam
update. It falls back to the dense gradient and optimizer paths when the model,
optimizer state, or device is incompatible. During compatible steady-state
steps this supersedes the coalesced dense-gradient and fused dense-Adam paths;
the first step remains dense to initialize Adam state (and may use coalesced
gradient materialization), while both dense options remain available on later
sparse fallback steps:

```sh
BRUSH_NATIVE_MSL_SPARSE_SH_ADAM=1 cargo run --release --features native-msl
```

Tracked SSIM training can optionally save the three f32 SSIM partials from
forward for reuse by backward. This removes the first image-load and blur pair
from loss backward without changing its formulas, but adds a `[9, H, W]` tape
tensor of 36 bytes per pixel: about 71.2 MiB at 1920x1080 and 284.8 MiB at
3840x2160. Eval, untracked, L1-only, non-Apple-Silicon, and default builds
continue to use the recompute path. The
1440x1920 egg replay uses about 94.9 MiB for this tape, so disable this option
explicitly under the preset on memory-constrained systems. Its standalone
opt-in remains:

```sh
BRUSH_NATIVE_MSL_SAVED_LOSS_PARTIALS=1 cargo run --release --features native-msl
```

### Web
Brush can be compiled to WASM. Run `npm run dev` to start the demo website using Next.js, see the web directory in app/brush-app/web.

Brush uses [`wasm-pack`](https://drager.github.io/wasm-pack/) to build the WASM bundle. You can also use it without a bundler, see [wasm-pack's documentation](https://drager.github.io/wasm-pack/book/).

WebGPU is still an upcoming standard, and as such, only Chrome 134+ on Windows and macOS is currently supported.

### Android

As a one time setup, make sure you have the Android SDK & NDK installed.
- Check if ANDROID_NDK_HOME and ANDROID_HOME are set
- Add the Android target to rust `rustup target add aarch64-linux-android`
- Install cargo-ndk to manage building a lib `cargo install cargo-ndk`

Each time you change the rust code, run
- `cargo ndk -t arm64-v8a -o crates/brush-app/app/src/main/jniLibs/ build`
- Nb:  Nb, for best performance, build in release mode. This is separate
  from the Android Studio app build configuration.
- `cargo ndk -t arm64-v8a -o crates/brush-app/app/src/main/jniLibs/  build --release`

You can now either run the project from Android Studio (Android Studio does NOT build the rust code), or run it from the command line:
```
./gradlew build
./gradlew installDebug
adb shell am start -n com.splats.app/.MainActivity
```

You can also open this folder as a project in Android Studio and run things from there. Nb: Running in Android Studio does _not_ rebuild the rust code automatically.

## Benchmarks

Rendering and training are generally faster than gsplat. You can run benchmarks of some of the kernels using `cargo bench`.

To benchmark native MSL code generation on macOS, run `cargo bench -p brush-bench-test --features native-msl`.

For a steady-state replay using an exported checkpoint and real dataset views,
use the standalone benchmark binary. Setup, image decoding, pipeline compilation,
and optimizer initialization happen before timing:

```sh
BRUSH_NATIVE_MSL_PRESET=1 \
cargo run --release -p brush-bench-test --bin brush-checkpoint-replay --features native-msl -- \
  --dataset /path/to/dataset \
  --ply /path/to/checkpoint.ply \
  --eval-split-every 20
```

The replay restores model parameters but starts fresh optimizer state. It is
intended to reproduce geometry-, visibility-, and resolution-dependent GPU work,
not to resume training numerically from the checkpoint.

The replay reports the preset and each resolved per-option request. These fields
show configuration intent; device and workload gates can still select a fallback
implementation.

Pass `--skip-refine-weight` to benchmark the late phase after high-gradient
densification stops. Production training selects that path automatically at
`--growth-stop-iter`; visibility and screen-radius refinement stats remain enabled.

For post-hoc quality evaluation, render the held-out dataset views from an
exported PLY with the standalone evaluator. Alpha interpretation is required so
comparisons cannot silently use different masking behavior:

```sh
cargo run --release -p brush-bench-test --bin brush-eval-checkpoint --features native-msl -- \
  --dataset /path/to/dataset \
  --ply /path/to/checkpoint.ply \
  --eval-split-every 20 \
  --alpha-mode masked \
  --save-dir /path/to/renders
```

The evaluator emits one `BRUSH_EVAL_VIEW` JSON record per held-out view and one
aggregate `BRUSH_EVAL_RESULT` record. See the
[egg 15k upstream-versus-macOS-preset bake-off](docs/performance/egg-15k-upstream-vs-macos-preset.md)
for the frozen performance and quality baseline used by raster redesign work.

# Acknowledgements

[**gSplat**](https://github.com/nerfstudio-project/gsplat), for their reference version of the kernels

**Peter Hedman, George Kopanas & Bernhard Kerbl**, for the many discussions & pointers.

**The Burn team**, for help & improvements to Burn along the way

**Raph Levien**, for the [original version](https://github.com/googlefonts/compute-shader-101/pull/31) of the GPU radix sort.

**GradeEterna**, for feedback and their scenes.

# Disclaimer

This is *not* an official Google product. This repository is a forked public version of [the google-research repository](https://github.com/google-research/google-research/tree/master/brush_splat)
