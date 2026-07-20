# Attribution

This fork of [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush) integrates several
in-flight community feature sets into one build. All original commit authorship is preserved
through the merge history; this file records the provenance of each feature.

## Upstream base
- **[ArthurBrussee/brush](https://github.com/ArthurBrussee/brush)** — upstream project (Apache-2.0).

## Integrated feature sets

| Feature | CLI flags | Source | Author |
|---|---|---|---|
| Per-view appearance compensation (bilateral grid + PPISP hybrid) | `--ppisp`, `--ppisp-grid`, `--bilateral-grid` | upstream PR [#483](https://github.com/ArthurBrussee/brush/pull/483) | [@gradeeterna](https://github.com/gradeeterna) |
| gsplat-style depth loss + depth-map viewer | `--depth-loss-weight` | upstream PR [#497](https://github.com/ArthurBrussee/brush/pull/497) | [@Deepthought73](https://github.com/Deepthought73) (Kilian Northoff) |
| DiG (DINO-embedded Gaussians) feature training | `--dino`, `--dino-view` | upstream PR [#511](https://github.com/ArthurBrussee/brush/pull/511) | [@connorsoohoo](https://github.com/connorsoohoo) (Connor Soohoo) |
| Rebase of #483 onto current `main`, hardening, macOS perf (native MSL preset + 16x8 raster tiles) | (build/runtime) | fork [@lanxinger/brush](https://github.com/lanxinger/brush) | [@lanxinger](https://github.com/lanxinger) |

## Integration notes
- The appearance port (#483) is taken via `@lanxinger`'s fork, which had already rebased it onto
  current `main`, hardened it, and added the macOS MSL / 16x8-tile perf work. On top of that base,
  PR #497 (depth) and PR #511 (DiG) were merged in and their conflicts against the appearance
  rewrite resolved by hand.
- The `--bilateral-grid` and `--ppisp` appearance models are mutually exclusive (upstream behavior).
- Backward/autodiff gradient math from each source PR was preserved unchanged; conflict resolution
  only re-threaded function signatures. The merged gradients were checked with the repo's own
  finite-difference test suite (`cargo test -p brush-bench-test --release finite_diff` +
  `dig_features`), which passes.
- Depth-loss (#497) forward + gradient auto-merged cleanly and passes the broad finite-diff
  rasterizer-backward tests; a dedicated depth-loss-enabled training run is the recommended final
  numerical A/B (no depth-specific finite-diff test exists upstream).

## Provenance branches in this fork
- `provenance/pr-497-depth` — upstream PR #497 head as fetched.
- `provenance/pr-511-dig` — upstream PR #511 head as fetched.
- `provenance/lanxinger-483-appearance-macos` — the `@lanxinger` fork base (#483 + macOS perf) this build sits on.
