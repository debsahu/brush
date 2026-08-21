# Prior-codec golden fixtures

Four tiny files that pin the depth/normal prior wire formats **across two
languages**. The Rust tests in `src/testdata.rs` (+ `load_depth.rs`,
`load_normal.rs`, `formats/mod.rs`) and the Python `test_prior_io.py` read the
*same bytes* against the *same literal expected-value tables*. If the two codecs
ever drift apart, one side stops matching its table and these files say which.

| file | shape | what it pins |
|---|---|---|
| `golden_depth_u16.png` | 4x3 `Luma16` | uint16-millimetre depth codec: the 0 sentinel, 1 mm, the 499/500/501 rounding neighbourhood, and the 65534/65535 ceiling (which stays **valid**) |
| `golden_normal_u8.png` | 4x3 `Rgb8` | uint8 normal codec: codes 0/127/128/129/255, plus an all-128 pixel that must decode to the `(0,0,0)` invalid sentinel |
| `predictor3_depth.tiff` | 4x3 f32 | Deflate + **FloatingPoint predictor (3)** decode — the lossless Part A format |
| `predictor3_normal.tiff` | 4x3x3 f32 | same, 3-channel |

## Why the TIFFs are checked in rather than generated

The Rust `tiff` crate's *encoder* cannot write predictor 3
(`tiff-0.11.3/src/encoder/mod.rs:38`, "FloatingPoint is currently not
supported") even though its *decoder* reads it. A Rust-only round-trip test
would therefore only ever exercise plain Deflate and would pass with the
predictor pass entirely broken. A `tifffile` + `imagecodecs`-written fixture is
the only honest test of the format Part A actually ships.

## Codec contracts (do not "clean up")

Both codecs are adopted verbatim from `gauss-surf` (Pablo Vela, Apache-2.0) so
his bundles and ours interoperate byte for byte:

- **depth** — encode `rint(clip(m * 1000, 0, 65535))`, decode `u16 / 1000.0`;
  `0` is the invalid sentinel. `rint` is numpy's **half-to-even**, so 0.5 mm
  rounds to code 0 and becomes invalid. Deliberate.
- **normal** — encode `rint((n + 1) / 2 * 255)`, decode `c / 255 * 2 - 1`
  **except code 128, which maps to exact 0.0**. That override is what lets the
  `(0,0,0)` invalid sentinel round-trip; its price is that code 127 decodes to
  about −1/255 while code 129 decodes to about **+3/255** — asymmetric on
  purpose. Rewriting this to `2c/255 − 1` or `(c − 127.5)/127.5` breaks the
  sentinel and silently disagrees with every file already written.

Operation order is pinned to numpy's so *every* code, not just the anchors, is
bit-identical between the languages.

## Regenerating

Any regeneration must reproduce the tables in `src/testdata.rs` byte for byte.
A fixture change that moves a table is a **codec change**, not a fixture
refresh, and needs the plan's §3 decisions revisited first.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow", "tifffile", "imagecodecs"]
# ///
import numpy as np, tifffile
from PIL import Image

DEPTH_MM = np.array([[0, 1, 499, 500],
                     [501, 1500, 1448, 2809],
                     [905, 65534, 65535, 12345]], dtype=np.uint16)
Image.fromarray(DEPTH_MM).save("golden_depth_u16.png", optimize=False)

NORMAL_U8 = np.array([
    [(0, 0, 255), (127, 128, 129), (128, 128, 128), (255, 255, 0)],
    [(128, 128, 0), (129, 127, 255), (0, 255, 128), (64, 192, 32)],
    [(1, 254, 127), (200, 55, 100), (128, 0, 128), (255, 128, 0)],
], dtype=np.uint8)
Image.fromarray(NORMAL_U8, mode="RGB").save("golden_normal_u8.png", optimize=False)

PRED_DEPTH = np.array([[0.0, 1.0, 0.905, 1.4482421875],
                       [2.809, 3.14159265358979, 1e-8, 65.535],
                       [-1.0, 1234.5678, 0.5, 2.0]], dtype=np.float32)
tifffile.imwrite("predictor3_depth.tiff", PRED_DEPTH,
                 compression="deflate", predictor=True)

PRED_NORMAL = np.stack([
    np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
    np.linspace(0.75, -0.75, 12, dtype=np.float32).reshape(3, 4),
    np.linspace(-0.125, -0.9375, 12, dtype=np.float32).reshape(3, 4),
], axis=-1).astype(np.float32)
PRED_NORMAL[1, 1] = 0.0  # invalid sentinel pixel
tifffile.imwrite("predictor3_normal.tiff", PRED_NORMAL,
                 compression="deflate", predictor=True)
```

`imagecodecs` is required — `tifffile` hard-errors on `predictor=True` for
float data without it.
