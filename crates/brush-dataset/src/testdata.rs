//! Cross-language golden fixtures for the depth/normal prior codecs.
//!
//! The four files under `crates/brush-dataset/testdata/` are the **shared**
//! half of plan test T18: the Rust tests here and `test_prior_io.py` on the
//! Python side read the *same bytes* against the *same literal tables*. If the
//! two codecs ever drift, one side stops matching the table and the fixture
//! says which.
//!
//! Every table below is a list of `f32::to_bits` values, not floats, because
//! the point is bit-identity across two languages — an approximate compare
//! would hide exactly the ulp drift this pins down (plan D7).
//!
//! Regenerating the fixtures (documented in `testdata/README.md`) must
//! reproduce these tables byte for byte; a fixture change that moves a table
//! is a codec change and needs the plan's §3 decisions revisited.

/// Golden fixture geometry: 4 wide x 3 tall, deliberately non-square so a
/// transposed size is a detectable error.
pub(crate) const GOLDEN_W: usize = 4;
pub(crate) const GOLDEN_H: usize = 3;

// ---------------------------------------------------------------------------
// Part B: quantized PNG priors
// ---------------------------------------------------------------------------

/// 4x3 uint16 PNG, depth in millimetres. Pixel list per plan §7.
pub(crate) const GOLDEN_DEPTH_U16_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/golden_depth_u16.png"
));

/// The millimetre codes stored in [`GOLDEN_DEPTH_U16_PNG`], row-major.
///
/// Covers the sentinel (0), the smallest representable depth (1 mm), the
/// half-to-even rounding neighbourhood (499/500/501 -- plan D6), a real scene
/// depth (905 mm, the measured playroom minimum), and the u16 ceiling
/// (65534/65535, which stays *valid* per D6).
#[rustfmt::skip]
pub(crate) const GOLDEN_DEPTH_MM: [u16; 12] = [
    0, 1, 499, 500,
    501, 1500, 1448, 2809,
    905, 65534, 65535, 12345,
];

/// Expected decode of [`GOLDEN_DEPTH_U16_PNG`] as metres, `f32::to_bits`.
///
/// Contract: `(mm as f32) / 1000.0` (plan D7; `gauss-surf` `render_io.py:161`).
#[rustfmt::skip]
pub(crate) const GOLDEN_DEPTH_METRES_BITS: [u32; 12] = [
    0x0000_0000, 0x3a83_126f, 0x3eff_7cee, 0x3f00_0000, // 0.0,   0.001,  0.499,  0.5
    0x3f00_4189, 0x3fc0_0000, 0x3fb9_5810, 0x4033_c6a8, // 0.501, 1.5,    1.448,  2.809
    0x3f67_ae14, 0x4283_1168, 0x4283_11ec, 0x4145_851f, // 0.905, 65.534, 65.535, 12.345
];

/// 4x3 RGB8 PNG of quantized unit normals. Covers codes 0/127/128/129/255 and
/// an all-128 (invalid) pixel, per plan §7.
pub(crate) const GOLDEN_NORMAL_U8_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/golden_normal_u8.png"
));

/// The uint8 codes stored in [`GOLDEN_NORMAL_U8_PNG`], row-major, interleaved.
#[rustfmt::skip]
pub(crate) const GOLDEN_NORMAL_CODES: [u8; 36] = [
    0, 0, 255,     127, 128, 129,   128, 128, 128,   255, 255, 0,
    128, 128, 0,   129, 127, 255,   0, 255, 128,     64, 192, 32,
    1, 254, 127,   200, 55, 100,    128, 0, 128,     255, 128, 0,
];

/// Expected decode of [`GOLDEN_NORMAL_U8_PNG`], `f32::to_bits`.
///
/// Contract: `(c as f32) / 255.0 * 2.0 - 1.0`, **except code 128 -> exact 0.0**
/// (plan D1/D7; `gauss-surf` `normals_encoding.py:52-53`). The 128 override is
/// what makes `(0,0,0)` invalid round-trip, and it is the reason 127 and 129
/// are not symmetric about zero.
#[rustfmt::skip]
pub(crate) const GOLDEN_NORMAL_UNIT_BITS: [u32; 36] = [
    0xbf80_0000, 0xbf80_0000, 0x3f80_0000, // (0,0,255)     -> (-1, -1, +1)
    0xbb80_8080, 0x0000_0000, 0x3c40_c100, // (127,128,129) -> (~-1/255, 0, ~+3/255)
    0x0000_0000, 0x0000_0000, 0x0000_0000, // (128,128,128) -> invalid (0,0,0)
    0x3f80_0000, 0x3f80_0000, 0xbf80_0000, // (255,255,0)   -> (+1, +1, -1)
    0x0000_0000, 0x0000_0000, 0xbf80_0000, // (128,128,0)   -> (0, 0, -1)
    0x3c40_c100, 0xbb80_8080, 0x3f80_0000, // (129,127,255)
    0xbf80_0000, 0x3f80_0000, 0x0000_0000, // (0,255,128)
    0xbefe_fefe, 0x3f01_8182, 0xbf3f_bfc0, // (64,192,32)
    0xbf7d_fdfe, 0x3f7d_fdfe, 0xbb80_8080, // (1,254,127)
    0x3f11_9192, 0xbf11_9192, 0xbe5c_dcdc, // (200,55,100)
    0x0000_0000, 0xbf80_0000, 0x0000_0000, // (128,0,128)
    0x3f80_0000, 0x0000_0000, 0xbf80_0000, // (255,128,0)
];

/// Flat index of the all-128 (invalid) pixel in [`GOLDEN_NORMAL_CODES`].
pub(crate) const GOLDEN_NORMAL_INVALID_PIXEL: usize = 2;

// ---------------------------------------------------------------------------
// Part A: lossless Deflate + FloatingPoint-predictor TIFFs
// ---------------------------------------------------------------------------

/// 4x3 float32 TIFF, `compression="deflate", predictor=True` (predictor 3),
/// written by `tifffile` + `imagecodecs`.
///
/// The Rust `tiff` encoder cannot emit predictor 3 (`encoder/mod.rs:38`), so a
/// checked-in fixture is the only honest test of the decode path Part A ships.
pub(crate) const PREDICTOR3_DEPTH_TIFF: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/predictor3_depth.tiff"
));

/// Expected decode of [`PREDICTOR3_DEPTH_TIFF`], `f32::to_bits`.
///
/// Values are deliberately *not* millimetre-quantizable (pi, 1e-8, 1234.5678)
/// so a Part-A regression that quietly rounded through the Part-B codec would
/// be caught.
#[rustfmt::skip]
pub(crate) const PREDICTOR3_DEPTH_BITS: [u32; 12] = [
    0x0000_0000, 0x3f80_0000, 0x3f67_ae14, 0x3fb9_6000, // 0, 1, 0.905, 1.4482422
    0x4033_c6a8, 0x4049_0fdb, 0x322b_cc77, 0x4283_11ec, // 2.809, pi, 1e-8, 65.535
    0xbf80_0000, 0x449a_522b, 0x3f00_0000, 0x4000_0000, // -1, 1234.5677, 0.5, 2
];

/// 4x3x3 float32 TIFF, same compression, with an all-zero (invalid) pixel.
pub(crate) const PREDICTOR3_NORMAL_TIFF: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/predictor3_normal.tiff"
));

/// Expected decode of [`PREDICTOR3_NORMAL_TIFF`], `f32::to_bits`.
#[rustfmt::skip]
pub(crate) const PREDICTOR3_NORMAL_BITS: [u32; 36] = [
    0xbf80_0000, 0x3f40_0000, 0xbe00_0000,
    0xbf51_745d, 0x3f1d_1746, 0xbe4b_a2e9,
    0xbf22_e8ba, 0x3ef4_5d17, 0xbe8b_a2e9,
    0xbee8_ba2f, 0x3eae_8ba3, 0xbeb1_745d,
    0xbe8b_a2e9, 0x3e51_745d, 0xbed7_45d1,
    0x0000_0000, 0x0000_0000, 0x0000_0000, // the invalid pixel
    0x3dba_2e8c, 0xbd8b_a2e9, 0xbf11_745d,
    0x3e8b_a2e9, 0xbe51_745d, 0xbf24_5d17,
    0x3ee8_ba2f, 0xbeae_8ba3, 0xbf37_45d1,
    0x3f22_e8ba, 0xbef4_5d17, 0xbf4a_2e8c,
    0x3f51_745d, 0xbf1d_1746, 0xbf5d_1746,
    0x3f80_0000, 0xbf40_0000, 0xbf70_0000,
];
