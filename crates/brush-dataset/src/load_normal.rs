use crate::load_depth::{PRIOR_MAGIC_HELP, PriorWireFormat, describe_magic, prior_wire_format};

use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use tokio::io::AsyncReadExt;

/// Lazily-loaded per-view surface-normal map, decoded to `[H, W, 3]` f32 in
/// row-major, channel-interleaved order.
///
/// Two wire formats are accepted, dispatched on magic bytes (never on the file
/// extension -- prior discovery in the `formats` module is extension-blind):
///
/// - **3-channel float32 TIFF** -- components stored directly.
/// - **RGB8 PNG** -- components quantized to uint8 (see `decode_normal_u8`).
///
/// Convention (must match whatever writes the priors), **identical for both
/// wire formats** -- nothing downstream can tell which one a prior came from:
/// - normals live in the **camera frame**, `OpenCV` axes (+X right, +Y down,
///   +Z forward),
/// - unit length, oriented toward the camera (`n.z <= 0`),
/// - `(0, 0, 0)` marks an invalid / unobserved pixel.
///
/// Note the sign convention is *ours*, not the quantization codec's: the codec
/// is sign-agnostic and `gauss-surf` stores away-from-camera normals. Flipping
/// foreign bundles is an extraction-time job, never loader logic (plan D2), so
/// this loader stays provenance-free.
///
/// This mirrors [`crate::load_depth::LoadDepth`]: lazy, size-checked against
/// the training resolution, and hard-erroring on a mismatch, because a prior
/// rendered at the wrong resolution is silently wrong supervision.
#[derive(Clone, Debug)]
pub struct LoadNormal {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
}

impl PartialEq for LoadNormal {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl LoadNormal {
    pub fn new(vfs: Arc<BrushVfs>, path: PathBuf) -> Self {
        Self { vfs, path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn load(
        &self,
        expected_h: usize,
        expected_w: usize,
    ) -> Result<TensorData, LoadNormalError> {
        let normal = self.load_vec(expected_h, expected_w).await?;
        Ok(TensorData::new(normal, [expected_h, expected_w, 3]))
    }

    pub async fn load_vec(
        &self,
        expected_h: usize,
        expected_w: usize,
    ) -> Result<Vec<f32>, LoadNormalError> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;
        decode_checked(&bytes, expected_h, expected_w)
    }
}

/// Decode a normal prior of either wire format and check it against the
/// expected size. Split out from `load_vec` so the decode + validation path is
/// unit-testable without a VFS.
pub(crate) fn decode_checked(
    bytes: &[u8],
    expected_h: usize,
    expected_w: usize,
) -> Result<Vec<f32>, LoadNormalError> {
    let Some(format) = prior_wire_format(bytes) else {
        return Err(LoadNormalError::UnsupportedFormat(format!(
            "unsupported prior format (expected float32 TIFF or {{Luma16|RGB8}} PNG magic); \
             leading bytes [{}] match none of {PRIOR_MAGIC_HELP}",
            describe_magic(bytes)
        )));
    };

    let (normal, w, h) = match format {
        PriorWireFormat::Tiff => decode_f32_rgb_tiff(bytes)?,
        PriorWireFormat::Png => decode_u8_normal_png(bytes)?,
    };

    if w != expected_w || h != expected_h {
        // Per-format variant so the message never claims the wrong container.
        let msg = format!("invalid normal size {w} x {h}, expected {expected_w} x {expected_h}");
        return Err(match format {
            PriorWireFormat::Tiff => LoadNormalError::ReadTiffError(msg),
            PriorWireFormat::Png => LoadNormalError::ReadPngError(msg),
        });
    }

    Ok(normal)
}

/// Dequantize one uint8 normal component to its signed unit value.
///
/// Codec contract taken verbatim from `gauss-surf` (Pablo Vela, Apache-2.0),
/// `normals_encoding.py:44, 52-53`: encode `rint((n + 1) / 2 * 255)`, decode
/// `c / 255 * 2 - 1` **except code 128, which maps to exact 0.0**.
///
/// The 128 override is deliberate and load-bearing (plan D1): it is what lets
/// the `(0, 0, 0)` invalid sentinel survive a round trip, since a symmetric
/// inverse would put 0.0 at code 127.5 -- unrepresentable. It has two prices,
/// both accepted:
///
/// 1. The two neighbours of the override are *asymmetric*: code 127 decodes to
///    about -1/255 while code 129 decodes to about +3/255, exactly 3x as far
///    from zero, not +/- the same step.
/// 2. **The worst-case round-trip error is 2/255, not 1/255.** Everything that
///    encodes to 128 -- the whole half-open interval `[0, 2/255)` -- decodes to
///    exactly 0, so a component just under 2/255 is dragged a full step. Every
///    other code stays within the usual 1/255. (The plan's section 5 prose says
///    1/255; its own *measured* 0.0078 figure is the correct one, and
///    0.007843 = 2/255. Corrected by WS-P against 480,000 random components:
///    976 exceeded 1/255, and every one of them was code 128.) That is still
///    ~20x inside the measured prior-error budget, so it moves the bound, not
///    the verdict.
///
/// **Do not "clean this up" to `2c/255 - 1` or `(c - 127.5)/127.5`.** Either
/// rewrite breaks the sentinel and creates a silent, permanent disagreement
/// with every file `gauss-surf` and `prior_io.py` have written.
///
/// The operation order is pinned to numpy's (plan D7) so every code, not just
/// the anchors, is bit-identical across the two languages.
#[inline]
pub(crate) fn decode_normal_u8(code: u8) -> f32 {
    if code == 128 {
        0.0
    } else {
        f32::from(code) / 255.0 * 2.0 - 1.0
    }
}

/// Decode a 3-channel uint8 PNG of quantized normals into `[H, W, 3]`.
///
/// The pixel type is matched **exactly**: an RGBA, 16-bit, grayscale, or
/// paletted PNG is rejected rather than converted. `image`'s `to_rgb8()` would
/// happily narrow a 16-bit map or drop an alpha channel, turning a wrong file
/// into plausible-looking supervision -- the same class of error the
/// float32-TIFF sample-format check exists to prevent.
fn decode_u8_normal_png(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), LoadNormalError> {
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?;
    let color = image.color();

    let image::DynamicImage::ImageRgb8(rgb) = image else {
        return Err(LoadNormalError::ReadPngError(format!(
            "unsupported normal PNG pixel type {color:?} (expected 8-bit 3-channel RGB8)"
        )));
    };

    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let normal: Vec<f32> = rgb.into_raw().into_iter().map(decode_normal_u8).collect();

    if w * h * 3 == normal.len() {
        Ok((normal, w, h))
    } else {
        Err(LoadNormalError::ReadPngError(format!(
            "expected {} samples for {w} x {h}, got {}",
            w * h * 3,
            normal.len()
        )))
    }
}

/// Decode a 3-channel float32 TIFF into `[H, W, 3]` row-major, interleaved order.
fn decode_f32_rgb_tiff(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), LoadNormalError> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes))?;
    let tiff::decoder::DecodingResult::F32(normal) = decoder.read_image()? else {
        return Err(LoadNormalError::ReadTiffError(
            "unsupported TIFF sample format (expected float32 normals)".to_owned(),
        ));
    };
    let (w, h) = decoder.dimensions()?;
    let (w, h) = (w as usize, h as usize);
    if w * h * 3 != normal.len() {
        Err(LoadNormalError::ReadTiffError(format!(
            "expected 3 channels ({} floats for {w} x {h}), got {}",
            w * h * 3,
            normal.len()
        )))
    } else {
        Ok((normal, w, h))
    }
}

#[derive(Error, Debug)]
pub enum LoadNormalError {
    #[error("I/O error while loading normal map: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error while loading TIFF file: {0}")]
    LoadTiffError(#[from] tiff::TiffError),

    #[error("Error while reading TIFF file: {0}")]
    ReadTiffError(String),

    #[error("Error while loading PNG file: {0}")]
    LoadPngError(#[from] image::ImageError),

    #[error("Error while reading PNG file: {0}")]
    ReadPngError(String),

    #[error("Error reading normal map: {0}")]
    UnsupportedFormat(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tiff::encoder::{TiffEncoder, colortype};

    /// Encode `[H, W, 3]` f32 samples as an uncompressed `RGB32Float` TIFF.
    fn encode_rgb_f32(values: &[f32], w: u32, h: u32) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf).expect("tiff encoder");
            encoder
                .write_image::<colortype::RGB32Float>(w, h, values)
                .expect("write rgb f32 tiff");
        }
        buf.into_inner()
    }

    #[test]
    fn round_trips_a_three_channel_f32_tiff() {
        let (w, h) = (3u32, 2u32);
        let values: Vec<f32> = (0..(w * h * 3)).map(|i| i as f32 * 0.25 - 1.0).collect();
        let bytes = encode_rgb_f32(&values, w, h);

        let decoded =
            decode_checked(&bytes, h as usize, w as usize).expect("3-channel f32 tiff must decode");
        assert_eq!(decoded.len(), (w * h * 3) as usize);
        for (got, want) in decoded.iter().zip(values.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn rejects_a_size_mismatch() {
        let (w, h) = (3u32, 2u32);
        let values = vec![0.0f32; (w * h * 3) as usize];
        let bytes = encode_rgb_f32(&values, w, h);

        // Same pixel count, transposed: must still be rejected.
        let err = decode_checked(&bytes, w as usize, h as usize)
            .expect_err("transposed size must be rejected");
        assert!(
            matches!(err, LoadNormalError::ReadTiffError(_)),
            "expected a read error, got {err:?}"
        );
    }

    #[test]
    fn rejects_a_single_channel_tiff() {
        let (w, h) = (3u32, 2u32);
        let values = vec![1.0f32; (w * h) as usize];
        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf).expect("tiff encoder");
            encoder
                .write_image::<colortype::Gray32Float>(w, h, &values)
                .expect("write gray f32 tiff");
        }
        let bytes = buf.into_inner();

        let err = decode_checked(&bytes, h as usize, w as usize)
            .expect_err("single-channel tiff must be rejected as a normal map");
        assert!(
            matches!(err, LoadNormalError::ReadTiffError(_)),
            "expected a read error, got {err:?}"
        );
    }

    // ---- T2 / T18: the uint8 codec's signed anchors, bit for bit -----------

    #[test]
    fn normal_png_signed_anchors_bit_exact() {
        use crate::testdata;

        let decoded = decode_checked(
            testdata::GOLDEN_NORMAL_U8_PNG,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("golden uint8 normal PNG must decode");

        assert_eq!(decoded.len(), testdata::GOLDEN_NORMAL_UNIT_BITS.len());
        for (i, (&got, &want_bits)) in decoded
            .iter()
            .zip(testdata::GOLDEN_NORMAL_UNIT_BITS.iter())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                want_bits,
                "component {i} (code {}): got {got} (bits {:#010x}), want {want_bits:#010x}",
                testdata::GOLDEN_NORMAL_CODES[i],
                got.to_bits()
            );
        }

        // Anchors, spelled out rather than trusted to the table.
        assert_eq!(decode_normal_u8(0).to_bits(), (-1.0f32).to_bits());
        assert_eq!(decode_normal_u8(128).to_bits(), 0.0f32.to_bits());
        assert_eq!(decode_normal_u8(255).to_bits(), 1.0f32.to_bits());

        // THE ASYMMETRY (plan D1, and the trap flagged in the 2026-08-19 plan
        // section 10d). 128 -> exact 0 means the two neighbours are NOT +/- one
        // step: 127 sits about one 1/255 below zero, 129 about three 1/255
        // above it. A "symmetric cleanup" that deletes the 128 override, or
        // rescales by 127.5, silently disagrees with every file gauss-surf ever
        // wrote.
        let c127 = decode_normal_u8(127);
        let c129 = decode_normal_u8(129);

        // ...and "about" is the point. The pinned op order does NOT produce the
        // bits of the obvious literal, so this asserts the exact drift rather
        // than hiding it under a tolerance that merely happens to be wider.
        // Measured, not assumed: the trailing `- 1.0` cancels ~6 decimal digits
        // near zero, so `assert_eq!(c129, 3.0 / 255.0)` FAILS against correct
        // code. That is why the golden tables ship as bit patterns.
        assert_eq!(c127.to_bits(), 0xBB80_8080, "code 127");
        assert_ne!(
            c127.to_bits(),
            (-1.0f32 / 255.0).to_bits(),
            "127 is one ulp off the naive literal; equality here means the op \
             order changed"
        );
        assert_eq!(c129.to_bits(), 0x3C40_C100, "code 129");
        assert_ne!(c129.to_bits(), (3.0f32 / 255.0).to_bits(), "code 129 drift");
        let drift = c129 - 3.0f32 / 255.0;
        assert!(
            (5.8e-8..6.0e-8).contains(&drift),
            "code 129 must sit 5.87e-8 above 3/255, got {drift:e}"
        );

        // Null model: had the codec been symmetric about 128, the two
        // neighbours would be equal and opposite. They are not -- 129 is three
        // times as far from zero as 127, which no symmetric map can produce.
        assert!(
            (c129 + c127).abs() > 0.5 * c129.abs(),
            "127/129 must NOT be symmetric about zero: {c127} / {c129}"
        );
        assert!(
            ((c129 / -c127) - 3.0).abs() < 1e-4,
            "the asymmetry ratio must be 3:1, got {}",
            c129 / -c127
        );

        // The 2/255 bound from the doc comment, as a test: everything in
        // [0, 2/255) encodes to 128 and decodes to exactly 0, so the override's
        // worst case is a full step, not the half step the plan's prose claims.
        assert_eq!(decode_normal_u8(128).to_bits(), 0.0f32.to_bits());
        let just_under = 2.0f32 / 255.0 - f32::EPSILON;
        let code = ((just_under + 1.0) / 2.0 * 255.0).round() as u8;
        assert_eq!(code, 128, "a component just under 2/255 must encode to 128");
        assert!(
            (just_under - decode_normal_u8(code)).abs() > 1.0 / 255.0,
            "the override's worst-case error must exceed 1/255"
        );

        // And the pinned table is not self-fulfilling: rederive it from the
        // codes through the D7 expression.
        for (i, &code) in testdata::GOLDEN_NORMAL_CODES.iter().enumerate() {
            assert_eq!(decoded[i].to_bits(), decode_normal_u8(code).to_bits());
        }
    }

    /// Exhaustive sweep of all 256 codes, pinning the properties the golden
    /// fixture can only sample.
    #[test]
    fn normal_codec_is_exhaustively_pinned() {
        let decoded: Vec<f32> = (0..=255u8).map(decode_normal_u8).collect();

        // Endpoints are exact, and 128 is the ONLY code that reaches zero.
        assert_eq!(decoded[0].to_bits(), (-1.0f32).to_bits());
        assert_eq!(decoded[255].to_bits(), 1.0f32.to_bits());
        let zeros: Vec<usize> = (0..256).filter(|&c| decoded[c] == 0.0).collect();
        assert_eq!(zeros, vec![128]);

        // Monotone non-decreasing across the whole range, INCLUDING over the
        // override: 127 -> -1/255, 128 -> 0, 129 -> +3/255 still ascends. A
        // codec that broke ordering would put two surfaces out of sequence.
        for c in 1..256 {
            assert!(
                decoded[c] > decoded[c - 1],
                "code {c} ({}) must exceed code {} ({})",
                decoded[c],
                c - 1,
                decoded[c - 1]
            );
        }

        // Error against the ideal signed value: within 1/255 everywhere EXCEPT
        // the override, where it reaches a full 2/255 (see decode_normal_u8's
        // doc comment -- the plan's prose says 1/255 and is wrong).
        let mut over_one_255 = vec![];
        for c in 0..256u32 {
            #[expect(clippy::cast_precision_loss, reason = "c <= 255, exact in f32")]
            let ideal = (c as f32) / 255.0 * 2.0 - 1.0;
            let err = (decoded[c as usize] - ideal).abs();
            assert!(
                err < 2.0 / 255.0,
                "code {c} error {err} reaches a full step"
            );
            if err > 1.0 / 255.0 {
                over_one_255.push(c);
            }
        }
        assert_eq!(
            over_one_255,
            vec![128],
            "only the override may exceed 1/255"
        );

        // The op-order sensitivity is specifically the `/ 255.0` and the
        // literal form, NOT the multiply: `x * 2.0` is exact in binary
        // floating point, so a fused `2.0.mul_add(c / 255.0, -1.0)` is
        // bit-identical for every code. Recorded because a mutation that
        // rewrote the expression that way survived the whole suite, and
        // "the test missed it" and "there is nothing to miss" look alike
        // until someone checks.
        for c in 0..=255u8 {
            if c == 128 {
                continue;
            }
            let fused = 2.0f32.mul_add(f32::from(c) / 255.0, -1.0);
            assert_eq!(
                fused.to_bits(),
                decoded[c as usize].to_bits(),
                "code {c}: multiply by two must be exact"
            );
        }
    }

    // ---- T3: all-128 pixel is the invalid sentinel -------------------------

    #[test]
    fn normal_png_all_128_is_invalid_zero_vector() {
        use crate::testdata;

        let decoded = decode_checked(
            testdata::GOLDEN_NORMAL_U8_PNG,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("golden uint8 normal PNG must decode");

        let base = testdata::GOLDEN_NORMAL_INVALID_PIXEL * 3;
        assert_eq!(
            &testdata::GOLDEN_NORMAL_CODES[base..base + 3],
            &[128u8, 128, 128]
        );
        let (x, y, z) = (decoded[base], decoded[base + 1], decoded[base + 2]);
        assert_eq!((x.to_bits(), y.to_bits(), z.to_bits()), (0, 0, 0));

        // brush-loss treats a prior as valid iff ||n|| > 0.5
        // (`normal_prior_valid_mask`, brush-loss/src/lib.rs:1800-1823), so the
        // decoded zero vector is what actually switches supervision off. Pin
        // the property the loss will test, not just the components.
        let norm = x.mul_add(x, y.mul_add(y, z * z)).sqrt();
        assert!(norm <= 0.5, "invalid pixel must fail the ||n|| > 0.5 gate");

        // It is also the ONLY pixel that decodes to exactly zero. Nothing else
        // in the fixture can be mistaken for the sentinel, and no unit normal
        // could ever encode to all-128 anyway (it would need ||n|| < 0.007).
        let exact_zero: Vec<usize> = (0..(testdata::GOLDEN_W * testdata::GOLDEN_H))
            .filter(|px| (0..3).all(|c| decoded[px * 3 + c].to_bits() == 0.0f32.to_bits()))
            .collect();
        assert_eq!(exact_zero, vec![testdata::GOLDEN_NORMAL_INVALID_PIXEL]);

        // Null model: the ||n|| > 0.5 gate must be discriminating, not trivially
        // true. The fixture deliberately contains several pixels whose codes
        // all sit in 127..=129 -- probes of the override's neighbourhood, not
        // physical normals -- and those legitimately land near zero. Derive
        // that set from the CODES rather than hardcoding indices, so a fixture
        // refresh cannot silently weaken the test.
        let near_override = |px: usize| {
            (0..3).all(|c| (127..=129).contains(&testdata::GOLDEN_NORMAL_CODES[px * 3 + c]))
        };
        let mut valid = 0;
        let mut probes = 0;
        for px in 0..(testdata::GOLDEN_W * testdata::GOLDEN_H) {
            let (x, y, z) = (decoded[px * 3], decoded[px * 3 + 1], decoded[px * 3 + 2]);
            let norm = x.mul_add(x, y.mul_add(y, z * z)).sqrt();
            if near_override(px) {
                assert!(norm <= 0.5, "probe pixel {px} should be near zero");
                probes += 1;
            } else {
                assert!(norm > 0.5, "pixel {px} should be valid, ||n|| = {norm}");
                valid += 1;
            }
        }
        // Both sides must be non-trivial: all-probes would make the gate
        // assertion vacuous, no-probes would make the invalid one vacuous.
        assert!(valid >= 8, "only {valid} pixels exercise the valid side");
        assert!(
            probes >= 2,
            "only {probes} pixels exercise the invalid side"
        );
        assert!(near_override(testdata::GOLDEN_NORMAL_INVALID_PIXEL));

        // A single 128 component does NOT invalidate a pixel: the override is
        // per-component. This one is (128, 0, 128) -> (0, -1, 0), a perfectly
        // good toward-camera normal that passes the gate above.
        let base = testdata::GOLDEN_NORMAL_PARTIAL_128_PIXEL * 3;
        assert_eq!(
            &testdata::GOLDEN_NORMAL_CODES[base..base + 3],
            &[128u8, 0, 128]
        );
        assert_eq!(decoded[base].to_bits(), 0.0f32.to_bits());
        assert_eq!(decoded[base + 1].to_bits(), (-1.0f32).to_bits());
        assert_eq!(decoded[base + 2].to_bits(), 0.0f32.to_bits());
    }

    // ---- T5: wrong PNG pixel type is a hard error, never a conversion ------

    #[test]
    fn normal_png_wrong_pixel_type_rejected() {
        use crate::testdata;

        let (w, h) = (testdata::GOLDEN_W as u32, testdata::GOLDEN_H as u32);
        let n = (w * h) as usize;

        // RGBA8: `to_rgb8()` would silently drop the alpha channel.
        let rgba = image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(w, h, vec![128u8; n * 4]).expect("rgba buffer"),
        );
        let err = decode_checked(&encode_png(&rgba), testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect_err("RGBA normal PNG must be rejected");
        assert!(
            matches!(err, LoadNormalError::ReadPngError(_)),
            "expected a PNG read error, got {err:?}"
        );

        // Luma16: the *depth* wire format handed to the normal loader.
        let luma16 = image::DynamicImage::ImageLuma16(
            image::ImageBuffer::from_raw(w, h, vec![128u16; n]).expect("luma16 buffer"),
        );
        let err = decode_checked(&encode_png(&luma16), testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect_err("16-bit grayscale normal PNG must be rejected");
        assert!(
            matches!(err, LoadNormalError::ReadPngError(_)),
            "expected a PNG read error, got {err:?}"
        );

        // Rgb16: right channel count, wrong depth.
        let rgb16 = image::DynamicImage::ImageRgb16(
            image::ImageBuffer::from_raw(w, h, vec![128u16; n * 3]).expect("rgb16 buffer"),
        );
        let err = decode_checked(&encode_png(&rgb16), testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect_err("16-bit RGB normal PNG must be rejected");
        assert!(
            matches!(err, LoadNormalError::ReadPngError(_)),
            "expected a PNG read error, got {err:?}"
        );

        // Luma8: single channel.
        let luma8 = image::DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w, h, vec![128u8; n]).expect("luma8 buffer"),
        );
        let err = decode_checked(&encode_png(&luma8), testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect_err("8-bit grayscale normal PNG must be rejected");
        assert!(
            matches!(err, LoadNormalError::ReadPngError(_)),
            "expected a PNG read error, got {err:?}"
        );

        // Control: the real RGB8 fixture still decodes, so the rejections above
        // are about the pixel type and not about PNGs in general.
        decode_checked(
            testdata::GOLDEN_NORMAL_U8_PNG,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("RGB8 normal PNG must decode");
    }

    fn encode_png(image: &image::DynamicImage) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        image
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("write png");
        buf.into_inner()
    }

    // ---- T10: size mismatch stays fatal on the PNG path too ----------------

    #[test]
    fn normal_png_size_mismatch_still_fatal() {
        use crate::testdata;

        let err = decode_checked(
            testdata::GOLDEN_NORMAL_U8_PNG,
            testdata::GOLDEN_W,
            testdata::GOLDEN_H,
        )
        .expect_err("transposed size must be rejected");
        assert!(
            matches!(err, LoadNormalError::ReadPngError(_)),
            "expected a PNG read error, got {err:?}"
        );
    }

    // ---- T7: Deflate + FloatingPoint-predictor TIFFs -----------------------

    #[test]
    fn compressed_normal_tiff_decodes_bit_identical() {
        use crate::testdata;
        use tiff::encoder::{Compression, DeflateLevel};

        let (w, h) = (testdata::GOLDEN_W as u32, testdata::GOLDEN_H as u32);
        let values: Vec<f32> = testdata::PREDICTOR3_NORMAL_BITS
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect();

        // (a) Rust-written Deflate (the encoder cannot emit predictor 3).
        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf)
                .expect("tiff encoder")
                .with_compression(Compression::Deflate(DeflateLevel::Balanced));
            encoder
                .write_image::<colortype::RGB32Float>(w, h, &values)
                .expect("write deflate rgb f32 tiff");
        }
        let deflated = buf.into_inner();
        assert!(
            deflated != encode_rgb_f32(&values, w, h),
            "the deflate arm must actually differ from the uncompressed one"
        );
        let decoded = decode_checked(&deflated, testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect("deflate rgb f32 tiff must decode");
        for (got, want) in decoded.iter().zip(values.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }

        // (b) tifffile-written Deflate + FloatingPoint predictor (3).
        let decoded = decode_checked(
            testdata::PREDICTOR3_NORMAL_TIFF,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("predictor-3 normal fixture must decode");
        assert_eq!(decoded.len(), testdata::PREDICTOR3_NORMAL_BITS.len());
        for (i, (&got, &want_bits)) in decoded
            .iter()
            .zip(testdata::PREDICTOR3_NORMAL_BITS.iter())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                want_bits,
                "component {i}: got bits {:#010x}, want {want_bits:#010x}",
                got.to_bits()
            );
        }

        // Null model: the fixture really is predictor-shuffled on disk, so a
        // decoder that skipped the predictor pass could not have passed above.
        let raw_le: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(
            !testdata::PREDICTOR3_NORMAL_TIFF
                .windows(raw_le.len())
                .any(|window| window == raw_le),
            "predictor-3 fixture must not contain the plain little-endian samples"
        );
    }

    // ---- magic-byte dispatch (plan D3) -------------------------------------

    #[test]
    fn normal_unknown_magic_is_a_hard_error() {
        let err = decode_checked(&[0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0], 3, 4)
            .expect_err("a JPEG must not be accepted as a normal prior");
        let LoadNormalError::UnsupportedFormat(msg) = err else {
            panic!("expected UnsupportedFormat, got {err:?}");
        };
        assert!(
            msg.contains("ff d8 ff e0"),
            "message must show the magic: {msg}"
        );
        assert!(msg.contains("PNG") && msg.contains("TIFF"), "{msg}");
    }
}
