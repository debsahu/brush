use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use tokio::io::AsyncReadExt;

/// Lazily-loaded per-view depth map. Two wire formats are accepted, dispatched
/// on magic bytes (never on the file extension — prior discovery in the
/// `formats` module is extension-blind):
///
/// - **float32 TIFF**, single channel — depth straight in metres.
/// - **uint16 PNG**, single channel (`Luma16`) — depth in *millimetres*, the
///   quantized wire format (see `decode_depth_u16_mm`).
///
/// The **decoded contract is identical for both**: `[H, W]` f32, depth in
/// metres, `0` marks an invalid depth. Nothing downstream of this loader can
/// tell which format a prior came from.
#[derive(Clone, Debug)]
pub struct LoadDepth {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
}

impl PartialEq for LoadDepth {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl LoadDepth {
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
    ) -> Result<TensorData, LoadDepthError> {
        let depth = self.load_vec(expected_h, expected_w).await?;
        Ok(TensorData::new(depth, [expected_h, expected_w]))
    }

    pub async fn load_vec(
        &self,
        expected_h: usize,
        expected_w: usize,
    ) -> Result<Vec<f32>, LoadDepthError> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;

        decode_checked(&bytes, expected_h, expected_w)
    }
}

/// Which wire format a prior byte buffer claims to be, decided by magic bytes
/// alone (plan decision D3).
///
/// Extension is never consulted: `find_prior_path` matches on the stem only, so
/// a dataset can carry `depth/img.tiff` for one frame and `depth/img.png` for
/// the next and both resolve. Sniffing the bytes is what makes that safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PriorWireFormat {
    Tiff,
    Png,
}

/// Human-readable list of the magics we accept, for error messages.
pub(crate) const PRIOR_MAGIC_HELP: &str =
    r#"II*\0 / MM\0* (TIFF), II+\0 / MM\0+ (BigTIFF), \x89PNG\r\n\x1a\n (PNG)"#;

/// Classify a prior buffer by its leading magic bytes.
///
/// Returns `None` for anything that is neither TIFF nor PNG; callers turn that
/// into a hard error rather than guessing. Both classic TIFF (version 42) and
/// `BigTIFF` (version 43) are accepted, because the `tiff` crate decodes both
/// and refusing `BigTIFF` here would drop a format that loads today.
///
/// Lives in this module (rather than a new one) so the diff stays inside the
/// two prior loaders; [`crate::load_normal`] imports it.
pub(crate) fn prior_wire_format(bytes: &[u8]) -> Option<PriorWireFormat> {
    const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";
    const TIFF_LE: &[u8] = b"II\x2a\x00";
    const TIFF_BE: &[u8] = b"MM\x00\x2a";
    const BIGTIFF_LE: &[u8] = b"II\x2b\x00";
    const BIGTIFF_BE: &[u8] = b"MM\x00\x2b";

    if bytes.starts_with(PNG_MAGIC) {
        Some(PriorWireFormat::Png)
    } else if bytes.starts_with(TIFF_LE)
        || bytes.starts_with(TIFF_BE)
        || bytes.starts_with(BIGTIFF_LE)
        || bytes.starts_with(BIGTIFF_BE)
    {
        Some(PriorWireFormat::Tiff)
    } else {
        None
    }
}

/// Render the first few bytes of an unrecognised prior for its error message.
pub(crate) fn describe_magic(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for byte in bytes.iter().take(8) {
        let _ = write!(out, "{byte:02x} ");
    }
    if out.is_empty() {
        "<empty file>".to_owned()
    } else {
        out.trim_end().to_owned()
    }
}

/// Decode a depth prior of either wire format and check it against the
/// expected size. Split out from `load_vec` so the decode + validation path is
/// unit-testable without a VFS (mirrors [`crate::load_normal::decode_checked`]).
pub(crate) fn decode_checked(
    bytes: &[u8],
    expected_h: usize,
    expected_w: usize,
) -> Result<Vec<f32>, LoadDepthError> {
    let Some(format) = prior_wire_format(bytes) else {
        return Err(LoadDepthError::UnsupportedFormat(format!(
            "unsupported prior format (expected float32 TIFF or {{Luma16|RGB8}} PNG magic); \
             leading bytes [{}] match none of {PRIOR_MAGIC_HELP}",
            describe_magic(bytes)
        )));
    };

    let (depth, w, h) = match format {
        PriorWireFormat::Tiff => decode_f32_tiff(bytes)?,
        PriorWireFormat::Png => decode_u16_mm_png(bytes)?,
    };

    if w != expected_w || h != expected_h {
        // Keep the per-format error variant so the message never claims the
        // wrong container.
        let msg = format!("invalid depth size {w} x {h}, expected {expected_w} x {expected_h}");
        return Err(match format {
            PriorWireFormat::Tiff => LoadDepthError::ReadTiffError(msg),
            PriorWireFormat::Png => LoadDepthError::ReadPngError(msg),
        });
    }

    Ok(depth)
}

/// Decode a single-channel float32 TIFF into in row-major order.
fn decode_f32_tiff(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), LoadDepthError> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes))?;

    let tiff::decoder::DecodingResult::F32(depth) = decoder.read_image()? else {
        return Err(LoadDepthError::ReadTiffError(
            "unsupported TIFF sample format (expected float32 depth)".to_owned(),
        ));
    };

    let (w, h) = decoder.dimensions()?;
    let (w, h) = (w as usize, h as usize);

    if w * h != depth.len() {
        Err(LoadDepthError::ReadTiffError(
            "expected only a single channel".to_owned(),
        ))
    } else {
        Ok((depth, w, h))
    }
}

/// Dequantize one uint16 millimetre depth code to metres.
///
/// Codec contract taken from `gauss-surf` (Pablo Vela, Apache-2.0):
/// `render_io.py:146-161` decode, `uw_geometry.py:234-238` encode. `0` is the
/// invalid sentinel on both sides, and `0 / 1000.0 == 0.0` carries it through
/// untouched — no special case needed. The operation order is pinned to match
/// numpy's (plan D7) so every code, not just the anchors, is bit-identical
/// across the two languages.
#[inline]
pub(crate) fn decode_depth_u16_mm(mm: u16) -> f32 {
    f32::from(mm) / 1000.0
}

/// Decode a single-channel uint16 PNG of millimetre depths.
///
/// The pixel type is matched **exactly**: an 8-bit, RGB, or alpha-carrying PNG
/// is rejected rather than converted. `image`'s `to_luma16()` would happily
/// widen an 8-bit depth map into silently-wrong supervision (255 mm max), which
/// is the same class of error the float32-TIFF sample-format check exists to
/// prevent.
fn decode_u16_mm_png(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), LoadDepthError> {
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?;
    let color = image.color();

    let image::DynamicImage::ImageLuma16(gray) = image else {
        return Err(LoadDepthError::ReadPngError(format!(
            "unsupported depth PNG pixel type {color:?} \
             (expected 16-bit single-channel Luma16, millimetres)"
        )));
    };

    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let depth: Vec<f32> = gray
        .into_raw()
        .into_iter()
        .map(decode_depth_u16_mm)
        .collect();

    if w * h == depth.len() {
        Ok((depth, w, h))
    } else {
        Err(LoadDepthError::ReadPngError(format!(
            "expected {} samples for {w} x {h}, got {}",
            w * h,
            depth.len()
        )))
    }
}

#[derive(Error, Debug)]
pub enum LoadDepthError {
    #[error("I/O error while loading depth map: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error while loading TIFF file: {0}")]
    LoadTiffError(#[from] tiff::TiffError),

    #[error("Error while reading TIFF file: {0}")]
    ReadTiffError(String),

    #[error("Error while loading PNG file: {0}")]
    LoadPngError(#[from] image::ImageError),

    #[error("Error while reading PNG file: {0}")]
    ReadPngError(String),

    #[error("Error reading depth map: {0}")]
    UnsupportedFormat(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata;

    /// Encode `[H, W]` f32 samples as an uncompressed `Gray32Float` TIFF.
    fn encode_gray_f32(values: &[f32], w: u32, h: u32) -> Vec<u8> {
        use tiff::encoder::{TiffEncoder, colortype};

        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf).expect("tiff encoder");
            encoder
                .write_image::<colortype::Gray32Float>(w, h, values)
                .expect("write gray f32 tiff");
        }
        buf.into_inner()
    }

    /// Same, but Deflate-compressed (T7's Rust-written half).
    fn encode_gray_f32_deflate(values: &[f32], w: u32, h: u32) -> Vec<u8> {
        use tiff::encoder::{Compression, DeflateLevel, TiffEncoder, colortype};

        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf)
                .expect("tiff encoder")
                .with_compression(Compression::Deflate(DeflateLevel::Balanced));
            encoder
                .write_image::<colortype::Gray32Float>(w, h, values)
                .expect("write deflate gray f32 tiff");
        }
        buf.into_inner()
    }

    // ---- T1 / T18: golden uint16-millimetre PNG -> metres --------------------

    #[test]
    fn depth_png_u16mm_decodes_to_metres() {
        let decoded = decode_checked(
            testdata::GOLDEN_DEPTH_U16_PNG,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("golden uint16 depth PNG must decode");

        assert_eq!(decoded.len(), testdata::GOLDEN_DEPTH_MM.len());
        for (i, (&got, &want_bits)) in decoded
            .iter()
            .zip(testdata::GOLDEN_DEPTH_METRES_BITS.iter())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                want_bits,
                "pixel {i}: got {got} (bits {:#010x}), want bits {want_bits:#010x}",
                got.to_bits()
            );
        }

        // The table is not just "whatever the decoder said": derive it again
        // from the millimetre codes through the pinned D7 expression.
        for (i, &mm) in testdata::GOLDEN_DEPTH_MM.iter().enumerate() {
            assert_eq!(decoded[i].to_bits(), (f32::from(mm) / 1000.0f32).to_bits());
        }

        // Null model: a *different* pinned table must not pass. Scaling by
        // 1/1024 (a plausible "fast" mistake) disagrees on every nonzero code.
        let wrong: Vec<u32> = testdata::GOLDEN_DEPTH_MM
            .iter()
            .map(|&mm| (f32::from(mm) / 1024.0f32).to_bits())
            .collect();
        let agree = decoded
            .iter()
            .zip(wrong.iter())
            .filter(|(g, w)| g.to_bits() == **w)
            .count();
        assert_eq!(
            agree, 1,
            "only the 0-sentinel may coincide between /1000 and /1024"
        );
    }

    // ---- T4: the zero sentinel survives, and nothing near it is touched ------

    #[test]
    fn depth_png_zero_sentinel_preserved() {
        let decoded = decode_checked(
            testdata::GOLDEN_DEPTH_U16_PNG,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("golden uint16 depth PNG must decode");

        // Pixel 0 is code 0 -> exactly 0.0 (and +0.0, not -0.0).
        assert_eq!(testdata::GOLDEN_DEPTH_MM[0], 0);
        assert_eq!(decoded[0].to_bits(), 0.0f32.to_bits());

        // Its neighbours are untouched: no smoothing, no resampling, no
        // "spread the invalid pixel" anywhere in the load path (plan D8).
        assert_eq!(testdata::GOLDEN_DEPTH_MM[1], 1);
        assert_eq!(decoded[1].to_bits(), 0.001f32.to_bits());
        assert_eq!(
            decoded[testdata::GOLDEN_W].to_bits(),
            (f32::from(testdata::GOLDEN_DEPTH_MM[testdata::GOLDEN_W]) / 1000.0f32).to_bits()
        );

        // Exactly one invalid pixel in the fixture, and it is the one we placed.
        let invalid: Vec<usize> = decoded
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(invalid, vec![0]);
    }

    // ---- T5: wrong PNG pixel type is a hard error, never a conversion --------

    #[test]
    fn png_wrong_pixel_type_rejected() {
        // 8-bit grayscale: `to_luma16()` would accept this and silently cap
        // depth at 255 mm.
        let luma8 = image::DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(
                testdata::GOLDEN_W as u32,
                testdata::GOLDEN_H as u32,
                vec![7u8; testdata::GOLDEN_W * testdata::GOLDEN_H],
            )
            .expect("luma8 buffer"),
        );
        let err = decode_checked(&encode_png(&luma8), testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect_err("8-bit depth PNG must be rejected");
        assert!(
            matches!(err, LoadDepthError::ReadPngError(_)),
            "expected a PNG read error, got {err:?}"
        );

        // RGB8, i.e. someone handed the depth loader a normal map.
        let rgb8 = image::DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(
                testdata::GOLDEN_W as u32,
                testdata::GOLDEN_H as u32,
                vec![7u8; testdata::GOLDEN_W * testdata::GOLDEN_H * 3],
            )
            .expect("rgb8 buffer"),
        );
        let err = decode_checked(&encode_png(&rgb8), testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect_err("RGB depth PNG must be rejected");
        assert!(
            matches!(err, LoadDepthError::ReadPngError(_)),
            "got {err:?}"
        );

        // 16-bit *with alpha* — still not Luma16.
        let la16 = image::DynamicImage::ImageLumaA16(
            image::ImageBuffer::from_raw(
                testdata::GOLDEN_W as u32,
                testdata::GOLDEN_H as u32,
                vec![7u16; testdata::GOLDEN_W * testdata::GOLDEN_H * 2],
            )
            .expect("luma-alpha16 buffer"),
        );
        let err = decode_checked(&encode_png(&la16), testdata::GOLDEN_H, testdata::GOLDEN_W)
            .expect_err("16-bit grey+alpha depth PNG must be rejected");
        assert!(
            matches!(err, LoadDepthError::ReadPngError(_)),
            "got {err:?}"
        );
    }

    fn encode_png(image: &image::DynamicImage) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        image
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("write png");
        buf.into_inner()
    }

    // ---- T10: size mismatch stays fatal on the PNG path too -----------------

    #[test]
    fn png_size_mismatch_still_fatal() {
        // Same pixel count, transposed: must still be rejected.
        let err = decode_checked(
            testdata::GOLDEN_DEPTH_U16_PNG,
            testdata::GOLDEN_W,
            testdata::GOLDEN_H,
        )
        .expect_err("transposed size must be rejected");
        assert!(
            matches!(err, LoadDepthError::ReadPngError(_)),
            "expected a PNG read error, got {err:?}"
        );

        // And the correct size still passes, so the assert is not vacuous.
        decode_checked(
            testdata::GOLDEN_DEPTH_U16_PNG,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("correct size must decode");
    }

    // ---- T6: the float32 TIFF path is untouched ----------------------------

    #[test]
    fn round_trips_a_single_channel_f32_tiff() {
        let (w, h) = (4u32, 3u32);
        let values: Vec<f32> = (0..(w * h)).map(|i| i as f32 * 0.25 + 0.5).collect();
        let bytes = encode_gray_f32(&values, w, h);

        let decoded = decode_checked(&bytes, h as usize, w as usize).expect("f32 tiff must decode");
        for (got, want) in decoded.iter().zip(values.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn rejects_a_size_mismatch_tiff() {
        let (w, h) = (4u32, 3u32);
        let bytes = encode_gray_f32(&vec![1.0f32; (w * h) as usize], w, h);
        let err = decode_checked(&bytes, w as usize, h as usize)
            .expect_err("transposed size must be rejected");
        assert!(
            matches!(err, LoadDepthError::ReadTiffError(_)),
            "expected a TIFF read error, got {err:?}"
        );
    }

    #[test]
    fn rejects_a_non_f32_tiff() {
        use tiff::encoder::{TiffEncoder, colortype};

        let (w, h) = (4u32, 3u32);
        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf).expect("tiff encoder");
            encoder
                .write_image::<colortype::Gray16>(w, h, &vec![1u16; (w * h) as usize])
                .expect("write gray16 tiff");
        }
        let err = decode_checked(&buf.into_inner(), h as usize, w as usize)
            .expect_err("uint16 TIFF must not be read as float32 depth");
        assert!(
            matches!(err, LoadDepthError::ReadTiffError(_)),
            "expected a TIFF read error, got {err:?}"
        );
    }

    // ---- T7: Deflate + FloatingPoint-predictor TIFFs -----------------------

    #[test]
    fn compressed_tiff_decodes_bit_identical() {
        let (w, h) = (4u32, 3u32);
        let values: Vec<f32> = testdata::PREDICTOR3_DEPTH_BITS
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect();

        // (a) Rust-written Deflate (no predictor — the encoder cannot emit
        //     predictor 3, tiff-0.11.3 encoder/mod.rs:38).
        let deflated = encode_gray_f32_deflate(&values, w, h);
        let plain = encode_gray_f32(&values, w, h);
        assert!(
            deflated != plain,
            "the deflate arm must actually differ from the uncompressed one"
        );
        let decoded = decode_checked(&deflated, h as usize, w as usize)
            .expect("deflate f32 tiff must decode");
        for (got, want) in decoded.iter().zip(values.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }

        // (b) tifffile-written Deflate + FloatingPoint predictor (3), the
        //     format Part A actually ships. Decoded against pinned bits, so a
        //     silently-wrong predictor pass cannot hide behind a round trip.
        let decoded = decode_checked(
            testdata::PREDICTOR3_DEPTH_TIFF,
            testdata::GOLDEN_H,
            testdata::GOLDEN_W,
        )
        .expect("predictor-3 depth fixture must decode");
        assert_eq!(decoded.len(), testdata::PREDICTOR3_DEPTH_BITS.len());
        for (i, (&got, &want_bits)) in decoded
            .iter()
            .zip(testdata::PREDICTOR3_DEPTH_BITS.iter())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                want_bits,
                "pixel {i}: got bits {:#010x}, want {want_bits:#010x}",
                got.to_bits()
            );
        }

        // Null model: the predictor fixture is byte-shuffled on disk, so a
        // decoder that skipped the predictor pass would produce garbage, not
        // a near-miss. Confirm the raw strip bytes are NOT the plain f32 LE
        // bytes of the expected values.
        let raw_le: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(
            !testdata::PREDICTOR3_DEPTH_TIFF
                .windows(raw_le.len())
                .any(|w| w == raw_le),
            "predictor-3 fixture must not contain the plain little-endian samples"
        );
    }

    // ---- magic-byte dispatch (plan D3) -------------------------------------

    #[test]
    fn unknown_magic_is_a_hard_error() {
        // JPEG SOI: neither TIFF nor PNG.
        let err = decode_checked(&[0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0], 3, 4)
            .expect_err("a JPEG must not be accepted as a depth prior");
        let LoadDepthError::UnsupportedFormat(msg) = err else {
            panic!("expected UnsupportedFormat, got {err:?}");
        };
        assert!(
            msg.contains("ff d8 ff e0"),
            "message must show the magic: {msg}"
        );
        assert!(msg.contains("PNG") && msg.contains("TIFF"), "{msg}");

        // Empty buffer must not panic.
        assert!(matches!(
            decode_checked(&[], 3, 4),
            Err(LoadDepthError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn magic_classification_is_exact() {
        assert_eq!(
            prior_wire_format(b"II\x2a\x00rest"),
            Some(PriorWireFormat::Tiff)
        );
        assert_eq!(
            prior_wire_format(b"MM\x00\x2arest"),
            Some(PriorWireFormat::Tiff)
        );
        assert_eq!(
            prior_wire_format(b"II\x2b\x00rest"),
            Some(PriorWireFormat::Tiff)
        );
        assert_eq!(
            prior_wire_format(b"MM\x00\x2brest"),
            Some(PriorWireFormat::Tiff)
        );
        assert_eq!(
            prior_wire_format(b"\x89PNG\r\n\x1a\nrest"),
            Some(PriorWireFormat::Png)
        );
        // Near misses: one byte off in either magic is not a match.
        // (`\x0a` IS `\n`, so the PNG near-miss has to differ elsewhere --
        // caught by this test failing when it did not.)
        assert_eq!(prior_wire_format(b"II\x2a\x01rest"), None);
        assert_eq!(prior_wire_format(b"\x89PNG\r\n\x1a\x0brest"), None);
        assert_eq!(prior_wire_format(b"\x88PNG\r\n\x1a\nrest"), None);
        assert_eq!(prior_wire_format(b"MM\x00\x2c"), None);
        assert_eq!(prior_wire_format(b"II\x2a"), None);
        assert_eq!(prior_wire_format(b""), None);

        // And the real fixtures classify as advertised.
        assert_eq!(
            prior_wire_format(testdata::GOLDEN_DEPTH_U16_PNG),
            Some(PriorWireFormat::Png)
        );
        assert_eq!(
            prior_wire_format(testdata::PREDICTOR3_DEPTH_TIFF),
            Some(PriorWireFormat::Tiff)
        );
    }
}
