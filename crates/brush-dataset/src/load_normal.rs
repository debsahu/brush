use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use tokio::io::AsyncReadExt;

/// Lazily-loaded per-view surface-normal map: a 3-channel float32 TIFF stored
/// as `[H, W, 3]` in row-major, channel-interleaved order.
///
/// Convention (must match whatever writes the priors):
/// - normals live in the **camera frame**, `OpenCV` axes (+X right, +Y down,
///   +Z forward),
/// - unit length, oriented toward the camera (`n.z <= 0`),
/// - `(0, 0, 0)` marks an invalid / unobserved pixel.
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

/// Decode a 3-channel float32 TIFF and check it against the expected size.
/// Split out from `load_vec` so the decode + validation path is unit-testable
/// without a VFS.
pub(crate) fn decode_checked(
    bytes: &[u8],
    expected_h: usize,
    expected_w: usize,
) -> Result<Vec<f32>, LoadNormalError> {
    let (normal, w, h) = decode_f32_rgb_tiff(bytes)?;
    if w != expected_w || h != expected_h {
        Err(LoadNormalError::ReadTiffError(format!(
            "invalid normal size {w} x {h}, expected {expected_w} x {expected_h}"
        )))
    } else {
        Ok(normal)
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
}
