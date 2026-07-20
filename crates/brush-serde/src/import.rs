use std::pin::pin;
use std::time::Duration;

use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid};
use brush_render::sh::rgb_to_sh;
use glam::{Vec3, Vec4Swizzles};
use serde::Deserialize;
use serde::de::{DeserializeSeed, Error};
use serde_ply::{DeserializeError, PlyChunkedReader, RowVisitor};
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio_stream::{Stream, StreamExt};

use crate::ply_gaussian::{PlyGaussian, QuantSh, QuantSplat};

type StreamEmitter = TryStreamEmitter<SplatMessage, DeserializeError>;

pub struct ParseMetadata {
    pub up_axis: Option<Vec3>,
    pub render_mode: Option<SplatRenderMode>,
    pub total_splats: u32,
    pub progress: f32,
}

/// Raw splat data parsed from a PLY file.
/// Fields are optional - only positions are guaranteed.
#[derive(Clone)]
pub struct SplatData {
    /// Position data (x, y, z) - always present
    pub means: Vec<f32>,
    pub rotations: Option<Vec<f32>>,
    pub log_scales: Option<Vec<f32>>,
    pub sh_coeffs: Option<Vec<f32>>,
    pub raw_opacities: Option<Vec<f32>>,
}

impl SplatData {
    pub fn num_splats(&self) -> usize {
        self.means.len() / 3
    }

    /// Strided subsample down to at most `max_splats` points.
    ///
    /// COLMAP / large PLY initialisations can hold far more points than the
    /// training budget. Constructing GPU tensors for all of them blows the
    /// buffer-size limit before training even starts, so cap the initial
    /// point count here. No-op when already within budget.
    pub fn subsample(self, max_splats: usize) -> Self {
        let n = self.num_splats();
        if max_splats == 0 || n <= max_splats {
            return self;
        }
        // Ceil so the result never exceeds `max_splats`.
        let step = n.div_ceil(max_splats);

        let pick = |v: &[f32], stride: usize| -> Vec<f32> {
            v.chunks_exact(stride)
                .step_by(step)
                .flatten()
                .copied()
                .collect()
        };

        let sh_stride = self.sh_coeffs.as_deref().map_or(0, |c| c.len() / n);

        Self {
            means: pick(&self.means, 3),
            rotations: self.rotations.as_deref().map(|v| pick(v, 4)),
            log_scales: self.log_scales.as_deref().map(|v| pick(v, 3)),
            sh_coeffs: self.sh_coeffs.as_deref().map(|v| pick(v, sh_stride)),
            raw_opacities: self.raw_opacities.as_deref().map(|v| pick(v, 1)),
        }
    }

    /// Convert into Splats using simple defaults for missing fields.
    pub fn into_splats(self, device: &burn::tensor::Device, mode: SplatRenderMode) -> Splats {
        let n_splats = self.num_splats();
        let rotations = self
            .rotations
            .unwrap_or_else(|| [1.0, 0.0, 0.0, 0.0].repeat(n_splats));
        let log_scales = self.log_scales.unwrap_or_else(|| vec![-4.0; n_splats * 3]);
        let sh_coeffs = self.sh_coeffs.unwrap_or_else(|| vec![0.5; n_splats * 3]);
        let opacities = self
            .raw_opacities
            .unwrap_or_else(|| vec![inverse_sigmoid(0.5); n_splats]);

        Splats::from_raw(
            self.means, rotations, log_scales, sh_coeffs, opacities, mode, device,
        )
    }
}

pub struct SplatMessage {
    pub meta: ParseMetadata,
    pub data: SplatData,
}

enum PlyFormat {
    Ply,
    SuperSplatCompressed,
}

struct TimedUpdate {
    last_update: web_time::Instant,
    update_every: Option<web_time::Duration>,
}

impl TimedUpdate {
    fn new(update_every: Option<web_time::Duration>) -> Self {
        Self {
            last_update: web_time::Instant::now(),
            update_every,
        }
    }

    fn should_update(&mut self, perc_done: f32) -> bool {
        // Don't bother updating if we're almost done
        if perc_done >= 0.95 {
            return false;
        }
        if let Some(duration) = self.update_every
            && self.last_update.elapsed() >= duration
        {
            self.last_update = web_time::Instant::now();
            return true;
        }

        false
    }
}

fn interleave_coeffs(sh_dc: Vec3, sh_rest: &[f32], result: &mut Vec<f32>) {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;

    result.extend([sh_dc.x, sh_dc.y, sh_dc.z]);
    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
}

async fn read_chunk<T: AsyncRead + Unpin>(
    mut reader: T,
    buf: &mut Vec<u8>,
) -> tokio::io::Result<usize> {
    buf.reserve(8 * 1024 * 1024);
    let mut total_read = 0;
    while buf.len() < buf.capacity() {
        let bytes_read = reader.read_buf(buf).await?;
        if bytes_read == 0 {
            break;
        }
        total_read += bytes_read;
        brush_async::yield_now().await;
    }
    Ok(total_read)
}

fn unexpected_eof() -> DeserializeError {
    DeserializeError::custom("Unexpected EOF while reading PLY data")
}

pub async fn load_splat_from_ply<T: AsyncRead + Unpin>(
    reader: T,
    subsample_points: Option<u32>,
) -> Result<SplatMessage, DeserializeError> {
    let stream = stream_splat_from_ply(reader, subsample_points, false);
    let mut stream = pin!(stream);
    let mut last = None;
    while let Some(message) = stream.next().await {
        last = Some(message?);
    }
    last.ok_or_else(|| DeserializeError::custom("Couldn't load single splat from ply"))
}

pub fn stream_splat_from_ply<T: AsyncRead + Unpin>(
    mut reader: T,
    subsample_points: Option<u32>,
    streaming: bool,
) -> impl Stream<Item = Result<SplatMessage, DeserializeError>> {
    try_fn_stream(|emitter| async move {
        let subsample = match subsample_points {
            Some(0) => {
                return Err(DeserializeError::custom(
                    "subsample_points must be greater than zero",
                ));
            }
            Some(value) => value as usize,
            None => 1,
        };
        let mut file = PlyChunkedReader::new();
        let bytes_read = read_chunk(&mut reader, file.buffer_mut()).await?;

        if bytes_read == 0 {
            return Err(unexpected_eof());
        }

        let header = file
            .header()
            .ok_or_else(|| DeserializeError::custom("missing PLY header"))?;
        // Parse some metadata.
        let up_axis = header
            .comments
            .iter()
            .filter_map(|c| {
                let s = c.to_lowercase();
                let suffix = s.strip_prefix("vertical axis: ")?.trim();
                match suffix {
                    "x" => Some(Vec3::X),
                    "y" => Some(Vec3::NEG_Y),
                    "z" => Some(Vec3::NEG_Z),
                    _ => {
                        let parts: Vec<f32> = suffix
                            .split(|ch: char| {
                                ch == ',' || ch.is_whitespace() || ch == '[' || ch == ']'
                            })
                            .filter(|s| !s.is_empty())
                            .filter_map(|p| p.parse::<f32>().ok())
                            .collect();
                        if parts.len() == 3 {
                            Some(Vec3::new(parts[0], parts[1], parts[2]))
                        } else {
                            None
                        }
                    }
                }
            })
            .next_back();

        let render_mode = header
            .comments
            .iter()
            .filter_map(|c| {
                match c
                    .to_lowercase()
                    .strip_prefix("splatrendermode: ")
                    .map(|s| s.trim())
                {
                    Some("mip") => Some(SplatRenderMode::Mip),
                    Some("default") => Some(SplatRenderMode::Default),
                    _ => None,
                }
            })
            .next_back();

        // Check whether there is a vertex header that has at least XYZ.
        let has_vertex = header.elem_defs.iter().any(|el| el.name == "vertex");

        let ply_type = if has_vertex
            && header
                .elem_defs
                .first()
                .is_some_and(|el| el.name == "chunk")
        {
            PlyFormat::SuperSplatCompressed
        } else if has_vertex {
            PlyFormat::Ply
        } else {
            return Err(DeserializeError::custom("Unknown format"));
        };

        let mut updater = TimedUpdate::new(streaming.then(|| Duration::from_millis(1500)));

        match ply_type {
            PlyFormat::Ply => {
                parse_ply(
                    reader,
                    subsample,
                    &mut file,
                    up_axis,
                    &emitter,
                    render_mode,
                    &mut updater,
                )
                .await?;
            }
            PlyFormat::SuperSplatCompressed => {
                parse_compressed_ply(
                    reader,
                    subsample,
                    file,
                    up_axis,
                    emitter,
                    render_mode,
                    updater,
                )
                .await?;
            }
        }
        Ok(())
    })
}

fn progress(completed: usize, len: usize) -> f32 {
    if len == 0 {
        1.0
    } else {
        completed.min(len) as f32 / len as f32
    }
}

fn validate_sh_rest_properties<'a>(
    names: impl Iterator<Item = &'a str>,
) -> Result<usize, DeserializeError> {
    let mut indices = names
        .map(|name| {
            name.strip_prefix("f_rest_")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .ok_or_else(|| {
                    DeserializeError::custom(format!("Invalid SH property name: {name}"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    indices.sort_unstable();

    if indices.iter().copied().ne(0..indices.len()) {
        return Err(DeserializeError::custom(
            "SH rest properties must be contiguous from f_rest_0",
        ));
    }
    if !matches!(indices.len(), 0 | 9 | 24 | 45 | 72) {
        return Err(DeserializeError::custom(format!(
            "Unsupported SH rest property count: {}",
            indices.len()
        )));
    }

    Ok(indices.len())
}

fn vec_exact(cap: usize) -> Vec<f32> {
    let mut r = vec![];
    r.reserve_exact(cap);
    r
}

async fn parse_ply<T: AsyncRead + Unpin>(
    mut reader: T,
    subsample: usize,
    file: &mut PlyChunkedReader,
    up_axis: Option<Vec3>,
    emitter: &StreamEmitter,
    render_mode: Option<SplatRenderMode>,
    update: &mut TimedUpdate,
) -> Result<(), DeserializeError> {
    let header = file
        .header()
        .ok_or_else(|| DeserializeError::custom("missing PLY header"))?;
    let vertex = header
        .get_element("vertex")
        .ok_or(DeserializeError::custom("Unknown format"))?;
    let is_ascii = header.format == serde_ply::PlyFormat::Ascii;
    let total_splats = vertex.count;
    let max_splats = total_splats.div_ceil(subsample);
    if total_splats == 0 {
        return Err(DeserializeError::custom("PLY contains no vertices"));
    }

    let rest_count = validate_sh_rest_properties(
        vertex
            .properties
            .iter()
            .filter(|property| property.name.starts_with("f_rest_"))
            .map(|property| property.name.as_str()),
    )?;
    let dc_count = vertex
        .properties
        .iter()
        .filter(|property| matches!(property.name.as_str(), "f_dc_0" | "f_dc_1" | "f_dc_2"))
        .count();
    let channel_count = |short: &str, long: &str| {
        vertex
            .properties
            .iter()
            .filter(|property| property.name == short || property.name == long)
            .count()
    };
    let rgb_channels = [
        channel_count("r", "red"),
        channel_count("g", "green"),
        channel_count("b", "blue"),
    ];
    let has_rgb = rgb_channels.iter().all(|&count| count == 1);
    if dc_count != 0 && dc_count != 3 {
        return Err(DeserializeError::custom(
            "PLY SH colors require f_dc_0, f_dc_1, and f_dc_2",
        ));
    }
    if rgb_channels.iter().any(|&count| count != 0) && !has_rgb {
        return Err(DeserializeError::custom(
            "PLY RGB colors require exactly one red, green, and blue property",
        ));
    }
    let has_color = dc_count == 3 || has_rgb;
    if rest_count > 0 && !has_color {
        return Err(DeserializeError::custom(
            "PLY SH rest properties require a complete DC or RGB color",
        ));
    }
    let sh_count = if has_color { rest_count + 3 } else { 0 };

    let mut data = SplatData {
        means: vec_exact(max_splats * 3),
        rotations: vertex
            .has_property("rot_0")
            .then(|| vec_exact(max_splats * 4)),
        log_scales: vertex
            .has_property("scale_0")
            .then(|| vec_exact(max_splats * 3)),
        sh_coeffs: (sh_count > 0).then(|| vec_exact(max_splats * sh_count)),
        raw_opacities: vertex
            .has_property("opacity")
            .then(|| vec_exact(max_splats)),
    };

    let mut row_index: usize = 0;

    loop {
        let bytes_read = read_chunk(&mut reader, file.buffer_mut()).await?;

        // ASCII values are delimiter-terminated. At a real EOF, a synthetic
        // newline lets the parser accept a complete final token while malformed
        // or incomplete rows still fail deserialization below.
        if bytes_read == 0 && is_ascii {
            let buffer = file.buffer_mut();
            if buffer
                .last()
                .is_some_and(|byte| !byte.is_ascii_whitespace())
            {
                buffer.push(b'\n');
            }
        }

        RowVisitor::new(|mut gauss: PlyGaussian| {
            row_index += 1;
            if !(row_index - 1).is_multiple_of(subsample) {
                return;
            }
            data.means.extend([gauss.x, gauss.y, gauss.z]);

            // Prefer rgb if specified.
            if let Some(r) = gauss.red
                && let Some(g) = gauss.green
                && let Some(b) = gauss.blue
            {
                let sh_dc = rgb_to_sh(Vec3::new(r, g, b));
                gauss.f_dc_0 = sh_dc.x;
                gauss.f_dc_1 = sh_dc.y;
                gauss.f_dc_2 = sh_dc.z;
            }

            if let Some(coeffs) = &mut data.sh_coeffs {
                interleave_coeffs(
                    Vec3::new(gauss.f_dc_0, gauss.f_dc_1, gauss.f_dc_2),
                    &gauss.sh_rest_coeffs()[..sh_count - 3],
                    coeffs,
                );
            }

            if let Some(scales) = &mut data.log_scales {
                scales.extend([gauss.scale_0, gauss.scale_1, gauss.scale_2]);
            }
            if let Some(rotation) = &mut data.rotations {
                rotation.extend([gauss.rot_0, gauss.rot_1, gauss.rot_2, gauss.rot_3]);
            }
            if let Some(opacity) = &mut data.raw_opacities {
                opacity.push(gauss.opacity);
            }
        })
        .deserialize(&mut *file)?;

        if update.should_update(row_index as f32 / total_splats as f32) || row_index == total_splats
        {
            let meta = ParseMetadata {
                total_splats: max_splats as u32,
                up_axis,
                progress: progress(row_index, total_splats),
                render_mode,
            };

            if row_index == total_splats {
                emitter.emit(SplatMessage { meta, data }).await;
                return Ok(());
            } else {
                emitter
                    .emit(SplatMessage {
                        meta,
                        data: data.clone(),
                    })
                    .await;
            }
        }

        if bytes_read == 0 {
            return Err(unexpected_eof());
        }
    }
}

async fn parse_compressed_ply<T: AsyncRead + Unpin>(
    mut reader: T,
    subsample: usize,
    mut file: PlyChunkedReader,
    up_axis: Option<Vec3>,
    emitter: StreamEmitter,
    render_mode: Option<SplatRenderMode>,
    mut update: TimedUpdate,
) -> Result<(), DeserializeError> {
    const SPLATS_PER_CHUNK: usize = 256;

    #[derive(Default, Deserialize)]
    struct QuantMeta {
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        min_z: f32,
        max_z: f32,
        min_scale_x: f32,
        max_scale_x: f32,
        min_scale_y: f32,
        max_scale_y: f32,
        min_scale_z: f32,
        max_scale_z: f32,
        min_r: f32,
        max_r: f32,
        min_g: f32,
        max_g: f32,
        min_b: f32,
        max_b: f32,
    }

    impl QuantMeta {
        fn mean(&self, raw: Vec3) -> Vec3 {
            let min = glam::vec3(self.min_x, self.min_y, self.min_z);
            let max = glam::vec3(self.max_x, self.max_y, self.max_z);
            raw * (max - min) + min
        }

        fn scale(&self, raw: Vec3) -> Vec3 {
            let min = glam::vec3(self.min_scale_x, self.min_scale_y, self.min_scale_z);
            let max = glam::vec3(self.max_scale_x, self.max_scale_y, self.max_scale_z);
            raw * (max - min) + min
        }

        fn color(&self, raw: Vec3) -> Vec3 {
            let min = glam::vec3(self.min_r, self.min_g, self.min_b);
            let max = glam::vec3(self.max_r, self.max_g, self.max_b);
            raw * (max - min) + min
        }
    }

    let mut quant_metas = vec![];

    while let Some(element) = file.current_element()
        && element.name == "chunk"
    {
        let bytes_read = if element.count == 0 {
            0
        } else {
            read_chunk(&mut reader, file.buffer_mut()).await?
        };
        RowVisitor::new(|meta: QuantMeta| {
            quant_metas.push(meta);
        })
        .deserialize(&mut file)?;

        if bytes_read == 0
            && file
                .current_element()
                .is_some_and(|element| element.name == "chunk")
        {
            return Err(unexpected_eof());
        }
    }

    let vertex = file
        .current_element()
        .ok_or(DeserializeError::custom("Unknown format"))?;

    if vertex.name != "vertex" {
        return Err(DeserializeError::custom("Unknown format"));
    }
    let total_splats = vertex.count;
    let max_splats = total_splats.div_ceil(subsample);
    if total_splats == 0 {
        return Err(DeserializeError::custom("PLY contains no vertices"));
    }
    let required_chunks = total_splats.div_ceil(SPLATS_PER_CHUNK);
    if quant_metas.len() < required_chunks {
        return Err(DeserializeError::custom(format!(
            "Compressed PLY has {} chunk rows but needs at least {required_chunks} for {total_splats} vertices",
            quant_metas.len()
        )));
    }

    let mut means = Vec::with_capacity(max_splats * 3);
    // Atm, unlike normal plys, these values aren't optional.
    let mut log_scales = Vec::with_capacity(max_splats * 3);
    let mut rotations = Vec::with_capacity(max_splats * 4);
    let mut sh_coeffs = Vec::with_capacity(max_splats * 3);
    let mut opacity = Vec::with_capacity(max_splats);

    let mut row_count = 0;

    let sh_vals = file
        .header()
        .ok_or_else(|| DeserializeError::custom("missing PLY header"))?
        .elem_defs
        .get(2)
        .cloned();
    if let Some(sh) = &sh_vals {
        if sh.name != "sh" || sh.count != total_splats {
            return Err(DeserializeError::custom(format!(
                "Invalid compressed PLY SH element: expected {total_splats} rows"
            )));
        }
        validate_sh_rest_properties(sh.properties.iter().map(|property| property.name.as_str()))?;
    }

    while let Some(element) = file.current_element()
        && element.name == "vertex"
    {
        let bytes_read = read_chunk(&mut reader, file.buffer_mut()).await?;

        RowVisitor::new(|splat: QuantSplat| {
            let quant_data = &quant_metas[row_count / SPLATS_PER_CHUNK];
            row_count += 1;
            if !(row_count - 1).is_multiple_of(subsample) {
                return;
            }
            means.extend(quant_data.mean(splat.mean).to_array());
            log_scales.extend(quant_data.scale(splat.log_scale).to_array());
            // Nb: Scalar order.
            rotations.extend([
                splat.rotation.w,
                splat.rotation.x,
                splat.rotation.y,
                splat.rotation.z,
            ]);
            // Compressed ply specifies things in post-activated values. Convert to pre-activated values.
            opacity.push(inverse_sigmoid(splat.rgba.w));
            // These come in as RGB colors. Convert to base SH coefficients.
            let sh_dc = rgb_to_sh(quant_data.color(splat.rgba.xyz()));
            sh_coeffs.extend([sh_dc.x, sh_dc.y, sh_dc.z]);
        })
        .deserialize(&mut file)?;

        // Occasionally send some updated splats.
        if update.should_update(row_count as f32 / total_splats as f32) || row_count == total_splats
        {
            // Leave 20% of progress for loading the SH's, just an estimate.
            let max_time = if sh_vals.is_some() { 0.8 } else { 1.0 };
            let progress = progress(row_count, total_splats) * max_time;
            let meta = ParseMetadata {
                total_splats: max_splats as u32,
                up_axis,
                progress,
                render_mode,
            };

            let data = SplatData {
                means: means.clone(),
                rotations: Some(rotations.clone()),
                log_scales: Some(log_scales.clone()),
                sh_coeffs: Some(sh_coeffs.clone()),
                raw_opacities: Some(opacity.clone()),
            };
            emitter.emit(SplatMessage { meta, data }).await;
        }

        if bytes_read == 0
            && file
                .current_element()
                .is_some_and(|element| element.name == "vertex")
        {
            return Err(unexpected_eof());
        }
    }

    if let Some(sh_vals) = sh_vals {
        let sh_count = sh_vals.properties.len();
        let mut total_coeffs = Vec::with_capacity(sh_vals.count * (3 + sh_count));
        let mut splat_index = 0;

        let mut row_count: usize = 0;

        while let Some(element) = file.current_element()
            && element.name == "sh"
        {
            let bytes_read = if element.count == 0 {
                0
            } else {
                read_chunk(&mut reader, file.buffer_mut()).await?
            };

            RowVisitor::new(|quant_sh: QuantSh| {
                row_count += 1;
                if !(row_count - 1).is_multiple_of(subsample) {
                    return;
                }
                let dc = glam::vec3(
                    sh_coeffs[splat_index * 3],
                    sh_coeffs[splat_index * 3 + 1],
                    sh_coeffs[splat_index * 3 + 2],
                );
                interleave_coeffs(
                    dc,
                    &quant_sh.sh_rest_coeffs()[..sh_count],
                    &mut total_coeffs,
                );
                splat_index += 1;
            })
            .deserialize(&mut file)?;

            if bytes_read == 0
                && file
                    .current_element()
                    .is_some_and(|element| element.name == "sh")
            {
                return Err(unexpected_eof());
            }
        }

        let meta = ParseMetadata {
            total_splats: (means.len() / 3) as u32,
            up_axis,
            progress: 1.0,
            render_mode,
        };
        let data = SplatData {
            means,
            rotations: Some(rotations),
            log_scales: Some(log_scales),
            sh_coeffs: Some(total_coeffs),
            raw_opacities: Some(opacity),
        };
        emitter.emit(SplatMessage { meta, data }).await;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::splat_to_ply;
    use crate::test_utils::{create_test_splats, create_test_splats_with_count};
    use brush_render::sh::sh_coeffs_for_degree;
    use std::io::Cursor;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn truncated_binary_ply() -> Vec<u8> {
        let mut data = b"ply\n\
format binary_little_endian 1.0\n\
element vertex 2\n\
property float x\n\
property float y\n\
property float z\n\
end_header\n"
            .to_vec();

        for value in [1.0_f32, 2.0, 3.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        // Only the first property of the second vertex is present.
        data.extend_from_slice(&4.0_f32.to_le_bytes());
        data
    }

    fn ascii_ply(vertex_count: usize, rows: &str) -> Vec<u8> {
        format!(
            "ply\n\
             format ascii 1.0\n\
             element vertex {vertex_count}\n\
             property float x\n\
             property float y\n\
             property float z\n\
             end_header\n\
             {rows}"
        )
        .into_bytes()
    }

    fn ascii_ply_with_properties(properties: &[String]) -> Vec<u8> {
        let mut data = b"ply\n\
format ascii 1.0\n\
element vertex 1\n\
property float x\n\
property float y\n\
property float z\n"
            .to_vec();
        for property in properties {
            data.extend_from_slice(format!("property float {property}\n").as_bytes());
        }
        data.extend_from_slice(b"end_header\n");
        data
    }

    fn compressed_ply_header(
        chunk_count: usize,
        vertex_count: usize,
        sh: Option<(usize, usize)>,
    ) -> Vec<u8> {
        let mut header = format!(
            "ply\n\
             format binary_little_endian 1.0\n\
             element chunk {chunk_count}\n\
             property float min_x\n\
             property float max_x\n\
             property float min_y\n\
             property float max_y\n\
             property float min_z\n\
             property float max_z\n\
             property float min_scale_x\n\
             property float max_scale_x\n\
             property float min_scale_y\n\
             property float max_scale_y\n\
             property float min_scale_z\n\
             property float max_scale_z\n\
             property float min_r\n\
             property float max_r\n\
             property float min_g\n\
             property float max_g\n\
             property float min_b\n\
             property float max_b\n\
             element vertex {vertex_count}\n\
             property uint packed_position\n\
             property uint packed_scale\n\
             property uint packed_rotation\n\
             property uint packed_color\n"
        )
        .into_bytes();
        if let Some((sh_count, property_count)) = sh {
            header.extend_from_slice(format!("element sh {sh_count}\n").as_bytes());
            for index in 0..property_count {
                header.extend_from_slice(format!("property uchar f_rest_{index}\n").as_bytes());
            }
        }
        header.extend_from_slice(b"end_header\n");
        header
    }

    fn truncated_compressed_chunk_ply() -> Vec<u8> {
        let mut data = compressed_ply_header(1, 1, None);
        data.extend_from_slice(&0_f32.to_le_bytes());
        data
    }

    fn truncated_compressed_vertex_ply() -> Vec<u8> {
        let mut data = compressed_ply_header(1, 1, None);
        data.extend_from_slice(&[0; 18 * std::mem::size_of::<f32>()]);
        data.extend_from_slice(&0_u32.to_le_bytes());
        data
    }

    fn truncated_compressed_sh_ply() -> Vec<u8> {
        let mut data = compressed_ply_header(1, 1, Some((1, 9)));
        data.extend_from_slice(&[0; 18 * std::mem::size_of::<f32>()]);
        data.extend_from_slice(&[0; 4 * std::mem::size_of::<u32>()]);
        data
    }

    fn compressed_ply_with_sh() -> Vec<u8> {
        let mut data = compressed_ply_header(1, 1, Some((1, 9)));
        data.extend_from_slice(&[0; 18 * std::mem::size_of::<f32>()]);
        data.extend_from_slice(&[0; 4 * std::mem::size_of::<u32>()]);
        data.extend_from_slice(&[0, 32, 64, 96, 127, 160, 192, 224, 255]);
        data
    }

    fn compressed_ply_one_vertex() -> Vec<u8> {
        let mut data = compressed_ply_header(1, 1, None);
        data.extend_from_slice(&[0; 18 * std::mem::size_of::<f32>()]);
        data.extend_from_slice(&[0; 4 * std::mem::size_of::<u32>()]);
        data
    }

    fn compressed_ply_with_oversized_sh_header() -> Vec<u8> {
        let mut data = compressed_ply_header(1, 1, Some((1, 75)));
        data.extend_from_slice(&[0; 18 * std::mem::size_of::<f32>()]);
        data
    }

    fn compressed_ply_without_chunk_metadata() -> Vec<u8> {
        let mut data = compressed_ply_header(0, 1, None);
        data.extend_from_slice(&[0; 4 * std::mem::size_of::<u32>()]);
        data
    }

    #[cfg(not(target_family = "wasm"))]
    async fn assert_stream_returns_error(data: Vec<u8>) -> DeserializeError {
        tokio::time::timeout(Duration::from_secs(1), async move {
            let stream = stream_splat_from_ply(Cursor::new(data), None, false);
            let mut stream = std::pin::pin!(stream);
            while let Some(result) = stream.next().await {
                if let Err(error) = result {
                    return error;
                }
            }
            panic!("truncated PLY stream ended without an error");
        })
        .await
        .expect("truncated PLY parser timed out")
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_truncated_ply_returns_error() {
        let error = assert_stream_returns_error(truncated_binary_ply()).await;
        assert!(error.to_string().contains("Unexpected EOF"));
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_partial_ascii_token_returns_error() {
        assert_stream_returns_error(ascii_ply(1, "1.0 2.0 3e")).await;
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_truncated_compressed_chunk_returns_error() {
        let error = assert_stream_returns_error(truncated_compressed_chunk_ply()).await;
        assert!(error.to_string().contains("Unexpected EOF"));
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_truncated_compressed_vertex_returns_error() {
        let error = assert_stream_returns_error(truncated_compressed_vertex_ply()).await;
        assert!(error.to_string().contains("Unexpected EOF"));
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_truncated_compressed_sh_returns_error() {
        let error = assert_stream_returns_error(truncated_compressed_sh_ply()).await;
        assert!(error.to_string().contains("Unexpected EOF"));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_one_shot_compressed_import_drains_to_final_message() {
        let imported = load_splat_from_ply(Cursor::new(compressed_ply_with_sh()), None)
            .await
            .expect("complete compressed PLY should load");

        assert_eq!(imported.meta.progress, 1.0);
        assert_eq!(imported.data.sh_coeffs.expect("SH coefficients").len(), 12);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_one_shot_compressed_import_reports_truncated_sh() {
        let error = load_splat_from_ply(Cursor::new(truncated_compressed_sh_ply()), None)
            .await
            .err()
            .expect("truncated SH payload should fail");
        assert!(error.to_string().contains("Unexpected EOF"));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_compressed_import_rejects_missing_chunk_metadata() {
        let error = load_splat_from_ply(Cursor::new(compressed_ply_without_chunk_metadata()), None)
            .await
            .err()
            .expect("missing chunk metadata should fail");
        assert!(error.to_string().contains("chunk rows"));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_import_rejects_zero_subsample() {
        let error = load_splat_from_ply(Cursor::new(ascii_ply(1, "1.0 2.0 3.0")), Some(0))
            .await
            .err()
            .expect("zero subsample should fail");
        assert!(error.to_string().contains("greater than zero"));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_regular_subsample_larger_than_input_keeps_first_row() {
        let imported = load_splat_from_ply(Cursor::new(ascii_ply(1, "1.0 2.0 3.0")), Some(2))
            .await
            .expect("first regular row should survive strided subsampling");
        assert_eq!(imported.data.means, vec![1.0, 2.0, 3.0]);
        assert_eq!(imported.meta.total_splats, 1);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_compressed_subsample_larger_than_input_keeps_first_row() {
        let imported = load_splat_from_ply(Cursor::new(compressed_ply_one_vertex()), Some(2))
            .await
            .expect("first compressed row should survive strided subsampling");
        assert_eq!(imported.data.num_splats(), 1);
        assert_eq!(imported.meta.total_splats, 1);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_regular_import_rejects_partial_rgb() {
        let error = load_splat_from_ply(
            Cursor::new(ascii_ply_with_properties(&["red".to_owned()])),
            None,
        )
        .await
        .err()
        .expect("partial RGB schema should fail");
        assert!(
            error
                .to_string()
                .contains("exactly one red, green, and blue")
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_regular_import_rejects_oversized_sh_schema() {
        let mut properties = vec![
            "f_dc_0".to_owned(),
            "f_dc_1".to_owned(),
            "f_dc_2".to_owned(),
        ];
        properties.extend((0..75).map(|index| format!("f_rest_{index}")));
        let error = load_splat_from_ply(Cursor::new(ascii_ply_with_properties(&properties)), None)
            .await
            .err()
            .expect("oversized SH schema should fail");
        assert!(
            error
                .to_string()
                .contains("Unsupported SH rest property count")
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_compressed_import_rejects_oversized_sh_schema() {
        let error =
            load_splat_from_ply(Cursor::new(compressed_ply_with_oversized_sh_header()), None)
                .await
                .err()
                .expect("oversized compressed SH schema should fail");
        assert!(
            error
                .to_string()
                .contains("Unsupported SH rest property count")
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_ascii_ply_accepts_final_token_without_newline() {
        let imported = load_splat_from_ply(Cursor::new(ascii_ply(1, "1.0 2.0 3.0")), None)
            .await
            .expect("complete final ASCII token should load");

        assert_eq!(imported.data.means, vec![1.0, 2.0, 3.0]);
        assert_eq!(imported.meta.progress, 1.0);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_zero_row_regular_ply_returns_error() {
        let error = load_splat_from_ply(Cursor::new(ascii_ply(0, "")), None)
            .await
            .err()
            .expect("zero-row PLY should fail");

        assert!(error.to_string().contains("no vertices"));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_zero_row_compressed_ply_returns_error() {
        let error = load_splat_from_ply(Cursor::new(compressed_ply_header(0, 0, None)), None)
            .await
            .err()
            .expect("zero-row compressed PLY should fail");

        assert!(error.to_string().contains("no vertices"));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_import_basic_functionality() {
        let _device = brush_cube::test_helpers::test_device().await;
        let original_splats = create_test_splats(1);
        let ply_bytes = splat_to_ply(original_splats.clone(), None).await.unwrap();

        let cursor = Cursor::new(ply_bytes);
        let imported_message = load_splat_from_ply(cursor, None).await.unwrap();

        assert_eq!(imported_message.data.num_splats(), 1);
        assert_eq!(imported_message.meta.total_splats, 1);
        // All fields should be present for a full PLY
        assert!(imported_message.data.rotations.is_some());
        assert!(imported_message.data.log_scales.is_some());
        assert!(imported_message.data.sh_coeffs.is_some());
        assert!(imported_message.data.raw_opacities.is_some());
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_import_different_sh_degrees() {
        let _device = brush_cube::test_helpers::test_device().await;
        for degree in [0, 1, 2] {
            let original_splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(original_splats, None).await.unwrap();

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None).await.unwrap();

            let n_splats = imported_message.data.num_splats();
            let sh_coeffs = imported_message.data.sh_coeffs.unwrap();
            let n_coeffs = sh_coeffs.len() / n_splats / 3;
            assert_eq!(n_coeffs, sh_coeffs_for_degree(degree) as usize);
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_import_with_subsample() {
        let _device = brush_cube::test_helpers::test_device().await;
        // Create 4 test splats
        let original_splats = create_test_splats_with_count(0, 4);
        assert_eq!(original_splats.num_splats(), 4);

        let ply_bytes = splat_to_ply(original_splats, None).await.unwrap();

        // Test no subsampling
        let cursor = Cursor::new(ply_bytes.clone());
        let imported_message = load_splat_from_ply(cursor, None).await.unwrap();
        assert_eq!(imported_message.data.num_splats(), 4);

        // Test subsampling every 2nd splat
        let cursor = Cursor::new(ply_bytes);
        let imported_message = load_splat_from_ply(cursor, Some(2)).await.unwrap();
        assert_eq!(imported_message.data.num_splats(), 2);
    }

    #[test]
    fn test_splat_data_subsample() {
        let n = 10;
        // Per-splat value == splat index, so we can check which rows survived.
        let make = |stride: usize| -> Vec<f32> {
            (0..n)
                .flat_map(|i| std::iter::repeat_n(i as f32, stride))
                .collect()
        };
        let data = SplatData {
            means: make(3),
            rotations: Some(make(4)),
            log_scales: Some(make(3)),
            sh_coeffs: Some(make(6)),
            raw_opacities: Some(make(1)),
        };

        // Within budget: untouched.
        let same = data.clone().subsample(10);
        assert_eq!(same.num_splats(), 10);
        let same = data.clone().subsample(0);
        assert_eq!(same.num_splats(), 10);

        // step = ceil(10 / 3) = 4 -> rows 0, 4, 8 survive.
        let sub = data.subsample(3);
        assert_eq!(sub.num_splats(), 3);
        assert!(sub.num_splats() <= 3);
        assert_eq!(sub.means, vec![0., 0., 0., 4., 4., 4., 8., 8., 8.]);
        assert_eq!(
            sub.rotations.unwrap(),
            vec![0., 0., 0., 0., 4., 4., 4., 4., 8., 8., 8., 8.]
        );
        assert_eq!(
            sub.log_scales.unwrap(),
            vec![0., 0., 0., 4., 4., 4., 8., 8., 8.]
        );
        let sh = sub.sh_coeffs.unwrap();
        assert_eq!(sh.len(), 3 * 6);
        assert_eq!(&sh[0..6], &[0., 0., 0., 0., 0., 0.]);
        assert_eq!(&sh[6..12], &[4., 4., 4., 4., 4., 4.]);
        assert_eq!(sub.raw_opacities.unwrap(), vec![0., 4., 8.]);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_import_custom_up_axis() {
        let _device = brush_cube::test_helpers::test_device().await;
        let original_splats = create_test_splats(1);
        let custom_up = Vec3::new(0.123, 0.456, -0.789);
        let ply_bytes = splat_to_ply(original_splats, Some(custom_up))
            .await
            .unwrap();

        let cursor = Cursor::new(ply_bytes);
        let imported_message = load_splat_from_ply(cursor, None).await.unwrap();

        assert!(imported_message.meta.up_axis.is_some());
        let imported_up = imported_message.meta.up_axis.unwrap();
        assert!((imported_up.x - custom_up.x).abs() < 1e-5);
        assert!((imported_up.y - custom_up.y).abs() < 1e-5);
        assert!((imported_up.z - custom_up.z).abs() < 1e-5);
    }
}
