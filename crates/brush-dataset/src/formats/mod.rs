use crate::{Dataset, config::LoadDatasetConfig, scene::SceneView};
use brush_serde::{DeserializeError, SplatMessage, load_splat_from_ply};

use brush_vfs::BrushVfs;
use image::ImageError;
use itertools::{Either, Itertools};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

pub mod colmap;
pub mod nerfstudio;
pub mod realitycapture;

use thiserror::Error;

pub struct DatasetLoadResult {
    pub init_splat: Option<SplatMessage>,
    pub dataset: Dataset,
    pub warnings: Vec<String>,
}

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("I/O error while loading dataset: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error decoding JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Error decoding camera parameters: {0}")]
    InvalidCamera(String),

    #[error("Error when decoding format: {0}")]
    InvalidFormat(String),

    #[error("Error loading splat data: {0}")]
    PlyError(#[from] DeserializeError),

    #[error("Error loading image in data: {0}")]
    ImageError(#[from] ImageError),

    #[error("Ambiguous geometry prior: {0}")]
    AmbiguousPrior(String),
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error(transparent)]
    FormatError(#[from] FormatError),

    #[error("Failed to load initial point cloud: {0}")]
    InitialPointCloudError(#[from] DeserializeError),

    #[error(
        "Format not recognized: only colmap, nerfstudio json and RealityCapture csv are supported"
    )]
    FormatNotSupported,
}

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Result<DatasetLoadResult, DatasetError> {
    load_args.validate().map_err(FormatError::InvalidFormat)?;

    let mut dataset = colmap::load_dataset(vfs.clone(), load_args).await;

    if dataset.is_none() {
        dataset = nerfstudio::read_dataset(vfs.clone(), load_args).await;
    }

    if dataset.is_none() {
        dataset = realitycapture::read_dataset(vfs.clone(), load_args).await;
    }

    let Some(dataset) = dataset else {
        return Err(DatasetError::FormatNotSupported);
    };

    let result = dataset?;

    // A dataset that parsed but has no usable training views (e.g. every image
    // was missing or filtered out) would otherwise "load" and then crash on the
    // first training batch. Reject it here with a typed error instead.
    if result.dataset.train.views.is_empty() {
        return Err(FormatError::InvalidFormat(
            "dataset contains no usable training views (all images missing or filtered out)"
                .to_owned(),
        )
        .into());
    }

    // If there's an initial ply file, override the init stream with that.
    let mut ply_paths: Vec<_> = vfs.files_with_extension("ply").collect();
    ply_paths.sort();

    let main_ply = ply_paths
        .iter()
        .find(|p| p.file_name().is_some_and(|n| n == "init.ply"))
        .or_else(|| ply_paths.last());

    let init_splat = if let Some(main_ply) = main_ply {
        log::info!("Using ply {main_ply:?} as initial point cloud.");
        let reader = vfs
            .reader_at_path(main_ply)
            .await
            .map_err(DeserializeError)?;
        Some(load_splat_from_ply(reader, load_args.subsample_points).await?)
    } else {
        result.init_splat
    };

    Ok(DatasetLoadResult {
        init_splat,
        dataset: result.dataset,
        warnings: result.warnings,
    })
}

/// Directory holding per-image masks.
const MASK_DIR_NAME: &str = "masks";

/// Directory holding per-image metric depth maps, as resolved by
/// [`find_depth_path`].
const PRIOR_DIR_DEPTH: &str = "depth";

/// Directory holding per-image surface-normal maps, as resolved by
/// [`find_normal_path`].
const PRIOR_DIR_NORMAL: &str = "normal";

/// Every directory whose contents are per-image *sidecars* rather than images.
///
/// This is the single source of truth: [`find_prior_path`]'s call sites and
/// image discovery both read it, so the two can never drift into disagreeing
/// about which files are priors. Adding a prior kind means adding it here, and
/// image discovery follows automatically.
const SIDECAR_DIR_NAMES: [&str; 3] = [MASK_DIR_NAME, PRIOR_DIR_DEPTH, PRIOR_DIR_NORMAL];

fn is_sidecar_dir(component: &str) -> bool {
    SIDECAR_DIR_NAMES
        .iter()
        .any(|name| component.eq_ignore_ascii_case(name))
}

/// True if any *directory* component of these normalized path components names
/// a sidecar tree. The final component is the file name and is not considered:
/// a file called `depth.png` is an ordinary image.
fn has_sidecar_dir(components: &[String]) -> bool {
    let Some((_file_name, directories)) = components.split_last() else {
        return false;
    };
    directories.iter().any(|part| is_sidecar_dir(part))
}

/// True if `path` lives inside a sidecar tree (mask / depth / normal), i.e. it
/// is per-image data and never a candidate image. Case-insensitive, matching
/// [`find_prior_path`]'s own comparison.
fn is_sidecar_path(path: &Path) -> bool {
    has_sidecar_dir(&normalized_components(path))
}

/// The images registered under one suffix key of the file index.
///
/// `chosen` is the lexicographically first, which is what resolution returns.
/// `others` is kept so an ambiguous key can be *reported* rather than silently
/// decided -- see [`DatasetFileIndex::ambiguity_warnings`].
#[derive(Debug)]
struct ImageCandidates {
    chosen: PathBuf,
    others: Vec<PathBuf>,
}

impl ImageCandidates {
    fn new(path: &Path) -> Self {
        Self {
            chosen: path.to_path_buf(),
            others: Vec::new(),
        }
    }

    fn insert(&mut self, path: &Path) {
        if path < self.chosen.as_path() {
            let previous = std::mem::replace(&mut self.chosen, path.to_path_buf());
            self.others.push(previous);
        } else {
            self.others.push(path.to_path_buf());
        }
    }

    /// Every candidate, `chosen` included, in a stable order.
    fn all(&self) -> Vec<&Path> {
        let mut all: Vec<&Path> = std::iter::once(self.chosen.as_path())
            .chain(self.others.iter().map(PathBuf::as_path))
            .collect();
        all.sort_unstable();
        all
    }
}

/// Paths used by dataset formats, indexed once so resolving every camera does
/// not repeatedly scan the entire VFS.
struct DatasetFileIndex {
    images_by_suffix: HashMap<String, ImageCandidates>,
    masks_by_key: HashMap<(String, String), PathBuf>,
}

impl DatasetFileIndex {
    fn new(vfs: &BrushVfs) -> Self {
        let mut images_by_suffix: HashMap<String, ImageCandidates> = HashMap::new();
        let mut masks_by_key = HashMap::new();

        for path in vfs.iter_files() {
            let components = normalized_components(path);
            let masks_index = components.iter().position(|part| part == MASK_DIR_NAME);

            // Sidecar trees -- masks and the geometry priors -- hold per-image
            // data, never images, so they are kept out of the image index
            // entirely. Without this a file is indexed under EVERY suffix of
            // its path including the bare file name, and COLMAP stores bare
            // names, so `depth/frame.png` and `images/frame.png` both claim the
            // key `frame.png`. See `is_sidecar_path`.
            if !has_sidecar_dir(&components) {
                for start in 0..components.len() {
                    let key = components[start..].join("/");
                    images_by_suffix
                        .entry(key)
                        .and_modify(|candidates| candidates.insert(path))
                        .or_insert_with(|| ImageCandidates::new(path));
                }
            }

            let Some(masks_index) = masks_index else {
                continue;
            };
            let Some(stem) = path.file_stem() else {
                continue;
            };
            let subdirectory = components[masks_index + 1..components.len() - 1].join("/");
            insert_min_path(
                &mut masks_by_key,
                (subdirectory, stem.to_string_lossy().to_lowercase()),
                path,
            );
        }

        Self {
            images_by_suffix,
            masks_by_key,
        }
    }

    /// Resolve a path suffix as stored by COLMAP or `RealityCapture`. Masks and
    /// geometry priors are excluded, so an image can never resolve to its own
    /// mask, depth map or normal map.
    fn find_image_by_name(&self, name: &str) -> Option<&Path> {
        let key = normalized_components(Path::new(name)).join("/");
        self.images_by_suffix
            .get(&key)
            .map(|candidates| candidates.chosen.as_path())
    }

    /// Suffix keys that more than one image claims, as a single aggregated
    /// warning for the loader to surface.
    ///
    /// Resolution keeps the lexicographically first candidate -- the choice
    /// this index has always made, so every dataset that loads today keeps
    /// loading, including the very common nerfstudio downscale pyramid
    /// (`images/`, `images_2/`, `images_4/`, ...) where every bare name is
    /// ambiguous by construction. That is why this is a warning and not the
    /// hard error `find_prior_path` raises: there, two candidates mean one is
    /// stale supervision and neither can be trusted; here they are usually the
    /// same view at different scales, and refusing the load would reject a
    /// large class of working datasets. The genuinely undecidable case -- an
    /// image against a *prior* of the same name -- is not decided by this rule
    /// at all, because priors never enter the index.
    ///
    /// Whatever the cause, the operator is told. A silently chosen image is
    /// precisely the failure this index shipped with.
    fn ambiguity_warnings(&self) -> Vec<String> {
        let mut ambiguous: Vec<(&String, &ImageCandidates)> = self
            .images_by_suffix
            .iter()
            .filter(|(_, candidates)| !candidates.others.is_empty())
            .collect();
        if ambiguous.is_empty() {
            return Vec::new();
        }
        // Sorted so the message is stable whatever order the VFS walked in.
        ambiguous.sort_unstable_by(|a, b| a.0.cmp(b.0));

        let examples = ambiguous
            .iter()
            .take(3)
            .map(|(key, candidates)| {
                format!(
                    "'{key}' matches {}",
                    candidates
                        .all()
                        .iter()
                        .map(|p| format!("'{}'", p.display()))
                        .join(", ")
                )
            })
            .join("; ");

        vec![format!(
            "{} image name(s) match more than one file; the lexicographically first is used \
             for each. If that is not the file you meant to train on, remove or rename the \
             others. {examples}",
            ambiguous.len()
        )]
    }

    fn find_mask_path(&self, path: &Path) -> Option<&Path> {
        let search_name = path.file_name()?.to_string_lossy().to_lowercase();
        let search_stem = path.file_stem()?.to_string_lossy().to_lowercase();
        let search_stems = [
            search_name,
            search_stem.clone(),
            format!("{search_stem}.mask"),
        ];
        let parent_components = normalized_components(path.parent()?);

        // A mask subdirectory may match any suffix of the image directory.
        // Select the smallest matching path to keep resolution deterministic.
        let mut result: Option<&PathBuf> = None;
        for start in 0..=parent_components.len() {
            let subdirectory = parent_components[start..].join("/");
            for stem in &search_stems {
                if let Some(candidate) =
                    self.masks_by_key.get(&(subdirectory.clone(), stem.clone()))
                    && result.is_none_or(|current| candidate < current)
                {
                    result = Some(candidate);
                }
            }
        }
        result.map(PathBuf::as_path)
    }
}

fn normalized_components(path: &Path) -> Vec<String> {
    let mut components = Vec::new();
    for component in path.to_string_lossy().replace('\\', "/").split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            component => components.push(component.to_lowercase()),
        }
    }
    components
}

fn insert_min_path<K>(map: &mut HashMap<K, PathBuf>, key: K, path: &Path)
where
    K: std::hash::Hash + Eq,
{
    let entry = map.entry(key).or_insert_with(|| path.to_path_buf());
    if path < entry.as_path() {
        *entry = path.to_path_buf();
    }
}

/// Convert an OpenGL/Blender camera-to-world matrix (the nerfstudio
/// `transform_matrix` convention: +X right, +Y up, +Z back) into brush's
/// camera pose (+X right, +Y down, +Z forward).
fn opengl_c2w_to_pose(mut c2w: glam::Mat4) -> (glam::Vec3, glam::Quat) {
    c2w.y_axis *= -1.0;
    c2w.z_axis *= -1.0;
    let (_, rotation, translation) = c2w.to_scale_rotation_translation();
    (translation, rotation)
}

/// Split views into (train, eval) by selecting every `eval_split_every`-th view
/// for eval. With `None`, every view is a train view. With `train_on_eval`,
/// eval views are additionally kept in the training set (so per-view
/// appearance corrections exist for them).
fn split_eval_every(
    views: Vec<SceneView>,
    eval_split_every: Option<usize>,
    train_on_eval: bool,
) -> (Vec<SceneView>, Vec<SceneView>) {
    if train_on_eval {
        let eval = views
            .iter()
            .enumerate()
            .filter(|(i, _)| eval_split_every.is_some_and(|split| i % split == 0))
            .map(|(_, v)| v.clone())
            .collect();
        return (views, eval);
    }
    views.into_iter().enumerate().partition_map(|(i, v)| {
        if let Some(split) = eval_split_every
            && i % split == 0
        {
            Either::Right(v)
        } else {
            Either::Left(v)
        }
    })
}

/// Resolve a bare image name (as stored by colmap / `RealityCapture`, which only
/// record a filename) to a path in the VFS by brute-force suffix search. Sidecar
/// trees are skipped so an image never resolves to its own mask, depth map or
/// normal map -- the same exclusion, and the same reason, as
/// [`DatasetFileIndex::new`]. Used by `estimate_metric_scale`, which runs before
/// the per-image `DatasetFileIndex` is built and is opt-in/rare enough that the
/// linear scan is fine.
pub(crate) fn find_image_by_name<'a>(vfs: &'a BrushVfs, name: &str) -> Option<&'a Path> {
    vfs.files_ending_in(name)
        .filter(|path| !is_sidecar_path(path))
        .min()
}

/// Locate a per-image feature map (`<features_dir_name>/<image_stem>.npy`).
pub(crate) fn find_features_path<'a>(
    vfs: &'a BrushVfs,
    path: &'a Path,
    features_dir_name: &str,
) -> Option<&'a Path> {
    let search_stem = path.file_stem().expect("File must have a name");

    vfs.iter_files().find(|candidate| {
        let Some(stem) = candidate.file_stem() else {
            return false;
        };

        let is_npy = candidate
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("npy"));
        if !is_npy || !stem.eq_ignore_ascii_case(search_stem) {
            return false;
        }

        let features_idx = candidate
            .components()
            .position(|c| c.as_os_str().eq_ignore_ascii_case(features_dir_name));
        features_idx.is_some_and(|idx| {
            let candidate_components: Vec<_> = candidate.components().collect();
            let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
            let features_dir_subpath =
                &candidate_components[idx + 1..candidate_components.len() - 1];
            path_dir_components.ends_with(features_dir_subpath)
        })
    })
}

/// Locate the `points3d.{txt,bin}` belonging to the chosen reconstruction.
fn find_points3d_path<'a>(vfs: &'a BrushVfs, points_dir: &'a Path) -> Option<(&'a Path, bool)> {
    let path = vfs
        .files_ending_in("points3d.txt")
        .chain(vfs.files_ending_in("points3d.bin"))
        .find(|p| p.parent() == Some(points_dir))?;
    let is_binary = matches!(path.extension().and_then(|e| e.to_str()), Some("bin"));
    Some((path, is_binary))
}

/// Locate a per-image prior map stored under a `<prior_dir>/` directory whose
/// tail matches the image's own directory tail, with a stem matching either the
/// image's full file name or its stem (so both `depth/img.png.tiff` and
/// `depth/img.tiff` resolve for `images/img.png`).
///
/// Matching is by **stem only** -- the candidate's extension is never
/// inspected, so `depth/img.tif` is found just as `depth/img.tiff` is, and a
/// quantized `depth/img.png` is found just as a float32 `depth/img.tiff` is.
/// The prior loaders dispatch on magic bytes, so one dataset may legitimately
/// mix both wire formats frame by frame.
///
/// What is **not** legitimate is two candidates for the same image. That means
/// two files claim to be the same prior -- overwhelmingly a half-finished
/// format migration that left the stale float32 file next to the new quantized
/// one. Picking one would mean picking by VFS iteration order, i.e.
/// nondeterministically training on stale supervision, so this returns an error
/// naming every candidate instead (plan D4).
fn find_prior_path<'a>(
    vfs: &'a BrushVfs,
    path: &'a Path,
    prior_dir: &str,
) -> Result<Option<&'a Path>, FormatError> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");

    let mut matches: Vec<&Path> = vfs
        .iter_files()
        .filter(|candidate| {
            let Some(stem) = candidate.file_stem() else {
                return false;
            };
            if !(stem.eq_ignore_ascii_case(search_name) || stem.eq_ignore_ascii_case(search_stem)) {
                return false;
            }
            let dir_idx = candidate
                .components()
                .position(|c| c.as_os_str().eq_ignore_ascii_case(prior_dir));
            dir_idx.is_some_and(|idx| {
                let candidate_components: Vec<_> = candidate.components().collect();
                let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
                let prior_dir_subpath =
                    &candidate_components[idx + 1..candidate_components.len() - 1];
                path_dir_components.ends_with(prior_dir_subpath)
            })
        })
        .collect();

    match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(matches[0])),
        _ => {
            // Sorted so the message is stable whatever order the VFS walked in.
            matches.sort_unstable();
            Err(FormatError::AmbiguousPrior(format!(
                "{} files claim to be the '{prior_dir}' prior of '{}': {}. \
                 Exactly one is required. This usually means a prior-format \
                 migration left a stale file behind -- delete all but the \
                 current one rather than letting the loader guess.",
                matches.len(),
                path.display(),
                matches
                    .iter()
                    .map(|p| format!("'{}'", p.display()))
                    .join(", ")
            )))
        }
    }
}

/// Locate a per-image depth map (`depth/<image stem>.{tiff,png}`).
fn find_depth_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Result<Option<&'a Path>, FormatError> {
    find_prior_path(vfs, path, PRIOR_DIR_DEPTH)
}

/// Locate a per-image surface-normal map (`normal/<image stem>.{tiff,png}`).
/// Same matching rules as [`find_depth_path`], different directory component.
fn find_normal_path<'a>(
    vfs: &'a BrushVfs,
    path: &'a Path,
) -> Result<Option<&'a Path>, FormatError> {
    find_prior_path(vfs, path, PRIOR_DIR_NORMAL)
}

/// Shared fixtures for the per-format prior-discovery tests (nerfstudio,
/// `RealityCapture`). Native-only: they write real files, because prior
/// discovery walks the VFS and can't be exercised on synthesised paths alone.
#[cfg(all(test, not(target_family = "wasm")))]
pub(crate) mod prior_test_support {
    use crate::config::LoadDatasetConfig;
    use std::io::Cursor;
    use std::path::Path;
    use tiff::encoder::{TiffEncoder, colortype};

    /// Single-channel float32 TIFF, the depth-prior wire format.
    pub(crate) fn encode_gray_f32(values: &[f32], w: u32, h: u32) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf).expect("tiff encoder");
            encoder
                .write_image::<colortype::Gray32Float>(w, h, values)
                .expect("write gray f32 tiff");
        }
        buf.into_inner()
    }

    /// 3-channel float32 TIFF, the normal-prior wire format.
    pub(crate) fn encode_rgb_f32(values: &[f32], w: u32, h: u32) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buf).expect("tiff encoder");
            encoder
                .write_image::<colortype::RGB32Float>(w, h, values)
                .expect("write rgb f32 tiff");
        }
        buf.into_inner()
    }

    pub(crate) async fn write_depth_tiff(path: &Path, w: u32, h: u32) {
        let values = vec![1.5f32; (w * h) as usize];
        tokio::fs::create_dir_all(path.parent().expect("has parent"))
            .await
            .expect("create prior dir");
        tokio::fs::write(path, encode_gray_f32(&values, w, h))
            .await
            .expect("write depth tiff");
    }

    pub(crate) async fn write_normal_tiff(path: &Path, w: u32, h: u32) {
        // Camera-frame OpenCV unit normals facing the camera (n.z <= 0).
        let values: Vec<f32> = (0..(w * h)).flat_map(|_| [0.0f32, 0.0, -1.0]).collect();
        tokio::fs::create_dir_all(path.parent().expect("has parent"))
            .await
            .expect("create prior dir");
        tokio::fs::write(path, encode_rgb_f32(&values, w, h))
            .await
            .expect("write normal tiff");
    }

    /// Single-channel uint16 PNG, the quantized depth wire format. Writes the
    /// committed golden fixture verbatim so the on-disk bytes the loader sees
    /// are exactly the ones `test_prior_io.py` reads (plan T18).
    pub(crate) async fn write_depth_png(path: &Path) {
        write_prior(path, crate::testdata::GOLDEN_DEPTH_U16_PNG).await;
    }

    /// 3-channel uint8 PNG, the quantized normal wire format. Same fixture
    /// discipline as [`write_depth_png`].
    pub(crate) async fn write_normal_png(path: &Path) {
        write_prior(path, crate::testdata::GOLDEN_NORMAL_U8_PNG).await;
    }

    async fn write_prior(path: &Path, bytes: &[u8]) {
        tokio::fs::create_dir_all(path.parent().expect("has parent"))
            .await
            .expect("create prior dir");
        tokio::fs::write(path, bytes).await.expect("write prior");
    }

    pub(crate) fn test_config() -> LoadDatasetConfig {
        LoadDatasetConfig {
            max_frames: None,
            max_resolution: 1920,
            eval_split_every: None,
            subsample_frames: None,
            subsample_points: None,
            alpha_mode: None,
            invert_masks: false,
            max_scene_batch_cache_size: 0,
            train_on_eval: false,
            estimate_metric_scale: false,
            features_dir_name: "dino_features".to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_mask() {
        // Basic matching with same extension
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.png"),
            PathBuf::from("masks/img.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_mask_path(Path::new("images/img.png")),
            Some(Path::new("masks/img.png"))
        );
        // Different extensions are ok.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.jpeg"),
            PathBuf::from("masks/img.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_mask_path(Path::new("images/img.jpeg")),
            Some(Path::new("masks/img.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_mask_formats() {
        // Test img.png.mask format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo.png"),
            PathBuf::from("masks/foo.png.mask"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_mask_path(Path::new("images/foo.png")),
            Some(Path::new("masks/foo.png.mask"))
        );

        // Test img.mask.png format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/bar.jpeg"),
            PathBuf::from("masks/bar.mask.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_mask_path(Path::new("images/bar.jpeg")),
            Some(Path::new("masks/bar.mask.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_nested_dirs() {
        // Nested directories must match
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo/bar/img.png"),
            PathBuf::from("masks/foo/bar/img.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_mask_path(Path::new("images/foo/bar/img.png")),
            Some(Path::new("masks/foo/bar/img.png"))
        );
        // Should not match wrong subpath
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/baz/img.png"),
            PathBuf::from("masks/foo/img.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(index.find_mask_path(Path::new("images/baz/img.png")), None);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_case_insensitive() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/IMG.PNG"),
            PathBuf::from("masks/img.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_mask_path(Path::new("images/IMG.PNG")),
            Some(Path::new("masks/img.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_indexed_image_suffix_lookup() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/nested/frame.png"),
            PathBuf::from("masks/nested/frame.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);

        assert_eq!(
            index.find_image_by_name("nested/frame.png"),
            Some(Path::new("images/nested/frame.png"))
        );
        assert_eq!(
            index.find_image_by_name("FRAME.PNG"),
            Some(Path::new("images/nested/frame.png"))
        );
    }

    /// A geometry prior sharing the image's file name must never be returned
    /// as the image. Reproduces the silent depth-for-image swap: COLMAP stores
    /// bare names, `depth` sorts before `images`, so the bare-name shortcut
    /// used to hand back the depth map.
    #[wasm_bindgen_test(unsupported = test)]
    fn test_prior_never_outranks_image_on_bare_name() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/frame_0001_back.png"),
            PathBuf::from("depth/frame_0001_back.png"),
            PathBuf::from("normal/frame_0001_back.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_image_by_name("frame_0001_back.png"),
            Some(Path::new("images/frame_0001_back.png"))
        );
    }

    /// The pre-index scan `estimate_metric_scale` uses must make the same
    /// exclusion, or the metric scale is fitted against a depth map that was
    /// read as a photograph.
    #[wasm_bindgen_test(unsupported = test)]
    fn test_free_find_image_by_name_skips_sidecars() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/frame.png"),
            PathBuf::from("depth/frame.png"),
            PathBuf::from("normal/frame.png"),
            PathBuf::from("masks/frame.png"),
        ]);
        assert_eq!(
            find_image_by_name(&vfs, "frame.png"),
            Some(Path::new("images/frame.png"))
        );
    }

    /// An image against a prior of the same name is not an ambiguity to be
    /// reported and tie-broken -- the loader knows which is which, so the
    /// prior never enters the index. If this ever starts warning, the
    /// exclusion has been replaced by a priority rule.
    #[wasm_bindgen_test(unsupported = test)]
    fn test_prior_collision_is_not_an_ambiguity() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/frame.png"),
            PathBuf::from("depth/frame.png"),
            PathBuf::from("normal/frame.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(index.ambiguity_warnings(), Vec::<String>::new());
    }

    /// Two real images of one bare name (the nerfstudio downscale pyramid) is
    /// still resolved -- deterministically, full resolution first -- but the
    /// operator is told rather than left to guess.
    #[wasm_bindgen_test(unsupported = test)]
    fn test_two_images_of_one_name_resolve_and_warn() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/frame.png"),
            PathBuf::from("images_2/frame.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_image_by_name("frame.png"),
            Some(Path::new("images/frame.png"))
        );

        let warnings = index.ambiguity_warnings();
        assert_eq!(warnings.len(), 1, "one aggregated message, not one per key");
        assert!(
            warnings[0].contains("images/frame.png") && warnings[0].contains("images_2/frame.png"),
            "the message must name both candidates: {}",
            warnings[0]
        );

        // A directory-prefixed name is unambiguous and must not be dragged in.
        assert_eq!(
            index.find_image_by_name("images_2/frame.png"),
            Some(Path::new("images_2/frame.png"))
        );
    }

    /// Excluding sidecars must not cost the legitimate lookups: a
    /// directory-prefixed COLMAP name, the bare-name shortcut, and both
    /// sidecars of that same image all still resolve.
    #[wasm_bindgen_test(unsupported = test)]
    fn test_nested_lookups_survive_the_sidecar_exclusion() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("data/images/nested/frame.png"),
            PathBuf::from("data/depth/nested/frame.png"),
            PathBuf::from("data/masks/nested/frame.png"),
        ]);
        let index = DatasetFileIndex::new(&vfs);

        assert_eq!(
            index.find_image_by_name("images/nested/frame.png"),
            Some(Path::new("data/images/nested/frame.png"))
        );
        assert_eq!(
            index.find_image_by_name("frame.png"),
            Some(Path::new("data/images/nested/frame.png"))
        );
        assert_eq!(
            index.find_mask_path(Path::new("data/images/nested/frame.png")),
            Some(Path::new("data/masks/nested/frame.png"))
        );
        assert_eq!(
            find_depth_path(&vfs, Path::new("data/images/nested/frame.png")).expect("unambiguous"),
            Some(Path::new("data/depth/nested/frame.png"))
        );
    }

    /// The exclusion keys on directory components only. A photograph that
    /// happens to be called `depth.png` is still a photograph.
    #[wasm_bindgen_test(unsupported = test)]
    fn test_sidecar_exclusion_ignores_the_file_name() {
        let vfs = BrushVfs::create_test_vfs(vec![PathBuf::from("images/depth.png")]);
        let index = DatasetFileIndex::new(&vfs);
        assert_eq!(
            index.find_image_by_name("depth.png"),
            Some(Path::new("images/depth.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_normal_matches_stem_and_full_name() {
        // `normal/<stem>.tiff` next to `images/<stem>.png`.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.png"),
            PathBuf::from("normal/img.tiff"),
        ]);
        assert_eq!(
            find_normal_path(&vfs, Path::new("images/img.png")).expect("unambiguous"),
            Some(Path::new("normal/img.tiff"))
        );

        // `normal/<full name>.tiff` is accepted too, same as depth.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.jpg"),
            PathBuf::from("normal/img.jpg.tiff"),
        ]);
        assert_eq!(
            find_normal_path(&vfs, Path::new("images/img.jpg")).expect("unambiguous"),
            Some(Path::new("normal/img.jpg.tiff"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_normal_requires_matching_subdirs() {
        // Nested dir tails must line up.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo/bar/img.png"),
            PathBuf::from("normal/foo/bar/img.tiff"),
        ]);
        assert_eq!(
            find_normal_path(&vfs, Path::new("images/foo/bar/img.png")).expect("unambiguous"),
            Some(Path::new("normal/foo/bar/img.tiff"))
        );

        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/baz/img.png"),
            PathBuf::from("normal/foo/img.tiff"),
        ]);
        assert_eq!(
            find_normal_path(&vfs, Path::new("images/baz/img.png")).expect("unambiguous"),
            None
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_normal_and_depth_dirs_do_not_cross_match() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.png"),
            PathBuf::from("depth/img.tiff"),
        ]);
        // A depth-only dataset must not resolve a normal prior, and vice versa.
        assert_eq!(
            find_normal_path(&vfs, Path::new("images/img.png")).expect("unambiguous"),
            None
        );
        assert_eq!(
            find_depth_path(&vfs, Path::new("images/img.png")).expect("unambiguous"),
            Some(Path::new("depth/img.tiff"))
        );

        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.png"),
            PathBuf::from("normal/img.tiff"),
        ]);
        assert_eq!(
            find_depth_path(&vfs, Path::new("images/img.png")).expect("unambiguous"),
            None
        );
    }
}

/// Cross-format prior discovery: a dataset may mix float32-TIFF and quantized
/// PNG priors frame by frame (plan D4 "works"), but two files claiming the same
/// prior is fatal (D4 "fails loudly").
///
/// These need a real on-disk dataset, so they are native-only, in the style of
/// the per-format prior tests.
#[cfg(all(test, not(target_family = "wasm")))]
mod prior_format_tests {
    use super::prior_test_support::{
        test_config, write_depth_png, write_depth_tiff, write_normal_png, write_normal_tiff,
    };
    use super::{DatasetError, DatasetLoadResult, FormatError, load_dataset};
    use crate::testdata;
    use brush_vfs::BrushVfs;
    use std::path::Path;
    use std::sync::Arc;

    const IMG_W: u32 = testdata::GOLDEN_W as u32;
    const IMG_H: u32 = testdata::GOLDEN_H as u32;

    /// A nerfstudio dataset with `n` identity-posed frames, sized to match the
    /// golden prior fixtures (no resize is ever applied to priors -- plan D8 --
    /// so the sizes must agree exactly).
    async fn write_dataset(dir: &Path, frames: &[&str]) {
        let frame_json: Vec<String> = frames
            .iter()
            .map(|name| {
                format!(
                    r#"{{
                        "file_path": "images/{name}.png",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]
                        ]
                    }}"#
                )
            })
            .collect();
        let transforms = format!(
            r#"{{
                "fl_x": 4.0, "fl_y": 3.0, "cx": 2.0, "cy": 1.5,
                "w": {IMG_W}, "h": {IMG_H},
                "frames": [{}]
            }}"#,
            frame_json.join(",")
        );
        tokio::fs::write(dir.join("transforms.json"), transforms)
            .await
            .expect("write transforms.json");

        let images_dir = dir.join("images");
        tokio::fs::create_dir_all(&images_dir)
            .await
            .expect("create images dir");
        for name in frames {
            image::RgbImage::from_pixel(IMG_W, IMG_H, image::Rgb([10, 20, 30]))
                .save(images_dir.join(format!("{name}.png")))
                .expect("write image");
        }
    }

    async fn try_load(dir: &Path) -> Result<DatasetLoadResult, DatasetError> {
        let vfs = Arc::new(BrushVfs::from_path(dir).await.expect("build vfs"));
        load_dataset(vfs, &test_config()).await
    }

    // ---- T8: one dataset, both wire formats --------------------------------

    #[tokio::test]
    async fn mixed_format_dataset_loads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        write_dataset(root, &["frame_001", "frame_002"]).await;

        // Frame 1 keeps the legacy float32 TIFFs; frame 2 is migrated.
        write_depth_tiff(&root.join("depth/frame_001.tiff"), IMG_W, IMG_H).await;
        write_normal_tiff(&root.join("normal/frame_001.tiff"), IMG_W, IMG_H).await;
        write_depth_png(&root.join("depth/frame_002.png")).await;
        write_normal_png(&root.join("normal/frame_002.png")).await;

        let Ok(result) = try_load(root).await else {
            panic!("mixed-format dataset must load");
        };
        let mut views: Vec<_> = result.dataset.train.views.as_ref().clone();
        views.sort_by(|a, b| a.image.path().cmp(b.image.path()));
        assert_eq!(views.len(), 2, "both frames must survive the load");

        let (h, w) = (IMG_H as usize, IMG_W as usize);

        // Frame 1: float32 TIFF, unchanged behaviour.
        let d1 = views[0]
            .depth
            .as_ref()
            .expect("frame 1 depth")
            .load_vec(h, w)
            .await
            .expect("tiff depth decodes");
        assert!(
            d1.iter().all(|&v| v == 1.5),
            "tiff fixture is a constant 1.5 m, got {d1:?}"
        );
        let n1 = views[0]
            .normal
            .as_ref()
            .expect("frame 1 normal")
            .load_vec(h, w)
            .await
            .expect("tiff normal decodes");
        assert_eq!(n1.len(), h * w * 3);

        // Frame 2: quantized PNG, decoded through the Part B codec -- and it
        // must land on the SAME pinned table the codec unit tests use, i.e.
        // going through discovery + VFS changes nothing.
        let d2 = views[1]
            .depth
            .as_ref()
            .expect("frame 2 depth")
            .load_vec(h, w)
            .await
            .expect("png depth decodes");
        for (got, &want) in d2.iter().zip(testdata::GOLDEN_DEPTH_METRES_BITS.iter()) {
            assert_eq!(got.to_bits(), want);
        }
        let n2 = views[1]
            .normal
            .as_ref()
            .expect("frame 2 normal")
            .load_vec(h, w)
            .await
            .expect("png normal decodes");
        for (got, &want) in n2.iter().zip(testdata::GOLDEN_NORMAL_UNIT_BITS.iter()) {
            assert_eq!(got.to_bits(), want);
        }

        // Null model: the two frames genuinely carry different data, so
        // "both decoded" is not one buffer counted twice.
        assert_ne!(d1, d2);
    }

    // ---- T9: two files claiming one prior is fatal -------------------------

    #[tokio::test]
    async fn ambiguous_prior_stem_is_fatal() {
        for (dir_name, ambiguous_ext) in [("depth", "png"), ("normal", "png")] {
            let dir = tempfile::tempdir().expect("tempdir");
            let root = dir.path();
            write_dataset(root, &["frame_001"]).await;

            // Baseline: the TIFF alone loads. This is the null model -- it
            // proves the failure below is caused by the SECOND file, not by
            // the dataset being broken some other way.
            write_depth_tiff(&root.join("depth/frame_001.tiff"), IMG_W, IMG_H).await;
            write_normal_tiff(&root.join("normal/frame_001.tiff"), IMG_W, IMG_H).await;
            assert!(
                try_load(root).await.is_ok(),
                "single-prior dataset must load"
            );

            // Now leave a stale sibling behind, exactly as a crashed migration
            // would.
            let stale = root.join(format!("{dir_name}/frame_001.{ambiguous_ext}"));
            if dir_name == "depth" {
                write_depth_png(&stale).await;
            } else {
                write_normal_png(&stale).await;
            }

            let Err(err) = try_load(root).await else {
                panic!("two files for one prior must be fatal");
            };
            let DatasetError::FormatError(FormatError::AmbiguousPrior(msg)) = err else {
                panic!("expected AmbiguousPrior, got {err:?}");
            };
            // The message must name EVERY candidate -- the operator has to know
            // which file to delete.
            assert!(
                msg.contains(&format!("{dir_name}/frame_001.tiff")),
                "message must name the tiff: {msg}"
            );
            assert!(
                msg.contains(&format!("{dir_name}/frame_001.{ambiguous_ext}")),
                "message must name the png: {msg}"
            );
            assert!(
                msg.contains("images/frame_001.png"),
                "message must name the image it belongs to: {msg}"
            );
        }
    }

    /// Two *stale* files of the same wire format are just as fatal -- the rule
    /// is one prior per image, not "one per extension".
    #[tokio::test]
    async fn ambiguous_prior_is_not_about_extensions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        write_dataset(root, &["frame_001"]).await;
        write_depth_tiff(&root.join("depth/frame_001.tiff"), IMG_W, IMG_H).await;
        write_depth_tiff(&root.join("depth/frame_001.tif"), IMG_W, IMG_H).await;

        let Err(err) = try_load(root).await else {
            panic!("two tiffs for one prior must be fatal");
        };
        assert!(
            matches!(
                err,
                DatasetError::FormatError(FormatError::AmbiguousPrior(_))
            ),
            "got {err:?}"
        );
    }
}
