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

/// Paths used by dataset formats, indexed once so resolving every camera does
/// not repeatedly scan the entire VFS.
struct DatasetFileIndex {
    images_by_suffix: HashMap<String, PathBuf>,
    masks_by_key: HashMap<(String, String), PathBuf>,
}

impl DatasetFileIndex {
    fn new(vfs: &BrushVfs) -> Self {
        let mut images_by_suffix = HashMap::new();
        let mut masks_by_key = HashMap::new();

        for path in vfs.iter_files() {
            let components = normalized_components(path);
            let masks_index = components.iter().position(|part| part == "masks");

            if masks_index.is_none() {
                for start in 0..components.len() {
                    insert_min_path(&mut images_by_suffix, components[start..].join("/"), path);
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

    /// Resolve a path suffix as stored by COLMAP or `RealityCapture`. Masks are
    /// excluded so an image cannot resolve to its own mask.
    fn find_image_by_name(&self, name: &str) -> Option<&Path> {
        let key = normalized_components(Path::new(name)).join("/");
        self.images_by_suffix.get(&key).map(PathBuf::as_path)
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
}
