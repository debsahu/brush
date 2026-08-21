use super::{
    DatasetFileIndex, DatasetLoadResult, FormatError, find_depth_path, find_normal_path,
    opengl_c2w_to_pose,
};
use crate::{
    Dataset,
    config::LoadDatasetConfig,
    scene::{LoadDepth, LoadImage, LoadNormal, SceneView},
};
use brush_render::camera::fov_to_focal;
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::CameraModel::{
    KannalaBrandt4, Pinhole, RadialTangential8,
};
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use brush_render::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use brush_serde::load_splat_from_ply;
use brush_vfs::BrushVfs;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;

#[derive(serde::Deserialize, Clone)]
struct JsonScene {
    // Horizontal FOV.
    camera_angle_x: Option<f64>,
    // Vertical FOV.
    camera_angle_y: Option<f64>,

    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,

    /// Nerfstudio camera model: `"OPENCV"`, `"OPENCV_FISHEYE"`, or unset (pinhole).
    camera_model: Option<String>,
    // Nerfstudio doesn't mention this in their format? But fine to include really.
    ply_file_path: Option<String>,

    /// Principal point x
    cx: Option<f64>,
    /// Principal point y
    cy: Option<f64>,
    /// Image width
    w: Option<f64>,
    /// Image height
    h: Option<f64>,

    /// First radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k1: Option<f64>,
    /// Second radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k2: Option<f64>,
    /// Third radial distortion parameter used by `OPENCV_FISHEYE`
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by `OPENCV_FISHEYE`
    k4: Option<f64>,
    /// First tangential distortion parameter used by `OPENCV`
    p1: Option<f64>,
    /// Second tangential distortion parameter used by `OPENCV`
    p2: Option<f64>,

    frames: Vec<FrameData>,
}

#[derive(serde::Deserialize, Clone)]
struct FrameData {
    /// Nerfstudio camera model override for this frame.
    camera_model: Option<String>,

    // Horizontal FOV.
    camera_angle_x: Option<f64>,
    // Vertical FOV.
    camera_angle_y: Option<f64>,

    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,

    /// Principal point x
    cx: Option<f64>,
    /// Principal point y
    cy: Option<f64>,
    /// Image width. Should be an integer but read as float, fine to truncate.
    w: Option<f64>,
    /// Image height. Should be an integer but read as float, fine to truncate.
    h: Option<f64>,

    /// First radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k1: Option<f64>,
    /// Second radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k2: Option<f64>,
    /// Third radial distortion parameter used by `OPENCV_FISHEYE`
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by `OPENCV_FISHEYE`
    k4: Option<f64>,
    /// First tangential distortion parameter used by `OPENCV`
    p1: Option<f64>,
    /// Second tangential distortion parameter used by `OPENCV`
    p2: Option<f64>,

    transform_matrix: Vec<Vec<f32>>,
    file_path: String,

    /// Optional per-frame depth prior, nerfstudio's own key. Path is relative
    /// to the transforms file, exactly like `file_path`. Contents must be a
    /// single-channel float32 TIFF in metres with `0` = invalid — see
    /// [`crate::load_depth::LoadDepth`]; the loader does **not** resize it, so
    /// it has to match the resolution the image is loaded at.
    depth_file_path: Option<String>,

    /// Optional per-frame surface-normal prior. Not part of the nerfstudio
    /// spec, but the natural sibling of `depth_file_path` and the same key the
    /// COLMAP-side `normal/` convention resolves to. 3-channel float32 TIFF,
    /// camera-frame `OpenCV` unit normals, `(0, 0, 0)` = invalid — see
    /// [`crate::load_normal::LoadNormal`]. Also never resized.
    normal_file_path: Option<String>,
}

/// Build a `CameraModel` from a nerfstudio `camera_model` string and the
/// k/p distortion coefficients available at this scope.
fn resolve_camera_model(
    model_name: Option<&str>,
    k1: Option<f64>,
    k2: Option<f64>,
    k3: Option<f64>,
    k4: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
) -> Result<CameraModel, FormatError> {
    let f = |o: Option<f64>| o.unwrap_or(0.0) as f32;
    match model_name {
        // `PINHOLE` is COLMAP's name for the same zero-distortion model, and it
        // turns up in transforms.json routinely (SplatCam writes it; the LFS
        // recipes push people toward PINHOLE everywhere). Genuinely unsupported
        // model names still hard-error below -- a silently substituted camera
        // model is worse than a failed load.
        None | Some("PERSPECTIVE" | "perspective" | "PINHOLE" | "pinhole") => Ok(Pinhole),
        Some("OPENCV" | "opencv") => Ok(RadialTangential8(RadialTangential8Params {
            k1: f(k1),
            k2: f(k2),
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: f(p1),
            p2: f(p2),
        })),
        Some("OPENCV_FISHEYE" | "opencv_fisheye") => Ok(KannalaBrandt4(KannalaBrandt4Params {
            k1: f(k1),
            k2: f(k2),
            k3: f(k3),
            k4: f(k4),
        })),
        Some(other) => Err(FormatError::InvalidCamera(format!(
            "Unsupported nerfstudio camera_model `{other}`"
        ))),
    }
}

async fn read_transforms_file(
    scene: JsonScene,
    transforms_path: &Path,
    vfs: Arc<BrushVfs>,
    file_index: &DatasetFileIndex,
    load_args: &LoadDatasetConfig,
    warnings: &mut Vec<String>,
) -> Result<Vec<SceneView>, FormatError> {
    let mut results = vec![];
    for frame in scene
        .frames
        .iter()
        .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
        .take(load_args.max_frames.unwrap_or(usize::MAX))
    {
        brush_async::yield_now().await;

        // NeRF 'transform_matrix' is a camera-to-world transform
        let transform_matrix: Vec<f32> = frame.transform_matrix.iter().flatten().copied().collect();
        if transform_matrix.len() != 16 {
            return Err(FormatError::InvalidFormat(format!(
                "frame '{}' has a {}-element transform_matrix, expected a 4x4 (16 elements)",
                frame.file_path,
                transform_matrix.len()
            )));
        }
        let transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();
        let (translation, rotation) = opengl_c2w_to_pose(transform);

        let Some(path) = resolve_frame_image_path(&vfs, transforms_path, &frame.file_path).await
        else {
            warnings.push(format!(
                "Skipped '{}': image file not found",
                frame.file_path
            ));
            continue;
        };

        let mask_path = file_index.find_mask_path(&path).map(Path::to_path_buf);

        // Geometry priors. An explicit `depth_file_path` / `normal_file_path`
        // in the json wins; otherwise fall back to the same `depth/<stem>` /
        // `normal/<stem>` directory convention the COLMAP loader uses
        // (`super::find_prior_path`), so one on-disk layout feeds either
        // loader. Before this, both were hardcoded `None` here and the depth
        // and normal supervision silently no-opped on every nerfstudio
        // dataset — which is why `ingest/splatcam/splatcam_to_brush.py`
        // transcodes to COLMAP format instead.
        let depth = match frame.depth_file_path.as_deref() {
            Some(declared) => {
                resolve_declared_prior_path(
                    &vfs,
                    transforms_path,
                    declared,
                    &frame.file_path,
                    "depth",
                    warnings,
                )
                .await
            }
            None => find_depth_path(&vfs, &path)?.map(Path::to_path_buf),
        }
        .map(|p| LoadDepth::new(vfs.clone(), p));

        let normal = match frame.normal_file_path.as_deref() {
            Some(declared) => {
                resolve_declared_prior_path(
                    &vfs,
                    transforms_path,
                    declared,
                    &frame.file_path,
                    "normal",
                    warnings,
                )
                .await
            }
            None => find_normal_path(&vfs, &path)?.map(Path::to_path_buf),
        }
        .map(|p| LoadNormal::new(vfs.clone(), p));

        let image = LoadImage::new(
            vfs.clone(),
            path,
            mask_path,
            load_args.max_resolution,
            load_args.alpha_mode,
            load_args.invert_masks,
        );

        let w = frame.w.or(scene.w);
        let h = frame.h.or(scene.h);
        // If the json omits the size, read it from the image header (cheap, no
        // full decode).
        let (w, h) = match (w, h) {
            (Some(w), Some(h)) => (w as u32, h as u32),
            _ => image.dimensions().await?,
        };

        let camera_model = resolve_camera_model(
            frame
                .camera_model
                .as_deref()
                .or(scene.camera_model.as_deref()),
            frame.k1.or(scene.k1),
            frame.k2.or(scene.k2),
            frame.k3.or(scene.k3),
            frame.k4.or(scene.k4),
            frame.p1.or(scene.p1),
            frame.p2.or(scene.p2),
        )?;

        let fovx = frame
            .camera_angle_x
            .or(frame.fl_x.map(|fx| focal_to_fov(fx, w, &camera_model)))
            .or(scene.camera_angle_x)
            .or(scene.fl_x.map(|fx| focal_to_fov(fx, w, &camera_model)));

        let fovy = frame
            .camera_angle_y
            .or(frame.fl_y.map(|fy| focal_to_fov(fy, h, &camera_model)))
            .or(scene.camera_angle_y)
            .or(scene.fl_y.map(|fy| focal_to_fov(fy, h, &camera_model)));

        let (fovx, fovy) = match (fovx, fovy) {
            (None, None) => Err(FormatError::InvalidCamera(
                "Must have some kind of focal length".to_owned(),
            ))?,
            (None, Some(fovy)) => {
                let fovx = focal_to_fov(fov_to_focal(fovy, h, &camera_model), w, &camera_model);
                (fovx, fovy)
            }
            (Some(fovx), None) => {
                let fovy = focal_to_fov(fov_to_focal(fovx, w, &camera_model), h, &camera_model);
                (fovx, fovy)
            }
            (Some(fovx), Some(fovy)) => (fovx, fovy),
        };

        let cx = frame.cx.or(scene.cx);
        let cy = frame.cy.or(scene.cy);

        let cuv = glam::vec2(
            cx.map_or(0.5, |v| v / w as f64) as f32,
            cy.map_or(0.5, |v| v / h as f64) as f32,
        );

        let camera = Camera::new(translation, rotation, fovx, fovy, cuv, camera_model);

        if !camera.is_valid() {
            let msg = format!(
                "Skipped '{}': camera contains nan or inf values",
                frame.file_path
            );
            warnings.push(msg);
            continue;
        }

        let view = SceneView {
            image,
            camera,
            features: None,
            depth,
            normal,
        };
        results.push(view);
    }
    Ok(results)
}

/// Resolve a prior path explicitly declared by a frame (`depth_file_path` /
/// `normal_file_path`). Interpreted relative to the transforms file, exactly
/// like `file_path`, and no extension guessing — priors are written with a
/// real extension by every generator we have.
///
/// A key that names a file the VFS does not hold produces a **warning** rather
/// than a silent `None`: an ignored prior is invisible in the loss curves, so
/// it must be loud at load time.
async fn resolve_declared_prior_path(
    vfs: &BrushVfs,
    transforms_path: &Path,
    declared: &str,
    frame_path: &str,
    kind: &str,
    warnings: &mut Vec<String>,
) -> Option<std::path::PathBuf> {
    let path = transforms_path
        .parent()
        .expect("Transforms path must be a filename")
        .join(declared);

    if vfs.reader_at_path(&path).await.is_ok() {
        Some(path)
    } else {
        warnings.push(format!(
            "Frame '{frame_path}': {kind} prior '{declared}' not found, ignoring it"
        ));
        None
    }
}

async fn resolve_frame_image_path(
    vfs: &BrushVfs,
    transforms_path: &Path,
    frame_path: &str,
) -> Option<std::path::PathBuf> {
    let mut path = transforms_path
        .parent()
        .expect("Transforms path must be a filename")
        .join(frame_path);

    // Nerfstudio commonly omits the extension and stores a PNG on disk.
    // Resolve that convention before checking whether the image exists.
    if path.extension().is_none() {
        path.set_extension("png");
    }

    vfs.reader_at_path(&path).await.ok().map(|_| path)
}

pub async fn read_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Option<Result<DatasetLoadResult, FormatError>> {
    log::info!("Loading nerfstudio dataset");

    let json_files: Vec<_> = vfs.files_with_extension("json").collect();

    let transforms_path = if json_files.len() == 1 {
        json_files.first()?
    } else {
        // If there's multiple options, only pick files which are either exactly
        // transforms.json or end with transforms_train.json (a la transforms_train.json)
        vfs.files_ending_in("transforms.json")
            .next()
            .or_else(|| vfs.files_ending_in("transforms_train.json").next())?
    };
    let transforms_path = transforms_path.to_path_buf();
    Some(read_dataset_inner(vfs, load_args, json_files, transforms_path).await)
}

async fn read_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
    json_files: Vec<std::path::PathBuf>,
    transforms_path: std::path::PathBuf,
) -> Result<DatasetLoadResult, FormatError> {
    let mut warnings = Vec::new();

    let mut buf = String::new();
    vfs.reader_at_path(&transforms_path)
        .await?
        .read_to_string(&mut buf)
        .await?;
    let train_scene: JsonScene = serde_json::from_str(&buf)?;
    let file_index = DatasetFileIndex::new(&vfs);
    let train_handles = read_transforms_file(
        train_scene.clone(),
        &transforms_path,
        vfs.clone(),
        &file_index,
        load_args,
        &mut warnings,
    )
    .await?;

    // Use transforms_val as eval, or _test if no _val is present. (Brush doesn't really have any notion of a test set).
    let eval_trans_path = json_files
        .iter()
        .find(|x| x.ends_with("transforms_val.json"))
        .or_else(|| {
            json_files
                .iter()
                .find(|x| x.ends_with("transforms_test.json"))
        });
    // If a separate eval file is specified, read it.
    let val_views = if let Some(eval_trans_path) = eval_trans_path {
        let mut json_str = String::new();
        vfs.reader_at_path(eval_trans_path)
            .await?
            .read_to_string(&mut json_str)
            .await?;
        let val_scene = serde_json::from_str(&json_str)?;
        Some(
            read_transforms_file(
                val_scene,
                eval_trans_path,
                vfs.clone(),
                &file_index,
                load_args,
                &mut warnings,
            )
            .await?,
        )
    } else {
        None
    };

    let mut train_views = vec![];
    let mut eval_views = vec![];
    for (i, view) in train_handles.into_iter().enumerate() {
        if let Some(eval_period) = load_args.eval_split_every {
            // Include extra eval images only when the dataset doesn't have them.
            if i % eval_period == 0 && val_views.is_none() {
                if load_args.train_on_eval {
                    train_views.push(view.clone());
                }
                eval_views.push(view);
            } else {
                train_views.push(view);
            }
        } else {
            train_views.push(view);
        }
    }

    if let Some(val_views) = val_views {
        if load_args.train_on_eval {
            let mut train_paths: HashSet<_> = train_views
                .iter()
                .map(|view| view.image.path().to_path_buf())
                .collect();
            train_views.extend(
                val_views
                    .iter()
                    .filter(|view| train_paths.insert(view.image.path().to_path_buf()))
                    .cloned(),
            );
        }
        eval_views.extend(val_views);
    }

    let dataset = Dataset::from_views(train_views, eval_views);

    let load_args = load_args.clone();

    let mut init_splat = None;

    if let Some(init_path) = train_scene.ply_file_path {
        let init_path = transforms_path
            .parent()
            .expect("Transforms path must be a filename")
            .join(init_path);

        let ply_data = vfs.reader_at_path(&init_path).await;

        if let Ok(ply_data) = ply_data {
            init_splat = Some(load_splat_from_ply(ply_data, load_args.subsample_points).await?);
        }
    }

    Ok(DatasetLoadResult {
        init_splat,
        dataset,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn resolves_extensionless_frame_to_png_before_lookup() {
        let vfs = BrushVfs::create_test_vfs(vec![PathBuf::from("images/frame_001.png")]);

        assert_eq!(
            resolve_frame_image_path(&vfs, Path::new("transforms.json"), "images/frame_001").await,
            Some(PathBuf::from("images/frame_001.png"))
        );
    }
}

/// Prior discovery needs a real on-disk dataset (tempfile + fs), so these live
/// in a native-only module, in the style of the COLMAP loader's tests.
#[cfg(all(test, not(target_family = "wasm")))]
mod prior_tests {
    use crate::formats::prior_test_support::{test_config, write_depth_tiff, write_normal_tiff};
    use crate::formats::{DatasetError, DatasetLoadResult, FormatError, load_dataset};
    use brush_render::kernels::camera_model::CameraModel;
    use brush_vfs::BrushVfs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const IMG_W: u32 = 4;
    const IMG_H: u32 = 3;

    /// `transforms.json` with a single frame, plus that frame's PNG. Extra
    /// scene-level and per-frame keys are spliced in as raw json so each test
    /// can declare exactly what it means to exercise.
    async fn write_dataset(dir: &Path, extra_scene_keys: &str, extra_frame_keys: &str) {
        let transforms = format!(
            r#"{{
                "fl_x": 4.0, "fl_y": 3.0, "cx": 2.0, "cy": 1.5,
                "w": {IMG_W}, "h": {IMG_H}{extra_scene_keys},
                "frames": [
                    {{
                        "file_path": "images/frame_001.png",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]
                        ]{extra_frame_keys}
                    }}
                ]
            }}"#
        );
        tokio::fs::write(dir.join("transforms.json"), transforms)
            .await
            .expect("write transforms.json");

        let images_dir = dir.join("images");
        tokio::fs::create_dir_all(&images_dir)
            .await
            .expect("create images dir");
        image::RgbImage::from_pixel(IMG_W, IMG_H, image::Rgb([10, 20, 30]))
            .save(images_dir.join("frame_001.png"))
            .expect("write png");
    }

    async fn load(dir: &Path) -> DatasetLoadResult {
        try_load(dir).await.expect("load")
    }

    async fn try_load(dir: &Path) -> Result<DatasetLoadResult, DatasetError> {
        let vfs = Arc::new(BrushVfs::from_path(dir).await.expect("build vfs"));
        load_dataset(vfs, &test_config()).await
    }

    /// The COLMAP-side `depth/<stem>.tiff` + `normal/<stem>.tiff` layout must
    /// resolve here too -- that is the whole point of mirroring the convention.
    #[tokio::test]
    async fn discovers_priors_by_directory_convention() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_dataset(dir.path(), "", "").await;
        write_depth_tiff(&dir.path().join("depth/frame_001.tiff"), IMG_W, IMG_H).await;
        write_normal_tiff(&dir.path().join("normal/frame_001.tiff"), IMG_W, IMG_H).await;

        let result = load(dir.path()).await;
        let view = &result.dataset.train.views[0];

        assert_eq!(
            view.depth.as_ref().map(|d| d.path().to_path_buf()),
            Some(PathBuf::from("depth/frame_001.tiff")),
        );
        assert_eq!(
            view.normal.as_ref().map(|n| n.path().to_path_buf()),
            Some(PathBuf::from("normal/frame_001.tiff")),
        );

        // Loading actually decodes at the image's own resolution: this is the
        // no-resize contract, and a wrong-sized prior is an error, not a warp.
        let depth = view
            .depth
            .as_ref()
            .expect("depth prior")
            .load(IMG_H as usize, IMG_W as usize)
            .await
            .expect("depth must decode at the image resolution");
        assert_eq!(depth.shape, vec![IMG_H as usize, IMG_W as usize].into());

        let normal = view
            .normal
            .as_ref()
            .expect("normal prior")
            .load(IMG_H as usize, IMG_W as usize)
            .await
            .expect("normal must decode at the image resolution");
        assert_eq!(normal.shape, vec![IMG_H as usize, IMG_W as usize, 3].into());
    }

    /// Explicit per-frame keys win, and may point anywhere relative to the
    /// transforms file -- not just at the conventional directories.
    #[tokio::test]
    async fn declared_frame_keys_resolve() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_dataset(
            dir.path(),
            "",
            ",\n \"depth_file_path\": \"priors/d/frame_001.tiff\",\n \
             \"normal_file_path\": \"priors/n/frame_001.tiff\"",
        )
        .await;
        write_depth_tiff(&dir.path().join("priors/d/frame_001.tiff"), IMG_W, IMG_H).await;
        write_normal_tiff(&dir.path().join("priors/n/frame_001.tiff"), IMG_W, IMG_H).await;

        let result = load(dir.path()).await;
        let view = &result.dataset.train.views[0];

        assert_eq!(
            view.depth.as_ref().map(|d| d.path().to_path_buf()),
            Some(PathBuf::from("priors/d/frame_001.tiff")),
        );
        assert_eq!(
            view.normal.as_ref().map(|n| n.path().to_path_buf()),
            Some(PathBuf::from("priors/n/frame_001.tiff")),
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    /// Default-inertness pin: a dataset with no priors loads exactly as before,
    /// with no warnings and no priors invented.
    #[tokio::test]
    async fn dataset_without_priors_is_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_dataset(dir.path(), "", "").await;

        let result = load(dir.path()).await;
        let views = &result.dataset.train.views;
        assert_eq!(views.len(), 1);
        assert!(views[0].depth.is_none());
        assert!(views[0].normal.is_none());
        assert!(views[0].features.is_none());
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    /// A declared prior that is not on disk must be loud. A silent `None` here
    /// is exactly the failure this loader is being fixed for.
    #[tokio::test]
    async fn declared_prior_that_is_missing_warns() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_dataset(
            dir.path(),
            "",
            ",\n \"normal_file_path\": \"normal/nope.tiff\"",
        )
        .await;

        let result = load(dir.path()).await;
        assert!(result.dataset.train.views[0].normal.is_none());
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("normal") && w.contains("nope.tiff")),
            "expected a warning naming the missing prior, got {:?}",
            result.warnings
        );
    }

    /// `PINHOLE` is COLMAP's spelling of the zero-distortion model and shows up
    /// in real transforms.json files (`SplatCam` writes it). It used to fail the
    /// whole load.
    #[tokio::test]
    async fn accepts_pinhole_camera_model() {
        for spelling in ["PINHOLE", "pinhole", "PERSPECTIVE", "perspective"] {
            let dir = tempfile::tempdir().expect("tempdir");
            write_dataset(
                dir.path(),
                &format!(", \"camera_model\": \"{spelling}\""),
                "",
            )
            .await;

            let result = load(dir.path()).await;
            let cam = &result.dataset.train.views[0].camera;
            assert!(
                matches!(cam.camera_model, CameraModel::Pinhole),
                "`{spelling}` must resolve to a pinhole camera",
            );
        }
    }

    /// ...but an unknown model still hard-errors. A silently substituted camera
    /// model would train a wrong scene without ever saying so.
    #[tokio::test]
    async fn rejects_an_unknown_camera_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_dataset(dir.path(), ", \"camera_model\": \"EQUIRECTANGULAR\"", "").await;

        // `DatasetLoadResult` isn't `Debug`, so unwrap the Err arm by hand.
        let Err(err) = try_load(dir.path()).await else {
            panic!("an unsupported camera model must fail the load");
        };
        assert!(
            matches!(
                err,
                DatasetError::FormatError(FormatError::InvalidCamera(_))
            ),
            "expected an InvalidCamera error, got {err:?}"
        );
    }
}
