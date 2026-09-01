use super::{
    DatasetFileIndex, DatasetLoadResult, FormatError, find_depth_path, find_normal_path,
    opengl_c2w_to_pose, split_eval_every,
};
use crate::{
    Dataset,
    config::LoadDatasetConfig,
    scene::{LoadDepth, LoadImage, LoadNormal, SceneView},
};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::CameraModel::{Pinhole, RadialTangential8};
use brush_render::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use brush_vfs::BrushVfs;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;

/// `RealityCapture` / `RealityScan` "Internal/External camera parameters" CSV —
/// the flat per-image format Postshot ingests. One row per image, e.g.:
/// `#name,x,y,alt,heading,pitch,roll,f,px,py,k1,k2,k3,k4,t1,t2`.
///
/// `x,y,alt` is the camera position and `heading,pitch,roll` its orientation
/// (degrees). `f`, `px` and `py` are in 35mm-film units (36mm reference): `f`
/// is the focal length and `px,py` the principal point offset from the image
/// center. All three scale by the larger image dimension to reach pixels.
/// `k1..k4` are Brown polynomial radial coeffs (r^2, r^4, r^6, r^8) and `t1,t2`
/// the tangential coeffs; brush has no r^8 term, so brown4's `k4` is dropped
/// (brown3 approximation) with a warning.
///
/// The column set is a user-customizable template, so optional columns may be
/// absent: distortion (`k*`/`t*`) defaults to none (pinhole) and the principal
/// point (`px`/`py`) to the image center. Only the pose and focal columns below
/// are required — they also serve as the format's detection signature.
const REQUIRED_COLUMNS: &[&str] = &["name", "x", "y", "alt", "heading", "pitch", "roll", "f"];

/// Map header names (lowercased, leading `#` stripped) to their column index.
fn parse_header(line: &str) -> Option<HashMap<String, usize>> {
    let cols: HashMap<String, usize> = line
        .split(',')
        .enumerate()
        .map(|(i, name)| (name.trim().trim_start_matches('#').to_lowercase(), i))
        .collect();
    REQUIRED_COLUMNS
        .iter()
        .all(|c| cols.contains_key(*c))
        .then_some(cols)
}

fn col<'a>(fields: &'a [&'a str], header: &HashMap<String, usize>, name: &str) -> Option<&'a str> {
    header.get(name).and_then(|&i| fields.get(i)).copied()
}

fn col_f64(fields: &[&str], header: &HashMap<String, usize>, name: &str) -> f64 {
    col(fields, header, name)
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

pub async fn read_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Option<Result<DatasetLoadResult, FormatError>> {
    let csv_paths: Vec<_> = vfs.files_with_extension("csv").collect();

    // Find a csv whose header looks like the RealityCapture camera format.
    for path in csv_paths {
        let Ok(mut reader) = vfs.reader_at_path(&path).await else {
            continue;
        };
        let mut buf = String::new();
        if reader.read_to_string(&mut buf).await.is_err() {
            continue;
        }
        let Some(first_line) = buf.lines().find(|l| !l.trim().is_empty()) else {
            continue;
        };
        if parse_header(first_line).is_some() {
            log::info!("Loading RealityCapture dataset from {path:?}");
            return Some(read_dataset_inner(vfs, load_args, buf).await);
        }
    }

    None
}

async fn read_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
    contents: String,
) -> Result<DatasetLoadResult, FormatError> {
    let mut lines = contents.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .and_then(parse_header)
        .ok_or_else(|| FormatError::InvalidFormat("RealityCapture csv has no header".to_owned()))?;

    let mut views = Vec::new();
    let mut warnings = Vec::new();
    let mut warned_brown4 = false;
    let file_index = DatasetFileIndex::new(&vfs);
    // The csv's `name` column is a bare file name; see colmap.rs.
    warnings.extend(file_index.ambiguity_warnings());

    for line in lines
        .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
        .take(load_args.max_frames.unwrap_or(usize::MAX))
    {
        brush_async::yield_now().await;

        let fields: Vec<&str> = line.split(',').collect();
        let Some(name) = col(&fields, &header, "name").map(str::trim) else {
            continue;
        };

        // brush's distortion model has no 4th-order (r^8) radial term, so the
        // brown4 `k4` coefficient can't be represented; fall back to the brown3
        // approximation and flag it once.
        if !warned_brown4 && col_f64(&fields, &header, "k4") != 0.0 {
            warnings.push(
                "RealityCapture brown4 radial term (k4) isn't supported; approximating with brown3"
                    .to_owned(),
            );
            warned_brown4 = true;
        }

        let Some(image_path) = file_index.find_image_by_name(name).map(Path::to_path_buf) else {
            warnings.push(format!("Skipped '{name}': image file not found"));
            continue;
        };

        let mask_path = file_index
            .find_mask_path(&image_path)
            .map(Path::to_path_buf);

        // Geometry priors, discovered exactly as in the COLMAP and nerfstudio
        // loaders: `depth/<stem>` and `normal/<stem>` alongside the image
        // (`super::find_prior_path`). The RealityCapture csv is a fixed
        // pose/intrinsics column template with no place to name a per-image
        // sidecar, so the directory convention is the only route here.
        let depth = find_depth_path(&vfs, &image_path)?
            .map(|p| LoadDepth::new(vfs.clone(), p.to_path_buf()));
        // Carries no scale, so -- as in colmap.rs -- it plays no part in any
        // metric-scale estimation.
        let normal = find_normal_path(&vfs, &image_path)?
            .map(|p| LoadNormal::new(vfs.clone(), p.to_path_buf()));

        let image = LoadImage::new(
            vfs.clone(),
            image_path,
            mask_path,
            load_args.max_resolution,
            load_args.alpha_mode,
            load_args.invert_masks,
        );

        // The csv carries no image dimensions; intrinsics are resolution
        // independent once expressed as fov + normalized center, so a
        // header-only dimension read (no full decode) is enough.
        let (w, h) = image.dimensions().await?;

        let camera = row_to_camera(&fields, &header, w, h);
        if !camera.is_valid() {
            warnings.push(format!(
                "Skipped '{name}': camera contains nan or inf values"
            ));
            continue;
        }

        views.push(SceneView {
            camera,
            image,
            features: None,
            depth,
            normal,
        });
    }

    let (train_views, eval_views) =
        split_eval_every(views, load_args.eval_split_every, load_args.train_on_eval);

    Ok(DatasetLoadResult {
        init_splat: None,
        dataset: Dataset::from_views(train_views, eval_views),
        warnings,
    })
}

/// Build a brush [`Camera`] from one csv row and the image dimensions.
fn row_to_camera(fields: &[&str], header: &HashMap<String, usize>, w: u32, h: u32) -> Camera {
    let scale = w.max(h) as f64;

    let focal = col_f64(fields, header, "f") * scale / 36.0;
    let cx = col_f64(fields, header, "px") * scale + w as f64 / 2.0;
    let cy = col_f64(fields, header, "py") * scale + h as f64 / 2.0;
    let center_uv = glam::vec2((cx / w as f64) as f32, (cy / h as f64) as f32);

    let camera_model = build_camera_model(
        col_f64(fields, header, "k1"),
        col_f64(fields, header, "k2"),
        col_f64(fields, header, "k3"),
        col_f64(fields, header, "t1"),
        col_f64(fields, header, "t2"),
    );
    let fov_x = focal_to_fov(focal, w, &camera_model);
    let fov_y = focal_to_fov(focal, h, &camera_model);

    let heading = col_f64(fields, header, "heading") as f32;
    let pitch = col_f64(fields, header, "pitch") as f32;
    let roll = col_f64(fields, header, "roll") as f32;

    // RealityCapture orientation: yaw(-heading) about Z, then pitch about X,
    // then roll about Y, yielding a camera-to-world rotation in the OpenGL
    // basis (matches nerfstudio's RealityCapture importer).
    let rotation = glam::Quat::from_rotation_z(-heading.to_radians())
        * glam::Quat::from_rotation_x(pitch.to_radians())
        * glam::Quat::from_rotation_y(roll.to_radians());
    let position = glam::vec3(
        col_f64(fields, header, "x") as f32,
        col_f64(fields, header, "y") as f32,
        col_f64(fields, header, "alt") as f32,
    );
    let c2w = glam::Mat4::from_rotation_translation(rotation, position);
    let (position, rotation) = opengl_c2w_to_pose(c2w);

    Camera::new(position, rotation, fov_x, fov_y, center_uv, camera_model)
}

/// `RealityCapture`'s brown3+tangential model maps onto `RadialTangential8`:
/// `k1,k2,k3` are the polynomial radial terms (the numerator, with the rational
/// denominator left at zero) and `t1,t2` the Brown-Conrady tangential terms.
/// The brown4 `k4` (an r^8 term) has no slot here and is handled by the caller.
fn build_camera_model(k1: f64, k2: f64, k3: f64, t1: f64, t2: f64) -> CameraModel {
    if [k1, k2, k3, t1, t2].iter().all(|v| *v == 0.0) {
        return Pinhole;
    }
    RadialTangential8(RadialTangential8Params {
        k1: k1 as f32,
        k2: k2 as f32,
        k3: k3 as f32,
        k4: 0.0,
        k5: 0.0,
        k6: 0.0,
        p1: t1 as f32,
        p2: t2 as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    const HEADER: &str = "#name,x,y,alt,heading,pitch,roll,f,px,py,k1,k2,k3,k4,t1,t2";
    // First data row of the Postshot sample (schliemann.csv).
    const ROW: &str = "frame_00001.jpeg,44.5876747664166,138.823621534044,6.821916401534405,170.3483067926429,85.18637269312288,-22.74995074830745,14.2390682243052,-9.482184385774318e-004,-2.446553068050568e-004,3.114799768048152e-003,4.026391718074555e-003,-1.795976992379612e-003,0,0,0";

    #[wasm_bindgen_test(unsupported = test)]
    fn test_parse_header() {
        let header = parse_header(HEADER).expect("header should parse");
        assert_eq!(header["name"], 0);
        assert_eq!(header["alt"], 3);
        assert_eq!(header["t2"], 15);
        // A csv that isn't the RealityCapture camera format is rejected.
        assert!(parse_header("a,b,c").is_none());
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_row_to_camera() {
        let header = parse_header(HEADER).expect("header should parse");
        let fields: Vec<&str> = ROW.split(',').collect();
        let cam = row_to_camera(&fields, &header, 3840, 2880);

        assert!(cam.is_valid());

        // x,y,alt map straight to the world position (the basis swap only
        // reorients the camera, it doesn't move it).
        assert!((cam.position - glam::vec3(44.587_674, 138.823_62, 6.821_916)).length() < 1e-3);

        // Near-zero px/py keep the principal point at the image center.
        assert!((cam.center_uv - glam::vec2(0.5, 0.5)).length() < 1e-2);

        // f=14.24 (35mm equiv) on a 4:3 frame is a wide lens; fov_x > fov_y > 0.
        assert!(cam.fov_x > cam.fov_y && cam.fov_y > 0.0);
        assert!((1.0..2.0).contains(&cam.fov_x));

        // Non-zero radial coeffs select the distortion model.
        assert!(matches!(cam.camera_model, RadialTangential8(_)));
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_minimal_header_no_distortion() {
        // A customized template can drop the optional principal-point and
        // distortion columns; the camera is then a centered pinhole.
        let header = parse_header("#name,x,y,alt,heading,pitch,roll,f")
            .expect("minimal header should parse");
        let fields: Vec<&str> = "img.png,1,2,3,10,20,30,20.0".split(',').collect();
        let cam = row_to_camera(&fields, &header, 1920, 1080);
        assert!(cam.is_valid());
        assert!(matches!(cam.camera_model, Pinhole));
        assert!((cam.center_uv - glam::vec2(0.5, 0.5)).length() < 1e-6);
        assert!((cam.position - glam::vec3(1.0, 2.0, 3.0)).length() < 1e-3);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_build_camera_model() {
        assert!(matches!(
            build_camera_model(0.0, 0.0, 0.0, 0.0, 0.0),
            Pinhole
        ));
        // brown3 radial (k1,k2,k3) -> numerator; tangential (t1,t2) -> p1,p2;
        // the rational denominator (k4,k5,k6) stays zero so it's a pure
        // polynomial.
        let RadialTangential8(p) = build_camera_model(1.0, 2.0, 3.0, 4.0, 5.0) else {
            panic!("expected RadialTangential8");
        };
        assert_eq!(
            (p.k1, p.k2, p.k3, p.k4, p.k5, p.k6, p.p1, p.p2),
            (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0)
        );
    }
}

/// Prior discovery walks the VFS, so it needs a real on-disk dataset. Mirrors
/// the nerfstudio loader's `prior_tests` module.
#[cfg(all(test, not(target_family = "wasm")))]
mod prior_tests {
    use crate::formats::prior_test_support::{test_config, write_depth_tiff, write_normal_tiff};
    use crate::formats::{DatasetLoadResult, load_dataset};
    use brush_vfs::BrushVfs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const IMG_W: u32 = 4;
    const IMG_H: u32 = 3;

    /// A minimal `RealityCapture` camera csv (pose + focal only -- the optional
    /// principal-point and distortion columns are a customizable template) plus
    /// the one image it names.
    async fn write_dataset(dir: &Path) {
        tokio::fs::write(
            dir.join("cameras.csv"),
            "#name,x,y,alt,heading,pitch,roll,f\nframe_001.png,1,2,3,10,20,30,20.0\n",
        )
        .await
        .expect("write csv");

        let images_dir = dir.join("images");
        tokio::fs::create_dir_all(&images_dir)
            .await
            .expect("create images dir");
        image::RgbImage::from_pixel(IMG_W, IMG_H, image::Rgb([10, 20, 30]))
            .save(images_dir.join("frame_001.png"))
            .expect("write png");
    }

    async fn load(dir: &Path) -> DatasetLoadResult {
        let vfs = Arc::new(BrushVfs::from_path(dir).await.expect("build vfs"));
        load_dataset(vfs, &test_config()).await.expect("load")
    }

    /// Same `depth/<stem>` + `normal/<stem>` convention as the COLMAP and
    /// nerfstudio loaders. The csv has no per-image sidecar column, so the
    /// directory layout is the only route in.
    #[tokio::test]
    async fn discovers_priors_by_directory_convention() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_dataset(dir.path()).await;
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

        // No-resize contract: the prior decodes at the image's own resolution.
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

    /// Default-inertness pin: a prior-free csv dataset loads exactly as before.
    #[tokio::test]
    async fn dataset_without_priors_is_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_dataset(dir.path()).await;

        let result = load(dir.path()).await;
        let views = &result.dataset.train.views;
        assert_eq!(views.len(), 1);
        assert!(views[0].depth.is_none());
        assert!(views[0].normal.is_none());
        assert!(views[0].features.is_none());
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }
}
