use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::{DatasetLoadResult, FormatError};
use crate::{
    Dataset,
    config::LoadDatasetConfig,
    formats::{
        DatasetFileIndex, find_depth_path, find_features_path, find_image_by_name,
        find_normal_path, find_points3d_path, split_eval_every,
    },
    scene::{LoadDepth, LoadFeatures, LoadImage, LoadNormal, SceneView},
};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::CameraModel::{
    KannalaBrandt4, Pinhole, RadialTangential8, ThinPrismFisheye,
};
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use brush_render::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use brush_render::kernels::camera_model::thin_prism_fisheye::ThinPrismFisheyeParams;
use brush_render::{
    camera::{self, Camera},
    sh::rgb_to_sh,
};
use brush_serde::{ParseMetadata, SplatData, SplatMessage};
use brush_vfs::BrushVfs;
use colmap_reader::{ColmapCamera, ColmapCameraModel};

/// COLMAP can emit several independent sparse reconstructions (`sparse/0`,
/// `sparse/1`, ...) when the image graph is disconnected. They share no
/// coordinate frame and cannot be merged here, so we pick the one that
/// registered the most images (COLMAP's own "largest first" convention,
/// determined empirically rather than trusting directory names).
async fn select_colmap_model(vfs: &BrushVfs) -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = vfs
        .files_ending_in("cameras.bin")
        .chain(vfs.files_ending_in("cameras.txt"))
        .map(Path::to_path_buf)
        .collect();

    if candidates.len() <= 1 {
        return candidates.into_iter().next();
    }

    let mut best: Option<(usize, PathBuf)> = None;
    for cam in &candidates {
        let dir = cam
            .parent()
            .expect("colmap cameras file must have a parent");
        let is_binary = cam.extension().and_then(|e| e.to_str()) == Some("bin");
        let img_path = dir.join(if is_binary {
            "images.bin"
        } else {
            "images.txt"
        });

        let Some(count) = count_registered_images(vfs, &img_path, is_binary).await else {
            log::warn!(
                "Skipping colmap model '{}': can't read images",
                dir.display()
            );
            continue;
        };
        log::info!("Colmap model '{}' registered {count} images", dir.display());

        // Tie-break on path so the choice is deterministic (VFS iteration isn't).
        let better = best
            .as_ref()
            .is_none_or(|(bc, bp)| count > *bc || (count == *bc && cam < bp));
        if better {
            best = Some((count, cam.clone()));
        }
    }

    // If every candidate failed to read, fall through to a deterministic pick
    // so the caller still surfaces a proper parse error downstream.
    let chosen = best
        .map(|(_, p)| p)
        .or_else(|| candidates.iter().min().cloned())?;
    log::info!(
        "Selected colmap model '{}'",
        chosen
            .parent()
            .expect("colmap cameras file must have a parent")
            .display()
    );
    Some(chosen)
}

async fn count_registered_images(
    vfs: &BrushVfs,
    img_path: &Path,
    is_binary: bool,
) -> Option<usize> {
    let mut file = vfs.reader_at_path(img_path).await.ok()?;
    let imgs = colmap_reader::read_images(&mut file, is_binary, false)
        .await
        .ok()?;
    Some(imgs.len())
}

pub(crate) async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Option<Result<DatasetLoadResult, FormatError>> {
    log::info!("Loading colmap dataset");

    let cam_path = select_colmap_model(&vfs).await?;
    let dir = cam_path
        .parent()
        .expect("colmap cameras file must have a parent");
    let is_binary = cam_path.extension().and_then(|e| e.to_str()) == Some("bin");
    let img_path = dir.join(if is_binary {
        "images.bin"
    } else {
        "images.txt"
    });

    Some(load_dataset_inner(vfs, load_args, cam_path, img_path).await)
}

async fn load_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
    cam_path: PathBuf,
    img_path: PathBuf,
) -> Result<DatasetLoadResult, FormatError> {
    let is_binary = cam_path.ends_with("cameras.bin");

    log::info!("Parsing colmap camera info");

    let load_args = load_args.clone();
    let vfs = vfs.clone();

    let vfs_init = vfs.clone();

    // Resolve points3d from the same reconstruction as the chosen cameras,
    // not an arbitrary one elsewhere in the VFS.
    let points_dir = cam_path
        .parent()
        .expect("colmap cameras file must have a parent")
        .to_path_buf();
    let points_dir_ = points_dir.clone();

    // One actor for both halves of the colmap load — the camera/image
    // parse and the points3d parse run concurrently on the same thread
    // (no cross-stream GPU concerns; this is pure CPU/I/O).
    let actor = brush_async::Actor::new("colmap-loader");
    let features_dir_name = load_args.features_dir_name.clone();
    let dataset = actor.run(move || async move {
        let mut cam_file = vfs.reader_at_path(&cam_path).await?;
        let cam_model_data = colmap_reader::read_cameras(&mut cam_file, is_binary).await?;
        let cam_model_data = cam_model_data
            .into_iter()
            .map(|cam| (cam.id, cam))
            .collect::<HashMap<_, _>>();
        let mut img_file = vfs.reader_at_path(&img_path).await?;
        let img_infos =
            colmap_reader::read_images(&mut img_file, is_binary, load_args.estimate_metric_scale)
                .await?;
        let mut img_info_list = img_infos.into_iter().collect::<Vec<_>>();
        img_info_list.sort_by(|img_a, img_b| img_a.name.cmp(&img_b.name));

        log::info!("Loading {} images for colmap dataset", img_info_list.len());

        // COLMAP is reconstructed up to an unknown global scale.
        // If metric depth maps are present, recover that scale so poses + points line
        // up with the depth.
        let metric_scale = if load_args.estimate_metric_scale {
            let scale = estimate_metric_scale(&vfs, &img_info_list, &cam_model_data, &points_dir_)
                .await?
                .expect("estimate metric scale failed");
            log::info!("Rescaling colmap reconstruction to metric depth (scale = {scale})");
            Some(scale)
        } else {
            None
        };

        let mut views = Vec::new();
        let mut warnings = Vec::new();
        let file_index = DatasetFileIndex::new(&vfs);
        // images.txt stores bare file names, so any name claimed by two files
        // is resolved by a tie-break the operator never asked for. Say so.
        warnings.extend(file_index.ambiguity_warnings());

        for img_info in img_info_list
            .iter()
            .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
            .take(load_args.max_frames.unwrap_or(usize::MAX))
        {
            let colmap_camera = cam_model_data
                .get(&img_info.camera_id)
                .ok_or_else(|| {
                    FormatError::InvalidFormat(format!(
                        "Image '{}' references camera ID {} which doesn't exist in camera data",
                        img_info.name, img_info.camera_id
                    ))
                })?
                .clone();

            // Create a future to handle loading the image.
            let camera_model = build_camera_model(&colmap_camera);
            let focal = colmap_camera.focal();
            let fovx = camera::focal_to_fov(focal.0, colmap_camera.width as u32, &camera_model);
            let fovy = camera::focal_to_fov(focal.1, colmap_camera.height as u32, &camera_model);
            let center = colmap_camera.principal_point();
            let center_uv =
                center / glam::vec2(colmap_camera.width as f32, colmap_camera.height as f32);

            let Some(path) = file_index.find_image_by_name(&img_info.name) else {
                warnings.push(format!("Skipped '{}': image file not found", img_info.name));
                continue;
            };

            let mask_path = file_index.find_mask_path(path);

            let features = find_features_path(&vfs, path, &features_dir_name)
                .map(|p| LoadFeatures::new(vfs.clone(), p.to_path_buf()));
            let depth =
                find_depth_path(&vfs, path)?.map(|p| LoadDepth::new(vfs.clone(), p.to_path_buf()));
            // Surface-normal prior, discovered exactly like depth. Carries no
            // scale, so it is deliberately NOT part of the metric-scale
            // estimation above.
            let normal = find_normal_path(&vfs, path)?
                .map(|p| LoadNormal::new(vfs.clone(), p.to_path_buf()));

            // Convert w2c to c2w.
            let world_to_cam = glam::Affine3A::from_rotation_translation(
                img_info.quat,
                img_info.tvec * metric_scale.unwrap_or(1.0),
            );
            let cam_to_world = world_to_cam.inverse();
            let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

            let camera = Camera::new(translation, quat, fovx, fovy, center_uv, camera_model);

            if !camera.is_valid() {
                warnings.push(format!(
                    "Skipped '{}': camera contains nan or inf values",
                    img_info.name
                ));
                continue;
            }

            let image = LoadImage::new(
                vfs.clone(),
                path.to_path_buf(),
                mask_path.map(|p| p.to_path_buf()),
                load_args.max_resolution,
                load_args.alpha_mode,
                load_args.invert_masks,
            );

            views.push(SceneView {
                camera,
                image,
                features,
                depth,
                normal,
            });
        }

        let (train_views, eval_views) =
            split_eval_every(views, load_args.eval_split_every, load_args.train_on_eval);

        Result::<_, FormatError>::Ok((
            Dataset::from_views(train_views, eval_views),
            warnings,
            metric_scale,
        ))
    });

    let load_args = load_args.clone();

    let init = actor.run(move || async move {
        let (points_path, is_binary) = find_points3d_path(&vfs_init, &points_dir)?;
        // At this point the VFS has said this file exists so just unwrap.
        let mut points_file = vfs_init
            .reader_at_path(points_path)
            .await
            .expect("unreachable");

        let step = load_args.subsample_points.unwrap_or(1) as usize;
        let points_data = colmap_reader::read_points3d(&mut points_file, is_binary, false)
            .await
            .ok()?;

        if points_data.is_empty() {
            return None;
        }

        let positions: Vec<f32> = points_data
            .iter()
            .step_by(step)
            .flat_map(|p| p.xyz.to_array())
            .collect();
        let colors: Vec<f32> = points_data
            .iter()
            .step_by(step)
            .flat_map(|p| {
                let sh = rgb_to_sh(glam::vec3(
                    p.rgb[0] as f32 / 255.0,
                    p.rgb[1] as f32 / 255.0,
                    p.rgb[2] as f32 / 255.0,
                ));
                [sh.x, sh.y, sh.z]
            })
            .collect();

        let n_splats = positions.len() / 3;
        log::info!("Starting from colmap points: {n_splats}");
        let data = SplatData {
            means: positions,
            rotations: None,
            log_scales: None,
            sh_coeffs: Some(colors),
            raw_opacities: None,
        };

        Some(SplatMessage {
            meta: ParseMetadata {
                up_axis: None,
                render_mode: None,
                total_splats: n_splats as u32,
                progress: 1.0,
            },
            data,
        })
    });

    // Wait for both halves.
    let (dataset, init) = tokio::join!(dataset, init);
    let ((dataset, warnings, metric_scale), mut init_splat) = (dataset?, init);

    if let (Some(scale), Some(splat)) = (metric_scale, init_splat.as_mut()) {
        for m in &mut splat.data.means {
            *m *= scale;
        }
    }

    Ok(DatasetLoadResult {
        init_splat,
        dataset,
        warnings,
    })
}

fn build_camera_model(colmap_camera: &ColmapCamera) -> CameraModel {
    let p = &colmap_camera.params;
    // Param layouts follow COLMAP's `src/colmap/sensor/models.h`. Indices
    // are 0-based positions into `p` after the intrinsics (fx, fy, cx, cy
    // or f, cx, cy depending on the model).
    match colmap_camera.model {
        // No distortion.
        ColmapCameraModel::SimplePinhole | ColmapCameraModel::Pinhole => Pinhole,
        // Pure-radial perspective models → RT8 with the higher-order /
        // tangential coefficients zeroed.
        // SIMPLE_RADIAL: f cx cy k1
        ColmapCameraModel::SimpleRadial => RadialTangential8(RadialTangential8Params {
            k1: p[3] as f32,
            ..Default::default()
        }),
        // RADIAL: f cx cy k1 k2
        ColmapCameraModel::Radial => RadialTangential8(RadialTangential8Params {
            k1: p[3] as f32,
            k2: p[4] as f32,
            ..Default::default()
        }),
        // OPENCV: fx fy cx cy k1 k2 p1 p2 (Brown-Conrady, 4 distortion coefficients).
        ColmapCameraModel::OpenCV => RadialTangential8(RadialTangential8Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            p1: p[6] as f32,
            p2: p[7] as f32,
            ..Default::default()
        }),
        // FULL_OPENCV: fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6.
        ColmapCameraModel::FullOpenCV => RadialTangential8(RadialTangential8Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            k3: p[8] as f32,
            k4: p[9] as f32,
            k5: p[10] as f32,
            k6: p[11] as f32,
            p1: p[6] as f32,
            p2: p[7] as f32,
        }),
        // Fisheye variants → KB4 with unused k's zeroed.
        // SIMPLE_RADIAL_FISHEYE: f cx cy k1
        ColmapCameraModel::SimpleRadialFisheye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[3] as f32,
            ..Default::default()
        }),
        // RADIAL_FISHEYE: f cx cy k1 k2
        ColmapCameraModel::RadialFisheye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[3] as f32,
            k2: p[4] as f32,
            ..Default::default()
        }),
        // OPENCV_FISHEYE: fx fy cx cy k1 k2 k3 k4
        ColmapCameraModel::OpenCvFishEye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            k3: p[6] as f32,
            k4: p[7] as f32,
        }),
        // THIN_PRISM_FISHEYE: fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1
        ColmapCameraModel::ThinPrismFisheye => ThinPrismFisheye(ThinPrismFisheyeParams {
            kb4: KannalaBrandt4Params {
                k1: p[4] as f32,
                k2: p[5] as f32,
                k3: p[8] as f32,
                k4: p[9] as f32,
            },
            p1: p[6] as f32,
            p2: p[7] as f32,
            sx1: p[10] as f32,
            sy1: p[11] as f32,
        }),
        // FOV uses a tan(ω r) / ω model that doesn't fit either of our
        // distortion polynomials. Fall back to pinhole — rare in practice.
        ColmapCameraModel::Fov => {
            log::warn!("COLMAP `FOV` model is not directly supported; falling back to pinhole.");
            Pinhole
        }
    }
}

/// Estimate the global scale that maps the COLMAP reconstruction into
/// the metric frame of the depth maps.
async fn estimate_metric_scale(
    vfs: &Arc<BrushVfs>,
    images: &[colmap_reader::Image],
    cameras: &HashMap<i32, ColmapCamera>,
    points_dir: &Path,
) -> Result<Option<f32>, FormatError> {
    // An ambiguous prior must surface as an error here, not as "no depth maps
    // found" -- that would turn a stale-file mistake into a bare panic on the
    // caller's `.expect`.
    let mut any_depth = false;
    for img in images {
        if let Some(image_path) = find_image_by_name(vfs, &img.name)
            && super::find_depth_path(vfs, image_path)?.is_some()
        {
            any_depth = true;
            break;
        }
    }
    if !any_depth {
        return Ok(None);
    }

    let Some((points_path, is_binary)) = find_points3d_path(vfs, points_dir) else {
        return Ok(None);
    };
    let Ok(mut points_file) = vfs.reader_at_path(points_path).await else {
        return Ok(None);
    };
    let Ok(points) = colmap_reader::read_points3d(&mut points_file, is_binary, false).await else {
        return Ok(None);
    };
    let point_map: HashMap<i64, glam::Vec3> = points.iter().map(|p| (p.id, p.xyz)).collect();

    let mut accumulated_colmap_depth = 0.0;
    let mut accumulated_dataset_depth = 0.0;

    for img in images {
        let Some(point_data) = &img.points else {
            continue;
        };
        let Some(image_path) = find_image_by_name(vfs, &img.name) else {
            continue;
        };
        let Some(depth_path) = find_depth_path(vfs, image_path)? else {
            continue;
        };
        let Some(cam) = cameras.get(&img.camera_id) else {
            continue;
        };

        let height = cam.height as usize;
        let width = cam.width as usize;

        let depth_loader = LoadDepth::new(vfs.clone(), depth_path.to_path_buf());
        let Ok(depth) = depth_loader.load_vec(height, width).await else {
            continue;
        };

        for (xy, &pid) in point_data.xys.iter().zip(point_data.point3d_ids.iter()) {
            if pid < 0 {
                continue;
            }
            let Some(&xyz) = point_map.get(&pid) else {
                continue;
            };
            let colmap_depth = (img.quat * xyz + img.tvec).z;

            let u = xy.x as i32;
            let v = xy.y as i32;

            if u < 0 || v < 0 || u as usize >= width || v as usize >= height {
                continue;
            }

            let expected_depth = depth[v as usize * width + u as usize];
            if expected_depth <= 0.0 || colmap_depth <= 0.0 {
                continue;
            }
            accumulated_colmap_depth += colmap_depth;
            accumulated_dataset_depth += expected_depth;
        }
    }

    let scale = accumulated_dataset_depth / accumulated_colmap_depth;

    log::info!("Estimated metric scale {scale}");
    Ok(Some(scale))
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::config::LoadDatasetConfig;
    use crate::formats::prior_test_support::write_depth_png;
    use crate::formats::{DatasetLoadResult, load_dataset};
    use brush_render::camera::focal_to_fov;
    use brush_render::kernels::camera_model::CameraModel;
    use brush_render::sh::rgb_to_sh;
    use brush_vfs::BrushVfs;
    use std::path::Path;
    use std::sync::Arc;

    const IMG_W: u32 = 64;
    const IMG_H: u32 = 48;
    const FX: f64 = 80.0;
    const FY: f64 = 70.0;
    const CX: f64 = 30.0;
    const CY: f64 = 20.0;

    /// World-to-cam pose written to images.txt for `img1.png`: a 90° rotation
    /// about Y plus a translation.
    fn img1_w2c() -> (glam::Quat, glam::Vec3) {
        (
            glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            glam::vec3(1.0, 0.0, 2.0),
        )
    }

    /// Write a minimal COLMAP text-format model plus tiny PNG files to `dir`.
    /// `missing.png` is referenced in images.txt but has no image file, so the
    /// loader should skip it with a warning.
    async fn write_test_dataset(dir: &Path) {
        let sparse = dir.join("sparse/0");
        tokio::fs::create_dir_all(&sparse).await.unwrap();

        let cameras = format!(
            "# Camera list with one line of data per camera:\n\
             #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\
             1 PINHOLE {IMG_W} {IMG_H} {FX} {FY} {CX} {CY}\n"
        );
        tokio::fs::write(sparse.join("cameras.txt"), cameras)
            .await
            .unwrap();

        let (q, t) = img1_w2c();
        let images = format!(
            "# Image list with two lines of data per image:\n\
             #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\
             #   POINTS2D[] as (X, Y, POINT3D_ID)\n\
             1 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 img0.png\n\
             \n\
             2 {} {} {} {} {} {} {} 1 img1.png\n\
             10.0 20.0 1 30.0 40.0 -1\n\
             3 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 img2.png\n\
             \n\
             4 1.0 0.0 0.0 0.0 5.0 5.0 5.0 1 missing.png\n\
             \n",
            q.w, q.x, q.y, q.z, t.x, t.y, t.z
        );
        tokio::fs::write(sparse.join("images.txt"), images)
            .await
            .unwrap();

        let points = "# 3D point list with one line of data per point:\n\
                      #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n\
                      1 1.5 2.5 3.5 255 0 0 0.5 1 0\n\
                      2 -1.0 0.0 1.0 0 255 0 0.5 2 1\n\
                      3 0.0 1.0 0.0 0 0 255 0.5 3 0\n\
                      4 2.0 2.0 2.0 128 128 128 0.5 1 1\n";
        tokio::fs::write(sparse.join("points3D.txt"), points)
            .await
            .unwrap();

        let images_dir = dir.join("images");
        tokio::fs::create_dir_all(&images_dir).await.unwrap();
        for name in ["img0.png", "img1.png", "img2.png"] {
            let img = image::RgbImage::from_pixel(4, 3, image::Rgb([10, 20, 30]));
            img.save(images_dir.join(name)).unwrap();
        }
    }

    fn test_config(eval_split_every: Option<usize>) -> LoadDatasetConfig {
        LoadDatasetConfig {
            max_frames: None,
            max_resolution: 1920,
            eval_split_every,
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

    async fn load_test_dataset(dir: &Path, eval_split_every: Option<usize>) -> DatasetLoadResult {
        let vfs = Arc::new(BrushVfs::from_path(dir).await.unwrap());
        load_dataset(vfs, &test_config(eval_split_every))
            .await
            .unwrap()
    }

    fn assert_vec3_close(a: glam::Vec3, b: glam::Vec3) {
        assert!((a - b).length() < 1e-4, "expected {b:?}, got {a:?}");
    }

    #[tokio::test]
    async fn loads_text_model() {
        let dir = tempfile::tempdir().unwrap();
        write_test_dataset(dir.path()).await;
        let result = load_test_dataset(dir.path(), None).await;

        // All three found images are train views; the missing one is skipped
        // with a warning instead of failing the load.
        let views = &result.dataset.train.views;
        assert_eq!(views.len(), 3);
        assert!(result.dataset.eval.is_none());
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("missing.png"));

        // Intrinsics: fov derives from the written focal, and roundtrips back.
        let cam = &views[0].camera;
        assert!(matches!(cam.camera_model, CameraModel::Pinhole));
        let expected_fov_x = focal_to_fov(FX, IMG_W, &CameraModel::Pinhole);
        let expected_fov_y = focal_to_fov(FY, IMG_H, &CameraModel::Pinhole);
        assert!((cam.fov_x - expected_fov_x).abs() < 1e-6);
        assert!((cam.fov_y - expected_fov_y).abs() < 1e-6);
        let focal = cam.focal(glam::uvec2(IMG_W, IMG_H));
        assert!((focal.x - FX as f32).abs() < 1e-3);
        assert!((focal.y - FY as f32).abs() < 1e-3);
        let center = cam.center(glam::uvec2(IMG_W, IMG_H));
        assert!((center.x - CX as f32).abs() < 1e-3);
        assert!((center.y - CY as f32).abs() < 1e-3);

        // Views are sorted by image name. img0 has identity rotation and
        // translation (1, 2, 3) in world-to-cam, so cam-to-world is just the
        // negated translation.
        assert_vec3_close(views[0].camera.position, glam::vec3(-1.0, -2.0, -3.0));
        // Note: angle_between amplifies ulp-level differences (acos near 1),
        // so 1e-3 rad (~0.06 degrees) is as tight as is meaningful here.
        assert!(views[0].camera.rotation.angle_between(glam::Quat::IDENTITY) < 1e-3);
        // img2 sits at the origin.
        assert_vec3_close(views[2].camera.position, glam::Vec3::ZERO);

        // img1 has a non-trivial pose: compare against inverting the same
        // world-to-cam transform that was written to images.txt.
        let (q, t) = img1_w2c();
        let cam_to_world = glam::Affine3A::from_rotation_translation(q, t).inverse();
        let (_, expected_rot, expected_pos) = cam_to_world.to_scale_rotation_translation();
        assert_vec3_close(views[1].camera.position, expected_pos);
        assert!(
            views[1].camera.rotation.angle_between(expected_rot) < 1e-3,
            "expected {expected_rot:?}, got {:?}",
            views[1].camera.rotation
        );
        // Sanity-check the hand-derived value too: -R^T * (1, 0, 2).
        assert_vec3_close(views[1].camera.position, glam::vec3(2.0, 0.0, -1.0));

        // Initial point cloud comes from points3D.txt.
        let init = result.init_splat.expect("expected an initial point cloud");
        assert_eq!(init.meta.total_splats, 4);
        assert_eq!(init.data.means.len(), 4 * 3);
        assert_eq!(&init.data.means[0..3], &[1.5, 2.5, 3.5]);
        let sh_coeffs = init.data.sh_coeffs.expect("expected sh coefficients");
        assert_eq!(sh_coeffs.len(), 4 * 3);
        let expected_sh = rgb_to_sh(glam::vec3(1.0, 0.0, 0.0));
        assert_eq!(&sh_coeffs[0..3], expected_sh.to_array().as_slice());
    }

    /// The quantized-PNG prior format gives `depth/img0.png` the image's own
    /// extension, and COLMAP's images.txt stores bare file names, so both files
    /// answer to `img0.png`. Training must use the photograph.
    ///
    /// This loaded *successfully* before the fix -- a depth map is a perfectly
    /// valid PNG -- and reported a spectacular PSNR against maps that are
    /// smooth and trivial to fit, which is why nothing caught it. The dataset
    /// below is exactly the shape that armed it.
    #[tokio::test]
    async fn depth_prior_is_never_loaded_as_the_image() {
        let dir = tempfile::tempdir().unwrap();
        write_test_dataset(dir.path()).await;
        for name in ["img0", "img1", "img2"] {
            write_depth_png(&dir.path().join(format!("depth/{name}.png"))).await;
        }

        let result = load_test_dataset(dir.path(), None).await;
        let views = &result.dataset.train.views;
        assert_eq!(views.len(), 3, "all three views must load");

        // The directory a path sits in is what identifies it here.
        fn parent_dir(path: &Path) -> String {
            path.parent()
                .and_then(Path::file_name)
                .map(|d| d.to_string_lossy().into_owned())
                .unwrap_or_default()
        }

        for view in views.iter() {
            let image_path = view.image.path().to_string_lossy().into_owned();
            assert_eq!(
                parent_dir(view.image.path()),
                "images",
                "trained on '{image_path}' instead of the photograph"
            );

            // Not just the path: the photograph is a flat RGB(10, 20, 30),
            // while the depth map is 16-bit grayscale, so no depth pixel can
            // impersonate it.
            let decoded = view.image.load().await.expect("image decodes");
            assert_eq!(
                decoded.to_rgb8().get_pixel(0, 0),
                &image::Rgb([10u8, 20, 30]),
                "'{image_path}' does not decode to the photograph's pixels"
            );

            // And the depth prior still resolves -- from the very file that
            // must not be the image.
            let depth = view
                .depth
                .as_ref()
                .expect("depth prior must still be discovered")
                .path();
            assert_eq!(parent_dir(depth), "depth", "depth prior at {depth:?}");
        }

        // No spurious ambiguity warning: only `missing.png` is reported.
        assert_eq!(result.warnings.len(), 1, "{:?}", result.warnings);
        assert!(result.warnings[0].contains("missing.png"));
    }

    #[tokio::test]
    async fn splits_eval_views() {
        let dir = tempfile::tempdir().unwrap();
        write_test_dataset(dir.path()).await;
        let result = load_test_dataset(dir.path(), Some(2)).await;

        // Every 2nd view (indices 0 and 2) goes to eval, the rest to train.
        let train = &result.dataset.train.views;
        let eval = &result
            .dataset
            .eval
            .as_ref()
            .expect("expected eval split")
            .views;
        assert_eq!(train.len(), 1);
        assert_eq!(eval.len(), 2);

        // Identify views by their distinct camera positions: img1 (the
        // rotated view) trains, img0 and img2 evaluate.
        let (q, t) = img1_w2c();
        let expected_train_pos = glam::Affine3A::from_rotation_translation(q, t)
            .inverse()
            .translation
            .into();
        assert_vec3_close(train[0].camera.position, expected_train_pos);
        assert_vec3_close(eval[0].camera.position, glam::vec3(-1.0, -2.0, -3.0));
        assert_vec3_close(eval[1].camera.position, glam::Vec3::ZERO);
    }
}
