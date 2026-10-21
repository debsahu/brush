//! Layer B of the walls-first plan, through the real trainer: `--plane-file`,
//! wall tags + projection, the behind-wall prune. (The consensus rule and the
//! file parser are unit-tested next to their code in `brush-train`.)

#![allow(clippy::missing_assert_message)]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_train::{
    config::TrainConfig,
    tidi::plane_coplanarity_loss,
    train::SplatTrainer,
    walls::{WallPlanes, quat_from_normal},
};
use burn::module::Module;
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, Vec3};

/// Plane 0: z = 0, normal +z, 4 x 4 cells of 0.1, cell (3, 3) an OPENING and
/// (0, 3) a CONFLICT. Plane 1: x = 5, normal -x, every cell CLOSED.
const PLANES: &str = r#"{
  "pitch": 0.1, "spacing": 0.01, "tau_a": 0.05, "kappa": 1.5,
  "planes": [
    {"index": 0, "source": "ransac", "normal": [0,0,1], "offset": 0.0,
     "grid_u": [1,0,0], "grid_v": [0,1,0], "a0": 0, "b0": 0, "na": 4, "nb": 4,
     "tiled_cells": 14, "u_min": 0.0, "u_max": 0.4, "v_min": 0.0, "v_max": 0.4,
     "verdict_rle": [[0,3],[1,1],[0,1],[3,1],[0,9],[2,1]]},
    {"index": 1, "source": "ransac", "normal": [-1,0,0], "offset": -5.0,
     "grid_u": [0,1,0], "grid_v": [0,0,1], "a0": 0, "b0": 0, "na": 4, "nb": 4,
     "tiled_cells": 16, "u_min": 0.0, "u_max": 0.4, "v_min": 0.0, "v_max": 0.4,
     "verdict_rle": [[5,16]]}
  ]}"#;

async fn device() -> Device {
    Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

fn walls() -> WallPlanes {
    WallPlanes::from_json(PLANES).expect("planes parse")
}

/// A camera 5 m in front of the origin looking down +z, and a smooth image.
fn batch() -> SceneBatch {
    let (w, h) = (64u32, 64u32);
    let img: Vec<i32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let r = 60 + x * 2;
            let g = 60 + y * 2;
            let b = 140 - (x + y);
            (r | g << 8 | b << 16 | 255 << 24) as i32
        })
        .collect();
    SceneBatch {
        img_packed: TensorData::new(img, [h as usize, w as usize]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        features: None,
        depth: None,
        normal: None,
        camera: Camera::new(
            Vec3::new(0.0, 0.0, -5.0),
            Quat::IDENTITY,
            0.8,
            0.8,
            glam::vec2(0.5, 0.5),
            Pinhole,
        ),
        view_index: 0,
    }
}

/// Splats from `(mean, raw opacity)` rows; round-ish scale 0.15, identity-ish
/// rotation unless given.
fn splats(device: &Device, rows: &[([f32; 3], f32)], rot: [f32; 4]) -> Splats {
    let n = rows.len();
    Splats::from_raw(
        rows.iter().flat_map(|r| r.0).collect(),
        rot.repeat(n),
        vec![0.15f32.ln(); n * 3],
        vec![0.5; n * 3],
        rows.iter().map(|r| r.1).collect(),
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0)
}

fn quiet_config() -> TrainConfig {
    TrainConfig {
        // Barely move anything in the one warm-up step.
        lr_mean: 1e-6,
        lr_mean_end: 1e-6,
        split_at_screen_size: 0.0,
        // No growth and no backfill: only the prune under test can change the count.
        growth_stop_iter: 0,
        stop_replace_iter: 1,
        error_map_densification: false,
        plane_coplanarity_assign_dist: 0.05,
        ..TrainConfig::default()
    }
}

fn trainer(config: &TrainConfig, device: &Device) -> SplatTrainer {
    SplatTrainer::new(
        config,
        device,
        BoundingBox::from_min_max(Vec3::splat(-10.0), Vec3::splat(10.0)),
    )
}

async fn vec(t: Tensor<2>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_into_vec::<f32>()
        .expect("f32")
}

async fn vec1(t: Tensor<1>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_into_vec::<f32>()
        .expect("f32")
}

/// Plan test 20. After ONE trainer step: the off-plane TAGGED splats are on
/// their wall to 1e-6 with `q == q_k`, while an UNTAGGED neighbour (also off the
/// plane) was moved by the step and left where it was.
#[tokio::test]
async fn one_step_projects_tagged_splats_and_only_tagged_splats() {
    let device = device().await;
    let w = walls();
    // 0: tagged plane 0, off by +0.3 in z, rotated away from q_k.
    // 1: UNTAGGED neighbour, off the plane by +0.4 in z.
    // 2: tagged plane 1 (x = 5), off by 0.2 in x.
    let rows = [
        ([0.2, 0.2, 0.3], 0.5),
        ([0.4, 0.2, 0.4], 0.5),
        ([5.2, 0.2, 0.2], 0.5),
    ];
    let rot = [0.7f32, 0.1, 0.5, 0.5];
    let before = splats(&device, &rows, rot).valid();
    let before_means = vec(before.means()).await;

    let mut t = trainer(&TrainConfig::default(), &device);
    t.set_wall_planes(w.clone());
    t.set_wall_tags(vec![0, -1, 1], &device).expect("tags");
    let (after, _) = t.step(batch(), before.clone().train()).await;
    let after = after.valid();
    let means = vec(after.means()).await;
    let quats = vec(after.rotations()).await;

    // Tagged splat 0: z = 0 to 1e-6; its in-plane position is still near where it was.
    assert!(means[2].abs() <= 1e-6, "tagged z = {}", means[2]);
    assert!((means[0] - 0.2).abs() < 0.05 && (means[1] - 0.2).abs() < 0.05);
    // Tagged splat 2: x = 5 on the -x wall.
    assert!((means[6] - 5.0).abs() <= 1e-6, "tagged x = {}", means[6]);
    // q == q_k exactly, and it was NOT q_k before.
    assert_eq!(&quats[0..4], &quat_from_normal([0.0, 0.0, 1.0]));
    assert_eq!(&quats[8..12], &quat_from_normal([-1.0, 0.0, 0.0]));
    assert_ne!(
        rot,
        quat_from_normal([0.0, 0.0, 1.0]),
        "fixture: it started elsewhere"
    );
    // Untagged splat 1: moved by the step, and still 0.4 off the plane.
    let moved: f32 = (0..3)
        .map(|k| (means[3 + k] - before_means[3 + k]).abs())
        .sum();
    assert!(moved > 1e-8, "the untagged neighbour did not move: {moved}");
    assert!((means[5] - 0.4).abs() < 0.05, "untagged z = {}", means[5]);
}

/// A tag change must change the result: with no tags the same step leaves the
/// splat off-plane (a control that the projection, not the step, did it).
#[tokio::test]
async fn without_tags_the_same_step_leaves_the_splat_off_the_plane() {
    let device = device().await;
    let rows = [([0.2, 0.2, 0.3], 0.5), ([0.4, 0.2, 0.4], 0.5)];
    let before = splats(&device, &rows, [0.7, 0.1, 0.5, 0.5]).valid();
    let mut t = trainer(&TrainConfig::default(), &device);
    t.set_wall_planes(walls());
    let (after, _) = t.step(batch(), before.train()).await;
    let means = vec(after.valid().means()).await;
    assert!((means[2] - 0.3).abs() < 0.05, "z = {}", means[2]);
}

/// Splats whose raw opacities are all distinct so a split can be traced.
fn traceable(device: &Device, n: usize) -> Splats {
    let rows: Vec<([f32; 3], f32)> = (0..n)
        .map(|i| {
            (
                [-0.5 + 0.2 * i as f32, 0.1, 0.2],
                0.2 + 0.1 * i as f32, // sigmoid in 0.55..0.8, far above min_opacity
            )
        })
        .collect();
    splats(device, &rows, [1.0, 0.0, 0.0, 0.0])
}

/// A tagged splat's children keep its tag; tags stay aligned to the splats.
#[tokio::test]
async fn children_inherit_the_parent_wall_tag() {
    let device = device().await;
    let n = 6;
    let tags_before = vec![0, -1, 1, -1, 0, 1];
    let config = TrainConfig {
        growth_grad_threshold: 0.0,
        growth_select_fraction: 1.0,
        growth_stop_iter: 5000,
        error_map_densification: false,
        split_at_screen_size: 0.0,
        ..quiet_config()
    };
    let mut t = trainer(&config, &device);
    t.set_wall_planes(walls());
    t.set_wall_tags(tags_before.clone(), &device).expect("tags");
    let mut s = traceable(&device, n).valid();
    for _ in 0..3 {
        let (next, _) = t.step(batch(), s.train()).await;
        s = next.valid();
    }
    let ops_before = vec1(s.opacities()).await;
    let (s, stats) = t.refine(100, s).await;
    assert!(stats.num_added > 0, "the fixture must split something");

    let tags = t.wall_tags_host().await.expect("tags");
    let total = s.num_splats() as usize;
    assert_eq!(tags.len(), total, "tags stay aligned to the splats");
    assert_eq!(total, n + stats.num_added as usize);
    assert_eq!(&tags[..n], &tags_before[..], "parents keep their tags");

    // Trace each child to its parent: a split gives parent and child the same
    // new opacity, and every opacity here started distinct.
    let ops = vec1(s.opacities()).await;
    let mut checked_tagged_child = false;
    for c in n..total {
        let parent = (0..n)
            .find(|&p| (ops[p] - ops[c]).abs() < 1e-7 && (ops[p] - ops_before[p]).abs() > 1e-4)
            .unwrap_or_else(|| panic!("child {c} matches no split parent"));
        assert_eq!(tags[c], tags_before[parent], "child {c} of parent {parent}");
        checked_tagged_child |= tags_before[parent] >= 0;
    }
    assert!(
        checked_tagged_child,
        "no tagged splat was split: the test saw nothing"
    );
}

/// Pruning one splat re-indexes the tags with it.
#[tokio::test]
async fn pruning_reindexes_the_wall_tags() {
    let device = device().await;
    let n = 6;
    let tags_before = vec![0, -1, 1, 0, -1, 1];
    let mut t = trainer(&quiet_config(), &device);
    t.set_wall_planes(walls());
    t.set_wall_tags(tags_before.clone(), &device).expect("tags");
    // Splat 2 is (almost) transparent: pruned by the min-opacity rule.
    let mut rows: Vec<([f32; 3], f32)> = (0..n)
        .map(|i| ([-0.5 + 0.2 * i as f32, 0.1, 0.2], 1.0))
        .collect();
    rows[2].1 = -12.0;
    let s = splats(&device, &rows, [1.0, 0.0, 0.0, 0.0]).valid();
    let (s, _) = t.step(batch(), s.train()).await;
    let (s, stats) = t.refine(100, s.valid()).await;
    assert_eq!(stats.num_pruned, 1);
    assert_eq!(s.num_splats() as usize, n - 1);
    assert_eq!(
        t.wall_tags_host().await.expect("tags"),
        [0, -1, 0, -1, 1],
        "the tag of the pruned splat left with it"
    );
}

/// Plan test 21 through the trainer. One tiled wall (plane 0, z = 0, +z), one
/// splat in each situation; only the one behind a tiled cell goes, and only
/// with the flag on.
#[tokio::test]
async fn behind_wall_prune_removes_exactly_the_splat_behind_a_tiled_cell() {
    let device = device().await;
    // opacity encodes identity so survivors can be read back.
    let situations: [(&str, [f32; 3], i32); 5] = [
        ("front", [0.15, 0.15, 0.5], -1),
        ("band", [0.15, 0.15, -0.01], -1),
        ("behind", [0.15, 0.15, -0.5], -1),
        ("behind OPENING", [0.35, 0.35, -0.5], -1),
        ("behind but tagged", [0.25, 0.15, -0.5], 0),
    ];
    let rows: Vec<([f32; 3], f32)> = situations
        .iter()
        .enumerate()
        .map(|(i, s)| (s.1, 1.0 + 0.2 * i as f32))
        .collect();
    let tags: Vec<i32> = situations.iter().map(|s| s.2).collect();

    let mut survivors = Vec::new();
    for enabled in [false, true] {
        let config = TrainConfig {
            behind_wall_prune: enabled,
            plane_file: Some("unused".into()),
            ..quiet_config()
        };
        let mut t = trainer(&config, &device);
        t.set_wall_planes(walls());
        t.set_wall_tags(tags.clone(), &device).expect("tags");
        let s = splats(&device, &rows, [1.0, 0.0, 0.0, 0.0]).valid();
        let (s, _) = t.step(batch(), s.train()).await;
        let (s, stats) = t.refine(100, s.valid()).await;
        // refine also decays every opacity by ~0.004, hence the tolerance;
        // the five opacities are >= 0.03 apart.
        let ops = vec1(s.opacities()).await;
        let alive: Vec<usize> = (0..situations.len())
            .filter(|&i| ops.iter().any(|o| (o - sigmoid(rows[i].1)).abs() < 0.01))
            .collect();
        survivors.push((enabled, stats.num_pruned, alive));
    }
    // Flag off: nothing is pruned (the control).
    assert_eq!(survivors[0], (false, 0, vec![0, 1, 2, 3, 4]));
    // Flag on: exactly situation 2 ("behind") is gone.
    assert_eq!(survivors[1], (true, 1, vec![0, 1, 3, 4]));
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Plan test 22. A file with two planes: `plane_summary` reports 2 although the
/// seed cloud holds THREE orthogonal planes (what RANSAC would find), and the
/// co-planarity term on a 3-splat fixture equals the hand-computed value.
#[tokio::test]
async fn plane_file_replaces_ransac() {
    let device = device().await;
    let config = TrainConfig {
        plane_coplanarity_weight: 20.0,
        plane_coplanarity_assign_dist: 0.05,
        ..TrainConfig::default()
    };
    let mut t = trainer(&config, &device);
    t.set_wall_planes(walls());

    // Three orthogonal 12 x 12 lattices at 0.1 spacing: RANSAC finds 3 planes.
    let mut cloud = Vec::new();
    for a in 0..12 {
        for b in 0..12 {
            let (a, b) = (a as f32 * 0.1, b as f32 * 0.1);
            cloud.extend([a, b, 0.0, 0.0, a, b, a, 0.0, b]);
        }
    }
    let n = cloud.len() / 3;
    let cloud = Tensor::<2>::from_data(TensorData::new(cloud, [n, 3]), &device.clone().inner());
    t.init_plane_priors(cloud, &device.clone().inner()).await;

    let summary = t.plane_summary().expect("planes present");
    assert!(summary.starts_with("2 planes"), "{summary}");

    // Hand-computed: s1 on plane 0 (z = 0.01; scales 0.1,0.1,0.02):
    //   pos 0.01^2 = 1e-4, flatten (0.02)^2 = 4e-4            -> 5e-4
    // s2 on plane 1 (x = 5.03, n = -x; scales 0.01,0.1,0.1):
    //   pos 0.03^2 = 9e-4, flatten (0.01)^2 = 1e-4            -> 1e-3
    // s3 far from both: unassigned. Mean over the 2 assigned: 7.5e-4.
    let dev = device.clone();
    let means = Tensor::<2>::from_data(
        TensorData::new(
            vec![0.2f32, 0.2, 0.01, 5.03, 0.2, 0.2, 0.2, 0.2, 0.5],
            [3, 3],
        ),
        &dev,
    );
    let rots = Tensor::<2>::from_data(
        TensorData::new([1.0f32, 0.0, 0.0, 0.0].repeat(3), [3, 4]),
        &dev,
    );
    let scales = Tensor::<2>::from_data(
        TensorData::new(
            vec![0.1f32, 0.1, 0.02, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1],
            [3, 3],
        ),
        &dev,
    );
    let planes = t.plane_set_ref().expect("plane set");
    let loss = plane_coplanarity_loss(means, rots, scales, planes, 0.05, &dev)
        .expect("term")
        .into_scalar_async::<f32>()
        .await
        .expect("scalar");
    assert!((loss - 7.5e-4).abs() < 1e-7, "loss {loss}");
}

/// The consensus rule through the real trainer: error map -> per-view vote ->
/// prune re-indexing -> candidate set -> growth. One splat is pruned in the
/// same refine so the host counters must be re-indexed (a length mismatch with
/// the post-prune tensors panics). Three arms on the same fixture separate the
/// rules: window-MAX (k = 0) grows, consensus k = 1 grows, consensus k = 100
/// (more votes than the 3 views can cast) grows nothing, so a growth path that
/// quietly ignored `--densify-consensus-k` is caught.
#[tokio::test]
async fn consensus_densification_runs_through_the_trainer() {
    let device = device().await;
    let n = 8u32;
    let mut grown = Vec::new();
    for k in [0u32, 1, 100] {
        let mut rows: Vec<([f32; 3], f32)> = (0..n)
            .map(|i| ([-0.7 + 0.2 * i as f32, 0.1, 0.2], 1.0 + 0.1 * i as f32))
            .collect();
        rows[3].1 = -12.0; // pruned by min-opacity
        let config = TrainConfig {
            error_map_densification: true,
            densify_consensus_k: k,
            densify_consensus_frac: 0.0,
            densify_hit_quantile: 0.5,
            growth_select_fraction: 1.0,
            growth_stop_iter: 5000,
            stop_replace_iter: 1,
            split_at_screen_size: 0.0,
            ..TrainConfig::default()
        };
        config.validate().expect("valid config");
        let mut t = trainer(&config, &device);
        let mut s = splats(&device, &rows, [1.0, 0.0, 0.0, 0.0]).valid();
        for _ in 0..3 {
            let (next, _) = t.step(batch(), s.train()).await;
            s = next.valid();
        }
        let (s, stats) = t.refine(100, s).await;
        assert_eq!(stats.num_pruned, 1, "k = {k}");
        assert_eq!(s.num_splats(), n - 1 + stats.num_added, "k = {k}");
        grown.push(stats.num_split_high_grad);
    }
    assert!(grown[0] > 0, "window-MAX control must grow: {grown:?}");
    assert!(grown[1] > 0, "consensus k=1 must grow: {grown:?}");
    assert_eq!(grown[2], 0, "k=100 cannot be satisfied: {grown:?}");
}

/// Flag validation: the consensus rule needs the error map, and the wall flags
/// need the plane file.
#[test]
fn walls_flags_are_validated() {
    let bad = |c: TrainConfig| c.validate().is_err();
    let base = TrainConfig::default();
    assert!(!bad(base.clone()));
    assert!(bad(TrainConfig {
        densify_consensus_k: 3,
        error_map_densification: false,
        ..base.clone()
    }));
    assert!(bad(TrainConfig {
        behind_wall_prune: true,
        ..base.clone()
    }));
    assert!(bad(TrainConfig {
        wall_tags: Some("t.bin".into()),
        ..base.clone()
    }));
    assert!(!bad(TrainConfig {
        behind_wall_prune: true,
        plane_file: Some("p.json".into()),
        ..base.clone()
    }));
    assert!(bad(TrainConfig {
        densify_consensus_k: 3,
        densify_hit_quantile: 0.0,
        ..base
    }));
}
