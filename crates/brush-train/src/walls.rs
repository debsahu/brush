//! Wall layer: the builder's `planes.json`, per-splat wall tags, the exact
//! projection onto a tagged splat's wall, and the behind-wall prune.
//!
//! The file is written by `ingest/osmo360/equirect_recipe/walls_first.py`
//! (`planes_json`). Everything here is host-side and pure so it can be tested
//! without a GPU; the trainer wires it to tensors.
//!
//! Conventions (all from the builder):
//! * a plane is `n . x = d` with `n` ORIENTED (unit, pointing to the side its
//!   confirming cameras are on);
//! * the cell grid belongs to the UNSIGNED plane: cell `(a, b)` covers
//!   `u in [(a + a0) h, (a + a0 + 1) h)` with `u = grid_u . x`, likewise `v`,
//!   `h = pitch`. For an oriented copy with a flipped normal `grid_u`/`grid_v`
//!   are unchanged, so `n` may equal `-(grid_u x grid_v)`;
//! * `verdict_rle` is run-length coded over the flat `na * nb` grid, a-major
//!   (`index = a * nb + b`). A cell is TILED iff its code is in
//!   [`TILED_CODES`].

use crate::tidi::{Plane, PlaneSet};
use burn::tensor::{Device, Int, Tensor, TensorData, s};
use serde::Deserialize;
use std::path::Path;

/// Verdict codes of a cell that carries a tile: SUPPORTED (0), FILLED (3),
/// CLOSED (5). CONFLICT (1), OPENING (2) and UNKNOWN (4) do not.
pub const TILED_CODES: [u8; 3] = [0, 3, 5];
const MAX_CODE: u8 = 5;

#[derive(Deserialize)]
struct RawFile {
    pitch: f64,
    spacing: f64,
    tau_a: f64,
    planes: Vec<RawPlane>,
}

#[derive(Deserialize)]
struct RawPlane {
    index: i64,
    normal: [f64; 3],
    offset: f64,
    grid_u: [f64; 3],
    grid_v: [f64; 3],
    a0: i64,
    b0: i64,
    na: usize,
    nb: usize,
    #[serde(default)]
    u_min: Option<f64>,
    #[serde(default)]
    u_max: Option<f64>,
    #[serde(default)]
    v_min: Option<f64>,
    #[serde(default)]
    v_max: Option<f64>,
    verdict_rle: Vec<[i64; 2]>,
}

/// One oriented wall plane plus its tiled-cell mask.
#[derive(Clone, Debug)]
pub struct WallPlane {
    pub normal: [f32; 3],
    pub offset: f32,
    pub grid_u: [f32; 3],
    pub grid_v: [f32; 3],
    pub a0: i64,
    pub b0: i64,
    pub na: usize,
    pub nb: usize,
    /// `na * nb`, a-major.
    pub tiled: Vec<bool>,
    /// Extent of the tiled cells along `grid_u` / `grid_v`; `None` when the
    /// plane has no tiled cell.
    pub extent: Option<[f32; 4]>,
}

impl WallPlane {
    pub fn tiled_cells(&self) -> usize {
        self.tiled.iter().filter(|t| **t).count()
    }

    /// Is the point's projection onto the plane inside a TILED cell?
    pub fn cell_is_tiled(&self, p: [f32; 3], pitch: f32) -> bool {
        let dot = |a: [f32; 3]| a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
        let a = (dot(self.grid_u) / pitch).floor() as i64 - self.a0;
        let b = (dot(self.grid_v) / pitch).floor() as i64 - self.b0;
        if a < 0 || b < 0 || a >= self.na as i64 || b >= self.nb as i64 {
            return false;
        }
        self.tiled[a as usize * self.nb + b as usize]
    }

    /// Signed distance `n . p - d`: positive on the side the normal points to.
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] - self.offset
    }
}

/// The parsed `planes.json`.
#[derive(Clone, Debug)]
pub struct WallPlanes {
    pub pitch: f32,
    /// The seed cloud's NN spacing, as measured by the builder.
    pub spacing: f32,
    /// The builder's own assign band. NOT what the trainer prunes with: the
    /// trainer uses the resolved `--plane-coplanarity-assign-dist`.
    pub tau_a: f32,
    /// In file order; `planes[i]` is the plane a tag value `i` refers to.
    pub planes: Vec<WallPlane>,
}

impl WallPlanes {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let raw: RawFile = serde_json::from_str(text).map_err(|e| format!("planes.json: {e}"))?;
        if !(raw.pitch.is_finite() && raw.pitch > 0.0) {
            return Err(format!(
                "planes.json: pitch must be positive, got {}",
                raw.pitch
            ));
        }
        let mut planes = Vec::with_capacity(raw.planes.len());
        for (pos, rp) in raw.planes.into_iter().enumerate() {
            if rp.index != pos as i64 {
                return Err(format!(
                    "planes.json: plane at position {pos} has index {} (tags refer to the index, \
                     so it must equal the position)",
                    rp.index
                ));
            }
            let norm = rp.normal.iter().map(|x| x * x).sum::<f64>().sqrt();
            if (norm - 1.0).abs() > 1e-3 {
                return Err(format!(
                    "planes.json: plane {pos} normal has norm {norm}, not unit"
                ));
            }
            let cells = rp.na * rp.nb;
            let mut tiled = Vec::with_capacity(cells);
            for [code, len] in &rp.verdict_rle {
                if !(0..=i64::from(MAX_CODE)).contains(code) || *len < 0 {
                    return Err(format!(
                        "planes.json: plane {pos} has verdict run [{code}, {len}]"
                    ));
                }
                let t = TILED_CODES.contains(&(*code as u8));
                tiled.extend(std::iter::repeat_n(t, *len as usize));
                if tiled.len() > cells {
                    break;
                }
            }
            if tiled.len() != cells {
                return Err(format!(
                    "planes.json: plane {pos} verdict_rle covers {} cells, na*nb = {cells}",
                    tiled.len()
                ));
            }
            let f3 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
            let extent = match (rp.u_min, rp.u_max, rp.v_min, rp.v_max) {
                (Some(a), Some(b), Some(c), Some(d)) => {
                    Some([a as f32, b as f32, c as f32, d as f32])
                }
                _ => None,
            };
            planes.push(WallPlane {
                normal: f3(rp.normal),
                offset: rp.offset as f32,
                grid_u: f3(rp.grid_u),
                grid_v: f3(rp.grid_v),
                a0: rp.a0,
                b0: rp.b0,
                na: rp.na,
                nb: rp.nb,
                tiled,
                extent,
            });
        }
        Ok(Self {
            pitch: raw.pitch as f32,
            spacing: raw.spacing as f32,
            tau_a: raw.tau_a as f32,
            planes,
        })
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    /// The planes as the trainer's co-planarity [`PlaneSet`]. Planes with no
    /// tiled cell are skipped: there is no wall there to pull toward. The
    /// bounding box is the tiled extent. `inlier_frac` is repurposed as the
    /// tiled fraction of the grid (the `Plane priors:` line prints it).
    pub fn plane_set(&self) -> PlaneSet {
        let planes = self
            .planes
            .iter()
            .filter_map(|p| {
                let [u_min, u_max, v_min, v_max] = p.extent?;
                (p.tiled_cells() > 0).then(|| Plane {
                    normal: p.normal,
                    offset: p.offset,
                    u_axis: p.grid_u,
                    v_axis: p.grid_v,
                    u_min,
                    u_max,
                    v_min,
                    v_max,
                    inlier_frac: p.tiled_cells() as f32 / p.tiled.len().max(1) as f32,
                })
            })
            .collect();
        PlaneSet {
            planes,
            spacing: self.spacing,
            threshold: self.tau_a,
        }
    }

    /// `q_k` for every plane, in file order.
    pub fn quats(&self) -> Vec<[f32; 4]> {
        self.planes
            .iter()
            .map(|p| quat_from_normal(p.normal))
            .collect()
    }
}

/// `(w, x, y, z)` rotating `e_z` onto `n`. The same formula as
/// `walls_first.quat_from_normal`, poles included: `n` within `1e-6` of the
/// z axis gives identity for `+z` and a half-turn about x for `-z`.
pub fn quat_from_normal(n: [f32; 3]) -> [f32; 4] {
    let n = n.map(f64::from);
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let n = [n[0] / len, n[1] / len, n[2] / len];
    // e_z x n
    let ax = [-n[1], n[0], 0.0];
    let s = ax[0].hypot(ax[1]);
    let c = n[2];
    let q = if s >= 1e-6 {
        let th = s.atan2(c);
        let k = (th / 2.0).sin() / s;
        [(th / 2.0).cos(), ax[0] * k, ax[1] * k, 0.0]
    } else if c > 0.0 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0, 0.0]
    };
    q.map(|v| v as f32)
}

/// Read `init.tags.bin`: little-endian `i32` per init row, `-1` for non-wall.
pub fn read_wall_tags(path: &Path) -> Result<Vec<i32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of i32",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Which splats the behind-wall prune removes (plan 7.5).
///
/// `means` is `[N * 3]` row-major, `tags` is `[N]`. A splat is pruned iff it
/// is UNTAGGED and there is a plane `k` for which
/// `n_k . mu - d_k < -tau_a` (strictly behind the oriented wall by more than
/// the assign band) AND its `(a, b)` cell on that plane is tiled. So a splat
/// in an OPENING cell, or past the edge of the grid, is never pruned, and a
/// splat inside a thin partition tiled from both sides is behind both walls
/// and is.
pub fn behind_wall_mask(means: &[f32], tags: &[i32], walls: &WallPlanes, tau_a: f32) -> Vec<bool> {
    let n = means.len() / 3;
    assert_eq!(tags.len(), n, "one tag per splat");
    (0..n)
        .map(|i| {
            if tags[i] >= 0 {
                return false;
            }
            let p = [means[i * 3], means[i * 3 + 1], means[i * 3 + 2]];
            walls
                .planes
                .iter()
                .any(|pl| pl.signed_distance(p) < -tau_a && pl.cell_is_tiled(p, walls.pitch))
        })
        .collect()
}

/// The per-plane constants of the projection, on the optimiser (inner) device.
pub(crate) struct WallTables {
    /// `[P, 3]` oriented unit normals.
    normals: Tensor<2>,
    /// `[P]` offsets `d` in `n . x = d`.
    offsets: Tensor<1>,
    /// `[P, 4]` `q_k` per plane, `(w, x, y, z)`.
    quats: Tensor<2>,
}

impl WallTables {
    pub(crate) fn new(walls: &WallPlanes, device: &Device) -> Self {
        let p = walls.planes.len();
        let normals: Vec<f32> = walls.planes.iter().flat_map(|pl| pl.normal).collect();
        let offsets: Vec<f32> = walls.planes.iter().map(|pl| pl.offset).collect();
        let quats: Vec<f32> = walls.quats().into_iter().flatten().collect();
        Self {
            normals: Tensor::from_data(TensorData::new(normals, [p, 3]), device),
            offsets: Tensor::from_data(TensorData::new(offsets, [p]), device),
            quats: Tensor::from_data(TensorData::new(quats, [p, 4]), device),
        }
    }
}

/// The exact Euclidean projection of the TAGGED rows of `transforms`
/// (`[N, 10]` = mean 3 | quat 4 | log-scale 3) onto their wall:
/// `mu <- mu - n (n . mu - d)` and `q <- q_k`. Untagged rows (`tag < 0`) are
/// returned bit for bit; the in-plane position, scales, opacity and colour of a
/// tagged row are untouched (the projection moves it along `n` only).
pub(crate) fn project_to_walls(
    transforms: Tensor<2>,
    tags: &Tensor<1, Int>,
    tables: &WallTables,
) -> Tensor<2> {
    let n = transforms.dims()[0];
    // Boolean per row, then broadcast across the 3 / 4 columns it gates.
    let tagged = tags.clone().greater_equal_elem(0).reshape([n, 1]);
    let idx = tags.clone().clamp_min(0);
    let nrm = tables.normals.clone().select(0, idx.clone()); // [N, 3]
    let off = tables
        .offsets
        .clone()
        .select(0, idx.clone())
        .reshape([n, 1]);
    let qk = tables.quats.clone().select(0, idx); // [N, 4]

    let means = transforms.clone().slice(s![.., 0..3]);
    let quats = transforms.clone().slice(s![.., 3..7]);
    let scales = transforms.slice(s![.., 7..10]);

    let dist = (means.clone() * nrm.clone()).sum_dim(1) - off; // [N, 1]
    let projected = means.clone() - nrm * dist;
    let means = means.mask_where(tagged.clone().expand([n, 3]), projected);
    let quats = quats.mask_where(tagged.expand([n, 4]), qk);
    Tensor::cat(vec![means, quats, scales], 1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    /// Two planes, 4 x 4 cells of 0.1 m. Plane 0: z = 0, normal +z, cell
    /// (3, 3) is an OPENING (code 2) and cell (0, 3) CONFLICT; the rest are
    /// SUPPORTED(0)/FILLED(3). Plane 1: x = 5, normal -x (flipped: its
    /// grid basis is (y, z) so n = -(u x v)), every cell CLOSED (5).
    pub(crate) fn two_plane_json() -> String {
        // plane 0 flat grid a-major index a*4+b: all 0 except (0,3)=1 (idx 3),
        // (3,3)=2 (idx 15), (1,1)=3 FILLED (idx 5).
        r#"{
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
          ]}"#
        .to_owned()
    }

    #[test]
    fn parses_planes_and_expands_the_verdict_rle() {
        let w = WallPlanes::from_json(&two_plane_json()).expect("parse");
        assert_eq!(w.planes.len(), 2);
        assert!((w.pitch - 0.1).abs() < 1e-9 && (w.spacing - 0.01).abs() < 1e-9);
        let p0 = &w.planes[0];
        assert_eq!(p0.tiled.len(), 16);
        // idx 3 CONFLICT and idx 15 OPENING are not tiled; 5 FILLED is.
        let untiled: Vec<usize> = (0..16).filter(|i| !p0.tiled[*i]).collect();
        assert_eq!(untiled, [3, 15]);
        assert_eq!(p0.tiled_cells(), 14);
        assert_eq!(w.planes[1].tiled_cells(), 16);
    }

    #[test]
    fn rejects_malformed_files() {
        let good = two_plane_json();
        // rle too short
        let bad = good.replace("[[5,16]]", "[[5,15]]");
        assert!(
            WallPlanes::from_json(&bad)
                .unwrap_err()
                .contains("covers 15 cells")
        );
        // index != position
        let bad = good.replace("\"index\": 1", "\"index\": 7");
        assert!(WallPlanes::from_json(&bad).unwrap_err().contains("index 7"));
        // unknown verdict code
        let bad = good.replace("[[5,16]]", "[[9,16]]");
        assert!(WallPlanes::from_json(&bad).is_err());
        // non-unit normal
        let bad = good.replace("[-1,0,0]", "[-2,0,0]");
        assert!(
            WallPlanes::from_json(&bad)
                .unwrap_err()
                .contains("not unit")
        );
    }

    #[test]
    fn plane_set_skips_planes_without_tiles() {
        let w = WallPlanes::from_json(&two_plane_json()).unwrap();
        assert_eq!(w.plane_set().planes.len(), 2);
        // Third plane with no tiles and null extents.
        let base = two_plane_json();
        let base = base
            .trim_end()
            .strip_suffix("]}")
            .expect("file ends with the planes list");
        let json = format!(
            "{base},{{\"index\": 2, \"source\": \"x\", \"normal\": [0,1,0], \"offset\": 1.0,
               \"grid_u\": [1,0,0], \"grid_v\": [0,0,1], \"a0\": 0, \"b0\": 0, \"na\": 2, \"nb\": 2,
               \"tiled_cells\": 0, \"u_min\": null, \"u_max\": null, \"v_min\": null, \"v_max\": null,
               \"verdict_rle\": [[2,2],[4,2]]}}]}}"
        );
        let w = WallPlanes::from_json(&json).expect("3-plane file");
        assert_eq!(w.planes.len(), 3, "tags still index all three");
        assert_eq!(
            w.plane_set().planes.len(),
            2,
            "but the tile-less one is no wall"
        );
        let ps = w.plane_set();
        assert_eq!(ps.planes[1].normal, [-1.0, 0.0, 0.0]);
        assert_eq!(ps.planes[1].u_axis, [0.0, 1.0, 0.0]);
        assert_eq!((ps.planes[0].u_min, ps.planes[0].u_max), (0.0, 0.4));
        assert!((ps.spacing - 0.01).abs() < 1e-9);
    }

    fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
        let [w, x, y, z] = q;
        [
            (1.0 - 2.0 * (y * y + z * z)) * v[0]
                + 2.0 * (x * y - z * w) * v[1]
                + 2.0 * (x * z + y * w) * v[2],
            2.0 * (x * y + z * w) * v[0]
                + (1.0 - 2.0 * (x * x + z * z)) * v[1]
                + 2.0 * (y * z - x * w) * v[2],
            2.0 * (x * z - y * w) * v[0]
                + 2.0 * (y * z + x * w) * v[1]
                + (1.0 - 2.0 * (x * x + y * y)) * v[2],
        ]
    }

    /// Values from a pure-python transcription of `walls_first.quat_from_normal`
    /// (numpy was not importable in that session), pinning the (w, x, y, z)
    /// order and both poles: an xyzw writer would fail every row but the poles.
    #[test]
    fn quat_from_normal_matches_the_builder() {
        let cases: [([f32; 3], [f32; 4]); 7] = [
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2, 0.0]),
            ([0.0, 1.0, 0.0], [FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0, 0.0]),
            ([0.6, 0.0, 0.8], [0.948_683_3, 0.0, 0.316_227_77, 0.0]),
            (
                [-0.48, 0.64, 0.6],
                [0.894_427_2, -0.357_770_9, -0.268_328_17, 0.0],
            ),
            ([0.0, -0.6, -0.8], [0.316_227_77, 0.948_683_3, 0.0, 0.0]),
        ];
        for (n, want) in cases {
            let got = quat_from_normal(n);
            for k in 0..4 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-6,
                    "n={n:?} got {got:?} want {want:?}"
                );
            }
            // And the defining property: R(q) e_z = n.
            let r = rotate(got, [0.0, 0.0, 1.0]);
            for k in 0..3 {
                assert!((r[k] - n[k]).abs() < 1e-6, "n={n:?} R e_z = {r:?}");
            }
        }
    }

    /// One tiled plane z = 0 (normal +z), cell (3, 3) an OPENING, `tau_a` 0.05.
    #[test]
    fn behind_wall_prunes_exactly_the_splat_behind_a_tiled_cell() {
        let w = WallPlanes::from_json(&two_plane_json()).unwrap();
        // Only plane 0 matters for these points (x is far from 5).
        let pts: [([f32; 3], i32, bool, &str); 7] = [
            ([0.15, 0.15, 0.5], -1, false, "in front"),
            ([0.15, 0.15, -0.01], -1, false, "inside the band"),
            ([0.15, 0.15, -0.5], -1, true, "behind a tiled cell"),
            ([0.35, 0.35, -0.5], -1, false, "behind an OPENING cell"),
            ([5.0, 5.0, -0.5], -1, false, "behind but past the grid"),
            ([0.15, 0.15, -0.5], 0, false, "behind but TAGGED (a tile)"),
            (
                [0.05, 0.35, -0.5],
                -1,
                false,
                "behind a CONFLICT cell (0,3)",
            ),
        ];
        let means: Vec<f32> = pts.iter().flat_map(|p| p.0).collect();
        let tags: Vec<i32> = pts.iter().map(|p| p.1).collect();
        let mask = behind_wall_mask(&means, &tags, &w, 0.05);
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(mask[i], p.2, "{}", p.3);
        }
        assert_eq!(mask.iter().filter(|m| **m).count(), 1);
    }

    /// Plane 1 has normal -x at x = 5 (so "behind" is x > 5) and a grid basis
    /// whose cross product is +x, i.e. n = -(u x v): the orientation lives in
    /// the normal, not in the basis.
    #[test]
    fn behind_wall_respects_an_oriented_plane_with_a_flipped_normal() {
        let w = WallPlanes::from_json(&two_plane_json()).unwrap();
        let pts = [
            ([4.5f32, 0.15, 0.15], false), // in front of the wall (x < 5)
            ([5.5, 0.15, 0.15], true),     // behind it
            ([5.5, 0.15, 3.0], false),     // behind but past the grid
        ];
        let means: Vec<f32> = pts.iter().flat_map(|p| p.0).collect();
        let mask = behind_wall_mask(&means, &[-1, -1, -1], &w, 0.05);
        assert_eq!(mask, [false, true, false]);
    }

    #[test]
    fn reads_tags_little_endian() {
        let dir = std::env::temp_dir().join(format!("walls_tags_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("init.tags.bin");
        let mut bytes = Vec::new();
        for v in [-1_i32, 0, 3, 258] {
            bytes.extend(v.to_le_bytes());
        }
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(read_wall_tags(&p).unwrap(), [-1, 0, 3, 258]);
        std::fs::write(&p, &bytes[..7]).unwrap();
        assert!(read_wall_tags(&p).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    async fn dev() -> Device {
        Device::from(brush_cube::test_helpers::test_device().await)
    }

    async fn host2(t: Tensor<2>) -> Vec<f32> {
        t.into_data_async()
            .await
            .unwrap()
            .try_into_vec::<f32>()
            .unwrap()
    }

    /// The projection on a hand-built `[N, 10]` table. Rows: 0 tagged plane 0
    /// (z = 0, +z) off by 0.3; 1 untagged off by 0.4; 2 tagged plane 1 (x = 5,
    /// n = -x) off by 0.2; 3 tagged plane 0 already on the wall.
    #[tokio::test]
    async fn projection_moves_tagged_rows_along_the_normal_and_nothing_else() {
        let device = dev().await;
        let w = WallPlanes::from_json(&two_plane_json()).unwrap();
        let rot = [0.7_f32, 0.1, 0.5, 0.5];
        let rows: [([f32; 3], i32); 4] = [
            ([0.2, 0.3, 0.3], 0),
            ([0.4, 0.2, 0.4], -1),
            ([5.2, 0.7, 0.9], 1),
            ([0.1, 0.1, 0.0], 0),
        ];
        let mut data = Vec::new();
        for (m, _) in &rows {
            data.extend(m);
            data.extend(rot);
            data.extend([-1.5_f32, -2.0, -3.5]);
        }
        let tr = Tensor::<2>::from_data(TensorData::new(data.clone(), [4, 10]), &device);
        let tags = Tensor::<1, Int>::from_data(
            TensorData::new(rows.iter().map(|r| r.1).collect::<Vec<_>>(), [4]),
            &device,
        );
        let out = host2(project_to_walls(tr, &tags, &WallTables::new(&w, &device))).await;
        let row = |i: usize| &out[i * 10..i * 10 + 10];

        // Tagged row 0: z -> 0, in-plane coordinates and scales untouched, q = q_k.
        assert!(row(0)[2].abs() <= 1e-6);
        assert_eq!(&row(0)[0..2], &[0.2, 0.3]);
        assert_eq!(&row(0)[3..7], &quat_from_normal([0.0, 0.0, 1.0]));
        assert_eq!(&row(0)[7..10], &[-1.5, -2.0, -3.5]);
        // Tagged row 2 on the flipped wall: x -> 5.
        assert!((row(2)[0] - 5.0).abs() <= 1e-6);
        assert_eq!(&row(2)[1..3], &[0.7, 0.9]);
        assert_eq!(&row(2)[3..7], &quat_from_normal([-1.0, 0.0, 0.0]));
        // Row 3 was already on the wall: position unchanged, quaternion still set.
        assert_eq!(&row(3)[0..3], &[0.1, 0.1, 0.0]);
        assert_eq!(&row(3)[3..7], &quat_from_normal([0.0, 0.0, 1.0]));
        // Untagged row 1: bit for bit the input.
        assert_eq!(row(1), &data[10..20]);
        // The fixture can tell the cases apart: rows 0 and 1 started equally far
        // from "their" plane, and only one of them is a wall tile.
        assert_ne!(&data[3..7], &quat_from_normal([0.0, 0.0, 1.0]));
    }

    /// Real builder output; the path is the one the walls-first run writes.
    #[test]
    #[ignore = "needs work/spotlight_dexter_home/walls_first/scene/planes.json"]
    fn parses_the_real_planes_json() {
        let p = Path::new(
            "/Users/debsahu/Workspace/slam/work/spotlight_dexter_home/walls_first/scene/planes.json",
        );
        let w = WallPlanes::from_path(p).expect("real planes.json parses");
        assert!(!w.planes.is_empty());
        assert!(w.planes.iter().any(|pl| pl.tiled_cells() > 0));
        assert!(w.plane_set().planes.len() <= w.planes.len());
    }
}
