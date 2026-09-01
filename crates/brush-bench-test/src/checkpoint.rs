//! Resolving the exported PLY that a checkpoint binary was pointed at.

use anyhow::{Result, bail};
use brush_vfs::BrushVfs;
use std::path::PathBuf;

/// Pick the exported Brush PLY inside a mounted checkpoint, or refuse.
///
/// Both checkpoint binaries used to do this with `vfs.file_paths().next()`.
/// `file_paths` iterates the VFS `HashMap`, and Rust randomizes hash-map
/// iteration per process, so pointing `--ply` at a *directory* -- which
/// `BrushVfs::from_path` accepts and walks -- scored a different checkpoint on
/// different runs of the identical command. Measured on the four-file fixture
/// in the tests below, before this function existed: 500 lookups split
/// 147/131/125 across the three exports, and the remaining **97 returned
/// `args.txt`**, which was then handed to `load_splat_from_ply`.
///
/// So there were two faults in that one expression, and the file-type one made
/// the other easy to miss: a stray non-PLY produced a parse error naming a file
/// nobody asked to load, while three plausible exports produced no error at
/// all -- just a benchmark or an eval number for a checkpoint the operator did
/// not choose. This is a real shape for us: a Brush export directory holds
/// `export_10000.ply` through `export_30000.ply`, and those files are not even
/// distinguishable by size (five plys on the M4 Max are byte-for-byte the same
/// length, all pinned at the 3M splat cap).
///
/// Candidates are therefore restricted to `.ply`, and **more than one is an
/// error naming every candidate** -- the shape `find_prior_path` uses in the
/// dataset loader, and the right one here: two exports of one run are
/// genuinely different things, nothing structural ranks iteration 10,000 above
/// 30,000, and either pick may be flatly wrong. Nothing that works today
/// breaks, because such a tree already picks at random. A single `.ply`
/// argument -- the normal invocation -- mounts a one-entry VFS and is
/// unambiguous.
pub fn select_checkpoint_ply(vfs: &BrushVfs) -> Result<PathBuf> {
    // Sorted so both the choice and the error message are stable whatever
    // order the VFS walked in.
    let mut plys: Vec<PathBuf> = vfs.files_with_extension("ply").collect();
    plys.sort();

    match plys.len() {
        0 => bail!(
            "checkpoint contained no .ply file (found {} file(s))",
            vfs.file_count()
        ),
        1 => Ok(plys.remove(0)),
        n => bail!(
            "checkpoint contains {n} .ply files and nothing distinguishes them: {}. \
             Brush used to pick one by hash-map iteration order, which differs between runs \
             of the identical command -- so the same invocation would score one export today \
             and another tomorrow, with nothing changed. Pass the single .ply you mean.",
            plys.iter()
                .map(|p| format!("'{}'", p.display()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::select_checkpoint_ply;
    use brush_vfs::BrushVfs;
    use std::path::PathBuf;

    /// Each iteration builds a fresh `BrushVfs`, hence a fresh `HashMap` with a
    /// fresh `RandomState`, so the walk order genuinely differs run to run --
    /// that is what the 147/131/125/97 split in the doc comment was measured
    /// with. At 500 repeats an order-dependent selection is not going to slip
    /// through.
    const ORDER_PROBE_RUNS: usize = 500;

    fn vfs(paths: &[&str]) -> BrushVfs {
        BrushVfs::create_test_vfs(paths.iter().map(PathBuf::from).collect())
    }

    /// The normal invocation: one export, with the sidecars a Brush export
    /// directory carries next to it. It must resolve to the ply every run, and
    /// never to `args.txt` -- which the old expression returned 97 times in 500.
    #[test]
    fn a_single_export_resolves_past_its_sidecars() {
        for _ in 0..ORDER_PROBE_RUNS {
            let chosen = select_checkpoint_ply(&vfs(&[
                "args.txt",
                "export_30000.ply",
                "export_30000_dig_mlp.json",
            ]))
            .expect("one ply must resolve");
            assert_eq!(chosen, PathBuf::from("export_30000.ply"));
        }
    }

    /// Several exports of one run. Nothing ranks them, so this must refuse --
    /// and name every candidate, with the same message every run, since an
    /// error chosen by iteration order would be no better than a checkpoint
    /// chosen that way.
    #[test]
    fn several_exports_are_fatal_and_name_every_candidate() {
        let mut messages = std::collections::BTreeSet::new();
        for _ in 0..ORDER_PROBE_RUNS {
            let err = select_checkpoint_ply(&vfs(&[
                "export_10000.ply",
                "export_20000.ply",
                "export_30000.ply",
                "args.txt",
            ]))
            .expect_err("an ambiguous checkpoint must be fatal");
            messages.insert(err.to_string());
        }
        assert_eq!(messages.len(), 1, "the message must not vary: {messages:?}");

        let message = messages.iter().next().expect("one message");
        for name in ["export_10000.ply", "export_20000.ply", "export_30000.ply"] {
            assert!(message.contains(name), "must name {name}: {message}");
        }
    }

    /// Null model for both tests above: a checkpoint holding no ply at all is
    /// an error rather than some other file being loaded as one.
    #[test]
    fn a_checkpoint_without_a_ply_is_an_error() {
        assert!(select_checkpoint_ply(&vfs(&["args.txt", "meta.json"])).is_err());
    }
}
