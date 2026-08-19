//! Stamps the binary with the exact commit it was built from.
//!
//! Upstream Brush reports a bare `CARGO_PKG_VERSION`, which cannot distinguish
//! stock Brush from this fork, nor one fork build from another. That is not
//! academic: on 2026-08-09 a `target/release/brush-cli` from the previous day
//! was silently missing two flags that had since landed, and the only way to
//! notice was diffing `--help` against the source. A build id makes that
//! impossible to miss.
//!
//! Emits `BRUSH_BUILD_ID`, consumed by `BRUSH_VERSION` in `lib.rs`.

use std::path::PathBuf;
use std::process::Command;

/// Run a git subcommand in `dir`, returning trimmed stdout on success.
fn git(dir: &PathBuf, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    // This crate lives at <root>/apps/brush-cli, so the workspace root is two up.
    let manifest = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo"),
    );
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .map_or_else(|| PathBuf::from("."), PathBuf::from);

    // `--always` falls back to a bare SHA when no tag is reachable; `--dirty`
    // marks a tree with uncommitted changes, which is exactly the case where a
    // binary's provenance is otherwise unknowable.
    let build_id = git(&root, &["describe", "--tags", "--always", "--dirty"])
        .unwrap_or_else(|| "nogit".to_owned());

    println!("cargo:rustc-env=BRUSH_BUILD_ID={build_id}");

    // Re-run when the checked-out commit or the index changes, so the stamp can
    // never lag the source. Resolve the git dir rather than assuming `<root>/.git`
    // is a directory: under a submodule or worktree checkout it is a gitlink file
    // pointing elsewhere, and this repo IS consumed as a submodule.
    if let Some(git_dir) = git(&root, &["rev-parse", "--absolute-git-dir"]) {
        let git_dir = PathBuf::from(git_dir);
        for f in ["HEAD", "index"] {
            let p = git_dir.join(f);
            if p.exists() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
}
