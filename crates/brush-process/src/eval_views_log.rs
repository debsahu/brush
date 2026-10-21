//! `BRUSH_EVAL_VIEWS_LOG`: one JSON line per eval view, appended and flushed
//! as each view is scored, so a run is pollable with `tail -f` and a crash
//! keeps every view that finished. Unset = nothing is opened, nothing written.

use brush_train::eval::EvalViewRecord;
use std::{fs::File, io::Write as _, path::Path};

/// The environment variable that names the file.
pub const ENV_VAR: &str = "BRUSH_EVAL_VIEWS_LOG";

/// An open eval-views log. Construct with [`EvalViewsLog::open`]; `None`
/// (variable unset, or file not openable) means "do not log".
pub struct EvalViewsLog {
    file: File,
}

impl EvalViewsLog {
    /// Open `path` for append. An unopenable path logs a warning and yields
    /// `None`: a missing diagnostic must not abort a training run.
    pub fn open(path: Option<&Path>) -> Option<Self> {
        let path = path?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(file) => Some(Self { file }),
            Err(error) => {
                log::warn!("{ENV_VAR} open failed ({}): {error}", path.display());
                None
            }
        }
    }

    /// Open the file named by `BRUSH_EVAL_VIEWS_LOG`, if set.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var_os(ENV_VAR).map(std::path::PathBuf::from);
        Self::open(path.as_deref())
    }

    /// Append one line and flush it. Failures are warnings, not errors.
    pub fn append(&mut self, record: &EvalViewRecord) {
        let line = record_json_line(record);
        if let Err(error) = writeln!(self.file, "{line}").and_then(|()| self.file.flush()) {
            log::warn!("{ENV_VAR} write failed: {error}");
        }
    }
}

/// The record as one JSON object. Non-finite numbers become `null` (an
/// undefined statistic, e.g. alpha over a fully-masked view, must not parse as
/// a number). `ssim_unmasked` is carried so the printed line can be derived
/// from the records alone.
pub fn record_json_line(record: &EvalViewRecord) -> String {
    serde_json::json!({
        "iter": record.iter,
        "image": record.image,
        "psnr_masked": record.psnr_masked,
        "ssim_masked": record.ssim_masked,
        "psnr_unmasked": record.psnr_unmasked,
        "ssim_unmasked": record.ssim_unmasked,
        "coverage": record.coverage,
        "alpha_mean": record.alpha_mean,
        "alpha_lt_half": record.alpha_lt_half,
    })
    .to_string()
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use brush_train::eval::eval_summary_line;

    fn rec(image: &str, psnr_masked: f32) -> EvalViewRecord {
        EvalViewRecord {
            iter: 2500,
            image: image.to_owned(),
            psnr_masked,
            ssim_masked: 0.875,
            psnr_unmasked: psnr_masked - 5.0,
            ssim_unmasked: 0.5,
            coverage: 0.75,
            alpha_mean: 0.625,
            alpha_lt_half: 0.125,
        }
    }

    #[test]
    fn unset_variable_opens_nothing() {
        assert!(EvalViewsLog::open(None).is_none());
    }

    #[test]
    fn writes_one_flushed_json_line_per_view_with_every_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eval_views.jsonl");
        let mut log = EvalViewsLog::open(Some(&path)).expect("open");
        log.append(&rec("images/a_front.png", 30.0));

        // Flushed per line: readable while the log is still open.
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 1);
        let v: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        let obj = v.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "alpha_lt_half",
                "alpha_mean",
                "coverage",
                "image",
                "iter",
                "psnr_masked",
                "psnr_unmasked",
                "ssim_masked",
                "ssim_unmasked"
            ]
        );
        assert_eq!(v["iter"], 2500);
        assert_eq!(v["image"], "images/a_front.png");
        assert_eq!(v["psnr_masked"], 30.0);
        assert_eq!(v["psnr_unmasked"], 25.0);
        assert_eq!(v["coverage"], 0.75);
        assert_eq!(v["alpha_mean"], 0.625);
        assert_eq!(v["alpha_lt_half"], 0.125);

        // A second open (the next eval) appends rather than truncating.
        drop(log);
        let mut log = EvalViewsLog::open(Some(&path)).expect("reopen");
        log.append(&rec("images/b_front.png", 32.0));
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 2);
    }

    #[test]
    fn undefined_statistics_serialise_as_null_not_a_number() {
        let mut r = rec("a.png", 30.0);
        r.alpha_mean = f32::NAN;
        r.alpha_lt_half = f32::NAN;
        let v: serde_json::Value = serde_json::from_str(&record_json_line(&r)).unwrap();
        assert!(v["alpha_mean"].is_null() && v["alpha_lt_half"].is_null());
        assert!(v["psnr_masked"].is_number());
    }

    /// The claim the Python scorer relies on: the mean of the JSONL
    /// `psnr_masked` / `coverage` IS the printed mean, because the printed line
    /// is formed from the very records that were written.
    #[test]
    fn mean_of_logged_psnr_masked_reproduces_the_printed_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eval_views.jsonl");
        let mut log = EvalViewsLog::open(Some(&path)).unwrap();
        let records = [rec("a.png", 30.0), rec("b.png", 33.5), rec("c.png", 41.0)];
        for r in &records {
            log.append(r);
        }
        let parsed: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(parsed.len(), 3);
        let mean = |key: &str| {
            parsed.iter().map(|v| v[key].as_f64().unwrap()).sum::<f64>() / parsed.len() as f64
        };

        let line = eval_summary_line(2500, &records);
        let printed_psnr: f64 = line
            .split("masked PSNR ")
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        // 3 decimals printed.
        assert!((mean("psnr_masked") - printed_psnr).abs() < 5e-4, "{line}");
        assert!((mean("psnr_masked") - 34.8333).abs() < 1e-3);
        assert!(line.ends_with(&format!("mean coverage {:.1}%", mean("coverage") * 100.0)));
    }
}
