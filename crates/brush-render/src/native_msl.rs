//! Shared runtime switches for experimental native-MSL optimizations.

#[cfg(not(target_family = "wasm"))]
use std::ffi::OsStr;

pub const PRESET_ENV: &str = "BRUSH_NATIVE_MSL_PRESET";
pub const UNCHECKED_RASTER_BWD_ENV: &str = "BRUSH_NATIVE_MSL_UNCHECKED_RASTER_BWD";
pub const FUSED_SH_ADAM_ENV: &str = "BRUSH_NATIVE_MSL_FUSED_SH_ADAM";
pub const COALESCED_SH_GRAD_ENV: &str = "BRUSH_NATIVE_MSL_COALESCED_SH_GRAD";
pub const SAVED_LOSS_PARTIALS_ENV: &str = "BRUSH_NATIVE_MSL_SAVED_LOSS_PARTIALS";
pub const SPARSE_SH_ADAM_ENV: &str = "BRUSH_NATIVE_MSL_SPARSE_SH_ADAM";
pub const FINE_RASTER_TILES_ENV: &str = "BRUSH_NATIVE_MSL_FINE_RASTER_TILES";

#[cfg(any(not(target_family = "wasm"), test))]
fn parse_bool(value: &str) -> Option<bool> {
    if value == "1" || value.eq_ignore_ascii_case("true") {
        Some(true)
    } else if value == "0" || value.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

#[cfg(not(target_family = "wasm"))]
fn value_enabled(value: &OsStr) -> bool {
    value.to_str().and_then(parse_bool).unwrap_or(false)
}

#[cfg(test)]
fn resolve_option(option: Option<&str>, preset: Option<&str>) -> bool {
    match option {
        Some(value) => parse_bool(value).unwrap_or(false),
        None => preset.and_then(parse_bool).unwrap_or(false),
    }
}

/// Whether the process requested the complete native-MSL optimization preset.
///
/// This reports the environment request only. Compile-time and device capability
/// gates still decide whether an individual optimization can run.
#[cfg(not(target_family = "wasm"))]
pub fn preset_requested() -> bool {
    std::env::var_os(PRESET_ENV)
        .as_deref()
        .is_some_and(value_enabled)
}

#[cfg(target_family = "wasm")]
pub const fn preset_requested() -> bool {
    false
}

/// Resolve one native-MSL optimization request.
///
/// An explicitly set option wins over [`PRESET_ENV`]. `1` and `true` enable it;
/// `0`, `false`, and unrecognized values disable it. When the option is absent,
/// it inherits the preset value.
#[cfg(not(target_family = "wasm"))]
pub fn option_requested(option_env: &str) -> bool {
    match std::env::var_os(option_env) {
        Some(value) => value_enabled(&value),
        None => preset_requested(),
    }
}

#[cfg(target_family = "wasm")]
pub const fn option_requested(_option_env: &str) -> bool {
    false
}

/// Whether the 16x8 training rasterizer was requested.
///
/// An explicit [`FINE_RASTER_TILES_ENV`] value wins over [`PRESET_ENV`], like
/// the other native-MSL options. Compile-time and platform gates still keep
/// this training-only specialization on supported Apple Silicon builds.
#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
pub fn fine_raster_tiles_requested() -> bool {
    option_requested(FINE_RASTER_TILES_ENV)
}

#[cfg(not(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
)))]
pub const fn fine_raster_tiles_requested() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::{parse_bool, resolve_option};

    #[test]
    fn parser_accepts_only_documented_values() {
        for value in ["1", "true", "TRUE", "True"] {
            assert_eq!(parse_bool(value), Some(true));
        }
        for value in ["0", "false", "FALSE", "FaLsE"] {
            assert_eq!(parse_bool(value), Some(false));
        }
        for value in ["", "yes", "on", "2", " true "] {
            assert_eq!(parse_bool(value), None);
        }
    }

    #[test]
    fn options_are_disabled_by_default() {
        assert!(!resolve_option(None, None));
        assert!(!resolve_option(None, Some("0")));
        assert!(!resolve_option(None, Some("false")));
        assert!(!resolve_option(None, Some("invalid")));
    }

    #[test]
    fn absent_option_inherits_enabled_preset() {
        assert!(resolve_option(None, Some("1")));
        assert!(resolve_option(None, Some("TRUE")));
    }

    #[test]
    fn explicit_option_overrides_preset() {
        assert!(resolve_option(Some("1"), Some("0")));
        assert!(resolve_option(Some("true"), None));
        assert!(!resolve_option(Some("0"), Some("1")));
        assert!(!resolve_option(Some("FALSE"), Some("true")));
    }

    #[test]
    fn invalid_explicit_option_fails_closed() {
        assert!(!resolve_option(Some(""), Some("1")));
        assert!(!resolve_option(Some("invalid"), Some("1")));
        assert!(!resolve_option(Some(" true "), Some("1")));
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_value_fails_closed() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let value = OsString::from_vec(vec![0xff]);
        assert!(!super::value_enabled(&value));
    }
}
