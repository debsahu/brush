use std::path::Path;

use brush_vfs::BrushVfs;
use clap::Parser;
use tokio::io::AsyncReadExt;

use crate::config::TrainStreamConfig;

/// Split an args.txt string into argument tokens. Malformed quoting falls back
/// to the legacy whitespace-only behavior for API compatibility; config
/// loading uses [`try_split_args_str`] so it can reject malformed files.
pub fn split_args_str(content: &str) -> Vec<String> {
    try_split_args_str(content)
        .unwrap_or_else(|_| content.split_whitespace().map(str::to_owned).collect())
}

pub fn try_split_args_str(content: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut token_started = false;
    let mut quoted = false;
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        if quoted {
            match ch {
                '"' => quoted = false,
                '\\' => match chars.peek().copied() {
                    Some('"' | '\\') => current.push(chars.next().expect("peeked character")),
                    _ => current.push('\\'),
                },
                _ => current.push(ch),
            }
            continue;
        }

        match ch {
            '"' if !token_started || current.ends_with('=') => {
                quoted = true;
                token_started = true;
            }
            '"' => {
                // Legacy args.txt used whitespace-only splitting, so a quote
                // embedded in an unquoted filename was a literal character.
                current.push('"');
                token_started = true;
            }
            ch if ch.is_whitespace() => {
                if token_started {
                    args.push(std::mem::take(&mut current));
                    token_started = false;
                }
            }
            _ => {
                current.push(ch);
                token_started = true;
            }
        }
    }

    if quoted {
        return Err("unterminated double quote in args.txt".to_owned());
    }
    if token_started {
        args.push(current);
    }
    Ok(args)
}

/// Load `TrainStreamConfig` from args.txt via VFS.
pub async fn load_config_from_vfs(vfs: &BrushVfs) -> Option<TrainStreamConfig> {
    let args_path = Path::new("args.txt");

    // Check if args.txt exists in the VFS
    if !vfs
        .file_paths()
        .any(|p| p.file_name().and_then(|n| n.to_str()) == Some("args.txt"))
    {
        return None;
    }

    let mut reader = match vfs.reader_at_path(args_path).await {
        Ok(r) => r,
        Err(e) => {
            log::warn!("Failed to open args: {e}");
            return None;
        }
    };

    let mut content = String::new();
    if let Err(e) = reader.read_to_string(&mut content).await {
        log::warn!("Failed to read args: {e}");
        return None;
    }

    let file_args = match try_split_args_str(&content) {
        Ok(args) => args,
        Err(error) => {
            log::warn!("Failed to parse args: {error}");
            return None;
        }
    };
    if file_args.is_empty() {
        return None;
    }

    log::info!("Loaded settings from args");
    let mut all_args = vec!["brush".to_owned()];
    all_args.extend(file_args);

    TrainStreamConfig::try_parse_from(&all_args).ok()
}

/// Convert a `TrainStreamConfig` to individual command-line argument tokens.
/// Only values that differ from the defaults are included. Scalar values use
/// `--key=value` so each returned string remains one token even when the value
/// contains whitespace.
pub fn config_to_args(config: &TrainStreamConfig) -> Vec<String> {
    use serde_json::Value;

    let config_json = serde_json::to_value(config).unwrap_or(Value::Null);
    let default_json = serde_json::to_value(TrainStreamConfig::default()).unwrap_or(Value::Null);

    let mut args = Vec::new();

    if let (Value::Object(config_map), Value::Object(default_map)) = (config_json, default_json) {
        for (key, value) in config_map {
            // Skip if value equals default
            if default_map.get(&key) == Some(&value) {
                continue;
            }

            // Skip null values (None options that are also None by default would be caught above)
            if value.is_null() {
                continue;
            }

            // Format one complete argument token.
            let arg_name = format!("--{key}");
            match value {
                Value::Bool(b) => {
                    // For booleans, only output flag if true (and different from default)
                    if b {
                        args.push(arg_name);
                    }
                }
                Value::String(s) => {
                    args.push(format!("{arg_name}={s}"));
                }
                Value::Number(n) => {
                    args.push(format!("{arg_name}={n}"));
                }
                Value::Array(items) => {
                    // Multi-value clap args in this config use a comma delimiter.
                    let joined = items
                        .iter()
                        .map(|v| match v {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    args.push(format!("{arg_name}={joined}"));
                }
                _ => {
                    args.push(format!("{arg_name}={value}"));
                }
            }
        }
    }

    args
}

fn quote_arg(arg: &str) -> String {
    if !arg.chars().any(|ch| ch.is_whitespace() || ch == '"') {
        return arg.to_owned();
    }

    let mut quoted = String::with_capacity(arg.len() + 2);
    quoted.push('"');
    for ch in arg.chars() {
        if matches!(ch, '"' | '\\') {
            quoted.push('\\');
        }
        quoted.push(ch);
    }
    quoted.push('"');
    quoted
}

/// Serialize non-default config arguments for `args.txt`. Backslashes outside
/// quotes remain literal for compatibility with legacy Windows paths;
/// [`try_split_args_str`] handles the quoting emitted here for whitespace values.
pub fn config_to_string(config: &TrainStreamConfig) -> String {
    config_to_args(config)
        .iter()
        .map(|arg| quote_arg(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn merge_config_fields(
    initial_config: &TrainStreamConfig,
    cli_config: &TrainStreamConfig,
    explicit_fields: &[String],
) -> Option<TrainStreamConfig> {
    use serde_json::Value;

    let Value::Object(mut initial) = serde_json::to_value(initial_config).ok()? else {
        return None;
    };
    let Value::Object(cli) = serde_json::to_value(cli_config).ok()? else {
        return None;
    };

    for field in explicit_fields {
        if let Some(value) = cli.get(field) {
            initial.insert(field.clone(), value.clone());
        } else {
            log::warn!("Ignoring unknown CLI config field '{field}'");
        }
    }

    // These flags are mutually exclusive. An explicit CLI choice replaces a
    // saved choice instead of making the merged config invalid.
    if explicit_fields.iter().any(|field| field == "ppisp")
        && cli.get("ppisp") == Some(&Value::Bool(true))
    {
        initial.insert("bilateral-grid".to_owned(), Value::Bool(false));
    }
    if explicit_fields
        .iter()
        .any(|field| field == "bilateral-grid")
        && cli.get("bilateral-grid") == Some(&Value::Bool(true))
    {
        initial.insert("ppisp".to_owned(), Value::Bool(false));
    }

    serde_json::from_value(Value::Object(initial)).ok()
}

/// Merge explicitly supplied CLI fields into an initial config. Unlike a
/// default-value comparison, this preserves the user's intent when the CLI
/// value happens to equal the program default.
pub fn merge_explicit_cli_fields(
    initial_config: &TrainStreamConfig,
    cli_config: &TrainStreamConfig,
    explicit_fields: &[String],
) -> TrainStreamConfig {
    merge_config_fields(initial_config, cli_config, explicit_fields).unwrap_or_else(|| {
        log::warn!("Failed to merge CLI config fields; using CLI config");
        cli_config.clone()
    })
}

/// Merge an initial config (e.g., from args.txt) with CLI arguments.
/// CLI arguments take precedence over the initial config values.
pub fn merge_configs(
    initial_config: &TrainStreamConfig,
    cli_config: &TrainStreamConfig,
) -> TrainStreamConfig {
    use serde_json::Value;

    let cli = serde_json::to_value(cli_config).unwrap_or(Value::Null);
    let defaults = serde_json::to_value(TrainStreamConfig::default()).unwrap_or(Value::Null);
    let explicit_fields = match (cli, defaults) {
        (Value::Object(cli), Value::Object(defaults)) => cli
            .iter()
            .filter(|(key, value)| defaults.get(*key) != Some(*value))
            .map(|(key, _)| key.clone())
            .collect(),
        _ => Vec::new(),
    };
    merge_explicit_cli_fields(initial_config, cli_config, &explicit_fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = test)]
    fn test_config_to_args_only_includes_changes() {
        let mut config = TrainStreamConfig::default();
        config.train_config.total_train_iters = 5000;
        config.model_config.sh_degree = 2;
        config.load_config.max_frames = Some(10);
        let args = config_to_args(&config);

        // Should have exactly 3 args for the 3 changes
        assert_eq!(args.len(), 3, "Should have 3 args, got: {args:?}");
        // Verify they contain the right values
        let args_str = args.join(" ");
        assert!(
            args_str.contains("--total-train-iters=5000"),
            "Missing total-train-iters"
        );
        assert!(args_str.contains("--sh-degree=2"), "Missing sh-degree");
        assert!(args_str.contains("--max-frames=10"), "Missing max-frames");
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_config_to_args_vec_round_trip() {
        let mut config = TrainStreamConfig::default();
        config.train_config.background_color = vec![1.0, 0.5, 0.25];
        let merged = merge_configs(&config, &TrainStreamConfig::default());
        assert_eq!(merged.train_config.background_color, vec![1.0, 0.5, 0.25]);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_config_round_trip() {
        let mut original = TrainStreamConfig::default();
        original.train_config.total_train_iters = 5000;
        original.model_config.sh_degree = 2;
        original.load_config.max_frames = Some(10);
        original.process_config.seed = 123;

        // Convert to args
        let args = config_to_args(&original);

        // Parse args back
        let mut cli_args = vec!["brush".to_owned()];
        cli_args.extend(args);

        let parsed = TrainStreamConfig::try_parse_from(&cli_args).expect("Should parse");
        assert_eq!(parsed.train_config.total_train_iters, 5000);
        assert_eq!(parsed.model_config.sh_degree, 2);
        assert_eq!(parsed.load_config.max_frames, Some(10));
        assert_eq!(parsed.process_config.seed, 123);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_config_string_round_trip_preserves_spaces() {
        let mut original = TrainStreamConfig::default();
        original.process_config.export_path = "./exports with spaces/{dataset}/".to_owned();

        let content = config_to_string(&original);
        let mut args = vec!["brush".to_owned()];
        args.extend(try_split_args_str(&content).expect("serialized args should parse"));
        let parsed = TrainStreamConfig::try_parse_from(args).expect("config should round-trip");

        assert_eq!(
            parsed.process_config.export_path,
            original.process_config.export_path
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_legacy_windows_paths_keep_backslashes_and_apostrophes() {
        let args = try_split_args_str(
            r#"--export-path C:\models\Markus's-scans --export-name export_{iter}.ply"#,
        )
        .expect("legacy args should parse");

        assert_eq!(args[1], r#"C:\models\Markus's-scans"#);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_unterminated_double_quote_is_rejected() {
        assert!(try_split_args_str(r#"--export-path "unfinished"#).is_err());
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_legacy_embedded_quote_remains_literal() {
        let args =
            try_split_args_str(r#"--export-path weird"name"#).expect("legacy args should parse");
        assert_eq!(args, ["--export-path", r#"weird"name"#]);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_infallible_splitter_preserves_legacy_signature() {
        assert_eq!(
            split_args_str(r#"--export-path "unfinished"#),
            ["--export-path", r#""unfinished"#]
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_explicit_default_overrides_saved_non_default() {
        let mut initial = TrainStreamConfig::default();
        initial.train_config.total_train_iters = 5000;
        let cli = TrainStreamConfig::default();

        let merged = merge_explicit_cli_fields(&initial, &cli, &["total-train-iters".to_owned()]);

        assert_eq!(merged.train_config.total_train_iters, 30_000);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_explicit_appearance_mode_replaces_saved_mode() {
        let mut initial = TrainStreamConfig::default();
        initial.train_config.bilateral_grid = true;
        let mut cli = TrainStreamConfig::default();
        cli.train_config.ppisp = true;

        let merged = merge_explicit_cli_fields(&initial, &cli, &["ppisp".to_owned()]);

        assert!(merged.train_config.ppisp);
        assert!(!merged.train_config.bilateral_grid);
    }
}
