# Brush env-gated JSONL metrics log

Adds a lightweight, env-gated JSONL metrics writer to the training loop so runs
are pollable (via `tail -f`) instead of only surfacing in the TUI.

## Usage

```
BRUSH_METRICS_LOG=/path/run.jsonl BRUSH_METRICS_EVERY=50 brush <dataset> [flags...]
```

- `BRUSH_METRICS_LOG` (unset by default): path to append JSONL to. When unset,
  behaviour is byte-identical to stock brush (one `Option` check per iter, no
  GPU readback).
- `BRUSH_METRICS_EVERY` (default `50`): log cadence in iters. First iter and last
  iter are always logged in addition.

Each line: `{"iter":<u32>,"num_splats":<u32>,"loss":<f64>,"elapsed_s":<f64>}`
(loss = per-step total loss scalar; elapsed_s = seconds since loop start).
Each write is flushed so `tail -f` sees lines live. The loss GPU readback only
happens on the throttled log iters.

File changed: crates/brush-process/src/train_stream.rs

Sample (30-iter smoke, BRUSH_METRICS_EVERY=10, --max-splats 200000):
{"iter":1,"num_splats":185890,"loss":0.050634875893592834,"elapsed_s":0.165395792}
{"iter":10,"num_splats":185890,"loss":0.10172834992408752,"elapsed_s":0.349220792}
{"iter":20,"num_splats":185890,"loss":0.05962536111474037,"elapsed_s":0.561540584}
{"iter":30,"num_splats":185890,"loss":0.03921128809452057,"elapsed_s":0.810611625}

## Diff


```diff
--- /tmp/train_stream.rs.bak	2026-07-21 11:45:50
+++ crates/brush-process/src/train_stream.rs	2026-07-21 11:45:50
@@ -337,6 +337,21 @@
     let process_config = &train_stream_config.process_config;
 
     log::info!("Start training loop.");
+
+    // Env-gated JSONL metrics writer. When `BRUSH_METRICS_LOG` points at a
+    // file, append one JSON line every `BRUSH_METRICS_EVERY` iters (default 50),
+    // plus the first and last iter, so a running train is pollable via
+    // `tail -f`. Reading the loss scalar forces a GPU readback, so it only
+    // happens on the iters we actually log. When the env var is unset this is a
+    // single `Option` check per iter, so default behaviour is byte-identical.
+    let metrics_log_path = std::env::var_os("BRUSH_METRICS_LOG").map(PathBuf::from);
+    let metrics_every: u32 = std::env::var("BRUSH_METRICS_EVERY")
+        .ok()
+        .and_then(|v| v.parse().ok())
+        .filter(|&n| n > 0)
+        .unwrap_or(50);
+    let metrics_start = Instant::now();
+
     for iter in process_config.start_iter..train_stream_config.train_config.total_iters() {
         let target_lod = if lod_levels == 0 || iter < training_steps {
             0u32
@@ -629,6 +644,37 @@
                     .log_splat_distribution_stats(iter, splats.clone())
                     .await
                     .unwrap();
+            }
+        }
+
+        // --- Env-gated JSONL metrics log ---
+        // `iter` here is the post-increment value that matches the reported
+        // iteration. Only touch the loss tensor (GPU readback) on log iters.
+        if let Some(metrics_path) = &metrics_log_path {
+            let is_first_step = iter == process_config.start_iter + 1;
+            if is_first_step || is_last_step || iter.is_multiple_of(metrics_every) {
+                let loss = stats.loss.clone().into_scalar_async::<f32>().await? as f64;
+                let num_splats = splats.num_splats();
+                let elapsed_s = metrics_start.elapsed().as_secs_f64();
+                let line = format!(
+                    "{{\"iter\":{iter},\"num_splats\":{num_splats},\"loss\":{loss},\"elapsed_s\":{elapsed_s}}}"
+                );
+                match std::fs::OpenOptions::new()
+                    .create(true)
+                    .append(true)
+                    .open(metrics_path)
+                {
+                    Ok(mut file) => {
+                        use std::io::Write as _;
+                        if let Err(error) = writeln!(file, "{line}").and_then(|()| file.flush()) {
+                            log::warn!("BRUSH_METRICS_LOG write failed: {error}");
+                        }
+                    }
+                    Err(error) => log::warn!(
+                        "BRUSH_METRICS_LOG open failed ({}): {error}",
+                        metrics_path.display()
+                    ),
+                }
             }
         }
 
```
