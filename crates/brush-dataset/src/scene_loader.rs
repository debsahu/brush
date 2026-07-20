use std::sync::Arc;

use brush_async::Actor;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::{
    config::LoadDatasetConfig,
    scene::{Scene, SceneBatch, sample_to_packed_data, view_to_sample_image},
};

const PREFETCH_BATCHES: usize = 4;

/// Shared cache of GPU-ready scene batches. Each slot holds at most one
/// batch; once the running total passes `budget_bytes`, new batches bypass
/// the cache and just get re-decoded + re-packed on every visit.
///
/// Caching the packed batch (instead of the decoded `DynamicImage`) skips
/// the per-hit decode → premultiply → repack work: a cache hit is now a
/// single copy of the already-packed `[H, W]` u32 buffer.
struct BatchCache {
    slots: Vec<Option<Arc<SceneBatch>>>,
    used_bytes: u64,
    budget_bytes: u64,
}

impl BatchCache {
    fn new(n_views: usize, budget_bytes: u64) -> Self {
        Self {
            slots: vec![None; n_views],
            used_bytes: 0,
            budget_bytes,
        }
    }

    fn get(&self, index: usize) -> Option<Arc<SceneBatch>> {
        self.slots[index].clone()
    }

    fn insert(&mut self, index: usize, batch: Arc<SceneBatch>) {
        if self.slots[index].is_some() {
            return;
        }
        // Track exact bytes: rounding to whole MB let sub-MB images slip in
        // for free and bypass the budget entirely.
        let size_bytes: u64 = batch
            .img_packed
            .as_bytes()
            .len()
            .try_into()
            .expect("shouldn't exceed ~18 Exabytes...");
        if self.used_bytes + size_bytes < self.budget_bytes {
            self.slots[index] = Some(batch);
            self.used_bytes += size_bytes;
        }
    }
}

pub struct SceneLoader {
    rx: mpsc::Receiver<SceneBatch>,
    // Owns the loader actor threads. Dropping cancels them; their
    // senders then drop, the channel closes, and `next_batch` returns.
    _actors: Vec<Actor>,
}

impl SceneLoader {
    pub fn new(scene: &Scene, seed: u64, config: &LoadDatasetConfig) -> Self {
        // Producers reserve a channel slot before decoding, so queued and
        // in-flight work together stay within this prefetch target.
        let (tx, rx) = mpsc::channel(PREFETCH_BATCHES);

        // Use up to one actor thread per producer so synchronous image decode
        // can actually run in parallel. When fewer CPU threads are available,
        // multiple async producers share each actor and still overlap I/O.
        let available_parallelism =
            std::thread::available_parallelism().map_or(1, |parallelism| parallelism.get());
        let n_actors = loader_actor_count(available_parallelism, cfg!(target_family = "wasm"));

        let views = scene.views.clone();
        let cache = Arc::new(Mutex::new(BatchCache::new(
            views.len(),
            config.max_scene_batch_cache_size,
        )));
        let load_locks = Arc::new((0..views.len()).map(|_| Mutex::new(())).collect::<Vec<_>>());

        let actors: Vec<Actor> = (0..n_actors)
            .map(|i| Actor::new(&format!("dataloader-{i}")))
            .collect();
        for producer_idx in 0..PREFETCH_BATCHES {
            let views = views.clone();
            let cache = cache.clone();
            let load_locks = load_locks.clone();
            let tx = tx.clone();
            let task_seed = seed.wrapping_add(producer_idx as u64);
            actors[producer_idx % n_actors]
                .run(move || run_loader(views, cache, load_locks, tx, task_seed))
                .detach();
        }

        Self {
            rx,
            _actors: actors,
        }
    }

    pub async fn next_batch(&mut self) -> SceneBatch {
        self.rx
            .recv()
            .await
            .expect("Scene loader channel closed unexpectedly")
    }
}

fn loader_actor_count(available_parallelism: usize, is_wasm: bool) -> usize {
    if is_wasm {
        1
    } else {
        available_parallelism.clamp(1, PREFETCH_BATCHES)
    }
}

async fn run_loader(
    views: Arc<Vec<crate::scene::SceneView>>,
    cache: Arc<Mutex<BatchCache>>,
    load_locks: Arc<Vec<Mutex<()>>>,
    tx: mpsc::Sender<SceneBatch>,
    seed: u64,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut shuffled: Vec<usize> = Vec::new();

    loop {
        let Ok(permit) = tx.reserve().await else {
            break;
        };

        if shuffled.is_empty() {
            shuffled = (0..views.len()).collect();
            shuffled.shuffle(&mut rng);
        }
        let index = shuffled.pop().expect("Need at least one view in dataset");
        let view = &views[index];

        let batch = if let Some(batch) = cache.lock().await.get(index) {
            batch
        } else {
            // A shuffled producer may pick the same uncached view. Serialize
            // only that view's miss and recheck the cache after waiting.
            let _load_guard = load_locks[index].lock().await;
            if let Some(batch) = cache.lock().await.get(index) {
                batch
            } else {
                let raw = view
                    .image
                    .load()
                    .await
                    .expect("Scene loader failed to load an image");
                let sample = view_to_sample_image(raw, view.image.alpha_mode());
                let (img_packed, has_alpha) = sample_to_packed_data(sample);

                let features = if let Some(load_features) = &view.features {
                    Some(
                        load_features
                            .load()
                            .await
                            .expect("Scene loader failed to load a feature map"),
                    )
                } else {
                    None
                };

                let batch = Arc::new(SceneBatch {
                    img_packed,
                    has_alpha,
                    alpha_mode: view.image.alpha_mode(),
                    features,
                    camera: view.camera,
                    view_index: index,
                });
                cache.lock().await.insert(index, batch.clone());
                batch
            }
        };

        // The channel takes an owned batch; clone the packed buffer out of
        // the shared cache entry.
        permit.send(batch.as_ref().clone());
        brush_async::yield_now().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = test)]
    fn loader_producers_are_bounded_by_prefetch_capacity() {
        assert_eq!(loader_actor_count(1, false), 1);
        assert_eq!(loader_actor_count(2, false), 2);
        assert_eq!(loader_actor_count(128, false), 4);
        assert_eq!(loader_actor_count(128, true), 1);

        assert!(
            loader_actor_count(128, false) <= PREFETCH_BATCHES,
            "loader actors exceeded prefetch capacity"
        );
    }
}
