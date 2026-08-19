use std::sync::Arc;

use brush_async::Actor;
use burn::tensor::TensorData;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::{
    config::LoadDatasetConfig,
    scene::{Scene, SceneBatch, view_to_packed_data},
};

const PREFETCH_BATCHES: usize = 4;

/// Shared cache of GPU-ready scene batches. Each slot holds at most one
/// batch; once the running total passes `budget_bytes`, new batches bypass
/// the cache and just get re-decoded + re-packed on every visit.
///
/// Caching the packed batch (instead of the decoded `DynamicImage`) skips
/// the per-hit decode → premultiply → repack work. Cached buffers are put
/// behind a refcount first (see `share_packed`), so a hit doesn't copy the
/// pixels either: it hands out a view of the same allocation.
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

    /// Whether `insert` would take this batch: nothing cached for the view
    /// yet, and it still fits the budget. Checked before caching so the
    /// packed bytes only get shared when they're actually going to be kept.
    ///
    /// Tracks exact bytes: rounding to whole MB let sub-MB images slip in
    /// for free and bypass the budget entirely.
    fn admits(&self, index: usize, batch: &SceneBatch) -> bool {
        self.slots[index].is_none() && self.used_bytes + batch.packed_bytes() < self.budget_bytes
    }

    fn insert(&mut self, index: usize, batch: Arc<SceneBatch>) {
        if !self.admits(index, &batch) {
            return;
        }
        self.used_bytes += batch.packed_bytes();
        self.slots[index] = Some(batch);
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

        let cached = cache.lock().await.get(index);

        let batch = if let Some(batch) = cached {
            // The cached buffer is refcounted, so this is a pointer bump
            // rather than a copy of the whole image.
            batch.as_ref().clone()
        } else {
            // A shuffled producer may pick the same uncached view. Serialize
            // only that view's miss and recheck the cache after waiting.
            let _load_guard = load_locks[index].lock().await;
            if let Some(batch) = cache.lock().await.get(index) {
                batch.as_ref().clone()
            } else {
                let raw = view
                    .image
                    .load()
                    .await
                    .expect("Scene loader failed to load an image");
                let (img_packed, has_alpha) = view_to_packed_data(raw, view.image.alpha_mode());

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

                let depth = if let Some(load_depth) = &view.depth {
                    let [h, w] = [img_packed.shape[0], img_packed.shape[1]];
                    Some(
                        load_depth
                            .load(h, w)
                            .await
                            .expect("Scene loader failed to load a depth map"),
                    )
                } else {
                    None
                };

                let normal = if let Some(load_normal) = &view.normal {
                    let [h, w] = [img_packed.shape[0], img_packed.shape[1]];
                    Some(
                        load_normal
                            .load(h, w)
                            .await
                            .expect("Scene loader failed to load a normal map"),
                    )
                } else {
                    None
                };

                let mut batch = SceneBatch {
                    img_packed,
                    has_alpha,
                    alpha_mode: view.image.alpha_mode(),
                    features,
                    depth,
                    normal,
                    camera: view.camera,
                    view_index: index,
                };

                let mut cache = cache.lock().await;
                if cache.admits(index, &batch) {
                    // Share the pixels before caching: this hand-off and
                    // every later hit then costs a refcount instead of a
                    // full copy.
                    batch.img_packed = share_packed(batch.img_packed);
                    cache.insert(index, Arc::new(batch.clone()));
                }
                batch
            }
        };

        // The slot was already reserved above, so this send is infallible.
        permit.send(batch);
        brush_async::yield_now().await;
    }
}

/// Move the packed pixels behind a refcount, so cloning the batch out of the
/// cache doesn't copy them. Uploading to the GPU is unaffected: that copies
/// into a staging buffer either way.
fn share_packed(data: TensorData) -> TensorData {
    TensorData::from_bytes(data.bytes.shared(), data.shape, data.dtype)
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
