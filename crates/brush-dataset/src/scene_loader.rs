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
/// behind a refcount first (see `share_buffer`), so a hit doesn't copy the
/// pixels or the priors either: it hands out a view of the same allocations.
///
/// The budget counts *every* buffer a cached batch keeps resident — the
/// packed image and the decoded depth/normal/feature priors alike (see
/// `SceneBatch::batch_bytes`). Counting the image alone under-counted a 4K
/// batch 5x, so the 6 GiB default admitted 194 views holding 32 GB and swapped
/// a 24 GB machine. Honest accounting caches 38 of those views instead: fewer
/// hits, but a bounded resident set.
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
    /// buffers only get shared when they're actually going to be kept.
    ///
    /// Tracks exact bytes: rounding to whole MB let sub-MB images slip in
    /// for free and bypass the budget entirely.
    fn admits(&self, index: usize, batch: &SceneBatch) -> bool {
        self.slots[index].is_none() && self.used_bytes + batch.batch_bytes() < self.budget_bytes
    }

    fn insert(&mut self, index: usize, batch: Arc<SceneBatch>) {
        if !self.admits(index, &batch) {
            return;
        }
        self.used_bytes += batch.batch_bytes();
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
                    batch = share_batch_buffers(batch);
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

/// Move a batch buffer (packed pixels or a decoded prior) behind a refcount,
/// so cloning the batch out of the cache doesn't copy it. Uploading to the GPU
/// is unaffected: that copies into a staging buffer either way.
fn share_buffer(data: TensorData) -> TensorData {
    TensorData::from_bytes(data.bytes.shared(), data.shape, data.dtype)
}

/// Put every heavy buffer of a batch behind a refcount, ready to be cached:
/// the hand-off into the cache and every later hit then cost a refcount
/// instead of a full copy.
///
/// The priors have to be shared too. At 4K they are 132.7 MB against the
/// image's 33.2 MB, so sharing the pixels alone still deep-copied 4x the
/// image on every cache hit.
fn share_batch_buffers(mut batch: SceneBatch) -> SceneBatch {
    batch.img_packed = share_buffer(batch.img_packed);
    batch.features = batch
        .features
        .take()
        .map(|(data, channels)| (share_buffer(data), channels));
    batch.depth = batch.depth.take().map(share_buffer);
    batch.normal = batch.normal.take().map(share_buffer);
    batch
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

    /// Bytes each buffer of a `[h, w]` test batch occupies on the host.
    const fn expected_bytes(h: usize, w: usize) -> (u64, u64, u64, u64) {
        let px = (h * w) as u64;
        // i32 packed pixels, f32 depth `[h, w]`, f32 normals `[h, w, 3]`,
        // f32 features `[h, w, 2]`.
        (px * 4, px * 4, px * 3 * 4, px * 2 * 4)
    }

    /// A batch shaped like a real view: packed pixels plus, optionally, the
    /// decoded f32 priors that ride along in the same cached batch.
    fn test_batch(h: usize, w: usize, with_priors: bool) -> SceneBatch {
        let (depth, normal, features) = if with_priors {
            (
                Some(TensorData::new(vec![1.0f32; h * w], [h, w])),
                Some(TensorData::new(vec![0.5f32; h * w * 3], [h, w, 3])),
                Some((TensorData::new(vec![0.25f32; h * w * 2], [h, w, 2]), 2)),
            )
        } else {
            (None, None, None)
        };

        SceneBatch {
            img_packed: TensorData::new(vec![0i32; h * w], [h, w]),
            has_alpha: false,
            alpha_mode: brush_render::AlphaMode::Masked,
            features,
            depth,
            normal,
            camera: brush_render::camera::Camera::new(
                glam::Vec3::ZERO,
                glam::Quat::IDENTITY,
                1.0,
                1.0,
                glam::Vec2::splat(0.5),
                brush_render::kernels::camera_model::CameraModel::default(),
            ),
            view_index: 0,
        }
    }

    /// T11 — the cache budget has to count the decoded priors, not just the
    /// packed image. A 4K view is 33.2 MB packed against 132.7 MB of f32
    /// depth + normals, so counting the image alone let the 6 GiB default
    /// admit 194 views holding 32 GB: the swap this repair exists to stop.
    #[wasm_bindgen_test(unsupported = test)]
    fn cache_budget_counts_prior_bytes() {
        let (h, w) = (8usize, 4usize);
        let (packed, depth, normal, features) = expected_bytes(h, w);
        let priors = depth + normal + features;
        let batch = test_batch(h, w, true);

        assert_eq!(batch.packed_bytes(), packed);
        assert_eq!(
            batch.prior_bytes(),
            priors,
            "prior_bytes must sum depth + normal + features"
        );
        assert_eq!(batch.batch_bytes(), packed + priors);

        // Null model: with no priors attached the two accountings agree, so
        // the assertions below are about the priors and nothing else.
        let bare = test_batch(h, w, false);
        assert_eq!(bare.prior_bytes(), 0);
        assert_eq!(bare.batch_bytes(), bare.packed_bytes());

        // A budget sized to the image alone must NOT admit the batch. Under
        // the old packed-only accounting it would have.
        let mut cache = BatchCache::new(2, packed + 1);
        assert!(
            !cache.admits(0, &batch),
            "priors must count against the budget"
        );
        cache.insert(0, Arc::new(batch.clone()));
        assert!(cache.get(0).is_none(), "over-budget batch must not cache");
        assert_eq!(cache.used_bytes, 0);

        // Sized to image + priors it fits, and the accounted bytes are the
        // full resident cost of the batch.
        let mut cache = BatchCache::new(2, packed + priors + 1);
        assert!(cache.admits(0, &batch));
        cache.insert(0, Arc::new(batch.clone()));
        assert!(cache.get(0).is_some());
        assert_eq!(cache.used_bytes, packed + priors);

        // The budget is now genuinely spent: a second view of the same size
        // is refused, where packed-only accounting would have taken several
        // more and blown past the budget.
        assert!(!cache.admits(1, &batch));
    }

    /// T12 — every heavy buffer is refcount-shared on insert, so a cache hit
    /// is a pointer bump. Sharing only `img_packed` left the priors to
    /// deep-copy on every hit: 132.7 MB of churn per 4K view.
    #[wasm_bindgen_test(unsupported = test)]
    fn cache_insert_shares_prior_buffers() {
        let (h, w) = (8usize, 4usize);
        let batch = test_batch(h, w, true);

        // Null model: an owned buffer really does deep-copy on clone, so an
        // equal-pointer assertion below is evidence of sharing and not just
        // a property `TensorData::clone` has anyway.
        let owned_clone = batch.clone();
        assert_ne!(
            owned_clone
                .depth
                .as_ref()
                .expect("depth")
                .as_bytes()
                .as_ptr(),
            batch.depth.as_ref().expect("depth").as_bytes().as_ptr(),
            "an owned prior buffer is expected to copy on clone"
        );

        // The same call `run_loader` makes before handing a batch to the
        // cache — the production path, not a copy of it.
        let batch = share_batch_buffers(batch);

        let mut cache = BatchCache::new(1, u64::MAX);
        cache.insert(0, Arc::new(batch.clone()));

        let cached = cache.get(0).expect("view must be cached");
        // The two batches `run_loader` hands the trainer: the one produced by
        // the miss that populated the cache, and the one every later hit
        // clones back out. Both must alias the cached allocations.
        let on_miss = batch;
        let on_hit = cached.as_ref().clone();

        for (label, handed_out) in [("miss", &on_miss), ("hit", &on_hit)] {
            assert_eq!(
                handed_out.img_packed.as_bytes().as_ptr(),
                cached.img_packed.as_bytes().as_ptr(),
                "cache {label} copied the packed image instead of sharing it"
            );
            assert_eq!(
                handed_out
                    .depth
                    .as_ref()
                    .expect("depth")
                    .as_bytes()
                    .as_ptr(),
                cached.depth.as_ref().expect("depth").as_bytes().as_ptr(),
                "cache {label} copied the depth prior instead of sharing it"
            );
            assert_eq!(
                handed_out
                    .normal
                    .as_ref()
                    .expect("normal")
                    .as_bytes()
                    .as_ptr(),
                cached.normal.as_ref().expect("normal").as_bytes().as_ptr(),
                "cache {label} copied the normal prior instead of sharing it"
            );
            assert_eq!(
                handed_out
                    .features
                    .as_ref()
                    .expect("features")
                    .0
                    .as_bytes()
                    .as_ptr(),
                cached
                    .features
                    .as_ref()
                    .expect("features")
                    .0
                    .as_bytes()
                    .as_ptr(),
                "cache {label} copied the feature map instead of sharing it"
            );
        }

        // Sharing must not disturb the accounting T11 pins.
        let (packed, depth, normal, features) = expected_bytes(h, w);
        assert_eq!(cached.batch_bytes(), packed + depth + normal + features);
    }
}
