use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, TextureMode,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
    render_splats,
};
use brush_render_bwd::render_splats as render_splats_diff;
use brush_train::train::SplatTrainer;
use burn::tensor::{Device, TensorData};
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};

const SEED: u64 = 42;
const ITERS_PER_SYNC: u32 = 4;

fn gen_splats(device: &Device, count: usize) -> Splats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            // Create clusters with some randomness
            let cluster_center = [
                rng.random_range(-5.0..5.0),
                rng.random_range(-3.0..3.0),
                rng.random_range(-10.0..10.0),
            ];
            let offset = [
                rng.random::<f32>() - 0.5,
                rng.random::<f32>() - 0.5,
                rng.random::<f32>() - 0.5,
            ];
            [
                cluster_center[0] + offset[0] * 2.0,
                cluster_center[1] + offset[1] * 2.0,
                cluster_center[2] + offset[2] * 3.0,
            ]
        })
        .collect();

    // Realistic scale distribution (log-normal-ish)
    let log_scales: Vec<f32> = (0..count)
        .flat_map(|_| {
            let base_scale = rng.random_range(0.01..0.1_f32).ln();
            let variation = rng.random_range(0.8..1.2);
            [base_scale, base_scale * variation, base_scale * variation]
        })
        .collect();

    // Random rotations using proper quaternion generation
    let rotations: Vec<f32> = (0..count)
        .flat_map(|_| {
            let u1 = rng.random::<f32>();
            let u2 = rng.random::<f32>();
            let u3 = rng.random::<f32>();

            let sqrt1_u1 = (1.0 - u1).sqrt();
            let sqrt_u1 = u1.sqrt();
            let theta1 = 2.0 * std::f32::consts::PI * u2;
            let theta2 = 2.0 * std::f32::consts::PI * u3;

            [
                sqrt1_u1 * theta1.sin(),
                sqrt1_u1 * theta1.cos(),
                sqrt_u1 * theta2.sin(),
                sqrt_u1 * theta2.cos(),
            ]
        })
        .collect();

    // Realistic color distribution
    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(0.1..0.9),
                rng.random_range(0.1..0.9),
                rng.random_range(0.1..0.9),
            ]
        })
        .collect();

    // Realistic opacity distribution (mostly opaque with some variation)
    let opacities: Vec<f32> = (0..count).map(|_| rng.random_range(0.05..1.0)).collect();

    Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh_coeffs,
        opacities,
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0)
}

fn generate_training_batch(resolution: (u32, u32), camera_pos: Vec3) -> SceneBatch {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED + camera_pos.x as u64);

    let (width, height) = resolution;
    let pixel_count = (width * height) as usize;

    let img_packed_data: Vec<i32> = (0..pixel_count)
        .map(|i| {
            let x = (i as u32) % width;
            let y = (i as u32) / width;
            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;
            let mut mk_byte = |v: f32| -> u32 {
                let v = (v + (rng.random::<f32>() - 0.5) * 0.1).clamp(0.0, 1.0);
                (v * 255.0).round() as u32
            };
            let r = mk_byte(nx * 0.6 + 0.2);
            let g = mk_byte(ny * 0.6 + 0.2);
            let b = mk_byte((nx + ny) * 0.3 + 0.4);
            (r | g << 8 | b << 16 | 255 << 24) as i32
        })
        .collect();

    let img_packed = TensorData::new(img_packed_data, [height as usize, width as usize]);
    let camera = Camera::new(
        camera_pos,
        Quat::IDENTITY,
        50.0,
        50.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );

    SceneBatch {
        img_packed,
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera,
    }
}

fn bench_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        50.0,
        50.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    )
}

/// Render `iters` frames with pre-built splats. The hot loop only — callers
/// own setup, so benches can keep splat generation out of the timed body.
async fn forward_iters(splats: &Splats, camera: &Camera, resolution: (u32, u32), iters: u32) {
    for _ in 0..iters {
        let _ = render_splats(
            splats.clone(),
            camera,
            glam::uvec2(resolution.0, resolution.1),
            Vec3::ZERO,
            None,
            TextureMode::Float,
        )
        .await;
    }
}

/// Render + backward `iters` times with pre-built splats. Hot loop only.
async fn backward_iters(splats: &Splats, camera: &Camera, resolution: (u32, u32), iters: u32) {
    for _ in 0..iters {
        let diff_out = render_splats_diff(
            splats.clone(),
            camera,
            glam::uvec2(resolution.0, resolution.1),
            Vec3::ZERO,
        )
        .await;
        let _ = diff_out.img.mean().backward();
    }
}

/// Run `iters` training steps against pre-built batches. `splats` is an
/// `Option` because `trainer.step` consumes and returns the splats, and
/// benches call this from an `FnMut` closure where moving out of a captured
/// variable needs a `take`. `step_offset` keeps the batch alternation going
/// across repeated calls.
async fn training_iters(
    trainer: &mut SplatTrainer,
    splats: &mut Option<Splats>,
    batches: &[SceneBatch],
    step_offset: usize,
    iters: u32,
) {
    for i in 0..iters as usize {
        let batch = batches[(step_offset + i) % batches.len()].clone();
        let cur_splats = splats.take().expect("splats always put back");
        let (new_splats, _) = trainer.step(batch, cur_splats).await;
        *splats = Some(new_splats);
    }
}

// Every bench below keeps setup (splat/GT generation, trainer init) outside
// the timed closure and runs one untimed warmup pass first, so samples
// measure steady-state GPU work rather than pipeline compilation, autotune,
// or CPU-side data generation.
#[cfg(not(target_family = "wasm"))]
#[divan::bench_group(max_time = 1)]
mod forward_rendering {
    const RESOLUTIONS: [(u32, u32); 4] = [(1024, 1024), (1920, 1080), (2560, 1440), (3200, 1800)];
    const SPLAT_COUNTS: [usize; 3] = [500_000, 1_000_000, 2_500_000];

    use burn::module::AutodiffModule;
    use burn::{backend::wgpu::WgpuDevice, prelude::Device};
    use burn_cubecl::cubecl::future::block_on;

    use crate::benches::{ITERS_PER_SYNC, bench_camera, forward_iters, gen_splats};

    fn bench_forward(bencher: divan::Bencher, splat_count: usize, resolution: (u32, u32)) {
        let device = Device::from(WgpuDevice::default()).autodiff();
        let splats = gen_splats(&device, splat_count).valid();
        let camera = bench_camera();
        block_on(async {
            forward_iters(&splats, &camera, resolution, ITERS_PER_SYNC).await;
            device.sync().expect("Failed to sync");
        });
        bencher
            .counter(divan::counter::ItemsCount::new(ITERS_PER_SYNC))
            .bench_local(move || {
                block_on(async {
                    forward_iters(&splats, &camera, resolution, ITERS_PER_SYNC).await;
                    device.sync().expect("Failed to sync");
                });
            });
    }

    #[divan::bench(args = SPLAT_COUNTS)]
    fn render_1080p(bencher: divan::Bencher, splat_count: usize) {
        bench_forward(bencher, splat_count, (1920, 1080));
    }

    #[divan::bench(args = RESOLUTIONS)]
    fn render_2m_splats(bencher: divan::Bencher, resolution: (u32, u32)) {
        bench_forward(bencher, 2_000_000, resolution);
    }
}

#[cfg(not(target_family = "wasm"))]
#[divan::bench_group(max_time = 2)]
mod backward_rendering {
    const RESOLUTIONS: [(u32, u32); 4] = [(1024, 1024), (1920, 1080), (2560, 1440), (3200, 1800)];

    use burn::{backend::wgpu::WgpuDevice, prelude::Device};
    use burn_cubecl::cubecl::future::block_on;

    use crate::benches::{ITERS_PER_SYNC, backward_iters, bench_camera, gen_splats};

    fn bench_backward(bencher: divan::Bencher, splat_count: usize, resolution: (u32, u32)) {
        let device = Device::from(WgpuDevice::default()).autodiff();
        let splats = gen_splats(&device, splat_count);
        let camera = bench_camera();
        block_on(async {
            backward_iters(&splats, &camera, resolution, ITERS_PER_SYNC).await;
            device.sync().expect("Failed to sync");
        });
        bencher
            .counter(divan::counter::ItemsCount::new(ITERS_PER_SYNC))
            .bench_local(move || {
                block_on(async {
                    backward_iters(&splats, &camera, resolution, ITERS_PER_SYNC).await;
                    device.sync().expect("Failed to sync");
                });
            });
    }

    #[divan::bench(args = [1_000_000, 2_000_000, 5_000_000])]
    fn render_grad_1080p(bencher: divan::Bencher, splat_count: usize) {
        bench_backward(bencher, splat_count, (1920, 1080));
    }

    #[divan::bench(args = RESOLUTIONS)]
    fn render_grad_2m_splats(bencher: divan::Bencher, resolution: (u32, u32)) {
        bench_backward(bencher, 2_000_000, resolution);
    }
}

#[cfg(not(target_family = "wasm"))]
// Canonical A/B runs must not override Divan's sample count, sample size, or
// minimum time; doing so changes the evolving training-step window.
#[divan::bench_group(sample_count = 10, sample_size = 1)]
mod training {
    const SPLAT_COUNTS: [usize; 3] = [500_000, 1_000_000, 2_500_000];

    use burn::{backend::wgpu::WgpuDevice, prelude::Device};
    use burn_cubecl::cubecl::future::block_on;
    use glam::Vec3;

    use crate::benches::{
        ITERS_PER_SYNC, SEED, gen_splats, generate_training_batch, training_iters,
    };
    use brush_render::bounding_box::BoundingBox;
    use brush_train::{config::TrainConfig, train::SplatTrainer};

    #[divan::bench(args = SPLAT_COUNTS)]
    fn train_steps(bencher: divan::Bencher, splat_count: usize) {
        let device = Device::from(WgpuDevice::default()).autodiff();
        let batches = [
            generate_training_batch((1920, 1080), Vec3::new(0.0, 0.0, 5.0)),
            generate_training_batch((1920, 1080), Vec3::new(2.0, 0.0, 5.0)),
        ];
        // CPU background noise uses an unseeded thread RNG. Disable it in the
        // regression bench so separate builds follow the same input sequence.
        let config = TrainConfig {
            background_noise_strength: 0.0,
            ..TrainConfig::default()
        };
        let mut trainer = SplatTrainer::new(
            &config,
            &device,
            BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
        );
        let mut splats = Some(gen_splats(&device, splat_count));
        // Seed CubeCL's RNG used by tensor noise. With one sample iteration
        // and a fixed sample count, every build consumes the same call window.
        device.seed(SEED);
        // Warmup also initializes the optimizer state (first-step lazy init).
        block_on(async {
            training_iters(&mut trainer, &mut splats, &batches, 0, ITERS_PER_SYNC).await;
            device.sync().expect("Failed to sync");
        });
        // Splats keep evolving across the fixed ten-sample window (no refine
        // runs, so the count stays fixed); batch cadence carries via `step`.
        let mut step = ITERS_PER_SYNC as usize;
        bencher
            .counter(divan::counter::ItemsCount::new(ITERS_PER_SYNC))
            .bench_local(move || {
                block_on(async {
                    training_iters(&mut trainer, &mut splats, &batches, step, ITERS_PER_SYNC).await;
                    step += ITERS_PER_SYNC as usize;
                    device.sync().expect("Failed to sync");
                });
            });
    }
}

// The bench target compiles this module under `cfg(test)` too, but with
// `harness = false` nothing collects `#[test]` fns — so everything here is
// "dead" in that build. Allow it rather than fight the target quirk.
#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use brush_render::bounding_box::BoundingBox;
    use brush_train::{config::TrainConfig, train::SplatTrainer};
    use burn::{module::AutodiffModule, tensor::Device};
    use glam::Vec3;
    use wasm_bindgen_test::wasm_bindgen_test;

    use crate::benches::{
        ITERS_PER_SYNC, backward_iters, bench_camera, forward_iters, gen_splats,
        generate_training_batch, training_iters,
    };

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    /// Run forward rendering loop. Generates splats once, then renders `iters` times.
    async fn run_forward_render(
        device: &Device,
        splat_count: usize,
        resolution: (u32, u32),
        iters: u32,
    ) {
        let splats = gen_splats(device, splat_count).valid();
        let camera = bench_camera();
        forward_iters(&splats, &camera, resolution, iters).await;
    }

    /// Run backward rendering loop. Generates splats once, then renders+backward `iters` times.
    async fn run_backward_render(
        device: &Device,
        splat_count: usize,
        resolution: (u32, u32),
        iters: u32,
    ) {
        let splats = gen_splats(device, splat_count);
        let camera = bench_camera();
        backward_iters(&splats, &camera, resolution, iters).await;
    }

    /// Run training loop. Generates splats once, then trains `iters` steps.
    async fn run_training_steps(
        device: &Device,
        splat_count: usize,
        resolution: (u32, u32),
        iters: u32,
    ) {
        let batches = [
            generate_training_batch(resolution, Vec3::new(0.0, 0.0, 5.0)),
            generate_training_batch(resolution, Vec3::new(2.0, 0.0, 5.0)),
        ];
        let config = TrainConfig::default();
        let mut trainer = SplatTrainer::new(
            &config,
            device,
            BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
        );
        let mut splats = Some(gen_splats(device, splat_count));
        training_iters(&mut trainer, &mut splats, &batches, 0, iters).await;
        let splats = splats.expect("splats always put back");
        assert!(splats.num_splats() > 0, "Failed smoke test");
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_fwd_render() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        run_forward_render(&device, 500_000, (1920, 1080), ITERS_PER_SYNC).await;
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_bwd_render() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        run_backward_render(&device, 500_000, (1920, 1080), ITERS_PER_SYNC).await;
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_training() {
        let device =
            burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        run_training_steps(&device, 500_000, (1920, 1080), ITERS_PER_SYNC).await;
    }
}
