use brush_async::{Actor, AsyncMap};
use brush_process::slot::Slot;
use brush_render::{
    TextureMode, camera::Camera, gaussian_splats::Splats, render_splats, render_splats_depth,
};
use burn::tensor::{Tensor, s};
use egui::Rect;
use glam::{UVec2, Vec3};
use std::sync::Arc;

use eframe::egui_wgpu::{self, CallbackTrait, wgpu};

#[derive(Clone)]
struct RenderRequest {
    splats: Splats,
    ctx: egui::Context,
    state: LastRenderState,
}

#[derive(Clone, PartialEq)]
struct LastRenderState {
    frame: usize,
    camera: Camera,
    background: Vec3,
    splat_scale: Option<f32>,
    depth_view: bool,
    img_size: UVec2,
}

/// A rendered frame read back from the training device: packed RGBA8, or in
/// depth mode one f32 depth per pixel plus the range it is normalised over.
#[derive(Clone)]
struct Frame {
    width: u32,
    height: u32,
    pixels: Arc<Vec<u8>>,
    depth_range: Option<(f32, f32)>,
}

pub struct SplatBackbuffer {
    pipe: AsyncMap<RenderRequest, Frame>,
}

impl SplatBackbuffer {
    pub fn new(state: &eframe::egui_wgpu::RenderState) -> Self {
        // Keep blocking Metal readbacks off the training actor.
        let actor = Actor::new("splat-view");
        // Register splat backbuffer resources
        state
            .renderer
            .write()
            .callback_resources
            .insert(SplatBackbufferResources::new(
                &state.device,
                state.target_format,
            ));

        let pipe = AsyncMap::new(
            actor,
            async move |req: &RenderRequest| {
                let (image, depth_range) = if req.state.depth_view {
                    render_depth(req.splats.clone(), req).await
                } else {
                    render_rgba(req.splats.clone(), req).await
                };

                let shape = image.shape();
                let (height, width) = (shape[0] as u32, shape[1] as u32);

                let data = image
                    .into_data_async()
                    .await
                    .expect("Failed to read back frame");

                Frame {
                    width,
                    height,
                    pixels: Arc::new(data.into_bytes().to_vec()),
                    depth_range,
                }
            },
            |req: &RenderRequest| req.ctx.request_repaint(),
        );

        Self { pipe }
    }

    pub fn paint(
        &self,
        rect: Rect,
        ui: &egui::Ui,
        splats: &Slot<Splats>,
        camera: &Camera,
        frame: usize,
        background: Vec3,
        splat_scale: Option<f32>,
        depth_view: bool,
        splats_dirty: bool,
    ) -> Option<(f32, f32)> {
        if rect.width() <= 0.0 || rect.height() <= 0.0 {
            return None;
        }

        // Calculate pixel size for rendering
        let ppp = ui.ctx().pixels_per_point();
        let img_size = UVec2::new(
            (rect.width() * ppp).round() as u32,
            (rect.height() * ppp).round() as u32,
        );
        if img_size.x == 0 || img_size.y == 0 {
            return None;
        }

        // Check if we need to re-render
        let current_state = LastRenderState {
            frame,
            camera: *camera,
            background,
            splat_scale,
            depth_view,
            img_size,
        };

        let dirty = splats_dirty
            || self.pipe.last_request().map(|r| r.state) != Some(current_state.clone());

        if dirty && let Some(splats) = splats.get(frame) {
            self.pipe.request(RenderRequest {
                splats,
                ctx: ui.ctx().clone(),
                state: current_state,
            });
        }

        if let Some(frame) = self.pipe.latest() {
            let depth_range = frame.depth_range;
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    SplatBackbufferPainter { frame },
                ));
            depth_range
        } else {
            None
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    img_width: u32,
    img_height: u32,
    /// 0 = packed RGBA image, 1 = float32 depth map.
    mode: u32,
    _pad: u32,
    depth_min: f32,
    depth_max: f32,
    _pad2: [f32; 2],
}

pub struct SplatBackbufferResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    // Per-frame bind group - created in prepare() with the current tensor buffer
    bind_group: Option<wgpu::BindGroup>,
    upload_buffer: Option<wgpu::Buffer>,
}

impl SplatBackbufferResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Splat Backbuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/splat_backbuffer.wgsl").into()),
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat Backbuffer Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Splat Backbuffer Bind Group Layout"),
            entries: &[
                // Uniform buffer for image dimensions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Storage buffer for image data (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Splat Backbuffer Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Splat Backbuffer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers - using fullscreen triangle trick
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group: None,
            upload_buffer: None,
        }
    }

    fn reserve_upload_buffer(&mut self, device: &wgpu::Device, size: u64) {
        let fits = self
            .upload_buffer
            .as_ref()
            .is_some_and(|b| b.size() >= size);
        if !fits {
            self.upload_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Splat Backbuffer Upload Buffer"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }
}

async fn render_rgba(splats: Splats, req: &RenderRequest) -> (Tensor<3>, Option<(f32, f32)>) {
    let (image, _) = render_splats(
        splats,
        &req.state.camera,
        req.state.img_size,
        req.state.background,
        req.state.splat_scale,
        TextureMode::Packed,
    )
    .await;
    (image, None)
}

async fn render_depth(splats: Splats, req: &RenderRequest) -> (Tensor<3>, Option<(f32, f32)>) {
    let max_depth = 100.0;

    let (image, _) = render_splats_depth(
        splats,
        &req.state.camera,
        req.state.img_size,
        req.state.background,
        req.state.splat_scale,
    )
    .await;

    let accumulated_depth = image.clone().slice(s![.., .., 4..5]);
    let alpha = image.clone().slice(s![.., .., 3..4]);
    let depth = accumulated_depth / alpha.clamp_min(1e-10);

    let invalid = depth.clone().lower_equal_elem(0.0) | depth.clone().greater_elem(max_depth);
    let depth_min = depth
        .clone()
        .mask_fill(invalid.clone(), f32::INFINITY)
        .min()
        .into_scalar_async::<f32>()
        .await
        .expect("Failed to read depth min");
    let depth_max = depth
        .clone()
        .mask_fill(invalid, f32::NEG_INFINITY)
        .max()
        .into_scalar_async::<f32>()
        .await
        .expect("Failed to read depth max");

    let depth_range = if depth_min.is_finite() && depth_max.is_finite() && depth_max > depth_min {
        (depth_min, depth_max)
    } else {
        (0.0, max_depth)
    };

    (depth, Some(depth_range))
}

struct SplatBackbufferPainter {
    frame: Frame,
}

impl CallbackTrait for SplatBackbufferPainter {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = resources.get_mut::<SplatBackbufferResources>() else {
            return Vec::new();
        };

        // Update uniform buffer with image dimensions and display mode.
        let (mode, depth_min, depth_max) = match self.frame.depth_range {
            Some((min, max)) => (1, min, max),
            None => (0, 0.0, 1.0),
        };
        queue.write_buffer(
            &res.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                img_width: self.frame.width,
                img_height: self.frame.height,
                mode,
                _pad: 0,
                depth_min,
                depth_max,
                _pad2: [0.0; 2],
            }]),
        );

        res.reserve_upload_buffer(device, self.frame.pixels.len() as u64);
        let img_buffer = res.upload_buffer.as_ref().expect("just reserved");
        queue.write_buffer(img_buffer, 0, &self.frame.pixels);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Splat Backbuffer Bind Group"),
            layout: &res.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: img_buffer.as_entire_binding(),
                },
            ],
        });

        res.bind_group = Some(bind_group);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = callback_resources.get::<SplatBackbufferResources>() else {
            return;
        };

        let Some(bind_group) = res.bind_group.as_ref() else {
            return;
        };

        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
