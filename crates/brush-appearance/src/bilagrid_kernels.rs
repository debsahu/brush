//! `CubeCL` implementation of per-view affine bilateral-grid slicing.
//!
//! The behavior follows the Apache-2.0 gsplat bilateral-grid reference:
//! trilinear sampling with aligned corners and border padding, using BT.601
//! luminance as the guidance coordinate. This implementation is written for
//! Brush's HWC tensors and custom autodiff interface.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use crate::AtomicAddF32;
use brush_cube::is_finite_f32;

const LUMA_R: f32 = 0.299;
const LUMA_G: f32 = 0.587;
const LUMA_B: f32 = 0.114;

pub const BLOCK_SIZE: u32 = 256;

#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
struct SamplePoint {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    z0: u32,
    z1: u32,
    tx: f32,
    ty: f32,
    tz: f32,
    guidance_active: bool,
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn sample_point(
    pixel_x: u32,
    pixel_y: u32,
    red: f32,
    green: f32,
    blue: f32,
    grid_l: u32,
    grid_h: u32,
    grid_w: u32,
    image_h: u32,
    image_w: u32,
) -> SamplePoint {
    let x = f32::cast_from(pixel_x) * f32::cast_from(grid_w - 1)
        / f32::cast_from(max(image_w - 1, 1u32));
    let y = f32::cast_from(pixel_y) * f32::cast_from(grid_h - 1)
        / f32::cast_from(max(image_h - 1, 1u32));
    let raw_z = (LUMA_R * red + LUMA_G * green + LUMA_B * blue) * f32::cast_from(grid_l - 1);
    let z = clamp(raw_z, 0.0f32, f32::cast_from(grid_l - 1));

    let x0 = u32::cast_from(f32::floor(x));
    let y0 = u32::cast_from(f32::floor(y));
    let z0 = u32::cast_from(f32::floor(z));
    SamplePoint {
        x0,
        x1: min(x0 + 1, grid_w - 1),
        y0,
        y1: min(y0 + 1, grid_h - 1),
        z0,
        z1: min(z0 + 1, grid_l - 1),
        tx: x - f32::floor(x),
        ty: y - f32::floor(y),
        tz: z - f32::floor(z),
        guidance_active: raw_z > 0.0f32 && raw_z < f32::cast_from(grid_l - 1),
    }
}

#[cube]
fn corner_x(point: SamplePoint, corner: u32) -> u32 {
    select((corner & 1u32) == 0u32, point.x0, point.x1)
}

#[cube]
fn corner_y(point: SamplePoint, corner: u32) -> u32 {
    select((corner & 2u32) == 0u32, point.y0, point.y1)
}

#[cube]
fn corner_z(point: SamplePoint, corner: u32) -> u32 {
    select((corner & 4u32) == 0u32, point.z0, point.z1)
}

#[cube]
fn corner_weight(point: SamplePoint, corner: u32) -> f32 {
    let wx = select((corner & 1u32) == 0u32, 1.0f32 - point.tx, point.tx);
    let wy = select((corner & 2u32) == 0u32, 1.0f32 - point.ty, point.ty);
    let wz = select((corner & 4u32) == 0u32, 1.0f32 - point.tz, point.tz);
    wx * wy * wz
}

#[cube]
fn corner_xy_weight(point: SamplePoint, corner: u32) -> f32 {
    let wx = select((corner & 1u32) == 0u32, 1.0f32 - point.tx, point.tx);
    let wy = select((corner & 2u32) == 0u32, 1.0f32 - point.ty, point.ty);
    wx * wy
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn grid_index(
    grid_offset: u32,
    coefficient: u32,
    z: u32,
    y: u32,
    x: u32,
    grid_l: u32,
    grid_h: u32,
    grid_w: u32,
) -> usize {
    let cells = grid_l * grid_h * grid_w;
    (grid_offset + coefficient * cells + (z * grid_h + y) * grid_w + x) as usize
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn interpolate(
    grid: &Tensor<f32>,
    point: SamplePoint,
    coefficient: u32,
    grid_offset: u32,
    grid_l: u32,
    grid_h: u32,
    grid_w: u32,
) -> f32 {
    let mut value = 0.0f32;
    #[unroll]
    for corner in 0u32..8u32 {
        let index = grid_index(
            grid_offset,
            coefficient,
            corner_z(point, corner),
            corner_y(point, corner),
            corner_x(point, corner),
            grid_l,
            grid_h,
            grid_w,
        );
        value += grid[index] * corner_weight(point, corner);
    }
    value
}

#[cube]
fn color_component(red: f32, green: f32, blue: f32, column: u32) -> f32 {
    select(
        column == 0u32,
        red,
        select(column == 1u32, green, select(column == 2u32, blue, 1.0f32)),
    )
}

#[cube]
fn output_gradient(v_out: &Tensor<f32>, base: usize, row: u32) -> f32 {
    let value = v_out[base + row as usize];
    select(is_finite_f32(value), value, 0.0f32)
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn bilagrid_slice_fwd_kernel(
    grid: &Tensor<f32>,
    rgb: &Tensor<f32>,
    out: &mut Tensor<f32>,
    grid_l: u32,
    grid_h: u32,
    grid_w: u32,
    image_h: u32,
    image_w: u32,
    grid_offset: u32,
    channels: u32,
    #[comptime] has_alpha: bool,
) {
    let pixel = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;
    if pixel >= image_h * image_w {
        terminate!();
    }
    let base = (pixel * channels) as usize;
    let red = rgb[base];
    let green = rgb[base + 1];
    let blue = rgb[base + 2];
    let point = sample_point(
        pixel % image_w,
        pixel / image_w,
        red,
        green,
        blue,
        grid_l,
        grid_h,
        grid_w,
        image_h,
        image_w,
    );

    #[unroll]
    for row in 0u32..3u32 {
        let mut value = 0.0f32;
        #[unroll]
        for column in 0u32..4u32 {
            let coefficient = row * 4 + column;
            value += interpolate(
                grid,
                point,
                coefficient,
                grid_offset,
                grid_l,
                grid_h,
                grid_w,
            ) * color_component(red, green, blue, column);
        }
        out[base + row as usize] = select(is_finite_f32(value), value, 0.5f32);
    }
    if has_alpha {
        out[base + 3] = rgb[base + 3];
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_ref_mut)]
pub fn bilagrid_slice_bwd_kernel<A: AtomicAddF32>(
    grid: &Tensor<f32>,
    rgb: &Tensor<f32>,
    v_out: &Tensor<f32>,
    grad_grid: &mut Tensor<Atomic<A::Storage>>,
    grad_rgb: &mut Tensor<f32>,
    grid_l: u32,
    grid_h: u32,
    grid_w: u32,
    image_h: u32,
    image_w: u32,
    grid_offset: u32,
    channels: u32,
    #[comptime] has_alpha: bool,
) {
    let pixel = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;
    if pixel >= image_h * image_w {
        terminate!();
    }
    let base = (pixel * channels) as usize;
    let red = rgb[base];
    let green = rgb[base + 1];
    let blue = rgb[base + 2];
    let point = sample_point(
        pixel % image_w,
        pixel / image_w,
        red,
        green,
        blue,
        grid_l,
        grid_h,
        grid_w,
        image_h,
        image_w,
    );

    let mut grad_red = 0.0f32;
    let mut grad_green = 0.0f32;
    let mut grad_blue = 0.0f32;
    let mut grad_guidance = 0.0f32;

    #[unroll]
    for row in 0u32..3u32 {
        let upstream = output_gradient(v_out, base, row);
        #[unroll]
        for column in 0u32..4u32 {
            let coefficient = row * 4 + column;
            let input = color_component(red, green, blue, column);
            let sampled = interpolate(
                grid,
                point,
                coefficient,
                grid_offset,
                grid_l,
                grid_h,
                grid_w,
            );
            grad_red += select(column == 0u32, upstream * sampled, 0.0f32);
            grad_green += select(column == 1u32, upstream * sampled, 0.0f32);
            grad_blue += select(column == 2u32, upstream * sampled, 0.0f32);

            let grad_coefficient = upstream * input;
            let mut lower = 0.0f32;
            let mut upper = 0.0f32;
            #[unroll]
            for corner in 0u32..8u32 {
                let z = corner_z(point, corner);
                let index = grid_index(
                    grid_offset,
                    coefficient,
                    z,
                    corner_y(point, corner),
                    corner_x(point, corner),
                    grid_l,
                    grid_h,
                    grid_w,
                );
                let weight = corner_weight(point, corner);
                A::add(&grad_grid[index], grad_coefficient * weight);

                // Bilinear x/y contribution on each guidance plane.
                let xy_weight = corner_xy_weight(point, corner);
                lower += select(z == point.z0, grid[index] * xy_weight, 0.0f32);
                upper += select(z == point.z1, grid[index] * xy_weight, 0.0f32);
            }
            grad_guidance += upstream * input * (upper - lower);
        }
    }

    let guidance_scale = select(
        point.guidance_active,
        f32::cast_from(grid_l - 1) * grad_guidance,
        0.0f32,
    );
    grad_rgb[base] = grad_red + LUMA_R * guidance_scale;
    grad_rgb[base + 1] = grad_green + LUMA_G * guidance_scale;
    grad_rgb[base + 2] = grad_blue + LUMA_B * guidance_scale;
    if has_alpha {
        grad_rgb[base + 3] = v_out[base + 3];
    }
}
