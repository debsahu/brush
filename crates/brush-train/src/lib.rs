#![recursion_limit = "256"]

pub mod config;
pub mod dig;
pub mod eval;
pub mod lod;
pub mod msg;
pub mod tidi;
pub mod train;
pub mod walls;

mod adam_scaled;
mod consensus;
mod edge;
mod error_map;
mod min_scale;
mod multinomial;
mod quat_vec;
#[cfg(all(
    feature = "native-msl",
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
mod sh_adam;
mod stats;

mod splat_init;

pub use splat_init::{RandomSplatsConfig, create_random_splats, to_init_splats};
