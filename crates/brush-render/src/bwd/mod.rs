//! Backward (differentiable) render path.
//!
//! Lives in `brush-render` rather than a separate crate because the
//! `#[backend_extension]`-generated `Dispatch` impl calls the `Autodiff` arm,
//! so `impl SplatOps for Autodiff<..>` must be visible from the crate that
//! defines `SplatOps`.
pub mod burn_glue;
mod features_bwd;
mod kernels;
mod render_bwd;

pub use burn_glue::{
    DeferredShGrad, DeferredShGradHandle, SplatOutputDiff, TrainingSplatOutputDiff,
    lift_splats_to_autodiff, render_splats, render_splats_for_training, render_splats_with_pass,
    render_splats_with_pass_and_rasterizer, render_splats_with_refine_weight,
};
pub use features_bwd::render_splat_features;
/// Stride of the compact per-splat backward-gradient buffer (`v_combined`),
/// re-exported so downstream consumers of `DeferredShGrad::compact_grads` (e.g.
/// the sparse SH Adam optimizer in brush-train) index it with the exact lane
/// count the render backward writes, instead of a hand-copied literal that can
/// drift when a lane is added. See `kernels::rasterize_backwards`.
pub use kernels::rasterize_backwards::COMPACT_GRAD_LANES;
