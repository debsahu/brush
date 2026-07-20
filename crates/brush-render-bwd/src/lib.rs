pub mod burn_glue;
mod features_bwd;
mod kernels;
mod render_bwd;

pub use burn_glue::{
    DeferredShGrad, DeferredShGradHandle, DeferredSplatGrads, RasterizeGrads, SplatBwdOps,
    SplatGrads, SplatOutputDiff, TrainingSplatOutputDiff, render_splats,
    render_splats_for_training, render_splats_with_pass, render_splats_with_pass_and_rasterizer,
    render_splats_with_refine_weight,
};
pub use features_bwd::render_splat_features;
