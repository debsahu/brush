#![recursion_limit = "256"]

/// Shared by the two checkpoint binaries, which are separate targets.
pub mod checkpoint;

#[cfg(test)]
mod reference;

#[cfg(test)]
mod safetensor_utils;

#[cfg(test)]
mod benches;
