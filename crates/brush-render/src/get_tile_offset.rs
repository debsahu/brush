use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::frontend::IndexMutExpand;
use burn_cubecl::cubecl::prelude::*;

#[doc(hidden)]
pub const CHECKS_PER_ITER: u32 = 8;

#[cube(launch)]
pub fn get_tile_offsets(
    num_inter: u32,
    num_tiles: u32,
    tile_id_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
) {
    // Compute linear position from 2D dispatch (for large dispatches that exceed 65535 workgroups)
    let workgroup_id = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    // Adjacent lanes read adjacent intersections on every unrolled iteration.
    // Each workgroup still covers one contiguous CUBE_DIM_X * CHECKS_PER_ITER span.
    let workgroup_base = workgroup_id * CUBE_DIM_X * CHECKS_PER_ITER;
    let base_id = workgroup_base + UNIT_POS;

    // `tile_id_from_isect` can contain the sentinel `num_tiles` produced by
    // `map_gaussians_to_intersect` whenever its predicate yields fewer hits
    // than PF reserved (separate optimisation passes). `tile_offsets` is sized
    // for valid tiles only, so we must gate every write on `tid < num_tiles` to
    // avoid stomping the slot one past the end.
    #[unroll]
    for i in 0..CHECKS_PER_ITER {
        let isect_id = base_id + i * CUBE_DIM_X;

        if isect_id < num_inter {
            let tid = tile_id_from_isect[isect_id as usize];

            if isect_id == 0 {
                if tid < num_tiles {
                    // First intersection: always write the start of its tile.
                    tile_offsets[tid as usize * 2] = 0;
                }
            } else {
                let prev_tid = tile_id_from_isect[isect_id as usize - 1];
                if tid != prev_tid {
                    if prev_tid < num_tiles {
                        // Write the end of the previous tile, including when
                        // the current row is the out-of-range sentinel.
                        tile_offsets[prev_tid as usize * 2 + 1] = isect_id;
                    }
                    if tid < num_tiles {
                        // Write the start of the current valid tile.
                        tile_offsets[tid as usize * 2] = isect_id;
                    }
                }
            }

            if isect_id == num_inter - 1 && tid < num_tiles {
                // Write the end of the last tile.
                tile_offsets[tid as usize * 2 + 1] = isect_id + 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CHECKS_PER_ITER, get_tile_offsets};
    use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor_from_slice};
    use burn::backend::ops::IntTensorOps;
    use burn::tensor::DType;
    use burn_cubecl::cubecl::CubeDim;
    use burn_wgpu::{CubeTensor, WgpuRuntime};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn read_u32(tensor: CubeTensor<WgpuRuntime>) -> Vec<u32> {
        let data = MainBackendBase::int_into_data(tensor)
            .await
            .expect("readback");
        data.as_slice::<i32>()
            .expect("wrong type")
            .iter()
            .map(|value| *value as u32)
            .collect()
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn coalesced_walk_finds_runs_across_workgroup_boundaries() {
        const NUM_TILES: u32 = 8;
        let run_lengths = [31usize, 1, 223, 1, 1_791, 1, 17, 0];
        let mut tile_ids = Vec::new();
        let mut expected = vec![0u32; NUM_TILES as usize * 2];

        for (tile_id, run_len) in run_lengths.into_iter().enumerate() {
            if run_len == 0 {
                continue;
            }
            let start = tile_ids.len() as u32;
            tile_ids.extend(std::iter::repeat_n(tile_id as u32, run_len));
            let end = tile_ids.len() as u32;
            expected[tile_id * 2] = start;
            expected[tile_id * 2 + 1] = end;
        }

        // Sentinel rows are sorted after every valid tile and must not write
        // past the end of the offsets tensor. Keep a partial final workgroup.
        tile_ids.extend(std::iter::repeat_n(NUM_TILES, 13));
        let num_inter = tile_ids.len() as u32;

        let device = brush_cube::test_helpers::test_device().await;
        let tile_ids = create_tensor_from_slice(&tile_ids, &device, DType::I32);
        let offsets =
            create_tensor_from_slice(&vec![0u32; NUM_TILES as usize * 2], &device, DType::I32);
        let client = tile_ids.client.clone();
        let cube_dim = CubeDim::new_1d(256);

        get_tile_offsets::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_inter, cube_dim.x * CHECKS_PER_ITER),
            cube_dim,
            num_inter,
            NUM_TILES,
            tile_ids.into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
        );

        assert_eq!(read_u32(offsets).await, expected);
    }
}
