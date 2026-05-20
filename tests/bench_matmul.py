import time

import mlx.core as mx
import numpy as np

from compute.fragments import BarrierFragment, CodegenContext
from compute.matmul import (
    MatmulAccumToDevFragment,
    MatmulAccumToTgFragment,
    MatmulComputeFragment,
    MatmulConfig,
    MatmulMainloopFragment,
    MatmulSetupFragment,
    MatmulTgToDevFragment,
    MatmulTileMappingFragment,
    SplitKComputeFragment,
    SplitKPartialStoreFragment,
    SplitKReduceComputeFragment,
    SplitKReduceStoreFragment,
    SplitKSetupFragment,
    ThreadIndexFragment,
)
from compute.matmul.config import (
    SplitKConfig,
    ceil_div,
    grid_for,
    is_aligned_shape,
    select_tile_config,
    should_use_splitk,
)
from memory import TgLoadFragment
from runtime import Allocate, Dispatch, Download, Fill, FromNumpy, Kernel, Runtime


WARMUP = 2
ITERS = 5


def mlx_matmul_ms(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    a_mx = mx.array(a)
    b_mx = mx.array(b)
    mx.eval(a_mx, b_mx)

    for _ in range(WARMUP):
        mx.eval(a_mx @ b_mx)

    times = []
    out = None
    for _ in range(ITERS):
        start = time.perf_counter()
        out = a_mx @ b_mx
        mx.eval(out)
        times.append((time.perf_counter() - start) * 1000)

    return np.array(out), float(np.mean(times))


def _splitk_kernels(config: SplitKConfig) -> tuple[Kernel, Kernel]:
    compute_fragments = (
        SplitKSetupFragment(config, "M", "N", "parts"),
        SplitKComputeFragment("A", "B", "K", "N", "part_K"),
        SplitKPartialStoreFragment("P", "M", "N"),
    )
    reduce_fragments = (
        SplitKSetupFragment(config, "M", "N", "parts", reduce=True),
        SplitKReduceComputeFragment("P", "M", "N", "parts"),
        SplitKReduceStoreFragment("C", "N"),
    )
    compute_ctx = CodegenContext(
        function_name="splitk_matmul_kernel",
        buffers=(
            "device const float* A [[buffer(0)]]",
            "device const float* B [[buffer(1)]]",
            "device float* P [[buffer(2)]]",
        ),
        dims=("M", "K", "N", "part_K", "parts"),
        tg_x=config.block_N,
        tg_y=config.block_M,
        position_type="uint3",
        thread_type="uint3",
    )
    reduce_ctx = CodegenContext(
        function_name="splitk_reduce_kernel",
        buffers=(
            "device const float* P [[buffer(0)]]",
            "device float* C [[buffer(1)]]",
        ),
        dims=("M", "K", "N", "part_K", "parts"),
        tg_x=config.block_N,
        tg_y=config.block_M,
        dims_buffer_index=2,
    )
    return (
        Kernel(fragments=compute_fragments, ctx=compute_ctx),
        Kernel(fragments=reduce_fragments, ctx=reduce_ctx),
    )


def _gemm_kernel(m: int, k: int, n: int) -> tuple[Kernel, object, bool]:
    tile = select_tile_config(m, k, n)
    aligned = is_aligned_shape(m, k, n, tile)
    config = MatmulConfig(
        tile, aligned, "M", "K", "N", "A", "B", "C", "A_tile", "B_tile", "C_tile"
    )
    row_limit = None if aligned else "M"
    k_limit = None if aligned else "K"
    n_limit = None if aligned else "N"
    c_tile_decl = (
        ()
        if aligned
        else (f"threadgroup float C_tile[{tile.tile_M}][{tile.tile_N + tile.c_pad}];",)
    )
    fragments = (
        MatmulTileMappingFragment(),
        ThreadIndexFragment(tile),
        MatmulSetupFragment(config),
        MatmulMainloopFragment(
            tile_K=tile.tile_K,
            fragments=(
                TgLoadFragment(
                    name="load_A_tile",
                    src_name="A",
                    src_row_stride="K",
                    row_start=f"tg.y * {tile.tile_M}",
                    col_start="k_chunk",
                    dst_name="A_tile",
                    tile_shape=(tile.tile_M, tile.tile_K),
                    num_threads=tile.num_threads,
                    row_limit=row_limit,
                    col_limit=k_limit,
                ),
                TgLoadFragment(
                    name="load_B_tile",
                    src_name="B",
                    src_row_stride="N",
                    row_start="k_chunk",
                    col_start=f"tg.x * {tile.tile_N}",
                    dst_name="B_tile",
                    tile_shape=(tile.tile_K, tile.tile_N),
                    num_threads=tile.num_threads,
                    row_limit=k_limit,
                    col_limit=n_limit,
                ),
                BarrierFragment("inputs_ready"),
                MatmulComputeFragment(config),
                BarrierFragment("compute_done"),
            ),
            K_dim_var="K",
        ),
        *(
            (MatmulAccumToDevFragment(config),)
            if aligned
            else (
                MatmulAccumToTgFragment(config),
                BarrierFragment("c_tile_ready"),
                MatmulTgToDevFragment(config),
            )
        ),
    )
    ctx = CodegenContext(
        function_name="matmul_kernel",
        buffers=(
            "device const float* A [[buffer(0)]]",
            "device const float* B [[buffer(1)]]",
            "device float* C [[buffer(2)]]",
        ),
        dims=("M", "K", "N"),
        tg_x=tile.tg_x,
        tg_y=tile.tg_y,
        threadgroup_decls=(
            f"threadgroup float A_tile[{tile.tile_M}][{tile.tile_K + tile.a_pad}];",
            f"threadgroup float B_tile[{tile.tile_K}][{tile.tile_N + tile.b_pad}];",
            *c_tile_decl,
        ),
    )
    return Kernel(fragments=fragments, ctx=ctx), tile, aligned


def ours_matmul_ms(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, object]:
    m, k = a.shape
    _, n = b.shape

    if should_use_splitk(m, k, n):
        config = SplitKConfig()
        parts = ceil_div(k, config.part_K)
        compute_kernel, reduce_kernel = _splitk_kernels(config)
        env = Runtime(
            (
                FromNumpy("A", a),
                FromNumpy("B", b),
                Allocate("P", parts * m * n * 4),
                Allocate("C", m * n * 4),
                compute_kernel,
                reduce_kernel,
            )
        ).run()
        dims = (m, k, n, config.part_K, parts)
        grid_compute = (ceil_div(n, config.block_N), ceil_div(m, config.block_M), parts)
        grid_reduce = (ceil_div(n, config.block_N), ceil_div(m, config.block_M), 1)
        threads = (config.block_N, config.block_M, 1)
        inner = Runtime(
            (
                Fill("P", 0),
                Fill("C", 0),
                Dispatch(
                    compute_kernel,
                    bindings=("A", "B", "P"),
                    dims=dims,
                    grid=grid_compute,
                    threads=threads,
                    time_key="t1",
                ),
                Dispatch(
                    reduce_kernel,
                    bindings=("P", "C"),
                    dims=dims,
                    grid=grid_reduce,
                    threads=threads,
                    time_key="t2",
                ),
            )
        )
        for _ in range(WARMUP):
            inner.run(env)
        times = []
        for _ in range(ITERS):
            inner.run(env)
            times.append(env["t1"] + env["t2"])
        Runtime((Download("C", shape=(m, n), dtype=np.float32, into="C_out"),)).run(env)
        spec = {
            "method": "splitk",
            "config_label": f"{config.block_M}x{config.block_N}",
            "aligned": False,
            "tile_order": "row_major",
        }
        return env["C_out"], float(np.mean(times)), spec

    kernel, tile, aligned = _gemm_kernel(m, k, n)
    env = Runtime(
        (
            FromNumpy("A", a),
            FromNumpy("B", b),
            Allocate("C", m * n * 4),
            kernel,
        )
    ).run()

    grid = grid_for(m, n, tile)
    threads = (tile.tg_x, tile.tg_y, 1)

    inner = Runtime(
        (
            Fill("C", 0),
            Dispatch(
                kernel,
                bindings=("A", "B", "C"),
                dims=(m, k, n),
                grid=grid,
                threads=threads,
                time_key="t",
            ),
        )
    )

    for _ in range(WARMUP):
        inner.run(env)
    times = []

    for _ in range(ITERS):
        inner.run(env)
        times.append(env["t"])

    Runtime((Download("C", shape=(m, n), dtype=np.float32, into="C_out"),)).run(env)
    spec = {
        "method": "gemm",
        "config_label": f"{tile.tile_M}x{tile.tile_N}x{tile.tile_K}",
        "aligned": aligned,
        "tile_order": "row_major",
    }
    return env["C_out"], float(np.mean(times)), spec


def check_shape(m: int, k: int, n: int) -> tuple[float, float, float, float, object]:
    rng = np.random.default_rng(m * 1_000_003 + k * 9_176 + n)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    ours, ours_ms, plan = ours_matmul_ms(a, b)
    mlx_out, mlx_ms = mlx_matmul_ms(a, b)

    max_abs = float(np.max(np.abs(ours - mlx_out)))
    mean_abs = float(np.mean(np.abs(ours - mlx_out)))
    np.testing.assert_allclose(ours, mlx_out, rtol=1e-3, atol=1e-2)
    return ours_ms, mlx_ms, max_abs, mean_abs, plan


def main() -> None:
    shapes = [
        (512, 512, 512),
        (1024, 64, 1024),
        (16, 4096, 16),
        (17, 19, 23),
        (63, 31, 65),
        (64, 33, 64),
        (65, 64, 17),
        (127, 129, 131),
        (1, 1, 1),
        (3, 70, 5),
        (130, 7, 96),
        (4096, 512, 4096),
        (8192, 64, 8192),
    ]

    for shape in shapes:
        ours_ms, mlx_ms, max_abs, mean_abs, spec = check_shape(*shape)
        print(
            f"shape={shape} method={spec['method']} config={spec['config_label']} "
            f"aligned={spec['aligned']} order={spec['tile_order']} "
            f"ours={ours_ms:.3f}ms mlx={mlx_ms:.3f}ms "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
        )


if __name__ == "__main__":
    main()
