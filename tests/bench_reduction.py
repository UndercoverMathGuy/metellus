import time

import mlx.core as mx
import numpy as np

from compute.fragments import CodegenContext
from compute.reduction import (
    LastAxisReductionComputeFragment,
    LastAxisReductionPartialStoreFragment,
    LastAxisReductionSetupFragment,
    StoreReductionResultFragment,
)
from runtime import Allocate, Dispatch, Download, Fill, FromNumpy, Kernel, Runtime


WARMUP = 2
ITERS = 5
TG_X = 256

REDUCTION_CASES = {
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "product": np.prod,
}

REDUCTION_MLX = {
    "sum": mx.sum,
    "max": mx.max,
    "min": mx.min,
    "product": mx.prod,
}


def _scratch_decl(name: str) -> str:
    return f"threadgroup float {name}[{(TG_X + 31) // 32}];"


def _single_kernel(op: str) -> Kernel:
    fragments = (
        LastAxisReductionSetupFragment("M", "single_setup"),
        LastAxisReductionComputeFragment(
            op, "X", "tg_reduce", "N", "N", "single_reduce"
        ),
        StoreReductionResultFragment("Out", "tg_reduce", "single_store"),
    )
    ctx = CodegenContext(
        function_name="reduce_last_axis_kernel",
        buffers=(
            "device const float* X [[buffer(0)]]",
            "device float* Out [[buffer(1)]]",
        ),
        dims=("M", "N"),
        tg_x=TG_X,
        tg_y=1,
        threadgroup_decls=(_scratch_decl("tg_reduce"),),
    )
    return Kernel(fragments=fragments, ctx=ctx)


def _tree_kernels(op: str) -> tuple[Kernel, Kernel]:
    partial_fragments = (
        LastAxisReductionSetupFragment("M", "partial_setup", tree=True),
        LastAxisReductionComputeFragment(
            op, "X", "tg_reduce", "N", "N", "partial_reduce", block_dim="block_N"
        ),
        LastAxisReductionPartialStoreFragment(
            "P", "tg_reduce", "num_blocks", "partial_store"
        ),
    )
    final_fragments = (
        LastAxisReductionSetupFragment("M", "final_setup"),
        LastAxisReductionComputeFragment(
            op, "P", "tg_reduce", "num_blocks", "num_blocks", "final_reduce"
        ),
        StoreReductionResultFragment("Out", "tg_reduce", "final_store"),
    )
    partial_ctx = CodegenContext(
        function_name="reduce_last_axis_partial_kernel",
        buffers=(
            "device const float* X [[buffer(0)]]",
            "device float* P [[buffer(1)]]",
        ),
        dims=("M", "N", "block_N", "num_blocks"),
        tg_x=TG_X,
        tg_y=1,
        threadgroup_decls=(_scratch_decl("tg_reduce"),),
    )
    final_ctx = CodegenContext(
        function_name="reduce_last_axis_final_kernel",
        buffers=(
            "device const float* P [[buffer(0)]]",
            "device float* Out [[buffer(1)]]",
        ),
        dims=("M", "N", "block_N", "num_blocks"),
        tg_x=TG_X,
        tg_y=1,
        threadgroup_decls=(_scratch_decl("tg_reduce"),),
        dims_buffer_index=2,
    )
    return (
        Kernel(fragments=partial_fragments, ctx=partial_ctx),
        Kernel(fragments=final_fragments, ctx=final_ctx),
    )


def run_ours_single(
    op: str, x: np.ndarray, timed: bool = False
) -> tuple[np.ndarray, float]:
    m, n = x.shape
    kernel = _single_kernel(op)
    env = Runtime(
        (
            FromNumpy("X", x),
            Allocate("Out", m * 4),
            kernel,
        )
    ).run()
    inner = Runtime(
        (
            Fill("Out", 0),
            Dispatch(
                kernel,
                bindings=("X", "Out"),
                dims=(m, n),
                grid=(m, 1, 1),
                threads=(TG_X, 1, 1),
                time_key="t",
            ),
        )
    )
    for _ in range(WARMUP if timed else 0):
        inner.run(env)
    times = []
    for _ in range(ITERS if timed else 1):
        inner.run(env)
        times.append(env["t"])
    Runtime((Download("Out", shape=(m,), dtype=np.float32, into="out"),)).run(env)
    return env["out"], float(np.mean(times))


def run_ours_tree(
    op: str, x: np.ndarray, block_n: int = 1024, timed: bool = False
) -> tuple[np.ndarray, float]:
    m, n = x.shape
    num_blocks = (n + block_n - 1) // block_n
    partial_kernel, final_kernel = _tree_kernels(op)
    env = Runtime(
        (
            FromNumpy("X", x),
            Allocate("P", m * num_blocks * 4),
            Allocate("Out", m * 4),
            partial_kernel,
            final_kernel,
        )
    ).run()

    dims = (m, n, block_n, num_blocks)
    inner = Runtime(
        (
            Fill("P", 0),
            Fill("Out", 0),
            Dispatch(
                partial_kernel,
                bindings=("X", "P"),
                dims=dims,
                grid=(num_blocks, m, 1),
                threads=(TG_X, 1, 1),
                time_key="t_partial",
            ),
            Dispatch(
                final_kernel,
                bindings=("P", "Out"),
                dims=dims,
                grid=(m, 1, 1),
                threads=(TG_X, 1, 1),
                time_key="t_final",
            ),
        )
    )

    for _ in range(WARMUP if timed else 0):
        inner.run(env)
    times = []
    for _ in range(ITERS if timed else 1):
        inner.run(env)
        times.append(env["t_partial"] + env["t_final"])
    Runtime((Download("Out", shape=(m,), dtype=np.float32, into="out"),)).run(env)
    return env["out"], float(np.mean(times))


def mlx_ms(op: str, x: np.ndarray) -> tuple[np.ndarray, float]:
    x_mx = mx.array(x)
    mx.eval(x_mx)

    for _ in range(WARMUP):
        mx.eval(REDUCTION_MLX[op](x_mx, axis=-1))

    times = []
    out = None
    for _ in range(ITERS):
        start = time.perf_counter()
        out = REDUCTION_MLX[op](x_mx, axis=-1)
        mx.eval(out)
        times.append((time.perf_counter() - start) * 1000)
    return np.array(out).astype(np.float32), float(np.mean(times))


def assert_close(actual: np.ndarray, expected: np.ndarray, op: str) -> None:
    rtol = 1e-4 if op in ("sum", "product") else 1e-5
    atol = 1e-3 if op in ("sum", "product") else 1e-5
    np.testing.assert_allclose(
        actual, expected.astype(np.float32), rtol=rtol, atol=atol
    )


def test_single_threadgroup_reductions() -> None:
    rng = np.random.default_rng(23)
    x = rng.uniform(0.75, 1.25, size=(17, 257)).astype(np.float32)
    for op, fn in REDUCTION_CASES.items():
        out, _ = run_ours_single(op, x)
        assert_close(out, fn(x, axis=-1), op)


def test_tree_reductions() -> None:
    rng = np.random.default_rng(29)
    x = rng.uniform(0.95, 1.05, size=(13, 4097)).astype(np.float32)
    for op, fn in REDUCTION_CASES.items():
        out, _ = run_ours_tree(op, x, block_n=512)
        assert_close(out, fn(x, axis=-1), op)


def bench_case(op: str, shape: tuple[int, int], tree: bool = False) -> None:
    rng = np.random.default_rng(shape[0] * 65537 + shape[1] * 17 + len(op))
    low, high = (0.95, 1.05) if op == "product" else (-1.0, 1.0)
    x = rng.uniform(low, high, size=shape).astype(np.float32)
    ours, ours_ms = (
        run_ours_tree(op, x, timed=True) if tree else run_ours_single(op, x, timed=True)
    )
    mlx_out, mlx_time = mlx_ms(op, x)
    assert_close(ours, mlx_out, op)
    mode = "tree" if tree else "single"
    print(
        f"mode={mode} op={op} shape={shape} ours={ours_ms:.4f}ms mlx={mlx_time:.4f}ms"
    )


def bench_reductions() -> None:
    for op in REDUCTION_CASES:
        bench_case(op, (1024, 1024), tree=False)
        bench_case(op, (512, 4096), tree=True)


if __name__ == "__main__":
    test_single_threadgroup_reductions()
    test_tree_reductions()
    bench_reductions()
