import time

import mlx.core as mx
import numpy as np

from compute.fragments import CodegenContext
from compute.reduction import (
    LastAxisReductionComputeFragment,
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
            op, "X", "tg_reduce", "N", "N", "single_reduce", TG_X
        ),
        StoreReductionResultFragment("Out", "tg_reduce", "single_store", TG_X),
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


def bench_case(op: str, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(shape[0] * 65537 + shape[1] * 17 + len(op))
    low, high = (0.95, 1.05) if op == "product" else (-1.0, 1.0)
    x = rng.uniform(low, high, size=shape).astype(np.float32)
    ours, ours_ms = run_ours_single(op, x, timed=True)
    mlx_out, mlx_time = mlx_ms(op, x)
    assert_close(ours, mlx_out, op)
    print(f"op={op} shape={shape} ours={ours_ms:.4f}ms mlx={mlx_time:.4f}ms")


def bench_reductions() -> None:
    for op in REDUCTION_CASES:
        bench_case(op, (1024, 1024))


if __name__ == "__main__":
    test_single_threadgroup_reductions()
    bench_reductions()
