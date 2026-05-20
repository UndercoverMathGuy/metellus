import time

import numpy as np

from compute.fragments import CodegenContext
from compute.init import ArangeFragment, CopyFragment, FillFragment
from runtime import (
    Allocate,
    Dispatch,
    Download,
    FromNumpy,
    Kernel,
    Runtime,
)


TG_X = 256


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _init_ctx(function_name: str, buffer_decls: tuple[str, ...]) -> CodegenContext:
    return CodegenContext(
        function_name=function_name,
        buffers=buffer_decls,
        dims=("N",),
        tg_x=TG_X,
        tg_y=1,
        preamble=(
            "uint2 tg = threadgroup_position_in_grid;",
            "uint2 lid = thread_position_in_threadgroup;",
            f"uint flat_tid = lid.y * {TG_X} + lid.x;",
        ),
    )


def _grid(n: int) -> tuple[int, int, int]:
    return (_ceil_div(n, TG_X), 1, 1)


THREADS = (TG_X, 1, 1)


def test_fill() -> None:
    for n in (1, 256, 1000, 1024, 4097):
        kernel = Kernel(
            fragments=(FillFragment("Out", "N", value=3.5),),
            ctx=_init_ctx("fill_kernel", ("device float* Out [[buffer(0)]]",)),
        )
        env = Runtime(
            (
                Allocate("Out", n * 4),
                kernel,
                Dispatch(
                    kernel, bindings=("Out",), dims=(n,), grid=_grid(n), threads=THREADS
                ),
                Download("Out", shape=(n,), dtype=np.float32, into="out"),
            )
        ).run()
        np.testing.assert_array_equal(env["out"], np.full(n, 3.5, dtype=np.float32))


def test_arange() -> None:
    for n, start, step in [(1024, 0.0, 1.0), (1000, 5.0, 0.25), (4097, -2.0, 0.5)]:
        kernel = Kernel(
            fragments=(ArangeFragment("Out", "N", start=start, step=step),),
            ctx=_init_ctx("arange_kernel", ("device float* Out [[buffer(0)]]",)),
        )
        env = Runtime(
            (
                Allocate("Out", n * 4),
                kernel,
                Dispatch(
                    kernel, bindings=("Out",), dims=(n,), grid=_grid(n), threads=THREADS
                ),
                Download("Out", shape=(n,), dtype=np.float32, into="out"),
            )
        ).run()
        expected = (start + step * np.arange(n)).astype(np.float32)
        np.testing.assert_allclose(env["out"], expected, rtol=0, atol=1e-5)


def test_copy() -> None:
    for n in (1, 1024, 1000, 4097):
        src_np = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        kernel = Kernel(
            fragments=(CopyFragment("Out", "In", "N"),),
            ctx=_init_ctx(
                "copy_kernel",
                (
                    "device float* Out [[buffer(0)]]",
                    "device const float* In [[buffer(1)]]",
                ),
            ),
        )
        env = Runtime(
            (
                FromNumpy("In", src_np),
                Allocate("Out", n * 4),
                kernel,
                Dispatch(
                    kernel,
                    bindings=("Out", "In"),
                    dims=(n,),
                    grid=_grid(n),
                    threads=THREADS,
                ),
                Download("Out", shape=(n,), dtype=np.float32, into="out"),
            )
        ).run()
        np.testing.assert_array_equal(env["out"], src_np)


def bench_fill() -> None:
    n = 1 << 22  # 4M floats = 16 MiB
    kernel = Kernel(
        fragments=(FillFragment("Out", "N", value=1.0),),
        ctx=_init_ctx("fill_kernel", ("device float* Out [[buffer(0)]]",)),
    )
    env = Runtime((Allocate("Out", n * 4), kernel)).run()
    dispatch = Runtime(
        (
            Dispatch(
                kernel,
                bindings=("Out",),
                dims=(n,),
                grid=_grid(n),
                threads=THREADS,
                time_key="t",
            ),
        )
    )
    for _ in range(2):
        dispatch.run(env)
    times = []
    for _ in range(5):
        dispatch.run(env)
        times.append(env["t"])
    print(f"fill n={n} ours={np.mean(times):.4f}ms")

    blit_times = []
    out = env["Out"]
    for _ in range(5):
        t0 = time.perf_counter()
        out.fill(0)
        blit_times.append((time.perf_counter() - t0) * 1000)
    print(f"fill n={n} blit={np.mean(blit_times):.4f}ms (Buffer.fill for comparison)")


if __name__ == "__main__":
    test_fill()
    test_arange()
    test_copy()
    bench_fill()
