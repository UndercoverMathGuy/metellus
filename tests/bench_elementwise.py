import time

import mlx.core as mx
import numpy as np

from compute.elementwise import ElementwiseComputeFragment, elementwise_arity
from compute.fragments import BarrierFragment, CodegenContext
from memory import TgLoadFragment, TgStoreFragment
from runtime import Allocate, Dispatch, Download, Fill, FromNumpy, Kernel, Runtime


WARMUP = 2
ITERS = 5

UNARY_CASES = {
    "negate": np.negative,
    "absolute": np.abs,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "recip": lambda x: 1.0 / x,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
}

UNARY_MLX = {
    "negate": lambda x: -x,
    "absolute": mx.abs,
    "exp": mx.exp,
    "log": mx.log,
    "sqrt": mx.sqrt,
    "recip": lambda x: 1.0 / x,
    "sin": mx.sin,
    "cos": mx.cos,
    "tanh": mx.tanh,
    "floor": mx.floor,
    "ceil": mx.ceil,
    "sign": mx.sign,
}

BINARY_CASES = {
    "add": np.add,
    "subtract": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
    "pow": np.power,
    "equal": np.equal,
    "not_equal": np.not_equal,
    "lt": np.less,
    "gt": np.greater,
    "ge": np.greater_equal,
    "le": np.less_equal,
}

BINARY_MLX = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "max": mx.maximum,
    "min": mx.minimum,
    "pow": mx.power,
    "equal": lambda x, y: x == y,
    "not_equal": lambda x, y: x != y,
    "lt": lambda x, y: x < y,
    "gt": lambda x, y: x > y,
    "ge": lambda x, y: x >= y,
    "le": lambda x, y: x <= y,
}


def _build_elementwise(
    op: str,
    M: int,
    N: int,
    y_shape: tuple[int, int] | None,
    cond_shape: tuple[int, int] | None,
    y_broadcast: str,
    cond_broadcast: str,
) -> tuple[
    Kernel, tuple[str, ...], tuple[int, ...], tuple[int, int, int], tuple[int, int, int]
]:
    tile_M = 16
    tile_N = 16
    tg_x = 32
    tg_y = 8
    tile_shape = (tile_M, tile_N)
    num_threads = tg_x * tg_y
    num_inputs = elementwise_arity(op)

    fragments = [
        TgLoadFragment(
            "load_X_tile",
            "X",
            "N",
            f"tg.y * {tile_M}",
            f"tg.x * {tile_N}",
            "X_tile",
            tile_shape,
            num_threads,
            "M",
            "N",
        )
    ]
    if num_inputs >= 2:
        fragments.append(
            TgLoadFragment(
                "load_Y_tile",
                "Y",
                "Y_stride",
                "0" if y_broadcast in ("scalar", "row") else f"tg.y * {tile_M}",
                "0" if y_broadcast in ("scalar", "col") else f"tg.x * {tile_N}",
                "Y_tile",
                tile_shape,
                num_threads,
                "Y_rows",
                "Y_cols",
            )
        )
    if num_inputs == 3:
        fragments.append(
            TgLoadFragment(
                "load_Cond_tile",
                "Cond",
                "C_stride",
                "0" if cond_broadcast in ("scalar", "row") else f"tg.y * {tile_M}",
                "0" if cond_broadcast in ("scalar", "col") else f"tg.x * {tile_N}",
                "Cond_tile",
                tile_shape,
                num_threads,
                "C_rows",
                "C_cols",
            )
        )
    fragments.extend(
        [
            BarrierFragment("inputs_ready"),
            ElementwiseComputeFragment(
                op,
                "Out_tile",
                "X_tile",
                tile_shape,
                num_threads,
                "Y_tile" if num_inputs >= 2 else None,
                "Cond_tile" if num_inputs == 3 else None,
                y_broadcast,
                cond_broadcast,
            ),
            BarrierFragment("output_ready"),
            TgStoreFragment(
                "store_Out_tile",
                "Out_tile",
                "Out",
                "N",
                f"tg.y * {tile_M}",
                f"tg.x * {tile_N}",
                tile_shape,
                num_threads,
                "M",
                "N",
            ),
        ]
    )

    buffer_decls = ["device const float* X [[buffer(0)]]"]
    if num_inputs >= 2:
        buffer_decls.append("device const float* Y [[buffer(1)]]")
    if num_inputs == 3:
        buffer_decls.append("device const float* Cond [[buffer(2)]]")
    out_idx = num_inputs if num_inputs <= 2 else 3
    buffer_decls.append(f"device float* Out [[buffer({out_idx})]]")

    dim_names = ["M", "N"]
    dim_values: list[int] = [M, N]
    if num_inputs >= 2:
        assert y_shape is not None
        dim_names.extend(["Y_rows", "Y_cols", "Y_stride"])
        dim_values.extend([y_shape[0], y_shape[1], y_shape[1]])
    if num_inputs == 3:
        assert cond_shape is not None
        dim_names.extend(["C_rows", "C_cols", "C_stride"])
        dim_values.extend([cond_shape[0], cond_shape[1], cond_shape[1]])

    tg_decls = [f"threadgroup float X_tile[{tile_M}][{tile_N}];"]
    if num_inputs >= 2:
        tg_decls.append(f"threadgroup float Y_tile[{tile_M}][{tile_N}];")
    if num_inputs == 3:
        tg_decls.append(f"threadgroup float Cond_tile[{tile_M}][{tile_N}];")
    tg_decls.append(f"threadgroup float Out_tile[{tile_M}][{tile_N}];")

    ctx = CodegenContext(
        function_name="elementwise_kernel",
        buffers=tuple(buffer_decls),
        dims=tuple(dim_names),
        tg_x=tg_x,
        tg_y=tg_y,
        threadgroup_decls=tuple(tg_decls),
        preamble=(
            "uint2 tg = threadgroup_position_in_grid;",
            "uint2 lid = thread_position_in_threadgroup;",
            f"uint flat_tid = lid.y * {tg_x} + lid.x;",
        ),
    )
    kernel = Kernel(fragments=tuple(fragments), ctx=ctx)
    binding_names: list[str] = ["X"]
    if num_inputs >= 2:
        binding_names.append("Y")
    if num_inputs == 3:
        binding_names.append("Cond")
    binding_names.append("Out")
    grid = ((N + tile_N - 1) // tile_N, (M + tile_M - 1) // tile_M, 1)
    threads = (tg_x, tg_y, 1)
    return kernel, tuple(binding_names), tuple(dim_values), grid, threads


def run_ours(
    op: str,
    x: np.ndarray,
    y: np.ndarray | None = None,
    cond: np.ndarray | None = None,
    y_broadcast: str = "none",
    cond_broadcast: str = "none",
    timed: bool = False,
) -> tuple[np.ndarray, float]:
    M, N = x.shape
    kernel, bindings, dims, grid, threads = _build_elementwise(
        op,
        M,
        N,
        y.shape if y is not None else None,
        cond.shape if cond is not None else None,
        y_broadcast,
        cond_broadcast,
    )

    setup_steps: list = [FromNumpy("X", x)]
    if y is not None:
        setup_steps.append(FromNumpy("Y", y))
    if cond is not None:
        setup_steps.append(FromNumpy("Cond", cond))
    setup_steps.extend([Allocate("Out", M * N * 4), kernel])
    env = Runtime(tuple(setup_steps)).run()

    inner = Runtime(
        (
            Fill("Out", 0),
            Dispatch(
                kernel,
                bindings=bindings,
                dims=dims,
                grid=grid,
                threads=threads,
                time_key="t",
            ),
        )
    )

    repeats = ITERS if timed else 1
    for _ in range(WARMUP if timed else 0):
        inner.run(env)

    times = []
    for _ in range(repeats):
        inner.run(env)
        times.append(env["t"])

    Runtime((Download("Out", shape=(M, N), dtype=np.float32, into="out"),)).run(env)
    return env["out"], float(np.mean(times))


def mlx_ms(fn, *arrays: np.ndarray) -> tuple[np.ndarray, float]:
    mx_arrays = [mx.array(a) for a in arrays]
    mx.eval(*mx_arrays)

    for _ in range(WARMUP):
        mx.eval(fn(*mx_arrays))

    times = []
    out = None
    for _ in range(ITERS):
        start = time.perf_counter()
        out = fn(*mx_arrays)
        mx.eval(out)
        times.append((time.perf_counter() - start) * 1000)
    return np.array(out).astype(np.float32), float(np.mean(times))


def assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        actual, expected.astype(np.float32), rtol=1e-5, atol=1e-5
    )


def test_unary_ops() -> None:
    x = np.linspace(0.25, 3.75, 130, dtype=np.float32).reshape(10, 13)
    for op, ref_fn in UNARY_CASES.items():
        out, _ = run_ours(op, x)
        assert_close(out, ref_fn(x))


def test_binary_ops() -> None:
    x = np.linspace(0.25, 3.75, 130, dtype=np.float32).reshape(10, 13)
    y = np.linspace(1.25, 2.25, 130, dtype=np.float32).reshape(10, 13)
    for op, ref_fn in BINARY_CASES.items():
        out, _ = run_ours(op, x, y)
        assert_close(out, ref_fn(x, y))


def test_broadcast_ops() -> None:
    x = np.linspace(0.25, 3.75, 130, dtype=np.float32).reshape(10, 13)
    scalar = np.array([[2.0]], dtype=np.float32)
    row = np.linspace(1.0, 2.0, 13, dtype=np.float32).reshape(1, 13)
    col = np.linspace(1.0, 2.0, 10, dtype=np.float32).reshape(10, 1)
    assert_close(run_ours("add", x, scalar, y_broadcast="scalar")[0], x + scalar)
    assert_close(run_ours("add", x, row, y_broadcast="row")[0], x + row)
    assert_close(run_ours("mul", x, col, y_broadcast="col")[0], x * col)


def test_where() -> None:
    x = np.linspace(0.25, 3.75, 130, dtype=np.float32).reshape(10, 13)
    y = np.flip(x, axis=1).copy()
    cond = (x > 2.0).astype(np.float32)
    out, _ = run_ours("where", x, y, cond)
    assert_close(out, np.where(cond != 0, x, y))


def bench_case(
    name: str,
    op: str,
    x: np.ndarray,
    y: np.ndarray | None,
    cond: np.ndarray | None,
    fn,
    y_broadcast: str = "none",
    cond_broadcast: str = "none",
) -> None:
    ours_out, ours_time = run_ours(
        op, x, y, cond, y_broadcast, cond_broadcast, timed=True
    )
    args = [x]
    if y is not None:
        args.append(y)
    if cond is not None:
        args = [cond, x, y]
    mlx_out, mlx_time = mlx_ms(fn, *args)
    assert_close(ours_out, mlx_out)
    print(f"case={name} op={op} ours={ours_time:.4f}ms mlx={mlx_time:.4f}ms")


def bench_elementwise() -> None:
    rng = np.random.default_rng(17)
    x = rng.uniform(0.25, 3.75, size=(1024, 1024)).astype(np.float32)
    y = rng.uniform(1.25, 2.25, size=(1024, 1024)).astype(np.float32)
    scalar = np.array([[1.5]], dtype=np.float32)
    row = rng.uniform(0.5, 1.5, size=(1, 1024)).astype(np.float32)
    col = rng.uniform(0.5, 1.5, size=(1024, 1)).astype(np.float32)
    cond = (x > 2.0).astype(np.float32)

    for op in ("negate", "exp", "sqrt", "tanh"):
        bench_case("unary", op, x, None, None, UNARY_MLX[op])
    for op in ("add", "mul", "div", "max", "lt"):
        bench_case("binary", op, x, y, None, BINARY_MLX[op])
    bench_case(
        "scalar_broadcast",
        "add",
        x,
        scalar,
        None,
        lambda a, b: a + b,
        y_broadcast="scalar",
    )
    bench_case(
        "row_broadcast", "add", x, row, None, lambda a, b: a + b, y_broadcast="row"
    )
    bench_case(
        "col_broadcast", "mul", x, col, None, lambda a, b: a * b, y_broadcast="col"
    )
    bench_case("where", "where", x, y, cond, lambda c, a, b: mx.where(c != 0, a, b))


if __name__ == "__main__":
    test_unary_ops()
    test_binary_ops()
    test_broadcast_ops()
    test_where()
    bench_elementwise()
