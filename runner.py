"""Tricky-kernels driver. Each scenario builds an `Operations` program
and runs it through the public `api.run` pipeline (fuse → assemble →
lifetime → schedule → execute), printing the generated MSL plus
per-kernel GPU timings. Goal: stress the compiler in ways that expose
where the optimizer's next-most-valuable wins live."""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import mlx.core as mx

from api import run as api_run
from orchestrator import Operations
from orchestrator.aliasing import alias_group
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse


def _hr(char: str = "─", width: int = 78) -> None:
    print(char * width)


def _describe_op(op) -> str:
    """One-line human-readable summary of an IR op for the kernel dump."""
    from orchestrator.ir import (
        ElementwiseOp,
        MatmulOp,
        ReductionOp,
        Scalar,
        ShapeOp,
    )

    cls = type(op).__name__
    if isinstance(op, MatmulOp):
        return f"{cls}: {op.out.name} = {op.a.name} @ {op.b.name}"
    if isinstance(op, ElementwiseOp):
        operands = ", ".join(
            (repr(o.value) if isinstance(o, Scalar) else o.name) for o in op.operands
        )
        bcasts = []
        if op.y_broadcast.value != "none":
            bcasts.append(f"y_bcast={op.y_broadcast.value}")
        if op.cond_broadcast.value != "none":
            bcasts.append(f"cond_bcast={op.cond_broadcast.value}")
        tail = f" [{', '.join(bcasts)}]" if bcasts else ""
        return f"{cls}: {op.out.name} = {op.op}({operands}){tail}"
    if isinstance(op, ReductionOp):
        return f"{cls}: {op.out.name} = {op.op}({op.input.name}, axis={op.axis})"
    if isinstance(op, ShapeOp):
        return f"{cls}: {op.out.name} = reshape({op.input.name}) [shape={op.out.shape}]"
    return f"{cls}: {op.out.name}"


def _dump_kernel(i: int, group) -> None:
    _hr("=")
    print(f"Kernel {i}: {group.strategy.value}")
    _hr("=")
    print(f"  function:  {group.kernel.function_name}")
    print(f"  bindings:  {group.bindings}")
    print(f"  dims:      {group.dims}")
    print(f"  grid:      {group.grid}")
    print(f"  threads:   {group.threads}")
    print(f"  absorbed ops ({len(group.ops)}):")
    for op in group.ops:
        print(f"    - {_describe_op(op)}")
    print()
    print(group.kernel.source)


def run_scenario(
    name: str, build_fn, mlx_fn=None, *, dump_msl: bool = True, execute: bool = True
) -> dict:
    """Build the IR and run it through `api.run` (or, when
    `execute=False`, only compile via fuse + assemble for structural
    inspection). When `mlx_fn` is provided and `execute=True`, also
    time the MLX equivalent (one shot — no warmup, no averaging, JIT
    included). Returns a summary dict; when `execute=False`, only the
    structural fields are populated."""
    _hr("█")
    print(f"SCENARIO: {name}")
    _hr("█")
    ops = build_fn()

    if execute:
        start = time.perf_counter()
        result = api_run(ops, profile=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        env = result.env
        groups = result.groups
    else:
        # Compile-only path: fuse + assemble + alias for structural
        # inspection. Mirror api.run so the dumped MSL matches what
        # the runtime would actually compile.
        vertices = fuse(ops.build())
        groups = tuple(alias_group(assemble(v)) for v in vertices)
        env = None
        elapsed_ms = 0.0

    program = ops.build()  # cheap; for IR-listing dump and op count.

    print(f"  primitive ops: {len(program)}")
    print(f"  fused kernels: {len(groups)}")
    print(f"  inputs: {sorted(ops.uploads.keys())}")
    print()

    if dump_msl:
        _hr("=")
        print(f"IR program ({len(program)} ops)")
        _hr("=")
        for i, op in enumerate(program):
            print(f"  {i:2d}  {_describe_op(op)}")
        print()
        for i, group in enumerate(groups):
            _dump_kernel(i, group)

    summary = {
        "name": name,
        "n_ops": len(program),
        "n_kernels": len(groups),
        "kernel_times": [],
        "total_gpu_ms": 0.0,
        "wall_ms": 0.0,
        "mlx_ms": 0.0,
        "strategies": [g.strategy.value for g in groups],
        "allclose_ok": None,  # None = not checked (no mlx_fn or scenario errored)
    }

    if not execute:
        _hr("=")
        print(f"Skipped run — {name} (use without --no-run to time)")
        _hr("=")
        print()
        return summary

    kernel_times = [float(env[f"t_{i}"]) for i in range(len(groups))]
    _hr("=")
    print(f"Timing — {name}")
    _hr("=")
    for i, group in enumerate(groups):
        print(
            f"  kernel {i:2d} [{group.strategy.value:38s}] "
            f"{kernel_times[i]:7.3f} ms  ({group.kernel.function_name})"
        )
    print(f"  total GPU: {sum(kernel_times):.3f} ms")
    print(f"  triton wall: {elapsed_ms:.3f} ms")

    if mlx_fn is not None:
        mlx_start = time.perf_counter()
        mlx_outputs = mlx_fn(ops.uploads)
        mlx_ms = (time.perf_counter() - mlx_start) * 1000.0
        print(f"  mlx wall:    {mlx_ms:.3f} ms  (one-shot, JIT included)")
        summary["mlx_ms"] = mlx_ms

        # Cross-check every MLX output against our env entry by name.
        # fp32 matmul accumulation drift makes tight tolerances unrealistic
        # for K=1024+; rtol=1e-3, atol=1e-2 is the "matches within fp32
        # ordering noise" line — anything past that is a real bug.
        rtol, atol = 1e-3, 1e-2
        all_ok = True
        for out_name, mlx_arr in mlx_outputs.items():
            ours = env.get(out_name)
            if ours is None:
                print(f"  allclose {out_name:6s}: MISSING from env")
                all_ok = False
                continue
            ours_arr = np.asarray(ours)
            diff = np.abs(ours_arr.astype(np.float32) - mlx_arr.astype(np.float32))
            max_abs = float(diff.max()) if diff.size else 0.0
            denom = np.maximum(np.abs(mlx_arr), 1e-12)
            max_rel = float((diff / denom).max()) if diff.size else 0.0
            ok = np.allclose(ours_arr, mlx_arr, rtol=rtol, atol=atol)
            status = "OK  " if ok else "DIFF"
            print(
                f"  allclose {out_name:6s}: {status}  "
                f"max|Δ|={max_abs:.2e}  max|Δ/x|={max_rel:.2e}"
            )
            if not ok:
                all_ok = False
        summary["allclose_ok"] = all_ok

    print()
    summary["kernel_times"] = kernel_times
    summary["total_gpu_ms"] = sum(kernel_times)
    summary["wall_ms"] = elapsed_ms
    return summary


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def s1_big_matmul_aligned():
    """Plain 1024×1024×1024 matmul. Baseline for raw throughput at scale."""
    rng = np.random.default_rng(0)
    M, K, N = 1024, 1024, 1024
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    return ops


def s2_mlp_layer():
    """Classic MLP block: y = relu(x @ W + b). Should fuse into 1 kernel
    via matmul + register epilogue (bias add is row-broadcast tensor, so
    actually forces tg-tile epilogue path)."""
    rng = np.random.default_rng(0)
    M, K, N = 512, 512, 512
    X = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.standard_normal((K, N)).astype(np.float32)
    b = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(W, "W")
    ops.from_numpy(b, "bias")
    ops.matmul(a="X", b="W", out="z")
    ops.elementwise("add", out="zb", operands=("z", "bias"), y_broadcast="row")
    ops.elementwise("max", out="y", operands=("zb", 0.0))
    return ops


def s3_gelu_chain():
    """Long unary chain after matmul — exercises register-epilogue depth.
    GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Expressed without aliasing intermediates so each op is unary."""
    rng = np.random.default_rng(0)
    M, K, N = 512, 512, 512
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="z")
    # x^3
    ops.elementwise("mul", out="z2", operands=("z", "z"))
    ops.elementwise("mul", out="z3", operands=("z2", "z"))
    # 0.044715 * x^3
    ops.elementwise("mul", out="zc", operands=("z3", 0.044715))
    # x + 0.044715 x^3
    ops.elementwise("add", out="zs", operands=("z", "zc"))
    # sqrt(2/pi) * (...)
    ops.elementwise("mul", out="zk", operands=("zs", math.sqrt(2.0 / math.pi)))
    # tanh
    ops.elementwise("tanh", out="zt", operands=("zk",))
    # 1 + tanh
    ops.elementwise("add", out="z1", operands=("zt", 1.0))
    # 0.5 * x
    ops.elementwise("mul", out="zh", operands=("z", 0.5))
    # final mul
    ops.elementwise("mul", out="y", operands=("zh", "z1"))
    return ops


def s4_diamond_multi_consumer():
    """Single matmul output feeds two parallel tails. Multi-consumer
    rule kills downstream fusion: t must materialize, two standalone
    elementwise kernels read it."""
    rng = np.random.default_rng(0)
    M, K, N = 256, 256, 256
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="t")
    ops.elementwise("max", out="r", operands=("t", 0.0))
    ops.elementwise("exp", out="e", operands=("t",))
    return ops


def s5_attention_block():
    """Toy attention: out = softmax(Q @ K^T) @ V, no scale, no mask.
    Exercises: matmul → transpose-view → reduction → elem → reduction
    → elem → matmul chain. Lots of small kernels with intermediates."""
    rng = np.random.default_rng(0)
    S, D = 128, 64
    Q = rng.standard_normal((S, D)).astype(np.float32)
    Kmat = rng.standard_normal((S, D)).astype(np.float32)
    V = rng.standard_normal((S, D)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(Q, "Q")
    ops.from_numpy(Kmat, "Kmat")
    ops.from_numpy(V, "V")
    ops.transpose("Kmat", out="Kt")
    ops.matmul(a="Q", b="Kt", out="scores")
    ops.reduction("max", out="row_max", x="scores", axis=-1)
    ops.reshape("row_max", (S, 1), out="row_max_2d")
    ops.elementwise(
        "subtract", out="shifted", operands=("scores", "row_max_2d"), y_broadcast="col"
    )
    ops.elementwise("exp", out="num", operands=("shifted",))
    ops.reduction("sum", out="row_sum", x="num", axis=-1)
    ops.reshape("row_sum", (S, 1), out="row_sum_2d")
    ops.elementwise(
        "div", out="probs", operands=("num", "row_sum_2d"), y_broadcast="col"
    )
    ops.matmul(a="probs", b="V", out="out")
    return ops


def s6_layernorm_like():
    """Layernorm-shaped graph: y = (x - mean) / sqrt(var + eps), with
    mean = sum(x)/N and var = sum((x-mean)^2)/N. Two reductions over
    the same row — the second depends on the first, so they can't run
    in parallel. Tests reduction prologue/epilogue fusion."""
    rng = np.random.default_rng(0)
    M, N = 256, 512
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    # mean
    ops.reduction("sum", out="row_sum", x="X", axis=-1)
    ops.elementwise("mul", out="mean", operands=("row_sum", 1.0 / N))
    ops.reshape("mean", (M, 1), out="mean_2d")
    # centered
    ops.elementwise(
        "subtract", out="centered", operands=("X", "mean_2d"), y_broadcast="col"
    )
    # squared
    ops.elementwise("mul", out="sq", operands=("centered", "centered"))
    # var (× 1/N)
    ops.reduction("sum", out="sq_sum", x="sq", axis=-1)
    ops.elementwise("mul", out="var", operands=("sq_sum", 1.0 / N))
    ops.elementwise("add", out="var_eps", operands=("var", 1e-5))
    ops.elementwise("sqrt", out="std", operands=("var_eps",))
    ops.reshape("std", (M, 1), out="std_2d")
    # normalize
    ops.elementwise("div", out="y", operands=("centered", "std_2d"), y_broadcast="col")
    return ops


def s7_nonaligned_matmul():
    """Awkward shape that forces the unaligned matmul path AND a
    non-aligned epilogue. Tests bounds-checking overhead and tail
    handling."""
    rng = np.random.default_rng(0)
    M, K, N = 257, 129, 193
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    b = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(b, "bias")
    ops.matmul(a="A", b="B", out="z")
    ops.elementwise("add", out="zb", operands=("z", "bias"), y_broadcast="row")
    ops.elementwise("max", out="y", operands=("zb", 0.0))
    return ops


def s8_tall_skinny_matmul():
    """4096×64 @ 64×4096 — many M rows, narrow K. Tests tile selection
    and grid utilization for skewed shapes."""
    rng = np.random.default_rng(0)
    M, K, N = 4096, 64, 4096
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    return ops


def s9_stacked_mlps():
    """Two-layer MLP: relu(relu(X @ W1 + b1) @ W2 + b2). Tests
    matmul→matmul chaining and whether the epilogue of layer 1 can be
    cleanly the input of layer 2."""
    rng = np.random.default_rng(0)
    M, K1, K2, N = 256, 256, 512, 256
    X = rng.standard_normal((M, K1)).astype(np.float32)
    W1 = rng.standard_normal((K1, K2)).astype(np.float32)
    b1 = rng.standard_normal((1, K2)).astype(np.float32)
    W2 = rng.standard_normal((K2, N)).astype(np.float32)
    b2 = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(W1, "W1")
    ops.from_numpy(b1, "b1")
    ops.from_numpy(W2, "W2")
    ops.from_numpy(b2, "b2")
    ops.matmul(a="X", b="W1", out="z1")
    ops.elementwise("add", out="z1b", operands=("z1", "b1"), y_broadcast="row")
    ops.elementwise("max", out="h1", operands=("z1b", 0.0))
    ops.matmul(a="h1", b="W2", out="z2")
    ops.elementwise("add", out="z2b", operands=("z2", "b2"), y_broadcast="row")
    ops.elementwise("max", out="y", operands=("z2b", 0.0))
    return ops


def s10_many_tensor_operands():
    """Elementwise chain that consumes many distinct input tensors —
    pushes the threadgroup-tile-per-operand strategy."""
    rng = np.random.default_rng(0)
    M, N = 256, 256
    shapes = [(M, N)] * 6
    arrs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    ops = Operations()
    for i, a in enumerate(arrs):
        ops.from_numpy(a, f"x{i}")
    ops.elementwise("add", out="s1", operands=("x0", "x1"))
    ops.elementwise("add", out="s2", operands=("s1", "x2"))
    ops.elementwise("add", out="s3", operands=("s2", "x3"))
    ops.elementwise("add", out="s4", operands=("s3", "x4"))
    ops.elementwise("add", out="y", operands=("s4", "x5"))
    return ops


# ---------------------------------------------------------------------------
# MLX counterparts — one shot each (no warmup, no averaging) so timing
# includes JIT compile, matching what we charge our kernels on first
# dispatch. `mx.eval` forces realisation since MLX is lazy.
# ---------------------------------------------------------------------------


def _mlx_s1(inputs):
    C = mx.array(inputs["A"]) @ mx.array(inputs["B"])
    mx.eval(C)
    return {"C": np.asarray(C)}


def _mlx_s2(inputs):
    X, W, b = mx.array(inputs["X"]), mx.array(inputs["W"]), mx.array(inputs["bias"])
    y = mx.maximum(X @ W + b, 0.0)
    mx.eval(y)
    return {"y": np.asarray(y)}


def _mlx_s3(inputs):
    A, B = mx.array(inputs["A"]), mx.array(inputs["B"])
    z = A @ B
    inner = math.sqrt(2.0 / math.pi) * (z + 0.044715 * z * z * z)
    y = 0.5 * z * (1.0 + mx.tanh(inner))
    mx.eval(y)
    return {"y": np.asarray(y)}


def _mlx_s4(inputs):
    A, B = mx.array(inputs["A"]), mx.array(inputs["B"])
    t = A @ B
    r = mx.maximum(t, 0.0)
    e = mx.exp(t)
    mx.eval([r, e])
    return {"r": np.asarray(r), "e": np.asarray(e)}


def _mlx_s5(inputs):
    Q, K, V = mx.array(inputs["Q"]), mx.array(inputs["Kmat"]), mx.array(inputs["V"])
    scores = Q @ K.T
    num = mx.exp(scores - mx.max(scores, axis=-1, keepdims=True))
    probs = num / mx.sum(num, axis=-1, keepdims=True)
    out = probs @ V
    mx.eval(out)
    return {"out": np.asarray(out)}


def _mlx_s6(inputs):
    X = mx.array(inputs["X"])
    _, N = X.shape
    mean = mx.sum(X, axis=-1, keepdims=True) / N
    centered = X - mean
    var = mx.sum(centered * centered, axis=-1, keepdims=True) / N
    y = centered / mx.sqrt(var + 1e-5)
    mx.eval(y)
    return {"y": np.asarray(y)}


def _mlx_s7(inputs):
    A, B, b = mx.array(inputs["A"]), mx.array(inputs["B"]), mx.array(inputs["bias"])
    y = mx.maximum(A @ B + b, 0.0)
    mx.eval(y)
    return {"y": np.asarray(y)}


def _mlx_s8(inputs):
    C = mx.array(inputs["A"]) @ mx.array(inputs["B"])
    mx.eval(C)
    return {"C": np.asarray(C)}


def _mlx_s9(inputs):
    X = mx.array(inputs["X"])
    W1, b1 = mx.array(inputs["W1"]), mx.array(inputs["b1"])
    W2, b2 = mx.array(inputs["W2"]), mx.array(inputs["b2"])
    h1 = mx.maximum(X @ W1 + b1, 0.0)
    y = mx.maximum(h1 @ W2 + b2, 0.0)
    mx.eval(y)
    return {"y": np.asarray(y)}


def _mlx_s10(inputs):
    xs = [mx.array(inputs[f"x{i}"]) for i in range(6)]
    y = xs[0] + xs[1] + xs[2] + xs[3] + xs[4] + xs[5]
    mx.eval(y)
    return {"y": np.asarray(y)}


SCENARIOS = [
    ("s1_big_matmul_aligned", s1_big_matmul_aligned, _mlx_s1),
    ("s2_mlp_layer", s2_mlp_layer, _mlx_s2),
    ("s3_gelu_chain", s3_gelu_chain, _mlx_s3),
    ("s4_diamond_multi_consumer", s4_diamond_multi_consumer, _mlx_s4),
    ("s5_attention_block", s5_attention_block, _mlx_s5),
    ("s6_layernorm_like", s6_layernorm_like, _mlx_s6),
    ("s7_nonaligned_matmul", s7_nonaligned_matmul, _mlx_s7),
    ("s8_tall_skinny_matmul", s8_tall_skinny_matmul, _mlx_s8),
    ("s9_stacked_mlps", s9_stacked_mlps, _mlx_s9),
    ("s10_many_tensor_operands", s10_many_tensor_operands, _mlx_s10),
]


def main() -> int:
    import traceback

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    dump_msl = "--no-msl" not in sys.argv
    execute = "--no-run" not in sys.argv
    summaries = []
    for name, fn, mlx_fn in SCENARIOS:
        if only is not None and only not in name:
            continue
        try:
            summaries.append(
                run_scenario(name, fn, mlx_fn, dump_msl=dump_msl, execute=execute)
            )
        except Exception as exc:
            _hr("!")
            print(f"FAILED — {name}")
            _hr("!")
            traceback.print_exc()
            print()
            summaries.append(
                {
                    "name": name,
                    "n_ops": 0,
                    "n_kernels": 0,
                    "kernel_times": [],
                    "total_gpu_ms": 0.0,
                    "wall_ms": 0.0,
                    "mlx_ms": 0.0,
                    "strategies": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    _hr("█")
    print("SUMMARY")
    _hr("█")
    if execute:
        print(
            f"{'scenario':30s} {'ops':>4s} {'kerns':>5s} {'gpu (ms)':>10s} "
            f"{'wall (ms)':>10s} {'mlx (ms)':>10s} {'allclose':>9s}  status"
        )
        for s in summaries:
            if "error" in s:
                allclose_str = "-"
                status = s["error"]
            else:
                ac = s.get("allclose_ok")
                allclose_str = {True: "OK", False: "DIFF", None: "n/a"}[ac]
                status = "ok" if ac is not False else "MISMATCH"
            print(
                f"{s['name']:30s} {s['n_ops']:>4d} {s['n_kernels']:>5d} "
                f"{s['total_gpu_ms']:>10.3f} {s['wall_ms']:>10.3f} "
                f"{s['mlx_ms']:>10.3f} {allclose_str:>9s}  {status}"
            )
    else:
        print(f"{'scenario':30s} {'ops':>4s} {'kerns':>5s}  strategies / error")
        for s in summaries:
            tail = s.get("error", ", ".join(s["strategies"]))
            print(f"{s['name']:30s} {s['n_ops']:>4d} {s['n_kernels']:>5d}  {tail}")
    n_failed = sum(
        1 for s in summaries if "error" in s or s.get("allclose_ok") is False
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
