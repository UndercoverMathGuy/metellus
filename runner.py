"""Tricky-kernels driver. Each scenario builds an `Operations` program,
fuses + assembles + schedules it, and prints the generated MSL plus
per-kernel GPU timings. Goal: stress the compiler in ways that expose
where the optimizer's next-most-valuable wins live."""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from orchestrator import Operations
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse
from orchestrator.scheduler import schedule


def _hr(char: str = "─", width: int = 78) -> None:
    print(char * width)


def _dump_kernel(i: int, group) -> None:
    _hr("=")
    print(f"Kernel {i}: {group.strategy.value}")
    _hr("=")
    print(f"  function:  {group.kernel.function_name}")
    print(f"  bindings:  {group.bindings}")
    print(f"  dims:      {group.dims}")
    print(f"  grid:      {group.grid}")
    print(f"  threads:   {group.threads}")
    print(f"  absorbed:  {[op.out.name for op in group.ops]}")
    print()
    print(group.kernel.source)


def run_scenario(name: str, build_fn, *, dump_msl: bool = True) -> dict:
    """Build, fuse, assemble, schedule, run. Returns timing summary."""
    _hr("█")
    print(f"SCENARIO: {name}")
    _hr("█")
    ops = build_fn()
    program = ops.build()
    decisions = fuse(program)
    groups = tuple(assemble(d) for d in decisions)

    print(f"  primitive ops: {len(program)}")
    print(f"  fused kernels: {len(groups)}")
    print(f"  inputs: {sorted(ops.uploads.keys())}")
    print()

    if dump_msl:
        for i, group in enumerate(groups):
            _dump_kernel(i, group)

    runtime = schedule(ops, groups, profile=True)
    start = time.perf_counter()
    env = runtime.run()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

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
    print(f"  wall:      {elapsed_ms:.3f} ms")
    print()
    return {
        "name": name,
        "n_ops": len(program),
        "n_kernels": len(groups),
        "kernel_times": kernel_times,
        "total_gpu_ms": sum(kernel_times),
        "wall_ms": elapsed_ms,
        "strategies": [g.strategy.value for g in groups],
    }


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


SCENARIOS = [
    ("s1_big_matmul_aligned", s1_big_matmul_aligned),
    ("s2_mlp_layer", s2_mlp_layer),
    ("s3_gelu_chain", s3_gelu_chain),
    ("s4_diamond_multi_consumer", s4_diamond_multi_consumer),
    ("s5_attention_block", s5_attention_block),
    ("s6_layernorm_like", s6_layernorm_like),
    ("s7_nonaligned_matmul", s7_nonaligned_matmul),
    ("s8_tall_skinny_matmul", s8_tall_skinny_matmul),
    ("s9_stacked_mlps", s9_stacked_mlps),
    ("s10_many_tensor_operands", s10_many_tensor_operands),
]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    dump_msl = "--no-msl" not in sys.argv
    summaries = []
    for name, fn in SCENARIOS:
        if only is not None and only not in name:
            continue
        summaries.append(run_scenario(name, fn, dump_msl=dump_msl))

    _hr("█")
    print("SUMMARY")
    _hr("█")
    print(
        f"{'scenario':30s} {'ops':>4s} {'kerns':>5s} {'gpu (ms)':>10s} {'wall (ms)':>10s}"
    )
    for s in summaries:
        print(
            f"{s['name']:30s} {s['n_ops']:>4d} {s['n_kernels']:>5d} "
            f"{s['total_gpu_ms']:>10.3f} {s['wall_ms']:>10.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
