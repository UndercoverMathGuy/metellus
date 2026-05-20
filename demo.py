"""End-to-end demo: harder DAG exercising multi-consumer fan-out,
mid-chain transpose feeding a matmul, both reduction ops, and long
elementwise chains. Verifies every materialized tensor against MLX.

The conceptual program:

    t1 = A @ B                      # (64, 64), THREE consumers
    t2 = t1 + bias1   (row bcast)   # two consumers
    t3 = t1 + C       (same shape)  # one consumer (transposed)
    t4 = relu(t1)                   # two consumers (reduce_max + sub)

    t5 = relu(t2)
    t6 = tanh(t2)

    t3T = transpose(t3)             # metadata only — no kernel
    t7  = t3T @ D                   # matmul with transposed (strided) A

    t4_max  = reduce_max(t4)        # (64,) → reshape to (64,1) view
    t4_sft  = t4 - t4_max           # col broadcast
    t4_exp  = exp(t4_sft)           # two consumers
    t4_sum  = reduce_sum(t4_exp)    # (64,) → reshape to (64,1) view
    t4_norm = t4_exp / t4_sum       # col broadcast

    t8 = t5 + t6
    t9 = t7 + t8
    out = t9 + t4_norm

Run:  uv run python demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import mlx.core as mx

from orchestrator import Operations
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse
from orchestrator.scheduler import schedule


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_program() -> tuple[Operations, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((64, 32)).astype(np.float32)
    B = rng.standard_normal((32, 64)).astype(np.float32)
    C = rng.standard_normal((64, 64)).astype(np.float32)
    D = rng.standard_normal((64, 64)).astype(np.float32)
    bias1 = rng.standard_normal((1, 64)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(C, "C")
    ops.from_numpy(D, "D")
    ops.from_numpy(bias1, "bias1")

    # t1 = A @ B — primary multi-consumer node (3 downstream readers).
    ops.matmul(a="A", b="B", out="t1")

    # Three consumers of t1 — none should be absorbed into matmul's epilogue
    # because t1 has multi-consumer; matmul stays standalone and t1 is
    # materialized.
    ops.elementwise("add", out="t2", operands=("t1", "bias1"), y_broadcast="row")
    ops.elementwise("add", out="t3", operands=("t1", "C"))
    ops.elementwise("max", out="t4", operands=("t1", 0.0))  # relu

    # t2 is consumed twice — multi-consumer too.
    ops.elementwise("max", out="t5", operands=("t2", 0.0))  # relu(t2)
    ops.elementwise("tanh", out="t6", operands=("t2",))

    # t3 → metadata-only transpose → matmul. No kernel for the transpose;
    # the matmul template reads t3's buffer with swapped strides.
    ops.transpose("t3", out="t3T")
    ops.matmul(a="t3T", b="D", out="t7")

    # Softmax-flavoured sub-DAG on t4. Reductions yield 1D; reshape views
    # back to (64, 1) so col-broadcast subtract/div can consume them.
    ops.reduction("max", out="t4_max", x="t4", axis=-1)  # (64,)
    ops.reshape("t4_max", (64, 1), out="t4_max_2d")
    ops.elementwise(
        "subtract",
        out="t4_shifted",
        operands=("t4", "t4_max_2d"),
        y_broadcast="col",
    )
    ops.elementwise("exp", out="t4_exp", operands=("t4_shifted",))
    ops.reduction("sum", out="t4_sum", x="t4_exp", axis=-1)  # (64,)
    ops.reshape("t4_sum", (64, 1), out="t4_sum_2d")
    ops.elementwise(
        "div",
        out="t4_norm",
        operands=("t4_exp", "t4_sum_2d"),
        y_broadcast="col",
    )

    # Final long chain.
    ops.elementwise("add", out="t8", operands=("t5", "t6"))
    ops.elementwise("add", out="t9", operands=("t7", "t8"))
    ops.elementwise("add", out="out", operands=("t9", "t4_norm"))

    inputs = {"A": A, "B": B, "C": C, "D": D, "bias1": bias1}
    return ops, inputs


# ---------------------------------------------------------------------------
# MLX ground truth
# ---------------------------------------------------------------------------


def mlx_reference(A, B, C, D, bias1):
    Am = mx.array(A)
    Bm = mx.array(B)
    Cm = mx.array(C)
    Dm = mx.array(D)
    b1m = mx.array(bias1)

    t1 = Am @ Bm
    t2 = t1 + b1m
    t3 = t1 + Cm
    t4 = mx.maximum(t1, 0.0)

    t5 = mx.maximum(t2, 0.0)
    t6 = mx.tanh(t2)

    t3T = t3.T
    t7 = t3T @ Dm

    t4_max = mx.max(t4, axis=-1, keepdims=True)
    t4_shifted = t4 - t4_max
    t4_exp = mx.exp(t4_shifted)
    t4_sum = mx.sum(t4_exp, axis=-1, keepdims=True)
    t4_norm = t4_exp / t4_sum

    t8 = t5 + t6
    t9 = t7 + t8
    out = t9 + t4_norm

    return {
        "t1": np.array(t1),
        "t2": np.array(t2),
        "t3": np.array(t3),
        "t4": np.array(t4),
        "t5": np.array(t5),
        "t6": np.array(t6),
        "t3T": np.array(t3T),
        "t7": np.array(t7),
        "t4_max": np.array(t4_max).reshape(-1),
        "t4_max_2d": np.array(t4_max),
        "t4_shifted": np.array(t4_shifted),
        "t4_exp": np.array(t4_exp),
        "t4_sum": np.array(t4_sum).reshape(-1),
        "t4_sum_2d": np.array(t4_sum),
        "t4_norm": np.array(t4_norm),
        "t8": np.array(t8),
        "t9": np.array(t9),
        "out": np.array(out),
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _hr(char: str = "─", width: int = 78) -> None:
    print(char * width)


def dump_kernel(i: int, group) -> None:
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


def compare(actual_env: dict, reference: dict) -> bool:
    _hr("=")
    print("Compare each materialized tensor against MLX")
    _hr("=")
    width = max(len(k) for k in reference)
    all_ok = True
    for name in reference:
        if name not in actual_env:
            print(f"  {name:<{width}}  (fused away — no buffer)")
            continue
        a = actual_env[name]
        b = reference[name]
        if a.shape != b.shape:
            print(f"  {name:<{width}}  SHAPE MISMATCH: {a.shape} vs {b.shape}")
            all_ok = False
            continue
        diff = np.abs(a - b)
        denom = np.abs(b) + 1e-9
        max_abs = float(np.max(diff))
        max_rel = float(np.max(diff / denom))
        ok = np.allclose(a, b, rtol=1e-3, atol=1e-4)
        status = "OK " if ok else "BAD"
        print(
            f"  [{status}] {name:<{width}}  shape={str(a.shape):<10}  "
            f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        )
        all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ops, inputs = build_program()
    program = ops.build()
    decisions = fuse(program)
    groups = tuple(assemble(d) for d in decisions)

    _hr("=")
    print(f"Program: {len(program)} primitive ops → {len(groups)} fused kernel(s)")
    _hr("=")
    print(f"  inputs:  {sorted(inputs.keys())}")
    print(f"  named tensors: {sorted(ops.tensors.keys())}")
    print()

    for i, g in enumerate(groups):
        dump_kernel(i, g)

    runtime = schedule(ops, groups)
    env = runtime.run()

    reference = mlx_reference(**inputs)
    ok = compare(env, reference)

    _hr("=")
    print("PASS" if ok else "FAIL")
    _hr("=")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
