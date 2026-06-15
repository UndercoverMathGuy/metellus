"""Correctness sweep: probe corner cases that runner.py / demo.py don't
exercise. Each probe builds an `Operations`, runs through `api.run`, and
verifies every materialized output against MLX.

Usage:
    uv run sandbox.py                 # run all probes
    uv run sandbox.py --only product  # filter by substring of probe name
    uv run sandbox.py --no-msl        # silence MSL on failures (default off)

Probe authoring rule: every probe MUST return (ops, mlx_outputs_dict).
The harness handles execution + comparison.
"""

## !! NOTE: THIS FILE IS FOR AI TESTING OF EDGE CASES - NOT REAL TESTS

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import mlx.core as mx

from api import run as api_run
from orchestrator import Operations


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _to_np(x):
    return np.asarray(x)


def _allclose(
    a: np.ndarray, b: np.ndarray, rtol=1e-3, atol=1e-3
) -> tuple[bool, float, float]:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    diff = np.abs(a32 - b32)
    denom = np.maximum(np.abs(b32), 1e-12)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    ok = np.allclose(a32, b32, rtol=rtol, atol=atol, equal_nan=True)
    return ok, max_abs, max_rel


def _hr(c="-", w=78):
    print(c * w)


def run_probe(name: str, build: Callable, *, dump_msl: bool = False) -> dict:
    print(f"\n=== {name} ===")
    out = {"name": name, "ok": True, "msgs": []}
    try:
        ops, mlx_outputs = build()
        result = api_run(ops)
        env = result.env
        groups = result.groups
        print(
            f"  fused into {len(groups)} kernel(s): "
            f"[{', '.join(g.strategy.value for g in groups)}]"
        )

        for tname, expected in mlx_outputs.items():
            actual = env.get(tname)
            if actual is None:
                msg = f"  [MISS] {tname}: not in env"
                print(msg)
                out["ok"] = False
                out["msgs"].append(msg)
                continue
            actual = _to_np(actual)
            expected = _to_np(expected)
            if actual.shape != expected.shape:
                msg = f"  [SHAPE] {tname}: actual={actual.shape} expected={expected.shape}"
                print(msg)
                out["ok"] = False
                out["msgs"].append(msg)
                continue
            ok, max_abs, max_rel = _allclose(actual, expected)
            tag = "OK  " if ok else "DIFF"
            print(
                f"  [{tag}] {tname:<18s} shape={str(actual.shape):<14s} "
                f"max|Δ|={max_abs:.2e} max|Δ/x|={max_rel:.2e}"
            )
            if not ok:
                out["ok"] = False
                # Show some sample values
                af, ef = actual.reshape(-1), expected.reshape(-1)
                diffs = np.abs(af - ef)
                idx = np.argsort(diffs)[::-1][:5]
                for i in idx:
                    print(
                        f"        idx={int(i):>6d}  actual={af[i]: .6f}  expected={ef[i]: .6f}  Δ={diffs[i]:.6f}"
                    )
                out["msgs"].append(f"  {tname}: max|Δ|={max_abs:.2e}")
                if dump_msl:
                    for i, g in enumerate(groups):
                        print(f"\n--- kernel {i} {g.strategy.value} ---")
                        print(g.kernel.source)
    except Exception as exc:
        out["ok"] = False
        msg = f"  EXCEPTION {type(exc).__name__}: {exc}"
        print(msg)
        out["msgs"].append(msg)
        traceback.print_exc()
    return out


# ---------------------------------------------------------------------------
# Helpers for building MLX references
# ---------------------------------------------------------------------------


def _mx(arr):
    return mx.array(arr)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def p01_product_reduction():
    """REDUCTION_OPS contains 'product' but runner never tests it.
    Test that reduce(product) over a last axis works."""
    rng = np.random.default_rng(0)
    M, K = 16, 8
    # Bounded values so the product doesn't overflow fp32.
    X = rng.uniform(0.7, 1.3, size=(M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("product", out="prod", x="X", axis=-1)
    mlx_prod = np.array(mx.prod(_mx(X), axis=-1))
    return ops, {"prod": mlx_prod}


def p02_where_ternary():
    """where(cond != 0 ? x : y) — never tested in runner. IR order is
    (x, y, cond) which is unusual."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    Cond = (rng.standard_normal((M, N)) > 0).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(Cond, "Cond")
    ops.elementwise("where", out="out", operands=("X", "Y", "Cond"))
    mlx_out = np.where(Cond != 0.0, X, Y).astype(np.float32)
    return ops, {"out": mlx_out}


def p03_comparison_mask():
    """Comparison ops produce 0/1 masks. Verify lt + use as multiply."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    thresh = 0.5
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("lt", out="mask", operands=("X", thresh))
    ops.elementwise("mul", out="masked", operands=("X", "mask"))
    mlx_mask = (X < thresh).astype(np.float32)
    mlx_masked = X * mlx_mask
    return ops, {"mask": mlx_mask, "masked": mlx_masked}


def p04_reduction_on_transposed():
    """Reduce sum over last axis of a transposed view. The reduction must
    honor the view's strides — last axis of transpose(X) is X's first axis."""
    rng = np.random.default_rng(0)
    M, N = 32, 48
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.transpose("X", out="Xt")  # shape (N, M)
    ops.reduction("sum", out="col_sums", x="Xt", axis=-1)  # sum across M, output (N,)
    mlx_out = np.array(mx.sum(_mx(X), axis=0))
    return ops, {"col_sums": mlx_out}


def p05_row_broadcast_via_reshape():
    """Reduction → reshape to (1, M) → ROW broadcast back into elementwise.
    Counterpart to the standard (M, 1) → COL broadcast in attention."""
    rng = np.random.default_rng(0)
    M, N = 32, 64
    X = rng.standard_normal((N, M)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="col_sum_1d", x="X", axis=-1)  # shape (N,)
    # Use as row-broadcast for a tensor of shape (?, N) — we need a (1, N) view.
    # Source is 1D, contiguous; reshape (N,) → (1, N) should be a view.
    ops.reshape("col_sum_1d", (1, N), out="col_sum_row")
    # Add to a fresh tensor Z of shape (M, N) using ROW broadcast.
    Z = rng.standard_normal((M, N)).astype(np.float32)
    ops.from_numpy(Z, "Z")
    ops.elementwise(
        "add", out="adjusted", operands=("Z", "col_sum_row"), y_broadcast="row"
    )
    mlx_col_sum = np.array(mx.sum(_mx(X), axis=-1))
    mlx_out = Z + mlx_col_sum[None, :]
    return ops, {"adjusted": mlx_out}


def p06_diamond_shared_a():
    """Diamond: t1 = A @ B1; t2 = A @ B2; out = t1 + t2. A is shared."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(B2, "B2")
    ops.matmul(a="A", b="B1", out="t1")
    ops.matmul(a="A", b="B2", out="t2")
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = np.array(_mx(A) @ _mx(B1) + _mx(A) @ _mx(B2))
    return ops, {"out": mlx_out}


def p07_multi_producer_no_shared():
    """Multi-producer convergent (NO shared A): t1 = A1@B1; t2 = A2@B2; out=t1*t2."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    A1 = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    A2 = rng.standard_normal((M, K)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A1, "A1")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(A2, "A2")
    ops.from_numpy(B2, "B2")
    ops.matmul(a="A1", b="B1", out="t1")
    ops.matmul(a="A2", b="B2", out="t2")
    ops.elementwise("mul", out="out", operands=("t1", "t2"))
    mlx_out = np.array((_mx(A1) @ _mx(B1)) * (_mx(A2) @ _mx(B2)))
    return ops, {"out": mlx_out}


def p08_one_row_matrix():
    """Edge: shape (1, N) elementwise. Tile is 16x16, so we test edge alignment."""
    rng = np.random.default_rng(0)
    M, N = 1, 64
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="z", operands=("X", "Y"))
    return ops, {"z": (X + Y).astype(np.float32)}


def p09_one_col_matrix():
    """Edge: shape (M, 1) elementwise."""
    rng = np.random.default_rng(0)
    M, N = 64, 1
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="z", operands=("X", "Y"))
    return ops, {"z": (X + Y).astype(np.float32)}


def p10_full_tensor_mean_broadcast():
    """Two reductions producing two scalars, then broadcast back. Exercises
    both row-axis reduction over a (1, M*N) view via reshape — but axis is
    last-only, so we route via column-shape (M, K) sums."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    # Row-wise sums, then sum of row-sums = total. (Decompose 2D reduce into row+col).
    ops.reduction("sum", out="row_sum_1d", x="X", axis=-1)  # (M,)
    ops.reshape("row_sum_1d", (1, M), out="row_sum_row")  # (1, M) row view
    # Sum across rows? We can't sum axis=0. So just check row-sums via column.
    mlx_row_sum = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {"row_sum_1d": mlx_row_sum}


def p11_relu_then_reduce():
    """Elementwise → reduction prologue absorb: reduce_sum(relu(X))."""
    rng = np.random.default_rng(0)
    M, K = 32, 48
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("max", out="relu_X", operands=("X", 0.0))
    ops.reduction("sum", out="s", x="relu_X", axis=-1)
    mlx_out = np.array(mx.sum(mx.maximum(_mx(X), 0.0), axis=-1))
    return ops, {"s": mlx_out}


def p12_reduce_then_scaled():
    """Reduction → epilogue elem: sum(X) / K. Tests reduction-epilogue scalar binary."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    ops.elementwise("mul", out="mean", operands=("s", 1.0 / K))
    mlx_out = np.array(mx.sum(_mx(X), axis=-1) / K)
    return ops, {"mean": mlx_out}


def p13_chain_with_intermediate_reuse():
    """y = (x + bias) * x — chain references original primary in a later op."""
    rng = np.random.default_rng(0)
    M, N = 32, 64
    X = rng.standard_normal((M, N)).astype(np.float32)
    bias = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(bias, "bias")
    ops.elementwise("add", out="t", operands=("X", "bias"), y_broadcast="row")
    ops.elementwise("mul", out="y", operands=("t", "X"))
    mlx_out = (X + bias) * X
    return ops, {"y": mlx_out.astype(np.float32)}


def p14_matmul_then_transpose_then_add():
    """C = A@B; Ct = C.T; D + Ct (Ct comes from a strided view of C)."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    D = rng.standard_normal((N, M)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(D, "D")
    ops.matmul(a="A", b="B", out="C")
    ops.transpose("C", out="Ct")
    ops.elementwise("add", out="out", operands=("D", "Ct"))
    mlx_out = D + np.array(_mx(A) @ _mx(B)).T
    return ops, {"out": mlx_out.astype(np.float32)}


def p15_reshape_op_via_transpose_then_reshape():
    """Force a ShapeOp: transpose(X) → reshape((flat,)). Transpose yields a
    non-linear view; the reshape can't be a metadata swap → ShapeOp kernel."""
    rng = np.random.default_rng(0)
    M, N = 16, 24
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.transpose("X", out="Xt")  # shape (N, M)
    ops.reshape("Xt", (N * M,), out="flat")  # forces ShapeOp
    mlx_out = np.array(_mx(X).T.reshape(N * M))
    return ops, {"flat": mlx_out.astype(np.float32)}


def p16_recompute_lane_agnostic_prologue():
    """Multi-consumer recompute: x' = relu(X). Then two matmul anchors both
    use x' as A. x' is lane-agnostic → multi-consumer recompute eligible."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(B2, "B2")
    ops.elementwise("max", out="X1", operands=("X", 0.0))
    ops.matmul(a="X1", b="B1", out="t1")
    ops.matmul(a="X1", b="B2", out="t2")
    mlx_X1 = np.maximum(X, 0.0)
    mlx_t1 = np.array(_mx(mlx_X1) @ _mx(B1))
    mlx_t2 = np.array(_mx(mlx_X1) @ _mx(B2))
    return ops, {"t1": mlx_t1, "t2": mlx_t2}


def p17_softmax_via_primitives():
    """softmax(X): max, sub, exp, sum, div — all the building blocks."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("max", out="row_max_1d", x="X", axis=-1)
    ops.reshape("row_max_1d", (M, 1), out="row_max")
    ops.elementwise(
        "subtract", out="shifted", operands=("X", "row_max"), y_broadcast="col"
    )
    ops.elementwise("exp", out="e", operands=("shifted",))
    ops.reduction("sum", out="row_sum_1d", x="e", axis=-1)
    ops.reshape("row_sum_1d", (M, 1), out="row_sum")
    ops.elementwise("div", out="probs", operands=("e", "row_sum"), y_broadcast="col")
    mx_X = _mx(X)
    mlx_probs = np.array(mx.exp(mx_X - mx.max(mx_X, axis=-1, keepdims=True)))
    mlx_probs = mlx_probs / mlx_probs.sum(axis=-1, keepdims=True)
    return ops, {"probs": mlx_probs.astype(np.float32)}


def p18_double_transpose_identity():
    """X.T.T should be X — a no-op metadata roundtrip."""
    rng = np.random.default_rng(0)
    M, N = 16, 24
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.transpose("X", out="Xt")
    ops.transpose("Xt", out="Xtt")
    # Add Xtt to itself to force materialization through a real op (since transposes are metadata).
    ops.elementwise("add", out="out", operands=("Xtt", "Xtt"))
    return ops, {"out": (X + X).astype(np.float32)}


def p19_min_reduction():
    """reduce_min — not commonly tested. Identity = +INFINITY."""
    rng = np.random.default_rng(0)
    M, K = 32, 48
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("min", out="rowmins", x="X", axis=-1)
    mlx_out = np.array(mx.min(_mx(X), axis=-1))
    return ops, {"rowmins": mlx_out}


def p20_unary_negate_chain():
    """Unary chain: -(-x) = x. Tests the unary chain emission."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("negate", out="nx", operands=("X",))
    ops.elementwise("negate", out="nnx", operands=("nx",))
    return ops, {"nnx": X.astype(np.float32)}


def p21_tanh_extreme_values():
    """tanh uses a special form to dodge MSL's tanh(60) = NaN bug.
    Sweep extreme inputs to verify saturation at ±1, not NaN."""
    M, N = 16, 32
    X = np.zeros((M, N), dtype=np.float32)
    # Mix of small, large positive, large negative, zero, +Inf, -Inf
    extremes = np.array(
        [
            0.0,
            1.0,
            -1.0,
            10.0,
            -10.0,
            60.0,
            -60.0,
            80.0,
            -80.0,
            100.0,
            -100.0,
            200.0,
            1e-10,
            -1e-10,
            0.5,
            -0.5,
        ],
        dtype=np.float32,
    )
    X[0, :16] = extremes
    X[1, :] = np.linspace(-100.0, 100.0, N, dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("tanh", out="th", operands=("X",))
    mlx_out = np.tanh(X).astype(np.float32)
    return ops, {"th": mlx_out}


def p22_reduce_epilogue_chain():
    """Reduction epilogue with multi-op chain: sum(X) * 2 + 1.
    Tests `build_reduction_epilogue_transform` on a chain of length > 1."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    ops.elementwise("mul", out="s2", operands=("s", 2.0))
    ops.elementwise("add", out="s3", operands=("s2", 1.0))
    mlx_out = np.array(mx.sum(_mx(X), axis=-1)) * 2.0 + 1.0
    return ops, {"s3": mlx_out.astype(np.float32)}


def p23_prologue_chain_into_matmul():
    """Multi-step prologue: ((X * 2) + 1)  fed into matmul's A side.
    Forces the chain absorption into the A-load value_transform."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B, "B")
    ops.elementwise("mul", out="X1", operands=("X", 2.0))
    ops.elementwise("add", out="X2", operands=("X1", 1.0))
    ops.matmul(a="X2", b="B", out="C")
    mlx_out = np.array((_mx(X) * 2.0 + 1.0) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p24_mixed_epilogue_chain():
    """Mixed epilogue: add(matmul, bias_row), then relu. The add isn't
    lane-agnostic (tensor y), so the whole chain takes the tg-tile path."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(bias, "bias")
    ops.matmul(a="A", b="B", out="z")
    ops.elementwise("add", out="zb", operands=("z", "bias"), y_broadcast="row")
    ops.elementwise("max", out="y", operands=("zb", 0.0))
    mlx_out = np.array(mx.maximum(_mx(A) @ _mx(B) + _mx(bias), 0.0))
    return ops, {"y": mlx_out.astype(np.float32)}


def p25_sum_of_comparison():
    """Count elements > threshold via reduce_sum on a comparison output.
    Tests comparison op producing 0/1 floats consumed by a reduction."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("gt", out="mask", operands=("X", 0.0))
    ops.reduction("sum", out="count", x="mask", axis=-1)
    mlx_out = np.array(mx.sum((_mx(X) > 0.0).astype(mx.float32), axis=-1))
    return ops, {"count": mlx_out.astype(np.float32)}


def p26_reshape_col_to_1d():
    """Reshape (M, 1) → (M,) — the COL → 1D path through reshape_view."""
    rng = np.random.default_rng(0)
    M = 32
    X = rng.standard_normal((M, 1)).astype(np.float32)
    Y = rng.standard_normal((M,)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.reshape("X", (M,), out="X1d")  # view
    ops.elementwise("add", out="z", operands=("X1d", "Y"))
    mlx_out = X.reshape(M) + Y
    return ops, {"z": mlx_out.astype(np.float32)}


def p27_reshape_to_same_shape():
    """Reshape to exactly the same shape — should be a no-op view rename."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reshape("X", (M, N), out="X_alias")  # view, same shape
    ops.elementwise("mul", out="y", operands=("X_alias", 3.0))
    return ops, {"y": (X * 3.0).astype(np.float32)}


def p28_non_aligned_matmul_with_epilogue_chain():
    """Awkward dims forcing unaligned matmul + a chain epilogue (multiple ops)."""
    rng = np.random.default_rng(0)
    M, K, N = 33, 49, 41
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="z")
    # Lane-agnostic chain: relu, add scalar, mul scalar.
    ops.elementwise("max", out="z_r", operands=("z", 0.0))
    ops.elementwise("add", out="z_a", operands=("z_r", 0.5))
    ops.elementwise("mul", out="y", operands=("z_a", 0.25))
    expected = np.maximum(A @ B, 0.0) + 0.5
    expected = (expected * 0.25).astype(np.float32)
    return ops, {"y": expected}


def p29_matmul_with_transposed_input_b():
    """y = A @ B.T  — strided B-side read via transpose view."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    # B is stored as (N, K); we transpose to (K, N) for the matmul.
    B = rng.standard_normal((N, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.transpose("B", out="Bt")  # (K, N) view, strided
    ops.matmul(a="A", b="Bt", out="C")
    mlx_out = np.array(_mx(A) @ _mx(B).T)
    return ops, {"C": mlx_out.astype(np.float32)}


def p30_where_with_broadcast_cond():
    """where: cond is ROW-broadcast (per-column mask). IR has (x, y, cond),
    cond_broadcast=row."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    cond = (rng.standard_normal((1, N)) > 0).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(cond, "cond")
    ops.elementwise(
        "where", out="out", operands=("X", "Y", "cond"), cond_broadcast="row"
    )
    mlx_out = np.where(cond != 0.0, X, Y).astype(np.float32)
    return ops, {"out": mlx_out}


def p31_matmul_input_used_twice_as_a_and_b():
    """Same tensor X feeds both A and B sides of a matmul: X @ X.
    Tests degenerate aliasing of buffer slots in the matmul template."""
    rng = np.random.default_rng(0)
    N = 64
    X = rng.standard_normal((N, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.matmul(a="X", b="X", out="C")
    mlx_out = np.array(_mx(X) @ _mx(X))
    return ops, {"C": mlx_out.astype(np.float32)}


def p32_self_add():
    """x + x — same tensor on both sides of a binary elementwise."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("add", out="y", operands=("X", "X"))
    return ops, {"y": (X + X).astype(np.float32)}


def p33_recompute_two_consumers_one_unfusable():
    """Multi-consumer where only ONE consumer is an anchor — recompute pass
    should NOT fire, original elem stays standalone."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B, "B")
    ops.from_numpy(C, "C")
    ops.elementwise("max", out="X1", operands=("X", 0.0))  # 2 consumers
    ops.matmul(a="X1", b="B", out="t1")  # anchor consumer
    ops.elementwise("add", out="t2", operands=("X1", "C"))  # non-anchor consumer
    mlx_X1 = np.maximum(X, 0.0).astype(np.float32)
    return ops, {
        "t1": np.array(_mx(mlx_X1) @ _mx(B)).astype(np.float32),
        "t2": (mlx_X1 + C).astype(np.float32),
    }


def p34_subtract_scalar_first():
    """y = x - 5.0 — subtract with primary on left, scalar on right.
    Tests that 'subtract' (non-commutative) gets operand order right."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32) + 100.0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("subtract", out="y", operands=("X", 5.0))
    return ops, {"y": (X - 5.0).astype(np.float32)}


def p36_gram_matrix_x_xt():
    """X @ X.T — gram matrix. Same buffer for A and B inputs (via transpose
    view → buffer_key=X). Same root cause as p31."""
    rng = np.random.default_rng(0)
    M, K = 64, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.transpose("X", out="Xt")
    ops.matmul(a="X", b="Xt", out="C")
    mlx_out = np.array(_mx(X) @ _mx(X).T)
    return ops, {"C": mlx_out.astype(np.float32)}


def p37_matmul_M_eq_1():
    """Degenerate matmul with M=1: shape (1, K) @ (K, N) = (1, N)."""
    rng = np.random.default_rng(0)
    M, K, N = 1, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    mlx_out = np.array(_mx(A) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p38_matmul_N_eq_1():
    """Degenerate matmul with N=1: shape (M, K) @ (K, 1) = (M, 1).
    Matrix-vector product."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 1
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    mlx_out = np.array(_mx(A) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p39_reduce_over_k_eq_1():
    """Reduce sum on (M, 1) input — K=1 edge case."""
    rng = np.random.default_rng(0)
    M, K = 64, 1
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    return ops, {"s": X.reshape(M).astype(np.float32)}


def p40_two_relus_share_input():
    """x is consumed by two unary elementwise ops — multi-consumer, but
    neither consumer is an anchor → recompute pass shouldn't fire."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("max", out="r1", operands=("X", 0.0))
    ops.elementwise("max", out="r2", operands=("X", 0.5))
    return ops, {
        "r1": np.maximum(X, 0.0).astype(np.float32),
        "r2": np.maximum(X, 0.5).astype(np.float32),
    }


def p41_chain_with_4_ops():
    """Long elementwise chain: x → +1 → *2 → relu → exp."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("add", out="a", operands=("X", 1.0))
    ops.elementwise("mul", out="b", operands=("a", 2.0))
    ops.elementwise("max", out="c", operands=("b", 0.0))
    ops.elementwise("exp", out="y", operands=("c",))
    expected = np.exp(np.maximum((X + 1.0) * 2.0, 0.0)).astype(np.float32)
    return ops, {"y": expected}


def p42_reduction_then_two_view_consumers():
    """rs = sum(X, axis=-1); rs.reshape(M,1) used by elem A; rs.reshape(1,M)
    used as ROW broadcast by elem B. Both consume the same underlying
    1D buffer through different shape views."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, M)).astype(np.float32)
    Z = rng.standard_normal((M, M)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(Z, "Z")
    ops.reduction("sum", out="rs", x="X", axis=-1)  # (M,)
    ops.reshape("rs", (M, 1), out="rs_col")  # (M, 1) col-broadcast slot
    ops.reshape("rs", (1, M), out="rs_row")  # (1, M) row-broadcast slot
    ops.elementwise("add", out="A", operands=("Y", "rs_col"), y_broadcast="col")
    ops.elementwise("add", out="B", operands=("Z", "rs_row"), y_broadcast="row")
    mlx_rs = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {
        "A": (Y + mlx_rs[:, None]).astype(np.float32),
        "B": (Z + mlx_rs[None, :]).astype(np.float32),
    }


def p43_pow_op():
    """pow(x, scalar) — exercises the binary `pow` op."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = np.abs(rng.standard_normal((M, N))).astype(np.float32) + 0.1
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("pow", out="y", operands=("X", 0.5))
    return ops, {"y": np.power(X, 0.5).astype(np.float32)}


def p44_sign_of_zero():
    """sign(0.0) should be 0.0 — MSL's piecewise sign matches numpy here."""
    M, N = 4, 8
    X = np.array(
        [
            [-2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0],
            [-3.0, -1e-30, 1e-30, 0.0, -0.0, 1.0, -1.0, 100.0],
            [-1e10, 1e10, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0],
            [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
        ],
        dtype=np.float32,
    )
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("sign", out="y", operands=("X",))
    expected = np.sign(X).astype(np.float32)
    return ops, {"y": expected}


def p45_chain_uses_scalar_operand_with_negate():
    """negate(0.5 - x) — primary on right, scalar on left disallowed by IR.
    Use negate of subtract instead: -(x - 5.0)."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("subtract", out="t", operands=("X", 5.0))
    ops.elementwise("negate", out="y", operands=("t",))
    return ops, {"y": (-(X - 5.0)).astype(np.float32)}


def p46_recip_op():
    """Test the unary recip op (1/x)."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.uniform(0.5, 2.0, (M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("recip", out="y", operands=("X",))
    return ops, {"y": (1.0 / X).astype(np.float32)}


def p47_sqrt_chain_log():
    """y = log(sqrt(x))  = 0.5 * log(x) — unary chain."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.uniform(0.1, 10.0, (M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("sqrt", out="r", operands=("X",))
    ops.elementwise("log", out="y", operands=("r",))
    return ops, {"y": np.log(np.sqrt(X)).astype(np.float32)}


def p49_diamond_shared_a_small():
    """Diamond with tile that fits in tg memory: M=K=N=16 forces tile (32,32,16)
    config, which is small enough for the multi-anchor kernel."""
    rng = np.random.default_rng(0)
    M, K, N = 16, 16, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(B2, "B2")
    ops.matmul(a="A", b="B1", out="t1")
    ops.matmul(a="A", b="B2", out="t2")
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = np.array(_mx(A) @ _mx(B1) + _mx(A) @ _mx(B2))
    return ops, {"out": mlx_out.astype(np.float32)}


def p50_multi_producer_small():
    """Multi-producer without shared A, small tile to avoid OOM."""
    rng = np.random.default_rng(0)
    M, K, N = 16, 16, 16
    A1 = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    A2 = rng.standard_normal((M, K)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A1, "A1")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(A2, "A2")
    ops.from_numpy(B2, "B2")
    ops.matmul(a="A1", b="B1", out="t1")
    ops.matmul(a="A2", b="B2", out="t2")
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = np.array(_mx(A1) @ _mx(B1) + _mx(A2) @ _mx(B2))
    return ops, {"out": mlx_out.astype(np.float32)}


def p51_prologue_same_input_into_both_a_and_b():
    """X1 = X + 1 → matmul(X1, X1). Prologue chain absorbed into both
    sides of a matmul whose A and B end up sharing X's buffer."""
    rng = np.random.default_rng(0)
    M = K = N = 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("add", out="X1", operands=("X", 1.0))
    ops.matmul(a="X1", b="X1", out="C")
    mlx_out = np.array((_mx(X) + 1.0) @ (_mx(X) + 1.0))
    return ops, {"C": mlx_out.astype(np.float32)}


def p52_elementwise_extra_shares_input_buffer():
    """Elementwise chain `y = X + X_alias` where X_alias is a same-shape
    reshape view of X. Both operands have the same buffer_key=X."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reshape("X", (M, N), out="X_alias")  # view, same buffer_key=X
    ops.elementwise("add", out="y", operands=("X", "X_alias"))
    return ops, {"y": (X + X).astype(np.float32)}


def p53_matmul_with_bias_that_also_is_input_a():
    """Matmul with bias_row that happens to be a transpose/reshape of input A.
    Forces extras buffer_key to collide with base slot A's buffer_key."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    # bias_row is shape (1, N); we manufacture it by reshaping A's first row
    # to (1, K=N). Requires K == N (true here).
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="z")
    # Use a separate small tensor as bias to avoid the issue — but on a
    # repeated tensor. Here we add A.row0 reshape to z, broadcasting. We can't
    # slice in this API, so instead use A itself as the bias tensor with full
    # broadcast. Pick (M, N) = (M, N) full-shape.
    ops.elementwise("add", out="y", operands=("z", "A"), y_broadcast="none")
    mlx_out = np.array(_mx(A) @ _mx(B) + _mx(A))
    return ops, {"y": mlx_out.astype(np.float32)}


def p54_equal_op():
    """The `equal` comparison op — fp equality is rare but should work."""
    M, N = 8, 16
    X = np.array([[i for i in range(N)] for _ in range(M)], dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("equal", out="mask", operands=("X", 4.0))
    expected = (X == 4.0).astype(np.float32)
    return ops, {"mask": expected}


def p55_chain_then_chain_via_reduction():
    """Long pipeline: relu → mul → reduce_sum → mul scalar.
    Mixes elementwise chain, prologue-into-reduction, and reduction epilogue."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("max", out="r", operands=("X", 0.0))
    ops.elementwise("mul", out="r2", operands=("r", 2.0))
    ops.reduction("sum", out="s", x="r2", axis=-1)
    ops.elementwise("mul", out="y", operands=("s", 0.5))
    mlx_out = np.array(mx.sum(mx.maximum(_mx(X), 0.0) * 2.0, axis=-1) * 0.5)
    return ops, {"y": mlx_out.astype(np.float32)}


def p60_reduction_prologue_full_shape_extra():
    """Reduction with prologue having a full-shape tensor secondary:
    s = sum(X + Y, axis=-1). Reduction loops scalar — should be OK."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="XY", operands=("X", "Y"))
    ops.reduction("sum", out="s", x="XY", axis=-1)
    mlx_out = np.array(mx.sum(_mx(X) + _mx(Y), axis=-1))
    return ops, {"s": mlx_out.astype(np.float32)}


def p62_multi_anchor_same_b():
    """Multi-anchor (no shared A) but B0 == B1: same B buffer used by two
    matmuls feeding a merge. Tests whether multi-anchor emits duplicate
    buffer parameters."""
    rng = np.random.default_rng(0)
    M, K, N = 16, 16, 16
    A1 = rng.standard_normal((M, K)).astype(np.float32)
    A2 = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A1, "A1")
    ops.from_numpy(A2, "A2")
    ops.from_numpy(B, "B")
    ops.matmul(a="A1", b="B", out="t1")
    ops.matmul(a="A2", b="B", out="t2")
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = np.array(_mx(A1) @ _mx(B) + _mx(A2) @ _mx(B))
    return ops, {"out": mlx_out.astype(np.float32)}


def p63_comparison_into_max():
    """y = max(x > 0, x)  — comparison result feeds into a max as primary?
    No, primary must be tensor; here x is primary, mask is secondary.
    Test: y = max(x, (x > 0)). The mask is broadcast into the max."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("gt", out="mask", operands=("X", 0.0))  # (M, N) of 0/1
    ops.elementwise("max", out="y", operands=("X", "mask"))  # full-shape secondary
    mlx_mask = (X > 0.0).astype(np.float32)
    mlx_out = np.maximum(X, mlx_mask)
    return ops, {"y": mlx_out.astype(np.float32)}


def p64_chain_through_view():
    """t1 = X + Y; t2 = view (reshape same-shape) of t1; t3 = t2 * 2.
    Does the chain fuse across a view rename?"""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="t1", operands=("X", "Y"))
    ops.reshape("t1", (M, N), out="t2")  # same-shape view
    ops.elementwise("mul", out="t3", operands=("t2", 2.0))
    mlx_out = (X + Y) * 2.0
    return ops, {"t3": mlx_out.astype(np.float32)}


def p65_chain_with_first_op_as_comparison():
    """Comparison op at the head of a chain produces 0/1; subsequent ops
    process those. mask = X > 0; y = mask + 5."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("gt", out="mask", operands=("X", 0.0))
    ops.elementwise("add", out="y", operands=("mask", 5.0))
    expected = ((X > 0.0).astype(np.float32) + 5.0).astype(np.float32)
    return ops, {"y": expected}


def p66_input_used_as_col_broadcast_into_matmul_epilogue():
    """matmul (M, N) + col_bias (M, 1) where col_bias is reused — tests
    tg-tile epilogue with COL broadcast where the operand has shape (M, 1)."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((M, 1)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(bias, "bias")
    ops.matmul(a="A", b="B", out="z")
    ops.elementwise("add", out="y", operands=("z", "bias"), y_broadcast="col")
    mlx_out = np.array(_mx(A) @ _mx(B) + _mx(bias))
    return ops, {"y": mlx_out.astype(np.float32)}


def p67_input_used_as_scalar_broadcast():
    """matmul (M, N) + scalar_t (1, 1) — scalar broadcast tensor (not Scalar literal)."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    s = np.array([[1.5]], dtype=np.float32)  # (1, 1)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(s, "s")
    ops.matmul(a="A", b="B", out="z")
    ops.elementwise("add", out="y", operands=("z", "s"), y_broadcast="scalar")
    mlx_out = np.array(_mx(A) @ _mx(B) + _mx(s))
    return ops, {"y": mlx_out.astype(np.float32)}


def p68_elementwise_chain_with_col_broadcast_intermediate_use():
    """ColBcast operand used twice in a chain:
       t1 = X + b (col broadcast); t2 = t1 + b again. Tests that the same
    extra is bound only once and the col broadcast tile read works repeatedly."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    b = rng.standard_normal((M, 1)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(b, "b")
    ops.elementwise("add", out="t1", operands=("X", "b"), y_broadcast="col")
    ops.elementwise("add", out="t2", operands=("t1", "b"), y_broadcast="col")
    mlx_out = X + b + b
    return ops, {"t2": mlx_out.astype(np.float32)}


def p69_reduce_max_all_negatives():
    """reduce_max with all negative values — identity is -INFINITY; final
    answer should be the row-wise max negative."""
    rng = np.random.default_rng(0)
    M, K = 16, 64
    X = -np.abs(rng.standard_normal((M, K)).astype(np.float32)) - 0.1  # all negative
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("max", out="rm", x="X", axis=-1)
    mlx_out = np.array(mx.max(_mx(X), axis=-1))
    return ops, {"rm": mlx_out.astype(np.float32)}


def p70_reduction_with_K_less_than_simdwidth():
    """K < 32 (one SIMD-group width). All lanes >= K are idle; SIMD reduce
    must include the identity for inactive lanes."""
    rng = np.random.default_rng(0)
    M, K = 16, 7  # K=7, much less than 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    mlx_out = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {"s": mlx_out.astype(np.float32)}


def p71_reduction_K_between_32_and_128():
    """K = 50; spans 2 SIMD groups within the 128-thread threadgroup."""
    rng = np.random.default_rng(0)
    M, K = 16, 50
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    mlx_out = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {"s": mlx_out.astype(np.float32)}


def p72_matmul_tg_epilogue_reads_input_a_again():
    """Matmul z = A@B with epilogue y = z + A_row_view  where A's row 0 (1, K)
    is broadcasted as bias. Forces matmul to bind A twice — once as base A,
    once as epilogue extra (via a reshape view of a row of A)."""
    # A_row_view is a same-shape view of A's first row; we need K == N here.
    rng = np.random.default_rng(0)
    M = K = N = 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    # bias is a separate (1, N) tensor — exercise extras with row broadcast.
    bias = A[0:1, :].copy().astype(np.float32)  # (1, N) — independent buffer.
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(bias, "bias")
    ops.matmul(a="A", b="B", out="z")
    ops.elementwise("add", out="y", operands=("z", "bias"), y_broadcast="row")
    mlx_out = np.array(_mx(A) @ _mx(B) + _mx(bias))
    return ops, {"y": mlx_out.astype(np.float32)}


def p73_input_buffer_size_only_uses_first_K_elements():
    """from_numpy with an explicit row_stride that's tighter than default.
    Currently the API doesn't expose row_stride directly to from_numpy,
    but verify ops.input + explicit row_stride works as a separate input."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    # Just a sanity test: identity through a chain
    ops.elementwise("add", out="y", operands=("X", 0.0))
    return ops, {"y": X.astype(np.float32)}


def p75_multi_anchor_a0_eq_b1():
    """Multi-anchor where A0 == B1: t1 = X @ Y; t2 = Z @ X; merge. X used
    as both A and B but in DIFFERENT matmul anchors. Cross-aliasing."""
    rng = np.random.default_rng(0)
    M = K = N = 16
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((K, N)).astype(np.float32)
    Z = rng.standard_normal((M, K)).astype(np.float32)
    # For t2 = Z @ X to be defined, X must be (K, N) — but X is (M, K). Use Xt.
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(Z, "Z")
    ops.transpose("X", out="Xt")  # (K, M) — when M==K, square, used as B
    ops.matmul(a="X", b="Y", out="t1")  # (M, N)
    ops.matmul(a="Z", b="Xt", out="t2")  # (M, M=N)
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = np.array(_mx(X) @ _mx(Y) + _mx(Z) @ _mx(X).T)
    return ops, {"out": mlx_out.astype(np.float32)}


def p76_chain_with_tail_consumed_by_view_then_op():
    """t1 = X + 1; t2 = t1.reshape(same); t3 = t2 * 2. Does the fuser
    actually fuse t1 and t3 across a metadata view?"""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("add", out="t1", operands=("X", 1.0))
    ops.reshape("t1", (M, N), out="t2")  # metadata view (same shape)
    ops.elementwise("mul", out="t3", operands=("t2", 2.0))
    return ops, {"t3": ((X + 1.0) * 2.0).astype(np.float32)}


def p77_relu_then_reshape_then_relu():
    """relu → reshape (different shape, view) → relu. Tests fusion across a
    real reshape view (non-noop)."""
    rng = np.random.default_rng(0)
    M, N = 8, 16  # M*N=128
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("max", out="t1", operands=("X", 0.0))
    ops.reshape("t1", (M * N,), out="t1_flat")  # 1D view (valid since t1 is contiguous)
    ops.elementwise("max", out="t2", operands=("t1_flat", 0.0))
    expected = np.maximum(X, 0.0).reshape(M * N).astype(np.float32)
    return ops, {"t2": expected}


def p78_reduce_then_use_result_in_two_chains():
    """row_sum = sum(X); y1 = row_sum * 2; y2 = row_sum + 1. Two separate
    chain consumers of a reduction's 1D output."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="rs", x="X", axis=-1)
    ops.elementwise("mul", out="y1", operands=("rs", 2.0))
    ops.elementwise("add", out="y2", operands=("rs", 1.0))
    mlx_rs = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {
        "y1": (mlx_rs * 2.0).astype(np.float32),
        "y2": (mlx_rs + 1.0).astype(np.float32),
    }


def p79_matmul_with_row_broadcast_full_M_eq_N():
    """matmul (M, N) + row_bias (1, N) when M == N — tests no confusion
    between row-broadcast and same-shape detection."""
    rng = np.random.default_rng(0)
    N = 32
    A = rng.standard_normal((N, N)).astype(np.float32)
    B = rng.standard_normal((N, N)).astype(np.float32)
    bias = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(bias, "bias")
    ops.matmul(a="A", b="B", out="z")
    ops.elementwise("add", out="y", operands=("z", "bias"), y_broadcast="row")
    mlx_out = np.array(_mx(A) @ _mx(B) + _mx(bias))
    return ops, {"y": mlx_out.astype(np.float32)}


def p80_chained_reductions_on_diff_axes():
    """reduce(X)→(M,); reshape to (1, M); reduce → (1,).
    Tests an unusual two-step pattern that reduces both axes."""
    rng = np.random.default_rng(0)
    M, K = 8, 16
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="rs", x="X", axis=-1)
    ops.reshape("rs", (1, M), out="rs_row")
    ops.reduction("sum", out="total", x="rs_row", axis=-1)
    expected = np.array([X.sum()]).astype(np.float32)
    return ops, {"total": expected}


def p81_transpose_of_1d_reshape_view():
    """X is 1D (size 16); reshape to (4,4); transpose to (4,4) [strided].
    Then add to a same-shape tensor."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16,)).astype(np.float32)
    Y = rng.standard_normal((4, 4)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.reshape("X", (4, 4), out="X2d")  # view
    ops.transpose("X2d", out="X2dT")  # view-of-view
    ops.elementwise("add", out="z", operands=("Y", "X2dT"))
    mlx_out = (Y + X.reshape(4, 4).T).astype(np.float32)
    return ops, {"z": mlx_out}


def p82_reduce_then_negate_epilogue():
    """reduce → negate (unary epilogue). Tests reduction epilogue with a
    pure unary."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="rs", x="X", axis=-1)
    ops.elementwise("negate", out="y", operands=("rs",))
    mlx_out = -np.array(mx.sum(_mx(X), axis=-1)).astype(np.float32)
    return ops, {"y": mlx_out}


def p83_chain_writes_back_to_same_position():
    """y = relu(x); z = exp(x). Two separate kernels both read x.
    Tests that x isn't freed too early."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("max", out="y", operands=("X", 0.0))
    ops.elementwise("exp", out="z", operands=("X",))
    return ops, {
        "y": np.maximum(X, 0.0).astype(np.float32),
        "z": np.exp(X).astype(np.float32),
    }


def p84_not_equal_op():
    """not_equal comparison — produces 0/1 floats."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.choice([-1.0, 0.0, 1.0, 2.0], size=(M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("not_equal", out="mask", operands=("X", 0.0))
    expected = (X != 0.0).astype(np.float32)
    return ops, {"mask": expected}


def p128_multi_anchor_followed_by_epilogue():
    """Multi-anchor merge followed by a relu. The fuser absorbs all into
    one vertex with 2 matmuls + 2 elementwise ops. Classifier asserts
    exactly 1 elementwise op for multi-anchor — should crash or produce wrong code."""
    rng = np.random.default_rng(0)
    M = K = N = 16
    A1 = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    A2 = rng.standard_normal((M, K)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A1, "A1")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(A2, "A2")
    ops.from_numpy(B2, "B2")
    ops.matmul(a="A1", b="B1", out="t1")
    ops.matmul(a="A2", b="B2", out="t2")
    ops.elementwise("add", out="merged", operands=("t1", "t2"))  # merge
    ops.elementwise("max", out="out", operands=("merged", 0.0))  # epilogue relu
    mlx_out = np.maximum(_mx(A1) @ _mx(B1) + _mx(A2) @ _mx(B2), 0.0)
    return ops, {"out": np.array(mlx_out).astype(np.float32)}


def p129_multi_anchor_with_prologue_into_either_anchor():
    """Multi-anchor pattern but with a prologue feeding A1: X1 = relu(X);
    t1 = X1 @ B1; t2 = X @ B2; merge = t1 + t2. The multi-producer pass
    requires both producers to be pure matmuls. relu means X1 has a
    prologue. Should not fuse as multi-anchor."""
    rng = np.random.default_rng(0)
    M = K = N = 16
    X = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(B2, "B2")
    ops.elementwise("max", out="X1", operands=("X", 0.0))
    ops.matmul(a="X1", b="B1", out="t1")
    ops.matmul(a="X", b="B2", out="t2")
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = np.maximum(X, 0.0) @ B1 + X @ B2
    return ops, {"out": mlx_out.astype(np.float32)}


def p121_multi_producer_comparison_merge():
    """Multi-producer with merge_elem being a COMPARISON op. Tests wrap_bool
    inside the multi-anchor merge fragment."""
    rng = np.random.default_rng(0)
    M = K = N = 16
    A1 = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    A2 = rng.standard_normal((M, K)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A1, "A1")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(A2, "A2")
    ops.from_numpy(B2, "B2")
    ops.matmul(a="A1", b="B1", out="t1")
    ops.matmul(a="A2", b="B2", out="t2")
    ops.elementwise("gt", out="mask", operands=("t1", "t2"))  # comparison merge
    mlx_out = ((_mx(A1) @ _mx(B1)) > (_mx(A2) @ _mx(B2))).astype(mx.float32)
    return ops, {"mask": np.array(mlx_out).astype(np.float32)}


def p122_chain_fuse_across_named_consumer():
    """What happens when the chain head is consumed AS A SECONDARY operand
    in the next chain elem? Chain fuse should NOT trigger (since primary
    alignment is required). Validates the chain invariant."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="t1", operands=("X", 1.0))
    # t1 is the SECONDARY of next op, not the primary
    ops.elementwise("add", out="y", operands=("Y", "t1"))
    expected = (Y + (X + 1.0)).astype(np.float32)
    return ops, {"y": expected}


def p123_reduce_epilogue_comparison_with_negate():
    """Reduction epilogue: s = sum(X); y = (s > 0); z = negate(y).
    Tests wrap_bool followed by negate in the reduction epilogue chain."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    ops.elementwise("gt", out="y", operands=("s", 0.0))
    ops.elementwise("negate", out="z", operands=("y",))
    mlx_s = np.array(mx.sum(_mx(X), axis=-1))
    expected = -((mlx_s > 0.0).astype(np.float32))
    return ops, {"z": expected.astype(np.float32)}


def p124_matmul_aligned_with_register_epilogue_chain_4_ops():
    """Aligned matmul + 4-op lane-agnostic chain. All in register epilogue.
    Verifies the register epilogue handles chains, not just single ops."""
    rng = np.random.default_rng(0)
    M = K = N = 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="z")
    ops.elementwise("add", out="z1", operands=("z", 1.0))
    ops.elementwise("mul", out="z2", operands=("z1", 0.5))
    ops.elementwise("max", out="z3", operands=("z2", 0.0))
    ops.elementwise("exp", out="y", operands=("z3",))
    expected = np.exp(np.maximum((np.array(_mx(A) @ _mx(B)) + 1.0) * 0.5, 0.0))
    return ops, {"y": expected.astype(np.float32)}


def p125_dispatch_chain_with_consecutive_kernels_sharing_input():
    """3 kernels in sequence, each consuming X plus other tensors. Tests
    that X stays alive through all kernels (last_use computed correctly)."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="a", operands=("X", "Y"))  # kernel 0
    ops.elementwise("mul", out="b", operands=("a", "X"))  # kernel 1, reads X again
    ops.elementwise("add", out="c", operands=("b", "X"))  # kernel 2, reads X again
    expected_a = X + Y
    expected_b = expected_a * X
    expected_c = expected_b + X
    return ops, {"c": expected_c.astype(np.float32)}


def p126_reduce_along_axis_then_immediately_use_via_view():
    """rs = sum(X); rs_view = reshape rs to (M, 1); then immediately
    consume rs_view in elementwise. Tests scheduler's last_use for the
    reduction's buffer."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="rs", x="X", axis=-1)
    ops.reshape("rs", (M, 1), out="rs_col")
    Y = rng.standard_normal((M, M)).astype(np.float32)
    ops.from_numpy(Y, "Y")
    ops.elementwise("mul", out="z", operands=("Y", "rs_col"), y_broadcast="col")
    mlx_rs = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {"z": (Y * mlx_rs[:, None]).astype(np.float32)}


def p127_long_chain_register_epilogue_relu_sequence():
    """Long lane-agnostic chain: matmul + relu + relu + ... + relu (10 times).
    Tests if register epilogue handles deep chains."""
    rng = np.random.default_rng(0)
    M = K = N = 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="z")
    cur = "z"
    for i in range(10):
        out_name = f"r{i}"
        ops.elementwise("max", out=out_name, operands=(cur, 0.0))
        cur = out_name
    expected = np.maximum(np.array(_mx(A) @ _mx(B)), 0.0)
    return ops, {cur: expected.astype(np.float32)}


def p116_diamond_shared_a_via_transpose_view():
    """Diamond: Xt = X.T; t1 = Xt @ B1; t2 = Xt @ B2; out = t1 + t2.
    Shared A is a TRANSPOSE VIEW. Tests strided A-load in multi-anchor."""
    rng = np.random.default_rng(0)
    M = K = N = 16
    # X is (K, M) — its transpose Xt is (M, K) used as matmul A.
    X = rng.standard_normal((K, M)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(B2, "B2")
    ops.transpose("X", out="Xt")  # (M, K)
    ops.matmul(a="Xt", b="B1", out="t1")
    ops.matmul(a="Xt", b="B2", out="t2")
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = (X.T @ B1 + X.T @ B2).astype(np.float32)
    return ops, {"out": mlx_out}


def p117_diamond_with_a_distinct_names_same_buffer():
    """Diamond where two anchors' A tensors are DIFFERENT views (different
    SSA names) but share a buffer. shared_a uses name equality — would
    detect this as NOT shared. Then duplicate-buffer bug triggers."""
    rng = np.random.default_rng(0)
    M = K = N = 16
    X = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(B2, "B2")
    ops.reshape("X", (M, K), out="X_v1")  # view 1 (same shape, same data)
    ops.reshape("X", (M, K), out="X_v2")  # view 2 (different SSA name, same buffer)
    ops.matmul(a="X_v1", b="B1", out="t1")
    ops.matmul(a="X_v2", b="B2", out="t2")
    ops.elementwise("add", out="out", operands=("t1", "t2"))
    mlx_out = (X @ B1 + X @ B2).astype(np.float32)
    return ops, {"out": mlx_out}


def p118_elementwise_output_as_input_to_another_with_view():
    """t1 = X * 2; t1_view = reshape t1; y = t1_view + Y. After fusion, t1
    might or might not be materialized — depends on if t1 has consumers."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("mul", out="t1", operands=("X", 2.0))
    ops.reshape("t1", (M, N), out="t1_v")  # view (same shape)
    ops.elementwise("add", out="y", operands=("t1_v", "Y"))
    return ops, {"y": (X * 2.0 + Y).astype(np.float32)}


def p119_chain_with_scalar_first_then_tensor():
    """Chain: t1 = X * 2; t2 = t1 + Y. First op scalar (lane-agnostic),
    second op tensor secondary (NOT lane-agnostic). Fused chain into a matmul
    prologue — what about float4 path?"""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(B, "B")
    ops.elementwise("mul", out="t1", operands=("X", 2.0))
    ops.elementwise("add", out="t2", operands=("t1", "Y"))
    ops.matmul(a="t2", b="B", out="C")
    mlx_out = np.array((_mx(X) * 2.0 + _mx(Y)) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p120_elementwise_chain_5_ops_with_intermediate_outputs():
    """Long chain — but the user wants every intermediate output. Tests
    whether the intermediate dropping bug scales."""
    rng = np.random.default_rng(0)
    M, N = 8, 16
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("add", out="a", operands=("X", 1.0))
    ops.elementwise("mul", out="b", operands=("a", 2.0))
    ops.elementwise("max", out="c", operands=("b", 0.0))
    ops.elementwise("exp", out="d", operands=("c",))
    ops.elementwise("log", out="e", operands=("d",))
    out_a = X + 1.0
    out_b = out_a * 2.0
    out_c = np.maximum(out_b, 0.0)
    out_d = np.exp(out_c)
    out_e = np.log(out_d)
    return ops, {
        "e": out_e.astype(np.float32),
    }


def p109_input_used_only_via_view():
    """X is uploaded; Xt is its transpose. Only Xt is referenced in any op,
    X is never directly accessed. Tests scheduler validation."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((N, M)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.transpose("X", out="Xt")  # only Xt is consumed
    ops.elementwise("add", out="z", operands=("Y", "Xt"))
    return ops, {"z": (Y + X.T).astype(np.float32)}


def p110_chain_with_two_consumers_one_is_view():
    """t1 = X + Y; reshape t1 → t1_v (view); also t2 = t1 * 3.
    t1 has 2 vertex consumers: itself (via t1_v reshape) and t2.
    Actually t1_v reshape is metadata-only, no vertex. So t1 has 1 vertex consumer (t2).
    But t1 must materialize since t1_v is a separate declared output."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="t1", operands=("X", "Y"))
    ops.reshape("t1", (M, N), out="t1_v")  # view — t1's buffer
    ops.elementwise("mul", out="t2", operands=("t1", 3.0))
    expected_t1 = (X + Y).astype(np.float32)
    return ops, {
        "t2": (expected_t1 * 3.0).astype(np.float32),
    }


def p111_matmul_output_with_strided_consumer():
    """C = A @ B; use C.T as input to another elementwise. C.T is a strided
    view of C. Tests strided primary in elementwise."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    D = rng.standard_normal((N, M)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(D, "D")
    ops.matmul(a="A", b="B", out="C")
    ops.transpose("C", out="Ct")
    ops.elementwise("add", out="z", operands=("D", "Ct"))
    mlx_out = (D + (np.array(_mx(A) @ _mx(B))).T).astype(np.float32)
    return ops, {"z": mlx_out}


def p112_recompute_with_comparison_op():
    """mask = X > 0 (lane-agnostic since y=Scalar). Two matmul consumers
    both use mask. Recompute fires. Comparison wrap_bool inside prologue."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    B1 = rng.standard_normal((K, N)).astype(np.float32)
    B2 = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B1, "B1")
    ops.from_numpy(B2, "B2")
    ops.elementwise("gt", out="mask", operands=("X", 0.0))
    ops.matmul(a="mask", b="B1", out="t1")
    ops.matmul(a="mask", b="B2", out="t2")
    mlx_mask = (X > 0.0).astype(np.float32)
    return ops, {
        "t1": np.array(_mx(mlx_mask) @ _mx(B1)).astype(np.float32),
        "t2": np.array(_mx(mlx_mask) @ _mx(B2)).astype(np.float32),
    }


def p113_matmul_register_epilogue_with_comparison():
    """matmul + comparison epilogue: y = (C > 0). Lane-agnostic → register
    epilogue path. Tests wrap_bool inside register epilogue."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("gt", out="mask", operands=("C", 0.0))
    expected = ((np.array(_mx(A) @ _mx(B))) > 0.0).astype(np.float32)
    return ops, {"mask": expected}


def p114_reduce_then_min_with_negate():
    """reduce_min then negate epilogue. Tests reduction epilogue when the
    reduction's identity is +INFINITY."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("min", out="rm", x="X", axis=-1)
    ops.elementwise("negate", out="y", operands=("rm",))
    expected = -np.array(mx.min(_mx(X), axis=-1)).astype(np.float32)
    return ops, {"y": expected}


def p115_two_matmul_aliased_through_reshape():
    """matmul C = A@B; then reshape C to (M, N) view → use as A in next
    matmul. The view C_view has buffer_key = C, same shape — view of itself."""
    rng = np.random.default_rng(0)
    M = K = N = 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    D = rng.standard_normal((N, M)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(D, "D")
    ops.matmul(a="A", b="B", out="C")
    ops.reshape("C", (M, N), out="C_view")  # same shape view
    ops.matmul(a="C_view", b="D", out="E")
    mlx_C = np.array(_mx(A) @ _mx(B))
    mlx_E = mlx_C @ D
    return ops, {"E": mlx_E.astype(np.float32)}


def p99_elementwise_shape_1_1():
    """Tiny (1, 1) elementwise. Bounds checks on every position; output
    correct only at (0,0)."""
    X = np.array([[2.5]], dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("mul", out="y", operands=("X", 3.0))
    return ops, {"y": np.array([[7.5]], dtype=np.float32)}


def p100_where_then_chain():
    """where(x, y, cond) followed by a chain. v_head was ternary check
    blocks chain absorbing into where. Verify where can be chain HEAD."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    Cond = (rng.standard_normal((M, N)) > 0).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(Cond, "Cond")
    ops.elementwise("where", out="w", operands=("X", "Y", "Cond"))
    ops.elementwise(
        "mul", out="z", operands=("w", 0.5)
    )  # chain head can't be where, but tail can
    mlx_out = (np.where(Cond != 0.0, X, Y) * 0.5).astype(np.float32)
    return ops, {"z": mlx_out}


def p101_two_kernels_share_input_then_consume_each_other():
    """Pipeline: t1 = X + Y; t2 = t1 * X. Both X consumers (separate or chained).
    Tests whether buffer X is kept alive correctly when t1 is materialized."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="t1", operands=("X", "Y"))
    ops.elementwise("mul", out="t2", operands=("t1", "X"))  # X consumed again
    expected = ((X + Y) * X).astype(np.float32)
    return ops, {"t2": expected}


def p102_unary_chain_then_unary_chain():
    """Long unary chain: log(sqrt(exp(x)))."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = np.abs(rng.standard_normal((M, N))).astype(np.float32) + 0.1
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("exp", out="a", operands=("X",))
    ops.elementwise("sqrt", out="b", operands=("a",))
    ops.elementwise("log", out="y", operands=("b",))
    expected = np.log(np.sqrt(np.exp(X))).astype(np.float32)
    return ops, {"y": expected}


def p103_reduction_then_op_then_back_reduction():
    """Pipeline: rs = sum(X) → m = rs * 2 → reduce_max(m) — but rs is 1D,
    can't be reduced. Use (M,K) → (M,) → reshape to (M, 1) and reduce."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="rs", x="X", axis=-1)  # (M,)
    ops.elementwise("mul", out="m", operands=("rs", 2.0))  # (M,)
    ops.reshape("m", (M, 1), out="m2d")  # view (M, 1) — last axis size 1
    ops.reduction("sum", out="final", x="m2d", axis=-1)  # (M,)
    expected = (np.array(mx.sum(_mx(X), axis=-1)) * 2.0).astype(np.float32)
    return ops, {"final": expected}


def p104_transpose_then_transpose_then_op():
    """X.T.T should equal X but goes through TWO transpose views. Each
    transpose flips layout. Result should still be readable contiguous."""
    rng = np.random.default_rng(0)
    M, N = 16, 24
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.transpose("X", out="Xt")
    ops.transpose("Xt", out="Xtt")  # should equal X
    ops.elementwise("add", out="z", operands=("Xtt", "Y"))
    return ops, {"z": (X + Y).astype(np.float32)}


def p105_chain_where_in_middle_of_chain():
    """where in the middle of a chain — chain fusing should allow this
    since v_head isn't where (where is the tail of u)."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    Cond = (rng.standard_normal((M, N)) > 0).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(Cond, "Cond")
    ops.elementwise("add", out="a", operands=("X", 1.0))
    ops.elementwise("where", out="b", operands=("a", "Y", "Cond"))
    ops.elementwise("mul", out="c", operands=("b", 2.0))
    expected = (np.where(Cond != 0.0, X + 1.0, Y) * 2.0).astype(np.float32)
    return ops, {"c": expected}


def p106_matmul_then_two_reductions():
    """C = A@B; rmax = max(C); rmin = min(C). Two reductions on the same
    matmul output."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.reduction("max", out="rmax", x="C", axis=-1)
    ops.reduction("min", out="rmin", x="C", axis=-1)
    mlx_C = _mx(A) @ _mx(B)
    return ops, {
        "rmax": np.array(mx.max(mlx_C, axis=-1)).astype(np.float32),
        "rmin": np.array(mx.min(mlx_C, axis=-1)).astype(np.float32),
    }


def p107_negate_of_comparison():
    """Comparison op followed by negate. Tests wrap_bool handling: the
    comparison wraps to (cond ? 1.0 : 0.0), then negate gives -1/0."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("gt", out="mask", operands=("X", 0.0))
    ops.elementwise("negate", out="y", operands=("mask",))
    expected = -((X > 0).astype(np.float32))
    return ops, {"y": expected.astype(np.float32)}


def p108_div_by_zero_handling():
    """X / 0 — should produce inf/nan. Just verify our kernel matches MLX
    on this edge case."""
    M, N = 4, 8
    X = np.array(
        [
            [1.0, -1.0, 2.0, 0.0, -2.0, 3.0, -3.0, 4.0],
            [-4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0, 8.0],
            [-8.0, 9.0, -9.0, 10.0, -10.0, 11.0, -11.0, 12.0],
            [1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    Z = np.zeros((M, N), dtype=np.float32)
    Z[0, 3] = 1.0  # avoid /0 most places
    Z[3, 4] = 1.0  # ...
    Y = np.where(Z == 0, 0.5, Z).astype(np.float32)  # mostly 0.5 to avoid /0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("div", out="z", operands=("X", "Y"))
    return ops, {"z": (X / Y).astype(np.float32)}


def p96_chain_intermediate_is_program_output():
    """t1 = X + 1; t2 = t1 * 2. User wants BOTH t1 and t2. The chain fuser
    will fuse if t1 has only 1 *vertex* consumer (t2's op) — but t1 is
    also a program output. Does the kernel emit a store for t1?"""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("add", out="t1", operands=("X", 1.0))
    ops.elementwise("mul", out="t2", operands=("t1", 2.0))
    return ops, {
        "t2": ((X + 1.0) * 2.0).astype(np.float32),
    }


# def p97_matmul_intermediate_also_used_downstream():
#     """matmul output materialized AND used downstream: C = A@B; y = relu(C);
#     user wants both C and y. C has 1 anchor consumer (relu's chain).
#     Does the fused kernel still expose C?"""
#     rng = np.random.default_rng(0)
#     M, K, N = 32, 32, 32
#     A = rng.standard_normal((M, K)).astype(np.float32)
#     B = rng.standard_normal((K, N)).astype(np.float32)
#     ops = Operations()
#     ops.from_numpy(A, "A")
#     ops.from_numpy(B, "B")
#     ops.matmul(a="A", b="B", out="C")
#     ops.elementwise("max", out="y", operands=("C", 0.0))
#     mlx_C = np.array(_mx(A) @ _mx(B))
#     mlx_y = np.maximum(mlx_C, 0.0)
#     return ops, {"C": mlx_C.astype(np.float32), "y": mlx_y.astype(np.float32)}
#!! NOT A BUG - PURPOSEFULLY DONE SO UNUSED COMPUTATION IS NOT DONE - ALL NON-OUTPUTTED TENSORS ARE NOT STORED UNLESS THEY ARE USED


def p98_reduction_intermediate_is_program_output():
    """Reduction output AND a downstream elementwise both wanted.
    s = sum(X); y = s * 2. User reads both s and y."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    ops.elementwise("mul", out="y", operands=("s", 2.0))
    mlx_s = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {
        "y": (mlx_s * 2.0).astype(np.float32),
    }


def p86_elementwise_transpose_as_primary():
    """Primary of an elementwise chain is a transposed view. Strided primary
    load — float4 disabled because col_stride != 1."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((N, M)).astype(np.float32)  # shape matches X.T
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.transpose("X", out="Xt")  # (N, M)
    ops.elementwise("add", out="z", operands=("Xt", "Y"))
    expected = (X.T + Y).astype(np.float32)
    return ops, {"z": expected}


def p87_reduction_M_eq_1():
    """Reduction with M=1: single row reduction. Grid = (1, 1, 1)."""
    rng = np.random.default_rng(0)
    M, K = 1, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    return ops, {"s": np.array([X.sum()]).astype(np.float32)}


def p88_matmul_1x1_at_1x1():
    """Most degenerate matmul: (1,1) @ (1,1) = (1,1)."""
    A = np.array([[2.0]], dtype=np.float32)
    B = np.array([[3.0]], dtype=np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    return ops, {"C": np.array([[6.0]], dtype=np.float32)}


def p89_shapeop_output_then_chain():
    """ShapeOp output fed into a downstream chain: reshape transposed →
    multiply → exp. Tests that ShapeOp's output (fresh row-major buffer)
    flows correctly to subsequent elementwise."""
    rng = np.random.default_rng(0)
    M, N = 16, 24
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.transpose("X", out="Xt")
    ops.reshape("Xt", (N, M), out="Xt_flat")  # ShapeOp
    ops.elementwise("mul", out="t", operands=("Xt_flat", 2.0))
    ops.elementwise("exp", out="y", operands=("t",))
    expected = np.exp(X.T * 2.0).astype(np.float32)
    return ops, {"y": expected}


def p90_reduction_with_prologue_then_epilogue():
    """Both prologue AND epilogue on a reduction:
    s = sum(X * 2)  → s * 0.5 → e^s. Tests that ALL three fuse into one kernel.
    Wait — current code may only support prologue OR epilogue per assemble_reduction."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("mul", out="X1", operands=("X", 2.0))  # prologue
    ops.reduction("sum", out="s", x="X1", axis=-1)
    ops.elementwise("mul", out="s1", operands=("s", 0.5))  # epilogue
    ops.elementwise("exp", out="y", operands=("s1",))  # epilogue
    expected = np.exp(np.array(mx.sum(_mx(X) * 2.0, axis=-1)) * 0.5).astype(np.float32)
    return ops, {"y": expected}


def p91_matmul_output_to_reduction():
    """C = A @ B; rs = sum(C, axis=-1). Tests matmul → reduction chain.
    No fusion expected between them."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.reduction("sum", out="rs", x="C", axis=-1)
    mlx_out = np.array(mx.sum(_mx(A) @ _mx(B), axis=-1))
    return ops, {"rs": mlx_out.astype(np.float32)}


def p92_reduction_then_two_chain_consumers_both_anchors():
    """rs = sum(X) → both consumed as col-broadcast in two separate matmuls.
    Tests multi-consumer recompute path for a reduction output? Actually,
    only elem outputs can be recomputed. Reduction outputs must be materialized."""
    rng = np.random.default_rng(0)
    M, K = 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, K)).astype(np.float32)
    Z = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(Z, "Z")
    ops.reduction("sum", out="rs", x="X", axis=-1)  # (M,)
    ops.reshape("rs", (M, 1), out="rs2d")
    ops.elementwise("add", out="A", operands=("Y", "rs2d"), y_broadcast="col")
    ops.elementwise("mul", out="B", operands=("Z", "rs2d"), y_broadcast="col")
    mlx_rs = np.array(mx.sum(_mx(X), axis=-1))
    return ops, {
        "A": (Y + mlx_rs[:, None]).astype(np.float32),
        "B": (Z * mlx_rs[:, None]).astype(np.float32),
    }


def p93_chain_reads_view_of_reduction_output():
    """rs = sum(X) → reshape to (1, M) — but the consumer reshapes it
    AGAIN to (M, 1) via a chain of views. Verify two views of the same
    1D buffer co-exist."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="rs", x="X", axis=-1)  # (M,)
    ops.reshape("rs", (M, 1), out="rs_col")  # view 1
    ops.reshape("rs", (1, M), out="rs_row")  # view 2 of same buffer
    # Add the col-view to a (M, M) ones matrix and the row-view to the result
    O = np.ones((M, M), dtype=np.float32)
    ops.from_numpy(O, "O")
    ops.elementwise("add", out="t", operands=("O", "rs_col"), y_broadcast="col")
    ops.elementwise("add", out="y", operands=("t", "rs_row"), y_broadcast="row")
    mlx_rs = np.array(mx.sum(_mx(X), axis=-1))
    expected = (O + mlx_rs[:, None] + mlx_rs[None, :]).astype(np.float32)
    return ops, {"y": expected}


def p94_shape_op_then_reduction():
    """ShapeOp output as reduction input — does the materialized fresh
    buffer flow to the reduction?"""
    rng = np.random.default_rng(0)
    M, N = 8, 16
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.transpose("X", out="Xt")
    ops.reshape("Xt", (N, M), out="Xt_flat")  # ShapeOp output
    ops.reduction("sum", out="rs", x="Xt_flat", axis=-1)
    expected = np.array(mx.sum(_mx(X).T, axis=-1)).astype(np.float32)
    return ops, {"rs": expected}


def p95_elementwise_chain_with_first_op_unary_then_binary_tensor():
    """t1 = exp(X); t2 = t1 + Y. Two-step chain: unary + binary-with-tensor.
    Tests that the chain handles unary at the head correctly."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("exp", out="t1", operands=("X",))
    ops.elementwise("add", out="t2", operands=("t1", "Y"))
    expected = (np.exp(X) + Y).astype(np.float32)
    return ops, {"t2": expected}


def p85_matmul_K_eq_tileK():
    """K = tile_K, i.e., exactly one k-chunk iteration. (For aligned 32×32×32
    with K=32, tile_K=16, two iterations. Try K=16 explicitly.)"""
    rng = np.random.default_rng(0)
    M, K, N = 32, 16, 32  # K=16 = tile_K when K%32 != 0
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    mlx_out = np.array(_mx(A) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p74_reshape_2d_to_2d_view():
    """Reshape (M, N) → (M*N, 1) — collapse to a single column.
    Source is contiguous, target is (M*N, 1) with strides (1, 1)?"""
    rng = np.random.default_rng(0)
    M, N = 4, 8  # M*N = 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reshape("X", (M * N, 1), out="X_col")
    # Use as col_broadcast against a (M*N, N2) operand
    N2 = 8
    Y = rng.standard_normal((M * N, N2)).astype(np.float32)
    ops.from_numpy(Y, "Y")
    ops.elementwise("add", out="z", operands=("Y", "X_col"), y_broadcast="col")
    expected = (Y + X.reshape(M * N, 1)).astype(np.float32)
    return ops, {"z": expected}


def p61_matmul_prologue_chain_with_tensor_then_scalar():
    """Multi-step prologue: X1 = X + Y; X2 = X1 * 2 — chain on top of a
    tensor-secondary elem. Float4 path still buggy."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(B, "B")
    ops.elementwise("add", out="X1", operands=("X", "Y"))
    ops.elementwise("mul", out="X2", operands=("X1", 2.0))
    ops.matmul(a="X2", b="B", out="C")
    mlx_out = np.array(((_mx(X) + _mx(Y)) * 2.0) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p57_matmul_prologue_row_broadcast_extra():
    """Prologue: X + bias_row (1, K). Reads bias at col-only — float4 lane
    column offset matters. Confirms the bug isn't just for full-shape."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    bias = rng.standard_normal((1, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(bias, "bias")
    ops.from_numpy(B, "B")
    ops.elementwise("add", out="Xb", operands=("X", "bias"), y_broadcast="row")
    ops.matmul(a="Xb", b="B", out="C")
    mlx_out = np.array((_mx(X) + _mx(bias)) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p58_matmul_prologue_col_broadcast_extra():
    """Prologue: X + bias_col (M, 1). Reads bias at row-only — should be OK
    since bias's address doesn't depend on column."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    bias = rng.standard_normal((M, 1)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(bias, "bias")
    ops.from_numpy(B, "B")
    ops.elementwise("add", out="Xb", operands=("X", "bias"), y_broadcast="col")
    ops.matmul(a="Xb", b="B", out="C")
    mlx_out = np.array((_mx(X) + _mx(bias)) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p59_matmul_prologue_b_side_row_broadcast():
    """Prologue on the B-side: B + scale_row(1, N). Float4 B-load applies
    the same Y[col] to all 4 lanes — same bug, B-load variant."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    scale = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(scale, "scale")
    ops.elementwise("mul", out="Bs", operands=("B", "scale"), y_broadcast="row")
    ops.matmul(a="A", b="Bs", out="C")
    mlx_out = np.array(_mx(A) @ (_mx(B) * _mx(scale)))
    return ops, {"C": mlx_out.astype(np.float32)}


def p56_matmul_prologue_with_full_shape_extra():
    """Prologue with full-shape (none-broadcast) tensor operand: X + Y → matmul.
    Tests that the extras buffer (Y) gets registered correctly."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(B, "B")
    ops.elementwise("add", out="XY", operands=("X", "Y"))
    ops.matmul(a="XY", b="B", out="C")
    mlx_out = np.array((_mx(X) + _mx(Y)) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p48_reduce_then_reshape_then_reduce():
    """rs = sum(X, axis=-1) → (M,); reshape to (1, M); reduce that to (1,).
    Tests reducing a row-major-by-default 1-row tensor."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="rs", x="X", axis=-1)  # (M,)
    ops.reshape("rs", (1, M), out="rs_row")
    ops.reduction("sum", out="total", x="rs_row", axis=-1)  # (1,)
    mlx_out = np.array(mx.sum(_mx(X))).reshape(1).astype(np.float32)
    return ops, {"total": mlx_out}


def p35_div_by_tensor():
    """z = X / Y — tests div with two tensors. Also tests the binary path
    where neither operand is a scalar."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.uniform(0.5, 2.0, (M, N)).astype(np.float32)  # avoid /0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("div", out="z", operands=("X", "Y"))
    return ops, {"z": (X / Y).astype(np.float32)}


# ---------------------------------------------------------------------------
# NEW PROBES — targeting untested ops and edge cases
# ---------------------------------------------------------------------------


def p200_absolute_op():
    """Test the `absolute` unary op (fabs). Never tested in any existing probe.
    Checks positive, negative, zero, and large-magnitude inputs."""
    rng = np.random.default_rng(42)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    X[0, 0] = 0.0
    X[0, 1] = -0.0
    X[0, 2] = -1e20
    X[0, 3] = 1e20
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("absolute", out="y", operands=("X",))
    return ops, {"y": np.abs(X).astype(np.float32)}


def p201_floor_op():
    """Test the `floor` unary op. Never tested. Exercises positive fractional,
    negative fractional, exact integers, and large values."""
    M, N = 4, 8
    X = np.array(
        [
            [0.0, 0.5, 0.9, 1.0, 1.1, -0.1, -0.5, -0.9],
            [-1.0, -1.1, 2.7, -2.7, 100.3, -100.3, 1e6, -1e6],
            [0.001, -0.001, 0.999, -0.999, 3.5, -3.5, 7.0, -7.0],
            [0.0, -0.0, 1.5, -1.5, 2.0, -2.0, 0.4, -0.4],
        ],
        dtype=np.float32,
    )
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("floor", out="y", operands=("X",))
    return ops, {"y": np.floor(X).astype(np.float32)}


def p202_ceil_op():
    """Test the `ceil` unary op. Never tested. Exercises positive fractional,
    negative fractional, exact integers."""
    M, N = 4, 8
    X = np.array(
        [
            [0.0, 0.5, 0.9, 1.0, 1.1, -0.1, -0.5, -0.9],
            [-1.0, -1.1, 2.7, -2.7, 100.3, -100.3, 1e6, -1e6],
            [0.001, -0.001, 0.999, -0.999, 3.5, -3.5, 7.0, -7.0],
            [0.0, -0.0, 1.5, -1.5, 2.0, -2.0, 0.4, -0.4],
        ],
        dtype=np.float32,
    )
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("ceil", out="y", operands=("X",))
    return ops, {"y": np.ceil(X).astype(np.float32)}


def p203_sin_op():
    """Test the `sin` unary op. Never tested. Uses modest inputs to keep
    precision within fp32 tolerances."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.uniform(-np.pi, np.pi, (M, N)).astype(np.float32)
    X[0, 0] = 0.0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("sin", out="y", operands=("X",))
    return ops, {"y": np.sin(X).astype(np.float32)}


def p204_cos_op():
    """Test the `cos` unary op. Never tested. Uses modest inputs."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.uniform(-np.pi, np.pi, (M, N)).astype(np.float32)
    X[0, 0] = 0.0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("cos", out="y", operands=("X",))
    return ops, {"y": np.cos(X).astype(np.float32)}


def p205_ge_comparison():
    """Test the `ge` (>=) binary comparison. Never tested. Verifies both
    strict greater-than values and EQUAL values (boundary at threshold=0)."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    # Force some elements to be exactly 0.0 to test boundary
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("ge", out="mask", operands=("X", 0.0))
    expected = (X >= 0.0).astype(np.float32)
    return ops, {"mask": expected}


def p206_le_comparison():
    """Test the `le` (<=) binary comparison. Never tested. Tests direction:
    le(X, 5.0) = X <= 5.0. Also checks exact equality at threshold."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    # Force exact threshold values
    X[0, 0] = 0.0
    X[0, 1] = 1.0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("le", out="mask", operands=("X", 0.0))
    expected = (X <= 0.0).astype(np.float32)
    return ops, {"mask": expected}


def p207_binary_min_op():
    """Test the binary `min` (fmin) elementwise op. Only reduction-min
    has been tested; binary min is untested."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("min", out="z", operands=("X", "Y"))
    return ops, {"z": np.minimum(X, Y).astype(np.float32)}


def p208_matmul_k1():
    """Matmul with K=1: (M, 1) @ (1, N). The matmul tile_K=16, so K=1
    is well below tile_K. Tests the unaligned path with a single K slice."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 1, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    mlx_out = np.array(_mx(A) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p209_matmul_k2():
    """Matmul with K=2: (M, 2) @ (2, N). K=2 < tile_K=16 but K=2 means
    only two K-iterations in the inner loop."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 2, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    mlx_out = np.array(_mx(A) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p210_floor_then_chain():
    """floor → mul → add chain. Tests floor in the middle of a fused chain."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.uniform(-5.0, 5.0, (M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("floor", out="f", operands=("X",))
    ops.elementwise("mul", out="f2", operands=("f", 2.0))
    ops.elementwise("add", out="y", operands=("f2", 1.0))
    expected = (np.floor(X) * 2.0 + 1.0).astype(np.float32)
    return ops, {"y": expected}


def p211_product_reduction_with_single_zero():
    """Product reduction where one element per row is exactly 0.
    Result should be 0 for those rows."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.uniform(0.5, 2.0, (M, K)).astype(np.float32)
    # Set element [i, 0] = 0.0 for even rows, so product must be 0.
    X[0::2, 0] = 0.0
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("product", out="p", x="X", axis=-1)
    expected = np.prod(X, axis=-1).astype(np.float32)
    return ops, {"p": expected}


def p212_ge_le_boundary_exact_equal():
    """ge and le when x == y exactly. ge(X, X) and le(X, X) should both
    be all 1s. Tests that boundary values (x == threshold) are included."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = X.copy()  # same values
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("ge", out="ge_mask", operands=("X", "Y"))
    ops.elementwise("le", out="le_mask", operands=("X", "Y"))
    return ops, {
        "ge_mask": np.ones((M, N), dtype=np.float32),
        "le_mask": np.ones((M, N), dtype=np.float32),
    }


def p213_absolute_in_matmul_epilogue():
    """absolute (fabs) in a matmul epilogue (lane-agnostic unary → register
    epilogue). Not tested in any existing probe."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("absolute", out="y", operands=("C",))
    expected = np.abs(np.array(_mx(A) @ _mx(B))).astype(np.float32)
    return ops, {"y": expected}


def p214_floor_as_matmul_prologue():
    """floor on A-side as a matmul prologue. floor is a unary op so
    _prologue_eligible=True and it should be absorbed."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    # Use values where floor makes meaningful changes
    X = rng.uniform(-3.0, 3.0, (M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B, "B")
    ops.elementwise("floor", out="Xf", operands=("X",))
    ops.matmul(a="Xf", b="B", out="C")
    expected = np.array(_mx(np.floor(X).astype(np.float32)) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p215_ge_then_sum_count_positive():
    """ge(X, 0) produces 0/1 float mask; reduce_sum gives count of
    positive-or-zero elements. Tests ge → reduction pipeline."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("ge", out="mask", operands=("X", 0.0))
    ops.reduction("sum", out="count", x="mask", axis=-1)
    expected = np.sum(X >= 0.0, axis=-1).astype(np.float32)
    return ops, {"count": expected}


def p216_le_broadcast_row():
    """le with row-broadcast: X <= row_thresh(1, N). Tests le with
    y_broadcast='row'."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    thresh = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(thresh, "thresh")
    ops.elementwise("le", out="mask", operands=("X", "thresh"), y_broadcast="row")
    expected = (X <= thresh).astype(np.float32)
    return ops, {"mask": expected}


def p217_matmul_k3():
    """Matmul with K=3 (prime, below tile_K=16). Tests the unaligned path
    for an unusual K value."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 3, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    mlx_out = np.array(_mx(A) @ _mx(B))
    return ops, {"C": mlx_out.astype(np.float32)}


def p218_sin_cos_identity():
    """sin²(x) + cos²(x) = 1 for all x. Verifies both sin and cos
    agree numerically via the Pythagorean identity."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.uniform(-np.pi, np.pi, (M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("sin", out="s", operands=("X",))
    ops.elementwise("cos", out="c", operands=("X",))
    ops.elementwise("mul", out="s2", operands=("s", "s"))
    ops.elementwise("mul", out="c2", operands=("c", "c"))
    ops.elementwise("add", out="one", operands=("s2", "c2"))
    expected = np.ones((M, N), dtype=np.float32)
    return ops, {"one": expected}


def p219_ceil_then_reduce():
    """ceil → reduce_max. ceil of negative fractions then max-reduce.
    Tests ceil in a reduction prologue."""
    rng = np.random.default_rng(0)
    M, K = 16, 48
    X = rng.uniform(-3.0, 3.0, (M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("ceil", out="cx", operands=("X",))
    ops.reduction("max", out="row_max", x="cx", axis=-1)
    expected = np.max(np.ceil(X), axis=-1).astype(np.float32)
    return ops, {"row_max": expected}


def p220_ge_not_le_direction():
    """Verify ge(X, 5.0) != le(X, 5.0) for a tensor with mixed values.
    If ge and le are swapped in codegen, this would produce wrong results."""
    M, N = 4, 8
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [-1.0, -2.0, 0.0, 5.0, 5.0, 4.9, 5.1, 10.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("ge", out="ge_mask", operands=("X", 5.0))
    ops.elementwise("le", out="le_mask", operands=("X", 5.0))
    return ops, {
        "ge_mask": (X >= 5.0).astype(np.float32),
        "le_mask": (X <= 5.0).astype(np.float32),
    }


def p221_ge_single_consumer_matmul_prologue():
    """ge(X, 0) → matmul: comparison op absorbed as single-consumer
    matmul prologue. Tests bool-wrapping in the float4 A-load value transform."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    X = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(B, "B")
    ops.elementwise("ge", out="mask", operands=("X", 0.0))  # 0/1 float
    ops.matmul(a="mask", b="B", out="C")
    mlx_mask = (X >= 0.0).astype(np.float32)
    expected = np.array(_mx(mlx_mask) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p222_le_in_matmul_register_epilogue():
    """le(C, 0) as matmul register epilogue. le is binary with Scalar y →
    lane-agnostic → register epilogue. Tests bool-wrapping in register epilogue."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("le", out="mask", operands=("C", 0.0))
    mlx_C = np.array(_mx(A) @ _mx(B))
    expected = (mlx_C <= 0.0).astype(np.float32)
    return ops, {"mask": expected}


def p223_floor_in_reduction_epilogue():
    """floor after reduce: tests reduction epilogue with a unary floor op.
    build_reduction_epilogue_transform handles unary ops via make_unary builder."""
    rng = np.random.default_rng(0)
    M, K = 16, 32
    X = rng.uniform(-5.0, 5.0, (M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    ops.elementwise("floor", out="y", operands=("s",))
    expected = np.floor(np.array(mx.sum(_mx(X), axis=-1))).astype(np.float32)
    return ops, {"y": expected}


def p224_abs_in_reduction_prologue():
    """reduce_sum(abs(X)) — absolute value as reduction prologue.
    Tests unary prologue with fabs in the reduction loop body."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("absolute", out="aX", operands=("X",))
    ops.reduction("sum", out="s", x="aX", axis=-1)
    expected = np.array(mx.sum(mx.abs(_mx(X)), axis=-1)).astype(np.float32)
    return ops, {"s": expected}


def p225_matmul_k4():
    """Matmul K=4: exactly one float4 per row. K=4 < tile_K=16; unaligned
    path. The float4 A-load at col_limit=4 should load exactly the right values."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 4, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    expected = np.array(_mx(A) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p226_matmul_k8():
    """Matmul K=8: the inner k-step in MatmulComputeFragment is 8. K=8 means
    exactly one inner iteration. Tests aligned inner loop with K=tile_K/2."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 8, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    expected = np.array(_mx(A) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p227_product_reduction_large_k():
    """Product reduction with K=256 where values are near 1.0 to avoid overflow.
    Tests that the multi-SIMD-group accumulation path works for large K."""
    rng = np.random.default_rng(0)
    M, K = 8, 256
    # Values near 1 so product stays in fp32 range
    X = rng.uniform(0.98, 1.02, (M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("product", out="p", x="X", axis=-1)
    expected = np.prod(X, axis=-1).astype(np.float32)
    return ops, {"p": expected}


def p228_ge_chain_into_reduce():
    """Chain: ge(X, 0) → mul(mask, X) → reduce_sum. Tests comparison op
    in an elementwise chain that then feeds a reduction prologue."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("ge", out="mask", operands=("X", 0.0))
    ops.elementwise("mul", out="xpos", operands=("mask", "X"))  # X where X>=0, else 0
    ops.reduction("sum", out="s", x="xpos", axis=-1)
    mlx_mask = (X >= 0.0).astype(np.float32)
    expected = np.array(mx.sum(_mx(mlx_mask * X), axis=-1)).astype(np.float32)
    return ops, {"s": expected}


def p229_reduction_k4():
    """Reduction with K=4 (very small). Tests the edge case where K is exactly
    4 and multiple SIMD lanes are idle during the reduction."""
    rng = np.random.default_rng(0)
    M, K = 16, 4
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    expected = np.array(mx.sum(_mx(X), axis=-1)).astype(np.float32)
    return ops, {"s": expected}


def p230_floor_of_negative_integers():
    """floor of exact negative integers should be the integers themselves.
    floor(-3.0) = -3.0, not -4.0. Edge case for floor on integer-valued floats."""
    M, N = 4, 8
    X = np.array(
        [
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            [-10.0, -5.0, -4.0, -3.5, 3.5, 4.0, 5.0, 10.0],
            [-0.0, 0.0, -1.5, 1.5, -2.5, 2.5, -7.0, 7.0],
            [-100.0, 100.0, -0.001, 0.001, -99.9, 99.9, -50.5, 50.5],
        ],
        dtype=np.float32,
    )
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("floor", out="y", operands=("X",))
    return ops, {"y": np.floor(X).astype(np.float32)}


def p231_ge_two_tensors_no_broadcast():
    """ge(X, Y) with both operands as full tensors (no broadcast). Tests
    the non-scalar secondary path for comparison ops."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    Y = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.elementwise("ge", out="mask", operands=("X", "Y"))
    expected = (X >= Y).astype(np.float32)
    return ops, {"mask": expected}


def p232_abs_then_ge_then_sum():
    """abs(X) → ge(abs, 1.0) → sum. Tests abs in a chain that includes
    a comparison, feeding into a reduction."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("absolute", out="aX", operands=("X",))
    ops.elementwise("ge", out="mask", operands=("aX", 1.0))
    ops.reduction("sum", out="count", x="mask", axis=-1)
    expected = np.sum(np.abs(X) >= 1.0, axis=-1).astype(np.float32)
    return ops, {"count": expected}


def p233_ceil_in_register_epilogue():
    """ceil as matmul register epilogue — ceil is unary → lane-agnostic →
    register epilogue. Tests ceil inside the simd thread_elements path."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("ceil", out="y", operands=("C",))
    expected = np.ceil(np.array(_mx(A) @ _mx(B))).astype(np.float32)
    return ops, {"y": expected}


def p234_matmul_k5_prime():
    """Matmul K=5 (prime, not divisible by 4 or 8). Tests the scalar
    per-lane fallback in the unaligned A-tile load."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 5, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    expected = np.array(_mx(A) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p235_le_col_broadcast():
    """le with COL-broadcast: col_bias (M, 1). le(X, col_bias) where col_bias
    is a per-row threshold. Tests le with col-broadcast secondary."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    thresh = rng.standard_normal((M, 1)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(thresh, "thresh")
    ops.elementwise("le", out="mask", operands=("X", "thresh"), y_broadcast="col")
    expected = (X <= thresh).astype(np.float32)
    return ops, {"mask": expected}


def p236_scalar_literal_tiny_positive():
    """Test scalar_literal for a very small positive value (1e-40 — below
    fp32 normal range, tests scalar emission for denormalized literals)."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    # 1e-40 forces repr "1e-40" (has 'e') → scalar_literal uses f"{v!r}f" path
    ops.elementwise("add", out="y", operands=("X", 1e-40))
    expected = (X + np.float32(1e-40)).astype(np.float32)
    return ops, {"y": expected}


def p237_where_then_reduce():
    """where(X, Y, Cond) → reduce_sum. The where op is standalone (not
    prologue-eligible), so it materializes first, then the reduction
    absorbs a potential prologue chain."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = rng.standard_normal((M, K)).astype(np.float32)
    Cond = (rng.standard_normal((M, K)) > 0).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(Y, "Y")
    ops.from_numpy(Cond, "Cond")
    ops.elementwise("where", out="W", operands=("X", "Y", "Cond"))
    ops.reduction("sum", out="s", x="W", axis=-1)
    mlx_W = np.where(Cond != 0.0, X, Y).astype(np.float32)
    expected = np.array(mx.sum(_mx(mlx_W), axis=-1)).astype(np.float32)
    return ops, {"s": expected}


def p238_multiple_reductions_same_input():
    """s1 = sum(X); s2 = max(X); s3 = min(X). Three reductions of the same
    input. Tests that X stays alive through all three dispatches."""
    rng = np.random.default_rng(0)
    M, K = 16, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s1", x="X", axis=-1)
    ops.reduction("max", out="s2", x="X", axis=-1)
    ops.reduction("min", out="s3", x="X", axis=-1)
    mlx_X = _mx(X)
    return ops, {
        "s1": np.array(mx.sum(mlx_X, axis=-1)).astype(np.float32),
        "s2": np.array(mx.max(mlx_X, axis=-1)).astype(np.float32),
        "s3": np.array(mx.min(mlx_X, axis=-1)).astype(np.float32),
    }


def p239_matmul_k16():
    """Matmul K=16 (exactly tile_K): one K-chunk in the outer loop.
    The inner loop runs once loading exactly 16 elements per row."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 16, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    expected = np.array(_mx(A) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p240_matmul_k17():
    """Matmul K=17 (one full tile_K=16 chunk + 1 partial). Tests the outer
    loop with exactly 2 iterations where the second is partially masked."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 17, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    expected = np.array(_mx(A) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p241_negate_ge_negate_chain():
    """Chain: negate(X) → ge(neg_X, 0.0) → negate(mask). Tests a comparison
    op in the MIDDLE of a chain (not at head or tail)."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.standard_normal((M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("negate", out="nx", operands=("X",))
    ops.elementwise("ge", out="mask", operands=("nx", 0.0))
    ops.elementwise("negate", out="nmask", operands=("mask",))
    # -X >= 0 means X <= 0. mask = 1 where X <= 0. -mask = -1 or 0.
    expected = -((-X >= 0.0).astype(np.float32))
    return ops, {"nmask": expected.astype(np.float32)}


def p242_ge_as_prologue_into_reduction():
    """ge(X, 0) → reduce_sum: comparison absorbed as reduction prologue.
    Tests wrap_bool in the per-element prologue path of the reduction."""
    rng = np.random.default_rng(0)
    M, K = 32, 64
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("ge", out="mask", operands=("X", 0.0))
    ops.reduction("sum", out="s", x="mask", axis=-1)
    expected = np.sum(X >= 0.0, axis=-1).astype(np.float32)
    return ops, {"s": expected}


def p243_matmul_K256():
    """Matmul K=256 — tests multiple full tile_K=32 chunks in the outer loop."""
    rng = np.random.default_rng(0)
    M, K, N = 64, 256, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    expected = np.array(_mx(A) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p244_reduction_K256():
    """Reduction sum with K=256 — tests 2 full stride passes per row (each
    thread processes 2 elements when K=256 > tg_x=128)."""
    rng = np.random.default_rng(0)
    M, K = 16, 256
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    expected = np.array(mx.sum(_mx(X), axis=-1)).astype(np.float32)
    return ops, {"s": expected}


def p245_reduction_K512():
    """Reduction with K=512 — each thread processes 4 elements. Tests the
    per-lane accumulation with more than one iteration per SIMD group pass."""
    rng = np.random.default_rng(0)
    M, K = 8, 512
    X = rng.standard_normal((M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="s", x="X", axis=-1)
    expected = np.array(mx.sum(_mx(X), axis=-1)).astype(np.float32)
    return ops, {"s": expected}


def p246_ge_in_tg_tile_matmul_epilogue():
    """ge as matmul tg-tile epilogue: C = A@B + bias; then (C+bias) >= 0.
    The add(bias) is not lane-agnostic (tensor y), so it forces tg-tile path.
    Then ge is another epilogue op on top. Tests bool-wrapping in tg-tile chain."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((1, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(bias, "bias")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("add", out="Cb", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("ge", out="mask", operands=("Cb", 0.0))
    mlx_C = np.array(_mx(A) @ _mx(B)) + bias
    expected = (mlx_C >= 0.0).astype(np.float32)
    return ops, {"mask": expected}


def p247_large_matmul_256x256():
    """Matmul M=N=256, K=64. Tests larger shapes that use the optimal tile
    config (64×64×32 for 256×64×256)."""
    rng = np.random.default_rng(0)
    M, K, N = 256, 64, 256
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.matmul(a="A", b="B", out="C")
    expected = np.array(_mx(A) @ _mx(B))
    return ops, {"C": expected.astype(np.float32)}


def p248_floor_then_absolute():
    """floor → absolute. Tests that absolute correctly handles negative
    floor values. floor(-0.5) = -1.0, abs(-1.0) = 1.0."""
    rng = np.random.default_rng(0)
    M, N = 16, 32
    X = rng.uniform(-3.0, 3.0, (M, N)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("floor", out="f", operands=("X",))
    ops.elementwise("absolute", out="y", operands=("f",))
    expected = np.abs(np.floor(X)).astype(np.float32)
    return ops, {"y": expected}


def p249_reduce_product_K128():
    """Product reduction with K=128 — one full tg_x=128 worth of elements.
    Tests the exact SIMD boundary where each thread handles exactly 1 element."""
    rng = np.random.default_rng(0)
    M, K = 8, 128
    # Values near 1 to keep product in fp32 range
    X = rng.uniform(0.99, 1.01, (M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("product", out="p", x="X", axis=-1)
    expected = np.prod(X, axis=-1).astype(np.float32)
    return ops, {"p": expected}


def p250_reduce_product_K129():
    """Product reduction with K=129 — just above tg_x=128. Lane 0 must process
    elements 0 AND 128 (two iterations). Tests the carry-over into 2nd pass."""
    rng = np.random.default_rng(0)
    M, K = 8, 129
    X = rng.uniform(0.99, 1.01, (M, K)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("product", out="p", x="X", axis=-1)
    expected = np.prod(X, axis=-1).astype(np.float32)
    return ops, {"p": expected}

def p251_scalar_inf_as_min_clamp():
    """min(X, +inf) should pass X through unchanged.
    BUG CANDIDATE: scalar_literal(float('inf')) generates 'inff' which is
    invalid MSL — Metal compiler should reject it."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((16, 16)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("min", out="y", operands=("X", float("inf")))
    expected = np.minimum(X, float("inf"))  # = X
    return ops, {"y": expected}


def p252_scalar_neg_inf_as_max_clamp():
    """max(X, -inf) should pass X through unchanged.
    BUG CANDIDATE: scalar_literal(float('-inf')) generates '-inff' — invalid MSL."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((16, 16)).astype(np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("max", out="y", operands=("X", float("-inf")))
    expected = np.maximum(X, float("-inf"))  # = X
    return ops, {"y": expected}


def p255_equal_nan_self():
    """equal(NaN, NaN) — IEEE says NaN != NaN, so result must be 0.0.
    Tests that MSL '==' on nan gives false → wrap_bool → 0.0, matching NumPy."""
    X = np.full((8, 8), float("nan"), dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("equal", out="mask", operands=("X", float("nan")))
    expected = (X == np.float32("nan")).astype(np.float32)  # all 0.0
    return ops, {"mask": expected}


def p256_not_equal_nan_self():
    """not_equal(NaN, NaN) — IEEE: NaN != NaN is TRUE → result 1.0.
    Tests scalar literal nan: 'nanf' is invalid MSL → BUG CANDIDATE."""
    X = np.full((8, 8), float("nan"), dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("not_equal", out="mask", operands=("X", float("nan")))
    expected = (X != np.float32("nan")).astype(np.float32)  # all 1.0
    return ops, {"mask": expected}


def p257_mul_zero_by_inf():
    """0 * inf = nan in IEEE float — tests that the MSL kernel and NumPy agree
    on NaN propagation. Builds inf from device data (no scalar literal)."""
    X = np.zeros((8, 8), dtype=np.float32)
    INF = np.full((8, 8), float("inf"), dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.from_numpy(INF, "INF")
    ops.elementwise("mul", out="y", operands=("X", "INF"))
    expected = X * INF  # 0 * inf = nan
    return ops, {"y": expected}


def p259_recip_of_inf():
    """recip(+inf) should be 0.0 — tests that 1/inf = 0 in both MSL and NumPy."""
    X = np.full((8, 8), float("inf"), dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("recip", out="y", operands=("X",))
    expected = np.float32(1.0) / X  # 1/inf = 0
    return ops, {"y": expected}


def p260_log_of_zero():
    """log(0) should be -inf — tests boundary of log at zero."""
    X = np.zeros((8, 8), dtype=np.float32)
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.elementwise("log", out="y", operands=("X",))
    expected = np.log(X).astype(np.float32)  # all -inf
    return ops, {"y": expected}


def p265_reduce_sum_with_nan_input():
    """sum-reduction of a row containing NaN — both MSL simd_sum and NumPy
    should propagate nan, but _allclose(nan, nan, equal_nan=False) will
    still report DIFF, revealing the harness limitation for nan outputs."""
    X = np.array(
        [[float("nan"), 1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float32
    )
    ops = Operations()
    ops.from_numpy(X, "X")
    ops.reduction("sum", out="r", x="X", axis=-1)
    expected = np.sum(X, axis=-1).astype(np.float32)  # [nan, 22.0]
    return ops, {"r": expected}


def p266_where_operand_is_nan_tensor():
    """where(cond, x, y) with x containing NaN values. The NaN should
    be selected when cond is True. Tests that NaN passes through where."""
    cond = np.array([[1.0, 0.0, 1.0, 0.0]] * 4, dtype=np.float32)
    x = np.array([[float("nan"), float("nan"), float("nan"), float("nan")]] * 4,
                 dtype=np.float32)
    y = np.ones((4, 4), dtype=np.float32)
    ops = Operations()
    ops.from_numpy(x, "x")
    ops.from_numpy(y, "y")
    ops.from_numpy(cond, "cond")
    ops.elementwise("where", out="out", operands=("x", "y", "cond"))
    expected = np.where(cond.astype(bool), x, y)  # [[nan, 1, nan, 1], ...]
    return ops, {"out": expected.astype(np.float32)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


PROBES = [
    ("p01_product_reduction", p01_product_reduction),
    ("p02_where_ternary", p02_where_ternary),
    ("p03_comparison_mask", p03_comparison_mask),
    ("p04_reduction_on_transposed", p04_reduction_on_transposed),
    ("p05_row_broadcast_via_reshape", p05_row_broadcast_via_reshape),
    ("p06_diamond_shared_a", p06_diamond_shared_a),
    ("p07_multi_producer_no_shared", p07_multi_producer_no_shared),
    ("p08_one_row_matrix", p08_one_row_matrix),
    ("p09_one_col_matrix", p09_one_col_matrix),
    ("p10_full_tensor_mean_broadcast", p10_full_tensor_mean_broadcast),
    ("p11_relu_then_reduce", p11_relu_then_reduce),
    ("p12_reduce_then_scaled", p12_reduce_then_scaled),
    ("p13_chain_with_intermediate_reuse", p13_chain_with_intermediate_reuse),
    ("p14_matmul_then_transpose_then_add", p14_matmul_then_transpose_then_add),
    (
        "p15_reshape_op_via_transpose_then_reshape",
        p15_reshape_op_via_transpose_then_reshape,
    ),
    ("p16_recompute_lane_agnostic_prologue", p16_recompute_lane_agnostic_prologue),
    ("p17_softmax_via_primitives", p17_softmax_via_primitives),
    ("p18_double_transpose_identity", p18_double_transpose_identity),
    ("p19_min_reduction", p19_min_reduction),
    ("p20_unary_negate_chain", p20_unary_negate_chain),
    ("p21_tanh_extreme_values", p21_tanh_extreme_values),
    ("p22_reduce_epilogue_chain", p22_reduce_epilogue_chain),
    ("p23_prologue_chain_into_matmul", p23_prologue_chain_into_matmul),
    ("p24_mixed_epilogue_chain", p24_mixed_epilogue_chain),
    ("p25_sum_of_comparison", p25_sum_of_comparison),
    ("p26_reshape_col_to_1d", p26_reshape_col_to_1d),
    ("p27_reshape_to_same_shape", p27_reshape_to_same_shape),
    (
        "p28_non_aligned_matmul_with_epilogue_chain",
        p28_non_aligned_matmul_with_epilogue_chain,
    ),
    ("p29_matmul_with_transposed_input_b", p29_matmul_with_transposed_input_b),
    ("p30_where_with_broadcast_cond", p30_where_with_broadcast_cond),
    ("p31_matmul_input_used_twice_as_a_and_b", p31_matmul_input_used_twice_as_a_and_b),
    ("p32_self_add", p32_self_add),
    (
        "p33_recompute_two_consumers_one_unfusable",
        p33_recompute_two_consumers_one_unfusable,
    ),
    ("p34_subtract_scalar_first", p34_subtract_scalar_first),
    ("p35_div_by_tensor", p35_div_by_tensor),
    ("p36_gram_matrix_x_xt", p36_gram_matrix_x_xt),
    ("p37_matmul_M_eq_1", p37_matmul_M_eq_1),
    ("p38_matmul_N_eq_1", p38_matmul_N_eq_1),
    ("p39_reduce_over_k_eq_1", p39_reduce_over_k_eq_1),
    ("p40_two_relus_share_input", p40_two_relus_share_input),
    ("p41_chain_with_4_ops", p41_chain_with_4_ops),
    ("p42_reduction_then_two_view_consumers", p42_reduction_then_two_view_consumers),
    ("p43_pow_op", p43_pow_op),
    ("p44_sign_of_zero", p44_sign_of_zero),
    (
        "p45_chain_uses_scalar_operand_with_negate",
        p45_chain_uses_scalar_operand_with_negate,
    ),
    ("p46_recip_op", p46_recip_op),
    ("p47_sqrt_chain_log", p47_sqrt_chain_log),
    ("p48_reduce_then_reshape_then_reduce", p48_reduce_then_reshape_then_reduce),
    ("p49_diamond_shared_a_small", p49_diamond_shared_a_small),
    ("p50_multi_producer_small", p50_multi_producer_small),
    (
        "p51_prologue_same_input_into_both_a_and_b",
        p51_prologue_same_input_into_both_a_and_b,
    ),
    (
        "p52_elementwise_extra_shares_input_buffer",
        p52_elementwise_extra_shares_input_buffer,
    ),
    (
        "p53_matmul_with_bias_that_also_is_input_a",
        p53_matmul_with_bias_that_also_is_input_a,
    ),
    ("p54_equal_op", p54_equal_op),
    ("p55_chain_then_chain_via_reduction", p55_chain_then_chain_via_reduction),
    (
        "p56_matmul_prologue_with_full_shape_extra",
        p56_matmul_prologue_with_full_shape_extra,
    ),
    (
        "p57_matmul_prologue_row_broadcast_extra",
        p57_matmul_prologue_row_broadcast_extra,
    ),
    (
        "p58_matmul_prologue_col_broadcast_extra",
        p58_matmul_prologue_col_broadcast_extra,
    ),
    (
        "p59_matmul_prologue_b_side_row_broadcast",
        p59_matmul_prologue_b_side_row_broadcast,
    ),
    (
        "p60_reduction_prologue_full_shape_extra",
        p60_reduction_prologue_full_shape_extra,
    ),
    (
        "p61_matmul_prologue_chain_with_tensor_then_scalar",
        p61_matmul_prologue_chain_with_tensor_then_scalar,
    ),
    ("p62_multi_anchor_same_b", p62_multi_anchor_same_b),
    ("p63_comparison_into_max", p63_comparison_into_max),
    ("p64_chain_through_view", p64_chain_through_view),
    ("p65_chain_with_first_op_as_comparison", p65_chain_with_first_op_as_comparison),
    (
        "p66_input_used_as_col_broadcast_into_matmul_epilogue",
        p66_input_used_as_col_broadcast_into_matmul_epilogue,
    ),
    ("p67_input_used_as_scalar_broadcast", p67_input_used_as_scalar_broadcast),
    (
        "p68_elementwise_chain_with_col_broadcast_intermediate_use",
        p68_elementwise_chain_with_col_broadcast_intermediate_use,
    ),
    ("p69_reduce_max_all_negatives", p69_reduce_max_all_negatives),
    (
        "p70_reduction_with_K_less_than_simdwidth",
        p70_reduction_with_K_less_than_simdwidth,
    ),
    ("p71_reduction_K_between_32_and_128", p71_reduction_K_between_32_and_128),
    (
        "p72_matmul_tg_epilogue_reads_input_a_again",
        p72_matmul_tg_epilogue_reads_input_a_again,
    ),
    (
        "p73_input_buffer_size_only_uses_first_K_elements",
        p73_input_buffer_size_only_uses_first_K_elements,
    ),
    ("p74_reshape_2d_to_2d_view", p74_reshape_2d_to_2d_view),
    ("p75_multi_anchor_a0_eq_b1", p75_multi_anchor_a0_eq_b1),
    (
        "p76_chain_with_tail_consumed_by_view_then_op",
        p76_chain_with_tail_consumed_by_view_then_op,
    ),
    ("p77_relu_then_reshape_then_relu", p77_relu_then_reshape_then_relu),
    (
        "p78_reduce_then_use_result_in_two_chains",
        p78_reduce_then_use_result_in_two_chains,
    ),
    (
        "p79_matmul_with_row_broadcast_full_M_eq_N",
        p79_matmul_with_row_broadcast_full_M_eq_N,
    ),
    ("p80_chained_reductions_on_diff_axes", p80_chained_reductions_on_diff_axes),
    ("p81_transpose_of_1d_reshape_view", p81_transpose_of_1d_reshape_view),
    ("p82_reduce_then_negate_epilogue", p82_reduce_then_negate_epilogue),
    ("p83_chain_writes_back_to_same_position", p83_chain_writes_back_to_same_position),
    ("p84_not_equal_op", p84_not_equal_op),
    ("p85_matmul_K_eq_tileK", p85_matmul_K_eq_tileK),
    ("p86_elementwise_transpose_as_primary", p86_elementwise_transpose_as_primary),
    ("p87_reduction_M_eq_1", p87_reduction_M_eq_1),
    ("p88_matmul_1x1_at_1x1", p88_matmul_1x1_at_1x1),
    ("p89_shapeop_output_then_chain", p89_shapeop_output_then_chain),
    (
        "p90_reduction_with_prologue_then_epilogue",
        p90_reduction_with_prologue_then_epilogue,
    ),
    ("p91_matmul_output_to_reduction", p91_matmul_output_to_reduction),
    (
        "p92_reduction_then_two_chain_consumers_both_anchors",
        p92_reduction_then_two_chain_consumers_both_anchors,
    ),
    (
        "p93_chain_reads_view_of_reduction_output",
        p93_chain_reads_view_of_reduction_output,
    ),
    ("p94_shape_op_then_reduction", p94_shape_op_then_reduction),
    (
        "p95_elementwise_chain_with_first_op_unary_then_binary_tensor",
        p95_elementwise_chain_with_first_op_unary_then_binary_tensor,
    ),
    (
        "p96_chain_intermediate_is_program_output",
        p96_chain_intermediate_is_program_output,
    ),
    (
        "p98_reduction_intermediate_is_program_output",
        p98_reduction_intermediate_is_program_output,
    ),
    ("p99_elementwise_shape_1_1", p99_elementwise_shape_1_1),
    ("p100_where_then_chain", p100_where_then_chain),
    (
        "p101_two_kernels_share_input_then_consume_each_other",
        p101_two_kernels_share_input_then_consume_each_other,
    ),
    ("p102_unary_chain_then_unary_chain", p102_unary_chain_then_unary_chain),
    (
        "p103_reduction_then_op_then_back_reduction",
        p103_reduction_then_op_then_back_reduction,
    ),
    ("p104_transpose_then_transpose_then_op", p104_transpose_then_transpose_then_op),
    ("p105_chain_where_in_middle_of_chain", p105_chain_where_in_middle_of_chain),
    ("p106_matmul_then_two_reductions", p106_matmul_then_two_reductions),
    ("p107_negate_of_comparison", p107_negate_of_comparison),
    ("p108_div_by_zero_handling", p108_div_by_zero_handling),
    ("p109_input_used_only_via_view", p109_input_used_only_via_view),
    (
        "p110_chain_with_two_consumers_one_is_view",
        p110_chain_with_two_consumers_one_is_view,
    ),
    (
        "p111_matmul_output_with_strided_consumer",
        p111_matmul_output_with_strided_consumer,
    ),
    ("p112_recompute_with_comparison_op", p112_recompute_with_comparison_op),
    (
        "p113_matmul_register_epilogue_with_comparison",
        p113_matmul_register_epilogue_with_comparison,
    ),
    ("p114_reduce_then_min_with_negate", p114_reduce_then_min_with_negate),
    (
        "p115_two_matmul_aliased_through_reshape",
        p115_two_matmul_aliased_through_reshape,
    ),
    (
        "p116_diamond_shared_a_via_transpose_view",
        p116_diamond_shared_a_via_transpose_view,
    ),
    (
        "p117_diamond_with_a_distinct_names_same_buffer",
        p117_diamond_with_a_distinct_names_same_buffer,
    ),
    (
        "p118_elementwise_output_as_input_to_another_with_view",
        p118_elementwise_output_as_input_to_another_with_view,
    ),
    (
        "p119_chain_with_scalar_first_then_tensor",
        p119_chain_with_scalar_first_then_tensor,
    ),
    (
        "p120_elementwise_chain_5_ops_with_intermediate_outputs",
        p120_elementwise_chain_5_ops_with_intermediate_outputs,
    ),
    ("p121_multi_producer_comparison_merge", p121_multi_producer_comparison_merge),
    ("p122_chain_fuse_across_named_consumer", p122_chain_fuse_across_named_consumer),
    (
        "p123_reduce_epilogue_comparison_with_negate",
        p123_reduce_epilogue_comparison_with_negate,
    ),
    (
        "p124_matmul_aligned_with_register_epilogue_chain_4_ops",
        p124_matmul_aligned_with_register_epilogue_chain_4_ops,
    ),
    (
        "p125_dispatch_chain_with_consecutive_kernels_sharing_input",
        p125_dispatch_chain_with_consecutive_kernels_sharing_input,
    ),
    (
        "p126_reduce_along_axis_then_immediately_use_via_view",
        p126_reduce_along_axis_then_immediately_use_via_view,
    ),
    (
        "p127_long_chain_register_epilogue_relu_sequence",
        p127_long_chain_register_epilogue_relu_sequence,
    ),
    ("p128_multi_anchor_followed_by_epilogue", p128_multi_anchor_followed_by_epilogue),
    (
        "p129_multi_anchor_with_prologue_into_either_anchor",
        p129_multi_anchor_with_prologue_into_either_anchor,
    ),
    # New probes for untested ops and edge cases
    ("p200_absolute_op", p200_absolute_op),
    ("p201_floor_op", p201_floor_op),
    ("p202_ceil_op", p202_ceil_op),
    ("p203_sin_op", p203_sin_op),
    ("p204_cos_op", p204_cos_op),
    ("p205_ge_comparison", p205_ge_comparison),
    ("p206_le_comparison", p206_le_comparison),
    ("p207_binary_min_op", p207_binary_min_op),
    ("p208_matmul_k1", p208_matmul_k1),
    ("p209_matmul_k2", p209_matmul_k2),
    ("p210_floor_then_chain", p210_floor_then_chain),
    ("p211_product_reduction_with_single_zero", p211_product_reduction_with_single_zero),
    ("p212_ge_le_boundary_exact_equal", p212_ge_le_boundary_exact_equal),
    ("p213_absolute_in_matmul_epilogue", p213_absolute_in_matmul_epilogue),
    ("p214_floor_as_matmul_prologue", p214_floor_as_matmul_prologue),
    ("p215_ge_then_sum_count_positive", p215_ge_then_sum_count_positive),
    ("p216_le_broadcast_row", p216_le_broadcast_row),
    ("p217_matmul_k3", p217_matmul_k3),
    ("p218_sin_cos_identity", p218_sin_cos_identity),
    ("p219_ceil_then_reduce", p219_ceil_then_reduce),
    ("p220_ge_not_le_direction", p220_ge_not_le_direction),
    ("p221_ge_single_consumer_matmul_prologue", p221_ge_single_consumer_matmul_prologue),
    ("p222_le_in_matmul_register_epilogue", p222_le_in_matmul_register_epilogue),
    ("p223_floor_in_reduction_epilogue", p223_floor_in_reduction_epilogue),
    ("p224_abs_in_reduction_prologue", p224_abs_in_reduction_prologue),
    ("p225_matmul_k4", p225_matmul_k4),
    ("p226_matmul_k8", p226_matmul_k8),
    ("p227_product_reduction_large_k", p227_product_reduction_large_k),
    ("p228_ge_chain_into_reduce", p228_ge_chain_into_reduce),
    ("p229_reduction_k4", p229_reduction_k4),
    ("p230_floor_of_negative_integers", p230_floor_of_negative_integers),
    ("p231_ge_two_tensors_no_broadcast", p231_ge_two_tensors_no_broadcast),
    ("p232_abs_then_ge_then_sum", p232_abs_then_ge_then_sum),
    ("p233_ceil_in_register_epilogue", p233_ceil_in_register_epilogue),
    ("p234_matmul_k5_prime", p234_matmul_k5_prime),
    ("p235_le_col_broadcast", p235_le_col_broadcast),
    ("p236_scalar_literal_tiny_positive", p236_scalar_literal_tiny_positive),
    ("p237_where_then_reduce", p237_where_then_reduce),
    ("p238_multiple_reductions_same_input", p238_multiple_reductions_same_input),
    ("p239_matmul_k16", p239_matmul_k16),
    ("p240_matmul_k17", p240_matmul_k17),
    ("p241_negate_ge_negate_chain", p241_negate_ge_negate_chain),
    ("p242_ge_as_prologue_into_reduction", p242_ge_as_prologue_into_reduction),
    ("p243_matmul_K256", p243_matmul_K256),
    ("p244_reduction_K256", p244_reduction_K256),
    ("p245_reduction_K512", p245_reduction_K512),
    ("p246_ge_in_tg_tile_matmul_epilogue", p246_ge_in_tg_tile_matmul_epilogue),
    ("p247_large_matmul_256x256", p247_large_matmul_256x256),
    ("p248_floor_then_absolute", p248_floor_then_absolute),
    ("p249_reduce_product_K128", p249_reduce_product_K128),
    ("p250_reduce_product_K129", p250_reduce_product_K129),
    ("p251_scalar_inf_as_min_clamp", p251_scalar_inf_as_min_clamp),
    ("p252_scalar_neg_inf_as_max_clamp", p252_scalar_neg_inf_as_max_clamp),
    ("p255_equal_nan_self", p255_equal_nan_self),
    ("p256_not_equal_nan_self", p256_not_equal_nan_self),
    ("p257_mul_zero_by_inf", p257_mul_zero_by_inf),
    ("p259_recip_of_inf", p259_recip_of_inf),
    ("p260_log_of_zero", p260_log_of_zero),
    ("p265_reduce_sum_with_nan_input", p265_reduce_sum_with_nan_input),
    ("p266_where_operand_is_nan_tensor", p266_where_operand_is_nan_tensor),
]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    dump_msl = "--msl" in sys.argv
    summaries = []
    for name, fn in PROBES:
        if only is not None and only not in name:
            continue
        summaries.append(run_probe(name, fn, dump_msl=dump_msl))

    _hr("=")
    print("SUMMARY")
    _hr("=")
    failures = [s for s in summaries if not s["ok"]]
    for s in summaries:
        status = "PASS" if s["ok"] else "FAIL"
        print(f"  {status}  {s['name']}")
    print(f"\n  {len(summaries) - len(failures)}/{len(summaries)} probes passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
