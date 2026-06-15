"""Playground for the Operations API. Edit `program()` below to write
your own ops; the harness compiles it, runs it on GPU, and shows you
what came out.

Run:
    uv run demo.py                # show inputs, fusion, named outputs
    uv run demo.py --msl          # also print the generated MSL
    uv run demo.py --ir           # also print the IR Program
    uv run demo.py --all-tensors  # print every tensor in the env, not just outputs
    uv run demo.py --full         # full numpy printing (no truncation)

Declaring tensors:

    ops.input(name, shape, layout=ROW_MAJOR,        # caller-supplied, no payload
              row_stride=None, col_stride=None)     # (override strides only when
                                                    # binding a non-contiguous buffer)
    ops.from_numpy(np_array, name)                  # declare + upload numpy payload
                                                    # (float32 only)

Compute primitives:

    ops.matmul(a=, b=, out=)                        # C[M,N] = A[M,K] @ B[K,N]
    ops.elementwise(op, out=, operands=,            # pointwise; operands are
                    y_broadcast="none",             # tensor names or scalar
                    cond_broadcast="none")          # literals (for `where`)
    ops.reduction(op, out=, x=, axis=-1)            # last-axis reduction over 2D

Metadata views (no IR op emitted when possible):

    ops.transpose(name, out=)                       # 2D transpose view
    ops.reshape(name, new_shape, out=)              # view if contiguous, else
                                                    # ShapeOp copy kernel

Introspection:

    ops.build()                                     # snapshot the IR Program
    ops.tensors                                     # name -> Tensor registry
    ops.uploads                                     # name -> numpy payload

Supported elementwise ops:

    unary    : negate, absolute, exp, log, sqrt, recip, sin, cos, tanh,
               floor, ceil, sign
    binary   : add, subtract, mul, div, max, min, pow,
               equal, not_equal, lt, gt, ge, le
    ternary  : where(x, y, cond)                    # cond is the THIRD operand

Supported reduction ops: sum, max, min, product

Broadcast modes for the secondary operand: "none", "scalar", "row", "col".
The same modes apply to `cond_broadcast` for `where`.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from api import run as api_run
from compute.elementwise.elementwise import supported_elementwise_ops
from compute.reduction.reduction import REDUCTION_OPS
from orchestrator import Operations
from orchestrator.ir import ElementwiseOp, MatmulOp, ReductionOp, Scalar, ShapeOp

#! Peek at sandbox.py for examples - those are AI generated tests so will have comprehensive examples


# =============================================================================
#! EDIT BELOW: write your own program.
#
# Return (ops, output_names) — output_names is the list of tensors whose
# values you want printed after the run. Anything you don't name still
# lives in the env (use --all-tensors to see it).
# =============================================================================


def program() -> tuple[Operations, list[str]]:
    rng = np.random.default_rng(0)
    M, K, N = 64, 64, 64

    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((1, N)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(A, "A")
    ops.from_numpy(B, "B")
    ops.from_numpy(bias, "bias")

    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("add", out="Cb", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("max", out="y", operands=("Cb", 0.0))  # relu

    return ops, ["y"]


# =============================================================================
# !Harness — you usually don't need to edit anything below here.
# =============================================================================


def _hr(c: str = "─", w: int = 78) -> None:
    print(c * w)


def _describe_op(op) -> str:
    if isinstance(op, MatmulOp):
        return f"{op.out.name} = matmul({op.a.name}, {op.b.name})"
    if isinstance(op, ElementwiseOp):
        operands = ", ".join(
            (repr(o.value) if isinstance(o, Scalar) else o.name) for o in op.operands
        )
        tail = []
        if op.y_broadcast.value != "none":
            tail.append(f"y={op.y_broadcast.value}")
        if op.cond_broadcast.value != "none":
            tail.append(f"cond={op.cond_broadcast.value}")
        suffix = f" [{', '.join(tail)}]" if tail else ""
        return f"{op.out.name} = {op.op}({operands}){suffix}"
    if isinstance(op, ReductionOp):
        return f"{op.out.name} = {op.op}({op.input.name}, axis={op.axis})"
    if isinstance(op, ShapeOp):
        return f"{op.out.name} = reshape({op.input.name}) -> {op.out.shape}"
    return f"{op.out.name} = {type(op).__name__}"


def _print_tensor(name: str, value, full: bool) -> None:
    arr = np.asarray(value)
    stats = (
        f"min={arr.min():.4g} max={arr.max():.4g} "
        f"mean={arr.mean():.4g} std={arr.std():.4g}"
        if arr.size
        else "empty"
    )
    print(f"  {name}  shape={tuple(arr.shape)}  dtype={arr.dtype}  {stats}")
    if full:
        with np.printoptions(threshold=np.inf, linewidth=120):
            print(arr)
    else:
        with np.printoptions(precision=4, suppress=True, linewidth=120, edgeitems=2):
            print(np.array2string(arr, prefix="    "))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Operations API playground — edit program() and run."
    )
    parser.add_argument("--msl", action="store_true", help="print generated MSL")
    parser.add_argument("--ir", action="store_true", help="print the IR Program")
    parser.add_argument(
        "--all-tensors",
        action="store_true",
        help="print every tensor in the env, not just declared outputs",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="print full tensor values without numpy truncation",
    )
    args = parser.parse_args()

    print("Supported elementwise ops:", ", ".join(supported_elementwise_ops()))
    print("Supported reduction ops: ", ", ".join(REDUCTION_OPS))
    _hr()

    ops, output_names = program()
    uploads = ops.uploads

    print("Inputs:")
    if not uploads:
        print("  (none — nothing uploaded via ops.from_numpy)")
    for name, arr in uploads.items():
        _print_tensor(name, arr, full=args.full)
    _hr()

    if args.ir:
        print("IR Program:")
        for i, op in enumerate(ops.build()):
            print(f"  {i:>2d}. {_describe_op(op)}")
        _hr()

    result = api_run(ops)

    print(f"Fused into {len(result.groups)} kernel(s):")
    for i, g in enumerate(result.groups):
        absorbed = ", ".join(_describe_op(o) for o in g.ops)
        print(f"  [{i}] {g.strategy.value}  ({absorbed})")
    _hr()

    if args.msl:
        for i, g in enumerate(result.groups):
            print(f"--- kernel {i}: {g.kernel.function_name} ({g.strategy.value}) ---")
            print(g.kernel.source)
            print()
        _hr()

    env = result.env
    if args.all_tensors:
        # Skip the input tensors we already printed and any profiler keys.
        names = [
            n
            for n in env
            if n not in uploads and not n.startswith("t_")
        ]
        print(f"All non-input tensors in env ({len(names)}):")
    else:
        names = list(output_names)
        print(f"Outputs ({len(names)}):")
    if not names:
        print("  (none)")
    for name in names:
        if name not in env:
            print(f"  {name}: NOT IN ENV — not materialized (likely absorbed by fusion)")
            continue
        _print_tensor(name, env[name], full=args.full)

    return 0


if __name__ == "__main__":
    sys.exit(main())
