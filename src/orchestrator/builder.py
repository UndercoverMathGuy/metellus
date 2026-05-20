"""Ergonomic builder for the primitive IR.

Callers reference tensors by string name; the builder tracks shape /
dtype / layout / strides in a name-keyed registry and emits primitive
Ops into a `Program`. Shape validity is the IR's job — each Op runs
shape / broadcast / dtype checks in its own `__post_init__`, so the
builder is a thin name-resolution layer that lets the Op dataclasses
do the talking.

    ops = Operations()
    ops.input("a", shape=(M, K))
    ops.input("b", shape=(K, N))
    ops.input("bias", shape=(1, N))
    ops.matmul(a="a", b="b", out="c")
    ops.elementwise("add", out="d",
                    operands=("c", "bias"),
                    y_broadcast=BroadcastSpec.ROW)
    ops.elementwise("max", out="relu_out",
                    operands=("d", 0.0))                # scalar literal
    ops.reduction("sum", out="r", x="relu_out", axis=-1)
    program = ops.build()

All inputs must be declared via `ops.input(...)`; shape is metadata on
the `Tensor`, never restated on the ops themselves.

Scope rule (per orchestrator-scope memory): only the three primitives
are exposed. Surface ops (relu, softmax, ...) decompose into
elementwise / reduction / matmul calls in a layer above this builder.
"""

from __future__ import annotations

import numpy as np

from compute.elementwise.elementwise import BroadcastSpec

from orchestrator.ir import (
    ElementwiseOp,
    Layout,
    MatmulOp,
    OperandValue,
    Program,
    ReductionOp,
    Scalar,
    ShapeOp,
    Tensor,
)


class Operations:
    """Mutable builder. One instance per program. `build()` returns a
    fresh list snapshot; the builder may still be appended to afterwards."""

    def __init__(self, dtype: str = "float32") -> None:
        self._dtype = dtype
        self._tensors: dict[str, Tensor] = {}
        self._ops: Program = []
        self._uploads: dict[str, np.ndarray] = {}

    # -- tensor registry ---------------------------------------------------

    def input(
        self,
        name: str,
        shape: tuple[int, ...],
        layout: Layout = Layout.ROW_MAJOR,
        row_stride: int | None = None,
        col_stride: int | None = None,
    ) -> Tensor:
        """Declare a caller-supplied tensor (no producing op). Strides
        default to the contiguous layout; override only when binding a
        buffer whose strides aren't contiguous."""
        return self._declare(
            name,
            shape,
            layout=layout,
            row_stride=row_stride,
            col_stride=col_stride,
        )

    def from_numpy(self, arr: np.ndarray, name: str) -> Tensor:
        """Declare an input tensor and stash its numpy payload for later
        upload by the scheduler. Equivalent to `ops.input(name, arr.shape)`
        plus recording `arr` against `name` in the uploads registry.

        Only storage owners can be uploaded — calling `from_numpy` on a
        view (a name produced by `transpose` / `reshape`) is rejected. The
        array must be `float32` and match the declared shape; non-float32
        inputs are rejected so silent float64 widening can't mask a bug."""
        if arr.dtype != np.float32:
            raise ValueError(
                f"from_numpy: array dtype must be float32; got {arr.dtype}"
            )
        t = self._declare(name, tuple(arr.shape))
        if t.buffer_key != name:
            raise ValueError(
                f"from_numpy: {name!r} is a view of {t.buffer_key!r}; only "
                "storage owners can be uploaded"
            )
        if tuple(arr.shape) != t.shape:
            raise ValueError(
                f"from_numpy: array shape {tuple(arr.shape)} does not match "
                f"declared tensor shape {t.shape}"
            )
        required = _required_storage_elements(t)
        if arr.size < required:
            raise ValueError(
                f"from_numpy: array with shape {tuple(arr.shape)} provides "
                f"{arr.size} elements, but tensor {name!r} with strides "
                f"(row_stride={t.row_stride}, col_stride={t.col_stride}) "
                f"can address {required} elements"
            )
        self._uploads[name] = arr
        return t

    def _declare(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        layout: Layout = Layout.ROW_MAJOR,
        row_stride: int | None = None,
        col_stride: int | None = None,
    ) -> Tensor:
        existing = self._tensors.get(name)
        if existing is not None:
            if existing.shape != tuple(shape):
                raise ValueError(
                    f"tensor {name!r} already declared with shape "
                    f"{existing.shape}, cannot redeclare as {tuple(shape)}"
                )
            return existing
        t = Tensor(
            name=name,
            shape=tuple(shape),
            dtype=self._dtype,
            layout=layout,
            row_stride=-1 if row_stride is None else row_stride,
            col_stride=-1 if col_stride is None else col_stride,
        )
        self._tensors[name] = t
        return t

    def _bind_existing(self, name: str) -> Tensor:
        t = self._tensors.get(name)
        if t is None:
            raise ValueError(
                f"tensor {name!r} is not declared — call ops.input(...) first "
                "or use an op that produces it"
            )
        return t

    def _reserve_output(self, name: str, shape: tuple[int, ...]) -> Tensor:
        """Build (but don't register) the output Tensor. SSA freshness is
        checked now; registration is deferred until the producing Op has
        constructed successfully so a failed Op-validation doesn't leave
        a stray name in the registry."""
        if name in self._tensors:
            raise ValueError(
                f"output tensor {name!r} already defined; IR is SSA — pick a fresh name"
            )
        return Tensor(name=name, shape=tuple(shape), dtype=self._dtype)

    def _commit_output(self, t: Tensor) -> None:
        self._tensors[t.name] = t

    # -- ops ---------------------------------------------------------------

    def matmul(self, a: str, b: str, out: str) -> Tensor:
        """C[M,N] = A[M,K] @ B[K,N]. Both inputs must be declared via
        `ops.input(...)` first; the output shape is taken from their
        shapes and the `MatmulOp` constructor validates the contract."""
        a_t = self._bind_existing(a)
        b_t = self._bind_existing(b)
        # Shape is inferred only so SSA can hold a fresh output Tensor;
        # MatmulOp.__post_init__ is the actual contract check.
        out_t = self._reserve_output(out, (a_t.shape[0], b_t.shape[1]))
        op = MatmulOp(out=out_t, a=a_t, b=b_t)
        self._commit_output(out_t)
        self._ops.append(op)
        return out_t

    def elementwise(
        self,
        op: str,
        out: str,
        operands: tuple[str | int | float, ...],
        y_broadcast: BroadcastSpec | str = BroadcastSpec.NONE,
        cond_broadcast: BroadcastSpec | str = BroadcastSpec.NONE,
    ) -> Tensor:
        """Pointwise op. Output shape = operands[0]'s shape.

        Each operand is either a registered tensor name (str) or a Python
        numeric literal (int/float) wrapped as a `Scalar` constant — used
        for things like relu = `elementwise("max", out=..., operands=("x", 0.0))`.

        Operand[0] must be a tensor; non-primary tensor operands must be
        pre-declared (auto-declaring them would mask bugs since the call
        site doesn't pin their shape). Shape, arity, and broadcast
        compatibility are all enforced by `ElementwiseOp.__post_init__`.
        """
        y_bc = BroadcastSpec(y_broadcast)  # raises ValueError on typos
        cond_bc = BroadcastSpec(cond_broadcast)

        resolved: list[OperandValue] = []
        for o in operands:
            if isinstance(o, str):
                resolved.append(self._bind_existing(o))
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                resolved.append(Scalar(value=float(o)))
            else:
                raise ValueError(
                    f"operand {o!r} must be a tensor name (str) or numeric "
                    "literal (int/float)"
                )

        primary = resolved[0]
        if not isinstance(primary, Tensor):
            # Surface the friendlier message here; the IR would also catch it.
            raise ValueError(
                f"elementwise {op!r}: operand[0] must be a tensor name; "
                f"got {operands[0]!r}. Scalars are only allowed as "
                "secondary operands."
            )
        out_t = self._reserve_output(out, primary.shape)
        ew = ElementwiseOp(
            out=out_t,
            op=op,
            operands=tuple(resolved),
            y_broadcast=y_bc,
            cond_broadcast=cond_bc,
        )
        self._commit_output(out_t)
        self._ops.append(ew)
        return out_t

    def transpose(self, name: str, out: str) -> Tensor:
        """Logical 2D transpose. No IR op is emitted — the result is a
        new SSA value `out` whose `buffer_key` aliases `name`'s
        storage, with shape, strides, and layout swapped. Downstream
        ops that consume `out` will see the new metadata; the matmul
        template honours arbitrary strides so a transposed input
        really is read transposed."""
        src = self._bind_existing(name)
        if out in self._tensors:
            raise ValueError(f"view tensor {out!r} already declared; pick a fresh name")
        view = src.transpose(out)
        self._tensors[out] = view
        return view

    def reshape(
        self,
        name: str,
        new_shape: tuple[int, ...],
        out: str,
    ) -> Tensor:
        """Reshape — pure metadata swap when possible (no IR op
        emitted), `ShapeOp` copy otherwise. The metadata-swap path
        applies whenever the source is row-major contiguous and the
        new shape has the same element count; the resulting tensor
        aliases the source buffer. Non-contiguous sources (e.g. after
        a transpose) or shape changes that would re-order the logical
        element stream fall back to a `ShapeOp` that reads the source
        via its strides and writes a fresh contiguous buffer named
        `out`."""
        src = self._bind_existing(name)
        if out in self._tensors:
            raise ValueError(
                f"reshape output {out!r} already declared; pick a fresh name"
            )
        new_shape_t = tuple(new_shape)
        if src.can_reshape_as_view(new_shape_t):
            view = src.reshape_view(new_shape_t, out)
            self._tensors[out] = view
            return view
        # Validate element count + rank up front so we get a clear error
        # before ShapeOp.__post_init__.
        if len(new_shape_t) == 0 or len(new_shape_t) > 2:
            raise ValueError(
                f"reshape target rank must be 1D or 2D in v0; got {new_shape_t}"
            )
        prod = 1
        for d in new_shape_t:
            prod *= d
        if prod != src.element_count:
            raise ValueError(
                f"reshape element-count mismatch: {src.shape} "
                f"({src.element_count}) → {new_shape_t} ({prod})"
            )
        out_t = self._reserve_output(out, new_shape_t)
        op = ShapeOp(out=out_t, input=src)
        self._commit_output(out_t)
        self._ops.append(op)
        return out_t

    def reduction(self, op: str, out: str, x: str, axis: int = -1) -> Tensor:
        """Last-axis reduction over a 2D tensor. Output is 1D of length
        x.shape[0]. Op validity (axis position, supported op name,
        output shape) is enforced by `ReductionOp.__post_init__`."""
        from compute.reduction.reduction import REDUCTION_OPS

        if op not in REDUCTION_OPS:
            raise ValueError(f"reduction op {op!r} not in {REDUCTION_OPS}")
        x_t = self._bind_existing(x)
        normalized = axis if axis >= 0 else len(x_t.shape) + axis
        # Best-effort shape so SSA can hold the output Tensor; if axis is
        # out of range we fall back to dropping the last dim and let
        # ReductionOp.__post_init__ produce the real error.
        if 0 <= normalized < len(x_t.shape):
            out_shape = x_t.shape[:normalized] + x_t.shape[normalized + 1 :]
        else:
            out_shape = x_t.shape[:-1] or (1,)
        out_t = self._reserve_output(out, out_shape)
        red = ReductionOp(out=out_t, op=op, input=x_t, axis=axis)
        self._commit_output(out_t)
        self._ops.append(red)
        return out_t

    # -- finalize ----------------------------------------------------------

    def build(self) -> Program:
        """Return a fresh copy of the IR Program. Builder remains usable
        for further appends if the caller wants to extend."""
        return list(self._ops)

    @property
    def tensors(self) -> dict[str, Tensor]:
        """Read-only view of the name → Tensor registry."""
        return dict(self._tensors)

    @property
    def uploads(self) -> dict[str, np.ndarray]:
        return dict(self._uploads)


def _required_storage_elements(t: Tensor) -> int:
    if len(t.shape) == 0:
        return 1
    if len(t.shape) == 1:
        return (t.shape[0] - 1) * t.row_stride + 1
    rows, cols = t.shape
    return (rows - 1) * t.row_stride + (cols - 1) * t.col_stride + 1
