"""Lowered-primitive IR.

The orchestrator's input. Every Op is one of three primitives — matmul,
elementwise, reduction — referencing Tensor SSA values by name. A Program
is just an ordered list of Ops; each Op defines exactly one output Tensor
whose name is unique across the program.

Shape correctness is enforced on the Op dataclasses themselves: every
Op validates its operand shapes (and the broadcast modes against them)
in `__post_init__`, so an ill-formed IR cannot be constructed even by
hand. Tensor metadata (shape, layout, strides, dtype) is the single
source of truth — ops never carry duplicate dim arguments.

Higher-level surface ops (relu / softmax / layernorm / ...) decompose into
this IR in a layer *above* the orchestrator; nothing in this package
introduces such shims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from compute.elementwise.elementwise import BroadcastSpec, elementwise_arity


class Layout(StrEnum):
    """Logical storage order of a 2D tensor. ROW_MAJOR is the canonical
    default; COL_MAJOR exists so a "logical transpose" is a metadata swap
    (shape + strides + layout) rather than a data copy."""

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


def _default_strides(shape: tuple[int, ...], layout: Layout) -> tuple[int, int]:
    """Strides in elements for the contiguous layout of `shape`.

    Returns (row_stride, col_stride). For 1D tensors only row_stride is
    meaningful; col_stride is 0. For 0D both are 0."""
    if len(shape) == 0:
        return (0, 0)
    if len(shape) == 1:
        return (1, 0)
    if len(shape) == 2:
        M, N = shape
        if layout is Layout.ROW_MAJOR:
            return (N, 1)
        return (1, M)
    raise ValueError(f"v0 supports 0D/1D/2D tensors only; got shape {shape}")


@dataclass(frozen=True)
class Tensor:
    """An SSA value. `name` is this SSA value's identifier. `buffer_key`
    is the env key the runtime uses to look up the backing `Buffer` —
    for a fresh tensor it equals `name`, but for a *view* (transpose or
    metadata-only reshape) it points at the storage owner so two SSA
    values can alias the same buffer with different shape / stride
    interpretations.

    `shape` is concrete (no symbolic dims in v0) and is the single
    source of truth for an operand's dims — ops never restate them.

    Layout / strides describe how logical (row, col) indices map to a
    linear element offset:  offset = row * row_stride + col * col_stride.
    Defaults are the contiguous strides for the given `layout`. Pass
    explicit strides only when interfacing with caller-supplied buffers
    whose strides aren't contiguous (or when constructing a view via
    `.transpose()` / `.reshape_view()`)."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    layout: Layout = Layout.ROW_MAJOR
    # Sentinels: row_stride/col_stride == -1 means "fill with contiguous default
    # from shape + layout"; buffer_key == "" means "alias to name". Post-init
    # replaces sentinels with concrete values, so post-construction these
    # attributes are always concrete (hence non-Optional types).
    row_stride: int = -1
    col_stride: int = -1
    buffer_key: str = ""

    def __post_init__(self) -> None:
        if any(d <= 0 for d in self.shape):
            raise ValueError(
                f"tensor dimensions must be positive; got shape {self.shape}"
            )
        if self.row_stride == -1 or self.col_stride == -1:
            rs, cs = _default_strides(self.shape, self.layout)
            if self.row_stride == -1:
                object.__setattr__(self, "row_stride", rs)
            if self.col_stride == -1:
                object.__setattr__(self, "col_stride", cs)
        if not self.buffer_key:
            object.__setattr__(self, "buffer_key", self.name)

    @property
    def nbytes(self) -> int:
        n = 4  # fp32 only in v0; widen when dtype field starts varying
        for d in self.shape:
            n *= d
        return n

    @property
    def element_count(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def linear_step(self) -> int | None:
        """Returns `c` such that row-major iteration over this tensor
        produces buffer offsets `0, c, 2c, …, (N-1)·c`. Returns `None`
        when iteration is *non-linear* — i.e., a real `ShapeOp` is
        required for any reshape other than same-shape.

        Conditions:

          - 0D: trivially linear (single element), `c = 1`.
          - 1D: linear with `c = row_stride`.
          - 2D with a singleton dim: only the non-singleton dim's
            stride matters; that's `c`.
          - 2D `(M, K)` with `M, K > 1`: linear iff
            `row_stride == K * col_stride`, then `c = col_stride`.
            (Row-major dense is the `c == 1` instance; any uniform-step
            "stride-c" view also qualifies.)
          - Anything else (e.g., col-major non-singleton, padded rows):
            `None`.

        When this is not `None`, the tensor can be reshaped to ANY
        rank-≤-2 same-element-count shape as a metadata view: the
        output gets strides `(K' * c, c)` for 2D targets or `(c,)` for
        1D, and aliases the source's buffer."""
        if len(self.shape) == 0:
            return 1
        if len(self.shape) == 1:
            return self.row_stride
        M, K = self.shape
        if M == 1:
            return self.col_stride
        if K == 1:
            return self.row_stride
        if self.row_stride == K * self.col_stride:
            return self.col_stride
        return None

    def is_contiguous(self) -> bool:
        """C-contiguous (dense row-major): `linear_step() == 1`. Kept
        as a convenience for the common case; reshape eligibility uses
        the more general `linear_step()`."""
        return self.linear_step() == 1

    def transpose(self, name: str) -> Tensor:
        """Logical 2D transpose — no data movement. Returns a fresh SSA
        value `name` that aliases this tensor's buffer (`buffer_key` is
        carried over) with shape, strides, and layout swapped. The
        caller chooses `name` so the builder can register the view
        without clashing with the source's name."""
        if len(self.shape) != 2:
            raise ValueError(f"transpose is 2D-only in v0; got shape {self.shape}")
        M, N = self.shape
        new_layout = (
            Layout.COL_MAJOR if self.layout is Layout.ROW_MAJOR else Layout.ROW_MAJOR
        )
        return Tensor(
            name=name,
            shape=(N, M),
            dtype=self.dtype,
            layout=new_layout,
            row_stride=self.col_stride,
            col_stride=self.row_stride,
            buffer_key=self.buffer_key,
        )

    def can_reshape_as_view(self, new_shape: tuple[int, ...]) -> bool:
        """Reshape is a pure metadata swap iff:

          (a) `new_shape` equals `self.shape` (no-op rename — safe at
              any strides), OR
          (b) `new_shape` has the same element count, rank ≤ 2, AND
              `linear_step()` is not None — i.e., row-major iteration
              over the source produces uniformly-stepped offsets, so
              the buffer can be re-iterated with output strides
              `(K' · c, c)` for any target shape.

        `ShapeOp` is only emitted when neither path applies — the most
        dire case where a logical row-major reinterpretation would have
        to physically reorder elements."""
        new_shape_t = tuple(new_shape)
        if new_shape_t == self.shape:
            return True
        if len(new_shape_t) > 2 or len(new_shape_t) == 0:
            return False
        prod = 1
        for d in new_shape_t:
            prod *= d
        if prod != self.element_count:
            return False
        return self.linear_step() is not None

    def reshape_view(self, new_shape: tuple[int, ...], name: str) -> Tensor:
        """Build a metadata-swap reshape view aliasing this buffer.
        Caller is responsible for checking `can_reshape_as_view` first;
        this raises otherwise.

          - Same-shape reshape: a no-op rename — layout / strides are
            carried over from the source.
          - Otherwise: the source's per-element step `c` (from
            `linear_step`) is the new tensor's inner stride. For a 1D
            target `(N,)` the row_stride is `c`. For a 2D target
            `(M', K')` the strides are `(K' · c, c)`.  Layout is
            reported as `ROW_MAJOR` (the strides have row-major
            ordering even when `c > 1`)."""
        if not self.can_reshape_as_view(new_shape):
            raise ValueError(
                f"cannot reshape {self.name!r} {self.shape} → {tuple(new_shape)} "
                "as a metadata swap: needs same shape OR a linearly-iterable "
                "source with matching element count (ranks 0D/1D/2D only)"
            )
        new_shape_t = tuple(new_shape)
        if new_shape_t == self.shape:
            return Tensor(
                name=name,
                shape=self.shape,
                dtype=self.dtype,
                layout=self.layout,
                row_stride=self.row_stride,
                col_stride=self.col_stride,
                buffer_key=self.buffer_key,
            )
        c = self.linear_step()
        assert c is not None  # guarded by can_reshape_as_view
        if len(new_shape_t) == 1:
            rs, cs = c, 0
        else:
            rs, cs = new_shape_t[1] * c, c
        return Tensor(
            name=name,
            shape=new_shape_t,
            dtype=self.dtype,
            layout=Layout.ROW_MAJOR,
            row_stride=rs,
            col_stride=cs,
            buffer_key=self.buffer_key,
        )


@dataclass(frozen=True)
class Scalar:
    """A compile-time constant elementwise operand. No buffer binding —
    codegen inlines `value` as a Metal literal. Used for things like the
    `0` in `max(x, 0)` (relu) without forcing the caller to allocate a
    (1,1) scalar buffer."""

    value: float


OperandValue = Tensor | Scalar


def _expected_broadcast_shape(
    primary_shape: tuple[int, ...], mode: BroadcastSpec
) -> tuple[int, ...]:
    """The operand shape implied by `mode` against `primary_shape`. NONE
    means same-shape; the others require a 2D primary."""
    if mode is BroadcastSpec.NONE:
        return primary_shape
    if len(primary_shape) != 2:
        raise ValueError(
            f"broadcast={mode!r} requires a 2D primary; got shape {primary_shape}"
        )
    M, N = primary_shape
    return {
        BroadcastSpec.SCALAR: (1, 1),
        BroadcastSpec.ROW: (1, N),
        BroadcastSpec.COL: (M, 1),
    }[mode]


def _check_operand_broadcast(
    label: str,
    primary: Tensor,
    operand: OperandValue,
    mode: BroadcastSpec,
) -> None:
    """Validate a single secondary operand. Scalars are inlined as
    literals at codegen time, so they have no broadcast slot and must
    carry mode=NONE. Tensor operands must have the shape implied by the
    declared broadcast mode."""
    if isinstance(operand, Scalar):
        if mode is not BroadcastSpec.NONE:
            raise ValueError(
                f"{label}_broadcast={mode!r} cannot be set for a Scalar "
                "operand — scalars are inlined, not broadcast"
            )
        return
    expected = _expected_broadcast_shape(primary.shape, mode)
    if operand.shape != expected:
        if mode is BroadcastSpec.NONE:
            raise ValueError(
                f"{label}-operand shape {operand.shape} must match "
                f"primary {primary.shape} when {label}_broadcast=NONE"
            )
        raise ValueError(
            f"{label}-operand shape {operand.shape} incompatible "
            f"with {label}_broadcast={mode!r} (expected {expected})"
        )


@dataclass(frozen=True)
class MatmulOp:
    """out = a @ b. 2D only in v0. Strides/layout on `a` and `b` describe
    how to read the inputs; the matmul template honours them so a
    transposed view (`a.transpose()`) is a metadata-only effect.

    Shapes and dtype are validated at construction: both operands must be
    2D, the contraction axes must match, `out.shape` must equal
    (M, N) = (a.shape[0], b.shape[1]), and all three tensors must share a
    dtype."""

    out: Tensor
    a: Tensor
    b: Tensor

    def __post_init__(self) -> None:
        if len(self.a.shape) != 2 or len(self.b.shape) != 2:
            raise ValueError(
                f"matmul operands must be 2D; got a={self.a.shape}, b={self.b.shape}"
            )
        if self.a.shape[1] != self.b.shape[0]:
            raise ValueError(
                f"matmul contraction-axis mismatch: a={self.a.shape}, b={self.b.shape}"
            )
        expected = (self.a.shape[0], self.b.shape[1])
        if self.out.shape != expected:
            raise ValueError(
                f"matmul out shape {self.out.shape} does not match a @ b = {expected}"
            )
        if not (self.a.dtype == self.b.dtype == self.out.dtype):
            raise ValueError(
                f"matmul dtypes must match; got a={self.a.dtype}, "
                f"b={self.b.dtype}, out={self.out.dtype}"
            )

    @property
    def inputs(self) -> tuple[Tensor, ...]:
        return (self.a, self.b)


@dataclass(frozen=True)
class ElementwiseOp:
    """Pointwise op. `op` is one of the names in
    compute.elementwise.{UNARY,BINARY,TERNARY}_EXPRESSIONS.

    `operands` may contain `Scalar` values for compile-time constants
    (e.g. relu = `max(x, 0)`); `inputs` filters them out so the DAG only
    tracks tensor dependencies.

    `y_broadcast` / `cond_broadcast` describe how the 2nd/3rd *tensor*
    operand maps to the output's (M, N) shape. They are validated
    against the operand's `Tensor.shape` at construction; for `Scalar`
    operands they must remain `NONE` (scalars are inlined, not
    broadcast).

    All validation — arity, primary type, output shape, operand shapes
    vs declared broadcast modes — happens in `__post_init__`, so the
    builder is just a name-resolution layer."""

    out: Tensor
    op: str
    operands: tuple[OperandValue, ...]
    y_broadcast: BroadcastSpec = BroadcastSpec.NONE
    cond_broadcast: BroadcastSpec = BroadcastSpec.NONE

    def __post_init__(self) -> None:
        arity = elementwise_arity(self.op)
        if len(self.operands) != arity:
            raise ValueError(
                f"elementwise {self.op!r} is arity-{arity}, got "
                f"{len(self.operands)} operands"
            )
        primary = self.operands[0]
        if not isinstance(primary, Tensor):
            raise ValueError(
                f"elementwise {self.op!r}: operand[0] must be a Tensor; "
                f"got {primary!r}. Scalars are only allowed as secondary "
                "operands."
            )
        if self.out.shape != primary.shape:
            raise ValueError(
                f"elementwise out shape {self.out.shape} must match "
                f"primary operand shape {primary.shape}"
            )
        if arity >= 2:
            _check_operand_broadcast("y", primary, self.operands[1], self.y_broadcast)
        elif self.y_broadcast is not BroadcastSpec.NONE:
            raise ValueError(
                f"y_broadcast={self.y_broadcast!r} set on arity-1 op {self.op!r}"
            )
        if arity == 3:
            _check_operand_broadcast(
                "cond", primary, self.operands[2], self.cond_broadcast
            )
        elif self.cond_broadcast is not BroadcastSpec.NONE:
            raise ValueError(
                f"cond_broadcast={self.cond_broadcast!r} set on arity-{arity} "
                f"op {self.op!r}"
            )

    @property
    def inputs(self) -> tuple[Tensor, ...]:
        return tuple(o for o in self.operands if isinstance(o, Tensor))


@dataclass(frozen=True)
class ReductionOp:
    """Last-axis reduction over a 2D tensor. `op` is one of
    compute.reduction.REDUCTION_OPS. `axis` is normalized against
    `input.shape` and must point at the last dim; `out.shape` must equal
    the input shape with that axis dropped."""

    out: Tensor
    op: str
    input: Tensor
    axis: int = -1

    def __post_init__(self) -> None:
        if len(self.input.shape) != 2:
            raise ValueError(
                f"reduction input must be 2D in v0; got shape {self.input.shape}"
            )
        normalized = self.axis if self.axis >= 0 else len(self.input.shape) + self.axis
        if normalized != len(self.input.shape) - 1:
            raise ValueError(
                f"reduction supports last axis only in v0; got axis="
                f"{self.axis} on shape {self.input.shape}"
            )
        expected = self.input.shape[:normalized] + self.input.shape[normalized + 1 :]
        if self.out.shape != expected:
            raise ValueError(
                f"reduction out shape {self.out.shape} does not match "
                f"input {self.input.shape} reduced along axis {self.axis} "
                f"(expected {expected})"
            )

    @property
    def inputs(self) -> tuple[Tensor, ...]:
        return (self.input,)


@dataclass(frozen=True)
class ShapeOp:
    """Pure data rearrangement. Emitted as the fallback for reshapes
    that can't be expressed as a metadata-swap view (source is
    non-contiguous, or the logical row-major element order differs from
    the source's storage order). Codegen reads every element of
    `input` via its strides and writes to `out` row-major contiguous.

    `out` owns its own buffer (`out.buffer_key == out.name`); the IR
    enforces element-count and dtype match. No compute, no fusion in
    v0 — the fuser treats ShapeOp as a standalone kernel."""

    out: Tensor
    input: Tensor

    def __post_init__(self) -> None:
        if self.input.element_count != self.out.element_count:
            raise ValueError(
                f"shape op element-count mismatch: input "
                f"{self.input.shape} ({self.input.element_count} elems) → "
                f"out {self.out.shape} ({self.out.element_count} elems)"
            )
        if self.input.dtype != self.out.dtype:
            raise ValueError(
                f"shape op dtype mismatch: input {self.input.dtype!r} → "
                f"out {self.out.dtype!r}"
            )
        if self.out.buffer_key != self.out.name:
            raise ValueError(
                f"shape op out must own its buffer (buffer_key == name); "
                f"got name={self.out.name!r}, buffer_key={self.out.buffer_key!r}"
            )

    @property
    def inputs(self) -> tuple[Tensor, ...]:
        return (self.input,)


Op = MatmulOp | ElementwiseOp | ReductionOp | ShapeOp
Program = list[Op]
