from dataclasses import dataclass
from typing import Callable

from compute.fragments import CodegenContext, TgmemAccess


REDUCTION_OPS = ("sum", "max", "min", "product")


def _scratch_floats(tg_x: int) -> int:
    # One float per SIMD group in a threadgroup of `tg_x` threads (SIMD width
    # is 32 on Apple Silicon).
    return (tg_x + 31) // 32


ValueTransform = Callable[[str], str]
"""Wraps a scalar value expression. Used to inline an elementwise transform
on the read side (elem→reduction fusion) or before the final store
(reduction→elem fusion). The orchestrator is responsible for declaring any
extra buffers referenced by the produced expression."""


@dataclass(frozen=True)
class LastAxisReductionSetupFragment:
    rows_dim: str
    name: str
    kind: str = "setup"

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        return ()

    def render(self, ctx: CodegenContext) -> str:
        return f"""\
uint row = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint simd_lane = lane & 31;
uint simd_id = lane >> 5;
uint simd_count = ({ctx.tg_x} + 31) >> 5;
if (row >= {self.rows_dim}) {{
    return;
}}"""


@dataclass(frozen=True)
class LastAxisReductionComputeFragment:
    """Per-row reduction. The canonical per-element fetch is
    `{input_name}[row * {row_stride} + idx_term]`. Pass `value_transform` to
    wrap that expression with an inline elementwise transform (the orchestrator
    uses this for elem→reduction fusion; the transform may also reference
    `row` and `idx` to weave in additional buffers loaded once per row or per
    element). The orchestrator owns declaring any extra buffers."""

    op: str
    input_name: str
    scratch_name: str
    reduce_dim: str
    row_stride: str
    name: str
    tg_x: int
    col_stride: str = "1"
    value_transform: ValueTransform | None = None
    kind: str = "compute"

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        size = _scratch_floats(self.tg_x)
        return (
            TgmemAccess(
                name=self.scratch_name,
                access="readwrite",
                size_floats=size,
                shape=(1, size),
            ),
        )

    def render(self, ctx: CodegenContext) -> str:
        if self.op not in REDUCTION_OPS:
            raise ValueError(f"unsupported reduction op: {self.op}")
        identity = _identity(self.op)
        combine = _combine_expr(self.op, "acc", "value")
        idx_term = "idx" if self.col_stride == "1" else f"idx * ({self.col_stride})"
        raw_fetch = f"{self.input_name}[row * {self.row_stride} + {idx_term}]"
        value_expr = (
            self.value_transform(raw_fetch)
            if self.value_transform is not None
            else raw_fetch
        )
        return f"""\
uint end = {self.reduce_dim};
float acc = {identity};
for (uint idx = lane; idx < end; idx += {ctx.tg_x}) {{
    float value = {value_expr};
    acc = {combine};
}}
acc = {_simd_reduce_expr(self.op, "acc")};
if (simd_lane == 0) {{
    {self.scratch_name}[simd_id] = acc;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {{
    float out = {identity};
    if (simd_lane < simd_count) {{
        out = {self.scratch_name}[simd_lane];
    }}
    out = {_simd_reduce_expr(self.op, "out")};
    if (simd_lane == 0) {{
        {self.scratch_name}[0] = out;
    }}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);"""


@dataclass(frozen=True)
class StoreReductionResultFragment:
    """Write the finished per-row reduction (held in `scratch[0]` by the
    leader thread) out to `output_name[row]`. Pass `value_transform` to apply
    an elementwise op to the reduced scalar before storing (reduction→elem
    fusion). `setup` is a tuple of extra MSL lines emitted inside the
    leader-thread guard before the transform — used to load row-indexed
    extras (e.g. `float bias_row = Bias[row];`). The orchestrator declares
    any extra buffers referenced."""

    output_name: str
    scratch_name: str
    name: str
    tg_x: int
    value_transform: ValueTransform | None = None
    setup: tuple[str, ...] = ()
    kind: str = "store"

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        size = _scratch_floats(self.tg_x)
        return (
            TgmemAccess(
                name=self.scratch_name,
                access="read",
                size_floats=size,
                shape=(1, size),
            ),
        )

    def render(self, ctx: CodegenContext) -> str:
        setup_block = "\n    ".join(self.setup)
        raw = f"{self.scratch_name}[0]"
        if self.value_transform is None and not self.setup:
            body = f"{self.output_name}[row] = {raw};"
        else:
            transformed = (
                self.value_transform("v") if self.value_transform is not None else "v"
            )
            body = "\n    ".join(
                filter(
                    None,
                    [
                        f"float v = {raw};",
                        setup_block,
                        f"v = {transformed};"
                        if self.value_transform is not None
                        else "",
                        f"{self.output_name}[row] = v;",
                    ],
                )
            )
        return f"""\
if (thread_position_in_threadgroup.x == 0) {{
    {body}
}}"""


def _identity(op: str) -> str:
    if op == "sum":
        return "0.0f"
    if op == "product":
        return "1.0f"
    if op == "max":
        return "-INFINITY"
    if op == "min":
        return "INFINITY"
    raise ValueError(f"unsupported reduction op: {op}")


def _combine_expr(op: str, lhs: str, rhs: str) -> str:
    if op == "sum":
        return f"{lhs} + {rhs}"
    if op == "product":
        return f"{lhs} * {rhs}"
    if op == "max":
        return f"max({lhs}, {rhs})"
    if op == "min":
        return f"min({lhs}, {rhs})"
    raise ValueError(f"unsupported reduction op: {op}")


def _simd_reduce_expr(op: str, value: str) -> str:
    if op == "sum":
        return f"simd_sum({value})"
    if op == "product":
        return f"simd_product({value})"
    if op == "max":
        return f"simd_max({value})"
    if op == "min":
        return f"simd_min({value})"
    raise ValueError(f"unsupported reduction op: {op}")
