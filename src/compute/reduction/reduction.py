from dataclasses import dataclass
from typing import Callable

from compute.fragments import CodegenContext


REDUCTION_OPS = ("sum", "max", "min", "product")


ValueTransform = Callable[[str], str]
"""Wraps a scalar value expression. Used to inline an elementwise transform
on the read side (elem→reduction fusion) or before the final store
(reduction→elem fusion). The orchestrator is responsible for declaring any
extra buffers referenced by the produced expression."""


@dataclass(frozen=True)
class LastAxisReductionSetupFragment:
    rows_dim: str
    name: str
    tree: bool = False
    kind: str = "setup"

    def render(self, ctx: CodegenContext) -> str:
        row_expr = (
            "threadgroup_position_in_grid.y"
            if self.tree
            else "threadgroup_position_in_grid.x"
        )
        block_line = "uint block = threadgroup_position_in_grid.x;" if self.tree else ""
        return f"""\
uint row = {row_expr};
{block_line}
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
    block_dim: str | None = None
    col_stride: str = "1"
    value_transform: ValueTransform | None = None
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        if self.op not in REDUCTION_OPS:
            raise ValueError(f"unsupported reduction op: {self.op}")
        identity = _identity(self.op)
        combine = _combine_expr(self.op, "acc", "value")
        start_line = "uint start = 0;"
        end_line = f"uint end = {self.reduce_dim};"
        loop_start = "lane"
        if self.block_dim is not None:
            start_line = f"uint start = block * {self.block_dim};"
            end_line = f"uint end = min({self.reduce_dim}, start + {self.block_dim});"
            loop_start = "start + lane"
        idx_term = "idx" if self.col_stride == "1" else f"idx * ({self.col_stride})"
        raw_fetch = f"{self.input_name}[row * {self.row_stride} + {idx_term}]"
        value_expr = (
            self.value_transform(raw_fetch)
            if self.value_transform is not None
            else raw_fetch
        )
        return f"""\
{start_line}
{end_line}
float acc = {identity};
for (uint idx = {loop_start}; idx < end; idx += {ctx.tg_x}) {{
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
    value_transform: ValueTransform | None = None
    setup: tuple[str, ...] = ()
    kind: str = "store"

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


@dataclass(frozen=True)
class LastAxisReductionPartialStoreFragment:
    partial_name: str
    scratch_name: str
    num_blocks_dim: str
    name: str
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        return f"""\
if (thread_position_in_threadgroup.x == 0) {{
    {self.partial_name}[row * {self.num_blocks_dim} + block] = {self.scratch_name}[0];
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
