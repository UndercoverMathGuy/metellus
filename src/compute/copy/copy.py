"""Strided copy kernel — the fallback for reshapes that can't be
expressed as a metadata swap.

Minimal by design: one thread per output element, scatter-load from
input via its strides, contiguous-store to output. No threadgroup
memory. The kernel walks output's linear (row-major) positions; each
position `i` is unraveled into the source's logical (in_row, in_col)
and read with the source's actual strides — so a transposed or
otherwise non-contiguous source still reads the right bytes.

The fragment hardcodes input shape and strides as MSL literals (no
runtime dim vars beyond `N`, the total element count). Kernels are
recompiled per (source shape, source strides) pair — the existing
matmul / elementwise pipelines already follow this recompile-per-shape
convention, so this matches.
"""

from __future__ import annotations

from dataclasses import dataclass

from compute.fragments import CodegenContext


@dataclass(frozen=True)
class StridedCopyFragment:
    """One thread = one element. Inputs and outputs share total element
    count `N`. `input_cols` is the source's logical 2nd-dim size used to
    unravel the linear index (use the 1D length for 1D sources)."""

    input_name: str
    output_name: str
    input_row_stride: int
    input_col_stride: int
    input_cols: int
    name: str = "strided_copy"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        # `flat_tid` is the in-threadgroup linear thread id. We add the
        # group's contribution to get a global linear index, then bound
        # against N (declared via the dims preamble).
        return f"""\
uint global_idx = (threadgroup_position_in_grid.x * {ctx.tg_x}) + flat_tid;
if (global_idx >= N) {{
    return;
}}
uint in_row = global_idx / {self.input_cols};
uint in_col = global_idx % {self.input_cols};
{self.output_name}[global_idx] = {self.input_name}[in_row * {self.input_row_stride} + in_col * {self.input_col_stride}];"""
