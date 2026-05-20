from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Any

KernelFragment = Any


@dataclass(frozen=True)
class CodegenContext:
    function_name: str
    buffers: tuple[str, ...]
    dims: tuple[str, ...]
    tg_x: int
    tg_y: int
    threadgroup_decls: tuple[str, ...] = ()
    helpers: tuple[str, ...] = ()
    position_type: str = "uint2"
    thread_type: str = "uint2"
    position_expr: str = "threadgroup_position_in_grid"
    thread_expr: str = "thread_position_in_threadgroup"
    preamble: tuple[str, ...] = ()
    dims_buffer_index: int | None = None


@dataclass(frozen=True)
class BarrierFragment:
    name: str = "barrier"
    kind: str = "barrier"

    def render(self, ctx: CodegenContext) -> str:
        return "threadgroup_barrier(mem_flags::mem_threadgroup);"


class CodegenEngine:
    def render(self, fragments: list[KernelFragment], ctx: CodegenContext) -> str:
        dims_index = (
            ctx.dims_buffer_index
            if ctx.dims_buffer_index is not None
            else len(ctx.buffers)
        )
        params = [
            *ctx.buffers,
            f"constant uint* dims [[buffer({dims_index})]]",
            f"{ctx.position_type} threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
            f"{ctx.thread_type} thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
        ]
        body = "\n".join(
            part
            for part in (
                self._dim_decls(ctx.dims),
                *ctx.preamble,
                *ctx.threadgroup_decls,
                *(fragment.render(ctx) for fragment in fragments),
            )
            if part.strip()
        )
        helpers = "\n".join(helper.strip() for helper in ctx.helpers if helper.strip())
        return f"""\
#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void {ctx.function_name}(
    {",\n    ".join(params)}
) {{
{self._indent(body, 4)}
}}
"""

    def render_many(
        self, stages: list[tuple[list[KernelFragment], CodegenContext]]
    ) -> tuple[str, ...]:
        return tuple(self.render(fragments, ctx) for fragments, ctx in stages)

    @staticmethod
    def _dim_decls(dims: tuple[str, ...]) -> str:
        return "\n".join(f"uint {name} = dims[{idx}];" for idx, name in enumerate(dims))

    @staticmethod
    def _indent(code: str, spaces: int) -> str:
        return textwrap.indent(code.strip(), " " * spaces)
