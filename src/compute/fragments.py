from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Any, Literal

KernelFragment = Any

TgmemAccessKind = Literal["read", "write", "readwrite"]


@dataclass(frozen=True)
class TgmemAccess:
    name: str
    access: TgmemAccessKind
    size_floats: int
    shape: tuple[int, int]


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

    @property
    def is_tgmem_barrier(self) -> bool:
        return True


@dataclass(frozen=True)
class AliasFragment:
    """Pointer-alias `new_name` onto `old_name`'s threadgroup storage.

    Emitted by the aliasing rewriter at the seam between two tenants
    that share a physical slot — the prior tenant (`old_name`) has
    just gone out of life, the next (`new_name`) is about to come in.
    After this fragment, every `new_name[r][c]` reference compiles to
    the same bytes as the corresponding region of `old_name`.

    Aliasing reuses storage, so every thread must finish with the
    prior tenant before any thread touches the new alias — i.e. a
    `threadgroup_barrier(mem_flags::mem_threadgroup)` is required
    between the last write/read of `old_name` and the first reference
    via `new_name`. `preceded_by_barrier` controls whether this
    fragment emits that barrier itself:

      - `True`  → the surrounding template already places a barrier
        immediately before this fragment; the alias line is emitted
        on its own and we don't double up.
      - `False` → emit the barrier, then the alias line.

    The pointer is typed `threadgroup float (*)[<new_cols>]` so 2D
    indexing through `new_name` decodes to the new tenant's stride
    independent of how the underlying storage was originally declared.
    """

    old_name: str
    new_name: str
    new_shape: tuple[int, int]
    preceded_by_barrier: bool
    name: str = "alias"
    kind: str = "alias"

    def render(self, ctx: CodegenContext) -> str:
        cols = self.new_shape[1]
        alias_line = f"threadgroup float (*{self.new_name})[{cols}] = (threadgroup float (*)[{cols}]){self.old_name};"
        if self.preceded_by_barrier:
            return alias_line
        return f"threadgroup_barrier(mem_flags::mem_threadgroup);\n{alias_line}"

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        return ()


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
