from dataclasses import dataclass
import textwrap


@dataclass(frozen=True)
class KernelBlock:
    name: str
    code: str


@dataclass(frozen=True)
class KernelScaffold:
    function_name: str
    buffers: tuple[str, ...]
    dims: tuple[str, ...]
    threadgroup_decls: tuple[str, ...]
    sections: tuple[str | KernelBlock, ...]
    tg_x: int
    tg_y: int = 1


def indent(code: str, spaces: int = 4) -> str:
    return textwrap.indent(code.strip(), " " * spaces)


def dim_decls(names: tuple[str, ...], dims_name: str = "dims") -> str:
    return "\n".join(
        f"uint {name} = {dims_name}[{idx}];" for idx, name in enumerate(names)
    )


def kernel_preamble(tg_x: int, dims: tuple[str, ...], dims_name: str = "dims") -> str:
    return f"""\
{dim_decls(dims, dims_name)}
uint2 tg = threadgroup_position_in_grid;
uint2 lid = thread_position_in_threadgroup;
uint flat_tid = lid.y * {tg_x} + lid.x;
"""


def block(name: str, code: str) -> KernelBlock:
    return KernelBlock(name=name, code=code)


def barrier(name: str = "threadgroup_barrier") -> KernelBlock:
    return block(name, "threadgroup_barrier(mem_flags::mem_threadgroup);")


def render_section(section: str | KernelBlock) -> str:
    if isinstance(section, KernelBlock):
        return f"// {section.name}\n{section.code.strip()}"
    return section.strip()


def metal_kernel(scaffold: KernelScaffold) -> str:
    params = [
        *scaffold.buffers,
        "constant uint* dims [[buffer(4)]]",
        "uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
        "uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
    ]
    body = "\n".join(
        (
            kernel_preamble(scaffold.tg_x, scaffold.dims),
            *scaffold.threadgroup_decls,
            *(render_section(section) for section in scaffold.sections),
        )
    )
    return f"""\
#include <metal_stdlib>
using namespace metal;

kernel void {scaffold.function_name}(
    {",\n    ".join(params)}
) {{
{indent(body, 4)}
}}
"""
