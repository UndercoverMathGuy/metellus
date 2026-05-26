"""Shape (copy) strategy: minimal device→device strided copy."""

from __future__ import annotations

from compute.copy.copy import StridedCopyFragment
from compute.fragments import CodegenContext
from orchestrator.ir import ShapeOp
from orchestrator.kernel_group import KernelGroup
from runtime.program import Kernel

from orchestrator.assembly.decision import DecisionView


def assemble_shape(decision: DecisionView, function_name: str | None) -> KernelGroup:
    """One thread per output element, scatter-load from input via its
    actual strides, contiguous store to output. No threadgroup memory.
    Block size 256 is a round number that fits any element count; the
    in-kernel `if (idx >= N) return;` handles partial tail blocks."""
    shape_op = decision.shape_op
    assert isinstance(shape_op, ShapeOp)
    in_t = shape_op.input
    out_t = shape_op.out
    N = out_t.element_count

    if len(in_t.shape) == 2:
        input_cols = in_t.shape[1]
    elif len(in_t.shape) == 1:
        input_cols = in_t.shape[0]
    else:
        raise ValueError(
            f"ShapeOp input must be 1D or 2D in v0; got shape {in_t.shape}"
        )

    tg_x = 256
    grid_x = (N + tg_x - 1) // tg_x

    copy = StridedCopyFragment(
        input_name=in_t.buffer_key,
        output_name=out_t.buffer_key,
        input_row_stride=in_t.row_stride,
        input_col_stride=in_t.col_stride,
        input_cols=input_cols,
    )
    fn_name = function_name or f"shape_{out_t.name}"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=(
            f"device const float* {in_t.buffer_key} [[buffer(0)]]",
            f"device float* {out_t.buffer_key} [[buffer(1)]]",
        ),
        dims=("N",),
        tg_x=tg_x,
        tg_y=1,
        preamble=(
            "uint2 lid = thread_position_in_threadgroup;",
            f"uint flat_tid = lid.y * {tg_x} + lid.x;",
        ),
    )
    kernel = Kernel(fragments=(copy,), ctx=ctx)
    return KernelGroup(
        kernel=kernel,
        bindings=(in_t.buffer_key, out_t.buffer_key),
        dims=(N,),
        grid=(grid_x, 1, 1),
        threads=(tg_x, 1, 1),
        ops=(shape_op,),
        strategy=decision.strategy,
    )
