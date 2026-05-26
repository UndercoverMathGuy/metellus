"""Reduction strategy: standalone, prologue-fused, or epilogue-fused
last-axis reduction."""

from __future__ import annotations

from compute.fragments import CodegenContext
from compute.reduction.reduction import (
    LastAxisReductionComputeFragment,
    LastAxisReductionSetupFragment,
    StoreReductionResultFragment,
)
from orchestrator.ir import ReductionOp, Tensor
from orchestrator.kernel_group import KernelGroup
from runtime.program import Kernel

from orchestrator.assembly.decision import DecisionView
from orchestrator.assembly.expressions import (
    build_reduction_epilogue_transform,
    chain_value_transform,
)
from orchestrator.assembly.extras import Extras


def assemble_reduction(
    decision: DecisionView, function_name: str | None
) -> KernelGroup:
    anchor = decision.anchor
    assert isinstance(anchor, ReductionOp)
    # When a prologue is absorbed, the actual device input is the
    # outermost prologue elem's primary — `anchor.input` is the IR name
    # of the post-prologue value, which has been fused away and has no
    # buffer. Strides / shape / buffer_key all come from the source.
    src = decision.prologue_a[0].operands[0] if decision.prologue_a else anchor.input
    assert isinstance(src, Tensor)
    M, K = src.shape
    final_out = decision.epilogue[-1].out if decision.epilogue else anchor.out

    base_slots = 2  # input=0, output=1
    extras = Extras(
        base_slots_by_key={
            src.buffer_key: 0,
            final_out.buffer_key: 1,
        }
    )

    # Prologue → value_transform on the per-element load inside the reduction.
    # `row`/`idx` are the in-scope vars from LastAxisReductionComputeFragment;
    # the prologue indexes the source via its actual row_stride.
    prologue_tf = None
    if decision.prologue_a:
        prologue_tf = chain_value_transform(
            decision.prologue_a,
            row_var="row",
            col_var="idx",
            extras=extras,
            base_slot=base_slots,
        )

    epilogue_tf = (
        build_reduction_epilogue_transform(decision.epilogue)
        if decision.epilogue
        else None
    )

    tg_x = 128
    setup = LastAxisReductionSetupFragment(rows_dim="M", name="reduction_setup")
    compute = LastAxisReductionComputeFragment(
        op=anchor.op,
        input_name=src.buffer_key,
        scratch_name="scratch",
        reduce_dim="K",
        row_stride=str(src.row_stride),
        col_stride=str(src.col_stride),
        name="reduction_compute",
        tg_x=tg_x,
        value_transform=prologue_tf,
    )
    store = StoreReductionResultFragment(
        output_name=final_out.buffer_key,
        scratch_name="scratch",
        name="reduction_store",
        tg_x=tg_x,
        value_transform=epilogue_tf,
    )

    base_buffers = [
        f"device const float* {src.buffer_key} [[buffer(0)]]",
        f"device float* {final_out.buffer_key} [[buffer(1)]]",
    ]
    fn_name = function_name or f"reduce_{anchor.out.name}_fused"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=tuple(base_buffers + extras.buffers),
        dims=("M", "K"),
        tg_x=tg_x,
        tg_y=1,
        threadgroup_decls=(f"threadgroup float scratch[{(tg_x + 31) // 32}];",),
    )
    kernel = Kernel(fragments=(setup, compute, store), ctx=ctx)
    bindings = (src.buffer_key, final_out.buffer_key, *extras.bindings)
    return KernelGroup(
        kernel=kernel,
        bindings=bindings,
        dims=(M, K),
        grid=(M, 1, 1),
        threads=(tg_x, 1, 1),
        ops=decision.ops,
        strategy=decision.strategy,
    )
