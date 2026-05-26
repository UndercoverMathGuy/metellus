"""Elementwise strategy: standalone op or chain of ops, no anchor."""

from __future__ import annotations

from compute.elementwise.tiled_chain import TiledElementwiseChainFragment
from compute.fragments import BarrierFragment, CodegenContext
from memory.memory import TgLoadFragment, TgStoreFragment
from orchestrator.ir import Tensor
from orchestrator.kernel_group import KernelGroup
from runtime.program import Kernel

from orchestrator.assembly.chain_tiles import build_chain_y_tile_loads
from orchestrator.assembly.decision import DecisionView
from orchestrator.assembly.extras import Extras


def assemble_elementwise(
    decision: DecisionView, function_name: str | None
) -> KernelGroup:
    chain = decision.chain_only
    assert chain, "elementwise decision must carry a chain"
    primary = chain[0].operands[0]
    assert isinstance(primary, Tensor)
    if len(primary.shape) == 1:
        M, N = primary.shape[0], 1
    else:
        M, N = primary.shape

    tile_M, tile_N = 16, 16
    tg_x, tg_y = 32, 8
    num_threads = tg_x * tg_y

    final_out = chain[-1].out
    base_slots = 2  # primary in=0, out=1
    extras = Extras(
        base_slots_by_key={
            primary.buffer_key: 0,
            final_out.buffer_key: 1,
        }
    )

    load_fragments, extra_tile_decls, y_tile_for, cond_tile_for = (
        build_chain_y_tile_loads(
            chain,
            seed_available={primary.name},
            tile_M=tile_M,
            tile_N=tile_N,
            num_threads=num_threads,
            extras=extras,
            base_slots=base_slots,
        )
    )
    tg_decls = [
        f"threadgroup float X_tile[{tile_M}][{tile_N}];",
        *extra_tile_decls,
    ]

    primary_load = TgLoadFragment(
        name="load_X_tile",
        src_name=primary.buffer_key,
        src_row_stride=str(primary.row_stride),
        src_col_stride=str(primary.col_stride),
        row_start=f"tg.y * {tile_M}",
        col_start=f"tg.x * {tile_N}",
        dst_name="X_tile",
        tile_shape=(tile_M, tile_N),
        num_threads=num_threads,
        row_limit="M",
        col_limit="N",
    )
    compute = TiledElementwiseChainFragment(
        chain=chain,
        primary_tile="X_tile",
        tile_shape=(tile_M, tile_N),
        num_threads=num_threads,
        y_tile_for=y_tile_for,
        cond_tile_for=cond_tile_for,
    )
    store = TgStoreFragment(
        name="store_X_tile",
        src_name="X_tile",
        dst_name=final_out.buffer_key,
        dst_row_stride="N",
        row_start=f"tg.y * {tile_M}",
        col_start=f"tg.x * {tile_N}",
        tile_shape=(tile_M, tile_N),
        num_threads=num_threads,
        row_limit="M",
        col_limit="N",
    )

    fragments = (
        primary_load,
        *load_fragments,
        BarrierFragment("inputs_ready"),
        compute,
        BarrierFragment("compute_done"),
        store,
    )
    base_buffers = [
        f"device const float* {primary.buffer_key} [[buffer(0)]]",
        f"device float* {final_out.buffer_key} [[buffer(1)]]",
    ]
    fn_name = function_name or f"elementwise_{final_out.name}_fused"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=tuple(base_buffers + extras.buffers),
        dims=("M", "N"),
        tg_x=tg_x,
        tg_y=tg_y,
        threadgroup_decls=tuple(tg_decls),
        preamble=(
            "uint2 tg = threadgroup_position_in_grid;",
            "uint2 lid = thread_position_in_threadgroup;",
            f"uint flat_tid = lid.y * {tg_x} + lid.x;",
        ),
    )
    kernel = Kernel(fragments=fragments, ctx=ctx)
    bindings = (primary.buffer_key, final_out.buffer_key, *extras.bindings)
    return KernelGroup(
        kernel=kernel,
        bindings=bindings,
        dims=(M, N),
        grid=((N + tile_N - 1) // tile_N, (M + tile_M - 1) // tile_M, 1),
        threads=(tg_x, tg_y, 1),
        ops=decision.ops,
        strategy=decision.strategy,
    )
