"""Matmul strategy: standalone, prologue-fused, tg-tile epilogue,
or register-epilogue matmul."""

from __future__ import annotations

from compute.fragments import BarrierFragment, CodegenContext
from compute.matmul import (
    MatmulAccumToDevFragment,
    MatmulAccumToTgFragment,
    MatmulComputeFragment,
    MatmulConfig,
    MatmulMainloopFragment,
    MatmulRegisterEpilogueFragment,
    MatmulSetupFragment,
    MatmulTgToDevFragment,
    MatmulTileMappingFragment,
    ThreadIndexFragment,
)
from compute.matmul.config import grid_for, is_aligned_shape, select_tile_config
from memory.memory import TgLoadFragment
from orchestrator.ir import MatmulOp, Tensor
from orchestrator.kernel_group import FusionStrategy, KernelGroup
from runtime.program import Kernel

from compute.elementwise.tiled_chain import TiledElementwiseChainFragment

from orchestrator.assembly.chain_tiles import build_chain_y_tile_loads
from orchestrator.assembly.decision import DecisionView
from orchestrator.assembly.expressions import chain_value_transform
from orchestrator.assembly.extras import Extras


def assemble_matmul(decision: DecisionView, function_name: str | None) -> KernelGroup:
    anchor = decision.anchor
    assert isinstance(anchor, MatmulOp)
    # Effective device-input tensors: when a prologue is absorbed, the
    # actual buffer/strides come from the chain's outermost primary, not
    # from anchor.{a,b} (which name post-prologue values that have no
    # materialized buffer).
    src_a = decision.prologue_a[0].operands[0] if decision.prologue_a else anchor.a
    src_b = decision.prologue_b[0].operands[0] if decision.prologue_b else anchor.b
    assert isinstance(src_a, Tensor) and isinstance(src_b, Tensor)
    M, K = anchor.a.shape
    _, N = anchor.b.shape

    tile = select_tile_config(M, K, N)
    aligned = is_aligned_shape(M, K, N, tile)
    has_epilogue = bool(decision.epilogue)
    use_register = decision.strategy is FusionStrategy.MATMUL_EPILOGUE_REGISTER
    # Tg-tile path is required whenever we have a non-lane-agnostic epilogue.
    use_tg_tile_epilogue = has_epilogue and not use_register

    final_out = decision.epilogue[-1].out if decision.epilogue else anchor.out
    final_out_key = final_out.buffer_key

    # Dedup input buffer slots by key so X@X (or any aliased pair) emits one
    # MSL param at one slot. The final out always gets its own slot after
    # the inputs; if it happens to alias an input we still want a writable
    # binding at a fresh slot rather than collapsing onto a const* param.
    input_key_to_slot: dict[str, int] = {}
    for k in (src_a.buffer_key, src_b.buffer_key):
        if k not in input_key_to_slot:
            input_key_to_slot[k] = len(input_key_to_slot)
    out_slot = len(input_key_to_slot)
    base_slots = out_slot + 1
    extras = Extras(base_slots_by_key={**input_key_to_slot, final_out_key: out_slot})

    # Prologue value-transforms (built once each, even when chain is empty).
    a_transform = (
        chain_value_transform(
            decision.prologue_a,
            row_var="global_row",
            col_var="global_col",
            extras=extras,
            base_slot=base_slots,
        )
        if decision.prologue_a
        else None
    )
    b_transform = (
        chain_value_transform(
            decision.prologue_b,
            row_var="global_row",
            col_var="global_col",
            extras=extras,
            base_slot=base_slots,
        )
        if decision.prologue_b
        else None
    )

    # Tg-tile epilogue: load any tensor-typed y operands into tiles
    # before the elementwise chain runs.
    if use_tg_tile_epilogue:
        epilogue_load_fragments, epilogue_tile_decls, y_tile_for, cond_tile_for = (
            build_chain_y_tile_loads(
                decision.epilogue,
                seed_available={anchor.out.name},
                tile_M=tile.tile_M,
                tile_N=tile.tile_N,
                num_threads=tile.num_threads,
                extras=extras,
                base_slots=base_slots,
                y_prefix="eY",
                cond_prefix="eCond",
                aligned=aligned,
            )
        )
    else:
        epilogue_load_fragments = []
        epilogue_tile_decls = []
        y_tile_for = {}
        cond_tile_for = {}

    # Build config — matmul's c_buffer_name must be the final output of
    # the fused chain (not the anchor's `out` if there's an epilogue).
    config = MatmulConfig(
        tile=tile,
        # Aligned fast path (MatmulAccumToDevFragment) is incompatible
        # with a tg-tile epilogue (we need C_tile to apply the chain).
        aligned=aligned and not use_tg_tile_epilogue,
        M_dim_var="M",
        K_dim_var="K",
        N_dim_var="N",
        a_buffer_name=anchor.a.buffer_key,
        b_buffer_name=anchor.b.buffer_key,
        c_buffer_name=final_out_key,
        a_tile_name="A_tile",
        b_tile_name="B_tile",
        c_tile_name="C_tile",
    )

    # Register epilogue value-transform (lane-agnostic chain only).
    register_epilogue: tuple = ()
    if use_register:
        # Each elem is unary or has a Scalar y. chain_value_transform on
        # the "lane element" expression gives us a callable; tile-coord
        # references aren't used because the chain is lane-agnostic.
        rt = chain_value_transform(
            decision.epilogue,
            row_var="0",  # unused — lane-agnostic chain ignores these
            col_var="0",
            extras=extras,
            base_slot=base_slots,
        )
        register_epilogue = (
            MatmulRegisterEpilogueFragment(config=config, value_transform=rt),
        )

    main_row_limit: str | None = None if aligned else "M"
    main_k_limit: str | None = None if aligned else "K"
    main_n_limit: str | None = None if aligned else "N"

    # Strides come straight off the operand Tensor metadata, so a
    # transposed or reshaped view feeds the right offsets into the
    # cooperative loads. Literal ints (vs. the K/N dim vars) are fine —
    # the matmul is already recompiled per shape via `select_tile_config`.
    a_load = TgLoadFragment(
        name="load_A_tile",
        src_name=src_a.buffer_key,
        src_row_stride=str(src_a.row_stride),
        src_col_stride=str(src_a.col_stride),
        row_start=f"tg.y * {tile.tile_M}",
        col_start="k_chunk",
        dst_name="A_tile",
        tile_shape=(tile.tile_M, tile.tile_K),
        num_threads=tile.num_threads,
        row_limit=main_row_limit,
        col_limit=main_k_limit,
        value_transform=a_transform,
    )
    b_load = TgLoadFragment(
        name="load_B_tile",
        src_name=src_b.buffer_key,
        src_row_stride=str(src_b.row_stride),
        src_col_stride=str(src_b.col_stride),
        row_start="k_chunk",
        col_start=f"tg.x * {tile.tile_N}",
        dst_name="B_tile",
        tile_shape=(tile.tile_K, tile.tile_N),
        num_threads=tile.num_threads,
        row_limit=main_k_limit,
        col_limit=main_n_limit,
        value_transform=b_transform,
    )

    mainloop = MatmulMainloopFragment(
        tile_K=tile.tile_K,
        fragments=(
            a_load,
            b_load,
            BarrierFragment("inputs_ready"),
            MatmulComputeFragment(config),
            BarrierFragment("compute_done"),
        ),
        K_dim_var="K",
    )

    store_fragments: list
    if use_tg_tile_epilogue:
        store_fragments = [
            MatmulAccumToTgFragment(config),
            BarrierFragment("c_tile_ready"),
            *epilogue_load_fragments,
            BarrierFragment("epilogue_inputs_ready"),
            TiledElementwiseChainFragment(
                chain=decision.epilogue,
                primary_tile="C_tile",
                tile_shape=(tile.tile_M, tile.tile_N),
                num_threads=tile.num_threads,
                y_tile_for=y_tile_for,
                cond_tile_for=cond_tile_for,
            ),
            BarrierFragment("epilogue_done"),
            MatmulTgToDevFragment(config),
        ]
    elif use_register:
        # Aligned fast path may still apply when shape allows. The
        # register epilogue is applied to the accumulators before
        # writing to device.
        if config.aligned:
            store_fragments = [*register_epilogue, MatmulAccumToDevFragment(config)]
        else:
            store_fragments = [
                *register_epilogue,
                MatmulAccumToTgFragment(config),
                BarrierFragment("c_tile_ready"),
                MatmulTgToDevFragment(config),
            ]
    else:
        if config.aligned:
            store_fragments = [MatmulAccumToDevFragment(config)]
        else:
            store_fragments = [
                MatmulAccumToTgFragment(config),
                BarrierFragment("c_tile_ready"),
                MatmulTgToDevFragment(config),
            ]

    fragments = (
        MatmulTileMappingFragment(),
        ThreadIndexFragment(tile),
        MatmulSetupFragment(config),
        mainloop,
        *store_fragments,
    )

    tg_decls = [
        f"threadgroup float A_tile[{tile.tile_M}][{tile.tile_K + tile.a_pad}];",
        f"threadgroup float B_tile[{tile.tile_K}][{tile.tile_N + tile.b_pad}];",
    ]
    if not config.aligned:
        tg_decls.append(
            f"threadgroup float C_tile[{tile.tile_M}][{tile.tile_N + tile.c_pad}];"
        )
    tg_decls.extend(epilogue_tile_decls)

    base_buffers = [
        f"device const float* {k} [[buffer({slot})]]"
        for k, slot in input_key_to_slot.items()
    ] + [f"device float* {final_out_key} [[buffer({out_slot})]]"]
    fn_name = function_name or f"matmul_{anchor.out.name}_fused"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=tuple(base_buffers + extras.buffers),
        dims=("M", "K", "N"),
        tg_x=tile.tg_x,
        tg_y=tile.tg_y,
        threadgroup_decls=tuple(tg_decls),
    )
    kernel = Kernel(fragments=fragments, ctx=ctx)
    bindings = (
        *input_key_to_slot.keys(),
        final_out_key,
        *extras.bindings,
    )
    return KernelGroup(
        kernel=kernel,
        bindings=bindings,
        dims=(M, K, N),
        grid=grid_for(M, N, tile),
        threads=(tile.tg_x, tile.tg_y, 1),
        ops=decision.ops,
        strategy=decision.strategy,
    )
