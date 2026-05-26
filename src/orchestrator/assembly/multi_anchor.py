"""Multi-anchor strategy: two matmul anchors + one merge elem.

Shape:
  - 2 anchors share output (M, N). Per-anchor K may differ
    (different inner k-loop bounds).
  - Each anchor accumulates into its own simdgroup matC set
    (suffixed `_0` / `_1`), then drains into its own threadgroup
    C tile (`C0_tile`, `C1_tile`).
  - The merge elem is applied via `TiledElementwiseChainFragment`
    with `primary_tile=C0_tile` and `y_tile_for[0]=C1_tile` — the
    same fragment that powers epilogue tg-tile chains. The merged
    result lands back in `C0_tile`, which then ships to device.

Diamond path (`shared_a=True`): both anchors load the same A from
the same source with matching K. One shared mainloop emits one
A-load plus two B-loads + two computes per k-chunk, amortising
the A-side bandwidth. Otherwise two sequential mainloops.

Tg-memory budget is ignored here (caps mainloop-time live tiles
at 2×A + 2×B + 2×C, which OOMs on the 32KB Apple Silicon floor
for typical tile shapes — the liveness/aliasing pass is the
designated cleanup).
"""

from __future__ import annotations

from compute.fragments import BarrierFragment, CodegenContext
from compute.matmul import (
    MatmulAccumToTgFragment,
    MatmulComputeFragment,
    MatmulConfig,
    MatmulMainloopFragment,
    MatmulSetupFragment,
    MatmulTileMappingFragment,
    ThreadIndexFragment,
)
from compute.matmul.config import grid_for, select_tile_config
from memory.memory import TgLoadFragment, TgStoreFragment
from orchestrator.kernel_group import KernelGroup
from runtime.program import Kernel

from compute.elementwise.tiled_chain import TiledElementwiseChainFragment
from orchestrator.ir import Tensor

from orchestrator.assembly.decision import DecisionView
from orchestrator.assembly.expressions import chain_value_transform
from orchestrator.assembly.extras import Extras


def assemble_multi_anchor(
    decision: DecisionView, function_name: str | None
) -> KernelGroup:
    anchors = decision.anchors
    merge_elem = decision.merge_elem
    assert anchors is not None and len(anchors) == 2
    assert merge_elem is not None

    a0, a1 = anchors
    M = a0.out.shape[0]
    N = a0.out.shape[1]
    assert a1.out.shape == (M, N), "multi-anchor: anchors must share output shape"
    K0 = a0.a.shape[1]
    K1 = a1.a.shape[1]
    shared_a = decision.shared_a and K0 == K1

    # Per-anchor prologue chains (one per anchor, per side). Empty tuple
    # when that side has no absorbed prologue.
    prologues_a = decision.prologues_a or ((), ())
    prologues_b = decision.prologues_b or ((), ())

    # Effective device-input tensors per anchor: when a prologue is
    # absorbed, the actual buffer/strides come from the chain's
    # outermost primary, not from the anchor's named operand (which has
    # no materialized buffer after fusion).
    src_a = tuple(
        prologues_a[i][0].operands[0] if prologues_a[i] else anchors[i].a
        for i in range(2)
    )
    src_b = tuple(
        prologues_b[i][0].operands[0] if prologues_b[i] else anchors[i].b
        for i in range(2)
    )
    for t in (*src_a, *src_b):
        assert isinstance(t, Tensor)

    final_out = merge_elem.out
    final_out_key = final_out.buffer_key

    # Tile config — same for both anchors (they share output (M, N)).
    # Pick the dominant anchor (larger K wins; more cycles, better tile
    # bias).
    tile = select_tile_config(M, max(K0, K1), N)

    # Dedup input buffer keys: when two anchor operands point at the same
    # storage (X@X, B0==B1, A0 cross-aliased with B1, aliased views like
    # reshape) we'd otherwise emit duplicate `[[buffer(N)]]` params with
    # the same MSL name and hit a redefinition error. Each loader already
    # references its operand by `buffer_key`, so collapsing to one param
    # per unique key is correct — the two loaders read the same buffer.
    bindings: tuple[str, ...]
    dims_names: tuple[str, ...]
    dims_values: tuple[int, ...]
    input_keys: list[str] = []
    candidates = (
        (src_a[0], src_b[0], src_b[1])
        if shared_a
        else (src_a[0], src_b[0], src_a[1], src_b[1])
    )
    for t in candidates:
        if t.buffer_key not in input_keys:
            input_keys.append(t.buffer_key)
    out_slot = len(input_keys)
    base_slots = out_slot + 1
    base_buffers = [
        f"device const float* {k} [[buffer({i})]]" for i, k in enumerate(input_keys)
    ] + [f"device float* {final_out_key} [[buffer({out_slot})]]"]

    # Prologue chains may pull in additional tensor operands (chain
    # secondaries: biases, broadcast vectors). Extras tracks them as
    # extra buffer slots after the deduped base set.
    extras = Extras(
        base_slots_by_key={
            **{k: i for i, k in enumerate(input_keys)},
            final_out_key: out_slot,
        }
    )

    # Per-anchor value transforms — build once per side. Empty prologue =
    # None (loader keeps its identity transform). Chain secondaries are
    # registered into `extras` as a side-effect.
    transforms_a = tuple(
        chain_value_transform(
            prologues_a[i],
            row_var="global_row",
            col_var="global_col",
            extras=extras,
            base_slot=base_slots,
        )
        if prologues_a[i]
        else None
        for i in range(2)
    )
    transforms_b = tuple(
        chain_value_transform(
            prologues_b[i],
            row_var="global_row",
            col_var="global_col",
            extras=extras,
            base_slot=base_slots,
        )
        if prologues_b[i]
        else None
        for i in range(2)
    )

    bindings = (*input_keys, final_out_key, *extras.bindings)
    if shared_a:
        dims_names = ("M", "K", "N")
        dims_values = (M, K0, N)  # K0 == K1
    else:
        dims_names = ("M", "K0", "K1", "N")
        dims_values = (M, K0, K1, N)

    # Per-anchor configs (c_buffer_name is unused here since we always
    # route through C tiles).
    configs = tuple(
        MatmulConfig(
            tile=tile,
            aligned=False,
            M_dim_var="M",
            K_dim_var="K" if shared_a else f"K{i}",
            N_dim_var="N",
            a_buffer_name=src_a[i].buffer_key,
            b_buffer_name=src_b[i].buffer_key,
            c_buffer_name=final_out_key,
            a_tile_name="A_tile" if shared_a else f"A{i}_tile",
            b_tile_name=f"B{i}_tile",
            c_tile_name=f"C{i}_tile",
        )
        for i in range(2)
    )

    suffixes = ("_0", "_1")
    setup_frags = [
        MatmulTileMappingFragment(),
        ThreadIndexFragment(tile),
        MatmulSetupFragment(
            configs[0], accumulator_suffix=suffixes[0], emit_thread_coords=True
        ),
        MatmulSetupFragment(
            configs[1], accumulator_suffix=suffixes[1], emit_thread_coords=False
        ),
    ]

    def _b_load(i: int) -> TgLoadFragment:
        return TgLoadFragment(
            name=f"load_B{i}_tile",
            src_name=src_b[i].buffer_key,
            src_row_stride=str(src_b[i].row_stride),
            src_col_stride=str(src_b[i].col_stride),
            row_start="k_chunk",
            col_start=f"tg.x * {tile.tile_N}",
            dst_name=f"B{i}_tile",
            tile_shape=(tile.tile_K, tile.tile_N),
            num_threads=tile.num_threads,
            row_limit=configs[i].K_dim_var,
            col_limit="N",
            value_transform=transforms_b[i],
        )

    if shared_a:
        # Single mainloop: one shared A-load + two B-loads + two computes.
        # `shared_a` is only set when neither anchor has an A prologue, so
        # no value_transform is needed on the shared A-load.
        shared_a_load = TgLoadFragment(
            name="load_A_tile",
            src_name=src_a[0].buffer_key,
            src_row_stride=str(src_a[0].row_stride),
            src_col_stride=str(src_a[0].col_stride),
            row_start=f"tg.y * {tile.tile_M}",
            col_start="k_chunk",
            dst_name="A_tile",
            tile_shape=(tile.tile_M, tile.tile_K),
            num_threads=tile.num_threads,
            row_limit="M",
            col_limit="K",
        )
        mainloops = [
            MatmulMainloopFragment(
                tile_K=tile.tile_K,
                fragments=(
                    shared_a_load,
                    _b_load(0),
                    _b_load(1),
                    BarrierFragment("inputs_ready"),
                    MatmulComputeFragment(configs[0], accumulator_suffix=suffixes[0]),
                    MatmulComputeFragment(configs[1], accumulator_suffix=suffixes[1]),
                    BarrierFragment("compute_done"),
                ),
                K_dim_var="K",
            )
        ]
    else:
        # Two sequential mainloops, each with its own A and B loads.
        mainloops = []
        for i in range(2):
            a_load = TgLoadFragment(
                name=f"load_A{i}_tile",
                src_name=src_a[i].buffer_key,
                src_row_stride=str(src_a[i].row_stride),
                src_col_stride=str(src_a[i].col_stride),
                row_start=f"tg.y * {tile.tile_M}",
                col_start="k_chunk",
                dst_name=f"A{i}_tile",
                tile_shape=(tile.tile_M, tile.tile_K),
                num_threads=tile.num_threads,
                row_limit="M",
                col_limit=configs[i].K_dim_var,
                value_transform=transforms_a[i],
            )
            mainloops.append(
                MatmulMainloopFragment(
                    tile_K=tile.tile_K,
                    fragments=(
                        a_load,
                        _b_load(i),
                        BarrierFragment(f"inputs{i}_ready"),
                        MatmulComputeFragment(
                            configs[i], accumulator_suffix=suffixes[i]
                        ),
                        BarrierFragment(f"compute{i}_done"),
                    ),
                    K_dim_var=configs[i].K_dim_var,
                )
            )

    # Drain accumulators → C tiles, merge, store.
    store_to_tg = [
        MatmulAccumToTgFragment(
            configs[0], accumulator_suffix=suffixes[0], emit_tg_coords=True
        ),
        MatmulAccumToTgFragment(
            configs[1], accumulator_suffix=suffixes[1], emit_tg_coords=False
        ),
    ]

    merge_frag = TiledElementwiseChainFragment(
        chain=(merge_elem,),
        primary_tile="C0_tile",
        tile_shape=(tile.tile_M, tile.tile_N),
        num_threads=tile.num_threads,
        y_tile_for={0: "C1_tile"},
    )

    store_out = TgStoreFragment(
        name="store_merged",
        src_name="C0_tile",
        dst_name=final_out_key,
        dst_row_stride="N",
        row_start=f"tg.y * {tile.tile_M}",
        col_start=f"tg.x * {tile.tile_N}",
        tile_shape=(tile.tile_M, tile.tile_N),
        num_threads=tile.num_threads,
        row_limit="M",
        col_limit="N",
    )

    fragments = (
        *setup_frags,
        *mainloops,
        *store_to_tg,
        BarrierFragment("c_tiles_ready"),
        merge_frag,
        BarrierFragment("merge_done"),
        store_out,
    )

    tg_decls: list[str] = []
    if shared_a:
        tg_decls.append(
            f"threadgroup float A_tile[{tile.tile_M}][{tile.tile_K + tile.a_pad}];"
        )
    else:
        for i in range(2):
            tg_decls.append(
                f"threadgroup float A{i}_tile[{tile.tile_M}][{tile.tile_K + tile.a_pad}];"
            )
    for i in range(2):
        tg_decls.append(
            f"threadgroup float B{i}_tile[{tile.tile_K}][{tile.tile_N + tile.b_pad}];"
        )
    for i in range(2):
        tg_decls.append(
            f"threadgroup float C{i}_tile[{tile.tile_M}][{tile.tile_N + tile.c_pad}];"
        )

    fn_name = function_name or f"multi_anchor_{final_out.name}_fused"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=tuple(base_buffers + extras.buffers),
        dims=dims_names,
        tg_x=tile.tg_x,
        tg_y=tile.tg_y,
        threadgroup_decls=tuple(tg_decls),
    )
    kernel = Kernel(fragments=fragments, ctx=ctx)
    return KernelGroup(
        kernel=kernel,
        bindings=bindings,
        dims=dims_values,
        grid=grid_for(M, N, tile),
        threads=(tile.tg_x, tile.tg_y, 1),
        ops=decision.ops,
        strategy=decision.strategy,
    )
