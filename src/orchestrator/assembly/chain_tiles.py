"""Set up Y/Cond threadgroup tiles for a tg-tile elementwise chain.

The matmul tg-tile epilogue path and the standalone elementwise
template both need to:
  1. Walk the chain in order.
  2. For each elem, register Y / Cond tensor operands not produced by
     an earlier chain element as device-buffer extras.
  3. Allocate a threadgroup tile per such operand and a TgLoadFragment
     that fills it from device.
  4. Return per-elem maps (`y_tile_for`, `cond_tile_for`) for
     `TiledElementwiseChainFragment` to dereference.

The two callers diverge only in the tile-name prefix and whether
they're in the aligned matmul fast-path (which suppresses row/col
limits). Everything else is shared and lives here.
"""

from __future__ import annotations

from compute.elementwise.elementwise import elementwise_arity
from memory.memory import TgLoadFragment
from orchestrator.ir import ElementwiseOp, Tensor

from orchestrator.assembly.extras import Extras
from orchestrator.assembly.load_geometry import broadcast_load_geometry


def build_chain_y_tile_loads(
    chain: tuple[ElementwiseOp, ...],
    *,
    seed_available: set[str],
    tile_M: int,
    tile_N: int,
    num_threads: int,
    extras: Extras,
    base_slots: int,
    y_prefix: str = "Y",
    cond_prefix: str = "Cond",
    aligned: bool = False,
) -> tuple[list[TgLoadFragment], list[str], dict[int, str], dict[int, str]]:
    """Returns (load_fragments, tile_decls, y_tile_for, cond_tile_for).

    `seed_available` is the set of value names already in scope before
    the chain runs (e.g. the anchor output for an epilogue, the primary
    tensor for a standalone elementwise chain). Chain outputs accrete
    as we iterate.
    """
    load_fragments: list[TgLoadFragment] = []
    tile_decls: list[str] = []
    y_tile_for: dict[int, str] = {}
    cond_tile_for: dict[int, str] = {}
    available_names = set(seed_available)

    for i, elem in enumerate(chain):
        for operand_index, broadcast, tile_for, prefix in (
            (1, elem.y_broadcast, y_tile_for, y_prefix),
            (2, elem.cond_broadcast, cond_tile_for, cond_prefix),
        ):
            if operand_index >= elementwise_arity(elem.op):
                continue
            operand = elem.operands[operand_index]
            if not isinstance(operand, Tensor) or operand.name in available_names:
                continue
            tile_name = f"{prefix}{i}_tile"
            tile_for[i] = tile_name
            extras.add_tensor(operand, base_slots)
            (
                row_start,
                col_start,
                tile_shape,
                row_limit,
                col_limit,
                row_stride,
                col_stride,
            ) = broadcast_load_geometry(broadcast, tile_M, tile_N, operand)
            tile_decls.append(
                f"threadgroup float {tile_name}[{tile_shape[0]}][{tile_shape[1]}];"
            )
            load_fragments.append(
                TgLoadFragment(
                    name=f"load_{tile_name}",
                    src_name=operand.buffer_key,
                    src_row_stride=row_stride,
                    src_col_stride=col_stride,
                    row_start=row_start,
                    col_start=col_start,
                    dst_name=tile_name,
                    tile_shape=tile_shape,
                    num_threads=num_threads,
                    row_limit=None if aligned else row_limit,
                    col_limit=None if aligned else col_limit,
                )
            )
        available_names.add(elem.out.name)

    return load_fragments, tile_decls, y_tile_for, cond_tile_for
