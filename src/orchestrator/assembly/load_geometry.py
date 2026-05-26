"""Unified broadcast-load geometry for tg-tile chain operands.

Both the matmul epilogue tile loader and the standalone elementwise
template need the same (row_start, col_start, tile_shape, row_limit,
col_limit, row_stride, col_stride) tuple per broadcast spec, derived
from the tile shape. The previous two near-identical functions are
collapsed here.

The dim-var names ("M" / "N" / "1") are baked in: every caller uses
the same canonical names in its CodegenContext.
"""

from __future__ import annotations

from compute.elementwise.elementwise import BroadcastSpec
from orchestrator.ir import Tensor


LoadGeometry = tuple[str, str, tuple[int, int], str, str, str, str]


def broadcast_load_geometry(
    bc: BroadcastSpec,
    tile_M: int,
    tile_N: int,
    operand: Tensor,
) -> LoadGeometry:
    """Returns (row_start, col_start, tile_shape, row_limit, col_limit,
    row_stride, col_stride)."""
    rs = str(operand.row_stride)
    cs = str(operand.col_stride)
    if bc is BroadcastSpec.NONE:
        return (
            f"tg.y * {tile_M}",
            f"tg.x * {tile_N}",
            (tile_M, tile_N),
            "M",
            "N",
            rs,
            cs,
        )
    if bc is BroadcastSpec.ROW:
        return ("0", f"tg.x * {tile_N}", (1, tile_N), "1", "N", rs, cs)
    if bc is BroadcastSpec.COL:
        return (f"tg.y * {tile_M}", "0", (tile_M, 1), "M", "1", rs, cs)
    if bc is BroadcastSpec.SCALAR:
        return ("0", "0", (1, 1), "1", "1", rs, cs)
    raise ValueError(f"unsupported broadcast {bc!r}")
