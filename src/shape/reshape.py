from __future__ import annotations

from dataclasses import dataclass

from compute.fragments import CodegenContext


@dataclass(frozen=True)
class ReshapeFragment:
    """Physical reshape: gather elements from `src_name` in row-major flat order
    and write them contiguously to `dst_name` in the new shape.

    Source is interpreted as 2D `(src_rows_dim, src_cols_dim)` with strides
    `src_row_stride` / `src_col_stride` (defaults to row-major contiguous).
    Destination is 2D `(dst_rows_dim, dst_cols_dim)` written contiguously
    row-major. The caller guarantees `src_rows * src_cols == dst_rows * dst_cols`
    and passes that product as `total_dim`.

    Reshape preserves the row-major flat ordering of elements, so output flat
    index `gid` reads source flat index `gid`; that flat index is decoded into
    source 2D coords using the source column count and addressed with the source
    strides. When the source is contiguous (`src_col_stride == "1"` and
    `src_row_stride == src_cols_dim`) the MSL compiler folds the indexing back
    to a flat copy.
    """

    src_name: str
    dst_name: str
    src_cols_dim: str
    total_dim: str
    src_row_stride: str
    src_col_stride: str = "1"
    name: str = "reshape"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        gid = f"tg.x * {ctx.tg_x} + flat_tid"
        col_term = (
            "src_col"
            if self.src_col_stride == "1"
            else f"src_col * ({self.src_col_stride})"
        )
        return f"""\
uint gid = {gid};
if (gid < {self.total_dim}) {{
    uint src_row = gid / ({self.src_cols_dim});
    uint src_col = gid % ({self.src_cols_dim});
    {self.dst_name}[gid] = {self.src_name}[src_row * ({self.src_row_stride}) + {col_term}];
}}"""
