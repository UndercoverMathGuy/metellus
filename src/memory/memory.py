import textwrap
from dataclasses import dataclass
from typing import Callable

from compute.fragments import CodegenContext, TgmemAccess


ValueTransform = Callable[[str], str]
"""Wraps a raw scalar load expression into a transformed expression. Applied
only to in-bounds reads — out-of-bounds reads stay 0.0f so they remain matmul
identity / reduction identity for downstream fusion consumers."""


def _identity_transform(expr: str) -> str:
    return expr


def _dedent(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def _validate_threads(num_threads: int) -> None:
    if num_threads % 32 != 0:
        raise ValueError(
            f"num_threads must be a multiple of 32 (Apple SIMD width); got {num_threads}"
        )


def _validate_vector_width(cols: int, vector_width: int) -> None:
    if vector_width not in (1, 4):
        raise ValueError(f"vector_width must be 1 or 4; got {vector_width}")
    if cols % vector_width != 0:
        raise ValueError(f"Tile cols must be divisible by {vector_width}")


def _cooperative_load(
    src_name: str,
    src_row_stride: str,
    row_start: str,
    col_start: str,
    dst_name: str,
    tile_shape: tuple[int, int],
    num_threads: int,
    row_limit: str | None = None,
    col_limit: str | None = None,
    vector_width: int = 1,
    aligned: bool = False,
    src_col_stride: str = "1",
    value_transform: ValueTransform | None = None,
) -> str:
    rows, cols = tile_shape
    _validate_threads(num_threads)
    _validate_vector_width(cols, vector_width)
    transform = value_transform or _identity_transform

    if vector_width == 1:
        return _scalar_load(
            src_name,
            src_row_stride,
            row_start,
            col_start,
            dst_name,
            rows,
            cols,
            num_threads,
            row_limit,
            col_limit,
            src_col_stride,
            transform,
        )

    return _float4_load(
        src_name,
        src_row_stride,
        row_start,
        col_start,
        dst_name,
        rows,
        cols,
        num_threads,
        row_limit,
        col_limit,
        aligned,
        transform,
    )


def _cooperative_store(
    src_name: str,
    dst_name: str,
    dst_row_stride: str,
    row_start: str,
    col_start: str,
    tile_shape: tuple[int, int],
    num_threads: int,
    row_limit: str | None = None,
    col_limit: str | None = None,
    vector_width: int = 1,
    aligned: bool = False,
    dst_col_stride: str = "1",
) -> str:
    rows, cols = tile_shape
    _validate_threads(num_threads)
    _validate_vector_width(cols, vector_width)

    if vector_width == 1:
        return _scalar_store(
            src_name,
            dst_name,
            dst_row_stride,
            row_start,
            col_start,
            rows,
            cols,
            num_threads,
            row_limit,
            col_limit,
            dst_col_stride,
        )

    return _float4_store(
        src_name,
        dst_name,
        dst_row_stride,
        row_start,
        col_start,
        rows,
        cols,
        num_threads,
        row_limit,
        col_limit,
        aligned,
    )


def _scalar_load(
    src_name: str,
    src_row_stride: str,
    row_start: str,
    col_start: str,
    dst_name: str,
    rows: int,
    cols: int,
    num_threads: int,
    row_limit: str | None,
    col_limit: str | None,
    src_col_stride: str = "1",
    transform: ValueTransform = _identity_transform,
) -> str:
    num_elements = rows * cols
    col_term = (
        "global_col" if src_col_stride == "1" else f"global_col * ({src_col_stride})"
    )
    raw = f"{src_name}[global_row * ({src_row_stride}) + {col_term}]"
    in_bounds_value = transform(raw)
    if row_limit is not None and col_limit is not None:
        value = f"(global_row < ({row_limit}) && global_col < ({col_limit})) ? ({in_bounds_value}) : 0.0f"
    else:
        value = in_bounds_value

    return _dedent(f"""
    for (uint idx = flat_tid; idx < {num_elements}; idx += {num_threads}) {{
        uint tile_row = idx / {cols};
        uint tile_col = idx % {cols};
        uint global_row = ({row_start}) + tile_row;
        uint global_col = ({col_start}) + tile_col;
        {dst_name}[tile_row][tile_col] = {value};
    }}
    """)


def _scalar_store(
    src_name: str,
    dst_name: str,
    dst_row_stride: str,
    row_start: str,
    col_start: str,
    rows: int,
    cols: int,
    num_threads: int,
    row_limit: str | None,
    col_limit: str | None,
    dst_col_stride: str = "1",
) -> str:
    num_elements = rows * cols
    col_term = (
        "global_col" if dst_col_stride == "1" else f"global_col * ({dst_col_stride})"
    )
    store = f"{dst_name}[global_row * ({dst_row_stride}) + {col_term}] = {src_name}[tile_row][tile_col];"
    if row_limit is not None and col_limit is not None:
        store = f"if (global_row < ({row_limit}) && global_col < ({col_limit})) {{\n        {store}\n    }}"

    return _dedent(f"""
    for (uint idx = flat_tid; idx < {num_elements}; idx += {num_threads}) {{
        uint tile_row = idx / {cols};
        uint tile_col = idx % {cols};
        uint global_row = ({row_start}) + tile_row;
        uint global_col = ({col_start}) + tile_col;
        {store}
    }}
    """)


def _float4_load(
    src_name: str,
    src_row_stride: str,
    row_start: str,
    col_start: str,
    dst_name: str,
    rows: int,
    cols: int,
    num_threads: int,
    row_limit: str | None,
    col_limit: str | None,
    aligned: bool,
    transform: ValueTransform = _identity_transform,
) -> str:
    cols_f4 = cols // 4
    num_f4 = rows * cols_f4
    address = f"{src_name}[global_row * ({src_row_stride}) + global_col]"

    # When transform is identity we keep the original (simpler) emission so
    # existing code paths are byte-identical.
    has_transform = transform is not _identity_transform

    # `transform` was built with col_var="global_col", so its emitted MSL
    # reads secondary operands at whatever the in-scope `global_col` is.
    # For lanes y/z/w we mutate `global_col` in place between assignments
    # so each transform call sees the correct per-lane column. (Nothing
    # after load_body reads global_col — stores use tile_col.)
    if aligned:
        if has_transform:
            load_body = _dedent(f"""
            float4 v = *((device const float4*)&{address});
            v.x = {transform("v.x")};
            global_col += 1; v.y = {transform("v.y")};
            global_col += 1; v.z = {transform("v.z")};
            global_col += 1; v.w = {transform("v.w")};
            """).strip()
        else:
            load_body = f"float4 v = *((device const float4*)&{address});"
    else:
        # In-bounds path applies the transform to each component; OOB lanes
        # stay 0.0f.
        if has_transform:
            bulk_load = _dedent(f"""
            v = *((device const float4*)&{address});
            v.x = {transform("v.x")};
            global_col += 1; v.y = {transform("v.y")};
            global_col += 1; v.z = {transform("v.z")};
            global_col += 1; v.w = {transform("v.w")};
            """).strip()
            per_lane = transform(
                f"{src_name}[global_row * ({src_row_stride}) + global_col]"
            )
            per_lane_body = _dedent(f"""
            v.x = (global_col < ({col_limit})) ? ({per_lane}) : 0.0f;
            global_col += 1;
            v.y = (global_col < ({col_limit})) ? ({per_lane}) : 0.0f;
            global_col += 1;
            v.z = (global_col < ({col_limit})) ? ({per_lane}) : 0.0f;
            global_col += 1;
            v.w = (global_col < ({col_limit})) ? ({per_lane}) : 0.0f;
            """).strip()
        else:
            bulk_load = f"v = *((device const float4*)&{address});"
            per_lane_x = f"{src_name}[global_row * ({src_row_stride}) + global_col]"
            per_lane_y = f"{src_name}[global_row * ({src_row_stride}) + global_col + 1]"
            per_lane_z = f"{src_name}[global_row * ({src_row_stride}) + global_col + 2]"
            per_lane_w = f"{src_name}[global_row * ({src_row_stride}) + global_col + 3]"
            per_lane_body = _dedent(f"""
            v.x = (global_col < ({col_limit})) ? ({per_lane_x}) : 0.0f;
            v.y = (global_col + 1 < ({col_limit})) ? ({per_lane_y}) : 0.0f;
            v.z = (global_col + 2 < ({col_limit})) ? ({per_lane_z}) : 0.0f;
            v.w = (global_col + 3 < ({col_limit})) ? ({per_lane_w}) : 0.0f;
            """).strip()
        load_body = _dedent(f"""
        float4 v = float4(0.0f);
        if (global_row < ({row_limit})) {{
            if (global_col + 3 < ({col_limit}) && ((global_row * ({src_row_stride}) + global_col) % 4 == 0)) {{
                {bulk_load}
            }} else {{
                {per_lane_body}
            }}
        }}
        """).strip()

    return _dedent(f"""
    for (uint idx = flat_tid; idx < {num_f4}; idx += {num_threads}) {{
        uint tile_row = idx / {cols_f4};
        uint tile_f4_col = idx % {cols_f4};
        uint tile_col = tile_f4_col * 4;
        uint global_row = ({row_start}) + tile_row;
        uint global_col = ({col_start}) + tile_col;
        {load_body}
        {dst_name}[tile_row][tile_col] = v.x;
        {dst_name}[tile_row][tile_col + 1] = v.y;
        {dst_name}[tile_row][tile_col + 2] = v.z;
        {dst_name}[tile_row][tile_col + 3] = v.w;
    }}
    """)


def _float4_store(
    src_name: str,
    dst_name: str,
    dst_row_stride: str,
    row_start: str,
    col_start: str,
    rows: int,
    cols: int,
    num_threads: int,
    row_limit: str | None,
    col_limit: str | None,
    aligned: bool,
) -> str:
    cols_f4 = cols // 4
    num_f4 = rows * cols_f4
    vector = f"float4({src_name}[tile_row][tile_col], {src_name}[tile_row][tile_col + 1], {src_name}[tile_row][tile_col + 2], {src_name}[tile_row][tile_col + 3])"
    address = f"{dst_name}[global_row * ({dst_row_stride}) + global_col]"

    if aligned:
        store_body = f"*((device float4*)&{address}) = v;"
    else:
        store_body = _dedent(f"""
        if (global_row < ({row_limit})) {{
            if (global_col + 3 < ({col_limit}) && ((global_row * ({dst_row_stride}) + global_col) % 4 == 0)) {{
                *((device float4*)&{address}) = v;
            }} else {{
                if (global_col < ({col_limit})) {{
                    {dst_name}[global_row * ({dst_row_stride}) + global_col] = {src_name}[tile_row][tile_col];
                }}
                if (global_col + 1 < ({col_limit})) {{
                    {dst_name}[global_row * ({dst_row_stride}) + global_col + 1] = {src_name}[tile_row][tile_col + 1];
                }}
                if (global_col + 2 < ({col_limit})) {{
                    {dst_name}[global_row * ({dst_row_stride}) + global_col + 2] = {src_name}[tile_row][tile_col + 2];
                }}
                if (global_col + 3 < ({col_limit})) {{
                    {dst_name}[global_row * ({dst_row_stride}) + global_col + 3] = {src_name}[tile_row][tile_col + 3];
                }}
            }}
        }}
        """).strip()

    return _dedent(f"""
    for (uint idx = flat_tid; idx < {num_f4}; idx += {num_threads}) {{
        uint tile_row = idx / {cols_f4};
        uint tile_f4_col = idx % {cols_f4};
        uint tile_col = tile_f4_col * 4;
        uint global_row = ({row_start}) + tile_row;
        uint global_col = ({col_start}) + tile_col;
        float4 v = {vector};
        {store_body}
    }}
    """)


def tg_load(
    src_name: str,
    src_row_stride: str,
    row_start: str,
    col_start: str,
    dst_name: str,
    tile_shape: tuple[int, int],
    num_threads: int,
    row_limit: str | None = None,
    col_limit: str | None = None,
    src_col_stride: str = "1",
    value_transform: ValueTransform | None = None,
) -> str:
    """Unified threadgroup load. Auto-selects float4 when tile cols divisible by 4
    and the column stride is the literal "1" (vectorized loads require contiguity).
    If row_limit and col_limit are None, emits aligned (no bounds check) code.
    `value_transform`, when supplied, wraps each in-bounds scalar read — used
    by the orchestrator to inline elementwise transforms on matmul/reduction
    input loads. Out-of-bounds reads stay 0.0f (matmul / reduction identity)."""
    _, cols = tile_shape
    contiguous = src_col_stride == "1"
    vector_width = 4 if cols % 4 == 0 and contiguous else 1
    aligned = row_limit is None and col_limit is None
    return _cooperative_load(
        src_name,
        src_row_stride,
        row_start,
        col_start,
        dst_name,
        tile_shape,
        num_threads,
        row_limit,
        col_limit,
        vector_width=vector_width,
        aligned=aligned,
        src_col_stride=src_col_stride,
        value_transform=value_transform,
    )


def tg_store(
    src_name: str,
    dst_name: str,
    dst_row_stride: str,
    row_start: str,
    col_start: str,
    tile_shape: tuple[int, int],
    num_threads: int,
    row_limit: str | None = None,
    col_limit: str | None = None,
    dst_col_stride: str = "1",
) -> str:
    """Unified threadgroup store. Auto-selects float4 when tile cols divisible by 4
    and the column stride is the literal "1" (vectorized stores require contiguity).
    If row_limit and col_limit are None, emits aligned (no bounds check) code."""
    _, cols = tile_shape
    contiguous = dst_col_stride == "1"
    vector_width = 4 if cols % 4 == 0 and contiguous else 1
    aligned = row_limit is None and col_limit is None
    return _cooperative_store(
        src_name,
        dst_name,
        dst_row_stride,
        row_start,
        col_start,
        tile_shape,
        num_threads,
        row_limit,
        col_limit,
        vector_width=vector_width,
        aligned=aligned,
        dst_col_stride=dst_col_stride,
    )


@dataclass(frozen=True)
class TgLoadFragment:
    name: str
    src_name: str
    src_row_stride: str
    row_start: str
    col_start: str
    dst_name: str
    tile_shape: tuple[int, int]
    num_threads: int
    row_limit: str | None = None
    col_limit: str | None = None
    src_col_stride: str = "1"
    value_transform: ValueTransform | None = None
    kind: str = "memory"

    def render(self, ctx: CodegenContext) -> str:
        return tg_load(
            src_name=self.src_name,
            src_row_stride=self.src_row_stride,
            row_start=self.row_start,
            col_start=self.col_start,
            dst_name=self.dst_name,
            tile_shape=self.tile_shape,
            num_threads=self.num_threads,
            row_limit=self.row_limit,
            col_limit=self.col_limit,
            src_col_stride=self.src_col_stride,
            value_transform=self.value_transform,
        )

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        rows, cols = self.tile_shape
        return (
            TgmemAccess(
                name=self.dst_name,
                access="write",
                size_floats=rows * cols,
                shape=self.tile_shape,
            ),
        )


@dataclass(frozen=True)
class TgStoreFragment:
    name: str
    src_name: str
    dst_name: str
    dst_row_stride: str
    row_start: str
    col_start: str
    tile_shape: tuple[int, int]
    num_threads: int
    row_limit: str | None = None
    col_limit: str | None = None
    dst_col_stride: str = "1"
    kind: str = "store"

    def render(self, ctx: CodegenContext) -> str:
        return tg_store(
            src_name=self.src_name,
            dst_name=self.dst_name,
            dst_row_stride=self.dst_row_stride,
            row_start=self.row_start,
            col_start=self.col_start,
            tile_shape=self.tile_shape,
            num_threads=self.num_threads,
            row_limit=self.row_limit,
            col_limit=self.col_limit,
            dst_col_stride=self.dst_col_stride,
        )

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        rows, cols = self.tile_shape
        return (
            TgmemAccess(
                name=self.src_name,
                access="read",
                size_floats=rows * cols,
                shape=self.tile_shape,
            ),
        )
