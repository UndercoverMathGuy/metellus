from dataclasses import dataclass

from typing import Any

from compute.fragments import CodegenContext
from compute.matmul.config import (
    SplitKConfig,
    TileConfig,
)
from memory.memory import TgStoreFragment, ValueTransform

KernelFragment = Any


@dataclass(frozen=True)
class MatmulConfig:
    tile: TileConfig
    aligned: bool
    M_dim_var: str
    K_dim_var: str
    N_dim_var: str
    a_buffer_name: str
    b_buffer_name: str
    c_buffer_name: str
    a_tile_name: str
    b_tile_name: str
    c_tile_name: str
    c_col_stride: str = "1"

    def validate(self) -> None:
        if self.tile.sg_M * self.tile.sg_N != self.tile.num_threads // 32:
            raise ValueError("sg_M * sg_N must equal num_threads/32")
        if self.aligned and self.c_col_stride != "1":
            raise ValueError(
                "aligned matmul store requires c_col_stride == '1' (simdgroup_store needs contiguous output)"
            )


@dataclass(frozen=True)
class MatmulTileMappingFragment:
    name: str = "tile_mapping"
    kind: str = "setup"

    def render(self, ctx: CodegenContext) -> str:
        return "uint2 tg = threadgroup_position_in_grid.xy;"


@dataclass(frozen=True)
class ThreadIndexFragment:
    tile: TileConfig
    name: str = "thread_index"
    kind: str = "setup"

    def render(self, ctx: CodegenContext) -> str:
        return f"""\
uint2 lid = thread_position_in_threadgroup.xy;
uint flat_tid = lid.y * {self.tile.tg_x} + lid.x;
uint sg_id = flat_tid / 32;"""


@dataclass(frozen=True)
class MatmulSetupFragment:
    config: MatmulConfig
    name: str = "matmul_setup"
    kind: str = "setup"

    def render(self, ctx: CodegenContext) -> str:
        tile = self.config.tile
        m_tiles, n_tiles = _simdgroup_tile_counts(tile)
        accum_decls = ", ".join(
            f"matC{i}{j}(0.0f)" for i in range(m_tiles) for j in range(n_tiles)
        )
        return f"""\
simdgroup_float8x8 {accum_decls};
uint sg_row = sg_id / {tile.sg_N};
uint sg_col = sg_id % {tile.sg_N};
uint a_row_origin = sg_row * {m_tiles * 8};
uint b_col_origin = sg_col * {n_tiles * 8};"""


@dataclass(frozen=True)
class MatmulMainloopFragment:
    tile_K: int
    fragments: tuple[KernelFragment, ...]
    K_dim_var: str
    name: str = "matmul_mainloop"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        body = "\n".join(fragment.render(ctx) for fragment in self.fragments)
        return f"""\
for (uint k_chunk = 0; k_chunk < {self.K_dim_var}; k_chunk += {self.tile_K}) {{
{_indent(body, 4)}
}}"""


@dataclass(frozen=True)
class MatmulComputeFragment:
    config: MatmulConfig
    name: str = "matmul_compute"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        tile = self.config.tile
        m_tiles, n_tiles = _simdgroup_tile_counts(tile)
        if tile.tile_K % 8 != 0:
            raise ValueError(f"tile_K ({tile.tile_K}) must be divisible by 8")
        matA_decls = ", ".join(f"matA{i}" for i in range(m_tiles))
        matB_decls = ", ".join(f"matB{j}" for j in range(n_tiles))
        matA_loads = "\n        ".join(
            f"simdgroup_load(matA{i}, &{self.config.a_tile_name}[a_row_origin + {i * 8}][k], {tile.tile_K + tile.a_pad});"
            for i in range(m_tiles)
        )
        matB_loads = "\n        ".join(
            f"simdgroup_load(matB{j}, &{self.config.b_tile_name}[k][b_col_origin + {j * 8}], {tile.tile_N + tile.b_pad});"
            for j in range(n_tiles)
        )
        muladds = "\n        ".join(
            f"simdgroup_multiply_accumulate(matC{i}{j}, matA{i}, matB{j}, matC{i}{j});"
            for i in range(m_tiles)
            for j in range(n_tiles)
        )
        return f"""\
for (uint k = 0; k < {tile.tile_K}; k += 8) {{
    simdgroup_float8x8 {matA_decls}, {matB_decls};
    {matA_loads}
    {matB_loads}
    {muladds}
}}"""


@dataclass(frozen=True)
class MatmulRegisterEpilogueFragment:
    """Apply elementwise ops to matmul accumulators while they're still in
    registers, via `simdgroup_float8x8::thread_elements()`. The transform is a
    Python callable that wraps each per-lane scalar expression (the two floats
    owned by this lane in each 8x8 accumulator block).

    Lane → (row, col) mapping inside an 8x8 simdgroup tile is
    implementation-defined in MSL, so transforms must be **lane-agnostic**:
    pure unary ops (relu, exp, ...) and scalar broadcasts. For ops that need
    per-element row/col (row/col bias, full-tile second operand) the
    orchestrator should fall back to the tg-tile epilogue (insert an
    ElementwiseComputeFragment between `MatmulAccumToTgFragment` and
    `MatmulTgToDevFragment` instead).

    `setup` is a tuple of extra MSL lines emitted before the per-tile bodies
    (e.g. declaring a broadcast scalar loaded once per threadgroup).

    With `value_transform=None` the fragment is a no-op — safe to always
    include in a template.
    """

    config: MatmulConfig
    value_transform: ValueTransform | None = None
    setup: tuple[str, ...] = ()
    name: str = "matmul_register_epilogue"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        if self.value_transform is None and not self.setup:
            return ""
        tile = self.config.tile
        m_tiles, n_tiles = _simdgroup_tile_counts(tile)
        lines = list(self.setup)
        if self.value_transform is not None:
            for i in range(m_tiles):
                for j in range(n_tiles):
                    elem = f"e{i}{j}"
                    # `thread_elements()` returns a per-lane reference into
                    # matC's distributed storage. Bind with `auto&` — plain
                    # `auto` decays to a copy and the writes below would
                    # land in a temporary, not matC.
                    lines.append(f"thread auto& {elem} = matC{i}{j}.thread_elements();")
                    lines.append(f"{elem}[0] = {self.value_transform(f'{elem}[0]')};")
                    lines.append(f"{elem}[1] = {self.value_transform(f'{elem}[1]')};")
        return "\n".join(lines)


@dataclass(frozen=True)
class MatmulAccumToDevFragment:
    """Aligned-shape fast path: simdgroup_store accumulators directly to the
    output device buffer, bypassing the C_tile in threadgroup memory. Only
    valid when `config.aligned` is True and no tg-tile epilogue is fused —
    when fusing a tg-tile epilogue, use `MatmulAccumToTgFragment` instead so
    there's a tile to operate on."""

    config: MatmulConfig
    name: str = "matmul_accum_to_dev"
    kind: str = "store"

    def render(self, ctx: CodegenContext) -> str:
        if not self.config.aligned:
            raise ValueError("MatmulAccumToDevFragment requires config.aligned=True")
        tile = self.config.tile
        m_tiles, n_tiles = _simdgroup_tile_counts(tile)
        stores = "\n".join(
            f"simdgroup_store(matC{i}{j}, &{self.config.c_buffer_name}[(c_row_origin + {i * 8}) * ({self.config.N_dim_var}) + (c_col_origin + {j * 8})], {self.config.N_dim_var});"
            for i in range(m_tiles)
            for j in range(n_tiles)
        )
        return f"""\
uint c_row_origin = (tg.y * {tile.tile_M}) + sg_row * {m_tiles * 8};
uint c_col_origin = (tg.x * {tile.tile_N}) + sg_col * {n_tiles * 8};
{stores}"""


@dataclass(frozen=True)
class MatmulAccumToTgFragment:
    """Store simdgroup accumulators into the threadgroup `C_tile`. Must be
    followed by a threadgroup barrier before any consumer reads `C_tile`."""

    config: MatmulConfig
    name: str = "matmul_accum_to_tg"
    kind: str = "store"

    def render(self, ctx: CodegenContext) -> str:
        tile = self.config.tile
        m_tiles, n_tiles = _simdgroup_tile_counts(tile)
        stores = "\n".join(
            f"simdgroup_store(matC{i}{j}, &{self.config.c_tile_name}[(c_tg_row_origin + {i * 8})][c_tg_col_origin + {j * 8}], {tile.tile_N + tile.c_pad});"
            for i in range(m_tiles)
            for j in range(n_tiles)
        )
        return f"""\
uint c_tg_row_origin = sg_row * {m_tiles * 8};
uint c_tg_col_origin = sg_col * {n_tiles * 8};
{stores}"""


@dataclass(frozen=True)
class MatmulTgToDevFragment:
    """Cooperative tg_store of `C_tile` to the C device buffer with
    out-of-bounds masking. Used in the unaligned path and whenever a tg-tile
    epilogue is fused."""

    config: MatmulConfig
    name: str = "matmul_tg_to_dev"
    kind: str = "store"

    def render(self, ctx: CodegenContext) -> str:
        tile = self.config.tile
        return TgStoreFragment(
            name="store_C_tile",
            src_name=self.config.c_tile_name,
            dst_name=self.config.c_buffer_name,
            dst_row_stride=self.config.N_dim_var,
            row_start=f"tg.y * {tile.tile_M}",
            col_start=f"tg.x * {tile.tile_N}",
            tile_shape=(tile.tile_M, tile.tile_N),
            num_threads=tile.num_threads,
            row_limit=self.config.M_dim_var,
            col_limit=self.config.N_dim_var,
            dst_col_stride=self.config.c_col_stride,
        ).render(ctx)


@dataclass(frozen=True)
class SplitKSetupFragment:
    config: SplitKConfig
    M_dim_var: str
    N_dim_var: str
    parts_dim_var: str
    reduce: bool = False
    name: str = "splitk_setup"
    kind: str = "setup"

    def render(self, ctx: CodegenContext) -> str:
        if self.reduce:
            guard = f"row >= {self.M_dim_var} || col >= {self.N_dim_var}"
            part_line = ""
        else:
            guard = f"row >= {self.M_dim_var} || col >= {self.N_dim_var} || part >= {self.parts_dim_var}"
            part_line = "uint part = threadgroup_position_in_grid.z;"
        return f"""\
uint row = threadgroup_position_in_grid.y * {self.config.block_M} + thread_position_in_threadgroup.y;
uint col = threadgroup_position_in_grid.x * {self.config.block_N} + thread_position_in_threadgroup.x;
{part_line}
if ({guard}) {{
    return;
}}
float acc = 0.0f;"""


@dataclass(frozen=True)
class SplitKComputeFragment:
    a_buffer_name: str
    b_buffer_name: str
    K_dim_var: str
    N_dim_var: str
    part_k_dim_var: str
    a_col_stride: str = "1"
    b_col_stride: str = "1"
    name: str = "splitk_compute"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        a_k = "k" if self.a_col_stride == "1" else f"k * ({self.a_col_stride})"
        b_col = "col" if self.b_col_stride == "1" else f"col * ({self.b_col_stride})"
        return f"""\
uint k_begin = part * {self.part_k_dim_var};
uint k_end = min({self.K_dim_var}, k_begin + {self.part_k_dim_var});
for (uint k = k_begin; k < k_end; k++) {{
    acc += {self.a_buffer_name}[row * {self.K_dim_var} + {a_k}] * {self.b_buffer_name}[k * {self.N_dim_var} + {b_col}];
}}"""


@dataclass(frozen=True)
class SplitKPartialStoreFragment:
    partial_name: str
    M_dim_var: str
    N_dim_var: str
    col_stride: str = "1"
    name: str = "splitk_partial_store"
    kind: str = "store"

    def render(self, ctx: CodegenContext) -> str:
        col_term = "col" if self.col_stride == "1" else f"col * ({self.col_stride})"
        return f"{self.partial_name}[(part * {self.M_dim_var} + row) * {self.N_dim_var} + {col_term}] = acc;"


@dataclass(frozen=True)
class SplitKReduceComputeFragment:
    partial_name: str
    M_dim_var: str
    N_dim_var: str
    parts_dim_var: str
    col_stride: str = "1"
    name: str = "splitk_reduce_compute"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        col_term = "col" if self.col_stride == "1" else f"col * ({self.col_stride})"
        return f"""\
for (uint part = 0; part < {self.parts_dim_var}; part++) {{
    acc += {self.partial_name}[(part * {self.M_dim_var} + row) * {self.N_dim_var} + {col_term}];
}}"""


@dataclass(frozen=True)
class SplitKReduceStoreFragment:
    c_buffer_name: str
    N_dim_var: str
    col_stride: str = "1"
    name: str = "splitk_reduce_store"
    kind: str = "store"

    def render(self, ctx: CodegenContext) -> str:
        col_term = "col" if self.col_stride == "1" else f"col * ({self.col_stride})"
        return f"{self.c_buffer_name}[row * {self.N_dim_var} + {col_term}] = acc;"


def _simdgroup_tile_counts(tile: TileConfig) -> tuple[int, int]:
    if tile.tile_M % (tile.sg_M * 8) != 0:
        raise ValueError(
            f"tile_M ({tile.tile_M}) must be divisible by sg_M*8 ({tile.sg_M * 8})"
        )
    if tile.tile_N % (tile.sg_N * 8) != 0:
        raise ValueError(
            f"tile_N ({tile.tile_N}) must be divisible by sg_N*8 ({tile.sg_N * 8})"
        )
    return tile.tile_M // (tile.sg_M * 8), tile.tile_N // (tile.sg_N * 8)


def _indent(code: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in code.splitlines())
