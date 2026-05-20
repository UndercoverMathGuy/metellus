from dataclasses import dataclass


@dataclass(frozen=True)
class TileConfig:
    tile_M: int
    tile_N: int
    tile_K: int
    sg_M: int
    sg_N: int
    tg_x: int = 32
    tg_y: int = 4
    a_pad: int = 0
    b_pad: int = 0
    c_pad: int = 0

    @property
    def num_threads(self) -> int:
        return self.tg_x * self.tg_y


@dataclass(frozen=True)
class SplitKConfig:
    block_M: int = 16
    block_N: int = 16
    part_K: int = 512


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def is_aligned_shape(M: int, K: int, N: int, config: TileConfig) -> bool:
    return (
        M % config.tile_M == 0
        and N % config.tile_N == 0
        and K % config.tile_K == 0
        and K % 4 == 0
        and N % 4 == 0
    )


def grid_for(M: int, N: int, config: TileConfig) -> tuple[int, int, int]:
    return (ceil_div(N, config.tile_N), ceil_div(M, config.tile_M), 1)


def should_use_splitk(M: int, K: int, N: int) -> bool:
    return M * N <= 4096 and K >= max(M, N, 1024)


def select_tile_config(M: int, K: int, N: int) -> TileConfig:
    if M % 64 != 0 or N % 64 != 0 or K % 32 != 0:
        return TileConfig(32, 32, 16, 2, 2, b_pad=1, c_pad=1)
    if M >= N * 2:
        return TileConfig(128, 64, 32, 2, 2, b_pad=1, c_pad=1)
    if N >= M * 2:
        return TileConfig(64, 128, 32, 2, 2, b_pad=1, c_pad=1)
    if M < 64 or N < 64:
        return TileConfig(32, 32, 16, 2, 2, b_pad=1, c_pad=1)
    return TileConfig(64, 64, 32, 2, 2, b_pad=1, c_pad=1)
