from __future__ import annotations

from dataclasses import dataclass

from compute.fragments import CodegenContext, TgmemAccess


def _msl_float(value: float) -> str:
    text = repr(float(value))
    if "inf" in text or "nan" in text:
        raise ValueError(f"Cannot embed non-finite float as MSL literal: {value!r}")
    return f"{text}f"


@dataclass(frozen=True)
class FillFragment:
    out_name: str
    count_dim: str
    value: float = 0.0
    name: str = "fill"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        gid = f"tg.x * {ctx.tg_x} + flat_tid"
        return f"""\
if ({gid} < {self.count_dim}) {{
    {self.out_name}[{gid}] = {_msl_float(self.value)};
}}"""

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        return ()


@dataclass(frozen=True)
class ArangeFragment:
    out_name: str
    count_dim: str
    start: float = 0.0
    step: float = 1.0
    name: str = "arange"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        gid = f"tg.x * {ctx.tg_x} + flat_tid"
        return f"""\
if ({gid} < {self.count_dim}) {{
    {self.out_name}[{gid}] = {_msl_float(self.start)} + {_msl_float(self.step)} * float({gid});
}}"""

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        return ()


@dataclass(frozen=True)
class CopyFragment:
    out_name: str
    in_name: str
    count_dim: str
    name: str = "copy"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        gid = f"tg.x * {ctx.tg_x} + flat_tid"
        return f"""\
if ({gid} < {self.count_dim}) {{
    {self.out_name}[{gid}] = {self.in_name}[{gid}];
}}"""

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        return ()
