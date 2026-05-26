"""Tile-resident elementwise chain fragment.

Walks a (rows, cols) threadgroup tile once using `flat_tid` to step
through elements, and applies a chain of elementwise ops in place.
Y and Cond tensor operands come from pre-loaded threadgroup tiles
named in `y_tile_for[op_index]` and `cond_tile_for[op_index]`.
Scalar operands are inlined as float literals. Broadcast settings
choose which (tile_row, tile_col) entry of the Y/Cond tile to read.

Used by both the matmul tg-tile epilogue path and the standalone
elementwise template — the same fragment, different surrounding
templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compute.elementwise.elementwise import (
    BroadcastSpec,
    elementwise_arity,
    elementwise_expression,
)
from compute.fragments import CodegenContext, TgmemAccess
from orchestrator.ir import ElementwiseOp, Scalar, Tensor

from orchestrator.assembly.expressions import scalar_literal, substitute, wrap_bool


def tile_operand_ref(tile: str, broadcast: BroadcastSpec) -> str:
    if broadcast is BroadcastSpec.SCALAR:
        return f"{tile}[0][0]"
    if broadcast is BroadcastSpec.ROW:
        return f"{tile}[0][tile_col]"
    if broadcast is BroadcastSpec.COL:
        return f"{tile}[tile_row][0]"
    return f"{tile}[tile_row][tile_col]"


# * * Look at render_tile_body for the main logic - rest are all helpers and annoying to read (i've tried trust me)


@dataclass(frozen=True)
class TiledElementwiseChainFragment:
    chain: tuple[ElementwiseOp, ...]
    primary_tile: str
    tile_shape: tuple[int, int]
    num_threads: int
    y_tile_for: dict[int, str]
    cond_tile_for: dict[int, str] = field(default_factory=dict)
    name: str = "tiled_elementwise_chain"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        rows, cols = self.tile_shape
        body = self.render_tile_body()
        return (
            f"for (uint idx = flat_tid; idx < {rows * cols}; idx += {self.num_threads}) {{\n"
            f"    uint tile_row = idx / {cols};\n"
            f"    uint tile_col = idx % {cols};\n"
            f"    {body}\n"
            f"}}"
        )

    def render_tile_body(self) -> str:
        current_value = self._primary_value_ref()  # primary operand reference
        local_values = self._initial_local_values(current_value)
        body_lines: list[str] = []

        for index, elem in enumerate(self.chain):
            tmp = f"v{index}"  # temporary intermediate value
            expr = self._element_expression(index, elem, current_value, local_values)
            body_lines.append(f"float {tmp} = {expr};")  # computing op
            current_value = tmp  # moving to next intermediate for chaining
            local_values[elem.out.name] = current_value

        body_lines.append(f"{self.primary_tile}[tile_row][tile_col] = {current_value};")
        body = "\n    ".join(body_lines)
        return body

    def _primary_value_ref(self) -> str:
        return f"{self.primary_tile}[tile_row][tile_col]"

    def _initial_local_values(self, primary_ref: str) -> dict[str, str]:
        local_values: dict[str, str] = {}
        primary_operand = self.chain[0].operands[0]
        if isinstance(primary_operand, Tensor):
            local_values[primary_operand.name] = (
                primary_ref  # first value is primary operand
            )
        return local_values

    def _element_expression(  # resolves elementwise expression for arity, values
        self,
        index: int,
        elem: ElementwiseOp,
        current_value: str,
        local_values: dict[str, str],
    ) -> str:
        substitutions = self._element_substitutions(
            index,
            elem,
            current_value,
            local_values,
        )
        expr = substitute(elementwise_expression(elem.op), **substitutions)
        return wrap_bool(elem.op, expr)

    def _element_substitutions(
        self,
        index: int,
        elem: ElementwiseOp,
        current_value: str,
        local_values: dict[str, str],
    ) -> dict[str, str]:
        substitutions = {"x": f"({current_value})"}
        arity = elementwise_arity(elem.op)

        if arity >= 2:
            substitutions["y"] = self._substitution_ref(
                elem.operands[1],
                elem.y_broadcast,
                self.y_tile_for,
                index,
                local_values,
            )

        if arity == 3:
            substitutions["cond"] = self._substitution_ref(
                elem.operands[2],
                elem.cond_broadcast,
                self.cond_tile_for,
                index,
                local_values,
            )

        return substitutions

    def _substitution_ref(
        self,
        operand: Tensor | Scalar,
        broadcast: BroadcastSpec,
        operand_tiles: dict[int, str],
        index: int,
        local_values: dict[str, str],
    ) -> str:
        operand_ref = self._operand_ref(
            operand,
            broadcast,
            operand_tiles,
            index,
            local_values,
        )
        return f"({operand_ref})"

    @property
    def tgmem_accesses(self) -> tuple[TgmemAccess, ...]:
        accesses = [
            TgmemAccess(
                name=self.primary_tile,
                access="readwrite",
                size_floats=self.tile_shape[0] * self.tile_shape[1],
                shape=self.tile_shape,
            )
        ]

        for index, elem in enumerate(self.chain):
            if index in self.y_tile_for:
                accesses.append(
                    self._readonly_tgmem_access(
                        self.y_tile_for[index],
                        elem.y_broadcast,
                    )
                )
            if index in self.cond_tile_for:
                accesses.append(
                    self._readonly_tgmem_access(
                        self.cond_tile_for[index],
                        elem.cond_broadcast,
                    )
                )
        return tuple(accesses)

    def _readonly_tgmem_access(
        self,
        tile: str,
        broadcast: BroadcastSpec,
    ) -> TgmemAccess:
        shape = self._broadcast_tile_shape(broadcast)
        return TgmemAccess(
            name=tile,
            access="read",
            size_floats=shape[0] * shape[1],
            shape=shape,
        )

    def _broadcast_tile_shape(self, broadcast: BroadcastSpec) -> tuple[int, int]:
        rows, cols = self.tile_shape
        if broadcast is BroadcastSpec.SCALAR:
            return (1, 1)
        if broadcast is BroadcastSpec.ROW:
            return (1, cols)
        if broadcast is BroadcastSpec.COL:
            return (rows, 1)
        return self.tile_shape

    def _operand_ref(
        self,
        operand: Tensor | Scalar,
        broadcast: BroadcastSpec,
        operand_tiles: dict[int, str],
        index: int,
        local_values: dict[str, str],
    ) -> str:
        if isinstance(operand, Scalar):
            return scalar_literal(operand.value)
        if operand.name in local_values:
            return local_values[operand.name]
        tile = operand_tiles[index]
        return tile_operand_ref(tile, broadcast)
