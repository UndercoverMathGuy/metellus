import re
from dataclasses import dataclass
from enum import StrEnum

from compute.fragments import CodegenContext


class BroadcastSpec(StrEnum):
    """How an elementwise op's secondary operand maps onto the (M, N) output
    tile. StrEnum so values compare equal to their string form — existing
    codegen that branches on `mode == "row"` keeps working, but `BroadcastSpec("rrow")`
    now raises at construction time."""

    NONE = "none"
    SCALAR = "scalar"
    ROW = "row"
    COL = "col"


UNARY_EXPRESSIONS = {
    "negate": "-x",
    "absolute": "fabs(x)",
    "exp": "exp(x)",
    "log": "log(x)",
    "sqrt": "sqrt(x)",
    "recip": "1.0f / x",
    "sin": "sin(x)",
    "cos": "cos(x)",
    "tanh": "tanh(x)",
    "floor": "floor(x)",
    "ceil": "ceil(x)",
    "sign": "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)",
}

BINARY_EXPRESSIONS = {
    "add": "x + y",
    "subtract": "x - y",
    "mul": "x * y",
    "div": "x / y",
    "max": "fmax(x, y)",
    "min": "fmin(x, y)",
    "pow": "pow(x, y)",
    "equal": "x == y",
    "not_equal": "x != y",
    "lt": "x < y",
    "gt": "x > y",
    "ge": "x >= y",
    "le": "x <= y",
}

TERNARY_EXPRESSIONS = {
    "where": "cond != 0.0f ? x : y",
}

COMPARISON_OPS = frozenset({"equal", "not_equal", "lt", "gt", "ge", "le"})
BROADCAST_MODES = frozenset(m.value for m in BroadcastSpec)


def elementwise_arity(op: str) -> int:
    if op in UNARY_EXPRESSIONS:
        return 1
    if op in BINARY_EXPRESSIONS:
        return 2
    if op in TERNARY_EXPRESSIONS:
        return 3
    raise ValueError(
        f"Unsupported elementwise op {op!r}; expected one of {supported_elementwise_ops()}"
    )


def elementwise_expression(op: str) -> str:
    if op in UNARY_EXPRESSIONS:
        return UNARY_EXPRESSIONS[op]
    if op in BINARY_EXPRESSIONS:
        return BINARY_EXPRESSIONS[op]
    if op in TERNARY_EXPRESSIONS:
        return TERNARY_EXPRESSIONS[op]
    raise ValueError(
        f"Unsupported elementwise op {op!r}; expected one of {supported_elementwise_ops()}"
    )


def elementwise_outputs_bool(op: str) -> bool:
    elementwise_expression(op)
    return op in COMPARISON_OPS


def supported_elementwise_ops() -> tuple[str, ...]:
    return tuple(
        sorted((*UNARY_EXPRESSIONS, *BINARY_EXPRESSIONS, *TERNARY_EXPRESSIONS))
    )


def _rewrite_expression(
    expression: str, x_name: str, y_name: str, cond_name: str
) -> str:
    replacements = {"cond": cond_name, "x": x_name, "y": y_name}
    return re.sub(
        r"\b(cond|x|y)\b", lambda match: replacements[match.group(0)], expression
    )


def elementwise_compute_block(
    op: str,
    output_name: str = "out",
    x_name: str = "x",
    y_name: str = "y",
    cond_name: str = "cond",
) -> str:
    expression = _rewrite_expression(
        elementwise_expression(op), x_name, y_name, cond_name
    )
    if elementwise_outputs_bool(op):
        return f"{output_name} = ({expression}) ? 1.0f : 0.0f;"
    return f"{output_name} = {expression};"


def _tile_ref(tile_name: str, row: str, col: str) -> str:
    return f"{tile_name}[{row}][{col}]"


def _broadcast_ref(tile_name: str, mode: str, row: str, col: str) -> str:
    if mode == "none":
        return _tile_ref(tile_name, row, col)
    if mode == "scalar":
        return f"{tile_name}[0][0]"
    if mode == "row":
        return f"{tile_name}[0][{col}]"
    if mode == "col":
        return f"{tile_name}[{row}][0]"
    raise ValueError(
        f"Unsupported broadcast mode {mode!r}; expected one of {sorted(BROADCAST_MODES)}"
    )


def elementwise_threadgroup_section(
    op: str,
    output_tile: str,
    x_tile: str,
    tile_shape: tuple[int, int],
    num_threads: int,
    y_tile: str | None = None,
    cond_tile: str | None = None,
    y_broadcast: str = "none",
    cond_broadcast: str = "none",
) -> str:
    rows, cols = tile_shape
    arity = elementwise_arity(op)
    if arity >= 2 and y_tile is None:
        raise ValueError(f"{op!r} requires y_tile")
    if arity == 3 and cond_tile is None:
        raise ValueError(f"{op!r} requires cond_tile")
    if y_broadcast not in BROADCAST_MODES:
        raise ValueError(
            f"Unsupported y_broadcast {y_broadcast!r}; expected one of {sorted(BROADCAST_MODES)}"
        )
    if cond_broadcast not in BROADCAST_MODES:
        raise ValueError(
            f"Unsupported cond_broadcast {cond_broadcast!r}; expected one of {sorted(BROADCAST_MODES)}"
        )

    x_ref = _tile_ref(x_tile, "tile_row", "tile_col")
    y_ref = (
        _broadcast_ref(y_tile, y_broadcast, "tile_row", "tile_col")
        if y_tile is not None
        else "y"
    )
    cond_ref = (
        _broadcast_ref(cond_tile, cond_broadcast, "tile_row", "tile_col")
        if cond_tile is not None
        else "cond"
    )
    out_ref = _tile_ref(output_tile, "tile_row", "tile_col")
    y_decl = f"float y = {y_ref};" if arity >= 2 else ""
    cond_decl = f"float cond = {cond_ref};" if arity == 3 else ""

    return f"""\
for (uint idx = flat_tid; idx < {rows * cols}; idx += {num_threads}) {{
    uint tile_row = idx / {cols};
    uint tile_col = idx % {cols};
    float x = {x_ref};
    {y_decl}
    {cond_decl}
    {elementwise_compute_block(op, output_name=out_ref)}
}}
"""


def elementwise_threadgroup_scalar_section(
    op: str,
    output_tile: str,
    x_tile: str,
    scalar_name: str,
    tile_shape: tuple[int, int],
    num_threads: int,
) -> str:
    return elementwise_threadgroup_section(
        op,
        output_tile,
        x_tile,
        tile_shape,
        num_threads,
        y_tile=scalar_name,
        y_broadcast="scalar",
    )


def elementwise_threadgroup_row_section(
    op: str,
    output_tile: str,
    x_tile: str,
    row_tile: str,
    tile_shape: tuple[int, int],
    num_threads: int,
) -> str:
    return elementwise_threadgroup_section(
        op,
        output_tile,
        x_tile,
        tile_shape,
        num_threads,
        y_tile=row_tile,
        y_broadcast="row",
    )


def elementwise_threadgroup_col_section(
    op: str,
    output_tile: str,
    x_tile: str,
    col_tile: str,
    tile_shape: tuple[int, int],
    num_threads: int,
) -> str:
    return elementwise_threadgroup_section(
        op,
        output_tile,
        x_tile,
        tile_shape,
        num_threads,
        y_tile=col_tile,
        y_broadcast="col",
    )


@dataclass(frozen=True)
class ElementwiseComputeFragment:
    op: str
    output_tile: str
    x_tile: str
    tile_shape: tuple[int, int]
    num_threads: int
    y_tile: str | None = None
    cond_tile: str | None = None
    y_broadcast: str = "none"
    cond_broadcast: str = "none"
    name: str = "elementwise_compute"
    kind: str = "compute"

    def render(self, ctx: CodegenContext) -> str:
        return elementwise_threadgroup_section(
            op=self.op,
            output_tile=self.output_tile,
            x_tile=self.x_tile,
            y_tile=self.y_tile,
            cond_tile=self.cond_tile,
            y_broadcast=self.y_broadcast,
            cond_broadcast=self.cond_broadcast,
            tile_shape=self.tile_shape,
            num_threads=self.num_threads,
        )
