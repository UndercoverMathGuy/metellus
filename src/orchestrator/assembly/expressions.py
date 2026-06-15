"""MSL expression building for elementwise chains.

`chain_value_transform` turns a prologue chain into a closure: given the
raw device-memory load expression for the primary, it returns the
post-chain expression at the same (row, col). The reduction epilogue
builder is the same idea but on a 1-element scalar instead of a 2D
element load.

Helpers `substitute`, `wrap_bool`, `scalar_literal` are shared with the
tiled-chain fragment for emitting the same per-op MSL.
"""

from __future__ import annotations

import math
import re
from typing import Callable

from compute.elementwise.elementwise import (
    BroadcastSpec,
    elementwise_arity,
    elementwise_expression,
    elementwise_outputs_bool,
)
from orchestrator.ir import ElementwiseOp, Scalar, Tensor

from orchestrator.assembly.extras import Extras


ValueTransform = Callable[[str], str]


def substitute(expr: str, **subs: str) -> str:
    """Replace whole-word identifiers (x, y, cond) in an elementwise
    expression template (e.g. 'x + y' or 'fmax(x, y)')."""

    def repl(m: re.Match) -> str:
        return subs.get(m.group(0), m.group(0))

    return re.sub(r"\b(cond|x|y)\b", repl, expr)


def wrap_bool(op: str, expr: str) -> str:
    """Comparison ops produce bool in MSL — wrap into 1.0f/0.0f to match
    fp32 outputs that the rest of the pipeline assumes."""
    if elementwise_outputs_bool(op):
        return f"(({expr}) ? 1.0f : 0.0f)"
    return expr


def scalar_literal(value: float) -> str:
    """fp32 literal with an 'f' suffix so MSL doesn't widen to double."""
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    return f"{value!r}f" if "e" in repr(value) else f"{value}f"


def _operand_expr(
    operand: Tensor | Scalar,
    *,
    broadcast: BroadcastSpec,
    row_var: str,
    col_var: str,
) -> str:
    """MSL expression for reading one element of an operand from device
    memory at the in-scope (row, col). Used by prologue value-transforms
    (where `row_var`/`col_var` are the source-tensor coordinates of the
    primary load)."""
    if isinstance(operand, Scalar):
        return scalar_literal(operand.value)
    key = operand.buffer_key
    if broadcast is BroadcastSpec.SCALAR:
        return f"{key}[0]"
    if broadcast is BroadcastSpec.ROW:
        col_term = (
            col_var
            if operand.col_stride == 1
            else f"{col_var} * ({operand.col_stride})"
        )
        return f"{key}[{col_term}]"
    if broadcast is BroadcastSpec.COL:
        row_term = (
            row_var
            if operand.row_stride == 1
            else f"{row_var} * ({operand.row_stride})"
        )
        return f"{key}[{row_term}]"
    row_term = (
        row_var if operand.row_stride == 1 else f"{row_var} * ({operand.row_stride})"
    )
    col_term = (
        col_var if operand.col_stride == 1 else f"{col_var} * ({operand.col_stride})"
    )
    return f"{key}[{row_term} + {col_term}]"


def _transform_operand_expr(
    operand: Tensor | Scalar,
    *,
    broadcast: BroadcastSpec,
    row_var: str,
    col_var: str,
    value_by_name: dict[str, str],
) -> str:
    if isinstance(operand, Tensor) and operand.name in value_by_name:
        return value_by_name[operand.name]
    return _operand_expr(
        operand,
        broadcast=broadcast,
        row_var=row_var,
        col_var=col_var,
    )


def chain_value_transform(
    chain: tuple[ElementwiseOp, ...],
    *,
    row_var: str,
    col_var: str,
    extras: Extras,
    base_slot: int,
) -> ValueTransform:
    """Build a value-transform closure for a prologue chain, where the
    raw expression is the device-memory load for the primary element at
    `(row_var, col_var)`. Each elem in `chain` wraps the previous
    expression. Binary tensor operands are registered as extras."""

    if not chain:
        return lambda raw: raw
    primary = chain[0].operands[0]
    assert isinstance(primary, Tensor)
    available_names: set[str] = {primary.name}
    for elem in chain:
        for operand in elem.operands[1 : elementwise_arity(elem.op)]:
            if isinstance(operand, Tensor) and operand.name not in available_names:
                extras.add_tensor(operand, base_slot)
        available_names.add(elem.out.name)

    def transform(raw: str) -> str:
        expr = raw
        value_by_name: dict[str, str] = {primary.name: raw}
        for elem in chain:
            tpl = elementwise_expression(elem.op)
            if elementwise_arity(elem.op) == 1:
                expr = wrap_bool(elem.op, substitute(tpl, x=f"({expr})"))
            else:
                y_expr = _transform_operand_expr(
                    elem.operands[1],
                    broadcast=elem.y_broadcast,
                    row_var=row_var,
                    col_var=col_var,
                    value_by_name=value_by_name,
                )
                subs = {"x": f"({expr})", "y": f"({y_expr})"}
                if elementwise_arity(elem.op) == 3:
                    cond_expr = _transform_operand_expr(
                        elem.operands[2],
                        broadcast=elem.cond_broadcast,
                        row_var=row_var,
                        col_var=col_var,
                        value_by_name=value_by_name,
                    )
                    subs["cond"] = f"({cond_expr})"
                expr = wrap_bool(elem.op, substitute(tpl, **subs))
            value_by_name[elem.out.name] = expr
        return expr

    return transform


def build_reduction_epilogue_transform(
    chain: tuple[ElementwiseOp, ...],
) -> ValueTransform:
    """Reduction epilogue elems are unary or Scalar-binary (enforced by
    the fuser's eligibility predicate). Build a closure that wraps the
    1-element scalar value."""
    builders: list[Callable[[str], str]] = []
    for elem in chain:
        tpl = elementwise_expression(elem.op)
        if elementwise_arity(elem.op) == 1:

            def make_unary(t: str, name: str) -> Callable[[str], str]:
                return lambda inner: wrap_bool(name, substitute(t, x=f"({inner})"))

            builders.append(make_unary(tpl, elem.op))
        else:
            assert isinstance(elem.operands[1], Scalar)
            y_lit = scalar_literal(elem.operands[1].value)

            def make_binary(t: str, name: str, y: str) -> Callable[[str], str]:
                return lambda inner: wrap_bool(
                    name, substitute(t, x=f"({inner})", y=f"({y})")
                )

            builders.append(make_binary(tpl, elem.op, y_lit))

    def transform(v: str) -> str:
        expr = v
        for build in builders:
            expr = build(expr)
        return expr

    return transform
