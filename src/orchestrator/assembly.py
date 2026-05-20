"""Assemble a FusionDecision → KernelGroup.

This is the only place where IR names get turned into MSL identifiers
and where the compute layer's fragments are wired into a CodegenContext.
Each strategy has a dedicated assembler — they share helpers for chain
expression building and "extra buffer" tracking (binary-tensor operands
that need to be threaded into the kernel's parameter list).

The compute / memory / fragments modules already provide every emission
primitive we need; this file does not write any new MSL. Fragment
identifiers (`global_row`, `global_col`, `flat_tid`, etc.) live in the
canonical preamble — see `compute/scaffold.py:kernel_preamble`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from compute.elementwise.elementwise import (
    BroadcastSpec,
    elementwise_arity,
    elementwise_expression,
    elementwise_outputs_bool,
)
from compute.copy.copy import StridedCopyFragment
from compute.fragments import BarrierFragment, CodegenContext
from compute.matmul import (
    MatmulAccumToDevFragment,
    MatmulAccumToTgFragment,
    MatmulComputeFragment,
    MatmulConfig,
    MatmulMainloopFragment,
    MatmulRegisterEpilogueFragment,
    MatmulSetupFragment,
    MatmulTgToDevFragment,
    MatmulTileMappingFragment,
    ThreadIndexFragment,
)
from compute.matmul.config import (
    TileConfig,
    grid_for,
    is_aligned_shape,
    select_tile_config,
)
from compute.reduction.reduction import (
    LastAxisReductionComputeFragment,
    LastAxisReductionSetupFragment,
    StoreReductionResultFragment,
)
from memory.memory import TgLoadFragment, TgStoreFragment
from orchestrator.fusion import FusionDecision
from orchestrator.ir import (
    ElementwiseOp,
    MatmulOp,
    ReductionOp,
    Scalar,
    ShapeOp,
    Tensor,
)
from orchestrator.kernel_group import FusionStrategy, KernelGroup
from runtime.program import Kernel


# ---------------------------------------------------------------------------
# Extras: extra device buffers + dim values that a fusion pulls in
# ---------------------------------------------------------------------------


@dataclass
class _Extras:
    """Tracks extra inputs a fusion pulls into the kernel beyond the
    anchor's primary inputs (e.g. bias tensors loaded for prologue or
    epilogue elementwise ops). Each extra has a slot in `ctx.buffers`
    and a binding env-key plus runtime row/col strides surfaced as dims.
    """

    buffers: list[str] = field(default_factory=list)
    bindings: list[str] = field(default_factory=list)
    dim_names: list[str] = field(default_factory=list)
    dim_values: list[int] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    base_slots_by_key: dict[str, int] = field(default_factory=dict)

    def add_tensor(self, t: Tensor, base_slot: int) -> int:
        """Register `t` as an input. Returns the buffer slot index. The
        MSL param name and env binding both use `buffer_key` so that an
        aliased view binds to its storage owner; already-added buffer_keys
        deduplicate."""
        key = t.buffer_key
        if key in self.base_slots_by_key:
            return self.base_slots_by_key[key]
        if key in self.seen:
            return self.bindings.index(key) + base_slot
        slot = base_slot + len(self.bindings)
        self.buffers.append(f"device const float* {key} [[buffer({slot})]]")
        self.bindings.append(key)
        self.seen.add(key)
        return slot


# ---------------------------------------------------------------------------
# Elementwise chain expression building
# ---------------------------------------------------------------------------


def _substitute(expr: str, **subs: str) -> str:
    """Replace whole-word identifiers (x, y, cond) in an elementwise
    expression template (e.g. 'x + y' or 'fmax(x, y)')."""

    def repl(m: re.Match) -> str:
        return subs.get(m.group(0), m.group(0))

    return re.sub(r"\b(cond|x|y)\b", repl, expr)


def _wrap_bool(op: str, expr: str) -> str:
    """Comparison ops produce bool in MSL — wrap into 1.0f/0.0f to match
    fp32 outputs that the rest of the pipeline assumes."""
    if elementwise_outputs_bool(op):
        return f"(({expr}) ? 1.0f : 0.0f)"
    return expr


def _scalar_literal(value: float) -> str:
    """fp32 literal with an 'f' suffix so MSL doesn't widen to double."""
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
        return _scalar_literal(operand.value)
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


ValueTransform = Callable[[str], str]


def _chain_value_transform(
    chain: tuple[ElementwiseOp, ...],
    *,
    row_var: str,
    col_var: str,
    extras: _Extras,
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
                expr = _wrap_bool(elem.op, _substitute(tpl, x=f"({expr})"))
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
                expr = _wrap_bool(elem.op, _substitute(tpl, **subs))
            value_by_name[elem.out.name] = expr
        return expr

    return transform


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


# ---------------------------------------------------------------------------
# Tile-resident elementwise chain (for matmul tg-tile epilogue and
# standalone elementwise chains).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TiledElementwiseChainFragment:
    """Walk a (rows, cols) threadgroup tile once, applying a chain of
    elementwise ops in-place. Binary-tensor operands are read from
    pre-loaded threadgroup tiles named in `y_tile_for[op]`; scalar
    operands inline as fp32 literals; broadcast direction picks the
    correct per-(tile_row, tile_col) index into the y_tile."""

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
        body_lines: list[str] = []
        cur = f"{self.primary_tile}[tile_row][tile_col]"
        value_by_name: dict[str, str] = {}
        primary = self.chain[0].operands[0]
        if isinstance(primary, Tensor):
            value_by_name[primary.name] = cur
        for i, elem in enumerate(self.chain):
            tpl = elementwise_expression(elem.op)
            if elementwise_arity(elem.op) == 1:
                expr = _substitute(tpl, x=f"({cur})")
            else:
                y_ref = self._operand_ref(
                    elem.operands[1],
                    elem.y_broadcast,
                    self.y_tile_for,
                    i,
                    value_by_name,
                )
                subs = {"x": f"({cur})", "y": f"({y_ref})"}
                if elementwise_arity(elem.op) == 3:
                    cond_ref = self._operand_ref(
                        elem.operands[2],
                        elem.cond_broadcast,
                        self.cond_tile_for,
                        i,
                        value_by_name,
                    )
                    subs["cond"] = f"({cond_ref})"
                expr = _substitute(tpl, **subs)
            expr = _wrap_bool(elem.op, expr)
            tmp = f"v{i}"
            body_lines.append(f"float {tmp} = {expr};")
            cur = tmp
            value_by_name[elem.out.name] = cur
        body_lines.append(f"{self.primary_tile}[tile_row][tile_col] = {cur};")
        body = "\n    ".join(body_lines)
        return (
            f"for (uint idx = flat_tid; idx < {rows * cols}; idx += {self.num_threads}) {{\n"
            f"    uint tile_row = idx / {cols};\n"
            f"    uint tile_col = idx % {cols};\n"
            f"    {body}\n"
            f"}}"
        )

    def _operand_ref(
        self,
        operand: Tensor | Scalar,
        broadcast: BroadcastSpec,
        tile_for: dict[int, str],
        index: int,
        value_by_name: dict[str, str],
    ) -> str:
        if isinstance(operand, Scalar):
            return _scalar_literal(operand.value)
        if operand.name in value_by_name:
            return value_by_name[operand.name]
        tile = tile_for[index]
        return _tile_operand_ref(tile, broadcast)


def _tile_operand_ref(tile: str, broadcast: BroadcastSpec) -> str:
    if broadcast is BroadcastSpec.SCALAR:
        return f"{tile}[0][0]"
    if broadcast is BroadcastSpec.ROW:
        return f"{tile}[0][tile_col]"
    if broadcast is BroadcastSpec.COL:
        return f"{tile}[tile_row][0]"
    return f"{tile}[tile_row][tile_col]"


# ---------------------------------------------------------------------------
# Per-strategy assemblers
# ---------------------------------------------------------------------------


def assemble(
    decision: FusionDecision, *, function_name: str | None = None
) -> KernelGroup:
    """Top-level dispatch on `decision.strategy`."""
    if decision.strategy in {
        FusionStrategy.STANDALONE_MATMUL,
        FusionStrategy.MATMUL_EPILOGUE_TG,
        FusionStrategy.MATMUL_EPILOGUE_REGISTER,
        FusionStrategy.ELEMENTWISE_PROLOGUE_MATMUL,
    }:
        return _assemble_matmul(decision, function_name)
    if decision.strategy in {
        FusionStrategy.STANDALONE_REDUCTION,
        FusionStrategy.REDUCTION_EPILOGUE,
        FusionStrategy.ELEMENTWISE_PROLOGUE_REDUCTION,
    }:
        return _assemble_reduction(decision, function_name)
    if decision.strategy in {
        FusionStrategy.STANDALONE_ELEMENTWISE,
        FusionStrategy.ELEMENTWISE_CHAIN,
    }:
        return _assemble_elementwise(decision, function_name)
    if decision.strategy is FusionStrategy.STANDALONE_SHAPE:
        return _assemble_shape(decision, function_name)
    raise ValueError(f"unknown strategy {decision.strategy!r}")


# ---------------------------------------------------------------------------
# Matmul assembly
# ---------------------------------------------------------------------------


def _assemble_matmul(
    decision: FusionDecision, function_name: str | None
) -> KernelGroup:
    anchor = decision.anchor
    assert isinstance(anchor, MatmulOp)
    # Effective device-input tensors: when a prologue is absorbed, the
    # actual buffer/strides come from the chain's outermost primary, not
    # from anchor.{a,b} (which name post-prologue values that have no
    # materialized buffer).
    src_a = decision.prologue_a[0].operands[0] if decision.prologue_a else anchor.a
    src_b = decision.prologue_b[0].operands[0] if decision.prologue_b else anchor.b
    assert isinstance(src_a, Tensor) and isinstance(src_b, Tensor)
    M, K = anchor.a.shape
    _, N = anchor.b.shape

    tile = select_tile_config(M, K, N)
    aligned = is_aligned_shape(M, K, N, tile)
    has_epilogue = bool(decision.epilogue)
    use_register = decision.strategy is FusionStrategy.MATMUL_EPILOGUE_REGISTER
    # Tg-tile path is required whenever we have a non-lane-agnostic epilogue.
    use_tg_tile_epilogue = has_epilogue and not use_register

    final_out = decision.epilogue[-1].out if decision.epilogue else anchor.out
    final_out_key = final_out.buffer_key

    base_slots = 3  # A=0, B=1, C=2
    extras = _Extras(
        base_slots_by_key={
            src_a.buffer_key: 0,
            src_b.buffer_key: 1,
            final_out_key: 2,
        }
    )

    # Prologue value-transforms (built once each, even when chain is empty).
    a_transform = (
        _chain_value_transform(
            decision.prologue_a,
            row_var="global_row",
            col_var="global_col",
            extras=extras,
            base_slot=base_slots,
        )
        if decision.prologue_a
        else None
    )
    b_transform = (
        _chain_value_transform(
            decision.prologue_b,
            row_var="global_row",
            col_var="global_col",
            extras=extras,
            base_slot=base_slots,
        )
        if decision.prologue_b
        else None
    )

    # For tg-tile epilogue, load any tensor-typed y operands into tiles
    # before the elementwise chain runs.
    epilogue_load_fragments: list = []
    epilogue_tile_decls: list[str] = []
    y_tile_for: dict[int, str] = {}
    cond_tile_for: dict[int, str] = {}
    if use_tg_tile_epilogue:
        available_names = {anchor.out.name}
        for i, elem in enumerate(decision.epilogue):
            for operand_index, broadcast, tile_for, prefix in (
                (1, elem.y_broadcast, y_tile_for, "eY"),
                (2, elem.cond_broadcast, cond_tile_for, "eCond"),
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
                ) = _broadcast_load_geometry(broadcast, tile, M, N, operand)
                epilogue_tile_decls.append(
                    f"threadgroup float {tile_name}[{tile_shape[0]}][{tile_shape[1]}];"
                )
                epilogue_load_fragments.append(
                    TgLoadFragment(
                        name=f"load_{tile_name}",
                        src_name=operand.buffer_key,
                        src_row_stride=row_stride,
                        src_col_stride=col_stride,
                        row_start=row_start,
                        col_start=col_start,
                        dst_name=tile_name,
                        tile_shape=tile_shape,
                        num_threads=tile.num_threads,
                        row_limit=None if aligned else row_limit,
                        col_limit=None if aligned else col_limit,
                    )
                )
            available_names.add(elem.out.name)

    # Build config — matmul's c_buffer_name must be the final output of
    # the fused chain (not the anchor's `out` if there's an epilogue).
    config = MatmulConfig(
        tile=tile,
        aligned=aligned and not use_tg_tile_epilogue,
        # ^ the aligned fast-path (MatmulAccumToDevFragment) is incompatible
        #   with a tg-tile epilogue (we need C_tile to apply the chain).
        M_dim_var="M",
        K_dim_var="K",
        N_dim_var="N",
        a_buffer_name=anchor.a.buffer_key,
        b_buffer_name=anchor.b.buffer_key,
        c_buffer_name=final_out_key,
        a_tile_name="A_tile",
        b_tile_name="B_tile",
        c_tile_name="C_tile",
    )

    # Register epilogue value-transform (lane-agnostic chain only).
    register_epilogue: tuple = ()
    if use_register:
        # Each elem is unary or has a Scalar y. _chain_value_transform on
        # the "lane element" expression gives us a callable; tile-coord
        # references aren't used because the chain is lane-agnostic.
        rt = _chain_value_transform(
            decision.epilogue,
            row_var="0",  # unused — lane-agnostic chain ignores these
            col_var="0",
            extras=extras,
            base_slot=base_slots,
        )
        register_epilogue = (
            MatmulRegisterEpilogueFragment(config=config, value_transform=rt),
        )

    main_row_limit: str | None = None if aligned else "M"
    main_k_limit: str | None = None if aligned else "K"
    main_n_limit: str | None = None if aligned else "N"

    # Strides come straight off the operand Tensor metadata, so a
    # transposed or reshaped view feeds the right offsets into the
    # cooperative loads. Literal ints (vs. the K/N dim vars) are fine —
    # the matmul is already recompiled per shape via `select_tile_config`.
    a_load = TgLoadFragment(
        name="load_A_tile",
        src_name=src_a.buffer_key,
        src_row_stride=str(src_a.row_stride),
        src_col_stride=str(src_a.col_stride),
        row_start=f"tg.y * {tile.tile_M}",
        col_start="k_chunk",
        dst_name="A_tile",
        tile_shape=(tile.tile_M, tile.tile_K),
        num_threads=tile.num_threads,
        row_limit=main_row_limit,
        col_limit=main_k_limit,
        value_transform=a_transform,
    )
    b_load = TgLoadFragment(
        name="load_B_tile",
        src_name=src_b.buffer_key,
        src_row_stride=str(src_b.row_stride),
        src_col_stride=str(src_b.col_stride),
        row_start="k_chunk",
        col_start=f"tg.x * {tile.tile_N}",
        dst_name="B_tile",
        tile_shape=(tile.tile_K, tile.tile_N),
        num_threads=tile.num_threads,
        row_limit=main_k_limit,
        col_limit=main_n_limit,
        value_transform=b_transform,
    )

    mainloop = MatmulMainloopFragment(
        tile_K=tile.tile_K,
        fragments=(
            a_load,
            b_load,
            BarrierFragment("inputs_ready"),
            MatmulComputeFragment(config),
            BarrierFragment("compute_done"),
        ),
        K_dim_var="K",
    )

    # Store path
    if use_tg_tile_epilogue:
        store_fragments: list = [
            MatmulAccumToTgFragment(config),
            BarrierFragment("c_tile_ready"),
            *epilogue_load_fragments,
            BarrierFragment("epilogue_inputs_ready"),
            _TiledElementwiseChainFragment(
                chain=decision.epilogue,
                primary_tile="C_tile",
                tile_shape=(tile.tile_M, tile.tile_N),
                num_threads=tile.num_threads,
                y_tile_for=y_tile_for,
                cond_tile_for=cond_tile_for,
            ),
            BarrierFragment("epilogue_done"),
            MatmulTgToDevFragment(config),
        ]
    elif use_register:
        # Aligned fast path may still apply when shape allows. The
        # register epilogue is applied to the accumulators before
        # writing to device.
        if config.aligned:
            store_fragments = [*register_epilogue, MatmulAccumToDevFragment(config)]
        else:
            store_fragments = [
                *register_epilogue,
                MatmulAccumToTgFragment(config),
                BarrierFragment("c_tile_ready"),
                MatmulTgToDevFragment(config),
            ]
    else:
        if config.aligned:
            store_fragments = [MatmulAccumToDevFragment(config)]
        else:
            store_fragments = [
                MatmulAccumToTgFragment(config),
                BarrierFragment("c_tile_ready"),
                MatmulTgToDevFragment(config),
            ]

    fragments = (
        MatmulTileMappingFragment(),
        ThreadIndexFragment(tile),
        MatmulSetupFragment(config),
        mainloop,
        *store_fragments,
    )

    # Threadgroup decls
    tg_decls = [
        f"threadgroup float A_tile[{tile.tile_M}][{tile.tile_K + tile.a_pad}];",
        f"threadgroup float B_tile[{tile.tile_K}][{tile.tile_N + tile.b_pad}];",
    ]
    if not config.aligned:
        tg_decls.append(
            f"threadgroup float C_tile[{tile.tile_M}][{tile.tile_N + tile.c_pad}];"
        )
    tg_decls.extend(epilogue_tile_decls)

    base_buffers = [
        f"device const float* {src_a.buffer_key} [[buffer(0)]]",
        f"device const float* {src_b.buffer_key} [[buffer(1)]]",
        f"device float* {final_out_key} [[buffer(2)]]",
    ]
    fn_name = function_name or f"matmul_{anchor.out.name}_fused"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=tuple(base_buffers + extras.buffers),
        dims=("M", "K", "N"),
        tg_x=tile.tg_x,
        tg_y=tile.tg_y,
        threadgroup_decls=tuple(tg_decls),
    )
    kernel = Kernel(fragments=fragments, ctx=ctx)
    bindings = (
        src_a.buffer_key,
        src_b.buffer_key,
        final_out_key,
        *extras.bindings,
    )
    return KernelGroup(
        kernel=kernel,
        bindings=bindings,
        dims=(M, K, N),
        grid=grid_for(M, N, tile),
        threads=(tile.tg_x, tile.tg_y, 1),
        ops=decision.ops,
        strategy=decision.strategy,
    )


def _broadcast_load_geometry(
    bc: BroadcastSpec,
    tile: TileConfig,
    M: int,
    N: int,
    operand: Tensor,
) -> tuple[str, str, tuple[int, int], str, str, str, str]:
    """Per-broadcast load geometry for an epilogue Y tile: returns
    (row_start, col_start, tile_shape, row_limit, col_limit, row_stride_dim).
    """
    if bc is BroadcastSpec.NONE:
        return (
            f"tg.y * {tile.tile_M}",
            f"tg.x * {tile.tile_N}",
            (tile.tile_M, tile.tile_N),
            "M",
            "N",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    if bc is BroadcastSpec.ROW:
        return (
            "0",
            f"tg.x * {tile.tile_N}",
            (1, tile.tile_N),
            "1",
            "N",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    if bc is BroadcastSpec.COL:
        return (
            f"tg.y * {tile.tile_M}",
            "0",
            (tile.tile_M, 1),
            "M",
            "1",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    if bc is BroadcastSpec.SCALAR:
        return (
            "0",
            "0",
            (1, 1),
            "1",
            "1",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    raise ValueError(f"unsupported broadcast {bc!r}")


# ---------------------------------------------------------------------------
# Reduction assembly
# ---------------------------------------------------------------------------


def _assemble_reduction(
    decision: FusionDecision, function_name: str | None
) -> KernelGroup:
    anchor = decision.anchor
    assert isinstance(anchor, ReductionOp)
    # When a prologue is absorbed, the actual device input is the
    # outermost prologue elem's primary — `anchor.input` is the IR name
    # of the post-prologue value, which has been fused away and has no
    # buffer. Strides / shape / buffer_key all come from the source.
    src = decision.prologue_a[0].operands[0] if decision.prologue_a else anchor.input
    assert isinstance(src, Tensor)
    M, K = src.shape
    final_out = decision.epilogue[-1].out if decision.epilogue else anchor.out

    base_slots = 2  # input=0, output=1
    extras = _Extras(
        base_slots_by_key={
            src.buffer_key: 0,
            final_out.buffer_key: 1,
        }
    )

    # Prologue → value_transform on the per-element load inside the reduction.
    # `row`/`idx` are the in-scope vars from LastAxisReductionComputeFragment;
    # the prologue indexes the source via its actual row_stride.
    prologue_tf = None
    if decision.prologue_a:
        prologue_tf = _chain_value_transform(
            decision.prologue_a,
            row_var="row",
            col_var="idx",
            extras=extras,
            base_slot=base_slots,
        )

    # Epilogue → value_transform on the per-row stored scalar.
    epilogue_tf = None
    if decision.epilogue:
        epilogue_tf = _build_reduction_epilogue_transform(decision.epilogue)

    tg_x = 128
    setup = LastAxisReductionSetupFragment(rows_dim="M", name="reduction_setup")
    compute = LastAxisReductionComputeFragment(
        op=anchor.op,
        input_name=src.buffer_key,
        scratch_name="scratch",
        reduce_dim="K",
        row_stride=str(src.row_stride),
        col_stride=str(src.col_stride),
        name="reduction_compute",
        value_transform=prologue_tf,
    )
    store = StoreReductionResultFragment(
        output_name=final_out.buffer_key,
        scratch_name="scratch",
        name="reduction_store",
        value_transform=epilogue_tf,
    )

    base_buffers = [
        f"device const float* {src.buffer_key} [[buffer(0)]]",
        f"device float* {final_out.buffer_key} [[buffer(1)]]",
    ]
    fn_name = function_name or f"reduce_{anchor.out.name}_fused"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=tuple(base_buffers + extras.buffers),
        dims=("M", "K"),
        tg_x=tg_x,
        tg_y=1,
        threadgroup_decls=(f"threadgroup float scratch[{(tg_x + 31) // 32}];",),
    )
    kernel = Kernel(fragments=(setup, compute, store), ctx=ctx)
    bindings = (src.buffer_key, final_out.buffer_key, *extras.bindings)
    return KernelGroup(
        kernel=kernel,
        bindings=bindings,
        dims=(M, K),
        grid=(M, 1, 1),
        threads=(tg_x, 1, 1),
        ops=decision.ops,
        strategy=decision.strategy,
    )


def _build_reduction_epilogue_transform(
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
                return lambda inner: _wrap_bool(name, _substitute(t, x=f"({inner})"))

            builders.append(make_unary(tpl, elem.op))
        else:
            assert isinstance(elem.operands[1], Scalar)
            y_lit = _scalar_literal(elem.operands[1].value)

            def make_binary(t: str, name: str, y: str) -> Callable[[str], str]:
                return lambda inner: _wrap_bool(
                    name, _substitute(t, x=f"({inner})", y=f"({y})")
                )

            builders.append(make_binary(tpl, elem.op, y_lit))

    def transform(v: str) -> str:
        expr = v
        for build in builders:
            expr = build(expr)
        return expr

    return transform


# ---------------------------------------------------------------------------
# Standalone / chained elementwise assembly
# ---------------------------------------------------------------------------


def _assemble_elementwise(
    decision: FusionDecision, function_name: str | None
) -> KernelGroup:
    chain = decision.chain_only
    assert chain, "elementwise decision must carry a chain"
    primary = chain[0].operands[0]
    assert isinstance(primary, Tensor)
    if len(primary.shape) == 1:
        M, N = primary.shape[0], 1
    else:
        M, N = primary.shape

    tile_M, tile_N = 16, 16
    tg_x, tg_y = 32, 8
    num_threads = tg_x * tg_y

    final_out = chain[-1].out
    base_slots = 2  # primary in=0, out=1
    extras = _Extras(
        base_slots_by_key={
            primary.buffer_key: 0,
            final_out.buffer_key: 1,
        }
    )

    load_fragments: list = []
    tg_decls: list[str] = [f"threadgroup float X_tile[{tile_M}][{tile_N}];"]
    y_tile_for: dict[int, str] = {}
    cond_tile_for: dict[int, str] = {}
    available_names = {primary.name}
    for i, elem in enumerate(chain):
        for operand_index, broadcast, tile_for, prefix in (
            (1, elem.y_broadcast, y_tile_for, "Y"),
            (2, elem.cond_broadcast, cond_tile_for, "Cond"),
        ):
            if operand_index >= elementwise_arity(elem.op):
                continue
            operand = elem.operands[operand_index]
            if not isinstance(operand, Tensor) or operand.name in available_names:
                continue
            tile_name = f"{prefix}{i}_tile"
            tile_for[i] = tile_name
            extras.add_tensor(operand, base_slots)
            row_start, col_start, ts, row_limit, col_limit, rs, cs = (
                _broadcast_load_geometry_2d(broadcast, tile_M, tile_N, operand)
            )
            tg_decls.append(f"threadgroup float {tile_name}[{ts[0]}][{ts[1]}];")
            load_fragments.append(
                TgLoadFragment(
                    name=f"load_{tile_name}",
                    src_name=operand.buffer_key,
                    src_row_stride=rs,
                    src_col_stride=cs,
                    row_start=row_start,
                    col_start=col_start,
                    dst_name=tile_name,
                    tile_shape=ts,
                    num_threads=num_threads,
                    row_limit=row_limit,
                    col_limit=col_limit,
                )
            )
        available_names.add(elem.out.name)

    primary_load = TgLoadFragment(
        name="load_X_tile",
        src_name=primary.buffer_key,
        src_row_stride=str(primary.row_stride),
        src_col_stride=str(primary.col_stride),
        row_start=f"tg.y * {tile_M}",
        col_start=f"tg.x * {tile_N}",
        dst_name="X_tile",
        tile_shape=(tile_M, tile_N),
        num_threads=num_threads,
        row_limit="M",
        col_limit="N",
    )
    compute = _TiledElementwiseChainFragment(
        chain=chain,
        primary_tile="X_tile",
        tile_shape=(tile_M, tile_N),
        num_threads=num_threads,
        y_tile_for=y_tile_for,
        cond_tile_for=cond_tile_for,
    )
    store = TgStoreFragment(
        name="store_X_tile",
        src_name="X_tile",
        dst_name=final_out.buffer_key,
        dst_row_stride="N",
        row_start=f"tg.y * {tile_M}",
        col_start=f"tg.x * {tile_N}",
        tile_shape=(tile_M, tile_N),
        num_threads=num_threads,
        row_limit="M",
        col_limit="N",
    )

    fragments = (
        primary_load,
        *load_fragments,
        BarrierFragment("inputs_ready"),
        compute,
        BarrierFragment("compute_done"),
        store,
    )
    base_buffers = [
        f"device const float* {primary.buffer_key} [[buffer(0)]]",
        f"device float* {final_out.buffer_key} [[buffer(1)]]",
    ]
    fn_name = function_name or f"elementwise_{final_out.name}_fused"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=tuple(base_buffers + extras.buffers),
        dims=("M", "N"),
        tg_x=tg_x,
        tg_y=tg_y,
        threadgroup_decls=tuple(tg_decls),
        preamble=(
            "uint2 tg = threadgroup_position_in_grid;",
            "uint2 lid = thread_position_in_threadgroup;",
            f"uint flat_tid = lid.y * {tg_x} + lid.x;",
        ),
    )
    kernel = Kernel(fragments=fragments, ctx=ctx)
    bindings = (primary.buffer_key, final_out.buffer_key, *extras.bindings)
    return KernelGroup(
        kernel=kernel,
        bindings=bindings,
        dims=(M, N),
        grid=((N + tile_N - 1) // tile_N, (M + tile_M - 1) // tile_M, 1),
        threads=(tg_x, tg_y, 1),
        ops=decision.ops,
        strategy=decision.strategy,
    )


# ---------------------------------------------------------------------------
# Shape (copy) assembly
# ---------------------------------------------------------------------------


def _assemble_shape(decision: FusionDecision, function_name: str | None) -> KernelGroup:
    """Minimal device→device copy. One thread per output element,
    scatter-load from input via its actual strides, contiguous store
    to output. No threadgroup memory. Block size 256 is a round number
    that fits any element count; the in-kernel `if (idx >= N) return;`
    handles partial tail blocks."""
    shape_op = decision.shape_op
    assert isinstance(shape_op, ShapeOp)
    in_t = shape_op.input
    out_t = shape_op.out
    N = out_t.element_count

    # Unravel the logical row-major index of the source. For 1D sources
    # we treat them as (1, M), so input_cols = M and in_row stays 0.
    if len(in_t.shape) == 2:
        input_cols = in_t.shape[1]
    elif len(in_t.shape) == 1:
        input_cols = in_t.shape[0]
    else:
        raise ValueError(
            f"ShapeOp input must be 1D or 2D in v0; got shape {in_t.shape}"
        )

    tg_x = 256
    block_size = tg_x
    grid_x = (N + block_size - 1) // block_size

    copy = StridedCopyFragment(
        input_name=in_t.buffer_key,
        output_name=out_t.buffer_key,
        input_row_stride=in_t.row_stride,
        input_col_stride=in_t.col_stride,
        input_cols=input_cols,
    )
    fn_name = function_name or f"shape_{out_t.name}"
    ctx = CodegenContext(
        function_name=fn_name,
        buffers=(
            f"device const float* {in_t.buffer_key} [[buffer(0)]]",
            f"device float* {out_t.buffer_key} [[buffer(1)]]",
        ),
        dims=("N",),
        tg_x=tg_x,
        tg_y=1,
        preamble=(
            "uint2 lid = thread_position_in_threadgroup;",
            f"uint flat_tid = lid.y * {tg_x} + lid.x;",
        ),
    )
    kernel = Kernel(fragments=(copy,), ctx=ctx)
    return KernelGroup(
        kernel=kernel,
        bindings=(in_t.buffer_key, out_t.buffer_key),
        dims=(N,),
        grid=(grid_x, 1, 1),
        threads=(tg_x, 1, 1),
        ops=(shape_op,),
        strategy=decision.strategy,
    )


def _broadcast_load_geometry_2d(
    bc: BroadcastSpec, tile_M: int, tile_N: int, operand: Tensor
) -> tuple[str, str, tuple[int, int], str, str, str, str]:
    if bc is BroadcastSpec.NONE:
        return (
            f"tg.y * {tile_M}",
            f"tg.x * {tile_N}",
            (tile_M, tile_N),
            "M",
            "N",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    if bc is BroadcastSpec.ROW:
        return (
            "0",
            f"tg.x * {tile_N}",
            (1, tile_N),
            "1",
            "N",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    if bc is BroadcastSpec.COL:
        return (
            f"tg.y * {tile_M}",
            "0",
            (tile_M, 1),
            "M",
            "1",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    if bc is BroadcastSpec.SCALAR:
        return (
            "0",
            "0",
            (1, 1),
            "1",
            "1",
            str(operand.row_stride),
            str(operand.col_stride),
        )
    raise ValueError(f"unsupported broadcast {bc!r}")
