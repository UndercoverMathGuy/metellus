"""Structural tests for the fuser.

Each test builds an IR program, runs `fuse`, and asserts which ops
landed in which decision and what strategy was picked. A small subset
exercise `assemble` to confirm the assembled MSL contains the expected
fragment markers (no GPU execution — that's for the bench scripts)."""

import pytest

from orchestrator import (
    MatmulOp,
    Operations,
)
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse
from orchestrator.kernel_group import FusionStrategy


def _op_types(decision) -> list[str]:
    return [type(o).__name__ for o in decision.ops]


# ---------------------------------------------------------------------------
# Per-pattern decision tests
# ---------------------------------------------------------------------------


def test_matmul_epilogue_tg_when_row_broadcast_present():
    # Use shapes that pick the 32x32 matmul tile (any dim failing
    # `M%64==0 && N%64==0 && K%32==0`). The 64x64 tile + tg-tile path
    # exceeds the 32KB tg-memory budget even before any Y tile, so the
    # fuser would refuse the epilogue absorption there.
    ops = Operations()
    ops.input("A", shape=(32, 64))
    ops.input("B", shape=(64, 128))
    ops.input("bias", shape=(1, 128))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("add", out="D", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("max", out="Y", operands=("D", 0.0))
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.strategy is FusionStrategy.MATMUL_EPILOGUE_TG
    assert _op_types(d) == ["MatmulOp", "ElementwiseOp", "ElementwiseOp"]
    assert len(d.epilogue) == 2


def test_matmul_epilogue_register_when_lane_agnostic_only():
    ops = Operations()
    ops.input("A", shape=(128, 64))
    ops.input("B", shape=(64, 128))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("max", out="Y", operands=("C", 0.0))  # relu — unary scalar
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.strategy is FusionStrategy.MATMUL_EPILOGUE_REGISTER
    assert _op_types(d) == ["MatmulOp", "ElementwiseOp"]


def test_elementwise_prologue_matmul_on_a():
    ops = Operations()
    ops.input("A", shape=(128, 64))
    ops.input("B", shape=(64, 128))
    # Apply relu to A before the matmul.
    ops.elementwise("max", out="A_relu", operands=("A", 0.0))
    ops.matmul(a="A_relu", b="B", out="C")
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.strategy is FusionStrategy.ELEMENTWISE_PROLOGUE_MATMUL
    assert len(d.prologue_a) == 1
    assert d.prologue_a[0].op == "max"


def test_elementwise_prologue_reduction():
    ops = Operations()
    ops.input("X", shape=(32, 64))
    ops.elementwise("exp", out="EX", operands=("X",))
    ops.reduction("sum", out="S", x="EX")
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.strategy is FusionStrategy.ELEMENTWISE_PROLOGUE_REDUCTION
    assert len(d.prologue_a) == 1
    assert d.prologue_a[0].op == "exp"


def test_reduction_epilogue_scalar_only():
    ops = Operations()
    ops.input("X", shape=(32, 64))
    ops.reduction("sum", out="S", x="X")
    ops.elementwise("mul", out="Y", operands=("S", 0.5))  # scalar binary on 1D
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.strategy is FusionStrategy.REDUCTION_EPILOGUE
    assert len(d.epilogue) == 1


def test_elementwise_chain_standalone():
    ops = Operations()
    ops.input("X", shape=(32, 64))
    ops.elementwise("exp", out="E", operands=("X",))
    ops.elementwise("log", out="L", operands=("E",))
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.strategy is FusionStrategy.ELEMENTWISE_CHAIN
    assert _op_types(d) == ["ElementwiseOp", "ElementwiseOp"]


def test_standalone_kernels_when_no_fusion():
    ops = Operations()
    ops.input("A", shape=(64, 64))
    ops.input("B", shape=(64, 64))
    ops.matmul(a="A", b="B", out="C")
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    assert decisions[0].strategy is FusionStrategy.STANDALONE_MATMUL


# ---------------------------------------------------------------------------
# Chain composition
# ---------------------------------------------------------------------------


def test_chained_prologue_and_epilogue_around_matmul():
    """elem → matmul → elem → elem all collapse into one kernel."""
    # Pick shapes that fall on the 32x32 matmul tile so the tg-tile
    # epilogue + row-broadcast bias tile fit under the 32KB budget.
    ops = Operations()
    ops.input("A", shape=(32, 64))
    ops.input("B", shape=(64, 64))
    ops.input("bias", shape=(1, 64))
    ops.elementwise("exp", out="A_exp", operands=("A",))  # prologue on A
    ops.matmul(a="A_exp", b="B", out="C")
    ops.elementwise("add", out="D", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("max", out="Y", operands=("D", 0.0))
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    d = decisions[0]
    assert d.strategy is FusionStrategy.MATMUL_EPILOGUE_TG
    assert len(d.prologue_a) == 1 and d.prologue_a[0].op == "exp"
    assert len(d.epilogue) == 2
    assert _op_types(d) == [
        "ElementwiseOp",
        "MatmulOp",
        "ElementwiseOp",
        "ElementwiseOp",
    ]


# ---------------------------------------------------------------------------
# Multi-consumer rules
# ---------------------------------------------------------------------------


def test_multi_consumer_downstream_blocks_epilogue_fusion():
    """matmul C → elem D (consumer 1) and elem E (consumer 2).
    Downstream fusion requires unique consumer; neither D nor E may be
    absorbed into the matmul. The matmul stays standalone."""
    ops = Operations()
    ops.input("A", shape=(64, 64))
    ops.input("B", shape=(64, 64))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("exp", out="D", operands=("C",))
    ops.elementwise("log", out="E", operands=("C",))
    decisions = fuse(ops.build())
    # Matmul standalone, plus two separate elementwise kernels.
    assert any(d.strategy is FusionStrategy.STANDALONE_MATMUL for d in decisions)
    elem_decisions = [
        d
        for d in decisions
        if d.strategy
        in (
            FusionStrategy.STANDALONE_ELEMENTWISE,
            FusionStrategy.ELEMENTWISE_CHAIN,
        )
    ]
    assert len(elem_decisions) == 2


@pytest.mark.xfail(
    reason="Multi-consumer prologue carve-out temporarily disabled — see "
    "fusion._extend_prologue. Re-enable with a primary-materialization "
    "guard so this case (lane-agnostic elem feeding multiple anchors from "
    "a program input) is sound while diamond chains stay rejected."
)
def test_multi_consumer_upstream_unary_fuses_into_both_and_elides():
    """Lane-agnostic elem fed into two matmuls (or matmul+reduc) should
    fuse into both AND be elided (the elem has no remaining standalone
    output)."""
    ops = Operations()
    ops.input("X", shape=(64, 64))
    ops.input("B1", shape=(64, 64))
    ops.input("B2", shape=(64, 64))
    ops.elementwise("max", out="Xr", operands=("X", 0.0))
    ops.matmul(a="Xr", b="B1", out="C1")
    ops.matmul(a="Xr", b="B2", out="C2")
    decisions = fuse(ops.build())
    # Two matmul decisions, no standalone Xr decision.
    matmul_decisions = [d for d in decisions if isinstance(d.anchor, MatmulOp)]
    assert len(matmul_decisions) == 2
    for d in matmul_decisions:
        assert len(d.prologue_a) == 1 and d.prologue_a[0].op == "max"
    # No standalone elementwise kernel for Xr.
    elem_decisions = [
        d
        for d in decisions
        if d.strategy
        in (
            FusionStrategy.STANDALONE_ELEMENTWISE,
            FusionStrategy.ELEMENTWISE_CHAIN,
        )
    ]
    assert elem_decisions == []


def test_multi_consumer_upstream_non_lane_agnostic_blocks_and_keeps_standalone():
    """Binary-tensor elem with a row broadcast is not lane-agnostic; the
    upstream carve-out doesn't apply, so the elem must stay materialized
    (its own kernel) and neither matmul absorbs it."""
    ops = Operations()
    ops.input("X", shape=(64, 64))
    ops.input("bias", shape=(1, 64))
    ops.input("B1", shape=(64, 64))
    ops.input("B2", shape=(64, 64))
    ops.elementwise("add", out="Xb", operands=("X", "bias"), y_broadcast="row")
    ops.matmul(a="Xb", b="B1", out="C1")
    ops.matmul(a="Xb", b="B2", out="C2")
    decisions = fuse(ops.build())
    # Xb stays as its own kernel; both matmuls are standalone.
    matmul_decisions = [d for d in decisions if isinstance(d.anchor, MatmulOp)]
    assert len(matmul_decisions) == 2
    for d in matmul_decisions:
        assert d.prologue_a == ()
        assert d.strategy is FusionStrategy.STANDALONE_MATMUL
    elem_decisions = [
        d
        for d in decisions
        if d.strategy
        in (
            FusionStrategy.STANDALONE_ELEMENTWISE,
            FusionStrategy.ELEMENTWISE_CHAIN,
        )
    ]
    assert len(elem_decisions) == 1


# ---------------------------------------------------------------------------
# Reduction shape boundary
# ---------------------------------------------------------------------------


def test_reduction_does_not_chain_through_matmul():
    """Reduction output is 1D — no matmul can consume it. The matmul
    later in the program lives in its own kernel."""
    ops = Operations()
    ops.input("X", shape=(8, 16))
    ops.input("A", shape=(8, 16))
    ops.input("B", shape=(16, 8))
    ops.reduction("sum", out="R", x="X")  # R is shape (8,)
    ops.matmul(a="A", b="B", out="C")
    decisions = fuse(ops.build())
    strategies = {d.strategy for d in decisions}
    assert FusionStrategy.STANDALONE_REDUCTION in strategies
    assert FusionStrategy.STANDALONE_MATMUL in strategies


# ---------------------------------------------------------------------------
# Assembly — sanity-check the MSL contains expected markers
# ---------------------------------------------------------------------------


def test_assemble_relu_matmul_bias_contains_chain_markers():
    # Small-tile-eligible shapes so the tg-tile epilogue + row-bias Y
    # tile fit in 32KB tg memory.
    ops = Operations()
    ops.input("A", shape=(32, 64))
    ops.input("B", shape=(64, 128))
    ops.input("bias", shape=(1, 128))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("add", out="D", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("max", out="Y", operands=("D", 0.0))
    decision = fuse(ops.build())[0]
    group = assemble(decision)
    src = group.kernel.source
    # Matmul mainloop present
    assert "simdgroup_multiply_accumulate" in src
    # Bias was loaded into a threadgroup tile
    assert "eY0_tile" in src
    # The fused chain is one inline loop with two ops (add then fmax)
    assert "fmax" in src
    assert "C_tile[tile_row][tile_col] = " in src
    # The result is written to Y (the chain tail)
    assert "device float* Y [[buffer(2)]]" in src
    # Bias appears as a device input
    assert "device const float* bias" in src
    # Bindings are A, B, Y, bias (extra)
    assert group.bindings == ("A", "B", "Y", "bias")


def test_assemble_register_epilogue_uses_thread_elements():
    ops = Operations()
    ops.input("A", shape=(128, 64))
    ops.input("B", shape=(64, 128))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("max", out="Y", operands=("C", 0.0))
    decision = fuse(ops.build())[0]
    group = assemble(decision)
    src = group.kernel.source
    assert group.strategy is FusionStrategy.MATMUL_EPILOGUE_REGISTER
    # MatmulRegisterEpilogueFragment emits per-lane `thread_elements()` writes.
    assert "thread_elements" in src
    # The fmax(., 0) transform is inlined per-lane.
    assert "fmax" in src


def test_assemble_standalone_where_loads_cond_operand():
    ops = Operations()
    ops.input("X", shape=(16, 16))
    ops.input("Y", shape=(16, 16))
    ops.input("Cond", shape=(16, 16))
    ops.elementwise("where", out="Z", operands=("X", "Y", "Cond"))
    group = assemble(fuse(ops.build())[0])
    src = group.kernel.source
    assert group.bindings == ("X", "Z", "Y", "Cond")
    assert "Cond0_tile" in src
    assert "device const float* Cond" in src
    assert "?" in src


def test_assemble_1d_elementwise_uses_column_extent_one():
    ops = Operations()
    ops.input("X", shape=(128,))
    ops.elementwise("exp", out="Y", operands=("X",))
    group = assemble(fuse(ops.build())[0])
    assert group.dims == (128, 1)
    assert group.grid == (1, 8, 1)


def test_assemble_binary_same_input_does_not_duplicate_base_binding():
    ops = Operations()
    ops.input("X", shape=(16, 16))
    ops.elementwise("mul", out="Y", operands=("X", "X"))
    group = assemble(fuse(ops.build())[0])
    src = group.kernel.source
    assert group.bindings == ("X", "Y")
    assert src.count("device const float* X") == 1


def test_assemble_chain_reuses_in_chain_secondary_value():
    ops = Operations()
    ops.input("X", shape=(16, 16))
    ops.elementwise("exp", out="E", operands=("X",))
    ops.elementwise("add", out="Y", operands=("E", "E"))
    decision = next(
        d
        for d in fuse(ops.build())
        if d.chain_only and d.chain_only[-1].out.name == "Y"
    )
    group = assemble(decision)
    src = group.kernel.source
    assert group.bindings == ("E", "Y")
    assert src.count("device const float* E") == 1
    assert "Y0_tile" not in src
    assert (
        "float v0 = (X_tile[tile_row][tile_col]) + (X_tile[tile_row][tile_col]);" in src
    )


def test_assemble_secondary_view_uses_col_stride():
    ops = Operations()
    ops.input("X", shape=(4, 8))
    ops.input("Ybase", shape=(8, 4))
    ops.transpose("Ybase", out="Y")
    ops.elementwise("add", out="Z", operands=("X", "Y"))
    group = assemble(fuse(ops.build())[0])
    src = group.kernel.source
    assert group.bindings == ("X", "Z", "Ybase")
    assert "global_col * (4)" in src
