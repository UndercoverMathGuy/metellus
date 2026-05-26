"""Structural tests for the fuser.

Each test builds an IR program, runs `fuse`, and asserts which ops
landed in which vertex and what strategy `assemble` picks for them.
A small subset exercise `assemble` to confirm the generated MSL
contains the expected fragment markers (no GPU execution — that's
for the bench scripts).

Vertex shape vs. KernelGroup strategy: `fuse()` returns `Vertex`es
holding IR ops in program order. The pre/post-anchor split (which the
old fuser exposed as `prologue_a` / `epilogue` tuples) is reconstructed
by `_split_around_anchor` here for assertions. Strategy labels come
from `assemble(v).strategy`."""

from orchestrator import MatmulOp, Operations
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse
from orchestrator.ir import ElementwiseOp, ReductionOp
from orchestrator.kernel_group import FusionStrategy


def _op_types(vertex) -> list[str]:
    return [type(o).__name__ for o in vertex.ops]


def _split_around_anchor(vertex):
    """For a single-anchor vertex, returns (pre_anchor_elems,
    post_anchor_elems). Both are lists of ElementwiseOps in program
    order; the anchor itself is excluded. Returns (None, None) for
    elem-only or shape vertices."""
    anchor_idx = next(
        (
            i
            for i, op in enumerate(vertex.ops)
            if isinstance(op, (MatmulOp, ReductionOp))
        ),
        None,
    )
    if anchor_idx is None:
        return None, None
    pre = [op for op in vertex.ops[:anchor_idx] if isinstance(op, ElementwiseOp)]
    post = [op for op in vertex.ops[anchor_idx + 1 :] if isinstance(op, ElementwiseOp)]
    return pre, post


# ---------------------------------------------------------------------------
# Per-pattern vertex tests
# ---------------------------------------------------------------------------


def test_matmul_epilogue_tg_when_row_broadcast_present():
    # Shapes that pick the 32x32 matmul tile. The 64x64 tile + tg-tile
    # path exceeds the 32KB tg-memory budget even before any Y tile, so
    # the fuser would refuse the epilogue absorption there — leave the
    # shape on the small-tile path.
    ops = Operations()
    ops.input("A", shape=(32, 64))
    ops.input("B", shape=(64, 128))
    ops.input("bias", shape=(1, 128))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("add", out="D", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("max", out="Y", operands=("D", 0.0))
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    assert _op_types(v) == ["MatmulOp", "ElementwiseOp", "ElementwiseOp"]
    pre, post = _split_around_anchor(v)
    assert len(pre) == 0 and len(post) == 2
    assert assemble(v).strategy is FusionStrategy.MATMUL_EPILOGUE_TG


def test_matmul_epilogue_register_when_lane_agnostic_only():
    ops = Operations()
    ops.input("A", shape=(128, 64))
    ops.input("B", shape=(64, 128))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("max", out="Y", operands=("C", 0.0))  # relu — unary scalar
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    assert _op_types(v) == ["MatmulOp", "ElementwiseOp"]
    assert assemble(v).strategy is FusionStrategy.MATMUL_EPILOGUE_REGISTER


def test_elementwise_prologue_matmul_on_a():
    ops = Operations()
    ops.input("A", shape=(128, 64))
    ops.input("B", shape=(64, 128))
    ops.elementwise("max", out="A_relu", operands=("A", 0.0))  # prologue on A
    ops.matmul(a="A_relu", b="B", out="C")
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    pre, post = _split_around_anchor(v)
    assert len(pre) == 1 and pre[0].op == "max"
    assert len(post) == 0
    assert assemble(v).strategy is FusionStrategy.ELEMENTWISE_PROLOGUE_MATMUL


def test_elementwise_prologue_reduction():
    ops = Operations()
    ops.input("X", shape=(32, 64))
    ops.elementwise("exp", out="EX", operands=("X",))
    ops.reduction("sum", out="S", x="EX")
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    pre, post = _split_around_anchor(v)
    assert len(pre) == 1 and pre[0].op == "exp"
    assert assemble(v).strategy is FusionStrategy.ELEMENTWISE_PROLOGUE_REDUCTION


def test_reduction_epilogue_scalar_only():
    ops = Operations()
    ops.input("X", shape=(32, 64))
    ops.reduction("sum", out="S", x="X")
    ops.elementwise("mul", out="Y", operands=("S", 0.5))  # scalar binary on 1D
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    pre, post = _split_around_anchor(v)
    assert len(pre) == 0 and len(post) == 1
    assert assemble(v).strategy is FusionStrategy.REDUCTION_EPILOGUE


def test_elementwise_chain_standalone():
    ops = Operations()
    ops.input("X", shape=(32, 64))
    ops.elementwise("exp", out="E", operands=("X",))
    ops.elementwise("log", out="L", operands=("E",))
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    assert _op_types(v) == ["ElementwiseOp", "ElementwiseOp"]
    assert assemble(v).strategy is FusionStrategy.ELEMENTWISE_CHAIN


def test_standalone_kernels_when_no_fusion():
    ops = Operations()
    ops.input("A", shape=(64, 64))
    ops.input("B", shape=(64, 64))
    ops.matmul(a="A", b="B", out="C")
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    assert assemble(vertices[0]).strategy is FusionStrategy.STANDALONE_MATMUL


# ---------------------------------------------------------------------------
# Chain composition
# ---------------------------------------------------------------------------


def test_chained_prologue_and_epilogue_around_matmul():
    """elem → matmul → elem → elem all collapse into one kernel."""
    # Small-tile-eligible shapes so the tg-tile epilogue + row-broadcast
    # bias tile fit under the 32KB budget.
    ops = Operations()
    ops.input("A", shape=(32, 64))
    ops.input("B", shape=(64, 64))
    ops.input("bias", shape=(1, 64))
    ops.elementwise("exp", out="A_exp", operands=("A",))  # prologue on A
    ops.matmul(a="A_exp", b="B", out="C")
    ops.elementwise("add", out="D", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("max", out="Y", operands=("D", 0.0))
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    assert _op_types(v) == [
        "ElementwiseOp",
        "MatmulOp",
        "ElementwiseOp",
        "ElementwiseOp",
    ]
    pre, post = _split_around_anchor(v)
    assert len(pre) == 1 and pre[0].op == "exp"
    assert len(post) == 2
    assert assemble(v).strategy is FusionStrategy.MATMUL_EPILOGUE_TG


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
    vertices = fuse(ops.build())
    strategies = [assemble(v).strategy for v in vertices]
    assert FusionStrategy.STANDALONE_MATMUL in strategies
    elem_count = sum(
        1
        for s in strategies
        if s
        in (FusionStrategy.STANDALONE_ELEMENTWISE, FusionStrategy.ELEMENTWISE_CHAIN)
    )
    assert elem_count == 2


def test_multi_consumer_upstream_unary_fuses_into_both_and_elides():
    """Lane-agnostic elem fed into two matmuls should fuse into both
    AND be elided (the elem has no remaining standalone output)."""
    ops = Operations()
    ops.input("X", shape=(64, 64))
    ops.input("B1", shape=(64, 64))
    ops.input("B2", shape=(64, 64))
    ops.elementwise("max", out="Xr", operands=("X", 0.0))
    ops.matmul(a="Xr", b="B1", out="C1")
    ops.matmul(a="Xr", b="B2", out="C2")
    vertices = fuse(ops.build())
    matmul_vs = [v for v in vertices if any(isinstance(op, MatmulOp) for op in v.ops)]
    assert len(matmul_vs) == 2
    for v in matmul_vs:
        pre, _ = _split_around_anchor(v)
        assert len(pre) == 1 and pre[0].op == "max"
    elem_only = [
        v for v in vertices if all(isinstance(op, ElementwiseOp) for op in v.ops)
    ]
    assert elem_only == []


def test_multi_consumer_upstream_non_lane_agnostic_blocks_and_keeps_standalone():
    """Binary-tensor elem with a row broadcast is not lane-agnostic; the
    upstream carve-out can't apply, so the elem stays materialized
    (its own kernel) and neither matmul absorbs it."""
    ops = Operations()
    ops.input("X", shape=(64, 64))
    ops.input("bias", shape=(1, 64))
    ops.input("B1", shape=(64, 64))
    ops.input("B2", shape=(64, 64))
    ops.elementwise("add", out="Xb", operands=("X", "bias"), y_broadcast="row")
    ops.matmul(a="Xb", b="B1", out="C1")
    ops.matmul(a="Xb", b="B2", out="C2")
    vertices = fuse(ops.build())
    matmul_vs = [v for v in vertices if any(isinstance(op, MatmulOp) for op in v.ops)]
    assert len(matmul_vs) == 2
    for v in matmul_vs:
        pre, _ = _split_around_anchor(v)
        assert pre == []
        assert assemble(v).strategy is FusionStrategy.STANDALONE_MATMUL
    elem_only = [
        v for v in vertices if all(isinstance(op, ElementwiseOp) for op in v.ops)
    ]
    assert len(elem_only) == 1


# ---------------------------------------------------------------------------
# Multi-producer convergent + diamond shared-load
# ---------------------------------------------------------------------------


def test_multi_producer_convergent_fuses_into_one_kernel():
    """c1 = A @ B; c2 = C @ D; z = c1 + c2 → one multi-anchor kernel.
    Both anchors materialise into per-anchor C tiles; the merge runs
    via the tg-tile path; the convergent elem (c1, c2) is fully fused.

    Shapes sized so the fused multi-anchor kernel fits under
    `TGMEM_CAP_BYTES`. The structural assertion is what matters; n=32
    is the largest power-of-two that still fits two C tiles + the
    aliased A/B tiles in one kernel."""
    ops = Operations()
    ops.input("A", shape=(32, 32))
    ops.input("B", shape=(32, 32))
    ops.input("C", shape=(32, 32))
    ops.input("D", shape=(32, 32))
    ops.matmul(a="A", b="B", out="c1")
    ops.matmul(a="C", b="D", out="c2")
    ops.elementwise("add", out="z", operands=("c1", "c2"))
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    assert _op_types(v) == ["MatmulOp", "MatmulOp", "ElementwiseOp"]
    assert assemble(v).strategy is FusionStrategy.MULTI_PRODUCER_CONVERGENT


def test_diamond_shared_a_picks_diamond_shared_strategy():
    """c1 = A @ B1; c2 = A @ B2; z = c1 + c2. Both anchors read the
    same A. Multi-producer fuses them; classifier upgrades to
    DIAMOND_SHARED so the assembler emits one shared A-load.

    Shapes sized so the fused kernel fits under `TGMEM_CAP_BYTES`."""
    ops = Operations()
    ops.input("A", shape=(32, 32))
    ops.input("B1", shape=(32, 32))
    ops.input("B2", shape=(32, 32))
    ops.matmul(a="A", b="B1", out="c1")
    ops.matmul(a="A", b="B2", out="c2")
    ops.elementwise("add", out="z", operands=("c1", "c2"))
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    assert _op_types(v) == ["MatmulOp", "MatmulOp", "ElementwiseOp"]
    assert assemble(v).strategy is FusionStrategy.DIAMOND_SHARED


def test_assemble_multi_producer_has_two_accumulator_sets():
    # n=32 keeps the fused kernel under `TGMEM_CAP_BYTES`; the
    # structural assertions about accumulator/tile naming are
    # shape-independent.
    ops = Operations()
    ops.input("A", shape=(32, 32))
    ops.input("B", shape=(32, 32))
    ops.input("C", shape=(32, 32))
    ops.input("D", shape=(32, 32))
    ops.matmul(a="A", b="B", out="c1")
    ops.matmul(a="C", b="D", out="c2")
    ops.elementwise("add", out="z", operands=("c1", "c2"))
    group = assemble(fuse(ops.build())[0])
    src = group.kernel.source
    # Two suffixed accumulator sets must coexist.
    assert "matC00_0" in src and "matC00_1" in src
    # Two C tiles and a merge expression combining them.
    assert "C0_tile" in src and "C1_tile" in src
    # Bindings: A, B, C, D, z.
    assert group.bindings == ("A", "B", "C", "D", "z")
    # Both A loads present (not shared).
    assert src.count("threadgroup float A0_tile") == 1
    assert src.count("threadgroup float A1_tile") == 1


def test_assemble_diamond_shared_emits_single_a_tile():
    # n=32 keeps the fused kernel under `TGMEM_CAP_BYTES`.
    ops = Operations()
    ops.input("A", shape=(32, 32))
    ops.input("B1", shape=(32, 32))
    ops.input("B2", shape=(32, 32))
    ops.matmul(a="A", b="B1", out="c1")
    ops.matmul(a="A", b="B2", out="c2")
    ops.elementwise("add", out="z", operands=("c1", "c2"))
    group = assemble(fuse(ops.build())[0])
    src = group.kernel.source
    # One shared A tile, no per-anchor A.
    assert "threadgroup float A_tile" in src
    assert "threadgroup float A0_tile" not in src
    assert "threadgroup float A1_tile" not in src
    # Two B tiles, two accumulator sets, two C tiles, merge.
    assert "B0_tile" in src and "B1_tile" in src
    assert "matC00_0" in src and "matC00_1" in src
    assert "C0_tile" in src and "C1_tile" in src
    # Bindings: A, B0, B1, Z (only three input buffers since A is shared).
    assert group.bindings == ("A", "B1", "B2", "z")


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
    vertices = fuse(ops.build())
    strategies = {assemble(v).strategy for v in vertices}
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
    group = assemble(fuse(ops.build())[0])
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
    group = assemble(fuse(ops.build())[0])
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
    """`add(E, E)` reads E twice. Graph-level consumer counting is by
    vertex-membership (one consumer, not two), so the new fuser pulls
    `exp` into the chain — strictly better than the old behaviour,
    which left `exp` standalone and materialised E. Verify the chain
    is `[exp, add]` and E is fully internal (no extra binding, no
    secondary tile, and both reads in the `add` expression resolve to
    the in-chain value `v0 = exp(X_tile[...])`)."""
    ops = Operations()
    ops.input("X", shape=(16, 16))
    ops.elementwise("exp", out="E", operands=("X",))
    ops.elementwise("add", out="Y", operands=("E", "E"))
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    v = vertices[0]
    assert _op_types(v) == ["ElementwiseOp", "ElementwiseOp"]
    group = assemble(v)
    src = group.kernel.source
    # E is internal — only X (primary input) and Y (output) cross the boundary.
    assert group.bindings == ("X", "Y")
    assert "device const float* E" not in src
    assert "Y0_tile" not in src
    # exp computes once into v0; both add operands reference v0.
    assert "float v0 = exp((X_tile[tile_row][tile_col]));" in src
    assert "float v1 = (v0) + (v0);" in src


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
