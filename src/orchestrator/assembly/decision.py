"""Classification: Vertex → DecisionView.

Decomposes a vertex's flat program-ordered ops into the structural
components each per-strategy template consumes (anchor, prologue_a,
prologue_b, epilogue, etc.). Pure function of `vertex.ops`.

Multi-anchor (multi-producer / diamond) decisions populate `anchors`
and `merge_elem` instead of `anchor`/`epilogue`. The `shared_a` flag
indicates the diamond case where both anchors consume the same A
tensor with matching K — assembler emits one shared A-load instead of
two.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.graph import Vertex
from orchestrator.ir import (
    ElementwiseOp,
    MatmulOp,
    Op,
    ReductionOp,
    Scalar,
    ShapeOp,
    Tensor,
)
from orchestrator.kernel_group import FusionStrategy


@dataclass
class DecisionView:
    strategy: FusionStrategy
    anchor: MatmulOp | ReductionOp | None = None
    prologue_a: tuple[ElementwiseOp, ...] = ()
    prologue_b: tuple[ElementwiseOp, ...] = ()
    epilogue: tuple[ElementwiseOp, ...] = ()
    chain_only: tuple[ElementwiseOp, ...] = ()
    shape_op: ShapeOp | None = None
    ops: tuple[Op, ...] = ()
    # Multi-anchor fields (v0: exactly 2 anchors).
    anchors: tuple[MatmulOp, ...] = ()
    merge_elem: ElementwiseOp | None = None
    shared_a: bool = False
    # Per-anchor prologue chains (parallel to `anchors`). Each entry is
    # the elementwise chain feeding that anchor's A or B input,
    # traced via `trace_prologue` over pre-anchor ops.
    prologues_a: tuple[tuple[ElementwiseOp, ...], ...] = ()
    prologues_b: tuple[tuple[ElementwiseOp, ...], ...] = ()


def classify(vertex: Vertex) -> DecisionView:
    ops = vertex.ops

    shape_op = next((op for op in ops if isinstance(op, ShapeOp)), None)
    if shape_op is not None:
        assert len(ops) == 1, "ShapeOp must be standalone (no fusion in v0)"
        return DecisionView(
            strategy=FusionStrategy.STANDALONE_SHAPE,
            shape_op=shape_op,
            ops=ops,
        )

    matmul_anchors = tuple(op for op in ops if isinstance(op, MatmulOp))
    reduction_anchors = tuple(op for op in ops if isinstance(op, ReductionOp))

    # Multi-anchor case (multi-producer / diamond): 2+ matmul anchors
    # followed by a convergent merge elem. v0 supports exactly 2,
    # optionally with per-anchor prologue chains absorbed in.
    if len(matmul_anchors) >= 2:
        assert len(matmul_anchors) == 2 and not reduction_anchors, (
            "v0: multi-anchor vertex must have exactly 2 matmul anchors "
            "and no reduction anchors"
        )
        a0, a1 = matmul_anchors
        a0_idx = ops.index(a0)
        a1_idx = ops.index(a1)
        last_anchor_idx = max(a0_idx, a1_idx)

        # The merge elem is the single op following both anchors. Anything
        # before either anchor must be an elementwise prologue feeding it.
        post_anchors_ops = ops[last_anchor_idx + 1 :]
        assert len(post_anchors_ops) == 1 and isinstance(
            post_anchors_ops[0], ElementwiseOp
        ), "v0: multi-anchor vertex has exactly one merge elem"
        merge_elem = post_anchors_ops[0]

        # Each anchor traces its own prologue chains. trace_prologue
        # filters by name match through `candidates`, so it naturally
        # ignores ops that don't feed this anchor.
        pre_a0 = tuple(op for op in ops[:a0_idx] if isinstance(op, ElementwiseOp))
        pre_a1 = tuple(op for op in ops[:a1_idx] if isinstance(op, ElementwiseOp))
        prologues_a = (
            trace_prologue(a0.a.name, pre_a0),
            trace_prologue(a1.a.name, pre_a1),
        )
        prologues_b = (
            trace_prologue(a0.b.name, pre_a0),
            trace_prologue(a1.b.name, pre_a1),
        )

        absorbed_ids = {
            id(op) for chain in (*prologues_a, *prologues_b) for op in chain
        }
        for op in ops:
            if isinstance(op, ElementwiseOp) and op is not merge_elem:
                assert id(op) in absorbed_ids, (
                    f"v0: pre-anchor elementwise op {op.out.name!r} "
                    "not absorbed by any anchor prologue"
                )

        # Shared-A optimization only fires when neither anchor has an A
        # prologue (since differing prologues mean differing load
        # transforms and we can't share the A_tile).
        shared_a = (
            a0.a.name == a1.a.name
            and a0.a.shape == a1.a.shape
            and not prologues_a[0]
            and not prologues_a[1]
        )
        strategy = (
            FusionStrategy.DIAMOND_SHARED
            if shared_a
            else FusionStrategy.MULTI_PRODUCER_CONVERGENT
        )
        return DecisionView(
            strategy=strategy,
            anchors=matmul_anchors,
            merge_elem=merge_elem,
            shared_a=shared_a,
            prologues_a=prologues_a,
            prologues_b=prologues_b,
            ops=ops,
        )

    anchor = next((op for op in ops if isinstance(op, (MatmulOp, ReductionOp))), None)
    if anchor is None:
        assert all(isinstance(op, ElementwiseOp) for op in ops)
        chain: tuple[ElementwiseOp, ...] = tuple(
            op for op in ops if isinstance(op, ElementwiseOp)
        )
        strategy = (
            FusionStrategy.ELEMENTWISE_CHAIN
            if len(chain) > 1
            else FusionStrategy.STANDALONE_ELEMENTWISE
        )
        return DecisionView(strategy=strategy, chain_only=chain, ops=ops)

    anchor_idx = ops.index(anchor)
    pre = ops[:anchor_idx]
    post = tuple(ops[anchor_idx + 1 :])
    epilogue: tuple[ElementwiseOp, ...] = tuple(
        op for op in post if isinstance(op, ElementwiseOp)
    )

    if isinstance(anchor, MatmulOp):
        prologue_a = trace_prologue(anchor.a.name, pre)
        prologue_b = trace_prologue(anchor.b.name, pre)
        if epilogue and all(is_lane_agnostic_pred(e) for e in epilogue):
            strategy = FusionStrategy.MATMUL_EPILOGUE_REGISTER
        elif epilogue:
            strategy = FusionStrategy.MATMUL_EPILOGUE_TG
        elif prologue_a or prologue_b:
            strategy = FusionStrategy.ELEMENTWISE_PROLOGUE_MATMUL
        else:
            strategy = FusionStrategy.STANDALONE_MATMUL
        return DecisionView(
            strategy=strategy,
            anchor=anchor,
            prologue_a=prologue_a,
            prologue_b=prologue_b,
            epilogue=epilogue,
            ops=ops,
        )

    # ReductionOp anchor
    prologue = trace_prologue(anchor.input.name, pre)
    if prologue:
        strategy = FusionStrategy.ELEMENTWISE_PROLOGUE_REDUCTION
    elif epilogue:
        strategy = FusionStrategy.REDUCTION_EPILOGUE
    else:
        strategy = FusionStrategy.STANDALONE_REDUCTION
    return DecisionView(
        strategy=strategy,
        anchor=anchor,
        prologue_a=prologue,
        epilogue=epilogue,
        ops=ops,
    )


def trace_prologue(
    target_name: str, candidates: tuple[Op, ...]
) -> tuple[ElementwiseOp, ...]:
    """Walk back from `target_name` through pre-anchor elementwise ops
    in `candidates`. Returns the chain in execution order
    (outermost-from-anchor first — the chain head reads from device,
    the chain tail feeds the anchor's load)."""
    chain: list[ElementwiseOp] = []
    current = target_name
    while True:
        producer = next((op for op in candidates if op.out.name == current), None)
        if not isinstance(producer, ElementwiseOp):
            break
        chain.append(producer)
        primary = producer.operands[0]
        if not isinstance(primary, Tensor):
            break
        current = primary.name
    chain.reverse()
    return tuple(chain)


def is_lane_agnostic_pred(op: ElementwiseOp) -> bool:
    """Inlined copy of `fusion._is_lane_agnostic` — kept local so the
    register-vs-tg-tile decision lives next to the strategy choice
    that consumes it, and so this module has no fusion-internals
    dependency beyond what's strictly needed."""
    from compute.elementwise.elementwise import TERNARY_EXPRESSIONS

    if op.op in TERNARY_EXPRESSIONS:
        return False
    return all(isinstance(o, Scalar) for o in op.operands[1:])
