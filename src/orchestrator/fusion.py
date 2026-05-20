"""Fusion decisions: IR Program → list of FusionDecision (one per kernel).

The fuser is an organizer. It does not emit MSL itself — it picks which
IR ops live in which kernel and which fragment composition strategy to
use. `kernel_group.assemble` then turns each decision into a compiled
KernelGroup.

Five fusion patterns, mirroring the fragment knobs the compute layer
already exposes:

  matmul → elem      (epilogue: register or tg-tile)
  elem → matmul      (prologue: TgLoadFragment.value_transform on A / B)
  elem → elem        (chains within any kernel; standalone group when no anchor)
  elem → reduc       (prologue: LastAxisReductionComputeFragment.value_transform)
  reduc → elem       (epilogue: StoreReductionResultFragment.value_transform)

Multi-consumer rule:

  * Downstream fusion (matmul→elem, reduc→elem, tail elem→elem) requires
    *unique consumer* of the producer's output. Otherwise the elem is
    not absorbed — we don't duplicate the producer in v0.
  * Upstream fusion (elem→matmul, elem→reduc) is allowed even when the
    elem has multiple consumers, provided the elem is **lane-agnostic**
    (unary or scalar-broadcast). Each consumer inlines the transform; the
    elem itself stays as its own kernel iff *some* consumer did not
    absorb it (so its materialized output is still needed elsewhere).
"""

from __future__ import annotations

from dataclasses import dataclass

from compute.elementwise.elementwise import (
    BINARY_EXPRESSIONS,
    BroadcastSpec,
    TERNARY_EXPRESSIONS,
    UNARY_EXPRESSIONS,
    elementwise_arity,
)
from compute.matmul.config import select_tile_config
from orchestrator.dag import DAG, build_dag
from orchestrator.ir import (
    ElementwiseOp,
    MatmulOp,
    Op,
    Program,
    ReductionOp,
    Scalar,
    ShapeOp,
    Tensor,
)
from orchestrator.kernel_group import FusionStrategy


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def _is_unary(op: ElementwiseOp) -> bool:
    return op.op in UNARY_EXPRESSIONS


def _is_binary(op: ElementwiseOp) -> bool:
    return op.op in BINARY_EXPRESSIONS


def _is_ternary(op: ElementwiseOp) -> bool:
    return op.op in TERNARY_EXPRESSIONS


def _is_lane_agnostic(op: ElementwiseOp) -> bool:
    """True iff every secondary operand is a `Scalar` (or there are none).
    Register-epilogue and upstream multi-consumer carve-out need this:
    the transform must not depend on lane→(row,col) layout."""
    if _is_ternary(op):
        return False
    for operand in op.operands[1:]:
        if not isinstance(operand, Scalar):
            return False
    return True


def _epilogue_eligible(elem: ElementwiseOp, primary: Tensor) -> bool:
    """An elem can sit in a matmul or reduction epilogue iff its primary
    matches the upstream output. v0 skips ternary (cond-tile plumbing is
    extra; not needed for the common cases)."""
    if _is_ternary(elem):
        return False
    return elem.operands[0] is primary or (
        isinstance(elem.operands[0], Tensor) and elem.operands[0].name == primary.name
    )


def _prologue_eligible(elem: ElementwiseOp) -> bool:
    """An elem can sit at the load side of a matmul/reduction iff it can
    be expressed as a value-transform on a single source element. v0:
    unary or binary (any broadcast — the load already exposes
    global_row/global_col so per-row/per-col bias indexing is free).
    Skip ternary."""
    return _is_unary(elem) or _is_binary(elem)


def _reduction_epilogue_eligible(elem: ElementwiseOp) -> bool:
    """Epilogue on a reduction's 1D output. The IR forces NONE broadcast
    on 1D primaries, so the only valid binary form is a Scalar second
    operand (constant literal). Skip non-Scalar binary in v0."""
    if _is_unary(elem):
        return True
    if _is_binary(elem):
        return isinstance(elem.operands[1], Scalar)
    return False


# ---------------------------------------------------------------------------
# FusionDecision
# ---------------------------------------------------------------------------


@dataclass
class FusionDecision:
    """One kernel-worth of fused work.

    Exactly one of `anchor` (a MatmulOp / ReductionOp) and
    `chain_only` (the list of elem ops in a standalone elementwise
    kernel) is populated.

    `prologue_a` / `prologue_b` are the elem chains feeding the anchor's
    inputs (B only for matmul). They are ordered nearest-to-anchor last
    (i.e. execution order: prologue[0] reads from device, prologue[-1]
    feeds the anchor's load).

    `epilogue` is the elem chain on the anchor's output, in execution
    order (epilogue[0] consumes the anchor's result, epilogue[-1]
    produces the final output).
    """

    strategy: FusionStrategy
    anchor: MatmulOp | ReductionOp | None = None
    prologue_a: tuple[ElementwiseOp, ...] = ()
    prologue_b: tuple[ElementwiseOp, ...] = ()
    epilogue: tuple[ElementwiseOp, ...] = ()
    chain_only: tuple[ElementwiseOp, ...] = ()
    shape_op: ShapeOp | None = None

    @property
    def ops(self) -> tuple[Op, ...]:
        """All IR ops absorbed by this decision, in program order."""
        if self.shape_op is not None:
            return (self.shape_op,)
        if self.chain_only:
            return self.chain_only
        assert self.anchor is not None
        return (
            *self.prologue_a,
            *self.prologue_b,
            self.anchor,
            *self.epilogue,
        )


# ---------------------------------------------------------------------------
# Fuser
# ---------------------------------------------------------------------------


def fuse(program: Program) -> tuple[FusionDecision, ...]:
    """Pick fusion decisions for `program`. Returns a tuple of decisions
    in program order; each decision is one KernelGroup-to-be."""
    dag = build_dag(program)
    # absorbed_into: id(elem) -> number of its IR consumers that have
    # absorbed the elem (either by inlining it as a prologue, or by
    # having it sit in the same downstream chain). When this count
    # equals the elem's consumer count, the elem is fully elided.
    absorbed_into: dict[int, int] = {}
    # Anchor ops (matmul/reduction) that have been folded into a
    # decision; tracked separately because they're never "elided" in the
    # multi-consumer sense — every anchor is its own kernel.
    anchored: set[int] = set()

    decisions: list[FusionDecision] = []
    consumer_counts: dict[int, int] = {
        id(op): len(dag.consumers_of.get(op.out.name, ())) for op in program
    }

    # Anchor pass: build decisions around each matmul and reduction.
    for op in program:
        if isinstance(op, MatmulOp) and id(op) not in anchored:
            decision = _build_matmul_decision(op, dag, absorbed_into)
            decisions.append(decision)
            anchored.add(id(op))
        elif isinstance(op, ReductionOp) and id(op) not in anchored:
            decision = _build_reduction_decision(op, dag, absorbed_into)
            decisions.append(decision)
            anchored.add(id(op))

    # Survivor pass: ops not fully absorbed go into standalone groups.
    handled: set[int] = set(anchored)
    for op in program:
        if id(op) in handled:
            continue
        if isinstance(op, ElementwiseOp):
            # Elide iff every consumer absorbed this elem. An elem with
            # zero consumers and absorbed >= 1 is also elided — the
            # absorbing anchor is producing the program output buffer.
            n_consumers = consumer_counts[id(op)]
            n_absorbed = absorbed_into.get(id(op), 0)
            if n_absorbed >= n_consumers and n_absorbed > 0:
                handled.add(id(op))
                continue
        # Build a standalone group / chain starting here.
        if isinstance(op, ElementwiseOp):
            chain = _extend_elementwise_chain(op, dag, handled)
            strategy = (
                FusionStrategy.ELEMENTWISE_CHAIN
                if len(chain) > 1
                else FusionStrategy.STANDALONE_ELEMENTWISE
            )
            decisions.append(FusionDecision(strategy=strategy, chain_only=chain))
            for ab in chain:
                handled.add(id(ab))
        elif isinstance(op, ShapeOp):
            decisions.append(
                FusionDecision(strategy=FusionStrategy.STANDALONE_SHAPE, shape_op=op)
            )
            handled.add(id(op))
        else:
            # Matmul / reduction without any fusion partners.
            assert False, "anchor pass should have handled this"

    # Topologically sort decisions by their LAST constituent op's program
    # position. A decision's chain can absorb an op at position P_a whose
    # operands are read at later position P_b (chain inputs that come from
    # other decisions). Sorting by min(P) would place such a decision
    # before its dependencies have run; max(P) puts it after the latest
    # op it absorbed, which by IR topo-ordering is after every kernel
    # that produces any input it touches.
    op_position = {id(op): i for i, op in enumerate(program)}
    decisions.sort(key=lambda d: max(op_position[id(o)] for o in d.ops))
    return tuple(decisions)


# ---------------------------------------------------------------------------
# Anchor-specific decision builders
# ---------------------------------------------------------------------------


def _build_matmul_decision(
    anchor: MatmulOp,
    dag: DAG,
    absorbed_into: dict[int, int],
) -> FusionDecision:
    prologue_a = _extend_prologue(anchor.a, dag, absorbed_into, anchor)
    prologue_b = _extend_prologue(anchor.b, dag, absorbed_into, anchor)
    epilogue = _extend_matmul_epilogue(anchor, dag, absorbed_into)

    # Strategy: register epilogue if every absorbed epilogue elem is
    # lane-agnostic AND there's no prologue forcing a tg-tile path. The
    # prologue does not actually force tg-tile (the loads always stage
    # through threadgroup tiles regardless), but the register epilogue
    # only makes sense when there IS an epilogue to apply in registers.
    if epilogue and all(_is_lane_agnostic(e) for e in epilogue):
        strategy = FusionStrategy.MATMUL_EPILOGUE_REGISTER
    elif epilogue or prologue_a or prologue_b:
        strategy = (
            FusionStrategy.MATMUL_EPILOGUE_TG
            if epilogue
            else FusionStrategy.ELEMENTWISE_PROLOGUE_MATMUL
        )
    else:
        strategy = FusionStrategy.STANDALONE_MATMUL
    return FusionDecision(
        strategy=strategy,
        anchor=anchor,
        prologue_a=prologue_a,
        prologue_b=prologue_b,
        epilogue=epilogue,
    )


def _build_reduction_decision(
    anchor: ReductionOp,
    dag: DAG,
    absorbed_into: dict[int, int],
) -> FusionDecision:
    prologue = _extend_prologue(anchor.input, dag, absorbed_into, anchor)
    epilogue = _extend_reduction_epilogue(anchor, dag, absorbed_into)
    if prologue and epilogue:
        # Both — prologue side already chose the strategy; epilogue is
        # additive on the store path. Pick a single label per the rule
        # that prologue wins when both exist (the prologue is the more
        # invasive fusion).
        strategy = FusionStrategy.ELEMENTWISE_PROLOGUE_REDUCTION
    elif prologue:
        strategy = FusionStrategy.ELEMENTWISE_PROLOGUE_REDUCTION
    elif epilogue:
        strategy = FusionStrategy.REDUCTION_EPILOGUE
    else:
        strategy = FusionStrategy.STANDALONE_REDUCTION
    return FusionDecision(
        strategy=strategy,
        anchor=anchor,
        prologue_a=prologue,
        epilogue=epilogue,
    )


# ---------------------------------------------------------------------------
# Chain extension helpers
# ---------------------------------------------------------------------------


def _extend_prologue(
    source: Tensor,
    dag: DAG,
    absorbed_into: dict[int, int],
    consumer_op: Op,
) -> tuple[ElementwiseOp, ...]:
    """Walk backward from `source`: while the producing op is an
    elementwise that's eligible to be inlined as a load-time
    value-transform AND has only one consumer (this anchor), absorb it.

    Unique-consumer is required at every step. The earlier multi-consumer
    "lane-agnostic carve-out" assumed each consumer could recompute the
    transform from the elem's primary input — but that breaks when the
    elem is downstream-absorbed by an upstream anchor (e.g. matmul +
    relu): the elem's primary input has no materialized buffer, and
    `anchor.input` is the already-transformed output. Re-applying the
    transform double-counts. Strict unique-consumer keeps prologue
    fusion sound at the cost of fewer fused chains in diamond patterns.

    Each absorbed elem increments `absorbed_into[id(elem)]` by 1 to
    record that the consuming anchor took it."""
    chain: list[ElementwiseOp] = []
    current = source
    while True:
        producer = dag.producer_of.get(current.name)
        if not isinstance(producer, ElementwiseOp):
            break
        if not _prologue_eligible(producer):
            break
        consumers = dag.consumers_of.get(producer.out.name, ())
        if len(consumers) != 1:
            break
        chain.append(producer)
        absorbed_into[id(producer)] = absorbed_into.get(id(producer), 0) + 1
        # Walk further upstream through this elem's primary operand.
        primary = producer.operands[0]
        if not isinstance(primary, Tensor):
            break
        current = primary
    chain.reverse()  # execution order: outermost-from-anchor first
    return tuple(chain)


def _extend_matmul_epilogue(
    anchor: MatmulOp,
    dag: DAG,
    absorbed_into: dict[int, int],
) -> tuple[ElementwiseOp, ...]:
    """Walk forward from the matmul's output: while the unique consumer
    is an eligible elem (and its primary is the matmul output / previous
    elem output), absorb it. Downstream fusion requires unique
    consumer — multi-consumer downstream is disallowed in v0.

    Threadgroup-memory budget: lane-agnostic-only chains compile to the
    register-epilogue path (no extra tg memory). Any non-lane-agnostic
    elem in the chain forces the tg-tile path, which declares C_tile
    plus a per-binary-tensor-operand Y tile. We size every candidate
    against Apple Silicon's 32KB tg-memory floor; absorption stops when
    the next elem would push the running total past the budget."""
    M, K = anchor.a.shape
    _, N = anchor.b.shape
    tile = select_tile_config(M, K, N)
    a_bytes = tile.tile_M * (tile.tile_K + tile.a_pad) * 4
    b_bytes = tile.tile_K * (tile.tile_N + tile.b_pad) * 4
    c_bytes = tile.tile_M * (tile.tile_N + tile.c_pad) * 4
    base_bytes = a_bytes + b_bytes  # always live during the mainloop
    BUDGET = 32 * 1024  # Apple Silicon's typical floor

    chain: list[ElementwiseOp] = []
    current_out = anchor.out
    y_tile_bytes_total = 0
    tg_tile_mode = False  # toggles on first non-lane-agnostic absorption
    while True:
        consumer = dag.unique_consumer(current_out.name)
        if not isinstance(consumer, ElementwiseOp):
            break
        if not _epilogue_eligible(consumer, current_out):
            break
        cand_tg_tile = tg_tile_mode or not _is_lane_agnostic(consumer)
        cand_y_bytes = y_tile_bytes_total + _epilogue_y_tile_bytes(
            consumer, tile.tile_M, tile.tile_N
        )
        cand_total = base_bytes + cand_y_bytes
        if cand_tg_tile:
            cand_total += c_bytes
        if cand_total > BUDGET:
            break
        tg_tile_mode = cand_tg_tile
        y_tile_bytes_total = cand_y_bytes
        chain.append(consumer)
        n_consumers = len(dag.consumers_of.get(consumer.out.name, ()))
        absorbed_into[id(consumer)] = max(n_consumers, 1)
        current_out = consumer.out
    return tuple(chain)


def _epilogue_y_tile_bytes(elem: ElementwiseOp, tile_M: int, tile_N: int) -> int:
    """Threadgroup memory cost (bytes) of the Y tile a tg-tile epilogue
    would declare for `elem`. Unary and Scalar-y ops cost zero; broadcast
    mode picks the tile shape for tensor operands."""
    if elementwise_arity(elem.op) != 2:
        return 0
    y = elem.operands[1]
    if not isinstance(y, Tensor):
        return 0
    if elem.y_broadcast is BroadcastSpec.NONE:
        return tile_M * tile_N * 4
    if elem.y_broadcast is BroadcastSpec.ROW:
        return tile_N * 4
    if elem.y_broadcast is BroadcastSpec.COL:
        return tile_M * 4
    if elem.y_broadcast is BroadcastSpec.SCALAR:
        return 4
    return 0


def _extend_reduction_epilogue(
    anchor: ReductionOp,
    dag: DAG,
    absorbed_into: dict[int, int],
) -> tuple[ElementwiseOp, ...]:
    """Walk forward from the reduction's 1D output. Same unique-consumer
    rule as matmul, but the eligibility predicate is tighter (only
    unary or Scalar-binary on 1D primaries)."""
    chain: list[ElementwiseOp] = []
    current_out = anchor.out
    while True:
        consumer = dag.unique_consumer(current_out.name)
        if not isinstance(consumer, ElementwiseOp):
            break
        if not _reduction_epilogue_eligible(consumer):
            break
        if consumer.operands[0] is not current_out and not (
            isinstance(consumer.operands[0], Tensor)
            and consumer.operands[0].name == current_out.name
        ):
            break
        chain.append(consumer)
        n_consumers = len(dag.consumers_of.get(consumer.out.name, ()))
        absorbed_into[id(consumer)] = max(n_consumers, 1)
        current_out = consumer.out
    return tuple(chain)


def _extend_elementwise_chain(
    head: ElementwiseOp,
    dag: DAG,
    absorbed: set[int],
) -> tuple[ElementwiseOp, ...]:
    """Standalone elementwise chain. Start at `head` (which itself is
    not absorbed); extend forward through unique-consumer elem links of
    matching shape. The chain stops at any non-elem, multi-consumer
    point, or ternary (cond plumbing not handled in elem-only kernels in
    v0)."""
    chain: list[ElementwiseOp] = [head]
    current_out = head.out
    while True:
        consumer = dag.unique_consumer(current_out.name)
        if not isinstance(consumer, ElementwiseOp):
            break
        if id(consumer) in absorbed:
            break
        if _is_ternary(consumer):
            break
        first_operand = consumer.operands[0]
        if not isinstance(first_operand, Tensor):
            break
        if first_operand.shape != current_out.shape:
            break
        if first_operand.name != current_out.name:
            break
        chain.append(consumer)
        current_out = consumer.out
    return tuple(chain)
