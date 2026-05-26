"""Fusion: IR Program → topo-ordered tuple of fused supervertices.

Replaces the previous greedy chain-walker. The new fuser:

  1. Decomposes the IR into a primitive `Graph` (one vertex per Op).
  2. Applies a priority-ordered set of rewrite passes over `graph.py`.
     Each pass matches a fusion pattern and collapses a subgraph into
     one supervertex via `Graph.replace`.
  3. Saturates: passes run in priority order, first match wins, restart
     at the top after each rewrite, until no pass fires.

Public entrypoint: `fuse(program) -> tuple[Vertex, ...]`. Each vertex
is one Metal kernel for `assembly.assemble()`.

The eligibility predicates (`_epilogue_eligible_matmul`,
`_prologue_eligible`, `_reduction_epilogue_eligible`,
`_is_lane_agnostic`) are carried over from the old fuser as the
policies that gate each pass — they say WHAT can fuse; the pass
priority says WHEN.

Pass priority (highest first):

  1. try_elem_chain_fuse        — elem → elem, unique consumer
  2. try_epilogue_absorb        — matmul/reduction → elem, unique consumer
  3. try_prologue_absorb_single — elem → matmul/reduction, unique consumer

Multi-consumer recompute, multi-producer convergent, and diamond
shared-load are future passes that slot into the same driver.

Note on chains: elem-chain-fuse runs at priority 1 so multi-step elem
chains become single vertices before any anchor sees them. That keeps
prologue/epilogue absorption a single rewrite per chain rather than
N iterative ones, and (when later passes look at chains-as-prologues)
the cost model has the whole chain in one place.
"""

from __future__ import annotations

from compute.elementwise.elementwise import (
    BINARY_EXPRESSIONS,
    BroadcastSpec,
    TERNARY_EXPRESSIONS,
    UNARY_EXPRESSIONS,
)
from orchestrator.aliasing import (
    TGMEM_CAP_BYTES,
    TgmemOverflowError,
    compute_group_budget,
)
from orchestrator.graph import Graph, Vertex
from orchestrator.ir import (
    ElementwiseOp,
    MatmulOp,
    Op,
    Program,
    ReductionOp,
    Scalar,
    Tensor,
)


# ---------------------------------------------------------------------------
# Eligibility predicates — the policies, preserved from the old fuser.
# These say WHAT can fuse; the pass priority says WHEN.
# ---------------------------------------------------------------------------


def _is_unary(op: ElementwiseOp) -> bool:
    return op.op in UNARY_EXPRESSIONS


def _is_binary(op: ElementwiseOp) -> bool:
    return op.op in BINARY_EXPRESSIONS


def _is_ternary(op: ElementwiseOp) -> bool:
    return op.op in TERNARY_EXPRESSIONS


def _is_lane_agnostic(op: ElementwiseOp) -> bool:
    """All secondary operands are Scalar (or arity 1) — the transform
    doesn't depend on lane→(row, col) layout. Gate for register
    epilogue and (future) multi-consumer recompute."""
    if _is_ternary(op):
        return False
    return all(isinstance(o, Scalar) for o in op.operands[1:])


def _epilogue_eligible_matmul(elem: ElementwiseOp, primary_name: str) -> bool:
    """An elem can sit in a matmul epilogue iff its primary input is
    the upstream output and it's not ternary. (Register vs tg-tile is
    chosen later from `_is_lane_agnostic`.)"""
    if _is_ternary(elem):
        return False
    primary = elem.operands[0]
    return isinstance(primary, Tensor) and primary.name == primary_name


def _prologue_eligible(elem: ElementwiseOp) -> bool:
    """An elem can be a matmul/reduction load-side prologue iff it's
    unary or binary (any broadcast). Skip ternary."""
    return _is_unary(elem) or _is_binary(elem)


def _reduction_epilogue_eligible(elem: ElementwiseOp) -> bool:
    """Reduction epilogue on a 1D output: unary or Scalar-binary
    (since the IR forces NONE broadcast on 1D primaries, non-Scalar
    binary has no valid shape and the elem can't ride on the scalar
    store path)."""
    if _is_unary(elem):
        return True
    if _is_binary(elem):
        return isinstance(elem.operands[1], Scalar)
    return False


# ---------------------------------------------------------------------------
# Vertex shape predicates
# ---------------------------------------------------------------------------


def _anchor_of(vertex: Vertex) -> Op | None:
    """The single matmul or reduction op in `vertex`, or None if the
    vertex has no anchor (elem-only or shape-only)."""
    for op in vertex.ops:
        if isinstance(op, (MatmulOp, ReductionOp)):
            return op
    return None


def _is_elementwise_vertex(vertex: Vertex) -> bool:
    """True iff every op in `vertex` is an ElementwiseOp."""
    return len(vertex.ops) > 0 and all(
        isinstance(op, ElementwiseOp) for op in vertex.ops
    )


# ---------------------------------------------------------------------------
# Pass implementations — each returns the rewritten graph on first match
# or None when no match exists. The driver applies them in priority
# order, restarting from the top after each rewrite.
# ---------------------------------------------------------------------------


def try_elem_chain_fuse(
    graph: Graph, blacklist: frozenset[frozenset[int]]
) -> Graph | None:
    """elem-vertex U → elem-vertex V where U's tail output is V's head
    primary, V's head is not ternary, and U's output has exactly one
    consumer (V). Collapses (U, V) into one elem-chain vertex."""
    for u in graph.vertices:
        if not _is_elementwise_vertex(u):
            continue
        u_tail_out = u.ops[-1].out.name
        if u_tail_out not in u.outputs:
            continue
        consumers = graph.consumers_of(u_tail_out)
        if len(consumers) != 1:
            continue
        v = consumers[0]
        if not _is_elementwise_vertex(v):
            continue
        v_head = v.ops[0]
        assert isinstance(v_head, ElementwiseOp)
        if _is_ternary(v_head):
            continue
        primary = v_head.operands[0]
        if not isinstance(primary, Tensor) or primary.name != u_tail_out:
            continue
        if (u.op_ids | v.op_ids) in blacklist:
            continue
        return graph.replace(absorbed=(u, v), ops=u.ops + v.ops)
    return None


def try_epilogue_absorb(
    graph: Graph, blacklist: frozenset[frozenset[int]]
) -> Graph | None:
    """anchor-vertex A → elem-vertex V where A's anchor-side tail
    output is V's head primary, V's head is epilogue-eligible (matmul
    or reduction predicate), and V is the unique consumer. Collapses
    (A, V) into one vertex with V's ops appended."""
    for a in graph.vertices:
        anchor = _anchor_of(a)
        if anchor is None:
            continue
        a_tail_out = a.ops[-1].out.name
        if a_tail_out not in a.outputs:
            continue
        consumers = graph.consumers_of(a_tail_out)
        if len(consumers) != 1:
            continue
        v = consumers[0]
        if not _is_elementwise_vertex(v):
            continue
        v_head = v.ops[0]
        assert isinstance(v_head, ElementwiseOp)
        if isinstance(anchor, MatmulOp):
            if not _epilogue_eligible_matmul(v_head, a_tail_out):
                continue
        else:  # ReductionOp
            if not _reduction_epilogue_eligible(v_head):
                continue
            primary = v_head.operands[0]
            if not (isinstance(primary, Tensor) and primary.name == a_tail_out):
                continue
        if (a.op_ids | v.op_ids) in blacklist:
            continue
        return graph.replace(absorbed=(a, v), ops=a.ops + v.ops)
    return None


def try_prologue_absorb_single(
    graph: Graph, blacklist: frozenset[frozenset[int]]
) -> Graph | None:
    """elem-vertex V → anchor-vertex A where V is A's unique consumer,
    every op in V is prologue-eligible, and V's tail output feeds one
    of A's anchor inputs (walking through any prologue chains already
    absorbed into A)."""
    for v in graph.vertices:
        if not _is_elementwise_vertex(v):
            continue
        v_tail_out = v.ops[-1].out.name
        if v_tail_out not in v.outputs:
            continue
        consumers = graph.consumers_of(v_tail_out)
        if len(consumers) != 1:
            continue
        a = consumers[0]
        # Collect boundary names from every anchor in the consumer vertex
        # so multi-anchor consumers (e.g. a multi-producer fuse already
        # collapsed [matmul, matmul, merge]) can still absorb a prologue
        # feeding either anchor.
        anchors_in_a = [op for op in a.ops if isinstance(op, (MatmulOp, ReductionOp))]
        if not anchors_in_a:
            continue
        boundary_names: set[str] = set()
        for anchor in anchors_in_a:
            boundary_names |= _anchor_boundary_input_names(anchor, a)
        if v_tail_out not in boundary_names:
            continue
        if not all(
            _prologue_eligible(op) for op in v.ops if isinstance(op, ElementwiseOp)
        ):
            continue
        if (v.op_ids | a.op_ids) in blacklist:
            continue
        # Merge in program order — prologues from different sides of
        # the same anchor can interleave with the anchor's already-
        # absorbed prologue/epilogue ops by IR position.
        merged = tuple(
            op
            for _, op in sorted((graph.op_index[id(op)], op) for op in (*v.ops, *a.ops))
        )
        return graph.replace(absorbed=(v, a), ops=merged)
    return None


def try_multi_producer_fuse(
    graph: Graph, blacklist: frozenset[frozenset[int]]
) -> Graph | None:
    """Convergent elem with two matmul-anchor producers, matching
    (M, N). Fuses [matmul_a, matmul_b, merge_elem] into one
    multi-anchor vertex. Must run before `try_epilogue_absorb`
    otherwise the elem gets stolen as a single-producer epilogue.

    Diamond shared-load (both anchors read the same A) is detected
    automatically by `assembly._classify` after this pass produces the
    multi-anchor vertex — the strategy flips to `DIAMOND_SHARED` and
    the assembler emits one shared A-load instead of two.

    v0 caps:
      - Producer vertices must be pure (one MatmulOp, no absorbed
        prologue/epilogue). Combining with prologue absorption needs
        per-anchor prologue tracing in the multi-anchor assembler,
        which is a later phase.
      - Convergent elem is binary, both operands Tensor, full-shape
        broadcast (`y_broadcast=NONE`)."""
    for v in graph.vertices:
        if not _is_elementwise_vertex(v) or len(v.ops) != 1:
            continue
        elem = v.ops[0]
        assert isinstance(elem, ElementwiseOp)
        if not _is_binary(elem) or elem.y_broadcast is not BroadcastSpec.NONE:
            continue
        x, y = elem.operands[0], elem.operands[1]
        if not (isinstance(x, Tensor) and isinstance(y, Tensor)):
            continue
        prod_x = graph.producer_of(x.name)
        prod_y = graph.producer_of(y.name)
        if prod_x is None or prod_y is None or prod_x is prod_y:
            continue
        # v0: pure matmul producer vertices only (no absorbed pro/epilogue).
        if len(prod_x.ops) != 1 or len(prod_y.ops) != 1:
            continue
        anchor_x = prod_x.ops[0]
        anchor_y = prod_y.ops[0]
        if not (isinstance(anchor_x, MatmulOp) and isinstance(anchor_y, MatmulOp)):
            continue
        # Unique-consumer at each producer (otherwise downstream
        # multi-consumer would force materialisation of the producer).
        if len(graph.consumers_of(x.name)) != 1:
            continue
        if len(graph.consumers_of(y.name)) != 1:
            continue
        if anchor_x.out.shape != anchor_y.out.shape:
            continue
        if (prod_x.op_ids | prod_y.op_ids | v.op_ids) in blacklist:
            continue
        merged = tuple(
            op
            for _, op in sorted(
                (graph.op_index[id(op)], op)
                for op in (*prod_x.ops, *prod_y.ops, *v.ops)
            )
        )
        return graph.replace(absorbed=(prod_x, prod_y, v), ops=merged)
    return None


def _should_recompute(elem: ElementwiseOp) -> bool:
    """v0 cost model: recompute iff lane-agnostic — i.e. the elem's
    per-element value depends only on its primary input at that lane.
    Each consumer can then recompute it using just the primary, which
    they were going to load anyway. For non-lane-agnostic elems
    (binary with tensor y, ternary), each consumer would have to
    additionally load the secondary operand at every position — defer
    to a bandwidth-based cost model before enabling those."""
    return _is_lane_agnostic(elem)


def try_prologue_absorb_multi(
    graph: Graph, blacklist: frozenset[frozenset[int]]
) -> Graph | None:
    """Multi-consumer recompute: an elem with ≥2 anchor consumers,
    all of which admit it as a prologue, and the cost model says
    recompute. Inline the elem's op into every consumer's prologue
    via `absorb_into`; elide the standalone elem vertex if no other
    readers remain.

    v0: single-op elem vertices only (no chains). Chain support
    requires extending the cost model to whole chains and verifying
    every link is recompute-safe."""
    for v in graph.vertices:
        if not _is_elementwise_vertex(v) or len(v.ops) != 1:
            continue
        elem = v.ops[0]
        assert isinstance(elem, ElementwiseOp)
        if not _prologue_eligible(elem) or not _should_recompute(elem):
            continue
        elem_out = elem.out.name
        if elem_out not in v.outputs:
            continue
        consumers = list(graph.consumers_of(elem_out))
        if len(consumers) < 2:
            continue
        # Every consumer must be an anchor vertex admitting `elem` as a prologue.
        collected_ids: list[int] = []
        bailed = False
        for c in consumers:
            ac = _anchor_of(c)
            if ac is None or elem_out not in _anchor_boundary_input_names(ac, c):
                bailed = True
                break
            collected_ids.append(id(ac))
        if bailed:
            continue
        anchor_ids = collected_ids
        # Signature for the whole fan-out: elem + every consumer vertex
        # it's being inlined into. Blacklisting this signature disables
        # the entire recompute event (not per-consumer); a future pass
        # could enable partial recompute, but v0 treats it atomically.
        recompute_sig = v.op_ids | frozenset(oid for c in consumers for oid in c.op_ids)
        if recompute_sig in blacklist:
            continue
        # Fold elem into every consumer; track via anchor identity since
        # vertex identity shifts across each absorb_into.
        new_graph = graph
        for anchor_id in anchor_ids:
            current = next(
                cv
                for cv in new_graph.vertices
                if any(id(op) == anchor_id for op in cv.ops)
            )
            new_graph = new_graph.absorb_into(current, v.ops)
        # `v` was never touched by absorb_into → preserved by identity.
        still_needed = elem_out in graph.program_outputs or any(
            elem_out in other.inputs for other in new_graph.vertices if other is not v
        )
        if not still_needed:
            new_graph = new_graph.remove_vertex(v)
        return new_graph
    return None


def _anchor_boundary_input_names(anchor: Op, anchor_vertex: Vertex) -> set[str]:
    """Tensor names that, when produced by a prologue chain, can absorb
    into `anchor`'s load value_transform.

    For a fresh anchor vertex (no prologue absorbed), these are
    `anchor.a` / `anchor.b` (matmul) or `anchor.input` (reduction).
    When a prologue chain has already been absorbed into the vertex on
    one of those inputs, the boundary moves — it's now the head
    primary of that absorbed chain (the actual device-load source).
    We walk back through pre-anchor ops in the vertex to find it."""
    if isinstance(anchor, MatmulOp):
        candidates = [anchor.a.name, anchor.b.name]
    elif isinstance(anchor, ReductionOp):
        candidates = [anchor.input.name]
    else:
        return set()
    anchor_idx = anchor_vertex.ops.index(anchor)
    pre = anchor_vertex.ops[:anchor_idx]
    result: set[str] = set()
    for name in candidates:
        current = name
        while True:
            producer = next((op for op in pre if op.out.name == current), None)
            if producer is None or not isinstance(producer, ElementwiseOp):
                break
            primary = producer.operands[0]
            if not isinstance(primary, Tensor):
                break
            current = primary.name
        result.add(current)
    return result


# ---------------------------------------------------------------------------
# Driver — priority-ordered saturation
# ---------------------------------------------------------------------------


_PASSES = (
    try_elem_chain_fuse,
    try_multi_producer_fuse,  # before epilogue_absorb — otherwise it'd steal the merge
    try_epilogue_absorb,
    try_prologue_absorb_single,
    try_prologue_absorb_multi,
)


def _check_primitive_budgets(graph: Graph) -> None:
    """Verify every primitive (pre-fusion) vertex fits under the
    tgmem cap. A vertex that overflows on its own can't be rescued by
    fusion — fusion only grows kernels. Raise upfront with the
    offending op named so callers can shrink the tile shape or pick
    a different strategy before we waste cycles on doomed fusion."""
    from orchestrator.assembly import assemble

    for v in graph.vertices:
        budget = compute_group_budget(assemble(v))
        if budget.size_bytes > TGMEM_CAP_BYTES:
            op_names = ", ".join(f"{type(op).__name__}({op.out.name})" for op in v.ops)
            raise TgmemOverflowError(
                f"primitive vertex [{op_names}] needs {budget.size_bytes}B "
                f"of threadgroup memory; cap is {TGMEM_CAP_BYTES}B. No "
                f"fusion can shrink a single op — reduce its tile shape "
                f"or pick a register-resident strategy."
            )


def _fits_budget(old: Graph, new: Graph) -> bool:
    """True iff every vertex that's new in `new` (relative to `old`,
    by Python identity) fits under the tgmem cap. Unchanged vertices
    are skipped — their footprints were already vetted on a prior
    iteration. Identity works because every rewrite (`replace`,
    `absorb_into`) constructs fresh `Vertex` objects for anything
    it modifies."""
    from orchestrator.assembly import assemble

    old_ids = {id(v) for v in old.vertices}
    for v in new.vertices:
        if id(v) in old_ids:
            continue
        if compute_group_budget(assemble(v)).size_bytes > TGMEM_CAP_BYTES:
            return False
    return True


def _modified_signature(old: Graph, new: Graph) -> frozenset[int]:
    """Op-id signature for the rewrite that produced `new` from `old`.
    Union of op_ids over vertices that are new in `new` — same key the
    passes test against the blacklist, so a returned straw signature
    plugs straight back into the next `_fuse_once` call."""
    old_ids = {id(v) for v in old.vertices}
    sig: frozenset[int] = frozenset()
    for v in new.vertices:
        if id(v) not in old_ids:
            sig = sig | v.op_ids
    return sig


def _fuse_once(
    program: Program, blacklist: frozenset[frozenset[int]]
) -> tuple[tuple[Vertex, ...], frozenset[int] | None]:
    """One deterministic fusion run with `blacklist` as the only state
    carried in from prior runs. Saturates the pass set in priority
    order; after each rewrite, checks the modified vertices against
    the tgmem cap.

    Returns `(vertices, None)` on a clean saturation (all fusions
    that fired fit under cap, no more passes match). Returns
    `(vertices, straw)` when a fusion just pushed a vertex over the
    cap — `straw` is the signature the outer loop should add to its
    blacklist before retrying. The returned vertices in the straw
    case include the offending fusion; callers shouldn't use them."""
    graph = Graph.from_program(program)
    _check_primitive_budgets(graph)
    while True:
        for pass_fn in _PASSES:
            new_graph = pass_fn(graph, blacklist)
            if new_graph is None:
                continue
            if not _fits_budget(graph, new_graph):
                return new_graph.topo_sort(), _modified_signature(graph, new_graph)
            graph = new_graph
            break
        else:
            return graph.topo_sort(), None


def fuse(program: Program) -> tuple[Vertex, ...]:
    """Build the primitive graph, saturate the pass set in priority
    order, return the resulting vertices in topo order. Each vertex
    is one Metal kernel for `assembly.assemble()`.

    Budget-gated: any fusion that would push a vertex over
    `TGMEM_CAP_BYTES` is identified as the straw, its op-id
    signature is blacklisted, and fusion is re-run deterministically
    from the primitive graph. The blacklist grows by one entry per
    overflow until the run saturates cleanly.

    The deterministic re-run is the entire mechanism — passes
    enumerate matches in a stable order, and a blacklist hit just
    `continue`s past the offending match to the next legal one.
    Convergence is guaranteed: the fusion search space is finite
    (≤ O(N²) signatures over N ops), so the blacklist can only grow
    so far. In practice, 0–2 outer iterations."""
    blacklist: frozenset[frozenset[int]] = frozenset()
    while True:
        vertices, straw = _fuse_once(program, blacklist)
        if straw is None:
            return vertices
        blacklist = blacklist | {straw}
