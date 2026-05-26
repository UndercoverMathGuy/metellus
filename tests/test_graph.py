"""Tests for the vertex-level rewriting substrate (`orchestrator.graph`).

`from_program` decomposes IR into one vertex per Op. `replace`
collapses subsets into supervertices with computed boundaries.
`topo_sort` / `flatten` give deterministic execution orders.
`check_dependencies` is the canonical soundness check against the
original IR — every rewrite is verified by it."""

import pytest

from orchestrator import Operations
from orchestrator.graph import Graph, Vertex
from orchestrator.ir import MatmulOp


def _simple_chain():
    ops = Operations()
    ops.input("a", shape=(8, 16))
    ops.input("b", shape=(16, 4))
    ops.input("bias", shape=(1, 4))
    ops.matmul(a="a", b="b", out="c")
    ops.elementwise("add", out="d", operands=("c", "bias"), y_broadcast="row")
    ops.elementwise("exp", out="e", operands=("d",))
    ops.reduction("sum", out="r", x="e")
    return ops.build()


def _by_out_name(graph: Graph) -> dict[str, Vertex]:
    """Helper: index single-op vertices by their op's output name."""
    return {v.ops[0].out.name: v for v in graph.vertices if v.is_primitive}


# ---------------------------------------------------------------------------
# from_program
# ---------------------------------------------------------------------------


def test_from_program_one_vertex_per_op():
    program = _simple_chain()
    graph = Graph.from_program(program)
    assert len(graph.vertices) == len(program)
    assert all(v.is_primitive for v in graph.vertices)


def test_from_program_boundary_for_chain():
    program = _simple_chain()
    graph = Graph.from_program(program)
    by_name = _by_out_name(graph)

    # matmul c: reads a, b; output consumed by d.
    assert by_name["c"].inputs == ("a", "b")
    assert by_name["c"].outputs == ("c",)
    # add d: reads c, bias; output consumed by e.
    assert set(by_name["d"].inputs) == {"c", "bias"}
    assert by_name["d"].outputs == ("d",)
    # exp e: reads d; output consumed by r.
    assert by_name["e"].inputs == ("d",)
    assert by_name["e"].outputs == ("e",)
    # reduction r: reads e; program sink → still materialized.
    assert by_name["r"].inputs == ("e",)
    assert by_name["r"].outputs == ("r",)


def test_from_program_default_program_outputs_are_sinks():
    graph = Graph.from_program(_simple_chain())
    assert graph.program_outputs == frozenset({"r"})


def test_from_program_explicit_program_outputs_override_default():
    program = _simple_chain()
    # Force an intermediate to also be materialized.
    graph = Graph.from_program(program, program_outputs=frozenset({"d", "r"}))
    assert graph.program_outputs == frozenset({"d", "r"})


def test_from_program_dead_output_has_empty_outputs_tuple():
    # When a producer's output is neither consumed nor in program_outputs,
    # its vertex has outputs == (). Synthesize by passing empty outputs.
    program = _simple_chain()
    graph = Graph.from_program(program, program_outputs=frozenset())
    by_name = _by_out_name(graph)
    # r is the only true sink; with no program_outputs, it's invisible.
    assert by_name["r"].outputs == ()
    # Internal vertices that feed someone are still materialized.
    assert by_name["c"].outputs == ("c",)


def test_from_program_view_appears_as_external_input():
    # Transpose / reshape views have no producing IR Op; consumers see
    # the view name as an external input (matches dag.py's behaviour).
    ops = Operations()
    ops.input("X", shape=(4, 8))
    ops.input("Ybase", shape=(8, 4))
    ops.transpose("Ybase", out="Yview")
    ops.elementwise("add", out="Z", operands=("X", "Yview"))
    graph = Graph.from_program(ops.build())
    assert len(graph.vertices) == 1  # only the elementwise op is in IR
    z = graph.vertices[0]
    assert set(z.inputs) == {"X", "Yview"}
    assert z.outputs == ("Z",)


# ---------------------------------------------------------------------------
# Vertex properties
# ---------------------------------------------------------------------------


def test_vertex_is_primitive_iff_single_op():
    program = _simple_chain()
    graph = Graph.from_program(program)
    for v in graph.vertices:
        assert v.is_primitive
    # After replace, the supervertex is not primitive.
    d = _by_out_name(graph)["d"]
    e = _by_out_name(graph)["e"]
    after = graph.replace(absorbed=(d, e), ops=(d.ops[0], e.ops[0]))
    super_v = next(v for v in after.vertices if len(v.ops) == 2)
    assert not super_v.is_primitive


def test_vertex_op_ids_matches_contained_ops():
    program = _simple_chain()
    graph = Graph.from_program(program)
    for v in graph.vertices:
        assert v.op_ids == frozenset(id(op) for op in v.ops)


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------


def test_replace_collapses_two_adjacent_vertices():
    program = _simple_chain()
    graph = Graph.from_program(program)
    by_name = _by_out_name(graph)
    d, e = by_name["d"], by_name["e"]
    after = graph.replace(absorbed=(d, e), ops=(d.ops[0], e.ops[0]))

    assert len(after.vertices) == len(graph.vertices) - 1
    super_v = next(v for v in after.vertices if len(v.ops) == 2)
    # Inputs: c (from add) and bias; d is produced internally.
    assert set(super_v.inputs) == {"c", "bias"}
    # Outputs: e (consumed by r). d is internal.
    assert super_v.outputs == ("e",)


def test_replace_keeps_internal_output_when_it_is_a_program_output():
    program = _simple_chain()
    # Mark d as a program output: it should stay in the supervertex's
    # outputs even though d's only IR consumer (e) is absorbed alongside.
    graph = Graph.from_program(program, program_outputs=frozenset({"d", "r"}))
    by_name = _by_out_name(graph)
    d, e = by_name["d"], by_name["e"]
    after = graph.replace(absorbed=(d, e), ops=(d.ops[0], e.ops[0]))
    super_v = next(v for v in after.vertices if len(v.ops) == 2)
    assert set(super_v.outputs) == {"d", "e"}


def test_replace_elides_fully_internal_output():
    program = _simple_chain()
    graph = Graph.from_program(program)
    # Absorb every vertex into one supervertex.
    after = graph.replace(absorbed=graph.vertices, ops=tuple(program))
    assert len(after.vertices) == 1
    super_v = after.vertices[0]
    assert set(super_v.inputs) == {"a", "b", "bias"}
    # Only program output is r; c, d, e are internal.
    assert super_v.outputs == ("r",)


def test_replace_propagates_input_from_inner_op():
    # When an absorbed vertex deeper in the chain reads an external
    # tensor (bias), that name must surface in the supervertex's inputs.
    program = _simple_chain()
    graph = Graph.from_program(program)
    by_name = _by_out_name(graph)
    c, d = by_name["c"], by_name["d"]
    after = graph.replace(absorbed=(c, d), ops=(c.ops[0], d.ops[0]))
    super_v = next(v for v in after.vertices if len(v.ops) == 2)
    assert set(super_v.inputs) == {"a", "b", "bias"}
    assert super_v.outputs == ("d",)


def test_replace_rejects_empty_absorbed():
    graph = Graph.from_program(_simple_chain())
    with pytest.raises(ValueError, match="empty"):
        graph.replace(absorbed=(), ops=())


def test_replace_rejects_vertex_from_other_graph():
    g1 = Graph.from_program(_simple_chain())
    g2 = Graph.from_program(_simple_chain())
    foreign = g2.vertices[0]
    with pytest.raises(ValueError, match="not in this graph"):
        g1.replace(absorbed=(foreign,), ops=foreign.ops)


def test_replace_rejects_duplicate_absorbed():
    graph = Graph.from_program(_simple_chain())
    v = graph.vertices[0]
    with pytest.raises(ValueError, match="duplicate"):
        graph.replace(absorbed=(v, v), ops=v.ops)


def test_replace_rejects_mismatched_ops():
    graph = Graph.from_program(_simple_chain())
    v = graph.vertices[0]
    # Pass empty ops when vertex has one op.
    with pytest.raises(ValueError, match="ops does not match"):
        graph.replace(absorbed=(v,), ops=())


def test_replace_rejects_out_of_program_order_ops():
    program = _simple_chain()
    graph = Graph.from_program(program)
    by_name = _by_out_name(graph)
    d, e = by_name["d"], by_name["e"]
    with pytest.raises(ValueError, match="program order"):
        graph.replace(absorbed=(d, e), ops=(e.ops[0], d.ops[0]))


def test_replace_rejects_non_convex_absorbed_set():
    # Absorbing the matmul (c) and the reduction (r) leaves add+exp on a
    # path between them: collapsing creates V → add → exp → V (cycle).
    program = _simple_chain()
    graph = Graph.from_program(program)
    by_name = _by_out_name(graph)
    c, r = by_name["c"], by_name["r"]
    with pytest.raises(ValueError, match="cycle|convex"):
        graph.replace(absorbed=(c, r), ops=(c.ops[0], r.ops[0]))


def test_replace_does_not_mutate_input_graph():
    program = _simple_chain()
    graph = Graph.from_program(program)
    snapshot = graph.vertices
    by_name = _by_out_name(graph)
    d, e = by_name["d"], by_name["e"]
    graph.replace(absorbed=(d, e), ops=(d.ops[0], e.ops[0]))
    assert graph.vertices is snapshot
    assert len(graph.vertices) == len(program)


# ---------------------------------------------------------------------------
# topo_sort
# ---------------------------------------------------------------------------


def test_topo_sort_matches_program_order_for_linear_chain():
    program = _simple_chain()
    graph = Graph.from_program(program)
    order = graph.topo_sort()
    assert [v.ops[0] for v in order] == list(program)


def test_topo_sort_is_deterministic_across_branches():
    # X feeds two independent consumers; the tiebreaker picks
    # program-order, so the result is the same every call.
    ops = Operations()
    ops.input("X", shape=(4, 4))
    ops.elementwise("exp", out="A", operands=("X",))
    ops.elementwise("log", out="B", operands=("X",))
    graph = Graph.from_program(ops.build())
    o1 = graph.topo_sort()
    o2 = graph.topo_sort()
    assert o1 == o2
    a_pos = next(i for i, v in enumerate(o1) if v.ops[0].out.name == "A")
    b_pos = next(i for i, v in enumerate(o1) if v.ops[0].out.name == "B")
    assert a_pos < b_pos


def test_topo_sort_raises_on_cycle():
    # Hand-construct a cyclic Graph directly (bypasses replace's check).
    ops = Operations()
    ops.input("X", shape=(4, 4))
    ops.elementwise("exp", out="Y", operands=("X",))
    ops.elementwise("log", out="Z", operands=("Y",))
    program = ops.build()
    # Two vertices, each claiming the other's output as input.
    v1 = Vertex(ops=(program[0],), inputs=("Z",), outputs=("Y",))
    v2 = Vertex(ops=(program[1],), inputs=("Y",), outputs=("Z",))
    graph = Graph(
        vertices=(v1, v2),
        op_index={id(op): i for i, op in enumerate(program)},
        program_outputs=frozenset({"Z"}),
    )
    with pytest.raises(ValueError, match="cycle"):
        graph.topo_sort()


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


def test_flatten_matches_program_for_primitive_graph():
    program = _simple_chain()
    graph = Graph.from_program(program)
    assert graph.flatten() == tuple(program)


def test_flatten_after_replace_preserves_intra_vertex_order():
    program = _simple_chain()
    graph = Graph.from_program(program)
    by_name = _by_out_name(graph)
    d, e = by_name["d"], by_name["e"]
    after = graph.replace(absorbed=(d, e), ops=(d.ops[0], e.ops[0]))
    flat = after.flatten()
    # Flattened sequence is a valid topo order of the original program.
    pos = {id(op): i for i, op in enumerate(flat)}
    producer = {op.out.name: op for op in program}
    for op in program:
        for t in op.inputs:
            p = producer.get(t.name)
            if p is None:
                continue
            assert pos[id(p)] < pos[id(op)]


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------


def test_check_dependencies_passes_for_primitive_graph():
    program = _simple_chain()
    Graph.from_program(program).check_dependencies(program)


def test_check_dependencies_passes_after_valid_replace():
    program = _simple_chain()
    graph = Graph.from_program(program)
    by_name = _by_out_name(graph)
    d, e = by_name["d"], by_name["e"]
    after = graph.replace(absorbed=(d, e), ops=(d.ops[0], e.ops[0]))
    after.check_dependencies(program)


def test_check_dependencies_detects_missing_op():
    program = _simple_chain()
    graph = Graph.from_program(program)
    truncated = Graph(
        vertices=graph.vertices[:-1],
        op_index=graph.op_index,
        program_outputs=graph.program_outputs,
    )
    with pytest.raises(ValueError, match="missing"):
        truncated.check_dependencies(program)


def test_check_dependencies_detects_intra_vertex_reordering():
    # A supervertex whose `ops` are in the wrong intra-vertex order is
    # the case the dependency check is most uniquely positioned to
    # catch (topo_sort can't see inside vertices). Construct directly
    # to bypass replace()'s program-order validation.
    program = _simple_chain()
    matmul_op, add_op, exp_op, reduce_op = program
    bad_super = Vertex(
        ops=(add_op, matmul_op),  # add reads 'c' before matmul produces it
        inputs=("a", "b", "bias"),
        outputs=("d",),
    )
    v_exp = Vertex(ops=(exp_op,), inputs=("d",), outputs=("e",))
    v_red = Vertex(ops=(reduce_op,), inputs=("e",), outputs=("r",))
    bad = Graph(
        vertices=(bad_super, v_exp, v_red),
        op_index={id(op): i for i, op in enumerate(program)},
        program_outputs=frozenset({"r"}),
    )
    with pytest.raises(ValueError, match="dependency violated"):
        bad.check_dependencies(program)


def test_preserves_dependencies_is_boolean_form():
    program = _simple_chain()
    assert Graph.from_program(program).preserves_dependencies(program)


# ---------------------------------------------------------------------------
# producer_of / consumers_of
# ---------------------------------------------------------------------------


def test_producer_of_returns_vertex_or_none():
    program = _simple_chain()
    graph = Graph.from_program(program)
    p = graph.producer_of("c")
    assert p is not None and isinstance(p.ops[0], MatmulOp)
    # Program inputs and unknown names → None.
    assert graph.producer_of("a") is None
    assert graph.producer_of("nope") is None


def test_consumers_of_returns_in_topo_order():
    ops = Operations()
    ops.input("X", shape=(4, 4))
    ops.elementwise("exp", out="A", operands=("X",))
    ops.elementwise("log", out="B", operands=("X",))
    graph = Graph.from_program(ops.build())
    cs = graph.consumers_of("X")
    assert tuple(v.ops[0].out.name for v in cs) == ("A", "B")
    # No consumers for an unknown name.
    assert graph.consumers_of("nope") == ()
