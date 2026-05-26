"""Vertex-level rewriting substrate above the IR.

A `Graph` is a DAG of `Vertex`es over an IR `Program`. Each vertex
contains one or more IR `Op`s and presents a typed boundary — inputs
(tensor names read from outside) and outputs (tensor names written
for outside readers). Operations:

  - `Graph.from_program(program)`: decompose into one vertex per Op.
  - `Graph.replace(absorbed, ops)`: collapse a connected subset of
    vertices into a single supervertex containing `ops`. Boundary is
    computed from the surrounding graph; cycle introduction is rejected.
  - `Graph.topo_sort()`: deterministic topological order of vertices,
    tie-broken by the earliest contained op's position in the original
    IR.
  - `Graph.flatten()`: the IR-op sequence implied by the graph
    (`topo_sort()` × each vertex's intra-vertex order).
  - `Graph.check_dependencies(original)`: the canonical soundness
    check — every (producer → consumer) data edge in `original` must
    be respected by `flatten()`. Call after every rewrite.

Views (transpose / reshape views in `Operations.tensors`) are not
graph nodes — like in `dag.py`, they appear as external inputs to
consuming vertices, since no IR Op produces them. The scheduler
handles the buffer_key aliasing downstream.

This module is independent of `dag.py`. The two co-exist: `dag.py`
gives a flat producer/consumer view used by the current fuser and
scheduler; `graph.py` is the substrate fusion passes will rewrite.
Until fusion is migrated, the graph here is read-only from the
perspective of the existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.ir import Op, Program


@dataclass(frozen=True)
class Vertex:
    """One node in the fused graph. Contains 1+ IR Ops in program order
    plus the boundary it presents to the rest of the graph.

    `inputs` are tensor names this vertex reads from outside (program
    inputs, transpose / reshape views, or outputs of other vertices).
    `outputs` are tensor names this vertex must materialize for
    outside readers (program outputs or inputs to other vertices)."""

    ops: tuple[Op, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def is_primitive(self) -> bool:
        """A vertex that wraps a single IR Op — the `from_program`
        baseline. Supervertices (post-fusion) wrap 2+ ops."""
        return len(self.ops) == 1

    @property
    def op_ids(self) -> frozenset[int]:
        """Identity set of contained ops, for fast 'does this vertex
        own op X?' queries during rule matching."""
        return frozenset(id(op) for op in self.ops)


@dataclass(frozen=True)
class Graph:
    """A DAG of `Vertex`es. Carries the position map of the original
    IR for deterministic topo-sort tiebreaks and for the dependency-
    preservation check. `program_outputs` is fixed at construction so
    a supervertex still surfaces a tensor as an output even when no
    other vertex consumes it — the user asked for it materialized."""

    vertices: tuple[Vertex, ...]
    op_index: dict[int, int]
    program_outputs: frozenset[str]

    @classmethod
    def from_program(
        cls,
        program: Program,
        *,
        program_outputs: frozenset[str] | None = None,
    ) -> Graph:
        """Decompose `program` into a primitive graph: one vertex per
        Op. `program_outputs` defaults to tensors produced but never
        consumed within the program (the natural sinks)."""
        op_index = {id(op): i for i, op in enumerate(program)}

        produced: set[str] = set()
        used: set[str] = set()
        for op in program:
            for t in op.inputs:
                used.add(t.name)
            produced.add(op.out.name)

        outs = (
            frozenset(produced - used)
            if program_outputs is None
            else frozenset(program_outputs)
        )

        vertices: list[Vertex] = []
        for op in program:
            in_names = tuple(t.name for t in op.inputs)
            name = op.out.name
            external = name in used or name in outs
            out_names = (name,) if external else ()
            vertices.append(Vertex(ops=(op,), inputs=in_names, outputs=out_names))

        return cls(
            vertices=tuple(vertices),
            op_index=op_index,
            program_outputs=outs,
        )

    def replace(
        self,
        absorbed: tuple[Vertex, ...],
        ops: tuple[Op, ...],
    ) -> Graph:
        """Collapse `absorbed` into a single new vertex containing
        `ops`. Returns a fresh graph; the input graph is untouched.

        Preconditions:
          - `absorbed` is a non-empty subset of `self.vertices` (by
            object identity), with no duplicates.
          - `ops` equals the union of `v.ops` over `absorbed`, in
            original-IR program order.
          - Collapsing the absorbed set must not introduce a cycle —
            i.e. `absorbed` is 'convex' in the dependency order (no
            non-absorbed vertex lies on a path between two absorbed
            ones). Violation raises ValueError after construction.

        Boundary inheritance:
          - inputs = (union of absorbed inputs) − (names produced
            internally by `ops`).
          - outputs = (names produced internally) ∩ (names still
            consumed by surviving vertices ∪ program_outputs)."""

        if not absorbed:
            raise ValueError("replace: absorbed set is empty")

        graph_vertex_ids = {id(v) for v in self.vertices}
        absorbed_id_set: set[int] = set()
        for v in absorbed:
            if id(v) not in graph_vertex_ids:
                raise ValueError(
                    "replace: absorbed contains a vertex not in this graph"
                )
            if id(v) in absorbed_id_set:
                raise ValueError("replace: absorbed contains duplicate vertices")
            absorbed_id_set.add(id(v))

        expected_op_ids = frozenset(id(op) for v in absorbed for op in v.ops)
        given_op_ids = frozenset(id(op) for op in ops)
        if expected_op_ids != given_op_ids:
            raise ValueError(
                "replace: ops does not match the union of absorbed vertices' ops"
            )

        positions = [self.op_index[id(op)] for op in ops]
        if positions != sorted(positions):
            raise ValueError("replace: ops must be in original-IR program order")

        produced_inside = {op.out.name for op in ops}
        survivor_inputs: set[str] = set()
        for v in self.vertices:
            if id(v) not in absorbed_id_set:
                survivor_inputs.update(v.inputs)

        absorbed_inputs: set[str] = set()
        for v in absorbed:
            absorbed_inputs.update(v.inputs)

        new_inputs = tuple(sorted(absorbed_inputs - produced_inside))
        new_outputs = tuple(
            sorted(
                n
                for n in produced_inside
                if n in survivor_inputs or n in self.program_outputs
            )
        )

        new_vertex = Vertex(ops=ops, inputs=new_inputs, outputs=new_outputs)

        new_vertices: list[Vertex] = []
        inserted = False
        for v in self.vertices:
            if id(v) in absorbed_id_set:
                if not inserted:
                    new_vertices.append(new_vertex)
                    inserted = True
            else:
                new_vertices.append(v)

        new_graph = Graph(
            vertices=tuple(new_vertices),
            op_index=self.op_index,
            program_outputs=self.program_outputs,
        )

        try:
            new_graph.topo_sort()
        except ValueError:
            raise ValueError(
                "replace: collapsing the absorbed set would introduce a "
                "cycle — the absorbed vertices are not convex in the "
                "dependency order"
            )
        return new_graph

    def topo_sort(self) -> tuple[Vertex, ...]:
        """Vertices in topological order. Ties broken by the earliest
        contained op's position in the original IR, so the result is
        deterministic across runs and across rewrite histories. Raises
        ValueError on cycles (a malformed graph)."""

        producer: dict[str, Vertex] = {}
        for v in self.vertices:
            for name in v.outputs:
                producer[name] = v

        by_id = {id(v): v for v in self.vertices}
        deps: dict[int, set[int]] = {vid: set() for vid in by_id}
        successors: dict[int, set[int]] = {vid: set() for vid in by_id}
        for v in self.vertices:
            for name in v.inputs:
                p = producer.get(name)
                if p is not None:
                    deps[id(v)].add(id(p))
                    successors[id(p)].add(id(v))

        indegree = {vid: len(ds) for vid, ds in deps.items()}

        def key(vid: int) -> int:
            return min(self.op_index[id(op)] for op in by_id[vid].ops)

        ready = sorted(
            [vid for vid, deg in indegree.items() if deg == 0],
            key=key,
        )
        result: list[Vertex] = []
        while ready:
            vid = ready.pop(0)
            result.append(by_id[vid])
            for s in successors[vid]:
                indegree[s] -= 1
                if indegree[s] == 0:
                    ready.append(s)
            ready.sort(key=key)

        if len(result) != len(self.vertices):
            raise ValueError("graph contains a cycle")
        return tuple(result)

    def flatten(self) -> tuple[Op, ...]:
        """The IR-op sequence implied by the graph: vertices in topo
        order, each vertex's ops in intra-vertex order (already
        program order by construction)."""
        return tuple(op for v in self.topo_sort() for op in v.ops)

    def check_dependencies(self, original: Program) -> None:
        """Verify that every (producer → consumer) data edge in
        `original` is respected by `self.flatten()`. Raises ValueError
        on the first violation, naming the offending tensor and ops.

        This is the canonical soundness check — run it after every
        rewrite during fuser development. It catches reordering bugs,
        misdeclared vertex boundaries (which propagate into topo-sort
        and then into flatten), and op set mismatches against
        `original`.

        Op multiplicity is allowed: after `absorb_into` replicates a
        producer into multiple consumer vertices, the same op appears
        in multiple positions of `flatten()`. The dependency check
        passes as long as, for each consumer position, at least one
        producer position appears strictly before it — i.e. every
        consumer kernel had its inputs computed by the time it ran."""

        flat = self.flatten()
        positions: dict[int, list[int]] = {}
        for i, op in enumerate(flat):
            positions.setdefault(id(op), []).append(i)

        original_ids = {id(op) for op in original}
        flat_ids = set(positions)
        missing = original_ids - flat_ids
        extra = flat_ids - original_ids
        if missing:
            names = [
                f"{type(op).__name__}({op.out.name})"
                for op in original
                if id(op) in missing
            ]
            raise ValueError(f"graph missing IR ops: {names}")
        if extra:
            raise ValueError(
                "graph contains ops not in original IR — replace() received "
                "ops that don't belong to this Program"
            )

        producer_of: dict[str, Op] = {}
        for op in original:
            producer_of[op.out.name] = op

        for op in original:
            op_positions = positions[id(op)]
            for t in op.inputs:
                producer = producer_of.get(t.name)
                if producer is None:
                    continue
                producer_positions = positions[id(producer)]
                for c_pos in op_positions:
                    if not any(p < c_pos for p in producer_positions):
                        raise ValueError(
                            f"dependency violated: "
                            f"{type(producer).__name__}({producer.out.name}) "
                            f"has no occurrence before "
                            f"{type(op).__name__}({op.out.name}) at "
                            f"flatten-pos {c_pos} (consumed via tensor "
                            f"{t.name!r})"
                        )

    def preserves_dependencies(self, original: Program) -> bool:
        """Boolean form of `check_dependencies` for predicate use in
        rule preconditions."""
        try:
            self.check_dependencies(original)
        except ValueError:
            return False
        return True

    def absorb_into(
        self,
        consumer: Vertex,
        additional_ops: tuple[Op, ...],
    ) -> Graph:
        """Replicate `additional_ops` into `consumer`'s vertex: returns
        a new graph where `consumer` is replaced by a vertex containing
        `additional_ops + consumer.ops` merged in program order. Unlike
        `replace`, this does NOT remove the source vertex of
        `additional_ops` — it's the rewrite for multi-consumer recompute,
        where the same producer op is inlined into multiple consumer
        kernels and the standalone producer vertex is removed via
        `remove_vertex` only once every consumer has absorbed (or never,
        if some consumer can't).

        Boundary inheritance:
          inputs  = (consumer.inputs ∪ inputs of additional_ops)
                    − outputs of additional_ops
          outputs = consumer.outputs (additional_ops' outputs become
                    internal to this vertex)"""
        if id(consumer) not in {id(v) for v in self.vertices}:
            raise ValueError("absorb_into: consumer not in this graph")
        if not additional_ops:
            raise ValueError("absorb_into: additional_ops is empty")

        merged = tuple(
            op
            for _, op in sorted(
                (self.op_index[id(op)], op) for op in (*additional_ops, *consumer.ops)
            )
        )
        additional_inputs: set[str] = set()
        additional_outputs: set[str] = set()
        for op in additional_ops:
            for t in op.inputs:
                additional_inputs.add(t.name)
            additional_outputs.add(op.out.name)

        new_inputs = tuple(
            sorted((set(consumer.inputs) | additional_inputs) - additional_outputs)
        )
        new_vertex = Vertex(ops=merged, inputs=new_inputs, outputs=consumer.outputs)
        new_vertices = tuple(new_vertex if v is consumer else v for v in self.vertices)
        new_graph = Graph(
            vertices=new_vertices,
            op_index=self.op_index,
            program_outputs=self.program_outputs,
        )
        try:
            new_graph.topo_sort()
        except ValueError:
            raise ValueError(
                "absorb_into: result is cyclic (additional_ops' inputs "
                "transitively depend on the consumer's outputs)"
            )
        return new_graph

    def remove_vertex(self, vertex: Vertex) -> Graph:
        """Drop `vertex` from the graph. Precondition: vertex's outputs
        are not in `program_outputs` and no other vertex consumes them.
        Used to elide a producer after every consumer has absorbed it
        via `absorb_into`."""
        if id(vertex) not in {id(v) for v in self.vertices}:
            raise ValueError("remove_vertex: vertex not in this graph")
        for name in vertex.outputs:
            if name in self.program_outputs:
                raise ValueError(
                    f"remove_vertex: output {name!r} is a program output; cannot remove"
                )
            for other in self.vertices:
                if other is vertex:
                    continue
                if name in other.inputs:
                    raise ValueError(
                        f"remove_vertex: output {name!r} still consumed by "
                        "another vertex"
                    )
        return Graph(
            vertices=tuple(v for v in self.vertices if v is not vertex),
            op_index=self.op_index,
            program_outputs=self.program_outputs,
        )

    def producer_of(self, name: str) -> Vertex | None:
        """The unique vertex producing tensor `name`, or None if the
        tensor is external (program input or view)."""
        for v in self.vertices:
            if name in v.outputs:
                return v
        return None

    def consumers_of(self, name: str) -> tuple[Vertex, ...]:
        """Vertices consuming `name`, in topo order."""
        return tuple(v for v in self.topo_sort() if name in v.inputs)
