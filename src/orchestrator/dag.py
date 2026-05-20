"""Producer/consumer view over an IR Program.

`build_dag(program)` returns a `DAG` exposing:

  - `producer_of[name]`     : the Op that defines tensor `name`
  - `consumers_of[name]`    : tuple of Ops that read `name`, in IR order
  - `inputs`                : tensor names read but never produced (caller-supplied)
  - `outputs`               : tensor names produced but never consumed (program sinks)

Validates SSA (each output name appears at most once across ops) and
def-before-use ordering (every operand is either an input or produced by
an earlier op in the list).

Downstream fusion / scheduling / liveness passes consume this view
instead of re-walking the program each time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from orchestrator.ir import Op, Program


@dataclass(frozen=True)
class DAG:
    program: Program
    producer_of: dict[str, Op]
    consumers_of: dict[str, tuple[Op, ...]]
    inputs: frozenset[str]
    outputs: frozenset[str]

    def unique_consumer(self, name: str) -> Op | None:
        """The single consuming op of `name`, or None if zero or >1
        consumers. Convenient for fusion eligibility checks."""
        cs = self.consumers_of.get(name, ())
        return cs[0] if len(cs) == 1 else None

    def topo_order(self) -> list[Op]:
        """The IR is already topologically ordered (validated at build
        time). Returns a fresh list copy for callers that want to mutate."""
        return list(self.program)


def build_dag(program: Program) -> DAG:
    producer_of: dict[str, Op] = {}
    consumer_lists: dict[str, list[Op]] = defaultdict(list)
    used: set[str] = set()

    for op in program:
        for t in op.inputs:
            consumer_lists[t.name].append(op)
            used.add(t.name)
        out_name = op.out.name
        if out_name in producer_of:
            raise ValueError(f"SSA violation: tensor {out_name!r} produced by two ops")
        if out_name in used:
            raise ValueError(
                f"def-after-use: tensor {out_name!r} consumed before it is "
                "defined (program must be in topological order)"
            )
        producer_of[out_name] = op

    produced = set(producer_of.keys())
    inputs = frozenset(used - produced)
    outputs = frozenset(produced - used)

    consumers_of = {name: tuple(ops) for name, ops in consumer_lists.items()}
    return DAG(
        program=list(program),
        producer_of=producer_of,
        consumers_of=consumers_of,
        inputs=inputs,
        outputs=outputs,
    )
