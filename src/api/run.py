"""End-to-end compile + run for an `Operations` graph.

Pipeline:

    ops.build()                  # IR Program
      → fuse(program)            # tuple[Vertex, ...]
      → assemble(v)              # tuple[KernelGroup, ...]
      → alias_group(g)           # tgmem aliasing rewrite
      → schedule(ops, groups)    # Runtime
      → runtime.run()            # env dict

Returns a `RunResult(env, groups)`:

  - `env` — the executed env dict; callers pull tensors out by their
    declared names (`result.env["y"]`, etc.). When `profile=True`
    it additionally contains per-kernel timings under keys
    `t_0`, `t_1`, ...
  - `groups` — the `KernelGroup` tuple in dispatch order, useful for
    introspection (strategies, function names, generated MSL).
"""

from __future__ import annotations

from typing import Any, NamedTuple

from orchestrator import Operations
from orchestrator.aliasing import alias_group
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse
from orchestrator.kernel_group import KernelGroup
from orchestrator.scheduler import schedule


class RunResult(NamedTuple):
    env: dict[str, Any]
    groups: tuple[KernelGroup, ...]


def run(ops: Operations, *, profile: bool = False) -> RunResult:
    """Compile the program described by `ops` and execute it. Inputs
    must have been supplied via `ops.from_numpy(...)`; outputs are
    materialized into the returned env under their tensor names.

    Set `profile=True` to record per-kernel GPU time_ms into the env
    under `t_0`, `t_1`, ..."""
    program = ops.build()
    vertices = fuse(program)
    groups = tuple(alias_group(assemble(v)) for v in vertices)
    runtime = schedule(ops, groups, profile=profile)
    return RunResult(env=runtime.run(), groups=groups)
