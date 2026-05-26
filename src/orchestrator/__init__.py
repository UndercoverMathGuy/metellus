"""Orchestrator: IR Program → (later) fused KernelGroups → Runtime.

Public surface for now: the primitive IR, the `Operations` builder that
constructs it, and a producer/consumer `DAG` view used by downstream
fusion / scheduling passes.
"""

from compute.elementwise.elementwise import BroadcastSpec

from orchestrator.builder import Operations
from orchestrator.dag import DAG, build_dag
from orchestrator.graph import Graph, Vertex
from orchestrator.ir import (
    ElementwiseOp,
    Layout,
    MatmulOp,
    Op,
    OperandValue,
    Program,
    ReductionOp,
    Scalar,
    ShapeOp,
    Tensor,
)

# Note: `fuse`, `assemble`, `schedule`, `KernelGroup`, `FusionDecision`,
# and `FusionStrategy` live in `orchestrator.fusion` / `orchestrator.assembly`
# / `orchestrator.scheduler` / `orchestrator.kernel_group`. They pull in
# `runtime.program.Kernel` (and therefore the Metal backend), so callers
# that only build IR / DAG don't have to pay that import cost. Import them
# directly from the submodules.

__all__ = [
    "BroadcastSpec",
    "DAG",
    "ElementwiseOp",
    "Graph",
    "Layout",
    "MatmulOp",
    "Op",
    "OperandValue",
    "Operations",
    "Program",
    "ReductionOp",
    "Scalar",
    "ShapeOp",
    "Tensor",
    "Vertex",
    "build_dag",
]
