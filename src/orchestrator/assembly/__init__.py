"""Assemble a fused `Vertex` → `KernelGroup`.

`assemble(vertex)` classifies the vertex by its contained ops + the
position of its (single, in v0) anchor, then dispatches to a
per-strategy template. The classification produces a private
`DecisionView` — the same shape as the old fuser's `FusionDecision`,
just constructed by walking the vertex's ops rather than handed in
pre-shaped — so all the per-strategy template code below is reused
without changes.

Each per-strategy module (`matmul`, `reduction`, `elementwise`,
`shape`, `multi_anchor`) wires the compute-layer fragments into a
CodegenContext and emits MSL. They share helpers for chain expression
building (`expressions`), broadcast load geometry (`load_geometry`),
tg-tile setup (`chain_tiles`), and extra-buffer tracking (`extras`).
The tile-resident chain fragment itself lives in the compute layer
(`compute.elementwise.tiled_chain.TiledElementwiseChainFragment`).

The compute / memory / fragments modules already provide every
emission primitive we need; this package does not write any new MSL.
Fragment identifiers (`global_row`, `global_col`, `flat_tid`, etc.)
live in the canonical preamble — see `compute/scaffold.py:kernel_preamble`.
"""

from __future__ import annotations

from orchestrator.graph import Vertex
from orchestrator.kernel_group import FusionStrategy, KernelGroup

from orchestrator.assembly.decision import classify
from orchestrator.assembly.elementwise import assemble_elementwise
from orchestrator.assembly.matmul import assemble_matmul
from orchestrator.assembly.multi_anchor import assemble_multi_anchor
from orchestrator.assembly.reduction import assemble_reduction
from orchestrator.assembly.shape import assemble_shape


_MATMUL_STRATEGIES = {
    FusionStrategy.STANDALONE_MATMUL,
    FusionStrategy.MATMUL_EPILOGUE_TG,
    FusionStrategy.MATMUL_EPILOGUE_REGISTER,
    FusionStrategy.ELEMENTWISE_PROLOGUE_MATMUL,
}
_REDUCTION_STRATEGIES = {
    FusionStrategy.STANDALONE_REDUCTION,
    FusionStrategy.REDUCTION_EPILOGUE,
    FusionStrategy.ELEMENTWISE_PROLOGUE_REDUCTION,
}
_ELEMENTWISE_STRATEGIES = {
    FusionStrategy.STANDALONE_ELEMENTWISE,
    FusionStrategy.ELEMENTWISE_CHAIN,
}
_MULTI_ANCHOR_STRATEGIES = {
    FusionStrategy.MULTI_PRODUCER_CONVERGENT,
    FusionStrategy.DIAMOND_SHARED,
}


def assemble(vertex: Vertex, *, function_name: str | None = None) -> KernelGroup:
    """Classify `vertex` and dispatch to the matching per-strategy
    template."""
    decision = classify(vertex)
    strategy = decision.strategy
    if strategy in _MATMUL_STRATEGIES:
        return assemble_matmul(decision, function_name)
    if strategy in _REDUCTION_STRATEGIES:
        return assemble_reduction(decision, function_name)
    if strategy in _ELEMENTWISE_STRATEGIES:
        return assemble_elementwise(decision, function_name)
    if strategy is FusionStrategy.STANDALONE_SHAPE:
        return assemble_shape(decision, function_name)
    if strategy in _MULTI_ANCHOR_STRATEGIES:
        return assemble_multi_anchor(decision, function_name)
    raise ValueError(f"unknown strategy {strategy!r}")


__all__ = ["assemble"]
