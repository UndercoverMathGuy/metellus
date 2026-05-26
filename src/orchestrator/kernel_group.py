"""KernelGroup: the fuser's output type.

A `KernelGroup` is one compiled MSL kernel together with everything the
runtime needs to dispatch it — bindings (env keys in MSL slot order),
the dim-value tuple matching `ctx.dims`, and a (grid, threads) launch
shape. It also records which IR `Op`s it absorbed and the
`FusionStrategy` that produced it, so callers and tests can introspect
the fusion plan.

The fuser returns a plain ordered tuple of `KernelGroup`s; the
`scheduler` walks that tuple directly to produce a `Runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from orchestrator.ir import Op
from runtime.program import Kernel


class FusionStrategy(StrEnum):
    """Identifies the chain shape of a KernelGroup. The strategy drives
    which assembly function builds the fragments and how the runtime
    Dispatch is shaped (grid/threads/bindings)."""

    STANDALONE_MATMUL = "standalone_matmul"
    STANDALONE_ELEMENTWISE = "standalone_elementwise"
    STANDALONE_REDUCTION = "standalone_reduction"
    # matmul → (elem chain) absorbed into matmul's epilogue.
    MATMUL_EPILOGUE_REGISTER = "matmul_epilogue_register"
    MATMUL_EPILOGUE_TG = "matmul_epilogue_tg"
    # (elem chain) → matmul on A and/or B inputs, optionally with
    # (elem chain) absorbed into the matmul's epilogue too.
    ELEMENTWISE_PROLOGUE_MATMUL = "elementwise_prologue_matmul"
    # (elem chain) → reduction with optional (elem chain) on the
    # post-reduction scalar.
    ELEMENTWISE_PROLOGUE_REDUCTION = "elementwise_prologue_reduction"
    # reduction → (elem chain) only (no prologue absorbed). Disjoint from
    # ELEMENTWISE_PROLOGUE_REDUCTION so callers can tell which side
    # extended.
    REDUCTION_EPILOGUE = "reduction_epilogue"
    # Standalone elementwise chain with no matmul/reduc anchor.
    ELEMENTWISE_CHAIN = "elementwise_chain"
    # Pure data rearrangement (ShapeOp). One kernel per op; no fusion in v0.
    STANDALONE_SHAPE = "standalone_shape"
    # Two matmul anchors feeding one convergent elem ('z = c1 + c2').
    # Sequential mainloops, both accumulators staged into per-anchor C
    # tiles, merged via the tg-tile path, stored to device.
    MULTI_PRODUCER_CONVERGENT = "multi_producer_convergent"
    # Same shape as above, but the two anchors share an input tensor
    # with matching K — one outer mainloop emits a single shared A-load
    # plus two B-loads + two computes per k-chunk, amortising the
    # shared load across both matmuls.
    DIAMOND_SHARED = "diamond_shared"


@dataclass(frozen=True)
class KernelGroup:
    """One fused kernel, ready to dispatch.

    `bindings` are env keys in MSL slot order (matching `kernel.ctx.buffers`).
    `dims` are the int runtime values for `kernel.ctx.dims`. `ops` lists the
    IR ops absorbed into this group, in IR order — useful for tests and for
    the scheduler to know which intermediate tensors are now elided.
    """

    kernel: Kernel
    bindings: tuple[str, ...]
    dims: tuple[int, ...]
    grid: tuple[int, int, int]
    threads: tuple[int, int, int]
    ops: tuple[Op, ...]
    strategy: FusionStrategy

    @property
    def outputs(self) -> tuple[str, ...]:
        """Tensor names this kernel writes (last op's output, plus any
        producer outputs that survive because they had non-absorbing
        consumers — tracked at fusion-decision time and surfaced here)."""
        return tuple(op.out.name for op in self.ops if op.out.name in self.bindings)
