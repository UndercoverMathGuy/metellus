from compute.elementwise.elementwise import (
    BROADCAST_MODES,
    BroadcastSpec,
    ElementwiseComputeFragment,
    elementwise_arity,
    elementwise_compute_block,
    elementwise_expression,
    elementwise_outputs_bool,
    elementwise_threadgroup_section,
    supported_elementwise_ops,
)


# Lazy export: TiledElementwiseChainFragment lives in `tiled_chain.py` but
# transitively imports `orchestrator.ir`, which itself imports from this
# package — eager import here would deadlock the package initialization.
# `__getattr__` defers the submodule load until first attribute access, by
# which point both packages have finished initializing.
def __getattr__(name):
    if name == "TiledElementwiseChainFragment":
        from compute.elementwise.tiled_chain import TiledElementwiseChainFragment

        return TiledElementwiseChainFragment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
