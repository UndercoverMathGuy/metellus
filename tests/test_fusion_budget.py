"""Budget-gated fusion behavior.

These tests cover the tgmem-cap guarantee `fuse()` provides:

  - Programs that fit are unchanged.
  - Programs whose naive fusion would overflow the cap have the
    offending fusion blacklisted; the run produces split kernels,
    each under cap.
  - A primitive op that on its own already exceeds the cap raises
    `TgmemOverflowError` up-front — fusion can never shrink it.
  - The blacklist mechanism is deterministic across runs.

Structural fusion shape under normal conditions is covered in
`test_fusion.py`; this file is specifically about the budget gate.
"""

from __future__ import annotations

import pytest

from orchestrator import Operations
from orchestrator.aliasing import (
    TGMEM_CAP_BYTES,
    TgmemOverflowError,
    compute_group_budget,
)
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse


def _budget_bytes(vertex) -> int:
    return compute_group_budget(assemble(vertex)).size_bytes


def test_under_cap_program_fuses_as_normal():
    """Sanity: a vanilla matmul+bias+relu chain fits well under cap
    and produces a single fused vertex. Budget gate is invisible when
    no fusion would overflow."""
    ops = Operations()
    ops.input("A", shape=(32, 64))
    ops.input("B", shape=(64, 128))
    ops.input("bias", shape=(1, 128))
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("add", out="Cb", operands=("C", "bias"), y_broadcast="row")
    ops.elementwise("max", out="y", operands=("Cb", 0.0))
    vertices = fuse(ops.build())
    assert len(vertices) == 1
    assert _budget_bytes(vertices[0]) <= TGMEM_CAP_BYTES


def test_overflowing_multi_producer_gets_blacklisted_and_splits():
    """Two parallel 64x64 matmuls feeding a convergent add overflow
    the cap when fused into one multi-anchor kernel (the per-anchor C
    tiles + B tiles push it past 32 KiB). The gate must blacklist
    that fusion and produce split kernels, each under cap."""
    ops = Operations()
    ops.input("A", shape=(64, 64))
    ops.input("B", shape=(64, 64))
    ops.input("C", shape=(64, 64))
    ops.input("D", shape=(64, 64))
    ops.matmul(a="A", b="B", out="c1")
    ops.matmul(a="C", b="D", out="c2")
    ops.elementwise("add", out="z", operands=("c1", "c2"))
    vertices = fuse(ops.build())
    # Without the gate this would be 1 multi-anchor vertex at ~33280 B.
    assert len(vertices) > 1
    for v in vertices:
        assert _budget_bytes(v) <= TGMEM_CAP_BYTES, (
            f"vertex {[type(op).__name__ for op in v.ops]} "
            f"is {_budget_bytes(v)} B, exceeds {TGMEM_CAP_BYTES} B"
        )


def test_irreducible_primitive_raises(monkeypatch):
    """A primitive vertex whose own footprint already exceeds the cap
    can't be helped by fusion. Surface it as `TgmemOverflowError` with
    the offending op named, so the user knows to shrink the tile shape
    or pick a different strategy. We force the situation by lowering
    the cap below the smallest primitive footprint."""
    monkeypatch.setattr("orchestrator.fusion.TGMEM_CAP_BYTES", 64)
    ops = Operations()
    ops.input("A", shape=(32, 32))
    ops.input("B", shape=(32, 32))
    ops.matmul(a="A", b="B", out="C")
    with pytest.raises(TgmemOverflowError) as exc_info:
        fuse(ops.build())
    # The error names the offending op so users can act on it.
    assert "MatmulOp" in str(exc_info.value)
    assert "C" in str(exc_info.value)


def test_fuse_is_deterministic_under_blacklist_retry():
    """The blacklist-and-retry loop is deterministic: repeat runs on
    the same program produce structurally identical partitions. This
    matters because the blacklist is purely a function of pass
    enumeration order, which must be stable across runs for results
    to be reproducible."""
    def build():
        ops = Operations()
        ops.input("A", shape=(64, 64))
        ops.input("B", shape=(64, 64))
        ops.input("C", shape=(64, 64))
        ops.input("D", shape=(64, 64))
        ops.matmul(a="A", b="B", out="c1")
        ops.matmul(a="C", b="D", out="c2")
        ops.elementwise("add", out="z", operands=("c1", "c2"))
        return ops.build()

    v1 = fuse(build())
    v2 = fuse(build())
    assert len(v1) == len(v2)
    for a, b in zip(v1, v2):
        # Same op types in same vertex slots is the structural fingerprint
        # — concrete op identities differ because each build() makes fresh Ops.
        assert [type(op).__name__ for op in a.ops] == [type(op).__name__ for op in b.ops]


def test_blacklisted_fusion_still_allows_other_fusions():
    """Blacklisting one fusion (multi-producer convergent) shouldn't
    nuke unrelated fusions. The two matmuls each become standalone
    kernels, but the convergent add still gets to fuse where it can —
    here it can't fuse into either matmul (each matmul has only one
    consumer, the add, but the add has two producers so it's not
    epilogue-eligible for either)."""
    ops = Operations()
    ops.input("A", shape=(64, 64))
    ops.input("B", shape=(64, 64))
    ops.input("C", shape=(64, 64))
    ops.input("D", shape=(64, 64))
    ops.matmul(a="A", b="B", out="c1")
    ops.matmul(a="C", b="D", out="c2")
    ops.elementwise("add", out="z", operands=("c1", "c2"))
    vertices = fuse(ops.build())
    # Every vertex under cap; partition is some legal split.
    for v in vertices:
        assert _budget_bytes(v) <= TGMEM_CAP_BYTES
    # The two matmuls run as their own kernels (no epilogue fusion
    # possible since `add` has two producers).
    op_groupings = sorted(
        tuple(sorted(type(op).__name__ for op in v.ops)) for v in vertices
    )
    assert ("MatmulOp",) in op_groupings
