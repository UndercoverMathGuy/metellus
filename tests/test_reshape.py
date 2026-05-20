"""Reshape / transpose machinery: views (no IR op emitted) for
metadata-only transforms; ShapeOp emitted when a copy is needed.

Also covers the end-to-end "make it smart" path: a transposed matmul
input produces a kernel whose load strides reflect the swap."""

import pytest

from orchestrator import (
    Operations,
    ShapeOp,
    Tensor,
)
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse


# ---------------------------------------------------------------------------
# Tensor-level
# ---------------------------------------------------------------------------


def test_default_tensor_buffer_key_matches_name():
    t = Tensor("x", (8, 16))
    assert t.buffer_key == "x"


def test_is_contiguous_for_default_row_major():
    assert Tensor("x", (8, 16)).is_contiguous()
    assert Tensor("v", (64,)).is_contiguous()


def test_is_contiguous_false_after_transpose():
    t = Tensor("x", (8, 16)).transpose("x_T")
    assert not t.is_contiguous()


def test_reshape_view_aliases_buffer():
    src = Tensor("x", (8, 16))
    view = src.reshape_view((128,), name="x_flat")
    assert view.name == "x_flat"
    assert view.buffer_key == "x"
    assert view.shape == (128,)
    assert view.row_stride == 1
    assert view.col_stride == 0


def test_reshape_view_2d_to_2d_aliases():
    src = Tensor("x", (8, 16))
    view = src.reshape_view((4, 32), name="x_r")
    assert view.buffer_key == "x"
    assert view.shape == (4, 32)
    assert view.row_stride == 32 and view.col_stride == 1


def test_reshape_view_raises_when_non_linear_and_shape_differs():
    t = Tensor("x", (8, 16)).transpose("x_T")  # non-linear: M=16, K=8, rs=1 ≠ K·cs
    with pytest.raises(ValueError, match="linearly-iterable"):
        t.reshape_view((128,), name="bad")


def test_same_shape_reshape_view_is_noop_regardless_of_strides():
    """A no-op reshape (target shape == source shape) is always a
    metadata view, even on a non-contiguous source."""
    t = Tensor("x", (8, 16)).transpose("x_T")  # col-major, strides (1, 16)
    v = t.reshape_view((16, 8), name="v")
    assert v.buffer_key == "x"
    # Strides carried over from the source (no-op semantics).
    assert (v.row_stride, v.col_stride) == (1, 16)
    assert v.layout is t.layout


def test_reshape_view_raises_on_element_count_mismatch():
    t = Tensor("x", (8, 16))
    with pytest.raises(ValueError):
        t.reshape_view((9, 16), name="bad")


def test_reshape_view_raises_on_3d():
    t = Tensor("x", (8, 16))
    with pytest.raises(ValueError):
        t.reshape_view((2, 4, 16), name="bad")


# ---------------------------------------------------------------------------
# Builder-level
# ---------------------------------------------------------------------------


def test_builder_transpose_emits_no_op():
    ops = Operations()
    ops.input("A", shape=(8, 16))
    at = ops.transpose("A", out="At")
    program = ops.build()
    assert program == []  # transpose is metadata-only
    assert at.buffer_key == "A"
    assert at.shape == (16, 8)


def test_builder_reshape_contiguous_emits_no_op():
    ops = Operations()
    ops.input("X", shape=(8, 16))
    v = ops.reshape("X", (128,), out="Xf")
    program = ops.build()
    assert program == []
    assert v.buffer_key == "X"
    assert v.shape == (128,)


def test_builder_reshape_after_transpose_emits_shape_op():
    ops = Operations()
    ops.input("X", shape=(8, 16))
    ops.transpose("X", out="Xt")  # non-contiguous now
    out = ops.reshape("Xt", (128,), out="Xtf")
    program = ops.build()
    assert len(program) == 1
    assert isinstance(program[0], ShapeOp)
    assert program[0].input.name == "Xt"
    assert program[0].out.name == "Xtf"
    # The copy materializes a fresh buffer (no aliasing).
    assert out.buffer_key == "Xtf"


def test_builder_reshape_element_count_mismatch_raises():
    ops = Operations()
    ops.input("X", shape=(8, 16))
    with pytest.raises(ValueError, match="element-count"):
        ops.reshape("X", (9, 16), out="bad")


def test_builder_reshape_rank_check():
    ops = Operations()
    ops.input("X", shape=(8, 16))
    ops.transpose("X", out="Xt")
    with pytest.raises(ValueError, match="rank"):
        ops.reshape("Xt", (2, 4, 16), out="bad")


def test_view_name_clash_raises():
    ops = Operations()
    ops.input("A", shape=(8, 16))
    ops.input("At", shape=(16, 8))
    with pytest.raises(ValueError, match="already declared"):
        ops.transpose("A", out="At")


# ---------------------------------------------------------------------------
# End-to-end: transpose-into-matmul produces correct kernel strides
# ---------------------------------------------------------------------------


def test_transposed_input_threads_through_to_matmul_kernel():
    """A_T @ B: the matmul kernel should read A's buffer with swapped
    row/col strides (row_stride=1, col_stride=M_orig), not the
    row-major default. Buffer binding aliases the original A."""
    M_orig, K_orig = 32, 64
    # A is (32, 64) row-major.  A.T is (64, 32) viewing the same buffer.
    # Matmul expects (M_new, K_new) @ (K_new, N) = (M_new, N).
    # With A.T as the left operand: M_new = 64, K_new = 32, N = 16.
    ops = Operations()
    ops.input("A", shape=(M_orig, K_orig))
    ops.input("B", shape=(M_orig, 16))  # B is (K_new, N) = (32, 16)
    ops.transpose("A", out="At")
    ops.matmul(a="At", b="B", out="C")
    program = ops.build()
    decisions = fuse(program)
    assert len(decisions) == 1
    group = assemble(decisions[0])
    src = group.kernel.source

    # The A buffer is bound under its original name, not "At".
    assert group.bindings[0] == "A"
    # MSL param uses the buffer_key (A), not the SSA view name.
    assert "device const float* A [[buffer(0)]]" in src
    # No "At" identifier appears anywhere in the kernel source — the
    # view name is purely an IR-level handle.
    assert "At[" not in src
    assert "At " not in src
    # The load expression for A uses the transposed strides:
    # row_stride == 1, col_stride == 64 (the original K).
    # The cooperative load substitutes these into
    # `A[global_row * (1) + global_col * (64)]`.
    assert "global_row * (1)" in src
    assert f"global_col * ({K_orig})" in src


def test_reshape_view_passes_to_matmul_unchanged():
    """A reshape that's a metadata swap doesn't perturb the matmul's
    view of its inputs (since the matmul reads the underlying buffer
    with the view's shape/strides)."""
    ops = Operations()
    ops.input("X", shape=(4, 8))
    # Reshape X (4,8) row-major contiguous → (32,1) — still contiguous;
    # buffer aliased. (Not directly useful in a matmul but exercises the
    # registry plumbing.)
    v = ops.reshape("X", (2, 16), out="Xr")
    program = ops.build()
    assert program == []
    assert v.buffer_key == "X"
    assert v.row_stride == 16 and v.col_stride == 1
