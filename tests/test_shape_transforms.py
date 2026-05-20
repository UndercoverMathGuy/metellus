"""All-around coverage for reshape / transpose / ShapeOp flows.

Two test categories:

1. **Compile-time structural tests** — assert the right IR + KernelGraph
   shape comes out (which views are metadata-only, when a ShapeOp gets
   emitted, what bindings/strides land in the assembled kernel).

2. **End-to-end execution tests** — actually dispatch the assembled
   ShapeOp kernel and compare against numpy. Skipped if the Metal
   backend isn't importable (e.g. cross-platform CI). Confirms the
   indexing math is correct under transpose/reshape combinations."""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator import Operations, ShapeOp, Tensor
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse
from orchestrator.kernel_group import FusionStrategy


try:
    from runtime import Allocate, Dispatch, Download, FromNumpy, Runtime

    _RUNTIME_AVAILABLE = True
except ImportError:
    _RUNTIME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metadata-only paths: no ShapeOp emitted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_shape,new_shape",
    [
        ((8, 16), (128,)),  # 2D flat
        ((8, 16), (16, 8)),  # 2D reshape
        ((8, 16), (4, 32)),  # 2D reshape, different ratio
        ((8, 16), (32, 4)),  # 2D reshape
        ((128,), (8, 16)),  # 1D → 2D
        ((128,), (16, 8)),  # 1D → 2D
        ((64,), (64,)),  # identity reshape
    ],
)
def test_contiguous_reshape_is_metadata_only(src_shape, new_shape):
    ops = Operations()
    ops.input("X", shape=src_shape)
    view = ops.reshape("X", new_shape, out="Y")
    program = ops.build()
    assert program == [], (
        f"contiguous reshape {src_shape} → {new_shape} should not emit a ShapeOp"
    )
    assert view.buffer_key == "X"
    assert view.shape == new_shape


@pytest.mark.parametrize(
    "src_shape,new_shape,explanation",
    [
        # Same-shape reshape — always a no-op, even on non-contiguous sources.
        ((8, 16), (8, 16), "identity reshape"),
        # Singleton-dim degeneracies: row-major iteration touches a
        # single row/col so the other stride is irrelevant. A col-major
        # (1, K) or (M, 1) view is still dense for reshape purposes.
        ((1, 64), (64,), "(1, K) row-major flatten"),
        ((64, 1), (64,), "(M, 1) row-major squeeze"),
    ],
)
def test_safe_metadata_reshapes_never_emit_shape_op(src_shape, new_shape, explanation):
    ops = Operations()
    ops.input("X", shape=src_shape)
    view = ops.reshape("X", new_shape, out="Y")
    assert ops.build() == [], f"{explanation}: should be a metadata view, no ShapeOp"
    assert view.buffer_key == "X"


@pytest.mark.parametrize(
    "src_shape,row_stride,col_stride,new_shape,expected_strides",
    [
        # 1D with non-unit stride (a strided slice). All reshapes work
        # as views with stride scaled by `c = row_stride`.
        ((12,), 3, 0, (4, 3), (9, 3)),
        ((12,), 3, 0, (3, 4), (12, 3)),
        ((12,), 3, 0, (2, 6), (18, 3)),
        ((12,), 3, 0, (12,), (3, 0)),  # same-shape no-op (carries 1D strides)
        # (1, K) with non-unit col_stride: row dim irrelevant, col_stride
        # is the per-element step.
        ((1, 6), 0, 5, (2, 3), (15, 5)),
        ((1, 6), 0, 5, (3, 2), (10, 5)),
        ((1, 6), 0, 5, (6,), (5, 0)),
        # (M, 1) with non-unit row_stride: symmetric.
        ((6, 1), 7, 0, (2, 3), (21, 7)),
        ((6, 1), 7, 0, (6,), (7, 0)),
        # 2D uniform-step source (rs = K * cs with cs > 1) — also linearly
        # iterable. E.g., a (2, 3) strided slice with cs=4 and rs=12.
        ((2, 3), 12, 4, (6,), (4, 0)),
        ((2, 3), 12, 4, (3, 2), (8, 4)),
        ((2, 3), 12, 4, (1, 6), (24, 4)),
    ],
)
def test_linearly_iterable_source_reshape_as_view(
    src_shape, row_stride, col_stride, new_shape, expected_strides
):
    """A source is a candidate for view-reshape iff its row-major
    iteration produces uniformly-stepped offsets. The output view
    inherits the per-element step `c`, yielding strides `(K'·c, c)` for
    2D targets or `(c, 0)` for 1D — regardless of the source's
    contiguity at unit stride."""
    src = Tensor(
        "x",
        src_shape,
        row_stride=row_stride if row_stride else None,
        col_stride=col_stride if col_stride else None,
    )
    assert src.can_reshape_as_view(new_shape)
    view = src.reshape_view(new_shape, name="v")
    assert view.buffer_key == "x"
    assert view.shape == new_shape
    assert (view.row_stride, view.col_stride) == expected_strides


def test_linear_step_detection():
    # Row-major dense: step 1.
    assert Tensor("x", (4, 8)).linear_step() == 1
    # Col-major non-singleton: not linear.
    assert Tensor("x", (4, 8)).transpose("xt").linear_step() is None
    # Singleton-dim cases: step = the non-singleton stride.
    assert Tensor("x", (1, 6), col_stride=5, row_stride=1).linear_step() == 5
    assert Tensor("x", (6, 1), row_stride=7, col_stride=1).linear_step() == 7
    # 2D uniform-step (rs = K * cs): step = cs.
    assert Tensor("x", (2, 3), row_stride=12, col_stride=4).linear_step() == 4
    # 2D non-uniform (rs ≠ K * cs): not linear.
    assert Tensor("x", (2, 3), row_stride=10, col_stride=1).linear_step() is None


def test_singleton_dim_after_transpose_still_flattens_as_view():
    """A (M, 1) tensor transposed becomes (1, M) col-major (rs=1, cs=M).
    Row-major iteration over (1, M) only touches row 0, so cs is
    irrelevant — the source is effectively dense and flattening is a
    metadata swap."""
    ops = Operations()
    ops.input("X", shape=(64, 1))
    ops.transpose("X", out="Xt")  # shape (1, 64), col-major, strides (1, 64)
    view = ops.reshape("Xt", (64,), out="Y")
    assert ops.build() == []
    assert view.buffer_key == "X"


def test_transpose_then_transpose_is_metadata_only_and_restores_strides():
    ops = Operations()
    ops.input("A", shape=(8, 16))
    ops.transpose("A", out="At")
    a_tt = ops.transpose("At", out="Att")
    program = ops.build()
    assert program == []
    assert a_tt.shape == (8, 16)
    assert a_tt.buffer_key == "A"
    # Round-trip restores the original row-major strides.
    assert (a_tt.row_stride, a_tt.col_stride) == (16, 1)


def test_reshape_chain_on_contiguous_is_metadata_only():
    ops = Operations()
    ops.input("X", shape=(8, 16))
    ops.reshape("X", (128,), out="Xf")
    ops.reshape("Xf", (4, 32), out="Xf_r")
    program = ops.build()
    assert program == []
    assert ops.tensors["Xf_r"].buffer_key == "X"
    assert ops.tensors["Xf_r"].shape == (4, 32)


# ---------------------------------------------------------------------------
# Copy paths: ShapeOp emitted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_shape,new_shape",
    [
        # All of these need a copy: source is col-major after transpose,
        # row-major iteration over the new shape would visit memory in a
        # different order than the source's storage.
        ((8, 16), (128,)),
        ((8, 16), (4, 32)),
        ((8, 16), (32, 4)),
    ],
)
def test_reshape_after_transpose_emits_shape_op(src_shape, new_shape):
    ops = Operations()
    ops.input("X", shape=src_shape)
    ops.transpose("X", out="Xt")
    ops.reshape("Xt", new_shape, out="Y")
    program = ops.build()
    assert len(program) == 1 and isinstance(program[0], ShapeOp)
    assert program[0].input.name == "Xt"
    assert program[0].out.name == "Y"
    assert program[0].out.buffer_key == "Y"  # owns its buffer


def test_same_shape_reshape_on_transposed_is_view():
    """Reshape to the *same* shape as the (transposed) source is always
    a no-op view — no ShapeOp regardless of strides. This is the most
    dire-case guard: the fuser should not pay for a redundant copy."""
    ops = Operations()
    ops.input("X", shape=(8, 16))
    ops.transpose("X", out="Xt")  # Xt shape = (16, 8)
    v = ops.reshape("Xt", (16, 8), out="Y")
    assert ops.build() == []
    assert v.buffer_key == "X"


def test_reshape_after_copy_is_metadata_only():
    """After a ShapeOp materializes a fresh contiguous buffer, any
    further reshape on that fresh tensor is a metadata swap."""
    ops = Operations()
    ops.input("X", shape=(8, 16))
    ops.transpose("X", out="Xt")
    ops.reshape("Xt", (128,), out="Y")  # ShapeOp materializes Y
    ops.reshape("Y", (16, 8), out="Yr")  # Yr is just metadata
    program = ops.build()
    assert len([o for o in program if isinstance(o, ShapeOp)]) == 1
    assert ops.tensors["Yr"].buffer_key == "Y"


# ---------------------------------------------------------------------------
# Fuser + assembly
# ---------------------------------------------------------------------------


def test_shape_op_becomes_standalone_kernel_group():
    ops = Operations()
    ops.input("X", shape=(8, 16))
    ops.transpose("X", out="Xt")
    ops.reshape("Xt", (128,), out="Y")
    decisions = fuse(ops.build())
    assert len(decisions) == 1
    assert decisions[0].strategy is FusionStrategy.STANDALONE_SHAPE
    assert decisions[0].shape_op is not None


def test_shape_op_alongside_matmul_runs_in_separate_kernels():
    """Matmul writes contiguous output → reshape to 1D is metadata,
    no ShapeOp. To get a ShapeOp here we transpose the matmul output
    first, then reshape it. Two kernels: the matmul, then the copy."""
    ops = Operations()
    ops.input("A", shape=(8, 16))
    ops.input("B", shape=(16, 8))
    ops.matmul(a="A", b="B", out="C")  # C is (8, 8) contiguous
    ops.transpose("C", out="Ct")
    ops.reshape("Ct", (64,), out="Cf")
    program = ops.build()
    decisions = fuse(program)
    assert len(decisions) == 2
    strategies = {d.strategy for d in decisions}
    assert FusionStrategy.STANDALONE_MATMUL in strategies
    assert FusionStrategy.STANDALONE_SHAPE in strategies


def test_assembled_shape_kernel_uses_input_strides():
    ops = Operations()
    ops.input("X", shape=(4, 8))  # row-major, strides (8, 1)
    ops.transpose("X", out="Xt")  # col-major, strides (1, 8)
    ops.reshape("Xt", (32,), out="Y")
    group = assemble(fuse(ops.build())[0])
    src = group.kernel.source
    # The transposed strides land as literals in the load expression.
    assert "in_row * 1" in src
    assert "in_col * 8" in src
    # input_cols = source's logical second dim = 4 (Xt.shape[1])
    assert "global_idx / 4" in src
    assert "global_idx % 4" in src
    # Bindings: original X buffer + new Y buffer.
    assert group.bindings == ("X", "Y")


def test_shape_kernel_grid_covers_all_elements():
    ops = Operations()
    ops.input("X", shape=(10, 13))
    ops.transpose("X", out="Xt")
    ops.reshape("Xt", (130,), out="Y")
    group = assemble(fuse(ops.build())[0])
    # 130 elements, block 256 → 1 group; the in-kernel bound check
    # handles the partial block.
    assert group.grid == (1, 1, 1)
    assert group.threads == (256, 1, 1)
    assert group.dims == (130,)


# ---------------------------------------------------------------------------
# End-to-end execution (skipped without Metal backend)
# ---------------------------------------------------------------------------


def _run_shape_kernel(x: np.ndarray, view_builder) -> np.ndarray:
    """Build a program where `view_builder(ops)` constructs a view that
    requires a ShapeOp, dispatch the assembled kernel, and return the
    Y output as numpy. `view_builder` calls ops.input / transpose /
    reshape and returns the output tensor name."""
    ops = Operations()
    out_name = view_builder(ops)
    program = ops.build()
    decisions = fuse(program)
    # We expect a single ShapeOp decision for these tests.
    assert (
        len(decisions) == 1 and decisions[0].strategy is FusionStrategy.STANDALONE_SHAPE
    )
    group = assemble(decisions[0])
    out_shape = ops.tensors[out_name].shape
    N = ops.tensors[out_name].element_count

    env = Runtime(
        (
            FromNumpy("X", x),
            Allocate(out_name, N * 4),
            group.kernel,
        )
    ).run()
    Runtime(
        (
            Dispatch(
                group.kernel,
                bindings=group.bindings,
                dims=group.dims,
                grid=group.grid,
                threads=group.threads,
            ),
            Download(out_name, shape=out_shape, dtype=np.float32, into="result"),
        )
    ).run(env)
    return env["result"]


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason="Metal backend not available")
def test_e2e_transpose_then_flatten():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    result = _run_shape_kernel(
        x,
        lambda ops: (
            ops.input("X", shape=x.shape),
            ops.transpose("X", out="Xt"),
            ops.reshape("Xt", (x.size,), out="Y"),
        )[-1].name,
    )
    np.testing.assert_allclose(result, x.T.reshape(-1), rtol=0, atol=0)


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason="Metal backend not available")
def test_e2e_transpose_then_2d_reshape():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    result = _run_shape_kernel(
        x,
        lambda ops: (
            ops.input("X", shape=x.shape),
            ops.transpose("X", out="Xt"),
            ops.reshape("Xt", (32, 4), out="Y"),
        )[-1].name,
    )
    np.testing.assert_allclose(result, x.T.reshape(32, 4), rtol=0, atol=0)


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason="Metal backend not available")
def test_e2e_partial_block_tail_handled():
    """N=130 doesn't divide block size 256 — the tail-guard `if (idx >= N)`
    must keep the kernel safe."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10, 13)).astype(np.float32)
    result = _run_shape_kernel(
        x,
        lambda ops: (
            ops.input("X", shape=x.shape),
            ops.transpose("X", out="Xt"),
            ops.reshape("Xt", (130,), out="Y"),
        )[-1].name,
    )
    np.testing.assert_allclose(result, x.T.reshape(-1), rtol=0, atol=0)


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason="Metal backend not available")
def test_e2e_grid_covers_multi_block_count():
    """N=1024 → 4 groups of 256 each."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 32)).astype(np.float32)
    result = _run_shape_kernel(
        x,
        lambda ops: (
            ops.input("X", shape=x.shape),
            ops.transpose("X", out="Xt"),
            ops.reshape("Xt", (1024,), out="Y"),
        )[-1].name,
    )
    np.testing.assert_allclose(result, x.T.reshape(-1), rtol=0, atol=0)
