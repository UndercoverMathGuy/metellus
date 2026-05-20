"""End-to-end tests for `orchestrator.scheduler.schedule`.

Each test declares ops in order via `Operations`, fuses + assembles,
hands the groups to `schedule(...)`, and runs the resulting `Runtime`
against a real Metal backend. Outputs in `env` are compared against
numpy ground truth. Skipped if `metal_backend` isn't importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator import Operations
from orchestrator.assembly import assemble
from orchestrator.fusion import fuse
from orchestrator.scheduler import schedule


try:
    import metal_backend  # noqa: F401

    _RUNTIME_AVAILABLE = True
except ImportError:
    _RUNTIME_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _RUNTIME_AVAILABLE, reason="Metal backend not available"
)


def _compile(ops: Operations):
    program = ops.build()
    decisions = fuse(program)
    return tuple(assemble(d) for d in decisions)


def test_single_matmul():
    """A @ B = C. One kernel, one named output, downloaded into env['C']."""
    rng = np.random.default_rng(0)
    M, K, N = 32, 16, 24
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(a, "A")
    ops.from_numpy(b, "B")
    ops.matmul(a="A", b="B", out="C")

    groups = _compile(ops)
    env = schedule(ops, groups).run()

    assert "C" in env
    np.testing.assert_allclose(env["C"], a @ b, rtol=1e-4, atol=1e-4)


def test_matmul_then_relu_fused():
    """C = a@b; D = max(C, 0). Fusion absorbs the relu into matmul's
    register epilogue, so C is never materialized — only D is downloaded.
    Confirms env['C'] does NOT exist and env['D'] is correct."""
    rng = np.random.default_rng(1)
    M, K, N = 32, 16, 24
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(a, "A")
    ops.from_numpy(b, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("max", out="D", operands=("C", 0.0))

    groups = _compile(ops)
    # Single fused kernel.
    assert len(groups) == 1
    env = schedule(ops, groups).run()

    assert "D" in env
    np.testing.assert_allclose(env["D"], np.maximum(a @ b, 0), rtol=1e-4, atol=1e-4)
    # C was fused away — no buffer, no download.
    assert "C" not in env


def test_matmul_with_two_consumers():
    """C = a@b; D = max(C, 0); E = abs(C). C has two consumers, so the
    fuser cannot absorb either elementwise into the matmul; we get three
    kernels. C is allocated as an intermediate, read by D's and E's
    kernels, then Freed after its last use. D and E are downloaded.

    Exercises the scheduler's intermediate Allocate + Free path, plus
    last-use computation across multiple groups."""
    rng = np.random.default_rng(1)
    M, K, N = 32, 16, 24
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(a, "A")
    ops.from_numpy(b, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("max", out="D", operands=("C", 0.0))
    ops.elementwise("absolute", out="E", operands=("C",))

    groups = _compile(ops)
    assert len(groups) == 3
    env = schedule(ops, groups).run()

    c_expected = a @ b
    np.testing.assert_allclose(env["C"], c_expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        env["D"], np.maximum(c_expected, 0), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(env["E"], np.abs(c_expected), rtol=1e-4, atol=1e-4)


def test_two_independent_matmuls():
    """Two unrelated matmuls run in one Runtime. Both outputs land in env;
    each gets its own Allocate / Dispatch / Download / Free pair."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)
    c = rng.standard_normal((16, 16)).astype(np.float32)
    d = rng.standard_normal((16, 16)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(a, "A")
    ops.from_numpy(b, "B")
    ops.from_numpy(c, "C")
    ops.from_numpy(d, "D")
    ops.matmul(a="A", b="B", out="AB")
    ops.matmul(a="C", b="D", out="CD")

    groups = _compile(ops)
    assert len(groups) == 2
    env = schedule(ops, groups).run()

    np.testing.assert_allclose(env["AB"], a @ b, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(env["CD"], c @ d, rtol=1e-4, atol=1e-4)


def test_chain_matmul_exp_relu_fused():
    """C = a@b; E = exp(C); D = max(E, 0). Both elementwise ops are
    lane-agnostic (unary exp, scalar-binary max), each has a single
    consumer chain, so the fuser should absorb both into the matmul's
    register epilogue → one kernel, D as the only materialized output."""
    rng = np.random.default_rng(3)
    M, K, N = 64, 32, 48
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(a, "A")
    ops.from_numpy(b, "B")
    ops.matmul(a="A", b="B", out="C")
    ops.elementwise("exp", out="E", operands=("C",))
    ops.elementwise("max", out="D", operands=("E", 0.0))

    groups = _compile(ops)
    assert len(groups) == 1, f"expected 1 fused kernel, got {len(groups)}"
    env = schedule(ops, groups).run()

    expected = np.maximum(np.exp(a @ b), 0)
    np.testing.assert_allclose(env["D"], expected, rtol=1e-3, atol=1e-3)
    assert "C" not in env
    assert "E" not in env


def test_download_transpose_view():
    """A is (M, K) uploaded; At = transpose(A) is a metadata-only view
    with buffer_key='A' and swapped strides. We use At as a matmul input
    (A.T @ B) and ALSO declare it for export. The scheduler emits a
    strided Download for At — host-side gather of A's flat row-major
    bytes via At's (col-major) strides — and compares it against np a.T.
    Confirms the non-contiguous Download path produces a true logical
    transpose, not just reshaped bytes."""
    rng = np.random.default_rng(4)
    M, K, N = 8, 6, 5
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((M, N)).astype(np.float32)

    ops = Operations()
    ops.from_numpy(a, "A")
    ops.from_numpy(b, "B")
    ops.transpose("A", out="At")  # (K, M) view of A
    ops.matmul(a="At", b="B", out="C")  # (K, N)

    groups = _compile(ops)
    env = schedule(ops, groups).run()

    np.testing.assert_allclose(env["At"], a.T, rtol=0, atol=0)
    np.testing.assert_allclose(env["C"], a.T @ b, rtol=1e-4, atol=1e-4)


def test_schedule_rejects_missing_upload():
    """Declaring an input without from_numpy is a hard error at schedule
    time — caught before any kernel runs."""
    ops = Operations()
    ops.input("A", shape=(8, 8))
    ops.input("B", shape=(8, 8))
    ops.matmul(a="A", b="B", out="C")
    groups = _compile(ops)
    with pytest.raises(ValueError, match="no from_numpy upload"):
        schedule(ops, groups)


def test_schedule_rejects_dtype_mismatch():
    """from_numpy must reject non-float32 arrays so float64 widening can't
    silently slip through."""
    ops = Operations()
    arr = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        ops.from_numpy(arr, "A")


def test_schedule_rejects_from_numpy_on_view():
    """Views are not storage owners — from_numpy on a view is rejected."""
    ops = Operations()
    a = np.zeros((4, 4), dtype=np.float32)
    ops.from_numpy(a, "A")
    ops.transpose("A", out="At")
    with pytest.raises(ValueError, match="view"):
        ops.from_numpy(a, "At")
