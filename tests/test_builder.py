import pytest
import numpy as np

from orchestrator import (
    ElementwiseOp,
    MatmulOp,
    Operations,
    ReductionOp,
    Tensor,
)


def test_matmul_infers_shape_from_declared_inputs():
    ops = Operations()
    ops.input("a", shape=(8, 16))
    ops.input("b", shape=(16, 4))
    c = ops.matmul(a="a", b="b", out="c")
    assert c.shape == (8, 4)
    program = ops.build()
    assert len(program) == 1
    assert isinstance(program[0], MatmulOp)
    assert program[0].out is c


def test_matmul_requires_declared_inputs():
    ops = Operations()
    with pytest.raises(ValueError, match="not declared"):
        ops.matmul(a="x", b="y", out="z")


def test_matmul_contraction_mismatch_raises():
    ops = Operations()
    ops.input("a", shape=(4, 8))
    ops.input("b", shape=(7, 4))
    with pytest.raises(ValueError, match="contraction-axis"):
        ops.matmul(a="a", b="b", out="c")


def test_elementwise_unary_infers_shape():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    y = ops.elementwise("exp", out="y", operands=("x",))
    assert y.shape == (4, 4)
    program = ops.build()
    assert isinstance(program[0], ElementwiseOp)
    assert program[0].op == "exp"


def test_elementwise_binary_row_broadcast():
    ops = Operations()
    ops.input("a", shape=(4, 8))
    ops.input("bias", shape=(1, 8))
    ops.elementwise("add", out="b", operands=("a", "bias"), y_broadcast="row")
    assert ops.tensors["b"].shape == (4, 8)


def test_elementwise_broadcast_shape_mismatch_raises():
    ops = Operations()
    ops.input("a", shape=(4, 8))
    ops.input("bias", shape=(4, 1))
    with pytest.raises(ValueError, match="incompatible with y_broadcast"):
        ops.elementwise("add", out="b", operands=("a", "bias"), y_broadcast="row")


def test_elementwise_unknown_op_raises():
    ops = Operations()
    ops.input("x", shape=(2, 2))
    with pytest.raises(ValueError):
        ops.elementwise("nope", out="y", operands=("x",))


def test_elementwise_arity_mismatch_raises():
    ops = Operations()
    ops.input("x", shape=(2, 2))
    with pytest.raises(ValueError, match="arity"):
        ops.elementwise("exp", out="y", operands=("x", "x"))


def test_reduction_drops_last_axis():
    ops = Operations()
    ops.input("x", shape=(8, 16))
    r = ops.reduction("sum", out="r", x="x")
    assert r.shape == (8,)
    program = ops.build()
    assert isinstance(program[0], ReductionOp)


def test_reduction_non_last_axis_raises():
    ops = Operations()
    ops.input("x", shape=(8, 16))
    with pytest.raises(ValueError, match="last axis"):
        ops.reduction("sum", out="r", x="x", axis=0)


def test_double_output_raises():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    ops.elementwise("exp", out="y", operands=("x",))
    with pytest.raises(ValueError, match="SSA"):
        ops.elementwise("log", out="y", operands=("x",))


def test_undeclared_operand_raises():
    ops = Operations()
    with pytest.raises(ValueError, match="not declared"):
        ops.elementwise("exp", out="y", operands=("ghost",))


def test_from_numpy_rejects_too_small_strided_storage():
    ops = Operations()
    ops.input("A", shape=(4, 4), row_stride=8, col_stride=1)
    with pytest.raises(ValueError, match="can address"):
        ops.from_numpy(np.zeros((4, 4), dtype=np.float32), "A")


def test_full_program_roundtrip():
    ops = Operations()
    ops.input("a", shape=(8, 16))
    ops.input("b", shape=(16, 4))
    ops.input("bias", shape=(1, 4))
    ops.matmul(a="a", b="b", out="c")
    ops.elementwise("add", out="d", operands=("c", "bias"), y_broadcast="row")
    ops.elementwise("exp", out="e", operands=("d",))
    ops.reduction("sum", out="r", x="e")
    program = ops.build()
    assert [type(op).__name__ for op in program] == [
        "MatmulOp",
        "ElementwiseOp",
        "ElementwiseOp",
        "ReductionOp",
    ]
    assert isinstance(ops.tensors["r"], Tensor)
    assert ops.tensors["r"].shape == (8,)
