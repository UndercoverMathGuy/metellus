import pytest

from orchestrator import (
    BroadcastSpec,
    ElementwiseOp,
    Operations,
    Scalar,
    Tensor,
    build_dag,
)


def test_relu_via_max_with_scalar():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    ops.elementwise("max", out="y", operands=("x", 0.0))
    program = ops.build()
    op = program[0]
    assert isinstance(op, ElementwiseOp)
    assert op.op == "max"
    assert isinstance(op.operands[0], Tensor)
    assert isinstance(op.operands[1], Scalar)
    assert op.operands[1].value == 0.0
    # inputs filters scalars — DAG only sees tensors
    assert op.inputs == (op.operands[0],)


def test_scalar_operand_excluded_from_dag_inputs():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    ops.elementwise("max", out="y", operands=("x", 0.0))
    dag = build_dag(ops.build())
    assert dag.inputs == frozenset({"x"})
    assert dag.outputs == frozenset({"y"})


def test_scalar_with_broadcast_set_raises():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    with pytest.raises(ValueError, match="cannot be set for a Scalar"):
        ops.elementwise(
            "max", out="y", operands=("x", 0.0), y_broadcast=BroadcastSpec.SCALAR
        )


def test_primary_must_be_tensor():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    with pytest.raises(ValueError, match="operand\\[0\\] must be a tensor"):
        ops.elementwise("max", out="y", operands=(0.0, "x"))


def test_invalid_operand_type_raises():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    with pytest.raises(ValueError, match="must be a tensor name"):
        ops.elementwise("max", out="y", operands=("x", [1.0]))


def test_string_broadcast_still_accepted():
    ops = Operations()
    ops.input("a", shape=(4, 8))
    ops.input("bias", shape=(1, 8))
    ops.elementwise("add", out="b", operands=("a", "bias"), y_broadcast="row")
    program = ops.build()
    assert program[0].y_broadcast is BroadcastSpec.ROW


def test_bool_not_accepted_as_scalar():
    # bool is a subclass of int but we exclude it — it's almost always a bug.
    ops = Operations()
    ops.input("x", shape=(4, 4))
    with pytest.raises(ValueError, match="must be a tensor name"):
        ops.elementwise("max", out="y", operands=("x", True))
