import pytest

from orchestrator import Operations, build_dag
from orchestrator.ir import ElementwiseOp, MatmulOp, ReductionOp, Tensor


def _simple_chain():
    ops = Operations()
    ops.input("a", shape=(8, 16))
    ops.input("b", shape=(16, 4))
    ops.input("bias", shape=(1, 4))
    ops.matmul(a="a", b="b", out="c")
    ops.elementwise("add", out="d", operands=("c", "bias"), y_broadcast="row")
    ops.elementwise("exp", out="e", operands=("d",))
    ops.reduction("sum", out="r", x="e")
    return ops.build()


def test_dag_producer_and_consumer_maps():
    program = _simple_chain()
    dag = build_dag(program)

    assert set(dag.producer_of) == {"c", "d", "e", "r"}
    assert isinstance(dag.producer_of["c"], MatmulOp)
    assert isinstance(dag.producer_of["d"], ElementwiseOp)
    assert isinstance(dag.producer_of["e"], ElementwiseOp)
    assert isinstance(dag.producer_of["r"], ReductionOp)

    assert dag.consumers_of["c"] == (dag.producer_of["d"],)
    assert dag.consumers_of["d"] == (dag.producer_of["e"],)
    assert dag.consumers_of["e"] == (dag.producer_of["r"],)
    assert "r" not in dag.consumers_of


def test_dag_inputs_and_outputs():
    dag = build_dag(_simple_chain())
    assert dag.inputs == frozenset({"a", "b", "bias"})
    assert dag.outputs == frozenset({"r"})


def test_unique_consumer_returns_none_when_zero_or_many():
    ops = Operations()
    ops.input("x", shape=(4, 4))
    ops.elementwise("exp", out="y", operands=("x",))
    ops.elementwise("log", out="z", operands=("x",))
    dag = build_dag(ops.build())
    assert dag.unique_consumer("x") is None  # two consumers
    assert dag.unique_consumer("y") is None  # zero consumers
    assert dag.unique_consumer("z") is None


def test_unique_consumer_returns_op_when_single():
    program = _simple_chain()
    dag = build_dag(program)
    assert dag.unique_consumer("c") is dag.producer_of["d"]


def test_topo_order_is_program_order():
    program = _simple_chain()
    dag = build_dag(program)
    assert dag.topo_order() == program
    # returns a fresh list — mutation must not affect the DAG
    order = dag.topo_order()
    order.clear()
    assert dag.topo_order() == program


def test_ssa_violation_raises():
    # Build a hand-rolled program that violates SSA (same out name twice).
    t_x = Tensor("x", (4, 4))
    t_y = Tensor("y", (4, 4))
    op1 = ElementwiseOp(out=t_y, op="exp", operands=(t_x,))
    op2 = ElementwiseOp(out=t_y, op="log", operands=(t_x,))
    with pytest.raises(ValueError, match="SSA violation"):
        build_dag([op1, op2])


def test_def_after_use_raises():
    # Operand of op1 is produced by op2 (later in list) → topological violation.
    t_x = Tensor("x", (4, 4))
    t_y = Tensor("y", (4, 4))
    t_z = Tensor("z", (4, 4))
    op1 = ElementwiseOp(out=t_z, op="exp", operands=(t_y,))
    op2 = ElementwiseOp(out=t_y, op="log", operands=(t_x,))
    with pytest.raises(ValueError, match="def-after-use"):
        build_dag([op1, op2])
