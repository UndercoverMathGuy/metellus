import pytest

from orchestrator import BroadcastSpec, Layout, Operations, Tensor


def test_default_strides_row_major_2d():
    t = Tensor("a", (8, 16))
    assert t.layout is Layout.ROW_MAJOR
    assert t.row_stride == 16
    assert t.col_stride == 1


def test_default_strides_col_major_2d():
    t = Tensor("a", (8, 16), layout=Layout.COL_MAJOR)
    assert t.row_stride == 1
    assert t.col_stride == 8


def test_default_strides_1d():
    t = Tensor("v", (8,))
    assert t.row_stride == 1
    assert t.col_stride == 0


def test_tensor_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="positive"):
        Tensor("z", (0, 4))
    with pytest.raises(ValueError, match="positive"):
        Tensor("n", (-1, 4))


def test_explicit_strides_preserved():
    t = Tensor("a", (4, 4), row_stride=8, col_stride=1)
    assert t.row_stride == 8
    assert t.col_stride == 1


def test_transpose_is_metadata_swap():
    a = Tensor("a", (8, 16))  # row-major: rs=16 cs=1
    at = a.transpose("a_T")
    assert at.name == "a_T"  # fresh SSA value
    assert at.buffer_key == "a"  # but aliases a's buffer
    assert at.shape == (16, 8)
    assert at.layout is Layout.COL_MAJOR
    assert at.row_stride == 1
    assert at.col_stride == 16

    # transpose of transpose round-trips shape and strides
    att = at.transpose("a_TT")
    assert att.shape == a.shape
    assert att.row_stride == a.row_stride
    assert att.col_stride == a.col_stride
    assert att.layout is a.layout
    assert att.buffer_key == "a"  # still aliasing the original


def test_transpose_rejects_non_2d():
    with pytest.raises(ValueError, match="2D-only"):
        Tensor("v", (8,)).transpose("v_T")


def test_builder_input_accepts_layout_and_strides():
    ops = Operations()
    t = ops.input("a", shape=(4, 4), layout=Layout.COL_MAJOR)
    assert t.layout is Layout.COL_MAJOR
    assert t.row_stride == 1
    assert t.col_stride == 4


def test_broadcast_spec_rejects_typo():
    with pytest.raises(ValueError):
        BroadcastSpec("rrow")


def test_broadcast_spec_string_compat():
    assert BroadcastSpec.ROW == "row"
    assert "row" == BroadcastSpec.ROW
