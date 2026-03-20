import numpy as np

from volumential.meshgen import _square_bbox_for_treebuilder


def test_square_bbox_for_treebuilder_expands_rectangular_box():
    bbox = np.array([[0.0, 2.0], [10.0, 13.0]], dtype=np.float64)

    square = _square_bbox_for_treebuilder(bbox)

    assert np.allclose(square[:, 0], bbox[:, 0])
    assert np.allclose(square[:, 1], np.array([3.0, 13.0]))


def test_square_bbox_for_treebuilder_keeps_square_box():
    bbox = np.array([[-1.0, 2.0], [4.0, 7.0], [8.0, 11.0]], dtype=np.float64)

    square = _square_bbox_for_treebuilder(bbox)

    assert np.allclose(square, bbox)
