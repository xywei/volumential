import numpy as np
import pytest

import pyopencl as cl

from volumential.tree_interactive_build import BoxTree, QuadratureOnBoxTree


def test_box_tree_refine_and_quadrature(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    tree = BoxTree()
    tree.generate_uniform_boxtree(
        queue, root_vertex=np.array([-1.0, -1.0]), root_extent=2.0, nlevels=2
    )

    assert tree.n_active_boxes == 4

    refine_flags = np.zeros(tree.nboxes, dtype=bool)
    refine_flags[int(tree.active_boxes.get()[0])] = True
    coarsen_flags = np.zeros(tree.nboxes, dtype=bool)
    tree.refine_and_coarsen(refine_flags, coarsen_flags, error_on_ignored_flags=False)

    assert tree.n_active_boxes > 4

    quad = QuadratureOnBoxTree(tree)
    q_points = quad.get_q_points(queue)
    q_weights = quad.get_q_weights(queue)
    cell_centers = quad.get_cell_centers(queue)
    cell_measures = quad.get_cell_measures(queue)

    assert len(q_points) == 2
    assert q_weights.size > 0
    assert len(cell_centers) == 2
    assert cell_measures.size == tree.n_active_boxes


def test_box_tree_coarsen_leaf_flags_reduce_uniform_tree(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    tree = BoxTree()
    tree.generate_uniform_boxtree(
        queue, root_vertex=np.array([-1.0, -1.0]), root_extent=2.0, nlevels=2
    )

    assert tree.n_active_boxes == 4

    refine_flags = np.zeros(tree.nboxes, dtype=bool)
    coarsen_flags = np.zeros(tree.nboxes, dtype=bool)
    coarsen_flags[int(tree.active_boxes.get()[0])] = True

    tree.refine_and_coarsen(
        refine_flags,
        coarsen_flags,
        error_on_ignored_flags=True,
    )

    assert tree.n_active_boxes == 1
    assert np.all(tree.box_levels.get() == 0)


def test_box_tree_coarsen_rejects_nonleaf_flags(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    tree = BoxTree()
    tree.generate_uniform_boxtree(
        queue, root_vertex=np.array([-1.0, -1.0]), root_extent=2.0, nlevels=2
    )

    refine_flags = np.zeros(tree.nboxes, dtype=bool)
    coarsen_flags = np.zeros(tree.nboxes, dtype=bool)
    coarsen_flags[0] = True

    with pytest.raises(ValueError, match="attempting to coarsen non-leaf"):
        tree.refine_and_coarsen(
            refine_flags,
            coarsen_flags,
            error_on_ignored_flags=True,
        )
