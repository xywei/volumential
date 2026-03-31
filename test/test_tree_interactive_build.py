import numpy as np
import pytest

import pyopencl as cl

from boxtree import refine_and_coarsen_tree_of_boxes

from volumential.tree_interactive_build import (
    BoxTree,
    QuadratureOnBoxTree,
    _box_keys_from_geometry,
    _coarsen_tree_of_boxes_compat,
    _enforce_level_restriction,
    _prune_unreachable_boxes,
    _rebuild_tob_from_geometry,
    _remap_bool_flags_by_box_key,
    _resize_bool_flags,
)


def _box_keys_from_tob(tob):
    return _box_keys_from_geometry(tob)


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


def test_box_tree_refine_and_coarsen_resizes_short_flags(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    tree = BoxTree()
    tree.generate_uniform_boxtree(
        queue, root_vertex=np.array([-1.0, -1.0]), root_extent=2.0, nlevels=2
    )

    refine_flags = np.zeros(0, dtype=bool)
    coarsen_flags = np.zeros(1, dtype=bool)

    tree.refine_and_coarsen(
        refine_flags,
        coarsen_flags,
        error_on_ignored_flags=True,
    )

    assert tree.n_active_boxes == 4


def test_box_tree_refine_and_coarsen_rejects_oversized_flags(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    tree = BoxTree()
    tree.generate_uniform_boxtree(
        queue, root_vertex=np.array([-1.0, -1.0]), root_extent=2.0, nlevels=2
    )

    refine_flags = np.zeros(tree.nboxes + 1, dtype=bool)
    coarsen_flags = np.zeros(tree.nboxes, dtype=bool)

    with pytest.raises(
        ValueError,
        match="flag arrays longer than the current number of boxes must be remapped",
    ):
        tree.refine_and_coarsen(
            refine_flags,
            coarsen_flags,
            error_on_ignored_flags=True,
        )


def test_box_tree_mixed_refine_coarsen_remaps_coarsen_flags(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    tree = BoxTree()
    tree.generate_uniform_boxtree(
        queue, root_vertex=np.array([-1.0, -1.0]), root_extent=2.0, nlevels=3
    )

    rng = np.random.default_rng(7)
    old_tob = tree._tree
    scenario = None

    for _ in range(30):
        leaves = np.where(np.all(old_tob.box_child_ids == 0, axis=0))[0]
        if leaves.size == 0:
            break

        warm_pick = rng.choice(leaves, size=min(2, leaves.size), replace=False)
        warm_refine = np.zeros(old_tob.nboxes, dtype=bool)
        warm_refine[warm_pick] = True
        old_tob = refine_and_coarsen_tree_of_boxes(
            old_tob,
            refine_flags=warm_refine,
            coarsen_flags=None,
            error_on_ignored_flags=True,
        )

        old_leaf = np.where(np.all(old_tob.box_child_ids == 0, axis=0))[0]
        old_parents = np.asarray(old_tob.box_parent_ids, dtype=np.int64)
        old_keys = _box_keys_from_tob(old_tob)

        coarsen_candidates = []
        for coarsen_leaf in old_leaf:
            parent_id = int(old_parents[int(coarsen_leaf)])
            if parent_id < 0:
                continue

            siblings = np.asarray(old_tob.box_child_ids)[:, parent_id]
            if np.any(siblings == 0):
                continue
            if not np.all(np.all(old_tob.box_child_ids[:, siblings] == 0, axis=0)):
                continue

            coarsen_candidates.append(int(coarsen_leaf))

        if not coarsen_candidates:
            continue

        for _ in range(80):
            coarsen_leaf = int(rng.choice(coarsen_candidates))
            refine_leaf = int(rng.choice(old_leaf))
            if refine_leaf == coarsen_leaf:
                continue

            refine_flags = np.zeros(old_tob.nboxes, dtype=bool)
            refine_flags[refine_leaf] = True

            refined_tob = refine_and_coarsen_tree_of_boxes(
                old_tob,
                refine_flags=refine_flags,
                coarsen_flags=None,
                error_on_ignored_flags=True,
            )

            refined_keys = _box_keys_from_tob(refined_tob)
            mapped_id = {key: idx for idx, key in enumerate(refined_keys)}.get(
                old_keys[coarsen_leaf]
            )
            if mapped_id is None or mapped_id == coarsen_leaf:
                continue

            parent_key = (
                old_keys[coarsen_leaf][0] - 1,
                tuple(v // 2 for v in old_keys[coarsen_leaf][1]),
            )

            wrong_parent_key = None
            if coarsen_leaf < refined_tob.nboxes:
                wrong_key = refined_keys[coarsen_leaf]
                wrong_parent_key = (
                    wrong_key[0] - 1,
                    tuple(v // 2 for v in wrong_key[1]),
                )

            if wrong_parent_key == parent_key:
                continue

            candidate_coarsen_flags = np.zeros(old_tob.nboxes, dtype=bool)
            candidate_coarsen_flags[coarsen_leaf] = True
            remapped_candidate_flags = _remap_bool_flags_by_box_key(
                candidate_coarsen_flags,
                old_tob,
                refined_tob,
            )
            try:
                _coarsen_tree_of_boxes_compat(
                    refined_tob,
                    remapped_candidate_flags,
                    error_on_ignored_flags=True,
                )
            except (RuntimeError, ValueError):
                continue

            scenario = (old_tob, refine_flags, coarsen_leaf)
            break

        if scenario is not None:
            break

    assert scenario is not None, (
        "failed to find a scenario where coarsen flags require remapping; "
        "try a different seed or increase iteration limits"
    )

    old_tob, refine_flags, coarsen_leaf = scenario
    coarsen_flags = np.zeros(old_tob.nboxes, dtype=bool)
    coarsen_flags[coarsen_leaf] = True

    def canonicalize_tob(tob):
        tob = _rebuild_tob_from_geometry(tob)
        tob = _enforce_level_restriction(tob)
        tob = _rebuild_tob_from_geometry(tob)
        tob = _prune_unreachable_boxes(tob)
        return tob

    refined_tob = refine_and_coarsen_tree_of_boxes(
        old_tob,
        refine_flags=refine_flags,
        coarsen_flags=None,
        error_on_ignored_flags=True,
    )
    remapped_coarsen_flags = _remap_bool_flags_by_box_key(
        coarsen_flags,
        old_tob,
        refined_tob,
    )
    expected_with_remap = _coarsen_tree_of_boxes_compat(
        refined_tob,
        remapped_coarsen_flags,
        error_on_ignored_flags=True,
    )
    expected_with_remap = canonicalize_tob(expected_with_remap)
    expected_with_remap_keys = _box_keys_from_tob(expected_with_remap)

    expected_without_remap_keys = None
    naive_coarsen_flags = _resize_bool_flags(coarsen_flags, refined_tob.nboxes)
    try:
        expected_without_remap = _coarsen_tree_of_boxes_compat(
            refined_tob,
            naive_coarsen_flags,
            error_on_ignored_flags=True,
        )
    except (ValueError, RuntimeError):
        expected_without_remap = None

    if expected_without_remap is not None:
        expected_without_remap = canonicalize_tob(expected_without_remap)
        expected_without_remap_keys = _box_keys_from_tob(expected_without_remap)

    tree._tree = old_tob

    tree.refine_and_coarsen(
        refine_flags,
        coarsen_flags,
        error_on_ignored_flags=True,
    )

    actual_keys = _box_keys_from_tob(tree._tree)
    assert actual_keys == expected_with_remap_keys
    if expected_without_remap_keys is not None:
        assert actual_keys != expected_without_remap_keys
