from __future__ import annotations

import logging
import os
import time
import warnings
from collections import deque
from itertools import product

import numpy as np

import pyopencl as cl
import pyopencl.array
from pytools.obj_array import new_1d as obj_array_1d

from boxtree import (
    Tree,
    box_flags_enum,
    make_tree_of_boxes_root,
    refine_and_coarsen_tree_of_boxes,
    uniformly_refine_tree_of_boxes,
)

logger = logging.getLogger(__name__)


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default

    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"ignoring invalid integer for {name}: {raw!r}",
            stacklevel=3,
        )
        return default


def _level_restriction_debug(msg):
    print(f"[volumential.level_restriction] {msg}", flush=True)


def _leaf_boxes_numpy(tob):
    return np.where(np.all(tob.box_child_ids == 0, axis=0))[0]


def _box_size(root_extent, level):
    return root_extent * (0.5 ** int(level))


def _are_adjacent(root_extent, levels, centers, ibox, jbox, tol=1.0e-15):
    si = _box_size(root_extent, levels[ibox])
    sj = _box_size(root_extent, levels[jbox])
    dist = np.max(np.abs(centers[:, ibox] - centers[:, jbox]))
    return dist <= 0.5 * (si + sj) + tol


def _child_slot_bits(dim):
    nchildren = 2**dim
    bits = np.empty((nchildren, dim), dtype=np.int64)
    for child_slot in range(nchildren):
        bits[child_slot, :] = [
            (child_slot >> (dim - 1 - iaxis)) & 1 for iaxis in range(dim)
        ]
    return bits


def _in_bounds_grid_index(level, grid_index):
    upper = 1 << int(level)
    return all(0 <= int(v) < upper for v in grid_index)


def _covering_leaf_key(leaf_keys, query_level, query_index):
    query_level = int(query_level)
    query_index = tuple(int(v) for v in query_index)

    for level in range(query_level, -1, -1):
        shift = query_level - level
        ancestor_index = tuple(v >> shift for v in query_index)
        key = (level, ancestor_index)
        if key in leaf_keys:
            return key

    return None


def _leaf_keys_from_tob(tob):
    _, levels, _, _, grid_indices = _geometry_grid_indices(tob)
    leaves = _leaf_boxes_numpy(tob)

    return {
        (int(levels[ibox]), tuple(int(v) for v in grid_indices[ibox]))
        for ibox in leaves
    }


def _tree_of_boxes_from_leaf_keys(template_tob, leaf_keys):
    from boxtree.tree import TreeOfBoxes

    dim = int(template_tob.dimensions)
    child_slot_bits = _child_slot_bits(dim)
    nchildren = int(child_slot_bits.shape[0])

    _, _, root_extent, root_min, _ = _geometry_grid_indices(template_tob)

    all_keys = set()
    for level, idx in leaf_keys:
        level = int(level)
        idx = tuple(int(v) for v in idx)

        all_keys.add((level, idx))
        parent_level = level
        parent_idx = idx
        while parent_level > 0:
            parent_idx = tuple(v // 2 for v in parent_idx)
            parent_level -= 1
            all_keys.add((parent_level, parent_idx))

    expected_root = tuple(0 for _ in range(dim))
    if (0, expected_root) not in all_keys:
        all_keys.add((0, expected_root))

    ordered_keys = sorted(all_keys, key=lambda key: (key[0],) + key[1])
    nboxes = len(ordered_keys)

    key_to_box_id = {key: ibox for ibox, key in enumerate(ordered_keys)}

    box_levels = np.asarray(
        [level for level, _ in ordered_keys],
        dtype=template_tob.box_level_dtype,
    )
    box_grid_indices = np.asarray([idx for _, idx in ordered_keys], dtype=np.int64)

    box_sizes = root_extent * np.exp2(-box_levels.astype(np.float64))
    box_centers = (
        root_min.reshape(dim, 1)
        + (box_grid_indices.T.astype(np.float64) + 0.5) * box_sizes
    )

    box_parent_ids = np.full(nboxes, -1, dtype=template_tob.box_id_dtype)
    box_child_ids = np.zeros((nchildren, nboxes), dtype=template_tob.box_id_dtype)

    for ibox, (level, idx) in enumerate(ordered_keys):
        idx_arr = np.asarray(idx, dtype=np.int64)

        if level > 0:
            parent_key = (level - 1, tuple((idx_arr // 2).tolist()))
            parent_id = key_to_box_id.get(parent_key)
            if parent_id is None:
                raise ValueError("missing parent while building balanced tree-of-boxes")
            box_parent_ids[ibox] = parent_id

        for child_slot, bits in enumerate(child_slot_bits):
            child_key = (level + 1, tuple((2 * idx_arr + bits).tolist()))
            child_id = key_to_box_id.get(child_key)
            if child_id is not None:
                box_child_ids[child_slot, ibox] = child_id

    max_level = int(np.max(box_levels)) if box_levels.size else 0
    level_counts = np.bincount(
        box_levels.astype(np.int64),
        minlength=max_level + 1,
    )
    level_start_box_nrs = np.zeros(max_level + 2, dtype=template_tob.box_id_dtype)
    level_start_box_nrs[1:] = np.cumsum(level_counts).astype(template_tob.box_id_dtype)

    return TreeOfBoxes(
        box_centers=np.asarray(box_centers, dtype=template_tob.coord_dtype),
        root_extent=template_tob.root_extent,
        box_parent_ids=np.asarray(box_parent_ids, dtype=template_tob.box_id_dtype),
        box_child_ids=np.asarray(box_child_ids, dtype=template_tob.box_id_dtype),
        box_levels=np.asarray(box_levels, dtype=template_tob.box_level_dtype),
        box_flags=np.asarray(
            _compute_box_flags(box_child_ids), dtype=box_flags_enum.dtype
        ),
        level_start_box_nrs=np.asarray(
            level_start_box_nrs,
            dtype=template_tob.box_id_dtype,
        ),
        box_id_dtype=template_tob.box_id_dtype,
        box_level_dtype=template_tob.box_level_dtype,
        coord_dtype=template_tob.coord_dtype,
        sources_have_extent=template_tob.sources_have_extent,
        targets_have_extent=template_tob.targets_have_extent,
        extent_norm=template_tob.extent_norm,
        stick_out_factor=template_tob.stick_out_factor,
        _is_pruned=template_tob._is_pruned,
    )


def _enforce_level_restriction(tob):
    """Enforce only 2:1 leaf-level balancing.

    Two adjacent leaves are allowed to differ by at most one level.
    No extra same-level colleague balancing is applied.
    """
    tob = _rebuild_tob_from_geometry(tob)

    debug = _env_flag("VOLUMENTIAL_LEVEL_RESTRICTION_DEBUG")
    profile = _env_flag("VOLUMENTIAL_LEVEL_RESTRICTION_PROFILE")
    profile_log_interval = max(
        1,
        _env_int("VOLUMENTIAL_LEVEL_RESTRICTION_PROFILE_LOG_INTERVAL", 50_000),
    )
    log_every = max(1, _env_int("VOLUMENTIAL_LEVEL_RESTRICTION_LOG_EVERY", 1))

    max_splits = _env_int("VOLUMENTIAL_LEVEL_RESTRICTION_MAX_ITERS", 0)
    if max_splits <= 0:
        max_splits = None

    max_work_items = _env_int("VOLUMENTIAL_LEVEL_RESTRICTION_MAX_PAIR_CHECKS", 0)
    if max_work_items <= 0:
        max_work_items = None

    max_nboxes = _env_int("VOLUMENTIAL_LEVEL_RESTRICTION_MAX_NBOXES", 0)
    if max_nboxes <= 0:
        max_nboxes = None

    max_requirement_cells = _env_int(
        "VOLUMENTIAL_LEVEL_RESTRICTION_MAX_LEAF_PAIR_TOTAL",
        0,
    )
    if max_requirement_cells <= 0:
        max_requirement_cells = None

    if max_nboxes is not None and tob.nboxes > max_nboxes:
        raise RuntimeError(
            "level-restriction nboxes limit exceeded "
            f"(nboxes={tob.nboxes}, max_nboxes={max_nboxes})"
        )

    leaf_keys = _leaf_keys_from_tob(tob)
    if not leaf_keys:
        return tob

    original_leaf_keys = set(leaf_keys)
    max_initial_leaf_level = max(level for level, _ in leaf_keys)
    max_allowed_level = _env_int(
        "VOLUMENTIAL_LEVEL_RESTRICTION_MAX_LEVEL",
        max_initial_leaf_level,
    )
    if max_allowed_level < max_initial_leaf_level:
        warnings.warn(
            "VOLUMENTIAL_LEVEL_RESTRICTION_MAX_LEVEL is below the current leaf "
            "maximum; clamping to the current maximum leaf level",
            stacklevel=3,
        )
        max_allowed_level = max_initial_leaf_level

    start_time = time.monotonic()
    dim = int(tob.dimensions)
    nchildren = 2**dim
    child_slot_bits = _child_slot_bits(dim)
    neighbor_offsets = tuple(product((-1, 0, 1), repeat=dim))

    split_count = 0
    work_items = 0

    worklist = deque()
    required_level_by_cell = {}

    def add_requirement(cell_level, cell_index, min_level):
        if min_level <= 0:
            return
        if not _in_bounds_grid_index(cell_level, cell_index):
            return

        key = (int(cell_level), tuple(int(v) for v in cell_index))
        old = required_level_by_cell.get(key)
        if old is None or min_level > old:
            required_level_by_cell[key] = int(min_level)
            worklist.append(key)

    for leaf_level, leaf_index in leaf_keys:
        min_neighbor_level = leaf_level - 1
        if min_neighbor_level <= 0:
            continue

        for offset in neighbor_offsets:
            neighbor_index = tuple(
                leaf_index[iaxis] + offset[iaxis] for iaxis in range(dim)
            )
            add_requirement(leaf_level, neighbor_index, min_neighbor_level)

    while worklist:
        work_items += 1
        if max_work_items is not None and work_items > max_work_items:
            raise RuntimeError(
                "level-restriction work-item limit exceeded "
                f"(work_items={work_items}, split_count={split_count})"
            )

        cell_level, cell_index = worklist.popleft()
        required_level = required_level_by_cell[(cell_level, cell_index)]

        covering_leaf = _covering_leaf_key(leaf_keys, cell_level, cell_index)
        if covering_leaf is None:
            continue

        leaf_level, leaf_index = covering_leaf
        if leaf_level >= required_level:
            continue

        if max_splits is not None and split_count >= max_splits:
            raise RuntimeError(
                "level-restriction split limit exceeded "
                f"(splits={split_count}, work_items={work_items})"
            )

        child_level = leaf_level + 1
        if child_level > max_allowed_level:
            raise RuntimeError(
                "level-restriction max-level bound exceeded "
                f"(requested_level={child_level}, max_allowed_level={max_allowed_level}, "
                f"cell_level={cell_level}, required_level={required_level})"
            )

        leaf_keys.remove(covering_leaf)
        leaf_index_arr = np.asarray(leaf_index, dtype=np.int64)

        new_children = []
        for child_bits in child_slot_bits:
            child_index = tuple((2 * leaf_index_arr + child_bits).tolist())
            child_key = (child_level, child_index)
            leaf_keys.add(child_key)
            new_children.append(child_key)

        split_count += 1
        if max_nboxes is not None:
            projected_nboxes = tob.nboxes + split_count * nchildren
            if projected_nboxes > max_nboxes:
                raise RuntimeError(
                    "level-restriction nboxes limit exceeded "
                    f"(projected_nboxes={projected_nboxes}, max_nboxes={max_nboxes}, "
                    f"split_count={split_count})"
                )

        worklist.append((cell_level, cell_index))

        for new_level, new_index in new_children:
            min_neighbor_level = new_level - 1
            if min_neighbor_level <= 0:
                continue

            for offset in neighbor_offsets:
                neighbor_index = tuple(
                    new_index[iaxis] + offset[iaxis] for iaxis in range(dim)
                )
                add_requirement(new_level, neighbor_index, min_neighbor_level)

        if (
            max_requirement_cells is not None
            and len(required_level_by_cell) > max_requirement_cells
        ):
            raise RuntimeError(
                "level-restriction requirement-map limit exceeded "
                f"(requirement_cells={len(required_level_by_cell)}, "
                f"max_requirement_cells={max_requirement_cells}, "
                f"split_count={split_count})"
            )

        if debug and split_count % log_every == 0:
            _level_restriction_debug(
                "closure "
                f"splits={split_count} work_items={work_items} "
                f"leaf_count={len(leaf_keys)} queue={len(worklist)}"
            )

        if profile and work_items % profile_log_interval == 0:
            _level_restriction_debug(
                "profile stage=closure-progress "
                f"elapsed={time.monotonic() - start_time:.2f}s "
                f"splits={split_count} work_items={work_items} "
                f"leaf_count={len(leaf_keys)} "
                f"requirement_cells={len(required_level_by_cell)}"
            )

    if leaf_keys == original_leaf_keys:
        if debug or profile:
            _level_restriction_debug(
                "done "
                f"elapsed={time.monotonic() - start_time:.2f}s "
                f"splits=0 nboxes={tob.nboxes} nleaves={len(leaf_keys)}"
            )
        return tob

    balanced_tob = _tree_of_boxes_from_leaf_keys(tob, leaf_keys)
    balanced_tob = _rebuild_tob_from_geometry(balanced_tob)

    if max_nboxes is not None and balanced_tob.nboxes > max_nboxes:
        raise RuntimeError(
            "level-restriction nboxes limit exceeded "
            f"(nboxes={balanced_tob.nboxes}, max_nboxes={max_nboxes})"
        )

    if debug or profile:
        _level_restriction_debug(
            "done "
            f"elapsed={time.monotonic() - start_time:.2f}s "
            f"splits={split_count} nboxes={balanced_tob.nboxes} "
            f"nleaves={len(leaf_keys)} work_items={work_items}"
        )

    return balanced_tob


def _enforce_same_level_colleagues(tob):
    while True:
        box_levels = tob.box_levels
        box_centers = tob.box_centers
        has_children = np.any(tob.box_child_ids != 0, axis=0)
        refine_flags = np.zeros(tob.nboxes, dtype=bool)

        for ibox in range(tob.nboxes):
            if not has_children[ibox]:
                continue

            level = int(box_levels[ibox])
            for jbox in range(tob.nboxes):
                if int(box_levels[jbox]) != level:
                    continue
                if not has_children[jbox]:
                    continue
                if ibox == jbox:
                    continue
                if not _are_adjacent(
                    tob.root_extent, box_levels, box_centers, ibox, jbox
                ):
                    continue

                for child in tob.box_child_ids[:, ibox]:
                    if child != 0 and np.all(tob.box_child_ids[:, child] == 0):
                        refine_flags[int(child)] = True
                for child in tob.box_child_ids[:, jbox]:
                    if child != 0 and np.all(tob.box_child_ids[:, child] == 0):
                        refine_flags[int(child)] = True

        if not np.any(refine_flags):
            return tob

        tob = refine_and_coarsen_tree_of_boxes(tob, refine_flags=refine_flags)


def _resize_bool_flags(flags, new_size):
    flags = np.asarray(flags, dtype=bool).ravel()
    if flags.size == new_size:
        return flags

    if flags.size > new_size:
        raise ValueError(
            "flag arrays longer than the current number of boxes must be remapped"
        )

    resized = np.zeros(new_size, dtype=bool)
    if flags.size:
        resized[: flags.size] = flags
    return resized


def _box_keys_from_geometry(tob):
    _, levels, _, _, grid_indices = _geometry_grid_indices(tob)

    keys = []
    for ibox in range(tob.nboxes):
        level = int(levels[ibox])
        idx = grid_indices[ibox]
        keys.append((level, tuple(int(v) for v in idx)))

    return keys


def _geometry_grid_indices(tob):
    centers = np.asarray(tob.box_centers)
    levels = np.asarray(tob.box_levels, dtype=np.int32)

    root_candidates = np.where(levels == 0)[0]
    if len(root_candidates) != 1:
        raise ValueError("expected exactly one root box at level 0")

    root_center = centers[:, int(root_candidates[0])]
    root_extent = float(tob.root_extent)
    root_min = root_center - 0.5 * root_extent

    nboxes = int(tob.nboxes)
    dim = int(tob.dimensions)

    grid_indices = np.empty((nboxes, dim), dtype=np.int64)
    for ibox in range(nboxes):
        level = int(levels[ibox])
        box_size = root_extent / (1 << level)
        grid_indices[ibox, :] = np.rint(
            (centers[:, ibox] - root_min) / box_size - 0.5
        ).astype(np.int64)

    return centers, levels, root_extent, root_min, grid_indices


def _box_paths_from_topology(tob, *, require_connected=True):
    parent_ids = np.asarray(tob.box_parent_ids, dtype=np.int64)
    child_ids = np.asarray(tob.box_child_ids, dtype=np.int64)
    levels = np.asarray(tob.box_levels, dtype=np.int64)
    nboxes = int(tob.nboxes)

    root_candidates = np.where(parent_ids < 0)[0]
    level_root_candidates = np.where(levels == 0)[0]
    if len(root_candidates) != 1 and len(level_root_candidates) == 1:
        root_candidates = level_root_candidates

    if len(root_candidates) != 1:
        raise ValueError("expected exactly one root box (parent id < 0 or level == 0)")

    root_id = int(root_candidates[0])
    box_paths = [None] * nboxes
    box_paths[root_id] = ()

    stack = [root_id]
    while stack:
        parent_id = stack.pop()
        parent_path = box_paths[parent_id]
        assert parent_path is not None

        for child_slot, child_id in enumerate(child_ids[:, parent_id]):
            child_id = int(child_id)
            if child_id == 0:
                continue
            if child_id < 0 or child_id >= nboxes:
                raise ValueError("tree-of-boxes contains invalid child id")

            child_path = parent_path + (child_slot,)
            prev_path = box_paths[child_id]

            if prev_path is None:
                box_paths[child_id] = child_path
                stack.append(child_id)
            elif prev_path != child_path:
                raise ValueError("child box has multiple parent paths")

    if require_connected and any(path is None for path in box_paths):
        raise ValueError("tree-of-boxes contains unreachable boxes")

    return box_paths


def _remap_bool_flags_by_box_key(
    flags,
    old_tob,
    new_tob,
    *,
    prefer_descendant_leaf=False,
):
    flags = _resize_bool_flags(flags, old_tob.nboxes)
    remapped = np.zeros(new_tob.nboxes, dtype=bool)

    if not np.any(flags):
        return remapped

    old_paths = _box_paths_from_topology(old_tob)
    new_paths = _box_paths_from_topology(new_tob)
    path_to_new = {path: idx for idx, path in enumerate(new_paths)}

    new_is_leaf = None
    new_child_ids = None
    if prefer_descendant_leaf:
        new_is_leaf = np.all(new_tob.box_child_ids == 0, axis=0)
        new_child_ids = np.asarray(new_tob.box_child_ids, dtype=np.int64)

    for old_id in np.flatnonzero(flags):
        new_id = path_to_new.get(old_paths[int(old_id)])
        if (
            new_id is not None
            and prefer_descendant_leaf
            and new_is_leaf is not None
            and not bool(new_is_leaf[int(new_id)])
        ):
            cursor = int(new_id)
            while not bool(new_is_leaf[cursor]):
                children = new_child_ids[:, cursor]
                nonzero_children = children[children != 0]
                if nonzero_children.size == 0:
                    break
                cursor = int(nonzero_children[0])

            if bool(new_is_leaf[cursor]):
                new_id = cursor
            else:
                new_id = None

        if new_id is not None:
            remapped[new_id] = True

    return remapped


def _coarsen_tree_of_boxes_compat(
    tob,
    coarsen_flags,
    *,
    error_on_ignored_flags=True,
):
    coarsen_flags = np.asarray(coarsen_flags, dtype=bool)
    if coarsen_flags.size != tob.nboxes:
        raise ValueError("coarsen_flags must match the number of boxes")

    box_is_leaf = np.all(tob.box_child_ids == 0, axis=0)
    if np.any(coarsen_flags[~box_is_leaf]):
        raise ValueError("attempting to coarsen non-leaf")

    (coarsen_sources,) = np.where(coarsen_flags)
    if coarsen_sources.size == 0:
        return tob

    source_parents = np.asarray(tob.box_parent_ids, dtype=np.int64)[coarsen_sources]
    valid_parent_flags = source_parents >= 0
    executable = np.zeros(coarsen_sources.size, dtype=bool)

    if np.any(valid_parent_flags):
        valid_parents = source_parents[valid_parent_flags]
        peer_ids = np.asarray(tob.box_child_ids, dtype=np.int64)[:, valid_parents]

        nonzero_peer_mask = peer_ids != 0
        leaf_flags = np.ones(peer_ids.shape, dtype=bool)
        leaf_flags[nonzero_peer_mask] = box_is_leaf[peer_ids[nonzero_peer_mask]]

        has_complete_peer_set = np.all(nonzero_peer_mask, axis=0)
        all_peers_leaf = np.all(leaf_flags, axis=0)
        executable[valid_parent_flags] = has_complete_peer_set & all_peers_leaf

    ignored_count = int(np.count_nonzero(~executable))
    if ignored_count:
        msg = (
            f"{ignored_count} out of {int(np.sum(coarsen_flags))} coarsening "
            "flags ignored to prevent removing non-leaf boxes"
        )
        if error_on_ignored_flags:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=3)

    executable_parents = np.unique(source_parents[executable])
    if executable_parents.size == 0:
        return tob

    box_parent_ids = np.asarray(tob.box_parent_ids).copy()
    box_child_ids = np.asarray(tob.box_child_ids).copy()

    for parent_id in executable_parents:
        peer_ids = box_child_ids[:, parent_id]
        peer_ids = peer_ids[peer_ids != 0]
        box_parent_ids[peer_ids] = -1
        box_child_ids[:, parent_id] = 0

    from boxtree.tree import TreeOfBoxes

    coarsened = TreeOfBoxes(
        box_centers=tob.box_centers,
        root_extent=tob.root_extent,
        box_parent_ids=box_parent_ids,
        box_child_ids=box_child_ids,
        box_levels=tob.box_levels,
        box_flags=np.asarray(
            _compute_box_flags(box_child_ids), dtype=box_flags_enum.dtype
        ),
        level_start_box_nrs=None,
        box_id_dtype=tob.box_id_dtype,
        box_level_dtype=tob.box_level_dtype,
        coord_dtype=tob.coord_dtype,
        sources_have_extent=tob.sources_have_extent,
        targets_have_extent=tob.targets_have_extent,
        extent_norm=tob.extent_norm,
        stick_out_factor=tob.stick_out_factor,
        _is_pruned=tob._is_pruned,
    )

    return _prune_unreachable_boxes(coarsened)


class BoxTree:
    """Compatibility wrapper for the old boxtree interactive build API.

    This vendors the small portion of the old ``xywei/boxtree`` API that
    ``volumential`` relied on for adaptive mesh generation, while using current
    upstream ``boxtree.tree_of_boxes`` data structures internally.
    """

    def __init__(self):
        self.queue: cl.CommandQueue | None = None
        self.root_vertex = None
        self.root_extent = None
        self.box_id_dtype = np.int32
        self.box_level_dtype = np.int32
        self.coord_dtype = np.float64
        self._tree = None

    def generate_uniform_boxtree(
        self,
        queue,
        root_vertex=np.zeros(2),
        root_extent=1,
        nlevels=1,
        box_id_dtype=np.int32,
        box_level_dtype=np.int32,
        coord_dtype=np.float64,
    ):
        self.queue = queue
        self.root_vertex = np.asarray(root_vertex, dtype=coord_dtype)
        self.root_extent = coord_dtype(root_extent)
        self.box_id_dtype = box_id_dtype
        self.box_level_dtype = box_level_dtype
        self.coord_dtype = coord_dtype

        bbox = (self.root_vertex, self.root_vertex + self.root_extent)
        tree = make_tree_of_boxes_root(bbox)
        for _ in range(max(0, nlevels - 1)):
            tree = uniformly_refine_tree_of_boxes(tree)

        self._tree = tree
        self._sync_device_views()

    def refine_and_coarsen(
        self, refine_flags, coarsen_flags, error_on_ignored_flags=False
    ):
        if isinstance(refine_flags, cl.array.Array):
            refine_flags = refine_flags.get()
        if isinstance(coarsen_flags, cl.array.Array):
            coarsen_flags = coarsen_flags.get()

        refine_flags = np.asarray(refine_flags, dtype=bool)
        coarsen_flags = np.asarray(coarsen_flags, dtype=bool)

        refine_flags = _resize_bool_flags(refine_flags, self._tree.nboxes)
        coarsen_flags = _resize_bool_flags(coarsen_flags, self._tree.nboxes)

        if np.any(refine_flags & coarsen_flags):
            raise ValueError(
                "some boxes are simultaneously marked to refine and coarsen"
            )

        if np.any(refine_flags):
            tree_before_refine = self._tree
            self._tree = refine_and_coarsen_tree_of_boxes(
                self._tree,
                refine_flags=refine_flags,
                coarsen_flags=None,
                error_on_ignored_flags=error_on_ignored_flags,
            )
            coarsen_flags = _remap_bool_flags_by_box_key(
                coarsen_flags,
                tree_before_refine,
                self._tree,
                prefer_descendant_leaf=True,
            )

            leaf_mask = np.all(self._tree.box_child_ids == 0, axis=0)
            remapped_nonleaf = coarsen_flags & ~leaf_mask
            if np.any(remapped_nonleaf):
                ignored_count = int(np.count_nonzero(remapped_nonleaf))
                msg = (
                    f"{ignored_count} coarsening flags ignored after refinement "
                    "because they now target non-leaf boxes"
                )
                coarsen_flags[remapped_nonleaf] = False
                if error_on_ignored_flags:
                    raise RuntimeError(msg)
                warnings.warn(msg, stacklevel=3)
        else:
            coarsen_flags = _resize_bool_flags(coarsen_flags, self._tree.nboxes)

        if np.any(coarsen_flags):
            self._tree = _coarsen_tree_of_boxes_compat(
                self._tree,
                coarsen_flags,
                error_on_ignored_flags=error_on_ignored_flags,
            )

        self._tree = _rebuild_tob_from_geometry(self._tree)
        self._sync_device_views()

    def _sync_device_views(self):
        assert self.queue is not None
        assert self._tree is not None

        self._tree = _rebuild_tob_from_geometry(self._tree)
        self._tree = _enforce_level_restriction(self._tree)
        self._tree = _rebuild_tob_from_geometry(self._tree)
        self._tree = _prune_unreachable_boxes(self._tree)

        box_levels = np.asarray(self._tree.box_levels, dtype=self.box_level_dtype)
        box_centers = np.asarray(self._tree.box_centers, dtype=self.coord_dtype)
        active_boxes = np.asarray(self._tree.leaf_boxes, dtype=self.box_id_dtype)

        self.box_levels = cl.array.to_device(self.queue, box_levels)
        self.box_centers = cl.array.to_device(self.queue, box_centers)
        self.active_boxes = cl.array.to_device(self.queue, active_boxes)

        nlevels = int(box_levels.max()) + 1 if box_levels.size else 0
        level_boxes = []
        for ilevel in range(nlevels):
            ids = np.where(box_levels == ilevel)[0].astype(self.box_id_dtype)
            level_boxes.append(cl.array.to_device(self.queue, ids))
        self.level_boxes = obj_array_1d(level_boxes)

    @property
    def dimensions(self):
        return self._tree.dimensions

    @property
    def nboxes(self):
        return self._tree.nboxes

    @property
    def nlevels(self):
        return int(np.max(self._tree.box_levels)) + 1

    @property
    def n_active_boxes(self):
        return len(self._tree.leaf_boxes)

    def get_box_extent(self, ibox):
        if isinstance(ibox, cl.array.Array):
            ibox = int(ibox.get())

        level = int(self._tree.box_levels[ibox])
        box_size = self.root_extent / (2**level)
        center = self._tree.box_centers[:, ibox]
        extent_low = center - 0.5 * box_size
        extent_high = center + 0.5 * box_size
        return extent_low.astype(self.coord_dtype), extent_high.astype(self.coord_dtype)


class QuadratureOnBoxTree:
    def __init__(self, boxtree, quadrature_formula=None):
        self.boxtree = boxtree

        if quadrature_formula is None:
            from modepy import LegendreGaussQuadrature

            quadrature_formula = LegendreGaussQuadrature(0, force_dim_axis=True)

        self.quadrature_formula = quadrature_formula

    def _reference_nodes(self):
        return np.asarray(self.quadrature_formula.nodes)

    def _reference_weights(self):
        return np.asarray(self.quadrature_formula.weights)

    def _leaf_boxes(self):
        return np.asarray(self.boxtree._tree.leaf_boxes)

    def _leaf_levels(self):
        return np.asarray(self.boxtree._tree.box_levels[self._leaf_boxes()])

    def _leaf_centers(self):
        return np.asarray(self.boxtree._tree.box_centers[:, self._leaf_boxes()])

    def get_q_points(self, queue):
        q_nodes = self._reference_nodes()
        dim = self.boxtree.dimensions
        centers = self._leaf_centers().T
        side_lengths = self.boxtree.root_extent / (2 ** self._leaf_levels())

        if dim == 1:
            grids = [q_nodes[:, None]]
        else:
            grids = np.meshgrid(*([q_nodes] * dim), indexing="ij")

        ref_points = np.vstack([g.ravel() for g in grids]).T
        q_points = (
            centers[:, None, :]
            + 0.5 * side_lengths[:, None, None] * ref_points[None, :, :]
        )
        q_points = q_points.reshape(-1, dim).T
        return obj_array_1d(
            [cl.array.to_device(queue, np.ascontiguousarray(comp)) for comp in q_points]
        )

    def get_q_weights(self, queue):
        dim = self.boxtree.dimensions
        q_weights_1d = self._reference_weights()
        grids = np.meshgrid(*([q_weights_1d] * dim), indexing="ij")
        ref_weights = np.prod(np.array(grids), axis=0).ravel()
        side_lengths = self.boxtree.root_extent / (2 ** self._leaf_levels())
        weights = (
            ((0.5 * side_lengths) ** dim)[:, None] * ref_weights[None, :]
        ).ravel()
        return cl.array.to_device(queue, weights.astype(self.boxtree.coord_dtype))

    def get_cell_centers(self, queue):
        centers = self._leaf_centers()
        return obj_array_1d(
            [cl.array.to_device(queue, np.ascontiguousarray(comp)) for comp in centers]
        )

    def get_cell_measures(self, queue):
        dim = self.boxtree.dimensions
        side_lengths = self.boxtree.root_extent / (2 ** self._leaf_levels())
        measures = (side_lengths**dim).astype(self.boxtree.coord_dtype)
        return cl.array.to_device(queue, measures)


def _leaf_dfs_order(box_child_ids, ibox=0):
    children = [int(ch) for ch in box_child_ids[:, ibox] if int(ch) != 0]
    if not children:
        return [ibox]

    result = []
    for child in children:
        result.extend(_leaf_dfs_order(box_child_ids, child))
    return result


def _level_order_boxes(box_child_ids, *, root_id=0):
    nboxes = int(box_child_ids.shape[1])

    result = []
    current = [int(root_id)]
    seen = set()

    while current:
        nxt = []
        queued = set()

        for ibox in current:
            ibox = int(ibox)
            if ibox < 0 or ibox >= nboxes:
                raise ValueError("tree-of-boxes contains invalid box id")
            if ibox in seen:
                raise ValueError(
                    "tree-of-boxes contains cyclic or repeated parent links"
                )

            seen.add(ibox)
            result.append(ibox)

            for ch in box_child_ids[:, ibox]:
                child_id = int(ch)
                if child_id == 0:
                    continue
                if child_id < 0 or child_id >= nboxes:
                    raise ValueError("tree-of-boxes contains invalid child id")
                if child_id in seen or child_id in queued:
                    raise ValueError(
                        "tree-of-boxes contains cyclic or repeated child links"
                    )

                queued.add(child_id)
                nxt.append(child_id)

        current = nxt

    return result


def _prune_unreachable_boxes(tob):
    parent_ids = np.asarray(tob.box_parent_ids, dtype=np.int64)
    levels = np.asarray(tob.box_levels, dtype=np.int64)

    root_candidates = np.where(parent_ids < 0)[0]
    level_root_candidates = np.where(levels == 0)[0]
    if len(root_candidates) != 1 and len(level_root_candidates) == 1:
        root_candidates = level_root_candidates

    if len(root_candidates) != 1:
        raise ValueError("expected exactly one root box (parent id < 0 or level == 0)")

    reachable = _level_order_boxes(tob.box_child_ids, root_id=int(root_candidates[0]))
    if len(reachable) == tob.nboxes:
        return tob

    reachable = np.array(reachable, dtype=np.int32)
    old_to_new = np.full(tob.nboxes, -1, dtype=np.int32)
    old_to_new[reachable] = np.arange(len(reachable), dtype=np.int32)

    box_centers = tob.box_centers[:, reachable]
    box_levels = tob.box_levels[reachable]
    old_parents = tob.box_parent_ids[reachable]
    box_parent_ids = np.where(old_parents < 0, -1, old_to_new[old_parents])
    old_child_ids = tob.box_child_ids[:, reachable]
    box_child_ids = np.where(old_child_ids == 0, 0, old_to_new[old_child_ids])

    from boxtree.tree import TreeOfBoxes

    return TreeOfBoxes(
        box_centers=box_centers,
        root_extent=tob.root_extent,
        box_parent_ids=box_parent_ids,
        box_child_ids=box_child_ids,
        box_levels=box_levels,
        box_flags=np.asarray(
            _compute_box_flags(box_child_ids), dtype=box_flags_enum.dtype
        ),
        level_start_box_nrs=None,
        box_id_dtype=tob.box_id_dtype,
        box_level_dtype=tob.box_level_dtype,
        coord_dtype=tob.coord_dtype,
        sources_have_extent=tob.sources_have_extent,
        targets_have_extent=tob.targets_have_extent,
        extent_norm=tob.extent_norm,
        stick_out_factor=tob.stick_out_factor,
        _is_pruned=tob._is_pruned,
    )


def _compute_box_flags(box_child_ids):
    box_flags = np.full(
        box_child_ids.shape[1],
        box_flags_enum.IS_SOURCE_BOX | box_flags_enum.IS_TARGET_BOX,
        dtype=box_flags_enum.dtype,
    )
    has_children = np.any(box_child_ids != 0, axis=0)
    box_flags[has_children] |= (
        box_flags_enum.HAS_SOURCE_CHILD_BOXES | box_flags_enum.HAS_TARGET_CHILD_BOXES
    )
    box_flags[~has_children] |= box_flags_enum.IS_LEAF_BOX
    return box_flags


def _rebuild_tob_from_geometry(tob):
    from boxtree.tree import TreeOfBoxes

    box_paths = _box_paths_from_topology(tob, require_connected=False)

    centers, levels, root_extent, _, grid_indices = _geometry_grid_indices(tob)

    nboxes = int(centers.shape[1])
    dim = int(tob.dimensions)
    nchildren = 2**dim

    def has_missing_parent(keys):
        key_set = set(keys)
        for level, idx in keys:
            if level == 0:
                continue
            parent_key = (level - 1, tuple(v // 2 for v in idx))
            if parent_key not in key_set:
                return True
        return False

    child_slot_bits = np.empty((nchildren, dim), dtype=np.int64)
    for child_slot in range(nchildren):
        child_slot_bits[child_slot, :] = [
            (child_slot >> (dim - 1 - iaxis)) & 1 for iaxis in range(dim)
        ]

    geo_keys = [
        (int(levels[i]), tuple(int(v) for v in grid_indices[i])) for i in range(nboxes)
    ]

    use_topology_keys = len(set(geo_keys)) != nboxes or has_missing_parent(geo_keys)

    if use_topology_keys:
        keys = []
        for ibox, path in enumerate(box_paths):
            if path is None:
                keys.append(geo_keys[ibox])
                continue

            grid_idx = np.zeros(dim, dtype=np.int64)
            for child_slot in path:
                grid_idx = 2 * grid_idx + child_slot_bits[int(child_slot)]
            keys.append((len(path), tuple(int(v) for v in grid_idx)))
    else:
        keys = geo_keys

    has_duplicate_keys = len(set(keys)) != nboxes
    has_parent_gaps = has_missing_parent(keys)
    if has_duplicate_keys or has_parent_gaps:
        pruned_tob = _prune_unreachable_boxes(tob)
        if pruned_tob.nboxes < tob.nboxes:
            return _rebuild_tob_from_geometry(pruned_tob)

        if has_duplicate_keys:
            raise ValueError("duplicate level/grid-index keys in tree-of-boxes")
        raise ValueError("missing parent while rebuilding tree-of-boxes")

    root_old_ids = [ibox for ibox, (level, _) in enumerate(keys) if level == 0]
    if len(root_old_ids) != 1:
        raise ValueError("expected exactly one root box at level 0")
    root_old_id = int(root_old_ids[0])

    root_center = np.asarray(centers[:, root_old_id], dtype=np.float64)
    if not np.all(np.isfinite(root_center)):
        raise ValueError("non-finite root center while rebuilding tree-of-boxes")

    root_min = root_center - 0.5 * root_extent

    new_order = sorted(range(nboxes), key=lambda i: (keys[i][0],) + keys[i][1])
    new_levels = np.asarray([keys[i][0] for i in new_order], dtype=tob.box_level_dtype)
    new_grid_indices = np.asarray([keys[i][1] for i in new_order], dtype=np.int64)

    new_box_sizes = root_extent * np.exp2(-new_levels.astype(np.float64))
    new_centers = (
        root_min.reshape(dim, 1)
        + (new_grid_indices.T.astype(np.float64) + 0.5) * new_box_sizes
    )

    key_to_new = {}
    for new_id, old_id in enumerate(new_order):
        key = keys[old_id]
        if key in key_to_new:
            raise ValueError("duplicate level/grid-index keys in tree-of-boxes")
        key_to_new[key] = new_id

    new_parent_ids = np.full(nboxes, -1, dtype=tob.box_id_dtype)
    new_child_ids = np.zeros((nchildren, nboxes), dtype=tob.box_id_dtype)

    for new_id in range(nboxes):
        level = int(new_levels[new_id])
        idx = new_grid_indices[new_id]

        if level > 0:
            parent_key = (level - 1, tuple((idx // 2).tolist()))
            parent_id = key_to_new.get(parent_key)
            if parent_id is None:
                raise ValueError("missing parent while rebuilding tree-of-boxes")
            new_parent_ids[new_id] = parent_id

        for child_slot in range(nchildren):
            bits = child_slot_bits[child_slot]
            child_key = (level + 1, tuple((2 * idx + bits).tolist()))
            child_id = key_to_new.get(child_key)
            if child_id is not None:
                new_child_ids[child_slot, new_id] = child_id

    max_level = int(np.max(new_levels))
    level_counts = np.bincount(new_levels, minlength=max_level + 1)
    level_start_box_nrs = np.zeros(max_level + 2, dtype=tob.box_id_dtype)
    level_start_box_nrs[1:] = np.cumsum(level_counts).astype(tob.box_id_dtype)

    return TreeOfBoxes(
        box_centers=np.asarray(new_centers, dtype=tob.coord_dtype),
        root_extent=tob.root_extent,
        box_parent_ids=np.asarray(new_parent_ids, dtype=tob.box_id_dtype),
        box_child_ids=np.asarray(new_child_ids, dtype=tob.box_id_dtype),
        box_levels=np.asarray(new_levels, dtype=tob.box_level_dtype),
        box_flags=np.asarray(
            _compute_box_flags(new_child_ids), dtype=box_flags_enum.dtype
        ),
        level_start_box_nrs=np.asarray(level_start_box_nrs, dtype=tob.box_id_dtype),
        box_id_dtype=tob.box_id_dtype,
        box_level_dtype=tob.box_level_dtype,
        coord_dtype=tob.coord_dtype,
        sources_have_extent=tob.sources_have_extent,
        targets_have_extent=tob.targets_have_extent,
        extent_norm=tob.extent_norm,
        stick_out_factor=tob.stick_out_factor,
        _is_pruned=tob._is_pruned,
    )


def build_particle_tree_from_box_tree(actx, box_tree, q_points_host):
    tob = box_tree._tree
    dim = tob.dimensions
    n_q_points = q_points_host.shape[0] // len(tob.leaf_boxes)
    box_level_dtype = np.dtype(np.uint8)

    box_order_old = _level_order_boxes(tob.box_child_ids)
    old_to_new = np.empty(tob.nboxes, dtype=tob.box_id_dtype)
    for new_id, old_id in enumerate(box_order_old):
        old_to_new[old_id] = new_id

    leaf_order_old = _leaf_dfs_order(tob.box_child_ids)
    leaf_order_old = [
        ibox for ibox in leaf_order_old if np.all(tob.box_child_ids[:, ibox] == 0)
    ]
    old_leaf_boxes = list(map(int, tob.leaf_boxes))
    old_leaf_pos = {ibox: i for i, ibox in enumerate(old_leaf_boxes)}
    particle_perm = np.concatenate(
        [
            np.arange(
                old_leaf_pos[ibox] * n_q_points,
                (old_leaf_pos[ibox] + 1) * n_q_points,
                dtype=tob.box_id_dtype,
            )
            for ibox in leaf_order_old
        ]
    )
    inverse_particle_perm = np.empty_like(particle_perm)
    inverse_particle_perm[particle_perm] = np.arange(
        len(particle_perm), dtype=tob.box_id_dtype
    )

    leaf_pos = {ibox: i for i, ibox in enumerate(leaf_order_old)}

    box_target_starts = np.zeros(tob.nboxes, dtype=tob.box_id_dtype)
    box_target_counts_nonchild = np.zeros(tob.nboxes, dtype=tob.box_id_dtype)
    box_target_counts_cumul = np.zeros(tob.nboxes, dtype=tob.box_id_dtype)

    def assign_particle_ranges(ibox):
        children = [int(ch) for ch in tob.box_child_ids[:, ibox] if int(ch) != 0]
        if not children:
            start = leaf_pos[ibox] * n_q_points
            count = n_q_points
            box_target_starts[ibox] = start
            box_target_counts_nonchild[ibox] = count
            box_target_counts_cumul[ibox] = count
            return start, count

        child_ranges = [assign_particle_ranges(ch) for ch in children]
        starts = [start for start, _ in child_ranges]
        counts = [count for _, count in child_ranges]
        start = min(starts)
        stop = max(st + ct for st, ct in child_ranges)
        assert sum(counts) == stop - start
        box_target_starts[ibox] = start
        box_target_counts_nonchild[ibox] = 0
        box_target_counts_cumul[ibox] = stop - start
        return start, stop - start

    assign_particle_ranges(0)

    box_source_starts = box_target_starts.copy()
    box_source_counts_nonchild = box_target_counts_nonchild.copy()
    box_source_counts_cumul = box_target_counts_cumul.copy()

    reordered_points = q_points_host[particle_perm]
    particle_id_dtype = np.int32

    sources = obj_array_1d(
        [
            actx.from_numpy(np.ascontiguousarray(reordered_points[:, iaxis]))
            for iaxis in range(dim)
        ]
    )

    level_counts = np.bincount(
        np.asarray(tob.box_levels, dtype=np.int32),
        minlength=int(np.max(tob.box_levels)) + 1,
    )
    level_start_box_nrs = np.zeros(len(level_counts) + 1, dtype=tob.box_id_dtype)
    level_start_box_nrs[1:] = np.cumsum(level_counts)

    aligned_nboxes = ((tob.nboxes + 31) // 32) * 32

    box_centers = np.zeros((dim, aligned_nboxes), dtype=box_tree.coord_dtype)
    box_centers[:, : tob.nboxes] = tob.box_centers[:, box_order_old]

    box_child_ids = np.zeros((2**dim, aligned_nboxes), dtype=tob.box_id_dtype)
    old_child_ids = tob.box_child_ids[:, box_order_old]
    mapped_child_ids = np.where(old_child_ids != 0, old_to_new[old_child_ids], 0)
    box_child_ids[:, : tob.nboxes] = mapped_child_ids

    box_parent_ids = np.zeros(tob.nboxes, dtype=tob.box_id_dtype)
    for parent in range(tob.nboxes):
        for child in box_child_ids[:, parent]:
            if child != 0:
                box_parent_ids[int(child)] = parent
    if tob.nboxes:
        box_parent_ids[0] = 0

    box_flags = np.zeros(tob.nboxes, dtype=box_flags_enum.dtype)
    has_children = np.any(box_child_ids[:, : tob.nboxes] != 0, axis=0)
    box_flags[~has_children] |= (
        box_flags_enum.IS_SOURCE_BOX | box_flags_enum.IS_TARGET_BOX
    )
    box_flags[has_children] |= (
        box_flags_enum.HAS_SOURCE_CHILD_BOXES | box_flags_enum.HAS_TARGET_CHILD_BOXES
    )

    zeros_bbox = obj_array_1d(
        [
            actx.from_numpy(np.zeros(aligned_nboxes, dtype=box_tree.coord_dtype))
            for _ in range(dim)
        ]
    )

    tree = Tree(
        root_extent=box_tree.root_extent,
        box_centers=actx.from_numpy(box_centers),
        box_parent_ids=actx.from_numpy(box_parent_ids),
        box_child_ids=actx.from_numpy(box_child_ids),
        box_levels=actx.from_numpy(
            np.asarray(tob.box_levels, dtype=box_level_dtype)[box_order_old]
        ),
        box_flags=actx.from_numpy(box_flags),
        level_start_box_nrs=actx.from_numpy(level_start_box_nrs),
        box_id_dtype=np.dtype(tob.box_id_dtype),
        box_level_dtype=box_level_dtype,
        coord_dtype=np.dtype(box_tree.coord_dtype),
        sources_have_extent=False,
        targets_have_extent=False,
        extent_norm=None,
        stick_out_factor=0,
        sources_are_targets=True,
        particle_id_dtype=np.dtype(tob.box_id_dtype),
        sources=sources,
        source_radii=None,
        targets=sources,
        target_radii=None,
        bounding_box=(
            np.asarray(box_tree.root_vertex, dtype=box_tree.coord_dtype),
            np.asarray(
                box_tree.root_vertex + box_tree.root_extent,
                dtype=box_tree.coord_dtype,
            ),
        ),
        user_source_ids=actx.from_numpy(particle_perm.astype(particle_id_dtype)),
        sorted_target_ids=actx.from_numpy(
            inverse_particle_perm.astype(particle_id_dtype)
        ),
        box_source_starts=actx.from_numpy(box_source_starts[box_order_old]),
        box_source_counts_nonchild=actx.from_numpy(
            box_source_counts_nonchild[box_order_old]
        ),
        box_source_counts_cumul=actx.from_numpy(box_source_counts_cumul[box_order_old]),
        box_target_starts=actx.from_numpy(box_target_starts[box_order_old]),
        box_target_counts_nonchild=actx.from_numpy(
            box_target_counts_nonchild[box_order_old]
        ),
        box_target_counts_cumul=actx.from_numpy(box_target_counts_cumul[box_order_old]),
        box_source_bounding_box_min=zeros_bbox,
        box_source_bounding_box_max=zeros_bbox,
        box_target_bounding_box_min=zeros_bbox,
        box_target_bounding_box_max=zeros_bbox,
        _is_pruned=tob._is_pruned,
    )

    return actx.freeze(tree)
