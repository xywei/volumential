from __future__ import annotations

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


_COARSENING_DISABLED_WARNING = (
    "Current upstream boxtree tree-of-boxes coarsening is incompatible with "
    "volumential's historical adaptive mesh workflow; retrying with refinement "
    "only."
)


def _leaf_boxes_numpy(tob):
    return np.where(np.all(tob.box_child_ids == 0, axis=0))[0]


def _box_size(root_extent, level):
    return root_extent * (0.5 ** int(level))


def _are_adjacent(root_extent, levels, centers, ibox, jbox, tol=1.0e-15):
    si = _box_size(root_extent, levels[ibox])
    sj = _box_size(root_extent, levels[jbox])
    dist = np.max(np.abs(centers[:, ibox] - centers[:, jbox]))
    return dist <= 0.5 * (si + sj) + tol


def _enforce_level_restriction(tob):
    """Enforce only 2:1 leaf-level balancing.

    Two adjacent leaves are allowed to differ by at most one level.
    No extra same-level colleague balancing is applied.
    """
    tob = _rebuild_tob_from_geometry(tob)

    while True:
        leaves = _leaf_boxes_numpy(tob)
        refine_flags = np.zeros(tob.nboxes, dtype=bool)

        for i, ibox in enumerate(leaves):
            for jbox in leaves[i + 1 :]:
                li = int(tob.box_levels[ibox])
                lj = int(tob.box_levels[jbox])
                if abs(li - lj) <= 1:
                    continue
                if not _are_adjacent(
                    tob.root_extent, tob.box_levels, tob.box_centers, ibox, jbox
                ):
                    continue
                if li < lj:
                    refine_flags[ibox] = True
                else:
                    refine_flags[jbox] = True

        if not np.any(refine_flags):
            return tob

        tob = refine_and_coarsen_tree_of_boxes(tob, refine_flags=refine_flags)
        tob = _rebuild_tob_from_geometry(tob)


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

        try:
            self._tree = refine_and_coarsen_tree_of_boxes(
                self._tree,
                refine_flags=refine_flags,
                coarsen_flags=coarsen_flags,
                error_on_ignored_flags=error_on_ignored_flags,
            )
        except OverflowError:
            import warnings

            warnings.warn(_COARSENING_DISABLED_WARNING, stacklevel=2)
            self._tree = refine_and_coarsen_tree_of_boxes(
                self._tree,
                refine_flags=refine_flags,
                coarsen_flags=np.zeros_like(coarsen_flags),
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

            quadrature_formula = LegendreGaussQuadrature(0)

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


def _level_order_boxes(box_child_ids):
    result = []
    current = [0]
    while current:
        result.extend(current)
        nxt = []
        for ibox in current:
            nxt.extend(int(ch) for ch in box_child_ids[:, ibox] if int(ch) != 0)
        current = nxt
    return result


def _prune_unreachable_boxes(tob):
    reachable = _level_order_boxes(tob.box_child_ids)
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

    centers = np.asarray(tob.box_centers)
    levels = np.asarray(tob.box_levels, dtype=np.int32)

    nboxes = int(tob.nboxes)
    dim = int(tob.dimensions)
    nchildren = 2**dim

    root_candidates = np.where(levels == 0)[0]
    if len(root_candidates) != 1:
        raise ValueError("expected exactly one root box at level 0")

    root_center = centers[:, int(root_candidates[0])]
    root_extent = float(tob.root_extent)
    root_min = root_center - 0.5 * root_extent

    grid_indices = np.empty((nboxes, dim), dtype=np.int64)
    for ibox in range(nboxes):
        lev = int(levels[ibox])
        box_size = root_extent / (1 << lev)
        grid_indices[ibox, :] = np.rint(
            (centers[:, ibox] - root_min) / box_size - 0.5
        ).astype(np.int64)

    old_keys = [
        (int(levels[i]), tuple(int(v) for v in grid_indices[i])) for i in range(nboxes)
    ]
    if len(set(old_keys)) != nboxes:
        return tob

    new_order = sorted(
        range(nboxes),
        key=lambda i: (int(levels[i]),) + tuple(int(v) for v in grid_indices[i]),
    )

    new_levels = levels[new_order]
    new_centers = centers[:, new_order]
    new_grid_indices = grid_indices[new_order]

    key_to_new = {old_keys[old_id]: new_id for new_id, old_id in enumerate(new_order)}

    new_parent_ids = np.full(nboxes, -1, dtype=tob.box_id_dtype)
    new_child_ids = np.zeros((nchildren, nboxes), dtype=tob.box_id_dtype)

    for new_id in range(nboxes):
        level = int(new_levels[new_id])
        idx = new_grid_indices[new_id]

        if level > 0:
            parent_key = (level - 1, tuple((idx // 2).tolist()))
            parent_id = key_to_new.get(parent_key)
            if parent_id is None:
                return tob
            new_parent_ids[new_id] = parent_id

        for morton_nr in range(nchildren):
            bits = np.array(
                [(morton_nr >> (dim - 1 - iaxis)) & 1 for iaxis in range(dim)],
                dtype=np.int64,
            )
            child_key = (level + 1, tuple((2 * idx + bits).tolist()))
            child_id = key_to_new.get(child_key)
            if child_id is not None:
                new_child_ids[morton_nr, new_id] = child_id

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
