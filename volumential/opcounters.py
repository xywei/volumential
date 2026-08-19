"""Explicit operation counters for the Paper 1 benchmark drivers (E3).

The categories mirror the operation-count cost model that replaces
wall-clock seconds as the primary cost currency (Section 6 of the paper):

- :data:`SINGULAR_NODES` — singular-quadrature node evaluations (Duffy
  triangle/cone nodes at which a radial profile or kernel is evaluated);
- :data:`SMOOTH_NODES` — smooth-rule (tensor Gauss-Legendre) node
  evaluations of the windowed remainder;
- :data:`PROFILE_NODES` — nodes passed through a windowed channel profile
  ``psi_m`` (each such node costs one special-function-class evaluation);
- :data:`KERNEL_EVALS` — kernel evaluations, keyed by the kernel's special
  function;
- :data:`SPECIAL_EVALS` — special-function evaluations keyed by function
  (``expn``, ``erfc``, ``exp``, ``k0``, ``hankel1``, ...);
- :data:`RECOMBINATION_FLOPS` — recombination fused multiply-adds and
  coefficient-recurrence steps of the assembly;
- :data:`TABLE_ENTRIES` — table entries built (one singular quadrature per
  stored entry).

Counting is strictly explicit: code that owns a loop calls
:func:`add` (or ``OpCounters.add``) with the executed element count; nothing
is monkeypatched or inferred from timers.  The module-level :func:`add` fans
out to every :class:`OpCounters` activated through the :func:`counting`
context manager and is a no-op (one truthiness check) when none is active,
so instrumented library code costs nothing outside instrumented runs.

The driver-level helpers at the bottom compute *analytic* counts from the
executed configuration — the surviving-node counts of the requested
quadrature rules, the symmetry-reduced entry set, the Duffy degeneracy
predicates, and the FMM traversal — so drivers can emit
measured-versus-analytic comparisons alongside their timing columns.
"""

from __future__ import annotations

import operator
from contextlib import contextmanager

import numpy as np

__all__ = [
    "KERNEL_EVALS",
    "PROFILE_NODES",
    "RECOMBINATION_FLOPS",
    "SINGULAR_NODES",
    "SMOOTH_NODES",
    "SPECIAL_EVALS",
    "TABLE_ENTRIES",
    "OpCounters",
    "add",
    "batched_duffy_nodes_per_entry",
    "counting",
    "duffy_block_geometry",
    "grouped_duffy_singular_nodes",
    "nearfield_point_pairs",
    "reduced_entry_count",
    "regular_node_count",
    "scalar_duffy_singular_nodes",
    "surviving_radial_node_count",
]

SINGULAR_NODES = "singular_quadrature_nodes"
SMOOTH_NODES = "smooth_rule_nodes"
PROFILE_NODES = "channel_profile_nodes"
KERNEL_EVALS = "kernel_evals"
SPECIAL_EVALS = "special_function_evals"
RECOMBINATION_FLOPS = "recombination_flops"
TABLE_ENTRIES = "table_entries_built"


def _require_count(count):
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("count must be an integer")
    try:
        result = operator.index(count)
    except TypeError as exc:
        raise ValueError("count must be an integer") from exc
    if result < 0:
        raise ValueError("count must be >= 0")
    return int(result)


class OpCounters:
    """A bag of named operation counters, ``category -> label -> count``."""

    def __init__(self):
        self._counts: dict[str, dict[str, int]] = {}

    def add(self, category: str, label: str, count) -> None:
        count = _require_count(count)
        by_label = self._counts.setdefault(str(category), {})
        by_label[str(label)] = by_label.get(str(label), 0) + count

    def labels(self, category: str) -> dict[str, int]:
        """Per-label counts of one category (a copy)."""
        return dict(self._counts.get(str(category), {}))

    def total(self, category: str) -> int:
        return sum(self._counts.get(str(category), {}).values())

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            category: dict(by_label)
            for category, by_label in self._counts.items()
        }

    def by_function(self, category: str) -> str:
        """Compact ``label:count`` listing, ';'-joined, sorted by label."""
        return ";".join(
            f"{label}:{count}"
            for label, count in sorted(self.labels(category).items())
        )


_ACTIVE: list[OpCounters] = []


@contextmanager
def counting(counters: OpCounters):
    """Activate ``counters`` for every :func:`add` call in the block."""
    if not isinstance(counters, OpCounters):
        raise ValueError("counting requires an OpCounters instance")
    _ACTIVE.append(counters)
    try:
        yield counters
    finally:
        _ACTIVE.remove(counters)


def add(category: str, label: str, count) -> None:
    """Increment every active counter; no-op when none is active."""
    if not _ACTIVE:
        return
    for counters in _ACTIVE:
        counters.add(category, label, count)


# {{{ analytic counts from the executed configuration

def surviving_radial_node_count(
    radial_order, *, radial_rule="tanh-sinh-fast", mp_dps=50
) -> int:
    """Surviving nodes of the requested radial rule (float64-saturating
    tanh-sinh nodes are masked by the builder, so the count is measured by
    executing the node builder, not assumed from the requested order)."""
    import volumential.singular_integral_2d as squad

    nodes, _ = squad._duffy_radial_nodes_weights(
        radial_rule, int(radial_order), int(mp_dps)
    )
    return int(nodes.size)


def regular_node_count(dim, regular_order) -> int:
    """Regular (non-radial) node count of one Duffy region: the angular
    Gauss order in 2D, the two-axis tensor count in 3D — measured from the
    same node builder the quadratures use."""
    dim = int(dim)
    if dim not in (2, 3):
        raise NotImplementedError("Duffy node counting supports 2D and 3D")
    if dim == 2:
        import scipy.special as sps

        return int(sps.roots_legendre(int(regular_order))[0].size)
    import volumential.singular_integral_2d as squad

    nodes, _ = squad._duffy_regular_nodes_weights(dim - 1, int(regular_order))
    return int(np.asarray(nodes).shape[0])


def batched_duffy_nodes_per_entry(
    dim, regular_order, radial_order, *, radial_rule="tanh-sinh-fast"
) -> int:
    """Node-template size of the batched direct builder per reduced entry:
    ``2^d d! * regular nodes * surviving radial nodes`` (degenerate
    sign-octants keep their nodes in the generated code)."""
    from math import factorial

    dim = int(dim)
    if dim not in (2, 3):
        raise NotImplementedError("Duffy node counting supports 2D and 3D")
    return (
        (2**dim)
        * factorial(dim)
        * regular_node_count(dim, regular_order)
        * surviving_radial_node_count(radial_order, radial_rule=radial_rule)
    )


def reduced_entry_count(table) -> int:
    """Number of symmetry-reduced (quadratured) entries of a table."""
    return int(
        np.asarray(table._get_invariant_entry_info()["entry_ids"]).size
    )


def duffy_block_geometry(table) -> dict:
    """Counts of the grouped Duffy entry enumeration for one table geometry.

    Mirrors the region-degeneracy predicates of the grouped channel builder
    (:func:`volumential.rke_table_assembly._duffy_channel_entry_values`),
    which agree with the scalar builder's collinear-triangle skip: 2D
    triangles are dropped when ``|det| < 1e-14 extent^2``, 3D sign-octants
    when any edge length vanishes (each surviving octant contributes the
    ``d!`` axis-permutation cones).

    :returns: dict with ``n_reduced_entries``, ``n_blocks`` (distinct
        (case, target) blocks), ``n_active_regions`` (non-degenerate
        triangles/cones summed over blocks — the grouped builders evaluate
        each region once per block), and ``entry_weighted_active_regions``
        (the same regions summed per member entry — the scalar per-entry
        builder pays each region once per entry).
    """
    from itertools import product as iproduct

    from volumential.rke_table_assembly import _reduced_entry_groups

    dim = int(table.dim)
    if dim not in (2, 3):
        raise NotImplementedError("Duffy node counting supports 2D and 3D")
    extent = float(table.source_box_extent)
    entry_ids, groups = _reduced_entry_groups(table)

    n_active_regions = 0
    entry_weighted = 0
    for (case_index, target_index), members in groups.items():
        target = np.asarray(
            table.find_target_point(target_index, case_index),
            dtype=np.float64,
        )
        singular = np.clip(target, 0.0, extent)
        if dim == 2:
            corners = [
                np.array([0.0, 0.0]),
                np.array([extent, 0.0]),
                np.array([extent, extent]),
                np.array([0.0, extent]),
            ]
            active = 0
            for corner_index in range(4):
                edge1 = corners[corner_index] - singular
                edge2 = corners[(corner_index + 1) % 4] - singular
                det = edge1[0] * edge2[1] - edge1[1] * edge2[0]
                if np.abs(det) >= 1.0e-14 * extent * extent:
                    active += 1
        else:
            active = 0
            for signs in iproduct((-1.0, 1.0), repeat=dim):
                lengths = [
                    singular[axis] if signs[axis] < 0
                    else extent - singular[axis]
                    for axis in range(dim)
                ]
                if all(length > 0.0 for length in lengths):
                    active += 6  # the 3! axis-permutation cones
        n_active_regions += active
        entry_weighted += active * len(members)

    return {
        "n_reduced_entries": int(entry_ids.size),
        "n_blocks": len(groups),
        "n_active_regions": int(n_active_regions),
        "entry_weighted_active_regions": int(entry_weighted),
    }


def grouped_duffy_singular_nodes(
    table, regular_order, radial_order, *, geometry=None
) -> int:
    """Singular-quadrature nodes of one grouped (per-block) build over the
    reduced entry set — what one windowed/complex channel build evaluates."""
    if geometry is None:
        geometry = duffy_block_geometry(table)
    return geometry["n_active_regions"] * regular_node_count(
        int(table.dim), regular_order
    ) * surviving_radial_node_count(radial_order)


def scalar_duffy_singular_nodes(
    table, regular_order, radial_order, *, geometry=None
) -> int:
    """Singular-quadrature nodes of one scalar (per-entry) build over the
    reduced entry set — the legacy interpreted path pays every region once
    per member entry rather than once per block."""
    if geometry is None:
        geometry = duffy_block_geometry(table)
    return geometry["entry_weighted_active_regions"] * regular_node_count(
        int(table.dim), regular_order
    ) * surviving_radial_node_count(radial_order)


def nearfield_point_pairs(queue, traversal) -> int:
    """Near-field (target point, source quadrature point) pairs of one List 1
    pass: for every target box, its targets times the quadrature points of
    its neighbor source boxes.  In the box-FMM near field, sources of a box
    are its target quadrature points (``box_target_counts_nonchild`` serves
    both sides), matching the wrangler's near-field kernel launch."""
    target_boxes = np.asarray(traversal.target_boxes.get(queue))
    starts = np.asarray(traversal.neighbor_source_boxes_starts.get(queue))
    lists = np.asarray(traversal.neighbor_source_boxes_lists.get(queue))
    counts = np.asarray(
        traversal.tree.box_target_counts_nonchild.get(queue),
        dtype=np.int64,
    )

    total = 0
    for index, box in enumerate(target_boxes):
        neighbors = lists[starts[index]:starts[index + 1]]
        total += int(counts[box]) * int(np.sum(counts[neighbors]))
    return int(total)

# }}}
