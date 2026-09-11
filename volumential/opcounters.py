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

The last section adds the per-phase far-field counts of experiment E6:
:func:`fmm_stage_operation_counts` prices the seven FMM stages of one solve
from the traversal's interaction lists and the wrangler's expansion sizes,
so that a driver pairing it with
:mod:`volumential.phase_profile` can report a solve's per-phase shares in
operations and in seconds side by side.
"""

from __future__ import annotations

import operator
from contextlib import contextmanager

import numpy as np

__all__ = [
    "FMM_FAR_FIELD_STAGES",
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
    "direct_build_fallback_reason",
    "direct_build_routing",
    "duffy_block_geometry",
    "expansion_coefficient_counts",
    "fmm_stage_operation_counts",
    "fmm_stage_operation_counts_from_traversal",
    "grouped_duffy_singular_nodes",
    "nearfield_point_pairs",
    "nearfield_point_pairs_from_counts",
    "reduced_entry_count",
    "regular_node_count",
    "scalar_duffy_singular_nodes",
    "surviving_radial_node_count",
]

#: Counter category for singular-quadrature node evaluations (Duffy
#: triangle/cone nodes at which a radial profile or kernel is evaluated).
SINGULAR_NODES = "singular_quadrature_nodes"

#: Counter category for smooth-rule (tensor Gauss-Legendre) node
#: evaluations of the windowed remainder.
SMOOTH_NODES = "smooth_rule_nodes"

#: Counter category for nodes passed through a windowed channel profile
#: ``psi_m``; each costs one special-function-class evaluation.
PROFILE_NODES = "channel_profile_nodes"

#: Counter category for kernel evaluations, keyed by the kernel's special
#: function.
KERNEL_EVALS = "kernel_evals"

#: Counter category for special-function evaluations keyed by function
#: (``expn``, ``erfc``, ``exp``, ``k0``, ``hankel1``, ...).
SPECIAL_EVALS = "special_function_evals"

#: Counter category for recombination fused multiply-adds and
#: coefficient-recurrence steps of the assembly.
RECOMBINATION_FLOPS = "recombination_flops"

#: Counter category for table entries built (one singular quadrature per
#: stored entry).
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


# {{{ executed build routing

def direct_build_routing(table) -> str:
    """The DuffyRadial builder that actually produced ``table``'s data.

    One of the recorded routings (``batched``, ``scalar``, ``scalar-adaptive``,
    ``scalar-fallback``), or ``unknown`` for a table whose payload predates the
    recording.  Read the *recorded* routing rather than re-evaluating the
    routing predicate: ``_supports_batched_duffy_builder`` reports what was
    attempted, and a batched build that failed and fell back to the scalar
    per-entry builder has a different cost class and a different converged
    accuracy at fixed orders.  A ``scalar-fallback`` here invalidates the
    batched node counts of the cost model for that table.
    """
    routing = getattr(table, "build_routing", None)
    if routing is None:
        return "unknown"
    return str(routing)


def direct_build_fallback_reason(table) -> str:
    """``"<ExceptionType>: <message>"`` of a recorded scalar fallback, else
    the empty string."""
    reason = getattr(table, "build_fallback_reason", None)
    if reason is None:
        return ""
    return str(reason)

# }}}


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
    (``volumential.rke_table_assembly._duffy_channel_entry_values``),
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


def nearfield_point_pairs_from_counts(
    *,
    target_boxes,
    neighbor_source_boxes_starts,
    neighbor_source_boxes_lists,
    box_target_counts_nonchild,
    box_source_counts_nonchild=None,
) -> int:
    """Near-field (target point, source point) pairs of one List 1 pass.

    For every target box, its targets times the source points of its
    neighbor source boxes.  ``box_source_counts_nonchild`` defaults to the
    target counts, which is the box-FMM near field's own convention (a
    box's sources are its target quadrature points).  Pass an explicit
    source-count array to price a pass that runs over a *different* source
    set — the online split remainder does exactly that when it evaluates on
    an interpolated smooth quadrature instead of the base nodes.
    """
    target_boxes = np.asarray(target_boxes, dtype=np.int64)
    starts = np.asarray(neighbor_source_boxes_starts, dtype=np.int64)
    lists = np.asarray(neighbor_source_boxes_lists, dtype=np.int64)
    target_counts = np.asarray(box_target_counts_nonchild, dtype=np.int64)
    if box_source_counts_nonchild is None:
        source_counts = target_counts
    else:
        source_counts = np.asarray(box_source_counts_nonchild, dtype=np.int64)

    if target_boxes.size == 0:
        return 0
    per_box_sources = _csr_group_sums(
        source_counts[lists] if lists.size else lists, starts
    )
    return int(np.sum(target_counts[target_boxes] * per_box_sources))


def nearfield_point_pairs(queue, traversal) -> int:
    """Near-field (target point, source quadrature point) pairs of one List 1
    pass: for every target box, its targets times the quadrature points of
    its neighbor source boxes.  In the box-FMM near field, sources of a box
    are its target quadrature points (``box_target_counts_nonchild`` serves
    both sides), matching the wrangler's near-field kernel launch."""
    return nearfield_point_pairs_from_counts(
        target_boxes=traversal.target_boxes.get(queue),
        neighbor_source_boxes_starts=(
            traversal.neighbor_source_boxes_starts.get(queue)
        ),
        neighbor_source_boxes_lists=(
            traversal.neighbor_source_boxes_lists.get(queue)
        ),
        box_target_counts_nonchild=(
            traversal.tree.box_target_counts_nonchild.get(queue)
        ),
    )

# }}}


# {{{ per-phase FMM stage operation counts (E6)

#: FMM far-field stage names, matching
#: :data:`volumential.phase_profile.FAR_FIELD_PHASES` without the ``far_``
#: prefix and the order in which ``drive_volume_fmm`` runs them.
FMM_FAR_FIELD_STAGES = (
    "form_multipoles",
    "coarsen_multipoles",
    "multipole_to_local",
    "eval_multipoles",
    "form_locals",
    "refine_locals",
    "eval_locals",
)


def _host(array, queue):
    """Host copy of a device or host array, without assuming which it is."""
    if array is None:
        return None
    if hasattr(array, "get"):
        return np.asarray(array.get(queue))
    return np.asarray(array)


def _csr_group_sums(values, starts):
    """Per-group sums of ``values`` under the CSR ``starts``, as int64.

    Handles empty groups and an empty ``lists`` array, which ``reduceat``
    does not.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(values, dtype=np.int64)]
    )
    return cumulative[starts[1:]] - cumulative[starts[:-1]]


def fmm_stage_operation_counts(
    *,
    box_levels,
    box_source_counts_nonchild,
    box_target_counts_nonchild,
    box_child_ids,
    source_boxes,
    source_parent_boxes,
    target_boxes,
    target_or_target_parent_boxes,
    from_sep_siblings_starts,
    from_sep_bigger_starts,
    from_sep_bigger_lists,
    sep_smaller_by_level,
    multipole_coeff_counts,
    local_coeff_counts,
) -> dict[str, int]:
    """Operation counts of the seven FMM far-field stages of one solve.

    Every count is a number of *coefficient touches*: one multiply-add
    against one expansion coefficient (or, for P2M/L2P/P2L, one
    source/target against one coefficient).  This is the dense-translation
    structural model, instantiated from the executed traversal's own
    interaction lists and the executed wrangler's own per-level expansion
    sizes -- nothing here is a constant, and nothing is a flop count of the
    generated OpenCL: a translation that sumpy accelerates (FFT-based or
    translation-class M2L, rotation-based operators) executes fewer machine
    operations than the dense count reported here, while still touching the
    same coefficients.  Consumers must read the counts as a structural
    decomposition of the stage graph, not as a hardware work estimate.

    The stage-by-stage rules follow the loops of
    :class:`sumpy.fmm.SumpyExpansionWrangler` at the executed revision, so
    where the textbook model and the implementation disagree the
    implementation wins.  Two such disagreements are reproduced here:

    * ``coarsen_multipoles`` (M2M) runs only for source levels
      ``nlevels-1 ... 3`` (``sumpy.fmm.SumpyExpansionWrangler
      .coarsen_multipoles``), because no level-1 box is well separated from
      another; translations into levels 0 and 1 are never performed and are
      therefore not counted.
    * ``refine_locals`` (L2L) is driven by the *target* box list, one
      parent-to-child translation per target-or-target-parent box at level
      ``>= 1``, not by an enumeration of children.

    :arg box_levels: ``tree.box_levels``, host array.
    :arg box_source_counts_nonchild: ``tree.box_source_counts_nonchild``.
    :arg box_target_counts_nonchild: ``tree.box_target_counts_nonchild``.
    :arg box_child_ids: ``tree.box_child_ids``, shape ``(2**dim, nboxes)``;
        a zero entry means "no such child" (boxtree convention).
    :arg source_boxes: ``traversal.source_boxes``.
    :arg source_parent_boxes: ``traversal.source_parent_boxes``.
    :arg target_boxes: ``traversal.target_boxes``.
    :arg target_or_target_parent_boxes:
        ``traversal.target_or_target_parent_boxes``.
    :arg from_sep_siblings_starts: List 2 CSR starts, indexed like
        ``target_or_target_parent_boxes``.
    :arg from_sep_bigger_starts: List 4 CSR starts, indexed like
        ``target_or_target_parent_boxes``.
    :arg from_sep_bigger_lists: List 4 CSR source-box lists.
    :arg sep_smaller_by_level: one ``(target_boxes, starts)`` pair per
        source level, from ``traversal.target_boxes_sep_smaller_by_source_level``
        and ``traversal.from_sep_smaller_by_level[i].starts`` (List 3 far).
    :arg multipole_coeff_counts: multipole coefficients per tree level.
    :arg local_coeff_counts: local coefficients per tree level.

    :returns: a dict with one key per entry of :data:`FMM_FAR_FIELD_STAGES`
        plus ``far_total``.
    """
    box_levels = np.asarray(box_levels, dtype=np.int64)
    nsources = np.asarray(box_source_counts_nonchild, dtype=np.int64)
    ntargets = np.asarray(box_target_counts_nonchild, dtype=np.int64)
    child_ids = np.asarray(box_child_ids, dtype=np.int64)
    source_boxes = np.asarray(source_boxes, dtype=np.int64)
    source_parent_boxes = np.asarray(source_parent_boxes, dtype=np.int64)
    target_boxes = np.asarray(target_boxes, dtype=np.int64)
    totpb = np.asarray(target_or_target_parent_boxes, dtype=np.int64)

    n_mpole = np.asarray(multipole_coeff_counts, dtype=np.int64)
    n_local = np.asarray(local_coeff_counts, dtype=np.int64)
    nlevels = int(n_mpole.size)
    if int(n_local.size) != nlevels:
        raise ValueError(
            "multipole and local coefficient counts must both be per level"
        )

    # P2M: every source of a source box enters every multipole coefficient
    # of that box's level.
    p2m = int(
        np.sum(nsources[source_boxes] * n_mpole[box_levels[source_boxes]])
    )

    # M2M: one dense child-to-parent translation per existing child of every
    # source parent box, for the source levels sumpy actually visits.
    m2m = 0
    for source_level in range(nlevels - 1, 2, -1):
        target_level = source_level - 1
        parents = source_parent_boxes[
            box_levels[source_parent_boxes] == target_level
        ]
        if parents.size == 0:
            continue
        n_children = int(np.count_nonzero(child_ids[:, parents]))
        m2m += n_children * int(n_mpole[source_level]) * int(
            n_mpole[target_level]
        )

    # M2L: one dense multipole-to-local translation per List 2 entry, source
    # and target on the same level.
    sib_starts = np.asarray(from_sep_siblings_starts, dtype=np.int64)
    n_list2 = sib_starts[1:] - sib_starts[:-1]
    totpb_levels = box_levels[totpb]
    m2l = int(
        np.sum(n_list2 * n_mpole[totpb_levels] * n_local[totpb_levels])
    )

    # M2P (List 3 far): every target of the target box is evaluated from
    # every listed multipole expansion at the source level.
    m2p = 0
    for source_level, (level_target_boxes, starts) in enumerate(
        sep_smaller_by_level
    ):
        level_target_boxes = np.asarray(level_target_boxes, dtype=np.int64)
        if level_target_boxes.size == 0:
            continue
        starts = np.asarray(starts, dtype=np.int64)
        n_list3 = starts[1:] - starts[:-1]
        m2p += int(
            np.sum(ntargets[level_target_boxes] * n_list3)
            * int(n_mpole[source_level])
        )

    # P2L (List 4 far): every source of every listed bigger box enters every
    # local coefficient of the target box's level.
    bigger_lists = np.asarray(from_sep_bigger_lists, dtype=np.int64)
    n_list4_sources = _csr_group_sums(
        nsources[bigger_lists] if bigger_lists.size else bigger_lists,
        from_sep_bigger_starts,
    )
    p2l = int(np.sum(n_list4_sources * n_local[totpb_levels]))

    # L2L: one dense parent-to-child translation per target-or-target-parent
    # box at level >= 1.
    l2l = 0
    for target_level in range(1, nlevels):
        n_boxes = int(np.count_nonzero(totpb_levels == target_level))
        if n_boxes == 0:
            continue
        l2l += n_boxes * int(n_local[target_level - 1]) * int(
            n_local[target_level]
        )

    # L2P: every target reads every local coefficient of its box's level.
    l2p = int(
        np.sum(ntargets[target_boxes] * n_local[box_levels[target_boxes]])
    )

    counts = {
        "form_multipoles": p2m,
        "coarsen_multipoles": m2m,
        "multipole_to_local": m2l,
        "eval_multipoles": m2p,
        "form_locals": p2l,
        "refine_locals": l2l,
        "eval_locals": l2p,
    }
    counts["far_total"] = int(sum(counts.values()))
    return counts


def expansion_coefficient_counts(wrangler) -> tuple[list[int], list[int]]:
    """Per-level ``(multipole, local)`` coefficient counts of a wrangler.

    Read off the executed expansion classes at the executed per-level
    orders, so a kernel-specific expansion (the 2D Helmholtz/Yukawa Bessel
    expansions, say, at ``2 * order + 1`` coefficients) is priced as the run
    actually priced it rather than by a Taylor-order formula.
    """
    tree_indep = wrangler.tree_indep
    orders = [int(order) for order in wrangler.level_orders]
    multipole = [len(tree_indep.multipole_expansion(order)) for order in orders]
    local = [len(tree_indep.local_expansion(order)) for order in orders]
    return multipole, local


def fmm_stage_operation_counts_from_traversal(
    queue, traversal, wrangler
) -> dict[str, int]:
    """:func:`fmm_stage_operation_counts` for an executed traversal/wrangler."""
    tree = traversal.tree
    multipole, local = expansion_coefficient_counts(wrangler)

    sep_smaller_by_level = []
    level_target_boxes = getattr(
        traversal, "target_boxes_sep_smaller_by_source_level", None
    )
    from_sep_smaller = getattr(traversal, "from_sep_smaller_by_level", None)
    if level_target_boxes is not None and from_sep_smaller is not None:
        for boxes, ssn in zip(level_target_boxes, from_sep_smaller, strict=True):
            host_boxes = _host(boxes, queue)
            if host_boxes is None or host_boxes.size == 0:
                sep_smaller_by_level.append(
                    (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64))
                )
                continue
            sep_smaller_by_level.append(
                (host_boxes, _host(ssn.starts, queue))
            )
    else:
        sep_smaller_by_level = [
            (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64))
            for _ in range(len(multipole))
        ]

    bigger_starts = _host(
        getattr(traversal, "from_sep_bigger_starts", None), queue
    )
    bigger_lists = _host(
        getattr(traversal, "from_sep_bigger_lists", None), queue
    )
    if bigger_starts is None:
        bigger_starts = np.zeros(1, dtype=np.int64)
        bigger_lists = np.zeros(0, dtype=np.int64)

    return fmm_stage_operation_counts(
        box_levels=_host(tree.box_levels, queue),
        box_source_counts_nonchild=_host(
            tree.box_source_counts_nonchild, queue
        ),
        box_target_counts_nonchild=_host(
            tree.box_target_counts_nonchild, queue
        ),
        box_child_ids=_host(tree.box_child_ids, queue),
        source_boxes=_host(traversal.source_boxes, queue),
        source_parent_boxes=_host(traversal.source_parent_boxes, queue),
        target_boxes=_host(traversal.target_boxes, queue),
        target_or_target_parent_boxes=_host(
            traversal.target_or_target_parent_boxes, queue
        ),
        from_sep_siblings_starts=_host(
            traversal.from_sep_siblings_starts, queue
        ),
        from_sep_bigger_starts=bigger_starts,
        from_sep_bigger_lists=bigger_lists,
        sep_smaller_by_level=sep_smaller_by_level,
        multipole_coeff_counts=multipole,
        local_coeff_counts=local,
    )

# }}}
