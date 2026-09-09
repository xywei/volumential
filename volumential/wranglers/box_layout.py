__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """Validation of the box/particle layout that table lookups assume.

Table-based near-field evaluation only makes sense when every active box
carries exactly one box-local quadrature rule; these helpers check that and
build the box-local particle index map.
"""

import sys

import numpy as np

import pyopencl as cl
import pyopencl.array


def _compute_box_local_ids(queue, tree, n_q_points: int) -> "cl.array.Array":
    """Index of each particle within its own box, in tree particle order."""
    if not getattr(tree, "sources_are_targets", False):
        raise ValueError(
            "table-based near-field evaluation requires sources and targets "
            "to coincide (tree.sources_are_targets=True)"
        )

    n_particles = tree.ntargets
    if n_particles % n_q_points != 0:
        raise ValueError("particle count is not divisible by box-local quadrature size")

    user_local_ids = np.tile(
        np.arange(n_q_points, dtype=np.int32),
        n_particles // n_q_points,
    )
    sorted_target_ids = tree.sorted_target_ids.get(queue)
    return cl.array.to_device(queue, user_local_ids[sorted_target_ids])


def _validate_table_box_particle_layout(
    queue,
    tree,
    target_boxes,
    source_boxes,
    n_q_points: int,
) -> None:
    """Raise unless every active box holds exactly *n_q_points* particles."""
    box_counts = tree.box_target_counts_nonchild.get(queue)

    if hasattr(target_boxes, "get"):
        target_box_ids = target_boxes.get(queue)
    else:
        target_box_ids = np.asarray(target_boxes)

    if hasattr(source_boxes, "get"):
        source_box_ids = source_boxes.get(queue)
    else:
        source_box_ids = np.asarray(source_boxes)

    target_counts = box_counts[target_box_ids]
    source_counts = box_counts[source_box_ids]

    target_ok = np.all(target_counts == n_q_points)
    source_ok = np.all(source_counts == n_q_points)
    if target_ok and source_ok:
        return

    bad_target = np.unique(target_counts[target_counts != n_q_points])
    bad_source = np.unique(source_counts[source_counts != n_q_points])

    raise ValueError(
        "table-based near-field evaluation requires exactly "
        f"{n_q_points} quadrature points per active source/target box; "
        f"found target counts {bad_target.tolist()} and source counts "
        f"{bad_source.tolist()}. Build the particle tree from the mesh box-tree "
        "(build_particle_tree_from_box_tree) to preserve per-cell quadrature layout."
    )


def _array_layout_cache_token(ary) -> tuple:
    """Cheap identity token for an index array, stable across re-wrapping."""
    if isinstance(ary, cl.array.Array):
        base_data = getattr(ary, "base_data", None)
        int_ptr = getattr(base_data, "int_ptr", None)
        if int_ptr is not None:
            return (
                "cl",
                int(int_ptr),
                int(getattr(ary, "offset", 0)),
                int(ary.size),
                ary.dtype.str,
            )
    return ("py", id(ary))


_LEGACY_MODULE = "volumential.expansion_wrangler_fpnd"


def _active_layout_validator():
    """Return the validator to call, honoring the legacy module's binding.

    :mod:`volumential.expansion_wrangler_fpnd` re-exports
    :func:`_validate_table_box_particle_layout`, and callers (tests included)
    replace the attribute *there* to observe or stub validation.  Looking the
    name up through that module keeps those replacements effective now that the
    two functions no longer share a module namespace.
    """
    legacy = sys.modules.get(_LEGACY_MODULE)
    return getattr(
        legacy, "_validate_table_box_particle_layout",
        _validate_table_box_particle_layout,
    )


def _validate_table_box_particle_layout_cached(
    queue,
    tree,
    target_boxes,
    source_boxes,
    n_q_points: int,
    validation_cache: set | None,
) -> None:
    """Validate the box layout once per distinct ``(boxes, n_q_points)`` triple.

    *validation_cache* is a mutable set owned by the caller; pass ``None`` to
    validate unconditionally.
    """
    validate = _active_layout_validator()

    if validation_cache is None:
        validate(
            queue,
            tree,
            target_boxes,
            source_boxes,
            n_q_points,
        )
        return

    cache_key = (
        _array_layout_cache_token(target_boxes),
        _array_layout_cache_token(source_boxes),
        int(n_q_points),
    )
    if cache_key in validation_cache:
        return

    validate(
        queue,
        tree,
        target_boxes,
        source_boxes,
        n_q_points,
    )
    validation_cache.add(cache_key)

# vim: filetype=pyopencl:foldmethod=marker
