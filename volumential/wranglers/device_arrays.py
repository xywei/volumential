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

__doc__ = """Device-array plumbing shared by both wrangler backends.

Command-queue discovery, index-permutation inversion and box level scaling.
"""

import numpy as np

import pyopencl as cl
import pyopencl.array


def level_to_rscale(tree, level):
    return tree.root_extent * (2**-level)


def inverse_id_map(queue, mapped_ids):
    """Given a index mapping as its mapped ids, compute its inverse,
    and return the inverse by the inversely-mapped ids.
    """
    cl_array = False
    if isinstance(mapped_ids, cl.array.Array):
        cl_array = True
        mapped_ids = mapped_ids.get(queue)

    inv_ids = np.zeros_like(mapped_ids)
    inv_ids[mapped_ids] = np.arange(len(mapped_ids))

    if cl_array:
        inv_ids = cl.array.to_device(queue, inv_ids)

    return inv_ids


def _queue_from_array_like(ary):
    if isinstance(ary, cl.array.Array):
        return ary.queue

    if isinstance(ary, np.ndarray) and ary.dtype == object:
        for entry in ary.flat:
            if isinstance(entry, cl.array.Array):
                return entry.queue

    return None


def _resolve_queue(queue, traversal, tree_indep):
    if queue is not None:
        return queue

    setup_actx = getattr(tree_indep, "_setup_actx", None)
    actx_queue = getattr(setup_actx, "queue", None)
    if actx_queue is not None:
        return actx_queue

    tree = getattr(traversal, "tree", None)
    if tree is not None:
        for ary_name in ("box_centers", "box_levels", "targets", "sources"):
            ary = getattr(tree, ary_name, None)
            ary_queue = _queue_from_array_like(ary)
            if ary_queue is not None:
                return ary_queue

    raise TypeError(
        "queue is required when it cannot be inferred from tree_indep/traversal"
    )

# vim: filetype=pyopencl:foldmethod=marker
