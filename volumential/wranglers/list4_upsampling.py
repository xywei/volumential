__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

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

__doc__ = """Upsampled P2L for List 4 sources.

On a 2:1 balanced tree a List 4 source box is a leaf colleague of the target
box's parent. It is twice the size of the target box and only half its own
size away from it, where a List 2 or List 3 source is never closer than its
own size. Point quadrature at the box's ``q_order**dim`` nodes is much less
accurate at that distance than for the other far pairs, and the gap grows
with ``q_order``.

So the wranglers form the local expansion of a List 4 pair from an upsampled
source instead: the box's nodal density is interpolated to a tensor-product
Gauss rule with more nodes per axis, and P2L runs from those nodes. The
interpolation and the change of weights fold into one matrix that depends only
on ``q_order``, the finer order and ``dim``
(:func:`list4_upsampling_matrix`), so a source box's upsampled strengths are
that matrix times its own strengths. Every other far pair keeps point
quadrature.

A higher-order Gauss rule on the whole box is used rather than the same rule
on each sub-box: for the same number of nodes it is far more accurate at the
List 4 distance. At :data:`LIST4_UPSAMPLING` times ``q_order`` nodes per axis,
the error of these pairs falls below that of the far pairs one to two source
sizes away, in 2-D and 3-D.

The upsampling is on by default in 1-D and 2-D and off in 3-D
(:func:`default_list4_upsampling`), by its cost: P2L from the finer nodes added
3 to 4% of the near-field time on 2-D graded trees and 10 to 45% on 3-D ones,
where the finer rule has about 3.4 times as many nodes and P2L costs more per
node than the near field.
"""

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from numbers import Real

import numpy as np

from volumential.wranglers.box_layout import _array_layout_cache_token


logger = logging.getLogger(__name__)


#: The upsampling factor the wranglers use where it is on by default: the
#: upsampled rule has ``ceil(1.5 * q_order)`` Gauss nodes per axis.
LIST4_UPSAMPLING = 1.5

# Largest distance, in the reference coordinates of a box ([-1, 1] per axis),
# between a source and the Gauss node it is matched to.
_NODE_MATCH_TOL = 1.0e-6


def default_list4_upsampling(dim: int) -> float:
    """The wranglers' ``list4_upsampling`` when none is given.

    :data:`LIST4_UPSAMPLING` in 1-D and 2-D, and 1 (point quadrature) in 3-D,
    where the upsampled P2L costs more than a tenth of the near-field time.
    """
    return float(LIST4_UPSAMPLING) if int(dim) < 3 else 1.0


def normalize_list4_upsampling(list4_upsampling, dim: int) -> float:
    """Validate a ``list4_upsampling`` argument and return it as a float.

    ``None`` stands for :func:`default_list4_upsampling` of *dim*; ``1`` turns
    the upsampling off.

    :raises TypeError: for a value that is not a real number, booleans
        included.
    :raises ValueError: for a value below 1 or not finite.
    """
    if list4_upsampling is None:
        return default_list4_upsampling(dim)
    if isinstance(list4_upsampling, bool) or not isinstance(
        list4_upsampling, Real
    ):
        raise TypeError(
            "list4_upsampling must be a real number >= 1 (1 turns it off), "
            f"not {list4_upsampling!r}"
        )
    factor = float(list4_upsampling)
    if not math.isfinite(factor) or factor < 1:
        raise ValueError(
            "list4_upsampling must be a finite number >= 1 (1 turns it off), "
            f"not {list4_upsampling!r}"
        )
    return factor


def list4_upsampled_q_order(q_order: int, list4_upsampling: float) -> int:
    """Gauss nodes per axis of the upsampled rule for a List 4 source box.

    That is ``ceil(list4_upsampling * q_order)``; a result equal to
    *q_order* means no upsampling.
    """
    q_order = int(q_order)
    # The tolerance keeps 1.5 * 4 at 6 in the presence of rounding.
    return max(q_order, math.ceil(float(list4_upsampling) * q_order - 1e-9))


def _gauss_legendre(order: int):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def _lagrange_basis_values(nodes, points):
    """``values[i, j]``: the Lagrange basis polynomial of ``nodes[j]`` at
    ``points[i]``."""
    values = np.ones((len(points), len(nodes)))
    for j, node_j in enumerate(nodes):
        for k, node_k in enumerate(nodes):
            if k != j:
                values[:, j] *= (points - node_k) / (node_j - node_k)
    return values


@lru_cache(maxsize=32)
def _list4_upsampling_rule(q_order: int, fine_q_order: int, dim: int):
    coarse_nodes, coarse_weights = _gauss_legendre(q_order)
    fine_nodes, fine_weights = _gauss_legendre(fine_q_order)
    basis = _lagrange_basis_values(coarse_nodes, fine_nodes)

    # Tensor indices in the order the box mesh lays out its nodes: numpy's
    # ``meshgrid(..., indexing="ij")``, the first axis varying slowest.
    fine_index = np.array(list(product(range(fine_q_order), repeat=dim)))
    coarse_index = np.array(list(product(range(q_order), repeat=dim)))

    matrix = np.ones((len(fine_index), len(coarse_index)))
    for axis in range(dim):
        matrix *= (
            fine_weights[fine_index[:, axis], None]
            * basis[fine_index[:, axis][:, None], coarse_index[None, :, axis]]
            / coarse_weights[None, coarse_index[:, axis]]
        )

    ref_nodes = np.ascontiguousarray(fine_nodes[fine_index].T)
    matrix.setflags(write=False)
    ref_nodes.setflags(write=False)
    return ref_nodes, matrix


def list4_upsampling_matrix(q_order: int, fine_q_order: int, dim: int):
    """The upsampled rule for one source box, on the reference box.

    :returns: ``(ref_nodes, matrix)``. *ref_nodes* has shape
        ``(dim, fine_q_order**dim)`` and holds the tensor-product Gauss nodes
        on :math:`[-1, 1]^d`. *matrix* has shape
        ``(fine_q_order**dim, q_order**dim)``: it maps a box's strengths
        (density times quadrature weight) at its own nodes, in tensor order,
        to the strengths at the finer nodes. Row ``i`` is the finer weight of
        node ``i`` times the value there of each coarse node's Lagrange basis
        polynomial, divided by that coarse node's weight. The box size cancels,
        so one matrix serves every level.

    Both arrays are cached per ``(q_order, fine_q_order, dim)`` and read-only.
    """
    return _list4_upsampling_rule(int(q_order), int(fine_q_order), int(dim))


@dataclass(frozen=True)
class List4UpsampledSources:
    """The upsampled sources of a traversal's List 4 source boxes, on the
    host. :attr:`box_source_starts` and :attr:`box_source_counts_nonchild`
    replace the tree's arrays of the same names in P2L, and :attr:`sources`
    the tree's sources."""

    #: The List 4 source boxes that hold sources, each once.
    source_boxes: np.ndarray
    #: Shape ``(len(source_boxes), q_order**dim)``: the tree-order source
    #: index of each box's nodes, in tensor order.
    gather: np.ndarray
    #: The matrix of :func:`list4_upsampling_matrix`.
    matrix: np.ndarray
    #: Shape ``(dim, len(source_boxes) * fine_q_order**dim)``: the finer
    #: nodes, box after box.
    sources: np.ndarray
    #: Per box, where its finer nodes start in :attr:`sources` (zero for the
    #: boxes that are not List 4 sources).
    box_source_starts: np.ndarray
    #: Per box, how many finer nodes it has (zero for the boxes that are not
    #: List 4 sources).
    box_source_counts_nonchild: np.ndarray

    @property
    def nsources(self) -> int:
        """Number of upsampled sources."""
        return int(self.sources.shape[1])

    def upsample(self, strengths: np.ndarray) -> np.ndarray:
        """Upsampled strengths, in the order of :attr:`sources`.

        :arg strengths: host array of the strengths of every source of the
            tree, in tree order.
        """
        return self.upsample_gathered(np.asarray(strengths)[self.gather])

    def upsample_gathered(self, coarse: np.ndarray) -> np.ndarray:
        """Upsampled strengths from the strengths at :attr:`gather` only.

        :arg coarse: host array of the strengths at the sources that
            :attr:`gather` lists, in its order (flat or of its shape).
        """
        coarse = np.asarray(coarse).reshape(self.gather.shape)
        dtype = np.result_type(coarse.dtype, self.matrix.dtype)
        matrix = self.matrix.astype(dtype, copy=False)
        return np.ascontiguousarray((coarse @ matrix.T).reshape(-1))


def build_list4_upsampled_sources(
    *,
    lists,
    sources,
    box_centers,
    box_levels,
    root_extent,
    box_source_starts,
    box_source_counts_nonchild,
    q_order,
    fine_q_order,
):
    """Place the upsampled nodes of the List 4 source boxes in *lists*.

    Every argument is a host array: *lists* is the traversal's
    ``from_sep_bigger_lists`` and the others are the tree's arrays of the same
    names (*sources* with shape ``(dim, nsources)``).

    Each source box must hold ``q_order**dim`` sources at the tensor-product
    Gauss nodes of the box, in any order, as the box mesh places them.

    :returns: ``(upsampled, reason)``: a :class:`List4UpsampledSources` and
        ``None``, or ``None`` and why the layout does not allow upsampling.
    """
    lists = np.asarray(lists)
    sources = np.asarray(sources)
    box_centers = np.asarray(box_centers)
    box_levels = np.asarray(box_levels)
    starts = np.asarray(box_source_starts)
    counts = np.asarray(box_source_counts_nonchild)

    dim = int(sources.shape[0])
    q_order = int(q_order)
    fine_q_order = int(fine_q_order)
    n_q_points = q_order**dim
    ref_nodes, matrix = list4_upsampling_matrix(q_order, fine_q_order, dim)
    n_fine = int(ref_nodes.shape[1])

    source_boxes = np.unique(lists.astype(np.int64))
    source_boxes = source_boxes[counts[source_boxes] > 0]

    nboxes = len(counts)
    new_starts = np.zeros(nboxes, dtype=starts.dtype)
    new_counts = np.zeros(nboxes, dtype=counts.dtype)
    nbox = len(source_boxes)

    if nbox == 0:
        return List4UpsampledSources(
            source_boxes=source_boxes,
            gather=np.zeros((0, n_q_points), dtype=np.int64),
            matrix=matrix,
            sources=np.zeros((dim, 0), dtype=sources.dtype),
            box_source_starts=new_starts,
            box_source_counts_nonchild=new_counts,
        ), None

    bad_counts = np.unique(counts[source_boxes][counts[source_boxes] != n_q_points])
    if len(bad_counts):
        return None, (
            f"List 4 source boxes hold {bad_counts.tolist()} sources where "
            f"q_order**dim = {n_q_points} is needed"
        )

    gather = starts[source_boxes].astype(np.int64)[:, None] + np.arange(n_q_points)
    centers = box_centers[:, source_boxes]
    half_sizes = 0.5 * float(root_extent) / 2.0 ** box_levels[source_boxes]

    # Reference coordinates of every node, shape (dim, nbox, n_q_points).
    ref = (sources[:, gather] - centers[:, :, None]) / half_sizes[None, :, None]
    gauss_nodes, _ = _gauss_legendre(q_order)
    nearest = np.argmin(np.abs(ref[..., None] - gauss_nodes), axis=-1)
    mismatch = float(np.max(np.abs(ref - gauss_nodes[nearest])))
    if mismatch > _NODE_MATCH_TOL:
        return None, (
            "the sources of a List 4 source box are not at the box's "
            f"{q_order}-point Gauss nodes (off by {mismatch:.1e} in reference "
            "coordinates)"
        )

    tensor_index = np.zeros(nearest.shape[1:], dtype=np.int64)
    for axis in range(dim):
        tensor_index = tensor_index * q_order + nearest[axis]
    if not np.all(np.sort(tensor_index, axis=1) == np.arange(n_q_points)):
        return None, "a List 4 source box has two sources at the same node"

    order = np.argsort(tensor_index, axis=1)
    gather = np.take_along_axis(gather, order, axis=1)

    fine_sources = (
        centers[:, :, None] + half_sizes[None, :, None] * ref_nodes[:, None, :]
    ).reshape(dim, -1)

    new_starts[source_boxes] = np.arange(nbox) * n_fine
    new_counts[source_boxes] = n_fine

    return List4UpsampledSources(
        source_boxes=source_boxes,
        gather=gather,
        matrix=matrix,
        sources=np.ascontiguousarray(fine_sources.astype(sources.dtype)),
        box_source_starts=new_starts,
        box_source_counts_nonchild=new_counts,
    ), None


class List4UpsamplingMixin:
    """Wrangler state for the upsampled List 4 P2L.

    A wrangler calls :meth:`_init_list4_upsampling` from its constructor,
    after it has set ``quad_order``, and :meth:`_get_list4_upsampled_sources`
    from ``form_locals``, which falls back to point quadrature when that
    returns ``None``.

    .. attribute:: list4_upsampling

        The upsampling factor, at least 1; 1 means point quadrature.
    """

    def _init_list4_upsampling(self, list4_upsampling, dim: int) -> None:
        self.list4_upsampling = normalize_list4_upsampling(list4_upsampling, dim)
        self._list4_upsampling_cache = None
        self._list4_upsampling_logged = False

    @property
    def list4_upsampled_q_order(self) -> int:
        """Gauss nodes per axis of the rule List 4 sources are upsampled to.

        Equal to ``quad_order`` when the upsampling is off.
        """
        return list4_upsampled_q_order(self.quad_order, self.list4_upsampling)

    def _list4_upsampling_unsupported_reason(self):
        """Why this wrangler's far field cannot be upsampled, or ``None``."""
        if getattr(self, "source_extra_kwargs", None):
            return (
                "the source kernels take per-source arguments "
                f"({', '.join(sorted(self.source_extra_kwargs))})"
            )
        return None

    def _get_list4_upsampled_sources(self, lists, to_host):
        """The upsampled List 4 sources for *lists*, or ``None``.

        ``None`` means point quadrature: the upsampling is off, the wrangler
        cannot use it, or the tree's layout does not allow it. The last two
        are logged once per wrangler.

        :arg lists: the traversal's ``from_sep_bigger_lists``.
        :arg to_host: callable that returns a host copy of a tree array.
        """
        fine_q_order = self.list4_upsampled_q_order
        if fine_q_order <= self.quad_order:
            return None

        reason = self._list4_upsampling_unsupported_reason()
        log = logger.info
        upsampled = None
        if reason is None:
            log = logger.warning
            key = (_array_layout_cache_token(lists), int(lists.size), fine_q_order)
            cache = self._list4_upsampling_cache
            if cache is not None and cache[0] == key:
                _, upsampled, reason = cache
            else:
                tree = self.tree
                dim = int(tree.dimensions)
                upsampled, reason = build_list4_upsampled_sources(
                    lists=to_host(lists),
                    sources=np.array([to_host(tree.sources[i]) for i in range(dim)]),
                    box_centers=to_host(tree.box_centers),
                    box_levels=to_host(tree.box_levels),
                    root_extent=tree.root_extent,
                    box_source_starts=to_host(tree.box_source_starts),
                    box_source_counts_nonchild=to_host(
                        tree.box_source_counts_nonchild
                    ),
                    q_order=self.quad_order,
                    fine_q_order=fine_q_order,
                )
                self._list4_upsampling_cache = (key, upsampled, reason)

        if upsampled is None and not self._list4_upsampling_logged:
            self._list4_upsampling_logged = True
            log(
                "List 4 sources use point quadrature, not the requested "
                "upsampling (list4_upsampling=%g): %s",
                self.list4_upsampling,
                reason,
            )
        return upsampled


# vim: filetype=pyopencl:foldmethod=marker
