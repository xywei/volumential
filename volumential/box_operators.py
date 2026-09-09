__copyright__ = "Copyright (C) 2018 Xiaoyu Wei"

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

__doc__ = """Box-local (box-specific) maps, reductions and modal filters.

A box-specific operator acts on the quadrature nodes of one leaf box at a time,
independently of every other box.  This module owns those operators and the
filter vectors they take; all of its public names are re-exported by
:mod:`volumential.tools` for backwards compatibility.

.. autoclass:: BoxSpecificMap
.. autoclass:: DiscreteLegendreTransform
.. autoclass:: InverseDiscreteLegendreTransform
.. autoclass:: BoxSpecificReduction
.. autoclass:: BoxSum
.. autofunction:: generate_leading_order_filtering
"""

import logging
from typing import Any

import numpy as np

import loopy as lp
import pyopencl as cl
import pyopencl.array

from volumential.kernel_cache import KernelCacheWrapper


logger = logging.getLogger(__name__)


def _box_filter_multiplier(queue, filtering, n_box_nodes: int):
    """Return the per-node filter multiplier for a box-specific operator.

    *filtering* is either *None* (no filtering, i.e. all-ones) or a
    :class:`pyopencl.array.Array` of shape ``(n_box_nodes,)``.
    """
    if filtering is None:
        return 1 + cl.array.zeros(queue, n_box_nodes, np.float64)

    if isinstance(filtering, cl.array.Array):
        assert filtering.shape == (n_box_nodes,)
        return filtering

    raise RuntimeError(f"Invalid filtering argument: {filtering!s}")


def _split_over_boxes(knl, ncpus=None):
    """Tag the box iname ``bid`` for parallel execution over *ncpus* groups."""
    if ncpus is None:
        import multiprocessing

        ncpus = multiprocessing.cpu_count()

    return lp.split_iname(
        knl, split_iname="bid", inner_length=ncpus, inner_tag="g.0"
    )


# {{{ box-specific maps


class BoxSpecificMap(KernelCacheWrapper):
    """
    Box-specific transform that maps between datum defined on quadrature
    nodes. Being box-specific means that the transform for each box is
    independent from the rest of the boxes.
    """


# {{{ discrete Legendre transform


class DiscreteLegendreTransform(BoxSpecificMap):
    """
    Transform from nodal values to Legendre polynomial coefficients
    for all cells (leaf boxes of a boxtree Tree object).
    It is assumed that the traversal is built over a tree where the
    sources and targets coincide.
    """

    def __init__(self, dim, degree):
        """
        :arg dim
        :arg degree Number of nodes in each axis direction.
        """
        assert dim > 0
        self.dim = dim
        assert degree > 0
        self.degree = degree

        # Template interval
        self.template_interval = [-1.0, 1.0]
        self.template_interval_extent = 2.0
        self.template_interval_center = 0.0

        self.leg_tplt_x, self.leg_tplt_w = np.polynomial.legendre.leggauss(degree)

        if self.dim == 1:
            self.V = np.polynomial.legendre.legvander(
                self.leg_tplt_x, self.degree - 1
            )
            self.W = self.leg_tplt_w.reshape(-1)

        elif self.dim == 2:
            x, y = np.meshgrid(self.leg_tplt_x, self.leg_tplt_x)
            self.V = np.polynomial.legendre.legvander2d(
                x.reshape(-1), y.reshape(-1), [self.degree - 1] * self.dim
            )
            self.W = (
                self.leg_tplt_w[None, :] * self.leg_tplt_w[:, None]
            ).reshape(-1)

        elif self.dim == 3:
            x, y, z = np.meshgrid(self.leg_tplt_x, self.leg_tplt_x, self.leg_tplt_x)
            self.V = np.polynomial.legendre.legvander3d(
                x.reshape(-1),
                y.reshape(-1),
                z.reshape(-1),
                [self.degree - 1] * self.dim,
            )
            self.W = (
                self.leg_tplt_w[:, None, None]
                * self.leg_tplt_w[None, :, None]
                * self.leg_tplt_w[None, None, :]
            ).reshape(-1)

        else:
            raise NotImplementedError(f"Dimension {self.dim} is not supported")

        # Vandermonde matrix: each column corresponds to one basis function
        assert self.V.shape == (self.degree**self.dim, self.degree**self.dim)
        assert self.W.shape == (self.degree**self.dim,)

        # Normalizers
        self.I = np.ascontiguousarray(np.diag((self.V.T * self.W) @ self.V))
        assert self.I.shape == (self.degree**self.dim,)

        # Fix strides for loopy
        self.V = np.ascontiguousarray(self.V)

        # Check orthogonality
        ortho_resid = np.linalg.norm(
            self.V.T * np.matmul(self.W, self.V) - np.diag(self.I)
        )
        if ortho_resid > 1e-13:
            logger.warning(
                "Legendre polynomials' orthogonality residual = %f", ortho_resid
            )

        self.name = "DiscreteLegendreTransform"

    def get_cache_key(self) -> tuple[Any, ...]:
        """Return a hashable key identifying the generated kernel."""
        return (type(self).__name__, f"{self.dim}D", f"degree={self.degree}")

    def get_kernel(self, **kwargs):
        """Return the nodal-to-modal transform kernel."""
        loopy_knl = lp.make_kernel(
            [
                "{ [ bid ] : 0 <= bid < n_boxes }",
                "{ [ mid ] : 0 <= mid < n_box_nodes }",
                "{ [ nid ] : 0 <= nid < n_box_nodes }",
            ],
            [
                """
                for bid
                    <> box_id       = boxes[bid]
                    <> box_node_beg = box_node_starts[box_id]

                    # Rescale weights based on template interval sizes.
                    # Not needed since the rscl in both the numerator and
                    # the denominator and is canceled.
                    #
                    # <> box_level    = box_levels[box_id]
                    # <> box_extent   = root_extent * (1.0 / (2**box_level))
                    # <> weight_rscl  = (box_extent / 2.0)**dim

                    for mid

                        <> mode_id = box_node_beg + mid

                        for nid
                            <> user_node_id = user_node_ids[box_node_beg + nid]
                        end

                        result[mode_id] = sum(
                                              nid,
                                              (
                                              func[user_node_id]
                                              * weight[nid]
                                              * vandermonde[nid, mid]
                                              ) * filter_multiplier[nid]
                                             ) / normalizer[mid]
                    end
                end
                """
            ],
            [
                lp.ValueArg("n_box_nodes, n_boxes", np.int32),
                # lp.ValueArg("root_extent", np.float64),
                lp.GlobalArg(
                    "weight, normalizer, filter_multiplier",
                    np.float64,
                    "n_box_nodes",
                ),
                lp.GlobalArg("vandermonde", np.float64, "n_box_nodes, n_box_nodes"),
                lp.GlobalArg("func", np.float64, "n_box_nodes * n_boxes"),
                "...",
            ],
            name="discrete_legendre_transform",
            lang_version=(2018, 2),
        )

        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        return loopy_knl

    def get_optimized_kernel(self, ncpus=None, **kwargs):
        """Return the transform kernel parallelized over boxes."""
        return _split_over_boxes(self.get_kernel(**kwargs), ncpus)

    def __call__(self, queue, traversal, nodal_vals, filtering=None, **kwargs):
        """
        :arg traversal
        :arg nodal_vals CL array of nodal values.
        :arg filtering Box-wide filter given by an CL array or None.
        """
        filter_multiplier = _box_filter_multiplier(
            queue, filtering, self.degree**self.dim
        )

        knl = self.get_cached_optimized_kernel()
        knl_exec = knl.executor(queue.context)

        _evt, res = knl_exec(
            queue,
            boxes=traversal.target_boxes,
            box_node_starts=traversal.tree.box_target_starts,
            user_node_ids=traversal.tree.user_source_ids,
            # box_levels=traversal.tree.box_levels,
            # root_extent=traversal.tree.root_extent,
            func=nodal_vals,
            weight=cl.array.to_device(queue, self.W),
            vandermonde=cl.array.to_device(queue, self.V),
            normalizer=cl.array.to_device(queue, self.I),
            n_box_nodes=self.degree**self.dim,
            n_boxes=traversal.target_boxes.shape[0],
            filter_multiplier=filter_multiplier,
            result=cl.array.zeros_like(nodal_vals),
        )

        return res["result"]


# }}} End discrete Legendre transform

# {{{ inverse discrete Legendre transform


class InverseDiscreteLegendreTransform(BoxSpecificMap):
    """
    Box-specific transform that maps box-local modal coefficients
    to nodal values. Inverse of :class:`DiscreteLegendreTransform`.
    """


# }}} End inverse discrete Legendre transform

# }}} End box-specific maps

# {{{ box-specific reductions


class BoxSpecificReduction(KernelCacheWrapper):
    """
    Box-specific reduction that maps for each box a data vector defined
    on the quadrature nodes to a scalar.
    Being box-specific means that the reductions for each box is
    independent from the rest of the boxes.
    """


# {{{ sum


class BoxSum(BoxSpecificReduction):
    """
    Adds up nodal values within each box.
    """

    def __init__(self, dim, degree):
        """
        :arg dim
        :arg degree Number of nodes in each axis direction.
        """
        assert dim > 0
        self.dim = dim
        assert degree > 0
        self.degree = degree

        self.name = "BoxSum"

    def get_cache_key(self) -> tuple[Any, ...]:
        """Return a hashable key identifying the generated kernel."""
        return (type(self).__name__, f"{self.dim}D", f"degree={self.degree}")

    def get_kernel(self, **kwargs):
        """Return the box-wise filtered sum kernel."""
        loopy_knl = lp.make_kernel(
            [
                "{ [ bid ] : 0 <= bid < n_boxes }",
                "{ [ nid ] : 0 <= nid < n_box_nodes }",
            ],
            [
                """
                for bid
                    <> box_id       = boxes[bid]
                    <> box_node_beg = box_node_starts[box_id]

                    result[bid] = sum(nid,
                                      func[box_node_beg + nid]
                                      * filter_multiplier[nid])
                end
                """
            ],
            [
                lp.ValueArg("n_box_nodes, n_boxes", np.int32),
                lp.GlobalArg("filter_multiplier", np.float64, "n_box_nodes"),
                lp.GlobalArg("func", np.float64, "n_box_nodes * n_boxes"),
                "...",
            ],
            name="box_filtered_sum",
            lang_version=(2018, 2),
        )

        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        return loopy_knl

    def get_optimized_kernel(self, ncpus=None, **kwargs):
        """Return the sum kernel parallelized over boxes."""
        return _split_over_boxes(self.get_kernel(**kwargs), ncpus)

    def __call__(self, queue, traversal, nodal_vals, filtering=None, **kwargs):
        """
        :arg traversal
        :arg nodal_vals CL array of nodal values.
        :arg filtering Box-wide filter given by an CL array or None.

        .. warning::
           The output of this kernel is ordered in :mod:`boxtree`'s box ids.
           It may not be the same as the order implied by the input (e.g.
           box mesh generated by dealii).
        """
        filter_multiplier = _box_filter_multiplier(
            queue, filtering, self.degree**self.dim
        )

        knl = self.get_cached_optimized_kernel()
        knl_exec = knl.executor(queue.context)
        n_boxes = traversal.target_boxes.shape[0]

        _evt, res = knl_exec(
            queue,
            boxes=traversal.target_boxes,
            box_node_starts=traversal.tree.box_target_starts,
            func=nodal_vals,
            n_box_nodes=self.degree**self.dim,
            n_boxes=n_boxes,
            filter_multiplier=filter_multiplier,
            result=cl.array.zeros(queue, n_boxes, nodal_vals.dtype),
        )

        return res["result"]


# }}} End sum

# }}} End box-specific reductions

# {{{ filters for box-specific operators


def generate_leading_order_filtering(dim, n_dofs) -> np.ndarray:
    """Returns a filtering vector that is an indicator function of the node
    that corresponds to the leading order modal values in the Fourier space.
    """
    mask1d = np.zeros(n_dofs)
    mask1d[-1] = 1

    if dim == 1:
        return mask1d

    elif dim == 2:
        return (
            mask1d[:, None] + mask1d[None, :] - mask1d[:, None] * mask1d[None, :]
        ).reshape(-1)

    elif dim == 3:
        return (
            mask1d[:, None, None]
            + mask1d[None, :, None]
            + mask1d[None, None, :]
            - mask1d[:, None, None] * mask1d[None, :, None]
            - mask1d[:, None, None] * mask1d[None, None, :]
            - mask1d[None, :, None] * mask1d[None, None, :]
            + mask1d[:, None, None] * mask1d[None, :, None] * mask1d[None, None, :]
        ).reshape(-1)

    else:
        raise NotImplementedError(f"Dimension {dim} not supported")


# }}} End filters for box-specific operators


__all__ = [
    "BoxSpecificMap",
    "BoxSpecificReduction",
    "BoxSum",
    "DiscreteLegendreTransform",
    "InverseDiscreteLegendreTransform",
    "generate_leading_order_filtering",
]

# vim: filetype=pyopencl.python:fdm=marker
