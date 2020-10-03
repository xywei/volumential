from __future__ import absolute_import, division, print_function
import six

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

import numpy as np
import loopy as lp
import pyopencl as cl
import pymbolic as pmbl
from pytools import memoize_method
from pymbolic.primitives import Variable as VariableType
from pymbolic.primitives import Expression as ExpressionType

import logging

logger = logging.getLogger(__name__)

# {{{ clean files


def clean_file(filename, new_name=None):
    """Remove/rename file if exists.
    Fails silently when the file does not exist.
    Useful for, for example, writing output files that
    are meant to overwrite existing ones.
    """
    import os

    if new_name is None:
        try:
            os.remove(filename)
        except OSError:
            pass
    else:
        try:
            os.rename(filename, new_name)
        except OSError:
            pass

# }}} End clean files

# {{{ loopy kernel cache wrapper


class KernelCacheWrapper(object):
    # FIXME: largely code duplication with sumpy.

    def __init__(self):
        self.name = "KernelCacheWrapper"
        raise RuntimeError(
                "KernelCacheWrapper objects should not be constructed")

    def get_cache_key(self):
        raise NotImplementedError("Unimplemented cache key")

    def get_kernel(self):
        raise NotImplementedError()

    def get_optimized_kernel(self):
        raise NotImplementedError()

    @memoize_method
    def get_cached_optimized_kernel(self, **kwargs):
        from sumpy import code_cache, CACHING_ENABLED, OPT_ENABLED

        if CACHING_ENABLED:
            import loopy.version
            from sumpy.version import KERNEL_VERSION as SUMPY_KERNEL_VERSION
            from volumential.version import KERNEL_VERSION
            cache_key = (
                    self.get_cache_key()
                    + tuple(sorted(six.iteritems(kwargs)))
                    + (loopy.version.DATA_MODEL_VERSION,)
                    + (SUMPY_KERNEL_VERSION,)
                    + (KERNEL_VERSION,)
                    + (OPT_ENABLED,))

            try:
                result = code_cache[cache_key]
                logger.debug("%s: kernel cache hit [key=%s]" % (
                    self.name, cache_key))
                return result
            except KeyError:
                pass

        logger.info("%s: kernel cache miss" % self.name)
        if CACHING_ENABLED:
            logger.info("%s: kernel cache miss [key=%s]" % (
                self.name, cache_key))

        from pytools import MinRecursionLimit
        with MinRecursionLimit(3000):
            if OPT_ENABLED:
                knl = self.get_optimized_kernel(**kwargs)
            else:
                knl = self.get_kernel()

        if CACHING_ENABLED:
            code_cache.store_if_not_present(cache_key, knl)

        return knl


# }}} End loopy kernel cache wrapper

# {{{ scalar field expression eval


class ScalarFieldExpressionEvaluation(KernelCacheWrapper):
    """
    Evaluate a field funciton on a set of D-d points.
    Useful for imposing analytic conditions efficiently.
    """

    def __init__(self, dim, expression, variables=None, dtype=np.float64,
                 function_manglers=None, preamble_generators=None):
        """
        :arg dim
        :arg expression A pymbolic expression for the function
        :arg variables A list of variables representing spacial coordinates
        """
        assert dim > 0
        self.dim = dim

        assert isinstance(expression, (ExpressionType, int, float, complex))
        self.expr = expression

        if variables is None:
            self.vars = [pmbl.var("x%d" % d) for d in range(self.dim)]
        else:
            assert isinstance(variables, list)
            for var in variables:
                assert isinstance(var, VariableType)
            self.vars = variables

        self.dtype = dtype
        self.function_manglers = function_manglers
        self.preamble_generators = preamble_generators

        self.name = "ScalarFieldExpressionEvaluation"

    def get_cache_key(self):
        return (
            type(self).__name__,
            str(self.dim) + "D",
            self.expr.__str__(),
            ",".join([x.__str__() for x in self.vars]),
        )

    def get_normalised_expr(self):
        nexpr = self.expr
        nvars = [pmbl.var("x%d" % d) for d in range(self.dim)]
        for var, nvar in zip(self.vars, nvars):
            nexpr = pmbl.substitute(nexpr, {var: nvar})
        return nexpr

    def get_variable_assignment_code(self):
        if self.dim == 1:
            return "<> x0 = target_points[0, itgt]"
        elif self.dim == 2:
            return """<> x0 = target_points[0, itgt]
                      <> x1 = target_points[1, itgt]"""
        elif self.dim == 3:
            return """<> x0 = target_points[0, itgt]
                      <> x1 = target_points[1, itgt]
                      <> x2 = target_points[2, itgt]"""
        else:
            raise NotImplementedError

    def get_kernel(self, **kwargs):

        extra_kernel_kwarg_types = ()
        if "extra_kernel_kwarg_types" in kwargs:
            extra_kernel_kwarg_types = kwargs["extra_kernel_kwarg_types"]

        eval_inames = frozenset(["itgt"])
        scalar_assignment = lp.Assignment(
            id=None,
            assignee="expr_val",
            expression=self.get_normalised_expr(),
            temp_var_type=None,
        )
        eval_insns = [
            insn.copy(within_inames=insn.within_inames | eval_inames)
            for insn in [scalar_assignment]
        ]

        loopy_knl = lp.make_kernel(  # NOQA
            "{ [itgt]: 0<=itgt<n_targets }",
            [
                """
                for itgt
                    VAR_ASSIGNMENT
                end
                """.replace(
                    "VAR_ASSIGNMENT", self.get_variable_assignment_code()
                )
            ]
            + eval_insns
            + [
                """
                for itgt
                    result[itgt] = expr_val
                end
                """
            ],
            [
                lp.ValueArg("dim, n_targets", np.int32),
                lp.GlobalArg("target_points", np.float64, "dim, n_targets"),
                lp.TemporaryVariable("expr_val", None, ()),
            ]
            + list(extra_kernel_kwarg_types)
            + ["...", ],
            name="eval_expr",
            lang_version=(2018, 2),
        )

        loopy_knl = lp.fix_parameters(loopy_knl, dim=self.dim)
        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        if self.function_manglers is not None:
            loopy_knl = lp.register_function_manglers(
                loopy_knl,
                self.function_manglers)

        if self.preamble_generators is not None:
            loopy_knl = lp.register_preamble_generators(
                loopy_knl,
                self.preamble_generators)

        return loopy_knl

    def get_optimized_kernel(self, ncpus=None, **kwargs):
        knl = self.get_kernel(**kwargs)
        if ncpus is None:
            import multiprocessing
            # NOTE: this detects the number of logical cores, which
            # may result in suboptimal performance.
            ncpus = multiprocessing.cpu_count()
        knl = lp.split_iname(
            knl,
            split_iname="itgt",
            inner_length=ncpus,
            inner_tag="g.0")
        return knl

    def __call__(self, queue, target_points, **kwargs):
        """
        :arg target_points
        :arg extra_kernel_kwargs
        """
        # handle target_points given as an obj_array of coords
        if (isinstance(target_points, np.ndarray)
                and target_points.dtype == np.object
                and isinstance(target_points[0], cl.array.Array)):
            target_points = cl.array.concatenate(
                    target_points).reshape([self.dim, -1])

        assert target_points.shape[0] == self.dim

        n_tgt_points = target_points[0].shape[0]
        for tgt_d in target_points:
            assert len(tgt_d) == n_tgt_points

        extra_kernel_kwargs = {}
        if "extra_kernel_kwargs" in kwargs:
            extra_kernel_kwargs = kwargs["extra_kernel_kwargs"]

        knl = self.get_cached_optimized_kernel()

        # FIXME: caching loses function mangler information
        if self.function_manglers is not None:
            knl = lp.register_function_manglers(knl, self.function_manglers)

        if self.preamble_generators is not None:
            knl = lp.register_preamble_generators(knl, self.preamble_generators)

        evt, res = knl(
            queue,
            target_points=target_points,
            n_targets=n_tgt_points,
            result=np.zeros(n_tgt_points, dtype=self.dtype),
            **extra_kernel_kwargs
        )

        return res["result"]

# }}} End scalar field expression eval

# {{{ import code


def import_code(code, name, add_to_sys_modules=True):
    """Dynamically generates a module.

    :arg code: can be any object containing code -- string, file object, or
    compiled code object. Returns a new module object initialized
    by dynamically importing the given code and optionally adds it
    to sys.modules under the given name.
    """
    import imp
    module = imp.new_module(name)

    if add_to_sys_modules:
        import sys
        sys.modules[name] = module

    exec(code, module.__dict__)

    return module

# }}} End import code

# {{{ box-specific maps


class BoxSpecificMap(KernelCacheWrapper):
    """
    Box-specific transform that maps between datum defined on quadrature
    nodes. Being box-specific means that the transform for each box is
    independent from the rest of the boxes.
    """
    pass

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
        self.template_interval = [-1., 1.]
        self.template_interval_extent = 2.
        self.template_interval_center = 0.

        self.leg_tplt_x, self.leg_tplt_w = \
                np.polynomial.legendre.leggauss(degree)

        if self.dim == 1:
            self.V = np.polynomial.legendre.legvander(
                    self.leg_tplt_x, self.degree - 1)
            self.W = self.leg_tplt_w.reshape(-1)

        elif self.dim == 2:
            x, y = np.meshgrid(self.leg_tplt_x, self.leg_tplt_x)
            self.V = np.polynomial.legendre.legvander2d(
                    x.reshape(-1), y.reshape(-1),
                    [self.degree - 1] * self.dim)
            self.W = (self.leg_tplt_w[None, :]
                    * self.leg_tplt_w[:, None]).reshape(-1)

        elif self.dim == 3:
            x, y, z = np.meshgrid(self.leg_tplt_x, self.leg_tplt_x, self.leg_tplt_x)
            self.V = np.polynomial.legendre.legvander3d(
                    x.reshape(-1), y.reshape(-1), z.reshape(-1),
                    [self.degree - 1] * self.dim)
            self.W = (self.leg_tplt_w[:, None, None]
                    * self.leg_tplt_w[None, :, None]
                    * self.leg_tplt_w[None, None, :]).reshape(-1)

        else:
            raise NotImplementedError("Dimension %d is not supported" % self.dim)

        # Vandermonde matrix: each column corresponds to one basis function
        assert self.V.shape == (self.degree**self.dim, self.degree**self.dim)
        assert self.W.shape == (self.degree**self.dim,)

        # Normalizers
        self.I = np.ascontiguousarray(  # noqa: E741
                np.diag(
                    (self.V.T * self.W) @ self.V))
        assert self.I.shape == (self.degree**self.dim,)

        # Fix strides for loopy
        self.V = np.ascontiguousarray(self.V)

        # Check orthogonality
        ortho_resid = np.linalg.norm(
                self.V.T * np.matmul(self.W, self.V) - np.diag(self.I))
        if ortho_resid > 1e-13:
            logger.warn("Legendre polynomials' orthogonality residual = %f"
                    % ortho_resid)

        self.name = "DiscreteLegendreTransform"

    def get_cache_key(self):
        return (
            type(self).__name__,
            str(self.dim) + "D",
            "degree=%d" % self.degree
        )

    def get_kernel(self, **kwargs):

        loopy_knl = lp.make_kernel(  # NOQA
                [
                    "{ [ bid ] : 0 <= bid < n_boxes }",
                    "{ [ mid ] : 0 <= mid < n_box_nodes }",
                    "{ [ nid ] : 0 <= nid < n_box_nodes }"
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
                    lp.GlobalArg("weight, normalizer, filter_multiplier",
                        np.float64, "n_box_nodes"),
                    lp.GlobalArg("vandermonde", np.float64,
                        "n_box_nodes, n_box_nodes"),
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
        knl = self.get_kernel(**kwargs)
        if ncpus is None:
            import multiprocessing
            ncpus = multiprocessing.cpu_count()
        knl = lp.split_iname(
            knl,
            split_iname="bid",
            inner_length=ncpus,
            inner_tag="g.0")
        return knl

    def __call__(self, queue, traversal, nodal_vals, filtering=None, **kwargs):
        """
        :arg traversal
        :arg nodal_vals CL array of nodal values.
        :arg filtering Box-wide filter given by an CL array or None.
        """

        if filtering is None:
            filter_multiplier = 1 + cl.array.zeros(
                queue, self.degree**self.dim, np.float64)
        elif isinstance(filtering, cl.array.Array):
            assert filtering.shape == (self.degree**self.dim,)
            filter_multiplier = filtering
        else:
            raise RuntimeError(f"Invalid filtering argument: {str(filtering)}")

        knl = self.get_cached_optimized_kernel()

        evt, res = knl(
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
    pass

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
    pass

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

    def get_cache_key(self):
        return (
            type(self).__name__,
            str(self.dim) + "D",
            "degree=%d" % self.degree
        )

    def get_kernel(self, **kwargs):

        loopy_knl = lp.make_kernel(  # NOQA
                [
                    "{ [ bid ] : 0 <= bid < n_boxes }",
                    "{ [ nid ] : 0 <= nid < n_box_nodes }"
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
        knl = self.get_kernel(**kwargs)
        if ncpus is None:
            import multiprocessing
            ncpus = multiprocessing.cpu_count()
        knl = lp.split_iname(
            knl,
            split_iname="bid",
            inner_length=ncpus,
            inner_tag="g.0")
        return knl

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

        if filtering is None:
            filter_multiplier = 1 + cl.array.zeros(queue,
                    self.degree**self.dim, np.float64)
        elif isinstance(filtering, cl.array.Array):
            assert filtering.shape == (self.degree**self.dim,)
            filter_multiplier = filtering
        else:
            raise RuntimeError("Invalid filtering argument: %s"
                    % str(filtering))

        knl = self.get_cached_optimized_kernel()
        n_boxes = traversal.target_boxes.shape[0]

        evt, res = knl(
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


def generate_leading_order_filtering(dim, n_dofs):
    """Returns a filtering vector that is an indicator function of the node
    that corresponds to the leading order modal values in the Fourier space.
    """

    mask1d = np.zeros(n_dofs)
    mask1d[-1] = 1

    if dim == 1:
        return mask1d

    elif dim == 2:
        return (
            mask1d[:, None] + mask1d[None, :]
            - mask1d[:, None] * mask1d[None, :]
        ).reshape(-1)

    elif dim == 3:
        return (
            mask1d[:, None, None] + mask1d[None, :, None] + mask1d[None, None, :]
            - mask1d[:, None, None] * mask1d[None, :, None]
            - mask1d[:, None, None] * mask1d[None, None, :]
            - mask1d[None, :, None] * mask1d[None, None, :]
            + mask1d[:, None, None] * mask1d[None, :, None] * mask1d[None, None, :]
        ).reshape(-1)

    else:
        raise NotImplementedError("Dimension %d not supported" % dim)

# }}} End filters for box-specific operators
