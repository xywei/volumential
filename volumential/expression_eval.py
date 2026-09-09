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

__doc__ = """Pointwise evaluation of symbolic scalar fields on the device.

This module owns :class:`ScalarFieldExpressionEvaluation` together with the
:mod:`pymbolic`/:mod:`loopy` plumbing that turns ``math.foo(x)`` lookups from a
symbolic expression into callables the OpenCL target understands.  It is
re-exported by :mod:`volumential.tools` for backwards compatibility.

.. autoclass:: ScalarFieldExpressionEvaluation
"""

import logging
from typing import Any

import numpy as np
from constantdict import constantdict

import loopy as lp
import pymbolic as pmbl
import pyopencl as cl
import pyopencl.array
from pymbolic.mapper import IdentityMapper, WalkMapper
from pymbolic.primitives import (
    ExpressionNode as ExpressionType,
    Variable as VariableType,
)

from volumential.kernel_cache import KernelCacheWrapper


logger = logging.getLogger(__name__)


# {{{ pymbolic function-call plumbing


class _MathLookupToBareCallMapper(IdentityMapper):
    """Rewrite ``math.foo(x)`` calls into bare ``foo(x)`` calls."""

    def map_call(self, expr):
        function = self.rec(expr.function)
        parameters = tuple(self.rec(par) for par in expr.parameters)

        if (
            isinstance(function, pmbl.primitives.Lookup)
            and isinstance(function.aggregate, pmbl.primitives.Variable)
            and function.aggregate.name == "math"
        ):
            function = pmbl.var(function.name)

        return pmbl.primitives.Call(function, parameters)


class _CallNameCollector(WalkMapper):
    """Collect the names of all functions called by an expression."""

    def __init__(self) -> None:
        super().__init__()
        self.names: set[str] = set()

    def map_call(self, expr):
        if isinstance(expr.function, pmbl.primitives.Variable):
            self.names.add(expr.function.name)

        return super().map_call(expr)


class _FunctionManglerCallable(lp.ScalarCallable):
    """Adapt legacy :mod:`loopy` function manglers to the callables interface."""

    def __init__(self, name, function_manglers):
        super().__init__(name=name, name_in_target=name)
        self.function_manglers = function_manglers

    def with_types(self, arg_id_to_dtype, clbl_inf_ctx):
        arg_num_to_dtype = {
            id: dtype for id, dtype in arg_id_to_dtype.items() if id >= 0
        }

        # wait for full type information
        if not arg_num_to_dtype or any(
            dtype is None for dtype in arg_num_to_dtype.values()
        ):
            return self.copy(
                arg_id_to_dtype=constantdict(arg_id_to_dtype)
            ), clbl_inf_ctx

        n_args = max(arg_num_to_dtype) + 1
        if any(i not in arg_num_to_dtype for i in range(n_args)):
            return self.copy(
                arg_id_to_dtype=constantdict(arg_id_to_dtype)
            ), clbl_inf_ctx

        arg_dtypes = tuple(arg_num_to_dtype[i] for i in range(n_args))

        for mangler in self.function_manglers:
            mangle_info = mangler(None, pmbl.var(self.name), arg_dtypes)
            if mangle_info is None:
                mangle_info = mangler(
                    None,
                    pmbl.primitives.Lookup(pmbl.var("math"), self.name),
                    arg_dtypes,
                )

            if mangle_info is None:
                continue

            updated_arg_id_to_dtype = dict(enumerate(mangle_info.arg_dtypes))
            updated_arg_id_to_dtype[-1] = mangle_info.result_dtypes[0]

            return (
                self.copy(
                    name_in_target=mangle_info.target_name,
                    arg_id_to_dtype=constantdict(updated_arg_id_to_dtype),
                ),
                clbl_inf_ctx,
            )

        # Fallback: assume return type matches the first argument.
        fallback_arg_id_to_dtype = dict(arg_num_to_dtype)
        fallback_arg_id_to_dtype[-1] = arg_num_to_dtype[0]
        return (
            self.copy(arg_id_to_dtype=constantdict(fallback_arg_id_to_dtype)),
            clbl_inf_ctx,
        )


def _collect_called_function_names(expr) -> set[str]:
    """Return the set of function names called inside *expr*."""
    collector = _CallNameCollector()
    collector(expr)
    return collector.names


# }}} End pymbolic function-call plumbing


# {{{ scalar field expression eval


class ScalarFieldExpressionEvaluation(KernelCacheWrapper):
    """
    Evaluate a field function on a set of D-d points.
    Useful for imposing analytic conditions efficiently.
    """

    def __init__(
        self,
        dim,
        expression,
        variables=None,
        dtype=np.float64,
        function_manglers=None,
        preamble_generators=None,
    ):
        """
        :arg dim
        :arg expression A pymbolic expression for the function
        :arg variables A list of variables representing spacial coordinates
        """
        assert dim > 0
        self.dim = dim

        sympy_to_pymbolic = None

        def _to_pymbolic(expr):
            nonlocal sympy_to_pymbolic
            if isinstance(expr, ExpressionType | int | float | complex):
                return expr

            if sympy_to_pymbolic is None:
                from pymbolic.interop.sympy import SympyToPymbolicMapper

                sympy_to_pymbolic = SympyToPymbolicMapper()

            return sympy_to_pymbolic(expr)

        self.expr = _to_pymbolic(expression)

        if variables is None:
            self.vars = [pmbl.var(f"x{d}") for d in range(self.dim)]
        else:
            assert isinstance(variables, list)
            self.vars = [
                var if isinstance(var, VariableType) else _to_pymbolic(var)
                for var in variables
            ]

        self.dtype = dtype
        self.function_manglers = function_manglers
        self.preamble_generators = preamble_generators

        self.name = "ScalarFieldExpressionEvaluation"

    def get_cache_key(self) -> tuple[Any, ...]:
        """Return a hashable key identifying the generated kernel."""
        return (
            type(self).__name__,
            f"{self.dim}D",
            str(self.expr),
            ",".join([str(x) for x in self.vars]),
            repr(self.function_manglers),
        )

    def _apply_function_manglers(self, loopy_knl):
        if self.function_manglers is None:
            return loopy_knl

        if hasattr(lp, "register_function_manglers"):
            return lp.register_function_manglers(loopy_knl, self.function_manglers)

        from loopy.target.opencl import get_opencl_callables

        known_callables = set(get_opencl_callables().keys())
        call_names = _collect_called_function_names(self.get_normalised_expr())
        for name in sorted(call_names - known_callables):
            loopy_knl = lp.register_callable(
                loopy_knl,
                name,
                _FunctionManglerCallable(name, self.function_manglers),
            )

        return loopy_knl

    def get_normalised_expr(self):
        """Return :attr:`expr` rewritten in terms of the ``x0, x1, ...`` names
        used by the generated kernel.
        """
        nexpr = self.expr
        nvars = [pmbl.var(f"x{d}") for d in range(self.dim)]
        # NOTE: strict=False preserves the historical behavior of substituting
        # only as many variables as the caller supplied.
        for var, nvar in zip(self.vars, nvars, strict=False):
            nexpr = pmbl.substitute(nexpr, {var: nvar})

        return _MathLookupToBareCallMapper()(nexpr)

    def get_variable_assignment_code(self) -> str:
        """Return the loopy snippet reading target coordinates into ``x0, ...``."""
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
        """Return the pointwise evaluation kernel."""
        extra_kernel_kwarg_types = ()
        if "extra_kernel_kwarg_types" in kwargs:
            extra_kernel_kwarg_types = kwargs["extra_kernel_kwarg_types"]

        eval_inames = frozenset(["itgt"])
        scalar_assignment = lp.Assignment(
            id=None,
            assignee="expr_val",
            expression=self.get_normalised_expr(),
            temp_var_type=lp.Optional(),
        )
        eval_insns = [
            insn.copy(within_inames=insn.within_inames | eval_inames)
            for insn in [scalar_assignment]
        ]

        loopy_knl = lp.make_kernel(
            "{ [itgt]: 0<=itgt<n_targets }",
            [
                """
                for itgt
                    VAR_ASSIGNMENT
                end
                """.replace("VAR_ASSIGNMENT", self.get_variable_assignment_code()),
                *eval_insns,
                """
                for itgt
                    result[itgt] = expr_val
                end
                """,
            ],
            [
                lp.ValueArg("dim, n_targets", np.int32),
                lp.GlobalArg("target_points", np.float64, "dim, n_targets"),
                lp.TemporaryVariable("expr_val", None, ()),
                *extra_kernel_kwarg_types,
                "...",
            ],
            name="eval_expr",
            lang_version=(2018, 2),
        )

        loopy_knl = lp.fix_parameters(loopy_knl, dim=self.dim)
        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        loopy_knl = self._apply_function_manglers(loopy_knl)

        if self.preamble_generators is not None:
            loopy_knl = lp.register_preamble_generators(
                loopy_knl, self.preamble_generators
            )

        return loopy_knl

    def get_optimized_kernel(self, ncpus=None, **kwargs):
        """Return the evaluation kernel parallelized over target points."""
        knl = self.get_kernel(**kwargs)
        if ncpus is None:
            import multiprocessing

            # NOTE: this detects the number of logical cores, which
            # may result in suboptimal performance.
            ncpus = multiprocessing.cpu_count()
        knl = lp.split_iname(
            knl, split_iname="itgt", inner_length=ncpus, inner_tag="g.0"
        )
        return knl

    def __call__(self, queue, target_points, **kwargs):
        """
        :arg target_points
        :arg extra_kernel_kwargs
        """
        # handle target_points given as an obj_array of coords
        if (
            isinstance(target_points, np.ndarray)
            and target_points.dtype == object
            and isinstance(target_points[0], cl.array.Array)
        ):
            target_points = cl.array.concatenate(target_points).reshape(
                [self.dim, -1]
            )

        assert target_points.shape[0] == self.dim

        n_tgt_points = target_points[0].shape[0]
        for tgt_d in target_points:
            assert len(tgt_d) == n_tgt_points

        extra_kernel_kwargs = {}
        if "extra_kernel_kwargs" in kwargs:
            extra_kernel_kwargs = kwargs["extra_kernel_kwargs"]

        knl = self.get_cached_optimized_kernel()

        if self.preamble_generators is not None:
            knl = lp.register_preamble_generators(knl, self.preamble_generators)

        knl_exec = knl.executor(queue.context)

        _evt, res = knl_exec(
            queue,
            target_points=target_points,
            n_targets=n_tgt_points,
            result=np.zeros(n_tgt_points, dtype=self.dtype),
            **extra_kernel_kwargs,
        )

        return res["result"]


# }}} End scalar field expression eval


__all__ = ["ScalarFieldExpressionEvaluation"]

# vim: filetype=pyopencl.python:fdm=marker
