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
import pymbolic as pmbl
from pymbolic.primitives import Variable as VariableType
from pymbolic.primitives import Expression as ExpressionType
from sumpy.tools import KernelCacheWrapper


def clean_file(filename, new_name=None):
    """Remove/rename file if exists.
    Fails silently when the file does not exist.
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


class ScalarFieldExpressionEvaluation(KernelCacheWrapper):
    """
    Evaluate a field funciton on a set of D-d points.
    """

    def __init__(self, dim, expression, variables=None, function_manglers=None):
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

        self.function_manglers = function_manglers

        self.name = "ScalarFieldExpressionEvaluation"

    def get_cache_key(self):
        return (type(self).__name__, str(self.dim) + "D", self.expr.__str__(),
                ','.join([x.__str__() for x in self.vars]))

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
        if 'extra_kernel_kwarg_types' in kwargs:
            extra_kernel_kwarg_types = kwargs['extra_kernel_kwarg_types']

        eval_inames = frozenset(["itgt"])
        scalar_assignment = lp.Assignment(
            id=None,
            assignee="expr_val",
            expression=self.get_normalised_expr(),
            temp_var_type=lp.auto)
        eval_insns = [
            insn.copy(within_inames=insn.within_inames | eval_inames)
            for insn in [scalar_assignment]
        ]

        loopy_knl = lp.make_kernel(  # NOQA
            "{ [itgt]: 0<=itgt<n_targets }", [
                """
                for itgt
                    VAR_ASSIGNMENT
                end
                """.replace("VAR_ASSIGNMENT", self.get_variable_assignment_code())
            ] + eval_insns + [
                """
                for itgt
                    result[itgt] = expr_val
                end
                """
            ], [
                lp.ValueArg("dim, n_targets", np.int32),
                lp.GlobalArg("target_points", np.float64, "dim, n_targets"),
                *extra_kernel_kwarg_types, "..."
            ],
            name="eval_expr")

        loopy_knl = lp.fix_parameters(loopy_knl, dim=self.dim)
        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        if self.function_manglers is not None:
            loopy_knl = lp.register_function_manglers(loopy_knl,
                                                      self.function_manglers)

        return loopy_knl

    def get_optimized_kernel(self, **kwargs):
        knl = self.get_kernel(**kwargs)
        knl = lp.split_iname(knl, "itgt", 16, outer_tag="g.0", inner_tag="l.0")
        return knl

    def __call__(self, queue, target_points, **kwargs):
        """
        :arg target_points
        :arg extra_kernel_kwargs
        """

        assert len(target_points) == self.dim
        n_tgt_points = len(target_points[0])
        for tgt_d in target_points:
            assert len(tgt_d) == n_tgt_points

        extra_kernel_kwargs = {}
        if 'extra_kernel_kwargs' in kwargs:
            extra_kernel_kwargs = kwargs['extra_kernel_kwargs']

        knl = self.get_cached_optimized_kernel()
        evt, res = knl(
            queue,
            target_points=target_points,
            n_targets=n_tgt_points,
            result=np.zeros(n_tgt_points),
            **extra_kernel_kwargs)

        return res['result']
