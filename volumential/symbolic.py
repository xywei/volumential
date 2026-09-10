__copyright__ = "Copyright (C) 2019 Xiaoyu Wei"

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

__doc__ = """Symbolic (:mod:`pymbolic`) building blocks for analytic fields.

This module owns

* one wrapper per OpenCL math function (:data:`CL_MATH_FUNCS`), each producing
  a ``math.<name>(x)`` :mod:`pymbolic` call node,
* the coordinate variables :data:`x`, :data:`y`, :data:`z`,
* :func:`der_laplacian` for symbolic Laplacians, and
* :func:`math_func_mangler` / :func:`get_evaluator`, which connect such
  expressions to :class:`volumential.tools.ScalarFieldExpressionEvaluation`.

.. autofunction:: der_laplacian
.. autofunction:: math_func_mangler
.. autofunction:: get_evaluator
"""

from collections.abc import Callable, Sequence

import numpy as np

import loopy as lp
import pymbolic as pmbl

from volumential.tools import ScalarFieldExpressionEvaluation


# {{{ math functions

CL_MATH_URL = (
    "https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/"
    "mathFunctions.html"
)

CL_MATH_FUNCS = [
    "acos",
    "acosh",
    "acospi",
    "asin",
    "asinh",
    "asinpi",
    "atan",
    "atan2",
    "atanh",
    "atanpi",
    "atan2pi",
    "cbrt",
    "ceil",
    "copysign",
    "cos",
    "cosh",
    "cospi",
    "erfc",
    "erf",
    "exp",
    "exp2",
    "exp10",
    "expm1",
    "fabs",
    "fdim",
    "floor",
    "fma",
    "fmax",
    "fmin",
    "fmod",
    "fract",
    "frexp",
    "hypot",
    "ilogb",
    "ldexp",
    "lgamma",
    "lgamma_r",
    "log",
    "log2",
    "log10",
    "log1p",
    "logb",
    "mad",
    "modf",
    "nan",
    "nextafter",
    "pow",
    "pown",
    "powr",
    "remainder",
    "remquo",
    "rint",
    "rootn",
    "round",
    "rsqrt",
    "sin",
    "sincos",
    "sinh",
    "sinpi",
    "sqrt",
    "tan",
    "tanh",
    "tanpi",
    "tgamma",
    "trunc",
]


def _make_cl_math_func(fname: str) -> Callable[[object], pmbl.primitives.Call]:
    """Build a wrapper emitting a ``math.<fname>(x)`` :mod:`pymbolic` call."""

    def cl_math_func(x):
        return pmbl.primitives.Call(
            pmbl.primitives.Lookup(pmbl.primitives.Variable("math"), fname),
            (x,),
        )

    cl_math_func.__name__ = fname
    cl_math_func.__qualname__ = fname
    cl_math_func.__doc__ = (
        f"CL math function {fname}.\n\nSee {CL_MATH_URL} for details."
    )
    return cl_math_func


# One module-level wrapper per OpenCL math function, e.g. ``symbolic.sin``.
for fname in CL_MATH_FUNCS:
    globals()[fname] = _make_cl_math_func(fname)
del fname

# }}} End math functions

x = pmbl.var("x")
y = pmbl.var("y")
z = pmbl.var("z")


def der_laplacian(func, coord_vars: Sequence[str] | None = None):
    """Return the symbolic Laplacian of *func* in the given coordinates.

    :arg func: a :mod:`pymbolic` expression.
    :arg coord_vars: names of the coordinate variables, ``("x", "y", "z")``
        by default.
    """
    if coord_vars is None:
        coord_vars = ["x", "y", "z"]

    return sum(pmbl.diff(pmbl.diff(func, var), var) for var in coord_vars)


# {{{ evaluation helper


def math_func_mangler(target, name, arg_dtypes):
    """Return :mod:`loopy` call-mangling info for ``math.<name>`` lookups.

    :returns: a :class:`loopy.CallMangleInfo`, or *None* if *name* is not a
        single-argument ``math`` lookup.
    """
    if len(arg_dtypes) == 1 and isinstance(name, pmbl.primitives.Lookup):
        (arg_dtype,) = arg_dtypes

        fname = name.name
        if not (
            isinstance(name.aggregate, pmbl.primitives.Variable)
            and name.aggregate.name == "math"
        ):
            raise RuntimeError(f"unexpected aggregate '{name.aggregate}'")

        if arg_dtype.is_complex():
            if arg_dtype.numpy_dtype == np.complex64:
                tpname = "cfloat"
            elif arg_dtype.numpy_dtype == np.complex128:
                tpname = "cdouble"
            else:
                raise RuntimeError(f"unexpected complex type '{arg_dtype}'")

            return lp.CallMangleInfo(
                target_name=f"{tpname}_{fname}",
                result_dtypes=(arg_dtype,),
                arg_dtypes=(arg_dtype,),
            )

        return lp.CallMangleInfo(
            target_name=str(fname),
            result_dtypes=(arg_dtype,),
            arg_dtypes=(arg_dtype,),
        )

    return None


def get_evaluator(
    dim: int, expression, variables=None
) -> ScalarFieldExpressionEvaluation:
    """Return an evaluator for *expression* over *dim*-dimensional points.

    :arg expression: a :mod:`pymbolic` expression in *variables*.
    :arg variables: coordinate variables, defaulting to :data:`x`, :data:`y`
        and :data:`z` truncated to *dim*.
    """
    if variables is None:
        if dim == 1:
            variables = [x]
        elif dim == 2:
            variables = [x, y]
        elif dim == 3:
            variables = [x, y, z]
    else:
        assert len(variables) == dim

    return ScalarFieldExpressionEvaluation(
        dim=dim,
        expression=expression,
        variables=variables,
        function_manglers=[math_func_mangler],
    )


# }}} End evaluation helper

# vim: filetype=pyopencl.python:fdm=marker
