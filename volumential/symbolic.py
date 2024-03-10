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

import numpy as np
import loopy as lp
import pymbolic as pmbl
from volumential.tools import ScalarFieldExpressionEvaluation

# {{{ math functions

CL_MATH_URL = r"https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/mathFunctions.html"  # noqa

CL_MATH_FUNCS = [
        'acos',     'acosh',     'acospi',  'asin',
        'asinh',    'asinpi',    'atan',    'atan2',
        'atanh',    'atanpi',    'atan2pi', 'cbrt',
        'ceil',     'copysign',  'cos',     'cosh',
        'cospi',    'erfc',      'erf',     'exp',
        'exp2',     'exp10',     'expm1',   'fabs',
        'fdim',     'floor',     'fma',     'fmax',
        'fmin',     'fmod',      'fract',   'frexp',
        'hypot',    'ilogb',     'ldexp',   'lgamma',
        'lgamma_r', 'log',       'log2',    'log10',
        'log1p',    'logb',      'mad',     'modf',
        'nan',      'nextafter', 'pow',     'pown',
        'powr',     'remainder', 'remquo',  'rint',
        'rootn',    'round',     'rsqrt',   'sin',
        'sincos',   'sinh',      'sinpi',   'sqrt',
        'tan',      'tanh',      'tanpi',   'tgamma',
        'trunc',
        ]

clmath_decl_code = r"""
def FUNC_NAME(x):
    "CL math function FUNC_NAME.\n\nSee CL_MATH_URL for details."
    return pmbl.primitives.Call(
            pmbl.primitives.Lookup(pmbl.primitives.Variable("math"), "FUNC_NAME"),
            (x,))
"""

for fname in CL_MATH_FUNCS:
    code = clmath_decl_code.replace(
            'FUNC_NAME', fname).replace(
                    'CL_MATH_URL', CL_MATH_URL)
    exec(code)

# }}} End math functions

x = pmbl.var("x")
y = pmbl.var("y")
z = pmbl.var("z")


def der_laplacian(func, coord_vars=["x", "y", "z"]):
    return sum(pmbl.diff(pmbl.diff(func, var), var)
            for var in coord_vars)

# {{{ evaluation helper


def math_func_mangler(target, name, arg_dtypes):
    """Magic function that is necessary for evaluating math functions
    """
    if len(arg_dtypes) == 1 and isinstance(name, pmbl.primitives.Lookup):
        arg_dtype, = arg_dtypes

        fname = name.name
        if not (isinstance(name.aggregate, pmbl.primitives.Variable)
                and name.aggregate.name == 'math'):
            raise RuntimeError("unexpected aggregate '%s'" %
                    str(name.aggregate))

        if arg_dtype.is_complex():
            if arg_dtype.numpy_dtype == np.complex64:
                tpname = "cfloat"
            elif arg_dtype.numpy_dtype == np.complex128:
                tpname = "cdouble"
            else:
                raise RuntimeError("unexpected complex type '%s'" %
                        arg_dtype)

            return lp.CallMangleInfo(
                   target_name=f"{tpname}_{fname}",
                   result_dtypes=(arg_dtype,),
                   arg_dtypes=(arg_dtype,))

        else:
            return lp.CallMangleInfo(
                   target_name="%s" % fname,
                   result_dtypes=(arg_dtype,),
                   arg_dtypes=(arg_dtype,))

    return None


def get_evaluator(dim, expression, variables=None):
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
            function_manglers=[math_func_mangler])

# }}} End evaluation helper
