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

import os
import time
import pyopencl as cl
import pyopencl.clrandom
import numpy as np
import loopy as lp
import pymbolic as pmbl
import pymbolic.primitives as primitives
from pymbolic.functions import sin, cos, exp, tan, log
from volumential.tools import ScalarFieldExpressionEvaluation

# {{{ math functions

# cf. https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/mathFunctions.html

def sqrt(x):
    return primitives.Call(
            primitives.Lookup(primitives.Variable("math"), "sqrt"), (x,))


def acos(x):
    return primitives.Call(
            primitives.Lookup(primitives.Variable("math"), "acos"), (x,))


def acosh(x):
    return primitives.Call(
            primitives.Lookup(primitives.Variable("math"), "acosh"), (x,))


# }}} End math functions

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
                   target_name="%s_%s" % (tpname, fname),
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
            variables=[x, y],
            function_manglers=[math_func_mangler])

# }}} End evaluation helper

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

ncpus = os.cpu_count()
hyperthreading = True
if hyperthreading:
    ncpus //= 2

x = pmbl.var("x0")
y = pmbl.var("x1")
z = pmbl.var("x2")

expr = x**2 + y * z + sin(z + x * 10.)

source_eval = get_evaluator(dim=3, expression=expr)

knl = source_eval.get_kernel()

knl_l = lp.split_iname(
        knl, split_iname="itgt", inner_length=ncpus,
        inner_tag="l.0")

knl_g = lp.split_iname(
        knl, split_iname="itgt", inner_length=ncpus,
        inner_tag="g.0")

pts = cl.clrandom.rand(queue, (3, 10**8), dtype=np.float64)

# tagged with local indices
queue.finish()
t0 = time.clock()
evt, vals = knl_l(queue, target_points=pts)
queue.finish()
t1 = time.clock()

# tagged with group indices
queue.finish()
t2 = time.clock()
evt, vals = knl_g(queue, target_points=pts)
queue.finish()
t3 = time.clock()

print("Tests run with %d threads." % ncpus)
print("With tag l.0:", t1 - t0)
print("With tag g.0:", t3 - t2)
