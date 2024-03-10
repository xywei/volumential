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
import warnings
import pyopencl as cl
import pyopencl.clrandom
import numpy as np
import loopy as lp
import pymbolic as pmbl
import pymbolic.primitives as primitives
from pymbolic.functions import sin, cos, exp, tan, log  # noqa: F401
from volumential.tools import ScalarFieldExpressionEvaluation

# {{{ math functions

# cf.
# https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/mathFunctions.html


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
            variables=[x, y],
            function_manglers=[math_func_mangler])

# }}} End evaluation helper


# Hide warnings
warnings.filterwarnings("ignore")

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

pts = cl.clrandom.rand(queue, (3, 10**8), dtype=np.float64)

knl = source_eval.get_kernel()

# needed for using loopy.statistics
knl = lp.add_and_infer_dtypes(
        knl,
        dict(x0=np.float64, x1=np.float64, x2=np.float64))
knl = lp.set_options(knl, ignore_boostable_into=True)

# {{{ wall time

knl_l = lp.split_iname(
        knl, split_iname="itgt", inner_length=ncpus,
        inner_tag="l.0")

knl_g = lp.split_iname(
        knl, split_iname="itgt", inner_length=ncpus,
        inner_tag="g.0")

# tagged with local indices
queue.finish()
t0 = time.time()
evt, vals = knl_l(queue, target_points=pts)
queue.finish()
t1 = time.time()

# tagged with group indices
queue.finish()
t2 = time.time()
evt, vals = knl_g(queue, target_points=pts)
queue.finish()
t3 = time.time()

print("Tests run with %d threads." % ncpus)
print("Wall time w/t tag l.0:", t1 - t0)
print("Wall time w/t tag g.0:", t3 - t2)

# }}} End wall time

# {{{ operation counts

# count the total work
op_map = lp.get_op_map(knl, subgroup_size=ncpus, count_redundant_work=True,
        count_within_subscripts=True)

params = dict(n_targets=pts.shape[1])
print('Operation counts:')
total_ops = 0
for op in op_map.keys():
    sub_count = op_map[op].eval_with_dict(params)
    total_ops += sub_count
    print('\t', op.name, op_map[op], sub_count)
print("Total:", total_ops)

# TODO: weight each operation by running micro-benchmarks
print("OP throughput w/t tag l.0 = %.2f GFLOPS/S" %
      (total_ops / (t1 - t0) * 1e-9))
print("OP throughput w/t tag g.0 = %.2f GFLOPS/S" %
      (total_ops / (t3 - t2) * 1e-9))

# }}} End operation counts

# {{{ mem access counts

# FIXME: warnings or write race on expr_val
# mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
#                                subgroup_size=ncpus)

# }}} End mem access counts
