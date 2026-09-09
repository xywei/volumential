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

import numpy as np
import pytest

import pymbolic as pmbl
from loopy.types import NumpyType

from volumential import symbolic


def test_cl_math_wrappers_are_defined_for_every_listed_function():
    for name in symbolic.CL_MATH_FUNCS:
        wrapper = getattr(symbolic, name)
        assert callable(wrapper)
        assert wrapper.__name__ == name
        assert name in wrapper.__doc__


def test_cl_math_wrapper_builds_math_lookup_call():
    call = symbolic.sin(symbolic.x)

    assert isinstance(call, pmbl.primitives.Call)
    assert isinstance(call.function, pmbl.primitives.Lookup)
    assert call.function.aggregate == pmbl.primitives.Variable("math")
    assert call.function.name == "sin"
    assert call.parameters == (symbolic.x,)


def test_coordinate_variables():
    assert symbolic.x == pmbl.var("x")
    assert symbolic.y == pmbl.var("y")
    assert symbolic.z == pmbl.var("z")


def test_der_laplacian_of_quadratic_is_constant():
    expr = symbolic.x**2 + symbolic.y**2 + symbolic.z**2

    assert symbolic.der_laplacian(expr) == 6


def test_der_laplacian_respects_coordinate_subset():
    expr = symbolic.x**2 + symbolic.y**2 + symbolic.z**2

    assert symbolic.der_laplacian(expr, coord_vars=["x", "y"]) == 4


def test_math_func_mangler_maps_real_call_to_bare_target():
    dtype = NumpyType(np.dtype(np.float64))
    name = pmbl.primitives.Lookup(pmbl.primitives.Variable("math"), "exp")

    info = symbolic.math_func_mangler(None, name, (dtype,))

    assert info.target_name == "exp"
    assert info.result_dtypes == (dtype,)
    assert info.arg_dtypes == (dtype,)


@pytest.mark.parametrize(
    ("numpy_dtype", "prefix"),
    [(np.complex64, "cfloat"), (np.complex128, "cdouble")],
)
def test_math_func_mangler_maps_complex_call_to_prefixed_target(numpy_dtype, prefix):
    dtype = NumpyType(np.dtype(numpy_dtype))
    name = pmbl.primitives.Lookup(pmbl.primitives.Variable("math"), "log")

    info = symbolic.math_func_mangler(None, name, (dtype,))

    assert info.target_name == f"{prefix}_log"


def test_math_func_mangler_ignores_non_lookup_names():
    dtype = NumpyType(np.dtype(np.float64))

    assert symbolic.math_func_mangler(None, pmbl.var("exp"), (dtype,)) is None
    assert symbolic.math_func_mangler(None, pmbl.var("exp"), (dtype, dtype)) is None


def test_math_func_mangler_rejects_foreign_aggregate():
    dtype = NumpyType(np.dtype(np.float64))
    name = pmbl.primitives.Lookup(pmbl.primitives.Variable("numpy"), "exp")

    with pytest.raises(RuntimeError):
        symbolic.math_func_mangler(None, name, (dtype,))


@pytest.mark.parametrize(
    ("dim", "expected_vars"),
    [(1, ["x"]), (2, ["x", "y"]), (3, ["x", "y", "z"])],
)
def test_get_evaluator_uses_default_coordinate_variables(dim, expected_vars):
    evaluator = symbolic.get_evaluator(dim, symbolic.x)

    assert evaluator.dim == dim
    assert [str(var) for var in evaluator.vars] == expected_vars
    assert evaluator.function_manglers == [symbolic.math_func_mangler]


def test_get_evaluator_normalises_math_calls():
    evaluator = symbolic.get_evaluator(2, symbolic.exp(symbolic.x) + symbolic.y)
    normalised = evaluator.get_normalised_expr()

    assert str(normalised) == "exp(x0) + x1"


# vim: filetype=pyopencl.python:fdm=marker
