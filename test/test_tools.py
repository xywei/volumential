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

import sys

import numpy as np
import pytest

import pymbolic as pmbl

from volumential.tools import (
    ScalarFieldExpressionEvaluation,
    clean_file,
    generate_leading_order_filtering,
    import_code,
)


# {{{ clean_file


def test_clean_file_removes_existing_file(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("data")

    clean_file(str(victim))

    assert not victim.exists()


def test_clean_file_renames_when_new_name_given(tmp_path):
    original = tmp_path / "original.txt"
    original.write_text("data")
    renamed = tmp_path / "renamed.txt"

    clean_file(str(original), str(renamed))

    assert not original.exists()
    assert renamed.read_text() == "data"


def test_clean_file_is_silent_on_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"

    clean_file(str(missing))
    clean_file(str(missing), str(tmp_path / "elsewhere.txt"))

    assert not missing.exists()


# }}} End clean_file

# {{{ import_code


def test_import_code_builds_module_without_registering_it():
    name = "volumential_test_import_code_unregistered"
    module = import_code("VALUE = 41 + 1\n", name, add_to_sys_modules=False)

    assert module.__name__ == name
    assert module.VALUE == 42
    assert name not in sys.modules


def test_import_code_registers_module_in_sys_modules():
    name = "volumential_test_import_code_registered"
    try:
        module = import_code("def twice(x):\n    return 2 * x\n", name)
        assert sys.modules[name] is module
        assert module.twice(3) == 6
    finally:
        sys.modules.pop(name, None)


# }}} End import_code

# {{{ leading order filtering


def test_leading_order_filtering_1d_is_last_node_indicator():
    mask = generate_leading_order_filtering(1, 4)
    assert np.array_equal(mask, np.array([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.parametrize("dim", [2, 3])
def test_leading_order_filtering_marks_highest_mode_in_any_axis(dim):
    n_dofs = 3
    mask = generate_leading_order_filtering(dim, n_dofs).reshape((n_dofs,) * dim)

    for index in np.ndindex(*mask.shape):
        expected = 1.0 if any(i == n_dofs - 1 for i in index) else 0.0
        assert mask[index] == expected


def test_leading_order_filtering_rejects_unsupported_dimension():
    with pytest.raises(NotImplementedError):
        generate_leading_order_filtering(4, 3)


# }}} End leading order filtering

# {{{ scalar field expression evaluation (host-side parts)


def test_scalar_field_eval_normalises_variable_names():
    xvar = pmbl.var("x")
    yvar = pmbl.var("y")
    evaluator = ScalarFieldExpressionEvaluation(
        dim=2, expression=xvar * yvar + 1, variables=[xvar, yvar]
    )

    assert str(evaluator.get_normalised_expr()) == str(
        pmbl.var("x0") * pmbl.var("x1") + 1
    )


def test_scalar_field_eval_rewrites_math_lookups_as_bare_calls():
    xvar = pmbl.var("x")
    expression = pmbl.primitives.Call(
        pmbl.primitives.Lookup(pmbl.primitives.Variable("math"), "sin"), (xvar,)
    )
    evaluator = ScalarFieldExpressionEvaluation(
        dim=1, expression=expression, variables=[xvar]
    )

    normalised = evaluator.get_normalised_expr()
    assert isinstance(normalised, pmbl.primitives.Call)
    assert normalised.function == pmbl.var("sin")
    assert normalised.parameters == (pmbl.var("x0"),)


def test_scalar_field_eval_cache_key_tracks_dimension_and_expression():
    xvar = pmbl.var("x")
    key_1d = ScalarFieldExpressionEvaluation(
        dim=1, expression=xvar, variables=[xvar]
    ).get_cache_key()
    key_1d_again = ScalarFieldExpressionEvaluation(
        dim=1, expression=xvar, variables=[xvar]
    ).get_cache_key()
    key_other_expr = ScalarFieldExpressionEvaluation(
        dim=1, expression=2 * xvar, variables=[xvar]
    ).get_cache_key()

    assert key_1d == key_1d_again
    assert key_1d != key_other_expr
    assert key_1d[0] == "ScalarFieldExpressionEvaluation"
    assert key_1d[1] == "1D"


@pytest.mark.parametrize(
    ("dim", "expected_lines"),
    [(1, 1), (2, 2), (3, 3)],
)
def test_scalar_field_eval_variable_assignment_code(dim, expected_lines):
    evaluator = ScalarFieldExpressionEvaluation(dim=dim, expression=pmbl.var("x0"))
    code = evaluator.get_variable_assignment_code()

    assert len(code.splitlines()) == expected_lines
    for axis in range(dim):
        assert f"<> x{axis} = target_points[{axis}, itgt]" in code


def test_scalar_field_eval_rejects_unsupported_dimension():
    evaluator = ScalarFieldExpressionEvaluation(dim=4, expression=pmbl.var("x0"))
    with pytest.raises(NotImplementedError):
        evaluator.get_variable_assignment_code()


# }}} End scalar field expression evaluation

# vim: filetype=pyopencl.python:fdm=marker
