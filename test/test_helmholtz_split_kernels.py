"""Backend-independence tests for the Helmholtz-split auxiliary kernels.

The kernels in :mod:`volumential.wranglers.kernels` are handed to
:mod:`sumpy`, whose symbolic backend is symengine whenever symengine is
importable and ``SUMPY_FORCE_SYMBOLIC_BACKEND`` does not say otherwise.  A
relational anywhere inside such a kernel's expression makes sumpy's common
subexpression elimination fail under that backend, because it rebuilds every
subexpression through ``expr.func(*args)`` and symengine's relationals carry
no ``func`` -- so the tests here pin the expressions as relational-free and
run the CSE that used to raise.

See xywei/volumential#151.
"""

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

import math

import pytest

from pymbolic.mapper import WalkMapper

from volumential.wranglers.kernels import (
    _LOG_ARG_FLOOR,
    _HelmholtzSplitSeriesRemainderKernel,
    _RadialPowerKernel,
    _RadialPowerLogKernel,
)


# {{{ helpers

class _RelationalCollector(WalkMapper):
    """Collect every relational and branch node of a pymbolic expression."""

    def __init__(self):
        super().__init__()
        self.found = []

    def map_comparison(self, expr, *args, **kwargs):
        self.found.append(expr)

    def map_if(self, expr, *args, **kwargs):
        self.found.append(expr)

    def map_logical_and(self, expr, *args, **kwargs):
        self.found.append(expr)

    def map_logical_or(self, expr, *args, **kwargs):
        self.found.append(expr)

    def map_logical_not(self, expr, *args, **kwargs):
        self.found.append(expr)


def _relationals_in(expr):
    collector = _RelationalCollector()
    collector(expr)
    return collector.found


def _split_kernels():
    """Every auxiliary kernel the Helmholtz split hands to sumpy."""

    return [
        _RadialPowerKernel(2, 0),
        _RadialPowerKernel(2, 4),
        _RadialPowerKernel(3, 3),
        _RadialPowerLogKernel(2, 2),
        _RadialPowerLogKernel(2, 4),
        # split_order 1 keeps every ``r**2n log r`` term in the remainder;
        # higher orders move the leading ones into pretabulated term tables.
        _HelmholtzSplitSeriesRemainderKernel(2, 8.0, 0.0, 1, 6),
        _HelmholtzSplitSeriesRemainderKernel(2, 8.0, 0.0, 2, 6),
        _HelmholtzSplitSeriesRemainderKernel(2, 4.0, 0.5, 3, 8),
        _HelmholtzSplitSeriesRemainderKernel(3, 8.0, 0.0, 1, 6),
        _HelmholtzSplitSeriesRemainderKernel(3, 4.0, 0.5, 3, 8),
    ]


def _kernel_id(knl):
    return repr(knl)

# }}}


@pytest.mark.parametrize("knl", _split_kernels(), ids=_kernel_id)
def test_split_kernel_expression_has_no_relational(knl):
    """No split kernel may put a relational into its sumpy expression."""

    assert _relationals_in(knl.expression) == []


@pytest.mark.parametrize("knl", _split_kernels(), ids=_kernel_id)
def test_split_kernel_survives_sumpy_global_cse(knl):
    """Run the CSE that ``AttributeError: 'LessThan' ...`` came out of.

    This is sumpy's own code generation path, minus the loopy and OpenCL
    parts, so it needs no device and runs under whichever symbolic backend
    sumpy picked.
    """

    import sumpy.symbolic as sym
    from sumpy.assignment_collection import SymbolicAssignmentCollection

    expr = knl.get_expression(sym.make_sym_vector("d", knl.dim))
    sac = SymbolicAssignmentCollection({"result": expr})
    sac.run_global_cse()

    assert sac.assignments


def test_log_arg_floor_vanishes_at_zero_distance():
    """``r**power * log(r + floor)`` is the branch the relational replaced.

    A P2P that does not exclude self interactions evaluates the kernel at
    ``r == 0``, where an unregularized ``r**power * log(r)`` is ``0 * -inf``,
    that is ``nan``.
    """

    assert _LOG_ARG_FLOOR > 0.0
    # A normal double, not a subnormal: a device that flushes subnormals to
    # zero would otherwise turn the floor back into ``log(0)``.
    assert _LOG_ARG_FLOOR > 2.3e-308

    for power in (1, 2, 4, 6):
        value = (0.0**power) * math.log(0.0 + _LOG_ARG_FLOOR)
        assert value == 0.0


@pytest.mark.parametrize(
    "r", [1.0e-160, 1.0e-30, 1.0e-12, 1.0e-6, 1.0e-3, 0.1, 0.5, 1.0, 2.0, 1.0e3]
)
def test_log_arg_floor_is_exact_at_working_distances(r):
    """The floor is invisible: ``r + floor`` is ``r`` bitwise, wherever the
    term it guards is not already zero by underflow."""

    assert r + _LOG_ARG_FLOOR == r
    assert math.log(r + _LOG_ARG_FLOOR) == math.log(r)


def test_log_arg_floor_leaves_underflowed_terms_at_zero():
    """Below the exactness range, ``r**power`` has underflowed anyway."""

    # Small enough that half an ulp of ``r`` is below the floor, so the floor
    # does move the sum here.
    r = 1.0e-290
    assert r + _LOG_ARG_FLOOR != r

    for power in (2, 4, 6):
        assert r**power == 0.0
        assert (r**power) * math.log(r + _LOG_ARG_FLOOR) == 0.0


# You can test individual routines by typing
# $ python test_helmholtz_split_kernels.py 'test_routine()'

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
