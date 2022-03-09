from __future__ import absolute_import, division, print_function

__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

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

import volumential.nearfield_potential_table as npt
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebval2d, chebval3d


def test_const_order_1():
    table = npt.NearFieldInteractionTable(
        quad_order=1, build_method='Transform', progress_bar=False)
    table.build_table()
    for ary in table.data:
        assert np.allclose(ary, 1)


def test_const_order_2(longrun):
    table = npt.NearFieldInteractionTable(
        quad_order=2, build_method='Transform', progress_bar=False)
    table.build_table()
    for ary in table.data:
        assert np.allclose(ary, 0.25)


def interp_modes(q_order):
    table = npt.NearFieldInteractionTable(
            quad_order=q_order, build_method='Transform', progress_bar=False)

    modes = [table.get_mode(i) for i in range(table.n_q_points)]

    def interpolate_function(x, y):
        return sum([mode(x, y) for mode in modes])

    h = 0.2
    xx = yy = np.arange(-1.0, 1.0, h)
    xi, yi = np.meshgrid(xx, yy)
    xi = xi.flatten()
    yi = yi.flatten()

    val = np.zeros(xi.shape)

    for i in range(len(xi)):
        val[i] = interpolate_function(xi[i], yi[i])

    return val


def test_modes():
    for q in [1, 2, 3, 4, 5, 6]:
        val = interp_modes(q)
        assert np.allclose(val, 1)


def cheb_eval(dim, coefs, coords):
    if dim == 1:
        return chebval(coords[0], coefs)
    elif dim == 2:
        return chebval2d(coords[0], coords[1], coefs)
    elif dim == 3:
        return chebval3d(coords[0], coords[1], coords[2], coefs)
    else:
        raise NotImplementedError('dimension %d not supported' % dim)


def drive_test_modes_cheb_coeffs(dim, q, cheb_order):
    if not cheb_order >= q:
        raise RuntimeError('Insufficient cheb_order to fully resolve the modes')

    sample_mode = np.random.randint(q**dim)
    table = npt.NearFieldInteractionTable(quad_order=q, dim=dim, progress_bar=False)
    ccoefs = table.get_mode_cheb_coeffs(sample_mode, cheb_order)
    shape = (cheb_order, ) * dim
    ccoefs = ccoefs.reshape(*shape)

    # Evaluate the mode at the interpolation nodes via
    # inverse Chebyshev transform.
    #
    # NOTE: table.q_points are over [0, 1]^dim,
    # while cheb_eval assumes points are over [-1, 1]^dim
    targets = np.array([
        [qpt[i] for qpt in table.q_points]
        for i in range(dim)]) * 2 - 1

    mode_vals = cheb_eval(dim, ccoefs, targets)
    mode_vals[np.abs(mode_vals) < 8 * np.finfo(mode_vals.dtype).eps] = 0

    mode_interp_vals = np.zeros(q**dim)
    mode_interp_vals[sample_mode] = 1

    assert np.allclose(mode_vals, mode_interp_vals)


def test_modes_cheb_coeffs():
    drive_test_modes_cheb_coeffs(1, 5, 5)
    drive_test_modes_cheb_coeffs(1, 5, 10)
    drive_test_modes_cheb_coeffs(1, 10, 10)
    drive_test_modes_cheb_coeffs(1, 15, 15)

    drive_test_modes_cheb_coeffs(2, 5, 5)
    drive_test_modes_cheb_coeffs(2, 5, 10)
    drive_test_modes_cheb_coeffs(2, 10, 10)
    drive_test_modes_cheb_coeffs(2, 15, 15)

    drive_test_modes_cheb_coeffs(3, 5, 5)
    drive_test_modes_cheb_coeffs(3, 5, 10)
    drive_test_modes_cheb_coeffs(3, 10, 10)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main

        main([__file__])

# vim: foldmethod=marker:filetype=pyopencl
