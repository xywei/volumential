import volumential.nearfield_potential_table as npt
import pytest # NOQA
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


# @pytest.mark.skipif("True")
def test_const_order_1():
    table = npt.NearFieldInteractionTable(
        quad_order=1, kernel_func=None)
    table.build_table()
    for ary in table.data:
        assert (np.allclose(ary, 1))


# @pytest.mark.skipif("True")
def test_const_order_2():
    table = npt.NearFieldInteractionTable(
        quad_order=2, kernel_func=None)
    table.build_table()
    for ary in table.data:
        assert (np.allclose(ary, 0.25))


def interp_modes(q_order):
    table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        kernel_func=None)

    modes = [
        table.get_mode(i)
        for i in range(table.n_q_points)
    ]

    def interpolate_function(x, y):
        return sum([
            mode(x, y) for mode in modes
        ])

    h = 0.2
    xx = yy = np.arange(-1.0, 1.0, h)
    xi, yi = np.meshgrid(xx, yy)
    xi = xi.flatten()
    yi = yi.flatten()

    val = np.zeros(xi.shape)
    print(xi)
    print(yi)

    for i in range(len(xi)):
        val[i] = interpolate_function(
            xi[i], yi[i])

    return val


def test_modes():
    for q in [1, 2, 3, 4, 5, 6, 7, 8]:
        val = interp_modes(q)
        assert (np.allclose(val, 1))


def test_modes_cheb_coeffs():
    q = 5
    cheb_order = 10
    sample_mode = 14
    window = [0, 1]
    table = npt.NearFieldInteractionTable(
        quad_order=q, kernel_func=None)
    # mode = table.get_mode(sample_mode)
    ccoefs = table.get_mode_cheb_coeffs(sample_mode, cheb_order)
    ccoefs = ccoefs.reshape(cheb_order, cheb_order)

    for itq, q_point in zip(
            range(table.n_q_points), table.q_points):

        # First reverse in x direction
        tmpc = np.zeros(cheb_order)
        for f0 in range(cheb_order):
            tmpc[f0] = Chebyshev(ccoefs[f0], window)(q_point[0])

        # Then in y direction
        sval = Chebyshev(tmpc, window)(q_point[1])

        if itq == sample_mode:
            assert(abs(sval - 1) < 1e-10)
        else:
            assert(abs(sval - 0) < 1e-10)


# vim: foldmethod=marker:filetype=pyopencl
