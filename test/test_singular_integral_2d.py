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

import numpy as np

import volumential.singular_integral_2d as sint


def test_quadrature_1d_interval():
    integrand = np.sin
    val, err = sint.quad(
        func=integrand,
        a=-np.pi,
        b=np.pi,
        args=(),
        tol=1e-8,
        rtol=1e-8,
        maxiter=50,
        vec_func=False,
        miniter=1,
    )
    assert np.abs(val) < 1e-8
    assert err < 1e-8


def test_quadrature_2d_box():
    def integrand(x, y):
        return np.sin(10 * x) * np.cos(3 * y)

    val, err = sint.qquad(
        func=integrand,
        a=-np.pi,
        b=np.pi,
        c=-np.pi,
        d=np.pi,
        args=(),
        tol=1e-8,
        rtol=1e-8,
        maxitero=50,
        maxiteri=50,
        vec_func=True,
        minitero=1,
        miniteri=1,
    )
    assert np.abs(val) < 1e-8
    assert err < 1e-8


def test_affine_mapping():
    tria1 = ((1, 1), (2, 3), (0, 4))
    tria2 = ((0, 0), (1, 0), (0, 1))
    mp, jac, imp, ivjac = sint.solve_affine_map_2d(tria1, tria2)
    assert np.isclose(jac * ivjac, 1)
    for i in range(3):
        assert np.allclose(mp(tria1[i]), tria2[i])
        assert np.allclose(imp(tria2[i]), tria1[i])


def test_is_in_t():
    p1 = (0.1, 0.1)
    p2 = (1.1, 0.1)
    p3 = (-0.1, 0.1)
    p4 = (0.1, -0.7)
    assert sint.is_in_t(p1)
    assert not sint.is_in_t(p2)
    assert not sint.is_in_t(p3)
    assert not sint.is_in_t(p4)


def test_is_in_r():
    p1 = (0.1, 0.1)
    p2 = (1.1, 0.1)
    p3 = (-0.1, 0.1)
    p4 = (0.1, -0.7)
    assert sint.is_in_r(p1)
    assert not sint.is_in_r(p2)
    assert not sint.is_in_r(p3)
    assert not sint.is_in_r(p4)


def test_tria2rect_map():
    t2r, jac_t2r, r2t, jac_r2t = sint.tria2rect_map_2d()
    assert np.allclose(t2r((0, 0)), (0, 0))
    assert np.allclose(t2r((1, 0)), (1, 0))
    assert np.allclose(t2r((0, 1)), (1, np.pi / 2))
    assert np.allclose(r2t((0, 0)), (0, 0))
    assert np.allclose(r2t((1, 0)), (1, 0))
    assert np.allclose(r2t((1, np.pi / 2)), (0, 1))
    assert np.allclose(r2t((0, np.pi / 2)), (0, 0))


def test_tria2rect_jacobian():
    t2r, jac_t2r, r2t, jac_r2t = sint.tria2rect_map_2d()
    p1 = (0.1, 0.1)
    assert np.isclose(jac_t2r(p1) * jac_r2t(t2r(p1)), 1)
    assert np.isclose(jac_r2t(p1) * jac_t2r(r2t(p1)), 1)


def run_test_tria_quad(func, region, exact):
    val, err = sint.tria_quad(
        func,
        region,
        args=(),
        tol=1e-10,
        rtol=1e-10,
        maxiter=50,
        vec_func=True,
        miniter=1,
    )

    assert np.isclose(exact, val, atol=1e-8)
    assert np.isclose(err, 0, atol=1e-8)


def test_tria_quad_1():
    def const_func(x, y):
        return 1

    # area = 0.5 * (10 * 9) = 45
    region = ((2.33, 1), (12.33, 1), (5, 10))

    run_test_tria_quad(const_func, region, 45)


def test_tria_quad_2():
    def greens_func(x, y, x0, y0):
        return -1 / (2 * np.pi) * np.log((x - x0) ** 2 + (y - y0) ** 2) * 0.5

    region = ((2.33, 1), (12.33, 1), (5, 10))

    val, err = sint.tria_quad(
        greens_func,
        region,
        args=region[0],
        tol=1e-8,
        rtol=1e-8,
        maxiter=80,
        vec_func=True,
        miniter=1,
    )

    assert np.isclose(err, 0, atol=1e-6)
    assert np.isfinite(val)


def test_box_quad_1():
    def const_func(x, y):
        return 1

    a = 1
    b = 2
    c = 23
    d = 40
    area = (b - a) * (d - c)
    sp = ((a + b) / 2, (c + d) / 2)

    val, err = sint.box_quad(
        const_func,
        a,
        b,
        c,
        d,
        sp,
        args=(),
        tol=1e-8,
        rtol=1e-8,
        maxiter=50,
        vec_func=True,
        miniter=1,
    )

    assert np.isclose(val, area, atol=1e-8)
    assert np.isclose(err, 0, atol=1e-8)


def test_box_quad_2():
    def const_func(x, y):
        return 1

    a = 1
    b = 2
    c = 23
    d = 40
    area = (b - a) * (d - c)
    sp = (0, 0)

    val, err = sint.box_quad(
        const_func,
        a,
        b,
        c,
        d,
        sp,
        args=(),
        tol=1e-8,
        rtol=1e-8,
        maxiter=50,
        vec_func=True,
        miniter=1,
    )

    print(area, val)
    print(err)
    assert np.isclose(val, area, atol=1e-8)
    assert np.isclose(err, 0, atol=1e-8)


def test_box_quad_3(longrun):
    def greens_func(x, y, x0, y0):
        return -1 / (2 * np.pi) * np.log((x - x0) ** 2 + (y - y0) ** 2) * 0.5

    a = 0
    b = 1
    c = 0
    d = 1
    # by symmetry, all four singular integrals should be equal
    sps = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]

    val = np.zeros(len(sps))
    err = np.zeros(len(sps))

    for i in range(len(sps)):
        val[i], err[i] = sint.box_quad(
            greens_func,
            a,
            b,
            c,
            d,
            sps[i],
            args=sps[i],
            tol=1e-8,
            rtol=1e-8,
            maxiter=50,
            vec_func=True,
            miniter=1,
        )

    val = val - np.mean(val)

    assert np.allclose(val, 0, atol=1e-8)
    assert np.allclose(err, 0, atol=1e-8)


def _unit_square_volume_integral(func):
    val, err = sint.qquad(
        func,
        0.0,
        1.0,
        0.0,
        1.0,
        args=(),
        tol=1e-11,
        rtol=1e-11,
        maxitero=60,
        maxiteri=60,
        vec_func=True,
        minitero=2,
        miniteri=2,
    )
    return float(val), float(err)


def _unit_square_boundary_integral(boundary_func):
    pieces = [
        lambda t: boundary_func(np.zeros_like(t), t),
        lambda t: boundary_func(np.ones_like(t), t),
        lambda t: boundary_func(t, np.zeros_like(t)),
        lambda t: boundary_func(t, np.ones_like(t)),
    ]

    total_val = 0.0
    total_err = 0.0
    for piece in pieces:
        val, err = sint.quad(
            piece,
            0.0,
            1.0,
            args=(),
            tol=1e-11,
            rtol=1e-11,
            maxiter=60,
            vec_func=True,
            miniter=2,
        )
        total_val += float(val)
        total_err += float(err)

    return total_val, total_err


def test_first_green_identity_unit_square():
    def u(x, y):
        return x * x + 2.0 * x * y + 3.0 * y * y + x + 1.0

    def v(x, y):
        return x * x * x - x * y + y * y + 2.0

    def du_dx(x, y):
        return 2.0 * x + 2.0 * y + 1.0

    def du_dy(x, y):
        return 2.0 * x + 6.0 * y

    def dv_dx(x, y):
        return 3.0 * x * x - y

    def dv_dy(x, y):
        return -x + 2.0 * y

    def lap_v(x, y):
        return 6.0 * x + 2.0

    lhs, lhs_err = _unit_square_volume_integral(
        lambda x, y: (
            du_dx(x, y) * dv_dx(x, y)
            + du_dy(x, y) * dv_dy(x, y)
            + u(x, y) * lap_v(x, y)
        )
    )

    rhs, rhs_err = _unit_square_boundary_integral(
        lambda x, y: (
            (
                np.where(np.isclose(x, 0.0), -1.0, 0.0)
                + np.where(np.isclose(x, 1.0), 1.0, 0.0)
            )
            * u(x, y)
            * dv_dx(x, y)
            + (
                np.where(np.isclose(y, 0.0), -1.0, 0.0)
                + np.where(np.isclose(y, 1.0), 1.0, 0.0)
            )
            * u(x, y)
            * dv_dy(x, y)
        )
    )

    assert abs(lhs - rhs) < max(1e-9, 100.0 * (lhs_err + rhs_err))


def test_second_green_identity_unit_square():
    def u(x, y):
        return x * x + 2.0 * x * y + 3.0 * y * y + x + 1.0

    def v(x, y):
        return x * x * x - x * y + y * y + 2.0

    def du_dx(x, y):
        return 2.0 * x + 2.0 * y + 1.0

    def du_dy(x, y):
        return 2.0 * x + 6.0 * y

    def dv_dx(x, y):
        return 3.0 * x * x - y

    def dv_dy(x, y):
        return -x + 2.0 * y

    def lap_u(x, y):
        return 8.0 + 0.0 * x + 0.0 * y

    def lap_v(x, y):
        return 6.0 * x + 2.0

    lhs, lhs_err = _unit_square_volume_integral(
        lambda x, y: u(x, y) * lap_v(x, y) - v(x, y) * lap_u(x, y)
    )

    rhs, rhs_err = _unit_square_boundary_integral(
        lambda x, y: (
            (
                np.where(np.isclose(x, 0.0), -1.0, 0.0)
                + np.where(np.isclose(x, 1.0), 1.0, 0.0)
            )
            * (u(x, y) * dv_dx(x, y) - v(x, y) * du_dx(x, y))
            + (
                np.where(np.isclose(y, 0.0), -1.0, 0.0)
                + np.where(np.isclose(y, 1.0), 1.0, 0.0)
            )
            * (u(x, y) * dv_dy(x, y) - v(x, y) * du_dy(x, y))
        )
    )

    assert abs(lhs - rhs) < max(1e-9, 100.0 * (lhs_err + rhs_err))


# vim: filetype=pyopencl.python:fdm=marker
