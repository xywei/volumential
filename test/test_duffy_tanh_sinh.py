import math

import numpy as np

from volumential.singular_integral_2d import (
    tria_quad,
    tria_quad_duffy_radial,
)


def test_radial_duffy_quadrature_matches_adaptive_triangle_baseline():
    tria = ((0.0, 0.0), (1.0, 0.0), (0.3, 0.8))

    cases = [
        lambda x, y: np.log(np.sqrt(x * x + y * y)),
        lambda x, y: 1.0 / np.sqrt(x * x + y * y),
        lambda x, y: math.exp(x - 0.3 * y) / np.sqrt(x * x + y * y),
    ]

    for func in cases:
        baseline, _ = tria_quad(
            func,
            tria,
            tol=1e-12,
            rtol=1e-12,
            maxiter=80,
            miniter=3,
            vec_func=False,
        )

        val, _ = tria_quad_duffy_radial(
            func,
            tria,
            radial_rule="tanh-sinh-fast",
            deg_theta=20,
            radial_quad_order=61,
            mp_dps=50,
        )

        assert abs(val - baseline) / max(1.0, abs(baseline)) < 1e-7


def test_radial_duffy_adaptive_matches_adaptive_triangle_baseline():
    tria = ((0.0, 0.0), (1.0, 0.0), (0.3, 0.8))

    func = lambda x, y: 1.0 / np.sqrt(x * x + y * y)

    baseline, _ = tria_quad(
        func,
        tria,
        tol=1e-12,
        rtol=1e-12,
        maxiter=80,
        miniter=3,
        vec_func=False,
    )

    val, _ = tria_quad_duffy_radial(
        func,
        tria,
        radial_rule="adaptive",
        deg_theta=20,
        mp_dps=50,
    )

    assert abs(val - baseline) / max(1.0, abs(baseline)) < 1e-10


def test_radial_duffy_3d_smoke_matches_adaptive_baseline():
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    singular_point = (0.0, 0.0, 0.0)

    func = lambda x, y, z: 1.0 / np.sqrt(x * x + y * y + z * z)

    from volumential.singular_integral_2d import box_quad_duffy_radial_nd

    baseline, _ = box_quad_duffy_radial_nd(
        func,
        bounds,
        singular_point,
        radial_rule="adaptive",
        deg_regular=10,
        radial_quad_order=31,
    )
    fast, _ = box_quad_duffy_radial_nd(
        func,
        bounds,
        singular_point,
        radial_rule="tanh-sinh-fast",
        deg_regular=10,
        radial_quad_order=61,
    )

    assert abs(fast - baseline) / max(1.0, abs(baseline)) < 1e-6
