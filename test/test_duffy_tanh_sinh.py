"""Tests that the tanh-sinh radial Duffy rules agree with the adaptive
triangle/box baselines on singular integrands.
"""

import math
import warnings

import numpy as np
import pytest

from volumential.singular_integral_2d import (
    box_quad,
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

    def func(x, y):
        return 1.0 / np.sqrt(x * x + y * y)

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

    def func(x, y, z):
        return 1.0 / np.sqrt(x * x + y * y + z * z)

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


def _log_r(x, y):
    return np.log(np.sqrt(x * x + y * y))


def _complex_log_r(x, y):
    return (1.0 - 2.0j) * _log_r(x, y)


@pytest.mark.parametrize("radial_rule", ["tanh-sinh-fast", "tanh-sinh", "adaptive"])
def test_radial_duffy_keeps_the_imaginary_part(radial_rule):
    """A complex integrand integrates to a complex value (#180).

    Every radial rule used to call ``float()`` on each integrand value, which
    keeps the real part: a complex kernel lost its imaginary part with only a
    ``ComplexWarning`` to show for it.
    """
    tria = ((0.0, 0.0), (1.0, 0.0), (0.3, 0.8))
    rule = {
        "radial_rule": radial_rule,
        "deg_theta": 8,
        "radial_quad_order": 61,
        "mp_dps": 20,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        real_val, _ = tria_quad_duffy_radial(_log_r, tria, **rule)
        complex_val, _ = tria_quad_duffy_radial(_complex_log_r, tria, **rule)

    assert not np.iscomplexobj(real_val)
    assert np.iscomplexobj(complex_val)
    # not bitwise: the adaptive rules may stop at another order for the
    # complex iterate, whose absolute change is sqrt(5) times larger
    assert abs(complex_val - (1.0 - 2.0j) * real_val) <= 1e-10 * abs(real_val)


def test_box_quad_keeps_the_imaginary_part():
    """:func:`box_quad` cast the integrand to ``float`` the same way."""
    box = (0.0, 1.0, 0.0, 1.0)
    singular_point = (0.3, 0.4)

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        real_val, _ = box_quad(_log_r, *box, singular_point, vec_func=False)
        complex_val, _ = box_quad(
            _complex_log_r, *box, singular_point, vec_func=False
        )

    assert np.iscomplexobj(complex_val)
    # within the rule's default tolerance, for the reason given above
    assert abs(complex_val - (1.0 - 2.0j) * real_val) <= 1e-6 * abs(real_val)
