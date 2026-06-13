import numpy as np

from volumential import dmk_split as dmk
from volumential import singular_integral_2d as sint


def _safe_laplace2d_table_integrand(coeffs, target):
    def integrand(x, y):
        r = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)
        density = dmk.evaluate_polynomial_2d(coeffs, x, y)

        if np.isscalar(r):
            if r == 0:
                return 0.0
            return density * dmk.laplace2d_kernel_radius(r)

        result = np.zeros_like(r, dtype=np.float64)
        mask = r > 0
        result[mask] = density[mask] * dmk.laplace2d_kernel_radius(r[mask])
        return result

    return integrand


def _duffy_reference_laplace2d(coeffs, target):
    value, _ = sint.box_quad_duffy_radial(
        _safe_laplace2d_table_integrand(coeffs, target),
        0,
        1,
        0,
        1,
        target,
        radial_rule="tanh-sinh-fast",
        deg_theta=80,
        radial_quad_order=181,
    )
    return value


def test_compact_window_support_avoids_gauss_node_boundary_intersection():
    q_order = 5
    target = dmk.tensor_gauss_point(q_order, 0)
    spacing = dmk.minimum_boundary_spacing(q_order)

    assert dmk.support_relation_to_box(
        target,
        dmk.CompactWindowConfig(sigma=0.99 * spacing),
    ) == "inside"

    assert dmk.support_relation_to_box(
        target,
        dmk.CompactWindowConfig(sigma=1.01 * spacing),
    ) == "intersects"

    assert dmk.support_relation_to_box(
        (-0.1, 0.5),
        dmk.CompactWindowConfig(sigma=0.05),
    ) == "outside"


def test_laplace2d_dmk_like_split_parameter_sweep_reaches_high_accuracy():
    q_order = 5
    mode_index = 12
    target_index = 12
    coeffs = dmk.tensor_lagrange_mode_coefficients(q_order, mode_index)
    target = dmk.tensor_gauss_point(q_order, target_index)
    reference = _duffy_reference_laplace2d(coeffs, target)

    rows = dmk.sweep_laplace2d_dmk_split(
        q_order=q_order,
        mode_index=mode_index,
        target_index=target_index,
        sigmas=[0.15, 0.25, 0.35],
        expansion_orders=[2, 4, 6, 8],
        smooth_orders=[64, 96, 160, 220],
        reference_value=reference,
    )

    low_expansion = next(
        row
        for row in rows
        if np.isclose(row["sigma"], 0.35)
        and row["expansion_order"] == 2
        and row["smooth_order"] == 220
    )
    high_expansion = next(
        row
        for row in rows
        if np.isclose(row["sigma"], 0.35)
        and row["expansion_order"] == 8
        and row["smooth_order"] == 220
    )
    assert high_expansion["support_relation"] == "inside"
    assert high_expansion["abs_error"] < 1.0e-11
    assert high_expansion["abs_error"] < 1.0e-8 * low_expansion["abs_error"]

    low_smooth_order = next(
        row
        for row in rows
        if np.isclose(row["sigma"], 0.35)
        and row["expansion_order"] == 8
        and row["smooth_order"] == 64
    )
    assert high_expansion["abs_error"] < 1.0e-3 * low_smooth_order["abs_error"]


def test_laplace2d_dmk_like_split_rejects_intersecting_local_support():
    q_order = 5
    mode_index = 12
    target_index = 0
    coeffs = dmk.tensor_lagrange_mode_coefficients(q_order, mode_index)
    target = dmk.tensor_gauss_point(q_order, target_index)

    config = dmk.CompactWindowConfig(
        sigma=1.1 * dmk.minimum_boundary_spacing(q_order),
    )

    try:
        dmk.laplace2d_dmk_split_integral(
            coeffs,
            target=target,
            config=config,
            expansion_order=8,
            smooth_order=64,
        )
    except ValueError as exc:
        assert "compact support intersects" in str(exc)
    else:
        raise AssertionError("intersecting compact support should be rejected")
