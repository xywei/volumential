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


def test_heat_kernel_split_recombines_laplace_kernel_off_target():
    config = dmk.HeatKernelConfig(sigma=0.06)
    radii = np.array([0.01, 0.05, 0.2, 1.0])

    local = dmk.laplace2d_heat_local_kernel_radius(radii, config)
    smooth = dmk.laplace2d_heat_smooth_kernel(radii, np.zeros_like(radii), config)

    np.testing.assert_allclose(
        local + smooth,
        dmk.laplace2d_kernel_radius(radii),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.isfinite(dmk.laplace2d_heat_smooth_kernel(0.0, 0.0, config))
    assert np.isclose(
        dmk.laplace2d_heat_local_moment(0, 0, config),
        config.sigma**2 / 4.0,
    )


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


def test_laplace2d_heat_split_reaches_high_accuracy_with_smooth_remainder():
    q_order = 5
    mode_index = 12
    target_index = 12
    coeffs = dmk.tensor_lagrange_mode_coefficients(q_order, mode_index)
    target = dmk.tensor_gauss_point(q_order, target_index)
    reference = _duffy_reference_laplace2d(coeffs, target)

    rows = dmk.sweep_laplace2d_heat_split(
        q_order=q_order,
        mode_index=mode_index,
        target_index=target_index,
        sigmas=[0.06],
        expansion_orders=[8],
        smooth_orders=[24, 48],
        reference_value=reference,
    )

    low_smooth_order = next(row for row in rows if row["smooth_order"] == 24)
    high_smooth_order = next(row for row in rows if row["smooth_order"] == 48)

    assert high_smooth_order["support_relation"] == "heat-tail"
    assert high_smooth_order["abs_error"] < 1.0e-12
    assert high_smooth_order["abs_error"] < 1.0e-6 * low_smooth_order["abs_error"]


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


def test_laplace2d_dmk_like_split_sweep_does_not_hide_invalid_orders():
    q_order = 5
    mode_index = 12
    target_index = 12

    try:
        dmk.sweep_laplace2d_dmk_split(
            q_order=q_order,
            mode_index=mode_index,
            target_index=target_index,
            sigmas=[0.35],
            expansion_orders=[8],
            smooth_orders=[0],
            reference_value=0.0,
        )
    except ValueError as exc:
        assert "order must be >= 1" in str(exc)
    else:
        raise AssertionError("invalid smooth order should not be reported as support")


def test_laplace2d_dmk_like_split_sweep_rejects_invalid_expansion_order():
    try:
        dmk.sweep_laplace2d_dmk_split(
            q_order=5,
            mode_index=12,
            target_index=12,
            sigmas=[0.35],
            expansion_orders=[-1],
            smooth_orders=[64],
            reference_value=0.0,
        )
    except ValueError as exc:
        assert "expansion_order must be >= 0" in str(exc)
    else:
        raise AssertionError("invalid expansion order should fail validation")


def test_gauss_legendre_rule_rejects_fractional_order():
    try:
        dmk.gauss_legendre_rule(64.9)
    except ValueError as exc:
        assert "order must be an integer" in str(exc)
    else:
        raise AssertionError("fractional quadrature order should fail validation")


def test_laplace2d_dmk_like_split_sweep_materializes_generator_inputs():
    rows = dmk.sweep_laplace2d_dmk_split(
        q_order=3,
        mode_index=4,
        target_index=4,
        sigmas=(sigma for sigma in [0.1, 0.2]),
        expansion_orders=(order for order in [0, 2]),
        smooth_orders=(order for order in [6, 8]),
        reference_value=0.0,
    )

    assert {
        (row["sigma"], row["expansion_order"], row["smooth_order"])
        for row in rows
    } == {
        (sigma, expansion_order, smooth_order)
        for sigma in [0.1, 0.2]
        for expansion_order in [0, 2]
        for smooth_order in [6, 8]
    }
