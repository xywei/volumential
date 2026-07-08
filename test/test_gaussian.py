import math

import numpy as np
import pytest

from volumential.gaussian import (
    GaussianComponent,
    GaussianMixture,
    axis_aligned_slice_grid,
    evaluate_gaussian_mixture,
    gaussian_mixture_tail_report,
    laplace3d_gaussian_potential,
    nearest_axis_slice,
)


def test_evaluate_gaussian_mixture_accepts_point_or_axis_major_points():
    mixture = GaussianMixture(
        "two-bumps",
        (
            GaussianComponent(2.0, (0.0, 0.0), 1.0),
            GaussianComponent(0.5, (1.0, 0.0), 4.0),
        ),
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])

    values = evaluate_gaussian_mixture(mixture, points)
    axis_major_values = evaluate_gaussian_mixture(mixture, points.T)

    np.testing.assert_allclose(values, axis_major_values)
    np.testing.assert_allclose(values[0], 2.0 + 0.5 * math.exp(-4.0))
    np.testing.assert_allclose(values[1], 2.0 * math.exp(-1.0) + 0.5)


def test_laplace3d_gaussian_potential_matches_center_limit_and_kernel_scale():
    mixture = GaussianMixture(
        "single",
        (GaussianComponent(3.0, (0.0, 0.0, 0.0), 2.0),),
    )

    center_value = laplace3d_gaussian_potential(mixture, np.array([[0.0, 0.0, 0.0]]))[0]
    classical_center_value = laplace3d_gaussian_potential(
        mixture,
        np.array([[0.0, 0.0, 0.0]]),
        kernel_scale=1.0 / (4.0 * math.pi),
    )[0]

    assert center_value == pytest.approx(2.0 * math.pi * 3.0 / 2.0)
    assert classical_center_value == pytest.approx(3.0 / (2.0 * 2.0))


def test_laplace3d_gaussian_potential_matches_radial_closed_form():
    amplitude = 1.25
    alpha = 5.0
    radius = 0.7
    mixture = GaussianMixture(
        "single",
        (GaussianComponent(amplitude, (0.0, 0.0, 0.0), alpha),),
    )

    value = laplace3d_gaussian_potential(mixture, np.array([[radius, 0.0, 0.0]]))[0]
    mass = amplitude * (math.pi / alpha) ** 1.5
    expected = mass * math.erf(math.sqrt(alpha) * radius) / radius

    assert value == pytest.approx(expected)


def test_gaussian_tail_report_uses_separable_box_mass():
    mixture = GaussianMixture(
        "single-1d",
        (GaussianComponent(1.0, (0.0,), 1.0),),
    )

    report = gaussian_mixture_tail_report(mixture, np.array([[-1.0, 1.0]]))

    assert report["total_abs_component_mass"] == pytest.approx(math.sqrt(math.pi))
    assert report["omitted_abs_fraction"] == pytest.approx(math.erfc(1.0))
    assert report["components"][0]["omitted_abs_fraction"] == pytest.approx(math.erfc(1.0))


def test_axis_aligned_slice_grid_embeds_2d_grid_in_3d_box():
    grid = axis_aligned_slice_grid(
        np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]),
        fixed_axis=1,
        fixed_value=0.25,
        shape=(3, 4),
    )

    assert grid.points.shape == (12, 3)
    assert grid.shape == (3, 4)
    np.testing.assert_allclose(grid.points[:, 1], 0.25)
    np.testing.assert_allclose(grid.axis_values[0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(grid.axis_values[1], [-3.0, -1.0, 1.0, 3.0])


def test_nearest_axis_slice_selects_nearest_plane_and_fields():
    points = np.array(
        [
            [0.0, 0.0, -0.4],
            [1.0, 0.0, -0.1],
            [2.0, 0.0, 0.1],
            [3.0, 0.0, 0.5],
        ]
    )
    result = nearest_axis_slice(
        points,
        {"value": np.array([10.0, 11.0, 12.0, 13.0])},
        axis=2,
        value=0.0,
    )

    np.testing.assert_array_equal(result["indices"], [1, 2])
    np.testing.assert_allclose(result["value"], [11.0, 12.0])
    np.testing.assert_allclose(result["axis_distances"], [0.1, 0.1])
