import importlib.util
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "branched_flow_helmholtz2d.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "branched_flow_helmholtz2d", _EXAMPLE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EXAMPLE
_SPEC.loader.exec_module(_EXAMPLE)

_axis_window = _EXAMPLE._axis_window
_apply_fixed_polynomial = _EXAMPLE._apply_fixed_polynomial
_box_window = _EXAMPLE._box_window
_chebyshev_lobatto_nodes = _EXAMPLE._chebyshev_lobatto_nodes
_default_config = _EXAMPLE._default_config
_fmm_level_order = _EXAMPLE._fmm_level_order
_lagrange_interpolation_weights = _EXAMPLE._lagrange_interpolation_weights
_random_refractive_perturbation = _EXAMPLE._random_refractive_perturbation
_restrict_neighbor_csr = _EXAMPLE._restrict_neighbor_csr
_RightPreconditionedOperator = _EXAMPLE._RightPreconditionedOperator
_validate_config = _EXAMPLE._validate_config


def test_smooth_box_window_has_exact_compact_support():
    config = _default_config(smoke=True)
    axis = np.linspace(-config.root_half_extent, config.root_half_extent, 257)
    x, y = np.meshgrid(axis, axis)
    x = x.ravel()
    y = y.ravel()

    window = _box_window(x, y, config)
    outer = (
        (x <= config.core_x_min - config.transition_width)
        | (x >= config.core_x_max + config.transition_width)
        | (y <= config.core_y_min - config.transition_width)
        | (y >= config.core_y_max + config.transition_width)
    )
    core = (
        (x >= config.core_x_min)
        & (x <= config.core_x_max)
        & (y >= config.core_y_min)
        & (y <= config.core_y_max)
    )

    assert np.all(window[outer] == 0.0)
    assert np.all(window[core] == 1.0)
    assert np.all((window >= 0.0) & (window <= 1.0))


def test_axis_window_is_monotone_through_each_transition():
    config = _default_config(smoke=True)
    left_axis = np.linspace(
        config.core_x_min - config.transition_width,
        config.core_x_min,
        101,
    )
    right_axis = np.linspace(
        config.core_x_max,
        config.core_x_max + config.transition_width,
        101,
    )

    left = _axis_window(
        left_axis,
        config.core_x_min,
        config.core_x_max,
        config.transition_width,
    )
    right = _axis_window(
        right_axis,
        config.core_x_min,
        config.core_x_max,
        config.transition_width,
    )

    assert left[0] == 0.0
    assert left[-1] == 1.0
    assert np.all(np.diff(left) >= 0.0)
    assert right[0] == 1.0
    assert right[-1] == 0.0
    assert np.all(np.diff(right) <= 0.0)


def test_random_medium_has_requested_core_rms_and_compact_support():
    config = _default_config(smoke=True)
    axis = np.linspace(-config.root_half_extent, config.root_half_extent, 129)
    x, y = np.meshgrid(axis, axis)
    x = x.ravel()
    y = y.ravel()
    weights = np.ones_like(x)
    window = _box_window(x, y, config)

    perturbation = _random_refractive_perturbation(
        x,
        y,
        weights,
        window,
        config,
    )
    core = (
        (x >= config.core_x_min)
        & (x <= config.core_x_max)
        & (y >= config.core_y_min)
        & (y <= config.core_y_max)
    )
    outer = window == 0.0

    np.testing.assert_allclose(
        np.sqrt(np.mean(perturbation[core] ** 2)),
        config.refractive_rms,
        rtol=1.0e-13,
    )
    assert np.all(perturbation[outer] == 0.0)


def test_nested_medium_matches_on_overlapping_core_points():
    base = replace(
        _default_config(smoke=True),
        medium_realization="nested",
        medium_normalization_samples=1024,
    )
    larger = replace(
        base,
        core_x_min=-1.2,
        core_x_max=0.8,
        core_y_min=-0.9,
        core_y_max=0.9,
    )
    axis = np.linspace(-0.15, 0.15, 17)
    x, y = np.meshgrid(axis, axis)
    x = x.ravel()
    y = y.ravel()
    weights = np.ones_like(x)

    base_values = _random_refractive_perturbation(
        x,
        y,
        weights,
        _box_window(x, y, base),
        base,
    )
    larger_values = _random_refractive_perturbation(
        x,
        y,
        weights,
        _box_window(x, y, larger),
        larger,
    )

    np.testing.assert_array_equal(base_values, larger_values)


def test_filtered_grid_medium_is_reproducible_and_order_independent():
    config = replace(
        _default_config(smoke=True),
        medium_generator="filtered-grid",
    )
    axis = np.linspace(-config.root_half_extent, config.root_half_extent, 65)
    x, y = np.meshgrid(axis, axis)
    x = x.ravel()
    y = y.ravel()
    weights = np.linspace(0.5, 1.5, x.size)
    window = _box_window(x, y, config)
    values = _random_refractive_perturbation(
        x, y, weights, window, config
    )
    repeated = _random_refractive_perturbation(
        x, y, weights, window, config
    )

    np.testing.assert_array_equal(values, repeated)
    core = (
        (x >= config.core_x_min)
        & (x <= config.core_x_max)
        & (y >= config.core_y_min)
        & (y <= config.core_y_max)
    )
    core_weight_sum = np.sum(weights[core])
    weighted_mean = np.sum(weights[core] * values[core]) / core_weight_sum
    weighted_rms = np.sqrt(
        np.sum(weights[core] * values[core] ** 2) / core_weight_sum
    )
    assert abs(weighted_mean) < 1.0e-14
    np.testing.assert_allclose(weighted_rms, config.refractive_rms, rtol=1.0e-14)
    assert np.all(values[window == 0.0] == 0.0)

    permutation = np.random.default_rng(9).permutation(x.size)
    permuted = _random_refractive_perturbation(
        x[permutation],
        y[permutation],
        weights[permutation],
        window[permutation],
        config,
    )
    np.testing.assert_allclose(
        permuted,
        values[permutation],
        rtol=5.0e-14,
        atol=1.0e-17,
    )
    changed_seed = _random_refractive_perturbation(
        x,
        y,
        weights,
        window,
        replace(config, random_seed=config.random_seed + 1),
    )
    assert not np.array_equal(values, changed_seed)


def test_publication_default_resolves_declared_local_parameter():
    config = _default_config(smoke=False)
    _validate_config(config)
    leaf_side_length = (
        2.0 * config.root_half_extent / 2 ** (config.nlevels - 1)
    )
    assert config.wave_number * leaf_side_length == 0.75
    assert config.fmm_backend == "fmmlib"
    assert config.fmm_order == 40
    assert config.fmm_order_mode == "helmholtz"
    assert config.fmm_order_padding == 20
    assert config.refractive_rms == 0.05
    assert config.gaussian_bumps == 2048
    assert config.medium_generator == "gaussian-bumps"
    assert config.preconditioner == "none"
    assert config.preconditioner_ordering == "target"
    assert config.preconditioner_degree == 1
    domain_width_wavelengths = (
        config.wave_number * 2.0 * config.root_half_extent / (2.0 * np.pi)
    )
    core_length_wavelengths = (
        config.wave_number * (config.core_x_max - config.core_x_min) / (2.0 * np.pi)
    )
    assert domain_width_wavelengths > 30.0
    assert core_length_wavelengths > 17.0


def test_helmholtz_fmm_order_increases_on_coarse_levels():
    config = _default_config(smoke=False)
    calibration_tree = type("Tree", (), {"root_extent": 8.0})()
    calibration_orders = [
        _fmm_level_order(config, calibration_tree, level) for level in range(8)
    ]

    assert calibration_orders == [88, 54, 40, 40, 40, 40, 40, 40]

    default_tree = type("Tree", (), {"root_extent": 16.0})()
    default_orders = [
        _fmm_level_order(config, default_tree, level) for level in range(9)
    ]
    assert default_orders == [156, 88, 54, 40, 40, 40, 40, 40, 40]


def test_source_frozen_palette_interpolates_quadratics_exactly():
    nodes = _chebyshev_lobatto_nodes(8.0, 12.0, 3)
    values = np.linspace(8.0, 12.0, 17)
    weights = _lagrange_interpolation_weights(values, nodes)

    np.testing.assert_allclose(np.sum(weights, axis=0), 1.0, atol=1.0e-15)
    np.testing.assert_allclose(
        np.sum(weights * nodes[:, np.newaxis] ** 2, axis=0),
        values**2,
        atol=5.0e-14,
    )


def test_neighbor_csr_restriction_preserves_selected_rows():
    targets = np.array([4, 7, 9], dtype=np.int32)
    starts = np.array([0, 2, 5, 6], dtype=np.int32)
    lists = np.array([1, 4, 2, 7, 8, 9], dtype=np.int32)
    active = np.zeros(10, dtype=bool)
    active[[4, 9]] = True

    restricted_targets, restricted_starts, restricted_lists = (
        _restrict_neighbor_csr(targets, starts, lists, active)
    )

    np.testing.assert_array_equal(restricted_targets, [4, 9])
    np.testing.assert_array_equal(restricted_starts, [0, 2, 3])
    np.testing.assert_array_equal(restricted_lists, [1, 4, 9])


def test_right_preconditioned_operator_applies_factors_in_right_order():
    matrix = np.array([[2.0, 1.0], [0.0, 3.0]])
    right_factor = np.array([[1.0, -1.0], [0.5, 2.0]])

    class MatrixOperator:
        shape = (2, 2)

        def __init__(self, array):
            self.array = array

        def __call__(self, vector):
            return self.array @ vector

    wrapped = _RightPreconditionedOperator(
        MatrixOperator(matrix),
        MatrixOperator(right_factor),
    )
    vector = np.array([0.25, -2.0])

    np.testing.assert_allclose(wrapped(vector), matrix @ right_factor @ vector)
    np.testing.assert_allclose(wrapped.recover(vector), right_factor @ vector)


def test_fixed_polynomial_matches_matrix_powers():
    matrix = np.array([[0.2, -0.1], [0.3, 0.15]])
    vector = np.array([1.5, -0.25])
    degree = 3
    damping = 0.7

    actual = _apply_fixed_polynomial(
        lambda operand: matrix @ operand,
        vector,
        degree,
        damping,
    )
    expected = vector.copy()
    power_action = vector.copy()
    for _ in range(degree):
        power_action = matrix @ power_action
        expected += damping * power_action

    np.testing.assert_allclose(actual, expected)


def test_preconditioner_configuration_validation():
    config = _default_config(smoke=False)
    _validate_config(replace(config, preconditioner="source-frozen"))

    try:
        _validate_config(replace(config, preconditioner_damping=1.1))
    except ValueError as exc:
        assert "damping" in str(exc)
    else:
        raise AssertionError("invalid damping was accepted")

    try:
        _validate_config(replace(config, preconditioner_degree=0))
    except ValueError as exc:
        assert "degree" in str(exc)
    else:
        raise AssertionError("invalid polynomial degree was accepted")
