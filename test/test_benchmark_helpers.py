import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark(name):
    path = _REPOSITORY_ROOT / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _accuracy_row(*, q_order, fmm_order, error):
    return {
        "path": "canonical_rescaled",
        "q_order": q_order,
        "n_levels": 2,
        "fmm_order": fmm_order,
        "regular_quad_order": fmm_order,
        "radial_quad_order": 2 * fmm_order + 1,
        "h_max": 0.25,
        "weighted_rel_l2_vs_exact": error,
        "h_observed_order_vs_exact": "",
        "q_error_ratio_vs_previous": "",
        "q_log_error_slope_vs_exact": "",
    }


def test_q_convergence_rates_require_fixed_solver_orders():
    module = _load_benchmark("accuracy_preservation")
    changed_order = _accuracy_row(q_order=2, fmm_order=8, error=0.5)
    fixed_order_coarse = _accuracy_row(q_order=3, fmm_order=10, error=0.25)
    fixed_order_fine = _accuracy_row(q_order=4, fmm_order=10, error=0.125)
    rows = [changed_order, fixed_order_coarse, fixed_order_fine]

    module._add_convergence_rates(rows)

    assert fixed_order_coarse["q_error_ratio_vs_previous"] == ""
    assert fixed_order_fine["q_error_ratio_vs_previous"] == pytest.approx(0.5)


@pytest.mark.parametrize(("repeat_count", "direct_levels", "message"), [
    (0, [2], "repeat_count"),
    (1, [0, 1], "direct_levels"),
])
def test_split_benchmark_validates_direct_call_invariants(
    tmp_path, repeat_count, direct_levels, message
):
    module = _load_benchmark("split_parameter_sweep")

    with pytest.raises(ValueError, match=message):
        module.run_benchmark(
            mode="smoke",
            backend="cpu",
            cache_dir=tmp_path,
            q_order=2,
            nlevels=2,
            fmm_order=8,
            split_orders=[1],
            helmholtz_k=[4.0],
            yukawa_lam=[],
            direct_levels=direct_levels,
            repeat_count=repeat_count,
        )


def test_split_benchmark_full_yukawa_accuracy_policy():
    module = _load_benchmark("split_parameter_sweep")

    direct = module._yukawa_reference_build_config(4, high_accuracy=True)
    channels = module._split_channel_build_config(4, high_accuracy=True)

    assert (direct.regular_quad_order, direct.radial_quad_order) == (80, 320)
    assert (channels.regular_quad_order, channels.radial_quad_order) == (48, 160)
    assert module._split_smooth_quad_order(4, 1, high_accuracy=True) == 4
    assert module._split_smooth_quad_order(4, 2, high_accuracy=True) == 8
    assert module._split_smooth_quad_order(4, 3, high_accuracy=True) == 8
    assert module._split_smooth_quad_order(4, 3, high_accuracy=False) == 4


def test_split_benchmark_rejects_full_yukawa_order_plateau():
    module = _load_benchmark("split_parameter_sweep")
    common = {
        "mode": "full",
        "kernel": "Yukawa",
        "parameter_value": 8.0,
    }

    module._validate_yukawa_order_convergence([
        {**common, "split_order": 1, "rel_l2_error": 3.0e-6},
        {**common, "split_order": 2, "rel_l2_error": 2.0e-11},
        {**common, "split_order": 3, "rel_l2_error": 7.0e-12},
    ])

    with pytest.raises(RuntimeError, match="did not improve"):
        module._validate_yukawa_order_convergence([
            {**common, "split_order": 1, "rel_l2_error": 1.19446e-5},
            {**common, "split_order": 2, "rel_l2_error": 1.16870e-5},
            {**common, "split_order": 3, "rel_l2_error": 1.16778e-5},
        ])


@pytest.mark.full_accuracy
def test_split_benchmark_full_yukawa_order_convergence(tmp_path):
    module = _load_benchmark("split_parameter_sweep")

    rows = module.run_benchmark(
        mode="full",
        backend="pocl-cpu",
        cache_dir=tmp_path,
        q_order=4,
        nlevels=3,
        fmm_order=16,
        split_orders=[1, 2, 3],
        helmholtz_k=[],
        yukawa_lam=[2.0],
        direct_levels=[3],
        repeat_count=1,
    )
    errors = {int(row["split_order"]): row["rel_l2_error"] for row in rows}

    assert errors[2] < 1.0e-3 * errors[1]
    assert errors[3] < 0.5 * errors[2]
    assert errors[3] < 1.0e-9


def test_keller_segel_critical_profile_is_mass_normalized():
    module = _load_benchmark("keller_segel_continuation")
    axis = np.linspace(-1.0, 1.0, 33)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    weights = np.ones(coords.shape[1])

    density = module._initial_density(
        coords,
        weights,
        mass=8.0 * np.pi,
        profile="critical",
        profile_scale=0.3,
        cutoff_inner_radius=0.7,
        cutoff_outer_radius=0.9,
    )

    assert np.sum(weights * density) == pytest.approx(8.0 * np.pi)
    assert density[np.argmin(np.sum(coords**2, axis=0))] == np.max(density)
    assert np.all(density[np.sqrt(np.sum(coords**2, axis=0)) >= 0.9] == 0.0)


def test_keller_segel_endpoint_planner_avoids_short_terminal_step():
    module = _load_benchmark("keller_segel_continuation")

    regular = module._plan_time_step(0.008, 0.005, 0.001, 2.0)
    assert regular == pytest.approx((1.0 / 16.0**2, False, False))

    terminal = module._plan_time_step(0.004, 0.005, 0.001, 2.0)
    assert terminal == pytest.approx((0.004, True, False))

    adjusted = module._plan_time_step(0.009, 0.005, 0.003, 2.0)
    assert adjusted == pytest.approx((0.0045, False, True))

    adaptive_endpoint = module._plan_time_step(
        0.000775871,
        0.000397,
        0.0003014,
        2.0**0.125,
    )
    assert adaptive_endpoint == pytest.approx(
        (0.000775871 / 2.0, False, True)
    )

    assert module._plan_time_step(0.0015, 0.0012, 0.001, 2.0) is None
