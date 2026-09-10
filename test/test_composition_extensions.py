"""Queue-free tests for the adaptive-split-composition evidence extensions
(E5): Helmholtz composition rows, the explicit quadrature-policy knob, and
the windowed-assembled composition row with its taxonomy and gates."""

__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

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

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"

# The committed composition CSV column prefix; appended extension columns
# must never move these.
COMMITTED_FIELD_PREFIX = (
    "case_id",
    "mode",
    "kernel",
    "q_order",
    "initial_nlevels",
    "adapt_steps",
    "parameter",
    "split_order",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
    "n_targets",
    "min_leaf_level",
    "max_leaf_level",
    "leaf_level_histogram_json",
    "max_adjacent_leaf_level_difference",
    "n_list1_interactions",
    "n_cross_level_list1_interactions",
    "cross_level_list1_fraction",
    "list1_source_target_level_pair_histogram_json",
    "populated_source_levels_json",
    "max_theta",
    "path_mismatch_norm_definition",
    "rke_vs_direct_weighted_rel_l2",
    "rke_vs_direct_linf",
    "direct_wall_s",
    "rke_wall_s",
    "direct_table_count",
    "direct_table_build_s",
    "direct_table_payload_bytes",
    "rke_base_table_build_s",
    "rke_base_table_payload_bytes",
    "rke_channel_table_count",
    "rke_channel_table_build_s",
    "rke_channel_table_payload_bytes",
    "rke_total_table_payload_bytes",
)


def _load_benchmark_module(name):
    if name in sys.modules:
        return sys.modules[name]
    if str(_BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(_BENCHMARK_DIR))
    spec = importlib.util.spec_from_file_location(
        name, _BENCHMARK_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def composition():
    return _load_benchmark_module("adaptive_split_composition")


def test_committed_csv_column_prefix_is_unchanged(composition):
    assert composition.FIELDS[: len(COMMITTED_FIELD_PREFIX)] == (
        COMMITTED_FIELD_PREFIX
    )
    assert len(set(composition.FIELDS)) == len(composition.FIELDS)


def test_windowed_fields_are_a_field_subset(composition):
    assert set(composition.WINDOWED_FIELDS) <= set(composition.FIELDS)
    for name in (
        "window_theta",
        "windowed_status",
        "windowed_refusal",
        "windowed_max_condition_number",
        "windowed_vs_direct_weighted_rel_l2",
    ):
        assert name in composition.FIELDS


def test_quadrature_policy_resolution(composition):
    assert composition._resolve_quadrature_policy("auto", "smoke") == "default"
    assert composition._resolve_quadrature_policy("auto", "full") == (
        "high-accuracy"
    )
    assert composition._resolve_quadrature_policy(
        "high-accuracy", "smoke"
    ) == "high-accuracy"
    assert composition._resolve_quadrature_policy("default", "full") == (
        "default"
    )
    with pytest.raises(ValueError, match="unknown quadrature policy"):
        composition._resolve_quadrature_policy("fast", "smoke")


def test_kernel_build_configs_mirror_the_sweep_policy(composition):
    direct, channel = composition._kernel_build_configs(
        "Yukawa", 4, high_accuracy=True
    )
    assert (direct.regular_quad_order, direct.radial_quad_order) == (80, 320)
    assert (channel.regular_quad_order, channel.radial_quad_order) == (48, 160)

    direct_default, channel_default = composition._kernel_build_configs(
        "Yukawa", 4, high_accuracy=False
    )
    assert (
        direct_default.regular_quad_order,
        direct_default.radial_quad_order,
    ) == (16, 40)
    assert (
        channel_default.regular_quad_order,
        channel_default.radial_quad_order,
    ) == (16, 40)

    # The high-accuracy policy is the Yukawa logarithmic-singularity policy;
    # Helmholtz stays on the sweep's default rule in either policy.
    for high_accuracy in (False, True):
        direct_h, channel_h = composition._kernel_build_configs(
            "Helmholtz", 4, high_accuracy=high_accuracy
        )
        assert (
            direct_h.regular_quad_order,
            direct_h.radial_quad_order,
        ) == (16, 40)
        assert (
            channel_h.regular_quad_order,
            channel_h.radial_quad_order,
        ) == (16, 40)

    with pytest.raises(ValueError, match="unknown kernel"):
        composition._kernel_build_configs("Stokes", 4, high_accuracy=False)


def _windowed_row(composition, **overrides):
    row = dict.fromkeys(composition.FIELDS, "")
    row.update(
        {
            "case_id": "yukawa2d-q3-l2-a1-lam2-windowed",
            "mode": "smoke",
            "kernel": "Yukawa",
            "max_theta": 0.5,
            "table_strategy": "windowed_assembled",
            "window_theta": 16.0,
            "windowed_status": "ok",
            "windowed_refusal": "",
            "windowed_vs_direct_weighted_rel_l2": 1.0e-7,
        }
    )
    row.update(overrides)
    return row


def test_windowed_composition_validation_passes_clean_rows(composition):
    composition._validate_windowed_composition_rows(
        [
            _windowed_row(composition),
            # refusal beyond the declaration is an expected certificate
            # outcome, not a failure
            _windowed_row(
                composition,
                max_theta=32.0,
                windowed_status="refused",
                windowed_refusal="RKEWindowCoverageError: outside",
                windowed_vs_direct_weighted_rel_l2="",
            ),
            # online rows are not the windowed validator's business
            {"table_strategy": "online_split"},
        ]
    )


def test_windowed_composition_validation_rejects_failures(composition):
    with pytest.raises(RuntimeError, match="failed"):
        composition._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition,
                    windowed_status="failed",
                    windowed_refusal="RuntimeError: boom",
                )
            ]
        )


def test_windowed_composition_rejects_refusal_inside_declaration(composition):
    with pytest.raises(RuntimeError, match="inside the declaration"):
        composition._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition,
                    max_theta=8.0,
                    windowed_status="refused",
                    windowed_refusal="RKEWindowConditioningError: bad",
                )
            ]
        )


def test_windowed_composition_gates_small_theta_agreement(composition):
    with pytest.raises(RuntimeError, match="disagrees"):
        composition._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition, windowed_vs_direct_weighted_rel_l2=0.5
                )
            ]
        )
    # large-theta rows are not gated on the direct reference agreement
    composition._validate_windowed_composition_rows(
        [
            _windowed_row(
                composition,
                max_theta=8.0,
                windowed_vs_direct_weighted_rel_l2=0.5,
            )
        ]
    )


def _online_row(composition, **overrides):
    row = dict.fromkeys(composition.FIELDS, "")
    row.update(
        {
            "case_id": "yukawa2d-q4-l4-a2-lam8-p2",
            "mode": "full",
            "kernel": "Yukawa",
            "q_order": 4,
            "initial_nlevels": 4,
            "adapt_steps": 2,
            "parameter": 8.0,
            "split_order": 2,
            "quadrature_policy": "high-accuracy",
            "table_strategy": "online_split",
            "rke_vs_direct_weighted_rel_l2": 1.0e-8,
        }
    )
    row.update(overrides)
    return row


def test_split_order_gate_requires_p2_improvement(composition):
    rows = [
        _online_row(
            composition, split_order=1, rke_vs_direct_weighted_rel_l2=1.0e-4
        ),
        _online_row(
            composition, split_order=2, rke_vs_direct_weighted_rel_l2=5.0e-8
        ),
        _online_row(
            composition, split_order=3, rke_vs_direct_weighted_rel_l2=5.0e-8
        ),
    ]
    composition._validate_split_order_convergence(rows)

    stalled = [
        _online_row(
            composition, split_order=1, rke_vs_direct_weighted_rel_l2=1.0e-4
        ),
        _online_row(
            composition, split_order=2, rke_vs_direct_weighted_rel_l2=5.0e-5
        ),
    ]
    with pytest.raises(RuntimeError, match="three orders of magnitude"):
        composition._validate_split_order_convergence(stalled)


def test_split_order_gate_rejects_material_degradation(composition):
    rows = [
        _online_row(
            composition, split_order=2, rke_vs_direct_weighted_rel_l2=1.0e-8
        ),
        _online_row(
            composition, split_order=3, rke_vs_direct_weighted_rel_l2=1.0e-6
        ),
    ]
    with pytest.raises(RuntimeError, match="materially degraded"):
        composition._validate_split_order_convergence(rows)


def test_split_order_gate_scope(composition):
    # smoke rows, default-policy rows, Helmholtz rows, and windowed rows are
    # all outside the gate
    rows = [
        _online_row(
            composition,
            mode="smoke",
            split_order=1,
            rke_vs_direct_weighted_rel_l2=1.0e-4,
        ),
        _online_row(
            composition,
            mode="smoke",
            split_order=2,
            rke_vs_direct_weighted_rel_l2=9.0e-5,
        ),
        _online_row(
            composition,
            quadrature_policy="default",
            split_order=1,
            rke_vs_direct_weighted_rel_l2=1.0e-4,
        ),
        _online_row(
            composition,
            quadrature_policy="default",
            split_order=2,
            rke_vs_direct_weighted_rel_l2=9.0e-5,
        ),
        _online_row(
            composition,
            kernel="Helmholtz",
            split_order=1,
            rke_vs_direct_weighted_rel_l2=1.0e-4,
        ),
        _online_row(
            composition,
            kernel="Helmholtz",
            split_order=2,
            rke_vs_direct_weighted_rel_l2=9.0e-5,
        ),
        _windowed_row(composition),
    ]
    composition._validate_split_order_convergence(rows)


def test_kernel_parameter_tags(composition):
    assert composition.KERNEL_PARAMETER_TAGS == {
        "Yukawa": "lam",
        "Helmholtz": "k",
    }


# {{{ windowed provisioning failure taxonomy


def _windowed_kwargs_2d(tmp_path):
    return {
        "cache_dir": tmp_path,
        "kernel": "Yukawa",
        "q_order": 2,
        "initial_nlevels": 3,
        "adapt_steps": 1,
        "parameter": 2.0,
        "source_levels": [3],
        "tree_root_extent": 2.0,
        "window_theta": 16.0,
        "windowed_p_star": 4,
        "windowed_chan_orders": (48, 61),
    }


def _stub_windowed_assembly(composition, monkeypatch):
    import volumential.rke_table_assembly as rke

    monkeypatch.setattr(
        composition,
        "_prepare_windowed_family",
        lambda **k: {"build_s": 0.0, "was_cold": False},
    )
    monkeypatch.setattr(
        rke,
        "assemble_windowed_parameterized_table",
        lambda *a, **k: (object(), {
            "condition_number": 2.0, "smooth_quad_order": 4,
        }),
    )


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("registration refused"),
        OSError("cache is unwritable"),
        sqlite3.OperationalError("database is locked"),
    ],
)
def test_registration_failures_become_a_failed_row(
    composition, tmp_path, monkeypatch, exc
):
    """Registration and reload belong to the assembly's taxonomy.

    ``main()`` writes the CSV only after every case completes, so an
    exception escaping here discards every measurement already taken, not
    just this row's.
    """
    _stub_windowed_assembly(composition, monkeypatch)

    def _raise(**kwargs):
        raise exc

    monkeypatch.setattr(
        composition, "_register_and_load_windowed_table", _raise
    )

    row = composition._run_windowed_composition(
        None, **_windowed_kwargs_2d(tmp_path)
    )

    assert row["windowed_status"] == "failed"
    assert type(exc).__name__ in row["windowed_refusal"]
    assert str(exc) in row["windowed_refusal"]
    assert "_tables" not in row
    # sqlite3's exceptions are the ones an OSError-only handler misses
    assert not issubclass(sqlite3.Error, OSError)


def test_partial_registration_metrics_reach_the_failed_row(
    composition, tmp_path, monkeypatch
):
    """Registration that completed before the reload failed is reported."""
    _stub_windowed_assembly(composition, monkeypatch)

    def _register_then_fail(**kwargs):
        exc = RuntimeError("cache reopen failed")
        exc.partial_windowed_transfer = {
            "register_s": 4.0,
            "register_payload_bytes": 8000,
            "load_s": 0.0,
            "load_payload_bytes": 0,
        }
        raise exc

    monkeypatch.setattr(
        composition, "_register_and_load_windowed_table", _register_then_fail
    )

    row = composition._run_windowed_composition(
        None, **_windowed_kwargs_2d(tmp_path)
    )

    assert row["windowed_status"] == "failed"
    assert row["windowed_register_s"] == pytest.approx(4.0)
    assert row["windowed_register_payload_bytes"] == 8000
    assert row["windowed_table_count"] == 0


# }}}
