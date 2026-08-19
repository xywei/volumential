"""Queue-free tests for the windowed-assembled benchmark strategies:

- the split-parameter sweep's windowed table-provisioning strategy (E1), and
- the Keller--Segel continuation's windowed strategy helpers (E4).
"""

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
import sys
from pathlib import Path

import numpy as np
import pytest


_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


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
def sweep():
    return _load_benchmark_module("split_parameter_sweep")


@pytest.fixture(scope="module")
def ks():
    return _load_benchmark_module("keller_segel_continuation")


# {{{ E1: split_parameter_sweep windowed strategy

def test_windowed_theta_parsing_defaults_and_validation(sweep):
    assert sweep._parse_windowed_thetas(None, "smoke") == list(
        sweep.DEFAULT_SMOKE_WINDOWED_THETAS
    )
    assert sweep._parse_windowed_thetas(None, "full") == list(
        sweep.DEFAULT_FULL_WINDOWED_THETAS
    )
    assert sweep._parse_windowed_thetas("none", "smoke") == []
    assert sweep._parse_windowed_thetas("1,2.5", "smoke") == [1.0, 2.5]
    with pytest.raises(ValueError):
        sweep._parse_windowed_thetas("1,-2", "smoke")
    with pytest.raises(ValueError):
        sweep._parse_windowed_thetas("1,1", "smoke")


def _windowed_row(sweep, **overrides):
    row = {field: "" for field in sweep.FIELDS}
    row.update(
        {
            "case_id": "yukawa2d-lambda4-windowed-theta1",
            "mode": "smoke",
            "table_strategy": "windowed_assembled",
            "theta": 1.0,
            "window_theta": 16.0,
            "windowed_status": "ok",
            "windowed_refusal": "",
            "classical_probe_status": "certified-truncation-only",
            "classical_probe_detail": "",
            "rel_l2_error": 1.0e-8,
        }
    )
    row.update(overrides)
    return row


def test_windowed_row_validation_passes_clean_rows(sweep):
    sweep._validate_windowed_rows(
        [
            _windowed_row(sweep),
            # refusal beyond the declaration is an expected certificate
            # outcome, not a failure
            _windowed_row(
                sweep,
                theta=32.0,
                windowed_status="refused",
                windowed_refusal="RKEWindowCoverageError: outside",
                rel_l2_error="",
            ),
            # online rows are not the windowed validator's business
            {"table_strategy": "online_split"},
        ]
    )


def test_windowed_row_validation_rejects_failures(sweep):
    with pytest.raises(RuntimeError, match="failed"):
        sweep._validate_windowed_rows(
            [
                _windowed_row(
                    sweep,
                    windowed_status="failed",
                    windowed_refusal="RuntimeError: boom",
                )
            ]
        )


def test_windowed_row_validation_rejects_refusal_inside_declaration(sweep):
    with pytest.raises(RuntimeError, match="inside the declaration"):
        sweep._validate_windowed_rows(
            [
                _windowed_row(
                    sweep,
                    theta=8.0,
                    windowed_status="refused",
                    windowed_refusal="RKEWindowConditioningError: bad",
                )
            ]
        )


def test_windowed_row_validation_gates_small_theta_agreement(sweep):
    with pytest.raises(RuntimeError, match="disagrees"):
        sweep._validate_windowed_rows(
            [_windowed_row(sweep, rel_l2_error=0.5)]
        )
    # large theta rows are not gated on the direct reference agreement
    sweep._validate_windowed_rows(
        [_windowed_row(sweep, theta=16.0, rel_l2_error=0.5)]
    )


def test_windowed_row_validation_rejects_probe_failure(sweep):
    with pytest.raises(RuntimeError, match="probe failed"):
        sweep._validate_windowed_rows(
            [
                _windowed_row(
                    sweep,
                    classical_probe_status="failed",
                    classical_probe_detail="RuntimeError: boom",
                )
            ]
        )


def test_windowed_row_base_covers_all_fields(sweep):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    row = sweep._windowed_row_base(
        mode="smoke",
        kernel="Yukawa",
        parameter_name="lambda",
        parameter=4.0,
        theta=1.0,
        window_theta=16.0,
        p_star=6,
        chan_orders=(48, 61),
        direct_build_config=DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=8,
            radial_quad_order=21,
        ),
        q_order=2,
        nlevels=2,
        fmm_order=8,
        repeat_count=1,
        classical_probe={
            "kind": "truncation",
            "status": "certified-truncation-only",
            "detail": "",
            "n_terms": 12,
            "condition_number": "",
            "probe_s": 0.01,
        },
    )
    assert set(row) == set(sweep.FIELDS)
    assert row["table_strategy"] == "windowed_assembled"
    assert row["reference_path"] == "direct_fixed_parameter_table"


def test_classical_truncation_probe_refuses_at_declaration_edge(sweep):
    # theta = 16 at the smoke geometry (level-2 tables, box extent 0.5):
    # the polynomial-completion series cannot certify 1e-11 within the
    # 60-term budget, which is the expected refusal mode.
    refused = sweep._classical_certificate_probe(
        queue=None,
        cache_path=Path("/nonexistent-not-touched"),
        kernel="Yukawa",
        q_order=2,
        parameter=32.0,
        source_box_level=2,
        tolerance=1.0e-11,
        probe_kind="truncation",
    )
    assert refused["status"] == "refused"
    assert "RKETruncationError" in refused["detail"]

    certified = sweep._classical_certificate_probe(
        queue=None,
        cache_path=Path("/nonexistent-not-touched"),
        kernel="Yukawa",
        q_order=2,
        parameter=2.0,
        source_box_level=2,
        tolerance=1.0e-11,
        probe_kind="truncation",
    )
    assert certified["status"] == "certified-truncation-only"
    assert int(certified["n_terms"]) >= 1

    off = sweep._classical_certificate_probe(
        queue=None,
        cache_path=Path("/nonexistent-not-touched"),
        kernel="Yukawa",
        q_order=2,
        parameter=2.0,
        source_box_level=2,
        tolerance=1.0e-11,
        probe_kind="off",
    )
    assert off["status"] == "skipped"

# }}}


# {{{ E4: keller_segel_continuation windowed strategy helpers

def test_provision_windowed_yukawa_table_roundtrip(ks, tmp_path):
    """Queue-free provisioning: assemble, register, load via the standard
    table-manager path, with the ok/refused taxonomy."""
    root_extent = 4.0
    level = 3
    box_extent = root_extent * 0.5**level
    lam = 4.0 / box_extent  # theta = 4

    family_cache = tmp_path / "family.sqlite"
    registered_cache = tmp_path / "registered.sqlite"
    table, info = ks._provision_windowed_yukawa_table(
        None,
        family_cache,
        registered_cache,
        2,
        lam,
        level,
        root_extent,
        16.0,
        4,
    )
    assert info["status"] == "ok"
    assert table is not None
    assert table.build_method == "ExternalAssembly"
    assert info["assemble_s"] > 0.0
    assert info["register_s"] > 0.0
    assert info["load_s"] > 0.0
    assert float(info["condition_number"]) < 10.0

    # beyond the declaration: certificate refusal, not a failure
    refused_table, refused_info = ks._provision_windowed_yukawa_table(
        None,
        family_cache,
        registered_cache,
        2,
        2.5 * 16.0 / box_extent,
        level,
        root_extent,
        16.0,
        4,
    )
    assert refused_table is None
    assert refused_info["status"] == "refused"
    assert "RKEWindowCoverageError" in refused_info["detail"]


def test_compare_checkpoints_matches_shared_times_only(ks):
    weights = np.full(4, 0.25)
    baseline = {
        "checkpoints": [
            (0.5, np.array([1.0, 2.0, 3.0, 4.0])),
            (1.0, np.array([2.0, 3.0, 4.0, 5.0])),
        ],
        "weights": weights,
    }
    windowed = {
        "checkpoints": [
            (0.5, np.array([1.0, 2.0, 3.0, 4.0])),
            # the windowed run stopped early: no t = 1.0 entry
        ],
        "weights": weights,
    }
    rows, agreement = ks._compare_checkpoints(
        baseline, windowed, case_id="w", baseline_case_id="b", mode="smoke"
    )
    assert len(rows) == 1
    assert rows[0]["checkpoint_time"] == 0.5
    assert rows[0]["windowed_vs_direct_weighted_rel_l2"] == 0.0
    assert agreement == [{"time": 0.5, "rel_l2": 0.0}]

    perturbed = {
        "checkpoints": [(0.5, np.array([1.0, 2.0, 3.0, 4.0]) * 1.01)],
        "weights": weights,
    }
    rows, agreement = ks._compare_checkpoints(
        baseline, perturbed, case_id="w", baseline_case_id="b", mode="smoke"
    )
    assert rows[0]["windowed_vs_direct_weighted_rel_l2"] == pytest.approx(
        0.01, rel=1e-12
    )


def test_apply_pair_outcome_groups_by_strategy(ks):
    def summary(strategy, regime, moment_ratio):
        return {
            "strategy": strategy,
            "regime": regime,
            "second_moment_ratio": moment_ratio,
            "admissible": 1,
            "trend_criterion_pass": 1,
            "pair_outcome_pass": 0,
            "pair_moment_ratio_separation": "",
        }

    rows = [
        summary("direct", "below_8pi_reference", 1.10),
        summary("direct", "above_8pi_reference", 0.95),
        summary("windowed", "below_8pi_reference", 1.09),
        summary("windowed", "above_8pi_reference", 0.96),
    ]
    ks._apply_pair_outcome(rows)
    assert all(row["pair_outcome_pass"] == 1 for row in rows)
    assert rows[0]["pair_moment_ratio_separation"] == pytest.approx(0.15)
    assert rows[2]["pair_moment_ratio_separation"] == pytest.approx(0.13)


def test_step_and_summary_field_tuples_are_consistent(ks):
    assert len(set(ks.STEP_FIELDS)) == len(ks.STEP_FIELDS)
    assert len(set(ks.SUMMARY_FIELDS)) == len(ks.SUMMARY_FIELDS)
    assert len(set(ks.CHECKPOINT_FIELDS)) == len(ks.CHECKPOINT_FIELDS)
    for name in (
        "strategy",
        "binding_constraint",
        "windowed_status",
        "windowed_assemble_s",
    ):
        assert name in ks.STEP_FIELDS
    for name in (
        "strategy",
        "window_theta",
        "go_no_go_binding_verdict",
        "binding_constraint_histogram_json",
        "checkpoint_agreement_json",
        "windowed_strategy_total_s",
    ):
        assert name in ks.SUMMARY_FIELDS

# }}}
