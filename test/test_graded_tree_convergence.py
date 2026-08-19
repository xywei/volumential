"""Queue-free tests for the graded-tree manufactured-solution convergence
driver (E9): the refinement-indicator math, observed-order and
asymptotic-regime bookkeeping, matched-error DOF accounting, and the
grading/modeling-gap validation gates."""

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
def gtc():
    return _load_benchmark_module("graded_tree_convergence")


def test_field_tuple_is_consistent(gtc):
    assert len(set(gtc.FIELDS)) == len(gtc.FIELDS)
    for name in (
        "refinement",
        "n_targets",
        "leaf_level_histogram_json",
        "max_adjacent_leaf_level_difference",
        "grading_status",
        "source_omitted_abs_fraction",
        "weighted_rel_l2_vs_analytic",
        "observed_order_dof",
        "in_asymptotic_regime",
        "ladder_verdict",
    ):
        assert name in gtc.FIELDS


def test_source_mixture_is_a_single_tight_gaussian(gtc):
    mixture = gtc._source_mixture(200.0, (0.11, 0.07, -0.05))
    assert len(mixture.components) == 1
    component = mixture.components[0]
    assert component.alpha == 200.0
    assert component.center == (0.11, 0.07, -0.05)
    assert component.amplitude == 1.0


def test_refinement_eta_scales_like_local_interpolation_error(gtc):
    values = np.array([1.0, 1.0, 0.5])
    levels = np.array([2, 3, 2])
    eta = gtc._refinement_eta(values, levels, 1.0, 3)
    # same value one level deeper: h halves, eta drops by 2**q_order
    assert eta[1] == pytest.approx(eta[0] / 8.0)
    # same level, half the value: eta halves
    assert eta[2] == pytest.approx(eta[0] / 2.0)
    assert np.all(eta > 0.0)


def test_observed_order_recovers_synthetic_rates(gtc):
    # error = DOF**(-1) in 3D is order 3
    assert gtc._observed_order(1.0e-2, 1.25e-3, 100, 800, 3) == pytest.approx(
        3.0
    )
    # non-increasing DOF or non-positive errors are not computable
    assert gtc._observed_order(1.0e-2, 1.0e-3, 800, 800, 3) == ""
    assert gtc._observed_order(0.0, 1.0e-3, 100, 800, 3) == ""
    assert gtc._observed_order(float("nan"), 1.0e-3, 100, 800, 3) == ""


def test_ladder_summary_detects_asymptotic_regime(gtc):
    # clean third-order ladder in 3D: DOF x8, error /8 per rung
    errors = [1.0e-2, 1.25e-3, 1.5625e-4, 1.953125e-5]
    dofs = [512, 4096, 32768, 262144]
    summary = gtc._ladder_summary(errors, dofs, 3)
    assert summary["observed_orders"][0] == ""
    assert summary["observed_orders"][1] == pytest.approx(3.0)
    assert summary["in_asymptotic_regime"] is True
    assert summary["verdict"].startswith("asymptotic")


def test_ladder_summary_states_the_limitation(gtc):
    # pre-asymptotic: the observed order is still drifting
    summary = gtc._ladder_summary(
        [1.0e-2, 5.0e-3, 1.0e-4], [512, 4096, 32768], 3
    )
    assert summary["in_asymptotic_regime"] is False
    assert summary["verdict"].startswith("limitation")
    assert "extend the ladder" in summary["verdict"]

    short = gtc._ladder_summary([1.0e-2, 1.25e-3], [512, 4096], 3)
    assert short["in_asymptotic_regime"] is False
    assert "extend the ladder" in short["verdict"]

    with pytest.raises(ValueError, match="equal length"):
        gtc._ladder_summary([1.0], [1, 2], 3)


def _ladder_rows(errors_and_dofs, refinement):
    return [
        {"weighted_rel_l2_vs_analytic": error, "n_targets": dof}
        for error, dof in errors_and_dofs
    ]


def test_matched_error_dof_advantage_brackets_either_curve(gtc):
    uniform = _ladder_rows(
        [(1.0e-2, 1000), (1.0e-3, 8000), (1.0e-4, 64000)], "uniform"
    )
    adaptive_inside = _ladder_rows(
        [(1.0e-2, 1000), (1.0e-3, 2000)], "adaptive"
    )
    advantage = gtc._matched_error_dof_advantage(uniform, adaptive_inside)
    assert advantage == pytest.approx(4.0)

    # the adaptive ladder ends below the whole uniform error range: match
    # the finest uniform error on the adaptive curve instead
    adaptive_below = _ladder_rows(
        [(1.0e-3, 2000), (1.0e-5, 4000)], "adaptive"
    )
    advantage = gtc._matched_error_dof_advantage(uniform, adaptive_below)
    assert advantage == pytest.approx(64000 / (2000 * 2 ** 0.5))

    disjoint = _ladder_rows([(1.0e-9, 10)], "adaptive")
    assert gtc._matched_error_dof_advantage(uniform, disjoint) == ""
    assert gtc._matched_error_dof_advantage([], adaptive_inside) == ""


def _row(gtc, **overrides):
    row = {field: "" for field in gtc.FIELDS}
    row.update(
        {
            "case_id": "graded-laplace3d-q2-adaptive-a1",
            "mode": "smoke",
            "refinement": "adaptive",
            "adapt_steps": 1,
            "min_leaf_level": 2,
            "max_leaf_level": 3,
            "max_adjacent_leaf_level_difference": 1,
            "n_cross_level_list1_interactions": 12,
            "grading_status": "graded",
            "source_omitted_abs_fraction": 1.0e-14,
            "weighted_rel_l2_vs_analytic": 1.0e-3,
            "n_targets": 512,
        }
    )
    row.update(overrides)
    return row


def test_validate_rows_passes_clean_ladders(gtc):
    gtc._validate_rows(
        [
            _row(
                gtc,
                case_id="graded-laplace3d-q2-uniform-l3",
                refinement="uniform",
                adapt_steps="",
                min_leaf_level=2,
                max_leaf_level=2,
                n_cross_level_list1_interactions=0,
                grading_status="uniform",
                weighted_rel_l2_vs_analytic=1.0e-2,
            ),
            _row(
                gtc,
                case_id="graded-laplace3d-q2-uniform-l4",
                refinement="uniform",
                adapt_steps="",
                min_leaf_level=3,
                max_leaf_level=3,
                n_cross_level_list1_interactions=0,
                grading_status="uniform",
                weighted_rel_l2_vs_analytic=1.0e-3,
            ),
            _row(
                gtc,
                case_id="graded-laplace3d-q2-adaptive-a0",
                adapt_steps=0,
                min_leaf_level=2,
                max_leaf_level=2,
                n_cross_level_list1_interactions=0,
                grading_status="uniform",
                weighted_rel_l2_vs_analytic=1.0e-2,
            ),
            _row(gtc),
        ]
    )


def test_validate_rows_requires_genuine_grading(gtc):
    with pytest.raises(RuntimeError, match="genuine grading"):
        gtc._validate_rows(
            [
                _row(
                    gtc,
                    min_leaf_level=3,
                    max_leaf_level=3,
                    grading_status="uniform",
                )
            ]
        )
    with pytest.raises(RuntimeError, match="cross-level"):
        gtc._validate_rows(
            [_row(gtc, n_cross_level_list1_interactions=0)]
        )


def test_validate_rows_gates_balance_and_modeling_gap(gtc):
    with pytest.raises(RuntimeError, match="not 2:1 balanced"):
        gtc._validate_rows(
            [_row(gtc, max_adjacent_leaf_level_difference=2)]
        )
    with pytest.raises(RuntimeError, match="modeling-gap gate"):
        gtc._validate_rows(
            [_row(gtc, source_omitted_abs_fraction=1.0e-6)]
        )
    with pytest.raises(RuntimeError, match="non-finite or non-positive"):
        gtc._validate_rows(
            [_row(gtc, weighted_rel_l2_vs_analytic=float("nan"))]
        )


def test_validate_rows_requires_ladder_convergence(gtc):
    with pytest.raises(RuntimeError, match="did not converge"):
        gtc._validate_rows(
            [
                _row(
                    gtc,
                    case_id="graded-laplace3d-q2-adaptive-a1",
                    weighted_rel_l2_vs_analytic=1.0e-3,
                ),
                _row(
                    gtc,
                    case_id="graded-laplace3d-q2-adaptive-a2",
                    adapt_steps=2,
                    weighted_rel_l2_vs_analytic=2.0e-3,
                ),
            ]
        )
