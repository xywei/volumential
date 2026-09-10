"""Tests for the E3/E6 extensions of the break-even driver: the lazy-direct
provisioning strategy, the operation-counter summary columns, and the
per-phase share columns."""

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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_break_even():
    path = _REPOSITORY_ROOT / "benchmarks" / "break_even_validation.py"
    spec = importlib.util.spec_from_file_location("break_even_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
    finally:
        sys.path[:] = previous_sys_path
    return module


def test_eager_provisioning_reproduces_committed_levels():
    module = _load_break_even()
    assert module._resolve_direct_levels(
        smoke=True, provisioning="eager", nlevels=2
    ) == [1, 2]
    assert module._resolve_direct_levels(
        smoke=False, provisioning="eager", nlevels=5
    ) == [0, 1, 2, 3, 4, 5]


def test_lazy_provisioning_builds_only_the_touched_level():
    module = _load_break_even()
    # only the leaf level owns List 1 work on the uniform benchmark tree
    assert module._resolve_direct_levels(
        smoke=True, provisioning="lazy", nlevels=2
    ) == [2]
    assert module._resolve_direct_levels(
        smoke=False, provisioning="lazy", nlevels=5
    ) == [5]


def test_unknown_provisioning_is_rejected():
    module = _load_break_even()
    with pytest.raises(ValueError, match="provisioning"):
        module._resolve_direct_levels(
            smoke=True, provisioning="opportunistic", nlevels=2
        )


def test_remainder_term_count_averages_over_every_parameter(monkeypatch):
    """The operation numerator must span the timing denominator's set.

    The profiled RKE seconds are a mean over every parameter, so pricing
    the remainder at the first parameter's term count would put an
    operation numerator and a timing denominator from different sweeps
    into the same share.
    """
    import numpy as np

    import volumential.opcounters as opcounters
    from volumential.expansion_wrangler_fpnd import (
        _HelmholtzSplitSeriesRemainderKernel,
    )

    module = _load_break_even()

    class _Wrangler:
        def __init__(self, nmax):
            self._kernel = _HelmholtzSplitSeriesRemainderKernel(
                2, 4.0, 0.0, 3, nmax
            )

        def _get_helmholtz_split_remainder_kernel(self):
            return self._kernel

    wranglers = [_Wrangler(nmax) for nmax in (10, 20, 30)]
    counts = [
        module._remainder_terms_per_pair(wrangler) for wrangler in wranglers
    ]
    # the parameter really does change the count, so averaging matters
    assert len(set(counts)) == len(counts)

    monkeypatch.setattr(
        opcounters,
        "fmm_stage_operation_counts_from_traversal",
        lambda queue, traversal, wrangler: dict.fromkeys(
            (*module.PHASE_FAR_STAGES, "far_total"), 0
        ),
    )
    monkeypatch.setattr(
        opcounters, "expansion_coefficient_counts", lambda wrangler: ([1], [1])
    )
    monkeypatch.setattr(
        opcounters, "nearfield_point_pairs", lambda queue, traversal: 100
    )
    monkeypatch.setattr(
        module,
        "_split_correction_operation_counts",
        lambda **kwargs: {
            "extra_table_fmas": 0,
            "remainder_pair_evals": 100,
            "beta_p2p_pair_evals": 0,
            "smooth_interp_fmas": 0,
            "smooth_sources_per_box": 4,
            "remainder_terms_per_pair": counts[0],
            "status": "base_quadrature",
        },
    )

    columns = module._phase_operation_counts(
        queue=None,
        traversal=None,
        direct_wrangler=None,
        rke_wrangler=wranglers[0],
        rke_wranglers=wranglers,
        q_order=2,
        smooth_quad_order=None,
        nmax_by_parameter=[10, 20, 30],
        split_table_count=1,
    )

    mean_terms = float(np.mean(counts))
    assert columns[
        "ops_phase_split_correction_remainder_terms_per_pair"
    ] == pytest.approx(mean_terms)
    assert columns[
        "ops_phase_split_correction_remainder_terms_by_parameter"
    ] == ";".join(f"{float(c):g}" for c in counts)
    # ... and the evaluations use the mean, not the first parameter's count
    assert columns[
        "ops_phase_split_correction_remainder_term_evals"
    ] == pytest.approx(100 * mean_terms)
    assert columns[
        "ops_phase_split_correction_remainder_term_evals"
    ] != pytest.approx(100 * counts[0])


def test_summary_fields_extend_the_committed_layout():
    module = _load_break_even()
    fields = list(module.SUMMARY_FIELDS)
    # append-only contract: the historical columns keep their positions
    assert fields.index("mode") == 0
    assert fields.index("benchmark_total_s") < fields.index(
        "direct_provisioning"
    )
    for name in (
        "direct_provisioning",
        "ops_reduced_entries_per_table",
        "ops_direct_build_routing",
        "ops_direct_singular_node_evals",
        "ops_direct_special_function",
        "ops_rke_channel_singular_node_evals",
        "ops_nearfield_point_pairs_per_solve",
        "ops_split_series_nmax_per_parameter",
    ):
        assert name in fields


# {{{ per-phase share columns (E6)

def _phase_summary_row(module, *, with_seconds=True):
    """A summary row shaped like a finished run, with round numbers."""
    row = {
        "mode": "smoke",
        "kernel": "Yukawa",
        "direct_provisioning": "eager",
        "phase_counting_rule": module.PHASE_COUNTING_RULE,
        "phase_profile_repeat_count": 2,
        "phase_profile_solves_per_strategy": "direct:2;rke:2",
        "phase_profile_nested_phases": "",
        "ops_phase_far_total": 0,
        "ops_phase_fmm_multipole_coefficients_by_level": "7;7",
        "ops_phase_fmm_local_coefficients_by_level": "9;9",
        "ops_phase_nearfield_table_apply_direct": 1000,
        "ops_phase_nearfield_table_apply_rke": 1000,
        "ops_phase_split_correction_rke": 9000.0,
        "ops_phase_split_correction_extra_table_fmas": 2000,
        "ops_phase_split_correction_remainder_pair_evals": 4000,
        "ops_phase_split_correction_remainder_term_evals": 4000.0,
        "ops_phase_split_correction_remainder_terms_per_pair": 1,
        "ops_phase_split_correction_remainder_terms_by_parameter": "1",
        "ops_phase_split_correction_beta_p2p_pair_evals": 2000,
        "ops_phase_split_correction_smooth_interp_fmas": 1000,
        "ops_phase_split_smooth_sources_per_box": 16,
        "ops_phase_split_correction_status": "interpolated_smooth_quadrature",
        "ops_phase_recombination_per_solve": 0,
        "ops_phase_solve_total_direct": 1000,
        "ops_phase_solve_total_rke": 10000.0,
        "ops_phase_setup_direct_table_build": 500,
        "ops_phase_setup_channel_family_build": 250,
        "ops_phase_setup_recombination_flops": 0,
        "s_phase_setup_direct_table_build": 8.0,
        "s_phase_setup_direct_table_cache_load": 2.0,
        "s_phase_setup_channel_family_build": 3.0,
        "s_phase_setup_channel_family_cache_load": 1.0,
        "s_phase_setup_recombination": 0.0,
    }
    for stage in module.PHASE_FAR_STAGES:
        row[f"ops_phase_far_{stage}"] = 100
        row["ops_phase_far_total"] += 100
    for strategy in module.PHASE_STRATEGIES:
        for stage in module.PHASE_FAR_STAGES:
            row[f"s_phase_far_{stage}_{strategy}"] = 0.1 if with_seconds else 0.0
        row[f"s_phase_far_total_{strategy}"] = (
            0.1 * len(module.PHASE_FAR_STAGES) if with_seconds else 0.0
        )
        row[f"s_phase_nearfield_table_apply_{strategy}"] = 0.2
        row[f"s_phase_split_correction_{strategy}"] = (
            1.0 if strategy == "rke" else 0.0
        )
        row[f"s_phase_other_{strategy}"] = 0.1
        row[f"s_phase_solve_total_{strategy}"] = (
            0.7 + 0.2 + 0.1 + (1.0 if strategy == "rke" else 0.0)
        )
    return row


def test_phase_columns_extend_the_layout_without_moving_anything():
    module = _load_break_even()
    fields = list(module.SUMMARY_FIELDS)
    # append-only: every pre-E6 column keeps its position
    assert fields.index("mode") == 0
    assert fields.index("direct_provisioning") < fields.index(
        "ops_reduced_entries_per_table"
    )
    last_pre_e6 = fields.index(
        "ops_split_remainder_term_flops_per_solve_per_parameter"
    )
    for name in module.PHASE_OPS_FIELDS + module.PHASE_SECONDS_FIELDS:
        assert fields.index(name) > last_pre_e6
    # and no column is emitted twice
    assert len(fields) == len(set(fields))


def test_phase_column_names_carry_the_required_prefixes():
    module = _load_break_even()
    for name in module.PHASE_OPS_FIELDS:
        assert name.startswith("ops_phase_") or name.startswith("phase_")
    for name in module.PHASE_SECONDS_FIELDS:
        assert name.startswith("s_phase_")
    # every far-field stage is priced and timed for both strategies
    for stage in module.PHASE_FAR_STAGES:
        assert f"ops_phase_far_{stage}" in module.PHASE_OPS_FIELDS
        for strategy in module.PHASE_STRATEGIES:
            assert (
                f"s_phase_far_{stage}_{strategy}"
                in module.PHASE_SECONDS_FIELDS
            )


def test_phase_rows_partition_each_strategy_into_shares():
    module = _load_break_even()
    row = _phase_summary_row(module)
    phase_rows = module._phase_rows(row)

    assert {entry["scope"] for entry in phase_rows} == {"solve", "setup"}
    assert set(phase_rows[0]) == set(module.PHASE_FIELDS)

    for strategy in module.PHASE_STRATEGIES:
        solve_rows = [
            entry
            for entry in phase_rows
            if entry["scope"] == "solve" and entry["strategy"] == strategy
        ]
        # seven far stages plus table apply, correction, recombination, other
        assert len(solve_rows) == len(module.PHASE_FAR_STAGES) + 4
        seconds_shares = [
            entry["seconds_share"] for entry in solve_rows
        ]
        assert sum(seconds_shares) == pytest.approx(1.0)
        ops_shares = [
            entry["ops_share"] for entry in solve_rows
            if entry["ops_share"] != ""
        ]
        assert sum(ops_shares) == pytest.approx(1.0)


def test_direct_strategy_rows_carry_no_split_correction_work():
    module = _load_break_even()
    phase_rows = module._phase_rows(_phase_summary_row(module))
    (direct_correction,) = [
        entry
        for entry in phase_rows
        if entry["scope"] == "solve"
        and entry["strategy"] == "direct"
        and entry["phase"] == "split_correction"
    ]
    assert direct_correction["ops"] == 0
    assert direct_correction["seconds"] == 0.0


def test_setup_rows_are_per_run_and_split_build_from_cache_load():
    module = _load_break_even()
    phase_rows = module._phase_rows(_phase_summary_row(module))
    setup = {
        (entry["strategy"], entry["phase"]): entry
        for entry in phase_rows
        if entry["scope"] == "setup"
    }
    assert setup[("direct", "direct_table_build")]["ops"] == 500
    assert setup[("direct", "direct_table_build")]["unit"] == "per_run"
    assert setup[("direct", "direct_table_build")]["seconds_share"] == (
        pytest.approx(0.8)
    )
    assert setup[("rke", "recombination")]["ops"] == 0


def test_setup_and_solve_operations_declare_different_currencies():
    module = _load_break_even()
    phase_rows = module._phase_rows(_phase_summary_row(module))
    by_scope = {entry["scope"]: entry["ops_unit"] for entry in phase_rows}
    assert by_scope == module.PHASE_OPS_UNITS
    assert by_scope["solve"] != by_scope["setup"]
    # a setup row never offers an operation share, so the two currencies
    # cannot be summed into one denominator by accident
    assert all(
        entry["ops_share"] == ""
        for entry in phase_rows
        if entry["scope"] == "setup"
    )


def test_an_unpriced_phase_withholds_the_whole_operation_partition():
    module = _load_break_even()
    row = _phase_summary_row(module)
    # what _split_correction_operation_counts writes when it cannot
    # interrogate the wrangler: the dominant phase of the split path has no
    # count at all
    row["ops_phase_split_correction_rke"] = ""
    row["ops_phase_solve_total_rke"] = ""
    phase_rows = module._phase_rows(row)

    rke_solve = [
        entry
        for entry in phase_rows
        if entry["scope"] == "solve" and entry["strategy"] == "rke"
    ]
    # no share is offered for any rke phase: far / table shares divided by
    # their own sum would be a confident number for the wrong denominator
    assert all(entry["ops_share"] == "" for entry in rke_solve)
    # the seconds partition is independent and survives
    assert sum(
        float(entry["seconds_share"]) for entry in rke_solve
    ) == pytest.approx(1.0)
    # and the direct path, whose phases are all priced, is unaffected
    direct_shares = [
        float(entry["ops_share"])
        for entry in phase_rows
        if entry["scope"] == "solve"
        and entry["strategy"] == "direct"
        and entry["ops_share"] != ""
    ]
    assert sum(direct_shares) == pytest.approx(1.0)


def test_zero_seconds_denominator_yields_blank_shares_not_a_crash():
    module = _load_break_even()
    row = _phase_summary_row(module)
    for strategy in module.PHASE_STRATEGIES:
        for name in module.PHASE_SECOND_NAMES:
            row[f"s_phase_{name}_{strategy}"] = 0.0
    phase_rows = module._phase_rows(row)
    solve_rows = [
        entry for entry in phase_rows if entry["scope"] == "solve"
    ]
    assert all(entry["seconds_share"] == "" for entry in solve_rows)
    # the operation shares are unaffected by an absent timing
    assert any(entry["ops_share"] != "" for entry in solve_rows)


def test_counting_rule_string_names_every_component():
    module = _load_break_even()
    rule = module.PHASE_COUNTING_RULE
    for token in (
        "far=",
        "nearfield=",
        "split_correction=",
        "smooth_interp=",
        "recombination=",
    ):
        assert token in rule


def test_smooth_interp_is_priced_as_the_tensor_product_that_runs():
    module = _load_break_even()
    # d = 2: interp_mat @ v costs q_s * q * q, then @ interp_mat.T costs
    # q_s * q * q_s, exactly what _interpolate_box_values_to_smooth_quad does
    assert module._tensor_product_interp_fmas(dim=2, q=4, q_smooth=8) == 384
    assert module._tensor_product_interp_fmas(dim=1, q=4, q_smooth=8) == 32
    assert module._tensor_product_interp_fmas(dim=3, q=3, q_smooth=6) == (
        6 * 27 + 36 * 9 + 216 * 3
    )
    # and it is strictly below the dense q_s**d x q**d reading it replaces
    for dim, q, q_smooth in ((2, 4, 8), (3, 3, 6)):
        assert module._tensor_product_interp_fmas(
            dim=dim, q=q, q_smooth=q_smooth
        ) < q_smooth**dim * q**dim


def test_smooth_interp_price_degenerates_when_orders_match():
    module = _load_break_even()
    # q_smooth == q is the non-interpolating path; the formula still returns
    # the axis-by-axis cost rather than something undefined
    assert module._tensor_product_interp_fmas(dim=2, q=4, q_smooth=4) == (
        4 * 16 + 16 * 4
    )

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
