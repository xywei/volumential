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
        dim=2,
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
        fmm_order_rule="fixed",
        fmm_order_floor=8,
        far_field_status="pinned",
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
        dim=2,
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
        dim=2,
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
        dim=2,
        kernel="Yukawa",
        q_order=2,
        parameter=2.0,
        source_box_level=2,
        tolerance=1.0e-11,
        probe_kind="off",
    )
    assert off["status"] == "skipped"

# }}}


# {{{ E1b: three-dimensional path and the resolved FMM-order rule

def test_field_set_is_unique_and_carries_far_field_columns(sweep):
    assert len(set(sweep.FIELDS)) == len(sweep.FIELDS)
    for name in (
        "fmm_order_rule",
        "fmm_order_floor",
        "fmm_expansion_radius",
        "far_field_status",
        "implied_reference_norm",
    ):
        assert name in sweep.FIELDS
    # the historical 106 columns keep their positions
    assert sweep.FIELDS[:106] == sweep.FIELDS[:-5]
    assert sweep.FIELDS[0] == "case_id"
    assert sweep.FIELDS[3] == "dim"


def test_dimension_validation(sweep):
    assert sweep._require_dimension(2) == 2
    assert sweep._require_dimension(3) == 3
    for bad in (1, 4, 0, -2):
        with pytest.raises(ValueError, match="dim must be"):
            sweep._require_dimension(bad)


def test_box_extent_and_theta_mapping_matches_committed_geometries(sweep):
    # 2D committed runs: nlevels = 3 gives leaf width 1/4 (k = 4 theta) and
    # nlevels = 5 gives 1/16 (k = 16 theta).
    assert sweep._box_extent(3) == pytest.approx(0.25)
    assert sweep._box_extent(5) == pytest.approx(0.0625)
    # the 3D windowed table sweep declares source level 2 at root extent 2
    assert sweep._box_extent(2) == pytest.approx(0.5)
    for nlevels, theta in ((5, 16.0), (5, 0.25), (3, 6.0), (2, 16.0)):
        parameter = theta / sweep._box_extent(nlevels)
        assert parameter * sweep._box_extent(nlevels) == pytest.approx(theta)


def test_uniform_target_count_matches_committed_runs(sweep):
    # 2D q = 4: nlevels 3 -> 256 targets, nlevels 5 -> 4096 (tbl:strategy-cost)
    assert sweep._uniform_target_count(2, 4, 3) == 256
    assert sweep._uniform_target_count(2, 4, 5) == 4096
    # 3D q = 3: the committed field demo at nlevels 4 carries 13824 nodes
    assert sweep._uniform_target_count(3, 3, 4) == 13824
    assert sweep._uniform_target_count(3, 3, 5) == 110592
    with pytest.raises(ValueError):
        sweep._uniform_target_count(3, 0, 4)


def test_fmm_expansion_radius_is_the_level_one_half_diagonal(sweep):
    assert sweep._fmm_expansion_radius(2) == pytest.approx(np.sqrt(2.0) / 4.0)
    assert sweep._fmm_expansion_radius(3) == pytest.approx(np.sqrt(3.0) / 4.0)


def test_resolved_fmm_order_reproduces_the_2d_order_ladder(sweep):
    """The committed 2D order-scaled run's band -> order mapping."""
    # nlevels = 3, k = 4 theta over theta = 0.25 .. 16
    expected = {1: 16, 2: 16, 4: 16, 8: 16, 16: 16, 24: 16,
                32: 20, 48: 26, 64: 33}
    for k, order in expected.items():
        assert sweep._resolved_fmm_order(2, k, floor=16) == order
    # nlevels = 5, k = 16 theta: the orders the manuscript quotes for the
    # matched-configuration run
    expected_l5 = {4: 16, 8: 16, 16: 16, 32: 20, 64: 33,
                   96: 45, 128: 57, 192: 81, 256: 105}
    for k, order in expected_l5.items():
        assert sweep._resolved_fmm_order(2, k, floor=16) == order


def test_resolved_fmm_order_is_monotone_and_floored(sweep):
    previous = 0
    for k in range(0, 300, 7):
        order = sweep._resolved_fmm_order(3, k, floor=12)
        assert order >= 12
        assert order >= previous
        previous = order
    with pytest.raises(ValueError):
        sweep._resolved_fmm_order(3, 1.0, floor=0)
    with pytest.raises(ValueError):
        sweep._resolved_fmm_order(3, float("inf"), floor=12)


def test_prescribed_fmm_order_only_scales_the_oscillatory_kernel(sweep):
    assert sweep._prescribed_fmm_order(
        3, "Yukawa", 256.0, floor=12, rule="resolved"
    ) == 12
    assert sweep._prescribed_fmm_order(
        3, "Helmholtz", 256.0, floor=12, rule="fixed"
    ) == 12
    assert sweep._prescribed_fmm_order(
        3, "Helmholtz", 64.0, floor=12, rule="resolved"
    ) == sweep._resolved_fmm_order(3, 64.0, floor=12)
    with pytest.raises(ValueError, match="fmm order rule"):
        sweep._prescribed_fmm_order(
            3, "Helmholtz", 1.0, floor=12, rule="nonsense"
        )


def test_implied_reference_norm(sweep):
    assert sweep._implied_reference_norm(9.06e-5, 6.234e-5) == pytest.approx(
        1.4533, rel=1.0e-3
    )
    assert sweep._implied_reference_norm(0.0, 0.0) == ""
    assert sweep._implied_reference_norm("", "") == ""
    assert sweep._implied_reference_norm(1.0, float("nan")) == ""


def test_far_field_resolution_failures_flag_the_documented_pathologies(sweep):
    clean = {
        "case_id": "helmholtz2d-k4-windowed-theta0.25",
        "kernel": "Helmholtz",
        "far_field_status": "resolved_by_rule",
        "fmm_order": 16,
        "rel_l2_error": 1.726e-7,
        "linf_error": 1.453e-7,
    }
    assert sweep._far_field_resolution_failures([clean]) == []

    zeroed = {**clean, "rel_l2_error": 0.0, "linf_error": 0.0,
              "far_field_status": "pinned"}
    (message,) = sweep._far_field_resolution_failures([zeroed])
    assert "exactly zero" in message

    diverged = {**clean, "rel_l2_error": 3.926e-8, "linf_error": 9.06e-5,
                "far_field_status": "pinned"}
    (message,) = sweep._far_field_resolution_failures([diverged])
    assert "implied reference-field norm" in message

    # The overflow end of the same pathology: the implied-norm ratio is not
    # computable for a non-finite column, so it must be caught on its own.
    for bad in (float("nan"), float("inf")):
        overflowed = {**clean, "rel_l2_error": bad, "linf_error": bad,
                      "far_field_status": "pinned"}
        (message,) = sweep._far_field_resolution_failures([overflowed])
        assert "non-finite" in message
        half = {**clean, "linf_error": bad, "far_field_status": "pinned"}
        (message,) = sweep._far_field_resolution_failures([half])
        assert "non-finite" in message

    # Yukawa rows and refused rows are not subject to the check
    assert sweep._far_field_resolution_failures(
        [{**zeroed, "kernel": "Yukawa"}]
    ) == []
    assert sweep._far_field_resolution_failures(
        [{**zeroed, "far_field_status": "refused_order_cap"}]
    ) == []


def test_three_dimensional_quadrature_policies(sweep):
    direct_tight = sweep._direct_build_config(
        3, "Helmholtz", 3, high_accuracy=True
    )
    direct_loose = sweep._direct_build_config(
        3, "Yukawa", 3, high_accuracy=False
    )
    channels = sweep._channel_build_config(3, "Yukawa", 3, high_accuracy=True)
    assert (
        direct_tight.regular_quad_order,
        direct_tight.radial_quad_order,
    ) == (24, 61)
    assert (
        direct_loose.regular_quad_order,
        direct_loose.radial_quad_order,
    ) == (16, 45)
    assert (channels.regular_quad_order, channels.radial_quad_order) == (12, 35)
    # 2D policies are unchanged
    yukawa_2d = sweep._direct_build_config(2, "Yukawa", 4, high_accuracy=True)
    helmholtz_2d = sweep._direct_build_config(
        2, "Helmholtz", 4, high_accuracy=True
    )
    assert (
        yukawa_2d.regular_quad_order,
        yukawa_2d.radial_quad_order,
    ) == (80, 320)
    assert (
        helmholtz_2d.regular_quad_order,
        helmholtz_2d.radial_quad_order,
    ) == (16, 40)
    assert sweep._smooth_quad_order(3, 3, 1, high_accuracy=True) == 3
    assert sweep._smooth_quad_order(3, 3, 2, high_accuracy=True) == 6
    assert sweep._smooth_quad_order(2, 4, 2, high_accuracy=True) == 8


def test_three_dimensional_channel_order_default(sweep):
    assert sweep.DEFAULT_WINDOWED_CHAN_ORDERS[2] == (48, 61)
    assert sweep.DEFAULT_WINDOWED_CHAN_ORDERS[3] == (20, 61)


def test_sources_are_dimension_generic_and_preserve_the_2d_forms(sweep):
    rng = np.random.default_rng(20260908)
    coords2 = rng.uniform(-0.5, 0.5, size=(2, 11))
    x, y = coords2
    assert np.allclose(
        sweep._gaussian_source_host(coords2),
        np.exp(-35.0 * ((x + 0.11) ** 2 + (y - 0.07) ** 2)),
    )
    source2, exact2 = sweep._helmholtz_manufactured_source_and_exact(
        coords2, 4.0
    )
    alpha = 80.0
    r2 = x * x + y * y
    assert np.allclose(exact2, np.exp(-alpha * r2))
    assert np.allclose(
        source2, (4 * alpha - 4 * alpha**2 * r2 - 16.0) * np.exp(-alpha * r2)
    )

    coords3 = rng.uniform(-0.5, 0.5, size=(3, 7))
    gaussian3 = sweep._gaussian_source_host(coords3)
    assert gaussian3.shape == (7,)
    assert np.all(gaussian3 > 0.0)
    source3, exact3 = sweep._helmholtz_manufactured_source_and_exact(
        coords3, 4.0
    )
    r3sq = (coords3**2).sum(axis=0)
    assert np.allclose(exact3, np.exp(-alpha * r3sq))
    assert np.allclose(
        source3,
        (6 * alpha - 4 * alpha**2 * r3sq - 16.0) * np.exp(-alpha * r3sq),
    )


def test_windowed_row_base_tags_the_dimension(sweep):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    row = sweep._windowed_row_base(
        mode="full",
        dim=3,
        kernel="Helmholtz",
        parameter_name="k",
        parameter=16.0,
        theta=1.0,
        window_theta=16.0,
        p_star=6,
        chan_orders=sweep.DEFAULT_WINDOWED_CHAN_ORDERS[3],
        direct_build_config=DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=24,
            radial_quad_order=61,
        ),
        q_order=3,
        nlevels=5,
        fmm_order=19,
        fmm_order_rule="resolved",
        fmm_order_floor=12,
        far_field_status="resolved_by_rule",
        repeat_count=5,
        classical_probe={
            "kind": "full",
            "status": "certified",
            "detail": "",
            "n_terms": 20,
            "condition_number": 3.0,
            "probe_s": 1.0,
        },
    )
    assert set(row) == set(sweep.FIELDS)
    assert row["dim"] == 3
    assert row["case_id"] == "helmholtz3d-k16-windowed-theta1"
    assert row["fmm_order"] == 19
    assert row["fmm_order_floor"] == 12
    assert row["fmm_expansion_radius"] == pytest.approx(np.sqrt(3.0) / 4.0)
    assert row["far_field_status"] == "resolved_by_rule"
    assert row["windowed_chan_regular_order"] == 20


def test_windowed_validation_skips_agreement_for_refused_order_rows(sweep):
    row = _windowed_row(
        sweep,
        far_field_status="refused_order_cap",
        rel_l2_error="",
        windowed_refusal="far-field order 45 exceeds the cap 20",
    )
    sweep._validate_windowed_rows([row])


def test_resolved_rule_is_refused_for_the_fixed_parameter_sweep(sweep, tmp_path):
    with pytest.raises(ValueError, match="windowed theta ladder"):
        sweep.run_benchmark(
            mode="smoke",
            backend="pocl-cpu",
            cache_dir=tmp_path,
            dim=3,
            q_order=2,
            nlevels=2,
            fmm_order=8,
            split_orders=[1],
            helmholtz_k=[4.0],
            yukawa_lam=[],
            direct_levels=[2],
            repeat_count=1,
            fmm_order_rule="resolved",
        )


def test_zero_wave_number_is_refused_as_a_helmholtz_row(sweep, tmp_path):
    """``k = 0`` is Laplace, and every Helmholtz diagnostic here assumes it is not.

    ``_far_field_resolution_failures`` reads two exactly-zero error columns as
    a diverged-then-cancelled far field, and the resolved-order rule
    prescribes an order from ``k a``; at ``k = 0`` the direct and split paths
    solve the same non-oscillatory problem, so both premises are void.
    """
    with pytest.raises(ValueError, match="degenerates to Laplace"):
        sweep.run_benchmark(
            mode="smoke",
            backend="this-backend-does-not-exist",
            cache_dir=tmp_path / "never-created",
            dim=3,
            q_order=2,
            nlevels=2,
            fmm_order=8,
            split_orders=[1],
            helmholtz_k=[4.0, 0.0],
            yukawa_lam=[],
            direct_levels=[2],
            repeat_count=1,
        )


def test_zero_decay_rate_is_refused_as_a_yukawa_row(sweep, tmp_path):
    with pytest.raises(ValueError, match="degenerates to Laplace"):
        sweep.run_benchmark(
            mode="smoke",
            backend="this-backend-does-not-exist",
            cache_dir=tmp_path / "never-created",
            dim=3,
            q_order=2,
            nlevels=2,
            fmm_order=8,
            split_orders=[1],
            helmholtz_k=[],
            yukawa_lam=[0.0],
            direct_levels=[2],
            repeat_count=1,
        )


def test_non_finite_wave_numbers_are_refused(sweep, tmp_path):
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            sweep.run_benchmark(
                mode="smoke",
                backend="this-backend-does-not-exist",
                cache_dir=tmp_path / "never-created",
                dim=3,
                q_order=2,
                nlevels=2,
                fmm_order=8,
                split_orders=[1],
                helmholtz_k=[bad],
                yukawa_lam=[],
                direct_levels=[2],
                repeat_count=1,
            )


def test_an_unknown_backend_is_refused_without_an_opencl_platform(sweep):
    """``--backend`` is validated before the ICD loader is touched.

    A misspelled backend must say so rather than surface whatever the driver
    says, and on a machine with no OpenCL platform at all ``cl.get_platforms``
    raises ``LogicError(PLATFORM_NOT_FOUND_KHR)`` before any later check could
    run.  The fake below has no platforms *and* raises on the call, so
    reaching the ``ValueError`` proves the check runs first.
    """

    class _NoPlatforms:
        @staticmethod
        def get_platforms():
            raise AssertionError("device enumeration must not be reached")

    with pytest.raises(ValueError, match="backend must be one of"):
        sweep._select_opencl_device(_NoPlatforms(), "this-backend-does-not-exist")
    # ... and the accepted values still pass the check and go on to enumerate
    for backend in sweep.SUPPORTED_BACKENDS:
        with pytest.raises(AssertionError, match="must not be reached"):
            sweep._select_opencl_device(_NoPlatforms(), backend)


def test_positive_parameters_still_pass_validation(sweep, tmp_path):
    """The guard must not reject the ordinary sweep it sits in front of.

    A nonexistent backend makes ``run_benchmark`` fail *after* validation, so
    reaching the backend error is the evidence that the parameter checks let a
    normal dispatch through.  That error is raised before any device
    enumeration (see the test above), so this stays queue-free.
    """
    with pytest.raises(ValueError, match="backend must be one of"):
        sweep.run_benchmark(
            mode="smoke",
            backend="this-backend-does-not-exist",
            cache_dir=tmp_path / "never-created",
            dim=3,
            q_order=2,
            nlevels=2,
            fmm_order=8,
            split_orders=[1],
            helmholtz_k=[4.0, 8.0],
            yukawa_lam=[4.0],
            direct_levels=[2],
            repeat_count=1,
        )


def test_max_fmm_order_must_not_undercut_the_floor(sweep, tmp_path):
    with pytest.raises(ValueError, match="max_fmm_order"):
        sweep.run_benchmark(
            mode="smoke",
            backend="pocl-cpu",
            cache_dir=tmp_path,
            dim=3,
            q_order=2,
            nlevels=2,
            fmm_order=12,
            split_orders=[1],
            helmholtz_k=[],
            yukawa_lam=[],
            direct_levels=[2],
            repeat_count=1,
            windowed_thetas=[1.0],
            max_fmm_order=8,
        )


def test_min_targets_is_refused_before_any_device_or_geometry(sweep, tmp_path):
    """The guard is a pure node count, so it fires without touching OpenCL.

    ``run_benchmark`` selects a device and builds geometry before it could
    measure a realized target count; a dispatch that asks for more targets
    than its (q, nlevels) can carry must fail immediately instead.
    """
    assert sweep._uniform_target_count(3, 2, 2) == 64
    with pytest.raises(RuntimeError, match="below the required minimum 4096"):
        sweep.run_benchmark(
            mode="smoke",
            backend="this-backend-does-not-exist",
            cache_dir=tmp_path / "never-created",
            dim=3,
            q_order=2,
            nlevels=2,
            fmm_order=8,
            split_orders=[1],
            helmholtz_k=[],
            yukawa_lam=[],
            direct_levels=[2],
            repeat_count=1,
            windowed_thetas=[1.0],
            min_targets=4096,
        )
    assert not (tmp_path / "never-created").exists()


def test_gate_failure_carries_its_rows_for_the_csv(sweep):
    """A failing gate must not cost a long run its measurements.

    ``run_benchmark`` wraps both post-run gates in ``_BenchmarkGateError``,
    which carries the complete rows so ``main`` can write the CSV before
    re-raising.  The class is a ``RuntimeError``, so callers that catch the
    historical exception type are unaffected.
    """
    assert issubclass(sweep._BenchmarkGateError, RuntimeError)
    rows = [_windowed_row(sweep)]
    error = sweep._BenchmarkGateError("gate says no", rows)
    assert error.rows is rows
    assert str(error) == "gate says no"

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
