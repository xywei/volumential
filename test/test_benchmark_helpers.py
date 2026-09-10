import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark(name):
    path = _REPOSITORY_ROOT / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
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


@pytest.mark.parametrize("name", [
    "adaptive_split_composition",
    "adaptive_timing_3d",
    "break_even_validation",
    "complex_bessel_parameterized",
    "complex_channel_closure",
    "derivative_log_preservation",
    "keller_segel_continuation",
    "rke_field_demo_3d",
])
def test_paper1_benchmark_module_imports(name):
    _load_benchmark(name)


def test_benchmark_loader_removes_failed_module(tmp_path, monkeypatch):
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()
    (benchmark_dir / "broken_import.py").write_text(
        "raise RuntimeError('broken import')\n", encoding="ascii"
    )
    monkeypatch.setattr(
        sys.modules[__name__], "_REPOSITORY_ROOT", tmp_path
    )

    with pytest.raises(RuntimeError, match="broken import"):
        _load_benchmark("broken_import")
    assert "broken_import" not in sys.modules


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
        "dim": 2,
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


def test_split_benchmark_yukawa_order_gate_is_dimension_aware():
    """The 3D gate is one order of magnitude, not three.

    The committed 3D field demo (data/benchmarks/rke-field-demo-3d) measures
    p=1 -> p=2 improvements of 1332x, 355x and 97x at lambda = 2, 4, 8, so the
    2D three-orders-of-magnitude gate would reject a healthy 3D run at the
    largest parameter.
    """
    module = _load_benchmark("split_parameter_sweep")
    common = {"mode": "full", "kernel": "Yukawa", "dim": 3}

    # the committed lambda = 8 triple, which the 2D gate would reject
    module._validate_yukawa_order_convergence([
        {**common, "parameter_value": 8.0, "split_order": 1,
         "rel_l2_error": 4.606911424028599e-4},
        {**common, "parameter_value": 8.0, "split_order": 2,
         "rel_l2_error": 4.747673362598595e-6},
        {**common, "parameter_value": 8.0, "split_order": 3,
         "rel_l2_error": 2.542248207701343e-6},
    ])
    with pytest.raises(RuntimeError, match="3D Yukawa"):
        module._validate_yukawa_order_convergence([
            {**common, "parameter_value": 8.0, "split_order": 1,
             "rel_l2_error": 4.6e-4},
            {**common, "parameter_value": 8.0, "split_order": 2,
             "rel_l2_error": 2.3e-4},
        ])


def test_rke_field_demo_full_accuracy_policy():
    module = _load_benchmark("rke_field_demo_3d")
    direct, channels = module._field_build_configs(3, high_accuracy=True)

    assert (direct.regular_quad_order, direct.radial_quad_order) == (16, 45)
    assert (channels.regular_quad_order, channels.radial_quad_order) == (12, 35)
    assert module._field_smooth_quad_order(3, 1, high_accuracy=True) == 3
    assert module._field_smooth_quad_order(3, 2, high_accuracy=True) == 6


def test_rke_field_demo_full_mode_requires_convergence_orders(tmp_path):
    module = _load_benchmark("rke_field_demo_3d")

    # Keep split-order validation ahead of OpenCL selection so normal CI stays
    # a cheap argument-policy test without requiring a local platform.
    with pytest.raises(ValueError, match="requires split orders 1, 2, and 3"):
        module.run_benchmark(
            mode="full",
            backend="pocl-cpu",
            cache_dir=tmp_path,
            q_order=3,
            nlevels=4,
            fmm_order=12,
            yukawa_lam=[4.0],
            split_orders=[1, 2],
            force_recompute=True,
        )


def test_rke_field_demo_rejects_full_order_plateau():
    module = _load_benchmark("rke_field_demo_3d")
    common = {"parameter": 4.0}
    module._validate_full_order_convergence([
        {
            **common,
            "split_order": 1,
            "split_vs_direct_weighted_rel_l2": 1.0e-3,
        },
        {
            **common,
            "split_order": 2,
            "split_vs_direct_weighted_rel_l2": 1.0e-7,
        },
        {
            **common,
            "split_order": 3,
            "split_vs_direct_weighted_rel_l2": 1.0e-10,
        },
    ])

    with pytest.raises(RuntimeError, match="did not converge"):
        module._validate_full_order_convergence([
            {
                **common,
                "split_order": 1,
                "split_vs_direct_weighted_rel_l2": 1.0e-3,
            },
            {
                **common,
                "split_order": 2,
                "split_vs_direct_weighted_rel_l2": 9.0e-4,
            },
            {
                **common,
                "split_order": 3,
                "split_vs_direct_weighted_rel_l2": 8.0e-4,
            },
        ])


def test_rke_field_demo_metadata_paths_are_sanitized(tmp_path, monkeypatch):
    module = _load_benchmark("rke_field_demo_3d")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "results" / "field.csv"

    assert module._public_path(output) == "results/field.csv"
    assert module._public_path(Path("/external/private/cache")) == "cache"
    assert module._public_argv(["driver.py", f"--out={output}"]) == [
        "driver.py",
        "--out=results/field.csv",
    ]
    assert module._public_argv(["driver.py", "--out=../private/field.csv"]) == [
        "driver.py",
        "--out=field.csv",
    ]


def test_complex_closure_exposes_nonfinite_candidate_entries():
    module = _load_benchmark("complex_channel_closure")
    count, max_abs, max_rel, reference_linf = module._mismatch_stats(
        np.array([1.0, 2.0]), np.array([1.0, np.nan])
    )

    assert count == 1
    assert np.isinf(max_abs)
    assert np.isinf(max_rel)
    assert reference_linf == pytest.approx(2.0)

    count, max_abs, max_rel, reference_linf = module._mismatch_stats(
        np.array([1.0, np.nan]), np.array([1.0, 2.0])
    )
    assert count == 1
    assert np.isinf(max_abs)
    assert np.isinf(max_rel)
    assert reference_linf == pytest.approx(1.0)


def test_complex_closure_full_roundoff_gate():
    module = _load_benchmark("complex_channel_closure")
    module._validate_closure_equivalence([{"max_rel_mismatch": 1.0e-13}])

    with pytest.raises(RuntimeError, match="roundoff gate"):
        module._validate_closure_equivalence([{"max_rel_mismatch": 1.0e-9}])


def test_complex_closure_rejects_nonfinite_alpha(monkeypatch):
    module = _load_benchmark("complex_channel_closure")
    monkeypatch.setattr(sys, "argv", ["complex_channel_closure.py", "--alpha=nan"])

    with pytest.raises(SystemExit, match="finite and positive"):
        module.main()


def test_adaptive_composition_validates_diagnostic_fields():
    module = _load_benchmark("adaptive_split_composition")
    diagnostics = {key: 0 for key in module.LEAF_DIAGNOSTIC_FIELDS}
    module._validate_diagnostic_fields(
        "leaf", diagnostics, module.LEAF_DIAGNOSTIC_FIELDS
    )

    diagnostics["unexpected"] = 0
    with pytest.raises(RuntimeError, match="unexpected"):
        module._validate_diagnostic_fields(
            "leaf", diagnostics, module.LEAF_DIAGNOSTIC_FIELDS
        )


def test_break_even_statistics_use_individual_solves():
    module = _load_benchmark("break_even_validation")
    rows = [
        {"strategy": "direct", "solve_wall_s": 1.0},
        {"strategy": "rke", "solve_wall_s": 10.0},
        {"strategy": "direct", "solve_wall_s": 3.0},
        {"strategy": "rke", "solve_wall_s": 14.0},
    ]

    assert module._solve_statistics(rows, "direct") == pytest.approx((2.0, 1.0))
    assert module._solve_statistics(rows, "rke") == pytest.approx((12.0, 2.0))


def test_break_even_model_requires_setup_advantage_and_solve_penalty():
    module = _load_benchmark("break_even_validation")
    common = {
        "parameter_count": 3,
        "direct_solve_mean_s": 1.0,
        "rke_solve_mean_s": 2.0,
    }

    assert module._modeled_break_even(
        **common, direct_build_s=20.0, rke_build_s=5.0
    ) == pytest.approx(5.0)
    assert module._modeled_break_even(
        **common, direct_build_s=5.0, rke_build_s=20.0
    ) == ""
    assert module._modeled_break_even(
        **{**common, "rke_solve_mean_s": 0.5},
        direct_build_s=20.0,
        rke_build_s=5.0,
    ) == ""


def test_keller_segel_vectorized_faces_match_boxwise_fluxes():
    module = _load_benchmark("keller_segel_continuation")
    transport = module.ConservativeDGTransport
    rho_minus = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rho_plus = rho_minus + 0.5
    velocity_minus = np.array([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]])
    velocity_plus = velocity_minus - 0.1
    minus_neighbors = np.array([-1, 0, 1])
    plus_neighbors = np.array([1, 2, -1])

    actual_minus, actual_plus = transport._numerical_faces(
        rho_minus,
        rho_plus,
        velocity_minus,
        velocity_plus,
        minus_neighbors,
        plus_neighbors,
    )
    expected_minus = np.empty_like(rho_minus)
    expected_plus = np.empty_like(rho_plus)
    for ibox in range(len(rho_minus)):
        minus = minus_neighbors[ibox]
        if minus < 0:
            expected_minus[ibox] = transport._rusanov(
                0.0,
                velocity_minus[ibox],
                rho_minus[ibox],
                velocity_minus[ibox],
            )
        else:
            expected_minus[ibox] = transport._rusanov(
                rho_plus[minus],
                velocity_plus[minus],
                rho_minus[ibox],
                velocity_minus[ibox],
            )

        plus = plus_neighbors[ibox]
        if plus < 0:
            expected_plus[ibox] = transport._rusanov(
                rho_plus[ibox],
                velocity_plus[ibox],
                0.0,
                velocity_plus[ibox],
            )
        else:
            expected_plus[ibox] = transport._rusanov(
                rho_plus[ibox],
                velocity_plus[ibox],
                rho_minus[plus],
                velocity_minus[plus],
            )

    assert np.allclose(actual_minus, expected_minus)
    assert np.allclose(actual_plus, expected_plus)


def test_table_timing_summary_includes_built_cache_payload():
    module = _load_benchmark("split_parameter_sweep")
    summary = module._summarize_table_get_timings([
        {
            "is_recomputed": True,
            "total_s": 3.0,
            "compute": {"table_build_s": 2.0, "payload_bytes": 128},
        },
        {
            "is_recomputed": False,
            "total_s": 0.5,
            "load": {"payload_bytes": 128},
        },
    ])

    assert summary["build_s"] == pytest.approx(3.0)
    assert summary["quadrature_build_s"] == pytest.approx(2.0)
    assert summary["build_cache_payload_bytes"] == 128
    assert summary["cache_payload_bytes"] == 128


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


def test_windowed_sweep_classifies_genuine_classical_refusals(tmp_path):
    """Pin both classical refusal kinds end to end.

    ``classical_refusal`` names the two certified refusal modes apart in the
    sweep CSV, and downstream analysis distinguishes them by name, so a
    silent reclassification (an exception type or message reworded in the
    assembler) has to fail here rather than quietly relabel rows.
    """
    module = _load_benchmark("windowed_rke_sweep")

    from volumential.rke_table_assembly import assemble_parameterized_table

    # lam * radius ~ 68 at level 0 exhausts the default series window, so the
    # truncation selector refuses before any channel is built (no queue).
    with pytest.raises(ValueError) as uncertifiable:
        assemble_parameterized_table(
            None,
            tmp_path / "uncertifiable.sqlite",
            2,
            "Yukawa",
            2,
            8.0,
            source_box_level=0,
            tolerance=1.0e-12,
        )
    assert module._classify_classical_refusal(uncertifiable.value) == (
        "uncertifiable"
    )

    import pyopencl as cl

    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception as exc:
        pytest.skip(f"no OpenCL context available: {exc}")

    # certifiable but cancellation-heavy: the channels assemble and the
    # deliberately tight max_condition then trips the conditioning guard
    with pytest.raises(RuntimeError) as ill_conditioned:
        assemble_parameterized_table(
            cl.CommandQueue(ctx),
            tmp_path / "ill-conditioned.sqlite",
            2,
            "Yukawa",
            2,
            4.0,
            source_box_level=3,
            tolerance=1.0e-8,
            max_condition=1.0,
        )
    assert module._classify_classical_refusal(ill_conditioned.value) == (
        "ill-conditioned"
    )


def test_windowed_sweep_refusal_classifier_message_fallback():
    """Unmarked exceptions still classify off the stable message fragments."""
    module = _load_benchmark("windowed_rke_sweep")

    assert module._classify_classical_refusal(
        ValueError("cannot certify tolerance 1e-11 within 60 series terms")
    ) == "uncertifiable"
    # the conditioning message also says "certified float64 recombination",
    # so the uncertifiable probe must not match it
    assert module._classify_classical_refusal(
        RuntimeError(
            "RKE table assembly is ill-conditioned for this parameter and "
            "box size (condition 1.0e+07 > 1.0e+06); the local parameter |k| "
            "times the separation radius (68.00) is too large for certified "
            "float64 recombination."
        )
    ) == "ill-conditioned"
    assert module._classify_classical_refusal(
        NotImplementedError("RKE table assembly supports 2D and 3D")
    ) == "NotImplementedError"


@pytest.mark.parametrize(("option", "value"), [
    ("--root-extent", "0"),
    ("--root-extent", "-1"),
    ("--root-extent", "nan"),
    ("--root-extent", "inf"),
    ("--root-extent", "-inf"),
    ("--window-theta", "0"),
    ("--window-theta", "-1"),
    ("--window-theta", "nan"),
    ("--window-theta", "inf"),
    ("--window-theta", "-inf"),
    ("--mus", "0"),
    ("--mus", "-1"),
    ("--mus", "nan"),
    ("--mus", "inf"),
    ("--mus", "-inf"),
])
def test_windowed_sweep_rejects_nonpositive_or_nonfinite_cli_values(
    option, value, monkeypatch
):
    module = _load_benchmark("windowed_rke_sweep")
    monkeypatch.setattr(sys, "argv", ["windowed_rke_sweep.py", option, value])
    monkeypatch.setattr(
        module,
        "run_sweep",
        lambda **kwargs: pytest.fail("run_sweep must not be called"),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize(("option", "value"), [
    ("--dim", "x"),
    ("--p-star", "x"),
    ("--smooth-orders", "x"),
])
def test_windowed_sweep_reports_malformed_csv_as_cli_error(
    option, value, monkeypatch, capsys
):
    module = _load_benchmark("windowed_rke_sweep")
    monkeypatch.setattr(sys, "argv", ["windowed_rke_sweep.py", option, value])
    monkeypatch.setattr(
        module,
        "run_sweep",
        lambda **kwargs: pytest.fail("run_sweep must not be called"),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(("option", "value"), [
    ("--dim", "2,2"),
    ("--kernels", "yukawa,yukawa"),
    ("--p-star", "4,4"),
    ("--smooth-orders", "8,8"),
    ("--mus", "1,1"),
    ("--chan-orders", "2,7;2,7"),
])
def test_windowed_sweep_reports_duplicate_axes_as_cli_errors(
    option, value, monkeypatch, capsys
):
    module = _load_benchmark("windowed_rke_sweep")
    monkeypatch.setattr(sys, "argv", ["windowed_rke_sweep.py", option, value])
    monkeypatch.setattr(
        module,
        "run_sweep",
        lambda **kwargs: pytest.fail("run_sweep must not be called"),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("raw", [
    "24,61;24,61",
    "24,61;23,160",
    "24,61;48,60",
])
def test_windowed_sweep_rejects_invalid_direct_policies(raw):
    module = _load_benchmark("windowed_rke_sweep")

    with pytest.raises(ValueError):
        module._parse_direct_policies(raw)

    assert module._parse_direct_policies("24,61;48,61") == [
        (24, 61),
        (48, 61),
    ]


@pytest.mark.parametrize("regular_order", [0, 1])
def test_windowed_sweep_rejects_unusable_regular_orders(regular_order):
    module = _load_benchmark("windowed_rke_sweep")
    invalid_inputs = [
        (module._parse_direct_policies, f"{regular_order},7;2,8"),
        (module._parse_order_pair, f"{regular_order},7"),
        (module._parse_order_pairs, f"{regular_order},7;2,8"),
    ]

    for parser, raw in invalid_inputs:
        with pytest.raises(ValueError, match="regular order must be >= 2"):
            parser(raw)


@pytest.mark.parametrize("radial_order", range(1, 7))
def test_windowed_sweep_rejects_unusable_radial_orders(radial_order):
    module = _load_benchmark("windowed_rke_sweep")
    invalid_inputs = [
        (module._parse_direct_policies, f"2,{radial_order};3,7"),
        (module._parse_order_pair, f"2,{radial_order}"),
        (module._parse_order_pairs, f"2,{radial_order};3,7"),
    ]

    for parser, raw in invalid_inputs:
        with pytest.raises(ValueError, match="radial order must be >= 7"):
            parser(raw)


def test_windowed_sweep_accepts_usable_channel_orders():
    module = _load_benchmark("windowed_rke_sweep")

    assert module._parse_order_pair("24,61") == (24, 61)
    assert module._parse_order_pairs("48,61;64,121") == [
        (48, 61),
        (64, 121),
    ]
    with pytest.raises(
        ValueError, match="channel-order policies must be unique"
    ):
        module._parse_order_pairs("48,61;48,61")


def test_windowed_sweep_float_identity_distinguishes_adjacent_values():
    module = _load_benchmark("windowed_rke_sweep")
    value = 1.0
    adjacent = np.nextafter(value, np.inf)

    assert f"{value:g}" == f"{adjacent:g}"
    assert value.hex() in module._parameter_identity_token(value)
    assert module._parameter_identity_token(value) != (
        module._parameter_identity_token(adjacent)
    )
    assert module._parameter_identity_token(value) == (
        module._parameter_identity_token(value)
    )


def test_windowed_sweep_classical_cache_identity_includes_geometry_and_policy(
    tmp_path,
):
    module = _load_benchmark("windowed_rke_sweep")
    adjacent_extent = np.nextafter(2.0, np.inf)

    baseline = module._classical_cache_path(tmp_path, 2, 3, 2.0, (24, 61))
    same = module._classical_cache_path(tmp_path, 2, 3, 2.0, (24, 61))
    changed_extent = module._classical_cache_path(
        tmp_path, 2, 3, adjacent_extent, (24, 61)
    )
    changed_policy = module._classical_cache_path(
        tmp_path, 2, 3, 2.0, (25, 61)
    )

    assert baseline == same
    assert module._exact_float_token(2.0) in baseline.name
    assert "c24x61" in baseline.name
    assert len({baseline, changed_extent, changed_policy}) == 3


def test_windowed_sweep_channel_cold_status_uses_cache_disposition(
    tmp_path, monkeypatch
):
    module = _load_benchmark("windowed_rke_sweep")
    import volumential.rke_table_assembly as rke

    class FakeChannel:
        def __init__(self, disposition):
            self._windowed_cache_disposition = disposition

        def get_reduced_entry_ids(self):
            return np.array([0], dtype=np.int64)

    def prepare(dispositions):
        dispositions = iter(dispositions)

        def get_channel(*args, **kwargs):
            return FakeChannel(next(dispositions))

        monkeypatch.setattr(rke, "get_windowed_channel_table", get_channel)
        return module._prepare_windowed_channels(
            cache_path=tmp_path / "channels.sqlite",
            dim=2,
            q_order=1,
            source_box_level=0,
            root_extent=2.0,
            window_theta=16.0,
            max_p_star=2,
            chan_regular_order=2,
            chan_radial_order=7,
        )

    # The first disposition represents an existing checksum-corrupt cache
    # that the core loader recovered by rebuilding.
    recovered = prepare(["rebuilt", "hit"])
    warm = prepare(["hit", "hit"])

    assert recovered["channel_build_was_cold"] is True
    assert warm["channel_build_was_cold"] is False

    for bad in (None, "unknown"):
        with pytest.raises(RuntimeError, match="cache disposition"):
            prepare([bad, "hit"])


@pytest.mark.parametrize(("root_extent", "window_theta"), [
    (0.0, 16.0),
    (np.nan, 16.0),
    (2.0, 0.0),
    (2.0, np.inf),
])
def test_windowed_sweep_programmatic_geometry_validation_precedes_side_effects(
    tmp_path, monkeypatch, root_extent, window_theta
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs.update(root_extent=root_extent, window_theta=window_theta)

    with pytest.raises(ValueError, match="finite and positive"):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


def _windowed_sweep_run_kwargs(cache_dir):
    return {
        "mode": "smoke",
        "dims": [2],
        "kernels": ["Yukawa"],
        "q_order_override": 1,
        "source_level_override": 0,
        "root_extent": 2.0,
        "window_theta": 16.0,
        "p_stars": [1],
        "smooth_orders": [2],
        "mus": [1.0],
        "direct_policies": [(2, 7), (3, 8)],
        "classical_channel_orders": (2, 7),
        "chan_orders": [(2, 7)],
        "cache_dir": cache_dir,
        "skip_3d_tight": False,
    }


@pytest.mark.parametrize("update", [
    {"q_order_override": 1.5},
    {"q_order_override": True},
    {"source_level_override": 0.5},
    {"source_level_override": False},
    {"smooth_orders": [1.5]},
    {"smooth_orders": [True]},
    {"p_stars": [1.5]},
    {"p_stars": [True]},
    {"direct_policies": [(2.5, 7), (3, 8)]},
    {"direct_policies": [(True, 7), (3, 8)]},
    {"classical_channel_orders": (2, 7.5)},
    {"classical_channel_orders": (2, True)},
    {"chan_orders": [(2.5, 7)]},
    {"chan_orders": [(2, False)]},
])
def test_windowed_sweep_programmatic_integer_validation_precedes_side_effects(
    tmp_path, monkeypatch, update
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs.update(update)

    with pytest.raises(ValueError, match="must be an integer"):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


def test_windowed_sweep_rejects_empty_dimensions_before_side_effects(
    tmp_path, monkeypatch
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs["dims"] = []

    with pytest.raises(ValueError, match="at least one dimension"):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


@pytest.mark.parametrize(("source_level", "match"), [
    (11, "outside the supported O\\(1\\) range"),
    (1074, "outside the supported O\\(1\\) range"),
    (1075, "box_extent must be finite and positive"),
])
def test_windowed_sweep_rejects_unsupported_geometry_before_side_effects(
    tmp_path, monkeypatch, source_level, match
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs.update(source_level_override=source_level, mus=None)

    with pytest.raises(ValueError, match=match):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


def test_windowed_sweep_rejects_underflowed_window_scale_before_side_effects(
    tmp_path, monkeypatch
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs.update(
        root_extent=2.0,
        window_theta=1.0e308,
        source_level_override=0,
        mus=[1.0],
    )

    with pytest.raises(ValueError, match="window_scale must be finite and positive"):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


def test_windowed_sweep_normalizes_mus_before_uniqueness_check(
    tmp_path, monkeypatch
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    lower = np.longdouble(1.0)
    adjacent = np.nextafter(lower, np.longdouble(2.0))
    assert lower != adjacent
    assert float(lower) == float(adjacent)
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs["mus"] = [lower, adjacent]

    with pytest.raises(ValueError, match="mu entries must be unique"):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


def test_windowed_sweep_direct_rejects_nonfinite_reference(
    tmp_path, monkeypatch
):
    module = _load_benchmark("windowed_rke_sweep")
    import volumential.table_manager as table_manager

    class FakeTable:
        def get_entry_data_for_full_indices(self, entry_ids):
            assert np.array_equal(entry_ids, np.array([0]))
            return np.array([np.nan])

    class FakeManager:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def get_table(self, *args, **kwargs):
            return FakeTable(), {}

    monkeypatch.setattr(
        table_manager, "NearFieldInteractionTableManager", FakeManager
    )

    result = module._build_direct_table(
        queue=None,
        cache_path=tmp_path / "direct.sqlite",
        dim=2,
        kernel="Yukawa",
        q_order=1,
        parameter=1.0,
        source_box_level=0,
        root_extent=2.0,
        regular_order=2,
        radial_order=7,
        entry_ids=np.array([0]),
    )

    assert result["status"].startswith("failed: RuntimeError:")
    assert "non-finite" in result["status"]
    assert result["values"] is None


@pytest.mark.parametrize(("update", "match"), [
    ({"mode": "invalid"}, "mode must"),
    ({"kernels": []}, "at least one kernel"),
    ({"kernels": ["Laplace"]}, "unknown kernel"),
    ({"p_stars": []}, "at least one p_star"),
    ({"smooth_orders": []}, "at least one smooth_order"),
    ({"mus": []}, "at least one mu"),
    ({"chan_orders": []}, "at least one channel policy"),
    ({"dims": [2, 2]}, "must be unique"),
    ({"kernels": ["Yukawa", "Yukawa"]}, "must be unique"),
    ({"p_stars": [1, 1]}, "must be unique"),
    ({"smooth_orders": [2, 2]}, "must be unique"),
    ({"mus": [1.0, 1.0]}, "must be unique"),
    ({"chan_orders": [(2, 7), (2, 7)]}, "must be unique"),
])
def test_windowed_sweep_rejects_empty_or_unknown_axes_before_side_effects(
    tmp_path, monkeypatch, update, match
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs.update(update)

    with pytest.raises(ValueError, match=match):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


@pytest.mark.parametrize("direct_policies", [
    [(2, 7)],
    [(2, 7), (3, 8), (4, 9)],
    [(2, 7), (2, 7)],
    [(3, 8), (2, 9)],
])
def test_windowed_sweep_programmatic_direct_policy_contract_precedes_side_effects(
    tmp_path, monkeypatch, direct_policies
):
    module = _load_benchmark("windowed_rke_sweep")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        module,
        "_make_queue",
        lambda: pytest.fail("queue creation must not be attempted"),
    )
    kwargs = _windowed_sweep_run_kwargs(cache_dir)
    kwargs["direct_policies"] = direct_policies

    with pytest.raises(ValueError, match="direct polic"):
        module.run_sweep(**kwargs)
    assert not cache_dir.exists()


@pytest.mark.parametrize("failed_column", [
    "windowed_status",
    "classical_status",
])
def test_windowed_sweep_main_fails_if_any_assembly_failed(
    tmp_path, monkeypatch, failed_column
):
    module = _load_benchmark("windowed_rke_sweep")
    usable = {
        "windowed_status": "ok",
        "classical_status": "ok",
        "direct_loose_status": "ok",
        "direct_tight_status": "ok",
        "direct_reference_policy": "tight",
    }
    failed = {**usable, failed_column: "failed"}
    monkeypatch.setattr(
        module,
        "run_sweep",
        lambda **kwargs: (
            [usable, failed],
            {"total_seconds": 0.0, "channel_prep": {}, "mus_by_dim": {}},
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        "windowed_rke_sweep.py",
        "--out-dir",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ])

    assert module.main() == 1


@pytest.mark.parametrize(("loose_status", "tight_status", "expected"), [
    ("failed: RuntimeError: loose", "ok", 1),
    ("ok", "failed: RuntimeError: tight", 1),
    ("ok", "skipped: --skip-3d-tight", 0),
])
def test_windowed_sweep_main_fails_on_unexpected_direct_build_failure(
    tmp_path, monkeypatch, loose_status, tight_status, expected
):
    module = _load_benchmark("windowed_rke_sweep")
    row = {
        "windowed_status": "ok",
        "classical_status": "ok",
        "direct_loose_status": loose_status,
        "direct_tight_status": tight_status,
        "direct_reference_policy": "tight" if tight_status == "ok" else "loose",
    }
    monkeypatch.setattr(
        module,
        "run_sweep",
        lambda **kwargs: (
            [row],
            {"total_seconds": 0.0, "channel_prep": {}, "mus_by_dim": {}},
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        "windowed_rke_sweep.py",
        "--out-dir",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ])

    assert module.main() == expected


def _windowed_sweep_classical_kwargs(tmp_path):
    return {
        "queue": None,
        "cache_path": tmp_path / "classical.sqlite",
        "dim": 2,
        "kernel": "Yukawa",
        "q_order": 2,
        "parameter": 1.0,
        "source_box_level": 3,
        "root_extent": 2.0,
        "channel_orders": (24, 61),
        "entry_ids": np.array([0], dtype=np.int64),
        "n_reduced_entries": 1,
    }


def _set_windowed_sweep_clock(module, monkeypatch, values):
    values = iter(values)

    class Clock:
        @staticmethod
        def perf_counter():
            return next(values)

    monkeypatch.setattr(module, "time", Clock())


def _windowed_sweep_windowed_kwargs(tmp_path):
    return {
        "cache_path": tmp_path / "windowed.sqlite",
        "dim": 2,
        "kernel": "Yukawa",
        "q_order": 1,
        "parameter": 1.0,
        "source_box_level": 0,
        "root_extent": 2.0,
        "window_theta": 16.0,
        "p_star": 1,
        "smooth_quad_order": 2,
        "chan_regular_order": 2,
        "chan_radial_order": 7,
        "entry_ids": np.array([0], dtype=np.int64),
        "n_reduced_entries": 1,
    }


def test_damped_phases_sweep_the_outgoing_half_plane():
    """The E8 path must stay on the branch the assembler selects.

    ``_selected_decay_root`` takes ``Re >= 0`` and, on the imaginary axis,
    ``Im <= 0`` -- the outgoing ``-i k``.  Sampling ``zeta = mu^2 e^{+i pi
    f}`` puts the selected root in the upper half plane, i.e. the incoming
    ``exp(-i b r)``, which flips discontinuously to outgoing at ``f = 1``.
    The conjugate path is continuous with that endpoint.
    """
    import numpy as np

    from volumential.rke_table_assembly import _selected_decay_root

    module = _load_benchmark("windowed_rke_sweep")
    mu = 8.0
    for fraction in (0.05, 0.25, 0.5, 0.75, 0.95):
        root = _selected_decay_root(module._damped_zeta(mu, fraction))
        assert root.real > 0.0
        # decaying and *outgoing*: exp(-root r) = exp(-a r) exp(+i b r)
        assert root.imag < 0.0

        # the conjugate path is the incoming one the sweep must not take
        incoming = complex((mu * mu) * np.exp(1j * np.pi * fraction))
        assert _selected_decay_root(incoming).imag > 0.0

    # the endpoints stay exactly where they were: both are real
    assert module._damped_zeta(mu, 0.0).imag == 0.0
    assert module._damped_zeta(mu, 0.0).real > 0.0
    assert module._damped_zeta(mu, 1.0).real < 0.0

    # continuous into the Helmholtz endpoint, where the selector pins -i k
    endpoint = _selected_decay_root(complex(-mu * mu))
    approaching = _selected_decay_root(
        module._damped_zeta(mu, 1.0 - 1.0e-9)
    )
    assert abs(approaching - endpoint) < 1.0e-6 * mu
    assert endpoint.imag < 0.0
    # ... which the +i pi path is not
    assert abs(
        _selected_decay_root(
            complex((mu * mu) * np.exp(1j * np.pi * (1.0 - 1.0e-9)))
        ) - endpoint
    ) > mu


@pytest.mark.parametrize(("values", "certificate", "expected"), [
    ([1.0, 2.0], {}, None),
    ([], {}, "no entries"),
    ([1.0, float("nan")], {}, "not all finite"),
    ([1.0, float("inf")], {}, "not all finite"),
    ([1.0], {"condition_number": float("inf")}, "'condition_number'"),
    ([1.0], {"remainder_peak": float("nan")}, "'remainder_peak'"),
    ([1.0], {"condition_number": 2.0}, None),
])
def test_a_nonfinite_assembly_is_not_a_successful_row(
        values, certificate, expected):
    """The smooth-remainder integration and the recombination can
    overflow or go nan -- most easily on the damped path -- and
    _relative_deviations propagates that while main() counts the row as
    usable from its status alone.
    """
    import numpy as np

    module = _load_benchmark("windowed_rke_sweep")
    reason = module._nonfinite_assembly_reason(np.asarray(values), certificate)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


def test_an_unrepresentable_mu_is_refused_before_provisioning():
    """float(mu)**2 raises OverflowError, which no row taxonomy covers,
    and the damped block runs after the real-parameter rows of the same
    sweep, so it would take their measurements down with it."""
    import math

    module = _load_benchmark("windowed_rke_sweep")

    with pytest.raises(ValueError, match="square is not representable"):
        module._damped_zeta(1.0e200, 0.5)

    # the boundary is where the square stops being finite, and everything
    # below it still works
    assert math.isfinite(module._damped_zeta(1.0e150, 0.5).real)
    assert math.isfinite(module._damped_zeta(1.0e150, 0.5).imag)


def test_damped_case_ids_keep_close_phases_apart():
    """Two phases agreeing in the default six significant digits used to
    collide on one case id with every other field equal, so tooling keyed
    on it merged or overwrote independently measured rows."""
    module = _load_benchmark("windowed_rke_sweep")
    ids = {
        module._damped_case_id(2, 8.0, phase, "c20r61", 4, 6)
        for phase in (0.50000001, 0.50000002)
    }
    assert len(ids) == 2
    # the same readable-prefix-plus-exact-hex identity the mu token uses,
    # so the phase can no longer collide where mu could not
    assert "-phi0.5-0x" in module._damped_case_id(
        2, 8.0, 0.5, "c20r61", 4, 6
    )
    assert module._damped_case_id(3, 8.0, 0.5, "c20r61", 4, 6).startswith(
        "damped3d-mu8-0x"
    )


@pytest.mark.parametrize(("error_name", "expected_status"), [
    ("RKEWindowCoverageError", "refused"),
    ("RKEWindowConditioningError", "refused"),
    ("ValueError", "failed"),
    ("RuntimeError", "failed"),
    ("NotImplementedError", "failed"),
    ("KeyError", "failed"),
    # the channel family is an .npz cache, so creating, writing or
    # atomically replacing one of its files can fail; main() writes the
    # CSV only after run_sweep() returns, so an escaping I/O error loses
    # every row the sweep already completed
    ("OSError", "failed"),
    ("PermissionError", "failed"),
    # sqlite3's exceptions descend from Exception, not OSError
    ("OperationalError", "failed"),
])
def test_windowed_sweep_windowed_errors_use_structured_refusal_taxonomy(
    tmp_path, monkeypatch, error_name, expected_status
):
    module = _load_benchmark("windowed_rke_sweep")
    import volumential.rke_table_assembly as rke

    error_types = {
        "RKEWindowCoverageError": rke.RKEWindowCoverageError,
        "RKEWindowConditioningError": rke.RKEWindowConditioningError,
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "NotImplementedError": NotImplementedError,
        "KeyError": KeyError,
        "OSError": OSError,
        "PermissionError": PermissionError,
        "OperationalError": sqlite3.OperationalError,
    }

    def assemble(*args, **kwargs):
        raise error_types[error_name]("probe failure")

    monkeypatch.setattr(rke, "assemble_windowed_parameterized_table", assemble)
    _set_windowed_sweep_clock(module, monkeypatch, [3.0, 4.5])

    result = module._run_windowed(
        **_windowed_sweep_windowed_kwargs(tmp_path)
    )

    assert result["windowed_status"] == expected_status
    assert error_name in result["windowed_refusal"]
    assert result["windowed_assemble_seconds"] == pytest.approx(1.5)
    assert result["values"] is None


def test_windowed_sweep_classical_timing_schema_is_truthful(
    tmp_path, monkeypatch
):
    module = _load_benchmark("windowed_rke_sweep")
    import volumential.rke_table_assembly as rke

    class FakeTable:
        def get_entry_data_for_full_indices(self, entry_ids):
            assert np.array_equal(entry_ids, np.array([0]))
            return np.array([2.0])

    calls = []

    def assemble(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeTable(), {
            "n_series_terms": 2,
            "channel_count": 6,
            "condition_number": 1.5,
        }

    monkeypatch.setattr(rke, "assemble_parameterized_table", assemble)
    _set_windowed_sweep_clock(
        module, monkeypatch, [1.0, 3.5, 10.0, 11.25]
    )

    result = module._run_classical(
        **_windowed_sweep_classical_kwargs(tmp_path)
    )

    assert len(calls) == 2
    assert result["classical_status"] == "ok"
    assert result["classical_warmup_seconds"] == pytest.approx(2.5)
    assert result["classical_assemble_seconds"] == pytest.approx(1.25)
    assert np.array_equal(result["values"], np.array([2.0]))
    assert "classical_build_was_cold" not in module.FIELDS
    assert "classical_channel_build_seconds" not in module.FIELDS


@pytest.mark.parametrize(("error_type", "refusal_kind"), [
    ("RKETruncationError", "uncertifiable"),
    ("RKEConditioningError", "ill-conditioned"),
])
def test_windowed_sweep_classical_structured_refusal_uses_warmup_time(
    tmp_path, monkeypatch, error_type, refusal_kind
):
    module = _load_benchmark("windowed_rke_sweep")
    import volumential.rke_table_assembly as rke

    def assemble(*args, **kwargs):
        raise getattr(rke, error_type)("numerical refusal")

    monkeypatch.setattr(rke, "assemble_parameterized_table", assemble)
    _set_windowed_sweep_clock(module, monkeypatch, [4.0, 6.5])

    result = module._run_classical(
        **_windowed_sweep_classical_kwargs(tmp_path)
    )

    assert result["classical_status"] == "refused"
    assert result["classical_refusal"] == refusal_kind
    assert error_type in result["classical_refusal_detail"]
    assert result["classical_warmup_seconds"] == pytest.approx(2.5)
    assert result["classical_assemble_seconds"] == ""
    assert result["values"] is None


@pytest.mark.parametrize("error", [
    ValueError("bad configuration"),
    RuntimeError("cache mismatch"),
    NotImplementedError("unsupported infrastructure"),
])
def test_windowed_sweep_classical_unexpected_errors_are_failures(
    tmp_path, monkeypatch, error
):
    module = _load_benchmark("windowed_rke_sweep")
    import volumential.rke_table_assembly as rke

    def assemble(*args, **kwargs):
        raise error

    monkeypatch.setattr(rke, "assemble_parameterized_table", assemble)
    _set_windowed_sweep_clock(module, monkeypatch, [8.0, 9.0])

    result = module._run_classical(
        **_windowed_sweep_classical_kwargs(tmp_path)
    )

    assert result["classical_status"] == "failed"
    assert result["classical_refusal"] == ""
    assert type(error).__name__ in result["classical_refusal_detail"]
    assert result["classical_warmup_seconds"] == pytest.approx(1.0)
    assert result["classical_assemble_seconds"] == ""
    assert result["values"] is None


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

    quantized_below_floor = module._plan_time_step(0.003, 0.0012, 0.001, 2.0)
    assert quantized_below_floor == pytest.approx((0.001, False, True))

    assert module._plan_time_step(0.0015, 0.0012, 0.001, 2.0) is None


# {{{ per-phase share columns of the split-parameter sweep (E6)

def test_sweep_phase_columns_are_appended_and_unique():
    module = _load_benchmark("split_parameter_sweep")
    fields = list(module.FIELDS)
    assert len(fields) == len(set(fields))
    last_pre_e6 = fields.index("classical_probe_s")
    for name in module.PHASE_FIELDS:
        assert fields.index(name) > last_pre_e6
    for name in module.PHASE_FIELDS:
        assert name.startswith(("ops_phase_", "s_phase_", "phase_"))


def test_sweep_phase_measurements_are_inert_when_disabled():
    module = _load_benchmark("split_parameter_sweep")

    def _explode():  # pragma: no cover - must never be called
        raise AssertionError("no solve may run when phase profiling is off")

    measurements = module._phase_measurements(
        queue=None,
        traversal=None,
        wrangler=None,
        solve=_explode,
        phase_repeat_count=0,
    )
    assert measurements["phase_profile_repeat_count"] == 0
    for key, value in measurements.items():
        if key != "phase_profile_repeat_count":
            assert value == module.PHASE_UNMEASURED


def test_sweep_phase_row_columns_map_both_paths():
    module = _load_benchmark("split_parameter_sweep")
    reference = module._phase_measurements(
        queue=None, traversal=None, wrangler=None, solve=None,
        phase_repeat_count=0,
    )
    split = dict(reference)
    reference["ops_phase_far_total"] = 1700
    reference["s_phase_solve_total"] = 0.5
    split["s_phase_solve_total"] = 2.0
    split["s_phase_split_correction"] = 1.5

    columns = module._phase_row_columns(
        reference_timing=reference, split_timing=split
    )
    assert set(columns) == set(module.PHASE_FIELDS)
    # the shared traversal's counts are taken from whichever path has them
    assert columns["ops_phase_far_total"] == 1700
    assert columns["s_phase_solve_total_reference"] == 0.5
    assert columns["s_phase_solve_total_split"] == 2.0
    assert columns["s_phase_split_correction_split"] == 1.5
    assert columns["s_phase_split_correction_reference"] == (
        module.PHASE_UNMEASURED
    )


def test_excluded_self_pairs_are_not_counted_in_the_remainder():
    """``exclude_self`` skips each target's own source; the count must too.

    On the base-quadrature path the correction keeps ``target_to_source``
    and passes the tree's ``exclude_self``, so the P2P skips the diagonal.
    Counting it overstates the remainder, and the beta P2P with it.
    """
    import numpy as _np

    sweep = _load_benchmark("split_parameter_sweep")

    class _Dev:
        def __init__(self, array):
            self._array = _np.asarray(array)

        def get(self, queue=None):
            return self._array

    # per_box must equal q_order**dim, since the remainder runs on the
    # smooth source set, which is the base quadrature here
    n_boxes, q_order, per_box = 4, 2, 4
    counts = _np.full(n_boxes, per_box, dtype=_np.int64)
    starts = _np.arange(n_boxes + 1, dtype=_np.int64)
    lists = _np.arange(n_boxes, dtype=_np.int64)

    class _Tree:
        dimensions = 2
        box_source_counts_nonchild = _Dev(counts)
        box_target_counts_nonchild = _Dev(counts)

    class _Traversal:
        tree = _Tree()
        target_boxes = _Dev(_np.arange(n_boxes, dtype=_np.int64))
        neighbor_source_boxes_starts = _Dev(starts)
        neighbor_source_boxes_lists = _Dev(lists)

    class _TreeIndep:
        def __init__(self, exclude_self):
            self.exclude_self = exclude_self

    class _Wrangler:
        helmholtz_split_order = 1
        helmholtz_split_order1_legacy_subtraction = False
        _helmholtz_split_auto_config = {}

        def __init__(self, exclude_self):
            self.tree_indep = _TreeIndep(exclude_self)

        def _helmholtz_split_extra_terms(self):
            return []

        def _get_helmholtz_split_remainder_kernel(self):
            from volumential.expansion_wrangler_fpnd import (
                _HelmholtzSplitSeriesRemainderKernel,
            )
            return _HelmholtzSplitSeriesRemainderKernel(2, 4.0, 0.0, 1, 3)

    def _count(exclude_self):
        return sweep._split_correction_operation_counts(
            queue=None,
            traversal=_Traversal(),
            wrangler=_Wrangler(exclude_self),
            q_order=q_order,
            smooth_quad_order=None,
        )

    kept = _count(False)
    skipped = _count(True)
    # the helper swallows any interrogation failure into `status`; surface
    # it rather than comparing against a blank
    assert not str(kept["status"]).startswith("unavailable"), kept["status"]
    assert not str(skipped["status"]).startswith("unavailable"), (
        skipped["status"]
    )

    # every box neighbours only itself here: 4 boxes x 5 targets x 5 sources
    assert kept["remainder_pair_evals"] == n_boxes * per_box * per_box
    # ... minus one skipped diagonal per target
    assert skipped["remainder_pair_evals"] == (
        n_boxes * per_box * per_box - n_boxes * per_box
    )
    assert kept["status"] == "base_quadrature"


def test_remainder_terms_are_counted_from_the_generated_expression():
    """The multiplier is the kernel's term count, not the series length.

    In 2D ``_HelmholtzSplitSeriesRemainderKernel`` emits a constant, one
    ``r**(2n)`` term for every ``n = 1 .. nmax``, and a second
    ``r**(2n) log r`` term for every ``n >= split_order``, so ``nmax``
    alone undercounts the remainder by roughly a factor of two.
    """
    from volumential.expansion_wrangler_fpnd import (
        _HelmholtzSplitSeriesRemainderKernel,
    )

    sweep = _load_benchmark("split_parameter_sweep")

    class _Wrangler:
        def __init__(self, kernel):
            self._kernel = kernel

        def _get_helmholtz_split_remainder_kernel(self):
            return self._kernel

    for split_order, nmax in ((1, 4), (2, 6), (3, 9)):
        kernel = _HelmholtzSplitSeriesRemainderKernel(
            2, 4.0, 0.0, split_order, nmax
        )
        counted = sweep._remainder_terms_per_pair(_Wrangler(kernel))
        # 1 constant + nmax power terms + one log term per n >= p
        expected = 1 + nmax + (nmax - split_order + 1)
        assert counted == expected, (split_order, nmax, counted, expected)
        assert counted > nmax

    # 3D drops the even powers the tables extract, so it is not 2n either
    for split_order, nmax in ((1, 5), (3, 9)):
        kernel = _HelmholtzSplitSeriesRemainderKernel(
            3, 4.0, 0.0, split_order, nmax
        )
        counted = sweep._remainder_terms_per_pair(_Wrangler(kernel))
        max_extracted_n = 2 * max(0, split_order - 1)
        expected = sum(
            1 for n in range(1, nmax + 1)
            if not (n % 2 == 0 and n <= max_extracted_n)
        )
        assert counted == expected, (split_order, nmax, counted, expected)


def test_sweep_correction_counts_come_from_the_break_even_function():
    """The two E6 artifacts must not be able to disagree.

    The sweep's split-correction operation counts are produced by the same
    ``_split_correction_operation_counts`` the break-even driver calls, and
    both drivers stamp the same ``phase_counting_rule``.
    """
    sweep = _load_benchmark("split_parameter_sweep")
    break_even = _load_benchmark("break_even_validation")

    assert (
        break_even._split_correction_operation_counts
        is sweep._split_correction_operation_counts
    )
    assert (
        break_even._tensor_product_interp_fmas
        is sweep._tensor_product_interp_fmas
    )
    # The two rules differ in exactly two documented ways: only the
    # break-even driver can state the recombination clause (the sweep does
    # provision windowed families), and only it averages the remainder term
    # count, because the sweep's rows are one parameter each.
    assert sweep.PHASE_COUNTING_RULE.startswith("e6-v3:")
    assert break_even.PHASE_COUNTING_RULE.startswith("e6-v3:")
    assert break_even.PHASE_COUNTING_RULE == (
        sweep.PHASE_COUNTING_RULE.replace(
            "*generated_remainder_term_count",
            "*mean_generated_remainder_term_count",
        )
        + ";recombination=0_per_solve_and_no_windowed_family_in_this_driver"
    )
    # the remainder multiplier is the generated term count, not nmax
    assert "generated_remainder_term_count" in sweep.PHASE_COUNTING_RULE
    assert "nmax" not in sweep.PHASE_COUNTING_RULE
    assert "nmax" not in break_even.PHASE_COUNTING_RULE


def test_sweep_non_split_paths_report_a_structural_zero_correction():
    """A blank must mean "not counted", never "there was none".

    The direct reference path, and the windowed-assembled path that rides
    the unchanged direct warm path, execute no split correction at all.
    """
    sweep = _load_benchmark("split_parameter_sweep")

    columns = sweep._phase_correction_op_columns(
        queue=None, traversal=None, wrangler=None, split=False,
        q_order=4, split_order=2, split_smooth_quad_order=None,
    )
    assert set(columns) == set(sweep.PHASE_CORRECTION_OPS_NAMES)
    assert columns["ops_phase_split_correction_status"] == "no_split_correction"
    for name in sweep.PHASE_CORRECTION_OPS_NAMES:
        if name != "ops_phase_split_correction_status":
            assert columns[name] == 0


def test_sweep_withholds_the_split_total_when_the_wrangler_is_opaque():
    """An uninterrogable wrangler blanks the correction *and* the total.

    Reporting a solve total that silently omits the correction phase would
    be worse than reporting nothing, since the correction is the split
    strategy's dominant near-field cost.
    """
    sweep = _load_benchmark("split_parameter_sweep")

    columns = sweep._phase_correction_op_columns(
        queue=None, traversal=None, wrangler=object(), split=True,
        q_order=4, split_order=2, split_smooth_quad_order=None,
    )
    assert columns["ops_phase_split_correction_status"].startswith(
        "unavailable:"
    )
    assert columns["ops_phase_split_correction_total"] == ""
    assert columns["ops_phase_split_correction_remainder_pair_evals"] == ""

    # ... and the blank propagates to the per-path total
    measurements = dict.fromkeys(
        sweep.PHASE_CORRECTION_OPS_NAMES, sweep.PHASE_UNMEASURED
    )
    measurements["ops_phase_solve_total"] = sweep.PHASE_UNMEASURED
    row = sweep._phase_row_columns(
        reference_timing=measurements, split_timing=measurements
    )
    assert row["ops_phase_solve_total_split"] == sweep.PHASE_UNMEASURED


def test_sweep_phase_row_columns_take_correction_from_the_split_path_only():
    sweep = _load_benchmark("split_parameter_sweep")
    reference = sweep._phase_measurements(
        queue=None, traversal=None, wrangler=None, solve=None,
        phase_repeat_count=0,
    )
    split = dict(reference)
    for index, name in enumerate(sweep.PHASE_CORRECTION_OPS_NAMES):
        split[name] = index
        # the reference path carries a decoy that must never be picked up
        reference[name] = "reference-decoy"
    split["ops_phase_solve_total"] = 9999
    reference["ops_phase_solve_total"] = 1111

    columns = sweep._phase_row_columns(
        reference_timing=reference, split_timing=split
    )
    for index, name in enumerate(sweep.PHASE_CORRECTION_OPS_NAMES):
        assert columns[name] == index
    assert columns["ops_phase_solve_total_split"] == 9999
    assert columns["ops_phase_solve_total_reference"] == 1111


def test_sweep_phase_row_columns_are_empty_for_an_unprofiled_run():
    module = _load_benchmark("split_parameter_sweep")
    unmeasured = module._phase_measurements(
        queue=None, traversal=None, wrangler=None, solve=None,
        phase_repeat_count=0,
    )
    columns = module._phase_row_columns(
        reference_timing=unmeasured, split_timing=unmeasured
    )
    assert columns["phase_profile_repeat_count"] == 0
    assert all(
        value == module.PHASE_UNMEASURED
        for key, value in columns.items()
        if key != "phase_profile_repeat_count"
    )


# {{{ build-routing provenance


def _minimal_split_row(module, *, direct_costs_routing, row_routing):
    """One ``_row_from_result`` row over neutral inputs."""
    from dataclasses import dataclass

    import numpy as np

    @dataclass
    class _Accounting:
        split_term_keys: tuple = ()

    class _Timing(dict):
        def __missing__(self, key):
            return 0.0

    class _BuildConfig:
        regular_quad_order = 20
        radial_quad_order = 40
        n_levels = 1

    values = np.zeros(4)
    kwargs = dict(
        mode="smoke",
        kernel="Yukawa",
        parameter_name="lambda",
        parameter=4.0,
        split_order=1,
        power_log_beta_mode="p2p",
        direct_build_config=_BuildConfig(),
        rke_channel_build_config=_BuildConfig(),
        split_smooth_quad_order=None,
        q_order=2,
        nlevels=2,
        fmm_order=8,
        reference_path="direct_fixed_parameter_table",
        reference_values=values,
        split_values=values,
        reference_timing=_Timing(),
        split_timing=_Timing(),
        accounting=_Accounting(),
        direct_costs=_Timing({"build_routing": direct_costs_routing}),
        rke_costs=_Timing(),
        amortization={},
        direct_levels=[2],
        repeat_count=1,
        dim=2,
    )
    if row_routing is not None:
        kwargs["direct_build_routing"] = row_routing
    return module._row_from_result(**kwargs)


def test_canonical_table_routing_reaches_the_reported_routing():
    """Each adaptive row times the canonical table, so it must be reported.

    ``direct_build_routing`` used to carry only the per-level tables'
    routing, so a canonical build that fell back while the per-level builds
    succeeded (or the reverse) was reported as ``batched``.
    """
    module = _load_benchmark("adaptive_timing")

    class _Table:
        def __init__(self, routing):
            self.build_routing = routing

    union = module._table_build_routing_union
    assert union("batched", _Table("batched")) == "batched"
    # the canonical table fell back, the per-level tables did not
    assert union("batched", _Table("scalar-fallback")) == (
        "batched;scalar-fallback"
    )
    # ... and the reverse
    assert union("scalar-fallback", _Table("batched")) == (
        "batched;scalar-fallback"
    )
    # an unrecorded canonical routing is still surfaced, not dropped
    assert union("batched", _Table(None)) == "batched;unknown"
    # no canonical table (an older caller) leaves the per-level set alone
    assert union("batched;scalar", None) == "batched;scalar"


def test_split_sweep_rows_report_their_own_parameter_s_routing():
    """One parameter falling back must not relabel every other row.

    ``direct_costs["build_routing"]`` is a union over the whole sweep,
    which is right for the shared setup-cost columns and wrong for a
    column documented as the routing of *this row's* reference tables.
    """
    module = _load_benchmark("split_parameter_sweep")

    row = _minimal_split_row(
        module,
        direct_costs_routing="batched;scalar-fallback",
        row_routing="batched",
    )
    assert row["direct_build_routing"] == "batched"

    # the row of the parameter that actually fell back says so
    fell_back = _minimal_split_row(
        module,
        direct_costs_routing="batched;scalar-fallback",
        row_routing="scalar-fallback",
    )
    assert fell_back["direct_build_routing"] == "scalar-fallback"

    # ... and a caller that has no per-row value still gets the aggregate
    aggregate = _minimal_split_row(
        module,
        direct_costs_routing="batched;scalar-fallback",
        row_routing=None,
    )
    assert aggregate["direct_build_routing"] == "batched;scalar-fallback"


# }}}
