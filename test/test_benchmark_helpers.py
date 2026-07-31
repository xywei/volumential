import importlib.util
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


def test_windowed_sweep_rejects_overflowed_default_mu_before_side_effects(
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
        root_extent=1.0e-3,
        window_theta=1.0e308,
        source_level_override=0,
        mus=None,
    )

    with pytest.raises(ValueError, match="mu must be finite and positive"):
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


@pytest.mark.parametrize(("error_name", "expected_status"), [
    ("RKEWindowCoverageError", "refused"),
    ("RKEWindowConditioningError", "refused"),
    ("ValueError", "failed"),
    ("RuntimeError", "failed"),
    ("NotImplementedError", "failed"),
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
