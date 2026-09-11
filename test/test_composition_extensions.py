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


@pytest.mark.parametrize("max_theta", [0.5, 8.0])
@pytest.mark.parametrize("rel_l2", [float("nan"), float("inf")])
def test_windowed_composition_rejects_a_nonfinite_mismatch_at_every_theta(
        composition, max_theta, rel_l2):
    """The accuracy tolerance applies only at small theta, but finiteness
    is not part of that scope restriction: a nan or inf mismatch is a
    broken measurement wherever it happens, and a large-theta row used to
    be written as valid evidence with windowed_status == "ok".
    """
    with pytest.raises(RuntimeError, match="non-finite mismatch"):
        composition._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition,
                    max_theta=max_theta,
                    windowed_vs_direct_weighted_rel_l2=rel_l2,
                )
            ]
        )


def test_windowed_composition_rejects_an_ok_row_without_a_mismatch(
        composition):
    with pytest.raises(RuntimeError, match="status ok without a mismatch"):
        composition._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition, windowed_vs_direct_weighted_rel_l2=""
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


@pytest.mark.parametrize("exc", [
    ValueError("unusable channel order"),
    OSError("channel cache is unwritable"),
    sqlite3.OperationalError("database is locked"),
])
def test_channel_family_failures_become_a_failed_row(
    composition, tmp_path, monkeypatch, exc
):
    """The family build is provisioning too.

    It used to sit outside the ``refused``/``failed`` handling, so an
    unusable channel order or an unwritable cache aborted the whole
    composition run instead of returning a failed windowed row.
    """
    def _raise(**kwargs):
        raise exc

    monkeypatch.setattr(composition, "_prepare_windowed_family", _raise)

    row = composition._run_windowed_composition(
        None, **_windowed_kwargs_2d(tmp_path)
    )

    assert row["windowed_status"] == "failed"
    assert type(exc).__name__ in row["windowed_refusal"]
    assert "windowed channel family" in row["windowed_refusal"]
    assert "_tables" not in row


def test_a_channel_family_refusal_stays_refused(
    composition, tmp_path, monkeypatch
):
    """A certificate refusal stays 'refused', wherever it is raised."""
    from volumential.rke_table_assembly import RKEWindowCoverageError

    def _raise(**kwargs):
        raise RKEWindowCoverageError("theta outside the declaration")

    monkeypatch.setattr(composition, "_prepare_windowed_family", _raise)

    row = composition._run_windowed_composition(
        None, **_windowed_kwargs_2d(tmp_path)
    )

    assert row["windowed_status"] == "refused"
    assert "RKEWindowCoverageError" in row["windowed_refusal"]


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

# {{{ case ids, argument validation and the CSV/gate ordering

def test_case_id_reproduces_the_committed_yukawa_ids(composition):
    assert composition._case_id("Yukawa", 3, 2, 1, 2.0, "p1") == (
        "yukawa2d-q3-l2-a1-lam2-p1"
    )
    assert composition._case_id("Yukawa", 4, 4, 2, 8.0, "p3") == (
        "yukawa2d-q4-l4-a2-lam8-p3"
    )
    assert composition._case_id("Helmholtz", 3, 2, 1, 4.0, "windowed") == (
        "helmholtz2d-q3-l2-a1-k4-windowed"
    )
    with pytest.raises(ValueError, match="unknown kernel"):
        composition._case_id("Stokes", 3, 2, 1, 2.0, "p1")


def test_case_id_separates_parameters_beyond_six_digits(composition):
    """Two parameters that differ in the eighth digit need distinct ids.

    ``%g`` renders both as ``1`` at the default six significant digits, so
    tooling keyed on ``case_id`` merged or overwrote two independently
    measured rows.
    """
    close = (1.0000001, 1.0000002)
    ids = {
        composition._case_id("Yukawa", 3, 2, 1, parameter, "p2")
        for parameter in close
    }
    assert len(ids) == len(close)
    # ... while the committed round-valued ids are byte-identical
    assert composition._case_id("Yukawa", 3, 2, 1, 2.0, "p1") == (
        "yukawa2d-q3-l2-a1-lam2-p1"
    )


@pytest.mark.parametrize(("option", "value", "message"), [
    ("--window-theta", "inf", "--window-theta must be finite and positive"),
    ("--window-theta", "0", "--window-theta must be finite and positive"),
    ("--window-theta", "-1", "--window-theta must be finite and positive"),
    ("--window-theta", "1e-200", "--window-theta is too small"),
    ("--windowed-p-star", "0", "--windowed-p-star must be >= 1"),
    # the assembler's own usable-order rule, quoted through parser.error
    ("--windowed-chan-orders", "0,61", "--windowed-chan-orders: "),
    ("--windowed-chan-orders", "1,61", "need at least 2"),
    ("--windowed-chan-orders", "20,2", "silent max(3, order) clamp"),
])
def test_main_rejects_bad_windowed_arguments_before_touching_a_device(
    composition, tmp_path, monkeypatch, capsys, option, value, message
):
    """Every windowed argument check runs before device selection.

    An infinite Theta passes a bare positivity test and is only refused by
    the channel builder, after the geometry and every direct and
    online-split table of the first case have been built.
    """
    monkeypatch.setattr(
        composition,
        "_select_opencl_device",
        lambda *a, **k: pytest.fail("device selection must not be reached"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adaptive_split_composition.py",
            "--include-windowed",
            f"{option}={value}",
            "--out", str(tmp_path / "never-written.csv"),
        ],
    )

    with pytest.raises(SystemExit) as exited:
        composition.main()

    assert exited.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "never-written.csv").exists()


def test_main_writes_the_csv_before_running_the_failure_gates(
    composition, tmp_path, monkeypatch
):
    """Converting provisioning problems into failed rows only preserves a
    long run's measurements if the CSV is written before the gates raise.
    The 3D twin already writes first.
    """
    out = tmp_path / "composition.csv"
    failed_row = _windowed_row(
        composition,
        windowed_status="failed",
        windowed_refusal="RuntimeError: boom",
    )

    monkeypatch.setattr(
        composition, "_select_opencl_device", lambda *a, **k: None
    )
    monkeypatch.setattr(composition.cl, "Context", lambda devices: None)
    monkeypatch.setattr(composition.cl, "CommandQueue", lambda ctx: None)
    monkeypatch.setattr(
        composition, "run_case", lambda *a, **k: [failed_row]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adaptive_split_composition.py",
            "--include-windowed",
            "--out", str(out),
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )

    with pytest.raises(RuntimeError, match="windowed assembly failed"):
        composition.main()

    # the gate still fires, but the measurements survive it
    assert out.exists()
    assert failed_row["case_id"] in out.read_text()


def test_online_split_rows_reject_a_nonfinite_mismatch_outside_the_scope(
    composition,
):
    """The convergence ratios are scoped to full-mode high-accuracy
    Yukawa; finiteness is not.  A smoke, default-policy or Helmholtz row
    with a nan or inf mismatch used to reach the CSV as successful
    numerical evidence, and inside the scope nan made the ratio
    comparisons False rather than raising.
    """
    for overrides in (
        {"mode": "smoke", "quadrature_policy": "default"},
        {"kernel": "Helmholtz"},
        {},
    ):
        for value in (float("nan"), float("inf")):
            with pytest.raises(RuntimeError, match="non-finite mismatch"):
                composition._validate_split_order_convergence(
                    [
                        _online_row(
                            composition,
                            rke_vs_direct_weighted_rel_l2=value,
                            **overrides,
                        )
                    ]
                )

# }}}


# {{{ parameter, split-order and resolved-order validation

@pytest.mark.parametrize(("parameters", "message"), [
    ((2.0, 0.0), "degenerates both 2D kernels to Laplace"),
    ((-1.0,), "must be positive"),
    ((float("nan"),), "must be finite"),
    ((float("inf"),), "must be finite"),
    ((2.0, 2.0), "must be unique"),
    ((), "at least one kernel parameter"),
])
def test_kernel_parameters_are_validated(composition, parameters, message):
    """argparse hands 0, nan and inf straight through; run_case must not."""
    with pytest.raises(ValueError, match=message):
        composition._validated_parameters(parameters)


def test_valid_kernel_parameters_pass_through(composition):
    assert composition._validated_parameters([2.0, 4.0, 8.0]) == (
        2.0, 4.0, 8.0
    )
    assert composition._validated_parameters(
        composition.FULL_PARAMETERS
    ) == tuple(composition.FULL_PARAMETERS)


@pytest.mark.parametrize(("split_orders", "message"), [
    ((2, 2), "must be unique"),
    ((0,), "must be >= 1"),
    ((), "at least one split order"),
])
def test_split_orders_are_validated(composition, split_orders, message):
    with pytest.raises(ValueError, match=message):
        composition._validated_split_orders(split_orders)


def test_every_shipped_2d_configuration_is_resolved(composition):
    """The guard must accept every configuration the driver ships with.

    Each mode's cases are paired with that mode's parameters, which is how
    ``main`` dispatches them.
    """
    for cases, parameters in (
        (composition.SMOKE_CASES, composition.SMOKE_PARAMETERS),
        (composition.FULL_CASES, composition.FULL_PARAMETERS),
    ):
        for q_order, _initial_nlevels, _adapt_steps in cases:
            composition._require_resolved_fmm_order(
                ("Helmholtz", "Yukawa"), parameters, max(8, 4 * q_order)
            )


def test_underresolved_helmholtz_parameters_are_refused(composition):
    """A wave number the fixed order cannot resolve must not run.

    The direct, online-split and windowed paths share one FMM here, so an
    underresolved far field diverges in all three and the reported path
    mismatch stays small while every number is wrong.
    """
    with pytest.raises(ValueError, match="needs FMM order 33") as refused:
        composition._require_resolved_fmm_order(
            ("Helmholtz",), (64.0,), max(8, 4 * 3)
        )
    # the remedy the message offers has to exist: this driver takes q from
    # its built-in case list and exposes no --q-order flag
    assert "--q-order" not in str(refused.value)
    assert "run_case()" in str(refused.value)
    # Yukawa is resolved at the floor whatever its decay rate, so the same
    # value is fine when no Helmholtz row is requested
    composition._require_resolved_fmm_order(
        ("Yukawa",), (64.0,), max(8, 4 * 3)
    )


def test_main_refuses_an_underresolved_wave_number_before_a_device(
    composition, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        composition,
        "_select_opencl_device",
        lambda *a, **k: pytest.fail("device selection must not be reached"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adaptive_split_composition.py",
            "--kernels", "Helmholtz",
            "--parameters", "64",
            "--out", str(tmp_path / "never-written.csv"),
        ],
    )

    with pytest.raises(SystemExit) as exited:
        composition.main()

    assert exited.value.code == 2
    assert "needs FMM order" in capsys.readouterr().err
    assert not (tmp_path / "never-written.csv").exists()

# }}}

@pytest.mark.parametrize(("key", "changed"), [
    ("window_theta", 16.0000001),
    ("windowed_p_star", 5),
    ("windowed_chan_orders", (24, 61)),
    ("quadrature_policy", "high-accuracy"),
])
def test_the_windowed_case_suffix_separates_campaigns(composition, key, changed):
    """Campaigns differing only in the window declaration, p_star, the
    channel orders or the quadrature policy assemble different tables and
    measure different errors, but all shared the literal "windowed"
    suffix."""
    base = {
        "window_theta": 16.0,
        "windowed_p_star": 4,
        "windowed_chan_orders": (20, 61),
        "quadrature_policy": "default",
    }
    suffix = composition._windowed_case_suffix(**base)
    assert suffix.startswith("windowed-cfg")
    assert composition._windowed_case_suffix(**base) == suffix
    assert composition._windowed_case_suffix(**{**base, key: changed}) != suffix

@pytest.mark.parametrize("exc", [
    RuntimeError("non-finite List 1 result"),
    ValueError("bad table"),
    OSError("device disappeared"),
])
def test_windowed_evaluator_failures_become_a_failed_row(
    composition, monkeypatch, exc
):
    """The evaluator is provisioning's last mile.

    drive_volume_fmm raises RuntimeError when its List 1 result turns
    non-finite, and main() writes the CSV only after every case
    completes, so an exception escaping the evaluator discarded every
    direct, online-split and windowed row already measured.
    """
    def _raise(**kwargs):
        raise exc

    monkeypatch.setattr(composition, "_build_path", _raise)

    updates = composition._windowed_evaluator_updates(
        ctx=None,
        queue=None,
        traversal=None,
        q_order=2,
        fmm_order=8,
        kernel="Yukawa",
        parameter=2.0,
        windowed_tables=[object()],
        q_weights=None,
        source_values_host=None,
        weights_host=None,
        direct_potential=None,
    )

    assert updates["windowed_status"] == "failed"
    assert "windowed evaluator" in updates["windowed_refusal"]
    assert type(exc).__name__ in updates["windowed_refusal"]
    assert updates["windowed_vs_direct_weighted_rel_l2"] == ""


def test_a_successful_windowed_evaluation_reports_its_columns(
    composition, monkeypatch
):
    monkeypatch.setattr(
        composition, "_build_path", lambda **kwargs: (object(), None, None)
    )
    monkeypatch.setattr(
        composition, "_drive", lambda *a, **k: ([0.0, 0.0, 0.0, 0.0], 1.25)
    )
    monkeypatch.setattr(
        composition, "_weighted_mismatch", lambda *a, **k: (1.0e-8, 2.0e-8)
    )

    updates = composition._windowed_evaluator_updates(
        ctx=None,
        queue=None,
        traversal=None,
        q_order=2,
        fmm_order=8,
        kernel="Yukawa",
        parameter=2.0,
        windowed_tables=[object()],
        q_weights=None,
        source_values_host=None,
        weights_host=None,
        direct_potential=None,
    )

    assert "windowed_status" not in updates
    assert updates["windowed_wall_s"] == pytest.approx(1.25)
    assert updates["windowed_vs_direct_weighted_rel_l2"] == pytest.approx(
        1.0e-8
    )
    assert updates["windowed_vs_direct_linf"] == pytest.approx(2.0e-8)

