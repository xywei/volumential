"""Queue-free tests for the 3D adaptive-split-composition evidence
extensions (E5b): Helmholtz composition rows, the explicit quadrature-policy
knob, the windowed-assembled composition row with its taxonomy and gates,
and the CSV-schema parity with the 2D composition driver that lets one
manuscript reader consume both files."""

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

import pytest


_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"

# The committed 3D composition CSV column prefix (the header of
# data/benchmarks/adaptive-split-composition-3d/
# adaptive_split_composition_3d.csv).  Appended extension columns must never
# move these, so the committed artifact stays readable byte-position-wise.
COMMITTED_FIELD_PREFIX_3D = (
    "case_id",
    "mode",
    "kernel",
    "dim",
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

# Committed 3D Yukawa full-mode mismatches (lambda = 2 / 4 / 8 at p = 1/2/3)
# from the archived artifact; the split-order gate is a regression check
# against exactly these numbers.
COMMITTED_3D_YUKAWA_MISMATCHES = {
    (2.0, 1): 6.3352206776730366e-06,
    (2.0, 2): 3.5855440419868202e-09,
    (2.0, 3): 3.5822551875700795e-09,
    (4.0, 1): 4.471348527653531e-05,
    (4.0, 2): 6.444981915031325e-09,
    (4.0, 3): 6.329213355612433e-09,
    (8.0, 1): 0.00040386978210934957,
    (8.0, 2): 2.6883242264613494e-08,
    (8.0, 3): 1.4341287071100777e-08,
}


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
def composition3d():
    return _load_benchmark_module("adaptive_split_composition_3d")


@pytest.fixture(scope="module")
def composition2d():
    return _load_benchmark_module("adaptive_split_composition")


def test_committed_csv_column_prefix_is_unchanged(composition3d):
    assert composition3d.FIELDS[: len(COMMITTED_FIELD_PREFIX_3D)] == (
        COMMITTED_FIELD_PREFIX_3D
    )
    assert len(set(composition3d.FIELDS)) == len(composition3d.FIELDS)


def test_csv_schema_is_the_2d_schema_plus_dim(composition3d, composition2d):
    """One manuscript reader must handle both composition CSVs: the 3D
    schema is the 2D schema with a single ``dim`` column inserted after
    ``kernel``, and nothing else."""
    fields_2d = composition2d.FIELDS
    kernel_index = fields_2d.index("kernel")
    expected = (
        fields_2d[: kernel_index + 1] + ("dim",) + fields_2d[kernel_index + 1:]
    )
    assert composition3d.FIELDS == expected
    assert len(composition3d.FIELDS) == len(fields_2d) + 1
    assert "dim" not in fields_2d


def test_windowed_fields_match_the_2d_driver(composition3d, composition2d):
    assert composition3d.WINDOWED_FIELDS == composition2d.WINDOWED_FIELDS
    assert set(composition3d.WINDOWED_FIELDS) <= set(composition3d.FIELDS)
    for name in (
        "window_theta",
        "table_strategy",
        "windowed_status",
        "windowed_refusal",
        "windowed_max_condition_number",
        "windowed_vs_direct_weighted_rel_l2",
    ):
        assert name in composition3d.FIELDS


def test_kernel_parameter_tags_match_the_2d_driver(composition3d, composition2d):
    assert composition3d.KERNEL_PARAMETER_TAGS == {
        "Yukawa": "lam",
        "Helmholtz": "k",
    }
    assert (
        composition3d.KERNEL_PARAMETER_TAGS
        == composition2d.KERNEL_PARAMETER_TAGS
    )


def test_quadrature_policy_resolution(composition3d):
    assert composition3d._resolve_quadrature_policy("auto", "smoke") == "default"
    assert composition3d._resolve_quadrature_policy("auto", "full") == (
        "high-accuracy"
    )
    assert composition3d._resolve_quadrature_policy(
        "high-accuracy", "smoke"
    ) == "high-accuracy"
    assert composition3d._resolve_quadrature_policy("default", "full") == (
        "default"
    )
    with pytest.raises(ValueError, match="unknown quadrature policy"):
        composition3d._resolve_quadrature_policy("fast", "smoke")


def test_kernel_build_configs_are_kernel_independent_in_3d(composition3d):
    """The 3D Yukawa and Helmholtz kernels share the canonical ``1/r``
    singularity, so one policy serves both (unlike 2D, where the
    high-accuracy policy exists for the Yukawa ``K_0`` logarithm)."""
    for high_accuracy in (False, True):
        yukawa = composition3d._kernel_build_configs(
            "Yukawa", 3, high_accuracy=high_accuracy
        )
        helmholtz = composition3d._kernel_build_configs(
            "Helmholtz", 3, high_accuracy=high_accuracy
        )
        for a, b in zip(yukawa, helmholtz, strict=True):
            assert (a.regular_quad_order, a.radial_quad_order) == (
                b.regular_quad_order,
                b.radial_quad_order,
            )

    # The high-accuracy orders are exactly the ones the committed 3D
    # artifact records (direct 16/45, channels 12/35 at q = 3).
    direct, channel = composition3d._kernel_build_configs(
        "Yukawa", 3, high_accuracy=True
    )
    assert (direct.regular_quad_order, direct.radial_quad_order) == (16, 45)
    assert (channel.regular_quad_order, channel.radial_quad_order) == (12, 35)

    direct_default, channel_default = composition3d._kernel_build_configs(
        "Helmholtz", 3, high_accuracy=False
    )
    assert (
        direct_default.regular_quad_order,
        direct_default.radial_quad_order,
    ) == (12, 30)
    assert (
        channel_default.regular_quad_order,
        channel_default.radial_quad_order,
    ) == (12, 30)

    with pytest.raises(ValueError, match="unknown kernel"):
        composition3d._kernel_build_configs("Stokes", 3, high_accuracy=False)


def test_case_id_reproduces_the_committed_yukawa_ids(composition3d):
    assert composition3d._case_id("Yukawa", 3, 4, 2, 2.0, "p1") == (
        "yukawa3d-q3-l4-a2-lam2-p1"
    )
    assert composition3d._case_id("Yukawa", 3, 4, 3, 8.0, "p3") == (
        "yukawa3d-q3-l4-a3-lam8-p3"
    )
    assert composition3d._case_id("Helmholtz", 3, 4, 2, 4.0, "p2") == (
        "helmholtz3d-q3-l4-a2-k4-p2"
    )
    assert composition3d._case_id("Helmholtz", 3, 4, 3, 8.0, "windowed") == (
        "helmholtz3d-q3-l4-a3-k8-windowed"
    )
    with pytest.raises(ValueError, match="unknown kernel"):
        composition3d._case_id("Stokes", 3, 4, 2, 2.0, "p1")


def test_channel_counting_rule(composition3d):
    """3D extracts the odd radial powers r^(2j-1), j = 1 .. p-1, so the
    channel count is p - 1 and the shared table count (base included) is p:
    the LP-to-p statement of the cost model."""
    assert composition3d._expected_channel_table_count(1) == 0
    assert composition3d._expected_channel_table_count(2) == 1
    assert composition3d._expected_channel_table_count(3) == 2
    assert composition3d._expected_channel_table_count(7) == 6
    with pytest.raises(ValueError, match="split_order must be >= 1"):
        composition3d._expected_channel_table_count(0)


def _windowed_row(composition3d, **overrides):
    row = {field: "" for field in composition3d.FIELDS}
    row.update(
        {
            "case_id": "yukawa3d-q2-l3-a1-lam2-windowed",
            "mode": "smoke",
            "kernel": "Yukawa",
            "dim": 3,
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


def test_windowed_composition_validation_passes_clean_rows(composition3d):
    composition3d._validate_windowed_composition_rows(
        [
            _windowed_row(composition3d),
            _windowed_row(composition3d, kernel="Helmholtz"),
            # refusal beyond the declaration is an expected certificate
            # outcome, not a failure
            _windowed_row(
                composition3d,
                max_theta=32.0,
                windowed_status="refused",
                windowed_refusal="RKEWindowCoverageError: outside",
                windowed_vs_direct_weighted_rel_l2="",
            ),
            # online rows are not the windowed validator's business
            {"table_strategy": "online_split"},
        ]
    )


def test_windowed_composition_validation_rejects_failures(composition3d):
    with pytest.raises(RuntimeError, match="failed"):
        composition3d._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition3d,
                    windowed_status="failed",
                    windowed_refusal="RuntimeError: boom",
                )
            ]
        )


def test_windowed_composition_rejects_refusal_inside_declaration(composition3d):
    with pytest.raises(RuntimeError, match="inside the declaration"):
        composition3d._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition3d,
                    max_theta=8.0,
                    windowed_status="refused",
                    windowed_refusal="RKEWindowConditioningError: bad",
                )
            ]
        )


def test_windowed_composition_gates_small_theta_agreement(composition3d):
    with pytest.raises(RuntimeError, match="disagrees"):
        composition3d._validate_windowed_composition_rows(
            [
                _windowed_row(
                    composition3d, windowed_vs_direct_weighted_rel_l2=0.5
                )
            ]
        )
    # large-theta rows are not gated on the direct reference agreement
    composition3d._validate_windowed_composition_rows(
        [
            _windowed_row(
                composition3d,
                max_theta=8.0,
                windowed_vs_direct_weighted_rel_l2=0.5,
            )
        ]
    )


def _online_row(composition3d, **overrides):
    row = {field: "" for field in composition3d.FIELDS}
    row.update(
        {
            "case_id": "yukawa3d-q3-l4-a2-lam2-p2",
            "mode": "full",
            "kernel": "Yukawa",
            "dim": 3,
            "q_order": 3,
            "initial_nlevels": 4,
            "adapt_steps": 2,
            "parameter": 2.0,
            "split_order": 2,
            "quadrature_policy": "high-accuracy",
            "table_strategy": "online_split",
            "rke_vs_direct_weighted_rel_l2": 1.0e-8,
        }
    )
    row.update(overrides)
    return row


def test_split_order_gate_accepts_the_committed_3d_artifact(composition3d):
    rows = [
        _online_row(
            composition3d,
            parameter=parameter,
            split_order=split_order,
            rke_vs_direct_weighted_rel_l2=value,
        )
        for (parameter, split_order), value in (
            COMMITTED_3D_YUKAWA_MISMATCHES.items()
        )
    ]
    composition3d._validate_split_order_convergence(rows)


def test_split_order_gate_requires_p2_improvement(composition3d):
    stalled = [
        _online_row(
            composition3d, split_order=1, rke_vs_direct_weighted_rel_l2=1.0e-4
        ),
        _online_row(
            composition3d, split_order=2, rke_vs_direct_weighted_rel_l2=5.0e-5
        ),
    ]
    with pytest.raises(RuntimeError, match="three orders of magnitude"):
        composition3d._validate_split_order_convergence(stalled)


def test_split_order_gate_rejects_material_degradation(composition3d):
    rows = [
        _online_row(
            composition3d, split_order=2, rke_vs_direct_weighted_rel_l2=1.0e-8
        ),
        _online_row(
            composition3d, split_order=3, rke_vs_direct_weighted_rel_l2=1.0e-6
        ),
    ]
    with pytest.raises(RuntimeError, match="materially degraded"):
        composition3d._validate_split_order_convergence(rows)


def test_split_order_gate_scope(composition3d):
    # smoke rows, default-policy rows, Helmholtz rows, and windowed rows are
    # all outside the gate
    rows = [
        _online_row(
            composition3d,
            mode="smoke",
            split_order=1,
            rke_vs_direct_weighted_rel_l2=1.0e-4,
        ),
        _online_row(
            composition3d,
            mode="smoke",
            split_order=2,
            rke_vs_direct_weighted_rel_l2=9.0e-5,
        ),
        _online_row(
            composition3d,
            quadrature_policy="default",
            split_order=1,
            rke_vs_direct_weighted_rel_l2=1.0e-4,
        ),
        _online_row(
            composition3d,
            quadrature_policy="default",
            split_order=2,
            rke_vs_direct_weighted_rel_l2=9.0e-5,
        ),
        _online_row(
            composition3d,
            kernel="Helmholtz",
            split_order=1,
            rke_vs_direct_weighted_rel_l2=1.0e-4,
        ),
        _online_row(
            composition3d,
            kernel="Helmholtz",
            split_order=2,
            rke_vs_direct_weighted_rel_l2=9.0e-5,
        ),
        _windowed_row(composition3d),
    ]
    composition3d._validate_split_order_convergence(rows)


def test_windowed_channel_order_default_matches_the_assembler(composition3d):
    """The driver's 3D channel-order default must be the assembler's tested
    per-dimension pair, not a hand-copied guess."""
    from volumential.rke_table_assembly import _resolve_channel_orders

    from split_parameter_sweep import (
        DEFAULT_WINDOWED_CHAN_ORDERS_2D,
        DEFAULT_WINDOWED_CHAN_ORDERS_3D,
    )

    assert DEFAULT_WINDOWED_CHAN_ORDERS_3D == _resolve_channel_orders(
        3, None, None
    )
    assert DEFAULT_WINDOWED_CHAN_ORDERS_2D == _resolve_channel_orders(
        2, None, None
    )
    assert DEFAULT_WINDOWED_CHAN_ORDERS_3D == (20, 61)


def test_shared_windowed_helpers_reject_unsupported_dimensions():
    """Both shared helpers refuse a dimension they have no geometry for.

    The refusal goes through the sweep's own ``_require_dimension``, so the
    message names ``SUPPORTED_DIMENSIONS`` rather than spelling "2D and 3D"
    twice.
    """
    if str(_BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(_BENCHMARK_DIR))
    sweep = _load_benchmark_module("split_parameter_sweep")
    assert sweep.SUPPORTED_DIMENSIONS == (2, 3)

    with pytest.raises(ValueError, match=r"dim must be one of"):
        sweep._prepare_windowed_family(
            cache_path=Path("unused.sqlite"),
            q_order=2,
            source_box_level=0,
            window_theta=16.0,
            p_star=1,
            chan_regular_order=20,
            chan_radial_order=61,
            dim=4,
        )
    with pytest.raises(ValueError, match=r"dim must be one of"):
        sweep._register_and_load_windowed_table(
            queue=None,
            cache_path=Path("unused.sqlite"),
            kernel="Yukawa",
            q_order=2,
            parameter=1.0,
            source_box_level=0,
            table=None,
            certificate={},
            dim=4,
        )
