#!/usr/bin/env python3
"""Emit Helmholtz/Yukawa split-parameter sweep CSVs for Paper 1.

The benchmark sweeps the Helmholtz wave number and Yukawa screening parameter
while recording split-table accounting. Rows compare against fixed-parameter
direct near-field tables and separate direct-table setup/apply costs from RKE
channel setup, coefficient, residual, and full split costs. Cold/warm strategy
totals and break-even roots expose the parameter/level/repeat amortization model.

``--dim`` selects the spatial dimension (2, the historical default, or 3).
The three-dimensional path uses the same conventions the other Paper 1 3D
drivers already exercise: ``MeshGen3D`` geometry on ``[-0.5, 0.5]^3``, the
3D Helmholtz ``exp(i k r) / (4 pi r)`` and Yukawa ``exp(-lambda r) / (4 pi r)``
kernels, the loose ``16/45`` and tight ``24/61`` direct Duffy policies of the
committed three-dimensional production dispatch of ``windowed_rke_sweep.py``
(``--direct-policies '16,45;24,61'``; the script's own default pair is the 2D
one), and its per-dimension windowed channel orders (``48/61`` in 2D,
``20/61`` in 3D).

``--fmm-order-rule resolved`` prescribes the surrounding FMM expansion order
per row from the row's own Helmholtz wave number instead of pinning it, so a
theta ladder that walks the wave number up stays resolved inside a single
cold-cache invocation (see :func:`_resolved_fmm_order`).  ``--max-fmm-order``
caps that prescription: a row whose prescribed order exceeds the cap is
recorded as refused rather than solved at an order that cannot represent its
far field.

Smoke mode is intended for CI/local validation. Full mode is intended for
metadata-wrapped runs on a controlled remote compute host.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np


# {{{ per-phase share columns (E6)

#: Far-field FMM stage names, matching
#: :data:`volumential.opcounters.FMM_FAR_FIELD_STAGES`.
PHASE_FAR_STAGES = (
    "form_multipoles",
    "coarsen_multipoles",
    "multipole_to_local",
    "eval_multipoles",
    "form_locals",
    "refine_locals",
    "eval_locals",
)

#: Timed phases reported per path, aggregated over the far-field stages.
PHASE_TIMED_NAMES = (
    "far_total",
    "nearfield_table_apply",
    "split_correction",
    "other",
    "solve_total",
)

#: Path prefixes: the fixed-parameter direct reference and the split path.
PHASE_PATHS = ("reference", "split")

#: Operation counts of the split path's correction phase, all produced by
#: :func:`_split_correction_operation_counts`, the same function
#: ``benchmarks/break_even_validation.py`` uses, so the two E6 artifacts
#: cannot drift apart.  Without them the ``ops_phase_*`` columns would be a
#: far-field decomposition plus one base near-field pair count and could not
#: be summed into a solve total: the correction phase is the split
#: strategy's dominant near-field cost.
PHASE_CORRECTION_OPS_NAMES = (
    "ops_phase_split_correction_extra_table_fmas",
    "ops_phase_split_correction_remainder_pair_evals",
    "ops_phase_split_correction_remainder_term_evals",
    "ops_phase_split_correction_remainder_terms_per_pair",
    "ops_phase_split_correction_beta_p2p_pair_evals",
    "ops_phase_split_correction_smooth_interp_fmas",
    "ops_phase_split_smooth_sources_per_box",
    "ops_phase_split_correction_status",
    "ops_phase_split_correction_total",
)

PHASE_FIELDS = (
    "phase_profile_repeat_count",
    "phase_counting_rule",
    *(f"ops_phase_far_{stage}" for stage in PHASE_FAR_STAGES),
    "ops_phase_far_total",
    "ops_phase_nearfield_point_pairs_per_solve",
    *PHASE_CORRECTION_OPS_NAMES,
    *(f"ops_phase_solve_total_{path}" for path in PHASE_PATHS),
    *(
        f"s_phase_{name}_{path}"
        for path in PHASE_PATHS
        for name in PHASE_TIMED_NAMES
    ),
)

#: Identifies the counting-rule revision the ``ops_phase_*`` columns follow,
#: so a consumer can tell two executions of different rules apart.  Kept
#: identical to ``break_even_validation``'s, because the counts come from
#: the same functions.
PHASE_COUNTING_RULE = (
    "e6-v3:far=dense_coefficient_touches_from_traversal_and_expansion_sizes;"
    "nearfield=fma_per_nearfield_pair_per_applied_table_dtype_blind;"
    "split_correction=extra_table_fmas"
    "+remainder_pair_evals*generated_remainder_term_count"
    "+beta_p2p_pair_evals+smooth_interp_fmas;"
    "smooth_interp=tensor_product_axis_by_axis_not_dense"
)

#: Value written into every per-phase column when phase profiling is off.
PHASE_UNMEASURED = ""

# }}}


FIELDS = (
    "case_id",
    "mode",
    "kernel",
    "dim",
    "parameter_name",
    "parameter_value",
    "split_order",
    "power_log_beta_mode",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
    "q_order",
    "nlevels",
    "fmm_order",
    "n_targets",
    "reference_path",
    "rel_l2_error",
    "linf_error",
    "reference_warm_s",
    "split_warm_s",
    "online_remainder_s",
    "online_remainder_time_kind",
    "parameter_count",
    "base_table_count",
    "split_term_table_count",
    "basis_table_count",
    "base_table_payload_bytes",
    "split_term_table_payload_bytes",
    "total_table_payload_bytes",
    "split_term_keys",
    "uses_online_coefficients",
    "uses_online_remainder",
    "level_count",
    "direct_levels",
    "repeat_count",
    "solve_count",
    "direct_table_count",
    "direct_table_payload_bytes",
    "direct_table_cache_payload_bytes",
    "direct_table_build_s",
    "direct_table_quadrature_build_s",
    "direct_table_load_s",
    "rke_channel_build_s",
    "rke_channel_quadrature_build_s",
    "rke_channel_load_s",
    "rke_channel_cache_payload_bytes",
    "direct_table_apply_total_s",
    "direct_table_apply_mean_s",
    "split_full_apply_total_s",
    "split_full_apply_mean_s",
    "split_correction_total_s",
    "split_correction_mean_s",
    "split_correction_time_kind",
    "smooth_residual_total_s",
    "smooth_residual_mean_s",
    "smooth_residual_time_kind",
    "coefficient_eval_mean_s",
    "coefficient_eval_time_kind",
    "reference_solve_total_s",
    "split_solve_total_s",
    "direct_strategy_cold_total_s",
    "rke_strategy_cold_total_s",
    "direct_strategy_warm_total_s",
    "rke_strategy_warm_total_s",
    "direct_amortized_cold_s_per_solve",
    "rke_amortized_cold_s_per_solve",
    "direct_amortized_warm_s_per_solve",
    "rke_amortized_warm_s_per_solve",
    "modeled_cold_savings_s",
    "direct_table_build_mean_s_per_parameter_level",
    "direct_solve_mean_s_per_solve",
    "split_solve_mean_s_per_solve",
    "amortization_time_kind",
    "break_even_parameter_count",
    "break_even_level_count",
    "break_even_repeat_count",
    "break_even_time_kind",
    "benchmark_total_s",
    "benchmark_total_time_kind",
    # table-provisioning strategy columns (E1).  "online_split" rows are the
    # historical online split-evaluator comparison; "windowed_assembled" rows
    # provision the near-field table by windowed offline assembly
    # (rke_table_assembly.assemble_windowed_parameterized_table), register it
    # through the standard table manager, and run the identical evaluator
    # path against the same direct fixed-parameter reference.
    "table_strategy",
    "theta",
    "window_theta",
    "windowed_p_star",
    "windowed_smooth_quad_order",
    "windowed_chan_regular_order",
    "windowed_chan_radial_order",
    "windowed_status",
    "windowed_refusal",
    "windowed_condition_number",
    "windowed_channel_build_s",
    "windowed_channel_build_was_cold",
    "windowed_assemble_s",
    "windowed_register_s",
    "windowed_register_payload_bytes",
    "windowed_table_load_s",
    "windowed_table_load_payload_bytes",
    "windowed_solve_warm_s",
    "windowed_table_apply_mean_s",
    "classical_probe_kind",
    "classical_probe_status",
    "classical_probe_detail",
    "classical_probe_n_terms",
    "classical_probe_condition_number",
    "classical_probe_s",
    # far-field resolution accounting (E1b/E1c).  Appended after the
    # historical 106 columns so committed artifacts stay readable by name.
    "fmm_order_rule",
    "fmm_order_floor",
    "fmm_expansion_radius",
    "far_field_status",
    "implied_reference_norm",
    # per-phase shares of one end-to-end solve (E6), opt-in through
    # --phase-repeat-count.  Off by default, so a run that does not ask for
    # them emits these columns empty and is otherwise unchanged.  Counting
    # rules and the split-correction caveats are documented in
    # benchmarks/break_even_validation.py, which owns the primary E6
    # artifact, and the counts here come from the same functions under the
    # same phase_counting_rule tag; here the far field is reported as one
    # aggregate in the seconds columns because the per-stage breakdown is
    # the same traversal in every row.  The "_reference" suffix is the
    # row's fixed-parameter direct reference path; "_split" is the row's
    # own table_strategy path -- the online split evaluator on online_split
    # rows, and the windowed-assembled table (which rides the unchanged
    # direct warm path, so it records neither split_correction seconds nor
    # correction operations) on windowed_assembled rows.
    #
    # ops_phase_solve_total_{reference,split} are per-path operation
    # totals, so an operation share is a division inside one row.  The
    # split path's total includes the correction phase; a blank
    # ops_phase_split_correction_* means the wrangler could not be
    # interrogated (see ops_phase_split_correction_status) and the split
    # total is withheld rather than reported without its dominant phase.
    *PHASE_FIELDS,
    # which DuffyRadial builder actually produced the direct reference tables
    # of this row (';'-joined over the built levels).  A "scalar-fallback"
    # here means the batched OpenCL build failed and the slower, differently
    # converged per-entry builder produced the numbers.
    "direct_build_routing",
)


class _BenchmarkGateError(RuntimeError):
    """A post-run gate failure that carries the rows it was measured on.

    The gates of :func:`_validate_yukawa_order_convergence` and
    :func:`_validate_windowed_rows` run on complete rows, so a failure means
    the measurements exist and only the verdict on them is negative.
    :func:`main` writes them out before re-raising: a multi-hour run must not
    lose its CSV to a failing gate, for the same reason the far-field
    resolution check reports after the write rather than before it.
    """

    def __init__(self, message: str, rows: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.rows = rows


# Root extent of the table-manager convention used throughout this driver
# (the [-0.5, 0.5]^d unit tree maps to table level ell + 1).
TABLE_ROOT_EXTENT = 2.0

SUPPORTED_DIMENSIONS = (2, 3)

# Per-dimension production discretizations.  The 2D values are the historical
# defaults of this driver; the 3D values are the production source order and
# smoke order of ``windowed_rke_sweep.py`` and ``rke_field_demo_3d.py``.
DEFAULT_Q_ORDER = {
    2: {"smoke": 2, "full": 4},
    3: {"smoke": 2, "full": 3},
}
DEFAULT_NLEVELS = {
    2: {"smoke": 2, "full": 3},
    3: {"smoke": 2, "full": 3},
}
DEFAULT_FMM_ORDER = {
    2: {"smoke": 8, "full": 16},
    3: {"smoke": 8, "full": 12},
}

# Windowed strategy defaults (E1): declared window and per-mode theta ladders.
# The smoke ladder holds one theta inside the polynomial-certified range, one
# in the polynomial refusal band, and the declaration edge itself.
DEFAULT_WINDOW_THETA = 16.0
DEFAULT_WINDOWED_P_STAR = 6
# Tested per-dimension channel quadrature orders; these mirror
# ``rke_table_assembly._resolve_channel_orders`` (48/61 in 2D, where the
# high-aspect Duffy triangles of edge-adjacent interpolation nodes converge
# slowly in the angular order, and 20/61 in 3D, whose cone geometry is mild).
DEFAULT_WINDOWED_CHAN_ORDERS = {2: (48, 61), 3: (20, 61)}
DEFAULT_WINDOWED_CHAN_ORDERS_2D = DEFAULT_WINDOWED_CHAN_ORDERS[2]
DEFAULT_WINDOWED_CHAN_ORDERS_3D = DEFAULT_WINDOWED_CHAN_ORDERS[3]
DEFAULT_SMOKE_WINDOWED_THETAS = (1.0, 6.0, 16.0)
DEFAULT_FULL_WINDOWED_THETAS = (
    0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0,
)
CLASSICAL_PROBE_TOLERANCE = 1.0e-11
# Evaluator-level agreement gates for the windowed-assembled strategy at
# small theta (theta <= 2), where the direct reference is well resolved.
# The smoke gate tolerates the loose smoke-mode direct quadrature; the full
# gate sits above the ~2e-7 channel-quadrature accuracy of the default
# 48/61 orders (see rke_table_assembly._resolve_channel_orders) with margin,
# while still catching any structural (sign/normalization/registration)
# error, which would show up at O(1).
WINDOWED_SMALL_THETA_AGREEMENT = {"smoke": 1.0e-2, "full": 1.0e-5}
WINDOWED_SMALL_THETA_MAX = 2.0

# Accuracy-margin term of the resolved FMM-order rule (see
# :func:`_resolved_fmm_order`).
FMM_ORDER_ACCURACY_TERMS = 3.0
# The implied reference-field norm ``linf_error / rel_l2_error`` of a solved
# row is a lower bound on the norm the relative column divides by.  A resolved
# far field keeps it O(1); an underresolved Helmholtz far field drives it
# through many orders of magnitude before both columns collapse to zero.  The
# band below is the pass criterion of the 2D order-scaled run, widened by two
# decades on each side so it flags divergence rather than ordinary variation.
FAR_FIELD_IMPLIED_NORM_BAND = (1.0e-2, 1.0e2)


def _require_dimension(dim: int) -> int:
    dim = int(dim)
    if dim not in SUPPORTED_DIMENSIONS:
        raise ValueError(
            f"dim must be one of {SUPPORTED_DIMENSIONS}; got {dim}"
        )
    return dim


def _box_extent(nlevels: int, root_extent: float = TABLE_ROOT_EXTENT) -> float:
    """Source-box extent of the leaf table level, in either dimension.

    The driver's tree is the unit box ``[-0.5, 0.5]^d`` while the table cache
    convention uses root extent 2, so the leaf tree box of a depth-``nlevels``
    uniform tree is the table manager's level-``nlevels`` box and the two
    extents agree: ``2 * 2**-nlevels == 1 * 2**-(nlevels - 1)``.
    """
    return float(root_extent) * 0.5**int(nlevels)


def _uniform_target_count(dim: int, q_order: int, nlevels: int) -> int:
    """Quadrature-node count of the uniform ``MeshGen`` tree this driver builds.

    ``MeshGen{2,3}D(q, nlevels, ...)`` refines to ``2**((nlevels - 1) * dim)``
    leaves carrying ``q**dim`` nodes each.  Kept as a pure function so the
    configuration guard can be checked without building geometry.
    """
    dim = _require_dimension(dim)
    if q_order < 1:
        raise ValueError("q_order must be >= 1")
    if nlevels < 1:
        raise ValueError("nlevels must be >= 1")
    return int(q_order) ** dim * 2 ** ((int(nlevels) - 1) * dim)


def _require_min_targets(n_targets: int, min_targets: int, dim: int) -> None:
    """Refuse a geometry smaller than the configuration the run declares."""
    if int(n_targets) < int(min_targets):
        raise RuntimeError(
            f"the requested {dim}D geometry carries {int(n_targets)} "
            f"targets, below the required minimum {int(min_targets)}; raise "
            "--nlevels or --q-order"
        )


def _fmm_expansion_radius(dim: int) -> float:
    """Radius the surrounding FMM's expansions must represent, in units of the
    unit root box.

    A level-1 box of the unit tree has side ``1/2``, so its half-diagonal is
    ``sqrt(dim) / 4``.  The 2D value ``sqrt(2)/4`` is not an assumption: it is
    calibrated in ``kb/reports/helmholtz-fmm-order-contamination.md`` against
    two committed sweeps whose Helmholtz breakdown threshold is
    ``nlevels``-independent, which falsifies a leaf-radius rule outright and
    fixes the binding radius at this domain-scale length.  The 3D value is the
    same geometric quantity in three dimensions and is checked, not assumed,
    by the implied-reference-norm criterion of
    :func:`_validate_far_field_resolution`.
    """
    return math.sqrt(_require_dimension(dim)) / 4.0


def _resolved_fmm_order(dim: int, wave_number: float, *, floor: int) -> int:
    """``p(k) = max(floor, ceil(k a + 3 ln(k a + pi)))`` with ``a`` the
    expansion radius of :func:`_fmm_expansion_radius`.

    The form is the standard multipole/local truncation estimate: the order
    must exceed the wave number times the radius of the region the expansion
    represents, plus a logarithmic accuracy term.  ``floor`` is a floor and
    never a ceiling, so no row is less resolved than a pinned-order run of
    the same configuration.
    """
    floor = int(floor)
    if floor < 1:
        raise ValueError("fmm order floor must be >= 1")
    wave_number = abs(float(wave_number))
    if not math.isfinite(wave_number):
        raise ValueError("wave number must be finite")
    ka = wave_number * _fmm_expansion_radius(dim)
    prescribed = math.ceil(ka + FMM_ORDER_ACCURACY_TERMS * math.log(ka + math.pi))
    return max(floor, int(prescribed))


def _prescribed_fmm_order(
    dim: int,
    kernel: str,
    parameter: float,
    *,
    floor: int,
    rule: str,
) -> int:
    """The FMM order this row is solved at.

    Only the oscillatory kernel drives the order: the Yukawa far field decays
    and its expansions are resolved at the floor, which is why the 2D
    order-scaled run leaves the Yukawa mismatch column unchanged to twelve
    significant digits at raised orders.
    """
    if rule not in {"fixed", "resolved"}:
        raise ValueError("fmm order rule must be 'fixed' or 'resolved'")
    if rule == "fixed" or kernel != "Helmholtz":
        return int(floor)
    return _resolved_fmm_order(dim, parameter, floor=floor)


def _implied_reference_norm(linf_error, rel_l2_error):
    """``linf / rel_l2``: a lower bound on the reference-field norm the
    relative error column divides by.  Blank when it is not computable."""
    try:
        linf = float(linf_error)
        rel_l2 = float(rel_l2_error)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(linf) or not math.isfinite(rel_l2) or rel_l2 == 0.0:
        return ""
    return linf / rel_l2


def _parse_csv_floats(raw: str, *, allow_empty: bool = False) -> list[float]:
    if allow_empty and raw.strip().lower() in {"", "none", "skip"}:
        return []
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        if allow_empty:
            return []
        raise ValueError("expected at least one float value")
    return values


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    if any(value < 1 for value in values):
        raise ValueError("integer values must be >= 1")
    return values


def _parse_csv_levels(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one source-box level")
    if any(value < 0 for value in values):
        raise ValueError("source-box levels must be >= 0")
    if len(set(values)) != len(values):
        raise ValueError("source-box levels must be unique")
    return sorted(values)


def _limit_parameter_values(values: list[float], count: int | None) -> list[float]:
    if count is None or not values:
        return values
    if count < 1:
        raise ValueError("parameter_count must be >= 1")
    if count > len(values):
        raise ValueError(
            f"parameter_count={count} exceeds the {len(values)} supplied values"
        )
    return values[:count]


def _table_payload_bytes(table) -> int:
    if bool(getattr(table, "table_data_is_symmetry_reduced", False)):
        if hasattr(table, "get_reduced_table_data"):
            _, values = table.get_reduced_table_data()
            return int(np.asarray(values).nbytes)
        data = np.asarray(table.data)
        return int(np.count_nonzero(np.isfinite(data)) * data.dtype.itemsize)
    return int(np.asarray(table.data).nbytes)


def _configure_logging() -> None:
    """Route library INFO/WARNING records to stderr for the run log.

    Without this the ``[duffy:builder] mode=...`` routing lines are dropped and
    a batched-to-scalar fallback warning only reaches stderr through Python's
    last-resort handler.  Only configures the root logger when the embedding
    process has not already installed handlers.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _case_parameter_token(parameter: float) -> str:
    """The parameter token of a ``case_id``.

    ``.17g`` round-trips a float64, so two parameters that differ beyond
    the sixth significant digit cannot collide on one case id and have
    their rows merged (or their saved fields overwritten) by tooling keyed
    on it.  ``%g`` strips trailing zeros, so the committed round-valued
    ids are byte-identical: 2.0 is still "2".
    """
    return f"{float(parameter):.17g}"


def _table_build_routing_counts(tables) -> dict[str, int]:
    """How many of ``tables`` were produced by each recorded routing.

    Eager provisioning can land on more than one routing across levels and
    parameters -- one batched invocation succeeding while another falls back
    -- and the two routings have different per-entry node counts, so a cost
    model that prices the whole set has to know how many tables each routing
    owns rather than sampling one of them.
    """
    import volumential.opcounters as opcounters

    counts: dict[str, int] = {}
    for table in tables:
        routing = opcounters.direct_build_routing(table)
        counts[routing] = counts.get(routing, 0) + 1
    return counts


def _table_build_routing(tables) -> str:
    """The distinct recorded DuffyRadial routings of ``tables``, ';'-joined."""
    return ";".join(sorted(_table_build_routing_counts(tables)))


def _clear_sqlite_cache(path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


@contextmanager
def _capture_table_get_timings():
    """Capture table-manager timings, including split-channel auto-builds."""
    from volumential.table_manager import NearFieldInteractionTableManager

    records = []
    original_get_table = NearFieldInteractionTableManager.get_table

    def timed_get_table(table_manager, *args, **kwargs):
        result = original_get_table(table_manager, *args, **kwargs)
        records.append(dict(table_manager.last_get_table_timings or {}))
        return result

    NearFieldInteractionTableManager.get_table = timed_get_table
    try:
        yield records
    finally:
        NearFieldInteractionTableManager.get_table = original_get_table


def _summarize_table_get_timings(records: list[dict[str, Any]]) -> dict[str, Any]:
    build_records = [record for record in records if record.get("is_recomputed")]
    load_records = [record for record in records if not record.get("is_recomputed")]
    return {
        "build_s": float(
            sum(record.get("total_s", 0.0) for record in build_records)
        ),
        "quadrature_build_s": float(
            sum((record.get("compute") or {}).get("table_build_s", 0.0)
                for record in build_records)
        ),
        "load_s": float(sum(record.get("total_s", 0.0) for record in load_records)),
        "cache_payload_bytes": int(
            sum((record.get("load") or {}).get("payload_bytes", 0)
                for record in load_records)
        ),
        "build_cache_payload_bytes": int(
            sum((record.get("compute") or {}).get("payload_bytes", 0)
                for record in build_records)
        ),
        "build_count": len(build_records),
        "load_count": len(load_records),
    }


def _device_supports_fp64(device) -> bool:
    extensions = set(getattr(device, "extensions", "").split())
    return "cl_khr_fp64" in extensions or bool(
        getattr(device, "double_fp_config", 0)
    )


#: Values ``--backend`` accepts.
SUPPORTED_BACKENDS = ("auto", "pocl-cpu", "cuda-gpu")


def _select_opencl_device(cl, backend: str):
    backend = backend.lower()
    # Validate the argument before touching the ICD loader: a misspelled
    # --backend should say so rather than surface whatever the driver says,
    # and on a machine with no OpenCL platform at all cl.get_platforms()
    # raises LogicError(PLATFORM_NOT_FOUND_KHR) before this check could run.
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            "backend must be one of: " + ", ".join(SUPPORTED_BACKENDS)
        )

    platforms = cl.get_platforms()

    if backend == "pocl-cpu":
        for platform in platforms:
            if "portable computing language" in platform.name.lower():
                for dev in platform.get_devices():
                    if dev.type & cl.device_type.CPU and _device_supports_fp64(dev):
                        return dev
        raise RuntimeError("PoCL CPU device with fp64 support not found")

    if backend == "cuda-gpu":
        for platform in platforms:
            if "nvidia cuda" in platform.name.lower():
                for dev in platform.get_devices():
                    if dev.type & cl.device_type.GPU and _device_supports_fp64(dev):
                        return dev
        raise RuntimeError("NVIDIA CUDA GPU device with fp64 support not found")

    # only "auto" is left; the whitelist above rejected everything else
    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl.device_type.GPU and _device_supports_fp64(dev):
                return dev

    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl.device_type.CPU and _device_supports_fp64(dev):
                return dev

    raise RuntimeError("No OpenCL GPU/CPU device with fp64 support found")


def _build_config(q_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(8, 4 * q_order),
        radial_quad_order=max(21, 10 * q_order),
    )


def _yukawa_reference_build_config(q_order: int, *, high_accuracy: bool):
    if not high_accuracy:
        return _build_config(q_order)

    from volumential.nearfield_potential_table import DuffyBuildConfig

    # The default scalar Duffy rule under-resolves the 2D Yukawa logarithmic
    # singularity and can hide split-order convergence behind table noise.
    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(80, 20 * q_order),
        radial_quad_order=max(320, 80 * q_order),
    )


def _split_channel_build_config(q_order: int, *, high_accuracy: bool):
    if not high_accuracy:
        return _build_config(q_order)

    from volumential.nearfield_potential_table import DuffyBuildConfig

    # Auto-built log-power channels inherit the canonical Laplace table's rule.
    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(32, 12 * q_order),
        radial_quad_order=max(80, 40 * q_order),
    )


def _split_smooth_quad_order(
        q_order: int, split_order: int, *, high_accuracy: bool):
    if split_order <= 1:
        return q_order
    return (2 if high_accuracy else 1) * q_order


def _duffy_config(regular_quad_order: int, radial_quad_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=int(regular_quad_order),
        radial_quad_order=int(radial_quad_order),
    )


def _direct_build_config_3d(q_order: int, *, high_accuracy: bool):
    """Direct fixed-parameter reference policy in 3D.

    The two policies are the loose/tight pair the committed 3D production
    dispatch of ``windowed_rke_sweep.py`` passes as
    ``--direct-policies '16,45;24,61'`` (the script's own default pair,
    ``24,61;48,160``, is the 2D one): ``16/45`` is also the accuracy the 3D
    field demo builds its direct references at, and ``24/61`` is the tight
    reference that 3D sweep reports its deviations against.  Both kernels
    share them, the 3D singularity being ``1/r`` for Helmholtz and Yukawa
    alike.
    """
    if high_accuracy:
        return _duffy_config(max(24, 8 * q_order), max(61, 20 * q_order))
    return _duffy_config(max(16, 4 * q_order), max(45, 12 * q_order))


def _split_channel_build_config_3d(q_order: int, *, high_accuracy: bool):
    """Auto-built 3D odd-power channel policy of ``rke_field_demo_3d.py``."""
    if high_accuracy:
        return _duffy_config(max(12, 4 * q_order), max(35, 10 * q_order))
    return _duffy_config(max(8, 4 * q_order), max(21, 10 * q_order))


def _direct_build_config(
    dim: int, kernel: str, q_order: int, *, high_accuracy: bool
):
    if _require_dimension(dim) == 3:
        return _direct_build_config_3d(q_order, high_accuracy=high_accuracy)
    if kernel == "Yukawa":
        return _yukawa_reference_build_config(
            q_order, high_accuracy=high_accuracy
        )
    return _build_config(q_order)


def _channel_build_config(
    dim: int, kernel: str, q_order: int, *, high_accuracy: bool
):
    if _require_dimension(dim) == 3:
        return _split_channel_build_config_3d(
            q_order, high_accuracy=high_accuracy
        )
    if kernel == "Yukawa":
        return _split_channel_build_config(
            q_order, high_accuracy=high_accuracy
        )
    return _build_config(q_order)


def _smooth_quad_order(
    dim: int, q_order: int, split_order: int, *, high_accuracy: bool
) -> int:
    """Online smooth-remainder order per axis.

    3D follows ``rke_field_demo_3d._field_smooth_quad_order``; 2D keeps this
    driver's historical rule.
    """
    if _require_dimension(dim) == 3:
        if high_accuracy and split_order > 1:
            return 2 * q_order
        return q_order
    return _split_smooth_quad_order(
        q_order, split_order, high_accuracy=high_accuracy
    )


def _get_laplace_table(
    queue,
    cache_path: Path,
    dim: int,
    q_order: int,
    *,
    force_recompute: bool = False,
    build_config=None,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    dim = _require_dimension(dim)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=TABLE_ROOT_EXTENT, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            dim,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            queue=queue,
            build_config=(
                _build_config(q_order) if build_config is None else build_config
            ),
        )
    return table


def _get_yukawa_table(
    queue,
    cache_path: Path,
    dim: int,
    q_order: int,
    lam: float,
    level: int,
    *,
    force_recompute: bool = False,
    build_config=None,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    dim = _require_dimension(dim)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=TABLE_ROOT_EXTENT, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            dim,
            "Yukawa",
            q_order,
            source_box_level=int(level),
            force_recompute=force_recompute,
            queue=queue,
            build_config=(
                _build_config(q_order) if build_config is None else build_config
            ),
            lam=float(lam),
        )
    return table


def _build_helmholtz_table(
    queue,
    cache_path: Path,
    dim: int,
    q_order: int,
    wave_number: float,
    level: int,
    *,
    force_recompute: bool = False,
    build_config=None,
):
    from sumpy.kernel import HelmholtzKernel
    from volumential.table_manager import NearFieldInteractionTableManager

    dim = _require_dimension(dim)
    kernel = HelmholtzKernel(dim)
    kernel_kwargs = {kernel.helmholtz_k_name: float(wave_number)}
    with NearFieldInteractionTableManager(
        str(cache_path),
        root_extent=TABLE_ROOT_EXTENT,
        dtype=np.complex128,
        queue=queue,
    ) as table_manager:
        table, _ = table_manager.get_table(
            dim,
            "Helmholtz-Reference",
            q_order,
            source_box_level=int(level),
            force_recompute=force_recompute,
            queue=queue,
            build_config=(
                _build_config(q_order) if build_config is None else build_config
            ),
            sumpy_knl=kernel,
            **kernel_kwargs,
        )
    return table


# Historical two-dimensional names, kept because the other Paper 1 drivers
# (adaptive_split_composition.py, keller_segel_continuation.py) import them.
def _get_laplace_2d_table(queue, cache_path: Path, q_order: int, **kwargs):
    return _get_laplace_table(queue, cache_path, 2, q_order, **kwargs)


def _get_yukawa_2d_table(
    queue, cache_path: Path, q_order: int, lam: float, level: int, **kwargs
):
    return _get_yukawa_table(
        queue, cache_path, 2, q_order, lam, level, **kwargs
    )


def _build_helmholtz_2d_table(
    queue,
    cache_path: Path,
    q_order: int,
    wave_number: float,
    level: int,
    **kwargs,
):
    return _build_helmholtz_table(
        queue, cache_path, 2, q_order, wave_number, level, **kwargs
    )


def _build_geometry(ctx, queue, q_order: int, nlevels: int, *, dim: int = 2):
    import volumential.meshgen as mg

    dim = _require_dimension(dim)
    mesh_cls = {2: mg.MeshGen2D, 3: mg.MeshGen3D}[dim]
    mesh = mesh_cls(q_order, nlevels, -0.5, 0.5, queue=queue)
    return mg.build_geometry_info(
        ctx,
        queue,
        dim,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * dim, dtype=np.float64),
    )


def _coords_host(queue, q_points):
    return np.array([axis.get(queue) for axis in q_points])


# Off-centre offsets of the screened-kernel Gaussian source, per axis.  The
# first two reproduce the historical 2D source exactly.
_GAUSSIAN_SOURCE_OFFSETS = (0.11, -0.07, 0.03)


def _gaussian_source_host(coords):
    coords = np.asarray(coords)
    dim = coords.shape[0]
    r2 = sum(
        (coords[axis] + _GAUSSIAN_SOURCE_OFFSETS[axis]) ** 2
        for axis in range(dim)
    )
    return np.exp(-35.0 * r2)


def _helmholtz_manufactured_source_and_exact(coords, wave_number: float):
    """``u = exp(-alpha |x|^2)`` and the source ``-(lap + k^2) u`` it solves.

    ``lap exp(-alpha r^2) = (4 alpha^2 r^2 - 2 d alpha) exp(-alpha r^2)`` in
    ``d`` dimensions, so the ``2 d alpha`` term below reduces to the
    historical ``4 alpha`` in 2D.
    """
    alpha = 80.0
    coords = np.asarray(coords)
    dim = coords.shape[0]
    r2 = sum(coords[axis] * coords[axis] for axis in range(dim))
    exact = np.exp(-alpha * r2)
    source = (
        2 * dim * alpha - 4 * alpha * alpha * r2 - wave_number * wave_number
    ) * exact
    return source, exact


def _source_values(queue, q_points, dtype, source_values_host=None):
    import pyopencl.array as cla

    if source_values_host is None:
        source_values_host = _gaussian_source_host(_coords_host(queue, q_points))
    return cla.to_device(
        queue, np.ascontiguousarray(source_values_host.astype(dtype))
    )


def _build_path(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    kernel: str,
    parameter: float,
    table,
    source_weights,
    q_points,
    source_values_host=None,
    split: bool,
    split_order: int,
    split_term_tables=None,
    split_auto_config=None,
    split_smooth_quad_order: int | None = None,
    dim: int = 2,
):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import HelmholtzKernel, YukawaKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    dim = _require_dimension(dim)
    if kernel == "Helmholtz":
        out_kernel = HelmholtzKernel(dim)
        kernel_kwargs = {out_kernel.helmholtz_k_name: float(parameter)}
        dtype = np.complex128
    elif kernel == "Yukawa":
        out_kernel = YukawaKernel(dim)
        kernel_kwargs = {out_kernel.yukawa_lambda_name: float(parameter)}
        # The shared split evaluator currently emits complex intermediates for
        # both kernel families. Keep direct and split paths on the same dtype.
        dtype = np.complex128
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    source_vals = _source_values(queue, q_points, dtype, source_values_host)
    weighted_sources = source_vals * source_weights.astype(dtype)

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(out_kernel)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(out_kernel)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, out_kernel),
        partial(local_expn_class, out_kernel),
        [out_kernel],
        exclude_self=True,
    )
    self_extra_kwargs = {}
    if traversal.tree.sources_are_targets:
        self_extra_kwargs["target_to_source"] = np.arange(
            traversal.tree.ntargets, dtype=np.int32
        )

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=table,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        kernel_extra_kwargs=kernel_kwargs,
        self_extra_kwargs=self_extra_kwargs,
        helmholtz_split=split,
        helmholtz_split_order=split_order,
        helmholtz_split_smooth_quad_order=split_smooth_quad_order,
        helmholtz_split_auto_config=split_auto_config,
        helmholtz_split_term_tables=split_term_tables,
    )

    return wrangler, weighted_sources, source_vals


def _time_repeated(queue, repeat_count: int, operation):
    _ = operation()
    queue.finish()
    start = time.perf_counter()
    result = None
    for _ in range(repeat_count):
        result = operation()
    queue.finish()
    total_s = time.perf_counter() - start
    return result, total_s, total_s / repeat_count


def _run_path(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    kernel: str,
    parameter: float,
    table,
    source_weights,
    q_points,
    source_values_host=None,
    split: bool,
    split_order: int,
    split_term_tables=None,
    split_auto_config=None,
    split_smooth_quad_order: int | None = None,
    repeat_count: int,
    dim: int = 2,
    phase_repeat_count: int = 0,
):
    from volumential.volume_fmm import drive_volume_fmm

    wrangler, weighted_sources, source_vals = _build_path(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        dim=dim,
        q_order=q_order,
        fmm_order=fmm_order,
        kernel=kernel,
        parameter=parameter,
        table=table,
        source_weights=source_weights,
        q_points=q_points,
        source_values_host=source_values_host,
        split=split,
        split_order=split_order,
        split_term_tables=split_term_tables,
        split_auto_config=split_auto_config,
        split_smooth_quad_order=split_smooth_quad_order,
    )

    def solve():
        return drive_volume_fmm(
            traversal,
            wrangler,
            weighted_sources,
            source_vals,
            direct_evaluation=False,
            list1_only=False,
        )

    potentials, solve_total_s, solve_mean_s = _time_repeated(
        queue, repeat_count, solve
    )
    (potential,) = potentials

    reordered_source_vals = wrangler.reorder_sources(source_vals)
    reordered_weighted_sources = wrangler.reorder_sources(weighted_sources)

    def table_apply():
        values, _ = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            reordered_source_vals,
        )
        return values

    _, table_apply_total_s, table_apply_mean_s = _time_repeated(
        queue, repeat_count, table_apply
    )

    timing = {
        "solve_total_s": solve_total_s,
        "solve_mean_s": solve_mean_s,
        "table_apply_total_s": table_apply_total_s,
        "table_apply_mean_s": table_apply_mean_s,
        "split_full_apply_total_s": 0.0,
        "split_full_apply_mean_s": 0.0,
        "split_correction_total_s": 0.0,
        "split_correction_mean_s": 0.0,
        "split_correction_time_kind": "not_applicable",
        "smooth_residual_total_s": 0.0,
        "smooth_residual_mean_s": 0.0,
        "smooth_residual_time_kind": "not_applicable",
        "coefficient_eval_mean_s": 0.0,
        "coefficient_eval_time_kind": "not_applicable_no_retained_channels",
    }

    if split:
        correction_args = (
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            reordered_weighted_sources,
        )

        def split_correction():
            correction, _ = wrangler.eval_direct_helmholtz_split_correction(
                *correction_args,
                src_func=reordered_source_vals,
            )
            return correction

        def split_full_apply():
            base_values = table_apply()
            correction = split_correction()
            return base_values[0] + correction[0]

        _, split_full_total_s, split_full_mean_s = _time_repeated(
            queue, repeat_count, split_full_apply
        )
        _, correction_total_s, correction_mean_s = _time_repeated(
            queue, repeat_count, split_correction
        )

        method_name = "_helmholtz_split_extra_terms"
        setattr(wrangler, method_name, MethodType(lambda self: [], wrangler))
        try:
            _, residual_total_s, residual_mean_s = _time_repeated(
                queue, repeat_count, split_correction
            )
        finally:
            delattr(wrangler, method_name)

        timing.update(
            {
                "split_full_apply_total_s": split_full_total_s,
                "split_full_apply_mean_s": split_full_mean_s,
                "split_correction_total_s": correction_total_s,
                "split_correction_mean_s": correction_mean_s,
                "split_correction_time_kind": "isolated_list1_split_correction",
                "smooth_residual_total_s": residual_total_s,
                "smooth_residual_mean_s": residual_mean_s,
                "smooth_residual_time_kind": (
                    "isolated_implemented_residual_correction_without_"
                    "retained_channels"
                ),
            }
        )

        if split_order > 1:
            coefficient_repeats = 1000
            _ = wrangler._helmholtz_split_extra_terms()
            start = time.perf_counter()
            for _ in range(coefficient_repeats):
                _ = wrangler._helmholtz_split_extra_terms()
            coefficient_total_s = time.perf_counter() - start
            timing["coefficient_eval_mean_s"] = (
                coefficient_total_s / coefficient_repeats
            )
            timing["coefficient_eval_time_kind"] = (
                "isolated_warm_python_scalar_coefficients"
            )

    timing.update(
        _phase_measurements(
            queue=queue,
            traversal=traversal,
            wrangler=wrangler,
            solve=solve,
            phase_repeat_count=phase_repeat_count,
            split=split,
            q_order=q_order,
            split_order=split_order,
            split_smooth_quad_order=split_smooth_quad_order,
        )
    )

    return potential.get(queue), timing, wrangler


def _tensor_product_interp_fmas(*, dim: int, q: int, q_smooth: int) -> int:
    """FMAs of one box's smooth-quadrature interpolation, as executed.

    ``_interpolate_box_values_to_smooth_quad`` applies the 1D barycentric
    matrix one axis at a time (``interp_mat @ v @ interp_mat.T`` in 2D, the
    optimized ``einsum`` path in 3D), so the cost is the tensor-product sum
    ``sum_{k=1..d} q_smooth**k * q**(d-k+1)`` and *not* the dense
    ``q_smooth**d * q**d`` a matrix-free reading would suggest.  At the
    production configuration (``d = 2``, ``q = 4``, ``q_smooth = 8``) this
    is 384 per box against 1024 dense, a factor of 2.67.
    """
    dim = int(dim)
    q = int(q)
    q_smooth = int(q_smooth)
    return int(
        sum(q_smooth ** (axis + 1) * q ** (dim - axis) for axis in range(dim))
    )


def _remainder_terms_per_pair(wrangler) -> int:
    """Terms the split remainder kernel evaluates at each near-field pair.

    Counted from the *generated* expression, not from a formula: the
    series length ``nmax`` is not the term count.  In 2D
    ``_HelmholtzSplitSeriesRemainderKernel`` emits a constant, one
    ``r**(2n)`` term for every ``n = 1 .. nmax``, and a second
    ``r**(2n) log r`` term for every ``n >= split_order`` -- roughly twice
    ``nmax``.  In 3D it emits one ``r**(n-1)`` term per retained ``n``,
    skipping the even powers the tables extract.  Deriving the multiplier
    from the expression means a change to either branch cannot leave this
    count stale.
    """
    import pymbolic.primitives as prim

    kernel = wrangler._get_helmholtz_split_remainder_kernel()
    expression = kernel.expression
    while True:
        # unwrap whatever the kernel-wrapper chain put around it
        inner = getattr(expression, "expression", None)
        if inner is None or inner is expression:
            break
        expression = inner

    def _count(expr):
        if isinstance(expr, prim.Sum):
            return sum(_count(child) for child in expr.children)
        # the 3D branch starts from a literal ``expr = 0`` that pymbolic
        # keeps as a Sum child; it is not an evaluated term
        if isinstance(expr, int | float | complex | np.number) and expr == 0:
            return 0
        return 1

    return int(_count(expression))


def _split_correction_operation_counts(
    *, queue, traversal, wrangler, q_order, smooth_quad_order
):
    """Executed operation counts of the online split correction phase.

    The correction phase runs, in this order (see
    ``FPNDSumpyExpansionWrangler.eval_direct_helmholtz_split_correction``):
    an optional interpolated smooth-source rebuild, one near-field P2P for
    the series remainder, and, per retained term, one table apply plus (for
    2D single-table ``power_log`` terms under the ``p2p`` beta mode) one
    further near-field P2P.  Every count below is read off the executed
    wrangler and traversal, so it prices what ran rather than the symbolic
    ``Delta W`` of the cost model.

    Two conventions are inherited from the pre-existing ``ops_*`` columns
    and are worth stating because they make these counts *lower bounds* on
    executed kernel launches rather than launch counts:

    * a table apply is priced at one FMA per near-field pair per applied
      table, blind to the arithmetic dtype.  This driver runs a complex128
      source function, and ``_eval_direct_helmholtz_split_term_table``
      dispatches a real-valued term kernel once for the real part and once
      for the imaginary part, so each retained channel executes *two* List 1
      passes for the one apply counted here.  Counting it once is what keeps
      ``base + extra == ops_split_table_fmas_per_solve``, the published
      ``p``-times-direct identity, and it prices the two strategies on the
      same dtype-blind footing;
    * the smooth-source rebuild is priced by
      :func:`_tensor_product_interp_fmas`, i.e. as the axis-by-axis
      contraction the implementation performs.

    :returns: a dict of counts plus a ``status`` string; on any failure to
        interrogate the wrangler the counts are ``""`` and ``status``
        carries the exception, so an implementation change shows up as a
        blank rather than as a wrong number.
    """
    import volumential.opcounters as opcounters
    from volumential.expansion_wrangler_fpnd import (
        _normalize_helmholtz_split_term_key,
    )

    blank = {
        "extra_table_fmas": "",
        "remainder_pair_evals": "",
        "beta_p2p_pair_evals": "",
        "smooth_interp_fmas": "",
        "smooth_sources_per_box": "",
        "remainder_terms_per_pair": "",
        "status": "",
    }

    try:
        tree = traversal.tree
        dim = int(tree.dimensions)
        n_quad_points = int(q_order) ** dim

        split_order = int(wrangler.helmholtz_split_order)
        use_series_remainder_path = split_order > 1 or (
            split_order == 1
            and not wrangler.helmholtz_split_order1_legacy_subtraction
        )
        smooth_order = (
            None if smooth_quad_order is None else int(smooth_quad_order)
        )
        use_interp_smooth_quad = (
            smooth_order is not None and smooth_order > int(q_order)
        )

        if use_interp_smooth_quad:
            interp_data = wrangler._get_helmholtz_split_smooth_interp_data(
                smooth_order,
                allow_node_overlap=use_series_remainder_path,
            )
            smooth_sources_per_box = int(interp_data["n_smooth_points"])
        else:
            smooth_sources_per_box = n_quad_points

        box_source_counts = np.asarray(
            tree.box_source_counts_nonchild.get(queue), dtype=np.int64
        )
        smooth_counts = np.where(
            box_source_counts > 0, smooth_sources_per_box, 0
        ).astype(np.int64)
        remainder_pair_evals = opcounters.nearfield_point_pairs_from_counts(
            target_boxes=traversal.target_boxes.get(queue),
            neighbor_source_boxes_starts=(
                traversal.neighbor_source_boxes_starts.get(queue)
            ),
            neighbor_source_boxes_lists=(
                traversal.neighbor_source_boxes_lists.get(queue)
            ),
            box_target_counts_nonchild=(
                tree.box_target_counts_nonchild.get(queue)
            ),
            box_source_counts_nonchild=smooth_counts,
        )

        # exclude_self: on the base-quadrature path the correction keeps
        # target_to_source and passes the tree's own exclude_self flag, so
        # the P2P skips each target's own source.  The interpolated path
        # pops target_to_source and passes False.  Counting the skipped
        # diagonal would overstate the remainder, and the beta P2P below,
        # by one pair per target.
        excluded_self_pairs = 0
        if not use_interp_smooth_quad and getattr(
            getattr(wrangler, "tree_indep", None), "exclude_self", False
        ):
            excluded_self_pairs = int(
                np.sum(
                    np.asarray(
                        tree.box_target_counts_nonchild.get(queue),
                        dtype=np.int64,
                    )[np.asarray(traversal.target_boxes.get(queue))]
                )
            )
        remainder_pair_evals = max(
            0, int(remainder_pair_evals) - excluded_self_pairs
        )

        if use_interp_smooth_quad:
            n_active_source_boxes = int(np.count_nonzero(box_source_counts))
            smooth_interp_fmas = (
                n_active_source_boxes
                * _tensor_product_interp_fmas(
                    dim=dim, q=int(q_order), q_smooth=int(smooth_order)
                )
            )
        else:
            smooth_interp_fmas = 0

        nearfield_pairs = opcounters.nearfield_point_pairs(queue, traversal)
        beta_mode = (
            str(
                wrangler._helmholtz_split_auto_config.get(
                    "power_log_single_table_beta_mode", "p2p"
                )
            )
            .strip()
            .lower()
        )
        extra_table_applies = 0
        beta_p2p_passes = 0
        for term_key, _kernel, _coeff in wrangler._helmholtz_split_extra_terms():
            term_tables = wrangler._get_helmholtz_split_term_tables(term_key)
            n_term_tables = (
                len(term_tables) if isinstance(term_tables, list) else 1
            )
            extra_table_applies += 1
            term_kind, _term_power = _normalize_helmholtz_split_term_key(
                term_key
            )
            if dim == 2 and term_kind == "power_log" and n_term_tables == 1:
                if beta_mode == "p2p":
                    beta_p2p_passes += 1
                else:
                    # the "table" beta mode applies one more table instead
                    extra_table_applies += 1

        return {
            "extra_table_fmas": extra_table_applies * nearfield_pairs,
            "remainder_pair_evals": remainder_pair_evals,
            "beta_p2p_pair_evals": beta_p2p_passes * remainder_pair_evals,
            "smooth_interp_fmas": smooth_interp_fmas,
            "smooth_sources_per_box": smooth_sources_per_box,
            "remainder_terms_per_pair": _remainder_terms_per_pair(wrangler),
            "status": (
                "interpolated_smooth_quadrature"
                if use_interp_smooth_quad
                else "base_quadrature"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - recorded, never guessed around
        return {**blank, "status": f"unavailable:{type(exc).__name__}: {exc}"}


def _phase_correction_op_columns(
    *, queue, traversal, wrangler, split, q_order, split_order,
    split_smooth_quad_order,
):
    """The ``ops_phase_split_correction_*`` columns of one path.

    A path that runs no online split correction -- the direct reference,
    and the windowed-assembled path, which rides the unchanged direct warm
    path -- reports a structural zero with an explicit status, so a blank
    always means "could not be counted" and never "there was none".
    """
    zero = {
        "ops_phase_split_correction_extra_table_fmas": 0,
        "ops_phase_split_correction_remainder_pair_evals": 0,
        "ops_phase_split_correction_remainder_term_evals": 0,
        "ops_phase_split_correction_remainder_terms_per_pair": 0,
        "ops_phase_split_correction_beta_p2p_pair_evals": 0,
        "ops_phase_split_correction_smooth_interp_fmas": 0,
        "ops_phase_split_smooth_sources_per_box": 0,
        "ops_phase_split_correction_status": "no_split_correction",
        "ops_phase_split_correction_total": 0,
    }
    if not split or q_order is None:
        return zero

    correction = _split_correction_operation_counts(
        queue=queue,
        traversal=traversal,
        wrangler=wrangler,
        q_order=q_order,
        smooth_quad_order=split_smooth_quad_order,
    )
    remainder_pairs = correction["remainder_pair_evals"]
    terms_per_pair = correction.get("remainder_terms_per_pair", "")
    if remainder_pairs == "":
        # the wrangler could not be interrogated: every count that depends
        # on it, and the solve total that would swallow it, stay blank
        return {
            f"ops_phase_split_correction_{name}": correction.get(name, "")
            for name in (
                "extra_table_fmas", "remainder_pair_evals",
                "beta_p2p_pair_evals", "smooth_interp_fmas",
                "remainder_terms_per_pair", "status",
            )
        } | {
            "ops_phase_split_correction_remainder_term_evals": "",
            "ops_phase_split_smooth_sources_per_box": (
                correction["smooth_sources_per_box"]
            ),
            "ops_phase_split_correction_total": "",
        }

    if terms_per_pair == "":
        # the remainder kernel could not be interrogated, so the term count
        # this multiplies by is unknown; the same policy as a blank pair
        # count applies
        return {
            **zero,
            "ops_phase_split_correction_status": correction["status"],
            "ops_phase_split_correction_total": "",
        }

    remainder_term_evals = float(remainder_pairs) * float(terms_per_pair)
    total = (
        float(correction["extra_table_fmas"])
        + remainder_term_evals
        + float(correction["beta_p2p_pair_evals"])
        + float(correction["smooth_interp_fmas"])
    )
    return {
        "ops_phase_split_correction_extra_table_fmas": (
            correction["extra_table_fmas"]
        ),
        "ops_phase_split_correction_remainder_pair_evals": remainder_pairs,
        "ops_phase_split_correction_remainder_term_evals": (
            remainder_term_evals
        ),
        "ops_phase_split_correction_remainder_terms_per_pair": (
            terms_per_pair
        ),
        "ops_phase_split_correction_beta_p2p_pair_evals": (
            correction["beta_p2p_pair_evals"]
        ),
        "ops_phase_split_correction_smooth_interp_fmas": (
            correction["smooth_interp_fmas"]
        ),
        "ops_phase_split_smooth_sources_per_box": (
            correction["smooth_sources_per_box"]
        ),
        "ops_phase_split_correction_status": correction["status"],
        "ops_phase_split_correction_total": total,
    }


def _phase_measurements(
    *,
    queue,
    traversal,
    wrangler,
    solve,
    phase_repeat_count: int,
    split: bool = False,
    q_order: int | None = None,
    split_order: int = 1,
    split_smooth_quad_order: int | None = None,
):
    """Per-phase operation counts and seconds of one end-to-end solve (E6).

    Runs ``phase_repeat_count`` extra, phase-instrumented solves *after*
    every timed phase, so no reported timing column is perturbed.  Phase
    profiling synchronizes the command queue at every phase boundary, which
    makes a profiled solve slower than an unprofiled one; the profiled total
    is reported next to the shares so the perturbation stays visible.  With
    ``phase_repeat_count == 0`` nothing runs and every phase value is left
    unmeasured.

    Counting rules are those of ``benchmarks/break_even_validation.py``,
    literally: the far-field counts are dense coefficient touches derived
    from the executed traversal and expansion sizes, the near-field pair
    count is one fused multiply-add per (target point, source quadrature
    point) pair per applied table, and the split path's correction counts
    come from :func:`_split_correction_operation_counts`, the same function
    that driver calls.  Both are recorded under the same
    :data:`PHASE_COUNTING_RULE` tag.

    ``ops_phase_solve_total_*`` is therefore a genuine per-path total: far
    field plus the base near-field apply, plus, on an online-split path, the
    correction phase -- the retained-channel applies, the series remainder
    over its (possibly interpolated) smooth source set, the ``power_log``
    beta P2P, and the smooth-source rebuild.  On a path that runs no online
    split (``split=False``: the direct reference, and the
    windowed-assembled path, which rides the unchanged direct warm path) the
    correction counts are a structural 0, not a blank.
    """
    measurements = {
        "phase_profile_repeat_count": phase_repeat_count,
        "phase_counting_rule": PHASE_UNMEASURED,
        "ops_phase_far_total": PHASE_UNMEASURED,
        "ops_phase_nearfield_point_pairs_per_solve": PHASE_UNMEASURED,
        "ops_phase_solve_total": PHASE_UNMEASURED,
        **{
            name: PHASE_UNMEASURED for name in PHASE_CORRECTION_OPS_NAMES
        },
        **{
            f"ops_phase_far_{stage}": PHASE_UNMEASURED
            for stage in PHASE_FAR_STAGES
        },
        **{
            f"s_phase_{name}": PHASE_UNMEASURED
            for name in PHASE_TIMED_NAMES
        },
    }
    if phase_repeat_count <= 0:
        return measurements

    import volumential.opcounters as opcounters
    from volumential.phase_profile import (
        FAR_FIELD_PHASES,
        PhaseProfile,
        profiling,
    )

    far = opcounters.fmm_stage_operation_counts_from_traversal(
        queue, traversal, wrangler
    )
    measurements["ops_phase_far_total"] = far["far_total"]
    for stage in PHASE_FAR_STAGES:
        measurements[f"ops_phase_far_{stage}"] = far[stage]
    nearfield_pairs = opcounters.nearfield_point_pairs(queue, traversal)
    measurements["ops_phase_nearfield_point_pairs_per_solve"] = nearfield_pairs
    measurements["phase_counting_rule"] = PHASE_COUNTING_RULE
    measurements.update(
        _phase_correction_op_columns(
            queue=queue,
            traversal=traversal,
            wrangler=wrangler,
            split=split,
            q_order=q_order,
            split_order=split_order,
            split_smooth_quad_order=split_smooth_quad_order,
        )
    )
    correction_total = measurements["ops_phase_split_correction_total"]
    if correction_total != PHASE_UNMEASURED:
        measurements["ops_phase_solve_total"] = (
            float(far["far_total"])
            + float(nearfield_pairs)
            + float(correction_total)
        )

    profile = PhaseProfile(sync=queue.finish)
    solve_total_s = 0.0
    for _ in range(phase_repeat_count):
        queue.finish()
        start = time.perf_counter()
        with profiling(profile):
            solve()
        queue.finish()
        solve_total_s += time.perf_counter() - start

    solve_total_s /= phase_repeat_count
    far_total_s = sum(
        profile.seconds(name) for name in FAR_FIELD_PHASES
    ) / phase_repeat_count
    table_s = (
        profile.seconds("nearfield_table_apply") / phase_repeat_count
    )
    correction_s = profile.seconds("split_correction") / phase_repeat_count
    measurements.update(
        {
            "s_phase_far_total": far_total_s,
            "s_phase_nearfield_table_apply": table_s,
            "s_phase_split_correction": correction_s,
            "s_phase_other": solve_total_s
            - far_total_s
            - table_s
            - correction_s,
            "s_phase_solve_total": solve_total_s,
        }
    )
    return measurements


def _get_direct_table(
    *,
    kernel: str,
    queue,
    cache_path: Path,
    q_order: int,
    parameter: float,
    level: int,
    build_config=None,
    dim: int = 2,
):
    if kernel == "Helmholtz":
        return _build_helmholtz_table(
            queue,
            cache_path,
            dim,
            q_order,
            parameter,
            level,
            build_config=build_config,
        )
    if kernel == "Yukawa":
        return _get_yukawa_table(
            queue,
            cache_path,
            dim,
            q_order,
            parameter,
            level,
            build_config=build_config,
        )
    raise ValueError(f"unknown kernel: {kernel}")


def _prepare_direct_tables(
    *,
    kernel: str,
    queue,
    cache_dir: Path,
    q_order: int,
    parameter: float,
    direct_levels: list[int],
    active_level: int,
    build_config=None,
    dim: int = 2,
):
    parameter_tag = f"{parameter:.17g}".replace("-", "m").replace(".", "p")
    cache_path = cache_dir / (
        f"cost-direct-{kernel.lower()}-{dim}d-parameter{parameter_tag}"
        f"-q{q_order}.sqlite"
    )
    _clear_sqlite_cache(cache_path)

    with _capture_table_get_timings() as cold_records:
        for level in direct_levels:
            _get_direct_table(
                kernel=kernel,
                queue=queue,
                cache_path=cache_path,
                dim=dim,
                q_order=q_order,
                parameter=parameter,
                level=level,
                build_config=build_config,
            )

    warm_tables = {}
    with _capture_table_get_timings() as warm_records:
        for level in direct_levels:
            warm_tables[level] = _get_direct_table(
                kernel=kernel,
                queue=queue,
                cache_path=cache_path,
                dim=dim,
                q_order=q_order,
                parameter=parameter,
                level=level,
                build_config=build_config,
            )

    cold = _summarize_table_get_timings(cold_records)
    warm = _summarize_table_get_timings(warm_records)
    if cold["build_count"] != len(direct_levels):
        raise RuntimeError("direct-table cold pass did not build every requested level")
    if warm["load_count"] != len(direct_levels) or warm["build_count"]:
        raise RuntimeError("direct-table warm pass was not a pure cache load")
    return warm_tables[active_level], {
        "build_s": cold["build_s"],
        "quadrature_build_s": cold["quadrature_build_s"],
        "load_s": warm["load_s"],
        "cache_payload_bytes": warm["cache_payload_bytes"],
        "table_count": len(warm_tables),
        "payload_bytes": sum(_table_payload_bytes(table)
                             for table in warm_tables.values()),
        # the routing recorded by the builder and carried through the cache
        # round trip (these tables are the warm, cache-loaded ones)
        "build_routing": _table_build_routing(warm_tables.values()),
        # ... and per routing, so a caller pricing every provisioned table
        # can weight the routings instead of sampling one of them
        "build_routing_counts": _table_build_routing_counts(
            warm_tables.values()
        ),
    }


def _prepare_rke_channels(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    kernel: str,
    parameter: float,
    split_order: int,
    source_weights,
    q_points,
    source_values_host,
    cache_dir: Path,
    split_auto_config=None,
    build_config=None,
    split_smooth_quad_order: int | None = None,
    cache_path: Path | None = None,
    clear_cache: bool = True,
    dim: int = 2,
):
    if cache_path is None:
        cache_path = cache_dir / (
            f"cost-rke-{kernel.lower()}-{dim}d-q{q_order}-p{split_order}.sqlite"
        )
    if clear_cache:
        _clear_sqlite_cache(cache_path)

    with _capture_table_get_timings() as cold_records:
        cold_base_table = _get_laplace_table(
            queue, cache_path, dim, q_order, build_config=build_config
        )
        _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            dim=dim,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=cold_base_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_auto_config=split_auto_config,
            split_smooth_quad_order=split_smooth_quad_order,
        )

    with _capture_table_get_timings() as warm_records:
        warm_base_table = _get_laplace_table(
            queue, cache_path, dim, q_order, build_config=build_config
        )
        warm_wrangler, _, _ = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            dim=dim,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=warm_base_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_auto_config=split_auto_config,
            split_smooth_quad_order=split_smooth_quad_order,
        )

    cold = _summarize_table_get_timings(cold_records)
    warm = _summarize_table_get_timings(warm_records)
    if cold["build_count"] < 1:
        raise RuntimeError("RKE cold pass did not build a new table")
    if cold["build_count"] + cold["load_count"] != warm["load_count"]:
        raise RuntimeError("RKE cold and warm channel request counts differ")
    if warm["build_count"]:
        raise RuntimeError("RKE warm pass unexpectedly rebuilt a channel table")
    return (
        warm_base_table,
        dict(warm_wrangler.helmholtz_split_term_tables),
        {
            "build_s": cold["build_s"],
            "quadrature_build_s": cold["quadrature_build_s"],
            "load_s": warm["load_s"],
            "cache_payload_bytes": warm["cache_payload_bytes"],
        },
    )


# {{{ windowed-assembled table-provisioning strategy (E1)

def _parse_windowed_thetas(raw: str | None, mode: str) -> list[float]:
    if raw is None:
        defaults = (
            DEFAULT_SMOKE_WINDOWED_THETAS
            if mode == "smoke"
            else DEFAULT_FULL_WINDOWED_THETAS
        )
        return list(defaults)
    if raw.strip().lower() in {"", "none", "skip"}:
        return []
    values = _parse_csv_floats(raw)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("windowed thetas must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("windowed thetas must be unique")
    return values


def _prepare_windowed_family(
    *,
    cache_path: Path,
    q_order: int,
    source_box_level: int,
    window_theta: float,
    p_star: int,
    chan_regular_order: int,
    chan_radial_order: int,
    root_extent: float = TABLE_ROOT_EXTENT,
    dim: int = 2,
) -> dict[str, Any]:
    """Build or reload the parameter-independent windowed channel family.

    ``dim`` defaults to 2 so every 2D caller keeps its exact behaviour; the
    3D composition driver passes ``dim=3`` to build the same family on the
    3D channel geometry.
    """
    from volumential.rke_table_assembly import get_windowed_channel_table

    dim = _require_dimension(dim)
    was_cold = False
    start = time.perf_counter()
    for m in range(p_star):
        channel = get_windowed_channel_table(
            cache_path,
            dim,
            q_order,
            m,
            source_box_level=source_box_level,
            root_extent=root_extent,
            window_theta=window_theta,
            chan_regular_order=chan_regular_order,
            chan_radial_order=chan_radial_order,
        )
        disposition = getattr(channel, "_windowed_cache_disposition", None)
        if disposition not in ("hit", "rebuilt"):
            raise RuntimeError(
                "windowed channel table did not report a valid cache "
                "disposition"
            )
        was_cold = was_cold or disposition == "rebuilt"
    return {
        "build_s": time.perf_counter() - start,
        "was_cold": was_cold,
    }


def _classical_certificate_probe(
    *,
    queue,
    cache_path: Path,
    kernel: str,
    q_order: int,
    parameter: float,
    source_box_level: int,
    tolerance: float,
    probe_kind: str,
    dim: int = 2,
) -> dict[str, Any]:
    """Certificate status of the polynomial-completion assembler at this
    parameter: ``certified`` / ``refused`` / ``failed`` (plus ``skipped``).

    ``probe_kind == "truncation"`` checks only the (cheap) series-tail
    certificate; ``"full"`` runs the complete certified assembly, exposing
    both refusal modes (term budget and recombination conditioning).
    """
    dim = _require_dimension(dim)
    empty = {
        "kind": probe_kind,
        "status": "skipped",
        "detail": "",
        "n_terms": "",
        "condition_number": "",
        "probe_s": "",
    }
    if probe_kind == "off":
        return empty

    from volumential.rke_table_assembly import (
        RKEConditioningError,
        RKETruncationError,
        assemble_parameterized_table,
        choose_truncation_order,
    )

    start = time.perf_counter()
    if probe_kind == "truncation":
        box_extent = _box_extent(source_box_level)
        # The conservative near-field separation radius of
        # rke_table_assembly.assemble_parameterized_table.
        radius = 3.0 * math.sqrt(dim) * box_extent
        k = (
            complex(parameter)
            if kernel == "Helmholtz"
            else complex(1j * parameter)
        )
        try:
            n_terms, _ = choose_truncation_order(dim, k, radius, tolerance)
        except RKETruncationError as exc:
            return {
                **empty,
                "status": "refused",
                "detail": f"{type(exc).__name__}: {exc}",
                "probe_s": time.perf_counter() - start,
            }
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            return {
                **empty,
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "probe_s": time.perf_counter() - start,
            }
        return {
            **empty,
            "status": "certified-truncation-only",
            "n_terms": int(n_terms),
            "probe_s": time.perf_counter() - start,
        }

    if probe_kind != "full":
        raise ValueError(f"unknown classical probe kind: {probe_kind}")
    try:
        _, certificate = assemble_parameterized_table(
            queue,
            cache_path,
            dim,
            kernel,
            q_order,
            parameter,
            source_box_level=source_box_level,
            root_extent=TABLE_ROOT_EXTENT,
            tolerance=tolerance,
        )
    except (RKETruncationError, RKEConditioningError) as exc:
        return {
            **empty,
            "status": "refused",
            "detail": f"{type(exc).__name__}: {exc}",
            "probe_s": time.perf_counter() - start,
        }
    except Exception as exc:  # noqa: BLE001 - recorded for the taxonomy
        return {
            **empty,
            "status": "failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "probe_s": time.perf_counter() - start,
        }
    return {
        **empty,
        "status": "certified",
        "n_terms": int(certificate["n_series_terms"]),
        "condition_number": float(certificate["condition_number"]),
        "probe_s": time.perf_counter() - start,
    }


def _register_and_load_windowed_table(
    *,
    queue,
    cache_path: Path,
    kernel: str,
    q_order: int,
    parameter: float,
    source_box_level: int,
    table,
    certificate: dict[str, Any],
    root_extent: float = TABLE_ROOT_EXTENT,
    dim: int = 2,
) -> tuple[Any, dict[str, Any]]:
    """Register the assembled table under the standard cache slot, then load
    it back through the ordinary ``get_table`` path (asserting a pure cache
    load), so the evaluator consumes it exactly like a direct-built table.

    ``dim`` defaults to 2 so every 2D caller keeps its exact behaviour.
    """
    from volumential.table_manager import NearFieldInteractionTableManager

    dim = _require_dimension(dim)
    manager_kwargs: dict[str, Any] = {}
    get_kwargs: dict[str, Any] = {}
    if kernel == "Helmholtz":
        from sumpy.kernel import HelmholtzKernel

        knl = HelmholtzKernel(dim)
        manager_kwargs["dtype"] = np.complex128
        get_kwargs["sumpy_knl"] = knl
        get_kwargs[knl.helmholtz_k_name] = float(parameter)
        kernel_request = "Helmholtz-Reference"
    elif kernel == "Yukawa":
        get_kwargs["lam"] = float(parameter)
        kernel_request = "Yukawa"
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    _clear_sqlite_cache(cache_path)
    provenance = {
        "kind": "windowed_rke_assembly",
        "window_theta": float(certificate["window_theta"]),
        "p_star": int(certificate["p_star"]),
        "smooth_quad_order": int(certificate["smooth_quad_order"]),
        "condition_number": float(certificate["condition_number"]),
    }
    register_start = time.perf_counter()
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=root_extent, queue=queue,
        **manager_kwargs,
    ) as table_manager:
        table_manager.register_external_table(
            dim,
            kernel_request,
            q_order,
            table,
            source_box_level=source_box_level,
            provenance=provenance,
            **get_kwargs,
        )
        register_payload_bytes = int(
            table_manager.last_register_timings["payload_bytes"]
        )
    register_s = time.perf_counter() - register_start

    # Registration is done.  If the reload below fails, a caller turning
    # that into a failure row still needs to account for it, or the row
    # charges the reload's wall time to registration while reporting a
    # zero payload for work that completed.  The partial metrics ride on
    # the exception under a name the caller looks for.
    #
    # The reload's own wall time counts too: a get_table() that raises
    # after a long read, and a reload that completes but is rejected below
    # for not being a pure cache hit, both spent time every caller copies
    # into its failed row.  The instrumented load_s is only available when
    # the capture completed, so this clock is the fallback.
    partial_transfer = {
        "register_s": register_s,
        "register_payload_bytes": register_payload_bytes,
        "load_s": 0.0,
        "load_payload_bytes": 0,
    }

    load_start = time.perf_counter()
    try:
        with _capture_table_get_timings() as load_records:
            with NearFieldInteractionTableManager(
                str(cache_path), root_extent=root_extent, queue=queue,
                **manager_kwargs,
            ) as table_manager:
                loaded_table, is_recomputed = table_manager.get_table(
                    dim,
                    kernel_request,
                    q_order,
                    source_box_level=source_box_level,
                    queue=queue,
                    **get_kwargs,
                )
        partial_transfer["load_s"] = time.perf_counter() - load_start
        if is_recomputed:
            raise RuntimeError(
                "registered windowed table did not load as a pure cache hit"
            )
        load_summary = _summarize_table_get_timings(load_records)
        partial_transfer["load_s"] = load_summary["load_s"]
        partial_transfer["load_payload_bytes"] = load_summary[
            "cache_payload_bytes"
        ]
        if load_summary["build_count"] or load_summary["load_count"] != 1:
            raise RuntimeError(
                "registered windowed table load pass was not a single pure "
                "load"
            )
    except BaseException as exc:
        if not partial_transfer["load_s"]:
            # nothing instrumented got recorded: fall back to the clock
            partial_transfer["load_s"] = time.perf_counter() - load_start
        with contextlib.suppress(AttributeError):
            exc.partial_windowed_transfer = dict(partial_transfer)
        raise

    return loaded_table, {
        "register_s": register_s,
        "register_payload_bytes": register_payload_bytes,
        "load_s": load_summary["load_s"],
        "load_payload_bytes": load_summary["cache_payload_bytes"],
    }


def _windowed_row_base(
    *,
    mode: str,
    dim: int,
    kernel: str,
    parameter_name: str,
    parameter: float,
    theta: float,
    window_theta: float,
    p_star: int,
    chan_orders: tuple[int, int],
    direct_build_config,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    fmm_order_rule: str,
    fmm_order_floor: int,
    far_field_status: str,
    repeat_count: int,
    classical_probe: dict[str, Any],
) -> dict[str, Any]:
    dim = _require_dimension(dim)
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "case_id": (
                f"{kernel.lower()}{dim}d-{parameter_name}"
                f"{_case_parameter_token(parameter)}"
                f"-windowed-theta{_case_parameter_token(theta)}"
            ),
            "mode": mode,
            "kernel": kernel,
            "dim": dim,
            "parameter_name": parameter_name,
            "parameter_value": parameter,
            "direct_regular_quad_order": direct_build_config.regular_quad_order,
            "direct_radial_quad_order": direct_build_config.radial_quad_order,
            "q_order": q_order,
            "nlevels": nlevels,
            "fmm_order": fmm_order,
            "fmm_order_rule": fmm_order_rule,
            "fmm_order_floor": fmm_order_floor,
            "fmm_expansion_radius": _fmm_expansion_radius(dim),
            "far_field_status": far_field_status,
            "reference_path": "direct_fixed_parameter_table",
            "repeat_count": repeat_count,
            "table_strategy": "windowed_assembled",
            "theta": theta,
            "window_theta": window_theta,
            "windowed_p_star": p_star,
            "windowed_chan_regular_order": chan_orders[0],
            "windowed_chan_radial_order": chan_orders[1],
            "classical_probe_kind": classical_probe["kind"],
            "classical_probe_status": classical_probe["status"],
            "classical_probe_detail": classical_probe["detail"],
            "classical_probe_n_terms": classical_probe["n_terms"],
            "classical_probe_condition_number": classical_probe[
                "condition_number"
            ],
            "classical_probe_s": classical_probe["probe_s"],
        }
    )
    return row


def _run_windowed_strategy(
    *,
    mode: str,
    ctx,
    queue,
    traversal,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    dim: int = 2,
    fmm_order_rule: str = "fixed",
    max_fmm_order: int | None = None,
    kernel: str,
    parameter_name: str,
    thetas: list[float],
    window_theta: float,
    p_star: int,
    chan_orders: tuple[int, int],
    classical_probe_kind: str,
    classical_probe_tolerance: float,
    direct_build_config,
    repeat_count: int,
    phase_repeat_count: int,
    source_weights,
    q_points,
    coords_host,
) -> list[dict[str, Any]]:
    from volumential.rke_table_assembly import (
        RKEWindowConditioningError,
        RKEWindowCoverageError,
        assemble_windowed_parameterized_table,
    )

    dim = _require_dimension(dim)
    box_extent = _box_extent(nlevels)
    family_cache = cache_dir / (
        f"windowed-channels-{dim}d-q{q_order}-l{nlevels}"
        f"-Theta{window_theta:g}.sqlite"
    )
    classical_cache = cache_dir / (
        f"classical-probe-{dim}d-q{q_order}-l{nlevels}.sqlite"
    )

    # The one-off family build is provisioning too, and the fixed-parameter
    # direct and online-split rows of this kernel have already been measured
    # when it runs.  main() preserves rows only for _BenchmarkGateError and
    # otherwise writes the CSV after run_benchmark() returns, so a corrupt,
    # locked or unwritable family cache would discard those measurements
    # instead of emitting a failed windowed row per requested theta.
    family_start = time.perf_counter()
    family_failure: tuple[str, str] | None = None
    try:
        family = _prepare_windowed_family(
            cache_path=family_cache,
            dim=dim,
            q_order=q_order,
            source_box_level=nlevels,
            window_theta=window_theta,
            p_star=p_star,
            chan_regular_order=chan_orders[0],
            chan_radial_order=chan_orders[1],
        )
    except (RKEWindowCoverageError, RKEWindowConditioningError) as exc:
        family_failure = ("refused", f"{type(exc).__name__}: {exc}")
    except (
        ValueError, RuntimeError, NotImplementedError, OSError, KeyError,
        # sqlite3's exceptions descend from Exception, not OSError
        sqlite3.Error,
    ) as exc:
        family_failure = ("failed", f"{type(exc).__name__}: {exc}")
    if family_failure is not None:
        family = {
            "build_s": time.perf_counter() - family_start,
            "was_cold": True,
        }

    rows: list[dict[str, Any]] = []
    for theta in thetas:
        parameter = theta / box_extent
        row_fmm_order = _prescribed_fmm_order(
            dim,
            kernel,
            parameter,
            floor=fmm_order,
            rule=fmm_order_rule,
        )
        if max_fmm_order is not None and row_fmm_order > max_fmm_order:
            far_field_status = "refused_order_cap"
        elif fmm_order_rule == "resolved" and kernel == "Helmholtz":
            far_field_status = "resolved_by_rule"
        else:
            far_field_status = "pinned"
        classical_probe = _classical_certificate_probe(
            queue=queue,
            cache_path=classical_cache,
            dim=dim,
            kernel=kernel,
            q_order=q_order,
            parameter=parameter,
            source_box_level=nlevels,
            tolerance=classical_probe_tolerance,
            probe_kind=classical_probe_kind,
        )
        row = _windowed_row_base(
            mode=mode,
            dim=dim,
            kernel=kernel,
            parameter_name=parameter_name,
            parameter=parameter,
            theta=theta,
            window_theta=window_theta,
            p_star=p_star,
            chan_orders=chan_orders,
            direct_build_config=direct_build_config,
            q_order=q_order,
            nlevels=nlevels,
            fmm_order=row_fmm_order,
            fmm_order_rule=fmm_order_rule,
            fmm_order_floor=fmm_order,
            far_field_status=far_field_status,
            repeat_count=repeat_count,
            classical_probe=classical_probe,
        )
        row["windowed_channel_build_s"] = family["build_s"]
        row["windowed_channel_build_was_cold"] = int(family["was_cold"])

        if family_failure is not None:
            # the family every theta of this kernel would have read never
            # got built: one row per requested theta, carrying the same
            # taxonomy the assembly and transfer use
            row["windowed_status"] = family_failure[0]
            row["windowed_refusal"] = (
                f"windowed channel family: {family_failure[1]}"
            )
            rows.append(row)
            continue

        assemble_start = time.perf_counter()
        try:
            assembled_table, certificate = (
                assemble_windowed_parameterized_table(
                    family_cache,
                    dim,
                    kernel,
                    q_order,
                    parameter,
                    source_box_level=nlevels,
                    root_extent=TABLE_ROOT_EXTENT,
                    window_theta=window_theta,
                    p_star=p_star,
                    chan_regular_order=chan_orders[0],
                    chan_radial_order=chan_orders[1],
                )
            )
        except (RKEWindowCoverageError, RKEWindowConditioningError) as exc:
            row["windowed_status"] = "refused"
            row["windowed_refusal"] = f"{type(exc).__name__}: {exc}"
            row["windowed_assemble_s"] = time.perf_counter() - assemble_start
            rows.append(row)
            continue
        except (
            ValueError, RuntimeError, NotImplementedError, KeyError,
            # a per-theta assembly can still have to rebuild a channel of
            # the family, and that is an .npz cache write
            OSError,
            # sqlite3's exceptions descend from Exception, not OSError
            sqlite3.Error,
        ) as exc:
            row["windowed_status"] = "failed"
            row["windowed_refusal"] = f"{type(exc).__name__}: {exc}"
            row["windowed_assemble_s"] = time.perf_counter() - assemble_start
            rows.append(row)
            continue
        row["windowed_assemble_s"] = time.perf_counter() - assemble_start
        row["windowed_status"] = "ok"
        row["windowed_condition_number"] = float(
            certificate["condition_number"]
        )
        row["windowed_smooth_quad_order"] = int(
            certificate["smooth_quad_order"]
        )

        parameter_tag = (
            f"{parameter:.17g}".replace("-", "m").replace(".", "p")
        )
        registered_cache = cache_dir / (
            f"windowed-registered-{kernel.lower()}-{dim}d"
            f"-parameter{parameter_tag}-q{q_order}.sqlite"
        )
        try:
            loaded_table, transfer = _register_and_load_windowed_table(
                queue=queue,
                cache_path=registered_cache,
                dim=dim,
                kernel=kernel,
                q_order=q_order,
                parameter=parameter,
                source_box_level=nlevels,
                table=assembled_table,
                certificate=certificate,
            )
        except (
            ValueError, RuntimeError, NotImplementedError, OSError, KeyError,
            # sqlite3's exceptions descend from Exception, not OSError, and
            # this phase is a SQLite round trip
            sqlite3.Error,
        ) as exc:
            # Registration and the pure-cache reload belong to the same
            # taxonomy as the assembly above.  main() preserves rows only
            # for _BenchmarkGateError and otherwise writes the CSV after
            # run_benchmark() returns, so an exception escaping here
            # discards every measurement already taken, not just this row.
            row["windowed_status"] = "failed"
            row["windowed_refusal"] = f"{type(exc).__name__}: {exc}"
            # whatever the helper got through before it raised
            partial = getattr(exc, "partial_windowed_transfer", None)
            if partial is not None:
                row["windowed_register_s"] = float(partial["register_s"])
                row["windowed_register_payload_bytes"] = int(
                    partial["register_payload_bytes"]
                )
                row["windowed_table_load_s"] = float(partial["load_s"])
                row["windowed_table_load_payload_bytes"] = int(
                    partial["load_payload_bytes"]
                )
            rows.append(row)
            continue
        row["windowed_register_s"] = transfer["register_s"]
        row["windowed_register_payload_bytes"] = transfer[
            "register_payload_bytes"
        ]
        row["windowed_table_load_s"] = transfer["load_s"]
        row["windowed_table_load_payload_bytes"] = transfer[
            "load_payload_bytes"
        ]

        if far_field_status == "refused_order_cap":
            # The certificate columns above are evaluator-independent and
            # stay measured; what is refused is the solve, because the order
            # this row's wave number requires exceeds --max-fmm-order and a
            # solve at a lower order would report agreement between two
            # unresolved far fields rather than a path mismatch.
            row["windowed_refusal"] = (
                f"far-field order {row_fmm_order} exceeds the "
                f"--max-fmm-order cap {max_fmm_order}"
            )
            rows.append(row)
            continue

        if kernel == "Helmholtz":
            source_values_host, _exact = (
                _helmholtz_manufactured_source_and_exact(
                    coords_host, parameter
                )
            )
        else:
            source_values_host = _gaussian_source_host(coords_host)

        direct_table, direct_costs = _prepare_direct_tables(
            kernel=kernel,
            queue=queue,
            cache_dir=cache_dir,
            dim=dim,
            q_order=q_order,
            parameter=parameter,
            direct_levels=[nlevels],
            active_level=nlevels,
            build_config=direct_build_config,
        )
        row.update(
            {
                "level_count": 1,
                "direct_levels": str(nlevels),
                "direct_table_count": direct_costs["table_count"],
                "direct_table_payload_bytes": direct_costs["payload_bytes"],
                "direct_table_cache_payload_bytes": direct_costs[
                    "cache_payload_bytes"
                ],
                "direct_table_build_s": direct_costs["build_s"],
                "direct_table_quadrature_build_s": direct_costs[
                    "quadrature_build_s"
                ],
                "direct_table_load_s": direct_costs["load_s"],
                "direct_build_routing": direct_costs.get(
                    "build_routing", "unknown"
                ),
            }
        )

        reference_values, reference_timing, _ = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            dim=dim,
            q_order=q_order,
            fmm_order=row_fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=direct_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=False,
            split_order=1,
            repeat_count=repeat_count,
            phase_repeat_count=phase_repeat_count,
        )
        windowed_values, windowed_timing, _ = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            dim=dim,
            q_order=q_order,
            fmm_order=row_fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=loaded_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=False,
            split_order=1,
            repeat_count=repeat_count,
            phase_repeat_count=phase_repeat_count,
        )

        diff = windowed_values - reference_values
        reference_norm = max(
            float(np.linalg.norm(reference_values)), 1.0e-300
        )
        row.update(
            {
                "n_targets": int(reference_values.size),
                "rel_l2_error": float(np.linalg.norm(diff) / reference_norm),
                "linf_error": float(np.max(np.abs(diff))),
                "reference_warm_s": reference_timing["solve_mean_s"],
                "direct_table_apply_total_s": reference_timing[
                    "table_apply_total_s"
                ],
                "direct_table_apply_mean_s": reference_timing[
                    "table_apply_mean_s"
                ],
                "reference_solve_total_s": reference_timing["solve_total_s"],
                "windowed_solve_warm_s": windowed_timing["solve_mean_s"],
                "windowed_table_apply_mean_s": windowed_timing[
                    "table_apply_mean_s"
                ],
                # on a windowed_assembled row the row's strategy path is the
                # assembled table, which rides the unchanged direct warm
                # evaluator, so its split_correction phase is empty by
                # construction
                **_phase_row_columns(
                    reference_timing=reference_timing,
                    split_timing=windowed_timing,
                ),
            }
        )
        row["implied_reference_norm"] = _implied_reference_norm(
            row["linf_error"], row["rel_l2_error"]
        )
        rows.append(row)
    return rows


def _validate_windowed_rows(rows: list[dict[str, Any]]) -> None:
    """Taxonomy and agreement gates for windowed-assembled strategy rows.

    A ``failed`` row, or a certificate refusal at a theta the declaration
    covers, is a driver failure.  At small theta the windowed-assembled
    evaluator path must agree with the direct fixed-parameter reference.
    """
    for row in rows:
        if row.get("table_strategy") != "windowed_assembled":
            continue
        theta = float(row["theta"])
        window_theta = float(row["window_theta"])
        status = row["windowed_status"]
        solved = row.get("far_field_status") != "refused_order_cap"
        if status == "failed":
            raise RuntimeError(
                f"windowed assembly failed for {row['case_id']}: "
                f"{row['windowed_refusal']}"
            )
        if row["classical_probe_status"] == "failed":
            raise RuntimeError(
                f"classical certificate probe failed for {row['case_id']}: "
                f"{row['classical_probe_detail']}"
            )
        if status == "refused" and theta <= window_theta * (1.0 + 1.0e-9):
            raise RuntimeError(
                f"windowed assembly refused inside the declaration for "
                f"{row['case_id']} (theta={theta:g} <= Theta="
                f"{window_theta:g}): {row['windowed_refusal']}"
            )
        if status == "ok" and solved:
            # Finiteness is not part of the small-theta scope
            # restriction: a nan or inf mismatch is a failed solve at
            # every theta, and the tolerance below would otherwise let a
            # large-theta row into the CSV as valid evidence.
            rel_l2 = float(row["rel_l2_error"])
            if not math.isfinite(rel_l2):
                raise RuntimeError(
                    "windowed-assembled evaluator path produced a "
                    f"non-finite error for {row['case_id']}: "
                    f"rel_l2_error={rel_l2}"
                )
        if (
            status == "ok"
            and solved
            and theta <= WINDOWED_SMALL_THETA_MAX
        ):
            gate = WINDOWED_SMALL_THETA_AGREEMENT[row["mode"]]
            rel_l2 = float(row["rel_l2_error"])
            if not rel_l2 <= gate:
                raise RuntimeError(
                    "windowed-assembled evaluator path disagrees with the "
                    f"direct reference at small theta for {row['case_id']}: "
                    f"rel_l2={rel_l2:.3e} > {gate:.1e}"
                )


def _far_field_resolution_failures(
    rows: list[dict[str, Any]],
    *,
    band: tuple[float, float] = FAR_FIELD_IMPLIED_NORM_BAND,
) -> list[str]:
    """Rows whose solved Helmholtz far field is not resolved at the order used.

    The diagnostic is the implied reference-field norm
    ``linf_error / rel_l2_error``, a lower bound on the norm the relative
    column divides by.  A pinned order on a wave number that outgrows it
    drives that norm through many decades and then collapses both error
    columns to exactly zero -- a difference of two overflowed fields, not an
    agreement.  A non-finite column is the same pathology one step further
    on, and is reported on its own because no ratio is computable from it.
    Returns the failure messages rather than raising, so a long run can still
    write its CSV before the caller reports the failure.
    """
    low, high = band
    failures: list[str] = []
    for row in rows:
        if row.get("kernel") != "Helmholtz":
            continue
        if row.get("far_field_status") in ("", None, "refused_order_cap"):
            continue
        rel_l2 = row.get("rel_l2_error", "")
        linf = row.get("linf_error", "")
        if rel_l2 == "" or linf == "":
            continue
        rel_l2 = float(rel_l2)
        linf = float(linf)
        if not math.isfinite(rel_l2) or not math.isfinite(linf):
            # The other end of the same pathology: a far field that overflows
            # instead of cancelling.  ``_implied_reference_norm`` declines to
            # report a ratio here, so the band test below would never see it.
            failures.append(
                f"{row['case_id']}: non-finite Helmholtz error columns "
                f"(rel_l2={rel_l2}, linf={linf}) at "
                f"fmm_order={row['fmm_order']}"
            )
            continue
        if rel_l2 == 0.0 and linf == 0.0:
            failures.append(
                f"{row['case_id']}: both Helmholtz error columns are exactly "
                "zero, the signature of a diverged-then-zeroed far field at "
                f"fmm_order={row['fmm_order']}"
            )
            continue
        implied = _implied_reference_norm(linf, rel_l2)
        if implied == "":
            continue
        if not low <= implied <= high:
            failures.append(
                f"{row['case_id']}: implied reference-field norm "
                f"{implied:.3e} outside [{low:g}, {high:g}] at "
                f"fmm_order={row['fmm_order']}"
            )
    return failures

# }}}


def _positive_root(value: float) -> float | str:
    if math.isfinite(value) and value > 0.0:
        return float(value)
    return ""


def _amortization_accounting(
    *,
    parameter_count: int,
    level_count: int,
    repeat_count: int,
    direct_build_s: float,
    direct_load_s: float,
    rke_build_s: float,
    rke_load_s: float,
    direct_solve_total_s: float,
    split_solve_total_s: float,
) -> dict[str, Any]:
    solve_count = parameter_count * repeat_count
    direct_apply_mean_s = direct_solve_total_s / solve_count
    split_apply_mean_s = split_solve_total_s / solve_count
    direct_build_per_parameter_level_s = direct_build_s / (
        parameter_count * level_count
    )

    direct_cold_total_s = direct_build_s + direct_solve_total_s
    rke_cold_total_s = rke_build_s + split_solve_total_s
    direct_warm_total_s = direct_load_s + direct_solve_total_s
    rke_warm_total_s = rke_load_s + split_solve_total_s

    parameter_denominator = (
        level_count * direct_build_per_parameter_level_s
        + repeat_count * (direct_apply_mean_s - split_apply_mean_s)
    )
    level_root = (
        rke_build_s
        + parameter_count * repeat_count
        * (split_apply_mean_s - direct_apply_mean_s)
    ) / (parameter_count * direct_build_per_parameter_level_s)
    repeat_denominator = parameter_count * (
        direct_apply_mean_s - split_apply_mean_s
    )

    if parameter_denominator == 0.0:
        parameter_root = math.inf
    else:
        parameter_root = rke_build_s / parameter_denominator
    if repeat_denominator == 0.0:
        repeat_root = math.inf
    else:
        repeat_root = (
            rke_build_s
            - parameter_count * level_count
            * direct_build_per_parameter_level_s
        ) / repeat_denominator

    return {
        "solve_count": solve_count,
        "direct_strategy_cold_total_s": direct_cold_total_s,
        "rke_strategy_cold_total_s": rke_cold_total_s,
        "direct_strategy_warm_total_s": direct_warm_total_s,
        "rke_strategy_warm_total_s": rke_warm_total_s,
        "direct_amortized_cold_s_per_solve": direct_cold_total_s / solve_count,
        "rke_amortized_cold_s_per_solve": rke_cold_total_s / solve_count,
        "direct_amortized_warm_s_per_solve": direct_warm_total_s / solve_count,
        "rke_amortized_warm_s_per_solve": rke_warm_total_s / solve_count,
        "modeled_cold_savings_s": direct_cold_total_s - rke_cold_total_s,
        "direct_table_build_mean_s_per_parameter_level": (
            direct_build_per_parameter_level_s
        ),
        "direct_solve_mean_s_per_solve": direct_apply_mean_s,
        "split_solve_mean_s_per_solve": split_apply_mean_s,
        "amortization_time_kind": (
            "measured_table_setup_plus_warm_full_solve;"
            "linear_projection_uses_observed_mean_costs"
        ),
        "break_even_parameter_count": _positive_root(parameter_root),
        "break_even_level_count": _positive_root(level_root),
        "break_even_repeat_count": _positive_root(repeat_root),
        "break_even_time_kind": (
            "positive_root_of_cold_setup_plus_warm_full_solve_linear_model;"
            "blank_means_no_positive_finite_root"
        ),
    }


def _split_term_keys(accounting) -> str:
    return ";".join(f"{kind}:{power}" for kind, power in accounting.split_term_keys)


# Order-to-order improvement the online split path must show for Yukawa.
# 2D keeps this driver's historical three-orders-of-magnitude gate.  The 3D
# gate is one order of magnitude, the range the committed 3D field demo
# measures across its parameters (p=1 -> p=2 improves by 97x at lambda = 8,
# 355x at 4 and 1332x at 2), so a stricter gate would reject healthy runs.
YUKAWA_ORDER_IMPROVEMENT_GATE = {2: 1.0e-3, 3: 1.0e-1}


def _validate_yukawa_order_convergence(rows: list[dict[str, Any]]) -> None:
    errors_by_parameter: dict[tuple[int, float], dict[int, float]] = {}
    for row in rows:
        if row.get("table_strategy", "online_split") != "online_split":
            continue

        # Finiteness first, and for *every* online-split row: a nan or inf
        # error is a failed solve whatever the mode or kernel, and the
        # scope filters below would otherwise let a smoke row carry one
        # into the CSV with a successful exit -- while inside the scope,
        # nan makes the ratio comparisons False rather than raising.  The
        # far-field resolution report main() prints after the write is
        # Helmholtz-only, so nothing else catches a Yukawa row.  Only the
        # convergence *ratios* are scoped.
        error = float(row["rel_l2_error"])
        if not math.isfinite(error):
            raise RuntimeError(
                "online-split solve produced a non-finite error for "
                f"{row['case_id']}: rel_l2_error={error}"
            )

        if row["mode"] != "full" or row["kernel"] != "Yukawa":
            continue
        key = (int(row["dim"]), float(row["parameter_value"]))
        errors_by_parameter.setdefault(key, {})[
            int(row["split_order"])
        ] = float(row["rel_l2_error"])

    for (dim, parameter), errors in errors_by_parameter.items():
        gate = YUKAWA_ORDER_IMPROVEMENT_GATE[dim]
        if 1 in errors and 2 in errors and errors[2] > gate * errors[1]:
            raise RuntimeError(
                f"full {dim}D Yukawa RKE p=2 error did not improve by the "
                f"required factor {1.0 / gate:g} at lambda={parameter:g}: "
                f"p=1 gives {errors[1]:.3e}, p=2 gives {errors[2]:.3e}"
            )
        if 2 in errors and 3 in errors and errors[3] > 1.1 * errors[2]:
            raise RuntimeError(
                f"full {dim}D Yukawa RKE p=3 error materially degraded from "
                f"p=2 at lambda={parameter:g}: p=2 gives {errors[2]:.3e}, "
                f"p=3 gives {errors[3]:.3e}"
            )


def _phase_row_columns(*, reference_timing, split_timing):
    """Map the two paths' phase measurements onto the row's phase columns.

    The operation counts describe the shared traversal and are therefore
    taken from whichever path measured them; the seconds are per path.  A
    row whose run did not ask for phase profiling gets every column empty.
    """
    by_path = {"reference": reference_timing, "split": split_timing}
    columns = {
        "phase_profile_repeat_count": reference_timing.get(
            "phase_profile_repeat_count", PHASE_UNMEASURED
        ),
    }
    for name in ("phase_counting_rule",
                 "ops_phase_far_total",
                 "ops_phase_nearfield_point_pairs_per_solve",
                 *(f"ops_phase_far_{stage}" for stage in PHASE_FAR_STAGES)):
        value = PHASE_UNMEASURED
        for timing in (reference_timing, split_timing):
            candidate = timing.get(name, PHASE_UNMEASURED)
            if candidate != PHASE_UNMEASURED:
                value = candidate
                break
        columns[name] = value
    # the correction phase belongs to the split path alone, so unlike the
    # shared far-field counts above these are read from that path only
    for name in PHASE_CORRECTION_OPS_NAMES:
        columns[name] = split_timing.get(name, PHASE_UNMEASURED)
    for path, timing in by_path.items():
        columns[f"ops_phase_solve_total_{path}"] = timing.get(
            "ops_phase_solve_total", PHASE_UNMEASURED
        )
    for path, timing in by_path.items():
        for name in PHASE_TIMED_NAMES:
            columns[f"s_phase_{name}_{path}"] = timing.get(
                f"s_phase_{name}", PHASE_UNMEASURED
            )
    return columns


def _row_from_result(
    *,
    mode: str,
    kernel: str,
    parameter_name: str,
    parameter: float,
    split_order: int,
    power_log_beta_mode: str,
    direct_build_config,
    rke_channel_build_config,
    split_smooth_quad_order: int | None,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    reference_path: str,
    reference_values,
    split_values,
    reference_timing: dict[str, Any],
    split_timing: dict[str, Any],
    accounting,
    direct_costs: dict[str, Any],
    rke_costs: dict[str, Any],
    amortization: dict[str, Any],
    direct_levels: list[int],
    repeat_count: int,
    dim: int = 2,
    direct_build_routing: str | None = None,
) -> dict[str, Any]:
    """``direct_build_routing`` defaults to the aggregate in *direct_costs*.

    Pass this row's own routing where the caller has it: ``direct_costs``
    carries a union over every parameter, which is right for the shared
    setup-cost columns beside it and wrong for a column documented as the
    routing of this row's direct reference tables.
    """
    dim = _require_dimension(dim)
    diff = split_values - reference_values
    reference_norm = max(float(np.linalg.norm(reference_values)), 1.0e-300)
    rel_l2_error = float(np.linalg.norm(diff) / reference_norm)
    linf_error = float(np.max(np.abs(diff)))
    accounting_dict = asdict(accounting)
    return {
        "case_id": (
            f"{kernel.lower()}{dim}d-{parameter_name}"
            f"{_case_parameter_token(parameter)}"
            f"-p{split_order}"
        ),
        "mode": mode,
        "kernel": kernel,
        "dim": dim,
        "parameter_name": parameter_name,
        "parameter_value": parameter,
        "split_order": split_order,
        "power_log_beta_mode": power_log_beta_mode,
        "direct_regular_quad_order": direct_build_config.regular_quad_order,
        "direct_radial_quad_order": direct_build_config.radial_quad_order,
        "rke_channel_regular_quad_order": (
            rke_channel_build_config.regular_quad_order
        ),
        "rke_channel_radial_quad_order": (
            rke_channel_build_config.radial_quad_order
        ),
        "split_smooth_quad_order": (
            "" if split_smooth_quad_order is None else split_smooth_quad_order
        ),
        "q_order": q_order,
        "nlevels": nlevels,
        "fmm_order": fmm_order,
        "n_targets": int(reference_values.size),
        "reference_path": reference_path,
        "rel_l2_error": rel_l2_error,
        "linf_error": linf_error,
        "reference_warm_s": reference_timing["solve_mean_s"],
        "split_warm_s": split_timing["solve_mean_s"],
        "online_remainder_s": split_timing["smooth_residual_mean_s"],
        "online_remainder_time_kind": split_timing["smooth_residual_time_kind"],
        "split_term_keys": _split_term_keys(accounting),
        **_phase_row_columns(
            reference_timing=reference_timing, split_timing=split_timing
        ),
        **{
            key: value
            for key, value in accounting_dict.items()
            if key not in {"split_enabled", "split_order", "split_term_keys"}
        },
        "level_count": len(direct_levels),
        "direct_levels": ";".join(str(level) for level in direct_levels),
        "repeat_count": repeat_count,
        "direct_table_count": direct_costs["table_count"],
        "direct_table_payload_bytes": direct_costs["payload_bytes"],
        "direct_table_cache_payload_bytes": direct_costs["cache_payload_bytes"],
        "direct_table_build_s": direct_costs["build_s"],
        "direct_table_quadrature_build_s": direct_costs["quadrature_build_s"],
        "direct_table_load_s": direct_costs["load_s"],
        "direct_build_routing": (
            direct_build_routing
            if direct_build_routing is not None
            else direct_costs.get("build_routing", "unknown")
        ),
        "rke_channel_build_s": rke_costs["build_s"],
        "rke_channel_quadrature_build_s": rke_costs["quadrature_build_s"],
        "rke_channel_load_s": rke_costs["load_s"],
        "rke_channel_cache_payload_bytes": rke_costs["cache_payload_bytes"],
        "direct_table_apply_total_s": reference_timing["table_apply_total_s"],
        "direct_table_apply_mean_s": reference_timing["table_apply_mean_s"],
        "split_full_apply_total_s": split_timing["split_full_apply_total_s"],
        "split_full_apply_mean_s": split_timing["split_full_apply_mean_s"],
        "split_correction_total_s": split_timing["split_correction_total_s"],
        "split_correction_mean_s": split_timing["split_correction_mean_s"],
        "split_correction_time_kind": split_timing["split_correction_time_kind"],
        "smooth_residual_total_s": split_timing["smooth_residual_total_s"],
        "smooth_residual_mean_s": split_timing["smooth_residual_mean_s"],
        "smooth_residual_time_kind": split_timing["smooth_residual_time_kind"],
        "coefficient_eval_mean_s": split_timing["coefficient_eval_mean_s"],
        "coefficient_eval_time_kind": split_timing["coefficient_eval_time_kind"],
        "reference_solve_total_s": reference_timing["solve_total_s"],
        "split_solve_total_s": split_timing["solve_total_s"],
        **amortization,
        "table_strategy": "online_split",
        "theta": parameter * _box_extent(nlevels),
        "fmm_order_rule": "fixed",
        "fmm_order_floor": fmm_order,
        "fmm_expansion_radius": _fmm_expansion_radius(dim),
        "far_field_status": "pinned",
        "implied_reference_norm": _implied_reference_norm(
            linf_error, rel_l2_error
        ),
    }


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    split_orders: list[int],
    helmholtz_k: list[float],
    yukawa_lam: list[float],
    direct_levels: list[int],
    repeat_count: int,
    dim: int = 2,
    phase_repeat_count: int = 0,
    power_log_beta_mode: str = "p2p",
    windowed_thetas: list[float] | None = None,
    window_theta: float = DEFAULT_WINDOW_THETA,
    windowed_p_star: int = DEFAULT_WINDOWED_P_STAR,
    windowed_chan_orders: tuple[int, int] | None = None,
    classical_probe_kind: str | None = None,
    classical_probe_tolerance: float = CLASSICAL_PROBE_TOLERANCE,
    fmm_order_rule: str = "fixed",
    max_fmm_order: int | None = None,
    min_targets: int | None = None,
) -> list[dict[str, Any]]:
    import pyopencl as cl

    dim = _require_dimension(dim)
    if windowed_chan_orders is None:
        windowed_chan_orders = DEFAULT_WINDOWED_CHAN_ORDERS[dim]
    if fmm_order_rule not in {"fixed", "resolved"}:
        raise ValueError("fmm_order_rule must be 'fixed' or 'resolved'")
    if max_fmm_order is not None and max_fmm_order < fmm_order:
        raise ValueError(
            "max_fmm_order must be at least the fmm_order floor"
        )
    if fmm_order_rule == "resolved" and helmholtz_k:
        # The online-split rows of one kernel share a single cold/warm
        # amortization account, so letting their order vary per parameter
        # would mix orders inside one cost column.  The resolved rule is
        # defined for the windowed theta ladder, where every row owns its
        # own solve pair.
        raise ValueError(
            "--fmm-order-rule resolved applies to the windowed theta ladder "
            "only; run the fixed-parameter Helmholtz sweep with "
            "--fmm-order-rule fixed (or --helmholtz-k none)"
        )
    if any(not math.isfinite(k) or k <= 0.0 for k in helmholtz_k):
        # Every Helmholtz diagnostic in this sweep is defined for an
        # oscillatory far field: the resolved-order rule prescribes an order
        # from ``k a``, and :func:`_far_field_resolution_failures` reads two
        # exactly-zero error columns as a diverged-then-cancelled field.  At
        # ``k = 0`` the kernel degenerates to Laplace, so both premises fail
        # and the row would be a Laplace measurement filed under a Helmholtz
        # label.  Run Laplace as Laplace instead of as a zero-frequency
        # Helmholtz.  (Mirrors the windowed ladder, whose thetas are already
        # required to be finite and positive.)
        raise ValueError(
            "Helmholtz wave numbers must be finite and positive; the zero "
            "wave number degenerates to Laplace and is not a Helmholtz row"
        )
    if any(not math.isfinite(lam) or lam <= 0.0 for lam in yukawa_lam):
        # Same degeneracy on the screened side: the Yukawa kernel at
        # ``lambda = 0`` is Laplace.
        raise ValueError(
            "Yukawa decay rates must be finite and positive; the zero decay "
            "rate degenerates to Laplace and is not a Yukawa row"
        )
    if repeat_count < 1:
        raise ValueError("repeat_count must be >= 1")
    if phase_repeat_count < 0:
        raise ValueError("phase_repeat_count must be >= 0")
    if len(set(split_orders)) != len(split_orders):
        raise ValueError("split_orders must be unique")
    if nlevels not in direct_levels:
        raise ValueError("direct_levels must include nlevels")
    if power_log_beta_mode not in {"p2p", "table"}:
        raise ValueError("power_log_beta_mode must be 'p2p' or 'table'")
    if windowed_thetas is None:
        windowed_thetas = []
    if classical_probe_kind is None:
        classical_probe_kind = "truncation" if mode == "smoke" else "full"
    if classical_probe_kind not in {"off", "truncation", "full"}:
        raise ValueError(
            "classical_probe_kind must be 'off', 'truncation', or 'full'"
        )
    if windowed_thetas:
        if not math.isfinite(window_theta) or window_theta <= 0.0:
            raise ValueError("window_theta must be finite and positive")
        if windowed_p_star < 1:
            raise ValueError("windowed_p_star must be >= 1")

    split_auto_config = {
        "power_log_single_table_beta_mode": power_log_beta_mode,
    }

    if min_targets is not None:
        # Checked from the pure node count first, before any device is
        # selected or any geometry built, so a misconfigured dispatch fails
        # immediately rather than after the setup it cannot use.
        _require_min_targets(
            _uniform_target_count(dim, q_order, nlevels), min_targets, dim
        )

    benchmark_start = time.perf_counter()
    device = _select_opencl_device(cl, backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    q_points, source_weights, _tree, traversal = _build_geometry(
        ctx, queue, q_order, nlevels, dim=dim
    )
    n_targets = int(q_points[0].shape[0])
    if min_targets is not None:
        # The realized count, in case the mesh generator ever departs from
        # the uniform node count the pre-check used.
        _require_min_targets(n_targets, min_targets, dim)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    coords_host = _coords_host(queue, q_points)

    sweep_specs = [
        ("Helmholtz", "k", helmholtz_k),
        ("Yukawa", "lambda", yukawa_lam),
    ]

    for kernel, parameter_name, parameters in sweep_specs:
        if not parameters and not windowed_thetas:
            continue

        high_accuracy = mode == "full"
        direct_build_config = _direct_build_config(
            dim, kernel, q_order, high_accuracy=high_accuracy
        )
        rke_channel_build_config = _channel_build_config(
            dim, kernel, q_order, high_accuracy=high_accuracy
        )
        parameter_cases = []
        direct_costs = {
            "build_s": 0.0,
            "quadrature_build_s": 0.0,
            "load_s": 0.0,
            "cache_payload_bytes": 0,
            "table_count": 0,
            "payload_bytes": 0,
        }
        # routings are unioned, not summed, so they stay out of direct_costs
        # until the per-parameter loop is done
        direct_routings: set[str] = set()

        for parameter in parameters:
            if kernel == "Helmholtz":
                source_values_host, _exact_values = (
                    _helmholtz_manufactured_source_and_exact(
                        coords_host, parameter
                    )
                )
            else:
                source_values_host = _gaussian_source_host(coords_host)

            direct_table, parameter_direct_costs = _prepare_direct_tables(
                kernel=kernel,
                queue=queue,
                cache_dir=cache_dir,
                dim=dim,
                q_order=q_order,
                parameter=parameter,
                direct_levels=direct_levels,
                active_level=nlevels,
                build_config=direct_build_config,
            )
            for key in direct_costs:
                direct_costs[key] += parameter_direct_costs[key]
            direct_routings.update(
                routing
                for routing in
                str(parameter_direct_costs["build_routing"]).split(";")
                if routing
            )

            reference_values, reference_timing, _ = _run_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                dim=dim,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel=kernel,
                parameter=parameter,
                table=direct_table,
                source_weights=source_weights,
                q_points=q_points,
                source_values_host=source_values_host,
                split=False,
                split_order=1,
                repeat_count=repeat_count,
                phase_repeat_count=phase_repeat_count,
            )
            parameter_cases.append(
                {
                    "parameter": parameter,
                    "source_values_host": source_values_host,
                    "reference_values": reference_values,
                    "reference_timing": reference_timing,
                    # this parameter's own routing, not the union below:
                    # direct_build_routing is documented as the routing of
                    # *this row's* direct reference tables, and one
                    # parameter falling back must not relabel the rest
                    "direct_build_routing": str(
                        parameter_direct_costs["build_routing"]
                    ),
                }
            )

        # The union over every parameter, describing the *aggregate* setup
        # this kernel's shared cost columns account for.  Each row reports
        # its own parameter's routing instead (see parameter_cases above).
        direct_costs["build_routing"] = ";".join(sorted(direct_routings))

        direct_solve_total_s = sum(
            case["reference_timing"]["solve_total_s"] for case in parameter_cases
        )

        # Keep one cache across orders so each pass measures only newly required
        # tables; _prepare_rke_channels otherwise defaults to a per-order file.
        rke_cache_path = cache_dir / (
            f"cost-rke-{kernel.lower()}-{dim}d-q{q_order}.sqlite"
        )
        cumulative_rke_build_s = 0.0
        cumulative_rke_quadrature_build_s = 0.0
        for split_index, split_order in enumerate(
            sorted(split_orders) if parameter_cases else []
        ):
            smooth_quad_order = _smooth_quad_order(
                dim, q_order, split_order, high_accuracy=high_accuracy
            )
            representative_case = parameter_cases[0]
            split_table, split_term_tables, rke_costs = _prepare_rke_channels(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                dim=dim,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel=kernel,
                parameter=representative_case["parameter"],
                split_order=split_order,
                source_weights=source_weights,
                q_points=q_points,
                source_values_host=representative_case["source_values_host"],
                cache_dir=cache_dir,
                split_auto_config=split_auto_config,
                build_config=rke_channel_build_config,
                split_smooth_quad_order=smooth_quad_order,
                cache_path=rke_cache_path,
                clear_cache=split_index == 0,
            )
            cumulative_rke_build_s += rke_costs["build_s"]
            cumulative_rke_quadrature_build_s += rke_costs[
                "quadrature_build_s"
            ]
            rke_costs["build_s"] = cumulative_rke_build_s
            rke_costs["quadrature_build_s"] = (
                cumulative_rke_quadrature_build_s
            )

            split_results = []
            for case in parameter_cases:
                split_values, split_timing, split_wrangler = _run_path(
                    ctx=ctx,
                    queue=queue,
                    traversal=traversal,
                    dim=dim,
                    q_order=q_order,
                    fmm_order=fmm_order,
                    kernel=kernel,
                    parameter=case["parameter"],
                    table=split_table,
                    source_weights=source_weights,
                    q_points=q_points,
                    source_values_host=case["source_values_host"],
                    split=True,
                    split_order=split_order,
                    split_term_tables=split_term_tables,
                    split_auto_config=split_auto_config,
                    split_smooth_quad_order=smooth_quad_order,
                    repeat_count=repeat_count,
                    phase_repeat_count=phase_repeat_count,
                )
                accounting = split_wrangler.get_helmholtz_split_cache_accounting(
                    parameter_count=len(parameters)
                )
                split_results.append(
                    (case, split_values, split_timing, accounting)
                )

            split_solve_total_s = sum(
                split_timing["solve_total_s"]
                for _, _, split_timing, _ in split_results
            )
            amortization = _amortization_accounting(
                parameter_count=len(parameters),
                level_count=len(direct_levels),
                repeat_count=repeat_count,
                direct_build_s=direct_costs["build_s"],
                direct_load_s=direct_costs["load_s"],
                rke_build_s=rke_costs["build_s"],
                rke_load_s=rke_costs["load_s"],
                direct_solve_total_s=direct_solve_total_s,
                split_solve_total_s=split_solve_total_s,
            )

            for case, split_values, split_timing, accounting in split_results:
                rows.append(
                    _row_from_result(
                        mode=mode,
                        dim=dim,
                        kernel=kernel,
                        parameter_name=parameter_name,
                        parameter=case["parameter"],
                        split_order=split_order,
                        power_log_beta_mode=power_log_beta_mode,
                        direct_build_config=direct_build_config,
                        rke_channel_build_config=rke_channel_build_config,
                        split_smooth_quad_order=smooth_quad_order,
                        q_order=q_order,
                        nlevels=nlevels,
                        fmm_order=fmm_order,
                        reference_path="direct_fixed_parameter_table",
                        reference_values=case["reference_values"],
                        split_values=split_values,
                        reference_timing=case["reference_timing"],
                        split_timing=split_timing,
                        accounting=accounting,
                        direct_costs=direct_costs,
                        rke_costs=rke_costs,
                        amortization=amortization,
                        direct_levels=direct_levels,
                        repeat_count=repeat_count,
                        direct_build_routing=case["direct_build_routing"],
                    )
                )

        if windowed_thetas:
            rows.extend(
                _run_windowed_strategy(
                    mode=mode,
                    ctx=ctx,
                    queue=queue,
                    traversal=traversal,
                    cache_dir=cache_dir,
                    dim=dim,
                    q_order=q_order,
                    nlevels=nlevels,
                    fmm_order=fmm_order,
                    fmm_order_rule=fmm_order_rule,
                    max_fmm_order=max_fmm_order,
                    kernel=kernel,
                    parameter_name=parameter_name,
                    thetas=windowed_thetas,
                    window_theta=window_theta,
                    p_star=windowed_p_star,
                    chan_orders=windowed_chan_orders,
                    classical_probe_kind=classical_probe_kind,
                    classical_probe_tolerance=classical_probe_tolerance,
                    direct_build_config=direct_build_config,
                    repeat_count=repeat_count,
                    phase_repeat_count=phase_repeat_count,
                    source_weights=source_weights,
                    q_points=q_points,
                    coords_host=coords_host,
                )
            )

    benchmark_total_s = time.perf_counter() - benchmark_start
    for row in rows:
        row["benchmark_total_s"] = benchmark_total_s
        row["benchmark_total_time_kind"] = (
            "driver_wall_including_all_cases_setup_and_isolated_diagnostics"
        )

    # The gates run last, on complete rows, and carry those rows out on the
    # exception so ``main`` can write the CSV before reporting the failure.
    try:
        _validate_yukawa_order_convergence(rows)
        _validate_windowed_rows(rows)
    except RuntimeError as exc:
        raise _BenchmarkGateError(str(exc), rows) from exc

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--dim",
        type=int,
        choices=SUPPORTED_DIMENSIONS,
        default=2,
        help="spatial dimension (default 2, the historical Paper 1 path)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/split-parameter-sweep.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/split-parameter-cache"),
    )
    parser.add_argument("--q-order", type=int)
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--split-orders")
    parser.add_argument(
        "--power-log-beta-mode",
        choices=("p2p", "table"),
        default="p2p",
    )
    parser.add_argument("--helmholtz-k")
    parser.add_argument("--yukawa-lambda")
    parser.add_argument(
        "--parameter-count",
        type=int,
        help="use the first COUNT values from each non-empty parameter list",
    )
    level_group = parser.add_mutually_exclusive_group()
    level_group.add_argument(
        "--level-count",
        type=int,
        help="number of direct-table levels ending at --nlevels",
    )
    level_group.add_argument(
        "--direct-levels",
        help=(
            "comma-separated direct-table source-box levels; "
            "must include --nlevels"
        ),
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        help="number of timed applications per parameter",
    )
    parser.add_argument(
        "--phase-repeat-count",
        type=int,
        default=0,
        help="extra phase-instrumented solves per row for the E6 per-phase "
        "share columns (default 0, i.e. off: the ops_phase_*/s_phase_* "
        "columns stay empty and the run is otherwise unchanged).  These "
        "solves run after every timed phase and never enter a timing "
        "column; profiling synchronizes the queue at each phase boundary, "
        "so s_phase_solve_total_* exceeds the unprofiled solve mean",
    )
    parser.add_argument(
        "--windowed-thetas",
        help=(
            "comma-separated theta = parameter * leaf-table-extent values "
            "for the windowed-assembled strategy (E1); 'none' disables; "
            "default smoke '1,6,16', full "
            "'0.25,0.5,1,2,4,6,8,12,16'"
        ),
    )
    parser.add_argument(
        "--window-theta",
        type=float,
        default=DEFAULT_WINDOW_THETA,
        help="declared window Theta for the windowed-assembled strategy",
    )
    parser.add_argument(
        "--windowed-p-star",
        type=int,
        default=DEFAULT_WINDOWED_P_STAR,
        help="number of tabulated windowed channels",
    )
    parser.add_argument(
        "--windowed-chan-orders",
        default=None,
        help=(
            "'regular,radial' channel quadrature orders "
            "(default 48,61: the assembler's tested 2D orders)"
        ),
    )
    parser.add_argument(
        "--classical-probe",
        choices=("off", "truncation", "full"),
        default=None,
        help=(
            "polynomial-completion certificate probe per windowed theta; "
            "default 'truncation' (cheap) in smoke, 'full' in full mode"
        ),
    )
    parser.add_argument(
        "--classical-probe-tolerance",
        type=float,
        default=CLASSICAL_PROBE_TOLERANCE,
    )
    parser.add_argument(
        "--fmm-order-rule",
        choices=("fixed", "resolved"),
        default="fixed",
        help=(
            "'fixed' pins the FMM expansion order at --fmm-order (default, "
            "the historical behaviour); 'resolved' prescribes it per "
            "windowed row from that row's Helmholtz wave number by "
            "p(k) = max(--fmm-order, ceil(k*a + 3*ln(k*a + pi))) with "
            "a = sqrt(dim)/4, leaving Yukawa rows at the floor"
        ),
    )
    parser.add_argument(
        "--max-fmm-order",
        type=int,
        default=None,
        help=(
            "refuse (rather than solve) any row whose prescribed FMM order "
            "exceeds this cap; the row keeps its certificate columns and "
            "records far_field_status = refused_order_cap"
        ),
    )
    parser.add_argument(
        "--min-targets",
        type=int,
        default=None,
        help="fail before any build if the geometry carries fewer targets",
    )
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    mode_key = "smoke" if smoke else "full"
    dim = _require_dimension(args.dim)
    q_order = (
        args.q_order
        if args.q_order is not None
        else DEFAULT_Q_ORDER[dim][mode_key]
    )
    nlevels = (
        args.nlevels
        if args.nlevels is not None
        else DEFAULT_NLEVELS[dim][mode_key]
    )
    fmm_order = (
        args.fmm_order
        if args.fmm_order is not None
        else DEFAULT_FMM_ORDER[dim][mode_key]
    )
    split_orders = _parse_csv_ints(
        args.split_orders or ("1,2" if smoke else "1,2,3")
    )
    helmholtz_k = _parse_csv_floats(
        args.helmholtz_k
        if args.helmholtz_k is not None
        else ("4" if smoke else "4,8,12"),
        allow_empty=True,
    )
    yukawa_lam = _parse_csv_floats(
        args.yukawa_lambda
        if args.yukawa_lambda is not None
        else ("4" if smoke else "4,8,12"),
        allow_empty=True,
    )
    helmholtz_k = _limit_parameter_values(helmholtz_k, args.parameter_count)
    yukawa_lam = _limit_parameter_values(yukawa_lam, args.parameter_count)

    repeat_count = (
        args.repeat_count if args.repeat_count is not None else (1 if smoke else 5)
    )
    if repeat_count < 1:
        parser.error("--repeat-count must be >= 1")
    if args.phase_repeat_count < 0:
        parser.error("--phase-repeat-count must be >= 0")

    if args.direct_levels:
        direct_levels = _parse_csv_levels(args.direct_levels)
    else:
        level_count = (
            args.level_count
            if args.level_count is not None
            else (1 if smoke else nlevels + 1)
        )
        if not 1 <= level_count <= nlevels + 1:
            parser.error("--level-count must be between 1 and nlevels + 1")
        direct_levels = list(range(nlevels - level_count + 1, nlevels + 1))
    if nlevels not in direct_levels:
        parser.error(
            "--direct-levels must include --nlevels for the direct reference"
        )
    try:
        windowed_thetas = _parse_windowed_thetas(args.windowed_thetas, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.windowed_chan_orders is None:
        windowed_chan_orders = DEFAULT_WINDOWED_CHAN_ORDERS[dim]
    else:
        parts = _parse_csv_ints(args.windowed_chan_orders)
        if len(parts) != 2:
            parser.error(
                "--windowed-chan-orders must be a 'regular,radial' pair"
            )
        windowed_chan_orders = (parts[0], parts[1])

    if not helmholtz_k and not yukawa_lam and not windowed_thetas:
        parser.error(
            "at least one of --helmholtz-k, --yukawa-lambda, or "
            "--windowed-thetas must be non-empty"
        )
    if args.max_fmm_order is not None and args.max_fmm_order < fmm_order:
        parser.error("--max-fmm-order must be at least --fmm-order")

    try:
        rows = run_benchmark(
            mode=args.mode,
            backend=args.backend,
            cache_dir=args.cache_dir,
            dim=dim,
            q_order=q_order,
            nlevels=nlevels,
            fmm_order=fmm_order,
            split_orders=split_orders,
            helmholtz_k=helmholtz_k,
            yukawa_lam=yukawa_lam,
            direct_levels=direct_levels,
            repeat_count=repeat_count,
            phase_repeat_count=args.phase_repeat_count,
            power_log_beta_mode=args.power_log_beta_mode,
            windowed_thetas=windowed_thetas,
            window_theta=args.window_theta,
            windowed_p_star=args.windowed_p_star,
            windowed_chan_orders=windowed_chan_orders,
            classical_probe_kind=args.classical_probe,
            classical_probe_tolerance=args.classical_probe_tolerance,
            fmm_order_rule=args.fmm_order_rule,
            max_fmm_order=args.max_fmm_order,
            min_targets=args.min_targets,
        )
    except _BenchmarkGateError as exc:
        # The rows are measured; only the verdict on them failed.  Write them
        # before re-raising, so a multi-hour run keeps its CSV.
        write_csv(args.out, exc.rows)
        print(f"GATE-FAILED (CSV written to {args.out}): {exc}")
        raise
    # Write first, then report the far-field resolution check, so a long run
    # never loses its measurements to a failing diagnostic.
    write_csv(args.out, rows)
    failures = _far_field_resolution_failures(rows)
    if failures:
        for message in failures:
            print(f"FAR-FIELD-UNRESOLVED: {message}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
