#!/usr/bin/env python3
"""Table-level certificate sweep for windowed RKE near-field assembly.

For each swept fixed parameter the driver produces the same near-field table
three ways and records certificates, wall times, payload accounting, and
cross-deviations:

(a) windowed RKE assembly (:func:`assemble_windowed_parameterized_table`),
    swept over ``p_star`` and the smooth-remainder quadrature order;
(b) classical series RKE assembly (:func:`assemble_parameterized_table`) at a
    matching certified tolerance, recording the refusal kind when the
    classical certificate cannot be issued (uncertifiable truncation or
    ill-conditioned recombination).  A separately timed warm-up call prepares
    any missing canonical channels.  ``classical_warmup_seconds`` records that
    complete first attempt, including cache loading, possible channel builds,
    and recombination; ``classical_assemble_seconds`` records a second
    warm-cache end-to-end assembly, including channel loading;
(c) direct fixed-parameter builds through the table manager at two Duffy
    quadrature policies, whose mutual disagreement estimates the reference
    floor below which assembled-vs-direct deviations are quadrature noise.

Deviations are reported against the tight direct policy as both relative
max-entry and relative L2 over the symmetry-reduced entries.  Windowed
channel families are built (or reused) once per ``(dim, q_order, level,
root_extent, window_theta, channel quadrature orders)`` from a per-family
cache, so the
one-off channel build cost and the marginal per-parameter assembly cost are
recorded separately.  The windowed channel quadrature orders are themselves
sweepable via ``--chan-orders``; the classical and direct reference builds do
not depend on them and are computed once per ``(dim, kernel, parameter)``.

Smoke mode is intended for CI/local validation.  Full mode is intended for
metadata-wrapped runs on a controlled remote compute host.
"""

from __future__ import annotations

import argparse
import csv
import json
import operator
import time
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = (
    "case_id",
    "mode",
    "dim",
    "kernel",
    "parameter_name",
    "parameter_value",
    "theta",
    "window_theta",
    "q_order",
    "source_box_level",
    "root_extent",
    "box_extent",
    "n_reduced_entries",
    "p_star",
    "smooth_quad_order_requested",
    "smooth_quad_order_used",
    "chan_regular_order",
    "chan_radial_order",
    "channel_build_was_cold",
    "channel_build_seconds",
    "windowed_status",
    "windowed_refusal",
    "windowed_assemble_seconds",
    "windowed_condition_number",
    "windowed_coefficient_bound",
    "windowed_remainder_peak",
    "windowed_truncation_tail_bound",
    "windowed_payload_bytes",
    "classical_status",
    "classical_refusal",
    "classical_refusal_detail",
    "classical_tolerance",
    "classical_n_series_terms",
    "classical_channel_count",
    "classical_condition_number",
    "classical_warmup_seconds",
    "classical_assemble_seconds",
    "classical_payload_bytes",
    "classical_channel_regular_order",
    "classical_channel_radial_order",
    "direct_loose_regular_order",
    "direct_loose_radial_order",
    "direct_loose_status",
    "direct_loose_build_seconds",
    "direct_tight_regular_order",
    "direct_tight_radial_order",
    "direct_tight_status",
    "direct_tight_build_seconds",
    "direct_policy_rel_max_entry_floor",
    "direct_reference_policy",
    "windowed_vs_direct_rel_max_entry",
    "windowed_vs_direct_rel_l2",
    "classical_vs_direct_rel_max_entry",
    "classical_vs_direct_rel_l2",
    "benchmark_total_seconds",
)

PARAMETER_NAMES = {"Helmholtz": "k", "Yukawa": "lambda"}
DEFAULT_Q_ORDER = {2: 3, 3: 2}
DEFAULT_SOURCE_LEVEL = {2: 3, 3: 2}
CLASSICAL_TOLERANCE = 1.0e-11

# What the windowed assembler covers is the dimensionless local parameter
# theta = mu * b, not mu itself, and the default source-box extent b differs
# per dimension (2D: 0.25, 3D: 0.5).  Declaring the default ladders as
# fractions of the declared window Theta and converting with mu = theta / b
# therefore keeps every default row inside the coverage disk in both
# dimensions -- a fixed mu ladder reaching theta = Theta in 2D overshoots to
# theta = 2 Theta in 3D and is refused outright.  At the default geometry
# (Theta = 16, root extent 2) these reproduce the historical fixed ladders
# exactly: smoke '4,64' and full '1,2,4,8,16,24,32,48,64' in 2D.
DEFAULT_SMOKE_THETA_FRACTIONS = (1.0 / 16.0, 1.0)
DEFAULT_FULL_THETA_FRACTIONS = (
    1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0,
    3.0 / 8.0, 1.0 / 2.0, 3.0 / 4.0, 1.0,
)


def _default_mus(mode: str, window_theta: float, box_extent: float):
    """Default parameter ladder for one dimension's box extent."""
    fractions = (
        DEFAULT_SMOKE_THETA_FRACTIONS
        if mode == "smoke"
        else DEFAULT_FULL_THETA_FRACTIONS
    )
    return [
        fraction * float(window_theta) / float(box_extent)
        for fraction in fractions
    ]


def _exact_float_token(value: float) -> str:
    """Filename-safe token preserving the exact binary64 value."""
    return float(value).hex()


def _parameter_identity_token(value: float) -> str:
    """Readable parameter prefix plus an exact collision-free identity."""
    return f"{float(value):g}-{_exact_float_token(value)}"


def _classical_cache_path(
    cache_dir: Path,
    dim: int,
    q_order: int,
    root_extent: float,
    channel_orders: tuple[int, int],
) -> Path:
    """Classical cache identity including file-wide geometry and policy."""
    root_token = _exact_float_token(root_extent)
    regular_order, radial_order = channel_orders
    return cache_dir / (
        f"classical-channels-d{int(dim)}-q{int(q_order)}"
        f"-r{root_token}-c{int(regular_order)}x{int(radial_order)}.sqlite"
    )


def _require_finite_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _require_integer(value, name: str, *, minimum=None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    result = int(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _require_usable_order_pair(
    pair: tuple[int, int], name: str
) -> tuple[int, int]:
    regular_order = _require_integer(pair[0], f"{name} regular order")
    radial_order = _require_integer(pair[1], f"{name} radial order")
    if regular_order < 2:
        raise ValueError(f"{name} regular order must be >= 2")
    if radial_order < 7:
        raise ValueError(f"{name} radial order must be >= 7")
    return regular_order, radial_order


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _parse_direct_policies(raw: str) -> list[tuple[int, int]]:
    """Parse ``'regular,radial;regular,radial'`` into [(loose), (tight)]."""
    policies = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(part.strip()) for part in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "each direct policy must be a 'regular,radial' pair"
            )
        policies.append((parts[0], parts[1]))
    if len(policies) != 2:
        raise ValueError(
            "exactly two direct policies (loose;tight) are required"
        )
    policies = [
        _require_usable_order_pair(policy, "direct policy")
        for policy in policies
    ]
    loose, tight = policies
    if any(
        tight_order < loose_order
        for loose_order, tight_order in zip(loose, tight, strict=True)
    ):
        raise ValueError(
            "tight direct policy must be componentwise >= the loose policy"
        )
    if tight == loose:
        raise ValueError(
            "tight direct policy must be strictly larger in at least one "
            "component"
        )
    return policies


def _parse_order_pairs(raw: str) -> list[tuple[int, int]]:
    """Parse ``'regular,radial;regular,radial;...'`` into a list of pairs."""
    pairs: list[tuple[int, int]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(part.strip()) for part in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "each channel-order policy must be a 'regular,radial' pair"
            )
        pairs.append(
            _require_usable_order_pair(
                (parts[0], parts[1]), "channel policy"
            )
        )
    if not pairs:
        raise ValueError("expected at least one 'regular,radial' pair")
    return pairs


def _parse_order_pair(raw: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("expected a 'regular,radial' integer pair")
    return _require_usable_order_pair(
        (parts[0], parts[1]), "classical channel policy"
    )


def _clear_sqlite_cache(path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _make_queue():
    import pyopencl as cl

    ctx = cl.create_some_context(interactive=False)
    return cl.CommandQueue(ctx)


def _relative_deviations(values, reference):
    """(rel max-entry, rel L2) of ``values`` against ``reference``."""
    values = np.asarray(values)
    reference = np.asarray(reference)
    diff = values.astype(np.complex128) - reference.astype(np.complex128)
    ref_max = max(float(np.max(np.abs(reference))), 1.0e-300)
    ref_l2 = max(float(np.linalg.norm(reference)), 1.0e-300)
    return (
        float(np.max(np.abs(diff)) / ref_max),
        float(np.linalg.norm(diff) / ref_l2),
    )


def _prepare_windowed_channels(
    *,
    cache_path: Path,
    dim: int,
    q_order: int,
    source_box_level: int,
    root_extent: float,
    window_theta: float,
    max_p_star: int,
    chan_regular_order: int,
    chan_radial_order: int,
) -> dict[str, Any]:
    """Build (or reload) the windowed channel family once and time it."""
    from volumential.rke_table_assembly import get_windowed_channel_table

    was_cold = False
    start = time.perf_counter()
    base_channel = None
    for m in range(max_p_star):
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
        disposition = getattr(
            channel,
            "_windowed_cache_disposition",
            getattr(channel, "_cache_disposition", None),
        )
        if disposition not in ("hit", "rebuilt"):
            raise RuntimeError(
                "windowed channel table did not report a valid private "
                "cache disposition"
            )
        was_cold = was_cold or disposition == "rebuilt"
        if m == 0:
            base_channel = channel
    build_seconds = time.perf_counter() - start

    entry_ids = np.asarray(base_channel.get_reduced_entry_ids(), dtype=np.int64)
    return {
        "entry_ids": entry_ids,
        "n_reduced_entries": int(entry_ids.size),
        "channel_build_seconds": build_seconds,
        "channel_build_was_cold": was_cold,
    }


def _run_windowed(
    *,
    cache_path: Path,
    dim: int,
    kernel: str,
    q_order: int,
    parameter: float,
    source_box_level: int,
    root_extent: float,
    window_theta: float,
    p_star: int,
    smooth_quad_order: int,
    chan_regular_order: int,
    chan_radial_order: int,
    entry_ids,
    n_reduced_entries: int,
) -> dict[str, Any]:
    from volumential.rke_table_assembly import (
        RKEWindowConditioningError,
        RKEWindowCoverageError,
        assemble_windowed_parameterized_table,
    )

    start = time.perf_counter()
    try:
        table, certificate = assemble_windowed_parameterized_table(
            cache_path,
            dim,
            kernel,
            q_order,
            parameter,
            source_box_level=source_box_level,
            root_extent=root_extent,
            window_theta=window_theta,
            p_star=p_star,
            smooth_quad_order=smooth_quad_order,
            chan_regular_order=chan_regular_order,
            chan_radial_order=chan_radial_order,
        )
    except (RKEWindowCoverageError, RKEWindowConditioningError) as exc:
        return {
            "windowed_status": "refused",
            "windowed_refusal": f"{type(exc).__name__}: {exc}",
            "windowed_assemble_seconds": time.perf_counter() - start,
            "values": None,
        }
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        return {
            "windowed_status": "failed",
            "windowed_refusal": f"{type(exc).__name__}: {exc}",
            "windowed_assemble_seconds": time.perf_counter() - start,
            "values": None,
        }
    assemble_seconds = time.perf_counter() - start
    values = np.asarray(table.get_entry_data_for_full_indices(entry_ids))
    return {
        "windowed_status": "ok",
        "windowed_refusal": "",
        "windowed_assemble_seconds": assemble_seconds,
        "windowed_condition_number": certificate["condition_number"],
        "windowed_coefficient_bound": certificate["coefficient_bound"],
        "windowed_remainder_peak": certificate["remainder_peak"],
        "windowed_truncation_tail_bound": (
            certificate["truncation_tail_bound"]
        ),
        "smooth_quad_order_used": certificate["smooth_quad_order"],
        # p_star real float64 channel tables over the reduced entries
        "windowed_payload_bytes": int(n_reduced_entries * 8 * p_star),
        "values": values,
    }


def _classify_classical_refusal(exc: BaseException) -> str:
    """Refusal kind of a classical assembly failure.

    The assembler names its two certified refusal modes structurally, via the
    ``refusal_kind`` attribute of :class:`~volumential.rke_table_assembly.\
RKETruncationError` / :class:`~volumential.rke_table_assembly.\
RKEConditioningError`, so a message rewording can no longer silently
    reclassify a row.  Message matching survives only as a fallback for
    exceptions raised without that marker, and keys on the stable part of
    each message ("certify", "ill-conditioned") rather than the full text.
    """
    kind = getattr(exc, "refusal_kind", None)
    if isinstance(kind, str) and kind:
        return kind
    message = str(exc).lower()
    if "certify" in message:
        return "uncertifiable"
    if "ill-conditioned" in message:
        return "ill-conditioned"
    return type(exc).__name__


def _run_classical(
    *,
    queue,
    cache_path: Path,
    dim: int,
    kernel: str,
    q_order: int,
    parameter: float,
    source_box_level: int,
    root_extent: float,
    channel_orders: tuple[int, int],
    entry_ids,
    n_reduced_entries: int,
) -> dict[str, Any]:
    """Classical assembly with a separate warm-up and warm-cache timing.

    The first call may load or build canonical channels and performs a full
    recombination, so its elapsed time is reported only as
    ``classical_warmup_seconds``.  The second call measures the repeatable
    warm-cache path, including channel deserialization and recombination.
    """
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.rke_table_assembly import (
        RKEConditioningError,
        RKETruncationError,
        assemble_parameterized_table,
    )

    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=channel_orders[0],
        radial_quad_order=channel_orders[1],
    )

    def assemble():
        return assemble_parameterized_table(
            queue,
            cache_path,
            dim,
            kernel,
            q_order,
            parameter,
            source_box_level=source_box_level,
            root_extent=root_extent,
            tolerance=CLASSICAL_TOLERANCE,
            build_config=build_config,
        )

    start = time.perf_counter()
    try:
        assemble()
    except (RKETruncationError, RKEConditioningError) as exc:
        return {
            "classical_status": "refused",
            "classical_refusal": exc.refusal_kind,
            "classical_refusal_detail": f"{type(exc).__name__}: {exc}",
            "classical_warmup_seconds": time.perf_counter() - start,
            "classical_assemble_seconds": "",
            "values": None,
        }
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        return {
            "classical_status": "failed",
            "classical_refusal": "",
            "classical_refusal_detail": f"{type(exc).__name__}: {exc}",
            "classical_warmup_seconds": time.perf_counter() - start,
            "classical_assemble_seconds": "",
            "values": None,
        }
    warmup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    try:
        table, certificate = assemble()
    except (RKETruncationError, RKEConditioningError) as exc:
        return {
            "classical_status": "refused",
            "classical_refusal": exc.refusal_kind,
            "classical_refusal_detail": f"{type(exc).__name__}: {exc}",
            "classical_warmup_seconds": warmup_seconds,
            "classical_assemble_seconds": "",
            "values": None,
        }
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        return {
            "classical_status": "failed",
            "classical_refusal": "",
            "classical_refusal_detail": f"{type(exc).__name__}: {exc}",
            "classical_warmup_seconds": warmup_seconds,
            "classical_assemble_seconds": "",
            "values": None,
        }
    assemble_seconds = time.perf_counter() - start
    values = np.asarray(table.get_entry_data_for_full_indices(entry_ids))
    channel_count = int(certificate["channel_count"])
    return {
        "classical_status": "ok",
        "classical_refusal": "",
        "classical_refusal_detail": "",
        "classical_n_series_terms": certificate["n_series_terms"],
        "classical_channel_count": channel_count,
        "classical_condition_number": certificate["condition_number"],
        "classical_warmup_seconds": warmup_seconds,
        "classical_assemble_seconds": assemble_seconds,
        # channel_count real float64 channel tables over the reduced entries
        "classical_payload_bytes": int(
            n_reduced_entries * 8 * channel_count
        ),
        "values": values,
    }


def _build_direct_table(
    *,
    queue,
    cache_path: Path,
    dim: int,
    kernel: str,
    q_order: int,
    parameter: float,
    source_box_level: int,
    root_extent: float,
    regular_order: int,
    radial_order: int,
    entry_ids,
) -> dict[str, Any]:
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_order,
        radial_quad_order=radial_order,
    )
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

    # A cold build per call keeps the recorded seconds an honest build cost
    # and prevents force_recompute from resurrecting a cached build config.
    _clear_sqlite_cache(cache_path)
    start = time.perf_counter()
    try:
        with NearFieldInteractionTableManager(
            str(cache_path),
            root_extent=float(root_extent),
            queue=queue,
            **manager_kwargs,
        ) as table_manager:
            table, _ = table_manager.get_table(
                dim,
                kernel_request,
                q_order,
                source_box_level=int(source_box_level),
                force_recompute=True,
                queue=queue,
                build_config=build_config,
                **get_kwargs,
            )
    except Exception as exc:
        return {
            "status": f"failed: {type(exc).__name__}: {exc}",
            "build_seconds": time.perf_counter() - start,
            "values": None,
        }
    build_seconds = time.perf_counter() - start
    values = np.asarray(table.get_entry_data_for_full_indices(entry_ids))
    return {"status": "ok", "build_seconds": build_seconds, "values": values}


def run_sweep(
    *,
    mode: str,
    dims: list[int],
    kernels: list[str],
    q_order_override: int | None,
    source_level_override: int | None,
    root_extent: float,
    window_theta: float,
    p_stars: list[int],
    smooth_orders: list[int],
    mus: list[float] | None,
    direct_policies: list[tuple[int, int]],
    classical_channel_orders: tuple[int, int],
    chan_orders: list[tuple[int, int]] | None,
    cache_dir: Path,
    skip_3d_tight: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be 'smoke' or 'full'")
    _require_finite_positive(root_extent, "root_extent")
    _require_finite_positive(window_theta, "window_theta")
    dims = [_require_integer(dim, "dim") for dim in dims]
    if not dims:
        raise ValueError("at least one dimension is required")
    if any(dim not in (2, 3) for dim in dims):
        raise ValueError("dim entries must be 2 or 3")
    kernels = list(kernels)
    if not kernels:
        raise ValueError("at least one kernel is required")
    unknown_kernels = [
        kernel for kernel in kernels if kernel not in PARAMETER_NAMES
    ]
    if unknown_kernels:
        raise ValueError(f"unknown kernel: {unknown_kernels[0]}")
    if q_order_override is not None:
        q_order_override = _require_integer(
            q_order_override, "q_order_override", minimum=1
        )
    if source_level_override is not None:
        source_level_override = _require_integer(
            source_level_override, "source_level_override", minimum=0
        )
    p_stars = [
        _require_integer(p_star, "p_star", minimum=1) for p_star in p_stars
    ]
    if not p_stars:
        raise ValueError("at least one p_star is required")
    smooth_orders = [
        _require_integer(order, "smooth_order", minimum=1)
        for order in smooth_orders
    ]
    if not smooth_orders:
        raise ValueError("at least one smooth_order is required")
    if mus is not None:
        mus = list(mus)
        if not mus:
            raise ValueError("at least one mu is required when mus is provided")
        for mu in mus:
            _require_finite_positive(mu, "mu")
    direct_policies = [
        _require_usable_order_pair(policy, "direct policy")
        for policy in direct_policies
    ]
    if len(direct_policies) != 2:
        raise ValueError(
            "exactly two direct policies (loose;tight) are required"
        )
    loose_policy, tight_policy = direct_policies
    if any(
        tight_order < loose_order
        for loose_order, tight_order in zip(
            loose_policy, tight_policy, strict=True
        )
    ) or tight_policy == loose_policy:
        raise ValueError(
            "tight direct policy must be componentwise >= the loose policy "
            "and strictly larger in at least one component"
        )
    classical_channel_orders = _require_usable_order_pair(
        classical_channel_orders, "classical channel policy"
    )
    if chan_orders is not None:
        chan_orders = [
            _require_usable_order_pair(policy, "channel policy")
            for policy in chan_orders
        ]
        if not chan_orders:
            raise ValueError(
                "at least one channel policy is required when chan_orders "
                "is provided"
            )

    dimension_configs: dict[int, tuple[int, int, float, list[float]]] = {}
    mus_by_dim: dict[int, list[float]] = {}
    for dim in sorted(dims):
        q_order = (
            q_order_override
            if q_order_override is not None
            else DEFAULT_Q_ORDER[dim]
        )
        source_level = (
            source_level_override
            if source_level_override is not None
            else DEFAULT_SOURCE_LEVEL[dim]
        )
        box_extent = float(root_extent) * 0.5**source_level
        _require_finite_positive(box_extent, "box_extent")
        dim_mus = sorted(
            mus
            if mus is not None
            else _default_mus(mode, window_theta, box_extent)
        )
        for mu in dim_mus:
            _require_finite_positive(mu, "mu")
        dimension_configs[dim] = (
            q_order, source_level, box_extent, dim_mus
        )
        mus_by_dim[dim] = dim_mus

    from volumential.rke_table_assembly import _resolve_channel_orders

    sweep_start = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    queue = _make_queue()

    rows: list[dict[str, Any]] = []
    channel_prep_records: dict[str, Any] = {}

    for dim in sorted(dims):
        q_order, source_level, box_extent, dim_mus = dimension_configs[dim]
        if chan_orders is None:
            dim_chan_orders = [_resolve_channel_orders(dim, None, None)]
        else:
            dim_chan_orders = [
                _resolve_channel_orders(dim, regular, radial)
                for regular, radial in chan_orders
            ]

        classical_cache = _classical_cache_path(
            cache_dir,
            dim,
            q_order,
            root_extent,
            classical_channel_orders,
        )

        print(
            f"[config] dim={dim} q_order={q_order} level={source_level} "
            f"box_extent={box_extent:g} window_theta={window_theta:g} "
            "chan_orders="
            + ";".join(
                f"{regular}/{radial}" for regular, radial in dim_chan_orders
            )
            + " mus="
            + ",".join(f"{mu:g}" for mu in dim_mus),
            flush=True,
        )

        # The windowed channel families depend only on (dim, q_order, level,
        # root_extent, window_theta, chan orders) -- not on the kernel or the
        # parameter -- so build each requested channel-order family once here.
        channel_families: list[dict[str, Any]] = []
        for chan_regular_order, chan_radial_order in dim_chan_orders:
            windowed_cache = cache_dir / (
                f"windowed-channels-d{dim}-q{q_order}"
                f"-c{chan_regular_order}x{chan_radial_order}.db"
            )
            channels = _prepare_windowed_channels(
                cache_path=windowed_cache,
                dim=dim,
                q_order=q_order,
                source_box_level=source_level,
                root_extent=root_extent,
                window_theta=window_theta,
                max_p_star=max(p_stars),
                chan_regular_order=chan_regular_order,
                chan_radial_order=chan_radial_order,
            )
            print(
                f"[channels] dim={dim} "
                f"chan_orders={chan_regular_order}/{chan_radial_order} "
                f"cold={channels['channel_build_was_cold']} "
                f"build_s={channels['channel_build_seconds']:.2f} "
                f"n_reduced_entries={channels['n_reduced_entries']}",
                flush=True,
            )
            channel_prep_records[
                f"dim{dim}_c{chan_regular_order}x{chan_radial_order}"
            ] = {
                "q_order": q_order,
                "source_box_level": source_level,
                "chan_regular_order": chan_regular_order,
                "chan_radial_order": chan_radial_order,
                "channel_build_seconds": channels["channel_build_seconds"],
                "channel_build_was_cold": channels["channel_build_was_cold"],
                "n_reduced_entries": channels["n_reduced_entries"],
            }
            channels["cache_path"] = windowed_cache
            channels["chan_regular_order"] = chan_regular_order
            channels["chan_radial_order"] = chan_radial_order
            channel_families.append(channels)

        # The reduced-entry index set is a property of (dim, q_order, level)
        # alone, so all families must agree; the direct and classical
        # references are then sampled once on that shared index set.
        entry_ids = channel_families[0]["entry_ids"]
        n_entries = channel_families[0]["n_reduced_entries"]
        for family in channel_families[1:]:
            if not np.array_equal(family["entry_ids"], entry_ids):
                raise RuntimeError(
                    "windowed channel families disagree on the reduced entry "
                    "index set across channel-order pairs"
                )

        for kernel in kernels:
            for mu in dim_mus:
                theta = float(mu) * box_extent
                mu_tag = _parameter_identity_token(mu)

                policy_results = []
                for policy_index, (regular, radial) in enumerate(
                    direct_policies
                ):
                    policy_name = "loose" if policy_index == 0 else "tight"
                    if policy_name == "tight" and dim == 3 and skip_3d_tight:
                        policy_results.append(
                            {
                                "status": "skipped: --skip-3d-tight",
                                "build_seconds": "",
                                "values": None,
                            }
                        )
                        continue
                    direct_cache = cache_dir / (
                        f"direct-d{dim}-{kernel.lower()}-mu{mu_tag}"
                        f"-{policy_name}.sqlite"
                    )
                    policy_results.append(
                        _build_direct_table(
                            queue=queue,
                            cache_path=direct_cache,
                            dim=dim,
                            kernel=kernel,
                            q_order=q_order,
                            parameter=mu,
                            source_box_level=source_level,
                            root_extent=root_extent,
                            regular_order=regular,
                            radial_order=radial,
                            entry_ids=entry_ids,
                        )
                    )
                loose, tight = policy_results

                if (
                    loose["values"] is not None
                    and tight["values"] is not None
                ):
                    floor_rel_max, _ = _relative_deviations(
                        loose["values"], tight["values"]
                    )
                else:
                    floor_rel_max = ""
                if tight["values"] is not None:
                    reference_policy = "tight"
                    reference_values = tight["values"]
                elif loose["values"] is not None:
                    reference_policy = "loose"
                    reference_values = loose["values"]
                else:
                    reference_policy = ""
                    reference_values = None

                classical = _run_classical(
                    queue=queue,
                    cache_path=classical_cache,
                    dim=dim,
                    kernel=kernel,
                    q_order=q_order,
                    parameter=mu,
                    source_box_level=source_level,
                    root_extent=root_extent,
                    channel_orders=classical_channel_orders,
                    entry_ids=entry_ids,
                    n_reduced_entries=n_entries,
                )
                if (
                    classical["values"] is not None
                    and reference_values is not None
                ):
                    classical_rel_max, classical_rel_l2 = (
                        _relative_deviations(
                            classical["values"], reference_values
                        )
                    )
                else:
                    classical_rel_max, classical_rel_l2 = "", ""

                print(
                    f"[case] dim={dim} kernel={kernel} mu={mu:g} "
                    f"theta={theta:g} "
                    f"direct_loose={loose['status']} "
                    f"direct_tight={tight['status']} "
                    f"classical={classical['classical_status']}"
                    + (
                        f" ({classical['classical_refusal']})"
                        if classical["classical_status"] == "refused"
                        else ""
                    ),
                    flush=True,
                )

                for family in channel_families:
                    windowed_cache = family["cache_path"]
                    chan_regular_order = family["chan_regular_order"]
                    chan_radial_order = family["chan_radial_order"]
                    chan_tag = f"c{chan_regular_order}x{chan_radial_order}"

                    for p_star in sorted(p_stars):
                        for smooth_order in sorted(smooth_orders):
                            windowed = _run_windowed(
                                cache_path=windowed_cache,
                                dim=dim,
                                kernel=kernel,
                                q_order=q_order,
                                parameter=mu,
                                source_box_level=source_level,
                                root_extent=root_extent,
                                window_theta=window_theta,
                                p_star=p_star,
                                smooth_quad_order=smooth_order,
                                chan_regular_order=chan_regular_order,
                                chan_radial_order=chan_radial_order,
                                entry_ids=entry_ids,
                                n_reduced_entries=n_entries,
                            )
                            if (
                                windowed["values"] is not None
                                and reference_values is not None
                            ):
                                windowed_rel_max, windowed_rel_l2 = (
                                    _relative_deviations(
                                        windowed["values"], reference_values
                                    )
                                )
                            else:
                                windowed_rel_max, windowed_rel_l2 = "", ""

                            print(
                                f"  [row] chan={chan_regular_order}/"
                                f"{chan_radial_order} "
                                f"p_star={p_star} "
                                f"smooth={smooth_order} "
                                f"windowed={windowed['windowed_status']} "
                                f"assemble_s="
                                f"{windowed['windowed_assemble_seconds']:.2f}"
                                + (
                                    f" rel_max={windowed_rel_max:.3e}"
                                    if windowed_rel_max != ""
                                    else ""
                                ),
                                flush=True,
                            )

                            row = {key: "" for key in FIELDS}
                            row.update(
                                {
                                    "case_id": (
                                        f"{kernel.lower()}{dim}d-mu{mu_tag}"
                                        f"-{chan_tag}"
                                        f"-p{p_star}-s{smooth_order}"
                                    ),
                                    "mode": mode,
                                    "dim": dim,
                                    "kernel": kernel,
                                    "parameter_name": PARAMETER_NAMES[kernel],
                                    "parameter_value": mu,
                                    "theta": theta,
                                    "window_theta": window_theta,
                                    "q_order": q_order,
                                    "source_box_level": source_level,
                                    "root_extent": root_extent,
                                    "box_extent": box_extent,
                                    "n_reduced_entries": n_entries,
                                    "p_star": p_star,
                                    "smooth_quad_order_requested": (
                                        smooth_order
                                    ),
                                    "chan_regular_order": chan_regular_order,
                                    "chan_radial_order": chan_radial_order,
                                    "channel_build_was_cold": family[
                                        "channel_build_was_cold"
                                    ],
                                    "channel_build_seconds": family[
                                        "channel_build_seconds"
                                    ],
                                    "classical_tolerance": CLASSICAL_TOLERANCE,
                                    "classical_channel_regular_order": (
                                        classical_channel_orders[0]
                                    ),
                                    "classical_channel_radial_order": (
                                        classical_channel_orders[1]
                                    ),
                                    "direct_loose_regular_order": (
                                        direct_policies[0][0]
                                    ),
                                    "direct_loose_radial_order": (
                                        direct_policies[0][1]
                                    ),
                                    "direct_loose_status": loose["status"],
                                    "direct_loose_build_seconds": loose[
                                        "build_seconds"
                                    ],
                                    "direct_tight_regular_order": (
                                        direct_policies[1][0]
                                    ),
                                    "direct_tight_radial_order": (
                                        direct_policies[1][1]
                                    ),
                                    "direct_tight_status": tight["status"],
                                    "direct_tight_build_seconds": tight[
                                        "build_seconds"
                                    ],
                                    "direct_policy_rel_max_entry_floor": (
                                        floor_rel_max
                                    ),
                                    "direct_reference_policy": (
                                        reference_policy
                                    ),
                                    "windowed_vs_direct_rel_max_entry": (
                                        windowed_rel_max
                                    ),
                                    "windowed_vs_direct_rel_l2": (
                                        windowed_rel_l2
                                    ),
                                    "classical_vs_direct_rel_max_entry": (
                                        classical_rel_max
                                    ),
                                    "classical_vs_direct_rel_l2": (
                                        classical_rel_l2
                                    ),
                                }
                            )
                            for key, value in windowed.items():
                                if key != "values" and key in FIELDS:
                                    row[key] = value
                            for key, value in classical.items():
                                if key != "values" and key in FIELDS:
                                    row[key] = value
                            rows.append(row)

    total_seconds = time.perf_counter() - sweep_start
    for row in rows:
        row["benchmark_total_seconds"] = total_seconds
    return rows, {
        "channel_prep": channel_prep_records,
        "mus_by_dim": {str(dim): values for dim, values in mus_by_dim.items()},
        "total_seconds": total_seconds,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--dim",
        help="comma-separated dimensions (2 and/or 3); "
        "defaults: smoke '2', full '2,3'",
    )
    parser.add_argument(
        "--kernels",
        default="helmholtz,yukawa",
        help="comma-separated kernels from {helmholtz, yukawa}",
    )
    parser.add_argument(
        "--q-order",
        type=int,
        help="override the per-dimension default q_order (2D: 3, 3D: 2)",
    )
    parser.add_argument(
        "--source-level",
        type=int,
        help="override the per-dimension default source-box level "
        "(2D: 3, 3D: 2)",
    )
    parser.add_argument("--root-extent", type=float, default=2.0)
    parser.add_argument("--window-theta", type=float, default=16.0)
    parser.add_argument(
        "--p-star",
        default="4,6",
        help="comma-separated windowed channel counts",
    )
    parser.add_argument(
        "--smooth-orders",
        help="comma-separated smooth-remainder Gauss orders; "
        "defaults: smoke '16', full '8,16,24,32'",
    )
    parser.add_argument(
        "--mus",
        help="comma-separated parameter values, applied to every dimension; "
        "the default ladder is resolved per dimension from the window "
        "declaration as mu = fraction * window_theta / box_extent, which at "
        "the default geometry gives smoke '4,64' and full "
        "'1,2,4,8,16,24,32,48,64' in 2D (and half of each in 3D, whose box "
        "extent is twice as large)",
    )
    parser.add_argument(
        "--direct-policies",
        default="24,61;48,160",
        help="two 'regular,radial' Duffy policies separated by ';' "
        "(loose;tight)",
    )
    parser.add_argument(
        "--classical-channel-orders",
        default="24,61",
        help="'regular,radial' Duffy orders for classical channel builds",
    )
    parser.add_argument(
        "--chan-orders",
        help="semicolon-separated 'regular,radial' Duffy orders for the "
        "windowed channel builds, e.g. '48,61;64,121'; each pair is swept "
        "as its own family with its own channel cache.  Default: the single "
        "per-dimension tested pair (2D: 48,61; 3D: 20,61)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/windowed-rke-cache"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/benchmarks/windowed-rke-sweep"),
    )
    parser.add_argument(
        "--skip-3d-tight",
        action="store_true",
        help="skip the tight direct policy in 3D to guard runtime "
        "(the loose policy then serves as the deviation reference)",
    )
    args = parser.parse_args()

    try:
        _require_finite_positive(args.root_extent, "--root-extent")
        _require_finite_positive(args.window_theta, "--window-theta")
    except ValueError as exc:
        parser.error(str(exc))

    smoke = args.mode == "smoke"
    try:
        dims = _parse_csv_ints(args.dim or ("2" if smoke else "2,3"))
    except ValueError as exc:
        parser.error(str(exc))
    if any(dim not in (2, 3) for dim in dims):
        parser.error("--dim entries must be 2 or 3")
    kernel_map = {"helmholtz": "Helmholtz", "yukawa": "Yukawa"}
    kernels = []
    for name in args.kernels.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name not in kernel_map:
            parser.error(f"unknown kernel: {name}")
        kernels.append(kernel_map[name])
    if not kernels:
        parser.error("at least one kernel is required")

    try:
        p_stars = _parse_csv_ints(args.p_star)
        smooth_orders = _parse_csv_ints(
            args.smooth_orders or ("16" if smoke else "8,16,24,32")
        )
    except ValueError as exc:
        parser.error(str(exc))
    if any(p < 1 for p in p_stars):
        parser.error("--p-star entries must be >= 1")
    if any(order < 1 for order in smooth_orders):
        parser.error("--smooth-orders entries must be >= 1")
    if args.q_order is not None and args.q_order < 1:
        parser.error("--q-order must be >= 1")
    if args.source_level is not None and args.source_level < 0:
        parser.error("--source-level must be >= 0")
    # ``None`` defers the ladder to run_sweep, which resolves it against each
    # dimension's own box extent.
    try:
        mus = sorted(_parse_csv_floats(args.mus)) if args.mus else None
        if mus is not None:
            for mu in mus:
                _require_finite_positive(mu, "--mus entries")
        direct_policies = _parse_direct_policies(args.direct_policies)
        classical_channel_orders = _parse_order_pair(
            args.classical_channel_orders
        )
    except ValueError as exc:
        parser.error(str(exc))
    # Both smoke and full default to the single tested per-dimension pair;
    # a channel-order sweep is always requested explicitly.
    try:
        chan_orders = (
            _parse_order_pairs(args.chan_orders) if args.chan_orders else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    rows, run_info = run_sweep(
        mode=args.mode,
        dims=dims,
        kernels=kernels,
        q_order_override=args.q_order,
        source_level_override=args.source_level,
        root_extent=args.root_extent,
        window_theta=args.window_theta,
        p_stars=p_stars,
        smooth_orders=smooth_orders,
        mus=mus,
        direct_policies=direct_policies,
        classical_channel_orders=classical_channel_orders,
        chan_orders=chan_orders,
        cache_dir=args.cache_dir,
        skip_3d_tight=args.skip_3d_tight,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "windowed_rke_sweep.csv"
    write_csv(csv_path, rows)

    config = {
        "mode": args.mode,
        "dims": dims,
        "kernels": kernels,
        "q_order_override": args.q_order,
        "source_level_override": args.source_level,
        "root_extent": args.root_extent,
        "window_theta": args.window_theta,
        "p_stars": sorted(p_stars),
        "smooth_orders": sorted(smooth_orders),
        "mus": mus,
        "direct_policies": [list(policy) for policy in direct_policies],
        "classical_channel_orders": list(classical_channel_orders),
        "chan_orders": (
            [list(pair) for pair in chan_orders]
            if chan_orders is not None
            else None
        ),
        "classical_tolerance": CLASSICAL_TOLERANCE,
        "cache_dir": str(args.cache_dir),
        "skip_3d_tight": args.skip_3d_tight,
        "csv_path": str(csv_path),
        "row_count": len(rows),
        **run_info,
    }
    json_path = out_dir / "windowed_rke_sweep_config.json"
    with json_path.open("w") as outfile:
        json.dump(config, outfile, indent=2, sort_keys=True)
        outfile.write("\n")

    # A sweep whose windowed assemblies were all refused, or whose direct
    # reference builds all failed, produces a CSV with no usable comparison
    # in it; that must not be indistinguishable from success (this driver is
    # also a CI smoke check).
    refused = sum(1 for row in rows if row["windowed_status"] == "refused")
    windowed_failed = sum(
        1 for row in rows if row["windowed_status"] == "failed"
    )
    classical_failed = sum(
        1 for row in rows if row["classical_status"] == "failed"
    )
    no_reference = sum(1 for row in rows if not row["direct_reference_policy"])
    direct_failed = sum(
        1
        for row in rows
        if row["direct_loose_status"].startswith("failed:")
        or row["direct_tight_status"].startswith("failed:")
    )
    usable = sum(
        1
        for row in rows
        if row["windowed_status"] == "ok" and row["direct_reference_policy"]
    )
    print(
        f"[done] rows={len(rows)} usable={usable} "
        f"windowed_refused={refused} windowed_failed={windowed_failed} "
        f"classical_failed={classical_failed} "
        f"direct_failed={direct_failed} no_direct_reference={no_reference} "
        f"csv={csv_path} json={json_path} "
        f"total_s={run_info['total_seconds']:.1f}",
        flush=True,
    )
    if windowed_failed or classical_failed or direct_failed or not usable:
        print(
            "[error] sweep has unexpected assembly failures or no row with "
            "both a windowed 'ok' assembly and a usable direct reference",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
