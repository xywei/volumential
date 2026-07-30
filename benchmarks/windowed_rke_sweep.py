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
    ill-conditioned recombination);
(c) direct fixed-parameter builds through the table manager at two Duffy
    quadrature policies, whose mutual disagreement estimates the reference
    floor below which assembled-vs-direct deviations are quadrature noise.

Deviations are reported against the tight direct policy as both relative
max-entry and relative L2 over the symmetry-reduced entries.  Windowed
channel families are built (or reused) once per ``(dim, q_order, level,
window_theta)`` from a shared cache, so the one-off channel build cost and
the marginal per-parameter assembly cost are recorded separately.

Smoke mode is intended for CI/local validation.  Full mode is intended for
metadata-wrapped runs on a controlled remote compute host.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    return policies


def _parse_order_pair(raw: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("expected a 'regular,radial' integer pair")
    return parts[0], parts[1]


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
    from volumential.rke_table_assembly import (
        _windowed_channel_cache_file,
        get_windowed_channel_table,
    )

    was_cold = False
    for m in range(max_p_star):
        cache_file, _ = _windowed_channel_cache_file(
            cache_path,
            dim,
            q_order,
            source_box_level,
            root_extent,
            window_theta,
            m,
            chan_regular_order,
            chan_radial_order,
        )
        if not cache_file.is_file():
            was_cold = True

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
    except (ValueError, RuntimeError) as exc:
        return {
            "windowed_status": "refused",
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
    message = str(exc)
    if "cannot certify" in message:
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
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.rke_table_assembly import assemble_parameterized_table

    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=channel_orders[0],
        radial_quad_order=channel_orders[1],
    )
    start = time.perf_counter()
    try:
        table, certificate = assemble_parameterized_table(
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
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        return {
            "classical_status": "refused",
            "classical_refusal": _classify_classical_refusal(exc),
            "classical_refusal_detail": f"{type(exc).__name__}: {exc}",
            "classical_assemble_seconds": time.perf_counter() - start,
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
    mus: list[float],
    direct_policies: list[tuple[int, int]],
    classical_channel_orders: tuple[int, int],
    cache_dir: Path,
    skip_3d_tight: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from volumential.rke_table_assembly import _resolve_channel_orders

    sweep_start = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    queue = _make_queue()

    rows: list[dict[str, Any]] = []
    channel_prep_records: dict[str, Any] = {}

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
        box_extent = float(root_extent) * 0.5**int(source_level)
        chan_regular_order, chan_radial_order = _resolve_channel_orders(
            dim, None, None
        )

        windowed_cache = cache_dir / f"windowed-channels-d{dim}-q{q_order}.db"
        classical_cache = (
            cache_dir / f"classical-channels-d{dim}-q{q_order}.sqlite"
        )

        print(
            f"[config] dim={dim} q_order={q_order} level={source_level} "
            f"box_extent={box_extent:g} window_theta={window_theta:g} "
            f"chan_orders={chan_regular_order}/{chan_radial_order}",
            flush=True,
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
            f"cold={channels['channel_build_was_cold']} "
            f"build_s={channels['channel_build_seconds']:.2f} "
            f"n_reduced_entries={channels['n_reduced_entries']}",
            flush=True,
        )
        channel_prep_records[f"dim{dim}"] = {
            "q_order": q_order,
            "source_box_level": source_level,
            "chan_regular_order": chan_regular_order,
            "chan_radial_order": chan_radial_order,
            "channel_build_seconds": channels["channel_build_seconds"],
            "channel_build_was_cold": channels["channel_build_was_cold"],
            "n_reduced_entries": channels["n_reduced_entries"],
        }
        entry_ids = channels["entry_ids"]
        n_entries = channels["n_reduced_entries"]

        for kernel in kernels:
            for mu in mus:
                theta = float(mu) * box_extent
                mu_tag = f"{mu:g}".replace("-", "m").replace(".", "p")

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
                            f"  [row] p_star={p_star} "
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
                                    f"{kernel.lower()}{dim}d-mu{mu:g}"
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
                                "smooth_quad_order_requested": smooth_order,
                                "chan_regular_order": chan_regular_order,
                                "chan_radial_order": chan_radial_order,
                                "channel_build_was_cold": channels[
                                    "channel_build_was_cold"
                                ],
                                "channel_build_seconds": channels[
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
                                "direct_reference_policy": reference_policy,
                                "windowed_vs_direct_rel_max_entry": (
                                    windowed_rel_max
                                ),
                                "windowed_vs_direct_rel_l2": windowed_rel_l2,
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
                        if "smooth_quad_order_used" in windowed:
                            row["smooth_quad_order_used"] = windowed[
                                "smooth_quad_order_used"
                            ]
                        for key, value in classical.items():
                            if key != "values" and key in FIELDS:
                                row[key] = value
                        rows.append(row)

    total_seconds = time.perf_counter() - sweep_start
    for row in rows:
        row["benchmark_total_seconds"] = total_seconds
    return rows, {
        "channel_prep": channel_prep_records,
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
        help="comma-separated parameter values; defaults: smoke '4,64', "
        "full '1,2,4,8,16,24,32,48,64'",
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

    smoke = args.mode == "smoke"
    dims = _parse_csv_ints(args.dim or ("2" if smoke else "2,3"))
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

    p_stars = _parse_csv_ints(args.p_star)
    if any(p < 1 for p in p_stars):
        parser.error("--p-star entries must be >= 1")
    smooth_orders = _parse_csv_ints(
        args.smooth_orders or ("16" if smoke else "8,16,24,32")
    )
    mus = sorted(
        _parse_csv_floats(args.mus or ("4,64" if smoke else "1,2,4,8,16,24,32,48,64"))
    )
    if any(mu <= 0 for mu in mus):
        parser.error("--mus entries must be positive")
    direct_policies = _parse_direct_policies(args.direct_policies)
    classical_channel_orders = _parse_order_pair(args.classical_channel_orders)

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

    print(
        f"[done] rows={len(rows)} csv={csv_path} json={json_path} "
        f"total_s={run_info['total_seconds']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
