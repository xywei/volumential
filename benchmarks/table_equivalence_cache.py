#!/usr/bin/env python3
"""Emit table-equivalence and cache-economics CSVs for Paper 1.

The benchmark builds near-field tables for a small matrix of kernels and source
box levels, compares per-level tables against a canonical level-0 table after
the expected scale factor is applied, and records cold-build/warm-load cache
timings exposed by :mod:`volumential.table_manager`.

Smoke mode is intended for CI/local validation. Full mode is intended for paper
artifact generation on a controlled machine such as ``ipa``.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dim: int
    kernel_type: str
    derivative: str
    q_orders: tuple[int, ...]
    max_level: int
    scale_exponent: int
    scale_rule: str = "power"


SMOKE_CASES = (
    BenchmarkCase(
        case_id="laplace3d-scalar-q1-l0-1",
        dim=3,
        kernel_type="Laplace",
        derivative="none",
        q_orders=(1,),
        max_level=1,
        scale_exponent=2,
    ),
    BenchmarkCase(
        case_id="laplace3d-dx-q1-l0-1",
        dim=3,
        kernel_type="Laplace-Dx",
        derivative="target_dx",
        q_orders=(1,),
        max_level=1,
        scale_exponent=1,
    ),
    BenchmarkCase(
        case_id="laplace2d-log-q1-l0-1",
        dim=2,
        kernel_type="Laplace",
        derivative="none",
        q_orders=(1,),
        max_level=1,
        scale_exponent=2,
        scale_rule="log",
    ),
    BenchmarkCase(
        case_id="laplace3d-source-dx-q1-l0-1",
        dim=3,
        kernel_type="Laplace-Sx",
        derivative="source_dx",
        q_orders=(1,),
        max_level=1,
        scale_exponent=1,
    ),
)

FULL_CASES = (
    BenchmarkCase(
        case_id="laplace3d-scalar-q1-3-l0-4",
        dim=3,
        kernel_type="Laplace",
        derivative="none",
        q_orders=(1, 2, 3),
        max_level=4,
        scale_exponent=2,
    ),
    BenchmarkCase(
        case_id="laplace3d-dx-q1-3-l0-4",
        dim=3,
        kernel_type="Laplace-Dx",
        derivative="target_dx",
        q_orders=(1, 2, 3),
        max_level=4,
        scale_exponent=1,
    ),
    BenchmarkCase(
        case_id="laplace2d-log-q1-4-l0-4",
        dim=2,
        kernel_type="Laplace",
        derivative="none",
        q_orders=(1, 2, 3, 4),
        max_level=4,
        scale_exponent=2,
        scale_rule="log",
    ),
    BenchmarkCase(
        case_id="laplace3d-source-dx-q1-3-l0-4",
        dim=3,
        kernel_type="Laplace-Sx",
        derivative="source_dx",
        q_orders=(1, 2, 3),
        max_level=4,
        scale_exponent=1,
    ),
)


EQUIVALENCE_FIELDS = (
    "case_id",
    "mode",
    "kernel",
    "dim",
    "derivative",
    "q_order",
    "source_box_level",
    "source_box_extent",
    "canonical_source_box_extent",
    "scale_exponent",
    "scale_rule",
    "scale_factor",
    "log_correction_applied",
    "finite_entry_count",
    "max_abs_mismatch",
    "max_rel_mismatch",
    "max_abs_mismatch_without_log_correction",
    "max_rel_mismatch_without_log_correction",
    "reference_linf",
)

CACHE_FIELDS = (
    "case_id",
    "mode",
    "policy",
    "kernel",
    "dim",
    "derivative",
    "q_order",
    "max_source_box_level",
    "source_box_level",
    "cache_state",
    "is_recomputed",
    "total_get_s",
    "table_build_s",
    "payload_serialize_s",
    "db_write_commit_s",
    "record_fetch_s",
    "payload_deserialize_s",
    "kwargs_load_s",
    "payload_bytes",
    "cache_bytes",
    "full_entry_count",
    "representative_count",
    "compression_ratio",
    "orbit_size_histogram",
    "sign_metadata_count",
    "negative_scale_count",
    "unreduced_payload_bytes",
    "reconstructed_payload_bytes",
    "metadata_payload_bytes",
    "max_reconstruction_error",
    "l2_reconstruction_error",
)


def _default_build_config(q_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    if q_order <= 2:
        regular_quad_order = 6
        radial_quad_order = 21
    else:
        regular_quad_order = 8
        radial_quad_order = 31

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _timing_value(timings: dict[str, Any], group: str, key: str) -> float | int | None:
    subgroup = timings.get(group)
    if not subgroup:
        return None
    return subgroup.get(key)


def _sqlite_database_bytes(connection) -> int:
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    return page_count * page_size


def _full_table_data(table) -> np.ndarray:
    if getattr(table, "table_data_is_symmetry_reduced", False):
        return np.asarray(table.reconstruct_full_table_from_symmetry(), dtype=table.dtype)
    return np.asarray(table.data, dtype=table.dtype)


def _apply_log_laplace_scaling(
    canonical_table,
    canonical_data: np.ndarray,
    *,
    extent_ratio: float,
    scale_factor: float,
):
    scaled = np.array(canonical_data, copy=True)
    for entry_id, value in enumerate(canonical_data):
        if not np.isfinite(value):
            continue
        source_mode_index = canonical_table.decode_index(entry_id)["source_mode_index"]
        displacement = (
            -0.5
            / np.pi
            * scale_factor
            * np.log(extent_ratio)
            * canonical_table.mode_normalizers[source_mode_index]
        )
        scaled[entry_id] = scale_factor * value + displacement
    return scaled


def _mismatch_stats(reference: np.ndarray, candidate: np.ndarray):
    finite_mask = np.isfinite(candidate) & np.isfinite(reference)
    if not np.any(finite_mask):
        return finite_mask, np.nan, np.nan, np.nan

    diff = np.abs(reference[finite_mask] - candidate[finite_mask])
    reference_linf = float(np.max(np.abs(reference[finite_mask])))
    max_abs_mismatch = float(np.max(diff))
    max_rel_mismatch = float(max_abs_mismatch / max(reference_linf, 1e-300))
    return finite_mask, max_abs_mismatch, max_rel_mismatch, reference_linf


def _equivalence_row(
    *,
    mode: str,
    case: BenchmarkCase,
    q_order: int,
    canonical_table,
    level_table,
) -> dict[str, Any]:
    canonical_extent = float(canonical_table.source_box_extent)
    level_extent = float(level_table.source_box_extent)
    extent_ratio = level_extent / canonical_extent
    scale_factor = extent_ratio ** case.scale_exponent

    canonical_data = _full_table_data(canonical_table)
    scaled_canonical_without_log_correction = scale_factor * canonical_data
    if case.scale_rule == "log":
        scaled_canonical = _apply_log_laplace_scaling(
            canonical_table,
            canonical_data,
            extent_ratio=extent_ratio,
            scale_factor=scale_factor,
        )
    else:
        scaled_canonical = scaled_canonical_without_log_correction
    level_data = _full_table_data(level_table)
    finite_mask, max_abs_mismatch, max_rel_mismatch, reference_linf = _mismatch_stats(
        level_data,
        scaled_canonical,
    )
    _, max_abs_mismatch_without_log, max_rel_mismatch_without_log, _ = _mismatch_stats(
        level_data,
        scaled_canonical_without_log_correction,
    )

    return {
        "case_id": case.case_id,
        "mode": mode,
        "kernel": case.kernel_type,
        "dim": case.dim,
        "derivative": case.derivative,
        "q_order": q_order,
        "source_box_level": int(level_table.source_box_level),
        "source_box_extent": level_extent,
        "canonical_source_box_extent": canonical_extent,
        "scale_exponent": case.scale_exponent,
        "scale_rule": case.scale_rule,
        "scale_factor": scale_factor,
        "log_correction_applied": case.scale_rule == "log",
        "finite_entry_count": int(np.count_nonzero(finite_mask)),
        "max_abs_mismatch": max_abs_mismatch,
        "max_rel_mismatch": max_rel_mismatch,
        "max_abs_mismatch_without_log_correction": (
            max_abs_mismatch_without_log if case.scale_rule == "log" else None
        ),
        "max_rel_mismatch_without_log_correction": (
            max_rel_mismatch_without_log if case.scale_rule == "log" else None
        ),
        "reference_linf": reference_linf,
    }


def _cache_row(
    *,
    mode: str,
    case: BenchmarkCase,
    q_order: int,
    max_level: int,
    level: int,
    cache_state: str,
    is_recomputed: bool,
    timings: dict[str, Any],
    table,
    cache_bytes: int,
) -> dict[str, Any]:
    diagnostics = table.get_symmetry_reduction_diagnostics()
    timing_group = "compute" if cache_state == "cold" else "load"
    payload_bytes = _timing_value(timings, timing_group, "payload_bytes")
    return {
        "case_id": case.case_id,
        "mode": mode,
        "policy": "per_level_direct" if level > 0 else "canonical_reference",
        "kernel": case.kernel_type,
        "dim": case.dim,
        "derivative": case.derivative,
        "q_order": q_order,
        "max_source_box_level": max_level,
        "source_box_level": level,
        "cache_state": cache_state,
        "is_recomputed": bool(is_recomputed),
        "total_get_s": _as_float(timings.get("total_s")),
        "table_build_s": _as_float(_timing_value(timings, "compute", "table_build_s")),
        "payload_serialize_s": _as_float(
            _timing_value(timings, "compute", "payload_serialize_s")
        ),
        "db_write_commit_s": _as_float(
            _timing_value(timings, "compute", "db_write_commit_s")
        ),
        "record_fetch_s": _as_float(_timing_value(timings, "load", "record_fetch_s")),
        "payload_deserialize_s": _as_float(
            _timing_value(timings, "load", "payload_deserialize_s")
        ),
        "kwargs_load_s": _as_float(_timing_value(timings, "load", "kwargs_load_s")),
        "payload_bytes": payload_bytes,
        "cache_bytes": cache_bytes,
        **asdict(diagnostics),
    }


def _build_or_load_table(tm, *, case: BenchmarkCase, q_order: int, level: int, queue):
    build_kwargs = {"build_config": _default_build_config(q_order)}
    if case.derivative == "source_dx":
        from sumpy.kernel import AxisSourceDerivative, LaplaceKernel

        build_kwargs["sumpy_knl"] = AxisSourceDerivative(0, LaplaceKernel(case.dim))

    return tm.get_table(
        case.dim,
        case.kernel_type,
        q_order,
        source_box_level=level,
        force_recompute=False,
        queue=queue,
        **build_kwargs,
    )


def run_benchmark(*, mode: str, cache_dir: Path):
    import pyopencl as cl

    from volumential.table_manager import NearFieldInteractionTableManager

    cases = SMOKE_CASES if mode == "smoke" else FULL_CASES
    cache_dir.mkdir(parents=True, exist_ok=True)

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    equivalence_rows = []
    cache_rows = []

    for case in cases:
        for q_order in case.q_orders:
            cache_path = cache_dir / f"{case.case_id}-q{q_order}.sqlite"
            if cache_path.exists():
                cache_path.unlink()

            canonical_table = None
            with NearFieldInteractionTableManager(
                str(cache_path), root_extent=2.0, queue=queue
            ) as tm:
                cold_tables = []
                for level in range(case.max_level + 1):
                    table, is_recomputed = _build_or_load_table(
                        tm,
                        case=case,
                        q_order=q_order,
                        level=level,
                        queue=queue,
                    )
                    if level == 0:
                        canonical_table = table
                    cold_tables.append(table)
                    cache_rows.append(
                        _cache_row(
                            mode=mode,
                            case=case,
                            q_order=q_order,
                            max_level=case.max_level,
                            level=level,
                            cache_state="cold",
                            is_recomputed=is_recomputed,
                            timings=dict(tm.last_get_table_timings),
                            table=table,
                            cache_bytes=_sqlite_database_bytes(tm.datafile),
                        )
                    )

                assert canonical_table is not None
                for table in cold_tables:
                    equivalence_rows.append(
                        _equivalence_row(
                            mode=mode,
                            case=case,
                            q_order=q_order,
                            canonical_table=canonical_table,
                            level_table=table,
                        )
                    )

                for level in range(case.max_level + 1):
                    table, is_recomputed = _build_or_load_table(
                        tm,
                        case=case,
                        q_order=q_order,
                        level=level,
                        queue=queue,
                    )
                    cache_rows.append(
                        _cache_row(
                            mode=mode,
                            case=case,
                            q_order=q_order,
                            max_level=case.max_level,
                            level=level,
                            cache_state="warm",
                            is_recomputed=is_recomputed,
                            timings=dict(tm.last_get_table_timings),
                            table=table,
                            cache_bytes=_sqlite_database_bytes(tm.datafile),
                        )
                    )

    return equivalence_rows, cache_rows


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/benchmarks/table-equivalence-cache"),
        help="directory for output CSV files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="directory for temporary SQLite table caches; defaults to a temp dir",
    )
    args = parser.parse_args()

    if args.cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="volumential-table-bench-") as temp_dir:
            equivalence_rows, cache_rows = run_benchmark(
                mode=args.mode,
                cache_dir=Path(temp_dir),
            )
    else:
        equivalence_rows, cache_rows = run_benchmark(
            mode=args.mode,
            cache_dir=args.cache_dir,
        )

    _write_csv(args.out_dir / "table_equivalence.csv", EQUIVALENCE_FIELDS, equivalence_rows)
    _write_csv(args.out_dir / "cache_economics.csv", CACHE_FIELDS, cache_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
