#!/usr/bin/env python3
"""Emit PDE target-compression diagnostics for near-field tables.

Smoke mode builds one small 3D Laplace table. Full mode sweeps a small set of
2D/3D orders and records table-size and shell-reconstruction-loss metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIELDS = (
    "case_id",
    "mode",
    "kernel",
    "dim",
    "q_order",
    "n_cases",
    "n_q_points",
    "boundary_target_points",
    "interior_target_points",
    "full_table_entries",
    "shell_table_entries",
    "self_correction_entries",
    "full_table_bytes",
    "shell_table_bytes",
    "self_correction_bytes",
    "recovery_metadata_bytes",
    "online_value_bytes",
    "value_compression_ratio",
    "pde_reconstruction_max_abs_error",
    "pde_reconstruction_rel_l2_error",
    "pde_reconstruction_rel_linf_error",
)


@dataclass(frozen=True)
class CompressionCase:
    case_id: str
    dim: int
    q_order: int


SMOKE_CASES = (
    CompressionCase(case_id="laplace3d-q3", dim=3, q_order=3),
)

FULL_CASES = (
    CompressionCase(case_id="laplace2d-q3", dim=2, q_order=3),
    CompressionCase(case_id="laplace2d-q4", dim=2, q_order=4),
    CompressionCase(case_id="laplace3d-q3", dim=3, q_order=3),
    CompressionCase(case_id="laplace3d-q4", dim=3, q_order=4),
)


def _make_queue():
    import pyopencl as cl

    for platform in cl.get_platforms():
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))
    raise RuntimeError("no OpenCL devices available")


def _build_config(q_order):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    if q_order <= 3:
        regular_quad_order = 8
        radial_quad_order = 31
    else:
        regular_quad_order = 10
        radial_quad_order = 41

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
    )


def _full_table_data(table):
    if getattr(table, "table_data_is_symmetry_reduced", False):
        return np.asarray(table.reconstruct_full_table_from_symmetry())
    return np.asarray(table.data)


def _build_table(case, queue):
    from sumpy.kernel import LaplaceKernel

    from volumential.nearfield_potential_table import NearFieldInteractionTable

    table = NearFieldInteractionTable(
        quad_order=case.q_order,
        dim=case.dim,
        sumpy_kernel=LaplaceKernel(case.dim),
        progress_bar=False,
    )
    table.build_table(queue=queue, build_config=_build_config(case.q_order))
    return table


def _row(case: CompressionCase, mode: str, table) -> dict[str, Any]:
    from volumential.expansion_wrangler_fpnd import _find_self_case_id
    from volumential.nearfield_pde_targets import (
        boundary_shell_reconstruction_diagnostics,
        build_laplace_boundary_shell_reduction,
    )

    reduction = build_laplace_boundary_shell_reduction(
        table.q_points,
        table.quad_order,
        table.dim,
    )
    full_data = _full_table_data(table)
    diagnostics = boundary_shell_reconstruction_diagnostics(
        full_data,
        reduction,
        _find_self_case_id(table),
        table.n_cases,
    )

    shell_entries = table.n_cases * table.n_q_points * reduction.n_boundary_points
    self_correction_entries = reduction.n_interior_points * table.n_q_points
    full_bytes = int(full_data.nbytes)
    shell_bytes = int(shell_entries * np.dtype(table.dtype).itemsize)
    self_correction_bytes = int(self_correction_entries * np.dtype(table.dtype).itemsize)
    metadata_bytes = int(
        reduction.boundary_ids.nbytes
        + reduction.full_to_boundary.nbytes
        + reduction.full_to_interior.nbytes
        + reduction.recovery_matrix.nbytes
    )
    online_value_bytes = shell_bytes + self_correction_bytes

    return {
        "case_id": case.case_id,
        "mode": mode,
        "kernel": "Laplace",
        "dim": case.dim,
        "q_order": case.q_order,
        "n_cases": table.n_cases,
        "n_q_points": table.n_q_points,
        "boundary_target_points": reduction.n_boundary_points,
        "interior_target_points": reduction.n_interior_points,
        "full_table_entries": int(full_data.size),
        "shell_table_entries": int(shell_entries),
        "self_correction_entries": int(self_correction_entries),
        "full_table_bytes": full_bytes,
        "shell_table_bytes": shell_bytes,
        "self_correction_bytes": self_correction_bytes,
        "recovery_metadata_bytes": metadata_bytes,
        "online_value_bytes": online_value_bytes,
        "value_compression_ratio": full_bytes / max(online_value_bytes, 1),
        **diagnostics,
    }


def run(mode: str):
    queue = _make_queue()
    cases = SMOKE_CASES if mode == "smoke" else FULL_CASES
    return [_row(case, mode, _build_table(case, queue)) for case in cases]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/pde-target-compression.csv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = run(args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
