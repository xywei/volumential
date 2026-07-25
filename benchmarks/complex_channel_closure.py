#!/usr/bin/env python3
"""Complex-channel exact-closure benchmark (log-periodic power kernel).

The kernel ``g(r) = r^(-alpha) * (1 + eps * cos(omega * log r))`` spans an
exactly dilation-closed three-channel family

    { r^(-alpha),  r^(-alpha) cos(omega log r),  r^(-alpha) sin(omega log r) }.

Dilation by ``rho`` acts as the scalar ``rho^(-alpha)`` times a 2x2 rotation by
``omega log rho`` on the (cos, sin) pair, with zero smooth residual and zero
singular tail.  The benchmark builds direct near-field tables for ``g`` at
source-box levels ``0..L`` and compares them against one canonical level-0
channel triple recombined through the rotation-block coefficient matrix:

    T_direct(h) = h^(d - alpha) * (T_base
                                   + eps * cos(omega log h) * T_cos
                                   - eps * sin(omega log h) * T_sin).

Because the Duffy-radial quadrature geometry is affinely similar across levels
and the identity holds pointwise, the mismatch is pure floating-point roundoff.
The tanh-sinh radial rule clusters nodes doubly-exponentially toward the
singular point, which resolves the bounded log-periodic oscillation; a
quadrature self-check at bumped orders quantifies the quadrature stability
separately from the closure mismatch.

Smoke mode is intended for CI/local validation.  Full mode is intended for
paper artifact generation on a controlled remote compute host.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

EQUIVALENCE_FIELDS = (
    "case_id",
    "mode",
    "dim",
    "q_order",
    "alpha",
    "eps",
    "omega",
    "source_box_level",
    "source_box_extent",
    "canonical_source_box_extent",
    "theta",
    "scale_factor",
    "cos_coefficient",
    "sin_coefficient",
    "finite_entry_count",
    "max_abs_mismatch",
    "max_rel_mismatch",
    "reference_linf",
)

BUILD_FIELDS = (
    "case_id",
    "mode",
    "dim",
    "q_order",
    "table_id",
    "source_box_level",
    "source_box_extent",
    "radial_rule",
    "regular_quad_order",
    "radial_quad_order",
    "n_jobs",
    "build_wall_s",
    "n_representative_entries",
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

QUADCHECK_FIELDS = (
    "case_id",
    "mode",
    "dim",
    "q_order",
    "table_id",
    "base_regular_quad_order",
    "base_radial_quad_order",
    "bumped_regular_quad_order",
    "bumped_radial_quad_order",
    "max_abs_difference",
    "max_rel_difference",
    "reference_linf",
)


@dataclass(frozen=True)
class ClosureCase:
    case_id: str
    dim: int
    q_order: int
    max_level: int
    regular_quad_order: int
    radial_quad_order: int


SMOKE_CASES = (
    ClosureCase(
        case_id="logper2d-q2-l0-1",
        dim=2,
        q_order=2,
        max_level=1,
        regular_quad_order=6,
        radial_quad_order=21,
    ),
)

FULL_CASES = (
    ClosureCase(
        case_id="logper2d-q4-l0-4",
        dim=2,
        q_order=4,
        max_level=4,
        regular_quad_order=12,
        radial_quad_order=41,
    ),
)

FULL_3D_CASE = ClosureCase(
    case_id="logper3d-q3-l0-4",
    dim=3,
    q_order=3,
    max_level=4,
    regular_quad_order=10,
    radial_quad_order=31,
)


# {{{ kernel channels

def _radius(coords):
    return math.sqrt(sum(c * c for c in coords))


class LogPeriodicKernel:
    """g(r) = r^(-alpha) (1 + eps cos(omega log r))."""

    def __init__(self, alpha, eps, omega):
        self.alpha = alpha
        self.eps = eps
        self.omega = omega

    def __call__(self, *coords):
        r = _radius(coords)
        return r ** (-self.alpha) * (
            1.0 + self.eps * math.cos(self.omega * math.log(r))
        )


class PowerKernel:
    """base channel: r^(-alpha)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, *coords):
        r = _radius(coords)
        return r ** (-self.alpha)


class PowerCosLogKernel:
    """cos channel: r^(-alpha) cos(omega log r)."""

    def __init__(self, alpha, omega):
        self.alpha = alpha
        self.omega = omega

    def __call__(self, *coords):
        r = _radius(coords)
        return r ** (-self.alpha) * math.cos(self.omega * math.log(r))


class PowerSinLogKernel:
    """sin channel: r^(-alpha) sin(omega log r)."""

    def __init__(self, alpha, omega):
        self.alpha = alpha
        self.omega = omega

    def __call__(self, *coords):
        r = _radius(coords)
        return r ** (-self.alpha) * math.sin(self.omega * math.log(r))


def closure_self_test(alpha, eps, omega, dim, rng):
    """Pointwise check of the dilation-closure identity used for recombination."""
    g = LogPeriodicKernel(alpha, eps, omega)
    base = PowerKernel(alpha)
    cosk = PowerCosLogKernel(alpha, omega)
    sink = PowerSinLogKernel(alpha, omega)

    for _ in range(100):
        coords = rng.uniform(0.05, 1.0, size=dim)
        h = 2.0 ** (-rng.integers(0, 6))
        theta = omega * math.log(h)
        direct = g(*(h * coords))
        recombined = (h ** (-alpha)) * (
            base(*coords)
            + eps * math.cos(theta) * cosk(*coords)
            - eps * math.sin(theta) * sink(*coords)
        )
        rel = abs(direct - recombined) / max(abs(direct), 1e-300)
        if rel > 1e-13:
            raise RuntimeError(
                f"closure identity self-test failed: rel={rel:.3e} "
                f"coords={coords} h={h}"
            )

# }}}


# {{{ table construction

def _tensor_gauss_legendre_q_points(q_order, dim, extent, dtype):
    """Reproduce the constructor's meshgen-derived quad points without OpenCL.

    Matches ``volumential.meshgen.make_uniform_cubic_grid(degree=q, level=1)``
    followed by the constructor's affine map to ``[0, extent]^dim`` and
    dictionary-order sort.
    """
    from modepy import LegendreGaussQuadrature

    nodes = np.asarray(
        LegendreGaussQuadrature(q_order - 1, force_dim_axis=True).nodes
    ).reshape(-1)
    grids = np.meshgrid(*([nodes] * dim), indexing="ij")
    pts = np.vstack([g.ravel() for g in grids]).T
    mapped = 0.5 * extent * (pts + 1.0)
    ordering = sorted(
        range(len(mapped)), key=lambda i: list(np.floor(mapped[i] * 10000))
    )
    return np.asarray(mapped[ordering], dtype=dtype)


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(table, radial_rule, regular_quad_order, radial_quad_order, mp_dps):
    _WORKER_STATE["table"] = table
    _WORKER_STATE["radial_rule"] = radial_rule
    _WORKER_STATE["regular_quad_order"] = regular_quad_order
    _WORKER_STATE["radial_quad_order"] = radial_quad_order
    _WORKER_STATE["mp_dps"] = mp_dps


def _compute_entry(entry_id):
    table = _WORKER_STATE["table"]
    return table.compute_table_entry_duffy_radial(
        entry_id,
        radial_rule=_WORKER_STATE["radial_rule"],
        deg_theta=_WORKER_STATE["regular_quad_order"],
        radial_quad_order=_WORKER_STATE["radial_quad_order"],
        mp_dps=_WORKER_STATE["mp_dps"],
    )


def build_channel_table(
    *,
    kernel_func,
    dim,
    q_order,
    level,
    root_extent,
    regular_quad_order,
    radial_quad_order,
    radial_rule="tanh-sinh-fast",
    mp_dps=50,
    n_jobs=1,
):
    """Build a symmetry-reduced near-field table for a scalar radial kernel.

    Follows the scalar Duffy-radial build path (representative entries only)
    with an optional fork-based parallel entry loop; the per-entry quadrature
    is identical to ``NearFieldInteractionTable.build_table``.
    """
    from volumential.nearfield_potential_table import NearFieldInteractionTable

    extent = root_extent * (2.0 ** (-level))
    table = NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        kernel_func=kernel_func,
        kernel_type=None,
        sumpy_kernel=None,
        source_box_extent=extent,
        precomputed_q_points=_tensor_gauss_legendre_q_points(
            q_order, dim, extent, np.float64
        ),
        progress_bar=False,
    )
    table.source_box_level = level

    t_start = time.perf_counter()
    entry_ids = [
        int(entry_id) for entry_id in table._get_invariant_entry_info()["entry_ids"]
    ]
    values = np.empty(len(entry_ids), dtype=table.dtype)

    if n_jobs > 1:
        ctx = get_context("fork")
        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(table, radial_rule, regular_quad_order, radial_quad_order,
                      mp_dps),
        ) as pool:
            id_to_slot = {eid: i for i, eid in enumerate(entry_ids)}
            for entry_id, value in pool.imap_unordered(
                _compute_entry, entry_ids, chunksize=8
            ):
                values[id_to_slot[int(entry_id)]] = value
    else:
        for islot, entry_id in enumerate(entry_ids):
            _, value = table.compute_table_entry_duffy_radial(
                entry_id,
                radial_rule=radial_rule,
                deg_theta=regular_quad_order,
                radial_quad_order=radial_quad_order,
                mp_dps=mp_dps,
            )
            values[islot] = value

    table.set_reduced_table_data(entry_ids, values)
    table.is_built = True
    build_wall_s = time.perf_counter() - t_start

    return table, build_wall_s


def _full_table_data(table):
    if getattr(table, "table_data_is_symmetry_reduced", False):
        return np.asarray(
            table.reconstruct_full_table_from_symmetry(), dtype=table.dtype
        )
    return np.asarray(table.data, dtype=table.dtype)

# }}}


# {{{ rows

def _mismatch_stats(reference, candidate):
    finite_mask = np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(finite_mask):
        return 0, np.nan, np.nan, np.nan
    diff = np.abs(reference[finite_mask] - candidate[finite_mask])
    reference_linf = float(np.max(np.abs(reference[finite_mask])))
    max_abs = float(np.max(diff))
    max_rel = float(max_abs / max(reference_linf, 1e-300))
    return int(np.count_nonzero(finite_mask)), max_abs, max_rel, reference_linf


def _build_row(
    *,
    case: ClosureCase,
    mode: str,
    table_id: str,
    table,
    build_wall_s: float,
    n_jobs: int,
    radial_rule: str,
) -> dict[str, Any]:
    diagnostics = table.get_symmetry_reduction_diagnostics()
    return {
        "case_id": case.case_id,
        "mode": mode,
        "dim": case.dim,
        "q_order": case.q_order,
        "table_id": table_id,
        "source_box_level": int(table.source_box_level),
        "source_box_extent": float(table.source_box_extent),
        "radial_rule": radial_rule,
        "regular_quad_order": case.regular_quad_order,
        "radial_quad_order": case.radial_quad_order,
        "n_jobs": n_jobs,
        "build_wall_s": build_wall_s,
        "n_representative_entries": int(len(table.reduced_entry_ids)),
        **asdict(diagnostics),
    }

# }}}


def run_case(
    *,
    case: ClosureCase,
    mode: str,
    alpha: float,
    eps: float,
    omega: float,
    root_extent: float,
    n_jobs: int,
    with_quadcheck: bool,
):
    radial_rule = "tanh-sinh-fast"
    equivalence_rows = []
    build_rows = []
    quadcheck_rows = []

    channels = {
        "channel-base": PowerKernel(alpha),
        "channel-cos": PowerCosLogKernel(alpha, omega),
        "channel-sin": PowerSinLogKernel(alpha, omega),
    }

    channel_tables = {}
    for table_id, kernel_func in channels.items():
        table, build_wall_s = build_channel_table(
            kernel_func=kernel_func,
            dim=case.dim,
            q_order=case.q_order,
            level=0,
            root_extent=root_extent,
            regular_quad_order=case.regular_quad_order,
            radial_quad_order=case.radial_quad_order,
            radial_rule=radial_rule,
            n_jobs=n_jobs,
        )
        channel_tables[table_id] = table
        build_rows.append(
            _build_row(
                case=case,
                mode=mode,
                table_id=table_id,
                table=table,
                build_wall_s=build_wall_s,
                n_jobs=n_jobs,
                radial_rule=radial_rule,
            )
        )
        print(
            f"[{case.case_id}] built {table_id} "
            f"({len(table.reduced_entry_ids)} representatives, "
            f"{build_wall_s:.1f} s)",
            flush=True,
        )

    channel_data = {
        table_id: _full_table_data(table)
        for table_id, table in channel_tables.items()
    }
    canonical_extent = float(channel_tables["channel-base"].source_box_extent)

    direct_kernel = LogPeriodicKernel(alpha, eps, omega)
    for level in range(case.max_level + 1):
        direct_table, build_wall_s = build_channel_table(
            kernel_func=direct_kernel,
            dim=case.dim,
            q_order=case.q_order,
            level=level,
            root_extent=root_extent,
            regular_quad_order=case.regular_quad_order,
            radial_quad_order=case.radial_quad_order,
            radial_rule=radial_rule,
            n_jobs=n_jobs,
        )
        build_rows.append(
            _build_row(
                case=case,
                mode=mode,
                table_id=f"direct-l{level}",
                table=direct_table,
                build_wall_s=build_wall_s,
                n_jobs=n_jobs,
                radial_rule=radial_rule,
            )
        )
        print(
            f"[{case.case_id}] built direct-l{level} "
            f"({build_wall_s:.1f} s)",
            flush=True,
        )

        extent = float(direct_table.source_box_extent)
        # dilation factor relative to the canonical (level-0) box
        rho = extent / canonical_extent
        theta = omega * math.log(rho)
        scale_factor = rho ** (case.dim - alpha)
        cos_coefficient = eps * math.cos(theta)
        sin_coefficient = -eps * math.sin(theta)

        recombined = scale_factor * (
            channel_data["channel-base"]
            + cos_coefficient * channel_data["channel-cos"]
            + sin_coefficient * channel_data["channel-sin"]
        )
        direct_data = _full_table_data(direct_table)
        count, max_abs, max_rel, reference_linf = _mismatch_stats(
            direct_data, recombined
        )

        equivalence_rows.append(
            {
                "case_id": case.case_id,
                "mode": mode,
                "dim": case.dim,
                "q_order": case.q_order,
                "alpha": alpha,
                "eps": eps,
                "omega": omega,
                "source_box_level": level,
                "source_box_extent": extent,
                "canonical_source_box_extent": canonical_extent,
                "theta": theta,
                "scale_factor": scale_factor,
                "cos_coefficient": cos_coefficient,
                "sin_coefficient": sin_coefficient,
                "finite_entry_count": count,
                "max_abs_mismatch": max_abs,
                "max_rel_mismatch": max_rel,
                "reference_linf": reference_linf,
            }
        )
        print(
            f"[{case.case_id}] level {level}: max_rel_mismatch={max_rel:.3e}",
            flush=True,
        )

    if with_quadcheck:
        bumped_regular = case.regular_quad_order + 8
        bumped_radial = case.radial_quad_order + 20
        for table_id, kernel_func in channels.items():
            bumped_table, _ = build_channel_table(
                kernel_func=kernel_func,
                dim=case.dim,
                q_order=case.q_order,
                level=0,
                root_extent=root_extent,
                regular_quad_order=bumped_regular,
                radial_quad_order=bumped_radial,
                radial_rule=radial_rule,
                n_jobs=n_jobs,
            )
            count, max_abs, max_rel, reference_linf = _mismatch_stats(
                _full_table_data(bumped_table), channel_data[table_id]
            )
            quadcheck_rows.append(
                {
                    "case_id": case.case_id,
                    "mode": mode,
                    "dim": case.dim,
                    "q_order": case.q_order,
                    "table_id": table_id,
                    "base_regular_quad_order": case.regular_quad_order,
                    "base_radial_quad_order": case.radial_quad_order,
                    "bumped_regular_quad_order": bumped_regular,
                    "bumped_radial_quad_order": bumped_radial,
                    "max_abs_difference": max_abs,
                    "max_rel_difference": max_rel,
                    "reference_linf": reference_linf,
                }
            )
            print(
                f"[{case.case_id}] quadcheck {table_id}: "
                f"max_rel_difference={max_rel:.3e}",
                flush=True,
            )

    return equivalence_rows, build_rows, quadcheck_rows


def _write_csv(path: Path, fieldnames, rows) -> None:
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
        default=Path("build/benchmarks/complex-channel-closure"),
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=4.0)
    parser.add_argument("--root-extent", type=float, default=1.0)
    parser.add_argument(
        "--jobs", type=int, default=1, help="parallel workers for table entries"
    )
    parser.add_argument(
        "--include-3d",
        action="store_true",
        help="also run the optional 3D q=3 case (full mode only)",
    )
    args = parser.parse_args()

    if not 0.0 < args.alpha:
        raise SystemExit("alpha must be positive")

    cases = list(SMOKE_CASES if args.mode == "smoke" else FULL_CASES)
    if args.include_3d and args.mode == "full":
        cases.append(FULL_3D_CASE)

    for case in cases:
        if not args.alpha < case.dim:
            raise SystemExit(
                f"alpha={args.alpha} must be < dim={case.dim} for integrability"
            )

    rng = np.random.default_rng(20260725)
    for dim in sorted({case.dim for case in cases}):
        closure_self_test(args.alpha, args.eps, args.omega, dim, rng)
    print("closure identity self-test passed", flush=True)

    equivalence_rows = []
    build_rows = []
    quadcheck_rows = []
    for case in cases:
        case_equivalence, case_builds, case_quadcheck = run_case(
            case=case,
            mode=args.mode,
            alpha=args.alpha,
            eps=args.eps,
            omega=args.omega,
            root_extent=args.root_extent,
            n_jobs=max(1, args.jobs),
            with_quadcheck=(args.mode == "full"),
        )
        equivalence_rows.extend(case_equivalence)
        build_rows.extend(case_builds)
        quadcheck_rows.extend(case_quadcheck)

    _write_csv(
        args.out_dir / "closure_equivalence.csv", EQUIVALENCE_FIELDS, equivalence_rows
    )
    _write_csv(args.out_dir / "closure_builds.csv", BUILD_FIELDS, build_rows)
    _write_csv(args.out_dir / "closure_quadcheck.csv", QUADCHECK_FIELDS, quadcheck_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
