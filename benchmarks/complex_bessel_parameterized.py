#!/usr/bin/env python3
"""Parameterized complex-RKE benchmark (imaginary-order modified Bessel kernel).

The model radial convolution kernel is ``g(r) = K_{i nu}(lambda r)`` in 2D.
From ``K_mu = pi / (2 sin(pi mu)) (I_{-mu} - I_mu)`` with ``mu = i nu``, the
small-argument expansion is real and carries the complex Euler exponents
``2m +- i nu``:

    K_{i nu}(z) = C_nu sum_m (z/2)^(2m) (rho_m / m!) sin(phi_m - nu log(z/2)),

with ``C_nu = pi / sinh(pi nu)``, ``rho_m = 1 / |Gamma(m+1+i nu)|`` and
``phi_m = arg Gamma(m+1+i nu)``.  The fixed channel family is

    { r^(2m) cos(nu log r),  r^(2m) sin(nu log r) },  m = 0 .. p-1,

tabulated once on the canonical level-0 box.  For a source box of extent ``h``
(canonical extent 1) and screening parameter ``lambda``, the online
recombination coefficients are pure functions of ``theta = lambda h``:

    A_m =  h^d C_nu (theta/2)^(2m) (rho_m/m!) sin(phi_m - nu log(theta/2))
    B_m = -h^d C_nu (theta/2)^(2m) (rho_m/m!) cos(phi_m - nu log(theta/2)),

i.e. the level and parameter dependence enter only through the scalar
``theta`` and rotations by ``nu log``.  Direct fixed-``lambda`` reference
tables are built from the converged series evaluation of ``K_{i nu}``, which
is validated against ``mpmath.besselk(1j*nu, z)`` at startup.  The reported
path mismatch at retained order ``p`` is the channel-truncation residual; the
documented tail bound sums the neglected ``I_{+-i nu}`` series terms at the
largest radius of the near-field case region (``3 sqrt(d) h``).

The experiment is table-level only; no far-field FMM for this kernel is
claimed.  Smoke mode is for CI/local validation; full mode is for paper
artifact generation on a controlled remote compute host.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from complex_channel_closure import (
    _build_row,
    _full_table_data,
    _mismatch_stats,
    _write_csv,
    build_channel_table,
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
    "warm_payload_load_ms",
    "serialized_payload_bytes",
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

MISMATCH_FIELDS = (
    "case_id",
    "mode",
    "dim",
    "q_order",
    "nu",
    "lam",
    "source_box_level",
    "source_box_extent",
    "theta",
    "retained_order_p",
    "z_max",
    "tail_bound_abs",
    "tail_bound_rel",
    "finite_entry_count",
    "max_abs_mismatch",
    "max_rel_mismatch",
    "reference_linf",
)

VALIDATION_FIELDS = (
    "mode",
    "nu",
    "n_samples",
    "z_min",
    "z_max",
    "max_rel_error_vs_mpmath",
    "pointwise_expansion_max_rel_error",
)


@dataclass(frozen=True)
class BesselCase:
    case_id: str
    dim: int
    q_order: int
    max_level: int
    regular_quad_order: int
    radial_quad_order: int
    lambdas: tuple[float, ...]
    retained_orders: tuple[int, ...]
    max_theta: float


SMOKE_CASES = (
    BesselCase(
        case_id="besselinu2d-q2-l0-1",
        dim=2,
        q_order=2,
        max_level=1,
        regular_quad_order=6,
        radial_quad_order=21,
        lambdas=(1.0,),
        retained_orders=(1, 2),
        max_theta=2.0,
    ),
)

# Same angular/radial orders as the log-periodic closure full case: the
# oscillatory log factor sets the quadrature burden, and the r^(2m) channel
# prefactors are bounded (no singularity).
FULL_CASES = (
    BesselCase(
        case_id="besselinu2d-q4-l0-3",
        dim=2,
        q_order=4,
        max_level=3,
        regular_quad_order=64,
        radial_quad_order=61,
        lambdas=(0.5, 1.0, 2.0, 4.0),
        retained_orders=(1, 2, 3, 4),
        max_theta=2.0,
    ),
)


# {{{ series evaluation of K_{i nu}

class BesselINuSeries:
    """Converged small-argument series for ``K_{i nu}(z)``, real ``z > 0``.

    Term magnitudes decay like ``(z^2/4)^m / (m!)^2``; the alternating-phase
    cancellation loses roughly ``0.43 z`` decimal digits, so the evaluation
    stays near machine precision for the ``z <~ 8.5`` range used here
    (``theta <= 2`` over the near-field case region).
    """

    def __init__(self, nu, max_terms=80):
        from scipy.special import loggamma

        self.nu = float(nu)
        self.prefactor = math.pi / math.sinh(math.pi * self.nu)
        orders = np.arange(max_terms)
        lg = loggamma(orders + 1.0 + 1j * self.nu)
        log_m_factorial = np.cumsum(
            np.concatenate([[0.0], np.log(np.arange(1, max_terms))])
        )
        # c_m = rho_m / m! with rho_m = 1 / |Gamma(m+1+i nu)|
        self.coefficients = np.exp(-lg.real - log_m_factorial)
        self.phases = lg.imag

    def series_coefficients(self, half_z):
        """Coefficients of the (cos, sin)(nu log r-like) channel pair.

        Returns arrays ``(a_m, b_m)`` such that
        ``K_{i nu}(z) = sum_m (a_m cos(nu log(z/2)') + ...)`` -- concretely,
        with ``s = half_z``,

            K = sum_m C (s^2)^m c_m sin(phi_m - nu log s).
        """
        log_s = math.log(half_z)
        angle = self.phases - self.nu * log_s
        weight = self.prefactor * self.coefficients * np.exp(
            2.0 * np.arange(len(self.coefficients)) * log_s
        )
        return weight * np.sin(angle)

    def __call__(self, z):
        if z <= 0.0:
            return math.inf
        return float(np.sum(self.series_coefficients(0.5 * z)))

    def tail_bound(self, z, p):
        """Bound on ``|sum_{m >= p}|`` at argument ``z`` (|sin| <= 1)."""
        log_s = math.log(0.5 * z)
        orders = np.arange(p, len(self.coefficients))
        return float(
            self.prefactor
            * np.sum(self.coefficients[p:] * np.exp(2.0 * orders * log_s))
        )


class BesselINuKernel:
    """Direct kernel ``g(x) = K_{i nu}(lambda |x|)``."""

    def __init__(self, series, lam):
        self.series = series
        self.lam = float(lam)

    def __call__(self, *coords):
        r = math.sqrt(sum(c * c for c in coords))
        return self.series(self.lam * r)


class PowerCosLogChannel:
    """channel: r^(2m) cos(nu log r)."""

    def __init__(self, order, nu):
        self.order = int(order)
        self.nu = float(nu)

    def __call__(self, *coords):
        r = math.sqrt(sum(c * c for c in coords))
        if r == 0.0:
            return math.inf
        return r ** (2 * self.order) * math.cos(self.nu * math.log(r))


class PowerSinLogChannel:
    """channel: r^(2m) sin(nu log r)."""

    def __init__(self, order, nu):
        self.order = int(order)
        self.nu = float(nu)

    def __call__(self, *coords):
        r = math.sqrt(sum(c * c for c in coords))
        if r == 0.0:
            return math.inf
        return r ** (2 * self.order) * math.sin(self.nu * math.log(r))


def recombination_coefficients(series, theta, h, dim, p):
    """Online coefficients (A_m, B_m) for the canonical channel pair m < p.

    Pure functions of ``theta = lambda h`` (rotations by ``nu log``), times
    the volume-element factor ``h^d``.
    """
    log_s = math.log(0.5 * theta)
    orders = np.arange(p)
    weight = (
        (h**dim)
        * series.prefactor
        * series.coefficients[:p]
        * np.exp(2.0 * orders * log_s)
    )
    angle = series.phases[:p] - series.nu * log_s
    return weight * np.sin(angle), -weight * np.cos(angle)


def validate_series(series, z_max, n_samples=120):
    """Max relative error of the series against mpmath.besselk(1j nu, z)."""
    import mpmath

    mpmath.mp.dps = 40
    z_values = np.geomspace(1e-8, z_max, n_samples)
    max_rel = 0.0
    for z in z_values:
        reference = complex(mpmath.besselk(1j * series.nu, mpmath.mpf(z)))
        if abs(reference.imag) > 1e-25 * max(abs(reference.real), 1e-300):
            raise RuntimeError(
                f"mpmath K_(i nu)({z}) has non-negligible imaginary part"
            )
        rel = abs(series(z) - reference.real) / max(abs(reference.real), 1e-300)
        max_rel = max(max_rel, rel)
    return float(max_rel), float(z_values[0]), float(z_values[-1])


def expansion_self_test(series, dim, rng, p=40):
    """Pointwise check of the channel expansion + theta-recombination."""
    max_rel = 0.0
    for _ in range(200):
        coords = rng.uniform(0.05, 1.0, size=dim)
        r_hat = math.sqrt(float(np.sum(coords**2)))
        h = 2.0 ** (-int(rng.integers(0, 5)))
        lam = float(rng.uniform(0.3, 2.0))
        theta = lam * h
        direct = series(lam * h * r_hat)
        a, b = recombination_coefficients(series, theta, h, dim, p)
        log_r = math.log(r_hat)
        recombined = float(
            np.sum(
                a
                * np.exp(2.0 * np.arange(p) * log_r)
                * math.cos(series.nu * log_r)
                + b
                * np.exp(2.0 * np.arange(p) * log_r)
                * math.sin(series.nu * log_r)
            )
        ) / (h**dim)
        rel = abs(direct - recombined) / max(abs(direct), 1e-300)
        max_rel = max(max_rel, rel)
    if max_rel > 1e-11:
        raise RuntimeError(
            f"channel expansion self-test failed: max rel={max_rel:.3e}"
        )
    return max_rel

# }}}


def run_case(
    *,
    case: BesselCase,
    mode: str,
    nu: float,
    series: BesselINuSeries,
    n_jobs: int,
):
    radial_rule = "tanh-sinh-fast"
    mismatch_rows = []
    build_rows = []
    p_max = max(case.retained_orders)

    channel_data = {}
    for m in range(p_max):
        for tag, kernel_func in (
            (f"channel-cos-m{m}", PowerCosLogChannel(m, nu)),
            (f"channel-sin-m{m}", PowerSinLogChannel(m, nu)),
        ):
            table, build_wall_s = build_channel_table(
                kernel_func=kernel_func,
                dim=case.dim,
                q_order=case.q_order,
                level=0,
                root_extent=1.0,
                regular_quad_order=case.regular_quad_order,
                radial_quad_order=case.radial_quad_order,
                radial_rule=radial_rule,
                n_jobs=n_jobs,
            )
            channel_data[tag] = _full_table_data(table)
            build_rows.append(
                _build_row(
                    case=case,
                    mode=mode,
                    table_id=tag,
                    table=table,
                    build_wall_s=build_wall_s,
                    n_jobs=n_jobs,
                    radial_rule=radial_rule,
                )
            )
            print(
                f"[{case.case_id}] built {tag} ({build_wall_s:.1f} s)",
                flush=True,
            )

    for lam in case.lambdas:
        for level in range(case.max_level + 1):
            h = 2.0 ** (-level)
            theta = lam * h
            if theta > case.max_theta:
                continue
            direct_table, build_wall_s = build_channel_table(
                kernel_func=BesselINuKernel(series, lam),
                dim=case.dim,
                q_order=case.q_order,
                level=level,
                root_extent=1.0,
                regular_quad_order=case.regular_quad_order,
                radial_quad_order=case.radial_quad_order,
                radial_rule=radial_rule,
                n_jobs=n_jobs,
            )
            build_rows.append(
                _build_row(
                    case=case,
                    mode=mode,
                    table_id=f"direct-lam{lam:g}-l{level}",
                    table=direct_table,
                    build_wall_s=build_wall_s,
                    n_jobs=n_jobs,
                    radial_rule=radial_rule,
                )
            )
            direct_data = _full_table_data(direct_table)
            print(
                f"[{case.case_id}] built direct lam={lam:g} l{level} "
                f"({build_wall_s:.1f} s)",
                flush=True,
            )

            # largest source-target distance over the List 1 case region
            r_max = 3.0 * math.sqrt(case.dim) * h
            z_max = lam * r_max

            for p in case.retained_orders:
                a, b = recombination_coefficients(
                    series, theta, h, case.dim, p
                )
                recombined = np.zeros_like(direct_data)
                for m in range(p):
                    recombined += a[m] * channel_data[f"channel-cos-m{m}"]
                    recombined += b[m] * channel_data[f"channel-sin-m{m}"]

                count, max_abs, max_rel, reference_linf = _mismatch_stats(
                    direct_data, recombined
                )
                # table entries integrate the kernel against unit-bounded
                # basis modes over a box of volume h^d, so the pointwise
                # tail bound times h^d bounds the per-entry tail
                tail_abs = series.tail_bound(z_max, p) * h**case.dim
                mismatch_rows.append(
                    {
                        "case_id": case.case_id,
                        "mode": mode,
                        "dim": case.dim,
                        "q_order": case.q_order,
                        "nu": nu,
                        "lam": lam,
                        "source_box_level": level,
                        "source_box_extent": h,
                        "theta": theta,
                        "retained_order_p": p,
                        "z_max": z_max,
                        "tail_bound_abs": tail_abs,
                        "tail_bound_rel": tail_abs
                        / max(reference_linf, 1e-300),
                        "finite_entry_count": count,
                        "max_abs_mismatch": max_abs,
                        "max_rel_mismatch": max_rel,
                        "reference_linf": reference_linf,
                    }
                )
                print(
                    f"[{case.case_id}] lam={lam:g} l{level} p={p}: "
                    f"max_rel_mismatch={max_rel:.3e} "
                    f"tail_bound_rel={mismatch_rows[-1]['tail_bound_rel']:.3e}",
                    flush=True,
                )

    return mismatch_rows, build_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/benchmarks/complex-bessel-parameterized"),
    )
    parser.add_argument("--nu", type=float, default=4.0)
    parser.add_argument(
        "--jobs", type=int, default=1, help="parallel workers for table entries"
    )
    args = parser.parse_args()

    if args.nu <= 0.0:
        raise SystemExit("nu must be positive")

    cases = SMOKE_CASES if args.mode == "smoke" else FULL_CASES
    series = BesselINuSeries(args.nu)

    z_max_global = max(
        3.0 * math.sqrt(case.dim) * case.max_theta for case in cases
    )
    max_rel_mpmath, z_lo, z_hi = validate_series(series, z_max_global)
    print(
        f"series validation vs mpmath: max rel error {max_rel_mpmath:.3e} "
        f"over z in [{z_lo:.1e}, {z_hi:.1e}]",
        flush=True,
    )
    if max_rel_mpmath > 1e-10:
        raise SystemExit("series evaluation failed mpmath validation")

    rng = np.random.default_rng(20260726)
    expansion_rel = max(
        expansion_self_test(series, case.dim, rng) for case in cases
    )
    print(
        f"channel expansion self-test passed (max rel {expansion_rel:.3e})",
        flush=True,
    )

    validation_rows = [
        {
            "mode": args.mode,
            "nu": args.nu,
            "n_samples": 120,
            "z_min": z_lo,
            "z_max": z_hi,
            "max_rel_error_vs_mpmath": max_rel_mpmath,
            "pointwise_expansion_max_rel_error": expansion_rel,
        }
    ]

    mismatch_rows = []
    build_rows = []
    for case in cases:
        case_mismatch, case_builds = run_case(
            case=case,
            mode=args.mode,
            nu=args.nu,
            series=series,
            n_jobs=max(1, args.jobs),
        )
        mismatch_rows.extend(case_mismatch)
        build_rows.extend(case_builds)

    _write_csv(
        args.out_dir / "bessel_mismatch.csv", MISMATCH_FIELDS, mismatch_rows
    )
    _write_csv(args.out_dir / "bessel_builds.csv", BUILD_FIELDS, build_rows)
    _write_csv(
        args.out_dir / "bessel_validation.csv",
        VALIDATION_FIELDS,
        validation_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
