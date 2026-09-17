"""Experiment A check 4: heat-time window versus Fourier-Gaussian multiplier
for the 3D Yukawa kernel (D3), with the stable closed form of D4.

Physical units, ``K_zeta(r) = exp(-mu r) / r``, ``zeta = mu**2``, window
``T = t_w = (h / Theta)**2``.  The heat-time short-range part is

    W_T(r) = (1 / (2 sqrt(pi))) int_0^T u^{-3/2} exp(-zeta u - r^2/(4u)) du
           = (1 / (2 r)) [ e^{-mu r} erfc(r/(2 sqrt T) - mu sqrt T)
                         + e^{+mu r} erfc(r/(2 sqrt T) + mu sqrt T) ].

The Critic addendum's stable evaluation is used throughout: with
``z = r/(2 sqrt T) - mu sqrt T`` and ``w = r/(2 sqrt T) + mu sqrt T``, both
``e^{-mu r} erfc(z)`` (for ``z >= 0``) and ``e^{mu r} erfc(w)`` equal
``exp(-r^2/(4T) - zeta T) erfcx(.)``; the naive form overflows once
``mu sqrt T = theta / Theta`` is large.

The Fourier-Gaussian multiplier ``e^{-T|k|^2} / (|k|^2 + zeta)`` equals
``e^{zeta T}`` times the heat-time long-range multiplier
``e^{-T(|k|^2 + zeta)} / (|k|^2 + zeta)``.  So (addendum correction to D3) the
two splits are the *same* decomposition up to one per-level scalar; the price
of the Fourier-Gaussian choice is that its short-range complement

    S_G = e^{zeta T} W_T - (e^{zeta T} - 1) K_zeta

carries a Yukawa-decaying, non-Gaussian tail of relative size
``e^{zeta T} - 1 = e^{(theta/Theta)^2} - 1``, whereas ``W_T`` is
Gaussian-localized for every ``zeta``.  This script tabulates both at
``r = h, 2h, 3h`` for ``theta/Theta`` in ``{0.25, 0.5, 1}`` (plus coarse-level
values where the naive form would overflow), and verifies the D4 closed form
against high-precision quadrature.

Needs SciPy; mpmath optional (used for the D4 verification and a 40-digit
reference).  Exploratory, not benchmark-grade.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from experiment_a_common import (
    WINDOW_THETA,
    environment_summary,
    get_pyplot,
    output_dir,
    write_csv,
    write_json,
)

MPMATH_DPS = 40
THETA_RATIOS = (0.25, 0.5, 1.0)
COARSE_RATIOS = (2.0, 4.0, 8.0)
AMPLIFICATION_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
RADII_OVER_H = (1.0, 2.0, 3.0)


def windowed_yukawa_stable(r: np.ndarray, mu: float, window: float) -> np.ndarray:
    """``W_T(r)``, the D4 closed form, evaluated without overflow."""
    import scipy.special as sps

    r = np.asarray(r, dtype=np.float64)
    root = math.sqrt(window)
    z = r / (2.0 * root) - mu * root
    w = r / (2.0 * root) + mu * root
    gaussian = np.exp(-(r * r) / (4.0 * window) - mu * mu * window)

    term_plus = gaussian * sps.erfcx(w)
    term_minus = np.where(
        z >= 0.0,
        gaussian * sps.erfcx(np.where(z >= 0.0, z, 0.0)),
        np.exp(-mu * r) * sps.erfc(np.where(z >= 0.0, 0.0, z)),
    )
    return (term_minus + term_plus) / (2.0 * r)


def windowed_yukawa_naive(r: np.ndarray, mu: float, window: float) -> np.ndarray:
    """The literal D4 expression, kept only to exhibit where it overflows."""
    import scipy.special as sps

    r = np.asarray(r, dtype=np.float64)
    root = math.sqrt(window)
    with np.errstate(over="ignore", invalid="ignore"):
        value = (
            np.exp(-mu * r) * sps.erfc(r / (2.0 * root) - mu * root)
            + np.exp(mu * r) * sps.erfc(r / (2.0 * root) + mu * root)
        ) / (2.0 * r)
    return value


def d4_reference_quadrature(mpmath, r: float, mu: float, window: float):
    """``W_T(r)`` by high-precision quadrature of the heat integral.

    The raw integrand on ``(0, T]`` is ``exp(-r^2/(4u))``-flat at the origin and
    concentrated at the upper endpoint.  Two substitutions fix that: ``u = T/s``
    gives ``T^{-1/2} int_1^inf s^{-1/2} exp(-zeta T / s - x s) ds`` with
    ``x = r^2/(4T)``, and then ``s = 1 + y/x`` rescales the decay to ``e^{-y}``
    independently of the parameters, leaving a smooth O(1) integrand that
    Gauss-Legendre panels resolve to full precision.
    """
    mpmath.mp.dps = MPMATH_DPS
    window = mpmath.mpf(window)
    x = mpmath.mpf(r) ** 2 / (4 * window)
    zeta_window = mpmath.mpf(mu) ** 2 * window

    def integrand(y):
        s = 1 + y / x
        return s ** mpmath.mpf(-0.5) * mpmath.e ** (-zeta_window / s - y)

    splits = [0, 1, 2, 5, 10, 20, 40, 80, 160, 320]
    value = mpmath.e ** (-x) * mpmath.quadgl(integrand, splits, maxdegree=10) / x
    return value / (mpmath.sqrt(window) * 2 * mpmath.sqrt(mpmath.pi))


def d4_reference_closed(mpmath, r: float, mu: float, window: float):
    """``W_T(r)`` from the D4 closed form evaluated at 40 digits."""
    mpmath.mp.dps = MPMATH_DPS
    r = mpmath.mpf(r)
    mu = mpmath.mpf(mu)
    root = mpmath.sqrt(mpmath.mpf(window))
    return (
        mpmath.e ** (-mu * r) * mpmath.erfc(r / (2 * root) - mu * root)
        + mpmath.e ** (mu * r) * mpmath.erfc(r / (2 * root) + mu * root)
    ) / (2 * r)


def run(box_size: float, theta_ratios, mpmath) -> tuple[list[dict], dict]:
    """Tabulate both splits over the requested ``theta / Theta`` ratios."""
    window = (box_size / WINDOW_THETA) ** 2
    rows = []
    worst_d4 = 0.0
    worst_d4_declared = 0.0
    worst_float = 0.0
    worst_float_declared = 0.0
    naive_overflow_ratio = None

    for ratio in theta_ratios:
        theta = ratio * WINDOW_THETA
        mu = theta / box_size
        zeta_window = (theta / WINDOW_THETA) ** 2
        amplification = math.exp(zeta_window) if zeta_window < 700.0 else math.inf
        radii = np.array([factor * box_size for factor in RADII_OVER_H])

        kernel = np.exp(-mu * radii) / radii
        short = windowed_yukawa_stable(radii, mu, window)
        naive = windowed_yukawa_naive(radii, mu, window)
        long_heat = kernel - short
        long_fourier = amplification * long_heat
        short_fourier = kernel - long_fourier
        tail = (amplification - 1.0) * kernel

        if not np.all(np.isfinite(naive)) and naive_overflow_ratio is None:
            naive_overflow_ratio = ratio

        for index, radius in enumerate(radii):
            if mpmath is not None:
                exact = d4_reference_quadrature(mpmath, float(radius), mu, window)
                closed = d4_reference_closed(mpmath, float(radius), mu, window)
                # compare in mpmath so that a reference below the double
                # underflow threshold (coarse levels) still has a meaning
                d4_dev = float(
                    abs(mpmath.mpf(float(short[index])) - exact) / abs(exact)
                )
                closed_dev = float(abs(closed - exact) / abs(exact))
                reference = float(exact)
                worst_d4 = max(worst_d4, closed_dev)
                worst_float = max(worst_float, d4_dev)
                if ratio <= 1.0:
                    worst_d4_declared = max(worst_d4_declared, closed_dev)
                    worst_float_declared = max(worst_float_declared, d4_dev)
            else:
                reference = float("nan")
                d4_dev = float("nan")
                closed_dev = float("nan")
            rows.append(
                {
                    "theta_over_Theta": ratio,
                    "theta": theta,
                    "mu": mu,
                    "zeta_times_T": zeta_window,
                    "exp_zetaT_minus_1": amplification - 1.0,
                    "r_over_h": RADII_OVER_H[index],
                    "r": float(radius),
                    "K_zeta": float(kernel[index]),
                    "W_T_heat_window_stable": float(short[index]),
                    "W_T_heat_window_naive": float(naive[index]),
                    "W_T_mpmath_reference": reference,
                    "rel_dev_D4_closed_form_vs_quadrature": closed_dev,
                    "rel_dev_float64_stable_vs_quadrature": d4_dev,
                    "long_range_heat": float(long_heat[index]),
                    "long_range_fourier_gaussian": float(long_fourier[index]),
                    "short_range_fourier_gaussian": float(short_fourier[index]),
                    "yukawa_tail_term": float(tail[index]),
                    "tail_over_W_T": float(abs(tail[index]) / abs(short[index]))
                    if short[index] != 0.0
                    else float("inf"),
                    "short_fourier_over_W_T": float(
                        abs(short_fourier[index]) / abs(short[index])
                    )
                    if short[index] != 0.0
                    else float("inf"),
                }
            )

    amplification_table = {
        f"{ratio:g}": {
            "zeta_T": ratio**2,
            "log10_exp_zetaT_minus_1": ratio**2 / math.log(10.0)
            if ratio**2 > 2.0
            else math.log10(math.expm1(ratio**2)),
        }
        for ratio in AMPLIFICATION_RATIOS
    }
    summary = {
        "box_size_h": box_size,
        "window_theta": WINDOW_THETA,
        "window_T": window,
        "sqrt_T_over_h": math.sqrt(window) / box_size,
        "max_rel_dev_D4_closed_form_vs_quadrature_declared_range": (
            worst_d4_declared if mpmath is not None else None
        ),
        "max_rel_dev_D4_closed_form_vs_quadrature_all_ratios": (
            worst_d4 if mpmath is not None else None
        ),
        "max_rel_dev_float64_stable_form_declared_range": (
            worst_float_declared if mpmath is not None else None
        ),
        "max_rel_dev_float64_stable_form_all_ratios": (
            worst_float if mpmath is not None else None
        ),
        "naive_form_first_overflow_theta_over_Theta": naive_overflow_ratio,
        "fourier_gaussian_amplification": amplification_table,
        "mpmath_available": mpmath is not None,
    }
    return rows, summary


def naive_breakdown(box_size: float, mpmath) -> tuple[list[dict], dict]:
    """Where the literal D4 expression stops being evaluable in float64.

    Scans ``theta / Theta`` well past the declaration.  The failure is not
    always a NaN: ``e^{mu r}`` stays finite up to ``mu r`` about 709 while
    ``erfc(w)`` has already underflowed to zero, so the naive form quietly
    drops its second term before it ever overflows.
    """
    window = (box_size / WINDOW_THETA) ** 2
    rows = []
    first_nonfinite = None
    first_wrong = None
    for ratio in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        mu = ratio * WINDOW_THETA / box_size
        radii = np.array([factor * box_size for factor in RADII_OVER_H])
        stable = windowed_yukawa_stable(radii, mu, window)
        naive = windowed_yukawa_naive(radii, mu, window)
        for index, radius in enumerate(radii):
            exact = (
                d4_reference_closed(mpmath, float(radius), mu, window)
                if mpmath is not None
                else None
            )
            stable_dev = (
                float(abs(mpmath.mpf(float(stable[index])) - exact) / abs(exact))
                if exact is not None and exact != 0
                else float("nan")
            )
            naive_value = float(naive[index])
            naive_dev = (
                float(abs(mpmath.mpf(naive_value) - exact) / abs(exact))
                if exact is not None and exact != 0 and math.isfinite(naive_value)
                else float("inf")
            )
            if not math.isfinite(naive_value) and first_nonfinite is None:
                first_nonfinite = ratio
            if naive_dev > 1.0e-12 and first_wrong is None:
                first_wrong = ratio
            rows.append(
                {
                    "theta_over_Theta": ratio,
                    "mu_sqrt_T": ratio,
                    "r_over_h": RADII_OVER_H[index],
                    "mu_r": float(mu * radius),
                    "W_T_exact_mpmath": float(exact) if exact is not None else "",
                    "W_T_stable_float64": float(stable[index]),
                    "W_T_naive_float64": naive_value,
                    "rel_dev_stable": stable_dev,
                    "rel_dev_naive": naive_dev,
                }
            )
    return rows, {
        "first_theta_over_Theta_with_nonfinite_naive_value": first_nonfinite,
        "first_theta_over_Theta_where_naive_loses_12_digits": first_wrong,
    }


def _plot(out, rows):
    """Gaussian versus Yukawa localization of the two short-range parts."""
    plt = get_pyplot()
    if plt is None:
        return []
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    ratios = sorted({row["theta_over_Theta"] for row in rows if row["theta_over_Theta"] <= 1.0})
    for ratio in ratios:
        selected = [row for row in rows if row["theta_over_Theta"] == ratio]
        selected.sort(key=lambda row: row["r_over_h"])
        axis.semilogy(
            [row["r_over_h"] for row in selected],
            [max(abs(row["W_T_heat_window_stable"]), 1e-300) for row in selected],
            marker="o",
            label=f"heat window, theta/Theta = {ratio}",
        )
        axis.semilogy(
            [row["r_over_h"] for row in selected],
            [max(abs(row["short_range_fourier_gaussian"]), 1e-300) for row in selected],
            marker="s",
            linestyle="--",
            label=f"Fourier-Gaussian, theta/Theta = {ratio}",
        )
    axis.set_xlabel("r / h")
    axis.set_ylabel("short-range part (absolute value)")
    axis.set_title("Check 4: Gaussian vs Yukawa-tailed short-range parts")
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = out / "experiment_a_check4_yukawa_window.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path.name]


def main() -> None:
    """Run check 4 and write CSV/JSON/PNG outputs."""
    parser = argparse.ArgumentParser(description="Experiment A check 4 (D3/D4)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--box-size", type=float, default=0.25)
    args = parser.parse_args()
    out = output_dir(args.out)

    started = time.time()
    try:
        import mpmath
    except ImportError:
        mpmath = None

    rows, summary = run(args.box_size, THETA_RATIOS + COARSE_RATIOS, mpmath)
    write_csv(out / "experiment_a_check4_yukawa.csv", list(rows[0]), rows)
    naive_rows, naive_summary = naive_breakdown(args.box_size, mpmath)
    write_csv(
        out / "experiment_a_check4_d4_stability.csv",
        list(naive_rows[0]),
        naive_rows,
    )
    summary["d4_float64_stability"] = naive_summary
    plots = _plot(out, rows)

    payload = {
        "experiment": "A (identity and normalization), check 4: D3/D4 Yukawa window",
        "exploratory": True,
        "environment": environment_summary(),
        "configuration": {
            "theta_over_Theta_declared_range": list(THETA_RATIOS),
            "theta_over_Theta_beyond_declaration": list(COARSE_RATIOS),
            "r_over_h": list(RADII_OVER_H),
        },
        "check4_yukawa_window": summary,
        "plots": plots,
        "wall_seconds": time.time() - started,
    }
    write_json(out / "experiment_a_check4_summary.json", payload)
    print(json.dumps(payload, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
