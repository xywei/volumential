"""Experiment A checks 1, 2 and 5: windowed-channel identity, adjacent-window
shell, and the Lemma E.1 germ constants.

Check 1 compares Volumential's ``windowed_channel_profile(3, 0, t_w)`` with the
Ewald short-range kernel ``erfc(r / (2 sqrt(t_w))) / r`` and with a 40-digit
mpmath reference.

Check 2 compares the adjacent-window difference ``chi_0(.; t) - chi_0(.; t/4)``
with the DMK difference kernel ``(erf(r/sigma_{l+1}) - erf(r/sigma_l)) / r``
under ``sigma_l = 2 sqrt(t_l)``, and probes smoothness at the origin.

Check 5 verifies the Lemma E.1 singular-germ constants for ``m <= 5`` in 2D and
3D by showing that ``chi_m - c_m psi_m`` coincides with an explicit power series
in ``r^2`` (the appendix's completion ``A_m``), together with the closed form
``Gamma(1/2 - m) = (-4)^m m! sqrt(pi) / (2m)!`` used in the 3D proof.

Exploratory, not benchmark-grade.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from experiment_a_common import (
    ROOT_EXTENT,
    WINDOW_THETA,
    environment_summary,
    get_pyplot,
    output_dir,
    relative_deviation,
    write_csv,
    write_json,
)

MPMATH_DPS = 40
GERM_DPS = 60


def _load_mpmath():
    """Return the mpmath module configured for this experiment, or None."""
    try:
        import mpmath
    except ImportError:
        return None
    mpmath.mp.dps = MPMATH_DPS
    return mpmath


def _volumential_profile(dim, m, window_scale):
    """Volumential's windowed channel profile, or None if unimportable."""
    try:
        from volumential.rke_table_assembly import windowed_channel_profile
    except Exception:  # noqa: BLE001 - any import-time failure is a fallback
        return None
    return windowed_channel_profile(dim, m, window_scale)


def _volumential_location() -> str | None:
    """Path-free provenance marker for the imported Volumential module."""
    try:
        import volumential.rke_table_assembly  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    return "imported (synced worktree; path withheld by campaign policy)"


# {{{ check 1


def check1_chi0_is_ewald(t_w: float, mpmath) -> tuple[list[dict], dict]:
    """chi_0 in 3D against erfc(r / (2 sqrt(t_w))) / r."""
    import scipy.special as sps

    ratios = np.logspace(-3.0, np.log10(20.0), 241)
    radius = ratios * math.sqrt(t_w)

    closed = sps.erfc(radius / (2.0 * math.sqrt(t_w))) / radius
    sigma = 2.0 * math.sqrt(t_w)
    dmk = sps.erfc(radius / sigma) / radius

    profile = _volumential_profile(3, 0, t_w)
    volumential = profile(radius) if profile is not None else None

    if mpmath is not None:
        reference = np.array(
            [
                float(
                    mpmath.erfc(mpmath.mpf(float(r)) / (2 * mpmath.sqrt(t_w)))
                    / mpmath.mpf(float(r))
                )
                for r in radius
            ]
        )
    else:
        reference = closed

    rows = []
    for index, r in enumerate(radius):
        row = {
            "r_over_sqrt_tw": float(ratios[index]),
            "r": float(r),
            "chi0_closed_form": float(closed[index]),
            "chi0_dmk_sigma_form": float(dmk[index]),
            "chi0_mpmath_40digit": float(reference[index]),
            "rel_dev_closed_vs_mpmath": float(
                relative_deviation(closed[index], reference[index])
            ),
            "rel_dev_dmk_vs_mpmath": float(
                relative_deviation(dmk[index], reference[index])
            ),
        }
        if volumential is not None:
            row["chi0_volumential"] = float(volumential[index])
            row["rel_dev_volumential_vs_mpmath"] = float(
                relative_deviation(volumential[index], reference[index])
            )
        else:
            row["chi0_volumential"] = ""
            row["rel_dev_volumential_vs_mpmath"] = ""
        rows.append(row)

    summary = {
        "n_samples": int(radius.size),
        "r_over_sqrt_tw_min": float(ratios[0]),
        "r_over_sqrt_tw_max": float(ratios[-1]),
        "window_scale_t_w": float(t_w),
        "sigma_from_t_w": float(sigma),
        "max_rel_dev_closed_vs_mpmath": float(
            np.max(relative_deviation(closed, reference))
        ),
        "max_rel_dev_dmk_sigma_vs_mpmath": float(
            np.max(relative_deviation(dmk, reference))
        ),
        "dmk_equals_closed_exactly": bool(np.array_equal(dmk, closed)),
        "mpmath_available": mpmath is not None,
    }
    if volumential is not None:
        summary["max_rel_dev_volumential_vs_mpmath"] = float(
            np.max(relative_deviation(volumential, reference))
        )
        summary["max_rel_dev_volumential_vs_closed"] = float(
            np.max(relative_deviation(volumential, closed))
        )
        summary["volumential_available"] = True
    else:
        summary["volumential_available"] = False
    return rows, summary


# }}}


# {{{ check 2


def check2_adjacent_window_shell(t_w: float, mpmath) -> tuple[list[dict], dict]:
    """Adjacent-window difference against the DMK shell kernel."""
    import scipy.special as sps

    t_coarse = t_w
    t_fine = t_w / 4.0
    sigma_coarse = 2.0 * math.sqrt(t_coarse)
    sigma_fine = 2.0 * math.sqrt(t_fine)

    ratios = np.logspace(-4.0, np.log10(8.0), 241)
    radius = ratios * sigma_coarse

    chi_coarse = sps.erfc(radius / sigma_coarse) / radius
    chi_fine = sps.erfc(radius / sigma_fine) / radius
    shell = chi_coarse - chi_fine
    dmk = (sps.erf(radius / sigma_fine) - sps.erf(radius / sigma_coarse)) / radius

    profile_coarse = _volumential_profile(3, 0, t_coarse)
    profile_fine = _volumential_profile(3, 0, t_fine)
    if profile_coarse is not None and profile_fine is not None:
        shell_vol = profile_coarse(radius) - profile_fine(radius)
    else:
        shell_vol = None

    scaled_profile = (
        sps.erf(2.0 * ratios) - sps.erf(ratios)
    ) / ratios / sigma_coarse

    rows = []
    for index, r in enumerate(radius):
        row = {
            "r_over_sigma_l": float(ratios[index]),
            "r": float(r),
            "shell_from_windows": float(shell[index]),
            "shell_dmk_erf_form": float(dmk[index]),
            "shell_scaled_profile": float(scaled_profile[index]),
            "rel_dev_windows_vs_dmk": float(
                relative_deviation(shell[index], dmk[index])
            ),
            "rel_dev_scaled_profile_vs_dmk": float(
                relative_deviation(scaled_profile[index], dmk[index])
            ),
        }
        if shell_vol is not None:
            row["shell_volumential"] = float(shell_vol[index])
            row["rel_dev_volumential_vs_dmk"] = float(
                relative_deviation(shell_vol[index], dmk[index])
            )
        else:
            row["shell_volumential"] = ""
            row["rel_dev_volumential_vs_dmk"] = ""
        rows.append(row)

    # The two float64 forms are algebraically identical but not numerically
    # interchangeable: ``erf(r/sigma_fine) - erf(r/sigma_coarse)`` is a
    # difference of two numbers that both round to 1 once the shell has
    # decayed, so the DMK erf form loses all relative accuracy and finally
    # returns exactly zero, while the erfc (window) form stays accurate.
    # Reporting a raw relative deviation against the erf form therefore says
    # nothing about the identity past that crossover; the mpmath reference
    # below decides which side is wrong.
    erf_form_zero = np.flatnonzero((dmk == 0.0) & (shell != 0.0))
    dmk_resolved = np.abs(dmk) > 0.0

    summary = {
        "sigma_l": float(sigma_coarse),
        "sigma_l_plus_1": float(sigma_fine),
        "sigma_ratio": float(sigma_coarse / sigma_fine),
        "max_rel_dev_windows_vs_dmk_all_samples": float(
            relative_deviation(shell, dmk).max()
        ),
        "max_rel_dev_scaled_profile_vs_dmk": float(
            relative_deviation(scaled_profile, dmk).max()
        ),
        "analytic_origin_value": float(2.0 / (math.sqrt(math.pi) * sigma_coarse)),
        "erf_form_first_exact_zero_r_over_sigma_l": (
            float(ratios[erf_form_zero[0]]) if erf_form_zero.size else None
        ),
        "shell_value_where_erf_form_first_vanishes": (
            float(shell[erf_form_zero[0]]) if erf_form_zero.size else None
        ),
    }
    if shell_vol is not None:
        summary["max_rel_dev_volumential_vs_dmk_all_samples"] = float(
            relative_deviation(shell_vol, dmk).max()
        )

    if mpmath is not None:
        mpmath.mp.dps = GERM_DPS

        # Cancellation-free high-precision reference: the erfc difference.
        def shell_reference(r):
            r = mpmath.mpf(float(r))
            return (
                mpmath.erfc(r / mpmath.mpf(sigma_coarse))
                - mpmath.erfc(r / mpmath.mpf(sigma_fine))
            ) / r

        reference = [shell_reference(r) for r in radius]
        worst_windows = 0.0
        worst_dmk = 0.0
        worst_vol = 0.0
        for index, exact in enumerate(reference):
            if exact == 0:
                rows[index]["shell_mpmath_reference"] = 0.0
                rows[index]["rel_dev_windows_vs_mpmath"] = ""
                rows[index]["rel_dev_dmk_erf_form_vs_mpmath"] = ""
                continue
            dev_windows = float(
                abs(mpmath.mpf(float(shell[index])) - exact) / abs(exact)
            )
            dev_dmk = float(abs(mpmath.mpf(float(dmk[index])) - exact) / abs(exact))
            rows[index]["shell_mpmath_reference"] = float(exact)
            rows[index]["rel_dev_windows_vs_mpmath"] = dev_windows
            rows[index]["rel_dev_dmk_erf_form_vs_mpmath"] = dev_dmk
            worst_windows = max(worst_windows, dev_windows)
            worst_dmk = max(worst_dmk, dev_dmk)
            if shell_vol is not None:
                dev_vol = float(
                    abs(mpmath.mpf(float(shell_vol[index])) - exact) / abs(exact)
                )
                rows[index]["rel_dev_volumential_vs_mpmath"] = dev_vol
                worst_vol = max(worst_vol, dev_vol)
        summary["max_rel_dev_windows_erfc_form_vs_mpmath"] = worst_windows
        summary["max_rel_dev_dmk_erf_form_vs_mpmath"] = worst_dmk
        if shell_vol is not None:
            summary["max_rel_dev_volumential_vs_mpmath"] = worst_vol
        # Restricted to the range where the erf form still resolves the shell
        # at all, the two float64 forms agree to roundoff.
        resolved = dmk_resolved & (ratios <= 2.0)
        summary["max_rel_dev_windows_vs_dmk_r_over_sigma_l_le_2"] = float(
            relative_deviation(shell[resolved], dmk[resolved]).max()
        )

        def shell_fn(r):
            r = mpmath.mpf(r)
            return (
                mpmath.erf(r / mpmath.mpf(sigma_fine))
                - mpmath.erf(r / mpmath.mpf(sigma_coarse))
            ) / r

        # Smoothness at the origin is settled analytically rather than by
        # numerical differentiation: D_l is the even entire function
        #   D_l(r) = (2/sqrt(pi)) sum_k (-1)^k r^{2k} / (k! (2k+1))
        #            * (sigma_fine^{-(2k+1)} - sigma_coarse^{-(2k+1)}),
        # so every odd derivative vanishes at r = 0 by parity.  The check is
        # that this series reproduces D_l to high precision on the whole
        # comparison range, and that its k = 0 term is the origin value.
        origin = (2.0 / math.sqrt(math.pi)) * (1.0 / sigma_fine - 1.0 / sigma_coarse)
        summary["analytic_origin_value"] = float(origin)
        summary["mpmath_origin_limit"] = float(shell_fn(mpmath.mpf("1e-30")))
        summary["origin_limit_rel_dev"] = float(
            abs(summary["mpmath_origin_limit"] - origin) / abs(origin)
        )

        # The even entire series is an alternating series whose largest term
        # grows like exp((r/sigma_fine)^2) while the sum stays bounded by the
        # shell itself, so a fixed term count and a fixed precision both fail
        # at large r for conditioning reasons, not because the identity fails.
        # Terms are therefore added until they stop contributing, the working
        # precision is raised to absorb the cancellation, and the cancellation
        # ratio (largest term over the sum) is reported alongside the deviation.
        series_dps = 200
        max_terms = 600

        def even_series(r):
            r = mpmath.mpf(r)
            total = mpmath.mpf(0)
            largest = mpmath.mpf(0)
            for k in range(max_terms):
                term = (
                    (-1) ** k
                    * r ** (2 * k)
                    / (mpmath.factorial(k) * (2 * k + 1))
                    * (
                        mpmath.mpf(sigma_fine) ** (-(2 * k + 1))
                        - mpmath.mpf(sigma_coarse) ** (-(2 * k + 1))
                    )
                )
                total += term
                largest = max(largest, abs(term))
                if k > 8 and abs(term) < mpmath.mpf(10) ** (-(series_dps - 20)) * (
                    largest if largest > 0 else 1
                ):
                    break
            return 2 / mpmath.sqrt(mpmath.pi) * total, largest, k + 1

        previous_dps = mpmath.mp.dps
        mpmath.mp.dps = series_dps
        worst_series = 0.0
        worst_cancellation = 0.0
        max_terms_used = 0
        series_range = radius[radius < 3.0 * sigma_coarse]
        for r in series_range:
            exact = shell_reference(float(r))
            approx, largest, used = even_series(float(r))
            worst_series = max(worst_series, float(abs(approx - exact) / abs(exact)))
            if exact != 0:
                worst_cancellation = max(
                    worst_cancellation, float(largest / abs(exact))
                )
            max_terms_used = max(max_terms_used, used)
        mpmath.mp.dps = previous_dps
        summary["max_rel_dev_even_series_vs_shell"] = worst_series
        summary["even_series_max_r_over_sigma_l"] = float(
            series_range.max() / sigma_coarse
        )
        summary["even_series_max_terms_used"] = max_terms_used
        summary["even_series_working_dps"] = series_dps
        summary["even_series_max_cancellation_ratio"] = worst_cancellation
        summary["even_series_digits_lost"] = (
            math.log10(worst_cancellation) if worst_cancellation > 1.0 else 0.0
        )
        summary["odd_derivatives_vanish_by_parity"] = True
        mpmath.mp.dps = MPMATH_DPS

    return rows, summary


# }}}


# {{{ check 5


def _chi_m_mp(mpmath, dim: int, m: int, t: float, r):
    """chi_m(r; t) in high precision from the incomplete-gamma closed form."""
    r = mpmath.mpf(r)
    x = r * r / (4 * mpmath.mpf(t))
    if dim == 2:
        return mpmath.mpf(1) / 2 * mpmath.mpf(t) ** m * mpmath.expint(m + 1, x)
    half = mpmath.mpf(1) / 2
    return (
        (r * r / 4) ** (mpmath.mpf(m) - half)
        * mpmath.gammainc(half - m, x, mpmath.inf)
        / (2 * mpmath.sqrt(mpmath.pi))
    )


def _germ_constant(dim: int, m: int) -> float:
    """Lemma E.1 germ constant c_m pairing chi_m with psi_m."""
    if dim == 2:
        return (-1.0) ** (m + 1) / (4.0**m * math.factorial(m))
    return (-1.0) ** m * math.factorial(m) / math.factorial(2 * m)


def _completion_series_mp(mpmath, dim: int, m: int, t: float, r, n_terms: int):
    """The appendix's completion A_m, evaluated from its explicit power
    series in r^2 (entire by construction)."""
    r = mpmath.mpf(r)
    t = mpmath.mpf(t)
    x = r * r / (4 * t)
    if dim == 2:
        prefactor = (
            mpmath.mpf(1) / 2 * (r * r / 4) ** m * (-1) ** m / mpmath.factorial(m)
        )
        bracket = -mpmath.euler + mpmath.log(4 * t)
        for k in range(1, n_terms + 1):
            bracket += (-1) ** (k + 1) * x**k / (k * mpmath.factorial(k))
        tail = mpmath.mpf(0)
        for j in range(m):
            tail += (-1) ** j * mpmath.factorial(j) * t ** (j + 1) * (
                r * r / 4
            ) ** (m - j - 1)
        return prefactor * bracket - (
            mpmath.mpf(1) / 2 * (-1) ** m / mpmath.factorial(m)
        ) * mpmath.e ** (-x) * tail
    half = mpmath.mpf(1) / 2
    total = mpmath.mpf(0)
    for k in range(n_terms + 1):
        total += (
            (-1) ** k
            * t ** (mpmath.mpf(m) - half - k)
            * (r * r / 4) ** k
            / (mpmath.factorial(k) * (half - m + k))
        )
    return -total / (2 * mpmath.sqrt(mpmath.pi))


def check5_germ_constants(t_loc: float, mpmath) -> tuple[list[dict], dict]:
    """Lemma E.1 germ constants for m <= 5 in both dimensions."""
    if mpmath is None:
        return [], {"status": "not-run", "reason": "mpmath unavailable"}

    mpmath.mp.dps = GERM_DPS
    rows = []
    worst = {2: 0.0, 3: 0.0}
    worst_control = {2: 0.0, 3: 0.0}
    gamma_worst = 0.0

    radii = [0.002, 0.008, 0.02, 0.05]
    for dim in (2, 3):
        for m in range(6):
            constant = _germ_constant(dim, m)
            if dim == 3:
                closed = (
                    mpmath.mpf(-4) ** m
                    * mpmath.factorial(m)
                    * mpmath.sqrt(mpmath.pi)
                    / mpmath.factorial(2 * m)
                )
                direct = mpmath.gamma(mpmath.mpf(1) / 2 - m)
                gamma_dev = float(abs(closed - direct) / abs(direct))
                gamma_worst = max(gamma_worst, gamma_dev)
            else:
                gamma_dev = float("nan")

            for r in radii:
                chi = _chi_m_mp(mpmath, dim, m, t_loc, r)
                if dim == 2:
                    psi = mpmath.mpf(r) ** (2 * m) * mpmath.log(r)
                else:
                    psi = mpmath.mpf(r) ** (2 * m - 1)
                completion = chi - constant * psi
                series = _completion_series_mp(mpmath, dim, m, t_loc, r, 90)
                scale = max(abs(completion), abs(chi))
                deviation = float(abs(completion - series) / scale)
                perturbed = chi - constant * mpmath.mpf("1.000001") * psi
                control = float(abs(perturbed - series) / scale)
                worst[dim] = max(worst[dim], deviation)
                worst_control[dim] = max(worst_control[dim], control)
                rows.append(
                    {
                        "dim": dim,
                        "m": m,
                        "r": r,
                        "germ_constant": constant,
                        "chi_m": float(chi),
                        "completion_A_m": float(completion),
                        "completion_series": float(series),
                        "rel_dev_completion_vs_series": deviation,
                        "rel_dev_with_1e-6_perturbed_constant": control,
                        "rel_dev_gamma_half_minus_m_closed_form": gamma_dev,
                    }
                )
    mpmath.mp.dps = MPMATH_DPS
    summary = {
        "status": "run",
        "dps": GERM_DPS,
        "max_rel_dev_2d": worst[2],
        "max_rel_dev_3d": worst[3],
        "min_control_rel_dev_2d": worst_control[2],
        "min_control_rel_dev_3d": worst_control[3],
        "max_rel_dev_gamma_closed_form": gamma_worst,
        "t_loc": float(t_loc),
    }
    return rows, summary


# }}}


def _plot(out, check1_rows, check2_rows):
    """Write the two PNG plots if matplotlib is present."""
    plt = get_pyplot()
    if plt is None:
        return []
    written = []

    ratios = [row["r_over_sqrt_tw"] for row in check1_rows]
    deviations = [row["rel_dev_closed_vs_mpmath"] for row in check1_rows]
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.loglog(ratios, np.maximum(deviations, 1e-20), label="closed form")
    if check1_rows[0]["rel_dev_volumential_vs_mpmath"] != "":
        axis.loglog(
            ratios,
            np.maximum(
                [row["rel_dev_volumential_vs_mpmath"] for row in check1_rows],
                1e-20,
            ),
            label="Volumential chi_0",
        )
    axis.axhline(2.2e-16, color="grey", linestyle=":", label="double eps")
    axis.set_xlabel("r / sqrt(t_w)")
    axis.set_ylabel("relative deviation from 40-digit reference")
    axis.set_title("Check 1: chi_0 (3D) is the Ewald short-range kernel")
    axis.legend()
    figure.tight_layout()
    path = out / "experiment_a_check1_chi0_identity.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path.name)

    scaled = [row["r_over_sigma_l"] for row in check2_rows]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].plot(
        scaled, [row["shell_dmk_erf_form"] for row in check2_rows], color="C0"
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("r / sigma_l")
    axes[0].set_ylabel("shell kernel D_l(r)")
    axes[0].set_title("Adjacent-window slab")
    axes[1].loglog(
        scaled,
        np.maximum([row["rel_dev_windows_vs_dmk"] for row in check2_rows], 1e-20),
    )
    axes[1].axhline(2.2e-16, color="grey", linestyle=":")
    axes[1].set_xlabel("r / sigma_l")
    axes[1].set_ylabel("relative deviation")
    axes[1].set_title("Check 2: windows vs DMK erf form")
    figure.tight_layout()
    path = out / "experiment_a_check2_shell.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path.name)
    return written


def main() -> None:
    """Run checks 1, 2 and 5 and write CSV/JSON/PNG outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()
    out = output_dir(args.out)

    started = time.time()
    mpmath = _load_mpmath()
    box_extent = ROOT_EXTENT
    t_w = (box_extent / WINDOW_THETA) ** 2
    t_loc = WINDOW_THETA ** -2.0

    rows1, summary1 = check1_chi0_is_ewald(t_w, mpmath)
    write_csv(out / "experiment_a_check1_chi0.csv", list(rows1[0]), rows1)

    rows2, summary2 = check2_adjacent_window_shell(t_w, mpmath)
    write_csv(out / "experiment_a_check2_shell.csv", list(rows2[0]), rows2)

    rows5, summary5 = check5_germ_constants(t_loc, mpmath)
    if rows5:
        write_csv(out / "experiment_a_check5_germs.csv", list(rows5[0]), rows5)

    plots = _plot(out, rows1, rows2)

    payload = {
        "experiment": "A (identity and normalization), checks 1, 2, 5",
        "exploratory": True,
        "environment": environment_summary(),
        "volumential_module": _volumential_location(),
        "configuration": {
            "window_theta": WINDOW_THETA,
            "box_extent": box_extent,
            "t_w_physical": t_w,
            "t_loc_box_units": t_loc,
        },
        "check1_chi0_is_ewald": summary1,
        "check2_adjacent_window_shell": summary2,
        "check5_germ_constants": summary5,
        "plots": plots,
        "wall_seconds": time.time() - started,
    }
    write_json(out / "experiment_a_identity_summary.json", payload)
    print(json.dumps(payload, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
