"""Render Experiment A plots from the CSVs the three check scripts write.

The scripts themselves skip plotting when matplotlib is not importable in the
interpreter that runs them (the Volumential environment used for check 3 has
SciPy but no matplotlib).  This script needs only ``numpy``, ``matplotlib`` and
the standard library, so it can be run afterwards with whatever interpreter on
the machine does have matplotlib:

    python experiments/windowed_dmk/experiment_a_plots.py --out OUT

Missing CSVs are skipped with a note; nothing here recomputes any number.

Exploratory, not benchmark-grade.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLOOR = 1.0e-18


def read_rows(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts, or an empty list if absent."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str, default: float = float("nan")) -> float:
    """Parse a CSV cell as a float, tolerating blanks."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def plot_check1(out: Path) -> list[str]:
    """Deviation of chi_0 from the 40-digit Ewald reference."""
    rows = read_rows(out / "experiment_a_check1_chi0.csv")
    if not rows:
        return []
    ratios = [as_float(row["r_over_sqrt_tw"]) for row in rows]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.loglog(
        ratios,
        [max(as_float(row["rel_dev_closed_vs_mpmath"]), FLOOR) for row in rows],
        label="erfc closed form",
    )
    volumential = [
        max(as_float(row.get("rel_dev_volumential_vs_mpmath", ""), FLOOR), FLOOR)
        for row in rows
    ]
    if any(value > FLOOR for value in volumential):
        axis.loglog(ratios, volumential, label="Volumential chi_0")
    axis.axhline(2.22e-16, color="grey", linestyle=":", label="double eps")
    axis.set_xlabel("r / sqrt(t_w)")
    axis.set_ylabel("relative deviation from 40-digit reference")
    axis.set_title("Check 1: chi_0 (3D) is the Ewald short-range kernel")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = out / "experiment_a_check1_chi0_identity.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path.name]


def plot_check2(out: Path) -> list[str]:
    """Adjacent-window slab profile and its deviation from DMK's D_l."""
    rows = read_rows(out / "experiment_a_check2_shell.csv")
    if not rows:
        return []
    scaled = [as_float(row["r_over_sigma_l"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].plot(scaled, [as_float(row["shell_dmk_erf_form"]) for row in rows])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("r / sigma_l")
    axes[0].set_ylabel("shell kernel D_l(r)")
    axes[0].set_title("Adjacent-window slab (finite, smooth at r = 0)")
    # Deviations are measured against the 60-digit erfc-difference reference,
    # not against each other: the two float64 forms lose relative accuracy at
    # opposite ends of the range, so a form-vs-form curve spans hundreds of
    # decades and says nothing about the identity.  Values are clipped into a
    # plottable window.
    ceiling = 1.0e2
    for column, label, style in (
        ("rel_dev_windows_vs_mpmath", "erfc (window) difference", "-"),
        ("rel_dev_dmk_erf_form_vs_mpmath", "erf (DMK) difference", "--"),
    ):
        values = [
            min(max(as_float(row.get(column, ""), FLOOR), FLOOR), ceiling)
            for row in rows
        ]
        if any(value > FLOOR for value in values):
            axes[1].loglog(scaled, values, linestyle=style, label=label)
    axes[1].axhline(2.22e-16, color="grey", linestyle=":", label="double eps")
    axes[1].set_ylim(1.0e-18, 1.0e3)
    axes[1].set_xlabel("r / sigma_l")
    axes[1].set_ylabel("relative deviation from 60-digit reference")
    axes[1].set_title("Check 2: float64 accuracy of the two algebraic forms")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    path = out / "experiment_a_check2_shell.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path.name]


def plot_check3(out: Path) -> list[str]:
    """Self-convergence of the graded v-rule and the Duffy disagreement."""
    written = []
    sweep = read_rows(out / "experiment_a_check3_v_convergence.csv")
    if sweep:
        dims = sorted({int(row["dim"]) for row in sweep})
        figure, axes = plt.subplots(
            1, len(dims), figsize=(4.8 * len(dims), 3.8), squeeze=False
        )
        for column, dim in enumerate(dims):
            axis = axes[0][column]
            labels = sorted(
                {row["target_label"] for row in sweep if int(row["dim"]) == dim}
            )
            for label in labels:
                selected = [
                    row
                    for row in sweep
                    if int(row["dim"]) == dim
                    and row["target_label"] == label
                    and int(row["m"]) == 0
                ]
                selected.sort(key=lambda row: int(row["n_v_nodes"]))
                if not selected:
                    continue
                axis.loglog(
                    [int(row["n_v_nodes"]) for row in selected],
                    [
                        max(as_float(row["max_rel_self_error"]), 1.0e-17)
                        for row in selected
                    ],
                    marker="o",
                    label=label,
                )
            axis.axhline(1.0e-13, color="grey", linestyle=":")
            axis.set_xlabel("graded v-quadrature nodes")
            axis.set_ylabel("max relative self-error")
            axis.set_title(f"Check 3: {dim}D, m = 0")
            axis.legend(fontsize=8)
        figure.tight_layout()
        path = out / "experiment_a_check3_separability_convergence.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path.name)

    entries = read_rows(out / "experiment_a_check3_entries.csv")
    if entries:
        figure, axis = plt.subplots(figsize=(6.6, 4.0))
        for dim in sorted({int(row["dim"]) for row in entries}):
            for label in sorted(
                {row["target_label"] for row in entries if int(row["dim"]) == dim}
            ):
                selected = [
                    row
                    for row in entries
                    if int(row["dim"]) == dim and row["target_label"] == label
                ]
                per_m = {}
                for row in selected:
                    m = int(row["m"])
                    per_m[m] = max(
                        per_m.get(m, 0.0), as_float(row["rel_diff_vs_target_max"])
                    )
                orders = sorted(per_m)
                axis.semilogy(
                    orders,
                    [max(per_m[m], FLOOR) for m in orders],
                    marker="o",
                    label=f"{dim}D {label}",
                )
        axis.set_xlabel("channel index m")
        axis.set_ylabel("max |Duffy - separable| / max |separable|")
        axis.set_title("Check 3: Duffy build against the separable u-integral")
        axis.legend(fontsize=7)
        figure.tight_layout()
        path = out / "experiment_a_check3_duffy_agreement.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path.name)
    return written


def plot_check4(out: Path) -> list[str]:
    """Gaussian versus Yukawa-tailed short-range parts."""
    rows = read_rows(out / "experiment_a_check4_yukawa.csv")
    if not rows:
        return []
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    ratios = sorted(
        {
            as_float(row["theta_over_Theta"])
            for row in rows
            if as_float(row["theta_over_Theta"]) <= 1.0
        }
    )
    for index, ratio in enumerate(ratios):
        selected = [
            row for row in rows if as_float(row["theta_over_Theta"]) == ratio
        ]
        selected.sort(key=lambda row: as_float(row["r_over_h"]))
        colour = f"C{index}"
        axis.semilogy(
            [as_float(row["r_over_h"]) for row in selected],
            [
                max(abs(as_float(row["W_T_heat_window_stable"])), 1.0e-300)
                for row in selected
            ],
            marker="o",
            color=colour,
            label=f"heat window, theta/Theta = {ratio:g}",
        )
        axis.semilogy(
            [as_float(row["r_over_h"]) for row in selected],
            [
                max(abs(as_float(row["short_range_fourier_gaussian"])), 1.0e-300)
                for row in selected
            ],
            marker="s",
            linestyle="--",
            color=colour,
            label=f"Fourier-Gaussian, theta/Theta = {ratio:g}",
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


def plot_check4_stability(out: Path) -> list[str]:
    """float64 accuracy of the two evaluations of the D4 closed form."""
    rows = read_rows(out / "experiment_a_check4_d4_stability.csv")
    if not rows:
        return []
    ceiling = 1.0e3
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for radius in sorted({as_float(row["r_over_h"]) for row in rows}):
        selected = [row for row in rows if as_float(row["r_over_h"]) == radius]
        selected.sort(key=lambda row: as_float(row["theta_over_Theta"]))
        ratios = [as_float(row["theta_over_Theta"]) for row in selected]
        for column, label, style in (
            ("rel_dev_stable", "erfcx form", "-"),
            ("rel_dev_naive", "naive erfc form", "--"),
        ):
            values = []
            for row in selected:
                value = as_float(row[column], ceiling)
                if not math.isfinite(value):
                    value = ceiling
                values.append(min(max(value, 1.0e-17), ceiling))
            axis.loglog(
                ratios, values, linestyle=style, marker="o",
                label=f"{label}, r/h = {radius:g}",
            )
    axis.axhline(2.22e-16, color="grey", linestyle=":")
    axis.set_xlabel("theta / Theta (equivalently mu sqrt(T))")
    axis.set_ylabel("relative deviation from the exact D4 value")
    axis.set_title("Check 4: stability of the closed-form windowed Yukawa kernel")
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = out / "experiment_a_check4_d4_stability.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path.name]


def main() -> None:
    """Render every available Experiment A plot."""
    parser = argparse.ArgumentParser(description="Experiment A plots from CSV")
    parser.add_argument("--out", required=True, help="directory holding the CSVs")
    args = parser.parse_args()
    out = Path(args.out).expanduser()

    written: list[str] = []
    for renderer in (
        plot_check1,
        plot_check2,
        plot_check3,
        plot_check4,
        plot_check4_stability,
    ):
        written.extend(renderer(out))
    print("wrote: " + (", ".join(written) if written else "(nothing)"))


if __name__ == "__main__":
    main()
