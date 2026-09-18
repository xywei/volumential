"""Render the Experiment C plots from the CSVs the driver script writes.

``experiment_c_leaf_residual.py`` skips plotting when matplotlib is not
importable in the interpreter that runs it (the Volumential environment used for
the numerics has SciPy but no matplotlib).  This script needs only matplotlib
and the standard library, so it can be run afterwards with whatever interpreter
on the machine does have matplotlib:

    python experiments/windowed_dmk/experiment_c_plots.py --out OUT

Missing CSVs are skipped with a note; nothing here recomputes any number.

Exploratory, not benchmark-grade.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLOOR = 1.0e-18

SWEEP_CASES = (
    ("case2_halfplane", "case 2: half-plane cut y1 <= 0.3, global cubic"),
    (
        "case2_halfplane_piecewise",
        "case 2p: half-plane cut, piecewise per-leaf density",
    ),
    ("case3_wedge90", "case 3: right-angle wedge, on the bisector"),
    ("case3_wedge90_face", "case 3f: right-angle wedge, one face, away from apex"),
    ("case4_wedge60", "case 4: 60-degree wedge, on the bisector"),
)

COLUMNS = (
    ("err_physical", "o-", "RKE + DMK (physical-side windowed prefix)"),
    ("err_extended", "s--", "box code with extended source"),
    ("err_legacy", "^:", "plain DMK, interior series"),
)

FRYKLUND_LABEL = "DMK line, Fryklund Lemma 4.5"
FRYKLUND_LABEL_AMBIGUOUS = (
    "DMK line, Fryklund Lemma 4.5\n(closest point not unique; hollow markers)"
)


def read_rows(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts, or an empty list if absent."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default: float = float("nan")) -> float:
    """Parse a CSV cell as a float, tolerating blanks."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def plot_sweeps(out: Path) -> str | None:
    """Relative error of the four leaf columns against distance to the cut."""
    available = []
    for name, title in SWEEP_CASES:
        rows = read_rows(out / f"experiment_c_{name}.csv")
        if rows:
            available.append((name, title, rows))
    if not available:
        return None
    ncols = min(3, len(available))
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows), squeeze=False
    )
    panels = [ax for row in axes for ax in row]
    for ax, (_, title, rows) in zip(panels, available, strict=False):
        delta = [as_float(r["delta_over_sqrt_t_leaf"]) for r in rows]
        for key, style, label in COLUMNS:
            values = [max(as_float(r[key], FLOOR), FLOOR) for r in rows]
            ax.loglog(delta, values, style, label=label, markersize=4)
        fryklund = [r for r in rows if r.get("err_fryklund", "") not in ("", None)]
        if fryklund:
            unique = str(fryklund[0].get("fryklund_defined", "")).strip() == "True"
            ax.loglog(
                [as_float(r["delta_over_sqrt_t_leaf"]) for r in fryklund],
                [max(as_float(r["err_fryklund"], FLOOR), FLOOR) for r in fryklund],
                "D-." if unique else "D--",
                label=FRYKLUND_LABEL if unique else FRYKLUND_LABEL_AMBIGUOUS,
                markersize=5,
                **({} if unique else {"markerfacecolor": "none"}),
            )
        ax.set_xlabel(
            "target distance to the cut (bisector sweeps: to the apex),\n"
            "in units of sqrt(t_L)"
        )
        ax.set_ylabel("relative error of the split total")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)
    for ax in panels[len(available):]:
        ax.axis("off")
    fig.tight_layout()
    path = out / "experiment_c_boundary_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def plot_smoothness(out: Path) -> str | None:
    """Tensor-Gauss order convergence of the split pieces on one leaf."""
    rows = read_rows(out / "experiment_c_smooth_quadrature.csv")
    if not rows:
        return None
    keys = [k for k in rows[0] if k.startswith("relerr_")]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    orders = [as_float(r["order"]) for r in rows]
    for key in keys:
        values = [max(as_float(r[key], FLOOR), FLOOR) for r in rows]
        ax.semilogy(orders, values, "o-", markersize=4, label=key[len("relerr_") :])
    ax.set_xlabel("tensor-Gauss order per direction on the target's own leaf")
    ax.set_ylabel("relative error")
    ax.set_title("only the finest window is singular", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "experiment_c_smooth_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def plot_prefix_magnitude(out: Path) -> str | None:
    """The four leaf columns themselves, next to the reference potential."""
    rows = read_rows(out / "experiment_c_case2_halfplane.csv")
    if not rows:
        return None
    delta = [as_float(r["delta_over_sqrt_t_leaf"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for key, style, label in (
        ("prefix_physical", "o-", "RKE + DMK (physical-side windowed prefix)"),
        ("prefix_extended", "s--", "box code with extended source"),
        ("prefix_legacy", "^:", "plain DMK, interior series"),
        ("prefix_fryklund", "D-.", FRYKLUND_LABEL),
    ):
        if key not in rows[0]:
            continue
        values = [as_float(r[key]) for r in rows]
        ax.semilogx(delta, values, style, markersize=4, label=label)
    ax.set_xlabel("target distance to the cut, in units of sqrt(t_L)")
    ax.set_ylabel("finest-window leaf contribution")
    ax.set_title("case 2: the leaf column itself", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "experiment_c_leaf_columns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main() -> None:
    """Render every Experiment C plot whose CSV is present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory holding the CSVs")
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    written = [plot_sweeps(out), plot_smoothness(out), plot_prefix_magnitude(out)]
    for name in written:
        print("wrote" if name else "skipped", name)


if __name__ == "__main__":
    main()
