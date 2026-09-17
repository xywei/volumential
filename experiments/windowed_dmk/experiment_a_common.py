"""Shared helpers for the windowed-DMK Experiment A scripts.

Exploratory campaign code for the windowed RKE / DMK unification study (windowed RKE / DMK
unification).  Not benchmark-grade: no warm-up, no repetition statistics, no
claim of optimality for any quadrature rule used here.

The helpers here are deliberately dependency-light: NumPy is required, SciPy
and mpmath are probed lazily by the callers, and matplotlib is optional.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

WINDOW_THETA = 16.0
"""Paper 1's committed window declaration ``Theta``."""

ROOT_EXTENT = 2.0
"""Canonical root-box extent used by the Volumential channel tables."""

Q_ORDER = 3
"""Paper 1's committed (production) source order."""

P_STAR = 6
"""Paper 1's committed retained channel count."""


def output_dir(raw: str) -> Path:
    """Create and return the output directory for a run."""
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write ``rows`` as a CSV with the given column order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as pretty-printed JSON."""
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def environment_summary() -> dict[str, Any]:
    """Versions and CPU class of the running interpreter, with no host,
    user or path identifiers (campaign policy forbids recording them)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "machine": platform.machine(),
        "processor_class": platform.machine(),
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = None
    try:
        import mpmath

        info["mpmath"] = mpmath.__version__
    except ImportError:
        info["mpmath"] = None
    try:
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        info["matplotlib"] = None
    return info


def get_pyplot():
    """Return ``matplotlib.pyplot`` with the Agg backend, or ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    return np.polynomial.legendre.leggauss(int(order))


def panel_quadrature(
    breakpoints: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes and weights over consecutive panels."""
    breakpoints = np.asarray(breakpoints, dtype=np.float64)
    nodes, weights = gauss_legendre(order)
    all_nodes = []
    all_weights = []
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=False):
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        all_nodes.append(mid + half * nodes)
        all_weights.append(half * weights)
    return np.concatenate(all_nodes), np.concatenate(all_weights)


def relative_deviation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise ``|a - b| / max(|b|, tiny)``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.maximum(np.abs(b), np.finfo(float).tiny)
    return np.abs(a - b) / denominator
