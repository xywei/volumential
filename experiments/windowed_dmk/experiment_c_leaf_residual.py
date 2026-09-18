"""Experiment C of the windowed-RKE / DMK unification study.

2D Laplace, kernel ``-log(r) / (2 pi)``, on a uniform three-level tree over
``[-1, 1]^2`` with piecewise tensor-polynomial sources of degree at most 3.
The telescoping split used throughout is

    K = (K - W_{t_0}) + sum_{l<L} (W_{t_l} - W_{t_{l+1}}) + W_{t_L},

with the heat-time windowed prefix (the ``m = 0`` windowed RKE channel in 2D)

    W_t(r) = (1 / (4 pi)) E_1(r^2 / (4 t)),   t_l = (h_l / Theta)^2, Theta = 8.

Only the finest residual ``W_{t_L}`` is singular; everything coarser is smooth
and is integrated here by direct high-order quadrature.  The experiment asks
whether the finest residual can be replaced by the windowed prefix evaluated on
the *physical* leaf geometry, and how the legacy DMK-style local treatment (the
asymptotic Laplacian series ``sum_j t_L^{j+1}/(j+1)! Delta^j rho(x)``, i.e. the
free-space moment operator) degrades when the window sees a cut.

Checks
------
1. kernel-level telescoping identity (all pieces summed against ``K``);
2. case 1, interior full box: split total against a high-accuracy reference,
   windowed prefix by the separable ``u``-integral (D7) against target-centred
   polar quadrature, with a globally smooth and a genuinely piecewise density;
3. case 2, half-plane cut ``y_1 <= 0.3``: four leaf columns (physical-side
   prefix, box-extended prefix, legacy asymptotic, and the Fryklund Lemma 4.5
   line described below) against the reference as the target approaches the
   cut, with the same sweep repeated for a genuinely piecewise per-leaf density;
4. case 3, right-angle wedge ``y_1 <= 0.3, y_2 <= 0.1``: the same four columns
   on the bisector, where no boundary point is the unique closest one, plus a
   second short sweep towards one face and away from the apex, where one is;
5. case 4, 60-degree wedge: physical-side prefix by polar quadrature only
   (the region is not separable in Cartesian coordinates);
6. smoothness evidence: tensor-Gauss order convergence of the shells and of the
   coarsest term over the target's own leaf, against the singular kernel;
7. node count of the ``u``-quadrature (in ``u = v^2``) needed by the separable
   prefix as a function of the target's distance to the cut.

The fourth column is the fairest comparison a DMK-style local treatment can
make at a flat wall: ``fryklund_VL`` of ``experiment_e_baseline``, that is Lemma
4.5 of Fryklund, Greengard, Jiang and Potter (2024) for the local volume
potential, reused unchanged and evaluated with window heat time ``delta = t_L``
(the same leaf window the other columns use), curvature ``kappa_b = 0``, and the
exact Taylor jet of the target leaf's polynomial in the frame of the chosen wall
(``xi`` along the tangent, ``eta`` along the inward normal).  The lemma is exact
through ``delta^2``; at a flat wall its first omitted group is the degree-three
part of the density, which ``flat_wall_cubic_remainder`` supplies in closed
form, so the measured gap can be checked against theory instead of merely
observed.  On a wedge bisector every boundary point is equidistant and the
expansion has no closest point; it is applied there anyway with the
deterministic choice ``y_1 = 0.3``, and those rows carry
``fryklund_defined = False``.

Reference values use target-centred polar (Duffy) quadrature over the fan of
triangles with the target as apex, with the radial ``r^k log r`` moments taken
in closed form; the only quadrature error is in the angular variable, and the
scripts reports its self-convergence.

Exploratory, not benchmark-grade: no warm-up, no repetitions, no claim that any
quadrature rule used here is optimal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import erf, erfc, exp1

try:  # Fryklund's Lemma 4.5 line is reused here, never reimplemented
    from experiment_e_baseline import fryklund_VL
except ImportError:  # pragma: no cover - run from another working directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_e_baseline import fryklund_VL


EULER_GAMMA = 0.5772156649015328606
THETA = 8.0
"""Window declaration used by this experiment."""

ROOT_HALF = 1.0
"""Half-extent of the root box ``[-1, 1]^2``."""

N_LEVELS = 3
"""Levels 0, 1, 2; the leaves live on level ``L = 2``."""

DENS_GLOBAL = np.array(
    [
        [1.0, -0.4, 0.2, 0.05],
        [0.7, -0.5, -0.15, 0.0],
        [0.3, 0.25, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0],
    ]
)
"""Coefficients ``c[i, j]`` of ``y_1^i y_2^j``; total degree at most 3."""


# ---------------------------------------------------------------------------
# small I/O helpers (kept local so this script stands alone)
# ---------------------------------------------------------------------------


def output_dir(raw: str) -> Path:
    """Create and return the output directory for a run."""
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` as CSV using the first row's key order."""
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as pretty-printed JSON."""
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def environment_summary() -> dict:
    """Interpreter and library versions, with no host, user or path identifiers."""
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "machine": platform.machine(),
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover - scipy is required here
        info["scipy"] = None
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


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def e1_plus_log(z):
    """``E_1(z) + log(z)``, evaluated without cancellation near ``z = 0``.

    The series ``E_1(z) + log(z) = -gamma + sum_{n>=1} (-1)^{n+1} z^n / (n n!)``
    is used below 1, the direct combination above it.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape, dtype=np.float64)
    small = z < 1.0
    zs = z[small]
    acc = np.zeros(zs.shape, dtype=np.float64)
    term = np.ones(zs.shape, dtype=np.float64)
    for n in range(1, 26):
        term = term * zs / n
        acc += ((-1.0) ** (n + 1)) * term / n
    out[small] = -EULER_GAMMA + acc
    zl = z[~small]
    out[~small] = exp1(zl) + np.log(zl)
    return out


def log_kernel(r):
    """The free-space 2D Laplace kernel ``-log(r) / (2 pi)``."""
    return -np.log(r) / (2.0 * np.pi)


def windowed_prefix(r, t):
    """The windowed prefix ``W_t(r) = E_1(r^2 / (4 t)) / (4 pi)``."""
    return exp1(np.asarray(r, dtype=np.float64) ** 2 / (4.0 * t)) / (4.0 * np.pi)


def prefix_smooth(r, t):
    """``W_t(r) + log(r) / (2 pi)``: the windowed prefix minus its log germ."""
    r = np.asarray(r, dtype=np.float64)
    z = r * r / (4.0 * t)
    return (e1_plus_log(z) + math.log(4.0 * t)) / (4.0 * np.pi)


class RadialKernel:
    """A radial kernel written as ``log_coeff * (-log r / (2 pi)) + smooth(r)``."""

    def __init__(self, name, log_coeff, smooth, scale):
        """Store the name, the log-germ coefficient, the smooth part and a scale."""
        self.name = name
        self.log_coeff = float(log_coeff)
        self.smooth = smooth
        self.scale = float(scale)

    def __call__(self, r):
        """Evaluate the kernel."""
        value = self.smooth(r) if self.smooth is not None else np.zeros_like(r)
        if self.log_coeff:
            value = value + self.log_coeff * log_kernel(r)
        return value


def make_kernels(windows):
    """Return the split pieces for the given decreasing window list."""
    t0 = windows[0]
    t_leaf = windows[-1]
    kernels = {
        "full": RadialKernel("full", 1.0, None, t_leaf**0.5),
        "coarse": RadialKernel(
            "coarse", 0.0, lambda r: -prefix_smooth(r, t0), t0**0.5
        ),
        "prefix": RadialKernel(
            "prefix", 1.0, lambda r: prefix_smooth(r, t_leaf), t_leaf**0.5
        ),
    }
    for idx in range(len(windows) - 1):
        t_hi = windows[idx]
        t_lo = windows[idx + 1]
        kernels[f"shell{idx}"] = RadialKernel(
            f"shell{idx}",
            0.0,
            (lambda r, a=t_hi, b=t_lo: prefix_smooth(r, a) - prefix_smooth(r, b)),
            t_lo**0.5,
        )
    return kernels


# ---------------------------------------------------------------------------
# polynomial helpers
# ---------------------------------------------------------------------------


def poly_eval(c, x):
    """Evaluate the coefficient array ``c`` at the point ``x``."""
    total = 0.0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j]:
                total += c[i, j] * x[0] ** i * x[1] ** j
    return float(total)


def poly_laplacian(c):
    """Return the coefficient array of the Laplacian of ``c``."""
    out = np.zeros_like(c)
    for i in range(2, c.shape[0]):
        for j in range(c.shape[1]):
            out[i - 2, j] += i * (i - 1) * c[i, j]
    for i in range(c.shape[0]):
        for j in range(2, c.shape[1]):
            out[i, j - 2] += j * (j - 1) * c[i, j]
    return out


def poly_partial(c, axis):
    """Coefficient array of the partial derivative of ``c`` along ``axis``."""
    out = np.zeros_like(c)
    if axis == 0:
        for i in range(1, c.shape[0]):
            out[i - 1, :] += i * c[i, :]
    else:
        for j in range(1, c.shape[1]):
            out[:, j - 1] += j * c[:, j]
    return out


def poly_jet_in_frame(c, x, tangent, inward_normal):
    """Exact Taylor jet of the polynomial ``c`` at ``x`` in a rotated frame.

    Returns ``(jet, f_xixieta, f_etaetaeta)``: Fryklund's six second-order jet
    entries in the frame whose first axis is ``tangent`` and whose second axis
    is ``inward_normal``, then the two third derivatives that Lemma 4.5 omits.
    Every derivative is taken on the coefficient array, so no numerical
    differencing of the density enters anywhere.
    """
    tan = np.asarray(tangent, dtype=np.float64)
    nrm = np.asarray(inward_normal, dtype=np.float64)
    arrays = {}
    arrays[(1, 0)] = poly_partial(c, 0)
    arrays[(0, 1)] = poly_partial(c, 1)
    arrays[(2, 0)] = poly_partial(arrays[(1, 0)], 0)
    arrays[(1, 1)] = poly_partial(arrays[(1, 0)], 1)
    arrays[(0, 2)] = poly_partial(arrays[(0, 1)], 1)
    arrays[(3, 0)] = poly_partial(arrays[(2, 0)], 0)
    arrays[(2, 1)] = poly_partial(arrays[(2, 0)], 1)
    arrays[(1, 2)] = poly_partial(arrays[(1, 1)], 1)
    arrays[(0, 3)] = poly_partial(arrays[(0, 2)], 1)
    value = {key: poly_eval(arr, x) for key, arr in arrays.items()}
    grad = np.array([value[(1, 0)], value[(0, 1)]])
    hess = np.array(
        [[value[(2, 0)], value[(1, 1)]], [value[(1, 1)], value[(0, 2)]]]
    )
    third = np.zeros((2, 2, 2))
    third[0, 0, 0] = value[(3, 0)]
    third[1, 1, 1] = value[(0, 3)]
    for index in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        third[index] = value[(2, 1)]
    for index in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        third[index] = value[(1, 2)]
    jet = {
        "f": poly_eval(c, x),
        "f_xi": float(tan @ grad),
        "f_eta": float(nrm @ grad),
        "f_xixi": float(tan @ hess @ tan),
        "f_xieta": float(tan @ hess @ nrm),
        "f_etaeta": float(nrm @ hess @ nrm),
    }
    f_xixieta = float(np.einsum("abc,a,b,c->", third, tan, tan, nrm))
    f_etaetaeta = float(np.einsum("abc,a,b,c->", third, nrm, nrm, nrm))
    return jet, f_xixieta, f_etaetaeta


def legacy_asymptotic(c, x, t_leaf):
    """DMK-style leaf asymptotics ``sum_j t^{j+1}/(j+1)! (Delta^j rho)(x)``.

    This is the free-space moment operator of the windowed prefix applied to the
    polynomial continued to all of space; the sum terminates for polynomials.
    """
    total = 0.0
    current = np.array(c, dtype=np.float64)
    order = 0
    while np.any(current != 0.0):
        total += t_leaf ** (order + 1) / math.factorial(order + 1) * poly_eval(
            current, x
        )
        current = poly_laplacian(current)
        order += 1
        if order > 12:
            break
    return total


def ray_coefficients(c, x, dirs):
    """Coefficients in ``r`` of ``rho(x + r n)`` for each direction ``n``.

    ``dirs`` has shape ``(N, 2)``; the result has shape ``(N, I + J - 1)``.
    """
    i_max, j_max = c.shape
    n_dir = dirs.shape[0]
    a = np.zeros((n_dir, i_max, i_max))
    a[:, 0, 0] = 1.0
    for i in range(1, i_max):
        a[:, i, 0] = x[0] * a[:, i - 1, 0]
        for k in range(1, i + 1):
            a[:, i, k] = x[0] * a[:, i - 1, k] + dirs[:, 0] * a[:, i - 1, k - 1]
    b = np.zeros((n_dir, j_max, j_max))
    b[:, 0, 0] = 1.0
    for j in range(1, j_max):
        b[:, j, 0] = x[1] * b[:, j - 1, 0]
        for k in range(1, j + 1):
            b[:, j, k] = x[1] * b[:, j - 1, k] + dirs[:, 1] * b[:, j - 1, k - 1]
    out = np.zeros((n_dir, i_max + j_max - 1))
    for i in range(i_max):
        for j in range(j_max):
            if not c[i, j]:
                continue
            for k1 in range(i + 1):
                for k2 in range(j + 1):
                    out[:, k1 + k2] += c[i, j] * a[:, i, k1] * b[:, j, k2]
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def rect_polygon(rect):
    """Counter-clockwise vertices of the axis-aligned rectangle ``rect``."""
    a1, b1, a2, b2 = rect
    return np.array([[a1, a2], [b1, a2], [b1, b2], [a1, b2]], dtype=np.float64)


def clip_polygon(vertices, half_planes):
    """Sutherland-Hodgman clip against ``(point, normal)`` half-planes.

    A half-plane is the set ``(y - point) . normal <= 0``.
    """
    poly = np.asarray(vertices, dtype=np.float64)
    for point, normal in half_planes:
        point = np.asarray(point, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        if poly.shape[0] == 0:
            return poly
        values = (poly - point) @ normal
        out = []
        n = poly.shape[0]
        for i in range(n):
            j = (i + 1) % n
            vi, vj = values[i], values[j]
            if vi <= 0.0:
                out.append(poly[i])
            if (vi < 0.0 < vj) or (vj < 0.0 < vi):
                s = vi / (vi - vj)
                out.append(poly[i] + s * (poly[j] - poly[i]))
        poly = np.array(out, dtype=np.float64) if out else np.zeros((0, 2))
    return poly


def polygon_area(vertices):
    """Signed area of a simple polygon."""
    if vertices.shape[0] < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def leaf_rects(n_side):
    """Axis-aligned leaf boxes of a uniform grid over the root box."""
    edges = np.linspace(-ROOT_HALF, ROOT_HALF, n_side + 1)
    rects = []
    for i in range(n_side):
        for j in range(n_side):
            rects.append((edges[i], edges[i + 1], edges[j], edges[j + 1]))
    return rects


# ---------------------------------------------------------------------------
# target-centred polar quadrature over a polygon
# ---------------------------------------------------------------------------


def _gauss(order):
    """Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    return np.polynomial.legendre.leggauss(int(order))


def _dyadic_breakpoints(lo, hi):
    """Breakpoints on ``[lo, hi]`` graded dyadically away from the origin."""
    marks = {lo, hi}
    limit = max(abs(lo), abs(hi))
    k = 0
    value = 1.0
    while value < limit:
        for signed in (value, -value):
            if lo < signed < hi:
                marks.add(signed)
        k += 1
        value = 2.0**k
    if lo < 0.0 < hi:
        marks.add(0.0)
    return np.array(sorted(marks), dtype=np.float64)


def _panel_rule(breakpoints, order):
    """Composite Gauss nodes and weights over consecutive panels."""
    gx, gw = _gauss(order)
    left = breakpoints[:-1]
    right = breakpoints[1:]
    mid = 0.5 * (left + right)
    half = 0.5 * (right - left)
    nodes = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    weights = (half[:, None] * gw[None, :]).ravel()
    return nodes, weights


def _log_moments(radius, k_max):
    """``int_0^R (-log r / (2 pi)) r^{k+1} dr`` for ``k = 0 .. k_max``."""
    radius = np.asarray(radius, dtype=np.float64)
    out = np.zeros((radius.size, k_max + 1))
    positive = radius > 0.0
    log_r = np.zeros_like(radius)
    log_r[positive] = np.log(radius[positive])
    for k in range(k_max + 1):
        power = radius ** (k + 2)
        out[:, k] = -(power * log_r / (k + 2) - power / (k + 2) ** 2) / (2.0 * np.pi)
    return out


def _smooth_moments(kernel, radius, k_max, order):
    """``int_0^R smooth(r) r^{k+1} dr`` by composite Gauss graded on the scale."""
    radius = np.asarray(radius, dtype=np.float64)
    template = kernel.scale * 2.0 ** np.arange(-3.0, 11.0)
    clipped = np.minimum(template[None, :], radius[:, None])
    bounds = np.concatenate(
        [np.zeros((radius.size, 1)), clipped, radius[:, None]], axis=1
    )
    left = bounds[:, :-1]
    right = np.maximum(bounds[:, 1:], bounds[:, :-1])
    gx, gw = _gauss(order)
    mid = 0.5 * (left + right)
    half = 0.5 * (right - left)
    nodes = mid[:, :, None] + half[:, :, None] * gx[None, None, :]
    weights = half[:, :, None] * gw[None, None, :]
    values = kernel.smooth(nodes) * weights
    out = np.zeros((radius.size, k_max + 1))
    for k in range(k_max + 1):
        out[:, k] = np.sum(values * nodes ** (k + 1), axis=(1, 2))
    return out


def radial_moments(kernel, radius, k_max, order):
    """``int_0^R kernel(r) r^{k+1} dr`` for ``k = 0 .. k_max``."""
    out = np.zeros((np.asarray(radius).size, k_max + 1))
    if kernel.log_coeff:
        out += kernel.log_coeff * _log_moments(radius, k_max)
    if kernel.smooth is not None:
        out += _smooth_moments(kernel, radius, k_max, order)
    return out


def polar_polygon_potential(vertices, target, dens, kernel, n_ang, n_rad):
    """``int_P kernel(|x - y|) rho(y) dy`` by the target-centred triangle fan.

    Each directed polygon edge spans a signed triangle with the target as apex;
    inside a triangle the radial integral of ``kernel(r) rho r`` is taken with
    exact ``r^k log r`` moments plus a graded Gauss rule for the smooth part,
    and the angular variable is substituted by ``w = tan(phi - phi_0)`` so the
    integrand is analytic in ``w`` even when the target is near the edge line.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] < 3:
        return 0.0
    target = np.asarray(target, dtype=np.float64)
    k_max = dens.shape[0] + dens.shape[1] - 2
    total = 0.0
    n = vertices.shape[0]
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-14:
            continue
        tangent = edge / length
        foot = a + float((target - a) @ tangent) * tangent
        offset = foot - target
        dist = float(np.hypot(offset[0], offset[1]))
        if dist < 1e-13:
            continue
        nhat = offset / dist
        that = np.array([-nhat[1], nhat[0]])
        w_a = float((a - target) @ that) / dist
        w_b = float((b - target) @ that) / dist
        if abs(w_b - w_a) < 1e-15:
            continue
        sign = 1.0 if w_b > w_a else -1.0
        lo, hi = min(w_a, w_b), max(w_a, w_b)
        nodes, weights = _panel_rule(_dyadic_breakpoints(lo, hi), n_ang)
        one_plus = 1.0 + nodes * nodes
        root = np.sqrt(one_plus)
        dirs = (nhat[None, :] + nodes[:, None] * that[None, :]) / root[:, None]
        radius = dist * root
        coeffs = ray_coefficients(dens, target, dirs)
        moments = radial_moments(kernel, radius, k_max, n_rad)
        inner = np.sum(coeffs * moments, axis=1) / one_plus
        total += sign * float(np.sum(weights * inner))
    return total


# ---------------------------------------------------------------------------
# separable u-integral for the windowed prefix over an axis-aligned rectangle
# ---------------------------------------------------------------------------


def _gaussian_moments(x, a, b, sigma, k_max):
    """``int_a^b exp(-(y - x)^2 / (2 sigma^2)) y^i dy`` for ``i = 0 .. k_max``.

    ``sigma`` is an array; the result has shape ``(k_max + 1, sigma.size)``.
    """
    za = (a - x) / sigma
    zb = (b - x) / sigma
    root2 = math.sqrt(2.0)
    scale = math.sqrt(math.pi / 2.0)
    both_pos = za >= 0.0
    both_neg = zb <= 0.0
    m0 = scale * (erf(zb / root2) - erf(za / root2))
    m0_pos = scale * (erfc(za / root2) - erfc(zb / root2))
    m0_neg = scale * (erfc(-zb / root2) - erfc(-za / root2))
    m0 = np.where(both_pos, m0_pos, np.where(both_neg, m0_neg, m0))
    ea = np.exp(-0.5 * za * za)
    eb = np.exp(-0.5 * zb * zb)
    moments = [m0, ea - eb]
    for k in range(2, k_max + 1):
        bracket = zb ** (k - 1) * eb - za ** (k - 1) * ea
        moments.append((k - 1) * moments[k - 2] - bracket)
    out = np.zeros((k_max + 1, sigma.size))
    for i in range(k_max + 1):
        acc = np.zeros(sigma.size)
        for k in range(i + 1):
            acc += math.comb(i, k) * x ** (i - k) * sigma**k * moments[k]
        out[i] = sigma * acc
    return out


def separable_prefix_rect(rect, target, dens, t_leaf, n_panel=26, order=16):
    """Windowed-prefix entry over a rectangle by the separable ``u``-integral.

    Uses ``u = v^2`` and dyadically graded panels in ``v`` on ``[0, sqrt(T)]``,
    so the integrand is analytic for every target position (including targets on
    a face of the rectangle).  Returns the value and the number of ``v`` nodes.
    """
    a1, b1, a2, b2 = rect
    if b1 <= a1 or b2 <= a2:
        return 0.0, 0
    v_max = math.sqrt(t_leaf)
    marks = [0.0] + [v_max * 2.0 ** (-k) for k in range(n_panel, -1, -1)]
    breakpoints = np.array(sorted(set(marks)), dtype=np.float64)
    nodes, weights = _panel_rule(breakpoints, order)
    sigma = math.sqrt(2.0) * nodes
    i_max = dens.shape[0] - 1
    j_max = dens.shape[1] - 1
    f1 = _gaussian_moments(target[0], a1, b1, sigma, i_max)
    f2 = _gaussian_moments(target[1], a2, b2, sigma, j_max)
    acc = np.zeros(nodes.size)
    for i in range(i_max + 1):
        for j in range(j_max + 1):
            if dens[i, j]:
                acc += dens[i, j] * f1[i] * f2[j]
    integrand = acc / (2.0 * np.pi * nodes)
    return float(np.sum(weights * integrand)), int(nodes.size)


# ---------------------------------------------------------------------------
# the Fryklund Lemma 4.5 line at a flat wall
# ---------------------------------------------------------------------------


def _t_power_gauss_moment(power, r, delta):
    """``int_0^delta t^p exp(-r^2 / (4 t)) dt``, by graded Gauss in ``t = delta w^2``.

    The substitution makes the integrand ``2 delta^{p+1} w^{2p+1}
    exp(-r^2 / (4 delta w^2))``, which is flat at ``w = 0``; the panels are
    graded dyadically towards ``0`` and refined uniformly near ``w = 1``.
    """
    marks = {0.0, 1.0}
    for k in range(31):
        marks.add(2.0**-k)
    for i in range(1, 20):
        marks.add(i / 20.0)
    nodes, weights = _panel_rule(np.array(sorted(marks), dtype=np.float64), 20)
    exponent = np.zeros_like(nodes)
    if r > 0.0:
        exponent = -(r * r) / (4.0 * delta * nodes * nodes)
    integrand = nodes ** (2.0 * power + 1.0) * np.exp(exponent)
    return float(2.0 * delta ** (power + 1.0) * np.sum(weights * integrand))


def flat_wall_cubic_remainder(f_xixieta, f_etaetaeta, r, delta):
    """The term Lemma 4.5 omits at a flat wall for a density of degree three.

    The lemma is exact through ``delta^2``, i.e. through the quadratic part of
    the density's Taylor expansion at the target.  Over the half-plane
    ``{eta > -r}`` the cubic part contributes, after the Gaussian ``xi``
    integral kills the terms odd in ``xi``,

        pi^{-1/2} [ (f_xixieta + 2 f_etaetaeta / 3) A_{3/2}
                    + f_etaetaeta r^2 A_{1/2} / 6 ],
        A_p = int_0^delta t^p exp(-r^2 / (4 t)) dt,

    which on the wall itself is ``2 delta^{5/2} (f_xixieta
    + 2 f_etaetaeta / 3) / (5 sqrt(pi))``.  For a density of degree at most
    three this is the *whole* remainder, so the lemma plus this term is the
    exact half-plane value and the sum can be checked to roundoff.
    """
    a_half = _t_power_gauss_moment(0.5, r, delta)
    a_three_half = _t_power_gauss_moment(1.5, r, delta)
    return float(
        (
            (f_xixieta + 2.0 * f_etaetaeta / 3.0) * a_three_half
            + f_etaetaeta * r * r * a_half / 6.0
        )
        / math.sqrt(math.pi)
    )


def fryklund_line(case, model, target, t_leaf):
    """Lemma 4.5 at ``target`` for the wall the case declares, or ``None``.

    The window heat time is the leaf window ``t_L`` the other columns use, the
    curvature is zero (every wall here is straight), and the density jet is the
    exact Taylor jet of the target leaf's polynomial in the frame whose inward
    normal points away from the wall.  ``defined`` is ``False`` when the case
    knows that no boundary point is the unique closest one, in which case the
    wall is the deterministic first choice and the number is reported anyway.
    """
    spec = case.get("fryklund")
    if spec is None:
        return None
    axis = int(spec["axis"])
    wall = float(spec["value"])
    base = np.array(target, dtype=np.float64)
    base[axis] = wall
    inward = np.zeros(2)
    inward[axis] = -1.0
    tangent = np.array([inward[1], -inward[0]])
    dens = model.dens[model.leaf_index(target)]
    jet, f_xixieta, f_etaetaeta = poly_jet_in_frame(dens, target, tangent, inward)
    r = float(wall - float(target[axis]))
    return {
        "value": float(fryklund_VL(target, base, 0.0, jet, t_leaf, sign=1.0)),
        "r": r,
        "remainder": flat_wall_cubic_remainder(f_xixieta, f_etaetaeta, r, t_leaf),
        "defined": bool(spec.get("unique_closest", True)),
    }


# ---------------------------------------------------------------------------
# tensor-Gauss quadrature over a triangle (smoothness evidence)
# ---------------------------------------------------------------------------


def triangle_tensor_gauss(tri, order):
    """Collapsed (Duffy) tensor-Gauss nodes and weights on a triangle."""
    v0, v1, v2 = np.asarray(tri, dtype=np.float64)
    gx, gw = _gauss(order)
    s = 0.5 * (gx + 1.0)
    ws = 0.5 * gw
    xi = s[:, None]
    eta = s[None, :]
    points = (
        (1.0 - xi)[:, :, None] * v0[None, None, :]
        + (xi * (1.0 - eta))[:, :, None] * v1[None, None, :]
        + (xi * eta)[:, :, None] * v2[None, None, :]
    )
    area = 0.5 * abs(
        (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
    )
    weights = 2.0 * area * xi * ws[:, None] * ws[None, :]
    return points.reshape(-1, 2), weights.reshape(-1)


def tensor_gauss_polygon_potential(vertices, target, dens, kernel, order):
    """Potential over a polygon by a plain tensor-Gauss rule.

    No singularity treatment: this is the smoothness probe of check 6.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] < 3:
        return 0.0
    total = 0.0
    for i in range(1, vertices.shape[0] - 1):
        tri = (vertices[0], vertices[i], vertices[i + 1])
        points, weights = triangle_tensor_gauss(tri, order)
        radius = np.hypot(points[:, 0] - target[0], points[:, 1] - target[1])
        values = np.zeros(points.shape[0])
        for a in range(dens.shape[0]):
            for b in range(dens.shape[1]):
                if dens[a, b]:
                    values += dens[a, b] * points[:, 0] ** a * points[:, 1] ** b
        total += float(np.sum(weights * kernel(radius) * values))
    return total


# ---------------------------------------------------------------------------
# model assembly
# ---------------------------------------------------------------------------


class Model:
    """A leaf partition of the physical domain with per-leaf densities."""

    def __init__(self, half_planes, axis_bounds, piecewise, n_side):
        """Clip every leaf of the uniform grid against the case's half-planes."""
        self.half_planes = half_planes
        self.axis_bounds = axis_bounds
        self.rects = leaf_rects(n_side)
        self.polygons = [
            clip_polygon(rect_polygon(r), half_planes) for r in self.rects
        ]
        self.dens = []
        for idx, rect in enumerate(self.rects):
            if piecewise:
                scale = 1.0 + 0.35 * math.sin(2.3 * idx + 0.7)
                tilt = 0.25 * math.cos(1.7 * idx)
                coeffs = DENS_GLOBAL * scale
                coeffs[1, 0] += tilt
                coeffs[0, 1] -= tilt
            else:
                coeffs = DENS_GLOBAL.copy()
            self.dens.append(coeffs)
        self.clipped_rects = None
        if axis_bounds is not None:
            self.clipped_rects = [self._clip_rect(r) for r in self.rects]

    def _clip_rect(self, rect):
        """Apply the case's axis-aligned upper bounds to a rectangle."""
        a1, b1, a2, b2 = rect
        for axis, bound in self.axis_bounds:
            if axis == 0:
                b1 = min(b1, bound)
            else:
                b2 = min(b2, bound)
        return (a1, b1, a2, b2)

    def leaf_index(self, target):
        """Index of the leaf rectangle containing ``target``."""
        for idx, (a1, b1, a2, b2) in enumerate(self.rects):
            if a1 <= target[0] <= b1 and a2 <= target[1] <= b2:
                return idx
        raise ValueError("target outside the root box")

    def polar_potential(self, target, kernel, n_ang, n_rad):
        """Sum the polar polygon potential over all leaves."""
        total = 0.0
        for polygon, dens in zip(self.polygons, self.dens, strict=False):
            if polygon.shape[0] >= 3:
                total += polar_polygon_potential(
                    polygon, target, dens, kernel, n_ang, n_rad
                )
        return total

    def separable_prefix(self, target, t_leaf, physical=True):
        """Sum the separable windowed-prefix entry over all leaves."""
        rects = self.clipped_rects if physical else self.rects
        if rects is None:
            return None, 0
        total = 0.0
        nodes = 0
        for rect, dens in zip(rects, self.dens, strict=False):
            value, count = separable_prefix_rect(rect, target, dens, t_leaf)
            total += value
            nodes = max(nodes, count)
        return total, nodes


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_kernel_identity(windows, kernels):
    """Check 1: the split pieces sum to the free-space kernel."""
    radii = np.logspace(-6.0, 0.6, 60)
    full = log_kernel(radii)
    pieces = kernels["coarse"](radii) + kernels["prefix"](radii)
    for idx in range(len(windows) - 1):
        pieces = pieces + kernels[f"shell{idx}"](radii)
    residual = np.abs(full - pieces)
    rows = []
    for k, r in enumerate(radii):
        row = {
            "r": r,
            "kernel": full[k],
            "coarse": float(kernels["coarse"](np.array([r]))[0]),
            "prefix": float(kernels["prefix"](np.array([r]))[0]),
            "sum": pieces[k],
            "abs_residual": residual[k],
        }
        for idx in range(len(windows) - 1):
            row[f"shell{idx}"] = float(kernels[f"shell{idx}"](np.array([r]))[0])
        rows.append(row)
    summary = {
        "max_abs_residual": float(np.max(residual)),
        "max_rel_residual": float(np.max(residual / np.abs(full))),
        "shell_value_at_origin": {
            f"shell{idx}": float(kernels[f"shell{idx}"](np.array([1e-14]))[0])
            for idx in range(len(windows) - 1)
        },
        "shell_value_at_origin_expected": {
            f"shell{idx}": math.log(windows[idx] / windows[idx + 1])
            / (4.0 * math.pi)
            for idx in range(len(windows) - 1)
        },
        "n_samples": int(radii.size),
    }
    return rows, summary


def evaluate_target(
    model, kernels, windows, target, n_ang, n_rad, want_polar_prefix
):
    """All columns of the split at one target."""
    t_leaf = windows[-1]
    reference = model.polar_potential(target, kernels["full"], n_ang, n_rad)
    smooth = model.polar_potential(target, kernels["coarse"], n_ang, n_rad)
    shells = []
    for idx in range(len(windows) - 1):
        shells.append(
            model.polar_potential(target, kernels[f"shell{idx}"], n_ang, n_rad)
        )
    smooth_total = smooth + sum(shells)
    prefix_phys, n_v = model.separable_prefix(target, t_leaf, physical=True)
    prefix_ext, _ = model.separable_prefix(target, t_leaf, physical=False)
    prefix_polar = None
    if want_polar_prefix:
        prefix_polar = model.polar_potential(target, kernels["prefix"], n_ang, n_rad)
    leaf = model.leaf_index(target)
    prefix_legacy = legacy_asymptotic(model.dens[leaf], target, t_leaf)
    out = {
        "reference": reference,
        "smooth_total": smooth_total,
        "smooth_coarse": smooth,
        "prefix_physical": prefix_phys,
        "prefix_extended": prefix_ext,
        "prefix_legacy": prefix_legacy,
        "prefix_polar": prefix_polar,
        "n_v_nodes": n_v,
    }
    for idx, value in enumerate(shells):
        out[f"smooth_shell{idx}"] = value
    return out


def sweep_case(case, model, kernels, windows, targets, n_ang, n_rad, polar_every):
    """Run the four-column comparison over a list of targets."""
    rows = []
    for k, (label, target) in enumerate(targets):
        want_polar = (k % polar_every == 0) or case.get("polar_prefix_only", False)
        data = evaluate_target(
            model, kernels, windows, np.asarray(target), n_ang, n_rad, want_polar
        )
        physical = data["prefix_physical"]
        if case.get("polar_prefix_only", False):
            physical = data["prefix_polar"]
        ref = data["reference"]
        fryklund = fryklund_line(case, model, np.asarray(target), windows[-1])
        row = {
            "case": case["name"],
            "label": label,
            "x1": float(target[0]),
            "x2": float(target[1]),
            "delta": float(label),
            "delta_over_sqrt_t_leaf": float(label) / math.sqrt(windows[-1]),
            "u_reference": ref,
            "u_smooth_total": data["smooth_total"],
            "prefix_physical": physical,
            "prefix_extended": data["prefix_extended"],
            "prefix_legacy": data["prefix_legacy"],
            "prefix_polar": data["prefix_polar"],
            "err_physical": abs(data["smooth_total"] + physical - ref) / abs(ref),
            "err_extended": abs(
                data["smooth_total"] + data["prefix_extended"] - ref
            )
            / abs(ref),
            "err_legacy": abs(data["smooth_total"] + data["prefix_legacy"] - ref)
            / abs(ref),
            "prefix_fryklund": fryklund["value"] if fryklund else "",
            "err_fryklund": (
                abs(data["smooth_total"] + fryklund["value"] - ref) / abs(ref)
                if fryklund
                else ""
            ),
            "fryklund_defined": fryklund["defined"] if fryklund else "",
            "fryklund_r": fryklund["r"] if fryklund else "",
            "fryklund_minus_physical": (
                abs(fryklund["value"] - physical) if fryklund else ""
            ),
            "fryklund_remainder_pred": fryklund["remainder"] if fryklund else "",
            "fryklund_remainder_residual": (
                abs(physical - fryklund["value"] - fryklund["remainder"])
                if fryklund
                else ""
            ),
            "prefix_separable_vs_polar": (
                abs(data["prefix_physical"] - data["prefix_polar"])
                if (data["prefix_polar"] is not None and not case.get(
                    "polar_prefix_only", False
                ))
                else ""
            ),
            "n_v_nodes": data["n_v_nodes"],
        }
        rows.append(row)
    return rows


def check_smoothness(model, kernels, windows, target, n_ang, n_rad, orders):
    """Check 6: tensor-Gauss order convergence of the split pieces on one leaf."""
    leaf = model.leaf_index(target)
    polygon = model.polygons[leaf]
    dens = model.dens[leaf]
    names = ["full", "coarse", "prefix"] + [
        f"shell{idx}" for idx in range(len(windows) - 1)
    ]
    exact = {
        name: polar_polygon_potential(
            polygon, target, dens, kernels[name], 4 * n_ang, 2 * n_rad
        )
        for name in names
    }
    rows = []
    for order in orders:
        row = {"order": order, "nodes": order * order}
        for name in names:
            value = tensor_gauss_polygon_potential(
                polygon, target, dens, kernels[name], order
            )
            denom = abs(exact[name]) if exact[name] else 1.0
            row[f"relerr_{name}"] = abs(value - exact[name]) / denom
        rows.append(row)
    return rows, {"polar_values": exact, "leaf": leaf}


def check_u_quadrature(model, windows, targets, tolerance=1e-13):
    """Check 7: ``v``-node count needed by the separable prefix per target."""
    t_leaf = windows[-1]
    rows = []
    for label, target in targets:
        target = np.asarray(target)
        rects = model.clipped_rects
        if rects is None:
            continue
        best = 0.0
        for rect, dens in zip(rects, model.dens, strict=False):
            value, _ = separable_prefix_rect(
                rect, target, dens, t_leaf, n_panel=34, order=24
            )
            best += value
        record = {"delta": float(label), "reference": best, "nodes_for_tol": ""}
        for order in (8, 10, 12, 16, 20):
            for n_panel in (6, 10, 14, 18, 22, 26, 30):
                total = 0.0
                nodes = 0
                for rect, dens in zip(rects, model.dens, strict=False):
                    value, count = separable_prefix_rect(
                        rect, target, dens, t_leaf, n_panel=n_panel, order=order
                    )
                    total += value
                    nodes = max(nodes, count)
                err = abs(total - best) / max(abs(best), 1e-300)
                record[f"relerr_o{order}_p{n_panel}"] = err
                if err < tolerance and not record["nodes_for_tol"]:
                    record["nodes_for_tol"] = nodes
        rows.append(record)
    return rows


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def build_cases(n_targets):
    """Return the four leaf-geometry cases with their targets."""
    deltas = np.logspace(math.log10(0.0022), math.log10(0.58), n_targets)
    cases = []

    interior = np.array(
        [
            [0.02, -0.07],
            [-0.19, 0.11],
            [0.23, 0.17],
            [-0.05, -0.23],
            [0.13, -0.29],
            [-0.27, -0.03],
            [0.07, 0.27],
            [-0.11, 0.21],
            [0.29, -0.13],
            [-0.23, -0.17],
        ]
    )
    interior_targets = [
        (float(min(ROOT_HALF - abs(p[0]), ROOT_HALF - abs(p[1]))), p)
        for p in interior[:n_targets]
    ]
    cases.append(
        {
            "name": "case1_interior_global",
            "half_planes": [],
            "axis_bounds": [],
            "piecewise": False,
            "targets": interior_targets,
        }
    )
    cases.append(
        {
            "name": "case1_interior_piecewise",
            "half_planes": [],
            "axis_bounds": [],
            "piecewise": True,
            "targets": interior_targets,
        }
    )

    cut = 0.3
    halfplane_targets = [(float(d), np.array([cut - d, -0.07])) for d in deltas]
    halfplane_wall = {"axis": 0, "value": cut, "unique_closest": True}
    cases.append(
        {
            "name": "case2_halfplane",
            "half_planes": [((cut, 0.0), (1.0, 0.0))],
            "axis_bounds": [(0, cut)],
            "piecewise": False,
            "fryklund": halfplane_wall,
            "targets": halfplane_targets,
        }
    )
    cases.append(
        {
            "name": "case2_halfplane_piecewise",
            "half_planes": [((cut, 0.0), (1.0, 0.0))],
            "axis_bounds": [(0, cut)],
            "piecewise": True,
            "fryklund": halfplane_wall,
            "targets": halfplane_targets,
        }
    )

    apex = np.array([0.3, 0.1])
    diag = np.array([-1.0, -1.0]) / math.sqrt(2.0)
    wedge_geometry = {
        "half_planes": [(apex, (1.0, 0.0)), (apex, (0.0, 1.0))],
        "axis_bounds": [(0, float(apex[0])), (1, float(apex[1]))],
        "piecewise": False,
    }
    cases.append(
        {
            "name": "case3_wedge90",
            **wedge_geometry,
            "fryklund": {
                "axis": 0,
                "value": float(apex[0]),
                "unique_closest": False,
            },
            "targets": [(float(d), apex + d * diag) for d in deltas],
        }
    )

    # A second, short sweep on one face of the same wedge, far enough from the
    # apex (at least 4 sqrt(t_L), here 0.55 = 8.8 sqrt(t_L)) that the closest
    # boundary point is unique and the other face is outside the window.
    face_offset = 0.55
    face_deltas = np.logspace(math.log10(0.0022), math.log10(0.30), 6)
    cases.append(
        {
            "name": "case3_wedge90_face",
            **wedge_geometry,
            "fryklund": {
                "axis": 0,
                "value": float(apex[0]),
                "unique_closest": True,
            },
            "targets": [
                (
                    float(d),
                    np.array([float(apex[0]) - d, float(apex[1]) - face_offset]),
                )
                for d in face_deltas
            ],
        }
    )

    angle = 2.0 * math.pi / 3.0
    normal2 = np.array([math.cos(angle), math.sin(angle)])
    inward = -(np.array([1.0, 0.0]) + normal2)
    inward = inward / float(np.hypot(inward[0], inward[1]))
    cases.append(
        {
            "name": "case4_wedge60",
            "half_planes": [(apex, (1.0, 0.0)), (apex, normal2)],
            "axis_bounds": None,
            "piecewise": False,
            "polar_prefix_only": True,
            "fryklund": {
                "axis": 0,
                "value": float(apex[0]),
                "unique_closest": False,
            },
            "targets": [(float(d), apex + d * inward) for d in deltas],
        }
    )
    return cases


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


SWEEP_CASE_ORDER = (
    "case2_halfplane",
    "case2_halfplane_piecewise",
    "case3_wedge90",
    "case3_wedge90_face",
    "case4_wedge60",
)
"""Cut-geometry sweeps that get a panel in the boundary-sweep figure."""

SWEEP_COLUMNS = (
    ("err_physical", "o-", "RKE + DMK (physical-side windowed prefix)"),
    ("err_extended", "s--", "box code with extended source"),
    ("err_legacy", "^:", "plain DMK, interior series"),
)
"""The three columns that do not depend on a closest boundary point."""

FRYKLUND_LABEL = "DMK line, Fryklund Lemma 4.5"
FRYKLUND_LABEL_AMBIGUOUS = (
    "DMK line, Fryklund Lemma 4.5\n(closest point not unique; hollow markers)"
)


def make_plots(out, case_rows, smooth_rows):
    """Write the sweep and smoothness plots, if matplotlib is importable."""
    plt = get_pyplot()
    if plt is None:
        return []
    written = []

    sweeps = [name for name in SWEEP_CASE_ORDER if name in case_rows]
    if sweeps:
        ncols = min(3, len(sweeps))
        nrows = (len(sweeps) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False
        )
        panels = [ax for row in axes for ax in row]
        for ax, name in zip(panels, sweeps, strict=False):
            rows = case_rows[name]
            delta = [r["delta_over_sqrt_t_leaf"] for r in rows]
            for key, style, label in SWEEP_COLUMNS:
                values = [max(r[key], 1e-18) for r in rows]
                ax.loglog(delta, values, style, label=label, markersize=4)
            fryklund = [r for r in rows if r["prefix_fryklund"] != ""]
            if fryklund:
                unique = bool(fryklund[0]["fryklund_defined"])
                ax.loglog(
                    [r["delta_over_sqrt_t_leaf"] for r in fryklund],
                    [max(r["err_fryklund"], 1e-18) for r in fryklund],
                    "D-." if unique else "D--",
                    label=FRYKLUND_LABEL if unique else FRYKLUND_LABEL_AMBIGUOUS,
                    markersize=5,
                    **({} if unique else {"markerfacecolor": "none"}),
                )
            ax.set_xlabel("distance to the cut / sqrt(t_L)")
            ax.set_ylabel("relative error of the split total")
            ax.set_title(name)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        for ax in panels[len(sweeps):]:
            ax.axis("off")
        fig.tight_layout()
        path = out / "experiment_c_boundary_sweep.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    if smooth_rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        orders = [r["order"] for r in smooth_rows]
        for key in smooth_rows[0]:
            if not key.startswith("relerr_"):
                continue
            values = [max(r[key], 1e-17) for r in smooth_rows]
            ax.semilogy(orders, values, "o-", label=key[len("relerr_") :])
        ax.set_xlabel("tensor-Gauss order per direction")
        ax.set_ylabel("relative error on the target's own leaf")
        ax.set_title("smoothness of the split pieces")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out / "experiment_c_smooth_convergence.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main():
    """Run experiment C and write CSV, JSON and PNG artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n-ang", type=int, default=16)
    parser.add_argument("--n-rad", type=int, default=16)
    parser.add_argument("--targets", type=int, default=10)
    parser.add_argument("--polar-every", type=int, default=3)
    parser.add_argument(
        "--no-refine", action="store_true", help="skip the quadrature self-check"
    )
    args = parser.parse_args()

    started = time.time()
    out = output_dir(args.out)

    n_side = 2 ** (N_LEVELS - 1)
    heights = [2.0 * ROOT_HALF * 2.0**-level for level in range(N_LEVELS)]
    windows = [(h / THETA) ** 2 for h in heights]
    kernels = make_kernels(windows)

    rows1, summary1 = check_kernel_identity(windows, kernels)
    write_csv(out / "experiment_c_kernel_split.csv", rows1)

    cases = build_cases(args.targets)
    case_rows = {}
    case_summary = {}
    refine_summary = {}
    smooth_rows = []
    smooth_summary = {}
    uquad_rows = []

    for case in cases:
        model = Model(
            case["half_planes"], case["axis_bounds"], case["piecewise"], n_side
        )
        rows = sweep_case(
            case,
            model,
            kernels,
            windows,
            case["targets"],
            args.n_ang,
            args.n_rad,
            args.polar_every,
        )
        case_rows[case["name"]] = rows
        write_csv(out / f"experiment_c_{case['name']}.csv", rows)
        separable_gaps = [
            r["prefix_separable_vs_polar"]
            for r in rows
            if r["prefix_separable_vs_polar"] != ""
        ]
        case_summary[case["name"]] = {
            "n_targets": len(rows),
            "physical_area": float(sum(polygon_area(p) for p in model.polygons)),
            "max_rel_err_physical": max(r["err_physical"] for r in rows),
            "min_rel_err_physical": min(r["err_physical"] for r in rows),
            "max_rel_err_extended": max(r["err_extended"] for r in rows),
            "max_rel_err_legacy": max(r["err_legacy"] for r in rows),
            "rel_err_legacy_at_closest": rows[0]["err_legacy"],
            "rel_err_legacy_at_farthest": rows[-1]["err_legacy"],
            "rel_err_physical_at_closest": rows[0]["err_physical"],
            "max_abs_separable_minus_polar_prefix": (
                max(separable_gaps) if separable_gaps else None
            ),
            "delta_over_sqrt_t_leaf_range": [
                rows[0]["delta_over_sqrt_t_leaf"],
                rows[-1]["delta_over_sqrt_t_leaf"],
            ],
        }

        fryklund_rows = [r for r in rows if r["prefix_fryklund"] != ""]
        if fryklund_rows:
            errors = [r["err_fryklund"] for r in fryklund_rows]
            case_summary[case["name"]].update(
                {
                    "fryklund_closest_point_unique": bool(
                        fryklund_rows[0]["fryklund_defined"]
                    ),
                    "max_rel_err_fryklund": max(errors),
                    "median_rel_err_fryklund": float(np.median(errors)),
                    "rel_err_fryklund_at_closest": fryklund_rows[0]["err_fryklund"],
                    "rel_err_fryklund_at_farthest": fryklund_rows[-1]["err_fryklund"],
                    "max_abs_fryklund_minus_physical": max(
                        r["fryklund_minus_physical"] for r in fryklund_rows
                    ),
                    "abs_fryklund_minus_physical_at_closest": fryklund_rows[0][
                        "fryklund_minus_physical"
                    ],
                    "max_abs_cubic_remainder_residual": max(
                        r["fryklund_remainder_residual"] for r in fryklund_rows
                    ),
                    "abs_cubic_remainder_at_closest": fryklund_rows[0][
                        "fryklund_remainder_pred"
                    ],
                }
            )

        if not args.no_refine:
            coarse_rows = sweep_case(
                case,
                model,
                kernels,
                windows,
                case["targets"],
                max(args.n_ang - 6, 6),
                max(args.n_rad - 6, 6),
                10**6,
            )
            gaps = [
                abs(a["u_reference"] - b["u_reference"]) / abs(a["u_reference"])
                for a, b in zip(rows, coarse_rows, strict=False)
            ]
            smooth_gaps = [
                abs(a["u_smooth_total"] - b["u_smooth_total"])
                / abs(a["u_reference"])
                for a, b in zip(rows, coarse_rows, strict=False)
            ]
            refine_summary[case["name"]] = {
                "reference_self_convergence": max(gaps) if gaps else None,
                "smooth_total_self_convergence": (
                    max(smooth_gaps) if smooth_gaps else None
                ),
                "n_ang_pair": [args.n_ang, max(args.n_ang - 6, 6)],
                "n_rad_pair": [args.n_rad, max(args.n_rad - 6, 6)],
            }

        if case["name"] == "case2_halfplane":
            smooth_rows, smooth_summary = check_smoothness(
                model,
                kernels,
                windows,
                np.asarray(case["targets"][-1][1]),
                args.n_ang,
                args.n_rad,
                [4, 6, 8, 12, 16, 24, 32, 48],
            )
            write_csv(out / "experiment_c_smooth_quadrature.csv", smooth_rows)
            uquad_rows = check_u_quadrature(model, windows, case["targets"])
            write_csv(out / "experiment_c_uquad_nodes.csv", uquad_rows)

    plots = make_plots(out, case_rows, smooth_rows)

    payload = {
        "experiment": "C (leaf residual replacement), 2D Laplace",
        "exploratory": True,
        "environment": environment_summary(),
        "configuration": {
            "theta": THETA,
            "levels": N_LEVELS,
            "leaves_per_side": n_side,
            "box_sides": heights,
            "windows_t_ell": windows,
            "sqrt_t_leaf": math.sqrt(windows[-1]),
            "density_degree": 3,
            "n_ang": args.n_ang,
            "n_rad": args.n_rad,
            "n_targets_per_case": args.targets,
        },
        "check1_kernel_identity": summary1,
        "cases": case_summary,
        "reference_quadrature_self_check": refine_summary,
        "check8_fryklund_line": {
            "source": (
                "experiment_e_baseline.fryklund_VL, Lemma 4.5 of Fryklund, "
                "Greengard, Jiang and Potter (2024), reused unchanged"
            ),
            "delta": windows[-1],
            "kappa_b": 0.0,
            "jet": (
                "exact Taylor jet of the target leaf's polynomial at the "
                "target, in the (tangent, inward normal) frame of the wall"
            ),
            "wall_choice": (
                "the first edge y_1 = 0.3 in every cut case; on a wedge "
                "bisector no boundary point is the unique closest one and "
                "those rows carry fryklund_defined = False"
            ),
            "remainder": (
                "flat_wall_cubic_remainder is the degree-three group the "
                "lemma omits at a flat wall; for a cubic density it is the "
                "whole remainder, of size delta^{5/2}, so "
                "fryklund_remainder_residual should be roundoff wherever the "
                "wall is flat, the closest point unique and the density one "
                "global polynomial"
            ),
        },
        "check6_smoothness": {
            "rows": smooth_rows,
            "polar_reference_values": smooth_summary.get("polar_values"),
        },
        "check7_u_quadrature": uquad_rows,
        "plots": plots,
        "wall_seconds": time.time() - started,
    }
    write_json(out / "experiment_c_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str)[:6000])


if __name__ == "__main__":
    main()
