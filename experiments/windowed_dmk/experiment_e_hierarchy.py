"""Experiment E stages: ``W_0``, plane-wave scale shells and the 2D leaf closure.

This module assembles the three stages of the windowed telescoping decomposition

    K = W_0 + sum_{l=0}^{L-1} D_l + chi_0(.; t_L),
    D_l(r) = chi_0(r; t_l) - chi_0(r; t_{l+1}),
    chi_0(r; t) = E_1(r^2 / (4 t)) / 2,

on a uniform quadtree of ``L`` levels whose leaves carry physical ``Piece``
objects and a per-leaf tensor Lagrange density proxy (``LeafPoly``).  Every
data structure, window, kernel, plane-wave grid and tree helper comes from
``experiment_e_common``; nothing here re-defines that vocabulary.

Stages
------
1. ``w0_stage``: ``W_0 = -log r - chi_0(r; t_0)`` is smooth on the scale
   ``sqrt(t_0)``, so its potential is taken by direct tensor Gauss quadrature
   over every leaf's pieces, with the order fixed by self-convergence.
2. ``shell_stage``: one level's shell by plane waves.  Per box at level ``l``
   the physical source is transformed to ``rho_B(k) = int_B e^{-i k . (y - c_B)}
   rho(y) dy`` (closed form for box pieces, quadrature otherwise), colleague
   boxes are translated in by the phase ``e^{i k . (c_B' - c_B)}``, the shell
   multiplier ``int_{t_{l+1}}^{t_l} e^{-u |k|^2} du`` is applied, and the
   inverse transform is evaluated directly at the targets.
3. ``closure_stage``: ``chi_0(.; t_L)`` over the colleague leaves, by the
   separable ``v = sqrt(u)`` heat-time quadrature on axis-aligned box pieces and
   by target-centred polar quadrature on polygon and curved pieces.

Every stage returns ``u``, ``grad u`` and ``Delta u`` analytically:

* plane waves differentiate to ``i k`` and ``-|k|^2`` multipliers;
* ``W_0`` differentiates under the quadrature, with
  ``W_0'(r) / r = -(1 - e^{-z}) / r^2 = -(1 - e^{-z}) / (4 t_0 z)`` and
  ``Delta W_0(r) = -e^{-z} / (2 t_0)``, ``z = r^2 / (4 t_0)``;
* the closure differentiates in closed form, using
  ``-d/dr chi_0(r; t) / (2 pi) = e^{-z} / (2 pi r)`` for the gradient and the
  distributional identity ``Delta [chi_0(.; t) * rho / (2 pi)] = G_t * rho -
  rho 1_Omega`` for the Laplacian, whose Gaussian term is the same closed-form
  box moment as the separable rule and the same polar rule on other pieces.
  On a box piece the gradient comes from the separable rule; on a polygon or
  curved piece it is the geometry module's vector polar fan, which shares the
  angular substitution, the panel rule and the radial moments with the scalar
  fan and carries a curved piece's arcs themselves.  The ``1_Omega`` factor is
  the exact domain predicate the geometry module attaches to every clipped
  piece, never a crossing test against a sampled chord loop.

``assemble`` returns the per-stage fields and their total; ``residual_report``
compares each stage's ``-Delta`` with the closed-form heat-smoothed right-hand
sides of the plan,

    -Delta[(2 pi)^{-1} W_0 * rho]           = G_{t_0} * rho,
    -Delta[(2 pi)^{-1} D_l * rho]           = (G_{t_{l+1}} - G_{t_l}) * rho,
    -Delta[(2 pi)^{-1} chi_0(.; t_L) * rho] = rho - G_{t_L} * rho,

which sum to ``rho`` inside ``Omega`` and to ``0`` outside.

Reuse
-----
``experiment_c_leaf_residual`` supplies the target-centred polar quadrature
(``polar_polygon_potential``, ``radial_moments``, ``ray_coefficients``,
``RadialKernel``) and the separable rectangle prefix
(``separable_prefix_rect``, used here as the cross-check reference);
``experiment_a_separability`` supplies the graded ``v``-quadrature whose panel
breakpoints sit at the target-to-face transition scales.  Polar and separable
routines from those modules expand the density in monomials about the *global*
origin, which is badly conditioned for a small leaf far from it, so every call
below is made in coordinates shifted to the source leaf's centre; the answer is
translation invariant.

If ``experiment_e_geometry`` is importable and exports ``plane_wave_coeffs``,
``prefix_potential`` or ``polar_reference``, those are used in preference to the
local implementations, which keep the same signatures.

Exploratory campaign code: no warm-up, no repetition statistics, no claim that
any rule here is optimal.

Self-test
---------
    python experiments/windowed_dmk/experiment_e_hierarchy.py [--out DIR] [--quick]

with ``PYTHONPATH`` pointing at the directory holding these scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.special import spherical_jn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import experiment_e_common as ec  # noqa: E402
from experiment_a_separability import v_quadrature  # noqa: E402
from experiment_c_leaf_residual import (  # noqa: E402
    RadialKernel,
    polar_polygon_potential,
    prefix_smooth,
    radial_moments,
    ray_coefficients,
    separable_prefix_rect,
)

try:  # the geometry module of the same wave may not exist yet
    import experiment_e_geometry as _geometry
except Exception:  # pragma: no cover - absence is the expected case today
    _geometry = None


__all__ = [
    "StageField",
    "assemble",
    "residual_report",
    "w0_stage",
    "shell_stage",
    "closure_stage",
    "plane_wave_coeffs",
    "prefix_potential",
    "polar_reference",
    "heat_smoothed_proxy",
    "proxy_density",
    "build_box_leaves",
    "case_b_probes",
    "polar_reference_total",
    "shell_polar_reference",
    "run_self_test",
    "main",
]


DEFAULT_W0_ORDER = 18
"""Tensor Gauss order per leaf for the ``W_0`` stage (10 points per direction).

Fixed by the ``w0_order_convergence`` check: at ``L = 3``, where the leaf side
equals ``sqrt(t_0)`` and the rule is hardest, order 18 already reproduces the
order-40 value to ``6e-17`` in ``u`` and ``3e-16`` in ``Delta u``.
"""

DEFAULT_POLAR_ANG = 24
"""Angular Gauss order per dyadic panel of the target-centred polar rule."""

DEFAULT_POLAR_RAD = 20
"""Radial Gauss order per graded panel of the polar rule's smooth moments."""

DEFAULT_V_ORDER = 16
"""Gauss order per graded panel of the separable ``v``-quadrature."""

DEFAULT_V_LEVELS = 26
"""Dyadic grading depth of the separable ``v``-quadrature."""

HEAT_CUTOFF = 13.0
"""Sources beyond ``HEAT_CUTOFF sqrt(t)`` of a target are dropped (below 1e-18)."""


# ---------------------------------------------------------------------------
# stage fields
# ---------------------------------------------------------------------------


@dataclass
class StageField:
    """Potential, gradient and Laplacian of one stage at a set of physical targets."""

    u: np.ndarray
    grad: np.ndarray
    lap: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "StageField":
        """A zero field for ``n`` targets."""
        return cls(np.zeros(n), np.zeros((n, 2)), np.zeros(n))

    def __add__(self, other: "StageField") -> "StageField":
        """Sum two stage fields componentwise."""
        return StageField(
            self.u + other.u, self.grad + other.grad, self.lap + other.lap
        )

    def as_dict(self) -> dict:
        """Plain lists, for the summary JSON."""
        return {
            "u": self.u.tolist(),
            "grad": self.grad.tolist(),
            "lap": self.lap.tolist(),
        }


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _resolve(name: str, fallback: Callable) -> Callable:
    """Prefer ``experiment_e_geometry``'s implementation of ``name`` when present."""
    if _geometry is not None:
        candidate = getattr(_geometry, name, None)
        if callable(candidate):
            return candidate
    return fallback


def _as_targets(points) -> np.ndarray:
    """Physical targets as a contiguous ``(n, 2)`` float array."""
    return np.ascontiguousarray(
        np.atleast_2d(np.asarray(points, dtype=np.float64)).reshape(-1, 2)
    )


def _leaf_index_map(leaves: Sequence, tree: ec.QuadTree) -> dict:
    """Map ``(i, j)`` finest-level grid indices to leaves."""
    out = {}
    for leaf in leaves:
        out[tree.indices_of(leaf.box)] = leaf
    return out


def _piece_is_full_box(piece, box: ec.Box) -> bool:
    """True when a ``box`` piece covers exactly the proxy's own box."""
    if piece.kind != "box":
        return False
    a1, b1, a2, b2 = piece.bounding_box()
    c1, c2 = box.center
    tol = 1e-12 * max(1.0, box.half)
    return (
        abs(a1 - (c1 - box.half)) < tol
        and abs(b1 - (c1 + box.half)) < tol
        and abs(a2 - (c2 - box.half)) < tol
        and abs(b2 - (c2 + box.half)) < tol
    )


def _piece_loop(piece, samples: int = 33) -> np.ndarray:
    """Closed vertex loop of a piece, densified along arcs for curved pieces."""
    if piece.kind != "curved" or not piece.edges:
        return piece.vertices
    pts = []
    for edge in piece.edges:
        s = np.linspace(edge.s0, edge.s1, samples)[:-1]
        pts.append(edge.curve(s))
    return np.vstack(pts) if pts else piece.vertices


def _points_in_loop(loop: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crossing-number test of points against a closed polygon loop."""
    pts = _as_targets(points)
    inside = np.zeros(pts.shape[0], dtype=bool)
    n = loop.shape[0]
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        cond = (y1 > pts[:, 1]) != (y2 > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xcross = x1 + (pts[:, 1] - y1) * (x2 - x1) / (y2 - y1)
        hit = cond & (pts[:, 0] < xcross)
        inside ^= hit
    return inside


def _inside_piece(piece, points: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside a piece, exactly for every piece kind.

    A box piece is the closed rectangle and a polygon piece is the crossing test
    against its own (exact) vertex loop.  A curved piece must *not* be tested
    against a sampled loop: its vertices are the chord polygon through the edge
    endpoints, and the chord sagitta on a cut leaf of the star is larger than the
    interface probe offsets this study places at ``10^{-4}`` and ``10^{-5}``
    times ``h_L``, so a loop test puts such probes on the wrong side of the
    boundary.  The geometry module therefore attaches ``inside_exact`` (the
    domain's own predicate restricted to the owning leaf box) to every clipped
    piece, and that is what a curved piece is tested against.
    """
    pts = _as_targets(points)
    if piece.kind == "box":
        a1, b1, a2, b2 = piece.bounding_box()
        tol = 1e-13 * max(1.0, b1 - a1)
        return (
            (pts[:, 0] >= a1 - tol)
            & (pts[:, 0] <= b1 + tol)
            & (pts[:, 1] >= a2 - tol)
            & (pts[:, 1] <= b2 + tol)
        )
    if piece.kind == "curved":
        exact = getattr(piece, "inside_exact", None)
        if exact is None:
            raise ValueError(
                "a curved piece needs the exact membership predicate attached by "
                "Domain.leaves; a sampled chord loop cannot resolve the probe "
                "offsets of this study"
            )
        return np.asarray(exact(pts), dtype=bool)
    return _points_in_loop(_piece_loop(piece), pts)


def proxy_density(
    leaves: Sequence, tree: ec.QuadTree, inside: Callable | None = None
) -> Callable:
    """Callable evaluating the piecewise per-leaf proxy, zero outside ``Omega``.

    This is the density the hierarchy actually convolves, so it is also the
    density the residual right-hand sides must use; the exact ``rho`` differs
    from it by the proxy's interpolation error.

    ``inside`` overrides the membership test.  The default reduces
    ``_inside_piece`` over the leaf's pieces, which is what ``closure_stage``
    subtracts, so a residual built on the default cannot see a geometric
    misclassification: the same wrong indicator would appear on both sides and
    cancel.  ``residual_report`` therefore passes the domain's own analytic
    predicate here, which makes the right-hand side independent of the stage.
    """
    index = _leaf_index_map(leaves, tree)
    level = tree.levels
    n_side = tree.n_side(level)
    h = tree.h(level)
    lo = np.asarray(tree.root.center, dtype=np.float64) - tree.root.half

    def _rho(points):
        """Proxy value at physical points, zero where no leaf piece covers them."""
        pts = _as_targets(points)
        out = np.zeros(pts.shape[0], dtype=np.float64)
        ii = np.clip(
            np.floor((pts[:, 0] - lo[0]) / h).astype(np.int64), 0, n_side - 1
        )
        jj = np.clip(
            np.floor((pts[:, 1] - lo[1]) / h).astype(np.int64), 0, n_side - 1
        )
        flat = ii * n_side + jj
        for key in np.unique(flat):
            leaf = index.get((int(key) // n_side, int(key) % n_side))
            if leaf is None or leaf.poly is None:
                continue
            sel = np.nonzero(flat == key)[0]
            if inside is not None:
                mask = np.asarray(inside(pts[sel]), dtype=bool)
            else:
                mask = np.zeros(sel.size, dtype=bool)
                for piece in leaf.pieces:
                    mask |= _inside_piece(piece, pts[sel])
            out[sel] = np.where(mask, leaf.poly.eval(pts[sel]), 0.0)
        return out

    return _rho


# ---------------------------------------------------------------------------
# axis Gaussian moments with derivatives (local: the common module has none)
# ---------------------------------------------------------------------------


def _axis_moments(
    x, a: float, b: float, sigma, k_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """``F_i = int_a^b y^i e^{-(y-x)^2/(2 s^2)} dy`` and ``dF_i/dx``, cancellation-free.

    ``x`` (the target offset) and ``sigma`` broadcast against each other; the
    returned arrays have shape ``(k_max + 1,) + broadcast_shape``.  Both are
    assembled from the standardized moments ``m_k = int z^k e^{-z^2/2} dz``, so
    the derivative is ``dF_i/dx = sum_k C(i, k) x^{i-k} s^k m_{k+1}`` and never
    forms the cancelling difference ``(F_{i+1} - x F_i) / s^2``.  This is the
    derivative-carrying variant of the moment helper used by the leaf-residual
    study's separable rectangle rule; the common module exposes no such helper.
    """
    from scipy.special import erf, erfc

    x_arr, sig = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    )
    za = (float(a) - x_arr) / sig
    zb = (float(b) - x_arr) / sig
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
    for k in range(2, k_max + 2):
        bracket = zb ** (k - 1) * eb - za ** (k - 1) * ea
        moments.append((k - 1) * moments[k - 2] - bracket)
    shape = (k_max + 1,) + x_arr.shape
    f_vals = np.zeros(shape, dtype=np.float64)
    d_vals = np.zeros(shape, dtype=np.float64)
    for i in range(k_max + 1):
        acc_f = np.zeros(x_arr.shape, dtype=np.float64)
        acc_d = np.zeros(x_arr.shape, dtype=np.float64)
        for k in range(i + 1):
            coeff = math.comb(i, k) * x_arr ** (i - k) * sig**k
            acc_f += coeff * moments[k]
            acc_d += coeff * moments[k + 1]
        f_vals[i] = sig * acc_f
        d_vals[i] = acc_d
    return f_vals, d_vals


def _shifted_monomials(poly: ec.LeafPoly) -> np.ndarray:
    """Monomial coefficients of the proxy about its own leaf centre (well conditioned)."""
    return poly.monomial_coeffs(origin=poly.box.center)


def _box_heat_integral(rect, offset, dens: np.ndarray, t: float) -> np.ndarray:
    """``int_rect G_t(|x - y|) rho(y) dy`` in closed form, physical units.

    ``rect`` and ``offset`` are already shifted so the proxy's monomial origin is
    at zero; ``offset`` has shape ``(n, 2)`` and the result shape ``(n,)``.
    """
    a1, b1, a2, b2 = rect
    if b1 <= a1 or b2 <= a2:
        return np.zeros(offset.shape[0], dtype=np.float64)
    sigma = math.sqrt(2.0 * float(t))
    f1, _ = _axis_moments(offset[:, 0], a1, b1, sigma, dens.shape[0] - 1)
    f2, _ = _axis_moments(offset[:, 1], a2, b2, sigma, dens.shape[1] - 1)
    acc = np.zeros(offset.shape[0], dtype=np.float64)
    for i in range(dens.shape[0]):
        for j in range(dens.shape[1]):
            if dens[i, j]:
                acc += dens[i, j] * f1[i] * f2[j]
    return acc / (4.0 * math.pi * float(t))


def _separable_box_closure(
    rect, offset, dens: np.ndarray, t_leaf: float, v_nodes, v_weights
) -> tuple[float, np.ndarray]:
    """Closure potential and gradient over a rectangle by the separable ``v``-rule.

    ``rect = (a1, b1, a2, b2)`` and the single target ``offset`` are in the
    proxy's shifted coordinates.  The rule is the ``u = v^2`` substitution of the
    separability proposition, so the integrand is analytic in ``v`` for every
    target position including targets on a face or corner.
    """
    a1, b1, a2, b2 = rect
    if b1 <= a1 or b2 <= a2:
        return 0.0, np.zeros(2)
    sigma = math.sqrt(2.0) * v_nodes
    f1, d1 = _axis_moments(float(offset[0]), a1, b1, sigma, dens.shape[0] - 1)
    f2, d2 = _axis_moments(float(offset[1]), a2, b2, sigma, dens.shape[1] - 1)
    acc = np.zeros(v_nodes.size, dtype=np.float64)
    acc_x = np.zeros(v_nodes.size, dtype=np.float64)
    acc_y = np.zeros(v_nodes.size, dtype=np.float64)
    for i in range(dens.shape[0]):
        for j in range(dens.shape[1]):
            if dens[i, j]:
                acc += dens[i, j] * f1[i] * f2[j]
                acc_x += dens[i, j] * d1[i] * f2[j]
                acc_y += dens[i, j] * f1[i] * d2[j]
    weight = v_weights / (2.0 * math.pi * v_nodes)
    value = float(np.sum(weight * acc))
    grad = np.array([float(np.sum(weight * acc_x)), float(np.sum(weight * acc_y))])
    return value, grad


# ---------------------------------------------------------------------------
# polar quadrature on a piece (shifted to the source leaf's centre)
# ---------------------------------------------------------------------------


def _polar_kernels(t_leaf: float) -> dict:
    """Radial kernels for the closure: potential, Gaussian and gradient profiles."""
    root = math.sqrt(float(t_leaf))
    return {
        "prefix": RadialKernel(
            "prefix", 1.0, lambda r, a=t_leaf: prefix_smooth(r, a), root
        ),
        "heat": RadialKernel(
            "heat", 0.0, lambda r, a=t_leaf: ec.heat_kernel(r, a), root
        ),
        "grad": RadialKernel(
            "grad",
            0.0,
            lambda r, a=t_leaf: np.exp(-np.asarray(r) ** 2 / (4.0 * a))
            / (2.0 * math.pi * np.asarray(r)),
            root,
        ),
    }


def _MISSING_VECTOR_FAN(*args, **kwargs):  # noqa: N802 - sentinel, never called
    """Sentinel standing for an absent ``experiment_e_geometry.polar_gradient``."""
    raise RuntimeError("sentinel")


def _polar_gradient(
    vertices: np.ndarray, target: np.ndarray, dens: np.ndarray, kernel, n_ang, n_rad
) -> np.ndarray:
    """Gradient of ``int_P w(|x - y|) rho(y) dy`` by the target-centred triangle fan.

    Straight-sided pieces only: the caller passes an exact vertex loop.  The
    production path for a curved piece is ``experiment_e_geometry.polar_gradient``,
    which carries the arcs themselves; this routine is kept as the independent
    cross-check that the two agree on a polygon.

    ``kernel`` is the radial profile ``-w'(r)``, so the gradient is
    ``int dphi dhat(phi) int_0^{R(phi)} kernel(r) rho(x + r dhat) r^{k+1} dr``
    summed over the ray coefficients; the angular substitution and the dyadic
    panels are those of the leaf-residual study's scalar routine.
    """
    from experiment_c_leaf_residual import _dyadic_breakpoints, _panel_rule

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] < 3:
        return np.zeros(2)
    target = np.asarray(target, dtype=np.float64)
    k_max = dens.shape[0] + dens.shape[1] - 2
    total = np.zeros(2)
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
        total += sign * (weights * inner) @ dirs
    return total


# ---------------------------------------------------------------------------
# plane-wave coefficients
# ---------------------------------------------------------------------------


_FACTOR_CACHE: dict = {}


def _legendre_plane_wave_factors(k1: np.ndarray, half: float, q: int) -> np.ndarray:
    """``I_m(kappa) = int_{-1}^{1} e^{-i kappa xi} L_m(xi) dxi = 2 (-i)^m j_m(kappa)``.

    Returned array has shape ``(q, k1.size)`` with ``kappa = k1 half``.  The
    parity ``j_m(-z) = (-1)^m j_m(z)`` is applied explicitly so negative
    wavenumbers are handled without relying on the library's branch.
    """
    k1 = np.asarray(k1, dtype=np.float64)
    key = (float(half), int(q), int(k1.size), k1.tobytes())
    hit = _FACTOR_CACHE.get(key)
    if hit is not None:
        return hit
    kappa = k1 * float(half)
    mag = np.abs(kappa)
    sign = np.sign(kappa)
    out = np.empty((int(q), kappa.size), dtype=np.complex128)
    for m in range(int(q)):
        out[m] = 2.0 * ((-1j) ** m) * spherical_jn(m, mag) * (sign**m)
    if len(_FACTOR_CACHE) < 32:
        _FACTOR_CACHE[key] = out
    return out


def _quadrature_order_for(kmax: float, extent: float, q: int) -> int:
    """Gauss order per direction resolving ``e^{-i k y}`` times a degree ``q-1`` proxy."""
    waves = float(kmax) * float(extent) / (2.0 * math.pi)
    return int(max(10, math.ceil(2.5 * waves) + int(q) + 4))


def _plane_wave_coeffs_local(piece, poly: ec.LeafPoly, kx, ky) -> np.ndarray:
    """``int_piece e^{-i k . y} rho(y) dy`` for flat wavenumber arrays, physical units.

    Box pieces covering the proxy's own box use the closed Legendre form; every
    other piece falls back to the piece's own quadrature at an order chosen from
    the highest wavenumber.  The two agree to the quadrature's own accuracy,
    which is the check ``plane_wave_coeffs_box_vs_quadrature`` below.
    """
    kx = np.atleast_1d(np.asarray(kx, dtype=np.float64)).reshape(-1)
    ky = np.atleast_1d(np.asarray(ky, dtype=np.float64)).reshape(-1)
    centre = np.asarray(poly.box.center, dtype=np.float64)
    if _piece_is_full_box(piece, poly.box):
        ix = _legendre_plane_wave_factors(kx, poly.box.half, poly.q)
        iy = _legendre_plane_wave_factors(ky, poly.box.half, poly.q)
        core = np.einsum("mk,mn,nk->k", ix, poly.legendre_coeffs, iy)
        phase = np.exp(-1j * (kx * centre[0] + ky * centre[1]))
        return (poly.box.half**2) * phase * core
    kmax = float(max(np.max(np.abs(kx)), np.max(np.abs(ky)), 0.0))
    a1, b1, a2, b2 = piece.bounding_box()
    extent = max(b1 - a1, b2 - a2)
    order = _quadrature_order_for(kmax, extent, poly.q)
    nodes, weights = piece.quadrature(order, max_diameter=None)
    if nodes.shape[0] == 0:
        return np.zeros(kx.size, dtype=np.complex128)
    src = weights * poly.eval(nodes)
    phase = np.exp(-1j * (np.outer(kx, nodes[:, 0]) + np.outer(ky, nodes[:, 1])))
    return phase @ src


def plane_wave_coeffs(piece, poly: ec.LeafPoly, kx, ky) -> np.ndarray:
    """``int_piece e^{-i k . y} rho(y) dy`` for flat wavenumber arrays, physical units.

    Dispatches to ``experiment_e_geometry.plane_wave_coeffs`` when that module is
    importable (it carries the divergence-theorem polygon form and the curved
    route); the local implementation keeps the same signature and is what the
    module falls back to.
    """
    fn = _resolve("plane_wave_coeffs", _plane_wave_coeffs_local)
    return np.asarray(fn(piece, poly, kx, ky), dtype=np.complex128)


def _curved_tensor_coeffs(piece, poly: ec.LeafPoly, k1: np.ndarray, centre):
    """Tensor-grid coefficients of a curved piece by one rank-``M`` contraction.

    The quadrature rule is exactly the one ``experiment_e_geometry`` would pick
    for the same grid (``curved_plane_wave_quadrature`` at the grid's largest
    ``|k|``), so this evaluates the same sum as the flat route; what changes is
    the contraction order.  The flat route forms an ``(n^2, M)`` phase matrix and
    therefore takes ``n^2 M`` complex exponentials and ``n^2 M`` words of memory;
    writing the phase as ``e^{-i k_i y_1} e^{-i k_j y_2}`` takes ``2 n M``
    exponentials and one ``n x M x n`` matrix product.  At the grid sizes of this
    study (``n = 147``, ``M`` of order ``10^4`` on a cut star leaf) that is the
    difference between minutes and a fraction of a second per piece per level,
    and it is what makes case S affordable at all.  Returns ``None`` when the
    geometry module is absent or the piece carries no nodes.
    """
    if _geometry is None:
        return None
    k1 = np.asarray(k1, dtype=np.float64)
    k_abs = float(np.max(np.abs(k1))) if k1.size else 0.0
    k_max = math.hypot(k_abs, k_abs)
    nodes, weights = _geometry.curved_plane_wave_quadrature(piece, k_max)
    if nodes.shape[0] == 0:
        return None
    values = weights * poly.eval(nodes)
    rel = nodes - np.asarray(centre, dtype=np.float64)[None, :]
    ex = np.exp(-1j * np.outer(k1, rel[:, 0]))
    ey = np.exp(-1j * np.outer(k1, rel[:, 1]))
    return (ex * values[None, :]) @ ey.T


def _tensor_plane_wave_coeffs(piece, poly: ec.LeafPoly, k1: np.ndarray) -> np.ndarray:
    """Box-centred coefficients on a tensor grid, shape ``(n, n)``, ``kx`` slowest.

    Centred at the proxy's own leaf centre, i.e. ``int e^{-i k . (y - c)} rho dy``;
    the caller supplies the phase to whatever expansion centre it wants.
    """
    n = int(np.asarray(k1).size)
    if _piece_is_full_box(piece, poly.box):
        factors = _legendre_plane_wave_factors(k1, poly.box.half, poly.q)
        return (poly.box.half**2) * (factors.T @ poly.legendre_coeffs @ factors)
    centre = np.asarray(poly.box.center, dtype=np.float64)
    if piece.kind == "curved":
        fast = _curved_tensor_coeffs(piece, poly, k1, centre)
        if fast is not None:
            return fast
    kxf = np.repeat(k1, n)
    kyf = np.tile(k1, n)
    absolute = plane_wave_coeffs(piece, poly, kxf, kyf)
    phase = np.exp(1j * (kxf * centre[0] + kyf * centre[1]))
    return (absolute * phase).reshape(n, n)


# ---------------------------------------------------------------------------
# stage 1: the coarsest part W_0
# ---------------------------------------------------------------------------


def _gather_sources(leaves: Sequence, order: int, max_diameter=None):
    """Concatenated quadrature nodes and ``weight * rho`` over every leaf piece."""
    nodes, source = [], []
    for leaf in leaves:
        if leaf.poly is None:
            raise ValueError("every leaf needs a density proxy")
        for piece in leaf.pieces:
            pts, wts = piece.quadrature(order, max_diameter)
            if pts.shape[0] == 0:
                continue
            nodes.append(pts)
            source.append(wts * leaf.poly.eval(pts))
    if not nodes:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(nodes), np.concatenate(source)


def w0_stage(
    leaves: Sequence,
    targets,
    t0: float,
    order: int = DEFAULT_W0_ORDER,
    max_diameter=None,
    chunk_elements: int = 4_000_000,
) -> StageField:
    """``(2 pi)^{-1} W_0 * rho`` and its derivatives by direct tensor Gauss quadrature.

    ``W_0(r) = -log r - chi_0(r; t_0)`` is analytic with scale ``sqrt(t_0)``, so
    the same smooth rule carries the potential, the gradient
    ``W_0'(r) (x - y) / r`` and the Laplacian ``Delta W_0 = -e^{-z} / (2 t_0)``.
    The Laplacian is therefore exactly ``-(G_{t_0} * rho)`` evaluated with this
    quadrature, which is what the residual check compares against.
    """
    pts = _as_targets(targets)
    nodes, src = _gather_sources(leaves, order, max_diameter)
    field = StageField.zeros(pts.shape[0])
    if nodes.shape[0] == 0:
        return field
    t0 = float(t0)
    log4t = math.log(4.0 * t0)
    step = max(1, int(chunk_elements // max(1, nodes.shape[0])))
    for lo in range(0, pts.shape[0], step):
        hi = min(pts.shape[0], lo + step)
        dx = pts[lo:hi, 0][:, None] - nodes[None, :, 0]
        dy = pts[lo:hi, 1][:, None] - nodes[None, :, 1]
        r2 = dx * dx + dy * dy
        z = r2 / (4.0 * t0)
        w0 = -0.5 * (ec.e1_plus_log(z) + log4t)
        safe = np.where(z > 0.0, z, 1.0)
        ratio = np.where(z > 0.0, -np.expm1(-safe) / safe, 1.0)
        dwr = -ratio / (4.0 * t0)
        lap = -np.exp(-z) / (2.0 * t0)
        field.u[lo:hi] = ec.INV_TWO_PI * (w0 @ src)
        field.grad[lo:hi, 0] = ec.INV_TWO_PI * ((dwr * dx) @ src)
        field.grad[lo:hi, 1] = ec.INV_TWO_PI * ((dwr * dy) @ src)
        field.lap[lo:hi] = ec.INV_TWO_PI * (lap @ src)
    return field


# ---------------------------------------------------------------------------
# stage 2: one scale shell by plane waves
# ---------------------------------------------------------------------------


def shell_stage(
    leaves: Sequence,
    targets,
    level: int,
    tree: ec.QuadTree,
    theta: float = ec.THETA_PRIMARY,
    nu: float = ec.NU_PERIOD,
    tol: float = 1e-12,
    grid: ec.PlaneWaveGrid | None = None,
) -> tuple[StageField, dict]:
    """``(2 pi)^{-1} D_l * rho`` over the colleague range at ``level``, by plane waves.

    Per box at ``level`` the leaf proxies inside it are transformed and phased to
    the box centre; colleague boxes are translated in with ``e^{i k . delta}``,
    the multiplier ``int_{t_{l+1}}^{t_l} e^{-u |k|^2} du`` is applied once, and
    the inverse transform is evaluated at the targets, with ``i k`` and
    ``-|k|^2`` giving the gradient and the Laplacian.  Returns the field and a
    dictionary of grid diagnostics, including the per-coordinate mode count
    (a convention-dependent number, not DMK's ``N_1``).
    """
    pts = _as_targets(targets)
    root_half = tree.root.half
    t_c = ec.t_l(level, theta, root_half)
    t_f = ec.t_l(level + 1, theta, root_half)
    if grid is None:
        grid = ec.plane_wave_grid(tree.h(level), t_c, t_f, nu=nu, tol=tol)
    k1 = grid.k1
    n = grid.n_per_coordinate
    k_sq = k1[:, None] ** 2 + k1[None, :] ** 2
    mult = ec.shell_multiplier(k_sq, t_c, t_f)

    target_boxes = [tree.locate(level, p) for p in pts]
    needed = set()
    for idx in set(target_boxes):
        needed.update(tree.colleague_indices(level, idx[0], idx[1]))

    coeffs: dict = {}
    for leaf in leaves:
        if leaf.poly is None:
            continue
        idx = tree.locate(level, leaf.box.center)
        if idx not in needed:
            continue
        box = tree.box_at(level, idx[0], idx[1])
        shift = np.asarray(leaf.box.center) - np.asarray(box.center)
        phase = np.exp(-1j * k1 * shift[0])[:, None] * np.exp(
            -1j * k1 * shift[1]
        )[None, :]
        acc = coeffs.get(idx)
        if acc is None:
            acc = np.zeros((n, n), dtype=np.complex128)
            coeffs[idx] = acc
        for piece in leaf.pieces:
            acc += phase * _tensor_plane_wave_coeffs(piece, leaf.poly, k1)

    field = StageField.zeros(pts.shape[0])
    ik = 1j * k1
    for idx in sorted(set(target_boxes)):
        box = tree.box_at(level, idx[0], idx[1])
        local = np.zeros((n, n), dtype=np.complex128)
        for jdx in tree.colleague_indices(level, idx[0], idx[1]):
            src = coeffs.get(jdx)
            if src is None:
                continue
            other = tree.box_at(level, jdx[0], jdx[1])
            delta = np.asarray(box.center) - np.asarray(other.center)
            phase = np.exp(1j * k1 * delta[0])[:, None] * np.exp(
                1j * k1 * delta[1]
            )[None, :]
            local += phase * src
        matrix = grid.weight * mult * local
        rows = [m for m, b in enumerate(target_boxes) if b == idx]
        for m in rows:
            d = pts[m] - np.asarray(box.center)
            ex = np.exp(1j * k1 * d[0])
            ey = np.exp(1j * k1 * d[1])
            me = matrix @ ey
            field.u[m] = float(np.real(ex @ me))
            field.grad[m, 0] = float(np.real((ik * ex) @ me))
            field.grad[m, 1] = float(np.real(ex @ (matrix @ (ik * ey))))
            field.lap[m] = float(np.real(ex @ ((-k_sq * matrix) @ ey)))
    info = {
        "level": int(level),
        "t_coarse": t_c,
        "t_fine": t_f,
        "h_level": tree.h(level),
        "dk": grid.dk,
        "period": grid.period,
        "n_per_coordinate": grid.n_per_coordinate,
        "n_modes": grid.n_modes,
        "trunc_rel": grid.trunc_rel,
        "alias_rel": grid.alias_rel,
        "n_boxes_with_source": len(coeffs),
    }
    return field, info


# ---------------------------------------------------------------------------
# stage 3: the leaf closure
# ---------------------------------------------------------------------------


def _closure_v_rule(target, box: ec.Box, t_leaf: float, order: int, levels: int):
    """Graded ``v``-quadrature for one target and one leaf box, from the A-wave rule."""
    lo = box.low
    shifted = np.asarray(target, dtype=np.float64) - lo
    return v_quadrature(shifted, box.side, float(t_leaf), int(order), int(levels))


def prefix_potential(
    piece,
    poly: ec.LeafPoly,
    targets,
    t: float,
    n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD,
    v_order: int = DEFAULT_V_ORDER,
    v_levels: int = DEFAULT_V_LEVELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, grad u, Delta u)`` of ``(2 pi)^{-1} chi_0(.; t) * rho`` over one piece.

    Box pieces use the separable ``v``-quadrature for the potential and gradient
    and the closed-form Gaussian box moment for ``Delta u``; polygon and curved
    pieces use the target-centred polar rule for all three, the gradient through
    the geometry module's vector fan, which dispatches a curved piece's arcs to
    the arc rule rather than to a chord polygon through them.  The Laplacian is
    the smooth part only (``G_t * rho`` over the piece); the ``-rho 1_Omega``
    term of the distributional identity is added once per target by
    ``closure_stage``.
    """
    pts = _as_targets(targets)
    u = np.zeros(pts.shape[0])
    grad = np.zeros((pts.shape[0], 2))
    lap = np.zeros(pts.shape[0])
    dens = _shifted_monomials(poly)
    centre = np.asarray(poly.box.center, dtype=np.float64)
    offsets = pts - centre[None, :]
    if piece.kind == "box":
        a1, b1, a2, b2 = piece.bounding_box()
        rect = (a1 - centre[0], b1 - centre[0], a2 - centre[1], b2 - centre[1])
        for m in range(pts.shape[0]):
            v_nodes, v_weights = _closure_v_rule(
                pts[m], poly.box, t, v_order, v_levels
            )
            value, gradient = _separable_box_closure(
                rect, offsets[m], dens, float(t), v_nodes, v_weights
            )
            u[m] = value
            grad[m] = gradient
        lap = _box_heat_integral(rect, offsets, dens, float(t))
        return u, grad, lap
    kernels = _polar_kernels(float(t))
    u = polar_reference(piece, poly, pts, kernels["prefix"], n_ang, n_rad)
    lap = polar_reference(piece, poly, pts, kernels["heat"], n_ang, n_rad)
    vector_fan = _resolve("polar_gradient", _MISSING_VECTOR_FAN)
    if vector_fan is not _MISSING_VECTOR_FAN:
        grad = vector_fan(piece, poly, pts, kernels["grad"], n_ang, n_rad)
        return u, grad, lap
    if piece.kind != "polygon":
        raise RuntimeError(
            "a curved piece's closure gradient needs "
            "experiment_e_geometry.polar_gradient; the local vertex-loop fan is "
            "exact only on straight-sided pieces"
        )
    loop = piece.vertices - centre[None, :]
    for m in range(pts.shape[0]):
        grad[m] = _polar_gradient(
            loop, offsets[m], dens, kernels["grad"], n_ang, n_rad
        )
    return u, grad, lap


def closure_stage(
    leaves: Sequence,
    targets,
    tree: ec.QuadTree,
    t_leaf: float,
    n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD,
    v_order: int = DEFAULT_V_ORDER,
    v_levels: int = DEFAULT_V_LEVELS,
) -> StageField:
    """``(2 pi)^{-1} chi_0(.; t_L) * rho`` over the colleague leaves of every target.

    ``Delta u = G_{t_L} * rho - rho 1_Omega``; the Gaussian term is the piecewise
    closed form or polar rule of ``prefix_potential`` restricted to the same
    colleague leaves, and the indicator term uses the proxy of the leaf holding
    the target.
    """
    pts = _as_targets(targets)
    field = StageField.zeros(pts.shape[0])
    index = _leaf_index_map(leaves, tree)
    level = tree.levels
    evaluate = _resolve("prefix_potential", prefix_potential)
    for m in range(pts.shape[0]):
        own = tree.locate(level, pts[m])
        for jdx in tree.colleague_indices(level, own[0], own[1]):
            leaf = index.get(jdx)
            if leaf is None or leaf.poly is None:
                continue
            for piece in leaf.pieces:
                if evaluate is prefix_potential:
                    u, grad, lap = evaluate(
                        piece,
                        leaf.poly,
                        pts[m : m + 1],
                        float(t_leaf),
                        n_ang=n_ang,
                        n_rad=n_rad,
                        v_order=v_order,
                        v_levels=v_levels,
                    )
                else:
                    u, grad, lap = evaluate(
                        piece, leaf.poly, pts[m : m + 1], float(t_leaf)
                    )
                field.u[m] += float(u[0])
                field.grad[m] += grad[0]
                field.lap[m] += float(lap[0])
        leaf = index.get(own)
        if leaf is not None and leaf.poly is not None:
            if any(_inside_piece(p, pts[m : m + 1])[0] for p in leaf.pieces):
                field.lap[m] -= float(leaf.poly.eval(pts[m : m + 1])[0])
    return field


# ---------------------------------------------------------------------------
# assembly and residual report
# ---------------------------------------------------------------------------


def assemble(
    leaves: Sequence,
    targets,
    theta: float = ec.THETA_PRIMARY,
    tree: ec.QuadTree | None = None,
    levels: int | None = None,
    root_half: float = ec.ROOT_HALF,
    w0_order: int = DEFAULT_W0_ORDER,
    nu: float = ec.NU_PERIOD,
    pw_tol: float = 1e-12,
    n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD,
    v_order: int = DEFAULT_V_ORDER,
    v_levels: int = DEFAULT_V_LEVELS,
) -> dict:
    """Assemble all three stages at ``targets`` and return per-stage and total fields.

    ``leaves`` are finest-level ``Leaf`` objects carrying pieces and proxies;
    ``theta`` is the single window declaration ``Theta`` of the plan.  The result
    holds ``StageField`` objects for ``W_0``, each shell, the shell total, the
    closure and the grand total, plus the per-level plane-wave diagnostics and
    wall times.
    """
    pts = _as_targets(targets)
    if levels is None:
        levels = int(leaves[0].box.level)
    if tree is None:
        tree = ec.build_quadtree(levels=int(levels), root_half=float(root_half))
    times = [ec.t_l(level, theta, tree.root.half) for level in range(levels + 1)]

    clock = time.perf_counter()
    w0 = w0_stage(leaves, pts, times[0], order=w0_order)
    t_w0 = time.perf_counter() - clock

    shells: list[StageField] = []
    infos: list[dict] = []
    clock = time.perf_counter()
    for level in range(levels):
        field, info = shell_stage(
            leaves, pts, level, tree, theta=theta, nu=nu, tol=pw_tol
        )
        shells.append(field)
        infos.append(info)
    t_shell = time.perf_counter() - clock

    shell_total = StageField.zeros(pts.shape[0])
    for field in shells:
        shell_total = shell_total + field

    clock = time.perf_counter()
    closure = closure_stage(
        leaves,
        pts,
        tree,
        times[-1],
        n_ang=n_ang,
        n_rad=n_rad,
        v_order=v_order,
        v_levels=v_levels,
    )
    t_closure = time.perf_counter() - clock

    return {
        "theta": float(theta),
        "levels": int(levels),
        "times": times,
        "targets": pts,
        "tree": tree,
        "w0": w0,
        "shells": shells,
        "shell_total": shell_total,
        "closure": closure,
        "total": w0 + shell_total + closure,
        "modes": infos,
        "seconds": {"w0": t_w0, "shells": t_shell, "closure": t_closure},
    }


def heat_smoothed_proxy(
    leaves: Sequence,
    targets,
    t: float,
    tree: ec.QuadTree,
    cutoff: float = HEAT_CUTOFF,
    n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD,
    restrict: Callable | None = None,
) -> np.ndarray:
    """``(G_t * rho_proxy)(x)`` over ``Omega``, exactly for box pieces.

    Leaves farther than ``cutoff sqrt(t)`` from a target are dropped (below
    ``1e-18`` of the peak).  Box pieces use the closed-form Gaussian moments of
    the proxy, polygon and curved pieces the polar rule with the heat kernel, so
    this is not a graded global quadrature and its cost does not blow up as
    ``t`` shrinks.  ``restrict(target_index, leaf_indices) -> bool`` can limit
    the sum to a sub-list (used to form the colleague-only variant).
    """
    pts = _as_targets(targets)
    out = np.zeros(pts.shape[0], dtype=np.float64)
    radius = float(cutoff) * math.sqrt(float(t))
    kernels = None
    for leaf in leaves:
        if leaf.poly is None:
            continue
        centre = np.asarray(leaf.box.center, dtype=np.float64)
        reach = radius + leaf.box.half * math.sqrt(2.0)
        near = np.nonzero(
            np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) <= reach
        )[0]
        if restrict is not None:
            idx = tree.indices_of(leaf.box)
            near = np.array(
                [m for m in near if restrict(int(m), idx)], dtype=np.int64
            )
        if near.size == 0:
            continue
        dens = _shifted_monomials(leaf.poly)
        offsets = pts[near] - centre[None, :]
        for piece in leaf.pieces:
            if piece.kind == "box":
                a1, b1, a2, b2 = piece.bounding_box()
                rect = (
                    a1 - centre[0],
                    b1 - centre[0],
                    a2 - centre[1],
                    b2 - centre[1],
                )
                out[near] += _box_heat_integral(rect, offsets, dens, float(t))
            else:
                if kernels is None:
                    kernels = _polar_kernels(float(t))
                out[near] += polar_reference(
                    piece, leaf.poly, pts[near],
                    kernels["heat"], n_ang, n_rad,
                )
    return out


def residual_report(
    leaves: Sequence,
    targets,
    theta: float = ec.THETA_PRIMARY,
    result: dict | None = None,
    tree: ec.QuadTree | None = None,
    labels: Sequence | None = None,
    indicator: Callable | None = None,
    **assemble_kwargs,
) -> dict:
    """Per-stage PDE residual against the closed-form heat-smoothed right-hand sides.

    Compares ``-Delta`` of every stage with ``G_{t_0} * rho``,
    ``(G_{t_{l+1}} - G_{t_l}) * rho`` and ``rho - G_{t_L} * rho``, and the total
    with ``rho 1_Omega``.  All right-hand sides use the same piecewise proxy the
    hierarchy convolves, so the numbers isolate the scheme, not the proxy's
    interpolation error.  Returns per-stage max and RMS residuals overall and by
    probe family when ``labels`` is given.

    One caveat on reading the ``closure`` row: that stage's Laplacian and the
    right-hand side ``rho - G_{t_L} * rho`` are both assembled from the same
    closed-form Gaussian box moments, so the row measures the colleague
    truncation of the closure and nothing else.  It is *not* an accuracy
    statement about the separable ``v``-rule or the polar prefix rule, which are
    the closure's actual quadratures and never enter this row.  The independent
    checks on those are ``closure_separable_vs_polar_u``, ``..._grad`` and
    ``..._lap`` of the module self-test, and the assembled total against
    ``polar_reference_total``.

    ``indicator`` is the exact domain membership predicate.  It decides the
    ``rho 1_Omega`` factor of the ``closure`` and ``total`` right-hand sides, and
    it must not be the predicate the stage itself uses, or a geometric
    misclassification appears identically on both sides and cancels.  The
    returned dict reports ``indicator_mismatch``, the number of targets where the
    two predicates disagree; anything other than zero invalidates the rows.
    """
    pts = _as_targets(targets)
    if result is None:
        result = assemble(leaves, pts, theta=theta, tree=tree, **assemble_kwargs)
    tree = result["tree"]
    times = result["times"]
    levels = result["levels"]

    smoothed = [
        heat_smoothed_proxy(leaves, pts, t, tree) for t in times
    ]
    rho_stage = proxy_density(leaves, tree)(pts)
    if indicator is None:
        rho_here = rho_stage
        mismatch = -1
    else:
        rho_here = proxy_density(leaves, tree, inside=indicator)(pts)
        mismatch = int(np.count_nonzero(rho_here != rho_stage))

    rhs = [smoothed[0]]
    names = ["w0"]
    stages = [result["w0"]]
    for level in range(levels):
        rhs.append(smoothed[level + 1] - smoothed[level])
        names.append(f"shell{level}")
        stages.append(result["shells"][level])
    rhs.append(rho_here - smoothed[levels])
    names.append("closure")
    stages.append(result["closure"])

    def _stats(err, scale):
        """Max and RMS of an error vector, plus a relative max against ``scale``."""
        denom = max(float(np.max(np.abs(scale))), 1e-300)
        return {
            "max_abs": float(np.max(np.abs(err))),
            "rms": float(np.sqrt(np.mean(err * err))),
            "max_rel": float(np.max(np.abs(err)) / denom),
        }

    rows = []
    for name, stage, target_rhs in zip(names, stages, rhs, strict=True):
        err = -stage.lap - target_rhs
        row = {"stage": name}
        row.update(_stats(err, target_rhs))
        if labels is not None:
            by_family = {}
            for family in sorted(set(labels)):
                mask = np.array([lab == family for lab in labels])
                by_family[family] = float(np.max(np.abs(err[mask])))
            if "jump" in by_family and len(by_family) > 1:
                keep = np.array([lab != "jump" for lab in labels])
                by_family["all_except_jump"] = float(np.max(np.abs(err[keep])))
            row["by_family"] = by_family
        rows.append(row)

    total_err = -result["total"].lap - rho_here
    total_row = {"stage": "total"}
    total_row.update(_stats(total_err, rho_here))
    if labels is not None:
        total_row["by_family"] = {
            family: float(
                np.max(
                    np.abs(
                        total_err[np.array([lab == family for lab in labels])]
                    )
                )
            )
            for family in sorted(set(labels))
        }
        if "jump" in total_row["by_family"] and len(total_row["by_family"]) > 1:
            keep = np.array([lab != "jump" for lab in labels])
            total_row["by_family"]["all_except_jump"] = float(
                np.max(np.abs(total_err[keep]))
            )
    rows.append(total_row)
    return {
        "stages": rows,
        "times": times,
        "rho": rho_here.tolist(),
        "modes": result["modes"],
        "indicator_mismatch": mismatch,
    }


# ---------------------------------------------------------------------------
# case B: the full-box control
# ---------------------------------------------------------------------------


def build_box_leaves(
    rho: Callable, levels: int, order: int, root_half: float = ec.ROOT_HALF
) -> tuple[list, ec.QuadTree]:
    """Case B: every finest-level box is a full-box leaf with a proxy of order ``order``."""
    tree = ec.build_quadtree(levels=int(levels), root_half=float(root_half))
    leaves = []
    for box in tree.leaves():
        poly = ec.LeafPoly.from_callable(box, rho, int(order))
        leaves.append(ec.Leaf(box, [ec.make_box_piece(box)], poly))
    return leaves, tree


def rho_gaussian(points) -> np.ndarray:
    """The plan's smooth test density ``rho_1(x) = exp(-3 |x|^2)``."""
    pts = _as_targets(points)
    return np.exp(-3.0 * (pts[:, 0] ** 2 + pts[:, 1] ** 2))


def case_b_probes(
    levels: int, root_half: float = ec.ROOT_HALF, seed: int = 20260918
) -> tuple[np.ndarray, list]:
    """24 deterministic probes: leaf centres, leaf faces, leaf corners and bulk points.

    The face and corner families are the worst case for a colleague-only window:
    the nearest source a colleague list omits is one full leaf away from a target
    on a leaf face, but one and a half leaves away from a target at the centre.
    """
    tree = ec.build_quadtree(levels=int(levels), root_half=float(root_half))
    limit = 0.7 * float(root_half)
    boxes = [
        box
        for box in tree.leaves()
        if abs(box.center[0]) < limit and abs(box.center[1]) < limit
    ]
    rng = np.random.default_rng(int(seed))
    picks = boxes[:: max(1, len(boxes) // 6)][:6]
    pts, labels = [], []
    for box in picks:
        pts.append(np.asarray(box.center, dtype=np.float64))
        labels.append("centre")
    for box in picks:
        pts.append(np.array([box.center[0] + box.half, box.center[1]]))
        labels.append("face")
    for box in picks:
        pts.append(np.array([box.center[0] + box.half, box.center[1] + box.half]))
        labels.append("corner")
    extra = rng.uniform(-limit, limit, size=(6, 2))
    for point in extra:
        pts.append(point)
        labels.append("bulk")
    return np.array(pts, dtype=np.float64), labels


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


def polar_reference(
    piece, poly: ec.LeafPoly, targets, kernel, n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD
) -> np.ndarray:
    """``int_piece kernel(|x - y|) rho(y) dy`` by the target-centred polar rule.

    Dispatches to ``experiment_e_geometry.polar_reference`` when that module is
    importable, because it also covers curved pieces; the local fallback is the
    straight-edge fan of the leaf-residual study, shifted to the leaf centre.
    """
    fn = _resolve("polar_reference", _polar_reference_local)
    return np.asarray(fn(piece, poly, targets, kernel, n_ang, n_rad), dtype=np.float64)


def _polar_reference_local(
    piece, poly: ec.LeafPoly, targets, kernel, n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD
) -> np.ndarray:
    """Straight-edge polar quadrature of one piece, in leaf-centred coordinates."""
    pts = _as_targets(targets)
    centre = np.asarray(poly.box.center, dtype=np.float64)
    dens = _shifted_monomials(poly)
    loop = _piece_loop(piece) - centre[None, :]
    return np.array(
        [
            polar_polygon_potential(
                loop, pts[m] - centre, dens, kernel, n_ang, n_rad
            )
            for m in range(pts.shape[0])
        ]
    )


def polar_reference_total(
    leaves: Sequence,
    targets,
    t_leaf: float,
    n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD,
    near_factor: float = 3.0,
    far_order: int = 24,
) -> np.ndarray:
    """Direct reference value of the full potential ``int_Omega K rho`` at the targets.

    Leaves nearer than ``near_factor`` leaf diameters use the polar rule of the
    leaf-residual study (exact for the ``r^k log r`` moments); farther leaves use
    a plain tensor Gauss rule of order ``far_order``, where the log kernel is
    analytic on the leaf scale.  ``far_order = 16`` at ``near_factor = 2.5``
    leaves the far part below roundoff, which the ``polar_reference_hybrid``
    check verifies against an all-polar reference.
    """
    pts = _as_targets(targets)
    out = np.zeros(pts.shape[0])
    kernel = RadialKernel("full", 1.0, None, math.sqrt(float(t_leaf)))
    for leaf in leaves:
        if leaf.poly is None:
            continue
        centre = np.asarray(leaf.box.center, dtype=np.float64)
        dist = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
        near = dist <= float(near_factor) * leaf.box.side
        for piece in leaf.pieces:
            idx_near = np.nonzero(near)[0]
            if idx_near.size:
                out[idx_near] += polar_reference(
                    piece, leaf.poly, pts[idx_near], kernel, n_ang, n_rad
                )
            far = np.nonzero(~near)[0]
            if far.size == 0:
                continue
            nodes, weights = piece.quadrature(int(far_order))
            if nodes.shape[0] == 0:
                continue
            src = weights * leaf.poly.eval(nodes)
            for m in far:
                r = np.hypot(nodes[:, 0] - pts[m, 0], nodes[:, 1] - pts[m, 1])
                out[m] += float(np.dot(src, ec.laplace_kernel(r)))
    return out


def shell_polar_reference(
    leaves: Sequence,
    targets,
    level: int,
    tree: ec.QuadTree,
    theta: float = ec.THETA_PRIMARY,
    n_ang: int = DEFAULT_POLAR_ANG,
    n_rad: int = DEFAULT_POLAR_RAD,
) -> np.ndarray:
    """Direct polar-quadrature value of one shell over the same colleague range.

    The shell is smooth (its log germs cancel analytically), so the polar rule
    with ``log_coeff = 0`` and the stable ``shell_kernel`` as the smooth profile
    is an independent check on the plane-wave route.
    """
    pts = _as_targets(targets)
    t_c = ec.t_l(level, theta, tree.root.half)
    t_f = ec.t_l(level + 1, theta, tree.root.half)
    kernel = RadialKernel(
        f"shell{level}",
        0.0,
        lambda r, a=t_c, b=t_f: ec.shell_kernel(r, a, b) * ec.INV_TWO_PI,
        math.sqrt(t_f),
    )
    owners = [(leaf, tree.locate(level, leaf.box.center)) for leaf in leaves]
    out = np.zeros(pts.shape[0])
    for m in range(pts.shape[0]):
        own = tree.locate(level, pts[m])
        colleagues = set(tree.colleague_indices(level, own[0], own[1]))
        for leaf, idx in owners:
            if idx not in colleagues or leaf.poly is None:
                continue
            for piece in leaf.pieces:
                out[m] += float(
                    polar_reference(
                        piece, leaf.poly, pts[m : m + 1], kernel, n_ang, n_rad
                    )[0]
                )
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def _check_plane_wave_coeffs() -> list[dict]:
    """Closed-form box coefficients against quadrature, and the tensor fast path."""
    box = ec.Box((0.125, -0.375), 0.125, 3)
    poly = ec.LeafPoly.from_callable(box, rho_gaussian, 6)
    piece = ec.make_box_piece(box)
    rng = np.random.default_rng(7)
    kx = rng.uniform(-80.0, 80.0, size=32)
    ky = rng.uniform(-80.0, 80.0, size=32)
    closed = _plane_wave_coeffs_local(piece, poly, kx, ky)
    nodes, weights = piece.quadrature(40)
    src = weights * poly.eval(nodes)
    quad = np.exp(
        -1j * (np.outer(kx, nodes[:, 0]) + np.outer(ky, nodes[:, 1]))
    ) @ src
    err = float(np.max(np.abs(closed - quad)) / np.max(np.abs(quad)))
    rows = [
        {
            "check": "plane_wave_coeffs_box_vs_quadrature",
            "max_rel_error": err,
            "passed": bool(err < 1e-12),
        }
    ]
    k1 = np.linspace(-60.0, 60.0, 21)
    tensor = _tensor_plane_wave_coeffs(piece, poly, k1)
    flat_x = np.repeat(k1, k1.size)
    flat_y = np.tile(k1, k1.size)
    phase = np.exp(1j * (flat_x * box.center[0] + flat_y * box.center[1]))
    flat = _plane_wave_coeffs_local(piece, poly, flat_x, flat_y) * phase
    err2 = float(
        np.max(np.abs(tensor.reshape(-1) - flat)) / np.max(np.abs(flat))
    )
    rows.append(
        {
            "check": "plane_wave_coeffs_tensor_vs_flat",
            "max_rel_error": err2,
            "passed": bool(err2 < 1e-13),
        }
    )
    return rows


def _check_geometry_interop() -> list[dict]:
    """Cross-module agreement with ``experiment_e_geometry`` when that module is present."""
    if _geometry is None:
        return [
            {
                "check": "geometry_module_interop",
                "note": "experiment_e_geometry not importable; local routines used",
                "passed": True,
            }
        ]
    box = ec.Box((0.125, -0.375), 0.125, 3)
    poly = ec.LeafPoly.from_callable(box, rho_gaussian, 6)
    piece = ec.make_box_piece(box)
    rng = np.random.default_rng(11)
    kx = rng.uniform(-90.0, 90.0, size=48)
    ky = rng.uniform(-90.0, 90.0, size=48)
    mine = _plane_wave_coeffs_local(piece, poly, kx, ky)
    theirs = np.asarray(_geometry.plane_wave_coeffs(piece, poly, kx, ky))
    err = float(np.max(np.abs(mine - theirs)) / np.max(np.abs(theirs)))
    rows = [
        {
            "check": "plane_wave_coeffs_vs_geometry_module",
            "max_rel_error": err,
            "passed": bool(err < 1e-12),
        }
    ]
    t_leaf = ec.t_l(3, ec.THETA_PRIMARY)
    kernel = _polar_kernels(t_leaf)["prefix"]
    targets = np.array([[0.125, -0.375], [0.25, -0.375], [0.2, -0.3], [0.4, -0.2]])
    mine_u = _polar_reference_local(piece, poly, targets, kernel)
    theirs_u = np.asarray(
        _geometry.polar_reference(piece, poly, targets, kernel)
    )
    err_u = float(np.max(np.abs(mine_u - theirs_u)) / np.max(np.abs(theirs_u)))
    rows.append(
        {
            "check": "polar_reference_vs_geometry_module",
            "max_rel_error": err_u,
            "passed": bool(err_u < 1e-11),
        }
    )
    return rows


def _check_shell_pitfall(theta: float = ec.THETA_PRIMARY) -> list[dict]:
    """The coarse-level shell pitfall: naive difference versus the scaled profile.

    At level 0 the shell ``D_0 = chi_0(.; t_0) - chi_0(.; t_1)`` is formed three
    ways: the plain double-precision difference of two ``chi_0`` values, the
    cancellation-free ``shell_kernel``, and the fixed scaled dyadic profile
    ``shell_profile(r / (2 sqrt(t_1)))`` evaluated once and rescaled.
    """
    eps = 2.220446049250313e-16
    rows = []
    for level in range(5):
        t_c = ec.t_l(level, theta)
        t_f = ec.t_l(level + 1, theta)
        h_l = ec.h_level(level)
        for name, radii in (
            ("colleague_range", np.geomspace(1e-4 * h_l, 2.0 * h_l, 240)),
            ("deep_near_field", np.geomspace(1e-30, 1e-4 * h_l, 240)),
        ):
            stable = ec.shell_kernel(radii, t_c, t_f)
            naive = ec.chi_0(radii, t_c) - ec.chi_0(radii, t_f)
            profile = ec.shell_profile(radii / (2.0 * math.sqrt(t_f)))
            rel_naive = float(np.max(np.abs(naive - stable) / np.abs(stable)))
            rel_profile = float(np.max(np.abs(profile - stable) / np.abs(stable)))
            rows.append(
                {
                    "check": "shell_pitfall_kernel",
                    "level": level,
                    "range": name,
                    "naive_max_rel_error": rel_naive,
                    "profile_max_rel_error": rel_profile,
                    "digits_lost_naive": float(
                        max(0.0, math.log10(max(rel_naive, eps) / eps))
                    ),
                    "passed": bool(rel_profile < 1e-13),
                }
            )
        grid = ec.plane_wave_grid(h_l, t_c, t_f)
        k_sq = grid.k_squared()
        stable_m = ec.shell_multiplier(k_sq, t_c, t_f)
        with np.errstate(divide="ignore", invalid="ignore"):
            naive_m = np.where(
                k_sq > 0.0,
                (np.exp(-t_f * k_sq) - np.exp(-t_c * k_sq)) / np.where(
                    k_sq > 0.0, k_sq, 1.0
                ),
                t_c - t_f,
            )
        rel_m = float(
            np.max(np.abs(naive_m - stable_m) / np.maximum(np.abs(stable_m), 1e-300))
        )
        rows.append(
            {
                "check": "shell_pitfall_multiplier",
                "level": level,
                "n_per_coordinate": grid.n_per_coordinate,
                "naive_max_rel_error": rel_m,
                "digits_lost_naive": float(
                    max(0.0, math.log10(max(rel_m, eps) / eps))
                ),
                "passed": True,
            }
        )
    return rows


def _check_separable_vs_polar(theta: float = ec.THETA_PRIMARY) -> list[dict]:
    """Closure over one box piece: separable ``v``-rule against the polar rule.

    The same region is presented once as a ``box`` piece (separable path) and
    once as a ``polygon`` piece with the same vertices (polar path), so the two
    independent quadratures of the same singular integral must agree.  The
    gradient and the Laplacian are compared on the same footing.
    """
    levels = 3
    box = ec.Box((0.125, 0.125), 0.125, levels)
    poly = ec.LeafPoly.from_callable(box, rho_gaussian, 6)
    t_leaf = ec.t_l(levels, theta)
    box_piece = ec.make_box_piece(box)
    poly_piece = ec.make_polygon_piece(box.vertices(), box)
    targets = np.array(
        [
            [0.125, 0.125],
            [0.25, 0.125],
            [0.25, 0.25],
            [0.3, 0.2],
            [0.06, 0.19],
            [0.375, 0.375],
        ]
    )
    u_sep, g_sep, l_sep = prefix_potential(box_piece, poly, targets, t_leaf)
    u_pol, g_pol, l_pol = prefix_potential(poly_piece, poly, targets, t_leaf)
    scale = max(float(np.max(np.abs(u_pol))), 1e-300)
    rows = [
        {
            "check": "closure_separable_vs_polar_u",
            "max_rel_error": float(np.max(np.abs(u_sep - u_pol)) / scale),
            "passed": bool(np.max(np.abs(u_sep - u_pol)) / scale < 1e-11),
        },
        {
            "check": "closure_separable_vs_polar_grad",
            "max_rel_error": float(
                np.max(np.abs(g_sep - g_pol))
                / max(float(np.max(np.abs(g_pol))), 1e-300)
            ),
            "passed": bool(
                np.max(np.abs(g_sep - g_pol))
                / max(float(np.max(np.abs(g_pol))), 1e-300)
                < 1e-10
            ),
        },
        {
            "check": "closure_separable_vs_polar_lap",
            "max_rel_error": float(
                np.max(np.abs(l_sep - l_pol))
                / max(float(np.max(np.abs(l_pol))), 1e-300)
            ),
            "passed": bool(
                np.max(np.abs(l_sep - l_pol))
                / max(float(np.max(np.abs(l_pol))), 1e-300)
                < 1e-10
            ),
        },
    ]
    ref = np.array(
        [
            separable_prefix_rect(
                (
                    box.low[0] - box.center[0],
                    box.high[0] - box.center[0],
                    box.low[1] - box.center[1],
                    box.high[1] - box.center[1],
                ),
                targets[m] - np.asarray(box.center),
                _shifted_monomials(poly),
                t_leaf,
            )[0]
            for m in range(targets.shape[0])
        ]
    )
    err = float(np.max(np.abs(u_sep - ref)) / scale)
    rows.append(
        {
            "check": "closure_separable_vs_experiment_c_rule",
            "max_rel_error": err,
            "passed": bool(err < 1e-12),
        }
    )
    return rows


def _check_gradients(levels: int = 3, theta: float = ec.THETA_PRIMARY) -> list[dict]:
    """Finite-difference check of every stage's analytic gradient and Laplacian."""
    leaves, tree = build_box_leaves(rho_gaussian, levels, 6)
    base = np.array([[0.13, -0.07], [0.31, 0.19], [-0.4, 0.28]])
    step = 1e-5
    offsets = [
        np.array([step, 0.0]),
        np.array([-step, 0.0]),
        np.array([0.0, step]),
        np.array([0.0, -step]),
    ]
    pts = np.vstack([base] + [base + off for off in offsets])
    res = assemble(leaves, pts, theta=theta, tree=tree, levels=levels)
    rows = []
    n = base.shape[0]
    for name in ("w0", "shell_total", "closure", "total"):
        field = res[name]
        u = field.u
        gx = (u[n : 2 * n] - u[2 * n : 3 * n]) / (2.0 * step)
        gy = (u[3 * n : 4 * n] - u[4 * n : 5 * n]) / (2.0 * step)
        lap = (
            u[n : 2 * n] + u[2 * n : 3 * n] + u[3 * n : 4 * n] + u[4 * n : 5 * n]
            - 4.0 * u[:n]
        ) / (step * step)
        scale_g = max(float(np.max(np.abs(field.grad[:n]))), 1e-12)
        scale_l = max(float(np.max(np.abs(field.lap[:n]))), 1e-12)
        err_g = float(
            max(
                np.max(np.abs(gx - field.grad[:n, 0])),
                np.max(np.abs(gy - field.grad[:n, 1])),
            )
            / scale_g
        )
        err_l = float(np.max(np.abs(lap - field.lap[:n])) / scale_l)
        rows.append(
            {
                "check": f"finite_difference_{name}",
                "grad_max_rel_error": err_g,
                "lap_max_rel_error": err_l,
                "passed": bool(err_g < 1e-6 and err_l < 1e-5),
            }
        )
    return rows


def _check_w0_order(levels: int = 3, theta: float = ec.THETA_PRIMARY) -> list[dict]:
    """Self-convergence of the ``W_0`` stage in the tensor Gauss order."""
    leaves, tree = build_box_leaves(rho_gaussian, levels, 6)
    targets, _ = case_b_probes(levels)
    t0 = ec.t_l(0, theta, tree.root.half)
    ref = w0_stage(leaves, targets, t0, order=40)
    rows = []
    for order in (6, 10, 14, 18, 22, 26, 30):
        field = w0_stage(leaves, targets, t0, order=order)
        err = float(np.max(np.abs(field.u - ref.u)))
        err_l = float(np.max(np.abs(field.lap - ref.lap)))
        rows.append(
            {
                "check": "w0_order_convergence",
                "levels": levels,
                "order": order,
                "max_abs_error_u": err,
                "max_abs_error_lap": err_l,
                "n_points_per_direction": ec.gauss_points_for_degree(order),
                "passed": bool(order < 22 or (err < 1e-13 and err_l < 1e-12)),
            }
        )
    return rows


def _check_shell_against_polar(
    levels: int = 3, theta: float = ec.THETA_PRIMARY
) -> list[dict]:
    """Plane-wave shells against direct polar quadrature of the same shell kernel."""
    leaves, tree = build_box_leaves(rho_gaussian, levels, 6)
    targets, _ = case_b_probes(levels)
    targets = targets[:6]
    rows = []
    for level in (0, levels - 1):
        field, info = shell_stage(leaves, targets, level, tree, theta=theta)
        ref = shell_polar_reference(leaves, targets, level, tree, theta=theta)
        scale = max(float(np.max(np.abs(ref))), 1e-300)
        err = float(np.max(np.abs(field.u - ref)) / scale)
        rows.append(
            {
                "check": "shell_planewave_vs_polar",
                "level": level,
                "n_per_coordinate": info["n_per_coordinate"],
                "max_rel_error": err,
                "passed": bool(err < 1e-9),
            }
        )
    return rows


def _check_heat_smoothed(levels: int = 3, theta: float = ec.THETA_PRIMARY) -> list[dict]:
    """Exact per-leaf ``G_t * rho_proxy`` against the common module's graded quadrature."""
    leaves, tree = build_box_leaves(rho_gaussian, levels, 6)
    targets, _ = case_b_probes(levels)
    targets = targets[:4]
    rho = proxy_density(leaves, tree)
    pieces = [piece for leaf in leaves for piece in leaf.pieces]
    rows = []
    for level in (0, levels):
        t = ec.t_l(level, theta, tree.root.half)
        mine = heat_smoothed_proxy(leaves, targets, t, tree)
        theirs = ec.heat_smoothed_density(pieces, rho, t, targets, order=10, grade=0.75)
        scale = max(float(np.max(np.abs(theirs))), 1e-300)
        err = float(np.max(np.abs(mine - theirs)) / scale)
        rows.append(
            {
                "check": "heat_smoothed_proxy_vs_common",
                "level": level,
                "max_rel_error": err,
                "passed": bool(err < 1e-10),
            }
        )
    return rows


def _check_polar_hybrid(levels: int = 3, theta: float = ec.THETA_PRIMARY) -> list[dict]:
    """Hybrid near-polar / far-Gauss total reference against an all-polar reference."""
    leaves, tree = build_box_leaves(rho_gaussian, levels, 6)
    targets, _ = case_b_probes(levels)
    targets = targets[:6]
    t_leaf = ec.t_l(levels, theta, tree.root.half)
    hybrid = polar_reference_total(leaves, targets, t_leaf)
    allpolar = polar_reference_total(leaves, targets, t_leaf, near_factor=1e9)
    refined = polar_reference_total(
        leaves, targets, t_leaf, n_ang=40, n_rad=32, near_factor=1e9
    )
    scale = max(float(np.max(np.abs(allpolar))), 1e-300)
    err = float(np.max(np.abs(hybrid - allpolar)) / scale)
    self_err = float(np.max(np.abs(refined - allpolar)) / scale)
    return [
        {
            "check": "polar_reference_hybrid",
            "max_rel_error": err,
            "passed": bool(err < 1e-12),
        },
        {
            "check": "polar_reference_self_convergence",
            "max_rel_error": self_err,
            "passed": bool(self_err < 1e-11),
        },
    ]


def _run_case_b(levels: int, theta: float, order: int = 6) -> dict:
    """Case B at one level count: per-stage residuals, total reference, mode counts."""
    leaves, tree = build_box_leaves(rho_gaussian, levels, order)
    targets, labels = case_b_probes(levels)
    clock = time.perf_counter()
    res = assemble(leaves, targets, theta=theta, tree=tree, levels=levels)
    seconds = time.perf_counter() - clock
    report = residual_report(
        leaves, targets, theta=theta, result=res, labels=labels
    )
    reference = polar_reference_total(leaves, targets, res["times"][-1])
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    diff = res["total"].u - reference
    return {
        "levels": levels,
        "theta": theta,
        "proxy_order": order,
        "n_leaves": len(leaves),
        "n_targets": int(targets.shape[0]),
        "seconds": seconds,
        "stage_seconds": res["seconds"],
        "residuals": report["stages"],
        "modes": [
            {
                "level": info["level"],
                "n_per_coordinate": info["n_per_coordinate"],
                "n_modes": info["n_modes"],
                "trunc_rel": info["trunc_rel"],
                "alias_rel": info["alias_rel"],
            }
            for info in res["modes"]
        ],
        "total_vs_polar_reference": {
            "max_abs": float(np.max(np.abs(diff))),
            "max_rel": float(np.max(np.abs(diff)) / scale),
            "by_family": {
                family: float(
                    np.max(
                        np.abs(diff[np.array([lab == family for lab in labels])])
                    )
                )
                for family in sorted(set(labels))
            },
        },
        "window_truncation": {
            "theta": theta,
            "tail_bound_exp_minus_theta_sq_over_4": math.exp(-theta * theta / 4.0),
            "centre_total": float(
                np.max(
                    np.abs(
                        np.array(
                            [
                                report["stages"][-1]["by_family"]["centre"],
                            ]
                        )
                    )
                )
            ),
            "face_corner_total": max(
                report["stages"][-1]["by_family"]["face"],
                report["stages"][-1]["by_family"]["corner"],
            ),
        },
        "labels": labels,
        "targets": targets.tolist(),
    }


def run_self_test(quick: bool = False, levels=(3, 4, 5)) -> tuple[list[dict], dict, bool]:
    """Run every unit check and the case B study; return rows, the study and a flag."""
    rows: list[dict] = []
    rows.extend(_check_plane_wave_coeffs())
    rows.extend(_check_geometry_interop())
    rows.extend(_check_shell_pitfall())
    rows.extend(_check_separable_vs_polar())
    rows.extend(_check_gradients())
    rows.extend(_check_w0_order())
    rows.extend(_check_shell_against_polar())
    rows.extend(_check_heat_smoothed())
    rows.extend(_check_polar_hybrid())
    study = {"case_b": [], "theta_sensitivity": []}
    wanted = (3,) if quick else tuple(levels)
    for level in wanted:
        study["case_b"].append(_run_case_b(level, ec.THETA_PRIMARY))
    study["theta_sensitivity"].append(_run_case_b(3, ec.THETA_SECONDARY))
    passed = all(bool(row.get("passed", False)) for row in rows)
    return rows, study, passed


def main(argv=None) -> int:
    """Command-line entry point: run the checks and the case B study, write JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="directory for the summary JSON")
    parser.add_argument("--quick", action="store_true", help="level 3 only")
    parser.add_argument(
        "--levels",
        default="3,4,5",
        help="comma-separated level counts for the case B study",
    )
    args = parser.parse_args(argv)
    levels = tuple(int(v) for v in str(args.levels).split(",") if v.strip())
    clock = time.perf_counter()
    rows, study, passed = run_self_test(quick=args.quick, levels=levels)
    payload = {
        "experiment": "E hierarchy (W_0, plane-wave shells, leaf closure)",
        "checks": rows,
        "study": study,
        "passed": passed,
        "seconds": time.perf_counter() - clock,
        "environment": ec.environment_summary(),
    }
    if args.out:
        out = Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "experiment_e_hierarchy.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str)
        )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
