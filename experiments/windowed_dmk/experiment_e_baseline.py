"""Experiment E baseline: asymptotic local volume potential and a Nystrom solver.

Two independent baselines for the two-dimensional leaf-closure study, both of
which are compared against the windowed leaf closure assembled elsewhere in this
experiment.

Part (a): the asymptotic local part of the volume heat potential
----------------------------------------------------------------
``fryklund_VL`` transcribes Lemma 4.5 of Fryklund, Greengard, Jiang and Potter
(2024), ``fryklund_lightweight_2024``, which expands

    V_L[f](x) = int_0^delta int_Omega G_t(x - y) f(y) dy dt
              = (4 pi)^{-1} int_Omega E_1(|x - y|^2 / (4 delta)) f(y) dy

through order ``delta^2`` with a remainder ``O(delta^{5/2})``.  This is exactly
the windowed leaf closure of the plan when ``delta = t_L``: in the chi units of
``experiment_e_common`` the same quantity is ``(2 pi)^{-1} chi_0(.; t_L) * rho``.

Frame and sign conventions (fixed once, used everywhere below):

* ``b`` is the closest point of ``d Omega`` to the target ``x`` and
  ``r = |x - b|``, ``c = r / sqrt(delta)``.
* The local frame has the target ``x`` at the origin, ``xi`` along the tangent
  at ``b`` and ``eta`` along the **inward** normal at ``b`` (pointing into
  ``Omega``).  For an interior target ``b`` sits at ``(0, -r)`` and ``Omega`` is
  locally ``{eta > gamma(xi)}`` with ``gamma(0) = -r``.
* ``kappa_b = gamma''(0)`` is the curvature of the boundary at ``b`` measured
  with respect to the *outward* normal: it is ``+1 / R`` for the interior of a
  disk of radius ``R`` and negative where the domain is locally concave.
* The density jet is Fryklund's, i.e. the partial derivatives of ``f`` at the
  target in the local frame,
  ``f(xi, eta) = f + f_xi xi + f_eta eta + f_xixi xi^2 / 2 + f_xieta xi eta
  + f_etaeta eta^2 / 2 + ...``.  ``f_xi`` and ``f_xieta`` are odd in ``xi`` and
  drop out of Lemma 4.5; they are accepted and ignored.

Coefficient transcription.  The published display (4.6) is typeset in fragments
and the ``delta^{3/2}`` bracket as it reads there, ``(kappa_b f - 2 f_eta)``,
disagrees in sign with the same paper's on-boundary corollary (4.11), which
reads ``(2 f_eta - kappa_b f)``.  An independent evaluation of the half-plane
integral (7) below, term by term, agrees with (4.11); the ``delta`` and
``delta^2`` groups of (4.6) are reproduced exactly as printed.  This module
therefore uses ``(2 f_eta - kappa_b f)`` and records the discrepancy in
``LEMMA_4_5_NOTES``.  ``lemma_4_5_terms()`` prints the three groups as
implemented.

Exterior targets are not covered by the published lemma (it is stated for
``x in Omega``).  ``sign = -1`` evaluates them through the exact complement
identity: over the whole plane the local potential of a quadratic density is
``delta f + delta^2 (f_xixi + f_etaeta) / 2``, so the exterior value is that
minus the interior value for the complementary domain, whose curvature is
``-kappa_b`` and whose inward normal is ``-eta``.

Part (b): Nystrom double-layer solver
-------------------------------------
The interior Dirichlet Laplace problem on a smooth closed curve is solved with
the double-layer representation ``u = D[mu]``,

    D[mu](x) = (2 pi)^{-1} int_Gamma ((x - y) . n_y) |x - y|^{-2} mu(y) ds_y,

whose interior limit is ``-mu / 2 + D_pv[mu]``, so the boundary equation is
``(-I / 2 + D) mu = g``.  With a counter-clockwise analytic parametrization the
kernel is analytic including the diagonal, where its limit is the Kress value
``-kappa(s) |gamma'(s)| / (4 pi)``; the periodic trapezoidal rule is then
spectrally accurate.  ``double_layer_eval`` and ``double_layer_grad`` evaluate
the representation and its gradient at interior points (smooth rule only, so
accuracy degrades within a few quadrature spacings of the curve).

Manufactured boundary value problem
-----------------------------------
``u_exact(x, y) = sin(2 x) cosh(2 y) + exp(-2 (x^2 + y^2))``, harmonic plus a
Gaussian bump, so ``rho = -Laplacian u_exact = exp(-2 r^2) (8 - 16 r^2)`` is
smooth, non-polynomial and not identically zero.  The plan's route is
``u = V[rho] + w`` with ``w`` harmonic and ``w = u_exact - V[rho]`` on the
boundary, which is one solve of the Nystrom system above.

References in this module are direct target-centred polar quadratures of
``(4 pi)^{-1} E_1(|x - y|^2 / (4 delta))`` with the radial moments in closed
form (``radial_log_moment``), so only the angular variable is quadrature.  For a
flat boundary and a density that is a polynomial of degree at most two the
expansion is exact, which is the sharpest available check on the printed
coefficients.

Exploratory campaign code: no warm-up, no repetition statistics, no claim that
any rule here is optimal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfc, exp1, gamma as gamma_fn, gammainc

from experiment_e_common import (
    TWO_PI,
    environment_summary,
    gauss_legendre,
)

__all__ = [
    "JET_KEYS",
    "LEMMA_4_5_NOTES",
    "NystromCurve",
    "STAR_AMPLITUDE",
    "STAR_BASE",
    "STAR_MODES",
    "as_jet",
    "boundary_frame",
    "circle_curve",
    "closest_point_on_curve",
    "disk_VL_reference",
    "double_layer_eval",
    "double_layer_grad",
    "double_layer_matrix",
    "fryklund_VL",
    "grad_u_exact",
    "halfplane_VL_reference",
    "jet_from_callable",
    "jet_of_coeffs",
    "lemma_4_5_terms",
    "radial_log_moment",
    "rho_manufactured",
    "solve_interior_dirichlet",
    "star_curve",
    "star_position",
    "u_exact",
]

SQRT_PI = math.sqrt(math.pi)

JET_KEYS = ("f", "f_xi", "f_eta", "f_xixi", "f_xieta", "f_etaeta")
"""Density jet keys, Fryklund's partial derivatives in the local frame."""

STAR_BASE = 0.55
"""Mean radius of the plan's smooth star boundary."""

STAR_AMPLITUDE = 0.3
"""Relative amplitude of the star boundary's radial modulation."""

STAR_MODES = 5
"""Angular mode number of the star boundary."""

LEMMA_4_5_NOTES = (
    "Display (4.6) as typeset carries the delta^{3/2} bracket (kappa_b f - "
    "2 f_eta); the on-boundary corollary (4.11) of the same paper carries "
    "(2 f_eta - kappa_b f). A term-by-term evaluation of the half-plane "
    "integral agrees with (4.11), so this module uses (2 f_eta - kappa_b f). "
    "The delta and delta^2 groups are used exactly as printed and are "
    "reproduced to machine precision by the flat-boundary reference."
)


def lemma_4_5_terms() -> dict:
    """The three groups of Lemma 4.5 as implemented, for the record."""
    return {
        "c": "c = r / sqrt(delta), r = |x - b|, E = exp(-c^2 / 4)",
        "delta_term": (
            "(delta / 4) * (2 erf(c/2) + 2 c E / sqrt(pi) - c^2 erfc(c/2) + 2) * f"
        ),
        "delta_three_half_term": (
            "delta^{3/2} * ((4 - 2 c^2) E + sqrt(pi) c^3 erfc(c/2)) "
            "* (2 f_eta - kappa_b f) / (12 sqrt(pi))"
        ),
        "delta_two_term": (
            "(delta^2 / 48) * ( erfc(c/2) * (4 c^4 kappa_b f_eta "
            "- 3 (c^4 + 4) f_etaeta + (c^4 - 12) f_xixi - 3 c^4 kappa_b^2 f) "
            "+ 2 c (c^2 - 2) E * (-4 kappa_b f_eta + 3 f_etaeta - f_xixi "
            "+ 3 kappa_b^2 f) / sqrt(pi) + 24 (f_xixi + f_etaeta) )"
        ),
        "remainder": "O(delta^{5/2})",
        "limits": (
            "c -> infinity: delta f + delta^2 (f_xixi + f_etaeta) / 2 (eq. 4.10); "
            "c = 0: delta f / 2 + delta^{3/2} (2 f_eta - kappa_b f) / (3 sqrt(pi)) "
            "+ delta^2 (f_xixi + f_etaeta) / 4 (eq. 4.11)"
        ),
        "sign_note": LEMMA_4_5_NOTES,
    }


# ---------------------------------------------------------------------------
# density jets in the local frame
# ---------------------------------------------------------------------------


def as_jet(jet) -> dict:
    """Normalize a density jet to a dict over ``JET_KEYS`` (missing entries zero)."""
    if isinstance(jet, Mapping):
        out = {}
        for key in JET_KEYS:
            out[key] = np.asarray(jet.get(key, 0.0), dtype=np.float64)
        unknown = set(jet) - set(JET_KEYS)
        if unknown:
            raise ValueError(f"unknown jet keys: {sorted(unknown)}")
        return out
    values = list(jet)
    if len(values) > len(JET_KEYS):
        raise ValueError("jet sequence longer than JET_KEYS")
    values = values + [0.0] * (len(JET_KEYS) - len(values))
    return {key: np.asarray(val, dtype=np.float64)
            for key, val in zip(JET_KEYS, values, strict=True)}


def jet_from_callable(
    func: Callable,
    target,
    tangent,
    inward_normal,
    step: float = 3.0e-3,
) -> dict:
    """Density jet at ``target`` in the ``(tangent, inward_normal)`` frame.

    ``func`` maps an ``(n, 2)`` array of physical points to an ``(n,)`` array.
    Fourth-order central differences with spacing ``step``; the second
    derivatives carry about ``1e-9`` absolute error at the default spacing,
    which is far below the ``delta^2`` weight they receive.
    """
    target = np.asarray(target, dtype=np.float64).reshape(2)
    tan = np.asarray(tangent, dtype=np.float64).reshape(2)
    nrm = np.asarray(inward_normal, dtype=np.float64).reshape(2)
    tan = tan / np.linalg.norm(tan)
    nrm = nrm / np.linalg.norm(nrm)
    h = float(step)

    offsets = [-2, -1, 0, 1, 2]
    grid = []
    for iy in offsets:
        for ix in offsets:
            grid.append(target + h * ix * tan + h * iy * nrm)
    vals = np.asarray(func(np.asarray(grid)), dtype=np.float64).reshape(5, 5)

    d1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)
    d2 = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h * h)
    mid = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    def apply(row_op, col_op):
        return float(row_op @ (vals @ col_op))

    return {
        "f": apply(mid, mid),
        "f_xi": apply(mid, d1),
        "f_eta": apply(d1, mid),
        "f_xixi": apply(mid, d2),
        "f_xieta": apply(d1, d1),
        "f_etaeta": apply(d2, mid),
    }


def boundary_frame(target, closest_point_b, sign: float = 1.0):
    """Return ``(r, tangent, inward_normal)`` for the local frame at ``b``.

    ``sign`` is ``+1`` for a target inside ``Omega`` and ``-1`` for a target
    outside it; it fixes which way the inward normal points when only the two
    points are known.  ``r = 0`` falls back to the ``+x`` axis.
    """
    target = np.asarray(target, dtype=np.float64).reshape(2)
    base = np.asarray(closest_point_b, dtype=np.float64).reshape(2)
    delta_vec = target - base
    r = float(np.linalg.norm(delta_vec))
    if r == 0.0:
        inward = np.array([0.0, 1.0])
    else:
        inward = float(np.sign(sign)) * delta_vec / r
    tangent = np.array([inward[1], -inward[0]])
    return r, tangent, inward


# ---------------------------------------------------------------------------
# part (a): Fryklund, Greengard, Jiang and Potter (2024), Lemma 4.5
# ---------------------------------------------------------------------------


def _fryklund_interior(r, kappa_b, jet: dict, delta: float):
    """Lemma 4.5 for a target inside ``Omega`` at distance ``r`` from ``b``."""
    delta = float(delta)
    r = np.asarray(r, dtype=np.float64)
    kappa = np.asarray(kappa_b, dtype=np.float64)
    c = r / math.sqrt(delta)

    f = jet["f"]
    f_eta = jet["f_eta"]
    f_xixi = jet["f_xixi"]
    f_etaeta = jet["f_etaeta"]

    half = 0.5 * c
    e_gauss = np.exp(-0.25 * c * c)
    erfc_half = erfc(half)
    erf_half = erf(half)
    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2

    term_delta = (delta / 4.0) * (
        2.0 * erf_half + 2.0 * c * e_gauss / SQRT_PI - c2 * erfc_half + 2.0
    ) * f

    shape = (4.0 - 2.0 * c2) * e_gauss + SQRT_PI * c3 * erfc_half
    term_three_half = (
        delta ** 1.5 * shape * (2.0 * f_eta - kappa * f) / (12.0 * SQRT_PI)
    )

    bracket_erfc = (
        4.0 * c4 * kappa * f_eta
        - 3.0 * (c4 + 4.0) * f_etaeta
        + (c4 - 12.0) * f_xixi
        - 3.0 * c4 * kappa * kappa * f
    )
    bracket_gauss = (
        -4.0 * kappa * f_eta + 3.0 * f_etaeta - f_xixi + 3.0 * kappa * kappa * f
    )
    term_two = (delta * delta / 48.0) * (
        erfc_half * bracket_erfc
        + 2.0 * c * (c2 - 2.0) * e_gauss * bracket_gauss / SQRT_PI
        + 24.0 * (f_xixi + f_etaeta)
    )

    return term_delta + term_three_half + term_two


def fryklund_VL(target, closest_point_b, kappa_b, rho_jet, delta, sign: float = 1.0):
    """Asymptotic local volume potential of Lemma 4.5, ``fryklund_lightweight_2024``.

    Parameters
    ----------
    target, closest_point_b:
        Physical points; only ``r = |target - b|`` enters.  Both may be ``(2,)``
        or ``(n, 2)``.
    kappa_b:
        Curvature at ``b`` with respect to the outward normal (``+1 / R`` for the
        interior of a disk of radius ``R``).
    rho_jet:
        Density jet at the *target* in the local frame, see ``JET_KEYS``.
    delta:
        Short-time cutoff ``delta = delta_1``; the leaf closure uses ``t_L``.
    sign:
        ``+1`` when the target lies in ``Omega`` (the published case), ``-1``
        when it lies outside, which is evaluated through the exact complement
        identity documented in the module docstring.

    Returns the scalar (or array) value of ``V_L[f](target)``.
    """
    target = np.asarray(target, dtype=np.float64)
    base = np.asarray(closest_point_b, dtype=np.float64)
    r = np.linalg.norm(target - base, axis=-1)
    jet = as_jet(rho_jet)
    kappa = np.asarray(kappa_b, dtype=np.float64)
    delta = float(delta)

    if float(sign) >= 0.0:
        out = _fryklund_interior(r, kappa, jet, delta)
    else:
        flipped = dict(jet)
        flipped["f_eta"] = -jet["f_eta"]
        flipped["f_xieta"] = -jet["f_xieta"]
        whole_plane = delta * jet["f"] + 0.5 * delta * delta * (
            jet["f_xixi"] + jet["f_etaeta"]
        )
        out = whole_plane - _fryklund_interior(r, -kappa, flipped, delta)
    return out if np.ndim(out) else float(out)


# ---------------------------------------------------------------------------
# direct references: target-centred polar quadrature with closed-form radials
# ---------------------------------------------------------------------------


def radial_log_moment(a, n: int, delta: float):
    """``int_0^a E_1(s^2 / (4 delta)) s^n ds`` in closed form, ``n >= 1``.

    Integration by parts turns the exponential integral into an incomplete
    gamma function:
    ``a^{n+1} E_1(a^2 / 4 delta) / (n + 1)
    + 2^{n+1} delta^{(n+1)/2} Gamma((n+1)/2, lower) / (n + 1)``.
    """
    n = int(n)
    if n < 1:
        raise ValueError("radial_log_moment needs n >= 1 for convergence at 0")
    a = np.asarray(a, dtype=np.float64)
    scalar = a.ndim == 0
    a = np.atleast_1d(a)
    out = np.zeros(a.shape, dtype=np.float64)
    pos = a > 0.0
    if np.any(pos):
        ap = a[pos]
        arg = ap * ap / (4.0 * float(delta))
        shape = 0.5 * (n + 1.0)
        lower = gamma_fn(shape) * gammainc(shape, arg)
        out[pos] = (
            ap ** (n + 1) * exp1(arg) / (n + 1.0)
            + (2.0 ** (n + 1)) * float(delta) ** shape * lower / (n + 1.0)
        )
    return float(out[0]) if scalar else out


def _polar_shell(theta, s_lo, s_hi, coeffs: Mapping, delta: float):
    """Angular integrand of the polar reference (without the ``1 / 4 pi``)."""
    theta = np.asarray(theta, dtype=np.float64)
    s_lo = np.broadcast_to(np.asarray(s_lo, dtype=np.float64), theta.shape)
    s_hi = np.broadcast_to(np.asarray(s_hi, dtype=np.float64), theta.shape)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    total = np.zeros(theta.shape, dtype=np.float64)
    for (i, j), coef in coeffs.items():
        if coef == 0.0:
            continue
        power = i + j + 1
        moment = (
            radial_log_moment(s_hi, power, delta)
            - radial_log_moment(s_lo, power, delta)
        )
        total = total + coef * (cos_t ** i) * (sin_t ** j) * moment
    return total


def _composite_gauss(lo: float, hi: float, npanels: int, ngauss: int):
    """Composite Gauss-Legendre nodes and weights on ``[lo, hi]``."""
    edges = np.linspace(lo, hi, int(npanels) + 1)
    gx, gw = gauss_legendre(int(ngauss))
    mid = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    nodes = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    weights = (half[:, None] * gw[None, :]).ravel()
    return nodes, weights


def jet_of_coeffs(coeffs: Mapping) -> dict:
    """Fryklund jet of the local polynomial ``sum coeffs[(i, j)] xi^i eta^j``."""
    def get(i, j):
        return float(coeffs.get((i, j), 0.0))

    return {
        "f": get(0, 0),
        "f_xi": get(1, 0),
        "f_eta": get(0, 1),
        "f_xixi": 2.0 * get(2, 0),
        "f_xieta": get(1, 1),
        "f_etaeta": 2.0 * get(0, 2),
    }


def halfplane_VL_reference(
    r: float,
    delta: float,
    coeffs: Mapping,
    sign: float = 1.0,
    cut_sigmas: float = 20.0,
    npanels: int = 48,
    ngauss: int = 24,
) -> float:
    """Exact ``V_L`` over a half-plane for a polynomial density, by polar quadrature.

    The target sits at the origin of the local frame; the boundary is the line
    ``eta = -r`` for ``sign = +1`` (target inside ``Omega = {eta > -r}``) and
    ``eta = +r`` for ``sign = -1`` (target outside ``Omega = {eta > r}``).
    ``coeffs`` maps ``(i, j)`` to the coefficient of ``xi^i eta^j``.
    """
    r = float(r)
    delta = float(delta)
    cut = float(cut_sigmas) * math.sqrt(delta)
    total = 0.0
    if float(sign) >= 0.0:
        panels = [(0.0, math.pi, lambda th: 0.0, lambda th: cut)]
        if r < cut:
            alpha = math.asin(min(1.0, r / cut))
            panels.append((math.pi, math.pi + alpha,
                           lambda th: 0.0, lambda th: cut))
            panels.append((math.pi + alpha, 2.0 * math.pi - alpha,
                           lambda th: 0.0, lambda th: r / (-np.sin(th))))
            panels.append((2.0 * math.pi - alpha, 2.0 * math.pi,
                           lambda th: 0.0, lambda th: cut))
        else:
            panels.append((math.pi, 2.0 * math.pi,
                           lambda th: 0.0, lambda th: cut))
    else:
        if r >= cut:
            return 0.0
        alpha = math.asin(min(1.0, r / cut))
        panels = [(alpha, math.pi - alpha,
                   lambda th: r / np.sin(th), lambda th: cut)]
    for lo, hi, s_lo_fn, s_hi_fn in panels:
        if hi <= lo:
            continue
        nodes, weights = _composite_gauss(lo, hi, npanels, ngauss)
        s_lo = np.broadcast_to(np.asarray(s_lo_fn(nodes), dtype=np.float64),
                               nodes.shape)
        s_hi = np.broadcast_to(np.asarray(s_hi_fn(nodes), dtype=np.float64),
                               nodes.shape)
        total += float(np.sum(weights * _polar_shell(nodes, s_lo, s_hi,
                                                     coeffs, delta)))
    return total / (4.0 * math.pi)


def disk_VL_reference(
    radius: float,
    r: float,
    delta: float,
    coeffs: Mapping,
    n_theta: int = 8192,
) -> float:
    """Exact ``V_L`` over a disk for a polynomial density, by polar quadrature.

    The target sits at the origin of the local frame at distance ``r`` inside the
    boundary, so the disk centre is at ``(0, radius - r)`` and the boundary
    curvature at the closest point is ``kappa_b = 1 / radius``.  The angular rule
    is the periodic trapezoidal rule.
    """
    radius = float(radius)
    r = float(r)
    if not 0.0 <= r < radius:
        raise ValueError("need 0 <= r < radius for a unique closest point")
    dist = radius - r
    theta = TWO_PI * np.arange(int(n_theta)) / float(n_theta)
    proj = dist * np.sin(theta)
    s_max = proj + np.sqrt(proj * proj + radius * radius - dist * dist)
    shell = _polar_shell(theta, np.zeros_like(theta), s_max, coeffs, delta)
    return float(np.sum(shell)) * (TWO_PI / float(n_theta)) / (4.0 * math.pi)


# ---------------------------------------------------------------------------
# part (b): Nystrom double-layer solver on a smooth closed curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NystromCurve:
    """Samples of an analytic counter-clockwise closed curve, trapezoidal rule.

    ``points`` are the physical nodes, ``normals`` the outward unit normals,
    ``speed`` is ``|gamma'|`` and ``curvature`` is positive where the enclosed
    domain is locally convex.  ``weights`` are the arclength weights
    ``2 pi speed / n``.
    """

    phi: np.ndarray
    points: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    speed: np.ndarray
    curvature: np.ndarray

    @property
    def size(self) -> int:
        """Number of quadrature nodes."""
        return int(self.points.shape[0])

    @property
    def weights(self) -> np.ndarray:
        """Arclength quadrature weights of the periodic trapezoidal rule."""
        return (TWO_PI / float(self.size)) * self.speed


def _curve_from_derivatives(phi, pos, dpos, ddpos) -> NystromCurve:
    """Assemble a ``NystromCurve`` from positions and their first two derivatives."""
    speed = np.hypot(dpos[:, 0], dpos[:, 1])
    normals = np.stack([dpos[:, 1], -dpos[:, 0]], axis=1) / speed[:, None]
    cross = dpos[:, 0] * ddpos[:, 1] - dpos[:, 1] * ddpos[:, 0]
    curvature = cross / speed ** 3
    return NystromCurve(
        phi=phi, points=pos, tangents=dpos, normals=normals,
        speed=speed, curvature=curvature,
    )


def star_position(phi, base: float = STAR_BASE, amplitude: float = STAR_AMPLITUDE,
                  modes: int = STAR_MODES):
    """Position, first and second derivative of the plan's star boundary.

    ``r(phi) = base (1 + amplitude cos(modes phi))``, traversed counter-clockwise.
    Local helper: ``experiment_e_common`` carries signed-distance utilities but no
    boundary parametrization, which the Nystrom rule needs.
    """
    phi = np.asarray(phi, dtype=np.float64)
    m = float(modes)
    rad = base * (1.0 + amplitude * np.cos(m * phi))
    drad = -base * amplitude * m * np.sin(m * phi)
    ddrad = -base * amplitude * m * m * np.cos(m * phi)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    pos = np.stack([rad * cos_p, rad * sin_p], axis=-1)
    dpos = np.stack([drad * cos_p - rad * sin_p,
                     drad * sin_p + rad * cos_p], axis=-1)
    ddpos = np.stack([ddrad * cos_p - 2.0 * drad * sin_p - rad * cos_p,
                      ddrad * sin_p + 2.0 * drad * cos_p - rad * sin_p], axis=-1)
    return pos, dpos, ddpos


def star_curve(n: int, base: float = STAR_BASE, amplitude: float = STAR_AMPLITUDE,
               modes: int = STAR_MODES) -> NystromCurve:
    """``n`` equispaced-in-parameter samples of the star boundary."""
    phi = TWO_PI * np.arange(int(n)) / float(n)
    pos, dpos, ddpos = star_position(phi, base, amplitude, modes)
    return _curve_from_derivatives(phi, pos, dpos, ddpos)


def circle_curve(n: int, radius: float = 1.0) -> NystromCurve:
    """``n`` equispaced samples of a circle, for solver sanity checks."""
    phi = TWO_PI * np.arange(int(n)) / float(n)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    pos = radius * np.stack([cos_p, sin_p], axis=-1)
    dpos = radius * np.stack([-sin_p, cos_p], axis=-1)
    ddpos = -radius * np.stack([cos_p, sin_p], axis=-1)
    return _curve_from_derivatives(phi, pos, dpos, ddpos)


def double_layer_matrix(curve: NystromCurve) -> np.ndarray:
    """Nystrom matrix of ``-I / 2 + D`` for the interior Dirichlet problem.

    Off-diagonal entries are the smooth kernel times the trapezoidal weight; the
    diagonal carries the Kress limiting value ``-kappa |gamma'| / (4 pi)``.
    """
    pts = curve.points
    nrm = curve.normals
    speed = curve.speed
    n = curve.size
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
    dot = np.einsum("ijk,jk->ij", diff, nrm)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = dot / dist_sq / TWO_PI
    kernel = kernel * speed[None, :]
    np.fill_diagonal(kernel, -curve.curvature * speed / (4.0 * math.pi))
    mat = (TWO_PI / float(n)) * kernel
    mat[np.diag_indices(n)] -= 0.5
    return mat


def solve_interior_dirichlet(curve: NystromCurve, boundary_values) -> np.ndarray:
    """Solve ``(-I / 2 + D) mu = g`` for the double-layer density ``mu``."""
    rhs = np.asarray(boundary_values, dtype=np.float64).reshape(curve.size)
    return np.linalg.solve(double_layer_matrix(curve), rhs)


def double_layer_eval(curve: NystromCurve, mu, targets) -> np.ndarray:
    """Evaluate ``D[mu]`` at interior ``targets`` with the smooth trapezoidal rule."""
    targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    mu = np.asarray(mu, dtype=np.float64).reshape(curve.size)
    diff = targets[:, None, :] - curve.points[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
    dot = np.einsum("ijk,jk->ij", diff, curve.normals)
    dens = mu * curve.weights / TWO_PI
    return (dot / dist_sq) @ dens


def double_layer_grad(curve: NystromCurve, mu, targets) -> np.ndarray:
    """Gradient of ``D[mu]`` at interior ``targets``; same smooth rule."""
    targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    mu = np.asarray(mu, dtype=np.float64).reshape(curve.size)
    diff = targets[:, None, :] - curve.points[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
    dot = np.einsum("ijk,jk->ij", diff, curve.normals)
    dens = (mu * curve.weights / TWO_PI)[None, :, None]
    term = (curve.normals[None, :, :] / dist_sq[:, :, None]
            - 2.0 * diff * (dot / dist_sq ** 2)[:, :, None])
    return np.sum(term * dens, axis=1)


def closest_point_on_curve(
    position_fn: Callable,
    target,
    phi0: float,
    iters: int = 60,
    tol: float = 1e-14,
):
    """Newton iteration for the closest curve point, returning ``(phi, point)``.

    ``position_fn(phi)`` must return ``(pos, dpos, ddpos)`` as ``star_position``
    does.  Minimizes ``|gamma(phi) - x|^2 / 2``; ``phi0`` must already be in the
    right basin (the polar angle of the target works for star-shaped curves).
    """
    target = np.asarray(target, dtype=np.float64).reshape(2)
    phi = float(phi0)
    for _ in range(int(iters)):
        pos, dpos, ddpos = position_fn(np.array([phi]))
        diff = pos[0] - target
        grad = float(diff @ dpos[0])
        hess = float(dpos[0] @ dpos[0] + diff @ ddpos[0])
        if hess <= 0.0:
            hess = float(dpos[0] @ dpos[0])
        step = grad / hess
        phi -= step
        if abs(step) < tol:
            break
    pos, _, _ = position_fn(np.array([phi]))
    return phi, pos[0]


# ---------------------------------------------------------------------------
# manufactured boundary value problem
# ---------------------------------------------------------------------------


def u_exact(points) -> np.ndarray:
    """``sin(2 x) cosh(2 y) + exp(-2 (x^2 + y^2))``, the manufactured solution."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    x = pts[:, 0]
    y = pts[:, 1]
    return np.sin(2.0 * x) * np.cosh(2.0 * y) + np.exp(-2.0 * (x * x + y * y))


def grad_u_exact(points) -> np.ndarray:
    """Gradient of ``u_exact``."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    x = pts[:, 0]
    y = pts[:, 1]
    bump = np.exp(-2.0 * (x * x + y * y))
    return np.stack([
        2.0 * np.cos(2.0 * x) * np.cosh(2.0 * y) - 4.0 * x * bump,
        2.0 * np.sin(2.0 * x) * np.sinh(2.0 * y) - 4.0 * y * bump,
    ], axis=1)


def rho_manufactured(points) -> np.ndarray:
    """``-Laplacian u_exact = exp(-2 r^2) (8 - 16 r^2)``, the manufactured density."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    r_sq = pts[:, 0] ** 2 + pts[:, 1] ** 2
    return np.exp(-2.0 * r_sq) * (8.0 - 16.0 * r_sq)


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------


def _record(results: list, name: str, ok: bool, detail: str, key_number) -> None:
    """Append one test record."""
    results.append({
        "name": name,
        "result": "pass" if ok else "fail",
        "detail": detail,
        "key_number": f"{key_number:.3e}" if isinstance(key_number, float)
        else str(key_number),
    })


_QUADRATIC = {
    (0, 0): 0.7, (1, 0): -0.4, (0, 1): 0.9,
    (2, 0): 0.35, (1, 1): -0.25, (0, 2): 0.6,
}
_CUBIC = dict(_QUADRATIC)
_CUBIC.update({(3, 0): 0.21, (2, 1): -0.17, (1, 2): 0.13, (0, 3): -0.29})


def _test_halfplane_exact(results: list) -> None:
    """Flat boundary, quadratic density: Lemma 4.5 must be exact."""
    worst_in = 0.0
    worst_out = 0.0
    jet = jet_of_coeffs(_QUADRATIC)
    rows = []
    for delta in (1.0e-2, 1.0e-4):
        for c_val in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            r = c_val * math.sqrt(delta)
            ref = halfplane_VL_reference(r, delta, _QUADRATIC, sign=1.0)
            got = fryklund_VL((0.0, 0.0), (0.0, -r), 0.0, jet, delta, sign=1.0)
            scale = max(abs(ref), delta * abs(jet["f"]))
            rel = abs(got - ref) / scale
            worst_in = max(worst_in, rel)
            rows.append({"delta": delta, "c": c_val, "sign": 1,
                         "ref": ref, "got": got, "rel": rel})
            if c_val > 0.0:
                ref_o = halfplane_VL_reference(r, delta, _QUADRATIC, sign=-1.0)
                got_o = fryklund_VL((0.0, 0.0), (0.0, r), 0.0, jet, delta,
                                    sign=-1.0)
                scale_o = max(abs(ref_o), delta * abs(jet["f"]))
                rel_o = abs(got_o - ref_o) / scale_o
                worst_out = max(worst_out, rel_o)
                rows.append({"delta": delta, "c": c_val, "sign": -1,
                             "ref": ref_o, "got": got_o, "rel": rel_o})
    _record(results, "halfplane_quadratic_exact_interior", worst_in < 5.0e-13,
            "flat boundary and a quadratic density make Lemma 4.5 exact; "
            "worst relative deviation over delta in {1e-2, 1e-4} and "
            "c = r/sqrt(delta) in {0, 0.5, 1, 2, 4, 8}", worst_in)
    _record(results, "halfplane_quadratic_exact_exterior", worst_out < 5.0e-13,
            "same check for exterior targets through the complement identity",
            worst_out)
    results[-1]["rows"] = rows


def _test_sign_of_eta_term(results: list) -> None:
    """The published (4.6) bracket sign is wrong; the (4.11) sign is right."""
    delta = 1.0e-3
    r = 0.5 * math.sqrt(delta)
    coeffs = {(0, 0): 1.0, (0, 1): 1.0}
    jet = jet_of_coeffs(coeffs)
    ref = halfplane_VL_reference(r, delta, coeffs, sign=1.0)
    good = fryklund_VL((0.0, 0.0), (0.0, -r), 0.0, jet, delta)
    bad_jet_value = _fryklund_interior(
        np.asarray(r), np.asarray(0.0),
        as_jet({**jet, "f_eta": -jet["f_eta"]}), delta)
    rel_good = abs(good - ref) / abs(ref)
    rel_bad = abs(float(bad_jet_value) - ref) / abs(ref)
    _record(results, "delta_three_half_sign", rel_good < 1e-13 < rel_bad,
            "with f = 1 + eta the implemented (2 f_eta - kappa f) matches the "
            f"reference to {rel_good:.2e} while the typeset (kappa f - 2 f_eta) "
            f"is off by {rel_bad:.2e}", rel_bad / max(rel_good, 1e-300))


def _test_disk_curvature(results: list) -> None:
    """Curved boundary: the expansion must converge at the stated order."""
    radius = 0.4
    coeffs = {(0, 0): 1.0, (0, 1): 0.6, (2, 0): 0.3, (0, 2): -0.45, (1, 1): 0.2}
    jet = jet_of_coeffs(coeffs)
    kappa = 1.0 / radius
    c_val = 1.0
    rows = []
    for delta in (4.0e-4, 1.0e-4, 2.5e-5, 6.25e-6):
        r = c_val * math.sqrt(delta)
        ref = disk_VL_reference(radius, r, delta, coeffs)
        got = fryklund_VL((0.0, 0.0), (0.0, -r), kappa, jet, delta)
        flat = fryklund_VL((0.0, 0.0), (0.0, -r), 0.0, jet, delta)
        rows.append({"delta": delta, "ref": ref, "got": got,
                     "err": abs(got - ref), "err_kappa_zero": abs(flat - ref)})
    rates = [math.log(rows[i]["err"] / rows[i + 1]["err"]) / math.log(4.0)
             for i in range(len(rows) - 1)]
    rate = rates[-1]
    gain = rows[-1]["err_kappa_zero"] / rows[-1]["err"]
    _record(results, "disk_curvature_order", 2.3 < rate < 2.7,
            "interior of a disk of radius 0.4 at c = 1: observed convergence "
            f"order of the Lemma 4.5 error in delta is {rate:.3f} "
            f"(expected 2.5); dropping kappa_b inflates the error by {gain:.1f}x",
            rate)
    results[-1]["rows"] = rows


def _test_delta_five_half_law(results: list) -> None:
    """Cubic density on a flat boundary: the remainder is exactly O(delta^{5/2})."""
    jet = jet_of_coeffs(_CUBIC)
    rows = []
    c_val = 1.5
    for delta in (1.0e-3, 2.5e-4, 6.25e-5, 1.5625e-5):
        r = c_val * math.sqrt(delta)
        ref = halfplane_VL_reference(r, delta, _CUBIC, sign=1.0)
        got = fryklund_VL((0.0, 0.0), (0.0, -r), 0.0, jet, delta)
        rows.append({"delta": delta, "ref": ref, "got": got,
                     "err": abs(got - ref),
                     "err_over_delta_2p5": abs(got - ref) / delta ** 2.5})
    rates = [math.log(rows[i]["err"] / rows[i + 1]["err"]) / math.log(4.0)
             for i in range(len(rows) - 1)]
    const = [row["err_over_delta_2p5"] for row in rows]
    spread = max(const) / min(const)
    _record(results, "remainder_order_five_half",
            abs(rates[-1] - 2.5) < 0.05 and spread < 1.2,
            "flat boundary, cubic density, c = 1.5: the error/delta^{5/2} "
            f"constant is {const[-1]:.4e} with spread {spread:.3f} over four "
            f"halvings and the observed order is {rates[-1]:.4f}", rates[-1])
    results[-1]["rows"] = rows


def _test_reference_self_convergence(results: list) -> None:
    """The polar references must be converged well below the tolerances used."""
    delta = 1.0e-4
    r = 1.5 * math.sqrt(delta)
    coarse = halfplane_VL_reference(r, delta, _CUBIC, npanels=24, ngauss=16)
    fine = halfplane_VL_reference(r, delta, _CUBIC, npanels=96, ngauss=32)
    rel_hp = abs(coarse - fine) / abs(fine)
    coeffs = {(0, 0): 1.0, (0, 1): 0.6, (2, 0): 0.3}
    d_coarse = disk_VL_reference(0.4, r, delta, coeffs, n_theta=4096)
    d_fine = disk_VL_reference(0.4, r, delta, coeffs, n_theta=16384)
    rel_disk = abs(d_coarse - d_fine) / abs(d_fine)
    worst = max(rel_hp, rel_disk)
    _record(results, "reference_self_convergence", worst < 1e-14,
            "half-plane panel refinement and disk trapezoid refinement of the "
            "polar references", worst)


def _test_radial_moment(results: list) -> None:
    """Closed-form radial moments against their analytic limits."""
    delta = 3.0e-3
    big = 60.0 * math.sqrt(delta)
    got1 = radial_log_moment(big, 1, delta)
    exact1 = 2.0 * delta
    got3 = radial_log_moment(big, 3, delta)
    exact3 = 4.0 * delta * delta
    err = max(abs(got1 - exact1) / exact1, abs(got3 - exact3) / exact3)
    _record(results, "radial_moment_limits", err < 1e-14,
            "int_0^inf E_1(s^2/4d) s ds = 2 d (int_0^inf E_1 = 1) and "
            "int_0^inf E_1(s^2/4d) s^3 ds = 4 d^2 (int_0^inf p E_1 = 1/2)",
            err)


def _harmonic_field(points, source=(1.3, 0.9)):
    """A harmonic test field: ``log|x - z0|`` with ``z0`` outside the star."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    diff = pts - np.asarray(source, dtype=np.float64)
    return 0.5 * np.log(np.einsum("ij,ij->i", diff, diff))


def _harmonic_grad(points, source=(1.3, 0.9)):
    """Gradient of ``_harmonic_field``."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    diff = pts - np.asarray(source, dtype=np.float64)
    return diff / np.einsum("ij,ij->i", diff, diff)[:, None]


def _interior_probes(scale: float, count: int = 41):
    """Interior probes of the star curve at a fixed fraction of the local radius."""
    phi = np.linspace(0.0, TWO_PI, count, endpoint=False) + 0.017
    pos, _, _ = star_position(phi)
    return scale * pos


def _test_nystrom_circle(results: list) -> None:
    """Sanity check of the operator on a circle, where everything is known."""
    curve = circle_curve(64, radius=0.8)
    mat = double_layer_matrix(curve)
    ones = np.ones(curve.size)
    err = float(np.max(np.abs(mat @ ones + ones)))
    _record(results, "nystrom_circle_operator", err < 1e-13,
            "(-I/2 + D)[1] = -1 on a circle (D[1] = -1/2 on the boundary)", err)


def _test_nystrom_convergence(results: list) -> None:
    """Spectral convergence of the interior Dirichlet solve on the star curve."""
    probes = _interior_probes(0.5)
    ref = _harmonic_field(probes)
    rows = []
    best = math.inf
    for n in (64, 128, 256, 512):
        curve = star_curve(n)
        mu = solve_interior_dirichlet(curve, _harmonic_field(curve.points))
        got = double_layer_eval(curve, mu, probes)
        err = float(np.max(np.abs(got - ref)))
        rows.append({"n": n, "max_error": err,
                     "cond": float(np.linalg.cond(double_layer_matrix(curve)))})
        best = min(best, err)
    _record(results, "nystrom_star_spectral", best < 1e-12,
            "interior Dirichlet solve on r(phi) = 0.55 (1 + 0.3 cos 5 phi) with "
            "harmonic data log|x - z0|, max error at 41 interior probes at half "
            "the local radius", best)
    results[-1]["rows"] = rows


def _test_nystrom_gradient(results: list) -> None:
    """Gradient of the double-layer representation against the exact gradient."""
    curve = star_curve(512)
    mu = solve_interior_dirichlet(curve, _harmonic_field(curve.points))
    probes = _interior_probes(0.5)
    got = double_layer_grad(curve, mu, probes)
    ref = _harmonic_grad(probes)
    err = float(np.max(np.abs(got - ref)))
    probes_near = _interior_probes(0.9)
    err_near = float(np.max(np.abs(
        double_layer_grad(curve, mu, probes_near) - _harmonic_grad(probes_near))))
    _record(results, "nystrom_gradient", err < 1e-12,
            "gradient at half the local radius; at 0.9 of the local radius the "
            f"smooth rule degrades to {err_near:.2e} as expected", err)


def _test_nystrom_second_field(results: list) -> None:
    """A second harmonic field, to rule out a lucky representation."""
    curve = star_curve(400)

    def field(points):
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        z = pts[:, 0] + 1j * pts[:, 1]
        return np.real((z - (1.4 - 0.6j)) ** -2 + z ** 3)

    mu = solve_interior_dirichlet(curve, field(curve.points))
    probes = _interior_probes(0.6)
    err = float(np.max(np.abs(double_layer_eval(curve, mu, probes)
                              - field(probes))))
    _record(results, "nystrom_second_field", err < 1e-12,
            "Re((z - z0)^{-2} + z^3) reproduced at 41 interior probes", err)


def _test_manufactured(results: list) -> None:
    """The manufactured pair must satisfy ``-Laplacian u = rho``."""
    rng = np.random.default_rng(20260918)
    pts = rng.uniform(-0.7, 0.7, size=(64, 2))
    h = 1.0e-4
    shifts = np.array([[h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    lap = -4.0 * u_exact(pts)
    for shift in shifts:
        lap = lap + u_exact(pts + shift)
    lap = lap / (h * h)
    err = float(np.max(np.abs(-lap - rho_manufactured(pts))))
    gh = 1.0e-6
    gx = (u_exact(pts + np.array([gh, 0.0]))
          - u_exact(pts - np.array([gh, 0.0]))) / (2.0 * gh)
    gy = (u_exact(pts + np.array([0.0, gh]))
          - u_exact(pts - np.array([0.0, gh]))) / (2.0 * gh)
    gerr = float(np.max(np.abs(np.stack([gx, gy], axis=1)
                               - grad_u_exact(pts))))
    _record(results, "manufactured_pair", err < 1e-6 and gerr < 1e-8,
            "u = sin(2x) cosh(2y) + exp(-2 r^2), rho = exp(-2 r^2)(8 - 16 r^2); "
            f"finite-difference gradient residual {gerr:.2e}", err)


def _test_bvp_skeleton(results: list) -> None:
    """The BVP assembly with a known volume potential stands in for V[rho].

    With ``V`` replaced by a smooth harmonic-free stand-in whose values on the
    boundary and at the probes are both known, ``u = V + w`` must reproduce the
    target field; this checks the plumbing of the manufactured route without the
    volume potential, which another module owns.
    """
    curve = star_curve(400)
    probes = _interior_probes(0.55)

    def stand_in(points):
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.exp(-2.0 * (pts[:, 0] ** 2 + pts[:, 1] ** 2))

    target = u_exact
    mu = solve_interior_dirichlet(curve, target(curve.points)
                                  - stand_in(curve.points))
    got = stand_in(probes) + double_layer_eval(curve, mu, probes)
    err = float(np.max(np.abs(got - target(probes))))
    _record(results, "bvp_assembly", err < 1e-12,
            "u = V + w with a stand-in V of known boundary and interior values; "
            "the harmonic correction closes the manufactured solution", err)


def _test_closest_point(results: list) -> None:
    """Newton closest-point search on the star boundary."""
    phi_ref = np.array([0.3, 1.1, 2.7, 4.4, 5.9])
    pos, dpos, _ = star_position(phi_ref)
    speed = np.hypot(dpos[:, 0], dpos[:, 1])
    nrm = np.stack([dpos[:, 1], -dpos[:, 0]], axis=1) / speed[:, None]
    dist = 0.01
    worst = 0.0
    for k in range(phi_ref.size):
        probe = pos[k] - dist * nrm[k]
        phi0 = math.atan2(probe[1], probe[0])
        _, found = closest_point_on_curve(star_position, probe, phi0)
        worst = max(worst, float(np.linalg.norm(found - pos[k])))
    _record(results, "closest_point_newton", worst < 1e-12,
            "probes placed one hundredth inward along the normal recover their "
            "boundary point", worst)


def run_self_test(quick: bool = False) -> tuple[list, bool]:
    """Run every check of this module; returns ``(records, all_passed)``."""
    results: list = []
    _test_radial_moment(results)
    _test_reference_self_convergence(results)
    _test_halfplane_exact(results)
    _test_sign_of_eta_term(results)
    _test_delta_five_half_law(results)
    if not quick:
        _test_disk_curvature(results)
    _test_nystrom_circle(results)
    _test_nystrom_convergence(results)
    _test_nystrom_gradient(results)
    if not quick:
        _test_nystrom_second_field(results)
    _test_manufactured(results)
    _test_bvp_skeleton(results)
    _test_closest_point(results)
    return results, all(rec["result"] == "pass" for rec in results)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the self test and print (and optionally save) a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="directory for the JSON summary")
    parser.add_argument("--quick", action="store_true", help="skip the slow checks")
    args = parser.parse_args(list(argv) if argv is not None else None)

    records, ok = run_self_test(quick=args.quick)
    summary = {
        "module": "experiment_e_baseline",
        "all_passed": ok,
        "lemma_4_5_terms": lemma_4_5_terms(),
        "manufactured": {
            "u_exact": "sin(2 x) cosh(2 y) + exp(-2 (x^2 + y^2))",
            "rho": "exp(-2 (x^2 + y^2)) (8 - 16 (x^2 + y^2))",
        },
        "tests": records,
        "environment": environment_summary(),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "experiment_e_baseline_tests.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
