"""Shared contracts for Experiment E: 2D windowed-DMK assembly on complex geometry.

This module is the single place where the data structures, kernels, windows,
plane-wave grids, tree, probe sets and Gaussian-smoothing quadrature of the
two-dimensional leaf-closure study are defined.  The hierarchy, geometry,
baseline and driver modules import from here and add nothing to this vocabulary.

Problem and normalization
-------------------------
Two-dimensional Laplace on a root box ``[-1, 1]^2`` (``ROOT_HALF = 1``).  The
physical free-space kernel is

    K(r) = -log(r) / (2 pi),

so that ``-Delta (K * rho) = rho``.  All *window* kernels in this module are
carried in **chi units**, i.e. multiplied by ``2 pi``, because that is the
normalization in which the telescoping identity has no stray constants:

    -log r = W_0(r) + sum_{l=0}^{L-1} D_l(r) + chi_0(r; t_L),
    chi_0(r; t) = E_1(r^2 / (4 t)) / 2,
    D_l(r)      = chi_0(r; t_l) - chi_0(r; t_{l+1}),
    W_0(r)      = -log r - chi_0(r; t_0).

The physical potential contributed by any stage ``S`` is therefore
``(2 pi)^{-1} (S * rho)``.  Multiply by ``INV_TWO_PI`` exactly once, at the
point where a stage kernel becomes a potential; never inside these helpers.
``chi_0(r; t) = 2 pi int_0^t G_u(r) du`` with ``G_u`` the 2D heat kernel
``heat_kernel``, which is where the ``2 pi`` comes from and why the per-stage
Laplacian identities of the plan read

    -Delta[(2 pi)^{-1} W_0 * rho]           = G_{t_0} * rho,
    -Delta[(2 pi)^{-1} D_l * rho]           = (G_{t_{l+1}} - G_{t_l}) * rho,
    -Delta[(2 pi)^{-1} chi_0(.; t_L) * rho] = rho - G_{t_L} * rho,

summing to ``rho`` inside ``Omega`` and to ``0`` outside.  ``heat_smoothed_density``
computes every right-hand side above.

Units
-----
Lengths, box centres, half-widths, vertices, probe coordinates and signed
distances are *physical* (root box ``[-1, 1]^2``).  ``h_level(l)`` is the leaf
**side length** ``2^{1-l}`` at level ``l``, not the half-width; ``Box.half`` is
the half-width, so ``box.side == h_level(box.level)``.  Heat times
``t_l = (h_l / Theta)^2`` are squared lengths.  Wavenumbers are physical inverse
lengths.  Box-unit quantities never appear in this module.

Sign conventions
----------------
* Signed-distance callables are **negative inside** ``Omega`` and positive
  outside; ``probe_sets`` and ``project_to_boundary`` assume that.
* Polygon and piece vertices are **counter-clockwise**, so ``area()`` is
  positive; ``polygon_area`` returns the signed value so callers can detect a
  reversed loop.
* The Fourier multiplier ``shell_multiplier`` is the transform of the *physical*
  shell ``D_l / (2 pi)``, i.e. ``int_{t_{l+1}}^{t_l} exp(-u |k|^2) du``; it is
  non-negative and has the removable value ``t_l - t_{l+1}`` at ``k = 0``.

Exploratory campaign code: no warm-up, no repetition statistics, no claim that
any rule here is optimal.

Self-test
---------
    python -m experiments.windowed_dmk.experiment_e_common [--out DIR] [--quick]

or equivalently ``python experiments/windowed_dmk/experiment_e_common.py``.
The self-test covers quadrature exactness on box and polygon pieces, curved-piece
area against closed forms, LeafPoly reproduction of a polynomial, the ``k = 0``
limit of the multiplier, the plane-wave truncation criterion, and the backbone
per-stage Laplacian identity on a Gaussian density in the full-box case against
an independent closed-form Gaussian-smoothing reference.

One measured limitation, recorded here because a caller can trip over it: the
*coefficients* returned by ``LeafPoly.monomial_coeffs`` carry an absolute error
of order ``eps / half^{i+j}``, which at ``q = 6`` on a level-3 leaf is about
``1e-7`` relative to the leading coefficient.  That is the intrinsic conditioning
of the monomial basis on a small box, not a defect of the conversion; polynomial
*values* reconstructed from those coefficients near the leaf are accurate to
roundoff (measured ``2e-15``) because the amplification cancels.  Consumers that
need the coefficients themselves should pass an ``origin`` near the piece.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.special import erf, exp1

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EULER_GAMMA = 0.5772156649015328606
"""Euler-Mascheroni constant, used by the stable ``E_1`` series."""

TWO_PI = 2.0 * math.pi
"""``2 pi``: the factor between chi-unit window kernels and physical potentials."""

INV_TWO_PI = 1.0 / TWO_PI
"""``1 / (2 pi)``: multiply a chi-unit stage kernel by this to get a potential."""

ROOT_HALF = 1.0
"""Half-width of the root box ``[-1, 1]^2``, in physical units."""

THETA_PRIMARY = 8.0
"""Primary window declaration ``Theta``; ``12 sqrt(t_L) = 1.5 h_L`` (colleague range)."""

THETA_SECONDARY = 12.0
"""Secondary window declaration ``Theta`` used for the sensitivity column."""

NU_PERIOD = 6.0
"""Periodization factor: the plane-wave trapezoidal period is ``nu h_l``."""

WINDOW_DECAY_SIGMAS = 12.0
"""Radius, in units of ``sqrt(t)``, beyond which the window is below ``1e-14``."""

PIECE_KINDS = ("box", "polygon", "curved")
"""Allowed ``Piece.kind`` values, in the order used by the plan."""


# ---------------------------------------------------------------------------
# elementary quadrature helpers
# ---------------------------------------------------------------------------


def gauss_legendre(npoints: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``[-1, 1]`` (physical units elsewhere)."""
    return np.polynomial.legendre.leggauss(int(npoints))


def gauss_points_for_degree(degree: int) -> int:
    """Number of Gauss points per direction that integrates degree ``degree`` exactly."""
    return max(1, (int(degree) + 2) // 2)


def _gauss_on(interval_lo: float, interval_hi: float, npoints: int):
    """Gauss-Legendre nodes and weights mapped to ``[lo, hi]`` (physical units)."""
    gx, gw = gauss_legendre(npoints)
    mid = 0.5 * (interval_lo + interval_hi)
    half = 0.5 * (interval_hi - interval_lo)
    return mid + half * gx, half * gw


def _composite_gauss(lo: float, hi: float, npoints: int, npanels: int):
    """Composite Gauss nodes and weights over ``npanels`` equal panels of ``[lo, hi]``."""
    edges = np.linspace(lo, hi, int(npanels) + 1)
    gx, gw = gauss_legendre(npoints)
    mid = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    nodes = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    weights = (half[:, None] * gw[None, :]).ravel()
    return nodes, weights


# ---------------------------------------------------------------------------
# polygon geometry
# ---------------------------------------------------------------------------


def polygon_area(vertices: np.ndarray) -> float:
    """Signed area of a simple polygon in physical units (positive when CCW)."""
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.shape[0] < 3:
        return 0.0
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def clip_polygon(vertices: np.ndarray, half_planes: Sequence) -> np.ndarray:
    """Sutherland-Hodgman clip of a CCW polygon by ``(point, normal)`` half-planes.

    A half-plane keeps the set ``(y - point) . normal <= 0``; the result stays CCW.
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


def _point_in_triangle(p, a, b, c, tol: float) -> bool:
    """True when ``p`` lies strictly inside triangle ``abc`` (physical units)."""
    d1 = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])
    d2 = (c[0] - b[0]) * (p[1] - b[1]) - (p[0] - b[0]) * (c[1] - b[1])
    d3 = (a[0] - c[0]) * (p[1] - c[1]) - (p[0] - c[0]) * (a[1] - c[1])
    return d1 > tol and d2 > tol and d3 > tol


def triangulate_polygon(vertices: np.ndarray) -> np.ndarray:
    """Ear-clip a simple (possibly non-convex) CCW polygon into positive triangles.

    Returns an array of shape ``(m, 3, 2)`` in physical units.  Ear clipping keeps
    every triangle inside the polygon, which matters for the peaked heat kernel:
    a signed fan would place cancelling nodes outside the piece.  If clipping
    stalls (a degenerate loop) the remaining vertices fall back to a signed fan,
    which ``triangle_gauss`` integrates correctly because it keeps the sign.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.shape[0] < 3:
        return np.zeros((0, 3, 2))
    if polygon_area(verts) < 0.0:
        verts = verts[::-1].copy()
    extent = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    tol = 1e-13 * max(extent * extent, 1e-30)
    idx = list(range(verts.shape[0]))
    tris: list[tuple] = []
    guard = 0
    guard_max = 4 * verts.shape[0] * verts.shape[0] + 16
    while len(idx) > 3 and guard < guard_max:
        guard += 1
        n = len(idx)
        clipped = False
        for a in range(n):
            i0, i1, i2 = idx[(a - 1) % n], idx[a], idx[(a + 1) % n]
            p0, p1, p2 = verts[i0], verts[i1], verts[i2]
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (
                p1[1] - p0[1]
            )
            if cross <= tol:
                continue
            if any(
                _point_in_triangle(verts[k], p0, p1, p2, -tol)
                for k in idx
                if k not in (i0, i1, i2)
            ):
                continue
            tris.append((p0, p1, p2))
            idx.pop(a)
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        tris.append((verts[idx[0]], verts[idx[1]], verts[idx[2]]))
    elif len(idx) > 3:
        for a in range(1, len(idx) - 1):
            tris.append((verts[idx[0]], verts[idx[a]], verts[idx[a + 1]]))
    if not tris:
        return np.zeros((0, 3, 2))
    return np.array(tris, dtype=np.float64)


def triangle_gauss(triangle, npoints: int) -> tuple[np.ndarray, np.ndarray]:
    """Collapsed (Duffy) tensor-Gauss rule on a triangle; weights carry the sign.

    ``npoints`` is the number of Gauss points per collapsed direction; the rule is
    exact for polynomials of total degree ``2 npoints - 2`` because of the extra
    Jacobian factor.  Nodes and weights are in physical units.
    """
    v0, v1, v2 = np.asarray(triangle, dtype=np.float64)
    gx, gw = gauss_legendre(npoints)
    s = 0.5 * (gx + 1.0)
    ws = 0.5 * gw
    xi = s[:, None]
    eta = s[None, :]
    points = (
        (1.0 - xi)[:, :, None] * v0[None, None, :]
        + (xi * (1.0 - eta))[:, :, None] * v1[None, None, :]
        + (xi * eta)[:, :, None] * v2[None, None, :]
    )
    signed_area = 0.5 * (
        (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
    )
    weights = 2.0 * signed_area * xi * ws[:, None] * ws[None, :]
    return points.reshape(-1, 2), weights.reshape(-1)


def split_triangle(triangle) -> list[np.ndarray]:
    """Split a triangle into four similar triangles by edge midpoints (orientation kept)."""
    v0, v1, v2 = np.asarray(triangle, dtype=np.float64)
    m01 = 0.5 * (v0 + v1)
    m12 = 0.5 * (v1 + v2)
    m20 = 0.5 * (v2 + v0)
    return [
        np.array([v0, m01, m20]),
        np.array([m01, v1, m12]),
        np.array([m20, m12, v2]),
        np.array([m01, m12, m20]),
    ]


def _triangle_diameter(triangle) -> float:
    """Longest edge of a triangle, in physical units."""
    v = np.asarray(triangle, dtype=np.float64)
    return float(
        max(
            np.hypot(*(v[1] - v[0])),
            np.hypot(*(v[2] - v[1])),
            np.hypot(*(v[0] - v[2])),
        )
    )


def graded_triangles(triangles, max_diameter: float | None) -> list[np.ndarray]:
    """Refine triangles by midpoint splitting until each diameter is at most ``max_diameter``."""
    out = [np.asarray(t, dtype=np.float64) for t in triangles]
    if max_diameter is None or max_diameter <= 0.0:
        return out
    done: list[np.ndarray] = []
    work = list(out)
    guard = 0
    while work and guard < 40:
        guard += 1
        nxt: list[np.ndarray] = []
        for tri in work:
            if _triangle_diameter(tri) <= max_diameter:
                done.append(tri)
            else:
                nxt.extend(split_triangle(tri))
        work = nxt
    done.extend(work)
    return done


def polygon_monomial_moment(vertices: np.ndarray, i: int, j: int) -> float:
    """Exact ``int_P x^i y^j dA`` over a CCW polygon by the divergence theorem.

    Used as the reference for the quadrature-exactness self-test and available to
    the geometry module for closed-form checks.  Physical units.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.shape[0] < 3:
        return 0.0
    npts = gauss_points_for_degree(i + j + 2)
    gx, gw = gauss_legendre(npts)
    tau = 0.5 * (gx + 1.0)
    wtau = 0.5 * gw
    total = 0.0
    n = verts.shape[0]
    for e in range(n):
        a = verts[e]
        b = verts[(e + 1) % n]
        dy = b[1] - a[1]
        if dy == 0.0:
            continue
        px = a[0] + tau * (b[0] - a[0])
        py = a[1] + tau * (b[1] - a[1])
        total += dy * float(np.sum(wtau * px ** (i + 1) * py**j))
    return total / (i + 1)


# ---------------------------------------------------------------------------
# boxes, pieces, leaves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned square cell: physical centre, half-width and tree level."""

    center: tuple[float, float]
    half: float
    level: int

    @property
    def side(self) -> float:
        """Side length ``h`` of the box in physical units (``2 * half``)."""
        return 2.0 * self.half

    @property
    def low(self) -> np.ndarray:
        """Lower-left corner of the box, physical units."""
        return np.array(self.center, dtype=np.float64) - self.half

    @property
    def high(self) -> np.ndarray:
        """Upper-right corner of the box, physical units."""
        return np.array(self.center, dtype=np.float64) + self.half

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Box bounds as ``(x_lo, x_hi, y_lo, y_hi)`` in physical units."""
        lo, hi = self.low, self.high
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])

    def vertices(self) -> np.ndarray:
        """Counter-clockwise corners of the box, shape ``(4, 2)``, physical units."""
        a1, b1, a2, b2 = self.bounds
        return np.array(
            [[a1, a2], [b1, a2], [b1, b2], [a1, b2]], dtype=np.float64
        )

    def contains(self, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Boolean mask of points inside the closed box, with an optional tolerance."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        lo, hi = self.low - tol, self.high + tol
        return np.all((pts >= lo) & (pts <= hi), axis=1)

    def piece(self) -> "Piece":
        """The full-box ``Piece`` for this cell (kind ``box``)."""
        return make_box_piece(self)


@dataclass(frozen=True)
class Segment:
    """A straight boundary edge from ``p0`` to ``p1``, parametrized on ``s in [0, 1]``."""

    p0: np.ndarray
    p1: np.ndarray
    s0: float = 0.0
    s1: float = 1.0
    is_straight: bool = True

    def curve(self, s: np.ndarray) -> np.ndarray:
        """Points on the segment at parameters ``s`` (shape ``(n, 2)``, physical units)."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        p0 = np.asarray(self.p0, dtype=np.float64)
        p1 = np.asarray(self.p1, dtype=np.float64)
        return p0[None, :] + s[:, None] * (p1 - p0)[None, :]

    def dcurve(self, s: np.ndarray) -> np.ndarray:
        """Tangent ``d curve / ds`` at parameters ``s`` (shape ``(n, 2)``)."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        p0 = np.asarray(self.p0, dtype=np.float64)
        p1 = np.asarray(self.p1, dtype=np.float64)
        return np.tile((p1 - p0)[None, :], (s.size, 1))


@dataclass
class Arc:
    """A parametrized boundary arc with ``curve(s)`` and ``dcurve(s)`` on ``[s0, s1]``.

    The arc is traversed in the counter-clockwise sense of the enclosing piece, so
    the interior of the piece lies to the left of ``dcurve``.
    """

    curve_fn: Callable[[np.ndarray], np.ndarray]
    dcurve_fn: Callable[[np.ndarray], np.ndarray]
    s0: float
    s1: float
    is_straight: bool = False

    def curve(self, s: np.ndarray) -> np.ndarray:
        """Points on the arc at parameters ``s`` (shape ``(n, 2)``, physical units)."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        return np.asarray(self.curve_fn(s), dtype=np.float64).reshape(s.size, 2)

    def dcurve(self, s: np.ndarray) -> np.ndarray:
        """Tangent ``d curve / ds`` at parameters ``s`` (shape ``(n, 2)``)."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        return np.asarray(self.dcurve_fn(s), dtype=np.float64).reshape(s.size, 2)

    @property
    def p0(self) -> np.ndarray:
        """Start point of the arc, physical units."""
        return self.curve(np.array([self.s0]))[0]

    @property
    def p1(self) -> np.ndarray:
        """End point of the arc, physical units."""
        return self.curve(np.array([self.s1]))[0]


class Piece:
    """A physical sub-region of a leaf: kind ``box``, ``polygon`` or ``curved``.

    ``vertices`` is the counter-clockwise vertex loop (for ``curved`` pieces, the
    chord polygon through the edge endpoints); ``edges`` is the ordered list of
    ``Segment`` and ``Arc`` objects for ``curved`` pieces and ``None`` otherwise.
    All coordinates are physical.
    """

    def __init__(self, kind: str, vertices, edges=None, box: Box | None = None):
        """Build a piece of the given kind from CCW vertices and optional edges."""
        if kind not in PIECE_KINDS:
            raise ValueError(f"unknown piece kind {kind!r}")
        self.kind = kind
        verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
        if verts.shape[0] >= 3 and polygon_area(verts) < 0.0 and kind != "curved":
            verts = verts[::-1].copy()
        self.vertices = verts
        self.edges = list(edges) if edges is not None else None
        self.box = box

    def __repr__(self) -> str:
        """Short representation naming the kind and the vertex count."""
        return f"Piece(kind={self.kind!r}, nverts={self.vertices.shape[0]})"

    # -- geometry -----------------------------------------------------------

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds ``(x_lo, x_hi, y_lo, y_hi)`` of the piece, physical units."""
        pts = [self.vertices] if self.vertices.shape[0] else []
        if self.kind == "curved" and self.edges:
            sample = np.linspace(0.0, 1.0, 17)
            for edge in self.edges:
                s = edge.s0 + sample * (edge.s1 - edge.s0)
                pts.append(edge.curve(s))
        if not pts:
            return (0.0, 0.0, 0.0, 0.0)
        allp = np.vstack(pts)
        return (
            float(allp[:, 0].min()),
            float(allp[:, 0].max()),
            float(allp[:, 1].min()),
            float(allp[:, 1].max()),
        )

    def area(self) -> float:
        """Area of the piece in physical units (closed form for box and polygon)."""
        if self.kind == "box":
            a1, b1, a2, b2 = self._box_bounds()
            return max(0.0, b1 - a1) * max(0.0, b2 - a2)
        if self.kind == "polygon":
            return abs(polygon_area(self.vertices))
        total = polygon_area(self.vertices) if self.vertices.shape[0] >= 3 else 0.0
        for edge in self.edges or []:
            if getattr(edge, "is_straight", False):
                continue
            panels_s, panels_sig = self._lune_panels(edge, None)
            _, weights = self._lune_quadrature(edge, 16, 8, panels_s, panels_sig)
            total += float(np.sum(weights))
        return total

    def _box_bounds(self) -> tuple[float, float, float, float]:
        """Bounds of a ``box`` piece, taken from its vertex loop."""
        v = self.vertices
        return (
            float(v[:, 0].min()),
            float(v[:, 0].max()),
            float(v[:, 1].min()),
            float(v[:, 1].max()),
        )

    def _lune_quadrature(self, edge, n_s: int, n_sigma: int, panels_s: int = 1,
                         panels_sigma: int = 1):
        """Signed quadrature of the region between an arc and its chord, physical units.

        The map ``x(s, sigma) = (1 - sigma) C(s) + sigma gamma(s)`` interpolates
        from the chord ``C`` to the arc ``gamma``; the Jacobian
        ``cross(dx/dsigma, dx/ds)`` is positive when the arc bulges away from the
        interior, so a concave arc subtracts automatically.
        """
        s_nodes, s_w = _composite_gauss(edge.s0, edge.s1, n_s, panels_s)
        g_nodes, g_w = _composite_gauss(0.0, 1.0, n_sigma, panels_sigma)
        p_start = edge.curve(np.array([edge.s0]))[0]
        p_end = edge.curve(np.array([edge.s1]))[0]
        span = edge.s1 - edge.s0
        lam = (s_nodes - edge.s0) / span
        chord = p_start[None, :] + lam[:, None] * (p_end - p_start)[None, :]
        dchord = ((p_end - p_start) / span)[None, :]
        gamma = edge.curve(s_nodes)
        dgamma = edge.dcurve(s_nodes)
        sig = g_nodes[None, :, None]
        pts = (1.0 - sig) * chord[:, None, :] + sig * gamma[:, None, :]
        dx_ds = (1.0 - sig) * dchord[:, None, :] + sig * dgamma[:, None, :]
        dx_dsig = (gamma - chord)[:, None, :]
        jac = (
            dx_dsig[:, :, 0] * dx_ds[:, :, 1] - dx_dsig[:, :, 1] * dx_ds[:, :, 0]
        )
        weights = s_w[:, None] * g_w[None, :] * jac
        return pts.reshape(-1, 2), weights.reshape(-1)

    def triangles(self, max_diameter: float | None = None) -> list[np.ndarray]:
        """Triangulation of the piece's straight part, refined to ``max_diameter``."""
        tris = triangulate_polygon(self.vertices)
        return graded_triangles(list(tris), max_diameter)

    # -- quadrature ---------------------------------------------------------

    def quadrature(
        self, order: int, max_diameter: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nodes and weights integrating total degree ``order`` exactly on box/polygon.

        ``order`` is the polynomial total degree reproduced exactly (exact for the
        ``box`` and ``polygon`` kinds; for ``curved`` pieces the straight part is
        exact and the arc lunes are resolved to the rule's own accuracy).
        ``max_diameter`` produces a composite rule whose cells all have diameter at
        most that value, which is what a peaked heat kernel needs.  Nodes are
        physical points of shape ``(m, 2)``; weights carry the area element.
        """
        order = int(order)
        if self.kind == "box":
            return self._box_quadrature(order, max_diameter)
        nodes: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        npts_tri = gauss_points_for_degree(order + 1)
        for tri in self.triangles(max_diameter):
            p, w = triangle_gauss(tri, npts_tri)
            nodes.append(p)
            weights.append(w)
        if self.kind == "curved":
            npts = gauss_points_for_degree(order + 2)
            for edge in self.edges or []:
                if getattr(edge, "is_straight", False):
                    continue
                panels_s, panels_sig = self._lune_panels(edge, max_diameter)
                p, w = self._lune_quadrature(
                    edge, npts, npts, panels_s, panels_sig
                )
                nodes.append(p)
                weights.append(w)
        if not nodes:
            return np.zeros((0, 2)), np.zeros(0)
        return np.vstack(nodes), np.concatenate(weights)

    def _lune_panels(self, edge, max_diameter: float | None) -> tuple[int, int]:
        """Panel counts in ``(s, sigma)`` for a lune: curvature first, then ``max_diameter``.

        Each ``s`` panel is limited to ``pi / 8`` of tangent turning so a fixed Gauss
        order resolves the arc regardless of how much it curves, and (when
        ``max_diameter`` is given) to a cell of that physical diameter.
        """
        sample = np.linspace(edge.s0, edge.s1, 65)
        pts = edge.curve(sample)
        tangents = edge.dcurve(sample)
        angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
        turning = float(np.sum(np.abs(np.diff(angles))))
        panels_s = max(1, int(math.ceil(turning / (math.pi / 8.0))))
        panels_sig = 1
        if max_diameter is not None and max_diameter > 0.0:
            arclen = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
            p_start, p_end = pts[0], pts[-1]
            lam = (sample - edge.s0) / (edge.s1 - edge.s0)
            chord = p_start[None, :] + lam[:, None] * (p_end - p_start)[None, :]
            offset = float(np.max(np.hypot(*(pts - chord).T)))
            panels_s = max(panels_s, int(math.ceil(arclen / max_diameter)))
            panels_sig = max(1, int(math.ceil(offset / max_diameter)))
        return panels_s, panels_sig

    def _box_quadrature(self, order: int, max_diameter: float | None):
        """Tensor Gauss rule on a ``box`` piece, subdivided to ``max_diameter``."""
        a1, b1, a2, b2 = self._box_bounds()
        if b1 <= a1 or b2 <= a2:
            return np.zeros((0, 2)), np.zeros(0)
        npts = gauss_points_for_degree(order)
        if max_diameter is None or max_diameter <= 0.0:
            n1 = n2 = 1
        else:
            cell = max_diameter / math.sqrt(2.0)
            n1 = max(1, int(math.ceil((b1 - a1) / cell)))
            n2 = max(1, int(math.ceil((b2 - a2) / cell)))
        x, wx = _composite_gauss(a1, b1, npts, n1)
        y, wy = _composite_gauss(a2, b2, npts, n2)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        ww = wx[:, None] * wy[None, :]
        return np.stack([xx.ravel(), yy.ravel()], axis=1), ww.ravel()


def make_box_piece(box: Box) -> Piece:
    """Full-box piece of a cell (kind ``box``); vertices CCW, physical units."""
    return Piece("box", box.vertices(), None, box)


def make_polygon_piece(vertices, box: Box | None = None) -> Piece:
    """Straight-sided piece from a CCW vertex loop (kind ``polygon``)."""
    return Piece("polygon", vertices, None, box)


def make_curved_piece(edges: Sequence, box: Box | None = None) -> Piece:
    """Piece bounded by an ordered CCW list of ``Segment`` and ``Arc`` edges."""
    verts = []
    for edge in edges:
        if isinstance(edge, Segment):
            verts.append(np.asarray(edge.p0, dtype=np.float64))
        else:
            verts.append(edge.p0)
    return Piece("curved", np.array(verts, dtype=np.float64), list(edges), box)


@dataclass
class Leaf:
    """A leaf cell with its physical pieces and (optionally) its density proxy."""

    box: Box
    pieces: list = field(default_factory=list)
    poly: "LeafPoly | None" = None

    def area(self) -> float:
        """Total physical area of the leaf's pieces (zero for an exterior leaf)."""
        return float(sum(p.area() for p in self.pieces))

    def is_cut(self) -> bool:
        """True when the leaf carries anything other than a single full-box piece."""
        return not (len(self.pieces) == 1 and self.pieces[0].kind == "box")

    def quadrature(self, order: int, max_diameter: float | None = None):
        """Concatenated quadrature of all pieces of the leaf, physical units."""
        nodes, weights = [], []
        for piece in self.pieces:
            p, w = piece.quadrature(order, max_diameter)
            nodes.append(p)
            weights.append(w)
        if not nodes:
            return np.zeros((0, 2)), np.zeros(0)
        return np.vstack(nodes), np.concatenate(weights)


# ---------------------------------------------------------------------------
# per-leaf tensor-Lagrange density proxy
# ---------------------------------------------------------------------------


def _leg2poly_2d(coeffs: np.ndarray) -> np.ndarray:
    """Convert 2D Legendre coefficients to monomial coefficients in the same variables."""
    leg2poly = np.polynomial.legendre.leg2poly
    tmp = np.column_stack([leg2poly(coeffs[:, j]) for j in range(coeffs.shape[1])])
    out = np.vstack([leg2poly(tmp[i, :]) for i in range(tmp.shape[0])])
    return out


def _shift_poly_1d(coeffs: np.ndarray, delta: float, axis: int) -> np.ndarray:
    """Re-expand monomial coefficients along ``axis`` from variable ``u`` to ``z = u - delta``."""
    coeffs = np.moveaxis(np.asarray(coeffs, dtype=np.float64), axis, 0)
    n = coeffs.shape[0]
    out = np.zeros_like(coeffs)
    for i in range(n):
        for k in range(i + 1):
            out[k] += math.comb(i, k) * (delta ** (i - k)) * coeffs[i]
    return np.moveaxis(out, 0, axis)


class LeafPoly:
    """Tensor Lagrange proxy of order ``q`` on a leaf's Gauss nodes (physical units).

    The proxy is stored as tensor Legendre coefficients in the leaf's reference
    coordinates ``xi = (x - center) / half``, which is the numerically stable form
    of the same interpolant; ``eval``, ``grad`` and ``laplacian`` accept arbitrary
    physical points (inside or outside the leaf, since the proxy is a polynomial).
    """

    def __init__(self, box: Box, values: np.ndarray, order: int | None = None):
        """Build the proxy from node values laid out as ``values[i, j]`` on the Gauss grid."""
        self.box = box
        vals = np.asarray(values, dtype=np.float64)
        if vals.ndim != 2 or vals.shape[0] != vals.shape[1]:
            raise ValueError("LeafPoly needs a square array of tensor node values")
        self.q = int(order if order is not None else vals.shape[0])
        if self.q != vals.shape[0]:
            raise ValueError("order does not match the value array")
        self.values = vals
        gx, gw = gauss_legendre(self.q)
        self._ref_nodes = gx
        vander = np.polynomial.legendre.legvander(gx, self.q - 1)
        scale = (2.0 * np.arange(self.q) + 1.0) / 2.0
        inner = vander.T @ (gw[:, None] * vals * gw[None, :]) @ vander
        self.legendre_coeffs = scale[:, None] * inner * scale[None, :]

    @classmethod
    def from_callable(cls, box: Box, func: Callable, order: int) -> "LeafPoly":
        """Interpolate a callable ``func(points) -> values`` on the leaf's Gauss grid."""
        pts, shape = _leaf_gauss_grid(box, order)
        vals = np.asarray(func(pts), dtype=np.float64).reshape(shape)
        return cls(box, vals, order)

    def nodes(self) -> np.ndarray:
        """Physical tensor Gauss nodes of the proxy, shape ``(q * q, 2)``, row-major in x."""
        pts, _ = _leaf_gauss_grid(self.box, self.q)
        return pts

    def _reference(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map physical points to the leaf reference square ``[-1, 1]^2``."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        c = np.asarray(self.box.center, dtype=np.float64)
        xi = (pts - c) / self.box.half
        return xi[:, 0], xi[:, 1]

    def eval(self, points: np.ndarray) -> np.ndarray:
        """Proxy value at arbitrary physical points, shape ``(n,)``."""
        x1, x2 = self._reference(points)
        return np.polynomial.legendre.legval2d(x1, x2, self.legendre_coeffs)

    def grad(self, points: np.ndarray) -> np.ndarray:
        """Proxy gradient ``(d/dx, d/dy)`` at physical points, shape ``(n, 2)``."""
        x1, x2 = self._reference(points)
        legder = np.polynomial.legendre.legder
        c = self.legendre_coeffs
        dx = legder(c, 1, axis=0) / self.box.half
        dy = legder(c, 1, axis=1) / self.box.half
        return np.stack(
            [
                np.polynomial.legendre.legval2d(x1, x2, dx),
                np.polynomial.legendre.legval2d(x1, x2, dy),
            ],
            axis=1,
        )

    def laplacian(self, points: np.ndarray) -> np.ndarray:
        """Proxy Laplacian at physical points, shape ``(n,)``."""
        x1, x2 = self._reference(points)
        legder = np.polynomial.legendre.legder
        c = self.legendre_coeffs
        h2 = self.box.half * self.box.half
        dxx = legder(c, 2, axis=0) / h2
        dyy = legder(c, 2, axis=1) / h2
        return np.polynomial.legendre.legval2d(
            x1, x2, dxx
        ) + np.polynomial.legendre.legval2d(x1, x2, dyy)

    def monomial_coeffs(self, origin=(0.0, 0.0)) -> np.ndarray:
        """Coefficients ``c[i, j]`` of ``(x - o1)^i (y - o2)^j`` for the proxy.

        ``origin`` defaults to the physical origin, matching the convention of the
        polar-quadrature helpers carried over from the leaf-residual study.  The
        monomial basis about a *distant* origin is ill-conditioned: recovering
        ``c[i, j]`` from node values amplifies roundoff by
        ``(|origin - center| + half)^{i+j} / half^{i+j}``, so pass an ``origin``
        near the piece (the leaf centre, or the polar target) whenever the
        coefficients themselves matter.  Values reconstructed at points inside the
        leaf are accurate for any ``origin``, because that amplification cancels.
        """
        mono_ref = _leg2poly_2d(self.legendre_coeffs)
        powers = self.box.half ** np.arange(self.q)
        mono = mono_ref / (powers[:, None] * powers[None, :])
        cx = float(origin[0]) - float(self.box.center[0])
        cy = float(origin[1]) - float(self.box.center[1])
        mono = _shift_poly_1d(mono, cx, axis=0)
        mono = _shift_poly_1d(mono, cy, axis=1)
        return mono


def eval_monomial(coeffs: np.ndarray, points: np.ndarray, origin=(0.0, 0.0)) -> np.ndarray:
    """Evaluate ``sum_ij c[i, j] (x - o1)^i (y - o2)^j`` at physical points."""
    c = np.asarray(coeffs, dtype=np.float64)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    dx = pts[:, 0] - float(origin[0])
    dy = pts[:, 1] - float(origin[1])
    px = np.vander(dx, c.shape[0], increasing=True)
    py = np.vander(dy, c.shape[1], increasing=True)
    return np.einsum("ni,ij,nj->n", px, c, py)


def _leaf_gauss_grid(box: Box, order: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Physical tensor Gauss grid of a box, row-major in the first coordinate."""
    gx, _ = gauss_legendre(order)
    c = np.asarray(box.center, dtype=np.float64)
    x = c[0] + box.half * gx
    y = c[1] + box.half * gx
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1), (order, order)


# ---------------------------------------------------------------------------
# windows, kernels and the Fourier multiplier
# ---------------------------------------------------------------------------


def h_level(level: int, root_half: float = ROOT_HALF) -> float:
    """Leaf **side length** ``h_l = 2 root_half / 2^l`` at tree level ``l``, physical units."""
    return 2.0 * root_half / (2.0 ** int(level))


def t_l(level: int, theta: float = THETA_PRIMARY, root_half: float = ROOT_HALF) -> float:
    """Heat time ``t_l = (h_l / Theta)^2`` at level ``l``; a squared physical length."""
    h = h_level(level, root_half)
    return (h / float(theta)) ** 2


def window_radius(t: float, sigmas: float = WINDOW_DECAY_SIGMAS) -> float:
    """Physical radius ``sigmas * sqrt(t)`` beyond which the window is negligible."""
    return float(sigmas) * math.sqrt(float(t))


def e1_plus_log(z):
    """``E_1(z) + log z``, cancellation-free near ``z = 0`` (series below 1, direct above)."""
    z = np.asarray(z, dtype=np.float64)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
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
    return float(out[0]) if scalar else out


def laplace_kernel(r):
    """Physical 2D Laplace kernel ``-log(r) / (2 pi)``; ``-Delta (K * rho) = rho``."""
    return -np.log(np.asarray(r, dtype=np.float64)) * INV_TWO_PI


def heat_kernel(r, t: float):
    """2D heat kernel ``G_t(r) = exp(-r^2 / (4 t)) / (4 pi t)``, physical normalization."""
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-r * r / (4.0 * float(t))) / (4.0 * math.pi * float(t))


def chi_0(r, t: float):
    """Window ``chi_0(r; t) = E_1(r^2 / (4 t)) / 2`` in chi units (``2 pi`` times physical)."""
    r = np.asarray(r, dtype=np.float64)
    return 0.5 * exp1(r * r / (4.0 * float(t)))


def coarse_kernel(r, t0: float):
    """``W_0(r) = -log r - chi_0(r; t_0)`` in chi units, evaluated without cancellation.

    Analytically ``W_0(r) = -(E_1(z) + log z + log(4 t_0)) / 2`` with ``z = r^2/(4 t_0)``,
    which is smooth at ``r = 0``; that is the form used here.
    """
    r = np.asarray(r, dtype=np.float64)
    z = r * r / (4.0 * float(t0))
    return -0.5 * (e1_plus_log(z) + math.log(4.0 * float(t0)))


def shell_kernel(r, t_coarse: float, t_fine: float):
    """Shell ``D_l(r) = chi_0(r; t_l) - chi_0(r; t_{l+1})`` in chi units, stably evaluated.

    Below ``r^2 = 4 t_fine`` the difference is formed from ``E_1 + log`` so the log
    germs cancel analytically (the first-wave pitfall: a plain double-precision
    difference of two ``chi_0`` values loses all relative accuracy at coarse
    levels); above it the direct difference is already well conditioned.
    """
    r = np.asarray(r, dtype=np.float64)
    tc, tf = float(t_coarse), float(t_fine)
    z_c = r * r / (4.0 * tc)
    z_f = r * r / (4.0 * tf)
    small = z_f < 1.0
    out = np.empty(np.atleast_1d(z_f).shape, dtype=np.float64)
    zc1 = np.atleast_1d(z_c)
    zf1 = np.atleast_1d(z_f)
    sm = np.atleast_1d(small)
    if np.any(sm):
        out[sm] = 0.5 * (
            e1_plus_log(zc1[sm]) - e1_plus_log(zf1[sm]) + math.log(tc / tf)
        )
    if np.any(~sm):
        out[~sm] = 0.5 * (exp1(zc1[~sm]) - exp1(zf1[~sm]))
    return float(out[0]) if np.ndim(r) == 0 else out.reshape(np.shape(r))


def shell_profile(s):
    """Fixed scaled dyadic shell profile ``S(s) = (E_1(s^2/4) - E_1(s^2)) / 2``, chi units.

    For a dyadic level pair (``t_l = 4 t_{l+1}``) the shell is a rescaling of this
    one profile: ``D_l(r) = shell_profile(r / (2 sqrt(t_{l+1})))``.  Evaluating the
    profile once per level and rescaling is what the first wave's pitfall requires
    instead of differencing two windows at coarse levels.
    """
    s = np.asarray(s, dtype=np.float64)
    return shell_kernel(2.0 * s, 4.0, 1.0)


def shell_peak(t_coarse: float, t_fine: float) -> float:
    """``D_l(0) = log(t_l / t_{l+1}) / 2`` in chi units; the shell's maximum."""
    return 0.5 * math.log(float(t_coarse) / float(t_fine))


def shell_multiplier(k_sq, t_coarse: float, t_fine: float):
    """Fourier multiplier of the **physical** shell ``D_l / (2 pi)`` at ``|k|^2 = k_sq``.

    Equals ``int_{t_{l+1}}^{t_l} exp(-u |k|^2) du``
    ``= (exp(-t_{l+1}|k|^2) - exp(-t_l|k|^2)) / |k|^2`` with the removable value
    ``t_l - t_{l+1}`` at ``k = 0``; formed with ``expm1`` so the small-``k`` regime
    is cancellation-free.  Non-negative for ``t_fine < t_coarse``.
    """
    k_sq = np.asarray(k_sq, dtype=np.float64)
    tc, tf = float(t_coarse), float(t_fine)
    dt = tc - tf
    safe = np.where(k_sq > 0.0, k_sq, 1.0)
    value = -np.expm1(-dt * safe) * np.exp(-tf * safe) / safe
    return np.where(k_sq > 0.0, value, dt)


def telescoping_kernels(levels: int, theta: float = THETA_PRIMARY,
                        root_half: float = ROOT_HALF) -> dict:
    """Heat times and chi-unit stage kernels for a uniform tree of ``levels + 1`` levels.

    Returns ``{"times": [t_0, ..., t_L], "W0": callable, "shells": [callable, ...],
    "closure": callable}``; every callable takes a physical radius and returns chi
    units, so the potential is ``(2 pi)^{-1}`` times the convolution.
    """
    times = [t_l(level, theta, root_half) for level in range(levels + 1)]
    shells = [
        (lambda r, a=times[i], b=times[i + 1]: shell_kernel(r, a, b))
        for i in range(levels)
    ]
    return {
        "times": times,
        "W0": (lambda r, a=times[0]: coarse_kernel(r, a)),
        "shells": shells,
        "closure": (lambda r, a=times[-1]: chi_0(r, a)),
    }


# ---------------------------------------------------------------------------
# tensor plane-wave grid
# ---------------------------------------------------------------------------


@dataclass
class PlaneWaveGrid:
    """A tensor trapezoidal plane-wave grid for one shell, in physical wavenumbers.

    ``k1`` is the per-coordinate wavenumber axis, ascending from ``-n_f dk`` to
    ``+n_f dk``; the tensor arrays ``kx`` and ``ky`` are flattened row-major with
    ``kx`` varying slowest, i.e. ``kx = repeat(k1, n)`` and ``ky = tile(k1, n)``
    with ``n = 2 n_f + 1``.  ``weight = (dk / (2 pi))^2`` is the scalar quadrature
    weight of the inverse transform, so a physical kernel is reconstructed as
    ``weight * sum_m khat(|k_m|) exp(i k_m . x)``.
    """

    dk: float
    n_f: int
    nu: float
    tol: float
    k1: np.ndarray
    weight: float
    trunc_rel: float
    alias_rel: float
    peak: float
    period: float

    @property
    def n_per_coordinate(self) -> int:
        """Number of modes per coordinate, ``2 n_f + 1``."""
        return 2 * self.n_f + 1

    @property
    def n_modes(self) -> int:
        """Total number of tensor modes, ``(2 n_f + 1)^2``."""
        return self.n_per_coordinate**2

    @property
    def kx(self) -> np.ndarray:
        """First wavenumber component of every mode, row-major (varies slowest)."""
        return np.repeat(self.k1, self.n_per_coordinate)

    @property
    def ky(self) -> np.ndarray:
        """Second wavenumber component of every mode, row-major (varies fastest)."""
        return np.tile(self.k1, self.n_per_coordinate)

    def k_squared(self) -> np.ndarray:
        """``|k|^2`` of every mode, in the same row-major ordering as ``kx``/``ky``."""
        kx, ky = self.kx, self.ky
        return kx * kx + ky * ky

    def multiplier(self, t_coarse: float, t_fine: float) -> np.ndarray:
        """Physical shell multiplier evaluated on the grid, same ordering as ``kx``."""
        return shell_multiplier(self.k_squared(), t_coarse, t_fine)


def plane_wave_grid(
    h_l: float,
    t_coarse: float,
    t_fine: float,
    nu: float = NU_PERIOD,
    tol: float = 1e-12,
    n_max: int = 512,
) -> PlaneWaveGrid:
    """Tensor plane-wave grid for a shell: period ``nu h_l``, truncation at ``tol``.

    The period is ``nu h_l`` physical units, so ``dk = 2 pi / (nu h_l)``.  Because
    the multiplier is non-negative, the max-norm truncation error over any
    evaluation set is attained at ``x = 0`` and equals
    ``(dk / 2 pi)^2 sum_{|m|_inf > n_f} Dhat(dk |m|)``; ``n_f`` is the smallest
    value bringing that below ``tol`` times the physical shell peak
    ``D_l(0) / (2 pi)``.  The per-coordinate mode count is ``2 n_f + 1``; it is
    convention-dependent through ``nu`` and is never to be called DMK's ``N_1``.
    """
    period = float(nu) * float(h_l)
    dk = TWO_PI / period
    weight = (dk / TWO_PI) ** 2
    peak = shell_peak(t_coarse, t_fine) * INV_TWO_PI
    m = np.arange(-int(n_max), int(n_max) + 1, dtype=np.float64)
    kx = dk * m
    k_sq = kx[:, None] ** 2 + kx[None, :] ** 2
    dhat = shell_multiplier(k_sq, t_coarse, t_fine)
    inf_norm = np.maximum(np.abs(m)[:, None], np.abs(m)[None, :]).astype(np.int64)
    shell_sums = np.bincount(
        inf_norm.ravel(), weights=dhat.ravel(), minlength=int(n_max) + 1
    )
    cumulative = np.cumsum(shell_sums)
    total = float(cumulative[-1])
    tail = total - cumulative
    threshold = float(tol) * peak / weight
    hits = np.nonzero(tail <= threshold)[0]
    n_f = int(hits[0]) if hits.size else int(n_max)
    trunc_rel = float(tail[n_f] * weight / peak)
    alias = _aliasing_bound(period, h_l, t_coarse, t_fine)
    return PlaneWaveGrid(
        dk=dk,
        n_f=n_f,
        nu=float(nu),
        tol=float(tol),
        k1=dk * np.arange(-n_f, n_f + 1, dtype=np.float64),
        weight=weight,
        trunc_rel=trunc_rel,
        alias_rel=float(alias / peak) if peak > 0.0 else float("inf"),
        peak=peak,
        period=period,
    )


def _aliasing_bound(period: float, h_l: float, t_coarse: float, t_fine: float) -> float:
    """Bound on the periodization error of the physical shell over the colleague range."""
    total = 0.0
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            if n1 == 0 and n2 == 0:
                continue
            d2 = sum(
                max(0.0, abs(ni) * period - 2.0 * h_l) ** 2 for ni in (n1, n2)
            )
            if d2 <= 0.0:
                return float("inf")
            total += float(shell_kernel(math.sqrt(d2), t_coarse, t_fine)) * INV_TWO_PI
    return total


# ---------------------------------------------------------------------------
# uniform quadtree with colleague lists
# ---------------------------------------------------------------------------


class QuadTree:
    """A uniform quadtree over a root box, with per-level colleague (List 1) lists."""

    def __init__(self, root: Box, levels: int):
        """Build ``levels + 1`` uniform levels below and including the root box."""
        self.root = root
        self.levels = int(levels)
        self._levels: list[list[Box]] = []
        for level in range(self.levels + 1):
            n = 2**level
            half = root.half / n
            lo = np.asarray(root.center, dtype=np.float64) - root.half
            boxes = []
            for i in range(n):
                for j in range(n):
                    center = (
                        float(lo[0] + (2 * i + 1) * half),
                        float(lo[1] + (2 * j + 1) * half),
                    )
                    boxes.append(Box(center, float(half), level))
            self._levels.append(boxes)

    def n_side(self, level: int) -> int:
        """Number of boxes per coordinate at ``level``."""
        return 2 ** int(level)

    def h(self, level: int) -> float:
        """Side length of a box at ``level``, physical units."""
        return 2.0 * self.root.half / self.n_side(level)

    def boxes(self, level: int) -> list:
        """All boxes at ``level``, row-major in the first coordinate index."""
        return self._levels[int(level)]

    def box_at(self, level: int, i: int, j: int) -> Box:
        """Box with grid indices ``(i, j)`` at ``level``."""
        return self._levels[int(level)][int(i) * self.n_side(level) + int(j)]

    def leaves(self) -> list:
        """All boxes at the finest level."""
        return self._levels[self.levels]

    def indices_of(self, box: Box) -> tuple[int, int]:
        """Grid indices ``(i, j)`` of a box at its own level."""
        n = self.n_side(box.level)
        lo = np.asarray(self.root.center, dtype=np.float64) - self.root.half
        i = int(round((box.center[0] - lo[0]) / (2.0 * box.half) - 0.5))
        j = int(round((box.center[1] - lo[1]) / (2.0 * box.half) - 0.5))
        return max(0, min(n - 1, i)), max(0, min(n - 1, j))

    def locate(self, level: int, point) -> tuple[int, int]:
        """Grid indices of the box at ``level`` containing a physical point (clamped)."""
        n = self.n_side(level)
        lo = np.asarray(self.root.center, dtype=np.float64) - self.root.half
        h = self.h(level)
        i = int(math.floor((float(point[0]) - lo[0]) / h))
        j = int(math.floor((float(point[1]) - lo[1]) / h))
        return max(0, min(n - 1, i)), max(0, min(n - 1, j))

    def colleague_indices(self, level: int, i: int, j: int) -> list:
        """Grid indices of the colleagues of ``(i, j)`` at ``level``, including itself."""
        n = self.n_side(level)
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    out.append((ii, jj))
        return out

    def colleagues(self, box: Box) -> list:
        """Colleague boxes of ``box`` at its own level, including ``box`` itself.

        This is the DMK colleague range and Volumential's List 1 for a uniform
        tree: the up to nine same-level boxes within one box of the target.
        """
        i, j = self.indices_of(box)
        return [self.box_at(box.level, ii, jj) for ii, jj in
                self.colleague_indices(box.level, i, j)]


def build_quadtree(root: Box | None = None, levels: int = 3,
                   root_half: float = ROOT_HALF) -> QuadTree:
    """Uniform quadtree of ``levels + 1`` levels over the root box (default ``[-1, 1]^2``)."""
    if root is None:
        root = Box((0.0, 0.0), float(root_half), 0)
    return QuadTree(root, levels)


# ---------------------------------------------------------------------------
# reproducible probe sets
# ---------------------------------------------------------------------------


def sdf_gradient(sdf: Callable, points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a signed-distance callable, physical units."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ex = np.array([eps, 0.0])
    ey = np.array([0.0, eps])
    gx = (np.asarray(sdf(pts + ex)) - np.asarray(sdf(pts - ex))) / (2.0 * eps)
    gy = (np.asarray(sdf(pts + ey)) - np.asarray(sdf(pts - ey))) / (2.0 * eps)
    return np.stack([gx, gy], axis=1)


def project_to_boundary(
    sdf: Callable, points: np.ndarray, iters: int = 40, eps: float = 1e-6
) -> np.ndarray:
    """Newton-project physical points onto ``{sdf = 0}`` along the signed-distance gradient."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64)).copy()
    for _ in range(int(iters)):
        value = np.asarray(sdf(pts), dtype=np.float64)
        grad = sdf_gradient(sdf, pts, eps)
        norm2 = np.sum(grad * grad, axis=1)
        norm2 = np.where(norm2 > 1e-30, norm2, 1.0)
        pts = pts - (value / norm2)[:, None] * grad
    return pts


def boundary_pairs(
    sdf: Callable, base_points: np.ndarray, distance: float, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Inside/outside probe pairs at signed distance ``-d`` and ``+d`` from the boundary.

    ``base_points`` are seeds; they are projected onto ``{sdf = 0}`` first.  The
    sign convention is negative inside, so the first array is the interior side.
    """
    feet = project_to_boundary(sdf, base_points, eps=eps)
    grad = sdf_gradient(sdf, feet, eps)
    norm = np.hypot(grad[:, 0], grad[:, 1])
    norm = np.where(norm > 1e-30, norm, 1.0)
    normal = grad / norm[:, None]
    return feet - distance * normal, feet + distance * normal


def probe_sets(
    sdf: Callable,
    h_leaf: float,
    t_leaf: float,
    root_half: float = ROOT_HALF,
    n_bulk: int = 32,
    n_band: int = 32,
    corner: Sequence[float] | None = None,
    bisector: Sequence[float] | None = None,
    n_corner: int = 12,
    seed: int = 20260918,
) -> dict:
    """Reproducible probe sets inside and outside ``Omega`` for the residual study.

    Bands are measured in physical distance to ``{sdf = 0}`` (negative inside):
    ``leaf_band`` covers ``[0.1 h_L, h_L]``, ``window_band`` covers
    ``[0.05 sqrt(t_L), 2 sqrt(t_L)]``, and ``bulk`` keeps points at least
    ``2 h_L`` from the boundary.  ``corner``/``bisector`` add a geometric
    sequence of probes along the corner bisector (inside) and its opposite
    (outside).  Every set is deterministic given ``seed``; returns a dict of
    ``(n, 2)`` physical point arrays keyed by set name.
    """
    rng = np.random.default_rng(int(seed))
    margin = 0.25 * float(h_leaf)
    limit = float(root_half) - margin

    def _clip(points):
        """Keep only points within the root box minus one quarter leaf."""
        pts = np.atleast_2d(points)
        if pts.size == 0:
            return pts.reshape(0, 2)
        keep = np.all(np.abs(pts) <= limit, axis=1)
        return pts[keep]

    sets: dict[str, np.ndarray] = {}

    candidates = rng.uniform(-limit, limit, size=(4000, 2))
    dist = np.asarray(sdf(candidates), dtype=np.float64)
    inside = candidates[dist <= -2.0 * float(h_leaf)]
    outside = candidates[dist >= 2.0 * float(h_leaf)]
    sets["bulk_inside"] = inside[: int(n_bulk)]
    sets["bulk_outside"] = outside[: int(n_bulk)]

    seeds = rng.uniform(-limit, limit, size=(6 * int(n_band) + 32, 2))
    feet = project_to_boundary(sdf, seeds)
    feet = _clip(feet)
    if feet.shape[0]:
        grad = sdf_gradient(sdf, feet)
        norm = np.hypot(grad[:, 0], grad[:, 1])
        norm = np.where(norm > 1e-30, norm, 1.0)
        normal = grad / norm[:, None]
        for name, lo, hi in (
            ("leaf_band", 0.1 * float(h_leaf), float(h_leaf)),
            (
                "window_band",
                0.05 * math.sqrt(float(t_leaf)),
                2.0 * math.sqrt(float(t_leaf)),
            ),
        ):
            take = min(int(n_band), feet.shape[0])
            logs = rng.uniform(math.log(lo), math.log(hi), size=take)
            offs = np.exp(logs)
            base = feet[:take]
            nrm = normal[:take]
            sets[f"{name}_inside"] = _clip(base - offs[:, None] * nrm)
            sets[f"{name}_outside"] = _clip(base + offs[:, None] * nrm)

    if corner is not None and bisector is not None:
        c = np.asarray(corner, dtype=np.float64)
        b = np.asarray(bisector, dtype=np.float64)
        b = b / max(float(np.hypot(b[0], b[1])), 1e-30)
        steps = float(h_leaf) * 2.0 ** (-np.arange(int(n_corner), dtype=np.float64))
        sets["corner_inside"] = _clip(c[None, :] + steps[:, None] * b[None, :])
        sets["corner_outside"] = _clip(c[None, :] - steps[:, None] * b[None, :])

    return {k: np.asarray(v, dtype=np.float64).reshape(-1, 2) for k, v in sets.items()}


# ---------------------------------------------------------------------------
# Gaussian-smoothed density: the right-hand side of the per-stage residual
# ---------------------------------------------------------------------------


class _NodeIndex:
    """Uniform bucket index over quadrature nodes, for radius queries in physical units."""

    def __init__(self, points: np.ndarray, cell: float):
        """Bucket ``points`` on a uniform grid of the given physical cell size."""
        self.cell = max(float(cell), 1e-12)
        self.origin = points.min(axis=0) - self.cell
        ij = np.floor((points - self.origin) / self.cell).astype(np.int64)
        self.nx = int(ij[:, 0].max()) + 2
        self.ny = int(ij[:, 1].max()) + 2
        flat = ij[:, 0] * self.ny + ij[:, 1]
        self.order = np.argsort(flat, kind="stable")
        sorted_flat = flat[self.order]
        keys = np.arange(self.nx * self.ny + 1)
        self.start = np.searchsorted(sorted_flat, keys[:-1], side="left")
        self.end = np.searchsorted(sorted_flat, keys[:-1], side="right")

    def query(self, point) -> np.ndarray:
        """Indices of nodes in the 3x3 bucket neighbourhood of a physical point."""
        i = int(math.floor((float(point[0]) - self.origin[0]) / self.cell))
        j = int(math.floor((float(point[1]) - self.origin[1]) / self.cell))
        parts = []
        for di in (-1, 0, 1):
            ii = i + di
            if ii < 0 or ii >= self.nx:
                continue
            for dj in (-1, 0, 1):
                jj = j + dj
                if jj < 0 or jj >= self.ny:
                    continue
                key = ii * self.ny + jj
                parts.append(self.order[self.start[key] : self.end[key]])
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(parts)


def domain_quadrature(
    domain_pieces, order: int = 10, max_diameter: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated quadrature over a piece or an iterable of pieces, physical units."""
    if isinstance(domain_pieces, Piece):
        pieces = [domain_pieces]
    elif isinstance(domain_pieces, Leaf):
        pieces = list(domain_pieces.pieces)
    else:
        pieces = []
        for item in domain_pieces:
            if isinstance(item, Leaf):
                pieces.extend(item.pieces)
            else:
                pieces.append(item)
    nodes, weights = [], []
    for piece in pieces:
        p, w = piece.quadrature(order, max_diameter)
        if p.shape[0]:
            nodes.append(p)
            weights.append(w)
    if not nodes:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(nodes), np.concatenate(weights)


def heat_smoothed_density(
    domain_pieces,
    rho: Callable,
    t: float,
    points: np.ndarray,
    order: int = 10,
    grade: float = 0.75,
    cutoff: float = 13.0,
) -> np.ndarray:
    """``(G_t * rho)(x)`` over ``Omega`` by graded quadrature: the residual right-hand side.

    ``domain_pieces`` is a ``Piece``, a ``Leaf`` or an iterable of either, covering
    ``Omega`` exactly once.  ``rho`` maps a physical ``(n, 2)`` array to values.
    The rule is graded so every cell has diameter at most ``grade sqrt(t)``, and
    sources beyond ``cutoff sqrt(t)`` of a target are dropped (``cutoff = 13``
    leaves an absolute truncation below ``1e-18`` of the peak).  Returns physical
    values of shape ``(n,)``; the ``2 pi`` of the chi-unit kernels never enters.
    """
    t = float(t)
    if t <= 0.0:
        raise ValueError("heat time must be positive")
    sigma = math.sqrt(t)
    nodes, weights = domain_quadrature(
        domain_pieces, order=order, max_diameter=grade * sigma
    )
    targets = np.atleast_2d(np.asarray(points, dtype=np.float64))
    out = np.zeros(targets.shape[0], dtype=np.float64)
    if nodes.shape[0] == 0:
        return out
    values = np.asarray(rho(nodes), dtype=np.float64).reshape(-1)
    contrib = weights * values
    radius = float(cutoff) * sigma
    index = _NodeIndex(nodes, radius)
    factor = 1.0 / (4.0 * math.pi * t)
    r2_max = radius * radius
    for n, target in enumerate(targets):
        cand = index.query(target)
        if cand.size == 0:
            continue
        dx = nodes[cand, 0] - target[0]
        dy = nodes[cand, 1] - target[1]
        r2 = dx * dx + dy * dy
        keep = r2 <= r2_max
        if not np.any(keep):
            continue
        out[n] = factor * float(
            np.dot(contrib[cand[keep]], np.exp(-r2[keep] / (4.0 * t)))
        )
    return out


def gaussian_box_heat_reference(
    points: np.ndarray, decay: float, t: float, bounds: tuple
) -> np.ndarray:
    """Closed-form ``(G_t * rho)`` over a rectangle for ``rho(y) = exp(-a |y|^2)``.

    Both factors are Gaussians, so the convolution restricted to the rectangle
    ``bounds = (x_lo, x_hi, y_lo, y_hi)`` separates into two one-dimensional
    integrals with error-function closed forms.  This is the independent
    reference used by the self-test; physical units throughout.
    """
    a = float(decay)
    t = float(t)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    a1, b1, a2, b2 = bounds
    coeff = 1.0 / (4.0 * math.pi * t)
    big_a = 1.0 / (4.0 * t) + a

    def _factor(x, lo, hi):
        """1D integral ``int_lo^hi exp(-(x-y)^2/(4t) - a y^2) dy``."""
        big_b = x / (2.0 * t)
        mu = big_b / (2.0 * big_a)
        pref = np.exp(big_b * big_b / (4.0 * big_a) - x * x / (4.0 * t))
        root = math.sqrt(big_a)
        return (
            pref
            * 0.5
            * math.sqrt(math.pi / big_a)
            * (erf(root * (hi - mu)) - erf(root * (lo - mu)))
        )

    return coeff * _factor(pts[:, 0], a1, b1) * _factor(pts[:, 1], a2, b2)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def _check_quadrature_exactness() -> list[dict]:
    """Check that box and polygon piece rules integrate polynomials of the stated degree."""
    rows = []
    box = Box((0.15, -0.2), 0.35, 2)
    box_piece = make_box_piece(box)
    lshape = np.array(
        [
            [-0.8, -0.8],
            [0.8, -0.8],
            [0.8, 0.0],
            [0.0, 0.0],
            [0.0, 0.8],
            [-0.8, 0.8],
        ]
    )
    poly_piece = make_polygon_piece(lshape)
    for name, piece, verts in (
        ("box", box_piece, box.vertices()),
        ("polygon_nonconvex", poly_piece, lshape),
    ):
        for degree in (0, 3, 6, 9):
            nodes, weights = piece.quadrature(degree)
            worst = 0.0
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    num = float(np.sum(weights * nodes[:, 0] ** i * nodes[:, 1] ** j))
                    ref = polygon_monomial_moment(verts, i, j)
                    scale = max(abs(ref), 1e-12)
                    worst = max(worst, abs(num - ref) / scale)
            rows.append(
                {
                    "check": "quadrature_exactness",
                    "piece": name,
                    "degree": degree,
                    "n_nodes": int(nodes.size // 2),
                    "max_rel_error": worst,
                    "passed": bool(worst < 1e-12),
                }
            )
        area_num = float(np.sum(piece.quadrature(2)[1]))
        rows.append(
            {
                "check": "piece_area",
                "piece": name,
                "degree": 2,
                "n_nodes": 0,
                "max_rel_error": abs(area_num - piece.area()) / abs(piece.area()),
                "passed": bool(
                    abs(area_num - piece.area()) / abs(piece.area()) < 1e-13
                ),
            }
        )
    return rows


def _circle_arc(center, radius, a0, a1) -> Arc:
    """Arc of a circle traversed from angle ``a0`` to ``a1`` (CCW when ``a1 > a0``)."""
    c = np.asarray(center, dtype=np.float64)

    def curve(s):
        """Points on the circle at angles ``s``."""
        s = np.atleast_1d(s)
        return c[None, :] + radius * np.stack([np.cos(s), np.sin(s)], axis=1)

    def dcurve(s):
        """Tangent of the circle at angles ``s``."""
        s = np.atleast_1d(s)
        return radius * np.stack([-np.sin(s), np.cos(s)], axis=1)

    return Arc(curve, dcurve, float(a0), float(a1))


def _check_curved_pieces() -> list[dict]:
    """Check curved-piece areas and moments against closed forms (arc sign convention)."""
    rows = []
    radius = 0.5
    half_disk = make_curved_piece(
        [
            _circle_arc((0.0, 0.0), radius, 0.0, math.pi),
            Segment(np.array([-radius, 0.0]), np.array([radius, 0.0])),
        ]
    )
    err = abs(half_disk.area() - 0.5 * math.pi * radius**2) / (
        0.5 * math.pi * radius**2
    )
    rows.append(
        {
            "check": "curved_area",
            "piece": "half_disk",
            "degree": -1,
            "n_nodes": 0,
            "max_rel_error": err,
            "passed": bool(err < 1e-12),
        }
    )
    quarters = [
        _circle_arc((0.0, 0.0), radius, k * math.pi / 2.0, (k + 1) * math.pi / 2.0)
        for k in range(4)
    ]
    disk = make_curved_piece(quarters)
    err_area = abs(disk.area() - math.pi * radius**2) / (math.pi * radius**2)
    nodes, weights = disk.quadrature(8)
    ref_x2 = math.pi * radius**4 / 4.0
    num_x2 = float(np.sum(weights * nodes[:, 0] ** 2))
    err_x2 = abs(num_x2 - ref_x2) / ref_x2
    rows.append(
        {
            "check": "curved_area",
            "piece": "disk",
            "degree": -1,
            "n_nodes": int(nodes.size // 2),
            "max_rel_error": max(err_area, err_x2),
            "passed": bool(max(err_area, err_x2) < 1e-10),
        }
    )
    return rows


def _check_leaf_poly() -> list[dict]:
    """Check that ``LeafPoly`` reproduces a polynomial and its derivatives to roundoff."""
    rng = np.random.default_rng(7)
    box = Box((0.3, -0.45), 0.125, 3)
    q = 6
    coeff = rng.normal(size=(q, q))

    def func(points):
        """The exact polynomial sampled by the proxy (degree ``q - 1`` per coordinate)."""
        pts = np.atleast_2d(points)
        out = np.zeros(pts.shape[0])
        for i in range(q):
            for j in range(q):
                out += coeff[i, j] * pts[:, 0] ** i * pts[:, 1] ** j
        return out

    def grad_exact(points):
        """Exact gradient of the polynomial."""
        pts = np.atleast_2d(points)
        gx = np.zeros(pts.shape[0])
        gy = np.zeros(pts.shape[0])
        for i in range(q):
            for j in range(q):
                if i >= 1:
                    gx += i * coeff[i, j] * pts[:, 0] ** (i - 1) * pts[:, 1] ** j
                if j >= 1:
                    gy += j * coeff[i, j] * pts[:, 0] ** i * pts[:, 1] ** (j - 1)
        return np.stack([gx, gy], axis=1)

    def lap_exact(points):
        """Exact Laplacian of the polynomial."""
        pts = np.atleast_2d(points)
        out = np.zeros(pts.shape[0])
        for i in range(q):
            for j in range(q):
                if i >= 2:
                    out += (
                        i * (i - 1) * coeff[i, j] * pts[:, 0] ** (i - 2) * pts[:, 1] ** j
                    )
                if j >= 2:
                    out += (
                        j * (j - 1) * coeff[i, j] * pts[:, 0] ** i * pts[:, 1] ** (j - 2)
                    )
        return out

    proxy = LeafPoly.from_callable(box, func, q)
    pts = box.center + box.half * rng.uniform(-1.0, 1.0, size=(64, 2))
    scale = max(float(np.max(np.abs(func(pts)))), 1e-12)
    e_val = float(np.max(np.abs(proxy.eval(pts) - func(pts)))) / scale
    gscale = max(float(np.max(np.abs(grad_exact(pts)))), 1e-12)
    e_grad = float(np.max(np.abs(proxy.grad(pts) - grad_exact(pts)))) / gscale
    lscale = max(float(np.max(np.abs(lap_exact(pts)))), 1e-12)
    e_lap = float(np.max(np.abs(proxy.laplacian(pts) - lap_exact(pts)))) / lscale
    centred = proxy.monomial_coeffs(origin=box.center)
    ref_shift = np.array(coeff, dtype=np.float64)
    ref_shift = _shift_poly_1d(ref_shift, float(box.center[0]), axis=0)
    ref_shift = _shift_poly_1d(ref_shift, float(box.center[1]), axis=1)
    e_shift = float(np.max(np.abs(centred - ref_shift))) / max(
        float(np.max(np.abs(ref_shift))), 1e-12
    )
    # Recovering physical-scale monomial coefficients from node values on a box of
    # half-width h amplifies roundoff by h^{-(i+j)}; the highest coefficient is the
    # worst case.  This asserts the conversion is no worse than that intrinsic bound,
    # which is why the value checks below, and not this one, are the usable contract.
    bound = 64.0 * float(np.finfo(np.float64).eps) / box.half ** (2 * (q - 1))
    e_mono = float(
        np.max(np.abs(eval_monomial(centred, pts, box.center) - func(pts)))
    ) / scale
    e_mono_far = float(
        np.max(np.abs(eval_monomial(proxy.monomial_coeffs(), pts) - func(pts)))
    ) / scale
    rows = [
        {
            "check": "leaf_poly_eval",
            "quantity": "value",
            "max_rel_error": e_val,
            "passed": bool(e_val < 1e-12),
        },
        {
            "check": "leaf_poly_grad",
            "quantity": "gradient",
            "max_rel_error": e_grad,
            "passed": bool(e_grad < 1e-11),
        },
        {
            "check": "leaf_poly_laplacian",
            "quantity": "laplacian",
            "max_rel_error": e_lap,
            "passed": bool(e_lap < 1e-10),
        },
        {
            "check": "leaf_poly_monomial_centred",
            "quantity": "coeffs_about_leaf_centre",
            "max_rel_error": e_shift,
            "conditioning_bound": bound,
            "passed": bool(e_shift <= bound),
        },
        {
            "check": "leaf_poly_monomial_value",
            "quantity": "reconstruction_centred",
            "max_rel_error": e_mono,
            "passed": bool(e_mono < 1e-12),
        },
        {
            "check": "leaf_poly_monomial_value_far_origin",
            "quantity": "reconstruction_origin",
            "max_rel_error": e_mono_far,
            "passed": bool(e_mono_far < 1e-9),
        },
    ]
    return rows


def _check_multiplier() -> list[dict]:
    """Check the ``k = 0`` limit and small-``k`` series of the shell multiplier."""
    tc, tf = t_l(2), t_l(3)
    dt = tc - tf
    zero = float(np.asarray(shell_multiplier(0.0, tc, tf)))
    e_zero = abs(zero - dt) / dt
    k2 = np.array([1e-14, 1e-10, 1e-6, 1e-3])
    num = np.asarray(shell_multiplier(k2, tc, tf))
    series = dt - 0.5 * (tc * tc - tf * tf) * k2 + (tc**3 - tf**3) * k2 * k2 / 6.0
    e_series = float(np.max(np.abs(num - series) / dt))
    grid_k = np.array([0.0, 1.0, 10.0, 100.0]) ** 2
    positive = bool(np.all(np.asarray(shell_multiplier(grid_k, tc, tf)) >= 0.0))
    peak_err = abs(shell_kernel(0.0, tc, tf) - shell_peak(tc, tf)) / shell_peak(tc, tf)
    prof_err = abs(
        float(np.asarray(shell_profile(0.7)))
        - float(np.asarray(shell_kernel(0.7 * 2.0 * math.sqrt(tf), tc, tf)))
    ) / abs(float(np.asarray(shell_profile(0.7))))
    return [
        {
            "check": "multiplier_k0",
            "quantity": "limit",
            "max_rel_error": e_zero,
            "passed": bool(e_zero < 1e-15),
        },
        {
            "check": "multiplier_small_k",
            "quantity": "series",
            "max_rel_error": e_series,
            "passed": bool(e_series < 1e-9),
        },
        {
            "check": "multiplier_sign",
            "quantity": "non_negative",
            "max_rel_error": 0.0 if positive else 1.0,
            "passed": positive,
        },
        {
            "check": "shell_peak",
            "quantity": "D_l(0)",
            "max_rel_error": peak_err,
            "passed": bool(peak_err < 1e-14),
        },
        {
            "check": "shell_profile_rescaling",
            "quantity": "dyadic_profile",
            "max_rel_error": prof_err,
            "passed": bool(prof_err < 1e-14),
        },
    ]


def _check_plane_wave_grid() -> list[dict]:
    """Check that the plane-wave grid meets its own truncation criterion."""
    rows = []
    for level in (1, 2, 3):
        tc, tf = t_l(level), t_l(level + 1)
        grid = plane_wave_grid(h_level(level), tc, tf, tol=1e-12)
        rows.append(
            {
                "check": "plane_wave_grid",
                "level": level,
                "n_f": grid.n_f,
                "n_per_coordinate": grid.n_per_coordinate,
                "dk_h": grid.dk * h_level(level),
                "trunc_rel": grid.trunc_rel,
                "alias_rel": grid.alias_rel,
                "max_rel_error": grid.trunc_rel,
                "passed": bool(0 < grid.n_f < 512 and grid.trunc_rel <= 1e-12),
            }
        )
    return rows


def _check_backbone(levels: int = 3, quick: bool = False) -> list[dict]:
    """Backbone check: per-stage Laplacian right-hand sides on the full-box control case.

    With ``Omega`` the root box and ``rho(y) = exp(-3 |y|^2)``, every stage's
    right-hand side is a Gaussian-smoothed density; each one is compared with the
    closed-form error-function reference, and the three stages
    ``G_{t_0} * rho``, ``sum_l (G_{t_{l+1}} - G_{t_l}) * rho`` and
    ``rho - G_{t_L} * rho`` are summed and compared with ``rho`` itself.
    """
    decay = 3.0
    root = Box((0.0, 0.0), ROOT_HALF, 0)
    piece = make_box_piece(root)
    bounds = root.bounds
    times = [t_l(level) for level in range(levels + 1)]

    def rho(points):
        """The Gaussian test density ``exp(-3 |y|^2)``."""
        pts = np.atleast_2d(points)
        return np.exp(-decay * (pts[:, 0] ** 2 + pts[:, 1] ** 2))

    rng = np.random.default_rng(101)
    n_probe = 12 if quick else 32
    probes = rng.uniform(-0.75, 0.75, size=(n_probe, 2))
    order = 8 if quick else 10
    grade = 1.0 if quick else 0.75

    rows = []
    smoothed = []
    for level, t in enumerate(times):
        num = heat_smoothed_density(
            piece, rho, t, probes, order=order, grade=grade
        )
        ref = gaussian_box_heat_reference(probes, decay, t, bounds)
        err = float(np.max(np.abs(num - ref))) / float(np.max(np.abs(ref)))
        smoothed.append(num)
        rows.append(
            {
                "check": "heat_smoothed_density",
                "level": level,
                "t": t,
                "max_rel_error": err,
                "passed": bool(err < 1e-11),
            }
        )

    stage_w0 = smoothed[0]
    stage_shells = np.zeros_like(stage_w0)
    for level in range(levels):
        stage_shells = stage_shells + (smoothed[level + 1] - smoothed[level])
    stage_closure = rho(probes) - smoothed[levels]
    total = stage_w0 + stage_shells + stage_closure
    exact = rho(probes)
    err_sum = float(np.max(np.abs(total - exact))) / float(np.max(np.abs(exact)))
    rows.append(
        {
            "check": "stage_laplacian_sum",
            "level": levels,
            "t": times[-1],
            "max_rel_error": err_sum,
            "passed": bool(err_sum < 1e-12),
        }
    )

    ref_stack = [
        gaussian_box_heat_reference(probes, decay, t, bounds) for t in times
    ]
    ref_total = (
        ref_stack[0]
        + sum(ref_stack[i + 1] - ref_stack[i] for i in range(levels))
        + (exact - ref_stack[levels])
    )
    err_ref = float(np.max(np.abs(total - ref_total))) / float(np.max(np.abs(exact)))
    rows.append(
        {
            "check": "stage_laplacian_vs_direct_quadrature",
            "level": levels,
            "t": times[-1],
            "max_rel_error": err_ref,
            "passed": bool(err_ref < 1e-11),
        }
    )
    return rows


def _check_tree_and_probes() -> list[dict]:
    """Check quadtree colleague lists and the determinism of the probe generator."""
    tree = build_quadtree(levels=3)
    corner = tree.box_at(3, 0, 0)
    interior = tree.box_at(3, 3, 4)
    ok_counts = len(tree.colleagues(corner)) == 4 and len(tree.colleagues(interior)) == 9
    i, j = tree.indices_of(interior)
    ok_index = (i, j) == (3, 4) and tree.locate(3, interior.center) == (3, 4)
    ok_h = abs(tree.h(3) - h_level(3)) < 1e-15

    def sdf(points):
        """Signed distance to a disk of radius 0.55 (negative inside)."""
        pts = np.atleast_2d(points)
        return np.hypot(pts[:, 0], pts[:, 1]) - 0.55

    sets_a = probe_sets(sdf, h_level(3), t_l(3), corner=(0.0, 0.0), bisector=(1.0, 1.0))
    sets_b = probe_sets(sdf, h_level(3), t_l(3), corner=(0.0, 0.0), bisector=(1.0, 1.0))
    same = all(
        np.array_equal(sets_a[k], sets_b[k]) for k in sets_a if k in sets_b
    ) and set(sets_a) == set(sets_b)
    band_ok = True
    for name, lo, hi in (
        ("leaf_band", 0.1 * h_level(3), h_level(3)),
        ("window_band", 0.05 * math.sqrt(t_l(3)), 2.0 * math.sqrt(t_l(3))),
    ):
        for side, sign in (("inside", -1.0), ("outside", 1.0)):
            pts = sets_a.get(f"{name}_{side}")
            if pts is None or pts.shape[0] == 0:
                band_ok = False
                continue
            d = np.asarray(sdf(pts))
            band_ok = band_ok and bool(
                np.all(sign * d > 0.0)
                and np.all(np.abs(d) >= lo * 0.5)
                and np.all(np.abs(d) <= hi * 2.0)
            )
    return [
        {
            "check": "quadtree_colleagues",
            "quantity": "counts_and_indices",
            "max_rel_error": 0.0 if (ok_counts and ok_index and ok_h) else 1.0,
            "passed": bool(ok_counts and ok_index and ok_h),
        },
        {
            "check": "probe_sets_reproducible",
            "quantity": "determinism",
            "max_rel_error": 0.0 if same else 1.0,
            "passed": bool(same),
        },
        {
            "check": "probe_sets_bands",
            "quantity": "band_membership",
            "max_rel_error": 0.0 if band_ok else 1.0,
            "passed": bool(band_ok),
        },
    ]


def environment_summary() -> dict:
    """Interpreter and library versions of the run, for the artifact JSON."""
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def run_self_test(quick: bool = False) -> tuple[list[dict], bool]:
    """Run every self-test check and return the rows and the overall pass flag."""
    rows: list[dict] = []
    rows.extend(_check_quadrature_exactness())
    rows.extend(_check_curved_pieces())
    rows.extend(_check_leaf_poly())
    rows.extend(_check_multiplier())
    rows.extend(_check_plane_wave_grid())
    rows.extend(_check_tree_and_probes())
    rows.extend(_check_backbone(levels=3, quick=quick))
    passed = all(bool(row.get("passed", False)) for row in rows)
    return rows, passed


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point running the self-test and optionally writing JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="directory for the summary JSON")
    parser.add_argument(
        "--quick", action="store_true", help="shrink the backbone probe set"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows, passed = run_self_test(quick=args.quick)
    width = max(len(str(row.get("check", ""))) for row in rows)
    for row in rows:
        tag = "ok  " if row.get("passed") else "FAIL"
        extra = row.get("piece") or row.get("quantity") or row.get("level")
        print(
            f"[E-common] {tag} {str(row['check']):<{width}} "
            f"{str(extra):<20} err={row.get('max_rel_error', float('nan')):.3e}",
            flush=True,
        )
    print(f"[E-common] overall: {'PASS' if passed else 'FAIL'}", flush=True)

    if args.out:
        out = Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "module": "experiment_e_common",
            "passed": passed,
            "rows": rows,
            "environment": environment_summary(),
            "conventions": {
                "kernel": "K(r) = -log(r)/(2 pi); window kernels carried in chi units",
                "chi_units": "chi_0(r; t) = E_1(r^2/(4t))/2 = 2 pi int_0^t G_u dr",
                "h_level": "leaf side length h_l = 2^{1-l} on the root box [-1,1]^2",
                "t_level": "t_l = (h_l / Theta)^2",
                "multiplier": (
                    "shell_multiplier is the transform of the physical shell "
                    "D_l/(2 pi), value t_l - t_{l+1} at k = 0"
                ),
                "sdf_sign": "signed distance negative inside Omega",
                "k_ordering": "kx varies slowest, ky fastest; k1 ascending",
            },
        }
        (out / "experiment_e_common_selftest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
