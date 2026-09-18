"""Experiment E geometry: domains, leaf clipping, plane waves and polar quadrature.

This module owns everything the two-dimensional windowed-DMK assembly needs to
know about *where* the source lives.  It supplies the three physical domains of
the study, the clipping of a quadtree leaf against them, the plane-wave
coefficients of a per-leaf polynomial proxy over a clipped piece, and the
target-centred polar quadrature that provides reference values and the leaf
closure on cut pieces.  The data structures (``Box``, ``Piece``, ``Leaf``,
``LeafPoly``, ``Segment``, ``Arc``) come from ``experiment_e_common`` and are
not redefined here; the polar machinery is carried over from the leaf-residual
study (``experiment_c_leaf_residual``) and generalized to curved edges.

Domains
-------
* ``"B"`` -- the full root box ``[-1, 1]^2``: the control case, never cut.
* ``"L"`` -- the re-entrant polygon ``[-0.8, 0.8]^2 \\ [0, 0.8]^2`` with its
  ``270`` degree corner at the origin.
* ``"S"`` -- the smooth star ``r(phi) = 0.55 (1 + 0.3 cos 5 phi)``.

Every domain exposes ``inside(points)``, ``signed_distance(points)`` (negative
inside, a true Euclidean distance; found by Newton on the curve for ``S``) and
``clip_leaf(box) -> list[Piece]``.  ``clip_leaf`` returns ``[]`` for an exterior
leaf, ``[box.piece()]`` for an interior leaf, and otherwise the physical
sub-regions of the leaf: axis-aligned rectangles and polygons for ``L``, curved
polygons (box-edge segments plus boundary arcs) for ``S``.

Two clipping conventions are worth stating because a consumer can trip on them.
First, ``L`` is clipped as ``(box ∩ [-0.8, 0.8]^2) \\ [0, 0.8]^2``, which for a
leaf straddling the re-entrant corner is *two* axis-aligned rectangles rather
than one L-shaped hexagon.  The two forms have the same union, but the
rectangles are exactly representable and need no ear clipping, and the corner
case at a leaf corner and the corner case strictly inside a leaf then go down
the same code path.  Second, a clipped rectangle that is not the whole leaf is
returned with ``kind == "polygon"``, never ``"box"``, so that
``Leaf.is_cut()`` keeps its meaning; use ``is_axis_rectangle(piece)`` to detect
the pieces on which a separable (tensor) rule still applies.

Plane-wave coefficients
-----------------------
``plane_wave_coeffs(piece, poly, kx, ky)`` returns ``int_piece e^{-i k . y}
p(y) dy`` for every mode of the grid.  Box pieces use the separable closed form
in the leaf's monomial basis; polygon pieces use the divergence theorem, with
the potential ``q`` solving ``q + i (k . grad q) / |k|^2 = -p`` by a terminating
Neumann series and each edge integral in closed form.  Two removable
singularities are branched on explicitly: ``k . (b - a) = 0`` (``k`` orthogonal
to an edge), where the edge moments ``int_0^1 tau^n e^{-i alpha tau} d tau``
revert to their Taylor series, and small ``|k|`` (including ``k = 0``), where
the divergence-theorem form loses relative accuracy to the ``|k|^{-2}`` of the
Neumann series and the whole integral is taken instead from exact polygon
monomial moments against the Taylor series of the exponential.  Curved pieces
are integrated by the piece's own Gauss rule, refined until each cell is short
against the shortest wavelength on the grid.

Polar quadrature
----------------
``polar_reference(piece, poly, targets, kernel)`` evaluates
``int_piece kernel(|x - y|) p(y) dy`` by the target-centred fan: each directed
edge of the piece contributes a signed sector, the radial integral is taken with
exact ``r^k log r`` moments plus a graded Gauss rule for the smooth remainder,
and the angular variable is ``w = tan(phi - phi_0)`` on a straight edge and the
curve parameter itself on an arc, where ``d phi / ds = cross(y - x, y'(s)) /
|y - x|^2``.  Arc panels are graded dyadically toward the point of the arc
closest to the target, which is what keeps the rule accurate for a target a
fraction of a leaf away from the boundary.  ``prefix_polar(piece, poly, targets,
t)`` is the same evaluation with the leaf-closure kernel
``chi_0(r; t) / (2 pi) = E_1(r^2 / (4 t)) / (4 pi)`` and returns a *physical*
potential (the ``2 pi`` of the chi-unit convention is already applied).
``polar_gradient(piece, poly, targets, kernel)`` is the vector form of the same
fan: it shares the angular substitution, the panel rule and the radial moments
edge by edge and carries the ray direction out of the angular integral, so a
curved piece's gradient is taken on its own arcs rather than on a sampled chord
polygon through them.

A curved ``Piece`` also carries ``inside_exact``, the exact ``Omega ∩ box``
membership predicate of the leaf it was clipped from, attached by
``Domain.leaves``.  It exists because a curved piece's ``vertices`` are only the
chord polygon through the edge endpoints, so no point-in-piece test built from
them can resolve an offset smaller than the chord sagitta.

Exploratory campaign code: no warm-up, no repetition statistics, no claim that
any rule here is optimal.

Self-test
---------
    python experiments/windowed_dmk/experiment_e_geometry.py --out DIR

runs the checks of this module (clipped areas against closed-form domain areas,
plane-wave closed form against quadrature, polar quadrature against the
leaf-residual study's own routine and against the separable rule, and the
leaf-decomposition plots).  ``experiment_e_test_geometry.py`` is the fuller
driver and writes the JSON consumed by the write-up.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:  # pragma: no cover - import shim for both invocation styles
    from experiments.windowed_dmk import experiment_c_leaf_residual as expc
    from experiments.windowed_dmk import experiment_e_common as ec
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import experiment_c_leaf_residual as expc  # type: ignore[no-redef]
    import experiment_e_common as ec  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

STAR_R0 = 0.55
"""Mean radius of the smooth star boundary, physical units."""

STAR_AMP = 0.3
"""Relative lobe amplitude of the smooth star boundary."""

STAR_LOBES = 5
"""Number of lobes of the smooth star boundary."""

L_HALF = 0.8
"""Half-side of the outer square of the L-shaped domain, physical units."""

DOMAIN_NAMES = ("B", "L", "S")
"""Case labels of the study: full box, L-shaped polygon, smooth star."""

DEFAULT_N_ANG = 16
"""Gauss points per angular panel of the polar fan (the leaf-residual default)."""

DEFAULT_N_RAD = 16
"""Gauss points per radial panel of the smooth part of the radial moments."""

CURVED_PW_ORDER = 16
"""Gauss order per direction used for plane-wave coefficients of curved pieces."""

PW_SMALL_KD = 1.0
"""Below this ``|k| * diameter`` the plane-wave closed form uses the moment series."""

PW_MOMENT_TERMS = 26
"""Terms of the exponential Taylor series in the small-``|k|`` moment branch."""

CROSSING_SAMPLES_MIN = 4096
"""Curve samples across a leaf's angular window used to bracket box crossings."""

_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# radial kernels (thin wrappers on the leaf-residual study's split form)
# ---------------------------------------------------------------------------


def full_radial_kernel() -> "expc.RadialKernel":
    """The free-space 2D Laplace kernel ``-log r / (2 pi)`` in split form."""
    return expc.RadialKernel("full", 1.0, None, 1.0)


def prefix_radial_kernel(t: float) -> "expc.RadialKernel":
    """The physical leaf-closure kernel ``chi_0(r; t) / (2 pi) = E_1(r^2/4t)/(4 pi)``."""
    t = float(t)
    return expc.RadialKernel(
        "prefix", 1.0, lambda r: expc.prefix_smooth(r, t), math.sqrt(t)
    )


def smooth_radial_kernel(func: Callable, scale: float, name: str = "smooth"):
    """A non-singular radial kernel ``func(r)`` with grading scale ``scale``."""
    return expc.RadialKernel(name, 0.0, func, float(scale))


# ---------------------------------------------------------------------------
# small geometric helpers
# ---------------------------------------------------------------------------


def is_axis_rectangle(piece, tol: float = 1e-14) -> bool:
    """True when the piece is an axis-aligned rectangle (a separable rule applies)."""
    if piece.kind == "box":
        return True
    if piece.kind != "polygon" or piece.vertices.shape[0] != 4:
        return False
    v = piece.vertices
    for i in range(4):
        a, b = v[i], v[(i + 1) % 4]
        if abs(a[0] - b[0]) > tol and abs(a[1] - b[1]) > tol:
            return False
    return True


def piece_bounds(piece) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a piece as ``(x_lo, x_hi, y_lo, y_hi)``."""
    return piece.bounding_box()


def piece_origin(piece) -> np.ndarray:
    """A well-conditioned local origin for the piece's monomial coefficients."""
    if piece.box is not None:
        return np.asarray(piece.box.center, dtype=np.float64)
    a1, b1, a2, b2 = piece.bounding_box()
    return np.array([0.5 * (a1 + b1), 0.5 * (a2 + b2)], dtype=np.float64)


def piece_diameter(piece) -> float:
    """Diameter of the piece's bounding box, physical units."""
    a1, b1, a2, b2 = piece.bounding_box()
    return float(math.hypot(b1 - a1, b2 - a2))


def loop_area_green(piece, npoints: int = 24) -> float:
    """Area of a piece from ``(1/2) oint (x dy - y dx)``, independent of ``Piece.area``.

    Straight edges are integrated exactly and arcs by composite Gauss on panels
    of at most ``pi / 8`` of tangent turning, so this is an independent check on
    the lune quadrature that ``Piece.area`` uses for curved pieces.
    """
    if piece.kind in ("box", "polygon"):
        return abs(ec.polygon_area(piece.vertices))
    total = 0.0
    for edge in piece.edges or []:
        if getattr(edge, "is_straight", False):
            a = np.asarray(edge.p0, dtype=np.float64)
            b = np.asarray(edge.p1, dtype=np.float64)
            total += 0.5 * (a[0] * b[1] - a[1] * b[0])
            continue
        sample = np.linspace(edge.s0, edge.s1, 65)
        tangents = edge.dcurve(sample)
        angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
        turning = float(np.sum(np.abs(np.diff(angles))))
        panels = max(1, int(math.ceil(turning / (math.pi / 8.0))))
        nodes, weights = ec._composite_gauss(edge.s0, edge.s1, npoints, panels)
        pts = edge.curve(nodes)
        der = edge.dcurve(nodes)
        total += 0.5 * float(
            np.sum(weights * (pts[:, 0] * der[:, 1] - pts[:, 1] * der[:, 0]))
        )
    return abs(total)


# ---------------------------------------------------------------------------
# domains
# ---------------------------------------------------------------------------


class Domain:
    """Base class: a physical region of the root box with a clipping rule."""

    name = "?"

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of the physical points strictly inside ``Omega``."""
        raise NotImplementedError

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Euclidean signed distance to ``partial Omega``, negative inside."""
        raise NotImplementedError

    def clip_leaf(self, box) -> list:
        """Physical pieces of ``Omega ∩ box``; ``[]`` when the leaf is exterior."""
        raise NotImplementedError

    def area(self) -> float:
        """Closed-form area of ``Omega``, physical units."""
        raise NotImplementedError

    def sdf(self) -> Callable:
        """The signed-distance callable, in the form ``probe_sets`` expects."""
        return lambda points: self.signed_distance(points)

    def corner(self):
        """The re-entrant corner and its interior bisector, or ``None``."""
        return None

    def piece_membership(self, box) -> Callable:
        """Exact ``Omega ∩ box`` membership test for the pieces clipped from ``box``.

        A consumer of a ``Piece`` has no other exact predicate available: a
        curved piece's ``vertices`` are only the chord polygon through its edge
        endpoints, so a crossing test against a sampled loop misclassifies every
        point within the chord sagitta of an arc, and on a cut leaf that sagitta
        is larger than the interface probe offsets this study uses.  Each piece
        clipped from one leaf carries this callable as ``inside_exact``.  The
        consumers reduce over the pieces of a leaf with ``or``, so a point lying
        in more than one piece of the same leaf is still counted once.
        """

        def _inside(points) -> np.ndarray:
            """True where a point is in the closed leaf box and inside ``Omega``."""
            pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
            tol = 1e-13 * max(1.0, float(box.half))
            return box.contains(pts, tol=tol) & self.inside(pts)

        return _inside

    def leaves(self, tree, rho: Callable | None = None, order: int | None = None):
        """Clip every leaf of ``tree`` and (optionally) attach a density proxy."""
        out = []
        for box in tree.leaves():
            pieces = self.clip_leaf(box)
            if not pieces:
                continue
            membership = self.piece_membership(box)
            for piece in pieces:
                piece.inside_exact = membership
            poly = None
            if rho is not None and order is not None:
                poly = ec.LeafPoly.from_callable(box, rho, int(order))
            out.append(ec.Leaf(box, pieces, poly))
        return out


class BoxDomain(Domain):
    """Case B: ``Omega`` is the root box itself; no leaf is ever cut."""

    name = "B"

    def __init__(self, half: float = ec.ROOT_HALF):
        """Store the half-width of the square domain, centred at the origin."""
        self.half = float(half)

    def inside(self, points: np.ndarray) -> np.ndarray:
        """True where ``max(|x|, |y|) < half``."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.max(np.abs(pts), axis=1) < self.half

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Exact signed distance to the square boundary, negative inside."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        d = np.abs(pts) - self.half
        outside = np.hypot(np.maximum(d[:, 0], 0.0), np.maximum(d[:, 1], 0.0))
        inside = np.minimum(np.max(d, axis=1), 0.0)
        return outside + inside

    def clip_leaf(self, box) -> list:
        """The whole leaf when it lies in the square, otherwise the clipped rectangle."""
        a1, b1, a2, b2 = box.bounds
        c1, d1 = max(a1, -self.half), min(b1, self.half)
        c2, d2 = max(a2, -self.half), min(b2, self.half)
        if d1 <= c1 or d2 <= c2:
            return []
        if (c1, d1, c2, d2) == (a1, b1, a2, b2):
            return [ec.make_box_piece(box)]
        return [_rectangle_piece((c1, d1, c2, d2), box)]

    def area(self) -> float:
        """``(2 half)^2``."""
        return 4.0 * self.half * self.half


class LDomain(Domain):
    """Case L: ``[-a, a]^2`` minus the quadrant square ``[c1, a] x [c2, a]``."""

    name = "L"

    def __init__(self, half: float = L_HALF, corner=(0.0, 0.0)):
        """Store the outer half-side and the re-entrant corner position."""
        self.half = float(half)
        self.corner_point = np.asarray(corner, dtype=np.float64)
        if not np.all(np.abs(self.corner_point) < self.half):
            raise ValueError("the re-entrant corner must lie inside the outer square")

    def polygon(self) -> np.ndarray:
        """Counter-clockwise vertex loop of the L-shaped polygon."""
        a = self.half
        c1, c2 = float(self.corner_point[0]), float(self.corner_point[1])
        return np.array(
            [
                [-a, -a],
                [a, -a],
                [a, c2],
                [c1, c2],
                [c1, a],
                [-a, a],
            ],
            dtype=np.float64,
        )

    def inside(self, points: np.ndarray) -> np.ndarray:
        """True inside the outer square and outside the removed quadrant."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        in_square = np.max(np.abs(pts), axis=1) < self.half
        in_notch = (pts[:, 0] > self.corner_point[0]) & (
            pts[:, 1] > self.corner_point[1]
        )
        return in_square & ~in_notch

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Exact signed distance to the polygon boundary, negative inside."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        verts = self.polygon()
        n = verts.shape[0]
        best = np.full(pts.shape[0], np.inf)
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            ab = b - a
            denom = float(ab @ ab)
            tau = ((pts - a) @ ab) / denom
            tau = np.clip(tau, 0.0, 1.0)
            foot = a[None, :] + tau[:, None] * ab[None, :]
            best = np.minimum(best, np.hypot(*(pts - foot).T))
        sign = np.where(self.inside(pts), -1.0, 1.0)
        return sign * best

    def clip_leaf(self, box) -> list:
        """``(box ∩ outer square) \\ notch`` as at most two axis-aligned rectangles."""
        a1, b1, a2, b2 = box.bounds
        c1, d1 = max(a1, -self.half), min(b1, self.half)
        c2, d2 = max(a2, -self.half), min(b2, self.half)
        if d1 <= c1 or d2 <= c2:
            return []
        x0, y0 = float(self.corner_point[0]), float(self.corner_point[1])
        rects = []
        left = (c1, min(d1, x0), c2, d2)
        if left[1] > left[0]:
            rects.append(left)
        lower = (max(c1, x0), d1, c2, min(d2, y0))
        if lower[1] > lower[0] and lower[3] > lower[2]:
            rects.append(lower)
        if not rects:
            return []
        if len(rects) == 1 and rects[0] == (a1, b1, a2, b2):
            return [ec.make_box_piece(box)]
        return [_rectangle_piece(r, box) for r in rects]

    def area(self) -> float:
        """Outer square minus the removed quadrant rectangle."""
        a = self.half
        x0, y0 = float(self.corner_point[0]), float(self.corner_point[1])
        return (2.0 * a) * (2.0 * a) - (a - x0) * (a - y0)

    def corner(self):
        """The re-entrant corner and the unit interior bisector at it."""
        return (
            self.corner_point.copy(),
            np.array([-_SQRT2 / 2.0, -_SQRT2 / 2.0]),
        )


class StarDomain(Domain):
    """Case S: the smooth star ``r(phi) = r0 (1 + amp cos(m phi))``."""

    name = "S"

    def __init__(self, r0: float = STAR_R0, amp: float = STAR_AMP,
                 lobes: int = STAR_LOBES):
        """Store the three shape parameters of the polar boundary."""
        self.r0 = float(r0)
        self.amp = float(amp)
        self.lobes = int(lobes)

    # -- boundary parametrization ------------------------------------------

    def radius(self, phi):
        """Boundary radius ``r(phi)``."""
        phi = np.asarray(phi, dtype=np.float64)
        return self.r0 * (1.0 + self.amp * np.cos(self.lobes * phi))

    def dradius(self, phi):
        """First derivative ``r'(phi)``."""
        phi = np.asarray(phi, dtype=np.float64)
        return -self.r0 * self.amp * self.lobes * np.sin(self.lobes * phi)

    def d2radius(self, phi):
        """Second derivative ``r''(phi)``."""
        phi = np.asarray(phi, dtype=np.float64)
        return -self.r0 * self.amp * self.lobes**2 * np.cos(self.lobes * phi)

    def boundary_point(self, phi) -> np.ndarray:
        """Boundary points ``gamma(phi)``, shape ``(n, 2)``, physical units."""
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        r = self.radius(phi)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    def boundary_tangent(self, phi) -> np.ndarray:
        """Tangent ``gamma'(phi)``; the interior lies to its left."""
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        r = self.radius(phi)
        dr = self.dradius(phi)
        c, s = np.cos(phi), np.sin(phi)
        return np.stack([dr * c - r * s, dr * s + r * c], axis=1)

    def boundary_second(self, phi) -> np.ndarray:
        """Second derivative ``gamma''(phi)``."""
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        r = self.radius(phi)
        dr = self.dradius(phi)
        d2r = self.d2radius(phi)
        c, s = np.cos(phi), np.sin(phi)
        return np.stack(
            [d2r * c - 2.0 * dr * s - r * c, d2r * s + 2.0 * dr * c - r * s], axis=1
        )

    def boundary_normal(self, phi) -> np.ndarray:
        """Unit outward normal at ``gamma(phi)``."""
        tan = self.boundary_tangent(phi)
        nrm = np.hypot(tan[:, 0], tan[:, 1])
        return np.stack([tan[:, 1] / nrm, -tan[:, 0] / nrm], axis=1)

    def curvature(self, phi) -> np.ndarray:
        """Signed curvature ``(r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^{3/2}``."""
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        r = self.radius(phi)
        dr = self.dradius(phi)
        d2r = self.d2radius(phi)
        return (r * r + 2.0 * dr * dr - r * d2r) / (r * r + dr * dr) ** 1.5

    def arclength_element(self, phi) -> np.ndarray:
        """``|gamma'(phi)|``, the arclength element of the parametrization."""
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        r = self.radius(phi)
        dr = self.dradius(phi)
        return np.sqrt(r * r + dr * dr)

    # -- point classification ----------------------------------------------

    def inside(self, points: np.ndarray) -> np.ndarray:
        """True where ``|x| < r(atan2(y, x))``; exact for a star-shaped boundary."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        return np.hypot(pts[:, 0], pts[:, 1]) < self.radius(phi)

    def closest_boundary(self, points: np.ndarray, samples: int = 512,
                         iters: int = 60) -> tuple[np.ndarray, np.ndarray]:
        """Parameter and distance of the closest boundary point, by Newton on the curve.

        A uniform sample of ``phi`` brackets the global minimum of
        ``|x - gamma(phi)|^2``; Newton on ``f(phi) = (x - gamma) . gamma'`` then
        converges quadratically, damped so a step never leaves the bracketing
        sample interval.  Returns ``(phi, distance)`` with distance unsigned.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        grid = np.linspace(0.0, 2.0 * math.pi, int(samples), endpoint=False)
        gamma = self.boundary_point(grid)
        step = 2.0 * math.pi / float(samples)
        phi = np.empty(pts.shape[0])
        chunk = max(1, int(4_000_000 // max(1, grid.size)))
        for start in range(0, pts.shape[0], chunk):
            block = pts[start : start + chunk]
            d2 = (block[:, None, 0] - gamma[None, :, 0]) ** 2 + (
                block[:, None, 1] - gamma[None, :, 1]
            ) ** 2
            phi[start : start + chunk] = grid[np.argmin(d2, axis=1)]
        for _ in range(int(iters)):
            g = self.boundary_point(phi)
            dg = self.boundary_tangent(phi)
            d2g = self.boundary_second(phi)
            rel = pts - g
            f = np.sum(rel * dg, axis=1)
            fp = -np.sum(dg * dg, axis=1) + np.sum(rel * d2g, axis=1)
            fp = np.where(np.abs(fp) < 1e-30, -1.0, fp)
            delta = np.clip(-f / fp, -step, step)
            phi = phi + delta
            if np.max(np.abs(delta)) < 1e-16:
                break
        dist = np.hypot(*(pts - self.boundary_point(phi)).T)
        return phi, dist

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Euclidean signed distance to the star boundary, negative inside."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        _, dist = self.closest_boundary(pts)
        return np.where(self.inside(pts), -dist, dist)

    def area(self) -> float:
        """``pi r0^2 (1 + amp^2 / 2)`` from ``(1/2) int r^2 d phi``."""
        return math.pi * self.r0**2 * (1.0 + 0.5 * self.amp**2)

    def max_radius(self) -> float:
        """Largest boundary radius, physical units."""
        return self.r0 * (1.0 + self.amp)

    # -- clipping -----------------------------------------------------------

    def clip_leaf(self, box, samples: int | None = None) -> list:
        """Clip a leaf against the star: box piece, nothing, or curved pieces.

        The cheap radial bracket (the leaf's smallest and largest distance from
        the origin against the smallest and largest boundary radius) settles the
        far interior and far exterior leaves without touching the curve; only the
        remaining leaves pay for the crossing search.
        """
        center = np.asarray(box.center, dtype=np.float64)
        r_lo, r_hi = _box_radial_range(box)
        if r_hi <= self.r0 * (1.0 - self.amp):
            return [ec.make_box_piece(box)]
        if r_lo >= self.max_radius():
            return []
        crossings = self._box_crossings(box, samples)
        if not crossings:
            probe = self.boundary_point(np.linspace(0.0, 2.0 * math.pi, 257))
            if np.all(box.contains(probe, tol=0.0)):
                return [self._closed_piece(box)]
            inside = bool(self.inside(center[None, :])[0])
            return [ec.make_box_piece(box)] if inside else []
        return self._assemble_pieces(box, crossings)

    def _closed_piece(self, box):
        """The whole star as one closed curved piece (the leaf contains ``Omega``)."""
        arc = ec.Arc(self.boundary_point, self.boundary_tangent, 0.0, 2.0 * math.pi)
        piece = ec.Piece("curved", self.boundary_point(np.array([0.0])), [arc], box)
        return piece

    def _box_crossings(self, box, samples: int | None = None) -> list[dict]:
        """Locate every crossing of the star boundary with the leaf's edges.

        The scalar ``f(phi) = max(|x - cx|, |y - cy|) - half`` is negative exactly
        inside the leaf, so its sign changes bracket the crossings; each bracket
        is bisected to machine precision and the resulting point is snapped onto
        the active box edge so that neighbouring leaves share it exactly.

        The scan runs over the angular window the leaf subtends at the origin
        rather than over the whole period, because the domain is star-shaped
        about the origin and the curve can meet the leaf only there.  For a leaf
        of side ``h`` at distance ``d >= h`` that resolves arcs down to roughly
        ``h / 1000`` of chord length, which is what a grazing incursion through
        one edge needs: at a uniform scan of the full period, a genuine
        ``2.4e-7`` sliver at level 5 was missed and showed up as a defect in the
        summed area.  Arcs far shorter than that can still be missed; the
        clipped-area sums are the check that reports it.
        """
        if samples is None:
            samples = CROSSING_SAMPLES_MIN
        center = np.asarray(box.center, dtype=np.float64)
        half = box.half
        lo_phi, hi_phi = _box_angular_window(box)
        phi = np.linspace(lo_phi, hi_phi, int(samples))

        def f(values):
            """Signed box-membership functional of the curve at parameters ``values``."""
            pts = self.boundary_point(values)
            return np.maximum(
                np.abs(pts[:, 0] - center[0]), np.abs(pts[:, 1] - center[1])
            ) - half

        vals = f(phi)
        mask = (vals[:-1] < 0.0) != (vals[1:] < 0.0)
        lo = phi[:-1][mask].copy()
        hi = phi[1:][mask].copy()
        if lo.size == 0:
            return []
        f_lo = f(lo)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            same = (f_mid < 0.0) == (f_lo < 0.0)
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
            f_lo = np.where(same, f_mid, f_lo)
        root = 0.5 * (lo + hi)
        pts = self.boundary_point(root)
        tan = self.boundary_tangent(root)
        a1, b1, a2, b2 = box.bounds
        out = []
        for i in range(root.size):
            p = pts[i].copy()
            dx = abs(p[0] - center[0]) - half
            dy = abs(p[1] - center[1]) - half
            if dx >= dy:
                edge = 1 if p[0] > center[0] else 3
                p[0] = b1 if edge == 1 else a1
                normal = np.array([1.0, 0.0]) if edge == 1 else np.array([-1.0, 0.0])
            else:
                edge = 2 if p[1] > center[1] else 0
                p[1] = b2 if edge == 2 else a2
                normal = np.array([0.0, 1.0]) if edge == 2 else np.array([0.0, -1.0])
            leaving = float(tan[i] @ normal) > 0.0
            out.append(
                {
                    "phi": float(root[i] % (2.0 * math.pi)),
                    "point": p,
                    "t": _box_param(box, p, edge),
                    "enters_omega": leaving,
                }
            )
        out.sort(key=lambda item: item["t"])
        return out

    def _assemble_pieces(self, box, crossings: list[dict]) -> list:
        """Walk the box boundary and the curve alternately into counter-clockwise loops."""
        if len(crossings) % 2 != 0:
            raise RuntimeError(
                f"odd number of boundary crossings ({len(crossings)}) on leaf {box}"
            )
        by_t = sorted(range(len(crossings)), key=lambda i: crossings[i]["t"])
        by_phi = sorted(range(len(crossings)), key=lambda i: crossings[i]["phi"])
        pos_t = {idx: n for n, idx in enumerate(by_t)}
        pos_phi = {idx: n for n, idx in enumerate(by_phi)}
        used = set()
        pieces = []
        for start in by_t:
            if start in used or not crossings[start]["enters_omega"]:
                continue
            edges = []
            cur = start
            guard = 0
            while True:
                guard += 1
                if guard > 4 * len(crossings) + 8:
                    raise RuntimeError("leaf clipping failed to close a loop")
                used.add(cur)
                nxt = by_t[(pos_t[cur] + 1) % len(crossings)]
                if crossings[nxt]["enters_omega"]:
                    raise RuntimeError("two consecutive box-entry crossings")
                path = [crossings[cur]["point"]]
                path.extend(
                    _box_walk(box, crossings[cur]["t"], crossings[nxt]["t"])
                )
                path.append(crossings[nxt]["point"])
                for a, b in zip(path[:-1], path[1:]):
                    if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-15:
                        edges.append(ec.Segment(np.asarray(a), np.asarray(b)))
                used.add(nxt)
                end = by_phi[(pos_phi[nxt] + 1) % len(crossings)]
                s0 = crossings[nxt]["phi"]
                s1 = crossings[end]["phi"]
                if s1 <= s0:
                    s1 += 2.0 * math.pi
                edges.append(
                    ec.Arc(self.boundary_point, self.boundary_tangent, s0, s1)
                )
                cur = end
                if cur == start:
                    break
            pieces.append(ec.make_curved_piece(edges, box))
        return pieces

    def corner(self):
        """The star has no corner."""
        return None


def make_domain(name: str, **kwargs) -> Domain:
    """Build one of the study's domains by its case label ``B``, ``L`` or ``S``."""
    key = str(name).upper()
    if key == "B":
        return BoxDomain(**kwargs)
    if key == "L":
        return LDomain(**kwargs)
    if key == "S":
        return StarDomain(**kwargs)
    raise ValueError(f"unknown domain {name!r}; expected one of {DOMAIN_NAMES}")


def _box_angular_window(box, pad: float = 1e-3) -> tuple[float, float]:
    """Angular interval the box subtends at the origin, padded; full period if it contains it.

    A convex set not containing the origin lies in a half-plane through it, so
    the four corner angles unwrapped about the centre's angle span less than
    ``pi`` and the window is unambiguous.
    """
    a1, b1, a2, b2 = box.bounds
    if a1 <= 0.0 <= b1 and a2 <= 0.0 <= b2:
        return 0.0, 2.0 * math.pi
    verts = box.vertices()
    ang = np.arctan2(verts[:, 1], verts[:, 0])
    mid = math.atan2(box.center[1], box.center[0])
    rel = (ang - mid + math.pi) % (2.0 * math.pi) - math.pi
    return mid + float(rel.min()) - pad, mid + float(rel.max()) + pad


def _box_radial_range(box) -> tuple[float, float]:
    """Smallest and largest distance from the origin to points of the box."""
    a1, b1, a2, b2 = box.bounds
    dx = max(a1, 0.0, -b1)
    dy = max(a2, 0.0, -b2)
    r_lo = math.hypot(dx, dy)
    r_hi = math.hypot(max(abs(a1), abs(b1)), max(abs(a2), abs(b2)))
    return r_lo, r_hi


def _rectangle_piece(rect, box):
    """A counter-clockwise polygon piece from ``(x_lo, x_hi, y_lo, y_hi)``."""
    a1, b1, a2, b2 = rect
    verts = np.array([[a1, a2], [b1, a2], [b1, b2], [a1, b2]], dtype=np.float64)
    return ec.make_polygon_piece(verts, box)


def _box_param(box, point, edge: int) -> float:
    """Counter-clockwise box-boundary parameter in ``[0, 4)`` of a point on ``edge``."""
    a1, b1, a2, b2 = box.bounds
    if edge == 0:
        return (point[0] - a1) / (b1 - a1)
    if edge == 1:
        return 1.0 + (point[1] - a2) / (b2 - a2)
    if edge == 2:
        return 2.0 + (b1 - point[0]) / (b1 - a1)
    return 3.0 + (b2 - point[1]) / (b2 - a2)


def _box_corner(box, index: int) -> np.ndarray:
    """Corner ``index`` of the box in counter-clockwise order from the lower left."""
    a1, b1, a2, b2 = box.bounds
    return np.array(
        [[a1, a2], [b1, a2], [b1, b2], [a1, b2]], dtype=np.float64
    )[index % 4]


def _box_walk(box, t_from: float, t_to: float) -> list:
    """Box corners strictly between two boundary parameters, walking counter-clockwise."""
    span = t_to - t_from
    if span <= 0.0:
        span += 4.0
    out = []
    k = math.floor(t_from) + 1.0
    while k < t_from + span:
        out.append(_box_corner(box, int(k)))
        k += 1.0
    return out


# ---------------------------------------------------------------------------
# density proxies on pieces
# ---------------------------------------------------------------------------


def _dens_and_origin(piece, poly, origin=None) -> tuple[np.ndarray, np.ndarray]:
    """Monomial coefficients of the proxy and the local origin they refer to."""
    o = np.asarray(piece_origin(piece) if origin is None else origin, dtype=np.float64)
    if poly is None:
        return np.ones((1, 1)), o
    if isinstance(poly, ec.LeafPoly):
        return poly.monomial_coeffs(origin=tuple(o)), o
    return np.asarray(poly, dtype=np.float64), o


def eval_proxy(poly, points: np.ndarray, origin=(0.0, 0.0)) -> np.ndarray:
    """Values of a ``LeafPoly`` or of a monomial coefficient array at physical points."""
    if poly is None:
        return np.ones(np.atleast_2d(points).shape[0])
    if isinstance(poly, ec.LeafPoly):
        return poly.eval(points)
    return ec.eval_monomial(np.asarray(poly, dtype=np.float64), points, origin=origin)


# ---------------------------------------------------------------------------
# plane-wave coefficients
# ---------------------------------------------------------------------------


def _unit_exp_moments(alpha: np.ndarray, n_max: int) -> np.ndarray:
    """``int_0^1 tau^n e^{-i alpha tau} d tau`` for ``n = 0 .. n_max``, shape ``(m, n+1)``.

    The upward recursion divides by ``alpha`` and is used only for ``|alpha| > 1``;
    below that (which includes ``k`` orthogonal to the edge, ``alpha = 0``) the
    Taylor series in ``alpha`` is summed instead, which is where the removable
    singularity of the divergence-theorem formula is disposed of.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    w = -1j * alpha
    out = np.zeros((alpha.size, int(n_max) + 1), dtype=np.complex128)
    narr = np.arange(int(n_max) + 1, dtype=np.float64)
    big = np.abs(alpha) > 1.0
    if np.any(~big):
        ws = w[~big]
        acc = np.zeros((ws.size, int(n_max) + 1), dtype=np.complex128)
        power = np.ones(ws.size, dtype=np.complex128)
        fact = 1.0
        for j in range(0, 34):
            if j:
                power = power * ws
                fact *= j
            acc += (power / fact)[:, None] / (narr + j + 1.0)[None, :]
        out[~big] = acc
    if np.any(big):
        wb = w[big]
        ew = np.exp(wb)
        cur = (ew - 1.0) / wb
        out[big, 0] = cur
        for n in range(1, int(n_max) + 1):
            cur = (ew - n * cur) / wb
            out[big, n] = cur
    return out


def _axis_moments(k: np.ndarray, d: float, i_max: int) -> np.ndarray:
    """``int_{-d}^{d} u^i e^{-i k u} du`` for ``i = 0 .. i_max``, shape ``(i_max+1, m)``."""
    k = np.asarray(k, dtype=np.float64)
    d = float(d)
    w = -1j * k
    out = np.zeros((int(i_max) + 1, k.size), dtype=np.complex128)
    big = np.abs(k) * d > 1.0
    if np.any(~big):
        ws = w[~big]
        power = np.ones(ws.size, dtype=np.complex128)
        fact = 1.0
        acc = np.zeros((int(i_max) + 1, ws.size), dtype=np.complex128)
        for n in range(0, 40):
            if n:
                power = power * ws
                fact *= n
            for i in range(int(i_max) + 1):
                if (i + n) % 2:
                    continue
                acc[i] += (power / fact) * (2.0 * d ** (i + n + 1) / (i + n + 1))
        out[:, ~big] = acc
    if np.any(big):
        wb = w[big]
        ep = np.exp(wb * d)
        em = np.exp(-wb * d)
        cur = (ep - em) / wb
        out[0, big] = cur
        for i in range(1, int(i_max) + 1):
            cur = (d**i * ep - (-d) ** i * em) / wb - (i / wb) * cur
            out[i, big] = cur
    return out


def _tau_expansion(a: np.ndarray, tvec: np.ndarray, i_max: int, j_max: int) -> np.ndarray:
    """Coefficients ``B[i, j, n]`` of ``tau^n`` in ``(a0 + tau t0)^i (a1 + tau t1)^j``."""
    px = np.zeros((i_max + 1, i_max + 1))
    for i in range(i_max + 1):
        for p in range(i + 1):
            px[i, p] = math.comb(i, p) * a[0] ** (i - p) * tvec[0] ** p
    py = np.zeros((j_max + 1, j_max + 1))
    for j in range(j_max + 1):
        for p in range(j + 1):
            py[j, p] = math.comb(j, p) * a[1] ** (j - p) * tvec[1] ** p
    n_max = i_max + j_max
    out = np.zeros((i_max + 1, j_max + 1, n_max + 1))
    for i in range(i_max + 1):
        for j in range(j_max + 1):
            for p in range(i + 1):
                for r in range(j + 1):
                    out[i, j, p + r] += px[i, p] * py[j, r]
    return out


def _apply_advection(term: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                     ksq: np.ndarray) -> np.ndarray:
    """Apply ``A = i (k . grad) / |k|^2`` to per-mode monomial coefficient arrays."""
    _, i_max, j_max = term.shape
    dx = np.zeros_like(term)
    dy = np.zeros_like(term)
    if i_max > 1:
        dx[:, :-1, :] = term[:, 1:, :] * np.arange(1, i_max, dtype=np.float64)[
            None, :, None
        ]
    if j_max > 1:
        dy[:, :, :-1] = term[:, :, 1:] * np.arange(1, j_max, dtype=np.float64)[
            None, None, :
        ]
    scale = 1j / ksq
    return scale[:, None, None] * (
        kx[:, None, None] * dx + ky[:, None, None] * dy
    )


def _pw_divergence(verts: np.ndarray, coeffs: np.ndarray, kx: np.ndarray,
                   ky: np.ndarray) -> np.ndarray:
    """Plane-wave coefficient of a polygon by the divergence theorem (vertices local)."""
    i_max, j_max = coeffs.shape
    ksq = kx * kx + ky * ky
    q = np.zeros((kx.size, i_max, j_max), dtype=np.complex128)
    term = np.broadcast_to(coeffs.astype(np.complex128), q.shape).copy()
    sign = 1.0
    for _ in range(i_max + j_max):
        q -= sign * term
        if not np.any(term):
            break
        term = _apply_advection(term, kx, ky, ksq)
        sign = -sign
    n_max = i_max + j_max - 2
    total = np.zeros(kx.size, dtype=np.complex128)
    n = verts.shape[0]
    for e in range(n):
        a = verts[e]
        b = verts[(e + 1) % n]
        tvec = b - a
        if math.hypot(tvec[0], tvec[1]) < 1e-15:
            continue
        cross = kx * tvec[1] - ky * tvec[0]
        alpha = kx * tvec[0] + ky * tvec[1]
        basis = _tau_expansion(a, tvec, i_max - 1, j_max - 1)
        g = np.einsum("mij,ijn->mn", q, basis)
        moments = _unit_exp_moments(alpha, n_max)
        phase = np.exp(-1j * (kx * a[0] + ky * a[1]))
        total += cross * phase * np.sum(g * moments, axis=1)
    return -1j * total / ksq


def _pw_moment_series(verts: np.ndarray, coeffs: np.ndarray, kx: np.ndarray,
                      ky: np.ndarray, terms: int = PW_MOMENT_TERMS) -> np.ndarray:
    """Plane-wave coefficient of a polygon from exact monomial moments (small ``|k|``).

    ``e^{-i k . z}`` is expanded about the piece's own origin and integrated term
    by term against exact polygon monomial moments, which stays accurate at and
    near ``k = 0`` where the divergence-theorem form has a removable ``|k|^{-2}``.
    """
    i_max, j_max = coeffs.shape
    n_max = int(terms)
    mom = np.zeros((i_max + n_max, j_max + n_max))
    for p in range(i_max + n_max):
        for r in range(j_max + n_max):
            if p + r > i_max + j_max + n_max:
                continue
            mom[p, r] = ec.polygon_monomial_moment(verts, p, r)
    shifted = np.zeros((n_max + 1, n_max + 1))
    for j1 in range(n_max + 1):
        for j2 in range(n_max + 1 - j1):
            shifted[j1, j2] = float(
                np.sum(coeffs * mom[j1 : j1 + i_max, j2 : j2 + j_max])
            )
    total = np.zeros(kx.size, dtype=np.complex128)
    for n in range(n_max + 1):
        acc = np.zeros(kx.size, dtype=np.float64)
        for j in range(n + 1):
            acc += math.comb(n, j) * kx**j * ky ** (n - j) * shifted[j, n - j]
        total += ((-1j) ** n / math.factorial(n)) * acc
    return total


def _pw_box(piece, coeffs: np.ndarray, origin: np.ndarray, kx: np.ndarray,
            ky: np.ndarray) -> np.ndarray:
    """Plane-wave coefficient of an axis-aligned rectangle in separable closed form."""
    a1, b1, a2, b2 = piece.bounding_box()
    cx, cy = 0.5 * (a1 + b1), 0.5 * (a2 + b2)
    d1, d2 = 0.5 * (b1 - a1), 0.5 * (b2 - a2)
    shift = np.array([cx - origin[0], cy - origin[1]])
    local = ec._shift_poly_1d(coeffs, shift[0], axis=0)
    local = ec._shift_poly_1d(local, shift[1], axis=1)
    i_max, j_max = local.shape
    mom_x = _axis_moments(kx, d1, i_max - 1)
    mom_y = _axis_moments(ky, d2, j_max - 1)
    body = np.einsum("ij,im,jm->m", local.astype(np.complex128), mom_x, mom_y)
    return np.exp(-1j * (kx * cx + ky * cy)) * body


def curved_plane_wave_max_diameter(piece, k_max: float, order: int = CURVED_PW_ORDER,
                                   refine: float = 1.0) -> float | None:
    """Cell size on a curved piece that resolves wavenumbers up to ``k_max``.

    ``Piece.quadrature`` puts ``(order + 3) // 2`` Gauss points per direction on
    each triangle, so a cell carrying ``k d = order / 3`` radians of phase gets
    roughly three Gauss points per radian.  That constant was chosen by measuring
    the rule against itself on cut star leaves: at ``k d = 2 order / 3`` the
    coefficients were only accurate to ``4e-12`` relative, and at ``order / 3``
    they agree with a rule two and a half times finer to ``1e-15``.
    """
    diam = piece_diameter(piece)
    if k_max <= 0.0:
        return None
    return min(diam, int(order) / (3.0 * float(k_max) * float(refine)))


def curved_plane_wave_quadrature(piece, k_max: float, order: int = CURVED_PW_ORDER,
                                 refine: float = 1.0):
    """Nodes and weights on a curved piece resolving wavenumbers up to ``k_max``."""
    return piece.quadrature(
        int(order), curved_plane_wave_max_diameter(piece, k_max, order, refine)
    )


def plane_wave_coeffs(piece, poly, kx, ky, origin=None, order: int = CURVED_PW_ORDER,
                      quadrature=None, chunk: int = 512,
                      method: str = "auto") -> np.ndarray:
    """``int_piece e^{-i k . y} p(y) dy`` for every mode, shape ``kx.shape``.

    Box and polygon pieces are evaluated in closed form (separable for a box, the
    divergence theorem for a polygon, with the small-``|k|`` and orthogonal-edge
    branches described in the module docstring); curved pieces are evaluated by
    the piece's own Gauss rule, refined to the shortest wavelength on the grid.
    ``poly`` is a ``LeafPoly``, a monomial coefficient array about ``origin``, or
    ``None`` for the constant density one.  ``method`` forces a route
    (``"box"``, ``"polygon"``, ``"quadrature"``) instead of the automatic
    choice, which is what the closed-form-versus-closed-form test uses.
    """
    kx_in = np.asarray(kx, dtype=np.float64)
    ky_in = np.asarray(ky, dtype=np.float64)
    shape = np.broadcast(kx_in, ky_in).shape
    kxf = np.broadcast_to(kx_in, shape).reshape(-1)
    kyf = np.broadcast_to(ky_in, shape).reshape(-1)
    coeffs, o = _dens_and_origin(piece, poly, origin)
    route = str(method)
    if route == "auto":
        if piece.kind == "box" or is_axis_rectangle(piece):
            route = "box"
        elif piece.kind == "polygon":
            route = "polygon"
        else:
            route = "quadrature"
    if route == "box":
        out = _pw_box(piece, coeffs, o, kxf, kyf)
        return out.reshape(shape)
    if route == "polygon":
        verts = piece.vertices - o[None, :]
        diam = piece_diameter(piece)
        kmag = np.hypot(kxf, kyf)
        small = kmag * diam <= PW_SMALL_KD
        out = np.zeros(kxf.size, dtype=np.complex128)
        if np.any(small):
            out[small] = _pw_moment_series(verts, coeffs, kxf[small], kyf[small])
        idx = np.nonzero(~small)[0]
        for start in range(0, idx.size, 8192):
            block = idx[start : start + 8192]
            out[block] = _pw_divergence(verts, coeffs, kxf[block], kyf[block])
        out *= np.exp(-1j * (kxf * o[0] + kyf * o[1]))
        return out.reshape(shape)
    if quadrature is None:
        k_max = float(np.max(np.hypot(kxf, kyf))) if kxf.size else 0.0
        quadrature = curved_plane_wave_quadrature(piece, k_max, order)
    nodes, weights = quadrature
    if nodes.shape[0] == 0:
        return np.zeros(shape, dtype=np.complex128)
    values = weights * eval_proxy(poly, nodes, origin=tuple(o))
    out = np.zeros(kxf.size, dtype=np.complex128)
    for start in range(0, kxf.size, int(chunk)):
        stop = min(start + int(chunk), kxf.size)
        phase = np.exp(
            -1j
            * (
                np.outer(kxf[start:stop], nodes[:, 0])
                + np.outer(kyf[start:stop], nodes[:, 1])
            )
        )
        out[start:stop] = phase @ values
    return out.reshape(shape)


def plane_wave_quadrature_reference(piece, poly, kx, ky, order: int = 20,
                                    refine: float = 2.0, chunk: int = 32):
    """Plane-wave coefficients by brute-force graded quadrature, for the closed-form test."""
    kx_in = np.asarray(kx, dtype=np.float64)
    ky_in = np.asarray(ky, dtype=np.float64)
    shape = np.broadcast(kx_in, ky_in).shape
    kxf = np.broadcast_to(kx_in, shape).reshape(-1)
    kyf = np.broadcast_to(ky_in, shape).reshape(-1)
    k_max = float(np.max(np.hypot(kxf, kyf))) if kxf.size else 0.0
    max_diameter = curved_plane_wave_max_diameter(piece, k_max, order, refine)
    nodes, weights = piece.quadrature(int(order), max_diameter)
    o = piece_origin(piece)
    values = weights * eval_proxy(poly, nodes, origin=tuple(o))
    out = np.zeros(kxf.size, dtype=np.complex128)
    for start in range(0, kxf.size, int(chunk)):
        stop = min(start + int(chunk), kxf.size)
        phase = np.exp(
            -1j
            * (
                np.outer(kxf[start:stop], nodes[:, 0])
                + np.outer(kyf[start:stop], nodes[:, 1])
            )
        )
        out[start:stop] = phase @ values
    return out.reshape(shape), int(nodes.shape[0])


# ---------------------------------------------------------------------------
# target-centred polar quadrature (carried over from the leaf-residual study)
# ---------------------------------------------------------------------------


def _fan_segment_parts(a: np.ndarray, b: np.ndarray, target: np.ndarray,
                       dens: np.ndarray, kernel, n_ang: int, n_rad: int):
    """Per-node contributions of one straight edge's sector, or ``None`` if empty.

    Returns ``(contrib, dirs)``: ``contrib[j]`` is the signed angular weight times
    the radial moment sum along the ray ``dirs[j]``, so the scalar sector is
    ``contrib.sum()`` and the vector sector (the gradient fan) is
    ``contrib @ dirs``.  Both callers therefore share one angular substitution,
    one panel rule and one radial moment evaluation.

    This is the edge body of ``experiment_c_leaf_residual.polar_polygon_potential``:
    the angular variable is ``w = tan(phi - phi_0)`` about the foot of the
    perpendicular, so the integrand is analytic in ``w`` even for a target on the
    edge's line, and the radial integral is the exact ``r^k log r`` moment plus a
    graded Gauss rule for the smooth part.
    """
    edge = b - a
    length = float(math.hypot(edge[0], edge[1]))
    if length < 1e-14:
        return None
    tangent = edge / length
    foot = a + float((target - a) @ tangent) * tangent
    offset = foot - target
    dist = float(math.hypot(offset[0], offset[1]))
    if dist < 1e-13:
        return None
    nhat = offset / dist
    that = np.array([-nhat[1], nhat[0]])
    w_a = float((a - target) @ that) / dist
    w_b = float((b - target) @ that) / dist
    if abs(w_b - w_a) < 1e-15:
        return None
    sign = 1.0 if w_b > w_a else -1.0
    lo, hi = min(w_a, w_b), max(w_a, w_b)
    nodes, weights = expc._panel_rule(expc._dyadic_breakpoints(lo, hi), n_ang)
    one_plus = 1.0 + nodes * nodes
    root = np.sqrt(one_plus)
    dirs = (nhat[None, :] + nodes[:, None] * that[None, :]) / root[:, None]
    radius = dist * root
    k_max = dens.shape[0] + dens.shape[1] - 2
    coeffs = expc.ray_coefficients(dens, target, dirs)
    moments = expc.radial_moments(kernel, radius, k_max, n_rad)
    inner = np.sum(coeffs * moments, axis=1) / one_plus
    return sign * weights * inner, dirs


def _fan_segment(a: np.ndarray, b: np.ndarray, target: np.ndarray, dens: np.ndarray,
                 kernel, n_ang: int, n_rad: int) -> float:
    """Signed scalar sector of one straight edge, proxy origin at zero."""
    parts = _fan_segment_parts(a, b, target, dens, kernel, n_ang, n_rad)
    if parts is None:
        return 0.0
    return float(np.sum(parts[0]))


def _fan_segment_vector(a: np.ndarray, b: np.ndarray, target: np.ndarray,
                        dens: np.ndarray, kernel, n_ang: int,
                        n_rad: int) -> np.ndarray:
    """Signed vector sector of one straight edge: the same rule times the ray direction."""
    parts = _fan_segment_parts(a, b, target, dens, kernel, n_ang, n_rad)
    if parts is None:
        return np.zeros(2)
    contrib, dirs = parts
    return contrib @ dirs


def _arc_nearest(edge, target: np.ndarray, origin: np.ndarray,
                 samples: int = 129) -> tuple[float, float]:
    """Parameter and distance of the arc point closest to a target (local frame)."""
    grid = np.linspace(edge.s0, edge.s1, int(samples))
    pts = edge.curve(grid) - origin[None, :]
    d2 = (pts[:, 0] - target[0]) ** 2 + (pts[:, 1] - target[1]) ** 2
    idx = int(np.argmin(d2))
    lo = grid[max(idx - 1, 0)]
    hi = grid[min(idx + 1, grid.size - 1)]

    def deriv(s):
        """``d/ds |gamma(s) - x|^2 / 2`` at a scalar parameter."""
        sv = np.array([s])
        rel = edge.curve(sv)[0] - origin - target
        return float(rel @ edge.dcurve(sv)[0])

    f_lo, f_hi = deriv(lo), deriv(hi)
    best = grid[idx]
    if f_lo * f_hi < 0.0:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = deriv(mid)
            if (f_mid < 0.0) == (f_lo < 0.0):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        best = 0.5 * (lo + hi)
    point = edge.curve(np.array([best]))[0] - origin
    return float(best), float(math.hypot(point[0] - target[0], point[1] - target[1]))


def _arc_breakpoints(edge, s_star: float, radius: float, levels: int = 60) -> np.ndarray:
    """Panel breakpoints graded dyadically toward the arc's closest point."""
    marks = {edge.s0, edge.s1, s_star}
    speed = float(np.hypot(*edge.dcurve(np.array([s_star]))[0]))
    floor = max(radius / max(speed, 1e-30), 1e-15)
    for side, span in ((-1.0, s_star - edge.s0), (1.0, edge.s1 - s_star)):
        if span <= 0.0:
            continue
        step = span
        for _ in range(int(levels)):
            step *= 0.5
            if step < floor:
                break
            marks.add(s_star + side * step)
    return np.array(sorted(marks), dtype=np.float64)


def _fan_arc_parts(edge, target: np.ndarray, origin: np.ndarray, dens: np.ndarray,
                   kernel, n_ang: int, n_rad: int):
    """Per-node contributions of one arc's sector, or ``None`` if it is empty.

    Returns ``(contrib, dirs)`` exactly as ``_fan_segment_parts`` does, so the
    scalar sector is ``contrib.sum()`` and the vector sector is ``contrib @
    dirs``.  In polar coordinates about the target the sector is
    ``int_s [int_0^{R(s)} kernel(r) rho r dr] (d theta / ds) ds`` with
    ``d theta / ds = cross(gamma - x, gamma') / |gamma - x|^2``; the radial
    bracket is the same exact-moment sum as on a straight edge.  Panels are
    graded toward the closest point of the arc, which is the only place where the
    angular integrand has a short scale.
    """
    s_star, radius = _arc_nearest(edge, target, origin)
    breaks = _arc_breakpoints(edge, s_star, radius)
    nodes, weights = expc._panel_rule(breaks, n_ang)
    pts = edge.curve(nodes) - origin[None, :]
    der = edge.dcurve(nodes)
    rel = pts - target[None, :]
    r2 = rel[:, 0] ** 2 + rel[:, 1] ** 2
    good = r2 > 1e-30
    if not np.any(good):
        return None
    rel = rel[good]
    der = der[good]
    r2 = r2[good]
    weights = weights[good]
    radii = np.sqrt(r2)
    dirs = rel / radii[:, None]
    dtheta = (rel[:, 0] * der[:, 1] - rel[:, 1] * der[:, 0]) / r2
    k_max = dens.shape[0] + dens.shape[1] - 2
    coeffs = expc.ray_coefficients(dens, target, dirs)
    moments = expc.radial_moments(kernel, radii, k_max, n_rad)
    inner = np.sum(coeffs * moments, axis=1)
    return weights * inner * dtheta, dirs


def _fan_arc(edge, target: np.ndarray, origin: np.ndarray, dens: np.ndarray,
             kernel, n_ang: int, n_rad: int) -> float:
    """Signed scalar sector of one arc, parametrized in the curve parameter."""
    parts = _fan_arc_parts(edge, target, origin, dens, kernel, n_ang, n_rad)
    if parts is None:
        return 0.0
    return float(np.sum(parts[0]))


def _fan_arc_vector(edge, target: np.ndarray, origin: np.ndarray, dens: np.ndarray,
                    kernel, n_ang: int, n_rad: int) -> np.ndarray:
    """Signed vector sector of one arc: the same rule times the ray direction."""
    parts = _fan_arc_parts(edge, target, origin, dens, kernel, n_ang, n_rad)
    if parts is None:
        return np.zeros(2)
    contrib, dirs = parts
    return contrib @ dirs


def polar_reference(piece, poly, targets, kernel, n_ang: int = DEFAULT_N_ANG,
                    n_rad: int = DEFAULT_N_RAD, origin=None) -> np.ndarray:
    """``int_piece kernel(|x - y|) p(y) dy`` at every target, by the target-centred fan.

    Works for box, polygon and curved pieces and for targets inside, on or
    outside the piece.  ``kernel`` is a ``RadialKernel`` in the split form
    ``log_coeff * (-log r / (2 pi)) + smooth(r)``; ``full_radial_kernel()`` and
    ``prefix_radial_kernel(t)`` are the two used by the study.
    """
    dens, o = _dens_and_origin(piece, poly, origin)
    pts = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    out = np.zeros(pts.shape[0], dtype=np.float64)
    if piece.kind in ("box", "polygon"):
        verts = piece.vertices - o[None, :]
        for idx in range(pts.shape[0]):
            x = pts[idx] - o
            total = 0.0
            for e in range(verts.shape[0]):
                total += _fan_segment(
                    verts[e], verts[(e + 1) % verts.shape[0]], x, dens,
                    kernel, n_ang, n_rad,
                )
            out[idx] = total
        return out
    for idx in range(pts.shape[0]):
        x = pts[idx] - o
        total = 0.0
        for edge in piece.edges or []:
            if getattr(edge, "is_straight", False):
                total += _fan_segment(
                    np.asarray(edge.p0, dtype=np.float64) - o,
                    np.asarray(edge.p1, dtype=np.float64) - o,
                    x, dens, kernel, n_ang, n_rad,
                )
            else:
                total += _fan_arc(edge, x, o, dens, kernel, n_ang, n_rad)
        out[idx] = total
    return out


def polar_gradient(piece, poly, targets, kernel, n_ang: int = DEFAULT_N_ANG,
                   n_rad: int = DEFAULT_N_RAD, origin=None) -> np.ndarray:
    """``grad_x int_piece w(|x - y|) p(y) dy`` at every target, shape ``(n, 2)``.

    ``kernel`` is the radial profile ``-w'(r)``, so the gradient is
    ``int dphi dhat(phi) int_0^{R(phi)} kernel(r) p(x + r dhat) r^{k+1} dr``; the
    angular and radial rules are exactly those of ``polar_reference``, edge by
    edge, with the ray direction carried out of the angular integral.  A curved
    piece dispatches its arcs to the arc fan, so the gradient is taken on the
    piece's own boundary and not on a sampled chord polygon through it.
    """
    dens, o = _dens_and_origin(piece, poly, origin)
    pts = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    out = np.zeros((pts.shape[0], 2), dtype=np.float64)
    if piece.kind in ("box", "polygon"):
        verts = piece.vertices - o[None, :]
        for idx in range(pts.shape[0]):
            x = pts[idx] - o
            total = np.zeros(2)
            for e in range(verts.shape[0]):
                total += _fan_segment_vector(
                    verts[e], verts[(e + 1) % verts.shape[0]], x, dens,
                    kernel, n_ang, n_rad,
                )
            out[idx] = total
        return out
    for idx in range(pts.shape[0]):
        x = pts[idx] - o
        total = np.zeros(2)
        for edge in piece.edges or []:
            if getattr(edge, "is_straight", False):
                total += _fan_segment_vector(
                    np.asarray(edge.p0, dtype=np.float64) - o,
                    np.asarray(edge.p1, dtype=np.float64) - o,
                    x, dens, kernel, n_ang, n_rad,
                )
            else:
                total += _fan_arc_vector(edge, x, o, dens, kernel, n_ang, n_rad)
        out[idx] = total
    return out


def prefix_polar(piece, poly, targets, t: float, n_ang: int = DEFAULT_N_ANG,
                 n_rad: int = DEFAULT_N_RAD, origin=None) -> np.ndarray:
    """Leaf closure ``(2 pi)^{-1} int_piece chi_0(|x - y|; t) p(y) dy``, physical units."""
    return polar_reference(
        piece, poly, targets, prefix_radial_kernel(t), n_ang, n_rad, origin
    )


def polar_leaf(leaf, targets, kernel, n_ang: int = DEFAULT_N_ANG,
               n_rad: int = DEFAULT_N_RAD) -> np.ndarray:
    """Sum of ``polar_reference`` over every piece of a leaf, using the leaf's proxy."""
    out = np.zeros(np.atleast_2d(targets).shape[0], dtype=np.float64)
    for piece in leaf.pieces:
        out += polar_reference(piece, leaf.poly, targets, kernel, n_ang, n_rad)
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def _check_areas(levels: Sequence[int] = (3, 4)) -> list[dict]:
    """Clipped leaf areas of each domain summed against the closed-form area."""
    rows = []
    for name in DOMAIN_NAMES:
        domain = make_domain(name)
        exact = domain.area()
        for level in levels:
            tree = ec.build_quadtree(levels=int(level))
            total = 0.0
            n_cut = 0
            n_pieces = 0
            for box in tree.leaves():
                pieces = domain.clip_leaf(box)
                n_pieces += len(pieces)
                if pieces and not (len(pieces) == 1 and pieces[0].kind == "box"):
                    n_cut += 1
                total += float(sum(p.area() for p in pieces))
            rows.append(
                {
                    "check": "clipped_area_sum",
                    "domain": name,
                    "level": int(level),
                    "area": total,
                    "exact": exact,
                    "abs_error": abs(total - exact),
                    "n_cut_leaves": n_cut,
                    "n_pieces": n_pieces,
                    "tolerance": 1e-13 if name in ("B", "L") else 1e-12,
                    "passed": abs(total - exact)
                    <= (1e-13 if name in ("B", "L") else 1e-12),
                }
            )
    return rows


def _default_poly(box, order: int = 5) -> ec.LeafPoly:
    """A smooth non-polynomial density sampled into the leaf's proxy."""

    def rho(points):
        """Gaussian test density of the study."""
        pts = np.atleast_2d(points)
        return np.exp(-3.0 * (pts[:, 0] ** 2 + pts[:, 1] ** 2))

    return ec.LeafPoly.from_callable(box, rho, int(order))


def _check_plane_waves() -> list[dict]:
    """Closed-form plane-wave coefficients against graded quadrature on polygons."""
    rows = []
    box = ec.Box((0.3125, -0.1875), 0.0625, 4)
    poly = _default_poly(box)
    star = StarDomain()
    ldom = LDomain()
    cases = {
        "leaf_box": ec.make_box_piece(box),
        "L_rectangle": ldom.clip_leaf(ec.Box((-0.75, 0.75), 0.125, 3))[0],
        "triangle": ec.make_polygon_piece(
            np.array([[0.26, -0.24], [0.375, -0.24], [0.30, -0.13]])
        ),
    }
    pieces_star = star.clip_leaf(ec.Box((0.375, 0.375), 0.125, 3))
    if pieces_star:
        cases["star_curved"] = pieces_star[0]
    dk = 2.0 * math.pi / (6.0 * box.side)
    modes = []
    for mx in (0, 1, -3, 7, 20):
        for my in (0, 1, 4, -11, 25):
            modes.append((mx * dk, my * dk))
    kx = np.array([m[0] for m in modes])
    ky = np.array([m[1] for m in modes])
    for label, piece in cases.items():
        closed = plane_wave_coeffs(piece, poly, kx, ky)
        quad, n_nodes = plane_wave_quadrature_reference(piece, poly, kx, ky)
        scale = float(np.max(np.abs(quad)))
        err = float(np.max(np.abs(closed - quad)))
        rows.append(
            {
                "check": "plane_wave_closed_vs_quadrature",
                "case": label,
                "kind": piece.kind,
                "max_abs_error": err,
                "rel_error": err / scale if scale else 0.0,
                "reference_nodes": n_nodes,
                "tolerance": 1e-13,
                "passed": (err / scale if scale else 0.0) <= 1e-13,
            }
        )
    return rows


def _check_polar() -> list[dict]:
    """Polar quadrature against the leaf-residual routine and the separable rule."""
    rows = []
    box = ec.Box((0.3125, -0.1875), 0.0625, 4)
    poly = _default_poly(box)
    piece = ec.make_box_piece(box)
    targets = np.array(
        [[0.3125, -0.1875], [0.30, -0.19], [0.40, -0.10], [0.3125, -0.125]]
    )
    t_leaf = ec.t_l(4)
    kernel = prefix_radial_kernel(t_leaf)
    mine = polar_reference(piece, poly, targets, kernel)
    o = piece_origin(piece)
    dens = poly.monomial_coeffs(origin=tuple(o))
    theirs = np.array(
        [
            expc.polar_polygon_potential(
                piece.vertices - o, x - o, dens, kernel,
                DEFAULT_N_ANG, DEFAULT_N_RAD,
            )
            for x in targets
        ]
    )
    err = float(np.max(np.abs(mine - theirs)))
    rows.append(
        {
            "check": "polar_vs_experiment_c_polygon_routine",
            "max_abs_error": err,
            "scale": float(np.max(np.abs(theirs))),
            "tolerance": 1e-14,
            "passed": err <= 1e-14,
        }
    )
    a1, b1, a2, b2 = box.bounds
    rect = (a1 - o[0], b1 - o[0], a2 - o[1], b2 - o[1])
    sep = np.array(
        [expc.separable_prefix_rect(rect, x - o, dens, t_leaf)[0] for x in targets]
    )
    err_sep = float(np.max(np.abs(mine - sep)))
    rows.append(
        {
            "check": "polar_vs_separable_prefix",
            "max_abs_error": err_sep,
            "scale": float(np.max(np.abs(sep))),
            "tolerance": 1e-12,
            "passed": err_sep <= 1e-12,
        }
    )
    return rows


def _chord_loop(piece, samples: int = 33) -> np.ndarray:
    """Chord polygon through a curved piece's edges, ``samples`` points per edge."""
    pts = []
    for edge in piece.edges or []:
        s = np.linspace(edge.s0, edge.s1, int(samples))[:-1]
        pts.append(edge.curve(s))
    return np.vstack(pts) if pts else piece.vertices


def _check_polar_gradient() -> list[dict]:
    """The vector fan against a difference of the scalar fan, on a box and on an arc.

    The third row is not a pass/fail check but the measurement that motivates the
    arc fan: the same gradient taken on a chord polygon through the arc, at the
    sampling an earlier revision used, against the arc-aware value.
    """
    rows = []
    t_leaf = ec.t_l(4)
    kernel = prefix_radial_kernel(t_leaf)
    grad_kernel = smooth_radial_kernel(
        lambda r, a=t_leaf: np.exp(-np.asarray(r) ** 2 / (4.0 * a))
        / (2.0 * math.pi * np.asarray(r)),
        math.sqrt(t_leaf),
        "grad",
    )
    step = 1.0e-6

    def _fd(piece, poly, x):
        """Central difference of ``polar_reference`` at one target."""
        out = np.zeros(2)
        for axis in range(2):
            shift = np.zeros(2)
            shift[axis] = step
            plus = polar_reference(piece, poly, (x + shift)[None, :], kernel)[0]
            minus = polar_reference(piece, poly, (x - shift)[None, :], kernel)[0]
            out[axis] = (plus - minus) / (2.0 * step)
        return out

    box = ec.Box((0.3125, -0.1875), 0.0625, 4)
    poly = _default_poly(box)
    piece = ec.make_box_piece(box)
    target = np.array([0.30, -0.19])
    mine = polar_gradient(piece, poly, target[None, :], grad_kernel)[0]
    err = float(np.max(np.abs(mine - _fd(piece, poly, target))))
    rows.append(
        {
            "check": "polar_gradient_vs_difference_box",
            "max_abs_error": err,
            "scale": float(np.max(np.abs(mine))),
            "tolerance": 1e-8,
            "passed": err <= 1e-8,
        }
    )

    star = make_domain("S")
    tree = ec.build_quadtree(levels=4)
    leaves = star.leaves(tree, lambda p: np.exp(-3.0 * np.sum(p * p, axis=1)), 6)
    cut = [
        lf for lf in leaves if any(p.kind == "curved" for p in lf.pieces)
    ]
    if cut:
        leaf = cut[len(cut) // 2]
        curved = next(p for p in leaf.pieces if p.kind == "curved")
        sample = curved.edges[0].curve(
            np.array([0.5 * (curved.edges[0].s0 + curved.edges[0].s1)])
        )[0]
        h_leaf = tree.h(4)
        normal = sample / max(float(np.hypot(*sample)), 1e-30)
        x = sample - 1.0e-4 * h_leaf * normal
        arc_grad = polar_gradient(curved, leaf.poly, x[None, :], grad_kernel)[0]
        err_arc = float(np.max(np.abs(arc_grad - _fd(curved, leaf.poly, x))))
        rows.append(
            {
                "check": "polar_gradient_vs_difference_arc",
                "max_abs_error": err_arc,
                "scale": float(np.max(np.abs(arc_grad))),
                "tolerance": 1e-7,
                "passed": err_arc <= 1e-7,
            }
        )
        chord = ec.make_polygon_piece(_chord_loop(curved, 33), curved.box)
        chord_grad = polar_gradient(chord, leaf.poly, x[None, :], grad_kernel)[0]
        fine = ec.make_polygon_piece(_chord_loop(curved, 513), curved.box)
        fine_grad = polar_gradient(fine, leaf.poly, x[None, :], grad_kernel)[0]
        rows.append(
            {
                "check": "curved_gradient_chord_loop_error",
                "chord_33_vs_arc": float(np.max(np.abs(chord_grad - arc_grad))),
                "chord_513_vs_arc": float(np.max(np.abs(fine_grad - arc_grad))),
                "scale": float(np.max(np.abs(arc_grad))),
                "note": (
                    "informational: the error a chord-polygon gradient makes on "
                    "an arc, which the arc fan removes"
                ),
                "passed": True,
            }
        )
    return rows


def run_self_test() -> tuple[list[dict], bool]:
    """Run the module's own checks and report the rows and an overall pass flag."""
    rows: list[dict] = []
    rows.extend(_check_areas())
    rows.extend(_check_plane_waves())
    rows.extend(_check_polar())
    rows.extend(_check_polar_gradient())
    return rows, all(bool(r.get("passed", True)) for r in rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: run the self-test and write a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="output directory for the summary")
    args = parser.parse_args(argv)
    rows, ok = run_self_test()
    payload = {"rows": rows, "all_passed": bool(ok)}
    text = json.dumps(payload, indent=2, sort_keys=True, default=float)
    print(text)
    if args.out:
        out = Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "experiment_e_geometry_selftest.json").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
