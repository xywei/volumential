__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """Experimental DMK-like near-field table-build splitting.

This module prototypes a table-entry decomposition that is deliberately
different from Volumential's existing Helmholtz/Yukawa ``S_p/R_p`` split.  The
kernel is split before quadrature into

* a compactly supported local singular part, evaluated from an asymptotic
  Taylor expansion of the source mode and analytic radial/angular moments, and
* a smooth remainder, evaluated by tensor-product Gauss quadrature that may use
  a higher order than the table's interpolation order.

The first implementation targets 2D Laplace table entries on the template box.
It is intended for convergence experiments and for scoping issue #104; it is
not wired into the production table manager yet.
"""

from dataclasses import dataclass
from math import comb, exp, lgamma, log, pi

import numpy as np
from scipy import special as sps


EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class CompactWindowConfig:
    """Compact radial window used by the DMK-like table split.

    ``sigma`` is the support radius.  The window is one for
    ``r <= plateau_fraction*sigma`` and zero for ``r >= sigma``.  Between those
    radii it uses a generalized smoothstep polynomial with the requested number
    of vanishing derivatives at both transition endpoints.
    """

    sigma: float
    plateau_fraction: float = 0.5
    smoothness_order: int = 4

    def __post_init__(self):
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError(f"sigma must be positive and finite, got {self.sigma!r}")
        object.__setattr__(
            self,
            "smoothness_order",
            _validate_integer_order(
                self.smoothness_order,
                "smoothness_order",
                minimum=0,
            ),
        )
        if not 0 < self.plateau_fraction < 1:
            raise ValueError(
                "plateau_fraction must be between 0 and 1, "
                f"got {self.plateau_fraction!r}"
            )


@dataclass(frozen=True)
class HeatKernelConfig:
    """Heat-kernel smoothing scale for the 2D Laplace split.

    This uses the Gaussian-smoothed Green's function.  In 3D this corresponds to
    the familiar ``erf(r/sigma)/r`` smooth part; for 2D Laplace the local residual
    is ``E1((r/sigma)**2)/(4*pi)``.
    """

    sigma: float
    tail_cutoff_sigma: float = 5.0

    def __post_init__(self):
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError(f"sigma must be positive and finite, got {self.sigma!r}")
        if not np.isfinite(self.tail_cutoff_sigma) or self.tail_cutoff_sigma <= 0:
            raise ValueError(
                "tail_cutoff_sigma must be positive and finite, "
                f"got {self.tail_cutoff_sigma!r}"
            )


@dataclass(frozen=True)
class Laplace2DDMKSplitResult:
    """Result of one split table-entry evaluation."""

    value: float
    local_value: float
    smooth_value: float
    support_relation: str


class CompactSupportIntersectionError(ValueError):
    """Raised when compact local support intersects the source box boundary."""


class HeatTailBoundaryError(ValueError):
    """Raised when heat-local full-space moments are unsafe for a finite box."""


def _validate_integer_order(order, name, minimum):
    if isinstance(order, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {order!r}")

    try:
        order_int = int(order)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {order!r}") from exc

    if order_int != order:
        raise ValueError(f"{name} must be an integer, got {order!r}")
    if order_int < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {order!r}")

    return order_int


def gauss_legendre_rule(order, a=0.0, b=1.0):
    """Return Gauss-Legendre nodes/weights on ``[a, b]``."""

    order = _validate_integer_order(order, "order", minimum=1)

    nodes, weights = np.polynomial.legendre.leggauss(order)
    center = 0.5 * (a + b)
    radius = 0.5 * (b - a)
    return center + radius * nodes, radius * weights


def gauss_legendre_tensor_points(order, bounds=((0.0, 1.0), (0.0, 1.0))):
    """Return flattened tensor-product Gauss nodes and weights in 2D."""

    x_nodes, x_weights = gauss_legendre_rule(order, *bounds[0])
    y_nodes, y_weights = gauss_legendre_rule(order, *bounds[1])
    xx, yy = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    wx, wy = np.meshgrid(x_weights, y_weights, indexing="ij")
    return xx.ravel(), yy.ravel(), (wx * wy).ravel()


def tensor_gauss_point(q_order, point_index, bounds=((0.0, 1.0), (0.0, 1.0))):
    """Return a Volumential-compatible 2D tensor Gauss target point."""

    if point_index < 0 or point_index >= q_order * q_order:
        raise ValueError(
            f"point_index must be in [0, {q_order * q_order}), got {point_index}"
        )

    ix = point_index // q_order
    iy = point_index % q_order
    x_nodes, _ = gauss_legendre_rule(q_order, *bounds[0])
    y_nodes, _ = gauss_legendre_rule(q_order, *bounds[1])
    return (float(x_nodes[ix]), float(y_nodes[iy]))


def minimum_boundary_spacing(q_order):
    """Return the nearest distance from any ``q_order`` Gauss node to ``[0, 1]``."""

    nodes, _ = gauss_legendre_rule(q_order)
    return float(min(np.min(nodes), np.min(1.0 - nodes)))


def lagrange_polynomial_coefficients(nodes, index):
    """Return monomial coefficients for the 1D Lagrange basis polynomial.

    Coefficients are in ascending order, i.e. ``c[k]`` multiplies ``x**k``.
    """

    nodes = np.asarray(nodes, dtype=np.float64)
    if index < 0 or index >= nodes.size:
        raise ValueError(f"basis index {index} outside node set of size {nodes.size}")

    roots = np.delete(nodes, index)
    coeffs = np.polynomial.polynomial.polyfromroots(roots)
    denom = np.prod(nodes[index] - roots)
    return coeffs / denom


def tensor_lagrange_mode_coefficients(q_order, mode_index):
    """Return monomial coefficients for a 2D tensor Lagrange source mode.

    The mode ordering matches :class:`NearFieldInteractionTable`: x-index first,
    y-index second.
    """

    if mode_index < 0 or mode_index >= q_order * q_order:
        raise ValueError(
            f"mode_index must be in [0, {q_order * q_order}), got {mode_index}"
        )

    ix = mode_index // q_order
    iy = mode_index % q_order
    nodes, _ = gauss_legendre_rule(q_order)
    cx = lagrange_polynomial_coefficients(nodes, ix)
    cy = lagrange_polynomial_coefficients(nodes, iy)
    return np.multiply.outer(cx, cy)


def evaluate_polynomial_2d(coeffs, x, y):
    """Evaluate an ascending-coefficient 2D polynomial."""

    return np.polynomial.polynomial.polyval2d(x, y, coeffs)


def shifted_polynomial_coefficients(coeffs, center, max_total_order=None):
    """Expand ``p(x, y)`` in powers of ``x-center[0]`` and ``y-center[1]``."""

    coeffs = np.asarray(coeffs, dtype=np.float64)
    tx, ty = center
    shifted = np.zeros_like(coeffs, dtype=np.float64)

    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            cij = coeffs[i, j]
            if cij == 0:
                continue

            for a in range(i + 1):
                x_scale = comb(i, a) * tx ** (i - a)
                for b in range(j + 1):
                    if max_total_order is not None and a + b > max_total_order:
                        continue
                    shifted[a, b] += cij * x_scale * comb(j, b) * ty ** (j - b)

    if max_total_order is not None:
        for a in range(shifted.shape[0]):
            for b in range(shifted.shape[1]):
                if a + b > max_total_order:
                    shifted[a, b] = 0.0

    return shifted


def _smoothstep_coefficients(smoothness_order):
    """Return coefficients of the generalized smoothstep polynomial."""

    n = int(smoothness_order)
    coeffs = np.zeros(2 * n + 2, dtype=np.float64)
    for k in range(n + 1):
        power = n + 1 + k
        coeffs[power] = ((-1) ** k) * comb(n + k, k) * comb(2 * n + 1, n - k)
    return coeffs


def _window_transition_polynomial(config):
    """Return ``eta(t)`` coefficients on ``plateau_fraction < t < 1``."""

    a = float(config.plateau_fraction)
    scale = 1.0 - a
    smoothstep = _smoothstep_coefficients(config.smoothness_order)
    coeffs = np.zeros_like(smoothstep)
    coeffs[0] = 1.0

    for j, sj in enumerate(smoothstep):
        if sj == 0:
            continue
        prefactor = -sj / scale**j
        for ell in range(j + 1):
            coeffs[ell] += prefactor * comb(j, ell) * (-a) ** (j - ell)

    return coeffs


def compact_window(t, config):
    """Evaluate the compact radial window at ``t = r/sigma``."""

    t_arr = np.asarray(t, dtype=np.float64)
    result = np.zeros(t_arr.shape, dtype=np.float64)
    plateau = t_arr <= config.plateau_fraction
    transition = np.logical_and(t_arr > config.plateau_fraction, t_arr < 1.0)
    result[plateau] = 1.0
    if np.any(transition):
        coeffs = _window_transition_polynomial(config)
        result[transition] = np.polynomial.polynomial.polyval(t_arr[transition], coeffs)
    result = np.clip(result, 0.0, 1.0)
    if np.isscalar(t):
        return float(result)
    return result


def laplace2d_kernel_radius(r):
    """2D Laplace Green's function as a function of radius."""

    return -np.log(r) / (2.0 * pi)


def laplace2d_smooth_kernel(dx, dy, config):
    """Smooth remainder kernel ``G(r)*(1-eta(r/sigma))``."""

    r = np.sqrt(np.asarray(dx) ** 2 + np.asarray(dy) ** 2)
    result = np.zeros(np.shape(r), dtype=np.float64)
    active = r > config.plateau_fraction * config.sigma
    if np.any(active):
        rr = r[active] if np.ndim(r) else r
        window = compact_window(rr / config.sigma, config)
        vals = laplace2d_kernel_radius(rr) * (1.0 - window)
        if np.ndim(r):
            result[active] = vals
        else:
            result = np.asarray(vals, dtype=np.float64)
    if np.isscalar(dx) and np.isscalar(dy):
        return float(result)
    return result


def laplace2d_heat_local_kernel_radius(r, config):
    """Heat-kernel local residual for 2D Laplace.

    The residual is ``E1((r/sigma)**2)/(4*pi)``.  It has the same logarithmic
    singularity as the original Green's function but decays exponentially away
    from the target.
    """

    r_arr = np.asarray(r, dtype=np.float64)
    z = (r_arr / config.sigma) ** 2
    result = sps.exp1(z) / (4.0 * pi)
    if np.isscalar(r):
        return float(result)
    return result


def laplace2d_heat_smooth_kernel(dx, dy, config):
    """Gaussian-smoothed 2D Laplace Green's function.

    This is ``G(r) - E1((r/sigma)**2)/(4*pi)`` and has the finite target value
    ``(EulerGamma - 2*log(sigma))/(4*pi)``.
    """

    r = np.sqrt(np.asarray(dx) ** 2 + np.asarray(dy) ** 2)
    if np.isscalar(dx) and np.isscalar(dy):
        if r == 0:
            return (EULER_GAMMA - 2.0 * log(config.sigma)) / (4.0 * pi)
        return float(
            laplace2d_kernel_radius(r)
            - laplace2d_heat_local_kernel_radius(r, config)
        )

    result = np.empty(np.shape(r), dtype=np.float64)
    zero = r == 0
    if np.any(~zero):
        rr = r[~zero]
        vals = laplace2d_kernel_radius(rr) - laplace2d_heat_local_kernel_radius(
            rr,
            config,
        )
        result[~zero] = vals
    if np.any(zero):
        result[zero] = (EULER_GAMMA - 2.0 * log(config.sigma)) / (4.0 * pi)
    return result


def support_relation_to_box(
    target,
    config,
    bounds=((0.0, 1.0), (0.0, 1.0)),
    tol=0.0,
):
    """Classify compact-window support relative to a rectangular source box.

    Returns ``"inside"`` if the entire support disk lies in the box, ``"outside"``
    if it is disjoint from the box, and ``"intersects"`` otherwise.
    """

    tx, ty = target
    (a, b), (c, d) = bounds
    sigma = config.sigma

    if a <= tx <= b and c <= ty <= d:
        boundary_distance = min(tx - a, b - tx, ty - c, d - ty)
        if boundary_distance + tol >= sigma:
            return "inside"

    dx = max(a - tx, 0.0, tx - b)
    dy = max(c - ty, 0.0, ty - d)
    if np.hypot(dx, dy) + tol >= sigma:
        return "outside"

    return "intersects"


def heat_tail_relation_to_box(
    target,
    config,
    bounds=((0.0, 1.0), (0.0, 1.0)),
    tol=0.0,
):
    """Classify the effective heat-local tail relative to a source box."""

    tx, ty = target
    (a, b), (c, d) = bounds
    radius = config.tail_cutoff_sigma * config.sigma

    if a <= tx <= b and c <= ty <= d:
        boundary_distance = min(tx - a, b - tx, ty - c, d - ty)
        if boundary_distance + tol >= radius:
            return "tail-contained"

    dx = max(a - tx, 0.0, tx - b)
    dy = max(c - ty, 0.0, ty - d)
    if np.hypot(dx, dy) + tol >= radius:
        return "tail-outside"

    return "tail-intersects"


def _integral_power(lo, hi, exponent):
    return (hi ** (exponent + 1) - lo ** (exponent + 1)) / (exponent + 1)


def _integral_power_log(lo, hi, exponent):
    power = exponent + 1

    def antiderivative(x):
        if x == 0:
            return 0.0
        return x**power * (log(x) / power - 1.0 / power**2)

    return antiderivative(hi) - antiderivative(lo)


def _window_power_integrals(config, exponent):
    """Return ``int t**exponent eta(t) dt`` and with an extra ``log(t)``."""

    a = float(config.plateau_fraction)
    coeffs = _window_transition_polynomial(config)

    plain = _integral_power(0.0, a, exponent)
    log_part = _integral_power_log(0.0, a, exponent)

    for degree, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        total_degree = exponent + degree
        plain += coeff * _integral_power(a, 1.0, total_degree)
        log_part += coeff * _integral_power_log(a, 1.0, total_degree)

    return plain, log_part


def _angular_moment(mx, my):
    if mx % 2 or my % 2:
        return 0.0

    a = mx // 2
    b = my // 2
    return float(2.0 * np.exp(lgamma(a + 0.5) + lgamma(b + 0.5) - lgamma(a + b + 1)))


def laplace2d_local_moment(mx, my, config):
    """Moment of the compactly supported 2D Laplace local kernel."""

    angular = _angular_moment(mx, my)
    if angular == 0.0:
        return 0.0

    radial_power = mx + my + 1
    plain, log_part = _window_power_integrals(config, radial_power)
    sigma_power = config.sigma ** (mx + my + 2)
    radial = -(sigma_power / (2.0 * pi)) * (log(config.sigma) * plain + log_part)
    return angular * radial


def laplace2d_heat_local_moment(mx, my, config):
    """Full-space moment of the heat-kernel local residual."""

    angular = _angular_moment(mx, my)
    if angular == 0.0:
        return 0.0

    total_degree = mx + my
    sigma_power = config.sigma ** (total_degree + 2)
    radial = (
        sigma_power
        * exp(lgamma(0.5 * total_degree + 1.0))
        / (8.0 * pi * (0.5 * total_degree + 1.0))
    )
    return angular * radial


def laplace2d_local_expansion_integral(coeffs, target, expansion_order, config):
    """Evaluate the local compact singular contribution by Taylor moments."""

    expansion_order = _validate_integer_order(
        expansion_order,
        "expansion_order",
        minimum=0,
    )
    shifted = shifted_polynomial_coefficients(
        coeffs,
        center=target,
        max_total_order=expansion_order,
    )
    total = 0.0
    for mx in range(shifted.shape[0]):
        for my in range(shifted.shape[1]):
            if mx + my > expansion_order:
                continue
            coeff = shifted[mx, my]
            if coeff != 0:
                total += coeff * laplace2d_local_moment(mx, my, config)
    return float(total)


def laplace2d_heat_local_expansion_integral(coeffs, target, expansion_order, config):
    """Evaluate the heat-local contribution by full-space Taylor moments."""

    expansion_order = _validate_integer_order(
        expansion_order,
        "expansion_order",
        minimum=0,
    )
    shifted = shifted_polynomial_coefficients(
        coeffs,
        center=target,
        max_total_order=expansion_order,
    )
    total = 0.0
    for mx in range(shifted.shape[0]):
        for my in range(shifted.shape[1]):
            if mx + my > expansion_order:
                continue
            coeff = shifted[mx, my]
            if coeff != 0:
                total += coeff * laplace2d_heat_local_moment(mx, my, config)
    return float(total)


def laplace2d_smooth_gauss_integral(
    coeffs,
    target,
    config,
    smooth_order,
    bounds=((0.0, 1.0), (0.0, 1.0)),
):
    """Evaluate the smooth remainder contribution by tensor-product Gauss."""

    xx, yy, weights = gauss_legendre_tensor_points(smooth_order, bounds=bounds)
    density = evaluate_polynomial_2d(coeffs, xx, yy)
    kernel = laplace2d_smooth_kernel(xx - target[0], yy - target[1], config)
    return float(np.dot(weights, density * kernel))


def laplace2d_heat_smooth_gauss_integral(
    coeffs,
    target,
    config,
    smooth_order,
    bounds=((0.0, 1.0), (0.0, 1.0)),
):
    """Evaluate the heat-smoothed contribution by tensor-product Gauss."""

    xx, yy, weights = gauss_legendre_tensor_points(smooth_order, bounds=bounds)
    density = evaluate_polynomial_2d(coeffs, xx, yy)
    kernel = laplace2d_heat_smooth_kernel(xx - target[0], yy - target[1], config)
    return float(np.dot(weights, density * kernel))


def laplace2d_dmk_split_integral(
    coeffs,
    target,
    config,
    expansion_order,
    smooth_order,
    bounds=((0.0, 1.0), (0.0, 1.0)),
):
    """Evaluate one 2D Laplace table entry by the experimental DMK-like split."""

    expansion_order = _validate_integer_order(
        expansion_order,
        "expansion_order",
        minimum=0,
    )
    smooth_order = _validate_integer_order(smooth_order, "smooth_order", minimum=1)

    relation = support_relation_to_box(target, config, bounds=bounds)
    if relation == "intersects":
        raise CompactSupportIntersectionError(
            "compact support intersects the source box boundary; choose a smaller "
            "sigma or use an intersection-aware local integration path"
        )

    if relation == "inside":
        local_value = laplace2d_local_expansion_integral(
            coeffs,
            target=target,
            expansion_order=expansion_order,
            config=config,
        )
    else:
        local_value = 0.0

    smooth_value = laplace2d_smooth_gauss_integral(
        coeffs,
        target=target,
        config=config,
        smooth_order=smooth_order,
        bounds=bounds,
    )
    return Laplace2DDMKSplitResult(
        value=local_value + smooth_value,
        local_value=local_value,
        smooth_value=smooth_value,
        support_relation=relation,
    )


def laplace2d_heat_split_integral(
    coeffs,
    target,
    config,
    expansion_order,
    smooth_order,
    bounds=((0.0, 1.0), (0.0, 1.0)),
):
    """Evaluate one 2D Laplace table entry by a heat-kernel split.

    The local part uses full-space Taylor moments of the exponentially decaying
    residual.  For finite boxes this is most accurate when ``sigma`` is small
    compared with the target's distance to the source-box boundary.
    """

    expansion_order = _validate_integer_order(
        expansion_order,
        "expansion_order",
        minimum=0,
    )
    smooth_order = _validate_integer_order(smooth_order, "smooth_order", minimum=1)

    relation = heat_tail_relation_to_box(target, config, bounds=bounds)
    if relation == "tail-intersects":
        raise HeatTailBoundaryError(
            "heat-local tail intersects the source box boundary; choose a smaller "
            "sigma or use boundary-corrected local moments"
        )

    if relation == "tail-contained":
        local_value = laplace2d_heat_local_expansion_integral(
            coeffs,
            target=target,
            expansion_order=expansion_order,
            config=config,
        )
    else:
        local_value = 0.0

    smooth_value = laplace2d_heat_smooth_gauss_integral(
        coeffs,
        target=target,
        config=config,
        smooth_order=smooth_order,
        bounds=bounds,
    )
    return Laplace2DDMKSplitResult(
        value=local_value + smooth_value,
        local_value=local_value,
        smooth_value=smooth_value,
        support_relation=relation,
    )


def laplace2d_full_gauss_integral(
    coeffs,
    target,
    quad_order,
    bounds=((0.0, 1.0), (0.0, 1.0)),
):
    """High-order tensor Gauss integral of the unsplit nonsingular case."""

    xx, yy, weights = gauss_legendre_tensor_points(quad_order, bounds=bounds)
    r = np.sqrt((xx - target[0]) ** 2 + (yy - target[1]) ** 2)
    if np.any(r == 0):
        raise ValueError("full Gauss reference hit the singular target point")
    integrand = evaluate_polynomial_2d(coeffs, xx, yy) * laplace2d_kernel_radius(r)
    return float(np.dot(weights, integrand))


def sweep_laplace2d_dmk_split(
    q_order,
    mode_index,
    target_index,
    sigmas,
    expansion_orders,
    smooth_orders,
    reference_value,
    plateau_fraction=0.5,
    smoothness_order=4,
):
    """Sweep split parameters for one 2D Laplace table entry."""

    coeffs = tensor_lagrange_mode_coefficients(q_order, mode_index)
    target = tensor_gauss_point(q_order, target_index)
    sigmas = tuple(sigmas)
    expansion_orders = tuple(
        _validate_integer_order(order, "expansion_order", minimum=0)
        for order in expansion_orders
    )
    smooth_orders = tuple(
        _validate_integer_order(order, "smooth_order", minimum=1)
        for order in smooth_orders
    )
    rows = []

    for sigma in sigmas:
        config = CompactWindowConfig(
            sigma=float(sigma),
            plateau_fraction=plateau_fraction,
            smoothness_order=smoothness_order,
        )
        for expansion_order in expansion_orders:
            for smooth_order in smooth_orders:
                try:
                    result = laplace2d_dmk_split_integral(
                        coeffs,
                        target=target,
                        config=config,
                        expansion_order=expansion_order,
                        smooth_order=smooth_order,
                    )
                    error = abs(result.value - reference_value)
                    rows.append(
                        {
                            "sigma": float(sigma),
                            "expansion_order": expansion_order,
                            "smooth_order": smooth_order,
                            "value": result.value,
                            "local_value": result.local_value,
                            "smooth_value": result.smooth_value,
                            "support_relation": result.support_relation,
                            "abs_error": float(error),
                        }
                    )
                except CompactSupportIntersectionError as exc:
                    rows.append(
                        {
                            "sigma": float(sigma),
                            "expansion_order": expansion_order,
                            "smooth_order": smooth_order,
                            "value": np.nan,
                            "local_value": np.nan,
                            "smooth_value": np.nan,
                            "support_relation": "intersects",
                            "abs_error": np.inf,
                            "error_message": str(exc),
                        }
                    )

    return rows


def sweep_laplace2d_heat_split(
    q_order,
    mode_index,
    target_index,
    sigmas,
    expansion_orders,
    smooth_orders,
    reference_value,
):
    """Sweep heat-kernel split parameters for one 2D Laplace table entry."""

    coeffs = tensor_lagrange_mode_coefficients(q_order, mode_index)
    target = tensor_gauss_point(q_order, target_index)
    sigmas = tuple(sigmas)
    expansion_orders = tuple(
        _validate_integer_order(order, "expansion_order", minimum=0)
        for order in expansion_orders
    )
    smooth_orders = tuple(
        _validate_integer_order(order, "smooth_order", minimum=1)
        for order in smooth_orders
    )
    rows = []

    for sigma in sigmas:
        config = HeatKernelConfig(sigma=float(sigma))
        for expansion_order in expansion_orders:
            for smooth_order in smooth_orders:
                try:
                    result = laplace2d_heat_split_integral(
                        coeffs,
                        target=target,
                        config=config,
                        expansion_order=expansion_order,
                        smooth_order=smooth_order,
                    )
                    error = abs(result.value - reference_value)
                    rows.append(
                        {
                            "sigma": float(sigma),
                            "expansion_order": expansion_order,
                            "smooth_order": smooth_order,
                            "value": result.value,
                            "local_value": result.local_value,
                            "smooth_value": result.smooth_value,
                            "support_relation": result.support_relation,
                            "abs_error": float(error),
                        }
                    )
                except HeatTailBoundaryError as exc:
                    rows.append(
                        {
                            "sigma": float(sigma),
                            "expansion_order": expansion_order,
                            "smooth_order": smooth_order,
                            "value": np.nan,
                            "local_value": np.nan,
                            "smooth_value": np.nan,
                            "support_relation": "tail-intersects",
                            "abs_error": np.inf,
                            "error_message": str(exc),
                        }
                    )

    return rows


# vim: foldmethod=marker:filetype=python
