"""Certified RKE assembly of fixed-parameter near-field tables.

This module assembles fixed-parameter Helmholtz and Yukawa near-field
interaction tables as linear combinations of canonical, parameter-independent
channel tables:

- 2D: ``T(k) = T[Laplace] + c0(k) T[const]
  + sum_n ( a_n(k) T[r^{2n} log r] + b_n(k) T[r^{2n}] )``
- 3D: ``T(k) = T[Laplace] + sum_n c_n(k) T[r^{n-1}]``

with the exact small-argument series coefficients (the same series that
defines :class:`~volumential.expansion_wrangler_fpnd.\
_HelmholtzSplitSeriesRemainderKernel`).  Yukawa uses the principal-branch
substitution ``k = i lam``, under which the assembled table is real.

Every channel integrand is elementary (powers and ``log``), so all channel
tables build through the batched (parallel) Duffy path; no special-function
quadrature is needed.  The channel family is parameter independent: once
built and cached, any further parameter costs only coefficient evaluation
and a linear combination over the symmetry-reduced entries.

The truncation order is chosen from the requested tolerance via an explicit
majorant of the omitted series tail on the near-field separation region, and
the returned certificate records that bound together with the numerically
computed :math:`L^1` norms of the source basis functions that convert the
kernel-space bound into a table-entry bound.

Windowed mode
-------------

The windowed assembler (:func:`assemble_windowed_parameterized_table`) keeps
the classical table count but replaces the polynomially growing channels
``r^{2m} log r`` / ``r^{m-1}`` by Gaussian-windowed channels

- 2D: ``chi_m(r) = (1/2) (r^2/4)^m Gamma(-m, x)``
- 3D: ``chi_m(r) = (1/(2 sqrt(pi))) (r^2/4)^(m-1/2) Gamma(1/2-m, x)``

with ``x = r^2 / (4 t_w)`` and window scale ``t_w = (b / Theta)^2`` for
source-box extent ``b`` and the single design declaration ``Theta``
(``window_theta``).  The channels carry the kernels' full singular germs but
die off beyond ``r ~ b / Theta``.  Internally, tables store the normalized
channels ``psi_m = chi_m / t_w^m`` and pair them with coefficients
``(-zeta t_w)^m / m!``.  Their magnitude is bounded by
``|zeta t_w|^m / m! = (theta/Theta)^{2m}/m!`` for every covered
squared-frequency parameter ``zeta`` (``zeta = lam^2`` for Yukawa,
``zeta = -k^2`` for Helmholtz, complex ``zeta`` for damped waves), so neither
factor carries the opposing ``t_w^m`` scaling that overflows or underflows
before their product does.

The online part is the smooth remainder
``R = G - sum_{m<p_star} ((-zeta t_w)^m/m!) psi_m``, evaluated pointwise from
the kernel and the closed channel forms and integrated against the source
modes by a tensor-product Gauss-Legendre rule.
Because ``R`` is defined as the exact difference, the assembly has no series
truncation anywhere: the certificate's ``truncation_tail_bound`` is
structurally ``0.0``, and the only error terms are the two already-owned
quadrature kinds (singular channel quadrature and smooth remainder
quadrature).  The channel family depends on the declaration ``Theta`` only —
never on the swept kernel parameter — so one cached family serves every
``theta = parameter * b`` up to ``Theta``.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import operator
import os
import uuid
from pathlib import Path

import numpy as np

import volumential.opcounters as opcounters

#: Relative slack on the declared-window coverage test.  The declaration
#: certifies the *closed* disk ``|zeta| <= (Theta/b)**2``, so a local theta
#: exactly at ``Theta`` must be accepted; this absorbs the rounding of
#: ``parameter * root_extent * 0.5**level``, and nothing more.  Exported
#: because a consumer that decides whether a refusal was legitimate has to
#: use the same boundary the assembler refused on -- a looser one turns a
#: correct out-of-window refusal into a reported gate failure.
WINDOW_COVERAGE_RELATIVE_TOLERANCE = 1.0e-12

__all__ = [
    "WINDOW_COVERAGE_RELATIVE_TOLERANCE",
    "RKEConditioningError",
    "RKETruncationError",
    "RKEWindowConditioningError",
    "RKEWindowCoverageError",
    "assemble_parameterized_table",
    "assemble_windowed_damped_table",
    "assemble_windowed_parameterized_table",
    "choose_truncation_order",
    "damped_kernel_radial",
    "get_windowed_channel_table",
    "windowed_channel_profile",
    "windowed_remainder_profile",
]


logger = logging.getLogger(__name__)

_EULER_GAMMA = np.euler_gamma
_WINDOWED_CHANNEL_CACHE_SCHEMA = 2
_WINDOWED_CHANNEL_NORMALIZATION = "psi=chi/t_w**m"


def _require_integer(name, value, *, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    result = int(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _require_finite_positive(name, value):
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _require_dimension(dim):
    dim = _require_integer("dim", dim)
    if dim not in (2, 3):
        raise NotImplementedError("RKE assembly supports only 2D and 3D")
    return dim


def _require_dim_q_order(dim, q_order):
    return (
        _require_dimension(dim),
        _require_integer("q_order", q_order, minimum=1),
    )


# {{{ refusal taxonomy

# The two certified refusal modes of the classical assembly are named on the
# exception itself, so callers (the sweep driver and its figures) classify
# them structurally instead of by matching free-form message text.  Both keep
# their historical base classes, so ``except (ValueError, RuntimeError)`` and
# message-matching callers are unaffected.

class RKETruncationError(ValueError):
    """The series tail majorant never falls below the requested tolerance."""

    refusal_kind = "uncertifiable"


class RKEConditioningError(RuntimeError):
    """Float64 recombination of the assembly is not certifiably conditioned."""

    refusal_kind = "ill-conditioned"


class RKEWindowCoverageError(ValueError):
    """The squared-frequency parameter lies outside the declared window."""

    refusal_kind = "outside-window"


class RKEWindowConditioningError(RuntimeError):
    """Windowed recombination is not certifiably conditioned."""

    refusal_kind = "ill-conditioned"

# }}}


# {{{ series coefficients

def _coefficients_2d(k: complex, n_terms: int):
    """Constant coefficient and per-order (log, power) coefficients for
    ``G_k - G_0`` in 2D, following the exact Bessel series."""
    k = np.complex128(k)
    log_k_half = np.log(0.5 * k)
    c0 = np.complex128(0.25j - (log_k_half + _EULER_GAMMA) / (2.0 * np.pi))

    coeff_log = np.empty(n_terms + 1, dtype=np.complex128)
    coeff_power = np.empty(n_terms + 1, dtype=np.complex128)
    coeff_log[0] = 0.0
    coeff_power[0] = 0.0
    harmonic = 0.0
    # series_scale(n) = (-1)^n (k^2/4)^n / (n!)^2, by the stable recurrence
    # series_scale(n) = series_scale(n-1) * (-(k^2/4) / n^2), which avoids
    # forming factorial(n) as an unbounded integer at high orders.
    series_scale = np.complex128(1.0)
    for n in range(1, n_terms + 1):
        harmonic += 1.0 / n
        series_scale = series_scale * (-(k * k / 4.0) / (n * n))
        coeff_log[n] = -series_scale / (2.0 * np.pi)
        coeff_power[n] = series_scale * (
            (harmonic - (log_k_half + _EULER_GAMMA)) / (2.0 * np.pi) + 0.25j
        )
    return c0, coeff_log, coeff_power


def _coefficients_3d(k: complex, n_terms: int):
    """Per-order coefficients of ``r^{n-1}`` for ``G_k - G_0`` in 3D,
    computed by the stable recurrence ``c_n = c_{n-1} * (i k) / n`` to
    avoid factorial overflow at high orders."""
    k = np.complex128(k)
    coefficients = np.empty(n_terms, dtype=np.complex128)
    value = np.complex128(1.0 / (4.0 * np.pi))
    for n in range(1, n_terms + 1):
        value = value * (1j * k) / n
        coefficients[n - 1] = value
    return coefficients

# }}}


# {{{ truncation certificate

def _exp_clipped(x: float) -> float:
    if x > 700.0:
        return float("inf")
    if x < -745.0:
        return 0.0
    return float(np.exp(x))


def _log_weight_max_log(radius: float, power: int) -> float:
    """log of max over 0 < r <= radius of r**power * |log(r)|.

    The stationary point of ``r**p |log r|`` on (0, 1) sits at
    ``r* = exp(-1/p)``; it contributes only when it lies inside the
    interval, i.e. when ``radius >= r*``.  Otherwise the function is
    increasing on (0, radius] and the boundary value is the maximum.
    """
    log_abs_log_radius = (
        float(np.log(abs(np.log(radius)))) if radius != 1.0 else -np.inf
    )
    boundary = power * float(np.log(radius)) + log_abs_log_radius
    if power * float(np.log(radius)) < -1.0:
        # radius < exp(-1/power): the interior extremum is inaccessible
        return boundary
    interior = -float(np.log(power)) - 1.0
    return max(interior, boundary)


def _tail_majorant(dim: int, k: complex, radius: float, n_terms: int) -> float:
    """Upper bound for the absolute sum of omitted series terms
    (orders ``n > n_terms``) on ``0 < r <= radius``, computed in log space
    to survive very large orders."""
    dim = _require_dimension(dim)
    from math import lgamma

    k_abs = float(np.abs(np.complex128(k)))
    if k_abs == 0.0:
        # Every omitted series term carries a positive power of k, so the
        # tail vanishes identically (the 3D zero-parameter kernels reduce
        # exactly to Laplace; the 2D series is rejected upstream).
        return 0.0
    log_k_half_gamma = float(
        np.abs(np.log(0.5 * np.complex128(k)) + _EULER_GAMMA)
    )
    log_radius = float(np.log(radius))
    log_two_pi = float(np.log(2.0 * np.pi))
    total = 0.0
    converged = False
    for n in range(n_terms + 1, n_terms + 400):
        if dim == 2:
            log_scale = 2.0 * n * float(np.log(k_abs / 2.0)) - 2.0 * lgamma(
                n + 1.0
            )
            harmonic = float(np.log(n) + 1.0)
            term = _exp_clipped(
                log_scale + _log_weight_max_log(radius, 2 * n) - log_two_pi
            ) + _exp_clipped(
                log_scale
                + 2.0 * n * log_radius
                + float(
                    np.log(
                        (harmonic + log_k_half_gamma) / (2.0 * np.pi) + 0.25
                    )
                )
            )
        else:
            term = _exp_clipped(
                n * float(np.log(k_abs))
                - lgamma(n + 1.0)
                + (n - 1) * log_radius
                - float(np.log(4.0 * np.pi))
            )
        total += term
        # Per-term growth ratio of the majorant: the 2D terms scale like
        # ((k R / 2) / (n+1))^2 up to a slowly varying factor (bounded by 2),
        # the 3D terms like (k R) / (n+1).  Once rho < 1/2, the uncomputed
        # remainder beyond this term is bounded by the geometric closure
        # term * rho / (1 - rho).
        if dim == 2:
            rho = 2.0 * (k_abs * radius / 2.0 / (n + 1.0)) ** 2
        else:
            rho = k_abs * radius / (n + 1.0)
        decaying = rho < 0.5
        # An exactly-zero computed term means the log-magnitude fell below
        # the float64 underflow threshold; in the decaying regime every
        # later term is smaller still, so this also counts as convergence
        # (covers tiny nonzero parameters whose whole tail underflows).
        if decaying and (
            term == 0.0 or term < 1e-30 * max(total, 1.0e-300)
        ):
            # close the series with the geometric remainder bound plus a
            # sub-underflow allowance for the exactly-zero case
            total += term * rho / (1.0 - rho) + 1.0e-300
            converged = True
            break
    if not converged:
        # The window did not reach numerically negligible terms, so the
        # partial sum is not a majorant; report an uncertifiable bound.
        return float("inf")
    return total


def choose_truncation_order(
    dim: int,
    k: complex,
    radius: float,
    tolerance: float,
    max_terms: int = 60,
):
    """Smallest series order whose tail majorant is below ``tolerance``.

    Returns ``(n_terms, tail_bound)``; raises :class:`RKETruncationError` if
    ``max_terms`` is not enough."""
    dim = _require_dimension(dim)
    max_terms = _require_integer("max_terms", max_terms, minimum=1)
    if dim == 2 and float(np.abs(np.complex128(k))) == 0.0:
        raise ValueError(
            "the 2D series is undefined at k = 0 (its coefficients contain "
            "log(k/2)); build the Laplace table directly"
        )
    bound = float("inf")
    for n_terms in range(1, max_terms + 1):
        bound = _tail_majorant(dim, k, radius, n_terms)
        if bound <= tolerance:
            return n_terms, bound
    raise RKETruncationError(
        f"cannot certify tolerance {tolerance:g} within {max_terms} series "
        f"terms (last tail bound {bound:g}); increase max_terms or relax "
        "the tolerance"
    )


def _clear_inherited_build_routing(table) -> None:
    """Drop the DuffyRadial build routing a deep copy inherited from its base.

    Both assemblers build their result as ``copy.deepcopy(base)`` and then
    overwrite the table data, so the copy would otherwise keep the base
    Laplace channel table's ``build_routing`` -- ``"batched"`` or
    ``"scalar-fallback"`` -- and a reloaded assembled table would claim its
    values came from Duffy quadrature that never touched them.  No
    :data:`~volumential.nearfield_potential_table.DUFFY_BUILD_ROUTINGS` value
    describes an RKE assembly, so clear the pair to *None*; the serializer
    then omits both keys and
    :func:`volumential.opcounters.direct_build_routing` reports ``unknown``,
    which is the honest answer for data no DuffyRadial builder produced.
    """
    for attr in ("build_routing", "build_fallback_reason"):
        if hasattr(table, attr):
            setattr(table, attr, None)


def _basis_l1_norms(table) -> np.ndarray:
    """Numerically computed L1 norms of the source basis functions on the
    source box, used to convert kernel-space bounds to entry bounds."""
    from itertools import product as iproduct

    q = int(table.quad_order)
    extent = float(table.source_box_extent)
    dim = int(table.dim)

    # 1D interpolation nodes of the tensor-product basis
    axis_nodes = np.array(
        [p[dim - 1] for p in table.q_points[:q]], dtype=np.float64
    )

    # The 1D Lagrange basis polynomial L_i has degree q - 1 and vanishes
    # exactly at the other interpolation nodes, so |L_i| is polynomial on
    # each subinterval between consecutive breakpoints {0, nodes, extent}.
    # Per-segment Gauss of order q integrates each piece exactly, making
    # the L1 norms exact up to roundoff rather than an estimate.
    breakpoints = np.unique(
        np.concatenate(([0.0], np.sort(axis_nodes), [extent]))
    )
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(q)

    def lagrange_values(i, x):
        num = np.ones_like(x)
        den = 1.0
        for j in range(q):
            if j == i:
                continue
            num *= x - axis_nodes[j]
            den *= axis_nodes[i] - axis_nodes[j]
        return num / den

    one_d_l1 = np.zeros(q, dtype=np.float64)
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        if right <= left:
            continue
        x = 0.5 * (right - left) * (gl_nodes + 1.0) + left
        w = 0.5 * (right - left) * gl_weights
        for i in range(q):
            one_d_l1[i] += abs(float(np.sum(w * lagrange_values(i, x))))
    norms = np.empty(q**dim, dtype=np.float64)
    for flat, multi in enumerate(iproduct(range(q), repeat=dim)):
        acc = 1.0
        for axis_index in multi:
            acc *= one_d_l1[axis_index]
        norms[flat] = acc
    return norms

# }}}


# {{{ channel table acquisition

def _channel_specs(dim: int, n_terms: int):
    """(term label, kernel factory kwargs) for every needed channel."""
    dim = _require_dimension(dim)
    specs = [("laplace", None)]
    if dim == 2:
        specs.append(("constant", 0))
        for n in range(1, n_terms + 1):
            specs.append((f"power:{2 * n}", 2 * n))
            specs.append((f"power_log:{2 * n}", 2 * n))
    else:
        for n in range(1, n_terms + 1):
            if n - 1 == 0:
                specs.append(("constant", 0))
            else:
                specs.append((f"power:{n - 1}", n - 1))
    return specs


def _channel_kernel(dim: int, label: str):
    dim = _require_dimension(dim)
    from volumential.expansion_wrangler_fpnd import (
        _RadialPowerKernel,
        _RadialPowerLogKernel,
    )
    from volumential.table_manager import ConstantKernel

    if label == "laplace":
        return "Laplace", None
    if label == "constant":
        return "Constant", ConstantKernel(dim)
    kind, power = label.split(":", 1)
    power = int(power)
    # Use the same kernel-type namespace as the expansion wrangler's
    # auto-built split-term tables, so caches interoperate and the labels
    # follow the established custom-sumpy-kernel convention (which carries
    # no scale-reuse type: the manager records these tables as
    # non-rescalable, exactly like direct fixed-parameter tables).
    if kind == "power":
        return f"SplitPower{power}", _RadialPowerKernel(dim, power)
    if kind == "power_log":
        return f"SplitPowerLog{power}", _RadialPowerLogKernel(dim, power)
    raise ValueError(f"unknown channel label: {label}")


def _get_channel_tables(
    queue,
    cache_path,
    dim,
    q_order,
    source_box_level,
    root_extent,
    labels,
    build_config,
    force_recompute,
):
    dim, q_order = _require_dim_q_order(dim, q_order)
    source_box_level = _require_integer(
        "source_box_level", source_box_level, minimum=0
    )
    from volumential.table_manager import NearFieldInteractionTableManager

    tables = {}
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=float(root_extent), queue=queue
    ) as table_manager:
        for label in labels:
            kernel_type, sumpy_knl = _channel_kernel(dim, label)
            kwargs = {
                "source_box_level": source_box_level,
                "force_recompute": bool(force_recompute),
                "queue": queue,
                "build_config": build_config,
            }
            if sumpy_knl is not None:
                kwargs["sumpy_knl"] = sumpy_knl
            table, _ = table_manager.get_table(
                dim, kernel_type, q_order, **kwargs
            )
            tables[label] = table
    return tables

# }}}


def assemble_parameterized_table(
    queue,
    cache_path,
    dim: int,
    kernel_type: str,
    q_order: int,
    parameter: float,
    *,
    source_box_level: int = 0,
    root_extent: float = 2.0,
    tolerance: float = 1.0e-12,
    max_terms: int = 60,
    max_condition: float = 1.0e6,
    build_config=None,
    force_channel_recompute: bool = False,
):
    """Assemble a fixed-parameter near-field table from canonical channels.

    :arg kernel_type: ``"Helmholtz"`` (outgoing, ``k = parameter``) or
        ``"Yukawa"`` (``lam = parameter``).
    :arg tolerance: certified absolute kernel-space truncation tolerance on
        the near-field separation region; the certificate also reports the
        induced per-entry bound.
    :arg force_channel_recompute: rebuild the channel tables even when the
        cache already holds them.  Cached channels are otherwise returned
        as stored, regardless of the ``build_config`` requested on this
        call, so pass ``True`` after tightening the quadrature
        configuration for an existing cache.
    :returns: ``(table, certificate)`` where ``table`` is a
        :class:`~volumential.nearfield_potential_table.\
NearFieldInteractionTable`
        equivalent to a direct fixed-parameter build up to the certified
        bound, and ``certificate`` is a dict of the assembly provenance.
    """
    dim, q_order = _require_dim_q_order(dim, q_order)
    source_box_level = _require_integer(
        "source_box_level", source_box_level, minimum=0
    )
    max_condition = _require_finite_positive("max_condition", max_condition)
    if kernel_type == "Helmholtz":
        k = np.complex128(float(parameter))
        result_dtype = np.complex128
    elif kernel_type == "Yukawa":
        k = np.complex128(1j * float(parameter))
        result_dtype = np.float64
    else:
        raise NotImplementedError(
            "RKE table assembly supports Helmholtz and Yukawa"
        )

    if float(parameter) == 0.0 and dim == 2:
        raise ValueError(
            "zero-parameter 2D assembly is not defined (the 2D series "
            "coefficients contain log(k/2)); build the Laplace table directly"
        )

    # Conservative bound for the near-field separation radius.  The adaptive
    # List 1 gallery contains center offsets up to 1.5 source-box extents
    # with target boxes up to twice the source size, so a source point and a
    # target point can be up to 1.5 + 1 + 0.5 = 3 extents apart per axis.
    box_extent = float(root_extent) * 0.5**source_box_level
    # The recombination evaluates channel integrals (which scale like
    # extent**(power + dim)) against coefficients (which scale like
    # k**power) as separate float64 factors, so extreme physical extents
    # can overflow or underflow one factor even when the dimensionless
    # product is moderate.  The table infrastructure's canonical convention
    # is an O(1) root extent; enforce it rather than certify garbage.
    if not 1.0e-3 <= box_extent <= 1.0e3:
        raise ValueError(
            f"source-box extent {box_extent:g} is outside the supported "
            "O(1) range for certified float64 recombination; rescale the "
            "problem to an O(1) root extent (the canonical table convention)"
        )
    radius = 3.0 * np.sqrt(dim) * box_extent

    n_terms, tail_bound = choose_truncation_order(
        dim, k, radius, tolerance, max_terms=max_terms
    )

    specs = _channel_specs(dim, n_terms)
    labels = [label for label, _ in specs]
    tables = _get_channel_tables(
        queue,
        cache_path,
        dim,
        q_order,
        source_box_level,
        root_extent,
        labels,
        build_config,
        force_channel_recompute,
    )

    base = tables["laplace"]
    entry_ids = np.asarray(base.get_reduced_entry_ids(), dtype=np.int64)

    def reduced(label):
        # Address every channel through full entry IDs so dense-stored
        # (legacy cache) and compact symmetry-reduced channels interoperate;
        # the accessor raises if a channel lacks any canonical entry.
        return np.asarray(
            tables[label].get_entry_data_for_full_indices(entry_ids)
        )

    contributions = [reduced("laplace").astype(np.complex128)]
    if dim == 2:
        c0, coeff_log, coeff_power = _coefficients_2d(k, n_terms)
        contributions.append(c0 * reduced("constant"))
        for n in range(1, n_terms + 1):
            contributions.append(coeff_log[n] * reduced(f"power_log:{2 * n}"))
            contributions.append(coeff_power[n] * reduced(f"power:{2 * n}"))
    else:
        coeffs = _coefficients_3d(k, n_terms)
        for n in range(1, n_terms + 1):
            label = "constant" if n - 1 == 0 else f"power:{n - 1}"
            contributions.append(coeffs[n - 1] * reduced(label))

    values = np.zeros_like(contributions[0])
    abs_accumulation = np.zeros(values.shape, dtype=np.float64)
    peak_contribution = 0.0
    sum_of_channel_maxima = 0.0
    for index, contribution in enumerate(contributions):
        if not np.all(np.isfinite(contribution)):
            raise RuntimeError(
                f"channel contribution {index} is not finite; the channel "
                "integrals or coefficients over/underflowed float64 "
                "(rescale the problem to an O(1) root extent)"
            )
        channel_max = float(np.max(np.abs(contribution)))
        peak_contribution = max(peak_contribution, channel_max)
        sum_of_channel_maxima += channel_max
        abs_accumulation += np.abs(contribution)
        values = values + contribution

    max_imag = float(np.max(np.abs(values.imag))) if values.size else 0.0
    if result_dtype == np.float64:
        reference_scale = max(float(np.max(np.abs(values.real))), 1e-300)
        if max_imag > 1.0e-10 * reference_scale:
            raise RuntimeError(
                "assembled Yukawa table has non-negligible imaginary part "
                f"({max_imag:g}); coefficient branch error suspected"
            )
        values = np.ascontiguousarray(values.real)

    result = copy.deepcopy(base)
    result.dtype = result_dtype
    # Do not inherit the Laplace channel's kernel identity: a fixed-parameter
    # Helmholtz/Yukawa table is not scale reusable, and the base table's
    # "log"/"inv_power" scale type would silently apply log-kernel or
    # homogeneous rescaling. ``None`` matches what a direct fixed-parameter
    # build records and makes scaling queries fail loudly.
    result.kernel_type = None
    for identity_attr in ("integral_knl", "kernel_func", "kernel_type_cached"):
        if hasattr(result, identity_attr):
            setattr(result, identity_attr, None)
    _clear_inherited_build_routing(result)
    result._data = None
    result.set_reduced_table_data(entry_ids, values.astype(result_dtype))
    result.is_built = True

    basis_l1 = _basis_l1_norms(base)
    entry_bound = tail_bound * float(np.max(basis_l1))
    max_entry = float(np.max(np.abs(values))) if values.size else 0.0

    # Floating-point recombination accounting: the assembly is an
    # alternating series, and its accuracy degrades when intermediate
    # contributions dwarf the result (large local parameter |k| * radius).
    # The bound covers, entrywise via sum_j |x_j|:
    # - the summation itself: standard forward bound gamma_m,
    # - the one rounding of each coefficient-times-channel product, and
    # - coefficient evaluation, modeled as 32 ulps for the seed (one
    #   complex log/exp under a few-ulp libm) plus 8 ulps per recurrence
    #   step: the order-n coefficient accumulates rounding from all n
    #   updates, so the allowance grows with the retained order; the
    #   modeled constants are recorded in the certificate.
    eps = float(np.finfo(np.float64).eps)
    m_terms = len(contributions)
    gamma_m = (m_terms + 1) * eps / max(1.0 - (m_terms + 1) * eps, 0.5)
    coefficient_eval_ulps = 32.0 + 8.0 * n_terms
    max_abs_sum = float(np.max(abs_accumulation)) if values.size else 0.0
    cancellation_bound = (gamma_m + coefficient_eval_ulps * eps) * max_abs_sum
    condition = max_abs_sum / max(max_entry, 1e-300)
    if condition > max_condition:
        raise RKEConditioningError(
            "RKE table assembly is ill-conditioned for this parameter and "
            f"box size (condition {condition:.3e} > {max_condition:.1e}); "
            "the local parameter |k| times the separation radius "
            f"({float(np.abs(k)) * radius:.2f}) is too large for certified "
            "float64 recombination. Assemble at a finer source-box level or "
            "use a direct batched build."
        )

    certificate = {
        "kernel_type": kernel_type,
        "parameter": float(parameter),
        "dim": int(dim),
        "q_order": int(q_order),
        "source_box_level": int(source_box_level),
        "n_series_terms": int(n_terms),
        "channel_count": len(labels),
        "separation_radius_bound": float(radius),
        "kernel_tail_majorant": float(tail_bound),
        "basis_l1_max": float(np.max(basis_l1)),
        "certified_entry_bound": float(entry_bound),
        "certified_entry_bound_relative": (
            float(entry_bound / max_entry) if max_entry > 0 else 0.0
        ),
        "peak_contribution": float(peak_contribution),
        "condition_number": float(condition),
        # Channel quadrature error is owned by ``build_config`` (exactly as
        # for direct builds) and is NOT part of the certified bounds below.
        # If every channel is built with relative quadrature error at most
        # eps_q (relative to its own max entry), the induced entry error of
        # the assembly is at most eps_q times this amplification factor:
        # the sum over channels of each coefficient-weighted channel's own
        # maximum (channel maxima need not share an entry, so this is the
        # honest amplification, larger than the entrywise abs-sum).
        "quadrature_amplification_bound": float(sum_of_channel_maxima),
        "coefficient_eval_ulps_model": float(coefficient_eval_ulps),
        "coefficient_eval_ulps_model_form": "32 + 8 * n_series_terms",
        "cancellation_entry_bound": float(cancellation_bound),
        "certified_entry_bound_total": float(entry_bound + cancellation_bound),
        "certified_entry_bound_total_relative": (
            float((entry_bound + cancellation_bound) / max_entry)
            if max_entry > 0
            else 0.0
        ),
        "assembled_max_abs_imag": max_imag,
    }
    return result, certificate


# {{{ windowed channels

def _generalized_exponential_integral_cf(order, x):
    """Evaluate ``E_order(x)`` for ``x > 1`` by a stable continued fraction."""
    order = float(order)
    x = np.asarray(x, dtype=np.float64)
    tiny = 1.0e-300
    b = x + order
    c = np.full_like(x, 1.0 / tiny)
    d = 1.0 / b
    value = d.copy()

    for iteration in range(1, 257):
        numerator = -iteration * (iteration + order - 1.0)
        b = b + 2.0
        d = b + numerator * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = b + numerator / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        update = d * c
        value = value * update
        if np.all(np.abs(update - 1.0) <= 8.0 * np.finfo(float).eps):
            return np.exp(-x) * value

    raise RuntimeError(
        "generalized exponential-integral continued fraction did not converge"
    )


def _windowed_channel_profile_impl(dim, m, window_scale, *, normalized):
    dim = _require_dimension(dim)
    import scipy.special as sps

    m = _require_integer("m", m, minimum=0)
    t_w = _require_finite_positive("window_scale", window_scale)

    if dim == 2:
        scale = 0.5 if normalized else 0.5 * t_w**m

        def profile(r):
            r_arr = np.asarray(r, dtype=np.float64)
            x = np.maximum(r_arr * r_arr / (4.0 * t_w), 1.0e-300)
            opcounters.add(opcounters.PROFILE_NODES, "psi_m", x.size)
            opcounters.add(opcounters.SPECIAL_EVALS, "expn", x.size)
            value = sps.expn(m + 1, x)
            value = scale * value
            if np.isscalar(r) or r_arr.ndim == 0:
                return float(value)
            return value

    else:
        if normalized:
            scale = t_w**-0.5 / (2.0 * np.sqrt(np.pi))
        else:
            scale = t_w ** (m - 0.5) / (2.0 * np.sqrt(np.pi))

        def profile(r):
            r_arr = np.asarray(r, dtype=np.float64)
            x = np.maximum(r_arr * r_arr / (4.0 * t_w), 1.0e-300)
            opcounters.add(opcounters.PROFILE_NODES, "psi_m", x.size)
            if m == 0:
                opcounters.add(opcounters.SPECIAL_EVALS, "erfc", x.size)
                sqrt_x = np.sqrt(x)
                value = np.sqrt(np.pi) * sps.erfc(sqrt_x) / sqrt_x
            else:
                large_x = x > 1.0
                opcounters.add(opcounters.SPECIAL_EVALS, "erfc", x.size)
                opcounters.add(opcounters.SPECIAL_EVALS, "exp", x.size)
                opcounters.add(
                    opcounters.SPECIAL_EVALS,
                    "expn_cf",
                    int(np.count_nonzero(large_x)),
                )
                recurrence_x = np.where(large_x, 0.0, x)
                sqrt_x = np.sqrt(np.maximum(recurrence_x, 1.0e-300))
                value = np.sqrt(np.pi) * sps.erfc(sqrt_x) / sqrt_x
                ex = np.exp(-recurrence_x)
                for order in range(1, m + 1):
                    value = (
                        ex - recurrence_x * value
                    ) / (order - 0.5)
                if np.any(large_x):
                    value = np.asarray(value)
                    value[large_x] = _generalized_exponential_integral_cf(
                        m + 0.5, x[large_x]
                    )
            value = scale * value
            if np.isscalar(r) or r_arr.ndim == 0:
                return float(value)
            return value

    return profile


def windowed_channel_profile(dim, m, window_scale):
    """Radial profile ``chi_m(r)`` of the physical windowed channel.

    Defined by the generalized exponential-integral recurrences

    - 2D: ``y_0(x) = E_1(x)``, ``y_m = (exp(-x) - x y_{m-1}) / m``,
      ``chi_m = (1/2) t_w^m y_m``
    - 3D: ``z_0(x) = sqrt(pi) erfc(sqrt(x)) / sqrt(x)``,
      ``z_m = (exp(-x) - x z_{m-1}) / (m - 1/2)``,
      ``chi_m = t_w^{m-1/2} z_m / (2 sqrt(pi))``

    with ``x = r^2 / (4 t_w)`` and ``t_w = window_scale``; equivalently
    ``chi_m = (1/2)(r^2/4)^m Gamma(-m, x)`` in 2D and
    ``chi_m = (r^2/4)^{m-1/2} Gamma(1/2-m, x) / (2 sqrt(pi))`` in 3D.
    ``chi_0`` in 3D is exactly the Ewald short-range kernel
    ``erfc(r / (2 sqrt(t_w))) / r``.
    Evaluation uses SciPy's stable integer-order ``expn`` in 2D and a direct
    continued fraction for the large-``x`` half-integer integral in 3D,
    avoiding cancellation in the displayed upward recurrences.

    :returns: a vectorized callable ``chi(r)`` accepting scalars or arrays.
    """
    return _windowed_channel_profile_impl(
        dim, m, window_scale, normalized=False
    )


def _normalized_windowed_channel_profile(dim, m, window_scale):
    """Numerically balanced channel ``psi_m = chi_m / t_w**m``."""
    return _windowed_channel_profile_impl(
        dim, m, window_scale, normalized=True
    )


def _windowed_channel_kernel_func(dim, m, window_scale, *, normalized=False):
    """Coordinate-space wrapper with the scalar Duffy path's
    ``kernel_func(x, y[, z])`` signature around a windowed profile."""
    if normalized:
        profile = _normalized_windowed_channel_profile(dim, m, window_scale)
    else:
        profile = windowed_channel_profile(dim, m, window_scale)

    def kernel_func(x, y=None, z=None):
        coords = [c for c in (x, y, z) if c is not None][:dim]
        r_squared = sum(np.asarray(c, dtype=np.float64) ** 2 for c in coords)
        return profile(np.sqrt(r_squared))

    return kernel_func


def _require_o1_box_extent(box_extent):
    """Refuse box extents outside the O(1) range the float64 recombination
    (and the extent-scaled degeneracy tests of the Duffy geometry) are
    certified for — the same contract the classical assembler enforces."""
    if not 1.0e-3 <= float(box_extent) <= 1.0e3:
        raise ValueError(
            f"source-box extent {float(box_extent):g} is outside the "
            "supported O(1) range for certified float64 recombination; "
            "rescale the problem to an O(1) root extent (the canonical "
            "table convention)"
        )


def _selected_decay_root(zeta):
    """The module's square-root branch contract for squared frequencies:
    the decaying root on the Yukawa ray (``Re > 0``) and the outgoing
    lower-half-plane limit on the negative (Helmholtz) ray (``-i k``).
    This is a pointwise branch selection, shared by the remainder profile
    and the damped-kernel entry point so they can never disagree."""
    decay = np.complex128(np.sqrt(np.complex128(zeta)))
    if decay.real < 0.0 or (decay.real == 0.0 and decay.imag > 0.0):
        decay = -decay
    return decay


def _windowed_coefficients(scaled_zeta, p_star):
    """Balanced coefficients ``(-scaled_zeta)^m / m!``."""
    coefficients = np.empty(p_star, dtype=np.complex128)
    value = np.complex128(1.0)
    for m in range(p_star):
        coefficients[m] = value
        value = value * (-complex(scaled_zeta)) / (m + 1)
    return coefficients


def windowed_remainder_profile(dim, zeta, kernel_radial, window_scale, p_star):
    """Radial profile of the exact windowed remainder

    ``R(r) = G(r) - pref * sum_{m < p_star}
    ((-zeta*t_w)^m / m!) psi_m(r)``

    with ``psi_m = chi_m / t_w**m``, ``pref = 1/(2 pi)`` in 2D, and
    ``pref = 1/(4 pi)`` in 3D.  This is algebraically identical to the
    physical-channel sum while keeping both factors representable.  It is
    exactly the smooth part the windowed assembler integrates, so tests can
    certify the coefficient/prefactor/sign conventions queue-free by probing
    ``R`` directly (boundedness and germ cancellation as ``r -> 0``).
    At ``r = 0`` the callable returns the analytic removable limit for the
    decaying Yukawa / outgoing Helmholtz branch instead of evaluating the two
    singular terms separately.  ``zeta = 0`` is rejected because this API does
    not define a zero-frequency kernel normalization or its origin limit.

    :returns: a vectorized callable ``R(r)`` (complex-valued).
    """
    dim = _require_dimension(dim)
    p_star = _require_integer("p_star", p_star, minimum=1)
    window_scale = _require_finite_positive("window_scale", window_scale)
    zeta = complex(zeta)
    if not np.isfinite(zeta.real) or not np.isfinite(zeta.imag):
        raise ValueError("zeta must be finite")
    if zeta == 0.0:
        raise ValueError("zeta must be nonzero")
    prefactor = 1.0 / (2.0 * np.pi) if dim == 2 else 1.0 / (4.0 * np.pi)
    coefficients = _windowed_coefficients(zeta * window_scale, p_star)
    profiles = [
        _normalized_windowed_channel_profile(dim, m, window_scale)
        for m in range(p_star)
    ]

    # Select the decaying Yukawa / outgoing Helmholtz square root.  On the
    # negative real axis the latter is the lower-half-plane limit, -i*k.
    decay = _selected_decay_root(zeta)
    if dim == 2:
        origin_value = (
            -np.log(decay * np.sqrt(window_scale)) - 0.5 * _EULER_GAMMA
        ) / (2.0 * np.pi)
        for m in range(1, p_star):
            origin_value -= prefactor * coefficients[m] / (2.0 * m)
    else:
        origin_value = (
            -decay + 1.0 / np.sqrt(np.pi * window_scale)
        ) / (4.0 * np.pi)
        for m in range(1, p_star):
            profile_at_origin = 1.0 / (
                2.0 * np.sqrt(np.pi * window_scale) * (m - 0.5)
            )
            origin_value -= (
                prefactor * coefficients[m] * profile_at_origin
            )

    def remainder_radial(r):
        r_array = np.asarray(r)
        at_origin = r_array == 0
        safe_r = np.where(at_origin, 1.0, r_array)
        channel_sum = coefficients[0] * profiles[0](safe_r)
        for m in range(1, p_star):
            channel_sum = channel_sum + coefficients[m] * profiles[m](safe_r)
        result = kernel_radial(safe_r) - prefactor * channel_sum
        result = np.where(at_origin, origin_value, result)
        if np.isscalar(r) or r_array.ndim == 0:
            return np.asarray(result).item()
        return result

    return remainder_radial


def _tensor_product_gauss_points(q_order, dim, extent):
    """Tensor-product Gauss-Legendre points on ``[0, extent]^dim`` in the
    lexicographic ordering the table constructor produces from the mesh
    generator (queue-free equivalent of ``mg.make_uniform_cubic_grid`` at
    level 1 followed by the constructor's mapping and dictionary sort)."""
    dim, q_order = _require_dim_q_order(dim, q_order)
    nodes = np.polynomial.legendre.leggauss(q_order)[0]
    axis_points = 0.5 * float(extent) * (nodes + 1.0)
    grids = np.meshgrid(*([axis_points] * dim), indexing="ij")
    return np.ascontiguousarray(
        np.stack([g.reshape(-1) for g in grids], axis=-1)
    )


def _windowed_channel_skeleton(
    dim, q_order, source_box_level, root_extent, window_theta, m,
):
    """Channel table shell (geometry and symmetry metadata, no data) at the
    source-box extent the table manager would use for this level."""
    dim, q_order = _require_dim_q_order(dim, q_order)
    source_box_level = _require_integer(
        "source_box_level", source_box_level, minimum=0
    )
    m = _require_integer("m", m, minimum=0)
    window_theta = _require_finite_positive("window_theta", window_theta)

    from volumential.nearfield_potential_table import NearFieldInteractionTable

    box_extent = float(root_extent) * 0.5**source_box_level
    window_scale = (box_extent / window_theta) ** 2
    table = NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        kernel_func=_windowed_channel_kernel_func(
            dim, m, window_scale, normalized=True
        ),
        kernel_type=None,
        sumpy_kernel=None,
        source_box_extent=box_extent,
        dtype=np.float64,
        progress_bar=False,
        precomputed_q_points=_tensor_product_gauss_points(
            q_order, dim, box_extent
        ),
    )
    table.source_box_level = source_box_level
    return table


def _reduced_entry_groups(table):
    """Reduced entry IDs grouped by ``(case, target)`` with the positions and
    source-mode indices of each member, mirroring the scalar Duffy builder's
    entry enumeration."""
    entry_ids = np.asarray(
        table._get_invariant_entry_info()["entry_ids"], dtype=np.int64
    )
    n_pairs = np.int64(table.n_pairs)
    n_q_points = np.int64(table.n_q_points)
    case_indices = (entry_ids // n_pairs).astype(np.int64)
    pair_ids = entry_ids % n_pairs
    source_modes = (pair_ids // n_q_points).astype(np.int64)
    target_points = (pair_ids % n_q_points).astype(np.int64)

    groups = {}
    for position in range(len(entry_ids)):
        key = (int(case_indices[position]), int(target_points[position]))
        groups.setdefault(key, []).append(
            (position, int(source_modes[position]))
        )
    return entry_ids, groups


def _axis_basis_values(table, coords, xi, bary_weights):
    """Per-axis Lagrange basis values (list over axes of list over basis
    indices) at the given coordinate arrays."""
    from volumential.lagrange import evaluate_lagrange_basis_1d

    q = int(table.quad_order)
    if q == 1:
        return [
            [np.ones_like(np.asarray(coord))] for coord in coords
        ]
    return [
        [
            evaluate_lagrange_basis_1d(xi, k, coord, weights=bary_weights)
            for k in range(q)
        ]
        for coord in coords
    ]


def _channel_profile_values(radial_profile, radius):
    """Channel profile values on a Duffy node block, with any node that
    rounds onto the singular point (``r == 0``) contributing zero.

    The Duffy Jacobian vanishes faster than the germ diverges (2D:
    ``rho log(1/rho)``, 3D: ``rho^2 / rho``), so the limit contribution of a
    target-coincident node is exactly zero — whereas
    :func:`windowed_channel_profile` would report the huge finite value its
    small-``x`` clip produces there (``sqrt(pi/1e-300)`` in 3D).
    """
    positive = radius > 0.0
    if positive.all():
        return radial_profile(radius)
    return np.where(
        positive, radial_profile(np.where(positive, radius, 1.0)), 0.0
    )


def _duffy_channel_entry_values(
    table, radial_profile, regular_order, radial_order,
):
    """Reduced table entries of a radial kernel via the scalar DuffyRadial
    node set, evaluated vectorized.

    Uses exactly the quadrature nodes and weights of
    ``compute_table_entry_duffy_radial`` with
    ``radial_rule="tanh-sinh-fast"`` (2D: four Duffy triangles with
    Gauss-Legendre in the angle; 3D: sign-octant/permutation cones with a
    tensor Gauss-Legendre tail), so it agrees with the scalar builder to
    roundoff while evaluating the radial profile on whole node blocks.
    """
    regular_order = _require_integer(
        "chan_regular_order", regular_order, minimum=1
    )
    radial_order = _require_integer(
        "chan_radial_order", radial_order, minimum=1
    )
    import scipy.special as sps

    import volumential.singular_integral_2d as squad
    from volumential.lagrange import barycentric_lagrange_weights

    dim = int(table.dim)
    q = int(table.quad_order)
    extent = float(table.source_box_extent)
    mode_axes = np.asarray(table._get_all_mode_axes(), dtype=np.int64)
    if q > 1:
        xi = np.asarray(
            [p[dim - 1] for p in table.q_points[:q]], dtype=np.float64
        )
        bary_weights = barycentric_lagrange_weights(xi)
    else:
        xi = None
        bary_weights = None

    entry_ids, groups = _reduced_entry_groups(table)
    values = np.zeros(len(entry_ids), dtype=np.float64)

    rho_nodes, rho_weights = squad._duffy_radial_nodes_weights(
        "tanh-sinh-fast", radial_order, 50
    )

    if dim == 2:
        th_nodes, th_weights = sps.roots_legendre(regular_order)
        theta = 0.25 * np.pi * (th_nodes + 1.0)
        w_theta = 0.25 * np.pi * th_weights
        cos_sq = np.cos(theta) ** 2
        sin_sq = np.sin(theta) ** 2
        duffy_factor = np.outer(np.sin(2.0 * theta), rho_nodes)
        weight_grid = np.outer(w_theta, rho_weights)
        corners = [
            np.array([0.0, 0.0]),
            np.array([extent, 0.0]),
            np.array([extent, extent]),
            np.array([0.0, extent]),
        ]

        for (case_index, target_index), members in groups.items():
            target = np.asarray(
                table.find_target_point(target_index, case_index),
                dtype=np.float64,
            )
            singular = np.clip(target, 0.0, extent)
            # source-to-target displacement is accumulated from the *offset*
            # of the Duffy origin, never by differencing absolute box
            # coordinates: for a self-interaction target the offset is
            # exactly zero, and a node closer to the target than one ulp of
            # the coordinate would otherwise be absorbed into it and report
            # r = 0 (see _channel_profile_values)
            base_offset = singular - target
            entry_acc = np.zeros(len(members), dtype=np.float64)
            for corner_index in range(4):
                edge1 = corners[corner_index] - singular
                edge2 = corners[(corner_index + 1) % 4] - singular
                det = edge1[0] * edge2[1] - edge1[1] * edge2[0]
                # degenerate-triangle test must scale with the box (det is
                # an area, ~ extent**2), or small extents would silently
                # drop every triangle
                if np.abs(det) < 1.0e-14 * extent * extent:
                    continue
                u = np.outer(cos_sq, rho_nodes)
                v = np.outer(sin_sq, rho_nodes)
                dx = base_offset[0] + u * edge1[0] + v * edge2[0]
                dy = base_offset[1] + u * edge1[1] + v * edge2[1]
                xx = target[0] + dx
                yy = target[1] + dy
                radius = np.hypot(dx, dy)
                opcounters.add(
                    opcounters.SINGULAR_NODES, "duffy_radial", radius.size
                )
                common = (
                    _channel_profile_values(radial_profile, radius)
                    * (det * duffy_factor)
                    * weight_grid
                )
                basis = _axis_basis_values(
                    table, (xx, yy), xi, bary_weights
                )
                for member_index, (_, source_mode) in enumerate(members):
                    a0, a1 = mode_axes[source_mode]
                    entry_acc[member_index] += float(
                        np.sum(common * basis[0][a0] * basis[1][a1])
                    )
            for member_index, (position, _) in enumerate(members):
                values[position] = entry_acc[member_index]
        return entry_ids, values

    from itertools import permutations
    from itertools import product as iproduct

    regular_nodes, regular_weights = squad._duffy_regular_nodes_weights(
        dim - 1, regular_order
    )
    regular_nodes = np.asarray(regular_nodes, dtype=np.float64)
    regular_weights = np.asarray(regular_weights, dtype=np.float64)
    n_tail = regular_nodes.shape[0]
    weight_grid = np.outer(regular_weights, rho_weights)
    tail0 = regular_nodes[:, 0:1]
    tail1 = regular_nodes[:, 1:2]
    rho_row = rho_nodes[np.newaxis, :]
    u_first = np.broadcast_to(rho_row, (n_tail, len(rho_nodes)))
    u_second = rho_row * tail0
    u_third = u_second * tail1
    jacobian_core = (rho_row**2) * tail0

    for (case_index, target_index), members in groups.items():
        target = np.asarray(
            table.find_target_point(target_index, case_index),
            dtype=np.float64,
        )
        singular = np.clip(target, 0.0, extent)
        # see the 2D branch: displacements are built from the Duffy origin's
        # offset, so a node nearer the target than one coordinate ulp still
        # carries its true (tiny) radius instead of being absorbed to r = 0
        base_offset = singular - target
        entry_acc = np.zeros(len(members), dtype=np.float64)
        for signs in iproduct((-1.0, 1.0), repeat=dim):
            lengths = np.array(
                [
                    singular[axis] if signs[axis] < 0
                    else extent - singular[axis]
                    for axis in range(dim)
                ],
                dtype=np.float64,
            )
            if np.any(lengths <= 0.0):
                continue
            box_scale = float(np.prod(lengths))
            for perm in permutations(range(dim)):
                u_by_axis = [None] * dim
                u_by_axis[perm[0]] = u_first
                u_by_axis[perm[1]] = u_second
                u_by_axis[perm[2]] = u_third
                offsets = [
                    base_offset[axis] + signs[axis] * lengths[axis]
                    * u_by_axis[axis]
                    for axis in range(dim)
                ]
                coords = [
                    target[axis] + offsets[axis] for axis in range(dim)
                ]
                radius = np.sqrt(
                    sum(offsets[axis] ** 2 for axis in range(dim))
                )
                opcounters.add(
                    opcounters.SINGULAR_NODES, "duffy_radial", radius.size
                )
                common = (
                    _channel_profile_values(radial_profile, radius)
                    * (box_scale * jacobian_core)
                    * weight_grid
                )
                basis = _axis_basis_values(
                    table, coords, xi, bary_weights
                )
                for member_index, (_, source_mode) in enumerate(members):
                    a0, a1, a2 = mode_axes[source_mode]
                    entry_acc[member_index] += float(
                        np.sum(
                            common
                            * basis[0][a0]
                            * basis[1][a1]
                            * basis[2][a2]
                        )
                    )
        for member_index, (position, _) in enumerate(members):
            values[position] = entry_acc[member_index]
    return entry_ids, values


def _resolve_channel_orders(dim, chan_regular_order, chan_radial_order):
    """Resolve ``None`` channel quadrature orders to the tested defaults.

    2D interpolation nodes near the box edge create high-aspect Duffy
    triangles whose angular convergence is slow (the same effect governs
    direct builds): angular order 48 holds the assembled 2D ``q = 3`` tables
    to about 2e-7 of a 64/91 reference, while order 20 drifts by about 1e-4
    on channel entries.  The 3D cone geometry is mild (order 14 already sits
    below 1e-7 relative drift), so order 20 is kept there.
    """
    dim = _require_dimension(dim)
    if chan_regular_order is None:
        chan_regular_order = 48 if dim == 2 else 20
    if chan_radial_order is None:
        chan_radial_order = 61
    return _validate_channel_orders(chan_regular_order, chan_radial_order)


# Duffy radial weights integrate the constant ``1`` over ``[0, 1]``, so the
# deviation of their sum from unity is a direct, order-independent measure of
# whether the requested order produced a usable rule.  Measured deviations of
# the ``tanh-sinh-fast`` set: order 3 (the builder's silent clamp target)
# 3.3e-5, order 5 4.1e-7, order 7 9.9e-9, order 15 1.2e-13, order 45 and
# above at roundoff.  The tolerance below therefore admits every order from 7
# up and rejects exactly the degenerate low end.
_CHANNEL_RADIAL_WEIGHT_TOL = 1.0e-8


def _validate_channel_orders(chan_regular_order, chan_radial_order):
    """Refuse channel quadrature orders the underlying node builders do not
    honour, instead of letting them degrade silently.

    Neither builder reports a bad request: ``scipy.special.p_roots`` accepts
    any positive order, and the ``tanh-sinh-fast`` radial rule passes its
    order through ``max(3, order)``, so 0, 1, 2 (or a negative order) all
    quietly become the same 7-node rule.  The radial order is validated by
    measurement against :data:`_CHANNEL_RADIAL_WEIGHT_TOL` rather than by a
    hard-coded list, so the rule — not this function — remains the
    authority on which orders are usable.
    """
    import volumential.singular_integral_2d as squad

    regular = _require_integer("chan_regular_order", chan_regular_order)
    radial = _require_integer("chan_radial_order", chan_radial_order)
    if regular < 2:
        raise ValueError(
            f"chan_regular_order {regular} is not a usable Gauss-Legendre "
            "order for the channel Duffy rule (need at least 2)"
        )
    if radial < 3:
        raise ValueError(
            f"chan_radial_order {radial} is below the tanh-sinh-fast node "
            "builder's silent max(3, order) clamp, so the requested order "
            "would be ignored; pass an order the rule actually honours"
        )
    rho_nodes, rho_weights = squad._duffy_radial_nodes_weights(
        "tanh-sinh-fast", radial, 50
    )
    weight_defect = abs(float(np.sum(rho_weights)) - 1.0)
    if (
        rho_nodes.size < 8
        or not np.all((rho_nodes > 0.0) & (rho_nodes < 1.0))
        or not np.all(np.isfinite(rho_weights))
        or weight_defect > _CHANNEL_RADIAL_WEIGHT_TOL
    ):
        raise ValueError(
            f"chan_radial_order {radial} yields a degenerate tanh-sinh-fast "
            f"rule ({rho_nodes.size} nodes, weight sum off unity by "
            f"{weight_defect:.3e} > {_CHANNEL_RADIAL_WEIGHT_TOL:.1e}); "
            "raise the radial order"
        )
    return regular, radial


def _channel_source_mode_sup(table):
    """Sup norm over the source box of the tensor-product source modes,
    ``max_i sup_y |phi_i(y)|``, bounded axis-wise (the tensor product of the
    per-axis sups dominates the sup of the product)."""
    from volumential.lagrange import (
        barycentric_lagrange_weights,
        evaluate_lagrange_basis_1d,
    )

    dim = int(table.dim)
    q = int(table.quad_order)
    if q == 1:
        return 1.0
    xi = np.asarray(
        [p[dim - 1] for p in table.q_points[:q]], dtype=np.float64
    )
    bary_weights = barycentric_lagrange_weights(xi)
    sample = np.linspace(0.0, float(table.source_box_extent), 1024)
    axis_sup = max(
        float(
            np.max(
                np.abs(
                    evaluate_lagrange_basis_1d(
                        xi, k, sample, weights=bary_weights
                    )
                )
            )
        )
        for k in range(q)
    )
    return axis_sup**dim


def _channel_entry_magnitude_bound(table, m, window_scale, safety=8.0):
    """Analytic upper bound on normalized ``|psi_m|`` table entries.

    A table entry is ``int_box psi_m(|x_t - y|) phi_i(y) dy``, so

    ``|entry| <= sup|phi_i| * int_{R^dim} psi_m(|z|) dz``

    and the total channel mass is elementary,

    ``int_{R^dim} psi_m = c_dim t_w / (m+1)``, ``c_2 = 2 pi``,
    ``c_3 = 4 pi``, because ``psi_m = chi_m/t_w^m``.

    (the ``m = 0`` cases are the Ewald identities ``int E_1(r^2/4t_w)/2 =
    2 pi t_w`` and ``int erfc(r/2 sqrt(t_w))/r = 4 pi t_w``; the recurrence
    carries the identity to every ``m``).  The bound is tight: without
    ``safety`` a built 3D ``q = 2`` table sits within a factor of three of
    it at every ``m``, so even the generous factor leaves a check that no
    quadrature error can trip but a blown-up build cannot survive.
    """
    dim = int(table.dim)
    mass_constant = 2.0 * np.pi if dim == 2 else 4.0 * np.pi
    channel_mass = mass_constant * float(window_scale) / (int(m) + 1)
    return float(safety) * _channel_source_mode_sup(table) * channel_mass


def _check_channel_table_values(table, values, m, window_scale):
    """Reject a channel table whose entries are not finite or exceed the
    analytic bound of :func:`_channel_entry_magnitude_bound`.

    Cheap (one pass over the reduced entries) and unconditional, so no build
    path — fresh, cached, or rebuilt — can hand back a poisoned
    channel table that the downstream condition estimate would rate as
    perfectly conditioned (a single blown-up channel dominates both the peak
    sum and the assembled maximum, so their ratio stays at 1).
    """
    values = np.asarray(values, dtype=np.float64)
    bound = _channel_entry_magnitude_bound(table, m, window_scale)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"windowed channel table m = {int(m)} has non-finite entries"
        )
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak > bound:
        raise RuntimeError(
            f"windowed channel table m = {int(m)} has entries of magnitude "
            f"{peak:.6e}, above the analytic bound {bound:.6e} for window "
            f"scale t_w = {float(window_scale):.6e}; the channel quadrature "
            "did not resolve the germ"
        )


def _windowed_channel_cache_file(
    cache_path,
    dim,
    q_order,
    source_box_level,
    root_extent,
    window_theta,
    m,
    chan_regular_order,
    chan_radial_order,
):
    dim, q_order = _require_dim_q_order(dim, q_order)
    source_box_level = _require_integer(
        "source_box_level", source_box_level, minimum=0
    )
    m = _require_integer("m", m, minimum=0)
    chan_regular_order = _require_integer(
        "chan_regular_order", chan_regular_order
    )
    chan_radial_order = _require_integer(
        "chan_radial_order", chan_radial_order
    )
    key = (
        ("cache_schema", _WINDOWED_CHANNEL_CACHE_SCHEMA),
        ("normalization", _WINDOWED_CHANNEL_NORMALIZATION),
        ("dim", dim),
        ("q_order", q_order),
        ("source_box_level", source_box_level),
        ("root_extent", repr(float(root_extent))),
        ("window_theta", repr(float(window_theta))),
        ("m", m),
        ("chan_regular_order", chan_regular_order),
        ("chan_radial_order", chan_radial_order),
        ("radial_rule", "tanh-sinh-fast"),
    )
    digest = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    directory = Path(str(cache_path) + ".windowed")
    filename = (
        f"channel-d{dim}-q{q_order}-l{source_box_level}"
        f"-m{m}-{digest}.npz"
    )
    return directory / filename, dict(key)


def _windowed_channel_payload_checksum(entry_ids, values):
    digest = hashlib.sha256()
    for label, array, dtype in (
        (b"entry_ids\0", entry_ids, "<i8"),
        (b"values\0", values, "<f8"),
    ):
        array = np.ascontiguousarray(array, dtype=dtype)
        digest.update(label)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def get_windowed_channel_table(
    cache_path,
    dim,
    q_order,
    m,
    *,
    source_box_level=0,
    root_extent=2.0,
    window_theta=16.0,
    chan_regular_order=None,
    chan_radial_order=None,
    force_recompute=False,
):
    """Build or load the normalized windowed channel table ``psi_m``.

    The channel family depends on the declaration ``window_theta`` (and the
    table geometry) only — never on any swept kernel parameter — and the
    cache key preserves that invariant.  Channel data is stored per channel
    in ``.npz`` files under ``str(cache_path) + ".windowed/"`` keyed by
    ``(schema, normalization, dim, q_order, source_box_level, root_extent,
    window_theta, m, chan_regular_order, chan_radial_order)``.  A load
    validates the full key, symmetry-reduced entry IDs, and a checksum over
    the IDs and values before accepting cached data.  Any unreadable, stale,
    or corrupted file is rebuilt and atomically replaced, with the discard
    reason logged as a warning on this module's logger.

    ``chan_regular_order`` / ``chan_radial_order`` default (via ``None``) to
    the tested per-dimension orders of :func:`_resolve_channel_orders`
    (48/61 in 2D, 20/61 in 3D); orders the underlying node builders would
    silently ignore are refused by :func:`_validate_channel_orders`.

    Every table handed back — freshly built or loaded from cache — passes
    :func:`_check_channel_table_values`, so a channel that violates the
    analytic entry bound can never reach the recombination (where a single
    blown-up channel would dominate both the peak sum and the assembled
    maximum, leaving the reported condition number at a reassuring 1.0).
    The private ``_windowed_cache_disposition`` attribute is ``"hit"`` only
    after a fully validated load and ``"rebuilt"`` after fresh or recovery
    construction.
    """
    dim, q_order = _require_dim_q_order(dim, q_order)
    source_box_level = _require_integer(
        "source_box_level", source_box_level, minimum=0
    )
    m = _require_integer("m", m, minimum=0)
    window_theta = _require_finite_positive("window_theta", window_theta)
    chan_regular_order, chan_radial_order = _resolve_channel_orders(
        dim, chan_regular_order, chan_radial_order
    )
    box_extent = float(root_extent) * 0.5**source_box_level
    _require_o1_box_extent(box_extent)
    window_scale = (box_extent / window_theta) ** 2
    table = _windowed_channel_skeleton(
        dim, q_order, source_box_level, root_extent, window_theta, m
    )
    cache_file, key = _windowed_channel_cache_file(
        cache_path,
        dim,
        q_order,
        source_box_level,
        root_extent,
        window_theta,
        m,
        chan_regular_order,
        chan_radial_order,
    )

    expected_entry_ids = np.asarray(
        table._get_invariant_entry_info()["entry_ids"], dtype=np.int64
    )

    if not force_recompute and cache_file.is_file():
        # A cache read must never wedge the assembly: a torn or corrupted
        # file (crash or concurrent writer mid-``np.savez``) or a stale
        # layout missing expected arrays falls through to a rebuild that
        # atomically overwrites the bad file.
        try:
            with np.load(cache_file, allow_pickle=False) as payload:
                stored_key = {
                    name: payload[f"key_{name}"].item() for name in key
                }
                stored_ids = np.asarray(
                    payload["entry_ids"], dtype=np.int64
                )
                stored_values = np.asarray(
                    payload["values"], dtype=np.float64
                )
                stored_checksum = str(payload["payload_checksum"].item())
        except Exception as exc:
            # never silent: a permanently unreadable cache file otherwise
            # looks exactly like a cache hit while paying a full rebuild on
            # every call
            logger.warning(
                "discarding unreadable windowed channel cache %s: %s: %s",
                cache_file, type(exc).__name__, exc,
            )
        else:
            # A cache written before the schema, checksum, or entry bound was
            # enforced is treated exactly like a torn file.  Shape checks come
            # before the table setter so malformed data cannot wedge later
            # calls on a permanently loadable bad file.
            try:
                if stored_key != key:
                    raise RuntimeError("stored cache key does not match")
                if not np.array_equal(stored_ids, expected_entry_ids):
                    raise RuntimeError("stored entry IDs do not match")
                if stored_values.shape != expected_entry_ids.shape:
                    raise RuntimeError(
                        f"stored value array of shape {stored_values.shape} "
                        f"does not match the {expected_entry_ids.shape} "
                        "reduced entries"
                    )
                computed_checksum = _windowed_channel_payload_checksum(
                    stored_ids, stored_values
                )
                if stored_checksum != computed_checksum:
                    raise RuntimeError("stored payload checksum does not match")
                _check_channel_table_values(
                    table, stored_values, m, window_scale
                )
            except RuntimeError as exc:
                logger.warning(
                    "rebuilding windowed channel cache %s: %s",
                    cache_file, exc,
                )
            else:
                table.set_reduced_table_data(
                    expected_entry_ids, stored_values
                )
                table.is_built = True
                table._windowed_cache_disposition = "hit"
                return table

    entry_ids, entry_values = _duffy_channel_entry_values(
        table,
        _normalized_windowed_channel_profile(dim, m, window_scale),
        chan_regular_order,
        chan_radial_order,
    )
    opcounters.add(
        opcounters.TABLE_ENTRIES, "windowed_channel", int(entry_values.size)
    )
    # ``entry_ids`` comes from the same memoized invariant-entry info on the
    # same table object as ``expected_entry_ids``, so an equality check here
    # could never fire; the meaningful validation is the load-path one above
    # (cached IDs against freshly computed ones).

    # Refuse (and never cache) a channel table that violates the analytic
    # entry bound: a poisoned channel is invisible to every downstream
    # certificate field, so it has to die here.
    _check_channel_table_values(table, entry_values, m, window_scale)

    # Publish atomically (write-then-rename) so a crash or a concurrent
    # writer can never leave a torn final file behind.
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_name(
        f"{cache_file.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with open(tmp_file, "wb") as stream:
            np.savez(
                stream,
                entry_ids=entry_ids,
                values=entry_values,
                payload_checksum=np.asarray(
                    _windowed_channel_payload_checksum(entry_ids, entry_values)
                ),
                **{
                    f"key_{name}": np.asarray(value)
                    for name, value in key.items()
                },
            )
        os.replace(tmp_file, cache_file)
    finally:
        tmp_file.unlink(missing_ok=True)
    table.set_reduced_table_data(entry_ids, entry_values)
    table.is_built = True
    table._windowed_cache_disposition = "rebuilt"
    return table


def _smooth_remainder_entry_values(
    table, entry_ids, remainder_radial, smooth_quad_order,
):
    """Entries ``integral over the source box of R(x_j - y) phi_i(y) dy``
    for the given full entry IDs, by a tensor-product Gauss-Legendre rule of
    ``smooth_quad_order`` points per axis on the source box, mirroring the
    case/target/mode geometry of ``compute_table_entry_duffy_radial``.
    Handles complex-valued ``R``."""
    n_nodes = _require_integer(
        "smooth_quad_order", smooth_quad_order, minimum=1
    )
    from volumential.lagrange import (
        barycentric_lagrange_weights,
        evaluate_lagrange_basis_1d,
    )

    dim = int(table.dim)
    q = int(table.quad_order)
    extent = float(table.source_box_extent)

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_nodes)
    axis_nodes = 0.5 * extent * (gl_nodes + 1.0)
    axis_weights = 0.5 * extent * gl_weights

    if q > 1:
        xi = np.asarray(
            [p[dim - 1] for p in table.q_points[:q]], dtype=np.float64
        )
        bary_weights = barycentric_lagrange_weights(xi)
        basis_matrix = np.stack(
            [
                evaluate_lagrange_basis_1d(
                    xi, k, axis_nodes, weights=bary_weights
                )
                for k in range(q)
            ]
        )
    else:
        basis_matrix = np.ones((1, n_nodes), dtype=np.float64)

    grids = np.meshgrid(*([axis_nodes] * dim), indexing="ij")
    weight_tensor = axis_weights
    for _ in range(dim - 1):
        weight_tensor = np.multiply.outer(weight_tensor, axis_weights)

    entry_ids = np.asarray(entry_ids, dtype=np.int64)
    n_pairs = np.int64(table.n_pairs)
    n_q_points = np.int64(table.n_q_points)
    case_indices = (entry_ids // n_pairs).astype(np.int64)
    pair_ids = entry_ids % n_pairs
    source_modes = (pair_ids // n_q_points).astype(np.int64)
    target_points = (pair_ids % n_q_points).astype(np.int64)
    mode_axes = np.asarray(table._get_all_mode_axes(), dtype=np.int64)

    groups = {}
    for position in range(len(entry_ids)):
        key = (int(case_indices[position]), int(target_points[position]))
        groups.setdefault(key, []).append(
            (position, int(source_modes[position]))
        )

    values = np.zeros(len(entry_ids), dtype=np.complex128)
    for (case_index, target_index), members in groups.items():
        target = np.asarray(
            table.find_target_point(target_index, case_index),
            dtype=np.float64,
        )
        radius = np.sqrt(
            sum((grids[axis] - target[axis]) ** 2 for axis in range(dim))
        )
        opcounters.add(opcounters.SMOOTH_NODES, "tensor_gauss", radius.size)
        weighted_remainder = (
            np.asarray(remainder_radial(radius)) * weight_tensor
        )
        if dim == 2:
            mode_table = basis_matrix @ weighted_remainder @ basis_matrix.T
        else:
            mode_table = np.einsum(
                "ai,bj,ck,ijk->abc",
                basis_matrix,
                basis_matrix,
                basis_matrix,
                weighted_remainder,
                optimize=True,
            )
        for position, source_mode in members:
            values[position] = mode_table[tuple(mode_axes[source_mode])]
    return values


def _assemble_windowed_for_zeta(
    cache_path,
    dim,
    q_order,
    zeta,
    kernel_radial,
    *,
    source_box_level=0,
    root_extent=2.0,
    window_theta=16.0,
    p_star=6,
    smooth_quad_order=None,
    max_condition=1.0e6,
    chan_regular_order=None,
    chan_radial_order=None,
    force_channel_recompute=False,
    result_dtype=np.complex128,
):
    """Windowed assembly at an explicit squared-frequency parameter ``zeta``
    and kernel radial profile; :func:`assemble_windowed_parameterized_table`
    wraps this with the standard Helmholtz/Yukawa identifications, and tests
    exercise complex ``zeta`` (damped waves) directly.  ``zeta`` must be
    nonzero because the explicit entry point does not define a zero-frequency
    kernel normalization or its origin limit."""
    dim, q_order = _require_dim_q_order(dim, q_order)
    source_box_level = _require_integer(
        "source_box_level", source_box_level, minimum=0
    )
    p_star = _require_integer("p_star", p_star, minimum=1)
    if smooth_quad_order is not None:
        smooth_quad_order = _require_integer(
            "smooth_quad_order", smooth_quad_order, minimum=1
        )
    window_theta = _require_finite_positive("window_theta", window_theta)
    max_condition = _require_finite_positive("max_condition", max_condition)
    zeta = complex(zeta)
    if not np.isfinite(zeta.real) or not np.isfinite(zeta.imag):
        raise ValueError("zeta must be finite")
    if zeta == 0.0:
        raise ValueError("zeta must be nonzero")
    chan_regular_order, chan_radial_order = _resolve_channel_orders(
        dim, chan_regular_order, chan_radial_order
    )

    box_extent = float(root_extent) * 0.5**source_box_level
    _require_o1_box_extent(box_extent)
    window_scale = (box_extent / window_theta) ** 2
    theta_abs = float(np.sqrt(abs(zeta)) * box_extent)

    # KB coverage contract: the declaration certifies the closed disk
    # |zeta| <= (Theta/b)^2, i.e. |zeta| t_w <= 1.  Outside it the
    # coefficients (theta/Theta)^{2m}/m! grow and the conditioning contract
    # no longer holds, so refuse rather than certify an uncovered point
    # (the real-parameter wrapper's theta guard is the same condition).
    if theta_abs > window_theta * (1.0 + WINDOW_COVERAGE_RELATIVE_TOLERANCE):
        raise RKEWindowCoverageError(
            f"local parameter |zeta|**0.5 * b = {theta_abs:g} exceeds the "
            f"declared window Theta = {window_theta:g}; the "
            "requested parameter is outside the declared coverage disk "
            "|zeta| <= (Theta/b)**2"
        )

    if smooth_quad_order is None:
        # Two resolution requirements: the kernel oscillation (theta/2 plus
        # margin) and the windowed channels inside R, whose transition
        # features live at scale sqrt(t_w) = b/Theta regardless of the
        # swept parameter.  Measured 2D convergence of assembled entries
        # (q=3, theta=1) against converged references: the window term
        # reaches ~1e-14 relative at about 1.25*Theta points per axis
        # (Theta=16: 24, Theta=32: 48, Theta=64: 80, Theta=128: ~160), so
        # 1.25*Theta + 8 carries margin at every declaration, not just the
        # default Theta = 16.
        smooth_quad_order = max(
            16,
            2 * q_order,
            int(np.ceil(theta_abs / 2.0)) + 16,
            int(np.ceil(1.25 * window_theta)) + 8,
        )
    # Gauss rules of the table's own order reproduce the self-interaction
    # target points, and any two odd-order Gauss rules share the interval
    # midpoint; either collision would sample the remainder at r = 0
    # (where the kernel diverges), so nudge the order clear of both.
    if smooth_quad_order == q_order or (
        smooth_quad_order % 2 == 1 and q_order % 2 == 1
    ):
        smooth_quad_order += 1

    channels = [
        get_windowed_channel_table(
            cache_path,
            dim,
            q_order,
            m,
            source_box_level=source_box_level,
            root_extent=root_extent,
            window_theta=window_theta,
            chan_regular_order=chan_regular_order,
            chan_radial_order=chan_radial_order,
            force_recompute=force_channel_recompute,
        )
        for m in range(p_star)
    ]
    base = channels[0]
    entry_ids = np.asarray(base.get_reduced_entry_ids(), dtype=np.int64)

    prefactor = 1.0 / (2.0 * np.pi) if dim == 2 else 1.0 / (4.0 * np.pi)
    coefficients = _windowed_coefficients(zeta * window_scale, p_star)
    remainder_radial = windowed_remainder_profile(
        dim, zeta, kernel_radial, window_scale, p_star
    )

    remainder_values = _smooth_remainder_entry_values(
        base, entry_ids, remainder_radial, smooth_quad_order
    )

    values = remainder_values.astype(np.complex128, copy=True)
    per_channel_peak = []
    for m in range(p_star):
        contribution = (
            prefactor
            * coefficients[m]
            * np.asarray(
                channels[m].get_entry_data_for_full_indices(entry_ids)
            )
        )
        if not np.all(np.isfinite(contribution)):
            raise RuntimeError(
                f"windowed channel contribution {m} is not finite"
            )
        per_channel_peak.append(float(np.max(np.abs(contribution))))
        values = values + contribution
    if not np.all(np.isfinite(values)):
        raise RuntimeError("windowed assembly produced non-finite entries")
    # Recombination cost: the remainder entry seeds the accumulator and each
    # of the p_star channels contributes one complex fused multiply-add per
    # entry; the coefficients come from a p_star-step recurrence.
    opcounters.add(
        opcounters.RECOMBINATION_FLOPS,
        "channel_fma",
        p_star * int(entry_ids.size),
    )
    opcounters.add(
        opcounters.RECOMBINATION_FLOPS, "coefficient_recurrence", p_star
    )

    remainder_peak = (
        float(np.max(np.abs(remainder_values)))
        if remainder_values.size
        else 0.0
    )
    max_entry = float(np.max(np.abs(values))) if values.size else 0.0
    condition = (sum(per_channel_peak) + remainder_peak) / max(
        max_entry, 1e-300
    )
    if condition > max_condition:
        raise RKEWindowConditioningError(
            "ill-conditioned windowed assembly (condition "
            f"{condition:.3e} > {max_condition:.1e}); the parameter "
            "lies outside the conditioning contract of the declared window"
        )

    coefficient_bound = 0.0
    magnitude = 1.0
    for m in range(p_star):
        coefficient_bound = max(coefficient_bound, magnitude)
        magnitude = magnitude * abs(zeta) * window_scale / (m + 1)

    max_imag = float(np.max(np.abs(values.imag))) if values.size else 0.0
    if np.dtype(result_dtype) == np.dtype(np.float64):
        reference_scale = max(float(np.max(np.abs(values.real))), 1e-300)
        if max_imag > 1.0e-10 * reference_scale:
            raise RuntimeError(
                "assembled windowed Yukawa table has non-negligible "
                f"imaginary part ({max_imag:g})"
            )
        values = np.ascontiguousarray(values.real)

    result = copy.deepcopy(base)
    result.dtype = np.dtype(result_dtype).type
    result.kernel_type = None
    for identity_attr in ("integral_knl", "kernel_func", "kernel_type_cached"):
        if hasattr(result, identity_attr):
            setattr(result, identity_attr, None)
    _clear_inherited_build_routing(result)
    result._data = None
    result.set_reduced_table_data(entry_ids, values.astype(result_dtype))
    result.is_built = True

    certificate = {
        "kernel_type": None,
        "parameter": None,
        "zeta": repr(zeta),
        "zeta_real": float(zeta.real),
        "zeta_imag": float(zeta.imag),
        "theta": theta_abs,
        "dim": dim,
        "q_order": q_order,
        "source_box_level": source_box_level,
        "window_theta": window_theta,
        "window_scale_t_w": float(window_scale),
        "p_star": p_star,
        "smooth_quad_order": smooth_quad_order,
        "coefficient_bound": float(coefficient_bound),
        "per_channel_peak": per_channel_peak,
        "remainder_peak": remainder_peak,
        "condition_number": float(condition),
        # The remainder is the exact kernel-minus-channels difference, so no
        # series truncation exists anywhere in the windowed assembly; this
        # zero is structural, not an estimate.
        "truncation_tail_bound": 0.0,
        "assembled_max_abs_imag": max_imag,
        "channel_quadrature_orders": {
            "regular": int(chan_regular_order),
            "radial": int(chan_radial_order),
        },
    }
    return result, certificate


def assemble_windowed_parameterized_table(
    cache_path,
    dim: int,
    kernel_type: str,
    q_order: int,
    parameter: float,
    *,
    source_box_level: int = 0,
    root_extent: float = 2.0,
    window_theta: float = 16.0,
    p_star: int = 6,
    smooth_quad_order=None,
    max_condition: float = 1.0e6,
    chan_regular_order: int | None = None,
    chan_radial_order: int | None = None,
    force_channel_recompute: bool = False,
):
    """Assemble a fixed-parameter near-field table from windowed channels.

    :arg kernel_type: ``"Helmholtz"`` (outgoing, ``k = parameter``) or
        ``"Yukawa"`` (``lam = parameter``).
    :arg window_theta: the declaration ``Theta``; the local parameter
        ``theta = parameter * b`` (with ``b`` the source-box extent) must
        not exceed it.
    :arg p_star: number of tabulated windowed channels.
    :arg smooth_quad_order: Gauss-Legendre points per axis for the smooth
        remainder; defaults to ``max(16, 2 q_order, ceil(theta/2) + 16,
        ceil(1.25 window_theta) + 8)`` — a Nyquist-style term for the kernel
        oscillation plus a declaration-scaled term for the windowed-channel
        transition features (width ``b/Theta``) inside the remainder.
    :arg chan_regular_order: angular (2D) / tail (3D) Gauss order of the
        one-time singular channel quadrature; ``None`` selects the tested
        per-dimension default (48 in 2D — the high-aspect Duffy triangles of
        edge-adjacent 2D interpolation nodes converge slowly in this order —
        and 20 in 3D).  ``chan_radial_order`` (radial tanh-sinh order)
        defaults to 61 in both dimensions.
    :returns: ``(table, certificate)`` with ``table`` a
        :class:`~volumential.nearfield_potential_table.\
NearFieldInteractionTable`
        (complex128 for Helmholtz, float64 for Yukawa) whose kernel identity
        is nulled exactly like the classical assembler's, and
        ``certificate`` a dict of the assembly provenance.
    """
    dim, q_order = _require_dim_q_order(dim, q_order)
    source_box_level = _require_integer(
        "source_box_level", source_box_level, minimum=0
    )
    window_theta = _require_finite_positive("window_theta", window_theta)
    parameter = float(parameter)
    if not parameter > 0.0:
        raise ValueError("windowed assembly requires a positive parameter")

    box_extent = float(root_extent) * 0.5**source_box_level
    theta = parameter * box_extent
    if theta > window_theta * (1.0 + WINDOW_COVERAGE_RELATIVE_TOLERANCE):
        raise RKEWindowCoverageError(
            f"local parameter theta = {theta:g} exceeds the declared window "
            f"Theta = {window_theta:g}; the requested parameter is "
            "outside the declaration"
        )

    import scipy.special as sps

    if kernel_type == "Yukawa":
        zeta = complex(parameter * parameter)
        result_dtype = np.float64
        if dim == 2:

            def kernel_radial(r):
                r = np.asarray(r)
                opcounters.add(opcounters.KERNEL_EVALS, "k0", r.size)
                opcounters.add(opcounters.SPECIAL_EVALS, "k0", r.size)
                return sps.k0(parameter * r) / (2.0 * np.pi)

        else:

            def kernel_radial(r):
                r = np.asarray(r)
                opcounters.add(opcounters.KERNEL_EVALS, "exp", r.size)
                opcounters.add(opcounters.SPECIAL_EVALS, "exp", r.size)
                return np.exp(-parameter * r) / (4.0 * np.pi * r)

    elif kernel_type == "Helmholtz":
        zeta = complex(-(parameter * parameter))
        result_dtype = np.complex128
        if dim == 2:

            def kernel_radial(r):
                r = np.asarray(r)
                opcounters.add(opcounters.KERNEL_EVALS, "hankel1", r.size)
                opcounters.add(opcounters.SPECIAL_EVALS, "hankel1", r.size)
                return 0.25j * sps.hankel1(0, parameter * r)

        else:

            def kernel_radial(r):
                r = np.asarray(r)
                opcounters.add(opcounters.KERNEL_EVALS, "exp", r.size)
                opcounters.add(opcounters.SPECIAL_EVALS, "exp", r.size)
                return np.exp(1j * parameter * r) / (4.0 * np.pi * r)

    else:
        raise NotImplementedError(
            "windowed RKE assembly supports Helmholtz and Yukawa"
        )

    table, certificate = _assemble_windowed_for_zeta(
        cache_path,
        dim,
        q_order,
        zeta,
        kernel_radial,
        source_box_level=source_box_level,
        root_extent=root_extent,
        window_theta=window_theta,
        p_star=p_star,
        smooth_quad_order=smooth_quad_order,
        max_condition=max_condition,
        chan_regular_order=chan_regular_order,
        chan_radial_order=chan_radial_order,
        force_channel_recompute=force_channel_recompute,
        result_dtype=result_dtype,
    )
    certificate["kernel_type"] = kernel_type
    certificate["parameter"] = parameter
    certificate["theta"] = theta
    return table, certificate


def damped_kernel_radial(dim, zeta):
    """Radial kernel profile for a complex squared frequency ``zeta``,
    with the module's square-root branch contract.

    The selected root ``mu = sqrt(zeta)`` is the decaying branch
    (``Re mu > 0``), continued to the outgoing lower-half-plane limit
    ``mu = -i k`` on the negative real axis, exactly as in
    :func:`windowed_remainder_profile` (both call the same selector).  The
    kernel is

    - 2D: ``K_0(mu r) / (2 pi)``, which on the negative ray equals the
      outgoing Helmholtz kernel ``(i/4) H_0^(1)(k r)`` through the exact
      continuation ``K_0(-i z) = (i pi / 2) H_0^(1)(z)``;
    - 3D: ``exp(-mu r) / (4 pi r)``.

    ``zeta`` must be finite and nonzero (this API does not define a
    zero-frequency kernel normalization).

    :returns: a vectorized complex-valued callable ``g(r)``.
    """
    dim = _require_dimension(dim)
    import scipy.special as sps

    zeta = complex(zeta)
    if not np.isfinite(zeta.real) or not np.isfinite(zeta.imag):
        raise ValueError("zeta must be finite")
    if zeta == 0.0:
        raise ValueError("zeta must be nonzero")
    decay = _selected_decay_root(zeta)

    if dim == 2:

        def kernel_radial(r):
            r = np.asarray(r)
            opcounters.add(opcounters.KERNEL_EVALS, "kv0_complex", r.size)
            opcounters.add(opcounters.SPECIAL_EVALS, "kv0_complex", r.size)
            return sps.kv(
                0, decay * np.asarray(r, dtype=np.complex128)
            ) / (2.0 * np.pi)

    else:

        def kernel_radial(r):
            r = np.asarray(r)
            opcounters.add(opcounters.KERNEL_EVALS, "exp_complex", r.size)
            opcounters.add(opcounters.SPECIAL_EVALS, "exp_complex", r.size)
            return np.exp(-decay * r) / (4.0 * np.pi * r)

    return kernel_radial


def assemble_windowed_damped_table(
    cache_path,
    dim: int,
    q_order: int,
    zeta,
    *,
    source_box_level: int = 0,
    root_extent: float = 2.0,
    window_theta: float = 16.0,
    p_star: int = 6,
    smooth_quad_order=None,
    max_condition: float = 1.0e6,
    chan_regular_order: int | None = None,
    chan_radial_order: int | None = None,
    force_channel_recompute: bool = False,
):
    """Assemble a fixed-parameter table at a damped complex frequency.

    ``zeta`` is the complex squared frequency; the supported coverage is the
    punctured closed disk ``0 < |zeta| <= (Theta / b)**2`` (``b`` the
    source-box extent), uniformly over the phase — the conditioning
    certificate depends on ``|zeta|`` only.  The kernel profile is
    :func:`damped_kernel_radial`, so the branch convention is pointwise
    consistent with the windowed remainder: the Yukawa ray (``zeta > 0``)
    reproduces :func:`assemble_windowed_parameterized_table` with
    ``kernel_type="Yukawa"`` and the negative ray (``zeta < 0``) the
    outgoing Helmholtz assembly, both through the identical channel family
    (the channels are real and depend only on the declaration ``Theta``).
    ``zeta = 0`` is rejected.

    :returns: ``(table, certificate)`` with a complex128 table; the
        certificate additionally records ``zeta_phase_fraction``
        (``arg(zeta) / pi`` in ``(-1, 1]``).
    """
    kernel_radial = damped_kernel_radial(dim, zeta)
    table, certificate = _assemble_windowed_for_zeta(
        cache_path,
        dim,
        q_order,
        complex(zeta),
        kernel_radial,
        source_box_level=source_box_level,
        root_extent=root_extent,
        window_theta=window_theta,
        p_star=p_star,
        smooth_quad_order=smooth_quad_order,
        max_condition=max_condition,
        chan_regular_order=chan_regular_order,
        chan_radial_order=chan_radial_order,
        force_channel_recompute=force_channel_recompute,
        result_dtype=np.complex128,
    )
    certificate["kernel_type"] = "Damped"
    certificate["zeta_phase_fraction"] = float(
        np.angle(complex(zeta)) / np.pi
    )
    return table, certificate

# }}}
