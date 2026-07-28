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
"""

from __future__ import annotations

import copy

import numpy as np

__all__ = [
    "assemble_parameterized_table",
    "choose_truncation_order",
]


_EULER_GAMMA = np.euler_gamma


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

    Returns ``(n_terms, tail_bound)``; raises if ``max_terms`` is not
    enough."""
    if dim not in (2, 3):
        raise NotImplementedError(
            "certified truncation supports only the 2D and 3D kernel series"
        )
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
    raise ValueError(
        f"cannot certify tolerance {tolerance:g} within {max_terms} series "
        f"terms (last tail bound {bound:g}); increase max_terms or relax "
        "the tolerance"
    )


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
    from volumential.table_manager import NearFieldInteractionTableManager

    tables = {}
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=float(root_extent), queue=queue
    ) as table_manager:
        for label in labels:
            kernel_type, sumpy_knl = _channel_kernel(dim, label)
            kwargs = {
                "source_box_level": int(source_box_level),
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
    if dim not in (2, 3):
        raise NotImplementedError("RKE table assembly supports 2D and 3D")
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
    box_extent = float(root_extent) * 0.5 ** int(source_box_level)
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
        raise RuntimeError(
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
