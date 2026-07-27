#!/usr/bin/env python3
"""Keller--Segel parameter-continuation application benchmark.

Free-space parabolic--elliptic Keller--Segel model with screened
chemoattractant, on a matched mass pair bracketing the classical unscreened
``8 pi`` reference:

    rho_t = Lap rho - div(rho grad c),     (-Lap + alpha) c = rho.

Discretization: uniform box tree with tensor Legendre-Gauss nodes.  Per step,

* the chemoattractant and its gradient come from one fixed-``lambda_alpha``
  Yukawa volume-potential pass with three output kernels (``Yukawa``,
  ``Yukawa-Dx``, ``Yukawa-Dy``), ``lambda_alpha = sqrt(alpha)``;
* the transport divergence ``div(rho grad c)`` is evaluated by a conservative
  tensor-product Gauss-DG derivative with one Rusanov flux per shared face;
* the semi-implicit diffusion step ``(I - dt Lap) rho = rhs`` is a Yukawa
  volume potential with the varying parameter ``lambda = 1/sqrt(dt)``,
  ``rho = (1/dt) Y_lambda * rhs``.

The varying-``lambda`` solve runs through both strategies every step: the
direct strategy builds (and caches) one fixed-``lambda`` table per distinct
ladder value, and the RKE strategy reuses one fixed channel family with
online coefficients.  The state advances with the direct solution; the
per-step RKE-direct mismatch is recorded.  Time steps are CFL-adaptive with
interior ``lambda`` values quantized on a geometric ladder (ratio ``2^(1/8)``)
so the direct strategy's table reuse is well defined. Endpoint-adjusted steps
are admitted only when both resulting intervals satisfy the resolved regime.

Regime consistency: the resolved-parameter bound ``theta = lambda h <= theta_max``
is a lower bound on ``dt``; when the advection CFL requires a smaller step,
the run stops and reports ``theta_regime_exit`` rather than silently
accepting the step. Above-reference runs additionally stop before blow-up at
a density-growth threshold.

This is in part a timing benchmark: full mode must run on a quiet host.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from split_parameter_sweep import (  # noqa: E402
    _build_path,
    _capture_table_get_timings,
    _clear_sqlite_cache,
    _coords_host,
    _get_laplace_2d_table,
    _select_opencl_device,
    _split_channel_build_config,
    _split_smooth_quad_order,
    _summarize_table_get_timings,
    _yukawa_reference_build_config,
)

STEP_FIELDS = (
    "case_id",
    "mode",
    "direct_only",
    "step",
    "time",
    "dt",
    "lam",
    "theta",
    "u_max",
    "rho_max",
    "rho_min",
    "mass",
    "mass_drift_rel",
    "second_moment",
    "second_moment_per_mass",
    "core_mass_fraction",
    "boundary_mass_fraction",
    "boundary_trace_rel_peak",
    "negative_mass_fraction",
    "centroid_radius",
    "half_mass_radius",
    "transport_mass_change_rel",
    "limiter_correction_rel",
    "cumulative_limiter_correction_rel",
    "terminal_step",
    "endpoint_adjusted",
    "admissible",
    "new_lambda",
    "strategy_order",
    "chemo_solve_s",
    "direct_table_build_s",
    "direct_wrangler_build_s",
    "direct_solve_s",
    "rke_wrangler_build_s",
    "rke_solve_s",
    "rke_vs_direct_weighted_rel_l2",
    "cumulative_direct_strategy_s",
    "cumulative_rke_strategy_s",
    "strategy_cost_gap_direct_minus_rke_s",
)

SUMMARY_FIELDS = (
    "case_id",
    "mode",
    "direct_only",
    "mass_factor",
    "mass",
    "reference_mass",
    "reference_model",
    "regime",
    "q_order",
    "nlevels",
    "n_targets",
    "fmm_order",
    "split_order",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
    "alpha",
    "lambda_alpha",
    "initial_profile",
    "profile_scale",
    "cfl",
    "theta_max",
    "root_extent",
    "leaf_level",
    "leaf_box_extent",
    "cutoff_inner_radius",
    "cutoff_outer_radius",
    "dt_max",
    "lambda_ladder_ratio",
    "rke_beta_mode",
    "requested_t_end",
    "max_steps",
    "blowup_factor",
    "n_steps",
    "final_time",
    "stop_reason",
    "rho_max_initial",
    "rho_max_final",
    "rho_max_ratio",
    "rho_min_final",
    "mass_drift_rel",
    "second_moment_ratio",
    "core_mass_fraction_initial",
    "core_mass_fraction_final",
    "max_boundary_mass_fraction",
    "max_boundary_trace_rel_peak",
    "max_negative_mass_fraction",
    "max_centroid_radius",
    "min_half_mass_radius",
    "cumulative_limiter_correction_rel",
    "admissible",
    "trend_criterion_pass",
    "pair_outcome_pass",
    "pair_moment_ratio_separation",
    "max_theta_seen",
    "lambda_min",
    "lambda_max",
    "n_distinct_lambdas",
    "rke_channel_build_s",
    "direct_table_build_total_s",
    "chemo_solve_total_s",
    "direct_strategy_total_s",
    "rke_strategy_total_s",
    "strategy_crossings_json",
    "rke_first_cheaper_step",
    "direct_first_cheaper_after_rke_step",
    "final_strategy_cost_gap_direct_minus_rke_s",
    "max_rke_vs_direct_weighted_rel_l2",
    "radial_gradient_tangential_rel_l2",
    "radial_gradient_outward_rel_l2",
    "radial_gradient_preflight_pass",
    "strategy_cost_definition",
)

UNSCREENED_REFERENCE_MASS = 8.0 * math.pi
DEFAULT_LAMBDA_LADDER_RATIO = 2.0 ** 0.125
RADIAL_PREFLIGHT_MAX_REL_L2 = 5.0e-3


# {{{ initial data and local differentiation

def _smooth_cutoff(radius, inner_radius, outer_radius):
    result = np.ones_like(radius)
    result[radius >= outer_radius] = 0.0
    transition = (radius > inner_radius) & (radius < outer_radius)
    scaled = (radius[transition] - inner_radius) / (
        outer_radius - inner_radius
    )
    rise = np.exp(-1.0 / scaled)
    fall = np.exp(-1.0 / (1.0 - scaled))
    result[transition] = fall / (rise + fall)
    return result


def _initial_density(
        coords, weights, mass, profile, profile_scale, cutoff_inner_radius,
        cutoff_outer_radius):
    r_sq = coords[0] ** 2 + coords[1] ** 2
    if profile == "critical":
        profile_values = 8.0 * profile_scale**2 / (profile_scale**2 + r_sq)**2
    elif profile == "gaussian":
        profile_values = np.exp(-r_sq / (2.0 * profile_scale**2))
    else:
        raise ValueError(f"unknown initial profile: {profile}")

    profile_values *= _smooth_cutoff(
        np.sqrt(r_sq), cutoff_inner_radius, cutoff_outer_radius
    )
    return mass * profile_values / float(np.sum(weights * profile_values))


def _differentiation_matrix(q_order):
    """Spectral differentiation matrix on the 1D Legendre-Gauss nodes."""
    from modepy import LegendreGaussQuadrature
    from numpy.polynomial import legendre as leg

    nodes = np.asarray(
        LegendreGaussQuadrature(q_order - 1, force_dim_axis=True).nodes
    ).reshape(-1)
    vander = leg.legvander(nodes, q_order - 1)
    dvander = np.empty_like(vander)
    for k in range(q_order):
        coefficients = np.zeros(q_order)
        coefficients[k] = 1.0
        dvander[:, k] = leg.legval(nodes, leg.legder(coefficients))
    return dvander @ np.linalg.inv(vander)


class ConservativeDGTransport:
    """Conservative tensor-product Gauss-DG transport on a uniform box tree.

    Node layout follows ``QuadratureOnBoxTree.get_q_points``: leaf boxes in
    order, and within each box the ``q x q`` tensor grid with ``ij``
    indexing (x-index slowest).
    """

    def __init__(self, q_order, coords, box_extent, root_extent):
        from modepy import LegendreGaussQuadrature
        from numpy.polynomial import legendre as leg

        self.q = q_order
        self.n_boxes = coords.shape[1] // q_order**2
        self.scale = 2.0 / box_extent
        self.diff = _differentiation_matrix(q_order)
        rule = LegendreGaussQuadrature(q_order - 1, force_dim_axis=True)
        nodes = np.asarray(rule.nodes).reshape(-1)
        weights = np.asarray(rule.weights).reshape(-1)
        vander = leg.legvander(nodes, q_order - 1)
        inverse_vander = np.linalg.inv(vander)
        self.face_minus = (
            leg.legvander(np.array([-1.0]), q_order - 1) @ inverse_vander
        ).reshape(-1)
        self.face_plus = (
            leg.legvander(np.array([1.0]), q_order - 1) @ inverse_vander
        ).reshape(-1)
        self.correction_minus = self.face_minus / weights
        self.correction_plus = self.face_plus / weights

        reference_gaps = np.diff(nodes)
        cross_box_gap = 2.0 * (1.0 - nodes[-1])
        self.minimum_node_gap = 0.5 * box_extent * min(
            float(np.min(reference_gaps)), float(cross_box_gap)
        )

        centers = coords.reshape(2, self.n_boxes, q_order, q_order).mean(
            axis=(2, 3)
        )
        nside = int(round(math.sqrt(self.n_boxes)))
        if nside**2 != self.n_boxes:
            raise ValueError("DG transport requires a uniform square box grid")
        lower = -0.5 * root_extent
        indices = np.rint((centers - lower) / box_extent - 0.5).astype(int)
        grid = np.full((nside, nside), -1, dtype=int)
        for ibox, (ix, iy) in enumerate(indices.T):
            if not (0 <= ix < nside and 0 <= iy < nside):
                raise ValueError("box center lies outside the uniform grid")
            grid[ix, iy] = ibox
        if np.any(grid < 0):
            raise ValueError("could not recover the uniform box connectivity")
        self.left = np.array([
            grid[ix - 1, iy] if ix else -1 for ix, iy in indices.T
        ])
        self.right = np.array([
            grid[ix + 1, iy] if ix + 1 < nside else -1
            for ix, iy in indices.T
        ])
        self.bottom = np.array([
            grid[ix, iy - 1] if iy else -1 for ix, iy in indices.T
        ])
        self.top = np.array([
            grid[ix, iy + 1] if iy + 1 < nside else -1
            for ix, iy in indices.T
        ])

    def _tensor(self, values):
        return np.asarray(values).reshape(self.n_boxes, self.q, self.q)

    def _x_faces(self, values):
        tensor = self._tensor(values)
        return (
            np.einsum("i,nij->nj", self.face_minus, tensor),
            np.einsum("i,nij->nj", self.face_plus, tensor),
        )

    def _y_faces(self, values):
        tensor = self._tensor(values)
        return (
            np.einsum("j,nij->ni", self.face_minus, tensor),
            np.einsum("j,nij->ni", self.face_plus, tensor),
        )

    @staticmethod
    def _rusanov(rho_left, velocity_left, rho_right, velocity_right):
        speed = np.maximum(np.abs(velocity_left), np.abs(velocity_right))
        return 0.5 * (
            rho_left * velocity_left + rho_right * velocity_right
            - speed * (rho_right - rho_left)
        )

    def divergence(self, rho, velocity_x, velocity_y):
        rho = np.asarray(rho)
        flux_x = rho * velocity_x
        flux_y = rho * velocity_y
        rho_x_minus, rho_x_plus = self._x_faces(rho)
        ux_minus, ux_plus = self._x_faces(velocity_x)
        fx_minus, fx_plus = self._x_faces(flux_x)
        rho_y_minus, rho_y_plus = self._y_faces(rho)
        uy_minus, uy_plus = self._y_faces(velocity_y)
        fy_minus, fy_plus = self._y_faces(flux_y)

        numerical_x_minus = np.empty_like(fx_minus)
        numerical_x_plus = np.empty_like(fx_plus)
        numerical_y_minus = np.empty_like(fy_minus)
        numerical_y_plus = np.empty_like(fy_plus)
        for ibox in range(self.n_boxes):
            left = self.left[ibox]
            if left < 0:
                numerical_x_minus[ibox] = self._rusanov(
                    0.0, ux_minus[ibox], rho_x_minus[ibox], ux_minus[ibox]
                )
            else:
                numerical_x_minus[ibox] = self._rusanov(
                    rho_x_plus[left], ux_plus[left],
                    rho_x_minus[ibox], ux_minus[ibox],
                )
            right = self.right[ibox]
            if right < 0:
                numerical_x_plus[ibox] = self._rusanov(
                    rho_x_plus[ibox], ux_plus[ibox], 0.0, ux_plus[ibox]
                )
            else:
                numerical_x_plus[ibox] = self._rusanov(
                    rho_x_plus[ibox], ux_plus[ibox],
                    rho_x_minus[right], ux_minus[right],
                )

            bottom = self.bottom[ibox]
            if bottom < 0:
                numerical_y_minus[ibox] = self._rusanov(
                    0.0, uy_minus[ibox], rho_y_minus[ibox], uy_minus[ibox]
                )
            else:
                numerical_y_minus[ibox] = self._rusanov(
                    rho_y_plus[bottom], uy_plus[bottom],
                    rho_y_minus[ibox], uy_minus[ibox],
                )
            top = self.top[ibox]
            if top < 0:
                numerical_y_plus[ibox] = self._rusanov(
                    rho_y_plus[ibox], uy_plus[ibox], 0.0, uy_plus[ibox]
                )
            else:
                numerical_y_plus[ibox] = self._rusanov(
                    rho_y_plus[ibox], uy_plus[ibox],
                    rho_y_minus[top], uy_minus[top],
                )

        derivative_x = np.einsum(
            "ib,nbj->nij", self.diff, self._tensor(flux_x)
        )
        derivative_x += np.einsum(
            "i,nj->nij", self.correction_plus,
            numerical_x_plus - fx_plus,
        )
        derivative_x -= np.einsum(
            "i,nj->nij", self.correction_minus,
            numerical_x_minus - fx_minus,
        )
        derivative_y = np.einsum(
            "jb,nib->nij", self.diff, self._tensor(flux_y)
        )
        derivative_y += np.einsum(
            "j,ni->nij", self.correction_plus,
            numerical_y_plus - fy_plus,
        )
        derivative_y -= np.einsum(
            "j,ni->nij", self.correction_minus,
            numerical_y_minus - fy_minus,
        )
        return (self.scale * (derivative_x + derivative_y)).reshape(-1)

    def positivity_limit(self, values):
        """Apply a cell-average-preserving linear scaling limiter."""
        from modepy import LegendreGaussQuadrature

        tensor = self._tensor(values).copy()
        weights = np.asarray(
            LegendreGaussQuadrature(
                self.q - 1, force_dim_axis=True
            ).weights
        ).reshape(-1)
        tensor_weights = np.outer(weights, weights) / 4.0
        averages = np.einsum("ij,nij->n", tensor_weights, tensor)
        minima = np.min(tensor, axis=(1, 2))
        for ibox in np.nonzero(minima < 0.0)[0]:
            average = averages[ibox]
            if average <= 0.0:
                tensor[ibox] = 0.0
            else:
                factor = min(
                    1.0,
                    average / max(average - minima[ibox], 1.0e-300),
                )
                tensor[ibox] = average + factor * (tensor[ibox] - average)
        return tensor.reshape(-1)

# }}}


# {{{ FMM paths

def _build_chemo_wrangler(ctx, queue, traversal, tables, q_order, fmm_order,
                          lambda_alpha):
    """Wrangler for one pass producing (c, dc/dx, dc/dy) at fixed lambda."""
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import AxisTargetDerivative, YukawaKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    base_kernel = YukawaKernel(2)
    out_kernels = [
        base_kernel,
        AxisTargetDerivative(0, base_kernel),
        AxisTargetDerivative(1, base_kernel),
    ]
    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(base_kernel)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(base_kernel)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, base_kernel),
        partial(local_expn_class, base_kernel),
        out_kernels,
        exclude_self=True,
    )
    near_field_table = {
        repr(out_knl): [table]
        for out_knl, table in zip(out_kernels, tables, strict=True)
    }
    target_to_source = np.arange(traversal.tree.ntargets, dtype=np.int32)
    return FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=near_field_table,
        dtype=np.complex128,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        kernel_extra_kwargs={base_kernel.yukawa_lambda_name: lambda_alpha},
        self_extra_kwargs={"target_to_source": target_to_source},
    )


def _drive(queue, traversal, wrangler, weighted_sources, source_vals):
    from volumential.volume_fmm import drive_volume_fmm

    queue.finish()
    start = time.perf_counter()
    outputs = drive_volume_fmm(
        traversal,
        wrangler,
        weighted_sources,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )
    queue.finish()
    wall_s = time.perf_counter() - start
    return [np.real(output.get(queue)) for output in outputs], wall_s


def _to_device_pair(queue, values_host, weights_dev):
    import pyopencl.array as cla

    source_vals = cla.to_device(
        queue, np.ascontiguousarray(values_host.astype(np.complex128))
    )
    weighted = source_vals * weights_dev.astype(np.complex128)
    return weighted, source_vals


def _get_chemo_tables(
        queue, cache_path, q_order, lambda_alpha, level, root_extent,
        build_config):
    from volumential.table_manager import NearFieldInteractionTableManager

    tables = []
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=root_extent, queue=queue
    ) as table_manager:
        for kernel_type in ("Yukawa", "Yukawa-Dx", "Yukawa-Dy"):
            table, _ = table_manager.get_table(
                2,
                kernel_type,
                q_order,
                source_box_level=int(level),
                queue=queue,
                build_config=build_config,
                lam=float(lambda_alpha),
            )
            tables.append(table)
    return tables


def _get_direct_yukawa_table_timed(
        queue, cache_path, q_order, lam, level, root_extent, build_config):
    from volumential.table_manager import NearFieldInteractionTableManager

    with _capture_table_get_timings() as records:
        with NearFieldInteractionTableManager(
            str(cache_path), root_extent=root_extent, queue=queue
        ) as table_manager:
            table, _ = table_manager.get_table(
                2,
                "Yukawa",
                q_order,
                source_box_level=int(level),
                queue=queue,
                build_config=build_config,
                lam=float(lam),
            )
    summary = _summarize_table_get_timings(records)
    return table, float(summary["build_s"])

# }}}


def _ladder_lambda(lambda_target, ladder_ratio):
    """Return the smallest ladder value above a target, anchored at one."""
    k = math.ceil(
        math.log(max(lambda_target, 1.0)) / math.log(ladder_ratio)
        - 1e-12
    )
    return ladder_ratio ** max(k, 0)


def _plan_time_step(remaining, dt_cap, dt_floor, ladder_ratio):
    """Choose a regime-admissible step without leaving a short remainder."""
    tolerance = 32.0 * np.finfo(float).eps * max(remaining, dt_cap, dt_floor)
    if remaining < dt_floor - tolerance or dt_cap < dt_floor - tolerance:
        return None

    if remaining <= dt_cap + tolerance:
        return remaining, True, False

    def equal_step_endpoint_plan():
        endpoint_steps = max(2, math.ceil((remaining - tolerance) / dt_cap))
        endpoint_dt = remaining / endpoint_steps
        if dt_floor - tolerance <= endpoint_dt <= dt_cap + tolerance:
            return endpoint_dt, False, True
        return None

    lam = _ladder_lambda(1.0 / math.sqrt(dt_cap), ladder_ratio)
    ladder_dt = 1.0 / lam**2
    if ladder_dt < dt_floor - tolerance:
        return equal_step_endpoint_plan()

    remainder = remaining - ladder_dt
    remainder_steps = max(1, math.ceil((remainder - tolerance) / dt_cap))
    if remainder >= remainder_steps * dt_floor - tolerance:
        return ladder_dt, False, False

    # A regular ladder step would strand an interval that cannot be covered by
    # steps satisfying both the CFL cap and theta floor. Use the smallest
    # equal-step endpoint plan admissible under the current cap.
    return equal_step_endpoint_plan()


def _radial_gradient_diagnostics(coords, weights, rho, gradient_x, gradient_y):
    radius_sq = coords[0] ** 2 + coords[1] ** 2
    radius = np.sqrt(radius_sq)
    nonzero = radius > 32.0 * np.finfo(float).eps
    radial = np.zeros_like(radius)
    tangential = np.zeros_like(radius)
    radial[nonzero] = (
        coords[0, nonzero] * gradient_x[nonzero]
        + coords[1, nonzero] * gradient_y[nonzero]
    ) / radius[nonzero]
    tangential[nonzero] = (
        -coords[1, nonzero] * gradient_x[nonzero]
        + coords[0, nonzero] * gradient_y[nonzero]
    ) / radius[nonzero]

    measure = weights * np.maximum(rho, 0.0)
    gradient_norm_sq = float(
        np.sum(measure * (gradient_x**2 + gradient_y**2))
    )
    denominator = max(gradient_norm_sq, 1.0e-300)
    return {
        "tangential_rel_l2": math.sqrt(
            float(np.sum(measure * tangential**2)) / denominator
        ),
        "outward_rel_l2": math.sqrt(
            float(np.sum(measure * np.maximum(radial, 0.0)**2)) / denominator
        ),
    }


def _build_ks_geometry(ctx, queue, q_order, nlevels, root_extent):
    import volumential.meshgen as mg

    radius = 0.5 * root_extent
    mesh = mg.MeshGen2D(q_order, nlevels, -radius, radius, queue=queue)
    return mg.build_geometry_info(
        ctx,
        queue,
        2,
        q_order,
        mesh,
        bbox=np.array([[-radius, radius]] * 2, dtype=np.float64),
    )


def _field_diagnostics(
        rho, coords, weights, initial_mass, root_extent, leaf_extent,
        core_radius):
    rho = np.asarray(rho)
    radius_sq = coords[0] ** 2 + coords[1] ** 2
    radius = np.sqrt(radius_sq)
    positive_rho = np.maximum(rho, 0.0)
    mass = float(np.sum(weights * rho))
    positive_mass = max(float(np.sum(weights * positive_rho)), 1.0e-300)
    negative_mass = float(np.sum(weights * np.maximum(-rho, 0.0)))
    peak = max(float(np.max(rho)), 1.0e-300)
    half_extent = 0.5 * root_extent
    max_coordinate = np.maximum(np.abs(coords[0]), np.abs(coords[1]))
    boundary_shell = max_coordinate > 0.9 * half_extent
    boundary_trace = max_coordinate > half_extent - leaf_extent
    weighted_positive = weights * positive_rho
    order = np.argsort(radius)
    cumulative = np.cumsum(weighted_positive[order])
    half_index = min(
        int(np.searchsorted(cumulative, 0.5 * positive_mass)), len(order) - 1
    )
    centroid_x = float(np.sum(weighted_positive * coords[0]) / positive_mass)
    centroid_y = float(np.sum(weighted_positive * coords[1]) / positive_mass)
    second_moment = float(np.sum(weights * rho * radius_sq))
    return {
        "rho_max": float(np.max(rho)),
        "rho_min": float(np.min(rho)),
        "mass": mass,
        "mass_drift_rel": abs(mass - initial_mass) / initial_mass,
        "second_moment": second_moment,
        "second_moment_per_mass": second_moment / max(mass, 1.0e-300),
        "core_mass_fraction": float(
            np.sum(weighted_positive[radius <= core_radius]) / positive_mass
        ),
        "boundary_mass_fraction": float(
            np.sum(weighted_positive[boundary_shell]) / positive_mass
        ),
        "boundary_trace_rel_peak": float(
            np.max(positive_rho[boundary_trace]) / peak
        ),
        "negative_mass_fraction": negative_mass / initial_mass,
        "centroid_radius": math.hypot(centroid_x, centroid_y),
        "half_mass_radius": float(radius[order[half_index]]),
    }


def run_case(
    ctx,
    queue,
    *,
    mode: str,
    case_id: str,
    mass_factor: float,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    split_order: int,
    alpha: float,
    initial_profile: str,
    profile_scale: float,
    cfl: float,
    theta_max: float,
    t_end: float,
    max_steps: int,
    blowup_factor: float,
    dt_max: float,
    root_extent: float,
    cutoff_inner_radius: float,
    cutoff_outer_radius: float,
    core_radius: float,
    ladder_ratio: float,
    rke_beta_mode: str,
    direct_only: bool,
    save_fields: Path | None,
):
    q_points, q_weights, tree, traversal = _build_ks_geometry(
        ctx, queue, q_order, nlevels, root_extent
    )
    coords = _coords_host(queue, q_points)
    weights_host = q_weights.get(queue)
    n_targets = int(coords.shape[1])

    leaf_level = int(tree.nlevels) - 1
    leaf_extent = float(tree.root_extent) * 0.5**leaf_level
    n_boxes = n_targets // q_order**2
    if n_boxes * q_order**2 != n_targets:
        raise ValueError("target count is not divisible into tensor boxes")
    transport = ConservativeDGTransport(
        q_order, coords, leaf_extent, float(tree.root_extent)
    )

    mass = mass_factor * UNSCREENED_REFERENCE_MASS
    lambda_alpha = math.sqrt(alpha)
    high_accuracy = mode == "full"
    direct_build_config = _yukawa_reference_build_config(
        q_order, high_accuracy=high_accuracy
    )
    rke_channel_build_config = _split_channel_build_config(
        q_order, high_accuracy=high_accuracy
    )
    smooth_quad_order = _split_smooth_quad_order(
        q_order, split_order, high_accuracy=high_accuracy
    )
    rho = _initial_density(
        coords,
        weights_host,
        mass,
        initial_profile,
        profile_scale,
        cutoff_inner_radius,
        cutoff_outer_radius,
    )
    rho_max_initial = float(np.max(rho))
    mass_initial = float(np.sum(weights_host * rho))
    initial_diagnostics = _field_diagnostics(
        rho, coords, weights_host, mass_initial, root_extent, leaf_extent,
        core_radius,
    )

    # dt floor from the resolved-parameter regime: theta = lam * h <= theta_max
    dt_theta_floor = (leaf_extent / theta_max) ** 2
    node_gap = transport.minimum_node_gap

    cache_dir.mkdir(parents=True, exist_ok=True)
    chemo_cache = cache_dir / f"ks-chemo-{case_id}.sqlite"
    rke_cache = cache_dir / f"ks-rke-{case_id}.sqlite"
    for path in (chemo_cache, rke_cache):
        _clear_sqlite_cache(path)

    print(f"[{case_id}] building fixed-alpha chemoattractant tables", flush=True)
    chemo_tables = _get_chemo_tables(
        queue,
        chemo_cache,
        q_order,
        lambda_alpha,
        leaf_level,
        root_extent,
        direct_build_config,
    )
    chemo_wrangler = _build_chemo_wrangler(
        ctx, queue, traversal, chemo_tables, q_order, fmm_order, lambda_alpha
    )

    rke_base_table = None
    split_term_tables = None
    rke_channel_build_s = 0.0
    if not direct_only:
        # Build one RKE channel family and reuse it for every ladder value.
        print(f"[{case_id}] building RKE channel family (p={split_order})",
              flush=True)
        channel_start = time.perf_counter()
        rke_base_table = _get_laplace_2d_table(
            queue,
            rke_cache,
            q_order,
            build_config=rke_channel_build_config,
        )
        seed_wrangler, _, _ = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel="Yukawa",
            parameter=1.0,
            table=rke_base_table,
            source_weights=q_weights,
            q_points=q_points,
            source_values_host=rho,
            split=True,
            split_order=split_order,
            split_smooth_quad_order=smooth_quad_order,
            split_auto_config={
                "power_log_single_table_beta_mode": rke_beta_mode,
            },
        )
        split_term_tables = dict(seed_wrangler.helmholtz_split_term_tables)
        rke_channel_build_s = time.perf_counter() - channel_start

    direct_wranglers: dict[float, Any] = {}
    rke_wranglers: dict[float, Any] = {}
    direct_table_build_total_s = 0.0
    chemo_solve_total_s = 0.0
    direct_strategy_total_s = 0.0
    rke_strategy_total_s = rke_channel_build_s
    strategy_crossings = []
    previous_cost_gap = -rke_channel_build_s
    rke_first_cheaper_step = ""
    direct_first_cheaper_after_rke_step = ""
    max_theta_seen = 0.0
    max_mismatch = 0.0
    max_boundary_mass_fraction = initial_diagnostics["boundary_mass_fraction"]
    max_boundary_trace_rel_peak = initial_diagnostics[
        "boundary_trace_rel_peak"
    ]
    max_negative_mass_fraction = 0.0
    max_centroid_radius = initial_diagnostics["centroid_radius"]
    min_half_mass_radius = initial_diagnostics["half_mass_radius"]
    cumulative_limiter_correction = 0.0
    admissible = True
    radial_preflight = None
    lambdas_seen = set()
    step_rows = []
    field_snapshots = {"step0": rho.copy()}

    t = 0.0
    stop_reason = "t_end"
    step = 0
    while t < t_end and step < max_steps:
        # chemoattractant and drift (shared physics cost, fixed lambda)
        weighted, source_vals = _to_device_pair(queue, rho, q_weights)
        (c_field, dcdx, dcdy), chemo_solve_s = _drive(
            queue, traversal, chemo_wrangler, weighted, source_vals
        )
        chemo_solve_total_s += chemo_solve_s

        if radial_preflight is None:
            radial_preflight = _radial_gradient_diagnostics(
                coords, weights_host, rho, dcdx, dcdy
            )
            radial_preflight["pass"] = (
                radial_preflight["tangential_rel_l2"]
                <= RADIAL_PREFLIGHT_MAX_REL_L2
                and radial_preflight["outward_rel_l2"]
                <= RADIAL_PREFLIGHT_MAX_REL_L2
            )
            print(
                f"[{case_id}] radial-gradient preflight: "
                f"tangential={radial_preflight['tangential_rel_l2']:.2e}, "
                f"outward={radial_preflight['outward_rel_l2']:.2e}",
                flush=True,
            )
            if not radial_preflight["pass"]:
                admissible = False
                stop_reason = "radial_gradient_preflight_exit"
                break

        u_max = float(np.max(np.hypot(dcdx, dcdy)))
        dt_cfl = cfl * node_gap / max(u_max, 1e-12)
        if dt_cfl < dt_theta_floor:
            stop_reason = "theta_regime_exit"
            print(
                f"[{case_id}] step {step}: CFL dt {dt_cfl:.3e} below theta "
                f"floor {dt_theta_floor:.3e}; stopping (reported, not "
                "silently accepted)",
                flush=True,
            )
            break

        step_plan = _plan_time_step(
            t_end - t,
            min(dt_cfl, dt_max),
            dt_theta_floor,
            ladder_ratio,
        )
        if step_plan is None:
            stop_reason = "endpoint_plan_exit"
            print(
                f"[{case_id}] step {step}: no step in "
                f"[{dt_theta_floor:.3e}, {min(dt_cfl, dt_max):.3e}] can "
                f"reach t_end={t_end:.6g}; stopping",
                flush=True,
            )
            break
        dt, terminal_step, endpoint_adjusted = step_plan
        lam = 1.0 / math.sqrt(dt)
        theta = lam * leaf_extent
        if theta > theta_max * (1.0 + 1.0e-14):
            stop_reason = (
                "terminal_theta_regime_exit" if terminal_step
                else "theta_regime_exit"
            )
            print(
                f"[{case_id}] step {step}: quantized lambda={lam:.6g} gives "
                f"theta={theta:.3f} above {theta_max:.3f}; stopping",
                flush=True,
            )
            break
        max_theta_seen = max(max_theta_seen, theta)
        new_lambda = lam not in lambdas_seen
        lambdas_seen.add(lam)

        # Conservative explicit transport followed by implicit diffusion.
        div_flux = transport.divergence(rho, dcdx, dcdy)
        transported_unlimited = rho - dt * div_flux
        transported = transport.positivity_limit(transported_unlimited)
        limiter_correction = float(
            np.sum(weights_host * np.abs(transported - transported_unlimited))
        )
        cumulative_limiter_correction += limiter_correction
        transport_mass_change_rel = abs(
            float(np.sum(weights_host * transported))
            - float(np.sum(weights_host * rho))
        ) / mass_initial
        rhs = transported / dt

        # direct strategy: per-lambda table (built once per ladder value)
        direct_table_build_s = 0.0
        direct_wrangler_build_s = 0.0
        if lam not in direct_wranglers:
            lam_tag = f"{lam:.17g}".replace(".", "p").replace("-", "m")
            direct_cache = cache_dir / (
                f"ks-direct-{case_id}-lam{lam_tag}.sqlite"
            )
            _clear_sqlite_cache(direct_cache)
            table, direct_table_build_s = _get_direct_yukawa_table_timed(
                queue,
                direct_cache,
                q_order,
                lam,
                leaf_level,
                root_extent,
                direct_build_config,
            )
            direct_table_build_total_s += direct_table_build_s
            wrangler_start = time.perf_counter()
            direct_wranglers[lam], _, _ = _build_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel="Yukawa",
                parameter=lam,
                table=table,
                source_weights=q_weights,
                q_points=q_points,
                source_values_host=rho,
                split=False,
                split_order=split_order,
            )
            direct_wrangler_build_s = time.perf_counter() - wrangler_start

        weighted, source_vals = _to_device_pair(queue, rhs, q_weights)

        # RKE strategy: fixed channel family, online coefficients per lambda.
        rke_wrangler_build_s = 0.0
        if not direct_only and lam not in rke_wranglers:
            wrangler_start = time.perf_counter()
            rke_wranglers[lam], _, _ = _build_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel="Yukawa",
                parameter=lam,
                table=rke_base_table,
                source_weights=q_weights,
                q_points=q_points,
                source_values_host=rho,
                split=True,
                split_order=split_order,
                split_term_tables=split_term_tables,
                split_smooth_quad_order=smooth_quad_order,
                split_auto_config={
                    "power_log_single_table_beta_mode": rke_beta_mode,
                },
            )
            rke_wrangler_build_s = time.perf_counter() - wrangler_start

        strategy_order = "direct-only"
        rho_rke = None
        rke_solve_s = 0.0
        if direct_only:
            (rho_direct,), direct_solve_s = _drive(
                queue, traversal, direct_wranglers[lam], weighted, source_vals
            )
        elif step % 2 == 0:
            strategy_order = "direct-first"
            (rho_direct,), direct_solve_s = _drive(
                queue, traversal, direct_wranglers[lam], weighted, source_vals
            )
            (rho_rke,), rke_solve_s = _drive(
                queue, traversal, rke_wranglers[lam], weighted, source_vals
            )
        else:
            strategy_order = "rke-first"
            (rho_rke,), rke_solve_s = _drive(
                queue, traversal, rke_wranglers[lam], weighted, source_vals
            )
            (rho_direct,), direct_solve_s = _drive(
                queue, traversal, direct_wranglers[lam], weighted, source_vals
            )

        mismatch = 0.0
        if rho_rke is not None:
            difference = rho_rke - rho_direct
            weighted_reference = max(
                float(np.sqrt(np.sum(weights_host * rho_direct**2))), 1e-300
            )
            mismatch = (
                float(np.sqrt(np.sum(weights_host * difference**2)))
                / weighted_reference
            )
        max_mismatch = max(max_mismatch, mismatch)

        # advance with the direct solution
        rho = rho_direct
        t = t_end if terminal_step else t + dt
        step += 1

        direct_step_s = (
            direct_table_build_s + direct_wrangler_build_s + direct_solve_s
        )
        rke_step_s = rke_wrangler_build_s + rke_solve_s
        direct_strategy_total_s += direct_step_s
        rke_strategy_total_s += rke_step_s
        cost_gap = direct_strategy_total_s - rke_strategy_total_s
        if not direct_only and cost_gap * previous_cost_gap < 0.0:
            direction = (
                "rke_became_cheaper" if cost_gap > 0.0
                else "direct_became_cheaper"
            )
            strategy_crossings.append({"step": step, "direction": direction})
            if direction == "rke_became_cheaper" and rke_first_cheaper_step == "":
                rke_first_cheaper_step = step
            elif (
                    direction == "direct_became_cheaper"
                    and rke_first_cheaper_step != ""
                    and direct_first_cheaper_after_rke_step == ""):
                direct_first_cheaper_after_rke_step = step
        previous_cost_gap = cost_gap

        diagnostics = _field_diagnostics(
            rho, coords, weights_host, mass_initial, root_extent, leaf_extent,
            core_radius,
        )
        limiter_correction_rel = limiter_correction / mass_initial
        cumulative_limiter_correction_rel = (
            cumulative_limiter_correction / mass_initial
        )
        step_admissible = (
            np.all(np.isfinite(rho))
            and diagnostics["negative_mass_fraction"] <= 1.0e-8
            and diagnostics["mass_drift_rel"] <= 5.0e-3
            and diagnostics["boundary_mass_fraction"] <= 1.0e-2
            and diagnostics["boundary_trace_rel_peak"] <= 5.0e-3
            and diagnostics["centroid_radius"] <= 0.25 * leaf_extent
            and diagnostics["half_mass_radius"] >= 6.0 * leaf_extent
            and cumulative_limiter_correction_rel <= 1.0e-4
            and mismatch <= 1.0e-4
        )
        max_boundary_mass_fraction = max(
            max_boundary_mass_fraction, diagnostics["boundary_mass_fraction"]
        )
        max_boundary_trace_rel_peak = max(
            max_boundary_trace_rel_peak,
            diagnostics["boundary_trace_rel_peak"],
        )
        max_negative_mass_fraction = max(
            max_negative_mass_fraction, diagnostics["negative_mass_fraction"]
        )
        max_centroid_radius = max(
            max_centroid_radius, diagnostics["centroid_radius"]
        )
        min_half_mass_radius = min(
            min_half_mass_radius, diagnostics["half_mass_radius"]
        )

        step_rows.append(
            {
                "case_id": case_id,
                "mode": mode,
                "direct_only": int(direct_only),
                "step": step,
                "time": t,
                "dt": dt,
                "lam": lam,
                "theta": theta,
                "u_max": u_max,
                **diagnostics,
                "transport_mass_change_rel": transport_mass_change_rel,
                "limiter_correction_rel": limiter_correction_rel,
                "cumulative_limiter_correction_rel": (
                    cumulative_limiter_correction_rel
                ),
                "terminal_step": int(terminal_step),
                "endpoint_adjusted": int(endpoint_adjusted),
                "admissible": int(step_admissible),
                "new_lambda": int(new_lambda),
                "strategy_order": strategy_order,
                "chemo_solve_s": chemo_solve_s,
                "direct_table_build_s": direct_table_build_s,
                "direct_wrangler_build_s": direct_wrangler_build_s,
                "direct_solve_s": direct_solve_s,
                "rke_wrangler_build_s": (
                    "" if direct_only else rke_wrangler_build_s
                ),
                "rke_solve_s": "" if direct_only else rke_solve_s,
                "rke_vs_direct_weighted_rel_l2": (
                    "" if direct_only else mismatch
                ),
                "cumulative_direct_strategy_s": direct_strategy_total_s,
                "cumulative_rke_strategy_s": (
                    "" if direct_only else rke_strategy_total_s
                ),
                "strategy_cost_gap_direct_minus_rke_s": (
                    "" if direct_only else cost_gap
                ),
            }
        )
        if step % 10 == 0 or step == 1:
            print(
                f"[{case_id}] step {step}: t={t:.4e} dt={dt:.3e} "
                f"lam={lam:.1f} theta={theta:.2f} "
                f"rho_max={diagnostics['rho_max']:.3e} "
                f"mismatch={mismatch:.2e}",
                flush=True,
            )
        if not step_admissible:
            admissible = False
            stop_reason = "admissibility_exit"
            print(
                f"[{case_id}] step {step}: admissibility gate failed; stopping",
                flush=True,
            )
            break
        if diagnostics["rho_max"] > blowup_factor * rho_max_initial:
            stop_reason = "pre_blowup_density_threshold"
            print(
                f"[{case_id}] step {step}: density growth factor "
                f"{diagnostics['rho_max'] / rho_max_initial:.1f} reached "
                "threshold; "
                "stopping before blow-up",
                flush=True,
            )
            break
    else:
        if t < t_end and step >= max_steps:
            stop_reason = "max_steps"

    field_snapshots["final"] = rho.copy()
    if save_fields is not None:
        save_fields.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_fields / f"{case_id}-fields.npz",
            coords=coords,
            weights=weights_host,
            **field_snapshots,
        )

    final_diagnostics = _field_diagnostics(
        rho, coords, weights_host, mass_initial, root_extent, leaf_extent,
        core_radius,
    )
    second_moment_ratio = (
        final_diagnostics["second_moment_per_mass"]
        / initial_diagnostics["second_moment_per_mass"]
    )
    if mass < UNSCREENED_REFERENCE_MASS:
        trend_criterion_pass = (
            second_moment_ratio >= 1.03
            and (
                final_diagnostics["rho_max"] < rho_max_initial
                or final_diagnostics["core_mass_fraction"]
                < initial_diagnostics["core_mass_fraction"]
            )
        )
    elif mass > UNSCREENED_REFERENCE_MASS:
        trend_criterion_pass = (
            second_moment_ratio <= 0.97
            and (
                final_diagnostics["rho_max"] > rho_max_initial
                or final_diagnostics["core_mass_fraction"]
                > initial_diagnostics["core_mass_fraction"]
            )
        )
    else:
        trend_criterion_pass = False
    regime = (
        "below_8pi_reference" if mass < UNSCREENED_REFERENCE_MASS
        else "above_8pi_reference" if mass > UNSCREENED_REFERENCE_MASS
        else "at_8pi_reference"
    )
    if radial_preflight is None:
        radial_preflight = {
            "tangential_rel_l2": math.nan,
            "outward_rel_l2": math.nan,
            "pass": False,
        }
    summary_row = {
        "case_id": case_id,
        "mode": mode,
        "direct_only": int(direct_only),
        "mass_factor": mass_factor,
        "mass": mass,
        "reference_mass": UNSCREENED_REFERENCE_MASS,
        "reference_model": "classical_unscreened_keller_segel",
        "regime": regime,
        "q_order": q_order,
        "nlevels": nlevels,
        "n_targets": n_targets,
        "fmm_order": fmm_order,
        "split_order": split_order,
        "direct_regular_quad_order": direct_build_config.regular_quad_order,
        "direct_radial_quad_order": direct_build_config.radial_quad_order,
        "rke_channel_regular_quad_order": (
            "" if direct_only
            else rke_channel_build_config.regular_quad_order
        ),
        "rke_channel_radial_quad_order": (
            "" if direct_only else rke_channel_build_config.radial_quad_order
        ),
        "split_smooth_quad_order": (
            "" if direct_only or smooth_quad_order is None
            else smooth_quad_order
        ),
        "alpha": alpha,
        "lambda_alpha": lambda_alpha,
        "initial_profile": initial_profile,
        "profile_scale": profile_scale,
        "cfl": cfl,
        "theta_max": theta_max,
        "root_extent": root_extent,
        "leaf_level": leaf_level,
        "leaf_box_extent": leaf_extent,
        "cutoff_inner_radius": cutoff_inner_radius,
        "cutoff_outer_radius": cutoff_outer_radius,
        "dt_max": dt_max,
        "lambda_ladder_ratio": ladder_ratio,
        "rke_beta_mode": "" if direct_only else rke_beta_mode,
        "requested_t_end": t_end,
        "max_steps": max_steps,
        "blowup_factor": blowup_factor,
        "n_steps": step,
        "final_time": t,
        "stop_reason": stop_reason,
        "rho_max_initial": rho_max_initial,
        "rho_max_final": final_diagnostics["rho_max"],
        "rho_max_ratio": final_diagnostics["rho_max"] / rho_max_initial,
        "rho_min_final": final_diagnostics["rho_min"],
        "mass_drift_rel": final_diagnostics["mass_drift_rel"],
        "second_moment_ratio": second_moment_ratio,
        "core_mass_fraction_initial": initial_diagnostics[
            "core_mass_fraction"
        ],
        "core_mass_fraction_final": final_diagnostics["core_mass_fraction"],
        "max_boundary_mass_fraction": max_boundary_mass_fraction,
        "max_boundary_trace_rel_peak": max_boundary_trace_rel_peak,
        "max_negative_mass_fraction": max_negative_mass_fraction,
        "max_centroid_radius": max_centroid_radius,
        "min_half_mass_radius": min_half_mass_radius,
        "cumulative_limiter_correction_rel": (
            cumulative_limiter_correction / mass_initial
        ),
        "admissible": int(admissible and stop_reason == "t_end"),
        "trend_criterion_pass": int(trend_criterion_pass),
        "pair_outcome_pass": 0,
        "pair_moment_ratio_separation": "",
        "max_theta_seen": max_theta_seen,
        "lambda_min": min(lambdas_seen) if lambdas_seen else "",
        "lambda_max": max(lambdas_seen) if lambdas_seen else "",
        "n_distinct_lambdas": len(lambdas_seen),
        "rke_channel_build_s": "" if direct_only else rke_channel_build_s,
        "direct_table_build_total_s": direct_table_build_total_s,
        "chemo_solve_total_s": chemo_solve_total_s,
        "direct_strategy_total_s": direct_strategy_total_s,
        "rke_strategy_total_s": "" if direct_only else rke_strategy_total_s,
        "strategy_crossings_json": (
            "" if direct_only else json.dumps(
                strategy_crossings, separators=(",", ":")
            )
        ),
        "rke_first_cheaper_step": (
            "" if direct_only else rke_first_cheaper_step
        ),
        "direct_first_cheaper_after_rke_step": (
            "" if direct_only else direct_first_cheaper_after_rke_step
        ),
        "final_strategy_cost_gap_direct_minus_rke_s": (
            "" if direct_only
            else direct_strategy_total_s - rke_strategy_total_s
        ),
        "max_rke_vs_direct_weighted_rel_l2": (
            "" if direct_only else max_mismatch
        ),
        "radial_gradient_tangential_rel_l2": radial_preflight[
            "tangential_rel_l2"
        ],
        "radial_gradient_outward_rel_l2": radial_preflight["outward_rel_l2"],
        "radial_gradient_preflight_pass": int(radial_preflight["pass"]),
        "strategy_cost_definition": (
            "direct=per-lambda table build+wrangler build+solve;"
            "rke=one channel family build+per-lambda wrangler build+solve;"
            "shared fixed-alpha chemoattractant pass, conservative transport,"
            "diagnostics, and output excluded from both;solve order alternates"
        ),
    }
    return step_rows, summary_row


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/benchmarks/keller-segel-continuation"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/keller-segel-cache"),
    )
    parser.add_argument(
        "--mass-factors", type=float, nargs="+", default=[0.85, 1.15]
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--initial-profile",
        choices=("critical", "gaussian"),
        default="critical",
    )
    parser.add_argument("--profile-scale", type=float, default=0.3)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--theta-max", type=float, default=0.9)
    parser.add_argument("--split-order", type=int, default=2)
    parser.add_argument("--q-order", type=int)
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--root-extent", type=float, default=4.0)
    parser.add_argument("--cutoff-inner-radius", type=float, default=1.6)
    parser.add_argument("--cutoff-outer-radius", type=float, default=1.8)
    parser.add_argument("--core-radius", type=float, default=0.3)
    parser.add_argument(
        "--lambda-ladder-ratio", type=float,
        default=DEFAULT_LAMBDA_LADDER_RATIO,
    )
    parser.add_argument("--dt-max", type=float, default=1.0e-3)
    parser.add_argument(
        "--rke-beta-mode", choices=("table", "p2p"), default="table"
    )
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--blowup-factor", type=float, default=10.0)
    parser.add_argument(
        "--save-fields", action="store_true",
        help="save density snapshots (initial/final) per case",
    )
    args = parser.parse_args()

    if not (0.0 < args.cutoff_inner_radius < args.cutoff_outer_radius):
        parser.error("cutoff radii must be positive and strictly increasing")
    if args.cutoff_outer_radius >= 0.5 * args.root_extent:
        parser.error("outer cutoff radius must lie inside the root box")
    if args.lambda_ladder_ratio <= 1.0:
        parser.error("lambda ladder ratio must exceed one")
    if not (0.0 < args.theta_max < 1.0):
        parser.error("theta-max must lie strictly between zero and one")
    if any(mass_factor <= 0.0 for mass_factor in args.mass_factors):
        parser.error("mass factors must be positive")
    if args.profile_scale <= 0.0:
        parser.error("profile-scale must be positive")

    import pyopencl as cl

    smoke = args.mode == "smoke"
    q_order = args.q_order if args.q_order is not None else 2
    nlevels = args.nlevels if args.nlevels is not None else 9
    fmm_order = args.fmm_order if args.fmm_order is not None else (
        8 if smoke else 16
    )
    t_end = args.t_end if args.t_end is not None else (
        5e-4 if smoke else 0.0144
    )
    max_steps = args.max_steps if args.max_steps is not None else (
        3 if smoke else 100
    )

    device = _select_opencl_device(cl, args.backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    step_rows = []
    summary_rows = []
    for mass_factor in args.mass_factors:
        case_id = (
            f"ks2d-{args.initial_profile}-q{q_order}-l{nlevels}"
            f"-a{args.alpha:g}-m{mass_factor:g}"
        )
        case_steps, case_summary = run_case(
            ctx,
            queue,
            mode=args.mode,
            case_id=case_id,
            mass_factor=mass_factor,
            cache_dir=args.cache_dir,
            q_order=q_order,
            nlevels=nlevels,
            fmm_order=fmm_order,
            split_order=args.split_order,
            alpha=args.alpha,
            initial_profile=args.initial_profile,
            profile_scale=args.profile_scale,
            cfl=args.cfl,
            theta_max=args.theta_max,
            t_end=t_end,
            max_steps=max_steps,
            blowup_factor=args.blowup_factor,
            dt_max=args.dt_max,
            root_extent=args.root_extent,
            cutoff_inner_radius=args.cutoff_inner_radius,
            cutoff_outer_radius=args.cutoff_outer_radius,
            core_radius=args.core_radius,
            ladder_ratio=args.lambda_ladder_ratio,
            rke_beta_mode=args.rke_beta_mode,
            direct_only=args.direct_only,
            save_fields=(args.out_dir / "fields") if args.save_fields else None,
        )
        step_rows.extend(case_steps)
        summary_rows.append(case_summary)
        print(
            f"[{case_id}] done: {case_summary['n_steps']} steps, "
            f"stop={case_summary['stop_reason']}, "
            f"rho_max_ratio={case_summary['rho_max_ratio']:.2f}",
            flush=True,
        )

    if len(summary_rows) == 2:
        subcritical = next(
            (
                row for row in summary_rows
                if row["regime"] == "below_8pi_reference"
            ),
            None,
        )
        supercritical = next(
            (
                row for row in summary_rows
                if row["regime"] == "above_8pi_reference"
            ),
            None,
        )
        if subcritical is not None and supercritical is not None:
            separation = (
                subcritical["second_moment_ratio"]
                - supercritical["second_moment_ratio"]
            )
            pair_pass = (
                bool(subcritical["admissible"])
                and bool(supercritical["admissible"])
                and bool(subcritical["trend_criterion_pass"])
                and bool(supercritical["trend_criterion_pass"])
                and separation >= 0.06
            )
            for row in summary_rows:
                row["pair_outcome_pass"] = int(pair_pass)
                row["pair_moment_ratio_separation"] = separation

    _write_csv(args.out_dir / "ks_steps.csv", STEP_FIELDS, step_rows)
    _write_csv(args.out_dir / "ks_summary.csv", SUMMARY_FIELDS, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
