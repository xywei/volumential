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
import sqlite3
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
    "strategy",
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
    # windowed-continuation (E4) step columns; empty for other strategies
    "binding_constraint",
    "checkpoint_landed",
    "windowed_status",
    "windowed_condition_number",
    "windowed_assemble_s",
    "windowed_register_s",
    "windowed_table_load_s",
    "windowed_wrangler_build_s",
    "windowed_solve_s",
)

SUMMARY_FIELDS = (
    "case_id",
    "mode",
    "strategy",
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
    # windowed-continuation (E4) summary columns; empty for other strategies
    "window_theta",
    "windowed_p_star",
    "min_theta_seen",
    "dt_min",
    "dt_max_seen",
    "lambda_quantization",
    "binding_constraint_histogram_json",
    "theta_floor_bound_steps",
    "go_no_go_binding_verdict",
    "windowed_channel_build_s",
    "windowed_assemble_total_s",
    "windowed_register_total_s",
    "windowed_table_load_total_s",
    "windowed_wrangler_build_total_s",
    "windowed_solve_total_s",
    "windowed_strategy_total_s",
    "windowed_mean_provisioning_s_per_step",
    "n_windowed_refused",
    "n_windowed_failed",
    "checkpoint_times_json",
    "checkpoint_agreement_json",
    "max_checkpoint_rel_l2",
    "baseline_case_id",
)

UNSCREENED_REFERENCE_MASS = 8.0 * math.pi
DEFAULT_LAMBDA_LADDER_RATIO = 2.0 ** 0.125
RADIAL_PREFLIGHT_MAX_REL_L2 = 5.0e-3
DEFAULT_WINDOW_THETA = 16.0
DEFAULT_WINDOWED_P_STAR = 6
STRATEGIES = ("paired", "direct", "windowed")


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

    @classmethod
    def _numerical_faces(
        cls,
        rho_minus,
        rho_plus,
        velocity_minus,
        velocity_plus,
        minus_neighbors,
        plus_neighbors,
    ):
        has_minus = minus_neighbors >= 0
        minus = np.maximum(minus_neighbors, 0)
        exterior_minus_rho = np.where(
            has_minus[:, np.newaxis], rho_plus[minus], 0.0
        )
        exterior_minus_velocity = np.where(
            has_minus[:, np.newaxis], velocity_plus[minus], velocity_minus
        )
        numerical_minus = cls._rusanov(
            exterior_minus_rho,
            exterior_minus_velocity,
            rho_minus,
            velocity_minus,
        )

        has_plus = plus_neighbors >= 0
        plus = np.maximum(plus_neighbors, 0)
        exterior_plus_rho = np.where(
            has_plus[:, np.newaxis], rho_minus[plus], 0.0
        )
        exterior_plus_velocity = np.where(
            has_plus[:, np.newaxis], velocity_minus[plus], velocity_plus
        )
        numerical_plus = cls._rusanov(
            rho_plus,
            velocity_plus,
            exterior_plus_rho,
            exterior_plus_velocity,
        )
        return numerical_minus, numerical_plus

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

        numerical_x_minus, numerical_x_plus = self._numerical_faces(
            rho_x_minus,
            rho_x_plus,
            ux_minus,
            ux_plus,
            self.left,
            self.right,
        )
        numerical_y_minus, numerical_y_plus = self._numerical_faces(
            rho_y_minus,
            rho_y_plus,
            uy_minus,
            uy_plus,
            self.bottom,
            self.top,
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


def _prepare_windowed_channel_family(
        cache_path, q_order, level, root_extent, window_theta, p_star):
    """One-time build (or reload) of the parameter-independent windowed
    channel family; every continuation lambda reuses these tables."""
    from volumential.rke_table_assembly import get_windowed_channel_table

    start = time.perf_counter()
    for m in range(p_star):
        get_windowed_channel_table(
            cache_path,
            2,
            q_order,
            m,
            source_box_level=int(level),
            root_extent=float(root_extent),
            window_theta=float(window_theta),
        )
    return time.perf_counter() - start


def _provision_windowed_yukawa_table(
        queue, family_cache_path, registered_cache_path, q_order, lam, level,
        root_extent, window_theta, p_star):
    """Offline-assemble the fixed-``lam`` table from the windowed channel
    family, register it under the standard table-manager slot, and load it
    back through the ordinary ``get_table`` path (pure cache hit), so the
    solve consumes it exactly like a direct-built table.

    Returns ``(table_or_None, info)`` where ``info['status']`` follows the
    ``ok`` / ``refused`` / ``failed`` taxonomy (``refused`` is the windowed
    certificate declining the parameter; anything unexpected is
    ``failed``).
    """
    from volumential.rke_table_assembly import (
        RKEWindowConditioningError,
        RKEWindowCoverageError,
        assemble_windowed_parameterized_table,
    )
    from volumential.table_manager import NearFieldInteractionTableManager

    info: dict[str, Any] = {
        "status": "ok",
        "detail": "",
        "assemble_s": 0.0,
        "register_s": 0.0,
        "load_s": 0.0,
        "condition_number": "",
    }
    start = time.perf_counter()
    try:
        table, certificate = assemble_windowed_parameterized_table(
            family_cache_path,
            2,
            "Yukawa",
            q_order,
            float(lam),
            source_box_level=int(level),
            root_extent=float(root_extent),
            window_theta=float(window_theta),
            p_star=int(p_star),
        )
    except (RKEWindowCoverageError, RKEWindowConditioningError) as exc:
        info["status"] = "refused"
        info["detail"] = f"{type(exc).__name__}: {exc}"
        info["assemble_s"] = time.perf_counter() - start
        return None, info
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        info["status"] = "failed"
        info["detail"] = f"{type(exc).__name__}: {exc}"
        info["assemble_s"] = time.perf_counter() - start
        return None, info
    info["assemble_s"] = time.perf_counter() - start
    info["condition_number"] = float(certificate["condition_number"])

    # Registration and the pure-cache reload are provisioning too, and this
    # helper's contract is that unexpected provisioning errors come back as
    # info["status"] == "failed".  The caller records n_windowed_failed and
    # writes its CSV only once this returns, so an escaping SQLite, I/O or
    # checksum error would abort the whole continuation and lose the
    # diagnostic outcome rather than reporting it.
    register_start = time.perf_counter()
    try:
        with NearFieldInteractionTableManager(
            str(registered_cache_path), root_extent=float(root_extent),
            queue=queue,
        ) as table_manager:
            table_manager.register_external_table(
                2,
                "Yukawa",
                q_order,
                table,
                source_box_level=int(level),
                provenance={
                    "kind": "windowed_rke_assembly",
                    "window_theta": float(window_theta),
                    "p_star": int(p_star),
                    "condition_number": float(
                        certificate["condition_number"]
                    ),
                },
                lam=float(lam),
            )
        info["register_s"] = time.perf_counter() - register_start

        load_start = time.perf_counter()
        with NearFieldInteractionTableManager(
            str(registered_cache_path), root_extent=float(root_extent),
            queue=queue,
        ) as table_manager:
            loaded_table, is_recomputed = table_manager.get_table(
                2,
                "Yukawa",
                q_order,
                source_box_level=int(level),
                queue=queue,
                lam=float(lam),
            )
    except (
        ValueError, RuntimeError, NotImplementedError, TypeError,
        OSError, KeyError,
        # sqlite3's exceptions descend from Exception, not OSError
        sqlite3.Error,
    ) as exc:
        info["status"] = "failed"
        info["detail"] = f"{type(exc).__name__}: {exc}"
        info.setdefault("register_s", time.perf_counter() - register_start)
        return None, info
    if is_recomputed:
        info["status"] = "failed"
        info["detail"] = (
            "registered windowed table did not load as a pure cache hit"
        )
        return None, info
    info["load_s"] = time.perf_counter() - load_start
    return loaded_table, info

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
    strategy: str = "paired",
    window_theta: float = DEFAULT_WINDOW_THETA,
    windowed_p_star: int = DEFAULT_WINDOWED_P_STAR,
    checkpoint_times: list[float] | None = None,
):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if direct_only and strategy == "paired":
        strategy = "direct"
    # "direct_only" keeps its historical CSV meaning: no online-RKE shadow
    # runs alongside the state-advancing solve.
    direct_only = strategy != "paired"
    windowed = strategy == "windowed"

    if checkpoint_times is None:
        checkpoint_times = [t_end]
    checkpoint_times = sorted(float(value) for value in checkpoint_times)
    if not checkpoint_times or any(
        value <= 0.0 or value > t_end * (1.0 + 1e-12)
        for value in checkpoint_times
    ):
        raise ValueError("checkpoint times must lie in (0, t_end]")
    if abs(checkpoint_times[-1] - t_end) > 1e-12 * t_end:
        checkpoint_times.append(t_end)

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
    # (direct/paired strategies).  The windowed strategy replaces this with
    # the certificate bound theta = lam * h <= Theta, i.e. an unquantized
    # lambda with a far smaller dt floor.
    if windowed:
        dt_theta_floor = (leaf_extent / window_theta) ** 2
    else:
        dt_theta_floor = (leaf_extent / theta_max) ** 2
    node_gap = transport.minimum_node_gap

    cache_dir.mkdir(parents=True, exist_ok=True)
    chemo_cache = cache_dir / f"ks-chemo-{case_id}.sqlite"
    rke_cache = cache_dir / f"ks-rke-{case_id}.sqlite"
    windowed_family_cache = cache_dir / f"ks-windowed-channels-{case_id}.sqlite"
    windowed_registered_cache = (
        cache_dir / f"ks-windowed-registered-{case_id}.sqlite"
    )
    for path in (chemo_cache, rke_cache, windowed_registered_cache):
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

    windowed_channel_build_s = 0.0
    if windowed:
        print(
            f"[{case_id}] building windowed channel family "
            f"(Theta={window_theta:g}, p*={windowed_p_star})",
            flush=True,
        )
        windowed_channel_build_s = _prepare_windowed_channel_family(
            windowed_family_cache,
            q_order,
            leaf_level,
            root_extent,
            window_theta,
            windowed_p_star,
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

    min_theta_seen = math.inf
    dt_min_seen = math.inf
    dt_max_seen = 0.0
    binding_histogram: dict[str, int] = {}
    windowed_totals = {
        "assemble_s": 0.0,
        "register_s": 0.0,
        "load_s": 0.0,
        "wrangler_build_s": 0.0,
        "solve_s": 0.0,
    }
    n_windowed_refused = 0
    n_windowed_failed = 0
    windowed_strategy_total_s = windowed_channel_build_s
    checkpoint_index = 0
    checkpoint_fields: list[tuple[float, np.ndarray]] = []

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
            stop_reason = (
                "windowed_theta_floor_exit" if windowed
                else "theta_regime_exit"
            )
            binding_histogram["theta_floor"] = (
                binding_histogram.get("theta_floor", 0) + 1
            )
            print(
                f"[{case_id}] step {step}: CFL dt {dt_cfl:.3e} below theta "
                f"floor {dt_theta_floor:.3e}; stopping (reported, not "
                "silently accepted)",
                flush=True,
            )
            break

        next_checkpoint = checkpoint_times[checkpoint_index]
        remaining = next_checkpoint - t
        binding_constraint = ""
        if windowed:
            # Unquantized lambda: every admissible dt is allowed; the only
            # parameter-side bound is the windowed certificate theta <= Theta,
            # i.e. dt >= (h / Theta)^2, checked above and re-checked below.
            dt_cap = min(dt_cfl, dt_max)
            endpoint_adjusted = False
            if remaining <= dt_cap * (1.0 + 1.0e-12):
                dt = remaining
                terminal_step = True
                binding_constraint = "checkpoint"
            elif remaining - dt_cap < dt_theta_floor:
                # Never strand a remainder below the certificate floor.
                dt = 0.5 * remaining
                terminal_step = False
                binding_constraint = "checkpoint_split"
                endpoint_adjusted = True
            else:
                dt = dt_cap
                terminal_step = False
                binding_constraint = (
                    "cfl" if dt_cfl <= dt_max else "dt_max"
                )
            if dt < dt_theta_floor * (1.0 - 1.0e-12):
                stop_reason = "windowed_theta_floor_exit"
                binding_constraint = "theta_floor"
                binding_histogram["theta_floor"] = (
                    binding_histogram.get("theta_floor", 0) + 1
                )
                print(
                    f"[{case_id}] step {step}: dt {dt:.3e} below the "
                    f"windowed certificate floor {dt_theta_floor:.3e} "
                    f"(Theta={window_theta:g}); stopping",
                    flush=True,
                )
                break
            binding_histogram[binding_constraint] = (
                binding_histogram.get(binding_constraint, 0) + 1
            )
        else:
            step_plan = _plan_time_step(
                remaining,
                min(dt_cfl, dt_max),
                dt_theta_floor,
                ladder_ratio,
            )
            if step_plan is None:
                stop_reason = "endpoint_plan_exit"
                print(
                    f"[{case_id}] step {step}: no step in "
                    f"[{dt_theta_floor:.3e}, {min(dt_cfl, dt_max):.3e}] can "
                    f"reach the next checkpoint {next_checkpoint:.6g}; "
                    "stopping",
                    flush=True,
                )
                break
            dt, terminal_step, endpoint_adjusted = step_plan
        lam = 1.0 / math.sqrt(dt)
        theta = lam * leaf_extent
        if not windowed and theta > theta_max * (1.0 + 1.0e-14):
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
        if windowed and theta > window_theta * (1.0 + 1.0e-12):
            stop_reason = "windowed_theta_floor_exit"
            print(
                f"[{case_id}] step {step}: lambda={lam:.6g} gives "
                f"theta={theta:.3f} above the declaration "
                f"Theta={window_theta:g}; stopping",
                flush=True,
            )
            break
        max_theta_seen = max(max_theta_seen, theta)
        min_theta_seen = min(min_theta_seen, theta)
        dt_min_seen = min(dt_min_seen, dt)
        dt_max_seen = max(dt_max_seen, dt)
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

        # windowed strategy (E4): per-step offline assembly at the exact
        # (unquantized) lambda, consumed through the standard table-manager
        # path; the state advances through this table.
        windowed_info = None
        windowed_wrangler = None
        windowed_wrangler_build_s = 0.0
        if windowed:
            windowed_table, windowed_info = _provision_windowed_yukawa_table(
                queue,
                windowed_family_cache,
                windowed_registered_cache,
                q_order,
                lam,
                leaf_level,
                root_extent,
                window_theta,
                windowed_p_star,
            )
            if windowed_info["status"] != "ok":
                if windowed_info["status"] == "refused":
                    n_windowed_refused += 1
                    stop_reason = "windowed_certificate_refusal"
                else:
                    n_windowed_failed += 1
                    stop_reason = "windowed_provisioning_failed"
                admissible = False
                windowed_strategy_total_s += (
                    windowed_info["assemble_s"]
                    + windowed_info["register_s"]
                    + windowed_info["load_s"]
                )
                print(
                    f"[{case_id}] step {step}: windowed table provisioning "
                    f"{windowed_info['status']} at lambda={lam:.6g} "
                    f"(theta={theta:.3f}): {windowed_info['detail']}; "
                    "stopping",
                    flush=True,
                )
                break
            windowed_totals["assemble_s"] += windowed_info["assemble_s"]
            windowed_totals["register_s"] += windowed_info["register_s"]
            windowed_totals["load_s"] += windowed_info["load_s"]
            wrangler_start = time.perf_counter()
            windowed_wrangler, _, _ = _build_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel="Yukawa",
                parameter=lam,
                table=windowed_table,
                source_weights=q_weights,
                q_points=q_points,
                source_values_host=rho,
                split=False,
                split_order=split_order,
            )
            windowed_wrangler_build_s = time.perf_counter() - wrangler_start
            windowed_totals["wrangler_build_s"] += windowed_wrangler_build_s

        # direct strategy: per-lambda table (built once per ladder value)
        direct_table_build_s = 0.0
        direct_wrangler_build_s = 0.0
        if not windowed and lam not in direct_wranglers:
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
        windowed_solve_s = 0.0
        if windowed:
            strategy_order = "windowed"
            (rho_windowed,), windowed_solve_s = _drive(
                queue, traversal, windowed_wrangler, weighted, source_vals
            )
            windowed_totals["solve_s"] += windowed_solve_s
            # downstream state advance and diagnostics read rho_direct
            rho_direct = rho_windowed
            direct_solve_s = 0.0
        elif direct_only:
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

        # advance the state (the windowed solution when the windowed
        # strategy is active, the direct solution otherwise)
        rho = rho_direct
        landed_checkpoint = terminal_step
        t = next_checkpoint if terminal_step else t + dt
        step += 1
        if landed_checkpoint:
            checkpoint_fields.append((t, rho.copy()))
            field_snapshots[f"checkpoint{checkpoint_index}"] = rho.copy()
            checkpoint_index += 1

        direct_step_s = (
            direct_table_build_s + direct_wrangler_build_s + direct_solve_s
        )
        rke_step_s = rke_wrangler_build_s + rke_solve_s
        direct_strategy_total_s += direct_step_s
        rke_strategy_total_s += rke_step_s
        if windowed:
            windowed_strategy_total_s += (
                windowed_info["assemble_s"]
                + windowed_info["register_s"]
                + windowed_info["load_s"]
                + windowed_wrangler_build_s
                + windowed_solve_s
            )
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
                "strategy": strategy,
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
                "direct_table_build_s": (
                    "" if windowed else direct_table_build_s
                ),
                "direct_wrangler_build_s": (
                    "" if windowed else direct_wrangler_build_s
                ),
                "direct_solve_s": "" if windowed else direct_solve_s,
                "rke_wrangler_build_s": (
                    "" if direct_only else rke_wrangler_build_s
                ),
                "rke_solve_s": "" if direct_only else rke_solve_s,
                "rke_vs_direct_weighted_rel_l2": (
                    "" if direct_only else mismatch
                ),
                "cumulative_direct_strategy_s": (
                    "" if windowed else direct_strategy_total_s
                ),
                "cumulative_rke_strategy_s": (
                    "" if direct_only else rke_strategy_total_s
                ),
                "strategy_cost_gap_direct_minus_rke_s": (
                    "" if direct_only else cost_gap
                ),
                "binding_constraint": binding_constraint,
                "checkpoint_landed": int(landed_checkpoint),
                "windowed_status": (
                    windowed_info["status"] if windowed else ""
                ),
                "windowed_condition_number": (
                    windowed_info["condition_number"] if windowed else ""
                ),
                "windowed_assemble_s": (
                    windowed_info["assemble_s"] if windowed else ""
                ),
                "windowed_register_s": (
                    windowed_info["register_s"] if windowed else ""
                ),
                "windowed_table_load_s": (
                    windowed_info["load_s"] if windowed else ""
                ),
                "windowed_wrangler_build_s": (
                    windowed_wrangler_build_s if windowed else ""
                ),
                "windowed_solve_s": (
                    windowed_solve_s if windowed else ""
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

    checkpoint_data = {
        "checkpoints": checkpoint_fields,
        "weights": weights_host,
    }

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
    if windowed:
        theta_floor_bound_steps = binding_histogram.get("theta_floor", 0)
        if theta_floor_bound_steps:
            binding_verdict = "theta_floor_binds"
        elif binding_histogram.get("cfl", 0):
            binding_verdict = (
                "cfl_cap_binds_step_size:constraint_relief_only_"
                "no_measured_step_size_gain"
            )
        elif binding_histogram.get("dt_max", 0):
            binding_verdict = (
                "dt_max_cap_binds_step_size:constraint_relief_only_"
                "no_measured_step_size_gain"
            )
        else:
            binding_verdict = "checkpoint_landing_bound_only"
        windowed_provisioning_total_s = (
            windowed_totals["assemble_s"]
            + windowed_totals["register_s"]
            + windowed_totals["load_s"]
            + windowed_totals["wrangler_build_s"]
        )

    summary_row = {
        "case_id": case_id,
        "mode": mode,
        "strategy": strategy,
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
        "direct_table_build_total_s": (
            "" if windowed else direct_table_build_total_s
        ),
        "chemo_solve_total_s": chemo_solve_total_s,
        "direct_strategy_total_s": (
            "" if windowed else direct_strategy_total_s
        ),
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
            (
                "windowed=one channel family build+per-step offline assembly"
                "+registration+standard cache load+wrangler build+solve;"
                "state advances through the windowed-assembled tables;"
                "shared fixed-alpha chemoattractant pass, conservative "
                "transport, diagnostics, and output excluded"
            )
            if windowed
            else (
                "direct=per-lambda table build+wrangler build+solve;"
                "rke=one channel family build+per-lambda wrangler build+solve;"
                "shared fixed-alpha chemoattractant pass, conservative "
                "transport,"
                "diagnostics, and output excluded from both;"
                "solve order alternates"
            )
        ),
        "window_theta": window_theta if windowed else "",
        "windowed_p_star": windowed_p_star if windowed else "",
        "min_theta_seen": (
            min_theta_seen if math.isfinite(min_theta_seen) else ""
        ),
        "dt_min": dt_min_seen if math.isfinite(dt_min_seen) else "",
        "dt_max_seen": dt_max_seen if dt_max_seen > 0.0 else "",
        "lambda_quantization": (
            "continuous_unquantized" if windowed
            else f"geometric_ladder_ratio_{ladder_ratio:.10g}"
        ),
        "binding_constraint_histogram_json": (
            json.dumps(binding_histogram, sort_keys=True,
                       separators=(",", ":"))
            if windowed else ""
        ),
        "theta_floor_bound_steps": (
            theta_floor_bound_steps if windowed else ""
        ),
        "go_no_go_binding_verdict": binding_verdict if windowed else "",
        "windowed_channel_build_s": (
            windowed_channel_build_s if windowed else ""
        ),
        "windowed_assemble_total_s": (
            windowed_totals["assemble_s"] if windowed else ""
        ),
        "windowed_register_total_s": (
            windowed_totals["register_s"] if windowed else ""
        ),
        "windowed_table_load_total_s": (
            windowed_totals["load_s"] if windowed else ""
        ),
        "windowed_wrangler_build_total_s": (
            windowed_totals["wrangler_build_s"] if windowed else ""
        ),
        "windowed_solve_total_s": (
            windowed_totals["solve_s"] if windowed else ""
        ),
        "windowed_strategy_total_s": (
            windowed_strategy_total_s if windowed else ""
        ),
        "windowed_mean_provisioning_s_per_step": (
            (windowed_provisioning_total_s / max(step, 1))
            if windowed else ""
        ),
        "n_windowed_refused": n_windowed_refused if windowed else "",
        "n_windowed_failed": n_windowed_failed if windowed else "",
        "checkpoint_times_json": json.dumps(
            checkpoint_times, separators=(",", ":")
        ),
        "checkpoint_agreement_json": "",
        "max_checkpoint_rel_l2": "",
        "baseline_case_id": "",
    }
    return step_rows, summary_row, checkpoint_data


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


CHECKPOINT_FIELDS = (
    "case_id",
    "baseline_case_id",
    "mode",
    "checkpoint_time",
    "windowed_vs_direct_weighted_rel_l2",
    "baseline_mass",
    "windowed_mass",
    "baseline_rho_max",
    "windowed_rho_max",
)


def _compare_checkpoints(
    baseline_data: dict[str, Any],
    windowed_data: dict[str, Any],
    *,
    case_id: str,
    baseline_case_id: str,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Weighted relative L2 agreement of the windowed trajectory against the
    resolved direct baseline at the checkpoints both runs reached."""
    weights = np.asarray(baseline_data["weights"])
    baseline_by_time = {
        round(t, 14): field for t, field in baseline_data["checkpoints"]
    }
    rows = []
    agreement = []
    for t, windowed_field in windowed_data["checkpoints"]:
        baseline_field = baseline_by_time.get(round(t, 14))
        if baseline_field is None:
            continue
        difference = np.asarray(windowed_field) - np.asarray(baseline_field)
        reference = max(
            float(np.sqrt(np.sum(weights * np.asarray(baseline_field) ** 2))),
            1e-300,
        )
        rel_l2 = float(np.sqrt(np.sum(weights * difference**2))) / reference
        rows.append(
            {
                "case_id": case_id,
                "baseline_case_id": baseline_case_id,
                "mode": mode,
                "checkpoint_time": t,
                "windowed_vs_direct_weighted_rel_l2": rel_l2,
                "baseline_mass": float(np.sum(weights * baseline_field)),
                "windowed_mass": float(np.sum(weights * windowed_field)),
                "baseline_rho_max": float(np.max(baseline_field)),
                "windowed_rho_max": float(np.max(windowed_field)),
            }
        )
        agreement.append({"time": t, "rel_l2": rel_l2})
    return rows, agreement


def _apply_pair_outcome(summary_rows: list[dict[str, Any]]) -> None:
    """Sub/supercritical pair verdict, applied per strategy group."""
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_strategy.setdefault(row["strategy"], []).append(row)
    for group in by_strategy.values():
        if len(group) != 2:
            continue
        subcritical = next(
            (
                row for row in group
                if row["regime"] == "below_8pi_reference"
            ),
            None,
        )
        supercritical = next(
            (
                row for row in group
                if row["regime"] == "above_8pi_reference"
            ),
            None,
        )
        if subcritical is None or supercritical is None:
            continue
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
        for row in group:
            row["pair_outcome_pass"] = int(pair_pass)
            row["pair_moment_ratio_separation"] = separation


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
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="paired",
        help=(
            "'paired' (state via direct, online-RKE shadow), 'direct' "
            "(no shadow), or 'windowed' (E4: state advances through "
            "windowed offline-assembled tables at an unquantized lambda, "
            "with a direct-strategy baseline run at its quantized ladder "
            "for comparison at shared checkpoints)"
        ),
    )
    parser.add_argument(
        "--window-theta", type=float, default=DEFAULT_WINDOW_THETA,
        help="declared window Theta certifying theta = lambda h <= Theta",
    )
    parser.add_argument(
        "--windowed-p-star", type=int, default=DEFAULT_WINDOWED_P_STAR,
    )
    parser.add_argument(
        "--checkpoint-fractions",
        help=(
            "comma-separated fractions of t_end at which trajectories are "
            "forced to land (and, for --strategy windowed, compared); "
            "default '1.0' in smoke and '0.25,0.5,0.75,1.0' in full mode "
            "for the windowed strategy, '1.0' otherwise"
        ),
    )
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
    if not (math.isfinite(args.window_theta) and args.window_theta > 0.0):
        # run_case() forms (leaf_extent / Theta)**2 for the windowed
        # strategy, so a zero raises ZeroDivisionError -- and only after
        # the geometry and the fixed-table setup are underway.  A negative
        # Theta certifies a meaningless declaration and fails later still.
        parser.error("--window-theta must be finite and positive")
    if args.windowed_p_star < 1:
        parser.error("--windowed-p-star must be >= 1")

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

    strategy = args.strategy
    if args.direct_only:
        if strategy == "windowed":
            parser.error("--direct-only conflicts with --strategy windowed")
        strategy = "direct"

    if args.checkpoint_fractions is not None:
        fractions = [
            float(part.strip())
            for part in args.checkpoint_fractions.split(",")
            if part.strip()
        ]
        if not fractions or any(
            not (0.0 < fraction <= 1.0) for fraction in fractions
        ):
            parser.error("checkpoint fractions must lie in (0, 1]")
        if len(set(fractions)) != len(fractions):
            parser.error("checkpoint fractions must be unique")
    elif strategy == "windowed" and not smoke:
        fractions = [0.25, 0.5, 0.75, 1.0]
    else:
        fractions = [1.0]
    checkpoint_times = sorted(fraction * t_end for fraction in fractions)

    device = _select_opencl_device(cl, args.backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    step_rows = []
    summary_rows = []
    checkpoint_rows = []
    windowed_failure_messages = []
    for mass_factor in args.mass_factors:
        case_id = (
            f"ks2d-{args.initial_profile}-q{q_order}-l{nlevels}"
            f"-a{args.alpha:g}-m{mass_factor:g}"
        )
        shared_case_kwargs = dict(
            mode=args.mode,
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
            direct_only=False,
            save_fields=(
                (args.out_dir / "fields") if args.save_fields else None
            ),
            window_theta=args.window_theta,
            windowed_p_star=args.windowed_p_star,
            checkpoint_times=checkpoint_times,
        )

        if strategy == "windowed":
            case_specs = [
                (f"{case_id}-direct-baseline", "direct"),
                (f"{case_id}-windowed", "windowed"),
            ]
        else:
            case_specs = [(case_id, strategy)]

        case_results = {}
        for run_case_id, run_strategy in case_specs:
            case_steps, case_summary, case_checkpoints = run_case(
                ctx,
                queue,
                case_id=run_case_id,
                strategy=run_strategy,
                **shared_case_kwargs,
            )
            step_rows.extend(case_steps)
            summary_rows.append(case_summary)
            case_results[run_strategy] = (case_summary, case_checkpoints)
            print(
                f"[{run_case_id}] done: {case_summary['n_steps']} steps, "
                f"stop={case_summary['stop_reason']}, "
                f"rho_max_ratio={case_summary['rho_max_ratio']:.2f}",
                flush=True,
            )

        if strategy == "windowed":
            baseline_summary, baseline_checkpoints = case_results["direct"]
            windowed_summary, windowed_checkpoints = case_results["windowed"]
            pair_rows, agreement = _compare_checkpoints(
                baseline_checkpoints,
                windowed_checkpoints,
                case_id=windowed_summary["case_id"],
                baseline_case_id=baseline_summary["case_id"],
                mode=args.mode,
            )
            checkpoint_rows.extend(pair_rows)
            windowed_summary["baseline_case_id"] = baseline_summary["case_id"]
            windowed_summary["checkpoint_agreement_json"] = json.dumps(
                agreement, separators=(",", ":")
            )
            if agreement:
                windowed_summary["max_checkpoint_rel_l2"] = max(
                    entry["rel_l2"] for entry in agreement
                )
                print(
                    f"[{windowed_summary['case_id']}] checkpoint agreement "
                    "vs direct baseline: "
                    + ", ".join(
                        f"t={entry['time']:.4g}: {entry['rel_l2']:.3e}"
                        for entry in agreement
                    ),
                    flush=True,
                )
            else:
                print(
                    f"[{windowed_summary['case_id']}] no shared checkpoints "
                    "reached by both runs (baseline stop: "
                    f"{baseline_summary['stop_reason']}, windowed stop: "
                    f"{windowed_summary['stop_reason']})",
                    flush=True,
                )
            print(
                f"[{windowed_summary['case_id']}] binding-constraint "
                f"verdict: {windowed_summary['go_no_go_binding_verdict']} "
                f"(histogram "
                f"{windowed_summary['binding_constraint_histogram_json']})",
                flush=True,
            )
            if int(windowed_summary["n_windowed_failed"] or 0) > 0:
                windowed_failure_messages.append(
                    f"{windowed_summary['case_id']}: windowed provisioning "
                    "failed"
                )
            if int(windowed_summary["n_windowed_refused"] or 0) > 0:
                windowed_failure_messages.append(
                    f"{windowed_summary['case_id']}: windowed certificate "
                    "refused inside the declared window"
                )

    _apply_pair_outcome(summary_rows)

    _write_csv(args.out_dir / "ks_steps.csv", STEP_FIELDS, step_rows)
    _write_csv(args.out_dir / "ks_summary.csv", SUMMARY_FIELDS, summary_rows)
    if strategy == "windowed":
        _write_csv(
            args.out_dir / "ks_windowed_checkpoints.csv",
            CHECKPOINT_FIELDS,
            checkpoint_rows,
        )
    if windowed_failure_messages:
        for message in windowed_failure_messages:
            print(f"FAILURE: {message}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
