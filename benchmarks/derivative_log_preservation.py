#!/usr/bin/env python3
"""Emit derivative/logarithmic FMM-preservation CSVs for Paper 1.

Two manufactured-solution studies close the FMM-preservation coverage gaps
stated in the manuscript:

* ``gradient3d``: the smooth 3D Gaussian Poisson problem evaluated through
  target-derivative tables (``Laplace-Dx/Dy/Dz``).  The canonical path uses one
  level-0 table per component with the inferred first-order scale law
  ``h / h_0``; the per-level path builds direct tables at every source-box
  level.  Errors are measured against the finite-box gradient reference
  obtained by differentiating the Green-identity boundary correction.

* ``log2d``: a smooth 2D Gaussian Poisson problem for the logarithmic kernel.
  The canonical path uses one level-0 table with the separable moment
  correction (``h^2`` scale plus ``-(1/2 pi) log(h/h_0)`` times the source-mode
  moment); the per-level path builds direct tables at every level.  Errors are
  measured against the finite-box Green-identity reference.

Both studies also run the direct point-to-point diagnostic through the same
volume-FMM driver.  Smoke mode is sized for CI/local checks; full mode is
intended for a controlled remote paper run.  ``--q-orders`` and ``--n-levels``
form a Cartesian product per study.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymbolic as pmbl
import pyopencl as cl

from volumential.tools import ScalarFieldExpressionEvaluation as Eval


ROOT_BOUNDS = (-0.5, 0.5)

GAUSSIAN_COMPONENTS_3D = (
    (1.0, 120.0, (-0.08, 0.06, -0.05)),
    (-0.65, 90.0, (0.09, -0.07, 0.08)),
)

GAUSSIAN_COMPONENTS_2D = (
    (1.0, 120.0, (-0.08, 0.06)),
    (-0.65, 90.0, (0.09, -0.07)),
)


@dataclass(frozen=True)
class PreservationCase:
    study: str  # "gradient3d" | "log2d"
    q_order: int
    n_levels: int
    fmm_order: int
    regular_quad_order: int
    radial_quad_order: int

    @property
    def dim(self) -> int:
        return 3 if self.study == "gradient3d" else 2


SMOKE_CASES = (
    PreservationCase(
        study="gradient3d",
        q_order=2,
        n_levels=2,
        fmm_order=8,
        regular_quad_order=8,
        radial_quad_order=25,
    ),
    PreservationCase(
        study="log2d",
        q_order=3,
        n_levels=3,
        fmm_order=10,
        regular_quad_order=10,
        radial_quad_order=35,
    ),
)

FULL_CASES = (
    PreservationCase("gradient3d", 2, 3, 14, 14, 55),
    PreservationCase("gradient3d", 3, 2, 14, 14, 55),
    PreservationCase("gradient3d", 3, 3, 14, 14, 55),
    PreservationCase("gradient3d", 3, 4, 14, 14, 55),
    PreservationCase("gradient3d", 4, 3, 14, 14, 55),
    PreservationCase("log2d", 2, 4, 14, 14, 55),
    PreservationCase("log2d", 3, 3, 14, 14, 55),
    PreservationCase("log2d", 3, 4, 14, 14, 55),
    PreservationCase("log2d", 3, 5, 14, 14, 55),
    PreservationCase("log2d", 3, 6, 14, 14, 55),
    PreservationCase("log2d", 4, 4, 14, 14, 55),
)

FIELDS = (
    "study",
    "case_id",
    "mode",
    "problem",
    "dim",
    "kernel",
    "kernel_normalization",
    "path",
    "component",
    "reference_path",
    "weighted_l2_definition",
    "q_order",
    "n_levels",
    "h_max",
    "fmm_order",
    "regular_quad_order",
    "radial_quad_order",
    "reference_quad_order",
    "reference_check_quad_order",
    "n_targets",
    "wall_s",
    "reference_boundary_weighted_l2",
    "reference_quad_delta_linf",
    "rel_l2_vs_exact",
    "weighted_l2_vs_exact",
    "weighted_rel_l2_vs_exact",
    "linf_vs_exact",
    "h_observed_order_vs_exact",
    "rel_l2_vs_canonical",
    "weighted_rel_l2_vs_canonical",
    "linf_vs_canonical",
    "rel_l2_vs_direct",
    "linf_vs_direct",
)


# {{{ manufactured problem

def _gaussian_components(dim):
    return GAUSSIAN_COMPONENTS_3D if dim == 3 else GAUSSIAN_COMPONENTS_2D


def _build_source_expression(dim):
    names = ("x", "y", "z")[:dim]
    variables = [pmbl.var(name) for name in names]
    expp = pmbl.var("exp")

    source_expr = 0
    for amplitude, alpha, center in _gaussian_components(dim):
        radius_sq = sum(
            (variable - center_i) ** 2
            for variable, center_i in zip(variables, center, strict=True)
        )
        gaussian = amplitude * expp(-alpha * radius_sq)
        # f = -Delta u; sumpy Laplace kernels carry the free-space Green's
        # function normalization, so no extra constant appears here.
        source_expr += (2 * dim * alpha - 4 * alpha**2 * radius_sq) * gaussian

    return source_expr, variables


def _evaluate_manufactured_solution(points, dim):
    """Return (u, grad u) for the manufactured Gaussian sum."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"points must have shape (npoints, {dim})")

    values = np.zeros(points.shape[0], dtype=np.float64)
    gradients = np.zeros_like(points)
    for amplitude, alpha, center in _gaussian_components(dim):
        offsets = points - np.asarray(center, dtype=np.float64)
        component = amplitude * np.exp(-alpha * np.sum(offsets**2, axis=1))
        values += component
        gradients += (-2.0 * alpha * component)[:, np.newaxis] * offsets
    return values, gradients

# }}}


# {{{ finite-box references via Green's identity

def _boundary_faces(bbox, quad_order, dim):
    """Yield (points, weights, normal_axis, normal_sign) per boundary face."""
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    for normal_axis in range(dim):
        tangential_axes = [axis for axis in range(dim) if axis != normal_axis]
        mapped_nodes = []
        mapped_weights = []
        for axis in tangential_axes:
            lo, hi = bbox[axis]
            mapped_nodes.append(0.5 * ((hi - lo) * nodes + hi + lo))
            mapped_weights.append(0.5 * (hi - lo) * weights)
        if dim == 2:
            face_points_t = mapped_nodes[0][:, np.newaxis]
            face_weights = mapped_weights[0]
        else:
            grids = np.meshgrid(*mapped_nodes, indexing="ij")
            weight_grids = np.meshgrid(*mapped_weights, indexing="ij")
            face_points_t = np.stack([g.ravel() for g in grids], axis=-1)
            face_weights = np.prod(np.asarray(weight_grids), axis=0).ravel()

        for side, normal_sign in ((0, -1.0), (1, 1.0)):
            points = np.empty((face_points_t.shape[0], dim), dtype=np.float64)
            points[:, normal_axis] = bbox[normal_axis, side]
            for column, axis in enumerate(tangential_axes):
                points[:, axis] = face_points_t[:, column]
            yield points, face_weights, normal_axis, normal_sign


def _finite_box_reference(targets, bbox, quad_order, dim, *, chunk_size=256):
    """Potential and gradient of the finite-box volume potential.

    Green's second identity for ``f = -Delta u`` gives

    ``integral_box G f = u + integral_boundary (u d_n G - G d_n u)``

    with ``G = 1/(4 pi r)`` in 3D and ``G = -log(r)/(2 pi)`` in 2D.  The
    gradient reference differentiates the boundary integrand in the target
    coordinate; all integrands are nonsingular because volume quadrature
    targets are strictly interior.
    """
    targets = np.asarray(targets, dtype=np.float64)
    solution, solution_gradient = _evaluate_manufactured_solution(targets, dim)
    potential_correction = np.zeros(targets.shape[0], dtype=np.float64)
    gradient_correction = np.zeros((targets.shape[0], dim), dtype=np.float64)

    kernel_scale = 1.0 / (4.0 * np.pi) if dim == 3 else 1.0 / (2.0 * np.pi)

    for face_points, face_weights, normal_axis, normal_sign in _boundary_faces(
        np.asarray(bbox, dtype=np.float64), quad_order, dim
    ):
        face_solution, face_gradient = _evaluate_manufactured_solution(
            face_points, dim
        )
        normal_derivative = normal_sign * face_gradient[:, normal_axis]

        for start in range(0, targets.shape[0], chunk_size):
            stop = min(start + chunk_size, targets.shape[0])
            displacement = (
                targets[start:stop, np.newaxis, :] - face_points[np.newaxis, :, :]
            )
            radius_sq = np.sum(displacement**2, axis=2)

            if dim == 3:
                radius = np.sqrt(radius_sq)
                green = kernel_scale / radius
                green_grad = (
                    -kernel_scale * displacement / radius_sq[:, :, np.newaxis] ** 1.5
                )
                dn_green = (
                    kernel_scale
                    * normal_sign
                    * displacement[:, :, normal_axis]
                    / radius_sq**1.5
                )
                # d/dx_i of dn_green:
                # ns * ks * (delta_{i,a} / r^3 - 3 d_i d_a / r^5)
                dn_green_grad = (
                    -3.0
                    * displacement
                    * displacement[:, :, normal_axis][:, :, np.newaxis]
                    / radius_sq[:, :, np.newaxis] ** 2.5
                )
                dn_green_grad[:, :, normal_axis] += 1.0 / radius_sq**1.5
                dn_green_grad *= kernel_scale * normal_sign
            else:
                green = -0.5 * kernel_scale * np.log(radius_sq)
                green_grad = (
                    -kernel_scale * displacement / radius_sq[:, :, np.newaxis]
                )
                dn_green = (
                    kernel_scale
                    * normal_sign
                    * displacement[:, :, normal_axis]
                    / radius_sq
                )
                # d/dx_i of dn_green:
                # ns * ks * (delta_{i,a} / r^2 - 2 d_i d_a / r^4)
                dn_green_grad = (
                    -2.0
                    * displacement
                    * displacement[:, :, normal_axis][:, :, np.newaxis]
                    / radius_sq[:, :, np.newaxis] ** 2
                )
                dn_green_grad[:, :, normal_axis] += 1.0 / radius_sq
                dn_green_grad *= kernel_scale * normal_sign

            potential_correction[start:stop] += (
                face_solution[np.newaxis, :] * dn_green
                - green * normal_derivative[np.newaxis, :]
            ) @ face_weights
            gradient_correction[start:stop] += np.einsum(
                "tfi,f->ti",
                face_solution[np.newaxis, :, np.newaxis] * dn_green_grad
                - green_grad * normal_derivative[np.newaxis, :, np.newaxis],
                face_weights,
            )

    return (
        solution + potential_correction,
        solution_gradient + gradient_correction,
        potential_correction,
    )

# }}}


# {{{ geometry, tables, wranglers

def _build_geometry(ctx, queue, case):
    import volumential.meshgen as mg

    dim = case.dim
    a, b = ROOT_BOUNDS
    mesh_cls = mg.MeshGen3D if dim == 3 else mg.MeshGen2D
    mesh = mesh_cls(case.q_order, case.n_levels, a, b, queue=queue)
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        dim,
        case.q_order,
        mesh,
        bbox=np.array([[a, b]] * dim, dtype=np.float64),
    )
    h_max = float(np.max(mesh.get_cell_measures()) ** (1.0 / dim))
    return q_points, q_weights, tree, traversal, h_max


def _kernel_types_for_study(study, dim):
    if study == "gradient3d":
        return ["Laplace-Dx", "Laplace-Dy", "Laplace-Dz"][:dim]
    return ["Laplace"]


def _out_kernels_for_study(study, dim):
    from sumpy.kernel import AxisTargetDerivative, LaplaceKernel

    base = LaplaceKernel(dim)
    if study == "gradient3d":
        return base, [AxisTargetDerivative(axis, base) for axis in range(dim)]
    return base, [base]


def _build_tables(queue, tree, case, cache_dir, path):
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"{case.study}-{path}-q{case.q_order}-l{case.n_levels}.sqlite"
    )
    if cache_path.exists():
        cache_path.unlink()

    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=case.regular_quad_order,
        radial_quad_order=case.radial_quad_order,
    )

    kernel_types = _kernel_types_for_study(case.study, case.dim)
    tables_per_kernel = []
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=1.0, queue=queue
    ) as tm:
        for kernel_type in kernel_types:
            if path == "per_level":
                tables = []
                for level in range(tree.nlevels + 1):
                    table, _ = tm.get_table(
                        case.dim,
                        kernel_type,
                        case.q_order,
                        source_box_level=level,
                        queue=queue,
                        build_config=build_config,
                    )
                    tables.append(table)
                tables_per_kernel.append(tables)
            else:
                table, _ = tm.get_table(
                    case.dim,
                    kernel_type,
                    case.q_order,
                    queue=queue,
                    build_config=build_config,
                )
                tables_per_kernel.append([table])
    return tables_per_kernel


def _build_wrangler(ctx, queue, traversal, tables_per_kernel, case):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    base_kernel, out_kernels = _out_kernels_for_study(case.study, case.dim)
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
        repr(out_knl): tables
        for out_knl, tables in zip(out_kernels, tables_per_kernel, strict=True)
    }
    target_to_source = np.arange(traversal.tree.ntargets, dtype=np.int32)
    return FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=near_field_table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: case.fmm_order,
        quad_order=case.q_order,
        self_extra_kwargs={"target_to_source": target_to_source},
    )


def _run_path(
    *,
    ctx,
    queue,
    traversal,
    tables_per_kernel,
    case,
    source_strengths,
    source_vals,
    direct_evaluation,
):
    from volumential.volume_fmm import drive_volume_fmm

    wrangler = _build_wrangler(ctx, queue, traversal, tables_per_kernel, case)
    queue.finish()
    start = time.perf_counter()
    outputs = drive_volume_fmm(
        traversal,
        wrangler,
        source_strengths,
        source_vals,
        direct_evaluation=direct_evaluation,
        list1_only=False,
    )
    queue.finish()
    wall_s = time.perf_counter() - start
    return [output.get() for output in outputs], wall_s

# }}}


# {{{ rows

def _metrics(diff, reference, weights):
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-300)
    weighted_reference_norm = max(
        float(np.sqrt(np.sum(weights * np.abs(reference) ** 2))), 1.0e-300
    )
    weighted_error = float(np.sqrt(np.sum(weights * np.abs(diff) ** 2)))
    return {
        "rel_l2": float(np.linalg.norm(diff) / reference_norm),
        "weighted_l2": weighted_error,
        "weighted_rel_l2": weighted_error / weighted_reference_norm,
        "linf": float(np.max(np.abs(diff))),
    }


def _component_rows(
    *,
    mode,
    case,
    path,
    components,  # dict component -> values array
    exact_components,
    canonical_components,
    direct_components,
    weights_map,
    h_max,
    reference_quad_order,
    reference_check_quad_order,
    reference_boundary_weighted_l2,
    reference_quad_delta_linf,
    wall_s,
):
    kernel = "Laplace-gradient" if case.study == "gradient3d" else "Laplace"
    normalization = (
        "sumpy_global_scaling_1_over_4pi_r"
        if case.dim == 3
        else "sumpy_global_scaling_minus_log_r_over_2pi"
    )
    rows = []
    for component, values in components.items():
        exact = exact_components[component]
        canonical = canonical_components[component]
        direct = direct_components[component]
        weights = weights_map[component]
        exact_metrics = _metrics(values - exact, exact, weights)
        canonical_metrics = _metrics(values - canonical, canonical, weights)
        direct_metrics = _metrics(values - direct, direct, weights)
        rows.append(
            {
                "study": case.study,
                "case_id": f"{case.study}-q{case.q_order}-l{case.n_levels}",
                "mode": mode,
                "problem": f"smooth-gaussian-poisson{case.dim}d",
                "dim": case.dim,
                "kernel": kernel,
                "kernel_normalization": normalization,
                "path": path,
                "component": component,
                "reference_path": "finite_box_green_identity",
                "weighted_l2_definition": (
                    "sqrt(sum_i quadrature_weight_i * abs(value_i)**2)"
                ),
                "q_order": case.q_order,
                "n_levels": case.n_levels,
                "h_max": h_max,
                "fmm_order": case.fmm_order,
                "regular_quad_order": case.regular_quad_order,
                "radial_quad_order": case.radial_quad_order,
                "reference_quad_order": reference_quad_order,
                "reference_check_quad_order": reference_check_quad_order,
                "n_targets": int(np.asarray(values).size),
                "wall_s": wall_s,
                "reference_boundary_weighted_l2": reference_boundary_weighted_l2,
                "reference_quad_delta_linf": reference_quad_delta_linf,
                "rel_l2_vs_exact": exact_metrics["rel_l2"],
                "weighted_l2_vs_exact": exact_metrics["weighted_l2"],
                "weighted_rel_l2_vs_exact": exact_metrics["weighted_rel_l2"],
                "linf_vs_exact": exact_metrics["linf"],
                "h_observed_order_vs_exact": "",
                "rel_l2_vs_canonical": canonical_metrics["rel_l2"],
                "weighted_rel_l2_vs_canonical": canonical_metrics[
                    "weighted_rel_l2"
                ],
                "linf_vs_canonical": canonical_metrics["linf"],
                "rel_l2_vs_direct": direct_metrics["rel_l2"],
                "linf_vs_direct": direct_metrics["linf"],
            }
        )
    return rows


def _componentize(case, outputs, exact_potential, exact_gradient, weights):
    """Split path outputs into named components, including vector norms."""
    if case.study == "gradient3d":
        names = ("dx", "dy", "dz")[: case.dim]
        components = dict(zip(names, outputs, strict=True))
        exact = {
            name: exact_gradient[:, axis] for axis, name in enumerate(names)
        }
        # vector norms use stacked arrays so weighted norms sum over
        # components as well as nodes
        components["grad"] = np.concatenate(outputs)
        exact["grad"] = exact_gradient.T.reshape(-1)
        weights_map = {name: weights for name in names}
        weights_map["grad"] = np.tile(weights, case.dim)
        return components, exact, weights_map
    components = {"potential": outputs[0]}
    exact = {"potential": exact_potential}
    return components, exact, {"potential": weights}


def _add_h_convergence(rows):
    groups = {}
    for row in rows:
        if row["component"] not in ("grad", "potential"):
            continue
        key = (
            row["study"],
            row["path"],
            row["component"],
            row["q_order"],
            row["fmm_order"],
            row["regular_quad_order"],
            row["radial_quad_order"],
        )
        groups.setdefault(key, []).append(row)

    for group in groups.values():
        group.sort(key=lambda row: row["h_max"], reverse=True)
        for coarse, fine in zip(group, group[1:], strict=False):
            coarse_h = float(coarse["h_max"])
            fine_h = float(fine["h_max"])
            coarse_error = float(coarse["weighted_rel_l2_vs_exact"])
            fine_error = float(fine["weighted_rel_l2_vs_exact"])
            if coarse_h > fine_h and coarse_error > 0.0 and fine_error > 0.0:
                fine["h_observed_order_vs_exact"] = float(
                    np.log(coarse_error / fine_error) / np.log(coarse_h / fine_h)
                )

# }}}


def run_case(*, ctx, queue, case, mode, cache_dir, reference_quad_order):
    q_points, q_weights, tree, traversal, h_max = _build_geometry(ctx, queue, case)
    source_expr, variables = _build_source_expression(case.dim)
    source_eval = Eval(case.dim, source_expr, variables)
    q_coords = np.array([coords.get() for coords in q_points])
    q_weights_host = q_weights.get(queue)
    source_vals = cl.array.to_device(queue, source_eval(queue, q_coords))
    source_strengths = source_vals * q_weights

    bbox = np.array([ROOT_BOUNDS] * case.dim, dtype=np.float64)
    reference_check_quad_order = max(2, reference_quad_order // 2)
    exact_potential, exact_gradient, reference_boundary = _finite_box_reference(
        q_coords.T, bbox, reference_quad_order, case.dim
    )
    check_potential, check_gradient, _ = _finite_box_reference(
        q_coords.T, bbox, reference_check_quad_order, case.dim
    )
    if case.study == "gradient3d":
        reference_quad_delta_linf = float(
            np.max(np.abs(exact_gradient - check_gradient))
        )
    else:
        reference_quad_delta_linf = float(
            np.max(np.abs(exact_potential - check_potential))
        )

    canonical_tables = _build_tables(queue, tree, case, cache_dir, "canonical")
    per_level_tables = _build_tables(queue, tree, case, cache_dir, "per_level")

    canonical_outputs, canonical_wall_s = _run_path(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        tables_per_kernel=canonical_tables,
        case=case,
        source_strengths=source_strengths,
        source_vals=source_vals,
        direct_evaluation=False,
    )
    per_level_outputs, per_level_wall_s = _run_path(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        tables_per_kernel=per_level_tables,
        case=case,
        source_strengths=source_strengths,
        source_vals=source_vals,
        direct_evaluation=False,
    )
    direct_outputs, direct_wall_s = _run_path(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        tables_per_kernel=per_level_tables,
        case=case,
        source_strengths=source_strengths,
        source_vals=source_vals,
        direct_evaluation=True,
    )

    canonical_components, exact_components, weights_map = _componentize(
        case, canonical_outputs, exact_potential, exact_gradient, q_weights_host
    )
    per_level_components, _, _ = _componentize(
        case, per_level_outputs, exact_potential, exact_gradient, q_weights_host
    )
    direct_components, _, _ = _componentize(
        case, direct_outputs, exact_potential, exact_gradient, q_weights_host
    )

    common = {
        "mode": mode,
        "case": case,
        "exact_components": exact_components,
        "canonical_components": canonical_components,
        "direct_components": direct_components,
        "weights_map": weights_map,
        "h_max": h_max,
        "reference_quad_order": reference_quad_order,
        "reference_check_quad_order": reference_check_quad_order,
        "reference_boundary_weighted_l2": float(
            np.sqrt(np.sum(q_weights_host * np.abs(reference_boundary) ** 2))
        ),
        "reference_quad_delta_linf": reference_quad_delta_linf,
    }
    rows = []
    rows.extend(
        _component_rows(
            path="canonical_rescaled",
            components=canonical_components,
            wall_s=canonical_wall_s,
            **common,
        )
    )
    rows.extend(
        _component_rows(
            path="per_level_tables",
            components=per_level_components,
            wall_s=per_level_wall_s,
            **common,
        )
    )
    rows.extend(
        _component_rows(
            path="direct_p2p",
            components=direct_components,
            wall_s=direct_wall_s,
            **common,
        )
    )
    return rows


def _select_cases(mode, studies, q_orders, n_levels):
    preset = SMOKE_CASES if mode == "smoke" else FULL_CASES
    selected = [case for case in preset if case.study in studies]
    if q_orders is None and n_levels is None:
        return tuple(selected)

    result = []
    for study in studies:
        study_cases = [case for case in selected if case.study == study]
        if not study_cases:
            continue
        template = study_cases[0]
        study_q_orders = (
            sorted({case.q_order for case in study_cases})
            if q_orders is None
            else list(dict.fromkeys(q_orders))
        )
        study_n_levels = (
            sorted({case.n_levels for case in study_cases})
            if n_levels is None
            else list(dict.fromkeys(n_levels))
        )
        for q_order in study_q_orders:
            for level in study_n_levels:
                result.append(
                    PreservationCase(
                        study=study,
                        q_order=q_order,
                        n_levels=level,
                        fmm_order=template.fmm_order,
                        regular_quad_order=template.regular_quad_order,
                        radial_quad_order=template.radial_quad_order,
                    )
                )
    return tuple(result)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=("gradient3d", "log2d"),
        default=["gradient3d", "log2d"],
    )
    parser.add_argument("--q-orders", type=int, nargs="+")
    parser.add_argument("--n-levels", type=int, nargs="+")
    parser.add_argument(
        "--reference-quad-order",
        type=int,
        help="Gauss-Legendre order per boundary-face axis (default: 24/40)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/derivative-log-preservation.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/derivative-log-cache"),
    )
    args = parser.parse_args()

    reference_quad_order = args.reference_quad_order
    if reference_quad_order is None:
        reference_quad_order = 24 if args.mode == "smoke" else 40

    cases = _select_cases(args.mode, args.studies, args.q_orders, args.n_levels)
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    rows = []
    for case in cases:
        print(f"running {case}", flush=True)
        rows.extend(
            run_case(
                ctx=ctx,
                queue=queue,
                case=case,
                mode=args.mode,
                cache_dir=args.cache_dir,
                reference_quad_order=reference_quad_order,
            )
        )
        print(f"finished {case.study}-q{case.q_order}-l{case.n_levels}", flush=True)

    _add_h_convergence(rows)
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
