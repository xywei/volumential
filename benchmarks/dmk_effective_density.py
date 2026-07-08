#!/usr/bin/env python3
"""Run a controlled DMK-style effective-density diagnostic.

This benchmark does not implement full adaptive DMK. It isolates one
translation-invariant Gaussian kernel split for the 3D Laplace Green's function

    K(x) = 1 / (4*pi*|x|),      K_hat(k) = 1 / |k|^2,

using the finest-level regularized multiplier

    K_sigma_hat(k) = exp(-sigma_l^2 |k|^2 / 4) / |k|^2.

The effective density is therefore defined by

    K * rho_eff = K_sigma * rho,
    rho_eff_hat(k) = exp(-sigma_l^2 |k|^2 / 4) rho_hat(k).

For a Gaussian source this filtered density is analytic: rho_eff is the source
convolved with a normalized Gaussian of standard deviation sigma_l/sqrt(2). The
diagnostic applies Volumential's singular 3D Laplace path to rho_eff on an
enlarged bounding box and compares it against the analytic DMK-regularized
potential. The unsmoothed analytic reference remains a secondary diagnostic for
the intentional regularization bias. The source Gaussian variance, DMK-style split
scale sigma_l, density smoothing length, and residual SOG approximation status are
recorded separately in the emitted metadata.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl

from gaussian_free_space import (  # noqa: PLC2701 - sibling benchmark helper reuse
    _build_geometry,
    _coords_host,
    _device_metadata,
    _get_table,
    _git_commit,
    _leaf_stats,
    _run_fmm,
    _select_opencl_device,
    _table_payload_bytes,
    _table_phase_seconds,
)
from volumential.gaussian import (
    GaussianComponent,
    GaussianMixture,
    axis_aligned_slice_grid,
    dmk_gaussian_split_sigma,
    evaluate_gaussian_mixture,
    gaussian_filter_mixture,
    gaussian_mixture_tail_report,
    laplace3d_gaussian_potential,
    mesh_leaf_box_arrays,
    nearest_axis_slice,
    write_json_metadata,
    write_npz,
)
from volumential.version import VERSION_TEXT


SUMMARY_FIELDS = (
    "case_id",
    "mode",
    "problem",
    "dim",
    "kernel",
    "kernel_normalization",
    "dmk_regularized_kernel_multiplier",
    "effective_density_multiplier",
    "primary_comparison",
    "source_alpha",
    "source_variance_per_axis",
    "split_epsilon",
    "split_sigma",
    "effective_density_smoothing_std",
    "split_box_side_length",
    "residual_sog_terms",
    "q_order",
    "nlevels",
    "fmm_order",
    "regular_quad_order",
    "radial_quad_order",
    "root_extent",
    "n_targets",
    "n_active_boxes",
    "n_total_boxes",
    "min_leaf_level",
    "max_leaf_level",
    "source_omitted_abs_fraction",
    "effective_omitted_abs_fraction",
    "quadrature_source_mass",
    "quadrature_effective_mass",
    "table_get_s",
    "table_build_s",
    "table_load_s",
    "table_payload_bytes",
    "fmm_wall_s",
    "rel_l2_volumential_vs_dmk_regularized",
    "weighted_rel_l2_volumential_vs_dmk_regularized",
    "linf_volumential_vs_dmk_regularized",
    "rel_l2_regularized_bias_vs_unsmoothed",
    "weighted_rel_l2_regularized_bias_vs_unsmoothed",
    "linf_regularized_bias_vs_unsmoothed",
    "rel_l2_volumential_vs_unsmoothed",
    "weighted_rel_l2_volumential_vs_unsmoothed",
    "linf_volumential_vs_unsmoothed",
)


def _source_mixture(source_alpha: float) -> GaussianMixture:
    return GaussianMixture(
        name="dmk-effective-density-source-3d",
        components=(GaussianComponent(1.0, (0.0, 0.0, 0.0), source_alpha),),
    )


def _safe_rel_l2(error: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(error) / max(float(np.linalg.norm(reference)), 1.0e-300))


def _safe_weighted_rel_l2(error: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sqrt(np.sum(weights * error**2))
        / max(float(np.sqrt(np.sum(weights * reference**2))), 1.0e-300)
    )


def _root_extent_tag(root_extent: float) -> str:
    return f"extent{root_extent:.12g}".replace("-", "m").replace(".", "p")


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    regular_quad_order: int,
    radial_quad_order: int,
    root_radius: float,
    source_alpha: float,
    split_epsilon: float,
    split_sigma: float | None,
    force_recompute: bool,
    slice_size: int,
) -> dict[str, Any]:
    mixture = _source_mixture(source_alpha)
    device = _select_opencl_device(backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    mesh, bbox, q_points, q_weights, tree, traversal = _build_geometry(
        ctx,
        queue,
        q_order=q_order,
        nlevels=nlevels,
        root_radius=root_radius,
        adapt_steps=0,
        adapt_fraction=0.35,
    )
    coords = _coords_host(queue, q_points)
    weights = q_weights.get(queue)
    root_extent = 2.0 * root_radius
    leaf_arrays = mesh_leaf_box_arrays(mesh)
    leaf_side_lengths = leaf_arrays["leaf_side_lengths"]
    split_box_side_length = float(np.max(leaf_side_lengths))
    if not np.allclose(leaf_side_lengths, split_box_side_length):
        raise RuntimeError("DMK effective-density diagnostic expects a uniform mesh")
    if split_sigma is None:
        split_sigma = dmk_gaussian_split_sigma(split_box_side_length, split_epsilon)
    effective_density_smoothing_std = split_sigma / np.sqrt(2.0)

    effective_mixture = gaussian_filter_mixture(
        mixture, effective_density_smoothing_std
    )
    source = evaluate_gaussian_mixture(mixture, coords)
    effective_density = evaluate_gaussian_mixture(effective_mixture, coords)
    kernel_scale = 1.0 / (4.0 * np.pi)
    analytic_reference = laplace3d_gaussian_potential(
        mixture, coords, kernel_scale=kernel_scale
    )
    analytic_regularized = laplace3d_gaussian_potential(
        effective_mixture, coords, kernel_scale=kernel_scale
    )

    table, table_timings = _get_table(
        queue,
        cache_path=cache_dir
        / f"dmk-effective-density-laplace3d-q{q_order}-{_root_extent_tag(root_extent)}.sqlite",
        root_extent=root_extent,
        q_order=q_order,
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        force_recompute=force_recompute,
    )
    potential, fmm_wall_s = _run_fmm(
        ctx,
        queue,
        traversal,
        table,
        q_order=q_order,
        fmm_order=fmm_order,
        q_weights=q_weights,
        source_host=effective_density,
    )

    min_leaf_level, max_leaf_level = _leaf_stats(leaf_arrays)
    source_tail = gaussian_mixture_tail_report(mixture, bbox)
    effective_tail = gaussian_mixture_tail_report(effective_mixture, bbox)
    error_vs_regularized = potential - analytic_regularized
    regularized_bias = analytic_regularized - analytic_reference
    error_vs_unsmoothed = potential - analytic_reference
    case_id = f"dmk-effective-density-laplace3d-q{q_order}-l{nlevels}"

    row = {
        "case_id": case_id,
        "mode": mode,
        "problem": "controlled-dmk-minimal-smoothing-effective-density",
        "dim": 3,
        "kernel": "Laplace",
        "kernel_normalization": "sumpy_global_scaling_1_over_4pi_r",
        "dmk_regularized_kernel_multiplier": "exp(-sigma_l^2*|k|^2/4)/|k|^2",
        "effective_density_multiplier": "exp(-sigma_l^2*|k|^2/4)",
        "primary_comparison": "full_finite_box_laplace_rho_eff_vs_dmk_regularized_potential",
        "source_alpha": source_alpha,
        "source_variance_per_axis": 1.0 / (2.0 * source_alpha),
        "split_epsilon": split_epsilon,
        "split_sigma": split_sigma,
        "effective_density_smoothing_std": effective_density_smoothing_std,
        "split_box_side_length": split_box_side_length,
        "residual_sog_terms": 0,
        "q_order": q_order,
        "nlevels": nlevels,
        "fmm_order": fmm_order,
        "regular_quad_order": regular_quad_order,
        "radial_quad_order": radial_quad_order,
        "root_extent": root_extent,
        "n_targets": int(tree.ntargets),
        "n_active_boxes": int(mesh.n_active_cells()),
        "n_total_boxes": int(mesh.n_cells()),
        "min_leaf_level": min_leaf_level,
        "max_leaf_level": max_leaf_level,
        "source_omitted_abs_fraction": source_tail["omitted_abs_fraction"],
        "effective_omitted_abs_fraction": effective_tail["omitted_abs_fraction"],
        "quadrature_source_mass": float(np.sum(weights * source)),
        "quadrature_effective_mass": float(np.sum(weights * effective_density)),
        "table_get_s": table_timings.get("total_s", "") if table_timings else "",
        "table_build_s": _table_phase_seconds(table_timings, "compute"),
        "table_load_s": _table_phase_seconds(table_timings, "load"),
        "table_payload_bytes": _table_payload_bytes(table_timings),
        "fmm_wall_s": fmm_wall_s,
        "rel_l2_volumential_vs_dmk_regularized": _safe_rel_l2(
            error_vs_regularized, analytic_regularized
        ),
        "weighted_rel_l2_volumential_vs_dmk_regularized": _safe_weighted_rel_l2(
            error_vs_regularized, analytic_regularized, weights
        ),
        "linf_volumential_vs_dmk_regularized": float(
            np.max(np.abs(error_vs_regularized))
        ),
        "rel_l2_regularized_bias_vs_unsmoothed": _safe_rel_l2(
            regularized_bias, analytic_reference
        ),
        "weighted_rel_l2_regularized_bias_vs_unsmoothed": _safe_weighted_rel_l2(
            regularized_bias, analytic_reference, weights
        ),
        "linf_regularized_bias_vs_unsmoothed": float(np.max(np.abs(regularized_bias))),
        "rel_l2_volumential_vs_unsmoothed": _safe_rel_l2(
            error_vs_unsmoothed, analytic_reference
        ),
        "weighted_rel_l2_volumential_vs_unsmoothed": _safe_weighted_rel_l2(
            error_vs_unsmoothed, analytic_reference, weights
        ),
        "linf_volumential_vs_unsmoothed": float(np.max(np.abs(error_vs_unsmoothed))),
    }

    node_slice = nearest_axis_slice(
        coords,
        {
            "source": source,
            "effective_density": effective_density,
            "potential": potential,
            "analytic_regularized": analytic_regularized,
            "analytic_reference": analytic_reference,
            "error_vs_regularized": error_vs_regularized,
            "regularized_bias": regularized_bias,
            "weights": weights,
        },
        axis=2,
        value=0.0,
    )
    analytic_slice = axis_aligned_slice_grid(
        bbox,
        fixed_axis=2,
        fixed_value=0.0,
        shape=(slice_size, slice_size),
    )
    analytic_slice_source = evaluate_gaussian_mixture(mixture, analytic_slice.points)
    analytic_slice_effective_density = evaluate_gaussian_mixture(
        effective_mixture, analytic_slice.points
    )
    analytic_slice_reference = laplace3d_gaussian_potential(
        mixture, analytic_slice.points, kernel_scale=kernel_scale
    )
    analytic_slice_regularized = laplace3d_gaussian_potential(
        effective_mixture, analytic_slice.points, kernel_scale=kernel_scale
    )

    arrays = {
        "node_coords": coords,
        "node_weights": weights,
        "node_source": source,
        "node_effective_density": effective_density,
        "node_potential": potential,
        "node_analytic_regularized": analytic_regularized,
        "node_analytic_split": analytic_regularized,
        "node_analytic_reference": analytic_reference,
        "node_error_vs_regularized": error_vs_regularized,
        "node_error_vs_split": error_vs_regularized,
        "node_regularized_bias_vs_reference": regularized_bias,
        "node_split_bias_vs_reference": regularized_bias,
        "slice_node_coords": node_slice["coords"],
        "slice_node_indices": node_slice["indices"],
        "slice_node_axis_distances": node_slice["axis_distances"],
        "slice_node_source": node_slice["source"],
        "slice_node_effective_density": node_slice["effective_density"],
        "slice_node_potential": node_slice["potential"],
        "slice_node_analytic_regularized": node_slice["analytic_regularized"],
        "slice_node_analytic_split": node_slice["analytic_regularized"],
        "slice_node_analytic_reference": node_slice["analytic_reference"],
        "slice_node_error_vs_regularized": node_slice["error_vs_regularized"],
        "slice_node_error_vs_split": node_slice["error_vs_regularized"],
        "slice_node_regularized_bias": node_slice["regularized_bias"],
        "slice_node_split_bias": node_slice["regularized_bias"],
        "slice_node_weights": node_slice["weights"],
        "analytic_slice_coords": analytic_slice.points,
        "analytic_slice_shape": np.asarray(analytic_slice.shape, dtype=np.int32),
        "analytic_slice_axis0": analytic_slice.axis_values[0],
        "analytic_slice_axis1": analytic_slice.axis_values[1],
        "analytic_slice_source": analytic_slice_source,
        "analytic_slice_effective_density": analytic_slice_effective_density,
        "analytic_slice_reference": analytic_slice_reference,
        "analytic_slice_regularized": analytic_slice_regularized,
        "analytic_slice_split": analytic_slice_regularized,
        "analytic_slice_regularized_bias": (
            analytic_slice_regularized - analytic_slice_reference
        ),
        "analytic_slice_split_bias": analytic_slice_regularized - analytic_slice_reference,
        **{f"tree_{name}": values for name, values in leaf_arrays.items()},
    }
    metadata = {
        "case_id": case_id,
        "mode": mode,
        "problem": "controlled-dmk-minimal-smoothing-effective-density",
        "primary_comparison": row["primary_comparison"],
        "operator_identity": {
            "fourier_convention": (
                "forward integral exp(-i k.x), "
                "inverse (2*pi)^-3 integral exp(i k.x)"
            ),
            "laplace_kernel": "K(x)=1/(4*pi*|x|)",
            "laplace_multiplier": "1/|k|^2",
            "dmk_regularized_kernel_multiplier": row[
                "dmk_regularized_kernel_multiplier"
            ],
            "effective_density_multiplier": row["effective_density_multiplier"],
            "identity": "K * rho_eff = K_sigma_l * rho",
            "rho_eff_formula": "rho_eff = gamma_sigma * rho",
            "primary_diagnostic": (
                "full finite-box singular Laplace VP of rho_eff is compared "
                "against the analytic DMK-regularized potential"
            ),
            "gamma_sigma": (
                "(2*pi*s_eff^2)^(-3/2) exp(-|x|^2/(2*s_eff^2)), "
                "where s_eff=sigma_l/sqrt(2)"
            ),
            "gaussian_component_map": (
                "alpha_eff=alpha/(1+2 alpha s_eff^2), "
                "amplitude_eff=amplitude*(alpha_eff/alpha)^(dim/2)"
            ),
        },
        "dmk_parameters": {
            "epsilon": split_epsilon,
            "box_side_length": split_box_side_length,
            "sigma": split_sigma,
            "sigma_rule": "sigma = box_side_length / sqrt(log(1/epsilon))",
            "effective_density_smoothing_std": effective_density_smoothing_std,
            "effective_density_smoothing_rule": "s_eff = sigma / sqrt(2)",
            "residual_sog_terms": 0,
            "residual_sog_status": (
                "not used; this diagnostic models the finest DMK regularized "
                "operator as a global effective density"
            ),
        },
        "kernel": {
            "name": "Laplace",
            "dimension": 3,
            "normalization": "sumpy_global_scaling_1_over_4pi_r",
            "kernel_scale": kernel_scale,
        },
        "fixture": {
            "source": mixture.as_metadata(),
            "effective_density": effective_mixture.as_metadata(),
            "source_variance_per_axis": row["source_variance_per_axis"],
            "effective_density_smoothing_std": effective_density_smoothing_std,
        },
        "bbox": bbox,
        "tail": {
            "source": source_tail,
            "effective_density": effective_tail,
        },
        "discretization": {
            "q_order": q_order,
            "nlevels": nlevels,
            "adapt_steps": 0,
            "fmm_order": fmm_order,
            "regular_quad_order": regular_quad_order,
            "radial_quad_order": radial_quad_order,
        },
        "tree": {
            "n_targets": int(tree.ntargets),
            "n_active_boxes": int(mesh.n_active_cells()),
            "n_total_boxes": int(mesh.n_cells()),
            "min_leaf_level": min_leaf_level,
            "max_leaf_level": max_leaf_level,
        },
        "errors": {
            key: row[key]
            for key in (
                "rel_l2_volumential_vs_dmk_regularized",
                "weighted_rel_l2_volumential_vs_dmk_regularized",
                "linf_volumential_vs_dmk_regularized",
                "rel_l2_regularized_bias_vs_unsmoothed",
                "weighted_rel_l2_regularized_bias_vs_unsmoothed",
                "linf_regularized_bias_vs_unsmoothed",
                "rel_l2_volumential_vs_unsmoothed",
                "weighted_rel_l2_volumential_vs_unsmoothed",
                "linf_volumential_vs_unsmoothed",
            )
        },
        "timing": {
            "table_get_s": row["table_get_s"],
            "table_build_s": row["table_build_s"],
            "table_load_s": row["table_load_s"],
            "table_payload_bytes": row["table_payload_bytes"],
            "fmm_wall_s": row["fmm_wall_s"],
        },
        "cache": {
            "cache_dir": str(cache_dir),
            "force_recompute": force_recompute,
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencl_device": _device_metadata(device),
        },
        "volumential": {
            "version": VERSION_TEXT,
            "git_commit": _git_commit(),
        },
        "stress_cases_designed_not_run": [
            "cut or clipped Gaussian support to expose boundary-layer rho_eff artifacts",
            "signed high-frequency residual density rho - gamma_sigma*rho",
            "box-boundary source center shifts with fixed split sigma",
        ],
    }
    return {"rows": [row], "arrays": arrays, "metadata": metadata}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--q-order", type=int)
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--regular-quad-order", type=int)
    parser.add_argument("--radial-quad-order", type=int)
    parser.add_argument(
        "--root-radius",
        type=float,
        default=None,
        help="root half-width; defaults to 0.5 in smoke mode and 1.0 in full mode",
    )
    parser.add_argument("--source-alpha", type=float, default=75.0)
    parser.add_argument("--split-epsilon", type=float, default=1.0e-6)
    parser.add_argument(
        "--split-sigma",
        type=float,
        default=None,
        help="override sigma; otherwise use box_side_length/sqrt(log(1/epsilon))",
    )
    parser.add_argument("--slice-size", type=int, default=64)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/dmk-effective-density.csv"),
    )
    parser.add_argument(
        "--arrays-out",
        type=Path,
        default=Path("build/benchmarks/dmk-effective-density-arrays.npz"),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("build/benchmarks/dmk-effective-density-metadata.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/dmk-effective-density-cache"),
    )
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = args.q_order if args.q_order is not None else 4
    nlevels = args.nlevels if args.nlevels is not None else (2 if smoke else 4)
    fmm_order = args.fmm_order if args.fmm_order is not None else (10 if smoke else 12)
    root_radius = args.root_radius if args.root_radius is not None else (0.5 if smoke else 1.0)
    regular_quad_order = (
        args.regular_quad_order
        if args.regular_quad_order is not None
        else max(8, 4 * q_order)
    )
    radial_quad_order = (
        args.radial_quad_order if args.radial_quad_order is not None else max(25, 10 * q_order)
    )

    result = run_benchmark(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        q_order=q_order,
        nlevels=nlevels,
        fmm_order=fmm_order,
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        root_radius=root_radius,
        source_alpha=args.source_alpha,
        split_epsilon=args.split_epsilon,
        split_sigma=args.split_sigma,
        force_recompute=args.force_recompute,
        slice_size=args.slice_size,
    )
    metadata = result["metadata"]
    metadata["command"] = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
    }
    metadata["outputs"] = {
        "summary_csv": str(args.out),
        "arrays_npz": str(args.arrays_out),
        "metadata_json": str(args.metadata_out),
    }
    write_csv(args.out, result["rows"])
    write_npz(args.arrays_out, **result["arrays"])
    write_json_metadata(args.metadata_out, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
