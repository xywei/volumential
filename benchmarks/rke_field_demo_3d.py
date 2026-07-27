#!/usr/bin/env python3
"""Run a 3D Yukawa RKE field demo for Paper 1.

The benchmark evaluates the 3D Yukawa volume potential of a compact Gaussian
mixture on a uniform box mesh at several screening parameters, once through
direct fixed-parameter tables and once through the RKE channel path at several
retained orders. It emits a summary CSV, an NPZ with full node fields plus a
mid-plane node slice of field and direct-vs-RKE pointwise difference, and a
JSON metadata sidecar, mirroring the Gaussian free-space demo artifacts.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from split_parameter_sweep import (  # noqa: E402
    _capture_table_get_timings,
    _clear_sqlite_cache,
    _select_opencl_device,
    _summarize_table_get_timings,
)
from volumential.gaussian import (  # noqa: E402
    default_overlapping_gaussian_mixture,
    evaluate_gaussian_mixture,
    nearest_axis_slice,
    write_json_metadata,
    write_npz,
)
from volumential.version import VERSION_TEXT  # noqa: E402


SUMMARY_FIELDS = (
    "case_id",
    "mode",
    "problem",
    "dim",
    "kernel",
    "parameter_name",
    "parameter",
    "effective_local_parameter",
    "split_order",
    "q_order",
    "nlevels",
    "fmm_order",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
    "root_extent",
    "n_targets",
    "n_boxes",
    "direct_table_build_s",
    "direct_table_load_s",
    "direct_table_payload_bytes",
    "rke_channel_build_s",
    "rke_channel_load_s",
    "rke_channel_payload_bytes",
    "direct_solve_wall_s",
    "split_solve_wall_s",
    "split_vs_direct_rel_l2",
    "split_vs_direct_weighted_rel_l2",
    "split_vs_direct_linf",
    "direct_max_abs_imag",
    "split_max_abs_imag",
    "field_max_abs",
)


def _build_config(regular_quad_order: int, radial_quad_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
    )


def _field_build_configs(q_order: int, *, high_accuracy: bool):
    if high_accuracy:
        direct = _build_config(max(16, 4 * q_order), max(45, 12 * q_order))
        channels = _build_config(
            max(12, 4 * q_order), max(35, 10 * q_order)
        )
        return direct, channels

    default = _build_config(max(8, 4 * q_order), max(21, 10 * q_order))
    return default, default


def _field_smooth_quad_order(
    q_order: int, split_order: int, *, high_accuracy: bool
) -> int:
    if high_accuracy and split_order > 1:
        return 2 * q_order
    return q_order


def _get_laplace_3d_table(
    queue,
    cache_path: Path,
    q_order: int,
    *,
    build_config,
    force_recompute: bool = False,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            3,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            queue=queue,
            build_config=build_config,
        )
    return table


def _get_yukawa_3d_table(
    queue,
    cache_path: Path,
    q_order: int,
    lam: float,
    level: int,
    *,
    build_config,
    force_recompute: bool = False,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            3,
            "Yukawa",
            q_order,
            source_box_level=int(level),
            force_recompute=force_recompute,
            queue=queue,
            build_config=build_config,
            lam=float(lam),
        )
    return table


def _build_geometry(ctx, queue, q_order: int, nlevels: int):
    import volumential.meshgen as mg

    mesh = mg.MeshGen3D(q_order, nlevels, -0.5, 0.5, queue=queue)  # pyright: ignore[reportArgumentType]
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        3,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * 3, dtype=np.float64),
    )
    return mesh, q_points, q_weights, tree, traversal


def _coords_host(queue, q_points):
    return np.array([axis.get(queue) for axis in q_points])


def _build_path(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    lam: float,
    table,
    source_weights,
    source_values_host,
    split: bool,
    split_order: int,
    split_term_tables=None,
    split_smooth_quad_order=None,
):
    from functools import partial

    import pyopencl.array as cla
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import YukawaKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    out_kernel = YukawaKernel(3)
    kernel_kwargs = {out_kernel.yukawa_lambda_name: float(lam)}
    # The shared split evaluator emits complex intermediates; keep direct and
    # split paths on the same dtype.
    dtype = np.complex128

    source_vals = cla.to_device(
        queue, np.ascontiguousarray(source_values_host.astype(dtype))
    )
    weighted_sources = source_vals * source_weights.astype(dtype)

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(out_kernel)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(out_kernel)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, out_kernel),
        partial(local_expn_class, out_kernel),
        [out_kernel],
        exclude_self=True,
    )
    self_extra_kwargs = {}
    if traversal.tree.sources_are_targets:
        self_extra_kwargs["target_to_source"] = np.arange(
            traversal.tree.ntargets, dtype=np.int32
        )

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=table,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        kernel_extra_kwargs=kernel_kwargs,
        self_extra_kwargs=self_extra_kwargs,
        helmholtz_split=split,
        helmholtz_split_order=split_order,
        helmholtz_split_term_tables=split_term_tables,
        helmholtz_split_smooth_quad_order=split_smooth_quad_order,
    )

    return wrangler, weighted_sources, source_vals


def _run_path(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    lam: float,
    table,
    source_weights,
    source_values_host,
    split: bool,
    split_order: int,
    split_term_tables=None,
    split_smooth_quad_order=None,
):
    from volumential.volume_fmm import drive_volume_fmm

    wrangler, weighted_sources, source_vals = _build_path(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        q_order=q_order,
        fmm_order=fmm_order,
        lam=lam,
        table=table,
        source_weights=source_weights,
        source_values_host=source_values_host,
        split=split,
        split_order=split_order,
        split_term_tables=split_term_tables,
        split_smooth_quad_order=split_smooth_quad_order,
    )

    def solve():
        return drive_volume_fmm(
            traversal,
            wrangler,
            weighted_sources,
            source_vals,
            direct_evaluation=False,
            list1_only=False,
        )

    # One untimed warmup absorbs JIT compilation, then one timed solve.
    solve()
    queue.finish()
    start = time.perf_counter()
    (potential,) = solve()
    queue.finish()
    wall_s = time.perf_counter() - start
    return potential.get(queue), wall_s, wrangler


def _prepare_direct_table(
    *,
    queue,
    cache_dir: Path,
    q_order: int,
    lam: float,
    level: int,
    build_config,
    force_recompute: bool,
):
    lam_tag = f"{lam:.17g}".replace("-", "m").replace(".", "p")
    cache_path = cache_dir / (
        f"field-direct-yukawa3d-lam{lam_tag}-q{q_order}.sqlite"
    )
    if force_recompute:
        _clear_sqlite_cache(cache_path)

    with _capture_table_get_timings() as cold_records:
        _get_yukawa_3d_table(
            queue, cache_path, q_order, lam, level, build_config=build_config
        )
    with _capture_table_get_timings() as warm_records:
        table = _get_yukawa_3d_table(
            queue, cache_path, q_order, lam, level, build_config=build_config
        )

    cold = _summarize_table_get_timings(cold_records)
    warm = _summarize_table_get_timings(warm_records)
    if warm["build_count"]:
        raise RuntimeError("direct-table warm pass was not a pure cache load")
    return table, {
        "build_s": cold["build_s"],
        "load_s": warm["load_s"],
        "payload_bytes": warm["cache_payload_bytes"],
    }


def _prepare_rke_channels(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    lam: float,
    split_order: int,
    source_weights,
    source_values_host,
    cache_dir: Path,
    build_config,
    split_smooth_quad_order: int,
    force_recompute: bool,
):
    cache_path = cache_dir / f"field-rke-yukawa3d-q{q_order}-p{split_order}.sqlite"
    if force_recompute:
        _clear_sqlite_cache(cache_path)

    with _capture_table_get_timings() as cold_records:
        cold_base_table = _get_laplace_3d_table(
            queue, cache_path, q_order, build_config=build_config
        )
        _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            lam=lam,
            table=cold_base_table,
            source_weights=source_weights,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_smooth_quad_order=split_smooth_quad_order,
        )

    with _capture_table_get_timings() as warm_records:
        warm_base_table = _get_laplace_3d_table(
            queue, cache_path, q_order, build_config=build_config
        )
        warm_wrangler, _, _ = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            lam=lam,
            table=warm_base_table,
            source_weights=source_weights,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_smooth_quad_order=split_smooth_quad_order,
        )

    cold = _summarize_table_get_timings(cold_records)
    warm = _summarize_table_get_timings(warm_records)
    if warm["build_count"]:
        raise RuntimeError("RKE warm pass unexpectedly rebuilt a channel table")
    return (
        warm_base_table,
        dict(warm_wrangler.helmholtz_split_term_tables),
        {
            "build_s": cold["build_s"],
            "load_s": warm["load_s"],
            "payload_bytes": warm["cache_payload_bytes"],
        },
    )


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _device_metadata(device) -> dict[str, Any]:
    return {
        "platform": device.platform.name,
        "device": device.name,
        "vendor": device.vendor,
        "version": device.version,
        "type": int(device.type),
    }


def _lam_tag(lam: float) -> str:
    return f"{lam:.12g}".replace("-", "m").replace(".", "p")


def _validate_full_order_convergence(rows: list[dict[str, Any]]) -> None:
    parameters = sorted({float(row["parameter"]) for row in rows})
    for parameter in parameters:
        errors = {
            int(row["split_order"]): float(row["split_vs_direct_weighted_rel_l2"])
            for row in rows
            if float(row["parameter"]) == parameter
        }
        if not {1, 2, 3}.issubset(errors):
            continue
        if not (
            errors[2] < 1.0e-2 * errors[1]
            and errors[3] < 0.5 * errors[2]
            and errors[3] < 1.0e-8
        ):
            raise RuntimeError(
                "full 3D Yukawa field sweep did not converge with split order "
                f"at lambda={parameter:g}: {errors}"
            )


def _public_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _public_argv(argv: list[str]) -> list[str]:
    result = []
    for token in argv:
        option, separator, value = token.partition("=")
        candidate = value if separator else token
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            candidate = _public_path(Path(candidate))
        result.append(option + separator + candidate if separator else candidate)
    return result


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    yukawa_lam: list[float],
    split_orders: list[int],
    force_recompute: bool,
) -> dict[str, Any]:
    high_accuracy = mode == "full"
    force_recompute = bool(force_recompute or high_accuracy)
    direct_build_config, rke_channel_build_config = _field_build_configs(
        q_order, high_accuracy=high_accuracy
    )
    if high_accuracy and not {1, 2, 3}.issubset(split_orders):
        raise ValueError("full mode requires split orders 1, 2, and 3")
    device = _select_opencl_device(cl, backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mesh, q_points, q_weights, tree, traversal = _build_geometry(
        ctx, queue, q_order, nlevels
    )
    coords = _coords_host(queue, q_points)
    weights = q_weights.get(queue)
    mixture = default_overlapping_gaussian_mixture(3)
    source_values_host = evaluate_gaussian_mixture(mixture, coords.T)
    leaf_level = int(tree.nlevels) - 1
    leaf_side = float(tree.root_extent) * 2.0**-leaf_level
    # Table caches use root extent 2 while this tree uses root extent 1, so the
    # matching table-manager level is one greater than the tree's leaf level.
    direct_table_level = leaf_level + 1
    if not np.isclose(2.0 * 2.0**-direct_table_level, leaf_side):
        raise RuntimeError("direct table level does not match the tree leaf size")

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "node_coords": coords,
        "node_weights": weights,
        "node_source": source_values_host,
    }
    slice_fields: dict[str, np.ndarray] = {"source": source_values_host}
    case_metadata: list[dict[str, Any]] = []

    direct_results = {}
    for lam in yukawa_lam:
        direct_table, direct_costs = _prepare_direct_table(
            queue=queue,
            cache_dir=cache_dir,
            q_order=q_order,
            lam=lam,
            level=direct_table_level,
            build_config=direct_build_config,
            force_recompute=force_recompute,
        )
        direct_potential, direct_wall_s, _ = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            lam=lam,
            table=direct_table,
            source_weights=q_weights,
            source_values_host=source_values_host,
            split=False,
            split_order=1,
        )
        direct_results[lam] = (direct_potential, direct_wall_s, direct_costs)
        tag = _lam_tag(lam)
        arrays[f"direct_potential_lam{tag}"] = direct_potential.real
        slice_fields[f"direct_lam{tag}"] = direct_potential.real

    for split_order in split_orders:
        representative_lam = yukawa_lam[0]
        split_smooth_quad_order = _field_smooth_quad_order(
            q_order, split_order, high_accuracy=high_accuracy
        )
        split_table, split_term_tables, rke_costs = _prepare_rke_channels(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            lam=representative_lam,
            split_order=split_order,
            source_weights=q_weights,
            source_values_host=source_values_host,
            cache_dir=cache_dir,
            build_config=rke_channel_build_config,
            split_smooth_quad_order=split_smooth_quad_order,
            force_recompute=force_recompute,
        )

        for lam in yukawa_lam:
            direct_potential, direct_wall_s, direct_costs = direct_results[lam]
            split_potential, split_wall_s, _ = _run_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                lam=lam,
                table=split_table,
                source_weights=q_weights,
                source_values_host=source_values_host,
                split=True,
                split_order=split_order,
                split_term_tables=split_term_tables,
                split_smooth_quad_order=split_smooth_quad_order,
            )

            difference = split_potential - direct_potential
            reference_norm = max(
                float(np.linalg.norm(np.abs(direct_potential))), 1.0e-300
            )
            weighted_reference_norm = max(
                float(np.sqrt(np.sum(weights * np.abs(direct_potential) ** 2))),
                1.0e-300,
            )

            tag = _lam_tag(lam)
            arrays[f"split_potential_lam{tag}_p{split_order}"] = split_potential.real
            arrays[f"split_vs_direct_absdiff_lam{tag}_p{split_order}"] = np.abs(
                difference
            )
            slice_fields[f"split_lam{tag}_p{split_order}"] = split_potential.real
            slice_fields[f"absdiff_lam{tag}_p{split_order}"] = np.abs(difference)

            case_id = (
                f"yukawa3d-field-lam{tag}-p{split_order}-q{q_order}-l{nlevels}"
            )
            row = {
                "case_id": case_id,
                "mode": mode,
                "problem": "gaussian-mixture-yukawa-field",
                "dim": 3,
                "kernel": "Yukawa",
                "parameter_name": "lambda",
                "parameter": lam,
                "effective_local_parameter": lam * leaf_side,
                "split_order": split_order,
                "q_order": q_order,
                "nlevels": nlevels,
                "fmm_order": fmm_order,
                "direct_regular_quad_order": (
                    direct_build_config.regular_quad_order
                ),
                "direct_radial_quad_order": (
                    direct_build_config.radial_quad_order
                ),
                "rke_channel_regular_quad_order": (
                    rke_channel_build_config.regular_quad_order
                ),
                "rke_channel_radial_quad_order": (
                    rke_channel_build_config.radial_quad_order
                ),
                "split_smooth_quad_order": split_smooth_quad_order,
                "root_extent": 1.0,
                "n_targets": int(tree.ntargets),
                "n_boxes": int(mesh.n_active_cells()),
                "direct_table_build_s": direct_costs["build_s"],
                "direct_table_load_s": direct_costs["load_s"],
                "direct_table_payload_bytes": direct_costs["payload_bytes"],
                "rke_channel_build_s": rke_costs["build_s"],
                "rke_channel_load_s": rke_costs["load_s"],
                "rke_channel_payload_bytes": rke_costs["payload_bytes"],
                "direct_solve_wall_s": direct_wall_s,
                "split_solve_wall_s": split_wall_s,
                "split_vs_direct_rel_l2": float(
                    np.linalg.norm(np.abs(difference)) / reference_norm
                ),
                "split_vs_direct_weighted_rel_l2": float(
                    np.sqrt(np.sum(weights * np.abs(difference) ** 2))
                    / weighted_reference_norm
                ),
                "split_vs_direct_linf": float(np.max(np.abs(difference))),
                "direct_max_abs_imag": float(np.max(np.abs(direct_potential.imag))),
                "split_max_abs_imag": float(np.max(np.abs(split_potential.imag))),
                "field_max_abs": float(np.max(np.abs(direct_potential))),
            }
            rows.append(row)
            case_metadata.append(
                {
                    "case_id": case_id,
                    "lambda": lam,
                    "split_order": split_order,
                    "errors": {
                        "rel_l2": row["split_vs_direct_rel_l2"],
                        "weighted_rel_l2": row["split_vs_direct_weighted_rel_l2"],
                        "linf": row["split_vs_direct_linf"],
                    },
                    "timing": {
                        "direct_solve_wall_s": direct_wall_s,
                        "split_solve_wall_s": split_wall_s,
                    },
                }
            )

    if high_accuracy:
        _validate_full_order_convergence(rows)

    node_slice = nearest_axis_slice(coords.T, slice_fields, axis=2, value=0.0)
    arrays["slice_node_coords"] = node_slice["coords"]
    arrays["slice_node_indices"] = node_slice["indices"]
    arrays["slice_node_axis_distances"] = node_slice["axis_distances"]
    for name in slice_fields:
        arrays[f"slice_node_{name}"] = node_slice[name]

    metadata = {
        "problem": "gaussian-mixture-yukawa-field",
        "mode": mode,
        "kernel": {
            "name": "Yukawa",
            "dimension": 3,
            "parameter_name": "lambda",
            "parameters": [float(lam) for lam in yukawa_lam],
            "normalization": "sumpy_global_scaling",
            "reference_path": "direct_fixed_parameter_table",
        },
        "fixture": mixture.as_metadata(),
        "discretization": {
            "q_order": q_order,
            "nlevels": nlevels,
            "fmm_order": fmm_order,
            "leaf_box_side": leaf_side,
            "effective_local_parameters": [
                float(lam) * leaf_side for lam in yukawa_lam
            ],
            "split_orders": [int(p) for p in split_orders],
            "quadrature": {
                "direct_regular_quad_order": (
                    direct_build_config.regular_quad_order
                ),
                "direct_radial_quad_order": (
                    direct_build_config.radial_quad_order
                ),
                "rke_channel_regular_quad_order": (
                    rke_channel_build_config.regular_quad_order
                ),
                "rke_channel_radial_quad_order": (
                    rke_channel_build_config.radial_quad_order
                ),
                "split_smooth_quad_order_by_retained_order": {
                    str(split_order): _field_smooth_quad_order(
                        q_order, split_order, high_accuracy=high_accuracy
                    )
                    for split_order in split_orders
                },
            },
        },
        "tree": {
            "n_targets": int(tree.ntargets),
            "n_active_boxes": int(mesh.n_active_cells()),
            "n_total_boxes": int(mesh.n_cells()),
        },
        "slice": {
            "node_slice_axis": int(node_slice["axis"]),
            "node_slice_value": float(node_slice["value"]),
            "node_slice_max_selected_distance": float(
                node_slice["max_selected_distance"]
            ),
            "node_slice_count": int(node_slice["coords"].shape[0]),
        },
        "cases": case_metadata,
        "cache": {
            "cache_dir": _public_path(cache_dir),
            "force_recompute": force_recompute,
        },
        "environment": {
            "hostname": "remote-compute-host",
            "python": platform.python_version(),
            "platform": platform.system(),
            "opencl_device": _device_metadata(device),
        },
        "volumential": {
            "version": VERSION_TEXT,
            "git_commit": _git_commit(),
        },
    }
    return {"rows": rows, "arrays": arrays, "metadata": metadata}


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
    parser.add_argument(
        "--yukawa-lambda",
        type=str,
        default=None,
        help="comma-separated screening parameters (default: 4 smoke, 2,4,8 full)",
    )
    parser.add_argument(
        "--split-orders",
        type=str,
        default=None,
        help="comma-separated RKE orders (default: 1,2 smoke, 1,2,3 full)",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/rke-field-demo-3d.csv"),
    )
    parser.add_argument(
        "--arrays-out",
        type=Path,
        default=Path("build/benchmarks/rke-field-demo-3d-arrays.npz"),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("build/benchmarks/rke-field-demo-3d-metadata.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/rke-field-demo-3d-cache"),
    )
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = args.q_order if args.q_order is not None else (2 if smoke else 3)
    nlevels = args.nlevels if args.nlevels is not None else (3 if smoke else 4)
    fmm_order = args.fmm_order if args.fmm_order is not None else (8 if smoke else 12)
    if args.yukawa_lambda is not None:
        yukawa_lam = [float(v) for v in args.yukawa_lambda.split(",") if v.strip()]
    else:
        yukawa_lam = [4.0] if smoke else [2.0, 4.0, 8.0]
    if args.split_orders is not None:
        split_orders = [int(v) for v in args.split_orders.split(",") if v.strip()]
    else:
        split_orders = [1, 2] if smoke else [1, 2, 3]
    if not yukawa_lam:
        parser.error("--yukawa-lambda must include at least one value")
    if not split_orders:
        parser.error("--split-orders must include at least one value")

    result = run_benchmark(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        q_order=q_order,
        nlevels=nlevels,
        fmm_order=fmm_order,
        yukawa_lam=yukawa_lam,
        split_orders=split_orders,
        force_recompute=args.force_recompute,
    )
    metadata = result["metadata"]
    metadata["command"] = {
        "argv": _public_argv(sys.argv),
        "cwd": ".",
    }
    metadata["outputs"] = {
        "summary_csv": _public_path(args.out),
        "arrays_npz": _public_path(args.arrays_out),
        "metadata_json": _public_path(args.metadata_out),
    }
    write_csv(args.out, result["rows"])
    write_npz(args.arrays_out, **result["arrays"])
    write_json_metadata(args.metadata_out, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
