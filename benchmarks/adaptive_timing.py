#!/usr/bin/env python3
"""Emit adaptive end-to-end timing CSVs for Paper 1.

The benchmark runs 2D Laplace volume-potential evaluations on meshes refined by
a fixed Gaussian indicator. It records graded-tree/List 1 diagnostics,
cold-cache and warm-cache setup timing, and independent post-warmup FMM samples
with median and spread statistics. Each case also performs one untimed comparison
between the canonical scaled table and direct tables for every populated source
level.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl


FIELDS = (
    "case_id",
    "mode",
    "cache_state",
    "q_order",
    "initial_nlevels",
    "adapt_steps",
    "n_active_boxes",
    "n_total_boxes",
    "n_targets",
    "min_leaf_level",
    "max_leaf_level",
    "leaf_level_histogram_json",
    "max_adjacent_leaf_level_difference",
    "n_list1_interactions",
    "n_cross_level_list1_interactions",
    "cross_level_list1_fraction",
    "list1_source_target_level_pair_histogram_json",
    "mesh_init_s",
    "adapt_s",
    "geometry_s",
    "table_get_s",
    "table_build_s",
    "table_load_s",
    "table_payload_bytes",
    "fmm_warmup_count",
    "fmm_trial_count",
    "fmm_wall_s",
    "fmm_wall_median_s",
    "fmm_wall_iqr_s",
    "fmm_wall_min_s",
    "fmm_wall_max_s",
    "fmm_wall_samples_s_json",
    "fmm_phase_samples_s_json",
    "timing_form_multipoles_s",
    "timing_coarsen_multipoles_s",
    "timing_eval_direct_s",
    "timing_multipole_to_local_s",
    "timing_eval_multipoles_s",
    "timing_form_locals_s",
    "timing_refine_locals_s",
    "timing_eval_locals_s",
    "path_mismatch_norm_definition",
    "canonical_vs_per_level_weighted_rel_l2",
    "canonical_vs_per_level_linf",
    "populated_source_levels_json",
    "canonical_table_count",
    "canonical_table_build_s",
    "canonical_table_payload_bytes",
    "per_level_table_count",
    "per_level_table_build_s",
    "per_level_table_payload_bytes",
)


SMOKE_CASES = ((3, 2, 1),)
FULL_CASES = ((3, 4, 2), (4, 4, 2), (4, 4, 3))

# Each successive shell is strictly inside the previous one. On the full-case
# level-3 base mesh, this leaves a one-level transition band around finer boxes.
REFINEMENT_THRESHOLDS = (1.0e-3, 0.5, 0.9)
SMOKE_WARMUP_COUNT = 1
SMOKE_TRIAL_COUNT = 3
FULL_WARMUP_COUNT = 2
FULL_TRIAL_COUNT = 7

FMM_TIMING_PHASES = (
    "form_multipoles",
    "coarsen_multipoles",
    "eval_direct",
    "multipole_to_local",
    "eval_multipoles",
    "form_locals",
    "refine_locals",
    "eval_locals",
)


def _device_supports_fp64(device) -> bool:
    extensions = set(getattr(device, "extensions", "").split())
    return "cl_khr_fp64" in extensions or bool(getattr(device, "double_fp_config", 0))


def _select_opencl_device(backend: str):
    backend = backend.lower()
    platforms = cl.get_platforms()

    if backend == "pocl-cpu":
        for platform in platforms:
            if "portable computing language" in platform.name.lower():
                for dev in platform.get_devices():
                    if dev.type & cl.device_type.CPU and _device_supports_fp64(dev):
                        return dev
        raise RuntimeError("PoCL CPU device with fp64 support not found")

    if backend == "cuda-gpu":
        for platform in platforms:
            if "nvidia cuda" in platform.name.lower():
                for dev in platform.get_devices():
                    if dev.type & cl.device_type.GPU and _device_supports_fp64(dev):
                        return dev
        raise RuntimeError("NVIDIA CUDA GPU device with fp64 support not found")

    if backend != "auto":
        raise ValueError("backend must be one of: auto, pocl-cpu, cuda-gpu")

    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl.device_type.GPU and _device_supports_fp64(dev):
                return dev
    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl.device_type.CPU and _device_supports_fp64(dev):
                return dev
    raise RuntimeError("No OpenCL GPU/CPU device with fp64 support found")


def _indicator(mesh) -> np.ndarray:
    centers = mesh.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]
    return np.exp(-90.0 * ((x - 0.12) ** 2 + (y + 0.08) ** 2))


def _build_adaptive_geometry(ctx, queue, q_order: int, initial_nlevels: int, adapt_steps: int):
    from boxtree import refine_and_coarsen_tree_of_boxes
    import volumential.meshgen as mg

    if adapt_steps > len(REFINEMENT_THRESHOLDS):
        raise ValueError(
            f"adapt_steps={adapt_steps} exceeds the supported nested refinement "
            f"shell count ({len(REFINEMENT_THRESHOLDS)})"
        )

    start = time.perf_counter()
    mesh = mg.MeshGen2D(q_order, initial_nlevels, -0.5, 0.5, queue=queue)  # pyright: ignore[reportArgumentType]
    mesh_init_s = time.perf_counter() - start

    start = time.perf_counter()
    for threshold in REFINEMENT_THRESHOLDS[:adapt_steps]:
        tree_of_boxes = mesh.boxtree._tree
        leaf_boxes = np.asarray(tree_of_boxes.leaf_boxes)
        refine_boxes = leaf_boxes[_indicator(mesh) >= threshold]
        if not len(refine_boxes):
            raise RuntimeError(
                f"Gaussian refinement shell {threshold} selected no leaf boxes"
            )

        refine_flags = np.zeros(tree_of_boxes.nboxes, dtype=bool)
        refine_flags[refine_boxes] = True

        # MeshGen's public update path also closes same-level colleagues, which
        # turns these compact cases uniform. The nested shells are independently
        # checked for 2:1 balance below, before any timing work is performed.
        mesh.boxtree._tree = refine_and_coarsen_tree_of_boxes(
            tree_of_boxes,
            refine_flags=refine_flags,
        )
    adapt_s = time.perf_counter() - start

    start = time.perf_counter()
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        2,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * 2, dtype=np.float64),
    )
    geometry_s = time.perf_counter() - start
    return mesh, q_points, q_weights, tree, traversal, mesh_init_s, adapt_s, geometry_s


def _leaf_diagnostics(mesh) -> dict[str, Any]:
    tree_of_boxes = mesh.boxtree._tree
    leaf_boxes = np.asarray(tree_of_boxes.leaf_boxes)
    levels = np.asarray(tree_of_boxes.box_levels)[leaf_boxes]
    unique_levels, level_counts = np.unique(levels, return_counts=True)
    histogram = {
        str(int(level)): int(count)
        for level, count in zip(unique_levels, level_counts, strict=True)
    }

    centers = np.asarray(tree_of_boxes.box_centers)[:, leaf_boxes].T
    side_lengths = float(tree_of_boxes.root_extent) * np.exp2(
        -levels.astype(np.float64)
    )
    max_adjacent_difference = 0
    for i in range(len(leaf_boxes)):
        touching = np.all(
            np.abs(centers[i + 1 :] - centers[i])
            <= 0.5 * (side_lengths[i + 1 :] + side_lengths[i])[:, np.newaxis]
            + 1.0e-14,
            axis=1,
        )
        if np.any(touching):
            max_adjacent_difference = max(
                max_adjacent_difference,
                int(np.max(np.abs(levels[i + 1 :][touching] - levels[i]))),
            )

    return {
        "min_leaf_level": int(np.min(levels)),
        "max_leaf_level": int(np.max(levels)),
        "leaf_level_histogram_json": json.dumps(
            histogram, sort_keys=True, separators=(",", ":")
        ),
        "max_adjacent_leaf_level_difference": max_adjacent_difference,
    }


def _to_host(queue, ary) -> np.ndarray:
    if hasattr(ary, "get"):
        return np.asarray(ary.get(queue))
    return np.asarray(ary)


def _list1_diagnostics(queue, tree, traversal) -> dict[str, Any]:
    box_levels = _to_host(queue, tree.box_levels)
    target_boxes = _to_host(queue, traversal.target_boxes)
    starts = _to_host(queue, traversal.neighbor_source_boxes_starts)
    source_boxes = _to_host(queue, traversal.neighbor_source_boxes_lists)

    interactions_per_target = np.diff(starts)
    target_levels = np.repeat(box_levels[target_boxes], interactions_per_target)
    source_levels = box_levels[source_boxes]
    if len(source_levels) != len(target_levels):
        raise RuntimeError("List 1 starts/lists arrays have inconsistent lengths")

    pair_histogram = {}
    if len(source_levels):
        pairs, counts = np.unique(
            np.column_stack((source_levels, target_levels)),
            axis=0,
            return_counts=True,
        )
        pair_histogram = {
            f"{int(source_level)}->{int(target_level)}": int(count)
            for (source_level, target_level), count in zip(pairs, counts, strict=True)
        }

    cross_level_count = int(np.count_nonzero(source_levels != target_levels))
    interaction_count = int(len(source_levels))
    return {
        "n_list1_interactions": interaction_count,
        "n_cross_level_list1_interactions": cross_level_count,
        "cross_level_list1_fraction": (
            cross_level_count / interaction_count if interaction_count else 0.0
        ),
        "list1_source_target_level_pair_histogram_json": json.dumps(
            pair_histogram, sort_keys=True, separators=(",", ":")
        ),
    }


def _populated_source_levels(queue, tree, traversal) -> list[int]:
    box_levels = _to_host(queue, tree.box_levels)
    target_boxes = _to_host(queue, traversal.target_boxes)
    source_boxes = _to_host(queue, traversal.neighbor_source_boxes_lists)
    leaf_levels = np.unique(box_levels[target_boxes])
    source_levels = np.unique(box_levels[source_boxes])
    if not np.array_equal(leaf_levels, source_levels):
        raise RuntimeError(
            "populated leaf and List 1 source levels do not cover the same levels"
        )
    return [int(level) for level in source_levels]


def _validate_adaptive_diagnostics(leaf_diagnostics, list1_diagnostics) -> None:
    if leaf_diagnostics["min_leaf_level"] >= leaf_diagnostics["max_leaf_level"]:
        raise RuntimeError("adaptive benchmark produced a uniform leaf level")
    if leaf_diagnostics["max_adjacent_leaf_level_difference"] > 1:
        raise RuntimeError("adaptive benchmark produced a tree that is not 2:1 balanced")
    if list1_diagnostics["n_cross_level_list1_interactions"] <= 0:
        raise RuntimeError("adaptive benchmark produced no cross-level List 1 work")


def _build_config(q_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(8, 4 * q_order),
        radial_quad_order=max(21, 10 * q_order),
    )


def _get_table(queue, cache_path: Path, q_order: int, *, force_recompute: bool):
    from volumential.table_manager import NearFieldInteractionTableManager
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            queue=queue,
            build_config=_build_config(q_order),
        )
        timings = table_manager.last_get_table_timings
    return table, timings


def _get_per_level_tables(
    queue,
    cache_path: Path,
    q_order: int,
    source_levels: list[int],
    *,
    tree_root_extent: float,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    if len(source_levels) < 2:
        raise RuntimeError("per-level equivalence requires mixed source levels")
    expected_levels = list(range(source_levels[0], source_levels[-1] + 1))
    if source_levels != expected_levels:
        raise RuntimeError(
            "populated source levels must be consecutive for the multilevel table path"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache_path.unlink()

    tables = []
    build_s = 0.0
    payload_bytes = 0
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=tree_root_extent, queue=queue
    ) as table_manager:
        for level in source_levels:
            table, _ = table_manager.get_table(
                2,
                "Laplace",
                q_order,
                source_box_level=level,
                force_recompute=True,
                queue=queue,
                build_config=_build_config(q_order),
            )
            timings = table_manager.last_get_table_timings
            if int(table.source_box_level) != level:
                raise RuntimeError(
                    f"requested direct table level {level}, got "
                    f"{int(table.source_box_level)}"
                )
            tables.append(table)
            build_s += _table_build_seconds(timings)
            payload_bytes += int(_table_payload_bytes(timings))

    return tables, {
        "per_level_table_count": len(tables),
        "per_level_table_build_s": build_s,
        "per_level_table_payload_bytes": payload_bytes,
    }


def _source_values(queue, q_points):
    import pyopencl.array as cla

    coords = np.array([axis.get(queue) for axis in q_points])
    x = coords[0]
    y = coords[1]
    values = np.exp(-35.0 * ((x + 0.10) ** 2 + (y - 0.05) ** 2))
    return cla.to_device(queue, np.ascontiguousarray(values.astype(np.float64)))


def _build_wrangler(ctx, queue, traversal, table, q_order: int, fmm_order: int):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    kernel = LaplaceKernel(2)
    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(kernel)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(kernel)
    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, kernel),
        partial(local_expn_class, kernel),
        [kernel],
        exclude_self=True,
    )
    self_extra_kwargs = {}
    if traversal.tree.sources_are_targets:
        self_extra_kwargs["target_to_source"] = np.arange(
            traversal.tree.ntargets, dtype=np.int32
        )
    return FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs,
    )


def _timing_seconds(value) -> float | str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return float(value)
    for name in ("wall_elapsed", "process_elapsed"):
        attr = getattr(value, name, None)
        if attr is None:
            continue
        try:
            elapsed = attr() if callable(attr) else attr
            return float(elapsed)  # pyright: ignore[reportArgumentType]
        except TypeError:
            pass
    return ""


def _run_fmm(ctx, queue, traversal, table, q_order: int, fmm_order: int, q_weights, q_points):
    from volumential.volume_fmm import drive_volume_fmm

    source_vals = _source_values(queue, q_points)
    wrangler = _build_wrangler(ctx, queue, traversal, table, q_order, fmm_order)
    timing_data: dict[str, Any] = {}
    queue.finish()
    start = time.perf_counter()
    (potential,) = drive_volume_fmm(
        traversal,
        wrangler,
        source_vals * q_weights,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
        timing_data=timing_data,
    )
    queue.finish()
    return time.perf_counter() - start, timing_data, potential


def _run_fmm_trials(
    ctx,
    queue,
    traversal,
    table,
    q_order: int,
    fmm_order: int,
    q_weights,
    q_points,
    *,
    warmup_count: int,
    trial_count: int,
):
    for _ in range(warmup_count):
        _run_fmm(
            ctx, queue, traversal, table, q_order, fmm_order, q_weights, q_points
        )

    wall_samples = []
    phase_samples = {phase: [] for phase in FMM_TIMING_PHASES}
    last_potential = None
    for _ in range(trial_count):
        wall_s, timing_data, last_potential = _run_fmm(
            ctx, queue, traversal, table, q_order, fmm_order, q_weights, q_points
        )
        wall_samples.append(wall_s)
        for phase in FMM_TIMING_PHASES:
            phase_s = _timing_seconds(timing_data.get(phase))
            if isinstance(phase_s, float):
                phase_samples[phase].append(phase_s)

    assert last_potential is not None
    return (
        wall_samples,
        {phase: samples for phase, samples in phase_samples.items() if samples},
        last_potential.get(queue),
    )


def _sample_statistics(samples) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    if not len(values):
        raise ValueError("at least one timing sample is required")
    q25, q75 = np.percentile(values, (25, 75))
    return {
        "median": float(np.median(values)),
        "iqr": float(q75 - q25),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _phase_median(phase_samples, phase: str) -> float | str:
    samples = phase_samples.get(phase, ())
    return _sample_statistics(samples)["median"] if samples else ""


def _table_phase_seconds(timings, phase: str) -> float | str:
    if not timings:
        return ""
    details = timings.get(phase)
    if not details:
        return ""
    return details.get("total_s", "")


def _table_payload_bytes(timings) -> int | str:
    if not timings:
        return ""
    details = timings.get("compute") or timings.get("load") or {}
    return int(details.get("payload_bytes", 0)) if details else ""


def _table_build_seconds(timings) -> float:
    if not timings:
        return 0.0
    details = timings.get("compute") or {}
    return float(details.get("table_build_s", details.get("total_s", 0.0)))


def _equivalence_diagnostics(
    canonical_potential,
    per_level_potential,
    q_weights,
    source_levels: list[int],
    canonical_table_timings,
    per_level_table_diagnostics,
) -> dict[str, Any]:
    if len(source_levels) < 2:
        raise RuntimeError("canonical/per-level comparison requires mixed source levels")
    if not (
        np.all(np.isfinite(canonical_potential))
        and np.all(np.isfinite(per_level_potential))
    ):
        raise RuntimeError("canonical/per-level comparison produced non-finite values")

    weights = q_weights.get()
    difference = canonical_potential - per_level_potential
    weighted_error = float(np.sqrt(np.sum(weights * np.abs(difference) ** 2)))
    weighted_reference = max(
        float(np.sqrt(np.sum(weights * np.abs(per_level_potential) ** 2))),
        1.0e-300,
    )
    return {
        "path_mismatch_norm_definition": (
            "weighted_rel_l2=sqrt(sum_i quadrature_weight_i*"
            "abs(canonical_i-per_level_i)**2)/max(sqrt(sum_i "
            "quadrature_weight_i*abs(per_level_i)**2),1e-300);"
            "linf=max_i abs(canonical_i-per_level_i)"
        ),
        "canonical_vs_per_level_weighted_rel_l2": (
            weighted_error / weighted_reference
        ),
        "canonical_vs_per_level_linf": float(np.max(np.abs(difference))),
        "populated_source_levels_json": json.dumps(source_levels, separators=(",", ":")),
        "canonical_table_count": 1,
        "canonical_table_build_s": _table_build_seconds(canonical_table_timings),
        "canonical_table_payload_bytes": _table_payload_bytes(
            canonical_table_timings
        ),
        **per_level_table_diagnostics,
    }


def _row(
    *,
    mode: str,
    cache_state: str,
    q_order: int,
    initial_nlevels: int,
    adapt_steps: int,
    mesh,
    tree,
    mesh_init_s: float,
    adapt_s: float,
    geometry_s: float,
    table_timings,
    leaf_diagnostics,
    list1_diagnostics,
    fmm_warmup_count: int,
    fmm_wall_samples,
    fmm_phase_samples,
) -> dict[str, Any]:
    wall_stats = _sample_statistics(fmm_wall_samples)
    return {
        "case_id": f"laplace2d-q{q_order}-l{initial_nlevels}-a{adapt_steps}",
        "mode": mode,
        "cache_state": cache_state,
        "q_order": q_order,
        "initial_nlevels": initial_nlevels,
        "adapt_steps": adapt_steps,
        "n_active_boxes": int(mesh.n_active_cells()),
        "n_total_boxes": int(mesh.n_cells()),
        "n_targets": int(tree.ntargets),
        **leaf_diagnostics,
        **list1_diagnostics,
        "mesh_init_s": mesh_init_s,
        "adapt_s": adapt_s,
        "geometry_s": geometry_s,
        "table_get_s": table_timings.get("total_s", "") if table_timings else "",
        "table_build_s": _table_phase_seconds(table_timings, "compute"),
        "table_load_s": _table_phase_seconds(table_timings, "load"),
        "table_payload_bytes": _table_payload_bytes(table_timings),
        "fmm_warmup_count": fmm_warmup_count,
        "fmm_trial_count": len(fmm_wall_samples),
        # Keep the legacy wall/phase columns, now as medians over measured trials.
        "fmm_wall_s": wall_stats["median"],
        "fmm_wall_median_s": wall_stats["median"],
        "fmm_wall_iqr_s": wall_stats["iqr"],
        "fmm_wall_min_s": wall_stats["min"],
        "fmm_wall_max_s": wall_stats["max"],
        "fmm_wall_samples_s_json": json.dumps(
            fmm_wall_samples, separators=(",", ":")
        ),
        "fmm_phase_samples_s_json": json.dumps(
            fmm_phase_samples, sort_keys=True, separators=(",", ":")
        ),
        "timing_form_multipoles_s": _phase_median(
            fmm_phase_samples, "form_multipoles"
        ),
        "timing_coarsen_multipoles_s": _phase_median(
            fmm_phase_samples, "coarsen_multipoles"
        ),
        "timing_eval_direct_s": _phase_median(fmm_phase_samples, "eval_direct"),
        "timing_multipole_to_local_s": _phase_median(
            fmm_phase_samples, "multipole_to_local"
        ),
        "timing_eval_multipoles_s": _phase_median(
            fmm_phase_samples, "eval_multipoles"
        ),
        "timing_form_locals_s": _phase_median(fmm_phase_samples, "form_locals"),
        "timing_refine_locals_s": _phase_median(
            fmm_phase_samples, "refine_locals"
        ),
        "timing_eval_locals_s": _phase_median(fmm_phase_samples, "eval_locals"),
    }


def run_case(
    ctx,
    queue,
    *,
    mode: str,
    cache_dir: Path,
    q_order: int,
    initial_nlevels: int,
    adapt_steps: int,
    warmup_count: int,
    trial_count: int,
):
    cache_path = cache_dir / f"adaptive-laplace2d-q{q_order}.sqlite"
    if cache_path.exists():
        cache_path.unlink()

    fmm_order = max(8, 4 * q_order)
    rows = []
    canonical_table_timings = None
    canonical_potential = None
    for cache_state, force_recompute in (("cold", True), ("warm", False)):
        mesh, q_points, q_weights, tree, traversal, mesh_init_s, adapt_s, geometry_s = (
            _build_adaptive_geometry(ctx, queue, q_order, initial_nlevels, adapt_steps)
        )
        leaf_diagnostics = _leaf_diagnostics(mesh)
        list1_diagnostics = _list1_diagnostics(queue, tree, traversal)
        _validate_adaptive_diagnostics(leaf_diagnostics, list1_diagnostics)
        table, table_timings = _get_table(
            queue, cache_path, q_order, force_recompute=force_recompute
        )
        fmm_wall_samples, fmm_phase_samples, canonical_potential = _run_fmm_trials(
            ctx,
            queue,
            traversal,
            table,
            q_order,
            fmm_order,
            q_weights,
            q_points,
            warmup_count=warmup_count,
            trial_count=trial_count,
        )
        if cache_state == "cold":
            canonical_table_timings = table_timings
        rows.append(
            _row(
                mode=mode,
                cache_state=cache_state,
                q_order=q_order,
                initial_nlevels=initial_nlevels,
                adapt_steps=adapt_steps,
                mesh=mesh,
                tree=tree,
                mesh_init_s=mesh_init_s,
                adapt_s=adapt_s,
                geometry_s=geometry_s,
                table_timings=table_timings,
                leaf_diagnostics=leaf_diagnostics,
                list1_diagnostics=list1_diagnostics,
                fmm_warmup_count=warmup_count,
                fmm_wall_samples=fmm_wall_samples,
                fmm_phase_samples=fmm_phase_samples,
            )
        )

    assert canonical_table_timings is not None
    assert canonical_potential is not None
    source_levels = _populated_source_levels(queue, tree, traversal)
    per_level_cache_path = cache_dir / (
        f"adaptive-laplace2d-q{q_order}-l{initial_nlevels}-a{adapt_steps}-"
        "per-level.sqlite"
    )
    per_level_tables, per_level_table_diagnostics = _get_per_level_tables(
        queue,
        per_level_cache_path,
        q_order,
        source_levels,
        tree_root_extent=float(tree.root_extent),
    )
    _, _, per_level_potential_dev = _run_fmm(
        ctx,
        queue,
        traversal,
        per_level_tables,
        q_order,
        fmm_order,
        q_weights,
        q_points,
    )
    equivalence_diagnostics = _equivalence_diagnostics(
        canonical_potential,
        per_level_potential_dev.get(queue),
        q_weights,
        source_levels,
        canonical_table_timings,
        per_level_table_diagnostics,
    )
    for row in rows:
        row.update(equivalence_diagnostics)
    return rows


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    warmup_count: int | None = None,
    trial_count: int | None = None,
):
    device = _select_opencl_device(backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    cases = SMOKE_CASES if mode == "smoke" else FULL_CASES
    if warmup_count is None:
        warmup_count = SMOKE_WARMUP_COUNT if mode == "smoke" else FULL_WARMUP_COUNT
    if trial_count is None:
        trial_count = SMOKE_TRIAL_COUNT if mode == "smoke" else FULL_TRIAL_COUNT
    if warmup_count < 0:
        raise ValueError("warmup_count must be nonnegative")
    if trial_count < 1:
        raise ValueError("trial_count must be at least one")
    rows = []
    for q_order, initial_nlevels, adapt_steps in cases:
        rows.extend(
            run_case(
                ctx,
                queue,
                mode=mode,
                cache_dir=cache_dir,
                q_order=q_order,
                initial_nlevels=initial_nlevels,
                adapt_steps=adapt_steps,
                warmup_count=warmup_count,
                trial_count=trial_count,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)  # pyright: ignore[reportArgumentType]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--warmups",
        type=int,
        default=None,
        help="FMM warmup runs per cache state (default: 1 smoke, 2 full)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="measured FMM runs per cache state (default: 3 smoke, 7 full)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/adaptive-timing.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/adaptive-timing-cache"),
    )
    args = parser.parse_args()
    if args.warmups is not None and args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be at least one")
    rows = run_benchmark(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        warmup_count=args.warmups,
        trial_count=args.trials,
    )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
