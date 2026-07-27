#!/usr/bin/env python3
"""Emit 3D graded adaptive end-to-end timing CSVs for Paper 1.

This is the 3D companion of ``adaptive_timing.py``. It runs 3D Laplace
volume-potential evaluations on box meshes refined by nested Gaussian
indicator shells so that leaves populate three or more levels, records
graded-tree/List 1 diagnostics, cold-cache and warm-cache setup timing,
independent post-warmup FMM samples, and one untimed comparison between the
canonical scaled table and direct per-level tables. It also exports leaf-box
arrays per case for visualization.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from adaptive_timing import (  # noqa: E402
    FMM_TIMING_PHASES,
    REFINEMENT_THRESHOLDS,
    _build_config,
    _equivalence_diagnostics,
    _leaf_diagnostics,
    _list1_diagnostics,
    _phase_median,
    _populated_source_levels,
    _sample_statistics,
    _select_opencl_device,
    _table_build_seconds,
    _table_payload_bytes,
    _table_phase_seconds,
    _timing_seconds,
    _validate_adaptive_diagnostics,
    write_csv,
)
from volumential.gaussian import write_npz  # noqa: E402


SMOKE_CASES = ((2, 3, 1),)
FULL_CASES = ((3, 4, 2), (3, 4, 3))

SMOKE_WARMUP_COUNT = 1
SMOKE_TRIAL_COUNT = 3
FULL_WARMUP_COUNT = 2
FULL_TRIAL_COUNT = 5


def _indicator(mesh) -> np.ndarray:
    centers = mesh.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]
    return np.exp(
        -90.0 * ((x - 0.12) ** 2 + (y + 0.08) ** 2 + (z - 0.06) ** 2)
    )


def _balance_closure(tree_of_boxes, *, max_rounds: int = 10):
    """Refine the coarse side of any 2:1-violating leaf pair until balanced.

    Unlike MeshGen's public update path, this closure does not refine
    same-level colleagues, so compact graded cases stay graded.
    """
    from boxtree import refine_and_coarsen_tree_of_boxes

    for _ in range(max_rounds):
        leaf_boxes = np.asarray(tree_of_boxes.leaf_boxes)
        levels = np.asarray(tree_of_boxes.box_levels)[leaf_boxes]
        centers = np.asarray(tree_of_boxes.box_centers)[:, leaf_boxes].T
        sides = float(tree_of_boxes.root_extent) * np.exp2(
            -levels.astype(np.float64)
        )
        refine = np.zeros(len(leaf_boxes), dtype=bool)
        for i in range(len(leaf_boxes)):
            touching = np.all(
                np.abs(centers[i + 1 :] - centers[i])
                <= 0.5 * (sides[i + 1 :] + sides[i])[:, np.newaxis] + 1.0e-14,
                axis=1,
            )
            if not np.any(touching):
                continue
            neighbor_indices = np.nonzero(touching)[0] + i + 1
            neighbor_levels = levels[neighbor_indices]
            if np.any(neighbor_levels - levels[i] >= 2):
                refine[i] = True
            refine[neighbor_indices[levels[i] - neighbor_levels >= 2]] = True

        if not np.any(refine):
            return tree_of_boxes

        refine_flags = np.zeros(tree_of_boxes.nboxes, dtype=bool)
        refine_flags[leaf_boxes[refine]] = True
        tree_of_boxes = refine_and_coarsen_tree_of_boxes(
            tree_of_boxes,
            refine_flags=refine_flags,
        )

    raise RuntimeError("2:1 balance closure did not converge")


def _build_adaptive_geometry(
    ctx, queue, q_order: int, initial_nlevels: int, adapt_steps: int
):
    from boxtree import refine_and_coarsen_tree_of_boxes
    import volumential.meshgen as mg

    if adapt_steps > len(REFINEMENT_THRESHOLDS):
        raise ValueError(
            f"adapt_steps={adapt_steps} exceeds the supported nested refinement "
            f"shell count ({len(REFINEMENT_THRESHOLDS)})"
        )

    start = time.perf_counter()
    mesh = mg.MeshGen3D(q_order, initial_nlevels, -0.5, 0.5, queue=queue)  # pyright: ignore[reportArgumentType]
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
    # In 3D the innermost indicator shell can leave a transition band thinner
    # than one box, so close the tree to 2:1 balance before any timing work.
    mesh.boxtree._tree = _balance_closure(mesh.boxtree._tree)
    adapt_s = time.perf_counter() - start

    start = time.perf_counter()
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        3,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * 3, dtype=np.float64),
    )
    geometry_s = time.perf_counter() - start
    return mesh, q_points, q_weights, tree, traversal, mesh_init_s, adapt_s, geometry_s


def _leaf_arrays(mesh) -> dict[str, np.ndarray]:
    # Read from the underlying tree of boxes rather than the cached device
    # views, which are stale after the direct refine_and_coarsen updates above.
    tree_of_boxes = mesh.boxtree._tree
    leaf_boxes = np.asarray(tree_of_boxes.leaf_boxes).astype(np.int64)
    levels = np.asarray(tree_of_boxes.box_levels)[leaf_boxes].astype(np.int32)
    centers = np.asarray(tree_of_boxes.box_centers)[:, leaf_boxes].T.astype(
        np.float64
    )
    side_lengths = float(tree_of_boxes.root_extent) / np.power(2.0, levels)
    return {
        "leaf_box_ids": leaf_boxes,
        "leaf_centers": centers,
        "leaf_levels": levels,
        "leaf_side_lengths": side_lengths.astype(np.float64),
        "leaf_measures": (side_lengths**3).astype(np.float64),
    }


def _get_table(queue, cache_path: Path, q_order: int, *, force_recompute: bool):
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            3,
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
                3,
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
    z = coords[2]
    values = np.exp(
        -35.0 * ((x + 0.10) ** 2 + (y - 0.05) ** 2 + (z + 0.04) ** 2)
    )
    return cla.to_device(queue, np.ascontiguousarray(values.astype(np.float64)))


def _build_wrangler(ctx, queue, traversal, table, q_order: int, fmm_order: int):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    kernel = LaplaceKernel(3)
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
    import json

    wall_stats = _sample_statistics(fmm_wall_samples)
    return {
        "case_id": f"laplace3d-q{q_order}-l{initial_nlevels}-a{adapt_steps}",
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
    arrays_dir: Path | None,
    q_order: int,
    initial_nlevels: int,
    adapt_steps: int,
    warmup_count: int,
    trial_count: int,
):
    cache_path = cache_dir / f"adaptive-laplace3d-q{q_order}.sqlite"
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

    if arrays_dir is not None:
        leaf_arrays = _leaf_arrays(mesh)
        write_npz(
            arrays_dir
            / f"adaptive-laplace3d-q{q_order}-l{initial_nlevels}-a{adapt_steps}-leaves.npz",
            **{f"tree_{name}": values for name, values in leaf_arrays.items()},
        )

    source_levels = _populated_source_levels(queue, tree, traversal)
    per_level_cache_path = cache_dir / (
        f"adaptive-laplace3d-q{q_order}-l{initial_nlevels}-a{adapt_steps}-"
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
    arrays_dir: Path | None,
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
    if arrays_dir is not None:
        arrays_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q_order, initial_nlevels, adapt_steps in cases:
        rows.extend(
            run_case(
                ctx,
                queue,
                mode=mode,
                cache_dir=cache_dir,
                arrays_dir=arrays_dir,
                q_order=q_order,
                initial_nlevels=initial_nlevels,
                adapt_steps=adapt_steps,
                warmup_count=warmup_count,
                trial_count=trial_count,
            )
        )
    return rows


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
        help="measured FMM runs per cache state (default: 3 smoke, 5 full)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/adaptive-timing-3d.csv"),
    )
    parser.add_argument(
        "--arrays-dir",
        type=Path,
        default=Path("build/benchmarks/adaptive-timing-3d-arrays"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/adaptive-timing-3d-cache"),
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
        arrays_dir=args.arrays_dir,
        warmup_count=args.warmups,
        trial_count=args.trials,
    )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
