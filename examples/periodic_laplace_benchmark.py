"""Benchmark periodic Laplace auto mode (2D/3D).

This script compares three direct-evaluation modes on split source/target point
clouds:

1. free-space baseline
2. periodic near-image correction only
3. periodic near + auto far mode (strict spectral runtime for Laplace)

The reference is a spectral periodic Laplace sum on a rectangular torus.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsources", type=int, default=96)
    parser.add_argument("--ntargets", type=int, default=96)
    parser.add_argument("--fmm-order", type=int, default=8)
    parser.add_argument("--far-shift-radius", type=int, default=3)
    parser.add_argument("--training-samples", type=int, default=48)
    parser.add_argument("--max-check-points", type=int, default=128)
    parser.add_argument("--spectral-kmax-2d", type=int, default=24)
    parser.add_argument("--spectral-kmax-3d", type=int, default=10)
    parser.add_argument("--kmax-2d", type=int, default=24)
    parser.add_argument("--kmax-3d", type=int, default=10)
    parser.add_argument(
        "--cache-file",
        default="nft-periodic-benchmark.sqlite",
        help=(
            "SQLite cache file (near-field tables; and periodic operator cache "
            "for non-Laplace auto paths)"
        ),
    )
    return parser.parse_args()


def _create_non_intel_context():
    platforms = cl.get_platforms()
    for platform in platforms:
        if platform.name == "Intel(R) OpenCL":
            continue
        devices = platform.get_devices()
        if devices:
            return cl.Context([devices[0]])

    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            return cl.Context([devices[0]])

    raise RuntimeError("No OpenCL devices available")


def _cache_file_for_case(base_cache_file, dim, root_extent):
    base_path = Path(base_cache_file)
    stem = base_path.stem
    suffix = base_path.suffix or ".sqlite"
    root_tag = f"{float(root_extent):.12g}".replace("-", "m").replace(".", "p")
    return str(base_path.with_name(f"{stem}.d{int(dim)}.re{root_tag}{suffix}"))


def _spectral_periodic_laplace_reference(
    *,
    src_points,
    strengths,
    tgt_points,
    cell_size,
    k_max,
):
    src_points = np.asarray(src_points, dtype=np.float64)
    tgt_points = np.asarray(tgt_points, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    cell_size = np.asarray(cell_size, dtype=np.float64)

    dim = src_points.shape[1]
    modes = np.asarray(
        [
            mode
            for mode in product(range(-int(k_max), int(k_max) + 1), repeat=dim)
            if any(comp != 0 for comp in mode)
        ],
        dtype=np.float64,
    )
    kvec = (2.0 * np.pi) * modes / cell_size.reshape(1, -1)
    k2 = np.sum(kvec * kvec, axis=1)

    src_phase = np.exp(-1j * (src_points @ kvec.T))
    rho_k = src_phase.T @ strengths

    tgt_phase = np.exp(1j * (tgt_points @ kvec.T))
    volume = float(np.prod(cell_size))
    ref = (tgt_phase * (rho_k / k2).reshape(1, -1)).sum(axis=1) / volume
    return np.real(ref)


def _build_split_tree_and_traversal(queue, src_points, tgt_points):
    from boxtree import TreeBuilder
    from boxtree.array_context import PyOpenCLArrayContext
    from boxtree.traversal import FMMTraversalBuilder
    from pytools.obj_array import new_1d as obj_array_1d

    dim = src_points.shape[1]
    src_dev = obj_array_1d(
        [
            cl.array.to_device(queue, np.ascontiguousarray(src_points[:, iax]))
            for iax in range(dim)
        ]
    )
    tgt_dev = obj_array_1d(
        [
            cl.array.to_device(queue, np.ascontiguousarray(tgt_points[:, iax]))
            for iax in range(dim)
        ]
    )

    actx = PyOpenCLArrayContext(queue)
    tb = TreeBuilder(actx)
    tree, _ = tb(
        actx,
        particles=src_dev,
        targets=tgt_dev,
        max_particles_in_box=max(32, src_points.shape[0] // 2),
        kind="adaptive-level-restricted",
    )

    tg = FMMTraversalBuilder(actx)
    traversal, _ = tg(actx, tree)
    return tree, traversal


def _build_wrangler(ctx, queue, traversal, near_field_table, fmm_order):
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    dim = int(traversal.tree.dimensions)
    knl = LaplaceKernel(dim)

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(knl)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        [knl],
        exclude_self=False,
    )

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=near_field_table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: int(fmm_order),
        quad_order=1,
        self_extra_kwargs={},
    )
    return wrangler


def _run_case(args, dim):
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import drive_volume_fmm

    ctx = _create_non_intel_context()
    queue = cl.CommandQueue(ctx)
    rng = np.random.default_rng(args.seed + dim)

    src_points = rng.uniform(0.15, 0.85, size=(args.nsources, dim))
    tgt_points = rng.uniform(0.10, 0.90, size=(args.ntargets, dim))

    strengths = rng.standard_normal(args.nsources)
    strengths -= np.mean(strengths)

    tree, traversal = _build_split_tree_and_traversal(queue, src_points, tgt_points)
    cell_size = np.full(dim, float(tree.root_extent), dtype=np.float64)
    cache_file = _cache_file_for_case(args.cache_file, dim, tree.root_extent)

    with NearFieldInteractionTableManager(
        cache_file,
        root_extent=float(tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(dim, "Laplace", q_order=1, queue=queue)
        wrangler = _build_wrangler(ctx, queue, traversal, table, args.fmm_order)

        strengths_dev = cl.array.to_device(queue, np.ascontiguousarray(strengths))

        t0 = time.perf_counter()
        (pot_free,) = drive_volume_fmm(
            traversal,
            wrangler,
            strengths_dev,
            strengths_dev,
            direct_evaluation=True,
            auto_interpolate_targets=False,
        )
        queue.finish()
        free_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        (pot_near,) = drive_volume_fmm(
            traversal,
            wrangler,
            strengths_dev,
            strengths_dev,
            periodic=True,
            direct_evaluation=True,
            auto_interpolate_targets=False,
            periodic_near_shifts="nearest",
            periodic_cell_size=cell_size,
        )
        queue.finish()
        near_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        (pot_full,) = drive_volume_fmm(
            traversal,
            wrangler,
            strengths_dev,
            strengths_dev,
            periodic=True,
            direct_evaluation=True,
            auto_interpolate_targets=False,
            periodic_near_shifts="nearest",
            periodic_cell_size=cell_size,
            periodic_far_operator="auto",
            periodic_far_operator_manager=tm,
            periodic_far_shift_radius=args.far_shift_radius,
            periodic_far_spectral_kmax_2d=args.spectral_kmax_2d,
            periodic_far_spectral_kmax_3d=args.spectral_kmax_3d,
            periodic_far_training_samples=args.training_samples,
            periodic_far_rng_seed=args.seed,
            periodic_far_max_check_points=args.max_check_points,
            periodic_far_force_recompute=False,
        )
        queue.finish()
        full_time = time.perf_counter() - t0

    k_max = args.kmax_2d if dim == 2 else args.kmax_3d
    ref = _spectral_periodic_laplace_reference(
        src_points=src_points,
        strengths=strengths,
        tgt_points=tgt_points,
        cell_size=cell_size,
        k_max=k_max,
    )

    pot_near_host = pot_near.get(queue)
    pot_full_host = pot_full.get(queue)
    near_rel = np.linalg.norm(pot_near_host - ref) / np.linalg.norm(ref)
    full_rel = np.linalg.norm(pot_full_host - ref) / np.linalg.norm(ref)

    return {
        "dim": dim,
        "free_time_s": free_time,
        "near_time_s": near_time,
        "full_time_s": full_time,
        "near_rel_l2": float(near_rel),
        "full_rel_l2": float(full_rel),
        "n_sources": int(args.nsources),
        "n_targets": int(args.ntargets),
        "cell_extent": float(tree.root_extent),
        "cache_file": cache_file,
    }


def main() -> None:
    args = _parse_args()

    print("Periodic Laplace benchmark (auto spectral mode)")
    print(f"cache file: {args.cache_file}")
    print(
        "config:",
        {
            "dims": args.dims,
            "nsources": args.nsources,
            "ntargets": args.ntargets,
            "fmm_order": args.fmm_order,
            "far_shift_radius": args.far_shift_radius,
            "spectral_kmax_2d": args.spectral_kmax_2d,
            "spectral_kmax_3d": args.spectral_kmax_3d,
            "training_samples": args.training_samples,
            "max_check_points": args.max_check_points,
        },
    )

    for dim in args.dims:
        result = _run_case(args, dim)
        print("---")
        print(
            f"dim={result['dim']} ns={result['n_sources']} nt={result['n_targets']} "
            f"cell_extent={result['cell_extent']:.6f}"
        )
        print(f"cache file: {result['cache_file']}")
        print(
            f"timings [s] free={result['free_time_s']:.3f} "
            f"near={result['near_time_s']:.3f} near+far={result['full_time_s']:.3f}"
        )
        print(
            f"rel_l2 vs spectral reference near={result['near_rel_l2']:.3e} "
            f"near+far={result['full_rel_l2']:.3e}"
        )


if __name__ == "__main__":
    main()
