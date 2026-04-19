"""Periodic Laplace volume examples with near/far decomposition.

This script uses true volume-discretized sources: source values are sampled on
tensor-product quadrature nodes and weighted by quadrature weights before
calling ``drive_volume_fmm``.

Optional adaptive mesh refinement can be enabled using a per-box Legendre
modal-tail indicator computed from the source density nodal values.

Modes
-----
1. ``benchmark``
   Compare free-space, periodic near-image-only, and periodic near+far solves
   for smooth volume source fields in 2D and/or 3D.

2. ``wrap-pair-figure`` (default)
   Reproduce a 2D boundary-wrap dipole-like volume source and generate plots
   showing that nearest-image correction alone is insufficient, while near+far
   periodic correction matches the spectral periodic reference.
   Potentials are solved on volume quadrature nodes and visualized on an
   interpolated uniform plotting grid.

Reference
---------
Alex H. Barnett, Gary R. Marple, Shravan Veerapaneni, and Lin Zhao,
"A Unified Integral Equation Scheme for Doubly Periodic Laplace and Stokes
Boundary Value Problems in Two Dimensions," Commun. Pure Appl. Math.
71(8):1694-1741, 2018. doi:10.1002/cpa.21759
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["wrap-pair-figure", "benchmark"],
        default="wrap-pair-figure",
        help="run benchmark table or wrap-pair figure reproduction",
    )

    parser.add_argument("--dims", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quad-order", type=int, default=4)
    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--fmm-order", type=int, default=8)

    parser.add_argument(
        "--adaptive-refine",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable adaptive box-mesh refinement using modal tail indicators; "
            "defaults to on in wrap-pair-figure mode and off in benchmark mode"
        ),
    )
    parser.add_argument(
        "--adaptive-max-loops",
        type=int,
        default=3,
        help="maximum adaptive refinement iterations",
    )
    parser.add_argument(
        "--adaptive-tail-tol",
        type=float,
        default=8.0e-2,
        help="stop refinement when max per-box modal tail ratio is below this",
    )
    parser.add_argument(
        "--adaptive-top-fraction",
        type=float,
        default=0.25,
        help="fraction of boxes to refine each adaptive step",
    )
    parser.add_argument(
        "--adaptive-bottom-fraction",
        type=float,
        default=0.0,
        help="fraction of boxes to coarsen each adaptive step",
    )
    parser.add_argument(
        "--adaptive-max-cells",
        type=int,
        default=2048,
        help="stop adaptive refinement once active cells exceed this",
    )

    parser.add_argument("--far-shift-radius", type=int, default=3)
    parser.add_argument("--spectral-kmax-2d", type=int, default=24)
    parser.add_argument("--spectral-kmax-3d", type=int, default=10)
    parser.add_argument("--kmax-2d", type=int, default=24)
    parser.add_argument("--kmax-3d", type=int, default=10)

    parser.add_argument(
        "--wrap-epsilon",
        type=float,
        default=0.02,
        help="distance of the wrap pair from x-boundaries (2D figure mode)",
    )
    parser.add_argument(
        "--background-amplitude",
        type=float,
        default=0.18,
        help="amplitude of smooth periodic background in wrap-pair mode",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=121,
        help="uniform plotting grid resolution per axis for figure mode",
    )
    parser.add_argument(
        "--plot-dir",
        default="periodic-laplace-plots",
        help="output directory for generated figures",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="compute and print metrics without generating figures",
    )

    parser.add_argument(
        "--cache-file",
        default="nft-periodic-benchmark.sqlite",
        help=(
            "SQLite cache file (near-field tables; periodic operator cache for "
            "non-Laplace auto paths)"
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


def _sanitize_tag(value):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)


def _cache_file_for_case(base_cache_file, dim, root_extent, case_tag):
    base_path = Path(base_cache_file)
    stem = base_path.stem
    suffix = base_path.suffix or ".sqlite"
    root_tag = f"{float(root_extent):.12g}".replace("-", "m").replace(".", "p")
    case_tag = _sanitize_tag(str(case_tag))
    return str(
        base_path.with_name(f"{stem}.{case_tag}.d{int(dim)}.re{root_tag}{suffix}")
    )


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


def _evaluate_density_on_mesh(density_builder, q_points_host, q_weights_host, rng_seed):
    rng = np.random.default_rng(int(rng_seed))
    values = np.asarray(
        density_builder(q_points_host, q_weights_host, rng),
        dtype=np.float64,
    )
    if values.shape != (len(q_points_host),):
        raise ValueError(
            f"density builder must return shape (n_modes,), got {values.shape}"
        )
    return values


def _build_legendre_modal_transform(dim, q_order):
    q_order = int(q_order)
    if q_order < 2:
        raise ValueError("quad_order must be >= 2")

    nodes_1d, _ = np.polynomial.legendre.leggauss(q_order)
    if int(dim) == 2:
        xx, yy = np.meshgrid(nodes_1d, nodes_1d, indexing="ij")
        vandermonde = np.polynomial.legendre.legvander2d(
            xx.reshape(-1),
            yy.reshape(-1),
            [q_order - 1, q_order - 1],
        )
    elif int(dim) == 3:
        xx, yy, zz = np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij")
        vandermonde = np.polynomial.legendre.legvander3d(
            xx.reshape(-1),
            yy.reshape(-1),
            zz.reshape(-1),
            [q_order - 1, q_order - 1, q_order - 1],
        )
    else:
        raise ValueError(f"unsupported dim={dim}; expected 2 or 3")

    inv_vandermonde = np.linalg.inv(vandermonde)
    multi_idx = np.asarray(
        list(product(range(q_order), repeat=int(dim))),
        dtype=np.int32,
    )
    tail_mask = np.any(multi_idx >= (q_order - 1), axis=1)

    return nodes_1d, inv_vandermonde, tail_mask


def _estimate_modal_tail_criteria(
    *,
    q_points_host,
    nodal_values,
    dim,
    q_order,
):
    q_points_host = np.asarray(q_points_host, dtype=np.float64)
    nodal_values = np.asarray(nodal_values, dtype=np.float64)
    dim = int(dim)
    n_q_per_cell = int(q_order) ** dim

    if len(q_points_host) % n_q_per_cell != 0:
        raise ValueError(
            "number of quadrature points is not divisible by q_order**dim: "
            f"n_points={len(q_points_host)}, n_q_per_cell={n_q_per_cell}"
        )

    n_cells = len(q_points_host) // n_q_per_cell
    if nodal_values.shape != (n_cells * n_q_per_cell,):
        raise ValueError(
            "nodal_values shape mismatch: expected "
            f"{(n_cells * n_q_per_cell,)}, got {nodal_values.shape}"
        )

    nodes_1d, inv_vandermonde, tail_mask = _build_legendre_modal_transform(dim, q_order)
    q_order = int(q_order)
    tail_ratio = np.zeros(n_cells, dtype=np.float64)
    mode_scale = np.zeros(n_cells, dtype=np.float64)

    for icell in range(n_cells):
        ibeg = icell * n_q_per_cell
        iend = ibeg + n_q_per_cell

        cell_points = q_points_host[ibeg:iend]
        cell_values = nodal_values[ibeg:iend]

        lo = np.min(cell_points, axis=0)
        hi = np.max(cell_points, axis=0)
        center = 0.5 * (lo + hi)
        extent = float(np.max(hi - lo))
        if extent <= 0:
            tail_ratio[icell] = 0.0
            mode_scale[icell] = 0.0
            continue

        local = 2.0 * (cell_points - center.reshape(1, -1)) / extent
        idx_axes = []
        for iax in range(dim):
            diff = np.abs(local[:, iax].reshape(-1, 1) - nodes_1d.reshape(1, -1))
            idx_axes.append(np.argmin(diff, axis=1).astype(np.int32))

        if dim == 2:
            mode_ids = idx_axes[0] * q_order + idx_axes[1]
        else:
            mode_ids = (
                idx_axes[0] * (q_order * q_order) + idx_axes[1] * q_order + idx_axes[2]
            )

        reorder = np.full(n_q_per_cell, -1, dtype=np.int32)
        reorder[mode_ids] = np.arange(n_q_per_cell, dtype=np.int32)
        if np.any(reorder < 0):
            raise RuntimeError(
                f"failed to map quadrature nodes to modal layout in cell {icell}"
            )

        nodal_lex = cell_values[reorder]
        modal = inv_vandermonde @ nodal_lex

        denom = float(np.linalg.norm(modal))
        if denom <= 0:
            tail_ratio[icell] = 0.0
            mode_scale[icell] = 0.0
            continue

        tail = np.asarray(modal[tail_mask], dtype=np.float64)
        tail_ratio[icell] = float(np.linalg.norm(tail) / denom)
        mode_scale[icell] = float(denom / np.sqrt(n_q_per_cell))

    ref_scale = float(np.percentile(mode_scale, 95.0))
    if not np.isfinite(ref_scale) or ref_scale <= 0:
        ref_scale = max(float(np.max(mode_scale)), 1.0e-16)

    amplitude_weight = np.minimum(1.0, mode_scale / ref_scale)
    criteria = tail_ratio * amplitude_weight
    return criteria, tail_ratio, amplitude_weight


def _build_volume_mesh_and_traversal(
    queue,
    dim,
    quad_order,
    nlevels,
    *,
    density_builder,
    rng_seed,
    adaptive_refine=False,
    adaptive_max_loops=0,
    adaptive_tail_tol=0.0,
    adaptive_top_fraction=0.25,
    adaptive_bottom_fraction=0.0,
    adaptive_max_cells=8192,
):
    import volumential.meshgen as mg
    from boxtree.array_context import PyOpenCLArrayContext
    from boxtree.traversal import FMMTraversalBuilder
    from volumential.tree_interactive_build import build_particle_tree_from_box_tree

    mesh_cls = {2: mg.MeshGen2D, 3: mg.MeshGen3D}[int(dim)]

    env_key = "VOLUMENTIAL_DISABLE_SAME_LEVEL_COLLEAGUES"
    old_env_value = os.environ.get(env_key)
    if bool(adaptive_refine):
        os.environ[env_key] = "1"

    try:
        mesh = mesh_cls(int(quad_order), int(nlevels), a=0.0, b=1.0, queue=queue)

        adaptive_info = {
            "enabled": bool(adaptive_refine),
            "loops": 0,
            "max_modal_tail": None,
            "max_modal_tail_ratio": None,
            "refine_candidates": None,
            "target_modal_tail": float(adaptive_tail_tol),
            "same_level_colleagues": not bool(adaptive_refine),
        }

        if bool(adaptive_refine):
            max_loops = int(max(0, adaptive_max_loops))
            tail_tol = float(max(0.0, adaptive_tail_tol))
            top_frac = float(np.clip(adaptive_top_fraction, 0.0, 1.0))
            bot_frac = float(np.clip(adaptive_bottom_fraction, 0.0, 1.0))
            max_cells = int(max(1, adaptive_max_cells))
            n_updates = 0

            for iloop in range(max_loops):
                q_points_loop = np.asarray(mesh.get_q_points(), dtype=np.float64)
                q_weights_loop = np.asarray(mesh.get_q_weights(), dtype=np.float64)
                source_loop = _evaluate_density_on_mesh(
                    density_builder,
                    q_points_loop,
                    q_weights_loop,
                    rng_seed,
                )
                modal_tail, tail_ratio, amplitude_weight = (
                    _estimate_modal_tail_criteria(
                        q_points_host=q_points_loop,
                        nodal_values=source_loop,
                        dim=dim,
                        q_order=quad_order,
                    )
                )
                max_tail = float(np.max(modal_tail)) if len(modal_tail) else 0.0
                significant = amplitude_weight > 0.05
                if np.any(significant):
                    max_tail_ratio = float(np.max(tail_ratio[significant]))
                else:
                    max_tail_ratio = (
                        float(np.max(tail_ratio)) if len(tail_ratio) else 0.0
                    )

                adaptive_info["loops"] = int(iloop)
                adaptive_info["max_modal_tail"] = max_tail
                adaptive_info["max_modal_tail_ratio"] = max_tail_ratio

                if max_tail <= tail_tol:
                    break
                if mesh.n_active_cells() >= max_cells:
                    break

                n_cells = int(len(modal_tail))
                if n_cells == 0:
                    break

                refine_floor = max(0.35 * max_tail, tail_tol)
                n_refine_candidates = int(np.count_nonzero(modal_tail >= refine_floor))
                adaptive_info["refine_candidates"] = n_refine_candidates
                if n_refine_candidates <= 0:
                    break

                top_frac_eff = min(top_frac, n_refine_candidates / n_cells)
                top_frac_eff = max(top_frac_eff, 1.0 / n_cells)

                mesh.update_mesh(
                    modal_tail,
                    top_fraction_of_cells=top_frac_eff,
                    bottom_fraction_of_cells=bot_frac,
                )
                n_updates += 1

            adaptive_info["loops"] = int(n_updates)

        q_points_host = np.asarray(mesh.get_q_points(), dtype=np.float64)
        q_weights_host = np.asarray(mesh.get_q_weights(), dtype=np.float64)
        if q_points_host.ndim != 2 or q_points_host.shape[1] != int(dim):
            raise RuntimeError(
                "meshgen returned unexpected q_points shape "
                f"{q_points_host.shape} for dim={dim}"
            )

        actx = PyOpenCLArrayContext(queue)
        tree = build_particle_tree_from_box_tree(actx, mesh.boxtree, q_points_host)

        tg = FMMTraversalBuilder(actx)
        traversal, _ = tg(actx, tree)
        q_weights_dev = cl.array.to_device(queue, np.ascontiguousarray(q_weights_host))

        adaptive_info["n_active_cells"] = int(mesh.n_active_cells())
        adaptive_info["n_q_points"] = int(len(q_points_host))
        adaptive_info["levels_present"] = [
            int(lev)
            for lev in np.unique(np.asarray(mesh._leaf_levels(), dtype=np.int32))
        ]

        return (
            q_points_host,
            q_weights_host,
            q_weights_dev,
            tree,
            traversal,
            adaptive_info,
        )
    finally:
        if old_env_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_env_value


def _build_wrangler(ctx, queue, traversal, near_field_table, fmm_order, quad_order):
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
        quad_order=int(quad_order),
        self_extra_kwargs={},
    )
    return wrangler


def _gaussian_field(points, center, sigma):
    delta = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    radius2 = np.sum(delta * delta, axis=1)
    sigma = float(sigma)
    return np.exp(-radius2 / (2.0 * sigma * sigma))


def _random_periodic_fourier_field(points, rng, *, n_modes=8, max_wave=3):
    points = np.asarray(points, dtype=np.float64)
    dim = points.shape[1]
    modes = rng.integers(1, int(max_wave) + 1, size=(int(n_modes), dim))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(n_modes))
    amplitudes = rng.standard_normal(int(n_modes))

    field = np.zeros(points.shape[0], dtype=np.float64)
    for mode, phase, amp in zip(modes, phases, amplitudes, strict=True):
        field += float(amp) * np.cos(2.0 * np.pi * (points @ mode) + float(phase))

    scale = float(np.max(np.abs(field)))
    if scale > 0:
        field /= scale
    return field


def _enforce_neutrality(density, q_weights):
    density = np.asarray(density, dtype=np.float64)
    q_weights = np.asarray(q_weights, dtype=np.float64)
    mean_val = float(np.dot(density, q_weights) / np.sum(q_weights))
    return density - mean_val


def _build_benchmark_density(points, q_weights, rng):
    dim = points.shape[1]

    c1 = np.linspace(0.2, 0.8, dim)
    c2 = np.linspace(0.75, 0.25, dim)
    c3 = np.roll(c1, 1)

    density = (
        0.95 * _gaussian_field(points, c1, sigma=0.10)
        - 0.80 * _gaussian_field(points, c2, sigma=0.16)
        + 0.35 * _gaussian_field(points, c3, sigma=0.08)
    )
    density += 0.25 * _random_periodic_fourier_field(points, rng, n_modes=7, max_wave=3)
    return _enforce_neutrality(density, q_weights)


def _build_wrap_pair_density(
    points, q_weights, rng, wrap_epsilon, background_amplitude
):
    if points.shape[1] != 2:
        raise ValueError("wrap-pair figure mode supports 2D only")

    eps = float(np.clip(wrap_epsilon, 1.0e-4, 0.1))
    density = (
        1.00 * _gaussian_field(points, [eps, 0.48], sigma=0.030)
        - 1.00 * _gaussian_field(points, [1.0 - eps, 0.52], sigma=0.030)
        + 0.35 * _gaussian_field(points, [0.30, 0.74], sigma=0.055)
        - 0.35 * _gaussian_field(points, [0.70, 0.26], sigma=0.055)
    )

    bg_amp = float(max(0.0, background_amplitude))
    if bg_amp > 0.0:
        density += bg_amp * _random_periodic_fourier_field(
            points,
            rng,
            n_modes=6,
            max_wave=2,
        )

    return _enforce_neutrality(density, q_weights)


def _relative_l2_error(values, reference):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denom = float(np.linalg.norm(reference))
    if denom <= 0:
        denom = 1.0
    return float(np.linalg.norm(values - reference) / denom)


def _compute_periodicity_probe_2d(*, src_points, strengths, cell_size, k_max):
    if np.asarray(src_points).shape[1] != 2:
        return None

    ny = 257
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny * float(cell_size[1])
    x = (np.arange(ny, dtype=np.float64) + 0.5) / ny * float(cell_size[0])

    left = np.column_stack([np.zeros(ny), y])
    right = np.column_stack([np.full(ny, float(cell_size[0])), y])
    bottom = np.column_stack([x, np.zeros(ny)])
    top = np.column_stack([x, np.full(ny, float(cell_size[1]))])

    u_left = _spectral_periodic_laplace_reference(
        src_points=src_points,
        strengths=strengths,
        tgt_points=left,
        cell_size=cell_size,
        k_max=k_max,
    )
    u_right = _spectral_periodic_laplace_reference(
        src_points=src_points,
        strengths=strengths,
        tgt_points=right,
        cell_size=cell_size,
        k_max=k_max,
    )
    u_bottom = _spectral_periodic_laplace_reference(
        src_points=src_points,
        strengths=strengths,
        tgt_points=bottom,
        cell_size=cell_size,
        k_max=k_max,
    )
    u_top = _spectral_periodic_laplace_reference(
        src_points=src_points,
        strengths=strengths,
        tgt_points=top,
        cell_size=cell_size,
        k_max=k_max,
    )

    dx = np.asarray(u_left - u_right, dtype=np.float64)
    dy = np.asarray(u_bottom - u_top, dtype=np.float64)

    return {
        "seam_x_abs_max": float(np.max(np.abs(dx))),
        "seam_y_abs_max": float(np.max(np.abs(dy))),
        "seam_x_rel_l2": float(
            np.linalg.norm(dx) / max(np.linalg.norm(u_left), 1.0e-16)
        ),
        "seam_y_rel_l2": float(
            np.linalg.norm(dy) / max(np.linalg.norm(u_bottom), 1.0e-16)
        ),
    }


def _solve_volume_case(
    args,
    *,
    ctx,
    dim,
    case_tag,
    density_builder,
    rng_seed,
    adaptive_refine,
):
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import drive_volume_fmm

    dim = int(dim)
    queue = cl.CommandQueue(ctx)
    q_points, q_weights, _, tree, traversal, adaptive_info = (
        _build_volume_mesh_and_traversal(
            queue,
            dim,
            args.quad_order,
            args.nlevels,
            density_builder=density_builder,
            rng_seed=rng_seed,
            adaptive_refine=bool(adaptive_refine),
            adaptive_max_loops=args.adaptive_max_loops,
            adaptive_tail_tol=args.adaptive_tail_tol,
            adaptive_top_fraction=args.adaptive_top_fraction,
            adaptive_bottom_fraction=args.adaptive_bottom_fraction,
            adaptive_max_cells=args.adaptive_max_cells,
        )
    )
    source_values = _evaluate_density_on_mesh(
        density_builder,
        q_points,
        q_weights,
        rng_seed,
    )

    source_strengths = np.ascontiguousarray(source_values * q_weights, dtype=np.float64)
    source_values = np.ascontiguousarray(source_values, dtype=np.float64)
    source_strengths_dev = cl.array.to_device(queue, source_strengths)
    source_values_dev = cl.array.to_device(queue, source_values)

    cell_size = np.full(dim, float(tree.root_extent), dtype=np.float64)
    cache_file = _cache_file_for_case(args.cache_file, dim, tree.root_extent, case_tag)

    with NearFieldInteractionTableManager(
        cache_file,
        root_extent=float(tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(dim, "Laplace", q_order=args.quad_order, queue=queue)
        wrangler = _build_wrangler(
            ctx,
            queue,
            traversal,
            table,
            args.fmm_order,
            args.quad_order,
        )

        t0 = time.perf_counter()
        (pot_free,) = drive_volume_fmm(
            traversal,
            wrangler,
            source_strengths_dev,
            source_values_dev,
            periodic=False,
            direct_evaluation=False,
            auto_interpolate_targets=False,
        )
        queue.finish()
        free_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        (pot_near,) = drive_volume_fmm(
            traversal,
            wrangler,
            source_strengths_dev,
            source_values_dev,
            periodic=True,
            direct_evaluation=False,
            auto_interpolate_targets=False,
            periodic_near_shifts="nearest",
            periodic_cell_size=cell_size,
            periodic_far_operator=None,
        )
        queue.finish()
        near_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        (pot_full,) = drive_volume_fmm(
            traversal,
            wrangler,
            source_strengths_dev,
            source_values_dev,
            periodic=True,
            direct_evaluation=False,
            auto_interpolate_targets=False,
            periodic_near_shifts="nearest",
            periodic_cell_size=cell_size,
            periodic_far_operator="auto",
            periodic_far_operator_manager=tm,
            periodic_far_shift_radius=args.far_shift_radius,
            periodic_far_spectral_kmax_2d=args.spectral_kmax_2d,
            periodic_far_spectral_kmax_3d=args.spectral_kmax_3d,
        )
        queue.finish()
        full_time = time.perf_counter() - t0

    k_max = args.kmax_2d if dim == 2 else args.kmax_3d
    ref_at_modes = _spectral_periodic_laplace_reference(
        src_points=q_points,
        strengths=source_strengths,
        tgt_points=q_points,
        cell_size=cell_size,
        k_max=k_max,
    )

    pot_free_host = np.asarray(pot_free.get(queue), dtype=np.float64)
    pot_near_host = np.asarray(pot_near.get(queue), dtype=np.float64)
    pot_full_host = np.asarray(pot_full.get(queue), dtype=np.float64)

    near_abs = np.abs(pot_near_host - ref_at_modes)
    full_abs = np.abs(pot_full_host - ref_at_modes)

    free_rel_modes = _relative_l2_error(pot_free_host, ref_at_modes)
    near_rel_modes = _relative_l2_error(pot_near_host, ref_at_modes)
    full_rel_modes = _relative_l2_error(pot_full_host, ref_at_modes)

    cell_boxes = _extract_cell_boxes_from_qpoints(q_points, dim, args.quad_order)
    cell_extents = np.max(cell_boxes[:, dim:] - cell_boxes[:, :dim], axis=1)
    level_raw = np.log(
        np.maximum(float(tree.root_extent), 1.0e-15) / np.maximum(cell_extents, 1.0e-15)
    ) / np.log(2.0)
    cell_levels = np.rint(level_raw).astype(np.int32)
    cell_levels = np.maximum(cell_levels, 0)

    result = {
        "case_tag": str(case_tag),
        "dim": dim,
        "n_sources": int(q_points.shape[0]),
        "n_targets": int(q_points.shape[0]),
        "cell_extent": float(tree.root_extent),
        "cell_size": np.asarray(cell_size, dtype=np.float64),
        "cache_file": cache_file,
        "quad_order": int(args.quad_order),
        "nlevels": int(args.nlevels),
        "source_points": q_points,
        "cell_boxes": cell_boxes,
        "cell_levels": cell_levels,
        "source_values": source_values,
        "source_strengths": source_strengths,
        "adaptive_info": adaptive_info,
        "free_time_s": float(free_time),
        "near_time_s": float(near_time),
        "full_time_s": float(full_time),
        "free_rel_l2": free_rel_modes,
        "near_rel_l2": near_rel_modes,
        "full_rel_l2": full_rel_modes,
        "free_rel_l2_modes": free_rel_modes,
        "near_rel_l2_modes": near_rel_modes,
        "full_rel_l2_modes": full_rel_modes,
        "near_abs_linf": float(np.max(near_abs)),
        "full_abs_linf": float(np.max(full_abs)),
        "near_abs_linf_modes": float(np.max(near_abs)),
        "full_abs_linf_modes": float(np.max(full_abs)),
        "pot_free": pot_free_host,
        "pot_near": pot_near_host,
        "pot_full": pot_full_host,
        "reference": ref_at_modes,
    }

    periodicity_probe = _compute_periodicity_probe_2d(
        src_points=q_points,
        strengths=source_strengths,
        cell_size=cell_size,
        k_max=k_max,
    )
    if periodicity_probe is not None:
        result.update(periodicity_probe)

    return result


def _print_case_summary(result):
    print("---")
    print(
        f"case={result['case_tag']} dim={result['dim']} "
        f"n_modes={result['n_sources']} q={result['quad_order']} "
        f"levels={result['nlevels']} cell_extent={result['cell_extent']:.6f}"
    )
    adaptive_info = result.get("adaptive_info")
    if adaptive_info is not None:
        print(
            "adaptive mesh: "
            f"enabled={adaptive_info.get('enabled', False)} "
            f"loops={adaptive_info.get('loops', 0)} "
            f"max_weighted_tail={adaptive_info.get('max_modal_tail', 'n/a')} "
            f"max_tail_ratio_sig={adaptive_info.get('max_modal_tail_ratio', 'n/a')} "
            f"refine_candidates={adaptive_info.get('refine_candidates', 'n/a')} "
            f"same_level_colleagues={adaptive_info.get('same_level_colleagues', 'n/a')} "
            f"target_tail={adaptive_info.get('target_modal_tail', 'n/a')} "
            f"cells={adaptive_info.get('n_active_cells', 'n/a')} "
            f"leaf_levels={adaptive_info.get('levels_present', [])}"
        )
    print(f"cache file: {result['cache_file']}")
    print(
        f"timings [s] free={result['free_time_s']:.3f} "
        f"near={result['near_time_s']:.3f} near+far={result['full_time_s']:.3f}"
    )
    print(
        f"rel_l2 vs spectral on source modes free={result['free_rel_l2_modes']:.3e} "
        f"near={result['near_rel_l2_modes']:.3e} "
        f"near+far={result['full_rel_l2_modes']:.3e}"
    )
    print(
        f"abs_linf vs spectral on source modes near={result['near_abs_linf_modes']:.3e} "
        f"near+far={result['full_abs_linf_modes']:.3e}"
    )
    if "seam_x_abs_max" in result:
        print(
            "periodicity seam check (spectral reference) "
            f"x-edge max={result['seam_x_abs_max']:.3e}, "
            f"y-edge max={result['seam_y_abs_max']:.3e}"
        )


def _extract_nearby_linecut(points, values, *, y0=0.5):
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ydist = np.abs(points[:, 1] - float(y0))

    npts = int(len(points))
    nkeep = max(48, min(npts, npts // 10))
    nearest = np.argsort(ydist)[:nkeep]
    nearest = nearest[np.argsort(points[nearest, 0])]

    x = points[nearest, 0]
    y = points[nearest, 1]
    v = values[nearest]
    band = float(np.max(np.abs(y - float(y0))))
    return x, v, band


def _tile_points_values_2d(points, values, cell_size):
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lx = float(cell_size[0])
    ly = float(cell_size[1])

    shifts = np.asarray(
        [(ix * lx, iy * ly) for ix in (-1, 0, 1) for iy in (-1, 0, 1)],
        dtype=np.float64,
    )
    tiled_points = np.vstack([points + shift for shift in shifts])
    tiled_values = np.tile(values, len(shifts))
    return tiled_points, tiled_values


def _extract_cell_boxes_from_qpoints(q_points, dim, q_order):
    q_points = np.asarray(q_points, dtype=np.float64)
    dim = int(dim)
    n_q_per_cell = int(q_order) ** dim

    if len(q_points) % n_q_per_cell != 0:
        raise ValueError(
            "q_points length is not divisible by q_order**dim: "
            f"n_points={len(q_points)}, n_q_per_cell={n_q_per_cell}"
        )

    n_cells = len(q_points) // n_q_per_cell
    boxes = np.empty((n_cells, dim * 2), dtype=np.float64)

    for icell in range(n_cells):
        ibeg = icell * n_q_per_cell
        iend = ibeg + n_q_per_cell
        cell_points = q_points[ibeg:iend]
        lo = np.min(cell_points, axis=0)
        hi = np.max(cell_points, axis=0)
        boxes[icell, :dim] = lo
        boxes[icell, dim:] = hi

    return boxes


def _build_plot_grid_2d(grid_size, cell_size):
    grid_size = int(grid_size)
    if grid_size < 8:
        raise ValueError("grid_size must be >= 8")

    lx = float(cell_size[0])
    ly = float(cell_size[1])
    x = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * lx
    y = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size * ly
    xx, yy = np.meshgrid(x, y, indexing="xy")
    query = np.ascontiguousarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    return query, xx, yy


def _interpolate_periodic_field_to_grid_2d(points, values, cell_size, grid_size):
    from scipy.interpolate import griddata

    query, xx, yy = _build_plot_grid_2d(grid_size, cell_size)
    tiled_points, tiled_values = _tile_points_values_2d(points, values, cell_size)

    interp_linear = griddata(tiled_points, tiled_values, query, method="linear")
    interp_nearest = griddata(tiled_points, tiled_values, query, method="nearest")

    if interp_linear is None:
        interp = np.asarray(interp_nearest, dtype=np.float64)
    else:
        interp = np.asarray(interp_linear, dtype=np.float64)
        interp_nearest = np.asarray(interp_nearest, dtype=np.float64)
        missing = ~np.isfinite(interp)
        if np.any(missing):
            interp[missing] = interp_nearest[missing]

    return interp.reshape(int(grid_size), int(grid_size)), xx, yy


def _save_wrap_pair_plots(*, result, plot_dir, grid_size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    src_points = np.asarray(result["source_points"], dtype=np.float64)
    free_vals = np.asarray(result["pot_free"], dtype=np.float64)
    near_vals = np.asarray(result["pot_near"], dtype=np.float64)
    full_vals = np.asarray(result["pot_full"], dtype=np.float64)
    ref_vals = np.asarray(result["reference"], dtype=np.float64)

    lx = float(result["cell_size"][0])
    ly = float(result["cell_size"][1])
    extent = (0.0, lx, 0.0, ly)

    free_grid, xx, yy = _interpolate_periodic_field_to_grid_2d(
        src_points,
        free_vals,
        result["cell_size"],
        grid_size,
    )
    near_grid, _, _ = _interpolate_periodic_field_to_grid_2d(
        src_points,
        near_vals,
        result["cell_size"],
        grid_size,
    )
    full_grid, _, _ = _interpolate_periodic_field_to_grid_2d(
        src_points,
        full_vals,
        result["cell_size"],
        grid_size,
    )
    ref_grid, _, _ = _interpolate_periodic_field_to_grid_2d(
        src_points,
        ref_vals,
        result["cell_size"],
        grid_size,
    )

    err_near = np.abs(near_grid - ref_grid)
    err_full = np.abs(full_grid - ref_grid)
    ref_norm_l2 = float(np.linalg.norm(ref_grid))
    if ref_norm_l2 <= 0:
        ref_norm_l2 = 1.0
    rel_err_near = err_near / ref_norm_l2
    rel_err_full = err_full / ref_norm_l2

    field_abs = np.abs(ref_grid)
    field_lim = float(np.percentile(field_abs, 99.0))
    if not np.isfinite(field_lim) or field_lim <= 0:
        field_lim = float(np.max(field_abs) + 1.0e-15)

    err_log_near = np.log10(np.maximum(rel_err_near, 1.0e-18))
    err_log_full = np.log10(np.maximum(rel_err_full, 1.0e-18))
    near_vmin = float(np.percentile(err_log_near, 2.0))
    near_vmax = float(np.percentile(err_log_near, 99.5))
    full_vmin = float(np.percentile(err_log_full, 2.0))
    full_vmax = float(np.percentile(err_log_full, 99.5))

    if (
        not np.isfinite(near_vmin)
        or not np.isfinite(near_vmax)
        or near_vmin >= near_vmax
    ):
        near_vmin, near_vmax = -12.0, -2.0
    if (
        not np.isfinite(full_vmin)
        or not np.isfinite(full_vmax)
        or full_vmin >= full_vmax
    ):
        full_vmin, full_vmax = -16.0, -8.0

    near_rel_vis = _relative_l2_error(near_grid, ref_grid)
    full_rel_vis = _relative_l2_error(full_grid, ref_grid)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    panels = [
        (
            free_grid,
            "free-space potential (interpolated)",
            "coolwarm",
            -field_lim,
            field_lim,
        ),
        (
            near_grid,
            "periodic near-image only (interpolated)",
            "coolwarm",
            -field_lim,
            field_lim,
        ),
        (
            full_grid,
            "periodic near+far (volumential, interpolated)",
            "coolwarm",
            -field_lim,
            field_lim,
        ),
        (
            ref_grid,
            "spectral periodic reference (interpolated)",
            "coolwarm",
            -field_lim,
            field_lim,
        ),
        (
            err_log_near,
            (f"log10 (|near - ref| / ||ref||_2)  (visual rel-L2={near_rel_vis:.2e})"),
            "viridis",
            near_vmin,
            near_vmax,
        ),
        (
            err_log_full,
            (
                "log10 (|near+far - ref| / ||ref||_2)  "
                f"(visual rel-L2={full_rel_vis:.2e})"
            ),
            "viridis",
            full_vmin,
            full_vmax,
        ),
    ]

    for ax, (field, title, cmap, vmin, vmax) in zip(axes.flat, panels, strict=True):
        im = ax.imshow(
            np.asarray(field, dtype=np.float64),
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, lx)
        ax.set_ylim(0.0, ly)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    field_path = plot_dir / "periodic_wrap_pair_fields.png"
    fig.savefig(field_path, dpi=180)
    plt.close(fig)

    y_axis = yy[:, 0]
    y_mid = 0.5 * ly
    row = int(np.argmin(np.abs(y_axis - y_mid)))
    x_axis = xx[row, :]
    ref_cut = ref_grid[row, :]
    free_cut = free_grid[row, :]
    near_cut = near_grid[row, :]
    full_cut = full_grid[row, :]
    cut_band = float(np.abs(y_axis[row] - y_mid))

    rel_cut_norm = max(float(np.linalg.norm(ref_cut)), 1.0e-16)
    free_cut_rel = np.abs(free_cut - ref_cut) / rel_cut_norm
    near_cut_rel = np.abs(near_cut - ref_cut) / rel_cut_norm
    full_cut_rel = np.abs(full_cut - ref_cut) / rel_cut_norm

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax0, ax1 = axes

    ax0.plot(x_axis, ref_cut, label="spectral reference", lw=2.2, color="black")
    ax0.plot(
        x_axis,
        near_cut,
        label="periodic near only",
        lw=1.5,
        color="#d95f02",
    )
    ax0.plot(
        x_axis,
        full_cut,
        label="periodic near+far",
        lw=1.8,
        color="#1b9e77",
        linestyle="--",
    )
    ax0.plot(
        x_axis,
        free_cut,
        label="free-space",
        lw=1.0,
        color="#6b7280",
        alpha=0.85,
    )
    ax0.set_xlabel("x")
    ax0.set_ylabel("u(x, y~Ly/2)")
    ax0.set_title(f"Interpolated line cut (|y-Ly/2|={cut_band:.2e})")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best")

    ax1.plot(
        x_axis, free_cut_rel, label="|free-ref| / ||ref||_2", lw=1.1, color="#6b7280"
    )
    ax1.plot(
        x_axis,
        near_cut_rel,
        label="|near-ref| / ||ref||_2",
        lw=1.3,
        color="#d95f02",
    )
    ax1.plot(
        x_axis,
        full_cut_rel,
        label="|near+far-ref| / ||ref||_2",
        lw=1.5,
        color="#1b9e77",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("x")
    ax1.set_ylabel("relative error")
    ax1.set_title("Line-cut relative error")
    ax1.grid(True, alpha=0.25, which="both")
    ax1.legend(loc="best")

    line_path = plot_dir / "periodic_wrap_pair_linecut_y0p5.png"
    fig.savefig(line_path, dpi=180)
    plt.close(fig)

    source_field = np.asarray(result["source_values"], dtype=np.float64)
    source_grid, _, _ = _interpolate_periodic_field_to_grid_2d(
        src_points,
        source_field,
        result["cell_size"],
        grid_size,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    src_lim = float(np.percentile(np.abs(source_grid), 99.0))
    if not np.isfinite(src_lim) or src_lim <= 0:
        src_lim = float(np.max(np.abs(source_grid)) + 1.0e-15)
    im = ax.imshow(
        source_grid,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-src_lim,
        vmax=src_lim,
        interpolation="bicubic",
    )
    ax.plot([0, lx, lx, 0, 0], [0, 0, ly, ly, 0], "k--", lw=0.9, alpha=0.7)
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Source density field (interpolated)")
    ax.grid(True, alpha=0.15)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    source_path = plot_dir / "periodic_wrap_pair_sources.png"
    fig.savefig(source_path, dpi=180)
    plt.close(fig)

    cell_boxes = np.asarray(result["cell_boxes"], dtype=np.float64)
    cell_levels = np.asarray(result["cell_levels"], dtype=np.int32)
    grid_segments = []
    grid_colors = []
    for box, lev in zip(cell_boxes, cell_levels, strict=True):
        xmin, ymin, xmax, ymax = box
        grid_segments.extend(
            [
                [(xmin, ymin), (xmax, ymin)],
                [(xmax, ymin), (xmax, ymax)],
                [(xmax, ymax), (xmin, ymax)],
                [(xmin, ymax), (xmin, ymin)],
            ]
        )
        grid_colors.extend([float(lev)] * 4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    ax_grid, ax_hist = axes

    line_width = max(0.15, min(0.45, 8.0 / np.sqrt(max(len(cell_boxes), 1))))
    line_collection = LineCollection(
        grid_segments,
        cmap="viridis",
        linewidths=line_width,
        alpha=0.8,
    )
    line_collection.set_array(np.asarray(grid_colors, dtype=np.float64))
    if len(grid_colors) > 0:
        line_collection.set_clim(
            float(np.min(grid_colors)),
            float(np.max(grid_colors)),
        )
    ax_grid.add_collection(line_collection)
    ax_grid.set_xlim(0, lx)
    ax_grid.set_ylim(0, ly)
    ax_grid.set_aspect("equal")
    ax_grid.set_xlabel("x")
    ax_grid.set_ylabel("y")
    ax_grid.set_title("Adaptive box grid (colored by level)")
    cbar = fig.colorbar(line_collection, ax=ax_grid, fraction=0.046, pad=0.02)
    cbar.set_label("box level")

    if len(cell_levels) > 0:
        unique_levels, counts = np.unique(cell_levels, return_counts=True)
        ax_hist.bar(unique_levels, counts, color="#4c78a8", width=0.8)
        ax_hist.set_xticks(unique_levels)
    ax_hist.set_xlabel("box level")
    ax_hist.set_ylabel("number of active cells")
    ax_hist.set_title("Adaptive level distribution")
    ax_hist.grid(True, axis="y", alpha=0.25)

    grid_path = plot_dir / "periodic_wrap_pair_adaptive_grid.png"
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)

    ref_tiled = np.tile(ref_grid, (3, 3))
    full_tiled = np.tile(full_grid, (3, 3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    tiled_extent = (-lx, 2.0 * lx, -ly, 2.0 * ly)

    im0 = axes[0].imshow(
        ref_tiled,
        origin="lower",
        extent=tiled_extent,
        cmap="coolwarm",
        vmin=-field_lim,
        vmax=field_lim,
        interpolation="bicubic",
    )
    axes[0].set_title("spectral periodic reference (3x3 tiled)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-lx, 2.0 * lx)
    axes[0].set_ylim(-ly, 2.0 * ly)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(
        full_tiled,
        origin="lower",
        extent=tiled_extent,
        cmap="coolwarm",
        vmin=-field_lim,
        vmax=field_lim,
        interpolation="bicubic",
    )
    axes[1].set_title("volumential near+far (3x3 tiled)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-lx, 2.0 * lx)
    axes[1].set_ylim(-ly, 2.0 * ly)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    tiled_path = plot_dir / "periodic_wrap_pair_tiled.png"
    fig.savefig(tiled_path, dpi=180)
    plt.close(fig)

    return [field_path, line_path, source_path, grid_path, tiled_path]


def _run_benchmark_mode(args):
    adaptive_refine = (
        False if args.adaptive_refine is None else bool(args.adaptive_refine)
    )

    print("Periodic Laplace volume benchmark (FMM path + periodic controls)")
    print(f"cache file: {args.cache_file}")
    print(
        "config:",
        {
            "mode": args.mode,
            "dims": args.dims,
            "quad_order": args.quad_order,
            "nlevels": args.nlevels,
            "adaptive_refine": adaptive_refine,
            "adaptive_max_loops": args.adaptive_max_loops,
            "adaptive_tail_tol": args.adaptive_tail_tol,
            "adaptive_top_fraction": args.adaptive_top_fraction,
            "adaptive_bottom_fraction": args.adaptive_bottom_fraction,
            "adaptive_max_cells": args.adaptive_max_cells,
            "fmm_order": args.fmm_order,
            "far_shift_radius": args.far_shift_radius,
            "spectral_kmax_2d": args.spectral_kmax_2d,
            "spectral_kmax_3d": args.spectral_kmax_3d,
        },
    )

    ctx = _create_non_intel_context()
    for dim in args.dims:
        dim = int(dim)
        if dim not in {2, 3}:
            print(f"skip dim={dim}: only 2D/3D supported")
            continue

        result = _solve_volume_case(
            args,
            ctx=ctx,
            dim=dim,
            case_tag=f"benchmark-volume-q{args.quad_order}-l{args.nlevels}",
            density_builder=_build_benchmark_density,
            rng_seed=args.seed + dim,
            adaptive_refine=adaptive_refine,
        )
        _print_case_summary(result)


def _run_wrap_pair_figure_mode(args):
    adaptive_refine = (
        True if args.adaptive_refine is None else bool(args.adaptive_refine)
    )

    print("Periodic Laplace volume wrap-pair reproduction")
    print(f"cache file: {args.cache_file}")
    print(
        "config:",
        {
            "mode": args.mode,
            "quad_order": args.quad_order,
            "nlevels": args.nlevels,
            "adaptive_refine": adaptive_refine,
            "adaptive_max_loops": args.adaptive_max_loops,
            "adaptive_tail_tol": args.adaptive_tail_tol,
            "adaptive_top_fraction": args.adaptive_top_fraction,
            "adaptive_bottom_fraction": args.adaptive_bottom_fraction,
            "adaptive_max_cells": args.adaptive_max_cells,
            "grid_size": args.grid_size,
            "wrap_epsilon": args.wrap_epsilon,
            "background_amplitude": args.background_amplitude,
            "fmm_order": args.fmm_order,
            "far_shift_radius": args.far_shift_radius,
            "spectral_kmax_2d": args.spectral_kmax_2d,
            "reference_kmax_2d": args.kmax_2d,
        },
    )

    ctx = _create_non_intel_context()

    def _density(points, q_weights, rng):
        return _build_wrap_pair_density(
            points,
            q_weights,
            rng,
            wrap_epsilon=args.wrap_epsilon,
            background_amplitude=args.background_amplitude,
        )

    result = _solve_volume_case(
        args,
        ctx=ctx,
        dim=2,
        case_tag=f"wrap-pair-volume-q{args.quad_order}-l{args.nlevels}",
        density_builder=_density,
        rng_seed=args.seed,
        adaptive_refine=adaptive_refine,
    )

    _print_case_summary(result)

    if args.skip_plots:
        print("plot generation skipped (--skip-plots)")
        return

    written = _save_wrap_pair_plots(
        result=result,
        plot_dir=args.plot_dir,
        grid_size=int(args.grid_size),
    )
    print("generated plots:")
    for path in written:
        print(f"- {path}")


def main() -> None:
    args = _parse_args()

    if args.mode == "benchmark":
        _run_benchmark_mode(args)
    elif args.mode == "wrap-pair-figure":
        _run_wrap_pair_figure_mode(args)
    else:
        raise ValueError(f"unsupported mode '{args.mode}'")


if __name__ == "__main__":
    main()
