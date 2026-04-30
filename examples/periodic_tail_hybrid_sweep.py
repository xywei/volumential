"""Parameter/seed sweep for 2D periodic tail hybrid experiment.

This script targets the most promising path identified in
``periodic_tail_hybrid_experiment.py``:

- use central-cell FMM + polynomial periodic tail correction,
- apply dipole-based spectral-convention correction,
- evaluate robustness across multiple random smooth neutral sources.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyopencl as cl
import pyopencl.array

import volumential.meshgen as mg
from volumential.table_manager import NearFieldInteractionTableManager
from volumential.volume_fmm import drive_volume_fmm, interpolate_volume_potential

from examples.periodic_tail_hybrid_experiment import (
    _affine_corrected_error_2d,
    _build_probe_points,
    _build_smooth_neutral_source,
    _build_wrangler,
    _create_non_intel_context,
    _dipole_linear_correction_2d,
    _direct_image_sum_square_2d,
    _fit_inverse_even_power_limit,
    _host_points_to_obj_array,
    _relative_l2_error,
    _ewald_periodic_laplace_reference,
    _spectral_periodic_laplace_reference,
    _to_host_array,
    build_periodic_tail_coefficients_2d,
    evaluate_tail_from_coefficients_2d,
)


def _parse_int_csv(text: str) -> list[int]:
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        raise ValueError("empty integer CSV list")
    return [int(item) for item in items]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-order", type=int, default=4)
    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--fmm-order", type=int, default=16)
    parser.add_argument("--kmax", type=int, default=80)
    parser.add_argument("--probe-grid-size", type=int, default=25)

    parser.add_argument("--seeds", default="3,7,11,17,23")
    parser.add_argument("--near-radii", default="1,2")
    parser.add_argument("--max-orders", default="8,10")

    parser.add_argument("--tail-exp", type=float, default=40.0)
    parser.add_argument("--eta-fit-points", type=int, default=8)
    parser.add_argument("--high-order-start-R", type=int, default=256)
    parser.add_argument("--high-order-max-R", type=int, default=32768)
    parser.add_argument("--high-order-selftol", type=float, default=1.0e-13)
    parser.add_argument(
        "--high-order-method",
        choices=["hard_richardson", "eisenstein"],
        default="eisenstein",
        help="estimator for even derivative sums with order > 2",
    )
    parser.add_argument(
        "--high-order-mp-dps",
        type=int,
        default=80,
        help="mpmath precision (decimal digits) for high-order eisenstein sums",
    )

    parser.add_argument(
        "--zero-mean-after-dipole-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove probe-grid mean after dipole linear correction",
    )
    parser.add_argument(
        "--tail-eval-extended-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="evaluate polynomial tail in np.longdouble before casting to float64",
    )
    parser.add_argument(
        "--pde-far-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "build a PDE-based non-central reference from periodic solution "
            "minus central-cell and near-image contributions"
        ),
    )
    parser.add_argument(
        "--pde-reference-apply-dipole",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "apply dipole linear correction when converting spectral PDE "
            "reference to square-sum convention"
        ),
    )
    parser.add_argument(
        "--pde-reference-kmax",
        type=int,
        default=0,
        help=(
            "spectral truncation for PDE far reference; if <=0, uses --kmax"
        ),
    )
    parser.add_argument(
        "--pde-reference-method",
        choices=["spectral", "ewald"],
        default="spectral",
        help="backend used to build PDE full-reference potential",
    )
    parser.add_argument(
        "--ewald-xi",
        type=float,
        default=8.0,
        help="Ewald splitting parameter xi for PDE reference",
    )
    parser.add_argument(
        "--ewald-real-cutoff-r",
        type=int,
        default=6,
        help="real-space image cutoff radius for Ewald PDE reference",
    )
    parser.add_argument(
        "--ewald-kmax",
        type=int,
        default=64,
        help="reciprocal-space mode cutoff radius for Ewald PDE reference",
    )
    parser.add_argument(
        "--spectral-chunk-size",
        type=int,
        default=8192,
        help=(
            "number of Fourier modes per chunk in spectral reference evaluation; "
            "use <=0 to process all modes at once"
        ),
    )
    parser.add_argument(
        "--spectral-compensated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use compensated accumulation for spectral periodic references",
    )
    parser.add_argument(
        "--spectral-source-block-size",
        type=int,
        default=0,
        help=(
            "if >0, accumulate spectral source modes in source blocks; "
            "0 keeps dense source-by-mode matmul"
        ),
    )
    parser.add_argument(
        "--spectral-accum-extended-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="accumulate spectral references in np.longdouble",
    )
    parser.add_argument(
        "--direct-sum-source-block-size",
        type=int,
        default=0,
        help=(
            "if >0, evaluate direct image sums in source blocks to reduce memory "
            "and improve compensated summation"
        ),
    )
    parser.add_argument(
        "--direct-sum-extended-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="accumulate direct image sums in np.longdouble",
    )
    parser.add_argument(
        "--cache-file",
        default="nft-periodic-tail-hybrid.sqlite",
        help="near-field table cache filename",
    )
    parser.add_argument(
        "--direct-far-cutoff-r",
        type=int,
        default=0,
        help=(
            "if >0, compute one-seed direct far-image reference (R,2R extrapolated) "
            "for each (near_radius,max_order)"
        ),
    )
    parser.add_argument(
        "--direct-far-second-cutoff-r",
        type=int,
        default=0,
        help=(
            "optional second extrapolation radius for direct-far checks; "
            "if not set, uses 2*direct-far-cutoff-r"
        ),
    )
    parser.add_argument(
        "--direct-far-fit-order",
        type=int,
        default=1,
        help="inverse-even-power extrapolation order for direct-far checks",
    )
    parser.add_argument(
        "--direct-far-num-radii",
        type=int,
        default=2,
        help="number of direct-far radii used in extrapolation checks",
    )
    return parser.parse_args()


def _summarize(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))


def main() -> None:
    args = _parse_args()

    seeds = _parse_int_csv(args.seeds)
    near_radii = _parse_int_csv(args.near_radii)
    max_orders = _parse_int_csv(args.max_orders)

    if any(rad < 0 for rad in near_radii):
        raise ValueError("near-radii must be non-negative")
    if any(order < 0 for order in max_orders):
        raise ValueError("max-orders must be non-negative")
    if int(args.spectral_source_block_size) < 0:
        raise ValueError("spectral-source-block-size must be >= 0")
    if int(args.direct_sum_source_block_size) < 0:
        raise ValueError("direct-sum-source-block-size must be >= 0")

    ctx = _create_non_intel_context()
    queue = cl.CommandQueue(ctx)

    mesh = mg.MeshGen2D(int(args.q_order), int(args.nlevels), 0.0, 1.0, queue=queue)
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        2,
        int(args.q_order),
        mesh,
        bbox=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
    )

    source_points = np.ascontiguousarray(
        np.stack([q_points[0].get(queue), q_points[1].get(queue)], axis=1),
        dtype=np.float64,
    )
    q_weights_host = np.asarray(q_weights.get(queue), dtype=np.float64)

    probe_points = _build_probe_points(
        cell_size=float(tree.root_extent),
        grid_size=int(args.probe_grid_size),
    )
    probe_points_dev = _host_points_to_obj_array(probe_points, queue)

    cell_size = float(tree.root_extent)
    center = np.asarray([0.5 * cell_size, 0.5 * cell_size], dtype=np.float64)

    spectral_chunk_size = int(args.spectral_chunk_size)
    spectral_compensated = bool(args.spectral_compensated)
    spectral_source_block_size = int(args.spectral_source_block_size)
    spectral_accum_extended_precision = bool(args.spectral_accum_extended_precision)
    direct_sum_source_block_size = int(args.direct_sum_source_block_size)
    direct_sum_extended_precision = bool(args.direct_sum_extended_precision)
    pde_reference_method = str(args.pde_reference_method)
    pde_ref_kmax = int(args.pde_reference_kmax)
    if pde_ref_kmax <= 0:
        pde_ref_kmax = int(args.kmax)

    seed_data: dict[int, dict] = {}

    with NearFieldInteractionTableManager(
        args.cache_file,
        root_extent=float(tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(2, "Laplace", q_order=int(args.q_order), queue=queue)
        wrangler = _build_wrangler(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            near_field_table=table,
            fmm_order=int(args.fmm_order),
            quad_order=int(args.q_order),
        )

        print("precomputing per-seed central/ref data...")
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            source_values_host = _build_smooth_neutral_source(source_points, q_weights_host, rng)
            source_values_dev = cl.array.to_device(
                queue,
                np.ascontiguousarray(source_values_host, dtype=np.float64),
            )
            source_strengths_dev = source_values_dev * q_weights
            source_strengths_host = np.asarray(source_values_host * q_weights_host, dtype=np.float64)

            (pot_center_tree,) = drive_volume_fmm(
                traversal,
                wrangler,
                source_strengths_dev,
                source_values_dev,
                direct_evaluation=False,
                auto_interpolate_targets=False,
                reorder_potentials=False,
            )
            pot_center = _to_host_array(
                interpolate_volume_potential(
                    probe_points_dev,
                    traversal,
                    wrangler,
                    pot_center_tree,
                    potential_in_tree_order=True,
                    use_mode_to_source_ids=True,
                ),
                queue,
            ).astype(np.float64)

            pot_near_by_radius = {}
            for radius in near_radii:
                if int(radius) > 0:
                    pot_near_by_radius[int(radius)] = _direct_image_sum_square_2d(
                        src_points=source_points,
                        strengths=source_strengths_host,
                        tgt_points=probe_points,
                        cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                        min_radius=1,
                        max_radius=int(radius),
                        source_block_size=direct_sum_source_block_size,
                        use_extended_precision=direct_sum_extended_precision,
                    )
                else:
                    pot_near_by_radius[int(radius)] = np.zeros(len(probe_points), dtype=np.float64)

            pot_ref = _spectral_periodic_laplace_reference(
                src_points=source_points,
                strengths=source_strengths_host,
                tgt_points=probe_points,
                cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                k_max=int(args.kmax),
                chunk_size=spectral_chunk_size,
                compensated=spectral_compensated,
                source_block_size=spectral_source_block_size,
                accum_extended_precision=spectral_accum_extended_precision,
            )

            if pde_reference_method == "spectral":
                if int(pde_ref_kmax) == int(args.kmax):
                    pot_ref_pde = pot_ref
                else:
                    pot_ref_pde = _spectral_periodic_laplace_reference(
                        src_points=source_points,
                        strengths=source_strengths_host,
                        tgt_points=probe_points,
                        cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                        k_max=int(pde_ref_kmax),
                        chunk_size=spectral_chunk_size,
                        compensated=spectral_compensated,
                        source_block_size=spectral_source_block_size,
                        accum_extended_precision=spectral_accum_extended_precision,
                    )
            elif pde_reference_method == "ewald":
                pot_ref_pde = _ewald_periodic_laplace_reference(
                    src_points=source_points,
                    strengths=source_strengths_host,
                    tgt_points=probe_points,
                    cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                    xi=float(args.ewald_xi),
                    real_cutoff_r=int(args.ewald_real_cutoff_r),
                    k_max=int(args.ewald_kmax),
                    mode_chunk_size=spectral_chunk_size,
                )
            else:
                raise ValueError(f"unknown pde-reference-method: {pde_reference_method}")

            seed_data[int(seed)] = {
                "source_strengths": source_strengths_host,
                "pot_center": pot_center,
                "pot_near_by_radius": pot_near_by_radius,
                "pot_ref": pot_ref,
                "pot_ref_pde": pot_ref_pde,
            }

        best_key = None
        best_metric = np.inf
        best_direct_key = None
        best_direct_metric = np.inf
        best_pde_key = None
        best_pde_metric = np.inf

        print("")
        print("=== periodic-tail hybrid sweep (fmm central) ===")
        print(
            f"n_modes={len(source_points)} probe_points={len(probe_points)} "
            f"q_order={args.q_order} nlevels={args.nlevels} kmax={args.kmax}"
        )
        print(f"spectral_chunk_size={spectral_chunk_size}")
        print(
            "direct_image_sum="
            f"block_size={direct_sum_source_block_size} "
            f"extended_precision={direct_sum_extended_precision}"
        )
        print(
            "spectral_reference="
            f"compensated={spectral_compensated} "
            f"source_block_size={spectral_source_block_size} "
            f"accum_extended_precision={spectral_accum_extended_precision}"
        )
        print(
            f"seeds={seeds} near_radii={near_radii} max_orders={max_orders} "
            f"zero_mean_after_dipole={bool(args.zero_mean_after_dipole_correction)}"
        )
        print("noncentral_images=direct_near_shell + tail (shifted-image P2P disabled)")
        print(
            "high-order derivative sum method: "
            f"{args.high_order_method} (mp_dps={int(args.high_order_mp_dps)})"
        )
        print(
            "tail_eval_extended_precision="
            f"{bool(args.tail_eval_extended_precision)}"
        )
        print(
            "pde_far_reference="
            f"{bool(args.pde_far_reference)} "
            f"pde_reference_method={pde_reference_method} "
            f"pde_reference_apply_dipole={bool(args.pde_reference_apply_dipole)}"
        )
        if pde_reference_method == "spectral":
            print(f"pde_reference_kmax={int(pde_ref_kmax)}")
        else:
            print(
                "ewald params: "
                f"xi={float(args.ewald_xi):.6g} "
                f"real_cutoff_r={int(args.ewald_real_cutoff_r)} "
                f"kmax={int(args.ewald_kmax)}"
            )
        if int(args.direct_far_cutoff_r) > 0:
            print(
                f"direct far-image check enabled: cutoff_r={int(args.direct_far_cutoff_r)} "
                f"seed={int(seeds[0])}"
            )
        print("")

        for near_radius in near_radii:
            direct_far_meta = None
            if int(args.direct_far_cutoff_r) > int(near_radius):
                fit_order = max(1, int(args.direct_far_fit_order))
                num_radii = max(2, int(args.direct_far_num_radii))

                r1 = int(args.direct_far_cutoff_r)
                r2 = int(args.direct_far_second_cutoff_r)
                if r2 <= 0:
                    r2 = 2 * r1
                if r2 <= r1:
                    raise ValueError(
                        "direct-far-second-cutoff-r must be > direct-far-cutoff-r"
                    )

                radii = sorted(set([int(r1), int(r2)]))
                while len(radii) < num_radii:
                    radii.append(int(2 * radii[-1]))

                seed0 = int(seeds[0])
                data0 = seed_data[seed0]
                t_direct_start = time.perf_counter()

                min_radius = int(near_radius) + 1
                far_running = np.zeros(len(probe_points), dtype=np.float64)
                far_running_comp = np.zeros_like(far_running)
                far_values_by_radius: dict[int, np.ndarray] = {}
                next_shell = int(min_radius)

                for radius in radii:
                    for shell_radius in range(next_shell, int(radius) + 1):
                        shell_val = _direct_image_sum_square_2d(
                            src_points=source_points,
                            strengths=data0["source_strengths"],
                            tgt_points=probe_points,
                            cell_size=np.asarray([cell_size, cell_size], dtype=np.float64),
                            min_radius=int(shell_radius),
                            max_radius=int(shell_radius),
                            source_block_size=direct_sum_source_block_size,
                            use_extended_precision=direct_sum_extended_precision,
                        )
                        y = shell_val - far_running_comp
                        t = far_running + y
                        far_running_comp = (t - far_running) - y
                        far_running = t

                    next_shell = int(radius) + 1
                    far_values_by_radius[int(radius)] = far_running.copy()

                far_values = [far_values_by_radius[int(radius)] for radius in radii]
                direct_ref_time = float(time.perf_counter() - t_direct_start)

                far_ref = _fit_inverse_even_power_limit(
                    radii=radii,
                    values=far_values,
                    fit_order=fit_order,
                )

                if len(radii) >= 3:
                    prev_ref = _fit_inverse_even_power_limit(
                        radii=radii[:-1],
                        values=far_values[:-1],
                        fit_order=fit_order,
                    )
                    direct_ref_self_rel = _relative_l2_error(prev_ref, far_ref)
                else:
                    direct_ref_self_rel = float("inf")

                direct_far_meta = {
                    "seed0": int(seed0),
                    "fit_order": int(min(fit_order, len(radii) - 1)),
                    "radii": list(radii),
                    "far_ref": far_ref,
                    "ref_time": float(direct_ref_time),
                    "self_rel": float(direct_ref_self_rel),
                }
                print(
                    f"near_radius={int(near_radius):2d} direct-far precompute: "
                    f"radii={radii} fit_order={min(fit_order, len(radii)-1)} "
                    f"self_rel={direct_ref_self_rel:.3e} ref_s={direct_ref_time:.2f}"
                )

            for max_order in max_orders:
                t_coeff_start = time.perf_counter()
                coeffs, _ = build_periodic_tail_coefficients_2d(
                    max_order=int(max_order),
                    near_radius=int(near_radius),
                    cell_size=cell_size,
                    tail_exp=float(args.tail_exp),
                    eta_fit_points=int(args.eta_fit_points),
                    high_order_start_r=int(args.high_order_start_R),
                    high_order_max_r=int(args.high_order_max_R),
                    high_order_selftol=float(args.high_order_selftol),
                    high_order_method=str(args.high_order_method),
                    high_order_mp_dps=int(args.high_order_mp_dps),
                )
                t_coeff = time.perf_counter() - t_coeff_start

                raw_rel_list = []
                dipole_rel_list = []
                affine_rel_list = []
                dipole_linf_list = []
                affine_linf_list = []
                tail_pde_rel_list = []
                tail_pde_rel_zm_list = []
                tail_pde_rel_affine_list = []
                tail_pde_abs_l2_list = []
                tail_pde_ref_l2_list = []
                pot_tail_by_seed = {}

                for seed in seeds:
                    data = seed_data[int(seed)]

                    pot_tail = evaluate_tail_from_coefficients_2d(
                        source_points=source_points,
                        source_strengths=data["source_strengths"],
                        target_points=probe_points,
                        coeffs=coeffs,
                        max_order=int(max_order),
                        center=center,
                        use_extended_precision=bool(args.tail_eval_extended_precision),
                    )
                    pot_tail_by_seed[int(seed)] = pot_tail

                    pot_hybrid_raw = (
                        data["pot_center"]
                        + data["pot_near_by_radius"][int(near_radius)]
                        + pot_tail
                    )

                    dipole_corr, _ = _dipole_linear_correction_2d(
                        source_points=source_points,
                        source_strengths=data["source_strengths"],
                        target_points=probe_points,
                        center=center,
                        cell_area=cell_size * cell_size,
                    )

                    if bool(args.pde_far_reference):
                        pde_full_model = data["pot_ref_pde"]
                        if bool(args.pde_reference_apply_dipole):
                            pde_full_model = pde_full_model - dipole_corr

                        pde_far_ref = (
                            pde_full_model
                            - data["pot_center"]
                            - data["pot_near_by_radius"][int(near_radius)]
                        )
                        tail_zm = pot_tail - np.mean(pot_tail)
                        pde_zm = pde_far_ref - np.mean(pde_far_ref)
                        aff = _affine_corrected_error_2d(
                            values=pot_tail,
                            reference=pde_far_ref,
                            points=probe_points,
                            center=center,
                        )

                        tail_pde_rel_list.append(
                            _relative_l2_error(pot_tail, pde_far_ref)
                        )
                        tail_pde_rel_zm_list.append(
                            _relative_l2_error(tail_zm, pde_zm)
                        )
                        tail_pde_rel_affine_list.append(float(aff["rel_l2"]))
                        tail_pde_abs_l2_list.append(
                            float(np.linalg.norm(pot_tail - pde_far_ref))
                        )
                        tail_pde_ref_l2_list.append(float(np.linalg.norm(pde_far_ref)))

                    pot_hybrid_dipole = pot_hybrid_raw + dipole_corr
                    if bool(args.zero_mean_after_dipole_correction):
                        pot_hybrid_dipole -= np.mean(pot_hybrid_dipole)

                    raw_rel_list.append(_relative_l2_error(pot_hybrid_raw, data["pot_ref"]))

                    dipole_rel_list.append(
                        _relative_l2_error(pot_hybrid_dipole, data["pot_ref"])
                    )
                    dipole_linf_list.append(
                        float(np.max(np.abs(pot_hybrid_dipole - data["pot_ref"])))
                    )

                    aff = _affine_corrected_error_2d(
                        values=pot_hybrid_raw,
                        reference=data["pot_ref"],
                        points=probe_points,
                        center=center,
                    )
                    affine_rel_list.append(float(aff["rel_l2"]))
                    affine_linf_list.append(float(aff["linf"]))

                raw_mean, raw_std, raw_max = _summarize(raw_rel_list)
                dip_mean, dip_std, dip_max = _summarize(dipole_rel_list)
                aff_mean, aff_std, aff_max = _summarize(affine_rel_list)
                dip_linf_mean, dip_linf_std, dip_linf_max = _summarize(dipole_linf_list)
                aff_linf_mean, aff_linf_std, aff_linf_max = _summarize(affine_linf_list)
                tail_pde_mean = None
                tail_pde_std = None
                tail_pde_max = None
                tail_pde_zm_mean = None
                tail_pde_zm_std = None
                tail_pde_zm_max = None
                tail_pde_affine_mean = None
                tail_pde_affine_std = None
                tail_pde_affine_max = None
                tail_pde_abs_mean = None
                tail_pde_ref_mean = None
                if tail_pde_rel_list:
                    tail_pde_mean, tail_pde_std, tail_pde_max = _summarize(tail_pde_rel_list)
                    tail_pde_zm_mean, tail_pde_zm_std, tail_pde_zm_max = _summarize(
                        tail_pde_rel_zm_list
                    )
                    (
                        tail_pde_affine_mean,
                        tail_pde_affine_std,
                        tail_pde_affine_max,
                    ) = _summarize(tail_pde_rel_affine_list)
                    tail_pde_abs_mean, _, _ = _summarize(tail_pde_abs_l2_list)
                    tail_pde_ref_mean, _, _ = _summarize(tail_pde_ref_l2_list)

                direct_full_rel = None
                direct_tail_rel = None
                direct_tail_abs_l2 = None
                direct_tail_ref_l2 = None
                direct_ref_time = None
                direct_ref_self_rel = None
                if direct_far_meta is not None:
                    seed0 = int(direct_far_meta["seed0"])
                    data0 = seed_data[seed0]
                    far_ref = np.asarray(direct_far_meta["far_ref"], dtype=np.float64)

                    direct_ref_time = float(direct_far_meta["ref_time"])
                    direct_ref_self_rel = float(direct_far_meta["self_rel"])

                    hybrid_seed0 = (
                        data0["pot_center"]
                        + data0["pot_near_by_radius"][int(near_radius)]
                        + pot_tail_by_seed[seed0]
                    )
                    direct_full_ref = (
                        data0["pot_center"]
                        + data0["pot_near_by_radius"][int(near_radius)]
                        + far_ref
                    )

                    direct_full_rel = _relative_l2_error(hybrid_seed0, direct_full_ref)
                    direct_tail_rel = _relative_l2_error(pot_tail_by_seed[seed0], far_ref)
                    direct_tail_abs_l2 = float(
                        np.linalg.norm(pot_tail_by_seed[seed0] - far_ref)
                    )
                    direct_tail_ref_l2 = float(np.linalg.norm(far_ref))

                print(
                    f"near_radius={near_radius:2d} max_order={max_order:2d} "
                    f"coeff_s={t_coeff:7.2f} | "
                    f"raw rel_l2 mean/std/max={raw_mean:.3e}/{raw_std:.1e}/{raw_max:.3e} | "
                    f"dipole rel_l2 mean/std/max={dip_mean:.3e}/{dip_std:.1e}/{dip_max:.3e} | "
                    f"affine-fit rel_l2 mean/std/max={aff_mean:.3e}/{aff_std:.1e}/{aff_max:.3e}"
                )
                print(
                    " " * 33
                    + f"dipole linf mean/std/max={dip_linf_mean:.3e}/{dip_linf_std:.1e}/{dip_linf_max:.3e} | "
                    + f"affine-fit linf mean/std/max={aff_linf_mean:.3e}/{aff_linf_std:.1e}/{aff_linf_max:.3e}"
                )
                if tail_pde_mean is not None:
                    print(
                        " " * 33
                        + f"tail-vs-pde rel_l2 mean/std/max={tail_pde_mean:.3e}/{tail_pde_std:.1e}/{tail_pde_max:.3e} | "
                        + f"zero-mean rel_l2 mean/std/max={tail_pde_zm_mean:.3e}/{tail_pde_zm_std:.1e}/{tail_pde_zm_max:.3e} | "
                        + f"affine-fit rel_l2 mean/std/max={tail_pde_affine_mean:.3e}/{tail_pde_affine_std:.1e}/{tail_pde_affine_max:.3e}"
                    )
                    print(
                        " " * 33
                        + f"tail-vs-pde abs_l2 mean={tail_pde_abs_mean:.3e} ref_l2 mean={tail_pde_ref_mean:.3e}"
                    )
                if direct_full_rel is not None:
                    print(
                        " " * 33
                        + f"direct-full rel_l2(seed={seed0})={direct_full_rel:.3e} | "
                        + f"tail-vs-direct rel_l2(seed={seed0})={direct_tail_rel:.3e} | "
                        + f"tail-vs-direct abs_l2={direct_tail_abs_l2:.3e} "
                        + f"ref_l2={direct_tail_ref_l2:.3e} | "
                        + f"fit_order={direct_far_meta['fit_order']} radii={direct_far_meta['radii']} | "
                        + f"direct-ref-self rel_l2={direct_ref_self_rel:.3e} | "
                        + f"direct_ref_s={direct_ref_time:.2f}"
                    )

                key = (int(near_radius), int(max_order))
                if dip_mean < best_metric:
                    best_metric = dip_mean
                    best_key = key
                if direct_tail_rel is not None and float(direct_tail_rel) < best_direct_metric:
                    best_direct_metric = float(direct_tail_rel)
                    best_direct_key = key
                if tail_pde_affine_mean is not None and float(tail_pde_affine_mean) < best_pde_metric:
                    best_pde_metric = float(tail_pde_affine_mean)
                    best_pde_key = key

        print("")
        if best_key is not None:
            print(
                "best by dipole-corrected rel_l2 mean: "
                f"near_radius={best_key[0]} max_order={best_key[1]} "
                f"mean_rel_l2={best_metric:.3e}"
            )
        if best_direct_key is not None:
            print(
                "best by tail-vs-direct rel_l2 (seed0): "
                f"near_radius={best_direct_key[0]} max_order={best_direct_key[1]} "
                f"rel_l2={best_direct_metric:.3e}"
            )
        if best_pde_key is not None:
            print(
                "best by tail-vs-pde affine-fit rel_l2 mean: "
                f"near_radius={best_pde_key[0]} max_order={best_pde_key[1]} "
                f"mean_rel_l2={best_pde_metric:.3e}"
            )


if __name__ == "__main__":
    main()
