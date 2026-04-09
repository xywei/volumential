"""Helmholtz 2D split-order p-convergence study with plot output.

This example evaluates a manufactured exact-solution case

    u(x, y) = exp(-alpha * (x^2 + y^2))

with right-hand side

    rho = (-Delta - k^2) u

and reports relative L2/Linf errors of the computed volume potential versus
``u`` for a sweep over ``helmholtz_split_order``.

By default this reproduces the tuned case:

    q=7, nlevels=4, fmm_order=20, k=20, alpha=120, split_orders=1..6

Output files:
- JSON summary
- CSV table (plot-friendly)
- PNG p-convergence plot (if matplotlib is available)

Set ``VOLUMENTIAL_EXAMPLE_SMOKE=1`` for a lighter run.
"""

from __future__ import annotations

__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import csv
import json
import logging
import os
import time
from functools import partial
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def _is_smoke_mode() -> bool:
    return os.environ.get("VOLUMENTIAL_EXAMPLE_SMOKE", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _parse_split_orders(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))

    if not vals:
        raise ValueError("split order list must not be empty")
    if any(v < 1 for v in vals):
        raise ValueError("split orders must be >= 1")

    return vals


def _get_laplace_table(queue, table_filename: str, q_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    regular_quad_order = max(8, 4 * q_order)
    radial_quad_order = max(21, 10 * q_order)

    with NearFieldInteractionTableManager(
        table_filename,
        root_extent=2.0,
        dtype=np.float64,
        queue=queue,
    ) as tm:
        table, _ = tm.get_table(
            2,
            "Laplace",
            q_order,
            queue=queue,
            build_config=DuffyBuildConfig(
                radial_rule="tanh-sinh-fast",
                regular_quad_order=regular_quad_order,
                radial_quad_order=radial_quad_order,
            ),
        )

    return table


def run_split_p_convergence(
    *,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    wave_number: float,
    alpha: float,
    split_orders: list[int],
    table_cache: str,
):
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import HelmholtzKernel

    import pyopencl as cl

    import volumential.meshgen as mg

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )
    from volumential.volume_fmm import drive_volume_fmm

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mesh = mg.MeshGen2D(q_order, nlevels, -0.5, 0.5, queue=queue)
    q_points, source_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        2,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * 2, dtype=np.float64),
    )

    coords = np.array([axis.get(queue) for axis in q_points])
    r2 = coords[0] * coords[0] + coords[1] * coords[1]

    u_exact = np.exp(-alpha * r2)
    rhs = (4 * alpha - 4 * alpha * alpha * r2 - wave_number * wave_number) * np.exp(
        -alpha * r2
    )

    source_vals = cl.array.to_device(
        queue,
        np.ascontiguousarray(rhs.astype(np.complex128)),
    )
    weighted_sources = source_vals * source_weights

    knl = HelmholtzKernel(2)
    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(knl)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        [knl],
        exclude_self=True,
    )

    laplace_table = _get_laplace_table(queue, table_cache, q_order)

    def make_wrangler(split_order):
        self_extra_kwargs = {}
        if tree.sources_are_targets:
            self_extra_kwargs["target_to_source"] = np.arange(
                tree.ntargets, dtype=np.int32
            )

        return FPNDExpansionWrangler(
            tree_indep=tree_indep,
            queue=queue,
            traversal=traversal,
            near_field_table=laplace_table,
            dtype=np.complex128,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
            quad_order=q_order,
            kernel_extra_kwargs={knl.helmholtz_k_name: wave_number},
            self_extra_kwargs=self_extra_kwargs,
            helmholtz_split=True,
            helmholtz_split_order=split_order,
            helmholtz_split_smooth_quad_order=q_order,
        )

    _ = make_wrangler(max(split_orders))

    rows = []
    for split_order in split_orders:
        wrangler = make_wrangler(split_order)

        _ = drive_volume_fmm(
            traversal,
            wrangler,
            weighted_sources,
            source_vals,
            direct_evaluation=False,
            list1_only=False,
        )
        queue.finish()

        t0 = time.perf_counter()
        (potentials,) = drive_volume_fmm(
            traversal,
            wrangler,
            weighted_sources,
            source_vals,
            direct_evaluation=False,
            list1_only=False,
        )
        queue.finish()
        warm_solve_seconds = time.perf_counter() - t0

        uh = potentials.get(queue)
        abs_err = np.abs(uh - u_exact)

        rows.append(
            {
                "split_order": int(split_order),
                "rel_l2": float(np.linalg.norm(abs_err) / np.linalg.norm(u_exact)),
                "rel_linf": float(
                    np.linalg.norm(abs_err, ord=np.inf)
                    / np.linalg.norm(u_exact, ord=np.inf)
                ),
                "warm_solve_seconds": float(warm_solve_seconds),
            }
        )

    for i, row in enumerate(rows):
        if i == 0:
            row["pct_improve_vs_prev"] = None
        else:
            prev = rows[i - 1]["rel_l2"]
            cur = row["rel_l2"]
            row["pct_improve_vs_prev"] = float((prev - cur) / prev * 100.0)

    return {
        "device": {
            "name": queue.device.name.strip(),
            "platform": queue.device.platform.name.strip(),
        },
        "case": {
            "q_order": int(q_order),
            "nlevels": int(nlevels),
            "fmm_order": int(fmm_order),
            "k": float(wave_number),
            "alpha": float(alpha),
            "n_points": int(u_exact.size),
            "smooth_quad_order": int(q_order),
        },
        "rows": rows,
    }


def _write_csv(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split_order",
                "rel_l2",
                "rel_linf",
                "warm_solve_seconds",
                "pct_improve_vs_prev",
            ]
        )
        for row in payload["rows"]:
            writer.writerow(
                [
                    row["split_order"],
                    f"{row['rel_l2']:.16e}",
                    f"{row['rel_linf']:.16e}",
                    f"{row['warm_solve_seconds']:.9f}",
                    ""
                    if row["pct_improve_vs_prev"] is None
                    else f"{row['pct_improve_vs_prev']:.6f}",
                ]
            )


def _make_plot(payload, path: Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.warning("matplotlib not available; skipping plot generation")
        return False

    split_orders = [row["split_order"] for row in payload["rows"]]
    rel_l2 = [row["rel_l2"] for row in payload["rows"]]
    solve_s = [row["warm_solve_seconds"] for row in payload["rows"]]

    fig, ax1 = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    ax1.plot(split_orders, rel_l2, marker="o", linewidth=1.8, label="rel L2")
    ax1.set_yscale("log")
    ax1.set_xlabel("split order p")
    ax1.set_ylabel("relative L2 error")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.55)

    ax2 = ax1.twinx()
    ax2.plot(
        split_orders,
        solve_s,
        marker="s",
        linewidth=1.4,
        color="tab:orange",
        label="warm solve seconds",
    )
    ax2.set_ylabel("warm solve seconds")

    case = payload.get("case", {})
    title = (
        "Helmholtz 2D split-order p-convergence"
        f" (q={case.get('q_order')}, nlevels={case.get('nlevels')},"
        f" m={case.get('fmm_order')}, k={case.get('k')}, alpha={case.get('alpha')})"
    )
    ax1.set_title(title)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _print_table(payload):
    print(
        "split_order  rel_l2            rel_linf          warm_s   improve_vs_prev(%)"
    )
    for row in payload["rows"]:
        improve = row["pct_improve_vs_prev"]
        improve_str = "-" if improve is None else f"{improve: .6f}"
        print(
            f"{row['split_order']:11d}  "
            f"{row['rel_l2']:.6e}  "
            f"{row['rel_linf']:.6e}  "
            f"{row['warm_solve_seconds']:.3f}  "
            f"{improve_str}"
        )


def _parse_args():
    smoke = _is_smoke_mode()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-order", type=int, default=5 if smoke else 7)
    parser.add_argument("--nlevels", type=int, default=3 if smoke else 4)
    parser.add_argument("--fmm-order", type=int, default=14 if smoke else 20)
    parser.add_argument("--k", type=float, default=12.0 if smoke else 20.0)
    parser.add_argument("--alpha", type=float, default=80.0 if smoke else 120.0)
    parser.add_argument("--split-orders", default="1,2,3,4,5,6")
    parser.add_argument(
        "--table-cache",
        default=(
            "nft_laplace2d_split_p_convergence_smoke.sqlite"
            if smoke
            else "nft_laplace2d_split_p_convergence.sqlite"
        ),
    )
    parser.add_argument(
        "--output-json",
        default=(
            "helmholtz2d_split_p_convergence_smoke.json"
            if smoke
            else "helmholtz2d_split_p_convergence.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "helmholtz2d_split_p_convergence_smoke.csv"
            if smoke
            else "helmholtz2d_split_p_convergence.csv"
        ),
    )
    parser.add_argument(
        "--output-plot",
        default=(
            "helmholtz2d_split_p_convergence_smoke.png"
            if smoke
            else "helmholtz2d_split_p_convergence.png"
        ),
    )
    parser.add_argument(
        "--input-json",
        help="Skip compute and load an existing JSON payload before CSV/plot output.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)

    args = _parse_args()
    split_orders = _parse_split_orders(args.split_orders)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_plot = Path(args.output_plot)

    if args.input_json:
        with Path(args.input_json).open() as f:
            payload = json.load(f)
    else:
        payload = run_split_p_convergence(
            q_order=args.q_order,
            nlevels=args.nlevels,
            fmm_order=args.fmm_order,
            wave_number=args.k,
            alpha=args.alpha,
            split_orders=split_orders,
            table_cache=args.table_cache,
        )

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote JSON: %s", output_json)

    _write_csv(payload, output_csv)
    logger.info("Wrote CSV: %s", output_csv)

    if not args.no_plot:
        if _make_plot(payload, output_plot):
            logger.info("Wrote plot: %s", output_plot)

    _print_table(payload)


if __name__ == "__main__":
    main()
