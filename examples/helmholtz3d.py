"""Helmholtz 3D volume potential convergence study.

This example evaluates a manufactured-source PDE residual
``(-Delta - k^2)u - rho`` on an interior calculus patch for a complex-valued
source over [-0.5, 0.5]^3.

Set ``VOLUMENTIAL_EXAMPLE_SMOKE=1`` for a lighter run.
"""

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

import logging
import os
import time
from functools import partial

import numpy as np
import pyopencl as cl

import volumential.meshgen as mg


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _is_smoke_mode():
    return os.environ.get("VOLUMENTIAL_EXAMPLE_SMOKE", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _int_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer %s=%s", name, raw)
        return default


def _float_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float %s=%s", name, raw)
        return default


def _get_laplace_table(queue, table_filename, q_order):
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    if q_order <= 1:
        regular_quad_order = 6
        radial_quad_order = 21
    else:
        regular_quad_order = 8
        radial_quad_order = 31

    with NearFieldInteractionTableManager(
        table_filename,
        root_extent=2.0,
        dtype=np.float64,
        queue=queue,
    ) as tm:
        table, _ = tm.get_table(
            3,
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


def _source_profile(x, y, z):
    amp = np.exp(-14.0 * ((x + 0.1) ** 2 + (y - 0.08) ** 2 + (z + 0.06) ** 2))
    phase = 0.9 * x - 0.5 * y + 0.3 * z
    return amp * (1.0 + 0.35j * np.cos(phase))


def _run_case(
    ctx,
    queue,
    near_field_table,
    *,
    q_order,
    nlevels,
    fmm_order,
    wave_number,
):
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import HelmholtzKernel
    from sumpy.point_calculus import CalculusPatch

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )
    from volumential.volume_fmm import drive_volume_fmm, interpolate_volume_potential

    dim = 3
    mesh = mg.MeshGen3D(q_order, nlevels, -0.5, 0.5, queue=queue)
    q_points, source_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        dim,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * dim, dtype=np.float64),
    )

    coords = np.array([axis.get(queue) for axis in q_points])
    x = coords[0]
    y = coords[1]
    z = coords[2]
    source_vals_host = _source_profile(x, y, z)
    source_vals = cl.array.to_device(
        queue,
        np.ascontiguousarray(source_vals_host.astype(np.complex128)),
    )

    knl = HelmholtzKernel(dim)
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

    self_extra_kwargs = {}
    if tree.sources_are_targets:
        self_extra_kwargs["target_to_source"] = np.arange(tree.ntargets, dtype=np.int32)

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=near_field_table,
        dtype=np.complex128,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        kernel_extra_kwargs={knl.helmholtz_k_name: wave_number},
        self_extra_kwargs=self_extra_kwargs,
        helmholtz_split=True,
    )

    weighted_sources = source_vals * source_weights
    t_start = time.perf_counter()
    (fmm_potentials,) = drive_volume_fmm(
        traversal,
        wrangler,
        weighted_sources,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )
    t_fmm = time.perf_counter() - t_start

    t_start = time.perf_counter()
    patch = CalculusPatch(center=[0.0, 0.0, 0.0], h=0.28, order=5)
    patch_targets = np.empty(3, dtype=object)
    patch_targets[0] = cl.array.to_device(queue, np.ascontiguousarray(patch.x))
    patch_targets[1] = cl.array.to_device(queue, np.ascontiguousarray(patch.y))
    patch_targets[2] = cl.array.to_device(queue, np.ascontiguousarray(patch.z))

    u_patch = interpolate_volume_potential(
        patch_targets,
        traversal,
        wrangler,
        fmm_potentials,
    ).get(queue)

    rho_patch = _source_profile(patch.x, patch.y, patch.z)
    residual = (
        -patch.laplace(u_patch) - (wave_number * wave_number) * u_patch - rho_patch
    )
    rel_pde_residual = np.linalg.norm(residual) / np.linalg.norm(rho_patch)
    t_pde_check = time.perf_counter() - t_start

    return {
        "q_order": int(q_order),
        "nlevels": int(nlevels),
        "n_points": int(source_vals_host.size),
        "rel_pde_residual": float(rel_pde_residual),
        "fmm_seconds": float(t_fmm),
        "pde_check_seconds": float(t_pde_check),
    }


def run_convergence_study(smoke_mode=None):
    if smoke_mode is None:
        smoke_mode = _is_smoke_mode()

    q_orders = [2, 3] if smoke_mode else [1, 2, 3]
    nlevels = _int_env("VOLUMENTIAL_HELMHOLTZ3D_N_LEVELS", 3)
    fmm_order = _int_env("VOLUMENTIAL_HELMHOLTZ3D_M_ORDER", 10 if smoke_mode else 14)
    wave_number = _float_env("VOLUMENTIAL_HELMHOLTZ3D_K", 6.0)
    table_filename = (
        "nft_laplace3d_for_helmholtz_smoke.sqlite"
        if smoke_mode
        else "nft_laplace3d_for_helmholtz.sqlite"
    )

    logger.info(
        "Helmholtz3D study: smoke=%s, q_orders=%s, nlevels=%d, fmm_order=%d, k=%.3f",
        smoke_mode,
        q_orders,
        nlevels,
        fmm_order,
        wave_number,
    )
    logger.info("Using table cache: %s", table_filename)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    results = []
    for q_order in q_orders:
        table = _get_laplace_table(
            queue,
            table_filename,
            q_order,
        )
        case = _run_case(
            ctx,
            queue,
            table,
            q_order=q_order,
            nlevels=nlevels,
            fmm_order=fmm_order,
            wave_number=wave_number,
        )
        logger.info(
            "q=%d n=%d rel_pde_res=%.3e fmm=%.2fs pde_check=%.2fs",
            case["q_order"],
            case["n_points"],
            case["rel_pde_residual"],
            case["fmm_seconds"],
            case["pde_check_seconds"],
        )
        results.append(case)

    if smoke_mode and len(results) >= 2:
        coarse = results[0]
        fine = results[-1]
        if not coarse["rel_pde_residual"] < 4.5:
            raise RuntimeError(
                "Helmholtz smoke regression failed: q2 PDE residual exceeded band "
                f"(q{coarse['q_order']}={coarse['rel_pde_residual']:.3e})"
            )

        if not fine["rel_pde_residual"] < 8.0e-1:
            raise RuntimeError(
                "Helmholtz smoke regression failed: q3 PDE residual exceeded band "
                f"(q{fine['q_order']}={fine['rel_pde_residual']:.3e})"
            )

        if not fine["rel_pde_residual"] < 0.2 * coarse["rel_pde_residual"]:
            raise RuntimeError(
                "Helmholtz smoke regression failed: q-order improvement too weak "
                f"(q{coarse['q_order']}={coarse['rel_pde_residual']:.3e}, "
                f"q{fine['q_order']}={fine['rel_pde_residual']:.3e})"
            )

    return results


def main():
    results = run_convergence_study(smoke_mode=None)
    print("q_order  n_points  rel_pde_residual  fmm_s  pde_check_s")
    for case in results:
        print(
            f"{case['q_order']:7d}  {case['n_points']:8d}  "
            f"{case['rel_pde_residual']:.6e}  "
            f"{case['fmm_seconds']:.2f}  {case['pde_check_seconds']:.2f}"
        )


if __name__ == "__main__":
    main()
