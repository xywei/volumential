__copyright__ = "Copyright (C) 2026"

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

from functools import partial
from itertools import product

import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: F401
import pytest


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
    if tgt_points.shape[1] != dim:
        raise ValueError("source/target dimensions do not match")

    mode_list = [
        mode
        for mode in product(range(-int(k_max), int(k_max) + 1), repeat=dim)
        if any(comp != 0 for comp in mode)
    ]
    modes = np.asarray(mode_list, dtype=np.float64)
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


def _build_wrangler(ctx, queue, traversal, near_field_table, fmm_order=8):
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


def _periodic_test_case(ctx_factory, dim, rng_seed):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    rng = np.random.default_rng(int(rng_seed))

    nsources = 64 if dim == 2 else 72
    ntargets = 56 if dim == 2 else 64

    src_points = rng.uniform(0.15, 0.85, size=(nsources, dim))
    tgt_points = rng.uniform(0.10, 0.90, size=(ntargets, dim))

    strengths = rng.standard_normal(nsources)
    strengths -= np.mean(strengths)

    tree, traversal = _build_split_tree_and_traversal(queue, src_points, tgt_points)

    return {
        "ctx": ctx,
        "queue": queue,
        "tree": tree,
        "traversal": traversal,
        "src_points": src_points,
        "tgt_points": tgt_points,
        "strengths": strengths,
    }


def test_periodic_far_operator_cache_reuse(ctx_factory, tmp_path):
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import build_periodic_far_operator

    case = _periodic_test_case(ctx_factory, dim=2, rng_seed=11)
    queue = case["queue"]
    traversal = case["traversal"]

    cache_file = tmp_path / "nft-periodic-cache-reuse.sqlite"
    with NearFieldInteractionTableManager(
        str(cache_file),
        root_extent=float(traversal.tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(2, "Laplace", q_order=1, queue=queue)
        wrangler = _build_wrangler(case["ctx"], queue, traversal, table, fmm_order=6)

        operator_a, info_a = build_periodic_far_operator(
            traversal,
            wrangler,
            periodic_cell_size=np.full(2, float(traversal.tree.root_extent)),
            periodic_near_shifts="nearest",
            far_shift_radius=2,
            n_training_samples=24,
            rng_seed=17,
            max_check_points=80,
            queue=queue,
            table_manager=tm,
            force_recompute=True,
        )
        operator_b, info_b = build_periodic_far_operator(
            traversal,
            wrangler,
            periodic_cell_size=np.full(2, float(traversal.tree.root_extent)),
            periodic_near_shifts="nearest",
            far_shift_radius=2,
            n_training_samples=24,
            rng_seed=17,
            max_check_points=80,
            queue=queue,
            table_manager=tm,
            force_recompute=False,
        )

    assert info_a["is_recomputed"] is True
    assert info_b["is_recomputed"] is False
    assert np.allclose(operator_a, operator_b)


@pytest.mark.full_accuracy
@pytest.mark.parametrize(
    ("dim", "k_max", "far_shift_radius", "tol_rel"),
    [
        (2, 24, 3, 3.0e-1),
        (3, 10, 2, 3.5e-1),
    ],
)
def test_periodic_laplace_matches_spectral_reference(
    ctx_factory,
    tmp_path,
    dim,
    k_max,
    far_shift_radius,
    tol_rel,
):
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import drive_volume_fmm

    case = _periodic_test_case(ctx_factory, dim=dim, rng_seed=37 + dim)
    queue = case["queue"]
    traversal = case["traversal"]
    strengths = case["strengths"]

    cache_file = tmp_path / f"nft-periodic-{dim}d.sqlite"
    cell_size = np.full(dim, float(traversal.tree.root_extent), dtype=np.float64)

    with NearFieldInteractionTableManager(
        str(cache_file),
        root_extent=float(traversal.tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(dim, "Laplace", q_order=1, queue=queue)
        wrangler = _build_wrangler(case["ctx"], queue, traversal, table, fmm_order=7)

        strengths_dev = cl.array.to_device(queue, np.ascontiguousarray(strengths))

        (pot_periodic,) = drive_volume_fmm(
            traversal,
            wrangler,
            strengths_dev,
            strengths_dev,
            direct_evaluation=True,
            auto_interpolate_targets=False,
            periodic_near_shifts="nearest",
            periodic_cell_size=cell_size,
            periodic_far_operator="auto",
            periodic_far_operator_manager=tm,
            periodic_far_shift_radius=far_shift_radius,
            periodic_far_training_samples=32 if dim == 2 else 48,
            periodic_far_rng_seed=23,
            periodic_far_max_check_points=96,
            periodic_far_force_recompute=True,
        )

    reference = _spectral_periodic_laplace_reference(
        src_points=case["src_points"],
        strengths=strengths,
        tgt_points=case["tgt_points"],
        cell_size=cell_size,
        k_max=k_max,
    )

    error = pot_periodic.get(queue) - reference
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(reference)
    assert rel_l2 < tol_rel
