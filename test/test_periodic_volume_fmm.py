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


_PARITY_REFERENCE_CASE_MATRIX = (
    {
        "id": "SMOKE-RND-2D-basic",
        "tier": "smoke",
        "dim": 2,
        "scenario": "random_independent",
        "rng_seed": 101,
        "nsources": 40,
        "ntargets": 36,
        "k_max": 24,
        "far_shift_radius": 3,
        "cell_scale": (1.0, 1.0),
        "tol_rel": 1.0e-10,
    },
    {
        "id": "SMOKE-RND-3D-basic",
        "tier": "smoke",
        "dim": 3,
        "scenario": "random_independent",
        "rng_seed": 102,
        "nsources": 48,
        "ntargets": 80,
        "k_max": 10,
        "far_shift_radius": 2,
        "cell_scale": (1.0, 1.0, 1.0),
        "tol_rel": 1.0e-10,
    },
    {
        "id": "FULL-RND-2D-basic",
        "tier": "full_accuracy",
        "dim": 2,
        "scenario": "random_independent",
        "rng_seed": 39,
        "nsources": 64,
        "ntargets": 56,
        "k_max": 24,
        "far_shift_radius": 3,
        "cell_scale": (1.0, 1.0),
        "tol_rel": 1.0e-10,
    },
    {
        "id": "FULL-RND-3D-basic",
        "tier": "full_accuracy",
        "dim": 3,
        "scenario": "random_independent",
        "rng_seed": 40,
        "nsources": 72,
        "ntargets": 64,
        "k_max": 10,
        "far_shift_radius": 2,
        "cell_scale": (1.0, 1.0, 1.0),
        "tol_rel": 1.0e-10,
    },
    {
        "id": "FULL-WRAP-CLOSE-PAIR-2D",
        "tier": "full_accuracy",
        "dim": 2,
        "scenario": "wrap_close_pair",
        "rng_seed": 141,
        "nsources": 64,
        "ntargets": 56,
        "k_max": 24,
        "far_shift_radius": 3,
        "cell_scale": (1.0, 1.0),
        "tol_rel": 1.0e-10,
    },
    {
        "id": "FULL-WRAP-CLOSE-PAIR-3D",
        "tier": "full_accuracy",
        "dim": 3,
        "scenario": "wrap_close_pair",
        "rng_seed": 142,
        "nsources": 72,
        "ntargets": 64,
        "k_max": 10,
        "far_shift_radius": 2,
        "cell_scale": (1.0, 1.0, 1.0),
        "tol_rel": 1.0e-10,
    },
    {
        "id": "FULL-ANISO-CELL-2D",
        "tier": "full_accuracy",
        "dim": 2,
        "scenario": "random_independent",
        "rng_seed": 251,
        "nsources": 64,
        "ntargets": 56,
        "k_max": 24,
        "far_shift_radius": 3,
        "cell_scale": (1.35, 0.85),
        "tol_rel": 1.0e-10,
    },
    {
        "id": "FULL-ANISO-CELL-3D",
        "tier": "full_accuracy",
        "dim": 3,
        "scenario": "random_independent",
        "rng_seed": 252,
        "nsources": 72,
        "ntargets": 64,
        "k_max": 10,
        "far_shift_radius": 2,
        "cell_scale": (1.30, 0.90, 1.55),
        "tol_rel": 1.0e-10,
    },
)


_PARITY_INVARIANCE_CASE_MATRIX = (
    {
        "id": "LONG-TRANSLATION-INVARIANCE-2D",
        "tier": "longrun",
        "kind": "translation",
        "dim": 2,
        "rng_seed": 311,
        "nsources": 96,
        "ntargets": 96,
        "k_max": 28,
        "far_shift_radius": 3,
        "cell_scale": (1.0, 1.0),
        "shift_fraction": (0.37, -0.19),
        "tol_rel": 1.0e-12,
    },
    {
        "id": "LONG-TRANSLATION-INVARIANCE-3D",
        "tier": "longrun",
        "kind": "translation",
        "dim": 3,
        "rng_seed": 312,
        "nsources": 104,
        "ntargets": 96,
        "k_max": 12,
        "far_shift_radius": 2,
        "cell_scale": (1.0, 1.0, 1.0),
        "shift_fraction": (0.31, -0.23, 0.17),
        "tol_rel": 1.0e-12,
    },
    {
        "id": "LONG-PERMUTATION-INVARIANCE-2D",
        "tier": "longrun",
        "kind": "permutation",
        "dim": 2,
        "rng_seed": 321,
        "nsources": 112,
        "ntargets": 96,
        "k_max": 28,
        "far_shift_radius": 3,
        "cell_scale": (1.0, 1.0),
        "tol_rel": 1.0e-12,
    },
    {
        "id": "LONG-PERMUTATION-INVARIANCE-3D",
        "tier": "longrun",
        "kind": "permutation",
        "dim": 3,
        "rng_seed": 322,
        "nsources": 120,
        "ntargets": 96,
        "k_max": 12,
        "far_shift_radius": 2,
        "cell_scale": (1.0, 1.0, 1.0),
        "tol_rel": 1.0e-12,
    },
)


_PARITY_FMM_GAP_CASE_MATRIX = (
    {
        "id": "FULL-FMM-GAP-2D-q1-l4",
        "tier": "full_accuracy",
        "dim": 2,
        "q_order": 1,
        "nlevels": 4,
        "rng_seed": 1302,
        "k_max": 24,
        "far_shift_radius": 3,
        "fmm_order": 6,
        "target_tol_rel": 1.0e-10,
        "sanity_upper_rel": 1.0,
    },
    {
        "id": "FULL-FMM-GAP-3D-q1-l3",
        "tier": "full_accuracy",
        "dim": 3,
        "q_order": 1,
        "nlevels": 3,
        "rng_seed": 1303,
        "k_max": 10,
        "far_shift_radius": 2,
        "fmm_order": 5,
        "target_tol_rel": 1.0e-10,
        "sanity_upper_rel": 1.0,
    },
)


def _case_params_by_tier(case_matrix, tier):
    return [
        pytest.param(case_spec, id=case_spec["id"])
        for case_spec in case_matrix
        if case_spec["tier"] == tier
    ]


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


def _build_point_configuration(*, dim, scenario, nsources, ntargets, rng):
    if scenario == "random_independent":
        src_points = rng.uniform(0.15, 0.85, size=(nsources, dim))
        tgt_points = rng.uniform(0.10, 0.90, size=(ntargets, dim))
        return src_points, tgt_points

    if scenario == "wrap_close_pair":
        src_points = rng.uniform(0.20, 0.80, size=(nsources, dim))
        tgt_points = rng.uniform(0.20, 0.80, size=(ntargets, dim))

        eps = 1.0e-10
        n_pairs = min(2 * dim, nsources, ntargets)
        for ipair in range(n_pairs):
            axis = ipair % dim
            if ipair % 2 == 0:
                src_points[ipair, axis] = eps
                tgt_points[ipair, axis] = 1.0 - 0.5 * eps
            else:
                src_points[ipair, axis] = 1.0 - eps
                tgt_points[ipair, axis] = 0.5 * eps

        return src_points, tgt_points

    raise ValueError(f"unsupported periodic parity scenario '{scenario}'")


def _case_cell_size(case_spec, traversal):
    dim = int(case_spec["dim"])
    cell_scale = np.asarray(case_spec.get("cell_scale", (1.0,) * dim), dtype=np.float64)
    if cell_scale.shape != (dim,):
        raise ValueError(f"cell_scale for {case_spec['id']} must have shape ({dim},)")

    root_extent = float(traversal.tree.root_extent)
    return np.ascontiguousarray(root_extent * cell_scale, dtype=np.float64)


def _evaluate_periodic_direct_solution(
    *,
    case,
    case_cache_id,
    tmp_path,
    cell_size,
    k_max,
    far_shift_radius,
    fmm_order,
    training_samples,
    max_check_points,
):
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import drive_volume_fmm

    queue = case["queue"]
    traversal = case["traversal"]
    strengths = case["strengths"]

    cache_file = tmp_path / f"nft-periodic-{case_cache_id}.sqlite"
    with NearFieldInteractionTableManager(
        str(cache_file),
        root_extent=float(traversal.tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(
            int(traversal.tree.dimensions), "Laplace", q_order=1, queue=queue
        )
        wrangler = _build_wrangler(
            case["ctx"],
            queue,
            traversal,
            table,
            fmm_order=int(fmm_order),
        )

        strengths_dev = cl.array.to_device(queue, np.ascontiguousarray(strengths))

        (pot_periodic,) = drive_volume_fmm(
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
            periodic_far_shift_radius=int(far_shift_radius),
            periodic_far_spectral_kmax_2d=int(k_max)
            if int(traversal.tree.dimensions) == 2
            else None,
            periodic_far_spectral_kmax_3d=int(k_max)
            if int(traversal.tree.dimensions) == 3
            else None,
            periodic_far_training_samples=int(training_samples),
            periodic_far_rng_seed=23,
            periodic_far_max_check_points=int(max_check_points),
            periodic_far_force_recompute=True,
        )

    return np.asarray(pot_periodic.get(queue), dtype=np.float64)


def _periodic_mesh_fmm_case(ctx_factory, case_spec):
    import volumential.meshgen as mg

    dim = int(case_spec["dim"])
    q_order = int(case_spec["q_order"])
    nlevels = int(case_spec["nlevels"])

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if dim == 2:
        mesh = mg.MeshGen2D(q_order, nlevels, -0.5, 0.5, queue=queue)
    elif dim == 3:
        mesh = mg.MeshGen3D(q_order, nlevels, -0.5, 0.5, queue=queue)
    else:
        raise ValueError(f"unsupported dimension {dim}")

    q_points, source_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        dim,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * dim, dtype=np.float64),
    )

    src_points = np.ascontiguousarray(
        np.array([coords.get(queue) for coords in q_points], dtype=np.float64).T
    )

    nsources = int(src_points.shape[0])
    rng = np.random.default_rng(int(case_spec["rng_seed"]))
    weighted_strengths = rng.standard_normal(nsources)
    weighted_strengths -= np.mean(weighted_strengths)

    source_weights_host = np.asarray(source_weights.get(queue), dtype=np.float64)
    source_vals_host = weighted_strengths / source_weights_host

    source_vals_dev = cl.array.to_device(
        queue,
        np.ascontiguousarray(source_vals_host, dtype=np.float64),
    )
    weighted_sources_dev = source_vals_dev * source_weights

    return {
        "ctx": ctx,
        "queue": queue,
        "tree": tree,
        "traversal": traversal,
        "src_points": src_points,
        "weighted_strengths": np.asarray(weighted_strengths, dtype=np.float64),
        "source_vals": source_vals_dev,
        "weighted_sources": weighted_sources_dev,
        "q_order": q_order,
    }


def _run_periodic_fmm_gap_case(ctx_factory, tmp_path, case_spec):
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import drive_volume_fmm

    case = _periodic_mesh_fmm_case(ctx_factory, case_spec)
    queue = case["queue"]
    traversal = case["traversal"]
    dim = int(case_spec["dim"])
    k_max = int(case_spec["k_max"])

    cell_size = np.full(dim, float(traversal.tree.root_extent), dtype=np.float64)

    cache_file = tmp_path / f"nft-periodic-fmm-gap-{case_spec['id']}.sqlite"
    with NearFieldInteractionTableManager(
        str(cache_file),
        root_extent=float(traversal.tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(
            dim, "Laplace", q_order=int(case_spec["q_order"]), queue=queue
        )
        wrangler = _build_wrangler(
            case["ctx"],
            queue,
            traversal,
            table,
            fmm_order=int(case_spec["fmm_order"]),
            quad_order=int(case_spec["q_order"]),
            exclude_self=True,
        )

        (pot_periodic_fmm,) = drive_volume_fmm(
            traversal,
            wrangler,
            case["weighted_sources"],
            case["source_vals"],
            periodic=True,
            direct_evaluation=False,
            auto_interpolate_targets=False,
            periodic_near_shifts="nearest",
            periodic_cell_size=cell_size,
            periodic_far_operator="auto",
            periodic_far_operator_manager=tm,
            periodic_far_shift_radius=int(case_spec["far_shift_radius"]),
            periodic_far_spectral_kmax_2d=k_max if dim == 2 else None,
            periodic_far_spectral_kmax_3d=k_max if dim == 3 else None,
            periodic_far_training_samples=64,
            periodic_far_rng_seed=23,
            periodic_far_max_check_points=256,
            periodic_far_force_recompute=True,
        )

    reference = _spectral_periodic_laplace_reference(
        src_points=case["src_points"],
        strengths=case["weighted_strengths"],
        tgt_points=case["src_points"],
        cell_size=cell_size,
        k_max=k_max,
    )

    error = np.asarray(pot_periodic_fmm.get(queue), dtype=np.float64) - reference
    rel_l2 = np.linalg.norm(error) / max(np.linalg.norm(reference), 1.0e-30)
    return rel_l2


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


def _build_wrangler(
    ctx,
    queue,
    traversal,
    near_field_table,
    fmm_order=8,
    quad_order=1,
    exclude_self=False,
):
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
        exclude_self=bool(exclude_self),
    )

    self_extra_kwargs = {}
    if bool(exclude_self) and bool(traversal.tree.sources_are_targets):
        self_extra_kwargs = {
            "target_to_source": np.arange(int(traversal.tree.ntargets), dtype=np.int32)
        }

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=near_field_table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: int(fmm_order),
        quad_order=int(quad_order),
        self_extra_kwargs=self_extra_kwargs,
    )

    return wrangler


def _periodic_test_case(
    ctx_factory,
    *,
    dim,
    rng_seed,
    nsources=None,
    ntargets=None,
    scenario="random_independent",
    src_points=None,
    tgt_points=None,
    strengths=None,
):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    rng = np.random.default_rng(int(rng_seed))

    if nsources is None:
        nsources = 64 if dim == 2 else 72
    if ntargets is None:
        ntargets = 56 if dim == 2 else 64

    nsources = int(nsources)
    ntargets = int(ntargets)

    if src_points is None or tgt_points is None:
        src_points, tgt_points = _build_point_configuration(
            dim=dim,
            scenario=scenario,
            nsources=nsources,
            ntargets=ntargets,
            rng=rng,
        )
    else:
        src_points = np.asarray(src_points, dtype=np.float64)
        tgt_points = np.asarray(tgt_points, dtype=np.float64)
        if src_points.ndim != 2 or src_points.shape[1] != dim:
            raise ValueError("src_points must have shape (nsources, dim)")
        if tgt_points.ndim != 2 or tgt_points.shape[1] != dim:
            raise ValueError("tgt_points must have shape (ntargets, dim)")
        nsources = int(src_points.shape[0])
        ntargets = int(tgt_points.shape[0])

    if strengths is None:
        strengths = rng.standard_normal(nsources)
        strengths -= np.mean(strengths)
    else:
        strengths = np.asarray(strengths, dtype=np.float64)
        if strengths.shape != (nsources,):
            raise ValueError("strengths must have shape (nsources,)")

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
    assert info_a.get("tail_model") == "spectral_2d"
    assert info_b.get("tail_model") == "spectral_2d"
    assert info_a.get("operator_basis") in {"multipole", "source"}
    assert info_b.get("operator_basis") == info_a.get("operator_basis")
    if info_a.get("operator_basis") == "source":
        assert operator_a.shape[1] == int(traversal.tree.nsources)
    assert np.allclose(operator_a, operator_b)


def test_periodic_laplace_auto_rejects_multipole_basis(ctx_factory, tmp_path):
    from volumential.table_manager import NearFieldInteractionTableManager
    from volumential.volume_fmm import drive_volume_fmm

    case = _periodic_test_case(ctx_factory, dim=2, rng_seed=19)
    queue = case["queue"]
    traversal = case["traversal"]
    strengths = case["strengths"]

    cache_file = tmp_path / "nft-periodic-auto-basis.sqlite"
    with NearFieldInteractionTableManager(
        str(cache_file),
        root_extent=float(traversal.tree.root_extent),
        progress_bar=False,
    ) as tm:
        table, _ = tm.get_table(2, "Laplace", q_order=1, queue=queue)
        wrangler = _build_wrangler(case["ctx"], queue, traversal, table, fmm_order=6)

        strengths_dev = cl.array.to_device(queue, np.ascontiguousarray(strengths))
        cell_size = np.full(2, float(traversal.tree.root_extent), dtype=np.float64)

        with pytest.raises(
            ValueError,
            match=(
                "periodic_far_operator='auto' for 2D/3D Laplace uses strict "
                "spectral runtime mode"
            ),
        ):
            drive_volume_fmm(
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
                periodic_far_operator_basis="multipole",
                periodic_far_operator_manager=tm,
            )


def _run_reference_parity_case(ctx_factory, tmp_path, case_spec):
    dim = int(case_spec["dim"])
    case = _periodic_test_case(
        ctx_factory,
        dim=dim,
        rng_seed=int(case_spec["rng_seed"]),
        nsources=int(case_spec["nsources"]),
        ntargets=int(case_spec["ntargets"]),
        scenario=case_spec["scenario"],
    )

    cell_size = _case_cell_size(case_spec, case["traversal"])
    k_max = int(case_spec["k_max"])

    potential = _evaluate_periodic_direct_solution(
        case=case,
        case_cache_id=case_spec["id"],
        tmp_path=tmp_path,
        cell_size=cell_size,
        k_max=k_max,
        far_shift_radius=int(case_spec["far_shift_radius"]),
        fmm_order=7,
        training_samples=32 if dim == 2 else 48,
        max_check_points=96,
    )

    reference = _spectral_periodic_laplace_reference(
        src_points=case["src_points"],
        strengths=case["strengths"],
        tgt_points=case["tgt_points"],
        cell_size=cell_size,
        k_max=k_max,
    )

    error = potential - reference
    denom = max(np.linalg.norm(reference), 1.0e-30)
    rel_l2 = np.linalg.norm(error) / denom
    return rel_l2, float(case_spec["tol_rel"])


@pytest.mark.parametrize(
    "case_spec",
    _case_params_by_tier(_PARITY_REFERENCE_CASE_MATRIX, "smoke"),
)
def test_periodic_laplace_reference_parity_smoke(ctx_factory, tmp_path, case_spec):
    rel_l2, tol_rel = _run_reference_parity_case(ctx_factory, tmp_path, case_spec)
    assert rel_l2 < tol_rel


@pytest.mark.full_accuracy
@pytest.mark.parametrize(
    "case_spec",
    _case_params_by_tier(_PARITY_REFERENCE_CASE_MATRIX, "full_accuracy"),
)
def test_periodic_laplace_reference_parity_full_accuracy(
    ctx_factory,
    tmp_path,
    case_spec,
):
    rel_l2, tol_rel = _run_reference_parity_case(ctx_factory, tmp_path, case_spec)
    assert rel_l2 < tol_rel


@pytest.mark.parametrize(
    "case_spec",
    _case_params_by_tier(_PARITY_INVARIANCE_CASE_MATRIX, "longrun"),
)
def test_periodic_laplace_longrun_invariance(longrun, ctx_factory, tmp_path, case_spec):
    del longrun

    dim = int(case_spec["dim"])
    rng_seed = int(case_spec["rng_seed"])
    base_case = _periodic_test_case(
        ctx_factory,
        dim=dim,
        rng_seed=rng_seed,
        nsources=int(case_spec["nsources"]),
        ntargets=int(case_spec["ntargets"]),
        scenario="random_independent",
    )

    cell_size = _case_cell_size(case_spec, base_case["traversal"])
    k_max = int(case_spec["k_max"])

    base_potential = _evaluate_periodic_direct_solution(
        case=base_case,
        case_cache_id=f"{case_spec['id']}-base",
        tmp_path=tmp_path,
        cell_size=cell_size,
        k_max=k_max,
        far_shift_radius=int(case_spec["far_shift_radius"]),
        fmm_order=8,
        training_samples=48 if dim == 2 else 64,
        max_check_points=128,
    )

    kind = case_spec["kind"]
    if kind == "translation":
        shift_fraction = np.asarray(case_spec["shift_fraction"], dtype=np.float64)
        shift_vector = shift_fraction * cell_size

        shifted_case = _periodic_test_case(
            ctx_factory,
            dim=dim,
            rng_seed=rng_seed,
            src_points=base_case["src_points"] + shift_vector.reshape(1, -1),
            tgt_points=base_case["tgt_points"] + shift_vector.reshape(1, -1),
            strengths=base_case["strengths"],
        )

        shifted_potential = _evaluate_periodic_direct_solution(
            case=shifted_case,
            case_cache_id=f"{case_spec['id']}-shifted",
            tmp_path=tmp_path,
            cell_size=cell_size,
            k_max=k_max,
            far_shift_radius=int(case_spec["far_shift_radius"]),
            fmm_order=8,
            training_samples=48 if dim == 2 else 64,
            max_check_points=128,
        )

        denom = max(np.linalg.norm(base_potential), 1.0e-30)
        rel_l2 = np.linalg.norm(base_potential - shifted_potential) / denom
        assert rel_l2 < float(case_spec["tol_rel"])
        return

    if kind == "permutation":
        rng = np.random.default_rng(rng_seed + 1000)
        perm = rng.permutation(base_case["src_points"].shape[0])

        permuted_case = _periodic_test_case(
            ctx_factory,
            dim=dim,
            rng_seed=rng_seed,
            src_points=base_case["src_points"][perm],
            tgt_points=base_case["tgt_points"],
            strengths=base_case["strengths"][perm],
        )

        permuted_potential = _evaluate_periodic_direct_solution(
            case=permuted_case,
            case_cache_id=f"{case_spec['id']}-permuted",
            tmp_path=tmp_path,
            cell_size=cell_size,
            k_max=k_max,
            far_shift_radius=int(case_spec["far_shift_radius"]),
            fmm_order=8,
            training_samples=48 if dim == 2 else 64,
            max_check_points=128,
        )

        denom = max(np.linalg.norm(base_potential), 1.0e-30)
        rel_l2 = np.linalg.norm(base_potential - permuted_potential) / denom
        assert rel_l2 < float(case_spec["tol_rel"])
        return

    raise ValueError(f"unsupported longrun parity kind '{kind}'")


@pytest.mark.full_accuracy
@pytest.mark.parametrize(
    "case_spec",
    _case_params_by_tier(_PARITY_FMM_GAP_CASE_MATRIX, "full_accuracy"),
)
def test_periodic_laplace_fmm_path_gap_vs_spectral_reference(
    ctx_factory,
    tmp_path,
    case_spec,
):
    rel_l2 = _run_periodic_fmm_gap_case(ctx_factory, tmp_path, case_spec)

    assert np.isfinite(rel_l2)
    assert rel_l2 < float(case_spec["sanity_upper_rel"])

    target_tol = float(case_spec["target_tol_rel"])
    assert rel_l2 < target_tol, (
        "periodic FMM path (direct_evaluation=False) missed target tolerance: "
        f"rel_l2={rel_l2:.3e}, target={target_tol:.1e}"
    )
