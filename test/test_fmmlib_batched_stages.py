__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

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

"""Exact-agreement tests for the batched far-field stages of
FPNDFMMLibExpansionWrangler (batched P2M via ``*formmp_imany`` and
GEMM-based L2P) against the inherited per-box implementations from
:mod:`boxtree.pyfmmlib_integration`.

These stages do not touch near-field tables, so the wrangler is built with a
minimal stand-in table object (an established pattern in this test suite).
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import pyopencl as cl
import pyopencl.array  # noqa: F401

from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler

import volumential.meshgen as mg


logger = logging.getLogger(__name__)


# {{{ setup helpers

def _make_pocl_context():
    """Build a context on the pocl CPU platform (never the Intel GPU)."""
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    for platform in platforms:
        if "Portable Computing Language" in platform.name:
            devices = platform.get_devices()
            if devices:
                return cl.Context(devices=[devices[0]])

    pytest.skip("pocl (Portable Computing Language) platform not available")


def _build_wrangler(ctx, queue, *, kernel_type, q_order, nlevels,
                    fmm_order, graded=False, helmholtz_k=2.0):
    from sumpy.kernel import HelmholtzKernel, LaplaceKernel

    from volumential.expansion_wrangler_fpnd import (
        FPNDFMMLibExpansionWrangler,
        FPNDFMMLibTreeIndependentDataForWrangler,
    )

    dim = 3
    mesh = mg.MeshGen3D(q_order, nlevels, -0.5, 0.5, queue=queue)

    if graded:
        # Refine the cells nearest to a corner to obtain a graded
        # (adaptive) tree.
        centers = mesh.get_cell_centers()
        criteria = -np.linalg.norm(centers - np.array([-0.5] * dim), axis=1)
        mesh.update_mesh(criteria, 0.2, 0.0)

    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        dim,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * dim, dtype=np.float64),
    )

    if kernel_type == "laplace":
        knl = LaplaceKernel(dim)
        kernel_extra_kwargs = {}
    elif kernel_type == "helmholtz":
        knl = HelmholtzKernel(dim)
        kernel_extra_kwargs = {knl.helmholtz_k_name: helmholtz_k}
    else:
        raise ValueError(kernel_type)

    tree_indep = FPNDFMMLibTreeIndependentDataForWrangler(
        ctx, None, None, [knl], exclude_self=True
    )

    # The far-field stages under test do not use near-field tables; a
    # minimal stand-in suffices for wrangler construction.
    host_tree_root_extent = 1.0
    fake_table = SimpleNamespace(
        source_box_extent=host_tree_root_extent,
        quad_order=q_order,
        is_built=True,
    )

    wrangler = FPNDFMMLibExpansionWrangler(
        tree_indep,
        queue,
        tree,
        [fake_table],
        np.complex128,
        lambda kernel, kernel_args, tree_, lev: fmm_order,
        q_order,
        kernel_extra_kwargs=kernel_extra_kwargs,
        traversal=traversal,
    )

    # smooth source density on the quadrature nodes (in tree order)
    coords = np.array([coords_i.get(queue) for coords_i in q_points])
    density = np.exp(-16 * np.sum(coords**2, axis=0)) * (
        1 + coords[0] - 2 * coords[1] * coords[2]
    )
    weights = density * q_weights.get(queue)
    weights = wrangler.reorder_sources(weights)

    return wrangler, weights


def _synthesize_local_expansions(wrangler, seed=17):
    """Random local expansions with the correct per-level shapes."""
    rng = np.random.default_rng(seed)
    local_exps = wrangler.local_expansion_zeros()
    local_exps[:] = (
        rng.standard_normal(local_exps.shape)
        + 1j * rng.standard_normal(local_exps.shape)
    )
    return local_exps

# }}}


@pytest.mark.parametrize("kernel_type", ["laplace", "helmholtz"])
@pytest.mark.parametrize("graded", [False, True])
def test_batched_form_multipoles_agrees_with_boxtree(kernel_type, graded):
    ctx = _make_pocl_context()
    queue = cl.CommandQueue(ctx)

    wrangler, weights = _build_wrangler(
        ctx, queue, kernel_type=kernel_type,
        q_order=4, nlevels=3, fmm_order=8, graded=graded,
    )

    # the batched path must actually be available for this configuration
    assert wrangler._get_batched_formmp_routine() is not None

    trav = wrangler.traversal
    args = (trav.level_start_source_box_nrs, trav.source_boxes, [weights])

    mpoles_new, _ = wrangler.form_multipoles(*args)
    mpoles_ref = FMMLibExpansionWrangler.form_multipoles(
        wrangler, wrangler._fmmlib_actx, *args
    )

    ref_scale = np.abs(mpoles_ref).max()
    assert ref_scale > 0
    rel_err = np.abs(mpoles_new - mpoles_ref).max() / ref_scale
    logger.info(
        "form_multipoles (%s, graded=%s): max rel diff %.3e",
        kernel_type, graded, rel_err,
    )
    assert rel_err <= 1e-14


def test_form_multipoles_fallback_path(monkeypatch):
    """The fallback to the inherited implementation must stay exercised."""
    ctx = _make_pocl_context()
    queue = cl.CommandQueue(ctx)

    wrangler, weights = _build_wrangler(
        ctx, queue, kernel_type="laplace",
        q_order=3, nlevels=2, fmm_order=6,
    )

    trav = wrangler.traversal
    args = (trav.level_start_source_box_nrs, trav.source_boxes, [weights])

    mpoles_batched, _ = wrangler.form_multipoles(*args)

    monkeypatch.setattr(
        type(wrangler), "_get_batched_formmp_routine", lambda self: None
    )
    mpoles_fallback, _ = wrangler.form_multipoles(*args)

    ref_scale = np.abs(mpoles_fallback).max()
    assert ref_scale > 0
    assert np.abs(mpoles_batched - mpoles_fallback).max() / ref_scale <= 1e-14


@pytest.mark.parametrize("kernel_type", ["laplace", "helmholtz"])
@pytest.mark.parametrize("graded", [False, True])
def test_gemm_eval_locals_agrees_with_boxtree(kernel_type, graded):
    ctx = _make_pocl_context()
    queue = cl.CommandQueue(ctx)

    wrangler, _weights = _build_wrangler(
        ctx, queue, kernel_type=kernel_type,
        q_order=4, nlevels=3, fmm_order=8, graded=graded,
    )

    assert wrangler._gemm_l2p_supported()

    local_exps = _synthesize_local_expansions(wrangler)

    trav = wrangler.traversal
    args = (trav.level_start_target_box_nrs, trav.target_boxes, local_exps)

    pot_new, _ = wrangler.eval_locals(*args)
    pot_ref = FMMLibExpansionWrangler.eval_locals(
        wrangler, wrangler._fmmlib_actx, *args
    )

    ref_norm = np.linalg.norm(pot_ref)
    assert ref_norm > 0
    rel_err = np.linalg.norm(pot_new - pot_ref) / ref_norm
    logger.info(
        "eval_locals (%s, graded=%s): rel l2 diff %.3e",
        kernel_type, graded, rel_err,
    )
    assert rel_err <= 1e-12


def test_eval_locals_fallback_path(monkeypatch):
    """When the GEMM path declares itself unsupported, the inherited
    implementation is used and produces the same potentials."""
    ctx = _make_pocl_context()
    queue = cl.CommandQueue(ctx)

    wrangler, _weights = _build_wrangler(
        ctx, queue, kernel_type="laplace",
        q_order=3, nlevels=2, fmm_order=6,
    )

    local_exps = _synthesize_local_expansions(wrangler)
    trav = wrangler.traversal
    args = (trav.level_start_target_box_nrs, trav.target_boxes, local_exps)

    pot_gemm, _ = wrangler.eval_locals(*args)

    monkeypatch.setattr(
        type(wrangler), "_gemm_l2p_supported", lambda self: False
    )
    pot_fallback, _ = wrangler.eval_locals(*args)

    ref_norm = np.linalg.norm(pot_fallback)
    assert ref_norm > 0
    assert np.linalg.norm(pot_gemm - pot_fallback) / ref_norm <= 1e-12


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__, "-v"])
