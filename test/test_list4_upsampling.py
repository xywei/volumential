"""Upsampled P2L for List 4 sources.

A List 4 source box is twice the size of the target box and only half its own
size away from it, too close for point quadrature at its nodes. The wranglers
interpolate its density to a finer Gauss rule and form the local expansion from
those nodes (:mod:`volumential.wranglers.list4_upsampling`). These tests check
the rule on its own, the placement of the finer nodes, and the List 4 far field
of both wranglers against the exact integral of each source box's interpolant.
"""

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

import logging
from functools import partial
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import pyopencl as cl
import pyopencl.array
from pytools.obj_array import new_1d as obj_array_1d

import volumential.meshgen as mg
from volumential.wranglers.list4_upsampling import (
    LIST4_UPSAMPLING,
    build_list4_upsampled_sources,
    list4_upsampled_q_order,
    list4_upsampling_matrix,
    normalize_list4_upsampling,
)


try:
    from _opencl_test_utils import create_fp64_context_or_skip
except ImportError:
    from test._opencl_test_utils import create_fp64_context_or_skip


logger = logging.getLogger(__name__)


# {{{ the rule


@pytest.mark.parametrize(
    ("q_order", "factor", "expected"),
    [(4, 1.5, 6), (3, 1.5, 5), (8, 1.5, 12), (6, 1.25, 8), (2, 1.5, 3),
     (5, 1, 5), (5, 2, 10)],
)
def test_upsampled_q_order(q_order, factor, expected):
    assert list4_upsampled_q_order(q_order, factor) == expected


def test_normalize_list4_upsampling():
    # on by default below 3D, where it is cheap; opt-in in 3D
    assert normalize_list4_upsampling(None, 1) == LIST4_UPSAMPLING
    assert normalize_list4_upsampling(None, 2) == LIST4_UPSAMPLING
    assert normalize_list4_upsampling(None, 3) == 1.0
    assert normalize_list4_upsampling(1, 2) == 1.0
    assert normalize_list4_upsampling(np.float32(2.5), 3) == 2.5
    for bad in (True, "1.5", [1.5]):
        with pytest.raises(TypeError):
            normalize_list4_upsampling(bad, 2)
    for bad in (0.5, 0, -2, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            normalize_list4_upsampling(bad, 2)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("q_order", [1, 3, 4])
def test_matrix_without_upsampling_is_the_identity(dim, q_order):
    nodes, matrix = list4_upsampling_matrix(q_order, q_order, dim)
    assert nodes.shape == (dim, q_order**dim)
    assert np.allclose(matrix, np.eye(q_order**dim), atol=1e-13)


def _tensor_gauss(order, dim):
    x, w = np.polynomial.legendre.leggauss(order)
    nodes = np.array(list(product(x, repeat=dim))).T
    weights = np.prod(np.array(list(product(w, repeat=dim))), axis=1)
    return nodes, weights


def _lagrange_1d(nodes, points):
    values = np.ones((len(points), len(nodes)))
    for j, node_j in enumerate(nodes):
        for k, node_k in enumerate(nodes):
            if k != j:
                values[:, j] *= (points - node_k) / (node_j - node_k)
    return values


def _interpolant(density, q_order, points):
    """The tensor-product interpolant of *density* (tensor order) at points."""
    dim = points.shape[0]
    gauss, _ = np.polynomial.legendre.leggauss(q_order)
    values = np.zeros(points.shape[1])
    for j, index in enumerate(product(range(q_order), repeat=dim)):
        basis = np.ones(points.shape[1])
        for axis in range(dim):
            basis *= _lagrange_1d(gauss, points[axis])[:, index[axis]]
        values += density[j] * basis
    return values


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(("q_order", "fine_q_order"), [(2, 3), (3, 5), (4, 6)])
def test_matrix_integrates_the_interpolant(dim, q_order, fine_q_order):
    """The finer strengths integrate the interpolant times a polynomial of
    degree 2 * fine_q_order - q_order per axis exactly."""
    rng = np.random.default_rng(7)
    _, coarse_weights = _tensor_gauss(q_order, dim)
    density = rng.standard_normal(q_order**dim)
    strengths = density * coarse_weights

    fine_nodes, matrix = list4_upsampling_matrix(q_order, fine_q_order, dim)
    expected_nodes, _ = _tensor_gauss(fine_q_order, dim)
    assert np.allclose(fine_nodes, expected_nodes, atol=1e-15)
    fine_strengths = matrix @ strengths

    degree = 2 * fine_q_order - q_order
    coeffs = rng.standard_normal((dim, degree + 1))

    def test_function(x):
        return np.prod(
            [np.polynomial.polynomial.polyval(x[axis], coeffs[axis])
             for axis in range(dim)],
            axis=0,
        )

    exact_nodes, exact_weights = _tensor_gauss(fine_q_order + q_order, dim)
    exact = np.sum(
        exact_weights
        * _interpolant(density, q_order, exact_nodes)
        * test_function(exact_nodes)
    )
    upsampled = np.sum(fine_strengths * test_function(fine_nodes))
    assert abs(upsampled - exact) <= 1e-12 * max(1.0, abs(exact))
    # Both rules integrate the interpolant itself exactly.
    assert abs(np.sum(fine_strengths) - np.sum(strengths)) <= 1e-13


def _two_box_layout(rng, dim, q_order, *, perturb=0.0, count=None):
    """Two level-1 boxes of a unit root, holding their Gauss nodes shuffled."""
    root_extent = 1.0
    box_centers = np.zeros((dim, 3))
    box_centers[:, 1] = -0.25
    box_centers[:, 2] = 0.25
    box_levels = np.array([0, 1, 1])
    ref, _ = _tensor_gauss(q_order, dim)
    sources, starts, counts = [], [], []
    for ibox in (1, 2):
        pts = box_centers[:, ibox, None] + 0.25 * ref
        pts = pts[:, rng.permutation(pts.shape[1])]
        starts.append(sum(p.shape[1] for p in sources))
        counts.append(pts.shape[1])
        sources.append(pts)
    sources = np.concatenate(sources, axis=1)
    sources[0, -1] += perturb
    counts = np.array([0, *counts], dtype=np.int32)
    if count is not None:
        counts[2] = count
    return {
        "sources": sources,
        "box_centers": box_centers,
        "box_levels": box_levels,
        "root_extent": root_extent,
        "box_source_starts": np.array([0, *starts], dtype=np.int32),
        "box_source_counts_nonchild": counts,
    }


@pytest.mark.parametrize("dim", [2, 3])
def test_upsampled_sources_follow_the_nodes_in_any_order(dim):
    rng = np.random.default_rng(3)
    q_order, fine_q_order = 3, 5
    layout = _two_box_layout(rng, dim, q_order)

    upsampled, reason = build_list4_upsampled_sources(
        lists=np.array([2, 1, 2]), q_order=q_order, fine_q_order=fine_q_order,
        **layout,
    )
    assert reason is None
    assert upsampled.source_boxes.tolist() == [1, 2]

    ref, _ = _tensor_gauss(q_order, dim)
    fine_ref, _ = _tensor_gauss(fine_q_order, dim)
    n_fine = fine_q_order**dim
    for row, ibox in enumerate(upsampled.source_boxes):
        center = layout["box_centers"][:, ibox, None]
        # gather lists the box's sources in tensor order
        assert np.allclose(
            layout["sources"][:, upsampled.gather[row]], center + 0.25 * ref
        )
        start = upsampled.box_source_starts[ibox]
        assert upsampled.box_source_counts_nonchild[ibox] == n_fine
        assert np.allclose(
            upsampled.sources[:, start:start + n_fine], center + 0.25 * fine_ref
        )
    assert upsampled.box_source_counts_nonchild[0] == 0

    # A linear density is reproduced at the finer nodes.
    density = 1 + layout["sources"][0] - 2 * layout["sources"][-1]
    _, weights = _tensor_gauss(q_order, dim)
    _, fine_weights = _tensor_gauss(fine_q_order, dim)
    strengths = np.empty_like(density)
    for row in range(2):
        strengths[upsampled.gather[row]] = (
            density[upsampled.gather[row]] * weights * 0.25**dim
        )
    fine_density = 1 + upsampled.sources[0] - 2 * upsampled.sources[-1]
    assert np.allclose(
        upsampled.upsample(strengths),
        fine_density * np.tile(fine_weights, 2) * 0.25**dim,
        atol=1e-14,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_upsampled_sources_refuse_a_foreign_layout(dim):
    rng = np.random.default_rng(5)
    upsampled, reason = build_list4_upsampled_sources(
        lists=np.array([1, 2]), q_order=3, fine_q_order=5,
        **_two_box_layout(rng, dim, 3, perturb=1e-3),
    )
    assert upsampled is None
    assert "Gauss nodes" in reason

    upsampled, reason = build_list4_upsampled_sources(
        lists=np.array([1, 2]), q_order=3, fine_q_order=5,
        **_two_box_layout(rng, dim, 3, count=3**dim - 1),
    )
    assert upsampled is None
    assert "sources where" in reason


# }}}


# {{{ the List 4 far field


def _split_twice_geometry(ctx, queue, dim, q_order):
    """A uniform tree with one leaf split twice, as in the #175 checks."""
    point = np.array([0.1, 0.07, 0.04][:dim])
    mesh_cls = {2: mg.MeshGen2D, 3: mg.MeshGen3D}[dim]
    mesh = mesh_cls(q_order, 4 if dim == 2 else 3, -0.5, 0.5, queue=queue)
    for _ in range(2):
        half_sides = 0.5 * mesh.get_cell_measures() ** (1 / dim)
        offsets = np.abs(mesh.get_cell_centers() - point)
        (ileaf,) = np.flatnonzero(np.all(offsets < half_sides[:, None], axis=1))
        refine_flags = np.zeros(mesh.boxtree.nboxes, dtype=bool)
        refine_flags[mesh.boxtree.active_boxes.get()[ileaf]] = True
        mesh.boxtree.refine_and_coarsen(
            refine_flags=refine_flags,
            coarsen_flags=np.zeros_like(refine_flags),
        )
    q_points, q_weights, tree, trav = mg.build_geometry_info(
        ctx, queue, dim, q_order, mesh, bbox=np.array([[-0.5, 0.5]] * dim)
    )
    return q_points, q_weights, tree, trav


def _density(x):
    return np.cos(3 * x[0] + 1) * np.exp(x[1]) + x[-1] ** 2


def _green(dim, targets, sources):
    dist = np.sqrt(
        np.sum((targets[:, :, None] - sources[:, None, :]) ** 2, axis=0)
    )
    if dim == 2:
        return -np.log(dist) / (2 * np.pi)
    return 1 / (4 * np.pi * dist)


def _list4_reference(queue, tree, trav, *, q_order, density, weights):
    """The List 4 far field at every node, in tree order, two ways.

    Returns the exact integral of each source box's interpolant (by a Gauss
    rule of order ``3 * q_order``, many orders past what the test resolves)
    and the point sum over the box's own nodes.
    """
    def host(ary):
        return ary.get(queue)

    dim = tree.dimensions
    nodes = np.array([host(tree.sources[i]) for i in range(dim)])
    centers = host(tree.box_centers)
    levels = host(tree.box_levels)
    parents = host(tree.box_parent_ids)
    starts = host(tree.box_source_starts)
    counts = host(tree.box_source_counts_nonchild)
    totpb = host(trav.target_or_target_parent_boxes)
    l4_starts = host(trav.from_sep_bigger_starts)
    l4_lists = host(trav.from_sep_bigger_lists)
    list4 = {
        int(box): l4_lists[l4_starts[i] : l4_starts[i + 1]]
        for i, box in enumerate(totpb)
    }

    fine_ref, fine_weights = _tensor_gauss(3 * q_order, dim)
    gauss, _ = np.polynomial.legendre.leggauss(q_order)

    exact = np.zeros(nodes.shape[1])
    point = np.zeros(nodes.shape[1])
    for leaf in np.flatnonzero(counts[: tree.nboxes]):
        targets = slice(starts[leaf], starts[leaf] + counts[leaf])
        sources_of_leaf = []
        box = int(leaf)
        while True:
            sources_of_leaf.extend(list4.get(box, []))
            if box == 0:
                break
            box = int(parents[box])
        for src in sources_of_leaf:
            src_nodes = slice(starts[src], starts[src] + counts[src])
            half = 0.5 * tree.root_extent / 2 ** levels[src]
            ref = (nodes[:, src_nodes] - centers[:, src, None]) / half
            # tensor position of each node, from its coordinates
            index = np.zeros(ref.shape[1], dtype=np.int64)
            for axis in range(dim):
                nearest = np.argmin(np.abs(ref[axis][:, None] - gauss), axis=1)
                index = index * q_order + nearest
            tensor_density = np.empty(q_order**dim)
            tensor_density[index] = density[src_nodes]
            fine_pts = centers[:, src, None] + half * fine_ref
            fine_strengths = (
                half**dim * fine_weights * _interpolant(tensor_density, q_order,
                                                        fine_ref)
            )
            exact[targets] += _green(dim, nodes[:, targets], fine_pts) @ (
                fine_strengths
            )
            point[targets] += _green(dim, nodes[:, targets],
                                     nodes[:, src_nodes]) @ weights[src_nodes]
    return exact, point


def _list4_far_field(wrangler, trav, weights):
    """Form locals from List 4 only, push them down, evaluate them."""
    local_exps, _ = wrangler.form_locals(
        trav.level_start_target_or_target_parent_box_nrs,
        trav.target_or_target_parent_boxes,
        trav.from_sep_bigger_starts,
        trav.from_sep_bigger_lists,
        weights,
    )
    local_exps, _ = wrangler.refine_locals(
        trav.level_start_target_or_target_parent_box_nrs,
        trav.target_or_target_parent_boxes,
        local_exps,
    )
    potentials, _ = wrangler.eval_locals(
        trav.level_start_target_box_nrs, trav.target_boxes, local_exps
    )
    # one potential per target kernel with sumpy, a plain array with FMMLib
    if potentials.dtype == object:
        (potentials,) = potentials
    return potentials


def _fake_table(q_order):
    # The far-field stages do not read near-field tables; a stand-in that
    # passes the constructor's checks suffices.
    return SimpleNamespace(source_box_extent=1.0, quad_order=q_order, is_built=True)


# (dim, q_order, FMM order, bounds). On a CPU device, against the exact
# integral of the interpolants, the List 4 far field is off by about 7e-11 (2D)
# and 8e-7 (3D) relative to its max from the upsampled sources, and by 3.7e-8
# and 1.2e-4 from point quadrature at the boxes' nodes. The latter is the FMM's
# rendering of the point sum, which it matches to 6e-12 and 2e-8.
LIST4_CASES = [
    (2, 6, 20, {"upsampled": 1e-9, "point_sum": 1e-10}),
    (3, 3, 12, {"upsampled": 1e-5, "point_sum": 1e-6}),
]


def _upsampled_kwargs(dim):
    # The upsampling is on by default in 2D only; 3D asks for it.
    return {} if dim < 3 else {"list4_upsampling": LIST4_UPSAMPLING}


def _check_list4_far_field(dim, q_order, bounds, *, far_field, make_wrangler,
                           exact, point):
    scale = np.max(np.abs(exact))
    wrangler = make_wrangler(**_upsampled_kwargs(dim))
    upsampled_error = np.max(np.abs(far_field(wrangler) - exact)) / scale
    logger.info(
        "%dD q_order %d: List 4 far field off the interpolants' integral by "
        "%.2e from upsampled sources",
        dim, q_order, upsampled_error,
    )
    assert upsampled_error < bounds["upsampled"], (
        f"List 4 far field off the interpolants' integral by "
        f"{upsampled_error:.3e}"
    )
    assert wrangler.list4_upsampling == LIST4_UPSAMPLING
    assert wrangler.list4_upsampled_q_order == list4_upsampled_q_order(
        q_order, LIST4_UPSAMPLING
    )
    if dim == 3:
        assert make_wrangler().list4_upsampling == 1

    plain = far_field(make_wrangler(list4_upsampling=1))
    plain_error = np.max(np.abs(plain - exact)) / scale
    point_sum_error = np.max(np.abs(plain - point)) / scale
    logger.info(
        "%dD q_order %d: with list4_upsampling=1, off the interpolants' "
        "integral by %.2e and off the point sum by %.2e",
        dim, q_order, plain_error, point_sum_error,
    )
    assert point_sum_error < bounds["point_sum"]
    assert plain_error > 30 * upsampled_error


@pytest.mark.parametrize(("dim", "q_order", "fmm_order", "bounds"), LIST4_CASES)
def test_list4_far_field_integrates_the_interpolant(
    ctx_factory, dim, q_order, fmm_order, bounds
):
    """By default the List 4 far field is the exact integral of each source
    box's interpolant, up to the upsampled rule's error; with
    ``list4_upsampling=1`` it is the point sum over the box's nodes."""
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    q_points, q_weights, tree, trav = _split_twice_geometry(ctx, queue, dim, q_order)
    assert len(trav.from_sep_bigger_lists) > 0

    knl = LaplaceKernel(dim)
    expn_factory = DefaultExpansionFactory()
    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(expn_factory.get_multipole_expansion_class(knl), knl),
        partial(expn_factory.get_local_expansion_class(knl), knl),
        [knl],
        exclude_self=True,
    )

    def make_wrangler(**kwargs):
        return FPNDExpansionWrangler(
            tree_indep=tree_indep,
            queue=queue,
            traversal=trav,
            near_field_table=[_fake_table(q_order)],
            dtype=np.float64,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
            quad_order=q_order,
            **kwargs,
        )

    user_nodes = np.array([q_points[i].get(queue) for i in range(dim)])
    user_weights = _density(user_nodes) * q_weights.get(queue)
    weights = make_wrangler().reorder_sources(
        cl.array.to_device(queue, user_weights)
    )
    tree_nodes = np.array([tree.sources[i].get(queue) for i in range(dim)])
    exact, point = _list4_reference(
        queue, tree, trav, q_order=q_order, density=_density(tree_nodes),
        weights=weights.get(queue),
    )

    def far_field(wrangler):
        return _list4_far_field(wrangler, trav, obj_array_1d([weights])).get(queue)

    _check_list4_far_field(dim, q_order, bounds, far_field=far_field,
                           make_wrangler=make_wrangler, exact=exact, point=point)


@pytest.mark.parametrize(("dim", "q_order", "fmm_order", "bounds"), LIST4_CASES)
def test_list4_far_field_integrates_the_interpolant_fmmlib(
    dim, q_order, fmm_order, bounds
):
    """The same with the FMMLib wrangler."""
    pytest.importorskip("pyfmmlib")
    from sumpy.kernel import LaplaceKernel

    from volumential.expansion_wrangler_fpnd import (
        FPNDFMMLibExpansionWrangler,
        FPNDFMMLibTreeIndependentDataForWrangler,
    )

    ctx = create_fp64_context_or_skip()
    queue = cl.CommandQueue(ctx)
    q_points, q_weights, tree, trav = _split_twice_geometry(ctx, queue, dim, q_order)

    knl = LaplaceKernel(dim)
    tree_indep = FPNDFMMLibTreeIndependentDataForWrangler(
        ctx, None, None, [knl], exclude_self=True
    )

    def make_wrangler(**kwargs):
        return FPNDFMMLibExpansionWrangler(
            tree_indep,
            queue,
            tree,
            [_fake_table(q_order)],
            np.complex128,
            lambda kernel, kernel_args, tree_, lev: fmm_order,
            q_order,
            traversal=trav,
            **kwargs,
        )

    user_nodes = np.array([q_points[i].get(queue) for i in range(dim)])
    user_weights = _density(user_nodes) * q_weights.get(queue)
    wrangler = make_wrangler()
    weights = wrangler.reorder_sources(user_weights)
    tree_nodes = np.array([tree.sources[i].get(queue) for i in range(dim)])
    exact, point = _list4_reference(
        queue, tree, trav, q_order=q_order, density=_density(tree_nodes),
        weights=weights,
    )
    scale_factor = wrangler.get_scale_factor()

    def far_field(wrangler):
        return np.real(
            scale_factor * _list4_far_field(wrangler, wrangler.traversal, [weights])
        )

    _check_list4_far_field(dim, q_order, bounds, far_field=far_field,
                           make_wrangler=make_wrangler, exact=exact, point=point)


# }}}

# vim: filetype=pyopencl:foldmethod=marker
