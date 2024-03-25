
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

import os
import subprocess
import numpy as np
import pyopencl as cl
import pytest
import volumential as vm
from shutil import copyfile
from volumential.table_manager import (
        NearFieldInteractionTableManager as NFTable)


def get_table(queue, q_order=1, dim=2):
    pid = os.getpid()

    # copy from shared cache if exists
    if os.path.exists("nft.hdf5"):
        copyfile("nft.hdf5",
                 f"nft-test-table-manager-{pid}.hdf5")

    subprocess.check_call(['rm', '-f', f'nft-test-table-manager-{pid}.hdf5'])

    with NFTable(f"nft-test-table-manager-{pid}.hdf5",
                 progress_bar=True) as table_manager:
        table, _ = table_manager.get_table(
                dim, "Laplace",
                q_order=q_order, force_recompute=False, queue=queue)

    return table


def test_case_id(ctx_factory, table_2d_order1):
    table = table_2d_order1
    case_same_box = len(table.interaction_case_vecs) // 2
    assert list(table.interaction_case_vecs[case_same_box]) == [0, 0]


def test_get_table_2d_order1(table_2d_order1):
    table = table_2d_order1
    assert table.dim == 2


def laplace_const_source_same_box(table_2d_order1, queue, q_order, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    n_pairs = nft.n_pairs
    n_q_points = nft.n_q_points
    pot = np.zeros(n_q_points)

    case_same_box = len(nft.interaction_case_vecs) // 2

    for source_mode_index in range(n_q_points):
        for target_point_index in range(n_q_points):
            pair_id = source_mode_index * n_q_points + target_point_index
            entry_id = case_same_box * n_pairs + pair_id
            # print(source_mode_index, target_point_index, pair_id, entry_id,
            # nft.data[entry_id])
            pot[target_point_index] += 1.0 * nft.data[entry_id]

    return pot


def laplace_cons_source_neighbor_box(
        table_2d_order1, queue, q_order, case_id, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    n_pairs = nft.n_pairs
    n_q_points = nft.n_q_points
    pot = np.zeros(n_q_points)

    for source_mode_index in range(n_q_points):
        for target_point_index in range(n_q_points):
            pair_id = source_mode_index * n_q_points + target_point_index
            entry_id = case_id * n_pairs + pair_id
            # print(source_mode_index, target_point_index, pair_id, entry_id,
            # nft.data[entry_id])
            pot[target_point_index] += 1.0 * nft.data[entry_id]

    return pot


def test_lcssb_1(ctx_factory, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    u = laplace_const_source_same_box(table_2d_order1, queue, 1)
    assert len(u) == 1


def interp_func(table_2d_order1, queue, q_order, coef, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    assert dim == 2

    modes = [nft.get_mode(i) for i in range(nft.n_q_points)]

    def func(x, y):
        z = np.zeros(np.array(x).shape)
        for i in range(nft.n_q_points):
            mode = modes[i]
            z += (coef[i] * mode(x, y)).reshape(z.shape)
        return z

    return func


def test_interp_func(longrun, ctx_factory, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    q_order = 3
    coef = np.ones(q_order ** 2)

    h = 0.1
    xx = yy = np.arange(-1.0, 1.0, h)
    xi, yi = np.meshgrid(xx, yy)
    func = interp_func(table_2d_order1, queue, q_order, coef)

    zi = func(xi, yi)

    assert np.allclose(zi, 1)


def direct_quad(source_func, target_point, dim=2):
    knl_func = vm.nearfield_potential_table.get_laplace(dim)

    def integrand(x, y):
        return source_func(x, y) * knl_func(x - target_point[0], y - target_point[1])
    import volumential.singular_integral_2d as squad
    integral, _ = squad.box_quad(
        func=integrand, a=0, b=1, c=0, d=1,
        singular_point=target_point, maxiter=1000)
    return integral


def drive_test_direct_quad_same_box(table_2d_order1, queue, q_order, dim=2):
    u = laplace_const_source_same_box(table_2d_order1, queue, q_order)
    func = interp_func(table_2d_order1, queue, q_order, u)

    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    def const_one_source_func(x, y):
        return 1

    # print(nft.compute_table_entry(1341))
    # print(nft.compute_table_entry(nft.lookup_by_symmetry(1341)))

    for it in range(nft.n_q_points):
        target = nft.q_points[it]
        v1 = func(target[0], target[1])
        v2 = direct_quad(const_one_source_func, target)
        v3 = 0
        for ids in range(nft.n_q_points):
            mode = nft.get_mode(ids)
            vv = direct_quad(mode, target)
            print(ids, it, vv)
            v3 += vv

        print(target, v1, v2, v3)
        assert np.abs(v1 - v2) < 2e-6
        assert np.abs(v1 - v3) < 1e-6


@pytest.mark.parametrize("q_order", [1, ])
def test_direct_quad(q_order, ctx_factory, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    drive_test_direct_quad_same_box(table_2d_order1, queue, q_order)


@pytest.mark.parametrize("q_order", [2, 3, 4, 5])
def test_direct_quad_longrun(longrun, ctx_factory, q_order, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    drive_test_direct_quad_same_box(table_2d_order1, queue, q_order)


def test_case_ids(ctx_factory, table_2d_order1):
    table = table_2d_order1
    for i in range(len(table.interaction_case_vecs)):
        code = table.case_encode(
                table.interaction_case_vecs[i])
        assert table.case_indices[code] == i


def get_target_point(case_id, target_id, table):
    case_vec = table.interaction_case_vecs[case_id]
    center = np.array([0.5, 0.5]) + np.array(case_vec) * 0.25
    dist = np.max(np.abs(case_vec)) - 2
    if dist == 1:
        scale = 0.5
    elif dist == 2:
        scale = 1
    elif dist == 4:
        scale = 2
    dx = table.q_points[target_id][0] - 0.5
    dy = table.q_points[target_id][1] - 0.5
    target_point = np.array([center[0] + dx * scale, center[1] + dy * scale])
    return target_point


def test_get_neighbor_target_point(ctx_factory, table_2d_order1):
    table = table_2d_order1
    case_same_box = len(table.interaction_case_vecs) // 2
    for cid in range(len(table.interaction_case_vecs)):
        if cid == case_same_box:
            continue
        for tpid in range(table.n_q_points):
            pt = table.find_target_point(tpid, cid)
            pt2 = get_target_point(cid, tpid, table)
        assert np.allclose(pt, pt2)


def laplace_const_source_neighbor_box(
        table_2d_order1, queue, q_order, case_id, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    n_pairs = nft.n_pairs
    n_q_points = nft.n_q_points
    pot = np.zeros(n_q_points)

    for source_mode_index in range(n_q_points):
        for target_point_index in range(n_q_points):
            pair_id = source_mode_index * n_q_points + target_point_index
            entry_id = case_id * n_pairs + pair_id
            pot[target_point_index] += 1.0 * nft.data[entry_id]
    return pot


def drive_test_direct_quad_neighbor_box(
        table_2d_order1, queue, q_order, case_id, dim=2):
    u = laplace_const_source_neighbor_box(
            table_2d_order1, queue, q_order, case_id)
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    def const_one_source_func(x, y):
        return 1

    for it in range(nft.n_q_points):
        target = nft.find_target_point(it, case_id)
        v1 = u[it]
        v2 = direct_quad(const_one_source_func, target)
        v3 = 0
        for ids in range(nft.n_q_points):
            mode = nft.get_mode(ids)
            vv = direct_quad(mode, target)
            print(ids, it, vv)
            v3 += vv

        print(target, v1, v2, v3)
        assert np.abs(v1 - v2) < 2e-6
        assert np.abs(v1 - v3) < 1e-6


@pytest.mark.parametrize("q_order", [1, ])
def test_direct_quad_neighbor_box(ctx_factory, q_order, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    table = table_2d_order1
    for case_id in range(len(table.interaction_case_vecs)):
        drive_test_direct_quad_neighbor_box(
                table_2d_order1, queue, q_order, case_id)


@pytest.mark.parametrize("q_order", [2, ])
def test_direct_quad_neighbor_box_longrun(
        longrun, ctx_factory, q_order, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    table = table_2d_order1
    for case_id in range(len(table.interaction_case_vecs)):
        drive_test_direct_quad_neighbor_box(
                table_2d_order1, queue, q_order, case_id)


# fdm=marker:ft=pyopencl
