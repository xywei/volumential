from __future__ import absolute_import, division, print_function

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

import subprocess
import pytest
import numpy as np
import volumential as vm
from volumential.table_manager import NearFieldInteractionTableManager


dim = 2
subprocess.check_call(['rm', '-f', 'nft.hdf5'])
table_manager = NearFieldInteractionTableManager()
table, _ = table_manager.get_table(dim, "Laplace", q_order=1, force_recompute=False)

case_same_box = len(table.interaction_case_vecs) // 2


def test_case_ids():
    for i in range(len(table.interaction_case_vecs)):
        code = table.case_encode(table.interaction_case_vecs[i])
        assert table.case_indices[code] == i


def get_target_point(case_id, target_id):
    assert case_id != case_same_box
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


def test_get_neighbor_target_point():

    for cid in range(len(table.interaction_case_vecs)):

        if cid == case_same_box:
            continue

        for tpid in range(table.n_q_points):
            pt = table.find_target_point(tpid, cid)
            pt2 = get_target_point(cid, tpid)

        assert np.allclose(pt, pt2)


def laplace_const_source_neighbor_box(q_order, case_id):
    nft, _ = table_manager.get_table(
        dim, "Laplace", q_order=q_order, force_recompute=False
    )

    n_pairs = nft.n_pairs
    n_q_points = nft.n_q_points
    pot = np.zeros(n_q_points)

    for source_mode_index in range(n_q_points):
        for target_point_index in range(n_q_points):
            pair_id = source_mode_index * n_q_points + target_point_index
            entry_id = case_id * n_pairs + pair_id
            pot[target_point_index] += 1.0 * nft.data[entry_id]

    return pot


def direct_quad(source_func, target_point):

    knl_func = vm.nearfield_potential_table.get_laplace(dim)

    def integrand(x, y):
        return source_func(x, y) * knl_func(x - target_point[0], y - target_point[1])

    import volumential.singular_integral_2d as squad

    integral, error = squad.box_quad(
        func=integrand, a=0, b=1, c=0, d=1, singular_point=target_point, maxiter=1000
    )

    return integral


def drive_test_direct_quad_neighbor_box(q_order, case_id):
    u = laplace_const_source_neighbor_box(q_order, case_id)

    nft, _ = table_manager.get_table(
        dim, "Laplace", q_order=q_order, force_recompute=False
    )

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
def test_direct_quad_neighbor_box(q_order):
    subprocess.check_call(['rm', '-f', 'nft.hdf5'])
    for case_id in range(len(table.interaction_case_vecs)):
        drive_test_direct_quad_neighbor_box(q_order, case_id)


@pytest.mark.parametrize("q_order", [2, ])
def test_direct_quad_neighbor_box_longrun(longrun, q_order):
    subprocess.check_call(['rm', '-f', 'nft.hdf5'])
    for case_id in range(len(table.interaction_case_vecs)):
        drive_test_direct_quad_neighbor_box(q_order, case_id)


# fdm=marker:ft=pyopencl


# fdm=marker:ft=pyopencl
