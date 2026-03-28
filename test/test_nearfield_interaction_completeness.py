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
import sys
from functools import partial

import numpy as np
import pytest

if (
    sys.platform == "darwin"
    and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
):
    pytest.skip(
        "nearfield completeness tests are unstable on macOS OpenCL CI "
        "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)",
        allow_module_level=True,
    )

import pyopencl as cl
import pyopencl.array  # noqa: F401


def drive_test_completeness(ctx, queue, dim, q_order):

    n_levels = 2  # 2^(n_levels-1) subintervals in 1D, must be at least 2

    # bounding box
    a = -1
    b = 1

    dtype = np.float64

    def source_field(x):
        assert len(x) == dim
        return 1

    # {{{ generate quad points

    import volumential.meshgen as mg

    mesh_cls = {2: mg.MeshGen2D, 3: mg.MeshGen3D}[dim]
    mesh = mesh_cls(q_order, n_levels, a, b)
    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    q_radii = None

    assert len(q_points) == len(q_weights)
    assert q_points.shape[1] == dim

    q_points_org = q_points
    q_points = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import new_1d as obj_array_1d

    q_points = obj_array_1d(
        [cl.array.to_device(queue, q_points[i]) for i in range(dim)]
    )

    q_weights = cl.array.to_device(queue, q_weights)
    if q_radii is not None:
        q_radii = cl.array.to_device(queue, q_radii)

    # }}}

    # {{{ discretize the source field

    source_vals = cl.array.to_device(
        queue, np.array([source_field(qp) for qp in q_points_org])
    )

    # }}} End discretize the source field

    # {{{ build tree and traversals

    # tune max_particles_in_box to reconstruct the mesh
    # TODO: use points from FieldPlotter are used as target points for better
    # visuals
    from boxtree.array_context import PyOpenCLArrayContext
    from volumential.tree_interactive_build import build_particle_tree_from_box_tree

    actx = PyOpenCLArrayContext(queue)
    tree = build_particle_tree_from_box_tree(actx, mesh.boxtree, q_points_org)

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(actx)
    trav, _ = tg(actx, tree)

    # }}} End build tree and traversals

    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    subprocess.check_call(["rm", "-f", "nft-test-completeness.hdf5"])
    with NearFieldInteractionTableManager(
        "nft-test-completeness.hdf5", progress_bar=False
    ) as tm:
        build_config = DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=20 if dim == 2 else 6,
            radial_quad_order=61 if dim == 2 else 21,
        )
        nft, _ = tm.get_table(
            dim,
            "Constant",
            q_order,
            queue=queue,
            build_config=build_config,
        )

    # {{{ expansion wrangler

    from sumpy.expansion.local import VolumeTaylorLocalExpansion
    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    from sumpy.kernel import LaplaceKernel

    knl = LaplaceKernel(dim)
    out_kernels = [knl]
    local_expn_class = partial(VolumeTaylorLocalExpansion, use_rscale=True)
    mpole_expn_class = partial(VolumeTaylorMultipoleExpansion, use_rscale=True)

    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        out_kernels,
        exclude_self=True,
    )

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=trav,
        near_field_table=nft,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: 1,
        quad_order=q_order,
    )

    # }}} End sumpy expansion for laplace kernel
    pot = wrangler.eval_direct(
        trav.target_boxes,
        trav.neighbor_source_boxes_starts,
        trav.neighbor_source_boxes_lists,
        mode_coefs=source_vals,
    )
    pot = actx.to_numpy(pot[0])
    for p in pot[0]:
        assert abs(p - 2**dim) < 1.0e-8


def test_completeness_1(ctx_factory):

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    drive_test_completeness(ctx, queue, 2, 1)
    drive_test_completeness(ctx, queue, 3, 1)


def test_completeness_q2_cpu_smoke(ctx_factory):

    ctx = ctx_factory()
    if not any(dev.type & cl.device_type.CPU for dev in ctx.devices):
        pytest.skip("q2 completeness smoke test runs on CPU contexts only")

    queue = cl.CommandQueue(ctx)

    drive_test_completeness(ctx, queue, 2, 2)
    drive_test_completeness(ctx, queue, 3, 2)


def test_completeness(longrun, ctx_factory):

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    for q in range(2, 4):
        drive_test_completeness(ctx, queue, 2, q)
        drive_test_completeness(ctx, queue, 3, q)
