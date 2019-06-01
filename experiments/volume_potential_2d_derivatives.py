""" This example evaluates the volume potential and its derivatives
    over [-1,1]^2 with the Laplace kernel.
"""
from __future__ import absolute_import, division, print_function

__copyright__ = "Copyright (C) 2019 Xiaoyu Wei"

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

logger = logging.getLogger(__name__)

if 1:
    # verbose
    logging.basicConfig(level=logging.INFO)
else:
    # clean
    logging.basicConfig(level=logging.CRITICAL)

import numpy as np
import pyopencl as cl
import boxtree as bt
import sumpy as sp
import volumential as vm

from functools import partial

print("*************************")
print("* Setting up...")
print("*************************")

dim = 2

table_filename = "nft_laplace2d.hdf5"
root_table_source_extent = 2

print("Using table cache:", table_filename)

q_order = 3   # quadrature order
n_levels = 9  # 2^(n_levels-1) subintervals in 1D

use_multilevel_table = True

adaptive_mesh = False
n_refinement_loops = 100
refined_n_cells = 2000
rratio_top = 0.2
rratio_bot = 0.5

dtype = np.float64

m_order = 20  # multipole order
force_direct_evaluation = False

print("Multipole order =", m_order)

if 0:
    # source 1 in a small region, to test for missing interation
    n_subintervals = 2 ** (n_levels - 1)
    h = 2 / n_subintervals
    kk = 0
    # RHS
    def source_field(x):
        assert len(x) == dim
        return 0
        if (
            x[0] > -kk * h
            and x[0] < max(kk, 1) * h
            and x[1] > -kk * h
            and x[1] < max(kk, 1) * h
        ):
            return -1
        else:
            return 0

    # analytical solution, up to a harmonic function
    def exact_solu(x, y):
        return 0.25 * (x ** 2 + y ** 2)


else:
    # a solution that is nearly zero at the boundary

    # alpha = 160 on [-0.5,0.5]^2 is a good choice
    alpha = 160

    def source_field(x):
        assert len(x) == dim
        assert dim == 2
        norm2 = x[0] ** 2 + x[1] ** 2
        lap_u = (4 * alpha ** 2 * norm2 - 4 * alpha) * np.exp(-alpha * norm2)
        return -lap_u

    def exact_solu(x, y):
        norm2 = x ** 2 + y ** 2
        return np.exp(-alpha * norm2)

    def exact_solu_dx(x, y):
        norm2 = x ** 2 + y ** 2
        return (-alpha) * np.exp(-alpha * norm2) * (2*x)

    def exact_solu_dy(x, y):
        norm2 = x ** 2 + y ** 2
        return (-alpha) * np.exp(-alpha * norm2) * (2*y)

# bounding box
a = -1
b = 1

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# {{{ generate quad points

import volumential.meshgen as mg

# Show meshgen info
mg.greet()

mesh = mg.MeshGen2D(q_order, n_levels, a, b)
if not adaptive_mesh:
    mesh.print_info()
    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    q_radii = None

else:
    iloop = -1
    while mesh.n_active_cells() < refined_n_cells:
        iloop += 1
        crtr = np.array(
            [
                np.abs(source_field(c) * m)
                for (c, m) in zip(mesh.get_cell_centers(), mesh.get_cell_measures())
            ]
        )
        mesh.update_mesh(crtr, rratio_top, rratio_bot)
        if iloop > n_refinement_loops:
            print("Max number of refinement loops reached.")
            break

    mesh.print_info()
    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    q_radii = None

assert len(q_points) == len(q_weights)
assert q_points.shape[1] == dim

q_points_org = q_points
q_points = np.ascontiguousarray(np.transpose(q_points))

from pytools.obj_array import make_obj_array

q_points = make_obj_array([cl.array.to_device(queue, q_points[i]) for i in range(dim)])

q_weights = cl.array.to_device(queue, q_weights)
# q_radii = cl.array.to_device(queue, q_radii)

# }}}

# {{{ discretize the source field

source_vals = cl.array.to_device(
    queue, np.array([source_field(qp) for qp in q_points_org])
)

# particle_weigt = source_val * q_weight

# }}} End discretize the source field

# {{{ build tree and traversals

from boxtree.tools import AXIS_NAMES

axis_names = AXIS_NAMES[:dim]

from pytools import single_valued

coord_dtype = single_valued(coord.dtype for coord in q_points)
from boxtree.bounding_box import make_bounding_box_dtype

bbox_type, _ = make_bounding_box_dtype(ctx.devices[0], dim, coord_dtype)
bbox = np.empty(1, bbox_type)
for ax in axis_names:
    bbox["min_" + ax] = a
    bbox["max_" + ax] = b


# tune max_particles_in_box to reconstruct the mesh
# TODO: use points from FieldPlotter are used as target points for better
# visuals
from boxtree import TreeBuilder

tb = TreeBuilder(ctx)
tree, _ = tb(
    queue,
    particles=q_points,
    targets=q_points,
    bbox=bbox,
    max_particles_in_box=q_order ** 2 * 4 - 1,
    kind="adaptive-level-restricted",
)

bbox2 = np.array([[a, b], [a, b]])
tree2, _ = tb(
    queue,
    particles=q_points,
    targets=q_points,
    bbox=bbox2,
    max_particles_in_box=q_order ** 2 * 4 - 1,
    kind="adaptive-level-restricted",
)

from boxtree.traversal import FMMTraversalBuilder

tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

# }}} End build tree and traversals

# {{{ build near field potential table

from volumential.table_manager import NearFieldInteractionTableManager
from volumential.list1_symmetry import Flip

tm = NearFieldInteractionTableManager(
    table_filename, root_extent=root_table_source_extent
)

if use_multilevel_table:
    assert (
        abs(
            int((b - a) / root_table_source_extent) * root_table_source_extent - (b - a)
        )
        < 1e-15
    )
    nftable_list = []
    nftable_dx_list = []
    nftable_dy_list = []
    for l in range(0, tree.nlevels + 1):
        if 1:
            print("Getting table at level", l)
            tb, _ = tm.get_table(dim, "Laplace", q_order,
                source_box_level=l, compute_method="DrosteSum",
                queue=queue, n_brick_quad_poitns=100,
                adaptive_level=False, use_symmetry=True,
                alpha=0.1, nlevels=15)
            nftable_list.append(tb)

        if 1:
            print("Getting table Dx at level", l)
            tb, _ = tm.get_table(dim, "Laplace-Dx", q_order,
                source_box_level=l, compute_method="DrosteSum",
                queue=queue, n_brick_quad_poitns=100,
                adaptive_level=False, use_symmetry=False,
                alpha=0.1, nlevels=15)
            nftable_dx_list.append(tb)

        if 1:
            print("Getting table Dy at level", l)
            tb, _ = tm.get_table(dim, "Laplace-Dy", q_order,
                source_box_level=l, compute_method="DrosteSum",
                queue=queue, n_brick_quad_poitns=100,
                adaptive_level=False, use_symmetry=False,
                alpha=0.1, nlevels=15)
            nftable_dy_list.append(tb)

    print("Using table list of length", len(nftable_list))
    nftable = {
            nftable_list[0].integral_knl.__repr__(): nftable_list,
            nftable_dx_list[0].integral_knl.__repr__(): nftable_dx_list,
            nftable_dy_list[0].integral_knl.__repr__(): nftable_dy_list,
            }

else:
    1/0

# }}} End build near field potential table

# {{{ sumpy expansion for laplace kernel

from sumpy.expansion import DefaultExpansionFactory
from sumpy.kernel import LaplaceKernel, AxisTargetDerivative

knl = LaplaceKernel(dim)
knl_dx = AxisTargetDerivative(0, knl)
knl_dy = AxisTargetDerivative(1, knl)

expn_factory = DefaultExpansionFactory()
local_expn_class = expn_factory.get_local_expansion_class(knl)
mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)
out_kernels = [knl, knl_dx, knl_dy]

exclude_self = True
from volumential.expansion_wrangler_interface import ExpansionWranglerCodeContainer

wcc = ExpansionWranglerCodeContainer(
    ctx,
    partial(mpole_expn_class, knl),
    partial(local_expn_class, knl),
    out_kernels,
    exclude_self=exclude_self,
)

if exclude_self:
    target_to_source = np.arange(tree.ntargets, dtype=np.int32)
    self_extra_kwargs = {"target_to_source": target_to_source}
else:
    self_extra_kwargs = {}

from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler

wrangler = FPNDExpansionWrangler(
    code_container=wcc,
    queue=queue,
    tree=tree,
    near_field_table=nftable,
    dtype=dtype,
    fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
    quad_order=q_order,
    self_extra_kwargs=self_extra_kwargs,
)

# }}} End sumpy expansion for laplace kernel

print("*************************")
print("* Performing FMM ...")
print("*************************")

# {{{ conduct fmm computation

from volumential.volume_fmm import drive_volume_fmm

import time
queue.finish()

t0 = time.clock()

pot = drive_volume_fmm(
    trav,
    wrangler,
    source_vals * q_weights,
    source_vals,
    direct_evaluation=force_direct_evaluation,
)
queue.finish()

t1 = time.clock()

print("Finished in %.2f seconds." % (t1 - t0))
print("(%e points per second)" % (
    len(q_weights) / (t1 - t0)
    ))

# }}} End conduct fmm computation

print("*************************")
print("* Postprocessing ...")
print("*************************")

# {{{ postprocess and plot

# print(pot)

x = q_points[0].get()
y = q_points[1].get()

ze = exact_solu(x, y)
zs = pot[0].get()

ze_dx = exact_solu_dx(x, y)
zs_dx = pot[1].get()

ze_dy = exact_solu_dy(x, y)
zs_dy = pot[2].get()

print_error = True
if print_error:
    err = np.max(np.abs(ze - zs))
    err_dx = np.max(np.abs(ze_dx - zs_dx))
    err_dy = np.max(np.abs(ze_dy - zs_dy))
    print("Error =", err)
    print("Error dx =", err_dx)
    print("Error dy =", err_dy)

# Interpolated surface
if 0:
    h = 0.005
    out_x = np.arange(a, b + h, h)
    out_y = np.arange(a, b + h, h)
    oxx, oyy = np.meshgrid(out_x, out_y)
    out_targets = make_obj_array(
        [
            cl.array.to_device(queue, oxx.flatten()),
            cl.array.to_device(queue, oyy.flatten()),
        ]
    )

    from volumential.volume_fmm import interpolate_volume_potential

    # src = source_field([q.get() for q in q_points])
    # src = cl.array.to_device(queue, src)
    interp_pot = interpolate_volume_potential(out_targets, trav, wrangler, pot)
    opot = interp_pot.get()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt3d = plt.figure()
    ax = Axes3D(plt3d)
    surf = ax.plot_surface(oxx, oyy, opot.reshape(oxx.shape))
    # ax.scatter(x, y, src.get())
    # ax.set_zlim(-0.25, 0.25)

    plt.draw()
    plt.show()

# Boxtree
if 0:
    import matplotlib.pyplot as plt

    if dim == 2:
        # plt.plot(q_points[0].get(), q_points[1].get(), ".")
        pass

    from boxtree.visualization import TreePlotter

    plotter = TreePlotter(tree.get(queue=queue))
    plotter.draw_tree(fill=False, edgecolor="black")
    # plotter.draw_box_numbers()
    plotter.set_bounding_box()
    plt.gca().set_aspect("equal")

    plt.draw()
    # plt.show()
    plt.savefig("tree.png")


# Direct p2p
if 0:
    print("Performing P2P")
    pot_direct, = drive_volume_fmm(
        trav, wrangler, source_vals * q_weights, source_vals, direct_evaluation=True
    )
    zds = pot_direct.get()
    zs = pot.get()

    print("P2P-FMM diff =", np.max(np.abs(zs - zds)))

    print("P2P Error =", np.max(np.abs(ze - zds)))

    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    x = q_points[0].get()
    y = q_points[1].get()
    plt.scatter(x, y, c=np.log(abs(zs-zds)) / np.log(10), cmap=cm.jet)
    plt.colorbar()

    plt.xlabel("Multipole order = " + str(m_order))

    plt.draw()
    plt.show()
    """

# Scatter plot
if 0:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    n = 2 ** (n_levels - 1) * q_order

    plt3d = plt.figure()
    ax = Axes3D(plt3d)
    ax.scatter(x, y, zs_dx, s=1)
    ax.scatter(x, y, ze_dx, s=1)
    # ax.scatter(x, y, source_field([q.get() for q in q_points]), s=1)
    import matplotlib.cm as cm

    # ax.scatter(x, y, zs, c=np.log(abs(zs-zds)), cmap=cm.jet)
    # plt.gca().set_aspect("equal")

    # ax.set_xlim3d([-1, 1])
    # ax.set_ylim3d([-1, 1])
    # ax.set_zlim3d([np.min(z), np.max(z)])
    # ax.set_zlim3d([-0.002, 0.00])

    plt.draw()
    plt.show()
    # plt.savefig("exact.png")

# }}} End postprocess and plot

# vim: filetype=pyopencl:foldmethod=marker
