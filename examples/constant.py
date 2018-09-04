# NOTE: to run this, you need to modify line 299 of sumpy/p2e.py
# and remove rscale from the loopy knl call

from __future__ import absolute_import, division, print_function

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pyopencl as cl
import boxtree as bt
import sumpy as sp
import volumential as vm

from functools import partial

dim = 2

q_order = 1  # quadrature order
n_levels = 4  # 2^(n_levels-1) subintervals in 1D, must be at least 2
dtype = np.float64

# NOTE: setting m_order too low (~5) does not work
m_order = 1  # multipole order


# RHS
def source_field(x):
    assert (len(x) == dim)
    return 1


# analytical solution
def exact_solu(x, y):
    return 0.5 * (x**2 + y**2)


# bounding box
a = -1
b = 1

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# {{{ generate quad points

import volumential.meshgen as mg
q_points, q_weights, q_radii = mg.make_uniform_cubic_grid(
    degree=q_order, level=n_levels)

assert (len(q_points) == len(q_weights))
assert (q_points.shape[1] == dim)

q_points_org = q_points
q_points = np.ascontiguousarray(np.transpose(q_points))

from pytools.obj_array import make_obj_array
q_points = make_obj_array(
    [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

q_weights = cl.array.to_device(queue, q_weights)
q_radii = cl.array.to_device(queue, q_radii)

# }}}

# {{{ discretize the source field

source_vals = cl.array.to_device(queue,
                                 np.array(
                                     [source_field(qp) for qp in q_points_org]))

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
    max_particles_in_box=q_order**2 * 4 - 1,
    kind="adaptive-level-restricted")

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

# }}} End build tree and traversals

# {{{ build near field potential table

from volumential.table_manager import NearFieldInteractionTableManager
tm = NearFieldInteractionTableManager("nft.hdf5")
nftable, _ = tm.get_table(dim, "Constant", q_order)

# }}} End build near field potential table

# {{{ sumpy expansion for constant kernel

from sumpy.kernel import ExpressionKernel
from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion

class ConstantKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self, dim=None):
        expr = 1
        scaling = 1

        super(ConstantKernel, self).__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                is_complex_valued=False)

    has_efficient_scale_adjustment = True

    def adjust_for_kernel_scaling(self, expr, rscale, nderivatives):
        return expr * rscale

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "ConKnl%dD" % self.dim

    # Using some existing mapping
    mapper_method = "map_expression_kernel"

knl = ConstantKernel(2)

out_kernels = [knl]
local_expn_class = VolumeTaylorLocalExpansion
mpole_expn_class = VolumeTaylorMultipoleExpansion

from volumential.expansion_wrangler_interface import ExpansionWranglerCodeContainer
wcc = ExpansionWranglerCodeContainer(ctx,
                                     partial(mpole_expn_class, knl),
                                     partial(local_expn_class, knl),
                                     out_kernels)

from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler
wrangler = FPNDExpansionWrangler(
    code_container=wcc,
    queue=queue,
    tree=tree,
    near_field_table=nftable,
    dtype=dtype,
    fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
    quad_order=q_order)

# }}} End sumpy expansion for laplace kernel

# {{{ conduct fmm computation

from volumential.volume_fmm import drive_volume_fmm
pot, = drive_volume_fmm(trav, wrangler, q_weights, source_vals)

# }}} End conduct fmm computation

# {{{ postprocess and plot

n_cells = (2**(n_levels-1))**2
cell_area = 4 / n_cells
print("Number of cells =", n_cells)
print("Cell area =", cell_area)

pot = pot.get()
pot /= cell_area

print(pot)

if 0:
    import matplotlib.pyplot as plt

    if dim == 2:
        plt.plot(q_points[0].get(), q_points[1].get(), ".")

    from boxtree.visualization import TreePlotter
    plotter = TreePlotter(tree.get(queue=queue))
    plotter.draw_tree(fill=False, edgecolor="black")
    plotter.draw_box_numbers()
    plotter.set_bounding_box()
    plt.gca().set_aspect("equal")
    plt.savefig("tree.png")

if 0:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x = q_points[0].get()
    y = q_points[1].get()
    zs = pot

    plt3d = plt.figure()
    ax = Axes3D(plt3d)
    ax.scatter(x, y, zs)

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])

    plt.draw()
    plt.show()
    #plt.savefig("exact.png")

# }}} End postprocess and plot

# vim: filetype=pyopencl:foldmethod=marker
