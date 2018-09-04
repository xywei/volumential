'''
FMM convergence test (laplace kernel)
'''
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

import logging
logger = logging.getLogger(__name__)

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

q_order = 2 # quadrature order
n_levels = 6  # 2^(n_levels-1) subintervals in 1D, must be at least 2 if not adaptive
adaptive_mesh = False
n_refinement_loops = 100
refined_n_cells = 1000
rratio_top = 0.2
rratio_bot = 0.5

dtype = np.float64

m_orders = [ i for i in range(10,31) ] # multipole order


if 1:
    # source 1 in a small region, to test for missing interation
    n_subintervals = 2**(n_levels-1)
    h = 2 / n_subintervals
    kk = 0  # radius
    from random import randint
    kx = randint(-n_levels + 1, n_levels - 1)
    ky = randint(-n_levels + 1, n_levels - 1)
    print(kx, ky)
    # RHS
    def source_field(x):
        assert (len(x) == dim)
        return 0
        if x[0] > (kx-kk)*h and x[0] < max(kx+kk,kx+1)*h \
                and x[1] > (ky-kk)*h and x[1] < max(ky+kk,ky+1)*h:
            return -1
        else:
            return 0

    # analytical solution, up to a harmonic function
    def exact_solu(x, y):
        return 0.25 * (x**2 + y**2)

else:

    # a solution that is nearly zero at the boundary

    # alpha = 160 on [-0.5,0.5]^2 is a good choice
    alpha = 160 / np.sqrt(2)

    def source_field(x):
        assert (len(x) == dim)
        assert (dim == 2)
        norm2 = x[0]**2 + x[1]**2
        lap_u = (4 * alpha**2 * norm2 - 4 * alpha) * np.exp(-alpha * norm2)
        return -lap_u

    def exact_solu(x, y):
        norm2 = x**2 + y**2
        return np.exp(-alpha * norm2)


# bounding box
a = -1
b = 1

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# {{{ generate quad points

import volumential.meshgen as mg

if not adaptive_mesh:
    q_points, q_weights, q_radii = mg.make_uniform_cubic_grid(
        degree=q_order, level=n_levels)
    print("Number of quad points:", len(q_points))
else:

    mesh = mg.MeshGen2D(q_order, n_levels)
    iloop = 0
    while mesh.n_active_cells() < refined_n_cells:
        iloop += 1
        crtr = np.array([
            np.abs(source_field(c) * m)
                 for (c, m) in
                 zip(mesh.get_cell_centers(), mesh.get_cell_measures()) ])
        mesh.update_mesh(crtr, rratio_top, rratio_bot)
        if iloop > n_refinement_loops:
            print("Max number of refinement loops reached.")
            break

    mesh.print_info()
    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    q_radii = None

assert (len(q_points) == len(q_weights))
assert (q_points.shape[1] == dim)
print(q_weights)

q_points_org = q_points
q_points = np.ascontiguousarray(np.transpose(q_points))

from pytools.obj_array import make_obj_array
q_points = make_obj_array(
    [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

q_weights = cl.array.to_device(queue, q_weights)
# q_radii = cl.array.to_device(queue, q_radii)

# }}}

# {{{ discretize the source field

def hs(f):
    if abs(f) > 1e-8:
        return 1
    else:
        return 0

source_vals = cl.array.to_device(queue,
        np.array(
            [source_field(qp) for qp in q_points_org]))

print(sum([ hs(source_field(qp)) for qp in q_points_org ]))


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
nftable, _ = tm.get_table(dim, "Laplace", q_order)

# }}} End build near field potential table

# {{{ sumpy expansion for laplace kernel

from sumpy.kernel import LaplaceKernel
from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion

from sumpy.expansion.multipole import LaplaceConformingVolumeTaylorMultipoleExpansion
from sumpy.expansion.local import LaplaceConformingVolumeTaylorLocalExpansion

knl = LaplaceKernel(dim)
out_kernels = [knl]
local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion
# local_expn_class = VolumeTaylorLocalExpansion
# mpole_expn_class = VolumeTaylorMultipoleExpansion

exclude_self = True
from volumential.expansion_wrangler_interface import ExpansionWranglerCodeContainer
wcc = ExpansionWranglerCodeContainer(ctx,
                                     partial(mpole_expn_class, knl),
                                     partial(local_expn_class, knl),
                                     out_kernels,
                                     exclude_self=exclude_self)

if exclude_self:
    target_to_source = np.arange(tree.ntargets, dtype=np.int32)
    self_extra_kwargs = {"target_to_source": target_to_source}
else:
    self_extra_kwargs = {}

# }}} End sumpy expansion for laplace kernel


print("*************************")
print("* Direct Evalulation ...")
print("*************************")

m_order = 1
from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler
from volumential.volume_fmm import drive_volume_fmm

wrangler = FPNDExpansionWrangler(
    code_container=wcc,
    queue=queue,
    tree=tree,
    near_field_table=nftable,
    dtype=dtype,
    fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
    quad_order=q_order,
    self_extra_kwargs=self_extra_kwargs)

direct_pot, = drive_volume_fmm(trav, wrangler,
        source_vals * q_weights, source_vals,
        direct_evaluation=True)
norm_pot = max(abs(direct_pot))
if norm_pot == 0:
    norm_pot = 1

err = []
for m_order in m_orders:

    print("*************************")
    print("* FMM Order =", m_order, "...")
    print("*************************")

    wrangler = FPNDExpansionWrangler(
        code_container=wcc,
        queue=queue,
        tree=tree,
        near_field_table=nftable,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs)

    pot, = drive_volume_fmm(trav, wrangler,
            source_vals * q_weights, source_vals,
            direct_evaluation=False)

    absolute_err = max(abs(direct_pot - pot))
    print("  Relative error =", absolute_err / norm_pot)
    err.append(absolute_err / norm_pot)

print(m_orders)
print(err)

# vim: ft=pyopencl
