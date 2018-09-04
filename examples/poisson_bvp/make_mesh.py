__copyright__ = "Copyright (C) 2017 Xiaoyu Wei"

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

# {{{ generate quad points

import volumential.meshgen as mg

if not adaptive_mesh:
    q_points, q_weights, q_radii = mg.make_uniform_cubic_grid(
        degree=q_order, level=n_levels)
else:

    mesh = mg.MeshGen2D(q_order, n_levels, a, b)
    iloop = 0
    while mesh.n_active_cells() < refined_n_cells:
        iloop += 1
        crtr = np.array([
            np.abs(refinement_flag(source_field(c[0], c[1])) * m)
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

    mesh.generate_gmsh("grid.msh")

assert (len(q_points) == len(q_weights))
assert (q_points.shape[1] == dim)

q_points_org = q_points
q_points = np.ascontiguousarray(np.transpose(q_points))

from pytools.obj_array import make_obj_array
q_points = make_obj_array(
    [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

q_weights = cl.array.to_device(queue, q_weights)
# q_radii = cl.array.to_device(queue, q_radii)

# }}}

# {{{ discretize the source field

source_vals = cl.array.to_device(queue,
                                 np.array(
                                     [source_field(qp[0], qp[1]) for qp in q_points_org]))

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
tm = NearFieldInteractionTableManager(nftable_datafile)
nftable, _ = tm.get_table(dim, "Laplace", q_order)

# }}} End build near field potential table

# {{{ sumpy expansion for laplace kernel

from sumpy.kernel import LaplaceKernel
from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion

knl = LaplaceKernel(dim)
out_kernels = [knl]
local_expn_class = VolumeTaylorLocalExpansion
mpole_expn_class = VolumeTaylorMultipoleExpansion

from volumential.expansion_wrangler_interface import ExpansionWranglerCodeContainer
wcc = ExpansionWranglerCodeContainer(ctx,
                                     partial(mpole_expn_class, knl),
                                     partial(local_expn_class, knl),
                                     out_kernels,
                                     exclude_self=True)

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

# {{{ make boundary mesh

vol_quad_order = q_order

from meshmode.mesh.generation import generate_regular_rect_mesh
from meshmode.mesh.io import generate_gmsh, FileSource  # noqa
if physical_domain == "circle":
    physical_mesh = generate_gmsh(
            FileSource("circle.step"), 2, order=mesh_order,
            force_ambient_dim=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h]
            )

logger.info("%d boundary elements" % physical_mesh.nelements)

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import  \
InterpolatoryQuadratureSimplexGroupFactory
vol_discr = Discretization(ctx, physical_mesh,
        InterpolatoryQuadratureSimplexGroupFactory(vol_quad_order))

from meshmode.mesh import BTAG_ALL
from meshmode.discretization.connection import make_face_restriction

bdry_connection = make_face_restriction(
        vol_discr, InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order),
        BTAG_ALL)
bdry_discr = bdry_connection.to_discr

logger.info("%d boundary node" % bdry_discr.nnodes)

# ensure bounding box properties
assert( a < np.min(physical_mesh.vertices) )
assert( b > np.max(physical_mesh.vertices) )
print(np.min(physical_mesh.vertices),
      np.max(physical_mesh.vertices))

# }}} End make boundary mesh

# {{{ make box discr

# convert from legacy format to whatever up-to-date
import os
os.system('gmsh grid.msh convert_grid -')

from meshmode.mesh.io import read_gmsh
modemesh = read_gmsh("grid.msh", force_ambient_dim=None)

# generate new discretization (not using FMM quad points)
from meshmode.discretization.poly_element import \
        LegendreGaussLobattoTensorProductGroupFactory
box_discr = Discretization(ctx, modemesh,
        LegendreGaussLobattoTensorProductGroupFactory(q_order))

# }}} End make box discr

