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

dim = 2

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym, norm  # noqa
from pytential.target import PointsTarget

from pytools.obj_array import make_obj_array
import pytential.symbolic.primitives as p

from sumpy.kernel import FactorizedBiharmonicKernel, YukawaKernel
from sumpy.kernel import AxisTargetDerivative, LaplacianTargetDerivative
from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion

import pymbolic as pmbl

from volumential.expansion_wrangler_interface import ExpansionWranglerCodeContainer
from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler

import loopy as lp
from functools import partial

from tabulate import tabulate

import warnings
from loopy import LoopyWarning
warnings.filterwarnings("ignore", category=LoopyWarning)

from enum import Enum
class Geometry(Enum):
    RegularRectangle = 1
    Circle = 2

import volumential.meshgen as mg
from volumential.table_manager import NearFieldInteractionTableManager
from volumential.volume_fmm import drive_volume_fmm

cl_ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(cl_ctx)
if "Portable Computing Language" not in queue.device.__str__():
    from warnings import warn
    warn("Not running on POCL.")

# clean up the mess
def clean_file(filename):
    import os
    try:
        os.remove(filename)
    except OSError:
        pass

# {{{ set some constants for use below

verbose = False

# {{{ all kinds of orders
h = 0.01
mesh_order = 2 # for gmsh

bdry_quad_order = 4
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 2 * bdry_quad_order

fmm_order = 8

visual_order = 2

adaptive_mesh = True
n_levels = 4
refined_n_cells = 10000
n_refinement_loops = 100
rratio_top = 1
rratio_bot = 0

vol_quad_order = 2
# }}}

# {{{ mesh generation
nelements = 20

from enum import Enum

class Geometry(Enum):
    RegularRectangle = 1
    Circle = 2

shape = Geometry.Circle
# shape = Geometry.RegularRectangle
# }}}

# {{{ physical parameters
# For energy stability,
# 1. s >= (3M^2-1)/2, where M=max(|phi|)
#    e.g., when M=1, need S>=1
# 2. alpha_tild >= N/2, where N=|sqrt(2)/3 * (pi/2)^2 * cos(theta_y)|
#    e.g., when theta_y=0, N=1.163144 is the maximum possible value of N
# alpha_tild = 0 in the notes.
s = 1.5
alpha_tild = 0

epsilon = 0.01
delta_t = 0.05
final_t = delta_t * 1
theta_y = 60. / 180. * np.pi

b = s / (epsilon**2)
c = 1. / (epsilon * delta_t)
bs = alpha_tild / epsilon
# }}}

# {{{ bounding box and pre-computed table
box_a = -1
box_b = 1
root_table_source_extent = 2

table_filename = "nft_ch.hdf5"
# }}} End box tree and pre-computed table

# {{{ PDE's rhs formulae


import pymbolic.functions as pf

x = pmbl.var("x")
y = pmbl.var("y")

# set a pair of desired solution

# solution id
solid = 1

if solid == 0:
    exact_phi = pf.sin(8 * np.pi * x) * pf.cos(16 * np.pi * y) + x**2 * y - y**3
    exact_mu  = pf.exp(x + y + 5) * pf.sin(4 * np.pi * x * y) + 10

elif solid == 1:
    exact_phi = 2 * x + y - 1.5
    exact_mu  = 0 * x


exact_grad_phi = [pmbl.differentiate(exact_phi, var) for var in ['x', 'y']]

# find the fhs and bc
d2 = lambda f, var : pmbl.differentiate(pmbl.differentiate(f, var), var)
lap = lambda f : d2(f, 'x') + d2(f, 'y')
bilap = lambda f : lap(lap(f))

op_L1 = lambda phi, mu : bilap(phi) - b * lap(phi) + c * phi
op_L2 = lambda phi, mu : (1 / epsilon) * mu + lap(phi) - b * phi

f1_expr = op_L1(exact_phi, exact_mu)
f2_expr = op_L2(exact_phi, exact_mu)

grad_f2_expr = [pmbl.differentiate(f2_expr, var) for var in ['x', 'y']]

# NOTE: b1 is mesh-dependent, so we only evaluate it later numerically

def expr_to_field(expr):
    import math
    def field_func(x):
        return pmbl.evaluate(expr,
                {"x": x[0], "y": x[1], "math": math})
    return field_func

f1_func = expr_to_field(f1_expr)
f2_func = expr_to_field(f2_expr)

# used for bc
f2_x_func, f2_y_func = (expr_to_field(expression) for expression in grad_f2_expr)
phi_func = expr_to_field(exact_phi)
phi_x_func, phi_y_func = (expr_to_field(expression) for expression  in exact_grad_phi)

# used to compare with numerical solutions
mu_func = expr_to_field(exact_mu)

# }}} End PDE rhs formulae

# }}}

def full_coverage(x):
    return 1

def circular_coverage(r, x):
    rad = np.linalg.norm(x)
    if rad > r:
        return 0
    else:
        return 1

def rectangular_coverage(a, b, c, d, x):
    if x[0] > a and x[0] < b and x[1] > c and x[1] < d:
        return 1
    else:
        return 0

geometry_mask = full_coverage

def main():
    from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_1
    import logging
    if verbose:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("volume mesh generation")

    # {{{ volume mesh generation

    if shape == Geometry.RegularRectangle:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        ext = 1.
        mesh = generate_regular_rect_mesh(
            a=(-ext / 2., -ext / 2.),
            b=(ext / 2., ext / 2.),
            n=(int(ext / h), int(ext / h)))

        geometry_mask = partial(rectangular_coverage,
                -ext / 2, ext / 2, -ext / 2, ext / 2)

    elif shape == Geometry.Circle:
        print("running on a circle")
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
            FileSource("circle.step"),
            2,
            order=mesh_order,
            force_ambient_dim=2,
            other_options=[
                "-string", "Mesh.CharacteristicLengthMax = %g;" % h
            ])

        geometry_mask = partial(circular_coverage, 0.25)

    else:
        RuntimeError("unsupported geometry")

    logger.info("%d elements" % mesh.nelements)

    # }}}

    print("discretization and connections")

    # {{{ discretization and connections

    vol_discr = Discretization(
        cl_ctx, mesh,
        InterpolatoryQuadratureSimplexGroupFactory(vol_quad_order))

    from meshmode.mesh import BTAG_ALL
    from meshmode.discretization.connection import make_face_restriction
    pre_density_connection = make_face_restriction(
        vol_discr,
        InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order), BTAG_ALL)
    pre_density_discr = pre_density_connection.to_discr

    from pytential.qbx import (QBXLayerPotentialSource,
                               QBXTargetAssociationFailedException)

    qbx, _ = QBXLayerPotentialSource(
        pre_density_discr,
        fine_order=bdry_ovsmp_quad_order,
        qbx_order=qbx_order,
        fmm_order=fmm_order
        ).with_refinement()

    density_discr = qbx.density_discr

    # composition of connetions
    # vol_discr --> pre_density_discr --> density_discr
    # via ChainedDiscretizationConnection

    #    from meshmode.mesh.generation import ellipse, make_curve_mesh
    #    from functools import partial
    #
    #    mesh = make_curve_mesh(
    #                partial(ellipse, 2),
    #                np.linspace(0, 1, nelements+1),
    #                mesh_order)
    #
    #    pre_density_discr = Discretization(
    #            cl_ctx, mesh,
    #            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))
    #
    #    from pytential.qbx import (
    #            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    #    qbx, _ = QBXLayerPotentialSource(
    #            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
    #            fmm_order=fmm_order,
    #            expansion_disks_in_tree_have_extent=True,
    #            ).with_refinement()
    #    density_discr = qbx.density_discr

    # boundary nodes
    bdry_nodes_x = density_discr.nodes()[0].with_queue(queue).get()
    bdry_nodes_y = density_discr.nodes()[1].with_queue(queue).get()
    bdry_nodes = make_obj_array( # get() first for CL compatibility issues
        [cl.array.to_device(queue, bdry_nodes_x),
         cl.array.to_device(queue, bdry_nodes_y)])
    print("density discr has", len(bdry_nodes_y), "nodes.")

    # boundary (unit) normals
    bdry_normals = bind(density_discr, sym.normal(dim))(queue).as_vector(dtype=object)
    bdry_normal_lens = np.sqrt(sum(vec_comp.get()**2 for vec_comp in bdry_normals))
    assert np.linalg.norm(bdry_normal_lens - 1) < 1e-12

    # gradient to normal derivative
    bdnormal_x = bdry_normals[0]
    bdnormal_y = bdry_normals[1]
    def grad_to_normal(der_x, der_y):
        return der_x * bdnormal_x + der_y * bdnormal_y

    # }}}

    print("quad points for box tree")

    # {{{ quad points for box tree

    if not adaptive_mesh:
        box_mesh = mg.MeshGen2D(vol_quad_order, n_levels)
        box_mesh.print_info()
        q_points = box_mesh.get_q_points()
        q_weights = box_mesh.get_q_weights()
        q_radii = None

    else:
        box_mesh = mg.MeshGen2D(vol_quad_order, n_levels)
        iloop = 0
        while box_mesh.n_active_cells() < refined_n_cells:
            iloop += 1
            crtr = np.array([
                np.abs(geometry_mask(c) * m)
                     for (c, m) in
                     zip(box_mesh.get_cell_centers(), box_mesh.get_cell_measures()) ])
            assert (np.max(np.abs(crtr))) > 0
            box_mesh.update_mesh(crtr, rratio_top, rratio_bot)
            if iloop > n_refinement_loops:
                print("Max number of refinement loops reached.")
                break

        box_mesh.print_info()
        q_points = box_mesh.get_q_points()
        q_weights = box_mesh.get_q_weights()
        q_radii = None

    box_mesh.generate_gmsh("box_grid.msh")

    assert (len(q_points) == len(q_weights))
    assert (q_points.shape[1] == dim)

    q_points_org = q_points
    q_points_org2 = np.ascontiguousarray(np.transpose(q_points))

    q_points = make_obj_array(
        [cl.array.to_device(queue, q_points_org2[i]) for i in range(dim)])

    # 1 inside, 0 outside
    mask_tmp = np.array([geometry_mask(qp) for qp in q_points_org])
    q_point_masks = cl.array.to_device(queue, mask_tmp)

    int_q_point_indicies = cl.array.to_device(queue, np.nonzero(mask_tmp)[0])
    ext_q_point_indicies = cl.array.to_device(queue, np.nonzero(1 - mask_tmp)[0])
    assert len(int_q_point_indicies) + len(ext_q_point_indicies) == len(q_points_org)

    ext_q_points = make_obj_array(
            [cl.array.to_device(queue, q_points_org2[i][mask_tmp==0])
                for i in range(dim)])
    for i in range(dim):
        assert len(ext_q_points[i]) == len(ext_q_point_indicies)

    q_weights = cl.array.to_device(queue, q_weights)

    # }}} End quad points for box tree

    print("box discretization")

    # {{{ box discretization

    # over the bounding box
    box_discr = PointsTarget(q_points, normals=None)

    # }}} End box discretization

    print("tree and trav")

    # {{{ tree and trav
    from boxtree.tools import AXIS_NAMES
    axis_names = AXIS_NAMES[:dim]

    from pytools import single_valued
    coord_dtype = single_valued(coord.dtype for coord in q_points)
    from boxtree.bounding_box import make_bounding_box_dtype
    bbox_type, _ = make_bounding_box_dtype(cl_ctx.devices[0], dim, coord_dtype)

    bbox = np.empty(1, bbox_type)
    for ax in axis_names:
        bbox["min_" + ax] = box_a
        bbox["max_" + ax] = box_b

    # tune max_particles_in_box to reconstruct the mesh
    from boxtree import TreeBuilder
    tb = TreeBuilder(cl_ctx)
    tree, _ = tb(
        queue,
        particles=q_points,
        targets=q_points,
        bbox=bbox,
        max_particles_in_box=vol_quad_order**2 * 4 - 1,
        kind="adaptive-level-restricted")

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(cl_ctx)
    trav, _ = tg(queue, tree)

    if 0:
        import matplotlib.pyplot as plt

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree.get(queue=queue))
        plotter.draw_tree(fill=False, edgecolor="black")
        #plotter.draw_box_numbers()
        plotter.set_bounding_box()
        plt.gca().set_aspect("equal")

        plt.draw()
        plt.savefig("ch_tree.png")
        print("Tree picture saved")

    # }}} End tree and trav

    print("The tree has", tree.nlevels, "levels")

    print("near field table")

    # {{{ near field table
    if False:
        clean_file(table_filename)
        clean_file("tmp_" + table_filename)

    tm = NearFieldInteractionTableManager(table_filename,
            root_extent=root_table_source_extent)

    import loopy as lp
    from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator
    chop = CahnHilliardOperator(b=b, c=c)

    n_brick_levels = 15
    brick_quad_order = 50

    extra_kernel_kwargs = {'lam1': chop.lambdas[0], 'lam2': chop.lambdas[1]}
    extra_kernel_kwarg_types = (lp.ValueArg("lam1", np.float64),
                               lp.ValueArg("lam2", np.float64))

    # box size should be integral times of largest table size
    # TODO: in principle the converse should also work
    assert abs(
            int((box_b-box_a)/root_table_source_extent) * root_table_source_extent
            - (box_b-box_a)) < 1e-15

    print("getting tables")
    near_field_table = {}

    # pass a debug parameter list to enable more checks
    table_debug = {"strict_loading": True}

    # {{{ cahn-hilliard kernel
    nftable = []
    for l in range(0, tree.nlevels):
        print("Getting table at level", l)
        tb, _ = tm.get_table(dim, "Cahn-Hilliard", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=b, c=c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types,
                # debug=table_debug
                )
        nftable.append(tb)
    near_field_table[nftable[0].integral_knl.__repr__()] = nftable

    nftable_dx = []
    for l in range(0, tree.nlevels):
        print("Getting table dx at level", l)
        tb, _ = tm.get_table(dim, "Cahn-Hilliard-Dx", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=b, c=c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types)
        nftable_dx.append(tb)
    near_field_table[nftable_dx[0].integral_knl.__repr__()] = nftable_dx

    nftable_dy = []
    for l in range(0, tree.nlevels):
        print("Getting table dy at level", l)
        tb, _ = tm.get_table(dim, "Cahn-Hilliard-Dy", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=b, c=c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types)
        nftable_dy.append(tb)
    near_field_table[nftable_dy[0].integral_knl.__repr__()] = nftable_dy

    if 0:
        # laplacian and its gradients
        nftable2 = []
        for l in range(0, tree.nlevels):
            print("Getting table 2 at level", l)
            tb, _ = tm.get_table(dim, "Cahn-Hilliard-Laplacian", vol_quad_order,
                    source_box_level=l, compute_method="DrosteSum",
                    adaptive_level=False,
                    n_levels=n_brick_levels,
                    n_brick_quad_points=brick_quad_order,
                    b=b, c=c,
                    extra_kernel_kwargs=extra_kernel_kwargs,
                    extra_kernel_kwarg_types=extra_kernel_kwarg_types)
            nftable2.append(tb)
        near_field_table[nftable2[0].integral_knl.__repr__()] = nftable2
        nftable2_dx = []

        for l in range(0, tree.nlevels):
            print("Getting table 2_dx at level", l)
            tb, _ = tm.get_table(dim, "Cahn-Hilliard-Laplacian-Dx", vol_quad_order,
                    source_box_level=l, compute_method="DrosteSum",
                    adaptive_level=False,
                    n_levels=n_brick_levels,
                    n_brick_quad_points=brick_quad_order,
                    b=b, c=c,
                    extra_kernel_kwargs=extra_kernel_kwargs,
                    extra_kernel_kwarg_types=extra_kernel_kwarg_types)
            nftable2_dx.append(tb)
        near_field_table[nftable2_dx[0].integral_knl.__repr__()] = nftable2_dx

        nftable2_dy = []
        for l in range(0, tree.nlevels):
            print("Getting table 2_dy at level", l)
            tb, _ = tm.get_table(dim, "Cahn-Hilliard-Laplacian-Dy", vol_quad_order,
                    source_box_level=l, compute_method="DrosteSum",
                    adaptive_level=False,
                    n_levels=n_brick_levels,
                    n_brick_quad_points=brick_quad_order,
                    b=b, c=c,
                    extra_kernel_kwargs=extra_kernel_kwargs,
                    extra_kernel_kwarg_types=extra_kernel_kwarg_types)
            nftable2_dy.append(tb)
        near_field_table[nftable2_dy[0].integral_knl.__repr__()] = nftable2_dy

    # }}} End cahn-hilliard kernel

    # {{{ yukawa kernel G1

    yukawa_extra_kernel_kwargs = {'lam': chop.lambdas[0]}
    yukawa_extra_kernel_kwarg_types = (lp.ValueArg("lam", np.float64), )

    nftable_ykw = []
    for l in range(0, tree.nlevels):
        print("Getting table yukawa at level", l)
        tb, _ = tm.get_table(dim, "Yukawa", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                lam=yukawa_extra_kernel_kwargs['lam'],
                extra_kernel_kwargs=yukawa_extra_kernel_kwargs,
                extra_kernel_kwarg_types=yukawa_extra_kernel_kwarg_types)
        nftable_ykw.append(tb)
    near_field_table[nftable_ykw[0].integral_knl.__repr__()] = nftable_ykw

    nftable_ykw_dx = []
    for l in range(0, tree.nlevels):
        print("Getting table yukawa-dx at level", l)
        tb, _ = tm.get_table(dim, "Yukawa-Dx", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                lam=yukawa_extra_kernel_kwargs['lam'],
                extra_kernel_kwargs=yukawa_extra_kernel_kwargs,
                extra_kernel_kwarg_types=yukawa_extra_kernel_kwarg_types)
        nftable_ykw_dx.append(tb)
    near_field_table[nftable_ykw_dx[0].integral_knl.__repr__()] = nftable_ykw_dx

    nftable_ykw_dy = []
    for l in range(0, tree.nlevels):
        print("Getting table yukawa-dy at level", l)
        tb, _ = tm.get_table(dim, "Yukawa-Dy", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                lam=yukawa_extra_kernel_kwargs['lam'],
                extra_kernel_kwargs=yukawa_extra_kernel_kwargs,
                extra_kernel_kwarg_types=yukawa_extra_kernel_kwarg_types)
        nftable_ykw_dy.append(tb)
    near_field_table[nftable_ykw_dy[0].integral_knl.__repr__()] = nftable_ykw_dy

    # }}} End  yukawa kernel

    # print("Using table list of length", len(nftable))
    # }}} End near field table

    print("sumpy kernel expansion")

    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    exclude_self = True
    dtype = np.complex128

    if exclude_self:
        target_to_source = np.arange(tree.ntargets, dtype=np.int32)
        self_extra_kwargs = {"target_to_source": target_to_source}
    else:
        self_extra_kwargs = {}

    # {{{ sumpy kernel expansion (cahn-hilliard)

    sumpy_knl = FactorizedBiharmonicKernel(dim)
    sumpy_knl_dx = AxisTargetDerivative(0, sumpy_knl)
    sumpy_knl_dy = AxisTargetDerivative(1, sumpy_knl)

    sumpy_knl2 = LaplacianTargetDerivative(sumpy_knl)
    sumpy_knl2_dx = AxisTargetDerivative(0, sumpy_knl2)
    sumpy_knl2_dy = AxisTargetDerivative(1, sumpy_knl2)

    ch_out_knls = [sumpy_knl, sumpy_knl_dx, sumpy_knl_dy]
            # sumpy_knl2, sumpy_knl2_dx, sumpy_knl2_dy]

    wcc = ExpansionWranglerCodeContainer(cl_ctx,
            partial(mpole_expn_class, sumpy_knl),
            partial(local_expn_class, sumpy_knl),
            ch_out_knls,
            exclude_self=exclude_self)

    wrangler = FPNDExpansionWrangler(code_container=wcc,
            queue=queue,
            tree=tree,
            near_field_table=near_field_table,
            dtype=dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
            quad_order=vol_quad_order,
            self_extra_kwargs=self_extra_kwargs,
            kernel_extra_kwargs=extra_kernel_kwargs)

    # }}} End sumpy kernel expansion

    # {{{ sumpy kernel expansion (yukawa)

    sumpy_knl_ykw = YukawaKernel(dim)
    sumpy_knl_ykw_dx = AxisTargetDerivative(0, sumpy_knl_ykw)
    sumpy_knl_ykw_dy = AxisTargetDerivative(1, sumpy_knl_ykw)

    ykw_out_knls = [sumpy_knl_ykw, sumpy_knl_ykw_dx, sumpy_knl_ykw_dy]

    wcc_ykw = ExpansionWranglerCodeContainer(cl_ctx,
            partial(mpole_expn_class, sumpy_knl_ykw),
            partial(local_expn_class, sumpy_knl_ykw),
            ykw_out_knls,
            exclude_self=exclude_self)

    wrangler_ykw = FPNDExpansionWrangler(code_container=wcc_ykw,
            queue=queue,
            tree=tree,
            near_field_table=near_field_table,
            dtype=dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
            quad_order=vol_quad_order,
            self_extra_kwargs=self_extra_kwargs,
            kernel_extra_kwargs=yukawa_extra_kernel_kwargs)

    # }}} End sumpy kernel expansion (yukawa)

    from volumential.volume_fmm import interpolate_volume_potential
    def pot_to_bdry(volume_pot):
        return interpolate_volume_potential(bdry_nodes, trav, wrangler, volume_pot
                    ).real

    # for internal debugging use
    force_direct_evaluation = False

    # {{{ time marching

    # {{{ volume potentials

    # {{{ source density

    f1_vals = cl.array.to_device(queue,
            np.array([f1_func(qp) for qp in q_points_org]))

    f2_vals = cl.array.to_device(queue,
            np.array([f2_func(qp) for qp in q_points_org]))

    print("Evaluated source density.")

    # }}} End source density

    pot = drive_volume_fmm(trav, wrangler,
            f1_vals * q_weights, f1_vals,
            direct_evaluation=force_direct_evaluation)
    # pot = np.real(pot)

    pot_ykw = drive_volume_fmm(trav, wrangler_ykw,
            f1_vals * q_weights, f1_vals,
            direct_evaluation=force_direct_evaluation)

    if 0:
        # p2p for testing
        wcc_direct = ExpansionWranglerCodeContainer(cl_ctx,
                partial(mpole_expn_class, sumpy_knl),
                partial(local_expn_class, sumpy_knl),
                [sumpy_knl],
                exclude_self=exclude_self)
        wrangler_direct = FPNDExpansionWrangler(code_container=wcc_direct,
                queue=queue,
                tree=tree,
                near_field_table=near_field_table,
                dtype=dtype,
                fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
                quad_order=vol_quad_order,
                self_extra_kwargs=self_extra_kwargs,
                kernel_extra_kwargs=extra_kernel_kwargs)
        pot_direct = drive_volume_fmm(trav, wrangler_direct,
                f1_vals * q_weights, f1_vals,
                direct_evaluation=True)
        print(pot_direct[0].real - pot[0].real)

    print("Evaluated volume potential.")

    u_tild_pot = pot[0]
    u_tild_x_pot = pot[1]
    u_tild_y_pot = pot[2]

    #pot2 = pot[3]
    #pot2_x = pot[4]
    #pot2_y = pot[5]

    pot2 = pot_ykw[0] + chop.lambdas[1]**2 * pot[0]
    pot2_x = pot_ykw[1] + chop.lambdas[1]**2 * pot[1]
    pot2_y = pot_ykw[2] + chop.lambdas[1]**2 * pot[2]

    # i.e., f2_vals - pot_ykw[0] + chop.lambdas[0]**2 * pot[0]
    v_tild_pot = f2_vals - pot2 + b * u_tild_pot

    # }}} End volume potentials (u and Lap_u)

    # {{{ prepare rhs bor BIE

    print("Preparing BC")

    bdry_u_tild = pot_to_bdry(u_tild_pot)
    bdry_u_tild_x = pot_to_bdry(u_tild_x_pot)
    bdry_u_tild_y = pot_to_bdry(u_tild_y_pot)

    bdry_v_tild = pot_to_bdry(v_tild_pot)
    bdry_pot2 = pot_to_bdry(pot2)

    # pot2 is short for laplace(u_tild)
    bdry_pot2_x = pot_to_bdry(pot2_x)
    bdry_pot2_y = pot_to_bdry(pot2_y)

    bdry_f2_x_vals = cl.array.to_device(queue,
            np.array([f2_x_func([x0, x1]) for (x0, x1) in zip(
                bdry_nodes_x, bdry_nodes_y)]))
    bdry_f2_y_vals = cl.array.to_device(queue,
            np.array([f2_y_func([x0, x1]) for (x0, x1) in zip(
                bdry_nodes_x, bdry_nodes_y)]))

    # numerically determin b1 values
    bdry_exact_phi_vals = cl.array.to_device(queue,
            np.array([phi_func([x0, x1]) for (x0, x1) in zip(
                bdry_nodes_x, bdry_nodes_y)]))
    bdry_exact_phi_x_vals = cl.array.to_device(queue,
            np.array([phi_x_func([x0, x1]) for (x0, x1) in zip(
                bdry_nodes_x, bdry_nodes_y)]))
    bdry_exact_phi_y_vals = cl.array.to_device(queue,
            np.array([phi_y_func([x0, x1]) for (x0, x1) in zip(
                bdry_nodes_x, bdry_nodes_y)]))

    bdry_b1_vals = grad_to_normal(bdry_exact_phi_x_vals, bdry_exact_phi_y_vals
            ) + c * bdry_exact_phi_vals

    bdry_u_tild_normal = grad_to_normal(bdry_u_tild_x, bdry_u_tild_y)
    bdry_f2_normal = grad_to_normal(bdry_f2_x_vals, bdry_f2_y_vals)
    bdry_pot2_normal = grad_to_normal(bdry_pot2_x, bdry_pot2_y)

    bdry_v_tild_normal = bdry_f2_normal - bdry_pot2_normal + b * bdry_u_tild_normal

    bdry_g1_vals = bdry_b1_vals - bdry_u_tild_normal - (c - bs) * bdry_u_tild
    bdry_g2_vals = -bdry_v_tild_normal

    # }}} End prepare rhs bor BIE

    # {{{ BIE solve

    from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator
    chop = CahnHilliardOperator(b=b, c=c)

    unk = chop.make_unknown("sigma")
    bound_op = bind(qbx, chop.operator(unk, bs))

    bc = sym.make_obj_array([
        bdry_g1_vals,
        bdry_g2_vals
    ])

    if 0:
        print("reconstructing matrix")
        # construct the coef matrix
        mop = bound_op.scipy_op(queue, "sigma", dtype=np.complex128)
        mm, mn = mop.shape
        assert mm == mn
        mcoef = np.zeros(mop.shape, dtype=np.complex128)

        for i in range(mm):
            print(i, "/", mm)
            ei = np.zeros(mm)
            ei[i] = 1
            coli = mop.matvec(ei)
            mcoef[:,i] = coli

        np.save("coef_matrix.npy", mcoef)

    from pytential.solve import gmres
    gmres_result = gmres(
        bound_op.scipy_op(queue, "sigma", dtype=np.complex128),
        bc,
        tol=1e-14,
        progress=True,
        stall_iterations=0,
        hard_failure=True)

    assert gmres_result.success

    sigma = gmres_result.solution

    # }}} End BIE solve

    # {{{ put pieces together (bootstrap for the next step)

    print("assembling the solution")

    qbx_stick_out = qbx.copy(target_association_tolerance=0.05)

    bvp_sol = bind(
            (qbx_stick_out, box_discr),
            chop.representation(unk))(queue, sigma=sigma)

    # interior side
    bdry_bvp_sol = bind(
            (qbx_stick_out, density_discr),
            chop.representation(unk, qbx_forced_limit=-1))(queue, sigma=sigma)

    # representaion:
    # u, v, u_x, u_y, lap(u), v, v_x, v_y

    solu_phi = u_tild_pot + bvp_sol[0]
    solu_mu = epsilon * (v_tild_pot + bvp_sol[1])
    solu_phi_x = u_tild_x_pot + bvp_sol[2]
    solu_phi_y = u_tild_y_pot + bvp_sol[3]
    solu_lap_phi = pot2 + bvp_sol[4]

    # For exterior part of layer potentials, do a continuous extension
    # (some layer potentials have jumps if used directly)

    bdry_solu_phi = bdry_u_tild + bdry_bvp_sol[0]
    bdry_solu_mu = epsilon * (bdry_v_tild + bdry_bvp_sol[1])
    bdry_solu_phi_x = bdry_u_tild_x + bdry_bvp_sol[2]
    bdry_solu_phi_y = bdry_u_tild_y + bdry_bvp_sol[3]
    bdry_solu_lap_phi = bdry_pot2 + bdry_bvp_sol[4]

    # }}} End put pieces together

    import pudb; pu.db
    1/0

    # {{{ postprocess
    print("postprocessing")

    if 0:
        # usual output on the unstructured mesh
        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(queue, vol_discr, visual_order)

        nodes_x = vol_discr.nodes()[0].with_queue(queue).get()
        nodes_y = vol_discr.nodes()[1].with_queue(queue).get()
        nodes = make_obj_array(
                [cl.array.to_device(queue, nodes_x),
                    cl.array.to_device(queue, nodes_y)])

        phi = interpolate_volume_potential(nodes,
                trav, wrangler, new_phi).real
        mu = interpolate_volume_potential(nodes,
                trav, wrangler, new_mu).real
        f1 = interpolate_volume_potential(nodes,
                trav, wrangler, f1_vals).real
        f2 = interpolate_volume_potential(nodes,
                trav, wrangler, f2_vals).real

        vtu_filename = "ch-solution-" + "{0:03}".format(i_iter+1) + ".vtu"
        clean_file(vtu_filename)

        vis.write_vtk_file(vtu_filename, [
            ("x", vol_discr.nodes()[0]),
            ("y", vol_discr.nodes()[1]),
            ("f1", f1),
            ("f2", f2),
            ("phi", phi),
            ("mu", mu)
            ])

        print("Written file " + vtu_filename)

    if 0:
        # layer potentials (to see the jump conditions)
        import os
        os.system('gmsh box_grid.msh convert_grid -')
        from meshmode.mesh.io import read_gmsh
        modemesh = read_gmsh("box_grid.msh", force_ambient_dim=None)
        from meshmode.discretization.poly_element import \
                LegendreGaussLobattoTensorProductGroupFactory
        immersed_discr = Discretization(
                cl_ctx, modemesh,
                LegendreGaussLobattoTensorProductGroupFactory(
                    vol_quad_order))

        one_sigma = make_obj_array([cl.array.to_device(queue, np.ones(len(sig)))
                    for sig in sigma])

        print(one_sigma)

        layer_potentials = bind(
                (qbx_stick_out, immersed_discr),
                chop.debug_representation(unk))(queue, sigma=one_sigma)
        #debug_rep_names = ["S0", "S0_n", "S1", "S1_n", "S2", "S2_n"]
        debug_rep_names = ["S0", "S1", "S2"]

        vtu_filename = "ch-layer-potential.vtu"
        clean_file(vtu_filename)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(queue, immersed_discr, visual_order)

        vis.write_vtk_file(vtu_filename, [
            ("x", immersed_discr.nodes()[0]),
            ("y", immersed_discr.nodes()[1])
            ] + [
                (debug_rep_names[i], layer_potentials[i])
                for i in range(len(layer_potentials))
                ])
        print("Written file " + vtu_filename)

    if 1:
        # immersed visualization (whole box)

        import os
        os.system('gmsh box_grid.msh convert_grid -')
        from meshmode.mesh.io import read_gmsh
        modemesh = read_gmsh("box_grid.msh", force_ambient_dim=None)
        from meshmode.discretization.poly_element import \
                LegendreGaussLobattoTensorProductGroupFactory
        immersed_discr = Discretization(
                cl_ctx, modemesh,
                LegendreGaussLobattoTensorProductGroupFactory(
                    vol_quad_order))

        immersed_nodes_x = immersed_discr.nodes()[0].with_queue(queue).get()
        immersed_nodes_y = immersed_discr.nodes()[1].with_queue(queue).get()
        immersed_nodes = make_obj_array(
                # get() first for CL compatibility issues
                [cl.array.to_device(queue, immersed_nodes_x),
                    cl.array.to_device(queue, immersed_nodes_y)])

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(queue, immersed_discr, visual_order)

        imm_phi = interpolate_volume_potential(immersed_nodes, trav,
                wrangler, new_phi).real
        imm_mu = interpolate_volume_potential(immersed_nodes, trav,
                wrangler, new_mu).real
        imm_f1 = interpolate_volume_potential(immersed_nodes, trav,
                wrangler, f1_vals).real
        imm_f2 = interpolate_volume_potential(immersed_nodes, trav,
                wrangler, f2_vals).real

        imm_bvp_solu = bind(
                (qbx_stick_out, immersed_discr),
                chop.representation(unk))(queue, sigma=sigma)
        imm_bvp_names = ["u", "v", "u_x", "u_y", "lap_u", "v2", "v_x", "v_y"]

        imm_pot = []
        for pp in pot:
            imm_pot.append(
                    interpolate_volume_potential(immersed_nodes, trav,
                        wrangler, pp).real
                    )
        imm_pot_names = ["ut", "ut_x", "ut_y", "lap_ut", "(lap_ut)_x", "(lap_ut)_y"]

        imm_pot_ykw = []
        for pp in pot_ykw:
            imm_pot_ykw.append(
                    interpolate_volume_potential(immersed_nodes, trav,
                        wrangler_ykw, pp).real)
        imm_pot_ykw_names = ["V1[f1]", "V1[f1]_x", "V1[f1]_y"]

        vtu_filename = "ch-solution-immersed-" + "{0:03}".format(i_iter+1) + ".vtu"
        clean_file(vtu_filename)

        vis.write_vtk_file(vtu_filename, [
            ("x", immersed_discr.nodes()[0]),
            ("y", immersed_discr.nodes()[1]),
            ("f1", imm_f1),
            ("f2", imm_f2),
            ("phi", imm_phi),
            ("mu", imm_mu)
            ] + [
                ("BVPSolu-" + imm_bvp_names[i], imm_bvp_solu[i].real) for i in range(len(imm_bvp_solu))
                ] + [
                    ("VolumePot-"+imm_pot_names[i], imm_pot[i]) for i in range(len(imm_pot))
                    ] + [
                        ("VolumePot-"+imm_pot_ykw_names[i], imm_pot_ykw[i])
                        for i in range(len(imm_pot_ykw))
                        ])
        print("Written file " + vtu_filename)

    if 0:
        # boundary only
        from meshmode.discretization.visualization import make_visualizer
        bdry_vis = make_visualizer(queue, density_discr, visual_order)

        bdry_normals = bind(density_discr, sym.normal(dim))(queue)\
                .as_vector(dtype=object)

        bdry_bvp_solu = bind(
                (qbx_stick_out, density_discr),
                chop.representation(unk, qbx_forced_limit=-1))(queue, sigma=sigma)
        bdry_bvp_names = ["u", "v", "u_x", "u_y", "lap_u", "v2", "v_x", "v_y"]

        vtu_filename = "ch-bdry-" + "{0:03}".format(i_iter+1) + ".vtu"
        clean_file(vtu_filename)

        bdry_vis.write_vtk_file(vtu_filename, [
            ("sigma[0]", sigma[0].real),
            ("sigma[1]", sigma[1].real),
            ("bdry_normals", bdry_normals),
            ("b1", bdry_b1_vals),
            ("u_tild_normal", bdry_u_tild_normal),
            ("v_tild_normal", bdry_v_tild_normal),
            ("f2_normal", bdry_f2_normal),
            ("lap_u_tild_normal", bdry_pot2_normal),
            ("u_tild", bdry_u_tild),
            ("g1", bdry_g1_vals),
            ("g2", bdry_g2_vals),
            ] + [
                ("BVPSolu-" + bdry_bvp_names[i], bdry_bvp_solu[i].real) for i in range(len(bdry_bvp_solu))
                ])
        print("Written file " + vtu_filename)

    # }}} End postprocess

    # }}} End time marching


if __name__ == "__main__":
    main()

# vim: fdm=marker:ft=pyopencl
