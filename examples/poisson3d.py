from __future__ import absolute_import, division, print_function

__copyright__ = "Copyright (C) 2018 Xiaoyu Wei"

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
import pymbolic as pmbl

from functools import partial

from meshmode.mesh.processing import affine_map
from meshmode.discretization.visualization import make_visualizer

from pytential import bind, sym, norm # noqa
from pytential.symbolic.pde.scalar import DirichletOperator
from sumpy.kernel import LaplaceKernel
from volumential.volume_fmm import interpolate_volume_potential
from volumential.tools import ScalarFieldExpressionEvaluation as Eval
from volumential.tools import clean_file

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("")
print("*************************")
print("* Setting up...")
print("*************************")

# {{{ configs

dim = 3
dtype = np.float64
nftable_datafile = "nft_laplace3d_1_to_9.hdf5"
physical_domain = "dice"
# physical_domain = "betterplane"
verbose = False

if verbose:
        logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

loc_sign = -1 # +1 for exterior, -1 for interior

q_order = 3         # (volumetric) quadrature order
m_order = 10         # multipole order

mesh_order = 2      # meshmode's volume mesh order, only useful for dice

bdry_quad_order = 2 # meshmode's boundary discr order
bdry_ovsmp_quad_order = 4 # qbx's fine_order
qbx_order = 10             # qbx order
# Converging: 10, 12
# Not converging: 15, 20
qbx_fmm_order = 10        # qbx's fmm_order

if physical_domain == "dice":
    # bounding box
    a = -0.3
    b = 0.3
    # boundary mesh control
    h = 0.02
else:
    a = None
    b = None
    h = 0.2

# volume (adaptive) mesh control
n_levels = 8  # 2^(n_levels-1) subintervals in 1D, must be at least 2 if not adaptive.
              # If adaptive mesh, this is the initial number of levels before further
              # adaptive refinements
adaptive_mesh = False
n_refinement_loops = 100
refined_n_cells = 50000
rratio_top = 0.1
rratio_bot = 0.5

def refinement_flag(f, u=0):
    '''
    Input source field value f and solution value u (if available),
    output non-negative flags used for mesh refinement.
    '''
    return np.abs(f)

visual_order = 1
write_bdry_vtu = True
write_vol_vtu = True
write_box_vtu = False

# }}} End configs

# {{{ analytic solution

x = pmbl.var("x")
y = pmbl.var("y")
z = pmbl.var("z")

norm2 = x**2 + y**2 + z**2
source_expr = -1
solu_expr = 1/6 * norm2
logger.info("Source expr: " + str(source_expr))
logger.info("Solu expr: " + str(solu_expr))

source_eval = Eval(dim, source_expr, [x, y, z])
solu_eval = Eval(dim, solu_expr, [x, y, z])

# }}} End analytic solution

print("Multipole order =", m_order)

# {{{ make boundary mesh

vol_quad_order = q_order

from meshmode.mesh.io import generate_gmsh, FileSource  # noqa
from meshmode.mesh.io import read_gmsh

disp = np.zeros(dim)

import os
mesh_file = "./gmsh_tmp/mesh.msh"
if os.path.isfile(mesh_file):
    logger.info("Reading mesh file mesh.msh")
    physical_mesh = read_gmsh(mesh_file, force_ambient_dim=dim)

else:

    if physical_domain == "dice":
        physical_mesh = generate_gmsh(
                FileSource("dice.step"), dim, order=mesh_order,
                force_ambient_dim=dim,
                other_options=[
                    "-string", "Mesh.CharacteristicLengthMax = %g;" % h +
                    "Mesh.CharacteristicLengthFromCurvature = %g;" % 1
                    ],
                output_file_name='mesh.msh',
                keep_tmp_dir=True
                )

    elif physical_domain == "betterplane":
        from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
        tmp_physical_mesh = generate_gmsh(
                ScriptWithFilesSource("""
                    Merge "betterplane.brep";

                    Mesh.CharacteristicLengthMax = %(lcmax)f;
                    Mesh.ElementOrder = 2;
                    Mesh.CharacteristicLengthExtendFromBoundary = 0;

                    // 2D mesh optimization
                    // Mesh.Lloyd = 1;

                    l_superfine() = Unique(Abs(Boundary{ Surface{
                        27, 25, 17, 13, 18  }; }));
                    l_fine() = Unique(Abs(Boundary{ Surface{ 2, 6, 7}; }));
                    l_coarse() = Unique(Abs(Boundary{ Surface{ 14, 16  }; }));

                    // p() = Unique(Abs(Boundary{ Line{l_fine()}; }));
                    // Characteristic Length{p()} = 0.05;

                    Field[1] = Attractor;
                    Field[1].NNodesByEdge = 100;
                    Field[1].EdgesList = {l_superfine()};

                    Field[2] = Threshold;
                    Field[2].IField = 1;
                    Field[2].LcMin = 0.075;
                    Field[2].LcMax = %(lcmax)f;
                    Field[2].DistMin = 0.1;
                    Field[2].DistMax = 0.4;

                    Field[3] = Attractor;
                    Field[3].NNodesByEdge = 100;
                    Field[3].EdgesList = {l_fine()};

                    Field[4] = Threshold;
                    Field[4].IField = 3;
                    Field[4].LcMin = 0.1;
                    Field[4].LcMax = %(lcmax)f;
                    Field[4].DistMin = 0.15;
                    Field[4].DistMax = 0.4;

                    Field[5] = Attractor;
                    Field[5].NNodesByEdge = 100;
                    Field[5].EdgesList = {l_coarse()};

                    Field[6] = Threshold;
                    Field[6].IField = 5;
                    Field[6].LcMin = 0.15;
                    Field[6].LcMax = %(lcmax)f;
                    Field[6].DistMin = 0.2;
                    Field[6].DistMax = 0.4;

                    Field[7] = Min;
                    Field[7].FieldsList = {2, 4, 6};

                    Background Field = 7;
                    """ % {
                        "lcmax": h,
                        }, ["betterplane.brep"]), 2,
                    output_file_name='mesh.msh',
                    keep_tmp_dir=True
                    )

        # Flip elements--gmsh generates inside-out geometry.
        from meshmode.mesh.processing import perform_flips
        physical_mesh = perform_flips(tmp_physical_mesh, np.ones(tmp_physical_mesh.nelements))

        # Translate vertices to make the mesh centered at origin
        for iaxis in range(dim):
            ai = np.min(physical_mesh.vertices[iaxis])
            bi = np.max(physical_mesh.vertices[iaxis])
            ci = (ai + bi) * 0.5
            disp[iaxis] = -ci
        #    assert len(physical_mesh.groups) == 1
        #    physical_mesh.vertices[iaxis] -= ci
        #    physical_mesh.groups[0].nodes[iaxis] -=ci
            print(iaxis, ":", ai, bi, ci)
        physical_mesh = affine_map(physical_mesh, None, disp)

    else:
        raise NotImplementedError

logger.info("%d boundary elements" % physical_mesh.nelements)

# debug output
if 0:
    from meshmode.mesh.visualization import write_vertex_vtk_file

    write_vertex_vtk_file(physical_mesh, "physical_mesh.vtu")

# infer bbox
if a is None or b is None:
    a = np.min(physical_mesh.vertices) * 1.01
    b = np.max(physical_mesh.vertices) * 1.01

if physical_domain == "dice":
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
else:
    # in this case, vol_discr is a test region used for visualization
    # and there is no needed connection between vol_discr and bdry_discr
    import meshmode.mesh.generation
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import  \
    InterpolatoryQuadratureSimplexGroupFactory
    if 0:
        test_mesh = meshmode.mesh.generation.generate_regular_rect_mesh(
                (a, a, a), (b, b, b), (50, 50, 50), order=1)

    vol_physical_mesh = generate_gmsh(
            FileSource("betterplane.brep"), dim, order=1,
            force_ambient_dim=dim,
            other_options=[
                "-string", "Mesh.CharacteristicLengthMax = %g;" % h +
                "Mesh.CharacteristicLengthFromCurvature = %g;" % 1
                ],
            output_file_name='vol_mesh.msh',
            keep_tmp_dir=True
            )

    logger.info("shifting volume mesh")
    print(disp)
    vol_physical_mesh = affine_map(vol_physical_mesh, None, disp)

    if 0:
        # shrik the plane a little to make better visuals
        print("shrinking volume mesh")
        A = np.eye(dim) * 0.95
        print(A)
        vol_physical_mesh = affine_map(vol_physical_mesh, A, None)

    vol_discr = Discretization(ctx, vol_physical_mesh,
            InterpolatoryQuadratureSimplexGroupFactory(vol_quad_order))
    bdry_discr = Discretization(ctx, physical_mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

logger.info("%d boundary node" % bdry_discr.nnodes)

# ensure bounding box properties
print("Boundary nodes are in range:",
        np.min(physical_mesh.vertices),
        np.max(physical_mesh.vertices))

if a is None or b is None:
    # Auto choose bbox from mesh
    pass
else:
    assert( a < np.min(physical_mesh.vertices) )
    assert( b > np.max(physical_mesh.vertices) )

# }}} End make boundary mesh

# {{{ generate quad points

import volumential.meshgen as mg

mesh = mg.MeshGen3D(q_order, n_levels, a, b)
if not adaptive_mesh:
    mesh.print_info()
    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    q_radii = None

else:

    iloop = 0
    while mesh.n_active_cells() < refined_n_cells:
        iloop += 1
        crtr = np.array([
            np.abs(refinement_flag(source_field(c[0], c[1], c[2])) * m)
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

if write_box_vtu:
    mesh.generate_gmsh("grid.msh")

assert (len(q_points) == len(q_weights))
assert (q_points.shape[1] == dim)

q_points_org = q_points
q_points = np.ascontiguousarray(np.transpose(q_points))

from pytools.obj_array import make_obj_array
q_points = make_obj_array(
    [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

q_weights = cl.array.to_device(queue, q_weights)

# }}}

# {{{ build tree and traversals

from boxtree.tools import AXIS_NAMES
axis_names = AXIS_NAMES[:dim]

from pytools import single_valued
coord_dtype = single_valued(coord.dtype for coord in q_points)
from boxtree.bounding_box import make_bounding_box_dtype
bbox_type, _ = make_bounding_box_dtype(ctx.devices[0], dim, coord_dtype)

bbox = np.empty(1, bbox_type)
if a is None or b is None:
    for ax, iax in zip(axis_names, range(dim)):
        bbox["min_" + ax] = a
        bbox["max_" + ax] = b
else:
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
    max_particles_in_box=q_order**dim * (2**dim) - 1,
    kind="adaptive-level-restricted")

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

# }}} End build tree and traversals

logger.info("discretizing source field")
source_vals = cl.array.to_device(queue,
        source_eval(queue, np.array(
            [coords.get() for coords in q_points])))

# {{{ build near field potential table

force_recompute = False
from volumential.table_manager import NearFieldInteractionTableManager
tm = NearFieldInteractionTableManager(nftable_datafile, root_extent=2)
nftable, _ = tm.get_table(dim, "Laplace", q_order,
        force_recompute=force_recompute,
        compute_method="DrosteSum", queue=queue,
        n_brick_quad_points=120,
        adaptive_level=False,
        use_symmetry=True,
        alpha=0, n_levels=1)

# }}} End build near field potential table

# {{{ sumpy expansion for laplace kernel

knl = LaplaceKernel(dim)
out_kernels = [knl]

if 1:
    from sumpy.expansion.multipole import LaplaceConformingVolumeTaylorMultipoleExpansion
    from sumpy.expansion.local import LaplaceConformingVolumeTaylorLocalExpansion
    local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion

else:
    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    from sumpy.expansion.local import VolumeTaylorLocalExpansion
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion

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

from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler
wrangler = FPNDExpansionWrangler(
    code_container=wcc,
    queue=queue,
    tree=tree,
    near_field_table=nftable,
    dtype=dtype,
    fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
    quad_order=q_order,
    self_extra_kwargs=self_extra_kwargs)

# }}} End sumpy expansion for laplace kernel

print("Bounding box: [", a, b, "]^2")
print("Number of boundary nodes: ", bdry_discr.nnodes)

# {{{ make box discr

if write_box_vtu:
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

print("")
print("*************************")
print("* Evaluating: ")
print("* Volume Potential... ")
print("*************************")

from volumential.volume_fmm import drive_volume_fmm
pot, = drive_volume_fmm(trav, wrangler, source_vals * q_weights, source_vals,
        direct_evaluation=False)

# interpolate solution
nodes_x = vol_discr.nodes()[0].with_queue(queue).get()
nodes_y = vol_discr.nodes()[1].with_queue(queue).get()
nodes_z = vol_discr.nodes()[2].with_queue(queue).get()
nodes = make_obj_array( # get() first for CL compatibility issues
        [cl.array.to_device(queue, nodes_x),
         cl.array.to_device(queue, nodes_y),
         cl.array.to_device(queue, nodes_z)])
vol_pot = interpolate_volume_potential(nodes, trav, wrangler, pot)

print("")
print("*************************")
print("* Evaluating: ")
print("* Boundary Conditions... ")
print("*************************")

from pytential.qbx import (
        QBXLayerPotentialSource, QBXTargetAssociationFailedException)
# TODO: use fmmlib as backend
qbx, _ = QBXLayerPotentialSource(
        bdry_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
        _from_sep_smaller_min_nsources_cumul=30,
        fmm_order=qbx_fmm_order,
        ).with_refinement()

bdry_discr = qbx.density_discr

bdry_nodes_x = bdry_discr.nodes()[0].with_queue(queue).get()
bdry_nodes_y = bdry_discr.nodes()[1].with_queue(queue).get()
bdry_nodes_z = bdry_discr.nodes()[2].with_queue(queue).get()
bdry_nodes = make_obj_array( # get() first for CL compatibility issues
        [cl.array.to_device(queue, bdry_nodes_x),
         cl.array.to_device(queue, bdry_nodes_y),
         cl.array.to_device(queue, bdry_nodes_z)])

bdry_pot = interpolate_volume_potential(bdry_nodes, trav, wrangler, pot)
assert(len(bdry_pot) == bdry_discr.nnodes)

bdry_condition = solu_eval(queue, np.array([
    bdry_nodes_x, bdry_nodes_y, bdry_nodes_z
    ]))
bdry_condition = cl.array.to_device(queue, bdry_condition)

bdry_vals = bdry_condition - bdry_pot

print("")
print("*************************")
print("* Solving BVP... ")
print("*************************")

op = DirichletOperator(LaplaceKernel(dim), loc_sign, use_l2_weighting=False)
sym_sigma = sym.var("sigma")
op_sigma = op.operator(sym_sigma)

if 0:
    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(queue, bdry_discr, visual_order)

    bdry_normals = bind(bdry_discr, sym.normal(dim))(queue).as_vector(dtype=object)

    clean_file("boudnary_normals.vtu")
    bdry_vis.write_vtk_file("boundary_nornmals.vtu", [
        ("bdry_normals", bdry_normals),
        ])

qbx_stick_out = qbx.copy(target_stick_out_factor=0.01)
bound_op = bind(qbx_stick_out, op_sigma)
bvp_rhs = bind(bdry_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bdry_vals)

from pytential.solve import gmres
gmres_result = gmres(
        bound_op.scipy_op(queue, "sigma", dtype=np.float64),
        bvp_rhs, tol=1e-14, progress=True,
        hard_failure=False)

sigma = bind(qbx_stick_out, sym_sigma)(queue, sigma=gmres_result.solution)
print("gmres state:", gmres_result.state)

print("")
print("*************************")
print("* Finalizing Solution... ")
print("*************************")

"""
qbx_forced_limit –

+1 if the output is required to originate from a QBX center on the “+” side
of the boundary. -1 for the other side. Evaluation at a target with a value of
+/- 1 in qbx_forced_limit will fail if no QBX center is found.

+2 may be used to allow evaluation QBX center on the “+” side of the (but
disallow evaluation using a center on the “-” side). Potential evaluation at the
target still succeeds if no applicable QBX center is found. (-2 for the analogous
behavior on the “-” side.)

None may be used to avoid expressing a side preference for close evaluation.

'avg' may be used as a shorthand to evaluate this potential as an average of the
+1 and the -1 value.
"""
from sumpy.visualization import FieldPlotter
fplot = FieldPlotter(np.zeros(dim), extent=20, npoints=50)
vol_qbx_stick_out = qbx.copy(target_stick_out_factor=0.1)
try:
    bvp_sol = bind(
            (vol_qbx_stick_out, vol_discr),
            op.representation(sym_sigma, qbx_forced_limit=None))(queue, sigma=sigma)
except QBXTargetAssociationFailedException as e:
    clean_file("poisson3d-failed-targets.vts")
    logger.warn("Failed targets are present.")
    bvp_sol = 0.
    vis = make_visualizer(queue, vol_discr, 1)
    vis.write_vtk_file(
            "poisson3d-failed-targets.vts",
            [
                ("failed", e.failed_target_flags)
                ]
            )

solu = bvp_sol + vol_pot

bdry_qbx_stick_out = qbx.copy(target_stick_out_factor=0.01)
bdry_bvp_sol = bind(
        (bdry_qbx_stick_out, bdry_discr),
        op.representation(sym_sigma, qbx_forced_limit=-1))(queue, sigma=sigma)

bdry_bvp_sol2 = bind(
        (bdry_qbx_stick_out, bdry_discr),
        op.representation(sym_sigma, qbx_forced_limit=+1))(queue, sigma=sigma)

bdry_bvp_sol_avg = bind(
        (bdry_qbx_stick_out, bdry_discr),
        op.representation(sym_sigma, qbx_forced_limit='avg'))(queue, sigma=sigma)

print("")
print("*************************")
print("* Postprocessing... ")
print("*************************")

if 0:
    test_nodes_x = np.array([0.0])
    test_nodes_y = np.array([0.0])
    test_nodes_z = np.array([0.0])
    test_nodes = make_obj_array([
        cl.array.to_device(queue, test_nodes_x),
        cl.array.to_device(queue, test_nodes_y),
        cl.array.to_device(queue, test_nodes_z)])
    from pytential.target import PointsTarget
    test_discr = PointsTarget(test_nodes, normals=None)

    poisson_true_sol = solu_eval(queue, np.array([
                test_nodes_x, test_nodes_y, test_nodes_z]))
    test_solu = (interpolate_volume_potential(test_nodes, trav, wrangler, pot).get()
            + bind((qbx, test_discr), op.representation(
                sym_sigma, qbx_forced_limit=-2))(queue, sigma=sigma).get())

    poisson_err = test_solu - poisson_true_sol

    #rel_err = (
    #        norm(vol_discr, queue, poisson_err)
    #        /
    #        norm(vol_discr, queue, poisson_true_sol))
    rel_err = np.linalg.norm(poisson_err)
    print("rel err: %g" % rel_err)

print("Writing vtu..")

# {{{ write vtu

if write_vol_vtu:
    vis = make_visualizer(queue, vol_discr, visual_order)
    clean_file("poisson-3d.vtu")
    vis.write_vtk_file("poisson-3d.vtu", [
        ("x", vol_discr.nodes()[0]),
        ("y", vol_discr.nodes()[1]),
        ("z", vol_discr.nodes()[2]),
        ("bvp_sol", bvp_sol),
        ("vol_pot", vol_pot),
        ("solu", solu)
        ])
else:
    pass

if write_bdry_vtu:
    bdry_normals = bind(bdry_discr, sym.normal(dim))(queue).as_vector(dtype=object)
    bdry_vis = make_visualizer(queue, bdry_discr, visual_order)
    clean_file("poisson-3d-bdry.vtu")
    bdry_vis.write_vtk_file("poisson-3d-bdry.vtu", [
        ("bdry_normals", bdry_normals),
        ("bdry_bvp_sol", bdry_bvp_sol),
        ("bdry_bvp_sol_+1", bdry_bvp_sol2),
        ("bdry_bvp_sol_avg", bdry_bvp_sol_avg),
        ("density_sigma", sigma),
        ("BC_vals", bdry_condition),
        ("VP_vals", bdry_pot),
        ("BC-VP", bdry_vals),
        ])
else:
    pass

if write_box_vtu:
    raise NotImplementedError
else:
    pass

# }}} End write vtu
