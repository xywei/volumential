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

dim = 2

import numpy as np
import sympy as sp
from pymbolic import var
import pyopencl as cl
import loopy as lp
import pyopencl.clmath
from pytential import bind, sym, norm
from pytential.solve import gmres
from pytential.symbolic.stokes import StressletWrapper, StokesletWrapper
from pytential.target import PointsTarget

from pytools.obj_array import make_obj_array
from volumential.tools import clean_file
from volumential.volume_fmm import drive_volume_fmm
from volumential.volume_fmm import interpolate_volume_potential
from sumpy.kernel import FactorizedBiharmonicKernel, YukawaKernel

import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from functools import partial
import pymbolic
import pymbolic as pmbl

from enum import Enum
class Geometry(Enum):
    RegularRectangle = 1
    Circle = 2

# {{{ Goursat kernels

from sumpy.kernel import ExpressionKernel, AxisTargetDerivative
from sumpy.kernel import KernelArgument

class ComplexLogKernel(ExpressionKernel):

    init_arg_names = ("dim",)

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            conj_z = d[0] - var("I") * d[1]
            r = var("sqrt")(np.dot(conj_z, z))
            expr = var("log")(r)
            scaling = 1/(4*var("pi"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super(ComplexLogKernel, self).__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                is_complex_valued=True)

    has_efficient_scale_adjustment = True

    def adjust_for_kernel_scaling(self, expr, rscale, nderivatives):
        if self.dim == 2:
            if nderivatives == 0:
                import sumpy.symbolic as sp
                return (expr + sp.log(rscale))
            else:
                return expr

        else:
            raise NotImplementedError("unsupported dimensionality")

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxLogKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


class ComplexLinearLogKernel(ExpressionKernel):

    init_arg_names = ("dim",)
    has_efficient_scale_adjustment = False

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            conj_z = d[0] - var("I") * d[1]
            r = var("sqrt")(np.dot(conj_z, z))
            expr = conj_z * var("log")(r)
            scaling = - 1/(4*var("pi"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super(ComplexLinearLogKernel, self).__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                is_complex_valued=True)

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxLinLogKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


class ComplexLinearKernel(ExpressionKernel):

    init_arg_names = ("dim",)
    has_efficient_scale_adjustment = False

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            expr = z
            scaling = - 1/(8*var("pi"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super(ComplexLinearKernel, self).__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                is_complex_valued=True)

    has_efficient_scale_adjustment = True

    def adjust_for_kernel_scaling(self, expr, rscale, nderivatives):
        if self.dim == 2:
            return (rscale * expr)

        else:
            raise NotImplementedError("unsupported dimensionality")

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxLinKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


class ComplexFractionalKernel(ExpressionKernel):

    init_arg_names = ("dim",)
    has_efficient_scale_adjustment = False

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            conj_z = d[0] - var("I") * d[1]
            expr = conj_z / z
            scaling = 1/(4*var("pi")*var("I"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super(ComplexFractionalKernel, self).__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                is_complex_valued=True)

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxFracKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


# }}} End Goursat kernels

# {{{ function details

def math_func_mangler(target, name, arg_dtypes):
    """Magic function that is necessary for evaluating initials
    """
    if len(arg_dtypes) == 1 and isinstance(name, pymbolic.primitives.Lookup):
        arg_dtype, = arg_dtypes

        fname = name.name
        if not (isinstance(name.aggregate, pymbolic.primitives.Variable)
                and name.aggregate.name == 'math'):
            raise RuntimeError("unexpected aggregate '%s'" %
                    str(name.aggregate))

        if arg_dtype.is_complex():
            if arg_dtype.numpy_dtype == np.complex64:
                tpname = "cfloat"
            elif arg_dtype.numpy_dtype == np.complex128:
                tpname = "cdouble"
            else:
                raise RuntimeError("unexpected complex type '%s'" %
                        arg_dtype)

            return lp.CallMangleInfo(
                   target_name="%s_%s" % (tpname, fname),
                   result_dtypes=(arg_dtype,),
                   arg_dtypes=(arg_dtype,))

        else:
            return lp.CallMangleInfo(
                   target_name="%s" % fname,
                   result_dtypes=(arg_dtype,),
                   arg_dtypes=(arg_dtype,))

    return None

def setup_cl_ctx(ctx=None, queue=None):
    if ctx is None:
        if queue is None:
            return cl.create_some_context()
        else:
            return queue.context
    else:
        return ctx

def setup_command_queue(ctx=None, queue=None):
    if queue is None:
        cl_ctx = setup_cl_ctx(ctx, queue)
        return cl.CommandQueue(cl_ctx)
    else:
        return queue

def get_output_filename(prefix, serial, extension=".vtu"):
    if serial is None:
        return prefix + extension
    else:
        return prefix + "{0:03}".format(serial) + extension

def write_visualization_file(visualizer, filename, field_names,
        field_data_contents):
    clen = min(len(field_names), len(field_data_contents))
    clean_file(filename)
    visualizer.write_vtk_file(filename, [
        (field_name, field_data) for field_name, field_data
        in zip(field_names[:clen], field_data_contents[:clen])])
    logger.info("written " + str(clen) + " field(s) in " + filename)

def print_sep():
    """Separate command line output
    """
    print()

def print_parameters_header():
    pass

def print_setup_header():
    print("**********************************************")
    print("* Setting up")
    print("**********************************************")
    print_sep()

def print_timestep_header(i_iter, delta_t):
    print("**********************************************")
    print("* Time step", i_iter, "at t =", str(delta_t * i_iter))
    print("**********************************************")
    print_sep()

def get_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='Biharmonic function extension.')
    parser.add_argument('parameters', metavar='P', type=float, nargs='*',
                   help='Solver parameters [h, ...]')
    return parser.parse_args()

def setup_initial_conditions(kind=1, **kwargs):
    """Setup some CL functions for evaluating initial function
    fields including phi, phi_x, phi_y and b1.
    """
    phi = pmbl.var("phi")
    phi_x = pmbl.var("phi_x")
    phi_y = pmbl.var("phi_y")

    x = pmbl.var("x")
    y = pmbl.var("y")
    r = (x**2 + y**2)**(1/2)

    if 'epsilon' in kwargs:
        epsilon = kwargs['epsilon']
    else:
        epsilon = 0.01

    if 'b1_expr' in kwargs:
        b1_expr = kwargs['b1_expr']

    # equilibrium (asymptotically) phi = tanh(x / sqrt (2 * epsilon))
    def tanh(z):
        # from pymbolic.functions import exp
        from pymbolic.functions import exp
        return (exp(z) - exp(-z)) / (exp(z) + exp(-z))

    def ellipse(x, y, axa, axb, thickness=0.1):
        import math
        rad = x**2 / axa**2 + y**2 / axb**2 - 1
        return tanh(rad / math.sqrt(2 * thickness))

    import math
    if kind==0:
        logger.info("Using interface (with CL) initials")
        phi_init = tanh(10 * x / math.sqrt(2 * epsilon))
    elif kind==1:
        logger.info("Using interface (without CL) initials")
        phi_init = tanh(10 * (r-0.1) / math.sqrt(2 * epsilon))
    elif kind==2:
        logger.info("Using constant initials")
        phi_init = x * 0 - 1

    phi_x_init = pmbl.differentiate(phi_init, 'x')
    phi_y_init = pmbl.differentiate(phi_init, 'y')
    laphi_init  = pmbl.differentiate(pmbl.differentiate(phi_init, 'x'), 'x') + \
             pmbl.differentiate(pmbl.differentiate(phi_init, 'y'), 'y')

    def expr_to_init(expr):
        return pmbl.substitute(expr, {
            "phi": phi_init,
            "phi_x": phi_x_init,
            "phi_y": phi_y_init,
            "lap_phi": laphi_init})


    # Initial values for phi, b1 etc.
    from volumential.tools import ScalarFieldExpressionEvaluation
    def get_init_func(expr):
        import math
        init_expr = expr_to_init(expr)
        return ScalarFieldExpressionEvaluation(dim=2,
                expression=init_expr,
                variables=[x, y],
                function_manglers=[math_func_mangler])

    initial_phi = get_init_func(phi)
    initial_phi_x = get_init_func(phi_x)
    initial_phi_y = get_init_func(phi_y)

    initial_b1 = get_init_func(b1_expr)

    return initial_phi, initial_phi_x, initial_phi_y, initial_b1

def setup_volume_mesh(shape=Geometry.Circle, mesh_order=2, h=0.005):
    # Cache the mesh for better speed & reproducibility
    volume_mesh_filename = "volume_mesh_" + str(shape) + ".meshmode"

    if os.path.isfile(volume_mesh_filename) and 0:
        logger.info("Loading cached volume mesh in " + volume_mesh_filename
                + " (make sure it is the desired one)")

        import pickle
        with open(volume_mesh_filename, 'rb') as handle:
            unserialized_mesh_data = pickle.load(handle)

        from meshmode.mesh import Mesh, SimplexElementGroup
        from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL

        groups = [
                SimplexElementGroup(order=g["order"],
                    vertex_indices=np.array(g["vertex_indices"], dtype=np.int32),
                    nodes=np.array(g["nodes"]),
                    # element_nr_base=g["element_nr_base"],
                    node_nr_base=g["node_nr_base"],
                    unit_nodes=np.array(g["unit_nodes"]),
                    dim=g["dim"])
                for g in unserialized_mesh_data["groups"]]

        mesh = Mesh(vertices=np.array(unserialized_mesh_data["vertices"]),
                groups=groups,
                skip_tests=False,
                node_vertex_consistency_tolerance=None,
                nodal_adjacency=None,
                facial_adjacency_groups=None,
                boundary_tags=[BTAG_ALL, BTAG_REALLY_ALL],
                vertex_id_dtype=np.int32,
                element_id_dtype=np.int32)

    else:

        if shape == Geometry.RegularRectangle:
            from meshmode.mesh.generation import generate_regular_rect_mesh
            ext = 1.
            mesh = generate_regular_rect_mesh(
                a=(-ext / 2., -ext / 2.),
                b=(ext / 2., ext / 2.),
                n=(int(ext / h), int(ext / h)))

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

        else:
            RuntimeError("unsupported geometry")

        logger.info("Saving cached volume mesh in " + volume_mesh_filename)
        import pickle
        from meshmode.mesh.io import to_json
        with open(volume_mesh_filename, 'wb') as handle:
            pickle.dump(to_json(mesh), handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Volume mesh has %d elements" % mesh.nelements)
    return mesh

def setup_qbx_and_discr(ctx, mesh,
        bdry_quad_order=4,
        vol_quad_order=2,
        bdry_ovsmp_quad_order=None,
        qbx_order=5, qbx_fmm_order=10):

    cl_ctx = setup_cl_ctx(ctx)

    if bdry_ovsmp_quad_order is None:
        bdry_ovsmp_quad_order = bdry_quad_order * 4

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
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
        fmm_order=qbx_fmm_order
        ).with_refinement()

    density_discr = qbx.density_discr

    return qbx, density_discr, vol_discr

def get_normal_vectors(queue, density_discr, loc_sign=-1):
    if loc_sign == 1:
        return -bind(density_discr, sym.normal(2).as_vector())(queue)
    elif loc_sign == -1:
        return bind(density_discr, sym.normal(2).as_vector())(queue)

def get_tangent_vectors(queue, density_discr, loc_sign):
    # the domain is on the left.
    normal = get_normal_vectors(queue, density_discr, loc_sign)
    return make_obj_array([-1 * normal[1], normal[0]])

def get_path_length(queue, density_discr):
    return bind(density_discr, sym.integral(2, 1, 1))(queue)

def get_arclength_parametrization_derivative(queue, density_discr, where=None):
    """Return x'(s), y'(s). s Follows the same direction as the parametrization.
    (Remark: t is the reference-to-global parametrization of the discretization).
    """
    queue = setup_command_queue(queue=queue)
    dim = 2

    # xp = dx/dt, yp = dy/dt, sp = ds/dt
    xp, yp = sym.parametrization_derivative_matrix(dim, dim-1, where)
    sp = (xp[0]**2 + yp[0]**2)**(1/2)

    xps = xp[0] / sp
    yps = yp[0] / sp

    return bind(density_discr, xps)(queue), bind(density_discr, yps)(queue)

def get_extension_bie_symbolic_operator(loc_sign=1):
    """
    loc_sign:
      -1 for interior Dirichlet
      +1 for exterior Dirichlet
    """
    logger = logging.getLogger("SETUP")
    logger.info(locals())

    dim = 2
    cse = sym.cse

    sigma_sym = sym.make_sym_vector("sigma", dim)
    int_sigma = sym.Ones() * sym.integral(2, 1, sigma_sym)

    nvec_sym = sym.make_sym_vector("normal", dim)
    mu_sym = sym.var("mu")

    stresslet_obj = StressletWrapper(dim=dim)
    stokeslet_obj = StokesletWrapper(dim=dim)
    bdry_op_sym = (
            loc_sign * 0.5 * sigma_sym
            - stresslet_obj.apply(sigma_sym, nvec_sym, mu_sym,
                qbx_forced_limit='avg')
            - stokeslet_obj.apply(sigma_sym, mu_sym,
                qbx_forced_limit='avg') + int_sigma)

    return bdry_op_sym

def setup_box_quad_points(queue=None, adaptive_mesh=False, vol_quad_order=1,
        n_levels=2, box_a=0., box_b=1.):

    import volumential.meshgen as mg
    queue = setup_command_queue(queue=queue)

    if not adaptive_mesh:
        box_mesh = mg.MeshGen2D(vol_quad_order, n_levels, a=box_a, b=box_b)
        box_mesh.print_info()
        q_points = box_mesh.get_q_points()
        q_weights = box_mesh.get_q_weights()
        q_radii = None

    else:
        box_mesh = mg.MeshGen2D(vol_quad_order, n_levels, a=box_a, b=box_b)
        iloop = 0
        while box_mesh.n_active_cells() < refined_n_cells:
            iloop += 1
            crtr = np.array([
                np.abs(geometry_mask(c) * m)
                     for (c, m) in
                     zip(box_mesh.get_cell_centers(),
                         box_mesh.get_cell_measures()) ])
            assert (np.max(np.abs(crtr))) > 0
            box_mesh.update_mesh(crtr, rratio_top, rratio_bot)
            if iloop > n_refinement_loops:
                print("Max number of refinement loops reached.")
                break

        box_mesh.print_info()
        q_points = box_mesh.get_q_points()
        q_weights = box_mesh.get_q_weights()
        q_radii = None

    try:
        box_mesh.generate_gmsh("box_grid.msh")
    except:
        pass

    legacy_msh_file = True
    if legacy_msh_file:
        import os
        os.system('gmsh box_grid.msh convert_grid -')

    assert (len(q_points) == len(q_weights))
    assert (q_points.shape[1] == dim)

    q_points_org = q_points
    q_points_org2 = np.ascontiguousarray(np.transpose(q_points))

    q_points = make_obj_array(
        [cl.array.to_device(queue, q_points_org2[i]) for i in range(dim)])
    q_weights = cl.array.to_device(queue, q_weights)

    return q_points, q_weights

def get_ch_extra_kernel_kwargs(ch_operator):
    return {'lam1': ch_operator.lambdas[0],
            'lam2': ch_operator.lambdas[1]}

def get_ch_extra_kernel_kwarg_types():
    return (lp.ValueArg("lam1", np.float64),
            lp.ValueArg("lam2", np.float64))

def get_ykw_extra_kernel_kwargs(ch_operator):
    return {'lam': ch_operator.lambdas[0]}

def get_ykw_extra_kernel_kwarg_types():
    return (lp.ValueArg("lam", np.float64), )

def setup_near_field_table(table_filename, root_extent,
        box_a, box_b, ch_operator,
        vol_quad_order, n_levels, n_brick_levels=None,
        brick_quad_order=None, force_recompute=False):

    if force_recompute:
        clean_file(table_filename)
        clean_file("tmp_" + table_filename)

    from volumential.table_manager import NearFieldInteractionTableManager
    tm = NearFieldInteractionTableManager(table_filename,
            root_extent=root_extent)

    nlevels = n_levels

    if n_brick_levels is None:
        n_brick_levels = 15

    if brick_quad_order is None:
        brick_quad_order = 50

    extra_kernel_kwargs = get_ch_extra_kernel_kwargs(ch_operator)
    extra_kernel_kwarg_types = get_ch_extra_kernel_kwarg_types()

    # box size should be integral times of largest table size
    assert (
            (abs(int((box_b-box_a)/root_extent)
                * root_extent - (box_b-box_a))
                < 1e-15)
            or
            (abs(int(root_extent/(box_b-box_a))
                * (box_b-box_a) - root_extent)
                < 1e-15)
            )

    logger.info("getting tables...")
    near_field_table = {}

    # pass a debug parameter list to enable more checks
    table_debug = {"strict_loading": True}

    nftable = []
    for l in range(0, nlevels):
        logger.info("Getting table at level" + str(l))
        tb, _ = tm.get_table(dim, "Cahn-Hilliard", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=ch_operator.b, c=ch_operator.c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types,
                # debug=table_debug
                )
        nftable.append(tb)
    near_field_table[nftable[0].integral_knl.__repr__()] = nftable

    nftable_dx = []
    for l in range(0, nlevels):
        logger.info("Getting table dx at level" + str(l))
        tb, _ = tm.get_table(dim, "Cahn-Hilliard-Dx", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=ch_operator.b, c=ch_operator.c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types)
        nftable_dx.append(tb)
    near_field_table[nftable_dx[0].integral_knl.__repr__()] = nftable_dx

    nftable_dy = []
    for l in range(0, nlevels):
        logger.info("Getting table dy at level" + str(l))
        tb, _ = tm.get_table(dim, "Cahn-Hilliard-Dy", vol_quad_order,
                source_box_level=l, compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=ch_operator.b, c=ch_operator.c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types)
        nftable_dy.append(tb)
    near_field_table[nftable_dy[0].integral_knl.__repr__()] = nftable_dy

    yukawa_extra_kernel_kwargs = get_ykw_extra_kernel_kwargs(ch_operator)
    yukawa_extra_kernel_kwarg_types = get_ykw_extra_kernel_kwarg_types()

    nftable_ykw = []
    for l in range(0, nlevels):
        logger.info("Getting table yukawa at level" + str(l))
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
    for l in range(0, nlevels):
        logger.info("Getting table yukawa-dx at level" + str(l))
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
    for l in range(0, nlevels):
        logger.info("Getting table yukawa-dy at level" + str(l))
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

    return near_field_table

def setup_tree_and_trav(ctx, queue,
        q_points, vol_quad_order,
        box_a=0, box_b=1, plot_tree=False):
    dim = 2
    from boxtree.tools import AXIS_NAMES
    axis_names = AXIS_NAMES[:dim]

    cl_ctx = setup_cl_ctx(ctx, queue)
    queue = setup_command_queue(ctx, queue)

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

    if plot_tree:

        # https://github.com/matplotlib/matplotlib/issues/9294/
        import matplotlib as mpl
        mpl.use('TkAgg')  # default can cause seg fault
        import matplotlib.pyplot as plt

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(tree.get(queue=queue))
        plotter.draw_tree(fill=False, edgecolor="black")
        #plotter.draw_box_numbers()
        plotter.set_bounding_box()
        plt.gca().set_aspect("equal")

        plt.draw()
        plt.savefig("ch_tree.png")
        logger.info("Tree picture saved in ch_tree.png.")

    # get an idea of the finest grid size
    delta_x = 2 * (box_b - box_a) * (0.5**tree.nlevels)
    logger.info("Boxtree's delta_x = " + str(delta_x))

    return tree, trav

def setup_pde_kernel_expansions(ctx, queue, tree, near_field_table,
        ch_operator,
        vol_quad_order, fmm_order_ch, fmm_order_ykw):
    from sumpy.kernel import AxisTargetDerivative
    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    from sumpy.expansion.local import VolumeTaylorLocalExpansion

    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    exclude_self = True
    dtype = np.complex128

    cl_ctx = setup_cl_ctx(ctx, queue)
    queue = setup_command_queue(ctx, queue)

    if exclude_self:
        target_to_source = np.arange(tree.ntargets, dtype=np.int32)
        self_extra_kwargs = {"target_to_source": target_to_source}
    else:
        self_extra_kwargs = {}

    extra_kernel_kwargs = get_ch_extra_kernel_kwargs(ch_operator)
    yukawa_extra_kernel_kwargs = get_ykw_extra_kernel_kwargs(ch_operator)

    from volumential.expansion_wrangler_interface import \
            ExpansionWranglerCodeContainer
    from volumential.expansion_wrangler_fpnd import \
            FPNDExpansionWrangler
    sumpy_knl = FactorizedBiharmonicKernel(dim)
    sumpy_knl_dx = AxisTargetDerivative(0, sumpy_knl)
    sumpy_knl_dy = AxisTargetDerivative(1, sumpy_knl)
    ch_out_knls = [sumpy_knl, sumpy_knl_dx, sumpy_knl_dy]
    wcc_ch = ExpansionWranglerCodeContainer(cl_ctx,
            partial(mpole_expn_class, sumpy_knl),
            partial(local_expn_class, sumpy_knl),
            ch_out_knls,
            exclude_self=exclude_self)
    wrangler_ch = FPNDExpansionWrangler(code_container=wcc_ch,
            queue=queue,
            tree=tree,
            near_field_table=near_field_table,
            dtype=dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order_ch,
            quad_order=vol_quad_order,
            self_extra_kwargs=self_extra_kwargs,
            kernel_extra_kwargs=extra_kernel_kwargs)

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
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order_ykw,
            quad_order=vol_quad_order,
            self_extra_kwargs=self_extra_kwargs,
            kernel_extra_kwargs=yukawa_extra_kernel_kwargs)

    return wrangler_ch, wrangler_ykw

def compute_biharmonic_extension(queue, target_discr,
        qbx, density_discr, f, fx, fy,
        target_association_tolerance=0.05):
    """Note that now the domain is the exterior.
    """
    dim = 2
    queue = setup_command_queue(queue=queue)
    qbx_forced_limit = 1

    nodes = density_discr.nodes().with_queue(queue)
    normal = get_normal_vectors(queue, density_discr, loc_sign=1)

    bdry_op_sym = get_extension_bie_symbolic_operator(loc_sign=1)
    bound_op = bind(qbx, bdry_op_sym)

    bc = [fy, -fx]
    bvp_rhs = bind(qbx, sym.make_sym_vector("bc", dim))(
            queue, bc=bc)
    gmres_result = gmres(
             bound_op.scipy_op(queue, "sigma", np.float64, mu=1., normal=normal),
             bvp_rhs, tol=1e-9, progress=True,
             stall_iterations=0,
             hard_failure=True)
    mu = gmres_result.solution

    arclength_parametrization_derivatives_sym = sym.make_sym_vector(
            "arclength_parametrization_derivatives", dim)
    density_mu_sym = sym.make_sym_vector("mu", dim)
    dxids_sym = arclength_parametrization_derivatives_sym[0] + \
            1j * arclength_parametrization_derivatives_sym[1]
    dxids_conj_sym = arclength_parametrization_derivatives_sym[0] - \
            1j * arclength_parametrization_derivatives_sym[1]
    density_rho_sym = density_mu_sym[1] - 1j * density_mu_sym[0]
    density_conj_rho_sym = density_mu_sym[1] + 1j * density_mu_sym[0]

    # convolutions
    GS1 = sym.IntG(
            ComplexLinearLogKernel(dim), density_rho_sym,
            qbx_forced_limit=None)
    GS2 = sym.IntG(
            ComplexLinearKernel(dim), density_conj_rho_sym,
            qbx_forced_limit=None)
    GD1 = sym.IntG(
            ComplexFractionalKernel(dim), density_rho_sym * dxids_sym,
            qbx_forced_limit=None)
    GD2 = [sym.IntG(
            AxisTargetDerivative(iaxis, ComplexLogKernel(dim)),
            density_conj_rho_sym * dxids_sym + density_rho_sym * dxids_conj_sym,
            qbx_forced_limit=qbx_forced_limit
            ) for iaxis in range(dim)]

    GS1_bdry = sym.IntG(
            ComplexLinearLogKernel(dim), density_rho_sym,
            qbx_forced_limit=qbx_forced_limit)
    GS2_bdry = sym.IntG(
            ComplexLinearKernel(dim), density_conj_rho_sym,
            qbx_forced_limit=qbx_forced_limit)
    GD1_bdry = sym.IntG(
            ComplexFractionalKernel(dim), density_rho_sym * dxids_sym,
            qbx_forced_limit=qbx_forced_limit)

    xp, yp = get_arclength_parametrization_derivative(queue, density_discr)
    xp = -xp
    yp = -yp
    tangent = get_tangent_vectors(queue, density_discr, loc_sign=qbx_forced_limit)

    # check and fix the direction of parametrization
    logger.info("Fix all negative signs in:" +
            str(xp * tangent[0] + yp * tangent[1]))

    grad_v2 = [
            bind(qbx, GD2[iaxis])(queue, mu=mu,
                arclength_parametrization_derivatives=make_obj_array([xp, yp])).real
            for iaxis in range(dim)]
    v2_tangent_der = sum(tangent[iaxis] * grad_v2[iaxis] for iaxis in range(dim))

    from pytential.symbolic.pde.scalar import NeumannOperator
    from sumpy.kernel import LaplaceKernel
    operator_v1 = NeumannOperator(LaplaceKernel(dim), loc_sign=qbx_forced_limit)
    bound_op_v1 = bind(qbx, operator_v1.operator(var("sigma")))
    # FIXME: the positive sign works here
    rhs_v1 = operator_v1.prepare_rhs(1 * v2_tangent_der)
    gmres_result = gmres(
            bound_op_v1.scipy_op(queue, "sigma", dtype=np.float64),
            rhs_v1, tol=1e-9, progress=True,
            stall_iterations=0,
            hard_failure=True)
    sigma = gmres_result.solution
    qbx_stick_out = qbx.copy(
            target_association_tolerance=target_association_tolerance)
    v1 = bind((qbx_stick_out, target_discr),
            operator_v1.representation(var("sigma"), qbx_forced_limit=None)
            )(queue, sigma=sigma)
    grad_v1 = bind((qbx_stick_out, target_discr),
            operator_v1.representation(var("sigma"), qbx_forced_limit=None,
                map_potentials=lambda pot: sym.grad(dim, pot))
            )(queue, sigma=sigma)
    v1_bdry = bind(qbx, operator_v1.representation(var("sigma"),
        qbx_forced_limit=qbx_forced_limit))(queue, sigma=sigma)

    z_conj = target_discr.nodes()[0] - 1j * target_discr.nodes()[1]
    z_conj_bdry = density_discr.nodes().with_queue(queue)[0] \
            - 1j * density_discr.nodes().with_queue(queue)[1]
    int_rho = 1 / (8 * np.pi) * bind(qbx,
            sym.integral(dim, dim - 1, density_rho_sym))(queue, mu=mu)

    # checking complex line integral with the area formula
    int_area = (1 / 2j) * bind(qbx,
            sym.integral(dim, dim - 1, var("z_conj") * dxids_sym))(queue,
                    z_conj=z_conj_bdry,
                    arclength_parametrization_derivatives=make_obj_array(
                        [xp, yp]))
    area_exact = np.pi * (0.25**2)
    logger.info(str(int_area) + " " + str(area_exact))

    omega_S1 = bind((qbx_stick_out, target_discr), GS1)(queue, mu=mu).real
    omega_S2 = - bind((qbx_stick_out, target_discr), GS2)(queue, mu=mu).real
    omega_S3 = (z_conj * int_rho).real
    omega_S = -(omega_S1 + omega_S2 + omega_S3)

    grad_omega_S1 = bind((qbx_stick_out, target_discr),
            sym.grad(dim, GS1))(queue, mu=mu).real
    grad_omega_S2 = - bind((qbx_stick_out, target_discr),
            sym.grad(dim, GS2))(queue, mu=mu).real
    grad_omega_S3 = (int_rho * make_obj_array([1., -1.])).real
    grad_omega_S = -(grad_omega_S1 + grad_omega_S2 + grad_omega_S3)

    omega_S1_bdry = bind(qbx, GS1_bdry)(queue, mu=mu).real
    omega_S2_bdry = - bind(qbx, GS2_bdry)(queue, mu=mu).real
    omega_S3_bdry = (z_conj_bdry * int_rho).real
    omega_S_bdry = -(omega_S1_bdry + omega_S2_bdry + omega_S3_bdry)

    omega_D1 = bind((qbx_stick_out, target_discr), GD1)(queue, mu=mu,
            arclength_parametrization_derivatives=make_obj_array([xp, yp])).real
    omega_D = (omega_D1 + v1)

    grad_omega_D1 = bind((qbx_stick_out, target_discr),
            sym.grad(dim, GD1)
            )(queue, mu=mu,
                    arclength_parametrization_derivatives=make_obj_array(
                        [xp, yp]
                        )
                    ).real
    grad_omega_D = grad_omega_D1 + grad_v1

    omega_D1_bdry = bind(qbx, GD1_bdry)(queue, mu=mu,
            arclength_parametrization_derivatives=make_obj_array([xp, yp])).real
    omega_D_bdry = (omega_D1_bdry + v1_bdry)

    int_bdry_mu = bind(qbx, sym.integral(dim, dim - 1, sym.make_sym_vector(
        "mu", dim)))(queue, mu=mu)
    omega_W = int_bdry_mu[0] * target_discr.nodes()[1] - \
            int_bdry_mu[1] * target_discr.nodes()[0]
    grad_omega_W = make_obj_array([
        -int_bdry_mu[1], int_bdry_mu[0]
        ])
    omega_W_bdry = int_bdry_mu[0] * density_discr.nodes().with_queue(queue)[1] - \
            int_bdry_mu[1] * density_discr.nodes().with_queue(queue)[0]

    int_bdry = bind(qbx, sym.integral(dim, dim - 1, var("integrand")))(
            queue, integrand=omega_S_bdry+omega_D_bdry+omega_W_bdry)

    debugging_info = {}
    debugging_info['omega_S'] = omega_S
    debugging_info['omega_D'] = omega_D
    debugging_info['omega_W'] = omega_W
    debugging_info['omega_v1'] = v1
    debugging_info['omega_D1'] = omega_D1

    int_interior_func_bdry = bind(qbx, sym.integral(2, 1, var("integrand")))(
            queue, integrand=f)

    path_length = get_path_length(queue, density_discr)
    ext_f = omega_S + omega_D + omega_W + (
            int_interior_func_bdry - int_bdry) / path_length
    grad_f = grad_omega_S + grad_omega_D + grad_omega_W

    return ext_f, grad_f[0], grad_f[1], debugging_info

# }}} End function details

def main(parameters=[]):

    # {{{ handling parameters

    print_parameters_header()

    n_iters = 300
    output_interval = 10

    bbox_a = -0.25; bbox_b = 0.25
    #bbox_a = -0.5; bbox_b = 0.5

    table_filename = "nft_ch.hdf5"
    root_table_source_extent=0.5

    vol_quad_order = 4
    n_boxtree_levels = 7

    visual_order = 2
    write_vol = True
    write_bdry = True
    write_imm = True

    write_coeff_matrix = False
    write_domain_mask = True

    epsilon = 0.01
    delta_t = 0.05
    theta_y = 60. / 180. * np.pi

    initial_kind = 0
    no_contact_line = False

    # For energy stability,
    # 1. s >= (3M^2-1)/2, where M=max(|phi|)
    #    e.g., when M=1, need S>=1
    # 2. alpha_tild >= N/2, where N=|sqrt(2)/3 * (pi/2)^2 * cos(theta_y)|
    #    e.g., when theta_y=0, N=1.163144 is the maximum possible value of N
    # alpha_tild = 0 in the notes.
    s = 1.5
    alpha_tild = 0
    # boundary convex splitting
    bs = alpha_tild / epsilon

    equation_param_b = s / (epsilon**2)
    equation_param_c = 1. / (epsilon * delta_t)

    from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator
    chop = CahnHilliardOperator(
            b=equation_param_b,
            c=equation_param_c)

    qbx_fmm_order = 5
    vp_fmm_order_ch = 5
    vp_fmm_order_ykw = 5

    # {{{ pde related formulas

    phi = pmbl.var("phi")
    phi_x = pmbl.var("phi_x")
    phi_y = pmbl.var("phi_y")
    lap_phi = pmbl.var("lap_phi")
    lap_phi3 = 6 * phi * (phi_x**2 + phi_y**2) + 3 * phi**2 * lap_phi

    def gamma(z):
        """specific surface energy
        """
        import math
        from pymbolic.functions import sin

        if 0:
            # NOTE: a different sign (as in [Gao, Wang, 2014]), correspons to
            # a different theta_y ( vs pi - theta_y )
            gamma_coef = math.sqrt(2) / 3 * math.cos(theta_y)

        else:
            # NOTE: as in the notes
            gamma_coef = -math.sqrt(2) / 3 * math.cos(theta_y)

        pi_val = math.pi
        return gamma_coef * sin(pi_val / 2 * z)

    gamma_expr = gamma(phi)
    b1_expr = (equation_param_c - bs) * phi + \
            1 / epsilon * pmbl.differentiate(gamma(phi), 'phi')

    f0_expr = equation_param_c * phi
    f1_expr = equation_param_c * phi - \
            (1 + s) / epsilon**2 * lap_phi + 1 / epsilon**2 * lap_phi3
    f2_expr = (phi*phi*phi - (1 + s) * phi) / epsilon**2
    f2_x_expr = pmbl.differentiate(f2_expr, 'phi') * phi_x
    f2_y_expr = pmbl.differentiate(f2_expr, 'phi') * phi_y

    # }}} End pde related formulas

    force_direct_evaluation = False

    print_sep()

    # }}} End handling parameters

    import time
    start_time = time.time()

    # {{{ setup

    print_setup_header()

    ctx = setup_cl_ctx()
    queue = setup_command_queue(ctx)
    check_time_cl = time.time()
    logger.info("Finished setting up OpenCL in "
            + str(check_time_cl - start_time))
    print_sep()

    vol_mesh = setup_volume_mesh()
    check_time_vm = time.time()
    logger.info("Finished volume mesh generation in "
            + str(check_time_vm - check_time_cl))
    print_sep()

    qbx, density_discr, vol_discr = setup_qbx_and_discr(ctx, vol_mesh,
            bdry_quad_order=4, vol_quad_order=vol_quad_order,
            qbx_order=5, qbx_fmm_order=qbx_fmm_order)
    qbx_stick_out = qbx.copy(target_association_tolerance=0.05)
    bdry_normals = get_normal_vectors(queue, density_discr)
    check_time_qd = time.time()
    logger.info("Finished qbx setup and density discretization in "
            + str(check_time_qd - check_time_vm))
    print_sep()

    q_points, q_weights = setup_box_quad_points(queue=queue,
            adaptive_mesh=False,
            vol_quad_order=vol_quad_order, n_levels=n_boxtree_levels,
            box_a=bbox_a, box_b=bbox_b)
    n_q_points = len(q_points[0])
    assert n_q_points == len(q_weights)
    q_points_host = np.array([qpc.get() for qpc in q_points])
    box_discr = PointsTarget(q_points, normals=None)
    check_time_pw = time.time()
    logger.info("Finished generating boxtree quad points in "
            + str(check_time_pw - check_time_qd))
    print_sep()

    tree, trav = setup_tree_and_trav(ctx, queue, q_points,
            vol_quad_order=vol_quad_order,
            box_a=bbox_a, box_b=bbox_b, plot_tree=False)
    check_time_tt = time.time()
    logger.info("Finished generating boxtee & its traversal in "
            + str(check_time_tt - check_time_pw))
    print_sep()

    near_field_table = setup_near_field_table(table_filename,
            root_extent=root_table_source_extent,
            box_a=bbox_a, box_b=bbox_b,
            ch_operator=chop,
            vol_quad_order=vol_quad_order,
            n_levels=tree.nlevels)
    check_time_nft = time.time()
    logger.info("Finished loading/generating precomputed tables in "
            + str(check_time_nft - check_time_tt))
    print_sep()

    wrangler_ch, wrangler_ykw = setup_pde_kernel_expansions(
            ctx, queue, tree, near_field_table,
            ch_operator=chop,
            vol_quad_order=vol_quad_order,
            fmm_order_ch=vp_fmm_order_ch,
            fmm_order_ykw=vp_fmm_order_ykw)
    check_time_ke = time.time()
    logger.info("Finished sumpy FMM setup in "
            + str(check_time_ke - check_time_nft))
    print_sep()

    # use a double layer potential to indicate the domain
    # for circular domain: interior -1, exterior 0
    # FIXME: true for noncircular domain?
    from sumpy.kernel import LaplaceKernel
    domain_mask_pre = bind(
            (qbx_stick_out, box_discr),
            sym.D(LaplaceKernel(dim), sym.Ones(),
                qbx_forced_limit=None))(queue).get()
    bdpot_lev = -0.5
    q_point_indicies = np.array(range(n_q_points))
    exterior_q_point_indicies = q_point_indicies[domain_mask_pre >= bdpot_lev]
    exterior_q_points = make_obj_array([
        cl.array.to_device(queue, q_points_host[i][exterior_q_point_indicies])
        for i in range(dim)])
    box_discr_exterior = PointsTarget(exterior_q_points, normals=None)
    interior_q_point_indicies = q_point_indicies[domain_mask_pre < bdpot_lev]
    interior_q_points = make_obj_array([
        cl.array.to_device(queue, q_points_host[i][interior_q_point_indicies])
        for i in range(dim)])
    box_discr_interior = PointsTarget(interior_q_points, normals=None)
    domain_mask_host = np.zeros(n_q_points)
    domain_mask_host[interior_q_point_indicies] = 1
    domain_mask = cl.array.to_device(queue, domain_mask_host)
    def fill_exterior(vec, ext_vec):
        """fill exterior points with new values.
        Scalar values for ext_vec are also accepted and are
        treated as constant vectors.
        """
        new_vec = domain_mask * vec + (1-domain_mask) * ext_vec
        return new_vec
    check_time_dm = time.time()
    logger.info("Finished computing domain mask in "
            + str(check_time_dm - check_time_ke))
    print_sep()

    from meshmode.discretization.visualization import make_visualizer
    vol_vis = make_visualizer(queue, vol_discr, visual_order)
    vol_discr_nodes = make_obj_array(
            [cl.array.to_device(queue,
                nodes_c.with_queue(queue).get()) for nodes_c
                in vol_discr.nodes()])
    bdry_vis = make_visualizer(queue, density_discr, 2*visual_order)
    bdry_discr_nodes = make_obj_array(
            [cl.array.to_device(queue,
                nodes_c.with_queue(queue).get()) for nodes_c
                in density_discr.nodes()])
    bdry_nodes_host = np.array([nc.get() for nc in bdry_discr_nodes])
    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=bbox_b - bbox_a, npoints=500)
    imm_discr_nodes = make_obj_array([
        cl.array.to_device(queue, fplot.points[i]) for i in range(dim)])
    if 0:
        # not yet supported by the Python implementation of meshgen
        from meshmode.mesh.io import read_gmsh
        modemesh = read_gmsh("box_grid.msh", force_ambient_dim=None)
        from meshmode.discretization.poly_element import \
                LegendreGaussLobattoTensorProductGroupFactory
        boxtree_discr = Discretization(ctx, modemesh,
                LegendreGaussLobattoTensorProductGroupFactory(vol_quad_order))
        boxtree_vis = make_visualizer(queue, boxtree_discr, visual_order)
    check_time_pp = time.time()
    logger.info("Finished setting up visualizers in "
            + str(check_time_pp - check_time_ke))
    print_sep()

    if write_domain_mask:
        # domain_mask has interpolation artifacts
        write_visualization_file(
                fplot,
                get_output_filename("ch-domain", None, ".vts"),
                ["domain", "layer_pot", "domain_mask"], [
                    (interpolate_volume_potential(imm_discr_nodes,
                        trav, wrangler_ch, domain_mask_pre).real
                        < bdpot_lev).get(),
                    interpolate_volume_potential(imm_discr_nodes,
                        trav, wrangler_ch, domain_mask_pre).real.get(),
                    interpolate_volume_potential(imm_discr_nodes,
                        trav, wrangler_ch, domain_mask_host).real.get()
                    ]
                )
        check_time_wvf = time.time()
        logger.info("Finished writing output for the domain maskup in "
                + str(check_time_wvf - check_time_pp))
        print_sep()

    initial_phi, initial_phi_x, initial_phi_y, initial_b1 = \
            setup_initial_conditions(kind=initial_kind, epsilon=epsilon,
                    b1_expr=b1_expr)

    def pot_to_bdry(volume_pot):
        # FIXME: cache the area queries for pot_to_bdry.
        return interpolate_volume_potential(
                bdry_discr_nodes, trav, wrangler_ch, volume_pot).real

    # }}} End setup

    setup_time = time.time() - start_time
    start_time = time.time()

    # {{{ time marching

    for i_iter in range(n_iters):
        print_timestep_header(i_iter, delta_t)

        # {{{ prepare source density

        if i_iter == 0:
            logger.info("Evaluating initials")
            old_phi = cl.array.to_device(queue,
                    initial_phi(queue, q_points_host))
            old_phi_x = cl.array.to_device(queue,
                    initial_phi_x(queue, q_points_host))
            old_phi_y = cl.array.to_device(queue,
                    initial_phi_y(queue, q_points_host))
        else:
            old_phi = new_phi.real
            old_phi_x = new_phi_x.real
            old_phi_y = new_phi_y.real

        # f0 = c * phi
        f0_vals = pmbl.evaluate(f0_expr, {"phi": old_phi})

        # f1 = f0 + laplace(f2)
        # not explicitly computed

        # f2 = (1/epsilon**2) * (phi**3 - (1+s)*phi)
        f2_vals = pmbl.evaluate(f2_expr, {"phi": old_phi})

        # used for post-processing volume potentials
        f2_x_vals = pmbl.evaluate(f2_x_expr,
                {"phi": old_phi, "phi_x": old_phi_x, "phi_y": old_phi_y})
        f2_y_vals = pmbl.evaluate(f2_y_expr,
                {"phi": old_phi, "phi_x": old_phi_x, "phi_y": old_phi_y})

        if i_iter == 0:
            if write_vol:
                write_visualization_file(
                        vol_vis,
                        get_output_filename("ch-vol-", i_iter, ".vtu"),
                        ["phi", "phi_x", "phi_y", "f2", "f2_x", "f2_y"],
                        [interpolate_volume_potential(vol_discr_nodes,
                            trav, wrangler_ch, field_var).real
                            for field_var in
                            [old_phi, old_phi_x, old_phi_y,
                                f2_vals, f2_x_vals, f2_y_vals]]
                            )
            if write_imm:
                write_visualization_file(
                        fplot,
                        get_output_filename("ch-imm-", i_iter, ".vts"),
                        ["phi", "phi_x", "phi_y", "f2", "f2_x", "f2_y"],
                        [interpolate_volume_potential(imm_discr_nodes,
                            trav, wrangler_ch, field_var).real.get()
                            for field_var in
                            [old_phi, old_phi_x, old_phi_y,
                                f2_vals, f2_x_vals, f2_y_vals]]
                            )

        logger.info("Updated source density.")

        # }}} End prepare source density

        # {{{ evaluate volume potentials

        logger.info("Evaluating volume potentials 1/4")
        pot_v0f0 = drive_volume_fmm(trav, wrangler_ch,
                f0_vals * q_weights, f0_vals,
                direct_evaluation=force_direct_evaluation)

        logger.info("Evaluating volume potentials 2/4")
        pot_v0f2 = drive_volume_fmm(trav, wrangler_ch,
                f2_vals * q_weights, f2_vals,
                direct_evaluation=force_direct_evaluation)

        logger.info("Evaluating volume potentials 3/4")
        pot_v1f0 = drive_volume_fmm(trav, wrangler_ykw,
                f0_vals * q_weights, f0_vals,
                direct_evaluation=force_direct_evaluation)

        logger.info("Evaluating volume potentials 4/4")
        pot_v1f2 = drive_volume_fmm(trav, wrangler_ykw,
                f2_vals * q_weights, f2_vals,
                direct_evaluation=force_direct_evaluation)

        logger.info("Processing volume potentials")
        # u_tild = V0[f1], which is replaced with
        #          V0[f0] + V1[f2] + lam2**2 * V0[f2]
        u_tild_pot, u_tild_x_pot, u_tild_y_pot = (
                pot_v0f0[i].real + pot_v1f2[i].real
                + chop.lambdas[1]**2 * pot_v0f2[i].real
                for i in range(3))

        # v_tild = -V1[f1] + lam1**2 * V0[f1]
        # V1[f1] is replaced with V1[f0] + lam1**2 * V1[f2] + f2
        # which differs by a layer potential (and the difference will be
        # taken into account by the BIE solve)
        v_tild_pot = -(pot_v1f0[0].real + chop.lambdas[0]**2 * pot_v1f2[0].real
                + f2_vals) + chop.lambdas[0]**2 * u_tild_pot
        v_tild_x_pot = -(pot_v1f0[1].real + chop.lambdas[0]**2 * pot_v1f2[1].real
                + f2_x_vals) + chop.lambdas[0]**2 * u_tild_x_pot
        v_tild_y_pot = -(pot_v1f0[2].real + chop.lambdas[0]**2 * pot_v1f2[2].real
                + f2_y_vals) + chop.lambdas[0]**2 * u_tild_y_pot

        # }}} End evaluate volume potentials

        # {{{ BIE solve

        logger.info("Preparing BC")

        bdry_u_tild = pot_to_bdry(u_tild_pot)
        bdry_grad_u_tild = [pot_to_bdry(u_tild_x_pot),
                pot_to_bdry(u_tild_y_pot)]
        bdry_u_tild_normal = sum(bdry_grad_u_tild[i] * bdry_normals[i]
                for i in range(dim))

        bdry_v_tild = pot_to_bdry(v_tild_pot)
        bdry_grad_v_tild = [pot_to_bdry(v_tild_x_pot),
                pot_to_bdry(v_tild_y_pot)]
        bdry_v_tild_normal = sum(bdry_grad_v_tild[i] * bdry_normals[i]
                for i in range(dim))

        if i_iter == 0:
            bdry_b1_vals = cl.array.to_device(queue,
                    initial_b1(queue, bdry_nodes_host))
        else:
            bdry_b1_vals = pmbl.evaluate(b1_expr, {
                "phi": bdry_new_phi, "math": cl.clmath})

        bdry_g1_vals = bdry_b1_vals - \
                bdry_u_tild_normal - \
                (equation_param_c - bs) * bdry_u_tild

        # du/dnormal ~ d(phi)/dnormal ~ O(1/epsilon)
        # mu ~ epsilon * laplace(phi) ~ O(1/epsilon)
        # dv/dnormal ~ d(mu/epsilon)/dnormal ~ O(1/epsilon**3)
        # Rescale the second BIE with epsilon*2 to get them to the same scale
        # which affects how GEMRES understands the residual.
        second_bie_rscale = epsilon**2
        bdry_g2_vals = - second_bie_rscale *  bdry_v_tild_normal

        unk = chop.make_unknown("sigma")
        bound_op = bind(qbx,
                chop.operator(unk,
                    rscale=(1, second_bie_rscale),
                    bdry_splitting_parameter=bs))

        bc = sym.make_obj_array([
            bdry_g1_vals,
            bdry_g2_vals
        ])

        if write_coeff_matrix and i_iter==0:
            logger.info("reconstructing matrix")
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

        # For later time steps, use previous sigma as initial guess
        guess = None
        if 1:
            if i_iter == 0:
                guess = None
            else:
                guess = cl.array.to_device(queue,
                        np.concatenate([sig.get() for sig in sigma]))

        gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.complex128),
            bc,
            x0=guess,
            tol=1e-9,
            progress=True,
            stall_iterations=0,
            hard_failure=True)
        sigma = gmres_result.solution

        # }}} End BIE solve

        # {{{ assemble solution

        logger.info("assembling the solution")

        # vol vals only useful inside the domain
        bvp_sol = bind(
                (qbx_stick_out, box_discr),
                chop.representation(unk))(queue, sigma=sigma)

        # dbry vals on the interior side
        bdry_bvp_sol = bind(
                (qbx_stick_out, density_discr),
                chop.representation(unk, qbx_forced_limit=-1))(queue, sigma=sigma)

        # representaion:
        # u, v, u_x, u_y, lap(u)
        new_phi = u_tild_pot + bvp_sol[0].real
        new_phi_x = u_tild_x_pot + bvp_sol[2].real
        new_phi_y = u_tild_y_pot + bvp_sol[3].real

        bdry_new_phi = bdry_u_tild + bdry_bvp_sol[0].real
        bdry_new_phi_x = bdry_grad_u_tild[0] + bdry_bvp_sol[2].real
        bdry_new_phi_y = bdry_grad_u_tild[1] + bdry_bvp_sol[3].real

        if no_contact_line:
            new_phi = fill_exterior(new_phi, 1)
            new_phi_x = fill_exterior(new_phi_x, 0)
            new_phi_y = fill_exterior(new_phi_y, 0)
        else:
            # replace the exterior vol vals with biharmonic extensions
            # from the boundary
            ext_new_phi, ext_new_phi_x, ext_new_phi_y, _ = \
                    compute_biharmonic_extension(queue,
                            target_discr=box_discr,
                            qbx=qbx, density_discr=density_discr,
                            f=bdry_new_phi,
                            fx=bdry_new_phi_x,
                            fy=bdry_new_phi_y)
            new_phi = fill_exterior(new_phi, ext_new_phi)
            new_phi_x = fill_exterior(new_phi_x, ext_new_phi_x)
            new_phi_y = fill_exterior(new_phi_y, ext_new_phi_y)

        # FIXME:
        # phi must be in [-1, 1]
        # overflow_phi_1 = (new_phi > 1)
        # overflow_phi_2 = (new_phi < -1)

        # }}} End put pieces together

        # {{{ postprocess
        logger.info("postprocessing")

        if write_vol and i_iter % output_interval == 0:
            write_visualization_file(vol_vis,
                    get_output_filename("ch-vol-", i_iter + 1, ".vtu"),
                    ["phi", "phi_x", "phi_y"],
                    [interpolate_volume_potential(vol_discr_nodes,
                        trav, wrangler_ch, field_var).real
                        for field_var in
                        [new_phi, new_phi_x, new_phi_y]]
                    )

        if write_bdry and i_iter % output_interval == 0:
            write_visualization_file(bdry_vis,
                    get_output_filename("ch-bdry-", i_iter + 1, ".vtu"),
                    ["phi", "phi_x", "phi_y"],
                    [bdry_new_phi, bdry_new_phi_x, bdry_new_phi_y]
                    )

        if write_imm and i_iter % output_interval == 0:
            write_visualization_file(fplot,
                    get_output_filename("ch-imm-", i_iter, ".vts"),
                    ["phi", "phi_x", "phi_y"],
                    [interpolate_volume_potential(imm_discr_nodes,
                        trav, wrangler_ch, field_var).real.get()
                        for field_var in
                        [new_phi, new_phi_x, new_phi_y]
                        ]
                    )

        # }}} End postprocess

    # }}} End time marching

    marching_time = time.time() - start_time
    start_time = time.time()

    print(setup_time, marching_time)

if __name__ == "__main__":
    args = get_command_line_args()
    parameters = args.parameters
    nparams = len(parameters)
    logger.info("parameters: " + str(parameters))
    main(parameters)
