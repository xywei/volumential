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

import logging
logging.basicConfig(level=logging.INFO)

# {{{ function details

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

def setup_qbx_and_discr(ctx, nelements,
        mesh_order=2, target_order=4, qbx_order=4,
        ovsmp_target_order=None,
        fmm_order=10, circle_rad=1.5,
        target_association_tolerance=0.05):

    logger = logging.getLogger("SETUP")
    logger.info(locals())

    if ovsmp_target_order is None:
        ovsmp_target_order = target_order * 4

    from meshmode.mesh.generation import (  # noqa
            make_curve_mesh, starfish, ellipse, drop)
    mesh = make_curve_mesh(
            lambda t: circle_rad * ellipse(1, t),
            np.linspace(0, 1, nelements+1),
            mesh_order)

    cl_ctx = setup_cl_ctx(ctx)
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    coarse_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx, _ = QBXLayerPotentialSource(
            coarse_density_discr, fine_order=ovsmp_target_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            target_association_tolerance=target_association_tolerance,
            _expansions_in_tree_have_extent=True,
            ).with_refinement()
    density_discr = qbx.density_discr

    return qbx, density_discr

def get_normal_vectors(queue, density_discr, loc_sign):
    if loc_sign == 1:
        return -bind(density_discr, sym.normal(2).as_vector())(queue)
    elif loc_sign == -1:
        return bind(density_discr, sym.normal(2).as_vector())(queue)

def get_tangent_vectors(queue, density_discr, loc_sign):
    # normal vecs rotated by pi/2 in counter-clockwise direction
    # so that the domain is on the left.
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

def get_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='Biharmonic function extension.')
    parser.add_argument('parameters', metavar='P', type=float, nargs='*',
                   help='Solver parameters [nelements, ...]')
    return parser.parse_args()

def get_bie_symbolic_operator(loc_sign=1):
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

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxLogKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


class ComplexLinearLogKernel(ExpressionKernel):

    init_arg_names = ("dim",)

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

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxLinKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


class ComplexFractionalKernel(ExpressionKernel):

    init_arg_names = ("dim",)

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

def Goursat_function(queue, mu, target_discr, density_discr, qbx,
        qbx_forced_limit, target_association_tolerance=0.05):
    """Evaluate the Goursat function associated with mu on the target
    discretization.
    """
    if qbx_forced_limit is None:
        raise RuntimeError("Must specify a side +1/-1")

    queue = setup_command_queue(queue=queue)
    dim = 2
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
    print("Fix all negative signs in:", xp * tangent[0] + yp * tangent[1])

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
    qbx_stick_out = qbx.copy(target_association_tolerance=target_association_tolerance)

    v1 = bind((qbx_stick_out, target_discr), operator_v1.representation(var("sigma"),
        qbx_forced_limit=None))(queue, sigma=sigma)
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
    area_exact = np.pi * (1.5**2)
    print(int_area, area_exact)

    omega_S1 = bind((qbx_stick_out, target_discr), GS1)(queue, mu=mu).real
    omega_S2 = - bind((qbx_stick_out, target_discr), GS2)(queue, mu=mu).real
    omega_S3 = (z_conj * int_rho).real
    omega_S = -(omega_S1 + omega_S2 + omega_S3)

    omega_S1_bdry = bind(qbx, GS1_bdry)(queue, mu=mu).real
    omega_S2_bdry = - bind(qbx, GS2_bdry)(queue, mu=mu).real
    omega_S3_bdry = (z_conj_bdry * int_rho).real
    omega_S_bdry = -(omega_S1_bdry + omega_S2_bdry + omega_S3_bdry)

    omega_D1 = bind((qbx_stick_out, target_discr), GD1)(queue, mu=mu,
            arclength_parametrization_derivatives=make_obj_array([xp, yp])).real
    omega_D = (omega_D1 + v1)

    omega_D1_bdry = bind(qbx, GD1_bdry)(queue, mu=mu,
            arclength_parametrization_derivatives=make_obj_array([xp, yp])).real
    omega_D_bdry = (omega_D1_bdry + v1_bdry)

    int_bdry_mu = bind(qbx, sym.integral(dim, dim - 1, sym.make_sym_vector(
        "mu", dim)))(queue, mu=mu)
    omega_W = int_bdry_mu[0] * target_discr.nodes()[1] - \
            int_bdry_mu[1] * target_discr.nodes()[0]
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

    return omega_S + omega_D + omega_W, int_bdry, debugging_info

# }}} End function details

def main():
    ctx = setup_cl_ctx()
    queue = setup_command_queue(ctx)

    qbx, density_discr = setup_qbx_and_discr(ctx, nelements=40, fmm_order=10,
        mesh_order=4, target_order=4, qbx_order=10,
        ovsmp_target_order=None,
        circle_rad=1.5)

    nodes = density_discr.nodes().with_queue(queue)
    normal = get_normal_vectors(queue, density_discr, loc_sign=1)
    xp, yp = get_arclength_parametrization_derivative(queue, density_discr)
    path_length = get_path_length(queue, density_discr)

    bdry_op_sym = get_bie_symbolic_operator(loc_sign=1)
    bound_op = bind(qbx, bdry_op_sym)

    strength = 0

    def interior_func(x, y, loc=np.array([0., 0.])):
        # w = x^2 * y + 3 * y + y^2
        return x * x * y + 3 * y + y * y
        #return x

    def boundary_vals(x, y, loc=np.array([0., 0.])):
        # u = (w_y, -w_x)
        # grad(w) = (-u2, u1)
        r = cl.clmath.sqrt((x - loc[0])**2 + (y - loc[1])**2)
        dwdx = 2 * x * y
        dwdy = x * x + 3 + 2 * y
        #dwdx = x * 0 + 1
        #dwdy = y * 0 + 0
        return [dwdy, -dwdx]

    bc = boundary_vals(nodes[0], nodes[1])

    bvp_rhs = bind(qbx, sym.make_sym_vector("bc", 2))(
            queue, bc=bc)

    gmres_result = gmres(
             bound_op.scipy_op(queue, "sigma", np.float64, mu=1., normal=normal),
             bvp_rhs, tol=1e-9, progress=True,
             stall_iterations=0,
             hard_failure=True)

    sigma = gmres_result.solution

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)
    targets = cl.array.to_device(queue, fplot.points)
    target_discr = PointsTarget(targets)

    goursat_func, goursat_int_bdry, goursat_debugging = Goursat_function(queue, sigma,
            target_discr, density_discr, qbx, qbx_forced_limit=1)

    int_interior_func_bdry = bind(qbx, sym.integral(2, 1, var("integrand")))(
            queue, integrand=interior_func(nodes[0], nodes[1]))
    goursat_func = goursat_func + (int_interior_func_bdry
            - goursat_int_bdry) / path_length

    # {{{ velocity field
    sigma_sym = sym.make_sym_vector("sigma", 2)
    int_sigma = sym.Ones() * sym.integral(2, 1, sigma_sym)
    nvec_sym = sym.make_sym_vector("normal", 2)
    mu_sym = sym.var("mu")
    stresslet_obj = StressletWrapper(dim=2)
    stokeslet_obj = StokesletWrapper(dim=2)
    vel_representation_sym = (
            - stresslet_obj.apply(sigma_sym, nvec_sym, mu_sym,
                qbx_forced_limit=1)
            - stokeslet_obj.apply(sigma_sym, mu_sym,
                qbx_forced_limit=1) + int_sigma)
    velocity = bind(qbx, vel_representation_sym)(
            queue, sigma=sigma, normal=normal, mu=1.)

    vel_representation_vol_sym = (
            - stresslet_obj.apply(sigma_sym, nvec_sym, mu_sym,
                qbx_forced_limit=None)
            - stokeslet_obj.apply(sigma_sym, mu_sym,
                qbx_forced_limit=None))
    velocity_vol = bind((qbx, target_discr), vel_representation_vol_sym)(
            queue, sigma=sigma, normal=normal, mu=1.)
    velocity_vol2 = bind(qbx, sym.integral(2,1,sigma_sym))(queue, sigma=sigma)
    print("velocity contribution (W) :", velocity_vol2)

    velocity_vol_S = bind((qbx, target_discr), - stokeslet_obj.apply(sigma_sym, mu_sym,
        qbx_forced_limit=None))(queue, sigma=sigma, mu=1.)
    velocity_vol_D = bind((qbx, target_discr), - stresslet_obj.apply(sigma_sym,
        nvec_sym, mu_sym, qbx_forced_limit=None))(queue, sigma=sigma, mu=1., normal=normal)
    # }}} End velocity field

    from volumential.tools import clean_file
    clean_file("biharmonic.vts")
    fplot.write_vtk_file(
            "biharmonic.vts",
            [
                ("goursat_func", goursat_func.get(queue)),
                ("interior_func", interior_func(targets[0], targets[1]).get(queue)),
                ("V_x", velocity_vol[0].get() + velocity_vol2[0]),
                ("V_y", velocity_vol[1].get() + velocity_vol2[1]),
                ("omega_S", goursat_debugging['omega_S'].get(queue)),
                ("omega_D", goursat_debugging['omega_D'].get(queue)),
                ("omega_D1", goursat_debugging['omega_D1'].get(queue)),
                ("omega_v1", goursat_debugging['omega_v1'].get(queue)),
                ("omega_W", goursat_debugging['omega_W'].get(queue)),
                ("VelS_x", velocity_vol_S[0].get()),
                ("VelS_y", velocity_vol_S[1].get()),
                ("VelD_x", velocity_vol_D[0].get()),
                ("VelD_y", velocity_vol_D[1].get()),
                ])

    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(queue, density_discr, 2)
    bdry_normals = normal
    bdry_velocity = velocity
    bdry_vals = boundary_vals(nodes[0], nodes[1])
    clean_file("biharmonic-bdry.vtu")
    bdry_vis.write_vtk_file("biharmonic-bdry.vtu",
            [
                ("bdry_normals", bdry_normals),
                ("velocity", bdry_velocity),
                ("bdry_vals", make_obj_array(bdry_vals)),
                ])



if __name__ == "__main__":
    args = get_command_line_args()
    parameters = args.parameters
    print("parameters:", parameters)
    main()
