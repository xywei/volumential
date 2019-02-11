"""Function extension with regularity contraints:
    1. L^2: extend with constant value
    2. C^0: harmonic extension
    3. C^1: biharmonic extension
"""

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

import numpy as np
import pyopencl as cl
from pytools.obj_array import make_obj_array
from pymbolic import var
from pytential import bind, sym, norm
from pytential.solve import gmres
from pytential.symbolic.stokes import StressletWrapper, StokesletWrapper

# {{{ helper functions

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



# }}} End helper functions

# {{{ constant extension

def compute_constant_extension(queue, target_discr,
        qbx, density_discr, constant_val=0):
    """Constant extension."""
    dim = qbx.ambient_dim
    queue = setup_command_queue(queue=queue)
    debugging_info = {}

    nodes = density_discr.nodes().with_queue(queue)

    ext_f = bind(
                 (qbx, target_discr),
                 sym.var("sigma") * 0 + constant_val,
                )(queue, sigma=1).real

    return ext_f, debugging_info

# }}} End constant extension

# {{{ harmonic extension

def compute_harmonic_extension(queue, target_discr,
        qbx, density_discr, f,
        loc_sign=1,
        target_association_tolerance=0.05,
        gmres_tolerance=1e-14):
    """Harmonic extension.
    loc_sign indicates the domain for extension, which
             equals to the negation of the loc_sign for
             the original problem.
    """
    dim = qbx.ambient_dim
    queue = setup_command_queue(queue=queue)

    nodes = density_discr.nodes().with_queue(queue)

    # {{{ describe bvp

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(dim)

    cse = sym.cse

    sigma_sym = sym.var("sigma")
    #sqrt_w = sym.sqrt_jac_q_weight(3)
    sqrt_w = 1
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)

    bdry_op_sym = (loc_sign*0.5*sigma_sym
            + sqrt_w*(
                sym.S(kernel, inv_sqrt_w_sigma)
                + sym.D(kernel, inv_sqrt_w_sigma)
                ))

    # }}}

    bound_op = bind(qbx, bdry_op_sym)

    # {{{ fix rhs and solve

    bvp_rhs = bind(qbx, sqrt_w*sym.var("bc"))(queue, bc=f)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.float64),
            bvp_rhs, tol=gmres_tolerance, progress=True,
            stall_iterations=0,
            hard_failure=True)

    sigma = bind(qbx, sym.var("sigma")/sqrt_w)(queue, sigma=gmres_result.solution)

    # }}}

    debugging_info = {}
    debugging_info['gmres_result'] = gmres_result

    # {{{ postprocess

    repr_kwargs = dict(qbx_forced_limit=None)
    representation_sym = (
            sym.S(kernel, inv_sqrt_w_sigma, **repr_kwargs)
            + sym.D(kernel, inv_sqrt_w_sigma, **repr_kwargs))

    qbx_stick_out = qbx.copy(target_stick_out_factor=target_association_tolerance)

    ext_f = bind(
            (qbx_stick_out, target_discr),
            representation_sym)(queue, sigma=sigma).real

    # }}}

    return ext_f, debugging_info


# }}} End harmonic extension

# {{{ biharmonic extension

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


def compute_biharmonic_extension(queue, target_discr,
        qbx, density_discr, f, fx, fy,
        target_association_tolerance=0.05):
    """Biharmoc extension. Currently only support
    interior domains in 2D (i.e., extension is on the exterior).
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
    # logger.info("Fix all negative signs in:" +
    #        str(xp * tangent[0] + yp * tangent[1]))

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
    # See logger.info(str(int_area) + " " + str(area_exact))

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

# }}} End biharmonic extension
