"""Function extension with regularity constraints:

1. :math:`L^2`: extend with constant value

.. autofunction:: compute_constant_extension

2. :math:`C^0`: harmonic extension

.. autofunction:: compute_harmonic_extension

3. :math:`C^1`: biharmonic extension

.. autofunction:: compute_biharmonic_extension

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
from arraycontext import flatten
from meshmode.dof_array import DOFArray
from pymbolic import var
from pytential import GeometryCollection, bind, sym
from pytential.array_context import PyOpenCLArrayContext
from pytential.linalg.gmres import gmres
from pytential.target import PointsTarget
from pytential.symbolic.stokes import StokesletWrapper, StressletWrapper
from pytools.obj_array import new_1d as obj_array_1d
from sumpy.kernel import AxisTargetDerivative, ExpressionKernel


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


def setup_array_context(actx=None, ctx=None, queue=None):
    if actx is not None:
        return actx

    queue = setup_command_queue(ctx=ctx, queue=queue)
    return PyOpenCLArrayContext(queue)


def _build_extension_places(target_geometry, qbx, target_association_tolerance):
    if not isinstance(target_geometry, PointsTarget):
        targets = target_geometry
    else:
        targets = target_geometry

    qbx_target_assoc = qbx.copy(
        target_association_tolerance=target_association_tolerance
    )

    places = GeometryCollection(
        {
            "qbx": qbx,
            "qbx_target_assoc": qbx_target_assoc,
            "targets": targets,
        },
        auto_where="qbx",
    )

    return places, qbx_target_assoc


def get_normal_vectors(queue, density_discr, loc_sign=-1):
    if loc_sign == 1:
        return -bind(density_discr, sym.normal(2).as_vector())(queue)
    elif loc_sign == -1:
        return bind(density_discr, sym.normal(2).as_vector())(queue)


def get_tangent_vectors(queue, density_discr, loc_sign):
    # the domain is on the left.
    normal = get_normal_vectors(queue, density_discr, loc_sign)
    return obj_array_1d([-1 * normal[1], normal[0]])


def get_path_length(queue, density_discr):
    return bind(density_discr, sym.integral(2, 1, 1))(queue)


def get_arclength_parametrization_derivative(queue, density_discr, where=None):
    """Return x'(s), y'(s). s Follows the same direction as the parametrization.
    (Remark: t is the reference-to-global parametrization of the discretization).
    """
    queue = setup_command_queue(queue=queue)
    dim = 2

    # xp = dx/dt, yp = dy/dt, sp = ds/dt
    xp, yp = sym.parametrization_derivative_matrix(dim, dim - 1, where)
    sp = (xp[0] ** 2 + yp[0] ** 2) ** (1 / 2)

    xps = xp[0] / sp
    yps = yp[0] / sp

    return bind(density_discr, xps)(queue), bind(density_discr, yps)(queue)


# }}} End helper functions

# {{{ constant extension


def compute_constant_extension(
    queue,
    target_discr,
    qbx,
    density_discr,
    constant_val=0,
    actx=None,
):
    """Constant extension."""
    actx = setup_array_context(actx=actx, queue=queue)
    places, _ = _build_extension_places(target_discr, qbx, 0.05)
    debugging_info = {}

    ext_f = bind(
        places,
        sym.var("sigma") * 0 + constant_val,
        auto_where=("qbx", "targets"),
    )(actx, sigma=1).real

    return ext_f, debugging_info


# }}} End constant extension

# {{{ harmonic extension


def compute_harmonic_extension(
    queue,
    target_discr,
    qbx,
    density_discr,
    f,
    loc_sign=1,
    representation_mode="auto",
    target_association_tolerance=0.05,
    gmres_tolerance=1e-14,
    actx=None,
):
    """Harmonic extension.

    :param loc_sign: Indicates the domain for extension,
        which equals the negation of the loc_sign for the original problem.
    :param representation_mode: One of "auto", "s_plus_d", "d_only".
        "auto" uses D-only for interior extension (loc_sign=-1) and S+D
        otherwise.
    """
    dim = qbx.ambient_dim
    actx = setup_array_context(actx=actx, queue=queue)
    places, qbx_stick_out = _build_extension_places(
        target_discr, qbx, target_association_tolerance
    )

    # {{{ describe bvp

    from sumpy.kernel import LaplaceKernel

    kernel = LaplaceKernel(dim)

    cse = sym.cse

    sigma_sym = sym.var("sigma")
    # sqrt_w = sym.sqrt_jac_q_weight(3)
    sqrt_w = 1
    inv_sqrt_w_sigma = cse(sigma_sym / sqrt_w)

    if representation_mode == "auto":
        use_d_only = loc_sign == -1
    elif representation_mode == "d_only":
        use_d_only = True
    elif representation_mode == "s_plus_d":
        use_d_only = False
    else:
        raise ValueError(
            "representation_mode must be one of 'auto', 's_plus_d', 'd_only'"
        )

    if use_d_only:
        bdry_op_sym = loc_sign * 0.5 * sigma_sym + sqrt_w * sym.D(
            kernel, inv_sqrt_w_sigma, qbx_forced_limit="avg"
        )
    else:
        bdry_op_sym = loc_sign * 0.5 * sigma_sym + sqrt_w * (
            sym.S(kernel, inv_sqrt_w_sigma, qbx_forced_limit="avg")
            + sym.D(kernel, inv_sqrt_w_sigma, qbx_forced_limit="avg")
        )

    # }}}

    bound_op = bind(places, bdry_op_sym, auto_where=("qbx", "qbx"))

    # {{{ fix rhs and solve

    bvp_rhs = bind(places, sqrt_w * sym.var("bc"), auto_where=("qbx", "qbx"))(
        actx, bc=f
    )

    gmres_result = gmres(
        bound_op.scipy_op(actx, "sigma", dtype=np.float64),
        bvp_rhs,
        tol=gmres_tolerance,
        progress=True,
        stall_iterations=0,
        hard_failure=True,
    )

    sigma = bind(places, sym.var("sigma") / sqrt_w, auto_where=("qbx", "qbx"))(
        actx, sigma=gmres_result.solution
    )

    # }}}

    debugging_info = {}
    debugging_info["gmres_result"] = gmres_result

    # {{{ postprocess

    repr_kwargs = {"qbx_forced_limit": None}
    if use_d_only:
        representation_sym = sym.D(kernel, inv_sqrt_w_sigma, **repr_kwargs)
    else:
        representation_sym = sym.S(kernel, inv_sqrt_w_sigma, **repr_kwargs) + sym.D(
            kernel, inv_sqrt_w_sigma, **repr_kwargs
        )

    debugging_info["qbx"] = qbx_stick_out
    debugging_info["places"] = places
    debugging_info["representation"] = representation_sym
    debugging_info["representation_mode"] = "d_only" if use_d_only else "s_plus_d"
    debugging_info["density"] = sigma

    ext_f = bind(
        places,
        representation_sym,
        auto_where=("qbx_target_assoc", "targets"),
    )(actx, sigma=sigma).real

    # }}}

    # NOTE: matching is needed if using
    # pytential.symbolic.pde.scalar.DirichletOperator
    # but here we are using a representation that does not have null
    # space for exterior Dirichlet problem
    if loc_sign == 1 and False:
        bdry_measure = bind(density_discr, sym.integral(dim, dim - 1, 1))(queue)

        int_func_bdry = bind(qbx, sym.integral(dim, dim - 1, var("integrand")))(
            queue, integrand=f
        )

        solu_bdry = bind((qbx, density_discr), representation_sym)(
            queue, sigma=sigma
        ).real
        int_solu_bdry = bind(qbx, sym.integral(dim, dim - 1, var("integrand")))(
            queue, integrand=solu_bdry
        )

        matching_const = (int_func_bdry - int_solu_bdry) / bdry_measure

    else:
        matching_const = 0.0

    ext_f = ext_f + matching_const

    def eval_ext_f(target_discr):
        eval_places = places.merge({"targets_eval": target_discr})
        return (
            bind(
                eval_places,
                representation_sym,
                auto_where=("qbx_target_assoc", "targets_eval"),
            )(actx, sigma=sigma).real
            + matching_const
        )

    debugging_info["eval_ext_f"] = eval_ext_f

    return ext_f, debugging_info


# }}} End harmonic extension

# {{{ biharmonic extension

# {{{ Goursat kernels


class ComplexLogKernel(ExpressionKernel):
    init_arg_names = ("dim",)
    is_complex_valued = True

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            conj_z = d[0] - var("I") * d[1]
            r = var("sqrt")(np.dot(conj_z, z))
            expr = var("log")(r)
            scaling = 1 / (4 * var("pi"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

    has_efficient_scale_adjustment = True

    def adjust_for_kernel_scaling(self, expr, rscale, nderivatives):
        """Efficient rescaling."""

        if self.dim == 2:
            if nderivatives == 0:
                # return expr + var("log")(rscale)
                import sumpy.symbolic as sp

                return expr + sp.log(rscale)
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
    is_complex_valued = True
    has_efficient_scale_adjustment = False

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            conj_z = d[0] - var("I") * d[1]
            r = var("sqrt")(np.dot(conj_z, z))
            expr = conj_z * var("log")(r)
            scaling = -1 / (4 * var("pi"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxLinLogKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


class ComplexLinearKernel(ExpressionKernel):
    init_arg_names = ("dim",)
    is_complex_valued = True
    has_efficient_scale_adjustment = False

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            expr = z
            scaling = -1 / (8 * var("pi"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

    has_efficient_scale_adjustment = True

    def adjust_for_kernel_scaling(self, expr, rscale, nderivatives):
        """Efficient rescaling of the kernel."""
        if self.dim == 2:
            return rscale * expr

        else:
            raise NotImplementedError("unsupported dimensionality")

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CplxLinKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


class ComplexFractionalKernel(ExpressionKernel):
    init_arg_names = ("dim",)
    is_complex_valued = True
    has_efficient_scale_adjustment = False

    def __init__(self, dim=None):
        if dim == 2:
            d = sym.make_sym_vector("d", dim)
            z = d[0] + var("I") * d[1]
            conj_z = d[0] - var("I") * d[1]
            expr = conj_z / z
            scaling = 1 / (4 * var("pi") * var("I"))
        else:
            raise NotImplementedError("unsupported dimensionality")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

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
        - stresslet_obj.apply(sigma_sym, nvec_sym, mu_sym, qbx_forced_limit="avg")
        - stokeslet_obj.apply(sigma_sym, mu_sym, qbx_forced_limit="avg")
        + int_sigma
    )

    return bdry_op_sym


def compute_biharmonic_extension(
    queue,
    target_discr,
    qbx,
    density_discr,
    f,
    fx,
    fy,
    loc_sign=1,
    target_association_tolerance=0.05,
    enforce_affine_match=False,
    actx=None,
):
    """Biharmoc extension. Currently only support
    2D domains.

    :param loc_sign: Side for extension evaluation, following the same
        convention used by pytential layer potentials.
        ``+1`` evaluates the exterior side and ``-1`` evaluates the
        interior side.
    :param enforce_affine_match: If *True*, perform a boundary least-squares
        affine correction.
    """
    # pylint: disable=invalid-unary-operand-type
    dim = 2
    actx = setup_array_context(actx=actx, queue=queue)
    queue = setup_command_queue(queue=queue)

    if loc_sign not in (-1, 1):
        raise ValueError("loc_sign must be -1 or +1")

    qbx_forced_limit = loc_sign

    normal = get_normal_vectors(actx, density_discr, loc_sign=loc_sign)

    bdry_op_sym = get_extension_bie_symbolic_operator(loc_sign=loc_sign)
    bound_op = bind(qbx, bdry_op_sym)

    bc = [fy, -fx]
    bvp_rhs = bind(qbx, sym.make_sym_vector("bc", dim))(actx, bc=bc)
    gmres_result_mu = gmres(
        bound_op.scipy_op(actx, "sigma", np.float64, mu=1.0, normal=normal),
        bvp_rhs,
        tol=1e-9,
        progress=True,
        stall_iterations=0,
        hard_failure=True,
    )
    mu = gmres_result_mu.solution

    arclength_parametrization_derivatives_sym = sym.make_sym_vector(
        "arclength_parametrization_derivatives", dim
    )
    density_mu_sym = sym.make_sym_vector("mu", dim)
    imag_unit = var("I")
    imag_kwargs = {"I": np.complex128(1j)}
    dxids_sym = (
        arclength_parametrization_derivatives_sym[0]
        + imag_unit * arclength_parametrization_derivatives_sym[1]
    )
    dxids_conj_sym = (
        arclength_parametrization_derivatives_sym[0]
        - imag_unit * arclength_parametrization_derivatives_sym[1]
    )
    density_rho_sym = density_mu_sym[1] - imag_unit * density_mu_sym[0]
    density_conj_rho_sym = density_mu_sym[1] + imag_unit * density_mu_sym[0]

    cplx_log_knl = ComplexLogKernel(dim)
    cplx_lin_log_knl = ComplexLinearLogKernel(dim)
    cplx_lin_knl = ComplexLinearKernel(dim)
    cplx_frac_knl = ComplexFractionalKernel(dim)

    # convolutions
    GS1 = sym.IntG(  # noqa: N806
        cplx_lin_log_knl,
        [cplx_lin_log_knl],
        [density_rho_sym],
        qbx_forced_limit=None,
    )
    GS2 = sym.IntG(  # noqa: N806
        cplx_lin_knl,
        [cplx_lin_knl],
        [density_conj_rho_sym],
        qbx_forced_limit=None,
    )
    GD1 = sym.IntG(  # noqa: N806
        cplx_frac_knl,
        [cplx_frac_knl],
        [density_rho_sym * dxids_sym],
        qbx_forced_limit=None,
    )
    GD2 = [
        sym.IntG(  # noqa: N806
            AxisTargetDerivative(iaxis, ComplexLogKernel(dim)),
            [cplx_log_knl],
            [density_conj_rho_sym * dxids_sym + density_rho_sym * dxids_conj_sym],
            qbx_forced_limit=qbx_forced_limit,
        )
        for iaxis in range(dim)
    ]

    GS1_bdry = sym.IntG(  # noqa: N806
        cplx_lin_log_knl,
        [cplx_lin_log_knl],
        [density_rho_sym],
        qbx_forced_limit=qbx_forced_limit,
    )
    GS2_bdry = sym.IntG(  # noqa: N806
        cplx_lin_knl,
        [cplx_lin_knl],
        [density_conj_rho_sym],
        qbx_forced_limit=qbx_forced_limit,
    )
    GD1_bdry = sym.IntG(  # noqa: N806
        cplx_frac_knl,
        [cplx_frac_knl],
        [density_rho_sym * dxids_sym],
        qbx_forced_limit=qbx_forced_limit,
    )

    xp, yp = get_arclength_parametrization_derivative(actx, density_discr)
    xp = -xp
    yp = -yp
    tangent = get_tangent_vectors(actx, density_discr, loc_sign=qbx_forced_limit)

    # check and fix the direction of parametrization
    # logger.info("Fix all negative signs in:" +
    #        str(xp * tangent[0] + yp * tangent[1]))

    grad_v2 = [
        bind(qbx, GD2[iaxis])(
            actx,
            mu=mu,
            arclength_parametrization_derivatives=obj_array_1d([xp, yp]),
            **imag_kwargs,
        ).real
        for iaxis in range(dim)
    ]
    v2_tangent_der = sum(tangent[iaxis] * grad_v2[iaxis] for iaxis in range(dim))

    from pytential.symbolic.pde.scalar import NeumannOperator
    from sumpy.kernel import LaplaceKernel

    operator_v1 = NeumannOperator(LaplaceKernel(dim), loc_sign=qbx_forced_limit)
    bound_op_v1 = bind(qbx, operator_v1.operator(var("sigma")))
    # Sign convention: v1 solves a Neumann problem that cancels the tangential
    # derivative contribution from v2 on the extension side.
    rhs_v1 = operator_v1.prepare_rhs(-1 * v2_tangent_der)
    gmres_result_sigma = gmres(
        bound_op_v1.scipy_op(actx, "sigma", dtype=np.float64),
        rhs_v1,
        tol=1e-9,
        progress=True,
        stall_iterations=0,
        hard_failure=True,
    )
    sigma = gmres_result_sigma.solution
    qbx_stick_out = qbx.copy(target_association_tolerance=target_association_tolerance)
    v1 = bind(
        (qbx_stick_out, target_discr),
        operator_v1.representation(var("sigma"), qbx_forced_limit=None),
    )(actx, sigma=sigma)
    grad_v1 = bind(
        (qbx_stick_out, target_discr),
        operator_v1.representation(
            var("sigma"),
            qbx_forced_limit=None,
            map_potentials=lambda pot: sym.grad(dim, pot),
        ),
    )(actx, sigma=sigma)
    v1_bdry = bind(
        qbx, operator_v1.representation(var("sigma"), qbx_forced_limit=qbx_forced_limit)
    )(actx, sigma=sigma)

    target_nodes = actx.thaw(target_discr.nodes())
    density_nodes = actx.thaw(density_discr.nodes())

    z_conj = target_nodes[0] - 1j * target_nodes[1]
    z_conj_bdry = density_nodes[0] - 1j * density_nodes[1]
    int_rho = (
        1
        / (8 * np.pi)
        * bind(qbx, sym.integral(dim, dim - 1, density_rho_sym))(
            actx, mu=mu, **imag_kwargs
        )
    )

    omega_S1 = bind(  # noqa: N806
        (qbx_stick_out, target_discr), GS1
    )(actx, mu=mu, **imag_kwargs).real
    omega_S2 = (
        -1
        * bind(  # noqa: N806
            (qbx_stick_out, target_discr), GS2
        )(actx, mu=mu, **imag_kwargs).real
    )
    omega_S3 = (z_conj * int_rho).real  # noqa: N806
    omega_S = -(omega_S1 + omega_S2 + omega_S3)  # noqa: N806

    grad_omega_S1 = bind(  # noqa: N806
        (qbx_stick_out, target_discr), sym.grad(dim, GS1)
    )(actx, mu=mu, **imag_kwargs).real
    grad_omega_S2 = (
        -1
        * bind(  # noqa: N806
            (qbx_stick_out, target_discr), sym.grad(dim, GS2)
        )(actx, mu=mu, **imag_kwargs).real
    )
    grad_omega_S3 = obj_array_1d([int_rho.real, int_rho.imag])  # noqa: N806
    grad_omega_S = -(grad_omega_S1 + grad_omega_S2 + grad_omega_S3)  # noqa: N806

    omega_S1_bdry = bind(qbx, GS1_bdry)(actx, mu=mu, **imag_kwargs).real  # noqa: N806
    omega_S2_bdry = (  # noqa: N806
        -1 * bind(qbx, GS2_bdry)(actx, mu=mu, **imag_kwargs).real
    )
    omega_S3_bdry = (z_conj_bdry * int_rho).real  # noqa: N806
    omega_S_bdry = -(omega_S1_bdry + omega_S2_bdry + omega_S3_bdry)  # noqa: N806

    omega_D1 = bind(  # noqa: N806
        (qbx_stick_out, target_discr), GD1
    )(
        actx,
        mu=mu,
        arclength_parametrization_derivatives=obj_array_1d([xp, yp]),
        **imag_kwargs,
    ).real
    omega_D = omega_D1 + v1  # noqa: N806

    grad_omega_D1 = bind(  # noqa: N806
        (qbx_stick_out, target_discr), sym.grad(dim, GD1)
    )(
        actx,
        mu=mu,
        arclength_parametrization_derivatives=obj_array_1d([xp, yp]),
        **imag_kwargs,
    ).real
    grad_omega_D = grad_omega_D1 + grad_v1  # noqa: N806

    omega_D1_bdry = bind(  # noqa: N806
        qbx, GD1_bdry
    )(
        actx,
        mu=mu,
        arclength_parametrization_derivatives=obj_array_1d([xp, yp]),
        **imag_kwargs,
    ).real
    omega_D_bdry = omega_D1_bdry + v1_bdry  # noqa: N806

    int_bdry_mu = bind(qbx, sym.integral(dim, dim - 1, sym.make_sym_vector("mu", dim)))(
        actx, mu=mu
    )
    omega_W = (  # noqa: N806
        int_bdry_mu[0] * target_nodes[1] - int_bdry_mu[1] * target_nodes[0]
    )
    grad_omega_W = obj_array_1d(  # noqa: N806
        [-int_bdry_mu[1], int_bdry_mu[0]]
    )
    omega_W_bdry = (  # noqa: N806
        int_bdry_mu[0] * density_nodes[1] - int_bdry_mu[1] * density_nodes[0]
    )

    int_bdry = bind(qbx, sym.integral(dim, dim - 1, var("integrand")))(
        actx, integrand=omega_S_bdry + omega_D_bdry + omega_W_bdry
    )

    debugging_info = {}
    debugging_info["gmres_result_mu"] = gmres_result_mu
    debugging_info["gmres_result_sigma"] = gmres_result_sigma
    debugging_info["omega_S"] = omega_S
    debugging_info["omega_D"] = omega_D
    debugging_info["omega_W"] = omega_W
    debugging_info["omega_v1"] = v1
    debugging_info["omega_D1"] = omega_D1

    int_interior_func_bdry = bind(qbx, sym.integral(2, 1, var("integrand")))(
        actx, integrand=f
    )

    path_length = get_path_length(actx, density_discr)
    matching_const = (int_interior_func_bdry - int_bdry) / path_length

    ext_f = omega_S + omega_D + omega_W + matching_const
    grad_f = grad_omega_S + grad_omega_D + grad_omega_W

    if enforce_affine_match:
        ext_bdry = omega_S_bdry + omega_D_bdry + omega_W_bdry + matching_const
        residual_flat = flatten((f - ext_bdry).real, actx, leaf_class=DOFArray)
        x_bdry_flat = flatten(density_nodes[0], actx, leaf_class=DOFArray)
        y_bdry_flat = flatten(density_nodes[1], actx, leaf_class=DOFArray)

        residual_host = actx.to_numpy(residual_flat)
        x_bdry_host = actx.to_numpy(x_bdry_flat)
        y_bdry_host = actx.to_numpy(y_bdry_flat)

        design = np.vstack([x_bdry_host, y_bdry_host, np.ones_like(x_bdry_host)]).T
        coeffs, *_ = np.linalg.lstsq(design, residual_host, rcond=None)
        affine_a, affine_b, affine_c = coeffs

        ext_f = (
            ext_f + affine_a * target_nodes[0] + affine_b * target_nodes[1] + affine_c
        )
        grad_f = grad_f + obj_array_1d([affine_a, affine_b])

        ext_bdry_corrected = ext_bdry + (
            affine_a * density_nodes[0] + affine_b * density_nodes[1] + affine_c
        )

        debugging_info["affine_coeffs"] = (affine_a, affine_b, affine_c)
        debugging_info["boundary_residual_before"] = residual_flat
        debugging_info["boundary_residual_after"] = flatten(
            (f - ext_bdry_corrected).real, actx, leaf_class=DOFArray
        )
    else:
        debugging_info["affine_coeffs"] = (0.0, 0.0, 0.0)

    return ext_f.real, grad_f[0].real, grad_f[1].real, debugging_info


# }}} End biharmonic extension
