import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from pymbolic import var

import pyopencl as cl

from sumpy.point_calculus import CalculusPatch
from sumpy.kernel import (
    AxisTargetDerivative,
    ExpressionKernel,
    LaplaceKernel,
    YukawaKernel,
)

import volumential.nearfield_potential_table as npt
from volumential.table_manager import ConstantKernel


class _Laplace1DKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self):
        d0 = var("d0")
        super().__init__(
            1,
            expression=-0.5 * (d0 * d0) ** 0.5,
            global_scaling_const=1,
        )

    @property
    def is_complex_valued(self):
        return False

    mapper_method = "map_expression_kernel"


class _HelmholtzPlaneWaveKernel(ExpressionKernel):
    init_arg_names = ("dim", "k")

    def __init__(self, dim, k):
        self.k = float(k)
        d0 = var("d0")
        super().__init__(
            dim, expression=var("cos")(self.k * d0), global_scaling_const=1
        )

    @property
    def is_complex_valued(self):
        return False

    mapper_method = "map_expression_kernel"

    def __getinitargs__(self):
        return (self.dim, self.k)


def _get_gpu_queue_or_skip(ctx_factory):
    ctx = ctx_factory()
    if not any(dev.type & cl.device_type.GPU for dev in ctx.devices):
        pytest.skip("manufactured batched checks run on GPU contexts only")
    return cl.CommandQueue(ctx)


def _pick_far_positive_case_id(table):
    case_vecs = np.asarray(table.interaction_case_vecs, dtype=np.int64)
    positive_ids = [i for i, vec in enumerate(case_vecs) if np.all(vec > 0)]
    if not positive_ids:
        positive_ids = list(range(len(case_vecs)))
    return max(positive_ids, key=lambda i: int(np.dot(case_vecs[i], case_vecs[i])))


def _tensor_box_integral(dim, order, func):
    nodes_1d, weights_1d = leggauss(order)
    nodes_1d = 0.5 * (nodes_1d + 1.0)
    weights_1d = 0.5 * weights_1d

    grids = np.meshgrid(*([nodes_1d] * dim), indexing="ij")
    wgrids = np.meshgrid(*([weights_1d] * dim), indexing="ij")
    weights = np.ones_like(grids[0])
    for wg in wgrids:
        weights *= wg

    values = np.asarray(func(*grids))
    if np.iscomplexobj(values):
        imag_max = float(np.max(np.abs(np.imag(values)))) if values.size else 0.0
        real_scale = (
            max(1.0, float(np.max(np.abs(np.real(values))))) if values.size else 1.0
        )
        imag_tol = 256.0 * np.finfo(np.float64).eps * real_scale
        if imag_max > imag_tol:
            raise AssertionError(
                "expected negligible imaginary residue in manufactured "
                f"integrand values, got max imag {imag_max:.3e}"
            )
        values = np.real(values)

    values = np.asarray(values, dtype=np.float64)
    return float(np.sum(weights * values))


def _target_x_derivative_via_calculus_patch(potential_at, target):
    cpatch = CalculusPatch(
        center=np.array([target[0]], dtype=np.float64),
        h=1.0e-3,
        order=4,
    )
    values = np.empty(cpatch.points.shape[1], dtype=np.float64)
    for i, x in enumerate(cpatch.x):
        shifted_target = np.array(target, copy=True)
        shifted_target[0] = x
        values[i] = potential_at(shifted_target)

    deriv_values = cpatch.dx(values)
    center_index = int(np.argmin(np.abs(cpatch.x - target[0])))
    center_value = float(deriv_values[center_index])

    try:
        centered = float(cpatch.eval_at_center(deriv_values))
    except Exception:
        return center_value

    if not np.isfinite(centered):
        return center_value
    if not np.isclose(centered, center_value, rtol=1.0e-3, atol=1.0e-14):
        return center_value

    return centered


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_duffy_batched_gpu_constant_kernel_matches_exact_volume(ctx_factory, dim):
    queue = _get_gpu_queue_or_skip(ctx_factory)

    table = npt.NearFieldInteractionTable(
        quad_order=1,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=npt.constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(dim),
        progress_bar=False,
    )
    table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=6,
        radial_quad_order=21,
    )

    exact = table.source_box_extent**dim
    for case_id in range(table.n_cases):
        entry_id = table.get_entry_index(0, 0, case_id)
        rel_err = abs(table.data[entry_id] - exact) / max(1.0, abs(exact))
        assert rel_err < 1e-11


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_duffy_batched_gpu_laplace_derivative_matches_finite_difference(
    ctx_factory, dim
):
    queue = _get_gpu_queue_or_skip(ctx_factory)

    if dim == 1:
        laplace_knl = _Laplace1DKernel()
        laplace_func = lambda x: -0.5 * np.abs(np.asarray(x))
        derivative_func = lambda x: 0.5 * np.sign(np.asarray(x))
        regular_quad_order = 6
        radial_quad_order = 21
        legendre_order = 250
    else:
        laplace_knl = LaplaceKernel(dim)
        laplace_func = npt.sumpy_kernel_to_lambda(laplace_knl)
        derivative_func = npt.sumpy_kernel_to_lambda(
            AxisTargetDerivative(0, laplace_knl)
        )
        regular_quad_order = 8 if dim == 2 else 6
        radial_quad_order = 31 if dim == 2 else 21
        legendre_order = 80 if dim == 2 else 30

    laplace_dx_knl = AxisTargetDerivative(0, laplace_knl)

    table_dx = npt.NearFieldInteractionTable(
        quad_order=1,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=derivative_func,
        kernel_type="rigid",
        sumpy_kernel=laplace_dx_knl,
        progress_bar=False,
    )
    table_dx.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
    )

    case_id = _pick_far_positive_case_id(table_dx)
    entry_id = table_dx.get_entry_index(0, 0, case_id)
    target = np.asarray(table_dx.find_target_point(0, case_id), dtype=np.float64)

    def potential_at(target_point):
        return _tensor_box_integral(
            dim,
            legendre_order,
            lambda *coords: laplace_func(
                *[coords[i] - target_point[i] for i in range(dim)]
            ),
        )

    fd_target_derivative = _target_x_derivative_via_calculus_patch(
        potential_at,
        target,
    )

    # AxisTargetDerivative in this table setup carries the opposite sign
    # of d/d(target_x) applied to the box integral assembled above.
    rel_err = abs(table_dx.data[entry_id] + fd_target_derivative) / max(
        1.0, abs(fd_target_derivative)
    )
    assert rel_err < 1e-8


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_duffy_batched_gpu_helmholtz_plane_wave_matches_exact_values(ctx_factory, dim):
    queue = _get_gpu_queue_or_skip(ctx_factory)

    k = 1.7
    helmholtz_knl = _HelmholtzPlaneWaveKernel(dim, k)
    helmholtz_dx_knl = AxisTargetDerivative(0, helmholtz_knl)

    kernel_func = lambda x, *rest: np.cos(k * np.asarray(x))
    kernel_dx_func = lambda x, *rest: k * np.sin(k * np.asarray(x))

    table = npt.NearFieldInteractionTable(
        quad_order=1,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=kernel_func,
        kernel_type="rigid",
        sumpy_kernel=helmholtz_knl,
        progress_bar=False,
    )
    table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=6,
        radial_quad_order=21,
    )

    table_dx = npt.NearFieldInteractionTable(
        quad_order=1,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=kernel_dx_func,
        kernel_type="rigid",
        sumpy_kernel=helmholtz_dx_knl,
        progress_bar=False,
    )
    table_dx.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=6,
        radial_quad_order=21,
    )

    case_id = _pick_far_positive_case_id(table)
    entry_id = table.get_entry_index(0, 0, case_id)
    target_x = float(table.find_target_point(0, case_id)[0])

    exact_potential = (np.sin(k * (1.0 - target_x)) - np.sin(-k * target_x)) / k
    exact_target_derivative = np.cos(k * target_x) - np.cos(k * (1.0 - target_x))

    pot_rel_err = abs(table.data[entry_id] - exact_potential) / max(
        1.0, abs(exact_potential)
    )
    assert pot_rel_err < 1e-8

    # Same sign convention as in the Laplace derivative test above.
    dpot_rel_err = abs(table_dx.data[entry_id] + exact_target_derivative) / max(
        1.0, abs(exact_target_derivative)
    )
    assert dpot_rel_err < 1e-8


@pytest.mark.parametrize("dim", [2, 3])
def test_duffy_batched_gpu_yukawa_derivative_matches_finite_difference(
    ctx_factory, dim
):
    queue = _get_gpu_queue_or_skip(ctx_factory)

    lam = 1.3
    yukawa_knl = YukawaKernel(dim)
    yukawa_dx_knl = AxisTargetDerivative(0, yukawa_knl)
    lam_name = yukawa_knl.yukawa_lambda_name

    yukawa_func = npt.sumpy_kernel_to_lambda(
        yukawa_knl,
        parameter_values={lam_name: lam},
    )
    yukawa_dx_func = npt.sumpy_kernel_to_lambda(
        yukawa_dx_knl,
        parameter_values={lam_name: lam},
    )

    regular_quad_order = 8 if dim == 2 else 6
    radial_quad_order = 31 if dim == 2 else 21
    legendre_order = 80 if dim == 2 else 30

    table_dx = npt.NearFieldInteractionTable(
        quad_order=1,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=yukawa_dx_func,
        kernel_type="rigid",
        sumpy_kernel=yukawa_dx_knl,
        progress_bar=False,
    )
    table_dx.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        lam=lam,
    )

    case_id = _pick_far_positive_case_id(table_dx)
    entry_id = table_dx.get_entry_index(0, 0, case_id)
    target = np.asarray(table_dx.find_target_point(0, case_id), dtype=np.float64)

    def potential_at(target_point):
        return _tensor_box_integral(
            dim,
            legendre_order,
            lambda *coords: yukawa_func(
                *[coords[i] - target_point[i] for i in range(dim)]
            ),
        )

    fd_target_derivative = _target_x_derivative_via_calculus_patch(
        potential_at,
        target,
    )

    rel_err = abs(table_dx.data[entry_id] - fd_target_derivative) / max(
        1.0, abs(fd_target_derivative)
    )
    assert rel_err < 1e-7
