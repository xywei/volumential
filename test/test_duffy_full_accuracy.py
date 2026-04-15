import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import pyopencl as cl

from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    LaplaceKernel,
    YukawaKernel,
)
from sumpy.point_calculus import CalculusPatch

import volumential.nearfield_potential_table as npt


def _get_gpu_queue_or_skip(ctx_factory):
    ctx = ctx_factory()
    if not any(dev.type & cl.device_type.GPU for dev in ctx.devices):
        pytest.skip("full-accuracy derivative checks run on GPU contexts only")
    return cl.CommandQueue(ctx)


def _pick_far_positive_case_id(table):
    case_vecs = np.asarray(table.interaction_case_vecs, dtype=np.int64)
    positive_ids = [i for i, vec in enumerate(case_vecs) if np.all(vec > 0)]
    if not positive_ids:
        positive_ids = list(range(len(case_vecs)))
    return max(positive_ids, key=lambda i: int(np.dot(case_vecs[i], case_vecs[i])))


def _tensor_box_integral_real(dim, order, func):
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
                "expected negligible imaginary residue in reference integrand "
                f"values, got max imag {imag_max:.3e}"
            )
        values = np.real(values)

    values = np.asarray(values, dtype=np.float64)
    return float(np.sum(weights * values))


def _target_x_derivative_via_calculus_patch(potential_at, target, h):
    cpatch = CalculusPatch(
        center=np.array([target[0]], dtype=np.float64),
        h=float(h),
        order=4,
    )

    values = np.empty(cpatch.points.shape[1], dtype=np.float64)
    for i, x in enumerate(cpatch.x):
        shifted_target = np.array(target, copy=True)
        shifted_target[0] = x
        values[i] = potential_at(shifted_target)

    deriv_values = cpatch.dx(values)
    center_index = int(np.argmin(np.abs(cpatch.x - target[0])))
    return float(deriv_values[center_index])


def _exp_bump_profile(*coords):
    centers = [0.37, 0.61, 0.28]
    alphas = [8.0, 7.0, 5.0]

    r2 = 0.0
    for i, coord in enumerate(coords):
        r2 = r2 + alphas[i] * (coord - centers[i]) ** 2

    return np.exp(-r2)


def _contract_table_for_source_values(
    table, source_values, target_point_index, case_id
):
    out = 0.0
    for source_mode_index, value in enumerate(source_values):
        entry_id = table.get_entry_index(source_mode_index, target_point_index, case_id)
        out = out + value * float(table.data[entry_id])
    return float(out)


def _make_interpolated_source_callable(table, source_values):
    modes = [table.get_mode(i) for i in range(table.n_q_points)]

    def source_interp(*coords):
        out = np.zeros_like(np.asarray(coords[0], dtype=np.float64))
        for mode_value, mode in zip(source_values, modes):
            out = out + mode_value * mode(*coords)
        return out

    return source_interp


@pytest.mark.full_accuracy
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kernel_family", ["laplace", "yukawa"])
def test_duffy_batched_derivative_full_accuracy_with_exponential_bump(
    ctx_factory,
    dim,
    kernel_family,
):
    queue = _get_gpu_queue_or_skip(ctx_factory)

    if dim == 2:
        q_order = 4
        regular_quad_order = 12
        radial_quad_order = 61
        reference_quad_order = 220
        fd_h = 2.0e-4
    else:
        q_order = 3
        regular_quad_order = 10
        radial_quad_order = 41
        reference_quad_order = 90
        fd_h = 5.0e-4

    kernel_kwargs = {}
    if kernel_family == "laplace":
        base_knl = LaplaceKernel(dim)
    elif kernel_family == "yukawa":
        base_knl = YukawaKernel(dim)
        kernel_kwargs[base_knl.yukawa_lambda_name] = 1.3
    else:
        raise ValueError(f"unsupported kernel_family: {kernel_family}")

    target_knl = AxisTargetDerivative(0, base_knl)
    source_knl = AxisSourceDerivative(0, base_knl)

    base_func = npt.sumpy_kernel_to_lambda(
        base_knl,
        parameter_values=kernel_kwargs if kernel_kwargs else None,
    )
    target_func = npt.sumpy_kernel_to_lambda(
        target_knl,
        parameter_values=kernel_kwargs if kernel_kwargs else None,
    )
    source_func = npt.sumpy_kernel_to_lambda(
        source_knl,
        parameter_values=kernel_kwargs if kernel_kwargs else None,
    )

    target_table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=target_func,
        kernel_type="rigid",
        sumpy_kernel=target_knl,
        progress_bar=False,
    )
    source_table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=source_func,
        kernel_type="rigid",
        sumpy_kernel=source_knl,
        progress_bar=False,
    )

    target_table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        **kernel_kwargs,
    )
    source_table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        **kernel_kwargs,
    )

    case_id = _pick_far_positive_case_id(target_table)
    target_point_index = target_table.n_q_points // 2
    target = np.asarray(
        target_table.find_target_point(target_point_index, case_id), dtype=np.float64
    )

    source_values = np.asarray(
        [_exp_bump_profile(*point) for point in target_table.q_points], dtype=np.float64
    )

    target_table_value = _contract_table_for_source_values(
        target_table, source_values, target_point_index, case_id
    )
    source_table_value = _contract_table_for_source_values(
        source_table, source_values, target_point_index, case_id
    )

    source_interp = _make_interpolated_source_callable(target_table, source_values)

    def potential_at(target_point):
        return _tensor_box_integral_real(
            dim,
            reference_quad_order,
            lambda *coords: (
                source_interp(*coords)
                * base_func(*[coords[i] - target_point[i] for i in range(dim)])
            ),
        )

    fd_target_derivative = _target_x_derivative_via_calculus_patch(
        potential_at,
        target,
        h=fd_h,
    )

    rel_target_error = min(
        abs(target_table_value - fd_target_derivative),
        abs(target_table_value + fd_target_derivative),
    ) / max(1.0, abs(fd_target_derivative))
    rel_source_error = min(
        abs(source_table_value - fd_target_derivative),
        abs(source_table_value + fd_target_derivative),
    ) / max(1.0, abs(fd_target_derivative))

    antisymmetry_error = abs(target_table_value + source_table_value) / max(
        1.0,
        max(abs(target_table_value), abs(source_table_value)),
    )

    assert rel_target_error < 1.0e-8
    assert rel_source_error < 1.0e-8
    assert antisymmetry_error < 1.0e-10
