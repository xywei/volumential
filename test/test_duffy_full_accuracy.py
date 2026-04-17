import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import pyopencl as cl

from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    HelmholtzKernel,
    LaplaceKernel,
    YukawaKernel,
)
from sumpy.point_calculus import CalculusPatch

import volumential.nearfield_potential_table as npt


_FP64_GPU_QUEUE_CACHE = {}


def _get_fp64_gpu_queue_or_skip(*, require_non_intel=False):
    cache_key = bool(require_non_intel)
    cached = _FP64_GPU_QUEUE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    for platform in platforms:
        if require_non_intel and platform.name == "Intel(R) OpenCL":
            continue

        for dev in platform.get_devices():
            if not (dev.type & cl.device_type.GPU):
                continue

            extensions = getattr(dev, "extensions", "")
            has_khr_fp64 = "cl_khr_fp64" in extensions.split()
            has_double_config = bool(getattr(dev, "double_fp_config", 0))
            if not (has_khr_fp64 or has_double_config):
                continue

            queue = cl.CommandQueue(cl.Context([dev]))
            _FP64_GPU_QUEUE_CACHE[cache_key] = queue
            return queue

    if require_non_intel:
        pytest.skip("No non-Intel GPU OpenCL device with fp64 support available")
    pytest.skip("No GPU OpenCL device with fp64 support available")


def _get_non_intel_gpu_queue_or_skip():
    return _get_fp64_gpu_queue_or_skip(require_non_intel=True)


def _pick_far_positive_case_id(table):
    case_vecs = np.asarray(table.interaction_case_vecs, dtype=np.int64)
    positive_ids = [i for i, vec in enumerate(case_vecs) if np.all(vec > 0)]
    if not positive_ids:
        positive_ids = list(range(len(case_vecs)))
    return max(positive_ids, key=lambda i: int(np.dot(case_vecs[i], case_vecs[i])))


def _tensor_box_integral_real(dim, order, func, *, box_extent=1.0):
    nodes_1d, weights_1d = leggauss(order)
    scale = float(box_extent)
    nodes_1d = 0.5 * scale * (nodes_1d + 1.0)
    weights_1d = 0.5 * scale * weights_1d

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


def _tensor_box_integral_complex(dim, order, func, *, box_extent=1.0):
    nodes_1d, weights_1d = leggauss(order)
    scale = float(box_extent)
    nodes_1d = 0.5 * scale * (nodes_1d + 1.0)
    weights_1d = 0.5 * scale * weights_1d

    grids = np.meshgrid(*([nodes_1d] * dim), indexing="ij")
    wgrids = np.meshgrid(*([weights_1d] * dim), indexing="ij")
    weights = np.ones_like(grids[0], dtype=np.float64)
    for wg in wgrids:
        weights *= wg

    values = np.asarray(func(*grids), dtype=np.complex128)
    return np.sum(weights * values)


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


def _target_x_derivative_via_fd4(potential_at, target, h):
    center = np.array(target, copy=True)
    x0 = float(np.real(center[0]))
    step = float(h)

    evals = {}
    for shift in (-2.0, -1.0, 1.0, 2.0):
        shifted = np.array(center, copy=True)
        shifted[0] = x0 + shift * step
        evals[shift] = potential_at(shifted)

    return (-evals[2.0] + 8.0 * evals[1.0] - 8.0 * evals[-1.0] + evals[-2.0]) / (
        12.0 * step
    )


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
    table_dtype = np.dtype(getattr(table, "dtype", np.float64))
    out = np.array(0, dtype=table_dtype)
    for source_mode_index, value in enumerate(source_values):
        entry_id = table.get_entry_index(source_mode_index, target_point_index, case_id)
        out = out + value * table.data[entry_id]
    return out.item()


def _make_interpolated_source_callable(table, source_values):
    modes = [table.get_mode(i) for i in range(table.n_q_points)]
    table_dtype = np.dtype(getattr(table, "dtype", np.float64))
    out_dtype = (
        np.complex128 if np.issubdtype(table_dtype, np.complexfloating) else np.float64
    )

    def source_interp(*coords):
        out = np.zeros_like(np.asarray(coords[0], dtype=out_dtype))
        for mode_value, mode in zip(source_values, modes):
            out = out + mode_value * mode(*coords)
        return out

    return source_interp


def _source_box_extent_from_level(source_box_level):
    return float(2.0 ** (-int(source_box_level)))


def _exp_bump_profile_on_box(source_box_extent, *coords):
    scale = float(source_box_extent)
    scaled_coords = [np.asarray(coord, dtype=np.float64) / scale for coord in coords]
    return _exp_bump_profile(*scaled_coords)


def _full_accuracy_case_setup(kernel_family, dim):
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

    if kernel_family == "laplace":
        queue = _get_fp64_gpu_queue_or_skip()
        base_knl = LaplaceKernel(dim)
        kernel_kwargs = {}
        kernel_type = "rigid"
        table_dtype = np.float64
        integral = _tensor_box_integral_real
        reference_derivative = _target_x_derivative_via_calculus_patch
        rel_tol = 1.0e-8
    elif kernel_family == "yukawa":
        queue = _get_fp64_gpu_queue_or_skip()
        base_knl = YukawaKernel(dim)
        kernel_kwargs = {base_knl.yukawa_lambda_name: 1.3}
        kernel_type = "rigid"
        table_dtype = np.float64
        integral = _tensor_box_integral_real
        reference_derivative = _target_x_derivative_via_calculus_patch
        rel_tol = 1.0e-8
    elif kernel_family == "helmholtz":
        queue = _get_non_intel_gpu_queue_or_skip()
        base_knl = HelmholtzKernel(dim)
        kernel_kwargs = {base_knl.helmholtz_k_name: 1.3}
        kernel_type = "helmholtz-rigid"
        table_dtype = np.complex128
        integral = _tensor_box_integral_complex
        reference_derivative = _target_x_derivative_via_fd4
        rel_tol = 1.0e-6
    else:
        raise ValueError(f"unsupported kernel_family: {kernel_family}")

    return {
        "queue": queue,
        "base_knl": base_knl,
        "kernel_kwargs": kernel_kwargs,
        "kernel_type": kernel_type,
        "table_dtype": table_dtype,
        "integral": integral,
        "reference_derivative": reference_derivative,
        "q_order": q_order,
        "regular_quad_order": regular_quad_order,
        "radial_quad_order": radial_quad_order,
        "reference_quad_order": reference_quad_order,
        "fd_h": fd_h,
        "rel_tol": rel_tol,
    }


def _build_duffy_table(
    queue,
    *,
    dim,
    q_order,
    output_kernel,
    kernel_func,
    kernel_type,
    table_dtype,
    source_box_extent,
    source_box_level,
    regular_quad_order,
    radial_quad_order,
    kernel_kwargs,
):
    table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=dim,
        build_method="DuffyRadial",
        kernel_func=kernel_func,
        kernel_type=kernel_type,
        sumpy_kernel=output_kernel,
        source_box_extent=source_box_extent,
        dtype=table_dtype,
        progress_bar=False,
    )
    table.source_box_level = int(source_box_level)
    table.table_root_extent = 1.0
    table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        **kernel_kwargs,
    )
    return table


_FULL_ACCURACY_SCALAR_MATRIX = [
    pytest.param(
        kernel_family,
        dim,
        source_box_level,
        id=f"{kernel_family}-{dim}d-L{source_box_level}-scalar",
    )
    for kernel_family in ("laplace", "yukawa", "helmholtz")
    for dim in (2, 3)
    for source_box_level in (0, 2)
]


_FULL_ACCURACY_DERIVATIVE_MATRIX = [
    pytest.param(
        kernel_family,
        dim,
        source_box_level,
        id=f"{kernel_family}-{dim}d-L{source_box_level}-axis-derivatives",
    )
    for kernel_family in ("laplace", "yukawa", "helmholtz")
    for dim in (2, 3)
    for source_box_level in (0, 2)
]


@pytest.mark.full_accuracy
@pytest.mark.parametrize(
    ("kernel_family", "dim", "source_box_level"),
    _FULL_ACCURACY_SCALAR_MATRIX,
)
def test_duffy_batched_scalar_full_accuracy_matrix(
    kernel_family,
    dim,
    source_box_level,
):
    cfg = _full_accuracy_case_setup(kernel_family, dim)
    queue = cfg["queue"]
    base_knl = cfg["base_knl"]
    kernel_kwargs = cfg["kernel_kwargs"]
    source_box_extent = _source_box_extent_from_level(source_box_level)

    base_func = npt.sumpy_kernel_to_lambda(base_knl, parameter_values=kernel_kwargs)
    scalar_table = _build_duffy_table(
        queue,
        dim=dim,
        q_order=cfg["q_order"],
        output_kernel=base_knl,
        kernel_func=base_func,
        kernel_type=cfg["kernel_type"],
        table_dtype=cfg["table_dtype"],
        source_box_extent=source_box_extent,
        source_box_level=source_box_level,
        regular_quad_order=cfg["regular_quad_order"],
        radial_quad_order=cfg["radial_quad_order"],
        kernel_kwargs=kernel_kwargs,
    )

    case_id = _pick_far_positive_case_id(scalar_table)
    target_point_index = scalar_table.n_q_points // 2
    target = np.asarray(
        scalar_table.find_target_point(target_point_index, case_id),
        dtype=cfg["table_dtype"],
    )

    source_values = np.asarray(
        [
            _exp_bump_profile_on_box(
                source_box_extent,
                *[float(np.real(coord)) for coord in point],
            )
            for point in scalar_table.q_points
        ],
        dtype=np.float64,
    )

    scalar_table_value = _contract_table_for_source_values(
        scalar_table,
        source_values,
        target_point_index,
        case_id,
    )

    source_interp = _make_interpolated_source_callable(scalar_table, source_values)

    def potential_at(target_point):
        def integrand(*coords):
            displacement = [target_point[i] - coords[i] for i in range(dim)]
            return source_interp(*coords) * base_func(*displacement)

        return cfg["integral"](
            dim,
            cfg["reference_quad_order"],
            integrand,
            box_extent=source_box_extent,
        )

    reference_value = potential_at(target)
    rel_error = abs(scalar_table_value - reference_value) / max(
        1.0, abs(reference_value)
    )

    assert rel_error < cfg["rel_tol"]


@pytest.mark.full_accuracy
@pytest.mark.parametrize(
    ("kernel_family", "dim", "source_box_level"),
    _FULL_ACCURACY_DERIVATIVE_MATRIX,
)
def test_duffy_batched_derivative_full_accuracy_matrix(
    kernel_family,
    dim,
    source_box_level,
):
    cfg = _full_accuracy_case_setup(kernel_family, dim)
    queue = cfg["queue"]
    base_knl = cfg["base_knl"]
    kernel_kwargs = cfg["kernel_kwargs"]
    source_box_extent = _source_box_extent_from_level(source_box_level)

    target_knl = AxisTargetDerivative(0, base_knl)
    source_knl = AxisSourceDerivative(0, base_knl)

    base_func = npt.sumpy_kernel_to_lambda(base_knl, parameter_values=kernel_kwargs)
    target_func = npt.sumpy_kernel_to_lambda(target_knl, parameter_values=kernel_kwargs)
    source_func = npt.sumpy_kernel_to_lambda(source_knl, parameter_values=kernel_kwargs)

    target_table = _build_duffy_table(
        queue,
        dim=dim,
        q_order=cfg["q_order"],
        output_kernel=target_knl,
        kernel_func=target_func,
        kernel_type=cfg["kernel_type"],
        table_dtype=cfg["table_dtype"],
        source_box_extent=source_box_extent,
        source_box_level=source_box_level,
        regular_quad_order=cfg["regular_quad_order"],
        radial_quad_order=cfg["radial_quad_order"],
        kernel_kwargs=kernel_kwargs,
    )
    source_table = _build_duffy_table(
        queue,
        dim=dim,
        q_order=cfg["q_order"],
        output_kernel=source_knl,
        kernel_func=source_func,
        kernel_type=cfg["kernel_type"],
        table_dtype=cfg["table_dtype"],
        source_box_extent=source_box_extent,
        source_box_level=source_box_level,
        regular_quad_order=cfg["regular_quad_order"],
        radial_quad_order=cfg["radial_quad_order"],
        kernel_kwargs=kernel_kwargs,
    )

    case_id = _pick_far_positive_case_id(target_table)
    target_point_index = target_table.n_q_points // 2
    target = np.asarray(
        target_table.find_target_point(target_point_index, case_id),
        dtype=cfg["table_dtype"],
    )

    source_values = np.asarray(
        [
            _exp_bump_profile_on_box(
                source_box_extent,
                *[float(np.real(coord)) for coord in point],
            )
            for point in target_table.q_points
        ],
        dtype=np.float64,
    )

    target_table_value = _contract_table_for_source_values(
        target_table,
        source_values,
        target_point_index,
        case_id,
    )
    source_table_value = _contract_table_for_source_values(
        source_table,
        source_values,
        target_point_index,
        case_id,
    )

    source_interp = _make_interpolated_source_callable(target_table, source_values)

    def potential_at(target_point):
        def integrand(*coords):
            displacement = [target_point[i] - coords[i] for i in range(dim)]
            return source_interp(*coords) * base_func(*displacement)

        return cfg["integral"](
            dim,
            cfg["reference_quad_order"],
            integrand,
            box_extent=source_box_extent,
        )

    fd_target_derivative = cfg["reference_derivative"](
        potential_at,
        target,
        h=cfg["fd_h"] * source_box_extent,
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

    assert rel_target_error < cfg["rel_tol"]
    assert rel_source_error < cfg["rel_tol"]
    assert antisymmetry_error < 1.0e-10
