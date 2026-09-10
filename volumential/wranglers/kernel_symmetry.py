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

__doc__ = """Introspection of :mod:`sumpy` kernels for near-field table symmetry.

Figures out how target kernels wrap their base kernel, which source kernels
they imply, and what constant source direction (if any) a directional source
derivative carries -- the near-field tables key their symmetry reduction on
that direction.
"""

import numpy as np

import pyopencl as cl
import pyopencl.array
from sumpy.kernel import AxisSourceDerivative, DirectionalSourceDerivative


def _find_directional_source_derivative_kernel(kernel):
    if isinstance(kernel, DirectionalSourceDerivative):
        return kernel
    inner = getattr(kernel, "inner_kernel", None)
    if inner is None:
        return None
    return _find_directional_source_derivative_kernel(inner)


def _extract_symmetry_source_direction(out_kernel, source_extra_kwargs, queue):
    dknl = _find_directional_source_derivative_kernel(out_kernel)
    if dknl is None:
        return None

    dir_vec_name = getattr(dknl, "dir_vec_name", None)
    if not dir_vec_name or dir_vec_name not in source_extra_kwargs:
        return None

    dim = int(dknl.dim)
    dir_vec = source_extra_kwargs[dir_vec_name]

    def _constant_component_value(comp_data):
        arr = np.asarray(comp_data).ravel()
        if arr.size == 0:
            return None

        if np.iscomplexobj(arr):
            arr_c = np.asarray(arr, dtype=np.complex128)
            if not np.all(np.isclose(np.imag(arr_c), 0.0)):
                raise ValueError(
                    "symmetry_source_direction must be real-valued for all sources"
                )
            arr = np.real(arr_c)

        arr = np.asarray(arr, dtype=np.float64)
        first = float(arr.ravel()[0])
        if arr.size > 1 and not np.allclose(arr, first):
            raise ValueError(
                "symmetry_source_direction must be constant across sources"
            )
        return first

    if isinstance(dir_vec, cl.array.Array):
        dir_vec_h = np.asarray(dir_vec.get(queue=queue))
    else:
        dir_vec_h = np.asarray(dir_vec)

    if dir_vec_h.size == 0:
        return None

    if dir_vec_h.dtype != object:
        if dir_vec_h.ndim == 1:
            if dir_vec_h.size != dim:
                raise ValueError(
                    "symmetry_source_direction must have one component per axis "
                    f"(expected length {dim}, got {dir_vec_h.size})"
                )

            if np.iscomplexobj(dir_vec_h):
                imag = np.imag(np.asarray(dir_vec_h, dtype=np.complex128)).ravel()
                if imag.size and not np.all(np.isclose(imag, 0.0)):
                    raise ValueError(
                        "symmetry_source_direction must be real-valued with one "
                        f"component per axis (expected length {dim})"
                    )
                dir_vec_h = np.real(dir_vec_h)

            return np.asarray(dir_vec_h, dtype=np.float64)

        if dir_vec_h.ndim == 2:
            if dir_vec_h.shape[0] == dim:
                comp_rows = dir_vec_h
            elif dir_vec_h.shape[1] == dim:
                comp_rows = dir_vec_h.T
            else:
                raise ValueError(
                    "symmetry_source_direction has incompatible shape "
                    f"{dir_vec_h.shape}; expected ({dim}, nsources) or (nsources, {dim})"
                )

            comps = []
            for comp_row in comp_rows:
                comp_val = _constant_component_value(comp_row)
                if comp_val is None:
                    return None
                comps.append(comp_val)

            return np.asarray(comps, dtype=np.float64)

        raise ValueError(
            "symmetry_source_direction must be a vector or matrix, got "
            f"{dir_vec_h.ndim}D array"
        )

    try:
        components = list(dir_vec)
    except TypeError as exc:
        raise ValueError(
            "symmetry_source_direction must be vector-like with one component per axis"
        ) from exc

    if len(components) != dim:
        raise ValueError(
            "symmetry_source_direction must have one component per axis "
            f"(expected {dim}, got {len(components)})"
        )

    comps = []
    for comp in components:
        if isinstance(comp, cl.array.Array):
            comp_h = np.asarray(comp.get(queue=queue))
        else:
            comp_h = np.asarray(comp)
        comp_val = _constant_component_value(comp_h)
        if comp_val is None:
            return None
        comps.append(comp_val)

    return np.asarray(comps, dtype=np.float64)


def _derive_source_kernels_from_target_kernels(target_kernels, *, require_single=True):
    from sumpy.kernel import TargetTransformationRemover

    txr = TargetTransformationRemover()

    if not target_kernels:
        return None

    source_kernels = tuple(txr(knl) for knl in target_kernels)
    if not require_single:
        return source_kernels

    from pytools import single_valued

    try:
        source_knl = single_valued(source_kernels)
    except (AssertionError, ValueError) as exc:
        raise ValueError(
            "target kernels must share a single source-side kernel "
            "after removing target derivatives"
        ) from exc

    return (source_knl,)


def _target_kernels_include_source_derivatives(target_kernels):
    if target_kernels is None:
        return False

    for knl in target_kernels:
        cur_knl = knl
        while cur_knl is not None:
            if isinstance(cur_knl, (AxisSourceDerivative, DirectionalSourceDerivative)):
                return True
            cur_knl = getattr(cur_knl, "inner_kernel", None)

    return False

# vim: filetype=pyopencl:foldmethod=marker
