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

import json
import logging
import hashlib
import math
from collections import OrderedDict

import numpy as np

import pyopencl as cl
import pyopencl.array
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
from pytools.obj_array import new_1d as obj_array_1d
from sumpy.array_context import PyOpenCLArrayContext
from sumpy.fmm import (
    SumpyExpansionWrangler,
    SumpyTreeIndependentDataForWrangler,
)
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    DirectionalSourceDerivative,
    ExpressionKernel,
    HelmholtzKernel,
    LaplaceKernel,
    YukawaKernel,
)

from volumential.expansion_wrangler_interface import (
    ExpansionWranglerInterface,
    TreeIndependentDataForWranglerInterface,
)
from volumential.nearfield_potential_table import NearFieldInteractionTable


logger = logging.getLogger(__name__)


def _table_data_fingerprint(arr, sample_bytes=4096):
    narr = np.asarray(arr)
    nbytes = int(narr.nbytes)
    if nbytes == 0:
        return (str(narr.dtype), tuple(int(s) for s in narr.shape), 0, "")

    raw = narr.view(np.uint8).reshape(-1)
    if nbytes <= 2 * sample_bytes:
        sample = bytes(raw)
    else:
        sample = bytes(raw[:sample_bytes]) + bytes(raw[-sample_bytes:])

    digest = hashlib.sha256(sample).hexdigest()
    return (str(narr.dtype), tuple(int(s) for s in narr.shape), nbytes, digest)


def _gauss_legendre_nodes_and_weights(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return (0.5 * (nodes + 1.0), 0.5 * weights)


def _barycentric_interp_matrix(source_nodes, target_nodes):
    from scipy.interpolate import BarycentricInterpolator as Interpolator

    source_nodes = np.asarray(source_nodes, dtype=np.float64)
    target_nodes = np.asarray(target_nodes, dtype=np.float64)

    if source_nodes.size == 1:
        return np.ones((target_nodes.size, 1), dtype=np.float64)

    weights = np.asarray(Interpolator(xi=source_nodes, yi=None).wi, dtype=np.float64)
    interp_mat = np.empty((target_nodes.size, source_nodes.size), dtype=np.float64)

    for i, x_tgt in enumerate(target_nodes):
        diff = x_tgt - source_nodes
        hit = np.where(np.abs(diff) < 1.0e-15)[0]
        if hit.size:
            interp_mat[i, :] = 0.0
            interp_mat[i, int(hit[0])] = 1.0
            continue

        terms = weights / diff
        interp_mat[i, :] = terms / np.sum(terms)

    return interp_mat


class _RadialPowerKernel(ExpressionKernel):
    init_arg_names = ("dim", "power")
    mapper_method = "map_expression_kernel"

    def __init__(self, dim, power):
        from pymbolic.primitives import make_sym_vector
        from sumpy.symbolic import pymbolic_real_norm_2

        if power < 0:
            raise ValueError("power must be non-negative")

        self.power = int(power)
        dim = int(dim)

        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = 1 if self.power == 0 else r**self.power

        super().__init__(
            dim,
            expression=expr,
            global_scaling_const=1,
        )

    @property
    def is_complex_valued(self):
        return False

    def __getinitargs__(self):
        return (self.dim, self.power)

    def __repr__(self):
        return f"RadialPowerKernel(dim={self.dim}, power={self.power})"


class _RadialPowerLogKernel(ExpressionKernel):
    init_arg_names = ("dim", "power")
    mapper_method = "map_expression_kernel"

    def __init__(self, dim, power):
        from pymbolic import var
        from pymbolic.primitives import make_sym_vector
        from sumpy.symbolic import pymbolic_real_norm_2

        self.power = int(power)
        if self.power <= 0:
            raise ValueError("power must be positive for r**power * log(r)")

        dim = int(dim)
        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = (r**self.power) * var("log")(r)

        super().__init__(
            dim,
            expression=expr,
            global_scaling_const=1,
        )

    @property
    def is_complex_valued(self):
        return False

    def __getinitargs__(self):
        return (self.dim, self.power)

    def __repr__(self):
        return f"RadialPowerLogKernel(dim={self.dim}, power={self.power})"


class _HelmholtzSplitSeriesRemainderKernel(ExpressionKernel):
    """Analytic near-field split remainder for ``G_k - G_0``.

    ``split_order`` controls which non-smooth terms are removed from this
    remainder and handled by prebuilt near-field tables:

    - 2D: remove :math:`r^{2n}\\log r` for :math:`n=1,\\dots,p-1`.
    - 3D: remove odd powers :math:`r^{2j-1}` for :math:`j=1,\\dots,p-1`.

    The remaining smooth polynomial terms stay in this kernel.
    """

    init_arg_names = (
        "dim",
        "wave_number_real",
        "wave_number_imag",
        "split_order",
        "series_nmax",
    )
    mapper_method = "map_expression_kernel"

    def __init__(
        self,
        dim,
        wave_number_real,
        wave_number_imag,
        split_order,
        series_nmax,
    ):
        from math import factorial

        from pymbolic import var
        from pymbolic.primitives import Comparison, If
        from pymbolic.primitives import make_sym_vector
        from sumpy.symbolic import pymbolic_real_norm_2

        dim = int(dim)
        self.wave_number_real = float(wave_number_real)
        self.wave_number_imag = float(wave_number_imag)
        self.split_order = int(split_order)
        self.series_nmax = int(series_nmax)

        if dim not in (2, 3):
            raise NotImplementedError("split remainder kernel supports only 2D/3D")
        if self.split_order < 1:
            raise ValueError("split_order must be >= 1")

        k = np.complex128(self.wave_number_real + 1j * self.wave_number_imag)
        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = 0

        if dim == 2:
            if np.abs(k) == 0.0:
                expr = np.complex128(0.0)
            else:
                expr = expr + np.complex128(
                    0.25j
                    - (1.0 / (2.0 * np.pi))
                    * (np.log(0.5 * k) + np.complex128(np.euler_gamma))
                )

                log_k_half = np.log(0.5 * k)
                euler_gamma = np.complex128(np.euler_gamma)
                for n in range(1, self.series_nmax + 1):
                    series_scale = (
                        ((-1) ** n) * (k * k / 4.0) ** n / (factorial(n) * factorial(n))
                    )
                    harmonic_n = np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64))
                    common = np.complex128(series_scale)

                    coeff_log = np.complex128(-common / (2.0 * np.pi))
                    coeff_power = np.complex128(
                        common
                        * (
                            (harmonic_n - (log_k_half + euler_gamma)) / (2.0 * np.pi)
                            + 0.25j
                        )
                    )

                    power = 2 * n
                    if n >= self.split_order:
                        log_term = If(
                            Comparison(r, "<=", np.float64(1.0e-300)),
                            np.float64(0.0),
                            (r**power) * var("log")(r),
                        )
                        expr = expr + coeff_log * log_term
                    expr = expr + coeff_power * (r**power)
        else:
            max_extracted_n = 2 * max(0, self.split_order - 1)
            for n in range(1, self.series_nmax + 1):
                if n % 2 == 0 and n <= max_extracted_n:
                    continue
                coeff = (1j * k) ** n / (4.0 * np.pi * factorial(n))
                expr = expr + np.complex128(coeff) * (r ** (n - 1))

        super().__init__(
            dim,
            expression=expr,
            global_scaling_const=1,
        )

    @property
    def is_complex_valued(self):
        return True

    def __getinitargs__(self):
        return (
            self.dim,
            self.wave_number_real,
            self.wave_number_imag,
            self.split_order,
            self.series_nmax,
        )

    def __repr__(self):
        return (
            "HelmholtzSplitSeriesRemainderKernel("
            f"dim={self.dim}, k=({self.wave_number_real}+{self.wave_number_imag}j), "
            f"split_order={self.split_order}, series_nmax={self.series_nmax})"
        )


def _normalize_helmholtz_split_term_key(term_key):
    if isinstance(term_key, tuple):
        if len(term_key) != 2:
            raise ValueError(
                "helmholtz split term key tuples must have length 2: (kind, power)"
            )
        kind, power = term_key
    elif isinstance(term_key, int):
        kind = "power"
        power = term_key
    elif isinstance(term_key, str):
        if term_key == "constant":
            kind = "power"
            power = 0
        elif term_key.startswith("power:"):
            kind = "power"
            power = term_key.split(":", 1)[1]
        elif term_key.startswith("power_log:"):
            kind = "power_log"
            power = term_key.split(":", 1)[1]
        else:
            raise ValueError(
                "invalid helmholtz split term key string; expected one of: "
                "constant, power:<n>, power_log:<n>"
            )
    else:
        raise TypeError(
            "helmholtz split term key must be int, str, or (kind, power) tuple"
        )

    kind = str(kind)
    if kind not in ("power", "power_log"):
        raise ValueError("helmholtz split term key kind must be 'power' or 'power_log'")

    ipower = int(power)
    if ipower < 0:
        raise ValueError("helmholtz split term key power must be >= 0")
    if kind == "power_log" and ipower == 0:
        raise ValueError("helmholtz split power_log term requires power >= 1")

    return (kind, ipower)


def _format_helmholtz_split_term_key(term_key):
    kind, power = _normalize_helmholtz_split_term_key(term_key)
    if kind == "power":
        return f"power:{power}"
    return f"power_log:{power}"


def _select_split_order_from_rho(rho_max, thresholds, orders):
    if len(orders) != len(thresholds) + 1:
        raise ValueError("orders must have one more element than thresholds")

    rho = float(rho_max)
    for idx, threshold in enumerate(thresholds):
        if rho <= float(threshold):
            return int(orders[idx])
    return int(orders[-1])


def _select_split_order_from_rho_components(
    rho_real,
    rho_imag,
    thresholds_real,
    thresholds_imag,
    orders,
):
    order_real = _select_split_order_from_rho(rho_real, thresholds_real, orders)
    order_imag = _select_split_order_from_rho(rho_imag, thresholds_imag, orders)
    return max(order_real, order_imag)


def _compute_box_local_ids(queue, tree, n_q_points):
    if not getattr(tree, "sources_are_targets", False):
        raise ValueError(
            "table-based near-field evaluation requires sources and targets "
            "to coincide (tree.sources_are_targets=True)"
        )

    n_particles = tree.ntargets
    if n_particles % n_q_points != 0:
        raise ValueError("particle count is not divisible by box-local quadrature size")

    user_local_ids = np.tile(
        np.arange(n_q_points, dtype=np.int32),
        n_particles // n_q_points,
    )
    sorted_target_ids = tree.sorted_target_ids.get(queue)
    return cl.array.to_device(queue, user_local_ids[sorted_target_ids])


def _validate_table_box_particle_layout(
    queue,
    tree,
    target_boxes,
    source_boxes,
    n_q_points,
):
    box_counts = tree.box_target_counts_nonchild.get(queue)

    if hasattr(target_boxes, "get"):
        target_box_ids = target_boxes.get(queue)
    else:
        target_box_ids = np.asarray(target_boxes)

    if hasattr(source_boxes, "get"):
        source_box_ids = source_boxes.get(queue)
    else:
        source_box_ids = np.asarray(source_boxes)

    target_counts = box_counts[target_box_ids]
    source_counts = box_counts[source_box_ids]

    target_ok = np.all(target_counts == n_q_points)
    source_ok = np.all(source_counts == n_q_points)
    if target_ok and source_ok:
        return

    bad_target = np.unique(target_counts[target_counts != n_q_points])
    bad_source = np.unique(source_counts[source_counts != n_q_points])

    raise ValueError(
        "table-based near-field evaluation requires exactly "
        f"{n_q_points} quadrature points per active source/target box; "
        f"found target counts {bad_target.tolist()} and source counts "
        f"{bad_source.tolist()}. Build the particle tree from the mesh box-tree "
        "(build_particle_tree_from_box_tree) to preserve per-cell quadrature layout."
    )


def _array_layout_cache_token(ary):
    if isinstance(ary, cl.array.Array):
        base_data = getattr(ary, "base_data", None)
        int_ptr = getattr(base_data, "int_ptr", None)
        if int_ptr is not None:
            return (
                "cl",
                int(int_ptr),
                int(getattr(ary, "offset", 0)),
                int(ary.size),
                ary.dtype.str,
            )
    return ("py", id(ary))


def _validate_table_box_particle_layout_cached(
    queue,
    tree,
    target_boxes,
    source_boxes,
    n_q_points,
    validation_cache,
):
    if validation_cache is None:
        _validate_table_box_particle_layout(
            queue,
            tree,
            target_boxes,
            source_boxes,
            n_q_points,
        )
        return

    cache_key = (
        _array_layout_cache_token(target_boxes),
        _array_layout_cache_token(source_boxes),
        int(n_q_points),
    )
    if cache_key in validation_cache:
        return

    _validate_table_box_particle_layout(
        queue,
        tree,
        target_boxes,
        source_boxes,
        n_q_points,
    )
    validation_cache.add(cache_key)


class SumpyTimingFuture:
    def __init__(self, queue, events):
        self.queue = queue
        self.events = [evt for evt in events if evt is not None]

    def __call__(self):
        for evt in self.events:
            evt.wait()
        return 0.0


def level_to_rscale(tree, level):
    return tree.root_extent * (2**-level)


def inverse_id_map(queue, mapped_ids):
    """Given a index mapping as its mapped ids, compute its inverse,
    and return the inverse by the inversely-mapped ids.
    """
    cl_array = False
    if isinstance(mapped_ids, cl.array.Array):
        cl_array = True
        mapped_ids = mapped_ids.get(queue)

    inv_ids = np.zeros_like(mapped_ids)
    inv_ids[mapped_ids] = np.arange(len(mapped_ids))

    if cl_array:
        inv_ids = cl.array.to_device(queue, inv_ids)

    return inv_ids


def _queue_from_array_like(ary):
    if isinstance(ary, cl.array.Array):
        return ary.queue

    if isinstance(ary, np.ndarray) and ary.dtype == object:
        for entry in ary.flat:
            if isinstance(entry, cl.array.Array):
                return entry.queue

    return None


def _resolve_queue(queue, traversal, tree_indep):
    if queue is not None:
        return queue

    setup_actx = getattr(tree_indep, "_setup_actx", None)
    actx_queue = getattr(setup_actx, "queue", None)
    if actx_queue is not None:
        return actx_queue

    tree = getattr(traversal, "tree", None)
    if tree is not None:
        for ary_name in ("box_centers", "box_levels", "targets", "sources"):
            ary = getattr(tree, ary_name, None)
            ary_queue = _queue_from_array_like(ary)
            if ary_queue is not None:
                return ary_queue

    raise TypeError(
        "queue is required when it cannot be inferred from tree_indep/traversal"
    )


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


def _derive_source_kernels_from_target_kernels(target_kernels):
    from pytools import single_valued
    from sumpy.kernel import TargetTransformationRemover

    txr = TargetTransformationRemover()

    if not target_kernels:
        return None

    try:
        source_knl = single_valued(txr(knl) for knl in target_kernels)
    except ValueError as exc:
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


def _prepare_table_data_and_entry_map(table_levels):
    if not table_levels:
        raise ValueError("table_levels cannot be empty")

    table0 = table_levels[0]
    n_full_entries = len(table0.data)

    reduced_flags = [
        bool(getattr(table, "table_data_is_symmetry_reduced", False))
        for table in table_levels
    ]
    if any(flag != reduced_flags[0] for flag in reduced_flags[1:]):
        raise RuntimeError("mixed full/reduced near-field table storage across levels")

    if reduced_flags[0]:
        table_entry_scales = np.ones(n_full_entries, dtype=table0.data.dtype)
        orbit_info = None
        if hasattr(table0, "_get_orbit_canonical_info"):
            orbit_info = table0._get_orbit_canonical_info()

        if orbit_info is not None and "canonical_entry_ids" in orbit_info:
            canonical_entry_ids = np.asarray(
                orbit_info["canonical_entry_ids"], dtype=np.int64
            )
            kept_entry_ids = np.asarray(orbit_info["entry_ids"], dtype=np.int64)
            if len(kept_entry_ids) == 0:
                raise RuntimeError("near-field table contains no canonical entries")

            finite_mask = np.isfinite(table0.data)
            if not np.all(finite_mask[kept_entry_ids]):
                raise RuntimeError(
                    "near-field table is missing finite values for canonical entries"
                )

            for table in table_levels[1:]:
                level_finite_mask = np.isfinite(table.data)
                if not np.all(level_finite_mask[kept_entry_ids]):
                    raise RuntimeError(
                        "near-field levels disagree on canonical entry availability"
                    )

            compact_ids = np.full(n_full_entries, -1, dtype=np.int32)
            compact_ids[kept_entry_ids] = np.arange(len(kept_entry_ids), dtype=np.int32)
            table_entry_ids = compact_ids[canonical_entry_ids]
            if "canonical_scales" in orbit_info:
                table_entry_scales = np.asarray(
                    orbit_info["canonical_scales"], dtype=table0.data.dtype
                )
        else:
            finite_mask = np.isfinite(table0.data)
            for table in table_levels[1:]:
                level_finite_mask = np.isfinite(table.data)
                if not np.array_equal(level_finite_mask, finite_mask):
                    raise RuntimeError(
                        "near-field levels disagree on symmetry-reduced entry ids"
                    )

            kept_entry_ids = np.flatnonzero(finite_mask).astype(np.int64)
            if len(kept_entry_ids) == 0:
                raise RuntimeError("near-field table contains no finite entries")

            table_entry_ids = np.full(n_full_entries, -1, dtype=np.int32)
            table_entry_ids[kept_entry_ids] = np.arange(
                len(kept_entry_ids), dtype=np.int32
            )
    else:
        for table in table_levels:
            if not np.all(np.isfinite(table.data)):
                raise RuntimeError("full near-field table contains non-finite entries")
        kept_entry_ids = np.arange(n_full_entries, dtype=np.int64)
        table_entry_ids = np.full(n_full_entries, -1, dtype=np.int32)
        table_entry_ids[kept_entry_ids] = np.arange(len(kept_entry_ids), dtype=np.int32)
        table_entry_scales = np.ones(n_full_entries, dtype=table0.data.dtype)

    table_data_combined = np.zeros(
        (len(table_levels), len(kept_entry_ids)),
        dtype=table0.data.dtype,
    )
    mode_nmlz_combined = np.zeros(
        (len(table_levels), len(table0.mode_normalizers)),
        dtype=table0.mode_normalizers.dtype,
    )
    exterior_mode_nmlz_combined = np.zeros(
        (len(table_levels), len(table0.kernel_exterior_normalizers)),
        dtype=table0.kernel_exterior_normalizers.dtype,
    )

    for lev, table in enumerate(table_levels):
        table_data_combined[lev, :] = table.data[kept_entry_ids]
        mode_nmlz_combined[lev, :] = table.mode_normalizers
        exterior_mode_nmlz_combined[lev, :] = table.kernel_exterior_normalizers

    return (
        table_data_combined,
        mode_nmlz_combined,
        exterior_mode_nmlz_combined,
        table_entry_ids,
        table_entry_scales,
    )


# {{{ sumpy backend


class FPNDSumpyTreeIndependentDataForWrangler(
    TreeIndependentDataForWranglerInterface, SumpyTreeIndependentDataForWrangler
):
    """Objects of this type serve as a place to keep the code needed
    for ExpansionWrangler if it is using sumpy to perform multipole
    expansion and manipulations.

    Since ``SumpyExpansionWrangler`` necessarily must have a
    ``pyopencl.CommandQueue``, but this queue is allowed to be
    more ephemeral than the code, the code's lifetime
    is decoupled by storing it in this object.
    """

    def __init__(
        self,
        cl_context,
        multipole_expansion_factory,
        local_expansion_factory,
        target_kernels,
        exclude_self=True,
        use_rscale=None,
        strength_usage=None,
        source_kernels=None,
    ):
        if source_kernels is None and not _target_kernels_include_source_derivatives(
            target_kernels
        ):
            source_kernels = _derive_source_kernels_from_target_kernels(target_kernels)

        queue = cl.CommandQueue(cl_context)
        actx = PyOpenCLArrayContext(queue)
        super_kwargs = dict(
            target_kernels=target_kernels,
            exclude_self=exclude_self,
            strength_usage=strength_usage,
            source_kernels=source_kernels,
        )
        if use_rscale is not None:
            super_kwargs["use_rscale"] = use_rscale

        super().__init__(
            actx,
            multipole_expansion_factory,
            local_expansion_factory,
            **super_kwargs,
        )

    def _for_queue(self, queue):
        setup_queue = getattr(self._setup_actx, "queue", None)
        if queue is None or setup_queue is queue:
            return self

        ctor_kwargs = dict(
            exclude_self=self.exclude_self,
            strength_usage=self.strength_usage,
            source_kernels=self.source_kernels,
        )
        use_rscale = getattr(self, "use_rscale", None)
        if use_rscale is not None:
            ctor_kwargs["use_rscale"] = use_rscale

        tree_indep = type(self)(
            queue.context,
            self.multipole_expansion_factory,
            self.local_expansion_factory,
            self.target_kernels,
            **ctor_kwargs,
        )
        tree_indep._setup_actx = PyOpenCLArrayContext(queue)
        return tree_indep

    def get_wrangler(
        self,
        queue,
        traversal,
        dtype,
        fmm_level_to_order,
        source_extra_kwargs=None,
        kernel_extra_kwargs=None,
        self_extra_kwargs=None,
        *args,
        **kwargs,
    ):
        tree_indep = self._for_queue(queue)

        return FPNDSumpyExpansionWrangler(
            tree_indep=tree_indep,
            queue=queue,
            traversal=traversal,
            *args,
            dtype=dtype,
            fmm_level_to_order=fmm_level_to_order,
            source_extra_kwargs=source_extra_kwargs,
            kernel_extra_kwargs=kernel_extra_kwargs,
            self_extra_kwargs=self_extra_kwargs,
            **kwargs,
        )

    def opencl_fft_app(self, shape, dtype, inverse):
        from sumpy.tools import get_opencl_fft_app

        return get_opencl_fft_app(self._setup_actx, shape, dtype, inverse=inverse)


class FPNDSumpyExpansionWrangler(ExpansionWranglerInterface, SumpyExpansionWrangler):
    """This expansion wrangler uses "fpnd" strategy. That is, Far field is
    computed via Particle approximation and Near field is computed Directly.
    The FMM is performed using sumpy backend.

    For Helmholtz split mode, neighbor-list (list1) interactions are evaluated
    with the decomposition

    .. math::

        G_k(r) = G_0(r) + S_p(r) + R_p(r),

    where ``G_0`` is tabulated by Laplace near-field tables,
    ``S_p`` is the sum of pretabulated non-smooth split terms,
    and ``R_p`` is an analytic remainder evaluated online by P2P.

    Split terms kept in ``S_p`` are:

    - 2D: :math:`r^{2n}\\log r`, :math:`n=1,\\dots,p-1`.
    - 3D: :math:`r^{2j-1}`, :math:`j=1,\\dots,p-1`.

    Smooth polynomial terms (2D: constant and :math:`r^{2n}`;
    3D: even powers including constant) remain in ``R_p``.

    .. attribute:: source_extra_kwargs

        Keyword arguments to be passed to interactions that involve
        the source field.

    .. attribute:: kernel_extra_kwargs

        Keyword arguments to be passed to interactions that involve
        expansions, but not the source field.

    .. attribute:: self_extra_kwargs

        Keyword arguments to be passed for handling
        self interactions (singular integrals)
    """

    # {{{ constructor

    def __init__(
        self,
        tree_indep,
        queue,
        traversal,
        dtype,
        fmm_level_to_order,
        near_field_table,
        quad_order,
        potential_kind=1,
        source_extra_kwargs=None,
        kernel_extra_kwargs=None,
        self_extra_kwargs=None,
        list1_extra_kwargs=None,
        helmholtz_split=None,
        helmholtz_split_order=1,
        helmholtz_split_smooth_quad_order=None,
        helmholtz_split_auto_config=None,
        helmholtz_split_term_tables=None,
        helmholtz_split_order1_legacy_subtraction=False,
        translation_classes_data=None,
        preprocessed_mpole_dtype=None,
    ):
        """
        near_field_table can either one of three things:
            1. a single table, when len(target_kernels) = 1 (single level)
            2. a list of tables, when len(target_kernels) = 1 (multiple levels)
            3. otherwise, a dictionary from kernel.__repr__() to a list of its tables

        If ``helmholtz_split`` is true, near-field table lookups are interpreted as
        Laplace contributions and list1 receives an additional online
        Helmholtz-minus-Laplace correction from neighbor-only direct evaluation.
        This mode supports Helmholtz/Yukawa output kernels in 2D or 3D,
        including axis target/source and directional source derivative wrappers.
        For multiple output kernels, split correction currently runs with
        ``helmholtz_split_order=1``.

        If ``helmholtz_split`` is ``None`` (default), split mode is enabled
        automatically for supported Helmholtz/Yukawa target kernels and
        disabled otherwise.

        ``helmholtz_split_order`` controls how many *non-smooth* correction
        terms are handled by prebuilt near-field split tables. Values mean:
        ``helmholtz_split_order=1`` keeps only the Laplace singular part in
        tables and leaves all Helmholtz-minus-Laplace smooth correction online.
        For 3D, higher orders add odd powers :math:`r, r^3, ...`.
        For 2D, higher orders add :math:`r^{2m}\\log r` terms.
        Smooth polynomial terms (:math:`1, r^2, r^4, ...`) stay in the online
        smooth correction.
        For 2D ``power_log`` terms, single-table scaling is supported with a
        level-dependent :math:`\\log(\\alpha_\\ell)` correction folded into online
        source strengths. The correction backend is controlled by
        ``helmholtz_split_auto_config["power_log_single_table_beta_mode"]``:
        ``"p2p"`` (default) or ``"table"``.

        ``helmholtz_split_smooth_quad_order`` controls the online tensor-product
        quadrature order used for the smooth split correction integral. For
        ``helmholtz_split_order > 1``, values ``>= q`` are supported; ``m=q`` uses
        an analytic split-remainder kernel on the base quadrature.

        ``helmholtz_split_term_tables`` optionally provides precomputed near-field
        tables for additional split terms. Keys may be integer powers (legacy),
        ``("power", p)`` for :math:`r^p`, and ``("power_log", p)`` for
        :math:`r^p\\log r`.

        ``helmholtz_split_order1_legacy_subtraction`` keeps the historical
        split-order-1 correction path that evaluates Helmholtz and Laplace
        kernels separately and subtracts them. By default this is ``False`` and
        split-order-1 uses the analytic series-remainder kernel path.
        """

        queue = _resolve_queue(queue, traversal, tree_indep)
        if hasattr(tree_indep, "_for_queue"):
            tree_indep = tree_indep._for_queue(queue)

        if source_extra_kwargs is None:
            source_extra_kwargs = {}

        if kernel_extra_kwargs is None:
            kernel_extra_kwargs = {}

        if self_extra_kwargs is None:
            self_extra_kwargs = {}

        super().__init__(
            tree_indep,
            traversal,
            dtype,
            fmm_level_to_order,
            source_extra_kwargs=source_extra_kwargs,
            kernel_extra_kwargs=kernel_extra_kwargs,
            self_extra_kwargs=self_extra_kwargs,
            translation_classes_data=translation_classes_data,
            preprocessed_mpole_dtype=preprocessed_mpole_dtype,
        )

        self.queue = queue

        if "target_to_source" in self.self_extra_kwargs and isinstance(
            self.self_extra_kwargs["target_to_source"], np.ndarray
        ):
            user_target_to_source = self.self_extra_kwargs["target_to_source"]
            sorted_target_ids = self.tree.sorted_target_ids.get(queue)
            if hasattr(self.tree, "sorted_source_ids"):
                sorted_source_ids = self.tree.sorted_source_ids.get(queue)
            else:
                sorted_source_ids = sorted_target_ids
            inverse_sorted_source_ids = np.empty_like(sorted_source_ids)
            inverse_sorted_source_ids[sorted_source_ids] = np.arange(
                len(sorted_source_ids), dtype=sorted_source_ids.dtype
            )
            target_to_source_sorted = inverse_sorted_source_ids[
                user_target_to_source[sorted_target_ids]
            ]

            self.self_extra_kwargs = dict(self.self_extra_kwargs)
            self.self_extra_kwargs["target_to_source"] = cl.array.to_device(
                self.queue, target_to_source_sorted
            )

        self.near_field_table = {}
        # list of tables for a single out kernel
        if isinstance(near_field_table, list):
            assert len(self.tree_indep.target_kernels) == 1
            self.near_field_table[self.tree_indep.target_kernels[0].__repr__()] = (
                near_field_table
            )
            self.n_tables = len(near_field_table)

        # single table
        elif isinstance(near_field_table, NearFieldInteractionTable):
            assert len(self.tree_indep.target_kernels) == 1
            self.near_field_table[self.tree_indep.target_kernels[0].__repr__()] = [
                near_field_table
            ]
            self.n_tables = 1

        # dictionary of lists of tables
        elif isinstance(near_field_table, dict):
            self.n_tables = {}
            for out_knl in self.tree_indep.target_kernels:
                if repr(out_knl) not in near_field_table:
                    raise RuntimeError(
                        "Missing nearfield table for %s." % repr(out_knl)
                    )
                if isinstance(
                    near_field_table[repr(out_knl)], NearFieldInteractionTable
                ):
                    near_field_table[repr(out_knl)] = [near_field_table[repr(out_knl)]]
                else:
                    assert isinstance(near_field_table[repr(out_knl)], list)

                self.n_tables[repr(out_knl)] = len(near_field_table[repr(out_knl)])

            self.near_field_table = near_field_table
        else:
            raise RuntimeError("Table type unrecognized.")

        self.quad_order = quad_order
        self.potential_kind = potential_kind

        # TODO: make all parameters table-specific (allow using inhomogeneous tables)
        kname = repr(self.tree_indep.target_kernels[0])
        self.root_table_source_box_extent = self.near_field_table[kname][
            0
        ].source_box_extent
        table_starting_level = int(
            np.round(
                np.log(self.tree.root_extent / self.root_table_source_box_extent)
                / np.log(2)
            )
        )
        self.table_starting_level = table_starting_level
        for kid in range(len(self.tree_indep.target_kernels)):
            kname = self.tree_indep.target_kernels[kid].__repr__()
            for lev, table in zip(
                range(len(self.near_field_table[kname])), self.near_field_table[kname]
            ):
                assert table.quad_order == self.quad_order

                if not table.is_built:
                    raise RuntimeError(
                        "Near field interaction table needs to be built "
                        "prior to being used"
                    )

                table_root_extent = table.source_box_extent * 2**lev
                assert (
                    abs(self.root_table_source_box_extent - table_root_extent) < 1e-15
                )

                # If the kernel cannot be scaled,
                # - tree_root_extent must be integral times of table_root_extent
                # - n_tables must be sufficient
                if not isinstance(self.n_tables, dict) and self.n_tables > 1:
                    if (
                        not abs(
                            int(self.tree.root_extent / table_root_extent)
                            * table_root_extent
                            - self.tree.root_extent
                        )
                        < 1e-15
                    ):
                        raise RuntimeError(
                            "Incompatible list of tables: the "
                            "source_box_extent of the root table must "
                            "divide the bounding box's extent by an integer."
                        )

            if not isinstance(self.n_tables, dict) and self.n_tables > 1:
                # this checks that the boxes at the highest level are covered
                if (
                    not self.tree.nlevels
                    <= len(self.near_field_table[kname]) + table_starting_level
                ):
                    raise RuntimeError(
                        "Insufficient list of tables: the "
                        "finest level mesh cells at level "
                        + str(self.tree.nlevels)
                        + " are not covered."
                    )

                # the check that the boxes at the coarsest level are covered is
                # deferred until trav.target_boxes is passed when invoking
                # eval_direct

        if list1_extra_kwargs is None:
            list1_extra_kwargs = {}
        else:
            list1_extra_kwargs = dict(list1_extra_kwargs)

        self._split_user_list1_extra_kwargs = dict(list1_extra_kwargs)
        self._helmholtz_split_multi_output = False

        self.helmholtz_split = (
            True if helmholtz_split is None else bool(helmholtz_split)
        )
        auto_cfg = dict(helmholtz_split_auto_config or {})
        self._helmholtz_split_auto_config = dict(auto_cfg)
        auto_mode = False

        if self.helmholtz_split:
            split_supported, split_reason = self._split_target_kernel_support_status()
            if not split_supported:
                logger.info(
                    "[split:disable] unsupported target kernels: %s",
                    split_reason,
                )
                self.helmholtz_split = False

        if self.helmholtz_split:
            base_table_supported, base_table_reason = (
                self._split_base_table_support_status()
            )
            if not base_table_supported:
                if helmholtz_split is None:
                    logger.info(
                        "[split:disable] %s",
                        base_table_reason,
                    )
                    self.helmholtz_split = False
                else:
                    raise RuntimeError(
                        "helmholtz_split requires Laplace-backed near-field tables; "
                        f"{base_table_reason}"
                    )

        if self.helmholtz_split:
            auto_enabled = bool(auto_cfg.get("enabled", False))
            auto_mode = auto_enabled or (
                isinstance(helmholtz_split_order, str)
                and helmholtz_split_order.strip().lower() == "auto"
            )

            if auto_mode:
                self.helmholtz_split_order = self._choose_auto_helmholtz_split_order(
                    auto_cfg
                )
            else:
                self.helmholtz_split_order = int(helmholtz_split_order)
        else:
            # Split order is irrelevant when split mode is disabled.
            self.helmholtz_split_order = 1

        if self.helmholtz_split_order < 1:
            raise ValueError("helmholtz_split_order must be >= 1")

        if self.helmholtz_split and len(self.tree_indep.target_kernels) > 1:
            self._helmholtz_split_multi_output = True
            if self.helmholtz_split_order > 1:
                logger.warning(
                    "split order %d requested with %d target kernels; "
                    "clamping to order 1 for multi-output split mode",
                    self.helmholtz_split_order,
                    len(self.tree_indep.target_kernels),
                )
                self.helmholtz_split_order = 1

        if auto_mode and self.helmholtz_split:
            max_rho_imag = float(auto_cfg.get("rho_imag_split_max", 8.0))
            disable_outside = bool(
                auto_cfg.get("disable_split_if_outside_coverage", False)
            )
            if (
                disable_outside
                and getattr(self, "_split_auto_rho_imag", 0.0) > max_rho_imag
            ):
                logger.warning(
                    "Imaginary rho %.3g exceeds configured split coverage %.3g. "
                    "Keeping split mode enabled because direct fallback requires "
                    "matching direct near-field tables.",
                    self._split_auto_rho_imag,
                    max_rho_imag,
                )
            elif getattr(self, "_split_auto_rho_imag", 0.0) > max_rho_imag:
                logger.warning(
                    "Imaginary rho %.3g exceeds configured split coverage %.3g; "
                    "continuing with split at configured max order",
                    self._split_auto_rho_imag,
                    max_rho_imag,
                )

        self.helmholtz_split_order1_legacy_subtraction = bool(
            helmholtz_split_order1_legacy_subtraction
        )

        self._helmholtz_split_p2p_pair_cache = {}
        self._helmholtz_split_p2p_pair_include_self_cache = {}
        self._helmholtz_split_term_p2p = {}
        self._helmholtz_split_term_p2p_include_self = {}
        self._helmholtz_split_remainder_p2p = None
        self._helmholtz_split_remainder_p2p_include_self = None
        self._helmholtz_split_remainder_p2p_cache_key = None
        self._helmholtz_split_remainder_p2p_include_self_cache_key = None
        self._helmholtz_split_remainder_kernel_cache = {}
        self._helmholtz_split_kernels = None
        self._helmholtz_split_constant_kernel = None
        self._helmholtz_split_power_kernels = {}
        self._helmholtz_split_power_log_kernels = {}
        self._helmholtz_split_kernel_wrapper_chain = ()
        self._helmholtz_split_kernel_wrapper_suffix = ""
        self._helmholtz_split_wrapper_derivative_order = 0
        self._helmholtz_split_smooth_interp_cache = {}
        self._helmholtz_split_table_cache_filename = None
        self._helmholtz_split_table_cache_root_extent = None
        self._helmholtz_split_table_build_config = None
        self._helmholtz_split_base_source_box_levels = None
        self._helmholtz_split_log_alpha_per_source_cache = {}
        self._helmholtz_split_log_alpha_per_source_dev_cache = {}
        self.helmholtz_split_term_tables = {}
        if helmholtz_split_term_tables is None:
            helmholtz_split_term_tables = {}
        elif not isinstance(helmholtz_split_term_tables, dict):
            raise TypeError(
                "helmholtz_split_term_tables must be a dict mapping term keys to tables"
            )

        for term_key, tables in helmholtz_split_term_tables.items():
            normalized_term_key = _normalize_helmholtz_split_term_key(term_key)

            if isinstance(tables, NearFieldInteractionTable):
                tables = [tables]
            elif not isinstance(tables, list):
                raise TypeError(
                    "helmholtz split term tables must be table or list of tables"
                )

            if len(tables) == 0:
                raise ValueError("helmholtz split term table list cannot be empty")

            for lev, table in enumerate(tables):
                if not isinstance(table, NearFieldInteractionTable):
                    raise TypeError(
                        "helmholtz split term list entries must be "
                        "NearFieldInteractionTable"
                    )
                if not table.is_built:
                    raise RuntimeError(
                        "helmholtz split term tables must be built before use"
                    )
                if table.quad_order != self.quad_order:
                    raise ValueError(
                        "helmholtz split term table quadrature order mismatch: "
                        f"expected {self.quad_order}, got {table.quad_order}"
                    )

                table_root_extent = table.source_box_extent * 2**lev
                if (
                    abs(self.root_table_source_box_extent - table_root_extent)
                    >= 1.0e-15
                ):
                    raise ValueError("helmholtz split term table root extent mismatch")

            self.helmholtz_split_term_tables[normalized_term_key] = tables

        if helmholtz_split_smooth_quad_order is None and self.helmholtz_split:
            if auto_mode:
                min_smooth = int(auto_cfg.get("smooth_quad_order_min", self.quad_order))
                add_per_order = int(auto_cfg.get("smooth_quad_order_per_order", 1))
                add_per_order_hard = int(
                    auto_cfg.get("smooth_quad_order_per_order_hard", 1)
                )
                if add_per_order < 0 or add_per_order_hard < 0:
                    raise ValueError(
                        "smooth quadrature per-order increments must be non-negative"
                    )

                hard_rho_imag = float(
                    auto_cfg.get(
                        "smooth_quad_order_hard_rho_imag",
                        4.0,
                    )
                )
                hard_rho_real = float(
                    auto_cfg.get("smooth_quad_order_hard_rho_real", 3.0)
                )
                if hard_rho_real <= 0.0:
                    raise ValueError("smooth_quad_order_hard_rho_real must be > 0")

                rho_real = float(getattr(self, "_split_auto_rho_real", 0.0))
                rho_imag = float(getattr(self, "_split_auto_rho_imag", 0.0))
                active_add_per_order = (
                    add_per_order_hard
                    if (rho_imag >= hard_rho_imag or rho_real >= hard_rho_real)
                    else add_per_order
                )

                rho_boost_start = float(
                    auto_cfg.get("smooth_quad_order_rho_boost_start", hard_rho_imag)
                )
                rho_boost_scale = float(
                    auto_cfg.get("smooth_quad_order_rho_boost_scale", 1.0)
                )
                if rho_boost_scale < 0.0:
                    raise ValueError("smooth_quad_order_rho_boost_scale must be >= 0")

                rho_boost = int(
                    math.ceil(max(0.0, rho_imag - rho_boost_start) * rho_boost_scale)
                )

                rho_boost_cap = auto_cfg.get("smooth_quad_order_rho_boost_cap", None)
                if rho_boost_cap is not None:
                    rho_boost = min(rho_boost, int(rho_boost_cap))

                rho_real_boost_start = float(
                    auto_cfg.get(
                        "smooth_quad_order_real_boost_start",
                        hard_rho_real,
                    )
                )
                rho_real_boost_scale = float(
                    auto_cfg.get("smooth_quad_order_real_boost_scale", 0.5)
                )
                if rho_real_boost_scale < 0.0:
                    raise ValueError("smooth_quad_order_real_boost_scale must be >= 0")

                rho_real_boost = int(
                    math.ceil(
                        max(0.0, rho_real - rho_real_boost_start) * rho_real_boost_scale
                    )
                )
                rho_real_boost_cap = auto_cfg.get(
                    "smooth_quad_order_real_boost_cap", None
                )
                if rho_real_boost_cap is not None:
                    rho_real_boost = min(rho_real_boost, int(rho_real_boost_cap))

                rho_boost += rho_real_boost

                base_smooth_order = (
                    self.quad_order
                    + max(0, self.helmholtz_split_order - 1) * active_add_per_order
                )
                helmholtz_split_smooth_quad_order = max(
                    min_smooth,
                    base_smooth_order + rho_boost,
                )

                smooth_quad_order_max = auto_cfg.get("smooth_quad_order_max", None)
                if smooth_quad_order_max is not None:
                    helmholtz_split_smooth_quad_order = min(
                        helmholtz_split_smooth_quad_order,
                        int(smooth_quad_order_max),
                    )
            else:
                helmholtz_split_smooth_quad_order = self.quad_order

        if helmholtz_split_smooth_quad_order is None:
            self.helmholtz_split_smooth_quad_order = None
        else:
            self.helmholtz_split_smooth_quad_order = int(
                helmholtz_split_smooth_quad_order
            )
            if self.helmholtz_split_smooth_quad_order < 1:
                raise ValueError("helmholtz_split_smooth_quad_order must be >= 1")

        self._helmholtz_split_smooth_quad_order_requested = (
            self.helmholtz_split_smooth_quad_order
        )

        if self.helmholtz_split:
            from sumpy.kernel import HelmholtzKernel, LaplaceKernel, YukawaKernel

            from volumential.table_manager import ConstantKernel

            out_knl = self.tree_indep.target_kernels[0]
            base_knl = out_knl.get_base_kernel()

            if not isinstance(base_knl, (HelmholtzKernel, YukawaKernel)):
                raise NotImplementedError(
                    "helmholtz_split currently supports only Helmholtz/Yukawa "
                    "kernels (optionally wrapped by supported derivatives)"
                )

            wrapper_chain = self._extract_helmholtz_split_kernel_wrapper_chain(
                out_knl,
                base_knl,
            )

            self._helmholtz_split_kernel_wrapper_chain = wrapper_chain
            self._helmholtz_split_kernel_wrapper_suffix = (
                self._format_helmholtz_split_kernel_wrapper_suffix(wrapper_chain)
            )
            self._helmholtz_split_wrapper_derivative_order = int(len(wrapper_chain))

            if base_knl.dim not in (2, 3):
                raise NotImplementedError(
                    "helmholtz_split currently supports only 2D/3D Helmholtz kernels"
                )

            self._initialize_helmholtz_split_table_umbrella(out_knl)

            for term_key, term_tables in self.helmholtz_split_term_tables.items():
                self._validate_helmholtz_split_term_table_umbrella(
                    term_key,
                    term_tables,
                )

            if self.helmholtz_split_order > 1:
                required_term_keys = set(
                    self._helmholtz_split_required_term_keys(base_knl.dim)
                )
                missing_term_keys = sorted(
                    required_term_keys - set(self.helmholtz_split_term_tables),
                )
                if missing_term_keys:
                    self._autobuild_helmholtz_split_term_tables(
                        out_knl,
                        missing_term_keys,
                    )

                missing_term_keys = sorted(
                    required_term_keys - set(self.helmholtz_split_term_tables),
                )
                if missing_term_keys:
                    missing_formatted = [
                        _format_helmholtz_split_term_key(term_key)
                        for term_key in missing_term_keys
                    ]
                    raise RuntimeError(
                        "helmholtz_split_order requires precomputed term tables for "
                        f"{missing_formatted}; provide helmholtz_split_term_tables "
                        "or use Laplace tables loaded from NearFieldInteractionTableManager "
                        "to enable auto-build"
                    )

            if list1_extra_kwargs.get("infer_kernel_scaling", False):
                logger.info(
                    "[split:init] infer_kernel_scaling requested; overriding with "
                    "split-aware scaling/displacement policy"
                )

            list1_extra_kwargs["infer_kernel_scaling"] = False
            list1_extra_kwargs.setdefault(
                "kernel_scaling_code",
                self._helmholtz_split_list1_scaling_code(base_knl.dim),
            )
            list1_extra_kwargs.setdefault(
                "kernel_displacement_code",
                self._helmholtz_split_list1_displacement_code(base_knl.dim),
            )

            self._helmholtz_split_kernels = (
                self._apply_helmholtz_split_kernel_wrappers(base_knl),
                self._apply_helmholtz_split_kernel_wrappers(
                    LaplaceKernel(base_knl.dim)
                ),
            )
            self._helmholtz_split_constant_kernel = ConstantKernel(base_knl.dim)

            wrapper_summary = (
                ",".join(f"{kind}:{value}" for kind, value in wrapper_chain)
                if wrapper_chain
                else "none"
            )
            term_key_summary = (
                ",".join(
                    _format_helmholtz_split_term_key(term_key)
                    for term_key in sorted(self.helmholtz_split_term_tables)
                )
                if self.helmholtz_split_term_tables
                else "none"
            )

            if auto_mode:
                rho_real = float(getattr(self, "_split_auto_rho_real", 0.0))
                rho_imag = float(getattr(self, "_split_auto_rho_imag", 0.0))
                rho_summary = f"{rho_real:.3g},{rho_imag:.3g}"
            else:
                rho_summary = "n/a"

            logger.info(
                "[split:init] dim=%d out=%s base=%s wrappers=%s order=%d "
                "smooth_q=%s auto=%s rho=%s term_keys=%s",
                base_knl.dim,
                out_knl.__class__.__name__,
                base_knl.__class__.__name__,
                wrapper_summary,
                self.helmholtz_split_order,
                self.helmholtz_split_smooth_quad_order,
                auto_mode,
                rho_summary,
                term_key_summary,
            )
        self.list1_extra_kwargs = list1_extra_kwargs
        self._table_layout_validation_cache = set()
        self._nearfield_device_payload_cache = OrderedDict()
        self._nearfield_device_payload_cache_max = 16

    # }}} End constructor

    # {{{ data vector utilities

    def _get_cached_nearfield_payload(
        self,
        cache_key,
        queue,
        table0,
        near_field_tables,
        eval_dtype,
    ):
        payload = self._nearfield_device_payload_cache.get(cache_key)
        if payload is not None:
            self._nearfield_device_payload_cache.move_to_end(cache_key)
            return payload

        distinct_numbers = set()
        for vec in table0.interaction_case_vecs:
            for cvc in vec:
                distinct_numbers.add(cvc)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        case_indices_dev = cl.array.to_device(queue, table0.case_indices)
        symmetry_maps = table0._get_online_symmetry_maps()
        mode_qpoint_map_dev = cl.array.to_device(
            queue, symmetry_maps["mode_qpoint_map"]
        )
        mode_case_map_dev = cl.array.to_device(queue, symmetry_maps["mode_case_map"])
        mode_case_scale = symmetry_maps.get("mode_case_scale")
        if mode_case_scale is None:
            mode_case_scale = np.ones(
                (table0.n_q_points, table0.n_cases),
                dtype=np.int8,
            )

        (
            table_data_combined,
            mode_nmlz_combined,
            exterior_mode_nmlz_combined,
            table_entry_ids,
            table_entry_scales,
        ) = _prepare_table_data_and_entry_map(near_field_tables)

        if table_data_combined.dtype != eval_dtype:
            table_data_combined = table_data_combined.astype(eval_dtype)
        if mode_nmlz_combined.dtype != eval_dtype:
            mode_nmlz_combined = mode_nmlz_combined.astype(eval_dtype)
        if exterior_mode_nmlz_combined.dtype != eval_dtype:
            exterior_mode_nmlz_combined = exterior_mode_nmlz_combined.astype(eval_dtype)
        if table_entry_scales.dtype != eval_dtype:
            table_entry_scales = table_entry_scales.astype(eval_dtype)
        if mode_case_scale.dtype != eval_dtype:
            mode_case_scale = mode_case_scale.astype(eval_dtype)

        table_data_shapes = {
            "n_tables": len(near_field_tables),
            "n_q_points": table0.n_q_points,
            "n_cases": table0.n_cases,
            "n_table_entries": table_data_combined.shape[1],
        }

        payload = {
            "base": base,
            "shift": shift,
            "case_indices_dev": case_indices_dev,
            "mode_qpoint_map_dev": mode_qpoint_map_dev,
            "mode_case_map_dev": mode_case_map_dev,
            "mode_case_scale_dev": cl.array.to_device(queue, mode_case_scale),
            "table_data_dev": cl.array.to_device(queue, table_data_combined),
            "mode_nmlz_dev": cl.array.to_device(queue, mode_nmlz_combined),
            "exterior_mode_nmlz_dev": cl.array.to_device(
                queue, exterior_mode_nmlz_combined
            ),
            "table_entry_ids_dev": cl.array.to_device(queue, table_entry_ids),
            "table_entry_scales_dev": cl.array.to_device(queue, table_entry_scales),
            "table_data_shapes": table_data_shapes,
        }
        self._nearfield_device_payload_cache[cache_key] = payload
        while (
            len(self._nearfield_device_payload_cache)
            > self._nearfield_device_payload_cache_max
        ):
            self._nearfield_device_payload_cache.popitem(last=False)
        return payload

    @property
    def _actx(self):
        return self.tree_indep._setup_actx

    def multipole_expansion_zeros(self, actx=None):
        if actx is None:
            actx = self._actx
        return SumpyExpansionWrangler.multipole_expansion_zeros(self, actx)

    def local_expansion_zeros(self, actx=None):
        if actx is None:
            actx = self._actx
        return SumpyExpansionWrangler.local_expansion_zeros(self, actx)

    def output_zeros(self, actx=None):
        if actx is None:
            actx = self._actx
        return SumpyExpansionWrangler.output_zeros(self, actx)

    def reorder_sources(self, source_array):
        return SumpyExpansionWrangler.reorder_sources(self, source_array)

    def reorder_targets(self, target_array):
        if not hasattr(self, "_user_target_ids"):
            self._user_target_ids = inverse_id_map(
                self.queue, self.tree.sorted_target_ids
            )
        return target_array.with_queue(self.queue)[self._user_target_ids]

    def reorder_potentials(self, potentials):
        return SumpyExpansionWrangler.reorder_potentials(self, potentials)

    def finalize_potentials(self, potentials):
        # return potentials
        return SumpyExpansionWrangler.finalize_potentials(self, self._actx, potentials)

    # }}} End data vector utilities

    # {{{ formation & coarsening of multipoles

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        mpoles = SumpyExpansionWrangler.form_multipoles(
            self, self._actx, level_start_source_box_nrs, source_boxes, src_weights
        )
        return mpoles, SumpyTimingFuture(self.queue, [])

    def coarsen_multipoles(
        self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
    ):
        mpoles = SumpyExpansionWrangler.coarsen_multipoles(
            self,
            self._actx,
            level_start_source_parent_box_nrs,
            source_parent_boxes,
            mpoles,
        )
        return mpoles, SumpyTimingFuture(self.queue, [])

    # }}} End formation & coarsening of multipoles

    # {{{ direct evaluation of near field interactions

    def eval_direct_single_out_kernel(
        self,
        out_pot,
        out_kernel,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        mode_coefs,
        near_field_tables=None,
        list1_extra_kwargs=None,
    ):

        # NOTE: mode_coefs are similar to source_weights BUT
        # do not include quadrature weights (purely function
        # expansiona coefficients)

        queue = self.queue

        if 0:
            print("Returns range for list1")
            out_pot[:] = cl.array.to_device(queue, np.arange(len(out_pot)))
            return out_pot, None

        kname = out_kernel.__repr__()

        if near_field_tables is None:
            near_field_tables = self.near_field_table[kname]

        if list1_extra_kwargs is None:
            list1_extra_kwargs = self.list1_extra_kwargs

        n_tables_local = len(near_field_tables)
        table0 = near_field_tables[0]
        table_root_extent = table0.source_box_extent
        table_starting_level = int(
            np.round(np.log(self.tree.root_extent / table_root_extent) / np.log(2))
        )

        use_multilevel_tables = n_tables_local > 1

        if use_multilevel_tables:
            # this checks that the boxes at the coarsest level
            # and allows for some round-off error
            min_lev = np.min(self.tree.box_levels.get(queue)[target_boxes.get(queue)])
            largest_cell_extent = self.tree.root_extent * 0.5**min_lev
            if not table0.source_box_extent >= (largest_cell_extent - 1e-15):
                raise RuntimeError(
                    "Insufficient list of tables: the "
                    "coarsest level mesh cells at level "
                    + str(min_lev)
                    + " are not covered."
                )

        symmetry_source_direction = _extract_symmetry_source_direction(
            out_kernel,
            self.source_extra_kwargs,
            queue,
        )
        for table in near_field_tables:
            table.symmetry_source_direction = symmetry_source_direction

        configured_dtype = np.dtype(self.dtype)
        if out_kernel.is_complex_valued:
            if configured_dtype.kind == "c":
                eval_dtype = configured_dtype
            else:
                eval_dtype = (
                    np.complex64 if configured_dtype == np.float32 else np.complex128
                )
        else:
            if configured_dtype.kind == "f":
                eval_dtype = configured_dtype
            else:
                eval_dtype = (
                    np.float32 if configured_dtype == np.complex64 else np.float64
                )

        cache_key = (
            kname,
            tuple(int(np.asarray(tbl.data).ctypes.data) for tbl in near_field_tables),
            tuple(int(np.asarray(tbl.data).size) for tbl in near_field_tables),
            tuple(_table_data_fingerprint(tbl.data) for tbl in near_field_tables),
            np.dtype(eval_dtype).str,
            None
            if symmetry_source_direction is None
            else tuple(np.asarray(symmetry_source_direction, dtype=float).tolist()),
        )
        payload = self._get_cached_nearfield_payload(
            cache_key,
            queue,
            table0,
            near_field_tables,
            eval_dtype,
        )

        base = payload["base"]
        shift = payload["shift"]
        case_indices_dev = payload["case_indices_dev"]
        mode_qpoint_map_dev = payload["mode_qpoint_map_dev"]
        mode_case_map_dev = payload["mode_case_map_dev"]
        mode_case_scale_dev = payload["mode_case_scale_dev"]
        table_data_shapes = payload["table_data_shapes"]
        assert table_data_shapes["n_q_points"] == len(table0.mode_normalizers)

        from volumential.list1 import NearFieldFromCSR

        near_field = NearFieldFromCSR(
            out_kernel,
            table_data_shapes,
            potential_kind=self.potential_kind,
            **list1_extra_kwargs,
        )

        table_data_combined = payload["table_data_dev"]
        mode_nmlz_combined = payload["mode_nmlz_dev"]
        exterior_mode_nmlz_combined = payload["exterior_mode_nmlz_dev"]
        table_entry_ids = payload["table_entry_ids_dev"]
        table_entry_scales = payload["table_entry_scales_dev"]
        particle_local_ids = _compute_box_local_ids(queue, self.tree, table0.n_q_points)

        _validate_table_box_particle_layout_cached(
            queue,
            self.tree,
            target_boxes,
            neighbor_source_boxes_lists,
            table0.n_q_points,
            self._table_layout_validation_cache,
        )

        aligned_nboxes = self.tree.box_centers.shape[1]
        source_counts_nonchild = np.zeros(aligned_nboxes, dtype=np.int32)
        target_counts_nonchild = np.zeros(aligned_nboxes, dtype=np.int32)
        source_counts_nonchild[: len(self.tree.box_target_counts_nonchild)] = (
            self.tree.box_target_counts_nonchild.get(queue)
        )
        target_counts_nonchild[: len(self.tree.box_target_counts_nonchild)] = (
            self.tree.box_target_counts_nonchild.get(queue)
        )
        source_counts_nonchild = cl.array.to_device(queue, source_counts_nonchild)
        target_counts_nonchild = cl.array.to_device(queue, target_counts_nonchild)

        # NOTE: box_sources for this evaluation should be "box_targets".
        # This is due to the special features of how box-FMM works.

        res, evt = near_field(
            queue,
            result=out_pot,
            box_centers=self.tree.box_centers,
            box_levels=self.tree.box_levels,
            box_source_counts_nonchild=source_counts_nonchild,
            box_source_starts=self.tree.box_target_starts,
            box_target_counts_nonchild=target_counts_nonchild,
            box_target_starts=self.tree.box_target_starts,
            case_indices=case_indices_dev,
            mode_qpoint_map=mode_qpoint_map_dev,
            mode_case_map=mode_case_map_dev,
            mode_case_scale=mode_case_scale_dev,
            encoding_base=base,
            encoding_shift=shift,
            mode_nmlz_combined=mode_nmlz_combined,
            exterior_mode_nmlz_combined=exterior_mode_nmlz_combined,
            table_entry_ids=table_entry_ids,
            table_entry_scales=table_entry_scales,
            neighbor_source_boxes_starts=neighbor_source_boxes_starts,
            root_extent=self.tree.root_extent,
            neighbor_source_boxes_lists=neighbor_source_boxes_lists,
            mode_coefs=mode_coefs,
            source_mode_ids=particle_local_ids,
            table_data_combined=table_data_combined,
            target_boxes=target_boxes,
            target_point_ids=particle_local_ids,
            table_root_extent=table_root_extent,
            table_starting_level=table_starting_level,
        )

        # print(near_field.get_kernel())
        # import pudb; pu.db

        assert res is out_pot

        # sorted_target_ids=self.tree.user_source_ids,
        # user_source_ids=self.tree.user_source_ids)

        # FIXME: lazy evaluation sometimes returns incorrect results
        res.finish()

        return out_pot, evt

    def eval_direct(
        self,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        mode_coefs,
    ):
        pot = self.output_zeros()
        events = []
        for i in range(len(self.tree_indep.target_kernels)):
            # print("processing near-field of out_kernel", i)
            out_knl = self.tree_indep.target_kernels[i]
            kernel_list1_extra_kwargs = None
            if self.helmholtz_split:
                kernel_list1_extra_kwargs = (
                    self._split_list1_extra_kwargs_for_out_kernel(out_knl)
                )

            pot[i], evt = self.eval_direct_single_out_kernel(
                pot[i],
                out_knl,
                target_boxes,
                neighbor_source_boxes_starts,
                neighbor_source_boxes_lists,
                mode_coefs,
                list1_extra_kwargs=kernel_list1_extra_kwargs,
            )
            events.append(evt)

        for out_pot in pot:
            out_pot.finish()

        return (pot, SumpyTimingFuture(self.queue, events))

    def eval_direct_helmholtz_split_correction(
        self,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        src_weights,
        src_func=None,
        _split_out_kernel=None,
    ):
        if not self.helmholtz_split:
            return self.output_zeros(), SumpyTimingFuture(self.queue, [])

        if _split_out_kernel is None and len(self.tree_indep.target_kernels) > 1:
            if self.helmholtz_split_order > 1:
                raise NotImplementedError(
                    "multi-output split correction currently supports split_order=1"
                )

            corrections = []
            aggregated_events = []
            for out_knl in self.tree_indep.target_kernels:
                corr_i, timing_i = self.eval_direct_helmholtz_split_correction(
                    target_boxes,
                    neighbor_source_boxes_starts,
                    neighbor_source_boxes_lists,
                    src_weights,
                    src_func=src_func,
                    _split_out_kernel=out_knl,
                )
                aggregated_events.extend(getattr(timing_i, "events", []) or [])
                corr_i_oa = (
                    corr_i
                    if isinstance(corr_i, np.ndarray) and corr_i.dtype == object
                    else obj_array_1d([corr_i])
                )
                if len(corr_i_oa) != 1:
                    raise RuntimeError(
                        "split correction per output kernel must have one component"
                    )
                corrections.append(corr_i_oa[0])

            return (
                obj_array_1d(corrections),
                SumpyTimingFuture(self.queue, aggregated_events),
            )

        if _split_out_kernel is None:
            _split_out_kernel = self.tree_indep.target_kernels[0]

        _, _, effective_split_smooth_quad_order = self._set_active_split_kernel(
            _split_out_kernel
        )

        shared_kwargs = {}
        shared_kwargs.update(self.self_extra_kwargs)
        shared_kwargs.update(self.source_extra_kwargs)
        shared_kwargs.update(self.box_source_list_kwargs())
        shared_kwargs.update(self.box_target_list_kwargs())

        if "target_to_source" in shared_kwargs and isinstance(
            shared_kwargs["target_to_source"], np.ndarray
        ):
            shared_kwargs["target_to_source"] = cl.array.to_device(
                self.queue, shared_kwargs["target_to_source"]
            )

        smooth_quad_order = effective_split_smooth_quad_order
        smooth_quad_order_int = (
            int(smooth_quad_order) if smooth_quad_order is not None else None
        )

        use_interp_smooth_quad = (
            smooth_quad_order_int is not None
            and smooth_quad_order_int > self.quad_order
            and src_func is not None
        )

        use_base_quad_series_remainder = (
            self.helmholtz_split_order > 1
            and smooth_quad_order_int is not None
            and smooth_quad_order_int == self.quad_order
            and src_func is not None
        )

        use_series_remainder_path = self.helmholtz_split_order > 1 or (
            self.helmholtz_split_order == 1
            and not self.helmholtz_split_order1_legacy_subtraction
        )

        if self.helmholtz_split_order > 1 and not (
            use_interp_smooth_quad or use_base_quad_series_remainder
        ):
            raise RuntimeError(
                "helmholtz_split_order > 1 requires source function samples and "
                "smooth quadrature order >= q"
            )

        if use_interp_smooth_quad:
            smooth_data = self._build_helmholtz_split_smooth_correction_sources(
                src_func,
                int(smooth_quad_order_int),
                allow_node_overlap=use_series_remainder_path,
            )
            strength = obj_array_1d([smooth_data["strength"]])
            split_shared_kwargs = dict(shared_kwargs)
            split_shared_kwargs.update(smooth_data["source_kwargs"])
            split_shared_kwargs.pop("target_to_source", None)
            exclude_self = False
            max_nsources_in_one_box = smooth_data["max_nsources_in_one_box"]
        else:
            strength = obj_array_1d([src_weights])
            split_shared_kwargs = shared_kwargs
            exclude_self = self.tree_indep.exclude_self
            max_nsources_in_one_box = self.max_nsources_in_one_box

        if use_series_remainder_path:
            p2p_remainder = self._get_helmholtz_split_remainder_p2p(
                exclude_self=exclude_self
            )
            p2p_helm = None
            p2p_lap = None
        else:
            p2p_helm, p2p_lap = self._get_helmholtz_split_p2p_pair(
                exclude_self=exclude_self
            )
            p2p_remainder = None

        def _run_p2p_from_csr(
            p2p_obj,
            extra_kwargs,
            *,
            strength_arg,
            max_nsources_in_one_box_arg,
        ):
            from sumpy.array_context import is_cl_cpu

            targets = extra_kwargs["targets"]
            sources = extra_kwargs["sources"]

            is_gpu = not is_cl_cpu(self._actx)
            if is_gpu:
                source_dtype = sources[0].dtype
                if (
                    isinstance(strength_arg, np.ndarray)
                    and strength_arg.dtype == object
                ):
                    strength_dtype = strength_arg[0].dtype
                else:
                    strength_dtype = strength_arg.dtype
            else:
                source_dtype = None
                strength_dtype = None

            knl = p2p_obj.get_cached_kernel(
                max_nsources_in_one_box=max_nsources_in_one_box_arg,
                max_ntargets_in_one_box=self.max_ntargets_in_one_box,
                local_mem_size=self._actx.queue.device.local_mem_size,
                is_gpu=is_gpu,
                source_dtype=source_dtype,
                strength_dtype=strength_dtype,
            )

            loopy_kwargs = dict(extra_kwargs)
            loopy_kwargs.pop("targets", None)
            loopy_kwargs.pop("sources", None)
            loopy_kwargs.update(
                {
                    "target_boxes": target_boxes,
                    "source_box_starts": neighbor_source_boxes_starts,
                    "source_box_lists": neighbor_source_boxes_lists,
                    "strength": strength_arg,
                }
            )

            result = self._actx.call_loopy(
                knl,
                targets=targets,
                sources=sources,
                **loopy_kwargs,
            )

            return obj_array_1d(
                [result[f"result_s{i}"] for i in range(p2p_obj.nresults)]
            )

        if use_series_remainder_path:
            correction = _run_p2p_from_csr(
                p2p_remainder,
                split_shared_kwargs,
                strength_arg=strength,
                max_nsources_in_one_box_arg=max_nsources_in_one_box,
            )

            for (
                term_key,
                term_kernel,
                term_coeff,
            ) in self._helmholtz_split_extra_terms():
                term_tables = self._get_helmholtz_split_term_tables(term_key)
                term_table = self._eval_direct_helmholtz_split_term_table(
                    target_boxes,
                    neighbor_source_boxes_starts,
                    neighbor_source_boxes_lists,
                    src_func,
                    term_kernel,
                    term_tables,
                    term_key=term_key,
                )

                term_contribution = term_table[0]
                term_kind, term_power = _normalize_helmholtz_split_term_key(term_key)

                if (
                    self.tree.dimensions == 2
                    and term_kind == "power_log"
                    and len(term_tables) == 1
                ):
                    power_kernel = self._get_helmholtz_split_power_kernel(term_power)
                    beta_mode = (
                        str(
                            self._helmholtz_split_auto_config.get(
                                "power_log_single_table_beta_mode",
                                "p2p",
                            )
                        )
                        .strip()
                        .lower()
                    )

                    if beta_mode == "p2p":
                        beta_kwargs = dict(split_shared_kwargs)
                        if not exclude_self:
                            beta_kwargs.pop("target_to_source", None)

                        if use_interp_smooth_quad:
                            beta_strength_scalar = (
                                self._fold_helmholtz_split_log_alpha_into_strength(
                                    smooth_data["strength"],
                                    float(term_tables[0].source_box_extent),
                                    box_source_starts=smooth_data["source_kwargs"][
                                        "box_source_starts"
                                    ],
                                    box_source_counts=smooth_data["source_kwargs"][
                                        "box_source_counts_nonchild"
                                    ],
                                )
                            )
                        else:
                            beta_strength_scalar = (
                                self._fold_helmholtz_split_log_alpha_into_strength(
                                    src_weights,
                                    float(term_tables[0].source_box_extent),
                                )
                            )

                        beta_strength = obj_array_1d([beta_strength_scalar])
                        p2p_power = self._get_helmholtz_split_term_p2p(
                            power_kernel,
                            exclude_self=exclude_self,
                        )
                        beta_power_term = _run_p2p_from_csr(
                            p2p_power,
                            beta_kwargs,
                            strength_arg=beta_strength,
                            max_nsources_in_one_box_arg=max_nsources_in_one_box,
                        )
                    elif beta_mode == "table":
                        beta_mode_coefs = (
                            self._fold_helmholtz_split_log_alpha_into_strength(
                                src_func,
                                float(term_tables[0].source_box_extent),
                            )
                        )

                        power_term_key = _normalize_helmholtz_split_term_key(
                            ("power", term_power)
                        )
                        power_term_tables = (
                            self._get_or_autobuild_helmholtz_split_term_tables(
                                power_term_key
                            )
                        )

                        beta_power_term = self._eval_direct_helmholtz_split_term_table(
                            target_boxes,
                            neighbor_source_boxes_starts,
                            neighbor_source_boxes_lists,
                            beta_mode_coefs,
                            power_kernel,
                            power_term_tables,
                            term_key=power_term_key,
                        )
                    else:
                        raise ValueError(
                            "power_log_single_table_beta_mode must be 'table' or 'p2p'"
                        )

                    beta_term = beta_power_term[0]
                    if isinstance(term_contribution, cl.array.Array) and not isinstance(
                        beta_term, cl.array.Array
                    ):
                        beta_term = cl.array.to_device(
                            self.queue,
                            np.ascontiguousarray(np.asarray(beta_term)),
                        )
                    elif isinstance(beta_term, cl.array.Array) and not isinstance(
                        term_contribution, cl.array.Array
                    ):
                        beta_term = beta_term.get(self.queue)

                    term_contribution = term_contribution + beta_term

                term_scale = correction[0].dtype.type(term_coeff)
                correction = obj_array_1d(
                    [correction[0] + term_contribution * term_scale]
                )
        else:
            helm_kwargs = dict(split_shared_kwargs)
            helm_kwargs.update(self.kernel_extra_kwargs)
            helm_result = _run_p2p_from_csr(
                p2p_helm,
                helm_kwargs,
                strength_arg=strength,
                max_nsources_in_one_box_arg=max_nsources_in_one_box,
            )
            lap_result = _run_p2p_from_csr(
                p2p_lap,
                split_shared_kwargs,
                strength_arg=strength,
                max_nsources_in_one_box_arg=max_nsources_in_one_box,
            )

            correction = obj_array_1d([helm_result[0] - lap_result[0]])

        # When self terms are excluded from correction P2P, add back the finite
        # r->0 limit of (Helmholtz-Laplace) analytically.
        if exclude_self:
            diag_term = self._helmholtz_split_self_diagonal_term(
                src_weights,
                shared_kwargs,
                like=correction[0],
            )
            if diag_term is not None:
                correction = obj_array_1d([correction[0] + diag_term])

        return correction, SumpyTimingFuture(self.queue, [])

    def _get_helmholtz_split_p2p_pair(self, *, exclude_self):
        from sumpy import P2PFromCSR

        wrapper_key = tuple(self._helmholtz_split_kernel_wrapper_chain)
        cache = (
            self._helmholtz_split_p2p_pair_cache
            if exclude_self
            else self._helmholtz_split_p2p_pair_include_self_cache
        )

        if wrapper_key not in cache:
            helm_knl, lap_knl = self._helmholtz_split_kernels
            cache[wrapper_key] = (
                P2PFromCSR(
                    [helm_knl],
                    bool(exclude_self),
                    value_dtypes=[self.dtype],
                ),
                P2PFromCSR(
                    [lap_knl],
                    bool(exclude_self),
                    value_dtypes=[self.dtype],
                ),
            )

        return cache[wrapper_key]

    def _get_helmholtz_split_term_p2p(self, term_kernel, *, exclude_self):
        from sumpy import P2PFromCSR

        kname = repr(term_kernel)
        cache = (
            self._helmholtz_split_term_p2p
            if exclude_self
            else self._helmholtz_split_term_p2p_include_self
        )

        if kname not in cache:
            cache[kname] = P2PFromCSR(
                [term_kernel],
                bool(exclude_self),
                value_dtypes=[self.dtype],
            )

        return cache[kname]

    def _extract_helmholtz_split_kernel_wrapper_chain(self, out_knl, base_knl):
        wrappers = []
        cur_knl = out_knl

        while cur_knl is not base_knl:
            if isinstance(cur_knl, AxisTargetDerivative):
                wrappers.append(("axis_target", int(cur_knl.axis)))
                cur_knl = cur_knl.inner_kernel
                continue

            if isinstance(cur_knl, AxisSourceDerivative):
                wrappers.append(("axis_source", int(cur_knl.axis)))
                cur_knl = cur_knl.inner_kernel
                continue

            if isinstance(cur_knl, DirectionalSourceDerivative):
                wrappers.append(("directional_source", str(cur_knl.dir_vec_name)))
                cur_knl = cur_knl.inner_kernel
                continue

            raise NotImplementedError(
                "helmholtz_split currently supports only axis target/source "
                "derivative wrappers around Helmholtz/Yukawa kernels"
            )

        return tuple(wrappers)

    def _format_helmholtz_split_kernel_wrapper_suffix(self, wrappers):
        if not wrappers:
            return ""

        parts = []
        for kind, value in wrappers:
            if kind == "axis_target":
                parts.append(f"td{int(value)}")
            elif kind == "axis_source":
                parts.append(f"sd{int(value)}")
            elif kind == "directional_source":
                parts.append(f"sdir_{value}")
            else:
                raise RuntimeError(f"unsupported split wrapper kind: {kind}")

        return "__" + "__".join(parts)

    @staticmethod
    def _apply_helmholtz_split_kernel_wrappers_with_chain(kernel, wrapper_chain):
        wrapped = kernel
        for kind, value in reversed(wrapper_chain):
            if kind == "axis_target":
                wrapped = AxisTargetDerivative(int(value), wrapped)
            elif kind == "axis_source":
                wrapped = AxisSourceDerivative(int(value), wrapped)
            elif kind == "directional_source":
                wrapped = DirectionalSourceDerivative(wrapped, str(value))
            else:
                raise RuntimeError(f"unsupported split wrapper kind: {kind}")

        return wrapped

    def _apply_helmholtz_split_kernel_wrappers(self, kernel):
        return self._apply_helmholtz_split_kernel_wrappers_with_chain(
            kernel,
            self._helmholtz_split_kernel_wrapper_chain,
        )

    @staticmethod
    def _split_table_kernel(table):
        table_kernel = getattr(table, "integral_knl", None)
        if table_kernel is None:
            table_kernel = getattr(table, "sumpy_kernel", None)
        return table_kernel

    def _split_base_table_support_status(self):
        target_kernels = list(self.tree_indep.target_kernels)
        if not target_kernels:
            return False, "no target kernels"

        for out_knl in target_kernels:
            base_knl = out_knl.get_base_kernel()
            wrapper_chain = self._extract_helmholtz_split_kernel_wrapper_chain(
                out_knl,
                base_knl,
            )

            expected_kernel = self._apply_helmholtz_split_kernel_wrappers_with_chain(
                LaplaceKernel(base_knl.dim),
                wrapper_chain,
            )
            expected_repr = repr(expected_kernel)

            table_key = repr(out_knl)
            base_tables = self.near_field_table.get(table_key, [])
            if not base_tables:
                return (
                    False,
                    f"missing near-field tables for {table_key}",
                )

            for lev, table in enumerate(base_tables):
                table_kernel = self._split_table_kernel(table)
                if table_kernel is None:
                    return (
                        False,
                        f"{table_key} level {lev} is missing table kernel metadata "
                        "(cache-key design cannot disambiguate split base kernel)",
                    )

                table_repr = repr(table_kernel)
                if table_repr != expected_repr:
                    return (
                        False,
                        f"{table_key} level {lev} kernel mismatch: expected "
                        f"{expected_repr}, got {table_repr} (cache-key design)",
                    )

        return True, ""

    def _split_target_kernel_support_status(self):
        from sumpy.kernel import HelmholtzKernel, YukawaKernel

        target_kernels = list(self.tree_indep.target_kernels)
        if not target_kernels:
            return False, "no target kernels"

        base_dim = None
        base_kind = None
        base_param_name = None

        for out_knl in target_kernels:
            base_knl = out_knl.get_base_kernel()
            if not isinstance(base_knl, (HelmholtzKernel, YukawaKernel)):
                return (
                    False,
                    f"{out_knl.__class__.__name__} is not a Helmholtz/Yukawa kernel",
                )

            if int(base_knl.dim) not in (2, 3):
                return False, f"unsupported dimension {base_knl.dim}"

            try:
                self._extract_helmholtz_split_kernel_wrapper_chain(out_knl, base_knl)
            except NotImplementedError as exc:
                return False, str(exc)

            param_name = self._split_wave_number_parameter_name(base_knl)
            if not isinstance(param_name, str) or not param_name:
                return (
                    False,
                    f"{base_knl.__class__.__name__} missing split parameter name",
                )

            if base_dim is None:
                base_dim = int(base_knl.dim)
                base_kind = type(base_knl)
                base_param_name = param_name
            else:
                if int(base_knl.dim) != base_dim:
                    return False, "mixed dimensions in split target kernels"
                if type(base_knl) is not base_kind:
                    return (
                        False,
                        "mixed Helmholtz/Yukawa base kernels in split mode are unsupported",
                    )
                if param_name != base_param_name:
                    return (
                        False,
                        "mixed split parameter names across target kernels are unsupported",
                    )

        return True, ""

    def _set_active_split_kernel(self, out_knl):
        from sumpy.kernel import HelmholtzKernel, LaplaceKernel, YukawaKernel

        from volumential.table_manager import ConstantKernel

        base_knl = out_knl.get_base_kernel()
        if not isinstance(base_knl, (HelmholtzKernel, YukawaKernel)):
            raise NotImplementedError(
                "split mode currently supports only Helmholtz/Yukawa output kernels"
            )

        wrapper_chain = self._extract_helmholtz_split_kernel_wrapper_chain(
            out_knl,
            base_knl,
        )

        self._helmholtz_split_kernel_wrapper_chain = wrapper_chain
        self._helmholtz_split_kernel_wrapper_suffix = (
            self._format_helmholtz_split_kernel_wrapper_suffix(wrapper_chain)
        )
        self._helmholtz_split_wrapper_derivative_order = int(len(wrapper_chain))
        self._helmholtz_split_kernels = (
            self._apply_helmholtz_split_kernel_wrappers(base_knl),
            self._apply_helmholtz_split_kernel_wrappers(LaplaceKernel(base_knl.dim)),
        )
        self._helmholtz_split_constant_kernel = ConstantKernel(base_knl.dim)

        effective_split_smooth_quad_order = getattr(
            self,
            "_helmholtz_split_smooth_quad_order_requested",
            self.helmholtz_split_smooth_quad_order,
        )

        has_directional_source_wrapper = any(
            kind == "directional_source" for kind, _ in wrapper_chain
        )
        if (
            has_directional_source_wrapper
            and effective_split_smooth_quad_order is not None
            and effective_split_smooth_quad_order > self.quad_order
        ):
            logger.warning(
                "split smooth quad order %d requested for directional source "
                "derivatives; clamping to base quadrature order %d",
                effective_split_smooth_quad_order,
                self.quad_order,
            )
            effective_split_smooth_quad_order = int(self.quad_order)

        return base_knl, wrapper_chain, effective_split_smooth_quad_order

    def _split_list1_extra_kwargs_for_out_kernel(self, out_knl):
        if not self.helmholtz_split:
            return self.list1_extra_kwargs

        base_knl, wrapper_chain, _ = self._set_active_split_kernel(out_knl)
        derivative_order = int(len(wrapper_chain))

        list1_kwargs = dict(self.list1_extra_kwargs)
        user_kwargs = self._split_user_list1_extra_kwargs
        list1_kwargs["infer_kernel_scaling"] = False

        if "kernel_scaling_code" not in user_kwargs:
            list1_kwargs["kernel_scaling_code"] = (
                self._helmholtz_split_list1_scaling_code(
                    base_knl.dim,
                    derivative_order=derivative_order,
                )
            )

        if "kernel_displacement_code" not in user_kwargs:
            list1_kwargs["kernel_displacement_code"] = (
                self._helmholtz_split_list1_displacement_code(
                    base_knl.dim,
                    derivative_order=derivative_order,
                )
            )

        return list1_kwargs

    def _helmholtz_split_kernel_type_name(self, base_name):
        suffix = self._helmholtz_split_kernel_wrapper_suffix
        if suffix:
            return f"{base_name}{suffix}"
        return base_name

    def _split_wave_number_parameter_name(self, base_knl):
        get_base_kernel = getattr(base_knl, "get_base_kernel", None)
        if callable(get_base_kernel):
            try:
                base_knl = get_base_kernel()
            except Exception:
                pass

        if isinstance(base_knl, HelmholtzKernel):
            return getattr(base_knl, "helmholtz_k_name", None)
        if isinstance(base_knl, YukawaKernel):
            return getattr(base_knl, "yukawa_lambda_name", None)
        return None

    def _split_wave_number(self, base_knl, *, what):
        get_base_kernel = getattr(base_knl, "get_base_kernel", None)
        if callable(get_base_kernel):
            try:
                base_knl = get_base_kernel()
            except Exception:
                pass

        param_name = self._split_wave_number_parameter_name(base_knl)
        if not isinstance(param_name, str) or not param_name:
            raise RuntimeError(
                f"{base_knl.__class__.__name__} does not expose a split parameter "
                f"name while evaluating {what}"
            )
        if param_name not in self.kernel_extra_kwargs:
            raise TypeError(
                f"missing kernel parameter {param_name!r} for "
                f"{base_knl.__class__.__name__} while evaluating {what}"
            )

        param = np.complex128(self.kernel_extra_kwargs[param_name])
        if isinstance(base_knl, HelmholtzKernel):
            return np.complex128(param)
        if isinstance(base_knl, YukawaKernel):
            if not np.isclose(np.imag(param), 0.0):
                raise NotImplementedError(
                    "Yukawa split mode requires real lam; use HelmholtzKernel "
                    "for complex wave numbers"
                )
            return np.complex128(1j * np.real(param))

        raise RuntimeError("split mode supports only Helmholtz/Yukawa kernels")

    def _compute_split_rho_max(self):
        if not self.tree_indep.target_kernels:
            return 0.0

        k_abs = 0.0
        for out_knl in self.tree_indep.target_kernels:
            base_knl = out_knl.get_base_kernel()
            k = self._split_wave_number(base_knl, what="split auto planning")
            k_abs = max(k_abs, float(abs(k)))

        box_source_counts = self.tree.box_source_counts_nonchild.get(self.queue)
        box_levels = self.tree.box_levels.get(self.queue)
        active = np.where(box_source_counts > 0)[0]
        if active.size == 0:
            return 0.0

        min_level = int(np.min(box_levels[active]))
        h_max = float(self.tree.root_extent) * (0.5**min_level)
        return k_abs * h_max

    def _compute_split_rho_components(self):
        if not self.tree_indep.target_kernels:
            return 0.0, 0.0

        k_real_abs = 0.0
        k_imag_abs = 0.0
        for out_knl in self.tree_indep.target_kernels:
            base_knl = out_knl.get_base_kernel()
            k = self._split_wave_number(base_knl, what="split auto planning")
            k_real_abs = max(k_real_abs, abs(float(np.real(k))))
            k_imag_abs = max(k_imag_abs, abs(float(np.imag(k))))

        box_source_counts = self.tree.box_source_counts_nonchild.get(self.queue)
        box_levels = self.tree.box_levels.get(self.queue)
        active = np.where(box_source_counts > 0)[0]
        if active.size == 0:
            return 0.0, 0.0

        min_level = int(np.min(box_levels[active]))
        h_max = float(self.tree.root_extent) * (0.5**min_level)
        rho_real = k_real_abs * h_max
        rho_imag = k_imag_abs * h_max
        return rho_real, rho_imag

    def _choose_auto_helmholtz_split_order(self, auto_cfg):
        order_min = int(auto_cfg.get("order_min", 2))
        order_max = int(auto_cfg.get("order_max", 12))
        if order_max < order_min:
            raise ValueError("order_max must be >= order_min")

        if "rho_thresholds" in auto_cfg or "orders" in auto_cfg:
            thresholds_real = auto_cfg.get("rho_thresholds", (0.5, 1.5, 3.0))
            thresholds_imag = thresholds_real
            if "orders" in auto_cfg:
                orders = auto_cfg["orders"]
            else:
                orders = tuple(range(order_min, order_min + len(thresholds_real) + 1))
        else:
            # Default to geometric ladders for real/imag parts separately.
            rho0_real = float(auto_cfg.get("rho_base_real", 0.25))
            rho0_imag = float(auto_cfg.get("rho_base_imag", 0.5))
            if rho0_real <= 0.0 or rho0_imag <= 0.0:
                raise ValueError("rho_base_real and rho_base_imag must be positive")
            count = max(0, order_max - order_min)
            thresholds_real = tuple(rho0_real * (2.0**j) for j in range(count))
            thresholds_imag = tuple(rho0_imag * (2.0**j) for j in range(count))
            orders = tuple(range(order_min, order_max + 1))

        if "rho_thresholds_real" in auto_cfg:
            thresholds_real = tuple(auto_cfg["rho_thresholds_real"])
        if "rho_thresholds_imag" in auto_cfg:
            thresholds_imag = tuple(auto_cfg["rho_thresholds_imag"])

        if "orders" not in auto_cfg:
            if len(thresholds_real) != len(thresholds_imag):
                raise ValueError(
                    "rho_thresholds_real and rho_thresholds_imag must have the "
                    "same length when orders are not provided"
                )
            orders = tuple(range(order_min, order_min + len(thresholds_real) + 1))

        rho_real, rho_imag = self._compute_split_rho_components()
        self._split_auto_rho_real = float(rho_real)
        self._split_auto_rho_imag = float(rho_imag)
        rho_max = max(rho_real, rho_imag)
        selected = _select_split_order_from_rho_components(
            rho_real,
            rho_imag,
            thresholds_real,
            thresholds_imag,
            orders,
        )

        if "orders" not in auto_cfg:
            selected = max(order_min, min(order_max, selected))

        coverage_max = max(
            float(thresholds_real[-1]) if len(thresholds_real) else 0.0,
            float(thresholds_imag[-1]) if len(thresholds_imag) else 0.0,
        )
        if coverage_max > 0.0 and rho_max > coverage_max:
            logger.warning(
                "rho_max=%.3g exceeds planner threshold coverage (max %.3g); "
                "clamping split order to %d",
                rho_max,
                coverage_max,
                selected,
            )

        logger.info(
            "Auto-selected helmholtz_split_order=%d (rho_real=%.3g, rho_imag=%.3g)",
            selected,
            rho_real,
            rho_imag,
        )
        return selected

    def _helmholtz_split_max_neighbor_distance(self):
        box_source_counts = self.tree.box_source_counts_nonchild.get(self.queue)
        box_levels = self.tree.box_levels.get(self.queue)

        active_levels = box_levels[box_source_counts > 0]
        if active_levels.size == 0:
            return 0.0

        coarsest_active_level = int(np.min(active_levels))
        max_box_extent = float(self.tree.root_extent) * (0.5**coarsest_active_level)
        return float(2.0 * np.sqrt(self.tree.dimensions) * max_box_extent)

    def _helmholtz_split_series_nmax(self, split_order):
        helm_knl, _ = self._helmholtz_split_kernels
        dim = int(helm_knl.dim)
        split_order = int(split_order)
        if split_order < 1:
            raise ValueError("split_order must be >= 1")

        k = self._split_wave_number(helm_knl, what="split terms")
        abs_k = float(np.abs(k))
        r_max = self._helmholtz_split_max_neighbor_distance()
        z_max = abs_k * r_max

        tol = 1.0e-16
        max_n = 96

        if dim == 3:
            start_n = 1
            if z_max == 0.0:
                return start_n

            term_mag = 1.0
            for i in range(1, start_n + 1):
                term_mag = term_mag * z_max / i

            n = start_n
            while n < max_n and term_mag > tol:
                n += 1
                term_mag = term_mag * z_max / n

            return min(max_n, n + 2)

        if dim != 2:
            raise NotImplementedError("split remainder series supports only 2D/3D")

        start_n = 1
        if z_max == 0.0:
            return start_n

        z2_over_4 = 0.25 * z_max * z_max
        coeff_mag = 1.0
        for i in range(1, start_n + 1):
            coeff_mag = coeff_mag * z2_over_4 / (i * i)

        log_r_max = abs(np.log(max(r_max, 1.0e-300)))
        log_k_half = abs(np.log(max(0.5 * abs_k, 1.0e-300)))
        coeff_scale = max(1.0, log_r_max, log_k_half + abs(np.euler_gamma))

        n = start_n
        while n < max_n and coeff_mag * coeff_scale > tol:
            n += 1
            coeff_mag = coeff_mag * z2_over_4 / (n * n)

        return min(max_n, n + 2)

    def _get_helmholtz_split_remainder_kernel(self, *, return_cache_key=False):
        helm_knl, _ = self._helmholtz_split_kernels
        dim = int(helm_knl.dim)
        split_order = int(self.helmholtz_split_order)
        k = self._split_wave_number(helm_knl, what="split remainder")
        series_nmax = self._helmholtz_split_series_nmax(split_order)

        cache_key = (
            dim,
            split_order,
            float(np.real(k)),
            float(np.imag(k)),
            int(series_nmax),
            tuple(self._helmholtz_split_kernel_wrapper_chain),
        )
        if cache_key not in self._helmholtz_split_remainder_kernel_cache:
            base_remainder_kernel = _HelmholtzSplitSeriesRemainderKernel(
                dim,
                np.real(k),
                np.imag(k),
                split_order,
                series_nmax,
            )
            self._helmholtz_split_remainder_kernel_cache[cache_key] = (
                self._apply_helmholtz_split_kernel_wrappers(base_remainder_kernel)
            )

        remainder_kernel = self._helmholtz_split_remainder_kernel_cache[cache_key]
        if return_cache_key:
            return cache_key, remainder_kernel
        return remainder_kernel

    def _get_helmholtz_split_remainder_p2p(self, *, exclude_self):
        from sumpy import P2PFromCSR

        cache_key, remainder_kernel = self._get_helmholtz_split_remainder_kernel(
            return_cache_key=True
        )

        if exclude_self:
            if (
                self._helmholtz_split_remainder_p2p is None
                or self._helmholtz_split_remainder_p2p_cache_key != cache_key
            ):
                self._helmholtz_split_remainder_p2p = P2PFromCSR(
                    [remainder_kernel],
                    True,
                    value_dtypes=[self.dtype],
                )
                self._helmholtz_split_remainder_p2p_cache_key = cache_key
            return self._helmholtz_split_remainder_p2p

        if (
            self._helmholtz_split_remainder_p2p_include_self is None
            or self._helmholtz_split_remainder_p2p_include_self_cache_key != cache_key
        ):
            self._helmholtz_split_remainder_p2p_include_self = P2PFromCSR(
                [remainder_kernel],
                False,
                value_dtypes=[self.dtype],
            )
            self._helmholtz_split_remainder_p2p_include_self_cache_key = cache_key
        return self._helmholtz_split_remainder_p2p_include_self

    def _get_helmholtz_split_term_tables(self, term_key):
        normalized_term_key = _normalize_helmholtz_split_term_key(term_key)
        if normalized_term_key not in self.helmholtz_split_term_tables:
            raise RuntimeError(
                "missing precomputed helmholtz split term tables for "
                f"{_format_helmholtz_split_term_key(normalized_term_key)}"
            )

        return self.helmholtz_split_term_tables[normalized_term_key]

    def _get_or_autobuild_helmholtz_split_term_tables(self, term_key):
        normalized_term_key = _normalize_helmholtz_split_term_key(term_key)
        if normalized_term_key in self.helmholtz_split_term_tables:
            return self.helmholtz_split_term_tables[normalized_term_key]

        if len(self.tree_indep.target_kernels) != 1:
            raise RuntimeError(
                "helmholtz split term table auto-build expects one target kernel"
            )

        out_knl = self.tree_indep.target_kernels[0]
        self._autobuild_helmholtz_split_term_tables(out_knl, [normalized_term_key])
        return self._get_helmholtz_split_term_tables(normalized_term_key)

    def _initialize_helmholtz_split_table_umbrella(self, out_knl):
        kname = repr(out_knl)
        base_tables = self.near_field_table.get(kname, [])
        if not base_tables:
            return

        reference_table = base_tables[0]
        self._helmholtz_split_table_cache_filename = getattr(
            reference_table,
            "_table_cache_filename",
            None,
        )

        if self._helmholtz_split_table_cache_filename is not None:
            self._helmholtz_split_table_cache_root_extent = float(
                getattr(
                    reference_table,
                    "_table_cache_root_extent",
                    self.root_table_source_box_extent,
                )
            )
        else:
            self._helmholtz_split_table_cache_root_extent = None

        self._helmholtz_split_table_build_config = (
            self._reference_helmholtz_split_term_build_config(reference_table)
        )

        source_box_levels = []
        for lev, table in enumerate(base_tables):
            source_box_level = getattr(table, "source_box_level", None)
            if source_box_level is None:
                source_box_level = self.table_starting_level + lev
            source_box_levels.append(int(source_box_level))

        self._helmholtz_split_base_source_box_levels = sorted(set(source_box_levels))

    def _validate_helmholtz_split_term_table_umbrella(self, term_key, term_tables):
        if self._helmholtz_split_table_cache_filename is None:
            return

        expected_filename = self._helmholtz_split_table_cache_filename
        expected_root_extent = self._helmholtz_split_table_cache_root_extent
        term_name = _format_helmholtz_split_term_key(term_key)

        for table in term_tables:
            table_filename = getattr(table, "_table_cache_filename", None)
            if table_filename is None:
                raise RuntimeError(
                    "helmholtz split term tables must come from the same "
                    "NearFieldInteractionTableManager cache as the Laplace "
                    f"table; missing cache metadata for {term_name}"
                )

            if table_filename != expected_filename:
                raise RuntimeError(
                    "helmholtz split term tables must share the same "
                    "NearFieldInteractionTableManager cache file as the "
                    f"Laplace table for {term_name}"
                )

            table_root_extent = getattr(table, "_table_cache_root_extent", None)
            if (
                expected_root_extent is not None
                and table_root_extent is not None
                and abs(float(table_root_extent) - float(expected_root_extent))
                >= 1.0e-15
            ):
                raise RuntimeError(
                    "helmholtz split term table cache root_extent mismatch "
                    f"for {term_name}"
                )

    def _helmholtz_split_required_term_keys(self, dim):
        if self.helmholtz_split_order <= 1:
            return []

        if dim == 3:
            return [
                _normalize_helmholtz_split_term_key(("power", 2 * j - 1))
                for j in range(1, self.helmholtz_split_order)
            ]

        if dim != 2:
            raise NotImplementedError(
                "helmholtz split extra terms are implemented only for 2D/3D"
            )

        return [
            _normalize_helmholtz_split_term_key(("power_log", 2 * n))
            for n in range(1, self.helmholtz_split_order)
        ]

    def _helmholtz_split_term_table_request(self, term_key):
        kind, power = _normalize_helmholtz_split_term_key(term_key)

        if kind == "power":
            if power == 0:
                if self._helmholtz_split_kernel_wrapper_chain:
                    return (
                        self._helmholtz_split_kernel_type_name("Constant"),
                        self._get_helmholtz_split_power_kernel(power),
                    )
                return "Constant", None
            return (
                self._helmholtz_split_kernel_type_name(f"SplitPower{power}"),
                self._get_helmholtz_split_power_kernel(power),
            )

        if kind == "power_log":
            return (
                self._helmholtz_split_kernel_type_name(f"SplitPowerLog{power}"),
                self._get_helmholtz_split_power_log_kernel(power),
            )

        raise RuntimeError(f"unsupported helmholtz split term key kind: {kind}")

    def _default_helmholtz_split_term_build_config(self):
        from volumential.nearfield_potential_table import DuffyBuildConfig

        q_order = int(self.quad_order)
        dim = int(self.tree.dimensions)

        if dim == 2:
            regular_quad_order = max(8, 4 * q_order)
            radial_quad_order = max(21, 10 * q_order)
        elif dim == 3:
            if q_order <= 2:
                regular_quad_order = 6
                radial_quad_order = 21
            else:
                regular_quad_order = 8
                radial_quad_order = 31
        else:
            raise NotImplementedError("split term tables currently support 2D/3D")

        return DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=regular_quad_order,
            radial_quad_order=radial_quad_order,
        )

    def _reference_helmholtz_split_term_build_config(self, base_table):
        build_config = getattr(base_table, "_table_cache_build_config", None)
        if build_config is not None:
            return build_config

        build_config_json = getattr(base_table, "build_config_json", None)
        if isinstance(build_config_json, str):
            try:
                from volumential.nearfield_potential_table import DuffyBuildConfig

                parsed = json.loads(build_config_json)
                if isinstance(parsed, dict):
                    return DuffyBuildConfig(**parsed)
            except Exception:
                pass

        return self._default_helmholtz_split_term_build_config()

    def _autobuild_helmholtz_split_source_box_levels(
        self, base_tables, missing_term_keys
    ):
        source_box_levels = []
        for lev, table in enumerate(base_tables):
            source_box_level = getattr(table, "source_box_level", None)
            if source_box_level is None:
                source_box_level = self.table_starting_level + lev
            source_box_levels.append(int(source_box_level))

        if not source_box_levels:
            return source_box_levels

        return sorted(set(source_box_levels))

    def _split_term_autobuild_direction_kwargs(self, out_knl):
        base_knl = out_knl.get_base_kernel()
        wrapper_chain = self._extract_helmholtz_split_kernel_wrapper_chain(
            out_knl,
            base_knl,
        )
        directional_names = [
            str(value) for kind, value in wrapper_chain if kind == "directional_source"
        ]
        if not directional_names:
            return {}

        unique_names = tuple(dict.fromkeys(directional_names))
        if len(unique_names) != 1:
            raise NotImplementedError(
                "split term table auto-build supports at most one directional "
                "source vector name"
            )

        dir_vec_name = unique_names[0]
        if dir_vec_name not in self.source_extra_kwargs:
            raise ValueError(
                "missing directional source parameter "
                f"{dir_vec_name!r} for split term table auto-build"
            )

        direction = _extract_symmetry_source_direction(
            out_knl,
            self.source_extra_kwargs,
            self.queue,
        )
        if direction is None:
            raise ValueError(
                "split term table auto-build requires directional source values "
                "that are present and constant across sources"
            )

        direction = np.asarray(direction, dtype=np.float64).ravel()
        if direction.size != int(self.tree.dimensions):
            raise ValueError(
                "directional source vector for split term table auto-build has "
                f"length {direction.size}; expected {self.tree.dimensions}"
            )

        return {dir_vec_name: direction}

    def _autobuild_helmholtz_split_term_tables(self, out_knl, missing_term_keys):
        if not missing_term_keys:
            return

        kname = repr(out_knl)
        base_tables = self.near_field_table.get(kname, [])
        if not base_tables:
            return

        reference_table = base_tables[0]
        cache_filename = getattr(reference_table, "_table_cache_filename", None)
        if cache_filename is None:
            return

        cache_root_extent = float(
            getattr(
                reference_table,
                "_table_cache_root_extent",
                self.root_table_source_box_extent,
            )
        )
        build_config = self._reference_helmholtz_split_term_build_config(
            reference_table
        )
        source_box_levels = self._autobuild_helmholtz_split_source_box_levels(
            base_tables,
            missing_term_keys,
        )
        directional_build_kwargs = self._split_term_autobuild_direction_kwargs(out_knl)

        from volumential.table_manager import NearFieldInteractionTableManager

        try:
            with NearFieldInteractionTableManager(
                cache_filename,
                root_extent=cache_root_extent,
            ) as table_manager:
                for term_key in missing_term_keys:
                    normalized_term_key = _normalize_helmholtz_split_term_key(term_key)
                    if normalized_term_key in self.helmholtz_split_term_tables:
                        continue

                    kernel_type, sumpy_knl = self._helmholtz_split_term_table_request(
                        normalized_term_key
                    )

                    tables = []
                    for source_box_level in source_box_levels:
                        get_table_kwargs = {
                            "source_box_level": int(source_box_level),
                            "force_recompute": False,
                            "queue": self.queue,
                            "build_config": build_config,
                        }
                        get_table_kwargs.update(directional_build_kwargs)
                        if sumpy_knl is not None:
                            get_table_kwargs["sumpy_knl"] = sumpy_knl

                        table, _ = table_manager.get_table(
                            self.tree.dimensions,
                            kernel_type,
                            self.quad_order,
                            **get_table_kwargs,
                        )
                        tables.append(table)

                    self.helmholtz_split_term_tables[normalized_term_key] = tables
        except Exception as exc:
            logger.warning(
                "helmholtz split term table auto-build failed for %s: %s",
                [
                    _format_helmholtz_split_term_key(term_key)
                    for term_key in missing_term_keys
                ],
                exc,
            )

    def _helmholtz_split_term_scaling_code(self, term_key, *, derivative_order=None):
        kind, power = _normalize_helmholtz_split_term_key(term_key)
        if kind not in {"power", "power_log"}:
            raise RuntimeError(
                f"unsupported split term kind for single-table scaling: {kind}"
            )

        if derivative_order is None:
            derivative_order = self._helmholtz_split_wrapper_derivative_order

        exponent = int(self.tree.dimensions) + int(power) - int(derivative_order)
        if exponent == 0:
            return "1.0"

        if exponent > 0:
            box_factor = " * ".join(["BOX_extent"] * exponent)
            table_factor = " * ".join(["table_root_extent"] * exponent)
            return f"({box_factor}) / ({table_factor})"

        exponent = -exponent
        box_factor = " * ".join(["BOX_extent"] * exponent)
        table_factor = " * ".join(["table_root_extent"] * exponent)
        return f"({table_factor}) / ({box_factor})"

    def _helmholtz_split_list1_scaling_code(self, dim, *, derivative_order=None):
        if derivative_order is None:
            derivative_order = self._helmholtz_split_wrapper_derivative_order
        derivative_order = int(derivative_order)
        if int(dim) == 2 and derivative_order == 0:
            exponent = 2
        elif int(dim) == 2:
            exponent = 2 - derivative_order
        else:
            exponent = int(dim) - 1 - derivative_order

        if exponent == 0:
            return "1.0"
        if exponent > 0:
            box_factor = " * ".join(["BOX_extent"] * exponent)
            table_factor = " * ".join(["table_root_extent"] * exponent)
            return f"({box_factor}) / ({table_factor})"

        exponent = -exponent
        box_factor = " * ".join(["BOX_extent"] * exponent)
        table_factor = " * ".join(["table_root_extent"] * exponent)
        return f"({table_factor}) / ({box_factor})"

    def _helmholtz_split_list1_displacement_code(self, dim, *, derivative_order=None):
        if derivative_order is None:
            derivative_order = self._helmholtz_split_wrapper_derivative_order
        if int(dim) == 2 and int(derivative_order) == 0:
            return (
                f"-0.5 / {np.pi!r} * scaling * "
                "log(BOX_extent / table_root_extent) * "
                "mode_nmlz[table_lev, sid]"
            )
        return "0.0"

    def _eval_direct_helmholtz_split_term_table(
        self,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        src_weights,
        term_kernel,
        term_tables,
        *,
        term_key,
    ):
        normalized_term_key = _normalize_helmholtz_split_term_key(term_key)

        if len(term_tables) > 1:
            list1_kwargs = {
                "infer_kernel_scaling": False,
            }
        else:
            list1_kwargs = {
                "infer_kernel_scaling": False,
                "kernel_scaling_code": self._helmholtz_split_term_scaling_code(
                    normalized_term_key
                ),
                "kernel_displacement_code": "0.0",
            }

        def _eval_term_table(term_src_weights):
            out_pot = self.output_zeros()[0]
            if not getattr(term_kernel, "is_complex_valued", False):
                out_dtype = np.empty((), dtype=self.dtype).real.dtype
                if out_pot.dtype != out_dtype:
                    if isinstance(out_pot, cl.array.Array):
                        out_pot = cl.array.zeros(
                            self.queue,
                            out_pot.shape,
                            dtype=out_dtype,
                        )
                    else:
                        out_pot = np.zeros_like(np.asarray(out_pot), dtype=out_dtype)

            out_pot, _ = self.eval_direct_single_out_kernel(
                out_pot,
                term_kernel,
                target_boxes,
                neighbor_source_boxes_starts,
                neighbor_source_boxes_lists,
                term_src_weights,
                near_field_tables=term_tables,
                list1_extra_kwargs=list1_kwargs,
            )
            return out_pot

        src_dtype = getattr(src_weights, "dtype", None)
        if src_dtype is None:
            is_complex_strength = np.iscomplexobj(src_weights)
        else:
            is_complex_strength = np.issubdtype(np.dtype(src_dtype), np.complexfloating)

        if not is_complex_strength:
            return obj_array_1d([_eval_term_table(src_weights)])

        if isinstance(src_weights, cl.array.Array):
            src_weights_real = src_weights.real
            src_weights_imag = src_weights.imag
        else:
            src_weights_host = np.asarray(src_weights)
            src_weights_real = np.ascontiguousarray(np.real(src_weights_host))
            src_weights_imag = np.ascontiguousarray(np.imag(src_weights_host))

        out_real = _eval_term_table(src_weights_real)
        out_imag = _eval_term_table(src_weights_imag)
        imag_scale = np.array(1j, dtype=self.dtype)
        return obj_array_1d([out_real + out_imag * imag_scale])

    def _get_helmholtz_split_log_alpha_per_source(
        self,
        table_root_extent,
        *,
        box_source_starts=None,
        box_source_counts=None,
    ):
        table_root_extent = float(table_root_extent)
        if table_root_extent <= 0.0:
            raise ValueError("table_root_extent must be positive")

        if box_source_starts is None and box_source_counts is None:
            cache_key = table_root_extent
            if cache_key in self._helmholtz_split_log_alpha_per_source_cache:
                return self._helmholtz_split_log_alpha_per_source_cache[cache_key]

            box_source_starts_h = self.tree.box_source_starts.get(self.queue)
            box_source_counts_h = self.tree.box_source_counts_nonchild.get(self.queue)
            use_cache = True
        else:
            if box_source_starts is None or box_source_counts is None:
                raise ValueError(
                    "box_source_starts and box_source_counts must be provided together"
                )

            if isinstance(box_source_starts, cl.array.Array):
                box_source_starts_h = box_source_starts.get(self.queue)
            else:
                box_source_starts_h = np.asarray(box_source_starts)

            if isinstance(box_source_counts, cl.array.Array):
                box_source_counts_h = box_source_counts.get(self.queue)
            else:
                box_source_counts_h = np.asarray(box_source_counts)

            use_cache = False

        box_levels = self.tree.box_levels.get(self.queue)

        if len(box_source_starts_h) != len(box_levels):
            raise ValueError("box_source_starts length must match number of boxes")
        if len(box_source_counts_h) != len(box_levels):
            raise ValueError("box_source_counts length must match number of boxes")

        if box_source_starts_h.size == 0:
            nsources = 0
        else:
            nsources = int(np.max(box_source_starts_h + box_source_counts_h))
        beta_host = np.zeros(nsources, dtype=np.float64)
        root_extent = float(self.tree.root_extent)

        for ibox in range(len(box_source_starts_h)):
            count = int(box_source_counts_h[ibox])
            if count <= 0:
                continue

            start = int(box_source_starts_h[ibox])
            stop = start + count
            box_extent = root_extent * (0.5 ** int(box_levels[ibox]))
            beta_value = np.log(box_extent / table_root_extent)
            beta_host[start:stop] = beta_value

        beta_host = np.ascontiguousarray(beta_host)
        if use_cache:
            self._helmholtz_split_log_alpha_per_source_cache[cache_key] = beta_host
        return beta_host

    def _fold_helmholtz_split_log_alpha_into_strength(
        self,
        src_weights,
        table_root_extent,
        *,
        box_source_starts=None,
        box_source_counts=None,
    ):
        beta_host = self._get_helmholtz_split_log_alpha_per_source(
            table_root_extent,
            box_source_starts=box_source_starts,
            box_source_counts=box_source_counts,
        )

        if isinstance(src_weights, cl.array.Array):
            dtype = np.dtype(src_weights.dtype)
            use_cache = box_source_starts is None and box_source_counts is None
            if use_cache:
                dev_key = (float(table_root_extent), dtype.str)
                beta_dev = self._helmholtz_split_log_alpha_per_source_dev_cache.get(
                    dev_key
                )
                if beta_dev is None:
                    beta_dev = cl.array.to_device(
                        self.queue,
                        beta_host.astype(dtype, copy=False),
                    )
                    self._helmholtz_split_log_alpha_per_source_dev_cache[dev_key] = (
                        beta_dev
                    )
            else:
                beta_dev = cl.array.to_device(
                    self.queue,
                    beta_host.astype(dtype, copy=False),
                )

            return src_weights * beta_dev

        src_weights_arr = np.asarray(src_weights)
        beta_arr = beta_host.astype(src_weights_arr.dtype, copy=False)
        return src_weights_arr * beta_arr

    def _get_helmholtz_split_power_kernel(self, power):
        power = int(power)
        if power < 0:
            raise ValueError("power must be non-negative")

        if power == 0:
            base_kernel = self._helmholtz_split_constant_kernel
            return self._apply_helmholtz_split_kernel_wrappers(base_kernel)

        if power not in self._helmholtz_split_power_kernels:
            self._helmholtz_split_power_kernels[power] = _RadialPowerKernel(
                self.tree.dimensions,
                power,
            )

        return self._apply_helmholtz_split_kernel_wrappers(
            self._helmholtz_split_power_kernels[power]
        )

    def _get_helmholtz_split_power_log_kernel(self, power):
        power = int(power)
        if power <= 0:
            raise ValueError("power must be positive for r**power * log(r)")

        if power not in self._helmholtz_split_power_log_kernels:
            self._helmholtz_split_power_log_kernels[power] = _RadialPowerLogKernel(
                self.tree.dimensions,
                power,
            )

        return self._apply_helmholtz_split_kernel_wrappers(
            self._helmholtz_split_power_log_kernels[power]
        )

    def _helmholtz_split_extra_terms(self):
        """Return pretabulated non-smooth split terms and coefficients.

        The returned list contains tuples ``(term_key, kernel, coeff)`` for the
        split-term table evaluation.

        - 2D: :math:`r^{2n}\\log r` for :math:`n=1,\\dots,p-1`.
        - 3D: :math:`r^{2j-1}` for :math:`j=1,\\dots,p-1`.
        """

        if self.helmholtz_split_order <= 1:
            return []

        helm_knl, _ = self._helmholtz_split_kernels
        dim = helm_knl.dim
        k = self._split_wave_number(helm_knl, what="split terms")

        terms = []

        if dim == 2:
            from math import factorial

            for n in range(1, self.helmholtz_split_order):
                power = 2 * n
                series_scale = (
                    ((-1) ** n) * (k * k / 4.0) ** n / (factorial(n) * factorial(n))
                )

                common = np.complex128(series_scale)

                coeff_log = np.complex128(-common / (2.0 * np.pi))

                terms.append(
                    (
                        _normalize_helmholtz_split_term_key(("power_log", power)),
                        self._get_helmholtz_split_power_log_kernel(power),
                        coeff_log,
                    )
                )

            return terms

        if dim != 3:
            raise NotImplementedError(
                "helmholtz split extra terms are implemented only for 2D/3D"
            )

        from math import factorial

        for j in range(1, self.helmholtz_split_order):
            n = 2 * j
            coeff = (1j * k) ** n / (4.0 * np.pi * factorial(n))
            power = n - 1
            kernel = self._get_helmholtz_split_power_kernel(power)
            terms.append(
                (
                    _normalize_helmholtz_split_term_key(("power", power)),
                    kernel,
                    np.complex128(coeff),
                )
            )

        return terms

    def _get_helmholtz_split_smooth_interp_data(
        self, smooth_quad_order, *, allow_node_overlap=False
    ):
        smooth_quad_order_int = int(smooth_quad_order)
        cache_key = (smooth_quad_order_int, bool(allow_node_overlap))
        if cache_key in self._helmholtz_split_smooth_interp_cache:
            return self._helmholtz_split_smooth_interp_cache[cache_key]

        dim = self.tree.dimensions
        nodes_q, _ = _gauss_legendre_nodes_and_weights(self.quad_order)
        nodes_smooth, weights_smooth = _gauss_legendre_nodes_and_weights(
            smooth_quad_order_int
        )

        overlap = np.any(
            np.abs(nodes_smooth[:, np.newaxis] - nodes_q[np.newaxis, :]) < 1.0e-15
        )
        if overlap and not allow_node_overlap:
            raise ValueError(
                "helmholtz split smooth quadrature order shares nodes with source "
                "quadrature; choose a non-overlapping order to avoid singular "
                "Helmholtz/Laplace evaluations"
            )

        interp_mat = _barycentric_interp_matrix(nodes_q, nodes_smooth)

        node_grids = np.meshgrid(*([nodes_smooth] * dim), indexing="ij")
        ref_nodes = np.asarray(
            [grid.reshape(-1) for grid in node_grids], dtype=np.float64
        )

        weight_grids = np.meshgrid(*([weights_smooth] * dim), indexing="ij")
        ref_weights = np.ones_like(weight_grids[0], dtype=np.float64)
        for weight_grid in weight_grids:
            ref_weights = ref_weights * weight_grid
        ref_weights = ref_weights.reshape(-1)

        interp_data = {
            "interp_mat": interp_mat,
            "ref_nodes": ref_nodes,
            "ref_weights": ref_weights,
            "n_smooth_points": smooth_quad_order_int**dim,
        }
        self._helmholtz_split_smooth_interp_cache[cache_key] = interp_data
        return interp_data

    def _interpolate_box_values_to_smooth_quad(
        self, box_values, interp_mat, smooth_quad_order
    ):
        dim = self.tree.dimensions
        q = self.quad_order

        box_values = np.asarray(box_values).reshape((q,) * dim)

        if dim == 1:
            smooth_values = interp_mat @ box_values
        elif dim == 2:
            smooth_values = interp_mat @ box_values @ interp_mat.T
        elif dim == 3:
            smooth_values = np.einsum(
                "ai,bj,ck,ijk->abc",
                interp_mat,
                interp_mat,
                interp_mat,
                box_values,
                optimize=True,
            )
        else:
            raise NotImplementedError("helmholtz split smooth correction supports 1-3D")

        return np.asarray(smooth_values).reshape(smooth_quad_order**dim)

    def _build_helmholtz_split_smooth_correction_sources(
        self, src_func, smooth_quad_order, *, allow_node_overlap=False
    ):
        dim = self.tree.dimensions
        q = self.quad_order
        n_q_points = q**dim

        interp_data = self._get_helmholtz_split_smooth_interp_data(
            smooth_quad_order,
            allow_node_overlap=allow_node_overlap,
        )
        interp_mat = interp_data["interp_mat"]
        ref_nodes = interp_data["ref_nodes"]
        ref_weights = interp_data["ref_weights"]
        n_smooth_points = interp_data["n_smooth_points"]

        if isinstance(src_func, cl.array.Array):
            src_func_host = src_func.get(self.queue)
        else:
            src_func_host = np.asarray(src_func)

        box_source_starts = self.tree.box_source_starts.get(self.queue)
        box_source_counts = self.tree.box_source_counts_nonchild.get(self.queue)
        box_centers = self.tree.box_centers.get(self.queue)
        box_levels = self.tree.box_levels.get(self.queue)

        nboxes = box_source_starts.size
        smooth_box_source_starts = np.zeros(nboxes, dtype=np.int32)
        smooth_box_source_counts = np.zeros(nboxes, dtype=np.int32)

        cursor = 0
        for ibox in range(nboxes):
            smooth_box_source_starts[ibox] = cursor
            count = int(box_source_counts[ibox])
            if count == 0:
                continue
            if count != n_q_points:
                raise ValueError(
                    "helmholtz split smooth correction requires exactly "
                    f"{n_q_points} sources per active box; found {count} "
                    f"in box {ibox}"
                )
            smooth_box_source_counts[ibox] = n_smooth_points
            cursor += n_smooth_points

        smooth_sources_host = np.empty((dim, cursor), dtype=np.float64)
        smooth_strength_host = np.empty(cursor, dtype=src_func_host.dtype)

        root_extent = float(self.tree.root_extent)

        for ibox in range(nboxes):
            count = int(box_source_counts[ibox])
            if count == 0:
                continue

            src_start = int(box_source_starts[ibox])
            src_stop = src_start + count
            box_values = src_func_host[src_start:src_stop]
            smooth_values = self._interpolate_box_values_to_smooth_quad(
                box_values,
                interp_mat,
                smooth_quad_order,
            )

            extent = root_extent * (0.5 ** int(box_levels[ibox]))
            lower_corner = box_centers[:, ibox] - 0.5 * extent

            smooth_start = int(smooth_box_source_starts[ibox])
            smooth_stop = smooth_start + n_smooth_points
            smooth_sources_host[:, smooth_start:smooth_stop] = (
                lower_corner[:, np.newaxis] + extent * ref_nodes
            )
            smooth_strength_host[smooth_start:smooth_stop] = smooth_values * (
                (extent**dim) * ref_weights
            )

        smooth_sources = obj_array_1d(
            [
                cl.array.to_device(
                    self.queue,
                    np.ascontiguousarray(smooth_sources_host[axis]),
                )
                for axis in range(dim)
            ]
        )

        return {
            "source_kwargs": {
                "box_source_starts": cl.array.to_device(
                    self.queue,
                    smooth_box_source_starts,
                ),
                "box_source_counts_nonchild": cl.array.to_device(
                    self.queue,
                    smooth_box_source_counts,
                ),
                "sources": smooth_sources,
            },
            "strength": cl.array.to_device(
                self.queue,
                np.ascontiguousarray(smooth_strength_host),
            ),
            "max_nsources_in_one_box": int(n_smooth_points),
        }

    def _helmholtz_split_self_diagonal_limit(self):
        if not self.helmholtz_split:
            return None

        if self._helmholtz_split_kernel_wrapper_chain:
            return np.complex128(0.0)

        helm_knl, _ = self._helmholtz_split_kernels
        try:
            k = self._split_wave_number(helm_knl, what="self diagonal limit")
        except RuntimeError:
            return None
        if helm_knl.dim == 3:
            return np.complex128(1j * k / (4.0 * np.pi))

        if helm_knl.dim == 2:
            if np.abs(k) == 0.0:
                return np.complex128(0.0)
            return np.complex128(
                0.25j
                - (1.0 / (2.0 * np.pi))
                * (np.log(0.5 * k) + np.complex128(np.euler_gamma))
            )

        return None

    def _helmholtz_split_self_diagonal_term(self, src_weights, shared_kwargs, like):
        limit = self._helmholtz_split_self_diagonal_limit()
        if limit is None:
            return None

        target_to_source = shared_kwargs.get("target_to_source")

        if target_to_source is not None:
            if isinstance(src_weights, cl.array.Array):
                if not isinstance(target_to_source, cl.array.Array):
                    target_to_source = cl.array.to_device(
                        self.queue, np.asarray(target_to_source)
                    )
                diag_strength = cl.array.take(src_weights, target_to_source)
            else:
                if isinstance(target_to_source, cl.array.Array):
                    target_to_source = target_to_source.get(self.queue)
                diag_strength = np.asarray(src_weights)[np.asarray(target_to_source)]
        elif (
            hasattr(src_weights, "shape")
            and hasattr(like, "shape")
            and tuple(src_weights.shape) == tuple(like.shape)
        ):
            diag_strength = src_weights
        else:
            if not self.tree.sources_are_targets:
                return None
            if self.tree.ntargets != self.tree.nsources:
                return None
            diag_strength = src_weights

        if isinstance(diag_strength, cl.array.Array):
            scalar = np.array(limit, dtype=diag_strength.dtype)
            return diag_strength * scalar

        if isinstance(like, cl.array.Array):
            diag_strength = cl.array.to_device(self.queue, np.asarray(diag_strength))
            return diag_strength * np.array(limit, dtype=like.dtype)

        return np.asarray(diag_strength) * np.array(limit, dtype=np.asarray(like).dtype)

    # }}} End direct evaluation of near field interactions

    # {{{ downward pass of fmm

    def multipole_to_local(
        self,
        level_start_target_box_nrs,
        target_boxes,
        src_box_starts,
        src_box_lists,
        mpole_exps,
    ):
        local_exps = SumpyExpansionWrangler.multipole_to_local(
            self,
            self._actx,
            level_start_target_box_nrs,
            target_boxes,
            src_box_starts,
            src_box_lists,
            mpole_exps,
        )
        return local_exps, SumpyTimingFuture(self.queue, [])

    def eval_multipoles(
        self, target_boxes_by_source_level, source_boxes_by_level, mpole_exps
    ):
        pot = SumpyExpansionWrangler.eval_multipoles(
            self,
            self._actx,
            target_boxes_by_source_level,
            source_boxes_by_level,
            mpole_exps,
        )
        return pot, SumpyTimingFuture(self.queue, [])

    def form_locals(
        self,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        starts,
        lists,
        src_weights,
    ):
        local_exps = SumpyExpansionWrangler.form_locals(
            self,
            self._actx,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            src_weights,
        )
        return local_exps, SumpyTimingFuture(self.queue, [])

    def refine_locals(
        self,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        local_exps = SumpyExpansionWrangler.refine_locals(
            self,
            self._actx,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        )
        return local_exps, SumpyTimingFuture(self.queue, [])

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        pot = SumpyExpansionWrangler.eval_locals(
            self, self._actx, level_start_target_box_nrs, target_boxes, local_exps
        )
        return pot, SumpyTimingFuture(self.queue, [])

    # }}} End downward pass of fmm

    # {{{ direct evaluation of p2p (discrete) interactions

    def eval_direct_p2p(
        self, target_boxes, source_box_starts, source_box_lists, src_weights
    ):
        pot = self.output_zeros(self._actx)

        kwargs = dict(self.extra_kwargs)
        kwargs.update(self.self_extra_kwargs)
        kwargs.update(self.box_source_list_kwargs())
        kwargs.update(self.box_target_list_kwargs())

        if "target_to_source" in kwargs and isinstance(
            kwargs["target_to_source"], np.ndarray
        ):
            kwargs["target_to_source"] = cl.array.to_device(
                self.queue, kwargs["target_to_source"]
            )

        pot_res = self.tree_indep.p2p()(
            self._actx,
            target_boxes=target_boxes,
            source_box_starts=source_box_starts,
            source_box_lists=source_box_lists,
            strength=src_weights,
            result=pot,
            max_nsources_in_one_box=self.max_nsources_in_one_box,
            max_ntargets_in_one_box=self.max_ntargets_in_one_box,
            **kwargs,
        )

        for pot_i, pot_res_i in zip(pot, pot_res, strict=True):
            assert pot_i is pot_res_i

        return pot, SumpyTimingFuture(self.queue, [])

    # }}} End direct evaluation of p2p interactions


# }}} End sumpy backend

# {{{ fmmlib backend (for laplace, helmholtz)


class FPNDFMMLibTreeIndependentDataForWrangler(
    TreeIndependentDataForWranglerInterface,
):
    """Objects of this type serve as a place to keep the code needed
    for ExpansionWrangler if it is using fmmlib to perform multipole
    expansion and manipulations.

    The interface is augmented with unnecessary arguments acting as
    placeholders, such that it can be a drop-in replacement of sumpy
    backend.
    """

    def __init__(
        self,
        cl_context,
        multipole_expansion_factory,
        local_expansion_factory,
        target_kernels,
        exclude_self=True,
        *args,
        **kwargs,
    ):
        self.cl_context = cl_context
        self.multipole_expansion_factory = multipole_expansion_factory
        self.local_expansion_factory = local_expansion_factory

        self.target_kernels = target_kernels
        self.exclude_self = True

    def get_wrangler(
        self,
        queue,
        tree,
        dtype,
        fmm_level_to_order,
        source_extra_kwargs=None,
        kernel_extra_kwargs=None,
        *args,
        **kwargs,
    ):
        if source_extra_kwargs is None:
            source_extra_kwargs = {}

        return FPNDFMMLibExpansionWrangler(
            self,
            queue,
            tree,
            dtype,
            fmm_level_to_order,
            source_extra_kwargs,
            kernel_extra_kwargs,
            *args,
            **kwargs,
        )


class FPNDFMMLibExpansionWrangler(ExpansionWranglerInterface, FMMLibExpansionWrangler):
    """This expansion wrangler uses "fpnd" strategy. That is, Far field is
    computed via Particle approximation and Near field is computed Directly.
    The FMM is performed using FMMLib backend.

    .. attribute:: source_extra_kwargs

        Keyword arguments to be passed to interactions that involve
        the source field.

    .. attribute:: kernel_extra_kwargs

        Keyword arguments to be passed to interactions that involve
        expansions, but not the source field.

    Much of this class is borrowed from pytential.qbx.fmmlib.
    """

    # {{{ constructor

    def __init__(
        self,
        tree_indep,
        queue,
        tree,
        near_field_table,
        dtype,
        fmm_level_to_order,
        quad_order,
        potential_kind=1,
        source_extra_kwargs=None,
        kernel_extra_kwargs=None,
        self_extra_kwargs=None,
        list1_extra_kwargs=None,
        *args,
        **kwargs,
    ):
        self.tree_indep = tree_indep
        self.queue = queue

        tree = tree.get(queue)
        self.tree = tree

        self.dtype = dtype
        self.quad_order = quad_order
        self.potential_kind = potential_kind

        # {{{ digest target_kernels

        ifgrad = False
        outputs = []
        source_deriv_names = []
        k_names = []

        for out_knl in self.tree_indep.target_kernels:
            if self.is_supported_helmknl(out_knl):
                outputs.append(())
                no_target_deriv_knl = out_knl

            elif isinstance(
                out_knl, AxisTargetDerivative
            ) and self.is_supported_helmknl(out_knl.inner_kernel):
                outputs.append((out_knl.axis,))
                ifgrad = True
                no_target_deriv_knl = out_knl.inner_kernel

            else:
                raise ValueError(
                    "only the 2/3D Laplace and Helmholtz kernel "
                    "and their derivatives are supported"
                )

            source_deriv_names.append(
                no_target_deriv_knl.dir_vec_name
                if isinstance(no_target_deriv_knl, DirectionalSourceDerivative)
                else None
            )

            base_knl = out_knl.get_base_kernel()
            k_names.append(
                base_knl.helmholtz_k_name
                if isinstance(base_knl, HelmholtzKernel)
                else None
            )

        self.outputs = outputs

        from pytools import is_single_valued

        if not is_single_valued(source_deriv_names):
            raise ValueError(
                "not all kernels passed are the same in "
                "whether they represent a source derivative"
            )

        source_deriv_name = source_deriv_names[0]

        if not is_single_valued(k_names):
            raise ValueError("not all kernels passed have the same Helmholtz parameter")

        k_name = k_names[0]

        if k_name is None:
            helmholtz_k = 0
        else:
            helmholtz_k = kernel_extra_kwargs[k_name]

        # }}}

        # {{{ table setup
        # TODO put this part into the inteferce class

        self.near_field_table = {}
        # list of tables for a single out kernel
        if isinstance(near_field_table, list):
            assert len(self.tree_indep.target_kernels) == 1
            self.near_field_table[self.tree_indep.target_kernels[0].__repr__()] = (
                near_field_table
            )
            self.n_tables = len(near_field_table)

        # single table
        elif isinstance(near_field_table, NearFieldInteractionTable):
            assert len(self.tree_indep.target_kernels) == 1
            self.near_field_table[self.tree_indep.target_kernels[0].__repr__()] = [
                near_field_table
            ]
            self.n_tables = 1

        # dictionary of lists of tables
        elif isinstance(near_field_table, dict):
            self.n_tables = {}
            for out_knl in self.tree_indep.target_kernels:
                if repr(out_knl) not in near_field_table:
                    raise RuntimeError(
                        "Missing nearfield table for %s." % repr(out_knl)
                    )
                if isinstance(
                    near_field_table[repr(out_knl)], NearFieldInteractionTable
                ):
                    near_field_table[repr(out_knl)] = [near_field_table[repr(out_knl)]]
                else:
                    assert isinstance(near_field_table[repr(out_knl)], list)

                self.n_tables[repr(out_knl)] = len(near_field_table[repr(out_knl)])

            self.near_field_table = near_field_table
        else:
            raise RuntimeError("Table type unrecognized.")

        # TODO: make all parameters table-specific (allow using inhomogeneous tables)
        kname = repr(self.tree_indep.target_kernels[0])
        self.root_table_source_box_extent = self.near_field_table[kname][
            0
        ].source_box_extent
        table_starting_level = int(
            np.round(
                np.log(self.tree.root_extent / self.root_table_source_box_extent)
                / np.log(2)
            )
        )
        self.table_starting_level = table_starting_level
        for kid in range(len(self.tree_indep.target_kernels)):
            kname = self.tree_indep.target_kernels[kid].__repr__()
            for lev, table in zip(
                range(len(self.near_field_table[kname])), self.near_field_table[kname]
            ):
                assert table.quad_order == self.quad_order

                if not table.is_built:
                    raise RuntimeError(
                        "Near field interaction table needs to be built "
                        "prior to being used"
                    )

                table_root_extent = table.source_box_extent * 2**lev
                assert (
                    abs(self.root_table_source_box_extent - table_root_extent) < 1e-15
                )

                # If the kernel cannot be scaled,
                # - tree_root_extent must be integral times of table_root_extent
                # - n_tables must be sufficient
                if not isinstance(self.n_tables, dict) and self.n_tables > 1:
                    if (
                        not abs(
                            int(self.tree.root_extent / table_root_extent)
                            * table_root_extent
                            - self.tree.root_extent
                        )
                        < 1e-15
                    ):
                        raise RuntimeError(
                            "Incompatible list of tables: the "
                            "source_box_extent of the root table must "
                            "divide the bounding box's extent by an integer."
                        )

            if not isinstance(self.n_tables, dict) and self.n_tables > 1:
                # this checks that the boxes at the highest level are covered
                if (
                    not tree.nlevels
                    <= len(self.near_field_table[kname]) + table_starting_level
                ):
                    raise RuntimeError(
                        "Insufficient list of tables: the "
                        "finest level mesh cells at level "
                        + str(tree.nlevels)
                        + " are not covered."
                    )

                # the check that the boxes at the coarsest level are covered is
                # deferred until trav.target_boxes is passed when invoking
                # eval_direct

        if source_extra_kwargs is None:
            source_extra_kwargs = {}

        if kernel_extra_kwargs is None:
            kernel_extra_kwargs = {}

        if self_extra_kwargs is None:
            self_extra_kwargs = {}

        if list1_extra_kwargs is None:
            list1_extra_kwargs = {}

        self.list1_extra_kwargs = list1_extra_kwargs
        self._table_layout_validation_cache = set()
        self._nearfield_device_payload_cache = OrderedDict()
        self._nearfield_device_payload_cache_max = 16

        # }}} End table setup

        if not callable(fmm_level_to_order):
            raise TypeError("fmm_level_to_order not passed")

        dipole_vec = None
        if source_deriv_name is not None:
            dipole_vec = np.array(
                [
                    d_i.get(queue=queue)
                    for d_i in source_extra_kwargs[source_deriv_name]
                ],
                order="F",
            )

        def inner_fmm_level_to_nterms(tree, level):
            if helmholtz_k == 0:
                return fmm_level_to_order(
                    LaplaceKernel(tree.dimensions), frozenset(), tree, level
                )
            else:
                return fmm_level_to_order(
                    HelmholtzKernel(tree.dimensions),
                    frozenset([("k", helmholtz_k)]),
                    tree,
                    level,
                )

        rotation_data = None
        if "traversal" in kwargs:
            # add rotation data if traversal is passed as a keyword argument
            from boxtree.pyfmmlib_integration import FMMLibRotationData

            rotation_data = FMMLibRotationData(self.queue, kwargs["traversal"])
        else:
            logger.warning(
                "Rotation data is not utilized since traversal is "
                "not known to FPNDFMMLibExpansionWrangler."
            )

        FMMLibExpansionWrangler.__init__(
            self,
            tree,
            helmholtz_k=helmholtz_k,
            dipole_vec=dipole_vec,
            dipoles_already_reordered=True,
            fmm_level_to_nterms=inner_fmm_level_to_nterms,
            rotation_data=rotation_data,
            ifgrad=ifgrad,
        )

    # }}} End constructor

    # {{{ scale factor for fmmlib

    def get_scale_factor(self):
        if self.eqn_letter == "l" and self.dim == 2:
            scale_factor = -1 / (2 * np.pi)
        elif self.eqn_letter == "h" and self.dim == 2:
            scale_factor = 1
        elif self.eqn_letter in ["l", "h"] and self.dim == 3:
            scale_factor = 1 / (4 * np.pi)
        else:
            raise NotImplementedError(
                "scale factor for pyfmmlib %s for %d dimensions"
                % (self.eqn_letter, self.dim)
            )

        return scale_factor

    # }}} End scale factor for fmmlib

    # {{{ data vector utilities

    def multipole_expansion_zeros(self):
        return FMMLibExpansionWrangler.multipole_expansion_zeros(self)

    def local_expansion_zeros(self):
        return FMMLibExpansionWrangler.local_expansion_zeros(self)

    def output_zeros(self):
        return FMMLibExpansionWrangler.output_zeros(self)

    def reorder_sources(self, source_array):
        return FMMLibExpansionWrangler.reorder_sources(self, source_array)

    def reorder_targets(self, target_array):
        if not hasattr(self.tree, "user_target_ids"):
            self.tree.user_target_ids = inverse_id_map(
                self.queue, self.tree.sorted_target_ids
            )
        return target_array[self.tree.user_target_ids]

    def reorder_potentials(self, potentials):
        return FMMLibExpansionWrangler.reorder_potentials(self, potentials)

    def finalize_potentials(self, potentials):
        # return potentials
        return FMMLibExpansionWrangler.finalize_potentials(self, potentials)

    # }}} End data vector utilities

    # {{{ formation & coarsening of multipoles

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        return FMMLibExpansionWrangler.form_multipoles(
            self, level_start_source_box_nrs, source_boxes, src_weights
        )

    def coarsen_multipoles(
        self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
    ):
        return FMMLibExpansionWrangler.coarsen_multipoles(
            self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
        )

    # }}} End formation & coarsening of multipoles

    # {{{ direct evaluation of near field interactions

    def eval_direct_single_out_kernel(
        self,
        out_pot,
        out_kernel,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        mode_coefs,
    ):

        # NOTE: mode_coefs are similar to source_weights BUT
        # do not include quadrature weights (purely function
        # expansiona coefficients)

        if 0:
            print("Returns range for list1")
            out_pot[:] = np.arange(len(out_pot))
            return out_pot, None

        kname = out_kernel.__repr__()

        if isinstance(self.n_tables, int) and self.n_tables > 1:
            use_multilevel_tables = True
        elif isinstance(self.n_tables, dict) and self.n_tables[kname] > 1:
            use_multilevel_tables = True
        else:
            use_multilevel_tables = False

        if use_multilevel_tables:
            # this checks that the boxes at the coarsest level
            # and allows for some round-off error
            min_lev = np.min(
                self.tree.box_levels.get(self.queue)[target_boxes.get(self.queue)]
            )
            largest_cell_extent = self.tree.root_extent * 0.5**min_lev
            if not self.near_field_table[kname][0].source_box_extent >= (
                largest_cell_extent - 1e-15
            ):
                raise RuntimeError(
                    "Insufficient list of tables: the "
                    "coarsest level mesh cells at level "
                    + str(min_lev)
                    + " are not covered."
                )

        symmetry_source_direction = _extract_symmetry_source_direction(
            out_kernel,
            self.source_extra_kwargs,
            self.queue,
        )
        for table in self.near_field_table[kname]:
            table.symmetry_source_direction = symmetry_source_direction

        near_field_tables = self.near_field_table[kname]
        table0 = near_field_tables[0]
        configured_dtype = np.dtype(self.dtype)
        if out_kernel.is_complex_valued:
            if configured_dtype.kind == "c":
                eval_dtype = configured_dtype
            else:
                eval_dtype = (
                    np.complex64 if configured_dtype == np.float32 else np.complex128
                )
        else:
            if configured_dtype.kind == "f":
                eval_dtype = configured_dtype
            else:
                eval_dtype = (
                    np.float32 if configured_dtype == np.complex64 else np.float64
                )
        cache_key = (
            kname,
            tuple(int(np.asarray(tbl.data).ctypes.data) for tbl in near_field_tables),
            tuple(int(np.asarray(tbl.data).size) for tbl in near_field_tables),
            tuple(_table_data_fingerprint(tbl.data) for tbl in near_field_tables),
            np.dtype(eval_dtype).str,
            None
            if symmetry_source_direction is None
            else tuple(np.asarray(symmetry_source_direction, dtype=float).tolist()),
        )
        payload = self._get_cached_nearfield_payload(
            cache_key,
            self.queue,
            table0,
            near_field_tables,
            eval_dtype,
        )

        base = payload["base"]
        shift = payload["shift"]
        case_indices_dev = payload["case_indices_dev"]
        mode_qpoint_map_dev = payload["mode_qpoint_map_dev"]
        mode_case_map_dev = payload["mode_case_map_dev"]
        table_data_shapes = payload["table_data_shapes"]
        assert table_data_shapes["n_q_points"] == len(
            self.near_field_table[kname][0].mode_normalizers
        )

        from volumential.list1 import NearFieldFromCSR

        near_field = NearFieldFromCSR(
            out_kernel,
            table_data_shapes,
            potential_kind=self.potential_kind,
            **self.list1_extra_kwargs,
        )

        table_data_combined = payload["table_data_dev"]
        mode_nmlz_combined = payload["mode_nmlz_dev"]
        exterior_mode_nmlz_combined = payload["exterior_mode_nmlz_dev"]
        table_entry_ids = payload["table_entry_ids_dev"]
        table_entry_scales = payload["table_entry_scales_dev"]
        particle_local_ids = _compute_box_local_ids(
            self.queue, self.tree, self.near_field_table[kname][0].n_q_points
        )

        _validate_table_box_particle_layout_cached(
            self.queue,
            self.tree,
            target_boxes,
            neighbor_source_boxes_lists,
            self.near_field_table[kname][0].n_q_points,
            self._table_layout_validation_cache,
        )

        aligned_nboxes = self.tree.box_centers.shape[1]
        source_counts_nonchild = np.zeros(aligned_nboxes, dtype=np.int32)
        target_counts_nonchild = np.zeros(aligned_nboxes, dtype=np.int32)
        source_counts_nonchild[: len(self.tree.box_target_counts_nonchild)] = (
            self.tree.box_target_counts_nonchild.get(self.queue)
        )
        target_counts_nonchild[: len(self.tree.box_target_counts_nonchild)] = (
            self.tree.box_target_counts_nonchild.get(self.queue)
        )
        source_counts_nonchild = cl.array.to_device(self.queue, source_counts_nonchild)
        target_counts_nonchild = cl.array.to_device(self.queue, target_counts_nonchild)

        res, evt = near_field(
            self.queue,
            result=out_pot,
            box_centers=self.tree.box_centers,
            box_levels=self.tree.box_levels,
            box_source_counts_nonchild=source_counts_nonchild,
            box_source_starts=self.tree.box_target_starts,
            box_target_counts_nonchild=target_counts_nonchild,
            box_target_starts=self.tree.box_target_starts,
            case_indices=case_indices_dev,
            mode_qpoint_map=mode_qpoint_map_dev,
            mode_case_map=mode_case_map_dev,
            mode_case_scale=payload["mode_case_scale_dev"],
            encoding_base=base,
            encoding_shift=shift,
            mode_nmlz_combined=mode_nmlz_combined,
            exterior_mode_nmlz_combined=exterior_mode_nmlz_combined,
            table_entry_ids=table_entry_ids,
            table_entry_scales=table_entry_scales,
            neighbor_source_boxes_starts=neighbor_source_boxes_starts,
            root_extent=self.tree.root_extent,
            neighbor_source_boxes_lists=neighbor_source_boxes_lists,
            mode_coefs=mode_coefs,
            source_mode_ids=particle_local_ids,
            table_data_combined=table_data_combined,
            target_boxes=target_boxes,
            target_point_ids=particle_local_ids,
            table_root_extent=self.root_table_source_box_extent,
            table_starting_level=self.table_starting_level,
        )

        if isinstance(out_pot, cl.array.Array):
            assert res is out_pot
            # FIXME: lazy evaluation sometimes returns incorrect results
            res.finish()
        else:
            assert isinstance(out_pot, np.ndarray)
            out_pot = res

        # sorted_target_ids=self.tree.user_source_ids,
        # user_source_ids=self.tree.user_source_ids)

        scale_factor = self.get_scale_factor()
        return out_pot / scale_factor, evt

    def _get_cached_nearfield_payload(
        self,
        cache_key,
        queue,
        table0,
        near_field_tables,
        eval_dtype,
    ):
        payload = self._nearfield_device_payload_cache.get(cache_key)
        if payload is not None:
            self._nearfield_device_payload_cache.move_to_end(cache_key)
            return payload

        distinct_numbers = set()
        for vec in table0.interaction_case_vecs:
            for cvc in vec:
                distinct_numbers.add(cvc)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        case_indices_dev = cl.array.to_device(queue, table0.case_indices)
        symmetry_maps = table0._get_online_symmetry_maps()
        mode_qpoint_map_dev = cl.array.to_device(
            queue, symmetry_maps["mode_qpoint_map"]
        )
        mode_case_map_dev = cl.array.to_device(queue, symmetry_maps["mode_case_map"])
        mode_case_scale = symmetry_maps.get("mode_case_scale")
        if mode_case_scale is None:
            mode_case_scale = np.ones(
                (table0.n_q_points, table0.n_cases),
                dtype=np.int8,
            )

        (
            table_data_combined,
            mode_nmlz_combined,
            exterior_mode_nmlz_combined,
            table_entry_ids,
            table_entry_scales,
        ) = _prepare_table_data_and_entry_map(near_field_tables)

        if table_data_combined.dtype != eval_dtype:
            table_data_combined = table_data_combined.astype(eval_dtype)
        if mode_nmlz_combined.dtype != eval_dtype:
            mode_nmlz_combined = mode_nmlz_combined.astype(eval_dtype)
        if exterior_mode_nmlz_combined.dtype != eval_dtype:
            exterior_mode_nmlz_combined = exterior_mode_nmlz_combined.astype(eval_dtype)
        if table_entry_scales.dtype != eval_dtype:
            table_entry_scales = table_entry_scales.astype(eval_dtype)
        if mode_case_scale.dtype != eval_dtype:
            mode_case_scale = mode_case_scale.astype(eval_dtype)

        table_data_shapes = {
            "n_tables": len(near_field_tables),
            "n_q_points": table0.n_q_points,
            "n_cases": table0.n_cases,
            "n_table_entries": table_data_combined.shape[1],
        }

        payload = {
            "base": base,
            "shift": shift,
            "case_indices_dev": case_indices_dev,
            "mode_qpoint_map_dev": mode_qpoint_map_dev,
            "mode_case_map_dev": mode_case_map_dev,
            "mode_case_scale_dev": cl.array.to_device(queue, mode_case_scale),
            "table_data_dev": cl.array.to_device(queue, table_data_combined),
            "mode_nmlz_dev": cl.array.to_device(queue, mode_nmlz_combined),
            "exterior_mode_nmlz_dev": cl.array.to_device(
                queue, exterior_mode_nmlz_combined
            ),
            "table_entry_ids_dev": cl.array.to_device(queue, table_entry_ids),
            "table_entry_scales_dev": cl.array.to_device(queue, table_entry_scales),
            "table_data_shapes": table_data_shapes,
        }
        self._nearfield_device_payload_cache[cache_key] = payload
        while (
            len(self._nearfield_device_payload_cache)
            > self._nearfield_device_payload_cache_max
        ):
            self._nearfield_device_payload_cache.popitem(last=False)
        return payload

    def eval_direct(
        self,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        mode_coefs,
    ):
        pot = self.output_zeros()
        if pot.dtype != object:
            pot = obj_array_1d(
                [
                    pot,
                ]
            )
        events = []
        for i in range(len(self.tree_indep.target_kernels)):
            # print("processing near-field of out_kernel", i)
            pot[i], evt = self.eval_direct_single_out_kernel(
                pot[i],
                self.tree_indep.target_kernels[i],
                target_boxes,
                neighbor_source_boxes_starts,
                neighbor_source_boxes_lists,
                mode_coefs,
            )
            events.append(evt)

        for out_pot in pot:
            if isinstance(out_pot, cl.array.Array):
                out_pot.finish()

        # boxtree.pyfmmlib_integration handles things differently
        # when target_kernels has only one element
        if len(pot) == 1:
            pot = pot[0]

        return (pot, SumpyTimingFuture(self.queue, events))

    # }}} End direct evaluation of near field interactions

    # {{{ downward pass of fmm

    def multipole_to_local(
        self,
        level_start_target_box_nrs,
        target_boxes,
        src_box_starts,
        src_box_lists,
        mpole_exps,
    ):
        return FMMLibExpansionWrangler.multipole_to_local(
            self,
            level_start_target_box_nrs,
            target_boxes,
            src_box_starts,
            src_box_lists,
            mpole_exps,
        )

    def eval_multipoles(
        self, target_boxes_by_source_level, source_boxes_by_level, mpole_exps
    ):
        return FMMLibExpansionWrangler.eval_multipoles(
            self, target_boxes_by_source_level, source_boxes_by_level, mpole_exps
        )

    def form_locals(
        self,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        starts,
        lists,
        src_weights,
    ):
        return FMMLibExpansionWrangler.form_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            src_weights,
        )

    def refine_locals(
        self,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        return FMMLibExpansionWrangler.refine_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        )

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        return FMMLibExpansionWrangler.eval_locals(
            self, level_start_target_box_nrs, target_boxes, local_exps
        )

    # }}} End downward pass of fmm

    # {{{ direct evaluation of p2p (discrete) interactions

    def eval_direct_p2p(
        self, target_boxes, source_box_starts, source_box_lists, src_weights
    ):
        return FMMLibExpansionWrangler.eval_direct(
            self, target_boxes, source_box_starts, source_box_lists, src_weights
        )

    # }}} End direct evaluation of p2p interactions

    @staticmethod
    def is_supported_helmknl(knl):
        if isinstance(knl, DirectionalSourceDerivative):
            knl = knl.inner_kernel

        return isinstance(knl, (LaplaceKernel, HelmholtzKernel)) and knl.dim in (2, 3)


# }}} End fmmlib backend (for laplace, helmholtz)


class FPNDTreeIndependentDataForWrangler(FPNDSumpyTreeIndependentDataForWrangler):
    """The default tree-independent-data class."""


class FPNDExpansionWrangler(FPNDSumpyExpansionWrangler):
    """The default wrangler class."""


# vim: filetype=pyopencl:foldmethod=marker
