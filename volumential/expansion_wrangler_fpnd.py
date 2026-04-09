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
    AxisTargetDerivative,
    DirectionalSourceDerivative,
    ExpressionKernel,
    HelmholtzKernel,
    LaplaceKernel,
)

from volumential.expansion_wrangler_interface import (
    ExpansionWranglerInterface,
    TreeIndependentDataForWranglerInterface,
)
from volumential.nearfield_potential_table import NearFieldInteractionTable


logger = logging.getLogger(__name__)


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
    else:
        for table in table_levels:
            if not np.all(np.isfinite(table.data)):
                raise RuntimeError("full near-field table contains non-finite entries")
        kept_entry_ids = np.arange(n_full_entries, dtype=np.int64)

    table_entry_ids = np.full(n_full_entries, -1, dtype=np.int32)
    table_entry_ids[kept_entry_ids] = np.arange(len(kept_entry_ids), dtype=np.int32)

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
        helmholtz_split=False,
        helmholtz_split_order=1,
        helmholtz_split_smooth_quad_order=None,
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
        This mode currently supports exactly one base Helmholtz target kernel in
        2D or 3D.

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
        source strengths.

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

        self.helmholtz_split = bool(helmholtz_split)
        self.helmholtz_split_order = int(helmholtz_split_order)
        if self.helmholtz_split_order < 1:
            raise ValueError("helmholtz_split_order must be >= 1")

        self.helmholtz_split_order1_legacy_subtraction = bool(
            helmholtz_split_order1_legacy_subtraction
        )

        self._helmholtz_split_p2p_helm = None
        self._helmholtz_split_p2p_lap = None
        self._helmholtz_split_p2p_helm_include_self = None
        self._helmholtz_split_p2p_lap_include_self = None
        self._helmholtz_split_term_p2p = {}
        self._helmholtz_split_term_p2p_include_self = {}
        self._helmholtz_split_remainder_p2p = None
        self._helmholtz_split_remainder_p2p_include_self = None
        self._helmholtz_split_remainder_kernel_cache = {}
        self._helmholtz_split_kernels = None
        self._helmholtz_split_constant_kernel = None
        self._helmholtz_split_power_kernels = {}
        self._helmholtz_split_power_log_kernels = {}
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
            helmholtz_split_smooth_quad_order = self.quad_order

        if helmholtz_split_smooth_quad_order is None:
            self.helmholtz_split_smooth_quad_order = None
        else:
            self.helmholtz_split_smooth_quad_order = int(
                helmholtz_split_smooth_quad_order
            )
            if self.helmholtz_split_smooth_quad_order < 1:
                raise ValueError("helmholtz_split_smooth_quad_order must be >= 1")

        if self.helmholtz_split:
            from sumpy.kernel import HelmholtzKernel, LaplaceKernel

            from volumential.table_manager import ConstantKernel

            if len(self.tree_indep.target_kernels) != 1:
                raise NotImplementedError(
                    "helmholtz_split currently supports exactly one target kernel"
                )

            out_knl = self.tree_indep.target_kernels[0]
            base_knl = out_knl.get_base_kernel()
            if out_knl is not base_knl or not isinstance(base_knl, HelmholtzKernel):
                raise NotImplementedError(
                    "helmholtz_split currently supports only base Helmholtz kernels"
                )

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

            list1_extra_kwargs = dict(list1_extra_kwargs)
            if list1_extra_kwargs.get("infer_kernel_scaling", False):
                raise RuntimeError(
                    "helmholtz_split does not support infer_kernel_scaling; "
                    "provide explicit split scaling/displacement only"
                )

            list1_extra_kwargs.setdefault("infer_kernel_scaling", False)
            if base_knl.dim == 3:
                list1_extra_kwargs.setdefault(
                    "kernel_scaling_code",
                    "BOX_extent * BOX_extent / (table_root_extent * table_root_extent)",
                )
                list1_extra_kwargs.setdefault("kernel_displacement_code", "0.0")
            else:
                list1_extra_kwargs.setdefault(
                    "kernel_scaling_code",
                    "BOX_extent * BOX_extent / (table_root_extent * table_root_extent)",
                )
                list1_extra_kwargs.setdefault(
                    "kernel_displacement_code",
                    "-0.5 / 3.141592653589793 * scaling * "
                    "log(BOX_extent / table_root_extent) * "
                    "mode_nmlz[table_lev, sid]",
                )

            self._helmholtz_split_kernels = (
                base_knl,
                LaplaceKernel(base_knl.dim),
            )
            self._helmholtz_split_constant_kernel = ConstantKernel(base_knl.dim)
        self.list1_extra_kwargs = list1_extra_kwargs
        self._table_layout_validation_cache = set()

    # }}} End constructor

    # {{{ data vector utilities

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

        # table.case_encode
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

        (
            table_data_combined,
            mode_nmlz_combined,
            exterior_mode_nmlz_combined,
            table_entry_ids,
        ) = _prepare_table_data_and_entry_map(near_field_tables)

        if getattr(out_kernel, "is_complex_valued", False):
            if table_data_combined.dtype != self.dtype:
                table_data_combined = table_data_combined.astype(self.dtype)
            if mode_nmlz_combined.dtype != self.dtype:
                mode_nmlz_combined = mode_nmlz_combined.astype(self.dtype)
            if exterior_mode_nmlz_combined.dtype != self.dtype:
                exterior_mode_nmlz_combined = exterior_mode_nmlz_combined.astype(
                    self.dtype
                )

        queue.finish()
        logger.info("table data for kernel " + out_kernel.__repr__() + " congregated")

        # The loop domain needs to know some info about the tables being used
        table_data_shapes = {
            "n_tables": n_tables_local,
            "n_q_points": table0.n_q_points,
            "n_cases": table0.n_cases,
            "n_table_entries": table_data_combined.shape[1],
        }
        assert table_data_shapes["n_q_points"] == len(table0.mode_normalizers)

        from volumential.list1 import NearFieldFromCSR

        near_field = NearFieldFromCSR(
            out_kernel,
            table_data_shapes,
            potential_kind=self.potential_kind,
            **list1_extra_kwargs,
        )

        table_data_combined = cl.array.to_device(queue, table_data_combined)
        mode_nmlz_combined = cl.array.to_device(queue, mode_nmlz_combined)
        exterior_mode_nmlz_combined = cl.array.to_device(
            queue, exterior_mode_nmlz_combined
        )
        table_entry_ids = cl.array.to_device(queue, table_entry_ids)
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

        queue.finish()
        logger.info("sent table data to device")

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
            encoding_base=base,
            encoding_shift=shift,
            mode_nmlz_combined=mode_nmlz_combined,
            exterior_mode_nmlz_combined=exterior_mode_nmlz_combined,
            table_entry_ids=table_entry_ids,
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
            out_pot.finish()

        return (pot, SumpyTimingFuture(self.queue, events))

    def eval_direct_helmholtz_split_correction(
        self,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        src_weights,
        src_func=None,
    ):
        if not self.helmholtz_split:
            return self.output_zeros(), SumpyTimingFuture(self.queue, [])

        shared_kwargs = {}
        shared_kwargs.update(self.self_extra_kwargs)
        shared_kwargs.update(self.box_source_list_kwargs())
        shared_kwargs.update(self.box_target_list_kwargs())

        if "target_to_source" in shared_kwargs and isinstance(
            shared_kwargs["target_to_source"], np.ndarray
        ):
            shared_kwargs["target_to_source"] = cl.array.to_device(
                self.queue, shared_kwargs["target_to_source"]
            )

        smooth_quad_order = self.helmholtz_split_smooth_quad_order
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
                    beta_weighted_strength = (
                        self._fold_helmholtz_split_log_alpha_into_strength(
                            src_weights,
                            float(term_tables[0].source_box_extent),
                        )
                    )

                    beta_strength = obj_array_1d([beta_weighted_strength])
                    power_kernel = self._get_helmholtz_split_power_kernel(term_power)
                    p2p_power = self._get_helmholtz_split_term_p2p(
                        power_kernel,
                        exclude_self=exclude_self,
                    )

                    beta_kwargs = dict(shared_kwargs)
                    if not exclude_self:
                        beta_kwargs.pop("target_to_source", None)

                    beta_power_term = _run_p2p_from_csr(
                        p2p_power,
                        beta_kwargs,
                        strength_arg=beta_strength,
                        max_nsources_in_one_box_arg=self.max_nsources_in_one_box,
                    )

                    term_contribution = term_contribution + beta_power_term[0]

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

        if exclude_self:
            if (
                self._helmholtz_split_p2p_helm is None
                or self._helmholtz_split_p2p_lap is None
            ):
                helm_knl, lap_knl = self._helmholtz_split_kernels
                self._helmholtz_split_p2p_helm = P2PFromCSR(
                    [helm_knl],
                    True,
                    value_dtypes=[self.dtype],
                )
                self._helmholtz_split_p2p_lap = P2PFromCSR(
                    [lap_knl],
                    True,
                    value_dtypes=[self.dtype],
                )
            return self._helmholtz_split_p2p_helm, self._helmholtz_split_p2p_lap

        if (
            self._helmholtz_split_p2p_helm_include_self is None
            or self._helmholtz_split_p2p_lap_include_self is None
        ):
            helm_knl, lap_knl = self._helmholtz_split_kernels
            self._helmholtz_split_p2p_helm_include_self = P2PFromCSR(
                [helm_knl],
                False,
                value_dtypes=[self.dtype],
            )
            self._helmholtz_split_p2p_lap_include_self = P2PFromCSR(
                [lap_knl],
                False,
                value_dtypes=[self.dtype],
            )

        return (
            self._helmholtz_split_p2p_helm_include_self,
            self._helmholtz_split_p2p_lap_include_self,
        )

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

        k_name = helm_knl.helmholtz_k_name
        if k_name not in self.kernel_extra_kwargs:
            raise RuntimeError("missing Helmholtz wave number for split terms")

        k = np.complex128(self.kernel_extra_kwargs[k_name])
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

    def _get_helmholtz_split_remainder_kernel(self):
        helm_knl, _ = self._helmholtz_split_kernels
        dim = int(helm_knl.dim)
        split_order = int(self.helmholtz_split_order)
        k_name = helm_knl.helmholtz_k_name
        if k_name not in self.kernel_extra_kwargs:
            raise RuntimeError("missing Helmholtz wave number for split remainder")

        k = np.complex128(self.kernel_extra_kwargs[k_name])
        series_nmax = self._helmholtz_split_series_nmax(split_order)

        cache_key = (
            dim,
            split_order,
            float(np.real(k)),
            float(np.imag(k)),
            int(series_nmax),
        )
        if cache_key not in self._helmholtz_split_remainder_kernel_cache:
            self._helmholtz_split_remainder_kernel_cache[cache_key] = (
                _HelmholtzSplitSeriesRemainderKernel(
                    dim,
                    np.real(k),
                    np.imag(k),
                    split_order,
                    series_nmax,
                )
            )

        return self._helmholtz_split_remainder_kernel_cache[cache_key]

    def _get_helmholtz_split_remainder_p2p(self, *, exclude_self):
        from sumpy import P2PFromCSR

        remainder_kernel = self._get_helmholtz_split_remainder_kernel()

        if exclude_self:
            if self._helmholtz_split_remainder_p2p is None:
                self._helmholtz_split_remainder_p2p = P2PFromCSR(
                    [remainder_kernel],
                    True,
                    value_dtypes=[self.dtype],
                )
            return self._helmholtz_split_remainder_p2p

        if self._helmholtz_split_remainder_p2p_include_self is None:
            self._helmholtz_split_remainder_p2p_include_self = P2PFromCSR(
                [remainder_kernel],
                False,
                value_dtypes=[self.dtype],
            )
        return self._helmholtz_split_remainder_p2p_include_self

    def _get_helmholtz_split_term_tables(self, term_key):
        normalized_term_key = _normalize_helmholtz_split_term_key(term_key)
        if normalized_term_key not in self.helmholtz_split_term_tables:
            raise RuntimeError(
                "missing precomputed helmholtz split term tables for "
                f"{_format_helmholtz_split_term_key(normalized_term_key)}"
            )

        return self.helmholtz_split_term_tables[normalized_term_key]

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
                return "Constant", None
            return f"SplitPower{power}", self._get_helmholtz_split_power_kernel(power)

        if kind == "power_log":
            return (
                f"SplitPowerLog{power}",
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

    def _helmholtz_split_term_scaling_code(self, term_key):
        kind, power = _normalize_helmholtz_split_term_key(term_key)
        if kind not in {"power", "power_log"}:
            raise RuntimeError(
                f"unsupported split term kind for single-table scaling: {kind}"
            )

        exponent = int(self.tree.dimensions) + int(power)
        if exponent <= 0:
            return "1.0"

        box_factor = " * ".join(["BOX_extent"] * exponent)
        table_factor = " * ".join(["table_root_extent"] * exponent)
        return f"({box_factor}) / ({table_factor})"

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

        out_pot = self.output_zeros()[0]
        out_pot, _ = self.eval_direct_single_out_kernel(
            out_pot,
            term_kernel,
            target_boxes,
            neighbor_source_boxes_starts,
            neighbor_source_boxes_lists,
            src_weights,
            near_field_tables=term_tables,
            list1_extra_kwargs=list1_kwargs,
        )

        return obj_array_1d([out_pot])

    def _get_helmholtz_split_log_alpha_per_source(self, table_root_extent):
        table_root_extent = float(table_root_extent)
        if table_root_extent <= 0.0:
            raise ValueError("table_root_extent must be positive")

        cache_key = table_root_extent
        if cache_key in self._helmholtz_split_log_alpha_per_source_cache:
            return self._helmholtz_split_log_alpha_per_source_cache[cache_key]

        box_source_starts = self.tree.box_source_starts.get(self.queue)
        box_source_counts = self.tree.box_source_counts_nonchild.get(self.queue)
        box_levels = self.tree.box_levels.get(self.queue)

        nsources = int(self.tree.nsources)
        beta_host = np.zeros(nsources, dtype=np.float64)
        root_extent = float(self.tree.root_extent)

        for ibox in range(len(box_source_starts)):
            count = int(box_source_counts[ibox])
            if count <= 0:
                continue

            start = int(box_source_starts[ibox])
            stop = start + count
            box_extent = root_extent * (0.5 ** int(box_levels[ibox]))
            beta_value = np.log(box_extent / table_root_extent)
            beta_host[start:stop] = beta_value

        beta_host = np.ascontiguousarray(beta_host)
        self._helmholtz_split_log_alpha_per_source_cache[cache_key] = beta_host
        return beta_host

    def _fold_helmholtz_split_log_alpha_into_strength(
        self, src_weights, table_root_extent
    ):
        beta_host = self._get_helmholtz_split_log_alpha_per_source(table_root_extent)

        if isinstance(src_weights, cl.array.Array):
            dtype = np.dtype(src_weights.dtype)
            dev_key = (float(table_root_extent), dtype.str)
            beta_dev = self._helmholtz_split_log_alpha_per_source_dev_cache.get(dev_key)
            if beta_dev is None:
                beta_dev = cl.array.to_device(
                    self.queue,
                    beta_host.astype(dtype, copy=False),
                )
                self._helmholtz_split_log_alpha_per_source_dev_cache[dev_key] = beta_dev

            return src_weights * beta_dev

        src_weights_arr = np.asarray(src_weights)
        beta_arr = beta_host.astype(src_weights_arr.dtype, copy=False)
        return src_weights_arr * beta_arr

    def _get_helmholtz_split_power_kernel(self, power):
        power = int(power)
        if power < 0:
            raise ValueError("power must be non-negative")

        if power == 0:
            return self._helmholtz_split_constant_kernel

        if power not in self._helmholtz_split_power_kernels:
            self._helmholtz_split_power_kernels[power] = _RadialPowerKernel(
                self.tree.dimensions,
                power,
            )

        return self._helmholtz_split_power_kernels[power]

    def _get_helmholtz_split_power_log_kernel(self, power):
        power = int(power)
        if power <= 0:
            raise ValueError("power must be positive for r**power * log(r)")

        if power not in self._helmholtz_split_power_log_kernels:
            self._helmholtz_split_power_log_kernels[power] = _RadialPowerLogKernel(
                self.tree.dimensions,
                power,
            )

        return self._helmholtz_split_power_log_kernels[power]

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
        k_name = helm_knl.helmholtz_k_name
        if k_name not in self.kernel_extra_kwargs:
            raise RuntimeError("missing Helmholtz wave number for split terms")

        k = np.complex128(self.kernel_extra_kwargs[k_name])

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

        helm_knl, _ = self._helmholtz_split_kernels
        k_name = helm_knl.helmholtz_k_name
        if k_name not in self.kernel_extra_kwargs:
            return None

        k = np.complex128(self.kernel_extra_kwargs[k_name])
        if helm_knl.dim == 3:
            return np.complex128(1j * k / (4.0 * np.pi))

        if helm_knl.dim == 2:
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

        if (
            hasattr(src_weights, "shape")
            and hasattr(like, "shape")
            and tuple(src_weights.shape) == tuple(like.shape)
        ):
            diag_strength = src_weights
        elif target_to_source is None:
            if not self.tree.sources_are_targets:
                return None
            if self.tree.ntargets != self.tree.nsources:
                return None
            diag_strength = src_weights
        else:
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

        if isinstance(diag_strength, cl.array.Array):
            scalar = np.array(limit, dtype=diag_strength.dtype)
            return diag_strength * scalar

        if isinstance(like, cl.array.Array):
            diag_strength = cl.array.to_device(self.queue, np.asarray(diag_strength))
            return diag_strength * np.array(limit, dtype=like.dtype)

        if isinstance(diag_strength, cl.array.Array):
            diag_strength = diag_strength.get(self.queue)

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

        # table.case_encode
        distinct_numbers = set()
        for vec in self.near_field_table[kname][0].interaction_case_vecs:
            for cvc in vec:
                distinct_numbers.add(cvc)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        case_indices_dev = cl.array.to_device(
            self.queue, self.near_field_table[kname][0].case_indices
        )
        symmetry_maps = self.near_field_table[kname][0]._get_online_symmetry_maps()
        mode_qpoint_map_dev = cl.array.to_device(
            self.queue, symmetry_maps["mode_qpoint_map"]
        )
        mode_case_map_dev = cl.array.to_device(
            self.queue, symmetry_maps["mode_case_map"]
        )

        (
            table_data_combined,
            mode_nmlz_combined,
            exterior_mode_nmlz_combined,
            table_entry_ids,
        ) = _prepare_table_data_and_entry_map(self.near_field_table[kname])

        logger.info("Table data for kernel " + out_kernel.__repr__() + " congregated")

        # The loop domain needs to know some info about the tables being used
        table_data_shapes = {
            "n_tables": len(self.near_field_table[kname]),
            "n_q_points": self.near_field_table[kname][0].n_q_points,
            "n_cases": self.near_field_table[kname][0].n_cases,
            "n_table_entries": table_data_combined.shape[1],
        }
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

        table_data_combined = cl.array.to_device(self.queue, table_data_combined)
        mode_nmlz_combined = cl.array.to_device(self.queue, mode_nmlz_combined)
        exterior_mode_nmlz_combined = cl.array.to_device(
            self.queue, exterior_mode_nmlz_combined
        )
        table_entry_ids = cl.array.to_device(self.queue, table_entry_ids)
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
            encoding_base=base,
            encoding_shift=shift,
            mode_nmlz_combined=mode_nmlz_combined,
            exterior_mode_nmlz_combined=exterior_mode_nmlz_combined,
            table_entry_ids=table_entry_ids,
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
