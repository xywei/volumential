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

__doc__ = """The sumpy-based fpnd expansion wrangler.

``fpnd`` = far field by Particle approximation, Near field Directly from
precomputed interaction tables.  This module owns the sumpy flavour of that
wrangler plus its tree-independent data, and the default aliases exported as
:class:`FPNDExpansionWrangler` / :class:`FPNDTreeIndependentDataForWrangler`.
"""

import logging
import math
from collections import OrderedDict

import numpy as np

import pyopencl as cl
import pyopencl.array
from pytools import memoize_method
from sumpy.array_context import PyOpenCLArrayContext
from sumpy.fmm import (
    SumpyExpansionWrangler,
    SumpyTreeIndependentDataForWrangler,
)

from volumential.expansion_wrangler_interface import (
    BoxIndexArray,
    ExpansionWranglerInterface,
    FMMArray,
    StageResult,
    TreeIndependentDataForWranglerInterface,
)
from volumential.nearfield_potential_table import NearFieldInteractionTable
from volumential.wranglers.arithmetic_orbits import (
    _ARITHMETIC_RECONSTRUCTION_KINDS,
)
from volumential.wranglers.box_layout import (
    _compute_box_local_ids,
    _validate_table_box_particle_layout_cached,
)
from volumential.wranglers.device_arrays import _resolve_queue, inverse_id_map
from volumential.wranglers.helmholtz_split import HelmholtzSplitCorrectionMixin
from volumential.wranglers.kernel_symmetry import (
    _derive_source_kernels_from_target_kernels,
    _extract_symmetry_source_direction,
    _target_kernels_include_source_derivatives,
)
from volumential.wranglers.nearfield_cache import NearFieldPayloadCacheMixin
from volumential.wranglers.split_terms import (
    _format_helmholtz_split_term_key,
    _normalize_helmholtz_split_term_key,
)
from volumential.wranglers.table_data import _table_data_fingerprint
from volumential.wranglers.timing import SumpyTimingFuture


logger = logging.getLogger(__name__)


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
        target_has_source_derivatives = _target_kernels_include_source_derivatives(
            target_kernels
        )
        expansion_source_kernels = source_kernels
        if source_kernels is None:
            expansion_source_kernels = _derive_source_kernels_from_target_kernels(
                target_kernels,
                require_single=not target_has_source_derivatives,
            )
            if not target_has_source_derivatives:
                source_kernels = expansion_source_kernels

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
        self.expansion_source_kernels = expansion_source_kernels

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
    ) -> "FPNDSumpyExpansionWrangler":
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

    @memoize_method
    def p2m(self, tgt_order: int):
        from sumpy.p2e import P2EFromSingleBox

        return P2EFromSingleBox(
            kernels=self.expansion_source_kernels,
            expansion=self.multipole_expansion(tgt_order),
            strength_usage=self.strength_usage,
            name="p2m",
        )

    @memoize_method
    def p2l(self, tgt_order: int):
        from sumpy.p2e import P2EFromCSR

        return P2EFromCSR(
            kernels=self.expansion_source_kernels,
            expansion=self.local_expansion(tgt_order),
            strength_usage=self.strength_usage,
            name="p2l",
        )

    def opencl_fft_app(self, shape, dtype, inverse: bool):
        from sumpy.tools import get_opencl_fft_app

        return get_opencl_fft_app(self._setup_actx, shape, dtype, inverse=inverse)


class FPNDSumpyExpansionWrangler(
    ExpansionWranglerInterface,
    NearFieldPayloadCacheMixin,
    HelmholtzSplitCorrectionMixin,
    SumpyExpansionWrangler,
):
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
                if helmholtz_split is None:
                    logger.info(
                        "[split:disable] unsupported target kernels: %s",
                        split_reason,
                    )
                    self.helmholtz_split = False
                else:
                    raise RuntimeError(
                        "helmholtz_split does not support the requested target "
                        f"kernels; {split_reason}"
                    )

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
                # Power-table beta corrections are needed only after a power-log
                # family resolves to one canonical table. Discover requirements
                # again after auto-building the primary power-log families.
                for _ in range(2):
                    required_term_keys = set(
                        self._helmholtz_split_required_term_keys(base_knl.dim)
                    )
                    missing_term_keys = sorted(
                        required_term_keys - set(self.helmholtz_split_term_tables),
                    )
                    if not missing_term_keys:
                        break
                    self._autobuild_helmholtz_split_term_tables(
                        out_knl, missing_term_keys
                    )

                required_term_keys = set(
                    self._helmholtz_split_required_term_keys(base_knl.dim)
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

    @property
    def _actx(self):
        return self.tree_indep._setup_actx

    def multipole_expansion_zeros(self, actx=None) -> FMMArray:
        if actx is None:
            actx = self._actx
        return SumpyExpansionWrangler.multipole_expansion_zeros(self, actx)

    def local_expansion_zeros(self, actx=None) -> FMMArray:
        if actx is None:
            actx = self._actx
        return SumpyExpansionWrangler.local_expansion_zeros(self, actx)

    def output_zeros(self, actx=None) -> FMMArray:
        if actx is None:
            actx = self._actx
        return SumpyExpansionWrangler.output_zeros(self, actx)

    def reorder_sources(self, source_array: FMMArray) -> FMMArray:
        return SumpyExpansionWrangler.reorder_sources(self, source_array)

    def reorder_targets(self, target_array: FMMArray) -> FMMArray:
        if not hasattr(self, "_user_target_ids"):
            self._user_target_ids = inverse_id_map(
                self.queue, self.tree.sorted_target_ids
            )
        return target_array.with_queue(self.queue)[self._user_target_ids]

    def reorder_potentials(self, potentials: FMMArray) -> FMMArray:
        return SumpyExpansionWrangler.reorder_potentials(self, potentials)

    def finalize_potentials(self, potentials: FMMArray) -> FMMArray:
        # return potentials
        return SumpyExpansionWrangler.finalize_potentials(self, self._actx, potentials)

    # }}} End data vector utilities

    # {{{ formation & coarsening of multipoles

    def form_multipoles(
        self,
        level_start_source_box_nrs: BoxIndexArray,
        source_boxes: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
        mpoles = SumpyExpansionWrangler.form_multipoles(
            self, self._actx, level_start_source_box_nrs, source_boxes, src_weights
        )
        return mpoles, SumpyTimingFuture(self.queue, [])

    def coarsen_multipoles(
        self,
        level_start_source_parent_box_nrs: BoxIndexArray,
        source_parent_boxes: BoxIndexArray,
        mpoles: FMMArray,
    ) -> StageResult:
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
        out_pot: FMMArray,
        out_kernel,
        target_boxes: BoxIndexArray,
        neighbor_source_boxes_starts: BoxIndexArray,
        neighbor_source_boxes_lists: BoxIndexArray,
        mode_coefs: FMMArray,
        near_field_tables=None,
        list1_extra_kwargs=None,
    ) -> tuple[FMMArray, object]:

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
        table_data_shapes = payload["table_data_shapes"]
        assert table_data_shapes["n_q_points"] == len(table0.mode_normalizers)
        reconstruction_kind = table_data_shapes.get("reconstruction_kind", "dense")

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
        if reconstruction_kind in _ARITHMETIC_RECONSTRUCTION_KINDS:
            reconstruction_kwargs = {
                "arithmetic_case_orbit_ranks": payload[
                    "arithmetic_case_orbit_ranks_dev"
                ],
                "arithmetic_case_axis_perm": payload[
                    "arithmetic_case_axis_perm_dev"
                ],
                "arithmetic_case_axis_sign": payload[
                    "arithmetic_case_axis_sign_dev"
                ],
                "arithmetic_case_axis_group": payload[
                    "arithmetic_case_axis_group_dev"
                ],
                "arithmetic_case_value_offsets": payload[
                    "arithmetic_case_value_offsets_dev"
                ],
                "arithmetic_axis_sign_power": payload[
                    "arithmetic_axis_sign_power_dev"
                ],
                "arithmetic_axis_direction_signs": payload[
                    "arithmetic_axis_direction_signs_dev"
                ],
                "arithmetic_direction_sign_axis": payload[
                    "arithmetic_direction_sign_axis"
                ],
            }
        elif reconstruction_kind == "generated-orbit":
            reconstruction_kwargs = {
                "reconstruction_qpoint_map": payload[
                    "reconstruction_qpoint_map_dev"
                ],
                "reconstruction_case_map": payload["reconstruction_case_map_dev"],
                "reconstruction_signs": payload["reconstruction_signs_dev"],
                "reconstruction_lookup_keys": payload[
                    "reconstruction_lookup_keys_dev"
                ],
                "reconstruction_lookup_values": payload[
                    "reconstruction_lookup_values_dev"
                ],
                "reconstruction_sign_lookup_keys": payload[
                    "reconstruction_sign_lookup_keys_dev"
                ],
                "reconstruction_sign_lookup_values": payload[
                    "reconstruction_sign_lookup_values_dev"
                ],
            }
        else:
            reconstruction_kwargs = {
                "mode_qpoint_map": payload["mode_qpoint_map_dev"],
                "mode_case_map": payload["mode_case_map_dev"],
                "mode_case_scale": payload["mode_case_scale_dev"],
                "table_entry_ids": payload["table_entry_ids_dev"],
                "table_entry_scales": payload["table_entry_scales_dev"],
            }
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
            encoding_base=base,
            encoding_shift=shift,
            mode_nmlz_combined=mode_nmlz_combined,
            exterior_mode_nmlz_combined=exterior_mode_nmlz_combined,
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
            **reconstruction_kwargs,
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
        target_boxes: BoxIndexArray,
        neighbor_source_boxes_starts: BoxIndexArray,
        neighbor_source_boxes_lists: BoxIndexArray,
        mode_coefs: FMMArray,
    ) -> StageResult:
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

    # }}} End direct evaluation of near field interactions

    # {{{ downward pass of fmm

    def multipole_to_local(
        self,
        level_start_target_box_nrs: BoxIndexArray,
        target_boxes: BoxIndexArray,
        src_box_starts: BoxIndexArray,
        src_box_lists: BoxIndexArray,
        mpole_exps: FMMArray,
    ) -> StageResult:
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
        self,
        target_boxes_by_source_level: BoxIndexArray,
        source_boxes_by_level: BoxIndexArray,
        mpole_exps: FMMArray,
    ) -> StageResult:
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
        level_start_target_or_target_parent_box_nrs: BoxIndexArray,
        target_or_target_parent_boxes: BoxIndexArray,
        starts: BoxIndexArray,
        lists: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
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
        level_start_target_or_target_parent_box_nrs: BoxIndexArray,
        target_or_target_parent_boxes: BoxIndexArray,
        local_exps: FMMArray,
    ) -> StageResult:
        local_exps = SumpyExpansionWrangler.refine_locals(
            self,
            self._actx,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        )
        return local_exps, SumpyTimingFuture(self.queue, [])

    def eval_locals(
        self,
        level_start_target_box_nrs: BoxIndexArray,
        target_boxes: BoxIndexArray,
        local_exps: FMMArray,
    ) -> StageResult:
        pot = SumpyExpansionWrangler.eval_locals(
            self, self._actx, level_start_target_box_nrs, target_boxes, local_exps
        )
        return pot, SumpyTimingFuture(self.queue, [])

    # }}} End downward pass of fmm

    # {{{ direct evaluation of p2p (discrete) interactions

    def eval_direct_p2p(
        self,
        target_boxes: BoxIndexArray,
        source_box_starts: BoxIndexArray,
        source_box_lists: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
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


class FPNDTreeIndependentDataForWrangler(FPNDSumpyTreeIndependentDataForWrangler):
    """The default tree-independent-data class."""


class FPNDExpansionWrangler(FPNDSumpyExpansionWrangler):
    """The default wrangler class."""

# vim: filetype=pyopencl:foldmethod=marker
