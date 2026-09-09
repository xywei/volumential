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

__doc__ = """Helmholtz near-field split correction for the sumpy backend.

Owns everything behind ``helmholtz_split``: choosing the split order from the
box-scaled wave number, building/validating the per-term near-field tables,
the series-remainder P2P, the smooth correction sources, and the self
-interaction diagonal limit.
"""

import json
import logging

import numpy as np

import pyopencl as cl
import pyopencl.array

from pytools.obj_array import new_1d as obj_array_1d
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    DirectionalSourceDerivative,
    HelmholtzKernel,
    LaplaceKernel,
    YukawaKernel,
)

from volumential.wranglers.barycentric import (
    _barycentric_interp_matrix,
    _gauss_legendre_nodes_and_weights,
)
from volumential.wranglers.kernel_symmetry import (
    _extract_symmetry_source_direction,
)
from volumential.wranglers.kernels import (
    _HelmholtzSplitSeriesRemainderKernel,
    _RadialPowerKernel,
    _RadialPowerLogKernel,
)
from volumential.wranglers.split_terms import (
    HelmholtzSplitCacheAccounting,
    _format_helmholtz_split_term_key,
    _nearfield_table_payload_bytes,
    _normalize_helmholtz_split_term_key,
    _select_split_order_from_rho_components,
)
from volumential.wranglers.timing import SumpyTimingFuture


logger = logging.getLogger(__name__)


class HelmholtzSplitCorrectionMixin:
    """Near-field Helmholtz split correction, mixed into the sumpy wrangler."""

    def get_helmholtz_split_cache_accounting(self, parameter_count=1):
        """Return table-storage accounting for Helmholtz/Yukawa split mode."""
        if not isinstance(parameter_count, int):
            try:
                parameter_count = len(parameter_count)
            except TypeError as exc:
                raise TypeError(
                    "parameter_count must be an integer or a sized parameter collection"
                ) from exc
        parameter_count = int(parameter_count)
        if parameter_count < 1:
            raise ValueError("parameter_count must be >= 1")

        base_tables = [
            table
            for tables in self.near_field_table.values()
            for table in tables
        ]
        split_term_items = sorted(
            self.helmholtz_split_term_tables.items(),
            key=lambda item: _format_helmholtz_split_term_key(item[0]),
        )
        split_term_tables = [
            table
            for _, tables in split_term_items
            for table in tables
        ]

        base_table_payload_bytes = sum(
            _nearfield_table_payload_bytes(table) for table in base_tables
        )
        split_term_table_payload_bytes = sum(
            _nearfield_table_payload_bytes(table) for table in split_term_tables
        )

        return HelmholtzSplitCacheAccounting(
            split_enabled=bool(self.helmholtz_split),
            split_order=int(self.helmholtz_split_order),
            parameter_count=parameter_count,
            base_table_count=len(base_tables),
            split_term_table_count=len(split_term_tables),
            basis_table_count=len(base_tables) + len(split_term_tables),
            base_table_payload_bytes=int(base_table_payload_bytes),
            split_term_table_payload_bytes=int(split_term_table_payload_bytes),
            total_table_payload_bytes=int(
                base_table_payload_bytes + split_term_table_payload_bytes
            ),
            split_term_keys=tuple(key for key, _ in split_term_items),
            uses_online_coefficients=bool(self.helmholtz_split and split_term_tables),
            uses_online_remainder=bool(self.helmholtz_split),
        )

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
    def _helmholtz_split_wrapper_chain_has_mixed_source_target(wrappers):
        has_target = any(kind == "axis_target" for kind, _value in wrappers)
        has_source = any(
            kind in ("axis_source", "directional_source") for kind, _value in wrappers
        )
        return has_target and has_source

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
                wrapper_chain = self._extract_helmholtz_split_kernel_wrapper_chain(
                    out_knl,
                    base_knl,
                )
            except NotImplementedError as exc:
                return False, str(exc)

            if self._helmholtz_split_wrapper_chain_has_mixed_source_target(
                wrapper_chain
            ):
                return (
                    False,
                    "mixed source/target derivative wrapper chains are unsupported "
                    "in split mode without a dedicated regression test and scaling rule",
                )

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

        power_log_keys = [
            _normalize_helmholtz_split_term_key(("power_log", 2 * n))
            for n in range(1, self.helmholtz_split_order)
        ]
        beta_mode = str(
            self._helmholtz_split_auto_config.get(
                "power_log_single_table_beta_mode",
                "p2p",
            )
        ).strip().lower()
        if beta_mode == "p2p":
            return power_log_keys
        if beta_mode == "table":
            required_keys = list(power_log_keys)
            term_tables = getattr(self, "helmholtz_split_term_tables", {})
            for power_log_key in power_log_keys:
                if len(term_tables.get(power_log_key, ())) == 1:
                    _, power = power_log_key
                    required_keys.append(
                        _normalize_helmholtz_split_term_key(("power", power))
                    )
            return required_keys
        raise ValueError(
            "power_log_single_table_beta_mode must be 'table' or 'p2p'"
        )

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

# vim: filetype=pyopencl:foldmethod=marker
