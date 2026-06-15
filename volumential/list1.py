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

__doc__ = """
.. autoclass:: KernelScalingPolicy
   :members:
.. autoclass:: NearFieldEvalBase
   :members:
.. autoclass:: NearFieldFromCSR
   :members:
"""

import logging
from dataclasses import dataclass, fields

import numpy as np

import loopy
import pyopencl as cl
from pytools import memoize_method

from volumential.tools import KernelCacheWrapper


logger = logging.getLogger(__name__)


CASE_ENCODING_BIAS_DEFAULT = 1.0e-10


@dataclass(frozen=True)
class KernelScalingPolicy:
    """Describes how a List 1 evaluator selects and rescales table data."""

    kernel_name: str
    mode: str
    infer_kernel_scaling: bool
    single_table_scaling_supported: bool
    reference_table_level: int | None
    scaling_code: str
    displacement_code: str
    table_level_code: str
    notes: str


def _array_layout_cache_token(ary, queue):
    if isinstance(ary, cl.array.Array):
        base_data = getattr(ary, "base_data", None)
        int_ptr = getattr(base_data, "int_ptr", None)
        if int_ptr is not None:
            return (
                "cl",
                int(id(queue)),
                int(int_ptr),
                int(getattr(ary, "offset", 0)),
                int(ary.size),
                ary.dtype.str,
            )
    return ("py", id(ary))


# {{{ near field eval base class


class NearFieldEvalBase(KernelCacheWrapper):
    """Base class of near-field evaluator."""

    default_name = "near_field_eval_base"

    def _supports_inferred_scaling(self):
        return False

    def __init__(
        self,
        integral_kernel,
        table_data_shapes,
        potential_kind=1,
        options=None,
        name=None,
        device=None,
        **kwargs,
    ):
        """potential_kind:

        1 - The (weakly singular) volume potentials.
        2 - The (hypersingular) inverse potentials, like the fractional Laplacian.
            Here, the fractional Laplacian is the inverse of the (weakly singular)
            Riesz potential operator.

        The two kinds share the same far-field code, but the second kind requires
        exterior_mode_nmlz when computing the list1 interactions.
        """
        if options is None:
            options = []

        self.integral_kernel = integral_kernel

        self.n_tables = table_data_shapes["n_tables"]
        self.n_q_points = table_data_shapes["n_q_points"]
        self.n_table_entries = table_data_shapes["n_table_entries"]
        self.n_cases = table_data_shapes.get("n_cases", 0)
        self.reconstruction_kind = table_data_shapes.get("reconstruction_kind", "dense")
        self.n_reconstruction_transforms = table_data_shapes.get(
            "n_reconstruction_transforms", 0
        )
        self.n_reconstruction_lookup_entries = table_data_shapes.get(
            "n_reconstruction_lookup_entries", 0
        )
        self.n_reconstruction_lookup_probes = table_data_shapes.get(
            "n_reconstruction_lookup_probes", 0
        )
        self.n_reconstruction_sign_lookup_entries = table_data_shapes.get(
            "n_reconstruction_sign_lookup_entries", 0
        )
        self.n_reconstruction_sign_lookup_probes = table_data_shapes.get(
            "n_reconstruction_sign_lookup_probes", 0
        )
        self.n_arithmetic_case_orbits = table_data_shapes.get(
            "n_arithmetic_case_orbits", 0
        )
        self.n_pde_boundary_target_points = table_data_shapes.get(
            "n_pde_boundary_target_points", 0
        )
        self.n_pde_interior_target_points = table_data_shapes.get(
            "n_pde_interior_target_points", 0
        )
        self.potential_kind = potential_kind

        assert np.isreal(self.n_tables)
        assert np.isreal(self.n_q_points)
        assert np.isreal(self.n_table_entries)
        assert np.isreal(self.n_cases)
        assert self.reconstruction_kind in {
            "dense",
            "generated-orbit",
            "scalar-arithmetic-orbit",
            "signed-arithmetic-orbit",
            "pde-boundary-shell",
        }

        self.options = options
        self.name = name or self.default_name
        self.device = device
        self.extra_kwargs = kwargs
        self.kname = self.integral_kernel.__repr__()
        self.dim = self.integral_kernel.dim
        self.quad_order = int(
            table_data_shapes.get(
                "quad_order",
                round(float(self.n_q_points) ** (1.0 / float(self.dim))),
            )
        )

        if "case_encoding_bias" not in self.extra_kwargs:
            self.extra_kwargs["case_encoding_bias"] = CASE_ENCODING_BIAS_DEFAULT
        self.extra_kwargs["case_encoding_bias"] = float(
            self.extra_kwargs["case_encoding_bias"]
        )

        if self.extra_kwargs.get("case_encoding_warn_tol") is not None:
            self.extra_kwargs["case_encoding_warn_tol"] = float(
                self.extra_kwargs["case_encoding_warn_tol"]
            )

        # Allow user to pass more tables to force using multiple tables
        # instead of performing kernel scaling
        if "infer_kernel_scaling" not in self.extra_kwargs:
            self.extra_kwargs["infer_kernel_scaling"] = (
                self.n_tables == 1 and self._supports_inferred_scaling()
            )

        if (
            self.extra_kwargs.get("infer_kernel_scaling", False)
            and not self._supports_inferred_scaling()
            and "kernel_scaling_code" not in self.extra_kwargs
        ):
            raise RuntimeError(
                "infer_kernel_scaling is unsupported for kernel "
                f"{self.integral_kernel!r}. For Helmholtz, "
                "K_k(h r) = h^{-1} K_{k h}(r): exact table scaling requires "
                "changing the effective wave number with box size, which a "
                "single fixed-k table cannot represent. Use multi-level tables "
                "or provide explicit custom scaling/displacement code."
            )

        # Do not infer scaling rules when user defined rules are present
        if ("kernel_scaling_code" in self.extra_kwargs) or (
            "kernel_displacement_code" in self.extra_kwargs
        ):
            self.extra_kwargs["infer_kernel_scaling"] = False
            # the two codes must be simultaneously given
            assert ("kernel_scaling_code" in self.extra_kwargs) and (
                "kernel_displacement_code" in self.extra_kwargs
            )

        self._single_table_level_check_cache_key = None

    def get_cache_key(self):
        return (
            type(self).__name__,
            self.name,
            self.kname,
            "infer_scaling=" + str(self.extra_kwargs["infer_kernel_scaling"]),
        )


# }}} End near field eval base class

# {{{ eval from CSR data


class NearFieldFromCSR(NearFieldEvalBase):
    """Evaluate the near-field potentials from CSR representation of the tree.
    The class supports auto-scaling of simple kernels.
    """

    default_name = "near_field_from_csr"

    def _base_kernel(self):
        return self.integral_kernel.get_base_kernel()

    def _is_laplace_kernel(self, dim):
        from sumpy.kernel import LaplaceKernel

        return isinstance(self._base_kernel(), LaplaceKernel) and self.dim == dim

    def _is_constant_kernel(self, dim):
        return self.kname in {f"CstKnl{dim}D", f"ConstantKernel{dim}D"} or (
            self._base_kernel().__class__.__name__ == "ConstantKernel"
            and self.dim == dim
        )

    def _is_axis_target_derivative_of_laplace(self, dim):
        from sumpy.kernel import AxisTargetDerivative, LaplaceKernel

        return (
            isinstance(self.integral_kernel, AxisTargetDerivative)
            and isinstance(
                self.integral_kernel.inner_kernel.get_base_kernel(), LaplaceKernel
            )
            and self.dim == dim
        )

    def _is_axis_source_derivative_of_laplace(self, dim):
        from sumpy.kernel import AxisSourceDerivative, LaplaceKernel

        return (
            isinstance(self.integral_kernel, AxisSourceDerivative)
            and isinstance(
                self.integral_kernel.inner_kernel.get_base_kernel(), LaplaceKernel
            )
            and self.dim == dim
        )

    def _supports_inferred_scaling(self):
        return (
            self._is_laplace_kernel(2)
            or self._is_laplace_kernel(3)
            or self._is_constant_kernel(2)
            or self._is_constant_kernel(3)
            or self._is_axis_target_derivative_of_laplace(2)
            or self._is_axis_target_derivative_of_laplace(3)
            or self._is_axis_source_derivative_of_laplace(2)
            or self._is_axis_source_derivative_of_laplace(3)
        )

    def _inferred_scaling_code(self):
        if self._is_axis_target_derivative_of_laplace(2):
            logger.info("scaling for Grad(LapKnl2D)")
            return "BOX_extent / table_root_extent"

        if self._is_axis_target_derivative_of_laplace(3):
            logger.info("scaling for Grad(LapKnl3D)")
            return "BOX_extent / table_root_extent"

        if self._is_axis_source_derivative_of_laplace(2):
            logger.info("scaling for SourceGrad(LapKnl2D)")
            return "BOX_extent / table_root_extent"

        if self._is_axis_source_derivative_of_laplace(3):
            logger.info("scaling for SourceGrad(LapKnl3D)")
            return "BOX_extent / table_root_extent"

        if self._is_laplace_kernel(2):
            logger.info("scaling for LapKnl2D")
            return "BOX_extent * BOX_extent / \
                    (table_root_extent * table_root_extent)"

        if self._is_constant_kernel(2):
            logger.info("scaling for CstKnl2D")
            return "BOX_extent * BOX_extent / \
                    (table_root_extent * table_root_extent)"

        if self._is_laplace_kernel(3):
            logger.info("scaling for Lapknl3D")
            return "BOX_extent * BOX_extent / \
                    (table_root_extent * table_root_extent)"

        if self._is_constant_kernel(3):
            logger.info("scaling for CstKnl3D")
            return "BOX_extent * BOX_extent * BOX_extent / \
                    (table_root_extent * table_root_extent * table_root_extent)"

        raise RuntimeError(f"no inferred scaling rule for {self.integral_kernel!r}")

    def _inferred_displacement_code(self):
        if self._is_axis_target_derivative_of_laplace(2):
            logger.info("no displacement for Grad(LapKnl2D)")
            return "0.0"

        if self._is_axis_target_derivative_of_laplace(3):
            logger.info("no displacement for Grad(LapKnl3D)")
            return "0.0"

        if self._is_axis_source_derivative_of_laplace(2):
            logger.info("no displacement for SourceGrad(LapKnl2D)")
            return "0.0"

        if self._is_axis_source_derivative_of_laplace(3):
            logger.info("no displacement for SourceGrad(LapKnl3D)")
            return "0.0"

        if self._is_laplace_kernel(2):
            logger.info("displacement for laplace 2D")
            s = "-0.5 / PI * scaling * \
                    log(BOX_extent / table_root_extent) * \
                    mode_nmlz[table_lev, sid]"
            import math

            return s.replace("PI", str(math.pi))

        if self._is_constant_kernel(2):
            logger.info("no displacement for CstKnl2D")
            return "0.0"

        if self._is_laplace_kernel(3):
            logger.info("no displacement for LapKnl3D")
            return "0.0"

        if self._is_constant_kernel(3):
            logger.info("no displacement for CstKnl3D")
            return "0.0"

        raise RuntimeError(f"no inferred displacement rule for {self.integral_kernel!r}")

    def _inferred_table_level_code(self):
        logger.info("scaling from table[0] for " + self.kname)
        return "0.0"

    def get_kernel_scaling_policy(self, box_name="sbox"):
        """Return the table-level and scaling policy used by generated List 1 code.

        ``mode == "canonical_single_table"`` means table level 0 is reused for
        all source-box levels with ``scaling_code`` and ``displacement_code``.
        ``mode == "fixed_single_table"`` means table level 0 is used without
        scaling and runtime checks require all source boxes to match the table
        starting level.
        ``mode == "per_level_tables"`` means no kernel scaling is applied and
        the table level is selected from the source-box level.
        """
        if self.extra_kwargs.get("infer_kernel_scaling", False):
            return KernelScalingPolicy(
                kernel_name=self.kname,
                mode="canonical_single_table",
                infer_kernel_scaling=True,
                single_table_scaling_supported=True,
                reference_table_level=0,
                scaling_code=self._inferred_scaling_code().replace("BOX", box_name),
                displacement_code=self._inferred_displacement_code().replace(
                    "BOX", box_name
                ),
                table_level_code=self._inferred_table_level_code().replace(
                    "BOX", box_name
                ),
                notes="Uses table level 0 for all source-box levels with inferred exact scaling.",
            )

        if "kernel_scaling_code" in self.extra_kwargs:
            return KernelScalingPolicy(
                kernel_name=self.kname,
                mode="custom_single_table",
                infer_kernel_scaling=False,
                single_table_scaling_supported=True,
                reference_table_level=0,
                scaling_code=self.extra_kwargs["kernel_scaling_code"].replace(
                    "BOX", box_name
                ),
                displacement_code=self.extra_kwargs["kernel_displacement_code"].replace(
                    "BOX", box_name
                ),
                table_level_code="0.0",
                notes="Uses user-provided scaling and displacement code with table level 0.",
            )

        if self.n_tables == 1:
            return KernelScalingPolicy(
                kernel_name=self.kname,
                mode="fixed_single_table",
                infer_kernel_scaling=False,
                single_table_scaling_supported=False,
                reference_table_level=0,
                scaling_code="1.0",
                displacement_code="0.0",
                table_level_code="0.0",
                notes=(
                    "Uses the only cached table level without kernel scaling; "
                    "runtime checks reject mixed source levels or a table level mismatch."
                ),
            )

        return KernelScalingPolicy(
            kernel_name=self.kname,
            mode="per_level_tables",
            infer_kernel_scaling=False,
            single_table_scaling_supported=False,
            reference_table_level=None,
            scaling_code="1.0",
            displacement_code="0.0",
            table_level_code=(
                "(0 if BOX_level < table_starting_level "
                "else (BOX_level - table_starting_level "
                "if BOX_level - table_starting_level < n_tables "
                "else n_tables - 1))"
            ).replace("BOX", box_name),
            notes="Selects one cached table level per source-box level without kernel scaling.",
        )

    def codegen_vec_component(self, d=None):
        if d is None:
            dimension = self.dim - 1
        else:
            dimension = d
        bias = float(self.extra_kwargs["case_encoding_bias"])
        # Keep encoded case ids away from integer truncation boundaries.
        # Some OpenCL stacks evaluate this expression slightly below the
        # mathematically integral value for non-dyadic box extents.
        return (
            "("
            + "(box_centers["
            + str(dimension)
            + ", target_box_id]"
            + "- box_centers["
            + str(dimension)
            + ", source_box_id]"
            + ") / sbox_extent * 4.0 + encoding_shift"
            + f" + {bias:.17g}"
            + ")"
        )

    def _to_host_array(self, queue, ary):
        if isinstance(ary, np.ndarray):
            return ary
        if hasattr(ary, "get"):
            return ary.get(queue)
        return np.asarray(ary)

    def _warn_case_encoding_drift(
        self,
        queue,
        box_centers,
        box_levels,
        root_extent,
        target_boxes,
        neighbor_source_boxes_starts,
        neighbor_source_boxes_lists,
        encoding_shift,
        warn_tol,
    ):
        if warn_tol is None:
            return

        box_centers_h = self._to_host_array(queue, box_centers)
        box_levels_h = self._to_host_array(queue, box_levels)
        target_boxes_h = self._to_host_array(queue, target_boxes)
        starts_h = self._to_host_array(queue, neighbor_source_boxes_starts)
        lists_h = self._to_host_array(queue, neighbor_source_boxes_lists)

        if len(target_boxes_h) == 0:
            return

        root_extent = float(root_extent)
        encoding_shift = float(encoding_shift)

        max_drift = 0.0
        worst_pair = None
        worst_dim = None
        worst_raw = None

        for i_tbox, target_box_id in enumerate(target_boxes_h):
            target_box_id = int(target_box_id)
            for i_sbox in range(int(starts_h[i_tbox]), int(starts_h[i_tbox + 1])):
                source_box_id = int(lists_h[i_sbox])
                source_box_level = int(box_levels_h[source_box_id])
                source_box_extent = root_extent * (0.5**source_box_level)

                for d in range(self.dim):
                    raw_component = (
                        box_centers_h[d, target_box_id]
                        - box_centers_h[d, source_box_id]
                    ) / source_box_extent * 4.0 + encoding_shift
                    drift = abs(raw_component - np.rint(raw_component))

                    if drift > max_drift:
                        max_drift = float(drift)
                        worst_pair = (target_box_id, source_box_id)
                        worst_dim = d
                        worst_raw = float(raw_component)

        if max_drift > warn_tol and worst_pair is not None:
            logger.warning(
                "List1 case encoding drift %.3e exceeds %.3e for "
                "target box %d, source box %d, dim %d (raw %.17g)",
                max_drift,
                warn_tol,
                worst_pair[0],
                worst_pair[1],
                worst_dim,
                worst_raw,
            )

    def codegen_vec_id(self):
        dim = self.dim
        code = "0.0"
        for d in range(dim):
            code = "(" + code + ") * encoding_base"
            code = code + "+" + self.codegen_vec_component(d)
        return code

    def codegen_compute_scaling(self, box_name="sbox"):
        """box_name: the name of the box whose extent is used."""
        if "kernel_scaling_code" in self.extra_kwargs:
            # user-defined scaling rule
            assert isinstance(self.extra_kwargs["kernel_scaling_code"], str)
            logger.info(
                "Using scaling rule %s for %s.",
                self.extra_kwargs["kernel_scaling_code"],
                self.kname,
            )
            return self.get_kernel_scaling_policy(box_name=box_name).scaling_code

        if not self.extra_kwargs.get("infer_kernel_scaling", False):
            logger.info("not scaling for " + self.kname)
            logger.info("(using multiple tables)")

        return self.get_kernel_scaling_policy(box_name=box_name).scaling_code

    def codegen_compute_displacement(self, box_name="sbox"):
        if "kernel_displacement_code" in self.extra_kwargs:
            # user-defined displacement rule
            assert isinstance(self.extra_kwargs["kernel_displacement_code"], str)
            logger.info(
                "Using displacement %s for %s.",
                self.extra_kwargs["kernel_displacement_code"],
                self.kname,
            )
            return self.get_kernel_scaling_policy(box_name=box_name).displacement_code

        if not self.extra_kwargs.get("infer_kernel_scaling", False):
            logger.info("no displacement for " + self.kname)
            logger.info("(using multiple tables)")

        return self.get_kernel_scaling_policy(box_name=box_name).displacement_code

    def codegen_get_table_level(self, box_name="sbox"):
        if "kernel_scaling_code" in self.extra_kwargs:
            # Using custom scaling
            return self.get_kernel_scaling_policy(box_name=box_name).table_level_code

        if not self.extra_kwargs.get("infer_kernel_scaling", False):
            logger.info("computing table level from box size")
            logger.info("(using multiple tables)")

        return self.get_kernel_scaling_policy(box_name=box_name).table_level_code

    def codegen_exterior_part(self):
        """Computes the exterior contribution. This is nonzero for
        inverse-type potentials like the fractional Laplacian.
        """
        if self.potential_kind == 1:
            return "0.0"
        elif self.potential_kind == 2:
            return "source_coefs[target_id] * ext_nmlz"
        else:
            raise ValueError("Unsupported potential kind %d" % self.potential_kind)

    def get_kernel(self):

        if self.integral_kernel.is_complex_valued:
            potential_dtype = np.complex128
        else:
            potential_dtype = np.float64

        if self.reconstruction_kind in {
            "scalar-arithmetic-orbit",
            "signed-arithmetic-orbit",
        }:
            reconstruction_domains = []

            def select_axis(prefix, perm_name):
                if self.dim == 2:
                    return (
                        f"({prefix}_raw_0 if {perm_name} == 0 "
                        f"else {prefix}_raw_1)"
                    )
                if self.dim == 3:
                    return (
                        f"({prefix}_raw_0 if {perm_name} == 0 "
                        f"else ({prefix}_raw_1 if {perm_name} == 1 "
                        f"else {prefix}_raw_2))"
                    )
                raise NotImplementedError("arithmetic ORBIT supports dim 2 or 3")

            if self.dim == 2:
                decode_code = """
                        <> src_raw_0 = source_mode_id // quad_order
                        <> src_raw_1 = source_mode_id % quad_order
                        <> tgt_raw_0 = target_point_id // quad_order
                        <> tgt_raw_1 = target_point_id % quad_order
                """
                sort_code = """
                        <> swap01 = (1 if group0 == group1
                            and (src1 < src0 or (src1 == src0 and tgt1 < tgt0))
                            else 0)
                        <> src0_s = (src1 if swap01 else src0)
                        <> tgt0_s = (tgt1 if swap01 else tgt0)
                        <> src1_s = (src0 if swap01 else src1)
                        <> tgt1_s = (tgt0 if swap01 else tgt1)
                """
            elif self.dim == 3:
                decode_code = """
                        <> src_raw_0 = source_mode_id // (quad_order * quad_order)
                        <> src_raw_1 = (source_mode_id // quad_order) % quad_order
                        <> src_raw_2 = source_mode_id % quad_order
                        <> tgt_raw_0 = target_point_id // (quad_order * quad_order)
                        <> tgt_raw_1 = (target_point_id // quad_order) % quad_order
                        <> tgt_raw_2 = target_point_id % quad_order
                """
                sort_code = """
                        <> swap01_a = (1 if group0 == group1
                            and (src1 < src0 or (src1 == src0 and tgt1 < tgt0))
                            else 0)
                        <> src0_a = (src1 if swap01_a else src0)
                        <> tgt0_a = (tgt1 if swap01_a else tgt0)
                        <> src1_a = (src0 if swap01_a else src1)
                        <> tgt1_a = (tgt0 if swap01_a else tgt1)
                        <> src2_a = src2
                        <> tgt2_a = tgt2

                        <> swap12_b = (1 if group1 == group2
                            and (src2_a < src1_a
                                or (src2_a == src1_a and tgt2_a < tgt1_a))
                            else 0)
                        <> src0_b = src0_a
                        <> tgt0_b = tgt0_a
                        <> src1_b = (src2_a if swap12_b else src1_a)
                        <> tgt1_b = (tgt2_a if swap12_b else tgt1_a)
                        <> src2_b = (src1_a if swap12_b else src2_a)
                        <> tgt2_b = (tgt1_a if swap12_b else tgt2_a)

                        <> swap01_c = (1 if group0 == group1
                            and (src1_b < src0_b
                                or (src1_b == src0_b and tgt1_b < tgt0_b))
                            else 0)
                        <> src0_c = (src1_b if swap01_c else src0_b)
                        <> tgt0_c = (tgt1_b if swap01_c else tgt0_b)
                        <> src1_c = (src0_b if swap01_c else src1_b)
                        <> tgt1_c = (tgt0_b if swap01_c else tgt1_b)
                        <> src2_c = src2_b
                        <> tgt2_c = tgt2_b

                        <> swap02_d = (1 if group0 == group2
                            and (src2_c < src0_c
                                or (src2_c == src0_c and tgt2_c < tgt0_c))
                            else 0)
                        <> src0_s = (src2_c if swap02_d else src0_c)
                        <> tgt0_s = (tgt2_c if swap02_d else tgt0_c)
                        <> src1_s = src1_c
                        <> tgt1_s = tgt1_c
                        <> src2_s = (src0_c if swap02_d else src2_c)
                        <> tgt2_s = (tgt0_c if swap02_d else tgt2_c)
                """
            else:
                raise NotImplementedError("arithmetic ORBIT supports dim 2 or 3")

            pair_code = ""
            for iaxis in range(self.dim):
                pair_code += f"""
                        <> pair{iaxis} = src{iaxis}_s * quad_order + tgt{iaxis}_s
                """
            pair_code += """
                        <> pair_count = quad_order * quad_order
                        <> folded_pair_count = (pair_count + 1) // 2
            """

            def min_expr(values):
                result = values[0]
                for value in values[1:]:
                    result = f"({result} if {result} <= {value} else {value})"
                return result

            def max_expr(values):
                result = values[0]
                for value in values[1:]:
                    result = f"({result} if {result} >= {value} else {value})"
                return result

            rank_code = ""
            for group_id in range(self.dim):
                size_terms = [
                    f"(1 if group{iaxis} == {group_id} else 0)"
                    for iaxis in range(self.dim)
                ]
                zero_terms = [
                    f"(1 if group{iaxis} == {group_id} "
                    f"and axis_sign{iaxis} == 0 else 0)"
                    for iaxis in range(self.dim)
                ]
                min_terms = [
                    f"(pair{iaxis} if group{iaxis} == {group_id} else pair_count)"
                    for iaxis in range(self.dim)
                ]
                max_terms = [
                    f"(pair{iaxis} if group{iaxis} == {group_id} else -1)"
                    for iaxis in range(self.dim)
                ]
                sum_terms = [
                    f"(pair{iaxis} if group{iaxis} == {group_id} else 0)"
                    for iaxis in range(self.dim)
                ]
                rank_code += f"""
                        <> group{group_id}_size = {' + '.join(size_terms)}
                        <> group{group_id}_zero_count = {' + '.join(zero_terms)}
                        <> group{group_id}_alphabet = (folded_pair_count
                            if group{group_id}_zero_count > 0 else pair_count)
                        <> group{group_id}_lo = {min_expr(min_terms)}
                        <> group{group_id}_hi = {max_expr(max_terms)}
                        <> group{group_id}_sum = {' + '.join(sum_terms)}
                        <> group{group_id}_mid = group{group_id}_sum \
                                - group{group_id}_lo - group{group_id}_hi
                        <> group{group_id}_value_count = (
                            1 if group{group_id}_size == 0 else (
                            group{group_id}_alphabet
                            if group{group_id}_size == 1 else (
                            (group{group_id}_alphabet
                            * (group{group_id}_alphabet + 1)) // 2
                            if group{group_id}_size == 2 else
                            (group{group_id}_alphabet
                            * (group{group_id}_alphabet + 1)
                            * (group{group_id}_alphabet + 2)) // 6)))
                        <> group{group_id}_rank = (
                            0 if group{group_id}_size == 0 else (
                            group{group_id}_lo
                            if group{group_id}_size == 1 else (
                            (group{group_id}_alphabet
                            * (group{group_id}_alphabet + 1)) // 2
                            - ((group{group_id}_alphabet - group{group_id}_lo)
                            * (group{group_id}_alphabet - group{group_id}_lo + 1)
                            ) // 2
                            + (group{group_id}_hi - group{group_id}_lo)
                            if group{group_id}_size == 2 else
                            (group{group_id}_alphabet
                            * (group{group_id}_alphabet + 1)
                            * (group{group_id}_alphabet + 2)) // 6
                            - ((group{group_id}_alphabet - group{group_id}_lo)
                            * (group{group_id}_alphabet - group{group_id}_lo + 1)
                            * (group{group_id}_alphabet - group{group_id}_lo + 2)
                            ) // 6
                            + ((group{group_id}_alphabet - group{group_id}_lo)
                            * (group{group_id}_alphabet - group{group_id}_lo + 1)
                            ) // 2
                            - ((group{group_id}_alphabet - group{group_id}_mid)
                            * (group{group_id}_alphabet - group{group_id}_mid + 1)
                            ) // 2
                            + (group{group_id}_hi - group{group_id}_mid))))
                """

            if self.dim == 2:
                compact_pair_rank = (
                    "group0_rank * group1_value_count + group1_rank"
                )
            elif self.dim == 3:
                compact_pair_rank = (
                    "(group0_rank * group1_value_count + group1_rank) "
                    "* group2_value_count + group2_rank"
                )
            else:
                raise NotImplementedError("arithmetic ORBIT supports dim 2 or 3")

            axis_code = ""
            for iaxis in range(self.dim):
                perm_name = f"perm{iaxis}"
                sign_name = f"axis_sign{iaxis}"
                axis_code += f"""
                        <> {perm_name} = arithmetic_case_axis_perm[case_id, {iaxis}]
                        <> {sign_name} = arithmetic_case_axis_sign[case_id, {iaxis}]
                        <> group{iaxis} = arithmetic_case_axis_group[case_id, {iaxis}]
                        <> sign_power{iaxis} = arithmetic_axis_sign_power[{perm_name}]
                        <> src{iaxis}_raw_perm = {select_axis("src", perm_name)}
                        <> tgt{iaxis}_raw_perm = {select_axis("tgt", perm_name)}
                        <> src{iaxis}_fixed = (quad_order - 1 - src{iaxis}_raw_perm
                            if {sign_name} < 0 else src{iaxis}_raw_perm)
                        <> tgt{iaxis}_fixed = (quad_order - 1 - tgt{iaxis}_raw_perm
                            if {sign_name} < 0 else tgt{iaxis}_raw_perm)
                        <> src{iaxis}_flip = quad_order - 1 - src{iaxis}_raw_perm
                        <> tgt{iaxis}_flip = quad_order - 1 - tgt{iaxis}_raw_perm
                        <> zero_flip{iaxis} = (1 if {sign_name} == 0
                            and (src{iaxis}_flip < src{iaxis}_raw_perm
                                or (src{iaxis}_flip == src{iaxis}_raw_perm
                                    and tgt{iaxis}_flip < tgt{iaxis}_raw_perm))
                            else 0)
                        <> src{iaxis} = (src{iaxis}_flip
                            if zero_flip{iaxis} else src{iaxis}_fixed)
                        <> tgt{iaxis} = (tgt{iaxis}_flip
                            if zero_flip{iaxis} else tgt{iaxis}_fixed)
                        <> transform_sign{iaxis} = (-1
                            if {sign_name} < 0 or zero_flip{iaxis} else 1)
                        <> direction_sign_factor{iaxis} = (
                            transform_sign{iaxis}
                            * arithmetic_axis_direction_signs[{perm_name}]
                            * arithmetic_axis_direction_signs[{iaxis}]
                            if arithmetic_direction_sign_axis == {iaxis}
                            else 1)
                        <> entry_sign_factor{iaxis} = (
                            (transform_sign{iaxis}
                                if sign_power{iaxis} != 0 else 1)
                            * direction_sign_factor{iaxis})
                """

            reconstruction_code = f"""
                        {decode_code}
                        {axis_code}
                        {sort_code}
                        {pair_code}
                        {rank_code}
                        <> arithmetic_case_rank = arithmetic_case_orbit_ranks[case_id]
                        <> compact_pair_rank = {compact_pair_rank}
                        <> entry_id = arithmetic_case_value_offsets[
                                arithmetic_case_rank] + compact_pair_rank
                        <> entry_sign = {" * ".join(
                            f"entry_sign_factor{iaxis}" for iaxis in range(self.dim)
                        )}
                        <> has_entry = 1
            """
            reconstruction_args = [
                loopy.GlobalArg(
                    "arithmetic_case_orbit_ranks",
                    np.uint16,
                    "n_cases",
                    is_input=True,
                ),
                loopy.GlobalArg(
                    "arithmetic_case_axis_perm",
                    np.uint8,
                    "n_cases, dim",
                    is_input=True,
                ),
                loopy.GlobalArg(
                    "arithmetic_case_axis_sign",
                    np.int8,
                    "n_cases, dim",
                    is_input=True,
                ),
                loopy.GlobalArg(
                    "arithmetic_case_axis_group",
                    np.uint8,
                    "n_cases, dim",
                    is_input=True,
                ),
                loopy.GlobalArg(
                    "arithmetic_case_value_offsets",
                    np.int32,
                    "n_arithmetic_case_orbits + 1",
                    is_input=True,
                ),
                loopy.GlobalArg(
                    "arithmetic_axis_sign_power",
                    np.uint8,
                    "dim",
                    is_input=True,
                ),
                loopy.GlobalArg(
                    "arithmetic_axis_direction_signs",
                    np.int8,
                    "dim",
                    is_input=True,
                ),
            ]
            reconstruction_value_arg_names = (
                ", quad_order, n_arithmetic_case_orbits, "
                "arithmetic_direction_sign_axis"
            )
        elif self.reconstruction_kind == "generated-orbit":
            reconstruction_domains = [
                "{ [ tr ] : 0 <= tr < n_reconstruction_transforms }",
                "{ [ probe ] : 0 <= probe < n_reconstruction_lookup_probes }",
                "{ [ sign_probe ] : "
                "0 <= sign_probe < n_reconstruction_sign_lookup_probes }",
                "{ [ sign_value_probe ] : "
                "0 <= sign_value_probe < n_reconstruction_sign_lookup_probes }",
            ]
            reconstruction_code = """
                        <> reconstruction_full_entry_id = \
                                case_id * (n_q_points * n_q_points) \
                                + source_mode_id * n_q_points + target_point_id
                        <> best_reconstruction_key = min((tr),
                            (
                                reconstruction_case_map[tr, case_id]
                                * (n_q_points * n_q_points)
                                + reconstruction_qpoint_map[tr, source_mode_id]
                                * n_q_points
                                + reconstruction_qpoint_map[tr, target_point_id]
                            ) * (2 * n_reconstruction_transforms)
                            + (0 if reconstruction_signs[tr] > 0 else 1)
                            * n_reconstruction_transforms
                            + tr)
                        <> best_transform_id = \
                                best_reconstruction_key % n_reconstruction_transforms
                        <> representative_entry_id = \
                                best_reconstruction_key \
                                // (2 * n_reconstruction_transforms)
                        <> lookup_start = (representative_entry_id * 33) \
                                % n_reconstruction_lookup_entries
                        <> lookup_match_count = sum((probe),
                            1
                            if reconstruction_lookup_keys[
                                (lookup_start + probe)
                                % n_reconstruction_lookup_entries]
                                == representative_entry_id
                            else 0)
                        <> entry_id = sum((probe),
                            reconstruction_lookup_values[
                                (lookup_start + probe)
                                % n_reconstruction_lookup_entries]
                            if reconstruction_lookup_keys[
                                (lookup_start + probe)
                                % n_reconstruction_lookup_entries]
                                == representative_entry_id
                            else 0)
                        <> sign_lookup_start = (reconstruction_full_entry_id * 33) \
                                % n_reconstruction_sign_lookup_entries
                        <> sign_correction_match_count = sum((sign_probe),
                            1
                            if reconstruction_sign_lookup_keys[
                                (sign_lookup_start + sign_probe)
                                % n_reconstruction_sign_lookup_entries]
                                == reconstruction_full_entry_id
                            else 0)
                        <> sign_correction = (
                            sum((sign_value_probe),
                                reconstruction_sign_lookup_values[
                                    (sign_lookup_start + sign_value_probe)
                                    % n_reconstruction_sign_lookup_entries]
                                if reconstruction_sign_lookup_keys[
                                    (sign_lookup_start + sign_value_probe)
                                    % n_reconstruction_sign_lookup_entries]
                                    == reconstruction_full_entry_id
                                else 0)
                            if sign_correction_match_count > 0
                            else 1)
                        <> entry_sign = reconstruction_signs[best_transform_id] \
                                * sign_correction
                        <> has_entry = lookup_match_count > 0
            """
            reconstruction_args = [
                loopy.GlobalArg(
                    "reconstruction_qpoint_map",
                    np.int32,
                    "n_reconstruction_transforms, n_q_points",
                ),
                loopy.GlobalArg(
                    "reconstruction_case_map",
                    np.int32,
                    "n_reconstruction_transforms, n_cases",
                ),
                loopy.GlobalArg(
                    "reconstruction_signs",
                    np.int8,
                    "n_reconstruction_transforms",
                ),
                loopy.GlobalArg(
                    "reconstruction_lookup_keys",
                    np.int32,
                    "n_reconstruction_lookup_entries",
                ),
                loopy.GlobalArg(
                    "reconstruction_lookup_values",
                    np.int32,
                    "n_reconstruction_lookup_entries",
                ),
                loopy.GlobalArg(
                    "reconstruction_sign_lookup_keys",
                    np.int32,
                    "n_reconstruction_sign_lookup_entries",
                ),
                loopy.GlobalArg(
                    "reconstruction_sign_lookup_values",
                    np.int8,
                    "n_reconstruction_sign_lookup_entries",
                ),
            ]
            reconstruction_value_arg_names = (
                ", n_reconstruction_transforms, "
                "n_reconstruction_lookup_entries, n_reconstruction_lookup_probes, "
                "n_reconstruction_sign_lookup_entries, "
                "n_reconstruction_sign_lookup_probes"
            )
        elif self.reconstruction_kind == "pde-boundary-shell":
            reconstruction_domains = []
            reconstruction_code = """
                        <> target_boundary_point_id = \
                                pde_target_boundary_ids[target_point_id]
                        <> entry_id = case_id \
                                * (n_q_points * n_pde_boundary_target_points) \
                                + source_mode_id * n_pde_boundary_target_points \
                                + target_boundary_point_id
                        <> entry_sign = 1
                        <> has_entry = target_boundary_point_id >= 0
            """
            reconstruction_args = [
                loopy.GlobalArg(
                    "pde_target_boundary_ids", np.int32, "n_q_points"
                ),
            ]
            reconstruction_value_arg_names = ", n_pde_boundary_target_points"
        else:
            reconstruction_domains = []
            reconstruction_code = """
                        <> source_mode_id_sym = mode_qpoint_map[source_mode_id, source_mode_id]
                        <> target_point_id_sym = mode_qpoint_map[source_mode_id, target_point_id]
                        <> case_id_sym = mode_case_map[source_mode_id, case_id]
                        <> mode_map_sign = mode_case_scale[source_mode_id, case_id]
                        <> pair_id = source_mode_id_sym * n_q_points + target_point_id_sym
                        <> entry_id_full = case_id_sym * (n_q_points * n_q_points) + pair_id
                        <> entry_id = table_entry_ids[entry_id_full]
                        <> entry_sign = table_entry_scales[entry_id_full] * mode_map_sign
                        <> has_entry = entry_id >= 0
            """
            reconstruction_args = [
                loopy.GlobalArg(
                    "table_entry_ids", np.int32, "n_cases*n_q_points*n_q_points"
                ),
                loopy.GlobalArg(
                    "table_entry_scales",
                    potential_dtype,
                    "n_cases*n_q_points*n_q_points",
                ),
                loopy.GlobalArg("mode_qpoint_map", np.int32, "n_q_points, n_q_points"),
                loopy.GlobalArg("mode_case_map", np.int32, "n_q_points, n_cases"),
                loopy.GlobalArg(
                    "mode_case_scale",
                    potential_dtype,
                    "n_q_points, n_cases",
                ),
            ]
            reconstruction_value_arg_names = ""

        kernel_domains = [
            "{ [ tbox ] : 0 <= tbox < n_tgt_boxes }",
            "{ [ tid ] : 0 <= tid < n_q_points }",
            "{ [ sbox ] : sbox_begin <= sbox < sbox_end }",
            "{ [ sid ] : 0 <= sid < n_box_sources }",
        ] + reconstruction_domains

        lpknl = loopy.make_kernel(
            kernel_domains,
            """
            for tbox
                <> target_box_id    = target_boxes[tbox]
                <> box_target_beg   = box_target_starts[target_box_id]
                <> n_box_targets    = box_target_counts_nonchild[target_box_id]

                <> sbox_begin = neighbor_source_boxes_starts[tbox]
                <> sbox_end   = neighbor_source_boxes_starts[tbox+1]

                <> tbox_level  = box_levels[target_box_id]
                <> tbox_extent = root_extent * (1.0 / (2**tbox_level))

                for tid
                    if tid < n_box_targets
                    <> target_id = box_target_beg + tid
                    end
                end

                for tid, sbox
                    if tid < n_box_targets
                    <> source_box_id  = source_boxes[sbox]
                    <> n_box_sources  = box_source_counts_nonchild[source_box_id]
                    <> box_source_beg = box_source_starts[source_box_id]

                    <> sbox_level  = box_levels[source_box_id]
                    <> sbox_extent = root_extent * (1.0 / (2**sbox_level))

                    table_lev_tmp = GET_TABLE_LEVEL {id=tab_lev_tmp}
                    table_lev = table_lev_tmp + 0.5 {id=tab_lev,dep=tab_lev_tmp}

                    vec_id_tmp = COMPUTE_VEC_ID {id=vec_id_tmp}
                    vec_id = vec_id_tmp + 0.5 {id=vec_id,dep=vec_id_tmp}
                    <> case_id = case_indices[vec_id] {dep=vec_id}

                    <> scaling = COMPUTE_SCALING

                    for sid

                        <> tgt_scaling = COMPUTE_TGT_SCALING
                        <> tgt_displacement = COMPUTE_TGT_DISPLACEMENT
                        tgt_table_lev_tmp = GET_TGT_TABLE_LEVEL {id=tgttab_lev_tmp}
                        tgt_table_lev = tgt_table_lev_tmp + 0.5 \
                                {id=tgttab_lev,dep=tgttab_lev_tmp}
                        <> target_point_id = target_point_ids[target_id]
                        <> ext_nmlz = exterior_mode_nmlz[tgt_table_lev, target_point_id] \
                                * tgt_scaling + tgt_displacement \
                                {id=extnmlz,dep=tgttab_lev}

                        <> source_id = box_source_beg + sid
                        <> source_mode_id = source_mode_ids[source_id]
                        RECONSTRUCT_ENTRY

                        <> displacement = COMPUTE_DISPLACEMENT

                        <> integ = (
                                table_data[table_lev, entry_id] * entry_sign * scaling
                                + displacement
                                if has_entry
                                else 0) {id=integ,dep=tab_lev}
                        # <> source_id_tree = user_source_ids[source_id]
                        <> coef = source_coefs[source_id] {id=coef}

                        # <> target_id_user = sorted_target_ids[target_id]

                        #db_table_lev[target_id] = table_lev_tmp {dep=tab_lev}
                        #db_case_id[target_id] = case_id
                        #db_vec_id[target_id] = vec_id
                        #db_n_box_targets[target_id] = n_box_targets
                        #db_n_box_sources[target_id] = n_box_sources
                        #db_entry_id[target_id] = entry_id

                    end
                    end
                end

                for tid
                    if tid < n_box_targets

                    result[target_id] = sum((sbox, sid),
                        coef * integ) + EXTERIOR_PART \
                            {id=write_result,dep=integ:coef:extnmlz}

                    # Try inspecting case_id if something goes wrong
                    # (like segmentation fault) and look for -1's
                    # result[target_id] = min((sbox, sid), case_id)
                    # result[target_id] = vec_id_tmp

                    end
                end
            end
            """.replace("COMPUTE_VEC_ID", self.codegen_vec_id())
            .replace("RECONSTRUCT_ENTRY", reconstruction_code)
            .replace("COMPUTE_SCALING", self.codegen_compute_scaling())
            .replace("COMPUTE_DISPLACEMENT", self.codegen_compute_displacement())
            .replace("COMPUTE_TGT_SCALING", self.codegen_compute_scaling("tbox"))
            .replace(
                "COMPUTE_TGT_DISPLACEMENT", self.codegen_compute_displacement("tbox")
            )
            .replace("GET_TABLE_LEVEL", self.codegen_get_table_level())
            .replace("GET_TGT_TABLE_LEVEL", self.codegen_get_table_level("tbox"))
            .replace("EXTERIOR_PART", self.codegen_exterior_part()),
            [
                loopy.TemporaryVariable("vec_id", np.int32),
                loopy.TemporaryVariable("vec_id_tmp", np.float64),
                loopy.TemporaryVariable("table_lev", np.int32),
                loopy.TemporaryVariable("table_lev_tmp", np.float64),
                loopy.TemporaryVariable("tgt_table_lev", np.int32),
                loopy.TemporaryVariable("tgt_table_lev_tmp", np.float64),
                loopy.ValueArg("encoding_base", np.int32),
                loopy.GlobalArg("mode_nmlz", potential_dtype, "n_tables, n_q_points"),
                loopy.GlobalArg(
                    "exterior_mode_nmlz", potential_dtype, "n_tables, n_q_points"
                ),
                loopy.GlobalArg(
                    "table_data", potential_dtype, "n_tables, n_table_entries"
                ),
                loopy.GlobalArg("source_boxes", np.int32, "n_source_boxes"),
                loopy.GlobalArg("box_centers", None, "dim, aligned_nboxes"),
                loopy.GlobalArg(
                    "box_source_counts_nonchild", np.int32, "aligned_nboxes"
                ),
                loopy.GlobalArg(
                    "box_target_counts_nonchild", np.int32, "aligned_nboxes"
                ),
                loopy.GlobalArg("source_mode_ids", np.int32, "n_source_particles"),
                loopy.GlobalArg("target_point_ids", np.int32, "n_target_particles"),
                loopy.ValueArg("aligned_nboxes", np.int32),
                loopy.ValueArg("table_root_extent", np.float64),
                loopy.ValueArg("table_starting_level", np.int32),
                loopy.ValueArg(
                    "dim, n_source_boxes, n_tables, n_q_points, n_cases, "
                    "n_table_entries, n_source_particles, n_target_particles"
                    + reconstruction_value_arg_names,
                    np.int32,
                ),
                "...",
            ]
            + reconstruction_args,
            name="near_field",
            lang_version=(2018, 2),
            silenced_warnings=("write_race(write_result)",),
        )

        # lpknl = loopy.set_options(lpknl, write_code=True)
        lpknl = loopy.set_options(lpknl, return_dict=True)

        return lpknl

    @memoize_method
    def get_pde_recovery_kernel(self):
        if self.integral_kernel.is_complex_valued:
            potential_dtype = np.complex128
        else:
            potential_dtype = np.float64

        lpknl = loopy.make_kernel(
            [
                "{ [ tbox ] : 0 <= tbox < n_tgt_boxes }",
                "{ [ tid ] : 0 <= tid < n_q_points }",
                "{ [ bid ] : 0 <= bid < n_pde_boundary_target_points }",
                "{ [ sid ] : 0 <= sid < n_q_points }",
            ],
            """
            for tbox
                <> target_box_id = target_boxes[tbox]
                <> box_target_beg = box_target_starts[target_box_id]
                <> n_box_targets = box_target_counts_nonchild[target_box_id]

                <> tbox_level = box_levels[target_box_id]
                <> tbox_extent = root_extent * (1.0 / (2**tbox_level))

                table_lev_tmp = GET_TABLE_LEVEL {id=tab_lev_tmp}
                table_lev = table_lev_tmp + 0.5 {id=tab_lev,dep=tab_lev_tmp}
                <> scaling = COMPUTE_SCALING

                for tid
                    if tid < n_box_targets
                    <> target_id = box_target_beg + tid
                    <> target_point_id = target_point_ids[target_id]
                    <> interior_id = pde_target_interior_ids[target_point_id]
                    result[target_id] = (
                        sum((bid),
                            pde_recovery_matrix[interior_id, bid]
                            * shell_result[pde_box_local_target_ids[
                                target_box_id * n_q_points
                                + pde_boundary_target_point_ids[bid]]]) \
                        + sum((sid),
                                pde_self_correction[
                                    table_lev, interior_id, sid]
                                * source_coefs[pde_box_local_target_ids[
                                    target_box_id * n_q_points + sid]]
                                * scaling)
                        if interior_id >= 0 else shell_result[target_id]) \
                        {id=write_recovered,dep=tab_lev}
                    end
                end
            end
            """
            .replace("COMPUTE_SCALING", self.codegen_compute_scaling("tbox"))
            .replace("GET_TABLE_LEVEL", self.codegen_get_table_level("tbox")),
            [
                loopy.TemporaryVariable("table_lev", np.int32),
                loopy.TemporaryVariable("table_lev_tmp", np.float64),
                loopy.GlobalArg("result", potential_dtype, "n_target_particles"),
                loopy.GlobalArg(
                    "shell_result", potential_dtype, "n_target_particles"
                ),
                loopy.GlobalArg("source_coefs", potential_dtype, "n_source_particles"),
                loopy.GlobalArg("target_boxes", np.int32, "n_tgt_boxes"),
                loopy.GlobalArg("box_centers", None, "dim, aligned_nboxes"),
                loopy.GlobalArg("box_levels", None, "nboxes"),
                loopy.GlobalArg(
                    "box_target_counts_nonchild", np.int32, "aligned_nboxes"
                ),
                loopy.GlobalArg("box_target_starts", np.int32, "nboxes"),
                loopy.GlobalArg("target_point_ids", np.int32, "n_target_particles"),
                loopy.GlobalArg(
                    "pde_target_interior_ids", np.int32, "n_q_points"
                ),
                loopy.GlobalArg(
                    "pde_boundary_target_point_ids",
                    np.int32,
                    "n_pde_boundary_target_points",
                ),
                loopy.GlobalArg(
                    "pde_box_local_target_ids",
                    np.int32,
                    "aligned_nboxes*n_q_points",
                ),
                loopy.GlobalArg(
                    "pde_recovery_matrix",
                    potential_dtype,
                    "n_pde_interior_target_points, n_pde_boundary_target_points",
                ),
                loopy.GlobalArg(
                    "pde_self_correction",
                    potential_dtype,
                    "n_tables, n_pde_interior_target_points, n_q_points",
                ),
                loopy.ValueArg("aligned_nboxes", np.int32),
                loopy.ValueArg("root_extent", np.float64),
                loopy.ValueArg("table_root_extent", np.float64),
                loopy.ValueArg("table_starting_level", np.int32),
                loopy.ValueArg(
                    "dim, nboxes, n_tables, n_q_points, n_tgt_boxes, "
                    "n_target_particles, n_source_particles, "
                    "n_pde_boundary_target_points, n_pde_interior_target_points",
                    np.int32,
                ),
                "...",
            ],
            name="near_field_pde_recovery",
            lang_version=(2018, 2),
            silenced_warnings=("write_race(write_recovered)",),
        )
        lpknl = loopy.set_options(lpknl, return_dict=True)
        lpknl = loopy.tag_inames(lpknl, {"tbox": "g.0"})
        lpknl = loopy.add_inames_for_unused_hw_axes(lpknl)
        return lpknl

    def get_cache_key(self):
        return (
            type(self).__name__,
            "kernel-v14",
            self.name,
            self.kname,
            "complex_kernel=" + str(self.integral_kernel.is_complex_valued),
            "potential_kind=%d" % self.potential_kind,
            "reconstruction_kind=" + self.reconstruction_kind,
            "n_reconstruction_transforms=%d" % self.n_reconstruction_transforms,
            "n_reconstruction_lookup_entries=%d"
            % self.n_reconstruction_lookup_entries,
            "n_reconstruction_lookup_probes=%d"
            % self.n_reconstruction_lookup_probes,
            "n_reconstruction_sign_lookup_entries=%d"
            % self.n_reconstruction_sign_lookup_entries,
            "n_reconstruction_sign_lookup_probes=%d"
            % self.n_reconstruction_sign_lookup_probes,
            "n_arithmetic_case_orbits=%d" % self.n_arithmetic_case_orbits,
            "n_pde_boundary_target_points=%d" % self.n_pde_boundary_target_points,
            "n_pde_interior_target_points=%d" % self.n_pde_interior_target_points,
            "list1_target_batch_size=%d" % self._target_batch_size(),
            "infer_scaling=" + str(self.extra_kwargs["infer_kernel_scaling"]),
            "case_encoding_bias=" + repr(self.extra_kwargs["case_encoding_bias"]),
            "scaling_policy=" + self.codegen_compute_scaling(),
            "displacement_policy=" + self.codegen_compute_displacement(),
            "table_level_policy=" + self.codegen_get_table_level(),
        )

    def _target_batch_size(self):
        target_batch_size = int(self.extra_kwargs.get("list1_target_batch_size", 1))
        if target_batch_size < 1:
            raise ValueError("list1_target_batch_size must be positive")
        return target_batch_size

    def get_optimized_kernel(self, ncpus=None):
        knl = self.get_kernel()
        target_batch_size = self._target_batch_size()
        if target_batch_size > 1:
            # Each target row is independent; keep the source-box/source-mode
            # reduction private to one work item while mapping neighboring target
            # points in a box to local lanes for warp/subgroup-friendly execution.
            knl = loopy.split_iname(
                knl,
                "tid",
                inner_length=target_batch_size,
                outer_tag="g.1",
                inner_tag="l.0",
                slabs=(0, 1),
            )

        knl = loopy.tag_inames(knl, {"tbox": "g.0"})
        knl = loopy.add_inames_for_unused_hw_axes(knl)
        return knl

    def __call__(self, queue, **kwargs):
        knl = self.get_cached_optimized_kernel()
        entry_knl = knl.default_entrypoint

        result = kwargs.pop("result")
        box_centers = kwargs.pop("box_centers")
        box_levels = kwargs.pop("box_levels")
        box_source_counts_nonchild = kwargs.pop("box_source_counts_nonchild")
        box_source_starts = kwargs.pop("box_source_starts")
        box_target_counts_nonchild = kwargs.pop("box_target_counts_nonchild")
        box_target_starts = kwargs.pop("box_target_starts")
        case_indices = kwargs.pop("case_indices")
        encoding_base = kwargs.pop("encoding_base")
        encoding_shift = kwargs.pop("encoding_shift")
        mode_nmlz_combined = kwargs.pop("mode_nmlz_combined")
        exterior_mode_nmlz_combined = kwargs.pop("exterior_mode_nmlz_combined")
        pde_recovery_kwargs = {}
        if self.reconstruction_kind in {
            "scalar-arithmetic-orbit",
            "signed-arithmetic-orbit",
        }:
            reconstruction_kwargs = {
                "arithmetic_case_orbit_ranks": kwargs.pop(
                    "arithmetic_case_orbit_ranks"
                ),
                "arithmetic_case_axis_perm": kwargs.pop(
                    "arithmetic_case_axis_perm"
                ),
                "arithmetic_case_axis_sign": kwargs.pop(
                    "arithmetic_case_axis_sign"
                ),
                "arithmetic_case_axis_group": kwargs.pop(
                    "arithmetic_case_axis_group"
                ),
                "arithmetic_case_value_offsets": kwargs.pop(
                    "arithmetic_case_value_offsets"
                ),
                "arithmetic_axis_sign_power": kwargs.pop(
                    "arithmetic_axis_sign_power"
                ),
                "arithmetic_axis_direction_signs": kwargs.pop(
                    "arithmetic_axis_direction_signs"
                ),
                "arithmetic_direction_sign_axis": kwargs.pop(
                    "arithmetic_direction_sign_axis"
                ),
                "quad_order": self.quad_order,
                "n_arithmetic_case_orbits": self.n_arithmetic_case_orbits,
            }
        elif self.reconstruction_kind == "generated-orbit":
            reconstruction_kwargs = {
                "reconstruction_qpoint_map": kwargs.pop("reconstruction_qpoint_map"),
                "reconstruction_case_map": kwargs.pop("reconstruction_case_map"),
                "reconstruction_signs": kwargs.pop("reconstruction_signs"),
                "reconstruction_lookup_keys": kwargs.pop(
                    "reconstruction_lookup_keys"
                ),
                "reconstruction_lookup_values": kwargs.pop(
                    "reconstruction_lookup_values"
                ),
                "reconstruction_sign_lookup_keys": kwargs.pop(
                    "reconstruction_sign_lookup_keys"
                ),
                "reconstruction_sign_lookup_values": kwargs.pop(
                    "reconstruction_sign_lookup_values"
                ),
                "n_reconstruction_transforms": self.n_reconstruction_transforms,
                "n_reconstruction_lookup_entries": (
                    self.n_reconstruction_lookup_entries
                ),
                "n_reconstruction_lookup_probes": self.n_reconstruction_lookup_probes,
                "n_reconstruction_sign_lookup_entries": (
                    self.n_reconstruction_sign_lookup_entries
                ),
                "n_reconstruction_sign_lookup_probes": (
                    self.n_reconstruction_sign_lookup_probes
                ),
            }
        elif self.reconstruction_kind == "pde-boundary-shell":
            reconstruction_kwargs = {
                "pde_target_boundary_ids": kwargs.pop("pde_target_boundary_ids"),
                "n_pde_boundary_target_points": self.n_pde_boundary_target_points,
            }
            pde_recovery_kwargs = {
                "pde_target_interior_ids": kwargs.pop("pde_target_interior_ids"),
                "pde_boundary_target_point_ids": kwargs.pop(
                    "pde_boundary_target_point_ids"
                ),
                "pde_box_local_target_ids": kwargs.pop("pde_box_local_target_ids"),
                "pde_recovery_matrix": kwargs.pop("pde_recovery_matrix"),
                "pde_self_correction": kwargs.pop("pde_self_correction"),
                "n_pde_interior_target_points": self.n_pde_interior_target_points,
            }
        else:
            reconstruction_kwargs = {
                "mode_qpoint_map": kwargs.pop("mode_qpoint_map"),
                "mode_case_map": kwargs.pop("mode_case_map"),
                "mode_case_scale": kwargs.pop("mode_case_scale"),
                "table_entry_ids": kwargs.pop("table_entry_ids"),
                "table_entry_scales": kwargs.pop("table_entry_scales"),
            }
        root_extent = kwargs.pop("root_extent")
        table_root_extent = kwargs.pop("table_root_extent")
        table_starting_level = kwargs.pop("table_starting_level")
        neighbor_source_boxes_starts = kwargs.pop("neighbor_source_boxes_starts")
        neighbor_source_boxes_lists = kwargs.pop("neighbor_source_boxes_lists")
        mode_coefs = kwargs.pop("mode_coefs")
        table_data_combined = kwargs.pop("table_data_combined")
        target_boxes = kwargs.pop("target_boxes")
        source_mode_ids = kwargs.pop("source_mode_ids")
        target_point_ids = kwargs.pop("target_point_ids")
        aligned_nboxes = box_centers.shape[1]

        if (
            self.n_tables == 1
            and not self.extra_kwargs.get("infer_kernel_scaling", False)
            and "kernel_scaling_code" not in self.extra_kwargs
        ):
            level_check_cache_key = (
                _array_layout_cache_token(neighbor_source_boxes_lists, queue),
                _array_layout_cache_token(box_levels, queue),
                int(table_starting_level),
            )

            if level_check_cache_key != self._single_table_level_check_cache_key:
                if hasattr(neighbor_source_boxes_lists, "get"):
                    source_boxes_h = neighbor_source_boxes_lists.get(queue)
                else:
                    source_boxes_h = np.asarray(neighbor_source_boxes_lists)

                if source_boxes_h.size:
                    if hasattr(box_levels, "get"):
                        box_levels_h = box_levels.get(queue)
                    else:
                        box_levels_h = np.asarray(box_levels)

                    source_levels = box_levels_h[source_boxes_h]
                    min_source_level = int(np.min(source_levels))
                    max_source_level = int(np.max(source_levels))

                    if min_source_level != max_source_level:
                        raise RuntimeError(
                            "Single near-field table without scaling cannot be used "
                            "with source boxes on multiple levels "
                            f"({min_source_level}..{max_source_level}); build per-level "
                            "tables or provide explicit scaling rules."
                        )

                    if min_source_level != int(table_starting_level):
                        raise RuntimeError(
                            "Single near-field table level mismatch: source level "
                            f"{min_source_level} but table_starting_level is "
                            f"{int(table_starting_level)}. Build the table at the active "
                            "source level or provide scaling rules."
                        )

                self._single_table_level_check_cache_key = level_check_cache_key

        integral_kernel_init_kargs = {
            field.name: getattr(self.integral_kernel, field.name)
            for field in fields(self.integral_kernel)
        }

        # help loopy's type inference
        for key, val in integral_kernel_init_kargs.items():
            if isinstance(val, int):
                integral_kernel_init_kargs[key] = np.int32(val)
            if isinstance(val, float):
                integral_kernel_init_kargs[key] = np.float64(val)

        extra_knl_args_from_init = {}
        for key, val in integral_kernel_init_kargs.items():
            if key in entry_knl.arg_dict:
                extra_knl_args_from_init[key] = val

        self._warn_case_encoding_drift(
            queue=queue,
            box_centers=box_centers,
            box_levels=box_levels,
            root_extent=root_extent,
            target_boxes=target_boxes,
            neighbor_source_boxes_starts=neighbor_source_boxes_starts,
            neighbor_source_boxes_lists=neighbor_source_boxes_lists,
            encoding_shift=encoding_shift,
            warn_tol=self.extra_kwargs.get("case_encoding_warn_tol"),
        )

        knl_exec = knl.executor(queue.context)
        first_pass_result = result
        if self.reconstruction_kind == "pde-boundary-shell":
            if isinstance(result, cl.array.Array):
                first_pass_result = cl.array.empty_like(result)
            else:
                first_pass_result = np.empty_like(result)

        evt, res = knl_exec(
            queue,
            result=first_pass_result,
            # db_table_lev=np.zeros(out_pot.shape),
            # db_case_id=np.zeros(out_pot.shape),
            # db_vec_id=np.zeros(out_pot.shape),
            # db_n_box_sources=np.zeros(out_pot.shape),
            # db_n_box_targets=np.zeros(out_pot.shape),
            # db_entry_id=np.zeros(out_pot.shape),
            box_centers=box_centers,
            box_levels=box_levels,
            box_source_counts_nonchild=box_source_counts_nonchild,
            box_source_starts=box_source_starts,
            box_target_counts_nonchild=box_target_counts_nonchild,
            box_target_starts=box_target_starts,
            case_indices=case_indices,
            n_tables=self.n_tables,
            n_table_entries=self.n_table_entries,
            n_q_points=self.n_q_points,
            n_cases=self.n_cases,
            aligned_nboxes=aligned_nboxes,
            encoding_base=encoding_base,
            encoding_shift=encoding_shift,
            mode_nmlz=mode_nmlz_combined,
            exterior_mode_nmlz=exterior_mode_nmlz_combined,
            n_tgt_boxes=len(target_boxes),
            neighbor_source_boxes_starts=neighbor_source_boxes_starts,
            root_extent=root_extent,
            source_boxes=neighbor_source_boxes_lists,
            n_source_boxes=len(neighbor_source_boxes_lists),
            source_mode_ids=source_mode_ids,
            n_source_particles=len(source_mode_ids),
            source_coefs=mode_coefs,
            table_data=table_data_combined,
            target_boxes=target_boxes,
            target_point_ids=target_point_ids,
            n_target_particles=len(target_point_ids),
            table_root_extent=table_root_extent,
            table_starting_level=table_starting_level,
            **reconstruction_kwargs,
            **extra_knl_args_from_init,
        )

        if (
            self.reconstruction_kind == "pde-boundary-shell"
            and self.n_pde_interior_target_points
        ):
            recovery_knl = self.get_pde_recovery_kernel()
            recovery_exec = recovery_knl.executor(queue.context)
            evt, recovery_res = recovery_exec(
                queue,
                result=result,
                shell_result=res["result"],
                source_coefs=mode_coefs,
                target_boxes=target_boxes,
                box_centers=box_centers,
                box_levels=box_levels,
                box_target_counts_nonchild=box_target_counts_nonchild,
                box_target_starts=box_target_starts,
                target_point_ids=target_point_ids,
                n_tgt_boxes=len(target_boxes),
                nboxes=len(box_levels),
                n_target_particles=len(target_point_ids),
                n_source_particles=len(source_mode_ids),
                n_tables=self.n_tables,
                n_q_points=self.n_q_points,
                aligned_nboxes=aligned_nboxes,
                root_extent=root_extent,
                table_root_extent=table_root_extent,
                table_starting_level=table_starting_level,
                pde_target_interior_ids=pde_recovery_kwargs[
                    "pde_target_interior_ids"
                ],
                pde_boundary_target_point_ids=pde_recovery_kwargs[
                    "pde_boundary_target_point_ids"
                ],
                pde_box_local_target_ids=pde_recovery_kwargs[
                    "pde_box_local_target_ids"
                ],
                pde_recovery_matrix=pde_recovery_kwargs["pde_recovery_matrix"],
                pde_self_correction=pde_recovery_kwargs["pde_self_correction"],
                n_pde_boundary_target_points=self.n_pde_boundary_target_points,
                n_pde_interior_target_points=self.n_pde_interior_target_points,
                **extra_knl_args_from_init,
            )
            res["result"] = recovery_res["result"]

        res["result"].add_event(evt)
        if isinstance(result, cl.array.Array):
            assert result is res["result"]
        else:
            assert isinstance(result, np.ndarray)
            result = res["result"].get(queue)

        queue.finish()
        logger.info("list1 evaluation finished")

        # check for data integrity
        # if not (np.max(np.abs(out_pot.get()))) < 100:
        #    import pudb; pu.db
        return result, evt


# }}} End eval from CSR data
