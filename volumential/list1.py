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
.. autoclass:: NearFieldEvalBase
   :members:
.. autoclass:: NearFieldFromCSR
   :members:
"""

import logging
from dataclasses import fields

import numpy as np

import loopy
import pyopencl as cl

from volumential.tools import KernelCacheWrapper


logger = logging.getLogger(__name__)


# {{{ near field eval base class


class NearFieldEvalBase(KernelCacheWrapper):
    """Base class of near-field evalulator."""

    default_name = "near_field_eval_base"

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
        self.potential_kind = potential_kind

        assert np.isreal(self.n_tables)
        assert np.isreal(self.n_q_points)
        assert np.isreal(self.n_table_entries)
        assert np.isreal(self.n_cases)

        self.options = options
        self.name = name or self.default_name
        self.divice = device
        self.extra_kwargs = kwargs

        # Allow user to pass more tables to force using multiple tables
        # instead of performing kernel scaling
        if "infer_kernel_scaling" not in self.extra_kwargs:
            self.extra_kwargs["infer_kernel_scaling"] = self.n_tables == 1

        # Do not infer scaling rules when user defined rules are present
        if ("kernel_scaling_code" in self.extra_kwargs) or (
            "kernel_displacement_code" in self.extra_kwargs
        ):
            self.extra_kwargs["infer_kernel_scaling"] = False
            # the two codes must be simultaneously given
            assert ("kernel_scaling_code" in self.extra_kwargs) and (
                "kernel_displacement_code" in self.extra_kwargs
            )

        self.kname = self.integral_kernel.__repr__()
        self.dim = self.integral_kernel.dim

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

    def codegen_vec_component(self, d=None):
        if d is None:
            dimension = self.dim - 1
        else:
            dimension = d
        return (
            "("
            + "(box_centers["
            + str(dimension)
            + ", target_box_id]"
            + "- box_centers["
            + str(dimension)
            + ", source_box_id]"
            + ") / sbox_extent * 4.0 + encoding_shift"
            + ")"
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
        if ("infer_kernel_scaling" in self.extra_kwargs) and (
            self.extra_kwargs["infer_kernel_scaling"]
        ):
            # Laplace 2D
            if self._is_laplace_kernel(2):
                logger.info("scaling for LapKnl2D")
                code = "BOX_extent * BOX_extent / \
                        (table_root_extent * table_root_extent)"

            elif self._is_axis_target_derivative_of_laplace(2):
                logger.info("scaling for Grad(LapKnl2D)")
                code = "BOX_extent / table_root_extent"

            # Constant 2D
            elif self._is_constant_kernel(2):
                logger.info("scaling for CstKnl2D")
                code = "BOX_extent * BOX_extent / \
                        (table_root_extent * table_root_extent)"

            # Laplace 3D
            elif self._is_laplace_kernel(3):
                logger.info("scaling for Lapknl3D")
                code = "BOX_extent * BOX_extent / \
                        (table_root_extent * table_root_extent)"

            elif self._is_axis_target_derivative_of_laplace(3):
                logger.info("scaling for Grad(LapKnl3D)")
                code = "BOX_extent / table_root_extent"

            # Constant 3D
            elif self._is_constant_kernel(3):
                logger.info("scaling for CstKnl3D")
                code = "BOX_extent * BOX_extent * BOX_extent / \
                        (table_root_extent * table_root_extent * table_root_extent)"

            else:
                logger.warn(
                    "Kernel not scalable and not using multiple tables, "
                    "to get correct results, please make sure that your "
                    "tree is uniform and only needs one table."
                )
                code = "1.0"

            return code.replace("BOX", box_name)

        elif "kernel_scaling_code" in self.extra_kwargs:
            # user-defined scaling rule
            assert isinstance(self.extra_kwargs["kernel_scaling_code"], str)
            logger.info(
                "Using scaling rule %s for %s.",
                self.extra_kwargs["kernel_scaling_code"],
                self.kname,
            )
            return self.extra_kwargs["kernel_scaling_code"]
        else:
            logger.info("not scaling for " + self.kname)
            logger.info("(using multiple tables)")
            return "1.0"

    def codegen_compute_displacement(self, box_name="sbox"):
        if ("infer_kernel_scaling" in self.extra_kwargs) and (
            self.extra_kwargs["infer_kernel_scaling"]
        ):
            # Laplace 2D
            if self._is_laplace_kernel(2):
                logger.info("displacement for laplace 2D")
                s = "-0.5 / PI * scaling * \
                        log(BOX_extent / table_root_extent) * \
                        mode_nmlz[table_lev, sid]"
                import math

                code = s.replace("PI", str(math.pi))

            # Constant 2D
            elif self._is_constant_kernel(2):
                logger.info("no displacement for CstKnl2D")
                code = "0.0"
            # Laplace 3D
            elif self._is_laplace_kernel(3):
                logger.info("no displacement for LapKnl3D")
                code = "0.0"
            # Constant 3D
            elif self._is_constant_kernel(3):
                logger.info("no displacement for CstKnl3D")
                code = "0.0"
            else:
                logger.warn(
                    "Kernel not scalable and not using multiple tables, "
                    "to get correct results, please make sure that either "
                    "no displacement is needed, or the box "
                    "tree is uniform and only needs one table."
                )
                code = "0.0"
        elif "kernel_displacement_code" in self.extra_kwargs:
            # user-defined displacement rule
            assert isinstance(self.extra_kwargs["kernel_displacement_code"], str)
            logger.info(
                "Using displacement %s for %s.",
                self.extra_kwargs["kernel_displacement_code"],
                self.kname,
            )
            code = self.extra_kwargs["kernel_displacement_code"]
        else:
            logger.info("no displacement for " + self.kname)
            logger.info("(using multiple tables)")
            code = "0.0"

        return code.replace("BOX", box_name)

    def codegen_get_table_level(self, box_name="sbox"):
        if ("infer_kernel_scaling" in self.extra_kwargs) and (
            self.extra_kwargs["infer_kernel_scaling"]
        ):
            if (
                self._is_laplace_kernel(2)
                or self._is_laplace_kernel(3)
                or self._is_constant_kernel(2)
                or self._is_constant_kernel(3)
            ):
                logger.info("scaling from table[0] for " + self.kname)
                code = "0.0"
            else:
                logger.warn(
                    "Kernel not scalable and not using multiple tables, "
                    "to get correct results, please make sure that your "
                    "tree is uniform and only needs one table."
                )
                code = "0.0"
        elif "kernel_scaling_code" in self.extra_kwargs:
            # Using custom scaling
            code = "0.0"
        else:
            logger.info("computing table level from box size")
            logger.info("(using multiple tables)")
            code = "log(table_root_extent / BOX_extent) / log(2.0)"

        return code.replace("BOX", box_name)

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

        lpknl = loopy.make_kernel(
            [
                "{ [ tbox ] : 0 <= tbox < n_tgt_boxes }",
                "{ [ tid, sbox ] : 0 <= tid < n_box_targets and \
                        sbox_begin <= sbox < sbox_end }",
                "{ [ sid ] : 0 <= sid < n_box_sources }",
            ],
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
                    <> target_id = box_target_beg + tid
                end

                for tid, sbox
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
                        <> source_mode_id_sym = mode_qpoint_map[source_mode_id, source_mode_id]
                        <> target_point_id_sym = mode_qpoint_map[source_mode_id, target_point_id]
                        <> case_id_sym = mode_case_map[source_mode_id, case_id]
                        <> pair_id = source_mode_id_sym * n_q_points + target_point_id_sym
                        <> entry_id_full = case_id_sym * (n_q_points * n_q_points) + pair_id
                        <> entry_id = table_entry_ids[entry_id_full]
                        <> has_entry = entry_id >= 0

                        <> displacement = COMPUTE_DISPLACEMENT

                        <> integ = if(has_entry,
                                table_data[table_lev, entry_id] * scaling
                                + displacement,
                                0) {id=integ,dep=tab_lev}
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

                for tid

                    result[target_id] = sum((sbox, sid),
                        coef * integ) + EXTERIOR_PART {dep=integ:coef:extnmlz}

                    # Try inspecting case_id if something goes wrong
                    # (like segmentation fault) and look for -1's
                    # result[target_id] = min((sbox, sid), case_id)
                    # result[target_id] = vec_id_tmp

                end
            end
            """.replace("COMPUTE_VEC_ID", self.codegen_vec_id())
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
                loopy.GlobalArg(
                    "table_entry_ids", np.int32, "n_cases*n_q_points*n_q_points"
                ),
                loopy.GlobalArg("mode_qpoint_map", np.int32, "n_q_points, n_q_points"),
                loopy.GlobalArg("mode_case_map", np.int32, "n_q_points, n_cases"),
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
                loopy.ValueArg(
                    "dim, n_source_boxes, n_tables, n_q_points, n_cases, n_table_entries, n_source_particles, n_target_particles",
                    np.int32,
                ),
                "...",
            ],
            name="near_field",
            lang_version=(2018, 2),
        )

        # lpknl = loopy.set_options(lpknl, write_code=True)
        lpknl = loopy.set_options(lpknl, return_dict=True)

        return lpknl

    def get_cache_key(self):
        return (
            type(self).__name__,
            "kernel-v8",
            self.name,
            self.kname,
            "complex_kernel=" + str(self.integral_kernel.is_complex_valued),
            "potential_kind=%d" % self.potential_kind,
            "infer_scaling=" + str(self.extra_kwargs["infer_kernel_scaling"]),
            "scaling_policy=" + self.codegen_compute_scaling(),
            "displacement_policy=" + self.codegen_compute_displacement(),
        )

    def get_optimized_kernel(self, ncpus=None):
        knl = self.get_kernel()
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
        mode_qpoint_map = kwargs.pop("mode_qpoint_map")
        mode_case_map = kwargs.pop("mode_case_map")
        table_entry_ids = kwargs.pop("table_entry_ids")
        root_extent = kwargs.pop("root_extent")
        table_root_extent = kwargs.pop("table_root_extent")
        neighbor_source_boxes_starts = kwargs.pop("neighbor_source_boxes_starts")
        neighbor_source_boxes_lists = kwargs.pop("neighbor_source_boxes_lists")
        mode_coefs = kwargs.pop("mode_coefs")
        table_data_combined = kwargs.pop("table_data_combined")
        target_boxes = kwargs.pop("target_boxes")
        source_mode_ids = kwargs.pop("source_mode_ids")
        target_point_ids = kwargs.pop("target_point_ids")

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

        evt, res = knl(
            queue,
            result=result,
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
            encoding_base=encoding_base,
            encoding_shift=encoding_shift,
            mode_nmlz=mode_nmlz_combined,
            exterior_mode_nmlz=exterior_mode_nmlz_combined,
            table_entry_ids=table_entry_ids,
            mode_qpoint_map=mode_qpoint_map,
            mode_case_map=mode_case_map,
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
            **extra_knl_args_from_init,
        )

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
