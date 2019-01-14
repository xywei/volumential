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

import numpy as np
import loopy

from sumpy.tools import KernelCacheWrapper

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# {{{ near field eval base class


class NearFieldEvalBase(KernelCacheWrapper):
    def __init__(
        self,
        integral_kernel,
        table_data_shapes,
        options=[],
        name=None,
        device=None,
        **kwargs
    ):

        self.integral_kernel = integral_kernel

        self.n_tables = table_data_shapes["n_tables"]
        self.n_q_points = table_data_shapes["n_q_points"]
        self.n_table_entries = table_data_shapes["n_table_entries"]

        self.options = options
        self.name = name or self.default_name
        self.divice = device
        self.extra_kwargs = kwargs

        # Allow user to pass more tables to force using multiple tables
        # instead of performing kernel scaling
        if not ("infer_kernel_scaling" in self.extra_kwargs):
            self.extra_kwargs["infer_kernel_scaling"] = self.n_tables == 1

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
    default_name = "near_field_from_csr"

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

    def codegen_compute_scaling(self):
        if ("infer_kernel_scaling" in self.extra_kwargs) and (
            self.extra_kwargs["infer_kernel_scaling"]
        ):
            # Laplace 2D
            if self.kname == "LapKnl2D":
                logger.info("scaling for LapKnl2D")
                return "sbox_extent * sbox_extent / \
                        (table_root_extent * table_root_extent)"

            # Constant 2D
            elif self.kname == "CstKnl2D":
                logger.info("scaling for CstKnl2D")
                return "sbox_extent * sbox_extent / \
                        (table_root_extent * table_root_extent)"

            # Laplace 3D
            elif self.kname == "LapKnl3D":
                logger.info("scaling for Lapknl3D")
                return "sbox_extent * sbox_extent / \
                        (table_root_extent * table_root_extent)"

            # Constant 3D
            elif self.kname == "CstKnl3D":
                logger.info("scaling for CstKnl3D")
                return "sbox_extent * sbox_extent * sbox_extent / \
                        (table_root_extent * table_root_extent * table_root_extent)"

            else:
                logger.warn(
                    "Kernel not scalable and not using multiple tables, "
                    "to get correct results, please make sure that your "
                    "tree is uniform and only needs one table."
                )
                return "1.0"
        else:
            logger.info("not scaling for " + self.kname)
            logger.info("(using multiple tables)")
            return "1.0"

    def codegen_compute_displacement(self):
        if ("infer_kernel_scaling" in self.extra_kwargs) and (
            self.extra_kwargs["infer_kernel_scaling"]
        ):
            # Laplace 2D
            if self.kname == "LapKnl2D":
                logger.info("displacement for laplace 2D")
                s = "-0.5 / PI * scaling * \
                        log(sbox_extent / table_root_extent) * \
                        mode_nmlz[table_lev, sid]"

                import math

                return s.replace("PI", str(math.pi))
            # Constant 2D
            elif self.kname == "CstKnl2D":
                logger.info("no displacement for CstKnl2D")
                return "0.0"
            # Laplace 3D
            elif self.kname == "LapKnl3D":
                logger.info("no displacement for LapKnl3D")
                return "0.0"
            # Constant 3D
            elif self.kname == "CstKnl3D":
                logger.info("no displacement for CstKnl3D")
                return "0.0"
            else:
                logger.warn(
                    "Kernel not scalable and not using multiple tables, "
                    "to get correct results, please make sure that your "
                    "tree is uniform and only needs one table."
                )
                return "0.0"
        else:
            logger.info("no displacement for " + self.kname)
            logger.info("(using multiple tables)")
            return "0.0"

    def codegen_get_table_level(self):
        if ("infer_kernel_scaling" in self.extra_kwargs) and (
            self.extra_kwargs["infer_kernel_scaling"]
        ):
            if (
                self.kname == "LapKnl2D"
                or self.kname == "LapKnl3D"
                or self.kname == "CstKnl2D"
                or self.kname == "CstKnl3D"
            ):
                logger.info("scaling from table[0] for " + self.kname)
                return "0.0"
            else:
                logger.warn(
                    "Kernel not scalable and not using multiple tables, "
                    "to get correct results, please make sure that your "
                    "tree is uniform and only needs one table."
                )
                return "0.0"
        else:
            logger.info("computing table level from box size")
            logger.info("(using multiple tables)")
            code = "log(table_root_extent / sbox_extent) / log(2.0)"
            return code

    def get_kernel(self):

        lpknl = loopy.make_kernel(  # NOQA
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
                <> n_box_targets    = box_target_counts_cumul[target_box_id]

                <> sbox_begin = neighbor_source_boxes_starts[tbox]
                <> sbox_end   = neighbor_source_boxes_starts[tbox+1]

                <> tbox_level  = box_levels[target_box_id]
                <> tbox_extent = root_extent * (1.0 / (2**tbox_level))

                for tid
                    <> target_id = box_target_beg + tid
                end

                for tid, sbox
                    <> source_box_id  = source_boxes[sbox]
                    <> n_box_sources  = box_source_counts_cuml[source_box_id]
                    <> box_source_beg = box_source_starts[source_box_id]

                    <> sbox_level  = box_levels[source_box_id]
                    <> sbox_extent = root_extent * (1.0 / (2**sbox_level))

                    table_lev_tmp = GET_TABLE_LEVEL {id=tab_lev_tmp}
                    table_lev = round(table_lev_tmp) {id=tab_lev,dep=tab_lev_tmp}

                    vec_id_tmp = COMPUTE_VEC_ID {id=vec_id_tmp}
                    vec_id = round(vec_id_tmp) {id=vec_id,dep=vec_id_tmp}
                    <> case_id = case_indices[vec_id] {dep=vec_id}

                    <> scaling = COMPUTE_SCALING

                    for sid
                        <> source_id = box_source_beg + sid
                        <> pair_id = sid * n_box_targets + tid
                        <> entry_id = case_id * \
                                      (n_box_targets * n_box_sources) \
                                      + pair_id

                        <> displacement = COMPUTE_DISPLACEMENT

                        <> integ = table_data[table_lev, entry_id] * scaling \
                                   + displacement {id=integ,dep=tab_lev}
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
                        coef * integ) {dep=integ:coef}

                # Try inspecting case_id if something goes wrong
                # (like segmentation fault) and look for -1's
                # result[target_id] = min((sbox, sid), case_id)
                # result[target_id] = vec_id_tmp

                end
            end
            """.replace(
                "COMPUTE_VEC_ID", self.codegen_vec_id()
            )
            .replace("COMPUTE_SCALING", self.codegen_compute_scaling())
            .replace("COMPUTE_DISPLACEMENT", self.codegen_compute_displacement())
            .replace("GET_TABLE_LEVEL", self.codegen_get_table_level()),
            [
                loopy.TemporaryVariable("vec_id", np.int32),
                loopy.TemporaryVariable("vec_id_tmp", np.float64),
                loopy.TemporaryVariable("table_lev", np.int32),
                loopy.TemporaryVariable("table_lev_tmp", np.float64),
                loopy.ValueArg("encoding_base", np.int32),
                loopy.GlobalArg(
                    "mode_nmlz",
                    np.float64,
                    # shape=(self.n_tables, self.n_q_points)
                    "n_tables, n_q_points",
                ),
                loopy.GlobalArg(
                    "table_data",
                    np.float64,
                    # shape=(self.n_tables, self.n_table_entries)
                    "n_tables, n_table_entries",
                ),
                loopy.GlobalArg("source_boxes", np.int32, "n_source_boxes"),
                loopy.GlobalArg("box_centers", None, "dim, aligned_nboxes"),
                loopy.ValueArg("aligned_nboxes", np.int32),
                loopy.ValueArg("table_root_extent", np.float64),
                loopy.ValueArg(
                    "dim, n_source_boxes, n_tables, " "n_q_points, n_table_entries",
                    np.int32,
                ),
                "...",
            ],
            name="near_field",
        )

        # print(lpknl)
        # lpknl = loopy.set_options(lpknl, write_code=True)

        lpknl = loopy.set_options(lpknl, return_dict=True)

        return lpknl

    def get_cache_key(self):
        return (
            type(self).__name__,
            self.name,
            self.kname,
            "infer_scaling=" + str(self.extra_kwargs["infer_kernel_scaling"]),
        )

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = loopy.split_iname(knl, "tbox", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        knl = self.get_cached_optimized_kernel()

        result = kwargs.pop("result")
        box_centers = kwargs.pop("box_centers")
        box_levels = kwargs.pop("box_levels")
        box_source_counts_cumul = kwargs.pop("box_source_counts_cumul")
        box_source_starts = kwargs.pop("box_source_starts")
        box_target_counts_cumul = kwargs.pop("box_target_counts_cumul")
        box_target_starts = kwargs.pop("box_target_starts")
        case_indices = kwargs.pop("case_indices")
        encoding_base = kwargs.pop("encoding_base")
        encoding_shift = kwargs.pop("encoding_shift")
        mode_nmlz_combined = kwargs.pop("mode_nmlz_combined")
        root_extent = kwargs.pop("root_extent")
        table_root_extent = kwargs.pop("table_root_extent")
        neighbor_source_boxes_starts = kwargs.pop("neighbor_source_boxes_starts")
        neighbor_source_boxes_lists = kwargs.pop("neighbor_source_boxes_lists")
        mode_coefs = kwargs.pop("mode_coefs")
        table_data_combined = kwargs.pop("table_data_combined")
        target_boxes = kwargs.pop("target_boxes")

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
            box_source_counts_cuml=box_source_counts_cumul,
            box_source_starts=box_source_starts,
            box_target_counts_cumul=box_target_counts_cumul,
            box_target_starts=box_target_starts,
            case_indices=case_indices,
            dim=self.dim,
            n_tables=self.n_tables,
            n_table_entries=self.n_table_entries,
            n_q_points=self.n_q_points,
            encoding_base=encoding_base,
            encoding_shift=encoding_shift,
            mode_nmlz=mode_nmlz_combined,
            n_tgt_boxes=len(target_boxes),
            neighbor_source_boxes_starts=neighbor_source_boxes_starts,
            root_extent=root_extent,
            source_boxes=neighbor_source_boxes_lists,
            n_source_boxes=len(neighbor_source_boxes_lists),
            source_coefs=mode_coefs,
            table_data=table_data_combined,
            target_boxes=target_boxes,
            table_root_extent=table_root_extent,
        )

        assert result is res["result"]
        result.add_event(evt)

        # check for data integrity
        # if not (np.max(np.abs(out_pot.get()))) < 100:
        #    import pudb; pu.db
        return result, evt


# }}} End eval from CSR data
