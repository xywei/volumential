from __future__ import division, absolute_import, print_function
import numpy as np
import pyopencl as cl
# from pytools import memoize_method
from volumential.nearfield_potential_table import NearFieldInteractionTable
from volumential.expansion_wrangler_interface import ExpansionWranglerInterface

import logging
logger = logging.getLogger(__name__)


def level_to_rscale(tree, level):
    return tree.root_extent * (2**-level)


from sumpy.fmm import SumpyExpansionWrangler


class FPNDExpansionWrangler(ExpansionWranglerInterface, SumpyExpansionWrangler):
    """This expansion wrangler uses "fpnd" strategy. That is, Far field is
    computed via Particle approximation and Near field is computed Directly.

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
    def __init__(self,
                 code_container,
                 queue,
                 tree,
                 near_field_table,
                 dtype,
                 fmm_level_to_order,
                 quad_order,
                 source_extra_kwargs=None,
                 kernel_extra_kwargs=None,
                 self_extra_kwargs=None):

        self.code = code_container
        self.queue = queue
        self.tree = tree
        if isinstance(near_field_table, list):
            self.near_field_table = near_field_table
            self.n_tables = len(near_field_table)
        elif isinstance(near_field_table, NearFieldInteractionTable):
            self.near_field_table = [near_field_table]
            self.n_tables = 1
        else:
            raise RuntimeError("Table type unrecognized")

        self.quad_order = quad_order

        self.root_table_source_box_extent = \
                self.near_field_table[0].source_box_extent
        table_starting_level = np.round(np.log(
                self.tree.root_extent / self.root_table_source_box_extent) /
                np.log(2))
        for lev, table in zip(range(len(self.near_field_table)),
                self.near_field_table):
            assert table.quad_order == self.quad_order

            if not table.is_built:
                raise RuntimeError(
                "Near field interaction table needs to be built prior to being used"
                )

            table_root_extent = table.source_box_extent * 2**lev
            assert abs(self.root_table_source_box_extent -
                table_root_extent) < 1e-15

            # If the kernel cannot be scaled,
            # - tree_root_extent must be integral times of table_root_extent
            # - n_tables must be sufficient
            if self.n_tables > 1:
                if not abs(
                        int(self.tree.root_extent / table_root_extent)
                        * table_root_extent - self.tree.root_extent) < 1e-15:
                    raise RuntimeError("Incompatible list of tables: the "
                            "source_box_extent of the root table must divide the "
                            "bounding box's extent by an integer.")

        if self.n_tables > 1:
            # this checks that the boxes at the highest level are covered
            if not tree.nlevels <= len(
                    self.near_field_table) + table_starting_level:
                raise RuntimeError("Insufficient list of tables: the "
                        "finest level mesh cells at level "
                        + str(tree.nlevels) + " are not covered.")

            # the check that the boxes at the coarsest level are covered is
            # deferred until trav.target_boxes is passed when invoking eval_direct

        self.dtype = dtype

        if source_extra_kwargs is None:
            source_extra_kwargs = {}

        if kernel_extra_kwargs is None:
            kernel_extra_kwargs = {}

        if self_extra_kwargs is None:
            self_extra_kwargs = {}

        if not callable(fmm_level_to_order):
            raise TypeError("fmm_level_to_order not passed")

        base_kernel = code_container.get_base_kernel()
        kernel_arg_set = frozenset(kernel_extra_kwargs.items())
        self.level_orders = [
            fmm_level_to_order(base_kernel, kernel_arg_set, tree, lev)
            for lev in range(tree.nlevels)
        ]

        # print("Multipole order = ",self.level_orders)

        self.source_extra_kwargs = source_extra_kwargs
        self.kernel_extra_kwargs = kernel_extra_kwargs
        self.self_extra_kwargs = self_extra_kwargs

        self.extra_kwargs = source_extra_kwargs.copy()
        self.extra_kwargs.update(self.kernel_extra_kwargs)

# }}} End constructor

# {{{ data vector utilities

    def multipole_expansion_zeros(self):
        return SumpyExpansionWrangler.multipole_expansion_zeros(self)

    def local_expansion_zeros(self):
        return SumpyExpansionWrangler.local_expansion_zeros(self)

    def output_zeros(self):
        return SumpyExpansionWrangler.output_zeros(self)

    def reorder_sources(self, source_array):
        return SumpyExpansionWrangler.reorder_sources(
                self, source_array)

    def reorder_potentials(self, potentials):
        return SumpyExpansionWrangler.reorder_potentials(
                self, potentials)

    def finalize_potentials(self, potentials):
        # return potentials
        return SumpyExpansionWrangler.finalize_potentials(
                self, potentials)

# }}} End data vector utilities

# {{{ formation & coarsening of multipoles

    def form_multipoles(self, level_start_source_box_nrs, source_boxes,
                        src_weights):
        return SumpyExpansionWrangler.form_multipoles(self,
                level_start_source_box_nrs,
                source_boxes,
                src_weights)

    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
                           source_parent_boxes, mpoles):
        return SumpyExpansionWrangler.coarsen_multipoles(self,
                level_start_source_parent_box_nrs,
                source_parent_boxes,
                mpoles)

# }}} End formation & coarsening of multipoles

# {{{ direct evaluation of near field interactions

    def eval_direct(self, target_boxes, neighbor_source_boxes_starts,
                    neighbor_source_boxes_lists, mode_coefs):

        # NOTE: mode_coefs are similar to source_weights BUT
        # do not include quadrature weights (purely function
        # expansiona coefficients)

        # NOTE: inputs should be all on the device main memory (cl.Arrays)

        if len(self.code.out_kernels) > 1:
            raise NotImplementedError(
                "Currently only one output kernel is supported")

        if self.n_tables > 1:
            # this checks that the boxes at the coarsest level
            # and allows for some round-off error
            min_lev = np.min(self.tree.box_levels.get(self.queue)[
                target_boxes.get(self.queue)])
            largest_cell_extent = self.tree.root_extent * 0.5**min_lev
            if not self.near_field_table[0].source_box_extent >= (
                    largest_cell_extent - 1e-15):
                raise RuntimeError("Insufficient list of tables: the "
                        "coarsest level mesh cells at level "
                        + str(min_lev) + " are not covered.")

        pot = self.output_zeros()

        # table.case_encode
        distinct_numbers = set()
        for vec in self.near_field_table[0].interaction_case_vecs:
            for l in vec:
                distinct_numbers.add(l)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        case_indices_dev = cl.array.to_device(self.queue, self.near_field_table[0].case_indices)

        # table.data
        from pytools.obj_array import make_obj_array
        table_data_combined = np.zeros(
                (len(self.near_field_table),
                    len(self.near_field_table[0].data)))
        mode_nmlz_combined = np.zeros(
                (len(self.near_field_table),
                    len(self.near_field_table[0].mode_normalizers)))
        for lev in range(len(self.near_field_table)):
            table_data_combined[lev,:] = self.near_field_table[lev].data
            mode_nmlz_combined[lev,:] = self.near_field_table[lev].mode_normalizers

        logger.debug("Table data congregated")

        # from pyopencl.tools import dtype_to_c_struct
        # box_id_dtype = np.int32

        def codegen_vec_component(dimension):
            return "(" + \
                "(box_centers[" + str(dimension) + ", target_box_id]" + \
                "- box_centers[" + str(dimension) + ", source_box_id]" + \
                ") / sbox_extent * 4.0 + encoding_shift" + \
                   ")"

        def codegen_vec_id(dim):
            code = "0.0"
            for d in range(dim):
                code = "(" + code + ") * encoding_base"
                code = code + "+" + codegen_vec_component(d)
            # Add additional 0.5 to make floor() to find the nearest integer
            code = code + " + 0.5"
            return code

        import loopy
        # Allow user to pass more tables to force using multiple tables
        # instead of performing kernel scaling
        if self.code.out_kernels[0].__repr__() == "LapKnl2D":

            if self.n_tables == 1:

                def codegen_compute_scaling():
                    return "sbox_extent * sbox_extent / \
                            (table_root_extent * table_root_extent)"

                def codegen_compute_displacement():
                    s = "-0.5 / PI * scaling * \
                            log(sbox_extent / table_root_extent) * \
                            mode_nmlz[table_lev, sid]"
                    return s.replace("PI", "3.1415926535897932384626433832795028")

                def codegen_get_table_level():
                    return "0"

            else:

                print("Using multilayer table for Laplace")

                def codegen_compute_scaling():
                    return "1.0"

                def codegen_compute_displacement():
                    return "0.0"

                def codegen_get_table_level():
                    code = "log(table_root_extent / sbox_extent) / log(2.0)"
                    return code

        elif self.code.out_kernels[0].__repr__() == "ConKnl2D":

            if self.n_tables == 1:

                def codegen_compute_scaling():
                    return "sbox_extent * sbox_extent"

                def codegen_compute_displacement():
                    return "0.0"

                def codegen_get_table_level():
                    return "0"

            else:

                def codegen_compute_scaling():
                    return "1.0"

                def codegen_compute_displacement():
                    return "0.0"

                def codegen_get_table_level():
                    code = "log(table_root_extent / sbox_extent) / log(2.0)"
                    return code

        else:
            # Kernel scaling not supported
            def codegen_compute_scaling():
                return "1.0"

            def codegen_compute_displacement():
                return "0.0"

            def codegen_get_table_level():
                code = "log(table_root_extent / sbox_extent) / log(2.0)"
                return code

            #raise NotImplementedError("Output kernel type " +
            #                          self.code.out_kernels[0].__repr__() +
            #                          " not supported")

        lpknl = loopy.make_kernel(  # NOQA
            [
                "{ [ tbox ] : 0 <= tbox < n_tgt_boxes }",
                "{ [ sbox ] : sbox_begin <= sbox < sbox_end }",
                "{ [ tid, sid ] : 0 <= tid < n_box_targets and \
                                      0 <= sid < n_box_sources }"
            ], """
            for tbox
                <> target_box_id    = target_boxes[tbox]
                <> box_target_beg   = box_target_starts[target_box_id]
                <> n_box_targets    = box_target_counts_cumul[target_box_id]

                <> sbox_begin = neighbor_source_boxes_starts[tbox]
                <> sbox_end   = neighbor_source_boxes_starts[tbox+1]

                <> tbox_level  = box_levels[target_box_id]
                <> tbox_extent = root_extent * (1.0 / (2**tbox_level))

                for sbox
                    <> source_box_id  = source_boxes[sbox]
                    <> n_box_sources  = box_source_counts_cuml[source_box_id]
                    <> box_source_beg = box_source_starts[source_box_id]

                    <> sbox_level  = box_levels[source_box_id]
                    <> sbox_extent = root_extent * (1.0 / (2**sbox_level))

                    <> table_lev_tmp = GET_TABLE_LEVEL
                    table_lev = round(table_lev_tmp) {id=tab_lev}

                    vec_id = COMPUTE_VEC_ID {id=vec_id}
                    <> case_id = case_indices[vec_id] {dep=vec_id}

                    <> scaling = COMPUTE_SCALING

                    for tid
                        <> target_id = box_target_beg + tid

                        for sid
                            <> source_id = box_source_beg + sid
                            <> pair_id = sid * n_box_targets + tid
                            <> entry_id = case_id * \
                                          (n_box_targets * n_box_sources) \
                                          + pair_id

                            <> displacement = COMPUTE_DISPLACEMENT

                            <> integ = table_data[table_lev, entry_id] * scaling \
                                       + displacement {dep=tab_lev}
                            # <> source_id_tree = user_source_ids[source_id]
                            <> coef = source_coefs[source_id]

                            # <> target_id_user = sorted_target_ids[target_id]

                            result[target_id] = result[target_id] \
                                                     + coef * integ
                            # db_table_lev[target_id] = table_lev_tmp {dep=tab_lev}
                        end
                    end
                end
            end
            """.replace("COMPUTE_VEC_ID",
                        codegen_vec_id(self.near_field_table[0].dim)).replace(
                            "COMPUTE_SCALING",
                            codegen_compute_scaling()).replace(
                                "COMPUTE_DISPLACEMENT",
                                codegen_compute_displacement()).replace(
                                        "GET_TABLE_LEVEL",
                                        codegen_get_table_level()),
            [
                loopy.TemporaryVariable("vec_id", np.int64),
                loopy.TemporaryVariable("table_lev", np.int64),
                loopy.ValueArg("encoding_base", np.int64),
                loopy.GlobalArg("mode_nmlz", np.float64,
                    shape=(self.n_tables,
                        self.near_field_table[0].n_q_points)),
                loopy.GlobalArg("table_data", np.float64,
                    shape=(self.n_tables,
                        len(self.near_field_table[0].data))),
                loopy.GlobalArg("source_boxes", np.int32,
                                len(neighbor_source_boxes_lists)),
                loopy.GlobalArg("box_centers", None, "dim, aligned_nboxes"),
                loopy.ValueArg("aligned_nboxes", np.int32),
                loopy.ValueArg("table_root_extent", np.float64),
                loopy.ValueArg("dim", np.int32), "..."
            ])

        # lpknl = loopy.set_options(lpknl_laplace, write_code=True)
        lpknl = loopy.set_options(lpknl, return_dict=True)
        # print(lpknl)

        evt, res = lpknl(
            self.queue,
            result=pot[0],
            # db_table_lev=np.zeros(pot[0].shape),
            box_centers=self.tree.box_centers,
            box_levels=self.tree.box_levels,
            box_source_counts_cuml=self.tree.box_source_counts_cumul,
            box_source_starts=self.tree.box_source_starts,
            box_target_counts_cumul=self.tree.box_target_counts_cumul,
            box_target_starts=self.tree.box_target_starts,
            case_indices=case_indices_dev,
            dim=self.near_field_table[0].dim,
            encoding_base=base,
            encoding_shift=shift,
            mode_nmlz=mode_nmlz_combined,
            n_tgt_boxes=len(target_boxes),
            neighbor_source_boxes_starts=neighbor_source_boxes_starts,
            root_extent=self.tree.root_extent,
            source_boxes=neighbor_source_boxes_lists,
            source_coefs=mode_coefs,
            table_data=table_data_combined,
            target_boxes=target_boxes,
            table_root_extent=self.root_table_source_box_extent
            )
        # sorted_target_ids=self.tree.user_source_ids,
        # user_source_ids=self.tree.user_source_ids)

        assert (pot[0] is res['result'])
        for pot_i in pot:
            pot_i.add_event(evt)
        # print(max(res['db_table_lev'].get()))

        # print(pot)

        return pot

# }}} End direct evaluation of near field interactions

# {{{ downward pass of fmm

    def multipole_to_local(self, level_start_target_box_nrs, target_boxes,
                           src_box_starts, src_box_lists, mpole_exps):
        return SumpyExpansionWrangler.multipole_to_local(self,
                level_start_target_box_nrs,
                target_boxes,
                src_box_starts,
                src_box_lists,
                mpole_exps)

    def eval_multipoles(self, level_start_target_box_nrs, target_boxes,
                        source_boxes_by_level, mpole_exps):
        return SumpyExpansionWrangler.eval_multipoles(self,
                level_start_target_box_nrs,
                target_boxes,
                source_boxes_by_level,
                mpole_exps)

    def form_locals(self, level_start_target_or_target_parent_box_nrs,
                    target_or_target_parent_boxes, starts, lists, src_weights):
        return SumpyExpansionWrangler.form_locals(self,
                level_start_target_or_target_parent_box_nrs,
                target_or_target_parent_boxes,
                starts,
                lists,
                src_weights)

    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
                      target_or_target_parent_boxes, local_exps):
        return SumpyExpansionWrangler.refine_locals(self,
                level_start_target_or_target_parent_box_nrs,
                target_or_target_parent_boxes,
                local_exps)

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        return SumpyExpansionWrangler.eval_locals(self,
                level_start_target_box_nrs,
                target_boxes,
                local_exps)

# }}} End downward pass of fmm

# {{{ direct evaluation of p2p (discrete) interactions

    def eval_direct_p2p(self, target_boxes, source_box_starts,
                    source_box_lists, src_weights):
        return SumpyExpansionWrangler.eval_direct(self,
                target_boxes,
                source_box_starts,
                source_box_lists,
                src_weights)

# }}} End direct evaluation of p2p interactions


# vim: filetype=pyopencl:foldmethod=marker
