from __future__ import division, absolute_import, print_function

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
        """
        near_field_table can either one of three things:
            1. a single table, when len(out_kernels) = 1 (single level)
            2. a list of tables, when len(out_kernels) = 1 (multiple levels)
            3. otherwise, a dictionary from kernel.__repr__() to a list of its tables
        """

        self.code = code_container
        self.queue = queue
        self.tree = tree

        self.near_field_table = {}
        if isinstance(near_field_table, list):
            assert len(self.code.out_kernels) == 1
            self.near_field_table[self.code.out_kernels[0].__repr__()] = \
                    near_field_table
            self.n_tables = len(near_field_table)

        # FIXME
        elif isinstance(near_field_table, NearFieldInteractionTable):
            assert len(self.code.out_kernels) == 1
            self.near_field_table[self.code.out_kernels[0].__repr__()] = \
                    [near_field_table]
            self.n_tables = 1

        elif isinstance(near_field_table, dict):
            assert len(self.code.out_kernels) <= len(near_field_table)
            self.near_field_table = near_field_table
            self.n_tables = len(
                    near_field_table[self.code.out_kernels[0].__repr__()])

        else:
            raise RuntimeError("Table type unrecognized.")

        # table -- kernel consistency check
        for kid in range(len(self.code.out_kernels)):
            kname = self.code.out_kernels[kid].__repr__()
            assert kname in self.near_field_table
            # n_tables = minimum length
            if len(self.near_field_table[kname]) < self.n_tables:
                self.n_tables = len(self.near_field_table[kname])
            assert self.n_tables > 0

        self.quad_order = quad_order

        self.root_table_source_box_extent = \
                self.near_field_table[kname][0].source_box_extent
        table_starting_level = np.round(np.log(
                self.tree.root_extent / self.root_table_source_box_extent) /
                np.log(2))
        for kid in range(len(self.code.out_kernels)):
            kname = self.code.out_kernels[kid].__repr__()
            for lev, table in zip(range(len(self.near_field_table[kname])),
                    self.near_field_table[kname]):
                assert table.quad_order == self.quad_order

                if not table.is_built:
                    raise RuntimeError(
                            "Near field interaction table needs to be built "
                            "prior to being used")

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
                                "source_box_extent of the root table must "
                                "divide the bounding box's extent by an integer.")

            if self.n_tables > 1:
                # this checks that the boxes at the highest level are covered
                if not tree.nlevels <= len(
                        self.near_field_table[kname]) + table_starting_level:
                    raise RuntimeError("Insufficient list of tables: the "
                            "finest level mesh cells at level "
                            + str(tree.nlevels) + " are not covered.")

                # the check that the boxes at the coarsest level are covered is
                # deferred until trav.target_boxes is passed when invoking
                # eval_direct

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

    def eval_direct_single_out_kernel(self, out_pot, out_kernel,
            target_boxes, neighbor_source_boxes_starts,
            neighbor_source_boxes_lists, mode_coefs):

        # NOTE: mode_coefs are similar to source_weights BUT
        # do not include quadrature weights (purely function
        # expansiona coefficients)

        # NOTE: inputs should be all on the device main memory (cl.Arrays)

        kname = out_kernel.__repr__()

        if self.n_tables > 1:
            # this checks that the boxes at the coarsest level
            # and allows for some round-off error
            min_lev = np.min(self.tree.box_levels.get(self.queue)[
                target_boxes.get(self.queue)])
            largest_cell_extent = self.tree.root_extent * 0.5**min_lev
            if not self.near_field_table[kname][0].source_box_extent >= (
                    largest_cell_extent - 1e-15):
                raise RuntimeError("Insufficient list of tables: the "
                        "coarsest level mesh cells at level "
                        + str(min_lev) + " are not covered.")

        # table.case_encode
        distinct_numbers = set()
        for vec in self.near_field_table[kname][0].interaction_case_vecs:
            for l in vec:
                distinct_numbers.add(l)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        case_indices_dev = cl.array.to_device(self.queue,
                self.near_field_table[kname][0].case_indices)

        # table.data
        table_data_combined = np.zeros(
                (len(self.near_field_table[kname]),
                    len(self.near_field_table[kname][0].data)))
        mode_nmlz_combined = np.zeros(
                (len(self.near_field_table[kname]),
                    len(self.near_field_table[kname][0].mode_normalizers)))
        for lev in range(len(self.near_field_table[kname])):
            table_data_combined[lev, :] = self.near_field_table[kname][lev].data
            mode_nmlz_combined[lev, :] = \
                    self.near_field_table[kname][lev].mode_normalizers

        logger.info("Table data for kernel "
                + out_kernel.__repr__() + " congregated")

        # The loop domain needs to know some info about the tables being used
        table_data_shapes = {
                'n_tables': self.n_tables,
                'n_q_points': self.near_field_table[kname][0].n_q_points,
                'n_table_entries': len(self.near_field_table[kname][0].data)
                }
        assert table_data_shapes['n_q_points'] == len(
                self.near_field_table[kname][0].mode_normalizers)

        from volumential.list1 import NearFieldFromCSR
        near_field = NearFieldFromCSR(out_kernel, table_data_shapes)

        res = near_field(self.queue,
                result=out_pot,
                box_centers=self.tree.box_centers,
                box_levels=self.tree.box_levels,
                box_source_counts_cumul=self.tree.box_source_counts_cumul,
                box_source_starts=self.tree.box_source_starts,
                box_target_counts_cumul=self.tree.box_target_counts_cumul,
                box_target_starts=self.tree.box_target_starts,
                case_indices=case_indices_dev,
                encoding_base=base,
                encoding_shift=shift,
                mode_nmlz_combined=mode_nmlz_combined,
                neighbor_source_boxes_starts=neighbor_source_boxes_starts,
                root_extent=self.tree.root_extent,
                neighbor_source_boxes_lists=neighbor_source_boxes_lists,
                mode_coefs=mode_coefs,
                table_data_combined=table_data_combined,
                target_boxes=target_boxes,
                table_root_extent=self.root_table_source_box_extent)

        #print(near_field.get_kernel())
        #import pudb; pu.db

        assert res is out_pot

        # sorted_target_ids=self.tree.user_source_ids,
        # user_source_ids=self.tree.user_source_ids)

        return out_pot

    def eval_direct(self, target_boxes, neighbor_source_boxes_starts,
                    neighbor_source_boxes_lists, mode_coefs):
        pot = self.output_zeros()
        for i in range(len(self.code.out_kernels)):
            # print("processing near-field of out_kernel", i)
            self.eval_direct_single_out_kernel(pot[i], self.code.out_kernels[i],
                    target_boxes,
                    neighbor_source_boxes_starts,
                    neighbor_source_boxes_lists,
                    mode_coefs)

        for out_pot in pot:
            out_pot.finish()

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

    def eval_multipoles(self, target_boxes_by_source_level,
                        source_boxes_by_level, mpole_exps):
        return SumpyExpansionWrangler.eval_multipoles(self,
                target_boxes_by_source_level,
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
