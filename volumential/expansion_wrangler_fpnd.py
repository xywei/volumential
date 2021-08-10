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
import logging
import pyopencl as cl
import pyopencl.array
from pytools.obj_array import make_obj_array

# from pytools import memoize_method
from volumential.nearfield_potential_table import NearFieldInteractionTable
from volumential.expansion_wrangler_interface import (
        ExpansionWranglerInterface, ExpansionWranglerCodeContainerInterface)
from sumpy.fmm import SumpyExpansionWrangler, \
        SumpyTimingFuture, SumpyExpansionWranglerCodeContainer
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler

from sumpy.kernel import (
        LaplaceKernel, HelmholtzKernel, AxisTargetDerivative,
        DirectionalSourceDerivative)

logger = logging.getLogger(__name__)


def level_to_rscale(tree, level):
    return tree.root_extent * (2 ** -level)


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


# {{{ sumpy backend


class FPNDSumpyExpansionWranglerCodeContainer(
        ExpansionWranglerCodeContainerInterface,
        SumpyExpansionWranglerCodeContainer):
    """Objects of this type serve as a place to keep the code needed
    for ExpansionWrangler if it is using sumpy to perform multipole
    expansion and manipulations.

    Since :class:`SumpyExpansionWrangler` necessarily must have a
    :class:`pyopencl.CommandQueue`, but this queue is allowed to be
    more ephemeral than the code, the code's lifetime
    is decoupled by storing it in this object.
    """
    get_wrangler = SumpyExpansionWranglerCodeContainer.get_wrangler


class FPNDSumpyExpansionWrangler(
        ExpansionWranglerInterface, SumpyExpansionWrangler):
    """This expansion wrangler uses "fpnd" strategy. That is, Far field is
    computed via Particle approximation and Near field is computed Directly.
    The FMM is performed using sumpy backend.

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
        code_container,
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
        translation_classes_data=None,
    ):
        """
        near_field_table can either one of three things:
            1. a single table, when len(target_kernels) = 1 (single level)
            2. a list of tables, when len(target_kernels) = 1 (multiple levels)
            3. otherwise, a dictionary from kernel.__repr__() to a list of its tables
        """
        if source_extra_kwargs is None:
            source_extra_kwargs = {}

        super().__init__(
            code_container, queue, tree, dtype, fmm_level_to_order,
            source_extra_kwargs, kernel_extra_kwargs, translation_classes_data)

        self.near_field_table = {}
        # list of tables for a single out kernel
        if isinstance(near_field_table, list):
            assert len(self.code.target_kernels) == 1
            self.near_field_table[
                self.code.target_kernels[0].__repr__()
            ] = near_field_table
            self.n_tables = len(near_field_table)

        # single table
        elif isinstance(near_field_table, NearFieldInteractionTable):
            assert len(self.code.target_kernels) == 1
            self.near_field_table[self.code.target_kernels[0].__repr__()] = [
                near_field_table
            ]
            self.n_tables = 1

        # dictionary of lists of tables
        elif isinstance(near_field_table, dict):
            self.n_tables = dict()
            for out_knl in self.code.target_kernels:
                if repr(out_knl) not in near_field_table:
                    raise RuntimeError(
                            "Missing nearfield table for %s." % repr(out_knl))
                if isinstance(near_field_table[repr(out_knl)],
                        NearFieldInteractionTable):
                    near_field_table[repr(out_knl)] = [
                            near_field_table[repr(out_knl)]]
                else:
                    assert isinstance(near_field_table[repr(out_knl)], list)

                self.n_tables[repr(out_knl)] = len(near_field_table[repr(out_knl)])

            self.near_field_table = near_field_table
        else:
            raise RuntimeError("Table type unrecognized.")

        self.quad_order = quad_order
        self.potential_kind = potential_kind

        # TODO: make all parameters table-specific (allow using inhomogeneous tables)
        kname = repr(self.code.target_kernels[0])
        self.root_table_source_box_extent = (
                self.near_field_table[kname][0].source_box_extent)
        table_starting_level = np.round(
            np.log(self.tree.root_extent / self.root_table_source_box_extent)
            / np.log(2)
            )
        for kid in range(len(self.code.target_kernels)):
            kname = self.code.target_kernels[kid].__repr__()
            for lev, table in zip(
                    range(len(self.near_field_table[kname])),
                    self.near_field_table[kname]
                    ):
                assert table.quad_order == self.quad_order

                if not table.is_built:
                    raise RuntimeError(
                        "Near field interaction table needs to be built "
                        "prior to being used"
                    )

                table_root_extent = table.source_box_extent * 2 ** lev
                assert (
                    abs(self.root_table_source_box_extent - table_root_extent)
                    < 1e-15
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

        if list1_extra_kwargs is None:
            list1_extra_kwargs = {}
        self.list1_extra_kwargs = list1_extra_kwargs

        # print("Multipole order = ",self.level_orders)

    # }}} End constructor

    # {{{ data vector utilities

    def multipole_expansion_zeros(self):
        return SumpyExpansionWrangler.multipole_expansion_zeros(self)

    def local_expansion_zeros(self):
        return SumpyExpansionWrangler.local_expansion_zeros(self)

    def output_zeros(self):
        return SumpyExpansionWrangler.output_zeros(self)

    def reorder_sources(self, source_array):
        return SumpyExpansionWrangler.reorder_sources(self, source_array)

    def reorder_targets(self, target_array):
        if not hasattr(self.tree, 'user_target_ids'):
            self.tree.user_target_ids = inverse_id_map(
                self.queue, self.tree.sorted_target_ids)
        return target_array.with_queue(self.queue)[self.tree.user_target_ids]

    def reorder_potentials(self, potentials):
        return SumpyExpansionWrangler.reorder_potentials(self, potentials)

    def finalize_potentials(self, potentials):
        # return potentials
        return SumpyExpansionWrangler.finalize_potentials(self, potentials)

    # }}} End data vector utilities

    # {{{ formation & coarsening of multipoles

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        return SumpyExpansionWrangler.form_multipoles(
            self, level_start_source_box_nrs, source_boxes, src_weights
        )

    def coarsen_multipoles(
        self, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
    ):
        return SumpyExpansionWrangler.coarsen_multipoles(
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
            out_pot[:] = cl.array.to_device(self.queue, np.arange(len(out_pot)))
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
            largest_cell_extent = self.tree.root_extent * 0.5 ** min_lev
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

        # table.data
        table_data_combined = np.zeros(
            (
                len(self.near_field_table[kname]),
                len(self.near_field_table[kname][0].data),
            ), dtype=self.near_field_table[kname][0].data.dtype
        )
        mode_nmlz_combined = np.zeros(
            (
                len(self.near_field_table[kname]),
                len(self.near_field_table[kname][0].mode_normalizers),
            ), dtype=self.near_field_table[kname][0].mode_normalizers.dtype
        )
        exterior_mode_nmlz_combined = np.zeros(
            (
                len(self.near_field_table[kname]),
                len(self.near_field_table[kname][0].kernel_exterior_normalizers),
            ),
            dtype=self.near_field_table[kname][0].kernel_exterior_normalizers.dtype
        )
        for lev in range(len(self.near_field_table[kname])):
            table_data_combined[lev, :] = self.near_field_table[kname][lev].data
            mode_nmlz_combined[lev, :] = \
                self.near_field_table[kname][lev].mode_normalizers
            exterior_mode_nmlz_combined[lev, :] = \
                self.near_field_table[kname][lev].kernel_exterior_normalizers

        self.queue.finish()
        logger.info(
                "table data for kernel "
                + out_kernel.__repr__() + " congregated")

        # The loop domain needs to know some info about the tables being used
        table_data_shapes = {
            "n_tables": len(self.near_field_table[kname]),
            "n_q_points": self.near_field_table[kname][0].n_q_points,
            "n_table_entries": len(self.near_field_table[kname][0].data),
        }
        assert table_data_shapes["n_q_points"] == len(
            self.near_field_table[kname][0].mode_normalizers
        )

        from volumential.list1 import NearFieldFromCSR

        near_field = NearFieldFromCSR(out_kernel, table_data_shapes,
            potential_kind=self.potential_kind,
            **self.list1_extra_kwargs)

        table_data_combined = cl.array.to_device(self.queue,
                table_data_combined)
        mode_nmlz_combined = cl.array.to_device(self.queue,
                mode_nmlz_combined)
        exterior_mode_nmlz_combined = cl.array.to_device(self.queue,
            exterior_mode_nmlz_combined)
        self.queue.finish()
        logger.info("sent table data to device")

        # NOTE: box_sources for this evaluation should be "box_targets".
        # This is due to the special features of how box-FMM works.

        res, evt = near_field(
            self.queue,
            result=out_pot,
            box_centers=self.tree.box_centers,
            box_levels=self.tree.box_levels,
            box_source_counts_cumul=self.tree.box_target_counts_cumul,
            box_source_starts=self.tree.box_target_starts,
            box_target_counts_cumul=self.tree.box_target_counts_cumul,
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
            table_data_combined=table_data_combined,
            target_boxes=target_boxes,
            table_root_extent=self.root_table_source_box_extent,
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
        for i in range(len(self.code.target_kernels)):
            # print("processing near-field of out_kernel", i)
            pot[i], evt = self.eval_direct_single_out_kernel(
                pot[i],
                self.code.target_kernels[i],
                target_boxes,
                neighbor_source_boxes_starts,
                neighbor_source_boxes_lists,
                mode_coefs,
            )
            events.append(evt)

        for out_pot in pot:
            out_pot.finish()

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
        return SumpyExpansionWrangler.multipole_to_local(
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
        return SumpyExpansionWrangler.eval_multipoles(
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
        return SumpyExpansionWrangler.form_locals(
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
        return SumpyExpansionWrangler.refine_locals(
            self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        )

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        return SumpyExpansionWrangler.eval_locals(
            self, level_start_target_box_nrs, target_boxes, local_exps
        )

    # }}} End downward pass of fmm

    # {{{ direct evaluation of p2p (discrete) interactions

    def eval_direct_p2p(
        self, target_boxes, source_box_starts, source_box_lists, src_weights
    ):
        return SumpyExpansionWrangler.eval_direct(
            self, target_boxes, source_box_starts, source_box_lists, src_weights
        )

    # }}} End direct evaluation of p2p interactions

# }}} End sumpy backend

# {{{ fmmlib backend (for laplace, helmholtz)


class FPNDFMMLibExpansionWranglerCodeContainer(
        ExpansionWranglerCodeContainerInterface,
        ):
    """Objects of this type serve as a place to keep the code needed
    for ExpansionWrangler if it is using fmmlib to perform multipole
    expansion and manipulations.

    The interface is augmented with unecessary arguments acting as
    placeholders, such that it can be a drop-in replacement of sumpy
    backend.
    """
    def __init__(self, cl_context,
            multipole_expansion_factory, local_expansion_factory,
            target_kernels, exclude_self=True, *args, **kwargs):
        self.cl_context = cl_context
        self.multipole_expansion_factory = multipole_expansion_factory
        self.local_expansion_factory = local_expansion_factory

        self.target_kernels = target_kernels
        self.exclude_self = True

    def get_wrangler(self, queue, tree, dtype, fmm_level_to_order,
            source_extra_kwargs={}, kernel_extra_kwargs=None,
            *args, **kwargs):
        return FPNDFMMLibExpansionWrangler(self, queue, tree,
                dtype, fmm_level_to_order,
                source_extra_kwargs, kernel_extra_kwargs,
                *args, **kwargs)


class FPNDFMMLibExpansionWrangler(
        ExpansionWranglerInterface, FMMLibExpansionWrangler):
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

    def __init__(self, code_container, queue, tree,
            near_field_table, dtype,
            fmm_level_to_order,
            quad_order,
            potential_kind=1,
            source_extra_kwargs=None,
            kernel_extra_kwargs=None,
            self_extra_kwargs=None,
            list1_extra_kwargs=None,
            *args, **kwargs):
        self.code = code_container
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

        for out_knl in self.code.target_kernels:

            if self.is_supported_helmknl(out_knl):
                outputs.append(())
                no_target_deriv_knl = out_knl

            elif (isinstance(out_knl, AxisTargetDerivative)
                    and self.is_supported_helmknl(out_knl.inner_kernel)):
                outputs.append((out_knl.axis,))
                ifgrad = True
                no_target_deriv_knl = out_knl.inner_kernel

            else:
                raise ValueError(
                        "only the 2/3D Laplace and Helmholtz kernel "
                        "and their derivatives are supported")

            source_deriv_names.append(no_target_deriv_knl.dir_vec_name
                    if isinstance(no_target_deriv_knl, DirectionalSourceDerivative)
                    else None)

            base_knl = out_knl.get_base_kernel()
            k_names.append(base_knl.helmholtz_k_name
                    if isinstance(base_knl, HelmholtzKernel)
                    else None)

        self.outputs = outputs

        from pytools import is_single_valued

        if not is_single_valued(source_deriv_names):
            raise ValueError("not all kernels passed are the same in "
                    "whether they represent a source derivative")

        source_deriv_name = source_deriv_names[0]

        if not is_single_valued(k_names):
            raise ValueError("not all kernels passed have the same "
                    "Helmholtz parameter")

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
            assert len(self.code.target_kernels) == 1
            self.near_field_table[
                self.code.target_kernels[0].__repr__()
            ] = near_field_table
            self.n_tables = len(near_field_table)

        # single table
        elif isinstance(near_field_table, NearFieldInteractionTable):
            assert len(self.code.target_kernels) == 1
            self.near_field_table[self.code.target_kernels[0].__repr__()] = [
                near_field_table
            ]
            self.n_tables = 1

        # dictionary of lists of tables
        elif isinstance(near_field_table, dict):
            self.n_tables = dict()
            for out_knl in self.code.target_kernels:
                if repr(out_knl) not in near_field_table:
                    raise RuntimeError(
                            "Missing nearfield table for %s." % repr(out_knl))
                if isinstance(near_field_table[repr(out_knl)],
                        NearFieldInteractionTable):
                    near_field_table[repr(out_knl)] = [
                            near_field_table[repr(out_knl)]]
                else:
                    assert isinstance(near_field_table[repr(out_knl)], list)

                self.n_tables[repr(out_knl)] = len(near_field_table[repr(out_knl)])

            self.near_field_table = near_field_table
        else:
            raise RuntimeError("Table type unrecognized.")

        # TODO: make all parameters table-specific (allow using inhomogeneous tables)
        kname = repr(self.code.target_kernels[0])
        self.root_table_source_box_extent = (
                self.near_field_table[kname][0].source_box_extent)
        table_starting_level = np.round(
            np.log(self.tree.root_extent / self.root_table_source_box_extent)
            / np.log(2)
            )
        for kid in range(len(self.code.target_kernels)):
            kname = self.code.target_kernels[kid].__repr__()
            for lev, table in zip(
                    range(len(self.near_field_table[kname])),
                    self.near_field_table[kname]
                    ):
                assert table.quad_order == self.quad_order

                if not table.is_built:
                    raise RuntimeError(
                        "Near field interaction table needs to be built "
                        "prior to being used"
                    )

                table_root_extent = table.source_box_extent * 2 ** lev
                assert (
                    abs(self.root_table_source_box_extent - table_root_extent)
                    < 1e-15
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

        # }}} End table setup

        if not callable(fmm_level_to_order):
            raise TypeError("fmm_level_to_order not passed")

        dipole_vec = None
        if source_deriv_name is not None:
            dipole_vec = np.array([
                    d_i.get(queue=queue)
                    for d_i in source_extra_kwargs[source_deriv_name]],
                    order="F")

        def inner_fmm_level_to_nterms(tree, level):
            if helmholtz_k == 0:
                return fmm_level_to_order(
                        LaplaceKernel(tree.dimensions),
                        frozenset(), tree, level)
            else:
                return fmm_level_to_order(
                        HelmholtzKernel(tree.dimensions),
                        frozenset([("k", helmholtz_k)]), tree, level)

        rotation_data = None
        if 'traversal' in kwargs:
            # add rotation data if traversal is passed as a keyword argument
            from boxtree.pyfmmlib_integration import FMMLibRotationData
            rotation_data = FMMLibRotationData(self.queue, kwargs['traversal'])
        else:
            logger.warning("Rotation data is not utilized since traversal is "
                           "not known to FPNDFMMLibExpansionWrangler.")

        FMMLibExpansionWrangler.__init__(
                self, tree,

                helmholtz_k=helmholtz_k,
                dipole_vec=dipole_vec,
                dipoles_already_reordered=True,

                fmm_level_to_nterms=inner_fmm_level_to_nterms,
                rotation_data=rotation_data,

                ifgrad=ifgrad)

    # }}} End constructor

# {{{ scale factor for fmmlib

    def get_scale_factor(self):
        if self.eqn_letter == "l" and self.dim == 2:
            scale_factor = -1/(2*np.pi)
        elif self.eqn_letter == "h" and self.dim == 2:
            scale_factor = 1
        elif self.eqn_letter in ["l", "h"] and self.dim == 3:
            scale_factor = 1/(4*np.pi)
        else:
            raise NotImplementedError(
                    "scale factor for pyfmmlib %s for %d dimensions" % (
                        self.eqn_letter,
                        self.dim))

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
        if not hasattr(self.tree, 'user_target_ids'):
            self.tree.user_target_ids = inverse_id_map(
                self.queue, self.tree.sorted_target_ids)
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
            largest_cell_extent = self.tree.root_extent * 0.5 ** min_lev
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

        # table.data
        table_data_combined = np.zeros(
            (
                len(self.near_field_table[kname]),
                len(self.near_field_table[kname][0].data),
            ), dtype=self.near_field_table[kname][0].data.dtype
        )
        mode_nmlz_combined = np.zeros(
            (
                len(self.near_field_table[kname]),
                len(self.near_field_table[kname][0].mode_normalizers),
            ), dtype=self.near_field_table[kname][0].mode_normalizers.dtype
        )
        for lev in range(len(self.near_field_table[kname])):
            table_data_combined[lev, :] = self.near_field_table[kname][lev].data
            mode_nmlz_combined[lev, :] = self.near_field_table[kname][
                lev
            ].mode_normalizers
        exterior_mode_nmlz_combined = np.zeros(
            (
                len(self.near_field_table[kname]),
                len(self.near_field_table[kname][0].kernel_exterior_normalizers),
            ),
            dtype=self.near_field_table[kname][0].kernel_exterior_normalizers.dtype
        )
        for lev in range(len(self.near_field_table[kname])):
            table_data_combined[lev, :] = self.near_field_table[kname][lev].data
            mode_nmlz_combined[lev, :] = \
                self.near_field_table[kname][lev].mode_normalizers
            exterior_mode_nmlz_combined[lev, :] = \
                self.near_field_table[kname][lev].kernel_exterior_normalizers

        logger.info(
                "Table data for kernel "
                + out_kernel.__repr__() + " congregated")

        # The loop domain needs to know some info about the tables being used
        table_data_shapes = {
            "n_tables": len(self.near_field_table[kname]),
            "n_q_points": self.near_field_table[kname][0].n_q_points,
            "n_table_entries": len(self.near_field_table[kname][0].data),
        }
        assert table_data_shapes["n_q_points"] == len(
            self.near_field_table[kname][0].mode_normalizers
        )

        from volumential.list1 import NearFieldFromCSR

        near_field = NearFieldFromCSR(out_kernel, table_data_shapes,
            potential_kind=self.potential_kind,
            **self.list1_extra_kwargs)

        res, evt = near_field(
            self.queue,
            result=out_pot,
            box_centers=self.tree.box_centers,
            box_levels=self.tree.box_levels,
            box_source_counts_cumul=self.tree.box_target_counts_cumul,
            box_source_starts=self.tree.box_target_starts,
            box_target_counts_cumul=self.tree.box_target_counts_cumul,
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
            table_data_combined=table_data_combined,
            target_boxes=target_boxes,
            table_root_extent=self.root_table_source_box_extent,
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
        if pot.dtype != np.object:
            pot = make_obj_array([pot, ])
        events = []
        for i in range(len(self.code.target_kernels)):
            # print("processing near-field of out_kernel", i)
            pot[i], evt = self.eval_direct_single_out_kernel(
                pot[i],
                self.code.target_kernels[i],
                target_boxes,
                neighbor_source_boxes_starts,
                neighbor_source_boxes_lists,
                mode_coefs,
            )
            events.append(evt)

        for out_pot in pot:
            if isinstance(out_pot, cl.array.Array):
                out_pot.finish()

        # boxtree.pyfmmlib_integration handles things diffferently
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

        return (isinstance(knl, (LaplaceKernel, HelmholtzKernel))
                and knl.dim in (2, 3))


# }}} End fmmlib backend (for laplace, helmholtz)


class FPNDExpansionWranglerCodeContainer(FPNDSumpyExpansionWranglerCodeContainer):
    """The default code container.
    """


class FPNDExpansionWrangler(FPNDSumpyExpansionWrangler):
    """The default wrangler class.
    """

# vim: filetype=pyopencl:foldmethod=marker
