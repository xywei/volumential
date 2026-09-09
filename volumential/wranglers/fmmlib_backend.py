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

__doc__ = """The pyfmmlib-based fpnd expansion wrangler.

Same fpnd strategy as :mod:`volumential.wranglers.sumpy_backend`, but the
far field runs through :mod:`boxtree.pyfmmlib_integration`.  Restricted to
the Laplace and Helmholtz kernels that pyfmmlib supports.
"""

from collections import OrderedDict

import numpy as np

import pyopencl as cl
import pyopencl.array
from boxtree.pyfmmlib_integration import (
    FMMLibExpansionWrangler,
    FMMLibTreeIndependentDataForWrangler,
    Kernel as FMMLibKernel,
)
from pytools.obj_array import new_1d as obj_array_1d
from sumpy.kernel import (
    AxisTargetDerivative,
    DirectionalSourceDerivative,
    HelmholtzKernel,
    LaplaceKernel,
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
from volumential.wranglers.device_arrays import inverse_id_map
from volumential.wranglers.fmmlib_batched import FMMLibBatchedStagesMixin
from volumential.wranglers.kernel_symmetry import (
    _extract_symmetry_source_direction,
)
from volumential.wranglers.nearfield_cache import NearFieldPayloadCacheMixin
from volumential.wranglers.table_data import _table_data_fingerprint
from volumential.wranglers.timing import SumpyTimingFuture


class FPNDFMMLibTreeIndependentDataForWrangler(
    TreeIndependentDataForWranglerInterface,
    FMMLibTreeIndependentDataForWrangler,
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

        base_kernels = [kernel.get_base_kernel() for kernel in target_kernels]
        if not base_kernels:
            raise ValueError("target_kernels must not be empty")
        if not all(type(kernel) is type(base_kernels[0]) for kernel in base_kernels):
            raise ValueError("FMMLib target kernels must share one base kernel type")

        base_kernel = base_kernels[0]
        if isinstance(base_kernel, LaplaceKernel):
            fmmlib_kernel = FMMLibKernel.LAPLACE
        elif isinstance(base_kernel, HelmholtzKernel):
            fmmlib_kernel = FMMLibKernel.HELMHOLTZ
        else:
            raise ValueError("FMMLib supports only Laplace and Helmholtz kernels")

        ifgrad = any(isinstance(kernel, AxisTargetDerivative)
                     for kernel in target_kernels)
        FMMLibTreeIndependentDataForWrangler.__init__(
            self,
            base_kernel.dim,
            fmmlib_kernel,
            ifgrad=ifgrad,
        )

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
    ) -> "FPNDFMMLibExpansionWrangler":
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


class FPNDFMMLibExpansionWrangler(
    ExpansionWranglerInterface,
    NearFieldPayloadCacheMixin,
    FMMLibBatchedStagesMixin,
    FMMLibExpansionWrangler,
):
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

        self.device_tree = tree

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
                np.log(tree.root_extent / self.root_table_source_box_extent)
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
                            int(tree.root_extent / table_root_extent)
                            * table_root_extent
                            - tree.root_extent
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

        self.source_extra_kwargs = source_extra_kwargs
        self.kernel_extra_kwargs = kernel_extra_kwargs
        self.self_extra_kwargs = self_extra_kwargs
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

        if "traversal" not in kwargs:
            raise TypeError("FMMLib wrangler requires traversal")

        from boxtree.array_context import (
            PyOpenCLArrayContext as BoxtreePyOpenCLArrayContext,
        )

        self._fmmlib_actx = BoxtreePyOpenCLArrayContext(queue)
        host_traversal = self._fmmlib_actx.to_numpy(kwargs["traversal"])

        rotation_data = None
        if tree.dimensions == 3:
            from boxtree.pyfmmlib_integration import FMMLibRotationData

            rotation_data = FMMLibRotationData(self._fmmlib_actx, host_traversal)

        FMMLibExpansionWrangler.__init__(
            self,
            tree_indep,
            host_traversal,
            helmholtz_k=helmholtz_k,
            dipole_vec=dipole_vec,
            dipoles_already_reordered=True,
            fmm_level_to_order=inner_fmm_level_to_nterms,
            rotation_data=rotation_data,
        )

    # }}} End constructor

    # {{{ scale factor for fmmlib

    def get_scale_factor(self) -> float:
        eqn_letter = self.tree_indep.eqn_letter
        if eqn_letter == "l" and self.dim == 2:
            scale_factor = -1 / (2 * np.pi)
        elif eqn_letter == "h" and self.dim == 2:
            scale_factor = 1
        elif eqn_letter in ["l", "h"] and self.dim == 3:
            scale_factor = 1 / (4 * np.pi)
        else:
            raise NotImplementedError(
                "scale factor for pyfmmlib %s for %d dimensions"
                % (eqn_letter, self.dim)
            )

        return scale_factor

    # }}} End scale factor for fmmlib

    # {{{ data vector utilities

    def multipole_expansion_zeros(self) -> FMMArray:
        return FMMLibExpansionWrangler.multipole_expansion_zeros(self)

    def local_expansion_zeros(self) -> FMMArray:
        return FMMLibExpansionWrangler.local_expansion_zeros(self)

    def output_zeros(self) -> FMMArray:
        return FMMLibExpansionWrangler.output_zeros(self)

    def reorder_sources(self, source_array: FMMArray) -> FMMArray:
        return FMMLibExpansionWrangler.reorder_sources(self, source_array)

    def reorder_targets(self, target_array: FMMArray) -> FMMArray:
        if not hasattr(self.tree, "user_target_ids"):
            self.tree.user_target_ids = inverse_id_map(
                self.queue, self.tree.sorted_target_ids
            )
        return target_array[self.tree.user_target_ids]

    def reorder_potentials(self, potentials: FMMArray) -> FMMArray:
        return FMMLibExpansionWrangler.reorder_potentials(self, potentials)

    def finalize_potentials(self, potentials: FMMArray) -> FMMArray:
        # return potentials
        return FMMLibExpansionWrangler.finalize_potentials(
            self, self._fmmlib_actx, potentials
        )

    # }}} End data vector utilities

    # {{{ formation & coarsening of multipoles

    def form_multipoles(
        self,
        level_start_source_box_nrs: BoxIndexArray,
        source_boxes: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
        formmp_imany = self._get_batched_formmp_routine()
        if formmp_imany is not None:
            result = self._form_multipoles_batched(
                formmp_imany,
                level_start_source_box_nrs,
                source_boxes,
                src_weights,
            )
        else:
            result = FMMLibExpansionWrangler.form_multipoles(
                self,
                self._fmmlib_actx,
                level_start_source_box_nrs,
                source_boxes,
                src_weights,
            )
        return result, None

    def coarsen_multipoles(
        self,
        level_start_source_parent_box_nrs: BoxIndexArray,
        source_parent_boxes: BoxIndexArray,
        mpoles: FMMArray,
    ) -> StageResult:
        result = FMMLibExpansionWrangler.coarsen_multipoles(
            self,
            self._fmmlib_actx,
            level_start_source_parent_box_nrs,
            source_parent_boxes,
            mpoles,
        )
        return result, None

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
    ) -> tuple[FMMArray, object]:

        # NOTE: mode_coefs are similar to source_weights BUT
        # do not include quadrature weights (purely function
        # expansiona coefficients)

        tree = self.device_tree
        output_is_device = isinstance(out_pot, cl.array.Array)

        def to_device(array):
            if isinstance(array, cl.array.Array):
                return array
            return cl.array.to_device(self.queue, np.ascontiguousarray(array))

        out_pot = to_device(out_pot)
        target_boxes = to_device(target_boxes)
        neighbor_source_boxes_starts = to_device(neighbor_source_boxes_starts)
        neighbor_source_boxes_lists = to_device(neighbor_source_boxes_lists)
        mode_coefs = to_device(mode_coefs)

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
                tree.box_levels.get(self.queue)[target_boxes.get(self.queue)]
            )
            largest_cell_extent = tree.root_extent * 0.5**min_lev
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
        table_data_shapes = payload["table_data_shapes"]
        assert table_data_shapes["n_q_points"] == len(
            self.near_field_table[kname][0].mode_normalizers
        )
        reconstruction_kind = table_data_shapes.get("reconstruction_kind", "dense")

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
        particle_local_ids = _compute_box_local_ids(
            self.queue, tree, self.near_field_table[kname][0].n_q_points
        )

        _validate_table_box_particle_layout_cached(
            self.queue,
            tree,
            target_boxes,
            neighbor_source_boxes_lists,
            self.near_field_table[kname][0].n_q_points,
            self._table_layout_validation_cache,
        )

        aligned_nboxes = tree.box_centers.shape[1]
        source_counts_nonchild = np.zeros(aligned_nboxes, dtype=np.int32)
        target_counts_nonchild = np.zeros(aligned_nboxes, dtype=np.int32)
        source_counts_nonchild[: len(tree.box_target_counts_nonchild)] = (
            tree.box_target_counts_nonchild.get(self.queue)
        )
        target_counts_nonchild[: len(tree.box_target_counts_nonchild)] = (
            tree.box_target_counts_nonchild.get(self.queue)
        )
        source_counts_nonchild = cl.array.to_device(self.queue, source_counts_nonchild)
        target_counts_nonchild = cl.array.to_device(self.queue, target_counts_nonchild)

        res, evt = near_field(
            self.queue,
            result=out_pot,
            box_centers=tree.box_centers,
            box_levels=tree.box_levels,
            box_source_counts_nonchild=source_counts_nonchild,
            box_source_starts=tree.box_target_starts,
            box_target_counts_nonchild=target_counts_nonchild,
            box_target_starts=tree.box_target_starts,
            case_indices=case_indices_dev,
            encoding_base=base,
            encoding_shift=shift,
            mode_nmlz_combined=mode_nmlz_combined,
            exterior_mode_nmlz_combined=exterior_mode_nmlz_combined,
            neighbor_source_boxes_starts=neighbor_source_boxes_starts,
            root_extent=tree.root_extent,
            neighbor_source_boxes_lists=neighbor_source_boxes_lists,
            mode_coefs=mode_coefs,
            source_mode_ids=particle_local_ids,
            table_data_combined=table_data_combined,
            target_boxes=target_boxes,
            target_point_ids=particle_local_ids,
            table_root_extent=self.root_table_source_box_extent,
            table_starting_level=self.table_starting_level,
            **reconstruction_kwargs,
        )

        if output_is_device:
            assert res is out_pot
            # FIXME: lazy evaluation sometimes returns incorrect results
            res.finish()
        else:
            out_pot = res.get(self.queue)

        # sorted_target_ids=self.tree.user_source_ids,
        # user_source_ids=self.tree.user_source_ids)

        scale_factor = self.get_scale_factor()
        return out_pot / scale_factor, evt

    def eval_direct(
        self,
        target_boxes: BoxIndexArray,
        neighbor_source_boxes_starts: BoxIndexArray,
        neighbor_source_boxes_lists: BoxIndexArray,
        mode_coefs: FMMArray,
    ) -> StageResult:
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
        level_start_target_box_nrs: BoxIndexArray,
        target_boxes: BoxIndexArray,
        src_box_starts: BoxIndexArray,
        src_box_lists: BoxIndexArray,
        mpole_exps: FMMArray,
    ) -> StageResult:
        result = FMMLibExpansionWrangler.multipole_to_local(
            self,
            self._fmmlib_actx,
            level_start_target_box_nrs,
            target_boxes,
            src_box_starts,
            src_box_lists,
            mpole_exps,
        )
        return result, None

    def eval_multipoles(
        self,
        target_boxes_by_source_level: BoxIndexArray,
        source_boxes_by_level: BoxIndexArray,
        mpole_exps: FMMArray,
    ) -> StageResult:
        result = FMMLibExpansionWrangler.eval_multipoles(
            self,
            self._fmmlib_actx,
            target_boxes_by_source_level,
            source_boxes_by_level,
            mpole_exps,
        )
        return result, None

    def form_locals(
        self,
        level_start_target_or_target_parent_box_nrs: BoxIndexArray,
        target_or_target_parent_boxes: BoxIndexArray,
        starts: BoxIndexArray,
        lists: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
        result = FMMLibExpansionWrangler.form_locals(
            self,
            self._fmmlib_actx,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts,
            lists,
            src_weights,
        )
        return result, None

    def refine_locals(
        self,
        level_start_target_or_target_parent_box_nrs: BoxIndexArray,
        target_or_target_parent_boxes: BoxIndexArray,
        local_exps: FMMArray,
    ) -> StageResult:
        result = FMMLibExpansionWrangler.refine_locals(
            self,
            self._fmmlib_actx,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps,
        )
        return result, None

    def eval_locals(
        self,
        level_start_target_box_nrs: BoxIndexArray,
        target_boxes: BoxIndexArray,
        local_exps: FMMArray,
    ) -> StageResult:
        if self._gemm_l2p_supported():
            result = self._eval_locals_gemm(
                level_start_target_box_nrs, target_boxes, local_exps
            )
        else:
            result = FMMLibExpansionWrangler.eval_locals(
                self,
                self._fmmlib_actx,
                level_start_target_box_nrs,
                target_boxes,
                local_exps,
            )
        return result, None

    # }}} End downward pass of fmm

    # {{{ direct evaluation of p2p (discrete) interactions

    def eval_direct_p2p(
        self,
        target_boxes: BoxIndexArray,
        source_box_starts: BoxIndexArray,
        source_box_lists: BoxIndexArray,
        src_weights: FMMArray,
    ) -> StageResult:
        result = FMMLibExpansionWrangler.eval_direct(
            self,
            self._fmmlib_actx,
            target_boxes,
            source_box_starts,
            source_box_lists,
            src_weights,
        )
        return result, None

    # }}} End direct evaluation of p2p interactions

    @staticmethod
    def is_supported_helmknl(knl) -> bool:
        if isinstance(knl, DirectionalSourceDerivative):
            knl = knl.inner_kernel

        return isinstance(knl, (LaplaceKernel, HelmholtzKernel)) and knl.dim in (2, 3)

# vim: filetype=pyopencl:foldmethod=marker
