from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__doc__ = """
.. autoclass:: NearFieldInteractionTableManager
   :members:
"""

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

import logging

import h5py as hdf
import numpy as np

import volumential as vm

# from sumpy.expansion.local import VolumeTaylorLocalExpansion
# from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
# from sumpy.kernel import LaplaceKernel
from volumential.nearfield_potential_table import NearFieldInteractionTable

logger = logging.getLogger(__name__)

# {{{ constant sumpy kernel

from sumpy.kernel import ExpressionKernel


class ConstantKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self, dim=None):

        expr = 1
        scaling = 1

        super(ConstantKernel, self).__init__(
            dim, expression=expr,
            global_scaling_const=scaling, is_complex_valued=False
        )

    has_efficient_scale_adjustment = True

    def adjust_for_kernel_scaling(self, expr, rscale, nderivatives):
        return expr / rscale

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CstKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


# }}} End constant sumpy kernel

# {{{ table dataset manager class


class NearFieldInteractionTableManager(object):
    """
    A class that manages near field interaction table computation and
    storage.

    Tables are stored under 'Dimension/KernelName/QuadOrder/BoxLevel/dataset_name'
    e.g., '2D/Laplace/Order_1/Level_0/data'

    Only one table manager can exist for a dataset file with write access.
    The access can be controlled with the read_only argument. By default,
    the constructor tries to open the dataset with write access, and falls
    back to read-only if that fails.
    """

    def __init__(self, dataset_filename="nft.hdf5",
            root_extent=1, dtype=np.float64,
            read_only='auto', **kwargs):
        """Constructor.
        """
        self.dtype = dtype

        self.filename = dataset_filename
        self.root_extent = root_extent

        if read_only == 'auto':
            try:
                self.datafile = hdf.File(self.filename, "a")
            except (IOError, OSError) as e:
                from warnings import warn
                warn("Trying to open in read/write mode failed: %s" % str(e))
                warn("Opening table dataset %s in read-only mode." % self.filename)
                self.datafile = hdf.File(self.filename, "r", swmr=True)
        elif read_only:
            self.datafile = hdf.File(self.filename, "r", swmr=True)
        else:
            # Read/write if exists, create otherwise
            self.datafile = hdf.File(self.filename, "a")

        self.table_extra_kwargs = kwargs

        # If the file exists, it must be for the same root_extent
        if "root_extent" not in self.datafile.attrs:
            self.datafile.attrs["root_extent"] = \
                    self.root_extent
        else:
            if not abs(
                    self.datafile.attrs["root_extent"] - self.root_extent
                    ) < 1e-15:
                raise RuntimeError(
                    "The table cache file "
                    + self.filename
                    + " was built with root_extent = "
                    + str(self.datafile.attrs["root_extent"])
                    + ", which is different from the requested value "
                    + str(self.root_extent)
                )

        self.supported_kernels = [
            "Laplace",
            "Laplace-Dx",
            "Laplace-Dy",
            "Laplace-Dz",
            "Constant",
            "Yukawa",
            "Yukawa-Dx",
            "Yukawa-Dy",
            "Cahn-Hilliard",
            "Cahn-Hilliard-Laplacian",
            "Cahn-Hilliard-Dx",
            "Cahn-Hilliard-Dy",
            "Cahn-Hilliard-Laplacian-Dx",
            "Cahn-Hilliard-Laplacian-Dy",
        ]

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.datafile.__exit__(exception_type, exception_value, traceback)

    def get_table(
        self,
        dim,
        kernel_type,
        q_order,
        source_box_level=0,
        force_recompute=False,
        compute_method=None,
        queue=None,
        **kwargs
    ):
        """Primary user interface. Get the specified table regardless of how.
        In the case of a cache miss or a forced re-computation, the method
        specified in the compute_method will be used.
        """

        dim = int(dim)
        assert dim >= 1

        if compute_method is None:
            compute_method = "Transform"

        is_recomputed = False

        q_order = int(q_order)
        assert q_order >= 1

        source_box_level = int(source_box_level)
        assert source_box_level >= 0

        if str(dim) + "D" not in self.datafile:
            self.datafile.create_group(str(dim) + "D")

        grp = self.datafile[str(dim) + "D"]

        if kernel_type not in grp:
            grp.create_group(kernel_type)

        grp = grp[kernel_type]

        if "Order_" + str(q_order) not in grp:
            grp.create_group("Order_" + str(q_order))

        grp = grp["Order_" + str(q_order)]

        if "Level_" + str(source_box_level) not in grp:
            logger.info("Table cache missing. Invoking fresh computation.")
            is_recomputed = True
            grp.create_group("Level_" + str(source_box_level))
            table = self.compute_and_update_table(
                dim,
                kernel_type,
                q_order,
                source_box_level,
                compute_method,
                queue=queue,
                **kwargs
            )

        elif force_recompute:
            logger.info("Invoking fresh computation since force_recompute is set")
            is_recomputed = True
            table = self.compute_and_update_table(
                dim,
                kernel_type,
                q_order,
                source_box_level,
                compute_method,
                queue=queue,
                **kwargs
            )

        else:
            try:

                table = self.load_saved_table(
                    dim,
                    kernel_type,
                    q_order,
                    source_box_level,
                    compute_method,
                    **kwargs
                )

            except KeyError:

                import traceback

                logger.debug(traceback.format_exc())

                logger.info("Recomputing due to table data corruption.")
                is_recomputed = True
                table = self.compute_and_update_table(
                    dim,
                    kernel_type,
                    q_order,
                    source_box_level,
                    compute_method,
                    queue=queue,
                    **kwargs
                )

            # Ensure loaded table matches requirements specified in kwargs
            for kkey, kval in kwargs.items():
                if kval is not None:
                    try:
                        tbval = getattr(table, kkey)
                        if isinstance(kval, (int, str)):
                            if not kval == tbval:
                                from warnings import warn

                                warn(
                                    "Table data loaded with a different value "
                                    + kkey
                                    + " = "
                                    + str(tbval)
                                    + " (expected "
                                    + str(kval)
                                    + ")"
                                )
                        else:
                            assert isinstance(kval, (float, complex))
                            tbval = kval.__class__(tbval)
                            if not abs(kval - tbval) < 1e-12:
                                from warnings import warn

                                warn(
                                    "Table data loaded with a different value "
                                    + kkey
                                    + " = "
                                    + str(tbval)
                                    + " (expected "
                                    + str(kval)
                                    + ")"
                                )

                    except AttributeError as e:
                        strict_loading = False
                        if "debug" in kwargs:
                            if "strict_loading" in kwargs["debug"]:
                                strict_loading = kwargs["debug"]["strict_loading"]

                        if strict_loading:
                            from warnings import warn

                            warn(
                                "Consistency is not fully ensured "
                                "(kwarg specified but cannot be loaded). "
                                "NOTE: this is most likely due to non-standard "
                                "arguements being passed, since only "
                                "(int, float, complex, str) "
                                "are stored in the cache. "
                                "Also, some parameters related to method for "
                                "table building are not critical for "
                                "consistency."
                            )
                            print(e)

        return table, is_recomputed

    def load_saved_table(
        self,
        dim,
        kernel_type,
        q_order,
        source_box_level=0,
        compute_method=None,
        **kwargs
    ):
        """Load a table saved in the hdf5 file.
        """

        q_order = int(q_order)
        assert q_order >= 1

        source_box_level = int(source_box_level)
        assert source_box_level >= 0

        # Check data table integrity
        assert str(dim) + "D" in self.datafile
        assert kernel_type in self.datafile[str(dim) + "D"]
        assert "Order_" + str(q_order) in self.datafile[str(dim) + "D"][kernel_type]
        assert (
            "Level_" + str(source_box_level)
            in self.datafile[str(dim) + "D"][kernel_type]["Order_" + str(q_order)]
        )

        grp = self.datafile[str(dim) + "D"][kernel_type]["Order_" + str(q_order)][
            "Level_" + str(source_box_level)
        ]

        assert dim == grp.attrs["dim"]
        assert q_order == grp.attrs["quad_order"]

        if compute_method == "Transform":
            if 'knl_func' not in kwargs:
                knl_func = self.get_kernel_function(dim, kernel_type, **kwargs)
            else:
                knl_func = kwargs['knl_func']
            sumpy_knl = None
        elif compute_method == "DrosteSum":
            knl_func = None
            if 'sumpy_knl' not in kwargs:
                sumpy_knl = self.get_sumpy_kernel(dim, kernel_type)
            else:
                sumpy_knl = kwargs['sumpy_knl']
        else:
            from warnings import warn

            warn("Unsupported compute_method: ", compute_method)
            knl_func = None
            sumpy_knl = None

        table = NearFieldInteractionTable(
            quad_order=q_order,
            dim=dim,
            dtype=self.dtype,
            build_method=compute_method,
            kernel_func=knl_func,
            kernel_type=self.get_kernel_function_type(dim, kernel_type),
            sumpy_kernel=sumpy_knl,
            source_box_extent=self.root_extent * (2 ** (-source_box_level)),
            **self.table_extra_kwargs
        )

        assert abs(table.source_box_extent - grp.attrs["source_box_extent"]) < 1e-15
        assert source_box_level == grp.attrs["source_box_level"]

        # Load data
        table.q_points[...] = grp["q_points"]
        table.data[...] = grp["data"]
        if 'mode_normalizers' in grp:
            table.mode_normalizers[...] = grp["mode_normalizers"]
        if 'kernel_exterior_normalizers' in grp:
            table.kernel_exterior_normalizers[...] = \
                    grp["kernel_exterior_normalizers"]

        tmp_case_vecs = np.array(table.interaction_case_vecs)
        tmp_case_vecs[...] = grp["interaction_case_vecs"]
        table.interaction_case_vecs = [list(vec) for vec in tmp_case_vecs]

        table.case_indices[...] = grp["case_indices"]

        assert table.n_q_points == grp.attrs["n_q_points"]
        assert table.n_pairs == grp.attrs["n_pairs"]
        assert table.quad_order == grp.attrs["quad_order"]

        base = grp.attrs["case_encoding_base"]
        shift = grp.attrs["case_encoding_shift"]

        def case_encode(case_vec):
            table_id = 0
            for l in case_vec:
                table_id = table_id * base + (l + shift)
            return int(table_id)

        table.case_encode = case_encode

        # load extra kwargs
        for atkey, atval in grp.attrs.items():
            setattr(table, atkey, atval)

        table.is_built = True

        return table

    def get_kernel_function(self, dim, kernel_type, **kwargs):
        """Kernel function is needed for building the table. This function
        provides support for some kernels such that the user can build and
        use the table without explicitly providing such information.
        """
        if kernel_type == "Laplace":
            knl_func = vm.nearfield_potential_table.get_laplace(dim)
        elif kernel_type == "Constant":
            knl_func = vm.nearfield_potential_table.constant_one
        elif kernel_type == "Cahn-Hilliard":
            knl_func = vm.nearfield_potential_table.get_cahn_hilliard(
                dim, kwargs["b"], kwargs["c"]
            )
        elif kernel_type in self.supported_kernels:
            knl = self.get_sumpy_kernel(dim, kernel_type)
            knl_func = vm.nearfield_potential_table.sumpy_kernel_to_lambda(knl)
        else:
            raise NotImplementedError("Kernel type not supported.")

        return knl_func

    def get_sumpy_kernel(self, dim, kernel_type):
        """Sumpy (symbolic) version of the kernel.
        """

        if kernel_type == "Laplace":
            from sumpy.kernel import LaplaceKernel

            return LaplaceKernel(dim)

        if kernel_type == "Laplace-Dx":
            from sumpy.kernel import LaplaceKernel, AxisTargetDerivative

            return AxisTargetDerivative(0, LaplaceKernel(dim))

        if kernel_type == "Laplace-Dy":
            from sumpy.kernel import LaplaceKernel, AxisTargetDerivative

            return AxisTargetDerivative(1, LaplaceKernel(dim))

        if kernel_type == "Laplace-Dz":
            from sumpy.kernel import LaplaceKernel, AxisTargetDerivative

            assert dim >= 3
            return AxisTargetDerivative(2, LaplaceKernel(dim))

        elif kernel_type == "Constant":
            return ConstantKernel(dim)

        elif kernel_type == "Yukawa":
            from sumpy.kernel import YukawaKernel

            return YukawaKernel(dim)

        elif kernel_type == "Yukawa-Dx":
            from sumpy.kernel import YukawaKernel, AxisTargetDerivative

            return AxisTargetDerivative(0, YukawaKernel(dim))

        elif kernel_type == "Yukawa-Dy":
            from sumpy.kernel import YukawaKernel, AxisTargetDerivative

            return AxisTargetDerivative(1, YukawaKernel(dim))

        elif kernel_type == "Cahn-Hilliard":
            from sumpy.kernel import FactorizedBiharmonicKernel

            return FactorizedBiharmonicKernel(dim)

        elif kernel_type == "Cahn-Hilliard-Laplacian":
            from sumpy.kernel import (
                FactorizedBiharmonicKernel,
                LaplacianTargetDerivative,
            )

            return LaplacianTargetDerivative(FactorizedBiharmonicKernel(dim))

        elif kernel_type == "Cahn-Hilliard-Dx":
            from sumpy.kernel import FactorizedBiharmonicKernel, AxisTargetDerivative

            return AxisTargetDerivative(0, FactorizedBiharmonicKernel(dim))

        elif kernel_type == "Cahn-Hilliard-Laplacian-Dx":
            from sumpy.kernel import (
                FactorizedBiharmonicKernel,
                LaplacianTargetDerivative,
            )
            from sumpy.kernel import AxisTargetDerivative

            return AxisTargetDerivative(
                0, LaplacianTargetDerivative(FactorizedBiharmonicKernel(dim))
            )

        elif kernel_type == "Cahn-Hilliard-Laplacian-Dy":
            from sumpy.kernel import (
                FactorizedBiharmonicKernel,
                LaplacianTargetDerivative,
            )
            from sumpy.kernel import AxisTargetDerivative

            return AxisTargetDerivative(
                1, LaplacianTargetDerivative(FactorizedBiharmonicKernel(dim))
            )

        elif kernel_type == "Cahn-Hilliard-Dy":
            from sumpy.kernel import FactorizedBiharmonicKernel, AxisTargetDerivative

            return AxisTargetDerivative(1, FactorizedBiharmonicKernel(dim))

        elif kernel_type in self.supported_kernels:
            return None

        else:
            raise NotImplementedError("Kernel type not supported.")

    def get_kernel_function_type(self, dim, kernel_type):
        """Determines how and to what extend the table data can be rescaled
        and reused.
        """

        if kernel_type == "Laplace":
            if dim == 2:
                return "log"
            elif dim >= 3:
                return "inv_power"
            else:
                raise NotImplementedError("Kernel scaling not supported")

        elif kernel_type == "Constant":
            if dim >= 1 and dim <= 3:
                return "const"
            else:
                raise NotImplementedError("Kernel scaling not supported")

        else:
            return None

    def update_dataset(self, group, dataset_name, data_array):
        """Update stored data.
        """
        if data_array is None:
            logger.debug("No data to save for %s" % dataset_name)
            return

        data_array = np.array(data_array)

        if dataset_name not in group:
            dset = group.create_dataset(
                dataset_name, data_array.shape, dtype=data_array.dtype
            )
        else:
            dset = group[dataset_name]

        dset[...] = data_array

    def compute_and_update_table(
        self,
        dim,
        kernel_type,
        q_order,
        source_box_level=0,
        compute_method=None,
        cl_ctx=None,
        queue=None,
        **kwargs
    ):
        """Performs the precomputation and stores the results.
        """

        if compute_method is None:
            logger.debug("Using default compute_method (Transform)")
            compute_method = "Transform"

        q_order = int(q_order)
        assert q_order >= 1

        assert str(dim) + "D" in self.datafile
        assert kernel_type in self.datafile[str(dim) + "D"]
        assert "Order_" + str(q_order) in self.datafile[str(dim) + "D"][kernel_type]

        if compute_method == "Transform":
            if 'knl_func' not in kwargs:
                knl_func = self.get_kernel_function(dim, kernel_type, **kwargs)
            else:
                knl_func = kwargs['knl_func']
            sumpy_knl = None
        elif compute_method == "DrosteSum":
            knl_func = None
            if 'sumpy_knl' not in kwargs:
                sumpy_knl = self.get_sumpy_kernel(dim, kernel_type)
            else:
                sumpy_knl = kwargs['sumpy_knl']
        else:
            raise NotImplementedError("Unsupported compute_method.")

        knl_type = self.get_kernel_function_type(dim, kernel_type)

        # compute table
        logger.debug("Start computing interaction table.")
        table = NearFieldInteractionTable(
            dim=dim,
            quad_order=q_order,
            dtype=self.dtype,
            kernel_func=knl_func,
            kernel_type=knl_type,
            sumpy_kernel=sumpy_knl,
            build_method=compute_method,
            source_box_extent=self.root_extent * (2 ** (-source_box_level)),
            **self.table_extra_kwargs
        )

        if 0:
            # self-similarly shrink delta
            if 'delta' in kwargs:
                delta = kwargs.pop('delta') * (2 ** (-source_box_level))
                kwargs['delta'] = delta

        table.build_table(cl_ctx, queue, **kwargs)
        assert table.is_built

        # update database
        logger.debug("Start updating database.")

        grp = self.datafile[str(dim) + "D"][kernel_type]["Order_" + str(q_order)][
            "Level_" + str(source_box_level)
        ]

        grp.attrs["n_q_points"] = table.n_q_points
        grp.attrs["quad_order"] = table.quad_order
        grp.attrs["dim"] = table.dim
        grp.attrs["n_pairs"] = table.n_pairs
        grp.attrs["source_box_level"] = source_box_level
        grp.attrs["source_box_extent"] = self.root_extent * (
                2 ** (-source_box_level))

        for key, kval in kwargs.items():
            if isinstance(kval, (int, float, complex, str)):
                grp.attrs[key] = kval

        self.update_dataset(grp, "q_points", table.q_points)
        self.update_dataset(grp, "data", table.data)
        self.update_dataset(grp, "mode_normalizers", table.mode_normalizers)
        self.update_dataset(grp, "kernel_exterior_normalizers",
                table.kernel_exterior_normalizers)
        self.update_dataset(grp, "interaction_case_vecs",
                table.interaction_case_vecs)
        self.update_dataset(grp, "case_indices", table.case_indices)

        distinct_numbers = set()
        for vec in table.interaction_case_vecs:
            for l in vec:
                distinct_numbers.add(l)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        grp.attrs["case_encoding_base"] = base
        grp.attrs["case_encoding_shift"] = shift

        return table


# }}} End table dataset manager class

# vim: filetype=pyopencl:foldmethod=marker
