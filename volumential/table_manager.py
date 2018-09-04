import logging

import h5py as hdf
import numpy as np

import volumential as vm
# from sumpy.expansion.local import VolumeTaylorLocalExpansion
# from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
# from sumpy.kernel import LaplaceKernel
from volumential.nearfield_potential_table import NearFieldInteractionTable

logger = logging.getLogger(__name__)

# {{{ table dataset manager class


class NearFieldInteractionTableManager(
        object):
    """
    A class that manages near field interaction table computation and
    storage.

    Tables are stored under 'Dimension/KernelName/QuadOrder/BoxLevel/dataset_name'
    e.g., '2D/Laplace/Order_1/Level_0/data'
    """

    def __init__(
            self,
            dataset_filename="nft.hdf5",
            root_extent=1
    ):
        self.filename = dataset_filename
        self.root_extent = root_extent

        # Read/write if exists, create otherwise
        self.datafile = hdf.File(self.filename, "a")

        # If the file exists, it must be for the same root_extent
        if "root_extent" not in self.datafile.attrs:
            self.datafile.attrs['root_extent'] = self.root_extent
        else:
            if not abs(self.datafile.attrs['root_extent']
                    - self.root_extent) < 1e-15:
                raise RuntimeError("The table cache file " + self.filename +
                        " was built with root_extent = " +
                        str(self.datafile.attrs['root_extent']) +
                        ", which is different from the requested value " +
                        str(self.root_extent))

        self.supported_kernels = [
            "Laplace", "Constant",
            "Cahn-Hilliard"
        ]

    def get_table(
            self,
            dim,
            kernel_type,
            q_order,
            source_box_level=0,
            force_recompute=False,
            compute_method=None,
            queue=None,
            **kwargs):
        """Primary user interface. Get the specified table regardless of how.
        In the case of a cache miss or a forced re-computation, the method
        specified in the compute_method will be used.
        """

        dim = int(dim)
        assert dim >= 1

        is_recomputed = False

        if kernel_type not in self.supported_kernels:
            raise NotImplementedError(
                "Kernel type not supported."
            )

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
            logger.debug("Table cache missing. Invoking fresh computation.")
            is_recomputed = True
            grp.create_group("Level_" + str(source_box_level))
            table = self.compute_and_update_table(
                dim, kernel_type, q_order, source_box_level,
                compute_method, queue, **kwargs)

        elif force_recompute:
            logger.debug("Invoking fresh computation since force_recompute is set")
            is_recomputed = True
            table = self.compute_and_update_table(
                dim, kernel_type, q_order, source_box_level,
                compute_method, queue, **kwargs)

        else:
            try:

                table = self.load_saved_table(
                        dim, kernel_type, q_order, source_box_level)

            except KeyError:

                logger.debug("Recomputing due to table data corruption.")
                is_recomputed = True
                table = self.compute_and_update_table(
                    dim, kernel_type, q_order, source_box_level,
                    compute_method, queue, **kwargs)

        return table, is_recomputed

    def load_saved_table(self, dim,
                         kernel_type,
                         q_order, source_box_level=0):

        if kernel_type not in self.supported_kernels:
            raise NotImplementedError(
                "Kernel type not supported."
            )

        q_order = int(q_order)
        assert q_order >= 1

        source_box_level = int(source_box_level)
        assert source_box_level >= 0

        # Check data table integrity
        assert (str(dim) + "D" in self.datafile)
        assert (kernel_type in self.datafile[str(dim) + "D"])
        assert ("Order_" + str(q_order) in
                self.datafile[str(dim) + "D"][kernel_type])
        assert ("Level_" + str(source_box_level) in
                self.datafile[str(dim) + "D"][kernel_type]["Order_" + str(q_order)])

        grp = self.datafile[str(dim) + "D"][kernel_type][
                "Order_" + str(q_order)]["Level_" + str(source_box_level)]

        assert (dim == grp.attrs['dim'])
        assert (q_order == grp.attrs['quad_order'])

        table = NearFieldInteractionTable(
            quad_order=q_order,
            dim=dim,
            kernel_func=self.get_kernel_function(dim, kernel_type),
            kernel_type=self.get_kernel_function_type(dim, kernel_type),
            sumpy_kernel=self.get_sumpy_kernel(dim, kernel_type),
            source_box_extent=self.root_extent * (2**(-source_box_level))
            )

        assert abs(table.source_box_extent - grp.attrs["source_box_extent"]
                ) < 1e-15
        assert source_box_level == grp.attrs["source_box_level"]

        # Load data
        table.q_points[...] = grp["q_points"]
        table.data[...] = grp["data"]
        table.mode_normalizers[...] = grp["mode_normalizers"]

        tmp_case_vecs = np.array(table.interaction_case_vecs)
        tmp_case_vecs[...] = grp["interaction_case_vecs"]
        table.interaction_case_vecs = [list(vec) for vec in tmp_case_vecs]

        table.case_indices[...] = grp["case_indices"]

        assert table.n_q_points == grp.attrs["n_q_points"]
        assert table.n_pairs == grp.attrs["n_pairs"]
        assert table.quad_order == grp.attrs["quad_order"]

        base = grp.attrs['case_encoding_base']
        shift = grp.attrs['case_encoding_shift']

        def case_encode(case_vec):
            table_id = 0
            for l in case_vec:
                table_id = table_id * base + (
                    l + shift)
            return int(table_id)

        table.case_encode = case_encode

        table.is_built = True

        return table

    def get_kernel_function(
            self, dim, kernel_type):

        if kernel_type == "Laplace":
            # knl = LaplaceKernel(dim)
            # For unknown reasons this does not work with multiprocess
            # knl_func = vm.nearfield_potential_table.sumpy_kernel_to_lambda(knl)
            knl_func = vm.nearfield_potential_table.get_laplace(dim)
        elif kernel_type == "Constant":
            knl_func = vm.nearfield_potential_table.constant_one
        elif kernel_type == "Cahn-Hilliard":
            knl_func = vm.nearfield_potential_table.get_cahn_hilliard(dim)
        elif kernel_type in self.supported_kernels:
            knl_func = None
        else:
            raise NotImplementedError(
                "Kernel type not supported."
            )

        return knl_func

    def get_sumpy_kernel(
            self, dim, kernel_type):

        if kernel_type == "Laplace":
            from sumpy.kernel import LaplaceKernel
            return LaplaceKernel(dim)

        elif kernel_type in self.supported_kernels:
            return None

        else:
            raise NotImplementedError(
                "Kernel type not supported."
            )

    def get_kernel_function_type(
            self, dim, kernel_type):

        if kernel_type == "Laplace":
            if dim == 2:
                return "log"
            elif dim >= 3:
                return "inv_power"
            else:
                raise NotImplementedError(
                    "Kernel scaling not supported"
                )

        elif kernel_type == "Constant":
            if dim >= 1 and dim <= 3:
                return "const"
            else:
                raise NotImplementedError(
                    "Kernel scaling not supported"
                )

        elif kernel_type == "Cahn-Hilliard":
            return None

        elif kernel_type in self.supported_kernels:
            return None

        else:
            raise NotImplementedError(
                "Kernel scaling not supported"
            )

    def update_dataset(self, group,
                       dataset_name,
                       data_array):

        data_array = np.array(data_array)

        if dataset_name not in group:
            dset = group.create_dataset(
                dataset_name,
                data_array.shape,
                dtype=data_array.dtype)
        else:
            dset = group[dataset_name]

        dset[...] = data_array

    def compute_and_update_table(
            self, dim, kernel_type, q_order, source_box_level=0,
            compute_method=None, queue=None, **kwargs):

        if kernel_type not in self.supported_kernels:
            raise NotImplementedError(
                "Kernel type not supported."
            )

        if compute_method is None:
            logger.debug("Using default compute_method (Transform)")
            compute_method = "Transform"

        q_order = int(q_order)
        assert (q_order >= 1)

        assert str(dim) + "D" in self.datafile
        assert kernel_type in self.datafile[str(dim) +"D"]
        assert "Order_" + str(q_order) in self.datafile[str(dim) + "D"][kernel_type]

        if compute_method == "Transform":
            knl_func = self.get_kernel_function(
                dim, kernel_type)
            sumpy_knl = None
        elif compute_method == "DrosteSum":
            knl_func = None
            sumpy_knl = self.get_sumpy_kernel(dim, kernel_type)
        else:
            raise NotImplementedError("Unsupported compute_method.")

        knl_type = self.get_kernel_function_type(dim, kernel_type)

        # compute table
        logger.debug("Start computing interaction table.")
        table = NearFieldInteractionTable(
            quad_order=q_order,
            kernel_func=knl_func,
            kernel_type=knl_type,
            sumpy_kernel=sumpy_knl,
            build_method=compute_method,
            source_box_extent=self.root_extent * (2**(-source_box_level))
            )
        table.build_table(queue, **kwargs)
        assert (table.is_built)

        # update database
        logger.debug("Start updating database.")

        grp = self.datafile[
                str(dim) + "D"][kernel_type][
                        "Order_" + str(q_order)]["Level_" + str(source_box_level)]

        grp.attrs['n_q_points'] = table.n_q_points
        grp.attrs['quad_order'] = table.quad_order
        grp.attrs['dim'] = table.dim
        grp.attrs['n_pairs'] = table.n_pairs
        grp.attrs['source_box_level'] = source_box_level
        grp.attrs['source_box_extent'] = self.root_extent * (2**(-source_box_level))

        for key, kval in kwargs.items():
            grp.attrs[key] = kval

        self.update_dataset(grp, "q_points", table.q_points)
        self.update_dataset(grp, "data", table.data)
        self.update_dataset(grp, "mode_normalizers", table.mode_normalizers)
        self.update_dataset(grp, "interaction_case_vecs", table.interaction_case_vecs)
        self.update_dataset(grp, "case_indices", table.case_indices)

        distinct_numbers = set()
        for vec in table.interaction_case_vecs:
            for l in vec:
                distinct_numbers.add(l)
        base = len(
            range(
                min(distinct_numbers),
                max(distinct_numbers) +
                1))
        shift = -min(distinct_numbers)

        grp.attrs['case_encoding_base'] = base
        grp.attrs['case_encoding_shift'] = shift

        return table


# }}} End table dataset manager class

# vim: filetype=pyopencl:foldmethod=marker
