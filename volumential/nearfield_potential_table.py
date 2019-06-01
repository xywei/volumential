from __future__ import absolute_import, division, print_function

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
from scipy.interpolate import BarycentricInterpolator as Interpolator

import volumential.list1_gallery as gallery
import volumential.singular_integral_2d as squad

import logging

logger = logging.getLogger(__name__)

# NOTE: Cannot use sumpy kernels for "Transform" since they
# do not work with multiprocess.
# e.g. with LaplaceKernel it will complain "AttributeError:
# 'LaplaceKernel' object has no attribute 'expression'"


def _self_tp(vec, tpd=2):
    """
    Self tensor product
    """
    assert len(vec.shape) == 1
    if tpd == 1:
        return vec
    elif tpd == 2:
        return vec.reshape([len(vec), 1]) * vec.reshape([1, len(vec)])
    elif tpd == 3:
        return (
            vec.reshape([len(vec), 1, 1])
            * vec.reshape([1, len(vec), 1])
            * vec.reshape([1, 1, len(vec)])
        )
    else:
        raise NotImplementedError


def _orthonormal(n, i):
    eb = np.zeros(n)
    eb[i] = 1
    return eb


def constant_one(x, y=None, z=None):
    return np.ones(np.array(x).shape)


# {{{ kernel function getters


def get_laplace(dim):
    if dim != 2:
        raise NotImplementedError(
            "Kernel function Laplace" + str(dim) + "D not implemented."
        )
    else:

        def laplace(x, y):
            return -0.25 * np.log(np.array(x) ** 2 + np.array(y) ** 2) / np.pi

        return laplace


def get_cahn_hilliard(dim, b=0, c=0, approx_at_origin=False):
    if dim != 2:
        raise NotImplementedError(
            "Kernel function Laplace" + str(dim) + "D not implemented."
        )
    else:

        def quadratic_formula_1(a, b, c):
            return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        def quadratic_formula_2(a, b, c):
            return (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        def citardauq_formula_1(a, b, c):
            return 2 * c / (-b - np.sqrt(b ** 2 - 4 * a * c))

        def citardauq_formula_2(a, b, c):
            return 2 * c / (-b + np.sqrt(b ** 2 - 4 * a * c))

        def f(x):
            return x ** 2 - b * x + c

        root11 = quadratic_formula_1(1, -b, c)
        root12 = citardauq_formula_1(1, -b, c)
        if np.abs(f(root11)) < np.abs(f(root12)):
            lam1 = np.sqrt(root11)
        else:
            lam1 = np.sqrt(root12)

        root21 = quadratic_formula_2(1, -b, c)
        root22 = citardauq_formula_2(1, -b, c)
        if np.abs(f(root21)) < np.abs(f(root22)):
            lam1 = np.sqrt(root21)
        else:
            lam2 = np.sqrt(root22)

        # assert np.abs(f(lam1**2)) < 1e-12
        # assert np.abs(f(lam2**2)) < 1e-12

        lambdas = sorted([lam1, lam2], key=abs, reverse=True)  # biggest first
        lam1 = lambdas[0]
        lam2 = lambdas[1]

        import scipy.special as sp

        def cahn_hilliard(x, y):
            r = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)

            # Leading order removed analytically
            def k0_approx(rr, lam):
                euler_constant = 0.57721566490153286060651209008240243104215933593992
                r = rr * lam
                return (
                    -(np.log(lam) + euler_constant)
                    - (np.log(r / 2) + euler_constant) * (r ** 2 / 4 + r ** 4 / 64)
                    + (r ** 2 / 4 + r ** 4 * 3 / 128)
                )

            if approx_at_origin:
                return (
                    -1
                    / (2 * np.pi * (lam1 ** 2 - lam2 ** 2))
                    * (k0_approx(r, lam1) - k0_approx(r, lam2))
                )
            else:
                return (
                    -1
                    / (2 * np.pi * (lam1 ** 2 - lam2 ** 2))
                    * (sp.kn(0, lam1 * r) - sp.kn(0, lam2 * r))
                )

        return cahn_hilliard


def get_cahn_hilliard_laplacian(dim, b=0, c=0):
    raise NotImplementedError(
        "Transform method under construction, " "use DrosteSum instead"
    )


# }}} End kernel function getters


def sumpy_kernel_to_lambda(sknl):
    from sympy import Symbol, symbols, lambdify

    var_name_prefix = "x"
    var_names = " ".join([var_name_prefix + str(i) for i in range(sknl.dim)])
    arg_names = symbols(var_names)
    args = [Symbol(var_name_prefix + str(i)) for i in range(sknl.dim)]

    def func(x, y=None, z=None):
        coord = (x, y, z)
        lmd = lambdify(
            arg_names, sknl.get_expression(args) * sknl.get_global_scaling_const()
        )
        return lmd(*coord[: sknl.dim])

    return func


# {{{ table data structure


class NearFieldInteractionTable(object):
    """Class for a near-field interaction table.

        A near-field interaction table stores precomputed singular integrals
        on template boxes and supports transforms to actual boxes on lookup.
        The query process is done through scaling the entries based on actual
        box sized.

        Orientations are ordered counter-clockwise.

        A template box is one of [0,1]^dim
    """

    # {{{ constructor

    def __init__(
        self,
        quad_order,
        method="gauss-legendre",
        dim=2,
        kernel_func=None,
        kernel_type=None,
        sumpy_kernel=None,
        build_method="Transform",
        source_box_extent=1,
        dtype=np.float64,
        inverse_droste=False
    ):
        """
        kernel_type determines how the kernel is scaled w.r.t. box size.
        build_method can be "Transform" or "DrosteSum".

        The source box is [0, source_box_extent]^dim

        :arg inverse_droste True if computing with the fractional Laplacian kernel.
        """
        self.quad_order = quad_order
        self.dim = dim
        self.dtype = dtype
        self.inverse_droste = inverse_droste

        assert source_box_extent > 0
        self.source_box_extent = source_box_extent

        self.center = np.ones(self.dim) * 0.5 * self.source_box_extent

        self.build_method = build_method

        if dim == 1:

            if build_method == "Transform":
                raise NotImplementedError("Use DrosteSum for 1d")

            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel

        elif dim == 2:

            # Constant kernel can be used for fun/testing
            if kernel_func is None:
                kernel_func = constant_one
                kernel_type = "const"
                # for DrosteSum kernel_func is unused
                if build_method == "Transform":
                    logger.warning("setting kernel_func to be constant.")

            # Kernel function differs from OpenCL's kernels
            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel

            if build_method == "DrosteSum":
                assert sumpy_kernel is not None

        elif dim == 3:

            if build_method == "Transform":
                raise NotImplementedError("Use DrosteSum for 3d")

            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel

        else:
            raise NotImplementedError

        # number of quad points per box
        # equals to the number of modes per box
        self.n_q_points = self.quad_order ** dim

        # Normalizers for polynomial modes
        # Needed only when we want to rescale log type kernels
        self.mode_normalizers = np.zeros(self.n_q_points, dtype=self.dtype)

        # number of (source_mode, target_point) pairs between two boxes
        self.n_pairs = self.n_q_points ** 2

        # possible interaction cases
        self.interaction_case_vecs, self.case_encode, self.case_indices = \
                gallery.generate_list1_gallery(self.dim)
        self.n_cases = len(self.interaction_case_vecs)

        if method == "gauss-legendre":
            # quad points in [-1,1]
            import volumential.meshgen as mg

            # FIXME: 3D support
            q_points, _, _ = mg.make_uniform_cubic_grid(
                degree=quad_order, level=1, dim=self.dim
            )
            # map to source box
            mapped_q_points = np.array(
                [
                    0.5 * self.source_box_extent * (qp + np.ones(self.dim))
                    for qp in q_points
                ]
            )
            # sort in dictionary order, preserve only the leading
            # digits to prevent floating point errors from polluting
            # the ordering.
            q_points_ordering = sorted(
                range(len(mapped_q_points)),
                key=lambda i: list(np.floor(mapped_q_points[i] * 10000)),
            )
            self.q_points = mapped_q_points[q_points_ordering]

        else:
            raise NotImplementedError

        self.data = np.empty(self.n_pairs * self.n_cases, dtype=self.dtype)
        self.data.fill(np.nan)

        total_evals = len(self.data) + self.n_q_points

        from pytools import ProgressBar

        self.pb = ProgressBar("Building table:", total_evals)

        self.is_built = False

    # }}} End constructor

    # {{{ encode to table index

    def get_entry_index(self, source_mode_index, target_point_index, case_id):

        assert source_mode_index >= 0 and source_mode_index < self.n_q_points
        assert target_point_index >= 0 and target_point_index < self.n_q_points
        pair_id = source_mode_index * self.n_q_points + target_point_index

        return case_id * self.n_pairs + pair_id

    # }}} End encode to table index

    # {{{ decode table index to entry info

    def decode_index(self, entry_id):
        """This is the inverse function of get_entry_index()
        """

        index_info = dict()

        case_id = entry_id // self.n_pairs
        pair_id = entry_id % self.n_pairs

        source_mode_index = pair_id // self.n_q_points
        target_point_index = pair_id % self.n_q_points

        index_info["case_index"] = case_id
        index_info["source_mode_index"] = source_mode_index
        index_info["target_point_index"] = target_point_index

        return index_info

    # }}} End decode table index to entry info

    # {{{ basis modes in the template box

    def unwrap_mode_index(self, mode_index):
        # NOTE: these two lines should be changed
        # in accordance with the mesh generator
        # to get correct xi (1d grid)
        if self.dim == 1:
            idx = [mode_index]
        elif self.dim == 2:
            idx = [mode_index // self.quad_order, mode_index % self.quad_order]
        elif self.dim == 3:
            idx = [
                mode_index // (self.quad_order ** 2),
                mode_index % (self.quad_order ** 2) // self.quad_order,
                mode_index % (self.quad_order ** 2) % self.quad_order,
            ]
        return idx

    def get_template_mode(self, mode_index):
        assert mode_index >= 0 and mode_index < self.n_q_points
        """
        template modes are defined on an l_infty circle.
        """
        idx = self.unwrap_mode_index(mode_index)

        xi = (
            np.array([p[self.dim - 1] for p in self.q_points[: self.quad_order]])
            / self.source_box_extent
        )
        assert len(xi) == self.quad_order

        yi = []
        for d in range(self.dim):
            yi.append(np.zeros(self.quad_order, dtype=self.dtype))
            yi[d][idx[d]] = 1

        axis_interp = [Interpolator(xi, yi[d]) for d in range(self.dim)]

        def mode(*coords):
            assert len(coords) == self.dim
            if isinstance(coords[0], (int, float, complex)):
                fvals = np.ones(1)
            else:
                fvals = np.ones(len(np.array(coords[0])))
            for d, coord in zip(range(self.dim), coords):
                fvals = np.multiply(fvals, axis_interp[d](np.array(coord)))
            return fvals

        return mode

    def get_mode(self, mode_index):
        """
        normal modes are deined on the source box
        """
        assert mode_index >= 0 and mode_index < self.n_q_points

        idx = self.unwrap_mode_index(mode_index)

        xi = np.array([p[self.dim - 1] for p in self.q_points[: self.quad_order]])
        assert len(xi) == self.quad_order

        yi = []
        for d in range(self.dim):
            yi.append(np.zeros(self.quad_order, dtype=self.dtype))
            yi[d][idx[d]] = 1

        axis_interp = [Interpolator(xi, yi[d]) for d in range(self.dim)]

        def mode(*coords):
            assert len(coords) == self.dim
            if isinstance(coords[0], (int, float, complex)):
                fvals = np.ones(1)
            else:
                fvals = np.ones(len(np.array(coords[0])))
            for d, coord in zip(range(self.dim), coords):
                fvals = np.multiply(fvals, axis_interp[d](np.array(coord)))
            return fvals

        return mode

    def get_mode_cheb_coeffs(self, mode_index, cheb_order):
        """
        Cheb coeffs of a mode.
        The projection process is performed on [0,1]^dim.
        """

        import scipy.special as sps

        cheby_nodes, _, cheby_weights = sps.chebyt(cheb_order).weights.T
        window = [0, 1]

        cheby_nodes = cheby_nodes * (window[1] - window[0]) / 2 + np.mean(window)
        cheby_weights = cheby_weights * (window[1] - window[0]) / 2

        mode = self.get_template_mode(mode_index)
        grid = np.meshgrid(*[cheby_nodes for d in range(self.dim)])
        mvals = mode(*grid)

        from numpy.polynomial.chebyshev import Chebyshev

        coef_scale = 2 * np.ones(cheb_order) / cheb_order
        coef_scale[0] /= 2
        basis_1d = np.array(
            [
                Chebyshev(
                    coef=_orthonormal(cheb_order, i),
                    domain=window)(cheby_nodes)
                for i in range(cheb_order)
            ]
        )

        from itertools import product

        if self.dim == 1:
            basis_set = basis_1d

        elif self.dim == 2:
            basis_set = np.array(
                [
                    b1.reshape([cheb_order, 1]) * b2.reshape([1, cheb_order])
                    for b1, b2 in product(*[basis_1d for d in range(self.dim)])
                ]
            )

        elif self.dim == 3:
            basis_set = np.array(
                [
                    b1.reshape([cheb_order, 1, 1])
                    * b2.reshape([1, cheb_order, 1])
                    * b3.reshape([1, 1, cheb_order])
                    for b1, b2, b3 in product(*[basis_1d for d in range(self.dim)])
                ]
            )

        mode_cheb_coeffs = np.array(
            [np.sum(mvals * basis) for basis in basis_set]
        ) * _self_tp(coef_scale, self.dim).reshape(-1)

        # purge small coeffs that are 15 digits away
        mm = np.max(np.abs(mode_cheb_coeffs))
        mode_cheb_coeffs[np.abs(mode_cheb_coeffs) < mm * 1e-15] = 0

        return mode_cheb_coeffs

    # }}} End basis modes in the template box

    # {{{ build table via transform

    def get_symmetry_transform(self, source_mode_index):
        """Apply proper transforms to map source mode to a reduced region

            Returns:
            - a transform that can be applied on the interaction case
            vectors connection box centers.
            - a transform that can be applied to the mode/point indices.
        """
        # mat = np.diag(np.ones(self.dim))

        # q_points must be sorted in (ascending) dictionary order
        k = np.zeros(self.dim)
        resid = source_mode_index
        for d in range(-1, -1 - self.dim, -1):
            k[d] = resid % self.quad_order
            resid = resid // self.quad_order

        s1 = np.sign((self.quad_order - 0.5) / 2 - k)
        for d in range(self.dim):
            if s1[d] < 0:
                k[d] = self.quad_order - 1 - k[d]

        s2 = sorted(range(len(k)), key=lambda i: abs(k[i]))

        def symmetry_transform(vec):
            nv = np.array(vec) * s1
            return nv[s2]

        def qpoint_index_transform(index):
            k = np.zeros(self.dim, dtype=int)
            resid = index
            for d in range(-1, -1 - self.dim, -1):
                k[d] = resid % self.quad_order
                resid = resid // self.quad_order
            assert resid == 0
            for d in range(self.dim):
                if s1[d] < 0:
                    k[d] = self.quad_order - 1 - k[d]
            k = k[s2]
            new_id = 0
            for d in range(self.dim):
                new_id = new_id * int(self.quad_order) + k[d]
            return new_id

        return (symmetry_transform, qpoint_index_transform)

    def find_target_point(self, target_point_index, case_index):
        """Apply proper transforms to find the target point's coordinate.

        Only translations and scalings are allowed in this step, avoiding the
        indices of quad points to be messed up.
        """
        assert target_point_index >= 0 and target_point_index < self.n_q_points

        # rescale to source box with size 1x1
        vec = (
            np.array(self.interaction_case_vecs[case_index])
            / 4.0
            * self.source_box_extent
        )

        new_cntr = (
            np.ones(self.dim, dtype=self.dtype) * 0.5 * self.source_box_extent + vec
        )

        if int(max(abs(np.array(self.interaction_case_vecs[case_index])))) == 0:
            new_size = 1
        else:
            new_size = (
                max([abs(l) - 2 for l in self.interaction_case_vecs[case_index]]) / 2
            )

        # print(vec, new_cntr, new_size)

        return new_cntr + new_size * (
                self.q_points[target_point_index] - self.center)

    def lookup_by_symmetry(self, entry_id):
        """Loop up table entry that is mapped to a region where:
            - k_i <= q/2 in all direction i
            - k_i's are sorted in ascending order

            Returns the mapped entry_id
        """

        entry_info = self.decode_index(entry_id)
        # source_mode = self.get_mode(
        #    entry_info[
        #        "source_mode_index"])
        # target_point = self.find_target_point(
        #    target_point_index=
        #    entry_info[
        #        "target_point_index"],
        #    case_index=entry_info[
        #        "case_index"])

        vec_map, qp_map = self.get_symmetry_transform(
                entry_info["source_mode_index"])
        # mapped (canonical) case_id
        case_vec = self.interaction_case_vecs[entry_info["case_index"]]
        cc_vec = vec_map(case_vec)
        cc_id = self.case_indices[self.case_encode(cc_vec)]
        cs_id = qp_map(entry_info["source_mode_index"])
        ct_id = qp_map(entry_info["target_point_index"])
        centry_id = self.get_entry_index(cs_id, ct_id, cc_id)

        return centry_id

    def compute_table_entry(self, entry_id):
        """Compute one entry in the table indexed by self.data[entry_id]

        Input kernel function should be centered at origin.
        """
        entry_info = self.decode_index(entry_id)
        source_mode = self.get_mode(entry_info["source_mode_index"])
        target_point = self.find_target_point(
            target_point_index=entry_info["target_point_index"],
            case_index=entry_info["case_index"],
        )

        # print(entry_info, target_point)
        # source_point = (
        #     self.q_points[entry_info[
        #         "source_mode_index"]])
        # print(source_mode(source_point[0], source_point[1]))

        if self.dim == 2:

            def integrand(x, y):
                return source_mode(x, y) * self.kernel_func(
                    x - target_point[0], y - target_point[1]
                )

            integral, error = squad.box_quad(
                func=integrand,
                a=0,
                b=self.source_box_extent,
                c=0,
                d=self.source_box_extent,
                singular_point=target_point,
                # tol=1e-10,
                # rtol=1e-10,
                # miniter=300,
                maxiter=301,
            )
        else:
            raise NotImplementedError

        return (entry_id, integral)

    def compute_nmlz(self, mode_id):
        mode_func = self.get_mode(mode_id)
        nmlz, _ = squad.qquad(
            func=mode_func,
            a=0,
            b=self.source_box_extent,
            c=0,
            d=self.source_box_extent,
            tol=1e-15,
            rtol=1e-15,
            minitero=25,
            miniteri=25,
            maxitero=100,
            maxiteri=100,
        )
        return (mode_id, nmlz)

    def build_normalizer_table(self, pool=None, pb=None):
        """
        Build normalizers only
        """
        assert self.dim == 2
        if pool is None:
            from multiprocess import Pool

            pool = Pool(processes=None)

        for mode_id, nmlz in pool.imap_unordered(
            self.compute_nmlz, [i for i in range(self.n_q_points)]
        ):
            self.mode_normalizers[mode_id] = nmlz
            if pb is not None:
                pb.progress(1)

    def build_table_via_transform(self):
        """
        Build the full data table using transforms to
        remove the singularity.
        """
        assert self.dim == 2

        # multiprocessing cannot handle member functions
        from multiprocess import Pool

        pool = Pool(processes=None)

        self.pb.draw()

        self.build_normalizer_table(pool, pb=self.pb)
        self.has_normalizers = True

        # First compute entries that are invariant under
        # symmetry lookup
        invariant_entry_ids = [
            i for i in range(len(self.data)) if self.lookup_by_symmetry(i) == i
        ]

        for entry_id, entry_val in pool.imap_unordered(
            self.compute_table_entry, invariant_entry_ids
        ):
            self.data[entry_id] = entry_val
            self.pb.progress(1)

        # Then complete the table via symmetry lookup
        for entry_id, centry_id in enumerate(
            pool.imap_unordered(
                self.lookup_by_symmetry, [i for i in range(len(self.data))]
            )
        ):
            assert not np.isnan(self.data[centry_id])
            if centry_id == entry_id:
                continue
            self.data[entry_id] = self.data[centry_id]
            self.pb.progress(1)

        self.pb.finished()

        for entry in self.data:
            assert not np.isnan(entry)

        self.is_built = True

    # }}} End build table via transform

    # {{{ build table via adding up a Droste of bricks

    def build_table_via_droste_bricks(
        self,
        n_brick_quad_points=50,
        alpha=0,
        queue=None,
        adaptive_level=True,
        use_symmetry=False,
        **kwargs
    ):

        if queue is None:
            import pyopencl as cl

            cl_ctx = cl.create_some_context(interactive=True)
            queue = cl.CommandQueue(cl_ctx)

        assert alpha >= 0 and alpha < 1
        if "nlevels" in kwargs:
            nlev = kwargs.pop("nlevels")
        else:
            nlev = 1

        # extra_kernel_kwargs = {}
        # if "extra_kernel_kwargs" in kwargs:
        #     extra_kernel_kwargs = kwargs["extra_kernel_kwargs"]

        cheb_coefs = [
            self.get_mode_cheb_coeffs(mid, max(self.quad_order, 3))
            for mid in range(self.n_q_points)
        ]

        if self.inverse_droste:
            from volumential.droste import InverseDrosteReduced

            if "knl_symietry_tags" in kwargs:
                knl_symmetry_tags = kwargs["knl_symmetry_tags"]
            else:
                # Maximum symmetry by default
                knl_symmetry_tags = None

            drf = InverseDrosteReduced(
                    self.integral_knl,
                    self.quad_order,
                    self.interaction_case_vecs,
                    n_brick_quad_points,
                    knl_symmetry_tags,
                    auto_windowing=True
                    )

        else:
            if not use_symmetry:
                from volumential.droste import DrosteFull

                drf = DrosteFull(
                    self.integral_knl,
                    self.quad_order,
                    self.interaction_case_vecs,
                    n_brick_quad_points,
                )
            else:
                from volumential.droste import DrosteReduced

                if "knl_symietry_tags" in kwargs:
                    knl_symmetry_tags = kwargs["knl_symmetry_tags"]
                else:
                    # Maximum symmetry by default
                    knl_symmetry_tags = None

                drf = DrosteReduced(
                    self.integral_knl,
                    self.quad_order,
                    self.interaction_case_vecs,
                    n_brick_quad_points,
                    knl_symmetry_tags,
                )

        # adaptive level refinement
        data0 = drf(
            queue,
            source_box_extent=self.source_box_extent,
            alpha=alpha,
            nlevels=nlev,
            # extra_kernel_kwargs=extra_kernel_kwargs,
            cheb_coefs=cheb_coefs,
            **kwargs
        )
        resid = -1

        if not adaptive_level:
            self.data = data0
        else:
            while alpha ** nlev > 1e-6:
                nlev = nlev + 1
                data1 = drf(
                    queue,
                    source_box_extent=self.source_box_extent,
                    alpha=alpha,
                    nlevels=nlev,
                    # extra_kernel_kwargs=extra_kernel_kwargs,
                    cheb_coefs=cheb_coefs,
                    **kwargs
                )

                resid = np.max(np.abs(data1 - data0)) / np.max(np.abs(data1))

                data0 = data1

                if resid < 1e-12 and resid > 0:
                    self.data = data0
                    logger.info(
                        "Adaptive level refinement "
                        "converged at level %d" % (nlev - 1)
                    )
                    break

                if np.isnan(resid):
                    logger.info(
                        "Adaptive level refinement terminated "
                        "at %d before converging due to NaNs" % nlev
                    )
                    break

            if resid >= 1e-12:
                logger.info("Adaptive level refinement failed to converge.")
                logger.info("Residual at level %d equals to %d" % (nlev, resid))

            if resid < 0:
                logger.info("No need for adaptive level refinement.")

            self.data = data0

        # print("Computing normalizers")
        # NOTE: normalizers are for log kernels and not needed in 3D
        if self.dim == 2:
            self.build_normalizer_table()
            self.has_normalizers = True
        else:
            self.has_normalizers = False

        if self.inverse_droste:
            self.build_kernel_exterior_normalizer_table()
        else:
            self.kernel_exterior_normalizers = None

        self.is_built = True

    # }}} End build table via adding up a Droste of bricks

    # {{{ build table (driver)

    def build_table(self, queue=None, **kwargs):
        method = self.build_method
        if method == "Transform":
            logger.info("Building table with transform method")
            self.build_table_via_transform()
        elif method == "DrosteSum":
            logger.info("Building table with Droste method")
            self.build_table_via_droste_bricks(queue=queue, **kwargs)
        else:
            raise NotImplementedError()

    # }}} End build table (driver)

    # {{{ build kernel exterior normalizer table

    def build_kernel_exterior_normalizer_table(self, cl_ctx, queue,
            pool=None,
            mesh_order=5, quad_order=10, mesh_size=0.03,
            remove_tmp_files=True):
        r"""Build the kernel exterior normalizer table.

        An exterior normalizer for kernel :math:`G(r)` and target
        :math:`x` is defined as

        .. math::

            \int_{B^c} G(\lVert x - y \rVert) dy

        where :math:`B` is the source box.
        """
        if pool is None:
            from multiprocessing import Pool, cpu_count
            pool = Pool(cpu_count)

        from meshmode.mesh.io import read_gmsh
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory

        # from pytential import bind, sym
        import gmsh

        # {{{ gmsh processing

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # meshmode does not support other versions
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)
        if mesh_order > 1:
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthFromCurvature", 1)

        # radius of source box
        hs = self.source_box_extent / 2
        # radius of bouding sphere
        r = hs * np.sqrt(self.dim)

        if self.dim == 2:
            tag_box = gmsh.model.occ.addRectangle(x=0, y=0, z=0,
                    dx=2*hs, dy=2*hs, tag=-1)
        elif self.dim == 3:
            tag_box = gmsh.model.occ.addBox(x=0, y=0, z=0,
                    dx=2*hs, dy=2*hs, dz=2*hs, tag=-1)
        else:
            raise NotImplementedError()

        if self.dim == 2:
            tag_ball = gmsh.model.occ.addDisk(xc=hs, yc=hs, zc=0,
                    rx=r, ry=r, tag=-1)
        elif self.dim == 3:
            tag_sphere = gmsh.model.occ.addSphere(xc=hs, yc=hs, zc=hs,
                    radius=r, tag=-1)
            tag_ball = gmsh.model.occ.addVolume([tag_sphere], tag=-1)
        else:
            raise NotImplementedError()

        dimtags_ints, dimtags_map_ints = gmsh.model.occ.cut(
                objectDimTags=[(self.dim, tag_ball)],
                toolDimTags=[(self.dim, tag_box)],
                tag=-1, removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(self.dim)

        from tempfile import mkdtemp
        from os.path import join
        temp_dir = mkdtemp(prefix="tmp_volumential_nft")
        msh_filename = join(temp_dir, 'chinese_lucky_coin.msh')
        gmsh.write(msh_filename)
        gmsh.finalize()

        mesh = read_gmsh(msh_filename)
        if remove_tmp_files:
            import shutil
            shutil.rmtree(temp_dir)

        # }}} End gmsh processing

        discr = Discretization(cl_ctx, mesh,  # noqa
                PolynomialWarpAndBlendGroupFactory(order=quad_order))

        # nodes = discr.nodes().with_queue(queue)[:self.dim]
        # weights = discr.quad_weights(queue).with_queue(queue)

        # TODO: evaluate kernel on nodes
        raise NotImplementedError(
                "Coming soon: this part of the code is under construction.")
        self.integral_knl

        self.kernel_exterior_normalizers = None

    # }}} End kernel exterior normalizer table

    # {{{ query table and transform to actual box

    def get_potential_scaler(
        self, entry_id, source_box_size=1, kernel_type=None, kernel_power=None
    ):
        """Returns a helper function to rescale the table entry based on
           source_box's actual size (edge length).
        """
        assert source_box_size > 0
        a = source_box_size

        if kernel_type is None:
            kernel_type = self.kernel_type

        if kernel_type is None:
            raise NotImplementedError(
                "Specify kernel type before performing scaling queries"
            )

        if kernel_type == "log":
            assert kernel_power is None
            source_mode_index = self.decode_index(entry_id)["source_mode_index"]
            displacement = (a ** 2) * np.log(a) \
                    * self.mode_normalizers[source_mode_index]
            scaling = a ** 2

        elif kernel_type == "const":
            displacement = 0
            scaling = 1

        elif kernel_type == "inv_power":
            assert kernel_power is not None
            displacement = 0
            scaling = source_box_size ** (2 + kernel_power)

        elif kernel_type == "rigid":
            # TODO: add assertion for source box size
            displacement = 0
            scaling = 1

        else:
            raise NotImplementedError("Unsupported kernel type")

        def scaler(pot_val):
            return pot_val * scaling + displacement

        return scaler


# }}} End query table and transform to actual box

# }}} End table data structure

# vim: foldmethod=marker:filetype=pyopencl
