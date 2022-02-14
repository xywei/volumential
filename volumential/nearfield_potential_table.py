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

import logging
import numpy as np
import loopy as lp
import pyopencl as cl
from scipy.interpolate import BarycentricInterpolator as Interpolator

from functools import partial

import volumential.list1_gallery as gallery
import volumential.singular_integral_2d as squad

logger = logging.getLogger('NearFieldInteractionTable')


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
        build_method=None,
        source_box_extent=1,
        dtype=np.float64,
        inverse_droste=False,
        progress_bar=True,
        **kwargs
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
                raise NotImplementedError("Use build_method=DrosteSum for 1d")

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
                raise NotImplementedError("Use build_method=DrosteSum for 3d")

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

        # Exterior normalizers for hypersingular kernels
        self.kernel_exterior_normalizers = np.zeros(
                self.n_q_points, dtype=self.dtype)

        # number of (source_mode, target_point) pairs between two boxes
        self.n_pairs = self.n_q_points ** 2

        # possible interaction cases
        self.interaction_case_vecs, self.case_encode, self.case_indices = \
                gallery.generate_list1_gallery(self.dim)
        self.n_cases = len(self.interaction_case_vecs)

        if method == "gauss-legendre":
            # quad points in [-1,1]
            import volumential.meshgen as mg

            if 'queue' in kwargs:
                queue = kwargs['queue']
            else:
                queue = None

            q_points, _, _ = mg.make_uniform_cubic_grid(
                nqpoints=quad_order-1, level=1, dim=self.dim,
                queue=queue)

            # map to source box
            mapped_q_points = np.array(
                [
                    0.5 * self.source_box_extent * (qp + np.ones(self.dim))
                    for qp in q_points.T
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

        if progress_bar:
            from pytools import ProgressBar
            self.pb = ProgressBar("Building table:", total_evals)
        else:
            self.pb = None

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
            np.array([p[self.dim - 1] for p in self.q_points[: self.quad_order + 1]])
            / self.source_box_extent
        )
        assert len(xi) == self.quad_order - 1
        print(xi)

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
                fvals = np.ones(np.array(coords[0]).shape)
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
                fvals = np.ones(np.array(coords[0]).shape)
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

        cheby_nodes, _, cheby_weights = \
            sps.chebyt(cheb_order).weights.T  # pylint: disable=E1136,E0633
        window = [0, 1]

        cheby_nodes = cheby_nodes * (window[1] - window[0]) / 2 + np.mean(window)
        cheby_weights = cheby_weights * (window[1] - window[0]) / 2

        mode = self.get_template_mode(mode_index)
        grid = np.meshgrid(
                *[cheby_nodes for d in range(self.dim)],
                indexing='ij')
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

        # purge small coeffs whose magnitude are less than 8 times machine epsilon
        mode_cheb_coeffs[
                np.abs(mode_cheb_coeffs) < 8 * np.finfo(mode_cheb_coeffs.dtype).eps
                ] = 0

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
                max([abs(cvc) - 2
                     for cvc in self.interaction_case_vecs[case_index]
                     ]) / 2
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

        return entry_id, centry_id

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
        nmlz, err = squad.qquad(
            func=mode_func,
            a=0,
            b=self.source_box_extent,
            c=0,
            d=self.source_box_extent,
            tol=1.,
            rtol=1.,
            minitero=25,
            miniteri=25,
            maxitero=100,
            maxiteri=100,
        )
        # FIXME: cannot pickle logger
        if err > 1e-15:
            logger.debug("Normalizer %d quad error is %e" % (
                mode_id, err))
        return (mode_id, nmlz)

    def build_normalizer_table(self, pool=None, pb=None):
        """
        Build normalizers, used for log-scaled kernels,
        currently only supported in 2D.
        """
        assert self.dim == 2
        if 0:
            # FIXME: make everything needed for compute_nmlz picklable
            if pool is None:
                from multiprocessing import Pool

                pool = Pool(processes=None)

            for mode_id, nmlz in pool.imap_unordered(
                self.compute_nmlz, [i for i in range(self.n_q_points)]
            ):
                self.mode_normalizers[mode_id] = nmlz
                if pb is not None:
                    pb.progress(1)
        else:
            for mode_id in range(self.n_q_points):
                _, nmlz = self.compute_nmlz(mode_id)
                self.mode_normalizers[mode_id] = nmlz
                if pb is not None:
                    pb.progress(1)

    def build_table_via_transform(self):
        """
        Build the full data table using transforms to
        remove the singularity.
        """
        assert self.dim == 2

        if 0:
            # FIXME: make everything needed for compute_nmlz picklable
            # multiprocessing cannot handle member functions
            from multiprocessing import Pool
            pool = Pool(processes=None)
        else:
            pool = None

        if self.pb is not None:
            self.pb.draw()

        self.build_normalizer_table(pool, pb=self.pb)
        self.has_normalizers = True

        # First compute entries that are invariant under
        # symmetry lookup
        invariant_entry_ids = [
            i for i in range(len(self.data)) if self.lookup_by_symmetry(i) == (i, i)
        ]

        if 0:
            # multiprocess disabled to remove dependency on dill/multiprocess
            for entry_id, entry_val in pool.imap_unordered(
                self.compute_table_entry, invariant_entry_ids
            ):
                self.data[entry_id] = entry_val
                if self.pb is not None:
                    self.pb.progress(1)
        else:
            for entry_id in invariant_entry_ids:
                _, entry_val = self.compute_table_entry(entry_id)
                self.data[entry_id] = entry_val
                if self.pb is not None:
                    self.pb.progress(1)

        if 0:
            # Then complete the table via symmetry lookup
            for entry_id, centry_id in pool.imap_unordered(
                    self.lookup_by_symmetry, [i for i in range(len(self.data))]):
                assert not np.isnan(self.data[centry_id])
                if centry_id == entry_id:
                    continue
                self.data[entry_id] = self.data[centry_id]
                if self.pb is not None:
                    self.pb.progress(1)
        else:
            for entry_id in range(len(self.data)):
                _, centry_id = self.lookup_by_symmetry(entry_id)
                assert not np.isnan(self.data[centry_id])
                if centry_id == entry_id:
                    continue
                self.data[entry_id] = self.data[centry_id]
                if self.pb is not None:
                    self.pb.progress(1)

        if self.pb is not None:
            self.pb.finished()

        for entry in self.data:
            assert not np.isnan(entry)

        self.is_built = True

    # }}} End build table via transform

    # {{{ build table via adding up a Droste of bricks

    def get_droste_table_builder(self, n_brick_quad_points,
                                 special_radial_brick_quadrature,
                                 nradial_brick_quad_points,
                                 use_symmetry=False,
                                 knl_symmetry_tags=None):
        if self.inverse_droste:
            from volumential.droste import InverseDrosteReduced

            drf = InverseDrosteReduced(
                self.integral_knl, self.quad_order, self.interaction_case_vecs,
                n_brick_quad_points, knl_symmetry_tags, auto_windowing=False,
                special_radial_quadrature=special_radial_brick_quadrature,
                nradial_quad_points=nradial_brick_quad_points)

        else:
            if not use_symmetry:
                from volumential.droste import DrosteFull

                drf = DrosteFull(
                    self.integral_knl, self.quad_order,
                    self.interaction_case_vecs, n_brick_quad_points,
                    special_radial_quadrature=special_radial_brick_quadrature,
                    nradial_quad_points=nradial_brick_quad_points)
            else:
                from volumential.droste import DrosteReduced

                drf = DrosteReduced(
                    self.integral_knl, self.quad_order, self.interaction_case_vecs,
                    n_brick_quad_points, knl_symmetry_tags,
                    special_radial_quadrature=special_radial_brick_quadrature,
                    nradial_quad_points=nradial_brick_quad_points)
        return drf

    def build_table_via_droste_bricks(
        self,
        n_brick_quad_points=50,
        alpha=0,
        cl_ctx=None,
        queue=None,
        adaptive_level=True,
        adaptive_quadrature=True,
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

        if "special_radial_brick_quadrature" in kwargs:
            special_radial_brick_quadrature = kwargs.pop(
                "special_radial_brick_quadrature")
            nradial_brick_quad_points = kwargs.pop("nradial_brick_quad_points")
        else:
            special_radial_brick_quadrature = False
            nradial_brick_quad_points = None

        if use_symmetry:
            if "knl_symmetry_tags" in kwargs:
                knl_symmetry_tags = kwargs["knl_symmetry_tags"]
            else:
                # Maximum symmetry by default
                logger.warn(
                        "use_symmetry is set to True, but knl_symmetry_tags is not "
                        "set. Using the default maximum symmetry. (Using maximum "
                        "symmetry for some kernels (e.g. derivatives of "
                        "LaplaceKernel will yield incorrect results)."
                        )
                knl_symmetry_tags = None

        # extra_kernel_kwargs = {}
        # if "extra_kernel_kwargs" in kwargs:
        #     extra_kernel_kwargs = kwargs["extra_kernel_kwargs"]

        cheb_coefs = [
            self.get_mode_cheb_coeffs(mid, self.quad_order)
            for mid in range(self.n_q_points)
            ]

        # compute an initial table
        drf = self.get_droste_table_builder(
            n_brick_quad_points,
            special_radial_brick_quadrature, nradial_brick_quad_points,
            use_symmetry, knl_symmetry_tags)
        data0 = drf(queue, source_box_extent=self.source_box_extent,
                    alpha=alpha, nlevels=nlev,
                    # extra_kernel_kwargs=extra_kernel_kwargs,
                    cheb_coefs=cheb_coefs, **kwargs)

        # {{{ adaptively determine number of levels

        resid = -1
        missing_measure = 1
        if adaptive_level:
            table_tol = np.finfo(self.dtype).eps * 256  # 5e-14 for float64
            logger.warn("Searching for nlevels since adaptive_level=True")

            while True:

                missing_measure = (
                        alpha ** nlev * self.source_box_extent
                        ) ** self.dim
                if missing_measure < np.finfo(self.dtype).eps * 128:
                    logger.warn(
                            "Adaptive level refinement terminated "
                            "at %d since missing measure is minuscule "
                            "(%e)" % (nlev, missing_measure))
                    break

                nlev = nlev + 1
                data1 = drf(queue, source_box_extent=self.source_box_extent,
                            alpha=alpha, nlevels=nlev,
                            # extra_kernel_kwargs=extra_kernel_kwargs,
                            cheb_coefs=cheb_coefs, **kwargs)

                resid = np.max(np.abs(data1 - data0)) / np.max(np.abs(data1))
                data0 = data1

                if abs(resid) < table_tol:
                    logger.warn(
                        "Adaptive level refinement "
                        "converged at level %d with residual %e" % (
                            nlev - 1, resid))
                    break

                if np.isnan(resid):
                    logger.warn(
                        "Adaptive level refinement terminated "
                        "at %d before converging due to NaNs" % nlev)
                    break

            if resid >= table_tol:
                logger.warn("Adaptive level refinement failed to converge.")
                logger.warn(f"Residual at level {nlev} equals to {resid}")

        # }}} End adaptively determine number of levels

        # {{{ adaptively determine brick quad order

        if adaptive_quadrature:
            table_tol = np.finfo(self.dtype).eps * 256  # 5e-14 for float64
            logger.warn("Searching for n_brick_quad_points since "
                        "adaptive_quadrature=True. Note that if you are using "
                        "special radial quadrature, the radial order will also be "
                        "adaptively refined.")

            max_n_quad_pts = 1000
            resid = np.inf

            while True:

                n_brick_quad_points += max(int(n_brick_quad_points * 0.2), 3)
                if special_radial_brick_quadrature:
                    nradial_brick_quad_points += max(
                        int(nradial_brick_quad_points * 0.2), 3)
                    logger.warn(
                        f"Trying n_brick_quad_points = {n_brick_quad_points}, "
                        f"nradial_brick_quad_points = {nradial_brick_quad_points}, "
                        f"resid = {resid}")
                else:
                    logger.warn(
                        f"Trying n_brick_quad_points = {n_brick_quad_points}, "
                        f"resid = {resid}")
                if n_brick_quad_points > max_n_quad_pts:
                    logger.warn(
                            "Adaptive quadrature refinement terminated "
                            "since order %d exceeds the max order "
                            "allowed (%d)" % (
                                n_brick_quad_points - 1,
                                max_n_quad_pts - 1))
                    break

                drf = self.get_droste_table_builder(
                    n_brick_quad_points,
                    special_radial_brick_quadrature, nradial_brick_quad_points,
                    use_symmetry, knl_symmetry_tags)
                data1 = drf(queue, source_box_extent=self.source_box_extent,
                            alpha=alpha, nlevels=nlev,
                            n_brick_quad_points=n_brick_quad_points,
                            # extra_kernel_kwargs=extra_kernel_kwargs,
                            cheb_coefs=cheb_coefs, **kwargs)

                resid_prev = resid
                resid = np.max(np.abs(data1 - data0)) / np.max(np.abs(data1))
                data0 = data1

                if resid < table_tol:
                    logger.warn(
                        "Adaptive quadrature "
                        "converged at order %d with residual %e" % (
                            n_brick_quad_points - 1, resid))
                    break

                if resid > resid_prev:
                    logger.warn("Non-monotonic residual, breaking..")
                    break

                if np.isnan(resid):
                    logger.warn(
                        "Adaptive quadrature terminated "
                        "at %d before converging due to NaNs" % nlev)
                    break

            if resid >= table_tol:
                logger.warn("Adaptive quadrature failed to converge.")
                logger.warn(f"Residual at order {n_brick_quad_points} "
                            f"equals to {resid}")

            if resid < 0:
                logger.warn("Failed to perform quadrature order refinement.")

        # }}} End adaptively determine brick quad order

        self.data = data0

        # {{{ (only for 2D) compute normalizers
        # NOTE: normalizers are for log kernels and not needed in 3D

        if self.dim == 2:
            self.build_normalizer_table()
            self.has_normalizers = True
        else:
            self.has_normalizers = False

        if self.inverse_droste:
            assert cl_ctx
            self.build_kernel_exterior_normalizer_table(cl_ctx, queue, **kwargs)

        # }}} End Compute normalizers

        self.is_built = True

    # }}} End build table via adding up a Droste of bricks

    # {{{ build table (driver)

    def build_table(self, cl_ctx=None, queue=None, **kwargs):
        method = self.build_method
        if method == "Transform":
            logger.info("Building table with transform method")
            self.build_table_via_transform()
        elif method == "DrosteSum":
            logger.info("Building table with Droste method")
            self.build_table_via_droste_bricks(cl_ctx=cl_ctx,
                    queue=queue, **kwargs)
        else:
            raise NotImplementedError()

    # }}} End build table (driver)

    # {{{ build kernel exterior normalizer table

    def build_kernel_exterior_normalizer_table(self, cl_ctx, queue,
            pool=None, ncpus=None,
            mesh_order=5, quad_order=10, mesh_size=0.03,
            remove_tmp_files=True,
            **kwargs):
        r"""Build the kernel exterior normalizer table for fractional Laplacians.

        An exterior normalizer for kernel :math:`G(r)` and target
        :math:`x` is defined as

        .. math::

            \int_{B^c} G(\lVert x - y \rVert) dy

        where :math:`B` is the source box :math:`[0, source_box_extent]^dim`.
        """
        logger.warn("this method is currently under construction.")

        if not self.inverse_droste:
            raise ValueError()

        if ncpus is None:
            import multiprocessing
            ncpus = multiprocessing.cpu_count()

        if pool is None:
            from multiprocessing import Pool
            pool = Pool(ncpus)

        def fl_scaling(k, s):
            # scaling constant
            from scipy.special import gamma
            return (
                    2**(2 * s) * s * gamma(s + k / 2)
                    ) / (
                            np.pi**(k / 2) * gamma(1 - s)
                            )

        # Directly compute and return in 1D
        if self.dim == 1:
            s = self.integral_knl.s

            targets = np.array(self.q_points).reshape(-1)
            r1 = targets
            r2 = self.source_box_extent - targets
            self.kernel_exterior_normalizers = 1/(2*s) * (
                    1 / r1**(2*s) + 1 / r2**(2*s)
                    ) * fl_scaling(k=self.dim, s=s)
            return

        from meshmode.array_context import PyOpenCLArrayContext
        from meshmode.dof_array import thaw, flatten
        from meshmode.mesh.io import read_gmsh
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory

        # {{{ gmsh processing

        import gmsh

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
        logger.debug("r_inner = %f, r_outer = %f" % (hs, r))

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

        arr_ctx = PyOpenCLArrayContext(queue)
        discr = Discretization(arr_ctx, mesh,
                PolynomialWarpAndBlendGroupFactory(order=quad_order))

        from pytential import bind, sym

        # {{{ optional checks

        if 1:
            if self.dim == 2:
                arerr = np.abs(
                        (np.pi * r**2 - (2 * hs)**2)
                        - bind(discr, sym.integral(self.dim, self.dim, 1))(queue)
                        ) / (np.pi * r**2 - (2 * hs)**2)
                if arerr > 1e-12:
                    log_to = logger.warn
                else:
                    log_to = logger.debug
                log_to("the numerical error when computing the measure of a "
                    "unit ball is %e" % arerr)

            elif self.dim == 3:
                arerr = np.abs(
                        (4 / 3 * np.pi * r**3 - (2 * hs)**3)
                        - bind(discr, sym.integral(self.dim, self.dim, 1))(queue)
                        ) / (4 / 3 * np.pi * r**3 - (2 * hs)**3)
                if arerr > 1e-12:
                    log_to = logger.warn
                else:
                    log_to = logger.debug
                logger.warn("The numerical error when computing the measure of a "
                    "unit ball is %e" % arerr)

        # }}} End optional checks

        # {{{ kernel evaluation

        # TODO: take advantage of symmetry if this is too slow

        from volumential.droste import InverseDrosteReduced

        # only for getting kernel evaluation related stuff
        drf = InverseDrosteReduced(
                self.integral_knl, self.quad_order,
                self.interaction_case_vecs, n_brick_quad_points=0,
                knl_symmetry_tags=[], auto_windowing=False)

        # uses "dist[dim]", assigned to "knl_val"
        knl_insns = drf.get_sumpy_kernel_insns()

        eval_kernel_insns = [
            insn.copy(within_inames=insn.within_inames | frozenset(["iqpt"]))
            for insn in knl_insns
        ]

        from sumpy.symbolic import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()

        scaling_assignment = lp.Assignment(
            id=None,
            assignee="knl_scaling",
            expression=sympy_conv(self.integral_knl.get_global_scaling_const()),
            temp_var_type=lp.Optional(),
        )

        extra_kernel_kwarg_types = ()
        if "extra_kernel_kwarg_types" in kwargs:
            extra_kernel_kwarg_types = kwargs["extra_kernel_kwarg_types"]

        lpknl = lp.make_kernel(  # NOQA
            "{ [iqpt, iaxis]: 0<=iqpt<n_q_points and 0<=iaxis<dim }",
            [
                """
                for iqpt
                    for iaxis
                        <> dist[iaxis] = (quad_points[iaxis, iqpt]
                            - target_point[iaxis])
                    end
                end
                """
            ]
            + eval_kernel_insns
            + [scaling_assignment]
            + [
                """
                for iqpt
                    result[iqpt] = knl_val * knl_scaling
                end
                """
                ],
            [
                lp.ValueArg("dim, n_q_points", np.int32),
                lp.GlobalArg("quad_points", np.float64, "dim, n_q_points"),
                lp.GlobalArg("target_point", np.float64, "dim")
                ] + list(extra_kernel_kwarg_types)
            + ["...", ],
            name="eval_kernel_lucky_coin",
            lang_version=(2018, 2),
        )

        lpknl = lp.fix_parameters(lpknl, dim=self.dim)
        lpknl = lp.set_options(lpknl, write_cl=False)
        lpknl = lp.set_options(lpknl, return_dict=True)

        # }}} End kernel evaluation

        node_coords = flatten(thaw(arr_ctx, discr.nodes()))
        nodes = cl.array.to_device(queue,
            np.vstack([crd.get() for crd in node_coords]))

        int_vals = []

        for target in self.q_points:
            evt, res = lpknl(queue, quad_points=nodes, target_point=target)
            knl_vals = res['result']

            integ = bind(discr,
                    sym.integral(self.dim, self.dim, sym.var("integrand")))(
                            queue,
                            integrand=knl_vals)
            queue.finish()
            int_vals.append(integ)

        int_vals_coins = np.array(int_vals)

        int_vals_inf = np.zeros(self.n_q_points)

        # {{{ integrate over the exterior of the ball

        if self.dim == 2:

            def rho_0(theta, target, radius):
                rho_x = np.linalg.norm(target, ord=2)
                return (
                    -1 * rho_x * np.cos(theta)
                    + np.sqrt(radius**2 - rho_x**2 * (np.sin(theta)**2))
                )

            def ext_inf_integrand(theta, s, target, radius):
                _rho_0 = rho_0(theta, target=target, radius=radius)
                return _rho_0**(-2 * s)

            def compute_ext_inf_integral(target, s, radius):
                # target: target point
                # s: fractional order
                # radius: radius of the circle
                import scipy.integrate as sint
                val, _ = sint.quadrature(
                    partial(ext_inf_integrand,
                        s=s, target=target, radius=radius),
                    a=0,
                    b=2*np.pi
                )
                return val * (1 / (2 * s)) * fl_scaling(k=self.dim, s=s)

            if 1:
                # optional test
                target = [0, 0]
                s = 0.5
                radius = 1
                scaling = fl_scaling(k=self.dim, s=s)
                val = compute_ext_inf_integral(target, s, radius)
                test_err = np.abs(
                        val
                        - radius**(-2 * s) * 2 * np.pi * (1 / (2 * s)) * scaling
                        ) / (radius**(-2 * s) * 2 * np.pi * (1 / (2 * s)) * scaling)
                if test_err > 1e-12:
                    logger.warn(
                            "Error evaluating at origin = %f" % test_err)

            for tid, target in enumerate(self.q_points):
                # The formula assumes that the source box is centered at origin
                int_vals_inf[tid] = compute_ext_inf_integral(
                        target=target - hs, s=self.integral_knl.s, radius=r)

        elif self.dim == 3:
            # FIXME
            raise NotImplementedError("3D not yet implemented.")

        else:
            raise NotImplementedError("Unsupported dimension")

        # }}} End integrate over the exterior of the ball

        self.kernel_exterior_normalizers = int_vals_coins + int_vals_inf
        return

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
