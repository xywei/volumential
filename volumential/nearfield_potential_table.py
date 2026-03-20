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
import time
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.interpolate import BarycentricInterpolator as Interpolator

import loopy as lp
import pyopencl as cl
from pytools import memoize_method

import volumential.list1_gallery as gallery
import volumential.singular_integral_2d as squad


logger = logging.getLogger("NearFieldInteractionTable")


_TABLE_BUILD_METHOD = "DuffyRadial"
_DUFFY_PROGRESS_STAGES = 3


@dataclass(frozen=True)
class DuffyBuildConfig:
    radial_rule: str = "tanh-sinh-fast"
    regular_quad_order: object = 20
    radial_quad_order: object = 61
    mp_dps: int = 50
    auto_tune_orders: bool = False
    auto_tune_samples: int = 5
    auto_tune_floor_factor: float = 8.0
    auto_tune_candidates: object = None


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


def _is_auto_quad_order(value):
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


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
            return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        def quadratic_formula_2(a, b, c):
            return (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        def citardauq_formula_1(a, b, c):
            return 2 * c / (-b - np.sqrt(b**2 - 4 * a * c))

        def citardauq_formula_2(a, b, c):
            return 2 * c / (-b + np.sqrt(b**2 - 4 * a * c))

        def f(x):
            return x**2 - b * x + c

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
                    - (np.log(r / 2) + euler_constant) * (r**2 / 4 + r**4 / 64)
                    + (r**2 / 4 + r**4 * 3 / 128)
                )

            if approx_at_origin:
                return (
                    -1
                    / (2 * np.pi * (lam1**2 - lam2**2))
                    * (k0_approx(r, lam1) - k0_approx(r, lam2))
                )
            else:
                return (
                    -1
                    / (2 * np.pi * (lam1**2 - lam2**2))
                    * (sp.kn(0, lam1 * r) - sp.kn(0, lam2 * r))
                )

        return cahn_hilliard


def get_cahn_hilliard_laplacian(dim, b=0, c=0):
    raise NotImplementedError("Cahn-Hilliard-Laplacian kernel function not implemented")


# }}} End kernel function getters


def sumpy_kernel_to_lambda(sknl):
    from sympy import Symbol, lambdify, symbols

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


class NearFieldInteractionTable:
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
        build_method=_TABLE_BUILD_METHOD,
        source_box_extent=1,
        dtype=np.float64,
        progress_bar=True,
        **kwargs,
    ):
        """
        kernel_type determines how the kernel is scaled w.r.t. box size.
        The table build method is DuffyRadial.

        The source box is [0, source_box_extent]^dim
        """
        self.quad_order = quad_order
        self.dim = dim
        self.dtype = dtype

        assert source_box_extent > 0
        self.source_box_extent = source_box_extent

        self.center = np.ones(self.dim) * 0.5 * self.source_box_extent

        if build_method is None:
            build_method = _TABLE_BUILD_METHOD
        if build_method != _TABLE_BUILD_METHOD:
            raise NotImplementedError(
                f"Unsupported build_method={build_method!r}; use {_TABLE_BUILD_METHOD}."
            )

        self.build_method = _TABLE_BUILD_METHOD
        self._auto_build_queue = None

        if kernel_func is None and sumpy_kernel is not None:
            kernel_func = sumpy_kernel_to_lambda(sumpy_kernel)

        if dim == 1:
            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel

        elif dim == 2:
            # Constant kernel can be used for fun/testing
            if kernel_func is None:
                kernel_func = constant_one
                kernel_type = "const"

            # Kernel function differs from OpenCL's kernels
            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel

        elif dim == 3:
            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel

        else:
            raise NotImplementedError

        # number of quad points per box
        # equals to the number of modes per box
        self.n_q_points = self.quad_order**dim

        # Normalizers for polynomial modes
        # Needed only when we want to rescale log type kernels
        self.mode_normalizers = np.zeros(self.n_q_points, dtype=self.dtype)

        # Exterior normalizers for hypersingular kernels
        self.kernel_exterior_normalizers = np.zeros(self.n_q_points, dtype=self.dtype)

        # number of (source_mode, target_point) pairs between two boxes
        self.n_pairs = self.n_q_points**2

        # possible interaction cases
        self.interaction_case_vecs, self.case_encode, self.case_indices = (
            gallery.generate_list1_gallery(self.dim)
        )
        self.n_cases = len(self.interaction_case_vecs)

        precomputed_q_points = kwargs.pop("precomputed_q_points", None)

        if precomputed_q_points is not None:
            q_points = np.asarray(precomputed_q_points, dtype=self.dtype)
            expected_shape = (self.n_q_points, self.dim)
            if q_points.shape != expected_shape:
                raise ValueError(
                    "precomputed_q_points has shape "
                    f"{q_points.shape}, expected {expected_shape}"
                )
            self.q_points = q_points

        elif method == "gauss-legendre":
            # quad points in [-1,1]
            import volumential.meshgen as mg

            if "queue" in kwargs:
                queue = kwargs["queue"]
            else:
                queue = None

            q_points, _, _ = mg.make_uniform_cubic_grid(
                degree=quad_order, level=1, dim=self.dim, queue=queue
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

        if self.dim == 2:
            total_evals = self.n_q_points + _DUFFY_PROGRESS_STAGES
        else:
            total_evals = _DUFFY_PROGRESS_STAGES

        if progress_bar:
            from pytools import ProgressBar

            self.pb = ProgressBar("Building table:", total_evals)
        else:
            self.pb = None

        self.is_built = False
        self.last_duffy_order_selection = None
        self.last_duffy_build_timings = None
        self.table_data_is_symmetry_reduced = False

    # }}} End constructor

    # {{{ encode to table index

    def get_entry_index(self, source_mode_index, target_point_index, case_id):

        assert source_mode_index >= 0 and source_mode_index < self.n_q_points
        assert target_point_index >= 0 and target_point_index < self.n_q_points
        pair_id = source_mode_index * self.n_q_points + target_point_index

        if self.table_data_is_symmetry_reduced:
            symmetry_maps = self._get_online_symmetry_maps()
            source_mode_index_sym = int(
                symmetry_maps["mode_qpoint_map"][source_mode_index, source_mode_index]
            )
            target_point_index_sym = int(
                symmetry_maps["mode_qpoint_map"][source_mode_index, target_point_index]
            )
            case_id = int(symmetry_maps["mode_case_map"][source_mode_index, case_id])
            pair_id = source_mode_index_sym * self.n_q_points + target_point_index_sym

        return case_id * self.n_pairs + pair_id

    # }}} End encode to table index

    # {{{ decode table index to entry info

    def decode_index(self, entry_id):
        """This is the inverse function of get_entry_index()"""

        index_info = {}

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
                mode_index // (self.quad_order**2),
                mode_index % (self.quad_order**2) // self.quad_order,
                mode_index % (self.quad_order**2) % self.quad_order,
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
            coords0 = np.asarray(coords[0])
            is_scalar = all(np.asarray(coord).ndim == 0 for coord in coords)
            fvals = np.ones(coords0.shape, dtype=self.dtype)
            for d, coord in zip(range(self.dim), coords):
                val = axis_interp[d](np.array(coord))
                if is_scalar:
                    val = np.asarray(val)
                    if val.size == 1:
                        val = val.item()
                fvals = np.multiply(fvals, val)
            if is_scalar and fvals.size == 1:
                return fvals.item()
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

        def _to_source_box_coords(coord):
            coord = np.asarray(coord)
            remapped = 0.5 * (coord + self.source_box_extent)
            return np.where(
                np.logical_or(coord < 0, coord > self.source_box_extent),
                remapped,
                coord,
            )

        def mode(*coords):
            assert len(coords) == self.dim
            coords0 = _to_source_box_coords(coords[0])
            is_scalar = all(np.asarray(coord).ndim == 0 for coord in coords)
            fvals = np.ones(coords0.shape, dtype=self.dtype)
            for d, coord in zip(range(self.dim), coords):
                val = axis_interp[d](_to_source_box_coords(coord))
                if is_scalar:
                    val = np.asarray(val)
                    if val.size == 1:
                        val = val.item()
                fvals = np.multiply(fvals, val)
            if is_scalar and fvals.size == 1:
                return fvals.item()
            return fvals

        return mode

    def get_mode_cheb_coeffs(self, mode_index, cheb_order):
        """
        Cheb coeffs of a mode.
        The projection process is performed on [0,1]^dim.
        """

        import scipy.special as sps

        cheby_nodes, _, cheby_weights = sps.chebyt(cheb_order).weights.T  # pylint: disable=E1136,E0633
        window = [0, 1]

        cheby_nodes = cheby_nodes * (window[1] - window[0]) / 2 + np.mean(window)
        cheby_weights = cheby_weights * (window[1] - window[0]) / 2

        mode = self.get_template_mode(mode_index)
        grid = np.meshgrid(*[cheby_nodes for d in range(self.dim)], indexing="ij")
        mvals = mode(*grid)

        from numpy.polynomial.chebyshev import Chebyshev

        coef_scale = 2 * np.ones(cheb_order) / cheb_order
        coef_scale[0] /= 2
        basis_1d = np.array(
            [
                Chebyshev(coef=_orthonormal(cheb_order, i), domain=window)(cheby_nodes)
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

    @memoize_method
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
                max([abs(cvc) - 2 for cvc in self.interaction_case_vecs[case_index]])
                / 2
            )

        # print(vec, new_cntr, new_size)

        return new_cntr + new_size * (self.q_points[target_point_index] - self.center)

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

        vec_map, qp_map = self.get_symmetry_transform(entry_info["source_mode_index"])
        # mapped (canonical) case_id
        case_vec = self.interaction_case_vecs[entry_info["case_index"]]
        cc_vec = vec_map(case_vec)
        cc_id = self.case_indices[self.case_encode(cc_vec)]
        cs_id = qp_map(entry_info["source_mode_index"])
        ct_id = qp_map(entry_info["target_point_index"])
        centry_id = self.get_entry_index(cs_id, ct_id, cc_id)

        return entry_id, centry_id

    def compute_table_entry_duffy_radial(
        self,
        entry_id,
        radial_rule="tanh-sinh-fast",
        deg_theta=20,
        radial_quad_order=61,
        mp_dps=50,
    ):
        entry_info = self.decode_index(entry_id)
        source_mode = self.get_mode(entry_info["source_mode_index"])
        target_point = self.find_target_point(
            target_point_index=entry_info["target_point_index"],
            case_index=entry_info["case_index"],
        )

        def integrand(x, y):
            return source_mode(x, y) * self.kernel_func(
                x - target_point[0], y - target_point[1]
            )

        if self.dim == 2:
            integral, error = squad.box_quad_duffy_radial(
                func=integrand,
                a=0,
                b=self.source_box_extent,
                c=0,
                d=self.source_box_extent,
                singular_point=target_point,
                args=(),
                radial_rule=radial_rule,
                deg_theta=deg_theta,
                radial_quad_order=radial_quad_order,
                mp_dps=mp_dps,
            )
        else:

            def integrand_nd(*coords):
                source_val = source_mode(*coords)
                shifted = [coords[i] - target_point[i] for i in range(self.dim)]
                return source_val * self.kernel_func(*shifted)

            bounds = [(0.0, self.source_box_extent) for _ in range(self.dim)]
            integral, error = squad.box_quad_duffy_radial_nd(
                func=integrand_nd,
                bounds=bounds,
                singular_point=target_point,
                args=(),
                radial_rule=radial_rule,
                deg_regular=deg_theta,
                radial_quad_order=radial_quad_order,
                mp_dps=mp_dps,
            )

        return (entry_id, integral)

    def _get_barycentric_data(self):
        xi = np.array([p[self.dim - 1] for p in self.q_points[: self.quad_order]])
        bw = np.ones(self.quad_order, dtype=self.dtype)
        for i in range(self.quad_order):
            for j in range(self.quad_order):
                if i != j:
                    bw[i] /= xi[i] - xi[j]
        return xi.astype(self.dtype), bw.astype(self.dtype)

    @memoize_method
    def _get_invariant_entry_info(self):
        sym_maps = self._get_online_symmetry_maps()
        mode_qpoint_map = sym_maps["mode_qpoint_map"]
        mode_case_map = sym_maps["mode_case_map"]
        all_mode_axes = self._get_all_mode_axes()

        qpoint_indices = np.arange(self.n_q_points, dtype=np.int32)
        case_indices_all = np.arange(self.n_cases, dtype=np.int32)

        entry_chunks = []
        case_chunks = []
        target_chunks = []
        source_chunks = []

        n_q_points_i64 = np.int64(self.n_q_points)
        n_pairs_i64 = np.int64(self.n_pairs)

        for source_mode_index in range(self.n_q_points):
            if (
                mode_qpoint_map[source_mode_index, source_mode_index]
                != source_mode_index
            ):
                continue

            fixed_targets = qpoint_indices[
                mode_qpoint_map[source_mode_index] == qpoint_indices
            ]
            if fixed_targets.size == 0:
                continue

            fixed_cases = case_indices_all[
                mode_case_map[source_mode_index] == case_indices_all
            ]
            if fixed_cases.size == 0:
                continue

            pair_base = np.int64(
                source_mode_index
            ) * n_q_points_i64 + fixed_targets.astype(np.int64)
            entry_block = (
                fixed_cases.astype(np.int64)[:, np.newaxis] * n_pairs_i64
                + pair_base[np.newaxis, :]
            )

            entry_chunks.append(entry_block.reshape(-1))
            case_chunks.append(np.repeat(fixed_cases, fixed_targets.size))
            target_chunks.append(np.tile(fixed_targets, fixed_cases.size))
            source_chunks.append(
                np.full(entry_block.size, source_mode_index, dtype=np.int32)
            )

        if entry_chunks:
            invariant_entry_ids = np.concatenate(entry_chunks).astype(
                np.int64, copy=False
            )
            case_indices = np.concatenate(case_chunks).astype(np.int32, copy=False)
            target_point_indices = np.concatenate(target_chunks).astype(
                np.int32, copy=False
            )
            source_mode_indices = np.concatenate(source_chunks).astype(
                np.int32, copy=False
            )
        else:
            invariant_entry_ids = np.empty(0, dtype=np.int64)
            case_indices = np.empty(0, dtype=np.int32)
            target_point_indices = np.empty(0, dtype=np.int32)
            source_mode_indices = np.empty(0, dtype=np.int32)

        mode_axes = np.ascontiguousarray(
            all_mode_axes[source_mode_indices], dtype=np.int32
        )

        return {
            "entry_ids": invariant_entry_ids,
            "case_indices": case_indices,
            "target_point_indices": target_point_indices,
            "source_mode_indices": source_mode_indices,
            "mode_axes": mode_axes,
        }

    @memoize_method
    def _get_case_target_points(self):
        target_points = np.empty(
            (self.dim, self.n_cases, self.n_q_points), dtype=self.dtype
        )

        q_offsets = (self.q_points - self.center).T
        for case_index in range(self.n_cases):
            case_vec = np.asarray(
                self.interaction_case_vecs[case_index], dtype=self.dtype
            )
            vec = case_vec / 4.0 * self.source_box_extent
            new_cntr = (
                np.ones(self.dim, dtype=self.dtype) * (0.5 * self.source_box_extent)
                + vec
            )

            if int(np.max(np.abs(case_vec))) == 0:
                new_size = self.dtype(1)
            else:
                new_size = self.dtype(np.max(np.abs(case_vec) - 2.0) / 2.0)

            target_points[:, case_index, :] = (
                new_cntr.reshape(self.dim, 1) + new_size * q_offsets
            )

        return target_points

    @memoize_method
    def _get_all_mode_axes(self):
        axes = np.empty((self.n_q_points, self.dim), dtype=np.int32)
        residual = np.arange(self.n_q_points, dtype=np.int64)
        for axis in range(self.dim - 1, -1, -1):
            axes[:, axis] = residual % self.quad_order
            residual //= self.quad_order
        return axes

    @memoize_method
    def _get_online_symmetry_maps(self):
        mode_qpoint_map = np.empty((self.n_q_points, self.n_q_points), dtype=np.int32)
        mode_case_map = np.empty((self.n_q_points, self.n_cases), dtype=np.int32)
        all_mode_axes = self._get_all_mode_axes()
        case_vecs = np.asarray(self.interaction_case_vecs, dtype=np.int32)
        multi_shape = (self.quad_order,) * self.dim

        for source_mode_index in range(self.n_q_points):
            source_axes = all_mode_axes[source_mode_index].astype(np.int64)
            s1 = np.sign((self.quad_order - 0.5) / 2.0 - source_axes)
            reflected_source = source_axes.copy()
            reflected_source[s1 < 0] = self.quad_order - 1 - reflected_source[s1 < 0]
            s2 = np.argsort(np.abs(reflected_source), kind="stable")

            mapped_axes = all_mode_axes
            if np.any(s1 < 0):
                mapped_axes = mapped_axes.copy()
                mapped_axes[:, s1 < 0] = self.quad_order - 1 - mapped_axes[:, s1 < 0]
            mapped_axes = mapped_axes[:, s2]
            mode_qpoint_map[source_mode_index, :] = np.ravel_multi_index(
                mapped_axes.T,
                multi_shape,
            ).astype(np.int32)

            mapped_case_vecs = case_vecs * s1.astype(np.int32)
            mapped_case_vecs = mapped_case_vecs[:, s2]
            encoded = np.array(
                [self.case_encode(case_vec.tolist()) for case_vec in mapped_case_vecs],
                dtype=np.int64,
            )
            mode_case_map[source_mode_index, :] = self.case_indices[encoded]

        return {
            "mode_qpoint_map": mode_qpoint_map,
            "mode_case_map": mode_case_map,
        }

    @memoize_method
    def _get_fused_invariant_duffy_table_tunit(self):
        from sumpy.assignment_collection import SymbolicAssignmentCollection
        from sumpy.codegen import to_loopy_insns
        from sumpy.symbolic import SympyToPymbolicMapper, make_sym_vector

        dvec = make_sym_vector("d", self.dim)
        sac = SymbolicAssignmentCollection()
        result_name = sac.assign_unique(
            "knl_val",
            self.integral_knl.postprocess_at_target(
                self.integral_knl.postprocess_at_source(
                    self.integral_knl.get_expression(dvec), dvec
                ),
                dvec,
            ),
        )
        sac.run_global_cse()
        loopy_insns = to_loopy_insns(
            sac.assignments.items(),
            vector_names=set(),
            pymbolic_expr_maps=[self.integral_knl.get_code_transformer()],
            retain_names=[result_name],
            complex_dtype=np.complex128,
        )
        quad_inames = frozenset(["ientry", "inode"])
        quad_kernel_insns = [
            insn.copy(within_inames=insn.within_inames | quad_inames)
            for insn in loopy_insns
        ]
        sympy_conv = SympyToPymbolicMapper()
        scaling_assignment = lp.Assignment(
            id=None,
            assignee="knl_scaling",
            expression=sympy_conv(self.integral_knl.get_global_scaling_const()),
            temp_var_type=lp.Optional(),
            within_inames=frozenset(["ientry", "inode"]),
        )

        setup_lines = ["<> active = 1"]
        len_terms = []
        dist_terms = []
        basis_terms = []
        interp_eps = 8 * np.finfo(self.dtype).eps
        for iaxis in range(self.dim):
            len_name = f"len{iaxis}"
            q_name = f"q{iaxis}"
            den_name = f"den{iaxis}"
            basis_name = f"basis{iaxis}"
            d_name = f"d{iaxis}"
            pos_name = f"len_pos{iaxis}"
            tol_name = f"interp_tol{iaxis}"
            hit_mode_name = f"hit_mode{iaxis}"
            hit_any_name = f"hit_any{iaxis}"
            num_name = f"num{iaxis}"
            setup_lines.extend(
                [
                    f"<> {len_name} = if(node_sign[{iaxis}, inode] < 0, decomposition_targets[{iaxis}, ientry], source_box_extent - decomposition_targets[{iaxis}, ientry])",
                    f"<> {pos_name} = ({len_name} > 0)",
                    f"active = active and {pos_name}",
                    f"<> {q_name} = decomposition_targets[{iaxis}, ientry] + node_sign[{iaxis}, inode] * {len_name} * node_u[{iaxis}, inode]",
                    f"<> {d_name} = if({pos_name}, {q_name} - target_points[{iaxis}, ientry], 1)",
                ]
            )
            if self.quad_order == 1:
                setup_lines.append(f"<> {basis_name} = 1")
            else:
                setup_lines.extend(
                    [
                        f"<> {tol_name} = {interp_eps} * source_box_extent",
                        f"<> {hit_mode_name} = if(abs({q_name} - interp_nodes[mode_i[{iaxis}, ientry]]) < {tol_name}, 1, 0)",
                        f"<> {hit_any_name} = sum(jq, if(abs({q_name} - interp_nodes[jq]) < {tol_name}, 1, 0))",
                        f"<> {den_name} = sum(jq, if(abs({q_name} - interp_nodes[jq]) < {tol_name}, 0, bary_w[jq] / ({q_name} - interp_nodes[jq])))",
                        f"<> {num_name} = if({hit_mode_name}, 0, bary_w[mode_i[{iaxis}, ientry]] / ({q_name} - interp_nodes[mode_i[{iaxis}, ientry]]))",
                        f"<> {basis_name} = if({hit_any_name} > 0, {hit_mode_name}, {num_name} / {den_name})",
                    ]
                )
            len_terms.append(len_name)
            dist_terms.append(d_name)
            basis_terms.append(basis_name)

        # Guard against quadrature nodes that hit the singular point exactly.
        # For such nodes, the transformed Jacobian factor drives the contribution
        # to zero, but finite-precision evaluation of the kernel may produce
        # non-finite values (e.g. log(0)) and then 0*inf -> nan.
        dist2_expr = " + ".join(f"{d_name} * {d_name}" for d_name in dist_terms)
        setup_lines.append(f"active = active and ({dist2_expr} > 0)")

        quad_w_expr = "node_jac_base[inode]" + "".join(
            f" * {len_name}" for len_name in len_terms
        )
        basis_expr = " * ".join(basis_terms) if basis_terms else "1"
        setup_body = "\n".join(f"                {line}" for line in setup_lines)

        tunit = lp.make_kernel(
            "{[ientry,inode,jq]: 0<=ientry<n_entries and 0<=inode<n_nodes and 0<=jq<q_order}",
            [
                f"""
            for ientry, inode
{setup_body}
                <> quad_w = {quad_w_expr}
            end"""
            ]
            + quad_kernel_insns
            + [scaling_assignment]
            + [
                """
            result[ientry] = sum(inode, if(active, knl_val * knl_scaling * quad_w * """
                + basis_expr
                + """
                , 0)
            )
            """
            ],
            [
                lp.ValueArg("dim,n_entries,n_nodes,q_order", np.int32),
                lp.ValueArg("source_box_extent", self.dtype),
                lp.GlobalArg("target_points", self.dtype, "dim,n_entries"),
                lp.GlobalArg("decomposition_targets", self.dtype, "dim,n_entries"),
                lp.GlobalArg("node_u", self.dtype, "dim,n_nodes"),
                lp.GlobalArg("node_sign", self.dtype, "dim,n_nodes"),
                lp.GlobalArg("node_jac_base", self.dtype, "n_nodes"),
                lp.GlobalArg("interp_nodes", self.dtype, "q_order"),
                lp.GlobalArg("bary_w", self.dtype, "q_order"),
                lp.GlobalArg("mode_i", np.int32, "dim,n_entries"),
                lp.GlobalArg("result", self.dtype, "n_entries", is_output=True),
                lp.TemporaryVariable("knl_scaling", self.dtype),
            ],
            name="duffy_invariant_fused_table",
            lang_version=(2018, 2),
        )
        tunit = lp.fix_parameters(tunit, dim=self.dim)
        tunit = lp.set_options(tunit, return_dict=True, no_numpy=True)
        tunit = lp.split_iname(tunit, "ientry", 64, outer_tag="g.0", inner_tag="l.0")
        tunit = lp.tag_inames(
            tunit,
            {
                "jq": "unr",
            },
        )
        return tunit

    def _get_fused_invariant_duffy_table_program(self, queue, n_entries, n_nodes):
        knl = lp.fix_parameters(
            self._get_fused_invariant_duffy_table_tunit(),
            n_entries=n_entries,
            n_nodes=n_nodes,
            q_order=self.quad_order,
        )
        return knl

    @memoize_method
    def _get_duffy_radial_node_data(
        self,
        radial_rule,
        regular_quad_order,
        radial_quad_order,
        mp_dps,
    ):
        from itertools import permutations, product

        if self.dim == 1:
            regular_nodes = np.zeros((1, 0), dtype=self.dtype)
            regular_weights = np.ones(1, dtype=self.dtype)
        else:
            regular_nodes, regular_weights = squad._duffy_regular_nodes_weights(
                self.dim - 1, regular_quad_order
            )
            regular_nodes = np.asarray(regular_nodes, dtype=self.dtype)
            regular_weights = np.asarray(regular_weights, dtype=self.dtype)

        rho_nodes, rho_weights = squad._duffy_radial_nodes_weights(
            radial_rule, radial_quad_order, mp_dps
        )
        rho_nodes = np.asarray(rho_nodes, dtype=self.dtype)
        rho_weights = np.asarray(rho_weights, dtype=self.dtype)

        sign_vectors = list(product([-1.0, 1.0], repeat=self.dim))
        axis_permutations = list(permutations(range(self.dim)))

        n_nodes = (
            len(sign_vectors)
            * len(axis_permutations)
            * len(regular_weights)
            * len(rho_weights)
        )

        node_u = np.empty((self.dim, n_nodes), dtype=self.dtype)
        node_sign = np.empty((self.dim, n_nodes), dtype=self.dtype)
        node_jac_base = np.empty(n_nodes, dtype=self.dtype)

        inode = 0
        for signs in sign_vectors:
            sign_arr = np.asarray(signs, dtype=self.dtype)
            for perm in axis_permutations:
                for ireg, w_tail in enumerate(regular_weights):
                    tail_rs = regular_nodes[ireg] if self.dim > 1 else np.array([])
                    for rho, w_rho in zip(rho_nodes, rho_weights):
                        rs = np.empty(self.dim, dtype=self.dtype)
                        rs[0] = rho
                        if self.dim > 1:
                            rs[1:] = tail_rs

                        u = np.empty(self.dim, dtype=self.dtype)
                        cumulative = 1.0
                        for i, axis in enumerate(perm):
                            cumulative *= rs[i]
                            u[axis] = cumulative

                        jac_base = 1.0
                        for i in range(self.dim - 1):
                            jac_base *= rs[i] ** (self.dim - 1 - i)
                        jac_base = jac_base * w_tail * w_rho

                        node_u[:, inode] = u
                        node_sign[:, inode] = sign_arr
                        node_jac_base[inode] = jac_base
                        inode += 1

        return {
            "n_nodes": n_nodes,
            "node_u": node_u,
            "node_sign": node_sign,
            "node_jac_base": node_jac_base,
        }

    def _default_duffy_order_candidates(self):
        if self.dim == 1:
            return [(2, 11), (2, 21), (2, 31), (2, 41)]
        if self.dim in (2, 3):
            return [(4, 11), (4, 21), (6, 21), (6, 31), (8, 31), (10, 41), (12, 61)]
        raise NotImplementedError(
            "Duffy order auto-tuning currently supports dimensions 1-3"
        )

    def _duffy_autotune_sample_entry_ids(self, sample_count):
        invariant_entry_ids = np.asarray(
            self._get_invariant_entry_info()["entry_ids"], dtype=np.int64
        )
        if invariant_entry_ids.size == 0:
            return []

        sample_count = max(1, int(sample_count))
        if sample_count >= invariant_entry_ids.size:
            idx = np.arange(invariant_entry_ids.size, dtype=np.int64)
        else:
            idx = np.linspace(
                0, invariant_entry_ids.size - 1, num=sample_count, dtype=np.int64
            )
            idx = np.unique(idx)

        return [int(invariant_entry_ids[i]) for i in idx]

    def _batched_duffy_values_for_local_indices(
        self,
        queue,
        invariant_info,
        local_entry_indices,
        radial_rule,
        regular_quad_order,
        radial_quad_order,
        mp_dps,
    ):
        local_entry_indices = np.asarray(local_entry_indices, dtype=np.int64)
        n_entries = int(local_entry_indices.size)
        if n_entries == 0:
            return np.empty(0, dtype=self.dtype)

        node_data = self._get_duffy_radial_node_data(
            radial_rule,
            regular_quad_order,
            radial_quad_order,
            mp_dps,
        )
        n_nodes = node_data["n_nodes"]

        all_target_points = self._get_case_target_points()

        mode_axes = invariant_info["mode_axes"][local_entry_indices]
        mode_i = np.ascontiguousarray(mode_axes.T, dtype=np.int32)

        case_indices = invariant_info["case_indices"][local_entry_indices]
        target_point_indices = invariant_info["target_point_indices"][
            local_entry_indices
        ]

        target_points = np.ascontiguousarray(
            all_target_points[:, case_indices, target_point_indices], dtype=self.dtype
        )
        decomposition_targets = np.ascontiguousarray(
            np.clip(target_points, 0.0, self.source_box_extent), dtype=self.dtype
        )

        xi, bw = self._get_barycentric_data()
        prg = self._get_fused_invariant_duffy_table_program(queue, n_entries, n_nodes)

        queue_is_cl = isinstance(queue, cl.CommandQueue)
        if queue_is_cl:
            import pyopencl.array as cla

            target_points_arg = cla.to_device(
                queue, np.ascontiguousarray(target_points, dtype=self.dtype)
            )
            decomposition_targets_arg = cla.to_device(
                queue, np.ascontiguousarray(decomposition_targets, dtype=self.dtype)
            )
            node_u_arg = cla.to_device(
                queue, np.ascontiguousarray(node_data["node_u"], dtype=self.dtype)
            )
            node_sign_arg = cla.to_device(
                queue, np.ascontiguousarray(node_data["node_sign"], dtype=self.dtype)
            )
            node_jac_base_arg = cla.to_device(
                queue,
                np.ascontiguousarray(node_data["node_jac_base"], dtype=self.dtype),
            )
            xi_arg = cla.to_device(queue, np.ascontiguousarray(xi, dtype=self.dtype))
            bw_arg = cla.to_device(queue, np.ascontiguousarray(bw, dtype=self.dtype))
            mode_i_arg = cla.to_device(queue, mode_i)
        else:
            target_points_arg = np.ascontiguousarray(target_points, dtype=self.dtype)
            decomposition_targets_arg = np.ascontiguousarray(
                decomposition_targets, dtype=self.dtype
            )
            node_u_arg = np.ascontiguousarray(node_data["node_u"], dtype=self.dtype)
            node_sign_arg = np.ascontiguousarray(
                node_data["node_sign"], dtype=self.dtype
            )
            node_jac_base_arg = np.ascontiguousarray(
                node_data["node_jac_base"], dtype=self.dtype
            )
            xi_arg = np.ascontiguousarray(xi, dtype=self.dtype)
            bw_arg = np.ascontiguousarray(bw, dtype=self.dtype)
            mode_i_arg = mode_i

        _, res = prg(
            queue,
            source_box_extent=self.dtype(self.source_box_extent),
            target_points=target_points_arg,
            decomposition_targets=decomposition_targets_arg,
            node_u=node_u_arg,
            node_sign=node_sign_arg,
            node_jac_base=node_jac_base_arg,
            interp_nodes=xi_arg,
            bary_w=bw_arg,
            mode_i=mode_i_arg,
        )

        result = res["result"]
        if hasattr(result, "get"):
            result = result.get()
        return np.ascontiguousarray(result, dtype=self.dtype)

    def _auto_tune_duffy_radial_orders(
        self,
        radial_rule,
        mp_dps,
        queue=None,
        sample_count=5,
        candidates=None,
        floor_factor=8.0,
    ):
        if radial_rule == "adaptive":
            raise ValueError(
                "Duffy order auto-tuning is only available for fixed radial rules"
            )

        if candidates is None:
            candidates = self._default_duffy_order_candidates()
        candidates = [tuple(int(x) for x in pair) for pair in candidates]
        if not candidates:
            raise ValueError("Duffy order auto-tuning candidate list cannot be empty")

        sample_entry_ids = self._duffy_autotune_sample_entry_ids(sample_count)
        if not sample_entry_ids:
            selected_regular, selected_radial = candidates[0]
            info = {
                "auto_tuned": True,
                "selected_regular_quad_order": selected_regular,
                "selected_radial_quad_order": selected_radial,
                "sample_entry_ids": [],
                "candidates": candidates,
                "relative_errors_vs_best": [0.0 for _ in candidates],
                "floor_estimate": 0.0,
                "acceptance_threshold": 0.0,
            }
            self.last_duffy_order_selection = info
            return selected_regular, selected_radial, info

        use_batched_eval = (
            isinstance(queue, cl.CommandQueue)
            and self.dim in (1, 2, 3)
            and self.integral_knl is not None
            and radial_rule in {"tanh-sinh-fast", "tanh-sinh"}
        )

        invariant_info = None
        sample_local_indices = None
        if use_batched_eval:
            invariant_info = self._get_invariant_entry_info()
            entry_to_local_index = {
                int(entry_id): i
                for i, entry_id in enumerate(invariant_info["entry_ids"])
            }
            sample_local_indices = np.asarray(
                [entry_to_local_index[entry_id] for entry_id in sample_entry_ids],
                dtype=np.int64,
            )

        candidate_values = []
        for regular_quad_order, radial_quad_order in candidates:
            if use_batched_eval:
                values = self._batched_duffy_values_for_local_indices(
                    queue,
                    invariant_info,
                    sample_local_indices,
                    radial_rule,
                    regular_quad_order,
                    radial_quad_order,
                    mp_dps,
                )
            else:
                values = []
                for entry_id in sample_entry_ids:
                    _, value = self.compute_table_entry_duffy_radial(
                        entry_id,
                        radial_rule=radial_rule,
                        deg_theta=regular_quad_order,
                        radial_quad_order=radial_quad_order,
                        mp_dps=mp_dps,
                    )
                    values.append(value)
                values = np.asarray(values)

            if np.iscomplexobj(values):
                values = values.astype(np.complex128)
            else:
                values = values.astype(np.float64)
            candidate_values.append(values)

        reference_values = candidate_values[-1]
        rel_errors = []
        for values in candidate_values:
            denom = np.maximum(1.0, np.abs(reference_values))
            rel_errors.append(float(np.max(np.abs(values - reference_values) / denom)))

        floor_estimate = 0.0
        if len(rel_errors) > 2:
            tail_errors = np.asarray(
                rel_errors[max(0, len(rel_errors) - 4) : -1],
                dtype=np.float64,
            )
            positive_tail_errors = tail_errors[tail_errors > 0]
            if positive_tail_errors.size >= 2:
                floor_estimate = float(np.min(positive_tail_errors))

        min_floor = 128.0 * np.finfo(np.float64).eps
        floor_estimate = max(floor_estimate, min_floor)
        acceptance_threshold = float(floor_factor) * floor_estimate

        selected_index = len(candidates) - 1
        for i, err in enumerate(rel_errors):
            if err <= acceptance_threshold:
                selected_index = i
                break

        selected_regular, selected_radial = candidates[selected_index]
        info = {
            "auto_tuned": True,
            "selected_regular_quad_order": selected_regular,
            "selected_radial_quad_order": selected_radial,
            "sample_entry_ids": sample_entry_ids,
            "candidates": candidates,
            "relative_errors_vs_best": rel_errors,
            "floor_estimate": floor_estimate,
            "acceptance_threshold": acceptance_threshold,
        }
        self.last_duffy_order_selection = info
        logger.info(
            "Auto-tuned Duffy orders (dim=%d): regular=%d radial=%d "
            "threshold=%.3e floor=%.3e",
            self.dim,
            selected_regular,
            selected_radial,
            acceptance_threshold,
            floor_estimate,
        )
        return selected_regular, selected_radial, info

    def build_table_via_duffy_radial_batched(
        self,
        queue,
        radial_rule="tanh-sinh-fast",
        deg_theta=20,
        radial_quad_order=61,
        mp_dps=50,
    ):
        t_total_start = time.perf_counter()

        if self.dim not in (1, 2, 3):
            raise NotImplementedError(
                "batched DuffyRadial path currently supports dimensions 1-3"
            )
        if self.integral_knl is None:
            raise ValueError("batched DuffyRadial path requires integral_knl")
        if radial_rule == "adaptive":
            raise ValueError("batched DuffyRadial path does not support adaptive rule")

        t_invariant_start = time.perf_counter()
        invariant_info = self._get_invariant_entry_info()
        t_invariant_end = time.perf_counter()
        invariant_entry_ids = invariant_info["entry_ids"]
        n_entries = len(invariant_entry_ids)

        if n_entries == 0:
            self.is_built = True
            self._progress_step(_DUFFY_PROGRESS_STAGES)
            t_total_end = time.perf_counter()
            self.last_duffy_build_timings = {
                "invariant_info_s": t_invariant_end - t_invariant_start,
                "quadrature_s": 0.0,
                "scatter_s": 0.0,
                "total_s": t_total_end - t_total_start,
                "n_entries": 0,
            }
            return

        local_entry_indices = np.arange(n_entries, dtype=np.int64)
        t_quadrature_start = time.perf_counter()
        table_host = self._batched_duffy_values_for_local_indices(
            queue,
            invariant_info,
            local_entry_indices,
            radial_rule,
            deg_theta,
            radial_quad_order,
            mp_dps,
        )
        t_quadrature_end = time.perf_counter()
        self._progress_step()

        t_scatter_start = time.perf_counter()
        for ientry, entry_id in enumerate(invariant_entry_ids):
            self.data[entry_id] = table_host[ientry]
        t_scatter_end = time.perf_counter()
        self._progress_step()

        # Keep table data symmetry-reduced to minimize cache/storage size.
        # Call progress step for the legacy "symmetry fill" stage even though
        # propagation is intentionally disabled.
        t_symmetry_fill_start = time.perf_counter()
        t_symmetry_fill_end = t_symmetry_fill_start
        self._progress_step()

        self.table_data_is_symmetry_reduced = bool(np.any(np.isnan(self.data)))
        self.is_built = True

        t_total_end = time.perf_counter()
        self.last_duffy_build_timings = {
            "invariant_info_s": t_invariant_end - t_invariant_start,
            "quadrature_s": t_quadrature_end - t_quadrature_start,
            "scatter_s": t_scatter_end - t_scatter_start,
            "symmetry_fill_s": t_symmetry_fill_end - t_symmetry_fill_start,
            "total_s": t_total_end - t_total_start,
            "n_entries": int(n_entries),
        }

    def build_table_via_duffy_radial_batched_2d(
        self,
        queue,
        radial_rule="tanh-sinh-fast",
        deg_theta=20,
        radial_quad_order=61,
        mp_dps=50,
    ):
        if self.dim != 2:
            raise ValueError("build_table_via_duffy_radial_batched_2d requires dim=2")
        return self.build_table_via_duffy_radial_batched(
            queue,
            radial_rule=radial_rule,
            deg_theta=deg_theta,
            radial_quad_order=radial_quad_order,
            mp_dps=mp_dps,
        )

    def compute_nmlz(self, mode_id):
        mode_func = self.get_mode(mode_id)
        nmlz, err = squad.qquad(
            func=mode_func,
            a=0,
            b=self.source_box_extent,
            c=0,
            d=self.source_box_extent,
            tol=1.0,
            rtol=1.0,
            minitero=25,
            miniteri=25,
            maxitero=100,
            maxiteri=100,
        )
        # FIXME: cannot pickle logger
        if err > 1e-15:
            logger.debug("Normalizer %d quad error is %e" % (mode_id, err))
        return (mode_id, nmlz)

    def _build_normalizer_table_fast_gauss_legendre(self, pb=None):
        if self.dim != 2:
            return False

        ref_nodes, ref_weights = np.polynomial.legendre.leggauss(self.quad_order)
        expected_nodes = 0.5 * self.source_box_extent * (ref_nodes + 1.0)

        # Mode construction assumes these are the 1D interpolation nodes in
        # ascending order. If this assumption is violated, use the generic
        # integration fallback.
        axis_nodes = np.array(
            [p[self.dim - 1] for p in self.q_points[: self.quad_order]],
            dtype=np.float64,
        )
        node_tol = 1024 * np.finfo(np.float64).eps * max(1.0, self.source_box_extent)
        if not np.allclose(axis_nodes, expected_nodes, atol=node_tol, rtol=0.0):
            return False

        axis_weights = (0.5 * self.source_box_extent * ref_weights).astype(self.dtype)
        self.mode_normalizers[:] = np.multiply.outer(
            axis_weights, axis_weights
        ).reshape(-1)

        if pb is not None:
            pb.progress(self.n_q_points)

        return True

    def build_normalizer_table(self, pool=None, pb=None):
        """
        Build normalizers, used for log-scaled kernels,
        currently only supported in 2D.
        """
        assert self.dim == 2

        if self._build_normalizer_table_fast_gauss_legendre(pb=pb):
            return

        if 0:
            # FIXME: make everything needed for compute_nmlz picklable
            if pool is None:
                from multiprocessing import Pool

                pool = Pool(processes=None)

            for mode_id, nmlz in pool.imap_unordered(
                self.compute_nmlz, range(self.n_q_points)
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

    def _progress_step(self, nsteps=1):
        if self.pb is not None and nsteps > 0:
            self.pb.progress(nsteps)

    def _make_duffy_build_config(
        self,
        radial_rule,
        regular_quad_order,
        radial_quad_order,
        mp_dps,
        auto_tune_orders,
        auto_tune_samples,
        auto_tune_floor_factor,
        auto_tune_candidates,
        kwargs,
    ):
        legacy_kwargs = dict(kwargs)
        if "deg_theta" in legacy_kwargs:
            legacy_deg_theta = legacy_kwargs.pop("deg_theta")
            if legacy_deg_theta is not None:
                regular_quad_order = legacy_deg_theta

        return DuffyBuildConfig(
            radial_rule=radial_rule,
            regular_quad_order=regular_quad_order,
            radial_quad_order=radial_quad_order,
            mp_dps=mp_dps,
            auto_tune_orders=auto_tune_orders,
            auto_tune_samples=auto_tune_samples,
            auto_tune_floor_factor=auto_tune_floor_factor,
            auto_tune_candidates=auto_tune_candidates,
        )

    def _resolve_duffy_build_config(self, build_config, queue):
        regular_quad_order = build_config.regular_quad_order
        radial_quad_order = build_config.radial_quad_order

        auto_requested = (
            build_config.auto_tune_orders
            or _is_auto_quad_order(regular_quad_order)
            or _is_auto_quad_order(radial_quad_order)
        )
        if auto_requested:
            tuned_regular, tuned_radial, _ = self._auto_tune_duffy_radial_orders(
                radial_rule=build_config.radial_rule,
                mp_dps=build_config.mp_dps,
                queue=queue,
                sample_count=build_config.auto_tune_samples,
                candidates=build_config.auto_tune_candidates,
                floor_factor=build_config.auto_tune_floor_factor,
            )
            if build_config.auto_tune_orders or _is_auto_quad_order(regular_quad_order):
                regular_quad_order = tuned_regular
            else:
                regular_quad_order = int(regular_quad_order)

            if build_config.auto_tune_orders or _is_auto_quad_order(radial_quad_order):
                radial_quad_order = tuned_radial
            else:
                radial_quad_order = int(radial_quad_order)
        else:
            regular_quad_order = int(regular_quad_order)
            radial_quad_order = int(radial_quad_order)
            self.last_duffy_order_selection = {
                "auto_tuned": False,
                "selected_regular_quad_order": regular_quad_order,
                "selected_radial_quad_order": radial_quad_order,
            }

        return DuffyBuildConfig(
            radial_rule=build_config.radial_rule,
            regular_quad_order=regular_quad_order,
            radial_quad_order=radial_quad_order,
            mp_dps=build_config.mp_dps,
            auto_tune_orders=build_config.auto_tune_orders,
            auto_tune_samples=build_config.auto_tune_samples,
            auto_tune_floor_factor=build_config.auto_tune_floor_factor,
            auto_tune_candidates=build_config.auto_tune_candidates,
        )

    def _build_table_via_duffy_radial_scalar(
        self,
        radial_rule,
        deg_theta,
        radial_quad_order,
        mp_dps,
    ):
        t_total_start = time.perf_counter()

        t_invariant_start = time.perf_counter()
        invariant_entry_ids = [
            i for i in range(len(self.data)) if self.lookup_by_symmetry(i) == (i, i)
        ]
        t_invariant_end = time.perf_counter()
        self._progress_step()

        t_quadrature_start = time.perf_counter()
        for entry_id in invariant_entry_ids:
            _, entry_val = self.compute_table_entry_duffy_radial(
                entry_id,
                radial_rule=radial_rule,
                deg_theta=deg_theta,
                radial_quad_order=radial_quad_order,
                mp_dps=mp_dps,
            )
            self.data[entry_id] = entry_val
        t_quadrature_end = time.perf_counter()
        self._progress_step()

        # Keep table data symmetry-reduced to minimize cache/storage size.
        # Call progress step for the legacy "symmetry fill" stage even though
        # propagation is intentionally disabled.
        t_symmetry_fill_start = time.perf_counter()
        t_symmetry_fill_end = t_symmetry_fill_start
        self._progress_step()

        self.table_data_is_symmetry_reduced = bool(np.isnan(self.data).any())
        self.is_built = True

        t_total_end = time.perf_counter()
        self.last_duffy_build_timings = {
            "invariant_info_s": t_invariant_end - t_invariant_start,
            "quadrature_s": t_quadrature_end - t_quadrature_start,
            "scatter_s": 0.0,
            "symmetry_fill_s": t_symmetry_fill_end - t_symmetry_fill_start,
            "total_s": t_total_end - t_total_start,
            "n_entries": int(len(invariant_entry_ids)),
        }

    def build_table_via_duffy_radial(
        self,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=20,
        radial_quad_order=61,
        mp_dps=50,
        queue=None,
        cl_ctx=None,
        build_config=None,
        auto_tune_orders=False,
        auto_tune_samples=5,
        auto_tune_floor_factor=8.0,
        auto_tune_candidates=None,
        **kwargs,
    ):
        if build_config is None:
            build_config = self._make_duffy_build_config(
                radial_rule=radial_rule,
                regular_quad_order=regular_quad_order,
                radial_quad_order=radial_quad_order,
                mp_dps=mp_dps,
                auto_tune_orders=auto_tune_orders,
                auto_tune_samples=auto_tune_samples,
                auto_tune_floor_factor=auto_tune_floor_factor,
                auto_tune_candidates=auto_tune_candidates,
                kwargs=kwargs,
            )
        else:
            if not isinstance(build_config, DuffyBuildConfig):
                raise TypeError("build_config must be a DuffyBuildConfig")
            if kwargs:
                build_config = self._make_duffy_build_config(
                    radial_rule=build_config.radial_rule,
                    regular_quad_order=build_config.regular_quad_order,
                    radial_quad_order=build_config.radial_quad_order,
                    mp_dps=build_config.mp_dps,
                    auto_tune_orders=build_config.auto_tune_orders,
                    auto_tune_samples=build_config.auto_tune_samples,
                    auto_tune_floor_factor=build_config.auto_tune_floor_factor,
                    auto_tune_candidates=build_config.auto_tune_candidates,
                    kwargs=kwargs,
                )
        build_config = self._resolve_duffy_build_config(build_config, queue=queue)

        if self.pb is not None:
            self.pb.draw()

        normalizer_s = 0.0
        if self.dim == 2:
            t_normalizer_start = time.perf_counter()
            self.build_normalizer_table(pb=self.pb)
            normalizer_s = time.perf_counter() - t_normalizer_start
            self.has_normalizers = True
        else:
            self.has_normalizers = False

        if build_config.radial_rule == "adaptive":
            logger.warning(
                "Using scalar CPU-backed %dD DuffyRadial table builder "
                "with adaptive radial rule",
                self.dim,
            )
            return_value = self._build_table_via_duffy_radial_scalar(
                radial_rule=build_config.radial_rule,
                deg_theta=int(build_config.regular_quad_order),
                radial_quad_order=int(build_config.radial_quad_order),
                mp_dps=build_config.mp_dps,
            )
            if self.last_duffy_build_timings is not None:
                self.last_duffy_build_timings["normalizer_s"] = normalizer_s
                self.last_duffy_build_timings["total_with_normalizer_s"] = (
                    self.last_duffy_build_timings["total_s"] + normalizer_s
                )
            if self.pb is not None:
                self.pb.finished()
            return return_value

        if queue is None:
            if cl_ctx is not None:
                queue = cl.CommandQueue(cl_ctx)
            else:
                if self._auto_build_queue is None:
                    auto_ctx = cl.create_some_context(interactive=False)
                    self._auto_build_queue = cl.CommandQueue(auto_ctx)
                queue = self._auto_build_queue

        if self.integral_knl is None:
            raise ValueError(
                "DuffyRadial loopy builder requires sumpy_kernel (integral_knl)"
            )

        if build_config.radial_rule not in {"tanh-sinh-fast", "tanh-sinh"}:
            raise ValueError(
                "DuffyRadial loopy builder supports only tanh-sinh and "
                "tanh-sinh-fast radial rules"
            )

        logger.warning(
            "Using batched GPU-backed %dD DuffyRadial table builder", self.dim
        )
        return_value = self.build_table_via_duffy_radial_batched(
            queue,
            radial_rule=build_config.radial_rule,
            deg_theta=int(build_config.regular_quad_order),
            radial_quad_order=int(build_config.radial_quad_order),
            mp_dps=build_config.mp_dps,
        )
        if self.last_duffy_build_timings is not None:
            self.last_duffy_build_timings["normalizer_s"] = normalizer_s
            self.last_duffy_build_timings["total_with_normalizer_s"] = (
                self.last_duffy_build_timings["total_s"] + normalizer_s
            )
        if self.pb is not None:
            self.pb.finished()
        return return_value

    # }}} End build table via DuffyRadial

    # {{{ build table (driver)

    def build_table(self, cl_ctx=None, queue=None, build_config=None, **kwargs):
        logger.info("Building table with Duffy+radial method")
        self.build_table_via_duffy_radial(
            queue=queue,
            cl_ctx=cl_ctx,
            build_config=build_config,
            **kwargs,
        )

    # }}} End build table (driver)

    # {{{ build kernel exterior normalizer table

    def build_kernel_exterior_normalizer_table(
        self,
        cl_ctx,
        queue,
        pool=None,
        ncpus=None,
        mesh_order=5,
        quad_order=10,
        mesh_size=0.03,
        remove_tmp_files=True,
        **kwargs,
    ):
        r"""Build the kernel exterior normalizer table for fractional Laplacians.

        An exterior normalizer for kernel :math:`G(r)` and target
        :math:`x` is defined as

        .. math::

            \int_{B^c} G(\lVert x - y \rVert) dy

        where :math:`B` is the source box :math:`[0, source_box_extent]^dim`.
        """
        logger.warn("this method is currently under construction.")

        if ncpus is None:
            import multiprocessing

            ncpus = multiprocessing.cpu_count()

        if pool is None:
            from multiprocessing import Pool

            pool = Pool(ncpus)

        def fl_scaling(k, s):
            # scaling constant
            from scipy.special import gamma

            return (2 ** (2 * s) * s * gamma(s + k / 2)) / (
                np.pi ** (k / 2) * gamma(1 - s)
            )

        # Directly compute and return in 1D
        if self.dim == 1:
            s = self.integral_knl.s

            targets = np.array(self.q_points).reshape(-1)
            r1 = targets
            r2 = self.source_box_extent - targets
            self.kernel_exterior_normalizers = (
                1
                / (2 * s)
                * (1 / r1 ** (2 * s) + 1 / r2 ** (2 * s))
                * fl_scaling(k=self.dim, s=s)
            )
            return

        # {{{ gmsh processing
        import gmsh

        from meshmode.array_context import PyOpenCLArrayContext
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import (
            PolynomialWarpAndBlendGroupFactory,
        )
        from meshmode.dof_array import flatten, thaw
        from meshmode.mesh.io import read_gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # meshmode does not support other versions
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)
        if mesh_order > 1:
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)

        # radius of source box
        hs = self.source_box_extent / 2
        # radius of bouding sphere
        r = hs * np.sqrt(self.dim)
        logger.debug(f"r_inner = {hs:f}, r_outer = {r:f}")

        if self.dim == 2:
            tag_box = gmsh.model.occ.addRectangle(
                x=0, y=0, z=0, dx=2 * hs, dy=2 * hs, tag=-1
            )
        elif self.dim == 3:
            tag_box = gmsh.model.occ.addBox(
                x=0, y=0, z=0, dx=2 * hs, dy=2 * hs, dz=2 * hs, tag=-1
            )
        else:
            raise NotImplementedError()

        if self.dim == 2:
            tag_ball = gmsh.model.occ.addDisk(xc=hs, yc=hs, zc=0, rx=r, ry=r, tag=-1)
        elif self.dim == 3:
            tag_sphere = gmsh.model.occ.addSphere(xc=hs, yc=hs, zc=hs, radius=r, tag=-1)
            tag_ball = gmsh.model.occ.addVolume([tag_sphere], tag=-1)
        else:
            raise NotImplementedError()

        dimtags_ints, dimtags_map_ints = gmsh.model.occ.cut(
            objectDimTags=[(self.dim, tag_ball)],
            toolDimTags=[(self.dim, tag_box)],
            tag=-1,
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(self.dim)

        from os.path import join
        from tempfile import mkdtemp

        temp_dir = mkdtemp(prefix="tmp_volumential_nft")
        msh_filename = join(temp_dir, "chinese_lucky_coin.msh")
        gmsh.write(msh_filename)
        gmsh.finalize()

        mesh = read_gmsh(msh_filename)
        if remove_tmp_files:
            import shutil

            shutil.rmtree(temp_dir)

        # }}} End gmsh processing

        arr_ctx = PyOpenCLArrayContext(queue)
        discr = Discretization(
            arr_ctx, mesh, PolynomialWarpAndBlendGroupFactory(order=quad_order)
        )

        from pytential import bind, sym

        # {{{ optional checks

        if 1:
            if self.dim == 2:
                arerr = np.abs(
                    (np.pi * r**2 - (2 * hs) ** 2)
                    - bind(discr, sym.integral(self.dim, self.dim, 1))(queue)
                ) / (np.pi * r**2 - (2 * hs) ** 2)
                if arerr > 1e-12:
                    log_to = logger.warn
                else:
                    log_to = logger.debug
                log_to(
                    "the numerical error when computing the measure of a "
                    "unit ball is %e" % arerr
                )

            elif self.dim == 3:
                arerr = np.abs(
                    (4 / 3 * np.pi * r**3 - (2 * hs) ** 3)
                    - bind(discr, sym.integral(self.dim, self.dim, 1))(queue)
                ) / (4 / 3 * np.pi * r**3 - (2 * hs) ** 3)
                if arerr > 1e-12:
                    log_to = logger.warn
                else:
                    log_to = logger.debug
                logger.warn(
                    "The numerical error when computing the measure of a "
                    "unit ball is %e" % arerr
                )

        # }}} End optional checks

        # {{{ kernel evaluation

        # TODO: take advantage of symmetry if this is too slow

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        from sumpy.codegen import to_loopy_insns
        from sumpy.symbolic import SympyToPymbolicMapper, make_sym_vector

        dvec = make_sym_vector("dist", self.dim)
        sac = SymbolicAssignmentCollection()
        result_name = sac.assign_unique(
            "knl_val",
            self.integral_knl.postprocess_at_target(
                self.integral_knl.postprocess_at_source(
                    self.integral_knl.get_expression(dvec), dvec
                ),
                dvec,
            ),
        )
        sac.run_global_cse()
        knl_insns = to_loopy_insns(
            sac.assignments.items(),
            vector_names={"dist"},
            pymbolic_expr_maps=[self.integral_knl.get_code_transformer()],
            retain_names=[result_name],
            complex_dtype=np.complex128,
        )

        eval_kernel_insns = [
            insn.copy(within_inames=insn.within_inames | frozenset(["iqpt"]))
            for insn in knl_insns
        ]

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

        lpknl = lp.make_kernel(
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
                lp.GlobalArg("target_point", np.float64, "dim"),
            ]
            + list(extra_kernel_kwarg_types)
            + [
                "...",
            ],
            name="eval_kernel_lucky_coin",
            lang_version=(2018, 2),
        )

        lpknl = lp.fix_parameters(lpknl, dim=self.dim)
        lpknl = lp.set_options(lpknl, write_cl=False)
        lpknl = lp.set_options(lpknl, return_dict=True)

        # }}} End kernel evaluation

        node_coords = flatten(thaw(arr_ctx, discr.nodes()))
        nodes = cl.array.to_device(queue, np.vstack([crd.get() for crd in node_coords]))

        int_vals = []

        for target in self.q_points:
            evt, res = lpknl(queue, quad_points=nodes, target_point=target)
            knl_vals = res["result"]

            integ = bind(discr, sym.integral(self.dim, self.dim, sym.var("integrand")))(
                queue, integrand=knl_vals
            )
            queue.finish()
            int_vals.append(integ)

        int_vals_coins = np.array(int_vals)

        int_vals_inf = np.zeros(self.n_q_points)

        # {{{ integrate over the exterior of the ball

        if self.dim == 2:

            def rho_0(theta, target, radius):
                rho_x = np.linalg.norm(target, ord=2)
                return -1 * rho_x * np.cos(theta) + np.sqrt(
                    radius**2 - rho_x**2 * (np.sin(theta) ** 2)
                )

            def ext_inf_integrand(theta, s, target, radius):
                _rho_0 = rho_0(theta, target=target, radius=radius)
                return _rho_0 ** (-2 * s)

            def compute_ext_inf_integral(target, s, radius):
                # target: target point
                # s: fractional order
                # radius: radius of the circle
                import scipy.integrate as sint

                val, _ = sint.quadrature(
                    partial(ext_inf_integrand, s=s, target=target, radius=radius),
                    a=0,
                    b=2 * np.pi,
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
                    val - radius ** (-2 * s) * 2 * np.pi * (1 / (2 * s)) * scaling
                ) / (radius ** (-2 * s) * 2 * np.pi * (1 / (2 * s)) * scaling)
                if test_err > 1e-12:
                    logger.warn("Error evaluating at origin = %f" % test_err)

            for tid, target in enumerate(self.q_points):
                # The formula assumes that the source box is centered at origin
                int_vals_inf[tid] = compute_ext_inf_integral(
                    target=target - hs, s=self.integral_knl.s, radius=r
                )

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
            displacement = (a**2) * np.log(a) * self.mode_normalizers[source_mode_index]
            scaling = a**2

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
