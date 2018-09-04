from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.interpolate import BarycentricInterpolator as Interpolator

import volumential.list1_gallery as gallery
import volumential.singular_integral_2d as squad

import loopy as lp
import islpy as isl
import six

# NOTE: Cannot use sumpy kernels for "Transform" since they
# do not work with multiprocess.
# e.g. with LaplaceKernel it will complain "AttributeError:
# 'LaplaceKernel' object has no attribute 'expression'"


def _self_tp(vec):
    assert len(vec.shape) == 1
    return vec.reshape([len(vec), 1]) * vec.reshape([1, len(vec)])


def _orthonormal(n, i):
    eb = np.zeros(n)
    eb[i] = 1
    return eb


def constant_one(x, y=None, z=None):
    return np.ones(np.array(x).shape)


def get_laplace(dim):
    if dim != 2:
        raise NotImplementedError(
            "Kernel function Laplace" +
            str(dim) +
            "D not implemented.")
    else:

        def laplace(x, y):
            return -0.25 * np.log(
                np.array(x)**2 +
                np.array(y)**2) / np.pi

        return laplace


def get_cahn_hilliard(dim):
    if dim != 2:
        raise NotImplementedError(
            "Kernel function Laplace" +
            str(dim) +
            "D not implemented.")
    else:

        def cahn_hilliard(x, y):
            return 0

        return cahn_hilliard


def sumpy_kernel_to_lambda(sknl):
    from sympy import Symbol, symbols, lambdify
    var_name_prefix = "x"
    var_names = " ".join([
        var_name_prefix + str(i)
        for i in range(sknl.dim)
    ])
    arg_names = symbols(var_names)
    args = [
        Symbol(var_name_prefix + str(i))
        for i in range(sknl.dim)
    ]

    def func(x, y=None, z=None):
        coord = (x, y, z)
        lmd = lambdify(
            arg_names,
            sknl.get_expression(args) *
            sknl.
            get_global_scaling_const())
        return lmd(*coord[:sknl.dim])

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
            dtype=np.float64):
        """
        kernel_type determines how the kernel is scaled w.r.t. box size.
        build_method can be "Transform" or "DrosteSum".

        The source box is [0, source_box_extent]^dim
        """
        self.quad_order = quad_order
        self.dim = dim
        self.dtype = dtype

        assert source_box_extent > 0
        self.source_box_extent = source_box_extent

        self.center = np.ones(
            self.dim) * 0.5 * self.source_box_extent

        self.build_method = build_method

        if dim == 2:

            # Constant kernel can be used for fun/testing
            if kernel_func is None:
                kernel_func = constant_one
                kernel_type = "const"

            # Kernel function differs from OpenCL's kernels
            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel

            if build_method == "DrosteSum":
                assert sumpy_kernel is not None

        else:
            raise NotImplemented

        # number of quad points per box
        # equals to the number of modes per box
        self.n_q_points = self.quad_order**dim

        # Normalizers for polynomial modes
        self.mode_normalizers = np.zeros(
            self.n_q_points,
            dtype=self.dtype)

        # number of (source_mode, target_point) pairs between two boxes
        self.n_pairs = self.n_q_points**2

        # possible interaction cases
        self.interaction_case_vecs, self.case_encode, self.case_indices \
            = gallery.generate_list1_gallery(self.dim)
        self.n_cases = len(
            self.interaction_case_vecs)

        if method == "gauss-legendre":
            # quad points in [-1,1]
            import volumential.meshgen as mg
            q_points, _, _ = mg.make_uniform_cubic_grid(
                degree=quad_order,
                level=1)
            # map to source box
            mapped_q_points = np.array([
                0.5 * self.source_box_extent *
                (qp + np.array([1, 1]))
                for qp in q_points
            ])
            # sort in dictionary order, preserve only the leading
            # digits to prevent floating point errors from polluting
            # the ordering.
            q_points_ordering = sorted(range(len(mapped_q_points)),
                                       key=lambda i: list(
                                           np.floor(
                                               mapped_q_points[i] * 10000)
                                           )
                                       )
            self.q_points = mapped_q_points[
                q_points_ordering]

        else:
            raise NotImplemented

        self.data = np.empty(
            self.n_pairs * self.n_cases,
            dtype=self.dtype)
        self.data.fill(np.nan)

        total_evals = len(
            self.data) + self.n_q_points

        from pytools import ProgressBar
        self.pb = ProgressBar(
            "Building table:",
            total_evals)

        self.is_built = False

    # }}} End constructor

    # {{{ encode to table index

    def get_entry_index(
            self, source_mode_index,
            target_point_index,
            case_id):

        assert (source_mode_index >= 0
                and source_mode_index <
                self.n_q_points)
        assert (target_point_index >= 0
                and target_point_index <
                self.n_q_points)
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

        index_info[
            "case_index"] = case_id
        index_info[
            "source_mode_index"] = source_mode_index
        index_info[
            "target_point_index"] = target_point_index

        return index_info

    # }}} End decode table index to entry info

    # {{{ basis modes in the template box

    def get_template_mode(self, mode_index):
        assert (mode_index >= 0 and
                mode_index <
                self.n_q_points)

        # NOTE: these two lines should be changed
        # in accordance with the mesh generator
        idx0 = mode_index // self.quad_order
        idx1 = mode_index % self.quad_order

        xi = np.array([
            p[self.dim - 1]
            for p in self.
            q_points[:self.quad_order]
        ]) / self.source_box_extent
        assert (
            len(xi) == self.quad_order)
        y0i = np.zeros(
            self.quad_order,
            dtype=self.dtype)
        y0i[idx0] = 1
        y1i = np.zeros(
            self.quad_order,
            dtype=self.dtype)
        y1i[idx1] = 1

        interp_x = Interpolator(xi, y0i)
        interp_y = Interpolator(xi, y1i)

        def mode(x, y):
            return np.multiply(
                interp_x(np.array(x)),
                interp_y(np.array(y)))

        return mode

    def get_mode(self, mode_index):
        assert (mode_index >= 0 and
                mode_index <
                self.n_q_points)

        # NOTE: these two lines should be changed
        # in accordance with the mesh generator
        idx0 = mode_index // self.quad_order
        idx1 = mode_index % self.quad_order

        xi = np.array([
            p[self.dim - 1]
            for p in self.
            q_points[:self.quad_order]
        ])
        assert (
            len(xi) == self.quad_order)
        y0i = np.zeros(
            self.quad_order,
            dtype=self.dtype)
        y0i[idx0] = 1
        y1i = np.zeros(
            self.quad_order,
            dtype=self.dtype)
        y1i[idx1] = 1

        interp_x = Interpolator(xi, y0i)
        interp_y = Interpolator(xi, y1i)

        def mode(x, y):
            return np.multiply(
                interp_x(np.array(x)),
                interp_y(np.array(y)))

        return mode

    def get_mode_cheb_coeffs(self, mode_index, cheb_order):

        import scipy.special as sps
        cheby_nodes, _, cheby_weights = sps.chebyt(cheb_order).weights.T
        window = [0, 1]

        cheby_nodes = cheby_nodes * (window[1] - window[0]) / 2 + np.mean(window)
        cheby_weights = cheby_weights * (window[1] - window[0]) / 2

        mode = self.get_template_mode(mode_index)
        xx, yy = np.meshgrid(cheby_nodes, cheby_nodes)
        mvals = mode(xx, yy)

        from numpy.polynomial.chebyshev import Chebyshev
        coef_scale = 2 * np.ones(cheb_order) / cheb_order
        coef_scale[0] /= 2
        basis_1d = np.array([Chebyshev(
            coef=_orthonormal(cheb_order, i),
            domain=window
            )(cheby_nodes) for i in range(cheb_order)])

        from itertools import product
        basis_2d = np.array([
            b1.reshape([cheb_order, 1]) * b2.reshape([1, cheb_order])
            for b1, b2 in product(basis_1d, basis_1d)])

        mode_cheb_coeffs = np.array([
            np.sum(mvals * basis) for basis in basis_2d
            ]) * _self_tp(coef_scale).reshape(-1)

        # purge small coeffs that are 15 digits away
        mm = np.max(np.abs(mode_cheb_coeffs))
        mode_cheb_coeffs[np.abs(mode_cheb_coeffs) < mm * 1e-15] = 0

        return mode_cheb_coeffs

    # }}} End basis modes in the template box

    # {{{ build table via transform

    def get_symmetry_transform(
            self, source_mode_index):
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
        for d in range(
                -1, -1 - self.dim, -1):
            k[d] = resid % self.quad_order
            resid = resid // self.quad_order

        s1 = np.sign((self.quad_order -
                      0.5) / 2 - k)
        for d in range(self.dim):
            if s1[d] < 0:
                k[d] = self.quad_order - 1 - k[d]

        s2 = sorted(
            range(len(k)),
            key=lambda i: abs(k[i]))

        def symmetry_transform(vec):
            nv = np.array(vec) * s1
            return nv[s2]

        def qpoint_index_transform(
                index):
            k = np.zeros(
                self.dim, dtype=int)
            resid = index
            for d in range(
                    -1, -1 - self.dim,
                    -1):
                k[d] = resid % self.quad_order
                resid = resid // self.quad_order
            assert (resid == 0)
            for d in range(self.dim):
                if s1[d] < 0:
                    k[d] = self.quad_order - 1 - k[d]
            k = k[s2]
            new_id = 0
            for d in range(self.dim):
                new_id = new_id * int(
                    self.quad_order
                ) + k[d]
            return new_id

        return (symmetry_transform,
                qpoint_index_transform)

    def find_target_point(
            self, target_point_index,
            case_index):
        """Apply proper transforms to find the target point's coordinate.

        Only translations and scalings are allowed in this step, avoiding the
        indices of quad points to be messed up.
        """
        assert (target_point_index >= 0
                and target_point_index <
                self.n_q_points)

        # rescale to source box with size 1x1
        vec = np.array(
            self.interaction_case_vecs[
                case_index]) / 4.0 * self.source_box_extent

        new_cntr = np.ones(
            self.dim, dtype=self.dtype
        ) * 0.5 * self.source_box_extent + vec

        if int(max(abs(np.array(
            self.interaction_case_vecs[case_index])))) == 0:
            new_size = 1
        else:
            new_size = max([
                abs(l) - 2
                for l in self.interaction_case_vecs[case_index]
            ]) / 2

        # print(vec, new_cntr, new_size)

        return new_cntr + new_size * (
            self.
            q_points[target_point_index]
            - self.center)

    def lookup_by_symmetry(self,
                           entry_id):
        """Loop up table entry that is mapped to a region where:
            - k_i <= q/2 in all direction i
            - k_i's are sorted in ascending order

            Returns the mapped entry_id
        """

        entry_info = self.decode_index(
            entry_id)
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
            entry_info[
                "source_mode_index"])
        # mapped (canonical) case_id
        case_vec = self.interaction_case_vecs[
            entry_info["case_index"]]
        cc_vec = vec_map(case_vec)
        cc_id = self.case_indices[
            self.case_encode(cc_vec)]
        cs_id = qp_map(entry_info[
            "source_mode_index"])
        ct_id = qp_map(entry_info[
            "target_point_index"])
        centry_id = self.get_entry_index(
            cs_id, ct_id, cc_id)

        return centry_id

    def compute_table_entry(self,
                            entry_id):
        """Compute one entry in the table indexed by self.data[entry_id]

        Input kernel function should be centered at origin.
        """
        entry_info = self.decode_index(
            entry_id)
        source_mode = self.get_mode(
            entry_info[
                "source_mode_index"])
        target_point = self.find_target_point(
            target_point_index=entry_info[
                "target_point_index"],
            case_index=entry_info[
                "case_index"])

        # print(entry_info, target_point)
        # source_point = (
        #     self.q_points[entry_info[
        #         "source_mode_index"]])
        # print(source_mode(source_point[0], source_point[1]))

        if self.dim == 2:

            def integrand(x, y):
                return source_mode(
                    x, y
                ) * self.kernel_func(
                    x - target_point[0],
                    y - target_point[1])

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
                maxiter=301)
        else:
            raise NotImplemented

        return (entry_id, integral)

    def compute_nmlz(self, mode_id):
        mode_func = self.get_mode(
            mode_id)
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
            maxiteri=100)
        return (mode_id, nmlz)

    def build_normalizer_table(self, pool=None, pb=None):
        """
        Build normalizers only
        """
        if pool is None:
            from multiprocess import Pool
            pool = Pool(processes=None)

        for mode_id, nmlz in pool.imap_unordered(
                self.compute_nmlz, [
                    i
                    for i in range(
                        self.n_q_points)
                ]):
            self.mode_normalizers[
                mode_id] = nmlz
            if pb is not None:
                pb.progress(1)

    def build_table_via_transform(self):
        """
        Build the full data table using transforms to
        remove the singularity.
        """

        # multiprocessing cannot handle member functions
        from multiprocess import Pool

        pool = Pool(processes=None)

        self.pb.draw()

        # import volumential.pickle_class_method

        self.build_normalizer_table(pool, pb=self.pb)

        # First compute entries that are invariant under
        # symmetry lookup
        invariant_entry_ids = [
            i
            for i in range(
                len(self.data))
            if self.lookup_by_symmetry(
                i) == i
        ]

        for entry_id, entry_val in pool.imap_unordered(
                self.
                compute_table_entry,
                invariant_entry_ids):
            self.data[
                entry_id] = entry_val
            self.pb.progress(1)

        # Then complete the table via symmetry lookup
        for entry_id, centry_id in enumerate(
                pool.imap_unordered(
                    self.
                    lookup_by_symmetry,
                    [
                        i
                        for i in range(
                            len(
                                self.
                                data))
                    ])):
            assert (not np.isnan(
                self.data[centry_id]))
            if centry_id == entry_id:
                continue
            self.data[
                entry_id] = self.data[
                    centry_id]
            self.pb.progress(1)

        self.pb.finished()

        for entry in self.data:
            assert (not np.isnan(entry))

        self.is_built = True

    # }}} End build table via transform

    # {{{ build table via adding up a Droste of bricks

    def get_sumpy_kernel_insns(self):

        from sumpy.symbolic import make_sym_vector
        dvec = make_sym_vector("dist", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        result_name = sac.assign_unique("knl_val",
                self.integral_knl.postprocess_at_target(
                    self.integral_knl.postprocess_at_source(
                        self.integral_knl.get_expression(dvec),
                        dvec),
                    dvec)
                )

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(six.iteritems(sac.assignments),
                vector_names=set(["dist"]),
                pymbolic_expr_maps=[
                    self.integral_knl.get_code_transformer()],
                retain_names=[result_name],
                complex_dtype=np.complex128  # FIXME
                )

        # for i in loopy_insns:
        #     print(i)

        return loopy_insns

    def build_table_via_droste_bricks(self,
            n_brick_quad_points=15, alpha=0.5, queue=None, adaptive_level=True,
            **kwargs):

        # FIXME: take advantage of symmetry? (Yet bad for vectorization)

        if queue is None:
            import pyopencl as cl
            cl_ctx = cl.create_some_context(interactive=True)
            queue = cl.CommandQueue(cl_ctx)

        d = self.dim
        integral_knl = self.integral_knl    # sumpy kernel, the integral kernel
        nfunctions = max(self.quad_order, 3)  # number of box-wide 1D Cheby modes
        ntgt_points = self.quad_order       # number of 1D target points
        nquad_points = n_brick_quad_points  # number of 1D brick quad points
        coar_nquad_points = int(n_brick_quad_points * 0.5 * (1-alpha))
        print(nquad_points, coar_nquad_points)

        case_vecs = np.array([
            [v[0] for v in self.interaction_case_vecs],
            [v[1] for v in self.interaction_case_vecs]])

        case_scls = np.array([
            1 if int(max(abs(np.array(vec)))) == 0
            else max([abs(l) - 0.5 for l in np.array(vec)/4]) * 2
            for vec in self.interaction_case_vecs])

        # print(case_vecs, case_scls)

        # {{{ build loop domain

        # use fewer quad points in some direction (q0)

        tgt_vars = ["t%d" % i for i in range(d)]
        basis_vars = ["f%d" % i for i in range(d)]
        fine_quad_vars = ["q%d" % i for i in range(1, d)]
        coar_quad_vars = ["q0"]
        basis_eval_vars = ["p%d" % i for i in range(d)]
        pwaffs = isl.make_zero_and_vars(
            tgt_vars + basis_vars + basis_eval_vars + fine_quad_vars + coar_quad_vars
            + ["iside", "iaxis", "ibrick_side", "ibrick_axis", "ilevel", "icase"],
            ["nlevels"])

        def make_brick_domain(pwaffs, variables, n, lbound=0):
            if isinstance(n, str):
                ubound_pwaff = pwaffs[n]
            else:
                ubound_pwaff = pwaffs[0] + n

            from functools import reduce
            import operator
            return reduce(operator.and_, [
                (pwaffs[0]+lbound).le_set(pwaffs[var]) for var in variables
                ] + [
                pwaffs[var].lt_set(ubound_pwaff) for var in variables
                ])

        domain = (
                make_brick_domain(pwaffs, tgt_vars, ntgt_points)
                & make_brick_domain(pwaffs, fine_quad_vars, nquad_points)
                & make_brick_domain(pwaffs, coar_quad_vars, coar_nquad_points)
                & make_brick_domain(pwaffs, basis_vars, nfunctions)
                & make_brick_domain(pwaffs, basis_eval_vars, nfunctions, lbound=2)
                & make_brick_domain(pwaffs, ["iside"], 2)
                & make_brick_domain(pwaffs, ["iaxis"], d)
                & make_brick_domain(pwaffs, ["ibrick_side"], 2)
                & make_brick_domain(pwaffs, ["ibrick_axis"], d)
                & make_brick_domain(pwaffs, ["ilevel"], "nlevels")
                & make_brick_domain(pwaffs, ["icase"], len(case_vecs[0]))
                )

        # print(domain)
        # }}} End build loop domain

        # {{{ sumpy kernel eval

        # integral kernel evaluation
        knl_loopy_insns = self.get_sumpy_kernel_insns()
        quad_inames = frozenset(["icase", "t0", "t1",
            "f0", "f1", "ilevel", "ibrick_axis", "ibrick_side", "q0", "q1"])
        quad_kernel_insns = [
                insn.copy(
                    within_inames=insn.within_inames | quad_inames
                    )
                for insn in knl_loopy_insns]

        from sumpy.symbolic import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()

        scaling_assignment = lp.Assignment(id=None,
                assignee="knl_scaling",
                expression=sympy_conv(integral_knl.get_global_scaling_const()),
                temp_var_type=lp.auto)
        # }}} End sumpy kernel eval

        # {{{ loopy kernel for heavy-lifting
        brick_map_knl = lp.make_kernel( # NOQA
                [domain],
                ["""
        for iaxis
            <> root_center[iaxis] = 0.5 * (
                    root_brick[iaxis, 1] + root_brick[iaxis, 0]) {dup=iaxis}
            <> root_extent[iaxis] = (root_brick[iaxis, 1]
                    - root_brick[iaxis, 0]) {dup=iaxis}
        end

        for ilevel, f0, f1, t0, t1, icase

            # Targets outside projected onto the boundary
            for iaxis
                if iaxis == 0
                    <> template_target[iaxis] = target_nodes[t0] \
                            {id=tplt_tgt1,dup=iaxis}
                else
                    template_target[iaxis] = target_nodes[t1] \
                            {id=tplt_tgt2,dup=iaxis}
                end
            end

            ... nop {id=tplt_tgt_pre,dep=tplt_tgt1:tplt_tgt2}

            # True targets are used for kernel evaluation
            for iaxis
                <> true_target[iaxis] = ( root_center[iaxis]
                         + interaction_case_vecs[iaxis, icase]
                            * 0.25 * root_extent[iaxis]
                         + interaction_case_scls[icase]
                            * (template_target[iaxis] - 0.5)
                            * root_extent[iaxis]
                         ) {id=true_targets,dup=iaxis,dep=tplt_tgt_pre}

            end

            # Projected targets are used for brick construction
            for iaxis
                <> target[iaxis] = if(
                    true_target[iaxis] > root_brick[iaxis, 1],
                    root_brick[iaxis, 1],
                    if(
                        true_target[iaxis] < root_brick[iaxis, 0],
                        root_brick[iaxis, 0],
                        true_target[iaxis])) {dup=iaxis,dep=true_targets}
            end

            for iaxis
                template_target[iaxis] = (0.5
                    + interaction_case_vecs[iaxis, icase] * 0.25
                    + interaction_case_scls[icase] * (template_target[iaxis] - 0.5)
                    ) {id=tplt_tgt3,dup=iaxis,dep=tplt_tgt_pre:true_targets}
            end

            for iaxis
                template_target[iaxis] = if(
                    template_target[iaxis] > 1,
                    1,
                    if(
                        template_target[iaxis] < 0,
                        0,
                        template_target[iaxis])
                    ) {id=tplt_tgt4,dup=iaxis,dep=tplt_tgt3}
            end

            ... nop {id=tplt_tgt,dep=tplt_tgt_pre:tplt_tgt4:true_targets}

            # FIXME: commenting out either of the 2 below ruins the result
            # Debug output
            #for iaxis
            #    projected_template_targets[t0, t1, icase, iaxis] = \
            #            template_target[iaxis] {dup=iaxis,dep=tplt_tgt}
            #end

            # Debug output
            #for iaxis
            #    target_points[ilevel, f0, f1, t0, t1, icase, iaxis
            #        ] = true_target[iaxis] {dup=iaxis,dep=true_targets}
            #end

            for iaxis, iside
                <> outer_brick[iaxis, iside] = (
                        (alpha**ilevel)*root_brick[iaxis,iside]
                        + (1-alpha**ilevel) * target[iaxis])  {dup=iaxis:iside}
                <> inner_brick[iaxis, iside] = (
                        alpha*outer_brick[iaxis,iside]
                        + (1-alpha) * target[iaxis])  {dup=iaxis:iside}
            end

            for iaxis
                <> ob_ext[iaxis] = (outer_brick[iaxis, 1]
                    - outer_brick[iaxis, 0])  {dup=iaxis}
            end

            for ibrick_axis, ibrick_side, q0, q1
                for iaxis
                    <> point[iaxis] = \
                        if(iaxis == ibrick_axis,
                            coar_quadrature_nodes[q0],
                            quadrature_nodes[q1])  {dup=iaxis}
                end

                <> deform = point[ibrick_axis]*(1-alpha)

                for iaxis
                    if iaxis == ibrick_axis
                        <> mapped_point_tmp[iaxis] = (
                            point[ibrick_axis]*(
                                inner_brick[ibrick_axis, ibrick_side]
                                - outer_brick[ibrick_axis, ibrick_side])
                                + outer_brick[ibrick_axis, ibrick_side]) \
                            {id=mpoint1}
                    else
                        <> pre_scale = (
                            point[iaxis]
                            + deform*(template_target[iaxis]-point[iaxis])
                            ) {dep=tplt_tgt}

                        mapped_point_tmp[iaxis] = \
                            ob_ext[iaxis] * pre_scale + outer_brick[iaxis, 0] \
                            {id=mpoint2}
                    end

                    ... nop {id=mpoint,dep=mpoint1:mpoint2}

                    # Re-map the mapped points to [-1,1]^2 for Chebyshev evals
                    <> template_mapped_point_tmp[iaxis] = 0.0 + (
                            mapped_point_tmp[iaxis] - root_center[iaxis]
                            ) / root_extent[iaxis] * 2 {dep=mpoint}

                    # Debug output
                    #mapped_points[ilevel, ibrick_axis,
                    #               ibrick_side, t0, t1, icase, iaxis, q0, q1] = \
                    #    mapped_point_tmp[iaxis]  {dep=mpoint}
                    #mapped_target[t0, t1, icase, iaxis] = target[iaxis]
                end

                for iaxis
                    if iaxis == ibrick_axis
                        <> jac_part = (
                            inner_brick[ibrick_axis, ibrick_side]
                            - outer_brick[ibrick_axis, ibrick_side]
                            )    {id=jpart1}
                    else
                        jac_part = ob_ext[iaxis] * (1-deform)  {id=jpart2}
                    end
                end

                <> jacobian = abs(product(iaxis,
                                          jac_part)) {id=jac,dep=jpart1:jpart2}

                <> dist[iaxis] = (true_target[iaxis]
                                - mapped_point_tmp[iaxis]) {dep=mpoint:true_targets}

            end

        end

        for f0, f1, t0, t1, icase
            for ilevel
                for ibrick_axis, ibrick_side, q0, q1
                    <> T0_0 = 1
                    <> T1_0 = template_mapped_point_tmp[0] {dep=mpoint}
                    <> Tprev_0 = T0_0 {id=t0_0}
                    <> Tcur_0 = T1_0 {id=t1_0,dep=t0_0}

                    for p0
                        <> Tnext_0 = (2 * template_mapped_point_tmp[0] * Tcur_0
                                        - Tprev_0) {id=tnext0,dep=t1_0}
                        Tprev_0 = Tcur_0 {id=tprev_update0,dep=tnext0}
                        Tcur_0 = Tnext_0 {id=tcur_update0,dep=tprev_update0}
                    end

                    <> T0_1 = 1
                    <> T1_1 = template_mapped_point_tmp[1] {dep=mpoint}
                    <> Tprev_1 = T0_1 {id=t0_1}
                    <> Tcur_1 = T1_1 {id=t1_1,dep=t0_1}

                    for p1
                        <> Tnext_1 = (2 * template_mapped_point_tmp[1] * Tcur_1
                                        - Tprev_1) {id=tnext1,dep=t1_1}
                        Tprev_1 = Tcur_1 {id=tprev_update1,dep=tnext1}
                        Tcur_1 = Tnext_1 {id=tcur_update1,dep=tprev_update1}
                    end

                    <> basis_eval0 = (
                        T0_0 * if(f0 == 0, 1, 0)
                        + T1_0 * if(f0 == 1, 1, 0)
                        + sum(p0, if(f0 == p0, Tcur_0, 0))
                        ) {id=basis0,dep=tcur_update0}

                    <> basis_eval1 = (
                        T0_1 * if(f1 == 0, 1, 0)
                        + T1_1 * if(f1 == 1, 1, 0)
                        + sum(p1, if(f1 == p1, Tcur_1, 0))
                        ) {id=basis1,dep=tcur_update1}

                    <> density_val = basis_eval0 * basis_eval1 \
                            {id=density,dep=basis0:basis1}
                end
            end
        end

        for f0, f1, t0, t1, icase
            # the first index is contiguous
            result[f1, f0, t1, t0, icase] = (sum(
                    (ilevel, ibrick_axis, ibrick_side, q0, q1),
                    coar_quadrature_weights[q0]
                    * quadrature_weights[q1]
                    * jacobian
                    * knl_val
                    * density_val
                    )
                    *  knl_scaling
                ) {dep=jac:mpoint:density}
        end
        """]
            + quad_kernel_insns + [scaling_assignment],
            [
            lp.ValueArg("alpha", np.float64),
            lp.ValueArg("n_cases, dim", np.int32),
            lp.GlobalArg("interaction_case_vecs", np.float64, "dim, n_cases"),
            lp.GlobalArg("interaction_case_scls", np.float64, "n_cases"),
            "..."
            ],
        name="brick_map")

        brick_map_knl = lp.fix_parameters(brick_map_knl, d=d)
        brick_map_knl = lp.set_options(brick_map_knl, write_cl=False)
        brick_map_knl = lp.set_options(brick_map_knl, return_dict=True)
        # }}} End loopy kernel for heavy-lifting

        root_brick = np.zeros((d, 2))
        root_brick[:, 1] = self.source_box_extent
        # root_displacement = 0.5 * sum(root_brick[0])
        # root_extent = root_brick[0, 1] - root_brick[0, 0]

        # target points in 1D
        t = np.array([pt[1] for pt in self.q_points[:ntgt_points]])
        t = t / self.source_box_extent

        # quad formula for each brick
        # sps.legendre blows up easily at high order
        import scipy.special as sps
        #legendre_nodes, _, legendre_weights = sps.legendre(
        #        nquad_points).weights.T
        legendre_nodes, legendre_weights = sps.p_roots(nquad_points)
        legendre_nodes = legendre_nodes*0.5 + 0.5
        legendre_weights = legendre_weights * 0.5

        coar_legendre_nodes, coar_legendre_weights = sps.p_roots(
                coar_nquad_points)
        coar_legendre_nodes = coar_legendre_nodes*0.5 + 0.5
        coar_legendre_weights = coar_legendre_weights * 0.5

        from time import time

        t0 = time()

        # target_nodes, quadrature_nodes and quadrature weights are
        # the templates in [0,1] unit interval.
        # They will be mapped accordingly to the bricks.
        # print("Running loopy kernel")

        self.cheb_table = 0
        if adaptive_level:
            min_nlevels = 5
            max_nlevels = 50
        elif 'n_levels' in kwargs:
            min_nlevels = kwargs['n_levels']
            max_nlevels = kwargs['n_levels']
        else:
            min_nlevels = 30
            max_nlevels = 30
        # Keep adding more layers until the results converge
        for nlevels in range(min_nlevels, max_nlevels+5, 5):
            evt, res = brick_map_knl(
                    queue, alpha=alpha,
                    root_brick=root_brick,
                    target_nodes=t.astype(np.float64, copy=True),
                    quadrature_nodes=legendre_nodes,
                    quadrature_weights=legendre_weights,
                    coar_quadrature_nodes=coar_legendre_nodes,
                    coar_quadrature_weights=coar_legendre_weights,
                    interaction_case_vecs=case_vecs.astype(np.float64, copy=True),
                    interaction_case_scls=case_scls.reshape(-1).astype(np.float64,
                        copy=True),
                    n_cases=len(case_vecs[0]),
                    nlevels=nlevels,
                    dim=2)

            if adaptive_level:
                resid = (np.max(np.abs(self.cheb_table - res["result"]))
                        / np.max(np.abs(res["result"])))
            else:
                resid = 1

            if resid < 1e-15:
                self.cheb_table = res["result"]
                print("Adaptive level refinement converged at level", nlevels)
                break

            # indexed by f0, f1, t0, t1, icase
            self.cheb_table = res["result"]

            if not adaptive_level:
                print("n_levels =", max_nlevels)
                break

            if nlevels >= max_nlevels:
                print("Adaptive level refinement failed to converge.")
                print("Residual at level", nlevels, " equals to", resid)

        t1 = time()

        #self.true_targets = res["target_points"]
        #self.mapped_points = res["mapped_points"]
        #self.mapped_target = res["mapped_target"]
        #self.projected_template_targets = res["projected_template_targets"]

        # print("Postprocessing")
        nfp_table = np.zeros([self.n_q_points, *self.cheb_table.shape[2:]])
        # transform to interpolatory basis functions
        for mid in range(self.n_q_points):
            mccoefs = self.get_mode_cheb_coeffs(
                    mid, cheb_order=nfunctions).reshape(
                            [nfunctions, nfunctions])
            nfp_table[mid] = np.tensordot(mccoefs, self.cheb_table,
                    axes=([0, 1], [0, 1]))

        # transform to self.data format
        self.data = nfp_table.transpose((3, 0, 2, 1)).reshape(-1, order='C')

        t2 = time()

        """
        self.data = np.zeros(nfp_table.reshape(-1).shape)
        for source_mode_id in range(self.n_q_points):
            for t0 in range(self.quad_order):
                for t1 in range(self.quad_order):
                    for case_id in range(len(self.interaction_case_vecs)):
                        target_mode_id = t0 * self.quad_order + t1
                        pair_id = source_mode_id * self.n_q_points + target_mode_id
                        entry_id = case_id * self.n_pairs + pair_id
                        self.data[entry_id] = nfp_table[source_mode_id,
                                                        t1, t0, case_id]
        """

        # print("Computing normalizers")
        self.build_normalizer_table()

        t3 = time()

        print("Loopy knl:", t1 - t0,
              ", InvCheb:", t2 - t1,
              ", NmlzInt:", t3 - t2)

        self.is_built = True

    # }}} End build table via adding up a Droste of bricks

    # {{{ build table (driver)

    def build_table(self, queue=None, **kwargs):
        method = self.build_method
        if method == "Transform":
            self.build_table_via_transform()
        elif method == "DrosteSum":
            self.build_table_via_droste_bricks(
                    queue=queue, **kwargs)
        else:
            raise NotImplementedError()

    # }}} End build table (driver)

    # {{{ query table and transform to actual box

    def get_potential_scaler(
            self,
            entry_id,
            source_box_size=1,
            kernel_type=None,
            kernel_power=None):
        """Returns a helper function to rescale the table entry based on
           source_box's actual size (edge length).
        """
        assert (source_box_size > 0)
        a = source_box_size

        if kernel_type is None:
            kernel_type = self.kernel_type

        if kernel_type is None:
            raise NotImplementedError(
                "Specify kernel type before performing scaling queries"
            )

        if kernel_type == "log":
            assert (kernel_power is
                    None)
            source_mode_index = self.decode_index(
                entry_id)[
                    "source_mode_index"]
            displacement = a**2 * np.log(
                a
            ) * self.mode_normalizers[
                source_mode_index]
            scaling = a**2

        elif kernel_type == "const":
            displacement = 0
            scaling = 1

        elif kernel_type == "inv_power":
            assert (kernel_power is
                    not None)
            displacement = 0
            scaling =ource_box_size**(
                2 + kernel_power)

        elif kernel_type == "rigid":
            # TODO: add assersion for source box size
            displacement = 0
            scaling = 1

        else:
            raise NotImplementedError(
                "Unsupported kernel type"
            )

        def scaler(pot_val):
            return pot_val * scaling + displacement

        return scaler


# }}} End query table and transform to actual box

# }}} End table data structure

# vim: foldmethod=marker:filetype=pyopencl
