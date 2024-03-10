__copyright__ = "Copyright (C) 2018 Xiaoyu Wei"

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
import loopy as lp
from volumential.tools import KernelCacheWrapper

import logging

logger = logging.getLogger(__name__)

__doc__ = """
.. autoclass:: DrosteBase
   :members:
.. autoclass:: DrosteFull
   :members:
.. autoclass:: DrosteReduced
   :members:
.. autoclass:: InverseDrosteReduced
   :members:
"""

if 0:
    logging.basicConfig(level=logging.INFO)

# {{{ Droste base class


class DrosteBase(KernelCacheWrapper):
    """
    Base class for Droste methods.
    It uses sumpy tools to cache the loopy kernel.

    .. attribute:: integral_knl

       The integral kernel of sumpy.kernel type.

    .. attribute:: interaction_case_vecs

       The relative positions of the target box for each case.

    .. attribute:: interaction_case_scls

       The relative sizes of the target box for each case.

    .. attribute:: special_radial_quadrature

        If True, the radial direction uses a different quadrature rule.
        The radial direction is identified with the condition
        ``ibrick_axis == iaxis`` and always uses iname ``q0``.
    """

    def __init__(self, integral_knl, quad_order, case_vecs, n_brick_quad_points,
                 special_radial_quadrature, nradial_quad_points):
        from sumpy.kernel import Kernel

        if integral_knl is not None:
            assert isinstance(integral_knl, Kernel)
            self.integral_knl = integral_knl
            self.dim = integral_knl.dim
        else:
            self.integral_knl = None
            self.dim = None

        if case_vecs is not None:
            self.ncases = len(case_vecs)
            for cvec in case_vecs:
                assert len(cvec) == self.dim
            self.interaction_case_vecs = np.array(
                [[v[d] for v in case_vecs] for d in range(self.dim)]
            )
        else:
            self.ncases = 0
            case_vecs = []
            self.interaction_case_vecs = np.array([])

        self.interaction_case_scls = np.array(
            [
                1
                if int(max(abs(np.array(vec)))) == 0
                else max([abs(cvc) - 0.5 for cvc in np.array(vec) / 4]) * 2
                for vec in case_vecs
            ]
        )

        self.nfunctions = quad_order
        self.ntgt_points = quad_order
        self.nquad_points = n_brick_quad_points

        if special_radial_quadrature and (self.dim == 1):
            raise ValueError(
                "please only use normal quadrature nodes for 1D quadrature")

        self.special_radial_quadrature = special_radial_quadrature
        self.nradial_quad_points = nradial_quad_points
        self.brick_quadrature_kwargs = self.make_brick_quadrature_kwargs()

        if self.dim is not None:
            self.n_q_points = quad_order ** self.dim
        else:
            self.n_q_points = None

        self.name = "DrosteBase"

    def make_basis_vars(self):
        d = self.dim
        self.basis_vars = ["f%d" % i for i in range(d)]
        return self.basis_vars

    def make_tgt_vars(self):
        d = self.dim
        self.tgt_vars = ["t%d" % i for i in range(d)]
        return self.tgt_vars

    def make_quad_vars(self):
        d = self.dim
        self.quad_vars = ["q%d" % i for i in range(d)]
        return self.quad_vars

    def make_basis_eval_vars(self):
        d = self.dim
        self.basis_eval_vars = ["p%d" % i for i in range(d)]
        return self.basis_eval_vars

    def make_pwaffs(self):
        import islpy as isl

        self.pwaffs = isl.make_zero_and_vars(
            self.tgt_vars
            + self.basis_vars
            + self.basis_eval_vars
            + self.quad_vars
            + ["iside", "iaxis", "ibrick_side", "ibrick_axis", "ilevel", "icase"],
            ["nlevels"],
        )
        return self.pwaffs

    def make_brick_domain(self, variables, n, lbound=0, ubound=0):
        pwaffs = self.pwaffs
        if isinstance(n, str):
            ubound_pwaff = pwaffs[n] + ubound
        else:
            ubound_pwaff = pwaffs[0] + n

        import operator
        from functools import reduce

        return reduce(
            operator.and_,
            [(pwaffs[0] + lbound).le_set(pwaffs[var]) for var in variables]
            + [pwaffs[var].lt_set(ubound_pwaff) for var in variables],
        )

    def make_brick_quadrature_kwargs(self):
        """Produce 1D quadrature formulae used for each brick.
        The rules returned are defined over [0, 1].
        """
        # sps.legendre blows up easily at high order
        import scipy.special as sps

        legendre_nodes, legendre_weights = sps.p_roots(self.nquad_points)
        legendre_nodes = legendre_nodes * 0.5 + 0.5
        legendre_weights = legendre_weights * 0.5

        if self.special_radial_quadrature:
            # Since there is no exact degree -> nqpts map for tanh-sinh rule,
            # we find the largest rule whose size dose not exceed the parameter
            # nradial_quad_points
            import mpmath
            mp_ctx = mpmath.mp
            mp_ctx.dps = 20  # decimal precision
            mp_ctx.pretty = True
            ts = mpmath.calculus.quadrature.TanhSinh(mp_ctx)
            prec = int(np.log(10) / np.log(2) * mp_ctx.dps)  # bits of precision

            for deg in range(1, 100):
                nodes = ts.calc_nodes(degree=deg, prec=prec)
                if len(nodes) >= self.nradial_quad_points:
                    self.nradial_quad_points = len(nodes)
                    break

            # extract quadrature formula over [-1, 1], note the 0.5**level scaling
            ts_nodes = np.array([p[0] for p in nodes], dtype=np.float64)
            ts_weights = np.array(
                [p[1] * 2 * mp_ctx.power(0.5, deg) for p in nodes],
                dtype=np.float64)
            if deg == 1:
                ts_weights *= 0.5  # peculiar scaling when deg=1

            # map to [0, 1]
            ts_nodes = ts_nodes * 0.5 + 0.5
            ts_weights *= 0.5

            return {
                "quadrature_nodes": legendre_nodes,
                "quadrature_weights": legendre_weights,
                "radial_quadrature_nodes": ts_nodes,
                "radial_quadrature_weights": ts_weights,
                }

        else:
            return {
                "quadrature_nodes": legendre_nodes,
                "quadrature_weights": legendre_weights,
                }

    def get_sumpy_kernel_insns(self):
        # get sumpy kernel insns
        from sumpy.symbolic import make_sym_vector

        dvec = make_sym_vector("dist", self.dim)
        from sumpy.assignment_collection import SymbolicAssignmentCollection

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
        from sumpy.codegen import to_loopy_insns

        loopy_insns = to_loopy_insns(
            sac.assignments.items(),
            vector_names={"dist"},
            pymbolic_expr_maps=[self.integral_knl.get_code_transformer()],
            retain_names=[result_name],
            complex_dtype=np.complex128,
        )
        return loopy_insns

    def get_sumpy_kernel_eval_insns(self):
        # actual integral kernel evaluation (within proper inames)
        knl_loopy_insns = self.get_sumpy_kernel_insns()
        quad_inames = frozenset(
            ["icase"]
            + self.tgt_vars
            + self.basis_vars
            + ["ilevel", "ibrick_axis", "ibrick_side"]
            + self.quad_vars
        )
        quad_kernel_insns = [
            insn.copy(within_inames=insn.within_inames | quad_inames)
            for insn in knl_loopy_insns
        ]

        from sumpy.symbolic import SympyToPymbolicMapper

        sympy_conv = SympyToPymbolicMapper()
        scaling_assignment = lp.Assignment(
            id=None,
            assignee="knl_scaling",
            expression=sympy_conv(self.integral_knl.get_global_scaling_const()),
            temp_var_type=lp.Optional(),
        )

        return quad_kernel_insns + [scaling_assignment]

    def codegen_basis_eval(self, iaxis):
        """Generate instructions to evaluate Chebyshev polynomial basis.
        (Chebyshev polynomials of the first kind T_n).
        """
        code = """
            <> T0_IAXIS = 1
            <> T1_IAXIS = template_mapped_point_tmp[IAXIS] {dep=mpoint}
            <> Tprev_IAXIS = T0_IAXIS {id=t0_IAXIS}
            <> Tcur_IAXIS = T1_IAXIS {id=t1_IAXIS,dep=t0_IAXIS}

            for pIAXIS
                <> Tnext_IAXIS = (2 * template_mapped_point_tmp[IAXIS] * Tcur_IAXIS
                                - Tprev_IAXIS) {id=tnextIAXIS,dep=t1_IAXIS}
                Tprev_IAXIS = Tcur_IAXIS {id=tprev_updateIAXIS,dep=tnextIAXIS}
                Tcur_IAXIS = Tnext_IAXIS {id=tcur_updateIAXIS,dep=tprev_updateIAXIS}
            end

            <> basis_evalIAXIS = (
                T0_IAXIS * if(fIAXIS == 0, 1, 0)
                + T1_IAXIS * if(fIAXIS == 1, 1, 0)
                + simul_reduce(sum, pIAXIS,
                    if(fIAXIS >= 2 and fIAXIS == pIAXIS + 2, Tcur_IAXIS, 0))
                ) {id=basisIAXIS,dep=tcur_updateIAXIS}
            """.replace(
            "IAXIS", str(iaxis)
        )
        return code

    def make_dim_independent(self, knlstring):

        # replace REV_* first
        resknl = knlstring.replace("REV_BASIS_VARS", ",".join(self.basis_vars[::-1]))
        resknl = resknl.replace("REV_TGT_VARS", ",".join(self.tgt_vars[::-1]))

        resknl = resknl.replace("BASIS_VARS", ",".join(self.basis_vars))
        resknl = resknl.replace("TGT_VARS", ",".join(self.tgt_vars))
        resknl = resknl.replace("QUAD_VARS", ",".join(self.quad_vars))

        resknl = resknl.replace(
                "POSTPROCESS_KNL_VAL",
                "<> knl_val_post = knl_val {id=pp_kval}"
                )

        if self.dim == 1:

            resknl = resknl.replace("TPLTGT_ASSIGNMENT", """target_nodes[t0]""")
            resknl = resknl.replace("QUAD_PT_ASSIGNMENT", """quadrature_nodes[q0]""")
            resknl = resknl.replace("PROD_QUAD_WEIGHT", """quadrature_weights[q0]""")
            resknl = resknl.replace("DENSITY_VAL_ASSIGNMENT", """basis_eval0""")
            resknl = resknl.replace(
                "PREPARE_BASIS_VALS",
                "\n".join([self.codegen_basis_eval(i) for i in range(self.dim)])
                + """
                ... nop {id=basis_evals,dep=basis0}
                """,
            )

        elif self.dim == 2:
            resknl = resknl.replace(
                "TPLTGT_ASSIGNMENT",
                """if(iaxis == 0, target_nodes[t0], target_nodes[t1])""",
            )
            if self.special_radial_quadrature:
                resknl = resknl.replace(
                    "QUAD_PT_ASSIGNMENT",
                    """if(iaxis == ibrick_axis,
                          radial_quadrature_nodes[q0], quadrature_nodes[q1])""",
                )
                resknl = resknl.replace(
                    "PROD_QUAD_WEIGHT",
                    """radial_quadrature_weights[q0] * quadrature_weights[q1]""")
            else:
                resknl = resknl.replace(
                    "QUAD_PT_ASSIGNMENT",
                    """if(iaxis == 0, quadrature_nodes[q0], quadrature_nodes[q1])""",
                )
                resknl = resknl.replace(
                    "PROD_QUAD_WEIGHT",
                    """quadrature_weights[q0] * quadrature_weights[q1]""")
            resknl = resknl.replace(
                "DENSITY_VAL_ASSIGNMENT", """basis_eval0 * basis_eval1"""
            )
            resknl = resknl.replace(
                "PREPARE_BASIS_VALS",
                "\n".join([self.codegen_basis_eval(i) for i in range(self.dim)])
                + """
                ... nop {id=basis_evals,dep=basis0:basis1}
                """,
            )

        elif self.dim == 3:
            resknl = resknl.replace(
                "TPLTGT_ASSIGNMENT",
                """if(iaxis == 0, target_nodes[t0], if(
                          iaxis == 1, target_nodes[t1], target_nodes[t2]))""",
            )
            if self.special_radial_quadrature:
                # depending on ibrick_axis, axes could be
                # [q0, q1, q2], [q1, q0, q2], or [q1, q2, q0]
                resknl = resknl.replace(
                    "QUAD_PT_ASSIGNMENT",
                    """if(iaxis == ibrick_axis, radial_quadrature_nodes[q0], if(
                      (iaxis == 0) or ((iaxis == 1) and (ibrick_axis == 0)),
                      quadrature_nodes[q1],
                      quadrature_nodes[q2]))""",
                )
                resknl = resknl.replace(
                    "PROD_QUAD_WEIGHT",
                    """radial_w[q0] * w[q1] * w[q2]""".replace(
                        "w", "quadrature_weights"))
            else:
                resknl = resknl.replace(
                    "QUAD_PT_ASSIGNMENT",
                    """if(iaxis == 0, quadrature_nodes[q0], if(
                      iaxis == 1, quadrature_nodes[q1], quadrature_nodes[q2]))""")
                resknl = resknl.replace(
                    "PROD_QUAD_WEIGHT",
                    """w[q0] * w[q1] * w[q2]""".replace("w", "quadrature_weights"))
            resknl = resknl.replace(
                "DENSITY_VAL_ASSIGNMENT",
                """basis_eval0 * basis_eval1 * basis_eval2"""
            )
            resknl = resknl.replace(
                "PREPARE_BASIS_VALS",
                "\n".join([self.codegen_basis_eval(i) for i in range(self.dim)])
                + """
                ... nop {id=basis_evals,dep=basis0:basis1:basis2}
                """,
            )

        else:
            raise NotImplementedError

        return resknl

    def make_result_array(self, **kwargs):
        """Allocate memory space for results.
        """
        # by default uses double type returns
        if "result_dtype" in kwargs:
            result_dtype = kwargs["result_dtype"]
        else:
            if self.integral_knl.is_complex_valued:
                result_dtype = np.complex128
            else:
                result_dtype = np.float64

        # allocate return arrays
        if self.dim == 1:
            result_array = (
                    np.zeros(
                        (self.nfunctions, self.ntgt_points, self.ncases),
                        result_dtype
                        )
                    + np.nan)
        elif self.dim == 2:
            result_array = (
                    np.zeros(
                        (
                            self.nfunctions,
                            self.nfunctions,
                            self.ntgt_points,
                            self.ntgt_points,
                            self.ncases,
                            ),
                        result_dtype
                        )
                    + np.nan
                    )
        elif self.dim == 3:
            result_array = (
                    np.zeros(
                        (
                            self.nfunctions,
                            self.nfunctions,
                            self.nfunctions,
                            self.ntgt_points,
                            self.ntgt_points,
                            self.ntgt_points,
                            self.ncases,
                            ),
                        result_dtype
                        )
                    + np.nan
                    )
        else:
            raise NotImplementedError
        return result_array

    def get_kernel_code(self):
        return [
            self.make_dim_independent(
                """  # noqa
        for iaxis
            <> root_center[iaxis] = 0.5 * (
                    root_brick[iaxis, 1] + root_brick[iaxis, 0]) {dup=iaxis}
            <> root_extent[iaxis] = (root_brick[iaxis, 1]
                    - root_brick[iaxis, 0]) {dup=iaxis}
        end

        for ilevel, BASIS_VARS, TGT_VARS, icase

            # Targets outside projected onto the boundary
            for iaxis
                <> template_target[iaxis] = TPLTGT_ASSIGNMENT \
                        {id=tplt_tgt_pre,dup=iaxis}
            end

            # True targets are used for kernel evaluation
            for iaxis
                <> true_target[iaxis] = ( root_center[iaxis]
                         + interaction_case_vecs[iaxis, icase]
                            * 0.25 * root_extent[iaxis]
                         + interaction_case_scls[icase]
                            * (template_target[iaxis] - 0.5)
                            * root_extent[iaxis]
                         ) {id=true_targets,dup=iaxis,dep=tplt_tgt_pre}

                # Re-map the mapped points to [-1,1]^dim for Chebyshev evals
                <> template_true_target[iaxis] = 0.0 + (
                        true_target[iaxis] - root_center[iaxis]
                        ) / root_extent[iaxis] * 2 {id=template_true_targets,dup=iaxis,dep=true_targets}
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

            # Debug output
            #for iaxis
            #    projected_template_targets[TGT_VARS, icase, iaxis] = \
            #            template_target[iaxis] {dup=iaxis,dep=tplt_tgt}
            #end

            # Debug output
            #for iaxis
            #    target_points[ilevel, BASIS_VARS, TGT_VARS, icase, iaxis
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

            for ibrick_axis, ibrick_side, QUAD_VARS
                for iaxis
                    <> point[iaxis] = QUAD_PT_ASSIGNMENT {dup=iaxis}
                end

                <> deform = point[ibrick_axis]*(1-alpha)

                for iaxis
                    if iaxis == ibrick_axis
                        <> mapped_point_tmp[iaxis] = (
                            point[ibrick_axis]*(
                                inner_brick[ibrick_axis, ibrick_side]
                                - outer_brick[ibrick_axis, ibrick_side])
                                + outer_brick[ibrick_axis, ibrick_side]) \
                            {id=mpoint1,nosync=mpoint2}
                    else
                        <> pre_scale = (
                            point[iaxis]
                            + deform*(template_target[iaxis]-point[iaxis])
                            ) {dep=tplt_tgt}

                        mapped_point_tmp[iaxis] = \
                            ob_ext[iaxis] * pre_scale + outer_brick[iaxis, 0] \
                            {id=mpoint2,nosync=mpoint1}
                    end

                    ... nop {id=mpoint,dep=mpoint1:mpoint2}

                    # Re-map the mapped points to [-1,1]^dim for Chebyshev evals
                    <> template_mapped_point_tmp[iaxis] = 0.0 + (
                            mapped_point_tmp[iaxis] - root_center[iaxis]
                            ) / root_extent[iaxis] * 2 {dep=mpoint}

                    # Debug output
                    #mapped_points[ilevel, ibrick_axis,
                    #          ibrick_side, TGT_VARS, icase, iaxis, QUAD_VARS] = \
                    #    mapped_point_tmp[iaxis]  {dep=mpoint}
                    #mapped_target[TGT_VARS, icase, iaxis] = target[iaxis]
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

                # optional postprocessing of kernel values
                POSTPROCESS_KNL_VAL

                # in our case 0 * inf = 0
                <> knl_val_finished = if(abs(knl_val_post) > 1e16,
                    0, knl_val_post) {id=finish_kval,dep=pp_kval}

            end

        end

        for BASIS_VARS, TGT_VARS, icase
            for ilevel
                for ibrick_axis, ibrick_side, QUAD_VARS

                    PREPARE_BASIS_VALS

                    <> density_val = DENSITY_VAL_ASSIGNMENT \
                            {id=density,dep=basis_evals}

                end
            end
        end

        for BASIS_VARS, TGT_VARS, icase
            # the first index is contiguous
            result[REV_BASIS_VARS, REV_TGT_VARS, icase] = (sum(
                    (ilevel, ibrick_axis, ibrick_side, QUAD_VARS),
                    PROD_QUAD_WEIGHT
                    * jacobian
                    * knl_val_finished
                    * density_val
                    )
                    *  knl_scaling
                ) {id=result,dep=jac:mpoint:density:finish_kval}
        end
        """)
        ]

    def get_target_points(self, queue=None):
        import volumential.meshgen as mg

        q_points, _, _ = mg.make_uniform_cubic_grid(
            degree=self.ntgt_points, level=1, dim=self.dim,
            queue=queue)

        # map to [0,1]^d
        mapped_q_points = np.array(
                [
                    0.5 * (qp + np.ones(self.dim))
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
        return mapped_q_points[q_points_ordering]

    def postprocess_cheb_table(self, cheb_table, cheb_coefs):
        nfp_table = np.zeros(
            [self.n_q_points, ] + list(cheb_table.shape[self.dim:]),
            dtype=cheb_table.dtype)

        # transform to interpolatory basis functions
        # NOTE: the reversed order of indices, e.g.,
        #       mccoefs[f0, f1, f2], and cheb_table[f2, f1, f0]
        concat_axes = list(range(self.dim))
        for mid in range(self.n_q_points):
            mccoefs = cheb_coefs[mid]
            nfp_table[mid] = np.tensordot(
                mccoefs.reshape([self.nfunctions for iaxis in range(self.dim)]),
                cheb_table,
                axes=(concat_axes, concat_axes[::-1]),
            )

        # transform to self.data format (icase, source_id, target_id)
        # NOTE: the directions of target_id are reversed since cheb_table
        #       is indexed by, e.g. cheb_table[t2, t1, t0]
        transpose_axes = (self.dim + 1, 0) + tuple(
            self.dim - i for i in range(self.dim)
        )

        return nfp_table.transpose(transpose_axes).reshape(-1, order="C")


# }}} End Droste base class

# {{{ full Droste method


class DrosteFull(DrosteBase):
    """
    Build the full table directly.
    """

    def __init__(self, integral_knl, quad_order, case_vecs,
                 n_brick_quad_points=50,
                 special_radial_quadrature=False,
                 nradial_quad_points=None):
        super().__init__(
            integral_knl, quad_order, case_vecs, n_brick_quad_points,
            special_radial_quadrature, nradial_quad_points)
        self.name = "DrosteFull"

    def make_loop_domain(self):
        tgt_vars = self.make_tgt_vars()
        quad_vars = self.make_quad_vars()
        basis_vars = self.make_basis_vars()
        basis_eval_vars = self.make_basis_eval_vars()
        pwaffs = self.make_pwaffs()  # noqa: F841
        if self.special_radial_quadrature:
            quad_vars_subdomain = self.make_brick_domain(
                quad_vars[1:], self.nquad_points) & self.make_brick_domain(
                    quad_vars[:1], self.nradial_quad_points)
        else:
            quad_vars_subdomain = self.make_brick_domain(
                quad_vars, self.nquad_points)
        self.loop_domain = (
            self.make_brick_domain(tgt_vars, self.ntgt_points)
            & quad_vars_subdomain
            & self.make_brick_domain(basis_vars, self.nfunctions)
            & self.make_brick_domain(basis_eval_vars, self.nfunctions)
            & self.make_brick_domain(["iside"], 2)
            & self.make_brick_domain(["iaxis"], self.dim)
            & self.make_brick_domain(["ibrick_side"], 2)
            & self.make_brick_domain(["ibrick_axis"], self.dim)
            & self.make_brick_domain(["ilevel"], "nlevels")
            & self.make_brick_domain(["icase"], self.ncases)
        )
        return self.loop_domain

    def get_kernel(self, **kwargs):

        domain = self.make_loop_domain()

        extra_kernel_kwarg_types = ()
        if "extra_kernel_kwarg_types" in kwargs:
            extra_kernel_kwarg_types = kwargs["extra_kernel_kwarg_types"]

        loopy_knl = lp.make_kernel(  # NOQA
            [domain],
            self.get_kernel_code()
            + self.get_sumpy_kernel_eval_insns(),
            [
                lp.ValueArg("alpha", np.float64),
                lp.ValueArg("n_cases, nfunctions, quad_order, dim", np.int32),
                lp.GlobalArg("interaction_case_vecs", np.float64, "dim, n_cases"),
                lp.GlobalArg("interaction_case_scls", np.float64, "n_cases"),
                lp.GlobalArg(
                    "result", None,
                    ", ".join(
                        ["nfunctions" for d in range(self.dim)]
                        + ["quad_order" for d in range(self.dim)]
                    )
                    + ", n_cases",
                ), ]
            + list(extra_kernel_kwarg_types)
            + ["...", ],
            name="brick_map",
            lang_version=(2018, 2),
        )

        loopy_knl = lp.fix_parameters(loopy_knl, d=self.dim)
        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        try:
            loopy_knl = self.integral_knl.prepare_loopy_kernel(loopy_knl)
        except Exception:  # noqa: B902
            pass

        return loopy_knl

    def get_optimized_kernel(self, ncpus=None, **kwargs):
        if ncpus is None:
            import multiprocessing

            # NOTE: this detects the number of logical cores, which
            # may result in suboptimal performance.
            ncpus = multiprocessing.cpu_count()
        knl = self.get_kernel(**kwargs)
        knl = lp.split_iname(knl, "icase", ncpus, inner_tag="g.0")
        knl = lp.add_inames_for_unused_hw_axes(knl)
        return knl

    def call_loopy_kernel(self, queue, **kwargs):
        """
        :param source_box_extent:
        :param alpha:
        :param nlevels:
        :param extra_kernel_kwargs:
        """

        if "source_box_extent" in kwargs:
            assert kwargs["source_box_extent"] > 0
            source_box_extent = kwargs["source_box_extent"]
        else:
            source_box_extent = 1

        extra_kernel_kwargs = {}
        if "extra_kernel_kwargs" in kwargs:
            extra_kernel_kwargs = kwargs["extra_kernel_kwargs"]

        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
            assert alpha >= 0 and alpha < 1
        else:
            alpha = 0

        if "nlevels" in kwargs:
            nlevels = kwargs["nlevels"]
            assert nlevels > 0
            if nlevels > 1 and self.special_radial_quadrature:
                logger.warn("When using tanh-sinh quadrature in the radial "
                            "direction, it is often best to use a single level.")
        else:
            # Single level is equivalent to Duffy transform
            nlevels = 1

        missing_measure = (alpha ** nlevels * source_box_extent) ** self.dim
        if abs(missing_measure) > 1e-6:
            from warnings import warn

            warn(
                "Droste probably has too few levels, missing measure = "
                + str(missing_measure)
            )

        if "result_array" in kwargs:
            result_array = kwargs["result_array"]

        else:
            result_array = self.make_result_array(**kwargs)

        # root brick
        root_brick = np.zeros((self.dim, 2))
        root_brick[:, 1] = source_box_extent

        # target points in 1D
        q_points = self.get_target_points(queue)
        assert len(q_points) == self.ntgt_points ** self.dim
        t = np.array([pt[-1] for pt in q_points[: self.ntgt_points]])

        knl = self.get_cached_optimized_kernel()
        evt, res = knl(
            queue, alpha=alpha, result=result_array,
            root_brick=root_brick,
            target_nodes=t.astype(np.float64, copy=True),
            interaction_case_vecs=self.interaction_case_vecs.astype(
                np.float64, copy=True),
            interaction_case_scls=self.interaction_case_scls.reshape(-1).astype(
                np.float64, copy=True),
            n_cases=self.ncases, nfunctions=self.nfunctions,
            quad_order=self.ntgt_points, nlevels=nlevels,
            **self.brick_quadrature_kwargs,
            **extra_kernel_kwargs)

        cheb_table = res["result"]

        return cheb_table

    def get_cache_key(self):
        return (
            type(self).__name__,
            str(self.dim) + "D",
            self.integral_knl.__str__(),
            "quad_order-" + str(self.ntgt_points),
            "brick_order-" + str(self.nquad_points),
            "brick_radial_order-" + str(self.nradial_quad_points),
        )

    def __call__(self, queue, **kwargs):
        """
        :arg source_box_extent
        :arg alpha
        :arg nlevels
        :arg extra_kernel_kwargs
        """
        if "cheb_coefs" in kwargs:
            assert len(kwargs["cheb_coefs"]) == self.n_q_points
            for ccoef in kwargs["cheb_coefs"]:
                assert len(ccoef) == self.nfunctions ** self.dim
            cheb_coefs = kwargs["cheb_coefs"]
            return self.postprocess_cheb_table(
                self.call_loopy_kernel(queue, **kwargs), cheb_coefs
            )
        else:
            logger.info("Returning cheb table directly")
            return self.call_loopy_kernel(queue, **kwargs)


# }}} End full Droste method

# {{{ reduced Droste method


class DrosteReduced(DrosteBase):
    """
    Reduce the workload by only building part of the table and infer the
    rest of the table by symmetry.
    """

    def __init__(
        self,
        integral_knl=None,
        quad_order=None,
        case_vecs=None,
        n_brick_quad_points=50,
        knl_symmetry_tags=None,
        special_radial_quadrature=False,
        nradial_quad_points=None,
    ):
        super().__init__(
            integral_knl, quad_order, case_vecs, n_brick_quad_points,
            special_radial_quadrature, nradial_quad_points)

        from volumential.list1_symmetry import CaseVecReduction

        # by default only self-interaction is counted
        if case_vecs is None:
            case_vecs = [np.array([0, 0, 0]), ]

        self.reduce_by_symmetry = CaseVecReduction(case_vecs, knl_symmetry_tags)
        logger.info(
            "Reduction ratio by symmetry = "
            + str(self.reduce_by_symmetry.get_full_reduction_ratio())
        )

        self.nbcases = len(self.reduce_by_symmetry.reduced_vecs)
        self.loop_domains = [None for bvec in self.reduce_by_symmetry.reduced_vecs]

        # state of the object that determines behavior of functions like
        # get_kernel()
        # get_kernel() returns the kernel for actual quadrature when
        #   get_kernel_id = 0,
        # while it returns the kernel for expansion within the same case when = 1.
        self.current_base_case = 0
        self.get_kernel_id = 0

    def make_loop_domain(self, base_case_id):
        # The icase is just a dummy variable that allows resusing some of the
        # code (e.g. kernel evals) from the base class
        assert base_case_id >= 0
        assert base_case_id < self.nbcases

        tgt_vars = self.make_tgt_vars()  # noqa: F841
        quad_vars = self.make_quad_vars()
        basis_vars = self.make_basis_vars()
        basis_eval_vars = self.make_basis_eval_vars()
        pwaffs = self.make_pwaffs()

        if self.special_radial_quadrature:
            quad_vars_subdomain = self.make_brick_domain(
                quad_vars[1:], self.nquad_points) & self.make_brick_domain(
                    quad_vars[:1], self.nradial_quad_points)
        else:
            quad_vars_subdomain = self.make_brick_domain(
                quad_vars, self.nquad_points)

        loop_domain_common_parts = (
            quad_vars_subdomain
            & self.make_brick_domain(basis_vars, self.nfunctions)
            & self.make_brick_domain(basis_eval_vars, self.nfunctions)
            & self.make_brick_domain(["iside"], 2)
            & self.make_brick_domain(["iaxis"], self.dim)
            & self.make_brick_domain(["ibrick_side"], 2)
            & self.make_brick_domain(["ibrick_axis"], self.dim)
            & self.make_brick_domain(["ilevel"], "nlevels")
            & self.make_brick_domain(["icase"], 1)
        )

        # target brick domain depends on symmetry group of the case vec
        # upper bounds account for flippings, and lower bounds account for swappings
        flippable, swappable_groups = self.reduce_by_symmetry.parse_symmetry_tags(
            self.reduce_by_symmetry.reduced_invariant_groups[base_case_id]
        )
        prev_swappable = -np.ones(self.dim, dtype=np.int32)
        for iaxis in range(self.dim):
            for group in swappable_groups:
                if iaxis in group:
                    axid = list(group).index(iaxis)
                    if axid == 0:
                        pass
                    else:
                        prev_swappable[iaxis] = list(group)[axid - 1]

        from functools import reduce
        import operator

        tgt_domain_ubounds = reduce(
            operator.and_,
            [
                pwaffs["t" + str(iaxis)].lt_set(
                    pwaffs[0]
                    + (
                        (self.ntgt_points + 1) // 2
                        if flippable[iaxis]
                        else self.ntgt_points
                    )
                )
                for iaxis in range(self.dim)
            ],
        )
        tgt_domain_lbounds = reduce(
            operator.and_,
            [
                (
                    pwaffs[0]
                    if prev_swappable[iaxis] == -1
                    else pwaffs["t" + str(prev_swappable[iaxis])]
                ).le_set(pwaffs["t" + str(iaxis)])
                for iaxis in range(self.dim)
            ],
        )
        tgt_domain = tgt_domain_ubounds & tgt_domain_lbounds

        self.loop_domains[base_case_id] = loop_domain_common_parts & tgt_domain
        return self.loop_domains[base_case_id]

    def make_result_array(self, **kwargs):
        """Allocate memory space for results.
        """
        # by default uses double type returns
        if "result_dtype" in kwargs:
            result_dtype = kwargs["result_dtype"]
        else:
            if self.integral_knl.is_complex_valued:
                result_dtype = np.complex128
            else:
                result_dtype = np.float64

        # allocate return arrays
        if self.dim == 1:
            result_array = (
                    np.zeros(
                        (self.nfunctions, self.ntgt_points, 1),
                        result_dtype
                        )
                    + np.nan)
        elif self.dim == 2:
            result_array = (
                    np.zeros(
                        (
                            self.nfunctions,
                            self.nfunctions,
                            self.ntgt_points,
                            self.ntgt_points,
                            1,
                            ),
                        result_dtype
                        )
                    + np.nan
                    )
        elif self.dim == 3:
            result_array = (
                    np.zeros(
                        (
                            self.nfunctions,
                            self.nfunctions,
                            self.nfunctions,
                            self.ntgt_points,
                            self.ntgt_points,
                            self.ntgt_points,
                            1,
                            ),
                        result_dtype
                        )
                    + np.nan
                    )
        else:
            raise NotImplementedError
        return result_array

    def get_kernel_expansion_by_symmetry_code(self):
        """
        Extra assignments that performs expansion by symmetry within the
        current case.
        """
        # case_id = self.reduce_by_symmetry.reduced_vec_ids[self.current_base_case]
        # case_vec = self.reduce_by_symmetry.reduced_vecs[self.current_base_case]
        invariant_group = self.reduce_by_symmetry.reduced_invariant_groups[
            self.current_base_case
        ]

        flippable, swappable_groups = self.reduce_by_symmetry.parse_symmetry_tags(
            invariant_group
        )
        nflippables = int(sum(flippable))
        assert len(flippable) == self.dim
        flippable_ids = [i for i in range(self.dim) if flippable[i]]
        base_tgt_ordering = self.make_dim_independent("TGT_VARS").split(",")
        base_fun_ordering = self.make_dim_independent("BASIS_VARS").split(",")
        assert len(base_tgt_ordering) == self.dim

        def flip_tgt(tgt_var):
            assert isinstance(tgt_var, str)
            return str(self.ntgt_points) + " - " + tgt_var + " - 1"

        def conjunct(var):
            # conjunct(v_i) = v_{dim-1-i}
            assert isinstance(var, str)
            vid = int(var[1:])
            assert vid >= 0 and vid < self.dim
            return var[0] + str(self.dim - 1 - vid)

        from itertools import permutations, product

        ext_ids = product(
            product([0, 1], repeat=nflippables),
            *[permutations(sgroup) for sgroup in swappable_groups]
        )

        # The idea is that, any member of the hyperoctahedral group
        # has the decomposition m = f * s, where f is a flip and s
        # is a swap.
        # Conversely, by iterating all combinations of flips and swaps,
        # the whole group is iterated over.

        expansion_code = []
        for ext_index, ext_id in zip(range(len(ext_ids)), ext_ids):
            ext_tgt_ordering = list(base_tgt_ordering)
            ext_fun_ordering = list(base_fun_ordering)
            # apply s
            for swap in ext_id[1:]:
                ns = len(swap)
                for i in range(ns):
                    original_part = sorted(swap)
                    ext_tgt_ordering[original_part[i]] = base_tgt_ordering[swap[i]]
                    ext_fun_ordering[original_part[i]] = base_fun_ordering[swap[i]]
            # apply f, also figure out the rule for sign changes
            ext_sign = "1"
            for fid in range(nflippables):
                if ext_id[0][fid]:
                    iaxis = flippable_ids[fid]
                    itgt = ext_tgt_ordering.index("t" + str(iaxis))
                    ext_tgt_ordering[itgt] = flip_tgt(ext_tgt_ordering[itgt])
                    ext_sign = (
                        ext_sign
                        + " * if("
                        + self.basis_vars[iaxis]
                        + " % 2 == 0, 1, -1)"
                    )

            ext_tgt_ordering.reverse()
            ext_fun_ordering.reverse()

            ext_instruction_ids = ":".join([
                "result_ext_" + str(eid)
                for eid in range(len(ext_ids))
                if eid != ext_index
                ])

            expansion_code += [
                self.make_dim_independent(
                    """
                for BASIS_VARS, TGT_VARS, icase
                    result[EXT_BASIS_VARS, EXT_TGT_VARS, icase] = (
                        EXT_SIGN *
                        result[REV_BASIS_VARS, REV_TGT_VARS, icase]
                        ) {id=result_ext_EXT_ID,nosync=EXT_INSN_IDS}
                end
                """.replace(
                        "EXT_TGT_VARS", ",".join(ext_tgt_ordering)
                    )
                    .replace("EXT_BASIS_VARS", ",".join(ext_fun_ordering))
                    .replace("EXT_SIGN", ext_sign)
                    .replace("EXT_ID", str(ext_index))
                    .replace("EXT_INSN_IDS", ext_instruction_ids)
                )
            ]

        return expansion_code

    def get_kernel(self, **kwargs):
        """Reduced Droste is a 2-staged algorithm. The first stage uses the
        kernel from DrosteBase to build part of the table. In the second
        stage, an expansion kernel is called to fill the empty entries.
        """

        domain = self.make_loop_domain(base_case_id=self.current_base_case)

        extra_kernel_kwarg_types = ()
        if "extra_kernel_kwarg_types" in kwargs:
            extra_kernel_kwarg_types = kwargs["extra_kernel_kwarg_types"]

        if self.get_kernel_id == 0:
            loopy_knl = lp.make_kernel(  # NOQA
                [domain],
                self.get_kernel_code()
                # FIXME: cannot have expansion in the same kernel, since it
                # will require a global barrier
                # + self.get_kernel_expansion_by_symmetry_code()
                + self.get_sumpy_kernel_eval_insns(),
                [
                    lp.ValueArg("alpha", np.float64),
                    lp.ValueArg("n_cases, nfunctions, quad_order, dim", np.int32),
                    lp.GlobalArg("interaction_case_vecs",
                        np.float64, "dim, n_cases"),
                    lp.GlobalArg("interaction_case_scls", np.float64, "n_cases"),
                    lp.GlobalArg("target_nodes", np.float64, "quad_order"),
                    lp.GlobalArg(
                        "result", None,
                        ", ".join(
                            ["nfunctions" for d in range(self.dim)]
                            + ["quad_order" for d in range(self.dim)]
                        )
                        + ", n_cases",
                    ), ] + list(extra_kernel_kwarg_types)
                + ["...", ],
                name="brick_map",
                lang_version=(2018, 2),
            )

        elif self.get_kernel_id == 1:
            loopy_knl = lp.make_kernel(  # NOQA
                [domain],
                self.get_kernel_expansion_by_symmetry_code(),
                [
                    lp.ValueArg("n_cases, nfunctions, quad_order, dim", np.int32),
                    lp.GlobalArg(
                        "result", None,
                        ", ".join(
                            ["nfunctions" for d in range(self.dim)]
                            + ["quad_order" for d in range(self.dim)]
                        )
                        + ", n_cases",
                    ), ] + list(extra_kernel_kwarg_types)
                + ["...", ],
                name="brick_map_expansion",
                lang_version=(2018, 2),
            )

        else:
            raise NotImplementedError

        loopy_knl = lp.fix_parameters(loopy_knl, dim=self.dim)
        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        try:
            loopy_knl = self.integral_knl.prepare_loopy_kernel(loopy_knl)
        except Exception:  # noqa: B902
            pass

        return loopy_knl

    def get_optimized_kernel(self, ncpus=None, **kwargs):
        # The returned kernel depends on the state variable
        # self.current_base_case, self.get_kernel_id
        if ncpus is None:
            import multiprocessing
            # NOTE: this detects the number of logical cores, which
            # may result in suboptimal performance.
            ncpus = multiprocessing.cpu_count()

        knl = self.get_kernel(**kwargs)
        knl = lp.join_inames(knl, inames=self.basis_vars, new_iname="func")
        knl = lp.split_iname(knl, "func", ncpus, inner_tag="g.0")
        knl = lp.add_inames_for_unused_hw_axes(knl)
        return knl

    def call_loopy_kernel_case(self, queue, base_case_id, **kwargs):
        """
        Call the table builder on one base case, as given in :self.current_base_case:
        :arg source_box_extent
        :arg alpha
        :arg nlevels
        :arg extra_kernel_kwargs
        """

        if base_case_id != self.current_base_case:
            self.current_base_case = base_case_id

        if "source_box_extent" in kwargs:
            assert kwargs["source_box_extent"] > 0
            source_box_extent = kwargs["source_box_extent"]
        else:
            source_box_extent = 1

        extra_kernel_kwargs = {}
        if "extra_kernel_kwargs" in kwargs:
            extra_kernel_kwargs = kwargs["extra_kernel_kwargs"]

        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
            assert alpha >= 0 and alpha < 1
        else:
            alpha = 0

        if "nlevels" in kwargs:
            nlevels = kwargs["nlevels"]
            assert nlevels > 0
        else:
            # Single level is equivalent to Duffy transform
            nlevels = 1

        missing_measure = (alpha ** nlevels * source_box_extent) ** self.dim
        if abs(missing_measure) > 1e-6:

            logger.warn(
                "Droste probably has too few levels, missing measure = %e"
                % missing_measure
            )

        if "result_array" in kwargs:
            result_array = kwargs["result_array"]
        else:
            result_array = self.make_result_array(**kwargs)

        # root brick
        root_brick = np.zeros((self.dim, 2))
        root_brick[:, 1] = source_box_extent

        # target points in 1D
        q_points = self.get_target_points(queue)
        assert len(q_points) == self.ntgt_points ** self.dim
        t = np.array([pt[-1] for pt in q_points[: self.ntgt_points]])

        base_case_vec = np.array(
            [
                [self.reduce_by_symmetry.reduced_vecs[self.current_base_case][d]]
                for d in range(self.dim)
            ]
        )
        base_case_scl = np.array(
            [
                self.interaction_case_scls[
                    self.reduce_by_symmetry.reduced_vec_ids[self.current_base_case]
                ]
            ]
        )

        self.get_kernel_id = 0
        try:
            delattr(self, "_memoize_dic_get_cached_optimized_kernel")
        except Exception:  # noqa: B902
            pass
        knl = self.get_cached_optimized_kernel()
        evt, res = knl(
            queue, alpha=alpha, result=result_array,
            root_brick=root_brick,
            target_nodes=t.astype(np.float64, copy=True),
            interaction_case_vecs=base_case_vec.astype(np.float64, copy=True),
            interaction_case_scls=base_case_scl.astype(np.float64, copy=True),
            n_cases=1, nfunctions=self.nfunctions,
            quad_order=self.ntgt_points, nlevels=nlevels,
            **extra_kernel_kwargs, **self.brick_quadrature_kwargs)

        raw_cheb_table_case = res["result"]

        self.get_kernel_id = 1
        try:
            delattr(self, "_memoize_dic_get_cached_optimized_kernel")
        except Exception:  # noqa: B902
            pass
        knl2 = self.get_cached_optimized_kernel()
        evt, res2 = knl2(
            queue,
            result=raw_cheb_table_case,
            n_cases=1,
            nfunctions=self.nfunctions,
            quad_order=self.ntgt_points,
            nlevels=nlevels,
        )
        cheb_table_case = res2["result"]

        return cheb_table_case

    def build_cheb_table(self, queue, **kwargs):
        # Build the table using Cheb modes as sources

        if self.integral_knl.is_complex_valued:
            dtype = np.complex128
        else:
            dtype = np.float64

        if self.dim == 1:
            cheb_table = (
                np.zeros((self.nfunctions, self.ntgt_points, self.ncases),
                         dtype) + np.nan)
        elif self.dim == 2:
            cheb_table = (
                np.zeros((self.nfunctions, self.nfunctions,
                          self.ntgt_points, self.ntgt_points,
                          self.ncases,), dtype) + np.nan)
        elif self.dim == 3:
            cheb_table = (
                np.zeros((self.nfunctions, self.nfunctions, self.nfunctions,
                          self.ntgt_points, self.ntgt_points, self.ntgt_points,
                          self.ncases,), dtype) + np.nan)
        else:
            raise NotImplementedError

        for base_case_id in range(self.nbcases):
            self.current_base_case = base_case_id
            case_id = self.reduce_by_symmetry.reduced_vec_ids[base_case_id]
            cheb_table[..., case_id] = self.call_loopy_kernel_case(
                queue, base_case_id, **kwargs
            )[..., 0]

            # {{{ expansion by symmetry
            flippable, swappable_groups = \
                    self.reduce_by_symmetry.parse_symmetry_tags(
                            self.reduce_by_symmetry.symmetry_tags
                            )
            nflippables = int(sum(flippable))
            assert len(flippable) == self.dim
            flippable_ids = [i for i in range(self.dim) if flippable[i]]

            from itertools import permutations, product

            ext_ids = product(
                product([0, 1], repeat=nflippables),
                *[permutations(sgroup) for sgroup in swappable_groups]
            )

            base_case_vec = self.reduce_by_symmetry.full_vecs[case_id]
            assert (base_case_vec
                    == self.reduce_by_symmetry.reduced_vecs[base_case_id])
            for _, ext_id in zip(range(len(ext_ids)), ext_ids):
                # start from the base vec
                ext_vec = list(base_case_vec)
                swapped_axes = list(range(self.dim))

                # apply s
                # Note that fi and ti are in reverse order
                for swap in ext_id[1:]:
                    ns = len(swap)
                    for i in range(ns):
                        original_part = sorted(swap)
                        ext_vec[original_part[i]] = base_case_vec[swap[i]]
                        swapped_axes[self.dim - 1 - original_part[i]] = (
                            self.dim - 1 - swap[i]
                        )
                swapped_basis_axes = swapped_axes
                swapped_tgt_axes = [ax + self.dim for ax in swapped_basis_axes]
                tmp_table_part = (
                    cheb_table[..., case_id]
                    .transpose(swapped_basis_axes + swapped_tgt_axes)
                    .copy()
                )

                # apply f, also figure out the rule for sign changes
                for fid in range(nflippables):
                    if ext_id[0][fid]:
                        iaxis = flippable_ids[fid]

                        if 0:  # ext_vec[swapped_iaxis] == 0:
                            continue
                        else:
                            ext_vec[iaxis] = -ext_vec[iaxis]
                            # flips.append(
                            #   lambda x:
                            #     np.flip(x, (self.dim - 1 - iaxis)
                            #     + self.dim))
                            full_slice = [
                                slice(None, None) for axid in range(2 * self.dim)
                            ]
                            for basis_id in range(self.nfunctions):
                                if basis_id % 2 == 0:
                                    continue
                                sign_slice = full_slice
                                sign_slice[self.dim - 1 - iaxis] = basis_id
                                np.multiply.at(tmp_table_part, tuple(sign_slice), -1)

                        tmp_table_part = np.flip(
                            tmp_table_part, (self.dim - 1 - iaxis) + self.dim
                        )

                assert tuple(ext_vec) in self.reduce_by_symmetry.full_vecs
                ext_case_id = self.reduce_by_symmetry.full_vecs.index(tuple(ext_vec))
                cheb_table[..., ext_case_id] = tmp_table_part

                # }}} End expansion by symmetry

        return cheb_table

    def get_cache_key(self):
        if self.reduce_by_symmetry.symmetry_tags is None:
            # Bn: the full n-dimensional hyperoctahedral group
            symmetry_info = "B%d" % self.dim
        else:
            symmetry_info = "Span{%s}" % ",".join([
                repr(tag) for tag in
                sorted(self.reduce_by_symmetry.symmetry_tags)
                ])

        return (
            type(self).__name__,
            str(self.dim) + "D",
            self.integral_knl.__str__(),
            "quad_order-" + str(self.ntgt_points),
            "brick_order-" + str(self.nquad_points),
            "brick_radial_order-" + str(self.nradial_quad_points),
            "case-" + str(self.current_base_case),
            "kernel_id-" + str(self.get_kernel_id),
            "symmetry-" + symmetry_info,
        )

    def __call__(self, queue, **kwargs):
        """
        :arg source_box_extent
        :arg alpha
        :arg nlevels
        :arg extra_kernel_kwargs
        """
        if "cheb_coefs" in kwargs:
            assert len(kwargs["cheb_coefs"]) == self.n_q_points
            for ccoef in kwargs["cheb_coefs"]:
                assert len(ccoef) == self.nfunctions ** self.dim
            cheb_coefs = kwargs["cheb_coefs"]
            return self.postprocess_cheb_table(
                self.build_cheb_table(queue, **kwargs), cheb_coefs
            )
        else:
            logger.info("Returning cheb table directly")
            return self.build_cheb_table(queue, **kwargs)


# }}} End reduced Droste method

# {{{ inverse Droste method


class InverseDrosteReduced(DrosteReduced):
    r"""
    A variant of the Droste method.
    Instead of computing a volume potential, it computes the "inverse" to a
    Riesz potential, aka a "fractional Laplacian".

    Specifically, given an integral kernel G(r), this class supports the
    precomputation for integrals of the form

    .. math::

       \int_B G(r) (u(x) - u(y)) dy

    For k-dimensional fractional Laplacian, :math:`G(r) = \frac{1}{r^{k+2s}}`.

    The core part of concern (that is to be modified based on DrosteReduces):

    .. code-block::

        ...

        for BASIS_VARS, TGT_VARS, icase
            for ilevel
                for ibrick_axis, ibrick_side, QUAD_VARS

                    PREPARE_BASIS_VALS

                    <> density_val = DENSITY_VAL_ASSIGNMENT \
                            {id=density,dep=basis_evals}

                end
            end
        end

        ...

    """

    def __init__(self, integral_knl, quad_order, case_vecs,
                 n_brick_quad_points=50, knl_symmetry_tags=None,
                 special_radial_quadrature=False, nradial_quad_points=None,
                 auto_windowing=True):
        """
        :param auto_windowing: auto-detect window radius.
        """
        super().__init__(
            integral_knl, quad_order, case_vecs,
            n_brick_quad_points, special_radial_quadrature,
            nradial_quad_points, knl_symmetry_tags)
        self.auto_windowing = auto_windowing

    def get_cache_key(self):
        return (
            type(self).__name__,
            str(self.dim) + "D",
            self.integral_knl.__str__(),
            "quad_order-" + str(self.ntgt_points),
            "brick_order-" + str(self.nquad_points),
            "brick_radial_order-" + str(self.nradial_quad_points),
            "case-" + str(self.current_base_case),
            "kernel_id-" + str(self.get_kernel_id),)

    def codegen_basis_tgt_eval(self, iaxis):
        """Generate instructions to evaluate Chebyshev polynomial basis
        at the target point, given that the target point lies in the
        source box. (Chebyshev polynomials of the first kind T_n).

        If the target point is not in the source box, the concerned instructions
        will return 0.
        """

        # only valid for self-interactions
        assert all(
            self.reduce_by_symmetry.reduced_vecs[self.current_base_case][d] == 0
            for d in range(self.dim))

        code = """  # noqa
            <> T0_tgt_IAXIS = 1
            <> T1_tgt_IAXIS = template_true_target[IAXIS] {dep=template_true_targets}
            <> Tprev_tgt_IAXIS = T0_tgt_IAXIS {id=t0_tgt_IAXIS}
            <> Tcur_tgt_IAXIS = T1_tgt_IAXIS {id=t1_tgt_IAXIS,dep=t0_tgt_IAXIS}

            for pIAXIS
                <> Tnext_tgt_IAXIS = (2 * template_true_target[IAXIS] * Tcur_tgt_IAXIS
                                - Tprev_tgt_IAXIS) {id=tnext_tgt_IAXIS,dep=t1_tgt_IAXIS}
                Tprev_tgt_IAXIS = Tcur_tgt_IAXIS {id=tprev_tgt_updateIAXIS,dep=tnext_tgt_IAXIS}
                Tcur_tgt_IAXIS = Tnext_tgt_IAXIS {id=tcur_tgt_updateIAXIS,dep=tprev_tgt_updateIAXIS}
            end

            <> basis_tgt_evalIAXIS = (
                T0_tgt_IAXIS * if(fIAXIS == 0, 1, 0)
                + T1_tgt_IAXIS * if(fIAXIS == 1, 1, 0)
                + simul_reduce(sum, pIAXIS,
                    if(fIAXIS >= 2 and fIAXIS == pIAXIS + 2, Tcur_tgt_IAXIS, 0))

                ) {id=tgtbasisIAXIS,dep=tcur_tgt_updateIAXIS}
            """.replace(
            "IAXIS", str(iaxis)
        )
        return code

    def codegen_der2_basis_tgt_eval(self, iaxis):
        r"""Generate instructions to evaluate the second order derivatives
        of Chebyshev polynomial basis at the target point, given that the target
        lies in the source box. (Chebyshev polynomials of the first kind T_n).

        If the target point is not in the source box, the concerned instructions
        will return 0.

        The evaluation is based on Chebyshev polynomials of the second kind
        :math:`U_n`.

        .. math::

           \frac{d^2 T_n}{dx^2} = n \frac{(n+1)T_n - U_n}{x^2 - 1}
        """

        # only valid for self-interactions
        assert all(
            self.reduce_by_symmetry.reduced_vecs[self.current_base_case][d] == 0
            for d in range(self.dim))

        code = """  # noqa
            <> U0_tgt_IAXIS = 1
            <> U1_tgt_IAXIS = 2 * template_true_target[IAXIS] {dep=template_true_targets}
            <> Uprev_tgt_IAXIS = U0_tgt_IAXIS {id=u0_tgt_IAXIS}
            <> Ucur_tgt_IAXIS = U1_tgt_IAXIS {id=u1_tgt_IAXIS,dep=u0_tgt_IAXIS}

            for pIAXIS
                <> Unext_tgt_IAXIS = (2 * template_true_target[IAXIS] * Ucur_tgt_IAXIS
                                - Uprev_tgt_IAXIS) {id=unext_tgt_IAXIS,dep=u1_tgt_IAXIS}
                Uprev_tgt_IAXIS = Ucur_tgt_IAXIS {id=uprev_tgt_updateIAXIS,dep=unext_tgt_IAXIS}
                Ucur_tgt_IAXIS = Unext_tgt_IAXIS {id=ucur_tgt_updateIAXIS,dep=uprev_tgt_updateIAXIS}
            end

            # U_n(target)
            <> basis2_tgt_evalIAXIS = (
                U0_tgt_IAXIS * if(fIAXIS == 0, 1, 0)
                + U1_tgt_IAXIS * if(fIAXIS == 1, 1, 0)
                + simul_reduce(sum, pIAXIS,
                    if(fIAXIS >= 2 and fIAXIS == pIAXIS + 2, Ucur_tgt_IAXIS, 0))
                ) {id=tgtbasis2IAXIS,dep=ucur_tgt_updateIAXIS}

            # this temp var helps with type deduction
            <> f_order_IAXIS = fIAXIS
            <> der2_basis_tgt_evalIAXIS = f_order_IAXIS * (
                    ((f_order_IAXIS + 1) * basis_tgt_evalIAXIS - basis2_tgt_evalIAXIS)
                    / (template_true_target[IAXIS]**2 - 1)
                ) * (2**2) / (root_extent[IAXIS]**2) {id=tgtd2basisIAXIS,dep=tgtbasisIAXIS:tgtbasis2IAXIS}
            """.replace(
                    "IAXIS", str(iaxis)
                    )
        return code

    def codegen_windowing_function(self):
        r"""Given :math:`dist = x - y`, compute the windowing function.
        """

        code = []
        if self.dim == 1:
            code.append("<> distsq = dist[0]*dist[0]")
        elif self.dim == 2:
            code.append("<> distsq = dist[0]*dist[0] + dist[1]*dist[1]")
        elif self.dim == 3:
            code.append("<> distsq = dist[0]*dist[0] + \
                                     dist[1]*dist[1] + dist[2]*dist[2]")
        # renormalized distance
        code.append("<> rndist = sqrt(distsq) / delta")

        if True:
            # polynomial windowing
            logger.info("Using polynomial windowing function.")
            code.append(r"""
                <> windowing = if(rndist >= 1,
                                  0,
                                  (1 - 35 * (rndist**4)
                                  + 84 * (rndist**5)
                                  - 70 * (rndist**6)
                                  + 20 * (rndist**7)
                                  )
                                 )
                """)
        elif False:
            # classical bump function
            logger.info("Using bump windowing function.")
            code.append(
                "<> windowing = exp(1) * exp(-(1/(1 - (distsq / (delta*delta)))))")
        else:
            # smooth transitions of 0-->1-->0
            logger.info("Using smooth transition windowing function.")
            code.append("<> fv = if(rndist > 0, exp(-1 / rndist), 0)")
            code.append("<> fc = if(1 - rndist > 0, exp(-1 / (1 - rndist)), 0)")
            code.append("<> windowing = 1 - fv / (fv + fc)")

        return "\n".join(code)

    def make_dim_independent(self, knlstring):
        r"""Produce the correct
        :math:`\text{DENSITY_VAL_ASSIGNMENT} = u(x) - u(y)`
        for self-interactions.
        """

        # detect for self-interactions
        if all(
                self.reduce_by_symmetry.reduced_vecs[self.current_base_case][d] == 0
                for d in range(self.dim)):
            target_box_is_source = True
        else:
            target_box_is_source = False

        # replace REV_* first
        resknl = knlstring.replace("REV_BASIS_VARS", ",".join(self.basis_vars[::-1]))
        resknl = resknl.replace("REV_TGT_VARS", ",".join(self.tgt_vars[::-1]))

        resknl = resknl.replace("BASIS_VARS", ",".join(self.basis_vars))
        resknl = resknl.replace("TGT_VARS", ",".join(self.tgt_vars))
        resknl = resknl.replace("QUAD_VARS", ",".join(self.quad_vars))

        if self.get_kernel_id == 0:
            resknl = resknl.replace(
                    "POSTPROCESS_KNL_VAL",
                    "\n".join([
                        self.codegen_windowing_function(),
                        "<> knl_val_post = windowing * knl_val {id=pp_kval}"
                        ])
                    )
        elif self.get_kernel_id == 1:
            resknl = resknl.replace(
                    "POSTPROCESS_KNL_VAL",
                    "\n".join([
                        self.codegen_windowing_function(),
                        "<> knl_val_post = (1 - windowing) * knl_val {id=pp_kval}"
                        ])
                    )
        else:
            pass

        # {{{ density evals

        basis_eval_insns = [self.codegen_basis_eval(i) for i in range(self.dim)]

        if target_box_is_source:
            basis_eval_insns += [
                    self.codegen_basis_tgt_eval(i) for i in range(self.dim)]

        if self.get_kernel_id == 0:
            # Given target x,
            # u(x) - u(y) p.v. integrated around a small region symmetric to x,
            # truncated to second order 0.5 * [(x - y)' * diag(Hess(u)(x)) * (x - y)]
            if target_box_is_source:
                basis_eval_insns += [
                        self.codegen_der2_basis_tgt_eval(i) for i in range(self.dim)]

                resknl = resknl.replace(
                        "PREPARE_BASIS_VALS",
                        "\n".join(basis_eval_insns + [
                            "... nop {id=basis_evals,dep=%s}"
                            % ":".join(
                                ["basis%d" % i for i in range(self.dim)]
                                + ["tgtbasis%d" % i for i in range(self.dim)]
                                + ["tgtd2basis%d" % i for i in range(self.dim)]
                                ),
                            ])
                        )
            else:
                resknl = resknl.replace(
                        "PREPARE_BASIS_VALS",
                        "\n".join(basis_eval_insns + [
                            "... nop {id=basis_evals,dep=%s}"
                            % ":".join(
                                ["basis%d" % i for i in range(self.dim)]
                                ),
                            ])
                        )

            if self.dim == 1:
                if target_box_is_source:
                    resknl = resknl.replace(
                            "DENSITY_VAL_ASSIGNMENT",
                            " ".join([
                                "0.5 * der2_basis_tgt_eval0 * (dist[0]**2)",
                                ])
                            )
                else:
                    resknl = resknl.replace(
                            "DENSITY_VAL_ASSIGNMENT",
                            "- basis_eval0"
                            )

            elif self.dim == 2:
                if target_box_is_source:
                    resknl = resknl.replace(
                            "DENSITY_VAL_ASSIGNMENT",
                            " ".join([
                                "  0.5 * der2_basis_tgt_eval0 * basis_tgt_eval1 * (dist[0]**2)",  # noqa: E501
                                "+ 0.5 * basis_tgt_eval0 * der2_basis_tgt_eval1 * (dist[1]**2)",  # noqa: E501
                                ])
                            )
                else:
                    resknl = resknl.replace(
                            "DENSITY_VAL_ASSIGNMENT",
                            "- basis_eval0 * basis_eval1"
                            )

            elif self.dim == 3:
                if target_box_is_source:
                    resknl = resknl.replace(
                            "DENSITY_VAL_ASSIGNMENT",
                            " ".join([
                                "  0.5 * der2_basis_tgt_eval0 * basis_tgt_eval1 * basis_tgt_eval2 * (dist[0]**2)",  # noqa: E501
                                "+ 0.5 * basis_tgt_eval0 * der2_basis_tgt_eval1 * basis_tgt_eval2 * (dist[1]**2)",  # noqa: E501
                                "+ 0.5 * basis_tgt_eval0 * basis_tgt_eval1 * der2_basis_tgt_eval2 * (dist[2]**2)",  # noqa: E501
                                ])
                            )
                else:
                    resknl = resknl.replace(
                            "DENSITY_VAL_ASSIGNMENT",
                            "- basis_eval0 * basis_eval1 * basis_eval2"
                            )

            else:  # self.dim not in [1, 2, 3]
                raise NotImplementedError("No support for dimension %d" % self.dim)

        elif self.get_kernel_id == 1:

            if target_box_is_source:
                # u(x) - u(y)
                resknl = resknl.replace(
                        "PREPARE_BASIS_VALS",
                        "\n".join(basis_eval_insns + [
                            "... nop {id=basis_evals,dep=%s}"
                            % ":".join(
                                ["basis%d" % i for i in range(self.dim)]
                                + ["tgtbasis%d" % i for i in range(self.dim)]
                                ),
                            ])
                        )
                resknl = resknl.replace(
                        "DENSITY_VAL_ASSIGNMENT",
                        " - ".join([
                            " * ".join(
                                ["basis_tgt_eval%d" % i for i in range(self.dim)]),
                            " * ".join(
                                ["basis_eval%d" % i for i in range(self.dim)]),
                            ])
                        )
            else:
                # - u(y)
                resknl = resknl.replace(
                        "PREPARE_BASIS_VALS",
                        "\n".join(basis_eval_insns + [
                            "... nop {id=basis_evals,dep=%s}"
                            % ":".join(
                                ["basis%d" % i for i in range(self.dim)]
                                ),
                            ])
                        )
                resknl = resknl.replace(
                        "DENSITY_VAL_ASSIGNMENT",
                        " - " + " * ".join(
                            ["basis_eval%d" % i for i in range(self.dim)]
                            )
                        )

        # }}} End density evals

        resknl = resknl.replace(
            "PROD_QUAD_WEIGHT",
            " * ".join(
                [
                    "quadrature_weights[QID]".replace("QID", qvar)
                    for qvar in self.quad_vars
                ]
            ),
        )

        if self.dim == 1:
            resknl = resknl.replace("TPLTGT_ASSIGNMENT", """target_nodes[t0]""")
            resknl = resknl.replace("QUAD_PT_ASSIGNMENT", """quadrature_nodes[q0]""")

        elif self.dim == 2:
            resknl = resknl.replace(
                "TPLTGT_ASSIGNMENT",
                """if(iaxis == 0, target_nodes[t0], target_nodes[t1])""",
            )
            resknl = resknl.replace(
                "QUAD_PT_ASSIGNMENT",
                """if(iaxis == 0, quadrature_nodes[q0], quadrature_nodes[q1])""",
            )

        elif self.dim == 3:
            resknl = resknl.replace(
                "TPLTGT_ASSIGNMENT",
                """if(iaxis == 0, target_nodes[t0], if(
                          iaxis == 1, target_nodes[t1], target_nodes[t2]))""",
            )
            resknl = resknl.replace(
                "QUAD_PT_ASSIGNMENT",
                """if(iaxis == 0, quadrature_nodes[q0], if(
                  iaxis == 1, quadrature_nodes[q1], quadrature_nodes[q2]))""",
            )

        else:
            raise NotImplementedError

        return resknl

    def get_kernel(self, **kwargs):
        """Get loopy kernel for computation, the get_kernel_id determines
        what task to perform.

        - 0:    Integrate :math:`W(r) G(r) [u(x) - u(y) + grad(u)(y - x)]`
        - 1:    Add the integral of :math:`[1 - W(r)] G(r) [u(x) - u(y)]`
        - 2:    Expansion by symmetry.
        """
        domain = self.make_loop_domain(base_case_id=self.current_base_case)

        extra_kernel_kwarg_types = ()
        if "extra_kernel_kwarg_types" in kwargs:
            extra_kernel_kwarg_types = kwargs["extra_kernel_kwarg_types"]

        extra_loopy_kernel_kwargs = {}
        if "extra_loopy_kernel_kwargs" in kwargs:
            extra_loopy_kernel_kwargs = kwargs["extra_loopy_kernel_kwargs"]

        if self.get_kernel_id == 0 or self.get_kernel_id == 1:
            loopy_knl = lp.make_kernel(  # NOQA
                [domain],
                self.get_kernel_code()
                + self.get_sumpy_kernel_eval_insns(),
                [
                    lp.ValueArg("alpha", np.float64),
                    lp.ValueArg("delta", np.float64),
                    lp.ValueArg("n_cases, nfunctions, quad_order, dim", np.int32),
                    lp.GlobalArg("interaction_case_vecs",
                        np.float64, "dim, n_cases"),
                    lp.GlobalArg("interaction_case_scls", np.float64, "n_cases"),
                    lp.GlobalArg("target_nodes", np.float64, "quad_order"),
                    lp.GlobalArg(
                        "result", None,
                        ", ".join(
                            ["nfunctions" for d in range(self.dim)]
                            + ["quad_order" for d in range(self.dim)]
                        )
                        + ", n_cases",
                    ), ] + list(extra_kernel_kwarg_types)
                + ["...", ],
                name="brick_map_%d" % self.get_kernel_id,
                lang_version=(2018, 2),
                **extra_loopy_kernel_kwargs
            )

        elif self.get_kernel_id == 2:
            loopy_knl = lp.make_kernel(  # NOQA
                [domain],
                self.get_kernel_expansion_by_symmetry_code(),
                [
                    lp.ValueArg("n_cases, nfunctions, quad_order, dim", np.int32),
                    lp.GlobalArg(
                        "result", None,
                        ", ".join(
                            ["nfunctions" for d in range(self.dim)]
                            + ["quad_order" for d in range(self.dim)]
                        )
                        + ", n_cases",
                    ), ] + list(extra_kernel_kwarg_types)
                + ["...", ],
                name="brick_map_expansion",
                lang_version=(2018, 2),
                **extra_loopy_kernel_kwargs
            )

        else:
            raise NotImplementedError

        loopy_knl = lp.fix_parameters(loopy_knl, dim=self.dim)
        loopy_knl = lp.set_options(loopy_knl, write_cl=False)
        loopy_knl = lp.set_options(loopy_knl, return_dict=True)

        # loopy_knl = lp.make_reduction_inames_unique(loopy_knl)

        try:
            loopy_knl = self.integral_knl.prepare_loopy_kernel(loopy_knl)
        except Exception:  # noqa: B902
            pass

        return loopy_knl

    def call_loopy_kernel_case(self, queue, base_case_id, **kwargs):
        """
        Call the table builder on one base case, as given in :self.current_base_case:
        :arg source_box_extent
        :arg alpha
        :arg delta
        :arg nlevels
        :arg extra_kernel_kwargs
        """

        if base_case_id != self.current_base_case:
            self.current_base_case = base_case_id

        if "source_box_extent" in kwargs:
            assert kwargs["source_box_extent"] > 0
            source_box_extent = kwargs["source_box_extent"]
        else:
            source_box_extent = 1

        extra_kernel_kwargs = {}
        if "extra_kernel_kwargs" in kwargs:
            extra_kernel_kwargs = kwargs["extra_kernel_kwargs"]

        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
            assert alpha >= 0 and alpha < 1
        else:
            alpha = 0

        # (template) target points in 1D over [0, 1]
        q_points = self.get_target_points(queue)
        assert len(q_points) == self.ntgt_points ** self.dim
        t = np.array([pt[-1] for pt in q_points[: self.ntgt_points]])

        tt = t * source_box_extent
        delta_max = min(np.min(tt), source_box_extent - np.max(tt))
        assert delta_max > 0
        if delta_max < 1e-6:
            logger.warn("Severe delta constraint (< %f)" % delta_max)

        if ("delta" in kwargs) and (not self.auto_windowing):
            delta = kwargs["delta"]
            logger.info("Using window radius %f" % delta)
            assert delta > 0 and 2 * delta < source_box_extent
        else:
            assert self.auto_windowing
            delta = 0.9 * delta_max
            logger.info("Using auto-determined window radius %f" % delta)

        if delta > delta_max:
            delta = min(delta, delta_max)
            logger.info("Shrinked delta to %f to fit inside the source box" % delta)

        if "nlevels" in kwargs:
            nlevels = kwargs["nlevels"]
            assert nlevels > 0
        else:
            # Single level is equivalent to Duffy transform
            nlevels = 1

        missing_measure = (alpha ** nlevels * source_box_extent) ** self.dim
        if abs(missing_measure) > 1e-6:
            from warnings import warn

            warn(
                "Droste probably has too few levels, missing measure = "
                + str(missing_measure)
            )

        if "result_array" in kwargs:
            result_array = kwargs["result_array"]
        else:
            result_array = self.make_result_array(**kwargs)

        # root brick
        root_brick = np.zeros((self.dim, 2))
        root_brick[:, 1] = source_box_extent

        base_case_vec = np.array(
            [
                [self.reduce_by_symmetry.reduced_vecs[self.current_base_case][d]]
                for d in range(self.dim)
            ]
        )
        base_case_scl = np.array(
            [
                self.interaction_case_scls[
                    self.reduce_by_symmetry.reduced_vec_ids[self.current_base_case]
                ]
            ]
        )

        # --------- call kernel 0 ----------
        self.get_kernel_id = 0
        try:
            delattr(self, "_memoize_dic_get_cached_optimized_kernel")
        except Exception:  # noqa: B902
            pass
        knl = self.get_cached_optimized_kernel()
        result_array_0 = self.make_result_array(**kwargs)
        evt0, res0 = knl(
            queue,
            alpha=alpha, delta=delta,
            result=result_array_0,
            root_brick=root_brick,
            target_nodes=t.astype(np.float64, copy=True),
            interaction_case_vecs=base_case_vec.astype(np.float64, copy=True),
            interaction_case_scls=base_case_scl.astype(np.float64, copy=True),
            n_cases=1,
            nfunctions=self.nfunctions,
            quad_order=self.ntgt_points,
            nlevels=nlevels,
            **extra_kernel_kwargs, **self.brick_quadrature_kwargs
        )

        # --------- call kernel 1 ----------
        self.get_kernel_id = 1
        try:
            delattr(self, "_memoize_dic_get_cached_optimized_kernel")
        except Exception:  # noqa: B902
            pass
        result_array_1 = self.make_result_array(**kwargs)
        knl = self.get_cached_optimized_kernel()
        evt1, res1 = knl(
            queue,
            alpha=alpha, delta=delta,
            result=result_array_1,
            root_brick=root_brick,
            target_nodes=t.astype(np.float64, copy=True),
            interaction_case_vecs=base_case_vec.astype(np.float64, copy=True),
            interaction_case_scls=base_case_scl.astype(np.float64, copy=True),
            n_cases=1,
            nfunctions=self.nfunctions,
            quad_order=self.ntgt_points,
            nlevels=nlevels,
            **extra_kernel_kwargs, **self.brick_quadrature_kwargs
        )

        # --------- call kernel 2 ----------
        self.get_kernel_id = 2
        try:
            delattr(self, "_memoize_dic_get_cached_optimized_kernel")
        except Exception:  # noqa: B902
            pass
        knl2 = self.get_cached_optimized_kernel()
        result_array = res0["result"] + res1["result"]

        evt2, res2 = knl2(
            queue,
            result=result_array,
            n_cases=1,
            nfunctions=self.nfunctions,
            quad_order=self.ntgt_points,
            nlevels=nlevels,
            **extra_kernel_kwargs
        )

        cheb_table_case = res2["result"]
        return cheb_table_case

# }}} End inverse Droste method
