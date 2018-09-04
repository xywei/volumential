from sumpy.kernel import LaplaceKernel
from pytential.symbolic.pde.scalar import DirichletOperator
op = DirichletOperator(LaplaceKernel(dim), loc_sign, use_l2_weighting=True)

sym_sigma = sym.var("sigma")
op_sigma = op.operator(sym_sigma)

from pytential.qbx import QBXLayerPotentialSource
qbx = QBXLayerPotentialSource(
        bdry_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
        fmm_order=qbx_fmm_order,
        )

bound_op = bind(qbx, op_sigma)

bvp_bc = bdry_vals
bdry_f = source_field(bdry_nodes_x, bdry_nodes_y)

bvp_rhs = bind(bdry_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bvp_bc)

from pytential.solve import gmres
gmres_result = gmres(
        bound_op.scipy_op(queue, "sigma", dtype=np.float64),
        bvp_rhs, tol=1e-14, progress=True,
        hard_failure=False)

sigma = gmres_result.solution
print("gmres state:", gmres_result.state)
