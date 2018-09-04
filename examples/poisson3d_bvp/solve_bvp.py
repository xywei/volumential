__copyright__ = "Copyright (C) 2017 Xiaoyu Wei"

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

from sumpy.kernel import LaplaceKernel
from pytential.symbolic.pde.scalar import DirichletOperator
op = DirichletOperator(LaplaceKernel(dim), loc_sign, use_l2_weighting=True)

sym_sigma = sym.var("sigma")
op_sigma = op.operator(sym_sigma)

from pytential.qbx import QBXLayerPotentialSource
qbx, _ = QBXLayerPotentialSource(
        bdry_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
        fmm_order=qbx_fmm_order,
        ).with_refinement()

bdry_discr = qbx.density_discr

# debug output
if 1:
    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(queue, bdry_discr, visual_order)

    bdry_normals = bind(bdry_discr, sym.normal(dim))(queue).as_vector(dtype=object)

    bdry_vis.write_vtk_file("boundary_nornmals.vtu", [
        ("bdry_normals", bdry_normals),
        ])

bound_op = bind(qbx, op_sigma)

bvp_bc = bdry_vals
bdry_f = source_field(bdry_nodes_x, bdry_nodes_y, bdry_nodes_z)

bvp_rhs = bind(bdry_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bvp_bc)

from pytential.solve import gmres
gmres_result = gmres(
        bound_op.scipy_op(queue, "sigma", dtype=np.float64),
        bvp_rhs, tol=1e-14, progress=True,
        hard_failure=False)

sigma = gmres_result.solution
print("gmres state:", gmres_result.state)
