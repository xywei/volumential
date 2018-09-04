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

# FIXME: add support to sumpy for multiple DirectionalSourceDerivative compositions
# (currently it raises a NotImplementedError)

# {{{ set some constants for use below

dim = 2
nelements = 20
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = 80
bdry_ovsmp_quad_order = 4*bdry_quad_order
fmm_order = 10

loc_sign = -1

# }}}

def main():
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info
    logger = logging.getLogger(__name__)

    import numpy as np
    import pyopencl as cl
    import pyopencl.clmath
    from pytential import sym
    from functools import partial

    from pytools.obj_array import make_obj_array

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ make mesh

    from meshmode.mesh.generation import ellipse, make_curve_mesh

    if 1 and dim == 2:
            # one circle
            mesh = make_curve_mesh(
                    partial(ellipse, 1),
                    np.linspace(0, 1, nelements+1),
                    mesh_order)
    elif dim == 2:
        # array of circles
        base_mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)

        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
        nx = 2
        ny = 2
        dx = 2 / nx
        meshes = [
                affine_map(
                    base_mesh,
                    A=np.diag([dx*0.25, dx*0.25]),
                    b=np.array([dx*(ix-nx/2), dx*(iy-ny/2)]))
                for ix in range(nx)
                for iy in range(ny)]

        mesh = merge_disjoint_meshes(meshes, single_group=True)
    else:
        raise NotImplementedError("Could not find a way to get the mesh.")

    if 0:
        from meshmode.mesh.visualization import draw_curve
        draw_curve(mesh)
        import matplotlib.pyplot as plt
        plt.show()

    # }}} End make mesh

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import (
            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order
            ).with_refinement()
    density_discr = qbx.density_discr

    # {{{ describe bvp

    unk = sym.make_sym_vector("sigma", 2)

    from sumpy.kernel import BiharmonicKernel, DirectionalSourceDerivative
    GB = BiharmonicKernel(dim)

    nrml = sym.normal(ambient_dim=dim, where=None)

    # only applies in 2D
    # TODO: deal with 3D with geometric algebra notation
    tngn = sym.MultiVector(
            nrml.xproject(1)[[1,0]] * np.array([-1, 1]))

    nabla = sym.MultiVector(make_obj_array(
        [sym.NablaComponent(axis, None) for axis in range(dim)]))

    dn = lambda kernel: DirectionalSourceDerivative(kernel, "normal")
    dtau = lambda kernel: DirectionalSourceDerivative(kernel, "tangential")
    kernel_arguments = {"normal": nrml.xproject(1), "tangential": tngn.xproject(1)}

    # Farkas representation
    def LP_KF_1(density, qbx_forced_limit=None):
        return sym.IntG(
                kernel=dn(dn(dn(GB))),
                density=density,
                qbx_forced_limit=qbx_forced_limit,
                kernel_arguments=kernel_arguments
                ) + 3 * sym.IntG(
                        kernel=dtau(dtau(dn(GB))),
                        density=density,
                        qbx_forced_limit=qbx_forced_limit,
                        kernel_arguments=kernel_arguments
                        )

    def LP_KF_2(density, qbx_forced_limit=None):
        return - sym.IntG(
                kernel=dn(dn(GB)),
                density=density,
                qbx_forced_limit=qbx_forced_limit,
                kernel_arguments=kernel_arguments
                ) + sym.IntG(
                        kernel=dtau(dtau(GB)),
                        density=density,
                        qbx_forced_limit=qbx_forced_limit,
                        kernel_arguments=kernel_arguments
                        )

    def operator(unknown):
        sig1, sig2 = unknown
        K11 = partial(LP_KF_1, qbx_forced_limit='avg')
        K12 = partial(LP_KF_2, qbx_forced_limit='avg')
        K21 = lambda density: sym.normal_derivative(dim, K11(density))
        K22 = lambda density: sym.normal_derivative(dim, K12(density))
        kappa = sym.mean_curvature(dim)

        d = - loc_sign * sym.make_obj_array([
            0.5 * sig1,
            - kappa * sig1 + 0.5 * sig2,
            ])
        a = sym.make_obj_array([
            K11(sig1) + K12(sig2),
            K21(sig1) + K22(sig2)
            ])
        return d + a

    from pytential import bind
    bdry_op_sym = operator(unk)

    # }}}

    bound_op = bind(qbx, bdry_op_sym)

    # {{{ fix rhs and solve

    nodes = density_discr.nodes().with_queue(queue)

    def bdry_conditions_func(x):
        if len(x) == 2:
            return sym.make_obj_array([
                x[0] * 0 + 2,
                x[0] * 0
                ])
        else:
            raise NotImplementedError

    bc = bdry_conditions_func(nodes)

    import pudb; pu.db

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.float64),
            bc, tol=1e-8, progress=True, maxiter=150,
            stall_iterations=0,
            hard_failure=False)

    # }}}

    # {{{ postprocess/visualize

    from datetime import datetime
    def file_id():
        # return str(datetime.now()).replace(' ', '::')
        return 'latest'

    sigma = gmres_result.solution

    if 0:
        logger.warn("Overwriting solutions for debugging")
        # test out layer potentials
        sigma[0] *= 0
        sigma[1] *= 0

        sigma[0] += -2
        sigma[1] += 10000

    representation_sym = biharmonic_operator.representation(unk, qbx_forced_limit=None, sscale=sscale)
    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(dim), extent=2.1, npoints=500)

    targets = cl.array.to_device(queue, fplot.points)

    bdry_conditions = bdry_conditions_func(targets)

    qbx_stick_out = qbx.copy(target_association_tolerance=0.75)

    if 1:
        # boundary plot for operator info
        visual_order = 2
        from meshmode.discretization.visualization import make_visualizer

        solu_on_bdry = bind((qbx, density_discr), bdry_op_sym)(queue, sigma=sigma)
        solu_bdry_representation = bind(
                (qbx, density_discr),
                biharmonic_operator.representation(unk, qbx_forced_limit=loc_sign, sscale=sscale)
                )(queue, sigma=sigma)
        solu_bdry_representation_other_side = bind(
                (qbx, density_discr),
                biharmonic_operator.representation(unk, qbx_forced_limit=-loc_sign, sscale=sscale)
                )(queue, sigma=sigma)

        bdplot = make_visualizer(queue, density_discr, visual_order)
        bdry_normals = bind(density_discr, sym.normal(dim))(queue)\
                .as_vector(dtype=object)

        bdplot.write_vtk_file(
                "potential-biharm-bdry-%s.vtu" % file_id(),
                [
                    ("sig1", sigma[0]),
                    ("sig2", sigma[1]),
                    ("bdry_normals", bdry_normals),
                    ("u", solu_on_bdry[0]),
                    ("u_n", solu_on_bdry[1]),
                    ("bc1", bc[0]),
                    ("bc2", bc[1]),
                    ("representation1", solu_bdry_representation[0]),
                    # ("representation2", solu_bdry_representation[1]),
                    ("representation_os1", solu_bdry_representation_other_side[0]),
                    # ("representation_os2", solu_bdry_representation_other_side[1]),
                    ]
                )

        sig1 = np.mean(sigma[0].get())
        sig2 = np.mean(sigma[1].get())
        actual_jump_u = np.mean(
                solu_bdry_representation_other_side[0].get() -
                solu_bdry_representation[0].get())

        print("sig1:", sig1)
        print("sig2:", sig2)
        print("u:", np.mean(solu_bdry_representation[0].get()),
                "\texpected:", np.mean(bc[0].get()))
        print("jump_u:", actual_jump_u, "\texpected:", 1*sig1)

        if len(solu_bdry_representation_other_side) > 1:
            actual_jump_un = np.mean(
                    solu_bdry_representation_other_side[1].get() -
                    solu_bdry_representation[1].get())

            print("un:", np.mean(solu_bdry_representation[1].get()),
                    "\texpected:", np.mean(bc[1].get()))
            print("jump_un:", actual_jump_un, "\texpected:", 2*sig1-sig2*sscale)

    try:
        from pytential.target import PointsTarget
        solu_representation = bind(
                (qbx_stick_out, PointsTarget(targets)),
                biharmonic_operator.representation(unk,
                    qbx_forced_limit=2*loc_sign,
                    sscale=sscale))(queue, sigma=sigma)
        fld_in_vol = solu_representation[0].get()
    except QBXTargetAssociationFailedException as e:
        fplot.write_vtk_file(
                "failed-targets-%s.vts" % file_id(),
                [
                    ("failed", e.failed_target_flags.get(queue))
                    ]
                )
        raise

    fplot.write_vtk_file(
            "potential-biharm-%s.vts" % file_id(),
            [
                ("potential", fld_in_vol),
                ]
            )

    # }}}

if __name__ == "__main__":
    main()
