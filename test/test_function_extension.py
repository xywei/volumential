import os
import sys

import numpy as np
import pytest

import pyopencl as cl
from arraycontext import flatten, unflatten
from meshmode.dof_array import DOFArray
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
)
from pytential.array_context import PyOpenCLArrayContext
from pytential.target import PointsTarget

from volumential.function_extension import (
    compute_biharmonic_extension,
    compute_constant_extension,
    compute_harmonic_extension,
)


def _skip_unstable_backend(ctx):
    platform_names = {dev.platform.name for dev in ctx.devices}
    if any(name == "Intel(R) OpenCL" for name in platform_names):
        import pytest

        pytest.skip("QBX function-extension tests are unstable on Intel(R) OpenCL")


def _make_closed_curve_mesh_skip_orientation(curve_f, element_boundaries, order):
    import modepy as mp
    from meshmode.mesh import SimplexElementGroup, make_mesh

    nelements = len(element_boundaries) - 1
    unit_nodes = mp.warp_and_blend_nodes(1, order)
    nodes_01 = 0.5 * (unit_nodes + 1)

    vertices = curve_f(element_boundaries)[:, :nelements]
    vertex_indices = np.vstack(
        [
            np.arange(0, nelements, dtype=np.int32),
            np.arange(1, nelements + 1, dtype=np.int32) % nelements,
        ]
    ).T

    el_lengths = np.diff(element_boundaries)
    el_starts = element_boundaries[:-1]
    t = el_starts[:, np.newaxis] + el_lengths[:, np.newaxis] * nodes_01
    nodes = curve_f(t.ravel()).reshape(vertices.shape[0], nelements, -1)

    egroup = SimplexElementGroup.make_group(
        order,
        vertex_indices=vertex_indices,
        nodes=nodes,
        unit_nodes=unit_nodes,
    )

    return make_mesh(
        vertices=vertices,
        groups=[egroup],
        is_conforming=True,
        skip_element_orientation_test=True,
    )


def _make_test_qbx(actx, nelements=40, order=4, qbx_order=3):
    from functools import partial

    from meshmode.mesh.generation import ellipse
    from pytential.qbx import QBXLayerPotentialSource

    mesh = _make_closed_curve_mesh_skip_orientation(
        partial(ellipse, 1.0),
        np.linspace(0, 1, nelements + 1),
        order,
    )
    discr = Discretization(
        actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(order)
    )
    qbx = QBXLayerPotentialSource(
        discr,
        fine_order=4 * order,
        qbx_order=qbx_order,
        fmm_order=False,
    )
    return qbx, discr


def test_constant_extension_geometry_collection(ctx_factory):
    ctx = ctx_factory()
    _skip_unstable_backend(ctx)
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    qbx, density_discr = _make_test_qbx(actx)
    targets = np.array([[1.5, -1.25, 0.0], [0.0, 0.5, -1.75]], dtype=np.float64)
    target_discr = PointsTarget(actx.freeze(actx.from_numpy(targets)))

    ext_f, _ = compute_constant_extension(
        queue,
        target_discr,
        qbx,
        density_discr,
        constant_val=2.5,
        actx=actx,
    )

    np.testing.assert_allclose(actx.to_numpy(ext_f), 2.5)


def test_harmonic_extension_geometry_collection_smoke(ctx_factory):
    if (
        sys.platform == "darwin"
        and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
    ):
        import pytest

        pytest.skip(
            "harmonic extension test is unstable on macOS OpenCL CI "
            "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)"
        )

    ctx = ctx_factory()
    _skip_unstable_backend(ctx)
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    qbx, density_discr = _make_test_qbx(actx)
    nodes = actx.thaw(density_discr.nodes())
    f = nodes[0]

    target_points = np.array(
        [[0.5, -0.4, 0.0, 0.2], [0.0, 0.4, -0.6, 0.1]],
        dtype=np.float64,
    )
    target_discr = PointsTarget(actx.freeze(actx.from_numpy(target_points)))

    ext_f, debug = compute_harmonic_extension(
        queue,
        target_discr,
        qbx,
        density_discr,
        f,
        loc_sign=-1,
        target_association_tolerance=0.05,
        gmres_tolerance=1e-10,
        actx=actx,
    )

    result = actx.to_numpy(ext_f)

    assert debug["gmres_result"].state == "success"
    assert result.shape == (target_points.shape[1],)
    assert np.isfinite(result).all()


def test_harmonic_extension_to_volume_mesh_points_smoke(ctx_factory):
    if (
        sys.platform == "darwin"
        and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
    ):
        import pytest

        pytest.skip(
            "harmonic extension test is unstable on macOS OpenCL CI "
            "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)"
        )

    ctx = ctx_factory()
    _skip_unstable_backend(ctx)
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    qbx, density_discr = _make_test_qbx(actx, nelements=24, order=3, qbx_order=2)
    nodes = actx.thaw(density_discr.nodes())
    f = nodes[0]

    import volumential.meshgen as mg

    mesh = mg.MeshGen2D(degree=2, nlevels=2, a=-0.4, b=0.4, queue=queue)
    target_points = np.ascontiguousarray(
        np.asarray(mesh.get_q_points()[:16], dtype=np.float64).T
    )
    target_discr = PointsTarget(actx.freeze(actx.from_numpy(target_points)))

    ext_f, debug = compute_harmonic_extension(
        queue,
        target_discr,
        qbx,
        density_discr,
        f,
        loc_sign=-1,
        target_association_tolerance=0.05,
        gmres_tolerance=1e-10,
        actx=actx,
    )

    result = actx.to_numpy(ext_f)

    assert debug["gmres_result"].state == "success"
    assert result.shape == (target_points.shape[1],)
    assert np.isfinite(result).all()


def test_harmonic_extension_interior_linear_accuracy(ctx_factory):
    if (
        sys.platform == "darwin"
        and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
    ):
        import pytest

        pytest.skip(
            "harmonic extension test is unstable on macOS OpenCL CI "
            "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)"
        )

    ctx = ctx_factory()
    _skip_unstable_backend(ctx)
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    qbx, density_discr = _make_test_qbx(actx, nelements=40, order=4, qbx_order=3)
    nodes = actx.thaw(density_discr.nodes())
    f = nodes[0]

    target_points = np.array(
        [[0.2, -0.15, 0.35, -0.3], [0.1, 0.25, -0.2, -0.1]],
        dtype=np.float64,
    )
    target_discr = PointsTarget(actx.freeze(actx.from_numpy(target_points)))

    ext_f, debug = compute_harmonic_extension(
        queue,
        target_discr,
        qbx,
        density_discr,
        f,
        loc_sign=-1,
        target_association_tolerance=0.05,
        gmres_tolerance=1e-10,
        actx=actx,
    )

    result = actx.to_numpy(ext_f)
    expected = target_points[0]

    assert debug["gmres_result"].state == "success"
    assert debug["representation_mode"] == "d_only"
    np.testing.assert_allclose(result, expected, atol=2.0e-2, rtol=0.0)


def test_biharmonic_extension_linear_accuracy(ctx_factory):
    if (
        sys.platform == "darwin"
        and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
    ):
        import pytest

        pytest.skip(
            "biharmonic extension test is unstable on macOS OpenCL CI "
            "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)"
        )

    ctx = ctx_factory()
    _skip_unstable_backend(ctx)
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    qbx, density_discr = _make_test_qbx(actx, nelements=40, order=4, qbx_order=3)

    bdry_nodes = actx.thaw(density_discr.nodes())
    xb = actx.to_numpy(flatten(bdry_nodes[0], actx, leaf_class=DOFArray))
    yb = actx.to_numpy(flatten(bdry_nodes[1], actx, leaf_class=DOFArray))

    f = unflatten(
        bdry_nodes[0],
        actx.from_numpy(np.ascontiguousarray(xb)),
        actx,
    )
    fx = unflatten(
        bdry_nodes[0],
        actx.from_numpy(np.ascontiguousarray(np.ones_like(xb))),
        actx,
    )
    fy = unflatten(
        bdry_nodes[0],
        actx.from_numpy(np.ascontiguousarray(np.zeros_like(yb))),
        actx,
    )

    target_points = np.array(
        [[0.2, -0.15, 0.35, -0.3], [0.1, 0.25, -0.2, -0.1]],
        dtype=np.float64,
    )
    target_discr = PointsTarget(actx.freeze(actx.from_numpy(target_points)))

    ext_f, ext_fx, ext_fy, debug = compute_biharmonic_extension(
        queue,
        target_discr,
        qbx,
        density_discr,
        f,
        fx,
        fy,
        loc_sign=-1,
        target_association_tolerance=0.05,
        enforce_affine_match=True,
        actx=actx,
    )

    result = actx.to_numpy(ext_f)
    result_fx = actx.to_numpy(ext_fx)
    result_fy = actx.to_numpy(ext_fy)

    assert debug["gmres_result_mu"].state == "success"
    assert debug["gmres_result_sigma"].state == "success"

    np.testing.assert_allclose(result, target_points[0], atol=5.0e-2, rtol=0.0)
    np.testing.assert_allclose(
        result_fx,
        np.ones_like(result_fx),
        atol=8.0e-2,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result_fy,
        np.zeros_like(result_fy),
        atol=8.0e-2,
        rtol=0.0,
    )


def test_biharmonic_extension_cubic_accuracy(ctx_factory):
    if (
        sys.platform == "darwin"
        and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
    ):
        pytest.skip(
            "biharmonic extension test is unstable on macOS OpenCL CI "
            "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)"
        )

    ctx = ctx_factory()
    _skip_unstable_backend(ctx)
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    qbx, density_discr = _make_test_qbx(actx, nelements=40, order=4, qbx_order=3)

    bdry_nodes = actx.thaw(density_discr.nodes())
    xb = actx.to_numpy(flatten(bdry_nodes[0], actx, leaf_class=DOFArray))
    yb = actx.to_numpy(flatten(bdry_nodes[1], actx, leaf_class=DOFArray))

    def u_exact(x, y):
        return x**3 + y**3

    def ux_exact(x, y):
        return 3.0 * x**2

    def uy_exact(x, y):
        return 3.0 * y**2

    f = unflatten(
        bdry_nodes[0],
        actx.from_numpy(np.ascontiguousarray(u_exact(xb, yb))),
        actx,
    )
    fx = unflatten(
        bdry_nodes[0],
        actx.from_numpy(np.ascontiguousarray(ux_exact(xb, yb))),
        actx,
    )
    fy = unflatten(
        bdry_nodes[0],
        actx.from_numpy(np.ascontiguousarray(uy_exact(xb, yb))),
        actx,
    )

    target_points = np.array(
        [[0.2, -0.15, 0.35, -0.3], [0.1, 0.25, -0.2, -0.1]],
        dtype=np.float64,
    )
    target_discr = PointsTarget(actx.freeze(actx.from_numpy(target_points)))

    ext_f, ext_fx, ext_fy, debug = compute_biharmonic_extension(
        queue,
        target_discr,
        qbx,
        density_discr,
        f,
        fx,
        fy,
        loc_sign=-1,
        target_association_tolerance=0.05,
        enforce_affine_match=True,
        actx=actx,
    )

    result = actx.to_numpy(ext_f)
    result_fx = actx.to_numpy(ext_fx)
    result_fy = actx.to_numpy(ext_fy)

    expected = u_exact(target_points[0], target_points[1])
    expected_fx = ux_exact(target_points[0], target_points[1])
    expected_fy = uy_exact(target_points[0], target_points[1])

    assert debug["gmres_result_mu"].state == "success"
    assert debug["gmres_result_sigma"].state == "success"

    np.testing.assert_allclose(result, expected, atol=5.0e-3, rtol=0.0)
    np.testing.assert_allclose(result_fx, expected_fx, atol=1.0e-2, rtol=0.0)
    np.testing.assert_allclose(result_fy, expected_fy, atol=1.0e-2, rtol=0.0)
