import os
import sys

import numpy as np

import pyopencl as cl
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
)
from pytential.array_context import PyOpenCLArrayContext
from pytential.target import PointsTarget

from volumential.function_extension import (
    compute_constant_extension,
    compute_harmonic_extension,
)


def _skip_unstable_backend(ctx):
    platform_names = {dev.platform.name for dev in ctx.devices}
    if any(name == "Intel(R) OpenCL" for name in platform_names):
        import pytest

        pytest.skip("QBX function-extension tests are unstable on Intel(R) OpenCL")


def _make_test_qbx(actx, nelements=40, order=4, qbx_order=3):
    from functools import partial

    from meshmode.mesh.generation import make_curve_mesh
    from meshmode.mesh.generation import ellipse
    from pytential.qbx import QBXLayerPotentialSource

    mesh = make_curve_mesh(
        partial(ellipse, 1.0), np.linspace(0, 1, nelements + 1), order
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
