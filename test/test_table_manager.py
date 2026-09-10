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

import os
import subprocess
import sys
from shutil import copyfile

import numpy as np
import pytest

if (
    sys.platform == "darwin"
    and os.environ.get("VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS") != "1"
):
    pytest.skip(
        "table manager tests are unstable on macOS OpenCL CI "
        "(set VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1 to run)",
        allow_module_level=True,
    )

import pyopencl as cl

import volumential as vm
from volumential.table_manager import NearFieldInteractionTableManager as NFTable
from volumential.table_manager import TableRequest


def get_table(queue, q_order=1, dim=2):
    pid = os.getpid()

    # copy from shared cache if exists
    if os.path.exists("nft.hdf5"):
        copyfile("nft.hdf5", f"nft-test-table-manager-{pid}.hdf5")

    subprocess.check_call(["rm", "-f", f"nft-test-table-manager-{pid}.hdf5"])

    with NFTable(
        f"nft-test-table-manager-{pid}.hdf5", progress_bar=True
    ) as table_manager:
        table, _ = table_manager.get_table(
            dim, "Laplace", q_order=q_order, force_recompute=False, queue=queue
        )

    return table


def test_case_id(ctx_factory, table_2d_order1):
    table = table_2d_order1
    case_same_box = len(table.interaction_case_vecs) // 2
    assert list(table.interaction_case_vecs[case_same_box]) == [0, 0]


def test_get_table_2d_order1(table_2d_order1):
    table = table_2d_order1
    assert table.dim == 2


def test_get_table_yukawa_requires_lambda(ctx_factory, tmp_path):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / "nft-yukawa-requires-lam.sqlite"
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        with pytest.raises(TypeError, match="missing kernel parameter"):
            table_manager.get_table(
                2,
                "Yukawa",
                q_order=1,
                force_recompute=True,
                queue=queue,
            )


def test_get_table_yukawa_builds_with_lambda(ctx_factory, tmp_path):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / "nft-yukawa-with-lam.sqlite"
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Yukawa",
            q_order=1,
            lam=3.0,
            force_recompute=True,
            queue=queue,
        )

    assert table.is_built
    values = np.array([table.get_entry_data(i) for i in range(len(table.data))])
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize(
    ("kernel_type", "extra_kwargs", "cache_name"),
    [
        ("Laplace-Dx", {}, "nft-laplace-dx.sqlite"),
        ("Yukawa-Dx", {"lam": 2.5}, "nft-yukawa-dx-with-lam.sqlite"),
    ],
)
def test_get_table_derivative_builds(
    ctx_factory,
    tmp_path,
    kernel_type,
    extra_kwargs,
    cache_name,
):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / cache_name
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        table, _ = table_manager.get_table(
            2,
            kernel_type,
            q_order=1,
            force_recompute=True,
            queue=queue,
            **extra_kwargs,
        )

    assert table.is_built
    values = np.array([table.get_entry_data(i) for i in range(len(table.data))])
    assert np.all(np.isfinite(values))


def test_get_table_yukawa_dx_requires_lambda(ctx_factory, tmp_path):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / "nft-yukawa-dx-requires-lam.sqlite"
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        with pytest.raises(TypeError, match="missing kernel parameter"):
            table_manager.get_table(
                2,
                "Yukawa-Dx",
                q_order=1,
                force_recompute=True,
                queue=queue,
            )


def test_load_saved_yukawa_table_rejects_lambda_mismatch(ctx_factory, tmp_path):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / "nft-yukawa-lam-mismatch.sqlite"
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        table_manager.get_table(
            2,
            "Yukawa",
            q_order=1,
            lam=3.0,
            force_recompute=True,
            queue=queue,
        )

        request = TableRequest.from_args(2, "Yukawa", 1, 0)
        with pytest.raises(KeyError, match="kernel parameter 'lam' mismatch"):
            table_manager._load_saved_table_for_request(request, lam=5.0)


def test_load_saved_yukawa_table_accepts_float32_roundtrip(ctx_factory, tmp_path):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / "nft-yukawa-lam-float32-roundtrip.sqlite"
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        table_manager.get_table(
            2,
            "Yukawa",
            q_order=1,
            lam=np.float32(0.1),
            force_recompute=True,
            queue=queue,
        )

        request = TableRequest.from_args(2, "Yukawa", 1, 0)
        table = table_manager._load_saved_table_for_request(request, lam=0.1)

    assert table.is_built


def test_build_routing_survives_the_cache_round_trip(ctx_factory, tmp_path):
    """A cached table remembers whether it was a batched build or a scalar
    fallback, so a warm run can still be told apart from a cold one."""
    import volumential.nearfield_potential_table as npt
    import volumential.opcounters as opcounters

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    original_batched = npt.NearFieldInteractionTable.\
        build_table_via_duffy_radial_batched

    def failing_batched(self, build_queue, *args, **kwargs):
        raise RuntimeError("synthetic batched build failure")

    cache_file = tmp_path / "nft-routing-roundtrip.sqlite"
    npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
        failing_batched
    )
    try:
        with NFTable(str(cache_file), progress_bar=False) as table_manager:
            with pytest.warns(RuntimeWarning, match="falling back to the"):
                built, _ = table_manager.get_table(
                    2,
                    "Laplace",
                    q_order=1,
                    force_recompute=True,
                    queue=queue,
                )
    finally:
        npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
            original_batched
        )

    assert built.build_routing == "scalar-fallback"
    assert built.build_fallback_reason == (
        "RuntimeError: synthetic batched build failure"
    )

    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        request = TableRequest.from_args(2, "Laplace", 1, 0)
        loaded = table_manager._load_saved_table_for_request(request)

    assert loaded.build_routing == "scalar-fallback"
    assert loaded.build_fallback_reason == (
        "RuntimeError: synthetic batched build failure"
    )
    assert opcounters.direct_build_routing(loaded) == "scalar-fallback"
    assert np.allclose(np.asarray(loaded.data), np.asarray(built.data),
                       equal_nan=True)


def test_batched_build_routing_survives_the_cache_round_trip(
    ctx_factory, tmp_path
):
    import volumential.opcounters as opcounters

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / "nft-routing-roundtrip-batched.sqlite"
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        built, _ = table_manager.get_table(
            2, "Laplace", q_order=1, force_recompute=True, queue=queue
        )
        assert built.build_routing == "batched"

        request = TableRequest.from_args(2, "Laplace", 1, 0)
        loaded = table_manager._load_saved_table_for_request(request)

    assert loaded.build_routing == "batched"
    assert loaded.build_fallback_reason is None
    assert opcounters.direct_build_routing(loaded) == "batched"


def _build_fallback_table_cache(cache_file, queue):
    """A cache file whose one entry was produced by the scalar fallback."""
    import volumential.nearfield_potential_table as npt

    original = npt.NearFieldInteractionTable.\
        build_table_via_duffy_radial_batched

    def failing_batched(self, build_queue, *args, **kwargs):
        raise RuntimeError("synthetic batched build failure")

    npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
        failing_batched
    )
    try:
        with NFTable(str(cache_file), progress_bar=False) as table_manager:
            with pytest.warns(RuntimeWarning, match="falling back to the"):
                table_manager.get_table(
                    2, "Laplace", q_order=1, force_recompute=True, queue=queue
                )
    finally:
        npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
            original
        )


def test_strict_mode_refuses_a_cached_scalar_fallback(
    ctx_factory, tmp_path, monkeypatch
):
    """A warmed cache must not smuggle fallback data past strict mode.

    ``VOLUMENTIAL_DUFFY_NO_FALLBACK`` turns the fallback into a build-time
    error, but a cached table skips the builder entirely, so without this
    check a strict campaign whose cache was warmed earlier would load and
    use exactly the data the switch exists to refuse.
    """
    from volumential.nearfield_potential_table import DUFFY_NO_FALLBACK_ENV_VAR
    from volumential.table_manager import UnverifiedBuildRoutingError

    queue = cl.CommandQueue(cl.Context([ctx_factory().devices[0]]))
    cache_file = tmp_path / "nft-strict-cached-fallback.sqlite"
    _build_fallback_table_cache(cache_file, queue)

    # without strict mode the cached fallback loads, routing and all
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        loaded, is_recomputed = table_manager.get_table(
            2, "Laplace", q_order=1, queue=queue
        )
    assert not is_recomputed
    assert loaded.build_routing == "scalar-fallback"

    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        with pytest.raises(UnverifiedBuildRoutingError) as refused:
            table_manager.get_table(2, "Laplace", q_order=1, queue=queue)

    message = str(refused.value)
    assert "scalar Duffy fallback" in message
    assert "synthetic batched build failure" in message
    assert DUFFY_NO_FALLBACK_ENV_VAR in message
    assert "force_recompute=True" in message
    # the refusal must not be mistaken for a cache miss and silently rebuilt
    assert not isinstance(refused.value, KeyError)


def test_strict_mode_refuses_a_cached_table_with_no_recorded_routing(
    ctx_factory, tmp_path, monkeypatch
):
    """A payload written before routing was recorded cannot be vouched for."""
    from volumential.nearfield_potential_table import DUFFY_NO_FALLBACK_ENV_VAR
    from volumential.table_manager import UnverifiedBuildRoutingError

    queue = cl.CommandQueue(cl.Context([ctx_factory().devices[0]]))
    cache_file = tmp_path / "nft-strict-legacy.sqlite"

    import volumential.table_manager as tm

    original = tm._serialize_table_payload

    def without_routing(table):
        # emulate a pre-recording payload
        saved = table.build_routing, table.build_fallback_reason
        table.build_routing = None
        table.build_fallback_reason = None
        try:
            return original(table)
        finally:
            table.build_routing, table.build_fallback_reason = saved

    monkeypatch.setattr(tm, "_serialize_table_payload", without_routing)
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        table_manager.get_table(
            2, "Laplace", q_order=1, force_recompute=True, queue=queue
        )
    monkeypatch.undo()

    import volumential.opcounters as opcounters

    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        loaded, _ = table_manager.get_table(
            2, "Laplace", q_order=1, queue=queue
        )
    assert opcounters.direct_build_routing(loaded) == "unknown"

    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        with pytest.raises(UnverifiedBuildRoutingError, match="no build routing"):
            table_manager.get_table(2, "Laplace", q_order=1, queue=queue)


def test_strict_mode_refusal_follows_the_compatibility_checks(
    ctx_factory, tmp_path, monkeypatch
):
    """An ineligible cache entry is a miss, not a strict-mode refusal.

    The compatibility checks raise ``KeyError``, which ``get_table`` reads
    as a cache miss and recomputes.  Refusing the routing before them would
    turn "this entry is for a different parameter" into a hard error, so a
    strict request for ``lam=5`` would fail merely because the slot still
    holds a fallback table for ``lam=3``.
    """
    import volumential.nearfield_potential_table as npt
    from volumential.nearfield_potential_table import DUFFY_NO_FALLBACK_ENV_VAR

    queue = cl.CommandQueue(cl.Context([ctx_factory().devices[0]]))
    cache_file = tmp_path / "nft-strict-other-parameter.sqlite"

    original = npt.NearFieldInteractionTable.\
        build_table_via_duffy_radial_batched

    def failing_batched(self, build_queue, *args, **kwargs):
        raise RuntimeError("synthetic batched build failure")

    npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
        failing_batched
    )
    try:
        with NFTable(str(cache_file), progress_bar=False) as table_manager:
            with pytest.warns(RuntimeWarning, match="falling back to the"):
                cached, _ = table_manager.get_table(
                    2, "Yukawa", q_order=1, lam=3.0,
                    force_recompute=True, queue=queue,
                )
    finally:
        npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
            original
        )
    assert cached.build_routing == "scalar-fallback"

    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        request = TableRequest.from_args(2, "Yukawa", 1, 0)
        # a request the cached entry cannot satisfy is a plain cache miss
        with pytest.raises(KeyError, match="kernel parameter 'lam' mismatch"):
            table_manager._load_saved_table_for_request(request, lam=5.0)

        # ... while the matching request is the one strict mode refuses
        from volumential.table_manager import UnverifiedBuildRoutingError

        with pytest.raises(UnverifiedBuildRoutingError):
            table_manager._load_saved_table_for_request(request, lam=3.0)


def test_public_batched_builder_records_its_own_routing(monkeypatch):
    """``build_table_via_duffy_radial_batched`` is public and used directly.

    A table it finished must not report ``unknown``; the routing
    dispatcher is not the only entry point, and the repository's own tests
    call this method straight through.
    """
    import numpy as _np

    import volumential.nearfield_potential_table as npt
    import volumential.opcounters as opcounters
    from sumpy.kernel import LaplaceKernel

    table = npt.NearFieldInteractionTable(
        quad_order=1, dim=2, sumpy_kernel=LaplaceKernel(2),
        progress_bar=False,
    )
    assert table.build_routing is None

    def fake_batched_values(
        queue, invariant_info, local_entry_indices, *args, **kwargs
    ):
        return _np.asarray(local_entry_indices, dtype=table.dtype) + 1

    monkeypatch.setattr(
        table, "_batched_duffy_values_for_local_indices", fake_batched_values
    )
    table.build_table_via_duffy_radial_batched(queue=None)

    assert table.is_built
    assert table.build_routing == "batched"
    assert table.build_fallback_reason is None
    assert opcounters.direct_build_routing(table) == "batched"


def test_a_cache_kwarg_cannot_overwrite_the_recorded_routing(
    ctx_factory, tmp_path, monkeypatch
):
    """The payload owns the provenance; a caller's kwarg does not.

    Cache kwargs are restored onto the table by a generic ``setattr``
    loop, so one named ``build_routing`` would have replaced the routing
    the payload just restored -- and walked past strict mode with it.
    """
    import volumential.nearfield_potential_table as npt
    from volumential.nearfield_potential_table import DUFFY_NO_FALLBACK_ENV_VAR
    from volumential.table_manager import UnverifiedBuildRoutingError

    queue = cl.CommandQueue(cl.Context([ctx_factory().devices[0]]))
    cache_file = tmp_path / "nft-routing-kwarg.sqlite"

    original = npt.NearFieldInteractionTable.\
        build_table_via_duffy_radial_batched

    def failing_batched(self, build_queue, *args, **kwargs):
        raise RuntimeError("synthetic batched build failure")

    npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
        failing_batched
    )
    try:
        with NFTable(str(cache_file), progress_bar=False) as table_manager:
            with pytest.warns(RuntimeWarning, match="falling back to the"):
                table_manager.get_table(
                    2, "Laplace", q_order=1,
                    force_recompute=True, queue=queue,
                    build_routing="batched",
                )
    finally:
        npt.NearFieldInteractionTable.build_table_via_duffy_radial_batched = (
            original
        )

    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        loaded, is_recomputed = table_manager.get_table(
            2, "Laplace", q_order=1, queue=queue, build_routing="batched"
        )
    assert not is_recomputed
    # the payload's routing wins over the caller's kwarg
    assert loaded.build_routing == "scalar-fallback"

    # ... so strict mode still refuses it
    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        with pytest.raises(UnverifiedBuildRoutingError):
            table_manager.get_table(
                2, "Laplace", q_order=1, queue=queue,
                build_routing="batched",
            )


def test_a_refused_batched_build_records_no_routing(ctx_factory, tmp_path,
                                                    monkeypatch):
    """A builder that raised produced nothing, so it claims nothing.

    The dispatcher used to record "batched" before calling the builder,
    so a failure with the fallback refused left that routing on a table
    whose data an earlier build had produced.
    """
    import volumential.nearfield_potential_table as npt
    from volumential.nearfield_potential_table import DUFFY_NO_FALLBACK_ENV_VAR
    from sumpy.kernel import LaplaceKernel

    table = npt.NearFieldInteractionTable(
        quad_order=1, dim=2, sumpy_kernel=LaplaceKernel(2),
        progress_bar=False,
    )

    def fake_batched_values(
        queue, invariant_info, local_entry_indices, *args, **kwargs
    ):
        return np.asarray(local_entry_indices, dtype=table.dtype) + 1

    monkeypatch.setattr(
        table, "_batched_duffy_values_for_local_indices", fake_batched_values
    )
    table.build_table_via_duffy_radial_batched(queue=None)
    assert table.build_routing == "batched"

    # now a rebuild whose batched builder fails, with the fallback refused
    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")

    def failing_batched(*args, **kwargs):
        raise RuntimeError("synthetic batched build failure")

    monkeypatch.setattr(
        table, "build_table_via_duffy_radial_batched", failing_batched
    )
    monkeypatch.setattr(
        table, "_supports_batched_duffy_builder", lambda: True
    )
    with pytest.raises(RuntimeError, match="scalar fallback is refused"):
        table.build_table_via_duffy_radial(queue=object())

    # the earlier successful build's routing is untouched, and no new
    # "batched" claim was recorded for the attempt that produced nothing
    assert table.build_routing == "batched"
    assert table.build_fallback_reason is None


def test_strict_mode_accepts_a_cached_batched_build(
    ctx_factory, tmp_path, monkeypatch
):
    """The guard must not reject the routing strict mode is asking for."""
    from volumential.nearfield_potential_table import DUFFY_NO_FALLBACK_ENV_VAR

    queue = cl.CommandQueue(cl.Context([ctx_factory().devices[0]]))
    cache_file = tmp_path / "nft-strict-batched.sqlite"
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        built, _ = table_manager.get_table(
            2, "Laplace", q_order=1, force_recompute=True, queue=queue
        )
    assert built.build_routing == "batched"

    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")
    with NFTable(str(cache_file), progress_bar=False) as table_manager:
        loaded, is_recomputed = table_manager.get_table(
            2, "Laplace", q_order=1, queue=queue
        )
    assert not is_recomputed
    assert loaded.build_routing == "batched"


def laplace_const_source_same_box(table_2d_order1, queue, q_order, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    n_pairs = nft.n_pairs
    n_q_points = nft.n_q_points
    pot = np.zeros(n_q_points)

    case_same_box = len(nft.interaction_case_vecs) // 2

    for source_mode_index in range(n_q_points):
        for target_point_index in range(n_q_points):
            pair_id = source_mode_index * n_q_points + target_point_index
            entry_id = case_same_box * n_pairs + pair_id
            # print(source_mode_index, target_point_index, pair_id, entry_id,
            # nft.get_entry_data(entry_id))
            pot[target_point_index] += 1.0 * nft.get_entry_data(entry_id)

    return pot


def laplace_cons_source_neighbor_box(table_2d_order1, queue, q_order, case_id, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    n_pairs = nft.n_pairs
    n_q_points = nft.n_q_points
    pot = np.zeros(n_q_points)

    for source_mode_index in range(n_q_points):
        for target_point_index in range(n_q_points):
            pair_id = source_mode_index * n_q_points + target_point_index
            entry_id = case_id * n_pairs + pair_id
            # print(source_mode_index, target_point_index, pair_id, entry_id,
            # nft.get_entry_data(entry_id))
            pot[target_point_index] += 1.0 * nft.get_entry_data(entry_id)

    return pot


def test_lcssb_1(ctx_factory, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    u = laplace_const_source_same_box(table_2d_order1, queue, 1)
    assert len(u) == 1


def interp_func(table_2d_order1, queue, q_order, coef, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    assert dim == 2

    modes = [nft.get_mode(i) for i in range(nft.n_q_points)]

    def func(x, y):
        z = np.zeros(np.array(x).shape)
        for i in range(nft.n_q_points):
            mode = modes[i]
            z += (coef[i] * mode(x, y)).reshape(z.shape)
        return z

    return func


def test_interp_func(longrun, ctx_factory, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    q_order = 3
    coef = np.ones(q_order**2)

    h = 0.1
    xx = yy = np.arange(-1.0, 1.0, h)
    xi, yi = np.meshgrid(xx, yy)
    func = interp_func(table_2d_order1, queue, q_order, coef)

    zi = func(xi, yi)

    assert np.allclose(zi, 1)


def direct_quad(source_func, target_point, dim=2):
    knl_func = vm.nearfield_potential_table.get_laplace(dim)

    def integrand(x, y):
        return source_func(x, y) * knl_func(x - target_point[0], y - target_point[1])

    import volumential.singular_integral_2d as squad

    integral, _ = squad.box_quad(
        func=integrand, a=0, b=1, c=0, d=1, singular_point=target_point, maxiter=1000
    )
    return integral


def drive_test_direct_quad_same_box(table_2d_order1, queue, q_order, dim=2):
    u = laplace_const_source_same_box(table_2d_order1, queue, q_order)
    func = interp_func(table_2d_order1, queue, q_order, u)

    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    def const_one_source_func(x, y):
        return 1

    # print(nft.compute_table_entry(1341))
    # print(nft.compute_table_entry(nft.lookup_by_symmetry(1341)))

    for it in range(nft.n_q_points):
        target = nft.q_points[it]
        v1 = func(target[0], target[1])
        v2 = direct_quad(const_one_source_func, target)
        v3 = 0
        for ids in range(nft.n_q_points):
            mode = nft.get_mode(ids)
            vv = direct_quad(mode, target)
            print(ids, it, vv)
            v3 += vv

        print(target, v1, v2, v3)
        assert np.abs(v1 - v2) < 2e-6
        assert np.abs(v1 - v3) < 1e-6


@pytest.mark.parametrize(
    "q_order",
    [
        1,
    ],
)
def test_direct_quad(q_order, ctx_factory, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    drive_test_direct_quad_same_box(table_2d_order1, queue, q_order)


@pytest.mark.parametrize("q_order", [2, 3, 4, 5])
def test_direct_quad_longrun(longrun, ctx_factory, q_order, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    drive_test_direct_quad_same_box(table_2d_order1, queue, q_order)


def test_case_ids(ctx_factory, table_2d_order1):
    table = table_2d_order1
    for i in range(len(table.interaction_case_vecs)):
        code = table.case_encode(table.interaction_case_vecs[i])
        assert table.case_indices[code] == i


def get_target_point(case_id, target_id, table):
    case_vec = table.interaction_case_vecs[case_id]
    center = np.array([0.5, 0.5]) + np.array(case_vec) * 0.25
    dist = np.max(np.abs(case_vec)) - 2
    if dist == 1:
        scale = 0.5
    elif dist == 2:
        scale = 1
    elif dist == 4:
        scale = 2
    dx = table.q_points[target_id][0] - 0.5
    dy = table.q_points[target_id][1] - 0.5
    target_point = np.array([center[0] + dx * scale, center[1] + dy * scale])
    return target_point


def test_get_neighbor_target_point(ctx_factory, table_2d_order1):
    table = table_2d_order1
    case_same_box = len(table.interaction_case_vecs) // 2
    for cid in range(len(table.interaction_case_vecs)):
        if cid == case_same_box:
            continue
        for tpid in range(table.n_q_points):
            pt = table.find_target_point(tpid, cid)
            pt2 = get_target_point(cid, tpid, table)
        assert np.allclose(pt, pt2)


def laplace_const_source_neighbor_box(table_2d_order1, queue, q_order, case_id, dim=2):
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    n_pairs = nft.n_pairs
    n_q_points = nft.n_q_points
    pot = np.zeros(n_q_points)

    for source_mode_index in range(n_q_points):
        for target_point_index in range(n_q_points):
            pair_id = source_mode_index * n_q_points + target_point_index
            entry_id = case_id * n_pairs + pair_id
            pot[target_point_index] += 1.0 * nft.get_entry_data(entry_id)
    return pot


def drive_test_direct_quad_neighbor_box(
    table_2d_order1, queue, q_order, case_id, dim=2
):
    u = laplace_const_source_neighbor_box(table_2d_order1, queue, q_order, case_id)
    if q_order == 1:
        nft = table_2d_order1
    else:
        nft = get_table(queue, q_order, dim)

    def const_one_source_func(x, y):
        return 1

    for it in range(nft.n_q_points):
        target = nft.find_target_point(it, case_id)
        v1 = u[it]
        v2 = direct_quad(const_one_source_func, target)
        v3 = 0
        for ids in range(nft.n_q_points):
            mode = nft.get_mode(ids)
            vv = direct_quad(mode, target)
            print(ids, it, vv)
            v3 += vv

        print(target, v1, v2, v3)
        assert np.abs(v1 - v2) < 2e-6
        assert np.abs(v1 - v3) < 1e-6


@pytest.mark.parametrize(
    "q_order",
    [
        1,
    ],
)
def test_direct_quad_neighbor_box(ctx_factory, q_order, table_2d_order1):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    table = table_2d_order1
    for case_id in range(len(table.interaction_case_vecs)):
        drive_test_direct_quad_neighbor_box(table_2d_order1, queue, q_order, case_id)


@pytest.mark.parametrize(
    "q_order",
    [
        2,
    ],
)
def test_direct_quad_neighbor_box_longrun(
    longrun, ctx_factory, q_order, table_2d_order1
):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    table = table_2d_order1
    for case_id in range(len(table.interaction_case_vecs)):
        drive_test_direct_quad_neighbor_box(table_2d_order1, queue, q_order, case_id)


# fdm=marker:ft=pyopencl
