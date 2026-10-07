"""Tests for the near-field table manager: cache lookup and rebuild, kernel
parameter validation, and agreement of cached table entries with direct
adaptive quadrature.
"""

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
import os
import subprocess
from shutil import copyfile

import numpy as np
import pytest

import pyopencl as cl

import volumential as vm
from volumential.table_manager import (
    NearFieldInteractionTableManager as NFTable,
    TableRequest,
)


logger = logging.getLogger(__name__)


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


def table_for_q_order(table_2d_order1, queue, q_order, dim=2):
    """Reuse the session-cached order-1 table, or build one for `q_order`."""
    if q_order == 1:
        return table_2d_order1
    return get_table(queue, q_order, dim)


def test_case_id(ctx_factory, table_2d_order1):
    table = table_2d_order1
    case_same_box = len(table.interaction_case_vecs) // 2
    assert list(table.interaction_case_vecs[case_same_box]) == [0, 0]


def test_get_table_2d_order1(table_2d_order1):
    table = table_2d_order1
    assert table.dim == 2


@pytest.mark.parametrize("kernel_name", ["Yukawa", "Yukawa-Dx"])
def test_get_table_yukawa_requires_lambda(ctx_factory, tmp_path, kernel_name):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    cache_file = tmp_path / f"nft-{kernel_name.lower()}-requires-lam.sqlite"
    with (
        NFTable(str(cache_file), progress_bar=False) as table_manager,
        pytest.raises(TypeError, match="missing kernel parameter"),
    ):
        table_manager.get_table(
            2,
            kernel_name,
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
    assert built.builder_revision is not None
    assert loaded.builder_revision == built.builder_revision


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


@pytest.mark.parametrize("routing", ["scalar-fallbac", "", "BATCHED", "42"])
def test_strict_mode_refuses_an_unrecognized_cached_routing(
    tmp_path, monkeypatch, routing
):
    """Provenance outside the recognized set is damaged, not verified.

    Only the exact strings "scalar-fallback" and "unknown" used to be
    refused, so a payload whose routing was corrupted to anything else --
    "scalar-fallbac", say -- passed as a vouched-for batched build and
    could enter a strict evidence run.
    """
    from volumential.nearfield_potential_table import (
        DUFFY_BUILD_ROUTINGS,
        DUFFY_NO_FALLBACK_ENV_VAR,
        NearFieldInteractionTable,
    )
    from volumential.table_manager import (
        UnverifiedBuildRoutingError,
        _refuse_unverified_build_routing,
    )

    assert routing not in DUFFY_BUILD_ROUTINGS

    class _Request:
        dim = 2
        kernel_type = "Laplace"
        q_order = 1
        source_box_level = 0

    table = NearFieldInteractionTable.__new__(NearFieldInteractionTable)
    table.build_routing = routing

    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")
    with pytest.raises(UnverifiedBuildRoutingError, match="unrecognized"):
        _refuse_unverified_build_routing(table, _Request)

    # every recognized non-fallback routing still loads
    for good in DUFFY_BUILD_ROUTINGS:
        table.build_routing = good
        if good == "scalar-fallback":
            with pytest.raises(UnverifiedBuildRoutingError, match="fallback"):
                _refuse_unverified_build_routing(table, _Request)
        else:
            _refuse_unverified_build_routing(table, _Request)

    # ... and without strict mode nothing is refused
    monkeypatch.delenv(DUFFY_NO_FALLBACK_ENV_VAR)
    table.build_routing = routing
    _refuse_unverified_build_routing(table, _Request)


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
    assert table.builder_revision is None

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
    assert table.builder_revision == npt.DUFFY_BUILDER_REVISION
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


def _rewrite_cached_payloads(cache_file, edit):
    """Replace the payload of every row of *cache_file* by ``edit(payload)``.

    *edit* takes and returns the deserialized payload, a dict of arrays.
    """
    import sqlite3
    from io import BytesIO

    from volumential.table_manager import _deserialize_table_payload

    conn = sqlite3.connect(str(cache_file))
    try:
        rows = conn.execute(
            "SELECT rowid, payload FROM nearfield_cache"
        ).fetchall()
        for rowid, blob in rows:
            payload = edit(_deserialize_table_payload(blob))
            with BytesIO() as f:
                np.savez(f, **payload)
                conn.execute(
                    "UPDATE nearfield_cache SET payload=? WHERE rowid=?",
                    (f.getvalue(), rowid),
                )
        conn.commit()
    finally:
        conn.close()


def _without_builder_revision(payload):
    """*payload* as a build written before builder revisions were recorded."""
    payload = dict(payload)
    payload.pop("builder_revision", None)
    return payload


def _as_cached_before_the_complex_fix(payload):
    """*payload* as the 2D scalar builder wrote it before #200 fixed #180:
    the real part of the entries, and no builder revision."""
    payload = _without_builder_revision(payload)
    key = "reduced_data" if "reduced_data" in payload else "data"
    payload[key] = payload[key].real.astype(payload[key].dtype)
    return payload


def test_a_complex_2d_scalar_table_cached_before_the_fix_is_rebuilt(tmp_path):
    """A cached complex 2D table that lost its imaginary part is rebuilt (#201).

    Until #200 fixed #180, the 2D scalar DuffyRadial rule cast every
    integrand value to ``float``, so a complex 2D table built by the scalar
    builder held the real part of the right table.  Its cache entry loaded
    cleanly and its routing was an ordinary ``scalar``, so a warm run went on
    using it.
    """
    import sqlite3
    from contextlib import closing

    from sumpy.kernel import HelmholtzKernel

    from volumential.nearfield_potential_table import DUFFY_BUILDER_REVISION

    kwargs = {"sumpy_knl": HelmholtzKernel(2), "k": 1.5}
    cache_file = tmp_path / "nft-helmholtz-before-the-complex-fix.sqlite"

    def manager(**manager_kwargs):
        return NFTable(
            str(cache_file), dtype=np.complex128, progress_bar=False,
            **manager_kwargs,
        )

    def get(**manager_kwargs):
        with manager(**manager_kwargs) as table_manager:
            return table_manager.get_table(2, "Helmholtz", q_order=1, **kwargs)

    def entries(table):
        return np.array(table.get_reduced_table_data()[1])

    # no queue, so the scalar builder
    built, _ = get()
    assert built.build_routing == "scalar"
    assert built.builder_revision == DUFFY_BUILDER_REVISION
    right = entries(built)
    # the part the old rule dropped is not small
    assert np.max(np.abs(right.imag)) > 1e-2

    # A table built after the fix, but cached before the revision was
    # recorded, keeps its imaginary part, and loads as it is.
    _rewrite_cached_payloads(cache_file, _without_builder_revision)
    loaded, is_recomputed = get()
    assert not is_recomputed
    assert loaded.builder_revision is None
    np.testing.assert_array_equal(entries(loaded), right)

    _rewrite_cached_payloads(cache_file, _as_cached_before_the_complex_fix)

    # A cache kwarg of the same name does not vouch for the table: the
    # payload owns the revision.
    with closing(sqlite3.connect(str(cache_file))) as conn, conn:
        conn.execute(
            "INSERT INTO nearfield_cache_kwargs (dim, kernel_type, q_order, "
            "source_box_level, key, value_type, value_text) "
            "VALUES (2, 'Helmholtz', 1, 0, 'builder_revision', 'int', ?)",
            (str(DUFFY_BUILDER_REVISION),),
        )

    # read-only, the table cannot be rebuilt, and is refused, not served
    with pytest.raises(RuntimeError, match="read-only") as refused:
        get(read_only=True)
    assert "only the real part" in str(refused.value.__cause__)

    rebuilt, is_recomputed = get()
    assert is_recomputed
    np.testing.assert_array_equal(entries(rebuilt), right)

    # the rebuilt entry records the revision, so it loads from now on
    loaded, is_recomputed = get()
    assert not is_recomputed
    assert loaded.builder_revision == DUFFY_BUILDER_REVISION
    np.testing.assert_array_equal(entries(loaded), right)


#: A complex 2D scalar build from before the fix: Helmholtz, complex entries
#: whose imaginary part is zero, no builder revision.  Each case below changes
#: one input.
_PRE_FIX_SCALAR_BUILD = {
    "dim": 2,
    "build_method": "DuffyRadial",
    "routing": "scalar",
    "revision": None,
    "values": np.array([1.0, -2.0, 0.5], dtype=np.complex128),
    "layout": "reduced",
    "kernel": "helmholtz",
}


@pytest.mark.parametrize(
    ("change", "stale"),
    [
        ({}, True),
        ({"routing": "scalar-adaptive"}, True),
        ({"routing": "scalar-fallback"}, True),
        # a payload written before routings were recorded
        ({"routing": None}, True),
        # a damaged routing is not a batched one
        ({"routing": "BATCHED"}, True),
        ({"revision": 0}, True),
        # a legacy row records no build method
        ({"build_method": None}, True),
        ({"layout": "dense"}, True),
        # its imaginary part is zero in fact, but nothing in the cache says so
        ({"kernel": "yukawa"}, True),
        # a load without a sumpy kernel cannot tell, nor a kernel that
        # does not say
        ({"kernel": None}, True),
        ({"kernel": "silent"}, True),
        ({"routing": "batched"}, False),
        ({"revision": 1}, False),
        ({"dim": 3}, False),
        ({"build_method": "ExternalAssembly"}, False),
        ({"values": np.array([1.0, -2.0, 0.5])}, False),
        ({"values": np.array([1.0, -2.0 + 1.0e-300j, 0.5])}, False),
        ({"kernel": "laplace"}, False),
        ({"kernel": "laplace-dx"}, False),
    ],
)
def test_which_cached_tables_predate_the_complex_fix(change, stale):
    """Only a table that can hold the old rule's real part is rebuilt."""
    from types import SimpleNamespace

    from sumpy.kernel import (
        AxisTargetDerivative,
        HelmholtzKernel,
        LaplaceKernel,
        YukawaKernel,
    )

    from volumential.table_manager import _stale_complex_2d_scalar_build

    inputs = {**_PRE_FIX_SCALAR_BUILD, **change}
    dim = inputs["dim"]
    kernel = {
        None: None,
        "silent": SimpleNamespace(),
        "helmholtz": HelmholtzKernel(dim),
        "yukawa": YukawaKernel(dim),
        "laplace": LaplaceKernel(dim),
        "laplace-dx": AxisTargetDerivative(0, LaplaceKernel(dim)),
    }[inputs["kernel"]]
    values = inputs["values"]
    if inputs["layout"] == "reduced":
        payload = {
            "reduced_entry_ids": np.arange(len(values)),
            "reduced_data": values,
        }
    else:
        payload = {"data": values}

    table = SimpleNamespace(
        build_routing=inputs["routing"],
        builder_revision=inputs["revision"],
    )
    reason = _stale_complex_2d_scalar_build(
        table,
        TableRequest.from_args(dim, "Helmholtz", 1, 0),
        inputs["build_method"],
        payload,
        kernel,
    )

    if stale:
        assert reason is not None
        assert "only the real part" in reason
        assert "discarding the cached data" in reason
    else:
        assert reason is None


def laplace_const_source_same_box(table_2d_order1, queue, q_order, dim=2):
    nft = table_for_q_order(table_2d_order1, queue, q_order, dim)

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
    nft = table_for_q_order(table_2d_order1, queue, q_order, dim)

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
    nft = table_for_q_order(table_2d_order1, queue, q_order, dim)

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

    nft = table_for_q_order(table_2d_order1, queue, q_order, dim)

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
            logger.info("mode=%s target=%s value=%s", ids, it, vv)
            v3 += vv

        logger.info("target=%s table=%s direct=%s modes=%s", target, v1, v2, v3)
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
    nft = table_for_q_order(table_2d_order1, queue, q_order, dim)

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
    nft = table_for_q_order(table_2d_order1, queue, q_order, dim)

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
            logger.info("mode=%s target=%s value=%s", ids, it, vv)
            v3 += vv

        logger.info("target=%s table=%s direct=%s modes=%s", target, v1, v2, v3)
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
