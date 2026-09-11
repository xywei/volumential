"""Tests for registering externally assembled tables with the table manager.

The registration path (``register_external_table``) stores an offline-built
table (e.g. windowed RKE assembly) under the standard cache slot so the
evaluator consumes it exactly like a direct-built table.  Everything here is
queue-free: the windowed assembler and the manager's load path require no
OpenCL context.
"""

__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

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

import sqlite3

import numpy as np
import pytest

from volumential.table_manager import (
    EXTERNAL_TABLE_BUILD_METHOD,
    NearFieldInteractionTableManager,
)


DIM = 2
Q_ORDER = 2
LEVEL = 2
ROOT_EXTENT = 2.0
WINDOW_THETA = 16.0
BOX_EXTENT = ROOT_EXTENT * 0.5**LEVEL
LAM = 4.0 / BOX_EXTENT  # theta = 4, well inside the declaration


@pytest.fixture(scope="module")
def assembled_yukawa(tmp_path_factory):
    from volumential.rke_table_assembly import (
        assemble_windowed_parameterized_table,
    )

    channel_cache = tmp_path_factory.mktemp("windowed-channels") / "chan.sqlite"
    table, certificate = assemble_windowed_parameterized_table(
        channel_cache,
        DIM,
        "Yukawa",
        Q_ORDER,
        LAM,
        source_box_level=LEVEL,
        root_extent=ROOT_EXTENT,
        window_theta=WINDOW_THETA,
        p_star=4,
    )
    return table, certificate


def _register(manager, table, certificate, **overrides):
    kwargs = {
        "source_box_level": LEVEL,
        "provenance": {
            "kind": "windowed_rke_assembly",
            "window_theta": WINDOW_THETA,
            "condition_number": certificate["condition_number"],
        },
        "lam": LAM,
    }
    kwargs.update(overrides)
    return manager.register_external_table(
        DIM, "Yukawa", Q_ORDER, table, **kwargs
    )


def test_register_and_load_roundtrip_is_exact(tmp_path, assembled_yukawa):
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)
        assert manager.last_register_timings["payload_bytes"] > 0

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        loaded, is_recomputed = manager.get_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )

    assert not is_recomputed
    assert loaded.build_method == EXTERNAL_TABLE_BUILD_METHOD
    assert loaded.table_provenance == "external_assembly"
    assert loaded.provenance_kind == "windowed_rke_assembly"
    assert loaded.table_data_is_symmetry_reduced

    ids0, vals0 = table.get_reduced_table_data()
    ids1, vals1 = loaded.get_reduced_table_data()
    assert np.array_equal(np.asarray(ids0), np.asarray(ids1))
    assert np.array_equal(np.asarray(vals0), np.asarray(vals1))


def test_series_assembly_does_not_inherit_the_channel_build_routing(tmp_path):
    """An assembled table must not claim Duffy quadrature produced its data.

    ``assemble_parameterized_table`` builds its result as
    ``copy.deepcopy(base)`` and then replaces the table data.  ``base`` is
    the Laplace channel table, which comes from
    ``NearFieldInteractionTableManager.get_table`` and therefore *does*
    carry a recorded DuffyRadial routing.  Without clearing it, the
    assembled table -- and the cache entry it is registered as -- would
    report ``batched`` or ``scalar-fallback`` for values no Duffy
    quadrature ever touched.
    """
    import pyopencl as cl

    import volumential.opcounters as opcounters
    from volumential.rke_table_assembly import (
        _get_channel_tables,
        assemble_parameterized_table,
    )
    from volumential.nearfield_potential_table import DUFFY_BUILD_ROUTINGS

    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception as exc:
        pytest.skip(f"no OpenCL context available: {exc}")
    queue = cl.CommandQueue(ctx)

    channel_cache = tmp_path / "series-channels.sqlite"
    table, certificate = assemble_parameterized_table(
        queue,
        channel_cache,
        DIM,
        "Yukawa",
        Q_ORDER,
        LAM,
        source_box_level=LEVEL,
        root_extent=ROOT_EXTENT,
    )

    # the base the assembler deep-copies is a real DuffyRadial build, so
    # there was something to inherit
    base = _get_channel_tables(
        queue, channel_cache, DIM, Q_ORDER, LEVEL, ROOT_EXTENT,
        ["laplace"], None, False,
    )["laplace"]
    assert base.build_routing in DUFFY_BUILD_ROUTINGS

    assert table.build_routing is None
    assert table.build_fallback_reason is None
    assert opcounters.direct_build_routing(table) == "unknown"

    cache = tmp_path / "registered-routing.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(
            manager, table, certificate,
            provenance={"kind": "rke_series_assembly"},
        )

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        loaded, is_recomputed = manager.get_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )

    assert not is_recomputed
    assert loaded.build_routing is None
    assert opcounters.direct_build_routing(loaded) == "unknown"


def test_strict_mode_still_loads_a_registered_external_assembly(
    tmp_path, assembled_yukawa, monkeypatch
):
    """The no-fallback switch must not reject externally assembled tables.

    It governs one substitution: a batched DuffyRadial build silently
    becoming a scalar one.  An assembled table was never a DuffyRadial
    build -- which is why the assemblers clear ``build_routing`` -- so its
    ``unknown`` routing is structural, not a missing record.  Its
    provenance is the ``ExternalAssembly`` build method, the provenance
    kind and the payload checksum the load path verifies.  Rejecting it
    would break the documented ``--include-windowed`` campaign flow, which
    is what sets this switch in the first place.
    """
    from volumential.nearfield_potential_table import DUFFY_NO_FALLBACK_ENV_VAR

    table, certificate = assembled_yukawa
    cache = tmp_path / "registered-strict.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    monkeypatch.setenv(DUFFY_NO_FALLBACK_ENV_VAR, "1")
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        loaded, is_recomputed = manager.get_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )

    assert not is_recomputed
    assert loaded.build_method == EXTERNAL_TABLE_BUILD_METHOD
    assert loaded.build_routing is None

    # the exemption is keyed on the build method, not on the routing being
    # unknown: an ordinary Duffy payload with the same unknown routing is
    # still refused
    from volumential.table_manager import (
        _refuse_unverified_build_routing,
        UnverifiedBuildRoutingError,
    )
    from volumential.table_manager import TableRequest

    request = TableRequest.from_args(DIM, "Yukawa", Q_ORDER, LEVEL)
    with pytest.raises(UnverifiedBuildRoutingError):
        _refuse_unverified_build_routing(loaded, request, "DuffyRadial")


def test_windowed_assembly_clears_the_build_routing_too(assembled_yukawa):
    """The windowed base is a skeleton table, so it has nothing to inherit
    today; the assembled result must still report no DuffyRadial routing so
    a future change of base cannot leak one."""
    import volumential.opcounters as opcounters

    table, _ = assembled_yukawa
    assert table.build_routing is None
    assert table.build_fallback_reason is None
    assert opcounters.direct_build_routing(table) == "unknown"


def test_load_misses_on_parameter_mismatch(tmp_path, assembled_yukawa):
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)
        with pytest.raises(KeyError, match="lam"):
            manager.load_saved_table(
                DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=1.5 * LAM
            )


def test_load_rejects_tampered_payload_checksum(tmp_path, assembled_yukawa):
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    conn = sqlite3.connect(str(cache))
    conn.execute(
        "UPDATE nearfield_cache_kwargs SET value_text='deadbeef' "
        "WHERE key='external_payload_checksum'"
    )
    conn.commit()
    conn.close()

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(KeyError, match="checksum"):
        manager.load_saved_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )


@pytest.mark.parametrize("corrupted", [
    "case_indices",
    "interaction_case_vecs",
    "q_points",
    "mode_normalizers",
    "kernel_exterior_normalizers",
])
def test_load_rejects_corrupted_payload_metadata(
        tmp_path, assembled_yukawa, corrupted):
    """The checksum covers the whole payload, not only entries and values.

    ``case_indices`` and ``interaction_case_vecs`` decide which interaction
    case an entry is read under, and ``q_points`` and the normalizers fix
    the basis geometry, so a same-shape corruption in any of them produces
    wrong potentials from entry data that is itself intact.
    """
    from io import BytesIO

    from volumential.table_manager import (
        _deserialize_table_payload,
        _external_payload_checksum,
    )

    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    conn = sqlite3.connect(str(cache))
    (blob,) = conn.execute("SELECT payload FROM nearfield_cache").fetchone()
    payload = _deserialize_table_payload(blob)
    before = _external_payload_checksum(payload)

    # one element, same shape, same dtype -- and the entry ids and values
    # are untouched
    array = np.array(payload[corrupted])
    flat = array.reshape(-1)
    flat[0] = flat[0] + array.dtype.type(1)
    payload[corrupted] = flat.reshape(array.shape)
    assert _external_payload_checksum(payload) != before

    with BytesIO() as f:
        np.savez(f, **payload)
        conn.execute(
            "UPDATE nearfield_cache SET payload=?", (f.getvalue(),)
        )
    conn.commit()
    conn.close()

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(KeyError, match="checksum"):
        manager.load_saved_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )


@pytest.mark.parametrize("column", [
    "case_encoding_base",
    "case_encoding_shift",
])
def test_load_rejects_corrupted_record_columns(
        tmp_path, assembled_yukawa, column):
    """The case encoding lives in the record, not the payload.

    The loader rebuilds ``table.case_encode`` from ``case_encoding_base``
    and ``case_encoding_shift``, and symmetry-reduced reconstruction maps
    interaction cases through it, so an in-range mutation there
    reconstructs entries from the wrong cases while the payload digest
    still verifies.

    (``n_q_points`` and ``n_pairs`` are in the digest too, but a
    pre-existing assertion in the payload loader catches those first, so
    they are not parametrized here.)
    """
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    conn = sqlite3.connect(str(cache))
    conn.execute(f"UPDATE nearfield_cache SET {column}={column}+1")
    conn.commit()
    conn.close()

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(KeyError, match="checksum"):
        manager.load_saved_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )


def test_a_helmholtz_table_registers_and_reloads(tmp_path):
    """The cache kwargs of a Helmholtz registration carry a live
    ``sumpy_knl`` kernel object, which is reconstruction-only and never
    stored.  The registration digest has to skip it exactly as the kwargs
    writer does, or every windowed Helmholtz row fails to provision.
    """
    from sumpy.kernel import HelmholtzKernel

    from volumential.rke_table_assembly import (
        assemble_windowed_parameterized_table,
    )

    channel_cache = tmp_path / "chan.sqlite"
    helmholtz_k = 4.0 / BOX_EXTENT
    table, certificate = assemble_windowed_parameterized_table(
        channel_cache,
        DIM,
        "Helmholtz",
        Q_ORDER,
        helmholtz_k,
        source_box_level=LEVEL,
        root_extent=ROOT_EXTENT,
        window_theta=WINDOW_THETA,
        p_star=4,
    )

    knl = HelmholtzKernel(DIM)
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT, dtype=np.complex128,
    ) as manager:
        manager.register_external_table(
            DIM, "Helmholtz-Reference", Q_ORDER, table,
            source_box_level=LEVEL,
            provenance={
                "kind": "windowed_rke_assembly",
                "condition_number": certificate["condition_number"],
            },
            sumpy_knl=knl,
            **{knl.helmholtz_k_name: helmholtz_k},
        )

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT, dtype=np.complex128,
    ) as manager:
        loaded = manager.load_saved_table(
            DIM, "Helmholtz-Reference", Q_ORDER,
            source_box_level=LEVEL,
            sumpy_knl=knl,
            **{knl.helmholtz_k_name: helmholtz_k},
        )

    assert loaded.build_method == EXTERNAL_TABLE_BUILD_METHOD
    entry_ids, values = table.get_reduced_table_data()
    loaded_ids, loaded_values = loaded.get_reduced_table_data()
    assert np.array_equal(np.asarray(loaded_ids), np.asarray(entry_ids))
    assert np.array_equal(np.asarray(loaded_values), np.asarray(values))


def test_a_failed_kwargs_write_leaves_the_previous_entry_intact(
        tmp_path, assembled_yukawa, monkeypatch):
    """The record and its kwargs are one entry.

    ``__exit__`` commits unconditionally, so a failure between the record
    upsert and the kwargs write used to commit the new payload with the
    old (or no) checksum rows -- destroying a valid cached entry and
    reloading as corruption.  Both writes are scoped by a SAVEPOINT.
    """
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    conn = sqlite3.connect(str(cache))
    (before,) = conn.execute("SELECT payload FROM nearfield_cache").fetchone()
    (checksum_before,) = conn.execute(
        "SELECT value_text FROM nearfield_cache_kwargs "
        "WHERE key='external_payload_checksum'"
    ).fetchone()
    conn.close()

    def _explode(self, table_request, cache_kwargs):
        # fail the way the finding describes: after the writer's opening
        # DELETE, so the entry is left with its checksum rows gone
        self.datafile.execute(
            "DELETE FROM nearfield_cache_kwargs WHERE dim=? AND "
            "kernel_type=? AND q_order=? AND source_box_level=?",
            (
                table_request.dim,
                table_request.kernel_type,
                table_request.q_order,
                table_request.source_box_level,
            ),
        )
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(
        NearFieldInteractionTableManager, "_store_record_kwargs", _explode
    )

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(sqlite3.OperationalError):
        _register(manager, table, certificate, lam=1.5 * LAM)

    conn = sqlite3.connect(str(cache))
    (after,) = conn.execute("SELECT payload FROM nearfield_cache").fetchone()
    (checksum_after,) = conn.execute(
        "SELECT value_text FROM nearfield_cache_kwargs "
        "WHERE key='external_payload_checksum'"
    ).fetchone()
    conn.close()
    assert after == before
    assert checksum_after == checksum_before

    # ... and the entry the slot held is still loadable
    monkeypatch.undo()
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        loaded = manager.load_saved_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )
    assert loaded.build_method == EXTERNAL_TABLE_BUILD_METHOD


def test_a_complex_table_refuses_to_load_through_a_float_manager(tmp_path):
    """The registration dtype check only proves the payload fitted the
    manager that wrote it.  Reopening a complex Helmholtz table through a
    default float manager used to cast every imaginary part away behind a
    ComplexWarning and serve wrong potentials from a payload that
    checksums perfectly.
    """
    from sumpy.kernel import HelmholtzKernel

    from volumential.rke_table_assembly import (
        assemble_windowed_parameterized_table,
    )

    helmholtz_k = 4.0 / BOX_EXTENT
    table, certificate = assemble_windowed_parameterized_table(
        tmp_path / "chan.sqlite",
        DIM,
        "Helmholtz",
        Q_ORDER,
        helmholtz_k,
        source_box_level=LEVEL,
        root_extent=ROOT_EXTENT,
        window_theta=WINDOW_THETA,
        p_star=4,
    )
    _entry_ids, values = table.get_reduced_table_data()
    assert np.asarray(values).dtype.kind == "c"
    assert np.any(np.asarray(values).imag != 0.0)

    knl = HelmholtzKernel(DIM)
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT, dtype=np.complex128,
    ) as manager:
        manager.register_external_table(
            DIM, "Helmholtz-Reference", Q_ORDER, table,
            source_box_level=LEVEL,
            provenance={"condition_number": certificate["condition_number"]},
            sumpy_knl=knl,
            **{knl.helmholtz_k_name: helmholtz_k},
        )

    # the default manager dtype is float64
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT,
    ) as manager, pytest.raises(KeyError, match="cannot be safely"):
        manager.load_saved_table(
            DIM, "Helmholtz-Reference", Q_ORDER,
            source_box_level=LEVEL,
            sumpy_knl=knl,
            **{knl.helmholtz_k_name: helmholtz_k},
        )

    # ... while the widening direction stays allowed
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT, dtype=np.complex128,
    ) as manager:
        loaded = manager.load_saved_table(
            DIM, "Helmholtz-Reference", Q_ORDER,
            source_box_level=LEVEL,
            sumpy_knl=knl,
            **{knl.helmholtz_k_name: helmholtz_k},
        )
    assert np.array_equal(
        np.asarray(loaded.get_reduced_table_data()[1]), np.asarray(values)
    )


@pytest.mark.parametrize("array_name", [
    "mode_normalizers",
    "kernel_exterior_normalizers",
])
def test_register_refuses_an_unsafe_auxiliary_dtype(
        tmp_path, assembled_yukawa, array_name):
    """The auxiliary arrays are reconstructed into dtype-constrained
    buffers too, so a complex normalizer on an otherwise float table
    would register, checksum cleanly, and lose its imaginary part on
    every reload behind a ComplexWarning.
    """
    import copy

    table, certificate = assembled_yukawa
    poisoned = copy.deepcopy(table)
    values = np.asarray(getattr(poisoned, array_name), dtype=np.complex128)
    values[0] = values[0] + 1j
    setattr(poisoned, array_name, values)

    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(
        ValueError, match=f"table {array_name} dtype"
    ):
        _register(manager, poisoned, certificate)


def test_the_load_side_dtype_guard_covers_every_reconstructed_array():
    """The loader assigns the entry values *and* the two normalizer arrays
    into buffers allocated at the manager's dtype, so the guard has to
    cover all three -- not only the entries."""
    from volumential.table_manager import _unsafe_payload_dtype

    payload = {
        "reduced_entry_ids": np.arange(4, dtype=np.int64),
        "reduced_data": np.zeros(4, dtype=np.float64),
        "mode_normalizers": np.zeros(4, dtype=np.float64),
        "kernel_exterior_normalizers": np.zeros(4, dtype=np.float64),
    }
    assert _unsafe_payload_dtype(payload, np.float64) is None
    assert _unsafe_payload_dtype(payload, np.complex128) is None

    for name, label in (
        ("reduced_data", "entry data"),
        ("mode_normalizers", "mode_normalizers"),
        ("kernel_exterior_normalizers", "kernel_exterior_normalizers"),
    ):
        widened = dict(payload)
        widened[name] = np.zeros(4, dtype=np.complex128)
        unsafe = _unsafe_payload_dtype(widened, np.float64)
        assert unsafe is not None
        assert unsafe[0] == label
        assert unsafe[1] == np.dtype(np.complex128)
        # ... and a complex manager accepts the same payload
        assert _unsafe_payload_dtype(widened, np.complex128) is None


def _record_fields(**overrides):
    fields = {
        "n_q_points": 4,
        "quad_order": 2,
        "n_pairs": 16,
        "source_box_extent": BOX_EXTENT,
        "case_encoding_base": 3,
        "case_encoding_shift": 1,
        "build_method": EXTERNAL_TABLE_BUILD_METHOD,
        "kernel_type_cached": "inverse_distance",
    }
    fields.update(overrides)
    return fields


def test_load_rejects_a_retargeted_kernel_parameter(
        tmp_path, assembled_yukawa):
    """The digest binds the payload to the identity it was registered
    under, not just to its own numbers.

    The kernel parameters live in ``nearfield_cache_kwargs``, outside the
    payload, and the loader compares them against the *request*: corrupt
    a stored ``lam`` from A to B and a request for B matched, the payload
    still verified, and the table assembled for A was evaluated as B.
    """
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    other_lam = 1.5 * LAM
    conn = sqlite3.connect(str(cache))
    conn.execute(
        "UPDATE nearfield_cache_kwargs SET value_text=? WHERE key='lam'",
        (repr(float(other_lam)),),
    )
    conn.commit()
    conn.close()

    # the request now agrees with the (corrupted) stored parameter, so the
    # parameter comparison is satisfied and only the digest can catch it
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(KeyError, match="checksum"):
        manager.load_saved_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=other_lam
        )


def test_load_rejects_tampered_provenance(tmp_path, assembled_yukawa):
    """Recorded provenance is part of the registered identity too."""
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    conn = sqlite3.connect(str(cache))
    conn.execute(
        "UPDATE nearfield_cache_kwargs SET value_text=? "
        "WHERE key='provenance_window_theta'",
        (repr(2.0 * WINDOW_THETA),),
    )
    conn.commit()
    conn.close()

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(KeyError, match="checksum"):
        manager.load_saved_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )


def test_identity_checksum_round_trips_the_stored_representation():
    """The digest is computed at registration from live values and at load
    from values deserialized out of SQLite; the two must agree by
    construction, or every honest load would fail.
    """
    from volumential.table_manager import (
        _deserialize_scalar,
        _external_identity_checksum,
        _serialize_scalar,
    )

    class _Request:
        dim = 2
        kernel_type = "Yukawa"
        q_order = 2
        source_box_level = 2

    record = _record_fields()
    live = {
        "lam": 8.0 / 3.0,
        "provenance_p_star": 4,
        "provenance_kind": "windowed_rke_assembly",
        "provenance_cold": True,
        "provenance_pole": complex(1.5, -0.25),
        # excluded from the digest: it is where the digest itself is stored
        "external_payload_checksum": "whatever",
    }
    stored = {
        key: _deserialize_scalar(*_serialize_scalar(value))
        for key, value in live.items()
    }
    assert _external_identity_checksum(_Request, stored, record) == (
        _external_identity_checksum(_Request, live, record)
    )

    # ... and every identifying field actually enters it
    for key, changed in (
        ("lam", 8.0 / 3.0 + 1.0e-15),
        ("provenance_p_star", 5),
        ("provenance_kind", "something_else"),
        ("provenance_cold", False),
        ("provenance_pole", complex(1.5, 0.25)),
    ):
        tampered = dict(stored)
        tampered[key] = changed
        assert _external_identity_checksum(_Request, tampered, record) != (
            _external_identity_checksum(_Request, stored, record)
        )

    # ... and so does every checksummed record column
    for name in ("case_encoding_base", "case_encoding_shift", "n_q_points",
                 "quad_order", "n_pairs", "source_box_extent",
                 "build_method", "kernel_type_cached"):
        tampered_record = dict(record)
        value = tampered_record[name]
        tampered_record[name] = (
            f"{value}-x" if isinstance(value, str) else value + 1
        )
        assert _external_identity_checksum(
            _Request, stored, tampered_record
        ) != _external_identity_checksum(_Request, stored, record)

    # a reconstruction-only object among the cache kwargs is skipped, not
    # hashed and not a TypeError: _store_record_kwargs drops it too, so
    # the loader never sees it
    from sumpy.kernel import HelmholtzKernel

    with_kernel = dict(stored)
    with_kernel["sumpy_knl"] = HelmholtzKernel(2)
    with_kernel["nothing"] = None
    assert _external_identity_checksum(_Request, with_kernel, record) == (
        _external_identity_checksum(_Request, stored, record)
    )

    # the checksum slot itself is not hashed, and neither is a missing key
    # confused with an empty one
    without = {k: v for k, v in stored.items()
               if k != "external_payload_checksum"}
    assert _external_identity_checksum(_Request, without, record) == (
        _external_identity_checksum(_Request, stored, record)
    )


def test_payload_checksum_covers_every_array():
    """A renamed, added or dropped array changes the digest too."""
    from volumential.table_manager import _external_payload_checksum

    base = {
        "reduced_entry_ids": np.arange(4, dtype=np.int64),
        "reduced_data": np.linspace(0.0, 1.0, 4),
        "case_indices": np.array([0, 1, 2, 3], dtype=np.int64),
    }
    digest = _external_payload_checksum(base)

    # order of insertion does not matter: keys are hashed sorted
    assert _external_payload_checksum(
        dict(reversed(list(base.items())))
    ) == digest

    dropped = {k: v for k, v in base.items() if k != "case_indices"}
    assert _external_payload_checksum(dropped) != digest

    renamed = dict(dropped)
    renamed["case_index"] = base["case_indices"]
    assert _external_payload_checksum(renamed) != digest

    added = dict(base)
    added["mode_normalizers"] = np.ones(4)
    assert _external_payload_checksum(added) != digest

    # ... and so does a dtype change that leaves the values equal
    retyped = dict(base)
    retyped["case_indices"] = base["case_indices"].astype(np.int32)
    assert _external_payload_checksum(retyped) != digest


def test_register_validates_geometry(tmp_path, assembled_yukawa):
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        with pytest.raises(ValueError, match="extent"):
            _register(manager, table, certificate,
                      source_box_level=LEVEL + 1)
        with pytest.raises(ValueError, match="quadrature order"):
            manager.register_external_table(
                DIM, "Yukawa", Q_ORDER + 1, table,
                source_box_level=LEVEL, lam=LAM,
            )

    with NearFieldInteractionTableManager(
        str(tmp_path / "other-root.sqlite"), root_extent=0.5 * ROOT_EXTENT
    ) as manager, pytest.raises(ValueError, match="extent"):
        _register(manager, table, certificate)


def test_register_requires_kernel_parameter(tmp_path, assembled_yukawa):
    table, _certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(TypeError, match="lam"):
        manager.register_external_table(
            DIM, "Yukawa", Q_ORDER, table, source_box_level=LEVEL
        )


def test_register_rejects_non_scalar_provenance(tmp_path, assembled_yukawa):
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(TypeError):
        _register(
            manager, table, certificate,
            provenance={"bad": np.zeros(3)},
        )


def test_register_rejects_non_finite_payload(tmp_path, assembled_yukawa):
    import copy

    table, certificate = assembled_yukawa
    poisoned = copy.deepcopy(table)
    entry_ids, values = poisoned.get_reduced_table_data()
    values = np.array(values, copy=True)
    values[0] = np.nan
    poisoned.set_reduced_table_data(entry_ids, values)
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(ValueError, match="non-finite"):
        _register(manager, poisoned, certificate)


@pytest.mark.parametrize("poisoned_array", [
    "mode_normalizers",
    "kernel_exterior_normalizers",
    "q_points",
])
def test_register_rejects_a_non_finite_auxiliary_array(
        tmp_path, assembled_yukawa, poisoned_array):
    """Finiteness is checked across the payload, not only the entries.

    The List 1 evaluator also consumes ``q_points`` and the two
    normalizer arrays, so a table with finite entries and one nan
    normalizer used to register, checksum cleanly, and then produce
    non-finite solves on every reload with no build step left to catch
    it.
    """
    import copy

    table, certificate = assembled_yukawa
    poisoned = copy.deepcopy(table)
    array = np.array(getattr(poisoned, poisoned_array), dtype=np.float64)
    flat = array.reshape(-1)
    flat[0] = np.nan
    setattr(poisoned, poisoned_array, flat.reshape(array.shape))

    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(
        ValueError, match=f"non-finite values in '{poisoned_array}'"
    ):
        _register(manager, poisoned, certificate)


def test_load_rejects_an_external_record_without_a_checksum(
        tmp_path, assembled_yukawa):
    """Every external registration writes the checksum row, so its
    absence on an ExternalAssembly record is a damaged cache -- not a
    legacy record to accept unchecked, which is the hole the checksum
    exists to close.
    """
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)

    conn = sqlite3.connect(str(cache))
    conn.execute(
        "DELETE FROM nearfield_cache_kwargs "
        "WHERE key='external_payload_checksum'"
    )
    conn.commit()
    conn.close()

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(KeyError, match="missing its payload"):
        manager.load_saved_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
        )


def test_register_rejects_unsafe_dtype_narrowing(tmp_path):
    from volumential.rke_table_assembly import (
        assemble_windowed_parameterized_table,
    )

    channel_cache = tmp_path / "chan.sqlite"
    complex_table, _ = assemble_windowed_parameterized_table(
        channel_cache,
        DIM,
        "Helmholtz",
        Q_ORDER,
        LAM,
        source_box_level=LEVEL,
        root_extent=ROOT_EXTENT,
        window_theta=WINDOW_THETA,
        p_star=4,
    )
    cache = tmp_path / "registered.sqlite"
    # default manager dtype is float64: complex payload must be refused
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager, pytest.raises(ValueError, match="dtype"):
        manager.register_external_table(
            DIM, "Yukawa", Q_ORDER, complex_table,
            source_box_level=LEVEL, lam=LAM,
        )


def test_register_refused_in_read_only_mode(tmp_path, assembled_yukawa):
    table, certificate = assembled_yukawa
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT, read_only=True
    ) as manager, pytest.raises(RuntimeError, match="read-only"):
        _register(manager, table, certificate)


def test_registered_slot_can_be_overwritten(tmp_path, assembled_yukawa):
    """The (dim, kernel, q, level) slot is reused across parameters (the
    parameter lives in the stored kwargs), as the KS continuation does."""
    from volumential.rke_table_assembly import (
        assemble_windowed_parameterized_table,
    )

    table, certificate = assembled_yukawa
    channel_cache = tmp_path / "chan.sqlite"
    other_lam = 0.5 * LAM
    other_table, other_certificate = assemble_windowed_parameterized_table(
        channel_cache,
        DIM,
        "Yukawa",
        Q_ORDER,
        other_lam,
        source_box_level=LEVEL,
        root_extent=ROOT_EXTENT,
        window_theta=WINDOW_THETA,
        p_star=4,
    )
    cache = tmp_path / "registered.sqlite"
    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        _register(manager, table, certificate)
        _register(manager, other_table, other_certificate, lam=other_lam)

    with NearFieldInteractionTableManager(
        str(cache), root_extent=ROOT_EXTENT
    ) as manager:
        loaded, is_recomputed = manager.get_table(
            DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=other_lam
        )
        assert not is_recomputed
        _, expected = other_table.get_reduced_table_data()
        _, got = loaded.get_reduced_table_data()
        assert np.array_equal(np.asarray(expected), np.asarray(got))
        # and the overwritten parameter now misses
        with pytest.raises(KeyError, match="lam"):
            manager.load_saved_table(
                DIM, "Yukawa", Q_ORDER, source_box_level=LEVEL, lam=LAM
            )
