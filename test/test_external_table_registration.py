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
