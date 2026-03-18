import numpy as np
import pytest
import sqlite3

from volumential.table_manager import (
    NearFieldInteractionTableManager as NFTable,
    TABLE_CACHE_SCHEMA_VERSION,
    _deserialize_table_payload,
    _deserialize_scalar,
    _serialize_table_payload,
    _serialize_scalar,
)


def test_legacy_hdf5_cache_error(tmp_path):
    filename = tmp_path / "legacy-cache.hdf5"
    filename.write_bytes(b"\x89HDF\r\n\x1a\nlegacy")

    with pytest.raises(RuntimeError, match="legacy HDF5 format"):
        with NFTable(str(filename), progress_bar=False):
            pass


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (np.float64(1.25), "float"),
        (np.complex128(1.25 + 2.5j), "complex"),
    ],
)
def test_numpy_scalar_kwarg_roundtrip(value, expected_type):
    value_type, value_text = _serialize_scalar(value)
    assert value_type == expected_type

    restored_value = _deserialize_scalar(value_type, value_text)
    if value_type == "float":
        assert restored_value == float(value)
    elif value_type == "complex":
        assert restored_value == complex(value)
    else:
        raise AssertionError("unexpected scalar type")


def test_read_only_cache_miss_fails_fast(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False):
        pass

    with NFTable(str(filename), read_only=True, progress_bar=False) as table_manager:
        monkeypatch.setattr(
            table_manager,
            "compute_and_update_table",
            lambda *args, **kwargs: pytest.fail(
                "unexpected recompute in read-only mode"
            ),
        )

        with pytest.raises(RuntimeError, match="cache miss in read-only mode"):
            table_manager.get_table(2, "Laplace", q_order=1)


def test_read_only_force_recompute_fails_fast(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False):
        pass

    with NFTable(str(filename), read_only=True, progress_bar=False) as table_manager:
        monkeypatch.setattr(table_manager, "_record_exists", lambda *args: True)
        monkeypatch.setattr(
            table_manager,
            "compute_and_update_table",
            lambda *args, **kwargs: pytest.fail(
                "unexpected recompute in read-only mode"
            ),
        )

        with pytest.raises(RuntimeError, match="force_recompute"):
            table_manager.get_table(2, "Laplace", q_order=1, force_recompute=True)


def test_schema_version_written_for_new_cache(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False):
        pass

    with sqlite3.connect(str(filename)) as db:
        row = db.execute(
            "SELECT value_type, value_text FROM nearfield_cache_meta "
            "WHERE key='schema_version'"
        ).fetchone()

    assert row is not None
    assert row[0] == "str"
    assert row[1] == TABLE_CACHE_SCHEMA_VERSION


def test_incompatible_future_schema_version_rejected(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False):
        pass

    with sqlite3.connect(str(filename)) as db:
        db.execute(
            "UPDATE nearfield_cache_meta SET value_type='str', value_text='999.0.0' "
            "WHERE key='schema_version'"
        )
        db.commit()

    with pytest.raises(RuntimeError, match="incompatible schema version"):
        with NFTable(str(filename), read_only=True, progress_bar=False):
            pass


def test_symmetry_reduced_payload_roundtrip_uses_sparse_arrays():
    class DummyTable:
        pass

    table = DummyTable()
    table.q_points = np.array([[0.5, 0.5], [0.25, 0.75]])
    table.data = np.array([np.nan, 1.5, np.nan, 2.5])
    table.mode_normalizers = np.array([1.0, 2.0])
    table.kernel_exterior_normalizers = np.array([0.0, 0.0])
    table.interaction_case_vecs = [[0, 0], [1, 0]]
    table.case_indices = np.array([0, 1, 2, 3])
    table.table_data_is_symmetry_reduced = True

    blob = _serialize_table_payload(table)
    payload = _deserialize_table_payload(blob)

    assert "reduced_entry_ids" in payload
    assert "reduced_data" in payload
    assert "data" not in payload
    assert np.array_equal(payload["reduced_entry_ids"], np.array([1, 3]))
    assert np.array_equal(payload["reduced_data"], np.array([1.5, 2.5]))


def test_load_saved_table_missing_payload_is_corruption(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(
            table_manager,
            "_load_record",
            lambda *args, **kwargs: {
                "dim": 2,
                "quad_order": 1,
                "payload": None,
            },
        )

        with pytest.raises(KeyError, match="payload"):
            table_manager.load_saved_table(
                2,
                "Laplace",
                q_order=1,
                compute_method="DuffyRadial",
            )


def test_get_table_recomputes_on_payload_corruption(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(table_manager, "_record_exists", lambda *args: True)
        monkeypatch.setattr(
            table_manager,
            "load_saved_table",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("missing payload")),
        )

        sentinel = object()
        recomputed = {"called": False}

        def fake_compute_and_update(*args, **kwargs):
            recomputed["called"] = True
            return sentinel

        monkeypatch.setattr(
            table_manager,
            "compute_and_update_table",
            fake_compute_and_update,
        )

        table, is_recomputed = table_manager.get_table(2, "Laplace", q_order=1)
        assert table is sentinel
        assert is_recomputed
        assert recomputed["called"]
