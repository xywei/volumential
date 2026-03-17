import numpy as np
import pytest

from volumential.table_manager import (
    NearFieldInteractionTableManager as NFTable,
    _deserialize_scalar,
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
