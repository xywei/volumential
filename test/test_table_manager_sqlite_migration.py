import pytest

from volumential.table_manager import NearFieldInteractionTableManager as NFTable


def test_legacy_hdf5_cache_error(tmp_path):
    filename = tmp_path / "legacy-cache.hdf5"
    filename.write_bytes(b"\x89HDF\r\n\x1a\nlegacy")

    with pytest.raises(RuntimeError, match="legacy HDF5 format"):
        with NFTable(str(filename), progress_bar=False):
            pass
