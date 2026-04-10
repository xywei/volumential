import numpy as np
import pytest
import sqlite3

from volumential.table_manager import (
    ConstantKernel,
    KernelSpec,
    NearFieldInteractionTableManager as NFTable,
    TableDiscretization,
    TableRequest,
    TABLE_CACHE_SCHEMA_VERSION,
    _coerce_sqlite_int,
    _deserialize_table_payload,
    _deserialize_scalar,
    _serialize_table_payload,
    _serialize_scalar,
)


def _insert_dummy_cache_row(db, payload_blob=None, build_method="DuffyRadial"):
    columns = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in db.execute("PRAGMA table_info(nearfield_cache)")
    }

    if "q_points" in columns:
        db.execute(
            """
            INSERT INTO nearfield_cache (
                dim, kernel_type, q_order, source_box_level,
                n_q_points, quad_order, n_pairs,
                source_box_extent, source_box_level_stored,
                case_encoding_base, case_encoding_shift,
                build_method, kernel_type_cached,
                q_points, data, mode_normalizers,
                kernel_exterior_normalizers, interaction_case_vecs,
                case_indices, payload
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                1,
                "Constant",
                1,
                0,
                1,
                1,
                1,
                1.0,
                0,
                1,
                0,
                build_method,
                "const",
                b"",
                b"",
                None,
                None,
                None,
                None,
                payload_blob,
            ),
        )
    else:
        db.execute(
            """
            INSERT INTO nearfield_cache (
                dim, kernel_type, q_order, source_box_level,
                n_q_points, quad_order, n_pairs,
                source_box_extent, source_box_level_stored,
                case_encoding_base, case_encoding_shift,
                build_method, kernel_type_cached, payload
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                1,
                "Constant",
                1,
                0,
                1,
                1,
                1,
                1.0,
                0,
                1,
                0,
                build_method,
                "const",
                payload_blob,
            ),
        )


def test_manager_rebuilds_legacy_blob_columns_cache_with_backup(tmp_path):
    filename = tmp_path / "cache.sqlite"

    db = sqlite3.connect(str(filename))
    db.execute(
        """
        CREATE TABLE nearfield_cache (
            dim INTEGER NOT NULL,
            kernel_type TEXT NOT NULL,
            q_order INTEGER NOT NULL,
            source_box_level INTEGER NOT NULL,
            n_q_points INTEGER NOT NULL,
            quad_order INTEGER NOT NULL,
            n_pairs INTEGER NOT NULL,
            source_box_extent REAL NOT NULL,
            source_box_level_stored INTEGER NOT NULL,
            case_encoding_base INTEGER,
            case_encoding_shift INTEGER,
            build_method TEXT,
            kernel_type_cached TEXT,
            q_points BLOB NOT NULL,
            data BLOB NOT NULL,
            mode_normalizers BLOB,
            kernel_exterior_normalizers BLOB,
            interaction_case_vecs BLOB,
            case_indices BLOB,
            payload BLOB,
            PRIMARY KEY (dim, kernel_type, q_order, source_box_level)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE nearfield_cache_kwargs (
            dim INTEGER NOT NULL,
            kernel_type TEXT NOT NULL,
            q_order INTEGER NOT NULL,
            source_box_level INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_type TEXT NOT NULL,
            value_text TEXT NOT NULL,
            PRIMARY KEY (dim, kernel_type, q_order, source_box_level, key)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE nearfield_cache_meta (
            key TEXT PRIMARY KEY,
            value_type TEXT NOT NULL,
            value_text TEXT NOT NULL
        )
        """
    )
    db.execute(
        "INSERT INTO nearfield_cache_meta (key, value_type, value_text) VALUES (?, ?, ?)",
        ("schema_version", "str", "2.0.0"),
    )
    _insert_dummy_cache_row(db, payload_blob=b"x")
    db.commit()
    db.close()

    with NFTable(str(filename), progress_bar=False):
        pass

    assert filename.exists()
    assert (tmp_path / "cache.sqlite.bak").exists()


def test_table_request_is_composed_from_stable_specs():
    request = TableRequest.from_args(3, "Laplace", 4, source_box_level=2)

    assert request.kernel == KernelSpec(dim=3, kernel_type="Laplace")
    assert request.discretization == TableDiscretization(q_order=4, source_box_level=2)
    assert request.dim == 3
    assert request.kernel_type == "Laplace"
    assert request.q_order == 4
    assert request.source_box_level == 2


def test_get_kernel_function_uses_sumpy_route_for_laplace3d(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        knl_func = table_manager.get_kernel_function(3, "Laplace")

    value = float(knl_func(1.0, 0.0, 0.0))
    assert np.isclose(value, 1.0 / (4.0 * np.pi))


def test_resolve_kernel_bundle_skips_python_kernel_lookup_for_sumpy(
    tmp_path, monkeypatch
):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(
            table_manager,
            "get_kernel_function",
            lambda *args, **kwargs: pytest.fail(
                "_resolve_kernel_bundle unexpectedly called get_kernel_function"
            ),
        )

        bundle = table_manager._resolve_kernel_bundle(
            TableRequest.from_args(3, "Laplace", 1),
            {},
            require_sumpy_kernel=False,
        )

    assert bundle.sumpy_kernel is not None
    assert bundle.kernel_func is None


def test_load_saved_table_uses_payload_without_python_kernel_lookup(
    tmp_path, monkeypatch
):
    import volumential.table_manager as table_manager_module

    filename = tmp_path / "cache.sqlite"

    class PayloadBackedTable:
        def __init__(
            self,
            quad_order,
            dim,
            kernel_func,
            kernel_type,
            sumpy_kernel,
            source_box_extent,
            precomputed_q_points=None,
            **kwargs,
        ):
            self.quad_order = quad_order
            self.dim = dim
            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel
            self.source_box_extent = source_box_extent

            self.n_q_points = 1
            self.n_pairs = 1
            self.q_points = np.zeros((1, dim), dtype=np.float64)
            if precomputed_q_points is not None:
                self.q_points[:] = precomputed_q_points

            self.data = np.zeros(1, dtype=np.float64)
            self.mode_normalizers = np.zeros(1, dtype=np.float64)
            self.kernel_exterior_normalizers = np.zeros(1, dtype=np.float64)
            self.interaction_case_vecs = [[0] * dim]
            self.case_indices = np.zeros(1, dtype=np.int32)
            self.table_data_is_symmetry_reduced = False

    from types import SimpleNamespace

    payload_table = SimpleNamespace(
        q_points=np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
        data=np.array([3.5], dtype=np.float64),
        mode_normalizers=np.array([1.0], dtype=np.float64),
        kernel_exterior_normalizers=np.array([0.0], dtype=np.float64),
        interaction_case_vecs=[[0, 0, 0]],
        case_indices=np.array([0], dtype=np.int32),
        table_data_is_symmetry_reduced=False,
    )
    payload_blob = _serialize_table_payload(payload_table)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(
            table_manager,
            "get_kernel_function",
            lambda *args, **kwargs: pytest.fail(
                "load_saved_table unexpectedly called get_kernel_function"
            ),
        )
        monkeypatch.setattr(
            table_manager_module,
            "NearFieldInteractionTable",
            PayloadBackedTable,
        )
        monkeypatch.setattr(
            table_manager,
            "_load_record",
            lambda *args, **kwargs: {
                "dim": 3,
                "quad_order": 1,
                "build_method": "DuffyRadial",
                "payload": payload_blob,
                "source_box_extent": 1.0,
                "source_box_level_stored": 0,
                "n_q_points": 1,
                "n_pairs": 1,
                "case_encoding_base": 1,
                "case_encoding_shift": 0,
                "kernel_type_cached": "inv_power",
            },
        )
        monkeypatch.setattr(
            table_manager, "_load_record_kwargs", lambda *args, **kwargs: {}
        )

        loaded = table_manager.load_saved_table(3, "Laplace", q_order=1)

    assert loaded.kernel_func is None
    assert np.allclose(loaded.q_points, payload_table.q_points)
    assert np.allclose(loaded.data, payload_table.data)


def test_get_table_recompute_skips_python_kernel_lookup_for_sumpy(
    tmp_path, monkeypatch
):
    import volumential.table_manager as table_manager_module

    filename = tmp_path / "cache.sqlite"

    class RecomputeTable:
        def __init__(
            self,
            quad_order,
            dim,
            kernel_func,
            kernel_type,
            sumpy_kernel,
            source_box_extent,
            **kwargs,
        ):
            self.quad_order = quad_order
            self.dim = dim
            self.kernel_func = kernel_func
            self.kernel_type = kernel_type
            self.integral_knl = sumpy_kernel
            self.source_box_extent = source_box_extent

            self.n_q_points = 1
            self.n_pairs = 1
            self.q_points = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
            self.data = np.array([1.25], dtype=np.float64)
            self.mode_normalizers = np.array([0.0], dtype=np.float64)
            self.kernel_exterior_normalizers = np.array([0.0], dtype=np.float64)
            self.interaction_case_vecs = [[0, 0, 0]]
            self.case_indices = np.array([0], dtype=np.int32)
            self.table_data_is_symmetry_reduced = False
            self.is_built = False

        def build_table(self, *args, **kwargs):
            self.is_built = True

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(
            table_manager,
            "get_kernel_function",
            lambda *args, **kwargs: pytest.fail(
                "get_table recompute unexpectedly called get_kernel_function"
            ),
        )
        monkeypatch.setattr(
            table_manager_module,
            "NearFieldInteractionTable",
            RecomputeTable,
        )

        table, is_recomputed = table_manager.get_table(3, "Laplace", q_order=1)

    assert is_recomputed
    assert table.kernel_func is None


def test_get_table_rejects_legacy_knl_func_kwarg(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        with pytest.raises(TypeError, match="knl_func has been removed"):
            table_manager.get_table(
                2,
                "Laplace",
                q_order=1,
                knl_func=lambda x, y: x + y,
            )


def test_coerce_sqlite_int_accepts_int32_blob():
    raw = np.array([-37], dtype=np.int32).tobytes()
    assert _coerce_sqlite_int(raw, field_name="case_encoding_shift") == -37


def test_get_kernel_function_preserves_cahn_hilliard_coefficients(tmp_path):
    filename = tmp_path / "cache.sqlite"

    b = 3.5
    c = 2.0

    with NFTable(str(filename), progress_bar=False) as table_manager:
        knl_func = table_manager.get_kernel_function(2, "Cahn-Hilliard", b=b, c=c)

    import volumential.nearfield_potential_table as npt

    reference = npt.get_cahn_hilliard(2, b=b, c=c)
    assert np.isclose(knl_func(0.2, 0.0), reference(0.2, 0.0))


def test_get_kernel_function_rejects_partial_cahn_hilliard_coefficients(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        with pytest.raises(TypeError, match="requires both b and c together"):
            table_manager.get_kernel_function(2, "Cahn-Hilliard", b=3.5)

        with pytest.raises(TypeError, match="requires both b and c together"):
            table_manager.get_kernel_function(2, "Cahn-Hilliard", c=2.0)

        with pytest.raises(TypeError, match="requires both b and c"):
            table_manager.get_kernel_function(
                2, "Cahn-Hilliard", approx_at_origin=False
            )


def test_legacy_hdf5_cache_rebuilt_with_backup(tmp_path):
    filename = tmp_path / "legacy-cache.hdf5"
    filename.write_bytes(b"\x89HDF\r\n\x1a\nlegacy")

    with NFTable(str(filename), progress_bar=False):
        pass

    assert filename.exists()
    assert (tmp_path / "legacy-cache.hdf5.bak").exists()


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
            "_compute_and_update_table_for_request",
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
            "_compute_and_update_table_for_request",
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
                "build_method": "DuffyRadial",
                "payload": None,
            },
        )

        with pytest.raises(KeyError, match="payload"):
            table_manager.load_saved_table(
                2,
                "Laplace",
                q_order=1,
            )


def test_get_table_recomputes_on_payload_corruption(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(table_manager, "_record_exists", lambda *args: True)
        monkeypatch.setattr(
            table_manager,
            "_load_saved_table_for_request",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("missing payload")),
        )

        sentinel = object()
        recomputed = {"called": False}

        def fake_compute_and_update(*args, **kwargs):
            recomputed["called"] = True
            return sentinel

        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        table, is_recomputed = table_manager.get_table(2, "Laplace", q_order=1)
        assert table is sentinel
        assert is_recomputed
        assert recomputed["called"]


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_cache_miss_recomputes_with_single_builder_path(tmp_path, monkeypatch, dim):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(table_manager, "_record_exists", lambda *args: False)

        seen = {"called": False}

        def fake_compute_and_update(*args, **kwargs):
            seen["called"] = True
            return object()

        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        _, is_recomputed = table_manager.get_table(dim, "Laplace", q_order=1)

        assert is_recomputed
        assert seen["called"]


def test_get_table_from_request_recomputes_on_cache_miss(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"
    table_request = TableRequest.from_args(2, "Laplace", 1)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(table_manager, "_record_exists", lambda *args: False)

        seen = {"called": False}

        def fake_compute_and_update(*args, **kwargs):
            seen["called"] = True
            return object()

        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        _, is_recomputed = table_manager.get_table_from_request(table_request)

        assert is_recomputed
        assert seen["called"]


def test_request_methods_reject_non_request_objects(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        with pytest.raises(TypeError, match="table_request must be"):
            table_manager.get_table_from_request((2, "Laplace", 1))

        with pytest.raises(TypeError, match="table_request must be"):
            table_manager.load_saved_table_from_request((2, "Laplace", 1))

        with pytest.raises(TypeError, match="table_request must be"):
            table_manager.compute_and_update_table_for_request((2, "Laplace", 1))


def test_force_recompute_recomputes_with_single_builder_path(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(table_manager, "_record_exists", lambda *args: True)

        seen = {"called": False}

        def fake_compute_and_update(*args, **kwargs):
            seen["called"] = True
            return object()

        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        _, is_recomputed = table_manager.get_table(
            2,
            "Laplace",
            q_order=1,
            force_recompute=True,
        )

        assert is_recomputed
        assert seen["called"]


def test_load_saved_table_rejects_legacy_transform_build_method(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(
            table_manager,
            "_load_record",
            lambda *args, **kwargs: {
                "dim": 2,
                "quad_order": 1,
                "build_method": "Transform",
            },
        )

        with pytest.raises(KeyError, match="build_method is unsupported"):
            table_manager.load_saved_table(
                2,
                "Laplace",
                q_order=1,
            )


def test_removed_compute_method_keyword_is_rejected(tmp_path):
    filename = tmp_path / "cache.sqlite"
    table_request = TableRequest.from_args(2, "Laplace", 1)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        with pytest.raises(TypeError, match="compute_method has been removed"):
            table_manager.get_table(
                2, "Laplace", q_order=1, compute_method="DuffyRadial"
            )

        with pytest.raises(TypeError, match="compute_method has been removed"):
            table_manager.load_saved_table(
                2,
                "Laplace",
                q_order=1,
                compute_method="DuffyRadial",
            )

        with pytest.raises(TypeError, match="compute_method has been removed"):
            table_manager.compute_and_update_table(
                2,
                "Laplace",
                q_order=1,
                compute_method="DuffyRadial",
            )

        with pytest.raises(TypeError, match="compute_method has been removed"):
            table_manager.load_saved_table_from_request(
                table_request,
                compute_method="DuffyRadial",
            )

        with pytest.raises(TypeError, match="compute_method has been removed"):
            table_manager.compute_and_update_table_for_request(
                table_request,
                compute_method="DuffyRadial",
            )


def test_removed_top_level_duffy_knobs_are_rejected(tmp_path):
    filename = tmp_path / "cache.sqlite"
    table_request = TableRequest.from_args(2, "Laplace", 1)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        with pytest.raises(TypeError, match="top-level Duffy knobs have been removed"):
            table_manager.get_table(
                2,
                "Laplace",
                q_order=1,
                regular_quad_order=12,
            )

        with pytest.raises(TypeError, match="top-level Duffy knobs have been removed"):
            table_manager.load_saved_table_from_request(
                table_request,
                radial_quad_order=61,
            )

        with pytest.raises(TypeError, match="top-level Duffy knobs have been removed"):
            table_manager.compute_and_update_table_for_request(
                table_request,
                radial_rule="tanh-sinh-fast",
            )


def test_load_saved_table_payload_decode_failure_is_corruption(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(
            table_manager,
            "_load_record",
            lambda *args, **kwargs: {
                "dim": 2,
                "quad_order": 1,
                "build_method": "DuffyRadial",
                "payload": b"not-a-valid-npz",
            },
        )

        with pytest.raises(KeyError, match="payload is corrupted"):
            table_manager.load_saved_table(
                2,
                "Laplace",
                q_order=1,
            )


def test_get_table_recomputes_on_payload_decode_failure(tmp_path, monkeypatch):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False) as table_manager:
        monkeypatch.setattr(table_manager, "_record_exists", lambda *args: True)
        monkeypatch.setattr(
            table_manager,
            "_load_record",
            lambda *args, **kwargs: {
                "dim": 2,
                "quad_order": 1,
                "build_method": "DuffyRadial",
                "payload": b"not-a-valid-npz",
            },
        )

        seen = {"called": False}

        def fake_compute_and_update(*args, **kwargs):
            seen["called"] = True
            return object()

        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        _, is_recomputed = table_manager.get_table(2, "Laplace", q_order=1)

        assert is_recomputed
        assert seen["called"]


def test_unversioned_cache_rows_rebuilt_in_write_mode(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False):
        pass

    with sqlite3.connect(str(filename)) as db:
        _insert_dummy_cache_row(db, payload_blob=None)
        db.execute("DELETE FROM nearfield_cache_meta WHERE key='schema_version'")
        db.commit()

    with NFTable(str(filename), progress_bar=False):
        pass

    assert filename.exists()
    assert (tmp_path / "cache.sqlite.bak").exists()


def test_unversioned_cache_rows_rejected_in_read_only_mode(tmp_path):
    filename = tmp_path / "cache.sqlite"

    with NFTable(str(filename), progress_bar=False):
        pass

    with sqlite3.connect(str(filename)) as db:
        _insert_dummy_cache_row(db, payload_blob=None)
        db.execute("DELETE FROM nearfield_cache_meta WHERE key='schema_version'")
        db.commit()

    with pytest.raises(RuntimeError, match="missing schema_version"):
        with NFTable(str(filename), read_only=True, progress_bar=False):
            pass


def test_get_table_loads_using_stored_build_method_when_unspecified(tmp_path):
    from volumential.nearfield_potential_table import (
        NearFieldInteractionTable,
        constant_one,
    )

    filename = tmp_path / "cache.sqlite"

    template_table = NearFieldInteractionTable(
        quad_order=1,
        dim=1,
        build_method="DuffyRadial",
        kernel_func=constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(1),
        progress_bar=False,
    )
    template_table.data[:] = 3.5
    payload_blob = _serialize_table_payload(template_table)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        _insert_dummy_cache_row(
            table_manager.datafile,
            payload_blob=payload_blob,
            build_method="DuffyRadial",
        )
        table_manager.datafile.commit()

        loaded_table, is_recomputed = table_manager.get_table(
            1,
            "Constant",
            q_order=1,
            force_recompute=False,
        )

    assert not is_recomputed
    assert loaded_table.build_method == "DuffyRadial"
    assert np.allclose(loaded_table.data, template_table.data)


def test_load_saved_table_ignores_manager_precomputed_q_points(tmp_path):
    from volumential.nearfield_potential_table import (
        NearFieldInteractionTable,
        constant_one,
    )

    filename = tmp_path / "cache.sqlite"

    template_table = NearFieldInteractionTable(
        quad_order=1,
        dim=1,
        build_method="DuffyRadial",
        kernel_func=constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(1),
        progress_bar=False,
    )
    template_table.data[:] = 3.5
    payload_blob = _serialize_table_payload(template_table)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        _insert_dummy_cache_row(
            table_manager.datafile,
            payload_blob=payload_blob,
            build_method="DuffyRadial",
        )
        table_manager.datafile.commit()

    with NFTable(
        str(filename),
        progress_bar=False,
        precomputed_q_points=np.array([999.0, -1.0]),
    ) as table_manager:
        loaded_table = table_manager.load_saved_table(1, "Constant", q_order=1)

    assert np.allclose(loaded_table.q_points, template_table.q_points)
    assert np.allclose(loaded_table.data, template_table.data)


def test_get_table_recomputes_on_build_config_mismatch(tmp_path, monkeypatch):
    from volumential.nearfield_potential_table import (
        DuffyBuildConfig,
        NearFieldInteractionTable,
        constant_one,
    )

    filename = tmp_path / "cache.sqlite"

    template_table = NearFieldInteractionTable(
        quad_order=1,
        dim=1,
        build_method="DuffyRadial",
        kernel_func=constant_one,
        kernel_type="const",
        sumpy_kernel=ConstantKernel(1),
        progress_bar=False,
    )
    template_table.data[:] = 3.5
    payload_blob = _serialize_table_payload(template_table)

    coarse = DuffyBuildConfig(regular_quad_order=4, radial_quad_order=11)
    fine = DuffyBuildConfig(regular_quad_order=12, radial_quad_order=61)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        _insert_dummy_cache_row(
            table_manager.datafile,
            payload_blob=payload_blob,
            build_method="DuffyRadial",
        )
        coarse_fingerprint = table_manager._build_config_fingerprint(
            {"build_config": coarse}
        )
        table_manager.datafile.execute(
            "INSERT INTO nearfield_cache_kwargs "
            "(dim, kernel_type, q_order, source_box_level, key, value_type, value_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "Constant",
                1,
                0,
                "build_config_fingerprint",
                "str",
                coarse_fingerprint,
            ),
        )
        table_manager.datafile.commit()

        sentinel = object()
        seen = {"called": False}

        def fake_compute_and_update(*args, **kwargs):
            seen["called"] = True
            return sentinel

        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        loaded_table, is_recomputed = table_manager.get_table(
            1,
            "Constant",
            q_order=1,
            build_config=fine,
        )

    assert is_recomputed
    assert seen["called"]
    assert loaded_table is sentinel


def test_force_recompute_reuses_cached_build_config_when_unspecified(
    tmp_path, monkeypatch
):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    filename = tmp_path / "cache.sqlite"
    cached_build_config = DuffyBuildConfig(regular_quad_order=10, radial_quad_order=41)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        _insert_dummy_cache_row(table_manager.datafile, payload_blob=b"x")

        serialized = table_manager._build_config_fingerprint(
            {"build_config": cached_build_config}
        )
        table_manager.datafile.executemany(
            "INSERT INTO nearfield_cache_kwargs "
            "(dim, kernel_type, q_order, source_box_level, key, value_type, value_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "Constant", 1, 0, "build_config_fingerprint", "str", serialized),
            ],
        )
        table_manager.datafile.commit()

        sentinel = object()
        seen = {}

        def fake_compute_and_update(*args, **kwargs):
            seen["build_config"] = kwargs.get("build_config")
            return sentinel

        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        table, is_recomputed = table_manager.get_table(
            1,
            "Constant",
            q_order=1,
            force_recompute=True,
        )

    assert is_recomputed
    assert table is sentinel
    assert seen["build_config"] == cached_build_config


def test_corruption_recompute_reuses_cached_build_config_when_unspecified(
    tmp_path, monkeypatch
):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    filename = tmp_path / "cache.sqlite"
    cached_build_config = DuffyBuildConfig(regular_quad_order=12, radial_quad_order=61)

    with NFTable(str(filename), progress_bar=False) as table_manager:
        _insert_dummy_cache_row(table_manager.datafile, payload_blob=b"x")

        serialized = table_manager._build_config_fingerprint(
            {"build_config": cached_build_config}
        )
        table_manager.datafile.executemany(
            "INSERT INTO nearfield_cache_kwargs "
            "(dim, kernel_type, q_order, source_box_level, key, value_type, value_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "Constant", 1, 0, "build_config_fingerprint", "str", serialized),
                (1, "Constant", 1, 0, "build_config_json", "str", serialized),
            ],
        )
        table_manager.datafile.commit()

        sentinel = object()
        seen = {}

        def fake_compute_and_update(*args, **kwargs):
            seen["build_config"] = kwargs.get("build_config")
            return sentinel

        monkeypatch.setattr(
            table_manager,
            "_load_saved_table_for_request",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("corrupt")),
        )
        monkeypatch.setattr(
            table_manager,
            "_compute_and_update_table_for_request",
            fake_compute_and_update,
        )

        table, is_recomputed = table_manager.get_table(1, "Constant", q_order=1)

    assert is_recomputed
    assert table is sentinel
    assert seen["build_config"] == cached_build_config
