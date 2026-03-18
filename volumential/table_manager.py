__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__doc__ = """
.. autoclass:: NearFieldInteractionTableManager
   :members:
"""

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
import re
import sqlite3
import time
from io import BytesIO
from urllib.parse import quote

import numpy as np

from sumpy.kernel import ExpressionKernel

import volumential as vm
from volumential.nearfield_potential_table import NearFieldInteractionTable


logger = logging.getLogger(__name__)


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


TABLE_CACHE_SCHEMA_VERSION = "2.0.0"
TABLE_CACHE_MIN_READABLE_SCHEMA_VERSION = "2.0.0"
_LEGACY_UNVERSIONED_SCHEMA_VERSION = "1.0.0"


def _parse_semver(version_text, what="version"):
    if not isinstance(version_text, str):
        raise ValueError(f"Invalid {what}: expected string, got {type(version_text)!r}")

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_text)
    if match is None:
        raise ValueError(
            f"Invalid {what} {version_text!r}: expected semantic version MAJOR.MINOR.PATCH"
        )

    return tuple(int(part) for part in match.groups())


def _quote_sqlite_path(path):
    return "file:{}".format(quote(str(path), safe="/"))


def _sqlite_uri(path, mode):
    return "{}?mode={}".format(_quote_sqlite_path(path), mode)


def _serialize_array(array):
    with BytesIO() as f:
        np.save(f, np.array(array), allow_pickle=False)
        return f.getvalue()


def _deserialize_array(blob):
    with BytesIO(blob) as f:
        return np.load(f, allow_pickle=False)


def _serialize_table_payload(table):
    data = np.array(table.data)
    table_data_is_symmetry_reduced = bool(
        getattr(table, "table_data_is_symmetry_reduced", False)
    )

    payload = {
        "q_points": np.array(table.q_points),
        "mode_normalizers": np.array(table.mode_normalizers),
        "kernel_exterior_normalizers": np.array(table.kernel_exterior_normalizers),
        "interaction_case_vecs": np.array(table.interaction_case_vecs),
        "case_indices": np.array(table.case_indices),
        "table_data_is_symmetry_reduced": np.array(
            [int(table_data_is_symmetry_reduced)],
            dtype=np.int8,
        ),
    }

    if table_data_is_symmetry_reduced:
        reduced_entry_ids = np.flatnonzero(np.isfinite(data)).astype(np.int64)
        payload["reduced_entry_ids"] = reduced_entry_ids
        payload["reduced_data"] = data[reduced_entry_ids]
    else:
        payload["data"] = data

    with BytesIO() as f:
        np.savez(f, **payload)
        return f.getvalue()


def _deserialize_table_payload(blob):
    with BytesIO(blob) as f:
        with np.load(f, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}


def _serialize_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return "bool", int(value)

    if isinstance(value, (int, np.integer)):
        return "int", int(value)

    if isinstance(value, (float, np.floating)):
        return "float", repr(float(value))

    if isinstance(value, (complex, np.complexfloating)):
        return "complex", repr(complex(value))

    if isinstance(value, str):
        return "str", value

    raise TypeError(f"Unsupported value type: {type(value)!r}")


def _deserialize_scalar(ty, value):
    if ty == "bool":
        return bool(int(value))

    if ty == "int":
        return int(value)

    if ty == "float":
        return float(value)

    if ty == "complex":
        return complex(value)

    if ty == "str":
        return str(value)

    raise ValueError("Unsupported serialized value type: %s" % ty)


def _looks_like_hdf5_file(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE
    except OSError:
        return False


# {{{ constant sumpy kernel


class ConstantKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self, dim=None):

        expr = 1
        scaling = 1

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

    has_efficient_scale_adjustment = True

    @property
    def is_complex_valued(self):
        return False

    def adjust_for_kernel_scaling(self, expr, rscale, nderivatives):
        return expr / rscale

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return "CstKnl%dD" % self.dim

    mapper_method = "map_expression_kernel"


# }}} End constant sumpy kernel

# {{{ table dataset manager class


class NearFieldInteractionTableManager:
    """
    A class that manages near field interaction table computation and
    storage.

    Tables are stored in an SQLite database and keyed by
    (dimension, kernel_type, quadrature order, source_box_level).

    Only one table manager can exist for a dataset file with write access.
    The access can be controlled with the read_only argument. By default,
    the constructor tries to open the dataset with write access, and falls
    back to read-only if that fails.
    """

    def __init__(
        self,
        dataset_filename="nft.hdf5",
        root_extent=1,
        dtype=np.float64,
        read_only="auto",
        **kwargs,
    ):
        """Constructor."""

        self.dtype = dtype

        self.filename = dataset_filename
        self.root_extent = root_extent
        self.last_compute_timings = None
        self.last_load_timings = None
        self.last_get_table_timings = None

        if read_only == "auto":
            try:
                self._read_only = False
                conn = sqlite3.connect(_sqlite_uri(self.filename, "rwc"), uri=True)
            except sqlite3.OperationalError as e:
                from warnings import warn

                warn("Trying to open in read/write mode failed: %s" % str(e))
                warn("Opening table dataset %s in read-only mode." % self.filename)
                self._read_only = True
                conn = sqlite3.connect(_sqlite_uri(self.filename, "ro"), uri=True)
        elif read_only:
            self._read_only = True
            conn = sqlite3.connect(_sqlite_uri(self.filename, "ro"), uri=True)
        else:
            # Read/write if exists, create otherwise
            self._read_only = False
            conn = sqlite3.connect(_sqlite_uri(self.filename, "rwc"), uri=True)

        self.datafile = conn
        self.datafile.row_factory = sqlite3.Row

        self.datafile.execute("PRAGMA foreign_keys = ON")
        try:
            self.datafile.execute("PRAGMA temp_store = MEMORY")
            self.datafile.execute("PRAGMA cache_size = -200000")
            if not self._read_only:
                self.datafile.execute("PRAGMA journal_mode = WAL")
                self.datafile.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:
            logger.warning("Failed to apply one or more SQLite tuning pragmas")

        try:
            self._init_schema()
            self._initialize_root_extent()
        except sqlite3.DatabaseError as exc:
            self.datafile.close()
            if _looks_like_hdf5_file(self.filename):
                raise RuntimeError(
                    "The table cache file "
                    + self.filename
                    + " appears to be in the legacy HDF5 format. "
                    "SQLite is now required. Remove this file to rebuild "
                    "the cache."
                ) from exc
            raise

        self.table_extra_kwargs = kwargs

        self.supported_kernels = [
            "Laplace",
            "Laplace-Dx",
            "Laplace-Dy",
            "Laplace-Dz",
            "Constant",
            "Yukawa",
            "Yukawa-Dx",
            "Yukawa-Dy",
            "Cahn-Hilliard",
            "Cahn-Hilliard-Laplacian",
            "Cahn-Hilliard-Dx",
            "Cahn-Hilliard-Dy",
            "Cahn-Hilliard-Laplacian-Dx",
            "Cahn-Hilliard-Laplacian-Dy",
        ]

    def _init_schema(self):
        """Create missing tables for a new cache file."""

        self.datafile.execute(
            """
            CREATE TABLE IF NOT EXISTS nearfield_cache (
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

        self.datafile.execute(
            """
            CREATE TABLE IF NOT EXISTS nearfield_cache_kwargs (
                dim INTEGER NOT NULL,
                kernel_type TEXT NOT NULL,
                q_order INTEGER NOT NULL,
                source_box_level INTEGER NOT NULL,
                key TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_text TEXT NOT NULL,

                PRIMARY KEY (
                    dim, kernel_type, q_order, source_box_level, key
                ),
                FOREIGN KEY (
                    dim, kernel_type, q_order, source_box_level
                ) REFERENCES nearfield_cache (
                    dim, kernel_type, q_order, source_box_level
                ) ON DELETE CASCADE
            )
            """
        )

        self.datafile.execute(
            """
            CREATE TABLE IF NOT EXISTS nearfield_cache_meta (
                key TEXT PRIMARY KEY,
                value_type TEXT NOT NULL,
                value_text TEXT NOT NULL
            )
            """
        )

        columns = {
            row["name"]
            for row in self.datafile.execute("PRAGMA table_info(nearfield_cache)")
        }
        if "payload" not in columns:
            if self._read_only:
                logger.warning(
                    "Table cache file %s uses legacy schema without payload column; "
                    "read-only mode cannot apply migrations.",
                    self.filename,
                )
            else:
                self.datafile.execute(
                    "ALTER TABLE nearfield_cache ADD COLUMN payload BLOB"
                )
                columns.add("payload")

        self._initialize_schema_version(columns)

        self.datafile.commit()

    def _get_meta_value(self, key):
        row = self.datafile.execute(
            "SELECT value_type, value_text FROM nearfield_cache_meta WHERE key=?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        return _deserialize_scalar(row["value_type"], row["value_text"])

    def _store_meta_value(self, key, value):
        value_type, value_text = _serialize_scalar(value)
        self.datafile.execute(
            "INSERT OR REPLACE INTO nearfield_cache_meta (key, value_type, value_text) "
            "VALUES (?, ?, ?)",
            (key, value_type, str(value_text)),
        )

    def _infer_schema_version_from_columns(self, cache_columns):
        if "payload" in cache_columns:
            return TABLE_CACHE_SCHEMA_VERSION
        return _LEGACY_UNVERSIONED_SCHEMA_VERSION

    def _initialize_schema_version(self, cache_columns):
        current_version_tuple = _parse_semver(
            TABLE_CACHE_SCHEMA_VERSION,
            what="current table cache schema version",
        )
        min_readable_version_tuple = _parse_semver(
            TABLE_CACHE_MIN_READABLE_SCHEMA_VERSION,
            what="minimum readable table cache schema version",
        )

        stored_schema_version = self._get_meta_value("schema_version")
        has_stored_schema_version = stored_schema_version is not None

        has_cache_rows = (
            self.datafile.execute("SELECT 1 FROM nearfield_cache LIMIT 1").fetchone()
            is not None
        )

        if not has_stored_schema_version:
            if self._read_only and has_cache_rows:
                raise RuntimeError(
                    "The table cache file "
                    + self.filename
                    + " is missing schema_version metadata and is treated as "
                    + "incompatible legacy cache."
                )

            if not self._read_only and has_cache_rows:
                logger.warning(
                    "Resetting legacy unversioned table cache data in %s.",
                    self.filename,
                )
                self.datafile.execute("DELETE FROM nearfield_cache_kwargs")
                self.datafile.execute("DELETE FROM nearfield_cache")

            if not self._read_only:
                self._store_meta_value("schema_version", TABLE_CACHE_SCHEMA_VERSION)

            self.cache_schema_version = TABLE_CACHE_SCHEMA_VERSION
            return

        if not isinstance(stored_schema_version, str):
            raise RuntimeError(
                "The table cache file "
                + self.filename
                + " has invalid schema_version metadata type "
                + str(type(stored_schema_version))
            )
        schema_version = stored_schema_version

        try:
            schema_version_tuple = _parse_semver(
                schema_version,
                what="table cache schema version",
            )
        except ValueError as exc:
            raise RuntimeError(
                "The table cache file "
                + self.filename
                + " has invalid schema_version metadata: "
                + str(exc)
            ) from exc

        if schema_version_tuple[0] > current_version_tuple[0]:
            raise RuntimeError(
                "The table cache file "
                + self.filename
                + " uses incompatible schema version "
                + schema_version
                + ". This build supports "
                + TABLE_CACHE_MIN_READABLE_SCHEMA_VERSION
                + " through "
                + TABLE_CACHE_SCHEMA_VERSION
                + "."
            )

        if schema_version_tuple < min_readable_version_tuple:
            if self._read_only:
                raise RuntimeError(
                    "The table cache file "
                    + self.filename
                    + " uses unsupported legacy schema version "
                    + schema_version
                    + "."
                )

            logger.warning(
                "Resetting legacy table cache schema %s in %s.",
                schema_version,
                self.filename,
            )
            self.datafile.execute("DELETE FROM nearfield_cache_kwargs")
            self.datafile.execute("DELETE FROM nearfield_cache")

            schema_version = TABLE_CACHE_SCHEMA_VERSION
            self._store_meta_value("schema_version", schema_version)
            self.cache_schema_version = schema_version
            return

        self.cache_schema_version = schema_version

    def _initialize_root_extent(self):
        stored_root_extent = self._get_meta_value("root_extent")

        if stored_root_extent is None:
            if self._read_only:
                return

            self._store_root_extent(self.root_extent)
            return

        stored_root_extent = float(stored_root_extent)
        if abs(stored_root_extent - float(self.root_extent)) >= 1e-15:
            raise RuntimeError(
                "The table cache file "
                + self.filename
                + " was built with root_extent = "
                + str(stored_root_extent)
                + ", which is different from the requested value "
                + str(self.root_extent)
            )

    def _store_root_extent(self, root_extent):
        self._store_meta_value("root_extent", root_extent)
        self.datafile.commit()

    def _record_exists(self, dim, kernel_type, q_order, source_box_level):
        row = self.datafile.execute(
            "SELECT 1 FROM nearfield_cache WHERE dim=? AND kernel_type=? "
            "AND q_order=? AND source_box_level=?",
            (dim, kernel_type, q_order, source_box_level),
        ).fetchone()

        return row is not None

    def _load_record(self, dim, kernel_type, q_order, source_box_level):
        return self.datafile.execute(
            "SELECT * FROM nearfield_cache WHERE dim=? AND kernel_type=? "
            "AND q_order=? AND source_box_level=?",
            (dim, kernel_type, q_order, source_box_level),
        ).fetchone()

    def _load_record_kwargs(self, dim, kernel_type, q_order, source_box_level):
        rows = self.datafile.execute(
            "SELECT key, value_type, value_text FROM nearfield_cache_kwargs "
            "WHERE dim=? AND kernel_type=? AND q_order=? AND source_box_level=?",
            (dim, kernel_type, q_order, source_box_level),
        ).fetchall()

        kwargs = {}
        for row in rows:
            kwargs[row["key"]] = _deserialize_scalar(
                row["value_type"], row["value_text"]
            )

        return kwargs

    def _store_record_kwargs(self, dim, kernel_type, q_order, source_box_level, kwargs):
        self.datafile.execute(
            "DELETE FROM nearfield_cache_kwargs WHERE dim=? AND kernel_type=? "
            "AND q_order=? AND source_box_level=?",
            (dim, kernel_type, q_order, source_box_level),
        )

        rows = []
        for key, value in kwargs.items():
            if value is None:
                continue

            try:
                value_type, value_text = _serialize_scalar(value)
            except TypeError:
                continue

            if isinstance(value, str):
                value_text = str(value)

            rows.append(
                (
                    dim,
                    kernel_type,
                    q_order,
                    source_box_level,
                    key,
                    value_type,
                    str(value_text),
                )
            )

        if rows:
            self.datafile.executemany(
                "INSERT INTO nearfield_cache_kwargs "
                "(dim, kernel_type, q_order, source_box_level, key, value_type, value_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.datafile.commit()
        self.datafile.close()

    def get_table(
        self,
        dim,
        kernel_type,
        q_order,
        source_box_level=0,
        force_recompute=False,
        compute_method=None,
        queue=None,
        **kwargs,
    ):
        """Primary user interface. Get the specified table regardless of how.
        In the case of a cache miss or a forced re-computation, the method
        specified in the compute_method will be used.
        """

        t_get_start = time.perf_counter()

        dim = int(dim)
        assert dim >= 1

        requested_compute_method = compute_method
        compute_method_for_compute = compute_method
        if compute_method_for_compute is None:
            compute_method_for_compute = "Transform"

        is_recomputed = False

        q_order = int(q_order)
        assert q_order >= 1

        source_box_level = int(source_box_level)
        assert source_box_level >= 0

        if not self._record_exists(dim, kernel_type, q_order, source_box_level):
            if self._read_only:
                raise RuntimeError(
                    "Table cache miss in read-only mode for "
                    f"(dim={dim}, kernel_type={kernel_type}, q_order={q_order}, "
                    f"source_box_level={source_box_level})."
                )
            logger.info("Table cache missing. Invoking fresh computation.")
            is_recomputed = True
            table = self.compute_and_update_table(
                dim,
                kernel_type,
                q_order,
                source_box_level,
                compute_method_for_compute,
                queue=queue,
                **kwargs,
            )

        elif force_recompute:
            if self._read_only:
                raise RuntimeError("force_recompute is not supported in read-only mode")
            logger.info("Invoking fresh computation since force_recompute is set")
            is_recomputed = True
            table = self.compute_and_update_table(
                dim,
                kernel_type,
                q_order,
                source_box_level,
                compute_method_for_compute,
                queue=queue,
                **kwargs,
            )

        else:
            try:
                table = self.load_saved_table(
                    dim,
                    kernel_type,
                    q_order,
                    source_box_level,
                    requested_compute_method,
                    **kwargs,
                )

            except KeyError:
                import traceback

                logger.debug(traceback.format_exc())

                if self._read_only:
                    raise RuntimeError(
                        "Cached table data is unavailable in read-only mode and "
                        "cannot be recomputed."
                    )

                logger.info("Recomputing due to table data corruption.")
                is_recomputed = True
                table = self.compute_and_update_table(
                    dim,
                    kernel_type,
                    q_order,
                    source_box_level,
                    compute_method_for_compute,
                    queue=queue,
                    **kwargs,
                )

            # Ensure loaded table matches requirements specified in kwargs
            for kkey, kval in kwargs.items():
                if kval is not None:
                    try:
                        tbval = getattr(table, kkey)
                        if isinstance(kval, (int, str)):
                            if not kval == tbval:
                                from warnings import warn

                                warn(
                                    "Table data loaded with a different value "
                                    + kkey
                                    + " = "
                                    + str(tbval)
                                    + " (expected "
                                    + str(kval)
                                    + ")"
                                )
                        else:
                            assert isinstance(kval, (float, complex))
                            tbval = kval.__class__(tbval)
                            if not abs(kval - tbval) < 1e-12:
                                from warnings import warn

                                warn(
                                    "Table data loaded with a different value "
                                    + kkey
                                    + " = "
                                    + str(tbval)
                                    + " (expected "
                                    + str(kval)
                                    + ")"
                                )

                    except AttributeError as e:
                        strict_loading = False
                        if "debug" in kwargs:
                            if "strict_loading" in kwargs["debug"]:
                                strict_loading = kwargs["debug"]["strict_loading"]

                        if strict_loading:
                            from warnings import warn

                            warn(
                                "Consistency is not fully ensured "
                                "(kwarg specified but cannot be loaded). "
                                "NOTE: this is most likely due to non-standard "
                                "arguements being passed, since only "
                                "(int, float, complex, bool, str) "
                                "are stored in the cache. "
                                "Also, some parameters related to method for "
                                "table building are not critical for "
                                "consistency."
                            )
                            print(e)

        t_get_end = time.perf_counter()
        self.last_get_table_timings = {
            "total_s": t_get_end - t_get_start,
            "is_recomputed": bool(is_recomputed),
            "compute": self.last_compute_timings if is_recomputed else None,
            "load": self.last_load_timings if not is_recomputed else None,
        }

        return table, is_recomputed

    def load_saved_table(
        self,
        dim,
        kernel_type,
        q_order,
        source_box_level=0,
        compute_method=None,
        **kwargs,
    ):
        """Load a table saved in the SQLite cache."""

        t_load_start = time.perf_counter()

        q_order = int(q_order)
        assert q_order >= 1

        source_box_level = int(source_box_level)
        assert source_box_level >= 0

        record = self._load_record(dim, kernel_type, q_order, source_box_level)
        t_record_fetch_end = time.perf_counter()
        if record is None:
            raise KeyError("missing table record")

        if dim != record["dim"]:
            raise AssertionError("cache record dimension mismatch")
        if q_order != record["quad_order"]:
            raise AssertionError("cache record quad order mismatch")

        stored_build_method = record["build_method"]
        if stored_build_method in {"Transform", "DuffyRadial"}:
            effective_compute_method = stored_build_method
        else:
            logger.warning(
                "Unknown cached build_method=%r, assuming DuffyRadial",
                stored_build_method,
            )
            effective_compute_method = "DuffyRadial"

        if (
            compute_method is not None
            and compute_method in {"Transform", "DuffyRadial"}
            and compute_method != effective_compute_method
        ):
            logger.warning(
                "Requested compute_method=%r differs from cached build_method=%r; "
                "loading cached table using the stored build method.",
                compute_method,
                effective_compute_method,
            )

        if effective_compute_method == "Transform":
            if "knl_func" not in kwargs:
                knl_func = self.get_kernel_function(dim, kernel_type, **kwargs)
            else:
                knl_func = kwargs["knl_func"]
            sumpy_knl = None
        elif effective_compute_method == "DuffyRadial":
            if "knl_func" not in kwargs:
                knl_func = self.get_kernel_function(dim, kernel_type, **kwargs)
            else:
                knl_func = kwargs["knl_func"]
            if "sumpy_knl" in kwargs:
                sumpy_knl = kwargs["sumpy_knl"]
            else:
                try:
                    sumpy_knl = self.get_sumpy_kernel(dim, kernel_type)
                except NotImplementedError:
                    sumpy_knl = None
        else:
            raise ValueError(
                f"unsupported compute_method={compute_method!r}; "
                "supported methods are 'Transform' and 'DuffyRadial'"
            )

        payload_blob = record["payload"] if "payload" in record.keys() else None
        used_payload = bool(payload_blob)
        payload = None
        t_payload_deser_start = time.perf_counter()
        if used_payload:
            payload = _deserialize_table_payload(payload_blob)
        else:
            raise KeyError("table cache payload is missing")
        t_payload_deser_end = time.perf_counter()

        precomputed_q_points = None
        if payload is not None and "q_points" in payload:
            precomputed_q_points = payload["q_points"]

        table = NearFieldInteractionTable(
            quad_order=q_order,
            dim=dim,
            dtype=self.dtype,
            build_method=effective_compute_method,
            kernel_func=knl_func,
            kernel_type=self.get_kernel_function_type(dim, kernel_type),
            sumpy_kernel=sumpy_knl,
            source_box_extent=self.root_extent * (2 ** (-source_box_level)),
            precomputed_q_points=precomputed_q_points,
            **self.table_extra_kwargs,
        )

        assert abs(table.source_box_extent - record["source_box_extent"]) < 1e-15
        assert source_box_level == record["source_box_level_stored"]

        # Load data
        table.q_points[:] = payload["q_points"]
        if "data" in payload:
            table.data[:] = payload["data"]
        elif "reduced_entry_ids" in payload and "reduced_data" in payload:
            table.data[:] = np.nan
            table.data[payload["reduced_entry_ids"]] = payload["reduced_data"]
        else:
            raise KeyError("payload is missing table data arrays")

        table.mode_normalizers[:] = payload["mode_normalizers"]
        table.kernel_exterior_normalizers[:] = payload["kernel_exterior_normalizers"]

        tmp_case_vecs = np.array(table.interaction_case_vecs)
        tmp_case_vecs[...] = payload["interaction_case_vecs"]
        table.interaction_case_vecs = [list(vec) for vec in tmp_case_vecs]

        table.case_indices[:] = payload["case_indices"]
        if "table_data_is_symmetry_reduced" in payload:
            table.table_data_is_symmetry_reduced = bool(
                payload["table_data_is_symmetry_reduced"][0]
            )

        assert table.n_q_points == record["n_q_points"]
        assert table.n_pairs == record["n_pairs"]
        assert table.quad_order == record["quad_order"]

        base = record["case_encoding_base"]
        shift = record["case_encoding_shift"]

        def case_encode(case_vec):
            table_id = 0
            for case_vec_comp in case_vec:
                table_id = table_id * base + (case_vec_comp + shift)
            return int(table_id)

        table.case_encode = case_encode

        table.source_box_level = source_box_level
        table.dim = record["dim"]
        table.n_q_points = record["n_q_points"]
        table.n_pairs = record["n_pairs"]
        table.case_encoding_base = base
        table.case_encoding_shift = shift
        table.build_method = effective_compute_method
        table.kernel_type_cached = record["kernel_type_cached"]
        table.source_box_extent = record["source_box_extent"]

        # load extra kwargs
        t_kwargs_load_start = time.perf_counter()
        for atkey, atval in self._load_record_kwargs(
            dim, kernel_type, q_order, source_box_level
        ).items():
            setattr(table, atkey, atval)
        t_kwargs_load_end = time.perf_counter()

        table.is_built = True

        t_load_end = time.perf_counter()
        self.last_load_timings = {
            "record_fetch_s": t_record_fetch_end - t_load_start,
            "payload_deserialize_s": t_payload_deser_end - t_payload_deser_start,
            "kwargs_load_s": t_kwargs_load_end - t_kwargs_load_start,
            "postprocess_s": t_load_end - t_kwargs_load_end,
            "total_s": t_load_end - t_load_start,
            "used_payload": used_payload,
            "payload_bytes": len(payload_blob) if payload_blob else 0,
        }

        return table

    def get_kernel_function(self, dim, kernel_type, **kwargs):
        """Kernel function is needed for building the table. This function
        provides support for some kernels such that the user can build and
        use the table without explicitly providing such information.
        """
        if kernel_type == "Laplace":
            knl_func = vm.nearfield_potential_table.get_laplace(dim)
        elif kernel_type == "Constant":
            knl_func = vm.nearfield_potential_table.constant_one
        elif kernel_type == "Cahn-Hilliard":
            knl_func = vm.nearfield_potential_table.get_cahn_hilliard(
                dim, kwargs["b"], kwargs["c"]
            )
        elif kernel_type in self.supported_kernels:
            knl = self.get_sumpy_kernel(dim, kernel_type)
            knl_func = vm.nearfield_potential_table.sumpy_kernel_to_lambda(knl)
        else:
            raise NotImplementedError("Kernel type not supported.")

        return knl_func

    def get_sumpy_kernel(self, dim, kernel_type):
        """Sumpy (symbolic) version of the kernel."""

        if kernel_type == "Laplace":
            from sumpy.kernel import LaplaceKernel

            return LaplaceKernel(dim)

        if kernel_type == "Laplace-Dx":
            from sumpy.kernel import AxisTargetDerivative, LaplaceKernel

            return AxisTargetDerivative(0, LaplaceKernel(dim))

        if kernel_type == "Laplace-Dy":
            from sumpy.kernel import AxisTargetDerivative, LaplaceKernel

            return AxisTargetDerivative(1, LaplaceKernel(dim))

        if kernel_type == "Laplace-Dz":
            from sumpy.kernel import AxisTargetDerivative, LaplaceKernel

            assert dim >= 3
            return AxisTargetDerivative(2, LaplaceKernel(dim))

        elif kernel_type == "Constant":
            return ConstantKernel(dim)

        elif kernel_type == "Yukawa":
            from sumpy.kernel import YukawaKernel

            return YukawaKernel(dim)

        elif kernel_type == "Yukawa-Dx":
            from sumpy.kernel import AxisTargetDerivative, YukawaKernel

            return AxisTargetDerivative(0, YukawaKernel(dim))

        elif kernel_type == "Yukawa-Dy":
            from sumpy.kernel import AxisTargetDerivative, YukawaKernel

            return AxisTargetDerivative(1, YukawaKernel(dim))

        elif kernel_type == "Cahn-Hilliard":
            from sumpy.kernel import FactorizedBiharmonicKernel

            return FactorizedBiharmonicKernel(dim)

        elif kernel_type == "Cahn-Hilliard-Laplacian":
            from sumpy.kernel import (
                FactorizedBiharmonicKernel,
                LaplacianTargetDerivative,
            )

            return LaplacianTargetDerivative(FactorizedBiharmonicKernel(dim))

        elif kernel_type == "Cahn-Hilliard-Dx":
            from sumpy.kernel import AxisTargetDerivative, FactorizedBiharmonicKernel

            return AxisTargetDerivative(0, FactorizedBiharmonicKernel(dim))

        elif kernel_type == "Cahn-Hilliard-Laplacian-Dx":
            from sumpy.kernel import (
                AxisTargetDerivative,
                FactorizedBiharmonicKernel,
                LaplacianTargetDerivative,
            )

            return AxisTargetDerivative(
                0, LaplacianTargetDerivative(FactorizedBiharmonicKernel(dim))
            )

        elif kernel_type == "Cahn-Hilliard-Laplacian-Dy":
            from sumpy.kernel import (
                AxisTargetDerivative,
                FactorizedBiharmonicKernel,
                LaplacianTargetDerivative,
            )

            return AxisTargetDerivative(
                1, LaplacianTargetDerivative(FactorizedBiharmonicKernel(dim))
            )

        elif kernel_type == "Cahn-Hilliard-Dy":
            from sumpy.kernel import AxisTargetDerivative, FactorizedBiharmonicKernel

            return AxisTargetDerivative(1, FactorizedBiharmonicKernel(dim))

        elif kernel_type in self.supported_kernels:
            return None

        else:
            raise NotImplementedError("Kernel type not supported.")

    def get_kernel_function_type(self, dim, kernel_type):
        """Determines how and to what extend the table data can be rescaled
        and reused.
        """

        if kernel_type == "Laplace":
            if dim == 2:
                return "log"
            elif dim >= 3:
                return "inv_power"
            else:
                raise NotImplementedError("Kernel scaling not supported")

        elif kernel_type == "Constant":
            if dim >= 1 and dim <= 3:
                return "const"
            else:
                raise NotImplementedError("Kernel scaling not supported")

        else:
            return None

    def compute_and_update_table(
        self,
        dim,
        kernel_type,
        q_order,
        source_box_level=0,
        compute_method=None,
        cl_ctx=None,
        queue=None,
        **kwargs,
    ):
        """Performs the precomputation and stores the results."""

        if compute_method is None:
            logger.debug("Using default compute_method (Transform)")
            compute_method = "Transform"

        q_order = int(q_order)
        assert q_order >= 1

        if compute_method == "Transform":
            if "knl_func" not in kwargs:
                knl_func = self.get_kernel_function(dim, kernel_type, **kwargs)
            else:
                knl_func = kwargs["knl_func"]
            sumpy_knl = None
        elif compute_method == "DuffyRadial":
            if "knl_func" not in kwargs:
                knl_func = self.get_kernel_function(dim, kernel_type, **kwargs)
            else:
                knl_func = kwargs["knl_func"]
            if "sumpy_knl" in kwargs:
                sumpy_knl = kwargs["sumpy_knl"]
            else:
                try:
                    sumpy_knl = self.get_sumpy_kernel(dim, kernel_type)
                except NotImplementedError:
                    sumpy_knl = None
        else:
            raise NotImplementedError("Unsupported compute_method.")

        knl_type = self.get_kernel_function_type(dim, kernel_type)

        # compute table
        logger.debug("Start computing interaction table.")
        table = NearFieldInteractionTable(
            dim=dim,
            quad_order=q_order,
            dtype=self.dtype,
            kernel_func=knl_func,
            kernel_type=knl_type,
            sumpy_kernel=sumpy_knl,
            build_method=compute_method,
            source_box_extent=self.root_extent * (2 ** (-source_box_level)),
            **self.table_extra_kwargs,
        )

        if 0:
            # self-similarly shrink delta
            if "delta" in kwargs:
                delta = kwargs.pop("delta") * (2 ** (-source_box_level))
                kwargs["delta"] = delta

        t_compute_start = time.perf_counter()
        table.build_table(cl_ctx, queue, **kwargs)
        t_compute_end = time.perf_counter()
        assert table.is_built

        # update database
        logger.debug("Start updating database.")

        source_box_extent = self.root_extent * (2 ** (-source_box_level))
        t_payload_serialize_start = time.perf_counter()
        payload_blob = _serialize_table_payload(table)
        empty_float_blob = _serialize_array(np.array([], dtype=self.dtype))
        t_payload_serialize_end = time.perf_counter()

        distinct_numbers = set()
        for vec in table.interaction_case_vecs:
            for case_vec_comp in vec:
                distinct_numbers.add(case_vec_comp)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        record_values = (
            dim,
            kernel_type,
            q_order,
            source_box_level,
            table.n_q_points,
            table.quad_order,
            table.n_pairs,
            source_box_extent,
            source_box_level,
            base,
            shift,
            compute_method,
            self.get_kernel_function_type(dim, kernel_type),
            empty_float_blob,
            empty_float_blob,
            None,
            None,
            None,
            None,
            payload_blob,
        )

        t_db_write_start = time.perf_counter()
        self.datafile.execute(
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
            ON CONFLICT(dim, kernel_type, q_order, source_box_level) DO UPDATE SET
                n_q_points=excluded.n_q_points,
                quad_order=excluded.quad_order,
                n_pairs=excluded.n_pairs,
                source_box_extent=excluded.source_box_extent,
                source_box_level_stored=excluded.source_box_level_stored,
                case_encoding_base=excluded.case_encoding_base,
                case_encoding_shift=excluded.case_encoding_shift,
                build_method=excluded.build_method,
                kernel_type_cached=excluded.kernel_type_cached,
                q_points=excluded.q_points,
                data=excluded.data,
                mode_normalizers=excluded.mode_normalizers,
                kernel_exterior_normalizers=excluded.kernel_exterior_normalizers,
                interaction_case_vecs=excluded.interaction_case_vecs,
                case_indices=excluded.case_indices,
                payload=excluded.payload
            """,
            record_values,
        )

        self._store_record_kwargs(dim, kernel_type, q_order, source_box_level, kwargs)

        self.datafile.commit()
        t_db_write_end = time.perf_counter()

        self.last_compute_timings = {
            "table_build_s": t_compute_end - t_compute_start,
            "payload_serialize_s": t_payload_serialize_end - t_payload_serialize_start,
            "db_write_commit_s": t_db_write_end - t_db_write_start,
            "total_s": t_db_write_end - t_compute_start,
            "payload_bytes": len(payload_blob),
        }

        return table


# }}} End table dataset manager class

# vim: filetype=pyopencl:foldmethod=marker
