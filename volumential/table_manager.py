__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__doc__ = """
.. autoclass:: TableRequest
   :members:

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
import json
import re
import sqlite3
import time
import zipfile
from dataclasses import dataclass, is_dataclass
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
_TABLE_BUILD_METHOD = "DuffyRadial"


@dataclass(frozen=True)
class KernelSpec:
    dim: int
    kernel_type: str

    @classmethod
    def from_args(cls, dim, kernel_type):
        dim = int(dim)
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")

        return cls(dim=dim, kernel_type=kernel_type)


@dataclass(frozen=True)
class TableDiscretization:
    q_order: int
    source_box_level: int = 0

    @classmethod
    def from_args(cls, q_order, source_box_level=0):
        q_order = int(q_order)
        if q_order < 1:
            raise ValueError(f"q_order must be >= 1, got {q_order}")

        source_box_level = int(source_box_level)
        if source_box_level < 0:
            raise ValueError(f"source_box_level must be >= 0, got {source_box_level}")

        return cls(
            q_order=q_order,
            source_box_level=source_box_level,
        )


@dataclass(frozen=True)
class TableRequest:
    kernel: KernelSpec
    discretization: TableDiscretization

    @classmethod
    def from_args(cls, dim, kernel_type, q_order, source_box_level=0):
        return cls(
            kernel=KernelSpec.from_args(dim=dim, kernel_type=kernel_type),
            discretization=TableDiscretization.from_args(
                q_order=q_order,
                source_box_level=source_box_level,
            ),
        )

    @property
    def dim(self):
        return self.kernel.dim

    @property
    def kernel_type(self):
        return self.kernel.kernel_type

    @property
    def q_order(self):
        return self.discretization.q_order

    @property
    def source_box_level(self):
        return self.discretization.source_box_level


@dataclass(frozen=True)
class TableKernelBundle:
    kernel_func: object
    kernel_scale_type: object
    sumpy_kernel: object


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


def _to_stable_jsonable(value):
    if isinstance(value, np.generic):
        return _to_stable_jsonable(value.item())

    if isinstance(value, np.ndarray):
        return _to_stable_jsonable(value.tolist())

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if is_dataclass(value) and not isinstance(value, type):
        return _to_stable_jsonable(
            {name: getattr(value, name) for name in value.__dataclass_fields__}
        )

    if isinstance(value, (list, tuple)):
        return [_to_stable_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _to_stable_jsonable(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    raise TypeError(f"unsupported value type for stable serialization: {type(value)!r}")


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


def _coerce_sqlite_int(value, field_name):
    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, str):
        return int(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)

        # Legacy rows could contain numpy scalar bytes if sqlite adapters
        # treated numpy integers as blobs.
        if len(raw) == 4:
            return int(np.frombuffer(raw, dtype=np.int32)[0])

        if len(raw) == 8:
            return int(np.frombuffer(raw, dtype=np.int64)[0])

        return int(raw.decode("ascii"))

    raise TypeError(f"unsupported integer value for {field_name}: {type(value)!r}")


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


# {{{ cahn-hilliard sumpy kernels


def _extract_cahn_hilliard_coefficients(kwargs, where):
    has_b = "b" in kwargs
    has_c = "c" in kwargs
    has_approx = "approx_at_origin" in kwargs

    if has_b != has_c:
        raise TypeError(f"{where}: Cahn-Hilliard requires both b and c together")

    if has_approx and not (has_b and has_c):
        raise TypeError(
            f"{where}: Cahn-Hilliard requires both b and c when approx_at_origin "
            "is specified"
        )

    if not (has_b and has_c):
        return None

    if kwargs.get("approx_at_origin", False):
        raise TypeError(
            f"{where}: approx_at_origin is unsupported for sumpy Cahn-Hilliard kernels"
        )

    b = kwargs["b"]
    c = kwargs["c"]
    if isinstance(b, np.generic):
        b = b.item()
    if isinstance(c, np.generic):
        c = c.item()

    if not np.isscalar(b) or not np.isscalar(c):
        raise TypeError(f"{where}: Cahn-Hilliard coefficients must be scalars")

    if isinstance(b, (bytes, bytearray, memoryview)) or isinstance(
        c, (bytes, bytearray, memoryview)
    ):
        raise TypeError(f"{where}: Cahn-Hilliard coefficients must be numeric scalars")

    try:
        b = complex(b)
        c = complex(c)
    except Exception as exc:
        raise TypeError(
            f"{where}: failed to parse Cahn-Hilliard coefficients b/c as scalars"
        ) from exc

    return b, c


def _compute_cahn_hilliard_lambdas(b, c):
    roots = np.roots(np.array([1.0, -b, c], dtype=np.complex128))
    lambdas = [np.lib.scimath.sqrt(root) for root in roots]
    lambdas.sort(key=abs, reverse=True)

    lam1, lam2 = lambdas
    if abs(lam1**2 - lam2**2) < 1e-15:
        raise ValueError(
            "degenerate Cahn-Hilliard coefficients: expected distinct roots"
        )

    return lam1, lam2


class CahnHilliardKernel(ExpressionKernel):
    init_arg_names = ("dim", "b", "c")

    def __init__(self, dim: int | None = None, b: complex = 0j, c: complex = 0j):
        if dim != 2:
            raise NotImplementedError(
                f"Cahn-Hilliard sumpy kernel only supports dim=2 (got {dim})"
            )

        from pymbolic import var
        from pymbolic.primitives import make_sym_vector
        from sumpy.symbolic import pymbolic_real_norm_2

        lam1, lam2 = _compute_cahn_hilliard_lambdas(b, c)
        denom = lam1**2 - lam2**2

        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = var("hankel_1")(0, var("I") * lam1 * r) - var("hankel_1")(
            0, var("I") * lam2 * r
        )
        scaling_for_k0 = var("pi") / 2 * var("I")
        scaling = -scaling_for_k0 / (2 * var("pi") * denom)

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)

    @property
    def is_complex_valued(self):
        return True

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import register_bessel_callables

        return register_bessel_callables(loopy_knl)

    def __getinitargs__(self):
        return (self.dim, self.b, self.c)

    def __repr__(self):
        return f"CahnHilliardKnl{self.dim}D(b={self.b}, c={self.c})"

    mapper_method = "map_expression_kernel"


# }}} End cahn-hilliard sumpy kernels

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

    def _record_exists(self, table_request):
        row = self.datafile.execute(
            "SELECT 1 FROM nearfield_cache WHERE dim=? AND kernel_type=? "
            "AND q_order=? AND source_box_level=?",
            (
                table_request.dim,
                table_request.kernel_type,
                table_request.q_order,
                table_request.source_box_level,
            ),
        ).fetchone()

        return row is not None

    def _load_record(self, table_request):
        return self.datafile.execute(
            "SELECT * FROM nearfield_cache WHERE dim=? AND kernel_type=? "
            "AND q_order=? AND source_box_level=?",
            (
                table_request.dim,
                table_request.kernel_type,
                table_request.q_order,
                table_request.source_box_level,
            ),
        ).fetchone()

    def _load_record_kwargs(self, table_request):
        rows = self.datafile.execute(
            "SELECT key, value_type, value_text FROM nearfield_cache_kwargs "
            "WHERE dim=? AND kernel_type=? AND q_order=? AND source_box_level=?",
            (
                table_request.dim,
                table_request.kernel_type,
                table_request.q_order,
                table_request.source_box_level,
            ),
        ).fetchall()

        kwargs = {}
        for row in rows:
            kwargs[row["key"]] = _deserialize_scalar(
                row["value_type"], row["value_text"]
            )

        return kwargs

    def _store_record_kwargs(self, table_request, kwargs):
        self.datafile.execute(
            "DELETE FROM nearfield_cache_kwargs WHERE dim=? AND kernel_type=? "
            "AND q_order=? AND source_box_level=?",
            (
                table_request.dim,
                table_request.kernel_type,
                table_request.q_order,
                table_request.source_box_level,
            ),
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
                    table_request.dim,
                    table_request.kernel_type,
                    table_request.q_order,
                    table_request.source_box_level,
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

    def _normalize_table_request(self, dim, kernel_type, q_order, source_box_level=0):
        return TableRequest.from_args(
            dim=dim,
            kernel_type=kernel_type,
            q_order=q_order,
            source_box_level=source_box_level,
        )

    def _coerce_table_request(self, table_request):
        if not isinstance(table_request, TableRequest):
            raise TypeError("table_request must be a TableRequest")

        return self._normalize_table_request(
            dim=table_request.dim,
            kernel_type=table_request.kernel_type,
            q_order=table_request.q_order,
            source_box_level=table_request.source_box_level,
        )

    def _source_box_extent_for_level(self, source_box_level):
        return self.root_extent * (2 ** (-source_box_level))

    def _reject_removed_compute_method_kwarg(self, kwargs, where):
        if "compute_method" in kwargs:
            raise TypeError(
                "compute_method has been removed; DuffyRadial is used for all "
                f"table builds ({where})"
            )

    def _build_config_fingerprint(self, kwargs):
        build_config = kwargs.get("build_config")
        if build_config is None:
            return None

        if not (is_dataclass(build_config) and not isinstance(build_config, type)):
            raise TypeError("build_config must be a dataclass instance")

        return json.dumps(
            _to_stable_jsonable(build_config),
            sort_keys=True,
            separators=(",", ":"),
        )

    def _reject_removed_top_level_duffy_knobs(self, kwargs, where):
        removed_knobs = (
            "radial_rule",
            "regular_quad_order",
            "radial_quad_order",
            "deg_theta",
            "mp_dps",
            "auto_tune_orders",
            "auto_tune_samples",
            "auto_tune_floor_factor",
            "auto_tune_candidates",
        )
        specified = [key for key in removed_knobs if key in kwargs]
        if specified:
            raise TypeError(
                "top-level Duffy knobs have been removed; pass a "
                "nearfield_potential_table.DuffyBuildConfig as build_config "
                f"({where}; got {', '.join(specified)})"
            )

    def _reject_removed_knl_func_kwarg(self, kwargs, where):
        if "knl_func" in kwargs:
            raise TypeError(
                f"knl_func has been removed; pass sumpy_knl instead ({where})"
            )

    def _kwargs_for_cache_storage(self, kwargs):
        cache_kwargs = dict(kwargs)
        build_config_fingerprint = self._build_config_fingerprint(kwargs)
        if build_config_fingerprint is not None:
            cache_kwargs["build_config_fingerprint"] = build_config_fingerprint
            cache_kwargs["build_config_json"] = build_config_fingerprint
            cache_kwargs.pop("build_config", None)

        return cache_kwargs

    def _deserialize_build_config(self, build_config_json):
        from volumential.nearfield_potential_table import DuffyBuildConfig

        decoded = json.loads(build_config_json)
        if not isinstance(decoded, dict):
            raise ValueError("build_config_json must decode to an object")

        return DuffyBuildConfig(**decoded)

    def _kwargs_with_cached_build_config(self, table_request, kwargs):
        if "build_config" in kwargs:
            return kwargs

        loaded_kwargs = self._load_record_kwargs(table_request)
        build_config_json = loaded_kwargs.get("build_config_json")
        if build_config_json is None:
            build_config_json = loaded_kwargs.get("build_config_fingerprint")
        if build_config_json is None:
            return kwargs

        try:
            cached_build_config = self._deserialize_build_config(build_config_json)
        except Exception:
            logger.warning("Ignoring malformed cached build_config_json")
            return kwargs

        updated_kwargs = dict(kwargs)
        updated_kwargs["build_config"] = cached_build_config
        return updated_kwargs

    def _resolve_kernel_bundle(self, table_request, kwargs, require_sumpy_kernel):
        self._reject_removed_knl_func_kwarg(kwargs, "_resolve_kernel_bundle")

        ch_kwargs = {
            key: kwargs[key] for key in ("b", "c", "approx_at_origin") if key in kwargs
        }

        if "sumpy_knl" in kwargs:
            if ch_kwargs:
                raise TypeError(
                    "_resolve_kernel_bundle cannot mix sumpy_knl with "
                    "Cahn-Hilliard coefficients b/c"
                )
            sumpy_knl = kwargs["sumpy_knl"]
        else:
            try:
                sumpy_knl = self.get_sumpy_kernel(
                    table_request.dim,
                    table_request.kernel_type,
                    **ch_kwargs,
                )
            except NotImplementedError:
                sumpy_knl = None

        kernel_scale_type = self.get_kernel_function_type(
            table_request.dim,
            table_request.kernel_type,
        )

        if require_sumpy_kernel and sumpy_knl is None:
            raise RuntimeError(
                "DuffyRadial table builder requires a sumpy kernel; "
                f"kernel_type={table_request.kernel_type!r} is unsupported "
                "for loopy table build."
            )

        return TableKernelBundle(
            kernel_func=None,
            kernel_scale_type=kernel_scale_type,
            sumpy_kernel=sumpy_knl,
        )

    def _warn_on_loaded_kwarg_mismatch(self, table, kwargs):
        for kkey, kval in kwargs.items():
            if kval is not None:
                if kkey == "build_config":
                    continue

                try:
                    tbval = getattr(table, kkey)
                    if isinstance(kval, (bool, int, str)):
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
                    elif isinstance(kval, (float, complex)):
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
                    else:
                        continue

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
                            "arguments being passed, since only "
                            "(int, float, complex, bool, str) "
                            "are stored in the cache. "
                            "Also, some parameters related to method for "
                            "table building are not critical for "
                            "consistency."
                        )
                        print(e)

    def get_table(
        self,
        dim,
        kernel_type,
        q_order,
        source_box_level=0,
        force_recompute=False,
        queue=None,
        **kwargs,
    ):
        """Primary user interface. Get or build a cached table."""
        table_request = self._normalize_table_request(
            dim=dim,
            kernel_type=kernel_type,
            q_order=q_order,
            source_box_level=source_box_level,
        )

        return self.get_table_from_request(
            table_request,
            force_recompute=force_recompute,
            queue=queue,
            **kwargs,
        )

    def get_table_from_request(
        self,
        table_request,
        force_recompute=False,
        queue=None,
        **kwargs,
    ):
        """Get or build a table using a :class:`volumential.table_manager.TableRequest`."""

        t_get_start = time.perf_counter()
        request_kwargs = dict(kwargs)
        self._reject_removed_compute_method_kwarg(
            request_kwargs,
            "get_table_from_request",
        )
        self._reject_removed_top_level_duffy_knobs(
            request_kwargs,
            "get_table_from_request",
        )
        self._reject_removed_knl_func_kwarg(
            request_kwargs,
            "get_table_from_request",
        )

        table_request = self._coerce_table_request(table_request)

        is_recomputed = False

        if not self._record_exists(table_request):
            if self._read_only:
                raise RuntimeError(
                    "Table cache miss in read-only mode for "
                    f"(dim={table_request.dim}, "
                    f"kernel_type={table_request.kernel_type}, "
                    f"q_order={table_request.q_order}, "
                    f"source_box_level={table_request.source_box_level})."
                )
            logger.info("Table cache missing. Invoking fresh computation.")
            is_recomputed = True
            table = self._compute_and_update_table_for_request(
                table_request,
                queue=queue,
                **request_kwargs,
            )

        elif force_recompute:
            if self._read_only:
                raise RuntimeError("force_recompute is not supported in read-only mode")

            logger.info("Invoking fresh computation since force_recompute is set")
            is_recomputed = True
            recompute_kwargs = self._kwargs_with_cached_build_config(
                table_request,
                request_kwargs,
            )
            table = self._compute_and_update_table_for_request(
                table_request,
                queue=queue,
                **recompute_kwargs,
            )

        else:
            try:
                table = self._load_saved_table_for_request(
                    table_request,
                    **request_kwargs,
                )

            except KeyError:
                import traceback

                logger.debug(traceback.format_exc())

                if self._read_only:
                    raise RuntimeError(
                        "Cached table data is unavailable in read-only mode and "
                        "cannot be recomputed."
                    )

                logger.info("Recomputing due to cache miss/corruption.")
                is_recomputed = True
                recompute_kwargs = self._kwargs_with_cached_build_config(
                    table_request,
                    request_kwargs,
                )
                table = self._compute_and_update_table_for_request(
                    table_request,
                    queue=queue,
                    **recompute_kwargs,
                )

            self._warn_on_loaded_kwarg_mismatch(table, request_kwargs)

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
        **kwargs,
    ):
        """Load a table saved in the SQLite cache."""
        table_request = self._normalize_table_request(
            dim=dim,
            kernel_type=kernel_type,
            q_order=q_order,
            source_box_level=source_box_level,
        )

        return self.load_saved_table_from_request(table_request, **kwargs)

    def load_saved_table_from_request(self, table_request, **kwargs):
        """Load a cached table using a :class:`volumential.table_manager.TableRequest`."""

        request_kwargs = dict(kwargs)
        self._reject_removed_compute_method_kwarg(
            request_kwargs,
            "load_saved_table_from_request",
        )
        self._reject_removed_top_level_duffy_knobs(
            request_kwargs,
            "load_saved_table_from_request",
        )
        self._reject_removed_knl_func_kwarg(
            request_kwargs,
            "load_saved_table_from_request",
        )
        table_request = self._coerce_table_request(table_request)

        return self._load_saved_table_for_request(table_request, **request_kwargs)

    def _load_saved_table_for_request(self, table_request, **kwargs):
        t_load_start = time.perf_counter()

        record = self._load_record(table_request)
        t_record_fetch_end = time.perf_counter()
        if record is None:
            raise KeyError("missing table record")

        if table_request.dim != record["dim"]:
            raise AssertionError("cache record dimension mismatch")
        if table_request.q_order != record["quad_order"]:
            raise AssertionError("cache record quad order mismatch")

        stored_build_method = record["build_method"]
        if (
            stored_build_method is not None
            and stored_build_method != _TABLE_BUILD_METHOD
        ):
            raise KeyError(
                "cached build_method is unsupported; expected " + _TABLE_BUILD_METHOD
            )

        kernel_bundle = self._resolve_kernel_bundle(
            table_request,
            kwargs,
            require_sumpy_kernel=False,
        )

        payload_blob = record["payload"] if "payload" in record.keys() else None
        used_payload = bool(payload_blob)
        t_payload_deser_start = time.perf_counter()
        try:
            if used_payload:
                payload = _deserialize_table_payload(payload_blob)
            else:
                raise KeyError("table cache payload is missing")

            t_payload_deser_end = time.perf_counter()

            precomputed_q_points = None
            if "q_points" in payload:
                precomputed_q_points = payload["q_points"]

            table_extra_kwargs = dict(self.table_extra_kwargs)
            table_extra_kwargs.pop("precomputed_q_points", None)

            table = NearFieldInteractionTable(
                quad_order=table_request.q_order,
                dim=table_request.dim,
                dtype=self.dtype,
                build_method=_TABLE_BUILD_METHOD,
                kernel_func=kernel_bundle.kernel_func,
                kernel_type=kernel_bundle.kernel_scale_type,
                sumpy_kernel=kernel_bundle.sumpy_kernel,
                derive_kernel_func=False,
                source_box_extent=self._source_box_extent_for_level(
                    table_request.source_box_level
                ),
                precomputed_q_points=precomputed_q_points,
                **table_extra_kwargs,
            )

            assert abs(table.source_box_extent - record["source_box_extent"]) < 1e-15
            assert table_request.source_box_level == record["source_box_level_stored"]

            table.q_points[:] = payload["q_points"]
            if "data" in payload:
                table.data[:] = payload["data"]
            elif "reduced_entry_ids" in payload and "reduced_data" in payload:
                table.data[:] = np.nan
                table.data[payload["reduced_entry_ids"]] = payload["reduced_data"]
            else:
                raise KeyError("payload is missing table data arrays")

            table.mode_normalizers[:] = payload["mode_normalizers"]
            table.kernel_exterior_normalizers[:] = payload[
                "kernel_exterior_normalizers"
            ]

            tmp_case_vecs = np.array(table.interaction_case_vecs)
            tmp_case_vecs[...] = payload["interaction_case_vecs"]
            table.interaction_case_vecs = [list(vec) for vec in tmp_case_vecs]

            table.case_indices[:] = payload["case_indices"]
            if "table_data_is_symmetry_reduced" in payload:
                table.table_data_is_symmetry_reduced = bool(
                    payload["table_data_is_symmetry_reduced"][0]
                )

        except KeyError:
            raise
        except (OSError, EOFError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            raise KeyError("table cache payload is corrupted") from exc

        assert table.n_q_points == record["n_q_points"]
        assert table.n_pairs == record["n_pairs"]
        assert table.quad_order == record["quad_order"]

        base = _coerce_sqlite_int(
            record["case_encoding_base"],
            field_name="case_encoding_base",
        )
        shift = _coerce_sqlite_int(
            record["case_encoding_shift"],
            field_name="case_encoding_shift",
        )

        def case_encode(case_vec):
            table_id = 0
            for case_vec_comp in case_vec:
                table_id = table_id * base + (case_vec_comp + shift)
            return int(table_id)

        table.case_encode = case_encode

        table.source_box_level = table_request.source_box_level
        table.dim = record["dim"]
        table.n_q_points = record["n_q_points"]
        table.n_pairs = record["n_pairs"]
        table.case_encoding_base = base
        table.case_encoding_shift = shift
        table.build_method = _TABLE_BUILD_METHOD
        table.kernel_type_cached = record["kernel_type_cached"]
        table.source_box_extent = record["source_box_extent"]

        t_kwargs_load_start = time.perf_counter()
        loaded_kwargs = self._load_record_kwargs(table_request)
        requested_build_config_fingerprint = self._build_config_fingerprint(kwargs)
        if requested_build_config_fingerprint is not None:
            loaded_build_config_fingerprint = loaded_kwargs.get(
                "build_config_fingerprint"
            )
            if loaded_build_config_fingerprint != requested_build_config_fingerprint:
                raise KeyError("cached build_config mismatch")

        for atkey, atval in loaded_kwargs.items():
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
        """Return a numerical kernel callable derived from a sumpy kernel."""
        allowed = {"sumpy_knl", "b", "c", "approx_at_origin"}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(
                "get_kernel_function only accepts sumpy_knl, b, c, and "
                f"approx_at_origin; got {', '.join(unknown)}"
            )

        ch_kwargs = {
            key: kwargs[key] for key in ("b", "c", "approx_at_origin") if key in kwargs
        }

        sumpy_knl = kwargs.get("sumpy_knl")
        if sumpy_knl is not None and ch_kwargs:
            raise TypeError(
                "get_kernel_function cannot mix sumpy_knl with Cahn-Hilliard "
                "coefficients b/c"
            )

        if sumpy_knl is None:
            sumpy_knl = self.get_sumpy_kernel(dim, kernel_type, **ch_kwargs)

        if sumpy_knl is None:
            raise RuntimeError(
                "Kernel function derivation requires a sumpy kernel; "
                f"kernel_type={kernel_type!r} is unsupported."
            )

        return vm.nearfield_potential_table.sumpy_kernel_to_lambda(sumpy_knl)

    def get_sumpy_kernel(self, dim, kernel_type, **kwargs):
        """Sumpy (symbolic) version of the kernel."""

        allowed = {"b", "c", "approx_at_origin"}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(
                "get_sumpy_kernel only accepts optional Cahn-Hilliard "
                f"kwargs b/c/approx_at_origin; got {', '.join(unknown)}"
            )

        ch_kwargs = {
            key: kwargs[key] for key in ("b", "c", "approx_at_origin") if key in kwargs
        }
        has_ch_coeffs = bool(ch_kwargs)
        if has_ch_coeffs and not kernel_type.startswith("Cahn-Hilliard"):
            raise TypeError(
                "Cahn-Hilliard coefficients b/c are only valid when "
                "kernel_type starts with 'Cahn-Hilliard'"
            )

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

        elif kernel_type.startswith("Cahn-Hilliard"):
            from sumpy.kernel import AxisTargetDerivative

            coeffs = _extract_cahn_hilliard_coefficients(ch_kwargs, "get_sumpy_kernel")
            if coeffs is None:
                try:
                    from sumpy.kernel import FactorizedBiharmonicKernel

                    base_knl = FactorizedBiharmonicKernel(dim)
                except (ImportError, AttributeError) as exc:
                    raise TypeError(
                        "Cahn-Hilliard kernels require both b and c when "
                        "FactorizedBiharmonicKernel is unavailable in sumpy"
                    ) from exc
            else:
                base_knl = CahnHilliardKernel(dim, b=coeffs[0], c=coeffs[1])

            if kernel_type == "Cahn-Hilliard":
                return base_knl

            if kernel_type == "Cahn-Hilliard-Laplacian":
                from sumpy.kernel import LaplacianTargetDerivative

                return LaplacianTargetDerivative(base_knl)

            if kernel_type == "Cahn-Hilliard-Dx":
                return AxisTargetDerivative(0, base_knl)

            if kernel_type == "Cahn-Hilliard-Laplacian-Dx":
                from sumpy.kernel import LaplacianTargetDerivative

                return AxisTargetDerivative(0, LaplacianTargetDerivative(base_knl))

            if kernel_type == "Cahn-Hilliard-Laplacian-Dy":
                from sumpy.kernel import LaplacianTargetDerivative

                return AxisTargetDerivative(1, LaplacianTargetDerivative(base_knl))

            if kernel_type == "Cahn-Hilliard-Dy":
                return AxisTargetDerivative(1, base_knl)

            raise NotImplementedError("Kernel type not supported.")

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
        cl_ctx=None,
        queue=None,
        **kwargs,
    ):
        """Performs the precomputation and stores the results."""
        table_request = self._normalize_table_request(
            dim=dim,
            kernel_type=kernel_type,
            q_order=q_order,
            source_box_level=source_box_level,
        )

        return self.compute_and_update_table_for_request(
            table_request,
            cl_ctx=cl_ctx,
            queue=queue,
            **kwargs,
        )

    def compute_and_update_table_for_request(
        self,
        table_request,
        cl_ctx=None,
        queue=None,
        **kwargs,
    ):
        """Build/update a cached table using a :class:`volumential.table_manager.TableRequest`."""

        request_kwargs = dict(kwargs)
        self._reject_removed_compute_method_kwarg(
            request_kwargs,
            "compute_and_update_table_for_request",
        )
        self._reject_removed_top_level_duffy_knobs(
            request_kwargs,
            "compute_and_update_table_for_request",
        )
        table_request = self._coerce_table_request(table_request)

        return self._compute_and_update_table_for_request(
            table_request,
            cl_ctx=cl_ctx,
            queue=queue,
            **request_kwargs,
        )

    def _compute_and_update_table_for_request(
        self,
        table_request,
        cl_ctx=None,
        queue=None,
        **kwargs,
    ):
        """Performs the precomputation and stores the results."""

        kernel_bundle = self._resolve_kernel_bundle(
            table_request,
            kwargs,
            require_sumpy_kernel=True,
        )

        logger.debug("Start computing interaction table.")
        table = NearFieldInteractionTable(
            dim=table_request.dim,
            quad_order=table_request.q_order,
            dtype=self.dtype,
            kernel_func=kernel_bundle.kernel_func,
            kernel_type=kernel_bundle.kernel_scale_type,
            sumpy_kernel=kernel_bundle.sumpy_kernel,
            derive_kernel_func=False,
            build_method=_TABLE_BUILD_METHOD,
            source_box_extent=self._source_box_extent_for_level(
                table_request.source_box_level
            ),
            **self.table_extra_kwargs,
        )

        if 0:
            if "delta" in kwargs:
                delta = kwargs.pop("delta") * (2 ** (-table_request.source_box_level))
                kwargs["delta"] = delta

        t_compute_start = time.perf_counter()
        table.build_table(cl_ctx=cl_ctx, queue=queue, **kwargs)
        t_compute_end = time.perf_counter()
        assert table.is_built

        logger.debug("Start updating database.")

        source_box_extent = self._source_box_extent_for_level(
            table_request.source_box_level
        )
        t_payload_serialize_start = time.perf_counter()
        payload_blob = _serialize_table_payload(table)
        empty_float_blob = _serialize_array(np.array([], dtype=self.dtype))
        t_payload_serialize_end = time.perf_counter()

        distinct_numbers = set()
        for vec in table.interaction_case_vecs:
            for case_vec_comp in vec:
                distinct_numbers.add(case_vec_comp)
        base = int(len(range(min(distinct_numbers), max(distinct_numbers) + 1)))
        shift = int(-min(distinct_numbers))

        record_values = (
            table_request.dim,
            table_request.kernel_type,
            table_request.q_order,
            table_request.source_box_level,
            table.n_q_points,
            table.quad_order,
            table.n_pairs,
            source_box_extent,
            table_request.source_box_level,
            base,
            shift,
            _TABLE_BUILD_METHOD,
            kernel_bundle.kernel_scale_type,
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

        self._store_record_kwargs(
            table_request,
            self._kwargs_for_cache_storage(kwargs),
        )

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
