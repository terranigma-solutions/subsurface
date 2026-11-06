"""Full validation of native .le files submitted for LiquidEarth ingestion.

``inspect_le`` reads headers only. This module reads a whole file, or the whole
set of files that makes one dataset, and checks it against the profile that a
LiquidEarth content family accepts: the binary layout, every payload value that
the layout constrains, and what the LiquidEarth (Unity) loaders can read. It
never changes or re-serializes the bytes, so a file that passes can be published
as it is.

Array order is not recorded in a .le file. The only accepted encoding,
``le-fortran-little-endian-v1``, is the public subsurface writers' default:
Fortran (column-major) arrays, little-endian. The caller declares it; plausible
values cannot prove it.

Errors carry one of four codes:

- ``malformed``: the bytes are not a valid .le file (or PNG).
- ``unsupported_profile``: a valid file outside what the family accepts.
- ``family_mismatch``: a valid dataset of another family.
- ``inconsistent_dataset``: files or attributes that do not agree.
"""

import json
import math
import mmap
import os
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np

from subsurface.core.structs.base_structures._liquid_earth_mesh import validate_unstructured_layout
from subsurface.core.utils._le_json import validate_le_json_nesting

LE_INGEST_ENCODING = "le-fortran-little-endian-v1"
MAX_INGEST_HEADER_BYTES = 1024 * 1024
MAX_INGEST_HEADER_DEPTH = 32
MAX_ATTRIBUTE_COLUMNS = 256
# Unity splits a static mesh into one object per run of equal (int) attribute values.
MAX_MESH_ATTRIBUTE_RUNS = 1024
MAX_TEXTURE_BYTES = 16 * 1024 * 1024
MAX_TEXTURE_DIMENSION = 4096
MAX_TEXTURE_PIXELS = 4 * 1024 * 1024
MAX_PNG_CHUNKS = 4096

ERROR_CODES = ("malformed", "unsupported_profile", "family_mismatch", "inconsistent_dataset")

# Roles of each family, in order: (role, required).
FAMILY_ROLES: Dict[str, Tuple[Tuple[str, bool], ...]] = {
        "mesh"      : (("mesh", True), ("texture", False)),
        "volume"    : (("volume", True),),
        "raster"    : (("raster", True),),
        "pointcloud": (("pointcloud", True),),
        "wells"     : (("collars", True), ("trajectory", True)),
}

_CHUNK = 1 << 22
_INT32_MIN, _INT32_MAX = -2 ** 31, 2 ** 31 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_UNSTRUCTURED_KEYS = frozenset({"format_version", "vertex_shape", "cell_shape", "cell_attrs", "vertex_attrs", "xarray_attrs"})
_STRUCTURED_KEYS = frozenset({"data_shape", "bounds", "transform", "dtype", "data_name"})
_COLUMN_KEYS = frozenset({"name", "dtype", "shape", "byte_length"})
# Exactly the dtype spellings that Unity's LiquidEarthMesh.ReadAttributeColumn reads.
_ATTRIBUTE_DTYPES = {
        "float32": np.dtype("<f4"), "float64": np.dtype("<f8"),
        "int32"  : np.dtype("<i4"), "int64": np.dtype("<i8"), "bool": np.dtype("u1"),
}
_TEMPORAL_KEYS = frozenset({"time_series_id", "timestamp", "time_series"})
_MASTER_KEYS = frozenset({"isMaster", "slaveResources"})
_COLOR_CHANNELS = {"rgb": 3, "rgba": 4}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

Source = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview]


class LEValidationError(ValueError):
    """A file or dataset outside the ingestion profile; ``code`` is one of ERROR_CODES."""

    def __init__(self, code: str, message: str, *, role: Optional[str] = None, detected_family: Optional[str] = None):
        if code not in ERROR_CODES:
            raise ValueError(f"Unknown LE validation code: {code!r}")
        super().__init__(message)
        self.code = code
        self.role = role
        self.detected_family = detected_family


@dataclass(frozen=True)
class LEAttribute:
    name: str
    dtype: str
    association: str  # "cell" or "vertex"


@dataclass(frozen=True)
class LEFileReport:
    """What a validated .le file holds; only counts, shapes and names, never values."""

    role: str
    kind: str  # "unstructured" or "structured"
    byte_size: int
    header_bytes: int
    vertex_count: Optional[int] = None
    cell_count: Optional[int] = None
    cell_width: Optional[int] = None
    attributes: Tuple[LEAttribute, ...] = ()
    data_shape: Optional[Tuple[int, ...]] = None
    data_name: Optional[str] = None
    channels: Optional[int] = None
    well_count: Optional[int] = None
    metadata_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PNGTextureReport:
    role: str
    byte_size: int
    width: int
    height: int
    channels: int


@dataclass(frozen=True)
class LEValidationReport:
    family: str
    encoding: str
    files: Mapping[str, Union[LEFileReport, PNGTextureReport]]
    well_count: Optional[int] = None


def validate_le(source: Source, *, family: str, role: Optional[str] = None,
                encoding: str = LE_INGEST_ENCODING) -> LEFileReport:
    """Validate one .le file for one role of ``family`` (default: its first role)."""
    roles = _roles(family)
    role = roles[0][0] if role is None else role
    if role not in dict(roles) or role == "texture":
        raise ValueError(f"{role!r} is not a .le role of {family!r}")
    _check_encoding(encoding, role)
    with _open(source) as buffer:
        return _validate_file(buffer, family, role)[0]


def validate_png_texture(source: Source, *, role: str = "texture") -> PNGTextureReport:
    """Validate a mesh texture: a complete non-interlaced 8-bit RGB or RGBA PNG."""
    with _open(source) as buffer:
        return _validate_png(buffer, role)


def validate_le_dataset(family: str, sources: Mapping[str, Source], *,
                        encoding: str = LE_INGEST_ENCODING) -> LEValidationReport:
    """Validate every file of one dataset and how they fit together.

    ``sources`` maps each role (see ``FAMILY_ROLES``) to a path or bytes. A
    mesh may have a ``texture`` PNG; wells need ``collars`` and ``trajectory``.
    """
    roles = _roles(family)
    known = dict(roles)
    for role in sources:
        if role not in known:
            raise LEValidationError("inconsistent_dataset", f"{family} datasets have no {role!r} file", role=role)
    for role, required in roles:
        if required and role not in sources:
            raise LEValidationError("inconsistent_dataset", f"{family} datasets need a {role} file", role=role)
    _check_encoding(encoding, None)
    reports, contexts = {}, {}
    for role, _ in roles:
        if role not in sources:
            continue
        with _open(sources[role]) as buffer:
            if role == "texture":
                reports[role] = _validate_png(buffer, role)
            else:
                reports[role], contexts[role] = _validate_file(buffer, family, role)
    well_count = None
    if family == "wells":
        well_count = reports["trajectory"].well_count
        if reports["collars"].vertex_count != well_count:
            raise LEValidationError(
                "inconsistent_dataset",
                f"collars has {reports['collars'].vertex_count} wells but trajectory has {well_count}",
                role="collars")
    if family == "mesh" and "texture" in reports and not contexts["mesh"]["uv"]:
        raise LEValidationError("inconsistent_dataset",
                                "a textured mesh needs finite numeric u and v vertex attributes", role="texture")
    return LEValidationReport(family=family, encoding=encoding, files=reports, well_count=well_count)


# region Files

def _roles(family):
    if family not in FAMILY_ROLES:
        raise ValueError(f"Unknown LiquidEarth family: {family!r}")
    return FAMILY_ROLES[family]


def _check_encoding(encoding, role):
    if encoding != LE_INGEST_ENCODING:
        raise LEValidationError("unsupported_profile", f"Unsupported .le encoding {encoding!r}; "
                                                       f"only {LE_INGEST_ENCODING} is accepted", role=role)


@contextmanager
def _open(source) -> Iterator[memoryview]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        yield memoryview(source).cast("B")
        return
    with open(os.fspath(source), "rb") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            yield memoryview(b"")
            return
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(mapped)
        try:
            yield view
        finally:
            try:
                view.release()
                mapped.close()
            except BufferError:
                pass  # An array view is still referenced (e.g. by a traceback); the map closes when it is freed.


def _fail(code, role, message, detected=None):
    raise LEValidationError(code, f"{role}: {message}", role=role, detected_family=detected)


def _header(buffer, role):
    if len(buffer) < 4:
        _fail("malformed", role, "file is too short for a .le length prefix")
    length = int.from_bytes(buffer[:4], "little")
    if length == 0:
        _fail("malformed", role, ".le header length is zero")
    if length > MAX_INGEST_HEADER_BYTES:
        _fail("unsupported_profile", role, f".le header is {length} bytes; at most {MAX_INGEST_HEADER_BYTES} are accepted")
    if len(buffer) < 4 + length:
        _fail("malformed", role, "file ends inside the .le header")
    raw = bytes(buffer[4:4 + length])
    try:
        validate_le_json_nesting(raw, max_depth=MAX_INGEST_HEADER_DEPTH)
    except ValueError:
        _fail("unsupported_profile", role, f".le header nests deeper than {MAX_INGEST_HEADER_DEPTH} levels")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("malformed", role, f"duplicate header key {key!r}")
            result[key] = value
        return result

    def constant(token):
        _fail("unsupported_profile", role, f"header contains the non-JSON number {token}")

    def number(token):
        value = float(token)
        if not math.isfinite(value):
            _fail("unsupported_profile", role, f"header number {token[:32]} overflows")
        return value

    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=constant, parse_float=number)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        _fail("malformed", role, ".le header is not valid UTF-8 JSON")
    if not isinstance(header, dict):
        _fail("malformed", role, ".le header must be a JSON object")
    return header, 4 + length


def _validate_file(buffer, family, role):
    header, offset = _header(buffer, role)
    if "data_shape" in header:
        if "vertex_shape" in header or "cell_shape" in header:
            _fail("malformed", role, "header mixes structured and unstructured fields")
        parsed = _structured(buffer, header, offset, role)
    elif "vertex_shape" in header or "cell_shape" in header:
        parsed = _unstructured(buffer, header, offset, role)
    else:
        _fail("malformed", role, "header is neither a structured nor an unstructured .le header")
    check = {
            "mesh"      : _mesh, "pointcloud": _pointcloud, "volume": _volume, "raster": _raster,
            "collars"   : _collars, "trajectory": _trajectory,
    }[role]
    report_fields, context = check(buffer, parsed, role)
    base = dict(role=role, kind=parsed.kind, byte_size=len(buffer), header_bytes=offset - 4)
    if parsed.kind == "unstructured":
        base.update(vertex_count=parsed.vertices, cell_count=parsed.rows, cell_width=parsed.width,
                    attributes=tuple(LEAttribute(c.name, c.dtype, c.association) for c in parsed.columns),
                    metadata_keys=tuple(sorted(parsed.metadata)))
    else:
        base.update(data_shape=parsed.shape, data_name=parsed.name, channels=parsed.channels)
    base.update(report_fields)
    return LEFileReport(**base), context


def _require(parsed, role, family, wanted):
    if parsed.family == wanted:
        return
    what = family if role == family else f"{family} {role}"
    if parsed.family is not None:
        _fail("family_mismatch", role, f"this is a {parsed.family} dataset, not a {what}", parsed.family)
    _fail("unsupported_profile", role, f"this .le file is not a supported {what}")

# endregion

# region Unstructured


@dataclass
class _Column:
    association: str
    name: str
    dtype: str
    offset: int
    count: int


@dataclass
class _Unstructured:
    kind: str
    vertices: int
    rows: int
    width: int
    vertex_offset: int
    cells_offset: int
    columns: List[_Column]
    metadata: Dict[str, Any]

    @property
    def family(self):
        if self.width == 3:
            return "mesh"
        if self.width in (0, 1):
            return "pointcloud"
        if self.width == 2 and "well_id_mapper" in self.metadata:
            return "wells"
        return None

    def column(self, association, name):
        return next((c for c in self.columns if c.association == association and c.name == name), None)


def _unstructured(buffer, header, offset, role):
    if "format_version" not in header:
        _fail("unsupported_profile", role, "legacy v1 .le files are not supported; re-export with a current subsurface writer")
    if type(header["format_version"]) is not int or header["format_version"] != 2:
        _fail("unsupported_profile", role, "unstructured .le files must have format_version 2")
    if "lods" in header:
        _fail("unsupported_profile", role, ".le files with LOD index buffers are not supported")
    extra = sorted(set(header) - _UNSTRUCTURED_KEYS)
    if extra:
        _fail("unsupported_profile", role, f"unsupported header fields: {', '.join(map(str, extra[:5]))}")
    missing = sorted(_UNSTRUCTURED_KEYS - set(header))
    if missing:
        _fail("malformed", role, f"missing header fields: {', '.join(missing)}")
    try:
        layout = validate_unstructured_layout(header, len(buffer) - offset)
    except (ValueError, TypeError, OverflowError) as error:
        _fail("malformed", role, str(error))
    vertices = layout["vertex_shape"][0]
    rows, width = layout["cell_shape"]
    if max(vertices, rows) > _INT32_MAX:
        _fail("unsupported_profile", role, "more vertices or cells than LiquidEarth can index")
    metadata = layout["data_attrs"]
    temporal = sorted(_TEMPORAL_KEYS & set(metadata))
    if temporal:
        _fail("unsupported_profile", role, "time-series .le files (snapshots and masters) are not supported")
    if _MASTER_KEYS & set(metadata):
        _fail("unsupported_profile", role, "master/slave .le resources are not supported")
    if "name" in metadata and not isinstance(metadata["name"], str):
        _fail("unsupported_profile", role, "xarray_attrs name must be a string")

    columns = []
    segments = iter(layout["segments"][2:])
    for association, key in (("cell", "cell_attrs"), ("vertex", "vertex_attrs")):
        declared = header[key]
        if len(declared) > MAX_ATTRIBUTE_COLUMNS:
            _fail("unsupported_profile", role, f"more than {MAX_ATTRIBUTE_COLUMNS} {association} attributes")
        for column in declared:
            segment = next(segments)
            if set(column) != _COLUMN_KEYS:
                _fail("unsupported_profile", role, f"{key} columns must have exactly name, dtype, shape and byte_length")
            name = column["name"]
            if not isinstance(name, str) or not name:
                _fail("unsupported_profile", role, f"{association} attribute names must be non-empty strings")
            if column["dtype"] not in _ATTRIBUTE_DTYPES:
                _fail("unsupported_profile", role,
                      f"attribute {name!r} has dtype {column['dtype']!r}; LiquidEarth reads float32, float64, int32, int64 and bool")
            columns.append(_Column(association, name, column["dtype"], offset + segment["offset"], segment["shape"][0]))

    parsed = _Unstructured("unstructured", vertices, rows, width, offset, offset + layout["segments"][1]["offset"],
                           columns, metadata)
    for _, chunk in _chunks(buffer, parsed.vertex_offset, np.dtype("<f4"), vertices * 3):
        if not np.isfinite(chunk).all():
            _fail("unsupported_profile", role, "vertex coordinates must be finite")
    for _, chunk in _chunks(buffer, parsed.cells_offset, np.dtype("<i4"), rows * width):
        if chunk.min() < 0 or chunk.max() >= vertices:
            _fail("malformed", role, "connectivity indices must be in range [0, vertex count)")
    for column in columns:
        dtype = _ATTRIBUTE_DTYPES[column.dtype]
        for _, chunk in _chunks(buffer, column.offset, dtype, column.count):
            if column.dtype == "bool":
                if chunk.max() > 1:
                    _fail("malformed", role, f"bool attribute {column.name!r} holds bytes other than 0 and 1")
            elif dtype.kind == "f":
                if np.isinf(chunk).any():
                    _fail("unsupported_profile", role, f"attribute {column.name!r} contains infinite values")
                if dtype.itemsize == 8 and (np.abs(chunk[np.isfinite(chunk)]) > _FLOAT32_MAX).any():
                    _fail("unsupported_profile", role, f"attribute {column.name!r} exceeds the float32 range LiquidEarth reads")
            elif column.dtype == "int64" and (chunk.min() < _INT32_MIN or chunk.max() > _INT32_MAX):
                _fail("unsupported_profile", role, f"attribute {column.name!r} has values outside the int32 range LiquidEarth reads")
    return parsed


def _mesh(buffer, parsed, role):
    _require(parsed, role, "mesh", "mesh")
    if parsed.rows < 1 or parsed.vertices < 3:
        _fail("unsupported_profile", role, "a mesh needs at least one triangle")
    cell_columns = [c for c in parsed.columns if c.association == "cell"]
    if cell_columns:
        _mesh_runs(buffer, parsed, role, cell_columns)
    uv = all(_finite_numeric(buffer, parsed.column("vertex", axis)) for axis in ("u", "v"))
    return {}, {"uv": uv}


def _mesh_runs(buffer, parsed, role, cell_columns):
    """Unity splits a mesh with cell attributes per run of each attribute's (int) value (MeshAnalysis)."""

    def check(column, values):
        runs = _runs(values)
        if runs > MAX_MESH_ATTRIBUTE_RUNS:
            _fail("unsupported_profile", role,
                  f"{column.association} attribute {column.name!r} changes value {runs} times along the mesh; "
                  f"LiquidEarth splits a mesh into one object per run, so at most {MAX_MESH_ATTRIBUTE_RUNS} runs are accepted")

    for column in cell_columns:
        check(column, _unity_ints(_whole(buffer, column)))
    vertex_columns = [c for c in parsed.columns if c.association == "vertex"]
    for column in vertex_columns:
        check(column, _unity_ints(_whole(buffer, column)))
    if vertex_columns:
        return
    # Without vertex attributes Unity derives one per cell attribute: each vertex takes the value of
    # the last cell (in row order) that references it, and unreferenced vertices stay 0.
    last = np.full(parsed.vertices, -1, dtype=np.int64)
    for k in range(parsed.width):
        for start, chunk in _chunks(buffer, parsed.cells_offset + k * parsed.rows * 4, np.dtype("<i4"), parsed.rows):
            np.maximum.at(last, chunk, (np.arange(start, start + chunk.size, dtype=np.int64) * parsed.width + k))
    referenced = last >= 0
    rows = last[referenced] // parsed.width
    for column in cell_columns:
        derived = np.zeros(parsed.vertices, dtype=np.int64)
        derived[referenced] = _unity_ints(_whole(buffer, column))[rows]
        check(column, derived)


def _pointcloud(buffer, parsed, role):
    _require(parsed, role, "pointcloud", "pointcloud")
    if parsed.vertices < 1:
        _fail("unsupported_profile", role, "a point cloud needs at least one point")
    if parsed.width == 1:
        _identity_cells(buffer, parsed, role, "point clouds with one-point cells must list every point once, in order")
    elif parsed.rows not in (0, parsed.vertices) or any(c.association == "cell" for c in parsed.columns):
        _fail("unsupported_profile", role, "point clouds without connectivity cannot have cell rows or cell attributes")
    colors = [parsed.column("vertex", name) for name in ("red", "green", "blue")]
    if all(c is not None and c.dtype in ("int32", "int64") for c in colors):
        for column in colors:
            for _, chunk in _chunks(buffer, column.offset, _ATTRIBUTE_DTYPES[column.dtype], column.count):
                if chunk.min() < 0 or chunk.max() > 255:
                    _fail("unsupported_profile", role, f"colour attribute {column.name!r} must be 0..255")
    return {}, {}


def _collars(buffer, parsed, role):
    _well_role(parsed, role, 1, "trajectory")
    if parsed.vertices < 1:
        _fail("unsupported_profile", role, "collars need at least one well")
    _identity_cells(buffer, parsed, role, "collars must have one point cell per well, in order")
    return {"well_count": parsed.vertices}, {}


def _trajectory(buffer, parsed, role):
    _well_role(parsed, role, 2, "collars")
    if "well_id_mapper" not in parsed.metadata:
        _fail("unsupported_profile", role, "trajectories need xarray_attrs.well_id_mapper")
    if parsed.rows < 1:
        _fail("unsupported_profile", role, "a trajectory needs at least one segment")
    mapper = _id_mapper(parsed.metadata["well_id_mapper"], role, "well_id_mapper")
    if sorted(mapper.values()) != list(range(len(mapper))):
        _fail("inconsistent_dataset", role, "well_id_mapper values must number the wells 0..N-1")
    if "lith_id_mapper" in parsed.metadata:
        _id_mapper(parsed.metadata["lith_id_mapper"], role, "lith_id_mapper")
    well_ids = parsed.column("vertex", "well_id")
    depths = parsed.column("vertex", "measured_depths")
    if well_ids is None or depths is None or depths.dtype == "bool" or well_ids.dtype == "bool":
        _fail("unsupported_profile", role, "trajectories need numeric vertex attributes well_id and measured_depths")

    rows, vertices = parsed.rows, parsed.vertices
    first = _view(buffer, parsed.cells_offset, np.dtype("<i4"), rows)
    second = _view(buffer, parsed.cells_offset + rows * 4, np.dtype("<i4"), rows)
    if first[0] != 0 or second[-1] != vertices - 1:
        _fail("inconsistent_dataset", role, "trajectory segments must cover every vertex, in order")
    for start in range(0, rows, _CHUNK):
        stop = min(rows, start + _CHUNK)
        if not np.array_equal(second[start:stop], first[start:stop] + 1):
            _fail("inconsistent_dataset", role, "trajectory segments must join consecutive vertices")
    # Within a well a segment starts where the previous one ended; the next well starts one vertex later.
    cell_starts = [0]
    for start in range(0, rows - 1, _CHUNK):
        stop = min(rows - 1, start + _CHUNK)
        gap = first[start + 1:stop + 1] - second[start:stop]
        if ((gap != 0) & (gap != 1)).any():
            _fail("inconsistent_dataset", role, "trajectory segments must form one unbroken polyline per well")
        cell_starts.extend((np.flatnonzero(gap == 1) + start + 1).tolist())
        if len(cell_starts) > len(mapper):
            break
    if len(cell_starts) != len(mapper):
        _fail("inconsistent_dataset", role,
              f"trajectory has {len(cell_starts)} polylines but well_id_mapper names {len(mapper)} wells")
    vertex_starts = [int(first[s]) for s in cell_starts] + [vertices]
    cell_bounds = cell_starts + [rows]
    ids = _whole(buffer, well_ids)
    md = _whole(buffer, depths)
    cell_ids = parsed.column("cell", "well_id")
    cell_values = None if cell_ids is None or cell_ids.dtype == "bool" else _whole(buffer, cell_ids)
    if not np.isfinite(md).all():
        _fail("unsupported_profile", role, "measured_depths must be finite")
    for well in range(len(mapper)):
        span = slice(vertex_starts[well], vertex_starts[well + 1])
        if not (ids[span] == well).all():
            _fail("inconsistent_dataset", role, f"vertices of polyline {well} do not all carry well_id {well}")
        if (np.diff(md[span]) < 0).any():
            _fail("inconsistent_dataset", role, f"measured_depths decrease along well {well}")
        if cell_values is not None and not (cell_values[cell_bounds[well]:cell_bounds[well + 1]] == well).all():
            _fail("inconsistent_dataset", role, f"segments of polyline {well} do not all carry well_id {well}")
    return {"well_count": len(mapper)}, {}


def _well_role(parsed, role, width, other):
    if parsed.kind == "unstructured" and parsed.width == width:
        return
    if parsed.kind == "unstructured" and parsed.width in (1, 2):
        _fail("inconsistent_dataset", role, f"this file looks like a {other} file; check the collars and trajectory roles")
    if parsed.family is not None:
        _fail("family_mismatch", role, f"this is a {parsed.family} dataset, not a wells {role} file", parsed.family)
    _fail("unsupported_profile", role, f"this .le file is not a supported wells {role} file")


def _id_mapper(value, role, label):
    if (not isinstance(value, dict) or
            any(not isinstance(k, str) or not k or type(v) is not int for k, v in value.items())):
        _fail("unsupported_profile", role, f"{label} must map names to integers")
    if len(set(value.values())) != len(value):
        _fail("inconsistent_dataset", role, f"{label} values must be unique")
    return value


def _identity_cells(buffer, parsed, role, message):
    if parsed.rows != parsed.vertices:
        _fail("unsupported_profile", role, message)
    for start, chunk in _chunks(buffer, parsed.cells_offset, np.dtype("<i4"), parsed.rows):
        if not np.array_equal(chunk, np.arange(start, start + chunk.size, dtype=np.int32)):
            _fail("unsupported_profile", role, message)


def _finite_numeric(buffer, column):
    # The writers store integral float columns (e.g. UVs of 0 and 1) as int64; Unity reads both.
    if column is None or column.dtype == "bool":
        return False
    return all(np.isfinite(chunk).all() for _, chunk in _chunks(buffer, column.offset, _ATTRIBUTE_DTYPES[column.dtype], column.count))


def _unity_ints(values):
    """Unity's ``(int)`` of each value after it was read as float32 (NaN and overflow become int.MinValue)."""
    floats = values.astype(np.float32).astype(np.float64)
    result = np.full(floats.shape, _INT32_MIN, dtype=np.int64)
    valid = np.isfinite(floats) & (floats > _INT32_MIN - 1.0) & (floats < _INT32_MAX + 1.0)
    result[valid] = np.trunc(floats[valid]).astype(np.int64)
    return result


def _runs(values):
    return 0 if values.size == 0 else 1 + int(np.count_nonzero(values[1:] != values[:-1]))

# endregion

# region Structured


@dataclass
class _Structured:
    kind: str
    shape: Tuple[int, ...]
    name: str
    channels: Optional[int]
    offset: int

    @property
    def family(self):
        if self.channels is not None or len(self.shape) == 2:
            return "raster"
        return "volume" if len(self.shape) == 3 else None


def _structured(buffer, header, offset, role):
    attrs = header.get("xarray_attrs")
    if isinstance(attrs, dict) and _TEMPORAL_KEYS & set(attrs):
        _fail("unsupported_profile", role, "time-series .le files (snapshots and masters) are not supported")
    extra = sorted(set(header) - _STRUCTURED_KEYS)
    if extra:
        _fail("unsupported_profile", role, f"unsupported header fields: {', '.join(map(str, extra[:5]))}")
    missing = sorted(_STRUCTURED_KEYS - set(header))
    if missing:
        _fail("malformed", role, f"missing header fields: {', '.join(missing)}")
    shape = header["data_shape"]
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 3 or
            any(type(size) is not int or not 1 <= size <= _INT32_MAX for size in shape)):
        _fail("malformed", role, "data_shape must hold one to three positive integer dimensions")
    if header["dtype"] != "float32":
        _fail("unsupported_profile", role, f"grid dtype {header['dtype']!r} is not supported; LiquidEarth reads float32 grids")
    if header["transform"] is not None:
        _fail("unsupported_profile", role, "grid transforms are not supported")
    name = header["data_name"]
    if not isinstance(name, str) or not name.strip():
        _fail("malformed", role, "data_name must be a non-empty string")
    channels = None
    if len(shape) == 3 and name.lower() in _COLOR_CHANNELS:
        if name not in _COLOR_CHANNELS or shape[2] != _COLOR_CHANNELS[name]:
            _fail("unsupported_profile", role, "colour rasters are [nx, ny, 3] named 'rgb' or [nx, ny, 4] named 'rgba'")
        channels = shape[2]
        dims, sizes = ("x", "y"), shape[:2]
    else:
        dims = ("dim0",) if len(shape) == 1 else ("x", "y", "z")[:len(shape)]
        sizes = shape
        if name in dims or name.lower() in _COLOR_CHANNELS:
            _fail("unsupported_profile", role, f"data_name {name!r} is reserved")
    bounds = header["bounds"]
    if not isinstance(bounds, dict) or set(bounds) != set(dims):
        _fail("malformed", role, f"bounds must map exactly the axes {', '.join(dims)} to sample extrema")
    for dim, size in zip(dims, sizes):
        pair = bounds[dim]
        if not isinstance(pair, list) or len(pair) != 2 or any(type(v) not in (int, float) for v in pair):
            _fail("malformed", role, f"bounds for axis {dim} must be [min, max]")
        try:
            low, high = float(pair[0]), float(pair[1])
        except OverflowError:
            _fail("unsupported_profile", role, f"bounds for axis {dim} exceed the float32 range")
        if low != pair[0] or high != pair[1] or not math.isfinite(high - low) or low > high or \
                (size == 1 and low != high) or (size > 1 and low == high):
            _fail("malformed", role, f"invalid sample extrema for axis {dim}")
        if max(abs(low), abs(high)) > _FLOAT32_MAX:
            _fail("unsupported_profile", role, f"bounds for axis {dim} exceed the float32 range")
    count = math.prod(shape)
    if len(buffer) - offset != count * 4:
        _fail("malformed", role, f"payload length mismatch: expected {count * 4} bytes, got {len(buffer) - offset}")
    for _, chunk in _chunks(buffer, offset, np.dtype("<f4"), count):
        if channels is not None:
            if (~np.isnan(chunk) & ((chunk < 0) | (chunk > 255))).any():
                _fail("unsupported_profile", role, "colour raster values must be 0..255 or NaN")
        elif np.isinf(chunk).any():
            _fail("unsupported_profile", role, "grid values must be finite or NaN")
    return _Structured("structured", tuple(shape), name, channels, offset)


def _volume(buffer, parsed, role):
    _require(parsed, role, "volume", "volume")
    return {}, {}


def _raster(buffer, parsed, role):
    _require(parsed, role, "raster", "raster")
    return {}, {}

# endregion

# region Arrays


def _view(buffer, offset, dtype, count):
    if count == 0:
        return np.empty(0, dtype=dtype)
    return np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)


def _whole(buffer, column):
    return _view(buffer, column.offset, _ATTRIBUTE_DTYPES[column.dtype], column.count)


def _chunks(buffer, offset, dtype, count):
    for start in range(0, count, _CHUNK):
        size = min(_CHUNK, count - start)
        yield start, np.frombuffer(buffer, dtype=dtype, count=size, offset=offset + start * dtype.itemsize)

# endregion

# region PNG


def _validate_png(buffer, role):
    size = len(buffer)
    if size > MAX_TEXTURE_BYTES:
        _fail("unsupported_profile", role, f"textures are limited to {MAX_TEXTURE_BYTES} bytes")
    if size < 8 or bytes(buffer[:8]) != _PNG_SIGNATURE:
        _fail("malformed", role, "not a PNG file")
    offset, chunks = 8, 0
    width = height = channels = 0
    header = data = data_closed = end = False
    compressed = bytearray()
    while offset < size:
        chunks += 1
        if chunks > MAX_PNG_CHUNKS or size - offset < 12:
            _fail("malformed", role, "truncated or oversized PNG chunk list")
        length = int.from_bytes(buffer[offset:offset + 4], "big")
        if length > size - offset - 12:
            _fail("malformed", role, "PNG chunk extends past the end of the file")
        kind = bytes(buffer[offset + 4:offset + 8])
        body = buffer[offset + 8:offset + 8 + length]
        if not kind.isalpha() or not kind.isascii() or kind[2] & 32:
            _fail("malformed", role, "invalid PNG chunk type")
        stored = int.from_bytes(buffer[offset + 8 + length:offset + 12 + length], "big")
        if zlib.crc32(body, zlib.crc32(kind)) != stored:
            _fail("malformed", role, "PNG chunk CRC mismatch")
        offset += length + 12
        if not header and kind != b"IHDR":
            _fail("malformed", role, "PNG must start with IHDR")
        if kind == b"IHDR":
            if header or length != 13:
                _fail("malformed", role, "invalid PNG IHDR")
            width, height = int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big")
            depth, color, compression, filtering, interlace = body[8], body[9], body[10], body[11], body[12]
            legal = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}.get(color, ())
            if depth not in legal or compression or filtering or interlace > 1 or not width or not height:
                _fail("malformed", role, "invalid PNG IHDR")
            if width > MAX_TEXTURE_DIMENSION or height > MAX_TEXTURE_DIMENSION or width * height > MAX_TEXTURE_PIXELS:
                _fail("unsupported_profile", role, f"textures are limited to {MAX_TEXTURE_DIMENSION} pixels a side "
                                                   f"and {MAX_TEXTURE_PIXELS} pixels")
            if depth != 8 or color not in (2, 6) or interlace:
                _fail("unsupported_profile", role, "textures must be non-interlaced 8-bit RGB or RGBA PNGs")
            channels = 3 if color == 2 else 4
            header = True
        elif kind == b"IDAT":
            if data_closed:
                _fail("malformed", role, "PNG image data chunks must be consecutive")
            data = True
            compressed += body
        elif kind == b"IEND":
            if not data or length or offset != size:
                _fail("malformed", role, "PNG must end with an empty IEND after its image data")
            end = True
            break
        elif kind == b"tEXt":
            keyword = bytes(body).split(b"\x00", 1)
            if (len(keyword) != 2 or not 1 <= len(keyword[0]) <= 79 or b"\x00" in keyword[1] or
                    keyword[0][:1] == b" " or keyword[0][-1:] == b" " or b"  " in keyword[0] or
                    any(c < 32 or 127 <= c <= 160 for c in keyword[0])):
                _fail("malformed", role, "invalid PNG tEXt chunk")
            if data:
                data_closed = True
        else:
            _fail("unsupported_profile", role, f"PNG chunk {kind.decode()} is not supported in textures")
    if not end or len(compressed) < 6:
        _fail("malformed", role, "PNG has no complete image data")
    if (compressed[0] & 15) != 8 or compressed[0] >> 4 > 7 or ((compressed[0] << 8) + compressed[1]) % 31 or compressed[1] & 32:
        _fail("malformed", role, "invalid PNG zlib stream")
    row = width * channels + 1
    expected = height * row
    inflater = zlib.decompressobj()
    try:
        pixels = inflater.decompress(bytes(compressed), expected + 1)
    except zlib.error:
        _fail("malformed", role, "corrupt PNG image data")
    if len(pixels) != expected or inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
        _fail("malformed", role, "PNG image data does not match its dimensions")
    if max(pixels[::row]) > 4:
        _fail("malformed", role, "invalid PNG row filter")
    return PNGTextureReport(role=role, byte_size=size, width=width, height=height, channels=channels)

# endregion
