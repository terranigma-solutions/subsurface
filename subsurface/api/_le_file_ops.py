"""Small shared safety boundary for unstructured LE file operations.

Keep meshes as LiquidEarthMesh objects: converting through UnstructuredData's
xarray attribute matrices can coerce mixed integer/float/bool columns.
"""

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


def load_le_mesh(source) -> LiquidEarthMesh:
    """Load a prefixed mesh using the validated decoder and default F order.

    Numeric columns remain separate pandas series with their wire dtypes.
    Output safety checks (including finite geometry) run on serialization.
    Sidecars and externally specified nondefault array orders are not supported.
    """
    with open(source, 'rb') as stream:
        return LiquidEarthMesh.from_binary(stream.read(), order='F')


def serialize_le_mesh(mesh: LiquidEarthMesh) -> bytes:
    """Preflight and serialize a mesh in F order, then validate the result.

    No object/nullable column coercion, dropping, or connectivity truncation is
    permitted. Geometry must be finite and within float32 range. Integer/bool
    columns retain width and values; float columns follow the shipped int64-if-
    exactly-integral, otherwise float32 rule. NaN/infinite attributes are retained;
    finite float overflow is rejected. Float32 rounding/underflow is allowed.
    Integral float columns outside int64 range are rejected before the writer's
    attempted integer conversion can overflow. Empty tables with named columns
    are rejected because the shipped writer would omit their schema.

    Metadata must use JSON-native objects with string dictionary keys, avoiding
    implicit key conversion or tuple-to-list loss. DataFrame row indices are not
    wire data; rows are associated positionally. This does not mutate mesh.
    """
    if not isinstance(mesh, LiquidEarthMesh):
        raise TypeError("Expected LiquidEarthMesh, not an xarray-backed container")
    vertex, cells = mesh.vertex, mesh.cells
    if (not isinstance(vertex, np.ndarray) or vertex.ndim != 2 or vertex.shape[1] != 3 or
            vertex.dtype.kind not in 'iuf'):
        raise ValueError("vertex must be a real numeric ndarray with shape (n_points, 3)")
    if (not isinstance(cells, np.ndarray) or cells.ndim != 2 or cells.shape[1] not in (0, 1, 2, 3, 4, 8) or
            cells.dtype.kind not in 'iu'):
        raise ValueError("cells must be an integer ndarray with a supported connectivity width")
    int32_max = np.iinfo(np.int32).max
    if vertex.shape[0] > int32_max + 1:
        raise ValueError("Vertex count exceeds int32 connectivity capacity")
    if cells.shape[1] == 0 and cells.shape[0] > vertex.shape[0]:
        raise ValueError("Zero-width connectivity cannot have more rows than vertices")
    if cells.size and (np.any(cells < 0) or np.any(cells > int32_max)):
        raise ValueError("Connectivity exceeds nonnegative int32 capacity")
    if cells.size and np.any(cells >= vertex.shape[0]):
        raise ValueError("Connectivity indices must be less than n_points")
    float32_max = np.finfo(np.float32).max
    if not np.all(np.isfinite(vertex)) or np.any(np.abs(vertex.astype(np.longdouble)) > float32_max):
        raise ValueError("Geometry must be finite and within float32 range")

    expected = []
    for label, frame, rows in (('cell', mesh.attributes, cells.shape[0]),
                               ('vertex', mesh.points_attributes, vertex.shape[0])):
        if not isinstance(frame, pd.DataFrame) or len(frame) != rows:
            raise ValueError(f"{label} attributes must be a DataFrame with {rows} rows")
        if not frame.columns.is_unique:
            raise ValueError(f"{label} attribute names must be unique")
        if rows == 0 and len(frame.columns):
            raise ValueError(f"Empty {label} attribute columns would be dropped by the writer")
        columns = {}
        for name in frame.columns:
            series = frame[name]
            dtype = series.dtype
            if (not isinstance(dtype, np.dtype) or not (
                    dtype.kind in 'iu' and dtype.itemsize in (1, 2, 4, 8) or
                    dtype.kind == 'b' and dtype.itemsize == 1 or
                    dtype.kind == 'f' and dtype.itemsize in (2, 4, 8))):
                raise TypeError(f"Unsupported {label} attribute {name!r} dtype: {dtype}; no coercion allowed")
            values = series.to_numpy()
            stored_dtype = dtype.newbyteorder('=')
            if dtype.kind == 'f':
                finite = np.isfinite(values)
                if np.any(np.abs(values[finite].astype(np.longdouble)) > float32_max):
                    raise ValueError(f"{label} attribute {name!r} exceeds float32 range")
                with np.errstate(invalid='ignore'):
                    integral = np.all(np.mod(values, 1) == 0)
                if integral:
                    wide = values.astype(np.longdouble)
                    if np.any(wide < -(2 ** 63)) or np.any(wide >= 2 ** 63):
                        raise ValueError(f"{label} attribute {name!r} integral float conversion exceeds int64 range")
                    integers = values.astype(np.int64)
                    stored_dtype = (np.dtype('int64') if np.all(integers.astype(np.float64) == values)
                                    else np.dtype('float32'))
                else:
                    stored_dtype = np.dtype('float32')
            columns[name] = (stored_dtype, values.astype(stored_dtype))
        expected.append(columns)

    def check_metadata(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Metadata dictionary keys must be strings")
            for item in value.values():
                check_metadata(item)
        elif isinstance(value, list):
            for item in value:
                check_metadata(item)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise TypeError("Metadata must contain only JSON-native values")

    if not isinstance(mesh.data_attrs, dict):
        raise TypeError("Dataset metadata must be a dictionary")
    check_metadata(mesh.data_attrs)
    # Infinite attributes remain infinite rather than producing modulo warnings.
    with np.errstate(invalid='ignore', under='ignore', over='raise'):
        binary = mesh.to_binary(order='F')
    restored = LiquidEarthMesh.from_binary(binary, order='F')
    if json.dumps(restored.data_attrs, sort_keys=True) != json.dumps(mesh.data_attrs, sort_keys=True):
        raise ValueError("Serialized dataset metadata differs from the input")
    if (not np.array_equal(restored.vertex, vertex.astype(np.float32)) or
            not np.array_equal(restored.cells, cells)):
        raise ValueError("Serialized geometry/connectivity differs from the expected wire values")
    for frame, original, columns in zip((restored.attributes, restored.points_attributes),
                                        (mesh.attributes, mesh.points_attributes), expected):
        if not frame.columns.equals(original.columns) or len(frame) != len(original):
            raise ValueError("Serialized attribute columns or row counts were lost")
        for name, (dtype, values) in columns.items():
            if frame[name].dtype != dtype or not np.array_equal(frame[name].to_numpy(), values, equal_nan=True):
                raise ValueError(f"Serialized attribute {name!r} differs from the expected wire values")
    return binary


def write_le_mesh(mesh: LiquidEarthMesh, destination, *, sources, overwrite=False) -> Path:
    """Validate and atomically publish a mesh, returning the destination Path.

    Callers must supply all input paths in ``sources`` (an iterable, possibly
    empty for generated data). A source/destination alias, including a symlink
    or hard link, is always rejected. Existing destinations are protected by an
    atomic no-clobber hard-link publish unless ``overwrite=True`` explicitly
    requests atomic replacement. The parent directory must already exist.
    Temporary files are closed and removed on success or failure. No fallback
    to a non-atomic publish is used if the filesystem does not support links.
    Concurrent hostile changes to directory/source identity are out of scope.
    """
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    if isinstance(sources, (str, bytes, os.PathLike)):
        raise TypeError("sources must be an iterable of source paths")
    destination = Path(destination).absolute()
    source_paths = tuple(Path(source).absolute() for source in sources)

    def check_destination():
        for source in source_paths:
            if destination.resolve() == source.resolve() or (
                    destination.exists() and source.exists() and os.path.samefile(destination, source)):
                raise ValueError("Destination must not alias any source path")
        if not overwrite and os.path.lexists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")

    check_destination()
    binary = serialize_le_mesh(mesh)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent,
                                         prefix=f'.{destination.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            if stream.write(binary) != len(binary):
                raise OSError("Incomplete temporary LE file write")
            stream.flush()
            os.fsync(stream.fileno())
        check_destination()
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return destination
