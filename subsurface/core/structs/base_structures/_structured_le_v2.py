"""Structured .le format version 2: every data variable of a StructuredData, with named dimensions.

Version 1 stores the active array only. Version 2 stores the whole dataset of a
regular axis-aligned grid: each data variable with its dimension names, its
attrs, the labels of non-spatial dimensions and the dataset attrs. See
``docs/le_spec.md``.

Every variable starts with the grid's spatial dimensions, in the order of
``data_shape`` (``x, y, z``; ``x, y``; or ``dim0``), and may add dimensions of
its own, for example a vector's ``component``. Variables are stored one after
another, in ``data_vars`` order, each little-endian in Fortran order (the first
dimension varies fastest). The active variable is stored first and is the active
one again when read.
"""
import json
import math
from typing import BinaryIO, Dict, List, Tuple

import numpy as np
import xarray as xr

FORMAT_VERSION = 2
MAX_HEADER_BYTES = 1024 * 1024
HEADER_KEYS = frozenset({"format_version", "data_shape", "dims", "bounds", "transform", "coords", "data_vars",
                         "xarray_attrs"})
VARIABLE_KEYS = frozenset({"name", "dims", "dtype", "byte_length", "attrs"})
SPATIAL_NAMES = frozenset({"x", "y", "z", "dim0"})
DTYPES = {name: np.dtype(name).newbyteorder("<") for name in (
        "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")}


def spatial_dims(rank: int) -> Tuple[str, ...]:
    """The spatial dimension names of a grid of ``rank`` axes, as in version 1."""
    return ("dim0",) if rank == 1 else ("x", "y", "z")[:rank]


def _json_safe(value, what: str):
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be JSON-serializable without NaN or infinity: {exc}") from exc
    return value


def _label(value, what: str):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{what} must be strings or numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{what} must be finite")
    return value


# region Write

def to_binary_v2(data, order: str = "F") -> bytes:
    """Serialize every data variable of ``data`` (a StructuredData). Raises ValueError for what the format cannot hold."""
    if order != "F":
        raise ValueError("Structured .le format 2 is written in Fortran order only; the file does not record order")
    if data._bounds is not None:
        raise ValueError("A bounds override cannot be stored; format 2 bounds come from the spatial coordinates")
    ds = data.data
    active = data.active_data_array_name
    names = [active] + [name for name in ds.data_vars if name != active]

    leading = ds[active].dims
    spatial = next((spatial_dims(rank) for rank in (3, 2, 1) if tuple(leading[:rank]) == spatial_dims(rank)), None)
    if spatial is None:
        raise ValueError(f"The active variable must start with the spatial dimensions x, y(, z) or dim0, not {leading}")

    dims: Dict[str, int] = {dim: ds.sizes[dim] for dim in spatial}
    for name in names:
        variable_dims = ds[name].dims
        if tuple(variable_dims[:len(spatial)]) != spatial:
            raise ValueError(f"Variable {name!r} must start with the spatial dimensions {spatial}, not {variable_dims}")
        for dim in variable_dims[len(spatial):]:
            if dim in SPATIAL_NAMES:
                raise ValueError(f"Variable {name!r} uses the spatial name {dim!r} for a non-spatial dimension")
            dims.setdefault(dim, ds.sizes[dim])

    bounds = {}
    for dim in spatial:
        bounds[dim] = _spatial_bounds(ds, dim)

    coords = {}
    for coord_name, coord in ds.coords.items():
        if coord.attrs:
            raise ValueError(f"Attributes of coordinate {coord_name!r} would be lost")
        if coord.dims != (coord_name,) or coord_name not in dims:
            raise ValueError(f"Coordinate {coord_name!r} is not the coordinate of a stored dimension and would be lost")
        if coord_name not in spatial:
            coords[coord_name] = [_label(value, f"Labels of {coord_name!r}") for value in coord.values.tolist()]

    dtype_name = np.dtype(data.dtype).name
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported structured .le numeric dtype: {data.dtype!r}")
    dtype = DTYPES[dtype_name]
    variables, body = [], []
    for name in names:
        array = ds[name]
        if array.dtype.kind not in "biuf":
            raise ValueError(f"Variable {name!r} is not numeric ({array.dtype})")
        payload = np.asarray(array.values).astype(dtype).tobytes(order="F")
        variables.append({"name": name, "dims": list(array.dims), "dtype": dtype_name, "byte_length": len(payload),
                          "attrs": _json_safe(dict(array.attrs), f"Attributes of {name!r}")})
        body.append(payload)

    header = {
            "format_version": FORMAT_VERSION,
            "data_shape"    : [dims[dim] for dim in spatial],
            "dims"          : dims,
            "bounds"        : bounds,
            "transform"     : None,
            "coords"        : coords,
            "data_vars"     : variables,
            "xarray_attrs"  : _json_safe(dict(ds.attrs), "Dataset attributes"),
    }
    header_bytes = json.dumps(header, allow_nan=False).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError(f"The header has {len(header_bytes)} bytes; readers accept at most {MAX_HEADER_BYTES}")
    return len(header_bytes).to_bytes(4, byteorder="little") + header_bytes + b"".join(body)


def _spatial_bounds(ds: xr.Dataset, dim: str) -> List[float]:
    """Sample extrema of a spatial axis, which must be regular: readers rebuild it with linspace."""
    if dim not in ds.coords:
        raise ValueError(f"The spatial dimension {dim!r} has no coordinate, so its bounds are unknown")
    values = np.asarray(ds.coords[dim].values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Coordinates of {dim!r} must be finite")
    low, high = float(values[0]), float(values[-1])
    if values.size > 1:
        tolerance = 1e-9 * max(abs(low), abs(high), high - low)
        if not (high > low and np.allclose(values, np.linspace(low, high, values.size), rtol=0, atol=tolerance)):
            raise ValueError(f"Coordinates of {dim!r} must be regular and increasing")
    return [low, high]

# endregion

# region Read


def header_layout_v2(header: dict):
    """Check a version 2 ``header`` without its bounds; return the spatial shape and dims, labels and variables.

    Each variable is ``(name, dims, shape, dtype, byte_length, attrs)``, in payload order.
    """
    if set(header) != HEADER_KEYS:
        raise ValueError("Unsupported structured .le format 2 header fields")
    if type(header["format_version"]) is not int or header["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported structured .le format_version")
    if header["transform"] is not None:
        raise ValueError("Non-null structured .le transforms are unsupported")
    xarray_attrs = header["xarray_attrs"]
    if not isinstance(xarray_attrs, dict):
        raise ValueError("xarray_attrs must be a JSON object")

    shape = header["data_shape"]
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 3
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError("data_shape must contain one to three positive integer dimensions")
    spatial = spatial_dims(len(shape))
    dims = header["dims"]
    if (not isinstance(dims, dict)
            or any(not isinstance(dim, str) or not dim or type(size) is not int or size <= 0 for dim, size in dims.items())):
        raise ValueError("dims must map dimension names to positive integer sizes")
    if any(dims.get(dim) != size for dim, size in zip(spatial, shape)):
        raise ValueError("dims must hold the spatial dimensions with the sizes of data_shape")
    if any(dim in SPATIAL_NAMES and dim not in spatial for dim in dims):
        raise ValueError("Non-spatial dimensions must not use the names x, y, z or dim0")

    labels = header["coords"]
    if not isinstance(labels, dict):
        raise ValueError("coords must be a JSON object")
    for dim, values in labels.items():
        if dim not in dims or dim in spatial:
            raise ValueError(f"coords may only label non-spatial dimensions, not {dim!r}")
        if not isinstance(values, list) or len(values) != dims[dim]:
            raise ValueError(f"coords of {dim!r} must list one label per index")
        for value in values:
            _label(value, f"Labels of {dim!r}")

    variables = header["data_vars"]
    if not isinstance(variables, list) or not variables:
        raise ValueError("data_vars must be a nonempty list")
    names, used, layout = set(), set(spatial), []
    for variable in variables:
        if not isinstance(variable, dict) or set(variable) != VARIABLE_KEYS:
            raise ValueError("Unsupported structured .le data variable fields")
        name, variable_dims = variable["name"], variable["dims"]
        if not isinstance(name, str) or not name.strip() or name in dims or name in names:
            raise ValueError("Data variable names must be unique nonempty strings distinct from dimension names")
        names.add(name)
        if (not isinstance(variable_dims, list) or any(not isinstance(dim, str) for dim in variable_dims)
                or tuple(variable_dims[:len(spatial)]) != spatial
                or len(set(variable_dims)) != len(variable_dims) or any(dim not in dims for dim in variable_dims)):
            raise ValueError(f"Variable {name!r} must use dims of the header, starting with {spatial}")
        used.update(variable_dims)
        if not isinstance(variable["dtype"], str) or variable["dtype"] not in DTYPES:
            raise ValueError(f"Unsupported structured .le numeric dtype for {name!r}")
        dtype = DTYPES[variable["dtype"]]
        variable_shape = tuple(dims[dim] for dim in variable_dims)
        if type(variable["byte_length"]) is not int or variable["byte_length"] != math.prod(variable_shape) * dtype.itemsize:
            raise ValueError(f"byte_length of {name!r} does not match its shape and dtype")
        if not isinstance(variable["attrs"], dict):
            raise ValueError(f"attrs of {name!r} must be a JSON object")
        layout.append((name, tuple(variable_dims), variable_shape, dtype, variable["byte_length"], variable["attrs"]))
    if used != set(dims):
        raise ValueError("dims lists a dimension that no data variable uses")
    return tuple(shape), spatial, labels, layout


def read_v2(cls, header: dict, source: BinaryIO, coords_from_bounds):
    """Read the payload after a version 2 ``header`` from ``source``; ``cls`` is StructuredData."""
    shape, spatial, labels, layout = header_layout_v2(header)
    coords = coords_from_bounds(header["bounds"], spatial, shape)
    expected_size = sum(entry[4] for entry in layout)
    payload_start = source.tell()
    source.seek(0, 2)
    if expected_size > np.iinfo(np.intp).max or source.tell() - payload_start != expected_size:
        raise ValueError("Structured .le payload length does not match the data variables")
    source.seek(payload_start)

    data_vars = {}
    for name, variable_dims, variable_shape, dtype, byte_length, attrs in layout:
        payload = source.read(byte_length)
        if len(payload) != byte_length:
            raise ValueError("Truncated structured .le payload")
        values = np.frombuffer(payload, dtype=dtype).reshape(variable_shape, order="F").astype(dtype.newbyteorder("="))
        data_vars[name] = xr.Variable(variable_dims, values, attrs=attrs)
    coords = {dim: np.linspace(*coords[dim], num=size) for dim, size in zip(spatial, shape)} | {
            dim: np.asarray(values) for dim, values in labels.items()}
    if any(np.any(np.diff(coords[dim]) <= 0) for dim in spatial):
        raise ValueError("Structured .le coordinate spacing is not representable in float64")

    first = layout[0]
    result = cls(xr.Dataset(data_vars, coords=coords, attrs=header["xarray_attrs"]), first[0])
    result.dtype = first[3].name
    return result

# endregion
