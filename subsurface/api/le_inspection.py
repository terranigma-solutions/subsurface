"""Bounded LiquidEarth header inspection and explicitly grouped object counts."""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class LEInspection:
    """File summary; header/length checks are not full payload validation."""

    file_kind: str
    format_version: int
    byte_size: int
    shapes: Dict[str, Tuple[int, ...]]
    attribute_schema: Dict[str, Tuple[Dict[str, Any], ...]]
    metadata: Dict[str, Any]
    vertex_count: Optional[int] = None
    cell_count: Optional[int] = None
    grid_sample_count: Optional[int] = None
    dataset_count: int = 1
    logical_object_count: Optional[int] = None
    object_ids: Optional[Tuple[Union[int, float], ...]] = None
    header_validated: bool = True
    payload_length_validated: bool = True
    payload_validated: bool = False
    grouping_validated: bool = False


def _shape(value, name, rank):
    if (not isinstance(value, list) or len(value) != rank
            or any(type(n) is not int or n < 0 for n in value)):
        raise ValueError(f"Invalid {name}: expected {rank} nonnegative integer dimensions")
    return tuple(value)


def _dtype(value):
    try:
        dtype = np.dtype(value) if isinstance(value, str) else None
    except (TypeError, ValueError):
        dtype = None
    if dtype is None or dtype.kind not in "biuf" or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported numeric dtype: {value!r}")
    return dtype


def group_object_ids(attributes, *, object_attribute: str, association: str, cell_width: int) -> np.ndarray:
    """Return sorted original IDs from a mapping/DataFrame of attribute columns.

    The caller supplies the attributes for the explicit association: point for
    cell_width <= 1, otherwise cell. The selected column must be one-dimensional
    numeric, non-boolean and finite. Empty columns return an empty ID array.
    This helper does not validate geometry or attribute-to-geometry row counts.
    """
    if not isinstance(object_attribute, str) or not object_attribute:
        raise ValueError("object_attribute must be a nonempty string")
    if association not in ("cell", "point"):
        raise ValueError("association must be 'cell' or 'point'")
    if type(cell_width) is not int or cell_width < 0:
        raise ValueError("cell_width must be a nonnegative integer")
    if (association == "point") != (cell_width <= 1):
        raise ValueError("Use point association for point clouds and cell association for meshes")
    if attributes is None or object_attribute not in attributes:
        raise ValueError(f"Missing grouping attribute {object_attribute!r} on {association}")
    values = np.asarray(attributes[object_attribute])
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError("Grouping IDs must be a one-dimensional numeric column, not boolean")
    if not np.isfinite(values).all():
        raise ValueError("Grouping IDs must not be missing or nonfinite")
    return np.unique(values)


def _header_layout(header):
    # Temporary integration seam: replace with foundation's validated layout.
    version = header.get("format_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError(f"Unsupported format_version: {version!r}")
    metadata = header.get("xarray_attrs", {})
    if not isinstance(metadata, dict):
        raise ValueError("xarray_attrs must be an object")
    schema = {"cell": (), "point": ()}
    if "data_shape" in header:
        if "vertex_shape" in header or "cell_shape" in header:
            raise ValueError("Mixed structured and unstructured header")
        if set(header) != {"data_shape", "bounds", "transform", "dtype", "data_name"}:
            raise ValueError("Unsupported structured header fields")
        raw_shape = header["data_shape"]
        if (not isinstance(raw_shape, list) or not 1 <= len(raw_shape) <= 3
                or any(type(size) is not int or size <= 0 for size in raw_shape)):
            raise ValueError("data_shape must contain one to three positive integer dimensions")
        shape = tuple(raw_shape)
        dims = ("dim0",) if len(shape) == 1 else ("x", "y", "z")[:len(shape)]
        dtype = _dtype(header.get("dtype"))
        if dtype.kind == "b":
            raise ValueError("Unsupported structured numeric dtype")
        bounds = header.get("bounds")
        if not isinstance(bounds, dict) or set(bounds) != set(dims):
            raise ValueError("bounds must map each standard axis to sample extrema; flat bounds are ambiguous")
        for dim, size in zip(dims, shape):
            pair = bounds[dim]
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(value) not in (int, float) for value in pair)):
                raise ValueError(f"Invalid bounds for axis {dim}")
            try:
                low, high = map(float, pair)
            except (OverflowError, ValueError) as error:
                raise ValueError(f"Invalid bounds for axis {dim}") from error
            if low != pair[0] or high != pair[1]:
                raise ValueError(f"Bounds for axis {dim} lose precision in float64 coordinates")
            if (not math.isfinite(low) or not math.isfinite(high)
                    or not math.isfinite(high - low) or low > high
                    or (size == 1 and low != high) or (size > 1 and low == high)):
                raise ValueError(f"Invalid sample extrema for axis {dim}")
        name = header["data_name"]
        if not isinstance(name, str) or not name.strip() or name in dims:
            raise ValueError("Structured data_name must be a nonempty string distinct from axis names")
        if header.get("transform") is not None:
            raise ValueError("Structured transforms are not supported")
        schema["grid"] = ({"name": header["data_name"], "dtype": str(dtype), "shape": shape},)
        metadata = dict(metadata, bounds=bounds, transform=header.get("transform"),
                        data_name=header["data_name"])
        payload_size = math.prod(shape) * dtype.itemsize
        if payload_size > np.iinfo(np.intp).max:
            raise ValueError("Structured payload length exceeds platform limits")
        return "structured", version, {"data": shape}, schema, metadata, payload_size

    vertex = _shape(header.get("vertex_shape"), "vertex_shape", 2)
    cells = _shape(header.get("cell_shape"), "cell_shape", 2)
    if vertex[1] != 3 and vertex != (0, 0):
        raise ValueError("vertex_shape must have three coordinates")
    if cells[1] not in (0, 1, 2, 3, 4, 8):
        raise ValueError("Unsupported or ambiguous flattened cell_shape")
    shapes = {"vertex": vertex, "cells": cells}
    offset = (math.prod(vertex) + math.prod(cells)) * 4
    for association, key, rows in (("cell", "cell", cells[0]), ("point", "vertex", vertex[0])):
        columns = []
        if version == 2:
            entries = header.get(key + "_attrs", [])
            if not isinstance(entries, list):
                raise ValueError(f"{key}_attrs must be a list")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("Attribute metadata must be an object")
                shape = _shape(entry.get("shape"), "attribute shape", 1)
                dtype = _dtype(entry.get("dtype"))
                length = entry.get("byte_length")
                if shape != (rows,) or type(length) is not int or length != rows * dtype.itemsize:
                    raise ValueError("Attribute row count or byte_length mismatch")
                columns.append(dict(name=entry.get("name"), dtype=str(dtype), shape=shape,
                                    byte_length=length, offset=offset))
                offset += length
        else:
            shape = _shape(header.get(key + "_attr_shape", [0, 0]), key + "_attr_shape", 2)
            names = header.get(key + "_attr_names", [])
            if not isinstance(names, list) or len(names) != shape[1] or (shape[1] and shape[0] != rows):
                raise ValueError("Legacy attribute names or row count mismatch")
            for index, name in enumerate(names):
                columns.append(dict(name=name, dtype="float32", shape=(rows,), byte_length=rows * 4,
                                    offset=offset + index * rows * 4))
            offset += math.prod(shape) * 4
        names = [column["name"] for column in columns]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Attribute names must be nonempty unique strings")
        schema[association] = tuple(columns)
    return "unstructured", version, shapes, schema, metadata, offset


def inspect_le(
    path: Union[str, os.PathLike], *, object_attribute: Optional[str] = None,
    association: Optional[str] = None, max_header_bytes: int = 8 * 1024 * 1024,
) -> LEInspection:
    """Inspect an embedded-header .le file without reading geometry.

    Grouping requires a numeric attribute and association ('cell' for meshes,
    'point' for point clouds). IDs must be finite; sorted original IDs are
    returned without integer-to-float conversion. Empty grouping counts zero.
    Header schema and exact file length are checked, not payload contents or
    connectivity. Version-2 grouping reads only the requested column.
    """
    if (object_attribute is None) != (association is None):
        raise ValueError("object_attribute and association must be supplied together")
    if association is not None and association not in ("cell", "point"):
        raise ValueError("association must be 'cell' or 'point'")
    if object_attribute is not None and (not isinstance(object_attribute, str) or not object_attribute):
        raise ValueError("object_attribute must be a nonempty string")
    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(4)
        if len(prefix) != 4:
            raise ValueError("Truncated header length prefix")
        length = int.from_bytes(prefix, "little")
        if length <= 0 or length > max_header_bytes:
            raise ValueError("Header length exceeds limit or is empty")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Truncated JSON header")
        try:
            header = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Invalid JSON header") from error
        if not isinstance(header, dict):
            raise ValueError("JSON header must be an object")
        kind, version, shapes, schema, metadata, payload_size = _header_layout(header)
        if size != 4 + length + payload_size:
            raise ValueError("File byte size does not match declared payload length")
        ids = None
        if association is not None:
            if kind != "unstructured":
                raise ValueError("Structured grid samples do not support logical object grouping")
            point_cloud = shapes["cells"][1] <= 1
            if (association == "point") != point_cloud:
                raise ValueError("Use point association for point clouds and cell association for meshes")
            column = next((col for col in schema[association] if col["name"] == object_attribute), None)
            if column is None:
                raise ValueError(f"Missing grouping attribute {object_attribute!r} on {association}")
            if np.dtype(column["dtype"]).kind == "b":
                raise ValueError("Grouping IDs must be numeric, not boolean")
            if version == 2:
                stream.seek(4 + length + column["offset"])
                raw = stream.read(column["byte_length"])
                if len(raw) != column["byte_length"]:
                    raise ValueError("Truncated grouping column")
                values = np.frombuffer(raw, dtype=column["dtype"])
            else:
                from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh

                columns = schema[association]
                block_length = sum(col["byte_length"] for col in columns)
                stream.seek(4 + length + columns[0]["offset"])
                raw = stream.read(block_length)
                if len(raw) != block_length:
                    raise ValueError("Truncated legacy attribute block")
                key = "cell_attr" if association == "cell" else "vertex_attr"
                frame, _ = LiquidEarthMesh._read_attr_v1(raw, 0, header, key, order="F")
                values = frame[object_attribute].to_numpy() if frame is not None else np.empty(0, dtype=np.float32)
            ids = tuple(group_object_ids(
                {object_attribute: values}, object_attribute=object_attribute,
                association=association, cell_width=shapes["cells"][1],
            ).tolist())
    return LEInspection(
        file_kind=kind, format_version=version, byte_size=size, shapes=shapes,
        attribute_schema=schema, metadata=metadata,
        vertex_count=shapes["vertex"][0] if kind == "unstructured" else None,
        cell_count=shapes["cells"][0] if kind == "unstructured" else None,
        grid_sample_count=math.prod(shapes["data"]) if kind == "structured" else None,
        logical_object_count=len(ids) if ids is not None else None,
        object_ids=ids, grouping_validated=ids is not None,
    )
