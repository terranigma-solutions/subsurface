"""Bounded LiquidEarth header inspection and explicitly grouped object counts."""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from subsurface.core.structs.base_structures._liquid_earth_mesh import (
    MAX_LE_HEADER_BYTES, read_le_header, validate_unstructured_layout,
)


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


def _header_layout(header, payload_length):
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
        metadata = dict(bounds=bounds, transform=header.get("transform"), data_name=header["data_name"])
        payload_size = math.prod(shape) * dtype.itemsize
        if payload_size > np.iinfo(np.intp).max:
            raise ValueError("Structured payload length exceeds platform limits")
        return "structured", 1, {"data": shape}, schema, metadata, payload_size

    layout = validate_unstructured_layout(header, payload_length)
    version = layout["format_version"]
    shapes = {"vertex": layout["vertex_shape"], "cells": layout["cell_shape"]}
    if layout["wire_cell_shape"] != layout["cell_shape"]:
        shapes["wire_cells"] = layout["wire_cell_shape"]
    columns = {"cell": [], "point": []}
    for segment in layout["segments"]:
        if segment["association"] == "geometry":
            continue
        association = "point" if segment["association"] == "vertex" else "cell"
        if version == 2:
            columns[association].append(dict(
                name=segment["name"], dtype=str(segment["dtype"]), shape=segment["shape"],
                byte_length=segment["byte_length"], offset=segment["offset"],
            ))
        else:
            rows = segment["shape"][0]
            length = rows * segment["dtype"].itemsize
            for index, name in enumerate(segment["names"]):
                columns[association].append(dict(
                    name=name, dtype=str(segment["dtype"]), shape=(rows,),
                    byte_length=length, offset=segment["offset"] + index * length,
                ))
    schema = {association: tuple(entries) for association, entries in columns.items()}
    return "unstructured", version, shapes, schema, layout["data_attrs"], layout["payload_length"]


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
        if length <= 0 or length > min(max_header_bytes, MAX_LE_HEADER_BYTES):
            raise ValueError("Header length exceeds limit or is empty")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Truncated JSON header")
        header, payload_start = read_le_header(prefix + raw)
        kind, version, shapes, schema, metadata, payload_size = _header_layout(header, size - payload_start)
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
            stream.seek(payload_start + column["offset"])
            raw = stream.read(column["byte_length"])
            if len(raw) != column["byte_length"]:
                raise ValueError("Truncated grouping column")
            values = np.frombuffer(raw, dtype=column["dtype"])
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
