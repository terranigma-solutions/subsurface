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
        shape = _shape(header["data_shape"], "data_shape", 3)
        dtype = _dtype(header.get("dtype"))
        bounds = header.get("bounds")
        if (not isinstance(bounds, list) or len(bounds) != 6
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds)
                or any(bounds[i] > bounds[i + 1] for i in (0, 2, 4))):
            raise ValueError("Invalid structured bounds")
        if not isinstance(header.get("data_name"), str):
            raise ValueError("Structured data_name must be a string")
        if header.get("transform") is not None:
            raise ValueError("Structured transforms are not supported")
        schema["grid"] = ({"name": header["data_name"], "dtype": str(dtype), "shape": shape},)
        metadata = dict(metadata, bounds=bounds, transform=header.get("transform"),
                        data_name=header["data_name"])
        return "structured", version, {"data": shape}, schema, metadata, math.prod(shape) * dtype.itemsize

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
            if not np.isfinite(values).all():
                raise ValueError("Grouping IDs must not be missing or nonfinite")
            ids = tuple(np.unique(values).tolist())
    return LEInspection(
        file_kind=kind, format_version=version, byte_size=size, shapes=shapes,
        attribute_schema=schema, metadata=metadata,
        vertex_count=shapes["vertex"][0] if kind == "unstructured" else None,
        cell_count=shapes["cells"][0] if kind == "unstructured" else None,
        grid_sample_count=math.prod(shapes["data"]) if kind == "structured" else None,
        logical_object_count=len(ids) if ids is not None else None,
        object_ids=ids, grouping_validated=ids is not None,
    )
