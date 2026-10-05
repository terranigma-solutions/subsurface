"""Synthetic inspection tests: no optional readers, plotting, or network."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest

from subsurface import inspect_le
from subsurface.api.le_inspection import group_object_ids
from subsurface.core.structs.base_structures._liquid_earth_mesh import (
    MAX_LE_HEADER_BYTES, LiquidEarthMesh, read_le_header, validate_unstructured_layout,
)
from subsurface.core.structs.base_structures.structured_data import StructuredData


def write_file(tmp_path, header, body=b""):
    path = tmp_path / "synthetic.le"
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(4, "little") + raw + body)
    return path


def mesh_file(tmp_path, ids=None, *, points=False, other=True):
    if ids is None:
        ids = np.array([2**60 + 1, 7, 2**60 + 1, 7], dtype=np.int64)
    rows = len(ids)
    vertex = np.zeros((rows if points else rows * 3, 3), dtype=np.float32)
    cells = np.empty((rows, 0), dtype=np.int32) if points else np.arange(rows * 3).reshape(rows, 3)
    attrs = pd.DataFrame({"objects": ids})
    if other:
        attrs.insert(0, "unused", np.arange(rows, dtype=np.float32) + 0.25)
    mesh = LiquidEarthMesh(vertex=vertex, cells=cells,
                          attributes=pd.DataFrame() if points else attrs,
                          points_attributes=attrs if points else pd.DataFrame(),
                          data_attrs={"crs": "local", "note": "two surfaces"})
    path = tmp_path / "mesh.le"
    path.write_bytes(mesh.to_binary(order="F"))
    return path


class ReadTracker:
    def __init__(self, stream, reads):
        self.stream = stream
        self.reads = reads

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def read(self, size=-1):
        assert size >= 0, "Inspection must never read an unbounded payload"
        self.reads.append((self.stream.tell(), size))
        return self.stream.read(size)


def track_reads(monkeypatch):
    reads = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: ReadTracker(real_open(*args, **kwargs), reads))
    return reads


def test_header_summary_and_explicit_objects(tmp_path):
    path = mesh_file(tmp_path)
    result = inspect_le(path)
    assert result.file_kind == "unstructured"
    assert result.format_version == 2
    assert result.byte_size == path.stat().st_size
    assert result.shapes == {"vertex": (12, 3), "cells": (4, 3)}
    assert (result.vertex_count, result.cell_count, result.dataset_count) == (12, 4, 1)
    assert result.logical_object_count is None
    assert result.object_ids is None
    assert result.grid_sample_count is None
    assert result.metadata == {"crs": "local", "note": "two surfaces"}
    assert [(col["name"], col["dtype"]) for col in result.attribute_schema["cell"]] == [
        ("unused", "float32"), ("objects", "int64")]
    assert result.header_validated and result.payload_length_validated
    assert not result.payload_validated and not result.grouping_validated
    grouped = inspect_le(path, object_attribute="objects", association="cell")
    assert grouped.logical_object_count == 2
    assert grouped.object_ids == (7, 2**60 + 1)
    assert grouped.grouping_validated and not grouped.payload_validated


def test_bounded_header_and_selective_column_reads(tmp_path, monkeypatch):
    path = mesh_file(tmp_path, points=True)
    reads = track_reads(monkeypatch)
    result = inspect_le(path)
    assert len(reads) == 2
    assert reads[0] == (0, 4)
    assert reads[1][0] == 4
    header_end = 4 + reads[1][1]
    reads.clear()
    result = inspect_le(path, object_attribute="objects", association="point")
    column = result.attribute_schema["point"][1]
    assert reads == [(0, 4), (4, header_end - 4), (header_end + column["offset"], 32)]
    assert result.logical_object_count == 2


@pytest.mark.parametrize("points", [False, True])
def test_empty_grouping(tmp_path, points):
    # The existing writer omits columns from empty DataFrames; represent an
    # explicitly declared empty ID column in the synthetic wire fixture.
    header = {"format_version": 2, "vertex_shape": [0, 3], "cell_shape": [0, 0 if points else 3],
              "cell_attrs": [], "vertex_attrs": []}
    header["vertex_attrs" if points else "cell_attrs"] = [
        {"name": "objects", "dtype": "int64", "shape": [0], "byte_length": 0}]
    path = write_file(tmp_path, header)
    result = inspect_le(path, object_attribute="objects", association="point" if points else "cell")
    assert result.logical_object_count == 0
    assert result.object_ids == ()


def test_empty_legacy_grouping(tmp_path):
    header = {"vertex_shape": [0, 3], "cell_shape": [0, 3],
              "cell_attr_shape": [0, 1], "cell_attr_names": ["objects"]}
    result = inspect_le(write_file(tmp_path, header), object_attribute="objects", association="cell")
    assert result.logical_object_count == 0 and result.object_ids == ()


@pytest.mark.parametrize("ids, expected", [
    (np.array([-19, -19, 400, 0], dtype=np.int32), (-19, 0, 400)),
    (np.array([0, 2**64 - 1], dtype=np.uint64), (0, 2**64 - 1)),
    (np.array([1.25, -3.5, 1.25], dtype=np.float32), (-3.5, 1.25)),
])
def test_numeric_ids_retained(tmp_path, ids, expected):
    result = inspect_le(mesh_file(tmp_path, ids), object_attribute="objects", association="cell")
    assert result.object_ids == expected
    assert result.logical_object_count == len(expected)


@pytest.mark.parametrize("ids", [np.array([np.nan]), np.array([np.inf]), np.array([-np.inf]), np.array([True])])
def test_invalid_grouping_values(tmp_path, ids):
    path = mesh_file(tmp_path, ids)
    with pytest.raises(ValueError, match="Grouping IDs"):
        inspect_le(path, object_attribute="objects", association="cell")


@pytest.mark.parametrize("kwargs, message", [
    ({"object_attribute": "objects"}, "supplied together"),
    ({"association": "cell"}, "supplied together"),
    ({"object_attribute": "objects", "association": "vertex"}, "association must"),
    ({"object_attribute": "objects", "association": "point"}, "point association"),
    ({"object_attribute": "absent", "association": "cell"}, "Missing grouping attribute"),
    ({"object_attribute": "", "association": "cell"}, "nonempty string"),
])
def test_invalid_grouping_requests(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        inspect_le(mesh_file(tmp_path), **kwargs)


def test_wrong_cell_association_for_point_cloud(tmp_path):
    with pytest.raises(ValueError, match="point association"):
        inspect_le(mesh_file(tmp_path, points=True), object_attribute="objects", association="cell")


def test_does_not_validate_connectivity_contents(tmp_path):
    header = {"vertex_shape": [3, 3], "cell_shape": [1, 3], "format_version": 2}
    body = np.zeros((3, 3), dtype=np.float32).tobytes() + np.array([-1, 99, 2], dtype=np.int32).tobytes()
    result = inspect_le(write_file(tmp_path, header, body))
    assert result.header_validated and not result.payload_validated
    assert result.cell_count == 1


@pytest.mark.parametrize("points", [False, True])
def test_legacy_embedded_header(tmp_path, points):
    header = {"vertex_shape": [4, 3], "cell_shape": [4, 0 if points else 3],
              "xarray_attrs": {"source": "legacy"}}
    key = "vertex_attr" if points else "cell_attr"
    header[key + "_shape"] = [4, 2]
    header[key + "_names"] = ["unused", "objects"]
    geometry = bytes((12 + (0 if points else 12)) * 4)
    attrs = np.array([[5, 9], [6, -2], [7, 9], [8, -2]], dtype=np.float32)
    path = write_file(tmp_path, header, geometry + attrs.tobytes(order="F"))
    result = inspect_le(path, object_attribute="objects", association="point" if points else "cell")
    assert result.format_version == 1
    assert result.logical_object_count == 2
    assert result.object_ids == (-2.0, 9.0)
    assert result.attribute_schema["point" if points else "cell"][0]["dtype"] == "float32"
    assert result.metadata == {"source": "legacy"}


def test_structured_sample_counts(tmp_path):
    header = {"data_shape": [2, 3, 4], "bounds": {"x": [0, 1], "y": [0, 2], "z": [0, 3]}, "dtype": "float32",
              "transform": None, "data_name": "density"}
    path = write_file(tmp_path, header, bytes(24 * 4))
    result = inspect_le(path)
    assert result.file_kind == "structured"
    assert result.format_version == 1
    assert result.shapes == {"data": (2, 3, 4)}
    assert result.grid_sample_count == 24
    assert result.vertex_count is None and result.cell_count is None
    assert result.dataset_count == 1 and result.logical_object_count is None
    assert result.attribute_schema["grid"][0]["name"] == "density"
    assert result.metadata["bounds"] == header["bounds"]
    with pytest.raises(ValueError, match="grid samples"):
        inspect_le(path, object_attribute="density", association="point")


@pytest.mark.parametrize("change, message", [
    ({"data_shape": []}, "data_shape"),
    ({"data_shape": [2, 3, 4, 5]}, "data_shape"),
    ({"data_shape": [0, 3, 4]}, "data_shape"),
    ({"data_shape": [2, True, 4]}, "data_shape"),
    ({"dtype": "object"}, "dtype"),
    ({"dtype": "bool"}, "dtype"),
    ({"dtype": "complex64"}, "dtype"),
    ({"bounds": [1, 0, 0, 2, 0, 3]}, "bounds"),
    ({"bounds": {"x": [0, 1], "y": [0, 2]}}, "bounds"),
    ({"bounds": {"x": [0, 1], "y": [0, 2], "wrong": [0, 3]}}, "bounds"),
    ({"bounds": {"x": [False, 1], "y": [0, 2], "z": [0, 3]}}, "bounds"),
    ({"bounds": {"x": [0], "y": [0, 2], "z": [0, 3]}}, "bounds"),
    ({"bounds": {"x": [0, float("nan")], "y": [0, 2], "z": [0, 3]}}, "precision"),
    ({"bounds": {"x": [1, 0], "y": [0, 2], "z": [0, 3]}}, "extrema"),
    ({"bounds": {"x": [0, 0], "y": [0, 2], "z": [0, 3]}}, "extrema"),
    ({"bounds": {"x": [2**60 + 1, 2**60 + 100], "y": [0, 2], "z": [0, 3]}}, "precision"),
    ({"data_name": 5}, "data_name"),
    ({"data_name": " "}, "data_name"),
    ({"data_name": "x"}, "data_name"),
    ({"transform": np.eye(4).tolist()}, "transforms"),
    ({"cell_shape": [0, 3]}, "Mixed"),
    ({"unexpected": 5}, "header fields"),
])
def test_invalid_structured_headers(tmp_path, change, message):
    header = {"data_shape": [2, 3, 4], "bounds": {"x": [0, 1], "y": [0, 2], "z": [0, 3]},
              "dtype": "float32", "transform": None, "data_name": "density"}
    header.update(change)
    with pytest.raises(ValueError, match=message):
        inspect_le(write_file(tmp_path, header, bytes(96)))


@pytest.mark.parametrize("shape", [(1,), (4,), (1, 3), (2, 1), (2, 3), (2, 3, 4), (1, 3, 1)])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint16", "float16"])
def test_public_structured_writer_inspection(tmp_path, monkeypatch, shape, dtype):
    dims = ["dim0"] if len(shape) == 1 else ["x", "y", "z"][:len(shape)]
    coords = {dim: np.arange(size, dtype=float) * 2.5 + 7 for dim, size in zip(dims, shape)}
    data = StructuredData.from_numpy(np.arange(np.prod(shape)).reshape(shape),
                                     coords=coords, data_array_name="density")
    data.dtype = dtype
    path = tmp_path / "public_writer.le"
    path.write_bytes(data.to_binary())
    reads = track_reads(monkeypatch)
    result = inspect_le(path)
    assert len(reads) == 2
    assert result.shapes == {"data": shape}
    assert result.grid_sample_count == np.prod(shape)
    assert result.attribute_schema["grid"][0]["dtype"] == dtype
    assert result.metadata["bounds"] == {dim: [values.min(), values.max()] for dim, values in coords.items()}
    assert result.logical_object_count is None and not result.payload_validated


def test_public_writer_flat_bounds_override_rejected(tmp_path):
    data = StructuredData.from_numpy(np.zeros((2, 3, 4)),
                                     coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]})
    data.bounds = (0, 1, 0, 2, 0, 3)
    path = tmp_path / "flat_bounds.le"
    path.write_bytes(data.to_binary())
    with pytest.raises(ValueError, match="flat bounds are ambiguous"):
        inspect_le(path)


def test_invalid_singleton_bounds(tmp_path):
    header = {"data_shape": [1], "bounds": {"dim0": [0, 1]}, "dtype": "float32",
              "transform": None, "data_name": "density"}
    with pytest.raises(ValueError, match="sample extrema"):
        inspect_le(write_file(tmp_path, header, bytes(4)))


@pytest.mark.parametrize("association, width", [("point", 0), ("point", 1), ("cell", 3)])
def test_in_memory_grouping(association, width):
    original = np.array([2**60 + 1, -7, 2**60 + 1], dtype=np.int64)
    attrs = pd.DataFrame({"objects": original})
    ids = group_object_ids(attrs, object_attribute="objects", association=association, cell_width=width)
    assert ids.dtype == np.dtype("int64")
    assert ids.tolist() == [-7, 2**60 + 1]
    np.testing.assert_array_equal(attrs["objects"], original)


def test_in_memory_empty_grouping():
    ids = group_object_ids({"objects": np.array([], dtype=np.int64)},
                           object_attribute="objects", association="cell", cell_width=2)
    assert ids.shape == (0,) and ids.dtype == np.dtype("int64")


@pytest.mark.parametrize("attrs, attribute, association, width, message", [
    ({}, "objects", "cell", 3, "Missing grouping"),
    (None, "objects", "cell", 3, "Missing grouping"),
    ({"objects": [True]}, "objects", "cell", 3, "not boolean"),
    ({"objects": [np.nan]}, "objects", "cell", 3, "missing or nonfinite"),
    ({"objects": [np.inf]}, "objects", "cell", 3, "missing or nonfinite"),
    ({"objects": [None]}, "objects", "cell", 3, "numeric column"),
    ({"objects": ["1"]}, "objects", "cell", 3, "numeric column"),
    ({"objects": [[1]]}, "objects", "cell", 3, "one-dimensional"),
    ({"objects": [1]}, "objects", "point", 3, "point association"),
    ({"objects": [1]}, "objects", "cell", 1, "point association"),
    ({"objects": [1]}, "objects", "vertex", 1, "association must"),
    ({"objects": [1]}, "", "cell", 3, "nonempty string"),
    ({"objects": [1]}, "objects", "cell", -1, "cell_width"),
])
def test_invalid_in_memory_grouping(attrs, attribute, association, width, message):
    with pytest.raises(ValueError, match=message):
        group_object_ids(attrs, object_attribute=attribute, association=association, cell_width=width)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 8])
def test_declared_geometry_counts(tmp_path, width):
    header = {"format_version": 2, "vertex_shape": [8, 3], "cell_shape": [2, width]}
    result = inspect_le(write_file(tmp_path, header, bytes((24 + 2 * width) * 4)))
    assert result.vertex_count == 8 and result.cell_count == 2
    assert result.logical_object_count is None


@pytest.mark.parametrize("header, message", [
    ([], "must be a JSON object"),
    ({"format_version": 3}, "Unsupported LE format_version"),
    ({"vertex_shape": [-1, 3], "cell_shape": [0, 3]}, "vertex_shape"),
    ({"vertex_shape": [1, 2], "cell_shape": [0, 3]}, "three XYZ columns"),
    ({"vertex_shape": [3, 3], "cell_shape": [1, 6]}, "Ambiguous"),
    ({"vertex_shape": [0, 3], "cell_shape": [0, 3], "xarray_attrs": []}, "xarray_attrs"),
])
def test_invalid_headers(tmp_path, header, message):
    with pytest.raises(ValueError, match=message):
        inspect_le(write_file(tmp_path, header))


@pytest.mark.parametrize("change, message", [
    ({"dtype": "object"}, "dtype"),
    ({"dtype": "complex64"}, "dtype"),
    ({"shape": [2]}, "row count"),
    ({"byte_length": 5}, "byte_length"),
    ({"name": []}, "unique JSON scalar"),
])
def test_invalid_attribute_schema(tmp_path, change, message):
    column = dict(name="objects", dtype="int64", shape=[1], byte_length=8)
    column.update(change)
    header = dict(format_version=2, vertex_shape=[3, 3], cell_shape=[1, 3], cell_attrs=[column])
    with pytest.raises(ValueError, match=message):
        inspect_le(write_file(tmp_path, header, bytes(56)))


def test_duplicate_attribute_names(tmp_path):
    column = dict(name="objects", dtype="int64", shape=[1], byte_length=8)
    header = dict(format_version=2, vertex_shape=[3, 3], cell_shape=[1, 3], cell_attrs=[column, column])
    with pytest.raises(ValueError, match="unique JSON scalar"):
        inspect_le(write_file(tmp_path, header, bytes(64)))


@pytest.mark.parametrize("contents, message", [
    (b"\x02", "prefix"),
    ((100).to_bytes(4, "little") + b"{}", "Truncated JSON"),
    ((2).to_bytes(4, "little") + b"xx", "Invalid LE JSON"),
    ((1).to_bytes(4, "little") + b"\xff", "Invalid LE JSON"),
])
def test_invalid_prefix_and_json(tmp_path, contents, message):
    path = tmp_path / "invalid.le"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match=message):
        inspect_le(path)


def test_header_limit_prevents_header_read(tmp_path, monkeypatch):
    path = tmp_path / "huge.le"
    path.write_bytes((2**32 - 1).to_bytes(4, "little"))
    reads = track_reads(monkeypatch)
    with pytest.raises(ValueError, match="Header length"):
        inspect_le(path)
    assert reads == [(0, 4)]


@pytest.mark.parametrize("body", [bytes(47), bytes(49)])
def test_file_length_validation(tmp_path, body):
    header = dict(format_version=2, vertex_shape=[3, 3], cell_shape=[1, 3])
    with pytest.raises(ValueError, match="payload length"):
        inspect_le(write_file(tmp_path, header, body))


@pytest.mark.parametrize("version", [None, 1, 2])
@pytest.mark.parametrize("cell_shape", [[0, 0], [0, 1], [0, 3], [0, 8]])
def test_foundation_empty_layout_parity(tmp_path, version, cell_shape):
    header = {"vertex_shape": [0, 0], "cell_shape": cell_shape}
    if version is not None:
        header["format_version"] = version
    layout = validate_unstructured_layout(header, 0)
    path = write_file(tmp_path, header)
    result = inspect_le(path)
    decoded = LiquidEarthMesh.from_binary(path.read_bytes())
    assert result.shapes == {"vertex": layout["vertex_shape"], "cells": layout["cell_shape"]}
    assert result.shapes["vertex"] == decoded.vertex.shape == (0, 3)
    assert result.shapes["cells"] == decoded.cells.shape
    assert result.vertex_count == result.cell_count == 0
    assert result.format_version == layout["format_version"]
    assert not result.payload_validated


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("label", [7, 1.5, True, None, ""])
def test_foundation_scalar_attribute_names_parity(tmp_path, version, label):
    header = {"format_version": version, "vertex_shape": [4, 3], "cell_shape": [2, 2]}
    values = np.array([[100, 7], [200, 9]], dtype=np.float32)
    if version == 2:
        header["cell_attrs"] = [dict(name=name, dtype="float32", shape=[2], byte_length=8)
                                for name in (label, "objects")]
    else:
        header.update(cell_attr_shape=[2, 2], cell_attr_names=[label, "objects"])
    body = bytes(64) + values.tobytes("F")
    path = write_file(tmp_path, header, body)
    layout = validate_unstructured_layout(header, len(body))
    result = inspect_le(path, object_attribute="objects", association="cell")
    assert result.logical_object_count == 2 and result.object_ids == (7.0, 9.0)
    actual = result.attribute_schema["cell"][0]
    assert actual["name"] == label and type(actual["name"]) is type(label)
    assert actual["offset"] == layout["segments"][2]["offset"] == 64
    decoded = LiquidEarthMesh.from_binary(path.read_bytes())
    pd.testing.assert_index_equal(decoded.attributes.columns, pd.Index([label, "objects"]))


@pytest.mark.parametrize("rows, width", [(2, 3), (3, 2), (4, 2)])
def test_foundation_flattened_legacy_grouping_parity(tmp_path, monkeypatch, rows, width):
    header = {"vertex_shape": [4, 3], "cell_shape": [1, rows * width],
              "cell_attr_shape": [rows, 2], "cell_attr_names": [None, "objects"],
              "cell_attr_types": ["float64", "int32"]}
    attrs = np.column_stack((np.arange(rows), np.arange(rows) % 2)).astype(np.float32)
    body = bytes((12 + rows * width) * 4) + attrs.tobytes("F")
    path = write_file(tmp_path, header, body)
    layout = validate_unstructured_layout(header, len(body))
    decoded = LiquidEarthMesh.from_binary(path.read_bytes())
    reads = track_reads(monkeypatch)
    result = inspect_le(path, object_attribute="objects", association="cell")
    assert result.cell_count == rows
    assert result.shapes["cells"] == decoded.cells.shape == layout["cell_shape"] == (rows, width)
    assert result.shapes["wire_cells"] == layout["wire_cell_shape"] == (1, rows * width)
    column = result.attribute_schema["cell"][1]
    assert column["dtype"] == "float32"
    assert result.object_ids == (0.0, 1.0)
    assert result.grouping_validated and not result.payload_validated
    assert len(reads) == 3
    assert reads[-1] == (4 + reads[1][1] + column["offset"], rows * 4)


@pytest.mark.parametrize("version", [1, 2])
def test_foundation_point_offsets_after_cell_attributes(tmp_path, monkeypatch, version):
    header = {"format_version": version, "vertex_shape": [4, 3], "cell_shape": [4, 0]}
    cell_values = np.arange(4, dtype=np.float32)
    point_values = np.array([9, 7, 9, 7], dtype=np.float32)
    if version == 2:
        header.update(cell_attrs=[dict(name=None, dtype="float32", shape=[4], byte_length=16)],
                      vertex_attrs=[dict(name="objects", dtype="float32", shape=[4], byte_length=16)])
    else:
        header.update(cell_attr_shape=[4, 1], cell_attr_names=[None],
                      vertex_attr_shape=[4, 1], vertex_attr_names=["objects"])
    body = bytes(48) + cell_values.tobytes() + point_values.tobytes()
    layout = validate_unstructured_layout(header, len(body))
    path = write_file(tmp_path, header, body)
    reads = track_reads(monkeypatch)
    result = inspect_le(path, object_attribute="objects", association="point")
    column = result.attribute_schema["point"][0]
    assert column["offset"] == layout["segments"][-1]["offset"] == 64
    assert result.attribute_schema["cell"][0]["name"] is None
    assert result.object_ids == (7.0, 9.0)
    assert reads[-1] == (4 + reads[1][1] + 64, 16)


def test_foundation_big_endian_numeric_grouping(tmp_path):
    values = np.array([2**60 + 1, -7], dtype=">i8")
    header = {"format_version": 2, "vertex_shape": [4, 3], "cell_shape": [2, 2],
              "cell_attrs": [dict(name="objects", dtype=">i8", shape=[2], byte_length=16)]}
    path = write_file(tmp_path, header, bytes(64) + values.tobytes())
    result = inspect_le(path, object_attribute="objects", association="cell")
    assert result.attribute_schema["cell"][0]["dtype"] == ">i8"
    assert result.object_ids == (-7, 2**60 + 1)


@pytest.mark.parametrize("change", [
    {"format_version": True}, {"format_version": 3}, {"vertex_shape": [4, 2]},
    {"cell_shape": [5, 0]}, {"cell_shape": [2, 6]}, {"xarray_attrs": []},
    {"cell_attrs": [{"name": "a", "dtype": "object", "shape": [2], "byte_length": 16}]},
    {"cell_attrs": [{"dtype": "float32", "shape": [2], "byte_length": 8}]},
    {"cell_attrs": [{"name": "a", "dtype": "int64", "shape": [2], "byte_length": 8}]},
    {"format_version": 1, "cell_attr_shape": [0, 1], "cell_attr_names": ["a"]},
    {"format_version": 1, "cell_attr_shape": [2, 1], "cell_attr_names": ["a"], "cell_attr_types": []},
    {"format_version": 1, "cell_shape": [1, 6]},
])
def test_malformed_header_foundation_error_parity(tmp_path, change):
    header = {"format_version": 2, "vertex_shape": [4, 3], "cell_shape": [2, 2]}
    header.update(change)
    with pytest.raises(ValueError) as foundation_error:
        validate_unstructured_layout(header, 64)
    with pytest.raises(ValueError) as inspection_error:
        inspect_le(write_file(tmp_path, header, bytes(64)))
    assert str(inspection_error.value) == str(foundation_error.value)


@pytest.mark.parametrize("raw", [
    b'{"vertex_shape":[0,3],"cell_shape":[0,3],"cell_shape":[0,3]}',
    b'{"vertex_shape":[0,3],"cell_shape":[0,3],"xarray_attrs":{"id":1,"id":2}}',
    b'{"data_shape":[1],"bounds":{"dim0":[0,0],"dim0":[0,0]},"dtype":"float32","data_name":"a","transform":null}',
])
def test_duplicate_json_keys_parser_parity(tmp_path, raw):
    binary = len(raw).to_bytes(4, "little") + raw
    path = tmp_path / "duplicate.le"
    path.write_bytes(binary)
    with pytest.raises(ValueError, match="Duplicate JSON key") as foundation_error:
        read_le_header(binary)
    with pytest.raises(ValueError, match="Duplicate JSON key") as inspection_error:
        inspect_le(path)
    assert str(inspection_error.value) == str(foundation_error.value)


def test_foundation_header_limit_cannot_be_bypassed(tmp_path, monkeypatch):
    path = tmp_path / "oversize.le"
    path.write_bytes((MAX_LE_HEADER_BYTES + 1).to_bytes(4, "little"))
    reads = track_reads(monkeypatch)
    with pytest.raises(ValueError, match="Header length"):
        inspect_le(path, max_header_bytes=MAX_LE_HEADER_BYTES * 2)
    assert reads == [(0, 4)]
