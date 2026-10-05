"""Synthetic inspection tests: no optional readers, plotting, or network."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest

from subsurface import inspect_le
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


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
    header = {"data_shape": [2, 3, 4], "bounds": [0, 1, 0, 2, 0, 3], "dtype": "float32",
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
    ({"data_shape": [2, 3]}, "data_shape"),
    ({"data_shape": [2, True, 4]}, "data_shape"),
    ({"dtype": "object"}, "dtype"),
    ({"bounds": [1, 0, 0, 2, 0, 3]}, "bounds"),
    ({"bounds": [0, float("nan"), 0, 2, 0, 3]}, "bounds"),
    ({"bounds": [0, 1]}, "bounds"),
    ({"data_name": 5}, "data_name"),
    ({"transform": np.eye(4).tolist()}, "transforms"),
    ({"cell_shape": [0, 3]}, "Mixed"),
])
def test_invalid_structured_headers(tmp_path, change, message):
    header = {"data_shape": [2, 3, 4], "bounds": [0, 1, 0, 2, 0, 3],
              "dtype": "float32", "transform": None, "data_name": "density"}
    header.update(change)
    with pytest.raises(ValueError, match=message):
        inspect_le(write_file(tmp_path, header, bytes(96)))


@pytest.mark.parametrize("width", [1, 2, 3, 4, 8])
def test_declared_geometry_counts(tmp_path, width):
    header = {"format_version": 2, "vertex_shape": [8, 3], "cell_shape": [2, width]}
    result = inspect_le(write_file(tmp_path, header, bytes((24 + 2 * width) * 4)))
    assert result.vertex_count == 8 and result.cell_count == 2
    assert result.logical_object_count is None


@pytest.mark.parametrize("header, message", [
    ([], "must be an object"),
    ({"format_version": 3}, "Unsupported format_version"),
    ({"vertex_shape": [-1, 3], "cell_shape": [0, 3]}, "Invalid vertex_shape"),
    ({"vertex_shape": [1, 2], "cell_shape": [0, 3]}, "three coordinates"),
    ({"vertex_shape": [3, 3], "cell_shape": [1, 6]}, "ambiguous"),
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
    ({"name": None}, "unique strings"),
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
    with pytest.raises(ValueError, match="unique strings"):
        inspect_le(write_file(tmp_path, header, bytes(64)))


@pytest.mark.parametrize("contents, message", [
    (b"\x02", "prefix"),
    ((100).to_bytes(4, "little") + b"{}", "Truncated JSON"),
    ((2).to_bytes(4, "little") + b"xx", "Invalid JSON"),
    ((1).to_bytes(4, "little") + b"\xff", "Invalid JSON"),
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
