"""Synthetic LE foundation contracts; no external data or optional backends."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from subsurface.core.structs.base_structures import UnstructuredData
from subsurface.core.structs.base_structures import _liquid_earth_mesh as le


def _frame(header, payload=b""):
    raw = json.dumps(header).encode("utf-8")
    return len(raw).to_bytes(4, "little") + raw + payload


def _header(version=2):
    header = {
        "vertex_shape": [4, 3],
        "cell_shape": [2, 2],
        "xarray_attrs": {"crs": "local", "nested": {"units": ["m", "s"]}},
    }
    if version == 2:
        header.update(format_version=2, cell_attrs=[], vertex_attrs=[])
    else:
        header.update(cell_attr_shape=[0, 0], cell_attr_names=[],
                      vertex_attr_shape=[0, 0], vertex_attr_names=[])
        if version is not None:
            header["format_version"] = version
    return header


def _geometry(order="F"):
    vertex = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
    cells = np.array([[0, 2], [1, 3]], dtype=np.int32)
    return vertex, cells, vertex.tobytes(order) + cells.tobytes(order)


def _public_read(tmp_path, binary, reader, order=None):
    path = tmp_path / "mesh.le"
    if reader == "current":
        path.write_bytes(binary)
        args = (path,)
        read = UnstructuredData.from_binary_le
    else:
        size = int.from_bytes(binary[:4], "little")
        sidecar = tmp_path / "mesh.json"
        sidecar.write_bytes(binary[4:4 + size])
        path.write_bytes(binary[4 + size:])
        args = (path, sidecar)
        read = UnstructuredData.from_binary_le_legacy
    return read(*args) if order is None else read(*args, order=order)


def _assert_attributes(actual, expected):
    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_allclose(actual.to_numpy(dtype=float),
                               expected.to_numpy(dtype=float), equal_nan=True)


@pytest.mark.parametrize("reader", ["current", "sidecar"])
@pytest.mark.parametrize("order", [None, "C"])
@pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 8],
                         ids=["zero-width-points", "points", "lines", "tris", "tet", "hex"])
def test_public_writer_round_trip(tmp_path, reader, order, width):
    vertex = np.arange(24, dtype=float).reshape(8, 3) / 7
    vertex[2, 1] = np.nan
    cells = (np.empty((8, 0), dtype=np.int32) if width == 0
             else np.arange(width, dtype=np.int32).reshape(1, width))
    cell_attrs = pd.DataFrame({"density": np.arange(len(cells)) + 0.25})
    vertex_attrs = pd.DataFrame({"sample": np.arange(8) + 0.5,
                                 "missing": [np.nan] + [1.25] * 7})
    metadata = {"crs": "synthetic", "nested": {"units": ["m"]}, "enabled": True}
    source = UnstructuredData.from_array(
        vertex, cells, cells_attr=cell_attrs, vertex_attr=vertex_attrs,
        xarray_attributes=metadata,
    )
    binary = source.to_binary() if order is None else source.to_binary(order=order)
    if reader == "sidecar":
        payload, header = source.to_binary_legacy(order=order or "F")
        assert _frame(header, payload) == binary
    restored = _public_read(tmp_path, binary, reader, order)
    np.testing.assert_allclose(restored.vertex, vertex.astype("float32"), equal_nan=True)
    np.testing.assert_array_equal(restored.cells, cells)
    assert restored.cells.shape == cells.shape
    assert restored.data.attrs == metadata
    _assert_attributes(restored.attributes, cell_attrs)
    _assert_attributes(restored.points_attributes, vertex_attrs)


@pytest.mark.parametrize("reader", ["current", "sidecar"])
@pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 8])
def test_empty_writer_round_trip(tmp_path, reader, width):
    source = UnstructuredData.from_array(
        np.empty((0, 3)), np.empty((0, width), dtype=np.int32),
        cells_attr=pd.DataFrame({"not_serialized": pd.Series(dtype="float32")}),
        vertex_attr=pd.DataFrame({"also_not_serialized": pd.Series(dtype="float32")}),
        xarray_attributes={"empty": True},
    )
    binary = source.to_binary()
    size = int.from_bytes(binary[:4], "little")
    header = json.loads(binary[4:4 + size])
    # The existing writer omits empty columns, so their names cannot round-trip.
    assert header["cell_attrs"] == header["vertex_attrs"] == []
    restored = _public_read(tmp_path, binary, reader)
    assert restored.vertex.shape == (0, 3)
    assert restored.cells.shape == (0, width)
    assert restored.attributes.shape == restored.points_attributes.shape == (0, 0)
    assert restored.data.attrs == {"empty": True}


@pytest.mark.parametrize("order", ["F", "C"])
def test_low_level_wire_attribute_dtypes(order):
    columns = {}
    for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        values = [1, np.iinfo(dtype).max] if dtype.startswith("uint") else [-1, np.iinfo(dtype).max]
        columns[dtype] = pd.Series(values, dtype=dtype)
    columns.update(
        boolean=pd.Series([True, False], dtype="bool"),
        fractional16=pd.Series([0.125, np.nan], dtype="float16"),
        fractional32=pd.Series([1.25, np.nan], dtype="float32"),
        fractional64=pd.Series([2.125, np.nan], dtype="float64"),
        integral16=pd.Series([1, -2], dtype="float16"),
        integral32=pd.Series([1, -2], dtype="float32"),
        integral64=pd.Series([2 ** 40, -3], dtype="float64"),
    )
    attrs = pd.DataFrame(columns)
    vertex = np.arange(6, dtype=float).reshape(2, 3)
    cells = np.array([[0], [1]], dtype=np.int32)
    mesh = le.LiquidEarthMesh(vertex, cells, attrs, attrs.copy(), {"source": "synthetic"})
    binary = mesh.to_binary(order=order)
    header, _ = le.read_le_header(binary)
    expected_dtypes = {name: str(series.dtype) for name, series in attrs.items()}
    expected_dtypes.update(fractional16="float32", fractional32="float32", fractional64="float32",
                           integral16="int64", integral32="int64", integral64="int64")
    for association in ("cell_attrs", "vertex_attrs"):
        assert {meta["name"]: meta["dtype"] for meta in header[association]} == expected_dtypes
    restored = le.LiquidEarthMesh.from_binary(binary, order=order)
    np.testing.assert_array_equal(restored.vertex, vertex)
    np.testing.assert_array_equal(restored.cells, cells)
    assert restored.data_attrs == mesh.data_attrs
    for actual in (restored.attributes, restored.points_attributes):
        assert list(actual.columns) == list(attrs.columns)
        for name, dtype in expected_dtypes.items():
            assert actual[name].dtype == np.dtype(dtype)
            np.testing.assert_array_equal(actual[name].to_numpy(), attrs[name].to_numpy().astype(dtype))


@pytest.mark.parametrize("version", [None, 1])
@pytest.mark.parametrize("order", [None, "C"])
@pytest.mark.parametrize("reader", ["current", "sidecar"])
def test_legacy_float32_attribute_matrices(tmp_path, version, order, reader):
    wire_order = order or "F"
    vertex, cells, payload = _geometry(wire_order)
    cell_attrs = np.array([[1.25, np.nan], [2.5, 3.75]], dtype=np.float32)
    vertex_attrs = np.arange(8, dtype=np.float32).reshape(4, 2) / 4
    header = _header(version)
    header.update(cell_attr_shape=[2, 2], cell_attr_names=["density", "missing"],
                  vertex_attr_shape=[4, 2], vertex_attr_names=["density", "sample"])
    payload += cell_attrs.tobytes(wire_order) + vertex_attrs.tobytes(wire_order)
    layout = le.validate_unstructured_layout(header, len(payload))
    assert layout["format_version"] == 1
    assert layout["payload_length"] == len(payload)
    segments = layout["segments"]
    assert len(segments) == 4
    for segment, shape, length, offset in zip(segments[2:], [(2, 2), (4, 2)], [16, 32], [64, 80]):
        assert np.dtype(segment["dtype"]) == np.dtype("float32")
        assert tuple(segment["shape"]) == shape
        assert segment["byte_length"] == length
        assert segment["offset"] == offset
    restored = _public_read(tmp_path, _frame(header, payload), reader, order)
    np.testing.assert_array_equal(restored.vertex, vertex)
    np.testing.assert_array_equal(restored.cells, cells)
    _assert_attributes(restored.attributes, pd.DataFrame(cell_attrs, columns=header["cell_attr_names"]))
    _assert_attributes(restored.points_attributes, pd.DataFrame(vertex_attrs, columns=header["vertex_attr_names"]))
    assert restored.data.attrs == header["xarray_attrs"]


def test_read_header_offset_and_payload_are_independent():
    header = _header()
    binary = _frame(header, b"not interpreted by the header reader")
    actual, offset = le.read_le_header(binary)
    assert actual == header
    assert offset == 4 + int.from_bytes(binary[:4], "little")
    assert binary[offset:] == b"not interpreted by the header reader"


@pytest.mark.parametrize("raw", [b"", b"{", b"[]", b"null", b"1", b'"text"',
                                 b"\xff", b'{"cell_shape":[0,0],"cell_shape":[1,2]}',
                                 b'{"xarray_attrs":{"a":1,"a":2}}'])
def test_read_header_rejects_invalid_json_objects(raw):
    with pytest.raises(ValueError):
        le.read_le_header(len(raw).to_bytes(4, "little") + raw)


@pytest.mark.parametrize("binary", [b"", b"\x01", b"\x01\x00", b"\x01\x00\x00",
                                    (100).to_bytes(4, "little") + b"{}",
                                    (16 * 1024 * 1024 + 1).to_bytes(4, "little"),
                                    (2 ** 32 - 1).to_bytes(4, "little")])
def test_read_header_rejects_short_or_unbounded_lengths(binary):
    with pytest.raises(ValueError):
        le.read_le_header(binary)


def test_read_header_accepts_exact_size_limit():
    limit = 16 * 1024 * 1024
    raw = b"{}" + b" " * (limit - 2)
    assert le.read_le_header(limit.to_bytes(4, "little") + raw) == ({}, limit + 4)
    with pytest.raises(ValueError):
        le.read_le_header((limit + 1).to_bytes(4, "little") + raw + b" ")


def test_deeply_nested_header_has_a_clear_error():
    raw = b'{"nested":' + b'[' * 10000 + b'0' + b']' * 10000 + b'}'
    with pytest.raises(ValueError, match="Invalid LE JSON"):
        le.read_le_header(len(raw).to_bytes(4, "little") + raw)


def test_layout_segment_offsets_and_associations():
    header = _header()
    header["cell_attrs"] = [
        {"name": "flag", "dtype": "bool", "shape": [2], "byte_length": 2},
        {"name": "id", "dtype": "int64", "shape": [2], "byte_length": 16},
    ]
    header["vertex_attrs"] = [
        {"name": "id", "dtype": "uint16", "shape": [4], "byte_length": 8},
    ]
    original = copy.deepcopy(header)
    layout = le.validate_unstructured_layout(header, payload_length=90)
    assert header == original
    assert layout["format_version"] == 2
    assert tuple(layout["vertex_shape"]) == (4, 3)
    assert tuple(layout["wire_cell_shape"]) == tuple(layout["cell_shape"]) == (2, 2)
    assert layout["payload_length"] == 90
    assert layout["data_attrs"] == header["xarray_attrs"]
    expected = [("vertex", "float32", (4, 3), 48, 0),
                ("cells", "int32", (2, 2), 16, 48),
                ("flag", "bool", (2,), 2, 64),
                ("id", "int64", (2,), 16, 66),
                ("id", "uint16", (4,), 8, 82)]
    assert len(layout["segments"]) == len(expected)
    for segment, (name, dtype, shape, length, offset) in zip(layout["segments"], expected):
        assert {"name", "association", "dtype", "shape", "byte_length", "offset"} <= segment.keys()
        assert segment["name"] == name
        assert np.dtype(segment["dtype"]) == np.dtype(dtype)
        assert tuple(segment["shape"]) == shape
        assert segment["byte_length"] == length
        assert segment["offset"] == offset
    assert layout["segments"][2]["association"] == layout["segments"][3]["association"]
    assert layout["segments"][3]["association"] != layout["segments"][4]["association"]
    assert le.validate_unstructured_layout(header) == layout


@pytest.mark.parametrize("version", [None, 1, 2])
@pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 8])
def test_canonical_cell_widths_never_inferred(version, width):
    header = _header(version)
    header.update(vertex_shape=[8, 3], cell_shape=[1, width])
    cells = np.arange(width, dtype=np.int32).reshape(1, width)
    payload = np.arange(24, dtype=np.float32).tobytes("F") + cells.tobytes("F")
    layout = le.validate_unstructured_layout(header, len(payload))
    assert layout["format_version"] == (1 if version is None else version)
    assert tuple(layout["wire_cell_shape"]) == tuple(layout["cell_shape"]) == (1, width)
    restored = le.LiquidEarthMesh.from_binary(_frame(header, payload))
    assert restored.cells.shape == (1, width)
    np.testing.assert_array_equal(restored.cells, cells)


@pytest.mark.parametrize("version", [None, 1])
@pytest.mark.parametrize("wire_width, rows, shape", [
    (10, 0, (5, 2)), (9, 0, (3, 3)), (6, 3, (3, 2)), (6, 2, (2, 3)),
    (12, 6, (6, 2)), (12, 4, (4, 3)),
    (4, 2, (2, 2)), (8, 4, (4, 2)),
])
def test_legacy_flattened_cells_only_infer_lines_or_triangles(version, wire_width, rows, shape):
    header = _header(version)
    header.update(vertex_shape=[wire_width, 3], cell_shape=[1, wire_width])
    vertex = np.arange(wire_width * 3, dtype=np.float32).reshape(wire_width, 3)
    flat = np.arange(wire_width, dtype=np.int32).reshape(1, wire_width)
    payload = vertex.tobytes("F") + flat.tobytes("F")
    if rows:
        header.update(cell_attr_shape=[rows, 1], cell_attr_names=["id"])
        payload += np.arange(rows, dtype=np.float32).tobytes()
    layout = le.validate_unstructured_layout(header, len(payload))
    assert tuple(layout["wire_cell_shape"]) == (1, wire_width)
    assert tuple(layout["cell_shape"]) == shape
    restored = le.LiquidEarthMesh.from_binary(_frame(header, payload))
    np.testing.assert_array_equal(restored.cells, flat.reshape(shape, order="C"))
    if rows:
        np.testing.assert_array_equal(restored.attributes["id"], np.arange(rows))


@pytest.mark.parametrize("version, width, rows", [
    (None, 6, 0), (1, 6, 0), (1, 12, 0), (1, 6, 1), (1, 6, 4),
    (1, 5, 0), (1, 7, 0), (2, 6, 0), (2, 9, 0), (2, 10, 0),
])
def test_ambiguous_or_nonhistorical_flattened_cells_rejected(version, width, rows):
    header = _header(version)
    header["cell_shape"] = [1, width]
    if rows:
        header.update(cell_attr_shape=[rows, 1], cell_attr_names=["id"])
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("key, value", [
    ("format_version", 0), ("format_version", 3), ("format_version", -1),
    ("format_version", True), ("format_version", 2.0), ("format_version", "2"),
    ("format_version", None), ("vertex_shape", None), ("vertex_shape", [4]),
    ("vertex_shape", [4, 3, 1]), ("vertex_shape", [4, 2]),
    ("vertex_shape", [-1, 3]), ("vertex_shape", [4.0, 3]),
    ("vertex_shape", [True, 3]), ("vertex_shape", ["4", 3]),
    ("cell_shape", None), ("cell_shape", [2]), ("cell_shape", [2, 2, 1]),
    ("cell_shape", [-1, 2]), ("cell_shape", [2, -1]),
    ("cell_shape", [2, 5]), ("cell_shape", [2, 6]),
    ("cell_shape", [2.0, 2]), ("cell_shape", [2, True]),
    ("cell_attrs", {}), ("cell_attrs", [None]), ("vertex_attrs", "bad"),
    ("xarray_attrs", []), ("xarray_attrs", "bad"),
])
def test_invalid_layout_schema(key, value):
    header = _header()
    header[key] = value
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("header", [None, [], "bad", {}, {"vertex_shape": [4, 3]},
                                    {"cell_shape": [2, 2]}])
def test_missing_or_nonobject_layout_schema(header):
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("association", ["cell_attrs", "vertex_attrs"])
@pytest.mark.parametrize("key, value", [
    ("dtype", "object"), ("dtype", "complex64"), ("dtype", "str"),
    ("dtype", "datetime64[ns]"), ("dtype", "not-a-dtype"), ("dtype", None),
    ("shape", []), ("shape", [2, 1]), ("shape", [-1]),
    ("shape", [2.0]), ("shape", [True]), ("shape", [99]),
    ("byte_length", -1), ("byte_length", 0), ("byte_length", 7),
    ("byte_length", 8.0), ("byte_length", True), ("name", []),
])
def test_invalid_v2_attribute_metadata(association, key, value):
    header = _header()
    rows = 2 if association == "cell_attrs" else 4
    meta = {"name": "value", "dtype": "float32", "shape": [rows], "byte_length": rows * 4}
    meta[key] = value
    header[association] = [meta]
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("key", ["name", "dtype", "shape", "byte_length"])
def test_v2_attribute_metadata_requires_all_fields(key):
    header = _header()
    meta = {"name": "value", "dtype": "float32", "shape": [2], "byte_length": 8}
    del meta[key]
    header["cell_attrs"] = [meta]
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("version", [None, 1, 2])
@pytest.mark.parametrize("association", ["cell", "vertex"])
def test_duplicate_attribute_names_rejected_per_association(version, association):
    header = _header(version)
    rows = 2 if association == "cell" else 4
    if version == 2:
        meta = {"name": "duplicate", "dtype": "float32", "shape": [rows], "byte_length": rows * 4}
        header[association + "_attrs"] = [meta, meta.copy()]
    else:
        header[association + "_attr_shape"] = [rows, 2]
        header[association + "_attr_names"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("key, value", [
    ("cell_attr_shape", [3, 1]), ("cell_attr_shape", [-1, 1]),
    ("cell_attr_shape", [2, 1, 1]), ("cell_attr_shape", [2.0, 1]),
    ("cell_attr_shape", [True, 1]), ("cell_attr_shape", [2, -1]),
    ("cell_attr_names", []), ("cell_attr_names", ["a", "b"]),
    ("cell_attr_names", "a"), ("vertex_attr_shape", [3, 1]),
    ("vertex_attr_names", []),
])
def test_invalid_legacy_attribute_matrix_schema(key, value):
    header = _header(1)
    header.update(cell_attr_shape=[2, 1], cell_attr_names=["a"],
                  vertex_attr_shape=[4, 1], vertex_attr_names=["b"])
    header[key] = value
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("delta", [-64, -1, 1, 8])
def test_layout_requires_exact_payload_length(delta):
    with pytest.raises(ValueError):
        le.validate_unstructured_layout(_header(), payload_length=64 + delta)


@pytest.mark.parametrize("reader", ["lowlevel", "current", "sidecar"])
@pytest.mark.parametrize("malformation", ["version", "shape", "dtype", "byte-length", "duplicate-name"])
def test_readers_apply_layout_validation(tmp_path, reader, malformation):
    header = _header()
    _, _, payload = _geometry()
    meta = {"name": "id", "dtype": "int32", "shape": [2], "byte_length": 8}
    header["cell_attrs"] = [meta]
    payload += np.array([10, 20], dtype=np.int32).tobytes()
    if malformation == "version":
        header["format_version"] = 999
    elif malformation == "shape":
        header["vertex_shape"] = [4, 3, 1]
    elif malformation == "dtype":
        meta["dtype"] = "object"
    elif malformation == "byte-length":
        meta["byte_length"] = 7
    else:
        header["cell_attrs"].append(meta.copy())
        payload += np.array([30, 40], dtype=np.int32).tobytes()
    with pytest.raises(ValueError):
        binary = _frame(header, payload)
        if reader == "lowlevel":
            le.LiquidEarthMesh.from_binary(binary)
        else:
            _public_read(tmp_path, binary, reader)


@pytest.mark.parametrize("reader", ["lowlevel", "current", "sidecar"])
@pytest.mark.parametrize("malformation", ["truncated-vertex", "truncated-cells", "truncated-attr",
                                          "trailing", "negative-index", "large-index",
                                          "duplicate-json-key"])
def test_readers_reject_invalid_payloads(tmp_path, reader, malformation):
    header = _header()
    vertex, cells, payload = _geometry()
    if malformation == "truncated-vertex":
        payload = payload[:47]
    elif malformation == "truncated-cells":
        payload = payload[:-1]
    elif malformation == "truncated-attr":
        header["cell_attrs"] = [{"name": "id", "dtype": "int64", "shape": [2], "byte_length": 16}]
        payload += b"\x00" * 15
    elif malformation == "trailing":
        payload += b"\x00"
    elif malformation in ("negative-index", "large-index"):
        cells[0, 0] = -1 if malformation == "negative-index" else len(vertex)
        payload = vertex.tobytes("F") + cells.tobytes("F")
    binary = _frame(header, payload)
    if malformation == "duplicate-json-key":
        raw = json.dumps(header).encode()[:-1] + b',"cell_shape":[2,2]}'
        binary = len(raw).to_bytes(4, "little") + raw + payload
    with pytest.raises(ValueError):
        if reader == "lowlevel":
            le.LiquidEarthMesh.from_binary(binary)
        else:
            _public_read(tmp_path, binary, reader)


@pytest.mark.parametrize("reader", ["current", "sidecar"])
def test_public_order_is_keyword_only(tmp_path, reader):
    _, _, payload = _geometry()
    binary = _frame(_header(), payload)
    _public_read(tmp_path, binary, reader)
    with pytest.raises(TypeError):
        if reader == "current":
            UnstructuredData.from_binary_le(tmp_path / "mesh.le", "F")
        else:
            UnstructuredData.from_binary_le_legacy(tmp_path / "mesh.le", tmp_path / "mesh.json", "F")


@pytest.mark.parametrize("reader", ["current", "sidecar"])
@pytest.mark.parametrize("dtype", ["bool", "int64", "uint64"])
def test_public_homogeneous_attributes_preserve_values(tmp_path, reader, dtype):
    values = ([True, False, True, False] if dtype == "bool" else
              [1, 2 ** 53 + 1, 3, np.iinfo(dtype).max])
    attrs = pd.DataFrame({"id": pd.Series(values, dtype=dtype)})
    source = UnstructuredData.from_array(
        np.zeros((4, 3)), "points", cells_attr=attrs, vertex_attr=attrs,
        xarray_attributes={"type": dtype},
    )
    restored = _public_read(tmp_path, source.to_binary(), reader)
    for actual in (restored.attributes, restored.points_attributes):
        np.testing.assert_array_equal(actual["id"].to_numpy(), attrs["id"].to_numpy())
        assert actual["id"].dtype == np.dtype(dtype)


@pytest.mark.parametrize("version", [None, 1, 2])
@pytest.mark.parametrize("cell_shape", [[0, 0], [0, 1], [0, 3], [0, 8]])
def test_empty_legacy_and_current_shapes(version, cell_shape):
    header = _header(version)
    header.update(vertex_shape=[0, 0], cell_shape=cell_shape)
    mesh = le.LiquidEarthMesh.from_binary(_frame(header))
    assert mesh.vertex.shape == (0, 3)
    assert mesh.cells.shape == tuple(cell_shape)


@pytest.mark.parametrize("reader", ["current", "sidecar"])
@pytest.mark.parametrize("dtype", [">i2", "<u8", ">f4", "float16", "float64"])
def test_supported_explicit_attribute_dtypes(tmp_path, reader, dtype):
    header = _header()
    vertex, cells, payload = _geometry()
    values = np.array([1, 2], dtype=dtype)
    header["cell_attrs"] = [{"name": "id", "dtype": dtype, "shape": [2],
                              "byte_length": values.nbytes}]
    mesh = le.LiquidEarthMesh.from_binary(_frame(header, payload + values.tobytes()))
    np.testing.assert_array_equal(mesh.attributes["id"].to_numpy(), values)
    restored = _public_read(tmp_path, _frame(header, payload + values.tobytes()), reader)
    np.testing.assert_array_equal(restored.attributes["id"].to_numpy(), values)


def test_boolean_payload_is_not_silently_coerced():
    header = _header()
    _, _, payload = _geometry()
    header["cell_attrs"] = [{"name": "flag", "dtype": "bool", "shape": [2], "byte_length": 2}]
    with pytest.raises(ValueError, match="Boolean"):
        le.LiquidEarthMesh.from_binary(_frame(header, payload + b"\x00\x02"))


@pytest.mark.parametrize("order", ["A", "K", "", None])
def test_unsupported_order_rejected(order):
    _, _, payload = _geometry()
    with pytest.raises(ValueError, match="order"):
        le.LiquidEarthMesh.from_binary(_frame(_header(), payload), order=order)


def test_python_integer_length_arithmetic_does_not_overflow():
    header = _header()
    header["vertex_shape"] = [2 ** 62, 3]
    layout = le.validate_unstructured_layout(header)
    assert layout["segments"][0]["byte_length"] == 12 * 2 ** 62
    with pytest.raises(ValueError, match="payload length"):
        le.LiquidEarthMesh.from_binary(_frame(header))


def test_structured_header_is_not_a_mesh():
    header = _header()
    header["data_shape"] = [4, 3, 2]
    with pytest.raises(ValueError, match="structured"):
        le.validate_unstructured_layout(header)


@pytest.mark.parametrize("reader", ["current", "sidecar"])
@pytest.mark.parametrize("name", [7, 1.5, True, None, ""])
def test_shipped_writer_scalar_column_names(tmp_path, reader, name):
    source = UnstructuredData.from_array(
        np.zeros((4, 3)), "points", cells_attr=pd.DataFrame({name: [1, 2, 3, 4]}),
        vertex_attr=pd.DataFrame({name: [5, 6, 7, 8]}),
    )
    restored = _public_read(tmp_path, source.to_binary(), reader)
    for actual, expected in ((restored.attributes, source.attributes),
                             (restored.points_attributes, source.points_attributes)):
        pd.testing.assert_index_equal(actual.columns, expected.columns)
        np.testing.assert_array_equal(actual.values, expected.values)


@pytest.mark.parametrize("reader", ["current", "sidecar"])
@pytest.mark.parametrize("version", [None, 1, 2])
def test_zero_byte_point_rows_cannot_bypass_length_validation(tmp_path, reader, version):
    header = _header(version)
    header.update(vertex_shape=[0, 3], cell_shape=[10 ** 9, 0])
    if version != 2:
        header["cell_attr_shape"] = [10 ** 9, 0]
    with pytest.raises(ValueError, match="Zero-width"):
        _public_read(tmp_path, _frame(header), reader)


def test_invalid_layout_is_rejected_before_array_decoding(monkeypatch):
    def unexpected_decode(*args, **kwargs):
        pytest.fail("Array decoding occurred before layout validation")
    monkeypatch.setattr(np, "frombuffer", unexpected_decode)
    header = _header()
    header["vertex_shape"] = [2 ** 62, 3]
    with pytest.raises(ValueError, match="payload length"):
        le.LiquidEarthMesh.from_binary(_frame(header))


def test_legacy_type_labels_are_descriptive_not_wire_dtypes():
    header = _header(1)
    _, _, payload = _geometry()
    header.update(cell_attr_shape=[2, 1], cell_attr_names=[7], cell_attr_types=["object"])
    binary = _frame(header, payload + np.array([1.25, 2.5], dtype=np.float32).tobytes())
    mesh = le.LiquidEarthMesh.from_binary(binary)
    assert mesh.attributes[7].dtype == np.dtype('float32')
    header['cell_attr_types'] = []
    with pytest.raises(ValueError, match="types"):
        le.validate_unstructured_layout(header)
