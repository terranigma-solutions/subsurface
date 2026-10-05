import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from subsurface import StructuredData, UnstructuredData


def _split_binary(binary):
    header_length = int.from_bytes(binary[:4], byteorder="little")
    return json.loads(binary[4:4 + header_length]), binary[4 + header_length:]


def _pack_binary(header, body):
    header_bytes = json.dumps(header).encode("utf-8")
    return len(header_bytes).to_bytes(4, byteorder="little") + header_bytes + body


@pytest.fixture
def temporal_attrs():
    return {
        "time_series_id": "temperature-series",
        "timestamp": "2025-12-12T10:45:39.123Z",
        "attribute_units": {"temperature": "degC"},
        "source_filename": "frame.csv",
    }


@pytest.fixture
def structured_snapshot():
    snapshot = StructuredData.from_numpy(
        np.arange(24).reshape(2, 3, 4) + 0.25,
        coords={"x": [10, 20], "y": [30, 40, 50], "z": [60, 70, 80, 90]},
        data_array_name="temperature",
    )
    return snapshot


def test_static_structured_bytes_unchanged(structured_snapshot):
    snapshot = structured_snapshot
    expected_header = {
        "data_shape": (2, 3, 4),
        "bounds": {"x": (10, 20), "y": (30, 50), "z": (60, 90)},
        "transform": None,
        "dtype": "float32",
        "data_name": "temperature",
    }
    expected_body = snapshot.values.astype("float32").tobytes("F")
    expected_binary = _pack_binary(expected_header, expected_body)
    assert snapshot.to_binary() == expected_binary

    # Static attrs were never exported, including attrs that cannot be JSON encoded.
    snapshot.data.attrs.update({"description": "static", "source": object()})
    assert snapshot.to_binary() == expected_binary
    assert snapshot.default_data_array_to_binary_legacy() == (expected_body, expected_header)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("explicit_bounds", [False, True])
def test_structured_temporal_header_and_body(
    structured_snapshot, temporal_attrs, dtype, order, explicit_bounds
):
    snapshot = structured_snapshot
    snapshot.dtype = dtype
    if explicit_bounds:
        snapshot.bounds = (10, 25, 30, 55, 60, 95)
    static_header, static_body = _split_binary(snapshot.to_binary(order=order))
    snapshot.data.attrs.update(temporal_attrs)

    header, body = _split_binary(snapshot.to_binary(order=order))
    assert header == {**static_header, "xarray_attrs": temporal_attrs}
    assert body == static_body
    np.testing.assert_array_equal(
        np.frombuffer(body, dtype=dtype).reshape(header["data_shape"], order=order),
        snapshot.values.astype(dtype),
    )
    legacy_body, legacy_header = snapshot.default_data_array_to_binary_legacy(order=order)
    assert legacy_body == body
    assert json.loads(json.dumps(legacy_header)) == header
    assert snapshot.data.attrs == temporal_attrs


@pytest.mark.parametrize("key", ["timestamp", "time_series_id"])
def test_structured_either_temporal_key_enables_metadata(structured_snapshot, key):
    structured_snapshot.data.attrs[key] = "example"
    header, _ = _split_binary(structured_snapshot.to_binary())
    assert header["xarray_attrs"] == {key: "example"}


@pytest.mark.parametrize(
    "value, error",
    [(datetime(2025, 12, 12), TypeError), (np.int64(1), TypeError),
     (np.array([1]), TypeError), (float("nan"), ValueError),
     (float("inf"), ValueError)],
)
def test_structured_temporal_attrs_must_be_json_safe(structured_snapshot, value, error):
    structured_snapshot.data.attrs.update({"timestamp": "2025-12-12T10:45:39Z", "source": value})
    with pytest.raises(error):
        structured_snapshot.to_binary()
    with pytest.raises(error):
        structured_snapshot.default_data_array_to_binary_legacy()


@pytest.fixture
def unstructured_snapshot():
    return UnstructuredData.from_array(
        vertex=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        cells=np.array([[0, 1], [1, 2]]),
        cells_attr=pd.DataFrame({"strain": [0.25, 0.5]}),
        vertex_attr=pd.DataFrame({"temperature": [10.25, np.nan, 11.75]}),
    )


def test_static_unstructured_v2_bytes_unchanged(unstructured_snapshot):
    snapshot = unstructured_snapshot
    header = {
        "format_version": 2,
        "vertex_shape": (3, 3),
        "cell_shape": (2, 2),
        "cell_attrs": [{"name": "strain", "dtype": "float32", "shape": [2], "byte_length": 8}],
        "vertex_attrs": [{"name": "temperature", "dtype": "float32", "shape": [3], "byte_length": 12}],
        "xarray_attrs": {},
    }
    body = (
        snapshot.vertex.astype("float32").tobytes("F")
        + snapshot.cells.astype("int32").tobytes("F")
        + np.array([0.25, 0.5], dtype="float32").tobytes()
        + np.array([10.25, np.nan, 11.75], dtype="float32").tobytes()
    )
    assert snapshot.to_binary() == _pack_binary(header, body)
    assert snapshot.to_binary_legacy() == (body, header)


@pytest.mark.parametrize("split_files", [False, True])
@pytest.mark.parametrize("format_version", [1, 2])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_unstructured_metadata_roundtrip(
    tmp_path, unstructured_snapshot, temporal_attrs, split_files, format_version, with_metadata
):
    snapshot = unstructured_snapshot
    static_header, static_body = _split_binary(snapshot.to_binary())
    if with_metadata:
        snapshot.data.attrs.update(temporal_attrs)
    header, body = _split_binary(snapshot.to_binary())
    assert body == static_body
    assert header == {**static_header, "xarray_attrs": temporal_attrs if with_metadata else {}}
    if format_version == 1:
        header = {
            "vertex_shape": [3, 3], "cell_shape": [2, 2],
            "cell_attr_shape": [2, 1], "vertex_attr_shape": [3, 1],
            "cell_attr_names": ["strain"], "vertex_attr_names": ["temperature"],
        }
        if with_metadata:
            header["xarray_attrs"] = temporal_attrs
    elif not with_metadata:
        del header["xarray_attrs"]

    binary_path = tmp_path / "snapshot.le"
    if split_files:
        binary_path.write_bytes(body)
        header_path = tmp_path / "snapshot.json"
        header_path.write_text(json.dumps(header), encoding="utf-8")
        restored = UnstructuredData.from_binary_le_legacy(binary_path, header_path)
    else:
        binary_path.write_bytes(_pack_binary(header, body))
        restored = UnstructuredData.from_binary_le(binary_path)

    assert restored.data.attrs == (temporal_attrs if with_metadata else {})
    np.testing.assert_array_equal(restored.vertex, snapshot.vertex)
    np.testing.assert_array_equal(restored.cells, snapshot.cells)
    np.testing.assert_array_equal(restored.attributes.values, snapshot.attributes.values)
    np.testing.assert_array_equal(restored.points_attributes.values, snapshot.points_attributes.values)
    if with_metadata:
        assert _split_binary(restored.to_binary())[0]["xarray_attrs"] == temporal_attrs
