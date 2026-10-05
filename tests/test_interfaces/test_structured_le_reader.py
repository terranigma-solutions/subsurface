import json
import sys

import numpy as np
import pytest
import xarray as xr

from subsurface import StructuredData
from subsurface.core.structs.base_structures.structured_data import StructuredDataType


pytestmark = pytest.mark.core


def write_file(tmp_path, header, payload=b""):
    encoded = json.dumps(header).encode("utf-8")
    path = tmp_path / "grid.le"
    path.write_bytes(len(encoded).to_bytes(4, "little") + encoded + payload)
    return path


def valid_header():
    return {"data_shape": [2, 3, 4], "bounds": {"x": [-2, 1], "y": [10, 14], "z": [0, 9]},
            "transform": None, "dtype": "float32", "data_name": "density"}


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8", "int16", "int32",
                                  "int64", "uint8", "uint16", "uint32", "uint64", ">i4", "<f8"])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 3, 4), (2, 1, 4), (2, 3, 1), (1, 1, 1), (2, 3), (1,)])
def test_public_round_trip(tmp_path, dtype, shape):
    values = np.arange(np.prod(shape)).reshape(shape)
    dims = StructuredData._default_dim_names(len(shape))
    coords = {dim: 10 * (i + 1) + np.arange(size) * (i + 2)
              for i, (dim, size) in enumerate(zip(dims, shape))}
    source = StructuredData.from_numpy(values, coords=coords, data_array_name="density")
    source.dtype = dtype
    source.data["discarded"] = source.active_data_array + 1
    source.data.attrs["crs"] = "not serialized"
    source.active_data_array.attrs["units"] = "not serialized"
    path = tmp_path / "grid.le"
    original = source.to_binary()
    path.write_bytes(original)

    result = StructuredData.from_binary_le(path)
    assert result.active_data_array_name == "density"
    assert result.type == StructuredDataType.REGULAR_AXIS_ALIGNED
    assert result.dtype == dtype
    assert result.shape == shape
    assert result.active_data_array.dims == tuple(dims)
    np.testing.assert_array_equal(result.values, values.astype(dtype))
    assert result.values.dtype == np.dtype(dtype)
    for dim in dims:
        np.testing.assert_allclose(result.data.coords[dim], coords[dim])
    assert list(result.data.data_vars) == ["density"]
    assert result.data.attrs == result.active_data_array.attrs == {}
    rewritten = result.to_binary()
    original_size = int.from_bytes(original[:4], "little")
    rewritten_size = int.from_bytes(rewritten[:4], "little")
    # JSON integers may become floats when coordinates are reconstructed.
    assert json.loads(original[4:4 + original_size]) == json.loads(rewritten[4:4 + rewritten_size])
    assert original[4 + original_size:] == rewritten[4 + rewritten_size:]
    path.write_bytes(rewritten)
    xr.testing.assert_identical(result.data, StructuredData.from_binary_le(path).data)
    result.values.flat[0] = 5  # The reconstructed container is writable.


def test_default_writer_precision(tmp_path):
    source = StructuredData.from_numpy(np.full((2, 3, 4), 1 / 3),
                                       coords={"x": [1, 2], "y": [4, 5, 6], "z": [8, 9, 10, 11]})
    path = tmp_path / "default.le"
    path.write_bytes(source.to_binary())
    result = StructuredData.from_binary_le(str(path))
    assert result.values.dtype == np.dtype("float32")
    np.testing.assert_array_equal(result.values, source.values.astype("float32"))


@pytest.mark.parametrize("values,dtype", [
    ([float("nan"), float("inf"), -float("inf")], "float64"),
    ([-2**63, 2**53 + 1, 2**63 - 1], "int64"),
    ([0, 2**53 + 1, 2**64 - 1], "uint64"),
])
def test_numeric_values_are_not_coerced(tmp_path, values, dtype):
    source = StructuredData.from_numpy(np.array(values, dtype=dtype), coords={"dim0": [0, 1, 2]})
    source.dtype = dtype
    path = tmp_path / "numeric.le"
    path.write_bytes(source.to_binary())
    result = StructuredData.from_binary_le(path)
    np.testing.assert_array_equal(result.values, source.values)


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 3, 1)])
def test_vtk_reference_coordinates_and_order(tmp_path, shape):
    # Reproduce the active adjacent VTK writer without importing VTK or plotting.
    outer_bounds = [-7, 9, 20, 35, 100, 140]
    bounds = {}
    for i, axis in enumerate("xyz"):
        low, high = outer_bounds[2 * i:2 * i + 2]
        bounds[axis] = [low, low + ((high - low) / shape[i]) * (shape[i] - 1)]
    x, y, z = np.indices(shape)
    expected = x + 10 * y + 100 * z
    header = valid_header()
    header.update(data_shape=list(shape), bounds=bounds)
    result = StructuredData.from_binary_le(write_file(tmp_path, header, expected.astype("<f4").tobytes(order="F")))
    np.testing.assert_array_equal(result.values, expected)
    for i, axis in enumerate("xyz"):
        np.testing.assert_allclose(result.data.coords[axis],
                                   np.linspace(*outer_bounds[2 * i:2 * i + 2], num=shape[i], endpoint=False))
    np.testing.assert_array_equal(result.bounds["x"], bounds["x"])


def test_bounds_key_order_does_not_transpose_axes(tmp_path):
    header = valid_header()
    header["bounds"] = {axis: header["bounds"][axis] for axis in ("z", "x", "y")}
    values = np.arange(24).reshape(2, 3, 4)
    result = StructuredData.from_binary_le(write_file(tmp_path, header, values.astype("f4").tobytes(order="F")))
    assert result.active_data_array.dims == ("x", "y", "z")
    np.testing.assert_array_equal(result.values, values)


@pytest.mark.parametrize("key,value,message", [
    ("data_shape", [], "data_shape"), ("data_shape", [2, 3, 4, 5], "data_shape"),
    ("data_shape", [2, 0, 4], "data_shape"), ("data_shape", [2, -1, 4], "data_shape"),
    ("data_shape", [True, 3, 4], "data_shape"), ("data_shape", [2.0, 3, 4], "data_shape"),
    ("data_shape", "2,3,4", "data_shape"), ("data_shape", [2**63, 3, 4], "payload length"),
    ("bounds", [0, 1, 0, 1, 0, 1], "bounds"), ("bounds", {}, "bounds"),
    ("bounds", {"x": [0, 1], "y": [0, 1], "depth": [0, 1]}, "bounds"),
    ("data_name", "", "data_name"), ("data_name", "  ", "data_name"),
    ("data_name", None, "data_name"), ("data_name", 1, "data_name"), ("data_name", "x", "data_name"),
    ("dtype", "object", "dtype"), ("dtype", "complex64", "dtype"), ("dtype", "bool", "dtype"),
    ("dtype", "S4", "dtype"), ("dtype", "datetime64[ns]", "dtype"),
    ("dtype", "V4", "dtype"), ("dtype", "not_a_dtype", "dtype"), ("dtype", None, "dtype"),
    ("dtype", ["int32"], "dtype"), ("dtype", "(2,)i4", "dtype"),
    ("transform", [], "transforms"), ("transform", 0, "transforms"),
    ("transform", np.eye(4).tolist(), "transforms"),
])
def test_invalid_header_values(tmp_path, key, value, message):
    header = valid_header()
    header[key] = value
    with pytest.raises(ValueError, match=message):
        StructuredData.from_binary_le(write_file(tmp_path, header, bytes(96)))


@pytest.mark.parametrize("pair", [[0], [0, 1, 2], "01", [False, 1], ["0", 1], [None, 1],
                                  [float("nan"), 1], [0, float("inf")], [2, 1], [1, 1],
                                  [-1e308, 1e308], [0, 10**400]])
def test_invalid_axis_bounds(tmp_path, pair):
    header = valid_header()
    header["bounds"]["x"] = pair
    with pytest.raises(ValueError, match="axis x"):
        StructuredData.from_binary_le(write_file(tmp_path, header, bytes(96)))


def test_singleton_requires_equal_extrema(tmp_path):
    header = valid_header()
    header["data_shape"][0] = 1
    with pytest.raises(ValueError, match="axis x"):
        StructuredData.from_binary_le(write_file(tmp_path, header, bytes(48)))


@pytest.mark.parametrize("size,pair", [(1, [2**53 + 1, 2**53 + 1]),
                                     (1, [2**53, 2**53 + 1]), (2, [2**53, 2**53 + 1])])
def test_lossy_integer_extrema(tmp_path, size, pair):
    header = valid_header()
    header["data_shape"][0] = size
    header["bounds"]["x"] = pair
    with pytest.raises(ValueError, match="precision"):
        StructuredData.from_binary_le(write_file(tmp_path, header, bytes(size * 3 * 4 * 4)))


def test_collapsed_float_spacing(tmp_path):
    header = valid_header()
    header["bounds"]["y"] = [1.0, np.nextafter(1.0, 2.0)]
    with pytest.raises(ValueError, match="spacing"):
        StructuredData.from_binary_le(write_file(tmp_path, header, bytes(96)))


@pytest.mark.parametrize("size", [0, 95, 97, 192])
def test_payload_length(tmp_path, size):
    with pytest.raises(ValueError, match="payload length"):
        StructuredData.from_binary_le(write_file(tmp_path, valid_header(), bytes(size)))


@pytest.mark.parametrize("header", [None, [], 1, {}, {"vertex_shape": [2, 3]},
                                   dict(valid_header(), version=2)])
def test_wrong_schema(tmp_path, header):
    with pytest.raises(ValueError, match="header fields"):
        StructuredData.from_binary_le(write_file(tmp_path, header))


@pytest.mark.parametrize("field", list(valid_header()))
def test_missing_header_fields(tmp_path, field):
    header = valid_header()
    del header[field]
    with pytest.raises(ValueError, match="header fields"):
        StructuredData.from_binary_le(write_file(tmp_path, header, bytes(96)))


@pytest.mark.parametrize("content,message", [
    (b"", "prefix"), (b"\x01\x00\x00", "prefix"),
    (bytes(4), "header size"), ((1024 * 1024 + 1).to_bytes(4, "little"), "header size"),
    ((10).to_bytes(4, "little") + b"{}", "Truncated.*header"),
    ((1).to_bytes(4, "little") + b"\xff", "JSON"),
    ((1).to_bytes(4, "little") + b"{", "JSON"),
])
def test_malformed_prefix_and_json(tmp_path, content, message):
    path = tmp_path / "bad.le"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        StructuredData.from_binary_le(path)


@pytest.mark.parametrize("duplicate", ["field", "bound", "escaped_bound"])
def test_duplicate_json_keys(tmp_path, duplicate):
    encoded = json.dumps(valid_header())
    if duplicate == "field":
        encoded = encoded[:-1] + ', "dtype": "float32"}'
    else:
        key = "x" if duplicate == "bound" else r"\u0078"
        encoded = encoded.replace('"x": [-2, 1]', f'"x": [-2, 1], "{key}": [-2, 1]')
    header = encoded.encode("utf-8")
    path = tmp_path / "duplicate.le"
    path.write_bytes(len(header).to_bytes(4, "little") + header + bytes(96))
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        StructuredData.from_binary_le(path)


@pytest.mark.parametrize("opening,closing", [(b"[", b"]"), (b'{"nested":', b"}")])
def test_excessive_json_nesting(tmp_path, opening, closing):
    depth = sys.getrecursionlimit() * 2
    header = opening * depth + b"0" + closing * depth
    path = tmp_path / "nested.le"
    path.write_bytes(len(header).to_bytes(4, "little") + header)
    with pytest.raises(ValueError, match="JSON header exceeds supported nesting depth"):
        StructuredData.from_binary_le(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        StructuredData.from_binary_le(tmp_path / "missing.le")
