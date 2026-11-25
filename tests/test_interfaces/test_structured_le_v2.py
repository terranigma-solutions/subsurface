"""Structured .le format 2: every data variable of a StructuredData, with named dimensions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from subsurface import StructuredData
from subsurface.api import LEValidationError, inspect_le, validate_le
from subsurface.api._le_grid_ops import load_le_grid
from subsurface.core.structs.base_structures import UnstructuredData

pytestmark = pytest.mark.core

DATA = Path(__file__).resolve().parents[1] / "data" / "le_ingest"
GOLDEN = DATA / "volume" / "dataset.le"
sys.path.insert(0, str(DATA))
from generate import dataset_v2  # noqa: E402


def split(data: bytes):
    length = int.from_bytes(data[:4], "little")
    return json.loads(data[4:4 + length]), data[4 + length:]


def join(header: dict, body: bytes) -> bytes:
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(4, "little") + encoded + body


def read(tmp_path, data: bytes) -> StructuredData:
    path = tmp_path / "grid.le"
    path.write_bytes(data)
    return StructuredData.from_binary_le(path)


def grid(**variables) -> StructuredData:
    ds = xr.Dataset(variables, coords={"x": np.linspace(0.0, 3.0, 4), "y": np.linspace(10.0, 12.0, 3),
                                       "z": np.linspace(-1.0, 0.0, 2)})
    return StructuredData(ds, next(iter(variables)))


SCALAR = (("x", "y", "z"), np.arange(24, dtype=np.float32).reshape(4, 3, 2))

# region Golden file


def test_the_writer_reproduces_the_golden_file():
    assert dataset_v2().to_binary(format_version=2) == GOLDEN.read_bytes()


def test_golden_header():
    header, body = split(GOLDEN.read_bytes())
    assert header == {
            "format_version": 2,
            "data_shape"    : [4, 3, 2],
            "dims"          : {"x": 4, "y": 3, "z": 2, "component": 3},
            "bounds"        : {"x": [100.0, 130.0], "y": [-20.0, 0.0], "z": [-50.0, -40.0]},
            "transform"     : None,
            "coords"        : {"component": ["x", "y", "z"]},
            "data_vars"     : [
                    {"name" : "temperature", "dims": ["x", "y", "z"], "dtype": "float32", "byte_length": 96,
                     "attrs": {"units": "degC"}},
                    {"name" : "displacement", "dims": ["x", "y", "z", "component"], "dtype": "float32",
                     "byte_length": 288, "attrs": {"units": "m", "kind": "displacement"}},
            ],
            "xarray_attrs"  : {},
    }
    assert len(body) == 96 + 288


def test_golden_body_is_little_endian_first_dimension_fastest():
    _, body = split(GOLDEN.read_bytes())
    temperature = np.frombuffer(body[:96], "<f4")
    displacement = np.frombuffer(body[96:], "<f4")
    nx, ny, nz = 4, 3, 2
    for i, j, k in [(0, 0, 0), (1, 0, 0), (3, 2, 0), (2, 1, 1)]:
        index = i + 10 * j + 100 * k
        assert temperature[i + nx * (j + ny * k)] == index
        for c in range(3):
            expected = index + 1000 * c if c % 2 == 0 else -index + 1000 * c
            assert displacement[i + nx * (j + ny * (k + nz * c))] == expected
    assert np.isnan(displacement[3 + nx * (2 + ny * (1 + nz * np.arange(3)))]).all()


def test_golden_file_reads_back_as_the_dataset():
    result = StructuredData.from_binary_le(GOLDEN)
    xr.testing.assert_identical(result.data, dataset_v2().data)
    assert result.active_data_array_name == "temperature"
    assert result.dtype == "float32"


def test_golden_file_is_a_valid_volume():
    report = validate_le(GOLDEN, family="volume", role="volume")
    assert report.data_shape == (4, 3, 2)
    assert report.data_name == "temperature"
    assert [(v.name, v.dims, v.shape) for v in report.grid_variables] == [
            ("temperature", ("x", "y", "z"), (4, 3, 2)),
            ("displacement", ("x", "y", "z", "component"), (4, 3, 2, 3))]

# endregion

# region Round trips


def test_version_1_stays_the_default():
    data = dataset_v2()
    assert data.to_binary() == data.to_binary(format_version=1)
    header, _ = split(data.to_binary())
    assert set(header) == {"data_shape", "bounds", "transform", "dtype", "data_name"}
    with pytest.raises(ValueError, match="format_version"):
        data.to_binary(format_version=3)


def test_the_active_variable_is_stored_first(tmp_path):
    data = dataset_v2()
    data.active_data_array_name = "displacement"
    header, _ = split(data.to_binary(format_version=2))
    assert [v["name"] for v in header["data_vars"]] == ["displacement", "temperature"]
    assert read(tmp_path, data.to_binary(format_version=2)).active_data_array_name == "displacement"


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16", "uint8"])
def test_variables_are_stored_in_the_grid_dtype(tmp_path, dtype):
    data = grid(a=SCALAR, b=(("x", "y", "z"), np.ones((4, 3, 2), np.int64)))
    data.dtype = dtype
    result = read(tmp_path, data.to_binary(format_version=2))
    assert result.dtype == dtype
    assert all(result.data[name].dtype == np.dtype(dtype) for name in ("a", "b"))
    np.testing.assert_array_equal(result.data["a"], SCALAR[1].astype(dtype))


def test_two_and_one_dimensional_grids(tmp_path):
    raster = StructuredData(xr.Dataset(
            {"rgb": (("x", "y", "channel"), np.arange(24, dtype=np.float32).reshape(4, 2, 3))},
            coords={"x": np.linspace(0.0, 3.0, 4), "y": np.array([5.0, 6.0]), "channel": ["r", "g", "b"]}), "rgb")
    xr.testing.assert_identical(read(tmp_path, raster.to_binary(format_version=2)).data, raster.data)
    line = StructuredData(xr.Dataset({"v": (("dim0",), np.arange(5, dtype=np.float32))},
                                     coords={"dim0": np.linspace(0.0, 1.0, 5)}), "v")
    xr.testing.assert_identical(read(tmp_path, line.to_binary(format_version=2)).data, line.data)


def test_singleton_axes_and_unlabelled_dimensions(tmp_path):
    ds = xr.Dataset({"v": (("x", "y", "z", "component"), np.ones((2, 1, 3, 2), np.float32))},
                    coords={"x": [0.0, 1.0], "y": [7.0], "z": [0.0, 0.5, 1.0]})
    data = StructuredData(ds, "v")
    header, _ = split(data.to_binary(format_version=2))
    assert header["coords"] == {} and header["bounds"]["y"] == [7.0, 7.0]
    xr.testing.assert_identical(read(tmp_path, data.to_binary(format_version=2)).data, ds)

# endregion

# region Writer errors


@pytest.mark.parametrize("change, match", [
        (lambda d: setattr(d, "bounds", (0, 1, 0, 1, 0, 1)), "bounds override"),
        (lambda d: d.data.__setitem__("t", (("y", "x", "z"), np.zeros((3, 4, 2)))), "must start with the spatial"),
        (lambda d: d.data.__setitem__("t", (("x", "y", "z", "time"), np.zeros((4, 3, 2, 2)))) or
                   d.data.coords.__setitem__("time", ("time", [0.0, 1.0], {"units": "s"})), "Attributes of coordinate"),
        (lambda d: d.data.coords.__setitem__("crs", 4326), "would be lost"),
        (lambda d: d.data.coords.__setitem__("x", [0.0, 1.0, 3.0, 4.0]), "regular and increasing"),
        (lambda d: d.data.coords.__setitem__("x", [3.0, 2.0, 1.0, 0.0]), "regular and increasing"),
        (lambda d: d.data.attrs.__setitem__("when", np.datetime64("2026-01-01")), "JSON-serializable"),
        (lambda d: d.data["a"].attrs.__setitem__("scale", float("nan")), "JSON-serializable"),
        (lambda d: d.data.__setitem__("s", (("x", "y", "z"), np.full((4, 3, 2), "a"))), "not numeric"),
])
def test_what_cannot_be_stored_raises(change, match):
    data = grid(a=SCALAR)
    change(data)
    with pytest.raises(ValueError, match=match):
        data.to_binary(format_version=2)


def test_the_writer_needs_fortran_order_and_spatial_coordinates():
    with pytest.raises(ValueError, match="Fortran"):
        grid(a=SCALAR).to_binary("C", format_version=2)
    data = StructuredData.from_numpy(SCALAR[1], data_array_name="a")
    with pytest.raises(ValueError, match="no coordinate"):
        data.to_binary(format_version=2)
    data = grid(a=(("x", "y", "z", "z2"), np.zeros((4, 3, 2, 1))))
    data.data = data.data.rename({"z2": "dim0"})
    with pytest.raises(ValueError, match="spatial name"):
        data.to_binary(format_version=2)


def test_the_header_must_fit_readers():
    data = grid(a=SCALAR)
    data.data.attrs["notes"] = "x" * (1024 * 1024)
    with pytest.raises(ValueError, match="at most"):
        data.to_binary(format_version=2)

# endregion

# region Reader errors


def _tamper(edit):
    header, body = split(GOLDEN.read_bytes())
    result = edit(header)
    return join(header, result if isinstance(result, bytes) else body)


@pytest.mark.parametrize("edit, match", [
        (lambda h: h.pop("coords"), "header fields"),
        (lambda h: h.__setitem__("format_version", 3), "format_version"),
        (lambda h: h.__setitem__("format_version", True), "format_version"),
        (lambda h: h.__setitem__("transform", [1]), "transforms"),
        (lambda h: h["dims"].__setitem__("x", 5), "sizes of data_shape"),
        (lambda h: h["dims"].__setitem__("extra", 2), "no data variable uses"),
        (lambda h: h["dims"].__setitem__("dim0", 3), "names x, y, z or dim0"),
        (lambda h: h["coords"].__setitem__("component", ["x", "y"]), "one label per index"),
        (lambda h: h["coords"].__setitem__("x", [1, 2, 3, 4]), "non-spatial"),
        (lambda h: h["coords"].__setitem__("component", ["x", True, "z"]), "strings or numbers"),
        (lambda h: h["data_vars"][1].__setitem__("name", "temperature"), "unique"),
        (lambda h: h["data_vars"][1].__setitem__("name", "component"), "unique"),
        (lambda h: h["data_vars"][1].__setitem__("dims", ["x", "y", "component", "z"]), "starting with"),
        (lambda h: h["data_vars"][1].__setitem__("dims", ["x", "y", "z", "time"]), "starting with"),
        (lambda h: h["data_vars"][1].__setitem__("dtype", ">f4"), "dtype"),
        (lambda h: h["data_vars"][1].__setitem__("dtype", []), "dtype"),
        (lambda h: h["data_vars"][1].__setitem__("dims", ["x", "y", "z", ["component"]]), "starting with"),
        (lambda h: h["dims"].__setitem__("x", 4.0), "integer sizes"),
        (lambda h: h["data_vars"][1].__setitem__("byte_length", 287), "byte_length"),
        (lambda h: h["data_vars"][1].pop("attrs"), "variable fields"),
        (lambda h: h.__setitem__("data_vars", []), "nonempty"),
        (lambda h: h.__setitem__("xarray_attrs", []), "JSON object"),
        (lambda h: h.__setitem__("bounds", {"x": [0, 1], "y": [0, 1]}), "bounds"),
        (lambda h: b"\0" * 383, "payload length"),
])
def test_the_reader_rejects_inconsistent_headers(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        read(tmp_path, _tamper(edit))

# endregion

# region Validator


@pytest.mark.parametrize("edit, code, match", [
        (lambda h: h["data_vars"][0].__setitem__("dtype", "float64") or
                   h["data_vars"][0].__setitem__("byte_length", 192) or b"\0" * (192 + 288), "unsupported_profile", "float32"),
        (lambda h: h["xarray_attrs"].__setitem__("timestamp", "2026-01-01T00:00:00Z"), "unsupported_profile", "time-series"),
        (lambda h: h["dims"].__setitem__("component", 4) or h["coords"].clear() or
                   h["data_vars"][1].__setitem__("byte_length", 384) or b"\0" * (96 + 384), "unsupported_profile", "2 or 3"),
        (lambda h: h["dims"].__setitem__("time", 2), "unsupported_profile", "component"),
        (lambda h: h["coords"].__setitem__("component", ["u", "v", "w"]), "unsupported_profile", "component labels"),
        (lambda h: h["coords"].__setitem__("component", ["z", "y", "x"]), "unsupported_profile", "component labels"),
        (lambda h: h["coords"].clear(), "unsupported_profile", "component labels"),
        (lambda h: h["dims"].__setitem__("x", 4.0), "malformed", "integer sizes"),
        (lambda h: h["dims"].__setitem__("component", 3.0), "malformed", "integer sizes"),
        (lambda h: h["data_vars"][1].__setitem__("dims", ["x", "y", "z", ["component"]]), "unsupported_profile", "dims"),
        (lambda h: h["data_vars"][0].__setitem__("name", "RGB"), "unsupported_profile", "reserved"),
        (lambda h: h["data_vars"].pop(1) and b"\0" * 96, "malformed", "no variable uses"),
        (lambda h: h.__setitem__("data_shape", [4, 3]), "unsupported_profile", "volumes"),
        (lambda h: h.__setitem__("transform", [1]), "unsupported_profile", "transforms"),
        (lambda h: h.__setitem__("extra", 1), "unsupported_profile", "header fields"),
        (lambda h: h["data_vars"][1].__setitem__("name", "temperature"), "malformed", "unique"),
        (lambda h: b"\0" * 380, "malformed", "payload length"),
])
def test_the_volume_profile(edit, code, match):
    with pytest.raises(LEValidationError, match=match) as error:
        validate_le(_tamper(edit), family="volume", role="volume")
    assert error.value.code == code


def test_infinite_values_are_rejected():
    data = dataset_v2()
    data.data["displacement"][0, 0, 0, 1] = np.inf
    with pytest.raises(LEValidationError, match="finite or NaN") as error:
        validate_le(data.to_binary(format_version=2), family="volume", role="volume")
    assert error.value.code == "unsupported_profile"


def test_a_format_2_grid_is_not_a_raster():
    with pytest.raises(LEValidationError) as error:
        validate_le(GOLDEN, family="raster", role="raster")
    assert error.value.code == "family_mismatch"

# endregion

# region Tools that read format 1 only


def test_inspection_summarizes_every_variable():
    result = inspect_le(GOLDEN)
    assert (result.file_kind, result.format_version, result.byte_size) == ("structured", 2, GOLDEN.stat().st_size)
    assert result.shapes == {"data": (4, 3, 2)}
    assert result.grid_sample_count == 24
    assert result.attribute_schema == {"cell": (), "point": (), "grid": (
            {"name": "temperature", "dtype": "float32", "shape": (4, 3, 2), "dims": ("x", "y", "z"),
             "byte_length": 96, "offset": 0},
            {"name": "displacement", "dtype": "float32", "shape": (4, 3, 2, 3), "dims": ("x", "y", "z", "component"),
             "byte_length": 288, "offset": 96})}
    assert result.metadata == {"bounds"   : {"x": [100.0, 130.0], "y": [-20.0, 0.0], "z": [-50.0, -40.0]},
                               "transform": None, "data_name": "temperature", "coords": {"component": ["x", "y", "z"]}}


def test_inspection_keeps_the_dataset_attrs(tmp_path):
    data = grid(t=SCALAR)
    data.data.attrs["timestamp"] = "2026-11-19T00:00:00Z"
    path = tmp_path / "grid.le"
    path.write_bytes(data.to_binary(format_version=2))
    assert inspect_le(path).metadata["timestamp"] == "2026-11-19T00:00:00Z"


@pytest.mark.parametrize("change, match", [
        (lambda h, b: (h, b[:-4]), "byte size"),
        (lambda h, b: ({**h, "format_version": 3}, b), "format_version"),
        (lambda h, b: ({**h, "bounds": {**h["bounds"], "x": [130.0, 100.0]}}, b), "sample extrema"),
        (lambda h, b: ({**h, "dtype": "float32"}, b), "header fields"),
])
def test_inspection_rejects_what_the_reader_rejects(tmp_path, change, match):
    path = tmp_path / "grid.le"
    path.write_bytes(join(*change(*split(GOLDEN.read_bytes()))))
    with pytest.raises(ValueError, match=match):
        inspect_le(path)
    with pytest.raises(ValueError):
        StructuredData.from_binary_le(path)


def test_inspection_refuses_grouping_a_grid():
    with pytest.raises(ValueError, match="grouping"):
        inspect_le(GOLDEN, object_attribute="temperature", association="point")


def test_grid_operations_refuse_variables_they_would_drop():
    with pytest.raises(ValueError, match="Extra arrays"):
        load_le_grid(GOLDEN)


def test_the_unstructured_reader_refuses_a_grid():
    with pytest.raises(ValueError, match="structured grid"):
        UnstructuredData.from_binary_le(GOLDEN)

# endregion
