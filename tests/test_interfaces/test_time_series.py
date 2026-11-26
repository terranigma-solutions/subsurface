import json
from datetime import datetime, timezone
from itertools import permutations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from subsurface.core.structs.base_structures import StructuredData, UnstructuredData
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh
from subsurface.modules.writer.time_series import export_time_series, export_volume_time_series, time_series_master


def trajectory(value=1.0):
    return UnstructuredData.from_array(
        vertex=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), cells="lines",
        vertex_attr=pd.DataFrame({"position": [0., 1., 2.], "temperature": [value, np.nan, 3.]}),
        xarray_attributes={"attribute_units": {"temperature": "C"}})


def volume():
    ds = xr.Dataset({"temperature": (("x", "time", "y", "z"),
                                    np.arange(16, dtype="float64").reshape(2, 2, 2, 2))},
                    coords={"x": [0., 1.], "y": [2., 3.], "z": [4., 5.],
                            "time": np.array(["2025-01-01T01:00:00.123456", "2025-01-01T00:00:00"],
                                             dtype="datetime64[us]")},
                    attrs={"units": "C", "source": "synthetic"})
    return StructuredData(ds, "temperature", dtype="float64", _bounds=(0., 2., 2., 4., 4., 6.))


def decode(path):
    return decode_bytes(path.read_bytes())


def decode_bytes(binary):
    length = int.from_bytes(binary[:4], "little")
    return json.loads(binary[4:4 + length]), binary[4 + length:]


def test_trajectory_frames_keep_stored_dtypes_when_values_are_integral(tmp_path):
    def frame(change):
        return UnstructuredData.from_array(
            vertex=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), cells="lines",
            vertex_attr=pd.DataFrame({"position": [0., 1., 2.], "change": change}))

    # Static writes still pack integral floats as int64; temporal frames must not.
    static_header, _ = decode_bytes(frame([0., 0., 0.]).to_binary())
    assert {a["name"]: a["dtype"] for a in static_header["vertex_attrs"]}["change"] == "int64"

    frames = [("2025-01-01T00:00:00Z", frame([0.5, -0.1, 0.])),
              ("2025-01-01T01:00:00Z", frame([0., 0., 0.]))]
    path = export_time_series(iter(frames), tmp_path / "series", time_series_id="m", kind="trajectory")
    layouts = []
    for entry in json.loads(path.read_text())["frames"]:
        header, _ = decode(path.parent / entry["path"])
        layouts.append([(a["name"], a["dtype"], a["byte_length"]) for a in header["vertex_attrs"]])
        mesh = LiquidEarthMesh.from_binary((path.parent / entry["path"]).read_bytes())
        assert mesh.points_attributes["change"].dtype == np.float32
    assert layouts[0] == layouts[1] == [("position", "float32", 12), ("change", "float32", 12)]


def test_trajectory_sorted_metadata_and_missing_values(tmp_path):
    first, second = trajectory(), trajectory(2.)
    frames = iter([("2025-01-01T02:00:00.123456+01:00", second),
                   (datetime(2025, 1, 1, tzinfo=timezone.utc), first)])
    path = export_time_series(frames, tmp_path / "series", time_series_id="measurements", kind="trajectory")
    index = json.loads(path.read_text())
    assert index == {"schema_version": 1, "time_series_id": "measurements", "kind": "trajectory",
                     "frames": [{"timestamp": "2025-01-01T00:00:00Z", "path": "trajectory_0001.le"},
                                {"timestamp": "2025-01-01T01:00:00.123456Z", "path": "trajectory_0000.le"}]}
    for entry in index["frames"]:
        mesh = LiquidEarthMesh.from_binary((path.parent / entry["path"]).read_bytes())
        assert mesh.data_attrs["timestamp"] == entry["timestamp"]
        assert mesh.data_attrs["time_series_id"] == "measurements"
        assert np.isnan(mesh.points_attributes["temperature"].iloc[1])
        np.testing.assert_array_equal(mesh.vertex, first.vertex)
    assert "timestamp" not in first.data.attrs
    assert sorted(p.name for p in path.parent.iterdir()) == ["series.json", "trajectory_0000.le", "trajectory_0001.le"]


def test_volume_axis_bounds_dtype_metadata(tmp_path):
    data = volume()
    path = export_volume_time_series(data, tmp_path / "volume", time_series_id="temperature", source_timezone="UTC")
    index = json.loads(path.read_text())
    assert [frame["timestamp"] for frame in index["frames"]] == ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00.123456Z"]
    for entry, i in zip(index["frames"], [1, 0]):
        header, body = decode(path.parent / entry["path"])
        assert header["xarray_attrs"] == dict(data.data.attrs, time_series_id="temperature", timestamp=entry["timestamp"])
        assert header["data_shape"] == [2, 2, 2]
        assert header["bounds"] == list(data.bounds)
        assert header["dtype"] == "float64"
        assert header["data_name"] == "temperature"
        np.testing.assert_array_equal(np.frombuffer(body, dtype="float64").reshape((2, 2, 2), order="F"),
                                      data.data.temperature.isel(time=i, drop=True).values)
    assert "timestamp" not in data.data.attrs


@pytest.mark.parametrize("timestamp", ["2025-01-01T00:00:00", "not-a-date", datetime(2025, 1, 1), 123])
def test_invalid_timestamp_fails_before_output(tmp_path, timestamp):
    with pytest.raises((ValueError, TypeError)):
        export_time_series([(timestamp, trajectory())], tmp_path / "bad", time_series_id="test", kind="trajectory")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("id,kind,data,error", [("", "trajectory", None, ValueError),
    ("../escape", "trajectory", None, ValueError), (None, "volume", None, ValueError),
    ("test", "mesh", None, ValueError), ("test", "volume", trajectory(), TypeError)])
def test_invalid_contract(tmp_path, id, kind, data, error):
    with pytest.raises(error):
        export_time_series([("2025-01-01T00:00:00Z", data)], tmp_path / "bad", time_series_id=id, kind=kind)
    assert not (tmp_path / "bad").exists()


def test_duplicate_normalized_time_cleans_output(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        export_time_series([(timestamp, trajectory()) for timestamp in
                            ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00+01:00"]],
                           tmp_path / "bad", time_series_id="test", kind="trajectory")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("change", ["xyz", "cells", "position", "schema", "units", "invalid_xyz"])
def test_trajectory_mismatch(tmp_path, change):
    first, second = trajectory(), trajectory()
    if change in ("xyz", "invalid_xyz"):
        second.data.vertex.values[0, 0] = np.nan if change == "invalid_xyz" else 10.
    elif change == "cells":
        second.data.cells.values[0] = [0, 2]
    elif change == "position":
        second.data["vertex_attrs"] = second.data.vertex_attrs.copy(deep=True)
        second.data.vertex_attrs.loc[dict(vertex_attr="position")] = [0., 1., 3.]
    elif change == "schema":
        second.data = second.data.assign_coords(vertex_attr=["position", "other"])
    else:
        second.data.attrs["attribute_units"] = {"temperature": "K"}
    with pytest.raises(ValueError):
        export_time_series([("2025-01-01T00:00:00Z", first), ("2025-01-01T01:00:00Z", second)],
                           tmp_path / "bad", time_series_id="test", kind="trajectory")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("location", ["vertex", "cell"])
@pytest.mark.parametrize("column", ["well_id", "measured_depths", "is_attr_point"])
def test_borehole_sample_identity_must_remain_fixed(tmp_path, location, column):
    first, second = trajectory(), trajectory(2.)
    for frame in (first, second):
        dim = location + "_attr"
        frame.data = frame.data.reindex({dim: list(frame.data[dim].values) + [column]},
                                        fill_value=1.).copy(deep=True)
    path = export_time_series([("2025-01-01T00:00:00Z", first), ("2025-01-01T01:00:00Z", second)],
                              tmp_path / "valid", time_series_id="test", kind="trajectory")
    assert len(json.loads(path.read_text())["frames"]) == 2
    second.data[location + "_attrs"].loc[{location + "_attr": column}] = 2.
    with pytest.raises(ValueError, match="sample identity"):
        export_time_series([("2025-01-01T00:00:00Z", first), ("2025-01-01T01:00:00Z", second)],
                           tmp_path / "bad", time_series_id="test", kind="trajectory")
    assert not (tmp_path / "bad").exists()


def test_borehole_well_id_mapping_must_remain_fixed(tmp_path):
    first, second = trajectory(), trajectory(2.)
    first.data.attrs["well_id_mapper"] = {"well-a": 0}
    second.data.attrs["well_id_mapper"] = {"well-b": 0}
    with pytest.raises(ValueError, match="sample identity"):
        export_time_series([("2025-01-01T00:00:00Z", first), ("2025-01-01T01:00:00Z", second)],
                           tmp_path / "bad", time_series_id="test", kind="trajectory")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("time_axis", range(4))
@pytest.mark.parametrize("spatial_dims", list(permutations(("x", "y", "z"))))
def test_volume_spatial_order_and_time_axis(tmp_path, spatial_dims, time_axis):
    data = volume()
    dims = list(spatial_dims)
    dims.insert(time_axis, "time")
    data.data = data.data.transpose(*dims)
    original_dims = data.active_data_array.dims
    if spatial_dims != ("x", "y", "z"):
        with pytest.raises(ValueError, match="ordered x, y, z"):
            export_volume_time_series(data, tmp_path / "bad", time_series_id="test", source_timezone="UTC")
        assert not (tmp_path / "bad").exists()
        snapshot = StructuredData(data.data.isel(time=0, drop=True), "temperature")
        with pytest.raises(ValueError, match="ordered x, y, z"):
            export_time_series([("2025-01-01T00:00:00Z", snapshot)], tmp_path / "bad",
                               time_series_id="test", kind="volume")
        assert not (tmp_path / "bad").exists()
    else:
        path = export_volume_time_series(data, tmp_path / "volume", time_series_id="test", source_timezone="UTC")
        _, body = decode(path.parent / "volume_0000.le")
        assert body == data.data.temperature.isel(time=0, drop=True).values.astype(data.dtype).tobytes("F")
    assert data.active_data_array.dims == original_dims


@pytest.mark.parametrize("change", ["coords", "bounds", "dtype", "field", "time"])
def test_volume_mismatch(tmp_path, change):
    data = volume()
    first = StructuredData(data.data.isel(time=0, drop=True), "temperature")
    second = StructuredData(data.data.isel(time=1, drop=True), "temperature")
    if change == "coords":
        second.data = second.data.assign_coords(x=[0., 2.])
    elif change == "bounds":
        second.bounds = (0., 3., 2., 4., 4., 6.)
    elif change == "dtype":
        second.dtype = "float64"
    elif change == "field":
        second.data = second.data.rename(temperature="other")
        second.active_data_array_name = "other"
    else:
        second = data
    with pytest.raises(ValueError):
        export_time_series([("2025-01-01T00:00:00Z", first), ("2025-01-01T01:00:00Z", second)],
                           tmp_path / "bad", time_series_id="test", kind="volume")
    assert not (tmp_path / "bad").exists()


def test_no_overwrite_or_consume(tmp_path):
    directory = tmp_path / "published"
    directory.mkdir()
    marker = directory / "series.json"
    marker.write_text("published")

    def frames():
        pytest.fail("existing output must be rejected before consuming input")
        yield

    with pytest.raises(FileExistsError):
        export_time_series(frames(), directory, time_series_id="test", kind="trajectory")
    assert marker.read_text() == "published"


def test_iterator_failure_never_advertises_index(tmp_path):
    def frames():
        yield "2025-01-01T00:00:00Z", trajectory()
        assert not (tmp_path / "bad" / "series.json").exists()
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        export_time_series(frames(), tmp_path / "bad", time_series_id="test", kind="trajectory")
    assert not (tmp_path / "bad").exists()


def test_empty_and_naive_volume(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        export_time_series([], tmp_path / "empty", time_series_id="test", kind="volume")
    with pytest.raises(ValueError, match="source_timezone"):
        export_volume_time_series(volume(), tmp_path / "naive", time_series_id="test")
    assert not (tmp_path / "naive").exists()


@pytest.mark.parametrize("time", ["2025-10-26T02:30:00", "2025-03-30T02:30:00"])
def test_ambiguous_or_nonexistent_volume_time(tmp_path, time):
    data = volume()
    data.data = data.data.isel(time=[0]).assign_coords(time=[np.datetime64(time)])
    with pytest.raises(Exception, match="2025"):
        export_volume_time_series(data, tmp_path / "bad", time_series_id="test", source_timezone="Europe/Berlin")
    assert not (tmp_path / "bad").exists()


def test_single_volume_frame(tmp_path):
    data = volume()
    data.data = data.data.isel(time=[0])
    path = export_volume_time_series(data, tmp_path / "single", time_series_id="test", source_timezone="UTC")
    assert len(json.loads(path.read_text())["frames"]) == 1


# region Format 2

def flow_volume(value=1.0, **variable_attrs):
    """A format 2 volume: a scalar temperature and a 3D velocity flow on a 3 x 2 x 2 grid."""
    shape = (3, 2, 2)
    temperature = np.full(shape, value, dtype="float32")
    flow = np.stack([np.full(shape, value), np.zeros(shape), -np.full(shape, value)], axis=-1).astype("float32")
    flow[2, 1, 1] = np.nan
    ds = xr.Dataset({"temperature": (("x", "y", "z"), temperature, {"units": "degC"}),
                     "flow"       : (("x", "y", "z", "component"), flow,
                                     dict({"units": "m/s", "kind": "velocity"}, **variable_attrs))},
                    coords={"x": [0., 1., 2.], "y": [5., 7.], "z": [-1., 0.], "component": ["x", "y", "z"]},
                    attrs={"source": "synthetic"})
    return StructuredData(ds, "temperature")


def test_volume_frames_in_format_2_keep_every_variable(tmp_path):
    frames = [("2025-01-01T01:00:00Z", flow_volume(2.0)), ("2025-01-01T00:00:00Z", flow_volume(1.0))]
    path = export_time_series(frames, tmp_path / "series", time_series_id="flow", kind="volume", format_version=2)
    index = json.loads(path.read_text())
    assert index["schema_version"] == 1
    assert [frame["path"] for frame in index["frames"]] == ["volume_0001.le", "volume_0000.le"]
    for entry, value in zip(index["frames"], [1.0, 2.0]):
        header, _ = decode(path.parent / entry["path"])
        assert header["format_version"] == 2
        assert header["xarray_attrs"] == {"source": "synthetic", "time_series_id": "flow", "timestamp": entry["timestamp"]}
        snapshot = StructuredData.from_binary_le(path.parent / entry["path"])
        expected = flow_volume(value).data
        assert list(snapshot.data.data_vars) == ["temperature", "flow"]
        xr.testing.assert_identical(snapshot.data.drop_attrs(deep=False), expected.drop_attrs(deep=False))


def test_volume_time_series_in_format_2(tmp_path):
    data = flow_volume().data
    stacked = xr.concat([data, data * 2], dim=xr.DataArray(
            np.array(["2025-01-01T00:00:00", "2025-01-01T01:00:00"], dtype="datetime64[us]"), dims="time", name="time"))
    for name in stacked.data_vars:
        stacked[name].attrs = data[name].attrs
    stacked.attrs = data.attrs
    path = export_volume_time_series(StructuredData(stacked, "temperature"), tmp_path / "series", time_series_id="flow",
                                     source_timezone="UTC", format_version=2)
    last = StructuredData.from_binary_le(path.parent / json.loads(path.read_text())["frames"][-1]["path"])
    np.testing.assert_array_equal(last.data["flow"].values, data["flow"].values * 2)
    assert last.data["flow"].attrs == {"units": "m/s", "kind": "velocity"}


@pytest.mark.parametrize("second, match", [
        (lambda: flow_volume(kind="displacement"), "attribute schema"),
        (lambda: StructuredData(flow_volume().data.drop_vars("flow").drop_vars("component"), "temperature"),
         "attribute schema"),
        (lambda: StructuredData(flow_volume().data, "temperature", _bounds=(0., 2., 5., 7., -1., 0.)), "bounds override"),
])
def test_format_2_frames_must_match(tmp_path, second, match):
    with pytest.raises(ValueError, match=match):
        export_time_series([("2025-01-01T00:00:00Z", flow_volume()), ("2025-01-01T01:00:00Z", second())],
                           tmp_path / "bad", time_series_id="flow", kind="volume", format_version=2)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind, data, format_version", [("trajectory", trajectory, 2), ("volume", flow_volume, 3),
                                                        ("volume", flow_volume, True)])
def test_format_2_is_for_volumes(tmp_path, kind, data, format_version):
    with pytest.raises(ValueError, match="format"):
        export_time_series([("2025-01-01T00:00:00Z", data())], tmp_path / "bad", time_series_id="flow", kind=kind,
                           format_version=format_version)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("format_version", [1, 2])
def test_time_series_master(tmp_path, format_version):
    frames = [("2025-01-01T00:00:00Z", flow_volume(1.0)), ("2025-01-01T01:00:00Z", flow_volume(2.0))]
    if format_version == 1:
        frames = [(stamp, StructuredData(data.data.drop_vars(["flow", "component"]), "temperature")) for stamp, data in frames]
    path = export_time_series(frames, tmp_path / "series", time_series_id="flow", kind="volume", format_version=format_version)
    index = json.loads(path.read_text())

    binary = time_series_master(frames[-1][1], index, directory="time-series/flow/abc", revision="abc",
                                format_version=format_version)
    header, _ = decode_bytes(binary)
    assert header.get("format_version", 1) == format_version
    assert header["xarray_attrs"] == {
            "source"     : "synthetic", "time_series_id": "flow", "timestamp": "2025-01-01T01:00:00Z",
            "time_series": {"schema_version": 2, "role": "master", "time_series_id": "flow", "revision": "abc",
                            "kind"          : "volume", "directory": "time-series/flow/abc", "index": "series.json",
                            "frames"        : index["frames"]}}
    master_path = tmp_path / "master.le"
    master_path.write_bytes(binary)
    np.testing.assert_array_equal(StructuredData.from_binary_le(master_path).data["temperature"].values,
                                  frames[-1][1].data["temperature"].values)
    assert "time_series" not in frames[-1][1].data.attrs


def test_trajectory_master_names_its_well(tmp_path):
    path = export_time_series([("2025-01-01T00:00:00Z", trajectory())], tmp_path / "series", time_series_id="m",
                              kind="trajectory")
    header, _ = decode_bytes(time_series_master(trajectory(), json.loads(path.read_text()), directory="d", revision="r"))
    assert header["xarray_attrs"]["well_id_mapper"] == {"m": 0}
    with pytest.raises(TypeError):
        time_series_master(flow_volume(), json.loads(path.read_text()), directory="d", revision="r")

# endregion
