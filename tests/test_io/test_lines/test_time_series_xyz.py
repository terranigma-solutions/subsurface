"""Synthetic CSV coverage for the explicit-XYZ trajectory example."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[3] / "examples" / "time_series_boreholes.py"
    spec = importlib.util.spec_from_file_location("time_series_boreholes_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def csv_frame(example):
    table = pd.DataFrame({
        "X [m]": [10., 11., 12.], "Y [m]": [20., 20., 20.],
        "Z [m]": [0., -1., -2.], "position [m]": [0., 0.5, 1.],
    })
    for source in list(example.FIELDS)[1:]:
        table[source] = [np.nan, 1., 2.]
    return table


def write_frame(directory, table, local_time="2025-12-12_10-45-39"):
    path = directory / f"{local_time}_strain_and_temperature.csv"
    table.to_csv(path, index=False)
    return path


def frames(example, directory, **kwargs):
    options = dict(source_timezone="UTC", time_series_id="synthetic", single_trajectory=True)
    options.update(kwargs)
    return (frame for _, frame in example.iter_borehole_frames(directory, **options))


def test_sorted_frames_preserve_xyz_connectivity_measurements_and_metadata(example, csv_frame, tmp_path):
    later = csv_frame.copy()
    later["temperature [\u00b0C]"] += 0.5
    write_frame(tmp_path, later, "2025-12-12_11-45-39")
    first_path = write_frame(tmp_path, csv_frame)
    result = list(frames(example, tmp_path, source_timezone="Europe/Berlin"))
    assert len(result) == 2
    assert [frame.data.attrs["timestamp"] for frame in result] == [
        "2025-12-12T09:45:39Z", "2025-12-12T10:45:39Z",
    ]
    for frame in result:
        np.testing.assert_array_equal(frame.vertex, csv_frame[example.XYZ_COLUMNS])
        np.testing.assert_array_equal(frame.cells, [[0, 1], [1, 2]])
        assert frame.n_points == 3
        assert frame.n_elements == 2
        assert set(frame.points_attributes) == {name for name, _ in example.FIELDS.values()}
        assert np.isnan(frame.points_attributes["raw_strain"].iloc[0])
        assert all(name.isascii() for name in frame.points_attributes)
        json.dumps(frame.data.attrs, allow_nan=False)
    for source, (canonical, _) in example.FIELDS.items():
        np.testing.assert_allclose(result[0].points_attributes[canonical], csv_frame[source], equal_nan=True)
    assert result[0].data.attrs["source_filename"] == first_path.name
    assert result[0].data.attrs["source_timezone"] == "Europe/Berlin"
    assert result[0].data.attrs["attribute_units"]["raw_strain"] == "um/m"
    assert result[0].data.attrs["source_fields"]["temperature"] == "temperature [\u00b0C]"
    assert result[0].data.attrs["time_series_id"] == "synthetic"


@pytest.mark.parametrize("options,match", [
    ({"single_trajectory": False}, "affirmation"),
    ({"source_timezone": None}, "source_timezone is required"),
    ({"source_timezone": ""}, "source_timezone is required"),
    ({"source_timezone": "Not/AZone"}, "Invalid source timezone"),
    ({"time_series_id": " "}, "nonempty"),
])
def test_required_confirmations(example, csv_frame, tmp_path, options, match):
    write_frame(tmp_path, csv_frame)
    with pytest.raises(ValueError, match=match):
        list(frames(example, tmp_path, **options))


@pytest.mark.parametrize("local_time,match", [
    ("2025-10-26_02-30-00", "Ambiguous"),
    ("2025-03-30_02-30-00", "Nonexistent"),
])
def test_reject_dst_fold_and_gap(example, csv_frame, tmp_path, local_time, match):
    write_frame(tmp_path, csv_frame, local_time)
    with pytest.raises(ValueError, match=match):
        list(frames(example, tmp_path, source_timezone="Europe/Berlin"))


@pytest.mark.parametrize("column,values,match", [
    ("X [m]", [10., np.nan, 12.], "finite"),
    ("Z [m]", [0., np.inf, -2.], "finite"),
    ("position [m]", [0., np.nan, 1.], "finite"),
    ("position [m]", [0., 0., 1.], "increasing and unique"),
    ("position [m]", [0., 1., 0.5], "increasing and unique"),
])
def test_invalid_geometry(example, csv_frame, tmp_path, column, values, match):
    csv_frame[column] = values
    write_frame(tmp_path, csv_frame)
    with pytest.raises(ValueError, match=match):
        list(frames(example, tmp_path))


@pytest.mark.parametrize("change", ["xyz", "position", "count", "unit", "missing", "extra"])
def test_reject_inconsistent_frames(example, csv_frame, tmp_path, change):
    write_frame(tmp_path, csv_frame)
    changed = csv_frame.copy()
    if change == "xyz":
        changed.loc[1, "X [m]"] += 0.1
    elif change == "position":
        changed.loc[1, "position [m]"] += 0.1
    elif change == "count":
        changed = changed.iloc[:2]
    elif change == "unit":
        changed = changed.rename(columns={"temperature [\u00b0C]": "temperature [K]"})
    elif change == "missing":
        changed = changed.drop(columns=["brillouin_strain [Ghz]"])
    else:
        changed["unexpected"] = 1.
    write_frame(tmp_path, changed, "2025-12-12_11-45-39")
    iterator = frames(example, tmp_path)
    assert next(iterator).n_points == 3
    with pytest.raises(ValueError, match="differs|fields/units"):
        next(iterator)


def test_csv_reading_is_lazy_and_bounded(example, csv_frame, tmp_path, monkeypatch):
    first = write_frame(tmp_path, csv_frame)
    second = write_frame(tmp_path, csv_frame, "2025-12-12_11-45-39")
    read_csv = pd.read_csv
    reads = []

    def record_read(path):
        reads.append(path)
        return read_csv(path)

    monkeypatch.setattr(example.pd, "read_csv", record_read)
    iterator = frames(example, tmp_path)
    assert reads == []
    next(iterator)
    assert reads == [first]
    next(iterator)
    assert reads == [first, second]


def test_reject_empty_directory_and_bad_filename(example, csv_frame, tmp_path):
    with pytest.raises(ValueError, match="No CSV"):
        list(frames(example, tmp_path))
    csv_frame.to_csv(tmp_path / "unknown.csv", index=False)
    with pytest.raises(ValueError, match="Invalid timestamp filename"):
        list(frames(example, tmp_path))


def test_reject_duplicate_normalized_timestamps(example, csv_frame, tmp_path):
    write_frame(tmp_path, csv_frame, "2025-01-02_10-45-39")
    write_frame(tmp_path, csv_frame, "2025-1-2_10-45-39")
    with pytest.raises(ValueError, match="Duplicate normalized timestamps"):
        list(frames(example, tmp_path))


def test_exported_snapshots_preserve_numeric_values_and_missing_masks(example, csv_frame, tmp_path):
    from subsurface.core.structs.base_structures import UnstructuredData
    from subsurface.modules.writer.time_series import export_time_series

    write_frame(tmp_path, csv_frame)
    later = csv_frame.copy()
    later["temperature [\u00b0C]"] += 0.5
    write_frame(tmp_path, later, "2025-12-12_11-45-39")
    output = tmp_path / "export"
    index_path = export_time_series(
        example.iter_borehole_frames(tmp_path, source_timezone="UTC",
                                     time_series_id="synthetic", single_trajectory=True),
        output, time_series_id="synthetic", kind="trajectory",
    )
    index = json.loads(index_path.read_text())
    assert index["kind"] == "trajectory"
    assert len(index["frames"]) == 2
    assert len(list(output.glob("*.le"))) == 2
    for entry, source_table in zip(index["frames"], (csv_frame, later)):
        restored = UnstructuredData.from_binary_le(output / entry["path"])
        assert restored.data.attrs["timestamp"] == entry["timestamp"]
        assert restored.data.attrs["time_series_id"] == "synthetic"
        assert restored.data.attrs["attribute_units"]["temperature"] == "degC"
        np.testing.assert_allclose(restored.vertex, source_table[example.XYZ_COLUMNS])
        np.testing.assert_array_equal(restored.cells, [[0, 1], [1, 2]])
        for source, (canonical, _) in example.FIELDS.items():
            actual = restored.points_attributes[canonical].to_numpy()
            expected = source_table[source].to_numpy()
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, equal_nan=True)


def test_non_numeric_measurements_are_not_silently_coerced(example, csv_frame, tmp_path):
    csv_frame["raw_strain [\u00b5m/m]"] = ["bad", "1", "2"]
    write_frame(tmp_path, csv_frame)
    with pytest.raises(ValueError):
        list(frames(example, tmp_path))


def test_cli_uses_exporter_api(example, csv_frame, tmp_path, monkeypatch):
    write_frame(tmp_path, csv_frame)
    exporter = ModuleType("subsurface.modules.writer.time_series")
    calls = []

    def export_time_series(frame_iterator, output_directory, *, time_series_id, kind):
        calls.append((list(frame_iterator), output_directory, time_series_id, kind))
        return output_directory / "index.json"

    exporter.export_time_series = export_time_series
    monkeypatch.setitem(sys.modules, exporter.__name__, exporter)
    output = tmp_path / "output"
    example.main([str(tmp_path), str(output), "--source-timezone", "UTC", "--single-trajectory"])
    result, directory, series_id, kind = calls[0]
    assert len(result) == 1
    timestamp, frame = result[0]
    assert timestamp == frame.data.attrs["timestamp"] == "2025-12-12T10:45:39Z"
    assert directory == output
    assert series_id == "2terranigma-strain-temperature"
    assert kind == "trajectory"


@pytest.mark.parametrize("flags", [[], ["--source-timezone", "UTC"], ["--single-trajectory"]])
def test_cli_requires_timezone_and_affirmation(example, tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        example.main([str(tmp_path), str(tmp_path / "output"), *flags])
    assert error.value.code == 2
