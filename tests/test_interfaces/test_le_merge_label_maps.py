"""Merging DXF-style label maps (``cell_attr_map``/``point_attr_map``)."""

import io

import numpy as np
import pandas as pd
import pytest

from subsurface.api._le_file_ops import load_le_mesh, write_le_mesh
from subsurface.api.le_merge import merge_le, merge_meshes
from subsurface.api.le_split import split_le
from subsurface.api.le_validation import validate_le
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


def layered(layers, metadata=None):
    """A triangle mesh with one triangle per (name, id) layer, DXF importer style."""
    count = len(layers)
    vertex = np.arange(9 * count, dtype=np.float32).reshape(-1, 3)
    cells = np.arange(3 * count, dtype=np.int32).reshape(-1, 3)
    attrs = pd.DataFrame({"Id": np.array([object_id for _, object_id in layers], dtype=np.int64)})
    data_attrs = {"cell_attr_map": dict(layers)}
    data_attrs.update(metadata or {})
    return LiquidEarthMesh(vertex, cells, attrs, pd.DataFrame(index=range(len(vertex))), data_attrs)


def save(tmp_path, meshes, names):
    paths = []
    for item, name in zip(meshes, names):
        paths.append(write_le_mesh(item, tmp_path / f"{name}.le", sources=[]))
    return paths


GROUP = dict(object_attribute="Id", association="cell")


def test_different_single_layer_maps_merge_and_split_by_layer(tmp_path):
    paths = save(tmp_path, [layered([("Igneous", 1)]), layered([("Rookery Fault", 1)])],
                 ["GM - Nymagee Igneous", "GM - Rookery Fault"])
    result = merge_le(paths, tmp_path / "merged.le", **GROUP)
    merged = load_le_mesh(result.destination)
    assert merged.data_attrs["cell_attr_map"] == {"Igneous": 0, "Rookery Fault": 1}
    assert merged.data_attrs["le_tools"]["renamed_labels"] == []
    assert [set(entry) for entry in result.id_mapping] == [{"source_index", "original_id", "merged_id"}] * 2
    validate_le(result.destination, family="mesh")


def test_shared_layer_name_gets_unique_source_qualified_label(tmp_path):
    paths = save(tmp_path, [layered([("Faults", 1), ("Granite", 2)]), layered([("Faults", 1)])],
                 ["GM - Nymagee Igneous", "GM - Rookery Fault"])
    merged = load_le_mesh(merge_le(paths, tmp_path / "merged.le", **GROUP).destination)
    assert merged.data_attrs["cell_attr_map"] == {
            "Faults": 0, "Granite": 1, "Faults (GM - Rookery Fault)": 2}
    assert merged.data_attrs["le_tools"]["renamed_labels"] == [
            {"source_index": 1, "label": "Faults", "merged_label": "Faults (GM - Rookery Fault)",
             "merged_id": 2}]


def test_repeated_label_without_source_names_and_repeated_names():
    item = layered([("Faults", 1)])
    merged = merge_meshes([item, item, item], **GROUP)
    assert merged.data_attrs["cell_attr_map"] == {
            "Faults": 0, "Faults (source 1)": 1, "Faults (source 2)": 2}
    merged = merge_meshes([item, item, item], source_labels=["x/a.le", "y/a.le", "z/a.le"], **GROUP)
    assert merged.data_attrs["cell_attr_map"] == {"Faults": 0, "Faults (a)": 1, "Faults (a 2)": 2}


@pytest.mark.parametrize("policy,expected", [("source", {"B": 0, "C": 1}), ("shared", {"B": 2, "C": 3})])
def test_split_parts_of_one_file_keep_only_present_ids(tmp_path, policy, expected):
    source = layered([("A", 1), ("B", 2), ("C", 3)])
    path = save(tmp_path, [source], ["original"])[0]
    directory = tmp_path / "parts"
    directory.mkdir()
    parts = split_le(path, directory, **GROUP)
    assert load_le_mesh(parts[2]).data_attrs["cell_attr_map"] == {"A": 1, "B": 2, "C": 3}
    merged = merge_meshes([load_le_mesh(parts[2]), load_le_mesh(parts[3])], id_policy=policy, **GROUP)
    assert merged.data_attrs["cell_attr_map"] == expected


def test_shared_policy_unions_compatible_maps():
    merged = merge_meshes([layered([("A", 1)]), layered([("B", 2), ("A", 1)])],
                          id_policy="shared", **GROUP)
    assert merged.data_attrs["cell_attr_map"] == {"A": 1, "B": 2}


@pytest.mark.parametrize("second,match", [([("B", 1)], 'ID 1 has different labels'),
                                          ([("A", 2)], 'Label "A" has different IDs')])
def test_shared_policy_rejects_conflicting_labels(second, match):
    with pytest.raises(ValueError, match=match):
        merge_meshes([layered([("A", 1)]), layered(second)], id_policy="shared", **GROUP)


@pytest.mark.parametrize("key", ["crs", "units"])
def test_real_metadata_conflicts_still_rejected_and_named(key):
    first = layered([("A", 1)], {"crs": "EPSG:28355", "units": "m"})
    second = layered([("B", 1)], {"crs": "EPSG:28355", "units": "m"})
    second.data_attrs[key] = "different"
    with pytest.raises(ValueError, match=f'Sources have different "{key}" metadata'):
        merge_meshes([first, second], **GROUP)


def test_missing_metadata_key_is_named():
    first, second = layered([("A", 1)], {"crs": "local"}), layered([("B", 1)])
    with pytest.raises(ValueError, match='Sources have different "crs" metadata'):
        merge_meshes([first, second], **GROUP)


def test_non_label_map_values_stay_strict():
    first, second = layered([("A", 1)]), layered([("A", 1)])
    second.data_attrs["cell_attr_map"] = {"A": "one"}
    with pytest.raises(ValueError, match='"cell_attr_map" metadata'):
        merge_meshes([first, second], **GROUP)


def test_ungrouped_association_map_stays_strict():
    first, second = layered([("A", 1)]), layered([("B", 1)])
    first.data_attrs["point_attr_map"] = {"p": 1}
    second.data_attrs["point_attr_map"] = {"q": 1}
    with pytest.raises(ValueError, match='"point_attr_map" metadata'):
        merge_meshes([first, second], **GROUP)


def test_merge_without_grouping_rejects_different_maps_and_keeps_equal_ones(tmp_path):
    with pytest.raises(ValueError, match="keep object IDs"):
        merge_meshes([layered([("A", 1)]), layered([("B", 1)])])
    with pytest.raises(ValueError, match="keep object IDs"):
        merge_meshes([layered([("A", 1)]), LiquidEarthMesh(
                np.zeros((3, 3), dtype=np.float32), np.arange(3, dtype=np.int32).reshape(1, 3),
                pd.DataFrame({"Id": np.array([1], dtype=np.int64)}), pd.DataFrame(index=range(3)))])
    merged = merge_meshes([layered([("A", 1)]), layered([("A", 1)])])
    assert merged.data_attrs["cell_attr_map"] == {"A": 1}
    assert "renamed_labels" not in merged.data_attrs["le_tools"]


def test_source_without_map_contributes_no_labels():
    plain = layered([("unused", 5)])
    del plain.data_attrs["cell_attr_map"]
    merged = merge_meshes([layered([("A", 1)]), plain], **GROUP)
    assert merged.data_attrs["cell_attr_map"] == {"A": 0}


def test_dxf_files_round_trip_through_merge_and_split(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    from subsurface.api.interfaces.stream import DXF_stream_to_unstruc

    def dxf_le(name, layers):
        document = ezdxf.new()
        for offset, layer in enumerate(layers):
            z = float(offset)
            document.modelspace().add_3dface([(0, 0, z), (1, 0, z), (0, 1, z)],
                                             dxfattribs={"layer": layer})
        stream = io.StringIO()
        document.write(stream)
        stream.seek(0)
        path = tmp_path / f"{name}.le"
        path.write_bytes(DXF_stream_to_unstruc(stream).to_binary())
        return path

    paths = [dxf_le("GM - Nymagee Igneous", ["Igneous", "Faults"]),
             dxf_le("GM - Rookery Fault", ["Faults", "Rookery Fault"])]
    result = merge_le(paths, tmp_path / "merged.le", **GROUP)
    validate_le(result.destination, family="mesh")
    directory = tmp_path / "split"
    directory.mkdir()
    parts = split_le(result.destination, directory, **GROUP)
    label_map = load_le_mesh(result.destination).data_attrs["cell_attr_map"]
    names = {object_id: name for name, object_id in label_map.items()}
    assert [names[object_id] for object_id in parts] == [
            "Faults", "Igneous", "Faults (GM - Rookery Fault)", "Rookery Fault"]
    for object_id, path in parts.items():
        part = load_le_mesh(path)
        assert part.attributes["Id"].tolist() == [object_id]
        assert part.data_attrs["cell_attr_map"] == label_map
