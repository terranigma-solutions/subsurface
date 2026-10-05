import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

from subsurface.api._le_file_ops import load_le_mesh, write_le_mesh
from subsurface.api.le_inspection import inspect_le
from subsurface.api.le_merge import merge_le, merge_meshes
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh
from subsurface.core.structs.base_structures.unstructured_data import UnstructuredData


def mesh(ids=(2 ** 60 + 7, 2 ** 60 + 8), width=3, metadata=None):
    vertex = np.arange(18, dtype=np.float32).reshape(6, 3)
    cells = np.arange(2 * width, dtype=np.int32).reshape(2, width)
    attrs = pd.DataFrame({"object_id": np.asarray(ids, dtype=np.int64),
                          "value": np.array([0.25, 0.75], dtype=np.float32),
                          "valid": np.array([True, False])})
    point_attrs = pd.DataFrame({"rank": np.arange(6, dtype=np.int16)})
    return LiquidEarthMesh(vertex, cells, attrs, point_attrs,
                           metadata if metadata is not None else {"crs": "local", "units": "m"})


def save(tmp_path, meshes):
    paths = []
    for index, item in enumerate(meshes):
        path = tmp_path / f"source{index}.le"
        write_le_mesh(item, path, sources=[])
        paths.append(path)
    return paths


def test_offsets_unused_vertices_public_read_and_unknown_groups(tmp_path):
    inputs = [mesh(width=2), mesh(width=2), mesh(width=2)]
    paths = save(tmp_path, inputs)
    result = merge_le(iter(paths), tmp_path / "merged.le")
    restored = load_le_mesh(result.destination)
    expected_cells = np.concatenate([item.cells + 6 * i for i, item in enumerate(inputs)])
    np.testing.assert_array_equal(restored.cells, expected_cells)
    np.testing.assert_array_equal(restored.vertex, np.concatenate([item.vertex for item in inputs]))
    for name in inputs[0].attributes:
        np.testing.assert_array_equal(restored.attributes[name],
                                      np.concatenate([item.attributes[name] for item in inputs]))
        assert restored.attributes[name].dtype == inputs[0].attributes[name].dtype
    assert restored.points_attributes["rank"].dtype == np.dtype("int16")
    public = UnstructuredData.from_binary_le(result.destination)
    np.testing.assert_array_equal(public.vertex, restored.vertex)
    np.testing.assert_array_equal(public.cells, expected_cells)
    assert public.data.attrs == restored.data_attrs
    assert result.id_mapping is None
    assert inspect_le(result.destination).logical_object_count is None
    assert restored.data_attrs["units"] == "m"
    assert [entry["name"] for entry in restored.data_attrs["le_tools"]["sources"]] == [p.name for p in paths]


@pytest.mark.parametrize("policy,count,expected", [("source", 4, [0, 1, 2, 3]),
                                                   ("shared", 2, [2 ** 60 + 7, 2 ** 60 + 8] * 2)])
def test_identity_collision_and_exact_provenance(tmp_path, policy, count, expected):
    paths = save(tmp_path, [mesh(), mesh()])
    result = merge_le(paths, tmp_path / "merged.le", object_attribute="object_id",
                      association="cell", id_policy=policy)
    restored = load_le_mesh(result.destination)
    np.testing.assert_array_equal(restored.attributes["object_id"], expected)
    assert inspect_le(result.destination, object_attribute="object_id",
                      association="cell").logical_object_count == count
    mapping = json.loads(json.dumps(result.id_mapping))
    assert [item["original_id"] for item in mapping] == [2 ** 60 + 7, 2 ** 60 + 8] * 2
    assert all(type(item["original_id"]) is int for item in mapping)
    assert [item["source_index"] for item in mapping] == [0, 0, 1, 1]
    assert restored.data_attrs["le_tools"]["id_mapping"] == mapping
    public = UnstructuredData.from_binary_le(result.destination)
    assert public.data.attrs["le_tools"]["id_mapping"] == mapping


def test_ordered_sorted_ids_and_input_immutability():
    first, second = mesh(ids=(9, -3)), mesh(ids=(9, 9))
    original = copy.deepcopy(first)
    merged = merge_meshes([first, second], object_attribute="object_id", association="cell")
    assert merged.attributes["object_id"].tolist() == [1, 0, 2, 2]
    assert [entry["original_id"] for entry in merged.data_attrs["le_tools"]["id_mapping"]] == [-3, 9, 9]
    pd.testing.assert_frame_equal(first.attributes, original.attributes)
    assert first.data_attrs == original.data_attrs
    np.testing.assert_array_equal(first.cells, original.cells)


@pytest.mark.parametrize("width", [0, 1])
def test_point_grouping(tmp_path, width):
    first = mesh(width=width)
    first.points_attributes["id"] = np.array([4, 4, 8, 8, 4, 4], dtype=np.int64)
    paths = save(tmp_path, [first, first])
    result = merge_le(paths, tmp_path / "points.le", object_attribute="id", association="point")
    assert inspect_le(result.destination, object_attribute="id", association="point").object_ids == (0, 1, 2, 3)
    np.testing.assert_array_equal(load_le_mesh(result.destination).cells,
                                  np.concatenate([first.cells, first.cells + 6]))


@pytest.mark.parametrize("kwargs", [{"object_attribute": "object_id"}, {"association": "cell"},
                                     {"id_policy": "invalid"}, {"id_policy": "shared"},
                                     {"object_attribute": "missing", "association": "cell"},
                                     {"object_attribute": "object_id", "association": "point"}])
def test_grouping_contract(kwargs):
    with pytest.raises(ValueError):
        merge_meshes([mesh()], **kwargs)


@pytest.mark.parametrize("ids", [[True, False], [np.nan, 1.5], [np.inf, 1.5]])
def test_invalid_grouping_values(ids):
    item = mesh()
    item.attributes["object_id"] = ids
    with pytest.raises(ValueError, match="Grouping"):
        merge_meshes([item], object_attribute="object_id", association="cell")


@pytest.mark.parametrize("change", ["missing", "dtype", "order", "point_schema", "width",
                                     "crs", "units", "semantic", "missing_metadata"])
def test_strict_compatibility_and_no_output(tmp_path, change):
    first, second = mesh(), mesh()
    if change == "missing":
        second.attributes = second.attributes.drop(columns="value")
    elif change == "dtype":
        second.attributes["object_id"] = second.attributes["object_id"].astype(np.uint64)
    elif change == "order":
        second.attributes = second.attributes[list(reversed(second.attributes.columns))]
    elif change == "point_schema":
        second.points_attributes = second.points_attributes.rename(columns={"rank": "other"})
    elif change == "width":
        second = mesh(width=2)
    elif change == "missing_metadata":
        second.data_attrs.pop("units")
    else:
        second.data_attrs[change] = "different"
    paths = save(tmp_path, [first, second])
    output = tmp_path / "merged.le"
    with pytest.raises(ValueError):
        merge_le(paths, output)
    assert not output.exists()


def test_different_sibling_provenance_is_nested():
    first, second = mesh(), mesh()
    first.data_attrs["le_tools"] = {"operation": "split", "original_id": 12}
    second.data_attrs["le_tools"] = {"operation": "split", "original_id": 19}
    merged = merge_meshes([first, second])
    sources = merged.data_attrs["le_tools"]["sources"]
    assert sources[0]["le_tools"] == first.data_attrs["le_tools"]
    assert sources[1]["le_tools"] == second.data_attrs["le_tools"]


def empty_mesh():
    return LiquidEarthMesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32),
                           pd.DataFrame(), pd.DataFrame(), {"crs": "local", "units": "m"})


def test_empty_sources_and_datasets(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        merge_le([], tmp_path / "output.le")
    with pytest.raises(ValueError, match="At least one"):
        merge_meshes([])
    paths = save(tmp_path, [empty_mesh(), empty_mesh()])
    result = merge_le(paths, tmp_path / "empty.le")
    assert inspect_le(result.destination).vertex_count == 0
    assert load_le_mesh(result.destination).cells.shape == (0, 3)
    item = empty_mesh()
    item.attributes["id"] = pd.Series([], dtype=np.int64)
    with pytest.raises(ValueError, match="Empty cell attribute"):
        merge_meshes([item], object_attribute="id", association="cell")
    nonempty = mesh()
    nonempty.attributes = pd.DataFrame(index=range(2))
    nonempty.points_attributes = pd.DataFrame(index=range(6))
    assert merge_meshes([empty_mesh(), nonempty]).vertex.shape == (6, 3)


def test_int32_capacity_precheck_without_allocating():
    item = mesh()
    item.vertex = np.broadcast_to(np.zeros((1, 3), dtype=np.float32), (2 ** 31, 3))
    with pytest.raises(ValueError, match="int32 connectivity capacity"):
        merge_meshes([item, mesh()])


def test_overwrite_aliases_and_failed_validation_preserve_files(tmp_path):
    paths = save(tmp_path, [mesh(), mesh()])
    before = [path.read_bytes() for path in paths]
    output = tmp_path / "output.le"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        merge_le(paths, output)
    assert output.read_bytes() == b"existing"
    merge_le(paths, output, overwrite=True)
    assert inspect_le(output).vertex_count == 12
    for alias in [paths[0], tmp_path / "symlink.le", tmp_path / "hardlink.le"]:
        if alias.name == "symlink.le":
            alias.symlink_to(paths[0])
        elif alias.name == "hardlink.le":
            os.link(paths[0], alias)
        with pytest.raises(ValueError, match="alias"):
            merge_le(paths, alias, overwrite=True)
    with pytest.raises(ValueError):
        merge_le(paths, output, object_attribute="missing", association="cell", overwrite=True)
    assert [path.read_bytes() for path in paths] == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_float_ids_and_repeated_source_paths(tmp_path):
    item = mesh()
    item.attributes["object_id"] = np.array([1.5, -0.25], dtype=np.float32)
    path = save(tmp_path, [item])[0]
    result = merge_le([path, path], tmp_path / "repeated.le", object_attribute="object_id",
                      association="cell")
    assert [entry["original_id"] for entry in result.id_mapping] == [-0.25, 1.5, -0.25, 1.5]
    assert load_le_mesh(result.destination).attributes["object_id"].tolist() == [1, 0, 3, 2]


def test_single_path_is_not_a_source_sequence(tmp_path):
    with pytest.raises(TypeError, match="iterable"):
        merge_le(tmp_path / "input.le", tmp_path / "output.le")


@pytest.mark.parametrize("width", [2, 3, 4, 8])
def test_supported_geometry_and_public_integer_attributes(tmp_path, width):
    vertices = np.arange(3 * (2 * width + 1), dtype=np.float32).reshape(-1, 3)
    cells = np.arange(2 * width, dtype=np.int32).reshape(2, width)
    ids = np.array([2 ** 60 + 7, 2 ** 60 + 8], dtype=np.int64)
    item = LiquidEarthMesh(vertices, cells, pd.DataFrame({"id": ids}),
                           pd.DataFrame(index=range(len(vertices))))
    paths = save(tmp_path, [item, item])
    result = merge_le(paths, tmp_path / "output.le", object_attribute="id",
                      association="cell", id_policy="shared")
    public = UnstructuredData.from_binary_le(result.destination)
    np.testing.assert_array_equal(public.cells, np.concatenate([cells, cells + len(vertices)]))
    np.testing.assert_array_equal(public.data["cell_attrs"].values[:, 0], np.tile(ids, 2))
    assert public.data["cell_attrs"].dtype == np.dtype("int64")


def test_remapping_uses_int64_not_original_small_integer_width():
    item = LiquidEarthMesh(np.zeros((128, 3), dtype=np.float32),
                           np.arange(128, dtype=np.int32).reshape(-1, 1),
                           pd.DataFrame(index=range(128)),
                           pd.DataFrame({"id": np.arange(128, dtype=np.int8)}))
    merged = merge_meshes([item, item], object_attribute="id", association="point")
    assert merged.points_attributes["id"].dtype == np.dtype("int64")
    np.testing.assert_array_equal(merged.points_attributes["id"], np.arange(256))
