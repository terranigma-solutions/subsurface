"""Public file-tool composition, independent of visualization and external data."""

import numpy as np
import pandas as pd
import pytest

import subsurface
from subsurface import StructuredData, UnstructuredData, inspect_le, merge_le, split_le, transform_le
from subsurface.api._le_file_ops import load_le_mesh, write_le_mesh
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


@pytest.mark.parametrize("name", ["inspect_le", "transform_le", "split_le", "merge_le"])
def test_public_exports(name):
    assert getattr(subsurface, name) is getattr(subsurface.api, name)


def test_triangle_read_inspect_transform_split_merge(tmp_path):
    vertex = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                       [2, 1, 0], [99, 99, 99]], dtype=np.float32)
    cells = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.int32)
    ids = np.array([2 ** 60 + 1, 2 ** 60 + 9, 2 ** 60 + 1], dtype=np.int64)
    attrs = pd.DataFrame({"object_id": ids, "grade": np.array([0.25, 0.5, 0.75], dtype=np.float32),
                          "valid": [True, False, True]})
    points = pd.DataFrame({"u": np.linspace(0.125, 0.875, len(vertex), dtype=np.float32),
                           "original_vertex": np.arange(len(vertex), dtype=np.int16)})
    mesh = LiquidEarthMesh(vertex, cells, attrs, points, {"crs": "local", "units": "m"})
    source = write_le_mesh(mesh, tmp_path / "source.le", sources=[])
    original = source.read_bytes()
    public = UnstructuredData.from_binary_le(source)
    np.testing.assert_array_equal(public.vertex, vertex)
    np.testing.assert_array_equal(public.cells, cells)
    assert public.data.attrs == mesh.data_attrs
    assert inspect_le(source).logical_object_count is None
    summary = inspect_le(source, object_attribute="object_id", association="cell")
    assert (summary.dataset_count, summary.vertex_count, summary.cell_count,
            summary.logical_object_count) == (1, 6, 3, 2)
    assert summary.object_ids == tuple(np.unique(ids).tolist())
    assert not summary.payload_validated

    matrix = np.array([[-2, 0.5, 0, 10], [0, 3, 0, -5], [0, 0, 1, 7], [0, 0, 0, 1]])
    transformed = transform_le(source, tmp_path / "transformed.le", matrix)
    decoded = load_le_mesh(transformed)
    expected = vertex @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_allclose(decoded.vertex, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(decoded.cells, cells[:, [0, 2, 1]])
    pd.testing.assert_frame_equal(decoded.attributes, attrs)
    pd.testing.assert_frame_equal(decoded.points_attributes, points)

    directory = tmp_path / "split"
    directory.mkdir()
    outputs = split_le(transformed, directory, object_attribute="object_id", association="cell")
    assert tuple(outputs) == summary.object_ids
    assert [path.name for path in outputs.values()] == ["object_000000.le", "object_000001.le"]
    assert sum(inspect_le(path).vertex_count for path in outputs.values()) == 8
    for object_id, path in outputs.items():
        grouped = inspect_le(path, object_attribute="object_id", association="cell")
        assert grouped.logical_object_count == 1
        assert grouped.object_ids == (object_id,)
        subset = load_le_mesh(path)
        assert subset.cells.min() >= 0 and subset.cells.max() < len(subset.vertex)
        assert 5 not in subset.points_attributes["original_vertex"].to_numpy()

    result = merge_le(outputs.values(), tmp_path / "merged.le",
                      object_attribute="object_id", association="cell")
    merged = load_le_mesh(result.destination)
    assert [entry["original_id"] for entry in result.id_mapping] == list(outputs)
    assert [entry["merged_id"] for entry in result.id_mapping] == [0, 1]
    assert inspect_le(result.destination, object_attribute="object_id", association="cell").object_ids == (0, 1)
    order = np.concatenate([np.flatnonzero(ids == object_id) for object_id in outputs])
    np.testing.assert_allclose(merged.vertex[merged.cells], decoded.vertex[decoded.cells[order]])
    np.testing.assert_array_equal(merged.attributes["grade"], attrs["grade"].iloc[order])
    np.testing.assert_array_equal(merged.attributes["valid"], attrs["valid"].iloc[order])
    assert merged.attributes["object_id"].dtype == np.dtype("int64")
    assert merged.attributes["valid"].dtype == np.dtype("bool")
    assert merged.data_attrs["crs"] == "local" and merged.data_attrs["units"] == "m"
    assert [entry["le_tools"]["object_id"] for entry in merged.data_attrs["le_tools"]["sources"]] == list(outputs)
    restored = UnstructuredData.from_binary_le(result.destination)
    np.testing.assert_array_equal(restored.cells, merged.cells)
    np.testing.assert_array_equal(restored.vertex, merged.vertex)
    assert restored.data.attrs == merged.data_attrs

    shared = merge_le(outputs.values(), tmp_path / "shared.le", object_attribute="object_id",
                      association="cell", id_policy="shared")
    assert inspect_le(shared.destination, object_attribute="object_id", association="cell").object_ids == summary.object_ids
    again = tmp_path / "split_again"
    again.mkdir()
    assert tuple(split_le(result.destination, again, object_attribute="object_id", association="cell")) == (0, 1)
    assert source.read_bytes() == original


@pytest.mark.parametrize("width", [0, 1])
def test_point_pipeline_and_collision_identity(tmp_path, width):
    vertex = np.arange(12, dtype=np.float32).reshape(4, 3)
    cells = np.empty((4, 0), dtype=np.int32) if width == 0 else np.arange(4, dtype=np.int32)[:, None]
    points = pd.DataFrame({"id": np.array([3, 9, 3, 9], dtype=np.int64),
                           "value": np.array([0.25, 0.5, 0.75, 0.125], dtype=np.float32)})
    mesh = LiquidEarthMesh(vertex, cells, pd.DataFrame(index=range(4)), points, {"units": "m"})
    source = write_le_mesh(mesh, tmp_path / "points.le", sources=[])
    matrix = np.eye(4)
    matrix[:3, 3] = [10, 20, 30]
    moved = transform_le(source, tmp_path / "moved.le", matrix)
    directory = tmp_path / "objects"
    directory.mkdir()
    outputs = split_le(moved, directory, object_attribute="id", association="point")
    merged = merge_le(outputs.values(), tmp_path / "merged.le", object_attribute="id", association="point")
    restored = UnstructuredData.from_binary_le(merged.destination)
    np.testing.assert_array_equal(restored.vertex, (vertex + [10, 20, 30])[[0, 2, 1, 3]])
    if width == 1:
        np.testing.assert_array_equal(restored.cells[:, 0], np.arange(4))
    assert inspect_le(merged.destination, object_attribute="id", association="point").logical_object_count == 2
    collisions = merge_le([merged.destination, merged.destination], tmp_path / "collisions.le",
                          object_attribute="id", association="point")
    assert inspect_le(collisions.destination, object_attribute="id", association="point").object_ids == (0, 1, 2, 3)
    shared = merge_le([merged.destination, merged.destination], tmp_path / "shared.le",
                      object_attribute="id", association="point", id_policy="shared")
    assert inspect_le(shared.destination, object_attribute="id", association="point").logical_object_count == 2


def test_public_structured_reader_and_inspection(tmp_path):
    values = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    grid = StructuredData.from_numpy(values, coords={"x": [10, 12], "y": [-1, 0, 1], "z": [7]},
                                     data_array_name="density")
    source = tmp_path / "grid.le"
    source.write_bytes(grid.to_binary())
    restored = StructuredData.from_binary_le(source)
    summary = inspect_le(source)
    assert summary.file_kind == "structured"
    assert summary.dataset_count == 1 and summary.grid_sample_count == 6
    assert summary.logical_object_count is None and not summary.payload_validated
    np.testing.assert_array_equal(restored.active_data_array.values, values)
    assert restored.active_data_array_name == "density"
    copy = tmp_path / "grid_copy.le"
    copy.write_bytes(restored.to_binary())
    reread = StructuredData.from_binary_le(copy)
    np.testing.assert_array_equal(reread.active_data_array.values, values)
    assert reread.bounds == grid.bounds
    assert inspect_le(copy).shapes == summary.shapes
    with pytest.raises(ValueError, match="Structured"):
        inspect_le(source, object_attribute="density", association="point")
    with pytest.raises(ValueError):
        transform_le(source, tmp_path / "unsupported.le", np.eye(4))
    assert not (tmp_path / "unsupported.le").exists()


def test_strict_merge_rejects_split_wire_dtype_divergence(tmp_path):
    mesh = LiquidEarthMesh(
        np.arange(9, dtype=np.float32).reshape(3, 3),
        np.array([[0, 1], [1, 2]], dtype=np.int32),
        pd.DataFrame({"id": np.array([1.0, 1.5], dtype=np.float32)}),
        pd.DataFrame(index=range(3)), {},
    )
    source = write_le_mesh(mesh, tmp_path / "source.le", sources=[])
    original = source.read_bytes()
    directory = tmp_path / "split"
    directory.mkdir()
    outputs = split_le(source, directory, object_attribute="id", association="cell")
    assert tuple(outputs) == (1.0, 1.5)
    assert load_le_mesh(outputs[1.0]).attributes["id"].dtype == np.dtype("int64")
    assert load_le_mesh(outputs[1.5]).attributes["id"].dtype == np.dtype("float32")
    destination = tmp_path / "incompatible.le"
    with pytest.raises(ValueError, match="dtypes"):
        merge_le(outputs.values(), destination, object_attribute="id", association="cell")
    assert not destination.exists()
    assert source.read_bytes() == original
