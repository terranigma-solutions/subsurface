"""Independent composition of the public non-resampling structured file tools."""

import numpy as np
import pytest

import subsurface
from subsurface import (
    StructuredData, inspect_le, merge_structured_le, split_structured_le,
    transform_structured_le,
)
from subsurface.api._le_grid_ops import write_le_grid


@pytest.mark.parametrize("name", ["transform_structured_le", "split_structured_le", "merge_structured_le"])
def test_public_structured_exports(name):
    assert getattr(subsurface, name) is getattr(subsurface.api, name)


@pytest.mark.parametrize("rank,axis_index", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("dtype", ["int64", "uint64", "float32", "float64"])
def test_read_inspect_transform_split_merge(tmp_path, rank, axis_index, dtype):
    shape = (4, 3, 2)[:rank]
    dims = ("dim0",) if rank == 1 else ("x", "y", "z")[:rank]
    count = int(np.prod(shape))
    if dtype == "int64":
        values = np.arange(count, dtype=dtype) + np.int64(2 ** 60 + 7)
    elif dtype == "uint64":
        values = np.arange(count, dtype=dtype) + np.uint64(2 ** 63 + 7)
    else:
        values = np.arange(count, dtype=dtype) / 4
        values[:2] = [np.nan, np.inf]
    values = values.reshape(shape, order="F")
    origins = [1e6, -12, 8]
    steps = [0.5, 2, 4]
    coords = {dim: origins[i] + np.arange(size) * steps[i]
              for i, (dim, size) in enumerate(zip(dims, shape))}
    grid = StructuredData.from_numpy(values, coords=coords, data_array_name="lithology")
    grid.dtype = dtype
    source = write_le_grid(grid, tmp_path / "source.le", sources=[])
    original = source.read_bytes()
    restored = StructuredData.from_binary_le(source)
    np.testing.assert_array_equal(restored.values, values)
    summary = inspect_le(source)
    assert summary.file_kind == "structured"
    assert summary.grid_sample_count == count and summary.logical_object_count is None
    assert not summary.payload_validated

    matrix = np.eye(4)
    scales = [2, 0.5, 3]
    translations = [1000, -20, 7]
    for index in range(rank):
        matrix[index, index] = scales[index]
        matrix[index, 3] = translations[index]
    transformed = transform_structured_le(source, tmp_path / "transformed.le", matrix)
    moved = StructuredData.from_binary_le(transformed)
    np.testing.assert_array_equal(moved.values, values)
    assert moved.dtype == dtype and moved.active_data_array_name == "lithology"
    for index, dim in enumerate(dims):
        np.testing.assert_array_equal(moved.data.coords[dim], coords[dim] * scales[index] + translations[index])

    axis = dims[axis_index]
    size = shape[axis_index]
    boundary = size // 2
    directory = tmp_path / "tiles"
    directory.mkdir()
    windows = {"left": {axis: (0, boundary)}, "right": {axis: (boundary, size)}}
    outputs = split_structured_le(transformed, directory, windows=windows)
    assert tuple(outputs) == ("left", "right")
    assert [path.name for path in outputs.values()] == ["grid_000000.le", "grid_000001.le"]
    for label, (start, stop) in (("left", (0, boundary)), ("right", (boundary, size))):
        tile = StructuredData.from_binary_le(outputs[label])
        selection = [slice(None)] * rank
        selection[axis_index] = slice(start, stop)
        np.testing.assert_array_equal(tile.values, values[tuple(selection)])
        np.testing.assert_array_equal(tile.data.coords[axis], moved.data.coords[axis][start:stop])
        assert tile.dtype == dtype and tile.active_data_array_name == "lithology"
    assert sum(inspect_le(path).grid_sample_count for path in outputs.values()) == count

    # A two-sample original axis becomes two singleton tiles; spacing was not
    # serialized in either tile, so the caller explicitly supplies it.
    spacing = steps[axis_index] * scales[axis_index] if size == 2 else None
    merged = merge_structured_le(outputs.values(), tmp_path / "merged.le", axis=axis, spacing=spacing)
    final = StructuredData.from_binary_le(merged)
    assert final.dtype == dtype and final.active_data_array_name == "lithology"
    assert final.shape == shape and final.values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(final.values, values)
    for dim in dims:
        np.testing.assert_array_equal(final.data.coords[dim], moved.data.coords[dim])
    assert inspect_le(merged).grid_sample_count == count

    inverse = transform_structured_le(merged, tmp_path / "inverse.le", np.linalg.inv(matrix))
    undone = StructuredData.from_binary_le(inverse)
    np.testing.assert_array_equal(undone.values, values)
    for dim in dims:
        np.testing.assert_allclose(undone.data.coords[dim], coords[dim], rtol=0, atol=1e-10)
    assert source.read_bytes() == original


def test_singleton_tiles_require_spacing_and_preserve_existing_output(tmp_path):
    grid = StructuredData.from_numpy(np.array([2 ** 60 + 1, 2 ** 60 + 2], dtype=np.int64),
                                     coords={"dim0": [10, 12]}, data_array_name="id")
    grid.dtype = "int64"
    source = write_le_grid(grid, tmp_path / "source.le", sources=[])
    directory = tmp_path / "tiles"
    directory.mkdir()
    tiles = split_structured_le(source, directory,
                                windows={"a": {"dim0": (0, 1)}, "b": {"dim0": (1, 2)}})
    destination = tmp_path / "merged.le"
    destination.write_bytes(b"existing")
    with pytest.raises(ValueError, match="spacing"):
        merge_structured_le(tiles.values(), destination, axis="dim0", overwrite=True)
    assert destination.read_bytes() == b"existing"
    merged = merge_structured_le(tiles.values(), destination, axis="dim0", spacing=2, overwrite=True)
    np.testing.assert_array_equal(StructuredData.from_binary_le(merged).values, grid.values)
    with pytest.raises(ValueError, match="alias"):
        merge_structured_le(tiles.values(), tiles["a"], axis="dim0", spacing=2, overwrite=True)


def test_overlap_gap_and_reversed_tiles_do_not_publish(tmp_path):
    grid = StructuredData.from_numpy(np.arange(6, dtype=np.int16), coords={"dim0": np.arange(6)},
                                     data_array_name="category")
    grid.dtype = "int16"
    source = write_le_grid(grid, tmp_path / "source.le", sources=[])
    original = source.read_bytes()
    for name, windows in (
        ("overlap", {"a": {"dim0": (0, 4)}, "b": {"dim0": (3, 6)}}),
        ("gap", {"a": {"dim0": (0, 2)}, "b": {"dim0": (3, 6)}}),
        ("reversed", {"b": {"dim0": (3, 6)}, "a": {"dim0": (0, 3)}}),
    ):
        directory = tmp_path / name
        directory.mkdir()
        tiles = split_structured_le(source, directory, windows=windows)
        output = tmp_path / f"{name}.le"
        with pytest.raises(ValueError, match="adjacent"):
            merge_structured_le(tiles.values(), output, axis="dim0")
        assert not output.exists()
    assert source.read_bytes() == original
