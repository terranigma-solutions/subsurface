import os

import numpy as np
import pytest

from subsurface import StructuredData
from subsurface.api import _le_grid_ops as ops
from subsurface.api.le_structured_transform import transform_structured_grid, transform_structured_le
from subsurface.core.structs.base_structures.structured_data import StructuredDataType


def make_grid(shape=(4,), dtype="int64", coords=None):
    dims = StructuredData._default_dim_names(len(shape))
    if coords is None:
        coords = {dim: -2.0 + np.arange(size) * 0.25 for dim, size in zip(dims, shape)}
    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    grid = StructuredData.from_numpy(values, coords=coords, data_array_name="facies")
    grid.dtype = dtype
    return grid


def affine(rank, scale=1.0, translation=0.0):
    matrix = np.eye(4)
    matrix[np.arange(rank), np.arange(rank)] = scale
    matrix[:rank, 3] = translation
    return matrix


@pytest.mark.parametrize("shape", [(5,), (1,), (2, 3), (1, 3), (2, 1, 4), (1, 1, 1)])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16",
                                   "uint32", "uint64", "float16", "float32", "float64", ">i8", ">f8"])
@pytest.mark.parametrize("operation", ["identity", "translation", "scale"])
def test_grid_and_file_transforms(tmp_path, shape, dtype, operation):
    grid = make_grid(shape, dtype)
    original = grid.data.copy(deep=True)
    rank = len(shape)
    matrix = affine(rank, scale=np.arange(1, rank + 1) * 2 if operation == "scale" else 1,
                    translation=np.arange(1, rank + 1) * 3 if operation == "translation" else 0)
    matrix_before = matrix.copy()
    result = transform_structured_grid(grid, matrix)
    assert result.dtype == grid.dtype
    assert result.values.dtype == grid.values.dtype
    assert result.active_data_array_name == grid.active_data_array_name
    assert result.shape == shape
    np.testing.assert_array_equal(result.values, grid.values)
    assert not np.shares_memory(result.values, grid.values)
    for index, dim in enumerate(grid.active_data_array.dims):
        expected = grid.data[dim].values * matrix[index, index] + matrix[index, 3]
        np.testing.assert_array_equal(result.data[dim], expected)
        assert result.data[dim].dtype == np.dtype("float64")
        assert not np.shares_memory(result.data[dim].values, grid.data[dim].values)
    assert grid.data.identical(original)
    np.testing.assert_array_equal(matrix, matrix_before)

    source = tmp_path / "source.le"
    source.write_bytes(grid.to_binary())
    before = source.read_bytes()
    output = transform_structured_le(source, tmp_path / "output.le", matrix)
    assert output == (tmp_path / "output.le").absolute()
    restored = ops.load_le_grid(output)
    assert restored.dtype == grid.dtype
    assert restored.data.identical(result.data)
    assert source.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("dtype", ["int64", "uint64", "float16", "float32", "float64"])
def test_extreme_and_nonfinite_scalar_values_are_unchanged(tmp_path, dtype):
    grid = make_grid(dtype=dtype)
    if dtype.startswith("float"):
        grid.values[:] = [np.nan, np.inf, -np.inf, -0.0]
    else:
        info = np.iinfo(dtype)
        grid.values[:] = [info.min, info.max, info.max - 1, 0]
    result = transform_structured_grid(grid, affine(1, 2, 3))
    assert result.values.tobytes() == grid.values.tobytes()
    source = tmp_path / "source.le"
    source.write_bytes(grid.to_binary())
    output = transform_structured_le(source, tmp_path / "out.le", affine(1, 2, 3))
    restored = ops.load_le_grid(output)
    assert restored.values.tobytes() == grid.values.tobytes()
    assert restored.dtype == dtype


def test_copy_can_be_mutated_independently():
    grid = make_grid()
    result = transform_structured_grid(grid, np.eye(4))
    result.values[0] = 100
    result.data = result.data.assign_coords(dim0=[-10, -1.75, -1.5, -1.25])
    result.active_data_array_name = "other"
    assert grid.values[0] == 0
    assert grid.data.dim0.values[0] == -2
    assert grid.active_data_array_name == "facies"


@pytest.mark.parametrize("shape", [(7,), (2, 5), (2, 3, 4), (1, 1, 1)])
def test_composition_and_inverse(shape):
    grid = make_grid(shape)
    rank = len(shape)
    first = affine(rank, np.arange(1, rank + 1) * 1.5, np.arange(rank) + 0.125)
    second = affine(rank, np.arange(1, rank + 1) * 0.5, np.arange(rank) - 0.5)
    sequential = transform_structured_grid(transform_structured_grid(grid, first), second)
    composed = transform_structured_grid(grid, second @ first)
    restored = transform_structured_grid(composed, np.linalg.inv(second @ first))
    for dim in grid.active_data_array.dims:
        np.testing.assert_allclose(sequential.data[dim], composed.data[dim], rtol=0, atol=1e-12)
        np.testing.assert_allclose(restored.data[dim], grid.data[dim], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(restored.values, grid.values)


@pytest.mark.parametrize("origin,span", [(1e6, 0.3), (1e12, 0.3), (1e16, 10)])
@pytest.mark.parametrize("scale,translation", [(1, 0), (1, 16), (2, 0)])
def test_large_origin_rounded_samples_are_not_regularized(origin, span, scale, translation):
    axis = np.linspace(origin, origin + span, 4)
    assert np.unique(np.diff(axis)).size > 1
    grid = make_grid(coords={"dim0": axis})
    result = transform_structured_grid(grid, affine(1, scale, translation))
    np.testing.assert_array_equal(result.data.dim0, axis * scale + translation)


def test_transform_that_cannot_reconstruct_samples_is_rejected(tmp_path):
    axis = np.linspace(1e6, 1e6 + 0.3, 4)
    grid = make_grid(coords={"dim0": axis})
    source = tmp_path / "source.le"
    source.write_bytes(grid.to_binary())
    original = source.read_bytes()
    # Removing the large origin exposes the rounded increments, not a new regular grid.
    matrix = affine(1, translation=-1e6)
    with pytest.raises(ValueError, match="reconstruction tolerance"):
        transform_structured_grid(grid, matrix)
    with pytest.raises(ValueError, match="reconstruction tolerance"):
        transform_structured_le(source, tmp_path / "out.le", matrix)
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("matrix", [np.eye(3), np.ones((4, 5)), [], 1])
def test_invalid_matrix_shape(matrix):
    with pytest.raises(ValueError, match="shape"):
        transform_structured_grid(make_grid(), matrix)


@pytest.mark.parametrize("matrix", [np.eye(4).astype(complex), np.eye(4).astype(bool),
                                    np.eye(4).astype(str), np.eye(4).astype(object)])
def test_nonreal_matrix_types(matrix):
    with pytest.raises(TypeError, match="real numeric"):
        transform_structured_grid(make_grid(), matrix)


@pytest.mark.parametrize("entry,value", [((0, 0), np.nan), ((0, 3), np.inf), ((3, 3), -np.inf),
                                        ((3, 0), 1e-20), ((3, 3), 1 + 1e-12),
                                        ((0, 1), 1e-20), ((1, 0), 1),
                                        ((0, 0), -1), ((1, 1), 0), ((2, 2), -0.1)])
def test_invalid_matrix_entries(entry, value):
    matrix = np.eye(4)
    matrix[entry] = value
    with pytest.raises(ValueError):
        transform_structured_grid(make_grid((2, 3, 4)), matrix)


def test_rotation_is_rejected():
    matrix = np.eye(4)
    matrix[:2, :2] = [[0, -1], [1, 0]]
    with pytest.raises(ValueError, match="positive diagonal"):
        transform_structured_grid(make_grid((2, 3)), matrix)


@pytest.mark.parametrize("entry", [(3, 0), (0, 1), (2, 3)])
def test_exact_restrictions_checked_before_float64_rounding(entry):
    matrix = np.eye(4, dtype=np.longdouble)
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    matrix[entry] = tiny
    with pytest.raises(ValueError):
        transform_structured_grid(make_grid(), matrix)


def test_unused_scale_checked_before_float64_rounding():
    matrix = np.eye(4, dtype=np.longdouble)
    matrix[2, 2] = np.nextafter(np.longdouble(1), np.longdouble(2))
    with pytest.raises(ValueError, match="Unused"):
        transform_structured_grid(make_grid(), matrix)


@pytest.mark.parametrize("rank,axis", [(1, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("field", ["scale", "translation"])
def test_unused_axes_cannot_be_silently_ignored(rank, axis, field):
    matrix = np.eye(4)
    matrix[axis, axis if field == "scale" else 3] = 2
    with pytest.raises(ValueError, match="Unused"):
        transform_structured_grid(make_grid((2,) * rank), matrix)


@pytest.mark.parametrize("coords,scale,translation", [([1e308], 2, 0), ([1e308], 1, 1e308),
                                                     ([0, 1, 2, 3], 1, 1e16),
                                                     ([0, 1, 2, 3], np.nextafter(0.0, 1.0), 1),
                                                     ([-8e307, 8e307], 1.2, 0)])
def test_overflow_and_collapsed_spacing(coords, scale, translation):
    grid = make_grid((len(coords),), coords={"dim0": coords})
    with pytest.raises(ValueError, match="overflow|collapsed"):
        transform_structured_grid(grid, affine(1, scale, translation))


@pytest.mark.parametrize("case", ["attrs", "encoding", "coord_attrs", "extra_array", "extra_coord",
                                   "dims", "bounds", "type", "dtype", "instance"])
def test_unsupported_metadata_and_layout_rejected_without_mutation(case):
    grid = make_grid()
    if case == "attrs":
        grid.data.attrs["crs"] = "EPSG:1234"
    elif case == "encoding":
        grid.active_data_array.encoding["scale_factor"] = 2
    elif case == "coord_attrs":
        grid.data.dim0.attrs["units"] = "metres"
    elif case == "extra_array":
        grid.data["other"] = grid.active_data_array.copy()
    elif case == "extra_coord":
        grid.data = grid.data.assign_coords(aux=1)
    elif case == "dims":
        grid.data = grid.data.rename(dim0="depth")
    elif case == "bounds":
        grid.bounds = (0, 1)
    elif case == "type":
        grid.type = StructuredDataType.IRREGULAR_AXIS_ALIGNED
    elif case == "dtype":
        grid.dtype = "float32"
    else:
        grid.crs = "EPSG:1234"
    original = grid.data.copy(deep=True)
    with pytest.raises(ValueError):
        transform_structured_grid(grid, np.eye(4))
    assert grid.data.identical(original)


@pytest.mark.parametrize("alias", ["path", "relative", "symlink", "hardlink", "source_symlink"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_source_aliases_always_rejected(tmp_path, monkeypatch, alias, overwrite):
    source = tmp_path / "source.le"
    original = make_grid().to_binary()
    source.write_bytes(original)
    destination = tmp_path / "alias.le"
    if alias == "path":
        destination = source
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        destination = "source.le"
    elif alias == "symlink":
        destination.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, destination)
    else:
        destination = source
        source = tmp_path / "link.le"
        source.symlink_to(destination)
    with pytest.raises(ValueError, match="alias"):
        transform_structured_le(source, destination, affine(1, 2, 3), overwrite=overwrite)
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_collision_and_explicit_overwrite(tmp_path):
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()
    destination = tmp_path / "out.le"
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        transform_structured_le(source, destination, affine(1, 2, 3))
    assert destination.read_bytes() == b"keep"
    transform_structured_le(source, destination, affine(1, 2, 3), overwrite=True)
    expected = transform_structured_grid(make_grid(), affine(1, 2, 3))
    assert ops.load_le_grid(destination).data.identical(expected.data)
    assert source.read_bytes() == original


def test_dangling_symlink_collision(tmp_path):
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()
    destination = tmp_path / "out.le"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        transform_structured_le(source, destination, np.eye(4))
    assert destination.is_symlink()
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_no_clobber_race(tmp_path, monkeypatch):
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()
    destination = tmp_path / "out.le"
    link = os.link

    def racing_link(temporary, output):
        output.write_bytes(b"racing winner")
        return link(temporary, output)

    monkeypatch.setattr(ops.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        transform_structured_le(source, destination, np.eye(4))
    assert destination.read_bytes() == b"racing winner"
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("stage", ["invalid_matrix", "serialization", "readback", "fsync", "replace"])
def test_safe_failures_preserve_existing_files(tmp_path, monkeypatch, stage):
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()
    destination = tmp_path / "out.le"
    destination.write_bytes(b"keep")
    matrix = affine(1, 2, 3)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if stage == "invalid_matrix":
        matrix[0, 1] = 1
    elif stage == "serialization":
        monkeypatch.setattr(StructuredData, "to_binary", fail)
    elif stage == "readback":
        load = ops.load_le_grid
        monkeypatch.setattr(ops, "load_le_grid", lambda path: make_grid() if path != source else load(path))
    else:
        monkeypatch.setattr(ops.os, stage, fail)
    with pytest.raises((ValueError, OSError)):
        transform_structured_le(source, destination, matrix, overwrite=True)
    assert source.read_bytes() == original
    assert destination.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".*.tmp"))


def test_wrong_container_and_missing_parent(tmp_path):
    with pytest.raises(TypeError):
        transform_structured_grid(None, np.eye(4))
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    with pytest.raises(FileNotFoundError):
        transform_structured_le(source, tmp_path / "missing" / "out.le", np.eye(4))
    with pytest.raises(TypeError):
        transform_structured_le(source, tmp_path / "out.le", np.eye(4), overwrite=1)
    assert not list(tmp_path.glob(".*.tmp"))
