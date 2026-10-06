import os

import numpy as np
import pytest

from subsurface import StructuredData
from subsurface.api import _le_grid_ops as ops
from subsurface.api.le_structured_merge import merge_structured_grids, merge_structured_le


def grid(shape=(3,), *, axis=None, start=0.0, step=0.5, dtype="int64", name="labels", offset=0):
    dims = StructuredData._default_dim_names(len(shape))
    axis = dims[0] if axis is None else axis
    coords = {dim: (start if dim == axis else 10.0 * (i + 1)) + np.arange(size) * step
              for i, (dim, size) in enumerate(zip(dims, shape))}
    values = (np.arange(np.prod(shape)).reshape(shape) + offset).astype(dtype)
    result = StructuredData.from_numpy(values, coords=coords, data_array_name=name)
    result.dtype = values.dtype.str
    return result


def sources_for(tmp_path, grids):
    paths = []
    for i, tile in enumerate(grids):
        path = tmp_path / f"tile-{i}.le"
        ops.write_le_grid(tile, path, sources=[])
        paths.append(path)
    return paths


@pytest.mark.parametrize("shape,axis", [((3,), "dim0"), ((2, 3), "x"), ((2, 3), "y"),
                                        ((2, 3, 4), "x"), ((2, 3, 4), "y"), ((2, 3, 4), "z")])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16",
                                   "uint32", "uint64", "float16", "float32", "float64", ">i8", ">f8"])
def test_each_axis_dtype_and_file_order(tmp_path, shape, axis, dtype):
    index = StructuredData._default_dim_names(len(shape)).index(axis)
    other_shape = list(shape)
    other_shape[index] += 1
    first = grid(shape, axis=axis, dtype=dtype)
    second = grid(other_shape, axis=axis, start=shape[index] * 0.5, dtype=dtype, offset=57)
    originals = [tile.data.copy(deep=True) for tile in (first, second)]
    expected = np.concatenate([first.values, second.values], axis=index)
    result = merge_structured_grids(iter([first, second]), axis=axis)
    assert result.values.dtype == first.values.dtype
    assert result.active_data_array_name == "labels"
    np.testing.assert_array_equal(result.values, expected)
    paths = sources_for(tmp_path, [first, second])
    binaries = [path.read_bytes() for path in paths]
    output = merge_structured_le(iter(paths), tmp_path / "merged.le", axis=axis)
    assert output == (tmp_path / "merged.le").absolute()
    restored = ops.load_le_grid(output)
    assert restored.data.identical(result.data)
    assert restored.values.dtype == first.values.dtype
    assert [path.read_bytes() for path in paths] == binaries
    for tile, original in zip((first, second), originals):
        assert tile.data.identical(original)
        assert not np.shares_memory(result.values, tile.values)
        for dim in tile.active_data_array.dims:
            assert not np.shares_memory(result.data[dim].values, tile.data[dim].values)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("dtype", ["int64", "uint64", ">i8"])
def test_exact_integer_extremes(tmp_path, dtype):
    first, second = grid(dtype=dtype), grid(start=1.5, dtype=dtype)
    info = np.iinfo(dtype)
    first.values[:] = [info.min, info.max, info.max - 1]
    second.values[:] = [info.max - 2, 0, info.min]
    paths = sources_for(tmp_path, [first, second])
    result = ops.load_le_grid(merge_structured_le(paths, tmp_path / "out.le", axis="dim0"))
    assert result.values.dtype == first.values.dtype
    np.testing.assert_array_equal(result.values, np.concatenate([first.values, second.values]))


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", ">f8"])
def test_nan_inf_and_signed_zero(tmp_path, dtype):
    first, second = grid(dtype=dtype), grid(start=1.5, dtype=dtype)
    first.values[:] = [np.nan, np.inf, -np.inf]
    second.values[:] = [-0.0, 0.0, np.nan]
    paths = sources_for(tmp_path, [first, second])
    result = ops.load_le_grid(merge_structured_le(paths, tmp_path / "out.le", axis="dim0"))
    np.testing.assert_array_equal(result.values, np.concatenate([first.values, second.values]))
    assert np.signbit(result.values[3])
    assert not np.signbit(result.values[4])


@pytest.mark.parametrize("sizes", [(1,), (1, 1, 1), (1, 3, 1), (3, 1, 2)])
@pytest.mark.parametrize("explicit", [False, True])
def test_singleton_spacing_policy(tmp_path, sizes, explicit):
    starts = np.cumsum([0] + list(sizes[:-1])) * 0.5
    tiles = [grid((size,), start=start, offset=i * 10) for i, (size, start) in enumerate(zip(sizes, starts))]
    paths = sources_for(tmp_path, tiles)
    kwargs = {"spacing": 0.5} if explicit else {}
    if all(size == 1 for size in sizes) and not explicit:
        with pytest.raises(ValueError, match="singleton.*spacing"):
            merge_structured_grids(tiles, axis="dim0", **kwargs)
        with pytest.raises(ValueError, match="singleton.*spacing"):
            merge_structured_le(paths, tmp_path / "out.le", axis="dim0", **kwargs)
        assert not (tmp_path / "out.le").exists()
    else:
        result = merge_structured_grids(tiles, axis="dim0", **kwargs)
        np.testing.assert_array_equal(result.data.dim0, np.arange(sum(sizes)) * 0.5)
        np.testing.assert_array_equal(result.values, np.concatenate([tile.values for tile in tiles]))
        output = merge_structured_le(paths, tmp_path / "out.le", axis="dim0", **kwargs)
        assert ops.load_le_grid(output).data.identical(result.data)


def test_single_input_independent_and_axis_required():
    source = grid()
    result = merge_structured_grids([source], axis="dim0")
    result.values[0] = 99
    result.data = result.data.assign_coords(dim0=[-100, 0.5, 1])
    assert source.values[0] == 0
    assert source.data.dim0.values[0] == 0
    with pytest.raises(ValueError, match="axis"):
        merge_structured_grids([source], axis="x")
    with pytest.raises(TypeError):
        merge_structured_grids([source])


@pytest.mark.parametrize("shape,axis", [((2, 1), "x"), ((1, 3), "y"),
                                        ((2, 1, 3), "x"), ((2, 1, 3), "z"), ((1, 3, 1), "y")])
def test_singleton_nonmerge_axes_preserved(tmp_path, shape, axis):
    index = StructuredData._default_dim_names(len(shape)).index(axis)
    tiles = [grid(shape, axis=axis), grid(shape, axis=axis, start=shape[index] * 0.5)]
    paths = sources_for(tmp_path, tiles)
    restored = ops.load_le_grid(merge_structured_le(paths, tmp_path / "out.le", axis=axis))
    for dim in restored.active_data_array.dims:
        if dim != axis:
            np.testing.assert_array_equal(restored.data[dim], tiles[0].data[dim])


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, -np.inf, 10**1000])
def test_invalid_numeric_spacing(spacing):
    with pytest.raises(ValueError, match="finite.*positive"):
        merge_structured_grids([grid()], axis="dim0", spacing=spacing)


@pytest.mark.parametrize("spacing", [True, np.bool_(False), "0.5", [0.5], 0.5j])
def test_spacing_not_real_scalar(spacing):
    with pytest.raises(TypeError, match="spacing"):
        merge_structured_grids([grid()], axis="dim0", spacing=spacing)


@pytest.mark.parametrize("axis", ["x", "depth", "", "DIM0", 0, None])
def test_invalid_axis(axis):
    with pytest.raises((ValueError, TypeError), match="axis"):
        merge_structured_grids([grid()], axis=axis)


@pytest.mark.parametrize("case", ["overlap", "duplicate_endpoint", "gap", "reverse", "resolution",
                                  "explicit_spacing", "dtype", "byteorder", "name", "rank",
                                  "dims", "nonmerge_size", "nonmerge_position", "singleton_position"])
def test_incompatible_inputs_fail_without_mutation(tmp_path, case):
    first = grid((2, 3), axis="x")
    second = grid((3, 3), axis="x", start=1)
    kwargs = {}
    if case in ("overlap", "duplicate_endpoint", "gap"):
        start = {"overlap": 0, "duplicate_endpoint": 0.5, "gap": 1.5}[case]
        second = grid((3, 3), axis="x", start=start)
    elif case == "reverse":
        first, second = second, first
    elif case == "resolution":
        second = grid((3, 3), axis="x", start=1, step=0.25)
        second.data = second.data.assign_coords(y=first.data.y.values.copy())
    elif case == "explicit_spacing":
        kwargs["spacing"] = 0.25
    elif case in ("dtype", "byteorder"):
        second = grid((3, 3), axis="x", start=1, dtype="float64" if case == "dtype" else ">i8")
    elif case == "name":
        second = grid((3, 3), axis="x", start=1, name="other")
    elif case == "rank":
        second = grid((3,), start=1)
    elif case == "dims":
        second.data = second.data.transpose("y", "x")
    elif case == "nonmerge_size":
        second = grid((3, 4), axis="x", start=1)
    elif case == "nonmerge_position":
        second.data = second.data.assign_coords(y=second.data.y.values + 0.125)
    else:
        first = grid((2, 1), axis="x")
        second = grid((3, 1), axis="x", start=1)
        second.data = second.data.assign_coords(y=[20 + 1e-12])
    originals = [tile.data.copy(deep=True) for tile in (first, second)]
    with pytest.raises(ValueError):
        merge_structured_grids([first, second], axis="x", **kwargs)
    assert all(tile.data.identical(original) for tile, original in zip((first, second), originals))
    # Custom dimension order is intentionally not serializable in the shared boundary.
    if case != "dims":
        paths = sources_for(tmp_path, [first, second])
        binaries = [path.read_bytes() for path in paths]
        destination = tmp_path / "out.le"
        destination.write_bytes(b"keep")
        with pytest.raises(ValueError):
            merge_structured_le(paths, destination, axis="x", overwrite=True, **kwargs)
        assert destination.read_bytes() == b"keep"
        assert [path.read_bytes() for path in paths] == binaries
        assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("origin", [1e6, 1e12, 1e16])
@pytest.mark.parametrize("target", ["boundary", "nonmerge"])
def test_world_magnitude_cannot_hide_sample_errors(origin, target):
    first = grid((3, 3), axis="x", start=origin, step=4)
    second = grid((3, 3), axis="x", start=origin + 12, step=4)
    if target == "boundary":
        second.data = second.data.assign_coords(x=second.data.x.values + 2)
    else:
        first.data = first.data.assign_coords(y=origin + np.arange(3) * 4)
        second.data = second.data.assign_coords(y=origin + np.arange(3) * 4 + 2)
    with pytest.raises(ValueError):
        merge_structured_grids([first, second], axis="x")


def test_large_origin_exact_spacing_merges():
    first, second = grid(start=1e16, step=4), grid(start=1e16 + 12, step=4)
    result = merge_structured_grids([first, second], axis="dim0")
    np.testing.assert_array_equal(result.data.dim0, 1e16 + np.arange(6) * 4)


def test_rounded_linspace_valid_single_tile_but_ambiguous_boundary_rejected():
    first = grid((4,))
    first.data = first.data.assign_coords(dim0=np.linspace(1e6, 1e6 + 0.3, 4))
    result = merge_structured_grids([first], axis="dim0")
    assert result.data.identical(first.data)
    second = grid((4,))
    second.data = second.data.assign_coords(dim0=np.linspace(1e6 + 0.4, 1e6 + 0.7, 4))
    with pytest.raises(ValueError, match="adjacent|spacings"):
        merge_structured_grids([first, second], axis="dim0")


def test_rounded_increments_merge_when_adjacency_can_be_established():
    first, second = grid((4,)), grid((4,))
    first.data = first.data.assign_coords(dim0=np.linspace(1e4, 1e4 + 1, 4))
    second.data = second.data.assign_coords(dim0=np.linspace(1e4 + 4 / 3, 1e4 + 7 / 3, 4))
    assert np.unique(np.diff(first.data.dim0.values)).size > 1
    result = merge_structured_grids([first, second], axis="dim0")
    supplied = np.concatenate([first.data.dim0.values, second.data.dim0.values])
    assert np.all(np.abs(result.data.dim0.values - supplied) <= (1 / 3) * ops.AXIS_SPACING_RTOL)


def test_reconstruction_tolerance_and_accumulated_drift():
    first, second = grid(step=1), grid(start=3 + 2e-11, step=1)
    result = merge_structured_grids([first, second], axis="dim0")
    supplied = np.concatenate([first.data.dim0.values, second.data.dim0.values])
    assert np.all(np.abs(result.data.dim0.values - supplied) <= 1e-10)
    tiles = [grid((1,), start=i + min(i, 19 - i) * 9e-11) for i in range(20)]
    # Boundaries individually pass, but cumulative interior positional drift does not.
    with pytest.raises(ValueError, match="Combined coordinates"):
        merge_structured_grids(tiles, axis="dim0", spacing=1)


def test_spacing_agreement_uses_endpoint_span_and_tight_tolerance():
    first, second = grid(step=1), grid(start=3, step=1 + 2e-11)
    merge_structured_grids([first, second], axis="dim0", spacing=1)
    with pytest.raises(ValueError, match="spacings"):
        merge_structured_grids([first, grid(start=3, step=1 + 2e-10)], axis="dim0")
    with pytest.raises(ValueError, match="spacings"):
        merge_structured_grids([first], axis="dim0", spacing=1 + 2e-10)


def test_singleton_explicit_spacing_does_not_override_adjacency():
    with pytest.raises(ValueError, match="adjacent"):
        merge_structured_grids([grid((1,)), grid((1,), start=0.5)], axis="dim0", spacing=1)


def test_tiny_spacing_has_no_absolute_tolerance_floor():
    first = grid((2,), step=1e-300)
    second = grid((2,), start=2e-300, step=1e-300)
    result = merge_structured_grids([first, second], axis="dim0")
    np.testing.assert_array_equal(result.data.dim0, np.linspace(0, 3e-300, 4))
    with pytest.raises(ValueError, match="adjacent"):
        merge_structured_grids([first, grid((2,), start=2.1e-300, step=1e-300)], axis="dim0")


def test_nonmerge_tight_reference_tolerance():
    first, second = grid((2, 3), axis="x"), grid((2, 3), axis="x", start=1)
    second.data = second.data.assign_coords(y=second.data.y.values + 2e-11)
    result = merge_structured_grids([first, second], axis="x")
    np.testing.assert_array_equal(result.data.y, first.data.y)
    second.data = second.data.assign_coords(y=second.data.y.values + 2e-10)
    with pytest.raises(ValueError, match="Nonmerge"):
        merge_structured_grids([first, second], axis="x")


@pytest.mark.parametrize("case", ["attrs", "extra_array", "extra_coord", "instance"])
def test_shared_metadata_preflight(case):
    source = grid()
    if case == "attrs":
        source.data.attrs["crs"] = "unknown"
    elif case == "extra_array":
        source.data["extra"] = source.active_data_array.copy()
    elif case == "extra_coord":
        source.data = source.data.assign_coords(extra=1)
    else:
        source.crs = "unknown"
    original = source.data.copy(deep=True)
    with pytest.raises(ValueError):
        merge_structured_grids([source], axis="dim0")
    assert source.data.identical(original)


@pytest.mark.parametrize("alias", ["path", "relative", "symlink", "hardlink", "source_symlink"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_every_source_alias_protected(tmp_path, monkeypatch, alias, overwrite):
    paths = sources_for(tmp_path, [grid(), grid(start=1.5)])
    originals = [path.read_bytes() for path in paths]
    destination = tmp_path / "alias.le"
    if alias == "path":
        destination = paths[-1]
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        destination = paths[-1].name
    elif alias == "symlink":
        destination.symlink_to(paths[-1])
    elif alias == "hardlink":
        os.link(paths[-1], destination)
    else:
        destination = paths[-1]
        link = tmp_path / "source-link.le"
        link.symlink_to(paths[-1])
        paths[-1] = link
    with pytest.raises(ValueError, match="alias"):
        merge_structured_le(paths, destination, axis="dim0", overwrite=overwrite)
    assert [path.read_bytes() for path in paths] == originals
    assert not list(tmp_path.glob(".*.tmp"))


def test_collision_overwrite_and_dangling_symlink(tmp_path):
    paths = sources_for(tmp_path, [grid(), grid(start=1.5)])
    destination = tmp_path / "out.le"
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        merge_structured_le(paths, destination, axis="dim0")
    assert destination.read_bytes() == b"keep"
    merge_structured_le(paths, destination, axis="dim0", overwrite=True)
    assert ops.load_le_grid(destination).shape == (6,)
    dangling = tmp_path / "dangling.le"
    dangling.symlink_to(tmp_path / "missing.le")
    with pytest.raises(FileExistsError):
        merge_structured_le(paths, dangling, axis="dim0")
    assert dangling.is_symlink()


@pytest.mark.parametrize("stage", ["serialize", "readback", "link", "replace"])
def test_atomic_failure_leaves_all_sources_and_destination(tmp_path, monkeypatch, stage):
    paths = sources_for(tmp_path, [grid(), grid(start=1.5)])
    originals = [path.read_bytes() for path in paths]
    destination = tmp_path / "out.le"
    if stage != "link":
        destination.write_bytes(b"keep")

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if stage == "serialize":
        monkeypatch.setattr(StructuredData, "to_binary", fail)
    elif stage == "readback":
        monkeypatch.setattr(ops, "load_le_grid", fail)
    else:
        monkeypatch.setattr(ops.os, stage, fail)
    with pytest.raises(OSError, match="injected"):
        merge_structured_le(paths, destination, axis="dim0", overwrite=stage != "link")
    assert [path.read_bytes() for path in paths] == originals
    if stage == "link":
        assert not destination.exists()
    else:
        assert destination.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".*.tmp"))


def test_empty_wrong_types_missing_parent_and_malformed_source(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        merge_structured_grids([], axis="dim0")
    with pytest.raises(ValueError, match="At least one"):
        merge_structured_le([], tmp_path / "out.le", axis="dim0")
    with pytest.raises(TypeError):
        merge_structured_grids([None], axis="dim0")
    with pytest.raises(TypeError):
        merge_structured_le(tmp_path / "source.le", tmp_path / "out.le", axis="dim0")
    paths = sources_for(tmp_path, [grid()])
    with pytest.raises(TypeError):
        merge_structured_le(paths, tmp_path / "out.le", axis="dim0", overwrite=1)
    with pytest.raises(FileNotFoundError):
        merge_structured_le(paths, tmp_path / "absent" / "out.le", axis="dim0")
    bad = tmp_path / "bad.le"
    bad.write_bytes(b"bad")
    with pytest.raises(ValueError):
        merge_structured_le([paths[0], bad], tmp_path / "out.le", axis="dim0")
    assert not (tmp_path / "out.le").exists()
    assert not list(tmp_path.glob(".*.tmp"))
