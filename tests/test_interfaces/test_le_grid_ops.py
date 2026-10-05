import json
import os

import numpy as np
import pytest

from subsurface import StructuredData
from subsurface.api import _le_grid_ops as ops
from subsurface.core.structs.base_structures.structured_data import StructuredDataType


def make_grid(shape=(4,), dtype="float64", coords=None, name="labels"):
    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    dims = StructuredData._default_dim_names(len(shape))
    if coords is None:
        coords = {dim: 12.5 + np.arange(size) * 0.25 for dim, size in zip(dims, shape)}
    grid = StructuredData.from_numpy(values, coords=coords, data_array_name=name)
    grid.dtype = np.dtype(dtype).str
    return grid


def assert_no_temps(directory):
    assert not list(directory.glob(".*.tmp"))


@pytest.mark.parametrize("shape", [(5,), (1,), (2, 3), (1, 3), (2, 1, 3), (1, 1, 1)])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16",
                                   "uint32", "uint64", "float16", "float32", "float64", ">i8", ">f8"])
def test_numeric_roundtrip(tmp_path, shape, dtype):
    grid = make_grid(shape, dtype)
    original = grid.data.copy(deep=True)
    output = ops.write_le_grid(grid, tmp_path / "grid.le", sources=[])
    assert output.is_absolute()
    restored = ops.load_le_grid(output)
    assert restored.active_data_array_name == "labels"
    assert restored.values.dtype == grid.values.dtype
    np.testing.assert_array_equal(restored.values, grid.values)
    for dim in grid.active_data_array.dims:
        np.testing.assert_array_equal(restored.data[dim], grid.data[dim])
    assert grid.data.identical(original)
    assert_no_temps(tmp_path)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_integer_extremes_are_exact(tmp_path, dtype):
    grid = make_grid(dtype=dtype)
    info = np.iinfo(dtype)
    grid.values[:] = [info.min, info.max, info.max - 1, 0]
    output = ops.write_le_grid(grid, tmp_path / "extremes.le", sources=[])
    np.testing.assert_array_equal(ops.load_le_grid(output).values, grid.values)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_nonfinite_scalars_are_preserved(tmp_path, dtype):
    grid = make_grid(dtype=dtype)
    grid.values[:] = [np.nan, np.inf, -np.inf, -0.0]
    output = ops.write_le_grid(grid, tmp_path / "nonfinite.le", sources=[])
    restored = ops.load_le_grid(output)
    np.testing.assert_array_equal(restored.values, grid.values)
    assert np.signbit(restored.values[-1])


@pytest.mark.parametrize("coords", [[0, 1, 2, 4], [3, 2, 1, 0], [0, 1, 1, 2],
                                    [0, 1, np.nan, 3], [0, 1, np.inf, 3],
                                    [True, False, True, False], ["0", "1", "2", "3"],
                                    np.array([2**63, 2**63 + 1, 2**63 + 2, 2**63 + 3], dtype="uint64"),
                                    [-1e308, -1e307, 1e307, 1e308]])
def test_invalid_coordinates_rejected(tmp_path, coords):
    grid = make_grid(coords={"dim0": coords})
    with pytest.raises(ValueError):
        ops.write_le_grid(grid, tmp_path / "bad.le", sources=[])
    assert not list(tmp_path.iterdir())


def test_spacing_not_origin_controls_tolerance(tmp_path):
    # Origin-relative allclose would accept this half-sample error.
    grid = make_grid(coords={"dim0": [1e12, 1e12 + 1, 1e12 + 2.5, 1e12 + 3]})
    with pytest.raises(ValueError, match="uniform"):
        ops.validate_le_grid(grid)
    # Even a reader-generated linspace can have sample-scale rounding jitter.
    grid = make_grid(coords={"dim0": np.linspace(1e16, 1e16 + 10, 4)})
    source = tmp_path / "rounded.le"
    source.write_bytes(grid.to_binary())
    with pytest.raises(ValueError, match="uniform"):
        ops.load_le_grid(source)


def test_tight_reconstruction_tolerance(tmp_path):
    grid = make_grid(coords={"dim0": [0, 1 + 2e-11, 2, 3]})
    output = ops.write_le_grid(grid, tmp_path / "tiny.le", sources=[])
    np.testing.assert_array_equal(ops.load_le_grid(output).data.dim0, [0, 1, 2, 3])
    grid.data = grid.data.assign_coords(dim0=[0, 1 + 2e-10, 2, 3])
    with pytest.raises(ValueError, match="uniform"):
        ops.validate_le_grid(grid)


@pytest.mark.parametrize("case", ["type", "bounds", "extra_array", "extra_coord", "missing_coord",
                                   "dataset_attrs", "array_attrs", "coord_attrs", "dataset_encoding",
                                   "array_encoding", "coord_encoding", "instance_metadata", "dims",
                                   "dtype_mismatch", "dtype_nonstring", "dtype_invalid", "missing_active"])
def test_loss_preflight(case):
    grid = make_grid()
    if case == "type":
        grid.type = StructuredDataType.IRREGULAR_AXIS_ALIGNED
    elif case == "bounds":
        grid.bounds = (0, 1, 0, 1, 0, 1)
    elif case == "extra_array":
        grid.data["other"] = grid.active_data_array.copy()
    elif case == "extra_coord":
        grid.data = grid.data.assign_coords(aux=7)
    elif case == "missing_coord":
        grid.data = grid.data.drop_vars("dim0")
    elif case.endswith("attrs") or case.endswith("encoding"):
        owner, field = case.split("_")
        target = {"dataset": grid.data, "array": grid.active_data_array, "coord": grid.data.dim0}[owner]
        getattr(target, field)["units"] = "metres"
    elif case == "instance_metadata":
        grid.crs = "EPSG:1234"
    elif case == "dims":
        grid.data = grid.data.rename({"dim0": "depth"})
    elif case == "dtype_mismatch":
        grid.dtype = "float32"
    elif case == "dtype_nonstring":
        grid.dtype = np.dtype("float64")
    elif case == "dtype_invalid":
        grid.dtype = "not-a-dtype"
    elif case == "missing_active":
        grid.active_data_array_name = "missing"
    with pytest.raises(ValueError):
        ops.validate_le_grid(grid)


def test_default_float32_field_cannot_downcast_integers():
    grid = StructuredData.from_numpy(np.array([2**63 - 1], dtype="int64"), coords={"dim0": [0]})
    with pytest.raises(ValueError, match="match.*dtype"):
        ops.validate_le_grid(grid)


def test_coordinate_must_use_its_own_dimension():
    grid = make_grid(coords={"dim0": ("other", [0, 1, 2, 3])})
    with pytest.raises(ValueError, match="one-dimensional"):
        ops.validate_le_grid(grid)


def test_nonrepresentable_singleton_position_rejected():
    grid = make_grid((1,), coords={"dim0": np.array([2**63 + 1], dtype="uint64")})
    with pytest.raises(ValueError, match="lose precision"):
        ops.validate_le_grid(grid)


@pytest.mark.parametrize("target", ["scalar", "coordinate"])
def test_numpy_dtype_metadata_rejected(target):
    dtype = np.dtype("float64", metadata={"units": "metres"})
    if target == "scalar":
        grid = make_grid(dtype=dtype)
    else:
        grid = make_grid(coords={"dim0": np.arange(4).astype(dtype)})
    with pytest.raises(ValueError, match="dtype metadata"):
        ops.validate_le_grid(grid)


@pytest.mark.parametrize("dtype", ["bool", "complex128", "object", "U2"])
def test_unsupported_scalar_dtype(dtype):
    with pytest.raises(ValueError):
        ops.validate_le_grid(make_grid(dtype=dtype))


@pytest.mark.parametrize("shape", [(), (0,), (1, 1, 1, 1)])
def test_invalid_rank_or_empty(shape):
    with pytest.raises(ValueError):
        ops.validate_le_grid(make_grid(shape))


@pytest.mark.parametrize("name", ["", "  ", "\ud800", 42])
def test_invalid_name(name):
    with pytest.raises(ValueError):
        ops.validate_le_grid(make_grid(name=name))


def test_invalid_inputs_and_missing_parent(tmp_path):
    with pytest.raises(TypeError):
        ops.validate_le_grid(None)
    with pytest.raises(TypeError):
        ops.write_le_grid(make_grid(), tmp_path / "out.le", sources="input.le")
    with pytest.raises(TypeError):
        ops.write_le_grid(make_grid(), tmp_path / "out.le", sources=[], overwrite=1)
    with pytest.raises(FileNotFoundError):
        ops.write_le_grid(make_grid(), tmp_path / "absent" / "out.le", sources=[])
    assert_no_temps(tmp_path)


@pytest.mark.parametrize("alias", ["path", "relative", "symlink", "hardlink", "source_symlink"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_source_aliases_are_never_writable(tmp_path, monkeypatch, alias, overwrite):
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
        ops.write_le_grid(make_grid(), destination, sources=[source], overwrite=overwrite)
    assert source.read_bytes() == original
    assert_no_temps(tmp_path)


def test_collision_and_explicit_overwrite(tmp_path):
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()
    destination = tmp_path / "existing.le"
    destination.write_bytes(b"keep")
    grid = ops.load_le_grid(source)
    with pytest.raises(FileExistsError):
        ops.write_le_grid(grid, destination, sources=[source])
    assert destination.read_bytes() == b"keep"
    ops.write_le_grid(grid, destination, sources=[source], overwrite=True)
    assert source.read_bytes() == original
    assert ops.load_le_grid(destination).data.identical(grid.data)
    assert_no_temps(tmp_path)


def test_dangling_symlink_collision(tmp_path):
    destination = tmp_path / "out.le"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        ops.write_le_grid(make_grid(), destination, sources=[])
    assert destination.is_symlink()
    assert_no_temps(tmp_path)


@pytest.mark.parametrize("change", ["values", "dtype", "name", "coords", "malformed"])
def test_written_temp_is_validated_before_replacement(tmp_path, monkeypatch, change):
    grid = make_grid()
    replacement = make_grid()
    if change == "values":
        replacement.values[0] = 99
    elif change == "dtype":
        replacement = make_grid(dtype="float32")
    elif change == "name":
        replacement = make_grid(name="changed")
    elif change == "coords":
        replacement = make_grid(coords={"dim0": [1, 2, 3, 4]})
    binary = b"bad" if change == "malformed" else replacement.to_binary()
    source = tmp_path / "source.le"
    original = grid.to_binary()
    source.write_bytes(original)
    monkeypatch.setattr(StructuredData, "to_binary", lambda self, order: binary)
    destination = tmp_path / "existing.le"
    destination.write_bytes(b"keep")
    with pytest.raises(ValueError):
        ops.write_le_grid(grid, destination, sources=[source], overwrite=True)
    assert destination.read_bytes() == b"keep"
    assert source.read_bytes() == original
    assert_no_temps(tmp_path)


def test_serialization_failure_preserves_source_and_destination(tmp_path, monkeypatch):
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()
    destination = tmp_path / "out.le"
    destination.write_bytes(b"keep")

    def fail(self, order):
        raise ValueError("serialization failed")

    monkeypatch.setattr(StructuredData, "to_binary", fail)
    with pytest.raises(ValueError, match="serialization"):
        ops.write_le_grid(make_grid(), destination, sources=[source], overwrite=True)
    assert source.read_bytes() == original
    assert destination.read_bytes() == b"keep"
    assert_no_temps(tmp_path)


@pytest.mark.parametrize("stage", ["short_write", "flush", "fsync", "link", "replace"])
def test_io_failure_cleanup(tmp_path, monkeypatch, stage):
    destination = tmp_path / "out.le"
    if stage == "replace":
        destination.write_bytes(b"keep")
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if stage in ("short_write", "flush"):
        factory = ops.tempfile.NamedTemporaryFile

        def temporary(**kwargs):
            stream = factory(**kwargs)
            if stage == "short_write":
                write = stream.write
                stream.write = lambda binary: write(binary[:-1])
            else:
                stream.flush = fail
            return stream

        monkeypatch.setattr(ops.tempfile, "NamedTemporaryFile", temporary)
    else:
        monkeypatch.setattr(ops.os, stage, fail)
    with pytest.raises(OSError):
        ops.write_le_grid(make_grid(), destination, sources=[source], overwrite=stage == "replace")
    assert source.read_bytes() == original
    if stage == "replace":
        assert destination.read_bytes() == b"keep"
    else:
        assert not destination.exists()
    assert_no_temps(tmp_path)


def test_atomic_no_clobber_race(tmp_path, monkeypatch):
    destination = tmp_path / "raced.le"
    link = os.link

    def racing_link(temporary, output):
        output.write_bytes(b"racing winner")
        return link(temporary, output)

    monkeypatch.setattr(ops.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        ops.write_le_grid(make_grid(), destination, sources=[])
    assert destination.read_bytes() == b"racing winner"
    assert_no_temps(tmp_path)


def test_alias_rechecked_after_temp_validation(tmp_path, monkeypatch):
    source = tmp_path / "source.le"
    source.write_bytes(make_grid().to_binary())
    original = source.read_bytes()
    destination = tmp_path / "out.le"
    load = ops.load_le_grid

    def race(path):
        result = load(path)
        os.link(source, destination)
        return result

    monkeypatch.setattr(ops, "load_le_grid", race)
    with pytest.raises(ValueError, match="alias"):
        ops.write_le_grid(make_grid(), destination, sources=[source], overwrite=True)
    assert source.read_bytes() == destination.read_bytes() == original
    assert_no_temps(tmp_path)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "name", "bounds", "transform", "truncated", "trailing"])
def test_load_rejects_malformed_files(tmp_path, mutation):
    binary = make_grid().to_binary()
    size = int.from_bytes(binary[:4], "little")
    header = json.loads(binary[4:4 + size])
    payload = binary[4 + size:]
    if mutation == "shape":
        header["data_shape"] = [0]
    elif mutation == "dtype":
        header["dtype"] = "object"
    elif mutation == "name":
        header["data_name"] = "dim0"
    elif mutation == "bounds":
        header["bounds"]["dim0"] = [2, 1]
    elif mutation == "transform":
        header["transform"] = np.eye(4).tolist()
    elif mutation == "truncated":
        payload = payload[:-1]
    else:
        payload += b"extra"
    encoded = json.dumps(header).encode()
    source = tmp_path / "bad.le"
    source.write_bytes(len(encoded).to_bytes(4, "little") + encoded + payload)
    with pytest.raises(ValueError):
        ops.load_le_grid(source)
