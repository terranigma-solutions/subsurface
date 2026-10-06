import os
from pathlib import Path

import numpy as np
import pytest

from subsurface import StructuredData
from subsurface.api import le_structured_split as split
from subsurface.api._le_grid_ops import load_le_grid, write_le_grid


def make_grid(shape=(5,), dtype="int64"):
    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if values.dtype.kind in "iu":
        values.flat[0] = np.iinfo(dtype).max
        values.flat[1] = np.iinfo(dtype).min
    else:
        values.flat[:4] = [np.nan, np.inf, -np.inf, -0.0]
    dims = StructuredData._default_dim_names(len(shape))
    coords = {dim: 10.5 + index * 100 + np.arange(size) * 0.25
              for index, (dim, size) in enumerate(zip(dims, shape))}
    grid = StructuredData.from_numpy(values, coords=coords, data_array_name="lithology")
    grid.dtype = np.dtype(dtype).str
    return grid


def source_file(tmp_path, grid=None):
    grid = make_grid() if grid is None else grid
    return write_le_grid(grid, tmp_path / "source.le", sources=())


def assert_clean(tmp_path, source, original):
    assert source.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("shape", [(5,), (4, 5), (3, 4, 5), (1, 4), (3, 1, 5)])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16",
                                   "uint32", "uint64", "float16", "float32", "float64", ">i8", ">f8"])
def test_exact_windows(tmp_path, shape, dtype):
    grid = make_grid(shape, dtype)
    before = grid.data.copy(deep=True)
    source = source_file(tmp_path, grid)
    original = source.read_bytes()
    dims = grid.active_data_array.dims
    singleton = {dim: (size - 1, size) for dim, size in zip(dims, shape)}
    partial = {dims[-1]: (np.int64(1), np.int64(shape[-1]))}
    windows = {"../unsafe.le": {}, "/absolute/path": partial, "one": singleton}
    outputs = split.split_structured_le(source, tmp_path, windows=windows)
    assert list(outputs) == list(windows)
    for index, (label, ranges) in enumerate(windows.items()):
        path = outputs[label]
        assert path == tmp_path / f"grid_{index:06d}.le"
        assert path.is_absolute()
        expected = split.split_structured_grid(grid, ranges)
        actual = load_le_grid(path)
        assert actual.shape == expected.shape
        assert actual.active_data_array.dims == dims
        assert actual.active_data_array_name == "lithology"
        assert actual.dtype == expected.dtype
        assert actual.values.dtype == grid.values.dtype
        assert actual.values.tobytes() == expected.values.tobytes()
        for dim in dims:
            np.testing.assert_array_equal(actual.data[dim], expected.data[dim])
        assert not actual.data.attrs
    assert load_le_grid(outputs["one"]).shape == (1,) * len(shape)
    assert grid.data.identical(before)
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".le_structured_split.*"))


def test_in_memory_independent_copy_and_half_open():
    grid = make_grid((3, 4, 5))
    original = grid.data.copy(deep=True)
    result = split.split_structured_grid(grid, {"x": [1, 3], "z": (2, 4)})
    np.testing.assert_array_equal(result.values, grid.values[1:3, :, 2:4])
    assert result.shape == (2, 4, 2)
    assert not np.shares_memory(result.data.coords["x"].values, grid.data.coords["x"].values)
    result.values[:] = 0
    result.data = result.data.assign_coords(x=[0, 1])
    assert grid.data.identical(original)


@pytest.mark.parametrize("ranges,exception", [
    ({"dim0": (-1, 2)}, ValueError), ({"dim0": (0, 6)}, ValueError),
    ({"dim0": (2, 2)}, ValueError), ({"dim0": (3, 2)}, ValueError),
    ({"dim0": (6, 7)}, ValueError), ({"dim0": (True, 2)}, TypeError),
    ({"dim0": (0, np.bool_(True))}, TypeError), ({"dim0": (0, 2.0)}, TypeError),
    ({"dim0": ("0", 2)}, TypeError), ({"dim0": (0, 3, 1)}, ValueError),
    ({"dim0": slice(0, 3)}, ValueError), ({"dim0": (0,)}, ValueError),
    ({"x": (0, 1)}, ValueError), ({"depth": (0, 1)}, ValueError),
    ([], TypeError), (None, TypeError),
])
def test_late_invalid_window_never_writes(tmp_path, monkeypatch, ranges, exception):
    source = source_file(tmp_path)
    original = source.read_bytes()

    def unexpected_write(*args, **kwargs):
        pytest.fail("No staging allowed until every window passes preflight")

    monkeypatch.setattr(split, "write_le_grid", unexpected_write)
    with pytest.raises(exception):
        split.split_structured_le(source, tmp_path, windows={"valid": {}, "bad": ranges})
    assert_clean(tmp_path, source, original)


@pytest.mark.parametrize("windows,exception", [(None, TypeError), ([], TypeError),
                                               ({"": {}}, ValueError), ({1: {}}, ValueError),
                                               ({Path("label"): {}}, ValueError)])
def test_invalid_windows(tmp_path, windows, exception):
    source = source_file(tmp_path)
    original = source.read_bytes()
    with pytest.raises(exception):
        split.split_structured_le(source, tmp_path, windows=windows)
    assert_clean(tmp_path, source, original)


def test_empty_windows_no_staging(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    original = source.read_bytes()
    monkeypatch.setattr(split, "TemporaryDirectory", lambda **kwargs: pytest.fail("No staging"))
    assert split.split_structured_le(source, tmp_path, windows={}) == {}
    assert_clean(tmp_path, source, original)


def test_existing_directory_required(tmp_path):
    source = source_file(tmp_path)
    for directory in (tmp_path / "missing", source):
        with pytest.raises(NotADirectoryError):
            split.split_structured_le(source, directory, windows={})
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("kind", ["file", "dangling", "symlink", "hardlink", "source"])
def test_collisions_and_aliases_preflight_all(tmp_path, kind):
    source = source_file(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / "grid_000001.le"
    if kind == "file":
        destination.write_bytes(b"unrelated")
    elif kind == "dangling":
        destination.symlink_to(tmp_path / "missing")
    elif kind == "symlink":
        destination.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, destination)
    else:
        source.rename(destination)
        source = destination
    expected = FileExistsError if kind in ("file", "dangling") else ValueError
    with pytest.raises(expected):
        split.split_structured_le(source, tmp_path, windows={"first": {}, "second": {}})
    assert source.read_bytes() == original
    assert not (tmp_path / "grid_000000.le").exists()
    assert os.path.lexists(destination)
    assert not list(tmp_path.glob(".le_structured_split.*"))


def test_staging_failure_removes_all_staging(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    original = source.read_bytes()
    real_write = split.write_le_grid
    calls = []

    def failing_write(grid, destination, **kwargs):
        calls.append(destination)
        result = real_write(grid, destination, **kwargs)
        if len(calls) == 2:
            raise OSError("staging failure")
        return result

    monkeypatch.setattr(split, "write_le_grid", failing_write)
    with pytest.raises(OSError, match="staging failure"):
        split.split_structured_le(source, tmp_path, windows={"a": {}, "b": {}})
    assert len(calls) == 2
    assert_clean(tmp_path, source, original)


@pytest.mark.parametrize("mode", ["before_link", "after_link", "racer", "replaced", "source_alias"])
def test_publication_failure_rollback_preserves_racers(tmp_path, monkeypatch, mode):
    source = source_file(tmp_path)
    original = source.read_bytes()
    real_link = os.link
    first = tmp_path / "grid_000000.le"
    second = tmp_path / "grid_000001.le"

    def failing_link(src, dst, *args, **kwargs):
        dst = Path(dst)
        if dst.parent != tmp_path:
            return real_link(src, dst, *args, **kwargs)
        if dst == first:
            result = real_link(src, dst, *args, **kwargs)
            if mode == "source_alias":
                real_link(source, second)
            return result
        assert dst == second
        if mode == "after_link":
            real_link(src, dst)
        elif mode == "racer":
            dst.write_bytes(b"racer")
            return real_link(src, dst)
        elif mode == "replaced":
            first.unlink()
            first.write_bytes(b"replacement")
        raise OSError("publication failure")

    monkeypatch.setattr(split.os, "link", failing_link)
    exception = ValueError if mode == "source_alias" else OSError
    with pytest.raises(exception):
        split.split_structured_le(source, tmp_path, windows={"a": {}, "b": {}})
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".le_structured_split.*"))
    if mode == "replaced":
        assert first.read_bytes() == b"replacement"
    else:
        assert not first.exists()
    if mode == "racer":
        assert second.read_bytes() == b"racer"
    elif mode == "source_alias":
        assert os.path.samefile(second, source)
    else:
        assert not second.exists()


def test_actual_staging_readback_failure(tmp_path, monkeypatch):
    from subsurface.api import _le_grid_ops

    source = source_file(tmp_path)
    original = source.read_bytes()
    real_load = _le_grid_ops.load_le_grid

    def corrupt_readback(path):
        restored = real_load(path)
        if str(path).endswith(".tmp"):
            restored.values.flat[0] = 7
        return restored

    monkeypatch.setattr(_le_grid_ops, "load_le_grid", corrupt_readback)
    with pytest.raises(ValueError, match="scalar values"):
        split.split_structured_le(source, tmp_path, windows={"a": {}, "b": {}})
    assert_clean(tmp_path, source, original)


def test_in_memory_metadata_rejected_without_mutation():
    grid = make_grid()
    grid.data.attrs["provenance"] = "unsupported"
    before = grid.data.copy(deep=True)
    with pytest.raises(ValueError, match="metadata"):
        split.split_structured_grid(grid, {})
    assert grid.data.identical(before)


def test_cleanup_failure_attempts_remaining_owned_files(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    original = source.read_bytes()
    first = tmp_path / "grid_000000.le"
    second = tmp_path / "grid_000001.le"
    real_link = os.link
    real_unlink = Path.unlink

    def publish_then_raise(src, dst, *args, **kwargs):
        result = real_link(src, dst, *args, **kwargs)
        if Path(dst) == second:
            raise OSError("publication failure")
        return result

    def failing_cleanup(path, *args, **kwargs):
        if path == second:
            raise PermissionError("cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(split.os, "link", publish_then_raise)
    monkeypatch.setattr(Path, "unlink", failing_cleanup)
    with pytest.raises(PermissionError, match="cleanup denied") as caught:
        split.split_structured_le(source, tmp_path, windows={"a": {}, "b": {}})
    assert str(caught.value.__cause__) == "publication failure"
    assert not first.exists()
    assert second.exists()
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".le_structured_split.*"))


@pytest.mark.parametrize("publish_current", [False, True])
def test_rollback_checks_keep_all_staged_inodes_alive(tmp_path, monkeypatch, publish_current):
    source = source_file(tmp_path)
    original = source.read_bytes()
    first = tmp_path / "grid_000000.le"
    second = tmp_path / "grid_000001.le"
    real_link = os.link
    real_lstat = Path.lstat
    staged = {}
    checked = []

    def fail_current_link(src, dst, *args, **kwargs):
        dst = Path(dst)
        if dst.parent == tmp_path:
            staged[dst] = Path(src)
            if dst == second:
                if publish_current:
                    real_link(src, dst, *args, **kwargs)
                raise OSError("publication failure")
        return real_link(src, dst, *args, **kwargs)

    def check_lifetime(path, *args, **kwargs):
        if path in staged:
            assert set(staged) == {first, second}
            assert all(staging_file.exists() for staging_file in staged.values())
            checked.append(path)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(split.os, "link", fail_current_link)
    monkeypatch.setattr(Path, "lstat", check_lifetime)
    with pytest.raises(OSError, match="publication failure"):
        split.split_structured_le(source, tmp_path, windows={"a": {}, "b": {}})
    assert checked == [second, first]
    assert_clean(tmp_path, source, original)


@pytest.mark.parametrize("deny_rollback", [False, True])
def test_recycled_unpublished_inode_after_staging_cleanup_is_not_checked(
        tmp_path, monkeypatch, deny_rollback):
    source = source_file(tmp_path)
    original = source.read_bytes()
    first = tmp_path / "grid_000000.le"
    second = tmp_path / "grid_000001.le"
    real_temporary = split.TemporaryDirectory
    real_link = os.link
    real_lstat = Path.lstat
    real_unlink = Path.unlink
    unpublished_stat = None
    cleanup_finished = False
    late_checks = []

    class RecycleAfterCleanup:
        def __init__(self, **kwargs):
            self.temporary = real_temporary(**kwargs)

        def __enter__(self):
            return self.temporary.__enter__()

        def __exit__(self, *args):
            nonlocal cleanup_finished
            result = self.temporary.__exit__(*args)
            cleanup_finished = True
            second.write_bytes(b"unrelated writer after cleanup")
            return result

    def fail_unpublished_link(src, dst, *args, **kwargs):
        nonlocal unpublished_stat
        if Path(dst) == second:
            unpublished_stat = Path(src).stat()
            raise OSError("publication failure")
        return real_link(src, dst, *args, **kwargs)

    def recycled_lstat(path, *args, **kwargs):
        if path == second and cleanup_finished:
            late_checks.append(path)
            # Deterministically model legal reuse of the now-unreferenced inode.
            # No source/staging file or live published inode is replaced.
            return unpublished_stat
        return real_lstat(path, *args, **kwargs)

    def denied_unlink(path, *args, **kwargs):
        if path == first and deny_rollback:
            raise PermissionError("rollback denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(split, "TemporaryDirectory", RecycleAfterCleanup)
    monkeypatch.setattr(split.os, "link", fail_unpublished_link)
    monkeypatch.setattr(Path, "lstat", recycled_lstat)
    monkeypatch.setattr(Path, "unlink", denied_unlink)
    message = "rollback denied" if deny_rollback else "publication failure"
    with pytest.raises(OSError, match=message):
        split.split_structured_le(source, tmp_path, windows={"a": {}, "b": {}})
    assert cleanup_finished
    assert late_checks == []
    assert second.read_bytes() == b"unrelated writer after cleanup"
    assert first.exists() == deny_rollback
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".le_structured_split.*"))


def test_staging_cleanup_failure_after_success_rolls_back_publications(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    original = source.read_bytes()
    real_temporary = split.TemporaryDirectory
    real_lstat = Path.lstat
    first = tmp_path / "grid_000000.le"
    second = tmp_path / "grid_000001.le"
    checked = []
    cleanup_finished = False

    class FailAfterCleanup:
        def __init__(self, **kwargs):
            self.temporary = real_temporary(**kwargs)

        def __enter__(self):
            return self.temporary.__enter__()

        def __exit__(self, *args):
            nonlocal cleanup_finished
            assert first.exists() and second.exists()
            self.temporary.__exit__(*args)
            cleanup_finished = True
            raise OSError("staging cleanup failure")

    def check_published_reference(path, *args, **kwargs):
        if cleanup_finished and path in (first, second):
            assert not list(tmp_path.glob(".le_structured_split.*"))
            assert path.exists()
            checked.append(path)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(split, "TemporaryDirectory", FailAfterCleanup)
    monkeypatch.setattr(Path, "lstat", check_published_reference)
    with pytest.raises(OSError, match="staging cleanup failure"):
        split.split_structured_le(source, tmp_path, windows={"a": {}, "b": {}})
    assert checked == [second, first]
    assert_clean(tmp_path, source, original)
