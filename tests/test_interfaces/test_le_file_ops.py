"""Synthetic safety contracts for LE file operations, without optional backends."""

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subsurface.api import _le_file_ops as ops
from subsurface.core.structs.base_structures._liquid_earth_mesh import (
    LiquidEarthMesh, read_le_header,
)


@pytest.fixture
def mesh():
    def attributes(rows):
        return pd.DataFrame({
            "signed_id": pd.Series([2 ** 63 - 1, -(2 ** 63), 2 ** 53 + 1, -7][:rows], dtype="int64"),
            "unsigned_id": pd.Series([2 ** 64 - 1, 2 ** 63 + 1, 2 ** 53 + 1, 0][:rows], dtype="uint64"),
            "fraction": pd.Series([0.125, -2.75, 3.5, 4.25][:rows], dtype="float64"),
            "enabled": pd.Series([True, False, False, True][:rows], dtype="bool"),
        })

    return LiquidEarthMesh(
        vertex=np.arange(12, dtype=np.float64).reshape(4, 3) / 7,
        cells=np.array([[0, 2, 1], [1, 2, 3]], dtype=np.int64),
        attributes=attributes(2), points_attributes=attributes(4),
        data_attrs={"crs": "local", "nested": {"units": ["m", None, True, 2, 0.5]}},
    )


def _assert_mesh(actual, expected):
    assert isinstance(actual, LiquidEarthMesh)
    assert actual.vertex.dtype == np.dtype("float32")
    assert actual.cells.dtype == np.dtype("int32")
    np.testing.assert_array_equal(actual.vertex, expected.vertex.astype("float32"))
    np.testing.assert_array_equal(actual.cells, expected.cells)
    assert actual.data_attrs == expected.data_attrs
    for restored, original in ((actual.attributes, expected.attributes),
                               (actual.points_attributes, expected.points_attributes)):
        wire = original.copy(deep=True)
        wire["fraction"] = wire["fraction"].astype("float32")
        pd.testing.assert_frame_equal(restored, wire)


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_default_f_serialization_and_load_preserve_columns(tmp_path, mesh):
    original = copy.deepcopy(mesh)
    binary = ops.serialize_le_mesh(mesh)
    assert isinstance(binary, bytes)
    _, offset = read_le_header(binary)
    geometry = mesh.vertex.astype("float32").tobytes("F") + mesh.cells.astype("int32").tobytes("F")
    assert binary[offset:offset + len(geometry)] == geometry
    source = tmp_path / "source.le"
    source.write_bytes(binary)
    _assert_mesh(ops.load_le_mesh(source), mesh)
    _assert_mesh(LiquidEarthMesh.from_binary(binary), mesh)
    np.testing.assert_array_equal(mesh.vertex, original.vertex)
    np.testing.assert_array_equal(mesh.cells, original.cells)
    pd.testing.assert_frame_equal(mesh.attributes, original.attributes)
    pd.testing.assert_frame_equal(mesh.points_attributes, original.points_attributes)
    assert mesh.data_attrs == original.data_attrs


@pytest.mark.parametrize("malformation", ["short-prefix", "truncated", "trailing", "invalid-index"])
def test_load_uses_validated_decoder(tmp_path, mesh, malformation):
    binary = mesh.to_binary(order="F")
    if malformation == "short-prefix":
        binary = b"\x01\x00"
    elif malformation == "truncated":
        binary = binary[:-1]
    elif malformation == "trailing":
        binary += b"x"
    else:
        mesh.cells[0, 0] = len(mesh.vertex)
        binary = mesh.to_binary(order="F")
    source = tmp_path / "source.le"
    source.write_bytes(binary)
    with pytest.raises(ValueError):
        ops.load_le_mesh(source)
    assert source.read_bytes() == binary


@pytest.mark.parametrize("value", [None, object(), {"vertex": []}])
def test_serialize_requires_liquid_earth_mesh(value):
    with pytest.raises(TypeError, match="Expected LiquidEarthMesh"):
        ops.serialize_le_mesh(value)


@pytest.mark.parametrize("field,value,message", [
    ("vertex", [[0, 1, 2]], "vertex must"),
    ("vertex", np.zeros((4, 2)), "vertex must"),
    ("vertex", np.zeros((4, 3), dtype=bool), "vertex must"),
    ("vertex", np.zeros((4, 3), dtype=complex), "vertex must"),
    ("vertex", np.zeros((4, 3), dtype=object), "vertex must"),
    ("vertex", np.full((4, 3), np.nan), "Geometry must be finite"),
    ("vertex", np.full((4, 3), np.inf), "Geometry must be finite"),
    ("vertex", np.full((4, 3), -np.inf), "Geometry must be finite"),
    ("vertex", np.full((4, 3), float(np.finfo("float32").max) * 2), "float32 range"),
    ("vertex", np.full((4, 3), -float(np.finfo("float32").max) * 2), "float32 range"),
    ("cells", [[0, 1]], "cells must"),
    ("cells", np.zeros((2, 5), dtype="int32"), "cells must"),
    ("cells", np.zeros((2, 2), dtype=float), "cells must"),
    ("cells", np.zeros((2, 2), dtype=bool), "cells must"),
    ("cells", np.array([[-1, 0]]), "nonnegative int32 capacity"),
    ("cells", np.array([[2 ** 31, 0]], dtype="uint64"), "int32 capacity"),
    ("cells", np.array([[4, 0]]), "less than n_points"),
    ("cells", np.empty((5, 0), dtype="int32"), "more rows than vertices"),
])
def test_geometry_preflight_precedes_writer(mesh, monkeypatch, field, value, message):
    setattr(mesh, field, value)

    def unexpected_writer(*args, **kwargs):
        pytest.fail("Invalid geometry reached the shipped writer")

    monkeypatch.setattr(LiquidEarthMesh, "to_binary", unexpected_writer)
    with pytest.raises(ValueError, match=message):
        ops.serialize_le_mesh(mesh)


def test_vertex_count_int32_capacity_without_large_allocation(mesh, monkeypatch):
    mesh.vertex = np.broadcast_to(np.zeros((1, 3)), (2 ** 31 + 1, 3))
    monkeypatch.setattr(LiquidEarthMesh, "to_binary", lambda *a, **k: pytest.fail("writer called"))
    with pytest.raises(ValueError, match="Vertex count exceeds int32"):
        ops.serialize_le_mesh(mesh)


@pytest.mark.parametrize("association", ["attributes", "points_attributes"])
@pytest.mark.parametrize("invalid", ["not-frame", "wrong-rows", "duplicates", "object-numeric",
                                      "object-null", "nullable-int", "nullable-float", "nullable-bool",
                                      "complex", "overflow", "int64-upper", "int64-lower"])
def test_attribute_preflight_no_coercion(mesh, monkeypatch, association, invalid):
    rows = len(getattr(mesh, association))
    error, message = ValueError, "attributes must be a DataFrame"
    if invalid == "not-frame":
        frame = np.zeros((rows, 1))
    elif invalid == "wrong-rows":
        frame = pd.DataFrame(index=range(rows + 1))
    elif invalid == "duplicates":
        frame = pd.DataFrame(np.zeros((rows, 2)), columns=["same", "same"])
        message = "names must be unique"
    else:
        values, dtype = {
            "object-numeric": (["1"] * rows, object),
            "object-null": ([None] * rows, object),
            "nullable-int": ([1] * rows, "Int64"),
            "nullable-float": ([1.25] * rows, "Float64"),
            "nullable-bool": ([True] * rows, "boolean"),
            "complex": ([1 + 2j] * rows, "complex128"),
            "overflow": ([float(np.finfo("float32").max) * 2] * rows, "float64"),
            "int64-upper": ([float(2 ** 63)] * rows, "float64"),
            "int64-lower": ([np.nextafter(float(-(2 ** 63)), -np.inf)] * rows, "float64"),
        }[invalid]
        frame = pd.DataFrame({"bad": pd.Series(values, dtype=dtype)})
        if invalid == "overflow":
            message = "exceeds float32 range"
        elif invalid.startswith("int64-"):
            message = "integral float conversion exceeds int64"
        else:
            error, message = TypeError, "no coercion allowed"
    setattr(mesh, association, frame)
    before = copy.deepcopy(frame)
    monkeypatch.setattr(LiquidEarthMesh, "to_binary", lambda *a, **k: pytest.fail("writer called"))
    with pytest.raises(error, match=message):
        ops.serialize_le_mesh(mesh)
    if isinstance(frame, pd.DataFrame):
        pd.testing.assert_frame_equal(frame, before)
    else:
        np.testing.assert_array_equal(frame, before)


@pytest.mark.parametrize("association", ["attributes", "points_attributes"])
def test_empty_named_columns_rejected(mesh, association):
    mesh.vertex = np.empty((0, 3))
    mesh.cells = np.empty((0, 3), dtype="int32")
    mesh.attributes = pd.DataFrame()
    mesh.points_attributes = pd.DataFrame()
    setattr(mesh, association, pd.DataFrame({"omitted": pd.Series(dtype="float64")}))
    with pytest.raises(ValueError, match="would be dropped by the writer"):
        ops.serialize_le_mesh(mesh)


@pytest.mark.parametrize("association", ["attributes", "points_attributes"])
def test_float_wire_rules_and_nonfinite_attributes(mesh, association):
    rows = len(getattr(mesh, association))
    frame = pd.DataFrame({
        "special": np.resize([np.nan, np.inf, -np.inf, 0.125], rows),
        "integral": np.resize([float(-(2 ** 63)), float(2 ** 40)], rows),
        "tiny": np.full(rows, 1e-100),
    })
    setattr(mesh, association, frame)
    restored = LiquidEarthMesh.from_binary(ops.serialize_le_mesh(mesh))
    expected = frame.astype({"special": "float32", "integral": "int64", "tiny": "float32"})
    pd.testing.assert_frame_equal(getattr(restored, association), expected)
    pd.testing.assert_frame_equal(getattr(mesh, association), frame)


@pytest.mark.parametrize("metadata", [[], {1: "bad"}, {"nested": [{False: "bad"}]},
                                      {"tuple": (1, 2)}, {"numpy": np.int64(1)},
                                      {"array": np.array([1])}, {"set": {1}}])
def test_metadata_requires_json_native_values_and_string_keys(mesh, metadata):
    mesh.data_attrs = metadata
    with pytest.raises(TypeError, match="dictionary|keys must be strings|JSON-native"):
        ops.serialize_le_mesh(mesh)
    assert mesh.data_attrs is metadata


@pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 8])
def test_supported_connectivity_and_empty_unnamed_tables(width):
    mesh = LiquidEarthMesh(np.zeros((8, 3)),
                           np.empty((8, 0), dtype="int32") if width == 0 else np.arange(width).reshape(1, width),
                           pd.DataFrame(index=range(8 if width == 0 else 1)),
                           pd.DataFrame(index=range(8)))
    restored = LiquidEarthMesh.from_binary(ops.serialize_le_mesh(mesh))
    np.testing.assert_array_equal(restored.cells, mesh.cells)
    assert restored.attributes.shape == mesh.attributes.shape
    empty = LiquidEarthMesh(np.empty((0, 3)), np.empty((0, width), dtype="int32"),
                            pd.DataFrame(), pd.DataFrame())
    assert LiquidEarthMesh.from_binary(ops.serialize_le_mesh(empty)).cells.shape == (0, width)


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_publication_uses_expected_primitive(tmp_path, mesh, monkeypatch, overwrite):
    source = tmp_path / "source.le"
    source.write_bytes(ops.serialize_le_mesh(mesh))
    destination = tmp_path / "output.le"
    if overwrite:
        destination.write_bytes(b"old destination")
    monkeypatch.chdir(tmp_path)
    primitive = "replace" if overwrite else "link"
    real_publish = getattr(os, primitive)
    calls = []

    def publish(temporary, target):
        assert Path(temporary).parent == tmp_path
        assert Path(temporary) != destination
        _assert_mesh(ops.load_le_mesh(temporary), mesh)
        calls.append((temporary, target))
        real_publish(temporary, target)

    monkeypatch.setattr(ops.os, primitive, publish)
    monkeypatch.setattr(ops.os, "link" if overwrite else "replace",
                        lambda *a, **k: pytest.fail("Wrong publication primitive"))
    result = ops.write_le_mesh(mesh, "output.le", sources=[source], overwrite=overwrite)
    assert result == destination and result.is_absolute()
    assert len(calls) == 1
    _assert_mesh(ops.load_le_mesh(result), mesh)
    assert source.read_bytes() == destination.read_bytes()
    assert set(_snapshot(tmp_path)) == {"source.le", "output.le"}


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_source_aliases_always_rejected(tmp_path, mesh, alias, overwrite):
    source = tmp_path / "source.le"
    source.write_bytes(ops.serialize_le_mesh(mesh))
    destination = source if alias == "direct" else tmp_path / "alias.le"
    if alias == "symlink":
        destination.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, destination)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="must not alias"):
        ops.write_le_mesh(mesh, destination, sources=[tmp_path / "unrelated.le", source], overwrite=overwrite)
    assert _snapshot(tmp_path) == before


def test_existing_and_dangling_destinations_not_clobbered(tmp_path, mesh):
    destination = tmp_path / "output.le"
    for dangling in (False, True):
        if dangling:
            destination.symlink_to(tmp_path / "missing.le")
        else:
            destination.write_bytes(b"existing")
        with pytest.raises(FileExistsError):
            ops.write_le_mesh(mesh, destination, sources=[])
        if dangling:
            assert destination.is_symlink() and not destination.exists()
        else:
            assert destination.read_bytes() == b"existing"
        assert list(tmp_path.iterdir()) == [destination]
        destination.unlink()


@pytest.mark.parametrize("failure", ["write", "short-write", "fsync", "link", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_io_failures_leave_files_unchanged_and_no_temporary(tmp_path, mesh, monkeypatch, failure, existing):
    source = tmp_path / "source.le"
    source.write_bytes(ops.serialize_le_mesh(mesh))
    destination = tmp_path / "output.le"
    if existing:
        destination.write_bytes(b"original destination")
    before = _snapshot(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("injected " + failure)

    if failure in ("write", "short-write"):
        real_temporary = ops.tempfile.NamedTemporaryFile

        class FailingWrite:
            def __init__(self, *args, **kwargs):
                self.stream = real_temporary(*args, **kwargs)

            def __enter__(self):
                self.stream.__enter__()
                return self

            @property
            def name(self):
                return self.stream.name

            def write(self, binary):
                self.stream.write(binary[:5])
                if failure == "short-write":
                    return 5
                fail()

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

        monkeypatch.setattr(ops.tempfile, "NamedTemporaryFile", FailingWrite)
    else:
        monkeypatch.setattr(ops.os, failure, fail)
    # Link failures exercise no-clobber; an existing target must instead fail early.
    overwrite = failure == "replace" or (existing and failure != "link")
    error = FileExistsError if existing and failure == "link" else OSError
    with pytest.raises(error):
        ops.write_le_mesh(mesh, destination, sources=[source], overwrite=overwrite)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("race_stage", ["after-fsync", "at-link"])
def test_publication_race_does_not_clobber_winner(tmp_path, mesh, monkeypatch, race_stage):
    source = tmp_path / "source.le"
    source.write_bytes(ops.serialize_le_mesh(mesh))
    source_bytes = source.read_bytes()
    destination = tmp_path / "output.le"
    winner = b"concurrent writer won"
    primitive = "fsync" if race_stage == "after-fsync" else "link"
    real_operation = getattr(os, primitive)

    def race(*args, **kwargs):
        destination.write_bytes(winner)
        return real_operation(*args, **kwargs)

    monkeypatch.setattr(ops.os, primitive, race)
    with pytest.raises(FileExistsError):
        ops.write_le_mesh(mesh, destination, sources=[source])
    assert _snapshot(tmp_path) == {"source.le": source_bytes, "output.le": winner}


@pytest.mark.parametrize("existing", [False, True])
def test_malformed_serialization_decoded_before_any_temporary(tmp_path, mesh, monkeypatch, existing):
    source = tmp_path / "source.le"
    source.write_bytes(ops.serialize_le_mesh(mesh))
    destination = tmp_path / "output.le"
    if existing:
        destination.write_bytes(b"unchanged")
    before = _snapshot(tmp_path)
    malformed = mesh.to_binary(order="F")[:-1]
    monkeypatch.setattr(LiquidEarthMesh, "to_binary", lambda *a, **k: malformed)
    monkeypatch.setattr(ops.tempfile, "NamedTemporaryFile",
                        lambda *a, **k: pytest.fail("Temporary created before decoder validation"))
    with pytest.raises(ValueError, match="payload length mismatch"):
        ops.write_le_mesh(mesh, destination, sources=[source], overwrite=existing)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("invalid", ["geometry", "connectivity", "attributes", "metadata"])
@pytest.mark.parametrize("existing", [False, True])
def test_preflight_failure_leaves_files_unchanged(tmp_path, mesh, monkeypatch, invalid, existing):
    source = tmp_path / "source.le"
    source.write_bytes(ops.serialize_le_mesh(mesh))
    destination = tmp_path / "output.le"
    if existing:
        destination.write_bytes(b"original destination")
    before = _snapshot(tmp_path)
    if invalid == "geometry":
        mesh.vertex[0, 0] = np.nan
    elif invalid == "connectivity":
        mesh.cells[0, 0] = -1
    elif invalid == "attributes":
        mesh.attributes["object"] = pd.Series(["1", "2"], dtype=object)
    else:
        mesh.data_attrs = {"nested": {1: "not a string key"}}
    monkeypatch.setattr(ops.tempfile, "NamedTemporaryFile",
                        lambda *a, **k: pytest.fail("Temporary created before preflight"))
    with pytest.raises((ValueError, TypeError)):
        ops.write_le_mesh(mesh, destination, sources=[source], overwrite=existing)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("alteration", ["geometry", "connectivity", "column-value", "column-dtype", "column-loss", "metadata"])
def test_valid_but_lossy_writer_output_rejected_before_temporary(tmp_path, mesh, monkeypatch, alteration):
    altered = copy.deepcopy(mesh)
    if alteration == "geometry":
        altered.vertex[0, 0] += 1
    elif alteration == "connectivity":
        altered.cells[0, 0] = 3
    elif alteration == "column-value":
        altered.attributes.loc[0, "signed_id"] = 17
    elif alteration == "column-dtype":
        altered.attributes["enabled"] = altered.attributes["enabled"].astype("uint8")
    elif alteration == "column-loss":
        altered.attributes = altered.attributes.drop(columns="enabled")
    else:
        altered.data_attrs = {}
    binary = altered.to_binary(order="F")
    # The decoder accepts this payload; the helper must also compare wire values.
    LiquidEarthMesh.from_binary(binary)
    monkeypatch.setattr(LiquidEarthMesh, "to_binary", lambda *a, **k: binary)
    monkeypatch.setattr(ops.tempfile, "NamedTemporaryFile",
                        lambda *a, **k: pytest.fail("Temporary created before fidelity validation"))
    with pytest.raises(ValueError, match="differs|were lost"):
        ops.write_le_mesh(mesh, tmp_path / "output.le", sources=[])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("overwrite", [1, None, "yes", np.bool_(True)])
def test_overwrite_requires_boolean(tmp_path, mesh, overwrite):
    with pytest.raises(TypeError, match="overwrite must be a boolean"):
        ops.write_le_mesh(mesh, tmp_path / "output.le", sources=[], overwrite=overwrite)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source_kind", [str, bytes, Path])
def test_sources_requires_iterable_not_single_path(tmp_path, mesh, source_kind):
    source = source_kind("source.le") if source_kind is not bytes else b"source.le"
    with pytest.raises(TypeError, match="iterable of source paths"):
        ops.write_le_mesh(mesh, tmp_path / "output.le", sources=source)
    assert not list(tmp_path.iterdir())


def test_parent_must_exist(tmp_path, mesh):
    with pytest.raises(FileNotFoundError):
        ops.write_le_mesh(mesh, tmp_path / "missing" / "output.le", sources=[])
    assert not list(tmp_path.iterdir())


def test_json_nonfinite_metadata_is_retained(mesh):
    mesh.data_attrs = {"nan": float("nan"), "pos_inf": float("inf"), "neg_inf": -float("inf")}
    restored = LiquidEarthMesh.from_binary(ops.serialize_le_mesh(mesh))
    assert np.isnan(restored.data_attrs["nan"])
    assert restored.data_attrs["pos_inf"] == float("inf")
    assert restored.data_attrs["neg_inf"] == -float("inf")
