"""Native .le ingestion profiles: every admit/reject rule of subsurface.api.le_validation."""

import json
import struct
import zlib

import numpy as np
import pytest

from subsurface.api import LEValidationError, validate_le, validate_le_dataset, validate_png_texture
from subsurface.api.le_validation import MAX_INGEST_HEADER_BYTES, MAX_MESH_ATTRIBUTE_RUNS


def _frame(header, payload=b"", raw=None):
    body = raw if raw is not None else json.dumps(header).encode()
    return len(body).to_bytes(4, "little") + body + payload


def _column(name, dtype, values):
    values = np.asarray(values)
    data = values.astype(np.uint8 if dtype == "bool" else dtype).tobytes()
    return {"name": name, "dtype": dtype, "shape": [len(values)], "byte_length": len(data)}, data


def _unstructured(vertices, cells, cell_attrs=(), vertex_attrs=(), attrs=None, **overrides):
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    cells = np.asarray(cells, dtype=np.int32)
    columns = {"cell": [_column(*c) for c in cell_attrs], "vertex": [_column(*c) for c in vertex_attrs]}
    header = {
            "format_version": 2, "vertex_shape": list(vertices.shape), "cell_shape": list(cells.shape),
            "cell_attrs"    : [c[0] for c in columns["cell"]], "vertex_attrs": [c[0] for c in columns["vertex"]],
            "xarray_attrs"  : {} if attrs is None else attrs,
    }
    header.update(overrides)
    payload = vertices.tobytes("F") + cells.tobytes("F") + b"".join(c[1] for c in columns["cell"] + columns["vertex"])
    return _frame(header, payload)


def _structured(values, bounds, name="density", **overrides):
    values = np.asarray(values, dtype=np.float32)
    header = {"data_shape": list(values.shape), "bounds": bounds, "transform": None, "dtype": "float32", "data_name": name}
    header.update(overrides)
    return _frame(header, values.tobytes("F"))


QUAD = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
TRIS = [[0, 1, 2], [0, 2, 3]]
GRID_BOUNDS = {"x": [0, 2], "y": [0, 3], "z": [0, 1]}


def _mesh(**kwargs):
    return _unstructured(QUAD, TRIS, **kwargs)


def _grid(shape=(3, 4, 2), bounds=None, **kwargs):
    return _structured(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), bounds or GRID_BOUNDS, **kwargs)


def _code(code, data, family="mesh", role=None, match=None):
    with pytest.raises(LEValidationError, match=match) as error:
        validate_le(data, family=family, role=role)
    assert error.value.code == code
    return error.value


# region Common


@pytest.mark.parametrize("data, code", [
        (b"\x01\x00", "malformed"),
        (b"\x00\x00\x00\x00", "malformed"),
        ((MAX_INGEST_HEADER_BYTES + 1).to_bytes(4, "little"), "unsupported_profile"),
        (b"\x10\x00\x00\x00{}", "malformed"),
        (_frame(None, raw=b"\xff\xfe"), "malformed"),
        (_frame(None, raw=b"[1, 2]"), "malformed"),
        (_frame(None, raw=b'{"a": 1, "a": 2}'), "malformed"),
        (_frame(None, raw=b'{"a": NaN}'), "unsupported_profile"),
        (_frame(None, raw=b'{"a": -Infinity}'), "unsupported_profile"),
        (_frame(None, raw=b'{"a": 1e400}'), "unsupported_profile"),
        (_frame(None, raw=b'{"a": ' + b"[" * 40 + b"]" * 40 + b"}"), "unsupported_profile"),
        (_frame({"vertex_shape": [1, 3], "data_shape": [1]}), "malformed"),
        (_frame({"format_version": 2}), "malformed"),
])
def test_common_framing_and_header(data, code):
    _code(code, data)


def test_encoding_must_be_declared_fortran_little_endian():
    with pytest.raises(LEValidationError) as error:
        validate_le(_mesh(), family="mesh", encoding="le-c-little-endian-v1")
    assert error.value.code == "unsupported_profile"


@pytest.mark.parametrize("key", ["time_series_id", "timestamp", "time_series"])
def test_temporal_files_are_unsupported(key):
    _code("unsupported_profile", _mesh(attrs={key: "x"}), match="time-series")
    _code("unsupported_profile", _grid(xarray_attrs={key: "x"}), family="volume", match="time-series")

# endregion

# region Unstructured


def test_unstructured_header_schema():
    legacy = _frame({"vertex_shape": [4, 3], "cell_shape": [2, 3], "cell_attr_shape": [0, 0]},
                    np.zeros((4, 3), np.float32).tobytes() + np.zeros((2, 3), np.int32).tobytes())
    _code("unsupported_profile", legacy, match="legacy v1")
    _code("unsupported_profile", _mesh(format_version=3))
    _code("unsupported_profile", _mesh(format_version=True))
    _code("unsupported_profile", _mesh(lods=[]), match="LOD")
    _code("unsupported_profile", _mesh(extra=1), match="unsupported header fields")
    header_without_attrs = _mesh()
    header = json.loads(header_without_attrs[4:4 + int.from_bytes(header_without_attrs[:4], "little")])
    del header["cell_attrs"]
    _code("malformed", _frame(header, header_without_attrs[4 + int.from_bytes(header_without_attrs[:4], "little"):]))
    _code("unsupported_profile", _mesh(attrs={"isMaster": True}))
    _code("unsupported_profile", _mesh(attrs={"name": 3}))


def test_payload_length_is_exact():
    _code("malformed", _mesh() + b"\x00")
    _code("malformed", _mesh()[:-1])


def test_vertices_must_be_finite():
    _code("unsupported_profile", _unstructured([[0, 0, np.nan]] + QUAD[1:], TRIS), match="finite")
    _code("unsupported_profile", _unstructured([[0, 0, np.inf]] + QUAD[1:], TRIS), match="finite")


def test_connectivity_must_be_in_range():
    _code("malformed", _unstructured(QUAD, [[0, 1, 4], [0, 2, 3]]))
    _code("malformed", _unstructured(QUAD, [[0, 1, -1], [0, 2, 3]]))


@pytest.mark.parametrize("dtype", ["int16", "uint8", "float16", "<i8", "int"])
def test_attribute_dtypes_are_the_ones_unity_reads(dtype):
    np_dtype = {"int": "int64"}.get(dtype, dtype)
    header, data = _column("a", np_dtype, [1, 2])
    header["dtype"] = dtype
    vertices = np.asarray(QUAD, np.float32)
    cells = np.asarray(TRIS, np.int32)
    raw = {"format_version": 2, "vertex_shape": [4, 3], "cell_shape": [2, 3], "cell_attrs": [header], "vertex_attrs": [],
           "xarray_attrs": {}}
    _code("unsupported_profile", _frame(raw, vertices.tobytes("F") + cells.tobytes("F") + data), match="LiquidEarth reads")


def test_attribute_names_and_fields():
    _code("unsupported_profile", _unstructured(QUAD, TRIS, cell_attrs=[(1, "int32", [1, 1])]), match="names")
    _code("unsupported_profile", _unstructured(QUAD, TRIS, cell_attrs=[("", "int32", [1, 1])]), match="names")
    data = _mesh(cell_attrs=[("a", "int32", [1, 1])])
    length = int.from_bytes(data[:4], "little")
    header = json.loads(data[4:4 + length])
    header["cell_attrs"][0]["unit"] = "m"
    _code("unsupported_profile", _frame(header, data[4 + length:]), match="exactly name")
    many = [(f"c{i}", "int32", [1, 1]) for i in range(257)]
    _code("unsupported_profile", _mesh(cell_attrs=many), match="more than 256")


def test_attribute_values():
    _code("malformed", _bool_two(), match="0 and 1")
    _code("unsupported_profile", _mesh(vertex_attrs=[("a", "float32", [0, 1, np.inf, 2])]), match="infinite")
    validate_le(_mesh(vertex_attrs=[("a", "float32", [0, 1, np.nan, 2])]), family="mesh")
    _code("unsupported_profile", _mesh(vertex_attrs=[("a", "float64", [0, 1, 1e39, 2])]), match="float32 range")
    _code("unsupported_profile", _mesh(vertex_attrs=[("id", "int64", [0, 1, 2 ** 31, 2])]), match="int32 range")
    _code("unsupported_profile", _mesh(vertex_attrs=[("id", "int64", [0, 1, -2 ** 31 - 1, 2])]), match="int32 range")
    report = validate_le(_mesh(vertex_attrs=[("id", "int64", [0, 1, 2 ** 31 - 1, -2 ** 31])]), family="mesh")
    assert [(a.name, a.dtype) for a in report.attributes] == [("id", "int64")]


def _bool_two():
    header, data = _column("flag", "bool", [0, 1])
    vertices = np.asarray(QUAD, np.float32).tobytes("F")
    cells = np.asarray(TRIS, np.int32).tobytes("F")
    raw = {"format_version": 2, "vertex_shape": [4, 3], "cell_shape": [2, 3], "cell_attrs": [header],
           "vertex_attrs": [], "xarray_attrs": {}}
    return _frame(raw, vertices + cells + b"\x00\x02")

# endregion

# region Mesh


def test_mesh_profile():
    report = validate_le(_mesh(cell_attrs=[("id", "int32", [3, 3])], vertex_attrs=[("z", "float32", [0, 1, 2, 3])]),
                         family="mesh")
    assert (report.vertex_count, report.cell_count, report.cell_width) == (4, 2, 3)
    _code("unsupported_profile", _unstructured(QUAD, [[0, 1], [1, 2]]), match="not a supported mesh")
    error = _code("family_mismatch", _unstructured(QUAD, [[0, 1], [1, 2]], attrs={"well_id_mapper": {"a": 0}}))
    assert error.detected_family == "wells"
    assert _code("family_mismatch", _unstructured(QUAD, [[0], [1], [2], [3]])).detected_family == "pointcloud"
    assert _code("family_mismatch", _grid()).detected_family == "volume"
    _code("unsupported_profile", _unstructured(QUAD, [[0, 1, 2, 3]]))
    _code("unsupported_profile", _unstructured(QUAD, np.zeros((0, 3))), match="triangle")


def _alternating(count):
    return [i % 2 for i in range(count)]


def test_mesh_attribute_runs_follow_unity_splitting():
    rows = MAX_MESH_ATTRIBUTE_RUNS + 1
    vertices = np.zeros((rows + 2, 3))
    cells = [[i, i + 1, i + 2] for i in range(rows)]
    # A cell attribute that changes on every triangle would split the mesh into one object per triangle.
    _code("unsupported_profile", _unstructured(vertices, cells, cell_attrs=[("dip", "float32", np.arange(rows) * 1.5)]),
          match="runs")
    # Float values are cut by their int value, as in Unity: 0.1..0.9 is one run.
    validate_le(_unstructured(vertices, cells, cell_attrs=[("dip", "float32", np.linspace(0.1, 0.9, rows))]), family="mesh")
    # With a cell attribute present, every vertex attribute is split on too.
    _code("unsupported_profile", _unstructured(vertices, cells, cell_attrs=[("id", "int32", [1] * rows)],
                                               vertex_attrs=[("u", "float32", _alternating(rows + 2))]), match="'u'")
    # Without cell attributes Unity does not split at all.
    validate_le(_unstructured(vertices, cells, vertex_attrs=[("u", "float32", _alternating(rows + 2))]), family="mesh")


def test_mesh_runs_of_vertex_values_derived_from_cells():
    # Two cell runs, but the derived vertex attribute alternates: even vertices belong to the first
    # half of the cells, odd vertices to the second half.
    half = MAX_MESH_ATTRIBUTE_RUNS
    vertices = np.zeros((2 * half, 3))
    cells = [[2 * k] * 3 for k in range(half)] + [[2 * k + 1] * 3 for k in range(half)]
    attr = [0] * half + [1] * half
    _code("unsupported_profile", _unstructured(vertices, cells, cell_attrs=[("id", "int32", attr)]), match="'id'")
    ordered = [[k] * 3 for k in range(2 * half)]
    validate_le(_unstructured(vertices, ordered, cell_attrs=[("id", "int32", attr)]), family="mesh")


def test_textured_mesh():
    png = _png(2, 2, 3)
    uv = [("u", "float32", [0, 1, 1, 0.5]), ("v", "float64", [0, 0, 1, 1])]
    validate_le_dataset("mesh", {"mesh": _mesh(vertex_attrs=uv), "texture": png})
    validate_le_dataset("mesh", {"mesh": _mesh(vertex_attrs=[("u", "int64", [0, 1, 1, 0]), ("v", "int32", [0, 0, 1, 1])]),
                                 "texture": png})
    for attrs in ([], [("u", "float32", [0, 1, 1, 0])], [("u", "float32", [0, 1, np.nan, 0]), ("v", "float32", [0, 0, 1, 1])],
                  [("u", "bool", [0, 1, 1, 0]), ("v", "bool", [0, 0, 1, 1])]):
        with pytest.raises(LEValidationError) as error:
            validate_le_dataset("mesh", {"mesh": _mesh(vertex_attrs=attrs), "texture": png})
        assert error.value.code == "inconsistent_dataset"
    validate_le_dataset("mesh", {"mesh": _mesh()})

# endregion

# region Point clouds


def test_pointcloud_profile():
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    colors = [("red", "int64", [0, 255, 3, 4, 5]), ("green", "int32", [0] * 5), ("blue", "int64", [255] * 5)]
    validate_le(_unstructured(points, np.arange(5).reshape(5, 1), vertex_attrs=colors), family="pointcloud")
    validate_le(_unstructured(points, np.empty((5, 0))), family="pointcloud")
    validate_le(_unstructured(points, np.empty((0, 0))), family="pointcloud")
    _code("unsupported_profile", _unstructured(points, [[0], [2], [1], [3], [4]]), "pointcloud", match="once, in order")
    _code("unsupported_profile", _unstructured(points, np.arange(4).reshape(4, 1)), "pointcloud")
    _code("unsupported_profile", _unstructured(points, np.empty((3, 0))), "pointcloud")
    _code("unsupported_profile", _unstructured(points, np.empty((5, 0)), cell_attrs=[("a", "int32", [1] * 5)]), "pointcloud")
    _code("unsupported_profile", _unstructured(points, np.arange(5).reshape(5, 1),
                                               vertex_attrs=colors[:2] + [("blue", "int64", [256] * 5)]), "pointcloud",
          match="0..255")
    _code("unsupported_profile", _unstructured(np.zeros((0, 3)), np.zeros((0, 1))), "pointcloud", match="at least one")
    assert _code("family_mismatch", _mesh(), "pointcloud").detected_family == "mesh"

# endregion

# region Grids


def test_volume_and_raster_profiles():
    report = validate_le(_grid(), family="volume")
    assert report.data_shape == (3, 4, 2)
    validate_le(_grid((3, 4, 1), bounds={"x": [0, 2], "y": [0, 3], "z": [5, 5]}), family="volume")
    validate_le(_grid((3, 4), bounds={"x": [0, 2], "y": [0, 3]}, name="topography"), family="raster")
    nan = np.arange(12, dtype=np.float32).reshape(3, 4)
    nan[1, 1] = np.nan
    validate_le(_structured(nan, {"x": [0, 2], "y": [0, 3]}), family="raster")
    assert _code("family_mismatch", _grid((3, 4), bounds={"x": [0, 2], "y": [0, 3]}), "volume").detected_family == "raster"
    assert _code("family_mismatch", _grid(), "raster").detected_family == "volume"
    assert _code("family_mismatch", _mesh(), "volume").detected_family == "mesh"
    _code("unsupported_profile", _grid((3,), bounds={"dim0": [0, 2]}), "volume")
    _code("unsupported_profile", _grid((3,), bounds={"dim0": [0, 2]}), "raster")


@pytest.mark.parametrize("overrides, code", [
        ({"dtype": "float64"}, "unsupported_profile"),
        ({"dtype": "int32"}, "unsupported_profile"),
        ({"transform": [1, 0, 0]}, "unsupported_profile"),
        ({"data_shape": [3, 4, 2, 1]}, "malformed"),
        ({"data_shape": [3, 0, 2]}, "malformed"),
        ({"data_name": ""}, "malformed"),
        ({"data_name": "x"}, "unsupported_profile"),
        ({"xarray_attrs": {"units": "m"}}, "unsupported_profile"),
        ({"bounds": {"x": [0, 2], "y": [0, 3]}}, "malformed"),
        ({"bounds": {"x": [0, 2], "y": [3, 0], "z": [0, 1]}}, "malformed"),
        ({"bounds": {"x": [0, 2], "y": [0, 0], "z": [0, 1]}}, "malformed"),
        ({"bounds": {"x": [0, 2], "y": [0, 3], "z": [0, True]}}, "malformed"),
        ({"bounds": {"x": [0, 1e39], "y": [0, 3], "z": [0, 1]}}, "unsupported_profile"),
        ({"bounds": {"x": [0, 10 ** 400], "y": [0, 3], "z": [0, 1]}}, "unsupported_profile"),
])
def test_grid_header_rules(overrides, code):
    _code(code, _grid(**overrides), "volume")


def test_grid_values_and_length():
    _code("malformed", _grid() + b"\x00\x00\x00\x00", "volume")
    values = np.zeros((3, 4, 2), np.float32)
    values[2, 3, 1] = np.inf
    _code("unsupported_profile", _structured(values, GRID_BOUNDS), "volume", match="finite")
    singleton = {"x": [0, 2], "y": [0, 3], "z": [5, 6]}
    _code("malformed", _grid((3, 4, 1), bounds=singleton), "volume")


def test_colour_rasters():
    xy = {"x": [0, 2], "y": [0, 3]}
    rgb = np.full((3, 4, 3), 128, np.float32)
    rgb[0, 0, :] = np.nan
    report = validate_le(_structured(rgb, xy, name="rgb"), family="raster")
    assert report.channels == 3
    assert validate_le(_structured(np.zeros((3, 4, 4)), xy, name="rgba"), family="raster").channels == 4
    _code("unsupported_profile", _structured(rgb, xy, name="RGB"), "raster")
    _code("unsupported_profile", _structured(rgb, xy, name="rgba"), "raster")
    _code("unsupported_profile", _structured(np.zeros((3, 4, 4)), xy, name="rgb"), "raster")
    _code("unsupported_profile", _structured(np.full((3, 4, 3), 256), xy, name="rgb"), "raster", match="0..255")
    _code("unsupported_profile", _structured(np.full((3, 4, 3), -1), xy, name="rgb"), "raster", match="0..255")
    _code("malformed", _structured(rgb, {"x": [0, 2], "y": [0, 3], "z": [0, 1]}, name="rgb"), "raster")
    _code("unsupported_profile", _structured(np.zeros((3, 4)), xy, name="rgb"), "raster", match="reserved")
    assert _code("family_mismatch", _structured(rgb, xy, name="rgb"), "volume").detected_family == "raster"

# endregion

# region Wells


def _wells(counts=(3, 2), md=None, ids=None, cell_ids=True, mapper=None, cells=None, collars=None):
    """Polylines of ``counts`` vertices each, laid out as the subsurface trajectory writer does."""
    vertices, segments, well_ids, cell_well_ids = [], [], [], []
    start = 0
    for well, count in enumerate(counts):
        for k in range(count):
            vertices.append([well, 0, -k])
            well_ids.append(well)
        for k in range(count - 1):
            segments.append([start + k, start + k + 1])
            cell_well_ids.append(well)
        start += count
    depths = md if md is not None else [k for count in counts for k in range(count)]
    vertex_attrs = [("well_id", "int64", ids if ids is not None else well_ids), ("measured_depths", "float32", depths)]
    attrs = {"well_id_mapper": mapper if mapper is not None else {f"W{i}": i for i in range(len(counts))}}
    trajectory = _unstructured(vertices, cells if cells is not None else segments,
                               cell_attrs=[("well_id", "int64", cell_well_ids)] if cell_ids else (),
                               vertex_attrs=vertex_attrs, attrs=attrs)
    n = len(counts) if collars is None else collars
    collar_file = _unstructured(np.arange(n * 3).reshape(n, 3), np.arange(n).reshape(n, 1))
    return {"collars": collar_file, "trajectory": trajectory}


def test_wells_dataset():
    report = validate_le_dataset("wells", _wells())
    assert report.well_count == 2
    assert report.files["trajectory"].well_count == 2
    validate_le_dataset("wells", _wells(cell_ids=False))


@pytest.mark.parametrize("kwargs, code, match", [
        ({"mapper": {"a": 0, "b": 2}}, "inconsistent_dataset", "0..N-1"),
        ({"mapper": {"a": 0, "b": 0}}, "inconsistent_dataset", "unique"),
        ({"mapper": {"a": 0, "b": 1.0}}, "unsupported_profile", "integers"),
        ({"mapper": {"a": 0}}, "inconsistent_dataset", "polylines"),
        ({"ids": [0, 0, 1, 0, 1]}, "inconsistent_dataset", "well_id"),
        ({"md": [0, 2, 1, 0, 1]}, "inconsistent_dataset", "decrease"),
        ({"md": [0, 1, np.nan, 0, 1]}, "unsupported_profile", "finite"),
        ({"cells": [[0, 1], [1, 2], [2, 4]]}, "inconsistent_dataset", "consecutive"),
        ({"cells": [[0, 1], [2, 3], [3, 4]]}, "inconsistent_dataset", "well_id"),
        ({"cells": [[0, 1], [1, 2], [0, 1], [3, 4]], "cell_ids": False}, "inconsistent_dataset", "unbroken"),
        ({"collars": 3}, "inconsistent_dataset", "collars has 3 wells"),
])
def test_wells_consistency(kwargs, code, match):
    sources = _wells(**kwargs)
    with pytest.raises(LEValidationError, match=match) as error:
        validate_le_dataset("wells", sources)
    assert error.value.code == code


def test_wells_roles():
    sources = _wells()
    swapped = {"collars": sources["trajectory"], "trajectory": sources["collars"]}
    with pytest.raises(LEValidationError, match="check the collars and trajectory roles") as error:
        validate_le_dataset("wells", swapped)
    assert error.value.code == "inconsistent_dataset"
    for roles in ({"collars": sources["collars"]}, {**sources, "texture": b""}):
        with pytest.raises(LEValidationError) as error:
            validate_le_dataset("wells", roles)
        assert error.value.code == "inconsistent_dataset"
    no_mapper = _unstructured(QUAD, [[0, 1], [1, 2]], vertex_attrs=[("well_id", "int64", [0] * 4)])
    _code("unsupported_profile", no_mapper, "wells", "trajectory", match="well_id_mapper")
    _code("unsupported_profile", _unstructured(QUAD, [[0, 1], [1, 2], [2, 3]], attrs={"well_id_mapper": {"a": 0}}),
          "wells", "trajectory", match="well_id and measured_depths")
    _code("unsupported_profile", _unstructured(QUAD, [[1], [0], [2], [3]]), "wells", "collars", match="one point cell")
    assert _code("family_mismatch", _mesh(), "wells", "collars").detected_family == "mesh"
    _code("unsupported_profile", _unstructured([[0, 0, np.nan]], [[0]]), "wells", "collars")

# endregion

# region PNG


def _png(width, height, channels, *, depth=8, color=None, interlace=0, chunks=(), filter_byte=0, pixels=None, tail=b""):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(body, zlib.crc32(kind)))

    color = (2 if channels == 3 else 6) if color is None else color
    row = bytes([filter_byte]) + bytes(width * channels * depth // 8)
    raw = row * height if pixels is None else pixels
    body = b"".join(chunk(kind, data) for kind, data in chunks)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, interlace)) +
            chunk(b"IDAT", zlib.compress(raw)) + body + chunk(b"IEND", b"") + tail)


def test_png_textures():
    report = validate_png_texture(_png(3, 2, 4))
    assert (report.width, report.height, report.channels) == (3, 2, 4)
    validate_png_texture(_png(2, 2, 3, chunks=[(b"tEXt", b"Software\x00subsurface")]))


@pytest.mark.parametrize("data, code", [
        (b"GIF89a", "malformed"),
        (_png(2, 2, 3)[:-4] + b"\x00\x00\x00\x00", "malformed"),
        (_png(2, 2, 3, tail=b"\x00"), "malformed"),
        (_png(2, 2, 3)[:40], "malformed"),
        (_png(2, 2, 1, color=0), "unsupported_profile"),
        (_png(2, 2, 3, depth=16), "unsupported_profile"),
        (_png(2, 2, 3, interlace=1), "unsupported_profile"),
        (_png(2, 2, 3, chunks=[(b"pHYs", bytes(9))]), "unsupported_profile"),
        (_png(2, 2, 3, filter_byte=5), "malformed"),
        (_png(2, 2, 3, pixels=bytes(5)), "malformed"),
        (_png(5000, 1, 3), "unsupported_profile"),
])
def test_png_rejections(data, code):
    with pytest.raises(LEValidationError) as error:
        validate_png_texture(data)
    assert error.value.code == code


def test_png_image_data_must_be_contiguous():
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(body, zlib.crc32(kind)))

    data = zlib.compress(b"\x00" + bytes(6) + b"\x00" + bytes(6))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", data[:5]) + chunk(b"tEXt", b"a\x00b") + chunk(b"IDAT", data[5:]) + chunk(b"IEND", b""))
    with pytest.raises(LEValidationError, match="consecutive"):
        validate_png_texture(png)

# endregion
