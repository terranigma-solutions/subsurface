"""Native .le ingestion fixtures written by the public subsurface writers (tests/data/le_ingest)."""

import hashlib
import importlib.util
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from subsurface.api import validate_le, validate_le_dataset, validate_png_texture
from subsurface.core.structs.base_structures import StructuredData
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh

DATA = Path(__file__).resolve().parents[1] / "data" / "le_ingest"
MANIFEST = json.loads((DATA / "manifest.json").read_text())["fixtures"]
LE_FILES = [entry for entry in MANIFEST if entry["role"] != "texture"]


def _validate(entry, source):
    if entry["role"] == "texture":
        return validate_png_texture(source)
    return validate_le(source, family=entry["family"], role=entry["role"])


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda entry: entry["path"])
def test_fixture_matches_manifest_and_validates(entry):
    data = (DATA / entry["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert len(data) == entry["size"]
    # Reading through a memory map and from bytes give the same report, and change nothing.
    copy = bytearray(data)
    assert _validate(entry, DATA / entry["path"]) == _validate(entry, copy)
    assert bytes(copy) == data
    assert hashlib.sha256((DATA / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("entry", [e for e in LE_FILES if e["family"] in ("mesh", "pointcloud", "wells")],
                         ids=lambda entry: entry["path"])
def test_unstructured_fixtures_decode_with_the_full_reader(entry):
    data = (DATA / entry["path"]).read_bytes()
    mesh = LiquidEarthMesh.from_binary(data, order="F")
    report = _validate(entry, data)
    assert mesh.vertex.shape[0] == report.vertex_count
    assert mesh.cells.shape == (report.cell_count, report.cell_width)


@pytest.mark.parametrize("entry", [e for e in LE_FILES if e["family"] in ("volume", "raster")
                                   and not e["path"].endswith(("rgb.le", "rgba.le"))], ids=lambda entry: entry["path"])
def test_scalar_grid_fixtures_decode_with_the_full_reader(entry):
    grid = StructuredData.from_binary_le(DATA / entry["path"])
    assert tuple(grid.shape) == _validate(entry, DATA / entry["path"]).data_shape


def test_datasets():
    mesh = DATA / "mesh"
    for name in ("textured.le", "textured_integral_uv.le"):
        for texture in ("textured.png", "textured_rgba.png"):
            report = validate_le_dataset("mesh", {"mesh": mesh / name, "texture": mesh / texture})
            assert set(report.files) == {"mesh", "texture"}
    wells = validate_le_dataset("wells", {"collars": DATA / "wells" / "supersimple_collars.le",
                                          "trajectory": DATA / "wells" / "supersimple_trajectory.le"})
    assert wells.well_count == 2


@pytest.mark.skipif(importlib.util.find_spec("wellpathpy") is None, reason="read_wells needs the wells requirements")
def test_generator_is_deterministic():
    spec = importlib.util.spec_from_file_location("le_ingest_generate", DATA / "generate.py")
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    produced = {path: hashlib.sha256(data).hexdigest() for path, *_, data in generate.fixtures()}
    assert produced == {entry["path"]: entry["sha256"] for entry in MANIFEST}


def test_validation_streams_large_files(tmp_path):
    shape = (256, 256, 256)  # 64 MiB of float32 samples
    header = json.dumps({"data_shape": list(shape), "bounds": {"x": [0, 1], "y": [0, 1], "z": [0, 1]},
                         "transform": None, "dtype": "float32", "data_name": "density"}).encode()
    path = tmp_path / "large.le"
    with open(path, "wb") as stream:
        stream.write(len(header).to_bytes(4, "little") + header)
        stream.write(np.zeros(shape[0] * shape[1], np.float32).tobytes() * shape[2])
    tracemalloc.start()
    try:
        validate_le(path, family="volume")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024
