"""Write the native .le ingestion fixtures with the public subsurface writers.

Every .le file here comes from a real writer (``UnstructuredData.to_binary`` /
``StructuredData.to_binary`` in Fortran order, and ``read_wells`` for wells),
so the validator is checked against what producers actually emit. The PNG
texture is encoded here byte by byte so its contents are fixed.

    python tests/data/le_ingest/generate.py           # rewrite files and manifest.json
    python tests/data/le_ingest/generate.py --check   # fail if a writer's output changed
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures import StructuredData, UnstructuredData
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase

HERE = Path(__file__).resolve().parent
BOREHOLE = HERE.parent / "borehole"

QUAD = np.array([[0.0, 0.0, 10.5], [4.0, 0.0, 11.0], [4.0, 3.0, 12.25], [0.0, 3.0, 11.5]])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _png(width, height, channels, pixels: bytes) -> bytes:
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(body, zlib.crc32(kind)))

    row = width * channels
    raw = b"".join(b"\x00" + pixels[y * row:(y + 1) * row] for y in range(height))
    color = 2 if channels == 3 else 6
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _mesh(**attrs):
    return UnstructuredData.from_array(QUAD, TRIANGLES, **attrs).to_binary(order="F")


def _grid(values, coords, name, dims=None):
    return StructuredData.from_numpy(values, coords=coords, data_array_name=name, dim_names=dims).to_binary("F")


def _wells():
    def reader(name):
        return GenericReaderFilesHelper(file_or_buffer=str(BOREHOLE / name))

    boreholes = read_wells(collars_reader=reader("supersimple_collars.csv"), surveys_reader=reader("supersimple_survey.csv"),
                           attrs_reader=reader("supersimple_lithology.csv"), is_lith_attr=True)
    return boreholes.collars.data.to_binary(), boreholes.combined_trajectory.data.to_binary()


def fixtures():
    """(relative path, family, role, writer, bytes) for every fixture."""
    rng = np.random.default_rng(20261106)
    collars, trajectory = _wells()
    raster_x, raster_y = np.linspace(500000.0, 500030.0, 4), np.linspace(4100000.0, 4100020.0, 3)
    color = rng.integers(0, 256, size=(4, 3, 4)).astype(np.float32)
    color[0, 0, :] = np.nan
    points = rng.normal(size=(6, 3)) * 10
    unstructured = "UnstructuredData.from_array(...).to_binary(order='F')"
    structured = "StructuredData.from_numpy(...).to_binary('F')"
    return [
            ("mesh/plain.le", "mesh", "mesh", unstructured, _mesh()),
            ("mesh/attributes.le", "mesh", "mesh", unstructured, _mesh(
                cells_attr=pd.DataFrame({"id": [7, 7]}),
                vertex_attr=pd.DataFrame({"elevation": QUAD[:, 2], "selected": [True, False, True, True]}))),
            ("mesh/textured.le", "mesh", "mesh", unstructured, _mesh(
                vertex_attr=pd.DataFrame({"u": [0.125, 0.875, 0.875, 0.125], "v": [0.25, 0.25, 0.75, 0.75]}))),
            ("mesh/textured.png", "mesh", "texture", "generate._png (RGB, 8-bit)",
             _png(2, 2, 3, bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]))),
            ("mesh/textured_integral_uv.le", "mesh", "mesh", unstructured + " (integral UVs are stored as int64)", _mesh(
                vertex_attr=pd.DataFrame({"u": [0.0, 1.0, 1.0, 0.0], "v": [0.0, 0.0, 1.0, 1.0]}))),
            ("mesh/textured_rgba.png", "mesh", "texture", "generate._png (RGBA, 8-bit)", _png(1, 1, 4, bytes([10, 20, 30, 128]))),
            ("pointcloud/points.le", "pointcloud", "pointcloud", unstructured + " with SpecialCellCase.POINTS",
             UnstructuredData.from_array(points, SpecialCellCase.POINTS, vertex_attr=pd.DataFrame({
                     "red"      : rng.integers(0, 256, 6), "green": rng.integers(0, 256, 6),
                     "blue"     : rng.integers(0, 256, 6), "intensity": rng.random(6)})).to_binary(order="F")),
            ("pointcloud/points_no_cells.le", "pointcloud", "pointcloud", unstructured + " with (n, 0) cells",
             UnstructuredData.from_array(points, np.empty((6, 0), dtype=np.int32)).to_binary(order="F")),
            ("volume/grid.le", "volume", "volume", structured, _grid(
                rng.random((3, 4, 2)).astype(np.float32),
                {"x": np.linspace(0, 20, 3), "y": np.linspace(0, 30, 4), "z": np.linspace(-50, -40, 2)}, "density")),
            ("volume/grid_singleton.le", "volume", "volume", structured + " (singleton z)", _grid(
                rng.random((3, 4, 1)).astype(np.float32),
                {"x": np.linspace(0, 20, 3), "y": np.linspace(0, 30, 4), "z": np.array([-45.0])}, "density")),
            ("raster/scalar.le", "raster", "raster", structured, _grid(
                np.where(np.arange(12).reshape(4, 3) == 5, np.nan, rng.random((4, 3)) * 100).astype(np.float32),
                {"x": raster_x, "y": raster_y}, "topography")),
            ("raster/rgb.le", "raster", "raster", structured + " (dims x, y, channel as in subsurface-le)", _grid(
                color[:, :, :3].copy(), {"x": raster_x, "y": raster_y}, "rgb", ["x", "y", "channel"])),
            ("raster/rgba.le", "raster", "raster", structured + " (dims x, y, channel as in subsurface-le)", _grid(
                color, {"x": raster_x, "y": raster_y}, "rgba", ["x", "y", "channel"])),
            ("wells/supersimple_collars.le", "wells", "collars", "read_wells(...).collars.data.to_binary()", collars),
            ("wells/supersimple_trajectory.le", "wells", "trajectory",
             "read_wells(...).combined_trajectory.data.to_binary()", trajectory),
    ]


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(check: bool) -> int:
    manifest_path = HERE / "manifest.json"
    entries = [{"path": path, "family": family, "role": role, "writer": writer,
                "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
               for path, family, role, writer, data in fixtures()]
    if check:
        recorded = json.loads(manifest_path.read_text())["fixtures"]
        changed = [e["path"] for e, r in zip(entries, recorded) if e != r] if len(entries) == len(recorded) else ["(list)"]
        on_disk = [e["path"] for e in recorded if hashlib.sha256((HERE / e["path"]).read_bytes()).hexdigest() != e["sha256"]]
        for path in changed + on_disk:
            print(f"changed: {path}", file=sys.stderr)
        return 1 if changed or on_disk else 0
    for (path, *_, data) in fixtures():
        target = HERE / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {
            "generator"       : "tests/data/le_ingest/generate.py",
            "generated_at_commit": _commit(),
            "versions"        : {"numpy": np.__version__, "pandas": pd.__version__, "xarray": xr.__version__},
            "fixtures"        : entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv[1:]))
