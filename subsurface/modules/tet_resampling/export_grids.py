"""Resample FEM results onto a regular grid and write them as a LiquidEarth grid dataset.

    python -m subsurface.modules.tet_resampling.export_grids results.xdmf --spacing 0.05 --start 2026-11-19T00:00:00Z --out /tmp/dataset

A dataset is one grid with every attribute, in grid format 2 (like borehole
files, which keep every attribute as a typed column): a scalar is a column of
shape (nx, ny, nz), a 2D or 3D vector one column of shape (nx, ny, nz, 2|3).
LiquidEarth shows each attribute as its own volume in the dataset's group, and
a vector as one volume with arrows and particles. Tensor fields are skipped.

With one time step the output is volumes/<name>.le. With several it is a time
series as subsurface-le publishes one: a frame file per step under
time-series/<name>/<revision>/, and volumes/<name>.le, a copy of the last frame
whose header lists every frame. Frame times are --start plus the result's
times in seconds, so the series plays on the shared timeline.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import fem_results
from . import tet_resampler

FORMAT_VERSION = 2
TIME_SERIES_SCHEMA = 2


@dataclass
class Column:
    """One attribute of a grid: float32 (nx, ny, nz) or (nx, ny, nz, components) per time step."""

    name: str
    steps: list[np.ndarray]
    kind: str | None = None

    @property
    def components(self) -> int:
        return self.steps[0].shape[3] if self.steps[0].ndim == 4 else 1


@dataclass
class Dataset:
    grid: tet_resampler.Grid
    columns: list[Column]
    times: np.ndarray
    units: dict[str, str] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return len(self.times)


def dataset_from_results(results: fem_results.FemResults, sampler: tet_resampler.TetSampler,
                         include: set[str] | None = None, step: int | None = None,
                         kinds: dict[str, str] | None = None) -> Dataset:
    """Samples every scalar and vector field onto the sampler's grid; NaN outside the mesh.

    Static fields repeat in every step, since every frame of a series has the same columns. With ``step``, only
    that time step is kept.
    """
    times = results.times if step is None else results.times[step:step + 1 or None]
    columns = []
    for center, fields in (("point", results.point_data), ("cell", results.cell_data)):
        sample = sampler.point_data if center == "point" else sampler.cell_data
        for name, f in fields.items():
            if include is not None and name not in include:
                continue
            n_comp = int(np.prod(f.components)) if f.components else 1
            if f.components and n_comp not in (2, 3):
                print(f"  skipped {name}: {n_comp} components (only scalars and 2D/3D vectors)")
                continue
            if f.static:
                frames = [sample(f.values[0])] * len(times)
            else:
                values = f.values if step is None else f.values[step:step + 1 or None]
                frames = [sample(v) for v in values]
            columns.append(Column(name, [np.asarray(v, np.float32) for v in frames], (kinds or {}).get(name)))
    return Dataset(sampler.grid, columns, np.asarray(times, float))


def grid_bytes(grid: tet_resampler.Grid, columns: list[tuple[Column, np.ndarray]], xarray_attrs: dict) -> bytes:
    """One grid format 2 file: a little-endian header length, the JSON header, then each column x fastest."""
    attrs, body = [], []
    for column, values in columns:
        data = np.asarray(values, np.float32).tobytes(order="F")
        meta = {"name": column.name, "dtype": "float32", "shape": list(values.shape), "byte_length": len(data)}
        if column.kind:
            meta["kind"] = column.kind
        attrs.append(meta)
        body.append(data)
    header = {
        "format_version": FORMAT_VERSION,
        "data_shape": list(grid.shape),
        "bounds": {axis: [float(lo), float(hi)] for axis, (lo, hi) in zip("xyz", grid.bounds)},
        "transform": None,
        "dtype": "float32",
        "grid_attrs": attrs,
        "xarray_attrs": xarray_attrs,
    }
    header_bytes = json.dumps(header, allow_nan=False).encode("utf-8")
    return len(header_bytes).to_bytes(4, "little") + header_bytes + b"".join(body)


def write_dataset(dataset: Dataset, name: str, out: Path, start: datetime | None, revision: str = "1") -> list[Path]:
    """Writes volumes/<name>.le and, for several steps, its frames; returns the files written."""
    base_attrs = {"attribute_units": dataset.units} if dataset.units else {}
    if dataset.steps == 1:
        frames = [grid_bytes(dataset.grid, [(c, c.steps[0]) for c in dataset.columns], base_attrs)]
        master_path = out / "volumes" / f"{name}.le"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(frames[0])
        return [master_path]

    if start is None:
        raise ValueError("a dataset with several time steps needs a start time (--start)")
    directory = f"time-series/{name}/{revision}"
    manifest = {
        "schema_version": TIME_SERIES_SCHEMA, "role": "master", "time_series_id": name, "revision": revision,
        "kind": "volume", "directory": directory, "index": "series.json",
        "frames": [{"timestamp": (start + timedelta(seconds=float(t))).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "path": f"volume_{i:04d}.le"} for i, t in enumerate(dataset.times)],
    }
    written = []
    frame_dir = out / directory
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(dataset.steps):
        path = frame_dir / manifest["frames"][i]["path"]
        path.write_bytes(grid_bytes(dataset.grid, [(c, c.steps[i]) for c in dataset.columns], base_attrs))
        written.append(path)
    (frame_dir / "series.json").write_text(json.dumps(manifest, indent=1))
    master = grid_bytes(dataset.grid, [(c, c.steps[-1]) for c in dataset.columns], {**base_attrs, "time_series": manifest})
    master_path = out / "volumes" / f"{name}.le"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(master)
    return [master_path] + written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help=".xdmf, .vtk.series, .vtk or .vtu")
    parser.add_argument("--spacing", type=float, help="voxel size in model units (default: a quarter of the mean edge)")
    parser.add_argument("--attributes", help="comma-separated fields to keep (default: all scalars and vectors)")
    parser.add_argument("--step", type=int, help="keep only this time step (negative counts from the end)")
    parser.add_argument("--name", help="dataset name (default: the input file name)")
    parser.add_argument("--start", help="time of t = 0 as ISO 8601, for example 2026-11-19T00:00:00Z")
    parser.add_argument("--unit", action="append", default=[], metavar="FIELD=UNIT", help="unit of a field (repeatable)")
    parser.add_argument("--kind", action="append", default=[], metavar="FIELD=KIND",
                        help="what a vector field is, for example displacement or velocity (repeatable)")
    parser.add_argument("--out", type=Path, help="write the dataset into this folder, laid out like a space's container")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    results = fem_results.read(args.input)
    t_read = time.perf_counter()
    spacing = args.spacing or tet_resampler.suggest_spacing(results.points, results.tets)
    sampler = tet_resampler.build_sampler(results.points, results.tets, spacing)
    t_build = time.perf_counter()
    include = set(args.attributes.split(",")) if args.attributes else None
    kinds = dict(item.split("=", 1) for item in args.kind)
    dataset = dataset_from_results(results, sampler, include, args.step, kinds)
    dataset.units = dict(item.split("=", 1) for item in args.unit)
    t_sample = time.perf_counter()

    grid = sampler.grid
    print(f"{args.input.name}: {len(results.points):,} nodes, {len(results.tets):,} tetrahedra, "
          f"{len(results.times)} time steps")
    print(f"grid {grid.shape[0]} x {grid.shape[1]} x {grid.shape[2]}, spacing "
          f"{' x '.join(f'{s:.4g}' for s in grid.spacing)}, {len(sampler.voxels):,} of {grid.size:,} voxels inside")
    print(f"attributes: {', '.join(f'{c.name} ({c.components})' if c.components > 1 else c.name for c in dataset.columns)}"
          f"; {dataset.steps} step(s)")
    print(f"read {t_read - started:.3f} s · locate {t_build - t_read:.3f} s · sample {t_sample - t_build:.3f} s")

    if args.out:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc) if args.start else None
        name = args.name or args.input.name.split(".")[0]
        written = write_dataset(dataset, name, args.out, start)
        print(f"wrote {len(written)} file(s), {sum(p.stat().st_size for p in written) / 1e6:.1f} MB, to {args.out}")


if __name__ == "__main__":
    main()
