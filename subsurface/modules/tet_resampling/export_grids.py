"""Resample FEM results onto a regular grid and write them as LiquidEarth volumes.

    python -m subsurface.modules.tet_resampling.export_grids results.xdmf --spacing 0.05 --start 2026-11-19T00:00:00Z --out /tmp/dataset

Each time step is a StructuredData with every field as a data variable, on the
voxel centres: a scalar on (x, y, z), a 2D or 3D vector on (x, y, z, component)
with the components labelled x, y(, z). It is written in structured .le format
2 (see docs/le_spec.md), so one file holds every field. LiquidEarth shows each
field as its own volume in the file's group, and a vector as one volume with
arrows and particles. Tensor fields are skipped.

With one time step the output is volumes/<name>.le. With several it is a time
series laid out as subsurface-le publishes one: a frame file per step under
time-series/<name>/<revision>/ with its series.json, and volumes/<name>.le, the
master (time_series_master). Frame times are --start plus the result's times
in seconds, so the series plays on the shared timeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from . import fem_results
from . import tet_resampler
from ...core.structs.base_structures import StructuredData
from ..writer.time_series import INDEX_NAME, export_time_series, time_series_master

FORMAT_VERSION = 2
AXES = ("x", "y", "z")


@dataclass
class Column:
    """One field on the grid: float32 (nx, ny, nz) or (nx, ny, nz, components) per time step."""

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

    def structured(self, step: int) -> StructuredData:
        """Time step ``step`` as a StructuredData: every field a variable, the first one active."""
        sizes = {c.components for c in self.columns if c.components > 1}
        if len(sizes) > 1:
            raise ValueError("2D and 3D vectors cannot share a dataset: they would need two component dimensions")
        data_vars = {}
        for column in self.columns:
            attrs = {}
            if column.name in self.units:
                attrs["units"] = self.units[column.name]
            if column.kind:
                attrs["kind"] = column.kind
            dims = AXES + ("component",) if column.components > 1 else AXES
            data_vars[column.name] = (dims, np.asarray(column.steps[step], np.float32), attrs)
        coords = {axis: self.grid.axis(a) for a, axis in enumerate(AXES)}
        if sizes:
            coords["component"] = list(AXES[:sizes.pop()])
        return StructuredData(xr.Dataset(data_vars, coords=coords), self.columns[0].name)


def dataset_from_results(results: fem_results.FemResults, sampler: tet_resampler.TetSampler,
                         include: set[str] | None = None, step: int | None = None,
                         kinds: dict[str, str] | None = None) -> Dataset:
    """Samples every scalar and vector field onto the sampler's grid; NaN outside the mesh.

    Static fields repeat in every step, since every frame of a series has the same variables. With ``step``, only
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


def write_dataset(dataset: Dataset, name: str, out: Path, start: datetime | None) -> list[Path]:
    """Writes volumes/<name>.le and, for several steps, its frames; returns the files written.

    The frames' folder is named after a hash of their times and contents, so writing the same dataset again reuses
    it, and a changed dataset gets a new folder, as subsurface-le publishes revisions.
    """
    master_path = out / "volumes" / f"{name}.le"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset.steps == 1:
        master_path.write_bytes(dataset.structured(0).to_binary(format_version=FORMAT_VERSION))
        return [master_path]

    if start is None:
        raise ValueError("a dataset with several time steps needs a start time (--start)")
    stamps = [start + timedelta(seconds=float(t)) for t in dataset.times]
    digest = hashlib.sha256()
    for i, stamp in enumerate(stamps):
        digest.update(stamp.isoformat().encode())
        digest.update(dataset.structured(i).to_binary(format_version=FORMAT_VERSION))
    revision = digest.hexdigest()[:16]
    directory = f"time-series/{name}/{revision}"
    frame_dir = out / directory
    index_path = frame_dir / INDEX_NAME
    if not index_path.exists():  # * The index is written last, so an existing one means a complete folder
        export_time_series(((stamp, dataset.structured(i)) for i, stamp in enumerate(stamps)), frame_dir,
                           time_series_id=name, kind="volume", format_version=FORMAT_VERSION)
    index = json.loads(index_path.read_text())
    master_path.write_bytes(time_series_master(dataset.structured(dataset.steps - 1), index, directory=directory,
                                               revision=revision, format_version=FORMAT_VERSION))
    return [master_path, index_path] + [frame_dir / frame["path"] for frame in index["frames"]]


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
