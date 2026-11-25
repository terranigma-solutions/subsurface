# Tetrahedral resampling

Resamples simulation results on linear tetrahedra onto a regular grid and
writes them as LiquidEarth volumes. LiquidEarth volumes are regular
grids, while finite-element results are unstructured meshes with several
attributes and time steps.

Moved from the LiquidEarth client repository, where it was a prototype under
`tools/vector-resampling`.

## Status in subsurface

- `export_grids` builds a `StructuredData` per time step and writes it with
  subsurface's writers: `StructuredData.to_binary(format_version=2)`,
  `export_time_series` and `time_series_master`. `StructuredData.from_binary_le`
  reads the files back, and `validate_le` accepts a single-step file
  (time-series files are refused there by design).
- `write_dataset` writes the container layout locally. Publishing it is
  subsurface-le's job, which does not yet write format 2.
- The readers return their own `FemResults`, not `UnstructuredData` /
  `TetraMesh`, and the resampler returns NumPy arrays.

## Setup

```sh
pip install -e .
pip install -r requirements/requirements_fem.txt
```

Numba is optional. Without it, NumPy runs the same arithmetic more slowly.
h5py and meshio are needed only by the XDMF and VTK readers.

## Run

```sh
# Resample and report timings only
python -m subsurface.modules.tet_resampling.export_grids results.xdmf --spacing 0.05

# Write every attribute and time step
python -m subsurface.modules.tet_resampling.export_grids results.xdmf --spacing 0.05 \
    --start 2026-11-19T00:00:00Z --unit displacement=m --kind displacement=displacement --out /tmp/dataset
```

Inputs: XDMF 3 with HDF5 data (`.xdmf`), a ParaView VTK series
(`.vtk.series`), or one `.vtk` or `.vtu` file. Without `--spacing`, the
voxel size is a quarter of the mean edge length.

## Output

Each time step is one `StructuredData` with every field as a data variable,
written in structured `.le` format 2 (`docs/le_spec.md`, "Version 2"):

- The coordinates are the voxel centres. A scalar has dims `(x, y, z)`; a 2D or
  3D vector `(x, y, z, component)`, with the components labelled `x, y(, z)` in
  the grid's axes. Tensor fields are skipped, and 2D and 3D vectors cannot
  share a dataset.
- Values are float32 (integer fields too), x fastest, then y, z and the vector
  component. NaN is no data.
- `--unit` sets a variable's `units` attr, `--kind` its `kind`: optional free
  text, where `velocity` marks real motion.
- `bounds` are the outer voxel centres; the voxels tile the box half a spacing
  beyond them.

With one time step (or `--step`) the output is `volumes/<name>.le`. With
several, it is a time series laid out as subsurface-le publishes one, so the
dataset plays on the shared timeline:

- `time-series/<name>/<revision>/volume_NNNN.le`: one file per step, every
  field in each (`export_time_series`). Static fields, such as `region`, repeat
  in every step. The revision is a hash of the frames' times and contents, so
  writing the same dataset again reuses the folder.
- `time-series/<name>/<revision>/series.json`: the index.
- `volumes/<name>.le`: the master (`time_series_master`), the last step whose
  `xarray_attrs.time_series` lists every frame (schema 2). Frame times are
  `--start` plus the result's times in seconds.

LiquidEarth shows each field as its own volume in the dataset's group,
and a vector as one volume with arrows and particles.

The tool writes files only. A single-step volume passes `validate_le`'s
`volume` profile, which the LiquidEarth ingest uses. Time series are not
ingested through `validate_le`, which rejects time-series attrs.

## Method

`tet_resampler.build_sampler` finds the tetrahedron that contains each voxel
centre, and the barycentric weights of the centre. It runs once per mesh.
`TetSampler.point_data` and `cell_data` then apply the weights to any
number of fields and time steps.

Point location rasterises each tetrahedron row by row. On a row of voxels,
each barycentric coordinate is linear in x, so the voxels inside the
tetrahedron form one interval. The interval is solved directly, so no
voxel of the bounding box is tested one by one. With Numba, z-slabs of the
grid run in parallel. Without Numba, NumPy runs the same arithmetic in
vectorised chunks. Both paths give the same result on every run: a voxel
centre on a face shared by two tetrahedra goes to the one with the higher
index.

Linear interpolation in a linear tetrahedron is the solver's own field, so
point data has no resampling error. On a linear test field the error is
below 1e-6 (float32 precision).

## Performance

Apple M1 Max, 10 cores, Numba 0.68.1. Best of three runs.

| Case | Locate | Sample |
|---|---|---|
| Transient model, 1,105 tetrahedra, 5 cm grid (104 × 24 × 68) | 1 ms warm, 0.1 s with Numba start-up | 8 volumes × 5 steps: 15 ms |
| Synthetic, 1,296,000 tetrahedra, 12,000,000 voxels | 0.16 s | 10 vector frames: 0.34 s |
| Same, NumPy without Numba | 5.9 s | |
| Same, VTK `pyvista.sample` | 6.2 s | |

The first call in a process loads the cached Numba kernels and starts its
threads, which takes about 0.1 s. The first run after an install also
compiles them, which takes 1 to 2 s. A long-running process, such as a
simulation connector that pushes each new result, pays this once.

`benchmark.py` repeats the synthetic case. With PyVista installed, it also
times VTK.

```sh
python -m subsurface.modules.tet_resampling.benchmark --cells 60 --spacing 0.01
```

## Agreement with VTK

On the transient model above, values agree with `pyvista.sample` to within
9 µm (largest displacement 6.4 cm). VTK marks a few more voxels as inside
(41 of 42,335 at 5 cm). All of these centres lie 0.7 µm to 0.4 mm outside the
mesh surface. VTK accepts them because of its default tolerance; this tool
sets them to no data.

## Limitations

- Linear tetrahedra only. Other cell types in the file are ignored.
- The mesh must be the same in every time step.
- LiquidEarth clients without format 2 support reject these files.

## Tests

```sh
pytest tests/test_tet_resampling
```

The tests use synthetic meshes. They check exact reproduction of linear
fields, no data outside a non-convex mesh, cell data, agreement of the
Numba and NumPy paths, the format 2 datasets and the series layout, and
that the XDMF and VTK readers return the same results. Tests that need h5py
or meshio are skipped when those are not installed.
