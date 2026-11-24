# Tetrahedral resampling

Resamples simulation results on linear tetrahedra onto a regular grid and
writes them as a LiquidEarth grid dataset. LiquidEarth volumes are regular
grids, while finite-element results are unstructured meshes with several
attributes and time steps.

Moved from the LiquidEarth client repository, where it was a prototype under
`tools/vector-resampling`.

## Status in subsurface

This package is a relocation, not an integration yet:

- `export_grids` writes grid format 2 with its own writer. Neither
  `StructuredData.from_binary_le` nor `subsurface.api.le_validation` accepts
  format 2 yet, so subsurface cannot read back or validate these files.
- `write_dataset` writes the container layout and the master `.le` file
  (schema 2) itself. In production that layout is published by subsurface-le.
- The readers return their own `FemResults`, not `UnstructuredData` /
  `TetraMesh`, and the resampler returns NumPy arrays, not `StructuredData`.

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

A dataset is one grid with every attribute, in grid format 2. Like borehole
files, which keep every attribute as a typed column, the header lists the
columns and the body holds them one after another:

```json
{"format_version": 2, "data_shape": [104, 24, 68], "bounds": {"x": [-2.6, 2.6], "y": [-0.6, 0.6], "z": [0, 3.4]},
 "transform": null, "dtype": "float32",
 "grid_attrs": [{"name": "temperature", "dtype": "float32", "shape": [104, 24, 68], "byte_length": 678912},
                {"name": "displacement", "dtype": "float32", "shape": [104, 24, 68, 3], "byte_length": 2036736, "kind": "displacement"}],
 "xarray_attrs": {"attribute_units": {"displacement": "m"}}}
```

- A scalar is a column of shape (nx, ny, nz); a 2D or 3D vector is one column
  of shape (nx, ny, nz, 2 or 3). Tensor fields are skipped.
- Values are float32, x fastest, then y, z and the vector component. NaN is
  no data.
- `kind` is optional free text for vectors; `velocity` marks real motion.
- The grid is cell-centred: the voxels tile `bounds`.

With one time step (or `--step`) the output is `volumes/<name>.le`. With
several, it is a time series laid out as subsurface-le publishes one, so the
dataset plays on the shared timeline:

- `time-series/<name>/1/volume_NNNN.le`: one grid per step, every attribute in
  each. Static fields, such as `region`, repeat in every step.
- `time-series/<name>/1/series.json`: the manifest.
- `volumes/<name>.le`: a copy of the last step whose `xarray_attrs.time_series`
  lists every frame (schema 2). Frame times are `--start` plus the result's
  times in seconds.

LiquidEarth shows each attribute as its own volume in the dataset's group,
and a vector as one volume with arrows and particles.

Neither the LiquidEarth SDK nor the CLI's `ops` service accepts grid format 2
yet, so the tool writes files only.

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
- Clients before grid format 2 reject these files.

## Tests

```sh
pytest tests/test_tet_resampling
```

The tests use synthetic meshes. They check exact reproduction of linear
fields, no data outside a non-convex mesh, cell data, agreement of the
Numba and NumPy paths, the grid format 2 layout and the series layout, and
that the XDMF and VTK readers return the same results. Tests that need h5py
or meshio are skipped when those are not installed.
