# Validating `.le` files for LiquidEarth ingestion

`subsurface.api.le_validation` checks complete `.le` files before LiquidEarth
publishes them as space content. Unlike `inspect_le`, it reads the whole payload.
It checks the binary layout, every value the layout constrains, and what the
LiquidEarth (Unity) loaders can read. It never changes or re-serializes the
bytes: a file that passes is published exactly as it is.

```python
from subsurface.api import validate_le, validate_le_dataset, validate_png_texture, LEValidationError

validate_le("terrain.le", family="mesh")
validate_le_dataset("mesh", {"mesh": "terrain.le", "texture": "terrain.png"})
validate_le_dataset("wells", {"collars": "w_collars.le", "trajectory": "w_trajectory.le"})
```

Sources are paths (read through a read-only memory map, in chunks) or bytes.
Reports hold counts, shapes and attribute names, never values.

## Families and roles

| Family | Roles | File |
|---|---|---|
| `mesh` | `mesh`, optional `texture` | unstructured triangles; the texture is a PNG |
| `volume` | `volume` | structured `[nx, ny, nz]` |
| `raster` | `raster` | structured `[nx, ny]`, or colour `[nx, ny, 3 or 4]` |
| `pointcloud` | `pointcloud` | unstructured points |
| `wells` | `collars`, `trajectory` | unstructured points and polylines |

## Encoding

Array order is not stored in a `.le` file. The only accepted encoding is
`le-fortran-little-endian-v1` (`LE_INGEST_ENCODING`): Fortran (column-major)
arrays in little-endian byte order, which is what `UnstructuredData.to_binary()`
and `StructuredData.to_binary()` write by default. The producer declares it;
plausible values cannot prove it. `LiquidEarthMesh.to_binary()` defaults to C
order and must not be used for ingestion.

## Error codes

`LEValidationError.code` is one of:

| Code | Meaning |
|---|---|
| `malformed` | not a valid `.le` file or PNG (truncated, bad JSON, wrong lengths, trailing bytes, indices out of range) |
| `unsupported_profile` | a valid file outside what LiquidEarth accepts |
| `family_mismatch` | a valid dataset of another family; `detected_family` names it |
| `inconsistent_dataset` | files or attributes that do not agree, or a missing or unknown role |

## Rules for every `.le` file

- The header is at most 1 MiB, nested at most 32 levels, strict UTF-8 JSON, with
  no duplicate keys and no `NaN`, `Infinity` or overflowing numbers.
- The payload has exactly the declared length.
- Time-series files (`time_series_id`, `timestamp` or `time_series` metadata) are
  rejected: snapshots and masters need their own lifecycle.

## Unstructured files (meshes, point clouds, wells)

- `format_version` must be 2. Legacy v1 files are rejected.
- The header has exactly `format_version`, `vertex_shape`, `cell_shape`,
  `cell_attrs`, `vertex_attrs` and `xarray_attrs`. There are no `lods`.
- Vertices are finite.
- Connectivity indices are in range.
- Each attribute has exactly `name`, `dtype`, `shape` and `byte_length`:
  - names are non-empty strings;
  - `dtype` is `float32`, `float64`, `int32`, `int64` or `bool`, the spellings Unity reads;
  - bool bytes are 0 or 1;
  - floats may be NaN but not infinite, and `float64` values fit in float32;
  - `int64` values fit in int32, because Unity casts them to `int`;
  - at most 256 attributes per association.
- `xarray_attrs` must not mark master/slave resources, and its `name` is a string.

| Role | Accepted |
|---|---|
| `mesh` | Cell width 3, at least one triangle. If the mesh has any cell attribute, every cell and vertex attribute changes its `(int)` value at most 1024 times along the mesh. This applies to vertex attributes derived from cell attributes too, because Unity splits such meshes into one object per run. |
| `texture` | A complete non-interlaced 8-bit RGB or RGBA PNG: at most 16 MiB, 4096 pixels a side and 4 Mpx. Only `IHDR`, `IDAT`, `tEXt` and `IEND` chunks, with valid CRCs. The mesh must have finite, non-bool `u` and `v` vertex attributes. |
| `pointcloud` | Width 1 with cell *i* = point *i*, or width 0 with no cell rows (or one per point) and no cell attributes. If integer `red`, `green` and `blue` are present, they are 0..255. |
| `collars` | Width 1 with cell *i* = point *i*, at least one well. |
| `trajectory` | Width 2. `xarray_attrs.well_id_mapper` maps names to the integers 0..N-1. Numeric vertex attributes `well_id` and finite `measured_depths`. The segments form N polylines in order; polyline *r* covers its vertices consecutively, all with `well_id` *r* (and cell `well_id` *r* if present), and its measured depths never decrease. |

A wells dataset also needs as many collars as the trajectory has wells.

## Structured files (volumes, rasters)

- The header has exactly `data_shape`, `bounds`, `transform` (null), `dtype` and `data_name`.
- `dtype` is `float32`.
- `bounds` maps each spatial axis to sample extrema `[min, max]` that fit in float32. A singleton axis has `min == max`; other axes have `min < max`.
- Values are finite or NaN.

| Role | Accepted |
|---|---|
| `volume` | `[nx, ny, nz]` with bounds `x`, `y`, `z` |
| `raster` | `[nx, ny]` with bounds `x`, `y`; or a colour raster `[nx, ny, 3]` named `rgb` or `[nx, ny, 4]` named `rgba`, with bounds `x`, `y` and values 0..255 or NaN |

`rgb` and `rgba` are reserved for colour rasters. One-dimensional grids are rejected.

## Fixtures

`tests/data/le_ingest/` holds a file for each profile, written by the public
writers (`generate.py`, with `manifest.json` recording hashes and versions).
Run `python tests/data/le_ingest/generate.py --check` to see whether a writer's
output has changed.
