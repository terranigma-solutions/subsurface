# Structured LiquidEarth Read-Back

`StructuredData.from_binary_le(path)` reads the existing regular axis-aligned
scalar format into the existing `StructuredData` container. It accepts a string
or `pathlib.Path`. Invalid or unsupported files raise `ValueError`; filesystem
errors are not masked.

```python
from pathlib import Path
from subsurface import StructuredData

grid = StructuredData.from_binary_le("volume.le")
Path("copy.le").write_bytes(grid.to_binary())
```

## Supported Layout

The four-byte little-endian unsigned prefix gives the UTF-8 JSON header length
(1 byte to 1 MiB). The current header has exactly `data_shape`, `bounds`,
`transform`, `dtype`, and `data_name`. There is no version or array-order field.
The payload is exactly one scalar array in Fortran order, with the first axis
varying fastest. Truncated payloads and trailing bytes are rejected before
allocating coordinates or reshaping values.

Supported shapes have one to three positive integer dimensions, including
singleton dimensions. Axis names match the current constructor defaults:
`dim0` for one dimension, `x,y` for two, and `x,y,z` for three. Bounds object
key order does not reorder the scalar axes. Custom axis names and auxiliary
coordinates cannot establish an unambiguous shape-to-axis mapping and are
rejected. Data names must be nonempty strings distinct from axis names.

Signed and unsigned integers of 8, 16, 32, or 64 bits and floats of 16, 32,
or 64 bits are supported. The reader retains the declared dtype for subsequent
writes. The writer casts to `StructuredData.dtype` (default `float32`), so it
does not promise original input precision. Explicit NumPy byte-order dtype
strings are respected. Unqualified dtype names use NumPy native byte order,
as does the current writer; files from a big-endian producer with an unqualified
dtype cannot be distinguished from little-endian exports. Bool, complex,
object, string, structured, and subarray dtypes are unsupported.

## Coordinate Convention

For each axis, the bounds object stores **coordinate minimum and maximum**.
The reader reconstructs a regular axis using `linspace(low, high, size)`, with
both stored endpoints included. Singleton axes require equal extrema and retain
one coordinate; nonsingleton axes require strictly increasing finite extrema.
Coordinates use float64; integer extrema that would lose precision and axes
whose reconstructed spacing collapses at that precision are rejected.

`StructuredData.from_pyvista()` reshapes cell data in Fortran order, and assigns
`linspace(outer_low, outer_high, cell_count, endpoint=False)` coordinates.
The resulting stored bounds are thus lower cell-edge coordinate extrema,
not cell centers and not the original outer VTK bounds. Do not apply another
half-cell offset or exclude the stored upper endpoint when reading.

The adjacent active `subsurface-le/subsurface_le/vtk_import/core.py::write_le`
uses the same convention: its stored upper bound is
`low + ((outer_high - low) / cell_count) * (cell_count - 1)`.
It writes the already-flat VTK cell array, whose x-fastest ordering matches
the reader's Fortran reshape; its flattening call does not reorder a 1D array.

## Limitations

Only the active scalar array, its name, shape, dtype, and coordinate extrema
are serialized. Other arrays, dataset/array attributes, CRS, units, nodata
semantics, original coordinate vectors, descending axis orientation, and
original outer bounds cannot be recovered. In particular, singleton cell axes
lose their original cell width. Bounds alone cannot verify that the source
coordinates were uniform or ascending. This reader performs no resampling.

Non-null transforms, including identity matrices, are rejected because the
current writer provides no transform convention. Flat six-number bounds emitted
by an explicit `bounds` override are also rejected: that override is independent
of coordinates and does not declare whether it means sample extrema or outer
edges. Supporting those files requires consumer confirmation, not an inferred
coordinate shift. No wire-format changes are made.

Use the public writer's default `order='F'`. Nondefault-order files carry no
marker and cannot be detected or safely interpreted automatically. Synthetic
tests cover the default public round trip and the VTK export formula, not
production qualification against representative consumer files. Inspection and
logical object counts are separate APIs; this reader does not equate voxels with
geological objects.
