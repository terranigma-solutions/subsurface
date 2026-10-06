# Restricted Structured LE Transforms

Import the APIs directly from `subsurface.api.le_structured_transform`:

```python
import numpy as np
from subsurface.api.le_structured_transform import (
    transform_structured_grid, transform_structured_le,
)

matrix = np.eye(4)
matrix[0, 0] = 2.0
matrix[:3, 3] = [100.0, -20.0, 5.0]
output = transform_structured_le("source.le", "translated.le", matrix)
```

This example requires a rank-three grid. `output` is an absolute `pathlib.Path`.
`transform_structured_grid(grid, matrix)` instead returns an independent
`StructuredData`: scalar arrays and coordinates do not share memory with the
input. Neither API mutates its source or matrix.

## Transform Contract

The convention is homogeneous column vectors: `p_out = matrix @ p_in`.
The matrix must be finite, real, and 4x4, with exact last row `[0, 0, 0, 1]`.
Its linear part must be strictly positive diagonal: each coordinate is computed
in float64 as `scale * position + translation`. No rotation, shear, reflection,
zero/negative scale, or projective operation is accepted, even at tiny magnitude.
Rank-one `dim0` maps to X; rank-two `x,y` maps to X/Y; rank-three `x,y,z` maps to
X/Y/Z. Unused axes must have scale 1 and translation 0, not silently ignored
entries. Singleton axes are used axes and their sample position is transformed.

One numeric scalar array on standard regular ascending axes is supported.
The active-array name, exact `grid.dtype` string (including byte order), shape,
and scalar values are preserved; values are copied without arithmetic.
Integer/unsigned categorical labels are never interpolated or converted to
floating point. Scalar NaN, infinities, and signed zeros are preserved. Output
coordinates are float64. There is no new CRS, provenance, or nodata metadata.
Unsupported metadata, extra arrays, auxiliary coordinates, custom dimensions,
and explicit bounds overrides are rejected before any file write. See
[`le_grid_ops_safety.md`](le_grid_ops_safety.md) for the shared boundary.

## Coordinates And Precision

Bounds are inclusive sample extrema, not outer voxel edges. Every existing
sample position is transformed directly, rather than rebuilding the output from
its endpoints. The format reconstructs positions using endpoint-derived
`linspace`, so transformed positions must agree with that reconstruction within
`AXIS_SPACING_RTOL = 1e-10` times output endpoint spacing. There is no tolerance
floor and no tolerance relative to the world-coordinate origin. Singletons have
exact positions and no implied spacing.

Reader-generated axes at large origins can have rounded unequal positive
increments and are accepted when they meet this reconstruction contract.
Translation/scaling can make those rounded samples incompatible with the output
endpoint reconstruction. Such operations are rejected explicitly, not snapped or
resampled. Coordinate overflow, nonfinite span/spacing, and float64 rounding or
underflow that collapses adjacent samples also fail. A mathematically valid
transform therefore need not be representable by this format. Composition and
inverse operations are subject to float64 rounding and the same checks; exact
coordinate reversibility is not promised.

## File Safety And Limits

`transform_structured_le(source, destination, matrix, *, overwrite=False)` loads
and validates the source, then uses the shared atomic writer in Fortran order.
The destination parent must exist. Existing files and dangling symlinks fail by
default; `overwrite=True` atomically replaces the destination directory entry.
Path, symlink, resolved-path, and hardlink aliases to the source always fail,
even with overwrite enabled. A temporary is read back and checked before
publication, and cleaned up on success or failure. Sources remain unchanged.

Invalid transform shape, nonfinite values, unsupported operations or grids, and
unrepresentable coordinates raise `ValueError`; non-real matrix element types
or wrong container types raise `TypeError`. Filesystem errors propagate.
These APIs fully load/copy/serialize arrays without resource budgets. Original
metadata and nondefault ordering absent from the wire format cannot be recovered.
Shared atomic publication limits, including lack of directory fsync and exclusion
of hostile concurrent identity changes, apply. Tests use synthetic offline grids;
representative production exports and consumer qualification remain separate.
