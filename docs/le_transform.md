# Affine LE Transforms

Import `transform_le` and `transform_mesh` from `subsurface.api.le_transform`.
These operations support validated unstructured, prefixed LE files in default
Fortran geometry order, not structured grids, sidecars, projective transforms,
or CRS conversion.

```python
import numpy as np
from subsurface.api.le_transform import transform_le

matrix = np.eye(4)
matrix[:3, 3] = [10, 20, -5]
output = transform_le(
    "source.le", "translated.le", matrix,
    normal_columns={"point": [("nx", "ny", "nz")]},
    vector_columns={"cell": [("vx", "vy", "vz")]},
)
```

`transform_le(source, destination, matrix, *, normal_columns=None,
vector_columns=None, overwrite=False)` returns the absolute destination `Path`.
It leaves the source unchanged. Existing destinations raise `FileExistsError`
unless `overwrite=True`; aliases of the source, including hard links and
symlinks, always raise `ValueError`. The parent directory must exist. Shared
file-operation helpers preflight serialization and publish atomically, cleaning
temporary files on failure. Filesystem errors propagate.

`transform_mesh(mesh, matrix, *, normal_columns=None, vector_columns=None)`
accepts a `LiquidEarthMesh` and returns an independent mesh without mutating any
input arrays, DataFrames, or nested metadata. It uses the same serialization
preflight, so unsupported columns or metadata are rejected, not dropped.

## Geometry And Columns

The convention is homogeneous column vectors: `p_out = matrix @ p_in`.
For NumPy row arrays, `xyz_out = xyz @ matrix[:3, :3].T + matrix[:3, 3]`.
Composition applies the rightmost matrix first. Matrices must be real, finite,
4x4, invertible in their linear part, with last row exactly `[0, 0, 0, 1]`.
Singular geometry transforms are rejected rather than collapsing cells.
Matrices whose determinant cannot be resolved finitely in float64 are also
rejected rather than guessing orientation.

Both column arguments are mappings from `point` or `cell` to a sequence of
three-column tuples in XYZ order. Multiple triples may be specified per
association. Names must exist and each column can occur only once across both
arguments. There is no name guessing. Selected columns must be real numeric
(not boolean) and finite. Vectors use `A @ v`; normals use `inv(A).T @ n` and
are renormalized to unit length. Zero normals are rejected. Translation does
not affect either. Selected columns become floating-point computations in
memory, subject to the existing numeric writer rules on disk.

Negative-determinant transforms swap triangle indices 1 and 2 to retain outward
orientation consistent with transformed normals. Reflected tetrahedra and
hexahedra (connectivity widths 4 and 8) are rejected. Point and line connectivity
is unchanged. Positive-determinant transforms preserve all connectivity.

Calculations use float64. Nonfinite results and geometry/vector values beyond
float32 range raise `ValueError` before publication. Wire geometry is float32:
rounding and underflow are allowed, and large coordinates can lose small spatial
details. Inverse/composition round-trips therefore require float32 tolerances;
no topology repair or guarantee against float32-induced degeneration is made.
Unselected scalar, ID, boolean, and UV columns retain values and separate
dtypes in memory; on disk the existing writer retains integer/bool dtypes and
stores floating columns as int64 when exactly integral, otherwise float32.

## Metadata

Top-level non-null `data_attrs['transform']` is rejected because its application
semantics are not established. Absent or null transform metadata is preserved;
no hidden transform is composed or applied. Callers must resolve any external
coordinate-frame conventions themselves.

`data_attrs['bounds']` is added or replaced with
`[xmin, xmax, ymin, ymax, zmin, zmax]`, calculated from the output's serialized
float32 vertices, or `None` for empty geometry. All other metadata is deep-copied
unchanged, including the shared reserved provenance key `le_tools`. The tool
does not add its own provenance, rename coordinate systems/units, or reinterpret
unrelated metadata as geometry.
