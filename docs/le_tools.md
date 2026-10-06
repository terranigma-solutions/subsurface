# LiquidEarth File Tools

The supported library pipeline uses one unstructured dataset per `.le` file.
Logical objects are explicit numeric groups, not datasets, connected components,
primitives, or voxels. The tools do not introduce an archive/container format.

```python
from pathlib import Path
import numpy as np
from subsurface import inspect_le, transform_le, split_le, merge_le

summary = inspect_le("source.le", object_attribute="object_id", association="cell")
matrix = np.eye(4)
matrix[:3, 3] = [10, 20, -5]
transformed = transform_le("source.le", "translated.le", matrix)

directory = Path("objects")
directory.mkdir()  # The split destination directory must already exist.
outputs = split_le(transformed, directory, object_attribute="object_id", association="cell")
merged = merge_le(outputs.values(), "merged.le", object_attribute="object_id", association="cell")
print(summary.logical_object_count, merged.destination, merged.id_mapping)
```

All four functions are exported from both `subsurface` and `subsurface.api`.
`UnstructuredData.from_binary_le(path)` reconstructs the existing data container;
`StructuredData.from_binary_le(path)` reconstructs the supported active scalar
array. The new file operations retain separate numeric attribute columns
internally so mixed integer/float/bool values do not pass through the existing
container's homogeneous xarray attribute matrices.

## Contracts

- `inspect_le` reports one serialized dataset and validated shapes, schema,
  metadata, and file length. Without grouping, logical count is unknown (`None`).
  It does not claim full payload/connectivity validation. With grouping it reads
  the selected ID column, rejects missing/nonfinite/boolean IDs, and returns
  sorted original IDs without integer-to-float conversion.
- Use `association="cell"` for meshes and `association="point"` for point clouds.
  Attribute and association must both be explicit; empty declared grouping
  counts zero. Empty objects without any associated rows cannot be represented.
- Transforms use homogeneous column vectors (`p_out = M @ p_in`). Singular and
  projective matrices are rejected. Declared normals/vectors are transformed;
  scalar attributes and undeclared columns are not guessed from their names.
  Reflections correct triangle winding but reject volumetric cells.
- Splitting remaps connectivity, duplicates shared vertices between outputs,
  drops unreferenced vertices from cell groups, and preserves original IDs.
  Filenames are safe deterministic ordinals; the return value maps IDs to paths.
- Merging concatenates without welding and requires identical attribute
  names/order/dtypes and equal semantic dataset metadata, including CRS/units.
  Default explicit grouping uses `(source_index, original_id)` identity and
  consecutive int64 output IDs. Only `id_policy="shared"` declares globally
  shared IDs. `le_tools` metadata records source provenance and exact ID mapping.
- Transform/merge default to no overwrite. Explicit overwrite still cannot
  replace source aliases. Split preflights and stages every output before
  no-clobber publication, with inode-aware rollback on errors. Multi-file split
  is not crash atomic; filesystem failures can prevent cleanup/rollback.

## Supported Limits

File operations support prefixed unstructured files in the existing default
Fortran geometry order. The reader also supports documented sidecars, but file
operations do not. Array order is not encoded in the format.

Prefixed JSON headers are limited to 16 MiB for unstructured read-back and 1 MiB
for structured read-back. Both readers reject more than 128 nested JSON
containers before decoding, independently of Python's recursion behavior.
Quoted brackets and escaped quotes do not count toward the nesting limit.

Coordinates serialize as float32 and connectivity as int32. Floating attributes
follow the shipped exact-integral-to-int64/otherwise-float32 rule. Float32
rounding/underflow are documented, while finite overflow, unsupported coercion,
and column dropping are rejected. Consequently a split floating column may
serialize to a different dtype when its subset is entirely integral; a strict
merge rejects differing resulting schemas rather than silently promoting them.
Named zero-row attribute schemas that the existing writer would drop are
rejected on output; an empty split produces no files. The existing xarray-backed
reader can still coerce mixed attribute dtypes; use inspection for exact IDs and
the file tools for loss-aware operations rather than relying on mixed attribute
matrix conversion for exact large integer identity.

Structured read-back and inspection use coordinate extrema, inclusive stored
endpoints, Fortran scalar ordering, and explicit singleton handling. Flat bounds
overrides and non-null structured transforms are rejected because their
semantics are unresolved. Only the serialized active array can be recovered.
Restricted structured operations (Task 7) are available as separate explicit APIs
described below. General resampling/mosaics remain outside scope. The user has
removed multi-object container design (Task 8) from scope. No CRS conversion,
resampling, clipping, or topology repair is performed.

## Structured Operations

```python
from pathlib import Path
import numpy as np
from subsurface import (
    transform_structured_le, split_structured_le, merge_structured_le,
)

matrix = np.eye(4)
matrix[:3, 3] = [10, 20, -5]  # Three-dimensional source grid.
translated = transform_structured_le("volume.le", "translated_volume.le", matrix)

directory = Path("tiles")
directory.mkdir()
tiles = split_structured_le(
    translated, directory,
    windows={"left": {"x": (0, 10)}, "right": {"x": (10, 20)}},
)
merged = merge_structured_le(tiles.values(), "merged_volume.le", axis="x")
```

The three file functions are exported from both `subsurface` and `subsurface.api`
and return absolute output paths (a label-to-path mapping for split). The example
selects twenty X samples; omitted axes retain all their samples.

- Transform supports only translation and strictly positive diagonal scaling.
  It changes coordinates, not scalar values. Rank-one `dim0` maps to X; rank-two
  maps to X/Y, with unused matrix axes required to remain identity/zero translation.
- Split windows use half-open integer `(start, stop)` index ranges. Axes are not
  squeezed, singleton samples retain their positions, and overlapping explicit
  windows are allowed. Filenames are deterministic ordinals, never raw labels.
- Merge concatenates adjacent grids along one explicit axis in caller order.
  Rank, active-array name, exact scalar dtype, and nonmerge coordinates must
  agree. Overlaps, gaps, reversed order, and incompatible spacing are rejected.
  If all merge-axis tiles are singletons, supply explicit positive `spacing`;
  their original spacing is not recoverable from the files.

The shared safety boundary verifies temporary files through the public reader
before publication. Unsupported in-memory metadata/arrays/layouts are rejected
instead of dropped. The existing volume format has no provenance/CRS metadata
fields; none are invented. Callers must establish compatible coordinate frames
and units because absent file metadata cannot be checked. Scalar dtype
declarations, integer bits, categorical labels, and NaN/Inf values are preserved
without interpolation. Endpoint coordinate
reconstruction uses a `1e-10 * spacing` tolerance, not world-coordinate magnitude.
Numerically ambiguous transformations or adjacency are rejected, including some
otherwise valid mathematical cases at large origins. Multi-file split is not
crash atomic, and operations process arrays in memory.

See [structured transform](le_structured_transform.md),
[structured split](le_structured_split.md), [structured merge](le_structured_merge.md),
and [output safety](le_grid_ops_safety.md) for detailed policies.

See [inspection](le_inspection.md), [transforms](le_transform.md),
[split](le_split.md), [merge](le_merge.md), [structured reader](structured_le_reader.md),
and the [implementation report](le_tools_implementation_report.md) for precise
policies, verified commits, and remaining qualification work.
