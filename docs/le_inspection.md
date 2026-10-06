# LiquidEarth Inspection

```python
from subsurface import inspect_le

summary = inspect_le("surfaces.le")
grouped = inspect_le("surfaces.le", object_attribute="surface_id", association="cell")
print(grouped.dataset_count, grouped.cell_count, grouped.logical_object_count)
print(grouped.object_ids)
```

## API

`inspect_le(path, *, object_attribute=None, association=None,
max_header_bytes=8 * 1024 * 1024)` accepts a filesystem path and returns an
`LEInspection` dataclass (defined in `subsurface.api.le_inspection`). It does not
modify files. Invalid headers, layouts, file lengths, or grouping requests raise
`ValueError`; filesystem errors propagate normally.

The summary includes:

- `file_kind`: `"unstructured"` or `"structured"`.
- `format_version`: 1 for existing headers without a version, or explicit 1/2.
- `byte_size`: total size including the four-byte prefix and JSON header.
- `shapes`: normalized `vertex`/`cells`, or structured `data`, as dimension
  tuples. Resolved legacy flattened connectivity also reports `wire_cells` for
  the original stored shape. Empty `[0, 0]` vertices normalize to `(0, 3)`.
- `vertex_count`, `cell_count`, `grid_sample_count`: applicable geometry counts;
  inapplicable counts are `None`.
- `dataset_count`: one serialized dataset, not a geological object count.
- `attribute_schema`: `cell` and `point` column tuples; structured files also
  have a `grid` entry for the serialized active array. Columns report `name`,
  wire `dtype`, and `shape`; unstructured columns also report `byte_length` and
  payload-relative `offset`.
  Foundation-supported JSON scalar column labels (including numeric, null, and
  empty string labels) remain unchanged in the schema. Grouping requests still
  require a nonempty string `object_attribute`.
- `metadata`: stored `xarray_attrs`; structured summaries additionally expose
  `bounds`, `transform`, and `data_name`. Unserialized metadata cannot be recovered.
- `logical_object_count` and `object_ids`: `None` without explicit grouping;
  otherwise the count and ascending unique original numeric IDs.
- `header_validated`, `payload_length_validated`, `grouping_validated`, and
  `payload_validated`: separate validation scopes.

## Object Semantics

Both `object_attribute` and `association` must be supplied together. Association
is `"cell"` for meshes (lines, triangles, tetrahedra, hexahedra), or `"point"`
for point clouds (zero- or one-node connectivity). Point grouping on a mesh,
cell grouping on a point cloud, and structured grouping are rejected explicitly.
The attribute must exist on the requested association. No arbitrary `id` column
name, disconnected component, cell, or grid sample is inferred to be an object.
Two grouped surfaces in one file are one dataset and two logical objects.

IDs are numeric integers or finite floating-point values, not booleans or
strings. NaN and infinity are rejected as missing/invalid IDs. Negative, sparse,
repeated, and fractional finite IDs are allowed. Integer IDs retain their stored
precision, including 64-bit values, and are never converted through floating
point. Returned IDs are sorted, not renumbered. Legacy float32 IDs retain only
the precision already present in that file.

An explicitly declared ID column with zero rows yields `object_ids=()` and
`logical_object_count=0`. Groups without any members cannot be represented by
these files. If an empty file has no ID column, requesting grouping still fails.
Structured geometry counts are grid samples, **not geological objects or a
voxel count**. Stored bounds describe coordinate sample extrema.

### In-Memory Grouping

Operation agents can reuse the helper directly without file I/O:

```python
from subsurface.api.le_inspection import group_object_ids

ids = group_object_ids(cell_attributes, object_attribute="surface_id",
                       association="cell", cell_width=3)
```

`group_object_ids(attributes, *, object_attribute, association, cell_width)`
accepts a mapping or DataFrame of columns for the requested association and
returns an ascending unique NumPy ID array, retaining its numeric dtype and
original values (including int64 precision). `cell_width` is the number of nodes
per cell: point grouping is required for widths 0/1, cell grouping otherwise.
It uses the same finite, non-boolean numeric ID policy as file inspection and
rejects absent columns, missing IDs, multidimensional columns, and incorrect
associations with `ValueError`. Empty columns produce empty ID arrays. It does
not modify attributes, validate geometry, or check geometry/attribute row counts;
operations must validate those separately. This helper is exported from its
module only, not the package root.

## Read And Validation Scope

Inspection reads only the four-byte little-endian length and bounded JSON header
unless grouping is requested. It checks recognized versions, dimensions, numeric
attribute schema and lengths, unique names, and exact total file length against
the declared layout. The header limit is checked before reading JSON bytes and
cannot exceed foundation's 16 MiB limit. Foundation's `read_le_header` parser
rejects duplicate JSON keys, including nested objects, for both file kinds.
Version-2 grouping seeks directly to and reads only the requested column.
Legacy embedded headers use foundation-validated float32 column offsets in
default Fortran order, also reading only the selected column without geometry.
Legacy sidecar files are not accepted. Flattened legacy connectivity is resolved
only when foundation's schema evidence is unambiguous, including attribute row
counts; ambiguous layouts are rejected rather than guessed.

`header_validated=True` and `payload_length_validated=True` do **not** mean
payload contents are valid. Inspection does not check coordinate finiteness,
connectivity indices, other attribute values, or topology. A corrupt connectivity
array of the correct byte size can pass inspection. `payload_validated` therefore
always remains `False`. Requested grouping validates only its ID values and sets
`grouping_validated=True`. Use the hardened full reader for payload/connectivity
validation; this API does not offer a full-validation mode.

Structured inspection follows the committed structured reader's header contract:
one to three positive dimensions with default axes `dim0` (1D), `x/y` (2D), or
`x/y/z` (3D). Bounds must map exactly those axes to finite ordered coordinate
sample extrema representable without loss in float64. Singleton axes require
equal extrema; larger axes require distinct extrema with a finite range. Flat
bounds overrides are ambiguous and rejected, even if produced by the writer's
optional bounds setter. The name must be nonempty, non-whitespace, and distinct
from the axis names. Integer (8/16/32/64-bit signed or unsigned) and floating
(16/32/64-bit) scalar dtypes are supported; boolean and complex are rejected.
The header must contain exactly `data_shape`, `bounds`, `transform`, `dtype`, and
`data_name`; transforms must be null. Geometry/payload validation and coordinate
array reconstruction remain the full reader's responsibility.

## Foundation Integration

Unstructured inspection delegates schema and exact payload-length validation to
foundation's `validate_unstructured_layout`, then adapts its normalized shapes
and payload-relative segments into summary fields. There is no separate
unstructured schema validator. Structured header checks retain the structured
reader contract above; all JSON parsing uses foundation's `read_le_header`.
No new runtime dependencies or changes to existing serializers are required.
