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
- `shapes`: `vertex`/`cells`, or structured `data`, as dimension tuples.
- `vertex_count`, `cell_count`, `grid_sample_count`: applicable geometry counts;
  inapplicable counts are `None`.
- `dataset_count`: one serialized dataset, not a geological object count.
- `attribute_schema`: `cell` and `point` column tuples; structured files also
  have a `grid` entry for the serialized active array. Columns report `name`,
  wire `dtype`, and `shape`; unstructured columns also report `byte_length` and
  payload-relative `offset`.
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
promise about voxel-center/edge semantics**.

## Read And Validation Scope

Inspection reads only the four-byte little-endian length and bounded JSON header
unless grouping is requested. It checks recognized versions, dimensions, numeric
attribute schema and lengths, unique names, and exact total file length against
the declared layout. The header limit is checked before reading JSON bytes.
Version-2 grouping seeks directly to and reads only the requested column.
Legacy embedded headers use the existing legacy float32 attribute decoder on
the relevant attribute block, in default Fortran order, without reading geometry.
Legacy sidecar files are not accepted. Ambiguous flattened connectivity is not
guessed; unsupported connectivity widths are rejected.

`header_validated=True` and `payload_length_validated=True` do **not** mean
payload contents are valid. Inspection does not check coordinate finiteness,
connectivity indices, other attribute values, or topology. A corrupt connectivity
array of the correct byte size can pass inspection. `payload_validated` therefore
always remains `False`. Requested grouping validates only its ID values and sets
`grouping_validated=True`. Use the hardened full reader for payload/connectivity
validation; this API does not offer a full-validation mode.

Structured inspection currently recognizes three-dimensional regular
axis-aligned scalar headers with finite ordered bounds and null transforms.
Non-null transforms are rejected rather than silently ignored.

## Foundation Integration

The private `_header_layout` adapter is a temporary integration seam while the
read-back foundation validator is developed on the parent branch. The coordinator
should replace its schema checks with that committed validator's normalized
layout, preserving bounded reads, offsets, summary fields, and validation-scope
semantics. This is not a second public parsing API. No new runtime dependencies
or changes to the existing serializers are required.
