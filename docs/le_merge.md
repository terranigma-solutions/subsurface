# Merging LiquidEarth Meshes

```python
from subsurface.api.le_merge import merge_le

result = merge_le(
    ["first.le", "second.le"], "merged.le",
    object_attribute="object_id", association="cell",
)
print(result.destination)
print(result.id_mapping)
```

`merge_le(sources, destination, *, object_attribute=None, association=None,
id_policy="source", overwrite=False, source_labels=None)` returns `LEMergeResult` containing an
absolute destination `Path` and a JSON-native list of ID mapping entries, or
`None` for the mapping if grouping was not requested. Sources are consumed in
caller order, including repeated paths. Pass an iterable of paths, not one path.

## Compatibility

- Only embedded-header unstructured meshes in the supported decoder formats
  are accepted. Geometry and payload validation use the shared file tools.
- Connectivity widths must match exactly, including width 0 versus width 1
  point representations. Lines, triangles, tetrahedra and hexahedra cannot mix.
- Width-0 point connectivity must have either zero rows or one row per point
  in every source and in the merged output. Partial row counts are ambiguous
  and rejected, even though the reader accepts them. Mixing zero-row and
  per-point policies on nonempty geometry is rejected because it would produce
  a partial output incompatible with point splitting. Empty geometry does not
  introduce a policy conflict. No rows are invented or discarded to normalize
  these representations.
- Attribute names, column order and per-column dtypes must match exactly on
  both associations. No schema union, missing-value fill or numeric coercion is
  performed. Mixed numeric and bool columns stay separate, without xarray
  matrix promotion. Only the chosen grouping column may be replaced with int64.
  The error names both sources and their columns with dtypes. When column
  names differ (e.g. a DXF `Id` column against an OBJ mesh without one), it
  also says that meshes from different file types cannot be merged yet.
- All semantic dataset metadata must be equal, including coordinate CRS, units,
  transforms and other keys. Missing metadata is distinct from present metadata.
  No CRS/unit inference or reprojection occurs. JSON object key order is ignored;
  JSON value representations must otherwise match (for example, `1` and `1.0`
  differ). Reserved `le_tools` provenance is exempt from compatibility equality.
  A conflict error names the differing keys, e.g. `Sources have different
  "crs" metadata`. Label maps (below) and `bounds` are the exceptions.
- `bounds` is derived from geometry, so it is never compared. When any source
  has it, the output's `bounds` is recalculated like `transform_le` does:
  `[xmin, xmax, ymin, ymax, zmin, zmax]` from the merged float32 vertices, or
  `None` for empty geometry.

## Label Maps

A label map is a `cell_attr_map` or `point_attr_map` metadata value that is an
object of string to integer, such as the DXF reader's layer names
(`{"<layer name>": <id>}`, IDs from 1 in alphabetical layer order). A value
under those keys that is not a string-to-integer object is compared strictly.

- With grouping, the map of the grouped association (`cell_attr_map` for
  `association="cell"`) is not compared. It is rebuilt from the ID mapping
  instead. Entries whose ID does not occur in their source are dropped, since
  split parts carry their whole original map. A source without a map
  contributes no labels. The other association's map is compared strictly.
- `id_policy="source"` writes `label -> merged_id` for every present entry, in
  source order. When a label is already taken by an earlier source, it becomes
  `"<label> (<source name>)"`, where the source name is the provenance `name`
  without its `.le` suffix (`source <index>` without one), with a counter if
  that is taken too. Each rename is recorded in `le_tools["renamed_labels"]`
  as `source_index`, `label`, `merged_label` and `merged_id`.
- `id_policy="shared"` takes the union of the present entries. A label with
  different IDs in different sources, or an ID with different labels in
  different sources, is rejected.
- Without grouping, IDs are not remapped, so both label maps must be equal in
  every source (including absent in every source). Differing maps are rejected
  with a message to keep object IDs, i.e. merge with grouping.

`le_tools["renamed_labels"]` is present only when a grouped label map was
rebuilt. `le_tools["id_mapping"]` is unchanged.
- Every vertex is concatenated, including unused vertices. Each source's cells
  are offset by the cumulative preceding vertex count. No vertex welding,
  topology repair or connectivity reordering occurs. More than `int32.max + 1`
  total vertices is rejected before concatenation and connectivity conversion.
- Empty source lists are rejected. Empty meshes with matching schemas and no
  named zero-row attributes can merge, including wholly empty outputs. Named
  zero-row attributes in any source are explicitly rejected because the shared
  serializer would drop their schema, even if other sources have nonempty rows.

## Object Identity

Grouping requires the explicit `object_attribute` and `association` pair.
Use `association="cell"` for connectivity widths above 1, or `"point"` for
point clouds. IDs must be finite numeric nonbool values; missing attributes,
NaN and infinite IDs are rejected. No grouping means logical object count is
unknown, even when an arbitrary ID column is present.

The default `id_policy="source"` treats `(source_index, original_id)` as identity.
Consecutive int64 IDs start at zero, in source order and then sorted original
ID order within each source. This separates equal IDs from different files.
`id_policy="shared"` preserves the original column and allows IDs in different
sources to represent the same object. It requires explicit grouping.

Each mapping entry has `source_index`, `original_id` and `merged_id`. Original
integer IDs are Python integers and are serialized as JSON integers, never
converted through float. This reconstructs exact int64 IDs, including values
above `2**53`; JSON consumers must likewise avoid converting them to double.
Float IDs retain their decoded wire values, not pre-export float64 precision.

## Provenance And Safety

Output `data_attrs["le_tools"]` records `operation="merge"`, grouping arguments,
ID policy, ID mapping and ordered sources. Each source records its index,
absolute path and filename, plus its entire prior `le_tools` value when present.
Optional `source_labels` (one non-empty string per source, e.g. blob path plus
version) are recorded verbatim as `path`, with their last `/` component as
`name`, so outputs do not depend on local download locations.
Different split/merge sibling provenance does not cause a metadata conflict.
When present, reserved `le_tools` provenance must be a dictionary; null, scalar
and list values are rejected rather than treated as arbitrary semantic metadata.
All other equal metadata is preserved. This uses the existing JSON header and
binary format, not a new container.

Existing destinations are protected by atomic no-clobber publication by default.
`overwrite=True` requests atomic replacement, but source aliases (including
symlinks and hard links) are always rejected. The destination parent must exist.
Failures leave inputs and existing outputs unchanged and clean temporary files.
Concurrent hostile filesystem identity changes are outside the shared writer's
contract. Invalid data raises `ValueError`/`TypeError`; filesystem errors propagate.

Coordinates use the existing float32 wire precision; integer/bool attributes
retain widths and values. Floating columns use the shipped exactly-integral to
int64 rule, otherwise float32. Finite float overflow is rejected; float32
rounding and underflow are allowed. Nonfinite nongrouping attributes are retained.

`merge_meshes(meshes, *, object_attribute=None, association=None,
id_policy="source", sources=None, source_labels=None)` is the in-memory equivalent. It accepts
`LiquidEarthMesh`, returns a new validated mesh and does not mutate inputs.
Optional `sources` supplies one path label per mesh for provenance; absent labels
produce null paths/names, not invented filesystem sources. `sources` labels are
made absolute; `source_labels` are kept verbatim. Supply at most one of the two. Its mapping is in
`mesh.data_attrs["le_tools"]["id_mapping"]`. It intentionally avoids conversion
through `UnstructuredData` to preserve independent column dtypes.
