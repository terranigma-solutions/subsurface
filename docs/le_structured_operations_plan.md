# Restricted Structured LE Operations

Task 7 is authorized for exact, non-resampling operations on the scalar format
supported by Task 6. Task 8 (multi-object container) is removed from scope.
This supplements the preserved original implementation plan.

## Shared Boundary

Use the existing `StructuredData` container and wire format. Support one active
numeric scalar array, standard one-to-three-dimensional axes, ascending regular
coordinates, inclusive coordinate extrema, and singleton sample positions.
Preserve active-array name, scalar values/dtype, and Fortran ordering. Reject
unsupported metadata, extra arrays, and coordinate layouts instead of silently
discarding them in in-memory operations. File inputs cannot recover information
already absent from the format. No new CRS or provenance metadata is invented.

Output helpers preflight and validate temporary files before atomic no-clobber
publication, reject every source/destination alias, and require explicit overwrite
for existing destinations. All source files remain unchanged.

## Public APIs

```python
transform_structured_le(source, destination, matrix, *, overwrite=False)
split_structured_le(source, output_directory, *, windows)
merge_structured_le(sources, destination, *, axis, spacing=None, overwrite=False)
```

Transform uses the same homogeneous column-vector convention as unstructured
transforms. Only translation and strictly positive diagonal scaling are accepted.
Rank-one `dim0` maps to X; rank-two maps to X/Y. Unused matrix axes must remain
identity with zero translation. Values are not resampled or changed. Coordinate
overflow, collapsed spacing, rotation, shear, reflections, and projective matrices
are rejected.

Split windows map caller labels to axis-name mappings of half-open integer
`(start, stop)` ranges. Omitted axes retain their full range; steps, negative
indices, empty ranges, and out-of-bounds ranges are rejected. Caller window order
determines safe ordinal output filenames, not raw labels. Return a label-to-path
mapping. Overlapping windows are allowed as independent explicit selections;
there is no implicit partition, clipping, or grouping by scalar values. Preflight
and stage all outputs before publication, with inode-aware rollback on errors.
The operation is not crash atomic across multiple files.

Merge concatenates along one explicit standard axis in caller-supplied order.
Require matching rank, dimension order, active-array name and exact scalar dtype,
and coincident non-merge-axis coordinates. Spacing and adjacency checks use a
documented tight tolerance relative to axis spacing, never relative to world
coordinate magnitude. Reject overlaps, gaps, different resolution, misalignment,
and reversed source order. Infer merge-axis spacing from nonsingleton sources;
if no source provides it, require explicit positive `spacing`. An explicit spacing
must agree with all recoverable source spacings. Do not silently snap coordinates
outside the documented serialization reconstruction tolerance.

## Verification

Cover ranks one, two, and three; nonsymmetric shapes; singleton axes; exact
integer/unsigned/float scalar values and dtype; NaN preservation; identity,
translation and scale; half-open windows; merge on each axis; singleton spacing;
alignment errors; invalid operations; aliases/collisions; atomic safe failures.
Coordination adds public-export and read/inspect/transform/split/merge round-trip
tests, including categorical labels and source immutability.

## Not Included

Arbitrary transforms, interpolation, target grids, categorical resampling,
overlap precedence, gap filling, general mosaics, nodata inference, and
performance qualification on representative production volumes remain separate
work. No multi-object container is planned.
