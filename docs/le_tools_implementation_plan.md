# LiquidEarth File Tools Implementation Plan

Status: proposed; implementation has not started.

Branch: `le-tools/readback-and-operations`, based on local `main` at `727e6aa`.

## Goals

Provide dependable library operations for reading and inspecting `.le` files,
counting logical objects, baking a 4x4 affine transform into geometry, splitting
files by object, and merging compatible files. Treat structured volumes as a
separate implementation track rather than promising mesh operations will work
unchanged on grids.

This plan covers library code, documentation, and automated tests. Azure
endpoints, user interfaces, deployment, and a new heterogeneous file container
are outside the initial implementation scope.

## Current State

The existing format is a four-byte little-endian JSON-header length, followed
by a UTF-8 JSON header and binary arrays. Each file represents one unstructured
dataset or one structured active data array, not a general object archive.

Relevant implementation files:

- `subsurface/core/structs/base_structures/unstructured_data.py`: current and
  sidecar-based legacy readers, unstructured serialization.
- `subsurface/core/structs/base_structures/_liquid_earth_mesh.py`: header and
  payload decoding, numeric attribute serialization and filtering.
- `subsurface/core/structs/base_structures/structured_data.py`: structured
  serialization; no matching `.le` reader currently present.
- `subsurface/modules/reader/mesh/mx_reader.py`: mesh concatenation precedent,
  including a generated cell-level `id` attribute.
- `subsurface/modules/writer/to_binary.py`: legacy sidecar file writer.
- `docs/le_spec.md`: format documentation; update its unstructured examples
  to distinguish legacy float32 attribute arrays from current version-2
  per-column dtype metadata.

Unstructured read-back supports geometry and numeric attributes, but is not
currently a complete, lossless source-object round-trip:

- `from_binary_le()` and its legacy counterpart discard decoded dataset attrs.
- Writing silently filters out nonnumeric attributes; numeric-looking object
  columns can be coerced.
- Coordinates are stored as float32 and connectivity as int32. Floating
  attributes can change precision or be encoded as integers.
- Readers need stronger schema, version, payload-length, and connectivity
  validation. Legacy flattened connectivity uses an ambiguous inference rule.
- Original element wrappers, texture objects, and high-level geological objects
  are not reconstructed automatically.
- Existing point-cloud tests check vertices and point attributes, but do not
  establish complete connectivity, cell-attribute, or metadata preservation.

The earlier evaluation also examined uncommitted work in the original
`time_dimension` checkout. That work is not included in this worktree. Use the
committed baseline and do not depend on those changes or their additional tests.

## Scope Decisions

The following are proposed defaults, to confirm against representative files
and LiquidEarth consumer requirements before implementation:

- An object is an explicitly identified group of cells, or points for a point
  cloud. Dataset count, primitive count, and logical object count are distinct.
- Callers specify the grouping attribute and its association. Do not guess
  object identity from arbitrary `id` columns or connected components.
- Missing grouping metadata means logical object count is unknown, not one.
  Inspection can still report one serialized dataset and its geometry counts.
- Keep the existing format for the initial release. A grouping convention or
  provenance metadata addition must be checked with the consuming application.
- File operations write new output files by default and do not mutate inputs.
- Reject incompatible inputs rather than silently dropping attributes,
  changing coordinate systems, or flattening mixed geometry types.
- Preserve existing numeric wire-format precision rules and document them.
  Changing coordinate precision is a separate format-compatibility decision.
- Preserve documented legacy formats because persisted files already exist;
  do not add speculative compatibility for unknown variants.

## Task 1: Harden Unstructured Read-Back

Estimate: 2-4 engineering days. Complexity: low to medium.

Implementation tasks:

- [ ] Restore `mesh.data_attrs` when reconstructing current and legacy datasets.
- [ ] Validate header prefix, bounded header size, JSON type, recognized format
  version, shapes, allowed numeric dtypes, attribute lengths, and unique names.
- [ ] Check declared byte lengths against shapes and dtypes before allocation;
  reject truncated payloads and unexpected trailing bytes for supported schemas.
- [ ] Validate connectivity indices, attribute row counts, and supported
  point-cloud special cases, including empty datasets.
- [ ] Define clear errors for ambiguous legacy flattened connectivity instead
  of silently choosing a topology without sufficient information.
- [ ] Document array order and endianness. Default-order round-trips must be
  reliable; reject or explicitly handle nondefault order where the file does
  not encode enough information to infer it.
- [ ] Make unsupported-attribute loss explicit in the new file tools. Decide
  separately whether changing existing writer behavior requires a warning or
  an explicit opt-in to dropping columns.
- [ ] Use context-managed, atomic output writing for new file operations;
  validate before replacing an existing destination.
- [ ] Update `le_spec.md` to describe current version-2 attributes accurately.

Acceptance tests:

- Round-trip points, lines, triangles, and tetrahedra, comparing connectivity
  values, numeric/bool attributes, names, and dataset metadata.
- Cover empty data, absent attributes, NaN numeric attributes, and legacy files.
- Assert documented float32 tolerance rather than original float64 identity.
- Reject malformed headers, unsupported versions/dtypes, length mismatches,
  duplicate columns, and out-of-range indices with useful exceptions.
- Use checked-in or generated small fixtures without plotting, network access,
  or optional native visualization dependencies.

## Task 2: Inspection And Object Counts

Estimate: 0.5-1 day for geometry inspection; 1-2 days for explicit object counts.
Complexity: low to medium.

Implementation tasks:

- [ ] Add inspection of file kind, format version, byte size, array shapes,
  attribute names/dtypes, and available metadata.
- [ ] Report vertex, cell, or grid-sample counts from validated headers without
  loading the entire geometry payload.
- [ ] Distinguish header inspection from full payload/connectivity validation.
- [ ] Accept an explicit grouping attribute and association for logical counts.
- [ ] Count unique IDs, rejecting missing IDs and invalid grouping data under
  a documented policy. Specify whether empty groups can be represented.
- [ ] For version-2 files, use attribute offsets to read only the requested ID
  column where practical; use the existing decoder for legacy layouts first.
- [ ] Return unknown logical count when no grouping is supplied. Do not equate
  disconnected components, cells, or voxels with geological objects.

Acceptance tests:

- Two surfaces in one dataset report one dataset, two logical objects, and the
  correct primitive counts when the grouping attribute is supplied.
- Repeated and sparse IDs count correctly; integer IDs do not lose precision.
- Point-level IDs work for point clouds; wrong associations fail clearly.
- Header-only inspection does not read the full payload, and does not claim
  that payload contents have been validated.

## Task 3: Bake A 4x4 Transform Into Unstructured Geometry

Estimate: 2-4 engineering days. Complexity: medium. Depends on Task 1.

Implementation tasks:

- [ ] Define the convention as homogeneous column vectors: `p_out = M @ p_in`.
  With NumPy row arrays, use `xyz_out = xyz @ M[:3, :3].T + M[:3, 3]`.
- [ ] Validate matrix shape, finite values, and affine last row `[0, 0, 0, 1]`.
  Projective transforms are outside the initial scope.
- [ ] Compute in float64, then validate finite values and float32 range before
  serialization. Document precision limitations for large world coordinates.
- [ ] Preserve connectivity, IDs, scalar attributes, UVs, and dataset metadata.
- [ ] Support explicitly declared vector and normal columns rather than
  inferring semantics from names. Transform normals with the inverse transpose
  and renormalize; reject singular matrices when normals require an inverse.
- [ ] Define the singular-transform policy for geometry itself. Prefer rejecting
  degenerate transforms in the initial tool rather than collapsing meshes.
- [ ] For triangle reflections, preserve outward orientation by swapping two
  cell indices. Reject reflected volumetric cells until their orientation
  policy is implemented and tested.
- [ ] Update derived bounds and document handling of existing transform metadata
  so baking never applies a transform twice or leaves stale metadata behind.
- [ ] Write a new file and leave the source unchanged, including on failure.

Acceptance tests:

- Identity, translation, rotation, nonuniform scale, shear, and reflection.
- Known coordinate results, transform composition, and inverse round-trips
  within the documented serialization tolerance.
- Normal orthogonality and normalization under nonuniform scale.
- Reject projective, nonfinite, unsupported singular, and overflow cases.
- Confirm attributes and metadata remain unchanged except intentional geometry
  and orientation updates.

## Task 4: Split Unstructured Files By Object

Estimate: 3-5 engineering days. Complexity: medium. Depends on Tasks 1 and 2.

Implementation tasks:

- [ ] Split by an explicit cell-level grouping attribute, or point-level IDs
  for point clouds. Reject unsupported associations.
- [ ] Select cells, gather referenced vertices, and remap connectivity to local
  indices. Duplicate shared vertices across outputs rather than welding.
- [ ] Subset cell and point attributes consistently and preserve dataset attrs.
- [ ] Define handling of unreferenced vertices: exclude them from cell-grouped
  outputs and document this; point-cloud splitting retains selected points.
- [ ] Preserve original object IDs. Use deterministic ordering and safe output
  names, with an ID-to-output mapping rather than raw IDs as filesystem paths.
- [ ] Preflight destinations and collisions before writing. Clean up temporary
  files on failure and define partial-output behavior for a multi-file split.
- [ ] Record sufficient provenance to relate outputs to their source without
  requiring a new binary container.

Acceptance tests:

- Shared vertices, unused vertices, sparse IDs, multiple attributes, point
  clouds, empty input, missing grouping, and conflicting destinations.
- Every output has valid local connectivity and the expected attributes.
- Split then merge reproduces grouped geometry and values up to documented
  vertex duplication, removal of unused vertices, and ordering changes.
- Spatial selection and geometric clipping are explicitly not part of this
  initial split operation.

## Task 5: Merge Compatible Unstructured Files

Estimate: 3-5 engineering days. Complexity: medium. Depends on Tasks 1 and 2.

Implementation tasks:

- [ ] Require matching geometry/connectivity kinds, coordinate systems/units,
  and compatible attribute schemas; do not automatically reproject inputs.
- [ ] Concatenate vertices and offset cell indices by cumulative vertex count,
  including unused vertices. Check int32 connectivity capacity before writing.
- [ ] Start with strict matching attribute names and compatible dtypes. Missing
  column filling or permissive schema union can be a later explicit policy.
- [ ] Resolve ID collisions deterministically using `(source, original_id)` as
  identity when requested, and retain a mapping to original IDs and sources.
- [ ] Allow callers to declare that IDs are globally shared only explicitly;
  preserve grouping consistently with the counting and splitting tools.
- [ ] Define metadata conflict handling. Preserve equal values, retain source
  provenance, and reject unresolved coordinate-frame or semantic conflicts.
- [ ] Avoid vertex welding, topology repair, and mixed triangle/line arrays.
- [ ] Produce deterministic outputs and atomic destination writes.

Acceptance tests:

- Correct offsets with unused vertices and multiple source files.
- Colliding IDs, globally shared IDs, empty inputs, metadata conflicts,
  incompatible schemas, incompatible geometry, and capacity checks.
- Read the merged output with the public reader and check all arrays and attrs.
- Verify subsequent object counting and splitting preserve source identity.

## Task 6: Structured-Volume Read-Back

Estimate: 2-4 engineering days. Complexity: medium. Can follow Task 1 independently
of the unstructured geometry tools.

Implementation tasks:

- [ ] Add a `.le` reader for the existing regular axis-aligned scalar-grid format.
- [ ] Validate shape, numeric dtype, bounds, data name, payload length, and
  supported transform values. Do not silently ignore non-null transforms.
- [ ] Establish whether bounds represent samples, voxel centers, or voxel edges
  using representative exports and existing `from_pyvista()` behavior.
- [ ] Reconstruct coordinates and active-array identity with correct axis order
  and spacing; explicitly handle singleton dimensions.
- [ ] Document that only the serialized active array can be reconstructed;
  discarded arrays or metadata cannot be recovered from existing files.
- [ ] Extend inspection to structured files without defining every voxel as a
  logical object.

Acceptance tests:

- Known-coordinate grids, nonsymmetric axis sizes, multiple supported numeric
  dtypes, singleton axes, and public write/read/write round-trips.
- Bounds and scalar ordering match reference exports, including the staged VTK
  exporter in the adjacent active `subsurface-le` project when available.
- Unsupported transforms and malformed dimensions fail explicitly.

## Task 7: Structured Transforms, Split, And Merge

Estimate: 1-3 weeks per scoped general-purpose feature. Complexity: high.
Depends on Task 6 and explicit grid semantics.

Start with a restricted scope:

- [ ] Bake translation and positive axis-aligned scaling through grid-coordinate
  or bounds updates without changing scalar values.
- [ ] Split rectangular index ranges, preserving sample positions and bounds.
- [ ] Merge compatible, aligned, equal-resolution tiles, with explicit overlap
  and missing-region policies.

Treat arbitrary transforms and heterogeneous grids as a separate extension:

- [ ] Require an explicit target grid, resolution, extent, fill value, and
  resource limit for rotation/shear baking or resolution-changing merges.
- [ ] Use nearest-neighbour resampling for categorical IDs and an explicit
  interpolation policy for continuous fields.
- [ ] Specify overlap precedence, nodata semantics, and conservation expectations.
- [ ] Benchmark representative volumes for runtime, peak memory, and output size.
- [ ] Reject unsupported cases rather than producing misleading axis-aligned
  bounds for a rotated or sheared volume.

Acceptance tests include exact axis-aligned cases, categorical-label preservation,
known interpolated fields, tile boundaries, overlaps, gaps, and resource limits.

## Task 8: Optional Multi-Object Container Design

Complexity: high; schedule requires format discovery and consumer agreement.
Not required for grouping-based tools above.

- [ ] Obtain representative files if the requested `.le` means a different
  LiquidEarth container from the subsurface serialization documented here.
- [ ] Define an object table with names/IDs, geometry kinds, payload offsets,
  object metadata, and coordinate-frame semantics.
- [ ] Define how textures/materials, mixed geometry, shared vertices, structured
  volumes, and high-level geological identity are preserved.
- [ ] Agree versioning and compatibility with the LiquidEarth application before
  changing the wire format or introducing a new container.
- [ ] Estimate reader/writer, migrations, count/split/merge, and consumer changes
  separately after the format contract is approved.

## API And Implementation Approach

Keep binary validation close to the existing decoder and reuse existing data
containers. Add a small library-level file-tools surface only where it makes
these operations composable; avoid introducing a generic framework.

Candidate API names, not yet committed public interfaces:

```python
inspect_le(path, *, object_attribute=None, association=None)
transform_le(source, destination, matrix, *, normal_columns=None, vector_columns=None)
split_le(source, output_directory, *, object_attribute, association)
merge_le(sources, destination, *, object_attribute=None)
StructuredData.from_binary_le(path)
```

Specify return values, exception behavior, overwrite policy, ID mapping, and
metadata rules before finalizing the public API. In-memory operations should be
testable independently of file writing. Preserve documented persisted-format
compatibility, but do not add new aliases or fallback schemas without evidence.

## Graphite Delivery Order

Prefer small, independently tested PRs stacked from `main`:

1. Read-back metadata preservation and complete round-trip tests.
2. Binary validation and accurate format documentation.
3. Header inspection and explicit object counts.
4. Unstructured affine transform baking.
5. Split-by-ID with connectivity remapping.
6. Compatible merge with ID/provenance policies.
7. Structured reader and grid-semantics tests.
8. Restricted grid operations, followed by separately scoped resampling work.

The current branch contains this plan only. Do not create commits or submit PRs
until requested. Estimates include focused library tests and documentation, but
exclude deployment and performance qualification. Shared work overlaps between
tasks, so the estimates are not a simple additive schedule.

## Verification And Completion

Put self-contained core tests under `tests/test_interfaces/` or the relevant
existing I/O test directory. Use temporary paths and generated small datasets;
do not require plotting, external downloads, or user-specific files.

Suggested commands once the test modules exist:

```bash
REQUIREMENT_LEVEL=CORE python -m pytest tests/test_interfaces/
REQUIREMENT_LEVEL=CORE python -m pytest tests/test_structs/
git diff --check
```

Run relevant existing mesh, point-cloud, and volume tests when their optional
dependencies and fixtures are available. Report skips separately from passes.
Inspect representative real `.le` exports in addition to synthetic fixtures
before calling the pipeline production-ready.

Completion requires documented object semantics, reliable supported-format
read-back, no silent losses in new file operations, deterministic grouping and
ID behavior, valid connectivity after split/merge, and safe failure behavior
that leaves source files intact.
