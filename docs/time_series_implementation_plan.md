# Time Series Implementation Plan

## Goal

Support time-dependent volume values and measurements on fixed borehole
trajectories with minimal code and binary changes. Store one ordinary `.le`
snapshot per timestamp, with temporal metadata in each JSON header and a small
index for discovery. Keep existing static imports and payload layouts unchanged.

Local metadata, snapshot export, and examples are implemented. This remains an
end-to-end plan: client and publication contracts still need agreement.

## Implementation Status

- Delivered locally: optional structured `xarray_attrs` without changing static
  output, and unstructured metadata restoration in both binary read interfaces.
  The header contract is documented in `docs/le_spec.md`.
- Delivered locally: `export_time_series()` consumes an iterable of aware
  timestamp/snapshot pairs, validates fixed geometry and schemas, writes ordinary
  `.le` frames, and publishes a chronologically sorted `series.json` last.
  `export_volume_time_series()` slices a named time axis and requires an explicit
  `source_timezone` for naive coordinates. The time dimension may appear anywhere,
  but spatial dimensions after dropping time must be exactly `("x", "y", "z")`
  in that order. Trajectory identity attributes `well_id`, `measured_depths`, and
  `is_attr_point`, and the `well_id_mapper` metadata, cannot change across frames.
  Both exporters require a new output directory.
- Delivered locally: `examples/time_series_boreholes.py` handles explicit XYZ
  measurements with a required source timezone and single-trajectory affirmation;
  `examples/time_series_volumes.py` exports a synthetic volume series. Measurement
  NaNs are preserved. See [Time Series Usage](time_series_usage.md).
- Remaining: source timezone/CRS/trajectory confirmations for the external data,
  Azure/backend upload and publication, client series grouping/discovery, seeking,
  caching, and runtime interpolation. None of these client/backend features is
  implemented by the local exporter; strict client header compatibility still
  needs validation. Steps 4 and 5 and their acceptance criteria remain future work.

## Decisions

- Use existing `StructuredData` and `UnstructuredData` objects; no new temporal
  geometry class or binary format version is needed initially.
- Represent a series as ordered snapshots sharing a stable series identifier.
- Export volumes as ordinary 3D snapshots, not 4D `.le` payloads.
- Export boreholes as ordinary line meshes with changing numeric attributes.
- Repeat geometry in each trajectory snapshot initially. Export collars once
  only when the source actually supplies collars.
- Store observation timestamps in binary headers and in a series index.
- Keep source timestamps separate from ingestion/publication timestamps.
- Do not impose an Azure-driven hard limit on stored frame count.
- Defer changing trajectories, changing connectivity, shared-geometry payloads,
  temporal compression, and live append APIs.

## Example Dataset

Local integration example:

`/home/leguark/DevOps/SubsurfaceTestData/boreholes/2terranigma`

| Source file | Filename time, timezone unspecified | Samples |
| --- | --- | --- |
| `2025-12-12_10-45-39_strain_and_temperature.csv` | `2025-12-12 10:45:39` | 2,650 |
| `2025-12-12_11-45-39_strain_and_temperature.csv` | `2025-12-12 11:45:39` | 2,650 |
| `2025-12-12_12-45-39_strain_and_temperature.csv` | `2025-12-12 12:45:39` | 2,650 |
| `2025-12-12_13-45-39_strain_and_temperature.csv` | `2025-12-12 13:45:39` | 2,650 |

Inspection of all rows found identical XYZ and position strings across the four
files. Position spans 0 to 137.886 m. Each file has 244 `NaN` samples in each of
the three strain columns, and none in the three temperature columns.

| Source column | Proposed canonical field | Interpretation |
| --- | --- | --- |
| `X [m]`, `Y [m]`, `Z [m]` | vertex XYZ | Explicit trajectory coordinates in metres |
| `position [m]` | `position` | Sample position along the measurement trajectory |
| `brillouin_strain [Ghz]` | `brillouin_strain` | GHz |
| `raw_strain [\u00b5m/m]` | `raw_strain` | Micrometres per metre |
| `strain_change [\u00b5m/m]` | `strain_change` | Micrometres per metre |
| `brillouin_temp [Ghz]` | `brillouin_temp` | GHz |
| `temperature [\u00b0C]` | `temperature` | Degrees Celsius |
| `temperature_change [\u00b0C]` | `temperature_change` | Degrees Celsius |

The escaped unit characters above denote the actual Unicode CSV headers.
Retain source labels and units as metadata while using ASCII canonical fields.
Do not reinterpret `position` as vertical depth or a conventional survey measured
depth without confirmation. Do not recalculate the supplied change columns or
assume what reference they use.

These files contain no well IDs, collars, survey angles, or timestamp column.
They are sampled XYZ measurements, not the existing three-file CSV-wells input.
Treat them as one ordered trajectory only after confirming that interpretation;
the directory name alone does not prove there is only one physical borehole.
Coordinate reference system and any discontinuities also need confirmation.

## Metadata Contract

Use the existing unstructured `xarray_attrs` header field and add the same optional
field to structured snapshot headers. Minimal temporal attributes:

```json
{
  "xarray_attrs": {
    "time_series_id": "2terranigma-strain-temperature",
    "timestamp": "2025-12-12T10:45:39Z"
  }
}
```

This timestamp is illustrative only: it assumes the filename times are UTC.
The importer must require a source timezone for naive filename timestamps, then
normalize to UTC ISO 8601. Never silently append `Z` or use the host timezone.
Preserve source precision; reject invalid or ambiguous local times unless the
caller supplies an explicit resolution. Require unique timestamps per series.

Store JSON-safe strings/numbers/dictionaries, not Python datetime or NumPy objects.
Optional source/unit metadata should retain the source filename, source timezone,
attribute units, and confirmed coordinate reference system. The series identifier
must remain stable across frames and must not be derived solely from a timestamp.

For structured output, add `xarray_attrs` only when temporal metadata exists, so
ordinary static files retain their current header and exact binary hashes. Avoid
dumping arbitrary non-JSON-safe xarray attributes into a previously static writer.
Unstructured output already writes dataset attributes; retain that behavior.

## Series Index

Use a small sidecar JSON initially unless the existing Liquid Earth dataset
metadata can carry this contract directly. Paths are relative to the index;
do not persist expiring SAS URLs. Resolve authorization when accessing the files.

```json
{
  "schema_version": 1,
  "time_series_id": "2terranigma-strain-temperature",
  "kind": "trajectory",
  "frames": [
    {"timestamp": "2025-12-12T10:45:39Z", "path": "trajectory_0000.le"},
    {"timestamp": "2025-12-12T11:45:39Z", "path": "trajectory_0001.le"},
    {"timestamp": "2025-12-12T12:45:39Z", "path": "trajectory_0002.le"},
    {"timestamp": "2025-12-12T13:45:39Z", "path": "trajectory_0003.le"}
  ]
}
```

Again, example UTC values are conditional on source timezone confirmation.
Sort by normalized time and verify header/index agreement. Include a shared
collars path only for datasets that have collars. The index must allow seeking
to two surrounding frames without listing the container or fetching all headers.

## Implementation Steps

### 1. Temporal Metadata and Round Trips

Relevant paths are relative to the `subsurface` repository.

- In `subsurface/core/structs/base_structures/structured_data.py`, add optional
  temporal `xarray_attrs` to the existing JSON header without changing payload
  dtype, ordering, bounds representation, or static output.
- In `subsurface/core/structs/base_structures/unstructured_data.py`, preserve
  `LiquidEarthMesh.data_attrs` when rebuilding objects in `from_binary_le()` and
   `from_binary_le_legacy()`. Both now restore decoded metadata.
- Reuse the current `LiquidEarthMesh` metadata writer/reader. Do not introduce
  another timestamp array or extend XYZ to XYZT.
- Validate paired series ID/timestamp at the temporal export boundary. Reject
  unsliced time-dependent arrays there instead of ambiguously exporting them.
- Update `docs/le_spec.md` with temporal metadata and index examples. Its current
  unstructured examples describe the legacy attribute layout; distinguish that
  from the current v2 columnar encoding rather than changing the encoding.

### 2. Borehole Example Import and Snapshot Export

- Add a focused example under `examples/` accepting a dataset directory and an
  explicit source timezone. Initially keep the file discovery and filename-time
  parsing here rather than introducing a broad new reader framework.
- Read/map numeric columns, parse filenames with
  `%Y-%m-%d_%H-%M-%S_strain_and_temperature.csv`, normalize time, and sort frames.
- Validate finite XYZ/position, unique sample positions, increasing position,
  consistent fields/units, equal sample counts, and matching geometry across all
  frames. Reject mismatches for v1; do not silently interpolate geometry.
- Build `UnstructuredData.from_array()` with supplied XYZ and sequential line
  connectivity, wrapped by `LineSet` where useful. Once the single-line
  interpretation is confirmed, each snapshot has 2,650 vertices and 2,649 segments.
- Keep `position` and the six measurement columns as vertex attributes. Preserve
  numeric `NaN` values and all 2,650 rows; never drop a vertex because strain is
  missing. No spatial interpolation is necessary for this example.
- Build connectivity once and reuse the same sample order for each export. Attach
  temporal metadata to each snapshot, then write four `.le` files and one index.
- Do not fabricate collars or route these files through `read_wells()`, which
  expects collars/survey/interval attributes and may resample the trajectory.
- For later conventional borehole datasets, keep geometry construction in the
  existing wells path, map measurements by well ID and measured depth onto one
  fixed trajectory, and explicitly define any required spatial resampling policy.

### 3. Volume Snapshot Export

- Keep the existing xarray representation with a named `time` dimension and
  explicit spatial dimensions, for example `("time", "x", "y", "z")`.
- Select one time using `isel(time=i, drop=True)` before wrapping/exporting the
  3D frame. Record the selected time as metadata before dropping its coordinate.
- Preserve active variable, structured type, dtype, explicit bounds, and other
  spatial metadata when constructing snapshots.
- Require consistent spatial axes, shape, coordinates/bounds, active field, and
  units across frames. Handle time coordinates with the same normalization rules.
- Exclude `time` from spatial bounds. Prefer calculating bounds on selected
  snapshots; do not broadly change non-spatial bounds behavior without tests.
- Continue using NetCDF for full in-memory time-series persistence. No new
  structured binary reader is required merely for export; test emitted headers
  and decode payload bytes directly in tests.

### 4. Liquid Earth Publication

Relevant paths are relative to the sibling `subsurface-le` repository.

- Add an explicit temporal import contract only after local exports work. Do not
  overload the existing three-file `ImportCSVWellsRequest` with these XYZ files.
- Review `subsurface_le/data_models/import_arguments_model/import_csv_wells.py`,
  `subsurface_le/subsurface_adapter/subsurface_interface.py`,
  `subsurface_le/importer.py`, and `subsurface_le/uploader_manager.py` for the
  smallest opt-in route supporting a frame collection and an index.
- Preserve existing static two-output borehole imports and their address behavior.
  Temporal imports must publish one logical series rather than patching the
  client-facing dataset address independently for each frame.
- Upload frames under an immutable import/revision prefix, upload the index last,
  and then publish the series address. A failed import must not replace the
  previous published index or advertise incomplete frames.
- Reuse existing job/publication identifiers for retry safety where possible.
  Account for source hashes and frame metadata in content identity. Define cleanup
  of failed/unreferenced revisions separately from successful publication.
- Review agent output policies/validation only if exposing this route through
  agent imports; do not change their existing fixed-node static-wells assumptions.

### 5. Client Selection and Interpolation

- Parse the index once, binary-search normalized timestamps, and fetch the
  surrounding two frames. Cache by decoded bytes, with bounded adjacent prefetch.
- Verify compatible geometry/sample order and attribute schemas before combining
  values. For volumes verify the spatial grid, not only the array shape.
- Interpolate continuous numeric measurements with
  `weight = (requested_time - time_a) / (time_b - time_a)` and
  `value = (1 - weight) * value_a + weight * value_b`.
- At an exact frame time, return that frame directly. If either endpoint sample
  is missing between frames, return missing; do not replace missing values with
  zero or bridge them implicitly.
- Do not extrapolate outside the series by default; show the boundary frame and
  identify the request as outside the observed range. Agree this UI behavior with
  the client team before release.
- Use step/nearest selection for categorical values. The supplied six example
  fields are numeric, but this must not become a blanket lithology policy.

## Capacity and Operational Safeguards

Azure documents no fixed blob-count limit per account. Request rate, bandwidth,
capacity, client memory, and import runtime are the relevant constraints.

- Do not encode a hard maximum frame count in the file/index schema.
- Consider a configurable warning above 1,000 frames per import; this is a
  provisional operational threshold, not an Azure limit or a finalized requirement.
- Estimate total output bytes and largest decoded frame before large imports;
  set any rejection thresholds from measured job/client budgets.
- Bound transfer concurrency and use the SDK retry/backoff facilities. Do not load
  all frames into memory during import or playback.
- For this four-frame example, duplication is small: XYZ plus line connectivity
  is about 53 KB per frame before attributes and headers.
- Revisit shared-geometry or chunked representations only after measuring a real
  storage, transfer, or request-overhead bottleneck.

References:

- https://learn.microsoft.com/en-us/azure/storage/common/scalability-targets-standard-account
- https://learn.microsoft.com/en-us/azure/storage/blobs/scalability-targets

## Verification and Acceptance Criteria

- Existing static structured hash tests and static borehole output tests pass
  unchanged. Unknown optional header fields must be tested with the actual client;
  compatibility with a strict client parser is not assumed.
- Temporal metadata survives unstructured binary round trips. Structured snapshot
  tests verify the header, shape, spatial bounds, dtype, and original body ordering.
- A small synthetic multi-frame fixture runs in CI without the external dataset.
  Test invalid/missing timezone, ambiguous timestamps, duplicates, unordered input,
  mismatched XYZ/positions/fields, invalid coordinates, and missing measurements.
- Keep the external dataset local and opt-in; do not copy it or generated binaries
  into the repository without permission. Document the command in the example.
- The full dataset exports exactly four snapshots and one index, each snapshot
  containing 2,650 vertices and 2,649 cells under the confirmed single-line model.
- All six measurements and their missing-value masks survive export within the
  existing stored dtype precision. Each strain field retains 244 NaNs per frame.
- Future client interpolation criterion: at the midpoint between the first two
  frames, sample position 0 has temperature
  approximately 11.95 degrees Celsius (12.0 and 11.9 endpoints), while missing
  strain remains missing. Exact timestamps return original values.
- Volume tests cover time-axis placement, explicit bounds, single-frame series,
  NetCDF coordinate preservation, and rejection of inconsistent spatial grids.
- Publication tests cover partial upload failure, retries, index-last ordering,
  unchanged static publication, and preservation of the previous published series.
- A client integration test demonstrates discovery, seeking, interpolation, and
  bounded caching without downloading the complete series.

## Delivery Order and Open Confirmations

1. Confirm source timezone, coordinate system, and single-trajectory interpretation
   for `2terranigma`; agree header/index names with the client.
2. Implement metadata preservation and optional structured metadata with focused
   tests. Keep static binary bytes unchanged.
3. Deliver the local borehole example and volume snapshot example with index
   generation; verify against the external four-file dataset.
4. Add opt-in backend publication and client time selection/interpolation.
5. Measure larger imports before choosing byte/runtime limits or optimizing away
   repeated geometry.

Remaining confirmations: the source meaning/reference of the change columns,
whether timestamp represents acquisition start/end or nominal frame time, source
data redistribution permission, and how the Liquid Earth series address should
reference the index. None should be silently inferred from the directory name.
