# Time Series Usage

Local export supports fixed-geometry trajectory measurements and volume values.
Each observation becomes an ordinary `.le` snapshot; `series.json` groups the
frames. There is no Azure/backend integration or runtime interpolation yet.

## Snapshot API

`export_time_series()` accepts an iterable of `(timestamp, snapshot)` pairs.
Timestamps must be timezone-aware Python datetimes or ISO strings with an offset
or `Z`. Use `kind="trajectory"` for `UnstructuredData` line meshes or
`kind="volume"` for already-sliced, three-dimensional `StructuredData` snapshots.

```python
from subsurface.modules.writer.time_series import export_time_series

# snapshot_a and snapshot_b are existing fixed-geometry snapshots.
frames = [
    ("2025-12-12T10:45:39+00:00", snapshot_a),
    ("2025-12-12T11:45:39+00:00", snapshot_b),
]
index_path = export_time_series(
    frames, "output/measurements-new",
    time_series_id="measurements", kind="trajectory",
)
```

Use a stable ID starting with a letter or digit and containing only letters,
digits, `_`, `.`, or `-`. Frames must share geometry, sample order, position,
attribute schemas, and units; volume grids/bounds must also agree. Trajectory
identity attributes `well_id`, `measured_depths`, and `is_attr_point`, and the
`well_id_mapper` metadata, cannot change across trajectory frames. Measurement
values may change, and numeric NaNs remain missing rather than becoming zero.
An unsliced `time` dimension, duplicate timestamps, or incompatible frames are
rejected. Input objects are not modified.

## Volume API

`export_volume_time_series()` accepts `StructuredData` whose active field has a
named `time` dimension and coordinate, plus three spatial dimensions. It selects
each time before export. The time dimension may appear anywhere, but after
dropping it the spatial dimensions must be exactly `("x", "y", "z")` in that
order. The exporter retains the active field, output dtype, spatial metadata,
and explicit bounds.

```python
from subsurface.modules.writer.time_series import export_volume_time_series

# volume_series is existing StructuredData with a time coordinate.
index_path = export_volume_time_series(
    volume_series, "output/volumes-new",
    time_series_id="temperature", source_timezone="Europe/Berlin",
)
```

The timezone above is illustrative, not a claim about any source dataset.
Confirm the source timezone before using it. Naive time coordinates require an
explicit `source_timezone` (timezone name or tzinfo); aware coordinates already
identify their timezone. Times normalize to UTC. Ambiguous or nonexistent local
times are rejected rather than guessed. The snapshot API does not localize
naive timestamps for you.

## CLI Examples

Run from the repository root with `subsurface` installed. Choose output
directories that do not already exist, even as empty directories; the exporter
creates them and returns the index path. It refuses to overwrite existing
directories and removes its newly created output directory on export failure.

```bash
python examples/time_series_boreholes.py \
  /home/leguark/DevOps/SubsurfaceTestData/boreholes/2terranigma \
  output/boreholes-new \
  --source-timezone Europe/Berlin --single-trajectory \
  --time-series-id 2terranigma-strain-temperature

python examples/time_series_volumes.py output/volumes-new \
  --time-series-id synthetic-temperature
```

`Europe/Berlin` is only an example: the external CSV filenames do not establish
a timezone. The borehole command requires both a confirmed source timezone and
`--single-trajectory`, affirming that sequential samples form one connected line.
These are explicit XYZ trajectory datasets, not collar/survey inputs: no well
IDs, collars, survey angles, CRS, or change-column reference are inferred. The
example does not use `read_wells()`, resample geometry, or fabricate collars.
It retains `position` and all six measurements as vertex attributes, including
rows with missing strain. XYZ and strictly increasing position must be finite
and identical across frames. Filenames must follow
`YYYY-MM-DD_HH-MM-SS_strain_and_temperature.csv`.

The volume CLI creates three synthetic frames and passes `source_timezone="UTC"`
explicitly for its deliberately naive synthetic times; it takes no timezone CLI
option. This says nothing about the external CSV source timezone.

## Outputs and Clients

- Each output directory contains `trajectory_0000.le`, etc., or `volume_0000.le`,
  etc., and `series.json`. Geometry is repeated in each trajectory snapshot.
- The index contains `schema_version: 1`, `time_series_id`, `kind`, and `frames`
  with normalized UTC `timestamp` and relative `path`. Entries are sorted by
  time; filenames are numbered in input order, not necessarily time order.
- Snapshot JSON headers carry matching `xarray_attrs.time_series_id` and
  `xarray_attrs.timestamp`. Dataset metadata must be JSON-safe; measurement NaNs
  are preserved in payloads, but non-finite metadata numbers are rejected.
- No new binary payload version or 4D volume payload is introduced. Static
  structured output remains unchanged, and unstructured readers restore header
  metadata. See [LE Specification](le_spec.md). Compatibility with strict client
  parsers of optional headers must still be tested with the actual client.
- Clients need explicit support for index discovery and grouping snapshots into
  one logical series. Local files alone do not enable time controls, playback,
  interpolation, caching, or Azure publication. Those remain planned work in
  [Time Series Implementation Plan](time_series_implementation_plan.md).
