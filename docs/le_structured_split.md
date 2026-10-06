# Structured LE Index Split

Import directly from `subsurface.api.le_structured_split`:

```python
from subsurface.api.le_structured_split import split_structured_le, split_structured_grid

outputs = split_structured_le(
    "source.le", "existing-output-directory",
    windows={"left": {"x": (0, 2)}, "overlap": {"x": (1, 3)}},
)
```

`split_structured_le(source, output_directory, *, windows)` returns a dictionary
mapping caller labels to absolute `Path` objects in caller iteration order.
The directory must already exist. Labels must be nonempty strings; even path-like
labels never enter filenames. Outputs are deterministic ordinal names
`grid_000000.le`, `grid_000001.le`, and so on. Existing files and dangling
symlinks are rejected; there is no overwrite option. An empty windows mapping
returns `{}` and creates no outputs, but still requires a valid source and
existing output directory.

Each window is a mapping from standard axes (`dim0` for rank one, `x,y` for rank
two, `x,y,z` for rank three) to half-open integer `(start, stop)` pairs. Tuple
and list pairs and NumPy integer bounds are accepted. Omitted axes select all
samples; an empty window selects the whole grid. Unknown axes, booleans,
nonintegers, negative indices, steps, empty ranges, and out-of-bounds ranges fail.
Overlapping windows are independent explicit selections, not an implicit
partition. There is no scalar-ID grouping, geometric clipping, or resampling.

`split_structured_grid(grid, ranges)` returns one independently copied
`StructuredData` window. Both APIs retain scalar values exactly, including
integer extremes, floating NaNs/infinities, dtype/byte order, active-array name,
standard axis order, and rank. Singleton selections are never squeezed and keep
their sample position. Bounds describe inclusive sample extrema, not voxel edges.
Serialization reconstructs nonsingleton coordinates within the shared
spacing-relative tolerance of `1e-10`; singleton coordinates are exact. A
subwindow that cannot meet that strict reconstruction contract is rejected,
not silently regularized. Inputs remain unchanged.

The shared [safety boundary](le_grid_ops_safety.md) rejects unsupported in-memory
metadata, extra arrays, and coordinate layouts. Existing scalar files cannot
carry provenance attrs, and discarded metadata cannot be recovered. This split
adds no wire fields or CRS/provenance metadata. Keep the returned label-to-path
mapping and caller window specification if that relationship is needed later.

## Publication And Limits

All windows, destinations, and source aliases (path, symlink, hardlink) are
preflighted before writing. Every output is staged in a temporary directory on
the destination filesystem using the shared writer, which validates actual
temporary-file readback, exact scalar values/dtype/name/shape, and coordinates.
Only after all staging succeeds are outputs published by atomic no-clobber
hardlinks. Source-alias and collision checks repeat immediately before each
publication; hardlinks protect against destinations appearing after that check.
There is no fallback for filesystems without hardlink support.

On exceptions, rollback deletes only destination entries whose device/inode
matches the owned staged file captured before publication. This includes a link
that succeeds before raising, while preserving unrelated files and concurrent
racers. Failed-publication rollback runs while all staging files remain alive,
so an unpublished inode cannot be recycled before its destination is checked.
These ownership records are consumed before staging cleanup, even if rollback
fails; they are never retried after their inode references are released. If
staging cleanup fails after successful publication, the published links retain
the inode references and those outputs are rolled back too.
Rollback attempts all owned destinations even if one cleanup fails;
a filesystem cleanup error is raised chained from the triggering error.
Temporary-directory cleanup is attempted on success and failure.

This is **not a crash-atomic multi-file transaction**: observers can see a
published prefix, and abrupt termination can leave partial outputs or staging
files. Permissions, filesystem failures, or concurrent entry replacement can
prevent cleanup. Hostile concurrent source/directory identity changes and the
race between rollback identity check and unlink are outside the contract.
Directory entries are not fsynced, so power-loss durability is not guaranteed.
Full arrays and copied windows are held in memory with no resource budget.
Tests use synthetic offline fixtures, not production consumer qualification.
