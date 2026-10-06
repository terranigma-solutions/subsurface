# Adjacent Structured LE Merge

```python
from subsurface.api.le_structured_merge import (
    merge_structured_grids, merge_structured_le,
)

output = merge_structured_le(
    ["left.le", "right.le"], "merged.le", axis="x", overwrite=False,
)
merged = merge_structured_grids([left_grid, right_grid], axis="x")
```

The file API returns an absolute `Path`. The in-memory API returns an independent
existing `StructuredData`, including for a single input. Sources are consumed in
caller order, never sorted. Empty inputs fail. Declare an existing standard axis:
rank one uses `dim0`, rank two uses `x,y`, and rank three uses `x,y,z`.

## Compatibility

Inputs must pass the [shared safety boundary](le_grid_ops_safety.md): one numeric
scalar array, ascending endpoint-reconstructible coordinates, no extra arrays or
unsupported metadata. Rank, dimension order, active-array name and scalar dtype
(including byte order) must match exactly. Scalar values are copied in input
order, without interpolation or conversion, including categorical integers,
signed/unsigned extremes, floating NaNs, infinities and signed zero. Output uses
the existing Fortran-order wire format; no provenance fields are added.

Nonmerge axes must have equal shapes and positions within `1e-10` times the
reference axis's endpoint-derived spacing. Nonmerge singleton positions must
match exactly: the merge-axis `spacing` argument does not establish spacing for
other axes.

## Spacing And Precision

For every nonsingleton merge axis, recover conceptual spacing as
`(last - first) / (size - 1)`, not from individual rounded increments. Use the
first recoverable spacing unless explicit `spacing` is supplied. All recovered
spacings must agree within `AXIS_SPACING_RTOL = 1e-10` times that spacing. Explicit
spacing must be a finite positive real scalar and agree with every recovered
spacing. All-singleton tiles require explicit spacing, even for a single source.
Mixed singleton/nonsingleton tiles use the recoverable spacing.

Each next first position must be strictly greater than the previous last and its
coordinate difference must equal one spacing within the same spacing-relative
tolerance. There is no origin-relative tolerance or absolute tolerance floor.
Overlaps, gaps, reversed order and changed resolution fail. If float64 rounding
at large origins makes the boundary difference disagree with the conceptual
spacing, adjacency cannot be established and the merge fails rather than guessing.
Reader-generated alternating positive increments within a tile remain valid;
this does not relax boundary checks. For example, a single endpoint-reconstructed
tile at origin `1e6` with span `0.3` is valid, but two such tiles may be rejected
because their rounded boundary difference cannot establish adjacency to `1e-10`
of their approximately `0.1` spacing. Large origins with exactly representable
spacing can merge normally.

The combined axis is reconstructed using `linspace(first, last, total_size)`.
Its spacing must agree with the established spacing, and every original position
must agree with reconstruction within `1e-10` times the smaller of the established
and combined spacings. No sample is snapped outside this bound. This global
check also refuses accumulated drift that passes individual boundary checks.
Tiny tolerances may underflow to zero, requiring exact agreement. Output receives
shared validation and file outputs receive serialization/read-back validation.

## Safety And Limits

Every source is passed to the atomic shared writer. Source/destination path,
symlink and hardlink aliases fail, even with explicit overwrite. Existing paths
require `overwrite=True`; publication replaces a directory entry rather than
writing through a symlink. Validation and staging failures leave all sources and
existing destinations unchanged. The destination parent must already exist.

No topology, CRS handling, resampling, overlap precedence, gap filling, nodata
inference or arbitrary mosaics are provided. Inputs and output arrays are loaded
in memory without a resource budget. The existing format cannot recover metadata
already discarded by another producer, nor identify nondefault array ordering.
Power-loss directory durability, hostile concurrent identity changes and real
production consumer qualification remain outside the shared safety contract.
