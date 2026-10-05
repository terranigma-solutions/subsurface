# Structured LE Operation Safety

Restricted Task7 operations share three internal helpers in
`subsurface.api._le_grid_ops`. They use the existing `StructuredData` container
and scalar wire format, without public exports or a new container. Task8 is out
of scope. Transform, rectangular split, and aligned merge implementations are
separate consumers of this boundary; these helpers do not perform operations.

```python
load_le_grid(source) -> StructuredData
validate_le_grid(grid: StructuredData) -> None
write_le_grid(grid: StructuredData, destination, *, sources, overwrite=False) -> Path
```

## Preflight Contract

Loading uses `StructuredData.from_binary_le`, then validates the reconstructed
grid. Writing validates the input without mutation. Unsupported or ambiguous
grids raise `ValueError`; wrong container/argument types raise `TypeError` and
filesystem errors propagate.

- Only `REGULAR_AXIS_ALIGNED`, one scalar array, and ranks 1-3 are supported.
- Axes must be exactly `dim0`, `x,y`, or `x,y,z`, in that order, with positive
  sizes and explicit one-dimensional coordinates. Auxiliary coordinates,
  additional arrays, custom dimensions, and explicit bounds overrides fail.
- Dataset, array, and coordinate attrs, encodings, and NumPy dtype metadata must
  be empty. Additional instance metadata fails too: CRS, units, nodata, or other
  semantics cannot be silently discarded. No provenance is added to the existing
  wire format.
- The active name must be a nonblank UTF-8 string distinct from axis names.
- Signed/unsigned 8/16/32/64-bit integers and 16/32/64-bit floats are accepted.
  `grid.dtype` must be a string describing exactly the scalar array dtype,
  including byte order. Bool, complex, object, and other scalar types fail.
  There is no coercion, float32 downcast, resampling, or invented nodata.
- Scalar NaN and infinities are retained. Integer values, including int64 and
  uint64 extremes, retain their dtype and exact values.

`StructuredData.from_numpy` defaults `dtype` to `float32` regardless of the input
array. Operation implementations must explicitly set the field, for example:

```python
result = StructuredData.from_numpy(values, coords=coords, data_array_name=name)
result.dtype = values.dtype.str
validate_le_grid(result)
```

## Coordinates And Tolerance

Bounds are inclusive coordinate extrema, not outer voxel edges. Positions must
be finite and exactly representable as float64. Coordinates must be strictly
ascending with finite positive spacing. Singleton axes preserve their position
exactly, but provide no evidence of sample spacing or cell width.

For a nonsingleton axis, let `s = (last - first) / (size - 1)` be the theoretical
endpoint spacing. Every adjacent step must be finite and positive, and every
coordinate must differ from the reader's `linspace(first, last, size)` by at most
`s * 1e-10`. Individual steps are not compared to `s`: float64 rounding can
produce alternating increments even when the coordinates reconstruct exactly,
for example `linspace(1e6, 1e6 + 0.3, 4)`. Regularity means compatibility with this
endpoint-derived reconstruction, not exact equality of rounded increments.
There is no absolute tolerance floor or tolerance relative to world magnitude.
`AXIS_SPACING_RTOL` is `1e-10`. A tiny origin-relative error that is substantial
relative to spacing is rejected. Reader-generated linspace axes are also checked
for finite positive steps; large origins with collapsed increments are refused,
but unequal positive increments caused by reconstruction rounding are accepted.
Very small spacing may underflow the tolerance to zero, requiring exactness.

Read-back coordinates are compared to the input using the same spacing-relative
bound; singletons use exact equality. Float32-originated coordinates with
sample-scale rounding irregularity can be rejected. This is intentional rather
than assuming uniformity or silently regularizing an ambiguous grid.

## Publication Safety

`sources` must be an iterable containing every input path, including inputs whose
values are not retained; use an empty iterable only for generated data. The
destination returned is an absolute `Path`; its parent must already exist.
Path, resolved symlink, and hardlink aliases to any source are always refused,
even with `overwrite=True`. Checks run before work and immediately before
publication. Existing destinations, including dangling symlinks, fail by default.

Writing uses the existing serializer in Fortran order. A context-managed
temporary in the destination directory is checked for short writes, flushed,
fsynced, and closed. Before publication, the actual temporary is read through
`StructuredData.from_binary_le` and preflighted. Its name, shape, scalar dtype,
and scalar values (NaNs compare equal) must match the input; coordinates must
match within the tight tolerance above.

Default publication is an atomic no-clobber hardlink, protecting against a
destination created after preflight. Explicit overwrite uses atomic replacement
of the destination directory entry, not a write through a destination symlink.
There is no non-atomic fallback if links are unavailable. Temporaries are removed
on success or failure, and sources are never opened for writing.

## Limits

The helpers load/serialize full arrays and do not impose resource budgets.
Multi-output transactions, overlap/gap policy, spacing inference for singletons,
rotation/shear, resampling, and merge compatibility remain operation-level work.
Original discarded metadata and nondefault array order cannot be recovered.
The format has no order marker, so a nondefault-order producer cannot be detected.

Atomic publication does not guarantee power-loss durability of the directory
entry (the directory is not fsynced). Hostile concurrent source/directory identity
changes and concurrent mutation of the in-memory grid are outside the contract.
Tests are synthetic and offline; production consumer qualification still requires
representative exports. No common class, export, or unstructured helper changes
are part of this implementation.
