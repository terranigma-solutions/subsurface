"""Strict shared safety boundary for the existing scalar structured LE format."""

import os
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

from subsurface.core.structs.base_structures.structured_data import StructuredData, StructuredDataType


AXIS_SPACING_RTOL = 1e-10


def _axis_values(coordinate, dim):
    """Require float64-representable positions and spacing-relative regularity."""
    values = coordinate.values
    if values.dtype.metadata:
        raise ValueError(f"Axis {dim} dtype metadata would be lost")
    if coordinate.dims != (dim,) or values.dtype.kind not in "iuf":
        raise ValueError(f"Axis {dim} must be a one-dimensional real numeric coordinate")
    with np.errstate(over="ignore", invalid="ignore"):
        axis = values.astype(np.float64)
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"Axis {dim} coordinates must be finite float64 values")
    if values.dtype.kind in "iu":
        # NumPy mixed integer/float equality can itself round int64/uint64.
        exact = all(int(original) == int(converted) for original, converted in zip(values, axis))
    else:
        exact = np.array_equal(axis.astype(values.dtype), values)
    if not exact:
        raise ValueError(f"Axis {dim} coordinates lose precision in float64")
    if axis.size > 1:
        with np.errstate(over="ignore", invalid="ignore"):
            span = axis[-1] - axis[0]
            spacing = span / (axis.size - 1)
            steps = np.diff(axis)
        if (not np.isfinite(span) or not np.isfinite(spacing) or spacing <= 0
                or not np.all(np.isfinite(steps)) or np.any(steps <= 0)):
            raise ValueError(f"Axis {dim} must have positive finite ascending spacing")
        reconstructed = np.linspace(axis[0], axis[-1], axis.size)
        tolerance = spacing * AXIS_SPACING_RTOL
        if np.any(np.abs(reconstructed - axis) > tolerance):
            raise ValueError(f"Axis {dim} is not uniform within the spacing-relative tolerance")
    return axis


def validate_le_grid(grid: StructuredData) -> None:
    """Reject grids whose values, coordinates, or metadata would be lost on write.

    Only one scalar array, standard axes, and the current regular axis-aligned
    type are supported. No mutation or dtype coercion is performed. Singleton
    positions are retained but do not establish spacing. NaN/Inf scalar values
    are permitted; all coordinates must be finite and exactly representable in
    float64. Coordinate reconstruction error is bounded by 1e-10 of spacing.
    """
    if not isinstance(grid, StructuredData):
        raise TypeError("Expected StructuredData")
    if grid.type is not StructuredDataType.REGULAR_AXIS_ALIGNED:
        raise ValueError("Only regular axis-aligned structured grids are supported")
    if grid._bounds is not None:
        raise ValueError("Explicit bounds overrides are unsupported")
    if set(vars(grid)) - {"data", "_active_data_array_name", "type", "dtype", "_bounds"}:
        raise ValueError("Unsupported StructuredData metadata would be lost")
    if not isinstance(grid.data, xr.Dataset):
        raise TypeError("StructuredData.data must be an xarray Dataset")
    name = grid.active_data_array_name
    array = grid.active_data_array
    dims = tuple(StructuredData._default_dim_names(array.ndim))
    if not 1 <= array.ndim <= 3 or array.dims != dims or any(size == 0 for size in array.shape):
        raise ValueError("Grid must have one to three nonempty standard axes in standard order")
    if not isinstance(name, str) or not name.strip() or name in dims:
        raise ValueError("Active data name must be a nonempty string distinct from axis names")
    try:
        name.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Active data name must be valid UTF-8") from exc
    if set(grid.data.data_vars) != {name} or set(grid.data.coords) != set(dims):
        raise ValueError("Extra arrays/coordinates or missing axis coordinates are unsupported")
    if (grid.data.attrs or grid.data.encoding or array.attrs or array.encoding
            or any(coord.attrs or coord.encoding for coord in grid.data.coords.values())):
        raise ValueError("Dataset, array, and coordinate metadata/encoding would be lost")
    try:
        if not isinstance(grid.dtype, str):
            raise TypeError("dtype must be a serialized dtype string")
        dtype = np.dtype(grid.dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid serialized numeric dtype") from exc
    if not (dtype.kind in "iu" and dtype.itemsize in (1, 2, 4, 8)
            or dtype.kind == "f" and dtype.itemsize in (2, 4, 8)):
        raise ValueError("Unsupported scalar numeric dtype")
    if array.dtype.metadata:
        raise ValueError("Scalar dtype metadata would be lost")
    if dtype != array.dtype:
        raise ValueError("grid.dtype must match the active scalar array dtype exactly; no coercion allowed")
    for dim in dims:
        _axis_values(grid.data.coords[dim], dim)


def load_le_grid(source) -> StructuredData:
    """Read a current F-order scalar LE file and preflight reconstructed axes."""
    grid = StructuredData.from_binary_le(source)
    validate_le_grid(grid)
    return grid


def write_le_grid(grid: StructuredData, destination, *, sources, overwrite=False) -> Path:
    """Atomically write and read-back-verify a grid, returning an absolute Path.

    Supply every input path in ``sources`` (possibly empty for generated data).
    Path, symlink, and hardlink source aliases are always rejected. No-clobber
    publication uses a hardlink; overwrite uses replacement. The parent must
    exist. No non-atomic fallback is provided. Hostile concurrent changes to
    source/directory identity are outside this contract.
    """
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    if isinstance(sources, (str, bytes, os.PathLike)):
        raise TypeError("sources must be an iterable of source paths")
    destination = Path(destination).absolute()
    source_paths = tuple(Path(source).absolute() for source in sources)

    def check_destination():
        for source in source_paths:
            if destination.resolve() == source.resolve() or (
                    destination.exists() and source.exists() and os.path.samefile(destination, source)):
                raise ValueError("Destination must not alias any source path")
        if not overwrite and os.path.lexists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")

    check_destination()
    validate_le_grid(grid)
    binary = grid.to_binary(order="F")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            if stream.write(binary) != len(binary):
                raise OSError("Incomplete temporary LE grid write")
            stream.flush()
            os.fsync(stream.fileno())
        restored = load_le_grid(temporary)
        if (restored.active_data_array_name != grid.active_data_array_name
                or restored.values.dtype != grid.values.dtype
                or restored.shape != grid.shape
                or not np.array_equal(restored.values, grid.values, equal_nan=True)):
            raise ValueError("Written grid scalar values, dtype, shape, or active name changed")
        for dim in grid.active_data_array.dims:
            original = _axis_values(grid.data.coords[dim], dim)
            actual = _axis_values(restored.data.coords[dim], dim)
            tolerance = (0.0 if original.size == 1 else
                         (original[-1] - original[0]) / (original.size - 1) * AXIS_SPACING_RTOL)
            if np.any(np.abs(actual - original) > tolerance):
                raise ValueError(f"Written axis {dim} coordinates changed")
        check_destination()
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return destination
