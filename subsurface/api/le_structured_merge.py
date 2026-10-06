"""Concatenate adjacent scalar structured LE tiles without resampling."""

import os
from numbers import Real
from pathlib import Path

import numpy as np

from subsurface.api._le_grid_ops import (
    AXIS_SPACING_RTOL, load_le_grid, validate_le_grid, write_le_grid,
)
from subsurface.core.structs.base_structures.structured_data import StructuredData


def merge_structured_grids(grids, *, axis, spacing=None) -> StructuredData:
    """Return an independent grid concatenated in caller order along ``axis``.

    Require matching standard dimensions, active name, exact scalar dtype and
    nonmerge coordinates. Infer spacing from nonsingleton endpoint spans; all
    singleton inputs require explicit positive spacing, even for one input.
    Coordinate errors are bounded by ``AXIS_SPACING_RTOL`` times spacing, never
    world magnitude. Ambiguous rounded boundary differences are rejected.
    Unsupported metadata and extra arrays fail shared LE preflight validation.
    """
    grids = tuple(grids)
    if not grids:
        raise ValueError("At least one grid is required")
    for grid in grids:
        validate_le_grid(grid)
    reference = grids[0]
    dims = reference.active_data_array.dims
    if not isinstance(axis, str):
        raise TypeError("axis must be a standard axis name")
    if axis not in dims:
        raise ValueError("axis must name an existing standard grid dimension")
    if spacing is not None:
        if isinstance(spacing, (bool, np.bool_)) or not isinstance(spacing, Real):
            raise TypeError("spacing must be a real numeric scalar")
        try:
            spacing = float(spacing)
        except OverflowError as exc:
            raise ValueError("spacing must be finite and positive") from exc
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")

    axes = []
    recovered = []
    for grid in grids:
        if (grid.active_data_array.dims != dims
                or grid.active_data_array_name != reference.active_data_array_name
                or grid.values.dtype != reference.values.dtype):
            raise ValueError("Grid rank, dimension order, active name and scalar dtype must match exactly")
        for dim in dims:
            if dim == axis:
                continue
            expected = reference.data.coords[dim].values.astype(np.float64)
            actual = grid.data.coords[dim].values.astype(np.float64)
            tolerance = (0.0 if expected.size == 1 else
                         (expected[-1] - expected[0]) / (expected.size - 1) * AXIS_SPACING_RTOL)
            with np.errstate(over="ignore", invalid="ignore"):
                if actual.shape != expected.shape or np.any(np.abs(actual - expected) > tolerance):
                    raise ValueError(f"Nonmerge axis {dim} coordinates must coincide")
        positions = grid.data.coords[axis].values.astype(np.float64)
        axes.append(positions)
        if positions.size > 1:
            recovered.append((positions[-1] - positions[0]) / (positions.size - 1))
    if spacing is None:
        if not recovered:
            raise ValueError("All singleton merge axes require explicit spacing")
        spacing = recovered[0]
    tolerance = spacing * AXIS_SPACING_RTOL
    if any(abs(actual - spacing) > tolerance for actual in recovered):
        raise ValueError("Merge-axis spacings must agree within the spacing-relative tolerance")
    for previous, following in zip(axes, axes[1:]):
        with np.errstate(over="ignore", invalid="ignore"):
            difference = following[0] - previous[-1]
        if not np.isfinite(difference) or difference <= 0 or abs(difference - spacing) > tolerance:
            raise ValueError("Tiles must be adjacent in caller order by one spacing; ambiguous boundaries fail")

    supplied = np.concatenate(axes)
    reconstructed = np.linspace(supplied[0], supplied[-1], supplied.size)
    if supplied.size > 1:
        with np.errstate(over="ignore", invalid="ignore"):
            combined_spacing = (supplied[-1] - supplied[0]) / (supplied.size - 1)
            errors = np.abs(reconstructed - supplied)
        if (not np.isfinite(combined_spacing) or combined_spacing <= 0
                or abs(combined_spacing - spacing) > tolerance
                or np.any(errors > min(spacing, combined_spacing) * AXIS_SPACING_RTOL)):
            raise ValueError("Combined coordinates cannot reconstruct without moving samples beyond tolerance")

    index = dims.index(axis)
    shape = list(reference.shape)
    shape[index] = supplied.size
    # Explicit allocation preserves nonnative byte order as well as integer bits.
    values = np.empty(shape, dtype=reference.values.dtype)
    start = 0
    for grid in grids:
        selection = [slice(None)] * len(dims)
        stop = start + grid.shape[index]
        selection[index] = slice(start, stop)
        values[tuple(selection)] = grid.values
        start = stop
    coords = {dim: (reconstructed if dim == axis else reference.data.coords[dim].values.copy())
              for dim in dims}
    result = StructuredData.from_numpy(values, coords=coords,
                                       data_array_name=reference.active_data_array_name)
    result.dtype = values.dtype.str
    validate_le_grid(result)
    return result


def merge_structured_le(sources, destination, *, axis, spacing=None, overwrite=False) -> Path:
    """Merge adjacent LE files into an atomically published absolute output path.

    Source order is retained; every source is protected against destination
    aliases, including symlinks/hardlinks and explicit overwrite. Existing output
    paths require ``overwrite=True``. No metadata or provenance is invented.
    See ``merge_structured_grids`` for alignment and singleton spacing policy.
    """
    if isinstance(sources, (str, bytes, os.PathLike)):
        raise TypeError("sources must be an iterable of source paths")
    sources = tuple(sources)
    if not sources:
        raise ValueError("At least one source is required")
    result = merge_structured_grids((load_le_grid(source) for source in sources),
                                    axis=axis, spacing=spacing)
    return write_le_grid(result, destination, sources=sources, overwrite=overwrite)
