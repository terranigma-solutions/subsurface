"""Non-resampling translation and positive axis-aligned scaling of scalar LE grids."""

from pathlib import Path

import numpy as np

from subsurface.api._le_grid_ops import (
    AXIS_SPACING_RTOL, load_le_grid, validate_le_grid, write_le_grid,
)
from subsurface.core.structs.base_structures.structured_data import StructuredData


def transform_structured_grid(grid: StructuredData, matrix) -> StructuredData:
    """Return an independent grid with transformed float64 sample coordinates.

    Use homogeneous column vectors, ``p_out = matrix @ p_in``. Only finite real
    4x4 affine matrices with strictly positive diagonal scaling are supported.
    Rank-one ``dim0`` maps to X, rank-two to X/Y; unused axes must be identity
    with zero translation. Scalars, their exact dtype string, and name are
    copied unchanged, including categorical values and scalar NaN/Inf.

    Unsupported grid metadata/layouts, overflow, collapsed spacing, and sample
    positions incompatible with endpoint-derived LE reconstruction raise
    ValueError. Wrong grid or matrix element types raise TypeError. Neither
    input is mutated. No interpolation, coordinate snapping, or CRS change is
    performed; reconstruction tolerance is 1e-10 relative to axis spacing.
    """
    validate_le_grid(grid)
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4):
        raise ValueError("Matrix must have shape (4, 4)")
    if matrix.dtype.kind not in "iuf":
        raise TypeError("Matrix must contain real numeric values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Matrix must contain finite values")
    if not np.array_equal(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Matrix must have exact affine last row [0, 0, 0, 1]")
    linear = matrix[:3, :3]
    scale = np.diag(linear)
    if not np.array_equal(linear, np.diag(scale)) or np.any(scale <= 0):
        raise ValueError("Matrix linear part must be strictly positive diagonal scaling only")
    rank = grid.active_data_array.ndim
    if np.any(scale[rank:] != 1) or np.any(matrix[rank:3, 3] != 0):
        raise ValueError("Unused matrix axes must remain identity with zero translation")
    # Validate exact structural restrictions before conversion can round entries away.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        matrix = matrix.astype(np.float64)
    scale = np.diag(matrix[:3, :3])
    if not np.all(np.isfinite(matrix)) or np.any(scale <= 0):
        raise ValueError("Matrix scale and translation must be representable in float64")

    coords = {}
    for index, dim in enumerate(grid.active_data_array.dims):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            axis = grid.data.coords[dim].values.astype(np.float64) * scale[index] + matrix[index, 3]
            spacing = (axis[-1] - axis[0]) / (axis.size - 1) if axis.size > 1 else 0.0
            steps = np.diff(axis)
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"Transformed axis {dim} coordinates overflow float64")
        if axis.size > 1:
            if (not np.isfinite(spacing) or spacing <= 0
                    or not np.all(np.isfinite(steps)) or np.any(steps <= 0)):
                raise ValueError(f"Transformed axis {dim} spacing collapsed or overflowed")
            reconstructed = np.linspace(axis[0], axis[-1], axis.size)
            if np.any(np.abs(reconstructed - axis) > spacing * AXIS_SPACING_RTOL):
                raise ValueError(f"Transformed axis {dim} sample positions cannot be represented "
                                 "within the LE spacing-relative reconstruction tolerance")
        coords[dim] = axis

    result = StructuredData.from_numpy(
        grid.values.copy(), coords=coords, data_array_name=grid.active_data_array_name,
    )
    result.dtype = grid.dtype
    validate_le_grid(result)
    return result


def transform_structured_le(source, destination, matrix, *, overwrite=False) -> Path:
    """Transform a scalar LE grid into a new file and return its absolute Path.

    See :func:`transform_structured_grid` for transform restrictions. The parent
    directory must exist. Source aliases are always rejected, even with explicit
    overwrite. Shared LE safety helpers validate and read back a temporary file
    before atomic publication; existing destinations fail unless overwrite is
    True. Filesystem errors propagate and the source is never written.
    """
    grid = transform_structured_grid(load_le_grid(source), matrix)
    return write_le_grid(grid, destination, sources=[source], overwrite=overwrite)
