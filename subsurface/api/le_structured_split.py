"""Exact rectangular index-window selection for scalar structured LE grids."""

from collections.abc import Mapping
from numbers import Integral
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import numpy as np

from subsurface.api._le_grid_ops import load_le_grid, validate_le_grid, write_le_grid
from subsurface.core.structs.base_structures.structured_data import StructuredData


def split_structured_grid(grid: StructuredData, ranges) -> StructuredData:
    """Copy one half-open index window without squeezing or resampling.

    ``ranges`` maps standard axis names to integer ``(start, stop)`` pairs.
    Omitted axes retain all samples. Metadata unsupported by the scalar LE
    format is rejected, rather than dropped. The input is never mutated.
    """
    validate_le_grid(grid)
    if not isinstance(ranges, Mapping):
        raise TypeError("Window ranges must be an axis-to-range mapping")
    dims = grid.active_data_array.dims
    if any(axis not in dims for axis in ranges):
        raise ValueError("Window contains an unknown axis")
    slices = []
    coords = {}
    for axis, size in zip(dims, grid.shape):
        bounds = ranges.get(axis, (0, size))
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("Each axis range must be a (start, stop) pair without a step")
        start, stop = bounds
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in bounds):
            raise TypeError("Window bounds must be integers, not booleans")
        if start < 0 or stop <= start or stop > size:
            raise ValueError(f"Window range for {axis} must satisfy 0 <= start < stop <= {size}")
        selection = slice(int(start), int(stop))
        slices.append(selection)
        coords[axis] = grid.data.coords[axis].values[selection].copy()
    result = StructuredData.from_numpy(
        grid.values[tuple(slices)].copy(), coords=coords,
        data_array_name=grid.active_data_array_name,
    )
    result.dtype = grid.dtype
    validate_le_grid(result)
    return result


def split_structured_le(source, output_directory, *, windows) -> Dict[str, Path]:
    """Write ordered independent windows and return labels to absolute paths.

    Output names are ``grid_000000.le``, etc., never caller labels. The output
    directory must exist and all destinations must be new and not source aliases.
    An empty mapping returns ``{}`` with no outputs after source validation.
    All windows are preflighted and staged with verified readback before any
    publication. Rollback removes only owned inodes, not racers. Multi-file
    publication is not crash atomic; see ``docs/le_structured_split.md``.
    """
    source = Path(source).absolute()
    directory = Path(output_directory).absolute()
    if not directory.is_dir():
        raise NotADirectoryError(f"Output directory must already exist: {directory}")
    if not isinstance(windows, Mapping):
        raise TypeError("windows must be an ordered label-to-window mapping")
    grid = load_le_grid(source)
    destinations = {}
    outputs = []

    def check_destination(destination):
        if destination.resolve() == source.resolve() or (
                destination.exists() and os.path.samefile(destination, source)):
            raise ValueError("Destination must not alias the source path")
        if os.path.lexists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")

    for ordinal, (label, ranges) in enumerate(windows.items()):
        if not isinstance(label, str) or not label:
            raise ValueError("Window labels must be nonempty strings")
        destination = directory / f"grid_{ordinal:06d}.le"
        check_destination(destination)
        output = split_structured_grid(grid, ranges)
        destinations[label] = destination
        outputs.append((destination, output))
    if not outputs:
        return {}

    created = []
    try:
        with TemporaryDirectory(dir=directory, prefix=".le_structured_split.") as staging:
            staged = []
            for destination, output in outputs:
                path = Path(staging) / destination.name
                write_le_grid(output, path, sources=(source,))
                stat = path.stat()
                staged.append((destination, path, stat.st_dev, stat.st_ino))
            for destination, path, device, inode in staged:
                check_destination(destination)
                # A link may publish successfully and then raise; track ownership first.
                created.append((destination, device, inode))
                os.link(path, destination)
    except BaseException as error:
        cleanup_error = None
        for destination, device, inode in reversed(created):
            try:
                stat = destination.lstat()
                if (stat.st_dev, stat.st_ino) == (device, inode):
                    destination.unlink()
            except FileNotFoundError:
                pass
            except OSError as failure:
                cleanup_error = failure
        if cleanup_error is not None:
            raise cleanup_error from error
        raise
    return destinations
