"""Split unstructured LiquidEarth files by explicit numeric object IDs."""

from copy import deepcopy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Union

import numpy as np

from subsurface.api._le_file_ops import load_le_mesh, serialize_le_mesh, write_le_mesh
from subsurface.api.le_inspection import group_object_ids
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


def split_le(source, output_directory, *, object_attribute: str,
             association: str) -> Dict[Union[int, float], Path]:
    """Write one new file per sorted original ID; return ID-to-absolute-Path mapping.

    Use cell association for meshes and point association for point clouds.
    The directory must exist; outputs named ``object_000000.le`` etc. must not
    exist or alias the source. All outputs are preflighted and serialized before
    publication. On publication failure, files created by this call are removed.
    This is not a crash-atomic multi-file transaction. See ``docs/le_split.md``
    for point connectivity, precision, and provenance policies.
    """
    source = Path(source).absolute()
    directory = Path(output_directory).absolute()
    if not directory.is_dir():
        raise NotADirectoryError(f"Output directory must already exist: {directory}")
    mesh = load_le_mesh(source)
    width = mesh.cells.shape[1]
    frame = mesh.points_attributes if association == 'point' else mesh.attributes
    ids = group_object_ids(frame, object_attribute=object_attribute,
                           association=association, cell_width=width)
    if not len(ids):
        return {}
    if width == 0 and mesh.cells.shape[0] not in (0, len(mesh.vertex)):
        raise ValueError("Zero-width point connectivity must have zero rows or one row per point")
    if 'le_tools' in mesh.data_attrs and not isinstance(mesh.data_attrs['le_tools'], dict):
        raise ValueError("Reserved le_tools metadata must be a dictionary")

    destinations = {
        object_id.item(): directory / f'object_{ordinal:06d}.le'
        for ordinal, object_id in enumerate(ids)
    }
    for destination in destinations.values():
        if destination.resolve() == source.resolve() or (
                destination.exists() and os.path.samefile(destination, source)):
            raise ValueError("Destination must not alias the source path")
        if os.path.lexists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")

    outputs = []
    values = frame[object_attribute].to_numpy()
    for object_id, destination in destinations.items():
        selected = np.flatnonzero(values == object_id)
        if association == 'cell':
            cell_rows = selected
            point_rows, inverse = np.unique(mesh.cells[cell_rows], return_inverse=True)
            cells = inverse.reshape(len(cell_rows), width).astype(np.int32)
        else:
            point_rows = selected
            if width == 0:
                cell_rows = selected if len(mesh.cells) else np.empty(0, dtype=np.intp)
                cells = np.empty((len(cell_rows), 0), dtype=np.int32)
            else:
                cell_rows = np.flatnonzero(np.isin(mesh.cells[:, 0], point_rows))
                cells = np.searchsorted(point_rows, mesh.cells[cell_rows]).astype(np.int32)
        metadata = deepcopy(mesh.data_attrs)
        provenance = dict(operation='split', source=str(source.resolve()),
                          object_attribute=object_attribute, association=association,
                          object_id=object_id)
        if 'le_tools' in metadata:
            provenance['previous'] = metadata['le_tools']
        metadata['le_tools'] = provenance
        output = LiquidEarthMesh(
            vertex=mesh.vertex[point_rows], cells=cells,
            attributes=mesh.attributes.iloc[cell_rows].reset_index(drop=True),
            points_attributes=mesh.points_attributes.iloc[point_rows].reset_index(drop=True),
            data_attrs=metadata,
        )
        # Complete schema/precision validation for every object before publishing any.
        serialize_le_mesh(output)
        outputs.append((destination, output))

    created = []
    try:
        with TemporaryDirectory(dir=directory, prefix='.le_split.') as staging:
            staged = []
            for destination, output in outputs:
                path = Path(staging) / destination.name
                write_le_mesh(output, path, sources=(source,))
                stat = path.stat()
                staged.append((destination, path, stat.st_dev, stat.st_ino))
            for destination, path, device, inode in staged:
                if destination.resolve() == source.resolve() or (
                        destination.exists() and os.path.samefile(destination, source)):
                    raise ValueError("Destination must not alias the source path")
                # Track the known staged inode before linking: publication can
                # succeed even if the link call then raises. Racers never match.
                created.append((destination, device, inode))
                os.link(path, destination)
    except BaseException:
        for destination, device, inode in reversed(created):
            try:
                stat = destination.lstat()
                if (stat.st_dev, stat.st_ino) == (device, inode):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    return destinations
