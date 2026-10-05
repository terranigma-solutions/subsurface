"""Strict, ordered concatenation of compatible unstructured LiquidEarth files."""

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from subsurface.api._le_file_ops import load_le_mesh, serialize_le_mesh, write_le_mesh
from subsurface.api.le_inspection import group_object_ids
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


@dataclass(frozen=True)
class LEMergeResult:
    """Published destination and JSON-native ID mapping (None without grouping)."""

    destination: Path
    id_mapping: Optional[List[Dict[str, Any]]]


def merge_meshes(meshes, *, object_attribute=None, association=None,
                 id_policy="source", sources=None) -> LiquidEarthMesh:
    """Concatenate meshes without welding, coercing schemas, or mutating inputs.

    Grouping requires both arguments and follows inspection's association rules.
    ``source`` assigns consecutive int64 IDs in source order, then sorted original
    ID order; ``shared`` explicitly preserves globally shared IDs. Original exact
    Python numeric values and nested source provenance are kept under ``le_tools``.
    Optional sources are path labels, one per mesh. See docs/le_merge.md.
    """
    if (object_attribute is None) != (association is None):
        raise ValueError("object_attribute and association must be supplied together")
    if id_policy not in ("source", "shared"):
        raise ValueError("id_policy must be 'source' or 'shared'")
    if object_attribute is None and id_policy != "source":
        raise ValueError("id_policy='shared' requires explicit grouping")
    meshes = tuple(meshes)
    if not meshes:
        raise ValueError("At least one source is required")
    if any(not isinstance(mesh, LiquidEarthMesh) for mesh in meshes):
        raise TypeError("Expected LiquidEarthMesh inputs")
    if sources is None:
        paths = [None] * len(meshes)
    else:
        if isinstance(sources, (str, bytes, os.PathLike)):
            raise TypeError("sources must be an iterable of paths")
        paths = [Path(source).absolute() for source in sources]
        if len(paths) != len(meshes):
            raise ValueError("sources must contain one path per mesh")

    # Count before concatenation or serialization, so index overflow never wraps.
    total_vertices = sum(mesh.vertex.shape[0] for mesh in meshes)
    if total_vertices > np.iinfo(np.int32).max + 1:
        raise ValueError("Merged vertex count exceeds int32 connectivity capacity")
    first = meshes[0]
    width = first.cells.shape[1]
    metadata = {key: value for key, value in first.data_attrs.items() if key != "le_tools"}
    metadata_key = json.dumps(metadata, sort_keys=True)
    groups = []
    for mesh in meshes:
        if mesh.cells.shape[1] != width:
            raise ValueError("Sources must have identical geometry/connectivity widths")
        semantic = {key: value for key, value in mesh.data_attrs.items() if key != "le_tools"}
        if json.dumps(semantic, sort_keys=True) != metadata_key:
            raise ValueError("Source semantic metadata (including CRS/units) conflicts")
        for label, frame, reference in (
                ("cell", mesh.attributes, first.attributes),
                ("point", mesh.points_attributes, first.points_attributes)):
            if (not frame.columns.equals(reference.columns) or
                    not frame.dtypes.equals(reference.dtypes)):
                raise ValueError(f"Sources must have identical {label} attribute names/order and dtypes")
        if object_attribute is not None:
            frame = mesh.attributes if association == "cell" else mesh.points_attributes
            groups.append(group_object_ids(frame, object_attribute=object_attribute,
                                           association=association, cell_width=width))
        # Explicitly reject empty named schemas and other writer losses per source.
        serialize_le_mesh(mesh)

    cell_frames, point_frames, cells = [], [], []
    mapping = [] if object_attribute is not None else None
    provenance = []
    offset = 0
    next_id = 0
    for index, (mesh, path) in enumerate(zip(meshes, paths)):
        cell_frame = mesh.attributes.copy(deep=True)
        point_frame = mesh.points_attributes.copy(deep=True)
        if object_attribute is not None:
            frame = cell_frame if association == "cell" else point_frame
            original = frame[object_attribute].to_numpy()
            ids = groups[index]
            if id_policy == "source":
                remapped = np.searchsorted(ids, original).astype(np.int64) + next_id
                frame[object_attribute] = remapped
            for local_index, original_id in enumerate(ids):
                mapping.append({"source_index": index, "original_id": original_id.item(),
                                "merged_id": (next_id + local_index if id_policy == "source"
                                              else original_id.item())})
            next_id += len(ids)
        cell_frames.append(cell_frame)
        point_frames.append(point_frame)
        cells.append(mesh.cells.astype(np.int64) + offset)
        offset += mesh.vertex.shape[0]
        entry = {"source_index": index, "path": str(path) if path is not None else None,
                 "name": path.name if path is not None else None}
        if "le_tools" in mesh.data_attrs:
            entry["le_tools"] = copy.deepcopy(mesh.data_attrs["le_tools"])
        provenance.append(entry)
    metadata = copy.deepcopy(metadata)
    metadata["le_tools"] = {"operation": "merge", "sources": provenance,
                            "object_attribute": object_attribute, "association": association,
                            "id_policy": id_policy if object_attribute is not None else None,
                            "id_mapping": mapping}
    merged = LiquidEarthMesh(
        vertex=np.concatenate([mesh.vertex for mesh in meshes]),
        cells=np.concatenate(cells).astype(np.int32),
        attributes=pd.concat(cell_frames, ignore_index=True),
        points_attributes=pd.concat(point_frames, ignore_index=True), data_attrs=metadata,
    )
    serialize_le_mesh(merged)
    return merged


def merge_le(sources, destination, *, object_attribute=None, association=None,
             id_policy="source", overwrite=False) -> LEMergeResult:
    """Atomically merge ordered .le paths; reject aliases and incompatible data.

    No grouping is inferred. The default source identity policy only remaps an
    explicitly selected ID column; all other attributes keep their wire values.
    Returns an absolute destination Path and the provenance ID mapping, or None
    without grouping. Existing outputs require overwrite=True; inputs are never
    replaced, even when overwrite is enabled.
    """
    if isinstance(sources, (str, bytes, os.PathLike)):
        raise TypeError("sources must be an iterable of source paths")
    paths = tuple(Path(source).absolute() for source in sources)
    if not paths:
        raise ValueError("At least one source is required")
    merged = merge_meshes([load_le_mesh(path) for path in paths], sources=paths,
                          object_attribute=object_attribute, association=association,
                          id_policy=id_policy)
    output = write_le_mesh(merged, destination, sources=paths, overwrite=overwrite)
    return LEMergeResult(output, copy.deepcopy(merged.data_attrs["le_tools"]["id_mapping"]))
