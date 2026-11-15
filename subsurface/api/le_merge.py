"""Strict, ordered concatenation of compatible unstructured LiquidEarth files."""

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
                 id_policy="source", sources=None, source_labels=None) -> LiquidEarthMesh:
    """Concatenate meshes without welding, coercing schemas, or mutating inputs.

    Grouping requires both arguments and follows inspection's association rules.
    ``source`` assigns consecutive int64 IDs in source order, then sorted original
    ID order; ``shared`` explicitly preserves globally shared IDs. Original exact
    Python numeric values and nested source provenance are kept under ``le_tools``.
    Optional sources are path labels, one per mesh. Optional source_labels are
    recorded verbatim instead (e.g. blob paths), never resolved against the local
    filesystem; supply at most one of the two. See docs/le_merge.md.
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
    if sources is not None and source_labels is not None:
        raise ValueError("Supply sources or source_labels, not both")
    if source_labels is not None:
        labels = _check_source_labels(source_labels, len(meshes))
    elif sources is None:
        labels = [None] * len(meshes)
    else:
        if isinstance(sources, (str, bytes, os.PathLike)):
            raise TypeError("sources must be an iterable of paths")
        labels = [str(Path(source).absolute()) for source in sources]
        if len(labels) != len(meshes):
            raise ValueError("sources must contain one path per mesh")

    # Count before concatenation or serialization, so index overflow never wraps.
    total_vertices = sum(mesh.vertex.shape[0] for mesh in meshes)
    if total_vertices > np.iinfo(np.int32).max + 1:
        raise ValueError("Merged vertex count exceeds int32 connectivity capacity")
    first = meshes[0]
    width = first.cells.shape[1]
    metadata = {key: value for key, value in first.data_attrs.items() if key != "le_tools"}
    # Label maps are combined below rather than compared: the grouped association's
    # map when grouping, both (for a clear error) without grouping.
    relaxed = (LABEL_MAP_KEYS[association],) if object_attribute is not None else tuple(LABEL_MAP_KEYS.values())
    semantic_reference = _semantic_metadata(first.data_attrs, relaxed)
    names = [_source_name(label, index) for index, label in enumerate(labels)]
    groups = []
    for index, mesh in enumerate(meshes):
        if mesh.cells.shape[1] != width:
            raise ValueError("Sources must have identical geometry/connectivity widths")
        if width == 0 and mesh.cells.shape[0] not in (0, mesh.vertex.shape[0]):
            raise ValueError("Width-0 point connectivity must have zero rows or one row per point")
        if "le_tools" in mesh.data_attrs and not isinstance(mesh.data_attrs["le_tools"], dict):
            raise ValueError("Reserved le_tools provenance must be a dictionary")
        differing = _differing_keys(semantic_reference, _semantic_metadata(mesh.data_attrs, relaxed))
        if differing:
            names = ", ".join(f'"{key}"' for key in differing)
            raise ValueError(f"Sources have different {names} metadata")
        for label, frame, reference in (
                ("cell", mesh.attributes, first.attributes),
                ("point", mesh.points_attributes, first.points_attributes)):
            if (not frame.columns.equals(reference.columns) or
                    not frame.dtypes.equals(reference.dtypes)):
                same_names = frame.columns.equals(reference.columns)
                raise ValueError(
                    f'Sources have different {label} attribute {"dtypes" if same_names else "names"}: '
                    f'"{names[0]}" has {_describe_columns(reference)}, "{names[index]}" has '
                    f'{_describe_columns(frame)}' + ("" if same_names else
                    ". Meshes from different file types (e.g. DXF and OBJ) cannot be merged yet"))
        if object_attribute is not None:
            frame = mesh.attributes if association == "cell" else mesh.points_attributes
            groups.append(group_object_ids(frame, object_attribute=object_attribute,
                                           association=association, cell_width=width))
        # Explicitly reject empty named schemas and other writer losses per source.
        serialize_le_mesh(mesh)

    if width == 0 and sum(mesh.cells.shape[0] for mesh in meshes) not in (0, total_vertices):
        raise ValueError("Width-0 point connectivity row policies would create a partial output")
    if object_attribute is None:
        for key in relaxed:
            values = [json.dumps(mesh.data_attrs.get(key), sort_keys=True) for mesh in meshes]
            if len(set(values)) > 1:
                raise ValueError(
                    f'Sources have different layer names ("{key}"); keep object IDs '
                    "(group by the ID attribute) to merge them")

    cell_frames, point_frames, cells = [], [], []
    mapping = [] if object_attribute is not None else None
    provenance = []
    offset = 0
    next_id = 0
    for index, (mesh, label) in enumerate(zip(meshes, labels)):
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
        entry = {"source_index": index, "path": label,
                 "name": PurePosixPath(label).name if label is not None else None}
        if "le_tools" in mesh.data_attrs:
            entry["le_tools"] = copy.deepcopy(mesh.data_attrs["le_tools"])
        provenance.append(entry)
    metadata = copy.deepcopy(metadata)
    renamed = None
    if object_attribute is not None:
        key = LABEL_MAP_KEYS[association]
        maps = [mesh.data_attrs.get(key) for mesh in meshes]
        if any(_is_label_map(value) for value in maps):
            merged_map, renamed = _merge_label_maps(
                key, [value if _is_label_map(value) else {} for value in maps],
                mapping, provenance, id_policy)
            metadata[key] = merged_map
    metadata["le_tools"] = {"operation": "merge", "sources": provenance,
                            "object_attribute": object_attribute, "association": association,
                            "id_policy": id_policy if object_attribute is not None else None,
                            "id_mapping": mapping}
    if renamed is not None:
        metadata["le_tools"]["renamed_labels"] = renamed
    vertex = np.concatenate([mesh.vertex for mesh in meshes])
    if any(BOUNDS_KEY in mesh.data_attrs for mesh in meshes):
        # Derived from geometry, like transform_le: recalculated, never compared.
        wire_vertex = vertex.astype(np.float32)
        metadata[BOUNDS_KEY] = (
            np.column_stack((wire_vertex.min(axis=0), wire_vertex.max(axis=0))).ravel().tolist()
            if len(wire_vertex) else None)
    merged = LiquidEarthMesh(
        vertex=vertex,
        cells=np.concatenate(cells).astype(np.int32),
        attributes=pd.concat(cell_frames, ignore_index=True),
        points_attributes=pd.concat(point_frames, ignore_index=True), data_attrs=metadata,
    )
    serialize_le_mesh(merged)
    return merged


LABEL_MAP_KEYS = {"cell": "cell_attr_map", "point": "point_attr_map"}
BOUNDS_KEY = "bounds"


def _is_label_map(value) -> bool:
    """Name-to-ID maps such as the DXF reader's layer names: an object of str -> int."""
    return isinstance(value, dict) and all(
        type(label) is str and type(object_id) is int for label, object_id in value.items())


def _semantic_metadata(data_attrs, relaxed_keys) -> Dict[str, Any]:
    return {key: value for key, value in data_attrs.items() if key not in ("le_tools", BOUNDS_KEY) and not (
            key in relaxed_keys and _is_label_map(value))}


def _source_name(label, index) -> str:
    return PurePosixPath(label).name if label is not None else f"source {index}"


def _describe_columns(frame) -> str:
    return ", ".join(f"{name} ({dtype})" for name, dtype in frame.dtypes.items()) or "none"


def _differing_keys(reference, semantic) -> List[str]:
    return sorted(key for key in set(reference) | set(semantic)
                  if key not in reference or key not in semantic or
                  json.dumps(reference[key], sort_keys=True) != json.dumps(semantic[key], sort_keys=True))


def _merge_label_maps(key, maps, mapping, provenance, id_policy):
    """Rebuild a label map for merged IDs; return (map, renamed labels).

    Entries whose ID does not occur in their source are dropped (split parts
    carry their whole original map). ``source`` renames repeated labels with
    their source name; ``shared`` rejects labels that disagree across sources.
    """
    present = [{} for _ in maps]
    for entry in mapping:
        present[entry["source_index"]][entry["original_id"]] = entry["merged_id"]
    merged, owners, renamed = {}, {}, []
    for index, labels in enumerate(maps):
        for label, original_id in sorted(labels.items(), key=lambda item: (item[1], item[0])):
            if original_id not in present[index]:
                continue
            merged_id = present[index][original_id]
            if id_policy == "shared":
                if label in merged and merged[label] != merged_id:
                    raise ValueError(f'Label "{label}" has different IDs ({merged[label]} and '
                                     f'{merged_id}) in different sources ("{key}")')
                owner = owners.get(merged_id)
                if owner is not None and owner[0] != label and owner[1] != index:
                    raise ValueError(f'ID {merged_id} has different labels ("{owner[0]}" and '
                                     f'"{label}") in different sources ("{key}")')
                owners.setdefault(merged_id, (label, index))
                merged[label] = merged_id
                continue
            unique = label
            if unique in merged:
                name = provenance[index]["name"] or f"source {index}"
                if name.lower().endswith(".le"):
                    name = name[:-3]
                unique, count = f"{label} ({name})", 2
                while unique in merged:
                    unique, count = f"{label} ({name} {count})", count + 1
                renamed.append({"source_index": index, "label": label,
                                "merged_label": unique, "merged_id": merged_id})
            merged[unique] = merged_id
    return merged, renamed


def _check_source_labels(source_labels, count) -> List[str]:
    if isinstance(source_labels, (str, bytes, os.PathLike)):
        raise TypeError("source_labels must be an iterable of strings")
    labels = list(source_labels)
    if any(type(label) is not str or not label for label in labels):
        raise TypeError("source_labels must contain non-empty strings")
    if len(labels) != count:
        raise ValueError("source_labels must contain one label per source")
    return labels


def merge_le(sources, destination, *, object_attribute=None, association=None,
             id_policy="source", overwrite=False, source_labels=None) -> LEMergeResult:
    """Atomically merge ordered .le paths; reject aliases and incompatible data.

    No grouping is inferred. The default source identity policy only remaps an
    explicitly selected ID column; all other attributes keep their wire values.
    Returns an absolute destination Path and the provenance ID mapping, or None
    without grouping. Existing outputs require overwrite=True; inputs are never
    replaced, even when overwrite is enabled. Optional ``source_labels`` (one
    string per source) replace the absolute paths recorded in provenance, so
    outputs do not depend on where the inputs happen to be stored locally.
    """
    if isinstance(sources, (str, bytes, os.PathLike)):
        raise TypeError("sources must be an iterable of source paths")
    paths = tuple(Path(source).absolute() for source in sources)
    if not paths:
        raise ValueError("At least one source is required")
    if source_labels is not None:
        provenance = dict(source_labels=_check_source_labels(source_labels, len(paths)))
    else:
        provenance = dict(sources=paths)
    merged = merge_meshes([load_le_mesh(path) for path in paths], **provenance,
                          object_attribute=object_attribute, association=association,
                          id_policy=id_policy)
    output = write_le_mesh(merged, destination, sources=paths, overwrite=overwrite)
    return LEMergeResult(output, copy.deepcopy(merged.data_attrs["le_tools"]["id_mapping"]))
