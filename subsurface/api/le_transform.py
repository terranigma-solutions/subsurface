"""Bake explicitly specified affine transforms into unstructured LE geometry."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np

from subsurface.api._le_file_ops import load_le_mesh, serialize_le_mesh, write_le_mesh
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


def transform_mesh(mesh: LiquidEarthMesh, matrix, *, normal_columns=None,
                   vector_columns=None) -> LiquidEarthMesh:
    """Return an independent mesh with an affine column-vector transform baked in.

    Column declarations map ``point`` or ``cell`` to a sequence of XYZ column
    triples, e.g. ``{"point": [("nx", "ny", "nz")]}``. No column may be
    selected twice. Vectors use the linear part; normals use its inverse
    transpose and are normalized. Unselected columns retain their values/dtypes.
    Bounds are XYZ min/max pairs of the serialized float32 geometry (None for
    an empty mesh). Non-null top-level transform metadata is unsupported.
    Input and output are preflighted by the shared loss-checking serializer.
    """
    raw = np.asarray(matrix)
    if raw.shape != (4, 4) or raw.dtype.kind not in 'iuf':
        raise ValueError("matrix must be a real numeric 4x4 affine matrix")
    matrix = raw.astype(np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be finite")
    if not np.array_equal(matrix[3], [0, 0, 0, 1]):
        raise ValueError("matrix must have affine last row [0, 0, 0, 1]; projective transforms are unsupported")
    linear = matrix[:3, :3]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        sign, log_determinant = np.linalg.slogdet(linear)
    if sign == 0 or not np.isfinite(sign) or not np.isfinite(log_determinant):
        raise ValueError("Singular or numerically unresolved transforms are unsupported for geometry")
    serialize_le_mesh(mesh)
    if mesh.data_attrs.get('transform') is not None:
        raise ValueError("Non-null transform metadata has no established baking semantics")
    if sign < 0 and mesh.cells.shape[1] in (4, 8):
        raise ValueError("Reflected volumetric cells are unsupported")

    result = LiquidEarthMesh(
        vertex=mesh.vertex.astype(np.float64, copy=True), cells=mesh.cells.copy(),
        attributes=mesh.attributes.copy(deep=True),
        points_attributes=mesh.points_attributes.copy(deep=True),
        data_attrs=deepcopy(mesh.data_attrs),
    )
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        result.vertex = result.vertex @ linear.T + matrix[:3, 3]
    limit = np.finfo(np.float32).max
    if not np.all(np.isfinite(result.vertex)) or np.any(np.abs(result.vertex) > limit):
        raise ValueError("Transformed geometry must be finite and within float32 range")
    if sign < 0 and result.cells.shape[1] == 3:
        result.cells[:, [1, 2]] = result.cells[:, [2, 1]]

    frames = {'point': result.points_attributes, 'cell': result.attributes}
    selected = {'point': set(), 'cell': set()}
    for is_normal, declarations in ((False, vector_columns), (True, normal_columns)):
        if declarations is None:
            continue
        if not isinstance(declarations, Mapping) or any(key not in frames for key in declarations):
            raise ValueError("Column declarations must map point/cell to sequences of XYZ triples")
        try:
            operator = np.linalg.inv(linear) if is_normal else linear.T
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular transforms are unsupported for normals") from exc
        for association, triples in declarations.items():
            if not isinstance(triples, (list, tuple)):
                raise ValueError("Column declarations must contain sequences of XYZ triples")
            frame = frames[association]
            for triple in triples:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    raise ValueError("Each vector/normal requires exactly three XYZ column names")
                for name in triple:
                    if name not in frame.columns:
                        raise ValueError(f"Missing {association} vector/normal column: {name!r}")
                    if name in selected[association]:
                        raise ValueError(f"Column selected more than once: {name!r}")
                    selected[association].add(name)
                    if frame[name].dtype.kind not in 'iuf':
                        raise ValueError("Vector/normal columns must be real numeric, not boolean")
                values = frame[list(triple)].to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError("Vector/normal inputs must be finite")
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    values = values @ operator
                if not np.all(np.isfinite(values)):
                    raise ValueError("Transformed vectors/normals must be finite")
                if is_normal and len(values):
                    # Scale before taking lengths to avoid square overflow/underflow.
                    scale = np.max(np.abs(values), axis=1)
                    if np.any(scale == 0):
                        raise ValueError("Zero normals are unsupported")
                    values /= scale[:, None]
                    values /= np.linalg.norm(values, axis=1)[:, None]
                if np.any(np.abs(values) > limit):
                    raise ValueError("Transformed vectors/normals exceed float32 range")
                for index, name in enumerate(triple):
                    frame[name] = values[:, index]

    wire_vertex = result.vertex.astype(np.float32)
    result.data_attrs['bounds'] = (
        np.column_stack((wire_vertex.min(axis=0), wire_vertex.max(axis=0))).ravel().tolist()
        if len(wire_vertex) else None
    )
    serialize_le_mesh(result)
    return result


def transform_le(source, destination, matrix, *, normal_columns=None,
                 vector_columns=None, overwrite=False) -> Path:
    """Bake an affine transform into a new LE file, returning its absolute Path.

    Inputs are never overwritten, even with ``overwrite=True``. Destination
    publication is atomic; existing destinations require explicit overwrite.
    See :func:`transform_mesh` for matrix, columns, and metadata conventions.
    """
    mesh = transform_mesh(load_le_mesh(source), matrix, normal_columns=normal_columns,
                          vector_columns=vector_columns)
    return write_le_mesh(mesh, destination, sources=[source], overwrite=overwrite)
