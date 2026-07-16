import numpy as np
import pandas as pd

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
from subsurface.core.structs.base_structures import UnstructuredData
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase


def read_ply_point_cloud_to_unstruct(reader_args: GenericReaderFilesHelper) -> UnstructuredData:
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError(
            "The 'plyfile' package is required to read PLY point clouds. "
            "Install it with: pip install subsurface-terra[pointcloud]"
        )

    if reader_args.format != SupportedFormats.PLY:
        raise ValueError(f"read_ply_point_cloud_to_unstruct only supports PLY format, got {reader_args.format}")

    plydata = PlyData.read(reader_args.file_or_buffer)

    _check_ply_vertex_element(plydata)
    _reject_ply_with_faces(plydata)

    vertex_data = plydata["vertex"].data

    x = vertex_data["x"]
    y = vertex_data["y"]
    z = vertex_data["z"]

    vertex = np.column_stack((x, y, z)).astype(np.float64)

    skip_names = {"x", "y", "z"}
    attr_dict = {}
    for name in vertex_data.dtype.names:
        if name in skip_names:
            continue
        attr_dict[name] = vertex_data[name]

    vertex_attr = pd.DataFrame(attr_dict) if attr_dict else None

    return UnstructuredData.from_array(
        vertex=vertex,
        cells=SpecialCellCase.POINTS,
        vertex_attr=vertex_attr,
        xarray_attributes={"source_format": "ply"}
    )


def _check_ply_vertex_element(plydata: "PlyData") -> None:
    if "vertex" not in plydata:
        raise ValueError("PLY file must contain a 'vertex' element")

    vertex_names = {p.name for p in plydata["vertex"].properties}
    missing = [coord for coord in ("x", "y", "z") if coord not in vertex_names]

    if missing:
        raise ValueError(f"PLY vertex element is missing coordinate fields: {missing}")


def _reject_ply_with_faces(plydata: "PlyData") -> None:
    if "face" in plydata:
        raise ValueError(
            "PLY file contains 'face' element which is not supported by the point cloud reader. "
            "Use the mesh reader for PLY files with triangle faces."
        )
