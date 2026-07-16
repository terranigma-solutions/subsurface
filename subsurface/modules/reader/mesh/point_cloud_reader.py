import numpy as np
import pandas as pd
from typing import Union, List

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
from subsurface.core.structs.base_structures import UnstructuredData
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.optional_requirements import require_pye57


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


def read_e57_point_clouds_to_unstruct(reader_args: GenericReaderFilesHelper) -> List[UnstructuredData]:
    pye57 = require_pye57()

    if reader_args.format != SupportedFormats.E57:
        raise ValueError(f"read_e57_point_clouds_to_unstruct only supports E57 format, got {reader_args.format}")

    e57 = pye57.E57(reader_args.file_or_buffer)

    if e57.scan_count == 0:
        raise ValueError("E57 file contains no scans")

    results = []
    for scan_index in range(e57.scan_count):
        scan_data = e57.read_scan(scan_index, intensity=True, colors=True, row_column=True, ignore_missing_fields=True)

        x = scan_data.get("cartesianX")
        y = scan_data.get("cartesianY")
        z = scan_data.get("cartesianZ")

        if x is None or y is None or z is None:
            raise ValueError(f"E57 scan {scan_index} is missing cartesian coordinate fields")

        vertex = np.column_stack((x, y, z)).astype(np.float64)

        attr_dict = {}
        if "intensity" in scan_data and scan_data["intensity"] is not None:
            attr_dict["intensity"] = scan_data["intensity"]
        if "colorRed" in scan_data and scan_data["colorRed"] is not None:
            attr_dict["red"] = scan_data["colorRed"]
            attr_dict["green"] = scan_data["colorGreen"]
            attr_dict["blue"] = scan_data["colorBlue"]
        if "rowIndex" in scan_data and scan_data["rowIndex"] is not None:
            attr_dict["rowIndex"] = scan_data["rowIndex"]
            attr_dict["columnIndex"] = scan_data["columnIndex"]

        vertex_attr = pd.DataFrame(attr_dict) if attr_dict else None

        header = e57.get_header(scan_index)
        scan_attrs = {
            "source_format": "e57",
            "scan_index": scan_index,
            "scan_point_count": header.point_count,
        }
        rotation = _safe_numpy_to_list(getattr(header, "rotation_matrix", None))
        if rotation is not None:
            scan_attrs["scan_rotation_matrix"] = rotation
        translation = _safe_numpy_to_list(getattr(header, "translation", None))
        if translation is not None:
            scan_attrs["scan_translation"] = translation

        ud = UnstructuredData.from_array(
            vertex=vertex,
            cells=SpecialCellCase.POINTS,
            vertex_attr=vertex_attr,
            xarray_attributes=scan_attrs,
        )
        results.append(ud)

    return results


def _safe_numpy_to_list(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


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
