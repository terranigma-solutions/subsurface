import pandas as pd
from typing import Union

import numpy as np
import warnings

from .surface_reader import read_mesh_file_to_vertex, read_mesh_file_to_cells, cells_from_delaunay, read_mesh_file_to_attr
from .point_cloud_reader import read_ply_point_cloud_to_unstruct
from ....core.reader_helpers.reader_unstruct import ReaderUnstructuredHelper
from ....core.structs import UnstructuredData

from ....core.structs.base_structures.base_structures_enum import SpecialCellCase
from ....core.reader_helpers.readers_data import SupportedFormats, GenericReaderFilesHelper


def read_2d_mesh_to_unstruct(
        reader_args: ReaderUnstructuredHelper,
        delaunay: bool = True
) -> UnstructuredData:
    
    vertex: np.ndarray = read_mesh_file_to_vertex(reader_args.reader_vertex_args)
    cells: Union[np.ndarray, SpecialCellCase]
    cells_attr: Union[pd.DataFrame, None] = None
    vertex_attr: Union[pd.DataFrame, None] = None
    if reader_args.reader_cells_args is not None:
        cells = read_mesh_file_to_cells(reader_args.reader_cells_args)
    elif delaunay:
        cells = cells_from_delaunay(vertex)
    else:
        warnings.warn("No arguments to compute cell")
        cells = SpecialCellCase.POINTS
    if reader_args.reader_cells_attr_args is not None:
        cells_attr: pd.DataFrame = read_mesh_file_to_attr(reader_args.reader_cells_attr_args)
    if reader_args.reader_vertex_attr_args is not None:
        vertex_attr = read_mesh_file_to_attr(reader_args.reader_vertex_attr_args)

    ud = UnstructuredData.from_array(
        vertex=vertex,
        cells=cells,
        cells_attr=cells_attr,
        vertex_attr=vertex_attr,
    )
    return ud


def read_point_cloud_to_unstruct(reader_args: GenericReaderFilesHelper) -> UnstructuredData:
    if reader_args.format == SupportedFormats.PLY:
        return read_ply_point_cloud_to_unstruct(reader_args)
    raise ValueError(f"Unsupported point cloud format: {reader_args.format}")
