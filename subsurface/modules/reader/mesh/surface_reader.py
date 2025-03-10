from .csv_mesh_reader import mesh_csv_to_vertex, mesh_csv_to_cells, mesh_csv_to_attributes
from .dxf_reader import dxf_from_file_to_vertex, dxf_from_stream_to_vertex, DXFEntityType
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
import numpy as np

__all__ = ['read_mesh_file_to_vertex', 'read_mesh_file_to_cells',
           'read_mesh_file_to_attr', 'cells_from_delaunay']


def read_mesh_file_to_vertex(reader_args: GenericReaderFilesHelper) -> np.ndarray:
    if reader_args.format is SupportedFormats.CSV:
        vertex = mesh_csv_to_vertex(reader_args.file_or_buffer, reader_args.columns_map,
                                    **reader_args.pandas_reader_kwargs)
    elif reader_args.format is SupportedFormats.DXF:
        vertex = dxf_from_file_to_vertex(
            file_path=reader_args.file_or_buffer,
            entity_type=reader_args.additional_reader_kwargs.get('entity_type', DXFEntityType.ALL)
        )
    elif reader_args.format is SupportedFormats.DXFStream:
        vertex = dxf_from_stream_to_vertex(reader_args.file_or_buffer)
    else:
        raise ValueError(f"Subsurface is not able to read the following extension: {reader_args.format}")
    return vertex


def read_mesh_file_to_cells(reader_args: GenericReaderFilesHelper) -> np.ndarray:
    extension = reader_args.format

    if extension == SupportedFormats.CSV:
        cells = mesh_csv_to_cells(
            path_to_file=reader_args.file_or_buffer,
            columns_map=reader_args.columns_map,
            **reader_args.pandas_reader_kwargs
        )
    else:
        raise ValueError(f"Subsurface is not able to read the following extension: {extension}")
    return cells


def read_mesh_file_to_attr(reader_args: GenericReaderFilesHelper):
    extension = reader_args.format
    if extension == SupportedFormats.CSV:
        attr = mesh_csv_to_attributes(reader_args.file_or_buffer,
                                      reader_args.columns_map,
                                      **reader_args.pandas_reader_kwargs)
    else:
        raise ValueError(f"Subsurface is not able to read the following extension: {extension}")
    return attr


def cells_from_delaunay(vertex):
    import pyvista as pv
    a = pv.PolyData(vertex)
    b = a.delaunay_2d().faces
    cells = b.reshape(-1, 4)[:, 1:]
    return cells
