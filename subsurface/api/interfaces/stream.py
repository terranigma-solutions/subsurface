from io import BytesIO
from typing import TextIO

import pandas

from ...core.structs import TriSurf
from ...core.reader_helpers.reader_unstruct import ReaderUnstructuredHelper
from ...core.reader_helpers.readers_data import GenericReaderFilesHelper
from ...core.geological_formats import BoreholeSet
from ...core.structs.base_structures import UnstructuredData, StructuredData

from ...modules import reader
from ...modules.reader.volume.read_volume import read_volumetric_mesh_to_subsurface, read_VTK_structured_grid
from ...modules.reader.mesh.surfaces_api import read_2d_mesh_to_unstruct
from ...modules.reader.volume.volume_utils import interpolate_unstructured_data_to_structured_data

from ..reader.read_wells import read_wells


def DXF_stream_to_unstruc(stream: TextIO) -> UnstructuredData:
    vertex, cells, cell_attr_int, cell_attr_map = reader.dxf_stream_to_unstruct_input(stream)

    unstruct = UnstructuredData.from_array(
        vertex,
        cells,
        cells_attr=pandas.DataFrame(cell_attr_int, columns=["Id"]),
        xarray_attributes={"cell_attr_map": cell_attr_map},
    )

    return unstruct


def OMF_stream_to_unstruc(stream: BytesIO) -> list[UnstructuredData]:
    list_unstruct: list[UnstructuredData] = reader.omf_stream_to_unstructs(stream)
    return list_unstruct


def MX_stream_to_unstruc(stream: TextIO) -> list[UnstructuredData]:
    list_unstruct: list[UnstructuredData] = [reader.mx_to_unstruc_from_binary(stream)]
    return list_unstruct


def OBJ_stream_to_trisurf(stream: BytesIO) -> TriSurf:
    tri_mesh: TriSurf = reader.load_obj_with_trimesh_from_binary(stream)
    breakpoint()
    raise NotImplementedError


def VTK_stream_to_struct(stream: BytesIO, attribute_name: str) -> list[StructuredData]:
    struct = read_VTK_structured_grid(stream, attribute_name)
    return [struct]


def CSV_wells_stream_to_unstruc(
        collars_reader: GenericReaderFilesHelper,
        surveys_reader: GenericReaderFilesHelper,
        attrs_reader: GenericReaderFilesHelper,
        is_lith_attr: bool,
        number_nodes: int,
        add_attrs_as_nodes: bool,
        duplicate_attr_depths: bool
) -> list[UnstructuredData]:
    borehole_set: BoreholeSet = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=is_lith_attr,
        number_nodes=number_nodes,
        add_attrs_as_nodes=add_attrs_as_nodes,
        duplicate_attr_depths=duplicate_attr_depths
    )

    list_unstruct: list[UnstructuredData] = [
            borehole_set.collars.data,
            borehole_set.combined_trajectory.data
    ]
    return list_unstruct


def CSV_mesh_stream_to_unstruc(
        vertex_reader: GenericReaderFilesHelper,
        edges_reader: GenericReaderFilesHelper,
        cells_attrs_reader: GenericReaderFilesHelper,
        vertex_attrs_reader: GenericReaderFilesHelper
) -> list[UnstructuredData]:
    reader_unstruc = ReaderUnstructuredHelper(vertex_reader, edges_reader, vertex_attrs_reader, cells_attrs_reader)
    ud = read_2d_mesh_to_unstruct(reader_unstruc)
    return [ud]


def CSV_volume_stream_to_unstruc(
        coord_reader: GenericReaderFilesHelper,
        attrs_reader: GenericReaderFilesHelper
) -> list[UnstructuredData]:
    ud = read_volumetric_mesh_to_subsurface(
        reader_helper_coord=coord_reader,
        reader_helper_attr=attrs_reader
    )
    return [ud]


def CSV_volume_stream_to_struct(
        coord_reader: GenericReaderFilesHelper,
        attrs_reader: GenericReaderFilesHelper,
        attr_name: str,
        resolution: list[int]
) -> list[StructuredData]:
    ud = read_volumetric_mesh_to_subsurface(
        reader_helper_coord=coord_reader,
        reader_helper_attr=attrs_reader
    )

    sd: StructuredData = interpolate_unstructured_data_to_structured_data(
        ud=ud,
        attr_name=attr_name,
        resolution=resolution
    )

    return [sd]
