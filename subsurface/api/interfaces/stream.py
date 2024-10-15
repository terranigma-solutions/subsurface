from io import BytesIO
from typing import TextIO

import pandas
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper

from subsurface.core.geological_formats import BoreholeSet

from ...core.structs.base_structures import UnstructuredData
from ...modules import reader
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


def CSV_wells_stream_to_unstruc(
        collars_reader: GenericReaderFilesHelper,
        surveys_reader: GenericReaderFilesHelper,
        attrs_reader: GenericReaderFilesHelper
) -> list[UnstructuredData]:
    
    borehole_set: BoreholeSet = read_wells(
        collars_reader= collars_reader,
        surveys_reader= surveys_reader,
        attrs_reader= attrs_reader
    )
    
    list_unstruct: list[UnstructuredData] = [
            borehole_set.collars.data,
            borehole_set.combined_trajectory.data
    ]
    return list_unstruct
