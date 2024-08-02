from typing import TextIO

import pandas

from ...core.structs.base_structures import UnstructuredData
from ...modules.reader import dxf_stream_to_unstruct_input


def DXF_stream_to_unstruc(stream: TextIO) -> UnstructuredData:
    
    vertex, cells, cell_attr_int, cell_attr_map = dxf_stream_to_unstruct_input(stream)

    unstruct = UnstructuredData.from_array(
        vertex,
        cells,
        cells_attr=pandas.DataFrame(cell_attr_int, columns=["Id"]),
        xarray_attributes={"cell_attr_map": cell_attr_map},
    )
    
    return unstruct
