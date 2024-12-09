import io

import numpy as np

import subsurface
from ....optional_requirements import require_omf, require_pyvista
from ....core.structs.base_structures import UnstructuredData


def omf_stream_to_unstructs(stream: io.BytesIO) -> UnstructuredData:
    pyvista = require_pyvista()
    omfvista = require_omf()
    omf = omfvista.load_project(stream)
    list_of_polydata: list[pyvista.PolyData] = []


    all_vertex = []
    all_cells = []
    cell_attr = []
    _last_cell: int = 0

    for i in range(omf.n_blocks):
        block: pyvista.PolyData = omf[i]
        cell_type = block.get_cell(0).type

        if cell_type == pyvista.CellType.TRIANGLE:
            pyvista_unstructured_grid: pyvista.UnstructuredGrid = block.cast_to_unstructured_grid()
            all_vertex.append(pyvista_unstructured_grid.points)
            cells: np.ndarray = pyvista_unstructured_grid.cells.reshape(-1, 4)[:, 1:]
            if len(all_cells) > 0:
                cells = cells + _last_cell

            all_cells.append(cells)
            cell_attr.append(np.ones(len(all_cells[-1])) * i)
            _last_cell = cells.max() + 1

    # * Create the unstructured data
    import pandas as pd

    unstructured_data = subsurface.UnstructuredData.from_array(
        vertex=np.vstack(all_vertex),
        cells=np.vstack(all_cells),
        cells_attr=pd.DataFrame(np.hstack(cell_attr), columns=["Block id"]),
    )
    return unstructured_data
