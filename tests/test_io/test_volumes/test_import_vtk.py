from __future__ import annotations

import pathlib

import pytest
from pyvista import examples
import pyvista as pv

import subsurface
from subsurface import StructuredGrid
from subsurface.modules.visualization import to_pyvista_grid, pv_plot
from tests.conftest import RequirementsLevel


def generate_vtk():
    """ Use this function only to generate vtk files for testing purposes """

    grid = examples.load_hexbeam()
    grid.cell_data['Cell Number'] = range(grid.n_cells)
    grid.plot(scalars='Cell Number')
    
    # Write vtk
    grid.save("test.vtk")
   
    
# TODO: [ ] Make vtk reader using probably pyvista
# TODO: [ ] Convert to structured data
# TODO: [ ] Export to le file
# new_file = open("test_volume.le", "wb")
# new_file.write(sd.to_binary())

pf = pathlib.Path(__file__).parent.absolute()
data_path = pf.joinpath('../../data/volume/')


pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_VOLUME) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_VOLUME"
)


def test_vtk_file_to_structured_data():
    # read vtk file with pyvista
    pyvista_obj: pv.UnstructuredGrid = pv.read(data_path.joinpath('test.vtk'))
    if PLOT:=False:
        pyvista_obj.plot()
    
    struct: subsurface.StructuredData = subsurface.StructuredData.from_pyvista_unstructured_grid(
        grid=pyvista_obj
    )

    sg = StructuredGrid(struct)

    mesh = to_pyvista_grid(sg)
    pv_plot([mesh], image_2d=False )
    pass