from __future__ import annotations

import pathlib
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

import subsurface
from subsurface import StructuredGrid, optional_requirements
from subsurface.modules.visualization import to_pyvista_grid, pv_plot
from tests.conftest import RequirementsLevel

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_VOLUME) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_VOLUME"
)


def generate_vtk_struct():
    """ Use this function only to generate vtk files for testing purposes """
    pyvista = optional_requirements.require_pyvista()
    grid: pyvista.ExplicitStructuredGrid = pyvista.examples.load_explicit_structured()
    grid.cell_data['Cell Number'] = range(grid.n_cells)
    grid.cell_data['Random'] = np.random.rand(grid.n_cells)
    grid.plot(scalars='Random')

    # Write vtk
    grid.save("test_structured.vtk")


# TODO: [x] Make vtk reader using probably pyvista
# TODO: [x] Convert to structured data
# TODO: [ ] Export to le file

pf = pathlib.Path(__file__).parent.absolute()
data_path = pf.joinpath('../../data/volume/')


@pytest.mark.liquid_earth
def test_vtk_file_to_structured_data() -> subsurface.StructuredData:
    # read vtk file with pyvista
    NamedTemporaryFile()
    pv = optional_requirements.require_pyvista()
    joinpath = data_path.joinpath('test_structured.vtk')
    pyvista_obj: pv.DataSet = pv.read(joinpath)
    pv.examples.download_angular_sector()
    try:
        pyvista_struct: pv.ExplicitStructuredGrid = pyvista_obj.cast_to_explicit_structured_grid()
    except Exception as e:
        raise "The file is not a structured grid"

    active_scalars = "Cell Number"

    if PLOT := False:
        pyvista_struct.set_active_scalars(active_scalars)
        pyvista_struct.plot()

    struct: subsurface.StructuredData = subsurface.StructuredData.from_pyvista_structured_grid(
        grid=pyvista_struct,
        data_array_name=active_scalars
    )

    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    mesh = to_pyvista_grid(sg)
    pv_plot([mesh], image_2d=True)
    return struct


def test_vtk_file_to_binary():
    struct: subsurface.StructuredData = test_vtk_file_to_structured_data()
    print(struct.bounds)
    
    struct.active_data_array_name = "Cell Number"
    binary_cell_number = struct.to_binary()

    
    struct.active_data_array_name = "Random"
    binary_random = struct.to_binary()

    if WRITE_TO_DISK:=False:
        new_file = open("test_volume_Cell Number.le", "wb")
        new_file.write(binary_cell_number)
        new_file = open("test_volume_Random.le", "wb")
        new_file.write(binary_random)

    
    
    
    
    