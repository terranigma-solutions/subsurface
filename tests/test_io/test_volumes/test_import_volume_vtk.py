from __future__ import annotations

import os
import pathlib
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

import subsurface
from subsurface import StructuredGrid, optional_requirements
from subsurface.modules.reader.volume.read_volume import pv_cast_to_structured_grid
from subsurface.modules.visualization import to_pyvista_grid, pv_plot
from tests.conftest import RequirementsLevel

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_VOLUME) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_VOLUME"
)


def generate_vtk_rectilinear():
    """ Use this function only to generate vtk files for testing purposes """
    pyvista = optional_requirements.require_pyvista()
    grid: pyvista.RectilinearGrid = pyvista.examples.load_rectilinear()
    grid.cell_data['Cell Number'] = range(grid.n_cells)
    grid.cell_data['Random'] = np.random.rand(grid.n_cells)
    grid.plot(scalars='Random')

    # Write vtk
    grid.save("test_rectilinear.vtk")


def generate_vtk_uniform():
    """ Use this function only to generate vtk files for testing purposes """
    pyvista = optional_requirements.require_pyvista()
    grid: pyvista.ImageData = pyvista.examples.load_uniform()
    grid.cell_data['Cell Number'] = range(grid.n_cells)
    grid.cell_data['Random'] = np.random.rand(grid.n_cells)
    grid.plot(scalars='Random')

    # Write vtk
    grid.save("test_uniform.vtk")


# TODO: [x] Make vtk reader using probably pyvista
# TODO: [x] Convert to structured data
# TODO: [ ] Export to le file

pf = pathlib.Path(__file__).parent.absolute()
data_path = pf.joinpath('../../data/volume/')


@pytest.mark.liquid_earth
def test_vtk_uniform_to_structured_data() -> subsurface.StructuredData:
    # read vtk file with pyvista
    NamedTemporaryFile()
    pv = optional_requirements.require_pyvista()
    joinpath = data_path.joinpath('test_uniform.vtk')
    pyvista_obj: pv.DataSet = pv.read(joinpath)
    pv.examples.download_angular_sector()

    pyvista_struct: pv.ImageData = pv_cast_to_structured_grid(pyvista_obj)

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

    assert struct.shape == (9, 9, 9)
    assert struct.bounds == {'x': (0.0, 8.0), 'y': (0.0, 8.0), 'z': (0.0, 8.0)}
    assert struct.values.min() == 0
    assert struct.values.max() == 728
    assert struct.values[2][4][6] == 524

    return struct


def test_vtk_file_to_binary():
    struct: subsurface.StructuredData = test_vtk_uniform_to_structured_data()
    print(struct.bounds)

    struct.active_data_array_name = "Cell Number"
    binary_cell_number = struct.to_binary()

    struct.active_data_array_name = "Random"
    binary_random = struct.to_binary()

    if WRITE_TO_DISK := False:
        new_file = open("test_volume_Cell Number.le", "wb")
        new_file.write(binary_cell_number)
        new_file = open("test_volume_Random.le", "wb")
        new_file.write(binary_random)


def test_vtk_file_to_structured_data__gen11818__idn63() -> subsurface.StructuredData:

    pv = optional_requirements.require_pyvista()

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    filepath = devops_path.joinpath(r"volume/VTK/IDN-63/muon.vtk")

    pyvista_obj: pv.DataSet = pv.read(filepath)

    pyvista_struct = pv_cast_to_structured_grid(pyvista_obj)

    struct: subsurface.StructuredData = subsurface.StructuredData.from_pyvista_structured_grid(
        grid=pyvista_struct,
        data_array_name="model_name"
    )

    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    mesh = to_pyvista_grid(sg)
    pv_plot([mesh], image_2d=True)

    assert struct.shape == (100, 139, 46)
    assert round((struct.bounds['x'][1] + struct.values.max()) / struct.values[50][70][20], 4) == 1.6388

    return struct


def test_vtk_file_to_structured_data__idn69__gen12023() -> subsurface.StructuredData:

    pv = optional_requirements.require_pyvista()

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    filepath = devops_path.joinpath(r"volume/VTK/IDN-69/idn69.vtk")

    pyvista_obj: pv.DataSet = pv.read(filepath)

    pyvista_struct = pv_cast_to_structured_grid(pyvista_obj)

    struct: subsurface.StructuredData = subsurface.StructuredData.from_pyvista_structured_grid(
        grid=pyvista_struct,
        data_array_name="density"
    )

    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    mesh = to_pyvista_grid(sg)
    pv_plot([mesh], image_2d=True)

    assert struct.shape == (175, 112, 163)
    assert round((struct.bounds['y'][0]/1000000 + struct.values.max()) / struct.values[90][60][80], 4) == 3.7342

    return struct
