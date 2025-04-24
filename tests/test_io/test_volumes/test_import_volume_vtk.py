from __future__ import annotations

import os
import pathlib
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

import subsurface
from subsurface import StructuredGrid, optional_requirements
from subsurface.modules.reader.volume.read_volume import pv_cast_to_explicit_structured_grid
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

    pyvista_struct: pv.ExplicitStructuredGrid = pv_cast_to_explicit_structured_grid(pyvista_obj)

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


def test_vtk_from_numpy():

    import numpy as np
    import pyvista as pv

    ni, nj, nk = 4, 5, 6
    si, sj, sk = 20, 10, 1

    xcorn = np.arange(0, (ni + 1) * si, si)
    xcorn = np.repeat(xcorn, 2)
    xcorn = xcorn[1:-1]
    xcorn = np.tile(xcorn, 4 * nj * nk)

    ycorn = np.arange(0, (nj + 1) * sj, sj)
    ycorn = np.repeat(ycorn, 2)
    ycorn = ycorn[1:-1]
    ycorn = np.tile(ycorn, (2 * ni, 2 * nk))
    ycorn = np.transpose(ycorn)
    ycorn = ycorn.flatten()

    zcorn = np.arange(0, (nk + 1) * sk, sk)
    zcorn = np.repeat(zcorn, 2)
    zcorn = zcorn[1:-1]
    zcorn = np.repeat(zcorn, (4 * ni * nj))

    corners = np.stack((xcorn, ycorn, zcorn))
    corners = corners.transpose()

    dims = np.asarray((ni, nj, nk)) + 1
    print("Dims: ", dims)
    print("Corners: ", corners.shape)
    print("Corners: ", corners)
    
    grid = pv.ExplicitStructuredGrid(dims, corners)
    grid = grid.compute_connectivity()

    # * Attributes
    grid.cell_data['Cell Number'] = range(grid.n_cells)

    # Temp file save
    with NamedTemporaryFile(suffix='.vtk', delete=False) as temp_file:
        grid.save(temp_file.name)
        print(f"VTK file saved to: {temp_file.name}")
    grid.plot(show_edges=True)


def test_vtk_from_numpy_II():
    path = os.getenv("PATH_TO_VOLUME_NUMPY")
    # read npz
    data = np.load(path + "mesh_10m.npz")
    # array
    vertex = data['arr_0']
    # get corners from vertex XYZ array
    corners = np.zeros((vertex.shape[0], 3))
    corners[:, 0] = vertex[:, 0]
    corners[:, 1] = vertex[:, 1]
    corners[:, 2] = vertex[:, 2]
    
    data = np.load(path + "mesh_10m_MacPass_Bayesian_inference_density_model.npz")
    attr = data['data']
    mask = data['mask']

    dims = attr.shape
    import pyvista as pv
    grid = pv.ExplicitStructuredGrid(dims, corners)
    grid = grid.compute_connectivity()

    # * Attributes
    grid.cell_data['Density'] = range(grid.n_cells)

    # Temp file save
    if False:
        with NamedTemporaryFile(suffix='.vtk', delete=False) as temp_file:
            grid.save(temp_file.name)
            print(f"VTK file saved to: {temp_file.name}")
    grid.plot(show_edges=True)

    pass


def test_vtk_file_to_binary():
    struct: subsurface.StructuredData = test_vtk_file_to_structured_data()
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
    pv.examples.download_angular_sector()

    pyvista_struct = pv_cast_to_explicit_structured_grid(pyvista_obj)

    struct: subsurface.StructuredData = subsurface.StructuredData.from_pyvista_structured_grid(
        grid=pyvista_struct,
        data_array_name="model_name"
    )

    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    mesh = to_pyvista_grid(sg)
    pv_plot([mesh], image_2d=True)
    return struct
