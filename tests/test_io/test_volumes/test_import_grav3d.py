import os
import pathlib

import dotenv
import numpy as np
from matplotlib import pyplot as plt

import subsurface
from subsurface import StructuredGrid
from subsurface.modules.reader.volume.read_grav3d import (
    GridData, read_msh_file, read_mod_file, structured_data_from
)

from subsurface.modules.visualization import init_plotter, pyvista_to_matplotlib

dotenv.load_dotenv()


def test_import_grav3d():
    """
    Test importing and visualizing a Grav3D model.

    This test reads a Grav3D mesh and model file, converts them to a StructuredGrid,
    and visualizes the result using PyVista.
    """

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))

    # Read the mesh file to get grid information
    msh_filepath = devops_path.joinpath(r"volume/MSH/IDN-0/grav3d.msh")
    grid: GridData = read_msh_file(msh_filepath)

    # Verify the grid was loaded correctly
    assert grid is not None
    assert grid.dimensions.nx == 250  # Using the new property name

    # Read the model file to get property values
    mod_filepath = devops_path.joinpath(r"volume/MSH/IDN-0/grav3d.mod")
    array: np.ndarray = read_mod_file(
        filepath=mod_filepath,
        grid=grid  # Using the new parameter name
    )

    # Convert the array and grid to a structured data format
    struct = structured_data_from(array, grid)

    assert struct.shape == (250, 222, 70)
    assert round(10 * struct.bounds['x'][0] / struct.bounds['y'][1], 4) == 1.6245
    real_values = struct.values[~np.isnan(struct.values)]
    assert round(real_values.max() / real_values.min(), 4) == 1.4025

    # Create a StructuredGrid from the structured data
    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    # Visualize the grid

    image_2d = True 
    p = init_plotter(image_2d=image_2d, ve=1, plotter_kwargs=None)
    p.add_volume(
        sg.active_attributes,
        opacity='linear'
    )
    p.add_axes()
    p.add_bounding_box()
    if image_2d is False:
        p.show()
    else:
        pyvista_to_matplotlib(p)


def test_import_grav3d_II():
    """
    Test importing and visualizing a Grav3D model.

    This test reads a Grav3D mesh and model file, converts them to a StructuredGrid,
    and visualizes the result using PyVista.
    """

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    folder_path = devops_path.joinpath(r"volume/MSH/IDN-65/ubc_mesh_mod/")

    # Read the mesh file to get grid information
    msh_filepath = folder_path.joinpath(r"mesh.msh")
    grid: GridData = read_msh_file(msh_filepath)

    # Verify the grid was loaded correctly
    assert grid is not None
    assert grid.dimensions.nx == 100  # Using the new property name

    # Read the model file to get property values
    den_filepath = folder_path.joinpath(r"grav.den")
    array: np.ndarray = read_mod_file(
        filepath=den_filepath,
        grid=grid,  # Using the new parameter name
        missing_value= -1e+08
    )

    # Convert the array and grid to a structured data format
    struct = structured_data_from(array, grid)
    struct.active_data_array.plot()
    plt.show()

    assert array is not None
    assert array.shape == (100, 139, 46)  # Check the shape of the array
    assert round(100 * struct.bounds['x'][0] / struct.bounds['y'][1], 4) == 5.1575
    real_values = struct.values[~np.isnan(struct.values)]
    assert round(real_values.max() / real_values.min(), 4) == 1.1209

    # Create a StructuredGrid from the structured data
    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    # Visualize the grid
    image_2d = True
    p = init_plotter(image_2d=image_2d, ve=1, plotter_kwargs=None)
    p.add_volume(
        sg.active_attributes,
        opacity='linear'
    )
    p.add_axes()
    p.add_bounding_box()
    image_2d = True
    if image_2d is False:
        p.show()
    else:
        pyvista_to_matplotlib(p)



def test_import_grav3d_III():
    """
    Test importing and visualizing a Grav3D model.

    This test reads a Grav3D mesh and model file, converts them to a StructuredGrid,
    and visualizes the result using PyVista.
    """

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    folder_path = devops_path.joinpath(r"volume/MSH/IDN-65/ubc_mesh_mod/")

    # Read the mesh file to get grid information
    msh_filepath = folder_path.joinpath(r"mesh.msh")
    grid: GridData = read_msh_file(msh_filepath)

    # Verify the grid was loaded correctly
    assert grid is not None
    assert grid.dimensions.nx == 100  # Using the new property name

    # Read the model file to get property values
    den_filepath = folder_path.joinpath(r"muon.den")
    array: np.ndarray = read_mod_file(
        filepath=pathlib.Path(den_filepath),
        grid=grid  # Using the new parameter name
    )
    assert array is not None
    assert array.shape == (100, 139, 46)  # Check the shape of the array
    # Convert the array and grid to a structured data format
    struct = structured_data_from(array, grid)

    assert struct.shape == (100, 139, 46)  # Check the shape of the array
    assert round(100 * struct.bounds['x'][0] / struct.bounds['y'][1], 4) == 5.1575
    real_values = struct.values[~np.isnan(struct.values)]
    assert round(real_values.max()/real_values.min(), 4) == 5.2809

    # Create a StructuredGrid from the structured data
    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    # Visualize the grid
    image_2d = True
    p = init_plotter(image_2d=image_2d, ve=1, plotter_kwargs=None)
    p.add_volume(
        sg.active_attributes,
        opacity='linear'
    )
    p.add_axes()
    p.add_bounding_box()
    image_2d = True
    if image_2d is False:
        p.show()
    else:
        pyvista_to_matplotlib(p)



def test_import_grav3d_IV():
    """
    Test importing and visualizing a Grav3D model.

    This test reads a Grav3D mesh and model file, converts them to a StructuredGrid,
    and visualizes the result using PyVista.
    """

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    folder_path = devops_path.joinpath(r"volume/MSH/")

    # Read the mesh file to get grid information
    msh_filepath = folder_path.joinpath(r"test_simple.msh")
    grid: GridData = read_msh_file(msh_filepath)

    # Verify the grid was loaded correctly
    assert grid is not None

    # Read the model file to get property values
    mod_filepath = folder_path.joinpath(r"test_simple.mod")
    array: np.ndarray = read_mod_file(
        filepath=mod_filepath,
        grid=grid  # Using the new parameter name
    )

    # Convert the array and grid to a structured data format
    struct = structured_data_from(array, grid)

    assert struct.shape == (2, 2, 3)  # Check the shape of the array
    assert struct.bounds == {'x': (5.0, 25.0), 'y': (7.5, 32.5), 'z': (72.5, 97.5)}
    assert struct.values.min() == 1.0
    assert struct.values.max() == 12.0
    assert struct.values[1][1][2] == 10.0

    # Create a StructuredGrid from the structured data
    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    # Visualize the grid

    image_2d = True
    p = init_plotter(image_2d=image_2d, ve=1, plotter_kwargs=None)
    p.add_volume(
        sg.active_attributes,
        opacity=1,
    )
    p.add_axes()
    p.add_bounding_box()
    if image_2d is False:
        p.show()
    else:
        pyvista_to_matplotlib(p)
