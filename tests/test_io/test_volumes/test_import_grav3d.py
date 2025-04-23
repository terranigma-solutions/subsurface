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
    # Read the mesh file to get grid information
    grid: GridData = read_msh_file(os.getenv("PATH_TO_GRAV3D_MSH"))

    # Verify the grid was loaded correctly
    assert grid is not None
    assert grid.dimensions.nx == 250  # Using the new property name

    # Read the model file to get property values
    array: np.ndarray = read_mod_file(
        filepath=pathlib.Path(os.getenv("PATH_TO_GRAV3D_MOD")),
        grid=grid  # Using the new parameter name
    )

    # Convert the array and grid to a structured data format
    struct = structured_data_from(array, grid)

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
    # Read the mesh file to get grid information
    path_to_folder = os.getenv("PATH_TO_GRAV3D_MSH_II")
    grid: GridData = read_msh_file(path_to_folder + "/mesh.msh")

    # Verify the grid was loaded correctly
    assert grid is not None
    assert grid.dimensions.nx == 100  # Using the new property name

    # Read the model file to get property values
    array: np.ndarray = read_mod_file(
        filepath=pathlib.Path(path_to_folder + "/grav.den"),
        grid=grid,  # Using the new parameter name
        missing_value= -1e+08
    )

    # Convert the array and grid to a structured data format
    struct = structured_data_from(array, grid)
    struct.active_data_array.plot()
    plt.show()

    assert array is not None
    assert array.shape == (139, 100, 46)  # Check the shape of the array

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
    # Read the mesh file to get grid information
    path_to_folder = os.getenv("PATH_TO_GRAV3D_MSH_II")
    grid: GridData = read_msh_file(path_to_folder + "/mesh.msh")

    # Verify the grid was loaded correctly
    assert grid is not None
    assert grid.dimensions.nx == 100  # Using the new property name

    # Read the model file to get property values
    array: np.ndarray = read_mod_file(
        filepath=pathlib.Path(path_to_folder + "/muon.den"),
        grid=grid  # Using the new parameter name
    )
    assert array is not None
    assert array.shape == (139, 100, 46)  # Check the shape of the array
    # Convert the array and grid to a structured data format
    struct = structured_data_from(array, grid)

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



