import os
import pathlib

import dotenv
import numpy as np

import subsurface
from subsurface import StructuredGrid
from subsurface.modules.reader.volume.read_grav3d import (
    GridData, read_msh_file, read_mod_file, structured_data_from
)
from subsurface.modules.visualization import init_plotter

dotenv.load_dotenv()


def test_import_grav3d():
    """
    Test importing and visualizing a Grav3D model.

    This test reads a Grav3D mesh and model file, converts them to a StructuredGrid,
    and visualizes the result using PyVista.
    """
    import pyvista as pv

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
    p = init_plotter(image_2d=False, ve=1, plotter_kwargs=None)
    p.add_volume(
        sg.active_attributes,
        opacity='linear'
    )
    p.add_axes()
    p.add_bounding_box()
    p.show()
