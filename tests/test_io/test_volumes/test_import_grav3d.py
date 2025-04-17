import os
import pathlib

import dotenv
import numpy as np

import subsurface
from subsurface import StructuredGrid
from subsurface.modules.reader.volume.read_grav3d import MeshData, read_msh_file, read_mod_file
from subsurface.modules.visualization import to_pyvista_grid, pv_plot

dotenv.load_dotenv()


def test_import_grav3d():
    mesh: MeshData = read_msh_file(os.getenv("PATH_TO_GRAV3D_MSH"))

    assert mesh is not None
    assert mesh.dimensions.ne == 250

    array: np.ndarray = read_mod_file(
        filepath=(pathlib.Path(os.getenv("PATH_TO_GRAV3D_MOD"))),
        mesh=mesh
    )
    
    foo = np.unique(array).min()
    # Compute cell-center coordinates for each axis.
    # For easting, start from origin.x0 and add cumulative cell widths.
    easting = np.array(mesh.cell_sizes.easting)
    x_centers = mesh.origin.x0 + np.cumsum(easting) - easting[0] / 2

    # For northing, start from origin.y0
    northing = np.array(mesh.cell_sizes.northing)
    y_centers = mesh.origin.y0 + np.cumsum(northing) - northing[0] / 2

    # For vertical, note that the top is given by origin.z0 and cells extend downward.
    vertical = np.array(mesh.cell_sizes.vertical)
    z_centers = mesh.origin.z0 - (np.cumsum(vertical) - vertical[0] / 2)

    # Create the DataArray.
    # The array shape is (nn, ne, nz). We use dimension names 'north', 'east' and 'vertical'
    import xarray as xr
    xr_data_array = xr.DataArray(
        data=array,
        dims=['x', 'y', 'z'],
        coords={
            'x': y_centers,
            'y': x_centers,
            'z': z_centers,
        },
        name='model'
    )
    
    xr_data_array.plot()

    # Optionally, wrap the xr.DataArray into a StructuredData instance
    struct: subsurface.StructuredData = subsurface.StructuredData.from_data_array(
        data_array=xr_data_array,
        data_array_name='model'
    )

    # sg: subsurface.StructuredGrid = StructuredGrid(struct)
    # import pyvista as pv
    # pyvista_mesh:pv.StructuredGrid = to_pyvista_grid(sg)
    # pyvista_mesh = pyvista_mesh.threshold(0, scalars="model")
    # 
    # scalars = pyvista_mesh.active_scalars
    # foo = np.unique(scalars)
    # # plot as points
    # plotter = pv.Plotter()
    # plotter.add_mesh(pyvista_mesh)
    # plotter.show()