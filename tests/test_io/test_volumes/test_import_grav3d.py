import os
import pathlib

import dotenv
import numpy as np
from matplotlib import pyplot as plt

import subsurface
from subsurface import StructuredGrid
from subsurface.modules.reader.volume.read_grav3d import MeshData, read_msh_file, read_mod_file
from subsurface.modules.visualization import to_pyvista_grid, pv_plot, init_plotter

dotenv.load_dotenv()


def test_import_grav3d():
    import pyvista as pv
    mesh: MeshData = read_msh_file(os.getenv("PATH_TO_GRAV3D_MSH"))

    assert mesh is not None
    assert mesh.dimensions.ne == 250

    array: np.ndarray = read_mod_file(
        filepath=(pathlib.Path(os.getenv("PATH_TO_GRAV3D_MOD"))),
        mesh=mesh
    )
    # arr = array
    # # arr = np.random.random((100, 100, 100))
    # arr.shape
    # ba = np.unique(arr)
    # vol = pv.ImageData()
    # vol.dimensions = arr.shape
    # vol["array"] = arr.ravel(order="F")
    # 
    # p: pv.Pll = init_plotter(image_2d=False, ve=1, plotter_kwargs=None)
    # 
    # p.add_volume(
    #     vol,
    #     opacity="sigmoid",
    #     # show_axes=True,
    #     # show_bounds=True,
    # )
    # p.show()
    # plot 3d numpy array as structured grid in pyvista
    # vol.plot(
    #     # nan_opacity=0
    #     opacity="sigmoid",
    #     # style="points",
    #     show_axes=True,
    #     show_bounds=True,
    #     volume=True,
    #     # show_edges=True    
    # )
    # # foo

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
    plt.show()

    # Optionally, wrap the xr.DataArray into a StructuredData instance
    struct: subsurface.StructuredData = subsurface.StructuredData.from_data_array(
        data_array=xr_data_array,
        data_array_name='model'
    )

    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    p: pv.Pll = init_plotter(image_2d=False, ve=1, plotter_kwargs=None)
    p.add_volume(
        sg.active_attributes,
        # opacity="sigmoid",
    )
    p.add_axes()
    p.add_bounding_box()
    p.show()

    # volume = to_pyvista_grid(sg)
    # volume.plot(
    #     volume=True,
    #          opacity="sigmoid",
    # )
    # Filter nans

    # p: pv.Pll = init_plotter(image_2d=False, ve=1, plotter_kwargs=None)
    # return 
    # p.add_volume(
    #     volume,
    #     opacity="sigmoid",
    #     scalars="model"
    #     # show_axes=True,
    #     # show_bounds=True,
    # )
    # p.show()
    # mesh.plot(
    #     nan_opacity=0,
    #     style="points"
    # )
    # pv_plot(
    #     meshes=[mesh],
    #     image_2d=False,
    #     plotter_kwargs=
    #     {
    #             "volume": True,
    #     },
    #     add_mesh_kwargs=
    #     {
    #             "nan_color"  : "white",
    #             "nan_opacity": 0,
    #             "scalars"    : "model"
    #     }
    # )
