import os
import pathlib

import dotenv
import numpy as np

import subsurface
from subsurface import StructuredGrid
from subsurface.modules.reader.volume.read_grav3d import MeshData, read_msh_file, read_mod_file, structured_data_from
from subsurface.modules.visualization import init_plotter

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
    
    # Compute cell-center coordinates for each axis.
    # For easting, start from origin.x0 and add cumulative cell widths.
    struct = structured_data_from(array, mesh)

    sg: subsurface.StructuredGrid = StructuredGrid(struct)

    p: pv.Pll = init_plotter(image_2d=False, ve=1, plotter_kwargs=None)
    p.add_volume(
        sg.active_attributes,
        opacity='linear'
    )
    p.add_axes()
    p.add_bounding_box()
    p.show()


