import os

import dotenv
import pytest

import subsurface
import subsurface.modules.visualization as sb_viz
from subsurface import TriSurf

dotenv.load_dotenv()
data_folder = os.getenv("PATH_TO_WEISWEILER")


@pytest.mark.liquid_earth
def test_le_mesh_v1_topo():
    unstruct = subsurface.UnstructuredData.from_binary_le_legacy(
        path_to_binary=data_folder + "topography.le",
        path_to_json=data_folder + "topography.json"
    )

    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)


@pytest.mark.liquid_earth
def test_le_mesh_v1():
    unstruct = subsurface.UnstructuredData.from_binary_le_legacy(
        path_to_binary=data_folder + "gempy_mesh.le",
        path_to_json=data_folder + "gempy_mesh.json"
    )

    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)
