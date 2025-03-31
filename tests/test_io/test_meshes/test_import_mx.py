import os

import dotenv
import pytest

import subsurface
import subsurface.modules.visualization as sb_viz
from subsurface import TriSurf
from subsurface.modules.reader.mesh._GOCAD_mesh import GOCADMesh
from subsurface.modules.visualization import init_plotter
from tests.conftest import RequirementsLevel

dotenv.load_dotenv()

PLOT = True

pytestmark = pytest.mark.read_mesh

pytestmark = pytest.mark.skipif(
    condition=RequirementsLevel.PLOT not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH"
)


@pytest.mark.skipif(os.getenv("TERRA_PATH_DEVOPS") is None, reason="Need to set the TERRA_PATH_DEVOPS")
@pytest.mark.liquid_earth
def test_read_gocad_from_file():
    from subsurface.modules.reader.mesh.mx_reader import mx_to_unstruct_from_file
    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(os.getenv("PATH_TO_MX"))
    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)


def _meshes_to_pyvista(meshes: list[GOCADMesh]):
    import pyvista as pv
    pyvista_meshes = []
    for mesh in meshes:
        faces = mesh.vectorized_edges

        # Create PyVista mesh
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        pv_mesh.color = mesh.color
        pyvista_meshes.append(pv_mesh)

    return pyvista_meshes
