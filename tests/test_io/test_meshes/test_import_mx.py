import os

import dotenv
import numpy as np
import pytest

from subsurface.modules.visualization import init_plotter

from tests.conftest import RequirementsLevel

dotenv.load_dotenv()

PLOT = True

pytestmark = pytest.mark.skipif(
    condition=RequirementsLevel.PLOT not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH"
)


# Skip if TERRA_PATH_DEVOPS not in .env due to lack of input data
@pytest.mark.skipif(os.getenv("TERRA_PATH_DEVOPS") is None, reason="Need to set the TERRA_PATH_DEVOPS")
def test_read_gocad():
    from subsurface.modules.reader.mesh.read_mx_files import parse_gocad_mx_file, meshes_to_pyvista
    # Parse the .mx file
    meshes = parse_gocad_mx_file(os.getenv("PATH_TO_MX"))

    p = init_plotter(image_2d=False)
    # Convert each mesh to PyVista and visualize
    pyvista_meshes = meshes_to_pyvista(meshes)
    for pv_mesh in pyvista_meshes:
        # Add random colors
        p.add_mesh(pv_mesh, color= pv_mesh.color, show_scalar_bar=False)

    p.show()