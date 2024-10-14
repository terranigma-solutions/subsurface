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
    from subsurface.modules.reader.mesh.mx_reader import parse_gocad_mx_file
    # Parse the .mx file
    meshes = parse_gocad_mx_file(os.getenv("PATH_TO_MX"))

    p = init_plotter(image_2d=False)
    # Convert each mesh to PyVista and visualize
    pyvista_meshes = _meshes_to_pyvista(meshes)
    for pv_mesh in pyvista_meshes:
        # Add random colors
        p.add_mesh(pv_mesh, color= pv_mesh.color, show_scalar_bar=False)

    p.show()

def _meshes_to_pyvista(meshes):
    pyvista_meshes = []
    for mesh in meshes:
        # Create index mapping from original to zero-based indices
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(mesh.vertex_indices)}

        # Map triangle indices
        try:
            triangles_mapped = np.vectorize(idx_map.get)(mesh.edges)
        except TypeError as e:
            print(f"Error mapping indices for mesh: {e}")
            continue

        # Create faces array for PyVista
        faces = np.hstack([np.full((triangles_mapped.shape[0], 1), 3), triangles_mapped]).flatten()

        # Create PyVista mesh
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        pv_mesh.color = mesh.color
        pyvista_meshes.append(pv_mesh)
    return pyvista_meshes
