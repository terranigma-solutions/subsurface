import os

import dotenv
import pytest

import subsurface
from subsurface.modules.reader.mesh._trimesh_reader import TriMeshTransformations
from subsurface.modules.reader.mesh.glb_reader import load_gltf_with_trimesh
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot
from tests.conftest import RequirementsLevel

dotenv.load_dotenv()

pytestmark = pytest.mark.read_mesh


@pytest.mark.skipif(
    condition=(RequirementsLevel.MESH | RequirementsLevel.PLOT) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH variable to run this test"
)
def test_trimesh_read_glb_complex():
    """
    Test loading a .glb (binary glTF) file with trimesh.
    If it's a scene, we iterate over submeshes; 
    if it's a single mesh, we inspect it directly.
    """

    glb_path = os.getenv("PATH_TO_GLB_COMPLEX")
    # glb_path = os.getenv("PATH_TO_GLB")
    assert os.path.exists(glb_path), f"GLB file does not exist: {glb_path}"

    # Trimesh can load GLB/GLTF natively

    ts: subsurface.TriSurf = load_gltf_with_trimesh(glb_path, TriMeshTransformations.RIGHT_HANDED_Z_UP)
    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)
    return


@pytest.mark.skipif(
    condition=(RequirementsLevel.MESH | RequirementsLevel.PLOT) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH variable to run this test"
)
def test_trimesh_read_glb():
    """
    Test loading a .glb (binary glTF) file with trimesh.
    If it's a scene, we iterate over submeshes; 
    if it's a single mesh, we inspect it directly.
    """

    glb_path = os.getenv("PATH_TO_GLB")
    assert os.path.exists(glb_path), f"GLB file does not exist: {glb_path}"

    # Trimesh can load GLB/GLTF natively

    ts = load_gltf_with_trimesh(glb_path, TriMeshTransformations.RIGHT_HANDED_Z_UP)
    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)
    return 
