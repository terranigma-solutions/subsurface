import os

import pytest
import dotenv

from subsurface import optional_requirements
from subsurface.modules.reader.mesh.obj_reader import load_obj_with_trimesh
from tests.conftest import RequirementsLevel

import subsurface
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot

from subsurface import optional_requirements, TriSurf
from subsurface.modules.reader.mesh.obj_reader import load_obj_with_trimesh, trimesh_obj_to_unstruct

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
    import trimesh

    glb_path = os.getenv("PATH_TO_GLB_COMPLEX")
    # glb_path = os.getenv("PATH_TO_GLB")
    assert os.path.exists(glb_path), f"GLB file does not exist: {glb_path}"

    # Trimesh can load GLB/GLTF natively

    trimesh_obj = load_obj_with_trimesh(glb_path, plot=False)
    ts: subsurface.TriSurf = trimesh_obj_to_unstruct(trimesh_obj)
    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=False)
    return


def test_trimesh_read_glb():
    """
    Test loading a .glb (binary glTF) file with trimesh.
    If it's a scene, we iterate over submeshes; 
    if it's a single mesh, we inspect it directly.
    """

    glb_path = os.getenv("PATH_TO_GLB")
    assert os.path.exists(glb_path), f"GLB file does not exist: {glb_path}"

    # Trimesh can load GLB/GLTF natively

    trimesh_obj = load_obj_with_trimesh(glb_path, plot=False)
    ts: subsurface.TriSurf = trimesh_obj_to_unstruct(trimesh_obj)
    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)
    return 
