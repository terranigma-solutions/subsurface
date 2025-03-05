import os

import pytest
import dotenv

from subsurface import optional_requirements
from subsurface.modules.reader.mesh.obj_reader import load_obj_with_trimesh
from tests.conftest import RequirementsLevel

dotenv.load_dotenv()

path_to_obj = os.getenv("PATH_TO_OBJ")
path_to_mtl = os.getenv("PATH_TO_MTL")
path_to_obj_no_material = os.getenv("PATH_TO_OBJ_GALLERIES_I")

pytestmark = pytest.mark.read_mesh


@pytest.mark.skipif(
    condition=(RequirementsLevel.MESH | RequirementsLevel.PLOT) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH variable to run this test"
)
def test_read_obj_mesh_no_materials():
    pv = optional_requirements.require_pyvista()
    reader: pv.OBJReader = pv.get_reader(path_to_obj_no_material)
    mesh = reader.read()
    # Access texture coordinates if present

    if plot := True:
        mesh.plot()


def test_trimesh_load_obj_with_mtl_submeshes_heavy():
    """This is a heavy test"""
    # Replace these with the actual paths in your environment

    assert os.path.exists(path_to_obj), f"OBJ not found: {path_to_obj}"
    assert os.path.exists(path_to_mtl), f"MTL not found: {path_to_mtl}"
    load_obj_with_trimesh(path_to_obj, plot=True)


def test_trimesh_load_obj_with_mtl_submeshes_II():
    # Replace these with the actual paths in your environment
    path_to_obj = os.getenv("PATH_TO_OBJ_MULTIMATERIAL_II")
    load_obj_with_trimesh(path_to_obj, plot=True)


def test_trimesh_load_obj_with_jpg_texture():
    path_to_obj = os.getenv("TERRA_PATH_DEVOPS") + "/meshes/OBJ/Portugal outcrop decimated/textured_output.obj"
    load_obj_with_trimesh(path_to_obj)


def test_trimesh_load_obj_with_face_I():
    path_to_obj = os.getenv("PATH_TO_OBJ_FACE_I")
    load_obj_with_trimesh(path_to_obj)


def test_trimesh_load_obj_with_face_II():
    path_to_obj = os.getenv("PATH_TO_OBJ_FACE_II")
    load_obj_with_trimesh(path_to_obj)


def test_trimesh_load_obj_boxes():
    path_to_obj = os.getenv("PATH_TO_OBJ_SCANS")
    load_obj_with_trimesh(path_to_obj)


def test_trimesh_load_obj_with_texture_II():
    """Penguin, material exist but png is not loading correctly"""
    path_to_obj = os.getenv("TERRA_PATH_DEVOPS") + "/meshes/OBJ/TexturedMesh/PenguinBaseMesh.obj"
    load_obj_with_trimesh(
        path_to_obj=path_to_obj,
        plot=True
    )
