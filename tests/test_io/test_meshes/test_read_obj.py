import os

import pytest
import dotenv

import subsurface
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot

from subsurface import optional_requirements
from subsurface.modules.reader.mesh._trimesh_reader import trimesh_to_unstruct, load_with_trimesh, TriMeshTransformations
from ...conftest import RequirementsLevel

dotenv.load_dotenv()

path_to_obj = os.getenv("PATH_TO_OBJ")
path_to_mtl = os.getenv("PATH_TO_MTL")
path_to_obj_no_material = os.getenv("PATH_TO_OBJ_GALLERIES_I")

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_MESH) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_WELL"
)


def test_read_obj_mesh_no_materials():
    pv = optional_requirements.require_pyvista()
    reader: pv.OBJReader = pv.get_reader(path_to_obj_no_material)
    mesh = reader.read()
    # Access texture coordinates if present

    if plot := False:
        mesh.plot()


def test_trimesh_load_obj_with_mtl_submeshes_heavy():
    """This is a heavy test"""
    # Replace these with the actual paths in your environment

    assert os.path.exists(path_to_obj), f"OBJ not found: {path_to_obj}"
    assert os.path.exists(path_to_mtl), f"MTL not found: {path_to_mtl}"
    load_with_trimesh(path_to_obj, plot=False)


def test_trimesh_load_obj_with_mtl_submeshes_II():
    # Replace these with the actual paths in your environment
    path_to_obj = os.getenv("PATH_TO_OBJ_MULTIMATERIAL_II")
    load_with_trimesh(path_to_obj, plot=False)


def test_trimesh_load_obj_with_jpg_texture():
    path_to_obj = os.getenv("TERRA_PATH_DEVOPS") + "/meshes/OBJ/Portugal outcrop decimated/textured_output.obj"
    trimesh_obj = load_with_trimesh(
        path_to_file_or_buffer=path_to_obj,
        coordinate_system=TriMeshTransformations.RIGHT_HANDED_Z_UP_Y_REVERSED,
    )

    ts = trimesh_to_unstruct(trimesh_obj)

    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)


def test_trimesh_load_obj_with_face_I():
    path_to_obj = os.getenv("PATH_TO_OBJ_FACE_I")
    load_with_trimesh(path_to_obj, plot=False)


def test_trimesh_load_obj_with_face_II():
    path_to_obj = os.getenv("PATH_TO_OBJ_FACE_II")
    load_with_trimesh(path_to_obj, plot=False)


def test_trimesh_load_obj_boxes():
    path_to_obj = os.getenv("PATH_TO_OBJ_SCANS")
    load_with_trimesh(path_to_obj)


def test_trimesh_one_element_no_texture_to_unstruct():
    path_to_obj = os.getenv("TERRA_PATH_DEVOPS") + "/meshes/OBJ/TexturedMesh/PenguinBaseMesh.obj"
    trimesh_obj = load_with_trimesh(
        path_to_file_or_buffer=path_to_obj,
        plot=False
    )
    ts: subsurface.TriSurf = trimesh_to_unstruct(trimesh_obj)

    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)


def test_trimesh_three_element_no_texture_to_unstruct():
    path_to_obj = os.getenv("PATH_TO_OBJ_MULTIMATERIAL_II")
    trimesh_obj = load_with_trimesh(
        path_to_file_or_buffer=path_to_obj,
        coordinate_system=TriMeshTransformations.RIGHT_HANDED_Z_UP_Y_REVERSED,
    )

    ts = trimesh_to_unstruct(trimesh_obj)

    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)


def test_trimesh_ONE_element_texture_to_unstruct():
    trimesh_obj = load_with_trimesh(
        path_to_file_or_buffer=(os.getenv("PATH_TO_OBJ_FACE_II")),
        plot=False
    )

    ts: subsurface.TriSurf = trimesh_to_unstruct(trimesh_obj)

    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)


def test_trimesh_three_element_texture_to_unstruct():
    """This prints only the uv since we do not want to read
    multiple images as structured objects
    """
    path_to_obj = os.getenv("PATH_TO_OBJ_SCANS")
    trimesh_obj = load_with_trimesh(
        path_to_obj,
        coordinate_system=TriMeshTransformations.ORIGINAL
    )

    ts = trimesh_to_unstruct(trimesh_obj)

    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)
