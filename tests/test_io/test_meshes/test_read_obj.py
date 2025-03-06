import os

import pytest
import dotenv

from subsurface import optional_requirements
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


def test_trimesh_load_obj_with_mtl_submeshes():
    # TODO: [ ] Use smaller example
    # Replace these with the actual paths in your environment

    assert os.path.exists(path_to_obj), f"OBJ not found: {path_to_obj}"
    assert os.path.exists(path_to_mtl), f"MTL not found: {path_to_mtl}"
    import trimesh

    # Load the OBJ with Trimesh
    # - By default, trimesh.load will return a Scene if the OBJ has multiple parts/materials
    # - If it's a single mesh, it returns a Trimesh
    scene_or_mesh = trimesh.load(path_to_obj)

    if isinstance(scene_or_mesh, trimesh.Scene):
        print("Loaded a Scene with multiple geometries.")

        # Each geometry in the scene can have its own visual/material
        geometries = scene_or_mesh.geometry
        assert len(geometries) > 0, "No geometries found in the scene."

        for geom_name, geom in geometries.items():
            # 'geom' should be a Trimesh object
            if geom.visual and hasattr(geom.visual, 'material'):
                material = geom.visual.material
                print(f"Geometry '{geom_name}' has material: {material}")
            else:
                print(f"Geometry '{geom_name}' has no material attribute.")

        # Show the scene (opens an interactive viewer if possible):
        # If you're running tests in a headless environment, comment this out
        if PLOT := False:
            scene_or_mesh.show()

    else:
        # Single Trimesh object
        print("Loaded a single Trimesh.")

        if scene_or_mesh.visual and hasattr(scene_or_mesh.visual, 'material'):
            material = scene_or_mesh.visual.material
            print("Trimesh material:", material)
        else:
            print("No material found on this single-mesh object.")

        # Show the mesh (interactive viewer)
        if PLOT := False:
            scene_or_mesh.show()


def test_trimesh_load_obj_with_mtl_submeshes_II():
    # Replace these with the actual paths in your environment
    path_to_obj = os.getenv("PATH_TO_OBJ_MULTIMATERIAL_II")
    import trimesh

    # Load the OBJ with Trimesh
    # - By default, trimesh.load will return a Scene if the OBJ has multiple parts/materials
    # - If it's a single mesh, it returns a Trimesh
    scene_or_mesh = trimesh.load(path_to_obj)

    if isinstance(scene_or_mesh, trimesh.Scene):
        print("Loaded a Scene with multiple geometries.")

        # Each geometry in the scene can have its own visual/material
        geometries = scene_or_mesh.geometry
        assert len(geometries) > 0, "No geometries found in the scene."

        for geom_name, geom in geometries.items():
            # 'geom' should be a Trimesh object
            if geom.visual and hasattr(geom.visual, 'material'):
                material = geom.visual.material
                print(f"Geometry '{geom_name}' has material: {material}")
            else:
                print(f"Geometry '{geom_name}' has no material attribute.")

        # Show the scene (opens an interactive viewer if possible):
        # If you're running tests in a headless environment, comment this out
        if PLOT := True:
            scene_or_mesh.show()

    else:
        # Single Trimesh object
        print("Loaded a single Trimesh.")

        if scene_or_mesh.visual and hasattr(scene_or_mesh.visual, 'material'):
            material = scene_or_mesh.visual.material
            print("Trimesh material:", material)
        else:
            print("No material found on this single-mesh object.")

        # Show the mesh (interactive viewer)
        if PLOT := False:
            scene_or_mesh.show()


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
    raise NotImplementedError("We need to add the option to point to at least one texture directly")
    path_to_obj = os.getenv("TERRA_PATH_DEVOPS") + "/meshes/OBJ/TexturedMesh/PenguinBaseMesh.obj"
    load_obj_with_trimesh(path_to_obj)


def load_obj_with_trimesh(path_to_obj):
    import trimesh

    # Load the OBJ with Trimesh
    # - By default, trimesh.load will return a Scene if the OBJ has multiple parts/materials
    # - If it's a single mesh, it returns a Trimesh
    scene_or_mesh = trimesh.load(path_to_obj)

    if isinstance(scene_or_mesh, trimesh.Scene):
        print("Loaded a Scene with multiple geometries.")

        # Each geometry in the scene can have its own visual/material
        geometries = scene_or_mesh.geometry
        assert len(geometries) > 0, "No geometries found in the scene."

        for geom_name, geom in geometries.items():
            # 'geom' should be a Trimesh object
            if geom.visual and hasattr(geom.visual, 'material'):
                material = geom.visual.material
                print(f"Geometry '{geom_name}' has material: {material}")
            else:
                print(f"Geometry '{geom_name}' has no material attribute.")

        # Show the scene (opens an interactive viewer if possible):
        # If you're running tests in a headless environment, comment this out
        if PLOT := True:
            scene_or_mesh.show()

    else:
        # Single Trimesh object
        print("Loaded a single Trimesh.")

        if scene_or_mesh.visual and hasattr(scene_or_mesh.visual, 'material'):
            material = scene_or_mesh.visual.material
            print("Trimesh material:", material)
        else:
            print("No material found on this single-mesh object.")

        # Show the mesh (interactive viewer)
        if PLOT := True:
            scene_or_mesh.show()
