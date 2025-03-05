import os

import pytest
import dotenv

from subsurface import optional_requirements
from tests.conftest import RequirementsLevel

dotenv.load_dotenv()

pytestmark = pytest.mark.read_mesh


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
    import trimesh

    glb_path = os.getenv("PATH_TO_GLB_COMPLEX")
    # glb_path = os.getenv("PATH_TO_GLB")
    assert os.path.exists(glb_path), f"GLB file does not exist: {glb_path}"

    # Trimesh can load GLB/GLTF natively
    scene_or_mesh = trimesh.load(glb_path)

    # Check if we got a scene or a single mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        print(f"Loaded a Scene with {len(scene_or_mesh.geometry)} geometry object(s).")

        for geom_name, geom in scene_or_mesh.geometry.items():
            print(f" Submesh: {geom_name}")
            print(f"  - Vertices: {len(geom.vertices)}")
            print(f"  - Faces: {len(geom.faces)}")

            # Check if we have a material/visual
            if geom.visual and hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                print(f"  - Material: {mat}")
            else:
                print("  - No material data on this submesh.")

        # Optionally show the scene (only works in a GUI-friendly environment)
        if PLOT := True:
            scene_or_mesh.show()

    else:
        # A single Trimesh object
        mesh = scene_or_mesh
        print("Loaded a single Trimesh object.")
        print(f" - Vertices: {len(mesh.vertices)}")
        print(f" - Faces: {len(mesh.faces)}")

        # Check material
        if mesh.visual and hasattr(mesh.visual, 'material'):
            mat = mesh.visual.material
            print(f"Material: {mat}")
        else:
            print("No material data on this mesh.")

        # Optionally show the scene (only works in a GUI-friendly environment)
        if PLOT := True:
            scene_or_mesh.show()