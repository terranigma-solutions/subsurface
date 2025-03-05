import pyvista as pv
import numpy as np
import os

from subsurface import optional_requirements


def load_obj_with_trimesh(path_to_obj, plot=False):
    """
    Load an OBJ file with optional texture path override.
    We explicitly inject the texture into each geometry's material
    if a texture_path is provided.
    """
    trimesh = optional_requirements.require_trimesh()

    # Load the OBJ with Trimesh, ignoring material_imgpath
    # because it often doesn't override properly for Wavefront
    # Prepare load options
    # If the user explicitly provides a texture, we tell Trimesh

    # Load the OBJ with Trimesh using the specified options
    scene_or_mesh = trimesh.load(path_to_obj)

    # Process single mesh vs. scene
    if isinstance(scene_or_mesh, trimesh.Scene):
        print("Loaded a Scene with multiple geometries.")
        _process_scene(scene_or_mesh)
        if plot:
            scene_or_mesh.show()
    else:
        print("Loaded a single Trimesh.")
        _handle_material_info(scene_or_mesh)
        if plot:
            scene_or_mesh.show()


def _validate_texture_path(texture_path):
    """Validate the texture file path."""
    if texture_path and not texture_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Texture path must be a PNG or JPEG file")


def _handle_material_info(geometry):
    """
    Handle and print material information for a single geometry,
    explicitly injecting the PIL image if provided.
    """
    if geometry.visual and hasattr(geometry.visual, 'material'):
        material = geometry.visual.material

        print("Trimesh material:", material)

        # If there's already an image reference in the material, let the user know
        if hasattr(material, 'image') and material.image is not None:
            print("  -> Material already has an image:", material.image)
    else:
        print("No material found or no 'material' attribute on this geometry.")


def _process_scene(scene):
    """Process a scene with multiple geometries."""
    geometries = scene.geometry
    assert len(geometries) > 0, "No geometries found in the scene."

    for geom_name, geom in geometries.items():
        print(f"Geometry '{geom_name}':")
        _handle_material_info(geom)
