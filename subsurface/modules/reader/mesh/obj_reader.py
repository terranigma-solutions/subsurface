from typing import Union

import numpy as np
import pandas

from subsurface.core.structs import UnstructuredData

import subsurface
from subsurface import optional_requirements, StructuredData, TriSurf


def load_obj_with_trimesh(path_to_obj: str, plot: bool = False) -> Union["trimesh.Trimesh", "trimesh.Scene"]:
    """
    Summary:
        Loads an OBJ file using Trimesh, processes it as either a single mesh or a 
        scene with multiple geometries, and optionally displays the result. The 
        function differentiates between single mesh and scene processing, applying 
        specific handling based on the loaded object's type.
    
    Note: 
        ! This is missing the option to force a png as texture from code. trimesh ignores the uv when the
        ! material does not have an image. This is a limitation of trimesh. We could rewrite the load obj
        ! function in trimesh to allow that

    Arguments:
        path_to_obj: str
            Path to the .obj file to be loaded.
        plot: bool, optional
            Flag indicating whether to display the loaded object using Trimesh's 
            visualization tools. Defaults to False.

    Returns:
        Union[trimesh.Trimesh, trimesh.Scene]
            Returns the loaded Trimesh object as either a single mesh or a scene 
            containing multiple geometries.
    """
    trimesh = optional_requirements.require_trimesh()

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
    
    return scene_or_mesh


def trimesh_obj_to_unstruct(scene_or_mesh: Union["trimesh.Trimesh", "trimesh.Scene"]) -> subsurface.UnstructuredData:
    trimesh = optional_requirements.require_trimesh()
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Process scene with multiple geometries
        unstruct = _unstruct_from_scene(scene_or_mesh, trimesh)

    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        # Process single mesh
        tri = scene_or_mesh
        frame = pandas.DataFrame(tri.face_attributes)
        # Check frame has a valid shape for cells_attr if not make None
        if frame.shape[0] != tri.faces.shape[0]:
            frame = None
            
        # Get UV coordinates if they exist
        vertex_attr = None
        if hasattr(tri.visual, 'uv') and tri.visual.uv is not None:
            vertex_attr = pandas.DataFrame(
                tri.visual.uv,
                columns=['u', 'v']
            )
        
        unstruct = UnstructuredData.from_array(
            np.array(tri.vertices),
            np.array(tri.faces),
            cells_attr=frame,
            vertex_attr=vertex_attr,
            xarray_attributes={
                "bounds": tri.bounds.tolist(),
            },
        )
        
        # If there is a texture
        texture = StructuredData.from_numpy(tri.visual.material.image)
        coords = tri.vertices

        ts = TriSurf(
            mesh=unstruct,
            texture=texture,
            # texture_origin=[coords[0][0], coords[0][1], zmin],
            # texture_point_u=[coords[-1][0], coords[-1][1], zmin],
            # texture_point_v=[coords[0][0], coords[0][1], zmax]
        )


    else:
        raise ValueError("Input must be a Trimesh object or a Scene with multiple geometries.")
        
    return unstruct


def _unstruct_from_scene(scene_or_mesh: 'Scene', trimesh: 'trimesh') -> 'UnstructuredData':
    import pandas as pd
    geometries = scene_or_mesh.geometry
    assert len(geometries) > 0, "No geometries found in the scene."
    all_vertex = []
    all_cells = []
    cell_attr = []
    _last_cell = 0
    for i, (geom_name, geom) in enumerate(geometries.items()):
        geom: trimesh.Trimesh
        _handle_material_info(geom)

        # Append vertices
        all_vertex.append(np.array(geom.vertices))

        # Adjust cell indices and append
        cells = np.array(geom.faces)
        if len(all_cells) > 0:
            cells = cells + _last_cell
        all_cells.append(cells)

        # Create attribute array for this geometry
        cell_attr.append(np.ones(len(cells)) * i)

        _last_cell = cells.max() + 1
    # Create the combined UnstructuredData
    unstruct = UnstructuredData.from_array(
        vertex=np.vstack(all_vertex),
        cells=np.vstack(all_cells),
        cells_attr=pd.DataFrame(np.hstack(cell_attr), columns=["Geometry id"]),
        xarray_attributes={
                "bounds": scene_or_mesh.bounds.tolist(),
        },
    )
    return unstruct


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
