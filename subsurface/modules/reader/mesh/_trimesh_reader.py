from typing import Union, io

import numpy as np
from subsurface.core.structs import UnstructuredData

import subsurface
from subsurface import optional_requirements
from subsurface.core.structs import TriSurf, StructuredData


def _load_with_trimesh(path_to_obj, plot=False):
    plot = True
    trimesh = optional_requirements.require_trimesh()
    # Load the OBJ with Trimesh using the specified options
    scene_or_mesh = trimesh.load(
        file_obj=path_to_obj,
        file_type="obj"
    )
    # Process single mesh vs. scene
    if isinstance(scene_or_mesh, trimesh.Scene):
        print("Loaded a Scene with multiple geometries.")
        _process_scene(scene_or_mesh)
        if plot:
            scene_or_mesh.show()
    else:
        print("Loaded a single Trimesh object.")
        print(f" - Vertices: {len(scene_or_mesh.vertices)}")
        print(f" - Faces: {len(scene_or_mesh.faces)}")
        _handle_material_info(scene_or_mesh)
        if plot:
            scene_or_mesh.show()
    return scene_or_mesh


def trimesh_to_unstruct(scene_or_mesh: Union["trimesh.Trimesh", "trimesh.Scene"]) -> TriSurf:
    """
    Convert a Trimesh or Scene object to a subsurface TriSurf object.

    This function takes either a `trimesh.Trimesh` object or a `trimesh.Scene` 
    object and converts it to a `subsurface.TriSurf` object. If the input is 
    a scene containing multiple geometries, it processes all geometries and 
    combines them into a single TriSurf object. If the input is a single 
    Trimesh object, it directly converts it to a TriSurf object. An error 
    is raised if the input is neither a `trimesh.Trimesh` nor a `trimesh.Scene` 
    object.

    Parameters:
        scene_or_mesh (Union[trimesh.Trimesh, trimesh.Scene]): 
            Input geometry data, either as a Trimesh object representing 
            a single mesh or a Scene object containing multiple geometries.

    Note:
        ! Multimesh with multiple materials will read the uvs but not the textures since in that case is better
        ! to read directly the multiple images (compressed) whenever the user wants to work with them. 

    Returns:
        subsurface.TriSurf: Converted subsurface representation of the 
        provided geometry data.

    Raises:
        ValueError: If the input is neither a `trimesh.Trimesh` object nor 
        a `trimesh.Scene` object.
    """
    trimesh = optional_requirements.require_trimesh()
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Process scene with multiple geometries
        ts = _trisurf_from_scene(scene_or_mesh, trimesh)

    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        ts = _trisurf_from_trimesh(scene_or_mesh)


    else:
        raise ValueError("Input must be a Trimesh object or a Scene with multiple geometries.")

    return ts


def _trisurf_from_trimesh(scene_or_mesh):
    # Process single mesh
    tri = scene_or_mesh
    pandas = optional_requirements.require_pandas()
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

    texture = _extract_texture_from_material(tri)

    ts = TriSurf(
        mesh=unstruct,
        texture=texture,
    )
    return ts


def _trisurf_from_scene(scene_or_mesh: 'Scene', trimesh: 'trimesh') -> TriSurf:
    pandas = optional_requirements.require_pandas()
    geometries = scene_or_mesh.geometry
    assert len(geometries) > 0, "No geometries found in the scene."
    all_vertex = []
    all_cells = []
    cell_attr = []
    all_vertex_attr = []
    _last_cell = 0
    texture = None
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

        # Get UV coordinates if they exist
        if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
            vertex_attr = pandas.DataFrame(
                geom.visual.uv,
                columns=['u', 'v']
            )
            all_vertex_attr.append(vertex_attr)

        # Extract texture from material if it is only one geometry
        if len(geometries) == 1:
            texture = _extract_texture_from_material(geom)

    # Create the combined UnstructuredData
    unstruct = UnstructuredData.from_array(
        vertex=np.vstack(all_vertex),
        cells=np.vstack(all_cells),
        vertex_attr=pandas.concat(all_vertex_attr, ignore_index=True) if len(all_vertex_attr) > 0 else None,
        cells_attr=pandas.DataFrame(np.hstack(cell_attr), columns=["Geometry id"]),
        xarray_attributes={
                "bounds": scene_or_mesh.bounds.tolist(),
        },
    )

    # If there is a texture
    ts = TriSurf(
        mesh=unstruct,
        texture=texture,
    )

    return ts


def _extract_texture_from_material(geom):
    from PIL.JpegImagePlugin import JpegImageFile
    from PIL.PngImagePlugin import PngImageFile
    import trimesh

    array = np.empty(0)
    if isinstance(geom.visual.material, trimesh.visual.material.SimpleMaterial):
        image: JpegImageFile = geom.visual.material.image
        if image is None:
            return None
        array = np.array(image)
    elif isinstance(geom.visual.material, trimesh.visual.material.PBRMaterial):
        image: PngImageFile = geom.visual.material.baseColorTexture
        array = np.array(image.convert('RGBA'))

        if image is None:
            return None
    else:
        raise ValueError(f"Unsupported material type: {type(geom.visual.material)}")

    # Asser that image has 3 channels    assert array.shape[2] == 3    from PIL.PngImagePlugin import PngImageFile
    assert array.shape[2] == 3 or array.shape[2] == 4
    texture = StructuredData.from_numpy(array)
    return texture


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

    print(f"Loaded a Scene with {len(scene.geometry)} geometry object(s).")
    for geom_name, geom in geometries.items():
        print(f" Submesh: {geom_name}")
        print(f"  - Vertices: {len(geom.vertices)}")
        print(f"  - Faces: {len(geom.faces)}")

        print(f"Geometry '{geom_name}':")
        _handle_material_info(geom)


class TriMeshReaderFromBlob:
    @classmethod
    def OBJ_stream_to_trisurf(cls, obj_stream: io.TextIO, mtl_stream: io.BytesIO, texture_stream: io.BytesIO) -> TriSurf:
        """
        Load an OBJ file from a stream and convert it to a TriSurf object.
        
        If space_id and path_in are provided, will attempt to load associated
        MTL and texture files from the same location.
        
        Parameters:
            obj_stream: BytesIO containing the OBJ file data
            space_id: Optional Azure space ID for loading associated files
            path_in: Optional path to the OBJ file for locating related files
        
        Returns:
            TriSurf: The loaded mesh with textures if available
        """
        trimesh = optional_requirements.require_trimesh()
        import tempfile
        import os

        path_in = "foo"
        # If no additional context provided, just load directly from stream
        # TODO: Add the option if no material

        # Create a temporary directory to store associated files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the OBJ content to a temp file
            obj_path = os.path.join(temp_dir, os.path.basename(path_in))
            with open(obj_path, 'wb') as f:
                obj_stream.seek(0)
                f.write(obj_stream.read())
                obj_stream.seek(0)

            # Extract mtl references from the OBJ file
            mtl_files = cls._extract_mtl_references(obj_stream)
            dir_path = os.path.dirname(path_in)

            # Download and save MTL files
            for mtl_file in mtl_files:
                mtl_path = f"{dir_path}/{mtl_file}" if dir_path else mtl_file
                try:
                    # mtl_stream = read_files_from_space(
                    #     path_in=mtl_path,
                    #     space_id=space_id
                    # )

                    # Save the MTL file to temp directory
                    mtl_temp_path = os.path.join(temp_dir, mtl_file)
                    with open(mtl_temp_path, 'wb') as f:
                        f.write(mtl_stream.read())

                    # Extract texture references from MTL
                    mtl_stream.seek(0)
                    texture_files = cls._extract_texture_references(mtl_stream)

                    # Download texture files
                    for texture_file in texture_files:
                        texture_path = f"{dir_path}/{texture_file}" if dir_path else texture_file
                        try:
                            # texture_stream = read_files_from_space(
                            #     path_in=texture_path,
                            #     space_id=space_id
                            # )

                            # Save the texture file to temp directory
                            with open(os.path.join(temp_dir, texture_file), 'wb') as f:
                                f.write(texture_stream.read())
                        except Exception as e:
                            print(f"Failed to load texture {texture_file}: {e}")
                except Exception as e:
                    print(f"Failed to load MTL file {mtl_file}: {e}")

            # Now load the OBJ with all associated files available
            scene_or_mesh = trimesh.load(obj_path)

            # Convert to a TriSurf object
            tri_surf = trimesh_to_unstruct(scene_or_mesh)

        return tri_surf

    @classmethod
    def _extract_mtl_references(cls, obj_stream):
        """Extract MTL file references from an OBJ file."""
        obj_stream.seek(0)
        mtl_files = []

        obj_text = obj_stream.read().decode('utf-8')
        obj_stream.seek(0)

        for line in obj_text.splitlines():
            if line.startswith('mtllib '):
                mtl_name = line.split(None, 1)[1].strip()
                mtl_files.append(mtl_name)

        return mtl_files

    @classmethod
    def _extract_texture_references(cls, mtl_stream):
        """Extract texture file references from an MTL file."""
        mtl_stream.seek(0)
        texture_files = []

        mtl_text = mtl_stream.read().decode('utf-8')
        mtl_stream.seek(0)

        for line in mtl_text.splitlines():
            # Check for texture map definitions
            if any(line.startswith(prefix) for prefix in ['map_Kd ', 'map_Ka ', 'map_Ks ', 'map_Bump ', 'map_d ']):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    texture_name = parts[1].strip()
                    texture_files.append(texture_name)

        return texture_files
