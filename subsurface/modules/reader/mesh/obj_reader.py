from typing import Union, TextIO
import io

from ._trimesh_reader import _load_with_trimesh, trimesh_to_unstruct, TriMeshReaderFromBlob
from ....core.structs import TriSurf



def load_obj_with_trimesh_from_binary(obj_stream: TextIO, mtl_stream: list[TextIO], texture_stream: list[io.BytesIO]) -> TriSurf:
    tri_surf: TriSurf = TriMeshReaderFromBlob.OBJ_stream_to_trisurf(
        obj_stream=obj_stream,
        mtl_stream=mtl_stream,
        texture_stream=texture_stream
    )
    
    return tri_surf
    

def load_obj_with_trimesh(path_to_obj: str, plot: bool = False) -> TriSurf:
    """
    Load and process an OBJ file, returning trimesh-compatible objects.

    This function loads an OBJ file using `trimesh`, optionally plots it,
    and converts the loaded mesh or scene into a suitable unstructured format
    using the `trimesh_to_unstruct` function. Depending on the input and the 
    contents of the OBJ file, it may return a Trimesh object or a Scene object.

    Note: 
    This implementation does not include the capability to force a PNG as a 
    texture if the material does not already have an associated image. 
    `trimesh` ignores UVs when this condition occurs. Modifications to 
    `trimesh`'s loading function would be necessary to address this limitation.

    Args:
        path_to_obj: Path to the OBJ file to be loaded.
                     This must be a valid file path to a 3D object in OBJ format.
        plot: Boolean flag indicating whether to visually plot the loaded model.
              Defaults to False.

    Returns:
        A `trimesh.Trimesh` object if a single mesh is loaded, or a `trimesh.Scene`
        object if the file contains multiple meshes or a scene.

    Raises:
        `FileNotFoundError`: If the provided file path does not exist.
        `ValueError`: If the OBJ file could not be properly processed.

    """
    trimesh = _load_with_trimesh(path_to_obj, plot)
    trisurf = trimesh_to_unstruct(trimesh)
    return trisurf
