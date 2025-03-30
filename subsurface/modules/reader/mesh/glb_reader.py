import io
from typing import Union

from ....core.structs import TriSurf
from ._trimesh_reader import load_with_trimesh, trimesh_to_unstruct, TriMeshTransformations


def load_gltf_with_trimesh(path_to_glb: Union[str | io.BytesIO], coordinate_system: TriMeshTransformations) -> TriSurf:
    """
    load_obj_with_trimesh(path_to_glb, plot=False)

    Loads a 3D object file in .glb format using trimesh, processes it, and converts it into 
    a subsurface TriSurf object for further analysis or usage. Optionally, it allows 
    plotting the loaded object.

    Parameters
    ----------
    path_to_glb : str
        Path to the .glb file containing the 3D object.
    plot : bool, optional
        A flag indicating whether the loaded 3D object should be plotted. Defaults to False.

    Returns
    -------
    subsurface.TriSurf
        A TriSurf object representing the processed 3D surface geometry.
    """
    trimesh = load_with_trimesh(path_to_glb, file_type="glb", coordinate_system=coordinate_system, plot=False)
    trisurf = trimesh_to_unstruct(trimesh)
    return trisurf
