from typing import Union

import numpy as np
from subsurface.core.structs import UnstructuredData

import subsurface
from subsurface import optional_requirements, StructuredData, TriSurf

def load_obj_with_trimesh(path_to_obj: str, plot: bool = False) -> Union["trimesh.Trimesh", "trimesh.Scene"]:
    raise NotImplementedError
