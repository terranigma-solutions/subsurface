import enum
from typing import List
import numpy as np
from subsurface.optional_requirements import require_scipy

from ....core.structs import UnstructuredData, StructuredData

__all__ = ['interpolate_unstructured_data_to_structured_data', ]


class InterpolationMethod(enum.Enum):
    linear = "linear"
    nearest = "nearest"


def interpolate_unstructured_data_to_structured_data(
        ud: UnstructuredData, attr_name: str,
        resolution: List[int] = None,
        interpolation_method: InterpolationMethod = InterpolationMethod.nearest) -> StructuredData:
    if resolution is None:
        resolution = [50, 50, 50]
    boundaries_max = ud.vertex.max(axis=0)
    boundaries_min = ud.vertex.min(axis=0)
    coords = dict()
    dims = ['x', 'y', 'z']
    for e, i in enumerate(dims):
        coords[i] = np.linspace(boundaries_min[e], boundaries_max[e], resolution[e], endpoint=False)

    grid = np.meshgrid(*coords.values())
    scipy = require_scipy()
    interpolated_attributes = scipy.interpolate.griddata(
        points=ud.vertex,
        values=ud.attributes.loc[:, attr_name],
        xi=tuple(grid),
        method=interpolation_method.value
    )

    sd = StructuredData.from_numpy(
        array=interpolated_attributes,
        data_array_name=attr_name,
        coords=coords
    )
    return sd
