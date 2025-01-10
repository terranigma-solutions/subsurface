from typing import Union

from subsurface import optional_requirements
from ....core.structs.base_structures import StructuredData
import numpy as np


def read_in_segy(filepath: str, ignore_geometry: bool = True, flip_y_axis: bool = True) -> StructuredData:
    """
    Reads a SEG-Y file and processes its data into a structured format.

    This function opens a SEG-Y file using the `segyio` library and extracts 
    the trace data. The data is optionally flipped along the y-axis and its 
    axes are rearranged before being converted into a `StructuredData` object. 
    The `ignore_geometry` option determines whether SEG-Y header geometry 
    information is considered during file read.

    Args:
        filepath (str): The path to the SEG-Y file to be read.
        ignore_geometry (bool): Whether to ignore the geometry information 
            from the SEG-Y header. Defaults to True.
        flip_y_axis (bool): Whether to flip the data vertically (on the y-axis). 
            Defaults to True.

    Returns:
        StructuredData: An instance of `StructuredData` containing the 
        processed trace data from the SEG-Y file.

    Raises:
        IOError: If there is an error in opening or reading the SEG-Y file.
        ValueError: If the data cannot be processed into the expected format.
    """
    segyio = optional_requirements.require_segyio()
    segyfile = segyio.open(filepath, ignore_geometry=ignore_geometry)

    data = np.asarray([np.copy(tr) for tr in segyfile.trace[:]])
    
    if flip_y_axis:
        data = np.flip(data, axis=1)
        
    data = np.swapaxes(data, 0, 1)
    
    sd = StructuredData.from_numpy(data)  # data holds traces * (samples per trace) values
    segyfile.close()
    return sd


def create_mesh_from_coords(coords: dict, zmin: Union[float, int], zmax: Union[float, int] = 0.0):
    """Creates a mesh for plotting StructuredData

    Args:
        coords (Union[dict, LineString]): the x and y, i.e. latitude and longitude, location of the traces of the seismic profile
        zmax (float): the maximum elevation of the seismic profile, by default 0.0
        zmin (float): the location in z where the lowest sample was taken

    Returns: vertices and faces for creating an UnstructuredData object

    """
    n = len(coords['x'])
    coords = np.array([coords['x'], coords['y']]).T
    # duplicating the line, once with z=lower and another with z=upper values
    vertices = np.zeros((2 * n, 3))
    vertices[:n, :2] = coords
    vertices[:n, 2] = zmin
    vertices[n:, :2] = coords
    vertices[n:, 2] = zmax
    # i+n --- i+n+1
    # |\      |
    # | \     |
    # |  \    |
    # |   \   |
    # i  --- i+1

    scipy = optional_requirements.require_scipy()
    tri = scipy.spatial.qhull.Delaunay(vertices[:, [0, 2]])
    faces = tri.simplices
    return vertices, faces


def apply_colormap_to_texture(texture: StructuredData, cmap_name="bwr"):
    """
    Convert a single-channel seismic texture.data array into RGB
    using a Matplotlib colormap (e.g., 'bwr', 'jet', 'RdBu_r', etc.).
    """
    # 'texture.data' should be a 2D array of amplitudes: shape = (height, width)
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    data = texture.values
    # 1. Normalize data to [0,1] range based on its min/max
    min_val, max_val = data.min(), data.max()
    norm = colors.Normalize(vmin=min_val, vmax=max_val)

    norm = colors.Normalize(vmin=-6, vmax=6)

    # 2. Get a Matplotlib colormap
    cmap = plt.get_cmap(cmap_name)

    # 3. Map normalized data -> RGBA, shape becomes (height, width, 4)
    rgba_data = cmap(norm(data))

    # 4. Convert from float in [0,1] to uint8 in [0,255], and drop alpha channel
    rgb_data = (rgba_data[..., :3] * 255).astype(np.uint8)

    texture = StructuredData.from_numpy(rgb_data)
    return texture
