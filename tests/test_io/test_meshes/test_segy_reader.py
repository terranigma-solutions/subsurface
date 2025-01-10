import time

from typing import List

import pytest
import os

from subsurface import TriSurf, optional_requirements
from subsurface.core.structs.base_structures import StructuredData, UnstructuredData
import matplotlib.pyplot as plt
import numpy as np

from subsurface.modules.reader.profiles.profiles_core import create_vertical_mesh
from subsurface.modules.reader.volume import segy_reader
from subsurface.modules.reader.volume.segy_reader import apply_colormap_to_texture
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot
from tests.conftest import RequirementsLevel

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_VOLUME) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set READ_VOLUME"
)

pv = optional_requirements.require_pyvista()


def _read_segy_file(file_path) -> dict:
    with open(file_path, 'r') as cf:
        lines = cf.readlines()
        xs = []
        ys = []
        for line in lines:
            x = line.split()[1]
            y = line.split()[2]
            xs.append(x)
            ys.append(y)
    return {'x': xs, 'y': ys}


input_path = os.path.dirname(__file__) + '/../../data/segy'
files = ['/E5_MIG_DMO_FINAL.sgy', '/E5_MIG_DMO_FINAL_DEPTH.sgy', '/E5_STACK_DMO_FINAL.sgy', '/test.segy', '/Linie01.segy']
images = ['/myplot_cropped.png', '/myplot2_cropped.png', '/myplot3_cropped.png', '/myplot4_cropped.png']
coords = _read_segy_file(input_path + '/E5_CMP_COORDS.txt')


@pytest.fixture(scope="module")
def get_structured_data() -> List[StructuredData]:
    file_array = [input_path + x for x in files]
    sd_array = [segy_reader.read_in_segy(fp) for fp in file_array]
    return sd_array


@pytest.fixture(scope="module")
def get_images() -> List[str]:
    image_array = [input_path + y for y in images]
    return image_array


def test_converted_to_structured_data(get_structured_data):
    for x in get_structured_data:
        assert isinstance(x, StructuredData)
        x.active_data_array.plot()
        plt.show(block=False)


def test_pyvista_grid(get_structured_data, get_images):
    for s, t in zip(get_structured_data, get_images):

        x = s.data['x']
        y = s.data['y']

        x2, y2 = np.meshgrid(x, y)
        print(x2, y2)
        tex = pv.read_texture(t)
        z = np.zeros((len(y), len(x)))
        # z.reshape(z, (-1, 1101))
        print(x2.shape, y2.shape, z.shape)

        # create a surface to host this texture
        surf = pv.StructuredGrid(z, x2, y2)
        print(surf)

        surf.texture_map_to_plane(inplace=True)
        if False:  # Taking screenshots of pyvista is not handle well by pycharm
            pv_plot([surf], image_2d=True)
            time.sleep(2)
        # use Trisurf with Structured Data for texture and UnstructuredData for geometry


def test_read_segy_to_struct_data_imageio(get_structured_data, get_images):
    imageio = optional_requirements.require_imageio()
    for x, image in zip(get_structured_data, get_images):
        vertex = np.array([[0, x.data['x'][0], x.data['y'][0]], [0, x.data['x'][-1], x.data['y'][0]], [0, x.data['x'][0], x.data['y'][-1]], [0, x.data['x'][-1], x.data['y'][-1]]])
        a = pv.PolyData(vertex)
        b = a.delaunay_2d().faces
        cells = b.reshape(-1, 4)[:, 1:]
        print('cells', cells)
        struct = StructuredData.from_numpy(np.array(imageio.imread(image)))
        unstruct = UnstructuredData.from_array(vertex, cells)
        ts = TriSurf(
            mesh=unstruct,
            texture=struct
        )

        s = to_pyvista_mesh(ts)

        if False:  # Taking screenshots of pyvista is not handle well by pycharm
            pv_plot([s], image_2d=True)
            time.sleep(2)


def test_plot_segy_as_struct_data_with_coords_dict(get_structured_data, get_images):
    imageio = optional_requirements.require_imageio()
    s = []
    i = 0
    for x, image in zip(get_structured_data, get_images):
        zmin = -6000.0
        zmax = 0.0
        v, e = segy_reader.create_mesh_from_coords(coords, zmin, zmax)
        v[:, 0] = + i * 1000

        struct = StructuredData.from_numpy(np.array(imageio.imread(image)))
        print(struct)  # normalize to number of samples
        unstruct = UnstructuredData.from_array(v, e)

        origin = [float(coords['x'][0]), float(coords['y'][0]), zmin]
        point_u = [float(coords['x'][-1]), float(coords['y'][-1]), zmin]
        point_v = [float(coords['x'][0]), float(coords['y'][0]), zmax]
        ts = TriSurf(
            mesh=unstruct,
            texture=struct,
            texture_origin=origin,
            texture_point_u=point_u,
            texture_point_v=point_v
        )

        s.append(to_pyvista_mesh(ts))
        i += 1

    if True:  # Taking screenshots of pyvista is not handle well by pycharm
        pv_plot(meshes=s, image_2d=False)
        time.sleep(2)


def test_seismic_profile():
    # filepath = os.getenv("PATH_TO_SEISMIC")
    sd_array: StructuredData = segy_reader.read_in_segy(
        filepath=(os.getenv("PATH_TO_SEISMIC_FINAL")),
        ignore_geometry=True,
        flip_y_axis=True
    )

    sd_array.active_data_array.plot()
    plt.show(block=False)


def test_seismic_profile_3D_from_segy():
    filepath = os.getenv("PATH_TO_SEISMIC")
    texture: StructuredData = segy_reader.read_in_segy(filepath, ignore_geometry=True)

    # region coords
    import pandas as pd
    file_path = os.getenv("PATH_TO_SECTION")
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        skiprows=4,  # Skip the header lines above 'CDP'
        delim_whitespace=True,  # Treat consecutive spaces as separators
        names=["CDP", "X_COORD", "Y_COORD"]  # Assign column names
    )

    coords = df[["X_COORD", "Y_COORD"]].to_numpy()
    # endregion

    zmin = -450
    zmax = 140
    vertices, faces = create_vertical_mesh(coords, zmin, zmax)
    geometry: UnstructuredData = UnstructuredData.from_array(vertices, faces)

    texture = apply_colormap_to_texture(texture, cmap_name="bwr")
    ts = TriSurf(
        mesh=geometry,
        texture=texture,
        texture_origin=[coords[0][0], coords[0][1], zmin],
        texture_point_u=[coords[-1][0], coords[-1][1], zmin],
        texture_point_v=[coords[0][0], coords[0][1], zmax]
    )

    pv_plot(
        meshes=[to_pyvista_mesh(ts)],
        image_2d=False
    )

def test_seismic_profile_3D_from_interpreted_tiff():

    filepath = os.getenv("PATH_TO_INTERPRETATION")

    import tifffile as tiff  # Install with pip install tifffile

    image = tiff.imread(filepath)

    # Define the crop region: [y_start:y_end, x_start:x_end]
    # Example: Crop a region starting at (100, 100) with a size of 200x300 (height x width)
    y_start, y_end = 83, 730  # Vertical range
    x_start, x_end = 60, 2080  # Horizontal range

    # Perform the crop
    cropped_image = image[y_start:y_end, x_start:x_end]
    texture = StructuredData.from_numpy(cropped_image)

    # region coords
    import pandas as pd
    file_path = os.getenv("PATH_TO_SECTION")
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        skiprows=4,  # Skip the header lines above 'CDP'
        delim_whitespace=True,  # Treat consecutive spaces as separators
        names=["CDP", "X_COORD", "Y_COORD"]  # Assign column names
    )

    coords = df[["X_COORD", "Y_COORD"]].to_numpy()
    # endregion

    zmin = -450
    zmax = 140
    vertices, faces = create_vertical_mesh(coords, zmin, zmax)
    geometry: UnstructuredData = UnstructuredData.from_array(vertices, faces)

    # texture = apply_colormap_to_texture(texture, cmap_name="bwr")
    ts = TriSurf(
        mesh=geometry,
        texture=texture,
        texture_origin=[coords[0][0], coords[0][1], zmin],
        texture_point_u=[coords[-1][0], coords[-1][1], zmin],
        texture_point_v=[coords[0][0], coords[0][1], zmax]
    )

    pv_plot(
        meshes=[to_pyvista_mesh(ts)],
        image_2d=False
    )

def test_interpreted_profile():
    filepath = os.getenv("PATH_TO_INTERPRETATION")

    import matplotlib.pyplot as plt
    import tifffile as tiff  # Install with pip install tifffile

    image = tiff.imread(filepath)

    # Define the crop region: [y_start:y_end, x_start:x_end]
    # Example: Crop a region starting at (100, 100) with a size of 200x300 (height x width)
    y_start, y_end = 83, 730  # Vertical range
    x_start, x_end = 60, 2080  # Horizontal range

    # Perform the crop
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Plot the cropped image
    plt.figure(figsize=(8, 6))
    plt.imshow(cropped_image, cmap='gray')  # Use 'gray' for grayscale images
    plt.colorbar(label='Pixel Intensity')
    plt.title("Cropped TIF Image")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.show()

