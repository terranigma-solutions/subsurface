import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from subsurface import TriSurf, optional_requirements
from subsurface.core.structs.base_structures import StructuredData, UnstructuredData
from subsurface.modules.reader.profiles.profiles_core import create_mesh_from_trace, \
    create_tri_surf_from_traces_texture, lineset_from_trace
from subsurface.modules.reader.profiles.profiles_core import create_vertical_mesh
from subsurface.modules.reader.volume import segy_reader
from subsurface.modules.reader.volume.segy_reader import apply_colormap_to_texture
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot
from subsurface.modules.visualization import to_pyvista_mesh_and_texture
from tests.conftest import RequirementsLevel


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_PROFILES) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_PROFILES flag to True in the conftest.py file to run this test"
)
def test_read_trace_to_unstruct(data_path):
    gpd = optional_requirements.require_geopandas()
    traces = gpd.read_file(data_path + '/profiles/Traces.shp')
    v, e = create_mesh_from_trace(
        traces.loc[0, 'geometry'],
        traces.loc[0, 'zmax'],
        traces.loc[0, 'zmin']
    )

    unstruct = UnstructuredData.from_array(v, e)

    imageio = optional_requirements.require_imageio()
    cross = imageio.imread(data_path + '/profiles/Profil1_cropped.png')
    struct = StructuredData.from_numpy(np.array(cross))

    origin = [traces.loc[0, 'geometry'].xy[0][0],
              traces.loc[0, 'geometry'].xy[1][0],
              traces.loc[0, 'zmin']]
    point_u = [traces.loc[0, 'geometry'].xy[0][-1],
               traces.loc[0, 'geometry'].xy[1][-1],
               traces.loc[0, 'zmin']]
    point_v = [traces.loc[0, 'geometry'].xy[0][0],
               traces.loc[0, 'geometry'].xy[1][0],
               traces.loc[0, 'zmax']]

    ts = TriSurf(
        mesh=unstruct,
        texture=struct,
        texture_origin=origin,
        texture_point_u=point_u,
        texture_point_v=point_v
    )
    s, uv = to_pyvista_mesh_and_texture(ts)
    pv_plot([s], image_2d=True)

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



@pytest.mark.skipif(
    condition=(RequirementsLevel.TRACES | RequirementsLevel.MESH) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_PROFILES flag to True in the conftest.py file to run this test"
)
def test_tri_surf_from_traces_and_png(data_path):
    us, mesh_list = create_tri_surf_from_traces_texture(
        data_path + '/profiles/Traces.shp',
        path_to_texture=[
                data_path + '/profiles/Profil1_cropped.png',
                data_path + '/profiles/Profil2_cropped.png',
                data_path + '/profiles/Profil3_cropped.png',
                data_path + '/profiles/Profil4_cropped.png',
                data_path + '/profiles/Profil5_cropped.png',
                data_path + '/profiles/Profil6_cropped.png',
                data_path + '/profiles/Profil7_cropped.png',
        ]
    )

    pv_plot(mesh_list, image_2d=True)  # * This plots the uv


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_PROFILES) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_PROFILES flag to True in the conftest.py file to run this test"
)
def test_tri_surf_from_traces_and_png_uv(data_path):
    tri_surf, mesh_list = create_tri_surf_from_traces_texture(
        data_path + '/profiles/Traces.shp',
        path_to_texture=[
                data_path + '/profiles/Profil1_cropped.png',
                data_path + '/profiles/Profil2_cropped.png',
                data_path + '/profiles/Profil3_cropped.png',
                data_path + '/profiles/Profil4_cropped.png',
                data_path + '/profiles/Profil5_cropped.png',
                data_path + '/profiles/Profil6_cropped.png',
                data_path + '/profiles/Profil7_cropped.png',
        ]
    )

    print(tri_surf[0].mesh.points_attributes)
    pv_plot(mesh_list, image_2d=True)

@pytest.mark.skipif(
    condition=(RequirementsLevel.TRACES) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_PROFILES flag to True in the conftest.py file to run this test"
)
def test_line_set_from_trace(data_path):
    m = lineset_from_trace(data_path + '/profiles/Traces.shp')
    pv_plot(m, image_2d=True)


class TestSeismicsProfiles:
    def test_interpreted_profile_seismics(self):
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
