import pytest

from tests.conftest import RequirementsLevel
from subsurface import UnstructuredData, TriSurf, StructuredData, optional_requirements
from subsurface.modules.reader.profiles.profiles_core import create_mesh_from_trace, \
    create_tri_surf_from_traces_texture, lineset_from_trace
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot, to_pyvista_mesh_and_texture
import numpy as np


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
