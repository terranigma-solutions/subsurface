import pytest
import numpy as np
import pandas as pd

from tests.conftest import RequirementsLevel
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
from subsurface.core.structs.base_structures import UnstructuredData
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.modules.reader.mesh.surfaces_api import read_point_cloud_to_unstruct

pytestmark = pytest.mark.read_mesh
pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.MESH) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set REQUIREMENT_LEVEL=MESH or REQUIREMENT_LEVEL=ALL"
)


def test_ply_format_detection():
    reader_args = GenericReaderFilesHelper(file_or_buffer="something.ply")
    assert reader_args.format == SupportedFormats.PLY


class TestReadPlyPointCloud:
    def test_read_ascii_xyz_only(self, data_path):
        path = data_path + '/pointcloud/points_xyz_ascii.ply'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        ud = read_point_cloud_to_unstruct(reader_args)

        assert isinstance(ud, UnstructuredData)
        assert ud.cells.shape[1] == 1
        assert ud.n_points == 4
        assert ud.n_elements == 4

        expected = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0]])
        assert np.allclose(ud.vertex, expected)

        ps = PointSet(ud)
        assert ps.n_points == 4
        assert isinstance(ps.points, np.ndarray)

    def test_read_binary_with_attributes(self, data_path):
        path = data_path + '/pointcloud/points_attributes_binary.ply'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        ud = read_point_cloud_to_unstruct(reader_args)

        assert isinstance(ud, UnstructuredData)
        assert ud.n_points == 3
        assert ud.cells.shape[1] == 1

        pa = ud.points_attributes
        expected_cols = ['red', 'green', 'blue', 'nx', 'ny', 'nz', 'scalar_Mineral']
        for col in expected_cols:
            assert col in pa.columns

        assert list(pa['red']) == [255, 0, 0]
        assert list(pa['green']) == [0, 255, 0]
        assert list(pa['blue']) == [0, 0, 255]

        assert np.allclose(pa['scalar_Mineral'].values, [0.5, 0.8, 0.3], atol=1e-6)

        assert np.allclose(ud.vertex[0], [0.0, 0.0, 0.0])
        assert np.allclose(ud.vertex[1], [1.0, 0.0, 0.0])
        assert np.allclose(ud.vertex[2], [1.0, 1.0, 0.0])

    def test_read_binary_pointset_point_data(self, data_path):
        path = data_path + '/pointcloud/points_attributes_binary.ply'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        ud = read_point_cloud_to_unstruct(reader_args)
        ps = PointSet(ud)

        pd_df = ps.point_data
        assert isinstance(pd_df, pd.DataFrame)
        assert 'red' in pd_df.columns
        assert len(pd_df) == 3

        pd_dict = ps.point_data_dict
        assert len(pd_dict['red']) == 3

    def test_read_binary_xarray_attrs(self, data_path):
        path = data_path + '/pointcloud/points_attributes_binary.ply'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        ud = read_point_cloud_to_unstruct(reader_args)

        assert ud.data.attrs['source_format'] == 'ply'

    def test_reject_ply_with_faces(self, data_path):
        path = data_path + '/pointcloud/points_with_faces.ply'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)

        with pytest.raises(ValueError, match="face"):
            read_point_cloud_to_unstruct(reader_args)

    def test_unsupported_format_raises(self):
        reader_args = GenericReaderFilesHelper(file_or_buffer="something.csv")
        with pytest.raises(ValueError, match="Unsupported point cloud format"):
            read_point_cloud_to_unstruct(reader_args)
