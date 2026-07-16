import pytest
import numpy as np

from tests.conftest import RequirementsLevel
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
from subsurface.core.structs.base_structures import UnstructuredData
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.api import read_point_cloud_to_unstruct

pytestmark = pytest.mark.read_mesh
pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.MESH) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set REQUIREMENT_LEVEL=MESH or REQUIREMENT_LEVEL=ALL"
)


def test_e57_format_detection():
    reader_args = GenericReaderFilesHelper(file_or_buffer="something.e57")
    assert reader_args.format == SupportedFormats.E57


class TestReadE57PointCloud:
    def test_read_two_scans_returns_list(self, data_path):
        """E57 reader returns a list of UnstructuredData, one per scan."""
        path = data_path + '/pointcloud/two_scans.e57'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        result = read_point_cloud_to_unstruct(reader_args)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(ud, UnstructuredData) for ud in result)

    def test_scan_0_has_xyz_and_attributes(self, data_path):
        """Scan 0 (3 points) has XYZ, intensity, and RGB."""
        path = data_path + '/pointcloud/two_scans.e57'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        result = read_point_cloud_to_unstruct(reader_args)

        ud0 = result[0]
        assert ud0.n_points == 3
        assert ud0.cells.shape[1] == 1

        expected = np.array([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 1.0],
                             [2.0, 0.0, 1.0]])
        assert np.allclose(ud0.vertex, expected)

        pa = ud0.points_attributes
        assert 'intensity' in pa.columns
        assert np.allclose(pa['intensity'].values, [0.1, 0.5, 0.9], atol=0.01)

        assert 'red' in pa.columns
        assert list(pa['red']) == [255, 128, 0]

    def test_scan_1_has_xyz_and_intensity_only(self, data_path):
        """Scan 1 (2 points) has XYZ and intensity, no RGB.

        Coordinates are in global space (translation [1,2,3] applied by pye57 read_scan).
        """
        path = data_path + '/pointcloud/two_scans.e57'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        result = read_point_cloud_to_unstruct(reader_args)

        ud1 = result[1]
        assert ud1.n_points == 2

        # raw [10,10,20] + [1,2,3] = [11,12,23], raw [11,10,21] + [1,2,3] = [12,12,24]
        expected = np.array([[11.0, 12.0, 23.0],
                             [12.0, 12.0, 24.0]])
        assert np.allclose(ud1.vertex, expected)

        pa = ud1.points_attributes
        assert 'intensity' in pa.columns
        assert 'red' not in pa.columns

    def test_scan_metadata_preserved(self, data_path):
        """Scan-level metadata is stored in xarray attrs."""
        path = data_path + '/pointcloud/two_scans.e57'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        result = read_point_cloud_to_unstruct(reader_args)

        attrs0 = result[0].data.attrs
        assert attrs0['source_format'] == 'e57'
        assert attrs0['scan_index'] == 0
        assert attrs0['scan_point_count'] == 3
        assert 'scan_translation' in attrs0
        assert attrs0['scan_translation'] == [0.0, 0.0, 0.0]

        attrs1 = result[1].data.attrs
        assert attrs1['scan_index'] == 1
        assert attrs1['scan_point_count'] == 2
        assert attrs1['scan_translation'] == [1.0, 2.0, 3.0]

    def test_pointset_construction(self, data_path):
        """Each scan can be wrapped in a PointSet."""
        path = data_path + '/pointcloud/two_scans.e57'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        result = read_point_cloud_to_unstruct(reader_args)

        for ud in result:
            ps = PointSet(ud)
            assert isinstance(ps.points, np.ndarray)
            assert ps.n_points > 0

    def test_scan_0_pointset_point_data(self, data_path):
        """Scan 0 PointSet exposes intensity and RGB through point_data."""
        path = data_path + '/pointcloud/two_scans.e57'
        reader_args = GenericReaderFilesHelper(file_or_buffer=path)
        result = read_point_cloud_to_unstruct(reader_args)

        ps = PointSet(result[0])
        point_data = ps.point_data
        assert 'intensity' in point_data.columns
        assert 'red' in point_data.columns
        assert len(point_data) == 3
