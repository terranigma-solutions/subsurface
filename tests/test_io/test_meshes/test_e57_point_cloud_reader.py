import os

import pytest
import numpy as np

from tests.conftest import RequirementsLevel, check_requirements
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
from subsurface.core.structs.base_structures import UnstructuredData
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.api import read_point_cloud_to_unstruct

pytestmark = pytest.mark.slow


def _build_e57_path():
    devops_path = os.getenv("TERRA_PATH_DEVOPS")
    if devops_path:
        primary = os.path.join(devops_path, "point cloud", "parking-lot-updated.e57")
        if os.path.exists(primary):
            return primary
    fallback = os.path.expanduser(
        "~/.cache/rclone/vfs/terranigma/DevOps/SubsurfaceTestData/point cloud/parking-lot-updated.e57"
    )
    if os.path.exists(fallback):
        return fallback
    direct = os.path.expanduser(
        "/home/leguark/Data/OneDrive/Terranigma/DevOps/SubsurfaceTestData/point cloud/parking-lot-updated.e57"
    )
    if os.path.exists(direct):
        return direct
    return None


_needs_e57_file = pytest.mark.skipif(
    _build_e57_path() is None,
    reason="parking-lot-updated.e57 not found. Set TERRA_PATH_DEVOPS or ensure OneDrive is mounted.",
)


def test_e57_format_detection():
    reader_args = GenericReaderFilesHelper(file_or_buffer="something.e57")
    assert reader_args.format == SupportedFormats.E57


@pytest.fixture(scope="module")
def e57_scans():
    path = _build_e57_path()
    if path is None:
        pytest.skip("parking-lot-updated.e57 not found")
    reader_args = GenericReaderFilesHelper(file_or_buffer=path)
    return read_point_cloud_to_unstruct(reader_args)


@pytest.mark.skipif(
    condition=check_requirements(RequirementsLevel.MESH | RequirementsLevel.PLOT),
    reason="Need to set REQUIREMENT_LEVEL=MESH or REQUIREMENT_LEVEL=ALL"
)
class TestReadE57ParkingLot:
    """
    Integration test against the real parking-lot-updated.e57 dataset.

    The file contains 19 scans from a Leica ScanStation 2:
      - 3 large scans (~2.5M points each) from different scanner positions
      - 16 smaller registration-target scans (~1.4K points each)

    Each scan carries global-space XYZ, intensity, RGB, row/column indices,
    and per-scan pose metadata (rotation matrix and translation vector).

    Requires pye57 (pip install subsurface-terra[pointcloud]) and pyvista.
    The dataset must be available at TERRA_PATH_DEVOPS or the OneDrive path.
    """

    @_needs_e57_file
    def test_returns_list_of_nineteen_scans(self, e57_scans):
        """E57 reader produces one UnstructuredData per scan."""
        assert isinstance(e57_scans, list)
        assert len(e57_scans) == 19
        assert all(isinstance(ud, UnstructuredData) for ud in e57_scans)

    @_needs_e57_file
    def test_all_scans_are_point_clouds(self, e57_scans):
        """Every scan has point cells (cells.shape[1] == 1)."""
        for i, ud in enumerate(e57_scans):
            assert ud.cells.shape[1] == 1, f"Scan {i} is not a point cloud"

    @_needs_e57_file
    def test_large_scans_have_expected_attributes(self, e57_scans):
        """Large scans carry intensity, RGB, and row/column indices."""
        large_scan_indices = [0, 7, 11]  # ~2.5M-point scans
        for idx in large_scan_indices:
            ud = e57_scans[idx]
            assert ud.n_points > 1_000_000, f"Scan {idx} has {ud.n_points} points, expected >1M"
            pa = ud.points_attributes
            expected_cols = ['intensity', 'red', 'green', 'blue', 'rowIndex', 'columnIndex']
            for col in expected_cols:
                assert col in pa.columns, f"Scan {idx} missing column {col}"

    @_needs_e57_file
    def test_scan_metadata_preserved(self, e57_scans):
        """Every scan stores source_format, scan_index, point_count, and pose."""
        for i, ud in enumerate(e57_scans):
            attrs = ud.data.attrs
            assert attrs.get('source_format') == 'e57', f"Scan {i}"
            assert attrs.get('scan_index') == i, f"Scan {i}"
            # header point_count is raw total; read_scan filters invalid points
            assert attrs.get('scan_point_count') >= ud.n_points, f"Scan {i}"
            assert 'scan_translation' in attrs, f"Scan {i}"
            assert isinstance(attrs['scan_translation'], list), f"Scan {i}"

    @_needs_e57_file
    def test_total_point_count_is_plausible(self, e57_scans):
        """Combined point count exceeds 7M (the dataset totals ~7.9M)."""
        total = sum(ud.n_points for ud in e57_scans)
        assert total > 7_000_000, f"Total point count {total} below expected minimum"

    @_needs_e57_file
    def test_pointset_wraps_all_scans(self, e57_scans):
        """Every scan can be wrapped in a PointSet."""
        for i, ud in enumerate(e57_scans):
            ps = PointSet(ud)
            assert isinstance(ps.points, np.ndarray), f"Scan {i}"
            assert ps.n_points > 0, f"Scan {i}"

    @_needs_e57_file
    def test_pointset_exposes_vertex_attributes(self, e57_scans):
        """PointSet.point_data exposes intensity and RGB from the reader."""
        ps = PointSet(e57_scans[0])
        point_data = ps.point_data
        assert 'intensity' in point_data.columns
        assert 'red' in point_data.columns
        assert len(point_data) == e57_scans[0].n_points

    @_needs_e57_file
    def test_visualize_intensity_with_viridis(self, e57_scans):
        """Render scan 0 with intensity mapped through viridis colormap."""
        pytest.importorskip("pyvista", reason="PyVista is required for visualization")
        from subsurface.modules.visualization import to_pyvista_points, pv_plot

        ps = PointSet(e57_scans[0])
        cloud = to_pyvista_points(ps)

        assert 'intensity' in cloud.point_data, "intensity not attached to PolyData"
        assert cloud.n_points == e57_scans[0].n_points

        pv_plot(
            [cloud],
            image_2d=True,
            add_mesh_kwargs={'scalars': 'intensity', 'point_size': 1},
        )

    @_needs_e57_file
    def test_visualize_rgb_colors(self, e57_scans):
        """Render scan 0 with original scanner RGB colors, bypassing colormap."""
        pytest.importorskip("pyvista", reason="PyVista is required for visualization")
        from subsurface.modules.visualization import to_pyvista_points, pv_plot

        ps = PointSet(e57_scans[0])
        cloud = to_pyvista_points(ps)

        assert 'red' in cloud.point_data, "red not attached to PolyData"
        assert 'green' in cloud.point_data
        assert 'blue' in cloud.point_data

        rgb = np.column_stack([
            cloud.point_data["red"],
            cloud.point_data["green"],
            cloud.point_data["blue"],
        ]).astype(np.uint8)
        assert rgb.shape == (cloud.n_points, 3)

        cloud.point_data["RGB"] = rgb

        pv_plot(
            [cloud],
            image_2d=True,
            add_mesh_kwargs={'scalars': 'RGB', 'rgb': True, 'point_size': 1},
        )
