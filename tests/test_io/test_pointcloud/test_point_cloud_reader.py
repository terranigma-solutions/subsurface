import os

import pytest
import numpy as np
import pandas as pd

from tests.conftest import RequirementsLevel, check_requirements
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
from subsurface.core.structs.base_structures import UnstructuredData
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.api import read_point_cloud_to_unstruct

pytestmark = [pytest.mark.read_mesh, pytest.mark.skipif(
    condition=(RequirementsLevel.MESH) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set REQUIREMENT_LEVEL=MESH or REQUIREMENT_LEVEL=ALL"
)]


def _build_ply_path(filename):
    devops_path = os.getenv("TERRA_PATH_DEVOPS")
    if devops_path:
        primary = os.path.join(devops_path, "point cloud", filename)
        if os.path.exists(primary):
            return primary
    fallback = os.path.expanduser(
        "~/.cache/rclone/vfs/terranigma/DevOps/SubsurfaceTestData/point cloud/" + filename
    )
    if os.path.exists(fallback):
        return fallback
    direct = os.path.expanduser(
        "/home/leguark/Data/OneDrive/Terranigma/DevOps/SubsurfaceTestData/point cloud/" + filename
    )
    if os.path.exists(direct):
        return direct
    return None


_needs_zinnwald = pytest.mark.skipif(
    _build_ply_path("zinnwald.ply") is None,
    reason="zinnwald.ply not found. Set TERRA_PATH_DEVOPS or ensure OneDrive is mounted.",
)


_needs_pred_mineralogy = pytest.mark.skipif(
    _build_ply_path("zinnwaldHSIMaps/Z1_PredMineralogy.ply") is None,
    reason="Z1_PredMineralogy.ply not found. Set TERRA_PATH_DEVOPS or ensure OneDrive is mounted.",
)

_WRITE_BINARIES = os.getenv("WRITE_POINT_CLOUD_BINARIES", "").strip() != ""

_needs_binary_write = pytest.mark.skipif(
    not _WRITE_BINARIES,
    reason="Set WRITE_POINT_CLOUD_BINARIES=1 to enable binary export tests",
)


def _build_binary_dir():
    devops_path = os.getenv("TERRA_PATH_DEVOPS")
    if devops_path:
        return os.path.join(devops_path, "point cloud", "binary")
    return os.path.expanduser(
        "/home/leguark/Data/OneDrive/Terranigma/DevOps/SubsurfaceTestData/point cloud/binary"
    )


def _write_and_verify_roundtrip(ud, le_path, label):
    binary = ud.to_binary()
    os.makedirs(os.path.dirname(le_path), exist_ok=True)
    with open(le_path, "wb") as f:
        f.write(binary)

    assert os.path.getsize(le_path) > 0, f"{label}: binary file is empty"

    rt = UnstructuredData.from_binary_le(le_path)

    assert rt.vertex.shape == ud.vertex.shape, f"{label}: vertex shape mismatch"
    assert rt.cells.shape == ud.cells.shape, f"{label}: cells shape mismatch"
    assert np.allclose(rt.vertex, ud.vertex, atol=1e-5), f"{label}: vertex values differ"

    pa_orig = ud.points_attributes
    pa_rt = rt.points_attributes
    assert list(pa_rt.columns) == list(pa_orig.columns), f"{label}: attribute column mismatch"

    for col in pa_orig.columns:
        orig_vals = pa_orig[col].values
        rt_vals = pa_rt[col].values
        assert np.allclose(rt_vals, orig_vals, atol=1e-5, equal_nan=True), f"{label}: column {col} differs"

    extent = ud.extent.tolist()
    file_mb = os.path.getsize(le_path) / (1024 * 1024)
    print(f"\n{label}:")
    print(f"  points: {ud.n_points}")
    print(f"  extent [xmin xmax ymin ymax zmin zmax]: {extent}")
    print(f"  binary size: {file_mb:.1f} MB")
    print(f"  output: {le_path}")


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


@pytest.fixture(scope="module")
def zinnwald_point_cloud():
    path = _build_ply_path("zinnwald.ply")
    if path is None:
        pytest.skip("zinnwald.ply not found")
    reader_args = GenericReaderFilesHelper(file_or_buffer=path)
    return read_point_cloud_to_unstruct(reader_args)


@pytest.fixture(scope="module")
def pred_mineralogy_point_cloud():
    path = _build_ply_path("zinnwaldHSIMaps/Z1_PredMineralogy.ply")
    if path is None:
        pytest.skip("Z1_PredMineralogy.ply not found")
    reader_args = GenericReaderFilesHelper(file_or_buffer=path)
    return read_point_cloud_to_unstruct(reader_args)


@pytest.mark.skipif(
    condition=check_requirements(RequirementsLevel.MESH | RequirementsLevel.PLOT),
    reason="Need to set REQUIREMENT_LEVEL=MESH or REQUIREMENT_LEVEL=ALL"
)
class TestReadZinnwaldPly:
    """
    Integration test against the real zinnwald.ply dataset.

    4,145,450 points from a hyperspectral scan of a mine face.
    Each point carries XYZ, scanner RGB (uchar), and normals (nx, ny, nz).

    Requires plyfile (pip install subsurface-terra[pointcloud]) and pyvista.
    The dataset must be available at TERRA_PATH_DEVOPS or the OneDrive path.
    """

    @_needs_zinnwald
    def test_returns_unstructured_data(self, zinnwald_point_cloud):
        """Public API returns a single UnstructuredData with ~4.1M points."""
        assert isinstance(zinnwald_point_cloud, UnstructuredData)
        assert zinnwald_point_cloud.n_points == 4_145_450
        assert zinnwald_point_cloud.cells.shape[1] == 1

    @_needs_zinnwald
    def test_has_rgb_and_normals(self, zinnwald_point_cloud):
        """Point attributes include scanner RGB and normals."""
        pa = zinnwald_point_cloud.points_attributes
        for col in ['red', 'green', 'blue', 'nx', 'ny', 'nz']:
            assert col in pa.columns, f"Missing column {col}"
        assert pa['red'].min() >= 0
        assert pa['red'].max() <= 255

    @_needs_zinnwald
    def test_source_format_metadata(self, zinnwald_point_cloud):
        assert zinnwald_point_cloud.data.attrs['source_format'] == 'ply'

    @_needs_zinnwald
    def test_pointset_wraps_cloud(self, zinnwald_point_cloud):
        ps = PointSet(zinnwald_point_cloud)
        assert ps.n_points == zinnwald_point_cloud.n_points

    @_needs_zinnwald
    def test_pointset_exposes_attributes(self, zinnwald_point_cloud):
        ps = PointSet(zinnwald_point_cloud)
        point_data = ps.point_data
        assert 'red' in point_data.columns
        assert 'nx' in point_data.columns
        assert len(point_data) == zinnwald_point_cloud.n_points

    @_needs_zinnwald
    def test_visualize_rgb_colors(self, zinnwald_point_cloud):
        """Render with original scanner RGB colors, bypassing colormap."""
        pytest.importorskip("pyvista", reason="PyVista is required for visualization")
        from subsurface.modules.visualization import to_pyvista_points, pv_plot

        ps = PointSet(zinnwald_point_cloud)
        cloud = to_pyvista_points(ps)

        assert 'red' in cloud.point_data
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

    @_needs_zinnwald
    @_needs_binary_write
    def test_export_binary_and_roundtrip(self, zinnwald_point_cloud):
        le_path = os.path.join(_build_binary_dir(), "zinnwald.le")
        _write_and_verify_roundtrip(zinnwald_point_cloud, le_path, "zinnwald")


@pytest.mark.skipif(
    condition=check_requirements(RequirementsLevel.MESH | RequirementsLevel.PLOT),
    reason="Need to set REQUIREMENT_LEVEL=MESH or REQUIREMENT_LEVEL=ALL"
)
class TestReadZinnwaldPredMineralogy:
    """
    Integration test against Z1_PredMineralogy.ply.

    4,314,311 points from a classified hyperspectral mine-face scan.
    Each point carries XYZ, scanner RGB (uchar), normals (nx, ny, nz),
    and five per-point mineral probability scalars:
      scalar_QuartzFeldspar, scalar_Zinnwaldite, scalar_WhiteMica,
      scalar_Kaolinite, scalar_Topaz.

    Requires plyfile (pip install subsurface-terra[pointcloud]) and pyvista.
    The dataset must be available at TERRA_PATH_DEVOPS or the OneDrive path.
    """

    @_needs_pred_mineralogy
    def test_returns_unstructured_data(self, pred_mineralogy_point_cloud):
        assert isinstance(pred_mineralogy_point_cloud, UnstructuredData)
        assert pred_mineralogy_point_cloud.n_points == 4_314_311
        assert pred_mineralogy_point_cloud.cells.shape[1] == 1

    @_needs_pred_mineralogy
    def test_has_all_expected_attributes(self, pred_mineralogy_point_cloud):
        pa = pred_mineralogy_point_cloud.points_attributes
        expected = [
            'red', 'green', 'blue',
            'nx', 'ny', 'nz',
            'scalar_QuartzFeldspar', 'scalar_Zinnwaldite',
            'scalar_WhiteMica', 'scalar_Kaolinite', 'scalar_Topaz',
        ]
        for col in expected:
            assert col in pa.columns, f"Missing column {col}"

    @_needs_pred_mineralogy
    def test_mineral_fields_have_variation(self, pred_mineralogy_point_cloud):
        pa = pred_mineralogy_point_cloud.points_attributes
        mineral_fields = [
            'scalar_QuartzFeldspar', 'scalar_Zinnwaldite',
            'scalar_WhiteMica', 'scalar_Kaolinite', 'scalar_Topaz',
        ]
        for field in mineral_fields:
            vals = pa[field]
            assert vals.count() > 0, f"{field} has no non-NaN values"
            # NaN sentinel values are expected for unclassified points
            finite = vals[~np.isnan(vals)]
            assert len(finite) > 0, f"{field} has only NaN values"
            assert finite.min() < finite.max(), f"{field} is constant among finite values"

    @_needs_pred_mineralogy
    def test_source_format_metadata(self, pred_mineralogy_point_cloud):
        assert pred_mineralogy_point_cloud.data.attrs['source_format'] == 'ply'

    @_needs_pred_mineralogy
    def test_pointset_wraps_cloud(self, pred_mineralogy_point_cloud):
        ps = PointSet(pred_mineralogy_point_cloud)
        assert ps.n_points == pred_mineralogy_point_cloud.n_points

    @_needs_pred_mineralogy
    def test_visualize_rgb_colors(self, pred_mineralogy_point_cloud):
        """Render with original scanner RGB colors."""
        pytest.importorskip("pyvista", reason="PyVista is required for visualization")
        from subsurface.modules.visualization import to_pyvista_points, pv_plot

        ps = PointSet(pred_mineralogy_point_cloud)
        cloud = to_pyvista_points(ps)

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

    @_needs_pred_mineralogy
    def test_visualize_zinnwaldite_scalar(self, pred_mineralogy_point_cloud):
        """Render scalar_Zinnwaldite through viridis colormap."""
        pytest.importorskip("pyvista", reason="PyVista is required for visualization")
        from subsurface.modules.visualization import to_pyvista_points, pv_plot

        ps = PointSet(pred_mineralogy_point_cloud)
        cloud = to_pyvista_points(ps)

        assert 'scalar_Zinnwaldite' in cloud.point_data
        vals = cloud.point_data['scalar_Zinnwaldite']
        finite = vals[np.isfinite(vals)]
        assert len(finite) > 0
        assert finite.min() >= 0
        assert finite.max() <= 1

        pv_plot(
            [cloud],
            image_2d=False,
            add_mesh_kwargs={'scalars': 'scalar_Zinnwaldite', 'point_size': 1},
        )

    @_needs_pred_mineralogy
    @_needs_binary_write
    def test_export_binary_and_roundtrip(self, pred_mineralogy_point_cloud):
        le_path = os.path.join(_build_binary_dir(), "zinnwaldHSIMaps", "Z1_PredMineralogy.le")
        _write_and_verify_roundtrip(pred_mineralogy_point_cloud, le_path, "Z1_PredMineralogy")
