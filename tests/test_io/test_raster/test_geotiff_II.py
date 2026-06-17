import os

import matplotlib.pyplot as plt
import pytest
import rasterio
from matplotlib.patches import Rectangle
from rasterio.plot import plotting_extent

from subsurface import StructuredGrid
from subsurface.modules.reader import read_structured_topography
from subsurface.modules.visualization import to_pyvista_grid, pv_plot
from tests.conftest import RequirementsLevel

pytestmark = pytest.mark.read_geospatial


def _raw_rasterio_plot(tif_path: str, title: str, crop_to_extent=None):
    """Plot a GeoTIFF directly via rasterio for comparison."""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        extent = plotting_extent(src)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='terrain', extent=extent)
    if crop_to_extent is not None:
        min_x, min_y, max_x, max_y = crop_to_extent
        crop_rectangle = Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            fill=False,
            edgecolor='red',
            linewidth=2,
            label='crop_to_extent'
        )
        ax.add_patch(crop_rectangle)
        ax.legend(loc='upper right')
    ax.set_title(f'{title} (raw rasterio)', fontsize=12)
    fig.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    return fig


def _build_path(filename):
    devops_path = os.getenv("TERRA_PATH_DEVOPS")
    primary = os.path.join(devops_path, "raster", filename)
    if os.path.exists(primary):
        return primary
    fallback = os.path.expanduser(
        "~/.cache/rclone/vfs/terranigma/DevOps/SubsurfaceTestData/raster/" + filename
    )
    if os.path.exists(fallback):
        return fallback
    return primary  # let it fail with a clear path


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_rasterio_rgb_byte():
    """3-band RGB Byte raster (EPSG:32618, uncompressed, stripped)."""
    tif_path = _build_path("rasterio-rgb-byte.tif")

    fig_raw = _raw_rasterio_plot(tif_path, 'rasterio-rgb-byte')
    fig_raw.show()

    struct = read_structured_topography(tif_path)

    # 3-band read: auto-selects the richest band (should be band 2 or 3)
    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0
    assert struct.data['topography'].shape[1] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_gadas_rgba():
    """4-band RGBA Byte raster (Web Mercator, uncompressed, with alpha)."""
    tif_path = _build_path("geotiff-testdata-gadas.tif")

    fig_raw = _raw_rasterio_plot(tif_path, 'gadas-rgba')
    fig_raw.show()

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_geogtowgs84_palette():
    """Palette raster with GeogTOWGS84 datum transform and PackBits compression."""
    tif_path = _build_path("geotiff-testdata-geogtowgs84.tif")

    fig_raw = _raw_rasterio_plot(tif_path, 'geogtowgs84-palette')
    fig_raw.show()

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_osgeo_gdal_cea():
    """Single-band Byte grayscale with user-defined Cylindrical Equal Area CRS."""
    tif_path = _build_path("osgeo-gdal-cea.tif")

    fig_raw = _raw_rasterio_plot(tif_path, 'osgeo-gdal-cea')
    fig_raw.show()

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_osgeo_usgs_i30dem():
    """Signed 16-bit Int16 DEM (EPSG:26710, PixelIsPoint)."""
    tif_path = _build_path("osgeo-usgs-i30dem.tif")

    fig_raw = _raw_rasterio_plot(tif_path, 'osgeo-usgs-i30dem')
    fig_raw.show()

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_vanderford_cog():
    """COG Float32 elevation (EPSG:3031 Antarctic, Deflate, tiled, overviews)."""
    tif_path = _build_path("geotiff-testdata-vanderford-cog.tif")

    fig_raw = _raw_rasterio_plot(tif_path, 'vanderford-cog')
    fig_raw.show()

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_tiff1():
    """Large Float32 DEM (EPSG:3034 LCC Europe, NoData=nan)."""
    tif_path = _build_path("tiff1.tif")

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0
    assert struct.data['topography'].shape[1] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_tiff1_cropped():
    """Large Float32 DEM (EPSG:3034) read with a crop window."""
    tif_path = _build_path("tiff1.tif")
    # Crop to a small area around the center
    crop_to_extent = [3715000.0, 2665000.0, 3720000.0, 2670000.0]

    struct = read_structured_topography(tif_path, crop_to_extent=crop_to_extent)

    # Cropped result should have smaller dimensions than the full raster (1408x1405)
    assert struct.data['topography'].shape[0] < 1400
    assert struct.data['topography'].shape[1] < 1400
    assert struct.data['topography'].shape[0] > 0
    assert struct.data['topography'].shape[1] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_tiff2_rgba():
    """4-band RGBA Byte (EPSG:3034, LZW compressed, INTERLEAVE=PIXEL)."""
    tif_path = _build_path("tiff2.tiff")

    fig_raw = _raw_rasterio_plot(tif_path, 'tiff2-rgba')
    fig_raw.show()

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_tiff3_large():
    """Large 4-band RGBA Byte (EPSG:3034, LZW, 12699x11797)."""
    tif_path = _build_path("tiff3.tiff")

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0
    assert struct.data['topography'].shape[1] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_tiff3_large_cropped():
    """Large 4-band RGBA (EPSG:3034) with a small crop."""
    tif_path = _build_path("tiff3.tiff")
    crop_to_extent = [3718000.0, 2668000.0, 3719000.0, 2669000.0]

    struct = read_structured_topography(tif_path, crop_to_extent=crop_to_extent)

    # Cropped result should be much smaller than 12699x11797
    assert struct.data['topography'].shape[0] < 1000
    assert struct.data['topography'].shape[1] < 1000
    assert struct.data['topography'].shape[0] > 0
    assert struct.data['topography'].shape[1] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_et_dtm():
    """ET_DTM_HC_2026_FINAL_50m.tif — large Float32 DEM with NoData=nan."""
    tif_path = _build_path("ET_DTM_HC_2026_FINAL_50m.tif")

    struct = read_structured_topography(tif_path)

    assert struct.data['topography'].ndim == 2
    assert struct.data['topography'].shape[0] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_et_dtm_cropped():
    """ET_DTM_HC_2026_FINAL_50m.tif with a crop window."""
    tif_path = _build_path("ET_DTM_HC_2026_FINAL_50m.tif")
    crop_to_extent = [3700000.0, 2650000.0, 3720000.0, 2680000.0]

    struct = read_structured_topography(tif_path, crop_to_extent=crop_to_extent)

    assert struct.data['topography'].shape[0] > 0
    assert struct.data['topography'].shape[1] > 0

    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)
