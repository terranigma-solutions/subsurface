import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from tests.conftest import RequirementsLevel
from subsurface import StructuredGrid
from subsurface.modules.reader import read_structured_topography
from subsurface.modules.reader.topography.topo_core import rasterio_dataset_to_structured_data
from subsurface.core.utils.utils_core import replace_outliers
from subsurface.modules.visualization import to_pyvista_grid, pv_plot

pytestmark = pytest.mark.read_geospatial


def _memory_raster(data, transform, crs='EPSG:3857', nodata=None):
    memory_file = MemoryFile()
    with memory_file.open(
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            transform=transform,
            crs=crs,
            nodata=nodata
    ) as dataset:
        dataset.write(data, 1)
    return memory_file


def test_rasterio_dataset_to_structured_data_uses_pixel_centers_and_masks_nodata():
    data = np.array([[1, -9999, 3], [4, 5, 6]], dtype=np.float32)
    transform = from_origin(100, 220, 10, 20)

    with _memory_raster(data, transform, nodata=-9999) as memory_file:
        with memory_file.open() as dataset:
            struct = rasterio_dataset_to_structured_data(dataset)

    topography = struct.data['topography']
    np.testing.assert_allclose(topography.coords['x'], [105, 115, 125])
    np.testing.assert_allclose(topography.coords['y'], [190, 210])
    np.testing.assert_allclose(
        topography.values,
        np.array([[4, 1], [5, np.nan], [6, 3]], dtype=np.float32)
    )


def test_rasterio_dataset_to_structured_data_projects_geographic_coords_to_metric():
    data = np.ones((2, 2), dtype=np.float32)
    transform = from_origin(-9.2, 38.8, 0.1, 0.1)

    with _memory_raster(data, transform, crs='EPSG:4326') as memory_file:
        with memory_file.open() as dataset:
            struct = rasterio_dataset_to_structured_data(dataset)

    x = struct.data.coords['x'].values
    y = struct.data.coords['y'].values
    assert np.diff(x)[0] > 8_000
    assert np.diff(y)[0] > 10_000


def _raw_rasterio_plot(tif_path: str, title: str):
    """Plot a GeoTIFF directly via rasterio for comparison."""
    with rasterio.open(tif_path) as src:
        data = np.fliplr(src.read(1).T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='terrain', origin='lower')
    ax.set_title(f'{title} (raw rasterio)', fontsize=12)
    fig.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    fig.tight_layout()
    return fig


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_lisbon_elevation_geotiff():
    devops_path = os.getenv("TERRA_PATH_DEVOPS")
    tif_path = os.path.join(devops_path, "raster/geotiff-testdata-lisbon-elevation.tif")

    # Plot directly from rasterio for comparison
    fig_raw = _raw_rasterio_plot(tif_path, 'wind-direction')
    fig_raw.show()
    
    struct = read_structured_topography(tif_path)
    replace_outliers(struct, 'topography', 0.99)
    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=False)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_wind_direction_geotiff():
    """Compare raw rasterio plot vs subsurface pipeline for wind_direction.tif."""
    devops_path = os.getenv("TERRA_PATH_DEVOPS")
    tif_path = os.path.join(devops_path, "raster/geotiff-testdata-wind-direction.tif")

    # Plot directly from rasterio for comparison
    fig_raw = _raw_rasterio_plot(tif_path, 'wind-direction')
    fig_raw.show()

    # Plot via subsurface pipeline
    struct = read_structured_topography(tif_path)
    # replace_outliers(struct, 'topography', 0.99)
    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=False)


@pytest.mark.skipif(
    condition=(RequirementsLevel.READ_GEOSPATIAL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_GEOSPATIAL variable to run this test"
)
def test_read_soricom_geotiff():
    """Import soricomDEM10m.tif and compare raw vs subsurface pipeline plots."""
    devops_path = os.getenv("TERRA_PATH_DEVOPS")
    tif_path = os.path.join(devops_path, "raster/soricomDEM10m.tif")

    # Plot directly from rasterio for comparison
    fig_raw = _raw_rasterio_plot(tif_path, 'soricomDEM10m')
    fig_raw.show()

    # Plot via subsurface pipeline
    struct = read_structured_topography(tif_path)
    # replace_outliers(struct, 'topography', 0.99)
    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=True)