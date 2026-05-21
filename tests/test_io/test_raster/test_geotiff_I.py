import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio

from tests.conftest import RequirementsLevel
from subsurface import StructuredGrid
from subsurface.modules.reader import read_structured_topography
from subsurface.core.utils.utils_core import replace_outliers
from subsurface.modules.visualization import to_pyvista_grid, pv_plot

pytestmark = pytest.mark.read_geospatial


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
    replace_outliers(struct, 'topography', 0.99)
    sg = StructuredGrid(struct)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    pv_plot([s], image_2d=False)