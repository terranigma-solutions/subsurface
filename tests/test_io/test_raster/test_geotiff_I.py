import os

import pytest

from tests.conftest import RequirementsLevel
from subsurface import StructuredGrid
from subsurface.modules.reader import read_structured_topography
from subsurface.core.utils.utils_core import replace_outliers
from subsurface.modules.visualization import to_pyvista_grid, pv_plot

pytestmark = pytest.mark.read_geospatial


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