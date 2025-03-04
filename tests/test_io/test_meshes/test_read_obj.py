import os

import pytest
import dotenv

from subsurface import optional_requirements
from tests.conftest import RequirementsLevel

dotenv.load_dotenv()
path_to_obj = os.getenv("PATH_TO_OBJ")
path_to_mtl = os.getenv("PATH_TO_MTL")

pytestmark = pytest.mark.read_mesh

@pytest.mark.skipif(
    condition=(RequirementsLevel.MESH | RequirementsLevel.PLOT) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH variable to run this test"
)
def test_read_obj():
    pv = optional_requirements.require_pyvista()
    reader = pv.get_reader(path_to_obj)
    mesh = reader.read()
    if plot := False:
        mesh.plot()
    
    