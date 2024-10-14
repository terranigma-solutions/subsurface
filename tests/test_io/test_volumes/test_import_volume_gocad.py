import os

import dotenv
import pytest

from tests.conftest import RequirementsLevel

dotenv.load_dotenv()

PLOT = True

pytestmark = pytest.mark.skipif(
    condition=RequirementsLevel.PLOT not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH"
)


# Skip if TERRA_PATH_DEVOPS not in .env due to lack of input data
@pytest.mark.skipif(os.getenv("TERRA_PATH_DEVOPS") is None, reason="Need to set the TERRA_PATH_DEVOPS")
def test_read_gocad():
    path = os.getenv("PATH_TO_MX")
