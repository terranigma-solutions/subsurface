import os
import dotenv
import pathlib
import pytest

import subsurface
import subsurface.modules.visualization as sb_viz
from subsurface import TriSurf
from subsurface.modules.reader.mesh._GOCAD_mesh import GOCADMesh
from subsurface.modules.visualization import init_plotter
from tests.conftest import RequirementsLevel


dotenv.load_dotenv()

PLOT = True

pytestmark = pytest.mark.read_mesh


def test_read_gocad_from_file():
    from subsurface.modules.reader.mesh.mx_reader import mx_to_unstruct_from_file

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    filepath = devops_path / 'meshes' / 'GOCAD' / 'mix' / 'horizons_faults.mx'

    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(str(filepath))
    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)


def _meshes_to_pyvista(meshes: list[GOCADMesh]):
    import pyvista as pv
    pyvista_meshes = []
    for mesh in meshes:
        faces = mesh.vectorized_edges

        # Create PyVista mesh
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        pv_mesh.color = mesh.color
        pyvista_meshes.append(pv_mesh)

    return pyvista_meshes


def test_read_mx_from_file__gen11818__idn64():
    from subsurface.modules.reader.mesh.mx_reader import mx_to_unstruct_from_file

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    filepath = devops_path / 'meshes' / 'GOCAD' / 'IDN-64' / 'mx_ubc' / 'muon_only.mx'
        # muon_only.mx uses the PVRTX vertex definition but does not actually provide any property values.

    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(str(filepath))
    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)


def test_read_mx_from_file__gen11818__idn64_2():
    from subsurface.modules.reader.mesh.mx_reader import mx_to_unstruct_from_file

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    filepath = devops_path / 'meshes' / 'GOCAD' / 'IDN-64' / 'mx_ubc' / 'U60A_surf.mx'
        # U60A_surf.mx actually provides property values in the last column of PVRTX

    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(str(filepath))
    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)


def test_read_mx_from_file__gen14201__idn85():
    """
        Regression test for reading a customer-provided GOCAD MX/TSurf variant (GEN-14201 / IDN-85).

        This fixture intentionally represents a minimal, “non-standard but valid in the wild” file layout
        that previously produced an empty mesh (0 vertices) because geometry records began immediately
        after metadata without an explicit `TFACE` section delimiter.

        Fixture characteristics:
          - `.mx` container with a TSurf-style payload
          - Minimal HEADER / coordinate system blocks
          - Geometry starts directly with `VRTX` lines (no `TFACE`)
          - Vertex coordinates use scientific notation (e.g., `7.32121000e+05`)
          - File is heavily subsampled to a single triangle to keep the test fast while still exercising
            vertex and triangle parsing paths.

        Expected outcome:
          - `mx_to_unstruct_from_file` returns an `UnstructuredData` instance without raising, and the
            resulting mesh contains vertices and at least one face.

        Notes:
          - The test relies on `TERRA_PATH_DEVOPS` being set to the DevOps repository root containing the
            fixture file under `meshes/GOCAD/IDN-85/`.
        """

    from subsurface.modules.reader.mesh.mx_reader import mx_to_unstruct_from_file

    devops_path = pathlib.Path(os.getenv('TERRA_PATH_DEVOPS'))
    filepath = devops_path / 'meshes' / 'GOCAD' / 'IDN-85' / '20250417_Topo_ROI_resamp5_first_triangle_only_for_test.mx'

    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(str(filepath))

    assert unstruct.vertex.__len__() == 3
    assert unstruct.cells.__len__() == 1
