import dotenv
import pathlib
import pytest
from dotenv import dotenv_values

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

    config = dotenv_values()

    devops_path = pathlib.Path(config.get('TERRA_PATH_DEVOPS'))
    filepath = devops_path.joinpath('meshes\GOCAD\mix\horizons_faults.mx')

    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(filepath)
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

    config = dotenv_values()

    devops_path = pathlib.Path(config.get('TERRA_PATH_DEVOPS'))
    filepath = devops_path.joinpath('meshes\GOCAD\IDN-64\mx_ubc\muon_only.mx')
        # muon_only.mx uses the PVRTX vertex definition but does not actually provide any property values.

    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(str(filepath))
    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)


def test_read_mx_from_file__gen11818__idn64_2():
    from subsurface.modules.reader.mesh.mx_reader import mx_to_unstruct_from_file

    config = dotenv_values()

    devops_path = pathlib.Path(config.get('TERRA_PATH_DEVOPS'))
    filepath = devops_path.joinpath(r"meshes\GOCAD\IDN-64\mx_ubc\U60A_surf.mx")
        # U60A_surf.mx actually provides property values in the last column of PVRTX

    unstruct: subsurface.UnstructuredData = mx_to_unstruct_from_file(str(filepath))
    ts = TriSurf(mesh=unstruct)
    s = sb_viz.to_pyvista_mesh(ts)
    sb_viz.pv_plot([s], image_2d=True)