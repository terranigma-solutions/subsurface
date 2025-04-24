import os
import pathlib

import numpy as np
import pytest

import subsurface
from subsurface.modules.reader.volume.volume_utils import interpolate_unstructured_data_to_structured_data
from subsurface.core.structs import PointSet, StructuredGrid

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.volume.read_volume import read_volumetric_mesh_coord_file, read_volumetric_mesh_attr_file, \
    read_volumetric_mesh_to_subsurface
from subsurface.modules.visualization import to_pyvista_points, pv_plot, to_pyvista_grid
from tests.conftest import RequirementsLevel

pf = pathlib.Path(__file__).parent.absolute()
data_path = pf.joinpath('../../data/volume/')

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_VOLUME) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH"
)


def test_read_volume_from_numpy():
    path = os.getenv("PATH_TO_VOLUME_NUMPY")
    # read npz
    data = np.load(path + "mesh_10m.npz")
    # array
    vertex = data['arr_0']

    models = [
            "mesh_10m_MacPass_Bayesian_inference_density_model",
            "mesh_10m_MacPass_Bayesian_inference_posterior_variance_model",
            "mesh_10m_MacPass_Joint_Muon_Gravity_constrained_full_model",
            "mesh_10m_MacPass_Joint_Muon_Gravity_Density_unconstrained",
            "mesh_10m_MacPass_Muon_Density_unconstrained"
    ]
    
    for density_model_name in models:
        data = np.load(path + density_model_name + ".npz")
        attr = data['data']
        mask = data['mask']

        # put nans to all the masked values
        attr[mask] = np.nan
        attr[attr == -99_999.] = np.nan

        # Calculate spacing
        vertex_max = vertex.max(axis=0)
        vertex_min = vertex.min(axis=0)

        struct: subsurface.StructuredData = subsurface.StructuredData.from_numpy(
            array=attr,
            data_array_name="Density"
        )

        struct.bounds = {
                "X": (vertex_min[0], vertex_max[0]),
                "Y": (vertex_min[1], vertex_max[1]),
                "Z": (vertex_min[2], vertex_max[2])
        }
        binary = struct.to_binary()
        # Save to file
        with open(path + density_model_name + ".le", "wb") as f:
            f.write(binary)

        sg: subsurface.StructuredGrid = StructuredGrid(struct)

        mesh = to_pyvista_grid(sg)
        pv_plot([mesh], image_2d=True)


def test_volumetric_mesh_to_subsurface():
    ud = read_volumetric_mesh_to_subsurface(
        reader_helper_coord=GenericReaderFilesHelper(
            file_or_buffer=data_path.joinpath('mesh'),
            header=None,
            index_col=False,
            col_names=['elem', '_2', '_3', 'x', 'y', 'z'],
            additional_reader_kwargs={
                    "skiprows"    : 1,
                    "delimiter"   : "\s{2,}",
                    "on_bad_lines": "error",
                    "nrows"       : None,
            }
        ),
        reader_helper_attr=GenericReaderFilesHelper(
            file_or_buffer=data_path.joinpath('out_all00'),
            index_col=False,
            additional_reader_kwargs={"sep": ","}
        )
    )
    ps = PointSet(ud)
    mesh = to_pyvista_points(ps)
    pv_plot([mesh], image_2d=True)

    return ud, mesh


@pytest.mark.liquid_earth
def test_interpolate_ud_to_sd():
    ud, ud_mesh = test_volumetric_mesh_to_subsurface()
    sd: subsurface.StructuredData = interpolate_unstructured_data_to_structured_data(
        ud=ud,
        attr_name="pres",
        resolution=[50, 50, 50]
    )

    sg = StructuredGrid(sd)

    mesh = to_pyvista_grid(sg)
    pv_plot([mesh, ud_mesh], image_2d=True)


def test_read_volumetric_mesh():
    vol_mesh_coord_df = read_volumetric_mesh_coord_file(
        GenericReaderFilesHelper(
            file_or_buffer=data_path.joinpath('mesh'),
            header=None,
            index_col=False,
            col_names=['elem', '_2', '_3', 'x', 'y', 'z'],
            additional_reader_kwargs={
                    "skiprows"    : 1,
                    "delimiter"   : "\s{2,}",
                    "on_bad_lines": "error",
                    "nrows"       : None,
            }
        )
    )

    print(vol_mesh_coord_df)

    vol_mesh_attr_df = read_volumetric_mesh_attr_file(
        GenericReaderFilesHelper(
            file_or_buffer=data_path.joinpath('out_all00'),
            index_col=False,
            additional_reader_kwargs={"sep": ","}
        )
    )

    print(vol_mesh_attr_df)
