import dotenv
import pandas as pd
import pathlib
import numpy as np

import pytest

from subsurface import UnstructuredData
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith
from ._aux_func import _plot
from ...conftest import RequirementsLevel

dotenv.load_dotenv()

PLOT = False

pf = pathlib.Path(__file__).parent.absolute()
data_path = pf.joinpath('../../data/borehole/')

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH"
)


@pytest.mark.liquid_earth
def test_read_kim():
    collar_df: pd.DataFrame = read_collar(
        GenericReaderFilesHelper(
            file_or_buffer=data_path.joinpath('kim_ready.csv'),
            index_col="name",
            usecols=['x', 'y', 'altitude', "name"],
            columns_map={
                    "name"    : "id",  # ? Index name is not mapped
                    "X"       : "x",
                    "Y"       : "y",
                    "altitude": "z"
            }
        )
    )
    assert not collar_df.empty
    assert "x" in collar_df.columns
    assert "y" in collar_df.columns
    assert "z" in collar_df.columns
    
    # TODO: df to unstruct
    unstruc: UnstructuredData = UnstructuredData.from_array(
        vertex=collar_df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )
    assert unstruc.n_points == len(collar_df)
    points = PointSet(data=unstruc)
    collars: Collars = Collars(
        ids=collar_df.index.to_list(),
        collar_loc=points
    )

    survey_df: pd.DataFrame = read_survey(
        GenericReaderFilesHelper(
            file_or_buffer=data_path.joinpath('kim_ready.csv'),
            index_col="name",
            usecols=["name", "md"]
        )
    )
    assert not survey_df.empty
    assert "md" in survey_df.columns
    
    survey: Survey = Survey.from_df(survey_df)
    assert survey.survey_trajectory.data.n_points > 0

    lith_reader = GenericReaderFilesHelper(file_or_buffer=data_path.joinpath('kim_ready.csv'), usecols=['name', 'top', 'base', 'formation'], columns_map={'top': 'top', 'base': 'base', 'formation': 'component lith', })
    lith: pd.DataFrame = read_lith( lith_reader )
    assert not lith.empty

    survey.update_survey_with_lith(lith)

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )
    assert borehole_set.combined_trajectory.data.n_points > 0
    assert "lith_ids" in borehole_set.survey.survey_trajectory.data.points_attributes.columns
    
    # Check that trajectory coordinates change (not all at collar)
    vertices = borehole_set.combined_trajectory.data.vertex
    assert not np.allclose(vertices[0], vertices[-1])
    # Check lithology IDs are present and not all NaN
    lith_ids = borehole_set.survey.survey_trajectory.data.points_attributes["lith_ids"]
    assert lith_ids.notna().any()
    
    # Verify specific points to ensure consistency
    n = vertices.shape[0]
    assert n == 2070
    
    def assert_contains_vertex(target, array, atol=1e-1):
        found = np.any(np.all(np.isclose(array, target, atol=atol), axis=1))
        assert found, f"Vertex {target} not found in trajectory"

    assert_contains_vertex([303412.0, 3913997.0, 108.7132874], vertices)
    assert_contains_vertex([318982.0, 3935253.0, 227.418152], vertices)
    assert_contains_vertex([275307.0, 3947074.0, 74.15509], vertices)
    assert_contains_vertex([318982.0, 3935253.0, -1429.5139], vertices)


    borehole_set.collars.data.to_binary()
    borehole_set.combined_trajectory.data.to_binary()

    if PLOT:
        _plot(
            scalar="lith_ids",
            trajectory=borehole_set.combined_trajectory,
            collars=collars,
            lut=14,
            image_2d=False
        )
