import dotenv
import os
import pandas as pd
import numpy as np
import pytest

from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith
import subsurface as ss
from ._aux_func import _plot
from ...conftest import RequirementsLevel

dotenv.load_dotenv()

PLOT = True

data_folder = os.getenv("PATH_TO_BGR")

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH"
)

def test_read():
    raw_borehole_data_csv = data_folder + "boreholes_test.csv"
    collar_df: pd.DataFrame = read_collar(
        GenericReaderFilesHelper(
            file_or_buffer=raw_borehole_data_csv,
            index_col="Name",
            usecols=['X', 'Y', 'Topo', "Name"],
            columns_map={
                    "Name": "id",  # ? Index name is not mapped
                    "X"   : "x",
                    "Y"   : "y",
                    "Topo": "z"
            }

        )
    )
    assert not collar_df.empty
    assert "x" in collar_df.columns
    assert "y" in collar_df.columns
    assert "z" in collar_df.columns

    # Convert to UnstructuredData
    unstruc: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=collar_df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )
    assert unstruc.n_points == len(collar_df)

    points = ss.PointSet(data=unstruc)
    collars: Collars = Collars(
        ids=collar_df.index.to_list(),
        collar_loc=points
    )

    survey_df: pd.DataFrame = read_survey(
        GenericReaderFilesHelper(
            file_or_buffer=raw_borehole_data_csv,
            usecols=["Name", "Depth"],
            columns_map={
                    'Depth': 'md',
            },
        )
    )

    # sort survey_df by column Name and md 
    survey_df = survey_df.sort_values(by=["Name", "md"])

    # remove duplicates
    survey_df = survey_df.drop_duplicates()
    assert not survey_df.empty
    assert "md" in survey_df.columns

    survey: Survey = Survey.from_df(survey_df)
    assert survey.survey_trajectory.data.n_points > 0

    lith = read_lith(
        GenericReaderFilesHelper(
            file_or_buffer=raw_borehole_data_csv,
            index_col="Name",
            usecols=['Name', 'Depth', 'Layer', 'Topo'],
            columns_map={
                    'Depth': 'base',
                    'Layer': 'component lith',
            }
        )
    )
    assert not lith.empty

    # Update survey data with lithology information
    survey.update_survey_with_lith(lith)

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )
    assert borehole_set.combined_trajectory.data.n_points > 0
    assert "lith_ids" in borehole_set.survey.survey_trajectory.data.points_attributes.columns
    
    # Coordinate and attribute checks
    vertices = borehole_set.combined_trajectory.data.vertex
    assert not np.allclose(vertices[0], vertices[-1])
    # check that we have some lith ids
    lith_ids = borehole_set.survey.survey_trajectory.data.points_attributes["lith_ids"]
    assert lith_ids.nunique() > 1
    
    # Verify specific points to ensure consistency
    n = vertices.shape[0]
    assert n == 120
    # Index 0
    np.testing.assert_allclose(vertices[0], [5450950.0, 5708499.9, 115.0])
    assert lith_ids.iloc[0] == 0.0
    # Index n // 2 (60)
    np.testing.assert_allclose(vertices[60], [5451361.3, 5709288.2, 110.9])
    assert lith_ids.iloc[60] == 2.0
    # Index n - 1 (119)
    np.testing.assert_allclose(vertices[119], [5450661.3, 5709527.8, 79.0])
    assert lith_ids.iloc[119] == 6.0

    # %%
    # Visualize boreholes with pyvista

    trajectory = borehole_set.combined_trajectory
    scalar = "lith_ids"
    _plot(scalar, trajectory, collars, lut=14, image_2d=True)
