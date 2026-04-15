import dotenv
import os
import pandas as pd
import numpy as np
import pytest
from subsurface.modules.visualization import to_pyvista_points, pv_plot

from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith, read_attributes
import subsurface as ss
from ._aux_func import _plot
from ...conftest import RequirementsLevel

dotenv.load_dotenv()

PLOT = False

data_folder = os.getenv("TERRA_PATH_DEVOPS")

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_MESH"
)


def test_read_survey():
    collar_df: pd.DataFrame = read_collar(
        GenericReaderFilesHelper(
            file_or_buffer=(data_folder + "boreholes/bgr/Stammdaten.CSV"),
            index_col="LOCID",
            usecols=["LOCID", 'RECHTSWERT', 'HOCHWERT', 'ANSATZH', "BOHRUNGSNAME"],
            columns_map={
                    "LOCID"     : "id",  # ? Index name is not mapped
                    "RECHTSWERT": "x",
                    "HOCHWERT"  : "y",
                    "ANSATZH"   : "z"
            },
            encoding="latin-1",
            additional_reader_kwargs={"decimal": ","}
        )
    )
    assert not collar_df.empty
    extent_from_collar_max = collar_df[['x', 'y', 'z']].max()
    extent_from_collar_min = collar_df[['x', 'y', 'z']].min()

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
    assert collars.collar_loc.n_points > 0

    if PLOT and False:
        s = to_pyvista_points(collars.collar_loc)
        pv_plot([s], image_2d=False)

    survey_df: pd.DataFrame = read_survey(
        GenericReaderFilesHelper(
            file_or_buffer=(data_folder + "boreholes/bgr/Schichtdaten.CSV"),
            index_col="LOCID",
            usecols=["LOCID", "TIEFE BIS"],
            columns_map={
                    'TIEFE BIS': 'md',
                    'LOCID'    : 'Name'

            },
            encoding="latin-1",
            additional_reader_kwargs={"decimal": ","}
        )
    )
    assert not survey_df.empty


    print("\nSurvey Dataframe Info:")
    print(survey_df.info())
    print("\nSurvey Dataframe Description:")
    print(survey_df.describe())
    print("\nNumber of unique boreholes:", survey_df.index.nunique())
    print("\nBorehole IDs:", survey_df.index.unique().tolist())

    # Calculate max MD per borehole
    max_md = survey_df.groupby(level=0)['md'].max()
    print("\nMax MD per borehole:")
    print(max_md)

    print("\nCollar Dataframe Info:")
    print(collar_df.info())
    print("\nCollar Coordinates Extent:")
    print("Max:\n", extent_from_collar_max)
    print("Min:\n", extent_from_collar_min)
    
    survey_df = survey_df.drop_duplicates()
    survey: Survey = Survey.from_df(
        survey_df=survey_df,
        # attr_df=attributes,
        number_nodes=10,
        duplicate_attr_depths=True
    )
    assert survey.survey_trajectory.data.n_points > 0


def test_read():
    collar_df: pd.DataFrame = read_collar(
        GenericReaderFilesHelper(
            file_or_buffer=(data_folder + "boreholes/bgr/Stammdaten.CSV"),
            index_col="LOCID",
            usecols=["LOCID", 'RECHTSWERT', 'HOCHWERT', 'ANSATZH', "BOHRUNGSNAME"],
            columns_map={
                    "LOCID"     : "id",  # ? Index name is not mapped
                    "RECHTSWERT": "x",
                    "HOCHWERT"  : "y",
                    "ANSATZH"   : "z"
            },
            encoding="latin-1",
            additional_reader_kwargs={"decimal": ","}
        )
    )
    assert not collar_df.empty
    extent_from_collar_max = collar_df[['x', 'y', 'z']].max()
    extent_from_collar_min = collar_df[['x', 'y', 'z']].min()

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
    assert collars.collar_loc.n_points > 0

    if PLOT and False:
        s = to_pyvista_points(collars.collar_loc)
        pv_plot([s], image_2d=False)

    survey_df: pd.DataFrame = read_survey(
        GenericReaderFilesHelper(
            file_or_buffer=(data_folder + "boreholes/bgr/Schichtdaten.CSV"),
            index_col="LOCID",
            usecols=["LOCID", "TIEFE BIS"],
            columns_map={
                    'TIEFE BIS': 'md',
                    'LOCID'    : 'Name'

            },
            encoding="latin-1",
            additional_reader_kwargs={"decimal": ","}
        )
    )
    assert not survey_df.empty
    
    
    

    attributes: pd.DataFrame = read_attributes(
        GenericReaderFilesHelper(
            file_or_buffer=(data_folder + "boreholes/bgr/Schichtdaten.CSV"),
            index_col="LOCID",
            columns_map={
                    'LOCID'    : 'id',
                    'TIEFE BIS': 'base'
            },
            encoding="latin-1",
            additional_reader_kwargs={"decimal": ","}
        ))
    assert not attributes.empty

    lith: pd.DataFrame = read_lith(
        GenericReaderFilesHelper(
            file_or_buffer=(data_folder + "boreholes/bgr/survey.csv"),
            index_col="LOCID",
            columns_map={
                    'LOCID'    : 'id',
                    'TIEFE BIS': 'base',
                    'STRAT'    : 'component lith'
            },
            encoding="latin-1",
            additional_reader_kwargs={"decimal": ","}
        ))
    assert not lith.empty
    lith['component lith'].unique()

    # sort survey_df by column Name and md 
    # remove duplicates
    survey_df = survey_df.drop_duplicates()
    survey: Survey = Survey.from_df(
        survey_df=survey_df,
        attr_df=lith,
        number_nodes=10,
        duplicate_attr_depths=True
    )

    # survey.update_survey_with_attr(attributes)
    survey.update_survey_with_lith(lith)

    survey.survey_trajectory.data.points_attributes["lith_ids"].unique()
    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )
    assert borehole_set.combined_trajectory.data.n_points > 0
    
    # Coordinate and attribute checks
    vertices = borehole_set.combined_trajectory.data.vertex
    assert not np.allclose(vertices[0], vertices[-1])
    # check that we have some lith ids
    lith_ids = borehole_set.survey.survey_trajectory.data.points_attributes["lith_ids"]
    assert lith_ids.nunique() > 1
    
    # Verify specific points to ensure consistency
    n = vertices.shape[0]
    assert n == 958
    
    def assert_contains_vertex(target, array, atol=1e-1):
        found = np.any(np.all(np.isclose(array, target, atol=atol), axis=1))
        assert found, f"Vertex {target} not found in trajectory"

    assert_contains_vertex([3529790.0, 5685410.0, 278.0], vertices)

    # %%
    # Visualize boreholes with pyvista
    if PLOT:
        trajectory = borehole_set.combined_trajectory
        _plot(
            scalar="lith_ids",
            # scalar="TO NUMBER(SD.STRATEINH)",
            trajectory=trajectory,
            collars=collars,
            lut=20,
            image_2d=True
        )
