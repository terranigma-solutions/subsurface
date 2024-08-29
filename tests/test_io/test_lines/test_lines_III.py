import dotenv
import pandas as pd
import pathlib

from subsurface import UnstructuredData
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith
from test_io.test_lines._aux_func import _plot

dotenv.load_dotenv()

PLOT = True

pf = pathlib.Path(__file__).parent.absolute()
data_path = pf.joinpath('../../data/borehole/')


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
    # TODO: df to unstruct
    unstruc: UnstructuredData = UnstructuredData.from_array(
        vertex=collar_df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )
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

    survey: Survey = Survey.from_df(survey_df)

    lith: pd.DataFrame = read_lith(
        GenericReaderFilesHelper(
            file_or_buffer=data_path.joinpath('kim_ready.csv'),
            usecols=['name', 'top', 'base', 'formation'],
            columns_map={'top'      : 'top',
                         'base'     : 'base',
                         'formation': 'component lith',
                         }
        )
    )

    survey.update_survey_with_lith(lith)

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    _plot(
        scalar="lith_ids",
        trajectory=borehole_set.combined_trajectory,
        collars=collars,
        lut=14
    )
