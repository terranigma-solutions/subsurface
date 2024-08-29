import dotenv
import os
import pandas as pd

from subsurface import UnstructuredData
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith, read_attributes
from subsurface.modules.visualization import to_pyvista_points, pv_plot, to_pyvista_line
from test_io.test_lines._aux_func import _plot

dotenv.load_dotenv()

PLOT = True

data_folder = os.getenv("PATH_TO_ASCII_DRILLHOLES")


def test_read_collar():
    collars = _read_collars()

    if PLOT:
        s = to_pyvista_points(collars.collar_loc)
        pv_plot([s], image_2d=True)


def _read_collars() -> Collars:
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + "collars.csv",
        header=0,
        usecols=[0, 1, 2, 3],
        columns_map={
                "HOLE_ID": "id",  # ? Index name is not mapped
                "X"      : "x",
                "Y"      : "y",
                "Z"      : "z"
        }
    )
    df = read_collar(reader)
    # TODO: df to unstruct
    unstruc: UnstructuredData = UnstructuredData.from_array(
        vertex=df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )
    points = PointSet(data=unstruc)
    collars = Collars(
        ids=df.index.to_list(),
        collar_loc=points
    )
    return collars


def test_read_survey():
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + "survey.csv",
        columns_map={
                'Distance': 'md',
                'Dip'     : 'dip',
                'Azimuth' : 'azi'
        },
    )
    df = read_survey(reader)

    survey: Survey = Survey.from_df(df)

    if PLOT and False:
        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="well_id"
        )
        pv_plot([s], image_2d=True)

    return survey


def test_read_stratigraphy():
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + "geology.csv",
        columns_map={
                'HOLE-ID': 'id',
                'FROM'   : 'top',
                'TO'     : 'base',
                'GEOLOGY': 'component lith'
        }
    )

    lith: pd.DataFrame = read_lith(reader)
    survey: Survey = test_read_survey()

    survey.update_survey_with_lith(lith)

    reader_collar: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + "collars.csv",
        header=0,
        usecols=[0, 1, 2, 3],
        columns_map={
                "HOLE_ID": "id",  # ? Index name is not mapped
                "X"      : "x",
                "Y"      : "y",
                "Z"      : "z"
        }
    )
    df_collar = read_collar(reader_collar)
    collar = Collars.from_df(df_collar)

    borehole_set = BoreholeSet(
        collars=collar,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )
    borehole_set.get_bottom_coords_for_each_lith()

    foo = borehole_set._merge_vertex_data_arrays_to_dataframe()
    well_id_mapper: dict[str, int] = borehole_set.survey.id_to_well_id
    # mapp well_id column to well_name
    foo["well_name"] = foo["well_id"].map(well_id_mapper)

    if PLOT and True:
        trajectory = borehole_set.combined_trajectory
        scalar = "lith_ids"
        _plot(scalar, trajectory, collar)


def test_read_attr():
    survey = _read_geochem_into_survey()

    if PLOT and True:
        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="Ag"
        )
        pv_plot([s], image_2d=False)

        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="MnO"
        )
        pv_plot([s], image_2d=False)

    pass


def test_read_attr_into_borehole():
    collars = _read_collars()
    survey = _read_geochem_into_survey()

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    _plot(
        scalar="MnO",
        trajectory=borehole_set.combined_trajectory,
        collars=collars
    )


def _read_geochem_into_survey() -> Survey:
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + "geochem.csv",
        columns_map={
                'HoleId': 'id',
                'from'  : 'top',
                'to'    : 'base',
        }
    )
    attributes: pd.DataFrame = read_attributes(reader)
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + "survey.csv",
        columns_map={
                'Distance': 'md',
                'Dip'     : 'dip',
                'Azimuth' : 'azi'
        },
    )
    df = read_survey(reader)
    survey: Survey = Survey.from_df(df)
    survey.update_survey_with_attr(attributes)
    return survey
