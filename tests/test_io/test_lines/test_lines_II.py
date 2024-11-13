import dotenv
import os
import pandas as pd
import pytest
import numpy as np
from subsurface import UnstructuredData
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith, read_attributes
from subsurface.modules.visualization import to_pyvista_points, pv_plot, to_pyvista_line
from tests.test_io.test_lines._aux_func import _plot

dotenv.load_dotenv()

PLOT = True

data_folder = os.getenv("PATH_TO_ASCII_DRILLHOLES")


@pytest.mark.liquid_earth
def test_read_attr_into_borehole():
    collars: Collars = _read_collars()
    survey: Survey = _read_segment_attr_into_survey("geochem.csv", )

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    if True:
        borehole_set.to_binary("ascii_wells")

    # Assert shape is 17378, 3
    np.testing.assert_array_equal(borehole_set.combined_trajectory.data.vertex.shape, (17378, 3))
    np.testing.assert_array_equal(borehole_set.collars.data.vertex.shape, (263, 3))

    _plot(
        scalar="MnO",
        trajectory=borehole_set.combined_trajectory,
        collars=borehole_set.collars,
        image_2d=False
    )


@pytest.mark.liquid_earth
def test_read_geophys_attr():
    collars = _read_collars()
    survey = _read_point_attr_into_survey("geophysics.csv", )

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    if True:
        borehole_set.to_binary("ascii_wells_geophysics")

    _4Q83 = borehole_set.combined_trajectory.data.vertex[13487:48000, :]
    if PLOT and True:
        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="Gamma_TC"
        )
        pv_plot([s], image_2d=False)

        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="eU"
        )
        pv_plot([s], image_2d=False)

    _plot(
        scalar="Gamma_TC",
        trajectory=borehole_set.combined_trajectory,
        collars=collars,
        image_2d=False
    )


@pytest.mark.liquid_earth
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
    survey_reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + "survey.csv",
        columns_map={
                'Distance': 'md',
                'Dip'     : 'dip',
                'Azimuth' : 'azi'
        },
    )
    survey: Survey = Survey.from_df(
        survey_df=read_survey(survey_reader),
        attr_df=lith,
        number_nodes=10,
        duplicate_attr_depths=True
    )
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

    # ? Not sure what was this for
    foo = borehole_set._merge_vertex_data_arrays_to_dataframe()
    well_id_mapper: dict[str, int] = borehole_set.survey.id_to_well_id
    foo["well_name"] = foo["well_id"].map(well_id_mapper)

    if PLOT and True:
        trajectory = borehole_set.combined_trajectory
        scalar = "lith_ids"
        _plot(scalar, trajectory, collar, lut=8, image_2d=False)


def test_read_collar():
    collars = _read_collars()

    point_cloud = collars.data.vertex
    # Find extent
    min_x, max_x = point_cloud[:, 0].min(), point_cloud[:, 0].max()
    min_y, max_y = point_cloud[:, 1].min(), point_cloud[:, 1].max()
    min_z, max_z = point_cloud[:, 2].min(), point_cloud[:, 2].max()

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


def test_read_geochem_attr():
    survey = _read_segment_attr_into_survey("geochem.csv", )

    if PLOT and True:
        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="Ag"
        )
        pv_plot([s], image_2d=True)

        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="MnO"
        )
        pv_plot([s], image_2d=True)


def _read_point_attr_into_survey(attr_file) -> Survey:
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + attr_file,
        columns_map={
                'HoleId'  : 'id',
                'Distance': 'base'
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
    survey: Survey = Survey.from_df(
        survey_df=read_survey(reader),
        attr_df=attributes,
        number_nodes=10,
        duplicate_attr_depths=True
    )

    survey.update_survey_with_attr(attributes)
    return survey


def _read_segment_attr_into_survey(attr_file) -> Survey:
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=data_folder + attr_file,
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
    survey: Survey = Survey.from_df(
        survey_df=read_survey(reader),
        attr_df=attributes,
        number_nodes=10,
        duplicate_attr_depths=True
    )

    survey.update_survey_with_attr(attributes)
    return survey
