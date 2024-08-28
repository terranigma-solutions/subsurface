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
from subsurface.modules.visualization import to_pyvista_points, pv_plot, to_pyvista_line, init_plotter

dotenv.load_dotenv()

PLOT = True

data_folder = os.getenv("PATH_TO_ASCII_DRILLHOLES")


def test_read_collar():
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

    if PLOT:
        s = to_pyvista_points(collars.collar_loc)
        pv_plot([s], image_2d=True)


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
        s = to_pyvista_line(
            line_set=trajectory,
            active_scalar="lith_ids",
            radius=40
        )
        clim = [0, 8]  # Threshold lith_ids
        s = s.threshold(clim)

        collar_mesh = to_pyvista_points(collar.collar_loc)

        p = init_plotter()
        import matplotlib.pyplot as plt
        boring_cmap = plt.get_cmap("viridis", 8)
        p.add_mesh(s, clim=clim, cmap=boring_cmap)
        p.add_mesh(collar_mesh, render_points_as_spheres=True)
        p.add_point_labels(
            points=collar.collar_loc.points,
            labels=collar.ids,
            point_size=10,
            shape_opacity=0.5,
            font_size=12,
            bold=True
        )
        p.show()


def test_read_attr_only_depth():
    pass


def test_read_attr():
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

    pass
