import dotenv
import os
import pandas as pd

from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith
import subsurface as ss
from subsurface.modules.visualization import to_pyvista_line, init_plotter, to_pyvista_points
from test_io.test_lines._aux_func import _plot

dotenv.load_dotenv()

PLOT = True

data_folder = os.getenv("PATH_TO_BGR")


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

    # Convert to UnstructuredData
    unstruc: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=collar_df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )

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
    survey: Survey = Survey.from_df(survey_df)

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

    # Update survey data with lithology information
    survey.update_survey_with_lith(lith)

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    # %%
    # Visualize boreholes with pyvista

    trajectory = borehole_set.combined_trajectory
    scalar = "lith_ids"
    _plot(scalar, trajectory, collars, lut=14)
    return 
    import matplotlib.pyplot as plt

    well_mesh = to_pyvista_line(
        line_set=borehole_set.combined_trajectory,
        active_scalar="lith_ids",
        radius=40
    )

    p = init_plotter()

    # Set colormap for lithologies
    boring_cmap = plt.get_cmap(name="viridis", lut=14)
    p.add_mesh(well_mesh, cmap=boring_cmap)

    collar_mesh = to_pyvista_points(collars.collar_loc)

    p.add_mesh(collar_mesh, render_points_as_spheres=True)
    p.add_point_labels(
        points=collars.collar_loc.points,
        labels=collars.ids,
        point_size=10,
        shape_opacity=0.5,
        font_size=12,
        bold=True
    )

    p.show()
