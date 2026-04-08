import dotenv
import os
import pandas as pd
import pytest

from subsurface import UnstructuredData
from subsurface.api.reader.read_wells import read_wells
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.core.structs.unstructured_elements import PointSet
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith, read_attributes
from subsurface.modules.visualization import to_pyvista_points, pv_plot, to_pyvista_line, init_plotter
from ._aux_func import _plot
from ...conftest import RequirementsLevel

dotenv.load_dotenv()

PLOT = True

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_WELL"
)

def test_read_collar():
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_COLLAR"),
        header=0,
        usecols=[0, 1, 2, 4],
        columns_map={
                "hole_id"            : "id",  # ? Index name is not mapped
                "X_GK5_incl_inserted": "x",
                "Y__incl_inserted"   : "y",
                "Z_GK"               : "z"
        }
    )
    df = read_collar(reader)
    assert not df.empty
    assert "x" in df.columns
    assert "y" in df.columns
    assert "z" in df.columns

    # TODO: df to unstruct
    unstruc: UnstructuredData = UnstructuredData.from_array(
        vertex=df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )
    assert unstruc.n_points == len(df)

    points = PointSet(data=unstruc)

    collars = Collars(
        ids=df.index.to_list(),
        collar_loc=points
    )
    assert len(collars.ids) == len(df)
    assert collars.collar_loc.n_points == len(df)

    if PLOT:
        s = to_pyvista_points(collars.collar_loc)
        pv_plot([s], image_2d=True)


def test_read_survey():
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
        columns_map={
                'depth'  : 'md',
                'dip'    : 'dip',
                'azimuth': 'azi'
        },
        encoding="ISO-8859-1"
    )
    df = read_survey(reader)
    assert not df.empty
    assert "md" in df.columns
    assert "inc" in df.columns
    assert "azi" in df.columns

    survey: Survey = Survey.from_df(df, None)
    assert survey.survey_trajectory.data.n_points > 0

    if PLOT and False:
        s = to_pyvista_line(
            line_set=survey.survey_trajectory,
            radius=10,
            active_scalar="well_id"
        )
        pv_plot([s], image_2d=True)

    return survey


def test_add_auxiliary_fields_to_survey():
    # TODO: Update this test to account for the ids mapping of assey or lith
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
        columns_map={
                'depth'  : 'md',
                'dip'    : 'dip',
                'azimuth': 'azi'
        },
        encoding="ISO-8859-1"
    )
    survey: Survey = Survey.from_df(read_survey(reader))
    import xarray as xr
    data_array: UnstructuredData = survey.survey_trajectory.data
    data_set: xr.Dataset = data_array.data
    assert "vertex" in data_set.data_vars
    assert "cells" in data_set.data_vars

    pass


def test_read_assay():
    reader_collar: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_COLLAR"),
        header=0,
        usecols=[0, 1, 2, 4],
        columns_map={
                "hole_id"            : "id",  # ? Index name is not mapped
                "X_GK5_incl_inserted": "x",
                "Y__incl_inserted"   : "y",
                "Z_GK"               : "z"
        }
    )
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
        columns_map={
                'depth'  : 'md',
                'dip'    : 'dip',
                'azimuth': 'azi'
        },
        encoding="ISO-8859-1"
    )

    reader_attr: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_ASSAY"),
        columns_map={
                'hole_id'   : 'id',
                'depth_from': 'top',
                'depth_to'  : 'base'
        },
        additional_reader_kwargs={
                'na_values': [-9999]
        },
        encoding="ISO-8859-1"
    )
    
    borehole_set: BoreholeSet = read_wells(
        collars_reader=reader_collar,
        surveys_reader=reader,
        attrs_reader=reader_attr,
        is_lith_attr=False,
        add_attrs_as_nodes=True
    )
    assert borehole_set.collars.collar_loc.n_points > 0
    assert borehole_set.survey.survey_trajectory.data.n_points > 0
    assert "Cu(%)_GDR" in borehole_set.survey.survey_trajectory.data.points_attributes.columns
    
    _plot(
        scalar="Cu(%)_GDR",
        trajectory=borehole_set.combined_trajectory,
        collars=borehole_set.collars,
        image_2d=True
    )



def test_read_stratigraphy():
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_STRATIGRAPHY"),
        columns_map={
                'hole_id'   : 'id',
                'depth_from': 'top',
                'depth_to'  : 'base',
                'lit_code'  : 'component lith'
        }
    )

    lith: pd.DataFrame = read_lith(reader)
    assert not lith.empty
    survey: Survey = test_read_survey()

    survey.update_survey_with_lith(lith)

    reader_collar: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_COLLAR"),
        header=0,
        usecols=[0, 1, 2, 4],
        columns_map={
                "hole_id"            : "id",  # ? Index name is not mapped
                "X_GK5_incl_inserted": "x",
                "Y__incl_inserted"   : "y",
                "Z_GK"               : "z"
        }
    )
    df_collar = read_collar(reader_collar)
    collar = Collars.from_df(df_collar)

    borehole_set = BoreholeSet(
        collars=collar,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )
    assert borehole_set.collars.collar_loc.n_points > 0
    borehole_set.get_bottom_coords_for_each_lith()
    
    foo = borehole_set._merge_vertex_data_arrays_to_dataframe()
    assert not foo.empty
    assert "lith_ids" in foo.columns
    well_id_mapper: dict[str, int] = borehole_set.survey.id_to_well_id
    # mapp well_id column to well_name
    foo["well_name"] = foo["well_id"].map(well_id_mapper)
    

    if PLOT and False:
        trajectory = borehole_set.combined_trajectory
        s = to_pyvista_line(
            line_set=trajectory,
            active_scalar="lith_ids",
            radius=40
        )
        clim = [0, 20]
        s = s.threshold(clim)

        collar_mesh = to_pyvista_points(collar.collar_loc)

        p = init_plotter()
        p.add_mesh(s, clim=clim, cmap="tab20c")
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


def test_merge_collar_survey():
    reader_collar: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_COLLAR"),
        header=0,
        usecols=[0, 1, 2, 4],
        columns_map={
                "hole_id"            : "id",  # ? Index name is not mapped
                "X_GK5_incl_inserted": "x",
                "Y__incl_inserted"   : "y",
                "Z_GK"               : "z"
        }
    )
    df_collar = read_collar(reader_collar)
    collar = Collars.from_df(df_collar)

    reader_survey: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
        columns_map={
                'depth'  : 'md',
                'dip'    : 'dip',
                'azimuth': 'azi'
        },
        encoding="ISO-8859-1"
        
    )

    survey = Survey.from_df(read_survey(reader_survey))

    borehole_set = BoreholeSet(
        collars=collar,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )
    assert borehole_set.collars.collar_loc.n_points > 0
    assert borehole_set.survey.survey_trajectory.data.n_points > 0

    if PLOT:
        s = to_pyvista_line(line_set=borehole_set.combined_trajectory, radius=50)
        pv_plot([s], image_2d=True)
