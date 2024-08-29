import enum
import numpy as np
import pandas as pd

from .collars import Collars
from .survey import Survey
from ...structs import LineSet
from ...structs.base_structures import UnstructuredData


class MergeOptions(enum.Enum):
    RAISE = enum.auto()
    INTERSECT = enum.auto()


def create_combined_trajectory(collars: Collars, survey: Survey, merge_option: MergeOptions, slice_: slice):
    collar_df = _create_collar_df(collars, slice_, survey.well_id_mapper)
    survey_df_vertex = _create_survey_df(survey)

    if merge_option == MergeOptions.RAISE:
        raise NotImplementedError("RAISE merge option not implemented")
        _validate_matching_ids(collar_df, survey_df_vertex)
    elif merge_option == MergeOptions.INTERSECT:
        return _Intersect.process_intersection(collar_df, survey_df_vertex, survey)
    else:
        raise ValueError("Unsupported merge option")


def _create_collar_df(collars, slice_, well_id_mapper: dict ):
    collar_df = pd.DataFrame(collars.collar_loc.points[slice_], columns=['X', 'Y', 'Z'])
    selected_collars:list = collars.ids[slice_]
    collar_df['well_id'] = pd.Series(selected_collars).map(well_id_mapper)
    return collar_df


def _create_survey_df(survey):
    survey_df_vertex = pd.DataFrame(survey.survey_trajectory.data.vertex, columns=['X', 'Y', 'Z'])
    vertex_attrs = survey.survey_trajectory.data.points_attributes
    id_int_vertex = vertex_attrs['well_id']
    survey_df_vertex['well_id'] = id_int_vertex.map(pd.Series(survey.ids))
    return survey_df_vertex


def _validate_matching_ids(collar_df, survey_df_vertex):
    missing_from_survey = set(collar_df['well_id']) - set(survey_df_vertex['well_id'])
    missing_from_collar = set(survey_df_vertex['well_id']) - set(collar_df['well_id'])
    if missing_from_survey or missing_from_collar:
        raise ValueError(f"Collars and survey ids do not match. Missing in survey: {missing_from_survey}, Missing in collars: {missing_from_collar}")


class _Intersect:
    """This class is just to create a namespace for the intersection method"""
    @staticmethod
    def process_intersection(collar_df: pd.DataFrame, survey_df_vertex: pd.DataFrame, survey: Survey) -> LineSet:
        # make sure well_id type is int in both dataframes
        collar_df['well_id'] = collar_df['well_id'].astype(int)
        survey_df_vertex['well_id'] = survey_df_vertex['well_id'].astype(int)
        
        combined_df_vertex = pd.merge(
            left=survey_df_vertex,
            right=collar_df,
            how='outer',
            indicator=True,
            on='well_id',
            suffixes=('_collar', '_survey')
        )
        combined_df_vertex = combined_df_vertex[combined_df_vertex['_merge'].isin(['left_only', 'both']) ]

        vertex_attrs = survey.survey_trajectory.data.points_attributes
        if len(combined_df_vertex) != len(vertex_attrs):
            raise ValueError("Vertex and vertex attributes have different lengths")
        _Intersect._add_collar_coordinates(combined_df_vertex)

        combined_df_cells = _Intersect._generate_cells(combined_df_vertex, survey)
        cell_attributes = survey.survey_trajectory.data.cell_attributes
        if len(combined_df_cells) != len(cell_attributes):
            raise ValueError("Cells and cell attributes have different lengths")

        return _Intersect._create_line_set(combined_df_vertex, combined_df_cells, survey)

    @staticmethod
    def _add_collar_coordinates(combined_df_vertex: pd.DataFrame):
        combined_df_vertex['X_survey'] += combined_df_vertex['X_collar']
        combined_df_vertex['Y_survey'] += combined_df_vertex['Y_collar']
        combined_df_vertex['Z_survey'] += combined_df_vertex['Z_collar']

    @staticmethod
    def _generate_cells(combined_df_vertex: pd.DataFrame, survey: Survey) -> pd.DataFrame:
        combined_df_cells = []
        previous_index = 0
        for e, well_id in enumerate(survey.ids):
            df_vertex_well = combined_df_vertex[combined_df_vertex['well_id'] == well_id]
            indices = np.arange(len(df_vertex_well)) + previous_index
            previous_index += len(df_vertex_well)
            cells = np.array([indices[:-1], indices[1:]]).T
            df_cells_well = pd.DataFrame(cells, columns=['cell1', 'cell2'])
            df_cells_well['well_id'] = well_id
            df_cells_well['well_id_int'] = e
            combined_df_cells.append(df_cells_well)

        return pd.concat(combined_df_cells, ignore_index=True)

    @staticmethod
    def _create_line_set(combined_df_vertex: pd.DataFrame, combined_df_cells: pd.DataFrame, survey: Survey) -> LineSet:
        vertex_attrs = survey.survey_trajectory.data.points_attributes
        cell_attributes = survey.survey_trajectory.data.cell_attributes

        combined_trajectory_unstruct = UnstructuredData.from_array(
            vertex=combined_df_vertex[['X_survey', 'Y_survey', 'Z_survey']].values,
            cells=combined_df_cells[['cell1', 'cell2']].values,
            vertex_attr=vertex_attrs,
            cells_attr=cell_attributes
        )

        return LineSet(data=combined_trajectory_unstruct, radius=500)
