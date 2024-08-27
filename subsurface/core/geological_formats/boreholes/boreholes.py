import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Hashable

from ._combine_trajectories import create_combined_trajectory, MergeOptions
from .collars import Collars
from .survey import Survey
from ...structs import LineSet


@dataclass
class BoreholeSet:
    __slots__ = ['collars', 'survey', 'combined_trajectory']
    collars: Collars
    survey: Survey
    combined_trajectory: LineSet

    def __init__(self, collars: Collars, survey: Survey, merge_option: MergeOptions, slice_=slice(None)):
        self.collars = collars
        self.survey = survey
        self.combined_trajectory: LineSet = create_combined_trajectory(collars, survey, merge_option, slice_)


    def get_top_coords_for_each_lith(self) -> dict[Hashable, np.ndarray]:
        merged_df = self._merge_vertex_data_arrays_to_dataframe()
        component_lith_arrays = {}
        for lith, group in merged_df.groupby('lith_ids'):
            lith = int(lith)
            first_vertices = group.groupby('well_id').first().reset_index()
            array = first_vertices[['X', 'Y', 'Z']].values
            component_lith_arrays[lith] = array

        return component_lith_arrays

    def get_bottom_coords_for_each_lith(self) -> dict[Hashable, np.ndarray]:
        merged_df = self._merge_vertex_data_arrays_to_dataframe()
        component_lith_arrays = {}
        for lith, group in merged_df.groupby('lith_ids'):
            lith = int(lith)
            first_vertices = group.groupby('well_id').last().reset_index()
            array = first_vertices[['X', 'Y', 'Z']].values
            component_lith_arrays[lith] = array

        return component_lith_arrays

    def _merge_vertex_data_arrays_to_dataframe(self):
        ds = self.combined_trajectory.data.data
        # Convert vertex attributes to a DataFrame for easier manipulation
        vertex_attrs_df = ds['vertex_attrs'].to_dataframe().reset_index()
        vertex_attrs_df = vertex_attrs_df.pivot(index='points', columns='vertex_attr', values='vertex_attrs').reset_index()
        # Convert vertex coordinates to a DataFrame
        vertex_df = ds['vertex'].to_dataframe().reset_index()
        vertex_df = vertex_df.pivot(index='points', columns='XYZ', values='vertex').reset_index()
        # Merge the attributes with the vertex coordinates
        merged_df = pd.merge(vertex_df, vertex_attrs_df, on='points')
        # Create a dictionary to hold the numpy arrays for each component lith
        return merged_df
