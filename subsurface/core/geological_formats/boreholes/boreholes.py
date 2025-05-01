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
    """
    This module provides a class, `BoreholeSet`, that represents a collection of boreholes. It contains methods for accessing coordinate data for each lithology in the boreholes.
    
    Notes: 
        - Collars is defined as 1 UnstructuredData
        - Combined trajectory is defined as 1 UnstructuredData

    Classes:
        - `BoreholeSet`: Represents a collection of boreholes.

    Methods:
        - `__init__`: Initializes a new `BoreholeSet` object with the specified input parameters.
        - `get_top_coords_for_each_lith`: Returns a dictionary of top coordinates for each lithology in the boreholes.
        - `get_bottom_coords_for_each_lith`: Returns a dictionary of bottom coordinates for each lithology in the boreholes.

    Attributes:
        - `collars`: A `Collars` object representing the collar information for the boreholes.
        - `survey`: A `Survey` object representing the survey information for the boreholes.
        - `combined_trajectory`: A `LineSet` object representing the combined trajectory of the boreholes.

    Usage:
        ```
        borehole_set = BoreholeSet(collars, survey, merge_option)
        top_coords = borehole_set.get_top_coords_for_each_lith()
        bottom_coords = borehole_set.get_bottom_coords_for_each_lith()
        ```

    Note: The example usage code provided above is for demonstration purposes only. Please replace `collars`, `survey`, and `merge_option` with the actual input parameters when using the `BoreholeSet` class.

    """
    __slots__ = ['collars', 'survey', 'combined_trajectory']
    collars: Collars
    survey: Survey
    combined_trajectory: LineSet

    def __init__(self, collars: Collars, survey: Survey, merge_option: MergeOptions, slice_=slice(None)):

        new_collars = self._remap_collars_with_survey(collars, survey)

        self.collars = new_collars
        self.survey = survey
        self.combined_trajectory: LineSet = create_combined_trajectory(collars, survey, merge_option, slice_)

    @staticmethod
    def _remap_collars_with_survey(collars, survey):
        import pandas as pd
        # Create a DataFrame from your first list
        df1 = pd.DataFrame({'name': collars.ids, 'x': collars.data.vertex[:, 0], 'y': collars.data.vertex[:, 1], 'z': collars.data.vertex[:, 2]})
        df1 = df1.set_index('name')
        # Reindex to match the second list of names
        df_reindexed = df1.reindex(survey.well_id_mapper.keys())
        new_collars = Collars.from_df(df_reindexed)
        return new_collars

    def to_binary(self, path: str) -> bool:
        # I need to implement the survey to and then name the files accordingly
        bytearray_le_collars: bytes = self.collars.data.to_binary()
        bytearray_le_trajectory: bytes = self.combined_trajectory.data.to_binary()
        
        new_file = open(f"{path}_collars.le", "wb")
        new_file.write(bytearray_le_collars)
        
        new_file = open(f"{path}_trajectory.le", "wb")
        new_file.write(bytearray_le_trajectory)
        return True

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
        groupby = merged_df.groupby('lith_ids')
        if groupby.ngroups == 0:
            raise ValueError("No components found")
        for lith, group in groupby:
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
