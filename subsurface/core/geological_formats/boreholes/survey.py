from dataclasses import dataclass
from typing import Union, Hashable, Optional

import pandas as pd

from ._map_attrs_to_survey import combine_survey_and_attrs
from ._survey_to_unstruct import data_frame_to_unstructured_data
from ...structs.base_structures import UnstructuredData
from ...structs.unstructured_elements import LineSet

NUMBER_NODES = 30
RADIUS = 10


@dataclass
class Survey:
    ids: list[str]
    survey_trajectory: LineSet
    well_id_mapper: dict[str, int] = None  #: This is following the order of the survey csv that can be different that the collars

    @property
    def id_to_well_id(self):
        # Reverse the well_id_mapper dictionary to map IDs to well names
        id_to_well_name_mapper = {v: k for k, v in self.well_id_mapper.items()}
        return id_to_well_name_mapper

    @classmethod
    def from_df(cls, survey_df: 'pd.DataFrame', attr_df: Optional['pd.DataFrame'] = None, number_nodes: Optional[int] = NUMBER_NODES,
                duplicate_attr_depths: bool = False) -> 'Survey':
        """
        Create a Survey object from two DataFrames containing survey and attribute data.

        :param survey_df: DataFrame containing survey data.
        :param attr_df: DataFrame containing attribute data. This is used to make sure the raw data is perfectly aligned.
        :param number_nodes: Optional parameter specifying the number of nodes.
        :return: A Survey object representing the input data.

        """
        trajectories: UnstructuredData = data_frame_to_unstructured_data(
            survey_df=_correct_angles(survey_df),
            attr_df=attr_df,
            number_nodes=number_nodes,
            duplicate_attr_depths=duplicate_attr_depths
        )
        # Grab the unique ids
        unique_ids = trajectories.points_attributes["well_id"].unique()

        return cls(
            ids=unique_ids,
            survey_trajectory=LineSet(data=trajectories, radius=RADIUS),
            well_id_mapper=trajectories.data.attrs["well_id_mapper"]
        )

    def get_well_string_id(self, well_id: int) -> str:
        return self.ids[well_id]

    def get_well_num_id(self, well_string_id: Union[str, Hashable]) -> int:
        return self.well_id_mapper.get(well_string_id, None)

    def update_survey_with_lith(self, lith: pd.DataFrame):
        unstruct: UnstructuredData = combine_survey_and_attrs(lith, self.survey_trajectory, self.well_id_mapper)
        self.survey_trajectory.data = unstruct

    def update_survey_with_attr(self, attrs: pd.DataFrame):
        self.survey_trajectory.data = combine_survey_and_attrs(attrs, self.survey_trajectory, self.well_id_mapper)


def _correct_angles(df: pd.DataFrame) -> pd.DataFrame:
    def correct_inclination(inc: float) -> float:
        if inc < 0:
            inc = inc % 360  # Normalize to 0-360 range first if negative
        if 0 <= inc <= 180:
            # add or subtract a very small number to make sure that 0 or 180 are never possible
            return inc + 1e-10 if inc == 0 else inc - 1e-10
        elif 180 < inc < 360:
            return 360 - inc  # Reflect angles greater than 180 back into the 0-180 range
        else:
            raise ValueError(f'Inclination value {inc} is out of the expected range of 0 to 360 degrees')

    def correct_azimuth(azi: float) -> float:
        return azi % 360  # Normalize azimuth to 0-360 range

    df['inc'] = df['inc'].apply(correct_inclination)
    df['azi'] = df['azi'].apply(correct_azimuth)

    return df
