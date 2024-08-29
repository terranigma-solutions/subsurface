import warnings

from typing import Union, Hashable

import pandas as pd
from dataclasses import dataclass
import numpy as np
import xarray as xr

from subsurface import optional_requirements
from ...structs.unstructured_elements import LineSet
from ...structs.base_structures import UnstructuredData

STEP = 30
RADIUS = 10


@dataclass
class Survey:
    ids: list[str]
    survey_trajectory: LineSet
    well_id_mapper: dict[str, int] = None

    @property
    def id_to_well_id(self):
        # Reverse the well_id_mapper dictionary to map IDs to well names
        id_to_well_name_mapper = {v: k for k, v in self.well_id_mapper.items()}
        return id_to_well_name_mapper

    @classmethod
    def from_df(cls, df: 'pd.DataFrame'):
        trajectories: UnstructuredData = _data_frame_to_unstructured_data(
            df=_correct_angles(df)
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
        unstruct: UnstructuredData = _combine_survey_and_attrs(lith, self)
        self.survey_trajectory.data = unstruct

    def update_survey_with_attr(self, attrs: pd.DataFrame):
        self.survey_trajectory.data = _combine_survey_and_attrs(attrs, self)


def _combine_survey_and_attr(lith: pd.DataFrame, survey: Survey) -> UnstructuredData:
    pass


def _combine_survey_and_attrs(attrs: pd.DataFrame, survey: Survey) -> UnstructuredData:
    # Import moved to top for clarity and possibly avoiding repeated imports if called multiple times
    from ...structs.base_structures._unstructured_data_constructor import raw_attributes_to_dict_data_arrays

    # Accessing trajectory data more succinctly
    trajectory: xr.DataArray = survey.survey_trajectory.data.data["vertex_attrs"]
    # Ensure all columns in lith exist in new_attrs, if not, add them as NaN
    new_attrs = _map_attrs_to_measured_depths(attrs, survey)

    if 'component lith' in new_attrs.columns:
        # Factorize lith components directly in-place
        new_attrs['lith_ids'], _ = pd.factorize(new_attrs['component lith'], use_na_sentinel=True)

    # Construct the final xarray dict without intermediate variable
    points_attributes_xarray_dict = raw_attributes_to_dict_data_arrays(
        default_attributes_name="vertex_attrs",
        n_items=trajectory.shape[0], # TODO: Can I look this on new_attrs to remove line 11?
        dims=["points", "vertex_attr"],
        raw_attributes=new_attrs
    )

    # Inline construction of UnstructuredData
    return UnstructuredData.from_data_arrays_dict(
        xarray_dict={
                "vertex"      : survey.survey_trajectory.data.data["vertex"],
                "cells"       : survey.survey_trajectory.data.data["cells"],
                "vertex_attrs": points_attributes_xarray_dict["vertex_attrs"],
                "cell_attrs"  : survey.survey_trajectory.data.data["cell_attrs"]
        },
        xarray_attributes=survey.survey_trajectory.data.data.attrs,
        default_cells_attributes_name=survey.survey_trajectory.data.cells_attr_name,
        default_points_attributes_name=survey.survey_trajectory.data.vertex_attr_name
    )


def _map_attrs_to_measured_depths(attrs: pd.DataFrame, survey: Survey) -> pd.DataFrame:
    trajectory: xr.DataArray = survey.survey_trajectory.data.data["vertex_attrs"]
    trajectory_well_id: xr.DataArray = trajectory.sel({'vertex_attr': 'well_id'})
    measured_depths: np.ndarray = trajectory.sel({'vertex_attr': 'measured_depths'}).values.astype(np.float_)

    # Start with a copy of the existing attributes DataFrame
    new_attrs = survey.survey_trajectory.data.points_attributes.copy()

    # Add missing columns from attrs, preserving their dtypes
    for col in attrs.columns.difference(new_attrs.columns):
        new_attrs[col] = np.nan if pd.api.types.is_numeric_dtype(attrs[col]) else None

    # Align well IDs between attrs and trajectory, perform interpolation, and map the attributes
    # Loop dict
    for survey_well_name in survey.well_id_mapper:
        # Select rows corresponding to the current well ID

        # use the well_id to get all the elements of attrs that have the well_id as index
        if survey_well_name in attrs.index:
            attrs_well = attrs.loc[[survey_well_name]]
            # Proceed with processing attrs_well
        else:
            print(f"Well '{survey_well_name}' does not exist in the attributes DataFrame.")
            continue

        survey_well_id = survey.get_well_num_id(survey_well_name)
        trajectory_well_mask = (trajectory_well_id == survey_well_id).values

        # Apply mask to measured depths for the current well
        well_measured_depths = measured_depths[trajectory_well_mask]

        # Interpolation for each attribute column
        for col in attrs_well.columns:
            # Interpolate the attribute values based on the measured depths
            if col in ['top', 'base', 'well_id']:
                continue
            attr_to_interpolate = attrs_well[col]
            # make sure the attr_to_interpolate is not a string
            if attr_to_interpolate.dtype == 'O':
                continue

            location_values_to_interpolate = (attrs_well['top'] + attrs_well['base']) / 2

            from scipy.interpolate import interp1d
            interp_func = interp1d(
                x=location_values_to_interpolate,
                y=attr_to_interpolate,
                bounds_error=False,
                fill_value=np.nan
            )

            # Assign the interpolated values to the new_attrs DataFrame
            new_attrs.loc[trajectory_well_mask, col] = interp_func(well_measured_depths)

    return new_attrs


def _map_attrs_to_measured_depths_(attrs: pd.DataFrame, new_attrs: pd.DataFrame, survey: Survey):
    warnings.warn("This function is obsolete. Use _map_attrs_to_measured_depths instead.", DeprecationWarning)
     
    trajectory: xr.DataArray = survey.survey_trajectory.data.data["vertex_attrs"]
    well_ids: xr.DataArray = trajectory.sel({'vertex_attr': 'well_id'})
    measured_depths: xr.DataArray = trajectory.sel({'vertex_attr': 'measured_depths'})

    new_columns = attrs.columns.difference(new_attrs.columns)
    new_attrs = pd.concat([new_attrs, pd.DataFrame(columns=new_columns)], axis=1)
    for index, row in attrs.iterrows():
        well_id = survey.get_well_num_id(index)
        if well_id is None:
            print(f'Well ID {index} not found in survey trajectory. Skipping lithology assignment.')

        well_id_mask = well_ids == well_id

        # TODO: Here we are going to need to interpolate

        spatial_mask = ((measured_depths <= row['top']) & (measured_depths >= row['base']))
        mask = well_id_mask & spatial_mask

        new_attrs.loc[mask.values, attrs.columns] = row.values
    return new_attrs


def _correct_angles(df: pd.DataFrame) -> pd.DataFrame:
    def correct_inclination(inc: float) -> float:
        if inc < 0:
            inc = inc % 360  # Normalize to 0-360 range first if negative
        if 0 <= inc <= 180:
            return inc + 0.000001
        elif 180 < inc < 360:
            return 360 - inc  # Reflect angles greater than 180 back into the 0-180 range
        else:
            raise ValueError(f'Inclination value {inc} is out of the expected range of 0 to 360 degrees')

    def correct_azimuth(azi: float) -> float:
        return azi % 360  # Normalize azimuth to 0-360 range

    df['inc'] = df['inc'].apply(correct_inclination)
    df['azi'] = df['azi'].apply(correct_azimuth)

    return df


def _data_frame_to_unstructured_data(df: 'pd.DataFrame'):
    import numpy as np

    wp = optional_requirements.require_wellpathpy()
    pd = optional_requirements.require_pandas()

    cum_vertex: np.ndarray = np.empty((0, 3), dtype=np.float_)
    cells: np.ndarray = np.empty((0, 2), dtype=np.int_)
    cell_attr: pd.DataFrame = pd.DataFrame(columns=['well_id'], dtype=np.float32)
    vertex_attr: pd.DataFrame = pd.DataFrame(columns=['well_id'], dtype=np.float32)

    for e, (borehole_id, data) in enumerate(df.groupby(level=0)):
        dev: wp.deviation = wp.deviation(
            md=data['md'],
            inc=data['inc'],
            azi=data['azi']
        )
        depths = list(range(0, int(dev.md[-1]) + 1, STEP))
        pos: wp.minimum_curvature = dev.minimum_curvature().resample(depths=depths)
        vertex_count = cum_vertex.shape[0]

        this_well_vertex = np.vstack([pos.easting, pos.northing, pos.depth]).T
        cum_vertex = np.vstack([cum_vertex, this_well_vertex])
        measured_depths = _calculate_distances(array_of_vertices=this_well_vertex)

        n_vertex_shift_0 = np.arange(0, len(pos.depth) - 1, dtype=np.int_)
        n_vertex_shift_1 = np.arange(1, len(pos.depth), dtype=np.int_)
        cell_per_well = np.vstack([n_vertex_shift_0, n_vertex_shift_1]).T + vertex_count
        cells = np.vstack([cells, cell_per_well], dtype=np.int_)

        # Add the id (e), to cell_attr and vertex_attr
        cell_attr = pd.concat([cell_attr, pd.DataFrame({'well_id': [e] * len(cell_per_well)})])
        vertex_attr = pd.concat([vertex_attr, pd.DataFrame(
            {
                    'well_id'        : [e] * len(pos.depth),
                    # 'well_name'      : borehole_id,
                    'measured_depths': measured_depths,
            }
        )])

    unstruct = UnstructuredData.from_array(
        vertex=cum_vertex,
        cells=cells,
        vertex_attr=vertex_attr.reset_index(),
        cells_attr=cell_attr.reset_index()
    )
    
    unstruct.data.attrs["well_id_mapper"] = {well_id: e for e, well_id in enumerate(df.index.unique(level=0))}

    return unstruct


def _calculate_distances(array_of_vertices: np.ndarray) -> np.ndarray:
    # Calculate the differences between consecutive points
    differences = np.diff(array_of_vertices, axis=0)

    # Calculate the Euclidean distance for each pair of consecutive points
    distances = np.linalg.norm(differences, axis=1)
    # Insert a 0 at the beginning to represent the starting point at the surface
    measured_depths = np.insert(np.cumsum(distances), 0, 0)
    return measured_depths
