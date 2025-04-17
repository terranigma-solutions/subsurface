import warnings

from typing import Union, Hashable, Optional

import pandas as pd
from dataclasses import dataclass
import numpy as np
import xarray as xr

from subsurface import optional_requirements
from ...structs.unstructured_elements import LineSet
from ...structs.base_structures import UnstructuredData

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
        trajectories: UnstructuredData = _data_frame_to_unstructured_data(
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

    # Construct the final xarray dict without intermediate variable
    points_attributes_xarray_dict = raw_attributes_to_dict_data_arrays(
        default_attributes_name="vertex_attrs",
        n_items=trajectory.shape[0],  # TODO: Can I look this on new_attrs to remove line 11?
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
    measured_depths: np.ndarray = trajectory.sel({'vertex_attr': 'measured_depths'}).values.astype(np.float64)

    # Start with a copy of the existing attributes DataFrame
    new_attrs = survey.survey_trajectory.data.points_attributes.copy()
    if 'component lith' in attrs.columns and 'lith_ids' not in attrs.columns:
        # Factorize lith components directly in-place
        attrs['lith_ids'], _ = pd.factorize(attrs['component lith'], use_na_sentinel=True)
    else:
        pass

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

        if "base" not in attrs_well.columns:
            raise ValueError(f"Base column must be present in the file for well '{survey_well_name}'.")
        elif "top" not in attrs_well.columns:
            location_values_to_interpolate = attrs_well['base']
        else:
            location_values_to_interpolate = (attrs_well['top'] + attrs_well['base']) / 2

        # Interpolation for each attribute column
        for col in attrs_well.columns:
            # Interpolate the attribute values based on the measured depths
            if col in ['top', 'base', 'well_id']:
                continue
            attr_to_interpolate = attrs_well[col]
            # make sure the attr_to_interpolate is not a string
            if attr_to_interpolate.dtype == 'O' or isinstance(attr_to_interpolate.dtype, pd.CategoricalDtype): 
                continue
            if col in ['lith_ids', 'component lith']:
                interp_kind = 'nearest'
            else:
                interp_kind = 'linear'

            from scipy.interpolate import interp1d
            interp_func = interp1d(
                x=location_values_to_interpolate,
                y=attr_to_interpolate,
                bounds_error=False,
                fill_value=np.nan,
                kind=interp_kind
            )

            # Assign the interpolated values to the new_attrs DataFrame
            vals = interp_func(well_measured_depths)
            new_attrs.loc[trajectory_well_mask, col] = vals

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


def _data_frame_to_unstructured_data(survey_df: 'pd.DataFrame', number_nodes: int, attr_df: Optional['pd.DataFrame'] = None,
                                     duplicate_attr_depths: bool = False) -> UnstructuredData:

    wp = optional_requirements.require_wellpathpy()

    cum_vertex: np.ndarray = np.empty((0, 3), dtype=np.float32)
    cells: np.ndarray = np.empty((0, 2), dtype=np.int_)
    cell_attr: pd.DataFrame = pd.DataFrame(columns=['well_id'], dtype=np.float32)
    vertex_attr: pd.DataFrame = pd.DataFrame()

    for e, (borehole_id, data) in enumerate(survey_df.groupby(level=0)):
        dev = wp.deviation(
            md=data['md'].values,
            inc=data['inc'].values,
            azi=data['azi'].values
        )

        md_min = dev.md.min()
        md_max = dev.md.max()

        attr_depths = _grab_depths_from_attr(
            attr_df=attr_df,
            borehole_id=borehole_id,
            duplicate_attr_depths=duplicate_attr_depths,
            md_max=md_max,
            md_min=md_min
        )

        # Now combine attr_depths with depths
        md_min = dev.md.min()
        md_max = dev.md.max()
        depths = np.linspace(md_min, md_max, number_nodes)
        depths = np.union1d(depths, attr_depths)
        depths.sort()

        # Resample positions at depths
        pos = dev.minimum_curvature().resample(depths=depths)
        vertex_count = cum_vertex.shape[0]

        this_well_vertex = np.vstack([pos.easting, pos.northing, pos.depth]).T
        cum_vertex = np.vstack([cum_vertex, this_well_vertex])
        measured_depths = _calculate_distances(array_of_vertices=this_well_vertex)

        n_vertex_shift_0 = np.arange(0, len(pos.depth) - 1, dtype=np.int_)
        n_vertex_shift_1 = np.arange(1, len(pos.depth), dtype=np.int_)
        cell_per_well = np.vstack([n_vertex_shift_0, n_vertex_shift_1]).T + vertex_count
        cells = np.vstack([cells, cell_per_well])

        attribute_values = np.isin(depths, attr_depths)

        vertex_attr_per_well = pd.DataFrame({
                'well_id'        : [e] * len(pos.depth),
                'measured_depths': measured_depths,
                'is_attr_point'  : attribute_values,
        })

        vertex_attr = pd.concat([vertex_attr, vertex_attr_per_well], ignore_index=True)

        # Add the id (e), to cell_attr
        cell_attr = pd.concat([cell_attr, pd.DataFrame({'well_id': [e] * len(cell_per_well)})], ignore_index=True)

    unstruct = UnstructuredData.from_array(
        vertex=cum_vertex,
        cells=cells.astype(int),
        vertex_attr=vertex_attr.reset_index(drop=True),
        cells_attr=cell_attr.reset_index(drop=True)
    )

    unstruct.data.attrs["well_id_mapper"] = {well_id: e for e, well_id in enumerate(survey_df.index.unique(level=0))}

    return unstruct


def _grab_depths_from_attr(
        attr_df: pd.DataFrame,
        borehole_id: Hashable,
        duplicate_attr_depths: bool,
        md_max: float,
        md_min: float
) -> np.ndarray:
    # Initialize attr_depths and attr_labels as empty arrays
    attr_depths = np.array([], dtype=float)
    attr_labels = np.array([], dtype='<U4')  # Initialize labels for 'top' and 'base'

    if attr_df is None or ("top" not in attr_df.columns and "base" not in attr_df.columns):
        return attr_depths

    try:
        vals = attr_df.loc[borehole_id]

        tops = np.array([], dtype=float)
        bases = np.array([], dtype=float)

        if 'top' in vals:
            if isinstance(vals, pd.DataFrame):
                tops = vals['top'].values.flatten()
            else:
                tops = np.array([vals['top']])
            # Convert to float and remove NaNs
            tops = tops.astype(float)
            tops = tops[~np.isnan(tops)]
            # Clip to within md range
            tops = tops[(tops >= md_min) & (tops <= md_max)]

        if 'base' in vals:
            if isinstance(vals, pd.DataFrame):
                bases = vals['base'].values.flatten()
            else:
                bases = np.array([vals['base']])
            # Convert to float and remove NaNs
            bases = bases.astype(float)
            bases = bases[~np.isnan(bases)]
            # Clip to within md range
            bases = bases[(bases >= md_min) & (bases <= md_max)]

        # Combine tops and bases into attr_depths with labels
        attr_depths = np.concatenate((tops, bases))
        attr_labels = np.array(['top'] * len(tops) + ['base'] * len(bases))

        # Drop duplicates while preserving order
        _, unique_indices = np.unique(attr_depths, return_index=True)
        attr_depths = attr_depths[unique_indices]
        attr_labels = attr_labels[unique_indices]

    except KeyError:
        # No attributes for this borehole_id or missing columns
        attr_depths = np.array([], dtype=float)
        attr_labels = np.array([], dtype='<U4')

    # If duplicate_attr_depths is True, duplicate attr_depths with a tiny offset
    if duplicate_attr_depths and len(attr_depths) > 0:
        tiny_offset = (md_max - md_min) * 1e-6  # A tiny fraction of the depth range
        # Create offsets: +tiny_offset for 'top', -tiny_offset for 'base'
        offsets = np.where(attr_labels == 'top', tiny_offset, -tiny_offset)
        duplicated_attr_depths = attr_depths + offsets
        # Ensure the duplicated depths are within the md range
        valid_indices = (duplicated_attr_depths >= md_min) & (duplicated_attr_depths <= md_max)
        duplicated_attr_depths = duplicated_attr_depths[valid_indices]
        # Original attribute depths
        original_attr_depths = attr_depths
        # Combine originals and duplicates
        attr_depths = np.hstack([original_attr_depths, duplicated_attr_depths])

    return attr_depths


def _calculate_distances(array_of_vertices: np.ndarray) -> np.ndarray:
    # Calculate the differences between consecutive points
    differences = np.diff(array_of_vertices, axis=0)

    # Calculate the Euclidean distance for each pair of consecutive points
    distances = np.linalg.norm(differences, axis=1)
    # Insert a 0 at the beginning to represent the starting point at the surface
    measured_depths = np.insert(np.cumsum(distances), 0, 0)
    return measured_depths
