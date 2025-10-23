import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Optional, Union, List, Any

from subsurface import optional_requirements
from ...structs.base_structures import UnstructuredData
from ...structs.base_structures._unstructured_data_constructor import raw_attributes_to_dict_data_arrays
from ...structs.unstructured_elements import LineSet


def combine_survey_and_attrs(attrs: pd.DataFrame, survey_trajectory: LineSet,well_id_mapper: dict[str, int]) -> UnstructuredData:
    # Import moved to top for clarity and possibly avoiding repeated imports if called multiple times

    # Ensure all columns in lith exist in new_attrs, if not, add them as NaN
    new_attrs = _map_attrs_to_measured_depths(attrs, survey_trajectory, well_id_mapper)

    # Construct the final xarray dict without intermediate variable
    points_attributes_xarray_dict: dict[str, xr.DataArray] = raw_attributes_to_dict_data_arrays(
        default_attributes_name="vertex_attrs",
        n_items=survey_trajectory.data.data["vertex_attrs"].shape[0],  # TODO: Can I look this on new_attrs to remove line 11?
        dims=["points", "vertex_attr"],
        raw_attributes=new_attrs
    )

    # Inline construction of UnstructuredData
    return UnstructuredData.from_data_arrays_dict(
        xarray_dict={
                "vertex"      : survey_trajectory.data.data["vertex"],
                "cells"       : survey_trajectory.data.data["cells"],
                "vertex_attrs": points_attributes_xarray_dict["vertex_attrs"],
                "cell_attrs"  : survey_trajectory.data.data["cell_attrs"]
        },
        xarray_attributes=survey_trajectory.data.data.attrs,
        default_cells_attributes_name=survey_trajectory.data.cells_attr_name,
        default_points_attributes_name=survey_trajectory.data.vertex_attr_name
    )


def _prepare_categorical_data(attrs: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare categorical data for interpolation by converting categorical columns to numeric IDs.

    Args:
        attrs: DataFrame containing attribute data

    Returns:
        Modified DataFrame with categorical data prepared for interpolation
    """
    # Create a copy to avoid modifying the original
    attrs_copy = attrs.copy()

    # If component lith exists but lith_ids doesn't, create lith_ids
    if 'component lith' in attrs_copy.columns and 'lith_ids' not in attrs_copy.columns:
        attrs_copy['lith_ids'], _ = pd.factorize(attrs_copy['component lith'], use_na_sentinel=True)

    return attrs_copy


def _prepare_new_attributes(attrs: pd.DataFrame, survey_trajectory: LineSet) -> pd.DataFrame:
    """
    Prepare the new attributes DataFrame by adding missing columns from attrs.

    Args:
        attrs: DataFrame containing attribute data
        survey_trajectory: LineSet containing trajectory data

    Returns:
        New attributes DataFrame with all necessary columns
    """
    # Start with a copy of the existing attributes DataFrame
    new_attrs = survey_trajectory.data.points_attributes.copy()

    # Add missing columns from attrs, preserving their dtypes
    for col in attrs.columns.difference(new_attrs.columns):
        new_attrs[col] = np.nan if pd.api.types.is_numeric_dtype(attrs[col]) else None

    return new_attrs


def _get_interpolation_locations(attrs_well: pd.DataFrame, well_name: str) -> np.ndarray:
    """
    Determine the locations to use for interpolation based on top and base values.

    Args:
        attrs_well: DataFrame containing well attribute data
        well_name: Name of the current well

    Returns:
        Array of location values to use for interpolation
    """
    if "base" not in attrs_well.columns:
        raise ValueError(f"Base column must be present in the file for well '{well_name}'.")
    elif "top" not in attrs_well.columns:
        return attrs_well['base'].values
    else:
        return ((attrs_well['top'] + attrs_well['base']) / 2).values


def _nearest_neighbor_categorical_interpolation(
    x_locations: np.ndarray,
    y_values: np.ndarray,
    target_depths: np.ndarray
) -> np.ndarray:
    """
    Custom nearest neighbor interpolation for categorical data.

    This function finds the nearest source point for each target point
    and assigns the corresponding categorical value.

    Args:
        x_locations: Array of source locations
        y_values: Array of categorical values at source locations
        target_depths: Array of target depths for interpolation

    Returns:
        Array of interpolated categorical values
    """
    # Initialize output array with NaN or None values
    result = np.full(target_depths.shape, np.nan, dtype=object)

    # For each target depth, find the nearest source location
    for i, depth in enumerate(target_depths):
        # Calculate distances to all source locations
        distances = np.abs(x_locations - depth)

        # Find the index of the minimum distance
        if len(distances) > 0:
            nearest_idx = np.argmin(distances)
            result[i] = y_values[nearest_idx]

    return result


def _interpolate_attribute(
    attr_values: pd.Series, 
    x_locations: np.ndarray, 
    target_depths: np.ndarray, 
    column_name: str,
    is_categorical: bool
) -> np.ndarray:
    """
    Interpolate attribute values to target depths.

    Args:
        attr_values: Series containing attribute values
        x_locations: Array of source locations for interpolation
        target_depths: Array of target depths for interpolation
        column_name: Name of the column being interpolated
        is_categorical: Whether the attribute is categorical

    Returns:
        Array of interpolated values
    """
    # For categorical data or specific columns, use custom nearest neighbor interpolation
    if is_categorical or column_name in ['lith_ids', 'component lith']:
        return _nearest_neighbor_categorical_interpolation(
            x_locations=x_locations,
            y_values=attr_values.values,
            target_depths=target_depths
        )
    else:
        # For numerical data, use scipy's interp1d with linear interpolation
        scipy = optional_requirements.require_scipy()
        interp_func = scipy.interpolate.interp1d(
            x=x_locations,
            y=attr_values.values,
            bounds_error=False,
            fill_value=np.nan,
            kind='linear'
        )
        return interp_func(target_depths)


def _map_attrs_to_measured_depths(attrs: pd.DataFrame, survey_trajectory: LineSet, well_id_mapper: dict[str, int]) -> pd.DataFrame:
    """
    Map attributes to measured depths for each well.

    Args:
        attrs: DataFrame containing attribute data
        survey_trajectory: LineSet containing trajectory data
        well_id_mapper: Dictionary mapping well names to IDs

    Returns:
        DataFrame with attributes mapped to measured depths
    """
    # Extract trajectory data
    trajectory: xr.DataArray = survey_trajectory.data.data["vertex_attrs"]
    trajectory_well_id: xr.DataArray = trajectory.sel({'vertex_attr': 'well_id'})
    measured_depths: np.ndarray = trajectory.sel({'vertex_attr': 'measured_depths'}).values.astype(np.float64)

    # Prepare data
    attrs: pd.DataFrame = _prepare_categorical_data(attrs)
    new_attrs: pd.DataFrame = _prepare_new_attributes(attrs, survey_trajectory)

    # Process each well
    for well_name in well_id_mapper:
        # Skip wells not in the attributes DataFrame
        if well_name not in attrs.index:
            print(f"Well '{well_name}' does not exist in the attributes DataFrame.")
            continue

        # Get well data
        attrs_well = attrs.loc[[well_name]]
        well_id = well_id_mapper.get(well_name)
        well_mask = (trajectory_well_id == well_id).values
        well_depths = measured_depths[well_mask]

        # Get interpolation locations
        interp_locations = _get_interpolation_locations(attrs_well, well_name)

        # Interpolate each attribute
        for col in attrs_well.columns:
            # Skip location and ID columns
            if col in ['top', 'base', 'well_id']:
                continue

            attr_values = attrs_well[col]
            is_categorical = attr_values.dtype == 'O' or isinstance(attr_values.dtype, pd.CategoricalDtype)

            # Skip columns that can't be interpolated and aren't categorical
            if is_categorical and col not in ['lith_ids', 'component lith']:
                continue

            # Interpolate and assign values
            interpolated_values = _interpolate_attribute(
                attr_values, 
                interp_locations, 
                well_depths, 
                col,
                is_categorical
            )

            new_attrs.loc[well_mask, col] = interpolated_values

    return new_attrs
