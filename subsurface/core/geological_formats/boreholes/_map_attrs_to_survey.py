import numpy as np
import pandas as pd
import xarray as xr

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

def _map_attrs_to_measured_depths(attrs: pd.DataFrame, survey_trajectory: LineSet, well_id_mapper: dict[str, int]) -> pd.DataFrame:
    trajectory: xr.DataArray = survey_trajectory.data.data["vertex_attrs"]
    trajectory_well_id: xr.DataArray = trajectory.sel({'vertex_attr': 'well_id'})
    measured_depths: np.ndarray = trajectory.sel({'vertex_attr': 'measured_depths'}).values.astype(np.float64)

    # Start with a copy of the existing attributes DataFrame
    new_attrs = survey_trajectory.data.points_attributes.copy()
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
    for survey_well_name in well_id_mapper:
        # Select rows corresponding to the current well ID

        # use the well_id to get all the elements of attrs that have the well_id as index
        if survey_well_name in attrs.index:
            attrs_well = attrs.loc[[survey_well_name]]
            # Proceed with processing attrs_well
        else:
            print(f"Well '{survey_well_name}' does not exist in the attributes DataFrame.")
            continue

        survey_well_id = well_id_mapper.get(survey_well_name, None)
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
