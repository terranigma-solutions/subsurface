import numpy as np
import warnings

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
import pandas as pd

from subsurface.modules.reader.wells._read_to_df import check_format_and_read_to_df
from subsurface.modules.reader.wells.wells_utils import add_tops_from_base_and_altitude_in_place


def read_collar(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    if reader_helper.index_col is False: reader_helper.index_col = 0

    # Check file_or_buffer type
    data_df: pd.DataFrame = check_format_and_read_to_df(reader_helper)
    _map_rows_and_cols_inplace(data_df, reader_helper)
    _coerce_numeric_columns(data_df, reader_helper)

    # Remove duplicates
    data_df = data_df[~data_df.index.duplicated(keep='first')]

    return data_df


def read_survey(reader_helper: GenericReaderFilesHelper, validate_survey: bool = True) -> pd.DataFrame:
    if reader_helper.index_col is False: reader_helper.index_col = 0

    d = check_format_and_read_to_df(reader_helper)
    _map_rows_and_cols_inplace(d, reader_helper)

    if validate_survey:
        d_no_singles = _validate_survey_data(d)
    else:
        d_no_singles = d

    return d_no_singles


def read_lith(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    return read_attributes(reader_helper, is_lith=True)


def read_attributes(reader_helper: GenericReaderFilesHelper, is_lith: bool = False, validate_attr: bool = True) -> pd.DataFrame:
    if reader_helper.index_col is False:
        reader_helper.index_col = 0
        
    d = check_format_and_read_to_df(reader_helper)

    _map_rows_and_cols_inplace(d, reader_helper)
    _coerce_numeric_columns(d, reader_helper)
    if validate_attr is False:
        return d
    
    if is_lith:
        d = _validate_lith_data(d, reader_helper)
    else:
        _validate_attr_data(d)
    return d


def _coerce_numeric_columns(d: pd.DataFrame, reader_helper: GenericReaderFilesHelper) -> None:
    if reader_helper.coerce_numeric is not None:
        for col in reader_helper.coerce_numeric:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')


def _map_rows_and_cols_inplace(d: pd.DataFrame, reader_helper: GenericReaderFilesHelper):
    if reader_helper.index_map is not None:
        d.rename(reader_helper.index_map, axis="index", inplace=True)  # d.index = d.index.map(reader_helper.index_map)
    if reader_helper.columns_map is not None:
        d.rename(reader_helper.columns_map, axis="columns", inplace=True)


def _validate_survey_data(d):
    # Check for essential column 'md'
    if 'md' not in d.columns:
        raise AttributeError(
            'md, inc, and azi columns must be present in the file. Use columns_map to assign column names to these fields.')

    # Drop rows with NaN in essential survey columns
    d.dropna(subset=['md'], inplace=True)

    # Check if 'dip' column exists and convert it to 'inc'
    if 'dip' in d.columns:
        d.dropna(subset=['dip'], inplace=True)
        # Convert dip to inclination (90 - dip)
        d['inc'] = 90 - d['dip']
        # Optionally, drop the 'dip' column if it's no longer needed
        d.drop(columns=['dip'], inplace=True)

    # Handle if inclination ('inc') or azimuth ('azi') columns are missing
    if not np.isin(['inc', 'azi'], d.columns).all():
        warnings.warn(
            'inc and/or azi columns are not present in the file. The boreholes will be straight.')
        d['inc'] = 180
        d['azi'] = 0

    # A positive-depth singleton is a valid total-depth survey. The trajectory
    # builder adds its missing collar station at measured depth zero.
    station_count = d.groupby(level=0)['md'].transform('size')
    valid_single_station = station_count.eq(1) & d['md'].gt(0)
    d_no_singles = d[station_count.gt(1) | valid_single_station]

    return d_no_singles


def _validate_attr_data(d):
    assert d.columns.isin(['base']).any(), ('base column must be present in the file. '
                                            'Use columns_map to assign column names to these fields.')


def _validate_lith_data(d: pd.DataFrame, reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    # Check component lith in column
    if 'component lith' not in d.columns:
        raise AttributeError('If wells attributes represent lithology, `component lith` column must be present in the file. '
                             'Use columns_map to assign column names to these fields. Maybe you are marking as lithology'
                             'the wrong file?')
    else:
        # TODO: Add categories to reader helper
        categories = sorted(d['component lith'].dropna().unique())
        d['component lith'] = pd.Categorical(
            d['component lith'],
            categories=categories,
            ordered=True
        )
        
        d['lith_ids'] = d['component lith'].cat.codes + 1

    given_top = np.isin(['top', 'base'], d.columns).all()
    given_altitude_and_base = np.isin(['altitude', 'base'], d.columns).all()
    given_only_base = np.isin(['base'], d.columns).all()
    if given_altitude_and_base and not given_top:
        warnings.warn('top column is not present in the file. The tops will be calculated from the base and altitude')
        d = add_tops_from_base_and_altitude_in_place(
            data=d,
            col_well_name=reader_helper.index_col,
            col_base='base',
            col_altitude='altitude'
        )
    elif given_only_base and not given_top:
        warnings.warn('top column is not present in the file. The tops will be calculated from the base assuming altitude=0')
        # add a top column with 0 and call add_tops_from_base_and_altitude_in_place
        d['altitude'] = 0
        d = add_tops_from_base_and_altitude_in_place(
            data=d,
            col_well_name=reader_helper.index_col,
            col_base='base',
            col_altitude='altitude'
        )


    elif not given_top and not given_altitude_and_base:
        raise ValueError('top column or base and altitude columns must be present in the file. '
                         'Use columns_map to assign column names to these fields. Maybe you are marking as lithology'
                         'the wrong file?')

    # * Make sure values are positive
    d['top'] = np.abs(d['top'])
    d['base'] = np.abs(d['base'])
    _warn_about_ambiguous_lithology_intervals(d)

    return d


def _warn_about_ambiguous_lithology_intervals(d: pd.DataFrame) -> None:
    zero_thickness_count = int(d['top'].eq(d['base']).sum())
    overlapping_well_count = 0
    conflicting_interval_count = 0

    for _, intervals in d.groupby(level=0, sort=False):
        sorted_intervals = intervals.sort_values(['top', 'base'])
        previous_base = sorted_intervals['base'].cummax().shift()
        overlapping_well_count += int(sorted_intervals['top'].lt(previous_base).any())

        lithologies_per_interval = intervals.groupby(
            ['top', 'base'],
            dropna=False,
            observed=True,
        )['component lith'].nunique(dropna=True)
        conflicting_interval_count += int(lithologies_per_interval.gt(1).sum())

    if not any((zero_thickness_count, overlapping_well_count, conflicting_interval_count)):
        return

    warnings.warn(
        'Lithology data contains '
        f'{zero_thickness_count} zero-thickness intervals, overlapping intervals in '
        f'{overlapping_well_count} wells, and {conflicting_interval_count} interval '
        'keys with conflicting lithologies. The records are preserved; resolve these '
        'ambiguities before treating them as a canonical lithology log.',
        UserWarning,
        stacklevel=2,
    )
