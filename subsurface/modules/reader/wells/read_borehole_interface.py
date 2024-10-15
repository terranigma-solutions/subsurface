import numpy as np
import warnings

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
import pandas as pd

from subsurface.modules.reader.wells._read_to_df import check_format_and_read_to_df
from subsurface.modules.reader.wells.wells_utils import add_tops_from_base_and_altitude_in_place


def read_collar(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    if reader_helper.usecols is None: reader_helper.usecols = [0, 1, 2, 3]
    if reader_helper.index_col is False: reader_helper.index_col = 0

    # Check file_or_buffer type
    data_df: pd.DataFrame = check_format_and_read_to_df(reader_helper)
    _map_rows_and_cols_inplace(data_df, reader_helper)

    # Remove duplicates
    data_df = data_df[~data_df.index.duplicated(keep='first')]

    return data_df


def read_survey(reader_helper: GenericReaderFilesHelper):
    if reader_helper.index_col is False: reader_helper.index_col = 0

    d = check_format_and_read_to_df(reader_helper)
    _map_rows_and_cols_inplace(d, reader_helper)

    d_no_singles = _validate_survey_data(d)

    return d_no_singles


def read_lith(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    return read_attributes(reader_helper, is_lith=True)


def read_attributes(reader_helper: GenericReaderFilesHelper, is_lith: bool = False) -> pd.DataFrame:
    if reader_helper.index_col is False: reader_helper.index_col = 0

    d = check_format_and_read_to_df(reader_helper)

    _map_rows_and_cols_inplace(d, reader_helper)
    if is_lith:
        d = _validate_lith_data(d, reader_helper)
    else:
        _validate_attr_data(d)
    return d


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

    # Check if 'dip' column exists and convert it to 'inc'
    if 'dip' in d.columns:
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

    # Drop wells that contain only one value, ensuring that we keep rows only when there are duplicates
    d_no_singles = d[d.index.duplicated(keep=False)]

    return d_no_singles


def _validate_attr_data(d):
    assert d.columns.isin(['base']).any(), ('base column must be present in the file. '
                                            'Use columns_map to assign column names to these fields.')


def _validate_lith_data(d: pd.DataFrame, reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    # Check component lith in column
    if 'component lith' not in d.columns:
        raise AttributeError('component lith column must be present in the file. '
                             'Use columns_map to assign column names to these fields. Maybe you are marking as lithology'
                             'the wrong file?')

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

    lith_df = d[['top', 'base', 'component lith']]

    # * Make sure values are positive
    lith_df['top'] = np.abs(lith_df['top'])
    lith_df['base'] = np.abs(lith_df['base'])

    return lith_df
