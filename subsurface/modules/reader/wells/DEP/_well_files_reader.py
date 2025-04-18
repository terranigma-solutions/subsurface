import pandas as pd
import warnings
from typing import Dict

from subsurface.reader.readers_data import ReaderFilesHelper, ReaderWellsHelper, SupportedFormats
from subsurface.reader.wells.wells_utils import add_tops_from_base_and_altitude_in_place


def read_borehole_files(reader_wells_helper: ReaderWellsHelper) -> Dict[str, pd.DataFrame]:
    data_frames = dict()

    data_frames['collar_df'] = read_collar(reader_wells_helper.reader_collars_args)

    data_frames['survey_df'] = read_survey(reader_wells_helper.reader_survey_args)

    if reader_wells_helper.reader_lith_args is not None:
        data_frames['lith_df'] = read_lith(reader_wells_helper.reader_lith_args)

    if reader_wells_helper.reader_attr_args is not None:
        attributes_ = list()
        for e in reader_wells_helper.reader_attr_args:
            attributes_.append(read_attributes(e))
        data_frames['attrib_dfs'] = attributes_

    return data_frames


def read_collar(reader_helper: ReaderFilesHelper) -> pd.DataFrame:
    if reader_helper.usecols is None: reader_helper.usecols = [0, 1, 2, 3]
    if reader_helper.index_col is False: reader_helper.index_col = 0

    # Check file_or_buffer type
    d = check_format_and_read_to_df(reader_helper)
    map_rows_and_cols_inplace(d, reader_helper)

    return d


def read_survey(reader_helper: ReaderFilesHelper):
    if reader_helper.index_col is False: reader_helper.index_col = 0

    d = check_format_and_read_to_df(reader_helper)
    map_rows_and_cols_inplace(d, reader_helper)

    d_no_singles = _validate_survey_data(d)

    return d_no_singles


def read_lith(reader_helper: ReaderFilesHelper):
    """Columns MUST contain:
        - top
        - base
        - component lith
    """
    if reader_helper.index_col is False: reader_helper.index_col = 0

    d = check_format_and_read_to_df(reader_helper)
    map_rows_and_cols_inplace(d, reader_helper)
    lith_df = _validate_lith_data(d, reader_helper)

    return lith_df


def read_attributes(reader_helper: ReaderFilesHelper) -> pd.DataFrame:
    if reader_helper.index_col is False: reader_helper.index_col = 0

    d = check_format_and_read_to_df(reader_helper)

    if reader_helper.columns_map is not None: d.rename(reader_helper.columns_map, axis=1, inplace=True)
    if reader_helper.drop_cols is not None: d.drop(reader_helper.drop_cols, axis=1, inplace=True)

    _validate_attr_data(d)
    return d


def read_survey_df_from_las(reader_helper: ReaderFilesHelper, well_name: str) -> pd.DataFrame:
    """
    Reads a las file and returns a dataframe.
    
    """
    from subsurface.reader.wells._welly_reader import _create_welly_well_from_las
    welly_well = _create_welly_well_from_las(well_name, reader_helper.file_or_buffer)
    survey_df = welly_well.df()[reader_helper.usecols]
    map_rows_and_cols_inplace(survey_df, reader_helper)
    survey_df["well_name"] = well_name
    survey_df.set_index("well_name", inplace=True)
    return survey_df


def read_assay_df_from_las(reader_helper: ReaderFilesHelper, well_name: str) -> pd.DataFrame:
    from subsurface.reader.wells._welly_reader import _create_welly_well_from_las
    welly_well = _create_welly_well_from_las(well_name, reader_helper.file_or_buffer)
    assay_df = welly_well.df()
    assay_df["well_name"] = well_name
    assay_df.set_index("well_name", inplace=True)
    return assay_df


def check_format_and_read_to_df(reader_helper: ReaderFilesHelper) -> pd.DataFrame:
    if reader_helper.format == ".json":
        d = pd.read_json(reader_helper.file_or_buffer, orient='split')
    elif reader_helper.is_file_in_disk:
        reader = _get_reader(reader_helper.format)
        d = reader(reader_helper.file_or_buffer, **reader_helper.pandas_reader_kwargs)
    elif reader_helper.is_bytes_string:
        reader = _get_reader('.csv')
        d = reader(reader_helper.file_or_buffer, **reader_helper.pandas_reader_kwargs)
    elif reader_helper.is_python_dict:
        reader = _get_reader('dict')
        d = reader(reader_helper.file_or_buffer)
    else:
        raise AttributeError('file_or_buffer must be either a path or a dict')

    if type(d.columns) is str:  d.columns = d.columns.str.strip()  # Remove spaces at the beginning and end
    if type(d.index) is str: d.index = d.index.str.strip()  # Remove spaces at the beginning and end
    return d




def _get_reader(file_format):
    if file_format == SupportedFormats.XLXS:
        reader = pd.read_excel
    elif file_format == 'dict':
        reader = _dict_reader
    elif file_format == SupportedFormats.CSV:
        reader = pd.read_csv
    elif file_format == SupportedFormats.JSON:
        reader = _dict_reader
    else:
        raise ValueError(f"Subsurface is not able to read the following extension: {file_format}")
    return reader


def _dict_reader(dict_):
    """

    Args:
        dict_: data, index, columns

    """
    return pd.DataFrame(data=dict_['data'],
                        columns=dict_['columns'],
                        index=dict_['index'])




def _validate_lith_data(d: pd.DataFrame, reader_helper: ReaderFilesHelper) -> pd.DataFrame:
    given_top = pd.np.isin(['top', 'base', 'component lith'], d.columns).all()
    given_altitude_and_base = pd.np.isin(['altitude', 'base', 'component lith'], d.columns).all()

    if given_altitude_and_base and not given_top:
        d = add_tops_from_base_and_altitude_in_place(d, reader_helper.index_col, 'base', 'altitude')
    elif not given_top and not given_altitude_and_base:
        raise ValueError('basis column must be present in the file. Use '
                         'columns_map to assign column names to these fields.')
    lith_df = d[['top', 'base', 'component lith']]
    return lith_df


def _validate_attr_data(d):
    assert d.columns.isin(['basis']).any(), 'basis column' \
                                            'must be present in the file.' \
                                            'Use columns_map to assign' \
                                            'column names to these fields.'
