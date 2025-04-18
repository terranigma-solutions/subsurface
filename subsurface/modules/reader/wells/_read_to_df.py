import io
import pathlib
from typing import Callable

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper, SupportedFormats
import pandas as pd


def check_format_and_read_to_df(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    # ? This swithch is veeery confusing
    match (reader_helper.file_or_buffer, reader_helper.format):
        case _, SupportedFormats.JSON | ".json":
            d = pd.read_json(reader_helper.file_or_buffer, orient='split')
        case str() | pathlib.Path(), _:
            reader: Callable = _get_reader(reader_helper.format)
            d = reader(
                filepath_or_buffer=reader_helper.file_or_buffer,
                sep=reader_helper.separator,
                **reader_helper.pandas_reader_kwargs
            )
        case (bytes() | io.BytesIO() | io.StringIO() | io.TextIOWrapper()), _:
            reader = _get_reader(reader_helper.format)
            d = reader(reader_helper.file_or_buffer, **reader_helper.pandas_reader_kwargs)
        case dict(), _:
            reader = _get_reader('dict')
            d = reader(reader_helper.file_or_buffer)
        case _:
            raise AttributeError('file_or_buffer must be either a path or a dict')
            
    if type(d.columns) is str:  d.columns = d.columns.str.strip()  # Remove spaces at the beginning and end
    if type(d.index) is str: d.index = d.index.str.strip()  # Remove spaces at the beginning and end
    return d




def _get_reader(file_format) -> Callable:
    def _dict_reader(dict_):
        return pd.DataFrame(
            data=dict_['data'],
            columns=dict_['columns'],
            index=dict_['index']
        )

    match file_format:
        case SupportedFormats.XLXS:
            raise NotImplemented("Pandas changed the backend for reading excel files and needs to be re-implemented")
            reader = pd.read_excel
        case 'dict':
            reader = _dict_reader
        case SupportedFormats.CSV:
            reader = pd.read_csv
        case SupportedFormats.JSON:
            reader = _dict_reader
        case _:
            raise ValueError(f"Subsurface is not able to read the following extension: {file_format}")
    return reader
