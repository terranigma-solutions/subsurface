import enum
import pathlib
import io
from dataclasses import dataclass, field
from typing import Union, List, Callable, Any

import pandas as pd

from subsurface.core.utils.utils_core import get_extension

if pd.__version__ < '1.4.0':
    pass
elif pd.__version__ >= '1.4.0':
    from pandas._typing import FilePath, ReadCsvBuffer

    fb = Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]]


class SupportedFormats(str, enum.Enum):
    DXF = "dxf"
    DXFStream = "dxfstream"
    CSV = "csv"
    JSON = "json"
    XLXS = "xlsx"


# 
# @dataclass
# class GenericReaderFilesHelper_:
#     file_or_buffer: fb
#     usecols: Union[List[str], List[int]] = None  # Use a subset of columns
#     col_names: List[Union[str, int]] = None  # Give a name
#     drop_cols: List[str] = None  # Drop a subset of columns
#     format: SupportedFormats = None
#     separator: str = None
#     index_map: Union[None, Callable, dict, pd.Series] = None
#     columns_map: Union[None, Callable, dict, pd.Series] = None
#     additional_reader_kwargs: dict = field(default_factory=dict)
#     file_or_buffer_type: Any = field(init=False)
#     encoding: str = "ISO-8859-1"
# 
#     index_col: Union[int, str] = False
#     header: Union[None, int, List[int]] = 0
# 
#     def __post_init__(self):
#         if self.format is None:
#             extension: str = get_extension(self.file_or_buffer)
#             if extension == ".dxf":
#                 self.format = SupportedFormats.DXF
#             elif extension == ".csv":
#                 self.format = SupportedFormats.CSV
#             elif extension == ".json":
#                 self.format = SupportedFormats.JSON
#             elif extension == ".xlsx":
#                 self.format = SupportedFormats.XLXS
# 
#         self.file_or_buffer_type = type(self.file_or_buffer)
# 
#     @property
#     def pandas_reader_kwargs(self):
#         attr_dict = {
#                 "names"    : self.col_names,
#                 "header"   : self.header,
#                 "index_col": self.index_col,
#                 "usecols"  : self.usecols,
#                 "encoding" : self.encoding
#         }
#         return {**attr_dict, **self.additional_reader_kwargs}
# 
#     @property
#     def is_file_in_disk(self):
#         return self.file_or_buffer_type == str or isinstance(self.file_or_buffer, pathlib.PurePath)
# 
#     @property
#     def is_bytes_string(self):
#         return self.file_or_buffer_type == io.BytesIO or self.file_or_buffer_type == io.StringIO
# 
#     @property
#     def is_python_dict(self):
#         return self.file_or_buffer_type == dict


from pydantic import BaseModel, Field, root_validator, model_validator, field_validator
from typing import Union, List, Optional, Any
import pathlib
import io


class GenericReaderFilesHelper(BaseModel):
    file_or_buffer: Union[str, bytes, pathlib.Path, dict]
    usecols: Optional[Union[List[str], List[int]]] = None
    col_names: Optional[List[Union[str, int]]] = None
    drop_cols: Optional[List[str]] = None
    format: Optional[SupportedFormats] = None
    separator: Optional[str] = None
    index_map: Optional[dict] = None  # Adjusted for serialization
    columns_map: Optional[dict] = None  # Adjusted for serialization
    additional_reader_kwargs: dict = Field(default_factory=dict)
    encoding: str = "ISO-8859-1"
    index_col: Union[int, str, bool] = False
    header: Union[None, int, List[int]] = 0

    # Computed fields
    file_or_buffer_type: str = Field(init=False)

    @model_validator(mode="before")
    def set_format_and_file_type(cls, values):
        file_or_buffer = values.get('file_or_buffer')
        format = values.get('format')

        # Determine format if not provided
        if format is None and file_or_buffer is not None:
            extension = get_extension(file_or_buffer)
            format_map = {
                    ".dxf" : SupportedFormats.DXF,
                    ".csv" : SupportedFormats.CSV,
                    ".json": SupportedFormats.JSON,
                    ".xlsx": SupportedFormats.XLXS,
            }
            format = format_map.get(extension.lower())
            values['format'] = format

        # Set file_or_buffer_type as a string representation
        if file_or_buffer is not None:
            values['file_or_buffer_type'] = type(file_or_buffer).__name__
        else:
            values['file_or_buffer_type'] = None

        return values

    @property
    def pandas_reader_kwargs(self):
        attr_dict = {
                "names"    : self.col_names,
                "header"   : self.header,
                "index_col": self.index_col,
                "usecols"  : self.usecols,
                "encoding" : self.encoding,
        }
        return {**attr_dict, **self.additional_reader_kwargs}

    @property
    def is_file_in_disk(self):
        return isinstance(self.file_or_buffer, (str, pathlib.Path))

    @property
    def is_bytes_string(self):
        return isinstance(self.file_or_buffer, (bytes, io.BytesIO, io.StringIO))

    @property
    def is_python_dict(self):
        return isinstance(self.file_or_buffer, dict)
