import enum
import pathlib
import pandas as pd

from subsurface.core.utils.utils_core import get_extension
from pydantic import BaseModel, Field, model_validator
from typing import Union, List, Optional

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
    index_col: Optional[Union[int, str, bool]] = False
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

        # Custom validation for index_col to explicitly handle None

    @model_validator(mode="before")
    def validate_additional_reader_kwargs(cls, values):
        additional_reader_kwargs = values.get('additional_reader_kwargs')
        # Make sure that if any of the values is a regex expression that it is properly parsed like "delimiter":"\\\\s{2,}" to delimiter="\s{2,}"
        if additional_reader_kwargs is not None:
            for key, value in additional_reader_kwargs.items():
                if isinstance(value, str):
                    additional_reader_kwargs[key] = value.replace("\\\\", "\\")
        
        return values
    
    
    @model_validator(mode="before")
    def validate_index_col(cls, values):
        index_col = values.get('index_col')
        # Allow None explicitly
        if index_col is None:
            values['index_col'] = False
        else:
            # Ensure index_col is either int, str, or bool
            if not isinstance(index_col, (int, str, bool)):
                raise ValueError(f"Invalid value for index_col: {index_col}. Must be int, str, bool, or None.")

        return values

    # Validator to handle negative header values. If -1 is the same as null, other raise an error
    @model_validator(mode="before")
    def validate_header(cls, values):
        header = values.get('header')
        if header == -1:
            values['header'] = None
            header = None
        if header is not None and header < 0:
            raise ValueError(f"Invalid value for header: {header}. Must be None, 0, or positive integer.")
        return values

    @property
    def pandas_reader_kwargs(self):
        attr_dict = {
                "names"    : self.col_names,
                "header"   : self.header,
                "index_col": self.index_col,
                "usecols"  : self.usecols,
                "encoding" : self.encoding
        }
        # Check if delimiter or separator is in additional_reader_kwargs if not add it here
        if self.additional_reader_kwargs:
            delimiter = self.additional_reader_kwargs.get("delimiter", None)
        else:
            delimiter = None
        if self.separator is not None and delimiter is None:
            attr_dict["sep"] = self.separator
        
        return {**attr_dict, **self.additional_reader_kwargs}
   