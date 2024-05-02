import enum
import pathlib
import io
from dataclasses import dataclass, field
from typing import Union, Literal, Dict, Optional, List, Callable, Any

import numpy as np
import pandas as pd
import xarray as xr

from subsurface.utils.utils_core import get_extension

__all__ = ['ReaderFilesHelper', 'ReaderUnstructuredHelper',
           'ReaderWellsHelper', 'RawDataOptions', 'RawDataUnstructured']

if pd.__version__ < '1.4.0':
    from pandas._typing import FilePathOrBuffer

    fb = FilePathOrBuffer
elif pd.__version__ >= '1.4.0':
    from pandas._typing import FilePath, ReadCsvBuffer

    fb = Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]]


class SupportedFormats(enum.Enum):
    DXF = "dxf"
    DXFStream = "dxfstream"
    CSV = "csv"
    JSON = "json"
    XLXS = "xlsx"


@dataclass
class ReaderFilesHelper:
    file_or_buffer: fb
    usecols: Union[List[str], List[int]] = None  # Use a subset of columns
    col_names: List[Union[str, int]] = None  # Give a name
    drop_cols: List[str] = None  # Drop a subset of columns
    format: SupportedFormats = None
    index_map: Union[None, Callable, dict, pd.Series] = None
    columns_map: Union[None, Callable, dict, pd.Series] = None
    additional_reader_kwargs: dict = field(default_factory=dict)
    file_or_buffer_type: Any = field(init=False)

    index_col: Union[int, str] = False
    header: Union[None, int, List[int]] = 0

    def __post_init__(self):
        if self.format is None:
            extension: str = get_extension(self.file_or_buffer)
            if extension == ".dxf":
                self.format = SupportedFormats.DXF
            elif extension == ".csv":
                self.format = SupportedFormats.CSV
            elif extension == ".json":
                self.format = SupportedFormats.JSON
            elif extension == ".xlsx":
                self.format = SupportedFormats.XLXS

        self.file_or_buffer_type = type(self.file_or_buffer)

    @property
    def pandas_reader_kwargs(self):
        attr_dict = {"names": self.col_names,
                     "header": self.header,
                     "index_col": self.index_col,
                     "usecols": self.usecols
                     }
        return {**attr_dict, **self.additional_reader_kwargs}

    @property
    def is_file_in_disk(self):
        return self.file_or_buffer_type == str or isinstance(self.file_or_buffer, pathlib.PurePath)

    @property
    def is_bytes_string(self):
        return self.file_or_buffer_type == io.BytesIO or self.file_or_buffer_type == io.StringIO

    @property
    def is_python_dict(self):
        return self.file_or_buffer_type == dict


@dataclass
class ReaderUnstructuredHelper:
    reader_vertex_args: ReaderFilesHelper
    reader_cells_args: ReaderFilesHelper = None
    reader_vertex_attr_args: ReaderFilesHelper = None
    reader_cells_attr_args: ReaderFilesHelper = None


@dataclass
class ReaderWellsHelper:
    reader_collars_args: ReaderFilesHelper
    reader_survey_args: ReaderFilesHelper
    reader_lith_args: ReaderFilesHelper = None
    reader_attr_args: List[ReaderFilesHelper] = None


@dataclass
class RawDataOptions:
    swap_yz_cells: bool = False


@dataclass(init=False)
class RawDataUnstructured:
    vertex: np.ndarray
    cells: Union[np.ndarray, Literal["lines", "points"]]
    cells_attr: Union[None, pd.DataFrame, Dict[str, xr.DataArray]] = None
    vertex_attr: Union[None, pd.DataFrame, Dict[str, xr.DataArray]] = None

    def swap_yz_col_cells(self):
        cells_aux = self.cells.copy()
        self.cells[:, 1] = cells_aux[:, 2]
        self.cells[:, 2] = cells_aux[:, 1]