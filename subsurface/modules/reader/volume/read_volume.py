import os
from io import BytesIO
from typing import Union

from subsurface.core.structs import StructuredData

from .... import optional_requirements
from ....core.structs import UnstructuredData
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
import numpy as np
import pandas as pd


def read_VTK_structured_grid(file_or_buffer: Union[str, BytesIO], active_scalars: str) -> StructuredData:
    pv = optional_requirements.require_pyvista()

    if isinstance(file_or_buffer, BytesIO):
        # If file_or_buffer is a BytesIO, write it to a temporary file
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile('wb', suffix='.vtk', delete=False) as temp_file:
            # Write the BytesIO content to the temporary file
            getvalue: bytes = file_or_buffer.getvalue()
            temp_file.write(getvalue)
            temp_file.flush()  # Make sure all data is written
            temp_file_name = temp_file.name  # Store the temporary file name
        try:
            # Use pyvista.read() to read from the temporary file
            pyvista_obj = pv.read(temp_file_name)
        finally:
            # Ensure the temporary file is deleted after reading
            os.remove(temp_file_name)
    else:
        # If it's a file path, read directly
        pyvista_obj = pv.read(file_or_buffer)
    try:
        pyvista_struct: pv.StructuredGrid = pv_cast_to_structured_grid(pyvista_obj)
    except Exception as e:
        raise ValueError(f"Failed to convert to StructuredGrid: {e}")

    if PLOT := False:
        pyvista_struct.set_active_scalars(active_scalars)
        pyvista_struct.plot()

    struct: StructuredData = StructuredData.from_pyvista_structured_grid(
        grid=pyvista_struct,
        data_array_name=active_scalars
    )

    return struct


def read_volumetric_mesh_to_subsurface(reader_helper_coord: GenericReaderFilesHelper,
                                       reader_helper_attr: GenericReaderFilesHelper) -> UnstructuredData:
    df_coord = read_volumetric_mesh_coord_file(reader_helper_coord)
    if len(df_coord.columns) == 1:
        raise ValueError(
            "The attributes file has only one column, probably the columns are not being separated correctly. Use 'sep' in Additional Reader Arguments"
        )

    df_attr = read_volumetric_mesh_attr_file(reader_helper_attr)
    # Check if there are more than one column and if it is only one raise an error that probably the columns have not been properly separated. Use "sep" in Additional Reader Arguments
    if len(df_attr.columns) == 1:
        raise ValueError(
            "The attributes file has only one column, probably the columns are not being separated correctly. Use 'sep' in Additional Reader Arguments"
        )
    
    combined_df = df_coord.merge(df_attr, left_index=True, right_index=True)
    ud = UnstructuredData.from_array(
        vertex=combined_df[['x', 'y', 'z']], cells="points",
        attributes=combined_df[['pres', 'temp', 'sg', 'xco2']]
    )
    return ud


def read_volumetric_mesh_coord_file(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    df = pd.read_csv(
        filepath_or_buffer=reader_helper.file_or_buffer,
        **reader_helper.pandas_reader_kwargs
    )
    if reader_helper.columns_map is not None:
        df.rename(
            mapper=reader_helper.columns_map,
            axis="columns",
            inplace=True
        )

    df.dropna(axis=0, inplace=True)

    df.x = df.x.astype(float)
    df.y = df.y.astype(float)
    df.z = df.z.astype(float)
    # Throw error if empty
    if df.empty:
        raise ValueError("The file is empty")
    
    return df


def read_volumetric_mesh_attr_file(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    df = pd.read_table(reader_helper.file_or_buffer, **reader_helper.pandas_reader_kwargs)
    df.columns = df.columns.astype(str).str.strip()
    return df


def pv_cast_to_structured_grid(pyvista_object: 'pv.DataSet') -> 'pv.StructuredGrid':
    pv = optional_requirements.require_pyvista()

    match pyvista_object:
        case pv.UnstructuredGrid():
            # In a previous version of Subsurface there was an ill-formed attempt at supporting some unstructured grids here.
            # I've left this function to minimize downstream changes and also in case we decide to revive anything in that direction.
            raise ValueError("Cannot generally convert unstructured grids to structured grids.")
        case _:
            return pyvista_object.cast_to_structured_grid()
