from io import BytesIO
from typing import Union

from subsurface.core.structs import StructuredData

from .... import optional_requirements
from ....core.structs import UnstructuredData
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
import pandas as pd


def read_VTK_structured_grid(file_or_buffer: Union[str, BytesIO]) -> StructuredData:
    pv = optional_requirements.require_pyvista()

    reader = pv.get_reader("vtk")
    pyvista_obj: pv.DataSet = pv.read(file_or_buffer)


    try:
        pyvista_struct: pv.ExplicitStructuredGrid = pyvista_obj.cast_to_explicit_structured_grid()
    except Exception as e:
        raise f"The file is not a structured grid: {e}"

    active_scalars = "Cell Number"

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
    df_attr = read_volumetric_mesh_attr_file(reader_helper_attr)
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
    return df


def read_volumetric_mesh_attr_file(reader_helper: GenericReaderFilesHelper) -> pd.DataFrame:
    df = pd.read_table(reader_helper.file_or_buffer, **reader_helper.pandas_reader_kwargs)
    df.columns = df.columns.str.strip()
    return df
