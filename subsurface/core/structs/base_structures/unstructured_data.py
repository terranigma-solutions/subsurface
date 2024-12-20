from dataclasses import dataclass
from typing import Union, Dict, Mapping, Hashable, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from subsurface.core.structs.base_structures._unstructured_data_constructor import vertex_and_cells_arrays_to_data_array, raw_attributes_to_dict_data_arrays
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase


@dataclass(frozen=False)
class UnstructuredData:
    data: xr.Dataset
    cells_attr_name: str = "cell_attrs"
    vertex_attr_name: str = "vertex_attrs"

    """Primary structure definition for unstructured data

    Attributes:
        data (`xarray.Dataset`): Data structure where we store

    Args:

        ds (xarray.Dataset): Directly a dataset with the expected structured. This
         arg is specially thought for loading data from disk

    Notes:
        Depending on the shape of `edge` the following unstructured elements can
        be created:

        - cells NDArray[(Any, 0), IntX] or NDArray[(Any, 1), IntX] -> *Point cloud*.
          E.g. Outcrop scan with lidar
        - cells NDArray[(Any, 2), IntX] -> *Lines*. E.g. Borehole
        - cells NDArray[(Any, 3), IntX] -> *Mesh*. E.g surface-DEM Topography
        - cells NDArray[(Any, 4), IntX]
           - -> *tetrahedron*
           - -> *quadrilateral (or tetragon)* UNSUPPORTED?
        - cells NDArray[(Any, 8), IntX] -> *Hexahedron: Unstructured grid/Prisms*
    """

    def __post_init__(self):
        self._validate()

    def __repr__(self):
        return self.data.__repr__()

    @classmethod
    def from_array(
            cls,
            vertex: np.ndarray,
            cells: Union[np.ndarray, Literal["lines", "points"], SpecialCellCase],
            *,
            cells_attr: Union[None, pd.DataFrame, Dict[str, xr.DataArray]] = None,
            vertex_attr: Union[None, pd.DataFrame, Dict[str, xr.DataArray]] = None,
            coords: Mapping[Hashable, Any] = None,
            xarray_attributes: Mapping[Hashable, Any] = None,
            default_cells_attr_name: str = "cell_attrs",
            default_points_attr_name: str = "vertex_attrs",
            attributes: Union[None, pd.DataFrame, Dict[str, xr.DataArray]] = None  # TODO Obsolete
    ):
        """ Constructor of UnstructuredData from arrays or pandas DataFrames.

        Args:
            vertex (np.ndarray): NDArray[(Any, 3), FloatX]: XYZ point data
            cells (Union[np.ndarray, Literal["lines", "points"]]): NDArray[(Any, ...), IntX]:
             Combination of vertex that create different geometric elements. If
             str use default values for either points or lines
            cells_attr (Union[None, pd.DataFrame, Dict[str, xr.DataArray]]: Number associated to an element
            vertex_attr (Union[None, pd.DataFrame, Dict[str, xr.DataArray]]: Number
             associated to points
            coords:
            xarray_attributes:
            attributes:
            default_cells_attr_name:
            default_points_attr_name:

        Returns:

        """
        if attributes is not None:
            cells_attr = attributes

        cells_data_array, n_cells, n_vertex, vertex_data_array = vertex_and_cells_arrays_to_data_array(
            cells=cells,
            vertex=vertex
        )
        points_attributes_xarray_dict: dict[str, xr.DataArray] = raw_attributes_to_dict_data_arrays(
            default_attributes_name=default_points_attr_name,
            n_items=n_vertex,
            dims=["points", "vertex_attr"],
            raw_attributes=vertex_attr
        )

        cells_attributes_xarray_dict: dict[str, xr.DataArray] = raw_attributes_to_dict_data_arrays(
            default_attributes_name=default_cells_attr_name,
            n_items=n_cells,
            dims=["cell", "cell_attr"],
            raw_attributes=cells_attr
        )

        xarray_dict = {
                "vertex": vertex_data_array,
                "cells" : cells_data_array,
                **cells_attributes_xarray_dict,
                **points_attributes_xarray_dict
        }

        return cls.from_data_arrays_dict(
            xarray_dict=xarray_dict,
            coords=coords,
            xarray_attributes=xarray_attributes,
            default_cells_attributes_name=default_cells_attr_name,
            default_points_attributes_name=default_points_attr_name
        )

    @classmethod
    def from_data_arrays_dict(
            cls,
            xarray_dict: Dict[str, xr.DataArray],
            coords: Mapping[Hashable, Any] = None,
            xarray_attributes: Mapping[Hashable, Any] = None,
            default_cells_attributes_name="cell_attrs",
            default_points_attributes_name="vertex_attrs"
    ):
        # TODO: xr.Dataset seems to have been changed with 2022.06. needs to be adapted for indexing
        ds = xr.Dataset(xarray_dict, coords=coords, attrs=xarray_attributes)

        # Try to unstack pandas dataframe if exist
        # TODO: This is an issue in wells. If it is only there maybe we should move it there
        try:
            ds = ds.reset_index('cell')
        except (KeyError, ValueError) as e:
            print(f"{e} xarray dataset must include 'cell' key (KeyError) or xarray 'cell' has no index (ValueError).")

        # Check that the Dataset data_vars names matches, "vertex", "cells", default_cells_attributes_name and default_points_attributes_name
        # and raise an error pointing out which one is missing
        _vars = [var in ds.data_vars for var in ["vertex", "cells", default_cells_attributes_name, default_points_attributes_name]]
        if not all(_vars):
            missing = ["vertex", "cells", default_cells_attributes_name, default_points_attributes_name]
            raise KeyError(f"DataArray must include the following keys: {missing}. For attributes you can"
                           f" change the default names with default_cells_attributes_name and default_points_attributes_name.")

        return cls(ds, default_cells_attributes_name, default_points_attributes_name)

    @classmethod
    def from_binary_le(cls, path: str):
        from ._liquid_earth_mesh import LiquidEarthMesh
        with open(path, 'rb') as f:
            bytes_data = f.read()
        mesh = LiquidEarthMesh.from_binary(bytes_data)
        unstruct = cls.from_array(
            vertex=mesh.vertex,
            cells=mesh.cells,
            cells_attr=mesh.attributes,
            vertex_attr=mesh.points_attributes,
            xarray_attributes=None
        )
        return unstruct

    @classmethod
    def from_binary_le_legacy(cls, path_to_binary: str, path_to_json: str):
        import json
        from ._liquid_earth_mesh import LiquidEarthMesh

        with open(path_to_binary, 'rb') as f:
            body_ = f.read()
        with open(path_to_json, 'r') as f:
            header_ = json.load(f)

        header_json = json.dumps(header_)
        header_json_bytes = header_json.encode('utf-8')
        header_json_length = len(header_json_bytes)
        header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
        file = header_json_length_bytes + header_json_bytes + body_
        mesh = LiquidEarthMesh.from_binary(file)
        unstruct = cls.from_array(
            vertex=mesh.vertex,
            cells=mesh.cells,
            cells_attr=mesh.attributes,
            vertex_attr=mesh.points_attributes,
            xarray_attributes=None
        )
        return unstruct

    @property
    def vertex(self) -> np.ndarray:
        return self.data['vertex'].values

    @property
    def cells(self):
        return self.data['cells'].values

    @property
    def attributes(self) -> pd.DataFrame:
        xarray = self.data[self.cells_attr_name]
        return xarray.to_dataframe()[self.cells_attr_name].unstack(level=1)

    @attributes.setter
    def attributes(self, dataframe):
        self.data[self.cells_attr_name] = xr.DataArray(dataframe, dims=['element', 'cell_attr'])

    @property
    def cell_attributes(self):
        return self.attributes

    @cell_attributes.setter
    def cell_attributes(self, dataframe):
        self.attributes = dataframe

    @property
    def points_attributes(self) -> pd.DataFrame:
        data_array: xr.DataArray = self.data[self.vertex_attr_name]
        dataframe: pd.DataFrame = data_array.to_dataframe()
        not_suer = dataframe[self.vertex_attr_name]
        unstack = not_suer.unstack(level=1)
        return unstack

    @points_attributes.setter
    def points_attributes(self, dataframe: pd.DataFrame):
        vertex_attr: xr.DataArray = self.data[self.vertex_attr_name]
        vertex_attr.values = dataframe.values

    @property
    def n_elements(self):
        return self.cells.shape[0]

    @property
    def n_vertex_per_element(self):
        return self.cells.shape[1]

    @property
    def n_points(self):
        return self.vertex.shape[0]

    @property
    def attributes_to_dict(
            self,
            orient: Literal["dict", "list", "series", "split", "tight", "index"] = "list"
    ):
        return self.attributes.to_dict(orient)

    @property
    def points_attributes_to_dict(
            self,
            orient: Literal["dict", "list", "series", "split", "tight", "index"] = "list"
    ):
        return self.points_attributes.to_dict(orient)

    @property
    def extent(self):
        max = self.vertex.max(axis=0)
        min = self.vertex.min(axis=0)
        extent = np.stack((min, max), axis=1).ravel()
        return extent

    def to_xarray(self):
        a = xr.DataArray(self.vertex, dims=['points', 'XYZ'])
        b = xr.DataArray(self.cells, dims=['cells', 'node'])
        e = xr.DataArray(self.attributes, dims=['element', 'cell_attr'])
        c = xr.Dataset({'v': a, 'e': b, 'a': e})
        return c

    def to_binary_legacy(self, order='F'):
        bytearray_le = self._to_bytearray(order)
        header = self._set_binary_header()
        return bytearray_le, header

    def to_binary(self, order='F') -> bytes:
        body_ = self._to_bytearray(order)
        header_ = self._set_binary_header()
        import json
        header_json = json.dumps(header_)
        header_json_bytes = header_json.encode('utf-8')
        header_json_length = len(header_json_bytes)
        header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
        file = header_json_length_bytes + header_json_bytes + body_
        return file

    def _set_binary_header(self):
        header = {
                "vertex_shape"     : self.vertex.shape,
                "cell_shape"       : self.cells.shape,
                "cell_attr_shape"  : self.attributes.shape,
                "vertex_attr_shape": self.points_attributes.shape,
                "cell_attr_names"  : self.attributes.columns.to_list(),
                "cell_attr_types"  : self.attributes.dtypes.astype(str).to_list(),
                "vertex_attr_names": self.points_attributes.columns.to_list(),
                "vertex_attr_types": self.attributes.dtypes.astype(str).to_list(),
                "xarray_attrs"     : self.data.attrs
        }
        return header

    def _to_bytearray(self, order):
        vertex = self.vertex.astype('float32').tobytes(order)
        cells = self.cells.astype('int32').tobytes(order)
        cell_attribute = self.attributes.values.astype('float32').tobytes(order)
        vertex_attribute = self.points_attributes.values.astype('float32').tobytes(order)
        bytearray_le = vertex + cells + cell_attribute + vertex_attribute
        return bytearray_le

    def _validate(self):
        try:
            _ = self.data[self.cells_attr_name]['cell']
            _ = self.data[self.cells_attr_name]['cell_attr']
        except KeyError:
            raise KeyError('Cell attribute DataArrays must contain dimension cell and cell_attr')
        try:
            _ = self.data[self.vertex_attr_name]['vertex_attr']
            _ = self.data[self.vertex_attr_name]['points']
        except KeyError:
            raise KeyError('Point attribute DataArrays must contain dimensions points and vertex_attr.')

        # Make sure the number of vertices matches the associated data.
        if self.data['cells']['cell'].size != self.data[self.cells_attr_name]['cell'].size:
            raise AttributeError('Attributes and cells must have the same length.')

        if self.n_points != self.data[self.vertex_attr_name]['points'].size:
            raise AttributeError('points_attributes and vertex must have the same length.')
