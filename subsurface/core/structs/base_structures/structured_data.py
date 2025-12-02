import enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Literal

import numpy as np
import xarray as xr

from ....optional_requirements import require_pyvista


class StructuredDataType(enum.Enum):
    REGULAR_AXIS_ALIGNED = 0  #: Regular axis aligned grid. Distance between consecutive points is constant
    REGULAR_AXIS_UNALIGNED = 1  #: Regular axis unaligned grid. Distance between consecutive points is constant
    IRREGULAR_AXIS_ALIGNED = 2  #: Irregular axis aligned grid. Distance between consecutive points is not constant
    IRREGULAR_AXIS_UNALIGNED = 3  #: Irregular axis unaligned grid. Distance between consecutive points is not constant

    # [CLN] This terminology looks odd to me.
    # "Uniform" vs. "non-uniform" is what PyVista uses instead of "regular" vs. "irregular".
    # "Rectilinear" vs. "curvilinear" is what Pyvista uses instead of "axis-aligned" vs "axis-unaligned".
    # As of right now it's not clear to me that anything other than REGULAR_AXIS_ALIGNED is actually valid in this file.

@dataclass(frozen=False)
class StructuredData:
    data: xr.Dataset
    _active_data_array_name: str = "data_array"
    type: StructuredDataType = StructuredDataType.REGULAR_AXIS_ALIGNED
    dtype: Literal["float32", "float64"] = "float32"

    """Primary structure definition for structured data

       Check out other constructors: `StructuredData.from_numpy`,
        `StructuredData.from_data_array` and `StructuredData.from_dict`

    Args:
        data (xr.Dataset): object containing
         structured data, i.e. data that can be stored in multidimensional
         numpy array. The preferred type to pass as data is directly a
         xr.Dataset to be sure all the attributes are set and named as the user
         wants.
        data_array_name (str): If data is a numpy array or xarray DataArray, data_name
         provides the name for the xarray data variable
     
    Attributes:
        data (xarray.Dataset)
    """

    @property
    def active_data_array_name(self):
        data_var_list = list(self.data.data_vars.keys())
        if self._active_data_array_name not in data_var_list:
            raise ValueError("data_array_name not found in data_vars: {}".format(data_var_list))
        return self._active_data_array_name

    @active_data_array_name.setter
    def active_data_array_name(self, data_array_name: str):
        self._active_data_array_name = data_array_name

    @classmethod
    def from_numpy(cls, array: np.ndarray, coords: dict = None, data_array_name: str = "data_array",
                   dim_names: List[str] = None):
        if dim_names is None:
            dim_names = cls._default_dim_names(array.ndim)
        # if they are more than 3 we do not know the dimension name but it should valid:

        dataset: xr.Dataset = xr.Dataset(
            data_vars=
            {
                    data_array_name: (dim_names, array)
            },
            coords=coords
        )

        return cls(dataset, data_array_name)

    @classmethod
    def from_data_array(cls, data_array: xr.DataArray, data_array_name: str = "data_array"):
        dataset: xr.Dataset = xr.Dataset(
            data_vars={
                    data_array_name: data_array
            },
            coords=data_array.coords
        )

        return cls(dataset, data_array_name)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, xr.DataArray], coords: Dict[str, str] = None, data_array_name: str = "data_array"):
        dataset: xr.Dataset = xr.Dataset(data_vars=data_dict, coords=coords)
        return cls(dataset, data_array_name)

    @classmethod
    def from_pyvista(
            cls,
            pyvista_object: 'pyvista.DataSet',
            data_array_name: str = "data_array"
    ):
        pyvista = require_pyvista()

        def rectilinear_is_uniform(
                rectilinear_grid: pyvista.RectilinearGrid,
                relative_tolerance: float = 1e-6,
                absolute_tolerance: float = 1e-12,
            ) -> bool:

            def axis_is_uniform(v: np.ndarray) -> bool:
                v = np.asarray(v, dtype=float)
                if v.size <= 2:
                    # 0, 1 or 2 points → treat as uniform for our purposes
                    return True
                diffs = np.diff(v)
                first = diffs[0]
                return np.allclose(diffs, first, rtol=relative_tolerance, atol=absolute_tolerance)

            return (axis_is_uniform(rectilinear_grid.x)
                and axis_is_uniform(rectilinear_grid.y)
                and axis_is_uniform(rectilinear_grid.z))

        extended_help_message = "Only uniform rectilinear grids are currently supported. The VTK format is developed by KitWare and you can use their free software ParaView to further inspect your file. In ParaView, in Information > Data Statistics, the Type must be Image (Uniform Rectilinear Grid). Furthermore, you can use ParaView to interpolate your data on to a uniform rectilinear grid and to export it as Image type."

        match pyvista_object:
            case pyvista.UnstructuredGrid():
                # In a previous version of Subsurface there was an ill-formed attempt at supporting some unstructured grids here.
                # I've left this function to minimize downstream changes and also in case we decide to revive anything in that direction.
                raise ValueError(f"Cannot generally convert unstructured grids to structured grids. {extended_help_message}")
            case pyvista.ImageData():
                pass
            case pyvista.RectilinearGrid() as rectilinear:
                if not rectilinear_is_uniform(rectilinear):
                    raise NotImplementedError(f"Non-uniform rectilinear grid conversion is not yet implemented. {extended_help_message}")
            case _:
                raise ValueError(f"Unexpected VTK grid type. {extended_help_message}")

        grid = pyvista_object.cast_to_structured_grid()

        # Extract p

        # Extract cell data and point data (if any)
        data_vars = {}

        # TODO: I need to do something with the bounds

        dimensions = np.array(grid.dimensions) - 1
        default_dim_names = cls._default_dim_names(dimensions.shape[0])

        bounds: tuple = grid.bounds
        coords = {}
        for i, dim in enumerate(default_dim_names):
            coords[dim] = np.linspace(
                start=bounds[i * 2],
                stop=bounds[i * 2 + 1],
                num=dimensions[i],
                endpoint=False
            )

        for name in grid.cell_data:
            cell_attr_data: pyvista.pyvista_ndarray = grid[name]
            cell_attr_data_reshaped = cell_attr_data.reshape(dimensions, order='F')

            data_vars[name] = xr.DataArray(
                data=cell_attr_data_reshaped,
                dims=default_dim_names,
                name=name
            )

        dataset: xr.Dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )
        return cls(dataset, data_array_name)

    @property
    def values(self) -> np.ndarray:
        return self.data[self.active_data_array_name].values

    _bounds: Tuple[float, float, float, float, float, float] = None

    @property
    def bounds(self):
        if self._bounds is not None:
            return self._bounds

        array_: xr.DataArray = self.data[self.active_data_array_name]
        bounds = self._get_bounds_from_coord(array_)
        return bounds

    @bounds.setter
    def bounds(self, bounds: Tuple[float, float, float, float, float, float]):
        """
        Set the bounds of the structured data. This is useful for defining the
        spatial extent of the data in a structured grid.

        Args:
            bounds (Tuple[float, float, float, float, float, float]): A tuple containing
                the minimum and maximum values for each dimension (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        self._bounds = bounds

    @property
    def shape(self):
        return self.active_data_array.shape

    @property
    def active_data_array(self):
        return self.data[self.active_data_array_name]

    @staticmethod
    def _get_bounds_from_coord(xr_obj: xr.DataArray):
        bounds = {}
        for coord in xr_obj.coords:
            bounds[coord] = (xr_obj[coord].min().item(), xr_obj[coord].max().item())
        return bounds

    def default_data_array_to_binary_legacy(self, order: Literal["K", "A", "C", "F"] = 'F'):
        bytearray_le = self._to_bytearray(order=order)
        header = self._set_binary_header()

        return bytearray_le, header

    def to_binary(self, order: Literal["K", "A", "C", "F"] = 'F') -> bytes:
        """Converts the structured data to a binary file
        
        Notes: 
            Only the active data array is converted to binary for now 
        """

        body_ = self._to_bytearray(order)
        header = self._set_binary_header()

        import json
        header_json = json.dumps(header)
        header_json_bytes = header_json.encode('utf-8')
        header_json_length = len(header_json_bytes)
        header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
        file = header_json_length_bytes + header_json_bytes + body_
        return file

    def _set_binary_header(self) -> Dict:
        data_array = self.active_data_array

        match self.type:
            case StructuredDataType.REGULAR_AXIS_ALIGNED:
                header = {
                        "data_shape": self.shape,
                        "bounds"    : self.bounds,
                        "transform" : None,
                        "dtype"     : self.dtype,
                        "data_name" : self.active_data_array_name
                }
            case _:
                raise NotImplementedError(f"StructuredDataType {self.type} not implemented yet")

        return header

    def _to_bytearray(self, order: Literal["K", "A", "C", "F"]) -> bytes:
        data_array = self.active_data_array

        data = data_array.values.astype(self.dtype).tobytes(order)
        bytearray_le = data
        return bytearray_le

    @classmethod
    def _default_dim_names(cls, n_dims: int):
        if n_dims == 2:
            dim_names = ['x', 'y']
        elif n_dims == 3:
            dim_names = ['x', 'y', 'z']
        else:
            dim_names = ['dim' + str(i) for i in range(n_dims)]
        return dim_names

    def to_netcdf(self, path: str, **to_netcdf_kwargs):
        """
        Serializes the current StructuredData instance to a NetCDF file.

        Args:
            path (str): The path (including file name) where the NetCDF file will be saved.
            **to_netcdf_kwargs: Additional keyword arguments forwarded to xarray's `to_netcdf`.
        """
        # Copy the dataset (shallow copy of the data structure, no copying of the underlying arrays)
        ds = self.data.copy(deep=False)

        # Store relevant metadata as global attributes:
        ds.attrs["active_data_array_name"] = self._active_data_array_name
        ds.attrs["structured_data_type"] = self.type.name  # e.g., "REGULAR_AXIS_ALIGNED"
        ds.attrs["dtype"] = self.dtype  # e.g., "float32"

        # Use xarray's to_netcdf
        ds.to_netcdf(path, **to_netcdf_kwargs)

    @classmethod
    def from_netcdf(cls, path: str, **from_netcdf_kwargs):
        """
        Deserializes a NetCDF file into a StructuredData instance.

        Args:
            path (str): The path to the NetCDF file to read.
            **from_netcdf_kwargs: Additional keyword arguments forwarded to xarray's `open_dataset`.

        Returns:
            StructuredData: A new instance of StructuredData loaded from the file.
        """
        ds = xr.open_dataset(path, **from_netcdf_kwargs)

        # Retrieve what was stored in attrs (with defaults if missing)
        data_array_name = ds.attrs.get("active_data_array_name", "data_array")
        dtype_str: str = ds.attrs.get("dtype", "float32")
        if dtype_str not in ["float32", "float64"]:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        sdt_str = ds.attrs.get("structured_data_type", "REGULAR_AXIS_ALIGNED")

        # Convert strings back to your enum or any other type
        # (assuming StructuredDataType is an Enum where name matches sdt_str)
        if sdt_str not in StructuredDataType.__members__:
            raise ValueError(f"Unsupported structured_data_type: {sdt_str}")
        structured_data_type: StructuredDataType = StructuredDataType[sdt_str]

        return cls(
            data=ds,
            _active_data_array_name=data_array_name,
            type=structured_data_type,
            dtype=dtype_str
        )
