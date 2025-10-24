from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union, TextIO
import numpy as np
import xarray as xr
from pathlib import Path

from ....core.structs import StructuredData


@dataclass
class GridDimensions:
    """
    Represents the dimensions of a 3D grid.

    Attributes:
        nx (int): Number of cells in the x-direction
        ny (int): Number of cells in the y-direction
        nz (int): Number of cells in the z-direction
    """
    nx: int
    ny: int
    nz: int


@dataclass
class GridOrigin:
    """
    Represents the origin point of a 3D grid.

    Attributes:
        x (float): X-coordinate of the origin
        y (float): Y-coordinate of the origin
        z (float): Z-coordinate of the origin
    """
    x: float
    y: float
    z: float


@dataclass
class GridCellSizes:
    """
    Represents the cell sizes in each direction of a 3D grid.

    Attributes:
        x (List[float]): Cell sizes in the x-direction
        y (List[float]): Cell sizes in the y-direction
        z (List[float]): Cell sizes in the z-direction
    """
    x: List[float]
    y: List[float]
    z: List[float]


@dataclass
class GridData:
    """
    Represents a 3D grid with dimensions, origin, and cell sizes.

    Attributes:
        dimensions (GridDimensions): The dimensions of the grid
        origin (GridOrigin): The origin point of the grid
        cell_sizes (GridCellSizes): The cell sizes in each direction
        metadata (Dict[str, Any]): Optional metadata about the grid
    """
    dimensions: GridDimensions
    origin: GridOrigin
    cell_sizes: GridCellSizes
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, grid_dict: Dict[str, Any]) -> 'GridData':
        """
        Converts a dictionary containing grid information into a GridData instance.

        Args:
            grid_dict: Dictionary with grid information

        Returns:
            GridData: A new GridData instance
        """
        dims = grid_dict["dimensions"]
        origin_dict = grid_dict["origin"]
        cell_sizes_dict = grid_dict["cell_sizes"]

        # Handle both new and legacy key names
        nx = dims.get("nx", dims.get("ne"))
        ny = dims.get("ny", dims.get("nn"))
        nz = dims.get("nz", dims.get("nz"))

        x = origin_dict.get("x", origin_dict.get("x0"))
        y = origin_dict.get("y", origin_dict.get("y0"))
        z = origin_dict.get("z", origin_dict.get("z0"))

        x_sizes = cell_sizes_dict.get("x", cell_sizes_dict.get("easting"))
        y_sizes = cell_sizes_dict.get("y", cell_sizes_dict.get("northing"))
        z_sizes = cell_sizes_dict.get("z", cell_sizes_dict.get("vertical"))

        metadata = grid_dict.get("metadata", {})

        return cls(
            dimensions=GridDimensions(nx=nx, ny=ny, nz=nz),
            origin=GridOrigin(x=x, y=y, z=z),
            cell_sizes=GridCellSizes(x=x_sizes, y=y_sizes, z=z_sizes),
            metadata=metadata
        )


from typing import Literal

def read_msh_structured_grid(grid_stream: TextIO, values_stream: TextIO, missing_value: Optional[float],
                             attr_name: Optional[str], ordering: Literal['ijk', 'xyz', 'xyz_reverse', 'yx-z'] = 'yx-z') -> StructuredData:
    """
    Read a structured grid mesh and values from streams and return a StructuredData object.

    This function is designed to work with streams (e.g., from Azure blob storage)
    rather than file paths.

    Args:
        grid_stream: TextIO stream containing the grid definition (.msh format)
        values_stream: TextIO stream containing the property values (.mod format)
        missing_value: Value to replace with NaN in the output array
        attr_name: Name for the data attribute
        ordering: Data ordering in the file:
                  - 'ijk': i (x) varies fastest, then j (y), then k (z)
                  - 'xyz': z varies fastest, then x, then y
                  - 'xyz_reverse': z varies fastest (reversed), then x, then y
                  - 'yx-z': y varies fastest, then x, then z (reversed) - Fortran order
                  Default is 'yx-z'.

    Returns:
        StructuredData object containing the grid and property values

    Raises:
        ValueError: If the stream format is invalid
    """
    # Read all lines from the grid stream
    lines = [line.strip() for line in grid_stream if line.strip()]

    # Create metadata for the grid
    metadata = {
            'file_format': 'grav3d',
            'source'     : 'stream'
    }

    # Parse grid information from lines
    try:
        grid = _parse_grid_from_lines(lines, metadata)
    except ValueError as e:
        # Add context about the stream to the error message
        raise ValueError(f"Error parsing grid stream: {e}") from e

    # Read values from the values stream
    try:
        # Read all values from the stream
        lines = [line.strip() for line in values_stream if line.strip()]

        model_array = _parse_mod_file(grid, lines, missing_value=missing_value, ordering=ordering)

    except Exception as e:
        # Add context to any errors
        raise ValueError(f"Error reading model stream: {str(e)}") from e

    # Create and return a StructuredData object
    return structured_data_from(model_array, grid, data_name=attr_name)


def read_msh_file(filepath: Union[str, Path]) -> GridData:
    """
    Read a structured grid mesh file and return a GridData object.

    Currently supports Grav3D mesh file format (.msh):
    - First line: NX NY NZ (number of cells in X, Y, Z directions)
    - Second line: X Y Z (coordinates of origin in meters)
    - Next section: X cell widths (either expanded or using N*value notation)
    - Next section: Y cell widths (either expanded or using N*value notation)
    - Next section: Z cell thicknesses (either expanded or using N*value notation)

    Args:
        filepath: Path to the mesh file

    Returns:
        GridData object containing the mesh information

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    metadata = {
            'file_format': 'grav3d',
            'filepath'   : str(filepath)
    }

    try:
        return _parse_grid_from_lines(lines, metadata)
    except ValueError as e:
        # Add context about the file to the error message
        raise ValueError(f"Error parsing mesh file {filepath}: {e}") from e


def read_mod_file(filepath: Union[str, Path], grid: GridData,
                  missing_value: float = -99_999.0,
                  ordering: Literal['ijk', 'xyz', 'xyz_reverse', 'yx-z'] = 'yx-z') -> np.ndarray:
    """
    Read a model file containing property values for a 3D grid.

    Currently supports Grav3D model file format (.mod) where each line contains
    a single property value.

    Args:
        filepath: Path to the model file
        grid: GridData object containing the grid dimensions
        missing_value: Value to replace with NaN in the output array (default: -99_999.0)
        ordering: Data ordering in the file. Options:
                  - 'ijk': i (x) varies fastest, then j (y), then k (z) - standard VTK/Fortran ordering
                  - 'xyz': z varies fastest, then x, then y - legacy Grav3D ordering
                  - 'xyz_reverse': z varies fastest (reversed direction), then x, then y
                  - 'yx-z': y varies fastest, then x, then z (reversed) 
                  Default is 'yx-z'.

    Returns:
        3D numpy array of property values with shape (ny, nx, nz)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the number of values doesn't match the grid dimensions
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        # Read all values from the file
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        model_array = _parse_mod_file(grid, lines, missing_value, ordering)

        return model_array

    except Exception as e:
        # Add context to any errors
        raise ValueError(f"Error reading model file {filepath}: {str(e)}") from e


def structured_data_from(array: np.ndarray, grid: GridData,
                         data_name: str = 'model') -> StructuredData:
    """
    Convert a 3D numpy array and grid information into a StructuredData object.

    Args:
        array: 3D numpy array of property values with shape (ny, nx, nz)
        grid: GridData object containing grid dimensions, origin, and cell sizes
        data_name: Name for the data array (default: 'model')

    Returns:
        StructuredData object containing the data array with proper coordinates

    Raises:
        ValueError: If array shape doesn't match grid dimensions
    """
    # Verify array shape matches grid dimensions
    expected_shape = (grid.dimensions.ny, grid.dimensions.nx, grid.dimensions.nz)
    if array.shape != expected_shape:
        raise ValueError(
            f"Array shape {array.shape} doesn't match grid dimensions {expected_shape}"
        )

    # Calculate cell center coordinates
    centers = _calculate_cell_centers(grid)

    # Create the xarray DataArray with proper coordinates
    xr_data_array = xr.DataArray(
        data=array,
        dims=['y', 'x', 'z'],  # Dimensions in the order they appear in the array
        coords={
                'x': centers['x'],
                'y': centers['y'],
                'z': centers['z'],
        },
        name=data_name,
        attrs=grid.metadata  # Include grid metadata in the data array
    )

    # Create a StructuredData instance from the xarray DataArray
    struct = StructuredData.from_data_array(
        data_array=xr_data_array,
        data_array_name=data_name
    )

    return struct


def _parse_grid_from_lines(lines: List[str], metadata: Dict[str, Any] = None) -> GridData:
    """
    Parse grid information from a list of lines.

    Args:
        lines: List of lines containing grid information
        metadata: Optional metadata to include in the GridData object

    Returns:
        GridData object containing the parsed grid information

    Raises:
        ValueError: If the lines format is invalid
    """
    if len(lines) < 2:
        raise ValueError("Invalid format: insufficient data")

    # Parse dimensions (first line)
    try:
        dims = lines[0].split()
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid dimensions: {e}")

    # Parse origin coordinates (second line)
    try:
        origin = lines[1].split()
        x, y, z = float(origin[0]), float(origin[1]), float(origin[2])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid origin: {e}")

    # Parse cell sizes
    try:
        current_line = 2
        x_sizes, current_line = _parse_cell_sizes(lines, current_line, nx)
        y_sizes, current_line = _parse_cell_sizes(lines, current_line, ny)
        z_sizes, _ = _parse_cell_sizes(lines, current_line, nz)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing cell sizes: {e}")

    # Create a GridData object with the parsed information
    grid_data_dict = {
            'dimensions': {'nx': nx, 'ny': ny, 'nz': nz},
            'origin'    : {'x': x, 'y': y, 'z': z},
            'cell_sizes': {
                    'x': x_sizes,
                    'y': y_sizes,
                    'z': z_sizes
            },
            'metadata'  : metadata or {}
    }

    return GridData.from_dict(grid_data_dict)


def _parse_cell_sizes(lines: List[str], start_index: int, count: int) -> Tuple[List[float], int]:
    """
    Parse cell sizes from file lines, handling both compact (N*value) and expanded notation.

    Args:
        lines: List of lines from the file
        start_index: Index to start parsing from
        count: Number of values to parse

    Returns:
        Tuple containing:
            - List of parsed values
            - Next line index after parsing
    """
    line = lines[start_index]

    # Check for compact notation (N*value)
    if '*' in line:
        parts = line.split('*')
        repetition = int(parts[0])
        value = float(parts[1])
        values = [value] * repetition
        return values, start_index + 1

    # Handle expanded notation across multiple lines
    values = []
    line_index = start_index

    while len(values) < count and line_index < len(lines):
        current_line = lines[line_index]

        # If we encounter a line with compact notation while parsing expanded, 
        # it's likely the next section
        if '*' in current_line and len(values) > 0:
            break

        # Add all numbers from the current line
        values.extend([float(x) for x in current_line.split()])
        line_index += 1

    # Take only the required number of values
    return values[:count], line_index


def _calculate_cell_centers(grid: GridData) -> Dict[str, np.ndarray]:
    """
    Calculate the center coordinates of each cell in the grid.

    Args:
        grid: GridData object containing grid dimensions, origin, and cell sizes

    Returns:
        Dictionary with 'x', 'y', and 'z' keys containing arrays of cell center coordinates
    """
    # Convert cell sizes to numpy arrays for vectorized operations
    x_sizes = np.array(grid.cell_sizes.x)
    y_sizes = np.array(grid.cell_sizes.y)
    z_sizes = np.array(grid.cell_sizes.z)

    # Calculate cell centers by adding cumulative sizes and offsetting by half the first cell size
    x_centers = grid.origin.x + np.cumsum(x_sizes) - x_sizes[0] / 2
    y_centers = grid.origin.y + np.cumsum(y_sizes) - y_sizes[0] / 2

    # For z, cells typically extend downward from the origin
    z_centers = grid.origin.z - (np.cumsum(z_sizes) - z_sizes[0] / 2)

    return {
            'x': x_centers,
            'y': y_centers,
            'z': z_centers
    }


def _parse_mod_file(grid: GridData, lines: List[str], missing_value: Optional[float],
                   ordering: Literal['ijk', 'xyz', 'xyz_reverse', 'yx-z'] = 'yx-z') -> np.ndarray:
    """
    Parse model file values into a 3D numpy array.

    Args:
        grid: GridData object containing grid dimensions
        lines: List of lines containing the values
        missing_value: Value to replace with NaN
        ordering: Data ordering in the file:
                  - 'ijk': i (x) varies fastest, then j (y), then k (z)
                  - 'xyz': z varies fastest, then x, then y
                  - 'xyz_reverse': z varies fastest (reversed), then x, then y
                  - 'yx-z': y varies fastest, then x, then z (reversed) 

    Returns:
        3D numpy array with shape (ny, nx, nz)
    """
    # Convert each line to a float
    values = np.array([float(line) for line in lines], dtype=float)
    
    # Calculate expected number of values based on grid dimensions
    nx, ny, nz = grid.dimensions.nx, grid.dimensions.ny, grid.dimensions.nz
    expected_count = nx * ny * nz
    
    if len(values) != expected_count:
        raise ValueError(
            f"Invalid model file: expected {expected_count} values, got {len(values)}"
        )
    
    # Reshape based on ordering
    if ordering == 'ijk':
        # i (x) varies fastest, then j (y), then k (z)
        # This is standard VTK/Fortran ordering: (k, j, i) in array dimensions
        model_array = values.reshape((nz, ny, nx), order='C')
        # Transpose to (ny, nx, nz) to match expected output shape
        model_array = np.transpose(model_array, (1, 2, 0))
    elif ordering == 'xyz':
        # z varies fastest, then x, then y (legacy Grav3D ordering)
        model_array = values.reshape((ny, nx, nz))
    elif ordering == 'xyz_reverse':
        # z varies fastest (but in reverse direction), then x, then y
        model_array = values.reshape((ny, nx, nz))
        # Reverse the z-axis (last dimension)
        model_array = np.flip(model_array, axis=2)
    elif ordering == 'yx-z':
        model_array = values.reshape((nz, nx, ny), order='F')
        model_array = np.transpose(model_array, (1, 2, 0))[:, :, ::-1]
    else:
        raise ValueError(f"Invalid ordering: {ordering}. Must be 'ijk', 'xyz', 'xyz_reverse', or 'yx-z'")
    
    # Replace missing values with NaN
    if missing_value is not None:
        model_array[model_array == missing_value] = np.nan
    
    return model_array
