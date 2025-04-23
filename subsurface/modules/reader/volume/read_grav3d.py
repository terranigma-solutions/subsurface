from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union
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


def parse_cell_sizes(lines: List[str], start_index: int, count: int) -> Tuple[List[float], int]:
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

    if len(lines) < 2:
        raise ValueError(f"Invalid mesh file format: {filepath}")

    # Parse dimensions (first line)
    try:
        dims = lines[0].split()
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid dimensions in mesh file: {e}")

    # Parse origin coordinates (second line)
    try:
        origin = lines[1].split()
        x, y, z = float(origin[0]), float(origin[1]), float(origin[2])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid origin in mesh file: {e}")

    # Parse cell sizes
    try:
        current_line = 2
        x_sizes, current_line = parse_cell_sizes(lines, current_line, nx)
        y_sizes, current_line = parse_cell_sizes(lines, current_line, ny)
        z_sizes, _ = parse_cell_sizes(lines, current_line, nz)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing cell sizes: {e}")

    # Create a dictionary with all the parsed information
    grid_data_dict = {
            'dimensions': {'nx': nx, 'ny': ny, 'nz': nz},
            'origin'    : {'x': x, 'y': y, 'z': z},
            'cell_sizes': {
                    'x': x_sizes,
                    'y': y_sizes,
                    'z': z_sizes
            },
            'metadata'  : {
                    'file_format': 'grav3d',
                    'filepath'   : str(filepath)
            }
    }

    return GridData.from_dict(grid_data_dict)


def calculate_cell_centers(grid: GridData) -> Dict[str, np.ndarray]:
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
    centers = calculate_cell_centers(grid)

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


def read_mod_file(filepath: Union[str, Path], grid: GridData,
                  missing_value: float = -99_999.0) -> np.ndarray:
    """
    Read a model file containing property values for a 3D grid.

    Currently supports Grav3D model file format (.mod) where each line contains
    a single property value. The values are ordered with the z-direction changing
    fastest, then x, then y.

    Args:
        filepath: Path to the model file
        grid: GridData object containing the grid dimensions
        missing_value: Value to replace with NaN in the output array (default: -99_999.0)

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

        # Convert each line to a float
        values = np.array([float(line) for line in lines], dtype=float)

        # Calculate expected number of values based on grid dimensions
        nx, ny, nz = grid.dimensions.nx, grid.dimensions.ny, grid.dimensions.nz
        expected_count = nx * ny * nz

        if len(values) != expected_count:
            raise ValueError(
                f"Invalid model file: expected {expected_count} values, got {len(values)}"
            )

        # Reshape to (ny, nx, nz) with z changing fastest
        model_array = values.reshape((ny, nx, nz))

        # Replace missing values with NaN
        if missing_value is not None:
            model_array[model_array == missing_value] = np.nan

        return model_array

    except Exception as e:
        # Add context to any errors
        raise ValueError(f"Error reading model file {filepath}: {str(e)}") from e
