from dataclasses import dataclass
from typing import List


@dataclass
class Dimensions:
    ne: int
    nn: int
    nz: int


@dataclass
class Origin:
    x0: float
    y0: float
    z0: float


@dataclass
class CellSizes:
    easting: List[float]
    northing: List[float]
    vertical: List[float]


@dataclass
class MeshData:
    dimensions: Dimensions
    origin: Origin
    cell_sizes: CellSizes

    @classmethod
    def from_dict(cls, mesh_dict):
        """
        Converts a dictionary containing mesh information into a MeshData data class.
        """
        dims = mesh_dict["dimensions"]
        origin_dict = mesh_dict["origin"]
        cell_sizes_dict = mesh_dict["cell_sizes"]

        return MeshData(
            dimensions=Dimensions(ne=dims["ne"], nn=dims["nn"], nz=dims["nz"]),
            origin=Origin(x0=origin_dict["x0"], y0=origin_dict["y0"], z0=origin_dict["z0"]),
            cell_sizes=CellSizes(
                easting=cell_sizes_dict["easting"],
                northing=cell_sizes_dict["northing"],
                vertical=cell_sizes_dict["vertical"]
            )
        )


def read_msh_file(filepath) -> MeshData:
    """
    Read a Grav3D mesh file (.msh) and return its contents as a structured dictionary.

    The mesh file format is:
    - First line: NE NN NZ (number of cells in East, North, Vertical directions)
    - Second line: X0 Y0 Z0 (coordinates of southwest top corner in meters)
    - Next section: East cell widths (either expanded or using N*value notation)
    - Next section: North cell widths (either expanded or using N*value notation)
    - Next section: Vertical cell thicknesses (either expanded or using N*value notation)

    Parameters:
    -----------
    filepath : str or Path
        Path to the .msh file

    Returns:
    --------
    dict
        Dictionary containing the mesh information
    """

    """
    This is in the msh file
    100 139 46
    313375 6072925 400
    25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25
    25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25
    25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Parse dimensions (first line)
    dims = lines[0].split()
    ne, nn, nz = int(dims[0]), int(dims[1]), int(dims[2])

    # Parse origin coordinates (second line)
    origin = lines[1].split()
    x0, y0, z0 = float(origin[0]), float(origin[1]), float(origin[2])

    # Helper function to parse cell sizes with different notations
    def parse_cell_sizes(start_index, count):
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

    # Parse easting cell widths
    current_line = 2
    e_widths, current_line = parse_cell_sizes(current_line, ne)

    # Parse northing cell widths
    n_widths, current_line = parse_cell_sizes(current_line, nn)

    # Parse vertical cell thicknesses
    z_thicknesses, _ = parse_cell_sizes(current_line, nz)

    # Create a dictionary with all the parsed information
    mesh_data_dict = {
            'dimensions': {'ne': ne, 'nn': nn, 'nz': nz},
            'origin'    : {'x0': x0, 'y0': y0, 'z0': z0},
            'cell_sizes': {
                    'easting' : e_widths,
                    'northing': n_widths,
                    'vertical': z_thicknesses
            }
    }

    mesh_data = MeshData.from_dict(mesh_data_dict)

    return mesh_data
import numpy as np

def read_mod_file(filepath: str, mesh: MeshData) -> np.ndarray:
    """
    Reads a Grav3D model file (.mod) and validates its shape against the provided mesh.
    
    The file should contain NN * NE * NZ lines where each line is a cell property value.
    The values are in the order:
      - First in the z-direction (top-to-bottom)
      - Then in the easting
      - Finally in the northing
    This function reshapes the data into an array of shape (NN, NE, NZ) using row-major order.
    
    Parameters:
    -----------
    filepath : str
        Path to the .mod file.
    mesh : MeshData
        Mesh information with dimensions (ne, nn, nz).
    
    Returns:
    --------
    np.ndarray
        Three-dimensional array of model values with shape (nn, ne, nz).
    
    Raises:
    -------
    ValueError
        If the total number of values does not equal NN * NE * NZ.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Convert each line to a float
    values = [float(line) for line in lines]
    
    # Expected total number of values using mesh dimensions. Note that mesh.dimensions is stored as (ne, nn, nz)
    expected_count = mesh.dimensions.nn * mesh.dimensions.ne * mesh.dimensions.nz
    if len(values) != expected_count:
        raise ValueError(f"Invalid .mod file: expected {expected_count} values, got {len(values)}")
    
    # Reshape to (nn, ne, nz) so that the fastest changing axis is the vertical (z-direction)
    model_array = np.array(values, dtype=float).reshape(
        (mesh.dimensions.nn, mesh.dimensions.ne, mesh.dimensions.nz)
    )
    
    # -9999 to nan
    model_array[model_array == -99_999.0] = np.nan
    return model_array