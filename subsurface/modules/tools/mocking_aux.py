import numpy as np
import pyvista as pv


def transform_gaussian_blur(grid, sigma=20.0):
    """
    Applies a Gaussian blur to the 'model_name' field of the structured grid.

    Parameters:
        grid - PyVista grid with 'model_name' field
        sigma - Standard deviation for the Gaussian kernel
    """
    from scipy.ndimage import gaussian_filter

    # Get the original dimensions of the grid
    dims = grid.dimensions

    # Reshape the data to 3D array matching grid dimensions
    values = np.array(grid['model_name'])
    values_3d = values.reshape(dims[2] - 1, dims[1] - 1, dims[0] - 1).transpose(2, 1, 0)

    # Apply Gaussian filter
    blurred_values = gaussian_filter(values_3d, sigma=sigma, axes=(2,))

    # Reshape back to 1D array
    grid['model_name'] = blurred_values.transpose(2, 1, 0).flatten()
    return grid


def transform_sinusoidal(values, amplitude=1.0, frequency=0.01, phase=0):
    """
    Apply a sinusoidal transformation to the values.
    """
    return values + amplitude * np.sin(frequency * values + phase)


def obfuscate_model_name(grid, transform_functions, attr):
    """
    Applies transformation functions to the 'model_name' field.
    Functions can operate on either the grid or the values array.
    """
    for func in transform_functions:
        if 'grid' in func.__code__.co_varnames:
            # Function expects the full grid
            grid = func(grid)
        else:
            # Function expects just the values array
            values = np.array(grid[attr])
            grid[attr] = func(values)

    return grid


# pyvista_struct = transform_xy_to_z_propagation(pyvista_struct, z_factor=0.3, noise_level=0.1)
def transform_subtract_mean(values):
    """
    Subtract the mean of the array from each element.
    """
    return values - np.mean(values)


def transform_scale(values, scale_factor=0.003):
    """
    Multiply each value by scale_factor.
    """
    return values * scale_factor




def update_extent(pyvista_grid, new_extent):
    # new_extent: array-like with 6 elements [xmin, xmax, ymin, ymax, zmin, zmax]
    old_bounds = np.array(pyvista_grid.bounds)  # [xmin, xmax, ymin, ymax, zmin, zmax]

    # Check for valid extents
    if any(new_extent[i] >= new_extent[i + 1] for i in range(0, 6, 2)):
        raise ValueError("Each min value must be less than the corresponding max value in the new extent.")

    # Compute old ranges and new ranges for each axis
    old_ranges = old_bounds[1::2] - old_bounds[0::2]  # [x_range, y_range, z_range]
    new_ranges = np.array([new_extent[1] - new_extent[0],
                           new_extent[3] - new_extent[2],
                           new_extent[5] - new_extent[4]])

    # Avoid division by zero if any old range is zero
    if np.any(old_ranges == 0):
        raise ValueError("One of the dimensions in the current grid has zero length.")

    # Get the old points and reshape for easier manipulation
    old_points = pyvista_grid.points  # shape (N, 3)

    # Compute normalized coordinates within the old extent
    norm_points = (old_points - old_bounds[0::2]) / old_ranges

    # Compute new points based on new extent
    new_mins = np.array([new_extent[0], new_extent[2], new_extent[4]])
    new_points = new_mins + norm_points * new_ranges

    # Update the grid's points
    pyvista_grid.points = new_points

    # Updating bounds is implicit once the points are modified.
    pyvista_grid.Modified()
    return pyvista_grid
