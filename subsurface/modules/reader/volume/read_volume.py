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
        pyvista_struct: pv.ExplicitStructuredGrid = pv_cast_to_explicit_structured_grid(pyvista_obj)
    except Exception as e:
        raise ValueError(f"The file is not a structured grid: {e}")

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


def pv_cast_to_explicit_structured_grid(pyvista_object: 'pv.DataSet') -> 'pv.ExplicitStructuredGrid':
    pv = optional_requirements.require_pyvista()

    match pyvista_object:
        case pv.RectilinearGrid() as rectl_grid:
            return __pv_convert_rectilinear_to_explicit(rectl_grid)
        case pv.UnstructuredGrid() as unstr_grid:
            return __pv_convert_unstructured_to_explicit(unstr_grid)
        case _:
            return pyvista_object.cast_to_explicit_structured_grid()


def __pv_convert_unstructured_to_explicit(unstr_grid):
    """
    Convert a PyVista UnstructuredGrid to an ExplicitStructuredGrid if possible.
    """
    pv = optional_requirements.require_pyvista()

    # First check if the grid has the necessary attributes to be treated as structured
    if not hasattr(unstr_grid, 'n_cells') or unstr_grid.n_cells == 0:
        raise ValueError("The unstructured grid has no cells.")
    
    # Try to detect if the grid has a structured topology
    # Check if the grid has cell type 11 (VTK_VOXEL) or 12 (VTK_HEXAHEDRON)
    cell_types = unstr_grid.celltypes
    
    # Voxels (11) and hexahedra (12) are the cell types used in structured grids
    if not all(ct in [11, 12] for ct in cell_types):
        raise ValueError("The unstructured grid contains non-hexahedral cells and cannot be converted to explicit structured.")
    
    # Try to infer dimensions from the grid
    try:
        # Method 1: Try PyVista's built-in conversion if available
        return unstr_grid.cast_to_explicit_structured_grid()
    except (AttributeError, TypeError):
        pass
    
    try:
        # Method 2: If the grid has dimensions stored as field data
        if "dimensions" in unstr_grid.field_data:
            dims = unstr_grid.field_data["dimensions"]
            if len(dims) == 3:
                nx, ny, nz = dims
                # Verify that dimensions match the number of cells
                if (nx-1)*(ny-1)*(nz-1) != unstr_grid.n_cells:
                    raise ValueError("Stored dimensions do not match the number of cells.")
                
                # Extract points and reorder if needed
                points = unstr_grid.points.reshape((nx, ny, nz, 3))
                
                # Create explicit structured grid
                explicit_grid = pv.ExplicitStructuredGrid((nx, ny, nz), points.reshape((-1, 3)))
                explicit_grid.compute_connectivity()
                
                # Transfer data arrays
                for name, array in unstr_grid.cell_data.items():
                    explicit_grid.cell_data[name] = array.copy()
                for name, array in unstr_grid.point_data.items():
                    explicit_grid.point_data[name] = array.copy()
                for name, array in unstr_grid.field_data.items():
                    if name != "dimensions":  # Skip dimensions field
                        explicit_grid.field_data[name] = array.copy()
                
                return explicit_grid
    except (ValueError, KeyError):
        pass
    
    # If none of the above methods work, use PyVista's extract_cells function
    # to reconstruct the structured grid if possible
    try:
        # This is a best-effort approach that tries multiple strategies
        return pv.core.filters.convert_unstructured_to_structured_grid(unstr_grid)
    except Exception as e:
        raise ValueError(f"Failed to convert unstructured grid to explicit structured grid: {e}")


def __pv_convert_rectilinear_to_explicit(rectl_grid, *, temp_dtype=None):
    """
    Convert a PyVista RectilinearGrid to an ExplicitStructuredGrid with low peak memory.

    Behavior:
      - Output points are in world coordinates, dtype matches rectl_grid.points.dtype.
      - Data arrays are shallow-transferred (no deep copies).
      - temp_dtype controls the large temporary `corners` buffer dtype.
        * If temp_dtype is None (default), use float32 when the output dtype is wider (e.g., float64),
          else use the output dtype. This reduces peak memory automatically.
        * If temp_dtype < output dtype, coordinates are recentred for precision and origin is added back after.

    Parameters
    ----------
    rectl_grid : pv.RectilinearGrid
    temp_dtype : numpy dtype or None
        Dtype for building the temporary `corners` array. Examples:
          - None (default): auto -> float32 if output dtype is wider, else output dtype.
          - np.float32: memory-friendly; auto recenters & restores origin.
          - np.float64: highest precision (more memory).

    Returns
    -------
    pv.ExplicitStructuredGrid
    """
    import numpy as np
    pv = optional_requirements.require_pyvista()

    # Output dtype follows source grid points (usually float64)
    out_dtype = getattr(rectl_grid.points, "dtype", np.float64)

    # Auto-pick temp dtype: prefer float32 when output is wider (e.g., float64)
    if temp_dtype is None:
        temp_dtype = np.float32 if (
                    np.dtype(out_dtype).kind == 'f' and np.dtype(out_dtype).itemsize > 4) else out_dtype

    # Coordinate arrays
    x = np.asarray(rectl_grid.x)
    y = np.asarray(rectl_grid.y)
    z = np.asarray(rectl_grid.z)

    # Decide if we must recenter (when temp dtype is lower precision than output dtype)
    def _is_lower_precision(src, dst):
        s, d = np.dtype(src), np.dtype(dst)
        if s.kind != 'f' or d.kind != 'f':
            return s != d
        return s.itemsize < d.itemsize

    if _is_lower_precision(temp_dtype, out_dtype):
        origin = np.array([x[0], y[0], z[0]], dtype=np.float64)
        x_base, y_base, z_base = x - origin[0], y - origin[1], z - origin[2]
    else:
        origin = None
        x_base, y_base, z_base = x, y, z

    # Double coordinates (interior duplication expected by ExplicitStructuredGrid ctor)
    def _doubled(arr):
        # [a,b,c,d] -> [a, b,b, c,c, d]
        return np.repeat(arr, 2)[1:-1]

    xcorn = _doubled(x_base)
    ycorn = _doubled(y_base)
    zcorn = _doubled(z_base)

    nx2, ny2, nz2 = len(xcorn), len(ycorn), len(zcorn)
    slab = ny2 * nz2
    N = nx2 * slab

    # Build corners via slab/chunked fill (avoids N-sized intermediates)
    yz = np.empty((slab, 2), dtype=temp_dtype)
    yz[:, 0] = np.repeat(ycorn, nz2).astype(temp_dtype, copy=False)  # Y pattern
    yz[:, 1] = np.tile(zcorn, ny2).astype(temp_dtype, copy=False)  # Z pattern

    corners = np.empty((N, 3), dtype=temp_dtype)
    for i, xv in enumerate(xcorn):
        start = i * slab
        end = start + slab
        corners[start:end, 0] = xv
        corners[start:end, 1:3] = yz

    # Construct explicit grid
    dims = (len(x), len(y), len(z))
    explicit = pv.ExplicitStructuredGrid(dims, corners)
    explicit.compute_connectivity()

    # Always return world coordinates; add origin back and cast to out_dtype in one fused pass
    if origin is not None:
        new_pts = np.empty_like(explicit.points, dtype=out_dtype)
        np.add(explicit.points, origin, out=new_pts, dtype=out_dtype)
        explicit.points = new_pts
    else:
        if explicit.points.dtype != out_dtype:
            explicit.points = explicit.points.astype(out_dtype, copy=False)

    # Shallow-transfer all data arrays (no deep copies)
    for name, arr in rectl_grid.cell_data.items():
        explicit.cell_data[name] = arr
    for name, arr in rectl_grid.point_data.items():
        explicit.point_data[name] = arr
    for name, arr in rectl_grid.field_data.items():
        explicit.field_data[name] = arr

    __validate_rectilinear_to_explicit_conversion(rectl_grid, explicit)

    return explicit


def __validate_rectilinear_to_explicit_conversion(rectl_grid, explicit_grid, *, atol=1e-6, rtol=1e-8) -> None:
    """
    Validate core equivalence between a RectilinearGrid and its ExplicitStructuredGrid.
    Raises ValueError on mismatch. Avoids large 3D uniques / big temporaries.
    """
    import numpy as np

    # dims & counts
    nx, ny, nz = map(int, rectl_grid.dimensions)
    if tuple(map(int, explicit_grid.dimensions)) != (nx, ny, nz):
        raise ValueError(f"Dimensions differ: explicit {tuple(explicit_grid.dimensions)} vs rect {tuple(rectl_grid.dimensions)}")

    expected_cells = (nx - 1) * (ny - 1) * (nz - 1)
    if explicit_grid.n_cells != expected_cells:
        raise ValueError(f"Cell count mismatch: explicit {explicit_grid.n_cells} vs expected {expected_cells}")

    # Accept either nodes (M) or corners (N) for n_points, depending on PyVista/VTK version
    M = nx * ny * nz
    N = (2 * (nx - 1)) * (2 * (ny - 1)) * (2 * (nz - 1))
    if explicit_grid.n_points not in (M, N):
        raise ValueError(
            f"Point count unexpected: explicit {explicit_grid.n_points}; expected either nodes {M} or corners {N}"
        )

    # bounds
    if not np.allclose(explicit_grid.bounds, rectl_grid.bounds, rtol=rtol, atol=atol):
        raise ValueError(f"Bounds differ: explicit {explicit_grid.bounds} vs rect {rectl_grid.bounds}")

    # axis coordinates (order-independent, light memory use: 1D uniques per axis)
    pts = explicit_grid.points  # may be M×3 (unique nodes) or N×3 (corner lattice)
    x_exp = np.unique(pts[:, 0])
    y_exp = np.unique(pts[:, 1])
    z_exp = np.unique(pts[:, 2])

    x_rect = np.asarray(rectl_grid.x)
    y_rect = np.asarray(rectl_grid.y)
    z_rect = np.asarray(rectl_grid.z)

    if len(x_exp) != len(x_rect) or not np.allclose(x_exp, x_rect, rtol=rtol, atol=atol):
        raise ValueError("X axis coordinates differ.")
    if len(y_exp) != len(y_rect) or not np.allclose(y_exp, y_rect, rtol=rtol, atol=atol):
        raise ValueError("Y axis coordinates differ.")
    if len(z_exp) != len(z_rect) or not np.allclose(z_exp, z_rect, rtol=rtol, atol=atol):
        raise ValueError("Z axis coordinates differ.")
