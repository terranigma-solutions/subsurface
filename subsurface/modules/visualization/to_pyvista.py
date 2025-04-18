import enum

import warnings

from typing import Union, Tuple, Optional

from ... import optional_requirements
from ...core.structs.unstructured_elements import PointSet, TriSurf, LineSet, TetraMesh
from ...core.structs.structured_elements.structured_grid import StructuredGrid
import numpy as np

try:
    import pyvista as pv
except ImportError:
    warnings.warn('Pyvista is not installed. Some visualization functions will not work.')


def pv_plot(meshes: list,
            image_2d=False,
            ve=None,
            cmap='viridis',
            plotter_kwargs: dict = None,
            add_mesh_kwargs: dict = None,
            background_plotter=False):
    """Function to plot meshes in vtk using pyvista

    Args:
        meshes (List[pv.PolyData]):
        image_2d (bool): If True convert plot to matplotlib imshow. This helps for visualizing
         the plot in IDEs
        ve (float): vertical exaggeration
        plotter_kwargs (dict): pyvista.Plotter kwargs
        add_mesh_kwargs (dict): pyvista.add_mesh kwargs
        background_plotter (bool): if true and pyvistaqt installed use pyvista
         backgroung plotter.
    """

    add_mesh_kwargs = dict() if add_mesh_kwargs is None else add_mesh_kwargs
    p: pv.Pll = init_plotter(image_2d, ve, plotter_kwargs)

    for m in meshes:
        # Check if m has texture data
        texture = None
        if hasattr(m, '_textures') and isinstance(m._textures, dict):
            texture = m._textures.get(0, None)
        
        p.add_mesh(
            mesh=m,
            cmap=cmap,
            categories=True,
            texture=texture,
            **add_mesh_kwargs
        )

    p.show_bounds()

    if image_2d is False:
        p.show()
        return p
    else:
        fig = pyvista_to_matplotlib(p)
        return fig


def pyvista_to_matplotlib(p: "pv.Plotter"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('Matplotlib is necessary for generating a 2D image.')
    img = p.show(screenshot=True)
    img = p.last_image
    fig = plt.imshow(img)
    plt.axis('off')
    plt.show(block=False)
    p.close()
    return fig


def init_plotter(
        image_2d=False,
        ve=None,
        plotter_kwargs: dict = None
) -> "pv.Plotter":
    plotter_kwargs = dict() if plotter_kwargs is None else plotter_kwargs
    off_screen = True if image_2d is True else None
    p = pv.Plotter(**plotter_kwargs, off_screen=off_screen)
    if ve is not None:
        p.set_scale(zscale=ve)
    return p


def to_pyvista_points(point_set: PointSet):
    """Create pyvista.PolyData from PointSet

    Args:
        point_set (PointSet): Class for pointset based data structures.

    Returns:
        pv.PolyData
    """
    pv = optional_requirements.require_pyvista()
    poly: pv.PolyData = pv.PolyData(point_set.data.vertex)
    poly.point_data.update(point_set.data.attributes_to_dict)

    return poly


def to_pyvista_mesh(triangular_surface: TriSurf) -> "pv.PolyData":
    """Create planar surface PolyData from unstructured element such as TriSurf

    Returns:
        mesh texture
    """
    nve = triangular_surface.mesh.n_vertex_per_element
    vertices = triangular_surface.mesh.vertex

    # ? We need better name for these variables
    num_vertex_elements = np.full(triangular_surface.mesh.n_elements, nve)
    x = triangular_surface.mesh.cells

    cells = np.c_[num_vertex_elements, x]

    pv = optional_requirements.require_pyvista()
    mesh = pv.PolyData(vertices, cells)
    mesh.cell_data.update(triangular_surface.mesh.attributes_to_dict)
    mesh.point_data.update(triangular_surface.mesh.points_attributes)

    # If UV coordinates exist in points_attributes, set them as texture coordinates
    if triangular_surface.has_texture_data_with_uv:
        uv = np.column_stack((
            triangular_surface.mesh.points_attributes['u'],
            triangular_surface.mesh.points_attributes['v']
        ))
        mesh.active_texture_coordinates = uv
        if triangular_surface.texture is not None:
            texture_data = np.asarray(triangular_surface.texture.values, dtype=np.float32)
            mesh._textures = {0: texture_data}
            mesh.active_scalars_name = None

    elif triangular_surface.has_texture_data_without_uv:
        mesh.texture_map_to_plane(
            inplace=True,
            origin=triangular_surface.texture_origin,
            point_u=triangular_surface.texture_point_u,
            point_v=triangular_surface.texture_point_v
        )

        texture_data = np.asarray(triangular_surface.texture.values, dtype=np.float32)
        mesh._textures = {0: texture_data}
        mesh.active_scalars_name = None

    return mesh


def to_pyvista_mesh_and_texture(triangular_surface: Union[TriSurf], ) -> Tuple["pv.PolyData", Optional[np.array]]:
    """Create planar surface PolyData from unstructured element such as TriSurf

    Returns:
        mesh texture
    """
    mesh = to_pyvista_mesh(triangular_surface)

    if triangular_surface.texture is None:
        raise ValueError('unstructured_element needs texture data to be mapped.')

    mesh.texture_map_to_plane(
        inplace=True,
        origin=triangular_surface.texture_origin,
        point_u=triangular_surface.texture_point_u,
        point_v=triangular_surface.texture_point_v
    )
    tex = pv.numpy_to_texture(triangular_surface.texture.values)
    mesh._textures = {0: tex}

    from vtkmodules.util.numpy_support import vtk_to_numpy
    uv = vtk_to_numpy(mesh.GetPointData().GetTCoords())
    return mesh, uv


class PyvistaScalarType(enum.Enum):
    POINT = 'point'
    CELL = 'cell'


def to_pyvista_line(
        line_set: LineSet,
        as_tube=True,
        radius=None,
        spline=False,
        n_interp_points=1000,
        scalar_type: PyvistaScalarType = PyvistaScalarType.POINT,
        active_scalar: Optional[str] = None
):
    nve = line_set.data.n_vertex_per_element
    vertices = line_set.data.vertex
    cells = np.c_[np.full(line_set.data.n_elements, nve),
    line_set.data.cells]
    if spline is False:
        mesh = pv.PolyData()
        mesh.points = vertices
        mesh.lines = cells
    else:
        raise NotImplementedError

    match scalar_type:
        case PyvistaScalarType.POINT:
            mesh.point_data.update(line_set.data.points_attributes_to_dict)
            if active_scalar is not None:
                mesh.set_active_scalars(active_scalar, preference='point')
        case PyvistaScalarType.CELL:
            mesh.cell_data.update(line_set.data.attributes_to_dict)
            if active_scalar is not None:
                mesh.set_active_scalars(active_scalar, preference='cell')
    if as_tube is True:
        return mesh.tube(radius=radius)
    else:
        return mesh


def to_pyvista_tetra(tetra_mesh: TetraMesh):
    """Create pyvista.UnstructuredGrid"""
    vertices = tetra_mesh.data.vertex
    tets = tetra_mesh.data.cells
    cells = np.c_[np.full(len(tets), 4), tets]
    import vtk
    ctypes = np.array([vtk.VTK_TETRA, ], np.int32)
    mesh = pv.UnstructuredGrid(cells, ctypes, vertices)
    mesh.cell_data.update(tetra_mesh.data.attributes_to_dict)
    return mesh


def to_pyvista_grid(
        structured_grid: StructuredGrid,
        data_set_name: str = None,
        attribute_slice: dict = None,
        data_order: str = 'F'
) -> "pyvista.StructuredGrid":
    """

    Args:
        structured_grid:
        data_set_name:
        attribute_slice: dictionary to select which 3D array will be displayed as color

    Returns:

    """
    if attribute_slice is None:
        attribute_slice = dict()

    if data_set_name is None:
        data_set_name = structured_grid.ds.active_data_array_name

    cart_dims = structured_grid.cartesian_dimensions
    data_dims = structured_grid.ds.data[data_set_name].sel(**attribute_slice).ndim
    if cart_dims < data_dims:
        raise AttributeError('Data dimension and cartesian dimensions must match.'
                             'Possibly there are not valid dimension name in the'
                             'xarray.DataArray. These are X Y Z x y z')

    if data_dims == 2:
        meshgrid = structured_grid.meshgrid_2d(data_set_name)
    elif data_dims == 3:
        meshgrid = structured_grid.meshgrid_3d
    else:
        raise AttributeError('The DataArray does not have valid dimensionality. '
                             'Possibly there are not valid dimension name in the'
                             'xarray.DataArray. These are X Y Z x y z')

    pv = optional_requirements.require_pyvista()
    mesh = pv.StructuredGrid(*meshgrid)
    update_grid_attribute(mesh, structured_grid, data_order,
                          attribute_slice, data_set_name)

    return mesh


def update_grid_attribute(
        mesh: 'pv.StructuredGrid',
        structured_grid: StructuredGrid,
        data_order='F',
        attribute_slice=None,
        data_set_name=None
):
    if attribute_slice is None:
        attribute_slice = dict()

    if data_set_name is None:
        data_set_name = structured_grid.ds.active_data_array_name
    import xarray as xr
    dataset: xr.DataArray = structured_grid.ds.data[data_set_name]

    attributeData = {data_set_name: dataset.sel(**attribute_slice).values.ravel(data_order)}
    mesh.point_data.update(attributeData)

    return mesh


def _n_cartesian_coord(attribute, structured_grid):
    coord_names = np.array(['X', 'Y', 'Z', 'x', 'y', 'z'])
    ndim = np.isin(coord_names, structured_grid.ds.data[attribute].dims).sum()
    return ndim


def _generate_colors_from_colormap(num_colors, cmap_name='viridis'):
    """
    Generate a sequence of colors from a given Matplotlib colormap.

    Parameters:
    num_colors (int): Number of colors to generate.
    cmap_name (str): Name of the Matplotlib colormap to use.

    Returns:
    list of tuple: List of RGB color tuples.
    """
    import matplotlib.pyplot as plt
    colormap = plt.cm.get_cmap(cmap_name)
    colors = colormap(np.linspace(0, 1, num_colors))
    # Convert from RGBA to RGB and scale to 0-255
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]
