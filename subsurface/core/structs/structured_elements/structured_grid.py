import numpy as np
from ..base_structures import StructuredData


class StructuredGrid:
    # TODO check structured_data has three coordinates
    """Container for curvilinear mesh grids.

        This is analogous to PyVista's StructuredGrid class or discretize's
        CurviMesh class.

    """

    def __init__(self, structured_data: StructuredData):
        self.ds = structured_data

    @property
    def cartesian_dimensions(self):
        return len(self.cartesian_coords_names)

    @property
    def cartesian_coords_names(self):
        coord_names = np.array(['X', 'Y', 'Z', 'x', 'y', 'z'])
        return coord_names[np.isin(coord_names, self.ds.data.dims)]

    @property
    def coord(self):
        return self.ds.data.coords

    @property
    def meshgrid_3d(self):
        cart_coord = [self.coord[i] for i in self.cartesian_coords_names]
        grid_3d = np.meshgrid(*cart_coord, indexing='ij')
        return grid_3d

    @property
    def active_attributes(self) -> np.ndarray:
        return self.ds.data[self.ds.active_data_array_name].values

    def meshgrid_2d(self, attribute_name_coord_name: str = None) -> list:
        """

        Args:
            attribute_name_coord_name(str): Name of the xarray.Dataset coord that
             will be used for the z direction. This must be 2d

        Returns:

        """
        grid_2d_: tuple = np.meshgrid(self.coord['x'], self.coord['y'])
        grid_2d = list(grid_2d_)
        if attribute_name_coord_name is not None:
            z_coord = self.ds.data[attribute_name_coord_name].values.T
            if z_coord.ndim != 2:
                raise AttributeError('The attribute must be a 2D array')

            grid_2d.append(z_coord)

        return grid_2d
