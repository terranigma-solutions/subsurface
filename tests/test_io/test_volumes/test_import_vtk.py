from __future__ import annotations
from pyvista import examples


def generate_vtk():

    grid = examples.load_hexbeam()
    grid.cell_data['Cell Number'] = range(grid.n_cells)
    grid.plot(scalars='Cell Number')
    
    # Write vtk
    grid.save("test.vtk")
   
    
