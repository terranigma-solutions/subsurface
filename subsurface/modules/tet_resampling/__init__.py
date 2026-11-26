"""Resample results on linear tetrahedra onto regular grids, and write them as LiquidEarth grid datasets.

Optional dependencies: Numba (faster point location; NumPy gives the same result), and h5py and meshio for the
XDMF and VTK readers in ``fem_results``. See README.md in this folder.
"""
