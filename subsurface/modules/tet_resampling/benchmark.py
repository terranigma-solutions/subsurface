"""Times the resampler on a synthetic mesh of linear tetrahedra.

The mesh is a box of n^3 hexahedra, each split into six tetrahedra, with a
linear field that the resampler must reproduce exactly. With PyVista
installed, VTK's resampling filter runs on the same data as a reference.

    python -m subsurface.modules.tet_resampling.benchmark --cells 60 --spacing 0.01
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from . import tet_resampler as tr


def kuhn_box(n: int, lo=(-1.0, 0.5, 2.0), hi=(3.0, 2.0, 4.0)):
    """Nodes and tetrahedra of a box of n^3 hexahedra, each split into the six Kuhn tetrahedra."""
    axes = [np.linspace(lo[a], hi[a], n + 1) for a in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    ijk = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1).reshape(-1, 3)
    node = lambda c: (c[:, 0] * (n + 1) + c[:, 1]) * (n + 1) + c[:, 2]
    tets = []
    for order in itertools.permutations(range(3)):
        corner = ijk.copy()
        path = [node(corner)]
        for axis in order:
            corner = corner.copy()
            corner[:, axis] += 1
            path.append(node(corner))
        tets.append(np.column_stack(path))
    return points, np.concatenate(tets)


def linear_field(p: np.ndarray) -> np.ndarray:
    return np.column_stack([2 * p[:, 0] - 3 * p[:, 1] + 0.5 * p[:, 2] + 1, p[:, 2] - p[:, 0], p[:, 1]])


def run(label, fn, repeat):
    best = min(_timed(fn) for _ in range(repeat))
    print(f"  {label:<34s} {best:8.3f} s")
    return best


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=60, help="hexahedra per axis (6 tetrahedra each)")
    parser.add_argument("--spacing", type=float, default=0.01)
    parser.add_argument("--frames", type=int, default=10, help="3-component frames sampled after the build")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    points, tets = kuhn_box(args.cells)
    values = linear_field(points).astype(np.float32)
    grid = tr.Grid.covering(points.min(0), points.max(0), args.spacing)
    print(f"{len(points):,} nodes, {len(tets):,} tetrahedra -> grid {grid.shape} = {grid.size:,} voxels")

    sampler = tr.build_sampler(points, tets, grid)  # * warms up the Numba kernels
    centres = np.stack(np.meshgrid(*(grid.axis(a) for a in range(3)), indexing="ij"), -1).reshape(-1, 3, order="F")
    error = np.nanmax(np.abs(sampler.point_data(values).reshape(-1, 3, order="F") - linear_field(centres)))
    print(f"  max error of the linear field: {error:.1e}")

    if tr.numba is not None:
        run("build, Numba", lambda: tr.build_sampler(points, tets, grid, use_numba=True), args.repeat)
    run("build, NumPy", lambda: tr.build_sampler(points, tets, grid, use_numba=False), args.repeat)
    run(f"sample {args.frames} vector frames", lambda: [sampler.point_data(values) for _ in range(args.frames)],
        args.repeat)

    try:
        import pyvista as pv
    except ImportError:
        print("  PyVista is not installed; the VTK reference is skipped")
        return
    mesh = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, points)
    mesh.point_data["u"] = values
    image = pv.ImageData(dimensions=grid.shape, spacing=grid.spacing, origin=grid.origin)
    run("VTK resample (pyvista.sample)", lambda: image.sample(mesh), 1)


if __name__ == "__main__":
    main()
