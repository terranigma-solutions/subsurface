import unittest

import numpy as np

from subsurface.modules.tet_resampling import tet_resampler as tr
from subsurface.modules.tet_resampling.benchmark import kuhn_box, linear_field

LO, HI = np.array([-1.0, 0.5, 2.0]), np.array([3.0, 2.0, 4.0])


def centres(grid: tr.Grid) -> np.ndarray:
    """(nx, ny, nz, 3) voxel centres."""
    return np.stack(np.meshgrid(*(grid.axis(a) for a in range(3)), indexing="ij"), -1)


def backends():
    return (True, False) if tr.numba is not None else (False,)


class GridTests(unittest.TestCase):
    def test_voxels_tile_the_box_and_are_no_larger_than_the_spacing(self):
        grid = tr.Grid.covering(LO, HI, 0.3)
        np.testing.assert_allclose(grid.bounds, np.column_stack([LO, HI]))
        self.assertEqual(grid.shape, (14, 5, 7))
        self.assertTrue(np.all(grid.spacing <= 0.3))

    def test_float32_rounding_of_the_extent_adds_no_voxel(self):
        lo, hi = np.float32(-0.6), np.float32(0.6)  # * 1.2000000477 after float32 rounding
        grid = tr.Grid.covering([lo, lo, lo], [hi, hi, hi], 0.1)
        self.assertEqual(grid.shape, (12, 12, 12))

    def test_centres_follow_fortran_order(self):
        grid = tr.Grid.covering(LO, HI, 0.5)
        nx, ny, _ = grid.shape
        i, j, k = 3, 2, 1
        np.testing.assert_allclose(grid.centres(np.array([i + nx * (j + ny * k)]))[0], centres(grid)[i, j, k])


class LinearFieldTests(unittest.TestCase):
    def test_linear_point_data_is_reproduced_exactly(self):
        points, tets = kuhn_box(6, LO, HI)
        values = linear_field(points)
        for use_numba in backends():
            with self.subTest(use_numba=use_numba):
                sampler = tr.build_sampler(points, tets, 0.07, use_numba=use_numba)
                sampled = sampler.point_data(values)
                expected = linear_field(centres(sampler.grid).reshape(-1, 3)).reshape(sampled.shape)
                self.assertEqual(sampled.shape, sampler.grid.shape + (3,))
                self.assertFalse(np.isnan(sampled).any(), "a convex mesh covers every voxel centre")
                np.testing.assert_allclose(sampled, expected, atol=2e-6)

    def test_scalar_point_data_has_the_grid_shape(self):
        points, tets = kuhn_box(3, LO, HI)
        sampler = tr.build_sampler(points, tets, 0.4)
        sampled = sampler.point_data(points[:, 0])
        self.assertEqual(sampled.shape, sampler.grid.shape)
        np.testing.assert_allclose(sampled, centres(sampler.grid)[..., 0], atol=2e-6)


class NonConvexTests(unittest.TestCase):
    def setUp(self):
        # * An L-shaped body: the box without the hexahedra where x > 1 and z > 3
        points, tets = kuhn_box(8, LO, HI)
        centroid = points[tets].mean(1)
        self.removed = lambda p: (p[..., 0] > 1) & (p[..., 2] > 3)
        self.points, self.tets = points, tets[~self.removed(centroid)]

    def test_voxels_outside_the_mesh_are_nan(self):
        for use_numba in backends():
            with self.subTest(use_numba=use_numba):
                sampler = tr.build_sampler(self.points, self.tets, 0.1, use_numba=use_numba)
                sampled = sampler.point_data(self.points[:, 2])
                outside = self.removed(centres(sampler.grid))
                self.assertTrue(outside.any() and (~outside).any())
                self.assertTrue(np.isnan(sampled[outside]).all())
                self.assertFalse(np.isnan(sampled[~outside]).any())
                np.testing.assert_array_equal(sampler.inside, ~outside)

    def test_cell_data_comes_from_a_tetrahedron_that_contains_the_centre(self):
        sampler = tr.build_sampler(self.points, self.tets, 0.1)
        ids = sampler.cell_data(np.arange(len(self.tets)))
        inside = sampler.inside
        corners = self.points[self.tets[ids[inside].astype(int)]]
        edges = np.stack([corners[:, c] - corners[:, 0] for c in (1, 2, 3)], -1)
        b = np.linalg.solve(edges, (centres(sampler.grid)[inside] - corners[:, 0])[..., None])[..., 0]
        weights = np.column_stack([1 - b.sum(1), b])
        self.assertGreaterEqual(weights.min(), -1e-5)


class RobustnessTests(unittest.TestCase):
    def test_degenerate_tetrahedra_are_ignored(self):
        points, tets = kuhn_box(2, LO, HI)
        flat = np.array([[0, 1, 2, 2]])  # * zero volume
        sampler = tr.build_sampler(points, np.concatenate([flat, tets]), 0.25)
        self.assertNotIn(0, sampler.cells)
        self.assertTrue(sampler.inside.all())

    @unittest.skipIf(tr.numba is None, "Numba is not installed")
    def test_backends_agree(self):
        points, tets = kuhn_box(7, LO, HI)
        rng = np.random.default_rng(1)
        points = points + rng.normal(0, 0.01, points.shape) * (points > LO + 1e-9).all(1)[:, None]
        a = tr.build_sampler(points, tets, 0.05, use_numba=True)
        b = tr.build_sampler(points, tets, 0.05, use_numba=False)
        np.testing.assert_array_equal(a.inside, b.inside)
        np.testing.assert_array_equal(a.cells, b.cells)
        values = rng.normal(size=(len(points), 3))
        np.testing.assert_allclose(a.point_data(values), b.point_data(values), atol=1e-5, equal_nan=True)

    def test_repeated_builds_are_identical(self):
        # * Voxel centres on shared faces must go to the same tetrahedron on every run, with any thread count
        points, tets = kuhn_box(12, LO, HI)
        first = tr.build_sampler(points, tets, 1 / 12 * 4 / 2)
        for _ in range(3):
            again = tr.build_sampler(points, tets, 1 / 12 * 4 / 2)
            np.testing.assert_array_equal(again.cells, first.cells)
            np.testing.assert_array_equal(again.weights, first.weights)


if __name__ == "__main__":
    unittest.main()
