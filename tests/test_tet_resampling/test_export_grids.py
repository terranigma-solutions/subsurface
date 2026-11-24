import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from subsurface.modules.tet_resampling import export_grids
from subsurface.modules.tet_resampling import fem_results
from subsurface.modules.tet_resampling import tet_resampler as tr
from subsurface.modules.tet_resampling.benchmark import kuhn_box

HAS_READERS = all(importlib.util.find_spec(m) is not None for m in ("h5py", "meshio"))
TIMES = [1.0, 2.0, 3.0]
START = datetime(2026, 11, 19, tzinfo=timezone.utc)


def results() -> fem_results.FemResults:
    """A box with a displacement that grows with time and a static cell scalar."""
    points, tets = kuhn_box(3, (0, 0, 0), (2, 1, 1))
    u = np.column_stack([points[:, 0], -points[:, 2], 0.5 * np.ones(len(points))])
    r = fem_results.FemResults(points, tets, np.array(TIMES))
    r.point_data["displacement"] = fem_results.Field(np.stack([t * u for t in TIMES]).astype(np.float32), False)
    r.cell_data["region"] = fem_results.Field((np.arange(len(tets)) % 3 + 1)[None].astype(np.int32), True)
    r.point_data["stress"] = fem_results.Field(np.zeros((3, len(points), 9), np.float32), False)
    return r


def read_le(data: bytes):
    length = int.from_bytes(data[:4], "little")
    return json.loads(data[4:4 + length]), data[4 + length:]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.sampler = tr.build_sampler(results().points, results().tets, 0.25)
        self.dataset = export_grids.dataset_from_results(results(), self.sampler, kinds={"displacement": "displacement"})

    def test_scalars_and_vectors_become_columns_and_tensors_are_skipped(self):
        columns = {c.name: c for c in self.dataset.columns}
        self.assertEqual(sorted(columns), ["displacement", "region"])
        self.assertEqual(columns["displacement"].components, 3)
        self.assertEqual(columns["displacement"].kind, "displacement")
        self.assertEqual(columns["region"].components, 1)
        self.assertEqual(self.dataset.steps, 3)

    def test_static_fields_repeat_in_every_step_and_vectors_change(self):
        columns = {c.name: c for c in self.dataset.columns}
        for t in range(3):
            np.testing.assert_array_equal(columns["region"].steps[t], columns["region"].steps[0])
        x = np.stack(np.meshgrid(*(self.sampler.grid.axis(a) for a in range(3)), indexing="ij"), -1)
        for t, time in enumerate(TIMES):
            np.testing.assert_allclose(columns["displacement"].steps[t][..., 0], time * x[..., 0], atol=1e-5)

    def test_one_step_keeps_only_that_step(self):
        last = export_grids.dataset_from_results(results(), self.sampler, step=-1)
        self.assertEqual(last.steps, 1)
        self.assertEqual(last.times.tolist(), [3.0])

    def test_columns_follow_the_header_x_fastest_component_slowest(self):
        column = next(c for c in self.dataset.columns if c.name == "displacement")
        values = column.steps[1]
        header, body = read_le(export_grids.grid_bytes(self.sampler.grid, [(column, values)], {}))
        nx, ny, nz = self.sampler.grid.shape
        self.assertEqual(header["format_version"], 2)
        self.assertEqual(header["data_shape"], [nx, ny, nz])
        self.assertEqual(header["grid_attrs"], [{"name": "displacement", "dtype": "float32", "shape": [nx, ny, nz, 3],
                                                 "byte_length": nx * ny * nz * 3 * 4, "kind": "displacement"}])
        floats = np.frombuffer(body, np.float32)
        i, j, k, c = 5, 2, 3, 2
        self.assertEqual(floats[i + nx * (j + ny * (k + nz * c))], values[i, j, k, c])


class SeriesTests(unittest.TestCase):
    def setUp(self):
        sampler = tr.build_sampler(results().points, results().tets, 0.25)
        self.dataset = export_grids.dataset_from_results(results(), sampler)
        self.dataset.units = {"displacement": "m"}

    def test_several_steps_need_a_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_grids.write_dataset(self.dataset, "box", Path(tmp), None)

    def test_series_is_laid_out_like_a_published_volume_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = export_grids.write_dataset(self.dataset, "box", out, START)
            master_header, master_body = read_le((out / "volumes/box.le").read_bytes())
            manifest = master_header["xarray_attrs"]["time_series"]
            self.assertEqual(len(written), 4)
            self.assertEqual((manifest["schema_version"], manifest["kind"], manifest["directory"]),
                             (2, "volume", "time-series/box/1"))
            self.assertEqual([f["timestamp"] for f in manifest["frames"]],
                             ["2026-11-19T00:00:01.000000Z", "2026-11-19T00:00:02.000000Z", "2026-11-19T00:00:03.000000Z"])
            self.assertEqual(master_header["xarray_attrs"]["attribute_units"], {"displacement": "m"})
            frames = [read_le((out / manifest["directory"] / f["path"]).read_bytes()) for f in manifest["frames"]]
            for header, _ in frames:
                self.assertEqual(header["grid_attrs"], master_header["grid_attrs"])
                self.assertNotIn("time_series", header["xarray_attrs"])
            self.assertEqual(frames[-1][1], master_body, "The master is the last frame.")
            self.assertNotEqual(frames[0][1], master_body)
            self.assertEqual(json.loads((out / manifest["directory"] / "series.json").read_text()), manifest)

    def test_one_step_is_a_plain_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = tr.build_sampler(results().points, results().tets, 0.25)
            last = export_grids.dataset_from_results(results(), sampler, step=-1)
            written = export_grids.write_dataset(last, "box", Path(tmp), None)
            header, _ = read_le(written[0].read_bytes())
            self.assertEqual([p.relative_to(tmp).as_posix() for p in written], ["volumes/box.le"])
            self.assertNotIn("time_series", header["xarray_attrs"])


@unittest.skipUnless(HAS_READERS, "h5py and meshio are needed")
class ReaderTests(unittest.TestCase):
    def test_xdmf_and_vtk_series_read_the_same_results(self):
        import h5py
        import meshio

        r = results()
        steps = r.point_data["displacement"].values
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with h5py.File(tmp / "r.h5", "w") as h5:
                h5["mesh/geometry"], h5["mesh/topology"] = r.points, r.tets.astype(np.int32)
                h5["mesh/region"] = r.cell_data["region"].values[0]
                for t in range(len(TIMES)):
                    h5[f"step_{t}/displacement"] = steps[t]
            grids = "".join(f"""
   <Grid Name="step_{t}" GridType="Uniform"><Time Value="{time}"/>
    <Topology TopologyType="Tetrahedron" NumberOfElements="{len(r.tets)}"><DataItem Format="HDF">r.h5:/mesh/topology</DataItem></Topology>
    <Geometry GeometryType="XYZ"><DataItem Format="HDF">r.h5:/mesh/geometry</DataItem></Geometry>
    <Attribute Name="region" Center="Cell"><DataItem Format="HDF">r.h5:/mesh/region</DataItem></Attribute>
    <Attribute Name="displacement" AttributeType="Vector" Center="Node"><DataItem Format="HDF">r.h5:/step_{t}/displacement</DataItem></Attribute>
   </Grid>""" for t, time in enumerate(TIMES))
            (tmp / "r.xdmf").write_text(f'<Xdmf Version="3.0"><Domain><Grid GridType="Collection" '
                                        f'CollectionType="Temporal">{grids}</Grid></Domain></Xdmf>')
            files = []
            for t, time in enumerate(TIMES):
                meshio.write(tmp / f"r_{t}.vtk", meshio.Mesh(
                    r.points, [("tetra", r.tets)], point_data={"displacement": steps[t]},
                    cell_data={"region": [r.cell_data["region"].values[0]]}), binary=False)
                files.append({"name": f"r_{t}.vtk", "time": time})
            (tmp / "r.vtk.series").write_text(json.dumps({"file-series-version": "1.0", "files": files}))

            xdmf, vtk = fem_results.read(tmp / "r.xdmf"), fem_results.read(tmp / "r.vtk.series")
        for got in (xdmf, vtk):
            np.testing.assert_allclose(got.points, r.points, atol=1e-6)
            np.testing.assert_array_equal(got.tets, r.tets)
            np.testing.assert_array_equal(got.times, TIMES)
            np.testing.assert_allclose(got.point_data["displacement"].values, steps, atol=1e-6)
            self.assertFalse(got.point_data["displacement"].static)
            self.assertTrue(got.cell_data["region"].static)
            np.testing.assert_array_equal(got.cell_data["region"].values[0], r.cell_data["region"].values[0])


if __name__ == "__main__":
    unittest.main()
