import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from subsurface import StructuredData
from subsurface.api.le_validation import validate_le
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

    def test_a_step_is_a_structured_dataset_on_the_voxel_centres(self):
        self.dataset.units = {"displacement": "m"}
        data = self.dataset.structured(1).data
        grid = self.sampler.grid
        self.assertEqual(list(data.data_vars), ["displacement", "region"])
        self.assertEqual(data["displacement"].dims, ("x", "y", "z", "component"))
        self.assertEqual(data["displacement"].attrs, {"units": "m", "kind": "displacement"})
        self.assertEqual(data["region"].attrs, {})
        self.assertEqual(data["component"].values.tolist(), ["x", "y", "z"])
        for a, axis in enumerate("xyz"):
            np.testing.assert_array_equal(data[axis].values, grid.axis(a))
        columns = {c.name: c for c in self.dataset.columns}
        np.testing.assert_array_equal(data["displacement"].values, columns["displacement"].steps[1])

    def test_format_2_keeps_every_field_x_fastest_component_slowest(self):
        binary = self.dataset.structured(1).to_binary(format_version=2)
        header, body = read_le(binary)
        nx, ny, nz = self.sampler.grid.shape
        self.assertEqual(header["data_shape"], [nx, ny, nz])
        self.assertEqual([(v["name"], v["dims"]) for v in header["data_vars"]],
                         [("displacement", ["x", "y", "z", "component"]), ("region", ["x", "y", "z"])])
        # * Bounds are the outer voxel centres; the voxels tile the mesh's box around them
        lo, hi = self.sampler.grid.bounds.T
        spacing = self.sampler.grid.spacing
        np.testing.assert_allclose([header["bounds"][axis] for axis in "xyz"], np.column_stack([lo + spacing / 2, hi - spacing / 2]))
        floats = np.frombuffer(body, np.float32)
        values = next(c for c in self.dataset.columns if c.name == "displacement").steps[1]
        i, j, k, c = 5, 2, 3, 2
        self.assertEqual(floats[i + nx * (j + ny * (k + nz * c))], values[i, j, k, c])
        self.assertEqual(validate_le(binary, family="volume", role="volume").grid_variables[0].name, "displacement")

    def test_2d_and_3d_vectors_cannot_share_a_dataset(self):
        flat = export_grids.Column("flat", [np.zeros(self.sampler.grid.shape + (2,), np.float32)] * 3)
        self.dataset.columns.append(flat)
        with self.assertRaises(ValueError):
            self.dataset.structured(0)


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
            self.assertEqual(len(written), 5)
            self.assertEqual(master_header["format_version"], 2)
            self.assertEqual((manifest["schema_version"], manifest["kind"]), (2, "volume"))
            self.assertRegex(manifest["directory"], r"^time-series/box/[0-9a-f]{16}$")
            self.assertEqual(manifest["directory"].rsplit("/", 1)[1], manifest["revision"])
            self.assertEqual([f["timestamp"] for f in manifest["frames"]],
                             ["2026-11-19T00:00:01Z", "2026-11-19T00:00:02Z", "2026-11-19T00:00:03Z"])
            frames = [out / manifest["directory"] / f["path"] for f in manifest["frames"]]
            for path in frames:
                frame = StructuredData.from_binary_le(path)
                self.assertEqual(frame.data["displacement"].attrs, {"units": "m"})
                self.assertNotIn("time_series", frame.data.attrs)
            self.assertEqual(read_le(frames[-1].read_bytes())[1], master_body, "The master is the last frame.")
            self.assertNotEqual(read_le(frames[0].read_bytes())[1], master_body)
            index = json.loads((out / manifest["directory"] / "series.json").read_text())
            self.assertEqual(index["frames"], manifest["frames"])

    def test_the_same_dataset_keeps_its_revision_and_a_changed_one_gets_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def revision():
                export_grids.write_dataset(self.dataset, "box", out, START)
                return read_le((out / "volumes/box.le").read_bytes())[0]["xarray_attrs"]["time_series"]["revision"]

            first = revision()
            self.assertEqual(revision(), first)
            self.dataset.columns[0].steps[0] = self.dataset.columns[0].steps[0] + 1
            self.assertNotEqual(revision(), first)
            self.assertEqual(len(list((out / "time-series/box").iterdir())), 2)

    def test_one_step_is_a_plain_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = tr.build_sampler(results().points, results().tets, 0.25)
            last = export_grids.dataset_from_results(results(), sampler, step=-1)
            written = export_grids.write_dataset(last, "box", Path(tmp), None)
            self.assertEqual([p.relative_to(tmp).as_posix() for p in written], ["volumes/box.le"])
            report = validate_le(written[0].read_bytes(), family="volume", role="volume")
            self.assertEqual([v.name for v in report.grid_variables], ["displacement", "region"])


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
