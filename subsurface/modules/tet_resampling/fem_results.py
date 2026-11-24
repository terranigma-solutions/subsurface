"""Read time-dependent results on linear tetrahedra from XDMF/HDF5 or a VTK file series.

Both readers return the same ``FemResults``. The mesh must be the same in
every time step, which is what lets one sampler serve all steps. A field
whose values never change (XDMF: the same HDF5 dataset in every step; VTK:
equal arrays) is read once and flagged as static.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Field:
    values: np.ndarray  # (n_steps, n_items, *components); n_steps is 1 when static
    static: bool

    @property
    def components(self) -> tuple[int, ...]:
        return self.values.shape[2:]


@dataclass
class FemResults:
    points: np.ndarray                 # (n_nodes, 3) float64
    tets: np.ndarray                   # (n_cells, 4) int64
    times: np.ndarray                  # (n_steps,) as given by the file
    point_data: dict[str, Field] = field(default_factory=dict)
    cell_data: dict[str, Field] = field(default_factory=dict)


def read(path: str | Path) -> FemResults:
    """Reads ``.xdmf`` / ``.xmf``, ``.vtk.series`` or a single ``.vtk`` / ``.vtu`` file."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".xdmf", ".xmf")):
        return read_xdmf(path)
    if name.endswith(".series"):
        return read_vtk_series(path)
    if name.endswith((".vtk", ".vtu")):
        return _from_steps([0.0], [_read_vtk_step(path)])
    raise ValueError(f"unsupported file type: {path.name}")


def read_xdmf(path: Path) -> FemResults:
    """A temporal collection of uniform tetrahedron grids, or one uniform grid, with HDF5 data items."""
    import h5py

    root = ET.parse(path).getroot()
    grids = [g for g in root.iter("Grid") if g.get("GridType", "Uniform") == "Uniform"]
    if not grids:
        raise ValueError(f"{path.name} has no uniform grid")

    files: dict[str, h5py.File] = {}
    cache: dict[str, np.ndarray] = {}

    def load(item: ET.Element) -> tuple[str, np.ndarray]:
        if item.get("Format") != "HDF":
            raise ValueError(f"only HDF data items are supported, got {item.get('Format')}")
        source = item.text.strip()
        if source not in cache:
            file_name, dataset = source.split(":", 1)
            if file_name not in files:
                files[file_name] = h5py.File(path.parent / file_name, "r")
            cache[source] = files[file_name][dataset][()]
        return source, cache[source]

    try:
        times, mesh, sources = [], None, []
        for grid in grids:
            time = grid.find("Time")
            times.append(float(time.get("Value")) if time is not None else 0.0)
            topology, geometry = grid.find("Topology"), grid.find("Geometry")
            if topology.get("TopologyType", "").lower() != "tetrahedron":
                raise ValueError(f"only tetrahedron topologies are supported, got {topology.get('TopologyType')}")
            if geometry.get("GeometryType", "XYZ") != "XYZ":
                raise ValueError(f"only XYZ geometry is supported, got {geometry.get('GeometryType')}")
            step_mesh = (load(geometry.find("DataItem"))[0], load(topology.find("DataItem"))[0])
            if mesh is not None and step_mesh != mesh:
                raise ValueError("the mesh changes between time steps; one sampler per mesh is not supported yet")
            mesh = step_mesh
            sources.append({(a.get("Center", "Node"), a.get("Name")): load(a.find("DataItem"))[0]
                            for a in grid.findall("Attribute")})
        points, tets = cache[mesh[0]], cache[mesh[1]]
        results = FemResults(np.asarray(points, np.float64), np.asarray(tets, np.int64), np.asarray(times))
        for center, name in sources[0]:
            paths = [step.get((center, name)) for step in sources]
            if any(p is None for p in paths):
                raise ValueError(f"attribute {name} is missing in some time steps")
            static = len(set(paths)) == 1
            values = np.stack([_tidy(cache[p]) for p in (paths[:1] if static else paths)])
            target = results.point_data if center == "Node" else results.cell_data
            target[name] = Field(values, static)
        return results
    finally:
        for f in files.values():
            f.close()


def read_vtk_series(path: Path) -> FemResults:
    """A ParaView ``.series`` file that lists one VTK file per time step."""
    series = json.loads(path.read_text())
    entries = sorted(series["files"], key=lambda e: e["time"])
    steps = [_read_vtk_step(path.parent / e["name"]) for e in entries]
    return _from_steps([e["time"] for e in entries], steps)


def _read_vtk_step(path: Path):
    import meshio

    mesh = meshio.read(path)
    blocks = [i for i, block in enumerate(mesh.cells) if block.type == "tetra"]
    if len(blocks) != 1:
        raise ValueError(f"{path.name}: expected one block of linear tetrahedra, found {len(blocks)}")
    block = blocks[0]
    return (mesh.points, mesh.cells[block].data,
            {k: _tidy(v) for k, v in mesh.point_data.items()},
            {k: _tidy(v[block]) for k, v in mesh.cell_data.items()})


def _from_steps(times, steps) -> FemResults:
    points, tets = steps[0][0], steps[0][1]
    for p, t, _, _ in steps[1:]:
        if p.shape != points.shape or t.shape != tets.shape or not (np.array_equal(p, points) and np.array_equal(t, tets)):
            raise ValueError("the mesh changes between time steps; one sampler per mesh is not supported yet")
    results = FemResults(np.asarray(points, np.float64), np.asarray(tets, np.int64), np.asarray(times, float))
    for slot, target in ((2, results.point_data), (3, results.cell_data)):
        for name in steps[0][slot]:
            frames = [s[slot][name] for s in steps]
            static = all(np.array_equal(f, frames[0]) for f in frames[1:])
            target[name] = Field(np.stack(frames[:1] if static else frames), static)
    return results


def _tidy(values) -> np.ndarray:
    """Drops a trailing component axis of size 1 and flattens 3 x 3 tensors to 9 components."""
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    if values.ndim == 3:
        return values.reshape(len(values), -1)
    return values
