"""Resample results on linear tetrahedra onto a regular grid.

The grid is cell-centred: voxel (i, j, k) has its centre at
``origin + (i, j, k) * spacing``, and the voxels tile ``Grid.bounds`` exactly.
A 3D texture spans the same box, so LiquidEarth shows each value where it was
sampled.

``build_sampler`` finds the tetrahedron that contains each voxel centre, and
the barycentric weights of the centre in it. This is the expensive step, and
it runs once per mesh. Sampling a field after that is one gather and one
weighted sum, so all time steps and attributes of a static mesh reuse one
sampler.

Point location rasterises each tetrahedron row by row. For a row of voxels
(fixed j and k), each barycentric coordinate is linear in x, so the voxels
inside the tetrahedron form one interval, which is solved directly instead of
testing every voxel of the bounding box. With Numba installed, z-slabs of
the grid are processed in parallel; otherwise NumPy processes the
tetrahedra in vectorised chunks with the same arithmetic. Both give the same
result on every run: a voxel centre on a shared face goes to the tetrahedron
with the higher index.

Linear interpolation inside a linear tetrahedron is the field the solver
computed, so point data is sampled without loss. Cell data takes the value of
the containing tetrahedron. Voxels whose centre is in no tetrahedron are NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import numba
    from numba import prange
except ImportError:  # * NumPy gives the same result, only more slowly
    numba = None
    prange = range

# * Voxel centres on a face shared by two tetrahedra are accepted by both, never by neither
DEFAULT_TOLERANCE = 1e-6

# * Bounds the temporary arrays of the NumPy path to a few hundred MB
_NUMPY_ROWS_PER_CHUNK = 1 << 21
_NUMPY_VOXELS_PER_CHUNK = 1 << 21


@dataclass(frozen=True)
class Grid:
    """A regular, axis-aligned, cell-centred grid."""

    shape: tuple[int, int, int]
    origin: np.ndarray   # centre of voxel (0, 0, 0)
    spacing: np.ndarray  # per axis, the requested spacing or less (to within float32 rounding)

    @classmethod
    def covering(cls, lo, hi, spacing) -> Grid:
        """The grid that tiles the box [lo, hi] with voxels no larger than ``spacing``."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        spacing = np.broadcast_to(np.asarray(spacing, float), (3,))
        if np.any(spacing <= 0):
            raise ValueError(f"spacing must be positive, got {spacing.tolist()}")
        extent = hi - lo
        # * Float32 coordinates make an exact multiple of the spacing slightly larger; do not add a voxel for that
        shape = np.maximum(1, np.ceil(extent / spacing * (1 - 1e-6))).astype(int)
        step = np.where(extent > 0, extent / shape, spacing)
        return cls(tuple(int(n) for n in shape), lo + step / 2, step)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def bounds(self) -> np.ndarray:
        """(3, 2) array of the minimum and maximum of each axis."""
        lo = self.origin - self.spacing / 2
        return np.column_stack([lo, lo + self.spacing * np.asarray(self.shape)])

    def axis(self, a: int) -> np.ndarray:
        """Voxel centre coordinates along axis ``a``."""
        return self.origin[a] + self.spacing[a] * np.arange(self.shape[a])

    def centres(self, voxels: np.ndarray) -> np.ndarray:
        """Centres of voxels given by Fortran-order index (x fastest)."""
        nx, ny, _ = self.shape
        ijk = np.column_stack([voxels % nx, (voxels // nx) % ny, voxels // (nx * ny)])
        return self.origin + ijk * self.spacing


@dataclass(frozen=True)
class TetSampler:
    """The voxels of ``grid`` that lie inside the mesh and how to interpolate them."""

    grid: Grid
    voxels: np.ndarray   # (k,) int64 Fortran-order index (x fastest) of each covered voxel
    cells: np.ndarray    # (k,) int32 tetrahedron that contains each of those voxels
    nodes: np.ndarray    # (k, 4) int32 nodes of that tetrahedron
    weights: np.ndarray  # (k, 4) float32 barycentric weights of the voxel centre

    @property
    def inside(self) -> np.ndarray:
        """Boolean grid of the voxels that have values."""
        mask = np.zeros(self.grid.size, bool)
        mask[self.voxels] = True
        return self._to_grid(mask)

    def point_data(self, values) -> np.ndarray:
        """Interpolates node values (n_nodes, *components) onto the grid (nx, ny, nz, *components)."""
        values = np.asarray(values, np.float32)
        flat = values.reshape(len(values), -1)
        out = np.full((self.grid.size, flat.shape[1]), np.nan, np.float32)
        if numba is not None:
            _apply_kernel(self.voxels, self.nodes, self.weights, np.ascontiguousarray(flat), out)
        else:
            for s in range(0, len(self.voxels), _NUMPY_VOXELS_PER_CHUNK):
                part = slice(s, s + _NUMPY_VOXELS_PER_CHUNK)
                out[self.voxels[part]] = np.einsum("kn,knc->kc", self.weights[part], flat[self.nodes[part]])
        return self._to_grid(out.reshape((self.grid.size,) + values.shape[1:]))

    def cell_data(self, values) -> np.ndarray:
        """Assigns cell values (n_cells, *components) to the grid (nx, ny, nz, *components)."""
        values = np.asarray(values)
        dtype = values.dtype if values.dtype.kind == "f" else np.float32
        out = np.full((self.grid.size,) + values.shape[1:], np.nan, dtype)
        out[self.voxels] = values[self.cells]
        return self._to_grid(out)

    def _to_grid(self, flat: np.ndarray) -> np.ndarray:
        # * A view: voxel axes are Fortran-ordered, so a scalar result writes to .le without reordering
        nx, ny, nz = self.grid.shape
        tail = flat.shape[1:]
        axes = (2, 1, 0) + tuple(range(3, 3 + len(tail)))
        return flat.reshape((nz, ny, nx) + tail).transpose(axes)


def suggest_spacing(points, tets) -> float:
    """A quarter of the mean edge length: finer than any detail linear tetrahedra can hold."""
    points, tets = np.asarray(points, float), np.asarray(tets)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    lengths = [np.linalg.norm(points[tets[:, a]] - points[tets[:, b]], axis=1) for a, b in edges]
    return float(np.mean(lengths)) / 4


def build_sampler(points, tets, spacing, *, tolerance: float = DEFAULT_TOLERANCE,
                  use_numba: bool | None = None) -> TetSampler:
    """Locates the voxel centres of a grid in a mesh of linear tetrahedra.

    Args:
        points: (n_nodes, 3) node coordinates.
        tets: (n_cells, 4) node indices of each tetrahedron.
        spacing: voxel size (scalar or per axis) of a grid around the mesh, or a ``Grid``.
        tolerance: barycentric tolerance for centres on faces.
        use_numba: None uses Numba when it is installed.
    """
    points = np.ascontiguousarray(points, np.float64)
    tets = np.ascontiguousarray(tets, np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"expected points (n, 3) and tets (m, 4), got {points.shape} and {tets.shape}")
    grid = spacing if isinstance(spacing, Grid) else Grid.covering(points.min(0), points.max(0), spacing)
    if use_numba is None:
        use_numba = numba is not None
    elif use_numba and numba is None:
        raise ImportError("use_numba=True, but Numba is not installed")

    nx, ny, _ = grid.shape
    owner = np.full(grid.size, -1, np.int32)
    if use_numba:
        n = len(tets)
        v0, inverse = np.empty((n, 3)), np.empty((n, 3, 3))
        lo_idx, hi_idx = np.empty((n, 3), np.int64), np.empty((n, 3), np.int64)
        # * Degenerate tetrahedra get an empty index range
        _prepare_kernel(points, tets, grid.origin, grid.spacing, np.asarray(grid.shape), tolerance,
                        v0, inverse, lo_idx, hi_idx)
        _locate_kernel(v0, inverse, lo_idx, hi_idx, np.arange(n, dtype=np.int32), grid.origin, grid.spacing,
                       nx, ny, grid.shape[2], tolerance, owner)
    else:
        v0, inverse, keep = _tet_frames(points, tets)
        lo_idx, hi_idx = _index_ranges(points, tets[keep], grid, tolerance)
        _locate_numpy(v0[keep], inverse[keep], lo_idx, hi_idx, np.flatnonzero(keep).astype(np.int32),
                      grid.origin, grid.spacing, nx, ny, tolerance, owner)

    voxels = np.flatnonzero(owner >= 0)
    cells = owner[voxels]
    nodes = np.empty((len(cells), 4), np.int32)
    weights = np.empty((len(cells), 4), np.float32)
    barycentric = _weights_kernel if use_numba else _weights_numpy
    barycentric(voxels, cells, tets, v0, inverse, grid.origin, grid.spacing, nx, ny, nodes, weights)
    return TetSampler(grid, voxels, cells, nodes, weights)


def _tet_frames(points, tets):
    """Corner 0 and the inverse edge matrix of each tetrahedron; degenerate ones are not kept."""
    v0 = points[tets[:, 0]]
    edges = np.stack([points[tets[:, c]] - v0 for c in (1, 2, 3)], axis=-1)  # columns are edges
    det = np.linalg.det(edges)
    scale = np.max(np.abs(edges), axis=(1, 2))
    keep = np.abs(det) > 1e-12 * scale ** 3
    inverse = np.zeros_like(edges)
    inverse[keep] = np.linalg.inv(edges[keep])
    return v0, np.ascontiguousarray(inverse), keep


def _index_ranges(points, tets, grid, tolerance):
    """First and last voxel index along each axis whose centre can lie in each tetrahedron."""
    corners = points[tets]
    pad = tolerance * (corners.max(1) - corners.min(1))
    lo = np.ceil((corners.min(1) - pad - grid.origin) / grid.spacing).astype(np.int64)
    hi = np.floor((corners.max(1) + pad - grid.origin) / grid.spacing).astype(np.int64)
    top = np.asarray(grid.shape) - 1
    return np.clip(lo, 0, None), np.minimum(hi, top)


def _weights_numpy(voxels, cells, tets, v0, inverse, origin, spacing, nx, ny, nodes, weights):
    for s in range(0, len(cells), _NUMPY_VOXELS_PER_CHUNK):
        part = slice(s, s + _NUMPY_VOXELS_PER_CHUNK)
        nodes[part] = tets[cells[part]]
        v = voxels[part]
        ijk = np.column_stack([v % nx, (v // nx) % ny, v // (nx * ny)])
        local = origin + ijk * spacing - v0[cells[part]]
        b = np.einsum("kij,kj->ki", inverse[cells[part]], local)
        weights[part, 0] = 1 - b.sum(1)
        weights[part, 1:] = b


def _weights_rows(voxels, cells, tets, v0, inverse, origin, spacing, nx, ny, nodes, weights):
    for n in prange(len(voxels)):
        v, t = voxels[n], cells[n]
        for c in range(4):
            nodes[n, c] = tets[t, c]
        p = (origin[0] + (v % nx) * spacing[0] - v0[t, 0],
             origin[1] + ((v // nx) % ny) * spacing[1] - v0[t, 1],
             origin[2] + (v // (nx * ny)) * spacing[2] - v0[t, 2])
        total = 0.0
        for m in range(3):
            b = inverse[t, m, 0] * p[0] + inverse[t, m, 1] * p[1] + inverse[t, m, 2] * p[2]
            weights[n, m + 1] = b
            total += b
        weights[n, 0] = 1.0 - total


def _prepare_rows(points, tets, origin, spacing, shape, tolerance, v0, inverse, lo_idx, hi_idx):
    """Per tetrahedron: corner 0, the inverse edge matrix and the voxel index range of its bounding box."""
    for t in prange(len(tets)):
        a = tets[t, 0]
        e = np.empty((3, 3))  # * column c is the edge from corner 0 to corner c + 1
        scale = 0.0
        for c in range(3):
            for r in range(3):
                e[r, c] = points[tets[t, c + 1], r] - points[a, r]
                scale = max(scale, abs(e[r, c]))
        det = (e[0, 0] * (e[1, 1] * e[2, 2] - e[1, 2] * e[2, 1])
               - e[0, 1] * (e[1, 0] * e[2, 2] - e[1, 2] * e[2, 0])
               + e[0, 2] * (e[1, 0] * e[2, 1] - e[1, 1] * e[2, 0]))
        for r in range(3):
            v0[t, r] = points[a, r]
        if abs(det) <= 1e-12 * scale ** 3:
            for r in range(3):
                lo_idx[t, r], hi_idx[t, r] = 0, -1
            continue
        inverse[t, 0, 0] = (e[1, 1] * e[2, 2] - e[1, 2] * e[2, 1]) / det
        inverse[t, 0, 1] = (e[0, 2] * e[2, 1] - e[0, 1] * e[2, 2]) / det
        inverse[t, 0, 2] = (e[0, 1] * e[1, 2] - e[0, 2] * e[1, 1]) / det
        inverse[t, 1, 0] = (e[1, 2] * e[2, 0] - e[1, 0] * e[2, 2]) / det
        inverse[t, 1, 1] = (e[0, 0] * e[2, 2] - e[0, 2] * e[2, 0]) / det
        inverse[t, 1, 2] = (e[0, 2] * e[1, 0] - e[0, 0] * e[1, 2]) / det
        inverse[t, 2, 0] = (e[1, 0] * e[2, 1] - e[1, 1] * e[2, 0]) / det
        inverse[t, 2, 1] = (e[0, 1] * e[2, 0] - e[0, 0] * e[2, 1]) / det
        inverse[t, 2, 2] = (e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0]) / det
        for r in range(3):
            lo, hi = points[a, r], points[a, r]
            for c in range(1, 4):
                lo = min(lo, points[tets[t, c], r])
                hi = max(hi, points[tets[t, c], r])
            pad = tolerance * (hi - lo)
            lo_idx[t, r] = max(0, int(math.ceil((lo - pad - origin[r]) / spacing[r])))
            hi_idx[t, r] = min(shape[r] - 1, int(math.floor((hi + pad - origin[r]) / spacing[r])))


def _row_interval(inv, py, pz, tolerance):
    """Range of x (relative to corner 0) inside a tetrahedron on the row (py, pz).

    Each barycentric coordinate on the row is a * x + c and must be at least
    -tolerance. Returns (lo, hi); the row misses the tetrahedron when lo > hi.
    """
    lo, hi = -math.inf, math.inf
    sum_a, sum_c = 0.0, 0.0
    for m in range(4):
        if m < 3:
            a = inv[m, 0]
            c = inv[m, 1] * py + inv[m, 2] * pz
            sum_a += a
            sum_c += c
        else:
            a, c = -sum_a, 1.0 - sum_c
        if a > 0:
            lo = max(lo, (-tolerance - c) / a)
        elif a < 0:
            hi = min(hi, (-tolerance - c) / a)
        elif c < -tolerance:
            return 1.0, 0.0
    return lo, hi


def _locate_slabs(v0, inverse, lo_idx, hi_idx, cell_ids, origin, spacing, nx, ny, nz, tolerance, owner):
    """Rasterises the tetrahedra one z-slab per thread.

    Within a slab the tetrahedra run in index order, so a centre on a face
    shared by two tetrahedra always goes to the later one, whatever the thread
    count. The NumPy path keeps the same tetrahedron.
    """
    starts = np.zeros(nz + 1, np.int64)
    for t in range(len(v0)):
        for k in range(lo_idx[t, 2], hi_idx[t, 2] + 1):
            starts[k + 1] += 1
    for k in range(nz):
        starts[k + 1] += starts[k]
    fill = starts[:-1].copy()
    slab_tets = np.empty(starts[nz], np.int64)
    for t in range(len(v0)):
        for k in range(lo_idx[t, 2], hi_idx[t, 2] + 1):
            slab_tets[fill[k]] = t
            fill[k] += 1

    for k in prange(nz):
        z = origin[2] + k * spacing[2]
        for entry in range(starts[k], starts[k + 1]):
            t = slab_tets[entry]
            inv = inverse[t]
            pz = z - v0[t, 2]
            for j in range(lo_idx[t, 1], hi_idx[t, 1] + 1):
                py = origin[1] + j * spacing[1] - v0[t, 1]
                x_lo, x_hi = _row_interval(inv, py, pz, tolerance)
                if x_lo > x_hi:
                    continue
                f0 = (x_lo + v0[t, 0] - origin[0]) / spacing[0]
                f1 = (x_hi + v0[t, 0] - origin[0]) / spacing[0]
                i0 = lo_idx[t, 0] if f0 <= lo_idx[t, 0] else int(math.ceil(f0))
                i1 = hi_idx[t, 0] if f1 >= hi_idx[t, 0] else int(math.floor(f1))
                row = nx * (j + ny * k)
                for i in range(i0, i1 + 1):
                    owner[row + i] = cell_ids[t]


def _locate_numpy(v0, inverse, lo_idx, hi_idx, cell_ids, origin, spacing, nx, ny, tolerance, owner):
    rows_per_tet = (np.maximum(0, hi_idx[:, 1] - lo_idx[:, 1] + 1) *
                    np.maximum(0, hi_idx[:, 2] - lo_idx[:, 2] + 1))
    ends = np.cumsum(rows_per_tet)
    start = 0
    while start < len(v0):
        stop = int(np.searchsorted(ends, ends[start] - rows_per_tet[start] + _NUMPY_ROWS_PER_CHUNK, "right"))
        stop = max(stop, start + 1)
        _locate_numpy_chunk(slice(start, stop), v0, inverse, lo_idx, hi_idx, rows_per_tet, cell_ids,
                            origin, spacing, nx, ny, tolerance, owner)
        start = stop


def _locate_numpy_chunk(part, v0, inverse, lo_idx, hi_idx, rows_per_tet, cell_ids,
                        origin, spacing, nx, ny, tolerance, owner):
    counts = rows_per_tet[part]
    t = np.repeat(np.arange(part.start, part.stop), counts)
    if len(t) == 0:
        return
    r = np.arange(len(t)) - np.repeat(np.cumsum(counts) - counts, counts)
    width = hi_idx[t, 1] - lo_idx[t, 1] + 1
    j = lo_idx[t, 1] + r % width
    k = lo_idx[t, 2] + r // width
    py = origin[1] + j * spacing[1] - v0[t, 1]
    pz = origin[2] + k * spacing[2] - v0[t, 2]

    inv = inverse[t]
    a = inv[:, :, 0]
    c = inv[:, :, 1] * py[:, None] + inv[:, :, 2] * pz[:, None]
    a = np.column_stack([a, -a.sum(1)])
    c = np.column_stack([c, 1.0 - c.sum(1)])
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = (-tolerance - c) / a
    x_lo = np.where(a > 0, bound, -np.inf).max(1)
    x_hi = np.where(a < 0, bound, np.inf).min(1)
    x_hi[np.any((a == 0) & (c < -tolerance), axis=1)] = -np.inf

    with np.errstate(invalid="ignore"):
        i0 = np.maximum(lo_idx[t, 0], np.ceil((x_lo + v0[t, 0] - origin[0]) / spacing[0]))
        i1 = np.minimum(hi_idx[t, 0], np.floor((x_hi + v0[t, 0] - origin[0]) / spacing[0]))
    hit = (x_lo <= x_hi) & (i0 <= i1)
    i0, i1, t, j, k = i0[hit].astype(np.int64), i1[hit].astype(np.int64), t[hit], j[hit], k[hit]
    lengths = i1 - i0 + 1
    first = np.repeat(i0 + nx * (j + ny * k), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    owner[first + offsets] = np.repeat(cell_ids[t], lengths)


def _apply_rows(voxels, nodes, weights, values, out):
    for v in prange(len(voxels)):
        target = voxels[v]
        for c in range(values.shape[1]):
            acc = 0.0
            for n in range(4):
                acc += weights[v, n] * values[nodes[v, n], c]
            out[target, c] = acc


if numba is not None:
    _row_interval = numba.njit(cache=True)(_row_interval)
    _prepare_kernel = numba.njit(parallel=True, cache=True)(_prepare_rows)
    _locate_kernel = numba.njit(parallel=True, cache=True)(_locate_slabs)
    _weights_kernel = numba.njit(parallel=True, cache=True)(_weights_rows)
    _apply_kernel = numba.njit(parallel=True, cache=True)(_apply_rows)
