import json
import struct

import numpy as np
import pytest

from subsurface.modules.reader.mesh._GOCAD_mesh import GOCADMesh
from subsurface.modules.reader.mesh.mx_reader import _meshes_to_unstruct


@pytest.mark.parametrize(
    "vertex_counts, expected_cells",
    [
        pytest.param([4, 3], [[0, 2, 1], [4, 6, 5]], id="trailing-unused-vertex"),
        pytest.param(
            [4, 5, 3], [[0, 2, 1], [4, 6, 5], [9, 11, 10]],
            id="unused-vertices-first-and-middle",
        ),
        pytest.param([3, 3], [[0, 2, 1], [3, 5, 4]], id="sparse-reused-ids-baseline"),
        pytest.param([4], [[0, 2, 1]], id="single-block-unused-vertex"),
        pytest.param([3, 4], [[0, 2, 1], [3, 5, 4]], id="final-block-unused-vertex"),
    ],
)
def test_mx_merge_preserves_vertices_connectivity_and_surface_ids(vertex_counts, expected_cells):
    meshes = []
    vertices = []
    for block, count in enumerate(vertex_counts):
        block_vertices = np.array(
            [[block * 10, 0, 0], [block * 10 + 1, 0, 0], [block * 10, 1, 0]]
            + [[block * 10 + 99, 99 + i, 99] for i in range(count - 3)],
            dtype=float,
        )
        # Sparse IDs are deliberately reused across blocks; declaration order sets local indices.
        meshes.append(GOCADMesh(
            vertices=block_vertices,
            vertex_indices=np.array([10, 40, 7, 90, 100][:count]),
            edges=np.array([[10, 7, 40]]),
        ))
        vertices.extend(block_vertices.tolist())

    merged = _meshes_to_unstruct(meshes)
    expected_ids = np.arange(1, len(meshes) + 1)
    np.testing.assert_array_equal(merged.vertex, vertices)
    np.testing.assert_array_equal(merged.cells, expected_cells)
    np.testing.assert_array_equal(merged.data["cell_attrs"].values[:, 0], expected_ids)

    # Independently decode the column-major binary geometry and generated ID column.
    binary = merged.to_binary()
    header_length = struct.unpack_from("<I", binary)[0]
    header = json.loads(binary[4:4 + header_length])
    offset = 4 + header_length
    vertex_count = sum(vertex_counts)
    decoded_vertices = np.frombuffer(binary, dtype=np.float32, count=vertex_count * 3, offset=offset)
    offset += vertex_count * 12
    decoded_cells = np.frombuffer(binary, dtype=np.int32, count=len(meshes) * 3, offset=offset)
    offset += len(meshes) * 12
    assert header["vertex_shape"] == [vertex_count, 3]
    assert header["cell_shape"] == [len(meshes), 3]
    assert header["cell_attrs"] == [
        {"name": "id", "dtype": "int64", "shape": [len(meshes)], "byte_length": len(meshes) * 8}
    ]
    np.testing.assert_array_equal(decoded_vertices.reshape((vertex_count, 3), order="F"), vertices)
    np.testing.assert_array_equal(decoded_cells.reshape((len(meshes), 3), order="F"), expected_cells)
    np.testing.assert_array_equal(np.frombuffer(binary, dtype=np.int64, offset=offset), expected_ids)
