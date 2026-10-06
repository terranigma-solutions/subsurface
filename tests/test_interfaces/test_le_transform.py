"""Offline synthetic contracts for affine mesh/file transforms."""

from copy import deepcopy
import os

import numpy as np
import pandas as pd
import pytest

from subsurface.api import _le_file_ops as ops
from subsurface.api.le_transform import transform_le, transform_mesh
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


@pytest.fixture
def mesh():
    return LiquidEarthMesh(
        vertex=np.array([[0., 0., 0.], [1., 0., 1.], [0., 1., 0.]]),
        cells=np.array([[0, 1, 2]], dtype=np.int32),
        attributes=pd.DataFrame({
            'id': np.array([2 ** 63 - 1], dtype=np.int64), 'flag': [True],
            'scalar': [0.125], 'vx': [1.], 'vy': [2.], 'vz': [3.],
            'nx': [-1.], 'ny': [0.], 'nz': [1.],
        }),
        points_attributes=pd.DataFrame({
            'id': np.array([2 ** 53 + 1, -7, 2 ** 63 - 1], dtype=np.int64),
            'flag': [True, False, True], 'u': [0., 0.25, 0.5], 'v': [0., 0.5, 0.75],
            'nx': [-1.] * 3, 'ny': [0.] * 3, 'nz': [1.] * 3,
            'vx': [1.] * 3, 'vy': [2.] * 3, 'vz': [3.] * 3,
        }),
        data_attrs={'crs': 'local', 'units': 'm', 'transform': None,
                    'nested': {'values': [1, 'x']}, 'bounds': [0] * 6,
                    'le_tools': {'operation': 'split', 'sources': ['original.le']}},
    )


def affine(linear=None, translation=None):
    matrix = np.eye(4)
    if linear is not None:
        matrix[:3, :3] = linear
    if translation is not None:
        matrix[:3, 3] = translation
    return matrix


MATRICES = [
    np.eye(4), affine(translation=[2, -3, 7]),
    affine([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    affine(np.diag([2, 3, 4])), affine([[1, 2, 0], [0, 1, 0.5], [0, 0, 1]]),
    affine(np.diag([-1, 1, 1])),
]


@pytest.mark.parametrize('matrix', MATRICES,
                         ids=['identity', 'translation', 'rotation', 'scale', 'shear', 'reflection'])
def test_known_geometry_and_preservation(mesh, matrix):
    original = deepcopy(mesh)
    result = transform_mesh(mesh, matrix)
    expected = mesh.vertex @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_array_equal(result.vertex, expected)
    cells = mesh.cells[:, [0, 2, 1]] if np.linalg.det(matrix[:3, :3]) < 0 else mesh.cells
    np.testing.assert_array_equal(result.cells, cells)
    pd.testing.assert_frame_equal(result.attributes, mesh.attributes)
    pd.testing.assert_frame_equal(result.points_attributes, mesh.points_attributes)
    assert {k: v for k, v in result.data_attrs.items() if k != 'bounds'} == {
        k: v for k, v in mesh.data_attrs.items() if k != 'bounds'}
    wire = expected.astype(np.float32)
    assert result.data_attrs['bounds'] == np.column_stack((wire.min(0), wire.max(0))).ravel().tolist()
    np.testing.assert_array_equal(mesh.vertex, original.vertex)
    np.testing.assert_array_equal(mesh.cells, original.cells)
    pd.testing.assert_frame_equal(mesh.attributes, original.attributes)
    pd.testing.assert_frame_equal(mesh.points_attributes, original.points_attributes)
    assert mesh.data_attrs == original.data_attrs
    result.vertex[0, 0] = 99
    result.cells[0, 0] = 2
    result.attributes.loc[0, 'id'] = 0
    result.points_attributes.loc[0, 'id'] = 0
    result.data_attrs['nested']['values'].append('changed')
    assert mesh.data_attrs == original.data_attrs
    pd.testing.assert_frame_equal(mesh.attributes, original.attributes)
    pd.testing.assert_frame_equal(mesh.points_attributes, original.points_attributes)
    np.testing.assert_array_equal(mesh.vertex, original.vertex)
    np.testing.assert_array_equal(mesh.cells, original.cells)


@pytest.mark.parametrize('matrix', MATRICES)
def test_vectors_and_normals_both_associations(mesh, matrix):
    declarations = {key: [('nx', 'ny', 'nz')] for key in ('point', 'cell')}
    vectors = {key: [('vx', 'vy', 'vz')] for key in ('point', 'cell')}
    result = transform_mesh(mesh, matrix, normal_columns=declarations, vector_columns=vectors)
    expected_vector = matrix[:3, :3] @ [1, 2, 3]
    expected_normal = np.linalg.inv(matrix[:3, :3]).T @ [-1, 0, 1]
    expected_normal /= np.linalg.norm(expected_normal)
    tangent = matrix[:3, :3] @ [1, 0, 1]
    for frame in (result.attributes, result.points_attributes):
        values = frame[['nx', 'ny', 'nz']].to_numpy()
        np.testing.assert_allclose(values, np.tile(expected_normal, (len(frame), 1)), atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(values @ tangent, 0, atol=1e-14)
        np.testing.assert_allclose(frame[['vx', 'vy', 'vz']],
                                   np.tile(expected_vector, (len(frame), 1)))
    face = result.vertex[result.cells[0]]
    face_normal = np.cross(face[1] - face[0], face[2] - face[0])
    assert face_normal @ expected_normal > 0


def test_file_composition_inverse_bounds_and_wire_dtypes(tmp_path, mesh):
    source = ops.write_le_mesh(mesh, tmp_path / 'source.le', sources=[])
    original = source.read_bytes()
    first = affine([[1, 0.3, 0], [0, 2, 0.2], [0, 0, 3]], [0.7, -2, 3])
    second = affine([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [1, 2, 0])
    one = transform_le(source, tmp_path / 'one.le', first)
    two = transform_le(one, tmp_path / 'two.le', second)
    combined = transform_le(source, tmp_path / 'combined.le', second @ first)
    restored = transform_le(two, tmp_path / 'restored.le', np.linalg.inv(second @ first))
    np.testing.assert_allclose(ops.load_le_mesh(two).vertex, ops.load_le_mesh(combined).vertex,
                               atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(ops.load_le_mesh(restored).vertex, mesh.vertex, atol=2e-6)
    output = ops.load_le_mesh(two)
    input_mesh = ops.load_le_mesh(source)
    for actual, expected in ((output.attributes, input_mesh.attributes),
                             (output.points_attributes, input_mesh.points_attributes)):
        pd.testing.assert_frame_equal(actual, expected)
        assert actual['id'].dtype == np.dtype('int64')
        assert actual['flag'].dtype == np.dtype('bool')
    assert output.data_attrs['bounds'] == np.column_stack((output.vertex.min(0), output.vertex.max(0))).ravel().tolist()
    assert source.read_bytes() == original
    assert two.is_absolute()


@pytest.mark.parametrize('matrix', [np.eye(3), np.zeros((4, 5)), np.eye(4, dtype=complex),
                                    np.eye(4, dtype=bool), np.full((4, 4), '1'),
                                    affine(np.diag([0, 1, 1])),
                                    affine([[1, 2, 3], [1, 2, 3], [0, 0, 1]]),
                                    affine(translation=[np.inf, 0, 0]),
                                    affine(translation=[np.nan, 0, 0]),
                                    np.eye(4) + np.diag([0, 0, 0, 1]),
                                    np.eye(4) + np.array([[0] * 4] * 3 + [[0.01, 0, 0, 0]]),
                                    affine(translation=[1e39, 0, 0]),
                                    affine(np.diag([1e308, 2, 3]), [1e308, 0, 0])])
def test_invalid_matrix_safe_failure(tmp_path, mesh, matrix):
    source = ops.write_le_mesh(mesh, tmp_path / 'source.le', sources=[])
    destination = tmp_path / 'output.le'
    destination.write_bytes(b'previous')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        transform_le(source, destination, matrix, overwrite=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('declarations', [[], {'vertex': [('nx', 'ny', 'nz')]},
                                         {'point': ('nx', 'ny', 'nz')},
                                         {'point': [('nx', 'ny')]},
                                         {'point': [('nx', 'nx', 'nz')]},
                                         {'point': [('missing', 'ny', 'nz')]},
                                         {'point': [('flag', 'ny', 'nz')]}])
def test_bad_column_declarations(mesh, declarations):
    with pytest.raises(ValueError):
        transform_mesh(mesh, np.eye(4), normal_columns=declarations)


def test_overlapping_normal_vector_rejected(mesh):
    columns = {'point': [('nx', 'ny', 'nz')]}
    with pytest.raises(ValueError, match='more than once'):
        transform_mesh(mesh, np.eye(4), normal_columns=columns, vector_columns=columns)


@pytest.mark.parametrize('normal', [[0, 0, 0], [np.nan, 0, 1], [np.inf, 0, 1]])
def test_invalid_normals(mesh, normal):
    mesh.attributes[['nx', 'ny', 'nz']] = [normal]
    with pytest.raises(ValueError, match='Zero normals|finite'):
        transform_mesh(mesh, np.eye(4), normal_columns={'cell': [('nx', 'ny', 'nz')]})


def test_vector_overflow(mesh):
    mesh.attributes['vz'] = 4.
    with pytest.raises(ValueError, match='float32 range'):
        transform_mesh(mesh, affine(np.diag([1, 1, 1e38])),
                       vector_columns={'cell': [('vx', 'vy', 'vz')]})


def test_file_normals_wire_tolerance_and_unselected_columns(tmp_path, mesh):
    source = ops.write_le_mesh(mesh, tmp_path / 'source.le', sources=[])
    original = source.read_bytes()
    matrix = affine([[2, 0.5, 0], [0, 3, 0.25], [0, 0, 4]], [5, -3, 1])
    result = ops.load_le_mesh(transform_le(
        source, tmp_path / 'output.le', matrix,
        normal_columns={'point': [('nx', 'ny', 'nz')], 'cell': [('nx', 'ny', 'nz')]},
        vector_columns={'point': [('vx', 'vy', 'vz')], 'cell': [('vx', 'vy', 'vz')]},
    ))
    for frame in (result.points_attributes, result.attributes):
        normal = frame[['nx', 'ny', 'nz']].to_numpy()
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-7)
        np.testing.assert_allclose(normal @ (matrix[:3, :3] @ [1, 0, 1]), 0, atol=2e-7)
    for actual, expected in ((result.attributes, mesh.attributes),
                             (result.points_attributes, mesh.points_attributes)):
        np.testing.assert_array_equal(actual['id'], expected['id'])
        np.testing.assert_array_equal(actual['flag'], expected['flag'])
    np.testing.assert_array_equal(result.points_attributes[['u', 'v']], mesh.points_attributes[['u', 'v']])
    np.testing.assert_array_equal(result.attributes['scalar'], mesh.attributes['scalar'])
    assert source.read_bytes() == original


@pytest.mark.parametrize('failure', ['zero-normal', 'metadata', 'vector-overflow', 'nonfinite-geometry'])
def test_semantic_failure_is_atomic(tmp_path, mesh, failure):
    if failure == 'zero-normal':
        mesh.attributes[['nx', 'ny', 'nz']] = 0.
    elif failure == 'metadata':
        mesh.data_attrs['transform'] = np.eye(4).tolist()
    elif failure == 'vector-overflow':
        mesh.attributes['vz'] = 4.
    else:
        mesh.vertex[0, 0] = np.nan
    source = tmp_path / 'source.le'
    source.write_bytes(mesh.to_binary(order='F'))
    destination = tmp_path / 'output.le'
    destination.write_bytes(b'previous')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    matrix = affine(np.diag([1, 1, 1e38])) if failure == 'vector-overflow' else np.eye(4)
    with pytest.raises(ValueError):
        transform_le(source, destination, matrix, overwrite=True,
                     normal_columns={'cell': [('nx', 'ny', 'nz')]},
                     vector_columns={'cell': [('vx', 'vy', 'vz')]})
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('width', [0, 1, 2, 3, 4, 8])
def test_topologies_and_reflection_policy(width):
    vertices = np.arange(24, dtype=float).reshape(8, 3)
    cells = np.empty((1, 0), dtype=np.int32) if width == 0 else np.arange(width).reshape(1, width)
    mesh = LiquidEarthMesh(vertices, cells, pd.DataFrame(index=range(1)), pd.DataFrame(index=range(8)))
    np.testing.assert_array_equal(transform_mesh(mesh, np.eye(4)).cells, cells)
    if width in (4, 8):
        with pytest.raises(ValueError, match='Reflected volumetric'):
            transform_mesh(mesh, MATRICES[-1])
    else:
        expected = cells[:, [0, 2, 1]] if width == 3 else cells
        np.testing.assert_array_equal(transform_mesh(mesh, MATRICES[-1]).cells, expected)


def test_empty_mesh():
    mesh = LiquidEarthMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32),
                           pd.DataFrame(), pd.DataFrame())
    result = transform_mesh(mesh, np.eye(4))
    assert result.vertex.shape == (0, 3)
    assert result.data_attrs == {'bounds': None}


def test_extreme_unresolved_matrix_rejected_even_for_zero_geometry(mesh):
    mesh.vertex[:] = 0
    linear = [[1e308, 1e308, 0], [-1e308, 1e308, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match='numerically unresolved'):
        transform_mesh(mesh, affine(linear))


@pytest.mark.parametrize('transform', [np.eye(4).tolist(), [], {}, False, 0, 'identity'])
def test_nonnull_transform_metadata_rejected(mesh, transform):
    mesh.data_attrs['transform'] = transform
    with pytest.raises(ValueError, match='Non-null transform metadata'):
        transform_mesh(mesh, np.eye(4))


def test_bounds_use_wire_rounding(mesh):
    result = transform_mesh(mesh, affine(translation=[1 / 7, 1 / 11, 1 / 13]))
    restored = LiquidEarthMesh.from_binary(ops.serialize_le_mesh(result))
    assert result.data_attrs['bounds'] == np.column_stack((restored.vertex.min(0), restored.vertex.max(0))).ravel().tolist()


@pytest.mark.parametrize('alias', ['direct', 'symlink', 'hardlink'])
@pytest.mark.parametrize('overwrite', [False, True])
def test_source_alias_protection(tmp_path, mesh, alias, overwrite):
    source = ops.write_le_mesh(mesh, tmp_path / 'source.le', sources=[])
    destination = source if alias == 'direct' else tmp_path / 'alias.le'
    if alias == 'symlink':
        destination.symlink_to(source)
    elif alias == 'hardlink':
        os.link(source, destination)
    original = source.read_bytes()
    with pytest.raises(ValueError, match='alias'):
        transform_le(source, destination, np.eye(4), overwrite=overwrite)
    assert source.read_bytes() == original
    assert all(p.suffix == '.le' for p in tmp_path.iterdir())


def test_destination_overwrite_and_publication_failure(tmp_path, mesh, monkeypatch):
    source = ops.write_le_mesh(mesh, tmp_path / 'source.le', sources=[])
    original = source.read_bytes()
    destination = tmp_path / 'output.le'
    destination.write_bytes(b'previous')
    with pytest.raises(FileExistsError):
        transform_le(source, destination, np.eye(4))
    assert destination.read_bytes() == b'previous'

    def fail(*args):
        raise OSError('injected publication failure')

    with monkeypatch.context() as patch:
        patch.setattr(ops.os, 'replace', fail)
        with pytest.raises(OSError, match='injected'):
            transform_le(source, destination, np.eye(4), overwrite=True)
    assert destination.read_bytes() == b'previous'
    assert {p.name for p in tmp_path.iterdir()} == {'source.le', 'output.le'}
    transform_le(source, destination, np.eye(4), overwrite=True)
    assert source.read_bytes() == original
    np.testing.assert_array_equal(ops.load_le_mesh(destination).vertex, mesh.vertex)


def test_malformed_source_leaves_destination_unchanged(tmp_path):
    source = tmp_path / 'source.le'
    source.write_bytes(b'bad')
    destination = tmp_path / 'output.le'
    destination.write_bytes(b'previous')
    with pytest.raises(ValueError):
        transform_le(source, destination, np.eye(4), overwrite=True)
    assert source.read_bytes() == b'bad'
    assert destination.read_bytes() == b'previous'
