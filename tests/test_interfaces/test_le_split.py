"""Offline topology, dtype, and publication safety coverage for LE splitting."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import subsurface.api.le_split as split_module
from subsurface.api._le_file_ops import load_le_mesh, write_le_mesh
from subsurface.api.le_split import split_le
from subsurface.core.structs.base_structures._liquid_earth_mesh import LiquidEarthMesh


def source_mesh(tmp_path, *, cells=None, ids=None, point_ids=None, metadata=None):
    vertex = np.arange(18, dtype=np.float32).reshape(6, 3)
    if cells is None:
        cells = np.array([[0, 2, 3], [2, 3, 4], [0, 3, 4]], dtype=np.int32)
    attrs = pd.DataFrame(index=range(len(cells)))
    if ids is not None:
        attrs['id'] = ids
    attrs['value'] = np.arange(len(cells), dtype=np.float32) + .25
    attrs['enabled'] = np.arange(len(cells)) % 2 == 0
    points = pd.DataFrame({
        'large': np.arange(6, dtype=np.int64) + 2 ** 60,
        'weight': np.arange(6, dtype=np.float32) + .125,
        'active': [True, False, True, False, True, False],
    })
    if point_ids is not None:
        points['id'] = point_ids
    mesh = LiquidEarthMesh(vertex, cells, attrs, points,
                           metadata if metadata is not None else {'crs': 'local', 'units': 'm'})
    # Fixtures intentionally permit empty named schemas and unsafe geometry so
    # splitting, rather than the fixture writer, can exercise their rejection.
    path = tmp_path / 'source.le'
    path.write_bytes(mesh.to_binary(order='F'))
    return path


def split(source, directory, association='cell', attribute='id'):
    return split_le(source, directory, object_attribute=attribute, association=association)


def test_shared_unused_vertices_sparse_int64_and_all_attributes(tmp_path):
    low, high = 2 ** 60 + 1, 2 ** 60 + 3
    source = source_mesh(tmp_path, ids=np.array([high, low, high], dtype=np.int64),
                         metadata={'crs': 'local', 'nested': {'units': ['m']},
                                   'le_tools': {'operation': 'prior'}})
    before = source.read_bytes()
    original = load_le_mesh(source)
    mapping = split(source, tmp_path)
    assert list(mapping) == [low, high]
    assert [path.name for path in mapping.values()] == ['object_000000.le', 'object_000001.le']
    for object_id, cell_rows, point_rows in ((low, [1], [2, 3, 4]),
                                            (high, [0, 2], [0, 2, 3, 4])):
        output = load_le_mesh(mapping[object_id])
        np.testing.assert_array_equal(output.vertex, original.vertex[point_rows])
        np.testing.assert_array_equal(output.vertex[output.cells],
                                      original.vertex[original.cells[cell_rows]])
        pd.testing.assert_frame_equal(output.attributes,
                                      original.attributes.iloc[cell_rows].reset_index(drop=True))
        pd.testing.assert_frame_equal(output.points_attributes,
                                      original.points_attributes.iloc[point_rows].reset_index(drop=True))
        assert output.attributes['id'].dtype == np.dtype('int64')
        assert output.cells.min() >= 0 and output.cells.max() < len(output.vertex)
        assert output.data_attrs['nested'] == original.data_attrs['nested']
        assert output.data_attrs['crs'] == 'local'
        assert output.data_attrs['le_tools'] == dict(
            operation='split', source=str(source.resolve()), object_attribute='id',
            association='cell', object_id=object_id, previous={'operation': 'prior'})
    assert source.read_bytes() == before


@pytest.mark.parametrize('width', [2, 3, 4, 8])
def test_mesh_connectivity_widths(tmp_path, width):
    cells = (np.arange(2 * width).reshape(2, width) % 5).astype(np.int32)
    source = source_mesh(tmp_path, cells=cells, ids=[9, -4])
    mapping = split(source, tmp_path)
    assert list(mapping) == [-4, 9]
    for object_id, row in ((-4, 1), (9, 0)):
        output = load_le_mesh(mapping[object_id])
        np.testing.assert_array_equal(output.vertex[output.cells],
                                      load_le_mesh(source).vertex[cells[[row]]])


def test_fractional_ids_are_sorted_and_original(tmp_path):
    source = source_mesh(tmp_path, ids=[4.25, -2.5, 4.25])
    mapping = split(source, tmp_path)
    assert list(mapping) == [-2.5, 4.25]
    for object_id, path in mapping.items():
        assert (load_le_mesh(path).attributes['id'] == object_id).all()


@pytest.mark.parametrize('width', [0, 1])
def test_point_cloud_connectivity_and_cell_attributes(tmp_path, width):
    cells = (np.empty((6, 0), dtype=np.int32) if width == 0 else
             np.array([[4], [0], [4], [3], [2], [5]], dtype=np.int32))
    source = source_mesh(tmp_path, cells=cells, point_ids=[10, 20, 10, 20, 10, 20])
    original = load_le_mesh(source)
    mapping = split(source, tmp_path, 'point')
    for object_id, point_rows in ((10, [0, 2, 4]), (20, [1, 3, 5])):
        output = load_le_mesh(mapping[object_id])
        cell_rows = (point_rows if width == 0 else
                     np.flatnonzero(np.isin(cells[:, 0], point_rows)))
        np.testing.assert_array_equal(output.vertex, original.vertex[point_rows])
        pd.testing.assert_frame_equal(output.attributes,
                                      original.attributes.iloc[cell_rows].reset_index(drop=True))
        pd.testing.assert_frame_equal(output.points_attributes,
                                      original.points_attributes.iloc[point_rows].reset_index(drop=True))
        assert output.cells.shape == (len(cell_rows), width)
        if width:
            np.testing.assert_array_equal(output.vertex[output.cells],
                                          original.vertex[cells[cell_rows]])


def test_points_without_cells(tmp_path):
    source = source_mesh(tmp_path, cells=np.empty((0, 0), dtype=np.int32),
                         point_ids=[0, 1, 0, 1, 0, 1])
    mapping = split(source, tmp_path, 'point')
    assert all(load_le_mesh(path).cells.shape == (0, 0) for path in mapping.values())


@pytest.mark.parametrize('bad_ids', [[True, False, True], [1., np.nan, 1.], [1., np.inf, 1.]])
def test_invalid_ids_do_not_write(tmp_path, bad_ids):
    source = source_mesh(tmp_path, ids=bad_ids)
    before = source.read_bytes()
    with pytest.raises(ValueError, match='Grouping IDs'):
        split(source, tmp_path)
    assert list(tmp_path.iterdir()) == [source]
    assert source.read_bytes() == before


@pytest.mark.parametrize('association,attribute', [('cell', 'missing'), ('point', 'id'),
                                                 ('vertex', 'id'), ('cell', '')])
def test_missing_grouping_and_wrong_associations(tmp_path, association, attribute):
    source = source_mesh(tmp_path, ids=[1, 2, 1])
    with pytest.raises(ValueError):
        split(source, tmp_path, association, attribute)
    assert list(tmp_path.iterdir()) == [source]


def test_empty_explicit_grouping(tmp_path):
    header = dict(format_version=2, vertex_shape=[0, 3], cell_shape=[0, 3],
                  cell_attrs=[dict(name='id', dtype='int64', shape=[0], byte_length=0)],
                  vertex_attrs=[], xarray_attrs={})
    encoded = json.dumps(header).encode('utf-8')
    source = tmp_path / 'empty.le'
    source.write_bytes(len(encoded).to_bytes(4, 'little') + encoded)
    assert split(source, tmp_path) == {}
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize('collision', ['file', 'directory', 'dangling', 'symlink', 'hardlink'])
def test_preflight_all_collisions_and_aliases(tmp_path, collision):
    source = source_mesh(tmp_path, ids=[1, 2, 1])
    before = source.read_bytes()
    destination = tmp_path / 'object_000001.le'
    if collision == 'file':
        destination.write_bytes(b'unrelated')
    elif collision == 'directory':
        destination.mkdir()
    elif collision == 'dangling':
        destination.symlink_to(tmp_path / 'absent')
    elif collision == 'symlink':
        destination.symlink_to(source)
    else:
        os.link(source, destination)
    with pytest.raises((ValueError, FileExistsError)):
        split(source, tmp_path)
    assert not (tmp_path / 'object_000000.le').exists()
    assert os.path.lexists(destination)
    assert source.read_bytes() == before


def test_output_directory_must_exist(tmp_path):
    source = source_mesh(tmp_path, ids=[1, 2, 1])
    with pytest.raises(NotADirectoryError):
        split(source, tmp_path / 'missing')
    assert not (tmp_path / 'missing').exists()


def test_partial_zero_width_rows_rejected(tmp_path):
    source = source_mesh(tmp_path, cells=np.empty((2, 0), dtype=np.int32),
                         point_ids=[0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match='Zero-width'):
        split(source, tmp_path, 'point')
    assert list(tmp_path.iterdir()) == [source]


def test_empty_cell_attribute_subset_rejected_before_any_publish(tmp_path):
    source = source_mesh(tmp_path, cells=np.array([[0], [2]], dtype=np.int32),
                         point_ids=[0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match='Empty cell attribute'):
        split(source, tmp_path, 'point')
    assert list(tmp_path.iterdir()) == [source]


def test_later_invalid_geometry_is_preflighted(tmp_path):
    source = source_mesh(tmp_path, ids=[1, 2, 1],
                         cells=np.array([[0, 2, 3], [2, 3, 5], [0, 3, 4]], dtype=np.int32))
    mesh = load_le_mesh(source)
    mesh.vertex = mesh.vertex.copy()
    mesh.vertex[5, 0] = np.inf
    source.write_bytes(mesh.to_binary(order='F'))
    with pytest.raises(ValueError, match='Geometry must be finite'):
        split(source, tmp_path)
    assert list(tmp_path.iterdir()) == [source]


def test_publication_failure_rolls_back_only_created_files(tmp_path, monkeypatch):
    source = source_mesh(tmp_path, ids=[1, 2, 1])
    before = source.read_bytes()
    unrelated = tmp_path / 'keep.txt'
    unrelated.write_bytes(b'keep')

    def fail_second(mesh, destination, **kwargs):
        if destination.name == 'object_000001.le':
            raise OSError('simulated publication failure')
        return write_le_mesh(mesh, destination, **kwargs)

    monkeypatch.setattr(split_module, 'write_le_mesh', fail_second)
    with pytest.raises(OSError, match='simulated'):
        split(source, tmp_path)
    assert set(tmp_path.iterdir()) == {source, unrelated}
    assert source.read_bytes() == before
    assert unrelated.read_bytes() == b'keep'


def test_racing_destination_is_not_deleted(tmp_path, monkeypatch):
    source = source_mesh(tmp_path, ids=[1, 2, 1])
    collision = tmp_path / 'object_000001.le'

    def collide_second(mesh, destination, **kwargs):
        if destination == collision:
            destination.write_bytes(b'other writer')
        return write_le_mesh(mesh, destination, **kwargs)

    monkeypatch.setattr(split_module, 'write_le_mesh', collide_second)
    with pytest.raises(FileExistsError):
        split(source, tmp_path)
    assert set(tmp_path.iterdir()) == {source, collision}
    assert collision.read_bytes() == b'other writer'


def test_rollback_does_not_delete_replaced_output(tmp_path, monkeypatch):
    source = source_mesh(tmp_path, ids=[1, 2, 1])
    first = tmp_path / 'object_000000.le'
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'unrelated replacement')

    def replace_then_fail(mesh, destination, **kwargs):
        if destination.name == 'object_000001.le':
            os.replace(replacement, first)
            raise OSError('publication failure after concurrent replacement')
        return write_le_mesh(mesh, destination, **kwargs)

    monkeypatch.setattr(split_module, 'write_le_mesh', replace_then_fail)
    with pytest.raises(OSError, match='publication failure'):
        split(source, tmp_path)
    assert set(tmp_path.iterdir()) == {source, first}
    assert first.read_bytes() == b'unrelated replacement'


def test_temporary_write_failure_cleans_and_rolls_back(tmp_path, monkeypatch):
    source = source_mesh(tmp_path, ids=[1, 2, 1])
    original_link = os.link

    def fail_second_link(temporary, destination):
        if destination.name == 'object_000001.le':
            raise OSError('link failure')
        original_link(temporary, destination)

    monkeypatch.setattr(os, 'link', fail_second_link)
    with pytest.raises(OSError, match='link failure'):
        split(source, tmp_path)
    assert list(tmp_path.iterdir()) == [source]


def test_reserved_metadata_requires_dictionary(tmp_path):
    source = source_mesh(tmp_path, ids=[1, 2, 1], metadata={'le_tools': 'not provenance'})
    with pytest.raises(ValueError, match='le_tools'):
        split(source, tmp_path)
    assert list(tmp_path.iterdir()) == [source]
