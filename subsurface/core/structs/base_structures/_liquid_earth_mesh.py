import json
import numpy as np
import pandas as pd


class LiquidEarthMesh:
    def __init__(self, vertex=None, cells=None, attributes=None, points_attributes=None, data_attrs=None):
        self.vertex = vertex  # Expected to be a numpy array of shape (N, 3)
        self.cells = cells  # Expected to be a numpy array of shape (M, K)
        self.attributes = attributes
        self.points_attributes = points_attributes
        self.data_attrs = data_attrs if data_attrs is not None else {}

    def to_binary(self, order='F') -> bytes:
        body_ = self._to_bytearray(order)
        header_ = self._set_binary_header()
        header_json = json.dumps(header_)
        header_json_bytes = header_json.encode('utf-8')
        header_json_length = len(header_json_bytes)
        header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
        file = header_json_length_bytes + header_json_bytes + body_
        return file

    def _set_binary_header(self):
        header = {
                "vertex_shape"     : self.vertex.shape if self.vertex is not None else [0, 0],
                "cell_shape"       : self.cells.shape if self.cells is not None else [0, 0],
                "cell_attr_shape"  : self.attributes.shape if not self.attributes.empty else [0, 0],
                "vertex_attr_shape": self.points_attributes.shape if not self.points_attributes.empty else [0, 0],
                "cell_attr_names"  : self.attributes.columns.tolist() if not self.attributes.empty else [],
                "cell_attr_types"  : self.attributes.dtypes.astype(str).tolist() if not self.attributes.empty else [],
                "vertex_attr_names": self.points_attributes.columns.tolist() if not self.points_attributes.empty else [],
                "vertex_attr_types": self.points_attributes.dtypes.astype(str).tolist() if not self.points_attributes.empty else [],
                "xarray_attrs"     : self.data_attrs
        }
        return header

    def _to_bytearray(self, order):
        parts = []
        if self.vertex is not None:
            vertex_bytes = self.vertex.astype('float32').tobytes(order)
            parts.append(vertex_bytes)
        if self.cells is not None:
            cells_bytes = self.cells.astype('int32').tobytes(order)
            parts.append(cells_bytes)
        if not self.attributes.empty:
            cell_attr_bytes = self.attributes.values.astype('float32').tobytes(order)
            parts.append(cell_attr_bytes)
        if not self.points_attributes.empty:
            vertex_attr_bytes = self.points_attributes.values.astype('float32').tobytes(order)
            parts.append(vertex_attr_bytes)
        bytearray_le = b''.join(parts)
        return bytearray_le

    @classmethod
    def from_binary(cls, binary_data, order='F'):
        # Read header length
        header_length_bytes = binary_data[:4]
        header_length = int.from_bytes(header_length_bytes, byteorder='little')
        # Read header
        header_json_bytes = binary_data[4:4 + header_length]
        header_json = header_json_bytes.decode('utf-8')
        header = json.loads(header_json)
        # Read body
        body = binary_data[4 + header_length:]
        offset = 0

        # Parse vertices
        vertex_shape = header['vertex_shape']
        if vertex_shape[0] > 0 and vertex_shape[1] > 0:
            num_vertices = np.prod(vertex_shape)
            num_bytes = num_vertices * 4  # float32
            vertex = np.frombuffer(body[offset:offset + num_bytes], dtype=np.float32, count=num_vertices)
            offset += num_bytes
            vertex = vertex.reshape(vertex_shape, order=order)
        else:
            vertex = None

        # Parse cells
        cell_shape = header['cell_shape']
        if cell_shape[0] > 0 and cell_shape[1] > 0:
            num_cells = np.prod(cell_shape)
            num_bytes = num_cells * 4  # int32
            cells = np.frombuffer(body[offset:offset + num_bytes], dtype=np.int32, count=num_cells)
            offset += num_bytes
            cells = cells.reshape(cell_shape, order=order)
        else:
            cells = None

        # Parse cell attributes
        attributes = pd.DataFrame()
        cell_attr_shape = header['cell_attr_shape']
        if cell_attr_shape[0] > 0 and cell_attr_shape[1] > 0:
            num_attrs = np.prod(cell_attr_shape)
            num_bytes = num_attrs * 4  # float32
            cell_attr_values = np.frombuffer(body[offset:offset + num_bytes], dtype=np.float32, count=num_attrs)
            offset += num_bytes
            cell_attr_values = cell_attr_values.reshape(cell_attr_shape, order=order)
            attr_names = header['cell_attr_names']
            attributes = pd.DataFrame(cell_attr_values, columns=attr_names)
        else:
            attributes = None

        # Parse vertex attributes
        points_attributes = pd.DataFrame()
        vertex_attr_shape = header['vertex_attr_shape']
        if vertex_attr_shape[0] > 0 and vertex_attr_shape[1] > 0:
            num_attrs = np.prod(vertex_attr_shape)
            num_bytes = num_attrs * 4  # float32
            vertex_attr_values = np.frombuffer(body[offset:offset + num_bytes], dtype=np.float32, count=num_attrs)
            offset += num_bytes
            vertex_attr_values = vertex_attr_values.reshape(vertex_attr_shape, order=order)
            attr_names = header['vertex_attr_names']
            points_attributes = pd.DataFrame(vertex_attr_values, columns=attr_names)
        else:
            points_attributes = None

        data_attrs = header.get('xarray_attrs', {})

        return cls(vertex=vertex, cells=cells, attributes=attributes, points_attributes=points_attributes, data_attrs=data_attrs)

