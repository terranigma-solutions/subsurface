import json
import numpy as np
import pandas as pd

_FORMAT_VERSION = 2


def _validate_attribute_dataframe(df: pd.DataFrame, attr_name: str):
    for col in df.columns:
        series = df[col]
        if np.issubdtype(series.dtype, np.integer) or np.issubdtype(series.dtype, np.bool_):
            continue
        if np.issubdtype(series.dtype, np.floating):
            continue
        raise TypeError(
            f"Column '{col}' in {attr_name} has dtype '{series.dtype}' which is not a supported "
            f"numeric or boolean type. Only integer, float, and bool columns are allowed."
        )


def _column_metadata(df: pd.DataFrame, order: str) -> list[dict]:
    meta = []
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        values = series.to_numpy()

        if np.issubdtype(dtype, np.integer):
            stored_dtype = str(dtype)
            byte_length = values.nbytes
        elif np.issubdtype(dtype, np.bool_):
            stored_dtype = 'bool'
            byte_length = values.size
        elif np.issubdtype(dtype, np.floating):
            if np.all(np.mod(values, 1) == 0):
                int_vals = values.astype(np.int64)
                if np.all(int_vals.astype(np.float64) == values):
                    stored_dtype = 'int64'
                    byte_length = values.size * 8
                    meta.append({
                        "name": col,
                        "dtype": stored_dtype,
                        "shape": list(values.shape),
                        "byte_length": byte_length,
                    })
                    continue
            stored_dtype = 'float32'
            byte_length = values.size * 4
        else:
            stored_dtype = str(dtype)
            byte_length = values.nbytes

        meta.append({
            "name": col,
            "dtype": stored_dtype,
            "shape": list(values.shape),
            "byte_length": byte_length,
        })
    return meta


def _serialize_column(values: np.ndarray) -> bytes:
    if np.issubdtype(values.dtype, np.integer):
        return values.tobytes('C')
    elif np.issubdtype(values.dtype, np.bool_):
        return values.astype(np.uint8).tobytes('C')
    else:
        if np.issubdtype(values.dtype, np.floating):
            if np.all(np.mod(values, 1) == 0):
                int_vals = values.astype(np.int64)
                if np.all(int_vals.astype(np.float64) == values):
                    return int_vals.tobytes('C')
        return values.astype(np.float32).tobytes('C')


def _filter_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    filtered = {}
    for col in df.columns:
        series = df[col]
        if np.issubdtype(series.dtype, np.integer) or np.issubdtype(series.dtype, np.bool_):
            filtered[col] = series
        elif np.issubdtype(series.dtype, np.floating):
            filtered[col] = series
        elif series.dtype == object:
            if not series.notna().any():
                continue
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().equals(series.notna()):
                filtered[col] = converted
    return pd.DataFrame(filtered, index=df.index)


class LiquidEarthMesh:
    def __init__(self, vertex=None, cells=None, attributes=None, points_attributes=None, data_attrs=None):
        self.vertex = vertex
        self.cells = cells
        self.attributes = attributes
        self.points_attributes = points_attributes
        self.data_attrs = data_attrs if data_attrs is not None else {}

        if self.attributes is not None and not self.attributes.empty:
            _validate_attribute_dataframe(self.attributes, 'cell_attrs')
        if self.points_attributes is not None and not self.points_attributes.empty:
            _validate_attribute_dataframe(self.points_attributes, 'vertex_attrs')

    def to_binary(self, order='C') -> bytes:
        header_ = self._set_binary_header()
        header_json = json.dumps(header_)
        header_json_bytes = header_json.encode('utf-8')
        header_json_length = len(header_json_bytes)
        header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
        body_ = self._to_bytearray(order)
        return header_json_length_bytes + header_json_bytes + body_

    def _set_binary_header(self):
        header = {
            "format_version": _FORMAT_VERSION,
            "vertex_shape": self.vertex.shape if self.vertex is not None else [0, 0],
            "cell_shape": self.cells.shape if self.cells is not None else [0, 0],
            "cell_attrs": _column_metadata(self.attributes, 'C') if not self.attributes.empty else [],
            "vertex_attrs": _column_metadata(self.points_attributes, 'C') if not self.points_attributes.empty else [],
            "xarray_attrs": self.data_attrs,
        }
        return header

    def _to_bytearray(self, order='C') -> bytes:
        parts = []
        if self.vertex is not None:
            parts.append(self.vertex.astype('float32').tobytes(order))
        if self.cells is not None:
            parts.append(self.cells.astype('int32').tobytes(order))
        if not self.attributes.empty:
            for col in self.attributes.columns:
                parts.append(_serialize_column(self.attributes[col].to_numpy()))
        if not self.points_attributes.empty:
            for col in self.points_attributes.columns:
                parts.append(_serialize_column(self.points_attributes[col].to_numpy()))
        return b''.join(parts)

    @staticmethod
    def _read_attr_v1(body: bytes, offset: int, header: dict, attr_key: str, order: str = 'F'
                     ) -> tuple[pd.DataFrame | None, int]:
        shape = header.get(attr_key + '_shape', [0, 0])
        if shape is None or shape[0] <= 0 or shape[1] <= 0:
            return None, offset
        num_attrs = int(np.prod(shape))
        num_bytes = num_attrs * 4
        values = np.frombuffer(body[offset:offset + num_bytes], dtype=np.float32, count=num_attrs)
        offset += num_bytes
        values = values.reshape(shape[:2], order=order)
        names = header.get(attr_key + '_names', [])
        return pd.DataFrame(values, columns=names), offset

    @staticmethod
    def _read_attr_v2(body: bytes, offset: int, columns_meta: list[dict]
                      ) -> tuple[pd.DataFrame, int]:
        columns = {}
        for meta in columns_meta:
            dtype_str = meta["dtype"]
            count = int(np.prod(meta["shape"]))
            byte_length = meta["byte_length"]

            if dtype_str == 'bool':
                raw = np.frombuffer(body[offset:offset + byte_length], dtype=np.uint8, count=count)
                values = raw.astype(bool)
            else:
                values = np.frombuffer(body[offset:offset + byte_length], dtype=np.dtype(dtype_str), count=count)

            columns[meta["name"]] = values
            offset += byte_length

        return pd.DataFrame(columns), offset

    @classmethod
    def from_binary(cls, binary_data, order='F'):
        header_length_bytes = binary_data[:4]
        header_length = int.from_bytes(header_length_bytes, byteorder='little')
        header_json_bytes = binary_data[4:4 + header_length]
        header = json.loads(header_json_bytes.decode('utf-8'))
        body = binary_data[4 + header_length:]
        offset = 0
        format_version = header.get("format_version", 1)

        vertex_shape = header.get('vertex_shape', [0, 0])
        if vertex_shape[0] > 0 and vertex_shape[1] > 0:
            num_vertices = int(np.prod(vertex_shape))
            num_bytes = num_vertices * 4
            vertex = np.frombuffer(body[offset:offset + num_bytes], dtype=np.float32, count=num_vertices)
            offset += num_bytes
            vertex = vertex.reshape(vertex_shape, order=order)
        else:
            vertex = None

        cell_shape = header.get('cell_shape', [0, 0])
        if cell_shape[0] > 0 and cell_shape[1] > 0:
            num_cells = int(np.prod(cell_shape))
            num_bytes = num_cells * 4
            cells = np.frombuffer(body[offset:offset + num_bytes], dtype=np.int32, count=num_cells)
            offset += num_bytes
            cells = cells.reshape(cell_shape, order=order)

            # Auto-reshape legacy flattened cells
            if format_version == 1 and cells.shape[0] == 1 and cells.shape[1] > 3:
                if cells.shape[1] % 3 == 0:
                    cells = cells.reshape((-1, 3), order='C')
                elif cells.shape[1] % 2 == 0:
                    cells = cells.reshape((-1, 2), order='C')
        else:
            cells = None

        if format_version >= 2:
            cell_attrs_meta = header.get('cell_attrs', [])
            if cell_attrs_meta:
                cell_attr_values, offset = cls._read_attr_v2(body, offset, cell_attrs_meta)
            else:
                cell_attr_values = None

            vertex_attrs_meta = header.get('vertex_attrs', [])
            if vertex_attrs_meta:
                vertex_attr_values, offset = cls._read_attr_v2(body, offset, vertex_attrs_meta)
            else:
                vertex_attr_values = None
        else:
            cell_attr_values, offset = cls._read_attr_v1(body, offset, header, 'cell_attr', order=order)
            vertex_attr_values, offset = cls._read_attr_v1(body, offset, header, 'vertex_attr', order=order)

        data_attrs = header.get('xarray_attrs', {})

        return cls(
            vertex=vertex,
            cells=cells,
            attributes=cell_attr_values,
            points_attributes=vertex_attr_values,
            data_attrs=data_attrs,
        )

