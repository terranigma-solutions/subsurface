import json
import numpy as np
import pandas as pd

from subsurface.core.utils._le_json import validate_le_json_nesting

_FORMAT_VERSION = 2
MAX_LE_HEADER_BYTES = 16 * 1024 * 1024


def read_le_header(binary_data):
    """Parse a bounded LE prefix/header; no payload is required or decoded."""
    if len(binary_data) < 4:
        raise ValueError("LE header requires a 4-byte length prefix")
    length = int.from_bytes(binary_data[:4], byteorder='little')
    if not 0 < length <= MAX_LE_HEADER_BYTES:
        raise ValueError("LE header length must be between 1 and 16 MiB")
    if len(binary_data) < 4 + length:
        raise ValueError("Truncated LE header")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    header_bytes = bytes(binary_data[4:4 + length])
    validate_le_json_nesting(header_bytes)
    try:
        header = json.loads(header_bytes.decode('utf-8'),
                            object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid LE JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("LE header must be a JSON object")
    return header, 4 + length


def validate_unstructured_layout(header, payload_length=None):
    """Validate supported schemas and return byte offsets without reading arrays.

    Segment offsets are relative to the payload. This validates layout only;
    connectivity values and boolean bytes are checked by ``from_binary``.
    """
    if not isinstance(header, dict):
        raise ValueError("LE header must be a JSON object")
    version = header.get('format_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported LE format_version; expected 1 or 2")
    if 'data_shape' in header:
        raise ValueError("Expected an unstructured LE header, not a structured grid")

    def shape(value, rank, label):
        if (not isinstance(value, (list, tuple)) or len(value) != rank or
                any(type(n) is not int or n < 0 for n in value)):
            raise ValueError(f"{label} must contain {rank} nonnegative integer dimensions")
        return tuple(value)

    vertex_shape = shape(header.get('vertex_shape'), 2, 'vertex_shape')
    if vertex_shape == (0, 0):
        vertex_shape = (0, 3)
    if vertex_shape[1] != 3:
        raise ValueError("vertex_shape must have three XYZ columns")
    wire_cell_shape = shape(header.get('cell_shape'), 2, 'cell_shape')
    cell_shape = wire_cell_shape
    if cell_shape[1] == 0 and cell_shape[0] > vertex_shape[0]:
        raise ValueError("Zero-width point connectivity cannot have more rows than vertices")
    widths = (0, 1, 2, 3, 4, 8)
    legacy_attr_rows = 0
    if version == 1 and 'cell_attr_shape' in header:
        legacy_attr_rows = shape(header['cell_attr_shape'], 2, 'cell_attr_shape')[0]
    # Four/eight indices can also be flattened lines. Attribute rows are the
    # only shipped schema evidence that distinguishes those from a single cell.
    infer_flattened = (version == 1 and cell_shape[0] == 1 and cell_shape[1] > 3 and
                      (cell_shape[1] not in widths or legacy_attr_rows > 1))
    if cell_shape[1] not in widths or infer_flattened:
        if not infer_flattened:
            raise ValueError("Unsupported cell_shape connectivity width")
        candidates = [(cell_shape[1] // width, width) for width in (2, 3)
                      if cell_shape[1] % width == 0]
        if legacy_attr_rows > 0:
            candidates = [candidate for candidate in candidates if candidate[0] == legacy_attr_rows]
        if len(candidates) != 1:
            raise ValueError("Ambiguous or unsupported legacy flattened connectivity; provide cell_attr_shape row count")
        cell_shape = candidates[0]

    data_attrs = header.get('xarray_attrs', {})
    if not isinstance(data_attrs, dict):
        raise ValueError("xarray_attrs must be a JSON object")
    segments = []
    offset = 0

    def valid_names(names):
        # JSON scalar column labels are emitted by the shipped pandas writer,
        # including numeric and null labels. Do not coerce them to strings.
        return (all(name is None or type(name) in (str, int, float, bool) for name in names) and
                pd.Index(names, dtype=object).is_unique)

    def segment(name, association, dtype, dimensions):
        nonlocal offset
        count = 1
        for dimension in dimensions:
            count *= dimension
        byte_length = count * dtype.itemsize
        segments.append(dict(name=name, association=association, dtype=dtype,
                             shape=dimensions, byte_length=byte_length, offset=offset))
        offset += byte_length

    segment('vertex', 'geometry', np.dtype('float32'), vertex_shape)
    segment('cells', 'geometry', np.dtype('int32'), wire_cell_shape)
    for association, rows in (('cell', cell_shape[0]), ('vertex', vertex_shape[0])):
        if version == 2:
            key = association + '_attrs'
            columns = header.get(key, [])
            if not isinstance(columns, list):
                raise ValueError(f"{key} must be a list")
            names = []
            for column in columns:
                if not isinstance(column, dict):
                    raise ValueError(f"{key} entries must be objects")
                if 'name' not in column:
                    raise ValueError(f"{key} requires column names")
                name = column['name']
                names.append(name)
                dimensions = shape(column.get('shape'), 1, key + ' column shape')
                if dimensions != (rows,):
                    raise ValueError(f"{key} column row count does not match geometry")
                try:
                    if not isinstance(column.get('dtype'), str):
                        raise TypeError()
                    dtype = np.dtype(column['dtype'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Unsupported {key} dtype") from exc
                if not (dtype.kind in 'iu' and dtype.itemsize in (1, 2, 4, 8) or
                        dtype.kind == 'f' and dtype.itemsize in (2, 4, 8) or
                        dtype.kind == 'b' and dtype.itemsize == 1):
                    raise ValueError(f"Unsupported {key} numeric dtype: {dtype}")
                declared = column.get('byte_length')
                expected = rows * dtype.itemsize
                if type(declared) is not int or declared != expected:
                    raise ValueError(f"{key} byte_length does not match shape and dtype")
                segment(name, association, dtype, dimensions)
            if not valid_names(names):
                raise ValueError(f"{key} requires unique JSON scalar column names")
        else:
            key = association + '_attr'
            dimensions = shape(header.get(key + '_shape', [0, 0]), 2, key + '_shape')
            if dimensions != (0, 0) and dimensions[0] != rows:
                raise ValueError(f"{key} row count does not match geometry")
            names = header.get(key + '_names', [])
            if (not isinstance(names, list) or len(names) != dimensions[1] or
                    not valid_names(names)):
                raise ValueError(f"{key} requires unique JSON scalar names matching its columns")
            types = header.get(key + '_types')
            # Shipped empty legacy dataframes can retain one descriptive dtype.
            type_counts = (0, 1) if dimensions == (0, 0) else (len(names),)
            if types is not None and (not isinstance(types, list) or len(types) not in type_counts):
                raise ValueError(f"{key}_types must match its columns (legacy payload is float32)")
            segment(key, association, np.dtype('float32'), dimensions)
            segments[-1]['names'] = names
    if payload_length is not None:
        if type(payload_length) is not int or payload_length < 0 or payload_length != offset:
            raise ValueError(f"LE payload length mismatch: expected {offset} bytes, got {payload_length}")
    return dict(format_version=version, vertex_shape=vertex_shape,
                wire_cell_shape=wire_cell_shape, cell_shape=cell_shape,
                segments=segments, payload_length=offset, data_attrs=data_attrs)


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


def _column_metadata(df: pd.DataFrame, order: str, pack_integral_floats: bool = True) -> list[dict]:
    """Describe serialized columns. Integral float columns are stored as int64
    unless ``pack_integral_floats`` is False (temporal snapshots need a stored
    dtype that does not depend on the values)."""
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
            if pack_integral_floats and np.all(np.mod(values, 1) == 0):
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


def _serialize_column(values: np.ndarray, pack_integral_floats: bool = True) -> bytes:
    if np.issubdtype(values.dtype, np.integer):
        return values.tobytes('C')
    elif np.issubdtype(values.dtype, np.bool_):
        return values.astype(np.uint8).tobytes('C')
    else:
        if pack_integral_floats and np.issubdtype(values.dtype, np.floating):
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

    @classmethod
    def from_binary(cls, binary_data, order='F'):
        """Decode a validated mesh; geometry order is supplied externally (F or C)."""
        if order not in ('F', 'C'):
            raise ValueError("LE array order must be 'F' or 'C'")
        header, payload_offset = read_le_header(binary_data)
        layout = validate_unstructured_layout(header, len(binary_data) - payload_offset)
        body = memoryview(binary_data)[payload_offset:]
        geometry = {}
        attributes = {'cell': {}, 'vertex': {}}
        frames = {}
        for segment in layout['segments']:
            start = segment['offset']
            raw = body[start:start + segment['byte_length']]
            dtype = segment['dtype']
            if dtype.kind == 'b':
                if np.any(np.frombuffer(raw, dtype=np.uint8) > 1):
                    raise ValueError("Boolean attribute payload must contain only 0 or 1")
            values = np.frombuffer(raw, dtype=dtype)
            association = segment['association']
            if association == 'geometry':
                geometry[segment['name']] = values.reshape(segment['shape'], order=order)
            elif layout['format_version'] == 1:
                frames[association] = pd.DataFrame(values.reshape(segment['shape'], order=order),
                                                   columns=segment['names'])
            else:
                # pandas/xarray operations require native-endian numeric buffers.
                if not dtype.isnative:
                    values = values.astype(dtype.newbyteorder('='))
                attributes[association][segment['name']] = values
        cells = geometry['cells']
        if layout['wire_cell_shape'] != layout['cell_shape']:
            cells = cells.reshape(layout['cell_shape'], order='C')
        if cells.size and (np.any(cells < 0) or np.any(cells >= layout['vertex_shape'][0])):
            raise ValueError("Connectivity indices must be in range [0, n_points)")
        for association, rows in (('cell', cells.shape[0]), ('vertex', geometry['vertex'].shape[0])):
            if association not in frames:
                frames[association] = pd.DataFrame(attributes[association], index=pd.RangeIndex(rows))
            elif frames[association].shape == (0, 0):
                frames[association] = pd.DataFrame(index=pd.RangeIndex(rows))
        return cls(vertex=geometry['vertex'], cells=cells, attributes=frames['cell'],
                   points_attributes=frames['vertex'], data_attrs=layout['data_attrs'])
