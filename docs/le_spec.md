# **LiquidEarth Mesh (`.le`) File Format Specification**

## **Overview**

A `.le` file comprises:

1. A **4-byte unsigned little-endian integer** specifying the length (in bytes) of a JSON header.
2. A **JSON header** (UTF-8 encoded) that contains metadata about the mesh or grid.  
3. A **binary payload** – a concatenation of one or more numeric arrays.

These files are produced by either:

- **Unstructured** meshes (via the `UnstructuredData` class)  
- **Structured** grids (via the `StructuredData` class)  

While both produce the same *high-level layout*, the **contents** of the JSON header and the **structure** of the binary payload differ between unstructured and structured data.

---

## **High-Level Layout**

```
+------------------+------------------------+---------------------------+
| 4-byte integer   | JSON header (UTF-8)   | Binary payload            |
| (little-endian)  |                       | (arrays of numeric data)  |
+------------------+------------------------+---------------------------+
```

1. **4-byte unsigned integer** (little-endian)
   - Denotes how many bytes are used by the JSON header. Call this `N_header_bytes`.

2. **JSON header** (UTF-8, length = `N_header_bytes`)  
   - Deserialized via `json.loads(...)` in Python.  
   - Contains metadata describing how to interpret the binary payload:
     - For **unstructured** data, includes shapes for vertices/cells, attribute shapes, and names/types of attributes, etc.  
     - For **structured** data, includes shape of the grid, bounding information, data type, etc.

3. **Binary payload**  
   - Immediately follows the header.  
   - Contains numeric geometry and attributes; array order is supplied externally, not recorded in the header.

---

## **A) Unstructured Meshes**

### **Version 2 Layout**

The current writers emit `format_version: 2`. The payload has no padding and contains, in order:

1. Vertices: `float32`, shape `[n_points, 3]`.
2. Connectivity: `int32`, shape `[n_cells, width]`.
3. Each cell attribute column, in `cell_attrs` list order.
4. Each vertex attribute column, in `vertex_attrs` list order.

Geometry is cast to **native-endian** `float32`/`int32` by the writer and read using native-endian dtypes. On little-endian platforms these geometry bytes are little-endian, but there is no byte-order header field and the entire body is not specified as fixed little-endian. Attribute dtype strings are interpreted by NumPy; explicit endian dtypes such as `"<i2"` and `">f4"` are recognized and normalized to native-endian numeric buffers after decoding for pandas/xarray use. Width, signedness, and values are unchanged.

`UnstructuredData.to_binary()` and `to_binary_legacy()` default to `order='F'`, as do `UnstructuredData.from_binary_le()`, `from_binary_le_legacy()`, and `LiquidEarthMesh.from_binary()`. The low-level `LiquidEarthMesh.to_binary()` defaults to `order='C'`. Readers accept `F` or `C`; **pass `order='C'` explicitly for C-order exports**. Order cannot be detected from the file. It controls geometry reshaping and legacy attribute matrices; v2 attributes are one-dimensional columns.

### **Version 2 Header**

This valid example describes three vertices, one triangle, one cell attribute, and two vertex attributes:

```json
{
  "format_version": 2,
  "vertex_shape": [3, 3],
  "cell_shape": [1, 3],
  "cell_attrs": [
    {"name": "rock_type", "dtype": "int16", "shape": [1], "byte_length": 2}
  ],
  "vertex_attrs": [
    {"name": "density", "dtype": "float32", "shape": [3], "byte_length": 12},
    {"name": "active", "dtype": "bool", "shape": [3], "byte_length": 3}
  ],
  "xarray_attrs": {
    "description": "A sample triangular mesh",
    "coordinate_system": "EPSG:4326"
  }
}
```

The payload is 65 bytes: vertices at offset 0 (36 bytes), cells at 36 (12 bytes), `rock_type` at 48 (2 bytes), `density` at 50 (12 bytes), and `active` at 62 (3 bytes). These offsets are relative to the payload, not the start of the file.

- Shapes contain nonnegative integer dimensions. Vertices require three XYZ columns; `[0, 0]` is accepted and normalized to `[0, 3]`.
- Canonical connectivity widths are `0`, `1`, `2`, `3`, `4`, and `8`. Widths `0`/`1` represent points, `2` lines, `3` triangles, `4` tetrahedra, and `8` hexahedra. Empty connectivity shapes, including `[0, 0]`, are preserved. Zero-width connectivity cannot declare more rows than vertices.
- Attribute lists may be absent or empty. Each entry requires a unique scalar `name` within its association, a one-dimensional `shape` matching its geometry row count, a numeric `dtype`, and an integer `byte_length` equal to row count times dtype item size. String names are conventional; numeric, boolean, and null column labels emitted by the shipped writer are retained, not coerced to strings. Python JSON's NaN labels are also accepted for compatibility with pandas missing column labels.
- Accepted attribute dtypes are signed/unsigned integers of 1, 2, 4, or 8 bytes; floats of 2, 4, or 8 bytes; and one-byte booleans. Boolean payload bytes must be `0` or `1`. Strings, objects, complex numbers, and datetime dtypes are not wire attribute types.
- `xarray_attrs` is an optional JSON object (default `{}`), containing JSON-serializable dataset metadata.

### **Reader Validation**

The entry points in `_liquid_earth_mesh.py` separate header inspection from payload decoding:

- `read_le_header(binary_data)` reads the unsigned little-endian prefix, requires `1 <= N_header_bytes <= 16 MiB`, and returns `(header, payload_offset)`. It requires the complete header but no payload, rejects truncated/invalid UTF-8 or JSON, duplicate JSON keys, and non-object headers.
- `validate_unstructured_layout(header, payload_length=None)` validates the v1/v2 schema without reading arrays. It rejects unsupported versions and structured headers (`data_shape`). It returns normalized geometry shapes, the wire connectivity shape, metadata, total payload length, and ordered segments with `name`, `association`, `dtype`, `shape`, `byte_length`, and payload-relative `offset`. If supplied, `payload_length` must match exactly, so trailing or missing bytes are rejected.
- Header/layout validation does **not** validate connectivity values or boolean bytes. `LiquidEarthMesh.from_binary()` performs those checks while decoding; connectivity indices must be in `[0, n_points)`.
- `UnstructuredData.from_binary_le(path, order='F')` reads a prefixed file. `from_binary_le_legacy(path_to_binary, path_to_json, order='F')` reads a raw body plus JSON sidecar through the same validated decoder. Both restore `xarray_attrs` to the resulting dataset, for either version.

### **Legacy Compatibility (Version 1)**

An absent `format_version` means v1; explicit integer `1` is also accepted. V1 uses the same geometry followed by two **float32 matrices**: cell attributes, then vertex attributes, reshaped using the supplied order. The header uses `cell_attr_shape`, `cell_attr_names`, `cell_attr_types`, and the corresponding `vertex_attr_*` fields. Shapes default to `[0, 0]`; names must be unique scalar labels matching the column count. Optional `*_attr_types` lists must match the names in length, but are descriptive source-type metadata, **not** the payload dtype.

Canonical connectivity shapes are preserved unless v1 attribute rows explicitly establish flattened connectivity. V1 noncanonical `[1, N]` shapes with `N > 3` can be treated as historical flattened line/triangle connectivity: candidates are `[N / 2, 2]` and `[N / 3, 3]` where divisible. A positive `cell_attr_shape` row count can resolve the candidate. It can also establish that `[1, 4]` or `[1, 8]` contains multiple lines rather than one tetrahedron/hexahedron. Without that evidence, those canonical single-cell shapes are preserved. Exactly one candidate must remain for flattened connectivity; ambiguous or unsupported shapes are rejected rather than guessed. The flattened sequence is reshaped in C order after wire decoding.

`UnstructuredData.to_binary_legacy()` returns a raw body and header for sidecar storage, but the current writer still emits a **v2** header and per-column body; the method name does not select v1 serialization.

### **Writer Semantics and Losses**

The existing public writer filters unsupported attribute columns. Object columns are coerced with `pandas.to_numeric(errors='coerce')` only if the non-null mask is unchanged; all-null object columns and other unsupported columns are omitted. Empty attribute tables emit empty metadata lists, so zero-row column names are omitted. This documents existing behavior, not a change to filtering or coercion.

Actual per-column serialization preserves integer width and signedness, stores booleans as one byte, and stores floating columns as `float32` unless all values are integral and conversion to `int64` round-trips through `float64`, in which case it stores `int64`. Reader support for `float16`/`float64` does not imply the writer preserves those floating dtypes.

The dtype seen by the writer is not necessarily the original source column dtype: mixed-dtype xarray attribute matrices can already have coerced columns to a common dtype. Large integers may lose precision **before serialization**, and restoring a mixed-dtype matrix after reading also cannot promise original per-column source dtypes. Geometry casts to `float32`/`int32` and floating attribute casts can introduce further precision loss or overflow.

New output tools **MUST preflight losses**, including omitted/coerced columns, earlier mixed-matrix precision loss, geometry range/precision, and attribute conversions, before replacing a destination. They **MUST use context-managed atomic destination writes**: write and close a temporary file, then atomically replace the destination only after successful serialization and validation. The existing byte-returning writers do not enforce these safeguards; this foundation work adds no new output tools.

Readers reconstruct geometry, supported numeric attributes, and dataset metadata only. Element wrappers, textures, and high-level geological objects are not reconstructed.

---

## **B) Structured Grids**

`.le` files can also be generated by the `StructuredData` class. The *outer* layout (4-byte length → JSON → binary payload) is identical, but the **JSON** and **binary payload** are typically simpler:

1. **JSON header** might look like:
   ```json
   {
     "data_shape": [nx, ny, nz],
     "bounds": {
       "x": [x_min, x_max],
       "y": [y_min, y_max],
       "z": [z_min, z_max]
     },
     "transform": null,
     "dtype": "float32",
     "data_name": "data_array"
   }
   ```
   - `data_shape`: shape of the single data array.  
   - `bounds`: optional bounding box info for each dimension.  
   - `dtype`: e.g. `"float32"` or `"float64"`.  
   - `data_name`: name of the “active” data array.

2. **Binary payload**  
   - Usually a *single* numeric array (the active data array) in Fortran order with shape = `data_shape`.

### **Reading/Writing (Structured)**

- **Write**:  
  1. Serialize the JSON header (`data_shape`, `dtype`, etc.).  
  2. Write the active data array (`float32`) in Fortran order.  
  3. Prepend the 4-byte header-length integer.  

- **Read**:  
  1. Read the 4-byte integer → parse JSON.  
  2. From `data_shape`, read the required bytes for the single data array.  
  3. Reshape the array in Fortran order (`[nx, ny, (nz)]`), and attach any bounding/metadata as needed.

---

## **Temporal Snapshots**

Temporal data uses one ordinary `.le` snapshot per observation time, without a
new payload format or an extra time coordinate in the binary body. Metadata is
stored in the JSON header, for example:

```json
{
  "xarray_attrs": {
    "time_series_id": "temperature-series",
    "timestamp": "2025-12-12T10:45:39Z",
    "attribute_units": {"temperature": "degC"}
  }
}
```

For structured data, `StructuredData.to_binary()` and the split body/header
interface include dataset attrs as `xarray_attrs` only when `timestamp` or
`time_series_id` is present. All included attrs must be JSON-serializable;
Python datetime and NumPy objects must be converted by the caller, and
non-finite metadata numbers are rejected. Static structured headers and bytes
remain unchanged, even when other dataset attrs exist. The active array,
shape, bounds, dtype, and payload ordering are unchanged.

Unstructured v2 headers already include dataset attrs. Both
`UnstructuredData.from_binary_le()` and `from_binary_le_legacy()` restore
`xarray_attrs` as dataset attrs, including when reading v1 payloads. Files
without this field restore empty attrs. Temporal metadata does not change the
v2 column encoding described above.

The low-level binary interfaces do not validate timestamp syntax, normalize
timezones, enforce paired metadata, or select a time slice. Callers exporting
a series must supply a stable series ID and a UTC ISO 8601 observation timestamp
after resolving the source timezone, and slice volumes to an ordinary 3D frame.
A sidecar index can identify frames without changing `.le` files, for example:

```json
{
  "schema_version": 1,
  "time_series_id": "temperature-series",
  "kind": "volume",
  "frames": [
    {"timestamp": "2025-12-12T10:45:39Z", "path": "volume_0000.le"},
    {"timestamp": "2025-12-12T11:45:39Z", "path": "volume_0001.le"}
  ]
}
```

Index discovery and client playback are separate contracts; this metadata
support does not implement them or establish compatibility with strict client
header parsers.

## **Summary**

- **File Layout**:  
  1) 4-byte unsigned integer (little-endian) -> 2) JSON header (UTF-8) -> 3) Binary payload (numeric arrays with externally supplied order).

- **Unstructured** `.le`:  
  - Contains geometry followed by per-column cell and vertex attributes in v2; v1 uses float32 attribute matrices.
  - The JSON describes geometry shapes, attribute layout, and dataset metadata; it does not record geometry byte order or array order.

- **Structured** `.le`:  
  - Typically contains a single array plus optional bounding and transform data.  
  - The JSON describes the grid dimensions (`data_shape`), data type, and possibly coordinates/bounds.

The **core principle** is always the same: parse the header size, parse the JSON, then read and reshape the binary data as specified. Differences lie in how many arrays are written and which header fields are present, depending on whether the source is unstructured or structured.
