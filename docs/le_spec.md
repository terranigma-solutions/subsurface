Below is a revised, polished version of the specification. Repetitive statements have been streamlined, and formatting has been made more consistent and concise.

---

# **LiquidEarth Mesh (`.le`) File Format Specification**

## **Overview**

A `.le` file comprises:

1. A **4-byte little-endian integer** specifying the length (in bytes) of a JSON header.  
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

1. **4-byte integer** (little-endian)  
   - Denotes how many bytes are used by the JSON header. Call this `N_header_bytes`.

2. **JSON header** (UTF-8, length = `N_header_bytes`)  
   - Deserialized via `json.loads(...)` in Python.  
   - Contains metadata describing how to interpret the binary payload:
     - For **unstructured** data, includes shapes for vertices/cells, attribute shapes, and names/types of attributes, etc.  
     - For **structured** data, includes shape of the grid, bounding information, data type, etc.

3. **Binary payload**  
   - Immediately follows the header.  
   - Contains the numeric arrays (in Fortran-contiguous order, by default).

---

## **A) Unstructured Meshes**

### **File Layout**

```
+------------------+------------------------+--------------------------------------+
| 4-byte integer   | JSON header (UTF-8)   | Binary payload                       |
| (little-endian)  |                       | (vertex, cells, cell_attr, pt_attr)  |
+------------------+------------------------+--------------------------------------+
```

1. **4-byte integer** (`N_header_bytes`)  
2. **JSON header**  
   - Typically has keys like:
     ```json
     {
       "vertex_shape": [n_points, 3],
       "cell_shape": [n_cells, n_vertices_per_cell],
       "cell_attr_shape": [n_cells, n_cell_attrs],
       "vertex_attr_shape": [n_points, n_point_attrs],
       "cell_attr_names": [...],
       "cell_attr_types": [...],
       "vertex_attr_names": [...],
       "vertex_attr_types": [...],
       "xarray_attrs": { ... }
     }
     ```
   - These fields indicate how many points/cells there are, how many attributes, and so on.

3. **Binary payload** (in this order):
   1. **Vertex array** (`float32`, shape = `[n_points, 3]`): point coordinates.  
   2. **Cells array** (`int32`, shape = `[n_cells, n_vertices_per_cell]`): connectivity.  
   3. **Cell attribute array** (`float32`, shape = `[n_cells, n_cell_attrs]`).  
   4. **Vertex (point) attribute array** (`float32`, shape = `[n_points, n_point_attrs]`).

The exact dimensions and data types are specified in the JSON header. Because it uses Fortran-contiguous bytes (`order='F'`), any reader must reshape the arrays accordingly.

---

### **JSON Header (Unstructured)**

A typical example:

```json
{
  "vertex_shape": [2563, 3],
  "cell_shape": [4821, 3],
  "cell_attr_shape": [4821, 4],
  "vertex_attr_shape": [2563, 2],
  "cell_attr_names": ["rock_type", "porosity", "perm_x", "perm_y"],
  "cell_attr_types": ["float32", "float32", "float32", "float32"],
  "vertex_attr_names": ["density", "temperature"],
  "vertex_attr_types": ["float32", "float32"],
  "xarray_attrs": {
    "description": "A sample triangular mesh",
    "coordinate_system": "EPSG:4326"
  }
}
```

**Key fields**:

- `*_shape` keys define the shapes of vertex, cell, and attribute arrays.  
- `*_attr_names` and `*_attr_types` list the column names and data types for each attribute dimension.  
- `xarray_attrs` can store arbitrary metadata.

---

### **Binary Payload Details (Unstructured)**

Once the JSON is parsed:

1. Read the **vertex** array (`float32`) with shape = `vertex_shape`.  
2. Read the **cells** array (`int32`) with shape = `cell_shape`.  
3. Read **cell attribute** array (`float32`) with shape = `cell_attr_shape`.  
4. Read **vertex attribute** array (`float32`) with shape = `vertex_attr_shape`.

In total, four arrays must be read in that specific order.  

---

### **Pseudocode for Writing (Unstructured)**

```python
import json

def to_binary(unstruct_data, order='F'):
    # 1) Build JSON header
    header_dict = {
      "vertex_shape": list(unstruct_data.vertex.shape),
      "cell_shape": list(unstruct_data.cells.shape),
      "cell_attr_shape": list(unstruct_data.attributes.shape),
      "vertex_attr_shape": list(unstruct_data.points_attributes.shape),
      "cell_attr_names": unstruct_data.attributes.columns.tolist(),
      "cell_attr_types": unstruct_data.attributes.dtypes.astype(str).tolist(),
      "vertex_attr_names": unstruct_data.points_attributes.columns.tolist(),
      "vertex_attr_types": unstruct_data.points_attributes.dtypes.astype(str).tolist(),
      "xarray_attrs": unstruct_data.data.attrs
    }
    
    # 2) Convert JSON to bytes
    header_bytes = json.dumps(header_dict).encode('utf-8')
    header_size = len(header_bytes)
    header_size_bytes = header_size.to_bytes(4, 'little')
    
    # 3) Build binary payload (in Fortran order)
    vertex_bytes = unstruct_data.vertex.astype('float32').tobytes(order=order)
    cells_bytes = unstruct_data.cells.astype('int32').tobytes(order=order)
    cell_attr_bytes = unstruct_data.attributes.values.astype('float32').tobytes(order=order)
    pt_attr_bytes = unstruct_data.points_attributes.values.astype('float32').tobytes(order=order)
    
    body = vertex_bytes + cells_bytes + cell_attr_bytes + pt_attr_bytes
    
    # 4) Concatenate everything
    return header_size_bytes + header_bytes + body
```

---

### **Pseudocode for Reading (Unstructured)**

1. **Read** the first 4 bytes -> `N_header_bytes`.  
2. **Read** the next `N_header_bytes` -> parse JSON header.  
3. **Extract** shapes (`vertex_shape`, etc.) from the header.  
4. **Read** the appropriate number of bytes for each array in the specified order:
   - Vertex → Cells → Cell Attributes → Vertex Attributes.  
5. **Reshape** each array in Fortran order (`np.frombuffer(...).reshape(shape, order='F')`).  
6. **Reconstruct** the unstructured mesh object (`UnstructuredData`).

---

### **Important Notes (Unstructured)**

1. **Endianness**: The 4-byte header size and any array data must be read as little-endian.  
2. **Types**: `float32` for vertices/attributes, `int32` for cells.  
3. **Consistency**: The shapes in the JSON must match the actual array lengths in the payload.  

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

## **Summary**

- **File Layout**:  
  1) 4-byte integer (little-endian) → 2) JSON header (UTF-8) → 3) Binary payload (numeric arrays in Fortran order).  

- **Unstructured** `.le`:  
  - Contains four arrays: vertices, cells, cell attributes, and vertex attributes (in that order).  
  - The JSON describes each array’s shape and any attribute names/types.

- **Structured** `.le`:  
  - Typically contains a single array plus optional bounding and transform data.  
  - The JSON describes the grid dimensions (`data_shape`), data type, and possibly coordinates/bounds.

The **core principle** is always the same: parse the header size, parse the JSON, then read and reshape the binary data as specified. Differences lie in how many arrays are written and which header fields are present, depending on whether the source is unstructured or structured.