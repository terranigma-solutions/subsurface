# Guide: Importing Borehole Data into Subsurface

This guide provides a step-by-step walkthrough for importing borehole (well) data. To ensure a smooth experience, especially when using a GUI, we recommend converting your data into the **Canonical Subsurface CSV Format** described below.

---

## 1. The Canonical Subsurface CSV Format

The "Canonical" format is the simplest way to represent borehole data. If your files follow this format, they can be read by Subsurface with default settings (no extra arguments required).

### A. Collars (`collars.csv`)
This file defines the starting position (the "top") of each borehole.

- **Mandatory First Column**: `id` (Well/Borehole Identifier)
- **Required Columns**: `x`, `y`, `z` (Coordinates)

**Example:**
```csv
id,x,y,z
well_01,100,200,50
well_02,150,250,60
```

**Corner Cases & Behavior:**
- If the `id` column is not explicitly set as the index, Subsurface automatically uses the **first column** (column 0) as the index.
- **Duplicate IDs**: If multiple rows share the same `id`, only the **first occurrence** is kept; all subsequent duplicates are silently dropped.
- If the required columns `x`, `y`, or `z` are missing (or named differently without a `columns_map`), a `ValueError` is raised with a message suggesting you check column names or use `columns_map`.

### B. Survey (`survey.csv`)
This file describes the trajectory (deviation) of the borehole.

- **Mandatory First Column**: `id` (Must match the IDs in the collars file)
- **Required Columns**: 
  - `md`: Measured Depth
  - `inc`: Inclination (0 to 180 degrees, where 180 is vertical down)
  - `azi`: Azimuth (0 to 360 degrees)

**Example:**
```csv
id,md,inc,azi
well_01,0,180,0
well_01,10,180,0
well_01,20,175,10
```

**Corner Cases & Behavior:**
- **Using `dip` instead of `inc`**: You can provide a `dip` column instead of `inc`. Subsurface will automatically convert it using the formula `inc = 90 - dip`, and the `dip` column will be dropped.
- **Missing `inc` and/or `azi`**: If neither `inc` nor `azi` columns are present after mapping, Subsurface issues a warning and defaults all boreholes to **perfectly vertical** (`inc=180`, `azi=0`). This is useful for simple vertical wells where only measured depth is known.
- **Missing `md`**: If the `md` column is not present after mapping, an `AttributeError` is raised.
- **Single-entry wells**: Wells that have only **one** survey row are automatically **dropped**. A valid trajectory requires at least two depth points. Make sure every well has at least two survey entries (e.g., one at `md=0` and one at the total depth).
- **Inclination correction**: Inclination values are automatically normalized:
  - Values of exactly `0` or `180` are nudged by a tiny amount (`1e-10`) to avoid numerical singularities in trajectory calculations.
  - Values between `180` and `360` are reflected back into the `0–180` range (e.g., `270` becomes `90`).
  - Negative values are first normalized to the `0–360` range using modulo arithmetic.
- **Azimuth correction**: Azimuth values are normalized to the `0–360` range using modulo arithmetic (e.g., `-90` becomes `270`, `400` becomes `40`).

### C. Lithology (`lithology.csv`)
This file contains interval-based geological data, such as rock types.

- **Mandatory First Column**: `id` (Must match the IDs in the collars file)
- **Required Columns**: 
  - `top`: Starting depth of the interval
  - `base`: Ending depth of the interval
  - `component lith`: The name of the lithology/rock type (Required if `is_lith_attr` is True)

**Example:**
```csv
id,top,base,component lith
well_01,0,15,Sandstone
well_01,15,30,Shale
```

**Corner Cases & Behavior:**
- **Missing `component lith`**: If `is_lith_attr=True` and the `component lith` column is not present, an `AttributeError` is raised. Double-check that you are not accidentally marking an assay file as lithology.
- **Missing `top` column**: If only `base` is provided (no `top`), Subsurface will automatically calculate `top` values by assuming the top of each interval starts where the previous one ended (altitude=0). A warning is issued.
- **`altitude` and `base` without `top`**: If your file has `altitude` and `base` columns but no `top`, Subsurface will compute `top` from these two columns. A warning is issued.
- **Missing both `top` and `base`**: A `ValueError` is raised.
- **Negative depths**: Both `top` and `base` values are automatically converted to their **absolute values**. If your data uses negative depths (e.g., elevation-based), they will be made positive.
- **Lithology encoding**: Lithology names are automatically converted to an ordered `Categorical` type (sorted alphabetically). A numeric `lith_ids` column is added (starting from 1) for visualization purposes.
- **Only `top`, `base`, and `component lith` are kept**: All other columns in the lithology file are dropped after validation.

### D. Assays (`assays.csv`)
This file contains measurement-based data, such as geochemistry or geophysical logs. It can be either **interval-based** (like lithology) or **point-based**.

- **Mandatory First Column**: `id` (Must match the IDs in the collars file)
- **Required Columns**:
  - `top` (Optional): Starting depth of the interval.
  - `base`: Ending depth of the interval (or the exact depth for point-based data).
  - Any number of attribute columns (e.g., `Cu`, `Au`, `Gamma`).

**Example (Interval-based):**
```csv
id,top,base,Cu,Au
well_01,0,10,0.5,0.1
well_01,10,20,1.2,0.3
```

**Example (Point-based):**
```csv
id,base,Gamma
well_01,5,120
well_01,10,145
```

**Corner Cases & Behavior:**
- **Missing `base` column**: An `AssertionError` is raised. The `base` column is always required for assay data.
- **All other columns are preserved**: Unlike lithology, assay data keeps all non-index columns.

---

## 2. Understanding the Difference: Lithology vs. Assays

When importing data, you will see a parameter called `is_lith_attr`. Here is why it matters:

| Feature | Lithology (`is_lith_attr=True`) | Assays (`is_lith_attr=False`) |
| :--- | :--- | :--- |
| **Data Type** | Categorical (Rock types, formations) | Numerical (Geochem, logs) or Categorical |
| **Required Columns** | Must have `component lith` | Any column name is allowed |
| **Validation** | Checks for `top`, `base`, and `component lith` | Only requires `base` column |
| **Depth Handling** | `top` and `base` are made absolute (positive) | No automatic sign correction |
| **Output Columns** | Only `top`, `base`, `component lith` are kept | All columns are preserved |
| **Visualization** | Often used for 3D volumes or colored intervals | Often used for logs, point clouds, or heatmaps |
| **Example File** | `lithology.csv` | `assays.csv` |

---

## 3. Key Arguments Explained

### A. `number_nodes`
- **What it is**: The number of sampling points used to reconstruct the 3D trajectory of the well.
- **Default**: `10`
- **Why it matters**: A higher number of nodes results in a smoother, more accurate 3D curve for deviated wells, but increases memory usage. For vertical wells, a small number (e.g., 2-5) is sufficient.

### B. `add_attrs_as_nodes`
- **What it is**: A flag that tells Subsurface to create explicit nodes at the depths specified in your attribute file (`top` and `base`).
- **Default**: `False`
- **Why it matters**: If you want your 3D trajectory to perfectly align with your data intervals (e.g., a node exactly at the boundary between two rock layers), set this to `True`. When `False`, the attribute DataFrame is **not** passed to the trajectory builder, and nodes are placed only at the regular `number_nodes` intervals.

### C. `duplicate_attr_depths`
- **What it is**: Controls how Subsurface handles nodes at the same depth.
- **Default**: `False`
- **Why it matters**: When set to `True`, Subsurface ensures that interval boundaries are sharp. It creates two nodes at the same depth—one for the end of the previous interval and one for the start of the next—allowing attributes to change instantaneously without "bleeding" into each other in 3D visualization.

### D. `is_lith_attr`
- **What it is**: A flag indicating whether the attribute file contains lithological (categorical rock type) data.
- **Why it matters**: Controls which validation path is used. When `True`, the `component lith` column is required and lithology-specific processing is applied (categorical encoding, `lith_ids` generation, depth sign correction). When `False`, only the `base` column is validated.

---

## 4. Step-by-Step Import Process

Follow these steps to import your data into Subsurface:

### Step 1: Prepare Your Files
Ensure your CSV files are saved with a header row. If your columns have different names (e.g., `Easting` instead of `x`), you can either:
1. Rename them to the canonical names.
2. Use the **Columns Map** feature in the importer to map them (e.g., `{"Easting": "x"}`).

### Step 2: Configure the Readers
For each file (Collars, Survey, Attributes), you need to set up a reader. If you are using the canonical format, simply providing the file path is enough.

### Step 3: Run the Import
Use the `read_wells` function (or the equivalent GUI action) to combine the three files into a `BoreholeSet`.

**Python Example (Canonical Format — Minimal Configuration):**
```python
from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper

# 1. Setup readers (no extra arguments needed for canonical format)
collars_reader = GenericReaderFilesHelper(file_or_buffer="collars.csv")
surveys_reader = GenericReaderFilesHelper(file_or_buffer="survey.csv")
attrs_reader = GenericReaderFilesHelper(file_or_buffer="lithology.csv")

# 2. Import
borehole_set = read_wells(
    collars_reader=collars_reader,
    surveys_reader=surveys_reader,
    attrs_reader=attrs_reader,
    is_lith_attr=True  # Set to True for lithology, False for assays
)
```

**Python Example (Non-Canonical Format — With Column Mapping):**
```python
from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper

collars_reader = GenericReaderFilesHelper(
    file_or_buffer="my_collars.csv",
    usecols=[0, 1, 2, 3],
    columns_map={
        "HOLE_ID": "id",
        "Easting": "x",
        "Northing": "y",
        "Elevation": "z"
    }
)

surveys_reader = GenericReaderFilesHelper(
    file_or_buffer="my_survey.csv",
    columns_map={
        "Distance": "md",
        "Dip": "dip",       # Will be auto-converted to inc
        "Azimuth": "azi"
    }
)

attrs_reader = GenericReaderFilesHelper(
    file_or_buffer="my_geology.csv",
    columns_map={
        "HOLE-ID": "id",
        "FROM": "top",
        "TO": "base",
        "GEOLOGY": "component lith"
    }
)

borehole_set = read_wells(
    collars_reader=collars_reader,
    surveys_reader=surveys_reader,
    attrs_reader=attrs_reader,
    is_lith_attr=True,
    number_nodes=10,
    add_attrs_as_nodes=True,
    duplicate_attr_depths=True
)
```

**Python Example (Special Encoding and Delimiters):**
```python
# For files with semicolon separators, Latin-1 encoding, and comma decimals
reader = GenericReaderFilesHelper(
    file_or_buffer="german_data.CSV",
    encoding="latin-1",
    separator=";",
    additional_reader_kwargs={"decimal": ","},
    columns_map={"RECHTSWERT": "x", "HOCHWERT": "y", "ANSATZH": "z"}
)
```

---

## 5. Advanced Reader Settings

If your data is NOT in the canonical format, you can use these `GenericReaderFilesHelper` parameters to help Subsurface understand it:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `file_or_buffer` | *(required)* | Path to the CSV file (string or `pathlib.Path`). |
| `separator` | `None` (auto-detect) | Column delimiter (e.g., `","`, `";"`, `"\t"`). |
| `header` | `0` | Row number for column names. Set to `None` (or `-1`) if there is no header. |
| `encoding` | `"utf-8"` | Text encoding (e.g., `"utf-8"`, `"latin-1"`, `"ISO-8859-1"`). |
| `index_col` | `False` (auto: col 0) | Column to use as the DataFrame index. For borehole data, this is automatically set to column 0 (the `id` column). |
| `usecols` | `None` (all columns) | List of column names or indices to read (e.g., `[0, 1, 2, 3]` or `["id", "x", "y"]`). |
| `col_names` | `None` | Override column names. Useful when the file has no header row. |
| `columns_map` | `None` | Dictionary to rename columns after reading (e.g., `{"HOLE_ID": "id", "X": "x"}`). |
| `index_map` | `None` | Dictionary to rename index values (row labels). |
| `drop_cols` | `None` | List of column names to drop after reading. |
| `additional_reader_kwargs` | `{}` | Extra keyword arguments passed directly to `pandas.read_csv()` (e.g., `{"decimal": ",", "na_values": ["NA", "-999"]}`). |

**Corner Cases for Reader Settings:**
- **`header=-1`**: Treated as `None` (no header row). Any other negative value raises a `ValueError`.
- **`col_names=[]`** (empty list): Treated as `None`.
- **`index_col=None`**: Internally converted to `False`. For collar, survey, and attribute readers, `False` is then automatically overridden to `0` (first column).
- **Regex in `additional_reader_kwargs`**: Double-escaped backslashes (e.g., `"\\\\s{2,}"`) are automatically unescaped to their proper form (e.g., `"\\s{2,}"`).
- **`delimiter` in `additional_reader_kwargs`**: If you set `delimiter` inside `additional_reader_kwargs`, it takes precedence over the `separator` parameter.

---

## 6. Merging Collars and Survey (The `BoreholeSet`)

After reading collars, survey, and attributes, Subsurface creates a `BoreholeSet` by merging them. The default merge strategy is `MergeOptions.INTERSECT`, which means:

- Only wells that appear in **both** the collars and the survey are included in the final result.
- Wells present in collars but missing from the survey (or vice versa) are silently excluded.

**Common Pitfall:** If your final `BoreholeSet` has fewer wells than expected, check that:
1. The well IDs are spelled **exactly** the same across all three files (collars, survey, attributes). IDs are case-sensitive.
2. Each well in the survey file has **at least two rows** (single-entry wells are dropped during validation).
3. The `id` column is correctly identified (either as the first column or via `columns_map`/`index_col`).

---

## 7. Troubleshooting Common Errors

### `ValueError: Error while reading collars: ...`
- **Cause**: The collar file could not be parsed, or the required columns (`x`, `y`, `z`) are missing.
- **Fix**: Verify column names match the canonical format, or provide a `columns_map`.

### `AttributeError: md, inc, and azi columns must be present...`
- **Cause**: The `md` column is missing from the survey file after column mapping.
- **Fix**: Ensure your survey file has a `md` column, or map it with `columns_map` (e.g., `{"Distance": "md"}`).

### `AttributeError: ... component lith column must be present...`
- **Cause**: You set `is_lith_attr=True` but the attribute file does not have a `component lith` column.
- **Fix**: Either add a `columns_map` entry (e.g., `{"GEOLOGY": "component lith"}`) or set `is_lith_attr=False` if the file contains assay data, not lithology.

### `AssertionError: base column must be present...`
- **Cause**: The attribute file (assay mode) is missing the `base` column.
- **Fix**: Map the appropriate depth column to `base` using `columns_map`.

### `ValueError: top column or base and altitude columns must be present...`
- **Cause**: Lithology file has neither `top` nor `base`/`altitude` columns.
- **Fix**: Ensure at least `base` is present (Subsurface can infer `top`), or provide both `top` and `base`.

### Empty or missing wells in the result
- **Cause**: ID mismatch between files, or single-entry survey wells being dropped.
- **Fix**: Verify consistent IDs across all files. Ensure each well has at least 2 survey rows.

### Garbled characters in data
- **Cause**: Encoding mismatch.
- **Fix**: Try `encoding="latin-1"` or `encoding="ISO-8859-1"` if the default `utf-8` produces errors.

### Wrong number parsing (e.g., `1.234` read as `1234`)
- **Cause**: The file uses a comma as the decimal separator.
- **Fix**: Use `additional_reader_kwargs={"decimal": ","}`.

---

## 8. Tips for Success

1. **Consistent IDs**: Ensure the well IDs are spelled exactly the same in all three files. IDs are case-sensitive.
2. **Positive Depths**: `md`, `top`, and `base` should generally be positive values representing depth from the surface. Lithology depths are automatically made positive; survey depths are not.
3. **Start with Defaults**: Try importing with the canonical format first to verify your data structure before adding complex mappings.
4. **At Least Two Survey Points Per Well**: Wells with only one survey entry are silently dropped. Always include at least a start point (`md=0`) and an end point.
5. **Use `dip` or `inc` — Not Both**: If your file has `dip`, it will be converted to `inc` automatically. Do not provide both columns.
6. **Check Encoding Early**: If you see garbled characters or read errors, change the `encoding` parameter before investigating other issues.
7. **Save and Reuse Configurations**: Once you have a working set of reader parameters for your data format, save them for future imports.
