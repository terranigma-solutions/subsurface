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

### B. Survey (`survey.csv`)
This file describes the trajectory (deviation) of the borehole.

- **Mandatory First Column**: `id` (Must match the IDs in the collars file)
- **Required Columns**: 
  - `md`: Measured Depth
  - `inc`: Inclination (0 to 180 degrees, where 180 is vertical down)
  - `azi`: Azimuth (0 to 360 degrees)

*Note: You can also use `dip` instead of `inc`. Subsurface will automatically convert it.*

**Example:**
```csv
id,md,inc,azi
well_01,0,180,0
well_01,10,180,0
well_01,20,175,10
```

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

---

## 2. Understanding the Difference: Lithology vs. Assays

When importing data, you will see a parameter called `is_lith_attr`. Here is why it matters:

| Feature | Lithology (`is_lith_attr=True`) | Assays (`is_lith_attr=False`) |
| :--- | :--- | :--- |
| **Data Type** | Categorical (Rock types, formations) | Numerical (Geochem, logs) or Categorical |
| **Required Columns** | Must have `component lith` | Any column name is allowed |
| **Validation** | Checks for `top`, `base`, and `component lith` | Only requires `base` column |
| **Visualization** | Often used for 3D volumes or colored intervals | Often used for logs, point clouds, or heatmaps |
| **Example File** | `lithology.csv` | `assays.csv` |

---

## 3. Key Arguments Explained

### A. `number_nodes`
- **What it is**: The number of sampling points used to reconstruct the 3D trajectory of the well.
- **Why it matters**: A higher number of nodes results in a smoother, more accurate 3D curve for deviated wells, but increases memory usage. For vertical wells, a small number (e.g., 2-5) is sufficient.

### B. `add_attrs_as_nodes`
- **What it is**: A flag that tells Subsurface to create explicit nodes at the depths specified in your attribute file (`top` and `base`).
- **Why it matters**: If you want your 3D trajectory to perfectly align with your data intervals (e.g., a node exactly at the boundary between two rock layers), set this to `True`.

### C. `duplicate_attr_depths`
- **What it is**: Controls how Subsurface handles nodes at the same depth.
- **Why it matters**: When set to `True`, Subsurface ensures that interval boundaries are sharp. It creates two nodes at the same depth—one for the end of the previous interval and one for the start of the next—allowing attributes to change instantaneously without "bleeding" into each other in 3D visualization.

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

**Python Example:**
```python
from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper

# 1. Setup readers
collars_reader = GenericReaderFilesHelper(file_or_buffer="collars.csv")
surveys_reader = GenericReaderFilesHelper(file_or_buffer="survey.csv")
attrs_reader = GenericReaderFilesHelper(file_or_buffer="lithology.csv")

# 2. Import
borehole_set = read_wells(
    collars_reader=collars_reader,
    surveys_reader=surveys_reader,
    attrs_reader=attrs_reader,
    is_lith_attr=False,  # Set to True if reading lithology, False for assays
    add_attrs_as_nodes=True # Set to True to add nodes at assay/lithology depths
)
```

---

## 3. Advanced Reader Settings

If your data is NOT in the canonical format, you can use these parameters to help Subsurface understand it:

- **Separator**: If your file uses something other than a comma (e.g., `;` or `\t`).
- **Header**: The row number where column names are located (Default is `0`).
- **Encoding**: Text encoding (e.g., `utf-8` or `ISO-8859-1`).
- **Columns Map**: A dictionary to rename your columns to the required names.
  - *Example:* `{"HOLE_ID": "id", "DEPTH_TO": "base"}`
- **Additional Reader Arguments**: Any valid `pandas.read_csv` argument (e.g., `{"na_values": ["NA", "-999"]}`).

---

# 5. Tips for Success

1. **Consistent IDs**: Ensure the well IDs are spelled exactly the same in all three files.
2. **Positive Depths**: `md`, `top`, and `base` should generally be positive values representing depth from the surface.
3. **Start with Defaults**: Try importing with the canonical format first to verify your data structure before adding complex mappings.
