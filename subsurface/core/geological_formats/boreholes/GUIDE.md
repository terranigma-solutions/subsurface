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

### C. Attributes/Lithology (`lithology.csv`)
This file contains interval-based data, such as rock types or assay values.

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

---

## 2. Step-by-Step Import Process

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
    is_lith_attr=True  # Set to True if reading lithology
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

## 4. Tips for Success

1. **Consistent IDs**: Ensure the well IDs are spelled exactly the same in all three files.
2. **Positive Depths**: `md`, `top`, and `base` should generally be positive values representing depth from the surface.
3. **Start with Defaults**: Try importing with the canonical format first to verify your data structure before adding complex mappings.
