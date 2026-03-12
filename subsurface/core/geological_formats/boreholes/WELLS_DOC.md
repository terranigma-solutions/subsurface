# 1. Overview of Borehole Data

When working with borehole (or well) data, there are typically three major components:

1. **Collars**
    - **What it represents**: The collar is essentially the “top” of the borehole, including the surface coordinates where it starts.
    - **Typical data**: This includes borehole/well ID, and the x,y,z coordinates of the collar.
2. **Survey**
    - **What it represents**: The survey data describes the trajectory (deviation) of the borehole from the collar downward.
    - **Typical data**: Measured depth (MD), azimuth, and dip data along the well trajectory.
3. **Attributes**
    - **What it represents**: Interval-based or point-based properties measured along the borehole. This can include geochemistry, geophysical logs, lithologies, or any other properties that vary with depth.
    - **Typical data**: The well ID (to match with collars), a *from* depth, a *to* depth, and the actual attribute values (e.g., lithology type, geochem assays, etc.).

### Why Three Separate Files?

The CSVs are commonly split so that each dataset (collars, survey, attributes) captures a different aspect of the borehole. Splitting the data makes it easier to manage and update individually. LiquidEarth merges these datasets internally by matching the well IDs (collar IDs) and by referencing the same depth ranges or measured depths in the survey and attributes.

### Special Note on Lithologies

If the attributes being read are lithologies, LiquidEarth may treat them differently—for instance, by using different internal logic or classification schemes to display them in 3D, or by requiring specific naming conventions for intervals. The import process is similar, but the resulting data might be displayed or analyzed in a different manner (e.g., color-coded strata intervals in a 3D model).

---

# 2. Arguments

Guide describing all the key arguments and parameters you’ll encounter when setting up borehole imports (Collars, Surveys, and Attributes) in LiquidEarth. The guide is split into two sections:

1. **High-Level Import Settings** (what you see directly in the import dialog).
2. **Detailed Reader Settings** (advanced options controlling how each CSV file is read via pandas).

## 2.1. High-Level Import Settings

These fields determine what data you’re loading and how LiquidEarth should handle it at a conceptual level.

1. **Content Name**
    - **What it is**: A short label for your imported dataset (e.g., “Spring2025 Geochem”).
    - **Why it matters**: Helps you identify the dataset in LiquidEarth once it’s loaded.
2. **Collar File**
    - **What it is**: A selection among the files you have uploaded to your project that contains the collar (starting) positions of your wells/boreholes.
    - **Typical columns**:
        - Well/borehole ID.
        - X, Y, Z coordinates (the collar location).
3. **Survey File**
    - **What it is**: A selection among the uploaded files that contains the well trajectory data.
    - **Typical columns**:
        - Measured Depth (MD).
        - Azimuth.
        - Dip.
4. **Attrs File**
    - **What it is**: A selection among the uploaded files that contains attributes (interval-based or point-based data), such as geochemistry or lithology.
    - **Typical columns**:
        - Well/borehole ID.
        - From depth.
        - To depth.
        - Attribute values (e.g., lithology type, assay values, etc.).
5. **Number Nodes per Well**
    - **What it is**: The number of sampling points per well used in the 3D trajectory reconstruction.
    - **Why it matters**: More nodes can yield smoother trajectories but requires more computation.
    - **Example**: Setting this to **10** (as in the example) means each well will be split into 10 segments for 3D visualization.
6. **Enable Interval Nodes**
    - **What it is**: A flag indicating whether nodes should also be created at each depth interval in the attribute file.
    - **Why it matters**: Useful if you want explicit 3D nodes where attribute changes occur (e.g., lithology boundaries).
7. **Is Lith Attr**
    - **What it is**: A checkbox (or toggle) to indicate that the attribute data is lithological.
    - **Why it matters**: If set, LiquidEarth may apply special rules for interval merging, default lithology handling, or color-coding in 3D.

---

## 2.2. Detailed Reader Settings

For each file type (Collar, Survey, Attributes), you have advanced options—**Reader Settings**—that control how the file is parsed. Under the hood, LiquidEarth uses something similar to Python’s pandas.read_csv() function. These settings let you handle various CSV formats, encodings, column mappings, etc.

1. **Uploaded File**
    - **What it is**: Links to the file selected above (Collar, Survey, or Attrs).
    - **Why it matters**: Ensures the reader knows which physical file to parse.
2. **Encoding** (Default: `ISO-8859-1`)
    - **What it is**: The text encoding used to read the file (e.g., `UTF-8`, `ISO-8859-1`).
    - **Why it matters**: If your file contains special characters (like accents), matching the correct encoding is crucial for readable data.
3. **Index Columns** (Default: `null`)
    - **What it is**: Tells the reader which column (by name or index) should become the DataFrame’s index.
    - **Why it matters**: Setting an index can make lookups more direct. If not needed, you can leave it at the default (`null`).
4. **Header Row** (Default: `0`)
    - **What it is**: Specifies which row in the CSV file is used for column names. For example, `0` means the first row is the header.
    - **Why it matters**: If your file doesn’t have a header (i.e., only data rows), you might set this to `None` or `1`.
5. **Separator** (Default: `null`)
    - **What it is**: The delimiter that separates columns in your CSV (e.g., `,`, `;`, or `\t`).
    - **Why it matters**: If left `null`, LiquidEarth might auto-detect. If that fails, specify the correct delimiter.
    - **Tip**: If your file is delimited by semicolons, you might manually set it to `";"`.
6. **Columns to Use** (List of columns, expressed as a semicolon-separated string)
    - **What it is**: Tells the reader which columns to include (by name or position).
    - **Why it matters**: Helps skip unnecessary columns in large files.
    - **Example**: `"0; 2; 5"` means read only columns 0, 2, and 5.
    - **Tip**: `null` means “use all columns”. If you only want columns 0 and 2, you might provide `0;2`
7. **Columns Names** (List of new column names, also semicolon-separated)
    - **What it is**: Overrides or sets the column names if the file lacks a proper header.
    - **Why it matters**: Ensures consistent naming, especially if your CSV has no header row or has inconsistent column names.
    - **Example**: `"id; x; y; z"` to explicitly rename columns 0, 1, 2, 3.
8. **Index Map** (Dictionary expressed as `original_name:new_name;...`)
    - **What it is**: A mapping of certain column names to serve as an index or partially rename them for indexing.
    - **Why it matters**: If you want a specific column to become your DataFrame’s index or unify naming across multiple files.
    - **Example**: `"HOLE_ID:id"` means the `HOLE_ID` column is remapped to `id`, potentially used as an index or for unique identification.
9. **Columns Map** (Dictionary expressed as `original_name:new_name;...`)
    - **What it is**: A mapping to rename columns (apart from indexing).  Critical for aligning your CSV columns with LiquidEarth’s internal naming (e.g., “id”, “x”, “y”, “z”, “md”, “dip”, “azi”, “top”, “base”).
    - **Why it matters**: Ensures standardized column naming across different CSV formats.
    - **Example**: `"X:x;Y:y;Z:z"` means rename column `X` to `x`, `Y` to `y`, `Z` to `z`.
    - **Examples** from the three readers:
        - **Collar Reader**:
            
            ```json
            {
              "HOLE_ID": "id",
              "X": "x",
              "Y": "y",
              "Z": "z"
            }
            ```
            
            This means:
            
            - Rename `HOLE_ID` in the CSV to `id`.
            - Rename `X` to `x`, `Y` to `y`, and `Z` to `z`.
        - **Survey Reader**:
            
            ```json
            {
              "Distance": "md",
              "Dip": "dip",
              "Azimuth": "azi"
            }
            ```
            
            This means:
            
            - Rename `Distance` to `md` (measured depth).
            - Rename `Dip` to `dip`.
            - Rename `Azimuth` to `azi`.
        - **Attrs Reader**:
            
            ```json
            {
              "HoleId": "id",
              "from": "top",
              "to": "base"
            }
            ```
            
            This means:
            
            - Rename `HoleId` to `id`.
            - Rename `from` to `top` and `to` to `base` for interval depths.
10. **Additional Reader Arguments** (Dictionary in `key:value;...` format)
    - **What it is**: A catch-all for extra keyword arguments you might pass to `pandas.read_csv()` (e.g., `parse_dates`, `dtype`).
    - **Why it matters**: Gives you advanced control over parsing (handling missing data, data types, date parsing, etc.).
    - **Example**: `{}` means none were provided. You might specify something like `{"na_values": "NA"}` if your data uses `"NA"` to represent missing values, or `{"dtype": "float"}` to force numeric columns.

---

# 3. Tips for Setting Up Your Import

1. **Start with Defaults**
    - For most straightforward CSV files (comma-delimited, first row as header, standard text encoding), leaving the advanced settings at default often works.
2. **Be Consistent**
    - Aim for consistent naming (e.g., always rename the “Hole ID” column to simply “id” in all files). This makes combining data simpler and avoids confusion.
3. **Check Encoding**
    - If you see gibberish characters or question marks, you might need to change from `ISO-8859-1` to `UTF-8`, or vice versa.
4. **Map the Key Columns**
    - Make sure `Index Map` and `Columns Map` highlight how your “id”, “x”, “y”, “z”, “md”, “from”, and “to” columns are named. This is critical for merging collars, surveys, and attributes correctly.
5. Save and load your configuration for later applications