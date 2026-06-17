# AGENTS.md — Subsurface Codebase Guide

## Project Overview

**Subsurface** is a Python library for geoscientific data — a "DataHub" that unifies geometric subsurface data into common data objects using numpy/xarray, and provides I/O, metadata, and visualization for those objects.

- **Package name**: `subsurface` (PyPI), `subsurface_terra` (in `setup.py`)
- **Python**: 3.8+
- **License**: Apache-2.0
- **Status**: Alpha
- **Origin**: Software Underground, maintained by Terranigma Solutions

---

## Essential Commands

| Category | Command | Notes |
|---|---|---|
| Install (dev) | `pip install -e .` | From project root |
| Install (all deps) | `pip install -r requirements/requirements_all.txt` | CI uses this |
| Install (dev deps) | `pip install -r requirements/requirements_dev.txt` | For docs/linting |
| Test all | `pytest --cov=subsurface` | CI command |
| Test specific | `pytest tests/test_io/test_lines/` | Run a directory |
| Test with marker | `pytest -m core` | Uses markers: `core`, `read_mesh`, `slow` |
| Requirement filtering | `REQUIREMENT_LEVEL=READ_WELL pytest` | Skips tests needing uninstalled deps |
| Docs | `cd docs && make html` | Builds Sphinx docs |

### Requirements split (in `requirements/`):
- `requirements.txt` — **core**: xarray, netcdf4, python-dotenv, pydantic
- `requirements_wells.txt` — welly, striplog
- `requirements_mesh.txt` — mesh readers (trimesh, ezdxf, etc.)
- `requirements_volume.txt` — volume readers (segyio, etc.)
- `requirements_plot.txt` — pyvista, matplotlib
- `requirements_geospatial.txt` — rasterio, geopandas
- `requirements_all.txt` — everything
- `requirements_dev.txt` — docs build tools

### Testing note:
The `conftest.py` has a `RequirementsLevel` flag system. Set `REQUIREMENT_LEVEL=CORE` to run only tests needing core deps, or `ALL` (default) for everything. Tests use `pytest.mark.skipif` with `RequirementsLevel` to gate themselves.

---

## Architecture — Data Level Hierarchy

From README.rst (top = human-facing, bottom = computer):

```
HUMAN
  geo_format     — geological concepts (boreholes, faults, seismic)
    geo_object   — geological objects
      element    — geometry type (PointSet, TriSurf, LineSet, TetraMesh)
        primary_struct — UnstructuredData / StructuredData
          DF/Xarray — labeled numpy arrays
            array — memory
COMPUTER
```

---

## Core Data Structures

### `UnstructuredData` (`subsurface/core/structs/base_structures/unstructured_data.py`)
The **central data container**. Wraps an `xr.Dataset` with four variables: `vertex`, `cells`, `cell_attrs`, `vertex_attrs`. Dimension names are strict: `("cell", "cell_attr")` for cell attrs, `("points", "vertex_attr")` for vertex attrs.

**Cells shape determines element type:**
| Cells shape | Element | Example |
|---|---|---|
| (N, 0) or (N, 1) | Point cloud | LiDAR scan |
| (N, 2) | Lines | Borehole trajectories |
| (N, 3) | Triangles | Surface DEM, mesh |
| (N, 4) | Tetrahedra | Volumetric mesh |
| (N, 8) | Hexahedra | Unstructured grid |

**Main constructor:**
```python
UnstructuredData.from_array(vertex, cells, cells_attr=None, vertex_attr=None)
```
- `cells` can be a numpy array OR `SpecialCellCase.POINTS` / `SpecialCellCase.LINES` (auto-generates cells)
- `cells_attr` / `vertex_attr` are `pd.DataFrame` or dict of `xr.DataArray`

Other constructors: `from_data_arrays_dict()`, `from_binary_le()`, `from_binary_le_legacy()`

**Gotcha**: `_to_bytearray` is defined twice in the file (lines 306–311 and 314–331). The second one overrides the first. Both exist in source — don't assume only one.

### `StructuredData` (`subsurface/core/structs/base_structures/structured_data.py`)
For gridded/structured data. Has a `type` enum: `REGULAR_AXIS_ALIGNED` (most common), `REGULAR_AXIS_UNALIGNED`, `IRREGULAR_AXIS_ALIGNED`, `IRREGULAR_AXIS_UNALIGNED`. Only `REGULAR_AXIS_ALIGNED` is fully implemented.

**Constructors**: `from_numpy()`, `from_data_array()`, `from_dict()`, `from_pyvista()`, `from_netcdf()`

**Serialization**: `to_netcdf(path)` / `from_netcdf(path)` round-trips metadata in attrs.

### Element wrappers (thin, add validation on construction):
| Class | File | Validation |
|---|---|---|
| `PointSet` | `core/structs/unstructured_elements/point_set.py` | cells shape[1] ≤ 1 |
| `TriSurf` | `core/structs/unstructured_elements/triangular_surface.py` | cells shape[1] == 3 |
| `LineSet` | `core/structs/unstructured_elements/line_set.py` | cells shape[1] == 2 |
| `TetraMesh` | `core/structs/unstructured_elements/tetrahedron_mesh.py` | cells shape[1] == 4 |

### Structured element wrappers:
- `StructuredGrid` (`core/structs/structured_elements/structured_grid.py`) — 3D curvilinear grids
- `StructuredSurface` (`core/structs/structured_elements/structured_mesh.py`) — 2D surfaces

---

## Geological Formats

### BoreholeSet (`core/geological_formats/boreholes/`)
Three components:
1. **Collars** — well surface locations (`PointSet` with `ids`)
2. **Survey** — wellbore trajectory (`LineSet` with `well_id_mapper`)
3. **Combined trajectory** — survey + collar offset merged into one `LineSet`

**Data flow**: CSV files → read_collar/read_survey/read_lith (as DataFrames) → `Survey.from_df()` → `Survey.update_survey_with_lith()` → `BoreholeSet(collars, survey, MergeOptions.INTERSECT)`

**Angle correction** (`survey.py:_correct_angles`): inclination normalized to 0–180°, azimuth to 0–360°.

**MergeOptions**: `INTERSECT` (default, keeps only wells in both collars and survey) or `RAISE` (not implemented — raises NotImplementedError).

**Gotcha**: `BoreholeSet.__init__` reindexes collars to match survey well order via `_remap_collars_with_survey`. This means `borehole_set.collars.ids` may differ from the original collars CSV order.

---

## Reader Architecture

### `GenericReaderFilesHelper` (`core/reader_helpers/readers_data.py`)
Pydantic model that configures file reading. Key fields:
- `file_or_buffer`: str, pathlib.Path, bytes, dict, or IO stream
- `format`: auto-detected from extension if not provided (CSV, JSON, XLXS, DXF, DXFStream)
- `columns_map`: dict mapping expected → actual column names
- `separator`: delimiter for text files
- `pandas_reader_kwargs`: property that aggregates `names`, `header`, `index_col`, `usecols`, `encoding`, `sep`

### Reader helpers:
- `ReaderUnstructuredHelper` — groups vertex/cells/vertex_attr/cells_attr reader configs
- `ReaderWellsHelper` — groups collars/survey/lith/attr reader configs

### Stream API (`api/interfaces/stream.py`)
Entry points for non-file I/O: `DXF_stream_to_unstruc`, `OMF_stream_to_unstruc`, `OBJ_stream_to_trisurf`, `VTK_stream_to_struct`, `CSV_wells_stream_to_unstruc`, etc.

### Reader modules (`modules/reader/`):
| Submodule | Key files | Formats |
|---|---|---|
| `wells/` | `read_borehole_interface.py`, `_read_to_df.py`, `DEP/` | CSV, XLSX, welly backend |
| `mesh/` | `surface_reader.py`, `_GOCAD_mesh.py`, `dxf_reader.py`, `obj_reader.py`, `glb_reader.py`, `omf_mesh_reader.py`, `csv_mesh_reader.py`, `mx_reader.py` | DXF, OBJ, GLB, OMF, MX, CSV |
| `volume/` | `read_volume.py`, `segy_reader.py`, `seismic.py` | VTK, SEG-Y, grav3d |
| `topography/` | `topo_core.py` | DTM, DXF |
| `profiles/` | `profiles_core.py` | Image profiles |
| `faults/` | `faults.py` | Fault sticks |

---

## Writer Architecture

### Binary format (`modules/writer/to_binary.py`, `modules/writer/to_rex/`)
Two binary output formats:
1. **Liquid Earth (.le)**: JSON header (with 4-byte length prefix) + binary body. Vertex: float32, cells: int32, attributes: float32. Default order: Fortran ('F').
2. **REX**: 3D visualization format (spec in `modules/writer/to_rex/doc/rex-spec-v1.md`). Supports mesh, material, and line set blocks.

### GemPy integration (`modules/writer/to_rex/gempy_to_rexfile.py`)
Converts GemPy geological models to REX binary files. Has `GemPyToRex` class with mesh preprocessing, backside triangle generation, and material color encoding.

---

## Visualization

All in `modules/visualization/to_pyvista.py`:

| Function | Input | Output |
|---|---|---|
| `to_pyvista_points(PointSet)` | PointSet | pv.PolyData |
| `to_pyvista_mesh(TriSurf)` | TriSurf | pv.PolyData (with texture support) |
| `to_pyvista_line(LineSet, as_tube=True)` | LineSet | pv.PolyData (tubed lines) |
| `to_pyvista_tetra(TetraMesh)` | TetraMesh | pv.UnstructuredGrid |
| `to_pyvista_grid(StructuredGrid)` | StructuredGrid | pv.StructuredGrid |
| `pv_plot(meshes, ...)` | list of pv objects | Plotter or matplotlib figure |

**Texture support**: TriSurf can carry texture data via `StructuredData`. UV coordinates in `points_attributes['u']` and `points_attributes['v']`, or via texture plane mapping.

---

## Testing Patterns

### Test layout
```
tests/
  conftest.py           — fixtures (unstruct_factory, point_set_fixture, tri_surf, line_set, tetra_set, struc_data, data_path)
  pytest.ini            — markers: core, read_mesh, slow
  test_io/
    test_lines/         — well reading tests
    test_meshes/        — mesh reader tests
    test_volumes/       — volume reader tests
    test_combined/      — OMF, VTK tests
  test_structs/         — base structure and mesh tests
  test_geological_formats/  — fault tests
  test_visualization/   — pyvista tests
  test_interfaces/      — binary I/O tests
```

### Requirement-based test skipping
```python
from ...conftest import RequirementsLevel

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_WELL"
)
```

Set `REQUIREMENT_LEVEL=CORE` env var to run only core tests, `ALL` for everything.

### Test data
`tests/data/` contains test fixtures: borehole CSVs (supersimple_*), DXF files, SEG-Y files, VTK files, topography data, etc. `data_path` fixture provides the absolute path.

---

## Naming Conventions & Style

- **Classes**: PascalCase (`UnstructuredData`, `BoreholeSet`, `GenericReaderFilesHelper`)
- **Methods/Functions**: snake_case (`from_array`, `to_pyvista_mesh`, `read_wells`)
- **Private**: `_prefix` for module-internal (`_correct_angles`, `_combine_trajectories`)
- **Type hints**: Heavy use of typing module, `Union`, `Optional`, `Literal`, `Hashable`
- **Documentation**: Docstrings on all public classes/methods
- **Imports**: Absolute imports within package (`subsurface.core.structs.base_structures.UnstructuredData`)
- **xarray dims**: Standardized dimension names — `points`, `XYZ`, `cell`, `nodes`, `cell_attr`, `vertex_attr`

---

## Gotchas & Non-Obvious Patterns

1. **`.env` loaded on import**: `subsurface/__init__.py` calls `dotenv.load_dotenv()` — environment variables from `.env` are available everywhere.

2. **`_to_bytearray` duplicate**: `unstructured_data.py` has two identical `_to_bytearray` methods (lines 306 and 314). The second shadows the first. Don't try to modify one expecting the other to stay — they're the same code.

3. **Version via setuptools_scm**: Version is derived from git tags. `subsurface/_version.py` is auto-generated. In CI (`docs.yml`), tags are explicitly fetched for correct versioning.

4. **Borehole collar reindexing**: `BoreholeSet._remap_collars_with_survey` reindexes collars to match survey well order. If a well is in the collars CSV but not in the survey, it gets dropped silently (MergeOptions.INTERSECT).

5. **`pandas_reader_kwargs` property**: `GenericReaderFilesHelper.pandas_reader_kwargs` aggregates all reader parameters. `additional_reader_kwargs` are merged in. Backslash-escaping in regex patterns uses double-escape (`\\\\s{2,}` → `\s{2,}`) due to Pydantic serialization.

6. **`cells="lines"` vs `cells="points"`**: Using string literals for cells auto-generates the connectivity array. "lines" creates sequential pairs (0-1, 1-2, ...), "points" creates single-element arrays.

7. **Fortran order default**: Binary serialization defaults to `order='F'` (column-major/Fortran ordering), not C order.

8. **xarray index quirks**: `UnstructuredData.from_data_arrays_dict` tries `ds.reset_index('cell')` which can fail silently with KeyError/ValueError depending on xarray version. A TODO notes this changed with xarray 2022.06.

9. **DXF as stream type**: There's a separate `SupportedFormats.DXFStream` for DXF data coming from IO streams (not files).

10. **`install_requires` is minimal**: The `setup.py` core dependencies are just xarray, netcdf4, python-dotenv, and pydantic. Everything else (pyvista, welly, striplog, trimesh, etc.) is optional.

11. **`TriSurf` texture plane**: When UV coords aren't in the data, texture mapping uses `texture_map_to_plane` with `texture_origin`, `texture_point_u`, `texture_point_v` — these must be provided via kwargs on construction.

12. **Wells backend is welly-only**: The `read_wells_to_unstruct` function in the old DEP module only supports `backend='welly'`. Any other backend raises `AttributeError`.