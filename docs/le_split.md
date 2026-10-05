# Splitting LiquidEarth Files

```python
from subsurface.api.le_split import split_le

outputs = split_le(
    "surfaces.le", "existing_output_directory",
    object_attribute="object_id", association="cell",
)
```

`split_le` returns an insertion-ordered dictionary from sorted original numeric
IDs to absolute `pathlib.Path` destinations. Integer IDs, including int64 values
above float64's exact integer range, are not converted to floats. IDs must be
finite, nonmissing, and nonboolean. Grouping is explicit: `cell` for lines,
triangles, tetrahedra, and hexahedra; `point` for connectivity widths zero or one.
Wrong associations or missing columns raise `ValueError`. An explicitly named
empty grouping column returns `{}` without writing files.

Outputs are deterministically named `object_000000.le`, `object_000001.le`, etc.
IDs never become path components. The output directory must already exist.
There is no overwrite option: existing files, directories, dangling symlinks,
and source aliases (including hard links) are rejected before publication.

## Geometry And Attributes

Cell grouping keeps cells in source order, gathers referenced vertices in source
index order, and remaps connectivity to valid local indices. Unused vertices are
excluded. Vertices shared by objects are duplicated across outputs, not welded.
Every cell and point attribute is subset consistently, including the original
grouping IDs, mixed integer/float columns, and boolean columns.

Point grouping keeps selected points in source order, including points not
referenced by connectivity. Width-one connectivity retains every source cell
referencing a selected point (including duplicates), remaps its index, and subsets
cell attributes accordingly. Width-zero connectivity must have either no rows or
one row per point; in the latter case rows and cell attributes are selected
positionally. Partial width-zero rows are ambiguous and rejected. A subset with
zero cell rows and named cell attributes is rejected rather than silently losing
the attribute schema under the current writer.

Files are loaded directly as `LiquidEarthMesh`, avoiding mixed-dtype xarray
attribute coercion. Embedded-header unstructured files use Fortran array order;
legacy sidecars and externally specified array orders are not supported.
Coordinates follow existing float32 wire precision. Integer and boolean columns
retain their values and widths; floating columns follow the shared writer's
int64-if-exactly-integral, otherwise float32 rule. Finite attribute overflow and
nonfinite or overflowing geometry are rejected. Nonfinite nongrouping numeric
attributes are retained.

## Metadata And Safety

All dataset metadata is preserved except the reserved `le_tools` provenance
dictionary. Each output records `operation`, absolute resolved `source`,
`object_attribute`, `association`, and original `object_id` there. Existing
`le_tools` dictionaries are retained under `previous`; a non-dictionary value is
rejected. Other metadata, including coordinate-system semantics, is unchanged.
Sibling merge operations can treat `le_tools` as provenance while requiring all
other dataset metadata to match strictly.

Every destination is checked and every output is serialized and validated before
any publishing begins. The shared writer writes all outputs into a private
temporary directory on the destination filesystem. Each staged inode is captured
before final publication via an atomic no-clobber hard link. Rollback tracks the
known inode before attempting the link, so even an error after successful
publication removes the current output as well as earlier outputs. No destination
stat is needed after publication. Inode checks ensure rollback does not remove
racing destinations or files replaced by another actor. Existing or unrelated
files are never removed. Source bytes are never modified. Staging files and the
temporary directory are cleaned on success and failure; filesystem errors can
prevent staging cleanup or rollback. Concurrent hostile directory/source
replacement is out of scope.
This multi-file operation is **not crash atomic**: process termination or machine
failure can leave a prefix of outputs. There is no spatial selection, clipping,
welding, topology repair, or CRS conversion.
