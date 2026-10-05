# LiquidEarth Tools Implementation Report

Integration branch: `le-tools/readback-and-operations`. Local commits only;
no push, PR, amend, main integration, or repository-wide restack/sync.
The original `time_dimension` checkout and unrelated repositories are untouched.
The requested implementation plan is preserved in commit `36ada1d`.
Status: Tasks 1-6 implemented and integrated for the documented supported scope.
All six implementation agents are settled and their worktrees are clean. No
running tasks or blocked approvals remain. Tasks 7-8 are explicitly deferred.
The plan is a preserved pre-implementation snapshot; this report records results.

## Verified Foundation

| Track | Implementation SHA | Integration SHA |
| --- | --- | --- |
| Unstructured reader and metadata | `608c9c0` | `e59151e` |
| Loss-aware atomic file helpers | `54dc03f` | `0802b6b` |
| Inspection and explicit grouping | `f458b55` (includes `17fc540`, `f13d21e`) | `ca15574` (includes `39bcf58`, `60d1e5d`) |
| Structured reader | `6d13433` | `d3353dc` |
| Structured duplicate-key/nesting validation | `439f9bb` | `538ac80` |

Foundation was integrated into inspection only while both agents were settled
and worktrees clean, using targeted branch-local `git rebase`. Operation branches
share inspection commit `f458b55` and are individually tracked with that exact
Graphite parent. Metadata commands were serialized.

Independent coordination verification before the final structured follow-up:
`755 passed, 5 skipped` in interface/structure suites. An earlier foundation and
volume run including point-cloud tests passed `475` and skipped `37`. Tests use
cached offline Python 3.11, NumPy, pandas, xarray, pydantic, pytest, and dotenv.
Skipped tests need optional dependencies, higher requirement levels, or explicit
OS-sensitive/export opt-ins. No plotting or network execution is required.

## Verified Operations

| Track | Implementation SHA | Integration SHA |
| --- | --- | --- |
| Affine transforms | `a5e9e3` | `6bb64b3` |
| Object split | `0685ee2` | `47fb482` |
| Split post-publication rollback fix | `d61149a` | `cccc663` |
| Compatible merge | `ac13cf4` | `c5e9f15` |
| Split-compatible point topology/provenance | `768990f` | `dd1dfcb` |
| Public exports, usage guide, end-to-end tests | coordination-owned | `a6a8e5d` |

Three isolated implementation agents owned separate operation modules, tests,
and documentation without competing edits to common classes or exports.
Coordination reviewed committed diffs, requested tested follow-up commits, and
cherry-picked sibling commits without merging to main. Public functions are
exported from `subsurface` and `subsurface.api`; see `docs/le_tools.md`.

Review closed two failure/composability gaps: split now stages and registers
known inodes before final publication, including errors after a successful link;
merge rejects partial or incompatible zero-width point-connectivity row policies
and non-dictionary reserved provenance.

Shared contracts: column-vector affine transforms; explicit numeric cell/point
grouping; missing IDs rejected; shared vertices duplicated on split; no welding;
strict compatible merges; deterministic source-ID identity and provenance.
The reserved `le_tools` dataset metadata key carries operation provenance.

Final independently run command:

```bash
uv run --offline --no-project --python 3.11 --with pytest --with numpy --with pandas --with xarray --with pydantic --with python-dotenv env REQUIREMENT_LEVEL=CORE python -m pytest tests/test_interfaces/ tests/test_structs/ tests/test_io/test_pointcloud/ -q
```

Result: **933 passed, 37 skipped, 7 warnings**. Nine coordination-owned tests
cover both export locations, public read/inspect/transform/split/merge for
triangles and point clouds, shared-vertex duplication and unused-vertex removal,
exact large int64 grouping/provenance, source/shared collision policies,
re-splitting merged files, source immutability, structured reader/inspection
composition, and explicit strict rejection of split wire-dtype divergence.
Warnings are optional PyVista absence and existing writer modulo operations on
nonfinite attribute fixtures. `git diff --check` and staged checks passed.

Final implementation branch tips:

| Branch | Full SHA |
| --- | --- |
| `le-tools/readback-foundation` | `54dc03fc21c926f6fea752c8a411a6d43d6b13b2` |
| `le-tools/inspection` | `f458b55e395c4d17a419fe27a307a81071a66d0b` |
| `le-tools/structured-reader` | `439f9bb16d3b31b538730896b82385039e26c4d1` |
| `le-tools/transform` | `a5e9e3215490e66dea9132fab6f3a4c501d119a8` |
| `le-tools/split` | `d61149a61f0ae0ae2ba7df4f39af9c8bb4b5aab5` |
| `le-tools/merge` | `768990fa139f07dfc1406d08a0a4f189e153ebd9` |

The code integration commit before this report is
`a6a8e5d0d467150364dfc50d6c96389c6b6288b6`. The report itself is committed
separately so its commit can record this stable code revision without self-reference.

## Deferred Scope

Task 7 remains deferred: target-grid/interpolation, overlap, missing-region,
nodata, and singleton voxel-width decisions are not fully resolved. No structured
transform/split/merge operations are claimed. Task 8 is design-only and deferred
pending consumer/container agreement. No new containers, projective transforms,
or CRS conversions are introduced.

Representative historical/consumer exports and production performance
qualification remain outstanding. Supported synthetic round trips do not imply
production qualification. The existing wire format does not encode array order;
new file tools use default Fortran order. Existing writer float32 precision rules
remain in force, and output helpers reject unsupported coercion or column loss.
The shipped writer can recode entirely integral floating subsets as int64;
strict merging rejects divergent output schemas rather than silently promoting
them. Named zero-row schemas are rejected on output because the existing writer
drops them. These explicit restrictions are tested and documented, not permissive
merge/format extensions. Existing xarray mixed-attribute coercion remains a
container limitation; file operations preserve separate numeric columns.
