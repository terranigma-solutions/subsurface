# LiquidEarth Tools Implementation Report

Integration branch: `le-tools/readback-and-operations`. Local commits only;
no push, PR, amend, main integration, or repository-wide restack/sync.
The original `time_dimension` checkout and unrelated repositories are untouched.
The requested implementation plan is preserved in commit `36ada1d`.
Status: Tasks 1-7 implemented and integrated for the documented supported scope,
including restricted non-resampling structured operations. Implementation agents
are settled and their worktrees are clean. No running tasks or blocked approvals
remain. General resampling/mosaics are not implemented; Task 8 is removed from
scope at the user's request, not awaiting implementation.
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

Task 1-6 independently run command:

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

Task 1-6 implementation branch tips (including the later split safety follow-up):

| Branch | Full SHA |
| --- | --- |
| `le-tools/readback-foundation` | `54dc03fc21c926f6fea752c8a411a6d43d6b13b2` |
| `le-tools/inspection` | `f458b55e395c4d17a419fe27a307a81071a66d0b` |
| `le-tools/structured-reader` | `439f9bb16d3b31b538730896b82385039e26c4d1` |
| `le-tools/transform` | `a5e9e3215490e66dea9132fab6f3a4c501d119a8` |
| `le-tools/split` | `3ac536397d3c2cf707e196068a9f4d72b68051ed` |
| `le-tools/merge` | `768990fa139f07dfc1406d08a0a4f189e153ebd9` |

The Task 1-6 code integration commit before the original report was
`a6a8e5d0d467150364dfc50d6c96389c6b6288b6`. The report itself is committed
separately so its commit can record this stable code revision without self-reference.

## Restricted Task 7

The user authorized translation/positive axis scaling, rectangular index windows,
and aligned adjacent tile concatenation without resampling. The narrowed contract
is committed in `b374da0` and `docs/le_structured_operations_plan.md`. Four new
Herdr worktrees were created without focus changes. Graphite tracks the shared
foundation on `le-tools/readback-and-operations`, and the three operation branches
on `le-tools/structured-ops-foundation`; metadata commands were serialized and
targeted. The shared foundation was rebased locally only while settled and clean.
No repository-wide restack/sync, push, PR, amend, or main integration occurred.

| Track | Implementation SHA | Integration SHA |
| --- | --- | --- |
| Strict structured output safety | `a1f9dd0` | `c6aa8ab` |
| Endpoint-coordinate regularity fix | `1f8b7a0` | `0e5edc2` |
| Structured transforms | `c7b3531` | `c2c7930` |
| Structured index-window split | `575b934` | `966c792` |
| Adjacent axis merge | `afcac3c` | `eb1d74c` |
| Preserve original merge dtype declaration | `77bf833` | `01c7cac` |
| Public exports and structured pipeline tests | coordination-owned | `26798c1` |
| Structured split rollback lifetime | `8165dfe` | `44680a8` |
| Related unstructured split rollback lifetime | `3ac5363` | `2630651` |
| Nonmerge singleton pipeline and frame caveat | coordination-owned | `4bfa083` |

Public APIs from both `subsurface` and `subsurface.api`:

```python
transform_structured_le(source, destination, matrix, *, overwrite=False)
split_structured_le(source, output_directory, *, windows)
merge_structured_le(sources, destination, *, axis, spacing=None, overwrite=False)
```

Scalar values, exact numeric dtype/byte order and declared dtype spelling, active
name, standard rank, and singleton sample positions are preserved. File outputs
are read-back-verified before atomic publication; unsupported in-memory metadata,
extra arrays, coercion, and coordinate layouts are rejected. The existing scalar
format cannot store provenance or CRS/units; callers must establish compatible
frames/units, and no new wire fields are invented.

Transform supports only finite affine translation and positive diagonal scaling.
Split uses half-open integer windows, preserves omitted axes, and stages every
output before no-clobber publication. Merge requires identical scalar schemas,
coincident nonmerge coordinates, positive compatible spacing and exact adjacency
within the documented `1e-10 * spacing` tolerance. Gaps, overlaps, reversed order,
rotation, shear, reflection, and projective transforms fail explicitly. All
singleton merge axes require explicit spacing; mixed tiles can use spacing from
nonsingleton sources. No origin-relative tolerance masks sample-scale errors.

Review corrected a tolerance bug that rejected valid large-origin linspace axes,
retained original dtype declarations on merge, and closed a split inode-recycling
race. Failed-publication rollback now checks ownership while staging links keep
inodes alive; ownership records are not retried after staging cleanup. Related
unstructured split received the same correction. Fault-injection tests cover
post-link failures, inode-recycling racers, and cleanup errors.
The follow-up review confirmed closure of the concrete failed-publication race
and found no introduced regressions in the committed functions and test diffs.

Final independent command is the same offline interface/structure/point-cloud
command above. Result: **1736 passed, 37 skipped, 7 warnings**. This includes
**30 structured pipeline tests** spanning ranks 1-3, every merge axis, exact large
signed/unsigned categorical IDs, float32/float64 with NaN/Inf, read/inspect/
transform/split/merge/inverse composition, all-singleton spacing, nonmerge
singleton preservation, aliases, overlap/gap/reversed-order rejection, original
dtype declarations, and source immutability. Skips and warnings retain the
previous optional-dependency/fixture explanations. Whitespace checks passed.

| Task 7 Branch | Full SHA |
| --- | --- |
| `le-tools/structured-ops-foundation` | `1f8b7a0515b9dc7d7f55264de71d1a0db0797e54` |
| `le-tools/structured-transform` | `c7b3531cd587b663ebc379ed2d52b16d5dd52bc9` |
| `le-tools/structured-split` | `8165dfe1eb53cc13f9f7e184913232525b2d9eef` |
| `le-tools/structured-merge` | `77bf833e76ed5c35bb0de30198e6fafb96b9e610` |

The stable Task 7 code revision before this report update is `4bfa083`.

## Deferred Scope

General structured resampling and mosaics remain outside the completed restricted
Task 7 scope: target-grid/interpolation, overlap precedence, missing-region fill,
nodata, and original singleton voxel-width semantics need separate decisions.
Numerically ambiguous coordinate reconstruction or adjacency is rejected rather
than snapping outside the documented tolerance. Multi-file split is not crash
atomic; filesystem failures can prevent rollback/cleanup. Arrays are processed in
memory with no production resource budget. Task 8 is removed from scope. No new
containers, projective transforms, or CRS conversions are introduced.

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
