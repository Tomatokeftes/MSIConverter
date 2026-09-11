# Handout I -- spatialdata table sharding (scverse/spatialdata#1178)

Upstream work in `scverse/spatialdata`, not in this repository. It is the follow-up
deliberately carved out of upstream PR #1106, and it is what lets Thyra eventually delete
`thyra/converters/spatialdata/_chunking.py::table_write_config()`.

Status as of 2026-09-01:

- Issue: https://github.com/scverse/spatialdata/issues/1178 (open, no assignee)
- Design comment posted: https://github.com/scverse/spatialdata/issues/1178#issuecomment-5494212751
  Awaiting a maintainer reply. Do not post a second one.
- Related zarr bug filed as a side effect: see handout J.
- Nothing written yet. Part 2 below is the prompt to start from.

Investigated against `upstream/main` @ ccf1ea0, anndata 0.12.16, zarr 3.2.1/3.3.0.

# PART 1 -- THE PLAN

## 1. What #1178 asks for, and why it matters

[scverse/spatialdata#1178](https://github.com/scverse/spatialdata/issues/1178) (LucaMarconato, 2026-08-18, OPEN, zero comments, no assignee, no linked PR) says: images and labels get write-time chunk/shard configuration via PR #1106, tables get nothing. `SpatialData.write()` hands a table to anndata with no seam of any kind, so the on-disk shard geometry of the largest array in many stores is whatever anndata's global defaults happen to be that release. The issue is one sentence long and settles no design question.

Why it matters, concretely: Thyra writes mass-spectrometry imaging data where the table *is* the payload. anndata >= 0.13 defaults `auto_shard_zarr_v3 = True`, and against a zarr that sizes the auto inner chunk with `max_bytes=1024` where 1 MiB was intended, one MSI table landed in **393,310 files** and took **434 s** to write. Owning the shard budget took it to **12 files and 12.4 s** -- a 36x wall-clock improvement -- at the cost of one accepted 1.26x regression on single-ion-image random reads. Because spatialdata exposes no write-time lever, Thyra's only option was to hold zarr's process-global config open across the entire multi-minute `sdata.write`:

```python
with zarr.config.set({"array.target_shard_size_bytes": 128 * 1024 * 1024}):
    sdata.write(path)
```

`zarr.config` is donfig-backed and verified process-global, not thread-local -- any other zarr write in the process silently inherits Thyra's budget for the duration. That is the wart #1178 exists to remove.

## 2. Sequencing -- recommendation: go independent, land now

**Do not wait for #1106 and do not stack on it.** #1106 is `mergeable: CONFLICTING` / `mergeStateStatus: DIRTY`, its last *code* push was 2026-06-30, it carries two never-dismissed CHANGES_REQUESTED from LucaMarconato, and its author said on 2026-08-03 "No I need to update this". Worse, its head does not import at all: `@docstring_parameter(raster_write_kwargs=RASTER_WRITE_KWARGS_DOCS)` is applied to docstrings containing the placeholder `{RASTER_WRITE_KWARGS_DOCS}`, so `.format(**kwargs)` raises `KeyError` while the `SpatialData` class body executes. Branching from it is not viable.

Also correct two beliefs about #1106 before quoting it in review: it **no longer touches `config.py` at all** (all Settings/persistence/env-var work was reverted in commit `1f49846`, "revert: config settings restored / We will opt for scverse-misc settings in a follow up PR"), and its `base_options` dict comprehension in `_write_raster` is therefore dead code that always evaluates to `{}`.

File overlap if we go independent:
- **#1106** touches all nine of the sites this PR edits in `src/spatialdata/_core/spatialdata.py` (hunks at 1105, 1113, 1161, 1193, 1211, 1244, 1308, 1331, 1367). Every conflict is a one-line signature or forwarding addition -- mechanical to resolve either way round.
- **#1055** (this user's lazy tables, currently OPEN / MERGEABLE / CLEAN, head 45a4adf) does **not** functionally collide: its `spatialdata.py` hunks are all at 1868–1915 (the `read()` method), and its `io_table.py` hunk is `@@ -24,8 +24,34 @@`, confined to `_read_table`. Both branches merge cleanly with this one. Note it edits `src/spatialdata/_utils.py` while this PR edits `src/spatialdata/_io/_utils.py` -- different files.

Post the design rationale on #1178 **before** opening the PR (see §7 talking points), and offer `raster_shard_size_bytes` as the symmetric form so landing first does not invert melonora's deliberately reserved `raster_`/`table_` prefix scheme.

Base: `upstream/main @ ccf1ea0`.

## 3. The chosen API

A table is a heterogeneous tree of zarr arrays of mixed rank, length and dtype that all receive **one shared `dataset_kwargs`**. No chunk or shard *shape* can be valid for all of them -- verified empirically, three distinct failing keys: a 2-D `chunks` dies on `obs/_index`, a 1-D `chunks` dies on 2-D `obsm`, and any `shards` tuple dies on `uns` scalars. The only shape-independent expression of intent is a **byte budget**, and zarr already implements exactly that as `array.target_shard_size_bytes`, which anndata explicitly stands down for when a caller has set it.

So: one keyword-only `int`, delivered by scoping zarr's own config key plus anndata's settings around the existing anndata call. **Nothing is passed into `dataset_kwargs`.** There is no entry-point swap.

### Public signatures -- `src/spatialdata/_core/spatialdata.py`

```python
@_deprecation_alias(format="sdata_formats", version="0.7.0")
def write(
    self,
    file_path: str | Path,
    overwrite: bool = False,
    consolidate_metadata: bool = True,
    update_sdata_path: bool = True,
    sdata_formats: SpatialDataFormatType | list[SpatialDataFormatType] | None = None,
    shapes_geometry_encoding: Literal["WKB", "geoarrow"] | None = None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    convert_table_strings_to_categoricals: bool = False,
    *,
    table_shard_size_bytes: int | None = None,
) -> None: ...

def _write_element(
    self,
    element: SpatialElement | AnnData,
    zarr_container_path: Path,
    element_type: str,
    element_name: str,
    overwrite: bool,
    parsed_formats: dict[str, SpatialDataFormatType] | None = None,
    shapes_geometry_encoding: Literal["WKB", "geoarrow"] | None = None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    convert_table_strings_to_categoricals: bool = False,
    *,
    table_shard_size_bytes: int | None = None,
) -> None: ...

def write_element(
    self,
    element_name: str | list[str],
    overwrite: bool = False,
    sdata_formats: SpatialDataFormatType | list[SpatialDataFormatType] | None = None,
    shapes_geometry_encoding: Literal["WKB", "geoarrow"] | None = None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    convert_table_strings_to_categoricals: bool = False,
    *,
    table_shard_size_bytes: int | None = None,
) -> None: ...
```

### `src/spatialdata/_io/io_table.py`

Renamed at the boundary, exactly as PR #1170 renames `convert_table_strings_to_categoricals` → `convert_strings_to_categoricals`:

```python
def write_table(
    table: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    element_format: Format = CurrentTablesFormat(),
    convert_strings_to_categoricals: bool = False,
    *,
    shard_size_bytes: int | None = None,
) -> None: ...
```

### `src/spatialdata/_io/exceptions.py`

```python
class TableWriteOptionsError(ValueError):
    """Exception raised when table write options cannot be honoured by the active backend."""
```

`ValueError` subclass so nothing that currently catches `ValueError` breaks. **Re-export it from `spatialdata/__init__.py`** so the docstring cross-reference points at something users can import -- neither `FormatVersionUnknownError` nor `WritingToZarrV2DeprecationWarning` is currently re-exported, and a docstring promising a private-module class is a review comment waiting to happen.

### `src/spatialdata/_io/_utils.py`

```python
_MIN_ZARR_FOR_SHARD_BUDGET = Version("3.1.6")

def _validate_table_shard_size_bytes(
    table_shard_size_bytes: int | None,
    tables_zarr_format: int | None = None,
) -> None: ...

@contextmanager
def _table_shard_budget(shard_size_bytes: int | None) -> Generator[None, None, None]: ...
```

### Config fields: **none**

`spatialdata.config.Settings` is not touched. `None` already means "whatever anndata does". Adding a field re-opens the argument commit `1f49846` closed, and would land a field already slated for migration to scverse-misc (already a hard dependency at `pyproject.toml:48`; anndata has already migrated -- `class Settings(scverse_misc.Settings)`). A settings-backed default is strictly additive later: one dataclass field plus one `if shard_size_bytes is None:` line.

### Deliberately not included: `table_write_kwargs`

#1178 asks for *sharding* configuration. Compressors/serializer/filters are a separate feature, they are the only thing that would force the `AnnData.write_zarr` → `anndata.io.write_zarr` entry-point swap on the exact lines PR #1184 just repaired, and their shape is precisely what Tomaz-Vieira has a standing open objection to ("I don't think we should be surfacing the complexities (and instabilities!) of our dependencies onto users of `spatialdata`"). It folds in later unchanged.

### The mechanism (verified, do not re-derive)

anndata's `zarr_v3_sharding` (`_io/specs/methods.py:122` in 0.12.16; identical logic minus a `None` branch in released 0.13.3) computes:

```
has_auto_shard_size = zarr >= 3.1.4 and isinstance(zarr.config.get("array.target_shard_size_bytes"), int)
```

and takes `nullcontext()` -- i.e. does **not** install its own 1 GB default -- when that is True. anndata's own source comment: *"Users can ovetrride this nonetheless, hence the above checks."* ilan-gold recommends this exact call publicly on anndata #2415. So setting the key wins.

zarr then, in `_auto_partition`: inner chunk = `_guess_chunks(shape, item_size, max_bytes=1048576)`; shard = `chunk * num_chunks_per_shard_axis`, clamped per axis. **Divisibility (`shard % chunk == 0`) and `shard <= array` hold by construction, per array, at every rank, length and dtype.** Setting a budget also flips `has_auto_shard`, bypassing zarr's `a_shape // c_shape > 8` axis gate.

Measured (anndata 0.12.16 + zarr 3.2.1, 4000×2000 density-0.05 CSR table with categorical obs, nullable Int32, object string column, 2-D obsm, sparse layer, uns array + string scalars), one int producing six distinct geometries:

| budget | `X/data` (f8) | `X/indices` (i4) | `X/indptr` | `obsm/spatial` | uns scalars | files |
|---|---|---|---|---|---|---|
| none | None | None | None | None | None | 47 |
| 512 KiB | chunks (50000,) shards (50000,) | shards (100000,) | (4001,) | (4000,2) | None | 47 |
| 2 MiB | shards (250000,) | shards (400000,) | (4001,) | (4000,2) | None | 38 |

Monotone in the budget, resolved independently per array and per dtype. On a 20000×3000 nnz-1.2M table: 63 files unsharded → 41 files at a 128 MiB budget, total bytes 11,400,335 → 11,406,265 (+0.05%).

### Precedence rules (complete)

- **R1.** `table_shard_size_bytes=None` (default): spatialdata sets nothing and validates nothing. Byte-identical to today. This is what keeps `test_roundtrip`'s `_are_directories_identical` (`tests/io/test_readwrite.py:248`) green.
- **R2.** Positive int: for the duration of **one table's** anndata write, set `zarr.config["array.target_shard_size_bytes"]` **and** `anndata.settings.override(zarr_write_format=3, auto_shard_zarr_v3=True)`. Both restored on exit and on exception.
- **R3.** The explicit argument beats ambient config: a value the user set in `zarr.config` outside the call is overridden for that table's write and restored afterwards. Narrower scope wins. Verified: outer `999` → inner `134217728` → restored to `999`.
- **R4.** Forcing `auto_shard_zarr_v3=True` is scoped and one-directional. Passing a budget can turn sharding on for a user whose anndata setting had it off -- including an explicit `False`, verified -- because they asked for a shard size. Passing `None` never changes anything. There is no value that turns sharding off; that remains `anndata.settings.auto_shard_zarr_v3 = False`. No `Literal[False]` in the signature. **Say the override-of-an-explicit-False in the docstring**, not just in the design doc.
- **R5.** `shards` NEVER enters `dataset_kwargs`. Not "should not" -- must not. See §7.
- **R6.** Per-table budgets are expressed with `write_element(name, table_shard_size_bytes=...)`. No name-keyed mapping.
- **R7.** Both write branches of `write_table` are wrapped by the same context manager, so semantics are uniform across the whole anndata compatibility matrix including the `anndata>=0.12,<0.13` hatch leg, with zero version-conditional code.
- **R8.** `write_element(<an image>, table_shard_size_bytes=...)` is silently ignored, matching the `raster_compressor`-passed-for-a-table precedent. **State this in the docstring** -- the design left it unspecified.

## 4. Implementation, file by file, in call order

All paths are on `upstream/main @ ccf1ea0` in the clone at `%USERPROFILE%/Desktop/spatialdata`.

### 4.1 `src/spatialdata/_io/exceptions.py` (file is 26 lines; append)

Add `TableWriteOptionsError(ValueError)` with the docstring above, alongside the two existing definitions.

### 4.2 `src/spatialdata/__init__.py`

Re-export `TableWriteOptionsError` via the existing lazy `__getattr__` map (the same mechanism that exposes `settings` at lines 72–73 / 130–131 / 223–224).

### 4.3 `src/spatialdata/_io/_utils.py` (append after `_validate_compressor_args`, which ends at line 599)

The module **already imports** `zarr`, `contextmanager`, `Generator`, `Literal`, `Any`. Add only: `import anndata as ad`, `from importlib.metadata import version`, `from packaging.version import Version`. Use `Generator[None, None, None]`, not `Iterator` -- match house style at line 533.

```python
_MIN_ZARR_FOR_SHARD_BUDGET = Version("3.1.6")


def _validate_table_shard_size_bytes(
    table_shard_size_bytes: int | None,
    tables_zarr_format: int | None = None,
) -> None:
    """Validate the table shard budget before any element is written."""
    if table_shard_size_bytes is None:
        return
    if isinstance(table_shard_size_bytes, bool) or not isinstance(table_shard_size_bytes, int) \
            or table_shard_size_bytes <= 0:
        raise TableWriteOptionsError(<E1>)
    if Version(version("zarr")) < _MIN_ZARR_FOR_SHARD_BUDGET:
        raise TableWriteOptionsError(<E2>)
    settings_obj = getattr(ad, "settings", None)
    if settings_obj is None or not hasattr(settings_obj, "auto_shard_zarr_v3"):
        raise TableWriteOptionsError(<E3>)
    if tables_zarr_format is not None and tables_zarr_format != 3:
        raise TableWriteOptionsError(<E4>)


@contextmanager
def _table_shard_budget(shard_size_bytes: int | None) -> Generator[None, None, None]:
    """Scope a per-array uncompressed shard byte budget over a single anndata write.

    anndata injects ``shards="auto"`` itself, at the four writers where it is safe, and yields to a
    caller-set ``array.target_shard_size_bytes`` instead of installing its own 1 GB default. So the
    budget is delivered by setting that key and enabling anndata's auto-sharding, both scoped to this
    call -- NEVER by putting ``shards`` into ``dataset_kwargs``.

    Doing the latter would push ``shards="auto"`` onto the rank-0 scalars in ``uns/spatialdata_attrs``
    (anndata's ``_remove_scalar_compression_args`` strips ``chunks`` from scalar kwargs but not
    ``shards``), where zarr's ``_guess_num_chunks_per_axis_shard`` NEVER TERMINATES: with
    ``num_axes == 0``, ``(n + 1) ** 0 == 1`` makes the byte condition constant-true and ``all()`` over
    an empty zip is vacuously true. Two independent anndata mechanisms keep us safe today --
    ``write_scalar_zarr``/``write_null_zarr`` never call ``zarr_v3_sharding``, and the
    ``@zero_dim_array_as_scalar`` decorator re-dispatches 0-d ndarrays to the scalar writer before
    ``write_basic``'s sharding is reached. Neither is a contract; do not rely on passing ``shards``.

    ``zarr_write_format=3`` is forced alongside: anndata's ``open_write_group`` opens the table group
    with ``mode="w"`` and ``kwargs["zarr_format"] = settings.zarr_write_format``, DESTROYING and
    recreating the group spatialdata just made. Sharding is then gated on that recreated group's
    format, not on spatialdata's ``element_format``, so a user with ``zarr_write_format = 2`` would
    silently get an unsharded v2 table group inside a v3 store.
    """
    if shard_size_bytes is None:
        yield
        return
    with (
        zarr.config.set({"array.target_shard_size_bytes": shard_size_bytes}),
        ad.settings.override(zarr_write_format=3, auto_shard_zarr_v3=True),
    ):
        yield
```

Both `override` implementations are order-preserving for exactly this reason (anndata 0.12.x `SettingsManager.override` at `_settings.py:363` carries the comment *"always override zarr version before sharding"*; 0.13.x inherits `scverse_misc.Settings.override` at `scverse_misc/_settings.py:131`, a `@contextmanager` restoring in a `finally`). Verified valid on both.

### 4.4 `src/spatialdata/_core/spatialdata.py` -- nine mechanical sites

Exactly the set `convert_table_strings_to_categoricals` (PR #1170, commit `2debe5f`) already runs through.

| Line | Edit |
|---|---|
| 1108–1118 | `write` signature: append `*, table_shard_size_bytes: int | None = None` |
| ~1170 | `write` docstring: append the numpydoc block (§4.6) |
| 1174 | deferred import line: add `_validate_table_shard_size_bytes` |
| 1178 | after `_validate_compressor_args(raster_compressor)`, add `_validate_table_shard_size_bytes(table_shard_size_bytes, tables_zarr_format=parsed["tables"].zarr_format)` -- `parsed` is already computed at line 1177, **before** `zarr.create_group` at 1187 |
| 1192–1202 | `write` → `self._write_element(...)`: forward |
| 1210–1221 | `_write_element` signature: append |
| 1282–1289 | `elif element_type == "tables": write_table(...)`: add `shard_size_bytes=table_shard_size_bytes,` |
| 1293–1301 | `write_element` signature: append |
| ~1336 | in `write_element`, right after `parsed_formats = _parse_formats(formats=sdata_formats)` at 1335, add its own deferred `from spatialdata._io._utils import _validate_table_shard_size_bytes` and the same validation call. This method has **no** up-front validation today, not even for `raster_compressor` |
| ~1346 | `write_element` list-branch recursion: forward |
| ~1383 | `write_element` → `self._write_element(...)`: forward |

`@_deprecation_alias` is `functools.wraps` + `*args/**kwargs`, so keyword-only params pass through unharmed -- verified in source.

### 4.5 `src/spatialdata/_io/io_table.py` -- the only file doing real work

1. Imports: add `_table_shard_budget, _validate_table_shard_size_bytes` to the existing `from spatialdata._io._utils import _resolve_zarr_store`; add `TableWriteOptionsError` to the existing `from spatialdata._io.exceptions import ...`.
2. `write_table` signature (line 57): append `*, shard_size_bytes: int | None = None` + numpydoc entry.
3. At the top of the body, call `_validate_table_shard_size_bytes(shard_size_bytes, tables_zarr_format=element_format.zarr_format)`. This is **defence in depth** for direct callers of the `spatialdata._io.write_table` entry point -- the real early gate is in `write()`/`write_element()`. Position it before `group.require_group(name=name)` (line 93) so no half-created group is left.
4. Wrap the existing `if element_format.zarr_format == 3 and Version(version("anndata")) >= Version("0.13"): ... else: ...` block (lines 101–120) in `with _table_shard_budget(shard_size_bytes):`. Re-indent one level. **Nothing inside changes.** `table.write_zarr(store=resolved_store, consolidate_metadata=False, convert_strings_to_categoricals=...)` stays byte-for-byte; `write_adata(group, name, table)` stays byte-for-byte.
5. **Everything from `table_group = group[name]` (line 126) down is untouched.** The re-fetch, its issue-#1183 comment, and the five `table_group.attrs[...]` assignments stay exactly where they are. The `with` block closes above them -- deliberately, because the re-fetch writes attributes, not arrays.

### 4.6 Docstring (identical in `write` and `write_element`; adapted in `write_table`)

```
table_shard_size_bytes
    Target size, in bytes of *uncompressed* data, of a single Zarr shard for every array written
    inside a table group -- ``X`` and, for a sparse ``X``, its ``data``/``indices``/``indptr``
    components; ``obs`` and ``var`` columns; ``obsm``; ``layers``; array-valued ``uns`` entries.
    Sharding packs many chunks into one file, keeping the number of files on disk manageable for
    large tables. Each array is sharded independently, so a single value is meaningful regardless
    of that array's rank, length or dtype; the shard is always a whole number of inner chunks and
    never larger than the array itself. If the requested budget is smaller than the automatically
    chosen inner chunk (~1 MiB), the layout degenerates to one chunk per shard and the resulting
    shard may exceed the budget. Scalars are never sharded. If `None` (default), shard sizing is
    left entirely to :mod:`anndata`, which currently targets 1 GB. Passing a value enables
    anndata's zarr v3 auto-sharding for the duration of the write, overriding
    ``anndata.settings.auto_shard_zarr_v3`` even where it was explicitly set to `False`. Ignored
    for non-table elements. Requires a table format with ``zarr_format == 3``, ``zarr>=3.1.6`` and
    an anndata that supports zarr v3 auto-sharding; otherwise
    :class:`~spatialdata.TableWriteOptionsError` (a `ValueError`) is raised before anything is
    written.
```

## 5. Test plan

**Location: append at the END of `%USERPROFILE%/Desktop/spatialdata/tests/io/test_readwrite.py`** (currently 1370 lines, last test at :1314). **Not** at ~:830 -- PR #1106's test diff is `@@ -820,6 +841,168 @@`, the exact same insertion point, and #1055's is `@@ -1368,3 +1372,172 @@`. Appending at the end reduces the conflict to a trivial ordering rebase.

**Fixtures.** The shipped `_get_table` (`tests/conftest.py:344`) is `AnnData(RNG.normal(size=(100, 10)))` -- 8 kB, proves nothing. Build a purpose-built CSR AnnData (**4000 × 2000, density 0.05**, with categorical obs, a 2-D `obsm`, and `uns` scalars) the way #1106's raster sharding tests build 800×1000 arrays. Budgets **512 KiB and 2 MiB** -- measured to actually differentiate on that fixture. A 128 MiB budget and anndata's own 1 GB default produce byte-identical stores on anything smaller than ~1.2M nnz, so a large budget tests nothing.

Use the module-level `SDATA_FORMATS` (line 58) + `@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)` idiom. Add `@pytest.mark.skipif(Version(version("zarr")) < Version("3.1.6"), ...)` on the geometry tests. **No `filterwarnings` decorators are needed** -- anndata's four `zarr_v3_sharding` call sites are already `@suppress_autoshard_warning`, and a full budgeted write inside `warnings.catch_warnings(record=True)` with `simplefilter("always")` recorded zero shard-related warnings. (Note `[tool.pytest]` in pyproject is the wrong table name -- `markers` and `filterwarnings` there are silently ignored anyway.)

**How to assert on-disk geometry -- this is the assertion class that catches the real bug.** Reopen the written store directly and read array properties; never round-trip values.

```python
g = zarr.open_group(tmpdir, mode="r")
arr = g["tables"]["table"]["X"]["data"]
assert arr.shards is not None
assert arr.shards[0] % arr.chunks[0] == 0
```

For the encoding-attrs guard, read raw JSON as the existing #1183 test does: `json.loads((tmpdir / "tables" / "table" / "zarr.json").read_text())["attributes"]`.

| # | Test | Assertion |
|---|---|---|
| T1 | `test_table_shard_size_bytes_bounds_shard_size` | Two budgets (512 KiB, 2 MiB). For `tables/<name>/X/data`: `shards is not None`; `shards[0] % chunks[0] == 0`; **`shard_bytes <= max(budget, chunk_bytes)`** -- *not* `<= budget*; and the small budget yields a **strictly smaller** shard than the large one. That last clause is what actually tests the lever rather than anndata's default. |
| T2 | `test_table_scalars_are_never_sharded` | With a budget set, walk the whole written table group and assert `shards is None` for **every** array with `shape == ()`. Do **not** assert on `uns/spatialdata_attrs/region` -- for a multi-region table that key is a 1-D vlen string array and *is* correctly sharded (measured: `region=["a","b"]` → shape (2,) shards (2,)). Parametrize over single-str and list `region`. `region_key`/`instance_key` are always rank-0. **This is the §7 hazard guard.** |
| T3 | `test_no_shards_key_reaches_anndata` | Monkeypatch `spatialdata._io.io_table.write_adata` (and drive the `write_zarr` branch with a `write_dispatched` callback) to assert `"shards" not in dataset_kwargs`, always. A **fast** guard: T2's failure mode is a **hang**, not an assertion failure, and under CI's `pytest -n auto --dist worksteal` a hang stalls a worker until job timeout with no diagnostic. |
| T4 | `test_table_shard_size_bytes_none_is_unchanged` | Default write is byte-identical to a write with the argument absent. Protects `test_roundtrip` (:236, `_are_directories_identical`). |
| T5 | `test_table_shard_budget_restores_global_state` | After the write -- **and after a write that raises** (monkeypatch `write_adata` to raise) -- `zarr.config.get("array.target_shard_size_bytes")`, `anndata.settings.auto_shard_zarr_v3` and `anndata.settings.zarr_write_format` are unchanged. **Mandatory**, not optional. |
| T6 | `test_table_shard_size_bytes_rejected_on_zarr_v2` | `pytest.raises(TableWriteOptionsError, match="requires a zarr v3 table format")` on a `TablesFormatV01` container, **and assert the store directory does not exist / is empty afterwards** -- this is the regression guard for the early-raise fix. |
| T7 | `test_table_shard_size_bytes_validation` | `0`, `-1`, `1.5`, `True` each raise `TableWriteOptionsError`, and **nothing was written to disk**. Same for `write_element`. |
| T8 | extend `test_table_group_keeps_anndata_encoding_metadata` (:691) | Re-run the #1183 guard with a budget set: `zarr.json` attributes still contain `encoding-type`, `encoding-version` alongside spatialdata's five. Verified they do. |

## 6. Version floors and zarr-format-2 behaviour

**No `pyproject.toml` change.** `zarr>=3.0.0`, `anndata>=0.9.1`, `dask>=2026.3.0` and `distributed>=2026.3.0` all stay exactly as they are. All seven CI legs and all three `test-anndata-pandas` hatch legs stay green with zero dependency churn. Dependency-bound edits are what drew contention on #1106; this PR does not need them, so it does not make them. Do **not** carry #1106's `platformdirs` addition or its `distributed` removal.

Floors are enforced as **runtime raises**, hit only when the user actually passes the argument:

- **E1** -- not an int, or a bool, or `<= 0`: `` `table_shard_size_bytes` must be a positive int, got {value!r}. ``
- **E2** -- zarr < 3.1.6: `` `table_shard_size_bytes` requires zarr >= 3.1.6, got {version}. zarr 3.1.4 added `array.target_shard_size_bytes`, but 3.1.4 and 3.1.5 still size the inner chunk with max_bytes=1024 where 1 MiB was intended (fixed by zarr-python#3603), which would put ~130k inner chunks in a 128 MiB shard. `` -- **verified at the tags**: v3.1.4 `chunk_grids.py:267` and v3.1.5:267 both read `max_bytes=1024`; v3.1.6:272 reads `max_bytes=1048576`. `target_shard_size_bytes` is present from 3.1.4. The widely repeated "3.1.4 fixes it" claim is wrong by two patch releases.
- **E3** -- anndata without zarr-v3 auto-sharding: guard as `getattr(ad, "settings", None)`, **not** `hasattr(ad.settings, ...)` -- `anndata.settings` did not exist before 0.10 and the pin admits 0.9.1, so the naive form raises a bare `AttributeError`. Use `hasattr`, not a `Version` comparison: the exact 0.12.x release that registered the setting is not pinned down, and 0.12.16 has it.
- **E4** -- `zarr_format == 2`: `` `table_shard_size_bytes` requires a zarr v3 table format; got {type(element_format).__name__} (zarr_format=2). Sharding does not exist in zarr format 2. ``

**E4 must be raised in `write()` and `write_element()`, not only in `write_table`.** This is the single most important verification correction. Applying the design verbatim and running it produced the correct error -- but the store on disk already contained `['.zattrs', '.zgroup', 'images', 'tables']` with `images/img` fully written. `_write_element` calls `_get_groups_for_element` (which *creates* the element group) before `write_table`, and `write()` loops `gen_elements()` writing every preceding element first. `write_table` is never reached until images/labels/points/shapes are on disk. Both entry points already have the format available: `parsed = _parse_formats(sdata_formats)` at spatialdata.py:1177 and `parsed_formats = _parse_formats(formats=sdata_formats)` at :1335, both before any write.

This implements melonora's 2026-07-03 maintainer decision ("write_kwargs for users still writing format 2 will throw an error") with spatialdata's own message, not zarr's `Zarr format 2 arrays can only be created with shard_shape set to None` raised deep inside `zarr/core/array.py`. Note E4 is a **usability** choice, not a crash guard: the underlying path degrades safely -- forcing the budget on a v2 group completes with `shards = None` because `zarr_v3_sharding`'s `format == 3` gate short-circuits.

**Not raised, deliberately:** nothing warns when the budget is below the ~1 MiB inner chunk. It degenerates to `shard == chunk`. The docstring says so.

## 7. Known traps -- "if you do X you will silently get Y"

1. **If you put `shards="auto"` (or any `shards`) into `dataset_kwargs`, the write will HANG FOREVER on any real SpatialData table.** Every SpatialData table carries rank-0 string scalars in `uns/spatialdata_attrs`, and anndata's `_remove_scalar_compression_args` strips `chunks` from scalar kwargs but **not** `shards`. zarr's `_guess_num_chunks_per_axis_shard` never terminates on a rank-0 array when `array.target_shard_size_bytes` is set: `num_axes == 0` makes `(n + 1) ** 0 == 1`, so the byte condition is constant-true and `all()` over an empty zip is vacuously true. Verified by direct call and end-to-end on **zarr 3.2.1 and zarr 3.3.0 (latest)**: no traceback, no timeout, just an unbounded pure-Python loop. Safety today rests on **two independent** anndata mechanisms -- `write_scalar_zarr`/`write_null_zarr` never call `zarr_v3_sharding`, and `@zero_dim_array_as_scalar` (`anndata/_io/utils.py:315`) re-dispatches 0-d ndarrays before `write_basic`'s sharding is reached. Neither is a contract. **File the upstream zarr issue; the fix is two lines (`if num_axes == 0: return 1`) and link it from the PR.**
2. **If you rely on `element_format.zarr_format == 3` alone, a user with `anndata.settings.zarr_write_format = 2` gets a silently inert argument.** `AnnData.write_zarr` → `open_write_group(store, *, mode="w", **kwargs)` sets `kwargs["zarr_format"] = settings.zarr_write_format` and **destroys and recreates** the group spatialdata just made. Sharding is gated on `f.metadata.zarr_format` of the *recreated* group. Measured on anndata 0.13.3.post0 + zarr 3.3.0: table group format before=3, **after=2**, `X/data` shards=None, 67 files, no error, no warning, and a zarr v2 table group now sitting inside a v3 store. Fixed by adding `zarr_write_format=3` to the same `override`.
3. **If you copy #1106's `raster_write_kwargs` shape for tables, the documented happy path cannot execute.** A 2-D `chunks` raises on `obs/_index`; a 1-D `chunks` raises on 2-D `obsm`; any `shards` tuple raises on `uns` scalars; `shards` without `chunks` raises on divisibility; `chunks=<int>` broadcasts but `shards=<int>` raises `TypeError: 'int' object is not iterable`. Three distinct failing keys, all verified.
4. **If you forward `chunks=` to anndata, you ship a no-op for the consumer that filed the issue.** anndata's `chunks` callback fires only when `elem_name.lstrip("/") == "X"` **and** `not isinstance(elem, sparse.spmatrix)`. MSI tables are always CSR/CSC.
5. **If you swap to `anndata.io.write_zarr` to get `**ds_kwargs`, you inherit a `consolidate_metadata` reconciliation problem you do not have.** `AnnData.write_zarr` (0.13.3: `(self, store, *, chunks=None, convert_strings_to_categoricals=True, consolidate_metadata=True)`) has no `**ds_kwargs` -- but this design passes nothing, so the existing call stays byte-for-byte. **Do not swap the entry point.**
6. **If you assert `shards[0] * itemsize <= budget`, the test encodes a false invariant.** When the budget is below the auto inner chunk, `_guess_num_chunks_per_axis_shard` returns 1 early and shard == chunk, which exceeds the budget. Measured: budget 262144 on a 36M f8 array → chunks (70313,), shards (70313,) = 562,504 bytes > budget.
7. **If T2 asserts on `uns/spatialdata_attrs/region`, it fails on any multi-region table** -- a first-class case (`_get_table(region: str | list[str])`, `table_multiple_annotations` fixture). Measured: `region=["a","b"]` → shape (2,) shards (2,).
8. **If you set `zarr.config` in a scope narrower than the whole anndata call, anndata's 1 GB default silently takes over for the rest** -- arrays are created lazily throughout `write_dispatched`.
9. **If you read the anndata at `%USERPROFILE%/AppData/Roaming/Python/Python313/site-packages/anndata`, you will produce a wrong design.** That tree is 0.13.0.dev62+g3c90d3cc2, a stale main snapshot with `auto_shard_zarr_v3` default `False`, `zarr_write_format` default `2`, a plain-function `zarr_v3_sharding`, and a `validate_zarr_sharding` gate that exists **nowhere else**. Use `%USERPROFILE%/Desktop/spatialdata/.venv-test/Lib/site-packages/anndata` (0.12.16, chronologically newer) and fetch the 0.13.3 tag for released behaviour (`zarr_write_format = 3`, `auto_shard_zarr_v3: bool | None = True`).
10. **If you branch from or rebase onto `pr-1106-sharding`, `import spatialdata` fails** (`KeyError: 'RASTER_WRITE_KWARGS_DOCS'` at class-body execution). It is also 33 commits behind main and its `io_table.py` predates `convert_strings_to_categoricals`, the zarr-v3 path and the #1183 re-fetch.
11. **If you move the `table_group = group[name]` re-fetch, you re-break issue #1183** -- the stale handle's cached empty-attrs view erases anndata's `encoding-type`/`encoding-version`. Verified intact under a budget: attributes come back as `['encoding-type', 'encoding-version', 'instance_key', 'region', 'region_key', 'spatialdata-encoding-type', 'version']`.
12. **If you add a `TablesFormatV03`, you cascade six edits for nothing.** Table format versions encode only zarr v2 vs v3; the read path (`_read_table` → `anndata.read_zarr`) is geometry-blind. Precedent: `raster_compressor` (PR #944, commit `68dade6`) changed on-disk bytes with no format bump.
13. **If you write a test asserting `obs`/`var` string columns are unsharded, it fails.** Vlen string arrays **are** sharded (`write_vlen_string_array_zarr`); only `write_recarray_zarr` has sharding commented out pending zarr-python#3546. The widely repeated "string arrays are never sharded" gotcha is wrong.
14. **If you add a marker or `filterwarnings` to `pyproject.toml`, it does nothing** -- the config sits under `[tool.pytest]`, not `[tool.pytest.ini_options]`, and is silently ignored.
15. **Ruff has B006 active** (only B008 is ignored), so the new kwarg must default to `None`. Line length 120, `from __future__ import annotations` required, numpy docstrings mandatory in `src/`. mypy has `disallow_any_generics`, so no bare `dict`/`list`.

## 8. Definition of done

**Upstream PR (scverse/spatialdata), 5 files, ~130 lines:**
- [ ] Comment posted on #1178 first, covering the three talking points below.
- [ ] Branch from `upstream/main @ ccf1ea0`. 5 files: `_io/exceptions.py`, `__init__.py`, `_io/_utils.py`, `_io/io_table.py`, `_core/spatialdata.py`, plus `tests/io/test_readwrite.py`.
- [ ] All seven CI matrix jobs green (windows/3.12 min-dask, windows/3.14, ubuntu/3.12+3.13+3.14, macos/3.12, macos/3.14 prerelease which installs anndata from git main). Codecov gates are effectively no-ops (project target 1%, patch/changes disabled) -- tests must nonetheless *execute*.
- [ ] pre-commit green: prettier 3.9.6, mirrors-mypy 2.3.1 (src only), ruff 0.16.4 + ruff-format. pre-commit.ci will push fix commits onto the branch.
- [ ] T1–T8 all present and passing; T3 in particular, because T2's failure mode is a hang.
- [ ] **Docs: no file to edit.** `SpatialData.write`/`write_element` are documented solely by `.. autoclass:: SpatialData` with `inherited-members`; the numpydoc Parameters block *is* the documentation. Precedent confirmed: grepping `docs/` for `raster_compressor`, `storage_options`, `shapes_geometry_encoding` returns zero hits. Do not touch `docs/api/data_formats.md` (no new format class).
- [ ] **Changelog: nothing to write.** `CHANGELOG.md` is a 0-byte file; `docs/changelog.md` points at GitHub Releases. Release notes come from the PR title plus a **`release-added`** label. If the title cannot carry the note, add a `# Release notes` section at the end of the PR's first message per `docs/contributing.md:164`.
- [ ] Upstream zarr issue filed for the rank-0 non-termination, linked from the PR.

**Follow-up Thyra PR (this repo, `%USERPROFILE%/Desktop/Thyra`), gated on a released spatialdata carrying the kwarg:**
- [ ] `thyra/converters/spatialdata/base_spatialdata_converter.py:2530-2531` becomes `with _suppress_upstream_warnings(): sdata.write(str(self.output_path), table_shard_size_bytes=TABLE_SHARD_TARGET_BYTES)`.
- [ ] **Delete** `table_write_config()` (`thyra/converters/spatialdata/_chunking.py:64-73` -- the entire body is the `zarr.config.set` block), its import at `base_spatialdata_converter.py:24`, and `TestTableWriteConfig` (`tests/unit/converters/test_chunking_policy.py:68-86`, three donfig set/restore tests that become vacuous).
- [ ] **Keep** `TABLE_SHARD_TARGET_BYTES` (128 MiB) as the constant Thyra now *passes* rather than *installs*, and `MIN_TABLE_SHARD_BYTES` as the regression floor. The on-disk geometry assertions at `test_chunking_policy.py:153-168` are unchanged and still pass -- identical layout, same zarr key, same value, scoped by the library instead of by application code.
- [ ] **Rewrite** `test_save_output_holds_the_policy_open` (:145) from "assert the context manager was entered" to "assert `sdata.write` received `table_shard_size_bytes=TABLE_SHARD_TARGET_BYTES`" -- strictly stronger, because it asserts the *value*.
- [ ] Move two pins in `pyproject.toml`: `spatialdata>=0.8.0,<0.9` → the first release carrying the kwarg (v0.8.0's `write_table` is a 20-line function whose whole write is `write_adata(group, name, table)`; no design can land inside that range); and `zarr>=3.1.4,<3.2` → `>=3.1.6,<3.2`. The second is a correction Thyra owes anyway -- `pyproject.toml:94-103` and the `_chunking.py:57-61` docstring both credit 3.1.4 with the `max_bytes` fix, which is wrong by two releases. The lockfile resolves 3.1.6 so the shipped artifact is fine, but the declared floor admits two versions where the budget produces a pathological inner-chunk count; E2 turns that latent mis-pin into a loud failure.
- [ ] Correct `_chunking.py`'s claim that spatialdata "hands the table straight to `anndata.write_elem`" -- main calls `AnnData.write_zarr` on the v3 path.
- [x] **Done in Thyra (issue #218, 2026-09-08):** `StreamingSpatialDataConverter` used to hand-write the final store with its own `zarr.open_group(mode="w")` and `create_array` calls (`z_chunk = 1_000_000`, no shards) and never routed the table through spatialdata's writer, so no upstream seam reached it. Every table now goes through the inherited `_save_output` as an AnnData over memmapped CSC arrays, so `table_write_config()` covers the summed table too and the deletion stays clean and total.

### What to say on #1178 before opening the PR

1. **The asymmetry with #1106 is deliberate.** A raster scale is one N-d array of known rank, so a dict of `zarr.create_array` kwargs is coherent. A table is a heterogeneous tree sharing one `dataset_kwargs`, where a 2-D `chunks` dies on `obs/_index`, a 1-D `chunks` dies on 2-D `obsm`, and any `shards` tuple dies on `uns` scalars. An API whose documented happy path cannot execute is worse than no API. This directly answers LucaMarconato's twice-stated objection to tuple-shaped settings ("for cyx data we would need `len(raster_chunks) == 3`, while for yx data we would need `len(raster_chunks) == 2` … I'd remove altogether"). **Offer the corollary:** when #1106's settings return, they should return as `raster_shard_size_bytes` -- keeping melonora's reserved prefix scheme intact while supplying the half he deferred.
2. **The global is narrowed, not removed.** spatialdata sets `zarr.config` and `anndata.settings` for the duration of one anndata call. That is unavoidable: anndata's `shards="auto"` injection is settings-driven, and the only way to bypass it -- passing `shards` yourself -- is the path that hangs on rank-0 arrays. What changes is scope: from "the whole multi-minute `sdata.write`, held by application code" to "one table's write, held by the library, restored on exception, expressed as a per-call argument". Say this plainly rather than claiming the wart is gone. It matches ilan-gold's stated position: global settings are acceptable "in a targeted way where you know it is what you want", scoped to bulk operations.
3. **Sequencing.** Independent of #1106, not stacked. State that the nine forwarding lines will trivially conflict with #1106 so melonora is not surprised, and that #1055 does not conflict at all.

## Open questions for the implementer

- Whether maintainers want `TableWriteOptionsError` re-exported from `spatialdata/__init__.py` (this plan says yes, so the docstring cross-reference resolves) or prefer the docstring to just say "a `ValueError`" and keep the exception private like the two existing ones.
- Whether forcing `zarr_write_format=3` inside the budget scope (which also incidentally fixes the pre-existing "v2 table group inside a v3 store" bug for that call) is acceptable as a side effect, or whether maintainers would rather validate `anndata.settings.zarr_write_format` and raise `TableWriteOptionsError` instead.

---

# PART 2 -- THE HANDOFF PROMPT

`````
Implement scverse/spatialdata issue #1178 ("No sharding configuration exposed for tables"):
https://github.com/scverse/spatialdata/issues/1178

## Repo and branch

Work in the spatialdata clone at %USERPROFILE%/Desktop/spatialdata (Windows; PowerShell is
primary, a Bash tool is also available). Remotes: origin = Tomatokeftes fork, upstream =
scverse/spatialdata. There is a test venv at .venv-test (anndata 0.12.16, zarr 3.2.1).

Branch FROM upstream/main @ ccf1ea0 (re-verified as the tip; if it has moved, re-check the line
numbers below before trusting them). Name the branch something like feat/table-shard-size-bytes.

DO NOT branch from, rebase onto, or cherry-pick the local branch pr-1106-sharding. It does not
import (KeyError: 'RASTER_WRITE_KWARGS_DOCS' raised at class-body execution of
_core/spatialdata.py:1131), it is 33 commits behind main, and its io_table.py predates the
convert_strings_to_categoricals parameter, the zarr-v3 write path, and the issue-#1183 re-fetch fix.
Treat #1106 only as a design reference, and note that its settings/config layer was REVERTED in
commit 1f49846 - config.py is byte-identical between main and that branch.

DO NOT read the anndata at %USERPROFILE%/AppData/Roaming/Python/Python313/site-packages/anndata.
It is a stale main snapshot (0.13.0.dev62) whose sharding API, setting types and defaults all
disagree with reality. Use .venv-test/Lib/site-packages/anndata (0.12.16) and, for released
behaviour, fetch the 0.13.3 tag from raw.githubusercontent.com/scverse/anndata/0.13.3/.

## The design (already decided; do not redesign)

Add ONE keyword-only int: `table_shard_size_bytes`, a target size in bytes of UNCOMPRESSED data for
a single zarr shard of every array inside a table group.

WHY a scalar byte budget and not a chunks/shards tuple: a table is a heterogeneous tree of zarr
arrays of mixed rank, length and dtype that all receive ONE shared `dataset_kwargs` from anndata.
Verified empirically: a 2-D `chunks` raises on obs/_index; a 1-D `chunks` raises on 2-D obsm; any
`shards` tuple raises on uns scalars; `shards` without `chunks` raises on divisibility; `chunks=<int>`
broadcasts but `shards=<int>` raises TypeError. Copying #1106's `raster_write_kwargs` shape would
ship an API whose documented happy path cannot execute.

HOW it is delivered: spatialdata passes NOTHING into dataset_kwargs. It scopes two process globals
around the existing anndata call, for one table's write only:
  - zarr.config["array.target_shard_size_bytes"] = <budget>
  - anndata.settings.override(zarr_write_format=3, auto_shard_zarr_v3=True)
anndata's zarr_v3_sharding then injects shards="auto" itself, at the four writers where it is safe,
and yields to the caller-set budget instead of installing its own 1 GB default (its `has_auto_shard_size`
check; its own source comment: "Users can ovetrride this nonetheless, hence the above checks").
zarr derives shard = chunk * n per array, so `shard % chunk == 0` and `shard <= array` hold BY
CONSTRUCTION at every rank, length and dtype.

DO NOT swap the anndata entry point. `AnnData.write_zarr` has no **ds_kwargs, but that is irrelevant
because nothing is passed. The existing `table.write_zarr(store=resolved_store,
consolidate_metadata=False, convert_strings_to_categoricals=...)` call and the `write_adata(group,
name, table)` fallback both stay byte-for-byte unchanged.

## Signatures

src/spatialdata/_core/spatialdata.py - append `*, table_shard_size_bytes: int | None = None` to
`write` (line 1108), `_write_element` (1210) and `write_element` (1293).

src/spatialdata/_io/io_table.py - append `*, shard_size_bytes: int | None = None` to `write_table`
(line 57). Renamed at the boundary, exactly as PR #1170 renames convert_table_strings_to_categoricals
-> convert_strings_to_categoricals.

src/spatialdata/_io/exceptions.py - add `class TableWriteOptionsError(ValueError)` with docstring
"Exception raised when table write options cannot be honoured by the active backend."

src/spatialdata/__init__.py - re-export TableWriteOptionsError via the existing lazy __getattr__ map
(same mechanism that exposes `settings` at lines 72-73 / 130-131 / 223-224).

src/spatialdata/_io/_utils.py - add, after `_validate_compressor_args` (ends at line 599):
    _MIN_ZARR_FOR_SHARD_BUDGET = Version("3.1.6")
    def _validate_table_shard_size_bytes(table_shard_size_bytes: int | None,
                                         tables_zarr_format: int | None = None) -> None
    @contextmanager
    def _table_shard_budget(shard_size_bytes: int | None) -> Generator[None, None, None]
The module ALREADY imports zarr, contextmanager, Generator, Literal, Any. Add only `import anndata as
ad`, `from importlib.metadata import version`, `from packaging.version import Version`. Use
Generator[None, None, None], not Iterator (house style, see line 533).

NO change to spatialdata/config.py. NO new Settings field. NO format.py change / no TablesFormatV03
(table format versions encode only zarr v2 vs v3; the read path is geometry-blind). NO pyproject.toml
change. NO docs file change. NO changelog (CHANGELOG.md is 0 bytes; release notes come from the PR
title plus a `release-added` label).

## Edit sites, in call order (line numbers on upstream/main @ ccf1ea0)

_core/spatialdata.py:
  1108-1118  write signature: append the kwarg
  ~1170      write docstring: append the numpydoc block
  1174       deferred import line: add _validate_table_shard_size_bytes
  1178       after _validate_compressor_args(raster_compressor), add
             _validate_table_shard_size_bytes(table_shard_size_bytes,
                                              tables_zarr_format=parsed["tables"].zarr_format)
             (`parsed` is computed at 1177, BEFORE zarr.create_group at 1187)
  1192-1202  write -> self._write_element(...): forward
  1210-1221  _write_element signature: append
  1282-1289  the `elif element_type == "tables": write_table(...)` branch: add
             shard_size_bytes=table_shard_size_bytes,
  1293-1301  write_element signature: append
  ~1336      in write_element, right after `parsed_formats = _parse_formats(formats=sdata_formats)`
             at 1335, add its own deferred import of _validate_table_shard_size_bytes and the same
             validation call. This method has NO up-front validation today.
  ~1346      write_element list-branch recursion: forward
  ~1383      write_element -> self._write_element(...): forward

_io/io_table.py:
  imports    add _table_shard_budget, _validate_table_shard_size_bytes to the existing
             `from spatialdata._io._utils import _resolve_zarr_store`; add TableWriteOptionsError
             to the existing exceptions import
  57         signature + numpydoc entry
  top of body  _validate_table_shard_size_bytes(shard_size_bytes,
                 tables_zarr_format=element_format.zarr_format) -- defence in depth for direct
                 callers; must sit BEFORE group.require_group(name=name) at line 93 so no
                 half-created group is left
  101-120    wrap the existing `if element_format.zarr_format == 3 and Version(version("anndata"))
             >= Version("0.13"): ... else: ...` block in `with _table_shard_budget(shard_size_bytes):`
             Re-indent one level. NOTHING INSIDE CHANGES.
  126-132    UNTOUCHED. `table_group = group[name]`, its issue-#1183 comment, and the five
             table_group.attrs[...] assignments stay exactly where they are. The `with` closes ABOVE
             them - deliberately, because the re-fetch writes attributes, not arrays.

## Error behaviour

All errors are TableWriteOptionsError (subclasses ValueError so nothing that catches ValueError
breaks). E1-E4 are ALL raised up front in write() and write_element(), before anything reaches disk.

  E1  not an int, or a bool, or <= 0:
      "`table_shard_size_bytes` must be a positive int, got {value!r}."
  E2  zarr < 3.1.6:
      "`table_shard_size_bytes` requires zarr >= 3.1.6, got {version}. zarr 3.1.4 added
       `array.target_shard_size_bytes`, but 3.1.4 and 3.1.5 still size the inner chunk with
       max_bytes=1024 where 1 MiB was intended (fixed by zarr-python#3603), which would put ~130k
       inner chunks in a 128 MiB shard."
      (Verified at the tags: v3.1.4 and v3.1.5 chunk_grids.py:267 both read max_bytes=1024;
       v3.1.6:272 reads max_bytes=1048576.)
  E3  anndata without zarr-v3 auto-sharding support. Guard as:
        settings_obj = getattr(ad, "settings", None)
        if settings_obj is None or not hasattr(settings_obj, "auto_shard_zarr_v3"): raise
      NOT `hasattr(ad.settings, ...)` - anndata.settings did not exist before 0.10 and pyproject
      pins anndata>=0.9.1, so the naive form raises a bare AttributeError. Use hasattr, not a
      Version comparison.
      Message: "`table_shard_size_bytes` requires an anndata that supports zarr v3 auto-sharding,
       got {version}."
  E4  tables zarr_format == 2:
      "`table_shard_size_bytes` requires a zarr v3 table format; got {name} (zarr_format=2).
       Sharding does not exist in zarr format 2."

E4 MUST be raised in write()/write_element(), NOT only inside write_table. This was verified by
running the naive version: the error fired correctly but the store already contained
['.zattrs', '.zgroup', 'images', 'tables'] with images/img fully written, because _write_element
calls _get_groups_for_element (which creates the element group) before write_table, and write()
loops gen_elements() writing every preceding element first. Both entry points already compute the
format via _parse_formats before any write.

E2/E3 are RUNTIME gates, not pyproject bumps. zarr>=3.0.0 and anndata>=0.9.1 stay untouched so all
seven CI legs and all three test-anndata-pandas hatch legs stay green with zero dependency churn.

Nothing warns when the budget is below the ~1 MiB inner chunk; it degenerates to shard == chunk.
Say so in the docstring.

## Precedence rules

R1  None (default) = set nothing, validate nothing. Byte-identical to today.
R2  positive int = both globals scoped to one table's write, restored on exit AND on exception.
R3  the explicit argument beats ambient zarr.config; the outer value is restored afterwards.
R4  forcing auto_shard_zarr_v3=True overrides even an explicit user False. Say this in the docstring.
    There is no value that turns sharding off. No Literal[False] in the signature.
R5  `shards` NEVER enters dataset_kwargs. See the HANG trap below.
R6  per-table budgets use write_element(name, table_shard_size_bytes=...). No name-keyed mapping.
R7  BOTH write branches of write_table are wrapped, so semantics are uniform across the anndata
    matrix including the anndata>=0.12,<0.13 hatch leg, with zero version-conditional code.
R8  write_element(<an image>, table_shard_size_bytes=...) is silently ignored, matching the
    raster_compressor-passed-for-a-table precedent. Say so in the docstring.

## TRAPS - read these before writing code

1. IF YOU PUT `shards` (INCLUDING "auto") INTO dataset_kwargs, THE WRITE HANGS FOREVER on any real
   SpatialData table. Every table carries rank-0 string scalars in uns/spatialdata_attrs, and
   anndata's _remove_scalar_compression_args strips `chunks` from scalar kwargs but NOT `shards`.
   zarr's _guess_num_chunks_per_axis_shard never terminates on a rank-0 array when
   array.target_shard_size_bytes is set: num_axes == 0 makes (n+1)**0 == 1, so the byte condition is
   constant-true and all() over an empty zip is vacuously true. Verified on zarr 3.2.1 AND 3.3.0
   (latest): no traceback, no timeout, an unbounded pure-Python loop. Safety today rests on TWO
   independent anndata mechanisms - write_scalar_zarr/write_null_zarr never call zarr_v3_sharding,
   and @zero_dim_array_as_scalar (anndata/_io/utils.py:315) re-dispatches 0-d ndarrays before
   write_basic's sharding is reached. Name BOTH in the _table_shard_budget docstring. This is ALREADY FILED
   upstream as zarr-developers/zarr-python#4304. Do not file another one. Link #4304 from the PR
   and from the _table_shard_budget docstring.
2. IF YOU GATE ONLY ON element_format.zarr_format == 3, a user with
   anndata.settings.zarr_write_format = 2 gets a SILENTLY INERT argument. AnndData.write_zarr ->
   open_write_group opens with mode="w" and kwargs["zarr_format"] = settings.zarr_write_format, i.e.
   it DESTROYS and RECREATES the group spatialdata just made; sharding is gated on that recreated
   group's format. Measured on anndata 0.13.3.post0 + zarr 3.3.0: table group format before=3,
   AFTER=2, X/data shards=None, 67 files, no error, no warning. This is why zarr_write_format=3 goes
   into the same override(). Both override implementations are order-preserving (anndata 0.12.x
   SettingsManager.override at _settings.py:363 carries "always override zarr version before
   sharding"; 0.13.x inherits scverse_misc.Settings.override at _settings.py:131).
3. IF YOU MOVE the `table_group = group[name]` re-fetch, you re-break issue #1183 - the stale
   handle's cached empty-attrs view erases anndata's encoding-type/encoding-version.
4. IF YOU FORWARD `chunks=` to anndata, it is a no-op for sparse X (anndata's callback fires only
   when elem_name == "X" AND not isinstance(elem, sparse.spmatrix)).
5. Vlen string arrays ARE sharded. Do not write a test asserting obs/var string columns are
   unsharded; only write_recarray_zarr has sharding disabled (pending zarr-python#3546).
6. pyproject's pytest config sits under [tool.pytest], not [tool.pytest.ini_options], so markers and
   filterwarnings there are silently ignored. You do not need any filterwarnings anyway: anndata's
   four zarr_v3_sharding call sites are already @suppress_autoshard_warning, and a full budgeted
   write inside warnings.catch_warnings(record=True) with simplefilter("always") recorded zero
   shard-related warnings.
7. Ruff: B006 is ACTIVE (only B008 ignored) so the kwarg must default to None; line length 120;
   `from __future__ import annotations` required; numpy docstrings mandatory in src/. mypy has
   disallow_any_generics, so no bare dict/list annotations.
8. @_deprecation_alias is functools.wraps + *args/**kwargs, so keyword-only params pass through
   unharmed. Verified.

## Tests

Append at the END of %USERPROFILE%/Desktop/spatialdata/tests/io/test_readwrite.py (currently
1370 lines, last test at :1314). NOT at ~:830 - PR #1106's test diff is @@ -820,6 +841,168 @@, the
exact same insertion point; #1055's is @@ -1368,3 +1372,172 @@.

Fixture: the shipped _get_table (tests/conftest.py:344) is AnnData(RNG.normal(size=(100, 10))) - 8 kB,
proves nothing. Build a purpose-built CSR AnnData, 4000 x 2000 at density 0.05, with categorical obs,
a 2-D obsm, and uns scalars. Use budgets 512 KiB and 2 MiB - measured to actually differentiate on
that fixture. A 128 MiB budget and anndata's own 1 GB default produce byte-identical stores on
anything smaller than ~1.2M nnz.

Use the module-level SDATA_FORMATS (line 58) + @pytest.mark.parametrize("sdata_container_format",
SDATA_FORMATS). Add @pytest.mark.skipif(Version(version("zarr")) < Version("3.1.6"), ...) on the
geometry tests.

Assert ON-DISK GEOMETRY by reopening the store directly - never round-trip values:
    g = zarr.open_group(tmpdir, mode="r")
    arr = g["tables"]["table"]["X"]["data"]
    assert arr.shards is not None
    assert arr.shards[0] % arr.chunks[0] == 0
For encoding attrs, read raw JSON as the existing #1183 test does:
    json.loads((tmpdir / "tables" / "table" / "zarr.json").read_text())["attributes"]

T1 test_table_shard_size_bytes_bounds_shard_size
   Two budgets. For tables/<name>/X/data: shards is not None; shards[0] % chunks[0] == 0;
   shard_bytes <= max(budget, chunk_bytes)  -- NOT `<= budget`, that is a FALSE INVARIANT: when the
   budget is below the auto inner chunk, _guess_num_chunks_per_axis_shard returns 1 early and
   shard == chunk, which exceeds the budget (measured: budget 262144 on a 36M f8 array -> chunks
   (70313,) shards (70313,) = 562,504 bytes). AND assert the small budget yields a STRICTLY SMALLER
   shard than the large one - that clause is what actually tests the lever.
T2 test_table_scalars_are_never_sharded
   With a budget set, walk the whole written table group and assert `shards is None` for EVERY array
   with shape == (). Do NOT assert on uns/spatialdata_attrs/region: for a multi-region table that key
   is a 1-D vlen string array and IS correctly sharded (measured: region=["a","b"] -> shape (2,)
   shards (2,)). region_key/instance_key are always rank-0. Parametrize over single-str and list
   region. This is the trap-1 regression guard.
T3 test_no_shards_key_reaches_anndata
   Assert `"shards" not in dataset_kwargs`, always - monkeypatch
   spatialdata._io.io_table.write_adata for the fallback branch, and use a write_dispatched callback
   for the write_zarr branch. A FAST guard, because T2's failure mode is a HANG, and under CI's
   `pytest -n auto --dist worksteal` a hang stalls a worker until job timeout with no diagnostic.
T4 test_table_shard_size_bytes_none_is_unchanged
   Default write byte-identical to a write with the argument absent. Protects test_roundtrip at :236
   (_are_directories_identical).
T5 test_table_shard_budget_restores_global_state
   After a normal write AND after a write that raises (monkeypatch write_adata to raise):
   zarr.config.get("array.target_shard_size_bytes"), anndata.settings.auto_shard_zarr_v3 and
   anndata.settings.zarr_write_format are all unchanged. MANDATORY.
T6 test_table_shard_size_bytes_rejected_on_zarr_v2
   pytest.raises(TableWriteOptionsError, match="requires a zarr v3 table format") on a
   TablesFormatV01 container, AND assert the store directory does not exist / is empty afterwards -
   the regression guard for the early-raise fix.
T7 test_table_shard_size_bytes_validation
   0, -1, 1.5, True each raise TableWriteOptionsError and write nothing to disk. Same for
   write_element.
T8 extend test_table_group_keeps_anndata_encoding_metadata (:691)
   Re-run the #1183 guard with a budget set: zarr.json attributes still contain encoding-type and
   encoding-version alongside spatialdata's five.

## Style

No emojis anywhere, including commit messages, the PR body and the issue comment. No em dashes.
Plain, short sentences. Do not add Claude or any AI tool as co-author or committer, and do not add
"Assisted-by" or "Generated with" trailers. Commit as the repo's configured git user only.

## Definition of done

- The design comment on issue #1178 is ALREADY POSTED:
  https://github.com/scverse/spatialdata/issues/1178#issuecomment-5494212751
  Do not post another one. It already states the API, why it is not symmetric with #1106, that the
  global is narrowed rather than removed, the offer of `raster_shard_size_bytes`, and the sequencing.
  Read it before you start; the PR description should be consistent with it. If maintainers reply
  asking for changes, follow their reply over this prompt.
- 5 source files + 1 test file, ~130 lines total.
- All seven CI matrix jobs green, including the macos/3.14 prerelease leg that installs anndata from
  git main. pre-commit green (prettier, mypy on src only, ruff + ruff-format); pre-commit.ci will
  push fix commits onto the branch.
- Label the PR `release-added`. Title must carry the release note, or add a `# Release notes` section
  at the end of the PR's first message (docs/contributing.md:164).
- Link zarr-developers/zarr-python#4304 (the rank-0 non-termination) from the PR. It is already
  filed; do not open a duplicate.

Verify locally with the repo's own venv before pushing:
  %USERPROFILE%/Desktop/spatialdata/.venv-test/Scripts/python.exe -m pytest tests/io/test_readwrite.py -k "roundtrip or table or shard or compress or chunk"
(The unmodified baseline of that selection is 44 passed, 79 deselected.)
`````
