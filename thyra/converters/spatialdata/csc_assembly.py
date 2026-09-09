"""Out-of-core assembly of a pixels x features CSC matrix, in two passes.

The sibling tables -- the mobility-resolved table and the demultiplexed
MS/MS table -- are each a sparse matrix whose columns are the *occupied*
cells of a key space known before anything is read: ``(m/z bin, mobility
channel)`` for the grid, ``(precursor, m/z bin)`` for the demultiplexer,
the source's own feature list for a shared mobility axis. A pixel's
entries can be enumerated from the source in one pass and again in a
second, identically, so nothing ever needs to be held for the whole
image. This module is that shape, factored out of the two tables so both
bound their memory the same way the summed table does (pre-scan, count,
scatter into a memmap); the summed table itself is now one of these too,
with every m/z bin a column (``StreamingSpatialDataConverter``).

**Pass 1 -- counting.** :meth:`CscAssembly.count` takes one row's unique
keys. A dense ``uint32`` count per key over the key span is the whole
state: ``span * 4`` bytes, known before the first pixel is read and
independent of how many pixels there are. :meth:`finish_counting` then
names the occupied keys, and that is the moment a caller checks its
feature ceiling -- before a single value has been buffered, which is what
the ceiling is for.

**Pass 2 -- scattering.** :meth:`allocate` sizes CSC ``indices`` and
``data`` arrays from the counts and backs them with files in a scratch
directory; :meth:`scatter` writes one row's values straight to their final
positions. Memory in this pass is one row plus the per-column write
cursor. When rows arrive in ascending order -- a raster acquisition read
in acquisition order, which is every source measured so far -- every
column's row indices come out ascending and the matrix is canonical
without a sort: the ``COO -> CSC`` conversion the tables used to pay for,
one scipy call over the whole image and the single largest phase of a
measured grid conversion, is gone rather than tuned. A source read in
some other order gets its columns sorted afterwards, a bounded chunk of
columns at a time (:func:`sort_csc_columns`, which the summed table
shares), so the stored matrix is canonical either way.

Canonical also means one entry per ``(row, column)``. A caller whose rows
can repeat a key -- the summed table, whose row is a grid position and
whose source may hold two spectra for one pixel -- sets
:attr:`CscAssembly.merge_duplicates`, and :func:`merge_duplicate_entries`
sums the repeats into one entry the same chunked way. That is what
``coo_matrix(...).tocsc()`` did for the route this replaced, and it is
what the tests in ``test_csc_assembly.py`` pin the whole engine against.

:meth:`matrix` wraps the memmaps as a :class:`scipy.sparse.csc_matrix`
without copying them (index dtypes are chosen up front by scipy's own
rule so its constructor takes the arrays as they are), and the AnnData
built on it is written from disk to disk. The scratch directory has to
outlive that write; :meth:`release` drops the mappings so the caller can
remove it, and :func:`release_when_collected` ties the directory's life to
the table for an API caller who has nowhere else to put it.
"""

import gc
import logging
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from ...errors import ConversionRefused

logger = logging.getLogger(__name__)

#: Bytes the per-key count array of pass 1 may take. It is ``span * 4``
#: with ``span`` the key space -- ``mass axis bins x mobility channels``
#: for the grid, which on a default-resampled timsTOF axis is 35.5M pairs
#: and 142 MB. A raw, unresampled axis of several million bins would push
#: it into gigabytes, and that is a source to resample rather than a
#: table to build: the same axis would also blow the feature ceiling.
MAX_COUNT_BYTES = 1 << 30

#: Entries loaded at once when a column chunk has to be sorted after the
#: scatter (a source read out of raster order). Bounds that pass to a few
#: hundred megabytes whatever the matrix size.
SORT_CHUNK_ENTRIES = 16_000_000

#: Process memory one m/z bin of the summed table costs, everything the
#: conversion builds per bin included. :data:`MAX_COUNT_BYTES` sizes the
#: 4 B/bin count array alone, and that is 2 percent of the real figure:
#: around it sit the axis itself, each table's ``average_spectrum`` and
#: the column sums it is divided out of (8 B/bin each), the ``var`` frame
#: with its index, the ``indptr`` and the write cursor, and anndata's own
#: copies during the write. Measured 2026-09-09 with a six-pixel
#: conversion at a forced bin count, peak working set above the
#: pre-conversion baseline:
#:
#: =============  ==================  ==========
#: bins           peak above baseline bytes/bin
#: =============  ==================  ==========
#: 200,000        56 MB               292
#: 1,000,000      231 MB              243
#: 2,000,000      429 MB              225
#: 5,000,000      966 MB              203
#: 10,000,000     1,802 MB            189
#: =============  ==================  ==========
#:
#: Issue #251 measured 208 B/bin at 20M bins and 337 B/bin at 200M, where
#: a 6-pixel dataset peaked at 67.5 GB. 200 is the figure those agree on.
#:
#: The ``var`` index is over half of it -- 117 B/bin peak at 20M bins --
#: and that is not something a cheaper construction fixes. Building the
#: labels as a numpy ``StringDType`` array rather than a Python list was
#: measured at both 5M and 20M bins: 583 MB against 584 MB, and 2,328 MB
#: against 2,329 MB. On pandas 3 the index becomes an Arrow-backed ``str``
#: either way and the peak is the conversion, not the labels. So the cost
#: is real, it is inherent to a string ``var`` index, and the answer is to
#: refuse an axis that cannot afford it rather than to shave the index.
#:
#: The figure is per bin of *one* table. A multi-slice source converted as
#: 2D writes one table per plane and each carries its own ``var`` copy and
#: its own mean spectrum, so its real per-bin cost is a multiple of this
#: and the guard under-projects it. That predates the per-table mean
#: (#243), which is close to neutral here: it dropped one dataset-wide
#: accumulator and one dataset-wide mean (8 B/bin each) and added 8 B/bin
#: per table.
AXIS_BYTES_PER_BIN = 200

#: Fractions of the machine's free memory the projected mass axis may
#: take before the conversion warns, and before it refuses. Fractions
#: rather than sizes, for the reason the grid's own guard gives: an axis
#: that is routine on a workstation is fatal on a laptop.
AXIS_WARN_FRACTION = 0.25
AXIS_REFUSE_FRACTION = 0.5

#: What to assume is free when the machine will not say. Generous on
#: purpose: a guess must never be the thing that refuses a conversion
#: that would have fitted.
ASSUMED_FREE_GB = 8.0

_COUNT_DTYPE = np.uint32


def available_memory_gb() -> float:
    """The machine's free memory in GB, or :data:`ASSUMED_FREE_GB`."""
    try:
        import psutil

        return float(psutil.virtual_memory().available) / 1024**3
    except Exception as e:  # pragma: no cover - platform-dependent
        logger.debug("Could not read available memory: %s", str(e))
        return ASSUMED_FREE_GB


def projected_axis_gb(n_bins: int) -> float:
    """What a mass axis of this many bins is expected to cost in memory."""
    return float(n_bins) * AXIS_BYTES_PER_BIN / 1024**3


def mass_axis_refusal(
    n_bins: int, available_gb: Optional[float] = None
) -> Optional[str]:
    """Why a mass axis this wide cannot be converted here, or ``None``.

    Asked *before* the axis is materialised, from the bin count the
    resampling plan resolved, because building it is already expensive:
    300M bins on a six-pixel dataset were refused in 2.3 s by the count
    array's own ceiling, but the process had reached 7.44 GB getting
    there. 200M bins passed that ceiling entirely -- ``span * 4`` is
    800 MB, under the 1 GiB it allows -- and took 217 s and 67.5 GB
    (issue #251).

    ``available_gb`` overrides the machine's own answer, for tests.
    """
    gb = projected_axis_gb(n_bins)
    free = available_memory_gb() if available_gb is None else float(available_gb)
    levers = (
        "resample to fewer mass bins (--resample-bins) or ask for a wider "
        "bin (--resample-width-at-mz)"
    )
    if gb > free * AXIS_REFUSE_FRACTION:
        return (
            f"the mass axis would have {int(n_bins):,} bins, and the "
            f"structures built per bin are projected to need {gb:.1f} GB "
            f"({AXIS_BYTES_PER_BIN} bytes per bin, measured) against "
            f"{free:.1f} GB free, more than the {AXIS_REFUSE_FRACTION:.0%} of "
            f"free memory a conversion may take; {levers}"
        )
    if gb > free * AXIS_WARN_FRACTION:
        logger.warning(
            "The mass axis has %s bins, projected to need %.1f GB of the "
            "%.1f GB free. It will be attempted; %s to make it smaller.",
            f"{int(n_bins):,}",
            gb,
            free,
            levers,
        )
    return None


def sort_csc_columns(
    indices: Any,
    data: Any,
    indptr: NDArray[Any],
    n_rows: int,
    chunk_entries: int = SORT_CHUNK_ENTRIES,
) -> None:
    """Put each column's entries in ascending row order, in place, chunk by chunk.

    A scatter writes a column's entries in the order the source handed its
    rows over. A raster read makes that ascending and the matrix canonical
    for free. Any other order -- a TDF acquisition of several areas
    measured one after another, whose frames come back area by area --
    leaves the row indices within a column unsorted, which scipy reports
    as non-canonical and which breaks every consumer that binary-searches
    a column. The columns are already grouped, so this is a sort *within*
    columns over a bounded slice of the arrays at a time. Every table --
    the summed one and its siblings -- is an assembly and sorts its
    memmaps with it.

    Args:
        indices: CSC row indices, one array-like (a memmap) of the
            non-zero count.
        data: CSC values, aligned with ``indices``.
        indptr: Column pointers, ``n_cols + 1`` entries of any integer
            dtype.
        n_rows: Row count of the matrix; every row index is below it.
        chunk_entries: Entries a chunk may hold. A column larger than
            this is sorted on its own.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    n_cols = int(indptr.size - 1)
    logger.info(
        "Rows arrived out of raster order; sorting %s columns in chunks",
        f"{n_cols:,}",
    )
    start_col = 0
    while start_col < n_cols:
        # The largest column range whose entries fit the chunk budget,
        # and at least one column even when that column alone exceeds it.
        end_col = int(
            np.searchsorted(indptr, indptr[start_col] + chunk_entries, side="right")
        )
        end_col = max(min(end_col - 1, n_cols), start_col + 1)
        lo, hi = int(indptr[start_col]), int(indptr[end_col])
        if hi > lo:
            rows = np.asarray(indices[lo:hi]).astype(np.int64)
            column = np.repeat(
                np.arange(start_col, end_col, dtype=np.int64),
                np.diff(indptr[start_col : end_col + 1]),
            )
            # One combined key, unique because a row occurs once per
            # column, sorted stably: the columns are already in order, so
            # the key is a sequence of nearly sorted runs that timsort
            # merges in a fraction of a general sort's time (measured
            # 0.5 s against 3.1 s for lexsort on 16M entries).
            order = np.argsort(column * n_rows + rows, kind="stable")
            indices[lo:hi] = rows[order]
            data[lo:hi] = np.asarray(data[lo:hi])[order]
        start_col = end_col


def merge_duplicate_entries(
    indices: Any,
    data: Any,
    indptr: NDArray[Any],
    chunk_entries: int = SORT_CHUNK_ENTRIES,
) -> Tuple[NDArray[np.int64], int]:
    """Sum entries that repeat a ``(column, row)`` and compact them away.

    ``coo_matrix(...).tocsc()`` -- the conversion this engine replaced --
    summed duplicate triples, so a pixel measured twice ended up as one
    row entry holding the sum. Scattering straight into the CSC arrays
    keeps both, which leaves the stored matrix non-canonical: scipy and
    dask quietly merge duplicates when they read it, but a consumer that
    binary-searches ``X/indices`` directly -- an ion-image column reader
    -- finds one of the two values and no way to know the other exists
    (issue #241). This restores the summation out of core.

    Each column's rows must already be ascending, which is what
    :func:`sort_csc_columns` guarantees; equal rows are then adjacent and
    a chunk's groups are one ``reduceat``. The compaction only ever moves
    entries left, so it is written into the same arrays in one forward
    pass, a bounded slice at a time.

    Args:
        indices: CSC row indices, sorted within each column, modified in
            place.
        data: CSC values aligned with ``indices``, modified in place.
        indptr: Column pointers of the matrix as scattered.
        chunk_entries: Entries a chunk may hold; column boundaries are
            never split, so a column larger than this is done on its own.

    Returns:
        The new column pointers and the surviving non-zero count. The
        first ``nnz`` entries of ``indices`` and ``data`` are the matrix;
        what follows them is stale.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    n_cols = int(indptr.size - 1)
    counts = np.zeros(n_cols, dtype=np.int64)
    out = 0
    start_col = 0
    while start_col < n_cols:
        end_col = int(
            np.searchsorted(indptr, indptr[start_col] + chunk_entries, side="right")
        )
        end_col = max(min(end_col - 1, n_cols), start_col + 1)
        lo, hi = int(indptr[start_col]), int(indptr[end_col])
        if hi > lo:
            # Copies, not memmap views: the compaction writes back over
            # this same range while these are still being read from.
            rows = np.array(indices[lo:hi], dtype=np.int64)
            values = np.array(data[lo:hi], dtype=np.float64)
            column = np.repeat(
                np.arange(start_col, end_col, dtype=np.int64),
                np.diff(indptr[start_col : end_col + 1]),
            )
            starts = np.empty(hi - lo, dtype=bool)
            starts[0] = True
            starts[1:] = (rows[1:] != rows[:-1]) | (column[1:] != column[:-1])
            group = np.flatnonzero(starts)
            merged_rows = rows[group]
            merged_values = np.add.reduceat(values, group)
            counts[start_col:end_col] = np.bincount(
                column[group] - start_col, minlength=end_col - start_col
            )
            n_new = int(group.size)
            indices[out : out + n_new] = merged_rows
            data[out : out + n_new] = merged_values
            out += n_new
        start_col = end_col
    merged_indptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(counts, out=merged_indptr[1:])
    return merged_indptr, out


def index_dtype(n_nonzeros: int, n_rows: int) -> Any:
    """The index dtype scipy would give a CSC matrix of this size.

    Its rule for a ``COO -> CSC`` conversion: ``int32`` unless the
    non-zero count or the row count needs more. Choosing the same dtype
    here is what lets the constructor take the memmaps without a copy,
    and what makes the stored ``indices`` and ``indptr`` identical to what
    the conversion route used to write.
    """
    limit = int(np.iinfo(np.int32).max)
    return np.int32 if max(int(n_nonzeros), int(n_rows)) < limit else np.int64


def count_bytes(span: int) -> int:
    """What pass 1 allocates for a key space of ``span`` keys."""
    return int(span) * int(np.dtype(_COUNT_DTYPE).itemsize)


def count_refusal(span: int) -> Optional[str]:
    """Why a key space this wide cannot be counted, or ``None``."""
    needed = count_bytes(span)
    if needed <= MAX_COUNT_BYTES:
        return None
    return (
        f"the feature space spans {int(span):,} keys and counting them needs "
        f"{needed / 1024**3:.1f} GB, above the {MAX_COUNT_BYTES / 1024**3:.0f} GB "
        f"this pass allows; resample to fewer mass bins (--resample-bins)"
    )


class CscAssembly:
    """A pixels x keys CSC matrix built out of core, one row at a time.

    Args:
        span: Size of the key space; every key handed in is in
            ``[0, span)``.
        n_rows: Number of rows of the finished matrix.
        keep_empty_columns: Whether every key of the span is a column of
            the matrix, occupied or not. Right when the span *is* a
            declared feature list (a shared mobility axis) whose empty
            features are still features; wrong for a grid, whose span is
            mostly empty by construction, so there only the occupied keys
            become columns.

    Keys within one :meth:`count` or :meth:`scatter` call must be unique
    and ascending -- one entry per occupied cell of that row, which is
    what the callers produce by collapsing a row before handing it over.
    :meth:`scatter` must be called for exactly the rows :meth:`count` was,
    in the same order; a row may be handed in more than once when its
    entries come in several disjoint groups (one per precursor, say).

    Those groups being disjoint is the caller's guarantee, not something
    checked here. A caller that *cannot* give it -- the summed table,
    whose row is a grid position and whose source may hold two spectra
    for one pixel -- sets :attr:`merge_duplicates`, and the repeated
    entries are summed into one when the matrix is handed out, which is
    what ``coo_matrix(...).tocsc()`` did for the route this replaced.
    """

    def __init__(
        self, span: int, n_rows: int, keep_empty_columns: bool = False
    ) -> None:
        """Allocate the count array; refuses a span too wide to count."""
        refusal = count_refusal(span)
        if refusal is not None:
            raise MemoryError(refusal)
        self.span = int(span)
        self.n_rows = int(n_rows)
        self._keep_empty = bool(keep_empty_columns)
        self._counts: Optional[NDArray[np.uint32]] = np.zeros(
            self.span, dtype=_COUNT_DTYPE
        )
        self.n_nonzeros = 0
        #: Whether every row so far arrived at or after the previous one.
        #: Ascending rows make the scattered columns canonical for free.
        self.rows_in_order = True
        #: Set by a caller that may hand the same ``(row, key)`` in twice --
        #: two spectra at one pixel. :meth:`matrix` then sums the repeats
        #: into one entry instead of storing both.
        self.merge_duplicates = False
        self._last_row = -1
        self.unique_keys: Optional[NDArray[np.int64]] = None
        self.indptr: Optional[NDArray[Any]] = None
        self._column_of_key: Optional[NDArray[np.uint32]] = None
        self._write_pos: Optional[NDArray[np.int64]] = None
        self._indices: Optional[np.memmap] = None
        self._data: Optional[np.memmap] = None
        self._index_dtype: Any = None

    def _note_row(self, row: int) -> None:
        if row < self._last_row:
            self.rows_in_order = False
        self._last_row = int(row)

    # -- pass 1 ----------------------------------------------------------

    def count(self, row: int, keys: NDArray[np.int64]) -> None:
        """Record the occupied keys of one row."""
        if self._counts is None:
            raise RuntimeError("Counting is finished; this row cannot be added")
        self._note_row(row)
        if keys.size:
            # Unique within the row, so a fancy-indexed increment is exact.
            self._counts[keys] += 1
            self.n_nonzeros += int(keys.size)

    def finish_counting(self) -> int:
        """Name the occupied keys; returns how many there are.

        Nothing wide is allocated here beyond the key list itself, so a
        caller can still refuse the matrix on that number.
        """
        if self._counts is None:
            raise RuntimeError("Counting is already finished")
        if self._keep_empty:
            self.unique_keys = np.arange(self.span, dtype=np.int64)
        else:
            self.unique_keys = np.flatnonzero(self._counts).astype(np.int64)
        self._last_row = -1
        return int(self.unique_keys.size)

    # -- pass 2 ----------------------------------------------------------

    def allocate(self, scratch: Path) -> None:
        """Size the CSC arrays from the counts and back them with files."""
        if self._counts is None or self.unique_keys is None:
            raise RuntimeError("allocate() needs finish_counting() first")
        counts = self._counts
        n_cols = int(self.unique_keys.size)
        self._index_dtype = index_dtype(self.n_nonzeros, self.n_rows)
        indptr = np.zeros(n_cols + 1, dtype=np.int64)
        np.cumsum(counts[self.unique_keys], out=indptr[1:])
        self.indptr = indptr.astype(self._index_dtype, copy=False)
        self._write_pos = indptr[:-1].copy()
        if self._keep_empty:
            # Every key is its own column; no map is needed.
            self._column_of_key = None
        else:
            # The key -> column map, in place of the counts: a running count
            # of occupied keys, so column = map[key] - 1 (an unoccupied key
            # would read as the column before it, and is never asked for).
            np.cumsum(counts != 0, out=counts)
            self._column_of_key = counts
        self._counts = None
        size = max(self.n_nonzeros, 1)
        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        self._indices = np.memmap(
            scratch / "csc_indices.bin",
            dtype=self._index_dtype,
            mode="w+",
            shape=(size,),
        )
        self._data = np.memmap(
            scratch / "csc_data.bin", dtype=np.float64, mode="w+", shape=(size,)
        )
        logger.info(
            "Scattering %s non-zeros over %s columns to %s (%.2f GB on disk)",
            f"{self.n_nonzeros:,}",
            f"{n_cols:,}",
            scratch,
            size * (np.dtype(self._index_dtype).itemsize + 8) / 1024**3,
        )

    def scatter(
        self, row: int, keys: NDArray[np.int64], values: NDArray[np.float64]
    ) -> None:
        """Write one row's values to their final CSC positions."""
        if self._write_pos is None:
            raise RuntimeError("scatter() needs allocate() first")
        if self._indices is None or self._data is None:
            raise RuntimeError("scatter() after release()")
        self._note_row(row)
        if not keys.size:
            return
        if self._column_of_key is None:
            columns = keys
        else:
            columns = self._column_of_key[keys].astype(np.int64) - 1
        destinations = self._write_pos[columns]
        self._indices[destinations] = row
        self._data[destinations] = values
        self._write_pos[columns] += 1

    def matrix(self) -> sparse.csc_matrix:
        """The finished, canonical matrix over the memmaps, without copying them.

        Every reserved slot must have been written: a column short of its
        count would carry zero-filled entries at row 0, out of order, which
        scipy reports as a non-canonical matrix. The callers guarantee it
        by scattering exactly the rows they counted; this checks.

        Raises:
            ConversionRefused: When pass 2 scattered fewer entries than
                pass 1 counted. That is a property of the source and its
                reader, not of this code, so it is spelled as a refusal
                like the per-row disagreement the converter checks for
                (issue #247) rather than as a traceback.
        """
        if self.indptr is None or self._write_pos is None:
            raise RuntimeError("matrix() needs allocate() first")
        if self._indices is None or self._data is None:
            raise RuntimeError("matrix() after release()")
        if not np.array_equal(self._write_pos, self.indptr[1:]):
            short = int(np.count_nonzero(self._write_pos != self.indptr[1:]))
            raise ConversionRefused(
                f"{short} columns were counted for more entries than were "
                "scattered; the two passes disagree on which rows exist"
            )
        if self.merge_duplicates:
            # The repeats need not have arrived next to each other -- a
            # coordinate can recur anywhere in the file -- so the columns
            # are sorted whatever order the rows came in, which is what
            # puts equal rows side by side for the merge.
            self._sort_columns()
            self._merge_duplicates()
        elif not self.rows_in_order:
            self._sort_columns()
        self._indices.flush()
        self._data.flush()
        n_cols = int(self.indptr.size - 1)
        nnz = self.n_nonzeros
        return sparse.csc_matrix(
            (self._data[:nnz], self._indices[:nnz], self.indptr),
            shape=(self.n_rows, n_cols),
        )

    def _merge_duplicates(self) -> None:
        """Sum the repeated ``(row, key)`` entries into one; see #241.

        Runs after the sort, and leaves the assembly describing the
        merged matrix: ``indptr`` and ``n_nonzeros`` shrink, the write
        cursor follows them so the slot check above still holds, and the
        flag is cleared so a second :meth:`matrix` call is a no-op rather
        than a second merge.
        """
        assert self.indptr is not None and self._indices is not None
        assert self._data is not None
        before = self.n_nonzeros
        merged, nnz = merge_duplicate_entries(
            self._indices,
            self._data,
            self.indptr,
            chunk_entries=SORT_CHUNK_ENTRIES,
        )
        self.indptr = merged.astype(self._index_dtype, copy=False)
        self._write_pos = merged[1:].copy()
        self.n_nonzeros = int(nnz)
        self.merge_duplicates = False
        if before != nnz:
            logger.info(
                "Summed %s entries that repeated a (row, column): %s "
                "non-zeros remain",
                f"{before - nnz:,}",
                f"{nnz:,}",
            )

    def _sort_columns(self) -> None:
        """Put each column's entries in ascending row order, chunk by chunk.

        Only needed when the source handed its rows out of order, or when
        a merge is coming and needs equal rows adjacent; see
        :func:`sort_csc_columns`. The chunk budget is read at call time
        so a test can shrink it through the module constant.
        """
        assert self.indptr is not None and self._indices is not None
        assert self._data is not None
        sort_csc_columns(
            self._indices,
            self._data,
            self.indptr,
            self.n_rows,
            chunk_entries=SORT_CHUNK_ENTRIES,
        )
        self.rows_in_order = True

    def release(self) -> None:
        """Drop the mappings so the scratch files can be removed.

        Arrays handed out by :meth:`matrix` are views on the same
        mappings, so the table built on them has to be dropped too before
        the directory can go -- Windows will not delete a mapped file.
        """
        self._indices = None
        self._data = None
        self._write_pos = None
        self._column_of_key = None


def scratch_directory(prefix: str, parent: Optional[Path] = None) -> Path:
    """A fresh directory for the memmaps of one assembly."""
    if parent is not None:
        Path(parent).mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=prefix, dir=None if parent is None else str(parent))
    )


def remove_scratch(path: Path) -> None:
    """Remove a scratch directory; say so if a mapping still pins it."""
    gc.collect()
    shutil.rmtree(path, ignore_errors=True)
    if Path(path).exists():
        logger.warning(
            "Could not remove the scratch directory %s (a table built on it "
            "is still referenced); remove it by hand once the store is written",
            path,
        )


def release_when_collected(owner: Any, assembly: CscAssembly, scratch: Path) -> None:
    """Tie a scratch directory's life to the table built on it.

    For callers of the table builders who supplied no scratch directory:
    the memmaps are released and the directory removed when ``owner`` (the
    AnnData) is garbage collected, or at interpreter exit, whichever comes
    first.
    """

    def _cleanup() -> None:
        assembly.release()
        remove_scratch(scratch)

    weakref.finalize(owner, _cleanup)
