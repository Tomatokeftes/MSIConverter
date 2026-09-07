"""Out-of-core assembly of a pixels x features CSC matrix, in two passes.

The sibling tables -- the mobility-resolved table and the demultiplexed
MS/MS table -- are each a sparse matrix whose columns are the *occupied*
cells of a key space known before anything is read: ``(m/z bin, mobility
channel)`` for the grid, ``(precursor, m/z bin)`` for the demultiplexer,
the source's own feature list for a shared mobility axis. A pixel's
entries can be enumerated from the source in one pass and again in a
second, identically, so nothing ever needs to be held for the whole
image. This module is that shape, factored out of the two tables so both
bound their memory the same way the summed table already does on the
streaming route (``StreamingSpatialDataConverter._convert_to_csc_no_cache``:
pre-scan, count, scatter into a memmap).

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
columns at a time (:func:`sort_csc_columns`, which the streaming route's
summed table shares), so the stored matrix is canonical either way.

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
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

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

_COUNT_DTYPE = np.uint32


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
    columns over a bounded slice of the arrays at a time. The assembly
    here and the streaming route's summed table
    (``StreamingSpatialDataConverter._convert_to_csc_no_cache``) both sort
    their memmaps with it.

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
        """
        if self.indptr is None or self._write_pos is None:
            raise RuntimeError("matrix() needs allocate() first")
        if self._indices is None or self._data is None:
            raise RuntimeError("matrix() after release()")
        if not np.array_equal(self._write_pos, self.indptr[1:]):
            short = int(np.count_nonzero(self._write_pos != self.indptr[1:]))
            raise RuntimeError(
                f"{short} columns were counted for more entries than were "
                "scattered; the two passes disagree on which rows exist"
            )
        if not self.rows_in_order:
            self._sort_columns()
        self._indices.flush()
        self._data.flush()
        n_cols = int(self.indptr.size - 1)
        return sparse.csc_matrix(
            (self._data, self._indices, self.indptr), shape=(self.n_rows, n_cols)
        )

    def _sort_columns(self) -> None:
        """Put each column's entries in ascending row order, chunk by chunk.

        Only needed when the source handed its rows out of order; see
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
