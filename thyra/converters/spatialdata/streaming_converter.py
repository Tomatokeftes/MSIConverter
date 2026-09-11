# thyra/converters/spatialdata/streaming_converter.py

"""The MSI-to-SpatialData converter: two passes, every table built out of core.

- Pass 1 counts entries per m/z column and records which grid positions
  carry a spectrum; pass 2 scatters straight into memory-mapped CSC
  arrays (``csc_assembly.CscAssembly``, shared with the sibling tables).
- Each table is an AnnData over those memmaps, parsed by spatialdata's
  ``TableModel`` and written by spatialdata's own writer, so the matrix is
  never a scipy object in RAM and the on-disk layout is anndata's.
- One table per z plane (``handle_3d=False``) or one for the whole volume
  (``handle_3d=True``), the way the in-memory converters it replaced wrote
  them.
"""

import logging
import math
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from numpy.typing import NDArray
from spatialdata.models import Image2DModel, Image3DModel, TableModel
from spatialdata.transformations import Affine, Scale
from tqdm import tqdm

from ...errors import ConversionRefused
from ...resampling import ResamplingMethod
from .base_spatialdata_converter import BaseSpatialDataConverter, _kept_mz_range
from .csc_assembly import CscAssembly, index_dtype

logger = logging.getLogger(__name__)


class _TableUnit:
    """One table of the store in the making: a z plane, or the whole volume.

    Holds what the two passes accumulate for it -- the column counts and
    the scattered matrix (``assembly``), which grid positions carry a
    spectrum (``occupancy``) and the TIC raster -- and, once pass 1 is
    done, the row layout: ``kept_grid`` is the grid index of each table
    row and ``row_of_grid`` the table row of each grid position (-1 when
    dropped as empty).

    Args:
        key: The table's element key (``{id}_z{z}`` or ``{id}``).
        region_key: The shapes element the table annotates.
        plane: The z plane this table covers, or ``None`` for a volume.
        n_grid: Grid positions the table may hold rows for.
        n_cols: Length of the mass axis.
        tic_shape: ``(n_y, n_x)`` for a plane, ``(n_z, n_y, n_x)`` for a
            volume.
    """

    def __init__(
        self,
        key: str,
        region_key: str,
        plane: Optional[int],
        n_grid: int,
        n_cols: int,
        tic_shape: Tuple[int, ...],
    ) -> None:
        """Allocate the pass-1 accumulators; the matrix comes after pass 1."""
        self.key = key
        self.region_key = region_key
        self.plane = plane
        self.n_grid = int(n_grid)
        # Every m/z bin is a column, occupied or not: the summed table's
        # var is the mass axis, and an empty column is still a bin.
        self.assembly = CscAssembly(n_cols, n_rows=0, keep_empty_columns=True)
        self.occupancy = np.zeros(self.n_grid, dtype=bool)
        #: Positions the source gave more than one spectrum for. Their
        #: entries are summed into the one row the position gets, so both
        #: the row's TIC and pass 2's per-row check have to accumulate
        #: rather than compare a single spectrum (#241, #247).
        self.repeats = np.zeros(self.n_grid, dtype=bool)
        #: Pass 2's running total per repeated position, with the
        #: coordinate to name if it does not come out as pass 1's.
        self.observed: Dict[int, Tuple[float, Tuple[int, int, int]]] = {}
        self.tic = np.zeros(tic_shape, dtype=np.float64)
        self.kept_grid: Optional[NDArray[np.int64]] = None
        self.row_of_grid: Optional[NDArray[np.int64]] = None
        self.n_rows = 0

    def count(
        self, grid: int, mz_indices: NDArray[np.int_], values: NDArray[np.float64]
    ) -> None:
        """Pass 1: one spectrum at grid position ``grid``."""
        self.assembly.count(grid, mz_indices)
        if self.occupancy[grid]:
            # A pixel the source measured twice. One position is one row,
            # so the two spectra are summed into it -- what the COO route
            # got from ``coo.tocsc()`` before this engine replaced it --
            # and the assembly is told the repeats are coming.
            self.repeats[grid] = True
            self.assembly.merge_duplicates = True
        self.occupancy[grid] = True
        self.tic.reshape(-1)[grid] += values.sum()

    def finish_counting(self) -> None:
        """Decide which grid positions become rows, and in what order.

        Kept rows stay in grid order and keep their **grid index** as
        ``instance_id``: the index has gaps where the empty positions
        were, so a consumer can still recover the position from it. Only
        the row *offsets* are compacted, because the matrix has to be
        dense in its rows. Acquisitions are polygon-shaped and the grid
        is their bounding box, so the corners are the usual casualties
        (#88; on real ``pea.imzML`` 4,686 of 17,423 positions).
        """
        self.kept_grid = np.flatnonzero(self.occupancy).astype(np.int64)
        self.row_of_grid = np.full(self.n_grid, -1, dtype=np.int64)
        self.row_of_grid[self.kept_grid] = np.arange(
            self.kept_grid.size, dtype=np.int64
        )
        self.n_rows = int(self.kept_grid.size)
        self.assembly.n_rows = self.n_rows
        self.assembly.finish_counting()
        n_dropped = self.n_grid - self.n_rows
        if n_dropped:
            logger.info(
                "%s: dropping %d empty pixel rows from obs (%d non-empty pixels "
                "remain). These positions are inside the bounding box but "
                "outside the acquisition polygon.",
                self.key,
                n_dropped,
                self.n_rows,
            )
        self.observed = {}
        n_repeats = int(np.count_nonzero(self.repeats))
        if n_repeats:
            logger.warning(
                "%s: %d pixel position(s) carry more than one spectrum. Their "
                "spectra are summed into the one row the position has, which "
                "is what the COO route stored before this one replaced it; "
                "the TIC image and the row agree on the sum.",
                self.key,
                n_repeats,
            )

    def check_against_pass_one(
        self,
        grid: int,
        coords: Tuple[int, int, int],
        values: NDArray[np.float64],
    ) -> None:
        """Pass 2: refuse unless this position's total is the one pass 1 saw.

        A position the source gives one spectrum for is settled here. One
        it gives several for cannot be: pass 1 recorded the sum, so the
        running total is accumulated and
        :meth:`check_repeated_positions` compares it once the pass is
        over.

        Raises:
            ConversionRefused: When the two passes disagree.
        """
        total = float(np.sum(values))
        expected = float(self.tic.reshape(-1)[grid])
        if self.repeats[grid]:
            running = self.observed.get(grid, (0.0, coords))[0] + total
            self.observed[grid] = (running, coords)
            return
        if not _passes_agree(total, expected):
            raise ConversionRefused(_disagreement(self.key, coords, expected, total))

    def check_repeated_positions(self) -> None:
        """The same check for the positions whose spectra had to be summed.

        Raises:
            ConversionRefused: When the two passes disagree.
        """
        flat = self.tic.reshape(-1)
        for grid, (total, coords) in self.observed.items():
            expected = float(flat[grid])
            if not _passes_agree(total, expected):
                raise ConversionRefused(
                    _disagreement(self.key, coords, expected, total)
                )


#: How far pass 2's total for a pixel may sit from pass 1's before the
#: conversion is refused. The two passes run the same deterministic code
#: over the same spectrum, so the difference is normally exactly zero;
#: the tolerance is only there so a reader whose arithmetic is merely
#: re-associated between iterations is not refused for it. The divergence
#: this exists to catch was a factor of two.
PASS_AGREEMENT_RTOL = 1e-9


def _passes_agree(observed: float, expected: float) -> bool:
    """Whether pass 2's total for a pixel is pass 1's."""
    return math.isclose(observed, expected, rel_tol=PASS_AGREEMENT_RTOL, abs_tol=0.0)


def _disagreement(
    key: str, coords: Tuple[int, int, int], expected: float, observed: float
) -> str:
    """What to tell someone whose reader did not repeat itself."""
    x, y, z = coords
    return (
        f"{key}: the two passes over the source disagree at pixel "
        f"(x={x}, y={y}, z={z}). The pre-scan totalled {expected:.6g} for it "
        f"and the second pass totalled {observed:.6g}. Every conversion reads "
        "the source twice and the second read has to reproduce the first "
        "exactly; a reader whose iteration is not repeatable would otherwise "
        "write a store whose TIC image and average spectrum describe "
        "different data from its matrix."
    )


class StreamingSpatialDataConverter(BaseSpatialDataConverter):
    """Convert MSI data to SpatialData in two passes over the source.

    **Pass 1** resamples every spectrum onto the common mass axis, counts
    the entries each m/z column will hold and notes which grid positions
    carry a spectrum at all. **Pass 2** resamples every spectrum again --
    the resampling is deterministic -- and scatters its values straight
    to their final positions in memory-mapped CSC arrays. Each table is
    then an AnnData over those memmaps and goes through spatialdata's
    writer like any other element, so the matrix is never a scipy object
    in RAM and nothing here composes a Zarr layout by hand.

    Before v3.23 this was one of three converters. Two in-memory ones held
    the matrix as COO triples for one pass and converted it at the end;
    this one, the streaming route, hand-wrote the table's Zarr layout and
    wrote a single plane. Measured on the real files in ``test_data/``,
    warm, ``--no-optical``, before they were folded in (peak process RSS
    over the whole process tree, sampled at 50 ms):

    ===================  ==========  ===================  ==================
    dataset              spectra     one pass, in memory  two passes (this)
    ===================  ==========  ===================  ==================
    pea.imzML            12,737      12.2 s / 4.0 GB      15.7 s / 1.0 GB
    bellini.imzML        (36M nnz)   8.3 s / 1.1 GB       10.8 s / 0.6 GB
    TSF (33,800 px)      33,800      19.2 s / 5.1 GB      28.2 s / 1.3 GB
    TDF PASEF (713 px)   713         10.7 s / 0.42 GB     10.0 s / 0.40 GB
    ===================  ==========  ===================  ==================

    The second pass costs a quarter to a third of a small conversion and
    buys a peak memory that does not grow with the matrix; on a TDF the
    fused sibling passes (D5) already made the two routes equal. Design
    decision D11 records why the one-pass route went anyway.

    The route that preceded this one's second pass -- COO, count per row,
    CSR to a temporary Zarr, ``.tocsc()`` in RAM -- was measured against
    it on the mock reader before it was removed (D9): 7.4 s / 540 MB
    against 10.4 s / 655 MB at 16M non-zeros, 35.6 s / 1.1 GB against
    58.5 s / 1.9 GB at 64M. Note what those numbers do *not* say: peak
    RSS is not "~200 MB regardless of dataset size". Much of it is memmap
    pages, which count toward the working set while remaining evictable,
    and the ``var`` frame is O(bins) -- but "bounded" was never true and
    is not claimed here.
    """

    def __init__(
        self,
        *args,
        use_csc: Union[bool, Literal["auto"]] = "auto",
        **kwargs,
    ):
        """Initialize the converter.

        Args:
            *args: Arguments passed to BaseSpatialDataConverter
            use_csc: Kept for the callers that pinned the PCS route while a
                second one existed (Ousia's import wizard passes ``True``).
                ``True`` and ``"auto"`` both mean this route, which is the
                only one; ``False`` used to select the COO route and now
                raises, since there is nothing left for it to select.
            **kwargs: Keyword arguments passed to BaseSpatialDataConverter;
                ``handle_3d=True`` writes one table for the whole volume,
                the default writes one per z plane.

        Raises:
            ValueError: On ``use_csc=False``. There is one route left for it
                to select, and falling through to it as if it had been
                chosen is how a caller ends up with the opposite of what
                they asked for. (``sparse_format`` is refused by the base
                converter: CSC is the only layout, design decision D10.)

        Note:
            Intensity thresholding (filtering noise below a minimum value) is
            handled at the reader level via the `intensity_threshold` parameter
            passed to the reader constructor.
        """
        super().__init__(*args, **kwargs)

        if use_csc is False:
            raise ConversionRefused(
                "use_csc=False selected the streaming COO route, which has been "
                "removed: the PCS route was faster and lighter at every size "
                "measured. Pass use_csc=True or leave it out."
            )

        # Resolved once here rather than per spectrum: _process_spectrum is
        # the hottest call in both passes, and re-importing ResamplingMethod
        # plus re-checking the attribute there measured ~35 us per call.
        self._nn_route: bool = (
            self._resampling_config is not None
            and getattr(self, "_resampling_method", None)
            == ResamplingMethod.NEAREST_NEIGHBOR
        )

    def _suppress_reader_progress(self) -> None:
        """Suppress progress output from reader during the passes."""
        setattr(self.reader, "_quiet_mode", True)

    def _matrix_size_gb(self, units: List[_TableUnit]) -> float:
        """What the counted matrices come to, uncompressed, in GB.

        Not an estimate: the pre-scan has counted every entry by the time
        this is called, and the matrix is those entries -- eight bytes of
        value and four or eight of row index each. The store itself is
        smaller, because zarr compresses it, and a source with repeated
        coordinates ends with fewer entries than were counted because
        those are summed into one. Measured on
        ``TIMS-test-data/02_tiny_longramp_1465px``, converted with the
        defaults: 9,149,840 non-zeros, 0.10 GB here, 27 MB on disk. So
        the number is an upper bound, and it is on the right side to be
        one.

        It replaces an "Estimated output size" line that multiplied the
        *bounding box* by the bin count as though the matrix were dense
        and logged 40.3 GB for that dataset -- 2.5 GB for a 7.6 MB store,
        240.7 GB for a 329 MB one (issue #254). Since the route stopped
        being chosen by size (design decision D11) that number selected
        nothing, so the only thing left for it to do was talk somebody out
        of a conversion they had room for.

        Args:
            units: The tables, after their pre-scan.

        Returns:
            Size in gigabytes.
        """
        entry_bytes = 0
        n_entries = 0
        for unit in units:
            nnz = int(unit.assembly.n_nonzeros)
            n_entries += nnz
            entry_bytes += nnz * (8 + np.dtype(index_dtype(nnz, unit.n_rows)).itemsize)
        size_gb = entry_bytes / (1024**3)
        logger.info(
            "Matrix: %s non-zero entries over %s rows, %.2f GB uncompressed",
            f"{n_entries:,}",
            f"{sum(unit.n_rows for unit in units):,}",
            size_gb,
        )
        return size_gb

    # ------------------------------------------------------------------
    # The passes
    # ------------------------------------------------------------------

    def _process_spectrum(
        self, mzs: np.ndarray, intensities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample one spectrum onto the axis: ``(bin indices, values)``.

        The indices are unique and ascending on every path -- what
        :class:`CscAssembly` requires of a row -- and carry no zeros.
        Deterministic, which is what lets pass 2 reproduce pass 1.

        Note: Intensity thresholding is handled at the reader level before data
        reaches this method. This method only handles resampling and zero filtering.

        Args:
            mzs: Mass values (already filtered by reader if threshold is set)
            intensities: Intensity values (already filtered by reader if threshold is set)

        Returns:
            Tuple of (mz_indices, resampled_intensities) with zeros filtered out
        """
        # Before either resampling method, so both are handed the same
        # spectrum and cannot disagree about it (issue #248).
        mzs, intensities = self._drop_unusable_intensities(mzs, intensities)

        if not self._resampling_config:
            # No resampling - map m/z values to indices directly. Entries
            # that share a bin (a repeated m/z) are summed so the CSC never
            # carries two values at one (row, col), and zeros are dropped
            # so a dense continuous-mode spectrum does not fill the matrix
            # with explicit zeros -- which the in-memory converters never
            # stored, and the hand-written layout did.
            mz_indices, values = self._coalesce_duplicate_bins(
                self._map_mass_to_indices(mzs), intensities
            )
            keep = values != 0
            if not bool(keep.all()):
                mz_indices, values = mz_indices[keep], values[keep]
            return mz_indices, values

        # Optimized nearest-neighbor path; the route is resolved once in
        # __init__ because this runs once per spectrum per pass.
        if self._nn_route:
            return self._nearest_neighbor_resample(mzs, intensities)

        # TIC-preserving, evaluated only where the interpolant can be
        # non-zero and returned already zero-filtered. On a zero-suppressed
        # profile source the dense form of this call -- interpolate onto
        # every bin, then mask -- was 35x the cost of reading the file.
        return self._tic_preserving_resample_sparse(mzs, intensities)

    def _plan_tables(self) -> List[_TableUnit]:
        """The tables this conversion writes: one per plane, or one volume.

        A plane's table indexes its rows by ``y * n_x + x`` and is named
        ``{id}_z{z}``; the volume's by ``z * n_x * n_y + y * n_x + x`` and
        is named ``{id}``, with ``z`` and ``spatial_z`` in ``obs``. Those
        are the layouts the per-slice and volume converters wrote before
        they were folded in here, kept so a store reads back the same.
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")
        n_x, n_y, n_z = self._dimensions
        n_cols = len(self._common_mass_axis)
        if self.handle_3d:
            return [
                _TableUnit(
                    self.dataset_id,
                    f"{self.dataset_id}_pixels",
                    None,
                    n_x * n_y * n_z,
                    n_cols,
                    (n_z, n_y, n_x),
                )
            ]
        return [
            _TableUnit(
                f"{self.dataset_id}_z{z}",
                f"{self.dataset_id}_z{z}_pixels",
                z,
                n_x * n_y,
                n_cols,
                (n_y, n_x),
            )
            for z in range(n_z)
        ]

    def _create_data_structures(self) -> Dict[str, Any]:
        """Plan the tables and the accumulators the two passes fill.

        Returns:
            The mapping the base workflow threads through
            :meth:`_process_spectra`, :meth:`_finalize_data` and
            :meth:`_save_output`: the table units, the sibling sinks (or
            ``None``), the dataset-wide intensity accumulators, and the
            ``tables`` / ``shapes`` / ``images`` the finalize step fills.
        """
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        units = self._plan_tables()
        n_cols = len(self._common_mass_axis)
        logger.info(
            "Streaming CSC: %s grid positions x %s m/z bins in %d table(s)",
            f"{sum(unit.n_grid for unit in units):,}",
            f"{n_cols:,}",
            len(units),
        )

        # The sibling tables' sinks, fed from the two passes below when the
        # reader hands its frames over as records (design decision D5): one
        # raw read per frame per pass for every table. Only for a single
        # table -- the sinks take one row space -- which every source with
        # frame records (a Bruker TDF) has; a multi-plane source scans on
        # its own per plane, as it always did.
        passes = self._fused_sibling_passes(units[0].key) if len(units) == 1 else None

        data_structures: Dict[str, Any] = {
            "mode": "3d_volume" if self.handle_3d else "2d_slices",
            "units": units,
            "passes": passes,
            "tables": {},
            "shapes": {},
            "images": {},
            "var_df": self._create_mass_dataframe(),
            "pixel_count": 0,
            "avg_spectrum_per_region": None,
        }

        # Per-region accumulators for multi-region datasets. Unlike the
        # per-table average these stay dataset-wide, because a region is:
        # ``get_region_map`` is keyed on ``(x, y)`` with no z, so one
        # region is one in-plane footprint sampled on every plane. See
        # _finalize_table for what that means for a multi-plane store.
        if self._region_map is not None:
            unique_regions = sorted(set(self._region_map.values()))
            data_structures["region_total_intensity"] = {
                r: np.zeros(n_cols, dtype=np.float64) for r in unique_regions
            }
            data_structures["region_row_count"] = {r: 0 for r in unique_regions}

        return data_structures

    def _locate(
        self, units: List[_TableUnit], x: int, y: int, z: int
    ) -> Tuple[Optional[_TableUnit], int]:
        """The table a coordinate belongs to and its grid index there.

        ``(None, -1)`` for a coordinate outside the declared grid. Left
        unchecked, a negative coordinate is a legal negative numpy index
        and wraps silently onto an unrelated pixel; a ``z`` past the
        planes would pick a table that does not exist.
        """
        n_x, n_y, n_z = self._dimensions
        if not (0 <= x < n_x and 0 <= y < n_y and 0 <= z < n_z):
            return None, -1
        if self.handle_3d:
            return units[0], z * n_x * n_y + y * n_x + x
        return units[z], y * n_x + x

    def _iter_pass_spectra(
        self, passes: Any, phase: str
    ) -> Generator[Tuple[Tuple[int, int, int], Any, Any, Any], None, None]:
        """The summed spectra of one pass, with the frame record they came from.

        Yields ``(coords, mzs, intensities, frame)``: ``frame`` is the
        record the sinks are fed from, or ``None`` when the pass reads
        the summed spectra directly (no sinks, or a reader without
        records). A frame whose summed spectrum is empty is not yielded,
        exactly as ``iter_spectra`` never yields it, but the sinks of
        ``phase`` (``"count"`` or ``"scatter"``) still see it with no row,
        which is what their own pass would have shown them.
        """
        if passes is None or passes.empty:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                yield coords, mzs, intensities, None
            return
        feed = getattr(passes, phase)
        for frame in self.reader.iter_frame_scans(batch_size=self._buffer_size):
            spectrum = frame.spectrum()
            if spectrum is None:
                feed(frame, None)
                continue
            yield frame.coords, spectrum[0], spectrum[1], frame

    def _process_spectra(self, data_structures: Dict[str, Any]) -> None:
        """Run both passes: count, size the arrays, scatter.

        Overrides the base's single pass. The pre-scan is light -- the
        same resampling as the main pass, no disk I/O -- and the 2x CPU
        it costs is far less than the disk I/O of caching every spectrum
        between the passes, which is what it replaced.
        """
        units: List[_TableUnit] = data_structures["units"]
        passes = data_structures["passes"]

        logger.info("Step 1/3: Pre-scan (counting entries per column)...")
        self._count_pass(data_structures)

        logger.info("Step 2/3: Allocating memory-mapped CSC arrays...")
        for unit in units:
            unit.finish_counting()

        # The rows that are written, not the spectra that were read: a
        # position measured twice is one row, and an empty or out-of-grid
        # spectrum is none. The root attrs report this as
        # ``non_empty_pixels``, where it used to be the reader's spectrum
        # count and so disagreed with the table it described (#241).
        self._non_empty_pixel_count = sum(unit.n_rows for unit in units)

        # Before anything is allocated and before the source is read a
        # second time: a conversion with no row has nothing left to do
        # and no store to write (#242).
        self._refuse_an_empty_conversion(data_structures)
        self._finish_region_averages(data_structures)

        for unit in units:
            unit.assembly.allocate(
                self._register_table_scratch("summed", unit.assembly)
            )

        self._matrix_size_gb(units)

        if passes is not None:
            passes.finish_counting(units[0].n_rows, self._register_table_scratch)
            if units[0].repeats.any():
                # A position measured twice is scattered twice into every
                # table fed from these passes, not just the summed one.
                passes.merge_duplicate_rows()

        logger.info("Step 3/3: Processing spectra and scattering to CSC...")
        self._scatter_pass(data_structures)
        if passes is not None:
            passes.finish_scattering()
            self._take_fused_results(passes)

    def _kept_in_plane_xy(
        self, unit: _TableUnit
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """The in-plane ``(x, y)`` of one table's rows, in row order."""
        if unit.kept_grid is None or self._dimensions is None:
            raise RuntimeError("The row layout is not decided yet")
        n_x, n_y, _ = self._dimensions
        kept = unit.kept_grid
        in_plane = kept % (n_x * n_y) if unit.plane is None else kept
        return in_plane % n_x, in_plane // n_x

    def _refuse_an_empty_conversion(self, data_structures: Dict[str, Any]) -> None:
        """Refuse a conversion in which no position carries a spectrum.

        Such a run used to report success: ``convert_msi`` returned
        ``True``, the CLI exited 0, and a store was written with no table,
        no image and no shapes -- while the log said "No non-zero entries
        found!" and, per plane, "no position carries a spectrum". A
        calling script saw a finished conversion and a zarr it could not
        read a spectrum out of (issue #242).

        Checked here, on the row count, rather than on the empty
        ``tables`` mapping in ``_save_output``: the pre-scan has just
        settled every table's rows, a full second read of the source is
        still ahead, and the same count covers every route into the empty
        store -- an all-zero source, every spectrum empty, every peak
        outside a narrowed resampling range, every coordinate off the
        grid.

        A **multi-plane** source with some empty planes is not refused.
        Those planes are dropped with a warning and the rest are written,
        which is the #88 behaviour: an acquisition is polygon-shaped and
        its bounding box has empty corners. Only a conversion with no row
        anywhere is refused.

        Raises:
            ConversionRefused: When no table would have a single row.
        """
        if self._non_empty_pixel_count > 0:
            return

        raise ConversionRefused(
            f"{self.dataset_id}: no pixel carries a spectrum, so there is no "
            f"table to write -- {self._why_nothing_survived(data_structures)}. "
            f"Nothing was written to {self.output_path}."
        )

    def _why_nothing_survived(self, data_structures: Dict[str, Any]) -> str:
        """Which of the routes into an empty store this conversion took.

        "Every spectrum was empty" and "every peak fell outside
        250-1200 m/z" send a user to different places, so the refusal has
        to tell them apart rather than name the possibilities. Each test
        is on a total rather than on a counter being non-zero: a
        resampled axis lands a few tenths of a mDa inside the source's
        own range, so two of a spectrum's twenty peaks falling off its
        ends is routine and says nothing about why the store is empty.

        Both peak counters are read after pass 1 and before pass 2, so
        they hold that one pass's totals -- they count resample calls,
        and an ordinary conversion resamples every spectrum twice.
        """
        pixel_count = int(data_structures.get("pixel_count", 0))
        off_grid = int(data_structures.get("out_of_grid_spectra", 0))
        peaks_in = int(data_structures.get("input_peaks", 0))

        if pixel_count == 0:
            return "the reader yielded no spectra at all"
        if off_grid == pixel_count:
            n_x, n_y, n_z = self._dimensions
            return (
                f"all {pixel_count} spectra sat outside the declared "
                f"{n_x}x{n_y}x{n_z} grid"
            )
        if peaks_in == 0:
            return "every spectrum the reader yielded was empty"
        if self._unusable_intensities >= peaks_in:
            return (
                "every intensity was dropped as not a measurement "
                "(non-finite, or negative)"
            )
        usable = peaks_in - self._unusable_intensities
        if self._out_of_range_peaks >= usable and self._common_mass_axis is not None:
            lo_mz, hi_mz = _kept_mz_range(self._common_mass_axis, self._axis_range)
            return (
                "every peak fell outside the target mass range "
                f"[{lo_mz:.4f}, {hi_mz:.4f}] m/z -- widen the "
                "resampling range (--resample-min-mz / --resample-max-mz) to "
                "keep them"
            )
        return "every intensity in the source is zero"

    def _finish_region_averages(self, data_structures: Dict[str, Any]) -> None:
        """Divide each region's summed intensity by the rows it covers.

        The numerator is every spectrum that got a row in the region, so
        the denominator has to be rows too, exactly as for the per-table
        average (#243): counting the spectra fed in instead made a region
        holding a position the source measured twice come out low by that
        position's share.

        Regions stay dataset-wide, spanning z. That is what a region *is*
        here -- ``get_region_map`` is keyed on ``(x, y)`` and has no z
        component, so one region is one in-plane footprint sampled on
        every plane, and splitting it per plane would answer a question
        the source never asked.
        """
        region_total = data_structures.get("region_total_intensity")
        if region_total is None:
            return

        rows: Dict[int, int] = data_structures["region_row_count"]
        for unit in data_structures["units"]:
            x_idx, y_idx = self._kept_in_plane_xy(unit)
            numbers, counts = np.unique(
                self.build_region_numbers(x_idx, y_idx), return_counts=True
            )
            for region, count in zip(numbers.tolist(), counts.tolist()):
                if region in rows:
                    rows[region] += int(count)

        data_structures["avg_spectrum_per_region"] = {
            str(region): total / max(rows.get(region, 0), 1)
            for region, total in region_total.items()
        }

    def _count_pass(self, data_structures: Dict[str, Any]) -> None:
        """Pass 1: count entries per column, TIC, occupancy, region totals.

        The mean spectrum is *not* accumulated here. It is each table's
        own column sums divided by its own rows, and both are read off the
        finished matrix in :meth:`_finalize_table` (#243).

        Also feeds the sibling tables' pass-1 sinks, when ``passes`` is
        set, from the same frame read (see ``fused_passes.py``).
        """
        units: List[_TableUnit] = data_structures["units"]
        passes = data_structures["passes"]
        region_total = data_structures.get("region_total_intensity")

        pixel_count = 0
        n_out_of_bounds = 0
        # Peaks handed in, before anything is dropped. The empty-store
        # refusal tells its causes apart by comparing the dropped totals
        # with this one (see _why_nothing_survived).
        n_input_peaks = 0
        self._suppress_reader_progress()

        with tqdm(
            total=self._get_total_spectra_count(),
            desc="Pre-scan" if passes is None else "Pre-scan + sibling tables",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities, frame in self._iter_pass_spectra(
                passes, "count"
            ):
                x, y, z = coords
                n_input_peaks += int(np.size(mzs))
                mz_indices, values = self._process_spectrum(mzs, intensities)
                nnz = int(mz_indices.size)
                unit, grid = self._locate(units, x, y, z)

                if frame is not None:
                    # The row the sinks count under is the grid position,
                    # which is the table row's own order: rows are numbered
                    # in grid order once the empty positions are dropped.
                    # A spectrum that gets no row (empty, or off the grid)
                    # is a pixel the siblings skip, as their own pass
                    # skips one that is not in obs.
                    gets_row = nnz > 0 and unit is not None
                    passes.count(frame, grid if gets_row else None)

                if nnz > 0:
                    # Column counts, TIC and occupancy all describe a
                    # spectrum that is going to get a row, so all three sit
                    # behind the bounds check: a reader yielding a
                    # coordinate outside the declared dimensions would
                    # otherwise wrap round and land on an unrelated pixel,
                    # and counting it would reserve slots the scatter pass
                    # never writes -- explicit zeros, out of order, which
                    # scipy reports as a non-canonical matrix.
                    if unit is not None:
                        unit.count(grid, mz_indices, values)

                        # Behind the same check, and for the same reason
                        # the per-table average is taken over rows: the
                        # numerator has to be the current the store
                        # actually holds. A spectrum off the grid is
                        # written nowhere, so it belongs in no mean
                        # (#243). Indices are unique within a spectrum,
                        # so the fancy-indexed add is exact.
                        if region_total is not None:
                            region = self._region_map.get((x, y), -1)
                            if region in region_total:
                                region_total[region][mz_indices] += values
                    else:
                        n_out_of_bounds += 1

                pixel_count += 1
                pbar.update(1)

        total_nnz = sum(unit.assembly.n_nonzeros for unit in units)
        if total_nnz == 0:
            logger.warning("No non-zero entries found!")
        logger.info(
            "  Pre-scan complete: %s entries across %s columns",
            f"{total_nnz:,}",
            f"{len(self._common_mass_axis):,}",
        )
        if n_out_of_bounds:
            n_x, n_y, n_z = self._dimensions
            logger.warning(
                "%d spectra sat outside the declared %dx%dx%d grid and were "
                "skipped: their coordinates fall beyond the raster the source "
                "declares, so there is no pixel to write them to.",
                n_out_of_bounds,
                n_x,
                n_y,
                n_z,
            )

        # ``pixel_count`` is every spectrum the reader yielded. It is not
        # the table's row count -- an empty spectrum gets no row, an
        # out-of-grid one gets no row, and a position measured twice gets
        # one -- so ``non_empty_pixels`` is set from the rows in
        # _process_spectra rather than from here (#241), and no average is
        # taken over it any more (#243). All it sizes now is pass 2's
        # progress bar, which counts the same spectra pass 1 walked.
        data_structures["pixel_count"] = pixel_count
        data_structures["out_of_grid_spectra"] = n_out_of_bounds
        data_structures["input_peaks"] = n_input_peaks

    def _scatter_pass(self, data_structures: Dict[str, Any]) -> None:
        """Pass 2: resample every spectrum again and scatter it into its table.

        Same resampling as the pre-scan, so the two passes agree on which
        rows exist and how many entries each column holds; the assembly
        checks that agreement before it hands the matrix out. With
        ``passes`` it also scatters the sibling tables from the same
        frame read.

        The assembly's check is on *counts*, and counts are not enough. A
        reader whose second iteration hands back the same number of
        entries with different values wrote a store in which ``X`` came
        from pass 2 while the TIC image and ``average_spectrum`` came from
        pass 1 -- measured, on a probe reader that scaled its values:
        ``TIC sum=147066`` against ``X.sum=294132``, and nothing said so
        (issue #247). Pass 1 already recorded each position's total in
        ``unit.tic``, so pass 2 compares the row it is about to scatter
        with it and refuses on the first disagreement. That also catches
        two pixels exchanging spectra, which no count can.
        """
        units: List[_TableUnit] = data_structures["units"]
        passes = data_structures["passes"]

        # Reset reader for second pass. For real readers (ImzML, Bruker),
        # iter_spectra() is a generator factory that creates a fresh
        # iterator each time; a mock reader with random data reseeds.
        if hasattr(self.reader, "reset"):
            self.reader.reset()
        self._suppress_reader_progress()

        with tqdm(
            total=data_structures["pixel_count"],
            desc="Scatter to CSC" if passes is None else "Scatter to CSC + siblings",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities, frame in self._iter_pass_spectra(
                passes, "scatter"
            ):
                x, y, z = coords
                mz_indices, values = self._process_spectrum(mzs, intensities)
                unit, grid = self._locate(units, x, y, z)

                # Table row for this grid position -- not the grid index
                # itself, because the empty positions are dropped and the
                # rows compacted. row_of_grid comes from the same pre-scan
                # that sized the columns, so the two agree by construction;
                # a position with no row (empty, or a coordinate outside
                # the grid, both skipped there) comes back as -1.
                row = -1
                if mz_indices.size > 0 and unit is not None:
                    unit.check_against_pass_one(grid, coords, values)
                    row = int(unit.row_of_grid[grid])

                if frame is not None:
                    passes.scatter(frame, row if row >= 0 else None)
                if row >= 0:
                    unit.assembly.scatter(row, mz_indices, values)

                pbar.update(1)

        for unit in units:
            unit.check_repeated_positions()

        logger.info("  Scatter complete")

    # ------------------------------------------------------------------
    # From memmaps to elements
    # ------------------------------------------------------------------

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Build every table, its shapes and its TIC image from the memmaps.

        Args:
            data_structures: What the passes filled.
        """
        for unit in data_structures["units"]:
            if unit.n_rows == 0:
                logger.warning(
                    "%s: no position carries a spectrum; no table is written " "for it",
                    unit.key,
                )
                continue
            self._finalize_table(data_structures, unit)

        # Add optical images if available
        self._add_optical_images(data_structures)

    def _finalize_table(
        self, data_structures: Dict[str, Any], unit: _TableUnit
    ) -> None:
        """One table over its memmaps, parsed, with its shapes and TIC image."""
        # The matrix is canonical as scattered when the reader handed its
        # pixels over in raster order (rows are numbered in raster order,
        # so every column's row indices are ascending); the assembly
        # sorts each column in place otherwise, and checks that pass 2
        # wrote every slot pass 1 counted. The constructor takes the
        # memmaps without copying them.
        matrix = unit.assembly.matrix()
        logger.info("%s: %s non-zero entries (CSC)", unit.key, f"{matrix.nnz:,}")

        adata = AnnData(
            X=matrix,
            obs=self._table_obs(unit),
            var=data_structures["var_df"].copy(),
        )

        # This table's own mean spectrum, taken from this table's own
        # matrix: the column sums over the rows that were written. Until
        # #243 a multi-slice source converted as 2D put one dataset-wide
        # vector in every plane's table, and the ratio to the plane's real
        # mean was measured at 1.96 / 0.996 / 0.67 across three planes of
        # a source scaled by z. Read off the matrix rather than
        # accumulated alongside it so it *is* ``X.mean(axis=0)``, which is
        # what docs/output-format.md promises and what Ousia reads, rather
        # than a second number that has to be kept in step with it.
        adata.uns["average_spectrum"] = np.asarray(
            matrix.sum(axis=0), dtype=np.float64
        ).ravel() / max(unit.n_rows, 1)
        # The per-region means stay dataset-wide, because a region has no
        # z: see _finish_region_averages. On a multi-plane store they
        # therefore describe a wider population than the key above, which
        # docs/output-format.md says out loud.
        per_region = data_structures.get("avg_spectrum_per_region")
        if per_region is not None:
            adata.uns["average_spectrum_per_region"] = per_region

        # Decide on the sibling tables first so uns can name them, then
        # run the raw mobility pass once for the heatmap and the grid's
        # discovery together, before uns is built -- unless both were fed
        # from the summed table's own passes already (see
        # _fused_sibling_passes, which also planned the siblings).
        if not self._siblings_planned:
            self._mobility_table_key = self._plan_mobility_table(unit.key)
            self._msms_table_key = self._plan_msms_table(unit.key)
        self._prepare_sibling_scans(adata.obs, z_value=unit.plane)

        # Add MSI metadata to .uns
        self._add_metadata_to_uns(adata)

        # Make sure instance_key is a string column
        adata.obs["instance_key"] = adata.obs.index.astype(str)

        table = TableModel.parse(
            adata,
            region=unit.region_key,
            region_key="region",
            instance_key="instance_key",
        )

        data_structures["tables"][unit.key] = table
        data_structures["shapes"][unit.region_key] = self._create_pixel_shapes(adata)
        self._attach_sibling_tables(
            data_structures, unit.key, unit.region_key, adata.obs, z_value=unit.plane
        )
        data_structures["images"][f"{unit.key}_tic"] = self._tic_image(unit)

    def _table_obs(self, unit: _TableUnit) -> pd.DataFrame:
        """``obs`` for the kept rows of one table.

        The positions are recovered from the grid indices. A plane's table
        carries ``y``, ``x`` and the in-plane positions; the volume's adds
        ``z`` and ``spatial_z``, the latter from the z spacing rather than
        the in-plane pitch so the table lands in the same micrometre frame
        as the TIC volume's ``Scale``. Column order and dtypes are the ones
        the two in-memory converters wrote, so a store reads back the same.
        """
        if unit.kept_grid is None or self._dimensions is None:
            raise RuntimeError("The row layout is not decided yet")
        kept = unit.kept_grid
        n_x, n_y, n_z = self._dimensions

        if unit.plane is None:
            z_idx = kept // (n_x * n_y)
            remainder = kept % (n_x * n_y)
            y_idx = remainder // n_x
            x_idx = remainder % n_x
            obs = pd.DataFrame(
                {
                    "x": x_idx,
                    "y": y_idx,
                    "z": z_idx if n_z > 1 else np.zeros(kept.size, dtype=np.int64),
                    "instance_id": kept.astype(str),
                    "region": pd.Categorical(np.full(kept.size, unit.region_key)),
                    "spatial_x": x_idx * self.pixel_size_um,
                    "spatial_y": y_idx * self.pixel_size_y_um,
                    "spatial_z": (
                        z_idx * self.z_spacing_um
                        if n_z > 1
                        else np.zeros(kept.size, dtype=np.float64)
                    ),
                }
            )
        else:
            y_idx = (kept // n_x).astype(np.int32)
            x_idx = (kept % n_x).astype(np.int32)
            obs = pd.DataFrame(
                {
                    "y": y_idx,
                    "x": x_idx,
                    "instance_id": kept.astype(str),
                    "region": pd.Categorical(np.full(kept.size, unit.region_key)),
                    "spatial_x": x_idx * self.pixel_size_um,
                    "spatial_y": y_idx * self.pixel_size_y_um,
                }
            )
        obs.set_index("instance_id", inplace=True)
        # Always add per-pixel region numbers for a consistent schema.
        obs["region_number"] = self.build_region_numbers(x_idx, y_idx)
        return obs

    def _tic_image(self, unit: _TableUnit) -> Any:
        """The TIC raster of one table as a SpatialData image element.

        Full-grid, and deliberately not subset with the rows: it is a
        dense image of the bounding box, not a per-pixel table. The image
        array is intrinsically in raster indices; its transformation to
        ``"global"`` expresses the conversion into the chosen global
        frame -- optical-image pixels when FlexImaging alignment is
        applied, physical micrometres otherwise, so that ``"global"``
        agrees with the pixel-polygon shapes.
        """
        n_z = self._dimensions[2]
        if unit.plane is None and n_z > 1:
            # A volume, on the axes Image3DModel declares. z gets its own
            # spacing (see BaseMSIConverter._resolve_z_spacing). Scale
            # pairs values with axis *names*, so ("x", "y", "z") against
            # a (c, z, y, x) image is deliberate.
            transform: Any = Scale(
                [self.pixel_size_um, self.pixel_size_y_um, self.z_spacing_um],
                axes=("x", "y", "z"),
            )
            return Image3DModel.parse(
                xr.DataArray(unit.tic[np.newaxis, ...], dims=("c", "z", "y", "x")),
                transformations={self.dataset_id: transform, "global": transform},
            )

        plane = unit.tic if unit.plane is not None else unit.tic[0]
        # Gate the alignment-based affine on apply_optical_alignment. When
        # the caller opts out (e.g. Ousia's wizard), MSI lands in pure
        # micrometer coordinates so downstream registration is the
        # canonical alignment step.
        if self._apply_optical_alignment and self._tic_to_image_matrix is not None:
            transform = Affine(
                self._tic_to_image_matrix,
                input_axes=("x", "y"),
                output_axes=("x", "y"),
            )
        else:
            transform = Scale(
                [self.pixel_size_um, self.pixel_size_y_um], axes=("x", "y")
            )
        return Image2DModel.parse(
            xr.DataArray(plane[np.newaxis, ...], dims=("c", "y", "x")),
            transformations={self.dataset_id: transform, "global": transform},
        )
