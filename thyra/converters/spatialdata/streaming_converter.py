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
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from ...errors import ConversionRefused
from ...resampling import ResamplingMethod
from .base_spatialdata_converter import SPATIALDATA_AVAILABLE, BaseSpatialDataConverter
from .csc_assembly import CscAssembly

if SPATIALDATA_AVAILABLE:
    import xarray as xr
    from anndata import AnnData
    from spatialdata.models import Image2DModel, Image3DModel, TableModel
    from spatialdata.transformations import Affine, Scale

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
        self.tic = np.zeros(tic_shape, dtype=np.float64)
        self.kept_grid: Optional[NDArray[np.int64]] = None
        self.row_of_grid: Optional[NDArray[np.int64]] = None
        self.n_rows = 0

    def count(
        self, grid: int, mz_indices: NDArray[np.int_], values: NDArray[np.float64]
    ) -> None:
        """Pass 1: one spectrum at grid position ``grid``."""
        self.assembly.count(grid, mz_indices)
        self.occupancy[grid] = True
        self.tic.reshape(-1)[grid] = values.sum()

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

    def _estimate_output_size_gb(self) -> float:
        """Estimate the output dataset size in GB, for the log line.

        Dense size, ``n_pixels * n_mz_bins * 4`` bytes (float32).

        **Diagnostic only.** Nothing routes on this: there is one route. It
        is kept because the number is genuinely useful in a support log and
        because getting it right was not free: the fallback below is wrong
        in both directions (see the comment in the body), and deleting the
        function would delete that finding along with the tests that pin
        it.

        Returns:
            Estimated size in gigabytes
        """
        metadata = self.reader.get_essential_metadata()

        # Get dimensions
        n_x, n_y, n_z = metadata.dimensions
        n_pixels = n_x * n_y * n_z

        # Prefer the axis that was actually built. ``convert()`` runs
        # ``_initialize_conversion()`` -- and so ``_setup_mass_axis()`` --
        # before this is reached, so the real bin count is known and there
        # is nothing to estimate. That matters most on the raw-axis path,
        # where the fallback below assumes a 10 mDa spacing that the data
        # need not have: a continuous file carrying 4,000 points over
        # 250-1200 m/z was scored as though it had 95,000, and a processed
        # file whose spectra share no m/z values was scored far too low.
        # While this drove the routing (issue #87) that mis-sent the
        # largest datasets to the method that holds the most in memory; it
        # now only makes the logged number honest.
        if self._common_mass_axis is not None:
            n_mz_bins = len(self._common_mass_axis)
        elif self._resampling_config:
            # Same resolution path the axis builder will use, so the
            # reported bin count is the real one.
            _, _, _, n_mz_bins = self._resolve_resampling_plan()
        else:
            min_mass, max_mass = metadata.mass_range
            n_mz_bins = int((max_mass - min_mass) / 0.01)

        # Dense matrix size in bytes (float32 = 4 bytes)
        dense_bytes = n_pixels * n_mz_bins * 4

        # Convert to GB
        size_gb = dense_bytes / (1024**3)

        logger.info(
            f"Estimated output size: {size_gb:.1f} GB "
            f"({n_pixels:,} pixels x {n_mz_bins:,} m/z bins)"
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

        # Diagnostic only -- nothing below depends on it. After
        # _initialize_conversion() so it reports the built axis.
        self._estimate_output_size_gb()

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
            "total_intensity": np.zeros(n_cols, dtype=np.float64),
            "pixel_count": 0,
            "avg_spectrum": None,
            "avg_spectrum_per_region": None,
        }

        # Per-region accumulators for multi-region datasets
        if self._region_map is not None:
            unique_regions = sorted(set(self._region_map.values()))
            data_structures["region_total_intensity"] = {
                r: np.zeros(n_cols, dtype=np.float64) for r in unique_regions
            }
            data_structures["region_pixel_count"] = {r: 0 for r in unique_regions}

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
            unit.assembly.allocate(
                self._register_table_scratch("summed", unit.assembly)
            )
        if passes is not None:
            passes.finish_counting(units[0].n_rows, self._register_table_scratch)

        logger.info("Step 3/3: Processing spectra and scattering to CSC...")
        self._scatter_pass(data_structures)
        if passes is not None:
            passes.finish_scattering()
            self._take_fused_results(passes)

    def _count_pass(self, data_structures: Dict[str, Any]) -> None:
        """Pass 1: count entries per column, TIC, occupancy, average spectrum.

        Also feeds the sibling tables' pass-1 sinks, when ``passes`` is
        set, from the same frame read (see ``fused_passes.py``).
        """
        units: List[_TableUnit] = data_structures["units"]
        passes = data_structures["passes"]
        total_intensity: NDArray[np.float64] = data_structures["total_intensity"]
        region_total = data_structures.get("region_total_intensity")
        region_count = data_structures.get("region_pixel_count")

        pixel_count = 0
        n_out_of_bounds = 0
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
                    # The average is over every spectrum read rather than
                    # every spectrum stored, so it sits outside the bounds
                    # check. Indices are unique within a spectrum, so the
                    # fancy-indexed add is exact.
                    total_intensity[mz_indices] += values

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
                    else:
                        n_out_of_bounds += 1

                    if region_total is not None:
                        region = self._region_map.get((x, y), -1)
                        if region in region_total:
                            region_total[region][mz_indices] += values
                            region_count[region] += 1

                pixel_count += 1
                pbar.update(1)

        total_nnz = sum(unit.assembly.n_nonzeros for unit in units)
        if total_nnz == 0:
            logger.warning("No non-zero entries found!")
        logger.info(
            "  Pre-scan complete: %s entries across %s columns",
            f"{total_nnz:,}",
            f"{total_intensity.size:,}",
        )
        if n_out_of_bounds:
            n_x, n_y, n_z = self._dimensions
            logger.warning(
                "%d spectra sat outside the declared %dx%dx%d grid and were "
                "skipped. Previously they were written to a wrapped-round "
                "row index, silently overwriting an unrelated pixel.",
                n_out_of_bounds,
                n_x,
                n_y,
                n_z,
            )

        # The other write paths used to set this in their finalize step;
        # the root attrs builder reads it for msi_dataset_info.
        self._non_empty_pixel_count = pixel_count
        data_structures["pixel_count"] = pixel_count
        data_structures["avg_spectrum"] = total_intensity / max(pixel_count, 1)
        if region_total is not None:
            data_structures["avg_spectrum_per_region"] = {
                str(r): total / max(region_count.get(r, 0), 1)
                for r, total in region_total.items()
            }

    def _scatter_pass(self, data_structures: Dict[str, Any]) -> None:
        """Pass 2: resample every spectrum again and scatter it into its table.

        Same resampling as the pre-scan, so the two passes agree on which
        rows exist and how many entries each column holds; the assembly
        checks that agreement before it hands the matrix out. With
        ``passes`` it also scatters the sibling tables from the same
        frame read.
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
                    row = int(unit.row_of_grid[grid])

                if frame is not None:
                    passes.scatter(frame, row if row >= 0 else None)
                if row >= 0:
                    unit.assembly.scatter(row, mz_indices, values)

                pbar.update(1)

        logger.info("  Scatter complete")

    # ------------------------------------------------------------------
    # From memmaps to elements
    # ------------------------------------------------------------------

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Build every table, its shapes and its TIC image from the memmaps.

        Args:
            data_structures: What the passes filled.
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

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

        # Add average spectrum to .uns. The dataset-wide mean, the same
        # on every table: it is the per-pixel mean everywhere, including
        # the per-region block below.
        adata.uns["average_spectrum"] = data_structures["avg_spectrum"]
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
                    "spatial_y": y_idx * self.pixel_size_um,
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
                    "spatial_y": y_idx * self.pixel_size_um,
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
                [self.pixel_size_um, self.pixel_size_um, self.z_spacing_um],
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
            transform = Scale([self.pixel_size_um, self.pixel_size_um], axes=("x", "y"))
        return Image2DModel.parse(
            xr.DataArray(plane[np.newaxis, ...], dims=("c", "y", "x")),
            transformations={self.dataset_id: transform, "global": transform},
        )
