# thyra/converters/spatialdata/streaming_converter.py

"""Streaming SpatialData converter with direct Zarr write.

This converter processes MSI data in a memory-efficient streaming manner:
- Two-pass approach: count entries per column, then scatter straight into
  memory-mapped CSC arrays
- Writes directly to the final output without a scipy matrix in memory
- Writes CSC only; ask the in-memory converter for CSR
"""

import gc
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from tqdm import tqdm

from ...resampling import ResamplingMethod
from ._chunking import table_write_config
from .base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
    BaseSpatialDataConverter,
    _suppress_upstream_warnings,
)
from .csc_assembly import sort_csc_columns

if SPATIALDATA_AVAILABLE:
    import geopandas as gpd
    import xarray as xr
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, ShapesModel
    from spatialdata.transformations import Affine, Identity, Scale

logger = logging.getLogger(__name__)

# Elements per chunk when building the string index arrays. Bounds the
# transient cost of formatting to this many entries rather than the whole
# axis; measured at 17.5 bytes per entry at peak against 88 for a one-shot
# build, and 32 for an unchunked vectorised one.
_INDEX_BUILD_CHUNK = 1_000_000


class StreamingSpatialDataConverter(BaseSpatialDataConverter):
    """Memory-efficient streaming converter for MSI data to SpatialData format.

    One route, two passes over the reader -- **PCS** (Pre-calculated
    Scatter). Pass 1 counts entries per column, pass 2 scatters straight
    into memory-mapped CSC arrays and streams those to Zarr. The matrix is
    never a scipy object in RAM, and the table's Zarr layout is written by
    hand rather than through anndata (see :meth:`_write_csc_arrays_to_zarr`).

    There used to be a second route, COO (``use_csc=False``): count
    non-zeros per row, write CSR components to a temporary Zarr, read them
    back whole into a ``scipy.sparse.csr_matrix`` and ``.tocsc()`` it. It was
    the default until v3.19 on the assumption that PCS bought memory safety
    at a cost in speed. Measured on ``MockMSIReader``, peak process RSS
    sampled at 20 ms, one subprocess per route, that trade did not exist:

    ===========  ==============  ==============
    nnz          PCS             COO
    ===========  ==============  ==============
    16M          7.4 s / 540 MB  10.4 s / 655 MB
    64M          35.6 s / 1.1 GB 58.5 s / 1.9 GB
    ===========  ==============  ==============

    Both routes iterated the reader twice, so the gap was the
    materialise-then-convert step, and it widened with the dataset. The
    route was removed once it had been the escape hatch for one release and
    nothing reachable from ``convert()`` or the CLI could select it. The
    table stays here because it is the reason there is one route and not
    two.

    Note what the numbers do *not* say: PCS is not "~200 MB regardless of
    dataset size". Its RSS grew 540 MB -> 1.1 GB across those two points.
    Much of that is memmap pages, which count toward working set while
    remaining evictable -- but "bounded" was never true and is not claimed
    here.
    """

    def __init__(
        self,
        *args,
        use_csc: Union[bool, Literal["auto"]] = "auto",
        **kwargs,
    ):
        """Initialize streaming converter.

        Args:
            *args: Arguments passed to BaseSpatialDataConverter
            use_csc: Kept for the callers that pinned the PCS route while a
                second one existed (Ousia's import wizard passes ``True``).
                ``True`` and ``"auto"`` both mean this route, which is the
                only one; ``False`` used to select the COO route and now
                raises, since there is nothing left for it to select.
            **kwargs: Keyword arguments passed to BaseSpatialDataConverter

        Raises:
            ValueError: On ``use_csc=False``, and on ``sparse_format="csr"``.
                This route scatters straight into CSC arrays and has no CSR
                layout to write; it used to accept ``csr`` and silently
                store CSC. The in-memory converter (``streaming=False``)
                writes CSR.

        Note:
            Intensity thresholding (filtering noise below a minimum value) is
            handled at the reader level via the `intensity_threshold` parameter
            passed to the reader constructor.
        """
        kwargs["handle_3d"] = False  # Force 2D mode for now
        super().__init__(*args, **kwargs)

        if use_csc is False:
            raise ValueError(
                "use_csc=False selected the streaming COO route, which has been "
                "removed: the PCS route was faster and lighter at every size "
                "measured. Pass use_csc=True or leave it out."
            )
        if self._sparse_format != "csc":
            raise ValueError(
                "The streaming converter writes CSC only, but "
                f"sparse_format='{self._sparse_format}' was requested. Pass "
                "streaming=False to write CSR through the in-memory converter, "
                "or sparse_format='csc'."
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
        """Suppress progress output from reader during streaming passes."""
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

    def convert(self) -> bool:
        """Stream-convert MSI data to SpatialData format.

        Overrides the base convert() method: the table is scattered into
        memmapped CSC arrays and streamed to Zarr, never held as a scipy
        matrix (see the class docstring).

        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            # Initialize (loads metadata, mass axis, etc.)
            self._initialize_conversion()
            self._refuse_multiple_z_planes()

            # Diagnostic only -- nothing below depends on it. Called after
            # _initialize_conversion() so it reports the built axis instead
            # of guessing at it.
            self._estimate_output_size_gb()

            result = self._convert_to_csc_no_cache()
            logger.info(
                f"Zero-copy CSC conversion complete: {result['total_nnz']:,} non-zeros"
            )
            return True

        except Exception as e:
            logger.error(f"Error during zero-copy conversion: {e}")
            import traceback

            logger.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return False

        finally:
            # Scratch cleanup must run on EVERY exit path: success,
            # exception, and KeyboardInterrupt. The route that preceded
            # this one released its temp directory on the success path
            # only, and a failure mid-conversion (most commonly OOM or a
            # downstream Zarr write error) leaked it -- 79.5 GiB
            # accumulated in one user's system temp before a manual sweep.
            self._release_sibling_scratch()
            self.reader.close()

    def _refuse_multiple_z_planes(self) -> None:
        """Refuse a multi-plane acquisition, which this route cannot write.

        ``__init__`` forces ``handle_3d = False`` and the write path is
        single-slice throughout: the table is named ``_z0``, the TIC image
        is ``(n_y, n_x)``, the shapes are one polygon per (x, y), and obs
        is built from the kept grid positions of one plane. Only the matrix
        was ever sized ``n_x * n_y * n_z``, so ``n_z > 1`` has always ended
        in an obs-length mismatch ("Length of values (9) does not match
        length of index (18)"). Nothing has ever converted.

        Raising here is not a new restriction, then; it names the
        restriction instead of letting it surface as an arithmetic
        complaint from anndata two passes later.

        It is also load-bearing rather than cosmetic. The scatter indexes
        rows by ``y * n_x + x`` with no ``z`` term. The obs-length mismatch
        is the *only* thing that stops that: now that the empty rows are
        dropped the lengths would line up, and a two-plane file would
        convert cleanly with both planes summed onto one. A loud failure
        would have become a silent wrong answer. Use the 2D or 3D
        converter, both of which handle depth properly.

        Raises:
            ValueError: If the dataset declares more than one z plane.
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_z = self._dimensions[2]
        if n_z > 1:
            raise ValueError(
                f"The streaming converter writes a single z plane, but this "
                f"dataset declares {n_z}. Use SpatialData2DConverter (one "
                f"table per plane) or SpatialData3DConverter (one volume)."
            )

    def _init_region_accumulators(self, n_cols: int) -> Tuple[
        Optional[Dict[tuple, int]],
        Optional[Dict[int, NDArray[np.float64]]],
        Optional[Dict[int, int]],
    ]:
        """Initialise per-region accumulation structures.

        Returns:
            (region_map, region_total, region_count) -- all None when no
            region map is available.
        """
        region_map = self._region_map if hasattr(self, "_region_map") else None
        if region_map is None:
            return None, None, None
        unique_regions = sorted(set(region_map.values()))
        region_total = {r: np.zeros(n_cols, dtype=np.float64) for r in unique_regions}
        region_count: Dict[int, int] = {r: 0 for r in unique_regions}
        return region_map, region_total, region_count

    @staticmethod
    def _accumulate_region(
        region_map: Dict[tuple, int],
        region_total: Dict[int, NDArray[np.float64]],
        region_count: Dict[int, int],
        x: int,
        y: int,
        mz_indices: NDArray[np.int_],
        resampled_ints: NDArray[np.float64],
    ) -> None:
        """Accumulate spectrum into the appropriate region bucket."""
        rn = region_map.get((x, y), -1)
        if rn in region_total:
            np.add.at(region_total[rn], mz_indices, resampled_ints)
            region_count[rn] += 1

    @staticmethod
    def _compute_region_averages(
        region_total: Optional[Dict[int, NDArray[np.float64]]],
        region_count: Optional[Dict[int, int]],
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        """Compute per-region mean spectra from accumulators."""
        if region_total is None or region_count is None:
            return None
        return {
            str(r): total / max(region_count.get(r, 0), 1)
            for r, total in region_total.items()
        }

    def _process_spectrum(
        self, mzs: np.ndarray, intensities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single spectrum - resample and return indices/values.

        Note: Intensity thresholding is handled at the reader level before data
        reaches this method. This method only handles resampling and zero filtering.

        Args:
            mzs: Mass values (already filtered by reader if threshold is set)
            intensities: Intensity values (already filtered by reader if threshold is set)

        Returns:
            Tuple of (mz_indices, resampled_intensities) with zeros filtered out
        """
        if not self._resampling_config:
            # No resampling - map m/z values to indices directly. Entries
            # that share a bin (a repeated m/z) are summed so the CSC never
            # carries two values at one (row, col).
            mz_indices = self._map_mass_to_indices(mzs)
            return self._coalesce_duplicate_bins(mz_indices, intensities)

        # Optimized nearest-neighbor path; the route is resolved once in
        # __init__ because this runs once per spectrum per pass.
        if self._nn_route:
            return self._nearest_neighbor_resample(mzs, intensities)

        # TIC-preserving, evaluated only where the interpolant can be
        # non-zero and returned already zero-filtered. On a zero-suppressed
        # profile source the dense form of this call -- interpolate onto
        # every bin, then mask -- was 35x the cost of reading the file.
        return self._tic_preserving_resample_sparse(mzs, intensities)

    def _create_data_structures(self) -> Dict[str, Any]:
        """Not used in streaming mode - required by ABC."""
        raise NotImplementedError(
            "Streaming converter uses _convert_to_csc_no_cache instead"
        )

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Not used in streaming mode - required by ABC.

        The in-memory converters build their tables, shapes and images
        here from an in-RAM matrix. This route has none: the table goes
        to disk from memmaps, and the elements beside it are written by
        :meth:`_add_tic_image_and_shapes_to_store`.
        """
        raise NotImplementedError(
            "Streaming converter writes its elements from "
            "_write_csc_arrays_to_zarr instead"
        )

    # ========================================================================
    # No-Cache CSC Conversion (Optimized)
    # Two-pass approach without disk caching - processes spectra twice
    # but eliminates ~200GB cache file I/O
    # ========================================================================

    def _convert_to_csc_no_cache(self) -> Dict[str, Any]:
        """Convert MSI data to CSC sparse format without disk caching.

        This optimized method processes spectra twice but eliminates the
        large cache file (~200GB for big datasets):

        1. Pre-scan: Count entries per column + compute TIC + average spectrum
           - Light pass: just resampling and counting, no disk I/O
        2. Allocate memory-mapped files for CSC arrays
        3. Main pass: Process spectra again and scatter directly to CSC
           - If the reader handed its pixels over out of raster order,
             sort each column's rows afterwards, in place on the memmap
             (``csc_assembly.sort_csc_columns``, shared with the sibling
             tables), so the stored matrix is canonical either way
        4. Write CSC arrays to Zarr

        This works because nearest-neighbor resampling is deterministic -
        same input always produces same output. The 2x CPU cost of resampling
        is far less than the disk I/O cost of caching.

        Returns:
            Dictionary with conversion statistics
        """
        from uuid import uuid4

        if self._dimensions is None:
            raise ValueError("Dimensions not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis not initialized")

        n_x, n_y, n_z = self._dimensions
        n_grid = n_x * n_y * n_z
        n_cols = len(self._common_mass_axis)

        logger.info(
            f"Streaming CSC (no-cache): {n_grid:,} grid positions x "
            f"{n_cols:,} m/z bins"
        )

        # Create temp directory for memmap files only (no cache file)
        temp_dir = self.output_path.parent / f".streaming_csc_{uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # The sibling tables' sinks, fed from the two passes below when
            # the reader hands its frames over as records (design decision
            # D5): one raw read per frame per pass for every table.
            passes = self._fused_sibling_passes(f"{self.dataset_id}_z0")

            # Step 1: Pre-scan - count entries per column (no caching)
            logger.info("Step 1/3: Pre-scan (counting entries per column)...")
            prescan_result = self._prescan_count_columns(
                n_grid, n_cols, n_x, n_y, passes
            )

            col_counts = prescan_result["col_counts"]
            total_nnz = prescan_result["total_nnz"]
            tic_values = prescan_result["tic_values"]
            avg_spectrum = prescan_result["avg_spectrum"]
            pixel_count = prescan_result["pixel_count"]
            avg_per_region = prescan_result.get("avg_spectrum_per_region")

            if total_nnz == 0:
                logger.warning("No non-zero entries found!")

            kept_grid, row_of_grid = self._plan_row_layout(
                prescan_result["occupancy"], n_grid
            )
            n_rows = int(kept_grid.size)
            if passes is not None:
                passes.finish_counting(n_rows, self._register_sibling_scratch)

            # The other paths set this in _finalize_data; the root attrs
            # builder reads it for msi_dataset_info["non_empty_pixels"],
            # which would otherwise report the initial 0 on this route.
            self._non_empty_pixel_count = pixel_count

            # Build indptr from col_counts
            indptr = np.zeros(n_cols + 1, dtype=np.int64)
            indptr[1:] = np.cumsum(col_counts)

            # Step 2: Allocate memory-mapped files for CSC arrays
            logger.info(f"Step 2/3: Allocating memmap ({total_nnz:,} entries)...")
            mm_indices, mm_data = self._allocate_csc_memmap_arrays(total_nnz, temp_dir)

            # Step 3: Main pass - process and scatter directly to CSC
            logger.info("Step 3/3: Processing spectra and scattering to CSC...")
            rows_in_order = self._scatter_spectra_direct(
                mm_indices, mm_data, indptr, n_x, n_y, row_of_grid, pixel_count, passes
            )
            if passes is not None:
                passes.finish_scattering()
                self._take_fused_results(passes)
            if not rows_in_order:
                sort_csc_columns(mm_indices, mm_data, indptr, n_rows)

            # Write CSC arrays to Zarr
            logger.info("Writing CSC arrays to Zarr...")
            self._write_csc_arrays_to_zarr(
                mm_indices,
                mm_data,
                indptr,
                kept_grid,
                n_cols,
                total_nnz,
                tic_values,
                avg_spectrum,
                avg_per_region,
            )

            # Cleanup memmap references before deleting files
            del mm_indices, mm_data
            gc.collect()

            logger.info(f"Streaming CSC (no-cache) complete: {total_nnz:,} non-zeros")

            return {
                "total_nnz": total_nnz,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "pixel_count": pixel_count,
            }

        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _plan_row_layout(
        self, occupancy: NDArray[np.bool_], n_grid: int
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Decide which grid positions become table rows, and in what order.

        The PCS path used to emit one row per grid position, where the
        other three emit one per acquired spectrum -- acquisitions are
        polygon-shaped but the grid is their bounding box, so the corners
        came out as all-zero rows (#88). On real ``pea.imzML`` that is
        17,423 rows against 12,737 spectra, 4,686 of them empty, with
        ``shapes/`` carrying a polygon for each phantom.

        Kept rows stay in grid order and keep their **grid index** as
        ``instance_id``, which is what ``_drop_empty_pixels`` leaves
        behind on the other paths: the index has gaps, and a consumer can
        still recover the position from it. Only the row *offsets* are
        compacted, because the matrix has to be dense in its rows.

        Args:
            occupancy: True per grid position carrying a spectrum.
            n_grid: Number of grid positions.

        Returns:
            ``(kept_grid, row_of_grid)`` -- the grid index of each table
            row, and the table row of each grid position (-1 if dropped).
        """
        kept_grid = np.flatnonzero(occupancy).astype(np.int64)
        row_of_grid = np.full(n_grid, -1, dtype=np.int64)
        row_of_grid[kept_grid] = np.arange(kept_grid.size, dtype=np.int64)

        n_dropped = n_grid - int(kept_grid.size)
        if n_dropped:
            logger.info(
                "Dropping %d empty pixel rows from obs (%d non-empty pixels "
                "remain). These positions are inside the bounding box but "
                "outside the acquisition polygon.",
                n_dropped,
                kept_grid.size,
            )
        return kept_grid, row_of_grid

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

    def _prescan_count_columns(
        self,
        n_grid: int,
        n_cols: int,
        n_x: int,
        n_y: int,
        passes: Any = None,
    ) -> Dict[str, Any]:
        """Pre-scan spectra to count entries per column without caching.

        This is a lightweight pass that:
        1. Counts how many entries each m/z column will have (for CSC indptr)
        2. Computes TIC values per pixel
        3. Accumulates total intensity for average spectrum
        4. Records which grid positions carry a spectrum at all
        5. Feeds the sibling tables' pass-1 sinks, when ``passes`` is given,
           from the same frame read (see ``fused_passes.py``)

        No data is cached to disk - we'll reprocess spectra in the main pass.

        Args:
            n_grid: Number of grid positions
            n_cols: Number of m/z bins
            n_x, n_y: Spatial dimensions
            passes: The sibling sinks to feed, or ``None``

        Returns:
            Dictionary with col_counts, total_nnz, tic_values, avg_spectrum,
            pixel_count and occupancy
        """
        # Allocate counting arrays (very small memory footprint)
        col_counts = np.zeros(n_cols, dtype=np.int64)
        total_intensity = np.zeros(n_cols, dtype=np.float64)
        tic_values = np.zeros((n_y, n_x), dtype=np.float64)
        # Which grid positions carry a spectrum, in row-major order. The
        # table keeps only these; see the drop in _convert_to_csc_no_cache.
        occupancy = np.zeros(n_grid, dtype=bool)

        total_nnz = 0
        pixel_count = 0
        n_out_of_bounds = 0

        region_map, region_total, region_count = self._init_region_accumulators(n_cols)

        self._suppress_reader_progress()
        total_spectra = self._get_total_spectra_count()

        with tqdm(
            total=total_spectra,
            desc="Pre-scan" if passes is None else "Pre-scan + sibling tables",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities, frame in self._iter_pass_spectra(
                passes, "count"
            ):
                x, y, z = coords

                # Process spectrum (same resampling as main pass)
                mz_indices, resampled_ints = self._process_spectrum(mzs, intensities)

                nnz = len(mz_indices)
                if frame is not None:
                    # The row the sinks count under is the grid position,
                    # which is the table row's own order: rows are numbered
                    # in grid order once the empty positions are dropped.
                    # A spectrum that gets no row (empty, or off the grid)
                    # is a pixel the siblings skip, as their own pass
                    # skips one that is not in obs.
                    gets_row = nnz > 0 and 0 <= y < n_y and 0 <= x < n_x
                    passes.count(frame, (y * n_x + x) if gets_row else None)
                if nnz > 0:
                    # Accumulate for average spectrum (vectorized). Left
                    # outside the bounds check on purpose: the average is
                    # over every spectrum read rather than every spectrum
                    # stored, as it is on the in-memory converters.
                    np.add.at(total_intensity, mz_indices, resampled_ints)

                    # Column counts, TIC and occupancy all describe a
                    # spectrum that is going to get a row, so all three sit
                    # behind the bounds check: a reader yielding a
                    # coordinate outside the declared dimensions would
                    # otherwise wrap round and land on an unrelated pixel.
                    #
                    # col_counts and total_nnz size the CSC arrays, and the
                    # scatter pass skips exactly the spectra this rejects
                    # (row_of_grid gives them no row). Counting them here
                    # reserved slots nothing ever wrote: the memmap is
                    # zero-filled, so they surfaced as explicit zeros at
                    # row 0, out of order within their column, which is a
                    # matrix scipy reports as non-canonical.
                    if 0 <= y < n_y and 0 <= x < n_x:
                        np.add.at(col_counts, mz_indices, 1)
                        total_nnz += nnz
                        tic_values[y, x] = resampled_ints.sum()
                        occupancy[y * n_x + x] = True
                    else:
                        n_out_of_bounds += 1

                    if region_map is not None:
                        self._accumulate_region(
                            region_map,
                            region_total,
                            region_count,
                            x,
                            y,
                            mz_indices,
                            resampled_ints,
                        )

                pixel_count += 1
                pbar.update(1)

        # Compute average spectrum
        avg_spectrum = total_intensity / max(pixel_count, 1)

        logger.info(
            f"  Pre-scan complete: {total_nnz:,} entries across {n_cols:,} columns"
        )
        if n_out_of_bounds:
            logger.warning(
                "%d spectra sat outside the declared %dx%d grid and were "
                "skipped. Previously they were written to a wrapped-round "
                "row index, silently overwriting an unrelated pixel.",
                n_out_of_bounds,
                n_x,
                n_y,
            )

        return {
            "col_counts": col_counts,
            "total_nnz": total_nnz,
            "tic_values": tic_values,
            "avg_spectrum": avg_spectrum,
            "pixel_count": pixel_count,
            "occupancy": occupancy,
            "avg_spectrum_per_region": self._compute_region_averages(
                region_total, region_count
            ),
        }

    def _scatter_spectra_direct(
        self,
        mm_indices: np.memmap,
        mm_data: np.memmap,
        indptr: NDArray[np.int64],
        n_x: int,
        n_y: int,
        row_of_grid: NDArray[np.int64],
        pixel_count: int,
        passes: Any = None,
    ) -> bool:
        """Process spectra and scatter directly to CSC arrays.

        This is the main pass that processes spectra again (same resampling
        as pre-scan) and scatters values directly to their CSC positions.
        With ``passes`` it also scatters the sibling tables from the same
        frame read (see ``fused_passes.py``).

        Args:
            mm_indices: Memory-mapped array for row indices
            mm_data: Memory-mapped array for values
            indptr: Column pointers array (from pre-scan)
            n_x: Number of columns in spatial grid
            n_y: Number of rows in spatial grid
            row_of_grid: Table row for each grid position, -1 for the
                positions dropped as empty (from the pre-scan occupancy).
            pixel_count: Number of spectra to process

        Returns:
            Whether every scattered row came at or after the previous one,
            which is to say whether the reader handed its pixels over in
            raster order. Rows are numbered in raster order, so when it
            did, every column's row indices are already ascending and the
            matrix is canonical as scattered; when it did not, the caller
            sorts the columns before writing them.
        """
        # Current write position for each column
        write_pos = indptr[:-1].copy()
        rows_in_order = True
        last_row = -1

        # Reset reader for second pass
        # For real readers (ImzML, Bruker), iter_spectra() is a generator factory
        # that creates a fresh iterator each time - no need to recreate the reader.
        # For mock readers with random data, we need to reset the random seed.
        if hasattr(self.reader, "reset"):
            self.reader.reset()

        self._suppress_reader_progress()

        with tqdm(
            total=pixel_count,
            desc="Scatter to CSC" if passes is None else "Scatter to CSC + siblings",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities, frame in self._iter_pass_spectra(
                passes, "scatter"
            ):
                x, y, z = coords

                # Process spectrum (deterministic - same result as pre-scan)
                mz_indices, resampled_ints = self._process_spectrum(mzs, intensities)

                # Table row for this grid position -- not the grid index
                # itself, because the empty positions are dropped and the
                # rows compacted. row_of_grid comes from the same pre-scan
                # that sized the columns, so the two agree by construction;
                # a position with no row (empty, or a coordinate outside
                # the grid, both skipped there) comes back as -1.
                row_idx = -1
                if len(mz_indices) > 0 and 0 <= y < n_y and 0 <= x < n_x:
                    row_idx = int(row_of_grid[y * n_x + x])

                if frame is not None:
                    passes.scatter(frame, row_idx if row_idx >= 0 else None)

                if row_idx >= 0:
                    if row_idx < last_row:
                        rows_in_order = False
                    last_row = row_idx

                    # Vectorized scatter
                    destinations = write_pos[mz_indices]
                    mm_indices[destinations] = row_idx
                    mm_data[destinations] = resampled_ints

                    # Increment write positions
                    write_pos[mz_indices] += 1

                pbar.update(1)

        # One flush at the end. A periodic flush used to run every 100k
        # spectra, but ``np.memmap.flush`` writes back the whole mapping
        # synchronously, serialising disk writeback into the compute loop;
        # the OS pager already writes dirty pages back in the background.
        # Scatter writes each position exactly once, so nothing is dirtied
        # twice and the deferred writeback costs the same total I/O.
        mm_indices.flush()
        mm_data.flush()

        logger.info("  Scatter complete, memmap flushed")
        return rows_in_order

    def _allocate_csc_memmap_arrays(
        self, total_nnz: int, temp_dir: Path
    ) -> Tuple[np.memmap, np.memmap]:
        """Allocate memory-mapped files for CSC indices and data arrays.

        Uses numpy memmap so the OS handles memory management via virtual memory.
        This keeps RAM usage minimal regardless of array size.

        Args:
            total_nnz: Total number of non-zero entries
            temp_dir: Directory for temporary files

        Returns:
            Tuple of (mm_indices, mm_data) memory-mapped arrays
        """
        # Ensure at least 1 element for empty datasets
        size = max(total_nnz, 1)

        # CSC indices (row indices for each non-zero) - int32 sufficient for rows
        mm_indices = np.memmap(
            temp_dir / "csc_indices.bin",
            dtype=np.int32,
            mode="w+",
            shape=(size,),
        )

        # CSC data (values for each non-zero)
        mm_data = np.memmap(
            temp_dir / "csc_data.bin",
            dtype=np.float64,
            mode="w+",
            shape=(size,),
        )

        logger.info(
            f"  Allocated memmap: {size * 4 / (1024**3):.2f} GB indices + "
            f"{size * 8 / (1024**3):.2f} GB data"
        )

        return mm_indices, mm_data

    def _write_uns_provenance(self, uns_group: "zarr.Group") -> None:
        """Write the shared provenance block into a hand-written ``uns`` group.

        This path composes the Zarr layout itself, so it cannot reach
        ``adata.uns`` the way ``_save_output`` does. It renders the same
        mapping through anndata's own element writer instead, which is
        what produces the ``encoding-type`` / ``encoding-version`` pairs
        ``read_lazy`` needs -- getting those right by hand is exactly
        what this path used to have to do, and exactly where it drifted
        from the other one.

        Only ``uns`` goes through anndata here. ``obs``/``var`` stay
        hand-written: they carry the pandas string dtypes this path
        exists to sidestep, whereas the provenance block is plain dicts,
        strings, lists and numbers.
        """
        from anndata.io import write_elem

        for key, value in self.build_uns_metadata().items():
            write_elem(uns_group, key, value)

    def _write_csc_arrays_to_zarr(
        self,
        mm_indices: np.memmap,
        mm_data: np.memmap,
        indptr: NDArray[np.int64],
        kept_grid: NDArray[np.int64],
        n_cols: int,
        total_nnz: int,
        tic_values: NDArray[np.float64],
        avg_spectrum: NDArray[np.float64],
        avg_spectrum_per_region: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """Write CSC arrays to Zarr store with SpatialData-compatible structure.

        Creates the complete Zarr directory structure including:
        - CSC sparse matrix (indptr, indices, data)
        - Observation metadata (coordinates, region keys)
        - Variable metadata (m/z values)
        - Essential metadata and average spectrum

        Reads from memmap sequentially and writes in aligned chunks for efficiency.

        Args:
            mm_indices: Memory-mapped CSC indices array.
            mm_data: Memory-mapped CSC data array.
            indptr: Column pointers for CSC format.
            kept_grid: Grid index of each table row, in row order (see
                :meth:`_plan_row_layout`). Its length is the row count.
            n_cols: Number of columns in the matrix.
            total_nnz: Total non-zero count.
            tic_values: TIC image array. Full-grid, and deliberately not
                subset with the rows: it is a dense 2D image, not a
                per-pixel table.
            avg_spectrum: Average spectrum.
            avg_spectrum_per_region: Per-region mean spectra, or None.
        """
        slice_id = f"{self.dataset_id}_z0"
        region_key = f"{slice_id}_pixels"
        if self._dimensions is None:
            raise ValueError("Dimensions not initialized")
        n_x, n_y, _ = self._dimensions
        n_rows = int(kept_grid.size)
        # Decide on the sibling tables before uns is written, so the
        # table's uns can name them (see _collect_mobility_axis), and run
        # the raw mobility pass once for the heatmap and the grid's
        # discovery together -- unless both were done from the summed
        # table's own passes already (see _fused_sibling_passes). The
        # siblings' obs mirrors this table's rows: one per kept grid
        # position, indexed by the grid index as a string.
        if not self._siblings_planned:
            self._mobility_table_key = self._plan_mobility_table(slice_id)
            self._msms_table_key = self._plan_msms_table(slice_id)
        sibling_obs = self._sibling_obs(kept_grid, n_x, region_key)
        self._prepare_sibling_scans(sibling_obs, z_value=0)

        # Clean output directory
        if self.output_path.exists():
            shutil.rmtree(self.output_path)

        # Create Zarr store
        store = zarr.open_group(str(self.output_path), mode="w")

        # Root attributes. Everything but ``spatialdata_attrs`` -- which
        # describes the on-disk layout this method hand-writes, not the
        # dataset -- comes from the shared builder, so a root attr added
        # for the other paths reaches a PCS store too. This used to be six
        # literals composed here and was short by ``coordinate_systems``,
        # ``format_specific_metadata`` and ``msi_dataset_info``.
        store.attrs["spatialdata_attrs"] = {
            "version": "0.2",
            "spatialdata_software_version": "0.6.1",
        }
        store.attrs.update(self.build_root_attrs())

        # Create table structure
        tables_group = store.create_group("tables")
        table_group = tables_group.create_group(slice_id)

        table_group.attrs["encoding-type"] = "anndata"
        table_group.attrs["encoding-version"] = "0.1.0"
        table_group.attrs["spatialdata-encoding-type"] = "ngff:regions_table"
        table_group.attrs["region"] = region_key
        table_group.attrs["region_key"] = "region"
        table_group.attrs["instance_key"] = "instance_key"
        table_group.attrs["version"] = "0.2"

        # Empty groups required by AnnData
        for group_name in ["layers", "obsm", "obsp", "varm", "varp"]:
            g = table_group.create_group(group_name)
            g.attrs["encoding-type"] = "dict"
            g.attrs["encoding-version"] = "0.1.0"

        raw_arr = table_group.create_array("raw", data=np.array(False))
        raw_arr.attrs["encoding-type"] = "array"
        raw_arr.attrs["encoding-version"] = "0.2.0"

        # X group (CSC matrix)
        X_group = table_group.create_group("X")
        X_group.attrs["encoding-type"] = "csc_matrix"
        X_group.attrs["encoding-version"] = "0.1.0"
        X_group.attrs["shape"] = [n_rows, n_cols]

        # Write indptr (small, do it directly) - use int64
        indptr_arr = X_group.create_array(
            "indptr",
            data=indptr.astype(np.int64),
            chunks=(min(len(indptr), 100_000),),
        )
        indptr_arr.attrs["encoding-type"] = "array"
        indptr_arr.attrs["encoding-version"] = "0.2.0"

        # Zarr chunk settings
        z_chunk = 1_000_000
        actual_nnz = max(total_nnz, 1)

        indices_arr = X_group.create_array(
            "indices",
            shape=(actual_nnz,),
            dtype=np.int32,
            chunks=(min(actual_nnz, z_chunk),),
        )
        indices_arr.attrs["encoding-type"] = "array"
        indices_arr.attrs["encoding-version"] = "0.2.0"
        data_arr = X_group.create_array(
            "data",
            shape=(actual_nnz,),
            dtype=np.float64,
            chunks=(min(actual_nnz, z_chunk),),
        )
        data_arr.attrs["encoding-type"] = "array"
        data_arr.attrs["encoding-version"] = "0.2.0"

        # Sequential transfer from memmap to Zarr (aligned to chunks)
        read_buffer_size = z_chunk * 50  # ~200 MB buffer

        with tqdm(
            total=total_nnz,
            desc="Step 4/4: Writing to Zarr",
            unit="entries",
            unit_scale=True,
        ) as pbar:
            for start in range(0, total_nnz, read_buffer_size):
                end = min(start + read_buffer_size, total_nnz)
                indices_arr[start:end] = mm_indices[start:end]
                data_arr[start:end] = mm_data[start:end]
                pbar.update(end - start)

        # obs (coordinates)
        str_dtype = np.dtypes.StringDType()
        obs_group = table_group.create_group("obs")
        obs_group.attrs["encoding-type"] = "dataframe"
        obs_group.attrs["encoding-version"] = "0.2.0"
        obs_group.attrs["_index"] = "instance_id"
        obs_group.attrs["column-order"] = [
            "y",
            "x",
            "region",
            "spatial_x",
            "spatial_y",
            "region_number",
            "instance_key",
        ]

        # Positions of the kept rows only, recovered from their grid
        # indices. The index stays the GRID index, gaps included, which is
        # what _drop_empty_pixels leaves behind on the other paths -- the
        # row offsets compact, the identities do not.
        y_values = (kept_grid // n_x).astype(np.int32)
        x_values = (kept_grid % n_x).astype(np.int32)
        # Same reasoning as the var index below, but n_rows is the pixel count
        # and stays far smaller than n_cols, so one shot needs no chunking.
        instance_ids = kept_grid.astype(str_dtype)
        spatial_x = x_values.astype(np.float64) * self.pixel_size_um
        spatial_y = y_values.astype(np.float64) * self.pixel_size_um
        region_numbers = self.build_region_numbers(x_values, y_values)

        a = obs_group.create_array("y", data=y_values)
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"
        a = obs_group.create_array("x", data=x_values)
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"
        a = obs_group.create_array("region_number", data=region_numbers)
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"
        a = obs_group.create_array("spatial_x", data=spatial_x)
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"
        a = obs_group.create_array("spatial_y", data=spatial_y)
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"
        a = obs_group.create_array("instance_id", data=instance_ids)
        a.attrs["encoding-type"] = "string-array"
        a.attrs["encoding-version"] = "0.2.0"
        a = obs_group.create_array("instance_key", data=instance_ids)
        a.attrs["encoding-type"] = "string-array"
        a.attrs["encoding-version"] = "0.2.0"

        # Region as categorical
        region_group = obs_group.create_group("region")
        region_group.attrs["encoding-type"] = "categorical"
        region_group.attrs["encoding-version"] = "0.2.0"
        region_group.attrs["ordered"] = False
        a = region_group.create_array(
            "categories", data=np.array([region_key], dtype=str_dtype)
        )
        a.attrs["encoding-type"] = "string-array"
        a.attrs["encoding-version"] = "0.2.0"
        a = region_group.create_array("codes", data=np.zeros(n_rows, dtype=np.int8))
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"

        # var (mass axis)
        var_group = table_group.create_group("var")
        var_group.attrs["encoding-type"] = "dataframe"
        var_group.attrs["encoding-version"] = "0.2.0"
        var_group.attrs["_index"] = "_index"
        # A reader whose native axis is not m/z keeps that axis here too, so
        # this route stores the same columns as the dataframe-based ones.
        annotations = self._validated_mass_axis_annotations()
        var_group.attrs["column-order"] = ["mz"] + sorted(annotations)

        mz_values = self._common_mass_axis
        if mz_values is None:
            raise RuntimeError("Common mass axis not initialized")
        # Built in chunks rather than from a list comprehension. Materialising
        # n_cols Python str objects first costs about 88 bytes per entry at
        # peak, against 17.5 for this; at 10 million bins that is 883 MB
        # versus 175 MB, on a path whose whole point is not holding the
        # dataset in RAM. The values are identical.
        mz_index = np.empty(n_cols, dtype=str_dtype)
        for start in range(0, n_cols, _INDEX_BUILD_CHUNK):
            stop = min(start + _INDEX_BUILD_CHUNK, n_cols)
            mz_index[start:stop] = np.strings.add(
                "mz_", np.arange(start, stop, dtype=np.int64).astype(str_dtype)
            )
        a = var_group.create_array("_index", data=mz_index)
        a.attrs["encoding-type"] = "string-array"
        a.attrs["encoding-version"] = "0.2.0"
        a = var_group.create_array("mz", data=mz_values)
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"

        for name in sorted(annotations):
            a = var_group.create_array(name, data=np.asarray(annotations[name]))
            a.attrs["encoding-type"] = "array"
            a.attrs["encoding-version"] = "0.2.0"

        # uns (metadata)
        uns_group = table_group.create_group("uns")
        uns_group.attrs["encoding-type"] = "dict"
        uns_group.attrs["encoding-version"] = "0.1.0"

        sd_attrs = uns_group.create_group("spatialdata_attrs")
        sd_attrs.attrs["encoding-type"] = "dict"
        sd_attrs.attrs["encoding-version"] = "0.1.0"
        a = sd_attrs.create_array("region", data=np.array(region_key, dtype=str_dtype))
        a.attrs["encoding-type"] = "string"
        a.attrs["encoding-version"] = "0.2.0"
        a = sd_attrs.create_array(
            "region_key", data=np.array("region", dtype=str_dtype)
        )
        a.attrs["encoding-type"] = "string"
        a.attrs["encoding-version"] = "0.2.0"
        a = sd_attrs.create_array(
            "instance_key",
            data=np.array("instance_key", dtype=str_dtype),
        )
        a.attrs["encoding-type"] = "string"
        a.attrs["encoding-version"] = "0.2.0"

        self._write_uns_provenance(uns_group)

        a = uns_group.create_array("average_spectrum", data=avg_spectrum)
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"

        # Per-region mean spectra for multi-region datasets
        if avg_spectrum_per_region is not None:
            pr_group = uns_group.create_group("average_spectrum_per_region")
            pr_group.attrs["encoding-type"] = "dict"
            pr_group.attrs["encoding-version"] = "0.1.0"
            for region_key_str, region_avg in avg_spectrum_per_region.items():
                ra = pr_group.create_array(region_key_str, data=region_avg)
                ra.attrs["encoding-type"] = "array"
                ra.attrs["encoding-version"] = "0.2.0"

        # Create empty images and shapes groups (will be populated below)
        store.create_group("images")
        store.create_group("shapes")

        # Add TIC image and pixel shapes using SpatialData
        logger.info("  Adding TIC image and pixel shapes...")
        self._add_tic_image_and_shapes_to_store(
            tic_values, kept_grid, n_x, n_y, slice_id, region_key, sibling_obs
        )

        # Consolidate metadata after all elements are written
        logger.info("  Consolidating metadata...")
        with _suppress_upstream_warnings():
            zarr.consolidate_metadata(str(self.output_path))

    @staticmethod
    def _sibling_obs(
        kept_grid: NDArray[np.int64], n_x: int, region_key: str
    ) -> pd.DataFrame:
        """The ``obs`` a sibling table mirrors on this route."""
        kept = np.asarray(kept_grid, dtype=np.int64)
        obs = pd.DataFrame(
            {
                "x": kept % n_x,
                "y": kept // n_x,
                "region": np.full(kept.size, region_key),
            },
            index=kept.astype(str),
        )
        obs.index.name = "instance_id"
        return obs

    def _add_tic_image_and_shapes_to_store(
        self,
        tic_values: NDArray[np.float64],
        kept_grid: NDArray[np.int64],
        n_x: int,
        n_y: int,
        slice_id: str,
        region_key: str,
        sibling_obs: Optional[pd.DataFrame] = None,
    ) -> None:
        """Add TIC image and pixel shapes to the Zarr store using SpatialData.

        This method is called after the main table has been written to Zarr.
        It uses SpatialData's models to create properly formatted images and
        shapes, then writes them to the existing store.

        The TIC image stays full-grid while the shapes follow the table:
        the image is a dense raster of the bounding box, the shapes are the
        polygons the obs rows annotate, and those are different things.

        Args:
            tic_values: 2D array of TIC values (n_y, n_x).
            kept_grid: Grid index of each table row (see
                :meth:`_plan_row_layout`); one shape is emitted per entry.
            n_x: Number of pixels in x dimension.
            n_y: Number of pixels in y dimension.
            slice_id: Identifier for this slice (e.g., "msi_dataset_z0").
            region_key: Region key for shapes (e.g., "msi_dataset_z0_pixels").
            sibling_obs: The ``obs`` the sibling tables mirror, when the
                caller already built it for the scans; built here otherwise.
        """
        if not SPATIALDATA_AVAILABLE:
            logger.warning("SpatialData not available, skipping TIC image and shapes")
            return

        # === Create TIC Image ===
        y_size, x_size = tic_values.shape

        # Add channel dimension (c, y, x) as required by SpatialData
        tic_values_3d = tic_values.reshape(1, y_size, x_size)

        # The image array is intrinsically in raster pixel indices.
        # The transform to "global" expresses the conversion from raster
        # indices into the chosen global frame:
        #   - With FlexImaging optical alignment: Affine into optical
        #     image pixel space. global = optical pixels.
        #   - Without alignment: Scale into physical micrometers, so
        #     "global" agrees with the pixel-polygon shapes (which are
        #     stored in um). global = micrometers.
        tic_xarray = xr.DataArray(
            tic_values_3d,
            dims=("c", "y", "x"),
        )
        # Gate on apply_optical_alignment so the wizard's opt-out
        # path leaves the MSI TIC in micrometers (and the optical
        # image gets the inverse-alignment treatment elsewhere).
        if self._apply_optical_alignment and self._tic_to_image_matrix is not None:
            tic_transform = Affine(
                self._tic_to_image_matrix,
                input_axes=("x", "y"),
                output_axes=("x", "y"),
            )
        else:
            tic_transform = Scale(
                [self.pixel_size_um, self.pixel_size_um],
                axes=("x", "y"),
            )

        tic_image = Image2DModel.parse(
            tic_xarray,
            transformations={
                self.dataset_id: tic_transform,
                "global": tic_transform,
            },
        )

        # === Create Pixel Shapes ===
        gdf = self._create_streaming_pixel_shapes(kept_grid, n_x, n_y)

        shape_transform = Identity()
        shapes = ShapesModel.parse(
            gdf,
            transformations={
                self.dataset_id: shape_transform,
                "global": shape_transform,
            },
        )

        # === Write images and shapes via spatialdata's own element writer ===
        #
        # The CSC table is already hand-written to the Zarr store to keep
        # memory bounded.  The TIC image, pixel shapes, and any optical
        # images are small, so we let spatialdata write them through its
        # normal element writer.  That guarantees the produced OME-NGFF
        # metadata (``ome.version`` + ``multiscales``) matches whatever
        # spatialdata version is installed, so ``spatialdata.read_zarr``
        # round-trips on both the stock package and the Ousia fork.
        #
        # We deliberately do NOT call ``SpatialData.read(self.output_path)``
        # first: re-reading the hand-written store coupled element writing
        # to the hand-crafted store layout and -- combined with the
        # swallow below -- previously let a half-written, unreadable image
        # group pass as a successful conversion.  A fresh in-memory
        # SpatialData pointed at the existing store writes only the named
        # elements and leaves the hand-written table untouched.
        #
        # No try/except: any failure here propagates to convert(), which
        # returns False.  A corrupt zarr must never be reported as success.
        tic_name = f"{slice_id}_tic"
        data_structures: Dict[str, Any] = {
            "images": {tic_name: tic_image},
            "shapes": {region_key: shapes},
            "tables": {},
        }

        # The sibling tables, when the source supports one. Their obs
        # mirrors the hand-written table's rows (see _sibling_obs).
        if self._mobility_table_key is not None or self._msms_table_key is not None:
            kept = np.asarray(kept_grid, dtype=np.int64)
            if sibling_obs is None:
                sibling_obs = self._sibling_obs(kept, n_x, region_key)
            self._attach_sibling_tables(
                data_structures,
                slice_id,
                region_key,
                sibling_obs,
                z_value=0,
                # The summed table went straight to disk on this route and
                # is not in hand; its per-pixel ion current is, because the
                # TIC image is exactly that.
                summed_row_totals=np.asarray(tic_values, dtype=np.float64).ravel()[
                    kept
                ],
            )

        # Load optical images through the base converter's path so they get
        # the same multi-scale pyramid + chunked layout and identical
        # transforms as the in-memory converters.  Honours
        # self._include_optical internally.
        self._add_optical_images(data_structures)

        with _suppress_upstream_warnings():
            # This SpatialData intentionally carries no table (the table is
            # already on disk), so suppress the "table is annotating ...
            # which is not present" warning for the region it annotates.
            warnings.filterwarnings(
                "ignore",
                message="The table is annotating.*which is not present",
                category=UserWarning,
            )
            sdata = SpatialData(
                images=data_structures["images"],
                shapes=data_structures["shapes"],
                tables=data_structures["tables"],
            )
            sdata.path = Path(self.output_path)
            element_names = (
                list(data_structures["images"].keys())
                + list(data_structures["shapes"].keys())
                + list(data_structures["tables"].keys())
            )
            with table_write_config():
                sdata.write_element(element_names, overwrite=True)
        # The siblings were written from their memmaps; drop every
        # reference so their scratch directories can go.
        del sdata
        self._release_sibling_scratch(data_structures["tables"])

        # The optical images were declared as placeholders; their pixels
        # stream into the store now that the elements exist. An unreadable
        # TIFF is dropped from the store with a warning in there; anything
        # else propagates, since a half-written image group is a corrupt
        # store.
        n_optical = self._stream_pending_optical_pixels()

        logger.info(
            f"  Wrote TIC image '{tic_name}' ({x_size}x{y_size}), "
            f"{kept_grid.size:,} pixel shapes, and {n_optical} optical image(s)"
        )

    def _create_streaming_pixel_shapes(
        self, kept_grid: NDArray[np.int64], n_x: int, n_y: int
    ) -> "gpd.GeoDataFrame":
        """Create pixel shape geometries for the streaming converter.

        One polygon per table row, indexed by the same grid index obs uses,
        so the shapes and the table stay in step. This used to walk the
        whole bounding box, which is what put a polygon on each of real
        ``pea``'s 4,686 phantom rows.

        Args:
            kept_grid: Grid index of each table row, in row order.
            n_x: Number of pixels in x dimension.
            n_y: Number of pixels in y dimension.

        Returns:
            GeoDataFrame with pixel box geometries.
        """
        n_rows = int(kept_grid.size)
        y_indices = kept_grid // n_x
        x_indices = kept_grid % n_x

        from shapely import box as shapely_box_vectorized
        from shapely.geometry import box as shapely_box_single

        valid_indices: Optional[List[int]] = None

        # Gate on apply_optical_alignment so the wizard's opt-out
        # path produces pixel-polygon shapes in pure micrometer
        # coordinates (matching the MSI TIC image, which also takes
        # the micrometer Scale branch above).
        if (
            self._apply_optical_alignment
            and self._alignment_result is not None
            and self._alignment_result.region_mappings
        ):
            # Use optical alignment - transform raster coords to image pixels.
            default_half_pixel = self._alignment_result.region_mappings[
                0
            ].get_half_pixel_size()

            valid_geometries: List[Any] = []
            valid_indices = []
            for i in range(n_rows):
                rx, ry = int(x_indices[i]), int(y_indices[i])
                img_coords = self._alignment_result.transform_point(rx, ry)

                if img_coords is not None:
                    ix, iy = img_coords
                    half_pixel = self._alignment_result.get_half_pixel_size(rx, ry)
                    if half_pixel is None:
                        half_pixel = default_half_pixel
                    half_x, half_y = half_pixel
                    valid_geometries.append(
                        shapely_box_single(
                            ix - half_x, iy - half_y, ix + half_x, iy + half_y
                        )
                    )
                    valid_indices.append(int(kept_grid[i]))

            n_skipped = n_rows - len(valid_indices)
            if n_skipped > 0:
                logger.info(
                    f"Created {len(valid_geometries)} shapes using optical "
                    f"alignment (skipped {n_skipped} positions the alignment "
                    f"could not transform)"
                )

            geometries = valid_geometries
        else:
            # Physical micrometer coordinates
            half_pixel_um = self.pixel_size_um / 2
            spatial_x = x_indices * self.pixel_size_um
            spatial_y = y_indices * self.pixel_size_um

            geometries = shapely_box_vectorized(
                spatial_x - half_pixel_um,
                spatial_y - half_pixel_um,
                spatial_x + half_pixel_um,
                spatial_y + half_pixel_um,
            )

        # Create GeoDataFrame with string indices matching obs
        if valid_indices is not None:
            instance_ids = [str(i) for i in valid_indices]
        else:
            instance_ids = [str(i) for i in kept_grid.tolist()]
        return gpd.GeoDataFrame(geometry=geometries, index=instance_ids)
