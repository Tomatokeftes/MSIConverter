# thyra/metadata/extractors/waters_extractor.py
"""Waters-specific metadata extractor for MSI data.

Extracts essential and comprehensive metadata from Waters .raw imaging
files using the MassLynx native library and pre-built imaging grid.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ...core.base_extractor import MetadataExtractor
from ...resampling.constants import SpectrumType
from ..types import ComprehensiveMetadata, EssentialMetadata

if TYPE_CHECKING:
    from ...readers.waters.instrument import WatersInstrument
    from ...readers.waters.masslynx_lib import MassLynxLib

logger = logging.getLogger(__name__)


class WatersMetadataExtractor(MetadataExtractor):
    """Waters-specific metadata extractor.

    Uses the pre-built ImagingGrid for spatial dimensions and pixel sizes,
    and scans all MS spectra via the native library for mass range and
    peak count information.
    """

    def __init__(
        self,
        ml: "MassLynxLib",
        handle,  # opaque file handle
        data_path: Path,
        imaging_grid,  # ImagingGrid instance
        function_types: Dict[int, Any],
        ms_functions: List[int],
        instrument: Optional["WatersInstrument"] = None,
        use_centroid: bool = True,
        excluded_functions: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """Initialize Waters metadata extractor.

        Args:
            ml: MassLynxLib instance for native library access.
            handle: Opaque file handle from ml.open_file().
            data_path: Path to the Waters .raw directory.
            imaging_grid: Pre-built ImagingGrid with spatial metadata.
            function_types: Map of function index to FunctionType.
            instrument: What the run's side files say about the analyser,
                from :func:`thyra.readers.waters.instrument.identify_waters_instrument`.
                ``None`` reports nothing instrument-specific.
            use_centroid: Whether the reader that owns ``handle`` delivers
                the vendor centroid (``True``) or the profile trace. The
                representation reported in the essential metadata is the
                one the reader *delivers*, not the one the file was
                acquired in: downstream axis and method selection act on
                the spectra they will actually receive.
            ms_functions: List of MS function indices that are converted.
            excluded_functions: MS functions the reader left out because
                the file also holds MS1 ones (MSe, data-dependent runs):
                function index -> ``{"ms_level", "precursor_mz", "n_scans"}``.
        """
        super().__init__(ml)
        self._ml = ml
        self._handle = handle
        self._data_path = data_path
        self._imaging_grid = imaging_grid
        self._function_types = function_types
        self._ms_functions = ms_functions
        self._instrument = instrument
        self._use_centroid = use_centroid
        self._excluded_functions = dict(excluded_functions or {})

    def _extract_essential_impl(self) -> EssentialMetadata:
        """Extract essential metadata.

        Uses imaging grid for dimensions/pixel_size,
        scans all MS function spectra for mass range and peak counts.
        """
        grid = self._imaging_grid
        dimensions = grid.dimensions  # (n_x, n_y, 1)

        # Coordinate bounds (0-based pixel indices)
        coordinate_bounds = (
            0.0,
            float(grid.pixel_count_x - 1),
            0.0,
            float(grid.pixel_count_y - 1),
        )

        # Pixel size in micrometers
        pixel_size: Optional[Tuple[float, float]] = None
        if grid.pixel_size_x > 0 and grid.pixel_size_y > 0:
            pixel_size = (grid.pixel_size_x, grid.pixel_size_y)

        # Scan all MS spectra for mass range, spectrum count, peak counts
        observed_range, n_spectra, total_peaks, peak_counts = self._scan_all_ms_spectra(
            dimensions
        )
        mass_range = self._axis_mass_range(observed_range)

        # Memory estimate: total_peaks * 2 values (mz + intensity) * 8 bytes
        estimated_memory_gb = (total_peaks * 2 * 8) / (1024**3)

        # Spectrum type -- determine from first MS function
        spectrum_type = self._detect_spectrum_type()

        return EssentialMetadata(
            dimensions=dimensions,
            coordinate_bounds=coordinate_bounds,
            mass_range=mass_range,
            pixel_size=pixel_size,
            n_spectra=n_spectra,
            total_peaks=total_peaks,
            estimated_memory_gb=estimated_memory_gb,
            source_path=str(self._data_path),
            spectrum_type=spectrum_type,
            peak_counts_per_pixel=peak_counts,
        )

    def _axis_mass_range(self, observed: Tuple[float, float]) -> Tuple[float, float]:
        """The range the resampled axis is built over.

        The acquisition setting (``getAcquisitionRangeStart/End``, e.g.
        100-1000) rather than the span of the stored values (100.007-1000.000
        on the reference MRT run), so that two runs acquired with the same
        method land on the same generated axis and share bins -- the same
        rule the timsTOF route follows with ``MzAcqRangeLower/Upper``. The
        stored span alone would differ from run to run by whatever the
        first and last stored samples happened to be.

        Falls back to the stored span when no MS function reports an
        acquisition range, and widens the declared one to cover the stored
        span when a value lies outside it, since a range that drops
        measured data is worse than one that varies.
        """
        acquired = [
            r
            for r in (
                self._ml.get_acquisition_range(self._handle, f)
                for f in self._ms_functions
            )
            if r is not None
        ]
        if not acquired:
            logger.info(
                "No acquisition mass range reported for %s; the resampled "
                "axis spans the stored m/z values %.4f-%.4f",
                self._data_path.name,
                *observed,
            )
            return observed

        lo = min(float(r[0]) for r in acquired)
        hi = max(float(r[1]) for r in acquired)
        if lo <= observed[0] and hi >= observed[1]:
            logger.info(
                "Mass range %.4f-%.4f taken from the acquisition setting "
                "(stored values span %.4f-%.4f), so runs acquired with the "
                "same method share one resampled axis",
                lo,
                hi,
                *observed,
            )
            return (lo, hi)

        # Widen rather than abandon. The old fallback returned the observed
        # span and said "so nothing is dropped", which the axis builder
        # then contradicted in the same run: it drops the peaks sitting
        # exactly on the span's bounds, so the store lost 15 peaks of
        # 19.7 M on every Xevo DESI conversion measured while the log said
        # nothing was lost. The overshoot is real and small -- the vendor
        # centroid and the profile trace both exceed a declared 100-1200
        # range by 0.01 to 0.25 Da on those runs, which is why the shared
        # axis PR #207 added was never once used on them (issue #230).
        #
        # A widened range is run-specific where a purely declared one is
        # not, so two runs of one method share an axis only when they
        # overshoot alike. That is the honest trade: an axis that drops
        # measured data is worse than one that varies.
        widened = (min(lo, observed[0]), max(hi, observed[1]))
        logger.warning(
            "Stored m/z values %.4f-%.4f fall outside the acquisition range "
            "%.4f-%.4f; the resampled axis is widened to %.4f-%.4f to cover "
            "them. Runs of this method share one axis only if they overshoot "
            "the acquisition range by the same amount.",
            *observed,
            lo,
            hi,
            *widened,
        )
        return widened

    def _detect_spectrum_type(self) -> Optional[str]:
        """The representation the reader delivers: profile trace or centroid.

        MassLynx can hand back either for a profile-acquired file; which one
        is a mode switch on the open handle that the owning reader has
        already set. A centroid-acquired file has no trace to give, so it
        is centroid whatever was asked for -- said out loud, since a caller
        who asked for the profile would otherwise get a centroid store
        that claims nothing was lost.
        """
        if not self._ms_functions:
            return None
        acquired_profile = self._ml.is_raw_spectrum_profile(
            self._handle, self._ms_functions[0]
        )
        if not acquired_profile:
            if not self._use_centroid:
                logger.warning(
                    "%s was acquired in centroid mode, so there is no profile "
                    "trace to read; the vendor centroids are delivered instead",
                    self._data_path.name,
                )
            return SpectrumType.CENTROID
        if self._use_centroid:
            return SpectrumType.CENTROID
        return SpectrumType.PROFILE

    def _read_scan_mzs(self, func: int, scan: int) -> Optional[NDArray[np.floating]]:
        """Read m/z array for a single scan, returning None on failure."""
        scan_info = self._imaging_grid.scan_map.get((func, scan))
        if scan_info is None or not scan_info.has_position:
            return None

        coords = self._imaging_grid.get_coordinates(scan_info)
        if coords is None:
            return None

        try:
            mzs, _ = self._ml.read_spectrum(self._handle, func, scan)
        except Exception as e:
            logger.debug(f"Failed to read spectrum func={func} scan={scan}: {e}")
            return None

        return mzs if len(mzs) > 0 else None

    def _update_scan_stats(
        self,
        func: int,
        scan: int,
        mzs: NDArray[np.floating],
        stats: Dict[str, Any],
    ) -> None:
        """Update running statistics with data from one scan."""
        n_peaks = len(mzs)
        stats["min_mass"] = min(stats["min_mass"], float(mzs[0]))
        stats["max_mass"] = max(stats["max_mass"], float(mzs[-1]))
        stats["total_peaks"] += n_peaks

        coords = self._imaging_grid.get_coordinates(
            self._imaging_grid.scan_map[(func, scan)]
        )
        x, y, z = coords
        n_x, n_y = stats["n_x"], stats["n_y"]
        pixel_idx = z * (n_x * n_y) + y * n_x + x
        if 0 <= pixel_idx < stats["n_pixels"]:
            # Accumulate, and count pixels rather than scans. Two scans can
            # report the same stage position -- the registry's 100 um MALDI
            # set has 1275 positioned scans on 1274 pixels, the stage having
            # stopped between two acquisitions 40 ms apart -- and the
            # converter sums them into the one pixel. This used to overwrite
            # with the last scan's count while ``n_spectra`` counted both, so
            # the per-pixel counts and the totals that size the memory
            # estimate described two different datasets (issue #233).
            if stats["peak_counts"][pixel_idx] == 0:
                stats["n_spectra"] += 1
            stats["peak_counts"][pixel_idx] += n_peaks

    def _scan_all_ms_spectra(
        self,
        dimensions: Tuple[int, int, int],
    ) -> Tuple[Tuple[float, float], int, int, Optional[NDArray[np.int32]]]:
        """Single pass over all MS scans for mass range, counts, per-pixel peak counts.

        Returns:
            (mass_range, n_spectra, total_peaks, peak_counts_per_pixel)

        Raises:
            ValueError: If no scan yielded a mass range. This used to
                substitute a fabricated ``(0.0, 1000.0)``, which reached
                disk as the dataset's declared mass range and, with
                resampling on, sized the whole common axis. There is no
                defensible fallback range for a mass spectrometry dataset.
        """
        n_x, n_y, n_z = dimensions
        n_pixels = n_x * n_y * n_z

        stats: Dict[str, Any] = {
            "min_mass": float("inf"),
            "max_mass": float("-inf"),
            "total_peaks": 0,
            "n_spectra": 0,
            "peak_counts": np.zeros(n_pixels, dtype=np.int32),
            "n_x": n_x,
            "n_y": n_y,
            "n_pixels": n_pixels,
        }

        total_scans = sum(
            self._ml.get_number_of_scans_in_function(self._handle, f)
            for f in self._ms_functions
        )

        with tqdm(
            total=total_scans,
            desc="Scanning Waters metadata",
            unit="scan",
        ) as pbar:
            for func in self._ms_functions:
                n_scans = self._ml.get_number_of_scans_in_function(self._handle, func)
                for scan in range(n_scans):
                    pbar.update(1)
                    mzs = self._read_scan_mzs(func, scan)
                    if mzs is not None:
                        self._update_scan_stats(func, scan, mzs, stats)

        if stats["min_mass"] == float("inf"):
            raise ValueError(
                f"Could not determine the mass range of {self._data_path}: "
                f"none of the {total_scans:,} MS scans yielded a peak."
            )

        logger.info(
            f"Waters metadata scan complete: {stats['n_spectra']} spectra, "
            f"mass range {stats['min_mass']:.2f}-{stats['max_mass']:.2f}, "
            f"total peaks {stats['total_peaks']:,}"
        )

        return (
            (stats["min_mass"], stats["max_mass"]),
            stats["n_spectra"],
            stats["total_peaks"],
            stats["peak_counts"],
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        """Extract comprehensive metadata including Waters-specific details."""
        essential = self.get_essential()

        return ComprehensiveMetadata(
            essential=essential,
            format_specific=self._extract_waters_specific(),
            acquisition_params=self._extract_acquisition_params(),
            instrument_info=self._extract_instrument_info(),
            raw_metadata=self._extract_raw_metadata(),
        )

    def _extract_waters_specific(self) -> Dict[str, Any]:
        """Extract Waters format-specific metadata.

        ``format`` is the key ``DataCharacteristics.from_metadata`` reads for
        its format flags (``is_waters_raw`` prefix-matches on "Waters ").
        ``data_format`` predates it and only ever fed the stored
        format-specific block, so both are kept: renaming it would change
        stored metadata for no downstream gain.

        ``is_mrt`` and ``profile_sample_spacing_da_at_1000`` are likewise
        read by ``DataCharacteristics.from_metadata``: the first selects the
        MRT bin width, the second lets another Waters instrument's profile
        conversion size its bins from its own digitiser.
        """
        specific: Dict[str, Any] = {
            "format": "Waters MassLynx raw",
            "data_format": "waters_raw",
            "data_path": str(self._data_path),
            "is_imaging": True,
            # What the reader delivers, as distinct from what was acquired
            # (``acquisition_params["function_N_is_profile"]``).
            "spectrum_source": (
                "vendor_centroid" if self._use_centroid else "profile_trace"
            ),
        }
        if self._instrument is not None:
            spacing = self._instrument.profile_sample_spacing_da(1000.0)
            specific.update(
                {
                    "instrument": self._instrument.name,
                    "is_mrt": self._instrument.is_mrt,
                    "instrument_decided_by": self._instrument.decided_by,
                    "profile_sample_spacing_da_at_1000": spacing,
                }
            )
        specific.update(self._function_layout())
        return specific

    def _function_layout(self) -> Dict[str, Any]:
        """The function and grid facts that have always been reported here."""
        return {
            "n_functions": self._ml.get_number_of_functions(self._handle),
            "ms_functions": self._ms_functions,
            "function_types": {
                str(f): ft.name if hasattr(ft, "name") else str(ft)
                for f, ft in self._function_types.items()
            },
            # MS functions left out because the file also holds MS1 ones;
            # the store's spectra are MS1, and this is where the MS/MS
            # functions of an MSe or data-dependent run are still visible.
            "excluded_functions": {
                str(f): dict(detail) for f, detail in self._excluded_functions.items()
            },
            "pixel_count_x": self._imaging_grid.pixel_count_x,
            "pixel_count_y": self._imaging_grid.pixel_count_y,
            "lateral_width_um": self._imaging_grid.lateral_width,
            "lateral_height_um": self._imaging_grid.lateral_height,
        }

    def _extract_acquisition_params(self) -> Dict[str, Any]:
        """Extract acquisition parameters."""
        params: Dict[str, Any] = {}

        acq_date = self._ml.get_acquisition_date(self._handle)
        if acq_date:
            params["acquisition_date"] = acq_date

        # Profile/centroid info per MS function
        for func in self._ms_functions:
            is_profile = self._ml.is_raw_spectrum_profile(self._handle, func)
            params[f"function_{func}_is_profile"] = is_profile

        # Acquisition mass range per function
        for func in self._ms_functions:
            acq_range = self._ml.get_acquisition_range(self._handle, func)
            if acq_range:
                params[f"function_{func}_acq_range_start"] = acq_range[0]
                params[f"function_{func}_acq_range_end"] = acq_range[1]

        # Lock mass info
        params["is_lockmass_corrected"] = self._ml.is_lockmass_corrected(self._handle)
        lm_func = self._ml.get_lockmass_function(self._handle)
        params["lockmass_function"] = lm_func if lm_func >= 0 else None

        return params

    def _extract_instrument_info(self) -> Dict[str, Any]:
        """Extract instrument information.

        The key is ``manufacturer`` because that is what
        ``DataCharacteristics.from_metadata`` reads; this used to say
        ``vendor``, which nothing downstream consumed, so the fact that a
        file was Waters never reached the resampling detector chain.
        """
        info: Dict[str, Any] = {
            "manufacturer": "Waters",
            "format": "MassLynx .raw",
        }
        if self._instrument is not None and self._instrument.is_mrt:
            info["instrument_model"] = self._instrument.name
        if self._instrument is not None and self._instrument.resolution is not None:
            info["declared_resolution"] = self._instrument.resolution
        return info

    def _extract_raw_metadata(self) -> Dict[str, Any]:
        """Extract raw metadata dictionary."""
        return {
            "data_path": str(self._data_path),
            "imaging_grid": {
                "pixel_count_x": self._imaging_grid.pixel_count_x,
                "pixel_count_y": self._imaging_grid.pixel_count_y,
                "pixel_size_x_um": self._imaging_grid.pixel_size_x,
                "pixel_size_y_um": self._imaging_grid.pixel_size_y,
                "lateral_width_um": self._imaging_grid.lateral_width,
                "lateral_height_um": self._imaging_grid.lateral_height,
            },
        }
