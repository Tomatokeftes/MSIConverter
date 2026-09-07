# thyra/readers/waters/waters_reader.py
"""Waters .raw MSI reader using MassLynx native libraries.

Reads Waters mass spectrometry imaging data by calling the MassLynxRaw
and MLReader native C libraries via ctypes. The spatial pixel grid is
reconstructed from laser X/Y positions stored in each scan's metadata.
"""

import ctypes
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ...core.base_extractor import MetadataExtractor
from ...core.base_reader import BaseMSIReader
from ...core.msms import (
    COLLISION_INDUCED_DISSOCIATION_ACCESSION,
    FragmentationSchedule,
    IsolationWindow,
)
from ...core.registry import register_reader
from ...metadata.extractors.waters_extractor import WatersMetadataExtractor
from .imaging_grid import ImagingGrid, build_imaging_grid
from .instrument import WatersInstrument, identify_waters_instrument
from .masslynx_lib import FunctionType, MassLynxLib, ScanInfoData

logger = logging.getLogger(__name__)

#: What the fragmentation schedule names as its origin.
FRAGMENTATION_SOURCE = "waters_masslynx"


@register_reader("waters")
class WatersReader(BaseMSIReader):
    """Reader for Waters .raw MSI data using MassLynx native libraries.

    Supports Waters imaging data (.raw directories) containing spatial
    laser coordinates. Uses the same native DLLs as mzmine for data access.

    The reader:
    1. Loads MassLynxRaw + MLReader native libraries via ctypes
    2. Opens the .raw directory and verifies it contains imaging data
    3. Classifies acquisition functions (MS, IMS, MRM, lockmass)
    4. Reconstructs the imaging pixel grid from laser coordinates
    5. Keeps the MS1 functions only when the file also holds MS/MS ones
    6. Iterates MS spectra yielding (coords, mzs, intensities) tuples

    Every MS function shares the one laser grid, so two functions yield
    two spectra at the same pixel. Summing an MS1 and an MS2 spectrum into
    one pixel would make a spectrum of nothing, which is why the level
    filter in step 5 exists: an MSe or data-dependent acquisition converts
    to its MS1 image, and the functions left out are recorded in the
    Waters-specific metadata block. A file with no MS1 function converts
    its MS/MS functions instead and says so through
    :meth:`get_fragmentation`.
    """

    def __init__(
        self,
        data_path: Path,
        use_centroid: Optional[bool] = None,
        intensity_threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize a Waters MSI reader.

        Args:
            data_path: Path to the Waters .raw directory.
            use_centroid: If True, request vendor centroiding from the DLL.
                If False, read the profile trace. ``None`` (the default)
                decides from the instrument: the profile trace on a SELECT
                SERIES MRT, whose vendor peak picker merges near-isobars the
                trace resolves, and the vendor centroid on every other
                Waters instrument. See :mod:`thyra.readers.waters.instrument`.
            intensity_threshold: Minimum intensity value to include.
            **kwargs: Additional arguments passed to BaseMSIReader.
        """
        super().__init__(data_path, intensity_threshold=intensity_threshold, **kwargs)

        # Validate the .raw directory structure
        self._validate_raw_directory()

        # Which analyser wrote the run decides the default representation.
        # Read from the side files, so it costs no native-library call and
        # is known before the handle is opened.
        self._instrument: WatersInstrument = identify_waters_instrument(self.data_path)
        if use_centroid is None:
            use_centroid = not self._instrument.is_mrt
            logger.info(
                "%s: reading the %s by default (%s; pass --waters-spectrum "
                "to override)",
                self.data_path.name,
                "vendor centroid" if use_centroid else "profile trace",
                self._instrument.decided_by,
            )
        else:
            logger.info(
                "%s: reading the %s as requested (%s)",
                self.data_path.name,
                "vendor centroid" if use_centroid else "profile trace",
                self._instrument.name,
            )

        # Lazy initialization fields
        self._ml: Optional[MassLynxLib] = None
        self._handle: Optional[ctypes.c_void_p] = None
        self._imaging_grid: Optional[ImagingGrid] = None
        self._function_types: Optional[Dict[int, FunctionType]] = None
        self._ms_functions: Optional[List[int]] = None
        #: MS functions left out of the conversion because the file also
        #: holds MS1 ones: function index -> what they were.
        self._excluded_functions: Dict[int, Dict[str, Any]] = {}
        self._common_mass_axis_cache: Optional[NDArray[np.float64]] = None
        self._use_centroid = use_centroid
        self._closed = False

    def _validate_raw_directory(self) -> None:
        """Validate that data_path is a Waters .raw directory with _FUNC*.DAT files."""
        if not self.data_path.is_dir():
            raise ValueError(f"Waters .raw path must be a directory: {self.data_path}")

        # Check for _FUNC*.DAT files (case-insensitive)
        func_files = list(self.data_path.glob("_FUNC[0-9][0-9][0-9].DAT"))
        if not func_files:
            func_files = list(self.data_path.glob("_func[0-9][0-9][0-9].dat"))
        if not func_files:
            # Try broader pattern
            func_files = [
                f
                for f in self.data_path.iterdir()
                if f.name.upper().startswith("_FUNC")
                and f.name.upper().endswith(".DAT")
            ]
        if not func_files:
            raise ValueError(
                f"No _FUNC*.DAT files found in {self.data_path}. "
                "Is this a valid Waters .raw directory?"
            )

    def _ensure_initialized(self) -> None:
        """Lazy initialization: load DLL, open file, build imaging grid.

        Called before any data access. Safe to call multiple times.
        """
        if self._handle is not None:
            return

        if self._closed:
            raise RuntimeError("Reader has been closed and cannot be reused")

        # Load the native library (singleton)
        self._ml = MassLynxLib.get_instance(lib_dir=Path(__file__).parent / "lib")

        # Open the .raw directory
        self._handle = self._ml.open_file(str(self.data_path))

        # Verify this is an imaging file
        if not self._ml.is_imaging_file(self._handle):
            self._ml.close_file(self._handle)
            self._handle = None
            raise ValueError(
                f"{self.data_path} is not a Waters imaging file. "
                "The native library reports no imaging data."
            )

        # Set centroid mode
        self._ml.set_centroid(self._handle, self._use_centroid)

        # Classify all functions
        n_funcs = self._ml.get_number_of_functions(self._handle)
        self._function_types = {}
        for f in range(n_funcs):
            ft = self._ml.classify_function(self._handle, f)
            self._function_types[f] = ft
            logger.debug(f"Function {f}: {ft.name}")

        # Filter to MS functions only (skip lockmass, MRM, IMS, NOT_MS)
        self._ms_functions = [
            f for f, ft in self._function_types.items() if ft == FunctionType.MS
        ]

        if not self._ms_functions:
            self._ml.close_file(self._handle)
            self._handle = None
            raise ValueError(
                f"No MS functions found in {self.data_path}. "
                f"Function types: {self._function_types}"
            )

        # Build the imaging grid (scans all functions/scans for laser coordinates)
        self._imaging_grid = build_imaging_grid(
            self._ml, self._handle, self._function_types
        )

        # One MS level per store: keep the MS1 functions when there are any
        self._ms_functions = self._select_functions_by_ms_level(
            self._ms_functions, self._imaging_grid
        )

        # Verify the grid has more than one position (otherwise not really imaging)
        if (
            self._imaging_grid.pixel_count_x <= 1
            and self._imaging_grid.pixel_count_y <= 1
        ):
            logger.warning(
                "Imaging grid has only 1 pixel -- this may not be true imaging data. "
                "Proceeding anyway."
            )

        logger.info(
            f"Initialized Waters reader: {self.data_path.name}, "
            f"{len(self._ms_functions)} MS functions, "
            f"grid {self._imaging_grid.pixel_count_x}x{self._imaging_grid.pixel_count_y}"
        )

    # ------------------------------------------------------------------
    # MS level: which functions become the stored spectrum
    # ------------------------------------------------------------------

    @staticmethod
    def _positioned_scans(func: int, grid: "ImagingGrid") -> List["ScanInfoData"]:
        """The scan records of one function that carry a laser position."""
        return [
            info
            for (f, _scan), info in sorted(grid.scan_map.items())
            if f == func and info.has_position
        ]

    @classmethod
    def _function_ms_level(cls, func: int, grid: "ImagingGrid") -> int:
        """The MS level MassLynx reports for a function's scans.

        Read off the first positioned scan: MassLynx sets the level per
        function, so one scan speaks for all of them. A function without a
        positioned scan contributes no pixel and is treated as MS1 so that
        it is never the reason an MS1 image is refused.
        """
        scans = cls._positioned_scans(func, grid)
        return max(1, int(scans[0].ms_level)) if scans else 1

    def _select_functions_by_ms_level(
        self, ms_functions: List[int], grid: "ImagingGrid"
    ) -> List[int]:
        """Keep the MS1 functions when the file has any, else all of them.

        Records what was left out in :attr:`_excluded_functions` so the
        store can say the acquisition fragmented something even though
        the stored spectra are intact-ion spectra.
        """
        levels = {f: self._function_ms_level(f, grid) for f in ms_functions}
        ms1 = [f for f in ms_functions if levels[f] == 1]
        if not ms1 or len(ms1) == len(ms_functions):
            self._excluded_functions = {}
            if len(ms_functions) > 1:
                logger.warning(
                    "%d MS functions (%s) share one laser grid; their spectra "
                    "are summed per pixel.",
                    len(ms_functions),
                    ", ".join(str(f) for f in ms_functions),
                )
            return list(ms_functions)

        self._excluded_functions = {}
        for f in ms_functions:
            if levels[f] == 1:
                continue
            precursors = self._function_precursors(f, grid)
            self._excluded_functions[f] = {
                "ms_level": levels[f],
                "precursor_mz": (
                    precursors[0]
                    if len(precursors) == 1
                    else (precursors if precursors else None)
                ),
                "n_scans": len(self._positioned_scans(f, grid)),
            }
        logger.warning(
            "Functions %s are MS level %s and share the laser grid with the "
            "MS1 function(s) %s. Only the MS1 spectra are converted; the "
            "others are recorded in the Waters metadata block as "
            "'excluded_functions'.",
            ", ".join(str(f) for f in self._excluded_functions),
            "/".join(
                sorted({str(v["ms_level"]) for v in self._excluded_functions.values()})
            ),
            ", ".join(str(f) for f in ms1),
        )
        return ms1

    @classmethod
    def _function_precursors(cls, func: int, grid: "ImagingGrid") -> List[float]:
        """The distinct precursor m/z values a function's scans report, ascending."""
        values = {
            round(float(info.precursor_mz), 4)
            for info in cls._positioned_scans(func, grid)
            if info.precursor_mz > 0
        }
        return sorted(values)

    def get_fragmentation(self) -> Optional[FragmentationSchedule]:
        """What the stored spectra are: MS1, or the fragment spectra of what.

        MS level 1 whenever an MS1 function was converted, including the
        MSe and data-dependent files whose MS/MS functions were left out
        (those are in the Waters metadata block, not here, because this
        describes the spectra in the store). When the file holds MS/MS
        functions only, the schedule is their precursors: one isolation
        window per function whose precursor is constant, and
        ``constant_across_pixels`` false as soon as any function's
        precursor changes from scan to scan.
        """
        _ml, _handle, grid, _types, ms_functions = self._require_initialized()
        levels = {f: self._function_ms_level(f, grid) for f in ms_functions}
        top = max(levels.values(), default=1)
        if top <= 1:
            return FragmentationSchedule(ms_level=1, source=FRAGMENTATION_SOURCE)

        windows: List[IsolationWindow] = []
        constant = True
        for f in ms_functions:
            precursors = self._function_precursors(f, grid)
            if len(precursors) != 1:
                constant = False
                continue
            windows.append(self._isolation_window(f, precursors[0], grid))
        return FragmentationSchedule(
            ms_level=top,
            windows=tuple(windows),
            constant_across_pixels=constant,
            dissociation_accession=(
                COLLISION_INDUCED_DISSOCIATION_ACCESSION if windows else None
            ),
            source=FRAGMENTATION_SOURCE,
        )

    @classmethod
    def _isolation_window(
        cls, func: int, target: float, grid: "ImagingGrid"
    ) -> IsolationWindow:
        """One function's isolation window from its first positioned scan."""
        info = cls._positioned_scans(func, grid)[0]
        lower = upper = None
        start, end = float(info.quad_isolation_start), float(info.quad_isolation_end)
        if 0.0 < start <= target <= end:
            lower, upper = target - start, end - target
        energy = float(info.collision_energy)
        return IsolationWindow(
            target=target,
            lower_offset=lower,
            upper_offset=upper,
            collision_energy=energy if energy > 0 else None,
        )

    def _require_initialized(
        self,
    ) -> Tuple[
        "MassLynxLib",
        ctypes.c_void_p,
        "ImagingGrid",
        Dict[int, "FunctionType"],
        List[int],
    ]:
        """Ensure the reader is initialized and return required fields."""
        self._ensure_initialized()
        if (
            self._ml is None
            or self._handle is None
            or self._imaging_grid is None
            or self._function_types is None
            or self._ms_functions is None
        ):
            raise RuntimeError("Reader not fully initialized")
        return (
            self._ml,
            self._handle,
            self._imaging_grid,
            self._function_types,
            self._ms_functions,
        )

    @property
    def has_shared_mass_axis(self) -> bool:
        """Waters MSI data is typically processed/centroided with varying m/z per pixel."""
        return False

    @property
    def instrument(self) -> WatersInstrument:
        """What the run's own metadata says about the analyser."""
        return self._instrument

    @property
    def use_centroid(self) -> bool:
        """Whether spectra come from the vendor peak picker (else the profile trace)."""
        return self._use_centroid

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create Waters metadata extractor."""
        ml, handle, imaging_grid, function_types, ms_functions = (
            self._require_initialized()
        )
        return WatersMetadataExtractor(
            ml=ml,
            handle=handle,
            data_path=self.data_path,
            imaging_grid=imaging_grid,
            function_types=function_types,
            ms_functions=ms_functions,
            instrument=self._instrument,
            use_centroid=self._use_centroid,
            excluded_functions=self._excluded_functions,
        )

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Build common mass axis from all unique m/z values across all spectra.

        Since Waters MSI data is typically processed/centroided, each spectrum
        can have different m/z values. This iterates all MS spectra to collect
        all unique values and returns them sorted.

        Returns:
            Sorted array of unique m/z values across all spectra.
        """
        if self._common_mass_axis_cache is not None:
            return self._common_mass_axis_cache

        ml, handle, imaging_grid, function_types, ms_functions = (
            self._require_initialized()
        )

        all_mzs: list = []
        total = sum(ml.get_number_of_scans_in_function(handle, f) for f in ms_functions)

        with tqdm(total=total, desc="Building mass axis", unit="scan") as pbar:
            for func in ms_functions:
                n_scans = ml.get_number_of_scans_in_function(handle, func)
                for scan in range(n_scans):
                    pbar.update(1)

                    scan_info = imaging_grid.scan_map.get((func, scan))
                    if scan_info is None or not scan_info.has_position:
                        continue

                    try:
                        mzs, _ = ml.read_spectrum(handle, func, scan)
                        if mzs.size > 0:
                            all_mzs.append(mzs)
                    except Exception as e:
                        logger.debug(
                            f"Error reading spectrum func={func} scan={scan}: {e}"
                        )

        if not all_mzs:
            raise ValueError("No spectra found to build common mass axis")

        combined = np.concatenate(all_mzs)
        self._common_mass_axis_cache = np.unique(combined)

        logger.info(
            f"Built common mass axis with {len(self._common_mass_axis_cache):,} "
            f"unique m/z values from {len(all_mzs):,} spectra"
        )
        return self._common_mass_axis_cache

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Iterate through all MS imaging spectra with spatial coordinates.

        Only yields spectra from MS functions (skips LOCKMASS, MRM, IMS, NOT_MS).
        Skips scans without valid laser positions.

        Args:
            batch_size: Ignored (included for interface compatibility).

        Yields:
            Tuple of ((x, y, z), mzs, intensities) where coordinates are
            0-based pixel indices.
        """
        ml, handle, imaging_grid, function_types, ms_functions = (
            self._require_initialized()
        )

        total = sum(ml.get_number_of_scans_in_function(handle, f) for f in ms_functions)

        with tqdm(total=total, desc="Reading spectra", unit="spectrum") as pbar:
            for func in ms_functions:
                n_scans = ml.get_number_of_scans_in_function(handle, func)
                for scan in range(n_scans):
                    pbar.update(1)

                    # Look up cached scan info from the grid build pass
                    scan_info = imaging_grid.scan_map.get((func, scan))
                    if scan_info is None or not scan_info.has_position:
                        continue

                    # Map laser position to pixel coordinates
                    coords = imaging_grid.get_coordinates(scan_info)
                    if coords is None:
                        continue

                    try:
                        mzs, intensities = ml.read_spectrum(handle, func, scan)

                        # Apply intensity threshold filtering (from BaseMSIReader)
                        mzs, intensities = self._apply_intensity_filter(
                            mzs, intensities
                        )

                        if mzs.size > 0 and intensities.size > 0:
                            yield coords, mzs, intensities
                    except Exception as e:
                        logger.warning(
                            f"Error reading spectrum func={func} scan={scan}: {e}"
                        )

    def close(self) -> None:
        """Close the file handle and release native resources."""
        if self._closed:
            return
        if self._handle is not None and self._ml is not None:
            try:
                self._ml.close_file(self._handle)
                logger.debug(f"Closed Waters file: {self.data_path}")
            except Exception as e:
                logger.error(f"Error closing Waters file: {e}")
            self._handle = None
        self._closed = True

    def reset(self) -> None:
        """Reset reader for re-iteration.

        Stateless iteration (no internal cursor), so this is a no-op.
        The scan_map cache persists across iterations for efficiency.
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the reader."""
        grid_info = ""
        if self._imaging_grid:
            grid_info = (
                f", grid={self._imaging_grid.pixel_count_x}"
                f"x{self._imaging_grid.pixel_count_y}"
            )
        return f"WatersReader(path={self.data_path}{grid_info}, centroid={self._use_centroid})"
