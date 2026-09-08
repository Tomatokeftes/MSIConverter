# thyra/readers/waters/waters_reader.py
"""Waters .raw MSI reader using MassLynx native libraries.

Reads Waters mass spectrometry imaging data by calling the MassLynxRaw
and MLReader native C libraries via ctypes. The spatial pixel grid is
reconstructed from laser X/Y positions stored in each scan's metadata.
"""

import ctypes
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

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
from .imaging_grid import ImagingGrid, build_imaging_grid, regrid_for_functions
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
    5. Works out which functions hold the image, from the laser positions
    6. Iterates their spectra yielding (coords, mzs, intensities) tuples

    Step 5 exists because a Waters function is not necessarily one
    acquisition function. MassLynx caps a ``_FUNC*.DAT`` file at about
    1.6 GB and opens a new *function* when a long imaging run reaches it,
    so one raster arrives as several functions that tile the stage: they
    never share a pixel. It also names the last of them as the file's
    lockmass function, and reports MS level 1 for the first chunk, 2 for
    the middle ones and 0 for the last. None of that survived contact with
    real files (see :doc:`the D7 entry </design-decisions>`), so the reader
    decides from the laser positions instead, which are the same
    measurement the pixel grid is built from:

    * A function landing on pixels no earlier function covers **extends the
      raster** and is converted, whatever level MassLynx reports for it --
      including the chunk MassLynx calls the lockmass function, except while
      the run is read as centroids, which that one function's spectra cannot
      be (see :meth:`_raster_candidates`).
    * Functions competing for the same pixels are a parallel acquisition --
      MSe low and high energy, a data-dependent run, a co-acquired lockmass
      reference. Summing an MS1 and an MS2 spectrum into one pixel would
      make a spectrum of nothing, so among those the MS1 functions win when
      the file has any, and the rest are recorded under
      ``excluded_functions`` in the Waters-specific metadata block.

    A file whose converted functions carry a precursor m/z holds fragment
    spectra and says so through :meth:`get_fragmentation`.
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
        #: Functions not converted: function index -> what they held and
        #: why they stayed out.
        self._excluded_functions: Dict[int, Dict[str, Any]] = {}
        #: Raster chunks MassLynx refuses to centroid: index -> pixels lost.
        self._uncentroidable_functions: Dict[int, int] = {}
        #: Functions covering their own pixels off the raster the MS
        #: functions describe: index -> pixels they held.
        self._off_raster_functions: Dict[int, int] = {}
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

    def _function_dat_files(self) -> Dict[int, Path]:
        """The ``_FUNC*.DAT`` files present, keyed by 0-based function index.

        MassLynx numbers the files from one, so ``_FUNC001.DAT`` holds
        function 0.
        """
        found: Dict[int, Path] = {}
        for path in self.data_path.iterdir():
            name = path.name.upper()
            if not (name.startswith("_FUNC") and name.endswith(".DAT")):
                continue
            stem = name[len("_FUNC") : -len(".DAT")]
            if stem.isdigit():
                found[int(stem) - 1] = path
        return found

    def _validate_function_files(self, n_functions: int) -> None:
        """Refuse a .raw directory that has lost one of its function files.

        The DLL keeps reporting a function whose ``_FUNC*.DAT`` is gone --
        with 0 scans, no error and no warning -- so the only symptom is a
        smaller scan total than the acquisition had. Measured on a 16
        function file with ``_FUNC001.DAT`` deleted: still 16 functions,
        scans [0, 50, 55, ...], 820 positioned scans instead of 848, and
        the reader opened it without a word. On a raster MassLynx split
        across functions (functions 0/1/2 holding rows 0-19 / 19-38 /
        38-45) the same damage drops 20 rows of the image (issue #229).
        """
        present = self._function_dat_files()
        missing = [f for f in range(n_functions) if f not in present]
        if not missing:
            return
        raise ValueError(
            f"{self.data_path.name} declares {n_functions} function(s) but "
            f"{len(missing)} of them have no data file: "
            f"{', '.join(f'_FUNC{f + 1:03d}.DAT' for f in missing)}. "
            f"MassLynx still reports those functions, with 0 scans, so "
            f"converting this directory would silently drop whatever they "
            f"held -- on a raster split across functions, entire rows of "
            f"the image. Restore the missing file(s) from the acquisition."
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
        ms_functions = [
            f for f, ft in self._function_types.items() if ft == FunctionType.MS
        ]
        self._ms_functions = ms_functions

        if not self._ms_functions:
            self._ml.close_file(self._handle)
            self._handle = None
            raise ValueError(
                f"No MS functions found in {self.data_path}. "
                f"Function types: {self._function_types}"
            )

        # Refuse a file whose functions have lost their data. The DLL
        # reports a function whose _FUNC*.DAT is gone, with 0 scans, so
        # nothing downstream notices: on a raster MassLynx split across
        # functions, one missing chunk file silently drops its rows of the
        # image (issue #229).
        self._validate_function_files(n_funcs)

        # Build the imaging grid from the MS functions alone. They are what
        # defines the raster; everything else is measured against it. The
        # pitch used to come from every function including the ones
        # selection then excluded, so one reference spot parked off the
        # raster turned a 10x5 image at 30 um into an 11x6 one at
        # 1097 x 440 um (issue #232).
        self._imaging_grid = build_imaging_grid(
            self._ml,
            self._handle,
            self._function_types,
            include_functions=ms_functions,
        )

        # Which functions hold the image: the raster chunks MassLynx named
        # lockmass belong to it, the parallel functions do not. The first
        # step also records the chunks that had to stay out, so it has to
        # run before the second.
        candidates = self._raster_candidates(
            ms_functions, self._function_types, self._imaging_grid
        )
        self._ms_functions = self._select_converted_functions(
            candidates, self._imaging_grid
        )

        # Re-fit the raster to what is actually converted. A rescued chunk
        # extends the image past the extent of the MS functions, so the
        # grid has to grow to hold it; an excluded function's readings must
        # not leave their pitch behind in a grid it contributes no pixel to.
        self._imaging_grid = regrid_for_functions(
            self._imaging_grid, self._ms_functions
        )

        # A single-pixel grid is refused by build_imaging_grid() above rather
        # than warned about here: "this may not be true imaging data,
        # proceeding anyway" produced a 1x1 store with a 0.0 um pitch and
        # reported success, which is the defect in issue #213.

        logger.info(
            f"Initialized Waters reader: {self.data_path.name}, "
            f"{len(self._ms_functions)} MS functions, "
            f"grid {self._imaging_grid.pixel_count_x}x{self._imaging_grid.pixel_count_y}"
        )

    # ------------------------------------------------------------------
    # Which functions become the stored spectrum
    # ------------------------------------------------------------------

    @staticmethod
    def _positioned_scans(func: int, grid: "ImagingGrid") -> List["ScanInfoData"]:
        """The scan records of one function that carry a laser position.

        Reads the per-function index the grid builds once. This used to
        sort and filter the whole scan map on every call, and function
        selection asks five different questions of each function's scans:
        O(F x N log N) over the file, measured at 87 s of selection against
        a 29 s grid build on a single-function 25 M scan raster, and a
        25-function Synapt G1 run pays it 25 times over (issue #235).
        """
        return grid.scans_by_function.get(func, [])

    @classmethod
    def _function_pixels(
        cls, func: int, grid: "ImagingGrid"
    ) -> Set[Tuple[int, int, int]]:
        """The pixels one function's positioned scans land on."""
        pixels = (
            grid.get_coordinates(info) for info in cls._positioned_scans(func, grid)
        )
        return {c for c in pixels if c is not None}

    @classmethod
    def _comes_back_centroided(cls, func: int, grid: "ImagingGrid") -> bool:
        """Whether MassLynx honours the centroid request for this function.

        ``ScanInfo.isProfile`` is read after ``setCentroid``, so it reports
        what ``getDataPoints`` will hand back rather than how the run was
        acquired. It does not always agree with the request: the library's
        centroider skips the function ``getLockmassFunction`` names, which
        on a chunked imaging file is the last chunk of the raster.
        """
        scans = cls._positioned_scans(func, grid)
        return bool(scans) and not scans[0].is_profile

    def _raster_candidates(
        self,
        ms_functions: List[int],
        function_types: Dict[int, FunctionType],
        grid: "ImagingGrid",
    ) -> List[int]:
        """The MS functions, plus any lockmass function that extends the raster.

        MassLynx names the last function of a chunked imaging file as the
        file's lockmass function even though it is the tail of the raster.
        A function it calls lockmass belongs to the image when it covers
        pixels no MS function covers; a real reference function is acquired
        alongside the image and so covers nothing new. Only ever widens a
        file that already has an MS function, so a run with no MS function
        at all is still refused.

        The catch is that the same library will not centroid that function.
        Rescuing it into a store of centroids would put a band of profile
        rows across the top of the image -- measured at 3.3x the neighbouring
        rows' TIC -- so while the run is being read as centroids it stays
        out, and :attr:`_excluded_functions` records how many pixels that
        costs. Reading the run as the profile trace converts the whole image.
        """
        covered: Set[Tuple[int, int, int]] = set()
        for f in ms_functions:
            covered |= self._function_pixels(f, grid)

        extra, uncentroidable, off_raster = [], [], []
        for f, ft in sorted(function_types.items()):
            if ft is not FunctionType.LOCKMASS:
                continue
            # The rescue rule reads "covers pixels no MS function covers"
            # as "is the tail of a split raster". A reference spot parked
            # away from the sample covers a pixel no MS function covers
            # too, and used to be converted as an image pixel on that
            # basis. The tail of a split raster continues the raster, so
            # it lands on the same lattice; a spot does not (issue #232).
            #
            # Tested before the pixel count, not after: an off-raster
            # reading has no pixel in this grid at all, so it would
            # otherwise fall out as "covers nothing new" and never be
            # named.
            scans = self._positioned_scans(f, grid)
            if scans and not all(grid.lies_on_raster(s) for s in scans):
                off_raster.append(f)
                continue
            new_pixels = self._function_pixels(f, grid) - covered
            if not new_pixels:
                continue
            if self._use_centroid and not self._comes_back_centroided(f, grid):
                uncentroidable.append((f, len(new_pixels)))
            else:
                extra.append(f)

        for f in off_raster:
            self._off_raster_functions[f] = len(self._positioned_scans(f, grid))
        if off_raster:
            logger.warning(
                "Function(s) %s cover pixels no MS function covers, but "
                "their stage positions do not lie on the raster the MS "
                "functions describe, so they are not the tail of a split "
                "raster. They are excluded: a reference or calibration "
                "spot converted as an image pixel would stretch the grid "
                "around it.",
                ", ".join(str(f) for f in off_raster),
            )

        if uncentroidable:
            lost = sum(n for _f, n in uncentroidable)
            logger.warning(
                "Function(s) %s hold %d pixels (%.1f%% of the image) that no "
                "other function covers, but MassLynx names them the lockmass "
                "function and will not centroid them. They stay out rather "
                "than put profile rows in a table of centroids: pass "
                "--waters-spectrum profile to convert the whole image.",
                ", ".join(str(f) for f, _n in uncentroidable),
                lost,
                100.0 * lost / max(len(covered) + lost, 1),
            )
        if extra:
            logger.warning(
                "MassLynx names function(s) %s the lockmass function, but they "
                "cover pixels no MS function covers, so they are the tail of a "
                "raster MassLynx split across functions. Converting them too.",
                ", ".join(str(f) for f in extra),
            )
        self._uncentroidable_functions = dict(uncentroidable)
        return sorted(set(ms_functions) | set(extra))

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

    @classmethod
    def _pixel_groups(
        cls, ms_functions: List[int], grid: "ImagingGrid"
    ) -> List[List[int]]:
        """Group the functions that compete for the same pixels.

        Functions in one group were acquired in parallel and only one of
        them can be the pixel's spectrum. Separate groups tile the stage --
        they are the chunks MassLynx makes when a raster outgrows one
        ``_FUNC*.DAT`` file -- and every one of them is part of the image.
        """
        groups: List[Tuple[Set[Tuple[int, int, int]], List[int]]] = []
        for f in ms_functions:
            pixels = cls._function_pixels(f, grid)
            for covered, members in groups:
                if covered & pixels:
                    covered |= pixels
                    members.append(f)
                    break
            else:
                groups.append((pixels, [f]))
        return [members for _covered, members in groups]

    def _select_converted_functions(
        self, ms_functions: List[int], grid: "ImagingGrid"
    ) -> List[int]:
        """Keep every raster chunk, and one MS level per pixel within each.

        Chunks of one raster are all converted. Where functions do compete
        for a pixel, the MS1 ones win when the group has any; the rest are
        recorded in :attr:`_excluded_functions` so the store can say the
        acquisition fragmented something even though the stored spectra are
        intact-ion spectra.
        """
        self._excluded_functions = {
            f: {
                **self._function_summary(f, grid),
                "n_unique_pixels": n_pixels,
                "reason": "MassLynx will not centroid this function",
            }
            for f, n_pixels in self._uncentroidable_functions.items()
        }
        # No n_unique_pixels: an off-raster reading has no pixel in this
        # grid, which is the point. _function_summary's n_scans says how
        # much was left out.
        self._excluded_functions.update(
            {
                f: {
                    **self._function_summary(f, grid),
                    "reason": "stage positions do not lie on the imaging raster",
                }
                for f in self._off_raster_functions
            }
        )
        kept: List[int] = []
        for group in self._pixel_groups(ms_functions, grid):
            kept.extend(self._select_within_group(group, grid))
        return sorted(kept)

    def _function_summary(self, func: int, grid: "ImagingGrid") -> Dict[str, Any]:
        """What a function that was not converted held."""
        precursors = self._function_precursors(func, grid)
        return {
            "ms_level": self._function_ms_level(func, grid),
            "precursor_mz": (
                precursors[0]
                if len(precursors) == 1
                else (precursors if precursors else None)
            ),
            "n_scans": len(self._positioned_scans(func, grid)),
        }

    def _select_within_group(self, group: List[int], grid: "ImagingGrid") -> List[int]:
        """Which of a set of functions competing for one pixel set is stored."""
        levels = {f: self._function_ms_level(f, grid) for f in group}
        ms1 = [f for f in group if levels[f] == 1]
        if not ms1 or len(ms1) == len(group):
            if len(group) > 1:
                logger.warning(
                    "%d MS functions (%s) cover the same pixels; their spectra "
                    "are summed per pixel.",
                    len(group),
                    ", ".join(str(f) for f in group),
                )
            return list(group)

        for f in group:
            if levels[f] == 1:
                continue
            self._excluded_functions[f] = {
                **self._function_summary(f, grid),
                "reason": "covers the same pixels as an MS1 function",
            }
        excluded = [f for f in group if levels[f] != 1]
        logger.warning(
            "Functions %s are MS level %s and cover the same pixels as the "
            "MS1 function(s) %s. Only the MS1 spectra are converted; the "
            "others are recorded in the Waters metadata block as "
            "'excluded_functions'.",
            ", ".join(str(f) for f in excluded),
            "/".join(sorted({str(levels[f]) for f in excluded})),
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

        A converted function holds fragment spectra when MassLynx reports a
        precursor m/z for it. The reported MS level alone is not evidence:
        on every real multi-function imaging file measured, the chunks of
        one raster come back as level 1, then 2, then 0, with no precursor
        anywhere (see the D7 entry of the decisions page). So an MSe or
        data-dependent file, whose MS/MS functions were left out, reads as
        MS1 here -- what they were is in the Waters metadata block, since
        this describes the spectra in the store -- and so does a chunked
        MS1 raster. A file whose converted functions do carry precursors
        reports them: one isolation window per distinct precursor, and
        ``constant_across_pixels`` false as soon as a function's precursor
        changes from scan to scan.
        """
        _ml, _handle, grid, _types, ms_functions = self._require_initialized()
        precursors = {f: self._function_precursors(f, grid) for f in ms_functions}
        fragment_functions = [f for f in ms_functions if precursors[f]]
        if not fragment_functions:
            return FragmentationSchedule(ms_level=1, source=FRAGMENTATION_SOURCE)

        windows: Dict[float, IsolationWindow] = {}
        constant = True
        for f in fragment_functions:
            if len(precursors[f]) != 1:
                constant = False
                continue
            target = precursors[f][0]
            windows.setdefault(target, self._isolation_window(f, target, grid))
        return FragmentationSchedule(
            ms_level=max(
                [2] + [self._function_ms_level(f, grid) for f in fragment_functions]
            ),
            windows=tuple(windows[target] for target in sorted(windows)),
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
