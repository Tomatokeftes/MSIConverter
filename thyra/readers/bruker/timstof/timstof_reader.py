"""Bruker reader implementation combining best features from all implementations.

This module provides a high-performance, memory-efficient reader for
Bruker TSF/TDF data formats with lazy loading, intelligent caching, and
comprehensive error handling.
"""

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Set OpenMP thread limit before any SDK imports to control Bruker DLL
# threading
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"
    logging.getLogger(__name__).info(
        "Set OMP_NUM_THREADS=4 to limit Bruker DLL threading"
    )
else:
    current_setting = os.environ.get("OMP_NUM_THREADS")
    logging.getLogger(__name__).info(
        f"Using existing OMP_NUM_THREADS={current_setting}"
    )

from ....core.base_extractor import MetadataExtractor
from ....core.mobility import (
    INVERSE_REDUCED_MOBILITY_ACCESSION,
    MOBILITY_KIND_NAMES,
    MobilityAxis,
)
from ....core.msms import (
    COLLISION_INDUCED_DISSOCIATION_ACCESSION,
    FragmentationSchedule,
    IsolationWindow,
)
from ....core.registry import register_reader
from ....errors import ConversionRefused
from ....metadata.extractors.bruker_extractor import BrukerMetadataExtractor
from ....utils.bruker_exceptions import DataError, FileFormatError, SDKError
from ..base_bruker_reader import BrukerBaseMSIReader
from ..folder_structure import BrukerFolderStructure, BrukerFormat
from ..mis_parser import parse_mis_file
from .sdk.dll_manager import DLLManager
from .sdk.sdk_functions import (
    DEFAULT_TDF_SPECTRUM,
    TDF_SPECTRUM_MODES,
    SDKFunctions,
    sum_scans_per_index,
)

logger = logging.getLogger(__name__)

# Module-level dedupe for the "No calibration.sqlite" warning.  The
# reader is constructed multiple times per dataset open (preview +
# metadata + actual extraction); warning every time floods consumer
# session logs.  Keys are the resolved data_path strings; entries
# never get cleared (process lifetime is short enough).
_NO_CALIBRATION_WARNED: set = set()

# The unit of 1/K0 as PSI-MS names it; the same strings the Bruker
# metadata extractor reports, so the store and the schema block agree.
_ONE_OVER_K0_UNIT_ACCESSION = "MS:1002814"
_ONE_OVER_K0_UNIT_NAME = "volt-second per square centimeter"

#: Constructor keywords retired with the timsTOF utils package they
#: configured (issue #301).  They are named in a refusal rather than left
#: to ``**kwargs``, which forwards them to ``BaseMSIReader.__init__`` --
#: which never reads kwargs, so a caller who still passes one would hear
#: nothing at all.  That silence is what D10's Python-API half refuses: a
#: keyword accepted and doing nothing is the shape of option being cleared
#: out, and D10 refuses even ``sparse_format="csc"``, the value that asked
#: for what it would have got anyway.
RETIRED_INIT_KEYWORDS = ("cache_coordinates", "memory_limit_gb", "batch_size")

#: Said after every type refusal on a parameter that moved into the slots
#: the three above vacated.  A keyword refusal cannot see an old
#: positional call -- ``BrukerReader(path, True, True, 4.0, 100)`` arrives
#: with an empty ``kwargs`` and three truthy values bound to the wrong
#: parameters -- so the types are what catch it.
_POSITIONAL_SHIFT_NOTE = (
    "cache_coordinates, memory_limit_gb and batch_size were removed from the "
    "positional slots that used to precede this parameter, so a positional "
    "call written against the old signature binds their values to the "
    "parameters that moved up. Delete those arguments, or pass by keyword."
)


def build_raw_mass_axis(
    spectra_iterator: Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[NDArray[np.float64], int]:
    """Build raw mass axis from spectra iterator.

    Raw Mass axis in case the user wants the full data. Not recommended for
    normal use.
    Future interpolation module will create optimized mass axis using
    min/max mass + bin width.

    Also counts total peaks for COO matrix pre-allocation.

    Args:
        spectra_iterator: Iterator yielding (coords, mzs, intensities) tuples
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (numpy array of unique m/z values in ascending order, total_peaks)
    """
    from tqdm import tqdm

    unique_mzs: set[float] = set()
    count = 0
    total_peaks = 0

    # Create progress bar
    pbar = tqdm(
        desc="Building raw mass axis and counting peaks",
        unit=" spectra",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt} spectra [{elapsed}<{remaining}]",
    )

    try:
        for coords, mzs, intensities in spectra_iterator:
            if mzs.size > 0:
                unique_mzs.update(mzs)
                total_peaks += len(mzs)
            count += 1
            pbar.update(1)

            # Log progress periodically
            if count % 10000 == 0:
                pbar.set_postfix(
                    {
                        "unique_mz": len(unique_mzs),
                        "total_peaks": total_peaks,
                        "memory_est_mb": len(unique_mzs) * 8 / 1024 / 1024,
                    }
                )

            if progress_callback and count % 100 == 0:
                progress_callback(count)
    finally:
        pbar.close()

    logger.info(f"Total peaks counted: {total_peaks:,}")
    return (np.array(sorted(unique_mzs)), total_peaks)


def _get_frame_coordinates(
    db_path: Path,
    frame_id: int,
    coordinate_offsets: Optional[Tuple[int, int, int]] = None,
) -> Optional[Tuple[int, int, int]]:
    """Get normalized coordinates for a specific frame directly from database.

    Args:
        db_path: Path to the SQLite database file
        frame_id: Frame ID to look up
        coordinate_offsets: Optional coordinate offsets for normalization
                           (x_offset, y_offset, z_offset)

    Returns:
        Tuple of normalized (x, y, z) coordinates (0-based), or None if not
        found
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()

            # Check if this is MALDI data
            try:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos FROM MaldiFrameInfo WHERE "
                    "Frame = ?",
                    (frame_id,),
                )
                result = cursor.fetchone()
                if result:
                    x, y = result
                    # Apply coordinate offsets if provided (Bruker-specific
                    # normalization)
                    if coordinate_offsets:
                        offset_x, offset_y, offset_z = coordinate_offsets
                        return (int(x) - offset_x, int(y) - offset_y, 0)
                    else:
                        return (int(x), int(y), 0)
            except sqlite3.OperationalError:
                # No MALDI table, use generated coordinates
                pass

            # For non-MALDI data, generate coordinates (simple sequential
            # mapping)
            return (frame_id - 1, 0, 0)

    except Exception as e:
        logger.warning(f"Error getting coordinates for frame {frame_id}: {e}")
        return None


def _optional_float(value: Any) -> Optional[float]:
    """A nullable numeric column as a float, keeping NULL as ``None``."""
    return None if value is None else float(value)


def _optional_int(value: Any) -> Optional[int]:
    """A nullable integer column, keeping NULL as ``None``."""
    return None if value is None else int(value)


def _get_frame_count(db_path: Path) -> int:
    """Get total frame count directly from database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        Total number of frames
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Frames")
            return int(cursor.fetchone()[0])
    except Exception as e:
        logger.error(f"Error getting frame count: {e}")
        return 0


class TdfFrameScans:
    """One TDF frame's raw scan read, with every view of it derived on demand.

    The :class:`~thyra.core.frames.FrameScans` record of the Bruker
    reader: one ``tims_read_scans_v2`` over the full ramp at construction,
    ``tims_index_to_mz`` on the frame's unique indices the first time any
    view needs m/z, and the three views computed by the very helpers the
    reader's iterators use, so a converter fed from records sees what its
    own pass would have read.
    """

    __slots__ = (
        "coords",
        "frame_id",
        "indices",
        "intensities",
        "scans",
        "unique_indices",
        "inverse",
        "_reader",
        "_unique_mz",
    )

    def __init__(
        self, reader: "BrukerReader", frame_id: int, coords: Tuple[int, int, int]
    ) -> None:
        """Read the frame's scans once; every view is derived from them."""
        self.coords = coords
        self.frame_id = int(frame_id)
        self._reader = reader
        self.indices, self.intensities, self.scans = reader.sdk.read_tdf_scans(
            reader.handle,
            frame_id,
            0,
            reader._frame_num_scans(frame_id),
            reader._num_peaks_cache.get(frame_id),
        )
        if self.indices.size:
            unique_indices, inverse = np.unique(self.indices, return_inverse=True)
            self.unique_indices = unique_indices
            self.inverse = np.asarray(inverse).ravel()
        else:
            self.unique_indices = np.zeros(0, dtype=self.indices.dtype)
            self.inverse = np.zeros(0, dtype=np.int64)
        self._unique_mz: Optional[NDArray[np.float64]] = None

    @property
    def unique_mz(self) -> NDArray[np.float64]:
        """m/z of each unique digitizer index, ascending; converted once."""
        if self._unique_mz is None:
            self._unique_mz = self._reader.sdk.index_to_mz(
                self._reader.handle,
                self.frame_id,
                self.unique_indices.astype(np.float64),
            )
        return self._unique_mz

    def spectrum(
        self,
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """The summed spectrum as :meth:`BrukerReader.iter_spectra` yields it."""
        reader = self._reader
        try:
            if reader.tdf_spectrum == "vendor_centroid":
                # Bruker's own picker over the same ramp: a second library
                # call, which is the price of that mode; the raw read still
                # serves every other view.
                mzs, intensities = reader._read_frame_spectrum(self.frame_id)
            else:
                if self.indices.size == 0:
                    return None
                mzs = self.unique_mz
                intensities = sum_scans_per_index(
                    self.inverse, self.intensities, self.unique_indices.size
                )
            mzs, intensities = reader._apply_intensity_filter(mzs, intensities)
        except Exception as e:
            logger.warning(f"Error reading spectrum for frame {self.frame_id}: {e}")
            return None
        if mzs.size > 0 and intensities.size > 0:
            return mzs, intensities
        return None

    def mobility_points(
        self,
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
        """The point cloud as :meth:`BrukerReader.iter_mobility_spectra` yields it."""
        reader = self._reader
        try:
            values, n_axis = reader._mobility_values()
            return reader._mobility_points_from(self, values, n_axis)
        except Exception as e:
            logger.warning(
                f"Error reading mobility scans for frame {self.frame_id}: {e}"
            )
            return None

    def mobility_points_indexed(
        self,
    ) -> Optional[
        Tuple[
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ]:
        """The point cloud with the m/z left factored (:class:`~thyra.core.frames.FrameScans`).

        ``unique_mz[inverse]`` is :meth:`mobility_points`' ``mzs``: the
        frame is read as digitizer indices, and this hands over the
        conversion of the unique ones together with the map back to the
        points, which is what the reader has anyway.
        """
        reader = self._reader
        try:
            values, n_axis = reader._mobility_values()
            return reader._indexed_mobility_points_from(self, values, n_axis)
        except Exception as e:
            logger.warning(
                f"Error reading mobility scans for frame {self.frame_id}: {e}"
            )
            return None

    def precursor_spectra(
        self,
    ) -> List[Tuple[int, NDArray[np.float64], NDArray[np.float64]]]:
        """The fragment spectra as :meth:`BrukerReader.iter_precursor_spectra` yields them."""
        context = self._reader._precursor_context()
        if context is None:
            return []
        scan_map, n_windows = context
        return self._reader._precursor_spectra_from(self, scan_map, n_windows)


def _refuse_retired_keywords(passed: Dict[str, Any]) -> None:
    """Answer the keywords that went with the timsTOF utils package.

    All of the ones that were passed are named in one message. Refusing
    them one at a time would make a caller with an old three-keyword call
    run the conversion three times to learn it needs three edits.
    """
    retired = [name for name in RETIRED_INIT_KEYWORDS if name in passed]
    if not retired:
        return

    if len(retired) == 1:
        named, verb, noun, pronoun = retired[0], "was", "the argument", "it"
    else:
        named = f"{', '.join(retired[:-1])} and {retired[-1]}"
        verb, noun, pronoun = "were", "the arguments", "they"

    raise ConversionRefused(
        f"{named} {verb} removed: coordinates are read straight from the .d "
        "database and spectra one frame at a time, so there is nothing left "
        f"to cache, to cap or to batch. Delete {noun}; there is no "
        f"replacement keyword, because what {pronoun} configured was deleted "
        "as dead code."
    )


def _refuse_shifted_positional_types(
    progress_callback: object, region: object, metadata_only: object
) -> None:
    """Answer a value that none of these three parameters ever accepted.

    These are the parameters that moved up when the retired three were
    removed, and a positional call written against the old signature lands
    a bool, a float and an int in exactly this window. ``progress_callback``
    is the one that has to catch it: positional arguments are contiguous,
    so no old call could reach ``memory_limit_gb``'s slot or
    ``batch_size``'s without also filling ``cache_coordinates``', which is
    this one, and a bool is not callable. The other two are the same rule
    applied to the rest of the window, and each is independently right --
    a float region resolves to the string ``"4.0"`` and fails much later
    in ``--region`` vocabulary an API caller never used, and
    ``metadata_only=100`` builds a reader whose ``iter_spectra`` raises
    SDKError about a handle that is None.

    The arguments are typed ``object`` so that mypy, which is told these
    parameters are already a callable, an int-or-str and a bool, does not
    call the checks unreachable.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ConversionRefused(
            "progress_callback takes a callable or None, got "
            f"{type(progress_callback).__name__}. {_POSITIONAL_SHIFT_NOTE}"
        )
    if region is not None and (
        isinstance(region, bool) or not isinstance(region, (int, str))
    ):
        raise ConversionRefused(
            "region takes a DB RegionNumber int, a .mis Area Name string or "
            f"None, got {type(region).__name__}. {_POSITIONAL_SHIFT_NOTE}"
        )
    if not isinstance(metadata_only, bool):
        raise ConversionRefused(
            "metadata_only takes True or False, got "
            f"{type(metadata_only).__name__}. {_POSITIONAL_SHIFT_NOTE}"
        )


@register_reader("bruker")
class BrukerReader(BrukerBaseMSIReader):
    """Bruker reader for TSF/TDF data formats.

    Features:
    - Sequential spectrum iteration
    - Direct database coordinate access
    - Robust SDK integration with fallback mechanisms
    - Comprehensive error handling and recovery
    - Compatible with spatialdata_converter.py interface
    """

    def __init__(
        self,
        data_path: Path,
        use_recalibrated_state: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        region: Optional[Union[int, str]] = None,
        metadata_only: bool = False,
        tdf_spectrum: str = DEFAULT_TDF_SPECTRUM,
        **kwargs,
    ):
        """Initialize the Bruker reader.

        Args:
            data_path: Path to Bruker .d directory, or parent folder containing it.
                If a parent folder is provided, the reader will automatically find
                the .d folder within it.
            use_recalibrated_state: Whether to use recalibrated/active calibration state.
                Defaults to True (use active calibration). Set to False to use original
                calibration from data acquisition.
            progress_callback: Optional callback for progress updates
            region: Region selector for multi-region datasets.
                None (default): convert all regions (no filtering).
                int: select that DB RegionNumber (0-indexed).
                str: select by FlexImaging .mis Area Name (e.g. "03"); falls back
                to integer parse if the string doesn't match any area name.
            metadata_only: If True, skip the Bruker SDK DLL load and the
                NumPeaks preload pass.  All metadata reads (essential +
                comprehensive) come from the sqlite database alone and
                work without the vendor SDK.  Spectrum iteration is NOT
                possible in this mode -- attempting to call iter_spectra
                will raise SDKError because the SDK handle is None.
                Used by ``thyra.preview_msi`` so the Import Wizard's
                step 2 doesn't need the Bruker SDK to be installed.
            tdf_spectrum: How a TDF (TIMS engaged) frame's mobility scans
                are collapsed into the one spectrum yielded per pixel.
                ``"scan_sum"`` (default) sums every scan per digitizer
                index and keeps all of the ion current, which is also
                Bruker's own per-frame total; ``"vendor_centroid"`` is
                Bruker's frame-level centroid extraction over the full
                ramp, the same peak picker behind the TSF line spectrum.
                Ignored for TSF. See
                :mod:`thyra.readers.bruker.timstof.sdk.sdk_functions`.
            **kwargs: Additional arguments, forwarded to
                :class:`~thyra.core.base_reader.BaseMSIReader` (which reads
                ``intensity_threshold`` and ignores the rest).  Three names
                are the exception and are refused rather than forwarded:
                ``cache_coordinates``, ``memory_limit_gb`` and
                ``batch_size`` configured the timsTOF utils package, which
                was deleted as dead code (issue #301), so they have no
                replacement and the call should drop them.  Note that they
                also vacated the three positional slots now held by
                ``progress_callback``, ``region`` and ``metadata_only``: a
                positional call written against the old signature is
                refused on the type of what lands there.

        Raises:
            ConversionRefused: If ``cache_coordinates``, ``memory_limit_gb``
                or ``batch_size`` is passed; if ``progress_callback``,
                ``region`` or ``metadata_only`` is given a value its
                declared type does not allow; or if ``tdf_spectrum`` is not
                one of the supported modes.
        """
        _refuse_retired_keywords(kwargs)
        _refuse_shifted_positional_types(progress_callback, region, metadata_only)

        super().__init__(data_path, **kwargs)
        self.use_recalibrated_state = use_recalibrated_state
        self.progress_callback = progress_callback
        self._requested_region = region
        self._metadata_only = bool(metadata_only)
        if tdf_spectrum not in TDF_SPECTRUM_MODES:
            raise ConversionRefused(
                f"tdf_spectrum must be one of {TDF_SPECTRUM_MODES}, "
                f"got {tdf_spectrum!r}"
            )
        self.tdf_spectrum = tdf_spectrum

        # Validate and setup paths
        self._validate_data_path()
        self._detect_file_type()

        # Read calibration metadata
        self._calibration_metadata = self._read_calibration_metadata()

        # Initialize SDK + connections.  In metadata-only mode we
        # skip the SDK (no DLL load, no open_file) since all metadata
        # reads come from the sqlite database.
        if not self._metadata_only:
            self._initialize_sdk()
        else:
            self.dll_manager = None
            self.sdk = None
            self.handle = None
            logger.debug(
                "BrukerReader: metadata_only=True, skipping SDK initialization"
            )
        self._initialize_database()

        # Optical alignment data (parsed from .mis file + database).
        # Loaded BEFORE _select_region so Area-Name -> RegionNumber resolution
        # and startup logging have access to the area list.
        self._mis_metadata: Dict[str, Any] = self._parse_mis_alignment()

        # Region detection and selection (must be after database init)
        self._region_info: List[Tuple[int, int]] = self._detect_regions()
        self._selected_region: Optional[int]
        self._region_frame_ids: Optional[set]
        self._selected_region, self._region_frame_ids = self._select_region()
        self._positions: List[Dict[str, Any]] = self._build_positions_from_db()
        self._header: Dict[str, Any] = self._build_header_alignment()

        # Cached properties (lazy loaded)
        self._common_mass_axis: Optional[np.ndarray] = None
        self._frame_count: Optional[int] = None
        self._coordinate_offsets: Optional[Tuple[int, int, int]] = None
        self._mobility_axis: Optional[MobilityAxis] = None
        self._fragmentation: Optional[FragmentationSchedule] = None
        self._fragmentation_read: bool = False
        self._mobility_scan_overflow_warned: bool = False
        # (scan -> window map, window count) for the frame records' precursor
        # view; built once per reader, like iter_precursor_spectra's own.
        self._precursor_scan_map_cache: Optional[Tuple[NDArray[np.int64], int]] = None
        self._closed: bool = False  # Track if resources have been closed

        # Preload the per-frame NumPeaks (buffer sizing) and, for TDF, the
        # NumScans every read needs.  Skipped in metadata-only mode because
        # both are only consulted during spectrum iteration, which
        # metadata-only mode forbids.
        self._num_scans_cache: Dict[int, int] = {}
        if not self._metadata_only:
            self._num_peaks_cache: Dict[int, int] = self._preload_frame_num_peaks()
        else:
            self._num_peaks_cache = {}

        cache_status = (
            "metadata-only (SDK + NumPeaks skipped)"
            if self._metadata_only
            else (
                f"with {len(self._num_peaks_cache)} NumPeaks cached"
                if self._num_peaks_cache
                else "with fallback spectrum reading"
            )
        )
        logger.info(
            f"Initialized BrukerReader for {self.file_type.upper()} data at "
            f"{data_path} ({cache_status})"
        )

    def _validate_data_path(self) -> None:
        """Validate the data path and find .d directory if needed.

        If the path is not a .d directory, uses BrukerFolderStructure to
        find a .d folder within the given path. This allows users to pass
        a parent directory containing the .d folder.
        """
        if not self.data_path.exists():
            raise FileFormatError(f"Data path does not exist: {self.data_path}")

        if not self.data_path.is_dir():
            raise FileFormatError(f"Data path must be a directory: {self.data_path}")

        # If already a .d directory, use it directly
        if self.data_path.suffix.lower() == ".d":
            return

        # Otherwise, use BrukerFolderStructure to find the .d folder
        try:
            folder = BrukerFolderStructure(self.data_path)
            info = folder.analyze()

            if info.format != BrukerFormat.TIMSTOF:
                raise FileFormatError(
                    f"Path does not contain timsTOF data: {self.data_path}. "
                    f"Detected format: {info.format.value}"
                )

            # Update data_path to point to the actual .d folder
            self.data_path = info.data_path
            logger.info(f"Found .d folder at: {self.data_path}")

        except ValueError as e:
            raise FileFormatError(str(e)) from e
        except Exception as e:
            raise FileFormatError(
                f"Could not find .d folder in {self.data_path}: {e}"
            ) from e

    def _detect_file_type(self) -> None:
        """Detect whether this is TSF or TDF data."""
        tsf_path = self.data_path / "analysis.tsf"
        tdf_path = self.data_path / "analysis.tdf"

        if tsf_path.exists():
            self.file_type = "tsf"
            self.db_path = tsf_path
        elif tdf_path.exists():
            self.file_type = "tdf"
            self.db_path = tdf_path
        else:
            raise FileFormatError(
                f"No analysis.tsf or analysis.tdf found in {self.data_path}"
            )

        logger.debug(f"Detected file type: {self.file_type.upper()}")

    def _read_calibration_metadata(self) -> Optional[Dict]:
        """Read calibration metadata from calibration.sqlite.

        Returns:
            Dictionary containing calibration metadata, or None if unavailable.
            Keys include:
            - calibration_id: ID of active calibration state
            - calibration_uuid: Unique identifier for this calibration
            - calibration_datetime: When calibration was performed
            - calibration_source: Software that created calibration
            - num_calibration_versions: Total number of calibration states
            - recalibrated: Whether data has been recalibrated after acquisition
            - original_calibration_datetime: Original calibration datetime (if recalibrated)
        """
        cal_file = self.data_path / "calibration.sqlite"

        if not cal_file.exists():
            # Dedupe: this reader is instantiated multiple times per
            # dataset open (preview + metadata extraction + actual
            # read), so warning every time floods consumer logs.
            # Warn once per unique data_path per process.
            key = str(self.data_path)
            if key not in _NO_CALIBRATION_WARNED:
                _NO_CALIBRATION_WARNED.add(key)
                logger.warning(f"No calibration.sqlite found in {self.data_path}")
            return None

        try:
            conn = sqlite3.connect(cal_file)
            cursor = conn.cursor()

            # Count total calibration versions
            cursor.execute("SELECT COUNT(*) FROM CalibrationState")
            num_versions = cursor.fetchone()[0]

            # Get ACTIVE calibration (highest ID = most recent)
            cursor.execute(
                """
                SELECT Id, Key, DateTime, Source
                FROM CalibrationState
                ORDER BY Id DESC LIMIT 1
            """
            )
            cal_id, cal_uuid, cal_datetime, cal_source = cursor.fetchone()

            # Get original calibration if recalibrated
            original_datetime = None
            if num_versions > 1:
                cursor.execute(
                    """
                    SELECT DateTime FROM CalibrationState
                    ORDER BY Id ASC LIMIT 1
                """
                )
                original_datetime = cursor.fetchone()[0]

            # Get additional metadata from CalibrationInfo
            cursor.execute(
                """
                SELECT KeyName, Value
                FROM CalibrationInfo
                WHERE CalibrationState = ?
                AND KeyName IN ('CalibrationSoftwareVersion', 'CalibrationUser')
            """,
                (cal_id,),
            )

            extra_info = dict(cursor.fetchall())

            conn.close()

            metadata = {
                "calibration_id": cal_id,
                "calibration_uuid": cal_uuid,
                "calibration_datetime": cal_datetime,
                "calibration_source": cal_source,
                "calibration_software_version": extra_info.get(
                    "CalibrationSoftwareVersion"
                ),
                "calibration_user": extra_info.get("CalibrationUser"),
                "num_calibration_versions": num_versions,
                "recalibrated": num_versions > 1,
                "original_calibration_datetime": original_datetime,
                "calibration_file_size": cal_file.stat().st_size,
            }

            # Log which calibration is being used
            if self.use_recalibrated_state:
                recal_info = (
                    f" (recalibrated {num_versions} times)" if num_versions > 1 else ""
                )
                logger.info(
                    f"Using active calibration state {cal_id} from {cal_datetime}"
                    f"{recal_info}"
                )
            else:
                active_info = f", active state is {cal_id}" if num_versions > 1 else ""
                logger.info(
                    f"Using original calibration (use_recalibrated_state=False)"
                    f"{active_info}"
                )

            return metadata

        except Exception as e:
            logger.error(f"Failed to read calibration metadata: {e}")
            return None

    def _initialize_sdk(self) -> None:
        """Initialize the Bruker SDK with error handling."""
        try:
            # Initialize DLL manager
            self.dll_manager = DLLManager(
                data_directory=self.data_path, force_reload=False
            )

            # Initialize SDK functions
            self.sdk = SDKFunctions(
                self.dll_manager, self.file_type, tdf_spectrum=self.tdf_spectrum
            )

            # Open the data file
            self.handle = self.sdk.open_file(
                str(self.data_path), self.use_recalibrated_state
            )

            logger.debug(f"Successfully initialized {self.file_type.upper()} SDK")

        except Exception as e:
            logger.error(f"Failed to initialize SDK: {e}")
            raise SDKError(f"Failed to initialize Bruker SDK: {e}") from e

    def _initialize_database(self) -> None:
        """Initialize database connection with optimizations."""
        try:
            # Open database in read-only mode to avoid locking issues
            # This allows reading from network drives and concurrent access
            db_uri = f"file:{self.db_path}?mode=ro&immutable=1"
            self.conn = sqlite3.connect(
                db_uri, uri=True, timeout=30.0, check_same_thread=False
            )

            # Apply read-only compatible SQLite optimizations
            # Note: journal_mode and synchronous are not needed for read-only access
            self.conn.execute("PRAGMA cache_size = 10000")
            self.conn.execute("PRAGMA temp_store = MEMORY")

            logger.debug("Initialized database connection in read-only mode")

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.error(
                    f"Database is locked. This may occur if another process "
                    f"(e.g., DataAnalysis) has the file open: {self.db_path}"
                )
                raise DataError(
                    f"Database is locked. Please close any other applications "
                    f"that may have this dataset open: {e}"
                ) from e
            elif "unable to open database file" in str(e):
                logger.error(f"Cannot access database file: {self.db_path}")
                raise DataError(f"Cannot access database file: {e}") from e
            else:
                raise DataError(f"Failed to open database: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DataError(f"Failed to open database: {e}") from e

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create Bruker metadata extractor."""
        if not hasattr(self, "conn") or self.conn is None:
            raise ValueError("Database connection not available")
        # Pass selected region so extractor computes per-region bounds
        region_for_extractor = None
        if self._region_frame_ids is not None:
            region_for_extractor = self._selected_region
        return BrukerMetadataExtractor(
            self.conn,
            self.data_path,
            self._calibration_metadata,
            region=region_for_extractor,
            skip_total_peaks=self._metadata_only,
        )

    def _detect_regions(self) -> List[Tuple[int, int]]:
        """Detect available regions in the dataset.

        Returns:
            List of (region_number, frame_count) tuples sorted by frame
            count descending. Empty list if no region info available.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT RegionNumber, COUNT(*) as n_frames "
                "FROM MaldiFrameInfo "
                "GROUP BY RegionNumber "
                "ORDER BY COUNT(*) DESC"
            )
            regions = [(int(r), int(n)) for r, n in cursor.fetchall()]
            if len(regions) > 1:
                region_desc = ", ".join(
                    f"Region {r} ({n:,} frames)" for r, n in regions
                )
                logger.info(f"Detected {len(regions)} regions: {region_desc}")
            return regions
        except sqlite3.OperationalError:
            return []

    def _get_region_name_map(self) -> Dict[int, str]:
        """Build a mapping from DB RegionNumber to .mis Area Name.

        FlexImaging assigns ``MaldiFrameInfo.RegionNumber`` in the same order
        the areas appear in the ``.mis`` document, so the mapping is purely
        positional: the i-th area in the .mis file corresponds to
        RegionNumber i.
        """
        areas = self._mis_metadata.get("areas") if self._mis_metadata else None
        if not areas:
            return {}
        return {i: str(area.get("name", "")) for i, area in enumerate(areas)}

    def _log_region_mapping(self) -> None:
        """Log the DB RegionNumber to .mis Area Name mapping at startup.

        Surfaces the (often confusing) fact that ``--region <N>`` selects by
        DB index, not by the human-readable area label. Logged once per init.
        """
        if not self._region_info or len(self._region_info) <= 1:
            return
        name_map = self._get_region_name_map()
        if not name_map:
            return
        lines = ["Region mapping (DB RegionNumber -> .mis Area Name):"]
        for region_num, n_frames in self._region_info:
            name = name_map.get(region_num, "?")
            lines.append(
                f"  RegionNumber {region_num} -> Area '{name}' ({n_frames:,} frames)"
            )
        logger.info("\n".join(lines))

    def _resolve_requested_region(self) -> Optional[int]:
        """Resolve ``self._requested_region`` to a DB RegionNumber int.

        Accepts either an int (used as RegionNumber directly) or a string. A
        string is first matched against ``.mis`` Area Names; if no name
        matches it is parsed as an int.
        """
        if self._requested_region is None:
            return None
        if isinstance(self._requested_region, int):
            return self._requested_region

        request_str = str(self._requested_region)
        name_map = self._get_region_name_map()
        # Reverse map: name -> region_number (last writer wins; .mis names
        # are conventionally unique).
        name_to_num = {name: num for num, name in name_map.items() if name}
        if request_str in name_to_num:
            resolved = name_to_num[request_str]
            logger.info(
                f"Resolved --region '{request_str}' (Area Name) to "
                f"RegionNumber {resolved}"
            )
            return resolved
        try:
            resolved = int(request_str)
        except ValueError:
            available = ", ".join(sorted(name_to_num)) or "(none)"
            raise ConversionRefused(
                f"--region '{request_str}' is not a recognised .mis Area "
                f"Name and is not a valid integer. Available area names: "
                f"{available}"
            )
        # Said out loud, because the two spellings look alike and pick
        # different areas: on a three-area set '02' matches no Area Name,
        # falls through to here as RegionNumber 2, and selects Area '04'.
        # The name path has always logged what it resolved; this one
        # logged nothing at all (issue #252).
        available = ", ".join(sorted(name_to_num)) or "(none)"
        logger.info(
            "--region '%s' matches no .mis Area Name (found: %s), so it is "
            "read as DB RegionNumber %d. Area Names are what flexImaging "
            "shows; pass one of those to select by name.",
            request_str,
            available,
            resolved,
        )
        return resolved

    def _select_region(self) -> Tuple[Optional[int], Optional[set]]:
        """Select region based on user request or convert all regions.

        By default, all regions are converted (no filtering). The user can
        explicitly select a single region via the region= parameter.

        Returns:
            Tuple of (selected_region_number, set_of_frame_ids).
            Both are None when no filtering is needed (default: all regions).
        """
        # Always log the DB-number/name mapping for multi-region datasets so
        # users can correlate the log with their .mis labels (#89).
        self._log_region_mapping()

        if not self._region_info or len(self._region_info) <= 1:
            # A request still has to be answered. Returning here first
            # meant --region was never even parsed on a single-area set:
            # '--region foo' converted all 713 pixels at exit 0 while the
            # three-area set refused the same word (issue #252).
            self._check_region_on_single_region_set()
            return (None, None)

        # Multiple regions exist
        valid_regions = [r for r, _ in self._region_info]
        total_spectra = sum(n for _, n in self._region_info)

        resolved = self._resolve_requested_region()
        if resolved is not None:
            # User explicitly requested a specific region
            if resolved not in valid_regions:
                raise ConversionRefused(
                    f"Region {resolved} not found. "
                    f"Available regions: {valid_regions}"
                )

            # Get frame IDs for selected region
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT Frame FROM MaldiFrameInfo WHERE RegionNumber = ?",
                (resolved,),
            )
            frame_ids = {int(row[0]) for row in cursor.fetchall()}
            logger.info(
                f"Region {resolved}: {len(frame_ids):,} frames selected "
                f"(use region=None to convert all regions)"
            )
            return (resolved, frame_ids)

        # Default: convert all regions, no filtering
        logger.info(
            f"Multi-region dataset: converting all {len(self._region_info)} "
            f"regions ({total_spectra:,} total spectra). "
            f"Use region= parameter to select a specific region."
        )
        return (None, None)

    def _check_region_on_single_region_set(self) -> None:
        """Answer a ``--region`` request on a set with nothing to select from.

        There is no filtering to do either way -- one region is the whole
        dataset -- so a request naming that region converts exactly as it
        would have without it. A request naming anything else is refused,
        in the same words the multi-region path uses, rather than being
        dropped on the floor.
        """
        if self._requested_region is None:
            return

        if not self._region_info:
            raise ConversionRefused(
                f"--region {self._requested_region!r} was given but this "
                "dataset carries no region information (no MaldiFrameInfo "
                "table), so there is nothing to select. Drop the option to "
                "convert it."
            )

        resolved = self._resolve_requested_region()
        valid_regions = [r for r, _ in self._region_info]
        if resolved not in valid_regions:
            raise ConversionRefused(
                f"Region {resolved} not found. Available regions: " f"{valid_regions}"
            )
        logger.info(
            "Region %d is the only region in this dataset; all %d frames are "
            "converted.",
            resolved,
            self._region_info[0][1],
        )

    def get_region_map(self) -> Optional[Dict[tuple, int]]:
        """Get per-pixel region mapping from MaldiFrameInfo.

        Maps each normalized (0-based) (x, y) coordinate to its acquisition
        region number. Uses the same coordinate offsets as iter_spectra() so
        the mapping is consistent with obs indices.

        Returns:
            Dict mapping (x, y) tuples to region numbers, or None if
            region information is not available (single-region dataset).
        """
        if not self._region_info or len(self._region_info) <= 1:
            return None

        coordinate_offsets = self._get_coordinate_offsets()
        region_map: Dict[tuple, int] = {}

        try:
            cursor = self.conn.cursor()
            if self._selected_region is not None:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos, RegionNumber "
                    "FROM MaldiFrameInfo WHERE RegionNumber = ?",
                    (self._selected_region,),
                )
            else:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos, RegionNumber " "FROM MaldiFrameInfo"
                )

            for x, y, region_num in cursor.fetchall():
                nx, ny = int(x), int(y)
                if coordinate_offsets:
                    nx -= coordinate_offsets[0]
                    ny -= coordinate_offsets[1]
                region_map[(nx, ny)] = int(region_num)

            logger.info(
                f"Built region map: {len(region_map)} pixels across "
                f"{len(set(region_map.values()))} regions"
            )
            return region_map

        except sqlite3.OperationalError as e:
            logger.warning(f"Could not build region map: {e}")
            return None

    def get_region_info(self) -> Optional[list]:
        """Get summary information about acquisition regions.

        Returns:
            List of region summary dicts with region_number, n_spectra,
            and name (from .mis Area definitions when available),
            or None if region information is not available.
        """
        if not self._region_info or len(self._region_info) <= 1:
            return None

        areas = self._mis_metadata.get("areas", [])
        result = []
        for region_num, n_frames in self._region_info:
            info: Dict[str, Any] = {
                "region_number": region_num,
                "n_spectra": n_frames,
            }
            if region_num < len(areas) and areas[region_num].get("name"):
                info["name"] = areas[region_num]["name"]
            result.append(info)
        return result

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Return the common mass axis composed of all unique m/z values.

        Returns:
            Array of unique m/z values in ascending order
        """
        if self._common_mass_axis is None:
            self._common_mass_axis = self._build_common_mass_axis()

        return self._common_mass_axis

    def _build_common_mass_axis(self) -> NDArray[np.float64]:
        """Build the common mass axis and cache total peaks."""
        logger.info("Building raw mass axis")

        # Create iterator for mass axis building
        def mz_iterator():
            for coords, mzs, intensities in self._iter_spectra_raw():
                yield coords, mzs, intensities

        # Build raw mass axis using simplified function (returns mass_axis and total_peaks)
        mass_axis, total_peaks = build_raw_mass_axis(
            mz_iterator(), self.progress_callback  # type: ignore[arg-type]
        )

        # Cache total_peaks for later retrieval (if not already set from NumPeaks cache)
        if not hasattr(self, "_total_peaks_from_mass_axis"):
            self._total_peaks_from_mass_axis = total_peaks

        if len(mass_axis) == 0:
            logger.warning("No m/z values found in dataset")
            return np.array([])

        logger.info(f"Built raw mass axis with {len(mass_axis)} unique m/z values")
        return mass_axis

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Iterate through all spectra sequentially.

        Args:
            batch_size: Ignored, maintained for compatibility

        Yields:
            Tuples of (coordinates, mz_array, intensity_array)
        """
        # Always use simple sequential iteration
        yield from self._iter_spectra_raw()

    def _iter_spectra_raw(
        self,
    ) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Raw spectrum iteration without batching.

        One summed spectrum per frame, over the frames
        :meth:`_iter_frames` selects. A frame whose read fails is logged
        and skipped; a frame that reads empty is skipped silently.
        """
        for frame_id, coords in self._iter_frames():
            try:
                mzs, intensities = self._read_frame_spectrum(frame_id)
                # Apply intensity threshold filtering if configured
                mzs, intensities = self._apply_intensity_filter(mzs, intensities)
            except Exception as e:
                logger.warning(f"Error reading spectrum for frame {frame_id}: {e}")
                continue
            if mzs.size > 0 and intensities.size > 0:
                yield coords, mzs, intensities

    def _iter_frames(
        self,
    ) -> Generator[Tuple[int, Tuple[int, int, int]], None, None]:
        """The frames to read, in order, each with its normalised coordinates.

        The one frame loop behind both :meth:`iter_spectra` and
        :meth:`iter_mobility_spectra`: the region filter, the source of
        frame ids (``MaldiFrameInfo`` when present, else ``1..N`` -- ids
        are the 1-based ``Frames.Id`` throughout and are never
        renumbered) and the coordinate normalisation live here so the two
        iterations cannot drift apart. A frame without coordinates is
        skipped with a warning. The progress callback fires once per
        frame handed out.
        """
        if self._region_frame_ids is not None:
            frame_ids: Any = sorted(self._region_frame_ids)
            total = len(frame_ids)
        else:
            # Use actual frame IDs from MaldiFrameInfo when available,
            # rather than assuming sequential 1..N. This handles .d files
            # where frame IDs are non-contiguous (e.g., region-split files).
            frame_ids = self._get_maldi_frame_ids()
            if frame_ids is not None:
                total = len(frame_ids)
            else:
                total = self._get_frame_count()
                frame_ids = range(1, total + 1)

        coordinate_offsets = self._get_coordinate_offsets()
        for frame_id in frame_ids:
            coords = self._get_frame_coordinates_cached(frame_id, coordinate_offsets)
            if coords is None:
                logger.warning(f"No coordinates found for frame {frame_id}")
                continue
            yield frame_id, coords
            if self.progress_callback:
                self.progress_callback(frame_id, total)

    # ------------------------------------------------------------------
    # Ion mobility (TDF only)
    #
    # A TDF frame's scans are its mobility dimension: scan number maps
    # onto 1/K0 through the file's TimsCalibration, identically for every
    # frame, with 1/K0 decreasing as the scan number grows. Mobility is a
    # coordinate on a feature, never on a pixel -- nothing here touches
    # coordinates, and iter_spectra keeps yielding the summed spectrum.
    # ------------------------------------------------------------------

    @property
    def has_ion_mobility(self) -> bool:
        """True for TDF (TIMS engaged); a TSF file has no mobility dimension."""
        return self.file_type == "tdf"

    def get_mobility_axis(self) -> Optional[MobilityAxis]:
        """The per-scan 1/K0 axis of a TDF file, or ``None`` for TSF.

        ``values[s]`` is the 1/K0 of scan ``s`` from the SDK's own
        calibration (never a linear model: the mapping is non-linear by
        up to 0.3%), for the longest ramp in the file. The declared
        acquisition range and the ``TimsCalibration`` row travel with it
        for provenance. Read once and cached; the calibration is one row
        per file. Without the SDK (``metadata_only``) the axis is
        described but carries no values.
        """
        if self.file_type != "tdf":
            return None
        if self._mobility_axis is None:
            self._mobility_axis = self._build_mobility_axis()
        return self._mobility_axis

    def _build_mobility_axis(self) -> MobilityAxis:
        n_scans, frame_id = self._mobility_ramp()
        values: Optional[NDArray[np.float64]] = None
        if getattr(self, "sdk", None) is not None and self.handle:
            values = self.sdk.scannum_to_oneoverk0(
                self.handle, frame_id, np.arange(n_scans, dtype=np.float64)
            )
        else:
            logger.info(
                "Mobility axis described without per-scan values: the Bruker "
                "library is not loaded (metadata-only mode)"
            )
        return MobilityAxis(
            kind_accession=INVERSE_REDUCED_MOBILITY_ACCESSION,
            kind_name=MOBILITY_KIND_NAMES[INVERSE_REDUCED_MOBILITY_ACCESSION],
            unit_accession=_ONE_OVER_K0_UNIT_ACCESSION,
            unit_name=_ONE_OVER_K0_UNIT_NAME,
            values=values,
            acq_range=self._one_over_k0_acq_range(),
            calibration=self._tims_calibration(),
            source="bruker_tdf",
        )

    def _mobility_ramp(self) -> Tuple[int, int]:
        """``(longest NumScans in the file, a frame id to calibrate against)``."""
        row = self.conn.execute("SELECT MAX(NumScans), MIN(Id) FROM Frames").fetchone()
        if row is None or row[0] is None:
            raise DataError("The Frames table carries no NumScans; not a TIMS file")
        return int(row[0]), int(row[1])

    def _one_over_k0_acq_range(self) -> Optional[Tuple[float, float]]:
        """The acquired 1/K0 range from ``GlobalMetadata``, when declared."""
        try:
            rows = self.conn.execute(
                "SELECT Key, Value FROM GlobalMetadata WHERE Key IN "
                "('OneOverK0AcqRangeLower', 'OneOverK0AcqRangeUpper')"
            ).fetchall()
            bounds = {key: float(value) for key, value in rows}
        except (sqlite3.OperationalError, TypeError, ValueError) as e:
            logger.debug(f"Could not read the 1/K0 acquisition range: {e}")
            return None
        lower = bounds.get("OneOverK0AcqRangeLower")
        upper = bounds.get("OneOverK0AcqRangeUpper")
        if lower is None or upper is None:
            return None
        return (lower, upper)

    def _tims_calibration(self) -> Optional[Dict[str, Any]]:
        """The ``TimsCalibration`` row as ``{model_type, coefficients}``.

        Provenance only: nothing in Thyra evaluates the model, the SDK
        does. Coefficients keep their column order (``C0``, ``C1``, ...);
        a NULL becomes NaN so positions are preserved.
        """
        try:
            cursor = self.conn.execute("SELECT * FROM TimsCalibration ORDER BY Id")
            row = cursor.fetchone()
        except sqlite3.OperationalError as e:
            logger.debug(f"No TimsCalibration table: {e}")
            return None
        if row is None:
            return None
        columns = [description[0] for description in cursor.description]
        by_name = dict(zip(columns, row))
        coefficients = [
            float("nan") if by_name[name] is None else float(by_name[name])
            for name in columns
            if name.startswith("C") and name[1:].isdigit()
        ]
        calibration: Dict[str, Any] = {"coefficients": coefficients}
        if by_name.get("ModelType") is not None:
            calibration["model_type"] = int(by_name["ModelType"])
        return calibration

    # ------------------------------------------------------------------
    # Fragmentation
    #
    # ``Frames.MsMsType`` says whether a frame fragmented anything: 0 is a
    # survey scan, everything else is MS2 of some flavour. Which table
    # carries the precursor detail depends on the flavour -- PASEF frames
    # (type 8) list one row per isolation window in ``PasefFrameMsMsInfo``,
    # while single-precursor frames (type 2) carry one row per frame in
    # ``FrameMsMsInfo``.
    # ------------------------------------------------------------------

    def get_fragmentation(self) -> Optional[FragmentationSchedule]:
        """The precursor schedule of a TDF acquisition, or ``None``.

        ``None`` for TSF and for any file whose ``Frames`` table has no
        ``MsMsType`` column: that is "cannot tell", not "MS1". Read once
        and cached -- the schedule is a property of the method, and on
        every MALDI file measured so far it is identical at every pixel.
        """
        if self.file_type != "tdf":
            return None
        if not self._fragmentation_read:
            self._fragmentation = self._build_fragmentation()
            self._fragmentation_read = True
        return self._fragmentation

    def _build_fragmentation(self) -> Optional[FragmentationSchedule]:
        try:
            rows = self.conn.execute(
                "SELECT MsMsType, COUNT(*) FROM Frames GROUP BY MsMsType"
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug(f"Frames has no MsMsType column: {e}")
            return None
        counts = {int(kind): int(n) for kind, n in rows if kind is not None}
        if not counts:
            return None

        msms_frames = sum(n for kind, n in counts.items() if kind != 0)
        if msms_frames == 0:
            return FragmentationSchedule(ms_level=1, source="bruker_tdf")

        # A file holding both survey and fragment frames has no single
        # schedule per pixel, whatever the precursor tables say.
        mixed = counts.get(0, 0) > 0
        windows = self._isolation_windows(msms_frames)
        return FragmentationSchedule(
            ms_level=2,
            windows=windows[0],
            constant_across_pixels=windows[1] and not mixed,
            dissociation_accession=(
                COLLISION_INDUCED_DISSOCIATION_ACCESSION if windows[0] else None
            ),
            source="bruker_tdf",
        )

    def _isolation_windows(
        self, msms_frames: int
    ) -> Tuple[Tuple[IsolationWindow, ...], bool]:
        """The distinct isolation windows, and whether every frame has them all.

        Grouped rather than read per frame: on the files measured so far a
        scheduled method repeats the same windows at every pixel, so the
        distinct set *is* the schedule, and its per-window frame count is
        what proves the repetition.
        """
        for query, build in (
            (
                "SELECT IsolationMz, IsolationWidth, CollisionEnergy, "
                "ScanNumBegin, ScanNumEnd, COUNT(DISTINCT Frame) "
                "FROM PasefFrameMsMsInfo GROUP BY IsolationMz, IsolationWidth, "
                "CollisionEnergy, ScanNumBegin, ScanNumEnd ORDER BY IsolationMz",
                lambda row: IsolationWindow.from_full_width(
                    float(row[0]),
                    _optional_float(row[1]),
                    collision_energy=_optional_float(row[2]),
                    scan_begin=_optional_int(row[3]),
                    scan_end=_optional_int(row[4]),
                ),
            ),
            (
                "SELECT TriggerMass, IsolationWidth, CollisionEnergy, "
                "COUNT(DISTINCT Frame) FROM FrameMsMsInfo "
                "GROUP BY TriggerMass, IsolationWidth, CollisionEnergy "
                "ORDER BY TriggerMass",
                lambda row: IsolationWindow.from_full_width(
                    float(row[0]),
                    _optional_float(row[1]),
                    collision_energy=_optional_float(row[2]),
                ),
            ),
        ):
            try:
                rows = self.conn.execute(query).fetchall()
            except sqlite3.OperationalError:
                continue
            rows = [row for row in rows if row[0] is not None]
            if not rows:
                continue
            windows = tuple(build(row) for row in rows)
            everywhere = all(int(row[-1]) == msms_frames for row in rows)
            if not everywhere:
                logger.info(
                    "The isolation windows are not identical at every frame; "
                    "the precursor schedule is recorded as varying."
                )
            return windows, everywhere

        logger.info(
            "Frames are marked MS/MS but neither PasefFrameMsMsInfo nor "
            "FrameMsMsInfo carries a precursor; recording the level only."
        )
        return (), True

    def iter_mobility_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[
            Tuple[int, int, int],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        None,
        None,
    ]:
        """Every ``(m/z, 1/K0, intensity)`` point of each TDF frame, unbinned.

        One ``tims_read_scans_v2`` over the full ramp per frame (the frame
        id passed as-is), ``tims_index_to_mz`` on the frame's unique
        digitizer indices only, and the scan number of each pair turned
        into 1/K0 through :meth:`get_mobility_axis` -- so the mobility
        value of a point is exactly ``values[scan]``. Points come out
        ordered by scan, then by index; the three arrays are parallel.
        Same frames, same order and same coordinates as
        :meth:`iter_spectra`, which keeps yielding the summed spectrum.

        Args:
            batch_size: Ignored, maintained for interface compatibility.

        Raises:
            NotImplementedError: On a TSF file, which has no mobility.
            SDKError: When the Bruker library is not loaded.
        """
        if self.file_type != "tdf":
            raise NotImplementedError(
                "Only a TDF (TIMS engaged) acquisition carries an ion mobility "
                "dimension; this is a TSF file"
            )
        if getattr(self, "sdk", None) is None or not self.handle:
            raise SDKError(
                "Reading mobility spectra needs the Bruker library; the reader "
                "was opened in metadata-only mode"
            )
        values, n_axis = self._mobility_values()

        for frame_id, coords in self._iter_frames():
            try:
                frame = TdfFrameScans(self, frame_id, coords)
                points = self._mobility_points_from(frame, values, n_axis)
            except Exception as e:
                logger.warning(
                    f"Error reading mobility scans for frame {frame_id}: {e}"
                )
                continue
            if points is not None:
                yield coords, points[0], points[1], points[2]

    def _mobility_values(self) -> Tuple[NDArray[np.float64], int]:
        """The per-scan 1/K0 values and their count, for the point cloud."""
        axis = self.get_mobility_axis()
        if axis is None or axis.values is None:
            raise SDKError("The per-scan 1/K0 axis is not available")
        return axis.values, int(axis.values.size)

    def _indexed_mobility_points_from(
        self,
        frame: "TdfFrameScans",
        values: NDArray[np.float64],
        n_axis: int,
    ) -> Optional[
        Tuple[
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ]:
        """One frame's points as ``(unique m/z, index of each point, 1/K0, intensity)``.

        The one derivation behind :meth:`iter_mobility_spectra`, the
        frame record's ``mobility_points`` and its indexed view:
        ``tims_index_to_mz`` on the frame's unique indices only, the scan
        number of each pair turned into ``values[scan]``. The m/z of the
        points is left as ``unique_mz[inverse]`` rather than expanded,
        because a consumer that maps m/z onto an axis can map the unique
        values once and gather -- a TDF frame carries about three fifths
        as many unique indices as points. ``None`` when the frame holds
        no points (or none above the intensity threshold).
        """
        if frame.indices.size == 0:
            return None
        scans = frame.scans
        if int(scans.max()) >= n_axis and not self._mobility_scan_overflow_warned:
            self._mobility_scan_overflow_warned = True
            logger.warning(
                "Frame %d has scan numbers beyond the %d-scan mobility "
                "axis; they are clipped to the last scan's 1/K0",
                frame.frame_id,
                n_axis,
            )
        inverse = frame.inverse
        mobility = np.take(values, scans, mode="clip")
        intensities = frame.intensities.astype(np.float64)
        if self._intensity_threshold is not None:
            keep = intensities >= self._intensity_threshold
            inverse, mobility, intensities = (
                inverse[keep],
                mobility[keep],
                intensities[keep],
            )
        if inverse.size == 0:
            return None
        return frame.unique_mz, inverse, mobility, intensities

    def _mobility_points_from(
        self,
        frame: "TdfFrameScans",
        values: NDArray[np.float64],
        n_axis: int,
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
        """One frame's ``(m/z, 1/K0, intensity)`` points, the indexed view expanded.

        ``None`` when the frame holds no points (or none above the
        intensity threshold).
        """
        indexed = self._indexed_mobility_points_from(frame, values, n_axis)
        if indexed is None:
            return None
        unique_mz, inverse, mobility, intensities = indexed
        return unique_mz[inverse], mobility, intensities

    def _precursor_scan_map(
        self, windows: Tuple[IsolationWindow, ...]
    ) -> NDArray[np.int64]:
        """``scan -> window index``, ``-1`` where no window isolated anything.

        The whole demultiplexer, precomputed once: the windows own
        disjoint scan ranges, so which precursor a point belongs to is a
        lookup on its scan number rather than a search. Sized to cover
        the longest ramp in the file as well as the last window, so
        ``np.take(..., mode="clip")`` cannot fold a real scan onto a
        window it does not belong to.
        """
        n_scans = max(
            self._mobility_ramp()[0],
            max(int(w.scan_end or 0) for w in windows),
        )
        scan_map = np.full(n_scans, -1, dtype=np.int64)
        for index, window in enumerate(windows):
            scan_map[int(window.scan_begin) : int(window.scan_end)] = index
        return scan_map

    def iter_precursor_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[
            Tuple[int, int, int],
            int,
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        None,
        None,
    ]:
        """One fragment spectrum per (pixel, precursor), not per pixel.

        A PASEF frame isolates several precursors, each in its own slice
        of the mobility ramp, and :meth:`iter_spectra` sums the whole
        frame into one spectrum -- so that spectrum holds fragments of
        every precursor at once. This yields them apart: for each frame,
        one ``tims_read_scans_v2`` over the full ramp (the frame id
        passed as-is), then the points of each isolation window picked
        out by ``ScanNumBegin <= scan < ScanNumEnd`` and summed per
        digitizer index, exactly as the summed spectrum sums the whole
        ramp.

        The split is a filter, not an estimate: the windows are disjoint,
        so every point belongs to exactly one precursor or to none, and
        nothing is shared out or apportioned. A window with no points in
        a frame is not yielded.

        Yields ``((x, y, z), window_index, mzs, intensities)``, where
        ``window_index`` is the position of the precursor in
        ``get_fragmentation().windows`` and ``mzs`` is ascending.

        Args:
            batch_size: Ignored, maintained for interface compatibility.

        Raises:
            NotImplementedError: On a TSF file, or when the acquisition
                has no mobility-resolved precursor schedule to split on.
            SDKError: When the Bruker library is not loaded.
        """
        if self.file_type != "tdf":
            raise NotImplementedError(
                "Only a TDF (TIMS engaged) acquisition separates its "
                "precursors by mobility; this is a TSF file"
            )
        schedule = self.get_fragmentation()
        if schedule is None or not schedule.windows:
            raise NotImplementedError(
                "This acquisition reports no isolation windows; there is "
                "nothing to demultiplex"
            )
        if not all(w.is_mobility_resolved for w in schedule.windows):
            raise NotImplementedError(
                "The isolation windows carry no mobility scan range, so the "
                "precursors cannot be separated by scan number"
            )
        if getattr(self, "sdk", None) is None or not self.handle:
            raise SDKError(
                "Splitting a frame by precursor needs the Bruker library; "
                "the reader was opened in metadata-only mode"
            )

        scan_map = self._precursor_scan_map(schedule.windows)
        n_windows = len(schedule.windows)
        for frame_id, coords in self._iter_frames():
            try:
                frame = TdfFrameScans(self, frame_id, coords)
            except Exception as e:
                logger.warning(
                    f"Error reading scans for frame {frame_id}: {e}",
                )
                continue
            for window_index, mzs, intensities in self._precursor_spectra_from(
                frame, scan_map, n_windows
            ):
                yield coords, window_index, mzs, intensities

    def _precursor_context(self) -> Optional[Tuple[NDArray[np.int64], int]]:
        """``(scan -> window map, window count)``, or ``None`` when not separable.

        The frame record's precursor view uses this where
        :meth:`iter_precursor_spectra` would have raised: a record of a
        frame that cannot be split simply has no precursor spectra.
        """
        if self.file_type != "tdf":
            return None
        schedule = self.get_fragmentation()
        if schedule is None or not schedule.windows:
            return None
        if not all(w.is_mobility_resolved for w in schedule.windows):
            return None
        if self._precursor_scan_map_cache is None:
            self._precursor_scan_map_cache = (
                self._precursor_scan_map(schedule.windows),
                len(schedule.windows),
            )
        return self._precursor_scan_map_cache

    def _precursor_spectra_from(
        self,
        frame: "TdfFrameScans",
        scan_map: NDArray[np.int64],
        n_windows: int,
    ) -> List[Tuple[int, NDArray[np.float64], NDArray[np.float64]]]:
        """One frame's fragment spectra, one per precursor with points.

        The one derivation behind :meth:`iter_precursor_spectra` and the
        frame record's ``precursor_spectra``. Sums over each window's
        scans per digitizer index, every window at once: the mobility
        dimension is collapsed inside the window, the way the summed
        spectrum collapses it over the whole ramp. One bincount over
        (window, index) keys instead of one pass over the frame per
        window.
        """
        if frame.indices.size == 0:
            return []
        unique_mz = frame.unique_mz
        inverse = frame.inverse
        intensities = frame.intensities.astype(np.float64)
        window_of_point = np.take(scan_map, frame.scans, mode="clip")
        isolated = window_of_point >= 0
        n_unique = int(frame.unique_indices.size)
        sums = np.bincount(
            window_of_point[isolated] * n_unique + inverse[isolated],
            weights=intensities[isolated],
            minlength=n_windows * n_unique,
        ).reshape(n_windows, n_unique)
        out: List[Tuple[int, NDArray[np.float64], NDArray[np.float64]]] = []
        for window_index in np.flatnonzero(sums.any(axis=1)).tolist():
            mzs, window_intensities = self._apply_intensity_filter(
                unique_mz, sums[window_index]
            )
            nonzero = np.flatnonzero(window_intensities)
            if nonzero.size:
                out.append((window_index, mzs[nonzero], window_intensities[nonzero]))
        return out

    # ------------------------------------------------------------------
    # One read per frame for every table (design decision D5)
    # ------------------------------------------------------------------

    @property
    def has_frame_scans(self) -> bool:
        """True for a TDF file read through the library: one raw read serves every table."""
        return (
            self.file_type == "tdf"
            and getattr(self, "sdk", None) is not None
            and bool(self.handle)
        )

    def iter_frame_scans(
        self, batch_size: Optional[int] = None
    ) -> Generator["TdfFrameScans", None, None]:
        """Every frame as a :class:`TdfFrameScans`: one ``tims_read_scans_v2`` each.

        Same frames, same order and same coordinates as
        :meth:`iter_spectra`; a frame whose read fails is logged and
        skipped, as the iterators skip it. See
        :mod:`thyra.core.frames`.
        """
        if not self.has_frame_scans:
            raise NotImplementedError(
                "Frame records need a TDF file read through the Bruker library"
            )
        for frame_id, coords in self._iter_frames():
            try:
                frame = TdfFrameScans(self, frame_id, coords)
            except Exception as e:
                logger.warning(f"Error reading scans for frame {frame_id}: {e}")
                continue
            yield frame

    def _get_maldi_frame_ids(self) -> Optional[List[int]]:
        """Get sorted frame IDs from MaldiFrameInfo table.

        Returns actual frame IDs rather than assuming sequential 1..N,
        which is necessary for .d files with non-contiguous frame IDs
        (e.g., files split by region from a multi-region acquisition).

        Returns:
            Sorted list of frame IDs, or None if MaldiFrameInfo is unavailable.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT Frame FROM MaldiFrameInfo ORDER BY Frame")
            rows = cursor.fetchall()
            if rows:
                return [int(row[0]) for row in rows]
            return None
        except sqlite3.OperationalError:
            return None

    def _get_frame_count(self) -> int:
        """Get the total number of frames (respects region filtering)."""
        if self._frame_count is None:
            if self._region_frame_ids is not None:
                self._frame_count = len(self._region_frame_ids)
            else:
                self._frame_count = _get_frame_count(self.db_path)

        return self._frame_count

    def _get_coordinate_offsets(self) -> Optional[Tuple[int, int, int]]:
        """Get coordinate offsets from metadata for normalization."""
        if self._coordinate_offsets is None:
            essential_metadata = self.get_essential_metadata()
            self._coordinate_offsets = essential_metadata.coordinate_offsets

        return self._coordinate_offsets

    def _get_frame_coordinates_cached(
        self,
        frame_id: int,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int]]:
        """Get normalized coordinates for a specific frame using persistent connection.

        This avoids opening new SQLite connections for every frame.

        Args:
            frame_id: Frame ID to look up
            coordinate_offsets: Optional coordinate offsets for normalization

        Returns:
            Tuple of normalized (x, y, z) coordinates (0-based), or None if not
            found
        """
        try:
            cursor = self.conn.cursor()

            # Check if this is MALDI data
            try:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos FROM MaldiFrameInfo WHERE "
                    "Frame = ?",
                    (frame_id,),
                )
                result = cursor.fetchone()
                if result:
                    x, y = result
                    # Apply coordinate offsets if provided (Bruker-specific
                    # normalization)
                    if coordinate_offsets:
                        offset_x, offset_y, offset_z = coordinate_offsets
                        return (int(x) - offset_x, int(y) - offset_y, 0)
                    else:
                        return (int(x), int(y), 0)
            except sqlite3.OperationalError:
                # No MALDI table, use generated coordinates
                pass

            # For non-MALDI data, generate coordinates (simple sequential
            # mapping)
            return (frame_id - 1, 0, 0)

        except Exception as e:
            logger.warning(f"Error getting coordinates for frame {frame_id}: {e}")
            return None

    def get_frame_id_by_coordinates(self, x: int, y: int, z: int = 0) -> Optional[int]:
        """Get frame ID for given coordinates.

        Args:
            x: X coordinate (0-based)
            y: Y coordinate (0-based)
            z: Z coordinate (0-based, default 0 for 2D data)

        Returns:
            Frame ID if found, None otherwise
        """
        try:
            # Adjust coordinates by offsets if needed
            offsets = self._get_coordinate_offsets()
            if offsets:
                x_offset, y_offset, z_offset = offsets
                x += x_offset
                y += y_offset
                z += z_offset

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT Frame FROM MaldiFrameInfo WHERE "
                    "XIndexPos = ? AND YIndexPos = ?",
                    (x, y),
                )
                result = cursor.fetchone()
                if result:
                    return int(result[0])
                return None

        except Exception as e:
            logger.warning(
                f"Error getting frame ID for coordinates ({x}, {y}, {z}): {e}"
            )
            return None

    def get_spectrum_by_coordinates(
        self, x: int, y: int, z: int = 0
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Get spectrum data for given coordinates.

        Args:
            x: X coordinate (0-based)
            y: Y coordinate (0-based)
            z: Z coordinate (0-based, default 0 for 2D data)

        Returns:
            Tuple of (mz_array, intensity_array) if found, None otherwise
        """
        frame_id = self.get_frame_id_by_coordinates(x, y, z)
        if frame_id is None:
            return None

        try:
            mzs, intensities = self._read_frame_spectrum(frame_id)

            # Apply intensity threshold filtering if configured
            mzs, intensities = self._apply_intensity_filter(mzs, intensities)

            if mzs.size > 0 and intensities.size > 0:
                return mzs, intensities
            return None

        except Exception as e:
            logger.warning(f"Error reading spectrum for frame {frame_id}: {e}")
            return None

    def _read_frame_spectrum(
        self, frame_id: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Read one frame through the SDK with the hints it needs.

        The frame id is the 1-based ``Frames.Id`` and is handed to the SDK
        unchanged. ``NumPeaks`` sizes the buffer; for TDF the frame's
        ``NumScans`` is required so the whole mobility ramp is read.
        """
        buffer_size_hint = self._num_peaks_cache.get(frame_id)
        num_scans = self._frame_num_scans(frame_id) if self.file_type == "tdf" else None
        mzs, intensities = self.sdk.read_spectrum(
            self.handle,
            frame_id,
            buffer_size_hint=buffer_size_hint,
            num_scans=num_scans,
        )
        return mzs, intensities

    def _frame_num_scans(self, frame_id: int) -> int:
        """``Frames.NumScans`` for one frame, from the preload or the database."""
        cached = self._num_scans_cache.get(frame_id)
        if cached is not None:
            return cached
        row = self.conn.execute(
            "SELECT NumScans FROM Frames WHERE Id = ?", (frame_id,)
        ).fetchone()
        if row is None or row[0] is None:
            raise DataError(
                f"Frame {frame_id} has no NumScans entry in the Frames table"
            )
        num_scans = int(row[0])
        self._num_scans_cache[frame_id] = num_scans
        return num_scans

    def _preload_frame_num_peaks(self) -> Dict[int, int]:
        """Preload per-frame NumPeaks (and NumScans for TDF) at initialization.

        NumPeaks gives the SDK an exact buffer size, which avoids the
        retry loop that used to pin a core. For TDF it counts the
        ``(index, scan)`` pairs across the whole mobility ramp -- on real
        imaging runs routinely above 65,535 per frame -- and is only an
        upper bound on the summed spectrum's length, so it must not be
        treated as a peak count (see :meth:`get_peak_counts_per_pixel`).

        For TDF the same query also fills ``_num_scans_cache``: every
        frame read needs its ``NumScans`` to cover the full ramp.

        When region filtering is active, only frames in the selected
        region are cached.

        Returns:
            Dictionary mapping frame_id -> NumPeaks for frames with peaks.
        """
        is_tdf = self.file_type == "tdf"
        query = (
            "SELECT Id, NumPeaks, NumScans FROM Frames ORDER BY Id"
            if is_tdf
            else "SELECT Id, NumPeaks, NULL FROM Frames ORDER BY Id"
        )
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(query)

                num_peaks_cache: Dict[int, int] = {}
                empty_count = 0

                for frame_id, num_peaks, num_scans in cursor.fetchall():
                    # Skip frames not in selected region
                    if (
                        self._region_frame_ids is not None
                        and frame_id not in self._region_frame_ids
                    ):
                        continue

                    if is_tdf and num_scans is not None:
                        self._num_scans_cache[int(frame_id)] = int(num_scans)

                    if num_peaks is not None and num_peaks > 0:
                        num_peaks_cache[int(frame_id)] = int(num_peaks)
                    else:
                        empty_count += 1

                if empty_count:
                    logger.debug(
                        f"{empty_count} frames report no peaks; they are read "
                        "without a buffer hint"
                    )
                logger.debug(
                    f"Cached NumPeaks for {len(num_peaks_cache)} frames"
                    + (
                        f" and NumScans for {len(self._num_scans_cache)}"
                        if is_tdf
                        else ""
                    )
                )
                return num_peaks_cache

        except Exception as e:
            logger.warning(f"Failed to preload frame info: {e}")
            logger.info("Will use fallback retry logic for spectrum reading")
            return {}  # Empty cache triggers fallback behavior

    def close(self) -> None:
        """Close all resources and connections.

        This method is idempotent - safe to call multiple times.
        """
        # Skip if already closed
        if getattr(self, "_closed", False):
            logger.debug("Resources already closed, skipping")
            return

        logger.debug("Closing Bruker reader")

        try:
            # Close SDK handle
            if hasattr(self, "handle") and self.handle:
                self.sdk.close_file(self.handle)
                self.handle = None

            # Close database connection
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
                self.conn = None

            # Mark as closed
            self._closed = True

            logger.info("Successfully closed all resources")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Even if there's an error, mark as closed to prevent repeated attempts
            self._closed = True

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get spatial dimensions (pixel grid) from metadata extractor.

        Note: This is the spatial pixel grid, not the mass axis
        dimensions.
        Mass axis interpolation to common m/z values is handled during
        conversion.

        Returns:
            Tuple of (x_pixels, y_pixels, z_pixels) spatial dimensions
        """
        essential_metadata = self.get_essential_metadata()
        return essential_metadata.dimensions

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Get mass range from metadata extractor.

        Note: This is the acquisition mass range, not the final
        interpolated axis.
        The actual common mass axis for interpolation is built from all
        unique m/z values.

        Returns:
            Tuple of (min_mass, max_mass) in m/z units
        """
        essential_metadata = self.get_essential_metadata()
        return essential_metadata.mass_range

    def __repr__(self) -> str:
        """String representation of the reader."""
        return (
            f"BrukerReader(path={self.data_path}, "
            f"type={self.file_type.upper()}, "
            f"frames={self._get_frame_count()})"
        )

    @property
    def mis_metadata(self) -> Dict[str, Any]:
        """Get parsed .mis metadata for optical alignment."""
        return self._mis_metadata

    def _parse_mis_alignment(self) -> Dict[str, Any]:
        """Parse .mis file for optical alignment if available.

        Uses BrukerFolderStructure to find the .mis file in the parent
        folder hierarchy, then parses it for area definitions and
        teaching points.

        Returns:
            Dictionary of parsed .mis metadata, empty if no .mis found

        Raises:
            ConversionRefused: If the .mis found is a document defusedxml
                refuses. Deliberately not caught: this runs in ``__init__``
                before ``_select_region``, and the areas it would have
                filled are what resolves ``--region <name>``, so swallowing
                it turned a refused file into "no such region" naming a
                region list that was empty for a reason nobody was told.
        """
        try:
            mis_path = self.get_teaching_points_file()
        except (ValueError, OSError):
            # Folder structure analysis can fail for non-standard paths.
            # Scoped to this one call on purpose: ConversionRefused is a
            # ValueError, and parse_mis_file below must not land here.
            return {}

        if mis_path is None:
            return {}

        logger.info(f"Found .mis file for optical alignment: {mis_path.name}")
        metadata = parse_mis_file(mis_path)

        if metadata.get("areas"):
            logger.info(
                f"Parsed {len(metadata['areas'])} area definitions " f"from .mis file"
            )
        return metadata

    def _build_positions_from_db(self) -> List[Dict[str, Any]]:
        """Build position list from MaldiFrameInfo for alignment.

        Creates position dictionaries compatible with the alignment module
        by querying raster coordinates and region numbers from the database.

        For single-region .d files on multi-region slides, the RegionNumber
        column is often 0 regardless of which area on the slide was acquired.
        In this case, we parse the SpotName column (format R{nn}X{nnn}Y{nnn})
        to recover the true region number from the original acquisition.

        Returns:
            List of position dicts with region, raster_x, raster_y keys.
            Empty list if no .mis areas are available.
        """
        if not self._mis_metadata.get("areas"):
            return []

        n_areas = len(self._mis_metadata["areas"])
        positions = self._query_maldi_frame_positions()

        if positions:
            self._log_position_summary(positions, n_areas)

        return positions

    def _query_maldi_frame_positions(self) -> List[Dict[str, Any]]:
        """Query MaldiFrameInfo for raster positions with SpotName fallback."""
        positions: List[Dict[str, Any]] = []
        spot_region_re = re.compile(r"^R(\d+)")

        try:
            cursor = self.conn.cursor()
            use_spotname = False
            try:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos, RegionNumber, SpotName "
                    "FROM MaldiFrameInfo"
                )
                use_spotname = True
            except sqlite3.OperationalError:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos, RegionNumber " "FROM MaldiFrameInfo"
                )

            for row in cursor.fetchall():
                x, y, region = row[0], row[1], row[2]
                if use_spotname and row[3]:
                    m = spot_region_re.match(str(row[3]))
                    if m and int(m.group(1)) != int(region):
                        region = int(m.group(1))
                positions.append(
                    {"region": int(region), "raster_x": int(x), "raster_y": int(y)}
                )
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not query positions for alignment: {e}")

        return positions

    def _log_position_summary(
        self, positions: List[Dict[str, Any]], n_areas: int
    ) -> None:
        """Log summary of queried positions and region-to-area mapping."""
        unique_regions = sorted(set(p["region"] for p in positions))
        logger.info(
            f"Built {len(positions)} positions from MaldiFrameInfo "
            f"for optical alignment "
            f"(regions: {unique_regions}, areas in .mis: {n_areas})"
        )
        if len(unique_regions) == 1 and n_areas > 1:
            rid = unique_regions[0]
            if rid < n_areas:
                area_name = self._mis_metadata["areas"][rid].get("name", f"Area {rid}")
                logger.info(f"Single region {rid} will align to Area '{area_name}'")
            else:
                logger.warning(
                    f"Region {rid} exceeds number of areas "
                    f"({n_areas}). Alignment may be incorrect."
                )

    def _build_header_alignment(self) -> Dict[str, Any]:
        """Build header dict with first_raster offsets for alignment.

        The first_raster_x/y values must match the coordinate offsets used
        by BrukerMetadataExtractor so that transform_point() correctly
        de-normalizes 0-based raster coordinates back to originals.

        When a specific region is selected, uses per-region minimums from
        MaldiFrameInfo. Otherwise uses global ImagingArea from GlobalMetadata.

        Returns:
            Dictionary with first_raster_x and first_raster_y keys.
            Empty dict if no .mis areas are available.
        """
        if not self._mis_metadata.get("areas"):
            return {}

        try:
            cursor = self.conn.cursor()

            if self._selected_region is not None:
                # Per-region: use actual min from the selected region's frames
                # This matches BrukerMetadataExtractor's per-region offsets
                cursor.execute(
                    "SELECT MIN(XIndexPos), MIN(YIndexPos) "
                    "FROM MaldiFrameInfo WHERE RegionNumber = ?",
                    (self._selected_region,),
                )
                result = cursor.fetchone()
                if result and result[0] is not None and result[1] is not None:
                    header = {
                        "first_raster_x": int(result[0]),
                        "first_raster_y": int(result[1]),
                    }
                else:
                    return {}
            else:
                # Global: use ImagingArea from GlobalMetadata
                cursor.execute(
                    "SELECT Value FROM GlobalMetadata "
                    "WHERE Key = 'ImagingAreaMinXIndexPos'"
                )
                result_x = cursor.fetchone()
                cursor.execute(
                    "SELECT Value FROM GlobalMetadata "
                    "WHERE Key = 'ImagingAreaMinYIndexPos'"
                )
                result_y = cursor.fetchone()

                if result_x and result_y:
                    header = {
                        "first_raster_x": int(float(result_x[0])),
                        "first_raster_y": int(float(result_y[0])),
                    }
                else:
                    return {}

            logger.info(
                f"Alignment offsets: first_raster=("
                f"{header['first_raster_x']}, "
                f"{header['first_raster_y']})"
            )
            return header
        except (sqlite3.OperationalError, TypeError) as e:
            logger.warning(f"Could not get alignment offsets: {e}")

        return {}

    @property
    def n_spectra(self) -> int:
        """Return the total number of spectra in the dataset.

        Returns:
            Total number of frames (efficient implementation using cached
            frame count)
        """
        return self._get_frame_count()

    def get_total_peak_count(self) -> int:
        """Get total number of peaks across all spectra from NumPeaks cache.

        This is very fast as NumPeaks data is cached from the database at
        initialization.

        Returns:
            Total number of peaks across all spectra
        """
        if not self._num_peaks_cache:
            logger.warning("NumPeaks cache not available, cannot get exact count")
            return 0

        total = sum(self._num_peaks_cache.values())
        if self.file_type == "tdf":
            logger.info(
                f"Total (index, scan) pairs from NumPeaks cache: {total:,}; an "
                "upper bound on the summed spectra, since TDF NumPeaks counts "
                "every mobility scan separately"
            )
        else:
            logger.info(f"Total peak count from NumPeaks cache: {total:,}")
        return total

    def get_peak_counts_per_pixel(self) -> Optional[np.ndarray]:
        """Get per-pixel peak counts for CSR indptr construction.

        Converts the frame-indexed NumPeaks cache to pixel-indexed array
        using coordinate mapping. When region filtering is active, only
        includes frames from the selected region.

        Returns:
            Array of size n_pixels where arr[pixel_idx] = peak_count.
            pixel_idx = z * (n_x * n_y) + y * n_x + x
            Returns None if NumPeaks cache not available, and always for
            TDF: there ``NumPeaks`` counts ``(index, scan)`` pairs across
            the mobility ramp, which exceeds the summed spectrum's length
            by a factor the database does not record, so the streaming
            converter has to measure instead.
        """
        if self.file_type == "tdf":
            logger.debug(
                "TDF NumPeaks counts mobility scans separately; per-pixel peak "
                "counts are left to the converter to measure"
            )
            return None
        if not self._num_peaks_cache:
            logger.warning("NumPeaks cache not available")
            return None

        # Get dimensions and coordinate offsets
        metadata = self.get_essential_metadata()
        n_x, n_y, n_z = metadata.dimensions
        n_pixels = n_x * n_y * n_z
        coordinate_offsets = metadata.coordinate_offsets

        # Create output array
        peak_counts = np.zeros(n_pixels, dtype=np.int32)

        # Map frame_id -> pixel_idx using coordinate lookup
        cursor = self.conn.cursor()
        try:
            if self._selected_region is not None:
                cursor.execute(
                    "SELECT Frame, XIndexPos, YIndexPos "
                    "FROM MaldiFrameInfo "
                    "WHERE RegionNumber = ?",
                    (self._selected_region,),
                )
            else:
                cursor.execute(
                    "SELECT Frame, XIndexPos, YIndexPos " "FROM MaldiFrameInfo"
                )
            for frame_id, x, y in cursor.fetchall():
                if frame_id not in self._num_peaks_cache:
                    continue

                # Apply coordinate offsets (normalize to 0-based)
                if coordinate_offsets:
                    x = int(x) - coordinate_offsets[0]
                    y = int(y) - coordinate_offsets[1]
                else:
                    x, y = int(x), int(y)

                # Calculate pixel index
                z = 0  # Bruker MSI is typically 2D
                pixel_idx = z * (n_x * n_y) + y * n_x + x

                if 0 <= pixel_idx < n_pixels:
                    peak_counts[pixel_idx] = self._num_peaks_cache[frame_id]

        except Exception as e:
            logger.warning(f"Error mapping peak counts to pixels: {e}")
            return None

        logger.info(f"Mapped peak counts for {n_pixels:,} pixels")
        return peak_counts

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception as e:
            logger.debug(
                f"Error during cleanup in destructor: {e}"
            )  # Log but don't raise during destruction
