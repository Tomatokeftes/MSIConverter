"""Bruker reader implementation combining best features from all implementations.

This module provides a high-performance, memory-efficient reader for Bruker TSF/TDF
data formats with lazy loading, intelligent caching, and comprehensive error handling.
"""

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Dict, Generator, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# Set OpenMP thread limit before any SDK imports to control Bruker DLL threading
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

from ...core.base_extractor import MetadataExtractor
from ...core.base_reader import BaseMSIReader
from ...core.registry import register_reader
from ...metadata.extractors.bruker_extractor import BrukerMetadataExtractor
from ...utils.bruker_exceptions import DataError, FileFormatError, SDKError
from .sdk.dll_manager import DLLManager
from .sdk.sdk_functions import SDKFunctions

logger = logging.getLogger(__name__)


def build_raw_mass_axis(
    spectra_iterator: Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> NDArray[np.float64]:
    """Build raw mass axis from spectra iterator.

    Raw Mass axis in case the user wants the full data. Not recommended for normal use.
    Future interpolation module will create optimized mass axis using min/max mass + bin width.

    Args:
        spectra_iterator: Iterator yielding (coords, mzs, intensities) tuples
        progress_callback: Optional callback for progress updates

    Returns:
        numpy array of unique m/z values in ascending order
    """
    unique_mzs = set()
    count = 0

    for coords, mzs, intensities in spectra_iterator:
        if mzs.size > 0:
            unique_mzs.update(mzs)
        count += 1
        if progress_callback and count % 100 == 0:
            progress_callback(count)

    return np.array(sorted(unique_mzs))


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
        Tuple of normalized (x, y, z) coordinates (0-based), or None if not found
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()

            # Check if this is MALDI data
            try:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos FROM MaldiFrameInfo WHERE Frame = ?",
                    (frame_id,),
                )
                result = cursor.fetchone()
                if result:
                    x, y = result
                    # Apply coordinate offsets if provided (Bruker-specific normalization)
                    if coordinate_offsets:
                        offset_x, offset_y, offset_z = coordinate_offsets
                        return (int(x) - offset_x, int(y) - offset_y, 0)
                    else:
                        return (int(x), int(y), 0)
            except sqlite3.OperationalError:
                # No MALDI table, use generated coordinates
                pass

            # For non-MALDI data, generate coordinates (simple sequential mapping)
            return (frame_id - 1, 0, 0)

    except Exception as e:
        logger.warning(f"Error getting coordinates for frame {frame_id}: {e}")
        return None


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
            return cursor.fetchone()[0]
    except Exception as e:
        logger.error(f"Error getting frame count: {e}")
        return 0


@register_reader("bruker")
class BrukerReader(BaseMSIReader):
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
        use_recalibrated_state: bool = False,
        cache_coordinates: bool = True,
        memory_limit_gb: Optional[float] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        """Initialize the Bruker reader.

        Args:
            data_path: Path to Bruker .d directory
            use_recalibrated_state: Whether to use recalibrated data
            cache_coordinates: Ignored, maintained for compatibility
            memory_limit_gb: Ignored, maintained for compatibility
            batch_size: Ignored, maintained for compatibility
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments
        """
        super().__init__(data_path, **kwargs)
        self.use_recalibrated_state = use_recalibrated_state
        self.progress_callback = progress_callback

        # Validate and setup paths
        self._validate_data_path()
        self._detect_file_type()

        # Initialize components
        self._setup_components(cache_coordinates, memory_limit_gb, batch_size)

        # Initialize SDK and thread-local storage
        self._initialize_sdk()
        self._initialize_database()

        # Cached properties (lazy loaded)
        self._common_mass_axis: Optional[np.ndarray] = None
        self._frame_count: Optional[int] = None
        self._coordinate_offsets: Optional[Tuple[int, int, int]] = None

        # Preload NumPeaks cache for buffer size optimization
        self._num_peaks_cache: Dict[int, int] = self._preload_frame_num_peaks()

        cache_status = (
            f"with {len(self._num_peaks_cache)} NumPeaks cached"
            if self._num_peaks_cache
            else "with fallback spectrum reading"
        )
        logger.info(
            f"Initialized BrukerReader for {self.file_type.upper()} data at "
            f"{data_path} ({cache_status})"
        )

    def _validate_data_path(self) -> None:
        """Validate the data path and check for required files."""
        if not self.data_path.exists():
            raise FileFormatError(f"Data path does not exist: {self.data_path}")

        if not self.data_path.is_dir():
            raise FileFormatError(f"Data path must be a directory: {self.data_path}")

        if not self.data_path.suffix == ".d":
            raise FileFormatError(f"Expected .d directory, got: {self.data_path}")

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

    def _setup_components(
        self,
        cache_coordinates: bool,
        memory_limit_gb: Optional[float],
        batch_size: Optional[int],
    ) -> None:
        """Setup utility components (now minimal)."""
        # No more coordinate cache - using direct database access
        pass

    def _initialize_sdk(self) -> None:
        """Initialize the Bruker SDK with error handling."""
        try:
            # Initialize DLL manager
            self.dll_manager = DLLManager(
                data_directory=self.data_path, force_reload=False
            )

            # Initialize SDK functions
            self.sdk = SDKFunctions(self.dll_manager, self.file_type)

            # Open the data file
            self.handle = self.sdk.open_file(
                str(self.data_path), self.use_recalibrated_state
            )

            logger.info(f"Successfully initialized {self.file_type.upper()} SDK")

        except Exception as e:
            logger.error(f"Failed to initialize SDK: {e}")
            raise SDKError(f"Failed to initialize Bruker SDK: {e}") from e

    def _initialize_database(self) -> None:
        """Initialize thread-local database storage."""
        try:
            # Initialize thread-local storage for SQLite connections
            self._local = threading.local()

            # Test that we can connect to the database
            test_conn = sqlite3.connect(str(self.db_path))
            test_conn.close()

            logger.debug("Initialized thread-local database storage")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DataError(f"Failed to open database: {e}") from e

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(str(self.db_path))

            # Apply SQLite optimizations to this connection
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection.execute("PRAGMA cache_size = 10000")
            self._local.connection.execute("PRAGMA temp_store = MEMORY")

        return self._local.connection

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create Bruker metadata extractor."""
        # Create a fresh connection for metadata extraction
        conn = sqlite3.connect(str(self.db_path))
        return BrukerMetadataExtractor(conn, self.data_path)

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Return the common mass axis composed of all unique m/z values.

        Returns:
            Array of unique m/z values in ascending order
        """
        if self._common_mass_axis is None:
            self._common_mass_axis = self._build_common_mass_axis()

        return self._common_mass_axis

    def _build_common_mass_axis(self) -> NDArray[np.float64]:
        """Build the common mass axis."""
        logger.info("Building raw mass axis")

        # Create iterator for mass axis building
        def mz_iterator():
            for coords, mzs, intensities in self._iter_spectra_raw():
                yield coords, mzs, intensities

        # Build raw mass axis using simplified function
        mass_axis = build_raw_mass_axis(mz_iterator(), self.progress_callback)

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
        """Raw spectrum iteration without batching."""
        frame_count = self._get_frame_count()
        coordinate_offsets = self._get_coordinate_offsets()

        # Setup progress tracking
        with tqdm(
            total=frame_count,
            desc="Reading spectra",
            unit="spectrum",
            disable=True,  # Disable to avoid double progress with converter
        ) as pbar:
            for frame_id in range(1, frame_count + 1):
                try:
                    # Get normalized coordinates using persistent connection
                    coords = self._get_frame_coordinates_cached(
                        frame_id, coordinate_offsets
                    )
                    if coords is None:
                        logger.warning(f"No coordinates found for frame {frame_id}")
                        pbar.update(1)
                        continue

                    # OPTIMIZATION: Get buffer size hint from NumPeaks cache
                    buffer_size_hint = self._num_peaks_cache.get(frame_id)

                    # Read spectrum with optimization (or fallback if no hint)
                    mzs, intensities = self.sdk.read_spectrum(
                        self.handle,
                        frame_id,
                        buffer_size_hint=buffer_size_hint,
                    )

                    if mzs.size > 0 and intensities.size > 0:
                        yield coords, mzs, intensities

                    pbar.update(1)

                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(frame_id, frame_count)

                except Exception as e:
                    logger.warning(f"Error reading spectrum for frame {frame_id}: {e}")
                    pbar.update(1)
                    continue

    def _get_frame_count(self) -> int:
        """Get the total number of frames."""
        if self._frame_count is None:
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
            Tuple of normalized (x, y, z) coordinates (0-based), or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if this is MALDI data
            try:
                cursor.execute(
                    "SELECT XIndexPos, YIndexPos FROM MaldiFrameInfo WHERE Frame = ?",
                    (frame_id,),
                )
                result = cursor.fetchone()
                if result:
                    x, y = result
                    # Apply coordinate offsets if provided (Bruker-specific normalization)
                    if coordinate_offsets:
                        offset_x, offset_y, offset_z = coordinate_offsets
                        return (int(x) - offset_x, int(y) - offset_y, 0)
                    else:
                        return (int(x), int(y), 0)
            except sqlite3.OperationalError:
                # No MALDI table, use generated coordinates
                pass

            # For non-MALDI data, generate coordinates (simple sequential mapping)
            return (frame_id - 1, 0, 0)

        except Exception as e:
            logger.warning(f"Error getting coordinates for frame {frame_id}: {e}")
            return None

    def _preload_frame_num_peaks(self) -> Dict[int, int]:
        """Preload NumPeaks values for all frames at initialization.

        This optimization avoids the busy wait loop in SDK by providing exact
        buffer sizes for spectrum reading, reducing CPU usage from 100% to normal levels.

        Returns:
            Dictionary mapping frame_id -> num_peaks (validated to be <= 65535)
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT Id, NumPeaks FROM Frames ORDER BY Id")

                # Use uint16 equivalent for memory efficiency (max 65,535 peaks)
                num_peaks_cache = {}
                invalid_count = 0

                for frame_id, num_peaks in cursor.fetchall():
                    if num_peaks is not None and 0 < num_peaks <= 65535:
                        num_peaks_cache[frame_id] = int(num_peaks)
                    else:
                        invalid_count += 1
                        if invalid_count <= 5:  # Log first few invalid values
                            logger.debug(
                                f"Invalid NumPeaks value {num_peaks} for frame {frame_id}"
                            )

                if invalid_count > 5:
                    logger.debug(
                        f"... and {invalid_count - 5} more invalid NumPeaks values"
                    )

                memory_mb = len(num_peaks_cache) * 2 / (1024 * 1024)  # uint16 = 2 bytes
                logger.info(
                    f"Cached NumPeaks for {len(num_peaks_cache)} frames ({memory_mb:.1f}MB)"
                )
                return num_peaks_cache

        except Exception as e:
            logger.warning(f"Failed to preload NumPeaks cache: {e}")
            logger.info("Will use fallback retry logic for spectrum reading")
            return {}  # Empty cache triggers fallback behavior

    def close(self) -> None:
        """Close all resources and connections."""
        logger.info("Closing Bruker reader")

        try:
            # Close SDK handle
            if hasattr(self, "handle") and self.handle:
                self.sdk.close_file(self.handle)
                self.handle = None

            # Close thread-local database connections
            if hasattr(self, "_local"):
                if hasattr(self._local, "connection") and self._local.connection:
                    self._local.connection.close()
                    self._local.connection = None

            # No caches to clear anymore

            logger.info("Successfully closed all resources")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get spatial dimensions (pixel grid) from metadata extractor.

        Note: This is the spatial pixel grid, not the mass axis dimensions.
        Mass axis interpolation to common m/z values is handled during conversion.

        Returns:
            Tuple of (x_pixels, y_pixels, z_pixels) spatial dimensions
        """
        essential_metadata = self.get_essential_metadata()
        return essential_metadata.dimensions

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Get mass range from metadata extractor.

        Note: This is the acquisition mass range, not the final interpolated axis.
        The actual common mass axis for interpolation is built from all unique m/z values.

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
    def n_spectra(self) -> int:
        """Return the total number of spectra in the dataset.

        Returns:
            Total number of frames (efficient implementation using cached frame count)
        """
        return self._get_frame_count()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception as e:
            logger.debug(
                f"Error during cleanup in destructor: {e}"
            )  # Log but don't raise during destruction
