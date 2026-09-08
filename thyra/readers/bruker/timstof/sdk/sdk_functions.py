"""SDK function definitions and wrappers for Bruker data access.

This module defines all the Bruker SDK functions with proper type
annotations and provides a clean interface for data access operations.

TDF (TIMS engaged) frames are read over the **full mobility ramp**. A TIMS
frame is one pixel, its scans are the mobility dimension, and the scan
number maps monotonically onto 1/K0. The one spectrum the reader yields per
pixel therefore has to collapse every scan of the frame, and there are two
correct ways to do that (see :data:`TDF_SPECTRUM_MODES`); ``scan_sum`` is
the default, ``vendor_centroid`` the opt-in:

``vendor_centroid``
    Bruker's own frame-level centroid extraction
    (``tims_extract_centroided_spectrum_for_frame_v2``) over scans
    ``0..NumScans``. This is the same peak picker behind the TSF line
    spectrum (``tsf_read_line_spectrum_v2``) and behind SCiLS Lab's import,
    so TSF and TDF stores from the same instrument family agree. It reports
    peak areas, merges neighbouring digitizer bins and drops every index
    bin its picker assigns to no peak, which on measured imaging
    acquisitions keeps 87-96% of the raw ion current. Bruker's own
    ``Frames.SummedIntensities`` is the scan sum, not the centroid total.

``scan_sum``
    The lossless alternative: every ``(index, scan)`` pair of the frame is
    read with ``tims_read_scans_v2`` and intensities are summed per
    digitizer index. Keeps 100% of the ion current, yields 1.5 to 3 times as
    many points per frame, equals Bruker's own quasi-profile export
    (``tims_extract_profile_for_frame``) and ``Frames.SummedIntensities``
    exactly, and is the only mode whose result is the mobility marginal
    of the per-scan data.

Frame ids are the 1-based ``Frames.Id`` of the SQLite database throughout;
the SDK takes them as-is.
"""

import logging
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
    create_string_buffer,
)
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .....errors import ConversionRefused
from .....utils.bruker_exceptions import SDKError
from .dll_manager import DLLManager

logger = logging.getLogger(__name__)


def sum_scans_per_index(
    inverse: NDArray[Any], intensities: NDArray[Any], n_unique: int
) -> NDArray[np.float64]:
    """The ``scan_sum`` collapse: every pair's intensity summed per unique index.

    ``inverse`` is ``np.unique(indices, return_inverse=True)``'s second
    answer. One expression, shared by the spectrum reader and the frame
    record (``thyra.core.frames``), so the two cannot sum differently.
    """
    return np.bincount(
        np.asarray(inverse).ravel(),
        weights=np.asarray(intensities).astype(np.float64),
        minlength=int(n_unique),
    )


#: How a TDF frame's TIMS scans are collapsed into the one spectrum the
#: reader yields per pixel. See the module docstring for what each means.
TdfSpectrumMode = Literal["vendor_centroid", "scan_sum"]
TDF_SPECTRUM_MODES: Tuple[str, ...] = ("vendor_centroid", "scan_sum")
#: ``scan_sum`` since the default was decided on measurement (see
#: ``docs/design-decisions.md`` D1): it is the instrument's own record and
#: equals Bruker's per-frame total; the centroid is the opt-in.
DEFAULT_TDF_SPECTRUM: str = "scan_sum"

# The callback the SDK's frame-level centroid extraction hands its result
# to: (precursor id, number of peaks, m/z values, area values). Declared
# once at module level -- a ctypes function-pointer type must outlive every
# call that uses it, and creating one per call is wasted work.
_MSMS_SPECTRUM_FUNCTOR = CFUNCTYPE(
    None, c_int64, c_uint32, POINTER(c_double), POINTER(c_float)
)

# Signature shared by the SDK's per-frame conversion functions:
# (handle, frame id, input values, output values, count) -> success flag.
_CONVERSION_ARGTYPES = [
    c_uint64,
    c_int64,
    POINTER(c_double),
    POINTER(c_double),
    c_uint32,
]

# Per-frame conversions bound for TDF. ``tims_index_to_mz`` is required;
# the rest are the mobility axis and are bound when the library has them.
_TDF_CONVERSIONS = (
    "tims_index_to_mz",
    "tims_mz_to_index",
    "tims_scannum_to_oneoverk0",
    "tims_oneoverk0_to_scannum",
    "tims_scannum_to_voltage",
)

# Words the scan-read buffer starts at when the frame carries no NumPeaks
# hint. The SDK reports the size it needs and the read is retried, so this
# only has to be a reasonable first guess.
_DEFAULT_SCAN_BUFFER_WORDS = 1 << 16


class SDKFunctions:
    """Wrapper class for all Bruker SDK functions.

    This class provides a clean, type-safe interface to the Bruker SDK
    with proper error handling and data conversion.
    """

    def __init__(
        self,
        dll_manager: DLLManager,
        file_type: str,
        tdf_spectrum: str = DEFAULT_TDF_SPECTRUM,
    ):
        """Initialize SDK functions for a specific file type.

        Args:
            dll_manager: Initialized DLL manager
            file_type: Either 'tsf' or 'tdf'
            tdf_spectrum: How TDF frames are collapsed into one spectrum
                per pixel, one of :data:`TDF_SPECTRUM_MODES`. Ignored for
                TSF, whose line spectrum is already per pixel.
        """
        if tdf_spectrum not in TDF_SPECTRUM_MODES:
            raise ConversionRefused(
                f"tdf_spectrum must be one of {TDF_SPECTRUM_MODES}, "
                f"got {tdf_spectrum!r}"
            )
        self.dll_manager = dll_manager
        self.file_type = file_type.lower()
        self.tdf_spectrum = tdf_spectrum
        self._has_centroid_extraction = False
        self._bound_conversions: Dict[str, bool] = {}
        self._setup_functions()

    def _setup_functions(self) -> None:
        """Setup function signatures based on file type."""
        dll = self.dll_manager.dll

        if self.file_type == "tsf":
            self._setup_tsf_functions(dll)
        elif self.file_type == "tdf":
            self._setup_tdf_functions(dll)
        else:
            raise SDKError(f"Unsupported file type: {self.file_type}")

    def _setup_tsf_functions(self, dll) -> None:
        """Setup TSF-specific function signatures."""
        # TSF open/close functions
        dll.tsf_open.argtypes = [c_char_p, c_uint32]
        dll.tsf_open.restype = c_uint64

        dll.tsf_close.argtypes = [c_uint64]
        dll.tsf_close.restype = None

        # Error handling
        dll.tsf_get_last_error_string.argtypes = [c_char_p, c_uint32]
        dll.tsf_get_last_error_string.restype = c_uint32

        # Spectrum reading
        dll.tsf_read_line_spectrum_v2.argtypes = [
            c_uint64,
            c_int64,
            POINTER(c_double),
            POINTER(c_float),
            c_int32,
        ]
        dll.tsf_read_line_spectrum_v2.restype = c_int32

        # Mass calibration
        dll.tsf_index_to_mz.argtypes = [
            c_int64,
            c_int64,
            POINTER(c_double),
            POINTER(c_double),
            c_uint32,
        ]
        dll.tsf_index_to_mz.restype = c_uint32

    def _setup_tdf_functions(self, dll) -> None:
        """Setup TDF-specific function signatures."""
        # TDF open/close functions
        dll.tims_open.argtypes = [c_char_p, c_uint32]
        dll.tims_open.restype = c_uint64

        dll.tims_close.argtypes = [c_uint64]
        dll.tims_close.restype = None

        # Error handling
        dll.tims_get_last_error_string.argtypes = [c_char_p, c_uint32]
        dll.tims_get_last_error_string.restype = c_uint32

        # Raw scan reading: every (index, intensity) pair of a scan range
        dll.tims_read_scans_v2.argtypes = [
            c_uint64,
            c_int64,
            c_uint32,
            c_uint32,
            POINTER(c_uint32),
            c_uint32,
        ]
        dll.tims_read_scans_v2.restype = c_uint32

        # Per-frame conversions: mass calibration (required) and the
        # mobility axis (bound when the library exports them).
        for name in _TDF_CONVERSIONS:
            func = getattr(dll, name, None)
            if func is None:
                if name == "tims_index_to_mz":
                    raise SDKError(
                        "The Bruker library does not export tims_index_to_mz; "
                        "TDF data cannot be calibrated with it"
                    )
                self._bound_conversions[name] = False
                logger.debug("Bruker library does not export %s", name)
                continue
            func.argtypes = _CONVERSION_ARGTYPES
            func.restype = c_uint32
            self._bound_conversions[name] = True

        # 1/K0 <-> CCS (Mason-Schamp); no file handle involved.
        for name in ("tims_oneoverk0_to_ccs_for_mz", "tims_ccs_to_oneoverk0_for_mz"):
            func = getattr(dll, name, None)
            if func is not None:
                func.argtypes = [c_double, c_int32, c_double]
                func.restype = c_double
            self._bound_conversions[name] = func is not None

        # Vendor frame-level centroid extraction over a scan range. This is
        # what ``vendor_centroid`` uses; without it the reader falls back to
        # summing scans, which is also correct, only denser.
        extract = getattr(dll, "tims_extract_centroided_spectrum_for_frame_v2", None)
        if extract is not None:
            extract.argtypes = [
                c_uint64,
                c_int64,
                c_uint32,
                c_uint32,
                _MSMS_SPECTRUM_FUNCTOR,
                c_void_p,
            ]
            extract.restype = c_uint32
            self._has_centroid_extraction = True
        elif self.tdf_spectrum == "vendor_centroid":
            logger.warning(
                "The Bruker library does not export "
                "tims_extract_centroided_spectrum_for_frame_v2; TDF frames "
                "will be summed over their scans instead (tdf_spectrum="
                "'scan_sum')"
            )
            self.tdf_spectrum = "scan_sum"

    def open_file(self, file_path: str, use_recalibrated: bool = False) -> int:
        """Open a Bruker data file.

        Args:
            file_path: Path to the data directory
            use_recalibrated: Whether to use recalibrated data

        Returns:
            File handle for subsequent operations

        Raises:
            SDKError: If file cannot be opened
        """
        dll = self.dll_manager.dll

        if self.file_type == "tsf":
            handle = dll.tsf_open(
                file_path.encode("utf-8"), 1 if use_recalibrated else 0
            )
        else:  # tdf
            handle = dll.tims_open(
                file_path.encode("utf-8"), 1 if use_recalibrated else 0
            )

        if handle == 0:
            error_msg = self._get_last_error()
            raise SDKError(f"Failed to open {self.file_type.upper()} file: {error_msg}")

        logger.debug(
            f"Opened {self.file_type.upper()} file: {file_path} (handle: {handle})"
        )
        return int(handle)

    def close_file(self, handle: int) -> None:
        """Close a Bruker data file.

        Args:
            handle: File handle to close
        """
        dll = self.dll_manager.dll

        if self.file_type == "tsf":
            dll.tsf_close(handle)
        else:  # tdf
            dll.tims_close(handle)

        logger.debug(f"Closed {self.file_type.upper()} file (handle: {handle})")

    def read_spectrum(
        self,
        handle: int,
        frame_id: int,
        buffer_size_hint: Optional[int] = None,
        num_scans: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read one frame as a single (m/z, intensity) spectrum.

        Args:
            handle: File handle
            frame_id: The 1-based ``Frames.Id`` of the frame to read
            buffer_size_hint: The frame's ``Frames.NumPeaks`` when known.
                For TSF it is the exact line-spectrum length and avoids a
                retry loop; for TDF it is the number of (index, scan) pairs
                and sizes the raw scan buffer.
            num_scans: The frame's ``Frames.NumScans``. Required for TDF:
                every scan of the mobility ramp is read, and the SDK
                does not report the scan count itself.

        Returns:
            Tuple of (m/z array, intensity array)

        Raises:
            SDKError: If spectrum cannot be read
        """
        # Use optimized buffer size if provided, otherwise default
        buffer_size = (
            buffer_size_hint if buffer_size_hint and buffer_size_hint > 0 else 1024
        )

        if self.file_type == "tsf":
            return self._read_tsf_spectrum(
                handle, frame_id, buffer_size, buffer_size_hint is not None
            )

        if num_scans is None:
            raise SDKError(
                f"Reading TDF frame {frame_id} requires its NumScans (the length "
                "of the mobility ramp); the reader passes it from the Frames table"
            )
        return self._read_tdf_spectrum(
            handle, frame_id, int(num_scans), buffer_size_hint
        )

    def _read_tsf_spectrum(
        self,
        handle: int,
        frame_id: int,
        buffer_size: int,
        is_optimized: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read spectrum from TSF file with optional optimization.

        Args:
            handle: File handle
            frame_id: Frame ID to read
            buffer_size: Buffer size to use
            is_optimized: Whether buffer_size is exact (avoids retry loop)
        """
        dll = self.dll_manager.dll

        # OPTIMIZED PATH: Try exact buffer size first (no retries expected)
        if is_optimized:
            try:
                # Allocate buffers with exact size
                mz_indices = np.empty(buffer_size, dtype=np.float64)
                intensities = np.empty(buffer_size, dtype=np.float32)

                # Read spectrum
                result = dll.tsf_read_line_spectrum_v2(
                    handle,
                    frame_id,
                    mz_indices.ctypes.data_as(POINTER(c_double)),
                    intensities.ctypes.data_as(POINTER(c_float)),
                    buffer_size,
                )

                if result < 0:
                    error_msg = self._get_last_error()
                    raise SDKError(f"Failed to read TSF spectrum: {error_msg}")

                if result == 0:
                    return np.array([]), np.array([])

                if result <= buffer_size:
                    # SUCCESS: Exact buffer size worked!
                    mzs = self._convert_indices_to_mz(
                        handle, frame_id, mz_indices[:result]
                    )
                    return mzs, intensities[:result].copy()
                else:
                    # Buffer hint was too small, fall back to retry logic
                    logger.debug(
                        f"Buffer hint {buffer_size} too small for frame {frame_id} "
                        f"(needed {result}), falling back"
                    )

            except Exception as e:
                logger.debug(
                    f"Optimized read failed for frame {frame_id}: {e}, falling back"
                )

        # FALLBACK PATH: Use original retry loop logic
        return self._read_tsf_spectrum_with_retries(handle, frame_id, buffer_size)

    def _read_tsf_spectrum_with_retries(
        self, handle: int, frame_id: int, initial_buffer_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Original TSF spectrum reading with retry loop (fallback)."""
        dll = self.dll_manager.dll
        buffer_size = initial_buffer_size

        while True:
            # Allocate buffers
            mz_indices = np.empty(buffer_size, dtype=np.float64)
            intensities = np.empty(buffer_size, dtype=np.float32)

            # Read spectrum
            result = dll.tsf_read_line_spectrum_v2(
                handle,
                frame_id,
                mz_indices.ctypes.data_as(POINTER(c_double)),
                intensities.ctypes.data_as(POINTER(c_float)),
                buffer_size,
            )

            if result < 0:
                error_msg = self._get_last_error()
                raise SDKError(f"Failed to read TSF spectrum: {error_msg}")

            if result > buffer_size:
                logger.debug(
                    f"Buffer resized from {buffer_size} to {result} for frame {frame_id}"
                )
                # Buffer too small, resize and try again (BUSY WAIT LOOP)
                buffer_size = result
                continue

            if result == 0:
                return np.array([]), np.array([])

            # Convert indices to m/z values
            mzs = self._convert_indices_to_mz(handle, frame_id, mz_indices[:result])
            return mzs, intensities[:result].copy()

    # ------------------------------------------------------------------
    # TDF: the mobility ramp collapsed into one spectrum per frame
    # ------------------------------------------------------------------

    def _read_tdf_spectrum(
        self,
        handle: int,
        frame_id: int,
        num_scans: int,
        num_peaks_hint: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse every scan of a TDF frame into one spectrum.

        Dispatches on :attr:`tdf_spectrum`; both routes read scans
        ``0..num_scans`` of the frame ``frame_id`` itself.
        """
        if self.tdf_spectrum == "vendor_centroid":
            return self.read_tdf_centroided_spectrum(handle, frame_id, 0, num_scans)
        return self._read_tdf_scan_sum(handle, frame_id, num_scans, num_peaks_hint)

    def read_tdf_centroided_spectrum(
        self, handle: int, frame_id: int, scan_begin: int, scan_end: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Bruker's centroid spectrum for a scan range of one frame.

        Wraps ``tims_extract_centroided_spectrum_for_frame_v2``. Passing
        ``0..NumScans`` yields the vendor's mobility-summed line spectrum;
        a narrower range yields the spectrum of one mobility window.

        Args:
            handle: File handle
            frame_id: The 1-based ``Frames.Id``
            scan_begin: First scan (inclusive)
            scan_end: End scan (exclusive)

        Returns:
            (m/z, intensity) as float64 arrays sorted by m/z; both empty
            when the range holds no peaks.
        """
        if not self._has_centroid_extraction:
            raise SDKError(
                "The Bruker library does not export "
                "tims_extract_centroided_spectrum_for_frame_v2"
            )
        dll = self.dll_manager.dll
        result: Dict[str, NDArray[np.float64]] = {}

        def _collect(precursor_id, num_peaks, mz_values, area_values):
            n = int(num_peaks)
            if n <= 0:
                return
            # Copy out inside the callback: the SDK owns these buffers only
            # for the duration of the call.
            result["mz"] = np.ctypeslib.as_array(mz_values, shape=(n,)).astype(
                np.float64, copy=True
            )
            result["intensity"] = np.ctypeslib.as_array(area_values, shape=(n,)).astype(
                np.float64, copy=True
            )

        callback = _MSMS_SPECTRUM_FUNCTOR(_collect)
        rc = dll.tims_extract_centroided_spectrum_for_frame_v2(
            handle, frame_id, scan_begin, scan_end, callback, None
        )
        if rc == 0:
            error_msg = self._get_last_error()
            raise SDKError(
                f"Failed to extract centroided spectrum for frame {frame_id}: "
                f"{error_msg}"
            )
        if "mz" not in result:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        mzs = result["mz"]
        intensities = result["intensity"]
        # The SDK returns ascending m/z already; the sort is cheap insurance
        # for the downstream code that relies on it.
        order = np.argsort(mzs, kind="stable")
        return mzs[order], intensities[order]

    def read_tdf_scans(
        self,
        handle: int,
        frame_id: int,
        scan_begin: int,
        scan_end: int,
        num_peaks_hint: Optional[int] = None,
    ) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.int32]]:
        """Every (index, intensity) pair of a scan range, with scan provenance.

        Wraps ``tims_read_scans_v2`` and parses its buffer in one
        vectorised pass. The buffer layout is: one peak count per scan,
        then for each scan its ``n`` mass indices followed by its ``n``
        intensities. The scan number is kept per pair because it *is* the
        mobility coordinate (``tims_scannum_to_oneoverk0`` turns it into
        1/K0); the plain spectrum readers discard it, a mobility-aware
        consumer must not.

        Args:
            handle: File handle
            frame_id: The 1-based ``Frames.Id``
            scan_begin: First scan (inclusive)
            scan_end: End scan (exclusive)
            num_peaks_hint: The frame's ``Frames.NumPeaks`` when known; it
                is the exact number of pairs in the whole frame and sizes
                the buffer so the read succeeds first time.

        Returns:
            ``(indices, intensities, scan_numbers)``, three flat arrays of
            equal length ordered by scan and then by index. Empty when the
            range holds no pairs.
        """
        dll = self.dll_manager.dll
        n_scans = scan_end - scan_begin
        empty = (
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.int32),
        )
        if n_scans <= 0:
            return empty

        if num_peaks_hint and num_peaks_hint > 0:
            n_words = n_scans + 2 * int(num_peaks_hint) + 64
        else:
            n_words = max(_DEFAULT_SCAN_BUFFER_WORDS, n_scans + 64)

        while True:
            buffer = np.empty(n_words, dtype=np.uint32)
            required_bytes = dll.tims_read_scans_v2(
                handle,
                frame_id,
                scan_begin,
                scan_end,
                buffer.ctypes.data_as(POINTER(c_uint32)),
                n_words * 4,
            )
            if required_bytes == 0:
                error_msg = self._get_last_error()
                raise SDKError(
                    f"Failed to read TDF scans of frame {frame_id}: {error_msg}"
                )
            if required_bytes > n_words * 4:
                logger.debug(
                    "Scan buffer grown from %d to %d words for frame %d",
                    n_words,
                    required_bytes // 4 + 1,
                    frame_id,
                )
                n_words = required_bytes // 4 + 1
                continue
            break

        counts = buffer[:n_scans].astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            return empty
        body = buffer[n_scans : n_scans + 2 * total]

        # Per scan s the body holds a block of 2*n_s words: n_s indices then
        # n_s intensities. Gather both halves for every pair at once.
        pair_offsets = np.cumsum(counts) - counts  # first pair of each scan
        block_starts = 2 * pair_offsets  # first word of each scan's block
        within_scan = np.arange(total, dtype=np.int64) - np.repeat(pair_offsets, counts)
        base = np.repeat(block_starts, counts) + within_scan
        indices = body[base]
        intensities = body[base + np.repeat(counts, counts)]
        scan_numbers = np.repeat(
            np.arange(scan_begin, scan_end, dtype=np.int32), counts
        )
        return indices, intensities, scan_numbers

    def _read_tdf_scan_sum(
        self,
        handle: int,
        frame_id: int,
        num_scans: int,
        num_peaks_hint: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sum every scan of the frame per digitizer index; lossless."""
        indices, intensities, _ = self.read_tdf_scans(
            handle, frame_id, 0, num_scans, num_peaks_hint
        )
        if indices.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        unique_indices, inverse = np.unique(indices, return_inverse=True)
        summed = sum_scans_per_index(inverse, intensities, unique_indices.size)
        # Unique indices are ascending and the calibration is monotonic, so
        # the m/z array comes out sorted without a second pass.
        mzs = self._convert_indices_to_mz(
            handle, frame_id, unique_indices.astype(np.float64)
        )
        return mzs, summed

    # ------------------------------------------------------------------
    # Axis conversions
    # ------------------------------------------------------------------

    def _convert(
        self, func_name: str, handle: int, frame_id: int, values: np.ndarray
    ) -> NDArray[np.float64]:
        """Run one of the SDK's per-frame value conversions."""
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.size == 0:
            return np.array([], dtype=np.float64)
        if not self._bound_conversions.get(func_name, False):
            raise SDKError(f"The Bruker library does not export {func_name}")
        func = getattr(self.dll_manager.dll, func_name)
        out = np.empty_like(values)
        success = func(
            handle,
            frame_id,
            values.ctypes.data_as(POINTER(c_double)),
            out.ctypes.data_as(POINTER(c_double)),
            values.size,
        )
        if success == 0:
            error_msg = self._get_last_error()
            raise SDKError(f"{func_name} failed for frame {frame_id}: {error_msg}")
        return out

    def _convert_indices_to_mz(
        self, handle: int, frame_id: int, indices: np.ndarray
    ) -> np.ndarray:
        """Convert mass indices to m/z values."""
        if indices.size == 0:
            return np.array([])

        if self.file_type == "tsf":
            dll = self.dll_manager.dll
            indices = np.ascontiguousarray(indices, dtype=np.float64)
            mzs = np.empty_like(indices)
            success = dll.tsf_index_to_mz(
                handle,
                frame_id,
                indices.ctypes.data_as(POINTER(c_double)),
                mzs.ctypes.data_as(POINTER(c_double)),
                indices.size,
            )
            if success == 0:
                error_msg = self._get_last_error()
                raise SDKError(f"Failed to convert indices to m/z: {error_msg}")
            return mzs

        return self._convert("tims_index_to_mz", handle, frame_id, indices)

    def index_to_mz(
        self, handle: int, frame_id: int, indices: np.ndarray
    ) -> NDArray[np.float64]:
        """Convert digitizer (mass) indices of a frame to m/z.

        The public face of the conversion the spectrum readers use
        internally; a mobility-aware consumer that reads raw scans needs
        it too, on the frame's unique indices rather than on every pair.
        """
        return np.asarray(
            self._convert_indices_to_mz(handle, frame_id, indices), dtype=np.float64
        )

    def scannum_to_oneoverk0(
        self, handle: int, frame_id: int, scan_numbers: np.ndarray
    ) -> NDArray[np.float64]:
        """Convert TIMS scan numbers of a frame to 1/K0 (V s cm^-2).

        Scan numbers may be fractional. On MALDI imaging acquisitions the
        calibration is one row per file, so the result is the same for
        every frame; it is still per-frame in the SDK's contract.
        """
        return self._convert(
            "tims_scannum_to_oneoverk0", handle, frame_id, scan_numbers
        )

    def oneoverk0_to_scannum(
        self, handle: int, frame_id: int, one_over_k0: np.ndarray
    ) -> NDArray[np.float64]:
        """Convert 1/K0 values to (fractional) TIMS scan numbers of a frame."""
        return self._convert("tims_oneoverk0_to_scannum", handle, frame_id, one_over_k0)

    def oneoverk0_to_ccs(
        self, one_over_k0: np.ndarray, charge: int, mz: np.ndarray
    ) -> NDArray[np.float64]:
        """Collision cross section (A^2) from 1/K0, charge and m/z.

        Bruker's Mason-Schamp implementation (nitrogen drift gas). The
        charge is an input, not something this layer can know: MALDI is
        mostly 1+, but that is the caller's assumption to make and record.
        """
        if not self._bound_conversions.get("tims_oneoverk0_to_ccs_for_mz", False):
            raise SDKError(
                "The Bruker library does not export tims_oneoverk0_to_ccs_for_mz"
            )
        func = self.dll_manager.dll.tims_oneoverk0_to_ccs_for_mz
        k0 = np.asarray(one_over_k0, dtype=np.float64)
        mz_arr = np.asarray(mz, dtype=np.float64)
        k0_b, mz_b = np.broadcast_arrays(k0, mz_arr)
        out = np.empty(k0_b.shape, dtype=np.float64)
        flat_out = out.reshape(-1)
        for i, (k, m) in enumerate(zip(k0_b.reshape(-1), mz_b.reshape(-1))):
            flat_out[i] = func(float(k), int(charge), float(m))
        return out

    def _get_last_error(self) -> str:
        """Get the last error message from the SDK."""
        dll = self.dll_manager.dll

        if self.file_type == "tsf":
            len_buf = dll.tsf_get_last_error_string(None, 0)
            buf = create_string_buffer(len_buf)
            dll.tsf_get_last_error_string(buf, len_buf)
        else:  # tdf
            len_buf = dll.tims_get_last_error_string(None, 0)
            buf = create_string_buffer(len_buf)
            dll.tims_get_last_error_string(buf, len_buf)

        return buf.value.decode("utf-8") if buf.value else "Unknown error"
