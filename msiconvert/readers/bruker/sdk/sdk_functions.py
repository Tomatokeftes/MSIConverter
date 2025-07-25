"""
SDK function definitions and wrappers for Bruker data access.

This module defines all the Bruker SDK functions with proper type annotations
and provides a clean interface for data access operations.
"""

import logging
from ctypes import (
    POINTER,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    create_string_buffer,
)
from typing import List, Optional, Tuple

import numpy as np

from ....utils.bruker_exceptions import SDKError
from .dll_manager import DLLManager

logger = logging.getLogger(__name__)


class SDKFunctions:
    """
    Wrapper class for all Bruker SDK functions.

    This class provides a clean, type-safe interface to the Bruker SDK
    with proper error handling and data conversion.
    """

    def __init__(self, dll_manager: DLLManager, file_type: str):
        """
        Initialize SDK functions for a specific file type.

        Args:
            dll_manager: Initialized DLL manager
            file_type: Either 'tsf' or 'tdf'
        """
        self.dll_manager = dll_manager
        self.file_type = file_type.lower()
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

        # Scan reading
        dll.tims_read_scans_v2.argtypes = [
            c_uint64,
            c_int64,
            c_uint32,
            c_uint32,
            POINTER(c_uint32),
            c_uint32,
        ]
        dll.tims_read_scans_v2.restype = c_uint32

        # Mass calibration
        dll.tims_index_to_mz.argtypes = [
            c_int64,
            c_int64,
            POINTER(c_double),
            POINTER(c_double),
            c_uint32,
        ]
        dll.tims_index_to_mz.restype = c_uint32

    def open_file(self, file_path: str, use_recalibrated: bool = False) -> int:
        """
        Open a Bruker data file.

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
        return handle

    def close_file(self, handle: int) -> None:
        """
        Close a Bruker data file.

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
        self, handle: int, frame_id: int, buffer_size_hint: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a spectrum from the file with optional buffer size optimization.

        Args:
            handle: File handle
            frame_id: Frame ID to read
            buffer_size_hint: Exact buffer size if known (avoids retry loop for TSF)

        Returns:
            Tuple of (m/z array, intensity array)

        Raises:
            SDKError: If spectrum cannot be read
        """
        # Use optimized buffer size if provided, otherwise default
        buffer_size = buffer_size_hint if buffer_size_hint and buffer_size_hint > 0 else 1024
        
        if self.file_type == "tsf":
            return self._read_tsf_spectrum(handle, frame_id, buffer_size, buffer_size_hint is not None)
        else:  # tdf
            return self._read_tdf_spectrum(handle, frame_id, buffer_size)

    def _read_tsf_spectrum(
        self, handle: int, frame_id: int, buffer_size: int, is_optimized: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read spectrum from TSF file with optional optimization.
        
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
                    mzs = self._convert_indices_to_mz(handle, frame_id, mz_indices[:result])
                    return mzs, intensities[:result].copy()
                else:
                    # Buffer hint was too small, fall back to retry logic
                    logger.debug(f"Buffer hint {buffer_size} too small for frame {frame_id} (needed {result}), falling back")
                    
            except Exception as e:
                logger.debug(f"Optimized read failed for frame {frame_id}: {e}, falling back")

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

    def _read_tdf_spectrum(
        self, handle: int, frame_id: int, buffer_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read spectrum from TDF file (simplified version)."""
        # For TDF files, we need to read scans and combine them
        # This is a simplified version - full implementation would handle mobility

        dll = self.dll_manager.dll

        # Read scans for the frame
        scan_data = self._read_scans(
            handle, frame_id - 1, 0, 100
        )  # Read first 100 scans

        if not scan_data:
            return np.array([]), np.array([])

        # Combine scan data
        all_mz_indices = []
        all_intensities = []

        for indices, intensities in scan_data:
            if len(indices) > 0:
                mzs = self._convert_indices_to_mz(handle, frame_id, indices)
                all_mz_indices.extend(mzs)
                all_intensities.extend(intensities)

        if not all_mz_indices:
            return np.array([]), np.array([])

        # Sort by m/z
        mz_array = np.array(all_mz_indices)
        intensity_array = np.array(all_intensities)
        sort_idx = np.argsort(mz_array)

        return mz_array[sort_idx], intensity_array[sort_idx]

    def _read_scans(
        self, handle: int, frame_id: int, scan_begin: int, scan_end: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Read scans from TDF file."""
        dll = self.dll_manager.dll
        buffer_size = 1024

        while True:
            buffer = np.empty(buffer_size, dtype=np.uint32)

            required_len = dll.tims_read_scans_v2(
                handle,
                frame_id,
                scan_begin,
                scan_end,
                buffer.ctypes.data_as(POINTER(c_uint32)),
                buffer_size * 4,
            )

            if required_len == 0:
                error_msg = self._get_last_error()
                raise SDKError(f"Failed to read TDF scans: {error_msg}")

            if required_len > buffer_size * 4:
                buffer_size = (required_len // 4) + 1
                continue

            break

        # Parse scan data (simplified)
        result = []
        offset = scan_end - scan_begin

        for i in range(scan_begin, scan_end):
            idx = i - scan_begin
            if idx >= len(buffer):
                break

            peak_count = buffer[idx]
            if peak_count > 0 and offset + peak_count * 2 <= len(buffer):
                indices = buffer[offset : offset + peak_count].astype(np.float64)
                offset += peak_count
                intensities = buffer[offset : offset + peak_count]
                offset += peak_count
                result.append((indices, intensities))

        return result

    def _convert_indices_to_mz(
        self, handle: int, frame_id: int, indices: np.ndarray
    ) -> np.ndarray:
        """Convert mass indices to m/z values."""
        if indices.size == 0:
            return np.array([])

        dll = self.dll_manager.dll
        mzs = np.empty_like(indices)

        if self.file_type == "tsf":
            func = dll.tsf_index_to_mz
        else:  # tdf
            func = dll.tims_index_to_mz

        success = func(
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
