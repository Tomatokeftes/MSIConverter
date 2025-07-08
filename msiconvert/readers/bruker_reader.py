# msiconvert/readers/bruker_reader.py
import numpy as np
import sqlite3
from typing import Dict, Any, Tuple, Generator, List, Optional, cast
from numpy.typing import NDArray

from pathlib import Path
import os
import logging
import platform
from ctypes import (
    c_double, c_float, c_int32, POINTER, c_int64, c_uint32, 
    c_uint64, c_char_p, create_string_buffer
)
from tqdm import tqdm

from ..core.base_reader import BaseMSIReader
from ..core.registry import register_reader

@register_reader('bruker')
class BrukerReader(BaseMSIReader):
    """Reader for Bruker TSF/TDF format files.
    
    This implementation focuses on memory efficiency and streaming access
    to avoid loading the entire dataset into memory at once.
    """
    
    def __init__(
        self, 
        analysis_directory: Path, 
        use_recalibrated_state: bool = False,
        cached_metadata: bool = True, 
        batch_size: int = 50
    ) -> None:
        """
        Initialize the Bruker reader with the path to the analysis directory.
        
        Args:
            analysis_directory: Path to a Bruker .d directory containing analysis.tsf or analysis.tdf
            use_recalibrated_state: Whether to use recalibrated data (True) or not (False)
            cached_metadata: Whether to cache metadata upfront (True) or load on demand
            batch_size: Default batch size for spectrum iteration
            
        Raises:
            FileNotFoundError: If no analysis.tsf or analysis.tdf file is found
        """
        self.analysis_directory: Path = Path(analysis_directory).resolve()
        self.use_recalibrated_state: bool = use_recalibrated_state
        self.handle: Optional[int] = None
        self.conn: Optional[sqlite3.Connection] = None
        self._dll: Any = None
        self.default_batch_size: int = batch_size
        self.file_type: str = ""
        
        # Determine the analysis file type (TSF or TDF)
        self.tsf_path: Path = self.analysis_directory / "analysis.tsf"
        self.tdf_path: Path = self.analysis_directory / "analysis.tdf"
        
        if self.tsf_path.exists():
            self.file_type = "tsf"
        elif self.tdf_path.exists():
            self.file_type = "tdf"
        else:
            raise FileNotFoundError(f"No analysis.tsf or analysis.tdf found in {self.analysis_directory}")
        
        # Cached properties
        self._dimensions: Optional[Tuple[int, int, int]] = None
        self._common_mass_axis: Optional[NDArray[np.float64]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._frame_positions: Optional[NDArray[np.int32]] = None
        self._frame_count: Optional[int] = None
        self._position_cache: Dict[int, Tuple[int, int]] = {}  # Cache for frame positions
        
        # Buffers for spectrum reading
        self._line_buffer_size: int = 1024
        self._mz_indices_buffer: Optional[NDArray[np.float64]] = None
        self._intensities_buffer: Optional[NDArray[np.float64]] = None
        self._tdf_mz_buffer: Optional[NDArray[np.float64]] = None
        self._tdf_intensity_buffer: Optional[NDArray[np.float64]] = None
        self._scan_buffer: Optional[NDArray[np.uint32]] = None
        self._scan_buffer_size: int = 128
        self._scan_count_cache: Dict[int, int] = {}
        
        # Initialize DLL and database connection
        self._load_dll()
        self._open_data()
        
        # Optionally preload all metadata
        if cached_metadata:
            self._preload_metadata()
        
    def _find_dll_path(self) -> Optional[Path]:
        """
        Search for the Bruker DLL/shared library in common locations.
        
        Returns:
            Path to the DLL/shared library if found, None otherwise.
        """
        dll_name: str
        if platform.system() == "Windows":
            dll_name = "timsdata.dll"
            # Common installation paths for Windows
            search_paths: List[Path] = [
                Path("C:/Program Files/Bruker/timsTOF/sdk"),
                Path("C:/Bruker/sdk"),
                Path(os.environ.get("BRUKER_SDK_PATH", "")), # Check environment variable
                Path(__file__).parent.parent.parent, # Project root
                Path(__file__).parent.parent.parent / "timsdata", # timsdata folder in project root
                Path(__file__).parent.parent.parent / "timsdata" / "windows", # timsdata/windows folder in project root
                Path(os.getcwd()), # Current working directory
            ]
        elif platform.system() == "Linux":
            dll_name = "libtimsdata.so"
            # Common installation paths for Linux
            search_paths = [
                Path("/usr/lib"),
                Path("/usr/local/lib"),
                Path(os.environ.get("BRUKER_SDK_PATH", "")), # Check environment variable
                Path(__file__).parent.parent.parent, # Project root
                Path(__file__).parent.parent.parent / "timsdata", # timsdata folder in project root
                Path(__file__).parent.parent.parent / "timsdata" / "linux", # timsdata/linux folder in project root
                Path(os.getcwd()), # Current working directory
            ]
        else:
            logging.error(f"Unsupported platform: {platform.system()}")
            return None

        # Filter out empty or non-existent paths
        search_paths = [p for p in search_paths if p and p.exists()]

        for p in search_paths:
            dll_path = p / dll_name
            if dll_path.is_file():
                logging.info(f"Found Bruker DLL/shared library at: {dll_path}")
                return dll_path
        
        logging.warning(f"Bruker DLL/shared library ({dll_name}) not found in common locations or specified paths.")
        return None

    def _load_dll(self) -> None:
        """
        Load the appropriate DLL based on platform.
        
        Raises:
            RuntimeError: If the required DLL/shared library cannot be found or loaded.
        """
        dll_path = self._find_dll_path()
        if not dll_path:
            raise RuntimeError(
                f"Bruker DLL/shared library not found. Please ensure 'timsdata.dll' (Windows) "
                f"or 'libtimsdata.so' (Linux) is installed and accessible, or set the BRUKER_SDK_PATH environment variable."
            )

        try:
            if platform.system() == "Windows":
                from ctypes import windll
                self._dll = windll.LoadLibrary(str(dll_path))
            elif platform.system() == "Linux":
                from ctypes import cdll
                self._dll = cdll.LoadLibrary(str(dll_path))
            else:
                raise RuntimeError(f"Unsupported platform: {platform.system()}")
            
            # Configure DLL functions
            self._define_dll_functions()
            
        except OSError as e:
            logging.error(f"Error loading Bruker SDK from {dll_path}: {e}")
            raise RuntimeError(f"Failed to load Bruker DLL/shared library from {dll_path}. Check file permissions and integrity.") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading Bruker SDK from {dll_path}: {e}")
            raise RuntimeError(f"An unexpected error occurred during DLL loading: {e}") from e
    
    def _define_dll_functions(self) -> None:
        """
        Define the argument and return types for DLL functions.
        
        This ensures proper type safety when calling external library functions.
        """
        if self._dll is None:
            raise RuntimeError("DLL not loaded")
            
        # Common conversion function arguments
        convfunc_argtypes: List[Any] = [
            c_int64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32
        ]
        
        # Define functions based on file type (TSF or TDF)
        if self.file_type == "tsf":
            # TSF-specific functions
            try:
                self._dll.tsf_open.argtypes = [c_char_p, c_uint32]
                self._dll.tsf_open.restype = c_uint64
                
                self._dll.tsf_close.argtypes = [c_uint64]
                self._dll.tsf_close.restype = None
                
                self._dll.tsf_get_last_error_string.argtypes = [c_char_p, c_uint32]
                self._dll.tsf_get_last_error_string.restype = c_uint32
                
                self._dll.tsf_read_line_spectrum_v2.argtypes = [c_uint64, c_int64, POINTER(c_double), POINTER(c_float), c_int32]
                self._dll.tsf_read_line_spectrum_v2.restype = c_int32
                
                self._dll.tsf_index_to_mz.argtypes = convfunc_argtypes
                self._dll.tsf_index_to_mz.restype = c_uint32
            except AttributeError as e:
                raise RuntimeError(f"Missing TSF DLL function: {e}. Ensure the correct Bruker SDK is installed.") from e
            
        elif self.file_type == "tdf":
            # TDF-specific functions
            try:
                self._dll.tims_open.argtypes = [c_char_p, c_uint32]
                self._dll.tims_open.restype = c_uint64
                
                self._dll.tims_close.argtypes = [c_uint64]
                self._dll.tims_close.restype = None
                
                self._dll.tims_get_last_error_string.argtypes = [c_char_p, c_uint32]
                self._dll.tims_get_last_error_string.restype = c_uint32
                
                self._dll.tims_read_scans_v2.argtypes = [c_uint64, c_int64, c_uint32, c_uint32, 
                                                      POINTER(c_uint32), c_uint32]
                self._dll.tims_read_scans_v2.restype = c_uint32
                
                self._dll.tims_index_to_mz.argtypes = convfunc_argtypes
                self._dll.tims_index_to_mz.restype = c_uint32
            except AttributeError as e:
                raise RuntimeError(f"Missing TDF DLL function: {e}. Ensure the correct Bruker SDK is installed.") from e
    
    def _throw_last_error(self) -> None:
        """
        Retrieve and raise the last error from the DLL.
        
        Raises:
            RuntimeError: With the error message from the DLL
        """
        if self._dll is None:
            raise RuntimeError("DLL not loaded")
            
        try:
            if self.file_type == "tsf":
                len_buf: int = self._dll.tsf_get_last_error_string(None, 0)
            else:  # TDF
                len_buf: int = self._dll.tims_get_last_error_string(None, 0)
        except AttributeError as e:
            raise RuntimeError(f"Error getting last error string from DLL: {e}. DLL functions might be incorrectly defined.") from e
            
        buf: Any = create_string_buffer(len_buf)
        
        try:
            if self.file_type == "tsf":
                self._dll.tsf_get_last_error_string(buf, len_buf)
            else:  # TDF
                self._dll.tims_get_last_error_string(buf, len_buf)
        except AttributeError as e:
            raise RuntimeError(f"Error retrieving last error string from DLL: {e}. DLL functions might be incorrectly defined.") from e
            
        raise RuntimeError(buf.value.decode('utf-8'))
    
    def _open_data(self) -> None:
        """
        Open the Bruker data file and database connection.
        
        Raises:
            RuntimeError: If DLL handle is invalid or database connection fails
        """
        if self._dll is None:
            raise RuntimeError("DLL not loaded. Call _load_dll() first.")
            
        try:
            # Open the data handle based on file type
            if self.file_type == "tsf":
                self.handle = int(self._dll.tsf_open(
                    str(self.analysis_directory).encode('utf-8'),
                    1 if self.use_recalibrated_state else 0
                ))
            else:  # TDF
                self.handle = int(self._dll.tims_open(
                    str(self.analysis_directory).encode('utf-8'),
                    1 if self.use_recalibrated_state else 0
                ))
            
            if self.handle == 0:
                self._throw_last_error()
                
        except Exception as e:
            error_msg = f"Failed to open Bruker data file handle for {self.analysis_directory}: {e}"
            logging.error(error_msg)
            self.close()
            raise RuntimeError(error_msg) from e
            
        try:
            # Initialize database connection
            db_path: Path = self.tsf_path if self.file_type == "tsf" else self.tdf_path
            self.conn = sqlite3.connect(str(db_path))
            
            # Load minimal required metadata
            self._load_minimal_metadata()
            
        except sqlite3.Error as e:
            error_msg = f"Failed to connect to or query SQLite database for {self.analysis_directory}: {e}"
            logging.error(error_msg)
            self.close()
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"An unexpected error occurred during data opening for {self.analysis_directory}: {e}"
            logging.error(error_msg)
            self.close()
            raise RuntimeError(error_msg) from e
    
    def _load_minimal_metadata(self) -> None:
        """
        Load minimal required metadata to determine dimensions and frame count.
        
        Raises:
            sqlite3.Error: If database queries fail
        """
        if self.conn is None:
            raise RuntimeError("Database connection not initialized")
            
        cursor: sqlite3.Cursor = self.conn.cursor()
        
        # Get frame count
        cursor.execute("SELECT COUNT(*) FROM Frames")
        result = cursor.fetchone()
        if result is not None:
            self._frame_count = int(result[0])
        else:
            self._frame_count = 0
        
        # Get frame positions if this is a MALDI dataset
        table_exists_query: str = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='MaldiFrameInfo'
        """
        cursor.execute(table_exists_query)
        if cursor.fetchone():
            cursor.execute("SELECT XIndexPos, YIndexPos FROM MaldiFrameInfo")
            self._frame_positions = np.array(cursor.fetchall(), dtype=np.int32)
        else:
            # For non-MALDI data, we'll determine dimensions differently
            self._frame_positions = None
            
        cursor.close()
    
    def _execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
        """
        Execute a SQL query against the database.
   
        Args:
            query: SQL query to execute
            params: Parameters for the query
            
        Returns:
            List of result rows
            
        Raises:
            RuntimeError: If database connection is not initialized
            sqlite3.Error: If query execution fails
        """
        if self.conn is None:
            # Reopen database connection if needed
            db_path: Path = self.tsf_path if self.file_type == "tsf" else self.tdf_path
            self.conn = sqlite3.connect(str(db_path))
            # Enable SQLite performance optimizations
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = 10000")
            self.conn.execute("PRAGMA temp_store = MEMORY")
            
        cursor: sqlite3.Cursor = self.conn.cursor()
        cursor.execute(query, params)
        results: List[Tuple[Any, ...]] = cursor.fetchall()
        cursor.close()
        return results
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the Bruker dataset.
        
        Returns:
            Dictionary of metadata
            
        Raises:
            RuntimeError: If _frame_count is not initialized
        """
        if self._metadata is None:
            if self._frame_count is None:
                raise RuntimeError("Frame count not initialized")
                
            self._metadata = {
                'source': str(self.analysis_directory),
                'file_type': self.file_type,
                'frame_count': self._frame_count
            }
            
            # Add global metadata
            try:
                rows: List[Tuple[Any, ...]] = self._execute_query("SELECT Key, Value FROM GlobalMetadata")
                for key, value in rows:
                    self._metadata[str(key)] = value
            except sqlite3.OperationalError:
                # If GlobalMetadata table doesn't exist
                pass
                
        return self._metadata
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """
        Return the dimensions of the MSI dataset (x, y, z) using 0-based indexing.
        
        Returns:
            Tuple containing (width, height, depth)
            
        Raises:
            RuntimeError: If frame count is not initialized
        """
        if self._dimensions is None:
            if self._frame_positions is not None:
                # For MALDI data, use the max XY positions
                # Adding 1 to convert from 0-based max index to dimension size
                x_max: int = int(np.max(self._frame_positions[:, 0])) + 1
                y_max: int = int(np.max(self._frame_positions[:, 1])) + 1
                z_max: int = 1  # Assume 2D data for now
                self._dimensions = (x_max, y_max, z_max)
            else:
                # For non-MALDI data, just use frame count for now
                # This is likely LC data, so dimensions aren't as meaningful
                if self._frame_count is None:
                    raise RuntimeError("Frame count not initialized")
                self._dimensions = (self._frame_count, 1, 1)
                
        return self._dimensions
    
    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """
        Return the common mass axis composed of all unique m/z values across spectra.
        
        This collects all unique m/z values in the dataset to create an accurate
        common mass axis, rather than using a linearly spaced approximation.
        
        Returns:
            Array of m/z values
            
        Raises:
            RuntimeError: If frame count is not initialized
        """
        if self._common_mass_axis is None:
            logging.info("Building common mass axis from all unique m/z values")
            
            if self._frame_count is None:
                raise RuntimeError("Frame count not initialized")
            
            # Collect all m/z values with progress tracking
            all_mzs: List[NDArray[np.float64]] = []
            with tqdm(total=self._frame_count, desc="Building common mass axis", unit="frame") as pbar:
                for frame_id in range(1, self._frame_count + 1):
                    try:
                        mzs, _ = self._get_spectrum_data(frame_id)
                        if mzs.size > 0:
                            all_mzs.append(mzs)
                    except Exception as e:
                        logging.warning(f"Error reading spectrum for frame {frame_id}: {e}")
                    pbar.update(1)

            if all_mzs:
                # Concatenate and find unique values
                try:
                    combined_mzs: NDArray[np.float64] = np.concatenate(all_mzs)
                    self._common_mass_axis = np.unique(combined_mzs)
                    logging.info(f"Created common mass axis with {len(self._common_mass_axis)} unique m/z values")
                except Exception as e:
                    logging.warning(f"Error creating common mass axis: {e}")
                    self._common_mass_axis = np.array([], dtype=np.float64)
            else:
                # Fallback if no spectra were found
                self._common_mass_axis = np.array([], dtype=np.float64)
                logging.warning("No spectra found to build common mass axis")
        
        return self._common_mass_axis
    
    def _get_spectrum_data(self, frame_id: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the m/z and intensity arrays for a specific frame.
        
        Args:
            frame_id: ID of the frame
            
        Returns:
            Tuple of (m/z array, intensity array)
            
        Raises:
            RuntimeError: If handle is not initialized
        """
        if self.handle is None:
            raise RuntimeError("Handle not initialized")
            
        try:
            if self.file_type == "tsf":
                return self._get_tsf_spectrum(frame_id)
            else:  # TDF
                return self._get_tdf_spectrum(frame_id)
        except Exception as e:
            logging.warning(f"Error reading spectrum for frame {frame_id}: {e}")
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    def _get_tsf_spectrum(self, frame_id: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get spectrum data from a TSF file.
        
        Args:
            frame_id: ID of the frame
            
        Returns:
            Tuple of (m/z array, intensity array)
            
        Raises:
            RuntimeError: If DLL or handle is not initialized
        """
        if self._dll is None:
            raise RuntimeError("DLL not loaded")
        if self.handle is None:
            raise RuntimeError("Handle not initialized")
            
        # Use instance variable for buffer size
        buffer_size: int = self._line_buffer_size
        
        while True:
            # Allocate buffers (try to reuse existing ones)
            if hasattr(self, '_mz_indices_buffer') and self._mz_indices_buffer is not None and len(self._mz_indices_buffer) >= buffer_size:
                mz_indices: NDArray[np.float64] = self._mz_indices_buffer
            else:
                self._mz_indices_buffer = np.empty(buffer_size, dtype=np.float64)
                mz_indices = self._mz_indices_buffer
            
            if hasattr(self, '_intensities_buffer') and self._intensities_buffer is not None and len(self._intensities_buffer) >= buffer_size:
                intensities: NDArray[np.float64] = self._intensities_buffer
            else:
                self._intensities_buffer = np.empty(buffer_size, dtype=np.float64)
                intensities = self._intensities_buffer
            
            # Read spectrum
            result: int = self._dll.tsf_read_line_spectrum_v2(
                self.handle, 
                frame_id, 
                mz_indices.ctypes.data_as(POINTER(c_double)),
                intensities.ctypes.data_as(POINTER(c_float)),
                buffer_size
            )
            
            if result < 0:
                self._throw_last_error()

            if result > buffer_size:
                # Buffer too small, resize and try again
                buffer_size = result
                self._line_buffer_size = buffer_size  # Remember for next time
            else:
                # Success - only copy the data we need
                if result == 0:
                    return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
                    
                # Fast path: directly use the converted mzs
                mzs: NDArray[np.float64] = self._convert_indices_to_mz(frame_id, mz_indices[:result])
                intensities_result: NDArray[np.float64] = intensities[:result].copy()  # Need to copy as buffer is reused
                
                return mzs, intensities_result
    
    def _get_tdf_spectrum(self, frame_id: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get spectrum data from a TDF file.
        
        Args:
            frame_id: ID of the frame
            
        Returns:
            Tuple of (m/z array, intensity array)
            
        Raises:
            RuntimeError: If handle is not initialized
        """
        if self.handle is None:
            raise RuntimeError("Handle not initialized")
            
        # Get scan count from cache or query
        scan_count: int
        if frame_id in self._scan_count_cache:
            scan_count = self._scan_count_cache[frame_id]
        else:
            scan_count = self._get_scan_count(frame_id)
            self._scan_count_cache[frame_id] = scan_count
        
        if scan_count == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        
        # Use a preallocated buffer to avoid continuous allocations
        # Estimate buffer sizes based on scan count
        estimated_peaks: int = scan_count * 10  # Rough estimate: 10 peaks per scan
        
        # Initialize or resize buffers if needed
        if not hasattr(self, '_tdf_mz_buffer') or self._tdf_mz_buffer is None or len(self._tdf_mz_buffer) < estimated_peaks:
            self._tdf_mz_buffer = np.zeros(estimated_peaks, dtype=np.float64)
            
        if not hasattr(self, '_tdf_intensity_buffer') or self._tdf_intensity_buffer is None or len(self._tdf_intensity_buffer) < estimated_peaks:
            self._tdf_intensity_buffer = np.zeros(estimated_peaks, dtype=np.float64)
        
        # Read scans (simplified version without mobility data)
        scans: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = self._read_scans(frame_id -1, 0, scan_count)
        
        # Use direct buffer filling instead of list appending
        current_idx: int = 0
        for scan_indices, scan_intensities in scans:
            n_peaks: int = len(scan_indices)
            if n_peaks > 0:
                # Convert indices to m/z values
                scan_mzs: NDArray[np.float64] = self._convert_indices_to_mz(frame_id, scan_indices)
                
                # Ensure buffer is large enough
                while current_idx + n_peaks > len(self._tdf_mz_buffer):  # type: ignore
                    self._tdf_mz_buffer = np.resize(self._tdf_mz_buffer, len(self._tdf_mz_buffer) * 2)  # type: ignore
                    self._tdf_intensity_buffer = np.resize(self._tdf_intensity_buffer, len(self._tdf_intensity_buffer) * 2)  # type: ignore
                
                # Copy data to buffers
                self._tdf_mz_buffer[current_idx:current_idx + n_peaks] = scan_mzs  # type: ignore
                self._tdf_intensity_buffer[current_idx:current_idx + n_peaks] = scan_intensities  # type: ignore
                current_idx += n_peaks
        
        if current_idx == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        
        # Get views of the filled portions of the buffers
        mzs: NDArray[np.float64] = cast(NDArray[np.float64], self._tdf_mz_buffer[:current_idx].copy()) # type: ignore
        intensities: NDArray[np.float64] = cast(NDArray[np.float64], self._tdf_intensity_buffer[:current_idx].copy()) # type: ignore

        # Specify sort index type
        sort_idx: NDArray[np.int_] = np.argsort(mzs)
        mzs = mzs[sort_idx]
        intensities = intensities[sort_idx]
        
        return mzs, intensities
    
    def _get_scan_count(self, frame_id: int) -> int:
        """
        Get the number of scans for a frame.
        
        Args:
            frame_id: ID of the frame
            
        Returns:
            Number of scans
        """
        try:
            result: List[Tuple[int]] = self._execute_query(
                "SELECT NumScans FROM Frames WHERE Id = ?", 
                (frame_id,)
            )
            return int(result[0][0]) if result else 0
        except Exception as e:
            logging.warning(f"Error getting scan count for frame {frame_id}: {e}")
            return 0
    
    def _read_scans(self, frame_id: int, scan_begin: int, scan_end: int) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Read a range of scans from a TDF file with optimizations.
        
        Args:
            frame_id: ID of the frame
            scan_begin: First scan to read
            scan_end: Last scan to read (exclusive)
            
        Returns:
            List of (indices, intensities) tuples
            
        Raises:
            RuntimeError: If DLL or handle is not initialized
        """
        if self._dll is None:
            raise RuntimeError("DLL not loaded")
        if self.handle is None:
            raise RuntimeError("Handle not initialized")
            
        # Use instance variable for buffer size
        buffer_size: int = self._scan_buffer_size
        
        # Try to reuse buffer if possible
        if hasattr(self, '_scan_buffer') and self._scan_buffer is not None and len(self._scan_buffer) >= buffer_size:
            buffer: NDArray[np.uint32] = self._scan_buffer
        else:
            buffer = np.empty(buffer_size, dtype=np.uint32)
            self._scan_buffer = buffer
        
        while True:
            # Read scans
            required_len: int = self._dll.tims_read_scans_v2(
                self.handle,
                frame_id,  # 0-based frame index
                scan_begin,
                scan_end,
                buffer.ctypes.data_as(POINTER(c_uint32)),
                buffer_size * 4  # Size in bytes
            )
            
            if required_len == 0:
                self._throw_last_error()
                
            if required_len > buffer_size * 4:
                # Buffer too small, resize and try again
                buffer_size = (required_len // 4) + 1
                self._scan_buffer_size = buffer_size  # Remember for next time
                buffer = np.empty(buffer_size, dtype=np.uint32)
                self._scan_buffer = buffer
            else:
                break
        
        # Parse the buffer efficiently
        result: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        offset: int = scan_end - scan_begin
        
        for i in range(scan_begin, scan_end):
            idx: int = i - scan_begin
            if idx >= len(buffer):
                break
                
            peak_count: int = int(buffer[idx])
            if peak_count > 0:
                if offset + peak_count > len(buffer):
                    # Buffer overrun protection
                    break
                    
                indices: NDArray[np.float64] = buffer[offset:offset + peak_count].astype(np.float64, copy=True)
                offset += peak_count
                
                if offset + peak_count > len(buffer):
                    # Buffer overrun protection
                    break
                    
                intensities: NDArray[np.float64] = buffer[offset:offset + peak_count].astype(np.float64, copy=True)
                offset += peak_count
                
                result.append((indices, intensities))
            else:
                # Skip empty scans
                offset += peak_count * 2
        
        return result
    
    def _convert_indices_to_mz(self, frame_id: int, indices: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert mass indices to m/z values.
        
        Args:
            frame_id: ID of the frame
            indices: Array of mass indices
            
        Returns:
            Array of m/z values
            
        Raises:
            RuntimeError: If DLL or handle is not initialized
        """
        if self._dll is None:
            raise RuntimeError("DLL not loaded")
        if self.handle is None:
            raise RuntimeError("Handle not initialized")
            
        if indices.size == 0:
            return np.array([], dtype=np.float64)
            
        # Allocate output array
        mzs: NDArray[np.float64] = np.empty_like(indices)
        
        # Call the appropriate conversion function
        if self.file_type == "tsf":
            func = self._dll.tsf_index_to_mz
        else:  # TDF
            func = self._dll.tims_index_to_mz
            
        success: int = func(
            self.handle,
            frame_id,
            indices.ctypes.data_as(POINTER(c_double)),
            mzs.ctypes.data_as(POINTER(c_double)),
            indices.size
        )
        
        if success == 0:
            self._throw_last_error()
            
        return mzs
    
    def _preload_metadata(self) -> None:
        """
        Preload all metadata for faster access.
        This can significantly speed up processing by eliminating repeated database queries.
        
        Raises:
            RuntimeError: If database connection is not initialized
        """
        if self.conn is None:
            raise RuntimeError("Database connection not initialized")
            
        logging.info("Preloading metadata for faster access")
        
        # Preload frame count and positions
        cursor: sqlite3.Cursor = self.conn.cursor()
        
        # Get frame count
        cursor.execute("SELECT COUNT(*) FROM Frames")
        result = cursor.fetchone()
        if result is not None:
            self._frame_count = int(result[0])
        else:
            self._frame_count = 0
        
        # Check if this is a MALDI dataset
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='MaldiFrameInfo'")
        has_maldi: bool = cursor.fetchone() is not None
        
        # Preload all frame positions if this is MALDI data
        if has_maldi:
            cursor.execute("SELECT Frame, XIndexPos, YIndexPos FROM MaldiFrameInfo")
            self._position_cache = {int(row[0]): (int(row[1]), int(row[2])) for row in cursor.fetchall()}
            
            # Also create frame_positions array for legacy support
            if self._position_cache:
                max_frame: int = max(self._position_cache.keys())
                self._frame_positions = np.zeros((max_frame, 2), dtype=np.int32)
                for frame, (x, y) in self._position_cache.items():
                    if frame - 1 < len(self._frame_positions):
                        self._frame_positions[frame - 1] = [x, y]
        
        # Preload dimensions
        if self._position_cache:
            x_coords: List[int] = [x for x, _ in self._position_cache.values()]
            y_coords: List[int] = [y for _, y in self._position_cache.values()]
            if x_coords and y_coords:
                x_max: int = max(x_coords) + 1
                y_max: int = max(y_coords) + 1
                self._dimensions = (x_max, y_max, 1)
        
        # Preload global metadata
        try:
            cursor.execute("SELECT Key, Value FROM GlobalMetadata")
            metadata: Dict[str, Any] = {row[0]: row[1] for row in cursor.fetchall()}
            self._metadata = {
                'source': str(self.analysis_directory),
                'file_type': self.file_type,
                'frame_count': self._frame_count,
                **metadata
            }
        except sqlite3.OperationalError:
            # If GlobalMetadata table doesn't exist
            self._metadata = {
                'source': str(self.analysis_directory),
                'file_type': self.file_type,
                'frame_count': self._frame_count
            }
        
        cursor.close()
        
        # Preload common mass axis
        _ = self.get_common_mass_axis()
        
        logging.info(
            f"Metadata preloaded: {self._frame_count} frames, "
            f"dimensions: {self._dimensions}, "
            f"common mass axis: {len(self._common_mass_axis) if self._common_mass_axis is not None else 0} points"
        )
    
    def iter_spectra(
        self, 
        batch_size: Optional[int] = None
    ) -> Generator[Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]], None, None]:
        """
        Iterate through all spectra in the dataset, with optional batching for efficiency.
        
        For each spectrum, yields the coordinates and raw m/z and intensity values.
        All coordinates are 0-based for internal consistency.
        
        Args:
            batch_size: Number of spectra to process in each batch (None for default)
        
        Yields:
            Tuple containing:
                - Coordinates (x, y, z) using 0-based indexing
                - m/z values array
                - Intensity values array
                
        Raises:
            ValueError: If reader is not initialized or already closed
        """
        if not hasattr(self, 'handle') or self.handle is None:
            raise ValueError("Reader not initialized or already closed")
        
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.default_batch_size
        
        if self._frame_count is None:
            raise ValueError("Frame count not initialized")
            
        total_spectra: int = self._frame_count
        dimensions: Tuple[int, int, int] = self.get_dimensions()
        total_pixels: int = dimensions[0] * dimensions[1] * dimensions[2]
        
        # Log information about spectra vs pixels
        logging.info(f"Processing {total_spectra} spectra in a grid of {total_pixels} pixels")
        
        # Determine if this is a MALDI dataset
        is_maldi: bool = bool(self._position_cache) or self._frame_positions is not None
        
        # Process in batches with standardized progress bar
        with tqdm(total=total_spectra, desc="Reading spectra", unit="spectrum") as pbar:
            if batch_size <= 1:
                # Process one at a time
                for frame_id in range(1, self._frame_count + 1):
                    try:
                        # Get coordinates from cache if possible (already 0-based in cache)
                        if is_maldi:
                            coords: Tuple[int, int, int]
                            if self._position_cache and frame_id in self._position_cache:
                                x, y = self._position_cache[frame_id]
                                coords = (x, y, 0)
                            elif self._frame_positions is not None and frame_id - 1 < len(self._frame_positions):
                                x = int(self._frame_positions[frame_id - 1][0])
                                y = int(self._frame_positions[frame_id - 1][1])
                                coords = (x, y, 0)
                            else:
                                pbar.update(1)
                                continue  # Skip if position not found
                        else:
                            # For non-MALDI, use frame ID (0-based)
                            coords = (frame_id - 1, 0, 0)
                        
                        # Get spectrum data
                        mzs, intensities = self._get_spectrum_data(frame_id)
                        
                        if mzs.size > 0 and intensities.size > 0:
                            yield coords, mzs, intensities
                        
                        pbar.update(1)
                    except Exception as e:
                        logging.warning(f"Error processing frame {frame_id}: {e}")
                        pbar.update(1)
                        continue
            else:
                # Process in batches
                for batch_start in range(1, self._frame_count + 1, batch_size):
                    batch_end: int = min(batch_start + batch_size, self._frame_count + 1)
                    batch_size_actual: int = batch_end - batch_start
                    
                    # Process each frame in the batch
                    for offset in range(batch_size_actual):
                        frame_id: int = batch_start + offset
                        try:
                            # Get coordinates (already 0-based in cache)
                            if is_maldi:
                                coords: Tuple[int, int, int]
                                if self._position_cache and frame_id in self._position_cache:
                                    x, y = self._position_cache[frame_id]
                                    coords = (int(x), int(y), 0)
                                elif self._frame_positions is not None and frame_id - 1 < len(self._frame_positions):
                                    x = int(self._frame_positions[frame_id - 1][0])
                                    y = int(self._frame_positions[frame_id - 1][1])
                                    coords = (x, y, 0)
                                else:
                                    pbar.update(1)
                                    continue  # Skip if position not found
                            else:
                                # For non-MALDI, use frame ID (0-based)
                                coords = (frame_id - 1, 0, 0)
                            
                            # Get spectrum data
                            mzs, intensities = self._get_spectrum_data(frame_id)
                            
                            if mzs.size > 0 and intensities.size > 0:
                                yield coords, mzs, intensities
                            
                            pbar.update(1)
                        except Exception as e:
                            logging.warning(f"Error processing frame {frame_id}: {e}")
                            pbar.update(1)
                            continue
    
    def close(self) -> None:
        """
        Close all open file handles and connections.
        """
        # Close database connection
        if self.conn:
            self.conn.close()
            self.conn = None
        
        # Close data handle
        if self.handle and self._dll:
            if self.file_type == "tsf":
                self._dll.tsf_close(self.handle)
            else:  # TDF
                self._dll.tims_close(self.handle)
            self.handle = None