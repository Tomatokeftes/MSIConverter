"""
Tests for the Bruker reader.
Note: Full testing requires actual Bruker files and the timsdata DLL.
These tests focus on structure and mocking.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
import sys

from msiconvert.readers.bruker_reader import BrukerReader


def normalize_path(path_str):
    """
    Normalize a path for reliable comparison between different Windows path formats.
    """
    path = Path(path_str)
    # Convert to absolute, resolve any symlinks, and use proper case
    path = path.absolute().resolve()
    # Convert to string and normalize separators
    norm_path = str(path).replace('\\', '/')
    return norm_path


class TestBrukerReaderStructure:
    """Test the structure and interface of the Bruker reader."""
    
    def test_class_registration(self):
        """Test that the BrukerReader class is properly registered."""
        from msiconvert.core.registry import reader_registry
        assert "bruker" in reader_registry
        assert reader_registry["bruker"] == BrukerReader
    
    def test_interface_implementation(self):
        """Test that the BrukerReader implements the BaseMSIReader interface."""
        from msiconvert.core.base_reader import BaseMSIReader
        assert issubclass(BrukerReader, BaseMSIReader)
        
        # Check that it implements all required methods
        required_methods = [
            'get_metadata',
            'get_dimensions',
            'get_common_mass_axis',
            'iter_spectra',
            'close'
        ]
        
        for method in required_methods:
            assert hasattr(BrukerReader, method)


@pytest.mark.skipif(not (Path("timsdata.dll").exists() or 
                     Path("C:/timsdata.dll").exists()) and 
                     not Path("/usr/lib/libtimsdata.so").exists(),
                    reason="Bruker DLL/shared library not available")
class TestBrukerReaderWithMocks:
    """Test Bruker reader functionality using mocks."""
    
    @patch('ctypes.windll', new_callable=MagicMock) if sys.platform.startswith("win32") else patch('ctypes.cdll', new_callable=MagicMock)
    @patch('sqlite3.connect')
    def test_initialization(self, mock_sqlite3, mock_dll, mock_bruker_data):
        """Test initialization with mocked dependencies."""
        # Setup DLL mock
        dll_mock = MagicMock()
        if sys.platform.startswith("win32"):
            mock_dll.LoadLibrary.return_value = dll_mock
        else:
            mock_dll.LoadLibrary.return_value = dll_mock
            
        dll_mock.tsf_open.return_value = 123  # Non-zero handle
        
        # Setup SQLite mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Need to handle different SQL queries
        def execute_side_effect(query, *args, **kwargs):
            if "MaldiFrameInfo" in query and "name" not in query:
                # Return Frame, X, Y coordinates when querying MaldiFrameInfo table
                mock_cursor.fetchall.return_value = [(1, 0, 0), (2, 0, 1), (3, 1, 0), (4, 1, 1)]
            elif "SELECT COUNT" in query:
                # Return frame count for the COUNT query
                mock_cursor.fetchone.return_value = (4,)
            else:
                # Empty result for other queries
                mock_cursor.fetchall.return_value = []
                mock_cursor.fetchone.return_value = None
            return mock_cursor
            
        mock_cursor.execute.side_effect = execute_side_effect
        
        # Create reader
        reader = BrukerReader(mock_bruker_data)
        
        # Check initialization
        assert reader.handle == 123
        dll_mock.tsf_open.assert_called_once()
        
        # Clean up
        reader.close()
    
    @patch('ctypes.windll', new_callable=MagicMock) if sys.platform.startswith("win32") else patch('ctypes.cdll', new_callable=MagicMock)
    @patch('sqlite3.connect')
    def test_get_metadata(self, mock_sqlite3, mock_dll, mock_bruker_data):
        """Test getting metadata with mocked dependencies."""
        # Setup DLL mock
        dll_mock = MagicMock()
        if sys.platform.startswith("win32"):
            mock_dll.LoadLibrary.return_value = dll_mock
        else:
            mock_dll.LoadLibrary.return_value = dll_mock
            
        dll_mock.tsf_open.return_value = 123
        
        # Setup SQLite mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Need to handle different SQL queries
        def execute_side_effect(query, *args, **kwargs):
            if "MaldiFrameInfo" in query and "name" not in query:
                # Return Frame, X, Y coordinates
                mock_cursor.fetchall.return_value = [(1, 0, 0), (2, 0, 1), (3, 1, 0), (4, 1, 1)]
            elif "SELECT COUNT" in query:
                # Return frame count
                mock_cursor.fetchone.return_value = (4,)
            elif "GlobalMetadata" in query:
                # Return some metadata
                mock_cursor.fetchall.return_value = [("key1", "value1"), ("key2", "value2")]
            else:
                # Empty result for other queries
                mock_cursor.fetchall.return_value = []
                mock_cursor.fetchone.return_value = None
            return mock_cursor
            
        mock_cursor.execute.side_effect = execute_side_effect
        
        # Create reader
        reader = BrukerReader(mock_bruker_data)
        
        # Get metadata
        metadata = reader.get_metadata()
        
        # Check metadata - use normalized paths for comparison
        assert "source" in metadata
        assert normalize_path(mock_bruker_data) == normalize_path(metadata["source"])
        assert metadata["frame_count"] == 4
        assert metadata["key1"] == "value1"
        assert metadata["key2"] == "value2"
        
        # Clean up
        reader.close()
    
    @patch('ctypes.windll', new_callable=MagicMock) if sys.platform.startswith("win32") else patch('ctypes.cdll', new_callable=MagicMock)
    @patch('sqlite3.connect')
    def test_get_dimensions(self, mock_sqlite3, mock_dll, mock_bruker_data):
        """Test getting dimensions with mocked dependencies."""
        # Setup DLL mock
        dll_mock = MagicMock()
        if sys.platform.startswith("win32"):
            mock_dll.LoadLibrary.return_value = dll_mock
        else:
            mock_dll.LoadLibrary.return_value = dll_mock
            
        dll_mock.tsf_open.return_value = 123
        
        # Setup SQLite mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Need to handle different SQL queries
        def execute_side_effect(query, *args, **kwargs):
            if "MaldiFrameInfo" in query and "name" not in query:
                # Return Frame, X, Y coordinates - use 0-based indexing for positions
                mock_cursor.fetchall.return_value = [(1, 0, 0), (2, 0, 1), (3, 1, 0), (4, 1, 1)]
            elif "SELECT COUNT" in query:
                # Return frame count
                mock_cursor.fetchone.return_value = (4,)
            else:
                # Empty result for other queries
                mock_cursor.fetchall.return_value = []
                mock_cursor.fetchone.return_value = None
            return mock_cursor
            
        mock_cursor.execute.side_effect = execute_side_effect
        
        # Patch the _position_cache directly to control dimensions
        with patch.object(BrukerReader, '_preload_metadata'):
            # Create reader but skip the actual preload
            reader = BrukerReader(mock_bruker_data)
            
            # Manually set the position cache to control dimensions
            reader._position_cache = {
                1: (0, 0),
                2: (0, 1),
                3: (1, 0),
                4: (1, 1)
            }
            
            # Also set frame positions for backward compatibility
            reader._frame_positions = np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ])
            
            # Reset dimensions so it will be recalculated
            reader._dimensions = None
            
            # Get dimensions
            dimensions = reader.get_dimensions()
            
            # Check dimensions (should be 2x2x1 from our mock data)
            assert len(dimensions) == 3
            assert dimensions[0] == 2  # 0, 1
            assert dimensions[1] == 2  # 0, 1
            assert dimensions[2] == 1  # Only one Z plane
            
            # Clean up
            reader.close()
    
    @patch('ctypes.windll', new_callable=MagicMock) if sys.platform.startswith("win32") else patch('ctypes.cdll', new_callable=MagicMock)
    @patch('sqlite3.connect')
    def test_close(self, mock_sqlite3, mock_dll, mock_bruker_data):
        """Test closing the reader with mocked dependencies."""
        # Setup DLL mock
        dll_mock = MagicMock()
        if sys.platform.startswith("win32"):
            mock_dll.LoadLibrary.return_value = dll_mock
        else:
            mock_dll.LoadLibrary.return_value = dll_mock
            
        dll_mock.tsf_open.return_value = 123
        
        # Setup SQLite mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Need to handle different SQL queries
        def execute_side_effect(query, *args, **kwargs):
            if "MaldiFrameInfo" in query and "name" not in query:
                # Return Frame, X, Y coordinates
                mock_cursor.fetchall.return_value = [(1, 0, 0), (2, 0, 1), (3, 1, 0), (4, 1, 1)]
            elif "SELECT COUNT" in query:
                # Return frame count
                mock_cursor.fetchone.return_value = (4,)
            else:
                # Empty result for other queries
                mock_cursor.fetchall.return_value = []
                mock_cursor.fetchone.return_value = None
            return mock_cursor
            
        mock_cursor.execute.side_effect = execute_side_effect
        
        # Create reader
        reader = BrukerReader(mock_bruker_data)
        
        # Close reader
        reader.close()
        
        # Check close calls
        dll_mock.tsf_close.assert_called_once_with(123)
        mock_conn.close.assert_called_once()
        
        # Check internal state
        assert reader.handle is None
        assert reader.conn is None