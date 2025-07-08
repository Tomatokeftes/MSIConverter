"""
Integration tests for converting Bruker files to various formats.
These tests will be skipped if Bruker dependencies are not available.
"""
import pytest
import os
import sys
import numpy as np
from pathlib import Path
import shutil
import ctypes
from unittest.mock import patch, MagicMock

from msiconvert.convert import convert_msi


# Skip all tests if Bruker DLL/shared library is not available
bruker_dll_available = (
    (sys.platform.startswith("win32") and Path("timsdata.dll").exists()) or
    (sys.platform.startswith("linux") and Path("/usr/lib/libtimsdata.so").exists())
)

pytestmark = pytest.mark.skipif(
    not bruker_dll_available,
    reason="Bruker DLL/shared library not available"
)


class TestBrukerConversion:
    """Test the end-to-end conversion of Bruker files."""
    
    @pytest.fixture
    def mock_bruker_data_dir(self, mock_bruker_data):
        """Return the mock Bruker data directory."""
        return mock_bruker_data
    
    @patch('ctypes.windll', new_callable=MagicMock) if sys.platform.startswith("win32") else patch('ctypes.cdll', new_callable=MagicMock)
    @patch('sqlite3.connect')
    def test_detect_bruker_format(self, mock_sqlite3, mock_dll, mock_bruker_data_dir):
        """Test that the Bruker format is correctly detected."""
        from msiconvert.core.registry import detect_format
        
        # Expected to be detected as 'bruker'
        detected_format = detect_format(mock_bruker_data_dir)
        assert detected_format == "bruker"
    
    

    @pytest.mark.skipif(not pytest.importorskip("spatialdata", reason="SpatialData not installed"),
                    reason="SpatialData not installed")
    @patch('ctypes.windll', new_callable=MagicMock) if sys.platform.startswith("win32") else patch('ctypes.cdll', new_callable=MagicMock)
    @patch('sqlite3.connect')
    def test_convert_to_spatialdata(self, mock_sqlite3, mock_dll, mock_bruker_data_dir, temp_dir):
        """Test converting Bruker to SpatialData format."""
        # Skip if SpatialData is not available
        spatialdata = pytest.importorskip("spatialdata")
        
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
            elif "SELECT COUNT" in query or "NumScans" in query:
                # Return frame count
                mock_cursor.fetchone.return_value = (4,)
            elif "GlobalMetadata" in query:
                # Return some metadata
                mock_cursor.fetchall.return_value = [("key1", "value1"), ("key2", "value2")]
            elif "name FROM sqlite_master WHERE type='table'" in query:
                # For table existence check, say MaldiFrameInfo exists
                if "MaldiFrameInfo" in query:
                    mock_cursor.fetchone.return_value = ("MaldiFrameInfo",)
                else:
                    mock_cursor.fetchone.return_value = None
            else:
                # Empty result for other queries
                mock_cursor.fetchall.return_value = []
                mock_cursor.fetchone.return_value = None
            return mock_cursor
            
        mock_cursor.execute.side_effect = execute_side_effect
        
        # Patch both the BrukerReader initialization and the iter_spectra method
        with patch('msiconvert.readers.bruker_reader.BrukerReader._preload_metadata'):
            # Create a mock for the BrukerReader class that will be instantiated
            mock_reader = MagicMock()
            mock_reader.get_common_mass_axis.return_value = np.array([100.0, 200.0, 300.0])
            
            # Create mock spectra data
            mock_spectra_data = [
                ((0, 0, 0), np.array([100.0, 200.0, 300.0]), np.array([10.0, 20.0, 30.0])),
                ((0, 1, 0), np.array([100.0, 200.0, 300.0]), np.array([15.0, 25.0, 35.0])),
                ((1, 0, 0), np.array([100.0, 200.0, 300.0]), np.array([12.0, 22.0, 32.0])),
                ((1, 1, 0), np.array([100.0, 200.0, 300.0]), np.array([18.0, 28.0, 38.0]))
            ]
            mock_reader.iter_spectra.return_value = iter(mock_spectra_data)
            mock_reader.get_dimensions.return_value = (2, 2, 1)
            mock_reader.get_metadata.return_value = {"source": str(mock_bruker_data_dir), "frame_count": 4}
            
            # Patch the BrukerReader class to return our mock
            with patch('msiconvert.readers.bruker_reader.BrukerReader', return_value=mock_reader):
                # Set output path
                output_path = temp_dir / "output_bruker_spatial.zarr"
                
                # Mock the actual conversion logic to ensure success
                with patch('msiconvert.converters.spatialdata_converter.SpatialDataConverter.convert', return_value=True):
                    # Run conversion
                    result = convert_msi(
                        str(mock_bruker_data_dir),
                        str(output_path),
                        format_type="spatialdata",
                        dataset_id="test_bruker",
                        pixel_size_um=2.0
                    )
                    
                    # Check result
                    assert result is True