"""
Test suite for Bruker reader after simplification.

This test suite validates:
- Raw mass axis building
- Direct spectrum iteration
- Direct coordinate extraction
- Core functionality after removing complex utilities
"""

import numpy as np
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from msiconvert.readers.bruker.bruker_reader import BrukerReader


class TestBrukerReader:
    """Test the Bruker reader core functionality."""

    @patch('msiconvert.readers.bruker.bruker_reader.DLLManager')
    @patch('msiconvert.readers.bruker.bruker_reader.SDKFunctions')
    def test_initialization(self, mock_sdk_functions, mock_dll_manager):
        """Test that reader initialization doesn't create BatchProcessor or MemoryManager."""
        mock_data_path = Path("/fake/bruker.d")
        
        # Mock the required components
        mock_dll_manager.return_value = MagicMock()
        mock_sdk = MagicMock()
        mock_sdk_functions.return_value = mock_sdk
        mock_sdk.open_file.return_value = MagicMock()
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch('sqlite3.connect') as mock_connect:
            
            # Mock database connection
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            reader = BrukerReader(mock_data_path)
            
            # Verify that these attributes don't exist
            assert not hasattr(reader, 'batch_processor')
            assert not hasattr(reader, 'memory_manager')
            assert not hasattr(reader, 'coordinate_cache')
            
            # Should have essential components
            assert hasattr(reader, 'sdk')
            assert hasattr(reader, 'db_path')

    def test_build_raw_mass_axis_function_exists(self):
        """Test that build_raw_mass_axis function is available and works."""
        from msiconvert.readers.bruker.bruker_reader import build_raw_mass_axis
        
        # Test with mock data
        def mock_spectra_iterator():
            yield (0, 0, 0), np.array([100.0, 200.0]), np.array([10.0, 20.0])
            yield (0, 0, 1), np.array([150.0, 250.0]), np.array([15.0, 25.0])
        
        mass_axis = build_raw_mass_axis(mock_spectra_iterator())
        expected = np.array([100.0, 150.0, 200.0, 250.0])
        np.testing.assert_array_equal(mass_axis, expected)

    @patch('msiconvert.readers.bruker.bruker_reader.DLLManager')
    @patch('msiconvert.readers.bruker.bruker_reader.SDKFunctions')
    def test_get_common_mass_axis(self, mock_sdk_functions, mock_dll_manager):
        """Test that get_common_mass_axis works without complex mass axis builder."""
        mock_data_path = Path("/fake/bruker.d")
        
        # Mock the required components
        mock_dll_manager.return_value = MagicMock()
        mock_sdk = MagicMock()
        mock_sdk_functions.return_value = mock_sdk
        mock_sdk.open_file.return_value = MagicMock()
        
        # Mock database connection
        with patch('sqlite3.connect') as mock_connect, \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True):
            
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            reader = BrukerReader(mock_data_path)
            
            # Mock _iter_spectra_raw to return test data
            def mock_iter_spectra():
                yield (0, 0, 0), np.array([100.0, 200.0]), np.array([10.0, 20.0])
                yield (0, 0, 1), np.array([150.0, 250.0]), np.array([15.0, 25.0])
            
            reader._iter_spectra_raw = mock_iter_spectra
            reader._get_frame_count = lambda: 2
            
            # Test get_common_mass_axis
            mass_axis = reader.get_common_mass_axis()
            
            # Should contain unique m/z values
            assert len(mass_axis) > 0
            # Should be sorted
            assert np.array_equal(mass_axis, np.sort(mass_axis))

    def test_spectrum_iteration(self):
        """Test that spectrum iteration works without BatchProcessor."""
        # This will be implemented after Phase 1
        # For now, just ensure the structure is testable
        assert True


class TestRawMassAxisBuilder:
    """Test the raw mass axis building function."""
    
    def test_build_raw_mass_axis_basic(self):
        """Test basic raw mass axis building functionality."""
        def mock_spectra_iterator():
            yield (0, 0, 0), np.array([100.0, 200.0, 300.0]), np.array([10.0, 20.0, 30.0])
            yield (0, 0, 1), np.array([150.0, 200.0, 350.0]), np.array([15.0, 20.0, 35.0])
            yield (0, 0, 2), np.array([100.0, 250.0]), np.array([10.0, 25.0])
        
        from msiconvert.readers.bruker.bruker_reader import build_raw_mass_axis
        mass_axis = build_raw_mass_axis(mock_spectra_iterator())
        
        expected = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
        np.testing.assert_array_equal(mass_axis, expected)

    def test_build_raw_mass_axis_empty_input(self):
        """Test raw mass axis building with empty input."""
        def empty_iterator():
            return
            yield  # Never reached
        
        from msiconvert.readers.bruker.bruker_reader import build_raw_mass_axis
        mass_axis = build_raw_mass_axis(empty_iterator())
        
        assert len(mass_axis) == 0

    def test_build_raw_mass_axis_with_progress_callback(self):
        """Test raw mass axis building with progress callback."""
        progress_calls = []
        
        def progress_callback(count):
            progress_calls.append(count)
        
        def mock_spectra_iterator():
            for i in range(250):  # Should trigger progress callback
                yield (0, 0, i), np.array([100.0 + i]), np.array([10.0])
        
        from msiconvert.readers.bruker.bruker_reader import build_raw_mass_axis
        mass_axis = build_raw_mass_axis(mock_spectra_iterator(), progress_callback)
        
        # Should have called progress callback
        assert len(progress_calls) > 0
        assert 100 in progress_calls  # Should call at multiples of 100
        assert 200 in progress_calls


class TestDirectCoordinateExtraction:
    """Test direct coordinate extraction without CoordinateCache."""
    
    def test_get_frame_coordinates_maldi_with_offsets(self):
        """Test direct coordinate extraction for MALDI data with normalization."""
        from msiconvert.readers.bruker.bruker_reader import _get_frame_coordinates
        
        # Mock database with MALDI data
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock MALDI coordinate query
            mock_cursor.fetchone.return_value = (15, 25)  # X=15, Y=25 for frame
            
            db_path = Path("/fake/analysis.tsf")
            coordinate_offsets = (5, 15, 0)  # Offsets from metadata
            coords = _get_frame_coordinates(db_path, 1, coordinate_offsets)
            
            # Should return normalized coordinates: (15-5, 25-15, 0) = (10, 10, 0)
            assert coords == (10, 10, 0)

    def test_get_frame_coordinates_maldi_without_offsets(self):
        """Test direct coordinate extraction for MALDI data without normalization."""
        from msiconvert.readers.bruker.bruker_reader import _get_frame_coordinates
        
        # Mock database with MALDI data
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock MALDI coordinate query
            mock_cursor.fetchone.return_value = (15, 25)  # X=15, Y=25 for frame
            
            db_path = Path("/fake/analysis.tsf")
            coords = _get_frame_coordinates(db_path, 1)  # No offsets provided
            
            # Should return raw coordinates: (15, 25, 0)
            assert coords == (15, 25, 0)
    
    def test_get_frame_coordinates_non_maldi(self):
        """Test direct coordinate extraction for non-MALDI data."""
        from msiconvert.readers.bruker.bruker_reader import _get_frame_coordinates
        
        # Mock database without MALDI data
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock no MALDI table
            mock_cursor.execute.side_effect = [sqlite3.OperationalError("no such table")]
            
            db_path = Path("/fake/analysis.tsf")
            coords = _get_frame_coordinates(db_path, 5)
            
            # Should return generated coordinates for non-MALDI (frame_id-1, 0, 0)
            assert coords == (4, 0, 0)
    
    def test_get_frame_count_direct(self):
        """Test direct frame count extraction."""
        from msiconvert.readers.bruker.bruker_reader import _get_frame_count
        
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock frame count query
            mock_cursor.fetchone.return_value = [100]
            
            db_path = Path("/fake/analysis.tsf")
            count = _get_frame_count(db_path)
            
            assert count == 100
            mock_cursor.execute.assert_called_with("SELECT COUNT(*) FROM Frames")


class TestReaderInterface:
    """Test that the reader maintains required interfaces."""

    def test_registration_works(self):
        """Test that the reader is properly registered."""
        from msiconvert.core.registry import get_reader_class
        
        reader_class = get_reader_class("bruker")
        assert reader_class == BrukerReader

    @patch('msiconvert.readers.bruker.bruker_reader.DLLManager')
    @patch('msiconvert.readers.bruker.bruker_reader.SDKFunctions')
    def test_required_methods_exist(self, mock_sdk_functions, mock_dll_manager):
        """Test that all required interface methods exist."""
        mock_data_path = Path("/fake/bruker.d")
        
        # Mock the required components
        mock_dll_manager.return_value = MagicMock()
        mock_sdk = MagicMock()
        mock_sdk_functions.return_value = mock_sdk
        mock_sdk.open_file.return_value = MagicMock()
        
        with patch('sqlite3.connect') as mock_connect, \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True):
            
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            reader = BrukerReader(mock_data_path)
            
            # Test that all required interface methods exist
            required_methods = [
                'get_essential_metadata',
                'get_comprehensive_metadata', 
                'get_common_mass_axis',
                'iter_spectra',
                'close'
            ]
            
            for method_name in required_methods:
                assert hasattr(reader, method_name)
                assert callable(getattr(reader, method_name))