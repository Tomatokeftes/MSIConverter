# tests/unit/converters/test_dask_streaming.py

"""
Tests for Dask streaming functionality in SpatialData converters.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from msiconvert.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
    DASK_AVAILABLE,
    SPATIALDATA_AVAILABLE
)
from msiconvert.processing.interpolation import InterpolationResult


@pytest.fixture
def mock_reader():
    """Create a mock MSI reader."""
    reader = Mock()
    
    # Mock essential metadata
    essential_metadata = Mock()
    essential_metadata.dimensions = (10, 10, 1)  # Small 10x10 dataset
    essential_metadata.coordinate_bounds = ((0, 9), (0, 9), (0, 0))
    essential_metadata.n_spectra = 100
    essential_metadata.estimated_memory_gb = 0.1
    essential_metadata.pixel_size = (25.0, 25.0)
    essential_metadata.mass_range = (100.0, 200.0)
    
    reader.get_essential_metadata.return_value = essential_metadata
    
    # Mock spectrum data
    def mock_get_spectrum(x, y, z):
        if x == 0 and y == 0:  # Empty pixel
            return None, None
        # Generate simple spectrum
        mzs = np.array([110.0, 150.0, 190.0])
        intensities = np.array([100.0 * (x + 1), 200.0 * (y + 1), 150.0])
        return mzs, intensities
    
    reader.get_spectrum = mock_get_spectrum
    reader.close = Mock()
    
    return reader


@pytest.fixture
def mock_spatialdata_converter(mock_reader):
    """Create a mock SpatialData converter for testing."""
    
    class TestConverter(BaseSpatialDataConverter):
        def _create_data_structures(self):
            return {"test": "structure"}
        
        def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
            pass
        
        def _finalize_data(self, data_structures):
            pass
    
    # Mock SpatialData availability
    with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
        converter = TestConverter(
            reader=mock_reader,
            output_path=Path("/tmp/test.zarr"),
            dataset_id="test_dataset"
        )
    
    return converter


class TestDaskDecisionLogic:
    """Test the logic for deciding when to use Dask processing."""
    
    def test_should_use_dask_processing_manual_override_true(self, mock_spatialdata_converter):
        """Test manual override to use Dask."""
        converter = mock_spatialdata_converter
        converter.use_dask = True
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.DASK_AVAILABLE', True):
            assert converter._should_use_dask_processing() is True
    
    def test_should_use_dask_processing_manual_override_false(self, mock_spatialdata_converter):
        """Test manual override to not use Dask."""
        converter = mock_spatialdata_converter
        converter.use_dask = False
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.DASK_AVAILABLE', True):
            assert converter._should_use_dask_processing() is False
    
    def test_should_use_dask_processing_unavailable(self, mock_spatialdata_converter):
        """Test that Dask is not used when unavailable."""
        converter = mock_spatialdata_converter
        converter.use_dask = None  # Auto-detect
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.DASK_AVAILABLE', False):
            assert converter._should_use_dask_processing() is False
    
    def test_should_use_dask_processing_auto_detect_large_dataset(self, mock_spatialdata_converter):
        """Test auto-detection uses Dask for large datasets."""
        converter = mock_spatialdata_converter
        converter.use_dask = None  # Auto-detect
        converter.target_memory_gb = 1.0
        
        # Initialize converter to set dimensions and mass axis
        converter._initialize_conversion()
        
        # Mock large memory requirement
        with patch.object(converter, '_estimate_memory_requirements', return_value=3.0):
            with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.DASK_AVAILABLE', True):
                assert converter._should_use_dask_processing() is True
    
    def test_should_use_dask_processing_auto_detect_small_dataset(self, mock_spatialdata_converter):
        """Test auto-detection doesn't use Dask for small datasets."""
        converter = mock_spatialdata_converter
        converter.use_dask = None  # Auto-detect
        converter.target_memory_gb = 8.0
        
        # Mock small memory requirement
        with patch.object(converter, '_estimate_memory_requirements', return_value=1.0):
            with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.DASK_AVAILABLE', True):
                assert converter._should_use_dask_processing() is False
    
    def test_should_use_chunked_processing_with_dask(self, mock_spatialdata_converter):
        """Test that chunked processing is disabled when Dask is used."""
        converter = mock_spatialdata_converter
        
        with patch.object(converter, '_should_use_dask_processing', return_value=True):
            assert converter._should_use_chunked_processing() is False


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestDaskPipelineCreation:
    """Test Dask pipeline creation and execution."""
    
    def test_determine_dask_chunk_size_manual(self, mock_spatialdata_converter):
        """Test manual Dask chunk size setting."""
        converter = mock_spatialdata_converter
        converter.dask_chunk_size = 500
        
        chunk_size = converter._determine_dask_chunk_size()
        assert chunk_size == 500
    
    def test_determine_dask_chunk_size_auto(self, mock_spatialdata_converter):
        """Test automatic Dask chunk size calculation."""
        converter = mock_spatialdata_converter
        converter.dask_chunk_size = None
        
        chunk_size = converter._determine_dask_chunk_size()
        
        # Should be reasonable value between bounds
        assert 50 <= chunk_size <= 2000
    
    def test_create_pixel_chunk_task(self, mock_spatialdata_converter):
        """Test creation of pixel chunk tasks."""
        converter = mock_spatialdata_converter
        
        # Initialize converter state
        converter._initialize_conversion()
        
        # Create a small chunk task
        task = converter._create_pixel_chunk_task(0, 5)
        
        # Should return a delayed object
        assert hasattr(task, 'compute')
        
        # Execute the task
        pixel_data = task.compute()
        
        # Should have some pixels (excluding empty (0,0) pixel)
        assert len(pixel_data) >= 3  # At least pixels (1,0), (2,0), (3,0), etc.
        
        # Check structure of returned data
        for pixel in pixel_data:
            assert 'coords' in pixel
            assert 'pixel_idx' in pixel
            assert 'mzs' in pixel
            assert 'intensities' in pixel
            assert isinstance(pixel['mzs'], np.ndarray)
            assert isinstance(pixel['intensities'], np.ndarray)
    
    def test_process_dask_chunk(self, mock_spatialdata_converter):
        """Test processing of Dask chunks through interpolation."""
        converter = mock_spatialdata_converter
        
        # Initialize converter state
        converter._initialize_conversion()
        
        # Create mock pixel data
        pixel_chunk = [
            {
                'coords': (1, 0, 0),
                'pixel_idx': 1,
                'mzs': np.array([110.0, 150.0, 190.0]),
                'intensities': np.array([100.0, 200.0, 150.0])
            },
            {
                'coords': (2, 0, 0),
                'pixel_idx': 2,
                'mzs': np.array([120.0, 160.0]),
                'intensities': np.array([300.0, 400.0])
            }
        ]
        
        # Process the chunk
        task = converter._process_dask_chunk(pixel_chunk)
        
        # Should return a delayed object
        assert hasattr(task, 'compute')
        
        # Execute the task
        results = task.compute()
        
        # Should return InterpolationResult objects
        assert len(results) == 2
        for result in results:
            assert isinstance(result, InterpolationResult)
            assert hasattr(result, 'coords')
            assert hasattr(result, 'sparse_indices')
            assert hasattr(result, 'sparse_values')
            assert hasattr(result, 'tic_value')
    
    def test_combine_dask_results(self, mock_spatialdata_converter):
        """Test combination of Dask results."""
        converter = mock_spatialdata_converter
        
        # Mock processed chunks (lists of InterpolationResult)
        mock_result1 = InterpolationResult(
            coords=(1, 0, 0),
            pixel_idx=1,
            sparse_indices=np.array([10, 50]),
            sparse_values=np.array([100.0, 200.0]),
            tic_value=300.0
        )
        mock_result2 = InterpolationResult(
            coords=(2, 0, 0),
            pixel_idx=2,
            sparse_indices=np.array([20, 60]),
            sparse_values=np.array([150.0, 250.0]),
            tic_value=400.0
        )
        
        processed_chunks = [[mock_result1], [mock_result2]]
        
        # Combine results
        task = converter._combine_dask_results(processed_chunks)
        
        # Should return a delayed object
        assert hasattr(task, 'compute')
        
        # Execute the task
        combined = task.compute()
        
        # Should have all results combined
        assert len(combined) == 2
        assert combined[0] is mock_result1
        assert combined[1] is mock_result2
    
    def test_create_dask_pipeline(self, mock_spatialdata_converter):
        """Test creation of complete Dask pipeline."""
        converter = mock_spatialdata_converter
        
        # Initialize converter state
        converter._initialize_conversion()
        
        # Create mock data structures
        data_structures = {"test": "structure"}
        
        # Create pipeline
        pipeline = converter._create_dask_pipeline(data_structures)
        
        # Should return a delayed object
        assert hasattr(pipeline, 'compute')
        
        # Execute pipeline (this will process actual data from mock reader)
        results = pipeline.compute()
        
        # Should return list of InterpolationResult objects
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, InterpolationResult)


class TestDaskIntegration:
    """Test integration of Dask processing with converter workflow."""
    
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_process_with_dask(self, mock_spatialdata_converter):
        """Test full Dask processing workflow."""
        converter = mock_spatialdata_converter
        
        # Initialize converter state
        converter._initialize_conversion()
        
        # Create mock data structures
        data_structures = {"test": "structure"}
        
        # Mock the specific result processing
        converter._process_dask_result_specific = Mock()
        
        # Process with Dask
        converter._process_with_dask(data_structures)
        
        # Should have called result processing
        converter._process_dask_result_specific.assert_called_once()
        
        # Check that non-empty pixel count was updated
        assert converter._non_empty_pixel_count > 0
    
    def test_process_with_dask_without_dask_available(self, mock_spatialdata_converter):
        """Test that Dask processing fails gracefully without Dask."""
        converter = mock_spatialdata_converter
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.DASK_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="Dask is not available but Dask processing was requested"):
                converter._process_with_dask({})
    
    def test_process_spectra_routing_to_dask(self, mock_spatialdata_converter):
        """Test that _process_spectra routes to Dask when appropriate."""
        converter = mock_spatialdata_converter
        
        # Mock Dask processing
        converter._process_with_dask = Mock()
        
        # Force Dask usage
        with patch.object(converter, '_should_use_dask_processing', return_value=True):
            data_structures = {"test": "structure"}
            converter._process_spectra(data_structures)
        
        # Should have called Dask processing
        converter._process_with_dask.assert_called_once_with(data_structures)
    
    def test_process_spectra_routing_to_chunked(self, mock_spatialdata_converter):
        """Test that _process_spectra routes to chunked when appropriate."""
        converter = mock_spatialdata_converter
        
        # Mock chunked processing
        converter._process_chunked = Mock()
        
        # Force chunked usage (not Dask, but chunked)
        with patch.object(converter, '_should_use_dask_processing', return_value=False):
            with patch.object(converter, '_should_use_chunked_processing', return_value=True):
                data_structures = {"test": "structure"}
                converter._process_spectra(data_structures)
        
        # Should have called chunked processing
        converter._process_chunked.assert_called_once_with(data_structures)
    
    def test_process_spectra_routing_to_standard(self, mock_spatialdata_converter):
        """Test that _process_spectra routes to standard when appropriate."""
        converter = mock_spatialdata_converter
        
        # Mock parent's _process_spectra
        with patch.object(BaseSpatialDataConverter.__bases__[0], '_process_spectra') as mock_parent:
            # Force standard usage (neither Dask nor chunked)
            with patch.object(converter, '_should_use_dask_processing', return_value=False):
                with patch.object(converter, '_should_use_chunked_processing', return_value=False):
                    data_structures = {"test": "structure"}
                    converter._process_spectra(data_structures)
        
        # Should have called parent processing
        mock_parent.assert_called_once_with(data_structures)


class TestInterpolatorIntegration:
    """Test integration between converter and interpolation module."""
    
    def test_interpolator_initialization(self, mock_spatialdata_converter):
        """Test that interpolator is properly initialized."""
        converter = mock_spatialdata_converter
        
        # Initialize conversion
        converter._initialize_conversion()
        
        # Should have interpolator
        assert converter._interpolator is not None
        assert converter._interpolator.mass_axis_length > 0
        assert converter._interpolator.interpolation_method == "linear"
    
    def test_process_dask_result_specific_default(self, mock_spatialdata_converter):
        """Test default implementation of _process_dask_result_specific."""
        converter = mock_spatialdata_converter
        
        # Initialize conversion
        converter._initialize_conversion()
        
        # Mock _process_single_spectrum
        converter._process_single_spectrum = Mock()
        
        # Create mock results
        results = [
            InterpolationResult(
                coords=(1, 0, 0),
                pixel_idx=1,
                sparse_indices=np.array([10, 50]),
                sparse_values=np.array([100.0, 200.0]),
                tic_value=300.0
            )
        ]
        
        data_structures = {"test": "structure"}
        
        # Process results
        converter._process_dask_result_specific(data_structures, results)
        
        # Should have called _process_single_spectrum
        converter._process_single_spectrum.assert_called_once()
        call_args = converter._process_single_spectrum.call_args
        assert call_args[0][0] is data_structures
        assert call_args[0][1] == (1, 0, 0)  # coords


class TestErrorHandling:
    """Test error handling in Dask streaming functionality."""
    
    def test_uninitialized_dimensions_error(self, mock_spatialdata_converter):
        """Test error handling for uninitialized dimensions."""
        converter = mock_spatialdata_converter
        converter._dimensions = None
        
        with pytest.raises(ValueError, match="Dimensions are not initialized"):
            converter._create_dask_pipeline({})
    
    def test_uninitialized_interpolator_error(self, mock_spatialdata_converter):
        """Test error handling for uninitialized interpolator."""
        converter = mock_spatialdata_converter
        converter._interpolator = None
        
        with pytest.raises(ValueError, match="Interpolator is not initialized"):
            converter._process_dask_chunk([])


if __name__ == "__main__":
    pytest.main([__file__])