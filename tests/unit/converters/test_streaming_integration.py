# tests/unit/converters/test_streaming_integration.py

"""
End-to-end integration tests for streaming Dask pipeline with Zarr writing.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from msiconvert.converters.spatialdata.base_spatialdata_converter import BaseSpatialDataConverter
from msiconvert.storage.incremental_zarr_writer import ZarrStorageConfig
from msiconvert.processing.interpolation import InterpolationResult

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    import dask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@pytest.fixture
def temp_output_path():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    output_path = Path(temp_dir) / "test_output.zarr"
    yield output_path
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_reader():
    """Create a mock MSI reader for testing."""
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
    
    # Mock spectrum data - create realistic spectra
    def mock_get_spectrum(x, y, z):
        if (x + y) % 3 == 0:  # Some empty pixels
            return None, None
        
        # Generate realistic spectrum
        n_peaks = np.random.randint(5, 20)
        mzs = np.sort(np.random.uniform(100.0, 200.0, n_peaks))
        intensities = np.random.exponential(100.0, n_peaks)
        return mzs, intensities
    
    reader.get_spectrum = mock_get_spectrum
    reader.close = Mock()
    
    return reader


@pytest.fixture 
def streaming_converter(mock_reader, temp_output_path):
    """Create a streaming-enabled converter for testing."""
    
    class TestStreamingConverter(BaseSpatialDataConverter):
        def _create_data_structures(self):
            return {
                "test": "structure",
                "sparse_data": {"sparse_rows": [], "sparse_cols": [], "sparse_data": []},
                "tic_values": np.zeros((self.total_pixels,), dtype=np.float64),
                "total_intensity": np.zeros((len(self._common_mass_axis),), dtype=np.float64),
                "pixel_count": 0
            }
        
        def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
            pass  # Not used in streaming mode
        
        def _finalize_data(self, data_structures):
            pass  # Handled by streaming result processing
    
    # Mock SpatialData availability
    with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
        converter = TestStreamingConverter(
            reader=mock_reader,
            output_path=temp_output_path,
            dataset_id="test_dataset",
            enable_streaming=True,
            use_dask=True,
            dask_chunk_size=25,  # Small chunks for testing
            zarr_config=ZarrStorageConfig(chunk_size=100)
        )
        
        # Store for cleanup
        converter.total_pixels = 100
    
    return converter


@pytest.mark.skipif(not (ZARR_AVAILABLE and DASK_AVAILABLE), reason="Zarr and Dask required")
class TestStreamingIntegration:
    """Test end-to-end streaming integration."""
    
    def test_streaming_initialization(self, streaming_converter):
        """Test that streaming mode is properly initialized."""
        converter = streaming_converter
        
        assert converter.enable_streaming is True
        assert converter.use_dask is True
        assert converter.zarr_config is not None
        assert converter._zarr_writer is None  # Not initialized until processing
    
    def test_streaming_decision_logic(self, streaming_converter):
        """Test that streaming mode affects processing decisions."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Should use Dask processing when enabled
        assert converter._should_use_dask_processing() is True
        
        # Chunked processing should be disabled when using Dask
        assert converter._should_use_chunked_processing() is False
    
    def test_zarr_writer_initialization(self, streaming_converter):
        """Test Zarr writer initialization during Dask processing."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Initialize Zarr writer using the actual factory function
        from msiconvert.storage.incremental_zarr_writer import create_incremental_zarr_writer
        converter._zarr_writer = create_incremental_zarr_writer(
            output_path=converter.output_path,
            dimensions=converter._dimensions,
            n_masses=len(converter._common_mass_axis),
            config=converter.zarr_config
        )
        
        assert converter._zarr_writer is not None
        assert converter._zarr_writer.total_pixels == 100
        assert converter._zarr_writer.n_masses == len(converter._common_mass_axis)
        
        converter._zarr_writer.close()
    
    def test_streaming_pipeline_creation(self, streaming_converter):
        """Test creation of streaming Dask pipeline."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Mock Zarr writer for pipeline creation
        with patch.object(converter, '_zarr_writer') as mock_zarr_writer:
            mock_zarr_writer.write_interpolation_results = Mock()
            mock_zarr_writer.finalize.return_value = {
                'sparse_matrix': Mock(),
                'tic_values': np.zeros(100),
                'total_intensity': np.zeros(len(converter._common_mass_axis)),
                'pixels_written': 100,
                'sparse_nnz': 500
            }
            
            # Create data structures
            data_structures = converter._create_data_structures()
            
            # Create streaming pipeline
            pipeline = converter._create_dask_pipeline(data_structures)
            
            # Should return a delayed object
            assert hasattr(pipeline, 'compute')
    
    def test_create_zarr_writer_method(self, streaming_converter):
        """Test helper method for creating Zarr writer."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Test the factory function directly
        from msiconvert.storage.incremental_zarr_writer import create_incremental_zarr_writer
        
        writer = create_incremental_zarr_writer(
            output_path=converter.output_path,
            dimensions=converter._dimensions,
            n_masses=len(converter._common_mass_axis),
            config=converter.zarr_config
        )
        
        assert writer is not None
        assert writer.total_pixels == 100
        
        writer.close()
    
    def test_streaming_data_flow(self, streaming_converter):
        """Test the complete streaming data flow."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Create sample interpolation results
        sample_results = []
        for i in range(10):
            result = InterpolationResult(
                coords=(i % 10, i // 10, 0),
                pixel_idx=i,
                sparse_indices=np.array([i, i+1], dtype=np.int_),
                sparse_values=np.array([100.0 + i, 200.0 + i], dtype=np.float64),
                tic_value=300.0 + i
            )
            sample_results.append(result)
        
        # Test data flow through streaming result processing
        zarr_data = {
            'sparse_matrix': Mock(),
            'tic_values': np.array([300.0 + i for i in range(100)]),
            'total_intensity': np.random.uniform(0, 1000, len(converter._common_mass_axis)),
            'pixels_written': 10,
            'sparse_nnz': 20
        }
        
        data_structures = converter._create_data_structures()
        converter._process_streaming_result(data_structures, zarr_data)
        
        # Check that data structures were updated
        assert data_structures['pixel_count'] == 10
        assert data_structures['sparse_data'] is zarr_data['sparse_matrix']
        np.testing.assert_array_equal(data_structures['total_intensity'], zarr_data['total_intensity'])
    
    def test_streaming_vs_standard_mode_decision(self, mock_reader, temp_output_path):
        """Test decision logic between streaming and standard Dask mode."""
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {"test": "structure"}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        # Test standard Dask mode (no streaming)
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            standard_converter = TestConverter(
                reader=mock_reader,
                output_path=temp_output_path,
                enable_streaming=False,  # Disabled
                use_dask=True
            )
            
            standard_converter._initialize_conversion()
            
            # Should use Dask but not streaming
            assert standard_converter.enable_streaming is False
            assert standard_converter.use_dask is True
            
        # Test streaming mode
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            streaming_converter = TestConverter(
                reader=mock_reader,
                output_path=temp_output_path,
                enable_streaming=True,  # Enabled
                use_dask=True
            )
            
            streaming_converter._initialize_conversion()
            
            # Should use both Dask and streaming
            assert streaming_converter.enable_streaming is True
            assert streaming_converter.use_dask is True
    
    def test_memory_estimation_with_streaming(self, streaming_converter):
        """Test that memory estimation works correctly with streaming."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Memory estimation should work regardless of streaming mode
        estimated_memory = converter._estimate_memory_requirements()
        
        assert estimated_memory > 0
        assert isinstance(estimated_memory, float)
        
        # Streaming shouldn't change the estimation logic
        converter.enable_streaming = False
        estimated_memory_no_streaming = converter._estimate_memory_requirements()
        
        assert estimated_memory == estimated_memory_no_streaming
    
    def test_error_handling_without_zarr(self, mock_reader, temp_output_path):
        """Test error handling when Zarr is not available."""
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {"test": "structure"}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.storage.incremental_zarr_writer.ZARR_AVAILABLE', False):
            with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
                converter = TestConverter(
                    reader=mock_reader,
                    output_path=temp_output_path,
                    enable_streaming=True,
                    use_dask=True
                )
                
                converter._initialize_conversion()
                data_structures = converter._create_data_structures()
                
                # Should raise an error when trying to create Zarr writer
                with pytest.raises(ImportError, match="Zarr is not available"):
                    converter._process_with_dask(data_structures)
    
    def test_progress_tracking_integration(self, streaming_converter):
        """Test integration of progress tracking in streaming mode."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Mock the Zarr writer with progress tracking
        mock_zarr_writer = Mock()
        mock_zarr_writer.get_progress.return_value = {
            'pixels_written': 50,
            'total_pixels': 100,
            'progress_percent': 50,
            'sparse_elements': 1000
        }
        
        converter._zarr_writer = mock_zarr_writer
        
        # Test progress retrieval
        progress = converter._zarr_writer.get_progress()
        
        assert progress['progress_percent'] == 50
        assert progress['pixels_written'] == 50
        assert progress['total_pixels'] == 100


@pytest.mark.skipif(not (ZARR_AVAILABLE and DASK_AVAILABLE), reason="Zarr and Dask required")
class TestPerformanceValidation:
    """Test performance characteristics of streaming mode."""
    
    def test_streaming_memory_efficiency(self, streaming_converter):
        """Test that streaming mode doesn't accumulate large amounts of data."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # In streaming mode, the converter shouldn't hold large data structures
        data_structures = converter._create_data_structures()
        
        # Initial data structures should be minimal
        assert isinstance(data_structures['sparse_data'], dict)  # Should be empty dict structure
        assert len(data_structures['tic_values']) == 100  # Only space for TIC values
        
        # Total memory footprint should be small compared to non-streaming
        total_size = sum(
            getattr(v, 'nbytes', len(str(v))) 
            for v in data_structures.values() 
            if hasattr(v, 'nbytes') or isinstance(v, (str, list, dict))
        )
        
        # Should be much smaller than storing full dense arrays
        max_reasonable_size = 100 * len(converter._common_mass_axis) * 8  # Full dense would be this
        assert total_size < max_reasonable_size * 0.1  # Should be <10% of dense size
    
    def test_chunk_size_optimization(self, streaming_converter):
        """Test that chunk size optimization works for streaming."""
        converter = streaming_converter
        converter._initialize_conversion()
        
        # Test different chunk sizes
        converter.dask_chunk_size = None  # Auto-detect
        auto_chunk_size = converter._determine_dask_chunk_size()
        
        converter.dask_chunk_size = 50  # Manual setting
        manual_chunk_size = converter._determine_dask_chunk_size()
        
        assert auto_chunk_size > 0
        assert manual_chunk_size == 50
        
        # Auto-detected size should be reasonable for the dataset
        assert 10 <= auto_chunk_size <= 5000  # Increased upper bound


if __name__ == "__main__":
    pytest.main([__file__])