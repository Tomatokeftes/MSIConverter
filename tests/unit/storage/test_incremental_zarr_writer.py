# tests/unit/storage/test_incremental_zarr_writer.py

"""
Tests for incremental Zarr writer functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from msiconvert.processing.interpolation import InterpolationResult

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

# Only import if Zarr is available
if ZARR_AVAILABLE:
    from msiconvert.storage.incremental_zarr_writer import (
        IncrementalZarrWriter,
        ZarrStorageConfig,
        create_incremental_zarr_writer
    )


@pytest.fixture
def temp_zarr_path():
    """Create a temporary directory for Zarr storage."""
    temp_dir = tempfile.mkdtemp()
    zarr_path = Path(temp_dir) / "test.zarr"
    yield zarr_path
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_interpolation_results():
    """Create sample interpolation results for testing."""
    results = [
        InterpolationResult(
            coords=(0, 0, 0),
            pixel_idx=0,
            sparse_indices=np.array([10, 20, 30], dtype=np.int_),
            sparse_values=np.array([100.0, 200.0, 150.0], dtype=np.float64),
            tic_value=450.0
        ),
        InterpolationResult(
            coords=(1, 0, 0),
            pixel_idx=1,
            sparse_indices=np.array([15, 25], dtype=np.int_),
            sparse_values=np.array([300.0, 250.0], dtype=np.float64),
            tic_value=550.0
        ),
        InterpolationResult(
            coords=(2, 0, 0),
            pixel_idx=2,
            sparse_indices=np.array([], dtype=np.int_),
            sparse_values=np.array([], dtype=np.float64),
            tic_value=0.0
        )
    ]
    return results


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="Zarr not available")
class TestZarrStorageConfig:
    """Test ZarrStorageConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ZarrStorageConfig()
        
        assert config.chunk_size == 10000
        assert config.compression == 'zstd'
        assert config.compression_level == 3
        assert config.sparse_data_dtype == 'float32'
        assert config.sparse_index_dtype == 'int32'
        assert config.resize_increment == 100000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ZarrStorageConfig(
            chunk_size=5000,
            compression='gzip',
            compression_level=6,
            sparse_data_dtype='float64',
            sparse_index_dtype='int64',
            resize_increment=50000
        )
        
        assert config.chunk_size == 5000
        assert config.compression == 'gzip'
        assert config.compression_level == 6
        assert config.sparse_data_dtype == 'float64'
        assert config.sparse_index_dtype == 'int64'
        assert config.resize_increment == 50000


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="Zarr not available")
class TestIncrementalZarrWriter:
    """Test IncrementalZarrWriter class."""
    
    def test_initialization(self, temp_zarr_path):
        """Test proper initialization of IncrementalZarrWriter."""
        total_pixels = 100
        n_masses = 1000
        
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=total_pixels,
            n_masses=n_masses
        )
        
        assert writer.total_pixels == total_pixels
        assert writer.n_masses == n_masses
        assert writer._current_sparse_pos == 0
        assert writer._pixels_written == 0
        
        # Check Zarr store was created
        assert writer.zarr_root is not None
        assert 'sparse_data' in writer.zarr_root
        assert 'sparse_rows' in writer.zarr_root
        assert 'sparse_cols' in writer.zarr_root
        assert 'tic_values' in writer.zarr_root
        assert 'total_intensity' in writer.zarr_root
        
        # Check metadata
        assert writer.zarr_root.attrs['total_pixels'] == total_pixels
        assert writer.zarr_root.attrs['n_masses'] == n_masses
        assert writer.zarr_root.attrs['sparse_format'] == 'coo'
        
        writer.close()
    
    def test_initialization_without_zarr(self, temp_zarr_path):
        """Test that initialization fails without Zarr."""
        with patch('msiconvert.storage.incremental_zarr_writer.ZARR_AVAILABLE', False):
            with pytest.raises(ImportError, match="Zarr is not available"):
                IncrementalZarrWriter(
                    output_path=temp_zarr_path,
                    total_pixels=100,
                    n_masses=1000
                )
    
    def test_custom_config(self, temp_zarr_path):
        """Test initialization with custom configuration."""
        config = ZarrStorageConfig(
            chunk_size=5000,
            compression='gzip',
            sparse_data_dtype='float64'
        )
        
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=100,
            n_masses=1000,
            config=config
        )
        
        assert writer.config.chunk_size == 5000
        assert writer.config.compression == 'gzip'
        assert writer.config.sparse_data_dtype == 'float64'
        
        # Check that arrays were created with custom settings
        assert writer.sparse_data_array.dtype == 'float64'
        assert writer.sparse_data_array.chunks[0] == 5000
        
        writer.close()
    
    def test_write_empty_results(self, temp_zarr_path):
        """Test writing empty interpolation results."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=10,
            n_masses=100
        )
        
        # Write empty results list
        writer.write_interpolation_results([])
        
        assert writer._pixels_written == 0
        assert writer._current_sparse_pos == 0
        
        writer.close()
    
    def test_write_single_result(self, temp_zarr_path):
        """Test writing a single interpolation result."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=10,
            n_masses=100
        )
        
        result = InterpolationResult(
            coords=(0, 0, 0),
            pixel_idx=0,
            sparse_indices=np.array([10, 20], dtype=np.int_),
            sparse_values=np.array([100.0, 200.0], dtype=np.float64),
            tic_value=300.0
        )
        
        writer.write_interpolation_results([result])
        
        assert writer._pixels_written == 1
        assert writer._current_sparse_pos == 2
        
        # Check TIC value was written
        assert writer.tic_values_array[0] == 300.0
        
        # Check sparse data was written
        assert writer.sparse_data_array[0] == 100.0
        assert writer.sparse_data_array[1] == 200.0
        assert writer.sparse_rows_array[0] == 0
        assert writer.sparse_rows_array[1] == 0
        assert writer.sparse_cols_array[0] == 10
        assert writer.sparse_cols_array[1] == 20
        
        writer.close()
    
    def test_write_multiple_results(self, temp_zarr_path, sample_interpolation_results):
        """Test writing multiple interpolation results."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=10,
            n_masses=100
        )
        
        writer.write_interpolation_results(sample_interpolation_results)
        
        assert writer._pixels_written == 3
        assert writer._current_sparse_pos == 5  # 3 + 2 + 0 sparse elements
        
        # Check TIC values
        assert writer.tic_values_array[0] == 450.0
        assert writer.tic_values_array[1] == 550.0
        assert writer.tic_values_array[2] == 0.0
        
        # Check first few sparse elements
        assert writer.sparse_data_array[0] == 100.0
        assert writer.sparse_rows_array[0] == 0
        assert writer.sparse_cols_array[0] == 10
        
        writer.close()
    
    def test_write_empty_pixel(self, temp_zarr_path):
        """Test writing results that include empty pixels."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=10,
            n_masses=100
        )
        
        empty_result = InterpolationResult(
            coords=(5, 5, 0),
            pixel_idx=5,
            sparse_indices=np.array([], dtype=np.int_),
            sparse_values=np.array([], dtype=np.float64),
            tic_value=0.0
        )
        
        writer.write_interpolation_results([empty_result])
        
        assert writer._pixels_written == 1
        assert writer._current_sparse_pos == 0  # No sparse elements
        assert writer.tic_values_array[5] == 0.0
        
        writer.close()
    
    def test_array_resizing(self, temp_zarr_path):
        """Test automatic array resizing when capacity is exceeded."""
        # Use small resize increment for testing
        config = ZarrStorageConfig(resize_increment=5)
        
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=100,
            n_masses=100,
            config=config
        )
        
        initial_size = writer.sparse_data_array.shape[0]
        
        # Create many results to force resizing
        results = []
        for i in range(10):
            result = InterpolationResult(
                coords=(i, 0, 0),
                pixel_idx=i,
                sparse_indices=np.array([i, i+1], dtype=np.int_),
                sparse_values=np.array([100.0 + i, 200.0 + i], dtype=np.float64),
                tic_value=300.0 + i
            )
            results.append(result)
        
        writer.write_interpolation_results(results)
        
        # Array should have been resized
        final_size = writer.sparse_data_array.shape[0]
        assert final_size > initial_size
        
        assert writer._pixels_written == 10
        assert writer._current_sparse_pos == 20  # 2 elements per pixel * 10 pixels
        
        writer.close()
    
    def test_finalize_with_data(self, temp_zarr_path, sample_interpolation_results):
        """Test finalizing writer with data."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=10,
            n_masses=100
        )
        
        writer.write_interpolation_results(sample_interpolation_results)
        result = writer.finalize()
        
        # Check returned data structure
        assert 'sparse_matrix' in result
        assert 'tic_values' in result
        assert 'total_intensity' in result
        assert 'pixels_written' in result
        assert 'sparse_nnz' in result
        
        assert result['pixels_written'] == 3
        assert result['sparse_nnz'] == 5
        
        # Check sparse matrix properties
        sparse_matrix = result['sparse_matrix']
        assert sparse_matrix.shape == (10, 100)
        assert sparse_matrix.nnz == 5
        
        # Check TIC values
        tic_values = result['tic_values']
        assert len(tic_values) == 10
        assert tic_values[0] == 450.0
        assert tic_values[1] == 550.0
        
        # Check metadata was updated
        assert writer.zarr_root.attrs['finalized'] == True
        assert writer.zarr_root.attrs['final_pixels_written'] == 3
        
        writer.close()
    
    def test_finalize_empty(self, temp_zarr_path):
        """Test finalizing writer without data."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=10,
            n_masses=100
        )
        
        result = writer.finalize()
        
        assert result['pixels_written'] == 0
        assert result['sparse_nnz'] == 0
        
        # Should return empty sparse matrix
        sparse_matrix = result['sparse_matrix']
        assert sparse_matrix.shape == (10, 100)
        assert sparse_matrix.nnz == 0
        
        writer.close()
    
    def test_get_progress(self, temp_zarr_path, sample_interpolation_results):
        """Test progress tracking."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=10,
            n_masses=100
        )
        
        # Initial progress
        progress = writer.get_progress()
        assert progress['pixels_written'] == 0
        assert progress['total_pixels'] == 10
        assert progress['progress_percent'] == 0
        
        # Write some data
        writer.write_interpolation_results(sample_interpolation_results[:2])
        
        progress = writer.get_progress()
        assert progress['pixels_written'] == 2
        assert progress['total_pixels'] == 10
        assert progress['progress_percent'] == 20
        assert progress['sparse_elements'] == 5  # 3 + 2 sparse elements
        
        writer.close()
    
    def test_thread_safety_simulation(self, temp_zarr_path):
        """Test that write operations are thread-safe (simulation)."""
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=100,
            n_masses=100
        )
        
        # Simulate concurrent writes by checking lock behavior
        assert hasattr(writer, 'write_lock')
        
        # Write data and ensure consistency
        results = []
        for i in range(5):
            result = InterpolationResult(
                coords=(i, 0, 0),
                pixel_idx=i,
                sparse_indices=np.array([i], dtype=np.int_),
                sparse_values=np.array([100.0 + i], dtype=np.float64),
                tic_value=200.0 + i
            )
            results.append(result)
        
        writer.write_interpolation_results(results)
        
        assert writer._pixels_written == 5
        assert writer._current_sparse_pos == 5
        
        writer.close()


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="Zarr not available")
class TestFactoryFunction:
    """Test factory function for creating Zarr writers."""
    
    def test_create_incremental_zarr_writer(self, temp_zarr_path):
        """Test factory function for creating incremental Zarr writer."""
        dimensions = (10, 5, 2)
        n_masses = 1000
        
        writer = create_incremental_zarr_writer(
            output_path=temp_zarr_path,
            dimensions=dimensions,
            n_masses=n_masses
        )
        
        expected_pixels = 10 * 5 * 2  # 100 pixels
        assert writer.total_pixels == expected_pixels
        assert writer.n_masses == n_masses
        
        # Check that path was modified for temporary Zarr
        assert str(writer.output_path).endswith('.zarr.tmp')
        
        writer.close()
    
    def test_create_with_custom_config(self, temp_zarr_path):
        """Test factory function with custom configuration."""
        config = ZarrStorageConfig(chunk_size=5000)
        
        writer = create_incremental_zarr_writer(
            output_path=temp_zarr_path,
            dimensions=(5, 5, 1),
            n_masses=500,
            config=config
        )
        
        assert writer.config.chunk_size == 5000
        assert writer.total_pixels == 25
        assert writer.n_masses == 500
        
        writer.close()


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="Zarr not available")
class TestIntegrationScenarios:
    """Test integration scenarios with realistic data."""
    
    def test_large_dataset_simulation(self, temp_zarr_path):
        """Test simulation of large dataset processing."""
        # Simulate 1000 pixels with varying sparsity
        total_pixels = 1000
        n_masses = 10000
        
        writer = IncrementalZarrWriter(
            output_path=temp_zarr_path,
            total_pixels=total_pixels,
            n_masses=n_masses
        )
        
        # Process in chunks of 100 pixels
        for chunk_start in range(0, total_pixels, 100):
            chunk_end = min(chunk_start + 100, total_pixels)
            chunk_results = []
            
            for pixel_idx in range(chunk_start, chunk_end):
                # Simulate realistic sparsity - some pixels empty, others with varying peaks
                if pixel_idx % 10 == 0:
                    # Empty pixel
                    sparse_indices = np.array([], dtype=np.int_)
                    sparse_values = np.array([], dtype=np.float64)
                    tic_value = 0.0
                else:
                    # Non-empty pixel with random number of peaks
                    n_peaks = np.random.randint(1, 20)
                    sparse_indices = np.random.randint(0, n_masses, n_peaks, dtype=np.int_)
                    sparse_values = np.random.uniform(10.0, 1000.0, n_peaks).astype(np.float64)
                    tic_value = float(np.sum(sparse_values))
                
                result = InterpolationResult(
                    coords=(pixel_idx % 100, pixel_idx // 100, 0),
                    pixel_idx=pixel_idx,
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                    tic_value=tic_value
                )
                chunk_results.append(result)
            
            writer.write_interpolation_results(chunk_results)
        
        # Finalize and check results
        final_data = writer.finalize()
        
        assert final_data['pixels_written'] == total_pixels
        assert final_data['sparse_matrix'].shape == (total_pixels, n_masses)
        
        # Check that we have realistic sparsity (should be much less than total possible)
        assert final_data['sparse_nnz'] < total_pixels * n_masses * 0.01  # Less than 1% density
        
        writer.close()
    
    def test_streaming_workflow_simulation(self, temp_zarr_path):
        """Test complete streaming workflow simulation."""
        dimensions = (20, 20, 1)
        n_masses = 5000
        total_pixels = 20 * 20
        
        writer = create_incremental_zarr_writer(
            output_path=temp_zarr_path,
            dimensions=dimensions,
            n_masses=n_masses
        )
        
        # Simulate Dask chunk processing
        chunk_size = 50
        all_chunks_data = []
        
        for chunk_start in range(0, total_pixels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pixels)
            chunk_results = []
            
            for pixel_idx in range(chunk_start, chunk_end):
                # Convert linear index to coordinates
                x = pixel_idx % 20
                y = pixel_idx // 20
                z = 0
                
                # Simulate realistic MS data
                if (x + y) % 3 == 0:  # Some pattern of empty pixels
                    sparse_indices = np.array([], dtype=np.int_)
                    sparse_values = np.array([], dtype=np.float64)
                    tic_value = 0.0
                else:
                    # Generate realistic spectrum with peaks
                    n_peaks = np.random.randint(5, 50)
                    sparse_indices = np.sort(np.random.choice(n_masses, n_peaks, replace=False))
                    sparse_values = np.random.exponential(100.0, n_peaks).astype(np.float64)
                    tic_value = float(np.sum(sparse_values))
                
                result = InterpolationResult(
                    coords=(x, y, z),
                    pixel_idx=pixel_idx,
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                    tic_value=tic_value
                )
                chunk_results.append(result)
            
            # Write chunk (simulating Dask delayed execution)
            writer.write_interpolation_results(chunk_results)
            all_chunks_data.extend(chunk_results)
        
        # Finalize
        final_data = writer.finalize()
        
        # Validate results
        assert final_data['pixels_written'] == total_pixels
        assert final_data['sparse_matrix'].shape == (total_pixels, n_masses)
        
        # Check TIC values shape for 2D data
        tic_values = final_data['tic_values']
        assert tic_values.shape == (total_pixels,)
        
        # Verify that total intensity was accumulated correctly
        total_intensity = final_data['total_intensity']
        assert total_intensity.shape == (n_masses,)
        assert np.sum(total_intensity) > 0  # Should have accumulated some intensity
        
        writer.close()


if __name__ == "__main__":
    pytest.main([__file__])