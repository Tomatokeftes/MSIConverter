# tests/unit/processing/test_interpolation.py

"""
Tests for the spectral interpolation module.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from msiconvert.processing.interpolation import (
    SpectralInterpolator,
    InterpolationResult,
    DaskInterpolationProcessor,
    create_interpolator_from_mass_axis,
    DASK_AVAILABLE
)


class TestInterpolationResult:
    """Test cases for InterpolationResult dataclass."""
    
    def test_valid_interpolation_result(self):
        """Test creation of valid InterpolationResult."""
        coords = (5, 10, 0)
        pixel_idx = 150
        sparse_indices = np.array([1, 3, 5], dtype=np.int_)
        sparse_values = np.array([100.0, 200.0, 150.0], dtype=np.float64)
        tic_value = 450.0
        
        result = InterpolationResult(
            coords=coords,
            pixel_idx=pixel_idx,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            tic_value=tic_value
        )
        
        assert result.coords == coords
        assert result.pixel_idx == pixel_idx
        np.testing.assert_array_equal(result.sparse_indices, sparse_indices)
        np.testing.assert_array_equal(result.sparse_values, sparse_values)
        assert result.tic_value == tic_value
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched sparse arrays raise ValueError."""
        with pytest.raises(ValueError, match="sparse_indices and sparse_values must have the same length"):
            InterpolationResult(
                coords=(0, 0, 0),
                pixel_idx=0,
                sparse_indices=np.array([1, 2, 3]),
                sparse_values=np.array([1.0, 2.0]),  # Different length
                tic_value=3.0
            )
    
    def test_negative_tic_raises_error(self):
        """Test that negative TIC value raises ValueError."""
        with pytest.raises(ValueError, match="tic_value must be non-negative"):
            InterpolationResult(
                coords=(0, 0, 0),
                pixel_idx=0,
                sparse_indices=np.array([1, 2]),
                sparse_values=np.array([1.0, 2.0]),
                tic_value=-1.0  # Negative TIC
            )


class TestSpectralInterpolator:
    """Test cases for SpectralInterpolator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.common_mass_axis = np.linspace(100.0, 200.0, 101)  # 100-200 m/z, 1 Da resolution
        self.interpolator = SpectralInterpolator(self.common_mass_axis)
    
    def test_interpolator_initialization(self):
        """Test proper initialization of SpectralInterpolator."""
        assert len(self.interpolator.common_mass_axis) == 101
        assert self.interpolator.interpolation_method == "linear"
        assert self.interpolator.fill_value == 0.0
        assert self.interpolator.sparsity_threshold == 1e-10
        assert self.interpolator.mass_axis_length == 101
        assert self.interpolator.mass_range == (100.0, 200.0)
    
    def test_empty_mass_axis_raises_error(self):
        """Test that empty mass axis raises ValueError."""
        with pytest.raises(ValueError, match="common_mass_axis cannot be empty"):
            SpectralInterpolator(np.array([]))
    
    def test_non_increasing_mass_axis_raises_error(self):
        """Test that non-increasing mass axis raises ValueError."""
        with pytest.raises(ValueError, match="common_mass_axis must be strictly increasing"):
            SpectralInterpolator(np.array([100.0, 99.0, 101.0]))
    
    def test_interpolate_empty_spectrum(self):
        """Test interpolation of empty spectrum."""
        result = self.interpolator.interpolate_spectrum(
            mzs=np.array([]),
            intensities=np.array([]),
            coords=(5, 10, 0),
            pixel_idx=150
        )
        
        assert result.coords == (5, 10, 0)
        assert result.pixel_idx == 150
        assert len(result.sparse_indices) == 0
        assert len(result.sparse_values) == 0
        assert result.tic_value == 0.0
    
    def test_interpolate_single_peak(self):
        """Test interpolation of spectrum with single peak."""
        mzs = np.array([150.0])
        intensities = np.array([1000.0])
        
        result = self.interpolator.interpolate_spectrum(
            mzs=mzs,
            intensities=intensities,
            coords=(0, 0, 0),
            pixel_idx=0
        )
        
        assert result.tic_value == 1000.0
        assert len(result.sparse_indices) == 1
        # Should interpolate to index 50 (150.0 m/z at 1 Da resolution starting from 100.0)
        expected_idx = 50
        assert result.sparse_indices[0] == expected_idx
        assert result.sparse_values[0] == 1000.0
    
    def test_interpolate_multiple_peaks(self):
        """Test interpolation of spectrum with multiple peaks."""
        mzs = np.array([110.0, 150.0, 190.0])
        intensities = np.array([500.0, 1000.0, 750.0])
        
        result = self.interpolator.interpolate_spectrum(
            mzs=mzs,
            intensities=intensities,
            coords=(1, 1, 0),
            pixel_idx=10
        )
        
        assert result.tic_value == 2250.0
        # Linear interpolation will create many points between input points
        assert len(result.sparse_indices) > 3
        
        # Check that key indices correspond to correct m/z values
        expected_indices = [10, 50, 90]  # 110, 150, 190 m/z
        for idx in expected_indices:
            assert idx in result.sparse_indices
        
        # Check that peaks are at expected locations
        idx_10 = np.where(result.sparse_indices == 10)[0][0]
        idx_50 = np.where(result.sparse_indices == 50)[0][0]
        idx_90 = np.where(result.sparse_indices == 90)[0][0]
        
        assert result.sparse_values[idx_10] == 500.0
        assert result.sparse_values[idx_50] == 1000.0
        assert result.sparse_values[idx_90] == 750.0
    
    def test_interpolate_with_sparsity_threshold(self):
        """Test that values below sparsity threshold are filtered out."""
        interpolator = SpectralInterpolator(
            self.common_mass_axis,
            sparsity_threshold=100.0  # High threshold
        )
        
        mzs = np.array([110.0, 150.0, 190.0])
        intensities = np.array([50.0, 1000.0, 75.0])  # First and last below threshold
        
        result = interpolator.interpolate_spectrum(
            mzs=mzs,
            intensities=intensities,
            coords=(1, 1, 0),
            pixel_idx=10
        )
        
        # Only values above threshold should survive
        # The middle peak and some interpolated values around it
        assert len(result.sparse_indices) >= 1
        assert 50 in result.sparse_indices  # 150 m/z should be present
        
        # Check that the peak value is correct
        idx_50 = np.where(result.sparse_indices == 50)[0][0]
        assert result.sparse_values[idx_50] == 1000.0
        assert result.tic_value == 1125.0  # TIC calculated before sparsity filtering
    
    def test_interpolate_out_of_range_values(self):
        """Test interpolation with values outside mass axis range."""
        mzs = np.array([50.0, 150.0, 250.0])  # First and last outside 100-200 range
        intensities = np.array([500.0, 1000.0, 750.0])
        
        result = self.interpolator.interpolate_spectrum(
            mzs=mzs,
            intensities=intensities,
            coords=(2, 2, 0),
            pixel_idx=20
        )
        
        # The middle peak should be present, others outside range get fill_value=0
        assert len(result.sparse_indices) >= 1
        assert 50 in result.sparse_indices  # 150 m/z should be present
        
        # Check that the in-range peak value is correct
        idx_50 = np.where(result.sparse_indices == 50)[0][0]
        assert result.sparse_values[idx_50] == 1000.0
        assert result.tic_value == 2250.0  # Original TIC before interpolation
    
    def test_mismatched_mz_intensity_lengths(self):
        """Test that mismatched m/z and intensity arrays raise ValueError."""
        with pytest.raises(ValueError, match="mzs and intensities must have the same length"):
            self.interpolator.interpolate_spectrum(
                mzs=np.array([100.0, 150.0]),
                intensities=np.array([500.0]),  # Different length
                coords=(0, 0, 0),
                pixel_idx=0
            )
    
    def test_interpolate_chunk(self):
        """Test interpolation of multiple pixels in a chunk."""
        pixel_chunk = [
            {
                'coords': (0, 0, 0),
                'pixel_idx': 0,
                'mzs': np.array([150.0]),
                'intensities': np.array([1000.0])
            },
            {
                'coords': (1, 0, 0),
                'pixel_idx': 1,
                'mzs': np.array([110.0, 190.0]),
                'intensities': np.array([500.0, 750.0])
            }
        ]
        
        results = self.interpolator.interpolate_chunk(pixel_chunk)
        
        assert len(results) == 2
        
        # First pixel
        assert results[0].coords == (0, 0, 0)
        assert results[0].tic_value == 1000.0
        assert len(results[0].sparse_indices) == 1
        assert results[0].sparse_indices[0] == 50  # 150.0 m/z
        
        # Second pixel - will have interpolated values between peaks
        assert results[1].coords == (1, 0, 0)
        assert results[1].tic_value == 1250.0
        assert len(results[1].sparse_indices) >= 2  # At least the two peaks
        assert 10 in results[1].sparse_indices  # 110.0 m/z
        assert 90 in results[1].sparse_indices  # 190.0 m/z
    
    def test_interpolate_chunk_with_error(self):
        """Test that chunk interpolation continues despite individual pixel errors."""
        pixel_chunk = [
            {
                'coords': (0, 0, 0),
                'pixel_idx': 0,
                'mzs': np.array([150.0]),
                'intensities': np.array([1000.0])
            },
            {
                'coords': (1, 0, 0),
                'pixel_idx': 1,
                'mzs': np.array([110.0, 190.0]),
                'intensities': np.array([500.0])  # Mismatched lengths - will cause error
            }
        ]
        
        with patch('msiconvert.processing.interpolation.logging.warning') as mock_warning:
            results = self.interpolator.interpolate_chunk(pixel_chunk)
        
        # Should have one successful result and one warning
        assert len(results) == 1
        assert results[0].coords == (0, 0, 0)
        mock_warning.assert_called_once()
    
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_create_dask_interpolation_task(self):
        """Test creation of Dask delayed task."""
        from dask.delayed import delayed
        
        pixel_chunk = [
            {
                'coords': (0, 0, 0),
                'pixel_idx': 0,
                'mzs': np.array([150.0]),
                'intensities': np.array([1000.0])
            }
        ]
        
        # Create a delayed version of the pixel chunk
        delayed_pixel_chunk = delayed(lambda: pixel_chunk)()
        
        task = self.interpolator.create_dask_interpolation_task(delayed_pixel_chunk)
        
        # Should return a delayed object
        assert hasattr(task, 'compute')
        
        # Execute the task
        results = task.compute()
        assert len(results) == 1
        assert results[0].coords == (0, 0, 0)
    
    def test_dask_interpolation_task_without_dask(self):
        """Test that Dask task creation fails gracefully without Dask."""
        with patch('msiconvert.processing.interpolation.DASK_AVAILABLE', False):
            interpolator = SpectralInterpolator(self.common_mass_axis)
            
            with pytest.raises(RuntimeError, match="Dask is not available"):
                interpolator.create_dask_interpolation_task([])
    
    def test_nearest_interpolation(self):
        """Test nearest neighbor interpolation method."""
        interpolator = SpectralInterpolator(
            self.common_mass_axis,
            interpolation_method="nearest"
        )
        
        mzs = np.array([149.7, 150.3])  # Close to 150.0
        intensities = np.array([500.0, 1000.0])
        
        result = interpolator.interpolate_spectrum(
            mzs=mzs,
            intensities=intensities,
            coords=(0, 0, 0),
            pixel_idx=0
        )
        
        # Both should map to index 50 (150.0 m/z) in nearest neighbor
        # The interpolation will take one or both values
        assert len(result.sparse_indices) >= 1
        assert 50 in result.sparse_indices
    
    def test_cubic_interpolation_with_scipy(self):
        """Test cubic interpolation when SciPy is available."""
        with patch('scipy.interpolate.interp1d') as mock_interp1d:
            # Mock SciPy's interp1d
            mock_f = Mock()
            mock_f.return_value = np.array([0.0, 1000.0, 0.0])  # Mock interpolated result
            mock_interp1d.return_value = mock_f
            
            interpolator = SpectralInterpolator(
                np.array([100.0, 150.0, 200.0]),
                interpolation_method="cubic"
            )
            
            mzs = np.array([100.0, 150.0, 200.0, 250.0])
            intensities = np.array([0.0, 1000.0, 0.0, 500.0])
            
            result = interpolator.interpolate_spectrum(
                mzs=mzs,
                intensities=intensities,
                coords=(0, 0, 0),
                pixel_idx=0
            )
            
            mock_interp1d.assert_called_once()
            assert len(result.sparse_indices) == 1
            assert result.sparse_values[0] == 1000.0
    
    def test_cubic_interpolation_fallback_insufficient_points(self):
        """Test that cubic interpolation falls back to linear with insufficient points."""
        interpolator = SpectralInterpolator(
            self.common_mass_axis,
            interpolation_method="cubic"
        )
        
        # Only 3 points - insufficient for cubic interpolation
        mzs = np.array([110.0, 150.0, 190.0])
        intensities = np.array([500.0, 1000.0, 750.0])
        
        result = interpolator.interpolate_spectrum(
            mzs=mzs,
            intensities=intensities,
            coords=(0, 0, 0),
            pixel_idx=0
        )
        
        # Should still work (falls back to linear)
        assert len(result.sparse_indices) >= 3  # Linear interpolation creates more points
        assert result.tic_value == 2250.0
        
        # Check that key peaks are present
        assert 10 in result.sparse_indices  # 110.0 m/z
        assert 50 in result.sparse_indices  # 150.0 m/z
        assert 90 in result.sparse_indices  # 190.0 m/z
    
    def test_cubic_interpolation_fallback_no_scipy(self):
        """Test that cubic interpolation falls back to linear without SciPy."""
        with patch('scipy.interpolate.interp1d', side_effect=ImportError):
            interpolator = SpectralInterpolator(
                self.common_mass_axis,
                interpolation_method="cubic"
            )
            
            mzs = np.array([110.0, 150.0, 190.0, 120.0, 180.0])
            intensities = np.array([500.0, 1000.0, 750.0, 300.0, 400.0])
            
            with patch('msiconvert.processing.interpolation.logging.warning') as mock_warning:
                result = interpolator.interpolate_spectrum(
                    mzs=mzs,
                    intensities=intensities,
                    coords=(0, 0, 0),
                    pixel_idx=0
                )
            
            mock_warning.assert_called_once_with(
                "SciPy not available, falling back to linear interpolation"
            )
            assert len(result.sparse_indices) >= 5  # Linear interpolation creates more points
    
    def test_unknown_interpolation_method(self):
        """Test that unknown interpolation method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown interpolation method: unknown"):
            interpolator = SpectralInterpolator(
                self.common_mass_axis,
                interpolation_method="unknown"
            )
            interpolator.interpolate_spectrum(
                np.array([150.0]),
                np.array([1000.0]),
                (0, 0, 0),
                0
            )


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestDaskInterpolationProcessor:
    """Test cases for DaskInterpolationProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.common_mass_axis = np.linspace(100.0, 200.0, 101)
        self.interpolator = SpectralInterpolator(self.common_mass_axis)
        self.processor = DaskInterpolationProcessor(self.interpolator)
    
    def test_processor_initialization(self):
        """Test proper initialization of DaskInterpolationProcessor."""
        assert self.processor.interpolator is self.interpolator
        assert self.processor.memory_limit == "2GB"
    
    def test_processor_initialization_without_dask(self):
        """Test that processor raises error without Dask."""
        with patch('msiconvert.processing.interpolation.DASK_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="Dask is not available"):
                DaskInterpolationProcessor(self.interpolator)
    
    def test_process_chunks(self):
        """Test processing multiple chunks with Dask."""
        pixel_chunks = [
            [
                {
                    'coords': (0, 0, 0),
                    'pixel_idx': 0,
                    'mzs': np.array([150.0]),
                    'intensities': np.array([1000.0])
                }
            ],
            [
                {
                    'coords': (1, 0, 0),
                    'pixel_idx': 1,
                    'mzs': np.array([110.0, 190.0]),
                    'intensities': np.array([500.0, 750.0])
                }
            ]
        ]
        
        results = self.processor.process_chunks(pixel_chunks)
        
        assert len(results) == 2
        assert results[0].coords == (0, 0, 0)
        assert results[1].coords == (1, 0, 0)


class TestFactoryFunction:
    """Test cases for factory functions."""
    
    def test_create_interpolator_from_mass_axis(self):
        """Test factory function for creating interpolator."""
        mass_axis = np.linspace(100.0, 200.0, 101)
        
        interpolator = create_interpolator_from_mass_axis(
            mass_axis,
            interpolation_method="nearest",
            fill_value=1.0
        )
        
        assert interpolator.interpolation_method == "nearest"
        assert interpolator.fill_value == 1.0
        np.testing.assert_array_equal(interpolator.common_mass_axis, mass_axis)
    
    def test_create_interpolator_with_invalid_mass_axis(self):
        """Test that factory function validates mass axis."""
        with pytest.raises(ValueError, match="Valid common_mass_axis is required"):
            create_interpolator_from_mass_axis(None)
        
        with pytest.raises(ValueError, match="Valid common_mass_axis is required"):
            create_interpolator_from_mass_axis(np.array([]))


if __name__ == "__main__":
    pytest.main([__file__])