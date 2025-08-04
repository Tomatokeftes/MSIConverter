# tests/unit/converters/test_mass_axis_generation.py

"""
Tests for enhanced mass axis generation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from msiconvert.converters.spatialdata.base_spatialdata_converter import BaseSpatialDataConverter


@pytest.fixture
def mock_reader():
    """Create a mock MSI reader."""
    reader = Mock()
    
    # Mock essential metadata with mass range
    essential_metadata = Mock()
    essential_metadata.dimensions = (10, 10, 1)
    essential_metadata.coordinate_bounds = ((0, 9), (0, 9), (0, 0))
    essential_metadata.n_spectra = 100
    essential_metadata.estimated_memory_gb = 0.1
    essential_metadata.pixel_size = (25.0, 25.0)
    essential_metadata.mass_range = (100.0, 500.0)  # Wide mass range
    
    reader.get_essential_metadata.return_value = essential_metadata
    reader.close = Mock()
    
    return reader


@pytest.fixture
def base_converter(mock_reader):
    """Create a base converter instance for testing."""
    
    class TestConverter(BaseSpatialDataConverter):
        def _create_data_structures(self):
            return {}
        
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


class TestMassAxisGenerationSetup:
    """Test mass axis generation configuration and setup."""
    
    def test_default_analyzer_type(self, base_converter):
        """Test default analyzer type is TOF."""
        assert base_converter.analyzer_type == "tof"
    
    def test_custom_analyzer_type(self, mock_reader):
        """Test setting custom analyzer type."""
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            converter = TestConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr"),
                analyzer_type="orbitrap"
            )
        
        assert converter.analyzer_type == "orbitrap"
    
    def test_analyzer_type_case_insensitive(self, mock_reader):
        """Test analyzer type is converted to lowercase."""
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            converter = TestConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr"),
                analyzer_type="ORBITRAP"
            )
        
        assert converter.analyzer_type == "orbitrap"
    
    def test_tof_parameters(self, mock_reader):
        """Test TOF-specific parameters."""
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            converter = TestConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr"),
                analyzer_type="tof",
                tof_bin_width_da=0.005,
                tof_reference_mz=800.0,
                tof_n_bins=5000
            )
        
        assert converter.tof_bin_width_da == 0.005
        assert converter.tof_reference_mz == 800.0
        assert converter.tof_n_bins == 5000
    
    def test_orbitrap_parameters(self, mock_reader):
        """Test Orbitrap-specific parameters."""
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            converter = TestConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr"),
                analyzer_type="orbitrap",
                fourier_bin_width_da=0.0005,
                fourier_n_bins=10000
            )
        
        assert converter.fourier_bin_width_da == 0.0005
        assert converter.fourier_n_bins == 10000


class TestMassAxisGeneration:
    """Test mass axis generation for different analyzer types."""
    
    def test_generate_common_mass_axis_with_metadata(self, base_converter):
        """Test mass axis generation using metadata mass range."""
        # Initialize conversion to trigger mass axis generation
        base_converter._initialize_conversion()
        
        mass_axis = base_converter._common_mass_axis
        
        # Should have generated a mass axis
        assert len(mass_axis) > 0
        assert mass_axis[0] >= 100.0  # Should start near min_mz
        assert mass_axis[-1] <= 500.0  # Should end near max_mz
        assert np.all(np.diff(mass_axis) > 0)  # Should be strictly increasing
    
    def test_generate_common_mass_axis_without_metadata(self, mock_reader):
        """Test mass axis generation fallback when no metadata mass range."""
        # Mock essential metadata without mass range
        essential_metadata = Mock()
        essential_metadata.dimensions = (2, 2, 1)  # Small for scan
        essential_metadata.coordinate_bounds = ((0, 1), (0, 1), (0, 0))
        essential_metadata.n_spectra = 4
        essential_metadata.estimated_memory_gb = 0.001
        essential_metadata.pixel_size = (25.0, 25.0)
        essential_metadata.mass_range = None  # No mass range
        
        mock_reader.get_essential_metadata.return_value = essential_metadata
        
        # Mock spectrum iteration for scanning
        def mock_iter_spectra():
            yield (0, 0, 0), np.array([150.0, 200.0]), np.array([100.0, 200.0])
            yield (1, 0, 0), np.array([120.0, 180.0]), np.array([150.0, 250.0])
        
        mock_reader.iter_spectra = mock_iter_spectra
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            converter = TestConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr")
            )
        
        # Initialize conversion - should trigger mass range scan
        converter._initialize_conversion()
        
        mass_axis = converter._common_mass_axis
        
        # Should have generated a mass axis from scanned range
        assert len(mass_axis) > 0
        assert mass_axis[0] >= 120.0  # Should include minimum found
        assert mass_axis[-1] <= 200.0  # Should include maximum found
    
    def test_tof_mass_axis_variable_width(self, base_converter):
        """Test TOF mass axis with variable bin width."""
        base_converter.analyzer_type = "tof"
        base_converter.tof_bin_width_da = 0.01
        base_converter.tof_reference_mz = 200.0
        
        mass_axis = base_converter._create_tof_mass_axis(100.0, 300.0)
        
        # Should have variable width bins
        assert len(mass_axis) > 0
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 300.0
        
        # Check that bin width increases with m/z (approximately)
        bin_widths = np.diff(mass_axis)
        # Later bins should generally be wider (allowing for some variation)
        assert np.mean(bin_widths[-10:]) > np.mean(bin_widths[:10])
    
    def test_tof_mass_axis_n_bins(self, base_converter):
        """Test TOF mass axis with specified number of bins."""
        base_converter.analyzer_type = "tof"
        base_converter.tof_n_bins = 1000
        
        mass_axis = base_converter._create_tof_mass_axis(100.0, 300.0)
        
        # Should have approximately the specified number of bins
        assert 950 <= len(mass_axis) <= 1050  # Allow some tolerance
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 300.0
    
    def test_orbitrap_mass_axis_constant_width(self, base_converter):
        """Test Orbitrap mass axis with constant bin width."""
        base_converter.analyzer_type = "orbitrap"
        base_converter.fourier_bin_width_da = 0.001
        
        mass_axis = base_converter._create_fourier_mass_axis(100.0, 200.0)
        
        # Should have constant width bins
        assert len(mass_axis) > 0
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 200.0
        
        # Check that bin widths are approximately constant
        bin_widths = np.diff(mass_axis)
        assert np.std(bin_widths) < 0.0001  # Very consistent bin widths
        assert np.mean(bin_widths) == pytest.approx(0.001, rel=0.01)
    
    def test_orbitrap_mass_axis_n_bins(self, base_converter):
        """Test Orbitrap mass axis with specified number of bins."""
        base_converter.analyzer_type = "orbitrap"
        base_converter.fourier_n_bins = 5000
        
        mass_axis = base_converter._create_fourier_mass_axis(100.0, 200.0)
        
        # Should have approximately the specified number of bins
        assert 4950 <= len(mass_axis) <= 5050  # Allow some tolerance
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 200.0
    
    def test_quadrupole_mass_axis_constant_steps(self, base_converter):
        """Test quadrupole mass axis with constant step size."""
        base_converter.analyzer_type = "quadrupole"
        base_converter.quadrupole_step_size_da = 0.1
        
        mass_axis = base_converter._create_quadrupole_mass_axis(100.0, 200.0)
        
        # Should have constant step sizes
        assert len(mass_axis) > 0
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 200.0
        
        # Check that step sizes are approximately constant
        bin_widths = np.diff(mass_axis)
        assert np.std(bin_widths) < 0.001  # Very consistent steps
        assert np.mean(bin_widths) == pytest.approx(0.1, rel=0.01)
    
    def test_quadrupole_mass_axis_n_bins(self, base_converter):
        """Test quadrupole mass axis with specified number of bins."""
        base_converter.analyzer_type = "quadrupole"
        base_converter.quadrupole_n_bins = 2000
        
        mass_axis = base_converter._create_quadrupole_mass_axis(100.0, 200.0)
        
        # Should have approximately the specified number of bins
        assert 1950 <= len(mass_axis) <= 2050  # Allow some tolerance
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 200.0
    
    def test_linear_mass_axis_fallback(self, base_converter):
        """Test linear mass axis for unknown analyzer types."""
        base_converter.analyzer_type = "unknown"
        base_converter.default_n_bins = 1500
        
        mass_axis = base_converter._create_linear_mass_axis(100.0, 200.0)
        
        # Should have linear spacing
        assert len(mass_axis) > 0
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 200.0
        
        # Check that spacing is linear
        bin_widths = np.diff(mass_axis)
        assert np.std(bin_widths) < 0.001  # Very consistent spacing
    
    def test_unknown_analyzer_type_warning(self, base_converter):
        """Test that unknown analyzer types generate warnings."""
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.logging.warning') as mock_warning:
            mass_axis = base_converter._create_mass_axis(100.0, 200.0, "unknown_analyzer")
        
        mock_warning.assert_called_once()
        assert "Unknown analyzer type" in mock_warning.call_args[0][0]
        
        # Should still generate a mass axis
        assert len(mass_axis) > 0
    
    def test_create_axis_from_n_bins_linear(self, base_converter):
        """Test linear axis creation from number of bins."""
        mass_axis = base_converter._create_axis_from_n_bins(100.0, 200.0, 1000, 'linear')
        
        assert 950 <= len(mass_axis) <= 1050
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 200.0
        
        # Should be evenly spaced
        bin_widths = np.diff(mass_axis)
        assert np.std(bin_widths) < 0.001
    
    def test_create_axis_from_n_bins_tof_approximation(self, base_converter):
        """Test TOF axis creation from number of bins (linear approximation)."""
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.logging.info') as mock_info:
            mass_axis = base_converter._create_axis_from_n_bins(100.0, 200.0, 1000, 'tof')
        
        # Should log that it's using linear approximation
        assert any("linear approximation" in str(call) for call in mock_info.call_args_list)
        
        assert 950 <= len(mass_axis) <= 1050
        assert mass_axis[0] >= 100.0
        assert mass_axis[-1] <= 200.0
    
    def test_create_axis_from_n_bins_unknown_spacing(self, base_converter):
        """Test error for unknown spacing type."""
        with pytest.raises(ValueError, match="Unknown spacing type"):
            base_converter._create_axis_from_n_bins(100.0, 200.0, 1000, 'unknown')


class TestMassAxisQuality:
    """Test quality and consistency of generated mass axes."""
    
    def test_mass_axis_strictly_increasing(self, base_converter):
        """Test that all generated mass axes are strictly increasing."""
        test_cases = [
            ("tof", {}),
            ("orbitrap", {}),
            ("quadrupole", {}),
            ("unknown", {})
        ]
        
        for analyzer_type, kwargs in test_cases:
            base_converter.analyzer_type = analyzer_type
            mass_axis = base_converter._create_mass_axis(100.0, 300.0, analyzer_type)
            
            # Should be strictly increasing
            assert np.all(np.diff(mass_axis) > 0), f"Mass axis not strictly increasing for {analyzer_type}"
    
    def test_mass_axis_covers_range(self, base_converter):
        """Test that mass axes cover the specified range."""
        min_mz, max_mz = 150.0, 250.0
        
        test_cases = ["tof", "orbitrap", "quadrupole", "unknown"]
        
        for analyzer_type in test_cases:
            base_converter.analyzer_type = analyzer_type
            mass_axis = base_converter._create_mass_axis(min_mz, max_mz, analyzer_type)
            
            # Should cover the range (allowing small tolerance for bin edges)
            assert mass_axis[0] >= min_mz - 0.1, f"Mass axis starts too low for {analyzer_type}"
            assert mass_axis[-1] <= max_mz + 0.1, f"Mass axis ends too high for {analyzer_type}"
            assert mass_axis[0] <= min_mz + 10.0, f"Mass axis starts too high for {analyzer_type}"
            assert mass_axis[-1] >= max_mz - 10.0, f"Mass axis ends too low for {analyzer_type}"
    
    def test_mass_axis_reasonable_length(self, base_converter):
        """Test that mass axes have reasonable lengths."""
        min_mz, max_mz = 100.0, 500.0  # 400 Da range
        
        test_cases = ["tof", "orbitrap", "quadrupole", "unknown"]
        
        for analyzer_type in test_cases:
            base_converter.analyzer_type = analyzer_type
            mass_axis = base_converter._create_mass_axis(min_mz, max_mz, analyzer_type)
            
            # Should have reasonable number of points (not too sparse or dense)
            assert 100 <= len(mass_axis) <= 1000000, f"Unreasonable mass axis length for {analyzer_type}: {len(mass_axis)}"
    
    def test_mass_axis_reproducible(self, base_converter):
        """Test that mass axis generation is reproducible."""
        base_converter.analyzer_type = "tof"
        
        mass_axis1 = base_converter._create_mass_axis(100.0, 200.0, "tof")
        mass_axis2 = base_converter._create_mass_axis(100.0, 200.0, "tof")
        
        # Should be identical
        np.testing.assert_array_equal(mass_axis1, mass_axis2)


class TestMassRangeScanning:
    """Test mass range scanning fallback functionality."""
    
    def test_scan_for_mass_range_empty_dataset(self, mock_reader):
        """Test mass range scanning with empty dataset."""
        # Mock empty spectrum iteration
        mock_reader.iter_spectra = lambda: iter([])
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            converter = TestConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr")
            )
        
        # Should handle empty dataset - returns inf values
        min_mz, max_mz = converter._scan_for_mass_range()
        
        # Should return infinity values for empty dataset
        assert min_mz == float('inf')
        assert max_mz == float('-inf')
    
    def test_scan_for_mass_range_with_data(self, mock_reader):
        """Test mass range scanning with actual data."""
        # Mock spectrum iteration
        def mock_iter_spectra():
            yield (0, 0, 0), np.array([120.0, 180.0]), np.array([100.0, 200.0])
            yield (1, 0, 0), np.array([110.0, 190.0]), np.array([150.0, 250.0])
            yield (2, 0, 0), np.array([]), np.array([])  # Empty spectrum
            yield (3, 0, 0), np.array([130.0, 170.0]), np.array([120.0, 180.0])
        
        mock_reader.iter_spectra = mock_iter_spectra
        
        class TestConverter(BaseSpatialDataConverter):
            def _create_data_structures(self):
                return {}
            def _process_single_spectrum(self, data_structures, coords, mzs, intensities):
                pass
            def _finalize_data(self, data_structures):
                pass
        
        with patch('msiconvert.converters.spatialdata.base_spatialdata_converter.SPATIALDATA_AVAILABLE', True):
            converter = TestConverter(
                reader=mock_reader,
                output_path=Path("/tmp/test.zarr")
            )
        
        min_mz, max_mz = converter._scan_for_mass_range()
        
        # Should find the correct range
        assert min_mz == 110.0
        assert max_mz == 190.0


if __name__ == "__main__":
    pytest.main([__file__])