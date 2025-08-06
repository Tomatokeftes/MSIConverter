"""Tests for LinearMassAxisGenerator."""

import numpy as np
import pytest

from msiconvert.interpolators.mass_axis.linear_generator import LinearMassAxisGenerator
from msiconvert.metadata.types import EssentialMetadata


class TestLinearMassAxisGenerator:
    """Tests for LinearMassAxisGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = LinearMassAxisGenerator()
        self.metadata = EssentialMetadata(
            dimensions=(10, 10, 1),
            coordinate_bounds=(0.0, 10.0, 0.0, 10.0),
            mass_range=(100.0, 500.0),
            pixel_size=None,
            n_spectra=100,
            estimated_memory_gb=0.01,
            source_path="/test/path",
        )

    def test_generate_basic_functionality(self):
        """Test basic linear axis generation."""
        axis = self.generator.generate(self.metadata, 100)

        # Check type and shape
        assert isinstance(axis, np.ndarray)
        assert axis.dtype == np.float64
        assert len(axis) == 100

        # Check endpoints
        assert axis[0] == 100.0
        assert axis[-1] == 500.0

        # Check monotonic increasing
        assert np.all(np.diff(axis) > 0)

    def test_generate_uniform_spacing(self):
        """Test that generated axis has uniform spacing."""
        axis = self.generator.generate(self.metadata, 101)  # 101 points for easy math

        # Calculate expected spacing
        expected_spacing = (500.0 - 100.0) / (101 - 1)  # 4.0

        # Check actual spacing
        spacings = np.diff(axis)
        assert np.allclose(spacings, expected_spacing, rtol=1e-10)

    def test_generate_custom_range(self):
        """Test generation with custom mass range."""
        custom_range = (200.0, 300.0)
        axis = self.generator.generate(self.metadata, 50, mass_range=custom_range)

        assert len(axis) == 50
        assert axis[0] == 200.0
        assert axis[-1] == 300.0

        # Check uniform spacing
        expected_spacing = (300.0 - 200.0) / (50 - 1)
        spacings = np.diff(axis)
        assert np.allclose(spacings, expected_spacing, rtol=1e-10)

    def test_generate_single_point(self):
        """Test generation with single point."""
        axis = self.generator.generate(self.metadata, 1)

        assert len(axis) == 1
        # For single point, numpy.linspace returns the start value
        assert axis[0] == 100.0

    def test_generate_two_points(self):
        """Test generation with two points."""
        axis = self.generator.generate(self.metadata, 2)

        assert len(axis) == 2
        assert axis[0] == 100.0
        assert axis[1] == 500.0

    def test_generate_large_bins(self):
        """Test generation with large number of bins."""
        axis = self.generator.generate(self.metadata, 10000)

        assert len(axis) == 10000
        assert axis[0] == 100.0
        assert axis[-1] == 500.0

        # Check monotonic and uniform
        spacings = np.diff(axis)
        assert np.allclose(spacings, spacings[0], rtol=1e-10)

    def test_generate_narrow_range(self):
        """Test generation with very narrow mass range."""
        narrow_range = (100.0, 100.1)
        axis = self.generator.generate(self.metadata, 100, mass_range=narrow_range)

        assert len(axis) == 100
        assert axis[0] == 100.0
        assert axis[-1] == 100.1

        # Check precision
        expected_spacing = 0.1 / (100 - 1)
        spacings = np.diff(axis)
        assert np.allclose(spacings, expected_spacing, rtol=1e-10)

    def test_generate_wide_range(self):
        """Test generation with very wide mass range."""
        wide_range = (50.0, 2000.0)
        axis = self.generator.generate(self.metadata, 1000, mass_range=wide_range)

        assert len(axis) == 1000
        assert axis[0] == 50.0
        assert axis[-1] == 2000.0

        # Check uniform spacing
        expected_spacing = (2000.0 - 50.0) / (1000 - 1)
        spacings = np.diff(axis)
        assert np.allclose(spacings, expected_spacing, rtol=1e-10)

    def test_generate_error_cases(self):
        """Test error handling in generate method."""
        # Invalid bins
        with pytest.raises(ValueError, match="bins must be positive"):
            self.generator.generate(self.metadata, 0)

        with pytest.raises(ValueError, match="bins must be positive"):
            self.generator.generate(self.metadata, -5)

        # Invalid custom range
        with pytest.raises(ValueError, match="Invalid mass range"):
            self.generator.generate(self.metadata, 100, mass_range=(500.0, 100.0))

        # Negative mass range
        with pytest.raises(ValueError, match="Mass values must be positive"):
            self.generator.generate(self.metadata, 100, mass_range=(-10.0, 100.0))

    def test_generate_no_metadata_range(self):
        """Test error when metadata has no mass range and none provided."""
        metadata_no_range = EssentialMetadata(
            dimensions=(10, 10, 1),
            coordinate_bounds=(0.0, 10.0, 0.0, 10.0),
            mass_range=None,
            pixel_size=None,
            n_spectra=100,
            estimated_memory_gb=0.01,
            source_path="/test/path",
        )

        with pytest.raises(ValueError, match="No mass range available"):
            self.generator.generate(metadata_no_range, 100)

    def test_generate_mathematical_properties(self):
        """Test mathematical properties of generated axis."""
        axis = self.generator.generate(self.metadata, 1000)

        # Test that it covers the full range
        assert axis.min() == 100.0
        assert axis.max() == 500.0

        # Test that it's strictly increasing
        assert np.all(np.diff(axis) > 0)

        # Test uniform spacing (all differences should be equal)
        spacings = np.diff(axis)
        assert np.allclose(spacings, spacings[0], rtol=1e-12)

        # Test that the axis is evenly distributed
        expected_values = np.linspace(100.0, 500.0, 1000)
        assert np.allclose(axis, expected_values, rtol=1e-12)

    def test_generate_precision_edge_cases(self):
        """Test precision in edge cases."""
        # Very small mass values
        small_range = (0.001, 0.002)
        axis_small = self.generator.generate(self.metadata, 100, mass_range=small_range)
        assert axis_small[0] == 0.001
        assert axis_small[-1] == 0.002

        # Very large mass values
        large_range = (10000.0, 50000.0)
        axis_large = self.generator.generate(self.metadata, 100, mass_range=large_range)
        assert axis_large[0] == 10000.0
        assert axis_large[-1] == 50000.0
