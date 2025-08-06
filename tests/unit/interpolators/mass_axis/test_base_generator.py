"""Tests for BaseMassAxisGenerator abstract class."""

import numpy as np
import pytest

from msiconvert.interpolators.mass_axis.base_generator import BaseMassAxisGenerator
from msiconvert.metadata.types import EssentialMetadata


class ConcreteMassAxisGenerator(BaseMassAxisGenerator):
    """Concrete implementation for testing abstract methods."""

    def generate(self, metadata, bins, mass_range=None):
        """Simple implementation for testing."""
        extracted_range = self._extract_mass_range(metadata, mass_range)
        self._validate_parameters(bins, extracted_range)

        # Simple linear implementation for testing
        min_mass, max_mass = extracted_range
        return np.linspace(min_mass, max_mass, bins)


class TestBaseMassAxisGenerator:
    """Tests for BaseMassAxisGenerator validation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ConcreteMassAxisGenerator()
        self.valid_metadata = EssentialMetadata(
            dimensions=(10, 10, 1),
            coordinate_bounds=(0.0, 10.0, 0.0, 10.0),
            mass_range=(100.0, 500.0),
            pixel_size=None,
            n_spectra=100,
            estimated_memory_gb=0.01,
            source_path="/test/path",
        )

    def test_validate_parameters_valid_input(self):
        """Test parameter validation with valid inputs."""
        # Should not raise any exceptions
        self.generator._validate_parameters(1000, (100.0, 500.0))
        self.generator._validate_parameters(1, (1.0, 2.0))

    def test_validate_parameters_invalid_bins(self):
        """Test parameter validation with invalid bin counts."""
        with pytest.raises(ValueError, match="bins must be positive"):
            self.generator._validate_parameters(0, (100.0, 500.0))

        with pytest.raises(ValueError, match="bins must be positive"):
            self.generator._validate_parameters(-1, (100.0, 500.0))

    def test_validate_parameters_invalid_mass_range(self):
        """Test parameter validation with invalid mass ranges."""
        # min >= max
        with pytest.raises(ValueError, match="Invalid mass range"):
            self.generator._validate_parameters(100, (500.0, 100.0))

        # min == max
        with pytest.raises(ValueError, match="Invalid mass range"):
            self.generator._validate_parameters(100, (100.0, 100.0))

        # Negative masses
        with pytest.raises(ValueError, match="Mass values must be positive"):
            self.generator._validate_parameters(100, (-10.0, 100.0))

    def test_extract_mass_range_from_custom(self):
        """Test mass range extraction from custom parameters."""
        custom_range = (200.0, 400.0)
        result = self.generator._extract_mass_range(self.valid_metadata, custom_range)
        assert result == custom_range

    def test_extract_mass_range_from_metadata(self):
        """Test mass range extraction from metadata."""
        result = self.generator._extract_mass_range(self.valid_metadata, None)
        assert result == (100.0, 500.0)

    def test_extract_mass_range_no_metadata(self):
        """Test mass range extraction with no metadata range."""
        metadata_no_range = EssentialMetadata(
            dimensions=(10, 10, 1),
            coordinate_bounds=(0.0, 10.0, 0.0, 10.0),
            mass_range=None,  # No mass range
            pixel_size=None,
            n_spectra=100,
            estimated_memory_gb=0.01,
            source_path="/test/path",
        )

        with pytest.raises(ValueError, match="No mass range available"):
            self.generator._extract_mass_range(metadata_no_range, None)

    def test_generate_integration(self):
        """Test full generate method integration."""
        # Test with metadata range
        axis = self.generator.generate(self.valid_metadata, 100)
        assert len(axis) == 100
        assert axis[0] == 100.0
        assert axis[-1] == 500.0

        # Test with custom range
        axis_custom = self.generator.generate(
            self.valid_metadata, 50, mass_range=(200.0, 300.0)
        )
        assert len(axis_custom) == 50
        assert axis_custom[0] == 200.0
        assert axis_custom[-1] == 300.0

    def test_generate_error_propagation(self):
        """Test that generate method propagates validation errors."""
        # Invalid bins
        with pytest.raises(ValueError, match="bins must be positive"):
            self.generator.generate(self.valid_metadata, -1)

        # Invalid custom range
        with pytest.raises(ValueError, match="Invalid mass range"):
            self.generator.generate(self.valid_metadata, 100, mass_range=(500.0, 100.0))
