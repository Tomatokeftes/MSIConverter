"""Integration tests for mass axis generators with existing metadata system."""

import numpy as np
import pytest

from msiconvert.interpolators.mass_axis.linear_generator import LinearMassAxisGenerator
from msiconvert.metadata.types import EssentialMetadata


class TestMassAxisIntegration:
    """Integration tests with existing metadata system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = LinearMassAxisGenerator()

    def test_integration_with_essential_metadata(self):
        """Test integration using EssentialMetadata structure."""
        # Create metadata that matches what readers would produce
        metadata = EssentialMetadata(
            dimensions=(50, 50, 1),
            coordinate_bounds=(0.0, 1000.0, 0.0, 1000.0),
            mass_range=(150.5, 850.3),
            pixel_size=20.0,
            n_spectra=2500,
            estimated_memory_gb=0.5,
            source_path="data/test_sample.imzML",
        )

        # Generate mass axis from this metadata
        axis = self.generator.generate(metadata, 1000)

        # Validate integration
        assert len(axis) == 1000
        assert axis[0] == 150.5
        assert axis[-1] == 850.3
        assert np.all(np.diff(axis) > 0)  # Monotonic

        # Check that spacing is uniform
        spacings = np.diff(axis)
        expected_spacing = (850.3 - 150.5) / (1000 - 1)
        assert np.allclose(spacings, expected_spacing, rtol=1e-10)

    def test_integration_realistic_mass_ranges(self):
        """Test with realistic mass ranges from different instruments."""
        test_cases = [
            # TOF instrument typical range
            {
                "mass_range": (50.0, 1000.0),
                "bins": 5000,
                "description": "TOF typical range",
            },
            # Orbitrap typical range
            {
                "mass_range": (100.0, 2000.0),
                "bins": 10000,
                "description": "Orbitrap typical range",
            },
            # High-resolution range
            {
                "mass_range": (200.0, 800.0),
                "bins": 50000,
                "description": "High-resolution narrow range",
            },
        ]

        for case in test_cases:
            metadata = EssentialMetadata(
                dimensions=(100, 100, 1),
                coordinate_bounds=(0.0, 2000.0, 0.0, 2000.0),
                mass_range=case["mass_range"],
                pixel_size=20.0,
                n_spectra=10000,
                estimated_memory_gb=1.0,
                source_path=f"data/{case['description']}.imzML",
            )

            axis = self.generator.generate(metadata, case["bins"])

            # Validate each case
            assert len(axis) == case["bins"], f"Failed for {case['description']}"
            assert axis[0] == case["mass_range"][0], f"Failed for {case['description']}"
            assert (
                axis[-1] == case["mass_range"][1]
            ), f"Failed for {case['description']}"

            # Check mathematical properties
            assert np.all(np.diff(axis) > 0), f"Not monotonic for {case['description']}"

            # Check uniform spacing
            spacings = np.diff(axis)
            expected_spacing = (case["mass_range"][1] - case["mass_range"][0]) / (
                case["bins"] - 1
            )
            assert np.allclose(
                spacings, expected_spacing, rtol=1e-10
            ), f"Non-uniform spacing for {case['description']}"

    def test_integration_edge_case_metadata(self):
        """Test with edge case metadata that readers might produce."""
        # Very small dataset
        small_metadata = EssentialMetadata(
            dimensions=(1, 1, 1),
            coordinate_bounds=(0.0, 1.0, 0.0, 1.0),
            mass_range=(100.0, 101.0),  # Very narrow range
            pixel_size=1.0,
            n_spectra=1,
            estimated_memory_gb=0.001,
            source_path="data/single_pixel.imzML",
        )

        axis = self.generator.generate(small_metadata, 10)
        assert len(axis) == 10
        assert axis[0] == 100.0
        assert axis[-1] == 101.0

        # Large 3D dataset
        large_metadata = EssentialMetadata(
            dimensions=(200, 200, 50),
            coordinate_bounds=(0.0, 4000.0, 0.0, 4000.0),
            mass_range=(50.0, 2000.0),
            pixel_size=20.0,
            n_spectra=2000000,
            estimated_memory_gb=50.0,
            source_path="data/large_3d_dataset.d",
        )

        axis = self.generator.generate(large_metadata, 100000)
        assert len(axis) == 100000
        assert axis[0] == 50.0
        assert axis[-1] == 2000.0

    def test_custom_range_overrides_metadata(self):
        """Test that custom range properly overrides metadata range."""
        metadata = EssentialMetadata(
            dimensions=(10, 10, 1),
            coordinate_bounds=(0.0, 200.0, 0.0, 200.0),
            mass_range=(100.0, 500.0),  # This should be ignored
            pixel_size=20.0,
            n_spectra=100,
            estimated_memory_gb=0.1,
            source_path="data/test.imzML",
        )

        # Override with custom range
        custom_range = (200.0, 300.0)
        axis = self.generator.generate(metadata, 100, mass_range=custom_range)

        # Should use custom range, not metadata range
        assert axis[0] == 200.0
        assert axis[-1] == 300.0
        assert len(axis) == 100

    def test_integration_error_handling(self):
        """Test error handling in integration scenarios."""
        # Metadata with None mass_range
        invalid_metadata = EssentialMetadata(
            dimensions=(10, 10, 1),
            coordinate_bounds=(0.0, 200.0, 0.0, 200.0),
            mass_range=None,  # Invalid!
            pixel_size=20.0,
            n_spectra=100,
            estimated_memory_gb=0.1,
            source_path="data/invalid.imzML",
        )

        with pytest.raises(ValueError, match="No mass range available"):
            self.generator.generate(invalid_metadata, 100)

    def test_precision_with_realistic_data(self):
        """Test numerical precision with realistic data."""
        # Use actual realistic mass range with high precision
        metadata = EssentialMetadata(
            dimensions=(128, 128, 1),
            coordinate_bounds=(0.0, 2560.0, 0.0, 2560.0),
            mass_range=(150.000123, 850.999876),  # High precision range
            pixel_size=20.0,
            n_spectra=16384,
            estimated_memory_gb=2.5,
            source_path="data/high_precision.imzML",
        )

        axis = self.generator.generate(metadata, 10000)

        # Check precision is maintained
        assert abs(axis[0] - 150.000123) < 1e-10
        assert abs(axis[-1] - 850.999876) < 1e-10

        # Check uniform spacing with high precision
        spacings = np.diff(axis)
        expected_spacing = (850.999876 - 150.000123) / (10000 - 1)
        assert np.allclose(spacings, expected_spacing, rtol=1e-12)

    def test_axis_properties_for_interpolation(self):
        """Test that generated axis has properties suitable for interpolation."""
        metadata = EssentialMetadata(
            dimensions=(64, 64, 1),
            coordinate_bounds=(0.0, 1280.0, 0.0, 1280.0),
            mass_range=(100.0, 800.0),
            pixel_size=20.0,
            n_spectra=4096,
            estimated_memory_gb=1.0,
            source_path="data/interpolation_test.imzML",
        )

        axis = self.generator.generate(metadata, 5000)

        # Properties needed for interpolation
        assert len(axis) == 5000  # Exact number of bins
        assert axis.dtype == np.float64  # Sufficient precision
        assert np.all(np.isfinite(axis))  # No NaN or inf values
        assert np.all(np.diff(axis) > 0)  # Strictly monotonic
        assert axis[0] >= 0  # Positive mass values

        # Check that axis covers full range
        assert axis[0] == metadata.mass_range[0]
        assert axis[-1] == metadata.mass_range[1]

        # Check that axis is suitable for np.searchsorted (what interpolation will use)
        test_values = np.array([150.0, 250.0, 400.0, 600.0, 750.0])
        indices = np.searchsorted(axis, test_values)

        # All indices should be valid
        assert np.all(indices < len(axis))
        assert np.all(indices >= 0)
