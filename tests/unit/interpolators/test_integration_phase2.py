"""Integration tests for interpolation components (Phase 2)."""

import numpy as np

from msiconvert.interpolators.mass_axis.linear_generator import LinearMassAxisGenerator
from msiconvert.interpolators.pchip_interpolator import PchipInterpolator
from msiconvert.metadata.types import EssentialMetadata


class TestInterpolationIntegration:
    """Integration tests for mass axis generation + interpolation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mass_axis_generator = LinearMassAxisGenerator()
        self.interpolator = PchipInterpolator(extrapolate=False)

        # Test metadata
        self.metadata = EssentialMetadata(
            dimensions=(50, 50, 1),
            coordinate_bounds=(0.0, 1000.0, 0.0, 1000.0),
            mass_range=(150.0, 850.0),
            pixel_size=20.0,
            n_spectra=2500,
            estimated_memory_gb=1.0,
            source_path="data/integration_test.imzML",
        )

    def test_basic_workflow(self):
        """Test basic mass axis generation -> interpolation workflow."""
        # Step 1: Generate target mass axis
        target_axis = self.mass_axis_generator.generate(self.metadata, 1000)

        # Step 2: Create synthetic spectrum data
        # Irregular m/z values (typical of real spectra)
        original_mz = np.array(
            [155.0, 172.5, 203.7, 245.2, 298.1, 356.8, 412.3, 567.9, 723.4, 798.6]
        )
        original_intensities = np.array(
            [25.0, 45.0, 120.0, 80.0, 200.0, 150.0, 60.0, 90.0, 110.0, 40.0]
        )

        # Step 3: Interpolate onto target axis
        interpolated = self.interpolator.interpolate(
            original_mz, original_intensities, target_axis
        )

        # Validate integration
        assert interpolated.shape == target_axis.shape
        assert target_axis.shape == (1000,)
        assert np.all(np.isfinite(interpolated))
        assert np.all(interpolated >= 0)

        # Check that interpolation preserves key features
        assert np.max(interpolated) > 0  # Should have some signal
        assert np.sum(interpolated > 0) < len(
            interpolated
        )  # Not all points should be positive

    def test_metadata_driven_workflow(self):
        """Test full metadata-driven interpolation workflow."""
        # Use metadata to determine parameters
        bins = 5000  # High resolution

        # Generate mass axis from metadata
        target_axis = self.mass_axis_generator.generate(self.metadata, bins)

        # Verify mass axis properties
        assert len(target_axis) == bins
        assert target_axis[0] == self.metadata.mass_range[0]
        assert target_axis[-1] == self.metadata.mass_range[1]

        # Create realistic spectrum within metadata range
        n_peaks = 20
        peak_mz = np.linspace(160.0, 840.0, n_peaks)  # Spread across range
        peak_intensities = np.random.exponential(
            50.0, n_peaks
        )  # Realistic distribution

        # Interpolate
        result = self.interpolator.interpolate(peak_mz, peak_intensities, target_axis)

        # Validate result
        assert result.shape == (bins,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

        # Check intensity conservation (approximately)
        original_total = self.interpolator.calculate_total_intensity(
            peak_intensities, peak_mz
        )
        interpolated_total = self.interpolator.calculate_total_intensity(
            result, target_axis
        )
        relative_error = abs(interpolated_total - original_total) / original_total
        assert relative_error < 0.15  # Within 15% is reasonable

    def test_edge_case_integration(self):
        """Test integration with edge cases."""
        test_cases = [
            {
                "name": "sparse_spectrum",
                "mz": np.array([200.0, 700.0]),
                "intensities": np.array([100.0, 50.0]),
                "bins": 100,
            },
            {
                "name": "single_peak",
                "mz": np.array([400.0]),
                "intensities": np.array([200.0]),
                "bins": 50,
            },
            {
                "name": "many_small_peaks",
                "mz": np.linspace(200.0, 800.0, 100),
                "intensities": np.random.exponential(10.0, 100),
                "bins": 1000,
            },
        ]

        for case in test_cases:
            target_axis = self.mass_axis_generator.generate(self.metadata, case["bins"])

            result = self.interpolator.interpolate(
                case["mz"], case["intensities"], target_axis
            )

            # All cases should produce valid results
            assert result.shape == (case["bins"],), f"Failed for {case['name']}"
            assert np.all(np.isfinite(result)), f"Non-finite values in {case['name']}"
            assert np.all(result >= 0), f"Negative values in {case['name']}"

    def test_different_mass_ranges(self):
        """Test integration with different mass ranges."""
        test_ranges = [
            (100.0, 200.0),  # Narrow range
            (50.0, 2000.0),  # Wide range
            (500.0, 600.0),  # High m/z narrow range
            (0.1, 50.0),  # Low m/z range
        ]

        for min_mz, max_mz in test_ranges:
            # Create metadata for this range
            custom_metadata = EssentialMetadata(
                dimensions=(10, 10, 1),
                coordinate_bounds=(0.0, 200.0, 0.0, 200.0),
                mass_range=(min_mz, max_mz),
                pixel_size=20.0,
                n_spectra=100,
                estimated_memory_gb=0.1,
                source_path=f"data/range_{min_mz}_{max_mz}.imzML",
            )

            # Generate target axis
            target_axis = self.mass_axis_generator.generate(custom_metadata, 500)

            # Create spectrum in this range
            spectrum_mz = np.linspace(min_mz + 0.1, max_mz - 0.1, 10)
            spectrum_intensities = np.random.exponential(20.0, 10)

            # Interpolate
            result = self.interpolator.interpolate(
                spectrum_mz, spectrum_intensities, target_axis
            )

            # Validate
            assert result.shape == (500,)
            assert np.all(np.isfinite(result))
            assert np.all(result >= 0)
            assert np.any(result > 0)  # Should have some signal

    def test_resolution_effects(self):
        """Test effects of different target axis resolutions."""
        # Fixed spectrum
        spectrum_mz = np.array([200.0, 300.0, 400.0, 500.0, 600.0])
        spectrum_intensities = np.array([50.0, 100.0, 75.0, 125.0, 60.0])

        resolutions = [10, 50, 100, 500, 1000, 5000]

        for bins in resolutions:
            target_axis = self.mass_axis_generator.generate(self.metadata, bins)

            result = self.interpolator.interpolate(
                spectrum_mz, spectrum_intensities, target_axis
            )

            # Higher resolution should provide more detail
            assert result.shape == (bins,)
            assert np.all(np.isfinite(result))
            assert np.all(result >= 0)

            # Check that total intensity is approximately preserved
            original_total = self.interpolator.calculate_total_intensity(
                spectrum_intensities, spectrum_mz
            )
            interpolated_total = self.interpolator.calculate_total_intensity(
                result, target_axis
            )

            relative_error = abs(interpolated_total - original_total) / original_total
            assert relative_error < 0.2, f"Large error at resolution {bins}"

    def test_realistic_msi_simulation(self):
        """Test with realistic MSI-like data."""
        # Simulate what a real MSI pixel might look like

        # Generate realistic m/z values (not perfectly uniform)
        base_mz = np.linspace(200.0, 800.0, 50)
        # Add some jitter to make it realistic
        realistic_mz = base_mz + np.random.normal(0, 0.5, 50)
        realistic_mz = np.sort(realistic_mz)

        # Generate realistic intensities (exponential distribution with some peaks)
        baseline_intensities = np.random.exponential(5.0, 50)
        # Add some prominent peaks
        peak_indices = [10, 25, 35, 42]
        for idx in peak_indices:
            baseline_intensities[idx] += np.random.exponential(100.0)

        # Generate target axis at typical MSI resolution
        target_axis = self.mass_axis_generator.generate(
            self.metadata, 3500
        )  # 0.2 Da spacing

        # Interpolate
        result = self.interpolator.interpolate(
            realistic_mz, baseline_intensities, target_axis
        )

        # Validate realistic properties
        assert result.shape == (3500,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

        # Should have reasonable peak structure
        num_nonzero = np.sum(result > 0)
        assert num_nonzero > 100  # Should have substantial coverage
        assert num_nonzero < len(result)  # But not absolutely everywhere

        # Should preserve approximate total intensity
        original_total = self.interpolator.calculate_total_intensity(
            baseline_intensities, realistic_mz
        )
        interpolated_total = self.interpolator.calculate_total_intensity(
            result, target_axis
        )
        relative_error = abs(interpolated_total - original_total) / original_total
        assert relative_error < 0.5  # Within 50% for realistic noisy data

    def test_performance_with_realistic_sizes(self):
        """Test performance with realistic dataset sizes."""
        # Typical MSI parameters
        large_bins = 10000  # High resolution mass axis
        large_spectrum_size = 200  # Typical number of peaks per pixel

        # Generate large target axis
        target_axis = self.mass_axis_generator.generate(self.metadata, large_bins)

        # Generate large spectrum
        large_mz = np.sort(np.random.uniform(150.0, 850.0, large_spectrum_size))
        large_intensities = np.random.exponential(30.0, large_spectrum_size)

        # This should complete without error in reasonable time
        result = self.interpolator.interpolate(large_mz, large_intensities, target_axis)

        # Validate
        assert result.shape == (large_bins,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_custom_mass_range_override(self):
        """Test that custom mass ranges work in integration."""
        # Use custom range that differs from metadata
        custom_range = (300.0, 600.0)

        # Generate axis with custom range
        target_axis = self.mass_axis_generator.generate(
            self.metadata, 500, mass_range=custom_range
        )

        # Create spectrum in custom range
        spectrum_mz = np.array([320.0, 400.0, 480.0, 560.0])
        spectrum_intensities = np.array([40.0, 80.0, 60.0, 50.0])

        # Interpolate
        result = self.interpolator.interpolate(
            spectrum_mz, spectrum_intensities, target_axis
        )

        # Validate custom range was used
        assert target_axis[0] == 300.0
        assert target_axis[-1] == 600.0
        assert result.shape == (500,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        assert np.any(result > 0)
