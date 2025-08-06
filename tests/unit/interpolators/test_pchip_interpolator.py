"""Tests for PchipInterpolator."""

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator as ScipyPchip

from msiconvert.interpolators.pchip_interpolator import PchipInterpolator


class TestPchipInterpolator:
    """Tests for PchipInterpolator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.interpolator = PchipInterpolator(extrapolate=False)
        self.interpolator_extrap = PchipInterpolator(extrapolate=True)

        # Standard test data
        self.mz_values = np.array([100.0, 200.0, 300.0, 400.0])
        self.intensities = np.array([10.0, 30.0, 20.0, 40.0])
        self.target_axis = np.linspace(100.0, 400.0, 31)

    def test_init_parameters(self):
        """Test interpolator initialization."""
        interp1 = PchipInterpolator()
        assert interp1.extrapolate is False

        interp2 = PchipInterpolator(extrapolate=True)
        assert interp2.extrapolate is True

    def test_basic_interpolation(self):
        """Test basic PCHIP interpolation functionality."""
        result = self.interpolator.interpolate(
            self.mz_values, self.intensities, self.target_axis
        )

        # Check basic properties
        assert result.shape == self.target_axis.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Should be non-negative

        # Check endpoint values (allow small numerical errors)
        np.testing.assert_allclose(
            result[0], self.intensities[0], rtol=1e-10
        )  # 10.0 at m/z=100
        np.testing.assert_allclose(
            result[-1], self.intensities[-1], rtol=1e-10
        )  # 40.0 at m/z=400

    def test_scipy_compatibility(self):
        """Test that our interpolator matches scipy PCHIP exactly."""
        result = self.interpolator_extrap.interpolate(
            self.mz_values, self.intensities, self.target_axis
        )

        # Compare with direct scipy usage
        scipy_pchip = ScipyPchip(self.mz_values, self.intensities)
        expected = scipy_pchip(self.target_axis)

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_monotonicity_preservation(self):
        """Test that PCHIP preserves monotonicity."""
        # Create monotonic increasing data
        mz_mono = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        int_mono = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        target = np.linspace(100.0, 500.0, 100)

        result = self.interpolator.interpolate(mz_mono, int_mono, target)

        # Result should be monotonic increasing
        assert np.all(np.diff(result) >= 0)

        # Test monotonic decreasing data
        int_mono_dec = np.array([25.0, 20.0, 15.0, 10.0, 5.0])
        result_dec = self.interpolator.interpolate(mz_mono, int_mono_dec, target)

        # Result should be monotonic decreasing
        assert np.all(np.diff(result_dec) <= 0)

    def test_single_point_handling(self):
        """Test interpolation with single input point."""
        single_mz = np.array([250.0])
        single_intensity = np.array([100.0])

        result = self.interpolator.interpolate(
            single_mz, single_intensity, self.target_axis
        )

        # Should be zero everywhere except possibly at the exact m/z
        assert np.sum(result) <= 100.0  # At most the input intensity

        # If target axis contains the exact m/z, it should have the intensity
        if 250.0 in self.target_axis:
            idx = np.where(np.isclose(self.target_axis, 250.0))[0]
            if len(idx) > 0:
                assert result[idx[0]] == 100.0

    def test_duplicate_mz_handling(self):
        """Test handling of duplicate m/z values."""
        mz_with_dups = np.array([100.0, 200.0, 200.0, 300.0])
        int_with_dups = np.array([10.0, 15.0, 25.0, 30.0])  # 200.0 -> 15+25=40

        result = self.interpolator.interpolate(
            mz_with_dups, int_with_dups, self.target_axis
        )

        # Should handle duplicates by summing intensities
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

        # Value at m/z=200 should reflect summed intensity
        idx_200 = np.argmin(np.abs(self.target_axis - 200.0))
        # The exact value depends on interpolation, but should be reasonable
        assert result[idx_200] > 0

    def test_extrapolation_disabled(self):
        """Test behavior when extrapolation is disabled."""
        # Target axis extending beyond input range
        extended_target = np.linspace(50.0, 450.0, 41)  # Extends beyond 100-400

        result = self.interpolator.interpolate(
            self.mz_values, self.intensities, extended_target
        )

        # Points outside input range should be zero
        below_range = extended_target < 100.0
        above_range = extended_target > 400.0

        assert np.all(result[below_range] == 0.0)
        assert np.all(result[above_range] == 0.0)

        # Points within range should be positive
        in_range = (extended_target >= 100.0) & (extended_target <= 400.0)
        assert np.any(result[in_range] > 0)

    def test_extrapolation_enabled(self):
        """Test behavior when extrapolation is enabled."""
        # Target axis extending beyond input range
        extended_target = np.linspace(50.0, 450.0, 41)

        result = self.interpolator_extrap.interpolate(
            self.mz_values, self.intensities, extended_target
        )

        # Should handle extrapolation gracefully
        # (PCHIP extrapolation may produce NaN, which we convert to zeros)
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

        # Points within range should still be positive
        in_range = (extended_target >= 100.0) & (extended_target <= 400.0)
        assert np.any(result[in_range] > 0)

    def test_intensity_conservation_approximate(self):
        """Test approximate intensity conservation."""
        result = self.interpolator.interpolate(
            self.mz_values, self.intensities, self.target_axis
        )

        # Calculate total intensities using trapezoidal integration
        original_total = self.interpolator.calculate_total_intensity(
            self.intensities, self.mz_values
        )
        interpolated_total = self.interpolator.calculate_total_intensity(
            result, self.target_axis
        )

        # Should be approximately conserved (within reasonable tolerance)
        relative_error = abs(interpolated_total - original_total) / original_total
        assert relative_error < 0.1  # Within 10% is reasonable for interpolation

    def test_edge_case_empty_intersection(self):
        """Test case where target axis doesn't overlap with input range."""
        non_overlapping_target = np.linspace(500.0, 600.0, 10)

        result = self.interpolator.interpolate(
            self.mz_values, self.intensities, non_overlapping_target
        )

        # Should be all zeros when no extrapolation
        assert np.all(result == 0.0)

    def test_edge_case_identical_values(self):
        """Test case with identical intensity values."""
        identical_intensities = np.array([25.0, 25.0, 25.0, 25.0])

        result = self.interpolator.interpolate(
            self.mz_values, identical_intensities, self.target_axis
        )

        # Should produce constant values
        assert np.all(np.isclose(result, 25.0, rtol=1e-10))

    def test_error_propagation(self):
        """Test that base class errors are properly propagated."""
        # Empty arrays
        with pytest.raises(ValueError, match="cannot be empty"):
            self.interpolator.interpolate(np.array([]), np.array([]), self.target_axis)

        # Mismatched shapes
        with pytest.raises(ValueError, match="must have same shape"):
            self.interpolator.interpolate(
                self.mz_values, np.array([10.0, 20.0]), self.target_axis
            )

        # Unsorted m/z values
        unsorted_mz = np.array([200.0, 100.0, 300.0])
        unsorted_int = np.array([20.0, 10.0, 30.0])
        with pytest.raises(ValueError, match="must be sorted"):
            self.interpolator.interpolate(unsorted_mz, unsorted_int, self.target_axis)

    def test_performance_large_arrays(self):
        """Test performance with larger arrays."""
        # Create larger test data
        large_mz = np.linspace(100.0, 1000.0, 1000)
        large_intensities = np.random.random(1000) * 100
        large_target = np.linspace(100.0, 1000.0, 10000)

        # Should complete without error
        result = self.interpolator.interpolate(
            large_mz, large_intensities, large_target
        )

        assert result.shape == large_target.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_get_interpolation_info(self):
        """Test interpolation information method."""
        info = self.interpolator.get_interpolation_info()

        expected_keys = {
            "method",
            "extrapolate",
            "preserves_monotonicity",
            "avoids_oscillations",
            "backend",
        }
        assert set(info.keys()) == expected_keys

        assert info["method"] == "PCHIP"
        assert info["extrapolate"] is False
        assert info["preserves_monotonicity"] is True
        assert info["avoids_oscillations"] is True

    def test_realistic_spectrum_data(self):
        """Test with realistic spectrum-like data."""
        # Simulate a realistic mass spectrum with multiple peaks
        mz_realistic = np.array(
            [
                150.0,
                175.0,
                200.0,
                225.0,
                250.0,
                275.0,
                300.0,
                325.0,
                350.0,
                375.0,
                400.0,
            ]
        )

        # Intensities with some peaks and valleys
        int_realistic = np.array(
            [5.0, 15.0, 45.0, 30.0, 80.0, 20.0, 60.0, 10.0, 35.0, 25.0, 15.0]
        )

        target_realistic = np.linspace(150.0, 400.0, 500)

        result = self.interpolator.interpolate(
            mz_realistic, int_realistic, target_realistic
        )

        # Should produce reasonable spectrum-like output
        assert result.shape == target_realistic.shape
        assert np.all(result >= 0)
        assert np.max(result) <= np.max(int_realistic) * 1.1  # Reasonable peak height
        assert np.sum(result > 0) > len(target_realistic) * 0.1  # Some coverage
