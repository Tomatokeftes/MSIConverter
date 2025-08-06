"""Tests for BaseInterpolator abstract class."""

import numpy as np
import pytest

from msiconvert.interpolators.base_interpolator import BaseInterpolator


class ConcreteInterpolator(BaseInterpolator):
    """Concrete implementation for testing abstract methods."""

    def interpolate(self, mz_values, intensities, target_axis):
        """Simple linear interpolation for testing."""
        self._validate_inputs(mz_values, intensities, target_axis)
        mz_clean, int_clean = self._remove_duplicates(mz_values, intensities)

        # Simple linear interpolation
        return np.interp(target_axis, mz_clean, int_clean)


class TestBaseInterpolator:
    """Tests for BaseInterpolator validation and utility methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.interpolator = ConcreteInterpolator()

        # Valid test data
        self.valid_mz = np.array([100.0, 200.0, 300.0, 400.0])
        self.valid_intensities = np.array([10.0, 20.0, 15.0, 25.0])
        self.valid_target = np.linspace(100.0, 400.0, 31)

    def test_validate_inputs_valid_data(self):
        """Test input validation with valid data."""
        # Should not raise any exceptions
        self.interpolator._validate_inputs(
            self.valid_mz, self.valid_intensities, self.valid_target
        )

    def test_validate_inputs_empty_arrays(self):
        """Test validation with empty arrays."""
        empty = np.array([])

        with pytest.raises(ValueError, match="m/z values array cannot be empty"):
            self.interpolator._validate_inputs(
                empty, self.valid_intensities, self.valid_target
            )

        with pytest.raises(ValueError, match="Intensities array cannot be empty"):
            self.interpolator._validate_inputs(self.valid_mz, empty, self.valid_target)

        with pytest.raises(ValueError, match="Target axis cannot be empty"):
            self.interpolator._validate_inputs(
                self.valid_mz, self.valid_intensities, empty
            )

    def test_validate_inputs_mismatched_shapes(self):
        """Test validation with mismatched array shapes."""
        wrong_intensities = np.array([10.0, 20.0])  # Wrong length

        with pytest.raises(ValueError, match="must have same shape"):
            self.interpolator._validate_inputs(
                self.valid_mz, wrong_intensities, self.valid_target
            )

    def test_validate_inputs_non_finite_values(self):
        """Test validation with non-finite values."""
        # NaN values
        mz_with_nan = np.array([100.0, np.nan, 300.0, 400.0])
        with pytest.raises(ValueError, match="m/z values must be finite"):
            self.interpolator._validate_inputs(
                mz_with_nan, self.valid_intensities, self.valid_target
            )

        # Infinite intensities
        int_with_inf = np.array([10.0, np.inf, 15.0, 25.0])
        with pytest.raises(ValueError, match="Intensities must be finite"):
            self.interpolator._validate_inputs(
                self.valid_mz, int_with_inf, self.valid_target
            )

        # NaN target axis
        target_with_nan = np.array([100.0, 150.0, np.nan, 250.0])
        with pytest.raises(ValueError, match="Target axis must be finite"):
            self.interpolator._validate_inputs(
                self.valid_mz, self.valid_intensities, target_with_nan
            )

    def test_validate_inputs_negative_values(self):
        """Test validation with negative values."""
        # Negative m/z
        mz_negative = np.array([-100.0, 200.0, 300.0, 400.0])
        with pytest.raises(ValueError, match="m/z values must be non-negative"):
            self.interpolator._validate_inputs(
                mz_negative, self.valid_intensities, self.valid_target
            )

        # Negative intensities
        int_negative = np.array([10.0, -20.0, 15.0, 25.0])
        with pytest.raises(ValueError, match="Intensities must be non-negative"):
            self.interpolator._validate_inputs(
                self.valid_mz, int_negative, self.valid_target
            )

        # Negative target axis
        target_negative = np.array([-50.0, 100.0, 200.0, 300.0])
        with pytest.raises(ValueError, match="Target axis must be non-negative"):
            self.interpolator._validate_inputs(
                self.valid_mz, self.valid_intensities, target_negative
            )

    def test_validate_inputs_sorting_requirements(self):
        """Test validation of sorting requirements."""
        # Unsorted m/z values
        mz_unsorted = np.array([200.0, 100.0, 400.0, 300.0])
        with pytest.raises(ValueError, match="m/z values must be sorted"):
            self.interpolator._validate_inputs(
                mz_unsorted, self.valid_intensities, self.valid_target
            )

        # Unsorted target axis
        target_unsorted = np.array([200.0, 100.0, 300.0, 400.0])
        with pytest.raises(ValueError, match="Target axis must be sorted"):
            self.interpolator._validate_inputs(
                self.valid_mz, self.valid_intensities, target_unsorted
            )

        # Target axis with duplicates (not strictly ascending)
        target_duplicates = np.array([100.0, 200.0, 200.0, 300.0])
        with pytest.raises(
            ValueError, match="Target axis must be sorted in strictly ascending order"
        ):
            self.interpolator._validate_inputs(
                self.valid_mz, self.valid_intensities, target_duplicates
            )

    def test_validate_inputs_allows_mz_duplicates(self):
        """Test that m/z duplicates are allowed (will be handled separately)."""
        mz_with_duplicates = np.array([100.0, 200.0, 200.0, 300.0])
        int_with_duplicates = np.array([10.0, 15.0, 20.0, 25.0])

        # Should not raise - duplicates are handled by _remove_duplicates
        self.interpolator._validate_inputs(
            mz_with_duplicates, int_with_duplicates, self.valid_target
        )

    def test_remove_duplicates_no_duplicates(self):
        """Test duplicate removal with no duplicates."""
        unique_mz, unique_int = self.interpolator._remove_duplicates(
            self.valid_mz, self.valid_intensities
        )

        np.testing.assert_array_equal(unique_mz, self.valid_mz)
        np.testing.assert_array_equal(unique_int, self.valid_intensities)

    def test_remove_duplicates_with_duplicates(self):
        """Test duplicate removal with actual duplicates."""
        mz_with_dups = np.array([100.0, 200.0, 200.0, 200.0, 300.0])
        int_with_dups = np.array([10.0, 5.0, 10.0, 15.0, 20.0])

        unique_mz, unique_int = self.interpolator._remove_duplicates(
            mz_with_dups, int_with_dups
        )

        expected_mz = np.array([100.0, 200.0, 300.0])
        expected_int = np.array([10.0, 30.0, 20.0])  # 5+10+15=30 for m/z=200

        np.testing.assert_array_equal(unique_mz, expected_mz)
        np.testing.assert_array_equal(unique_int, expected_int)

    def test_remove_duplicates_single_value(self):
        """Test duplicate removal with single value."""
        single_mz = np.array([200.0])
        single_int = np.array([25.0])

        unique_mz, unique_int = self.interpolator._remove_duplicates(
            single_mz, single_int
        )

        np.testing.assert_array_equal(unique_mz, single_mz)
        np.testing.assert_array_equal(unique_int, single_int)

    def test_remove_duplicates_empty_arrays(self):
        """Test duplicate removal with empty arrays."""
        empty = np.array([])

        unique_mz, unique_int = self.interpolator._remove_duplicates(empty, empty)

        assert unique_mz.size == 0
        assert unique_int.size == 0

    def test_handle_extrapolation_full_coverage(self):
        """Test extrapolation handling when target is fully covered."""
        mz_range = (90.0, 450.0)  # Covers full target range
        mask = self.interpolator._handle_extrapolation(self.valid_target, mz_range)

        # All points should be within range
        assert np.all(mask)

    def test_handle_extrapolation_partial_coverage(self):
        """Test extrapolation handling with partial coverage."""
        mz_range = (150.0, 350.0)  # Partial coverage
        mask = self.interpolator._handle_extrapolation(self.valid_target, mz_range)

        # Check which points are in range
        expected_mask = (self.valid_target >= 150.0) & (self.valid_target <= 350.0)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_handle_extrapolation_no_coverage(self):
        """Test extrapolation handling with no coverage."""
        mz_range = (500.0, 600.0)  # No coverage
        mask = self.interpolator._handle_extrapolation(self.valid_target, mz_range)

        # No points should be within range
        assert not np.any(mask)

    def test_calculate_total_intensity_basic(self):
        """Test total intensity calculation."""
        intensities = np.array([10.0, 20.0, 15.0])
        axis = np.array([100.0, 200.0, 300.0])

        total = self.interpolator.calculate_total_intensity(intensities, axis)

        # Should use trapezoidal integration
        expected = np.trapz(intensities, axis)
        assert abs(total - expected) < 1e-10

    def test_calculate_total_intensity_single_point(self):
        """Test total intensity with single point."""
        intensities = np.array([25.0])
        axis = np.array([200.0])

        total = self.interpolator.calculate_total_intensity(intensities, axis)

        # Single point should just return the intensity
        assert total == 25.0

    def test_calculate_total_intensity_mismatched_sizes(self):
        """Test total intensity calculation with mismatched array sizes."""
        intensities = np.array([10.0, 20.0])
        axis = np.array([100.0, 200.0, 300.0])

        with pytest.raises(ValueError, match="must have same size"):
            self.interpolator.calculate_total_intensity(intensities, axis)

    def test_interpolate_integration(self):
        """Test full interpolation integration."""
        result = self.interpolator.interpolate(
            self.valid_mz, self.valid_intensities, self.valid_target
        )

        # Check result properties
        assert result.shape == self.valid_target.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Linear interpolation preserves non-negativity

        # Check endpoint values
        assert result[0] == self.valid_intensities[0]  # First point should match
        assert result[-1] == self.valid_intensities[-1]  # Last point should match
