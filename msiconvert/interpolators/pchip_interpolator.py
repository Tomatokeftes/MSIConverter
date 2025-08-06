"""PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator."""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator as ScipyPchip

from .base_interpolator import BaseInterpolator


class PchipInterpolator(BaseInterpolator):
    """PCHIP interpolator for spectral data.

    Uses scipy's PchipInterpolator for monotonicity-preserving interpolation
    that avoids spurious oscillations in spectral data.
    """

    def __init__(self, extrapolate: bool = False):
        """Initialize PCHIP interpolator.

        Args:
            extrapolate: If True, extrapolate beyond input range. If False,
                        return zeros for points outside input range.
        """
        self.extrapolate = extrapolate

    def interpolate(
        self,
        mz_values: NDArray[np.float64],
        intensities: NDArray[np.float64],
        target_axis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate spectrum using PCHIP method.

        Args:
            mz_values: Original m/z values (must be sorted)
            intensities: Corresponding intensity values
            target_axis: Target mass axis for interpolation (must be sorted)

        Returns:
            Interpolated intensity values on target axis

        Raises:
            ValueError: If inputs are invalid or PCHIP interpolation fails
        """
        # Validate inputs
        self._validate_inputs(mz_values, intensities, target_axis)

        # Handle edge cases
        if mz_values.size == 1:
            return self._handle_single_point(mz_values, intensities, target_axis)

        # Remove duplicates by summing intensities
        mz_clean, intensities_clean = self._remove_duplicates(mz_values, intensities)

        # Check if we still have enough points after duplicate removal
        if mz_clean.size == 1:
            return self._handle_single_point(mz_clean, intensities_clean, target_axis)

        # Create PCHIP interpolator
        try:
            pchip = ScipyPchip(mz_clean, intensities_clean, extrapolate=False)
        except ValueError as e:
            raise ValueError(f"Failed to create PCHIP interpolator: {e}") from e

        # Interpolate values
        result = np.zeros_like(target_axis)

        if self.extrapolate:
            # Extrapolate for all points, but handle NaN values
            result = pchip(target_axis)
            # Replace NaN values with zeros (safer than extrapolation artifacts)
            result = np.where(np.isfinite(result), result, 0.0)
        else:
            # Only interpolate points within the input range
            mz_range = (mz_clean[0], mz_clean[-1])
            in_range_mask = self._handle_extrapolation(target_axis, mz_range)

            if np.any(in_range_mask):
                result[in_range_mask] = pchip(target_axis[in_range_mask])
            # Points outside range remain zero

        # Ensure non-negative values (PCHIP can sometimes produce small negative values)
        result = np.maximum(result, 0.0)

        return result

    def _handle_single_point(
        self,
        mz_values: NDArray[np.float64],
        intensities: NDArray[np.float64],
        target_axis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Handle interpolation when input has only one point.

        Args:
            mz_values: Single m/z value
            intensities: Single intensity value
            target_axis: Target mass axis

        Returns:
            Interpolated values (intensity at exact m/z, zeros elsewhere)
        """
        result = np.zeros_like(target_axis)

        # Find exact matches in target axis
        exact_matches = np.isclose(target_axis, mz_values[0], rtol=1e-10)

        if np.any(exact_matches):
            result[exact_matches] = intensities[0]

        return result

    def get_interpolation_info(self) -> dict:
        """Get information about the interpolation settings.

        Returns:
            Dictionary with interpolation parameters
        """
        return {
            "method": "PCHIP",
            "extrapolate": self.extrapolate,
            "preserves_monotonicity": True,
            "avoids_oscillations": True,
            "backend": "scipy.interpolate.PchipInterpolator",
        }
