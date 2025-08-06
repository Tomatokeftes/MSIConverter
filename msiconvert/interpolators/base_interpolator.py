"""Base abstract class for spectral interpolators."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class BaseInterpolator(ABC):
    """Abstract base class for spectral interpolators.

    Interpolators take irregular m/z and intensity arrays and interpolate
    them onto a uniform target mass axis for consistent spectral processing.
    """

    @abstractmethod
    def interpolate(
        self,
        mz_values: NDArray[np.float64],
        intensities: NDArray[np.float64],
        target_axis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate spectrum onto target mass axis.

        Args:
            mz_values: Original m/z values (must be sorted)
            intensities: Corresponding intensity values
            target_axis: Target mass axis for interpolation (must be sorted)

        Returns:
            Interpolated intensity values on target axis

        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        pass

    def _validate_inputs(
        self,
        mz_values: NDArray[np.float64],
        intensities: NDArray[np.float64],
        target_axis: NDArray[np.float64],
    ) -> None:
        """Validate interpolation inputs.

        Args:
            mz_values: Original m/z values
            intensities: Corresponding intensity values
            target_axis: Target mass axis for interpolation

        Raises:
            ValueError: If inputs are invalid
        """
        self._validate_array_sizes(mz_values, intensities, target_axis)
        self._validate_array_values(mz_values, intensities, target_axis)
        self._validate_array_sorting(mz_values, target_axis)

    def _validate_array_sizes(
        self,
        mz_values: NDArray[np.float64],
        intensities: NDArray[np.float64],
        target_axis: NDArray[np.float64],
    ) -> None:
        """Validate array sizes and shapes."""
        if mz_values.size == 0:
            raise ValueError("m/z values array cannot be empty")
        if intensities.size == 0:
            raise ValueError("Intensities array cannot be empty")
        if target_axis.size == 0:
            raise ValueError("Target axis cannot be empty")

        if mz_values.shape != intensities.shape:
            raise ValueError(
                f"m/z values ({mz_values.shape}) and intensities "
                f"({intensities.shape}) must have same shape"
            )

    def _validate_array_values(
        self,
        mz_values: NDArray[np.float64],
        intensities: NDArray[np.float64],
        target_axis: NDArray[np.float64],
    ) -> None:
        """Validate array values are finite and non-negative."""
        # Check for valid values
        if not np.all(np.isfinite(mz_values)):
            raise ValueError("m/z values must be finite")
        if not np.all(np.isfinite(intensities)):
            raise ValueError("Intensities must be finite")
        if not np.all(np.isfinite(target_axis)):
            raise ValueError("Target axis must be finite")

        if np.any(mz_values < 0):
            raise ValueError("m/z values must be non-negative")
        if np.any(intensities < 0):
            raise ValueError("Intensities must be non-negative")
        if np.any(target_axis < 0):
            raise ValueError("Target axis must be non-negative")

    def _validate_array_sorting(
        self,
        mz_values: NDArray[np.float64],
        target_axis: NDArray[np.float64],
    ) -> None:
        """Validate array sorting requirements."""
        if not np.all(np.diff(mz_values) >= 0):
            raise ValueError("m/z values must be sorted in ascending order")
        if not np.all(np.diff(target_axis) > 0):
            raise ValueError("Target axis must be sorted in strictly ascending order")

    def _remove_duplicates(
        self, mz_values: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Remove duplicate m/z values by summing their intensities.

        Args:
            mz_values: Original m/z values (must be sorted)
            intensities: Corresponding intensity values

        Returns:
            Tuple of (unique_mz, summed_intensities)
        """
        if mz_values.size <= 1:
            return mz_values.copy(), intensities.copy()

        # Find unique values and their indices
        unique_mz, inverse_indices = np.unique(mz_values, return_inverse=True)

        # Sum intensities for duplicate m/z values
        unique_intensities = np.zeros_like(unique_mz)
        np.add.at(unique_intensities, inverse_indices, intensities)

        return unique_mz, unique_intensities

    def _handle_extrapolation(
        self,
        target_axis: NDArray[np.float64],
        mz_range: Tuple[float, float],
    ) -> NDArray[np.bool_]:
        """Identify points that require extrapolation.

        Args:
            target_axis: Target mass axis
            mz_range: (min_mz, max_mz) range of input spectrum

        Returns:
            Boolean mask indicating which target points are within input range
        """
        min_mz, max_mz = mz_range
        return (target_axis >= min_mz) & (target_axis <= max_mz)

    def calculate_total_intensity(
        self,
        intensities: NDArray[np.float64],
        axis: NDArray[np.float64],
    ) -> float:
        """Calculate total intensity using trapezoidal integration.

        Args:
            intensities: Intensity values
            axis: Corresponding mass axis

        Returns:
            Total integrated intensity
        """
        if intensities.size != axis.size:
            raise ValueError("Intensities and axis must have same size")
        if intensities.size < 2:
            return float(np.sum(intensities))

        return float(np.trapz(intensities, axis))
