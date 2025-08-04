# msiconvert/processing/interpolation.py

"""
Spectral interpolation utilities for MSI data processing.

This module provides classes and functions for interpolating mass spectra
to common mass axes, with support for different interpolation strategies
and sparse data representation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Check Dask availability
DASK_AVAILABLE = False
try:
    from dask import delayed

    DASK_AVAILABLE = True
except ImportError:

    def delayed(func):
        return func

    DASK_AVAILABLE = False


@dataclass
class InterpolationResult:
    """Result of spectral interpolation containing sparse representation."""

    coords: Tuple[int, int, int]
    pixel_idx: int
    sparse_indices: NDArray[np.int_]
    sparse_values: NDArray[np.float64]
    tic_value: float

    def __post_init__(self):
        """Validate the interpolation result after initialization."""
        if len(self.sparse_indices) != len(self.sparse_values):
            raise ValueError(
                "sparse_indices and sparse_values must have the same length"
            )
        if self.tic_value < 0:
            raise ValueError("tic_value must be non-negative")


class SpectralInterpolator:
    """
    Handles interpolation of mass spectra to a common mass axis.

    This class provides methods for interpolating individual spectra or
    processing chunks of spectra with various interpolation strategies.
    """

    def __init__(
        self,
        common_mass_axis: NDArray[np.float64],
        interpolation_method: str = "linear",
        fill_value: float = 0.0,
        sparsity_threshold: float = 1e-10,
    ):
        """
        Initialize the spectral interpolator.

        Args:
            common_mass_axis: Target mass axis for interpolation
            interpolation_method: Interpolation method ('linear', 'nearest', 'cubic')
            fill_value: Value to use for out-of-range interpolation
            sparsity_threshold: Minimum value to consider non-zero for sparse representation
        """
        self.common_mass_axis = common_mass_axis
        self.interpolation_method = interpolation_method
        self.fill_value = fill_value
        self.sparsity_threshold = sparsity_threshold

        # Validate inputs
        if len(self.common_mass_axis) == 0:
            raise ValueError("common_mass_axis cannot be empty")
        if not np.all(np.diff(self.common_mass_axis) > 0):
            raise ValueError("common_mass_axis must be strictly increasing")

        logging.debug(
            f"Initialized SpectralInterpolator with {len(self.common_mass_axis)} mass points"
        )

    def interpolate_spectrum(
        self,
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
        coords: Tuple[int, int, int],
        pixel_idx: int,
    ) -> InterpolationResult:
        """
        Interpolate a single spectrum to the common mass axis.

        Args:
            mzs: Original m/z values
            intensities: Original intensity values
            coords: Pixel coordinates (x, y, z)
            pixel_idx: Linear pixel index

        Returns:
            InterpolationResult with sparse representation
        """
        if len(mzs) != len(intensities):
            raise ValueError("mzs and intensities must have the same length")

        if len(mzs) == 0:
            # Empty spectrum
            return InterpolationResult(
                coords=coords,
                pixel_idx=pixel_idx,
                sparse_indices=np.array([], dtype=np.int_),
                sparse_values=np.array([], dtype=np.float64),
                tic_value=0.0,
            )

        # Zero out negative intensities to prevent data corruption
        intensities = np.maximum(intensities, 0.0)

        # Calculate TIC after zeroing negatives
        tic_value = float(np.sum(intensities))

        # Skip empty spectra after negative value removal
        if tic_value == 0.0:
            return InterpolationResult(
                coords=coords,
                pixel_idx=pixel_idx,
                sparse_indices=np.array([], dtype=np.int_),
                sparse_values=np.array([], dtype=np.float64),
                tic_value=0.0,
            )

        # Perform interpolation
        if self.interpolation_method == "linear":
            interpolated = np.interp(
                self.common_mass_axis,
                mzs,
                intensities,
                left=self.fill_value,
                right=self.fill_value,
            )
        elif self.interpolation_method == "nearest":
            interpolated = self._nearest_interpolation(mzs, intensities)
        elif self.interpolation_method == "cubic":
            interpolated = self._cubic_interpolation(mzs, intensities)
        else:
            raise ValueError(
                f"Unknown interpolation method: {self.interpolation_method}"
            )

        # Ensure no negative values in interpolated result
        interpolated = np.maximum(interpolated, 0.0)

        # Convert to sparse representation
        sparse_indices, sparse_values = self._to_sparse(interpolated)

        return InterpolationResult(
            coords=coords,
            pixel_idx=pixel_idx,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            tic_value=tic_value,
        )

    def interpolate_chunk(
        self, pixel_chunk: List[Dict[str, Any]]
    ) -> List[InterpolationResult]:
        """
        Interpolate a chunk of pixels to the common mass axis.

        Args:
            pixel_chunk: List of pixel data dictionaries containing 'coords', 'pixel_idx', 'mzs', 'intensities'

        Returns:
            List of InterpolationResult objects
        """
        results = []

        for pixel_data in pixel_chunk:
            try:
                result = self.interpolate_spectrum(
                    pixel_data["mzs"],
                    pixel_data["intensities"],
                    pixel_data["coords"],
                    pixel_data["pixel_idx"],
                )
                results.append(result)
            except Exception as e:
                logging.warning(
                    f"Error interpolating pixel {pixel_data.get('coords', 'unknown')}: {e}"
                )
                continue

        return results

    def create_dask_interpolation_task(self, pixel_chunk_delayed):
        """
        Create a Dask delayed task for interpolating a chunk of pixels.

        Args:
            pixel_chunk_delayed: Delayed object containing pixel data dictionaries

        Returns:
            Dask delayed task that will return List[InterpolationResult]
        """
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")

        @delayed
        def _delayed_interpolation(pixel_chunk):
            return self.interpolate_chunk(pixel_chunk)

        return _delayed_interpolation(pixel_chunk_delayed)

    def _nearest_interpolation(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Perform nearest neighbor interpolation."""
        interpolated = np.zeros_like(self.common_mass_axis, dtype=np.float64)

        for i, target_mz in enumerate(self.common_mass_axis):
            # Find nearest m/z
            idx = np.argmin(np.abs(mzs - target_mz))
            interpolated[i] = intensities[idx]

        return interpolated

    def _cubic_interpolation(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Perform cubic spline interpolation."""
        try:
            from scipy.interpolate import interp1d

            # Need at least 4 points for cubic interpolation
            if len(mzs) < 4:
                # Fall back to linear interpolation
                return np.interp(
                    self.common_mass_axis,
                    mzs,
                    intensities,
                    left=self.fill_value,
                    right=self.fill_value,
                )

            # Create cubic interpolator
            f = interp1d(
                mzs,
                intensities,
                kind="cubic",
                bounds_error=False,
                fill_value=self.fill_value,
            )

            return f(self.common_mass_axis)

        except ImportError:
            logging.warning("SciPy not available, falling back to linear interpolation")
            return np.interp(
                self.common_mass_axis,
                mzs,
                intensities,
                left=self.fill_value,
                right=self.fill_value,
            )

    def _to_sparse(
        self, interpolated: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """
        Convert interpolated spectrum to sparse representation.

        Args:
            interpolated: Dense interpolated spectrum

        Returns:
            Tuple of (indices, values) for non-zero elements
        """
        # Find non-zero elements above threshold
        nonzero_mask = interpolated > self.sparsity_threshold

        if not np.any(nonzero_mask):
            # No significant values
            return np.array([], dtype=np.int_), np.array([], dtype=np.float64)

        sparse_indices = np.where(nonzero_mask)[0].astype(np.int_)
        sparse_values = interpolated[nonzero_mask].astype(np.float64)

        return sparse_indices, sparse_values

    @property
    def mass_axis_length(self) -> int:
        """Get the length of the common mass axis."""
        return len(self.common_mass_axis)

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Get the mass range of the common axis."""
        return float(self.common_mass_axis[0]), float(self.common_mass_axis[-1])

    def _sparse_linear_interpolation(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """
        Perform sparse linear interpolation without creating dense arrays.

        This method provides memory-efficient linear interpolation by directly
        computing sparse output without temporary dense arrays.

        Args:
            mzs: Original m/z values
            intensities: Original intensity values

        Returns:
            Tuple of (sparse_indices, sparse_values) for interpolated spectrum
        """
        if len(mzs) == 0:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float64)

        sparse_indices = []
        sparse_values = []

        # For each original peak, perform true linear interpolation
        for mz, intensity in zip(mzs, intensities):
            if intensity <= self.sparsity_threshold:
                continue

            # Find the insertion point in common mass axis
            insert_idx = np.searchsorted(self.common_mass_axis, mz)

            # Handle edge cases
            if insert_idx == 0:
                # Before first bin - assign to first bin
                sparse_indices.append(0)
                sparse_values.append(intensity)
            elif insert_idx >= len(self.common_mass_axis):
                # After last bin - assign to last bin
                sparse_indices.append(len(self.common_mass_axis) - 1)
                sparse_values.append(intensity)
            else:
                # True linear interpolation between two adjacent bins
                left_idx = insert_idx - 1
                right_idx = insert_idx

                left_mz = self.common_mass_axis[left_idx]
                right_mz = self.common_mass_axis[right_idx]

                # Calculate interpolation weights
                total_distance = right_mz - left_mz
                if total_distance > 0:
                    left_weight = (right_mz - mz) / total_distance
                    right_weight = (mz - left_mz) / total_distance

                    # Add contributions to both bins (this is TRUE interpolation)
                    if left_weight > 0:
                        sparse_indices.append(left_idx)
                        sparse_values.append(intensity * left_weight)
                    if right_weight > 0:
                        sparse_indices.append(right_idx)
                        sparse_values.append(intensity * right_weight)
                else:
                    # Exact match
                    sparse_indices.append(left_idx)
                    sparse_values.append(intensity)

        if not sparse_indices:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float64)

        # Combine contributions to same bins
        sparse_indices = np.array(sparse_indices, dtype=np.int_)
        sparse_values = np.array(sparse_values, dtype=np.float64)

        # Group by index and sum values (handles multiple peaks contributing to same bin)
        unique_indices, inverse_indices = np.unique(sparse_indices, return_inverse=True)
        summed_values = np.zeros(len(unique_indices), dtype=np.float64)
        np.add.at(summed_values, inverse_indices, sparse_values)

        # Filter out values below threshold and ensure no negative values
        summed_values = np.maximum(summed_values, 0.0)
        keep_mask = summed_values > self.sparsity_threshold
        final_indices = unique_indices[keep_mask]
        final_values = summed_values[keep_mask]

        return final_indices, final_values

    def _sparse_nearest_interpolation(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Sparse nearest neighbor interpolation."""
        if len(mzs) == 0:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float64)

        # Find nearest bins
        bin_indices = np.searchsorted(self.common_mass_axis, mzs)
        bin_indices = np.clip(bin_indices, 0, len(self.common_mass_axis) - 1)

        # Choose closer bin for values between two bins
        for i, (mz, bin_idx) in enumerate(zip(mzs, bin_indices)):
            if bin_idx > 0 and bin_idx < len(self.common_mass_axis):
                left_distance = abs(mz - self.common_mass_axis[bin_idx - 1])
                right_distance = abs(mz - self.common_mass_axis[bin_idx])
                if left_distance < right_distance:
                    bin_indices[i] = bin_idx - 1

        # Filter by intensity threshold
        keep_mask = intensities > self.sparsity_threshold
        final_indices = bin_indices[keep_mask]
        final_intensities = intensities[keep_mask]

        # Combine intensities for peaks assigned to same bin
        unique_indices, inverse_indices = np.unique(final_indices, return_inverse=True)
        summed_intensities = np.zeros(len(unique_indices), dtype=np.float64)
        np.add.at(summed_intensities, inverse_indices, final_intensities)

        return unique_indices.astype(np.int_), summed_intensities

    def _sparse_cubic_interpolation(
        self, mzs: NDArray[np.float64], intensities: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """
        Sparse cubic interpolation - falls back to linear for memory efficiency.

        Cubic interpolation would require more complex sparse logic and may not
        provide significant benefits for typical MSI data. Falls back to linear.
        """
        logging.warning(
            "Cubic interpolation falls back to linear for memory efficiency"
        )
        return self._sparse_linear_interpolation(mzs, intensities)


class DaskInterpolationProcessor:
    """
    Handles Dask-based processing of spectral interpolation tasks.

    This class manages the creation and execution of Dask task graphs
    for large-scale spectral interpolation workflows.
    """

    def __init__(self, interpolator: SpectralInterpolator, memory_limit: str = "2GB"):
        """
        Initialize the Dask interpolation processor.

        Args:
            interpolator: SpectralInterpolator instance
            memory_limit: Dask memory limit per worker
        """
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available")

        self.interpolator = interpolator
        self.memory_limit = memory_limit

    def process_chunks(
        self, pixel_chunks: List[List[Dict[str, Any]]]
    ) -> List[InterpolationResult]:
        """
        Process multiple chunks of pixels using Dask.

        Args:
            pixel_chunks: List of pixel chunk lists

        Returns:
            Combined list of InterpolationResult objects
        """
        import dask

        # Create delayed tasks for each chunk
        delayed_tasks = [
            self.interpolator.create_dask_interpolation_task(chunk)
            for chunk in pixel_chunks
        ]

        # Combine results
        @delayed
        def combine_results(results_list):
            combined = []
            for chunk_results in results_list:
                combined.extend(chunk_results)
            return combined

        combined_task = combine_results(delayed_tasks)

        # Execute with memory management
        with dask.config.set({"array.chunk-size": self.memory_limit}):
            return combined_task.compute()


def create_interpolator_from_mass_axis(
    common_mass_axis: NDArray[np.float64], **kwargs
) -> SpectralInterpolator:
    """
    Factory function to create a SpectralInterpolator with validation.

    Args:
        common_mass_axis: Target mass axis for interpolation
        **kwargs: Additional arguments for SpectralInterpolator

    Returns:
        Configured SpectralInterpolator instance
    """
    if common_mass_axis is None or len(common_mass_axis) == 0:
        raise ValueError("Valid common_mass_axis is required")

    return SpectralInterpolator(common_mass_axis, **kwargs)
