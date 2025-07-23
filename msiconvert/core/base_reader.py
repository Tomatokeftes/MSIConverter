# msiconvert/core/base_reader.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class BaseMSIReader(ABC):
    """Abstract base class for reading MSI data formats."""

    def __init__(self, data_path: Path, **kwargs):
        """
        Initialize the reader with the path to the data.

        Args:
            data_path: Path to the data file or directory
            **kwargs: Additional reader-specific parameters
        """
        self.data_path = Path(data_path)

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the MSI dataset."""
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Return the dimensions of the MSI dataset (x, y, z)."""
        pass

    @abstractmethod
    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """
        Return the common mass axis for all spectra.

        This method must always return a valid array.
        If no common mass axis can be created, implementations should raise an exception.
        """
        pass

    @abstractmethod
    def iter_spectra(
        self, batch_size: Optional[int] = None
    ) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """
        Iterate through spectra with optional batch processing.

        Args:
            batch_size: Optional batch size for spectrum iteration

        Yields:
            Tuple containing:

                - Coordinates (x, y, z) using 0-based indexing
                - m/z values array

                - Intensity values array
        """
        pass

    @staticmethod
    def map_mz_to_common_axis(
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
        common_axis: NDArray[np.float64],
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """
        Map m/z values to indices in the common mass axis with high accuracy.

        This method ensures exact mapping of m/z values to the common mass axis
        without interpolation, preserving the original intensity values.

        Args:
            mzs: NDArray[np.float64] - Array of m/z values
            intensities: NDArray[np.float64] - Array of intensity values
            common_axis: NDArray[np.float64] - Common mass axis (sorted array of unique m/z values)

        Returns:
            Tuple of (indices in common mass axis, corresponding intensities)
        """
        if mzs.size == 0 or intensities.size == 0:
            return np.array([], dtype=int), np.array([])

        # Use searchsorted to find indices in common mass axis
        indices = np.searchsorted(common_axis, mzs)

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, len(common_axis) - 1)

        # Verify that we're actually finding the right m/z values
        max_diff = (
            1e-6  # A very small tolerance threshold for floating point differences
        )
        indices_valid = np.abs(common_axis[indices] - mzs) <= max_diff

        # Return only the valid indices and their corresponding intensities
        return indices[indices_valid], intensities[indices_valid]

    @abstractmethod
    def close(self) -> None:
        """Close all open file handles."""
        pass

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return the shape of the dataset (x, y, z dimensions).

        Returns:
            Tuple of (x, y, z) dimensions
        """
        return self.get_dimensions()

    @property
    def n_spectra(self) -> int:
        """
        Return the total number of spectra in the dataset.

        Returns:
            Total number of spectra
        """
        # Default implementation counts actual spectra
        count = 0
        for _ in self.iter_spectra():
            count += 1
        return count

    @property
    def mass_range(self) -> Tuple[float, float]:
        """
        Return the mass range (min_mass, max_mass) of the dataset.

        Returns:
            Tuple of (min_mass, max_mass) values
        """
        mass_axis = self.get_common_mass_axis()
        if len(mass_axis) == 0:
            return (0.0, 0.0)
        return (float(np.min(mass_axis)), float(np.max(mass_axis)))

    def get_pixel_size(self) -> Optional[Tuple[float, float]]:
        """
        Extract pixel size from format-specific metadata.

        Returns:
            Optional[Tuple[float, float]]: Pixel size as (x_size, y_size) in micrometers,
                                         or None if not available in metadata.

        Notes:
            - For ImzML: Extracts from cvParam IMS:1000046 and IMS:1000047
            - For Bruker: Extracts from MaldiFrameLaserInfo table (BeamScanSizeX/Y)
            - Default implementation returns None (no automatic detection)
        """
        return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    # --- Interpolation Support Methods ---

    def get_mass_bounds(self) -> Tuple[float, float]:
        """
        Get mass bounds WITHOUT scanning all spectra (for interpolation efficiency).
        
        Returns:
            Tuple of (min_mz, max_mz)
            
        Note:
            Base implementation uses mass_range property as fallback.
            Subclasses should override this for efficient metadata-based bounds detection.
        """
        return self.mass_range

    def get_spatial_bounds(self) -> Dict[str, int]:
        """
        Get spatial bounds WITHOUT scanning all spectra (for interpolation efficiency).
        
        Returns:
            Dictionary with keys: min_x, max_x, min_y, max_y
            
        Note:
            Base implementation derives from dimensions.
            Subclasses should override this for efficient metadata-based bounds detection.
        """
        dimensions = self.get_dimensions()
        return {
            'min_x': 0,
            'max_x': dimensions[0] - 1,
            'min_y': 0,
            'max_y': dimensions[1] - 1
        }

    def get_estimated_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory requirements for interpolation processing.
        
        Returns:
            Dictionary with memory usage estimates in bytes
        """
        dimensions = self.get_dimensions()
        mass_axis_length = len(self.get_common_mass_axis())
        
        # Estimate based on typical spectrum characteristics
        n_spectra = dimensions[0] * dimensions[1] * dimensions[2]
        avg_spectrum_points = mass_axis_length // 10  # Assume sparse data
        
        return {
            'total_spectra': n_spectra,
            'avg_spectrum_points': avg_spectrum_points,
            'estimated_raw_memory_mb': (n_spectra * avg_spectrum_points * 12) / 1e6,  # mz + intensity
            'estimated_interpolated_memory_mb': (n_spectra * mass_axis_length * 4) / 1e6  # float32 intensities
        }

    def iter_spectra_buffered(self, buffer_pool: 'SpectrumBufferPool') -> Generator['SpectrumBuffer', None, None]:
        """
        Iterate through spectra using pre-allocated buffers for zero-copy operations.
        
        Args:
            buffer_pool: Pool of pre-allocated spectrum buffers
            
        Yields:
            SpectrumBuffer: Buffer containing spectrum data
            
        Note:
            Base implementation converts regular iteration to buffered format.
            Subclasses should override this for native buffered iteration.
        """
        for coords, mzs, intensities in self.iter_spectra():
            # Get buffer from pool
            buffer = buffer_pool.get_buffer()
            
            # Fill buffer with spectrum data
            buffer.coords = coords
            buffer.fill(mzs, intensities.astype(np.float64))
            
            yield buffer
