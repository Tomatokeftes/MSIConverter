"""
MSI Interpolation Module

This module provides intelligent interpolation capabilities for Mass Spectrometry Imaging data,
enabling physics-based mass axis resampling to reduce file sizes by 50-90% while preserving
data quality.
"""

from typing import TypedDict, NamedTuple, Protocol
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# Core data types used throughout the interpolation system
class SpectrumData(NamedTuple):
    """Standard spectrum representation"""
    coords: Tuple[int, int, int]  # (x, y, z) coordinates
    mz_values: NDArray[np.float64]  # m/z values
    intensities: NDArray[np.float64]  # intensity values

# Bounds information from metadata
@dataclass
class BoundsInfo:
    """Dataset bounds from metadata extraction"""
    min_mz: float
    max_mz: float
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    n_spectra: int
    
# Buffer for zero-copy operations
@dataclass
class SpectrumBuffer:
    """Reusable buffer for spectrum data"""
    buffer_id: int
    mz_buffer: NDArray[np.float64]  # Pre-allocated
    intensity_buffer: NDArray[np.float64]  # Pre-allocated
    actual_size: int = 0  # Actual data size
    coords: Tuple[int, int, int] = (0, 0, 0)  # Spectrum coordinates
    
    def fill(self, mz: NDArray, intensity: NDArray) -> None:
        """Fill buffer with data"""
        self.actual_size = len(mz)
        # Handle case where input data is larger than buffer
        if self.actual_size > len(self.mz_buffer):
            # Resize buffers to accommodate larger data
            self.mz_buffer = np.empty(self.actual_size, dtype=np.float64)
            self.intensity_buffer = np.empty(self.actual_size, dtype=np.float64)
        self.mz_buffer[:self.actual_size] = mz
        self.intensity_buffer[:self.actual_size] = intensity
        
    def get_data(self) -> Tuple[NDArray, NDArray]:
        """Get actual data from buffer"""
        return (self.mz_buffer[:self.actual_size], 
                self.intensity_buffer[:self.actual_size])

__all__ = [
    'SpectrumData',
    'BoundsInfo', 
    'SpectrumBuffer'
]