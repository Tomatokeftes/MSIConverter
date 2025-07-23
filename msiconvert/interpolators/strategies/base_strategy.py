from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    """Quality metrics for interpolation validation"""
    tic_ratio: float  # Total Ion Current preservation ratio
    peak_preservation: float  # Fraction of peaks preserved
    n_peaks_original: int  # Number of peaks in original spectrum
    n_peaks_interpolated: int  # Number of peaks in interpolated spectrum
    interpolation_time: float  # Time taken for interpolation (seconds)
    
class InterpolationStrategy(ABC):
    """Abstract base class for interpolation strategies"""
    
    def __init__(self, validate_quality: bool = True):
        self.validate_quality = validate_quality
        
    @abstractmethod
    def interpolate_spectrum(self, 
                           mz_old: NDArray[np.float64],
                           intensity_old: NDArray[np.float32],
                           mz_new: NDArray[np.float64]) -> NDArray[np.float32]:
        """
        Interpolate spectrum to new mass axis.
        
        Args:
            mz_old: Original m/z values
            intensity_old: Original intensity values  
            mz_new: Target m/z values for interpolation
            
        Returns:
            Interpolated intensity values on new mass axis
        """
        pass
        
    def validate_interpolation(self, 
                             mz_old: NDArray[np.float64],
                             intensity_old: NDArray[np.float32],
                             mz_new: NDArray[np.float64],
                             intensity_new: NDArray[np.float32],
                             interpolation_time: float) -> QualityMetrics:
        """
        Validate interpolation quality.
        
        Args:
            mz_old: Original m/z values
            intensity_old: Original intensity values
            mz_new: New m/z values
            intensity_new: Interpolated intensity values
            interpolation_time: Time taken for interpolation
            
        Returns:
            Quality metrics for the interpolation
        """
        # Total Ion Current preservation
        tic_old = np.sum(intensity_old) if len(intensity_old) > 0 else 0.0
        tic_new = np.sum(intensity_new) if len(intensity_new) > 0 else 0.0
        
        tic_ratio = tic_new / tic_old if tic_old > 0 else 1.0
        
        # Peak preservation (simple peak detection)
        peaks_old = self._find_peaks(intensity_old)
        peaks_new = self._find_peaks(intensity_new)
        
        peak_preservation = len(peaks_new) / len(peaks_old) if len(peaks_old) > 0 else 1.0
        
        return QualityMetrics(
            tic_ratio=tic_ratio,
            peak_preservation=peak_preservation,
            n_peaks_original=len(peaks_old),
            n_peaks_interpolated=len(peaks_new),
            interpolation_time=interpolation_time
        )
        
    def _find_peaks(self, intensities: NDArray[np.float32], 
                   min_height_ratio: float = 0.01) -> NDArray[np.int32]:
        """
        Simple peak detection for quality assessment.
        
        Args:
            intensities: Intensity array
            min_height_ratio: Minimum peak height as fraction of maximum
            
        Returns:
            Array of peak indices
        """
        if len(intensities) <= 2:
            return np.array([], dtype=np.int32)
            
        max_intensity = np.max(intensities)
        if max_intensity == 0:
            return np.array([], dtype=np.int32)
            
        min_height = max_intensity * min_height_ratio
        peaks = []
        
        # Find local maxima above threshold
        for i in range(1, len(intensities) - 1):
            if (intensities[i] > intensities[i-1] and 
                intensities[i] > intensities[i+1] and
                intensities[i] > min_height):
                peaks.append(i)
                
        return np.array(peaks, dtype=np.int32)
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for identification"""
        pass
        
    @property
    @abstractmethod  
    def description(self) -> str:
        """Strategy description"""
        pass