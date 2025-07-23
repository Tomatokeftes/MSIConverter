from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

class InstrumentPhysics(ABC):
    """Base class for instrument-specific physics models"""
    
    @abstractmethod
    def calculate_resolution_at_mz(self, mz: float) -> float:
        """Calculate mass resolution at given m/z"""
        pass
        
    @abstractmethod
    def calculate_width_at_mz(self, mz: float, width_at_reference: float, reference_mz: float) -> float:
        """Calculate bin width at given m/z based on width at reference m/z"""
        pass
        
    def create_optimal_mass_axis(self, 
                                min_mz: float, 
                                max_mz: float,
                                target_bins: Optional[int] = None,
                                width_at_mz: Optional[Tuple[float, float]] = None) -> NDArray[np.float64]:
        """
        Create physics-based optimal mass axis (like SCiLS).
        
        Args:
            min_mz: Minimum m/z value
            max_mz: Maximum m/z value  
            target_bins: Target number of bins (option 1)
            width_at_mz: Tuple of (width, reference_mz) for width at specific m/z (option 2)
            
        Must specify either target_bins OR width_at_mz, not both.
        """
        if (target_bins is None) == (width_at_mz is None):
            raise ValueError("Must specify either target_bins OR width_at_mz, not both or neither")
            
        if width_at_mz is not None:
            # Option 2: Use specified width at reference m/z (like SCiLS)
            width_at_ref, ref_mz = width_at_mz
            return self._create_axis_from_width(min_mz, max_mz, width_at_ref, ref_mz)
        else:
            # Option 1: Use target number of bins (like SCiLS)
            return self._create_axis_from_bins(min_mz, max_mz, target_bins)
            
    def _create_axis_from_width(self, min_mz: float, max_mz: float, 
                               width_at_ref: float, ref_mz: float) -> NDArray[np.float64]:
        """Create axis using specified width at reference m/z"""
        axis = [min_mz]
        current_mz = min_mz
        
        while current_mz < max_mz:
            # Calculate width at current m/z based on physics
            width = self.calculate_width_at_mz(current_mz, width_at_ref, ref_mz)
            current_mz += width
            
            if current_mz <= max_mz:
                axis.append(current_mz)
                
        return np.array(axis)
        
    def _create_axis_from_bins(self, min_mz: float, max_mz: float, 
                              target_bins: int) -> NDArray[np.float64]:
        """Create axis with approximately target number of bins"""
        # First, estimate average width needed
        avg_width_estimate = (max_mz - min_mz) / target_bins
        
        # Use this to estimate width at reference m/z (e.g., 400)
        ref_mz = 400.0
        if min_mz <= ref_mz <= max_mz:
            width_at_ref = avg_width_estimate
        else:
            # Use middle of range as reference
            ref_mz = (min_mz + max_mz) / 2
            width_at_ref = avg_width_estimate
            
        # Create initial axis
        axis = self._create_axis_from_width(min_mz, max_mz, width_at_ref, ref_mz)
        
        # Adjust if we're far from target
        actual_bins = len(axis)
        if abs(actual_bins - target_bins) > target_bins * 0.1:  # >10% off
            # Scale the reference width
            scaling_factor = actual_bins / target_bins
            adjusted_width = width_at_ref / scaling_factor
            axis = self._create_axis_from_width(min_mz, max_mz, adjusted_width, ref_mz)
            
        return axis

class TOFPhysics(InstrumentPhysics):
    """Time-of-Flight physics model"""
    
    def __init__(self, resolution_at_400: float = 10000):
        self.resolution_at_400 = resolution_at_400
        
    def calculate_resolution_at_mz(self, mz: float) -> float:
        """R  sqrt(m/z) for linear TOF"""
        return self.resolution_at_400 * np.sqrt(mz / 400.0)
        
    def calculate_width_at_mz(self, mz: float, width_at_reference: float, reference_mz: float) -> float:
        """
        Width scales with sqrt(m/z) for TOF.
        width(mz) = width_ref * sqrt(mz / mz_ref)
        """
        return width_at_reference * np.sqrt(mz / reference_mz)

class OrbitrapPhysics(InstrumentPhysics):
    """Orbitrap physics model"""
    
    def __init__(self, resolution_at_400: float = 120000):
        self.resolution_at_400 = resolution_at_400
        
    def calculate_resolution_at_mz(self, mz: float) -> float:
        """R = R_400 * sqrt(400/m)"""
        return self.resolution_at_400 * np.sqrt(400.0 / mz)
        
    def calculate_width_at_mz(self, mz: float, width_at_reference: float, reference_mz: float) -> float:
        """
        Width scales with sqrt(m/z) for Orbitrap (inverse of resolution).
        width(mz) = width_ref * sqrt(mz / mz_ref)
        """
        return width_at_reference * np.sqrt(mz / reference_mz)

class FTICRPhysics(InstrumentPhysics):
    """FTICR physics model"""
    
    def __init__(self, resolution_at_400: float = 1000000):
        self.resolution_at_400 = resolution_at_400
        
    def calculate_resolution_at_mz(self, mz: float) -> float:
        """R = R_400 * (400/m) for FTICR"""
        return self.resolution_at_400 * (400.0 / mz)
        
    def calculate_width_at_mz(self, mz: float, width_at_reference: float, reference_mz: float) -> float:
        """
        Width scales linearly with m/z for FTICR.
        width(mz) = width_ref * (mz / mz_ref)
        """
        return width_at_reference * (mz / reference_mz)

# Factory function for creating physics models
def create_physics_model(instrument_type: str, resolution_at_400: Optional[float] = None) -> InstrumentPhysics:
    """Factory function to create appropriate physics model"""
    instrument_type = instrument_type.lower()
    
    if instrument_type == "tof":
        return TOFPhysics(resolution_at_400 or 10000)
    elif instrument_type == "orbitrap":
        return OrbitrapPhysics(resolution_at_400 or 120000)
    elif instrument_type == "fticr":
        return FTICRPhysics(resolution_at_400 or 1000000)
    else:
        # Default to TOF if unknown
        return TOFPhysics(resolution_at_400 or 10000)