import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any
import logging
from .base_strategy import InterpolationStrategy, QualityMetrics
from .pchip_strategy import PchipInterpolationStrategy

class AdaptiveInterpolationStrategy(InterpolationStrategy):
    """
    Adaptive interpolation strategy that selects the best method based on spectrum characteristics.
    
    This strategy analyzes the input spectrum and chooses between different interpolation
    methods based on factors like:
    - Number of data points
    - Data density
    - Presence of noise
    - Performance requirements
    """
    
    def __init__(self, 
                 validate_quality: bool = True,
                 performance_priority: str = "balanced"):
        """
        Initialize adaptive interpolation strategy.
        
        Args:
            validate_quality: Whether to validate interpolation quality
            performance_priority: "speed", "quality", or "balanced"
        """
        super().__init__(validate_quality)
        self.performance_priority = performance_priority
        
        # Initialize available strategies
        self.pchip_strategy = PchipInterpolationStrategy(validate_quality=False)
        
        # Strategy selection thresholds
        self.sparse_data_threshold = 10  # Use linear for very sparse data
        self.dense_data_threshold = 1000  # Use PCHIP for dense data
        
    def interpolate_spectrum(self, 
                           mz_old: NDArray[np.float64],
                           intensity_old: NDArray[np.float64],
                           mz_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Adaptively interpolate spectrum using best method for the data.
        
        Args:
            mz_old: Original m/z values
            intensity_old: Original intensity values
            mz_new: Target m/z values
            
        Returns:
            Interpolated intensity values
        """
        # Analyze spectrum characteristics
        characteristics = self._analyze_spectrum(mz_old, intensity_old, mz_new)
        
        # Select strategy based on characteristics
        strategy = self._select_strategy(characteristics)
        
        # Perform interpolation
        if strategy == "linear":
            return self._linear_interpolation(mz_old, intensity_old, mz_new)
        elif strategy == "pchip":
            return self.pchip_strategy.interpolate_spectrum(mz_old, intensity_old, mz_new)
        else:
            # Default to PCHIP
            return self.pchip_strategy.interpolate_spectrum(mz_old, intensity_old, mz_new)
    
    def _analyze_spectrum(self, 
                         mz_old: NDArray[np.float64],
                         intensity_old: NDArray[np.float64],
                         mz_new: NDArray[np.float64]) -> Dict[str, Any]:
        """
        Analyze spectrum characteristics to guide strategy selection.
        
        Args:
            mz_old: Original m/z values
            intensity_old: Original intensity values
            mz_new: Target m/z values
            
        Returns:
            Dictionary with spectrum characteristics
        """
        n_points_old = len(mz_old)
        n_points_new = len(mz_new)
        
        if n_points_old == 0:
            return {"type": "empty", "n_points_old": 0, "n_points_new": n_points_new}
        
        # Calculate data density
        mz_range = mz_old[-1] - mz_old[0] if n_points_old > 1 else 0
        data_density = n_points_old / mz_range if mz_range > 0 else 0
        
        # Calculate interpolation factor
        interpolation_factor = n_points_new / n_points_old if n_points_old > 0 else 1
        
        # Estimate noise level (coefficient of variation of intensity differences)
        noise_level = 0.0
        if n_points_old > 2:
            intensity_diffs = np.diff(intensity_old)
            mean_diff = np.mean(np.abs(intensity_diffs))
            std_diff = np.std(intensity_diffs)
            if mean_diff > 0:
                noise_level = std_diff / mean_diff
        
        # Calculate sparsity (fraction of zero/near-zero intensities)
        max_intensity = np.max(intensity_old) if n_points_old > 0 else 0
        threshold = max_intensity * 0.001  # 0.1% of max
        sparsity = np.sum(intensity_old <= threshold) / n_points_old if n_points_old > 0 else 1
        
        return {
            "type": "spectrum",
            "n_points_old": n_points_old,
            "n_points_new": n_points_new, 
            "data_density": data_density,
            "interpolation_factor": interpolation_factor,
            "noise_level": noise_level,
            "sparsity": sparsity,
            "mz_range": mz_range
        }
    
    def _select_strategy(self, characteristics: Dict[str, Any]) -> str:
        """
        Select interpolation strategy based on spectrum characteristics.
        
        Args:
            characteristics: Spectrum analysis results
            
        Returns:
            Strategy name to use
        """
        if characteristics["type"] == "empty":
            return "linear"  # Doesn't matter for empty spectra
        
        n_points_old = characteristics["n_points_old"]
        interpolation_factor = characteristics["interpolation_factor"]
        sparsity = characteristics["sparsity"]
        
        # Very sparse data - use linear interpolation
        if n_points_old < self.sparse_data_threshold or sparsity > 0.9:
            return "linear"
        
        # Performance priority considerations
        if self.performance_priority == "speed":
            # Prefer linear for speed, unless data is very dense
            if n_points_old < self.dense_data_threshold:
                return "linear"
            else:
                return "pchip"
                
        elif self.performance_priority == "quality":
            # Always prefer PCHIP for quality unless data is too sparse
            if n_points_old >= self.sparse_data_threshold:
                return "pchip"
            else:
                return "linear"
                
        else:  # balanced
            # Use PCHIP for moderate to dense data
            if n_points_old >= 50:  # Reasonable threshold for PCHIP
                return "pchip"
            else:
                return "linear"
    
    def _linear_interpolation(self, 
                            mz_old: NDArray[np.float64],
                            intensity_old: NDArray[np.float64],
                            mz_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Linear interpolation implementation.
        
        Args:
            mz_old: Original m/z values
            intensity_old: Original intensity values
            mz_new: Target m/z values
            
        Returns:
            Linearly interpolated intensity values
        """
        if len(mz_old) == 0:
            return np.zeros(len(mz_new), dtype=np.float64)
            
        intensity_new = np.interp(
            mz_new, 
            mz_old, 
            intensity_old, 
            left=0.0, 
            right=0.0
        )
        return intensity_new.astype(np.float64)
    
    def get_strategy_selection_info(self, 
                                  mz_old: NDArray[np.float64],
                                  intensity_old: NDArray[np.float64],
                                  mz_new: NDArray[np.float64]) -> Dict[str, Any]:
        """
        Get information about strategy selection for this spectrum.
        
        Args:
            mz_old: Original m/z values
            intensity_old: Original intensity values
            mz_new: Target m/z values
            
        Returns:
            Dictionary with strategy selection information
        """
        characteristics = self._analyze_spectrum(mz_old, intensity_old, mz_new)
        selected_strategy = self._select_strategy(characteristics)
        
        return {
            "selected_strategy": selected_strategy,
            "characteristics": characteristics,
            "reasoning": self._get_selection_reasoning(characteristics, selected_strategy)
        }
    
    def _get_selection_reasoning(self, 
                               characteristics: Dict[str, Any], 
                               selected_strategy: str) -> str:
        """
        Get human-readable reasoning for strategy selection.
        
        Args:
            characteristics: Spectrum characteristics
            selected_strategy: Selected strategy name
            
        Returns:
            Reasoning string
        """
        if selected_strategy == "linear":
            if characteristics["n_points_old"] < self.sparse_data_threshold:
                return f"Linear: Too few points ({characteristics['n_points_old']}) for PCHIP"
            elif characteristics["sparsity"] > 0.9:
                return f"Linear: High sparsity ({characteristics['sparsity']:.2f})"
            else:
                return "Linear: Performance optimization"
        else:
            return f"PCHIP: Sufficient data density ({characteristics['n_points_old']} points)"
    
    @property
    def name(self) -> str:
        """Strategy name"""
        return "adaptive"
    
    @property
    def description(self) -> str:
        """Strategy description"""
        return ("Adaptive interpolation that selects optimal method based on spectrum "
                "characteristics including data density, sparsity, and performance requirements.")
    
    def get_performance_info(self) -> Dict[str, Any]:
        """
        Get performance characteristics of this strategy.
        
        Returns:
            Dictionary with performance information
        """
        return {
            "name": self.name,
            "description": self.description,
            "adapts_to_data": True,
            "strategies_used": ["linear", "pchip"],
            "performance_priority": self.performance_priority,
            "recommended_for": "Mixed datasets with varying spectrum characteristics"
        }