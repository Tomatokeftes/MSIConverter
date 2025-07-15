# msiconvert/binning_module/domain/strategies.py
"""Abstract and concrete binning strategies for different instrument types."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BinningStrategy(ABC):
    """Abstract base class defining the interface for binning algorithms."""
    
    @abstractmethod
    def calculate_num_bins(
        self,
        target_width_da: float,
        reference_mz: float,
        min_mz: float,
        max_mz: float
    ) -> int:
        """
        Calculate number of bins needed to achieve target width at reference m/z.
        
        Parameters
        ----------
        target_width_da : float
            Target bin width in Daltons at reference m/z
        reference_mz : float
            Reference m/z value for bin width calculation
        min_mz : float
            Minimum m/z value in range
        max_mz : float
            Maximum m/z value in range
            
        Returns
        -------
        int
            Number of bins required
        """
        pass
    
    @abstractmethod
    def calculate_target_width(
        self,
        num_bins: int,
        reference_mz: float,
        min_mz: float,
        max_mz: float
    ) -> float:
        """
        Calculate resulting bin width in Daltons at reference m/z for given number of bins.
        
        Parameters
        ----------
        num_bins : int
            Number of bins
        reference_mz : float
            Reference m/z value for bin width calculation
        min_mz : float
            Minimum m/z value in range
        max_mz : float
            Maximum m/z value in range
            
        Returns
        -------
        float
            Bin width in Daltons at reference m/z
        """
        pass
    
    @abstractmethod
    def generate_bin_edges(
        self,
        min_mz: float,
        max_mz: float,
        num_bins: int
    ) -> NDArray[np.float64]:
        """
        Generate the actual non-linear bin edges array.
        
        Parameters
        ----------
        min_mz : float
            Minimum m/z value
        max_mz : float
            Maximum m/z value
        num_bins : int
            Number of bins
            
        Returns
        -------
        NDArray[np.float64]
            Array of bin edge values in m/z (length = num_bins + 1)
        """
        pass


class LinearTOFStrategy(BinningStrategy):
    """
    Implement binning for Linear TOF instruments using square root transformation.
    
    Linear TOF instruments have flight time proportional to sqrt(m/z),
    so uniform bins in flight time space create appropriate non-linear bins in m/z space.
    """
    
    def calculate_num_bins(
        self,
        target_width_da: float,
        reference_mz: float,
        min_mz: float,
        max_mz: float
    ) -> int:
        """Calculate number of bins for Linear TOF."""
        # Transform to flight time space
        min_ft = np.sqrt(min_mz)
        max_ft = np.sqrt(max_mz)
        ref_ft = np.sqrt(reference_mz)
        
        # Calculate bin width in flight time space at reference
        # Derivative: d(sqrt(mz))/d(mz) = 1/(2*sqrt(mz))
        ft_bin_width = target_width_da / (2 * ref_ft)
        
        # Calculate number of bins
        num_bins = int(np.ceil((max_ft - min_ft) / ft_bin_width))
        
        return max(1, num_bins)
    
    def calculate_target_width(
        self,
        num_bins: int,
        reference_mz: float,
        min_mz: float,
        max_mz: float
    ) -> float:
        """Calculate bin width for Linear TOF."""
        # Transform to flight time space
        min_ft = np.sqrt(min_mz)
        max_ft = np.sqrt(max_mz)
        ref_ft = np.sqrt(reference_mz)
        
        # Calculate bin width in flight time space
        ft_bin_width = (max_ft - min_ft) / num_bins
        
        # Convert back to m/z space at reference
        # Inverse of derivative calculation
        mz_bin_width = ft_bin_width * 2 * ref_ft
        
        return mz_bin_width
    
    def generate_bin_edges(
        self,
        min_mz: float,
        max_mz: float,
        num_bins: int
    ) -> NDArray[np.float64]:
        """Generate bin edges for Linear TOF."""
        # Transform to flight time space
        min_ft = np.sqrt(min_mz)
        max_ft = np.sqrt(max_mz)
        
        # Create uniform bins in flight time space
        ft_edges = np.linspace(min_ft, max_ft, num_bins + 1)
        
        # Transform back to m/z space
        mz_edges = ft_edges ** 2
        
        # Ensure exact boundaries
        mz_edges[0] = min_mz
        mz_edges[-1] = max_mz
        
        return mz_edges


class ReflectorTOFStrategy(BinningStrategy):
    """
    Implement binning for Reflector TOF instruments using logarithmic transformation.
    
    Reflector TOF instruments have approximately logarithmic relationship between
    flight time and m/z, so uniform bins in log space create appropriate bins.
    """
    
    def calculate_num_bins(
        self,
        target_width_da: float,
        reference_mz: float,
        min_mz: float,
        max_mz: float
    ) -> int:
        """Calculate number of bins for Reflector TOF."""
        # Transform to log space
        min_log = np.log(min_mz)
        max_log = np.log(max_mz)
        
        # Calculate bin width in log space at reference
        # Derivative: d(log(mz))/d(mz) = 1/mz
        log_bin_width = target_width_da / reference_mz
        
        # Calculate number of bins
        num_bins = int(np.ceil((max_log - min_log) / log_bin_width))
        
        return max(1, num_bins)
    
    def calculate_target_width(
        self,
        num_bins: int,
        reference_mz: float,
        min_mz: float,
        max_mz: float
    ) -> float:
        """Calculate bin width for Reflector TOF."""
        # Transform to log space
        min_log = np.log(min_mz)
        max_log = np.log(max_mz)
        
        # Calculate bin width in log space
        log_bin_width = (max_log - min_log) / num_bins
        
        # Convert back to m/z space at reference
        mz_bin_width = log_bin_width * reference_mz
        
        return mz_bin_width
    
    def generate_bin_edges(
        self,
        min_mz: float,
        max_mz: float,
        num_bins: int
    ) -> NDArray[np.float64]:
        """Generate bin edges for Reflector TOF."""
        # Transform to log space
        min_log = np.log(min_mz)
        max_log = np.log(max_mz)
        
        # Create uniform bins in log space
        log_edges = np.linspace(min_log, max_log, num_bins + 1)
        
        # Transform back to m/z space
        mz_edges = np.exp(log_edges)
        
        # Ensure exact boundaries
        mz_edges[0] = min_mz
        mz_edges[-1] = max_mz
        
        return mz_edges