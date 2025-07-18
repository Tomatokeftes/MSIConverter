# msiconvert/core/base_reader.py
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Tuple, Generator, Optional

class BaseMSIReader(ABC):
    """Abstract base class for reading MSI data formats."""
    
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
    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]], None, None]:
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
    
    @abstractmethod
    def close(self) -> None:
        """Close all open file handles."""
        pass

