# msiconvert/binning_module/application/models.py
"""Data Transfer Objects for binning requests."""

from dataclasses import dataclass
from typing import Optional

from ..exceptions import InvalidParametersError


@dataclass
class BinningRequest:
    """
    Data Transfer Object for validated binning parameters.
    
    Attributes
    ----------
    min_mz : float
        Minimum m/z value
    max_mz : float
        Maximum m/z value
    model_type : str
        Instrument type identifier ('linear' or 'reflector')
    num_bins : Optional[int]
        Total number of bins (if specified)
    bin_size_mu : Optional[float]
        Bin size in milli-Daltons at reference m/z (if specified)
    reference_mz : float
        Reference m/z for bin size calculation (default: 1000.0)
    """
    min_mz: float
    max_mz: float
    model_type: str
    num_bins: Optional[int] = None
    bin_size_mu: Optional[float] = None
    reference_mz: float = 1000.0
    
    def __post_init__(self):
        """Validate request parameters."""
        # Validate m/z range
        if self.min_mz <= 0:
            raise InvalidParametersError("min_mz must be positive")
        
        if self.max_mz <= 0:
            raise InvalidParametersError("max_mz must be positive")
        
        if self.min_mz >= self.max_mz:
            raise InvalidParametersError("min_mz must be less than max_mz")
        
        # Validate reference m/z
        if self.reference_mz <= 0:
            raise InvalidParametersError("reference_mz must be positive")
        
        # Validate model type
        valid_models = {'linear', 'reflector'}
        if self.model_type.lower() not in valid_models:
            raise InvalidParametersError(
                f"Invalid model_type '{self.model_type}'. "
                f"Must be one of: {', '.join(valid_models)}"
            )
        
        # Normalize model type to lowercase
        self.model_type = self.model_type.lower()
        
        # Validate that exactly one of num_bins or bin_size_mu is provided
        if self.num_bins is None and self.bin_size_mu is None:
            raise InvalidParametersError(
                "Either num_bins or bin_size_mu must be provided"
            )
        
        if self.num_bins is not None and self.bin_size_mu is not None:
            raise InvalidParametersError(
                "Only one of num_bins or bin_size_mu can be provided"
            )
        
        # Validate num_bins if provided
        if self.num_bins is not None:
            if not isinstance(self.num_bins, int):
                raise InvalidParametersError("num_bins must be an integer")
            if self.num_bins <= 0:
                raise InvalidParametersError("num_bins must be positive")
        
        # Validate bin_size_mu if provided
        if self.bin_size_mu is not None:
            if self.bin_size_mu <= 0:
                raise InvalidParametersError("bin_size_mu must be positive")