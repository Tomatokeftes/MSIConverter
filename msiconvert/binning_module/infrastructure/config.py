# msiconvert/binning_module/infrastructure/config.py
"""System-wide configuration and limits for the binning module."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BinningConfig:
    """
    System-wide configuration and limits.
    
    Attributes
    ----------
    DEFAULT_REFERENCE_MZ : float
        Default reference m/z value for bin width calculations
    DEFAULT_BIN_SIZE_MU : float
        Default bin size in milli-Daltons
    MAX_ALLOWED_BINS : int
        Maximum number of bins allowed to prevent memory issues
    """
    DEFAULT_REFERENCE_MZ: float = 1000.0
    DEFAULT_BIN_SIZE_MU: float = 5.0
    MAX_ALLOWED_BINS: int = 500_000
    
    # Additional configuration can be added here
    MIN_MZ_VALUE: float = 1.0
    MAX_MZ_VALUE: float = 1e6
    
    # Precision thresholds
    BIN_WIDTH_TOLERANCE: float = 1e-6  # Relative tolerance for bin width validation