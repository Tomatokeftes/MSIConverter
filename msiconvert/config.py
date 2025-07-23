"""
Configuration constants for MSIConverter.

This module centralizes all configuration values, magic numbers, and default
settings used throughout the codebase to improve maintainability and allow
easier tuning for different use cases.
"""

# Memory and performance settings
DEFAULT_BUFFER_SIZE = 100000  # Default buffer size for processing spectra
DEFAULT_BATCH_SIZE_IMZML = 50  # Default batch size for ImzML processing
DEFAULT_BATCH_SIZE_BRUKER = 100  # Default batch size for Bruker processing

# Memory constraints
MB_TO_BYTES = 1024 * 1024
BYTES_PER_FLOAT64 = 8
BYTES_PER_FLOAT32 = 4
BYTES_PER_UINT32 = 4
DEFAULT_MEMORY_LIMIT_MB = 1024  # Default memory limit in MB
LOG_FILE_MAX_SIZE_MB = 10  # Max log file size before rotation
LOG_BACKUP_COUNT = 5

# Batch processing limits
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE_BRUKER = 1000
ADAPTIVE_BATCH_ADJUSTMENT = 10  # Amount to adjust batch size

# Performance thresholds
FAST_PROCESSING_THRESHOLD_SEC = 0.5  # Under this is considered fast
SLOW_PROCESSING_THRESHOLD_SEC = 2.0  # Above this is considered slow

# Coordinate cache settings
COORDINATE_CACHE_BATCH_SIZE = 100
BUFFER_POOL_SIZE = 20

# SDK buffer settings
INITIAL_BUFFER_SIZE = 1024  # Initial buffer size for SDK operations

# Mass axis building
MASS_AXIS_BATCH_SIZE = 50  # Batch size for mass axis construction

# Memory pool settings
MAX_BUFFERS_PER_SIZE = 10  # Maximum number of buffers to keep per size

# Pixel size detection
PIXEL_SIZE_TOLERANCE = 0.01  # Tolerance for pixel size comparison (micrometers)

# File size estimation (rough)
ESTIMATED_BYTES_PER_SPECTRUM_POINT = 4  # For float32 values

# Progress reporting
DEFAULT_PROGRESS_UPDATE_INTERVAL = 1.0  # Seconds between progress updates

# Interpolation settings
DEFAULT_INTERPOLATION_BINS = 90000  # Default number of bins for interpolation
DEFAULT_INTERPOLATION_METHOD = "pchip"  # Default interpolation method
DEFAULT_INTERPOLATION_WORKERS = 4  # Default number of interpolation workers
MAX_INTERPOLATION_WORKERS = 80  # Maximum workers for interpolation (as per review)
MIN_INTERPOLATION_WORKERS = 4  # Minimum workers for interpolation
DEFAULT_INTERPOLATION_MEMORY_GB = 8.0  # Default memory limit for interpolation
DEFAULT_INTERPOLATION_WIDTH_MZ = 400.0  # Default reference m/z for width specification

# Configuration classes
from dataclasses import dataclass
from typing import Optional

@dataclass
class InterpolationConfig:
    """Configuration for interpolation functionality"""
    enabled: bool = False
    method: str = DEFAULT_INTERPOLATION_METHOD
    
    # SCiLS-like bin specification (either bins OR width, not both)
    interpolation_bins: Optional[int] = None
    interpolation_width: Optional[float] = None
    interpolation_width_mz: float = DEFAULT_INTERPOLATION_WIDTH_MZ
    
    # Performance settings
    max_workers: int = MAX_INTERPOLATION_WORKERS
    min_workers: int = MIN_INTERPOLATION_WORKERS
    max_memory_gb: float = DEFAULT_INTERPOLATION_MEMORY_GB
    adaptive_workers: bool = True
    buffer_size: int = 1000
    
    # Quality settings
    validate_quality: bool = True
    tic_deviation_threshold: float = 0.01  # 1% TIC deviation allowed
    peak_preservation_threshold: float = 0.95  # 95% peak preservation required
    
    # Physics model settings
    use_physics_model: bool = True
    physics_model: str = "auto"  # auto, tof, orbitrap, fticr
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.interpolation_bins and self.interpolation_width:
            raise ValueError("Cannot specify both interpolation_bins and interpolation_width")
        
        if self.enabled and not self.interpolation_bins and not self.interpolation_width:
            # Set default if none specified
            self.interpolation_bins = DEFAULT_INTERPOLATION_BINS
            
        if not 1 <= self.min_workers <= self.max_workers:
            raise ValueError("Worker count limits must satisfy: 1 <= min_workers <= max_workers")
            
        if self.method not in ["pchip", "linear", "adaptive"]:
            raise ValueError(f"Invalid interpolation method: {self.method}")
            
    def get_summary(self) -> dict:
        """Get configuration summary for logging"""
        return {
            "enabled": self.enabled,
            "method": self.method,
            "target_bins": self.interpolation_bins,
            "target_width": self.interpolation_width,
            "max_workers": self.max_workers,
            "physics_model": self.physics_model,
            "validate_quality": self.validate_quality
        }
