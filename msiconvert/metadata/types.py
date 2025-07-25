# msiconvert/metadata/core/metadata_types.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EssentialMetadata:
    """Critical metadata for processing decisions and interpolation setup."""

    dimensions: Tuple[int, int, int]  # (x, y, z) grid dimensions
    coordinate_bounds: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    mass_range: Tuple[float, float]  # (min_mass, max_mass)
    pixel_size: Optional[Tuple[float, float]]  # (x_size, y_size) in micrometers
    n_spectra: int  # Total number of spectra
    estimated_memory_gb: float  # Memory usage estimate
    source_path: str  # Path to source data
    coordinate_offsets: Optional[Tuple[int, int, int]] = None  # (x_offset, y_offset, z_offset) for raw coordinate normalization

    @property
    def spatial_extent(self) -> Tuple[float, float]:
        """Calculate spatial extent for interpolation grid setup."""
        return (
            self.coordinate_bounds[1] - self.coordinate_bounds[0],  # x extent
            self.coordinate_bounds[3] - self.coordinate_bounds[2],  # y extent
        )

    @property
    def has_pixel_size(self) -> bool:
        """Check if pixel size information is available."""
        return self.pixel_size is not None

    @property
    def is_3d(self) -> bool:
        """Check if dataset is 3D (z > 1)."""
        return self.dimensions[2] > 1


@dataclass
class ComprehensiveMetadata:
    """Complete metadata including format-specific details."""

    essential: EssentialMetadata
    format_specific: Dict[str, Any]  # Format-specific metadata (ImzML/Bruker)
    acquisition_params: Dict[str, Any]  # Acquisition parameters
    instrument_info: Dict[str, Any]  # Instrument information
    raw_metadata: Dict[str, Any]  # Original format metadata (unprocessed)

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Convenience access to dimensions from essential metadata."""
        return self.essential.dimensions

    @property
    def pixel_size(self) -> Optional[Tuple[float, float]]:
        """Convenience access to pixel size from essential metadata."""
        return self.essential.pixel_size

    @property
    def coordinate_bounds(self) -> Tuple[float, float, float, float]:
        """Convenience access to coordinate bounds from essential metadata."""
        return self.essential.coordinate_bounds
