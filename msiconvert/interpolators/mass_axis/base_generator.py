"""Base abstract class for mass axis generators."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ...metadata.types import EssentialMetadata


class BaseMassAxisGenerator(ABC):
    """Abstract base class for mass axis generators.

    Mass axis generators create uniform mass axes for spectral interpolation
    based on dataset metadata and user-specified parameters.
    """

    @abstractmethod
    def generate(
        self,
        metadata: EssentialMetadata,
        bins: int,
        mass_range: Optional[tuple[float, float]] = None,
    ) -> NDArray[np.float64]:
        """Generate a mass axis for interpolation.

        Args:
            metadata: Essential metadata containing mass range information
            bins: Number of points in the generated mass axis
            mass_range: Optional custom mass range (min, max). If None, uses metadata.

        Returns:
            Array of mass values for interpolation

        Raises:
            ValueError: If bins <= 0 or mass_range is invalid
        """
        pass

    def _validate_parameters(
        self,
        bins: int,
        mass_range: tuple[float, float],
    ) -> None:
        """Validate input parameters.

        Args:
            bins: Number of points in mass axis
            mass_range: Mass range tuple (min, max)

        Raises:
            ValueError: If parameters are invalid
        """
        if bins <= 0:
            raise ValueError(f"bins must be positive, got {bins}")

        min_mass, max_mass = mass_range
        if min_mass >= max_mass:
            raise ValueError(
                f"Invalid mass range: min ({min_mass}) >= max ({max_mass})"
            )

        if min_mass <= 0:
            raise ValueError(f"Mass values must be positive, got min={min_mass}")

    def _extract_mass_range(
        self,
        metadata: EssentialMetadata,
        custom_range: Optional[tuple[float, float]],
    ) -> tuple[float, float]:
        """Extract mass range from metadata or custom parameters.

        Args:
            metadata: Essential metadata
            custom_range: Optional custom mass range

        Returns:
            Mass range tuple (min, max)

        Raises:
            ValueError: If no valid mass range can be determined
        """
        if custom_range is not None:
            return custom_range

        if metadata.mass_range is None:
            raise ValueError("No mass range available in metadata and none provided")

        return metadata.mass_range
