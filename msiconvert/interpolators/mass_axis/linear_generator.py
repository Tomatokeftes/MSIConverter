"""Linear mass axis generator for uniform spectral interpolation."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ...metadata.types import EssentialMetadata
from .base_generator import BaseMassAxisGenerator


class LinearMassAxisGenerator(BaseMassAxisGenerator):
    """Generate linearly spaced mass axis for spectral interpolation.

    Creates a uniform mass axis with equal spacing between consecutive
    mass values, suitable for most spectral interpolation applications.
    """

    def generate(
        self,
        metadata: EssentialMetadata,
        bins: int,
        mass_range: Optional[tuple[float, float]] = None,
    ) -> NDArray[np.float64]:
        """Generate a linearly spaced mass axis.

        Args:
            metadata: Essential metadata containing mass range information
            bins: Number of points in the generated mass axis
            mass_range: Optional custom mass range (min, max). If None, uses metadata.

        Returns:
            Array of linearly spaced mass values

        Raises:
            ValueError: If bins <= 0 or mass_range is invalid
        """
        # Extract and validate mass range
        extracted_range = self._extract_mass_range(metadata, mass_range)
        self._validate_parameters(bins, extracted_range)

        min_mass, max_mass = extracted_range

        # Generate linearly spaced mass axis
        return np.linspace(min_mass, max_mass, bins, dtype=np.float64)
