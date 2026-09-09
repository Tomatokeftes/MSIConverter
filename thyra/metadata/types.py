# thyra/metadata/types.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EssentialMetadata:
    """Critical metadata for processing decisions and interpolation setup.

    Attributes:
        dimensions: Grid dimensions as ``(x, y, z)``.
        coordinate_bounds: Spatial extent as ``(min_x, max_x, min_y, max_y)``.
        mass_range: Mass-to-charge range as ``(min_mz, max_mz)``.
        pixel_size: In-plane pixel dimensions as ``(x_um, y_um)`` in
            micrometres, or ``None`` when not detected.  This is the
            raster pitch and says nothing about z; see ``z_spacing_um``.
        n_spectra: Total number of spectra in the dataset -- the ones
            actually present, not the positions the raster covers.
            Meaningless unless ``n_spectra_counted`` is True.
        n_spectra_counted: Whether ``n_spectra`` was counted at all. False
            only on the metadata-only path of a format that cannot count
            spectra without decoding them: PHI stores a stream of ion
            events rather than a list of spectra, so which pixels carry one
            is a property of the data, not of the header (issue #240). When
            False, ``n_spectra`` and ``total_peaks`` are 0 meaning "not
            counted", which is not the same as "none present" -- callers
            that display a count must say so rather than print the zero.
        total_peaks: Total number of peaks across all spectra (used for
            sparse matrix pre-allocation).
        estimated_memory_gb: Estimated dense memory footprint in GB.
        source_path: Absolute path to the source data.
        coordinate_offsets: Raw coordinate offsets ``(x, y, z)`` used to
            normalise coordinates to 0-based indexing.
        spectrum_type: Spectrum type string (e.g. ``"centroid spectrum"``),
            used to guide resampling decisions.
        peak_counts_per_pixel: Per-pixel peak counts for CSR ``indptr``
            construction in the streaming converter.  Array of size
            ``n_pixels`` where ``arr[pixel_idx] = peak_count`` and
            ``pixel_idx = z * (n_x * n_y) + y * n_x + x``.
        z_spacing_um: Distance between consecutive slices in micrometres,
            or ``None`` when the source cannot report it.  This is a
            *physical* distance set by how the sections were cut, and is
            unrelated to the in-plane pitch in ``pixel_size`` -- the two
            agree only by coincidence.  No reader populates this today:
            imzML has no standard term for it, and 3D acquisitions are
            frequently non-consecutive sections, so the value usually has
            to be supplied by the user.  The field exists so a format that
            *does* record it has somewhere to report it, and so the
            converter's precedence chain has something to consult.
    """

    dimensions: Tuple[int, int, int]
    coordinate_bounds: Tuple[float, float, float, float]
    mass_range: Tuple[float, float]
    pixel_size: Optional[Tuple[float, float]]
    n_spectra: int
    total_peaks: int
    estimated_memory_gb: float
    source_path: str
    coordinate_offsets: Optional[Tuple[int, int, int]] = None
    spectrum_type: Optional[str] = None
    peak_counts_per_pixel: Optional[NDArray[np.int32]] = None
    z_spacing_um: Optional[float] = None
    n_spectra_counted: bool = True

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
    """Complete metadata including format-specific details.

    Wraps :class:`EssentialMetadata` and adds vendor-specific information
    that is not needed for conversion but useful for provenance and QC.

    Attributes:
        essential: Core metadata required for conversion.
        format_specific: Vendor-specific metadata (e.g. ImzML CV params,
            Bruker property tables).
        acquisition_params: Acquisition parameters such as polarity,
            scan range, and laser settings.
        instrument_info: Instrument model, serial number, and software
            version.
        raw_metadata: Unprocessed metadata exactly as read from the
            source file, preserved for round-trip fidelity.
    """

    essential: EssentialMetadata
    format_specific: Dict[str, Any]
    acquisition_params: Dict[str, Any]
    instrument_info: Dict[str, Any]
    raw_metadata: Dict[str, Any]

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
