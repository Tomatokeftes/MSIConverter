"""Data characteristics for resampling decisions."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import SpectrumType, Thresholds


@dataclass
class DataCharacteristics:
    """Detected characteristics of MSI data for resampling decisions.

    This dataclass consolidates all the information needed to make
    intelligent resampling decisions, extracted from metadata.
    """

    # Core data properties
    has_shared_mass_axis: bool = False  # True for continuous, False for processed
    spectrum_type: Optional[str] = None  # "centroid spectrum" or "profile spectrum"

    # Instrument identification
    instrument_type: Optional[str] = None  # "MALDI-TOF", "timsTOF", "FT-ICR", etc.
    instrument_name: Optional[str] = None  # Specific model name
    manufacturer: Optional[str] = None  # "Bruker", etc.

    # Software provenance
    software_source: Optional[str] = None  # "SCiLS Lab", etc.

    # Data statistics
    total_peaks: Optional[int] = None
    n_spectra: Optional[int] = None

    # Format-specific flags
    is_rapiflex_format: bool = False
    is_timstof: bool = False
    is_phi_tofsims: bool = False
    is_waters_raw: bool = False
    # Waters only: the analyser is a SELECT SERIES MRT, per _extern.inf.
    is_waters_mrt: bool = False
    # Profile sources whose digitiser grid is known: the spacing between
    # consecutive stored samples at m/z 1000, in Da. Lets a detector anchor
    # the default bin width to the data's own sampling rather than to a
    # constant that fits one instrument.
    profile_sample_spacing_da_at_1000: Optional[float] = None

    @property
    def needs_resampling(self) -> bool:
        """Determine if data needs mass axis alignment.

        Continuous data (shared mass axis) doesn't need resampling for alignment.
        Processed data (different m/z per spectrum) needs resampling.
        """
        return not self.has_shared_mass_axis

    @property
    def is_profile_data(self) -> bool:
        """Check if this is profile (continuous signal) data."""
        return self.spectrum_type == SpectrumType.PROFILE

    @property
    def is_centroid_data(self) -> bool:
        """Check if this is centroid (discrete peaks) data."""
        return self.spectrum_type == SpectrumType.CENTROID

    @property
    def avg_peaks_per_spectrum(self) -> Optional[float]:
        """Calculate average peaks per spectrum."""
        if self.n_spectra and self.n_spectra > 0 and self.total_peaks:
            return self.total_peaks / self.n_spectra
        return None

    @property
    def is_high_density_profile(self) -> bool:
        """Check if this appears to be high-density profile data.

        Profile data typically has >5000 points per spectrum,
        indicating continuous signal rather than centroid peaks.

        This says how densely the spectra are sampled. It says nothing about
        which instrument produced them, so it **must not be used to pick a
        resampling strategy or an axis type**: those encode assumptions about
        the acquisition's physics. ``RapiflexDetector`` used to match on this
        alone, which handed MALDI-TOF treatment to any dense profile data --
        TOF-SIMS, Orbitrap, anything -- with no check that the data was
        MALDI-TOF at all.
        """
        avg = self.avg_peaks_per_spectrum
        return (
            self.is_profile_data
            and avg is not None
            and avg > Thresholds.PROFILE_PEAK_DENSITY
        )

    @property
    def is_maldi_tof(self) -> bool:
        """Check if this is MALDI-TOF data.

        Carries the same caveat as :attr:`is_high_density_profile`, which its
        last clause consults: density plus a vendor name is not a modality.
        No detector calls this.
        """
        return (
            self.is_rapiflex_format
            or self.instrument_type == "MALDI-TOF"
            or (self.manufacturer == "Bruker" and self.is_high_density_profile)
        )

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "DataCharacteristics":
        """Create DataCharacteristics from metadata dictionary.

        Args:
            metadata: Metadata dictionary with nested essential_metadata,
                     instrument_info, format_specific, etc.

        Returns:
            DataCharacteristics instance populated from metadata.
        """
        essential = metadata.get("essential_metadata", {})
        instrument_info = metadata.get("instrument_info", {})
        format_specific = metadata.get("format_specific", {})
        global_meta = metadata.get("GlobalMetadata", {})

        # Extract values with safe defaults
        spectrum_type = essential.get("spectrum_type") if essential else None
        total_peaks = essential.get("total_peaks") if essential else None
        n_spectra = essential.get("n_spectra") if essential else None

        # Instrument info
        instrument_type = (
            instrument_info.get("instrument_type") if instrument_info else None
        )
        manufacturer = instrument_info.get("manufacturer") if instrument_info else None
        # Bruker .d surfaces the name in GlobalMetadata, which stays
        # authoritative; an imzML surfaces the declared model term through
        # instrument_info instead. Folding the two here keeps ``is_timstof``
        # below the single deciding path for both routes -- a timsTOF fleX
        # exported to imzML used to lose its identity entirely because only
        # GlobalMetadata was consulted.
        instrument_name = global_meta.get("InstrumentName") if global_meta else None
        if not instrument_name and instrument_info:
            instrument_name = instrument_info.get(
                "instrument_model"
            ) or instrument_info.get("instrument_name")

        # Format detection
        is_rapiflex = (
            format_specific.get("format") == "Rapiflex" if format_specific else False
        )
        # Substring match: Bruker names timsTOF variants several ways
        # ("timsTOF Maldi 2", "timsTOF Pro 2", "timsTOF fleX MALDI-2",
        # "timsTOF SCP", etc.).  An exact match was missing every
        # variant except the original Maldi 2, so the InstrumentDetectorChain
        # silently fell through TimsTOFDetector and ended up at the
        # DefaultDetector returning CONSTANT.  This match is permissive
        # but case-sensitive on the brand "timsTOF" / "timstof".
        is_timstof = bool(instrument_name and "timstof" in instrument_name.lower())

        # PhiMetadataExtractor stamps the exact string "PHI SmartSoft-TOF raw";
        # the prefix match leaves room for a future SmartSoft revision to
        # extend it without silently dropping back to DefaultDetector.
        source_format = format_specific.get("format") if format_specific else None
        is_phi_tofsims = bool(source_format and source_format.startswith("PHI "))

        # WatersMetadataExtractor stamps "Waters MassLynx raw"; same prefix
        # convention as the PHI flag above.
        is_waters_raw = bool(source_format and source_format.startswith("Waters "))
        is_waters_mrt = bool(is_waters_raw and format_specific.get("is_mrt"))
        spacing = (
            format_specific.get("profile_sample_spacing_da_at_1000")
            if format_specific
            else None
        )
        profile_sample_spacing = (
            float(spacing)
            if isinstance(spacing, (int, float)) and spacing > 0
            else None
        )

        return cls(
            spectrum_type=spectrum_type,
            total_peaks=total_peaks,
            n_spectra=n_spectra,
            instrument_type=instrument_type,
            instrument_name=instrument_name,
            manufacturer=manufacturer,
            is_rapiflex_format=is_rapiflex,
            is_timstof=is_timstof,
            is_phi_tofsims=is_phi_tofsims,
            is_waters_raw=is_waters_raw,
            is_waters_mrt=is_waters_mrt,
            profile_sample_spacing_da_at_1000=profile_sample_spacing,
        )
