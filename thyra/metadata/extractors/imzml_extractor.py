# thyra/metadata/extractors/imzml_extractor.py
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pyimzml.ImzMLParser import ImzMLParser

from ...core.base_extractor import MetadataExtractor
from ...resampling.constants import (
    BinaryDataType,
    ImzMLAccessions,
    SpectrumType,
    normalize_spectrum_type,
)
from ...utils.imzml_coordinate_base import coordinate_bases
from ...utils.pyimzml_direct import read_spectrum_mzs_only
from ..ontology.cache import ONTOLOGY
from ..types import ComprehensiveMetadata, EssentialMetadata

logger = logging.getLogger(__name__)

# Micrometres per declared pixel-size unit, keyed by unitAccession. Everything
# downstream of the extractor -- ``EssentialMetadata.pixel_size``,
# ``obs/spatial_*`` -- is micrometres, so a unit outside this table is refused
# rather than passed through as a silent scale error.
UM_PER_UNIT = {
    "UO:0000016": 1000.0,  # millimeter
    "UO:0000017": 1.0,  # micrometer
    "UO:0000018": 0.001,  # nanometer
}

# Analyzer-component terms, in preference order. The first one found under a
# <componentList><analyzer> is surfaced as ``instrument_info["analyzer"]``, so
# on a hybrid that declares several analyzers the highest-resolution stage --
# the one that defines the mass axis of the stored spectra -- wins.
_ANALYZER_ACCESSIONS = (
    "MS:1000079",  # fourier transform ion cyclotron resonance mass spectrometer
    "MS:1000484",  # orbitrap
    "MS:1000084",  # time-of-flight
    "MS:1000264",  # ion trap
)

# The subset of analyzers whose axis-spacing law the resampling detector chain
# distinguishes. Values are the exact ``instrument_type`` strings
# ``FTICRDetector`` and ``OrbitrapDetector`` match on -- change both together.
# TOF is deliberately absent: an imzML declaring a TOF analyzer says nothing
# about which vendor grid the spectra arrived on, and stamping a TOF type here
# could arm detectors whose source-grid claims only hold for their own reader.
_ANALYZER_FAMILIES = {
    "MS:1000079": "FT-ICR",
    "MS:1000484": "Orbitrap",
}

# Instrument-model terms that imply an analyzer family when no analyzer
# component is declared. The local ontology tables are flat -- no is-a links --
# so family membership is enumerated rather than walked.
_FTICR_MODEL_ACCESSIONS = frozenset(
    {
        "MS:1000141",  # apex IV
        "MS:1000142",  # apex Q
        "MS:1000448",  # LTQ FT
        "MS:1000557",  # LTQ FT Ultra
        "MS:1000695",  # apex ultra
        "MS:1001548",  # Bruker Daltonics solarix series
        "MS:1001549",  # solariX
        "MS:1001556",  # Bruker Daltonics apex series
    }
)

_ORBITRAP_MODEL_ACCESSIONS = frozenset(
    {
        "MS:1000449",  # LTQ Orbitrap
        "MS:1000555",  # LTQ Orbitrap Discovery
        "MS:1000556",  # LTQ Orbitrap XL
        "MS:1000639",  # LTQ Orbitrap XL ETD
        "MS:1000643",  # MALDI LTQ Orbitrap
        "MS:1000649",  # Exactive
        "MS:1001742",  # LTQ Orbitrap Velos
        "MS:1001910",  # LTQ Orbitrap Elite
        "MS:1001911",  # Q Exactive
        "MS:1002416",  # Orbitrap Fusion
        "MS:1002417",  # Orbitrap Fusion ETD
        "MS:1002523",  # Q Exactive HF
        "MS:1002526",  # Exactive Plus
        "MS:1002634",  # Q Exactive Plus
        "MS:1002732",  # Orbitrap Fusion Lumos
        "MS:1002835",  # LTQ Orbitrap Classic
        "MS:1002877",  # Q Exactive HF-X
        "MS:1003028",  # Orbitrap Exploris 480
        "MS:1003029",  # Orbitrap Eclipse
        "MS:1003094",  # Orbitrap Exploris 240
        "MS:1003095",  # Orbitrap Exploris 120
        "MS:1003096",  # LTQ Orbitrap Velos Pro
        "MS:1003112",  # Orbitrap ID-X
    }
)

# timsTOF models carry no family here on purpose: the timsTOF decision is
# ``DataCharacteristics.is_timstof``, a substring match on the surfaced model
# name -- the same single path the native .d route goes through. Stamping an
# ``instrument_type`` as well would create a second, divergable path.
_TIMSTOF_MODEL_ACCESSIONS = frozenset(
    {
        "MS:1003005",  # timsTOF Pro
        "MS:1003123",  # Bruker Daltonics timsTOF series
        "MS:1003124",  # timsTOF fleX
        "MS:1003229",  # timsTOF
        "MS:1003230",  # timsTOF Pro 2
        "MS:1003231",  # timsTOF SCP
    }
)

_MODEL_FAMILIES: Dict[str, Optional[str]] = {
    **{acc: "FT-ICR" for acc in _FTICR_MODEL_ACCESSIONS},
    **{acc: "Orbitrap" for acc in _ORBITRAP_MODEL_ACCESSIONS},
    **{acc: None for acc in _TIMSTOF_MODEL_ACCESSIONS},
}

# Last-resort family match, applied to model text the file provides itself:
# the free-text value of MS:1000031 "instrument model", or the name of a model
# term newer than the shipped ontology tables. Tokens are vendor product names
# unambiguous enough that a substring hit identifies the family ("scimaX MRMS",
# "Orbitrap Astral", "solariX XR" all resolve correctly); generic words like
# "ft" or "tof" are deliberately not in the list.
_FAMILY_NAME_TOKENS = (
    ("solarix", "FT-ICR"),
    ("scimax", "FT-ICR"),
    ("mrms", "FT-ICR"),
    ("ft-icr", "FT-ICR"),
    ("fticr", "FT-ICR"),
    ("orbitrap", "Orbitrap"),
    ("exactive", "Orbitrap"),
    ("exploris", "Orbitrap"),
)


def _family_from_model_text(text: str) -> Optional[str]:
    """Resolve an analyzer family from free-form instrument-model text."""
    lowered = text.lower()
    for token, family in _FAMILY_NAME_TOKENS:
        if token in lowered:
            return family
    return None


class ImzMLMetadataExtractor(MetadataExtractor):
    """ImzML-specific metadata extractor with optimized two-phase extraction."""

    def __init__(
        self,
        parser: ImzMLParser,
        imzml_path: Path,
        spectrum_type: Optional[str] = None,
    ):
        """Initialize ImzML metadata extractor.

        Args:
            parser: Initialized ImzML parser
            imzml_path: Path to the ImzML file
            spectrum_type: Optional override for the spectrum representation,
                ``"profile"`` or ``"centroid"`` (SCiLS Lab spells this
                ``--rep_type``). When given, it wins over anything the file
                says. ``None`` -- the default -- leaves detection alone.

        Raises:
            ValueError: If ``spectrum_type`` is not a recognised
                representation.
        """
        super().__init__(parser)
        self.parser = parser
        self.imzml_path = imzml_path
        # Normalised at construction so a typo fails here rather than silently
        # falling through to auto-detection during extraction.
        self.spectrum_type_override = normalize_spectrum_type(spectrum_type)
        # See _coordinate_bases(): one memoised pass gives all three.
        self._coordinate_bases_value: Optional[Tuple[int, int, int]] = None

    def _extract_essential_impl(self) -> EssentialMetadata:
        """Extract essential metadata optimized for speed."""
        # Single coordinate scan for efficiency
        coords = np.array(self.parser.coordinates)

        if len(coords) == 0:
            raise ValueError("No coordinates found in ImzML file")

        dimensions = self._calculate_dimensions(coords)
        coordinate_bounds = self._calculate_bounds(coords)
        # Pass dimensions to collect per-pixel peak counts during scan
        mass_range, total_peaks, peak_counts = self._get_mass_range_complete(
            dimensions=dimensions
        )
        pixel_size = self._extract_pixel_size_fast()
        n_spectra = len(coords)
        estimated_memory = self._estimate_memory(n_spectra)

        # Check for centroid spectrum
        spectrum_type = self._detect_centroid_spectrum()

        return EssentialMetadata(
            dimensions=dimensions,
            coordinate_bounds=coordinate_bounds,
            mass_range=mass_range,
            pixel_size=pixel_size,
            n_spectra=n_spectra,
            total_peaks=total_peaks,
            estimated_memory_gb=estimated_memory,
            source_path=str(self.imzml_path),
            # What normalising the file's coordinates subtracted, so the
            # store can say where its origin came from: the converter
            # writes it to coordinate_systems.global.coordinate_offsets_px.
            # (1, 1, 1) for an ordinary 1-based file, (0, 0, ...) for a
            # 0-based one -- see _coordinate_bases() (issue #244).
            coordinate_offsets=self._coordinate_bases(coords),
            spectrum_type=spectrum_type,
            peak_counts_per_pixel=peak_counts,
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        """Extract comprehensive metadata with full XML parsing."""
        essential = self.get_essential()

        return ComprehensiveMetadata(
            essential=essential,
            format_specific=self._extract_imzml_specific(),
            acquisition_params=self._extract_acquisition_params(),
            instrument_info=self._extract_instrument_info(),
            raw_metadata=self._extract_raw_metadata(),
        )

    def _calculate_dimensions(self, coords: NDArray[np.int_]) -> Tuple[int, int, int]:
        """Calculate dataset dimensions from coordinates.

        Every axis is sized from the maximum *above its own base* (see
        :meth:`_coordinate_bases`), never from the maximum alone. Assuming
        1 for x and y under-counted a 0-based file by one row and one
        column, and the spectra that fell off the resulting grid were
        dropped with a warning naming a grid the file never declared
        (issue #244); assuming it for z under-counted a 0-based two-plane
        file as ``n_z = 1``.
        """
        if len(coords) == 0:
            return (0, 0, 0)

        max_coords = np.max(coords, axis=0)
        x_base, y_base, z_base = self._coordinate_bases(coords)
        return (
            int(max_coords[0]) - x_base + 1,
            int(max_coords[1]) - y_base + 1,
            int(max_coords[2]) - z_base + 1,
        )

    def _calculate_bounds(
        self, coords: NDArray[np.int_]
    ) -> Tuple[float, float, float, float]:
        """Calculate coordinate bounds (min_x, max_x, min_y, max_y).

        The file's **own** coordinates, not the 0-based indices the reader
        yields: a 1-based file reports ``1..n`` where :meth:`_calculate_dimensions`
        counts ``n``. Left that way deliberately when #244 rebased
        everything else -- nothing places a pixel from this field, and
        normalising it would move the stored metadata of every imzML that
        converts correctly today for no gain. What was subtracted to reach
        the store's indices is reported separately, and exactly, as
        ``EssentialMetadata.coordinate_offsets``.
        """
        if len(coords) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        x_coords = coords[:, 0].astype(float)
        y_coords = coords[:, 1].astype(float)

        return (
            float(np.min(x_coords)),
            float(np.max(x_coords)),
            float(np.min(y_coords)),
            float(np.max(y_coords)),
        )

    def _is_continuous_mode(self) -> bool:
        """Check if the ImzML file is in continuous mode.

        Continuous mode means all spectra share the same m/z axis.
        Detection checks file_description.param_by_name for "continuous" key.

        Returns:
            True if continuous mode, False otherwise (processed mode).
        """
        try:
            if not hasattr(self.parser, "metadata"):
                return False
            if self.parser.metadata is None:
                return False

            file_desc = getattr(self.parser.metadata, "file_description", None)
            if file_desc is None:
                return False

            param_by_name = getattr(file_desc, "param_by_name", None)
            if param_by_name is None or not isinstance(param_by_name, dict):
                return False

            return BinaryDataType.CONTINUOUS in param_by_name
        except Exception:
            return False

    def _get_mass_range_complete(
        self,
        dimensions: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Tuple[float, float], int, Optional[NDArray[np.int32]]]:
        """Complete mass range extraction.

        For continuous mode: reads only the first spectrum (all share same m/z axis).
        For processed mode: scans ALL spectra to find complete mass range.

        Also counts total peaks for COO matrix pre-allocation and
        optionally collects per-pixel peak counts for streaming conversion.

        Args:
            dimensions: Optional (n_x, n_y, n_z) grid dimensions.
                If provided, per-pixel peak counts will be collected.

        Returns:
            Tuple of ((min_mass, max_mass), total_peaks, peak_counts_per_pixel)
            peak_counts_per_pixel is None if dimensions not provided.

        Raises:
            ValueError: If no spectrum yields a mass range. There is no
                defensible fallback range for a mass spectrometry dataset:
                the invented ``(0.0, 1000.0)`` this used to return reached
                disk as ``uns/essential_metadata/mass_range`` and, with
                resampling on, sized the whole common axis. The same file
                with resampling off was refused loudly, so a flag decided
                whether a dataset with no readable spectra failed or was
                silently described wrongly.
        """
        try:
            coords = self.parser.coordinates
            n_spectra = len(coords)

            # Check if continuous mode - all spectra share the same m/z axis
            # Detection: check file_description.param_by_name for "continuous" key
            is_continuous = self._is_continuous_mode()

            if is_continuous:
                return self._get_mass_range_continuous(n_spectra, dimensions)
            else:
                return self._get_mass_range_processed(coords, n_spectra, dimensions)

        except ValueError:
            # The two branches above raise this when they find no mass range.
            # Re-wrapping it here would double the message, since both of
            # them already name the file.
            raise
        except Exception as e:
            logger.error(f"Mass range extraction failed: {e}")
            raise ValueError(
                f"Could not determine the mass range of {self.imzml_path}: {e}"
            ) from e

    def _get_mass_range_continuous(
        self,
        n_spectra: int,
        dimensions: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Tuple[float, float], int, Optional[NDArray[np.int32]]]:
        """Get mass range for continuous mode - read only first spectrum.

        In continuous mode, all spectra share the same m/z axis, so we only
        need to read one spectrum to get the mass range and peak count.

        Raises:
            ValueError: If spectrum 0 is empty. In continuous mode it is the
                only spectrum read, so an empty one leaves nothing to derive
                a range from.
        """
        logger.info(
            "Continuous mode detected - reading m/z axis from first spectrum only"
        )

        # Read first spectrum to get shared m/z axis
        mzs, _ = self.parser.getspectrum(0)
        n_peaks_per_spectrum = len(mzs)

        if n_peaks_per_spectrum == 0:
            raise ValueError(
                f"Could not determine the mass range of {self.imzml_path}: "
                "the file declares continuous mode, so every spectrum shares "
                "the m/z axis of spectrum 0 -- and spectrum 0 has no peaks."
            )

        min_mass = float(np.min(mzs))
        max_mass = float(np.max(mzs))
        total_peaks = n_peaks_per_spectrum * n_spectra

        # For continuous mode, all pixels have the same peak count
        peak_counts = None
        if dimensions is not None:
            n_x, n_y, n_z = dimensions
            n_pixels = n_x * n_y * n_z
            peak_counts = np.full(n_pixels, n_peaks_per_spectrum, dtype=np.int32)
            logger.info(
                f"All {n_pixels:,} pixels have {n_peaks_per_spectrum:,} peaks (continuous mode)"
            )

        logger.info(f"Mass range: {min_mass:.2f} - {max_mass:.2f} m/z")
        logger.info(
            f"Total peaks: {total_peaks:,} ({n_peaks_per_spectrum:,} per spectrum)"
        )
        return ((min_mass, max_mass), total_peaks, peak_counts)

    def _get_mass_range_processed(
        self,
        coords: List,
        n_spectra: int,
        dimensions: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Tuple[float, float], int, Optional[NDArray[np.int32]]]:
        """Get mass range for processed mode - scan all spectra.

        In processed mode, each spectrum can have different m/z values,
        so we must scan all spectra to find the complete mass range.

        Raises:
            ValueError: If not one of the scanned spectra yielded a peak.
        """
        logger.info("Processed mode - scanning ALL spectra for complete mass range...")

        peak_counts = self._init_peak_counts_array(dimensions)

        min_mass, max_mass, total_peaks = self._scan_all_spectra(
            coords, n_spectra, dimensions, peak_counts
        )

        if min_mass == float("inf"):
            raise ValueError(
                f"Could not determine the mass range of {self.imzml_path}: "
                f"none of the {n_spectra:,} spectra scanned yielded a peak. "
                "The .ibd may be truncated or unreadable."
            )

        logger.info(f"Complete mass range: {min_mass:.2f} - {max_mass:.2f} m/z")
        logger.info(f"Total peaks: {total_peaks:,}")
        return ((min_mass, max_mass), total_peaks, peak_counts)

    def _init_peak_counts_array(
        self, dimensions: Optional[Tuple[int, int, int]]
    ) -> Optional[NDArray[np.int32]]:
        """Initialize per-pixel peak counts array if dimensions provided.

        Args:
            dimensions: Optional (n_x, n_y, n_z) grid dimensions.

        Returns:
            Array of zeros or None if dimensions not provided.
        """
        if dimensions is None:
            return None
        n_x, n_y, n_z = dimensions
        n_pixels = n_x * n_y * n_z
        logger.info(f"Collecting per-pixel peak counts ({n_pixels:,} pixels)")
        return np.zeros(n_pixels, dtype=np.int32)

    def _scan_all_spectra(
        self,
        coords: List,
        n_spectra: int,
        dimensions: Optional[Tuple[int, int, int]],
        peak_counts: Optional[NDArray[np.int32]],
    ) -> Tuple[float, float, int]:
        """Scan all spectra to find mass range and count peaks.

        Args:
            coords: List of spectrum coordinates.
            n_spectra: Total number of spectra.
            dimensions: Optional grid dimensions for pixel indexing.
            peak_counts: Optional array to store per-pixel peak counts.

        Returns:
            Tuple of (min_mass, max_mass, total_peaks).
        """
        from tqdm import tqdm

        min_mass = float("inf")
        max_mass = float("-inf")
        total_peaks = 0

        with tqdm(
            total=n_spectra,
            desc="Scanning mass range and counting peaks",
            unit="spectrum",
        ) as pbar:
            for idx in range(n_spectra):
                result = self._process_spectrum_for_range(
                    idx, coords, dimensions, peak_counts
                )
                if result is not None:
                    spec_min, spec_max, n_peaks = result
                    min_mass = min(min_mass, spec_min)
                    max_mass = max(max_mass, spec_max)
                    total_peaks += n_peaks

                if idx % 50000 == 0 and idx > 0:
                    gc.collect()

                pbar.update(1)

        return min_mass, max_mass, total_peaks

    def _process_spectrum_for_range(
        self,
        idx: int,
        coords: List,
        dimensions: Optional[Tuple[int, int, int]],
        peak_counts: Optional[NDArray[np.int32]],
    ) -> Optional[Tuple[float, float, int]]:
        """Process a single spectrum for mass range and peak count.

        Args:
            idx: Spectrum index.
            coords: List of spectrum coordinates.
            dimensions: Optional grid dimensions.
            peak_counts: Optional array for per-pixel counts.

        Returns:
            Tuple of (min_mz, max_mz, n_peaks) or None if failed.
        """
        try:
            # Read and decode the m/z array only. The scan needs the m/z
            # extrema and the peak count; the intensity array -- half the
            # bytes read per spectrum -- was fetched and immediately
            # discarded. Parsers without pyimzml's offset tables fall back
            # to the documented getspectrum.
            mzs = read_spectrum_mzs_only(self.parser, idx)
            n_peaks = len(mzs)

            # Store per-pixel count if tracking
            if peak_counts is not None and dimensions is not None:
                self._store_pixel_peak_count(
                    idx, coords, dimensions, peak_counts, n_peaks
                )

            if n_peaks > 0:
                return (float(np.min(mzs)), float(np.max(mzs)), n_peaks)

            return None

        except Exception as e:
            logger.debug(f"Failed to read spectrum {idx}: {e}")
            return None

    def _coordinate_bases(self, coords: List) -> Tuple[int, int, int]:
        """The ``(x, y, z)`` values in this file that map onto index 0.

        The same rebasing :meth:`ImzMLReader._coordinate_bases` does, and
        it has to agree with it: these counts become the CSR ``indptr``, so
        a base the reader does not share lands a pixel's peak count on
        another pixel's row. Both call
        :func:`thyra.utils.imzml_coordinate_base.coordinate_bases`, which
        holds the rule and the reasoning -- a 0 folds down on x and y, z
        takes the smallest value present.

        Memoised; the scan is one pass over the coordinate list.

        Args:
            coords: The parser's coordinate list.

        Returns:
            ``(x_base, y_base, z_base)``.
        """
        if self._coordinate_bases_value is None:
            self._coordinate_bases_value = coordinate_bases(coords)
        return self._coordinate_bases_value

    def _store_pixel_peak_count(
        self,
        idx: int,
        coords: List,
        dimensions: Tuple[int, int, int],
        peak_counts: NDArray[np.int32],
        n_peaks: int,
    ) -> None:
        """Store peak count for a pixel.

        Args:
            idx: Spectrum index.
            coords: List of spectrum coordinates.
            dimensions: Grid dimensions (n_x, n_y, n_z).
            peak_counts: Array to store counts.
            n_peaks: Number of peaks in this spectrum.
        """
        # Every axis is rebased on the file -- see _coordinate_bases().
        x, y, z = coords[idx]
        x_base, y_base, z_base = self._coordinate_bases(coords)
        x, y, z = x - x_base, y - y_base, z - z_base
        n_x, n_y, n_z = dimensions
        pixel_idx = z * (n_x * n_y) + y * n_x + x
        if 0 <= pixel_idx < len(peak_counts):
            peak_counts[pixel_idx] = n_peaks

    def get_mass_range_for_resampling(self) -> Tuple[float, float]:
        """Get accurate mass range required for resampling.

        This performs a complete scan of all spectra to ensure no m/z
        values are missed when building the resampled axis.

        Raises:
            ValueError: If the file yields no mass range at all. Propagated
                deliberately: resampling is precisely the path that used to
                turn the invented ``(0.0, 1000.0)`` into a real axis on disk.
        """
        mass_range, _, _ = self._get_mass_range_complete()
        return mass_range

    def _extract_pixel_size_fast(self) -> Optional[Tuple[float, float]]:
        """Fast pixel size extraction from imzmldict, converted to micrometres.

        ``imzmldict`` discards ``unitAccession`` -- ``convert_cv_param`` takes
        no unit argument -- so ``pixel size x`` is a bare number in whatever
        unit the vendor declared. Taking that number at face value stored a
        nanometre pixel size as micrometres, 1000x too large, and
        ``convert_msi`` still returned ``True``. The unit survives on the
        ParamGroup path, so it is read from there and the value converted.
        """
        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            # Check for pixel size parameters in the parsed dictionary
            x_size = self.parser.imzmldict.get("pixel size x")
            y_size = self.parser.imzmldict.get("pixel size y")

            if x_size is not None and y_size is not None:
                try:
                    x_size, y_size = float(x_size), float(y_size)
                except (ValueError, TypeError):
                    return None  # Defer to comprehensive extraction

                units = self._pixel_size_unit_accessions()
                return (
                    self._pixel_size_to_um(
                        x_size, units.get(ImzMLAccessions.PIXEL_SIZE_X)
                    ),
                    self._pixel_size_to_um(
                        y_size, units.get(ImzMLAccessions.PIXEL_SIZE_Y)
                    ),
                )

        return None  # Defer to comprehensive extraction

    def _pixel_size_unit_accessions(self) -> Dict[str, Optional[str]]:
        """The declared ``unitAccession`` of each pixel-size cvParam.

        ``__readimzmlmeta`` resolves each accession by first match anywhere in
        the document, so the unit is looked up the same way: scan-settings
        blocks in document order, first declaration of each accession wins.
        That keeps the unit paired with the value ``imzmldict`` actually holds
        when a file carries more than one ``<scanSettings>`` block.
        """
        units: Dict[str, Optional[str]] = {}
        metadata = getattr(self.parser, "metadata", None)
        scan_settings = getattr(metadata, "scan_settings", None)
        if not isinstance(scan_settings, dict):
            return units

        wanted = (ImzMLAccessions.PIXEL_SIZE_X, ImzMLAccessions.PIXEL_SIZE_Y)
        for group in scan_settings.values():
            for param in getattr(group, "cv_params", []):
                # (name, accession, value, raw_name, raw_value, unit_name,
                #  unit_accession)
                accession, unit_accession = param[1], param[6]
                if accession in wanted and accession not in units:
                    units[accession] = unit_accession
        return units

    def _pixel_size_to_um(self, value: float, unit_accession: Optional[str]) -> float:
        """Convert a declared pixel size to micrometres, or refuse.

        No declared unit keeps the historical reading: the bare number is
        micrometres. Real vendor files (the IONTOF class among them) write
        ``IMS:1000046`` with no ``unitAccession`` at all, so refusing here
        would reject files that were being read correctly.

        Raises:
            ValueError: If the file declares a unit outside ``UM_PER_UNIT``.
                Guessing a factor for, say, centimetre would be the same
                silent scale error this method exists to close.
        """
        if unit_accession is None:
            return value
        factor = UM_PER_UNIT.get(unit_accession)
        if factor is None:
            raise ValueError(
                f"{self.imzml_path} declares its pixel size in unit "
                f"{unit_accession!r}, which Thyra cannot convert to "
                f"micrometres. Supported units: UO:0000016 (millimeter), "
                f"UO:0000017 (micrometer), UO:0000018 (nanometer)."
            )
        return value * factor

    def _estimate_memory(self, n_spectra: int) -> float:
        """Estimate memory usage in GB."""
        # Rough estimate: assume average 1000 peaks per spectrum,
        # 8 bytes per float
        avg_peaks_per_spectrum = 1000
        bytes_per_value = 8  # float64
        estimated_bytes = (
            n_spectra * avg_peaks_per_spectrum * 2 * bytes_per_value
        )  # mz + intensity
        return estimated_bytes / (1024**3)  # Convert to GB

    def _extract_imzml_specific(self) -> Dict[str, Any]:
        """Extract ImzML format-specific metadata.

        ``file_mode`` goes through :meth:`_is_continuous_mode`, which reads
        ``IMS:1000030``/``IMS:1000031`` off the file description. It used to
        read ``getattr(self.parser, "continuous", False)`` -- an attribute
        ``ImzMLParser`` does not define, so the default always won and every
        store Thyra has ever written says ``"processed"`` regardless of what
        the file declares.
        """
        format_specific: Dict[str, Any] = {
            "imzml_version": "1.1.0",  # Default version
            "file_mode": (
                BinaryDataType.CONTINUOUS
                if self._is_continuous_mode()
                else BinaryDataType.PROCESSED
            ),
            "ibd_file": str(self.imzml_path.with_suffix(".ibd")),
            "uuid": self._extract_uuid(),
            "spectrum_count": len(self.parser.coordinates),
            "scan_settings": {},
            "ion_mobility": self._extract_ion_mobility(),
        }

        return format_specific

    def _extract_ion_mobility(self) -> Dict[str, Any]:
        """Whether the file declares a third, ion mobility, binary array.

        Reported from the referenceableParamGroups alone -- no spectrum is
        read here -- so this says what the file *declares* (the array term,
        its quantity and unit), not whether every pixel shares one axis;
        the reader settles that when it reads.
        """
        from ...core.mobility import classify_mobility_array
        from ...readers.imzml.mobility_array import detect_mobility_array

        spec = detect_mobility_array(getattr(self.parser, "metadata", None))
        if spec is None:
            return {"present": False}
        kind_accession, kind_name = classify_mobility_array(
            spec.array_accession, spec.unit_accession
        )
        report: Dict[str, Any] = {
            "present": True,
            "array_accession": spec.array_accession,
            "array_name": spec.array_name,
        }
        if kind_name is not None:
            report["separation"] = kind_name
            report["separation_accession"] = kind_accession
        if spec.unit_name is not None:
            report["unit"] = spec.unit_name
        if spec.unit_accession is not None:
            report["unit_accession"] = spec.unit_accession
        return report

    def _extract_uuid(self) -> Optional[str]:
        """Read the ``IMS:1000080`` binary-file UUID from the file description.

        Keyed by name rather than by position. The previous form was
        ``cv_params[0][2]`` -- the value of whichever cvParam the vendor
        happened to write first -- which on a real IONTOF file is
        ``MS:1000128 profile spectrum`` and so stored the boolean ``True``
        as the dataset's UUID. SCiLS puts the UUID first, so the same
        expression was correct there by luck.

        Vendors differ on whether the value carries the registry-format
        braces (IONTOF writes ``{...}``, SCiLS does not); they are stripped
        so the stored field is one shape whatever wrote the file.

        Returns:
            The UUID string, or ``None`` if the file declares none.
        """
        try:
            metadata = getattr(self.parser, "metadata", None)
            file_desc = getattr(metadata, "file_description", None)
            param_by_name = getattr(file_desc, "param_by_name", None)
            if not isinstance(param_by_name, dict):
                return None
            value = param_by_name.get(ImzMLAccessions.UUID_NAME)
            if not isinstance(value, str):
                return None
            return value.strip().strip("{}")
        except Exception as e:
            logger.debug(f"Could not extract UUID: {e}")
            return None

    def _extract_acquisition_params(self) -> Dict[str, Any]:
        """Extract acquisition parameters from XML metadata."""
        params = {}

        # Extract pixel size with full XML parsing if not found in fast
        # extraction
        if not self.get_essential().has_pixel_size:
            pixel_size = self._extract_pixel_size_from_xml()
            if pixel_size:
                params["pixel_size_x_um"] = pixel_size[0]
                params["pixel_size_y_um"] = pixel_size[1]

        # Add other acquisition parameters from imzmldict
        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            acquisition_keys = [
                "scan direction",
                "scan pattern",
                "scan type",
                "laser power",
                "laser frequency",
                "laser spot size",
            ]
            for key in acquisition_keys:
                if key in self.parser.imzmldict:
                    params[key.replace(" ", "_")] = self.parser.imzmldict[key]

        return params

    def _extract_instrument_info(self) -> Dict[str, Any]:
        """Extract instrument information from the instrumentConfiguration blocks.

        This used to read ``imzmldict`` for "instrument model" and friends, but
        pyimzml's ``__readimzmlmeta`` only ever collects pixel-geometry and
        laser parameters -- none of those keys can exist, so every store Thyra
        wrote carried an empty ``instrument_information`` for imzML sources,
        and neither ``FTICRDetector`` nor ``OrbitrapDetector`` could ever fire
        on an imzML. The declarations survive on ``parser.metadata``, so they
        are read from there:

        - ``instrument_model`` / ``instrument_serial_number`` -- from the
          instrumentConfiguration's own cvParams, referenceableParamGroup
          inheritance included (Bruker and SCiLS exports put the model term in
          a ``CommonInstrumentParams`` group).
        - ``analyzer`` -- the CV name of the first recognised
          ``<componentList><analyzer>`` term, which is what
          ``normalize_analyzer`` in the metadata schema accepts.
        - ``instrument_type`` -- the exact string the resampling detector
          chain matches on (``"FT-ICR"`` / ``"Orbitrap"``), when the analyzer
          or the model resolves to a family the chain distinguishes.  Absent
          otherwise: an unstated analyzer stays unstated.
        """
        info: Dict[str, Any] = {}
        configs = self._instrument_configurations()
        if not configs:
            return info

        model = self._instrument_model(configs)
        if model:
            info["instrument_model"] = model

        serial = self._config_param(configs, "MS:1000529")
        if isinstance(serial, str) and serial.strip():
            info["instrument_serial_number"] = serial.strip()

        analyzer_accession = self._analyzer_accession(configs)
        if analyzer_accession is not None:
            info["analyzer"] = ONTOLOGY.terms[analyzer_accession][0]

        family = self._analyzer_family(configs, analyzer_accession, model)
        if family is not None:
            info["instrument_type"] = family

        return info

    def _instrument_configurations(self) -> List[Any]:
        """The parsed instrumentConfiguration ParamGroups, in document order."""
        metadata = getattr(self.parser, "metadata", None)
        configs = getattr(metadata, "instrument_configurations", None)
        if not isinstance(configs, dict):
            return []
        return list(configs.values())

    @staticmethod
    def _config_param(configs: List[Any], accession: str) -> Any:
        """The first value any configuration declares for ``accession``."""
        for config in configs:
            params = getattr(config, "param_by_accession", None)
            if isinstance(params, dict) and accession in params:
                return params[accession]
        return None

    def _instrument_model(self, configs: List[Any]) -> Optional[str]:
        """The declared instrument model, from the shape the exporter chose.

        Vendors write the model two ways: generic ``MS:1000031 instrument
        model`` carrying the name as free text, or the model's own child term
        as a valueless flag (``MS:1001549 solariX``, ``MS:1003124 timsTOF
        fleX``).  The explicit value wins; a known model accession resolves
        through the shipped ontology table; and a term newer than the table is
        still usable when its *name* identifies a family.
        """
        value = self._config_param(configs, "MS:1000031")
        if isinstance(value, str) and value.strip():
            return value.strip()

        for config in configs:
            params = getattr(config, "param_by_accession", None)
            if not isinstance(params, dict):
                continue
            for accession in params:
                if accession in _MODEL_FAMILIES:
                    return ONTOLOGY.terms[accession][0]

        for config in configs:
            params = getattr(config, "param_by_name", None)
            if not isinstance(params, dict):
                continue
            for name, value in params.items():
                if (
                    isinstance(name, str)
                    and value is True
                    and _family_from_model_text(name) is not None
                ):
                    return name

        return None

    @staticmethod
    def _analyzer_accession(configs: List[Any]) -> Optional[str]:
        """The highest-priority analyzer term declared on any componentList."""
        declared: set = set()
        for config in configs:
            for component in getattr(config, "components", None) or []:
                if getattr(component, "type", None) != "analyzer":
                    continue
                params = getattr(component, "param_by_accession", None)
                if isinstance(params, dict):
                    declared.update(params)

        for accession in _ANALYZER_ACCESSIONS:
            if accession in declared:
                return accession
        return None

    def _analyzer_family(
        self,
        configs: List[Any],
        analyzer_accession: Optional[str],
        model: Optional[str],
    ) -> Optional[str]:
        """Resolve the axis family, preferring the analyzer declaration.

        The analyzer cvParam states the physics outright, so it outranks the
        model term, which outranks a substring match on free-form model text.
        A file that declares none of the three resolves to ``None``.
        """
        if analyzer_accession in _ANALYZER_FAMILIES:
            return _ANALYZER_FAMILIES[analyzer_accession]
        if analyzer_accession is not None:
            # A recognised analyzer outside the family table (TOF, ion trap)
            # is a real declaration; do not second-guess it from the model.
            return None

        for config in configs:
            params = getattr(config, "param_by_accession", None)
            if not isinstance(params, dict):
                continue
            for accession in params:
                family = _MODEL_FAMILIES.get(accession)
                if family is not None:
                    return family

        if model is not None:
            return _family_from_model_text(model)
        return None

    def _extract_raw_metadata(self) -> Dict[str, Any]:
        """Extract raw metadata from imzmldict and spectrum cvParams."""
        raw_metadata = {}

        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            raw_metadata = dict(self.parser.imzmldict)

        # Extract spectrum-level cvParams for centroid detection
        cv_params = self._extract_spectrum_cvparams()
        if cv_params:
            raw_metadata["cvParams"] = cv_params

        return raw_metadata

    def _extract_spectrum_cvparams(self) -> Optional[List[Dict[str, Any]]]:
        """Extract the file-description cvParams, accessions included.

        The tuples on ``ParamGroup.cv_params`` are
        ``(name, accession, value, raw_name, raw_value, unit_name,
        unit_accession)``.  The accession is the part that makes the
        stored copy lossless: the name alone cannot be resolved back to
        the CV concept, and the whole point of carrying these through is
        that the converted store is annotated with the same PSI CV terms
        as the raw file it came from.  Unit fields are included only
        when the source set them, following the omit-vs-empty
        convention.
        """
        try:
            if not hasattr(self.parser, "metadata") or not self.parser.metadata:
                return None

            file_desc = getattr(self.parser.metadata, "file_description", None)
            if file_desc is None or not hasattr(file_desc, "cv_params"):
                return None

            cv_params = []
            for (
                name,
                accession,
                value,
                _,
                _,
                unit_name,
                unit_acc,
            ) in file_desc.cv_params:
                entry: Dict[str, Any] = {
                    "name": name,
                    "accession": accession,
                    "value": value,
                }
                if unit_name:
                    entry["unit_name"] = unit_name
                if unit_acc:
                    entry["unit_accession"] = unit_acc
                cv_params.append(entry)

            return cv_params or None
        except Exception as e:
            logger.debug(f"Could not extract spectrum cvParams: {e}")
            return None

    def _detect_centroid_spectrum(self) -> Optional[str]:
        """Detect spectrum type by looking for MS:1000127 (centroid) or MS:1000128 (profile).

        An explicit ``spectrum_type`` override wins over everything, so a user
        can correct a file that declares the wrong thing. Otherwise what the
        file *declares* wins, wherever it is written, and only when the file
        declares nothing does Thyra guess -- see
        :meth:`_guess_spectrum_type_from_storage_mode` for what that guess is
        worth.
        """
        try:
            # 0. An explicit override outranks the file.
            if self.spectrum_type_override is not None:
                self._report_spectrum_type_override()
                return self.spectrum_type_override

            # 1. The declaration, from metadata the parser already holds.
            result = self._check_parser_metadata_for_spectrum_type()
            if result:
                return result

            # 2. The same declaration, wherever else in the document it sits.
            #    Streams the XML, so it is tried only if step 1 came up empty.
            result = self._check_xml_for_spectrum_type()
            if result:
                return result

            # 3. Nothing declared. Only now is a guess appropriate.
            return self._guess_spectrum_type_from_storage_mode()
        except Exception as e:
            logger.debug(f"Could not detect spectrum type: {e}")
            return None

    def _report_spectrum_type_override(self) -> None:
        """Log an override, loudly when it contradicts the file's declaration.

        Overriding a declared ``MS:1000127``/``MS:1000128`` is a legitimate
        thing to want -- files do declare the wrong representation -- and it is
        also a good way to corrupt a conversion silently. So the two cases are
        not logged alike: agreeing with the file, or overriding a file that
        declares nothing, is INFO; contradicting an explicit declaration is
        WARNING and names both values.

        This still reads the file's *declarations* (steps 1 and 2) in order to
        compare, but never its guess -- there is nothing informative about
        contradicting a guess. That costs at most the same bounded XML scan
        detection would have done, and only on the opt-in path.
        """
        override = self.spectrum_type_override
        declared = self._check_parser_metadata_for_spectrum_type()
        if declared is None:
            declared = self._check_xml_for_spectrum_type()

        if declared is None:
            logger.info(
                "Using spectrum_type override %r; the file declares neither %s "
                "(centroid) nor %s (profile).",
                override,
                ImzMLAccessions.CENTROID_SPECTRUM,
                ImzMLAccessions.PROFILE_SPECTRUM,
            )
        elif declared == override:
            logger.info(
                "Using spectrum_type override %r, which agrees with the file.",
                override,
            )
        else:
            logger.warning(
                "spectrum_type override %r CONTRADICTS the file, which declares "
                "%r. Proceeding with the override -- the stored spectrum type, "
                "and any decision Thyra makes from it, will not match what the "
                "file says about itself. Remove the override to trust the file.",
                override,
                declared,
            )

    def _check_xml_for_spectrum_type(self) -> Optional[str]:
        """Check XML for spectrum type markers using streaming parser.

        Looks for PSI-MS controlled vocabulary accession codes:
        - MS:1000127 = centroid spectrum
        - MS:1000128 = profile spectrum

        Uses iterparse for memory-efficient streaming - stops as soon as a
        spectrum type marker is found, avoiding full file load for large datasets.
        """
        try:
            import xml.etree.ElementTree as ET  # nosec B405

            # Use iterparse for streaming - only parses until we find what we need
            # The spectrum type cvParam is typically in the fileDescription
            # section near the beginning of the file
            context = ET.iterparse(str(self.imzml_path), events=("end",))  # nosec B314

            elements_checked = 0
            max_elements = 10000  # Limit search to first 10k elements

            for event, elem in context:
                elements_checked += 1

                if elem.tag.endswith("cvParam"):
                    accession = elem.get("accession", "")
                    if accession == ImzMLAccessions.CENTROID_SPECTRUM:
                        logger.info(
                            f"Detected centroid spectrum from {ImzMLAccessions.CENTROID_SPECTRUM}"
                        )
                        del context
                        return SpectrumType.CENTROID
                    if accession == ImzMLAccessions.PROFILE_SPECTRUM:
                        logger.info(
                            f"Detected profile spectrum from {ImzMLAccessions.PROFILE_SPECTRUM}"
                        )
                        del context
                        return SpectrumType.PROFILE

                # Clear processed elements to save memory
                elem.clear()

                # Stop after checking enough elements - spectrum type info is at the start
                if elements_checked >= max_elements:
                    logger.debug(
                        f"Spectrum type marker not found in first {max_elements} elements"
                    )
                    break

            del context
        except Exception as e:
            logger.debug(f"XML streaming parse failed: {e}")
        return None

    def _get_xml_parser(self):
        """Get XML parser, preferring defusedxml for security."""
        try:
            # Use defusedxml for secure parsing
            import defusedxml.ElementTree as ET

            return ET
        except ImportError:
            # Fallback to standard library with warning
            import xml.etree.ElementTree as ET  # nosec B405

            logger.warning("defusedxml not available, using xml.etree.ElementTree")
            return ET

    def _file_description_params(self) -> Optional[Dict[str, Any]]:
        """Return the fileDescription cvParams the parser already parsed.

        ``param_by_name`` maps a term's CV *name* to its parsed value, and a
        valueless cvParam -- which is what a flag like ``profile spectrum``
        is -- parses to ``True``.
        """
        if not (hasattr(self.parser, "metadata") and self.parser.metadata):
            return None

        if not hasattr(self.parser.metadata, "file_description"):
            return None

        file_desc = self.parser.metadata.file_description
        params = getattr(file_desc, "param_by_name", None)
        if not isinstance(params, dict):
            return None

        return params

    def _check_parser_metadata_for_spectrum_type(self) -> Optional[str]:
        """Read the declared spectrum representation out of the fileDescription.

        Cheap: the parser has already read this, so no XML pass is needed.
        ``SpectrumType.CENTROID`` and ``SpectrumType.PROFILE`` are the CV
        names of ``MS:1000127`` and ``MS:1000128``, which is what
        ``param_by_name`` is keyed on.
        """
        params = self._file_description_params()
        if params is None:
            return None

        if params.get(SpectrumType.CENTROID, False):
            logger.info(
                f"Detected centroid spectrum from declared "
                f"{ImzMLAccessions.CENTROID_SPECTRUM} in fileDescription"
            )
            return SpectrumType.CENTROID

        if params.get(SpectrumType.PROFILE, False):
            logger.info(
                f"Detected profile spectrum from declared "
                f"{ImzMLAccessions.PROFILE_SPECTRUM} in fileDescription"
            )
            return SpectrumType.PROFILE

        return None

    def _guess_spectrum_type_from_storage_mode(self) -> Optional[str]:
        """Last-resort guess: assume a processed-mode file is centroided.

        Processed and centroid are orthogonal. ``IMS:1000031`` says each
        spectrum carries its own m/z array; it says nothing about whether
        the peaks in it are centroided or profile. Plenty of instruments
        export profile spectra in processed mode -- ``bellini.imzML`` is
        processed *and* declares ``MS:1000128 profile spectrum``.

        This used to run before the declared accession was ever read, so
        every processed file was reported as centroid whatever it said about
        itself. It now runs only when the file declares neither
        ``MS:1000127`` nor ``MS:1000128`` anywhere, where a guess is all
        there is. Peak lists are the common case for processed exports, so
        centroid is the better guess -- but it is still a guess, hence the
        warning.
        """
        params = self._file_description_params()
        if params is None:
            return None

        if params.get(BinaryDataType.PROCESSED, False):
            logger.warning(
                "This imzML declares neither %s (centroid) nor %s (profile). "
                "Assuming centroid, because it is stored in processed mode -- "
                "but the two are independent, so check the result if the "
                "spectra are actually profile.",
                ImzMLAccessions.CENTROID_SPECTRUM,
                ImzMLAccessions.PROFILE_SPECTRUM,
            )
            return SpectrumType.CENTROID

        return None

    def _extract_pixel_size_from_xml(self) -> Optional[Tuple[float, float]]:
        """Extract pixel size using full XML parsing as fallback.

        The unit conversion happens outside the ``try`` so a refused unit
        propagates as loudly as it does on the fast path, while a merely
        unparseable document still degrades to "no pixel size".
        """
        x_size = None
        y_size = None
        x_unit = None
        y_unit = None

        try:
            if not hasattr(self.parser, "metadata") or not hasattr(
                self.parser.metadata, "root"
            ):
                return None

            root = self.parser.metadata.root

            # Define namespaces for XML parsing
            namespaces = {
                "mzml": "http://psi.hupo.org/ms/mzml",
                "ims": "http://www.maldi-msi.org/download/imzml/imagingMS.obo",
            }

            # Search for cvParam elements with the pixel size accessions
            for cvparam in root.findall(".//mzml:cvParam", namespaces):
                accession = cvparam.get("accession")
                if accession == ImzMLAccessions.PIXEL_SIZE_X:
                    x_size = float(cvparam.get("value", 0))
                    x_unit = cvparam.get("unitAccession")
                elif accession == ImzMLAccessions.PIXEL_SIZE_Y:
                    y_size = float(cvparam.get("value", 0))
                    y_unit = cvparam.get("unitAccession")

        except Exception as e:
            logger.warning(f"Failed to parse XML metadata for pixel size: {e}")
            return None

        if x_size is not None and y_size is not None:
            x_um = self._pixel_size_to_um(x_size, x_unit)
            y_um = self._pixel_size_to_um(y_size, y_unit)
            logger.info(f"Detected pixel size from XML: x={x_um}μm, y={y_um}μm")
            return (x_um, y_um)

        return None
