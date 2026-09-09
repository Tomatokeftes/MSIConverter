# thyra/metadata/extractors/phi_extractor.py
"""Metadata extractor for PHI SmartSoft-TOF ToF-SIMS data.

Reads from the parsed acquisition header and the reader's aggregated event
index, so no additional file passes are needed.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from ...core.base_extractor import MetadataExtractor
from ..types import ComprehensiveMetadata, EssentialMetadata

if TYPE_CHECKING:
    from ...readers.phi.phi_reader import PhiReader

logger = logging.getLogger(__name__)


class PhiMetadataExtractor(MetadataExtractor):
    """PHI-specific metadata extractor.

    Args:
        reader: The :class:`PhiReader` to describe.
    """

    def __init__(self, reader: "PhiReader", skip_event_aggregate: bool = False):
        """Initialise the extractor.

        Args:
            reader: The :class:`PhiReader` to describe.
            skip_event_aggregate: If True, answer from the acquisition
                header and the block chain alone -- ``n_spectra`` and
                ``total_peaks`` come back as 0 with
                ``n_spectra_counted=False``, and ``peak_counts_per_pixel``
                as None. Used by metadata-only callers (``preview_msi``)
                to avoid the pass over every ion event that
                :meth:`PhiReader.get_peak_counts_per_pixel` performs, which
                is linear in file size and so made a preview cost what a
                conversion costs (issue #240). The same role
                ``skip_total_peaks`` plays for Bruker.
        """
        super().__init__(reader)
        self._reader = reader
        self._skip_event_aggregate = bool(skip_event_aggregate)

    def _extract_essential_impl(self) -> EssentialMetadata:
        """Extract metadata needed to drive conversion.

        The header knows the tile geometry, so ``dimensions`` and
        ``coordinate_bounds`` are free. It does **not** know which of those
        pixels recorded an ion: PHI stores a stream of events, not a list
        of spectra, so the only way to count occupied pixels is to decode
        every event. When ``skip_event_aggregate`` is set that count is
        reported as absent rather than guessed at -- see
        :attr:`EssentialMetadata.n_spectra_counted`. Reporting the raster
        size instead would have made ``n_spectra`` mean "positions the
        raster covers" for PHI and "spectra present" for every other
        format.
        """
        reader = self._reader
        dimensions = reader.dimensions
        n_x, n_y, _ = dimensions

        if self._skip_event_aggregate:
            peak_counts = None
            total_peaks = 0
            n_spectra = 0
        else:
            peak_counts = reader.get_peak_counts_per_pixel()
            assert peak_counts is not None
            total_peaks = int(peak_counts.sum())
            n_spectra = int(np.count_nonzero(peak_counts))

        pixel_size = None
        size_um = reader.pixel_size_um
        if size_um:
            pixel_size = (float(size_um), float(size_um))

        return EssentialMetadata(
            dimensions=dimensions,
            coordinate_bounds=(0.0, float(n_x - 1), 0.0, float(n_y - 1)),
            mass_range=reader.mass_axis.mass_range,
            pixel_size=pixel_size,
            n_spectra=n_spectra,
            n_spectra_counted=not self._skip_event_aggregate,
            total_peaks=total_peaks,
            estimated_memory_gb=(total_peaks * 2 * 8) / (1024**3),
            source_path=str(reader.data_path),
            coordinate_offsets=(0, 0, 0),
            # A sparse histogram on the detector's regular time-channel grid,
            # not a peak-picked list.
            spectrum_type="profile spectrum",
            peak_counts_per_pixel=peak_counts,
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        """Extract full metadata including vendor-specific detail."""
        return ComprehensiveMetadata(
            essential=self.get_essential(),
            format_specific=self._extract_format_specific(),
            acquisition_params=self._extract_acquisition_params(),
            instrument_info=self._extract_instrument_info(),
            raw_metadata=self._extract_raw_metadata(),
        )

    def _extract_format_specific(self) -> Dict[str, Any]:
        reader = self._reader
        header = reader.header
        index = reader.block_index
        axis = reader.mass_axis
        return {
            "format": "PHI SmartSoft-TOF raw",
            "raw_file_format_version": header.entries.get("RawFileFormatVersion"),
            "header_size": header.header_size,
            "image_pixels": header.image_pixels,
            "n_mass_channels": len(axis),
            "bin_size_ns": header.bin_size_ns,
            "flight_time_range_us": (
                header.start_flight_time_us,
                header.stop_flight_time_us,
            ),
            "n_event_blocks": len(index.data_spans),
            "n_records": index.n_records,
            "frames_in_stream": index.n_frames,
            # Zarr group members must be named, so the block ids are keyed
            # as strings rather than the ints they are in the file.
            "block_id_counts": {
                f"block_{block_id}": int(count)
                for block_id, count in sorted(index.block_counts.items())
            },
            "block_chain_complete": index.clean,
            "mosaic_tiles": index.tile_grid,
            "declared_tiles": header.declared_tiles,
            "pixel_size_source": reader.pixel_size_source,
            "raster_size_um": header.raster_size_um,
            "raster_size_calibration": header.raster_size_calibration,
        }

    def _extract_acquisition_params(self) -> Dict[str, Any]:
        header = self._reader.header
        acq = header.sections.get("Acq Base", {})
        return {
            "polarity": header.polarity,
            "technique": header.entries.get("Technique"),
            "acquisition_type": header.entries.get("AcquisitionType"),
            "n_frames": header.n_frames,
            "msms_active": header.msms_active,
            "primary_species": header.entries.get("AcqPrimSpecies"),
            "primary_current_na": header.entries.get("AcqPrimCurrent(nA)"),
            "pulse_width_ns": header.entries.get("AcqPulseWidth"),
            "raster_pattern": acq.get("Raster Pattern"),
            "raster_resolution": acq.get("Raster Resolution"),
            "mass_range_amu": (header.start_mz, header.stop_mz),
            "acquisition_date": header.entries.get("AcqFileDate"),
        }

    def _extract_instrument_info(self) -> Dict[str, Any]:
        header = self._reader.header
        return {
            "vendor": "Physical Electronics (PHI)",
            "software_version": header.entries.get("SoftwareVersion"),
            "platform": header.entries.get("Platform"),
            "primary_gun": header.entries.get("AcqPrimGun"),
            "emitter": header.entries.get("LmigEmitter"),
            "detector_scan_mode": header.entries.get("DetectorScanMode"),
        }

    def _extract_raw_metadata(self) -> Dict[str, Any]:
        reader = self._reader
        header = reader.header
        axis = reader.mass_axis
        calibration: Dict[str, Any] = {
            "source": reader.calibration_source,
            "mass_slope_used": axis.slope,
            "mass_offset_used": axis.offset,
            "header_mass_slope": header.mass_slope,
            "header_mass_offset": header.mass_offset,
            "header_calibrated_flag": header.entries.get("Calibrated"),
            # JSON strings: AnnData/zarr cannot round-trip a list of dicts.
            "acquisition_calibrants": json.dumps(
                [
                    {
                        "measured_mz": c.measured_mz,
                        "species": c.species,
                        "theoretical_mz": c.theoretical_mz,
                    }
                    for c in header.calibrants
                ]
            ),
        }
        appended = reader.block_index.appended
        if appended:
            calibration["appended_blocks"] = json.dumps(appended)
        # The vendor's own key names are not valid Zarr group members --
        # 'Mass/Time' contains a forward slash -- so the header is preserved
        # verbatim as JSON rather than being mangled into a group hierarchy.
        return {
            "header_entries": json.dumps(dict(header.entries)),
            "header_sections": json.dumps(
                {k: dict(v) for k, v in header.sections.items()}
            ),
            "calibration": calibration,
        }
