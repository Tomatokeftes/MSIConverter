# msiconvert/metadata/extractors/bruker_extractor.py
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...core.base_extractor import MetadataExtractor
from ..types import ComprehensiveMetadata, EssentialMetadata

logger = logging.getLogger(__name__)


class BrukerMetadataExtractor(MetadataExtractor):
    """Bruker-specific metadata extractor with optimized single-query extraction."""

    def __init__(self, conn: sqlite3.Connection, data_path: Path):
        """
        Initialize Bruker metadata extractor.

        Args:
            conn: Active SQLite database connection
            data_path: Path to the Bruker .d directory
        """
        super().__init__(conn)
        self.conn = conn
        self.data_path = data_path

    def _extract_essential_impl(self) -> EssentialMetadata:
        """Extract essential metadata with proper coordinate normalization."""
        cursor = self.conn.cursor()

        # Get imaging area bounds from GlobalMetadata for coordinate normalization
        imaging_bounds_query = """
        SELECT Key, Value FROM GlobalMetadata
        WHERE Key IN ('ImagingAreaMinXIndexPos', 'ImagingAreaMaxXIndexPos',
                      'ImagingAreaMinYIndexPos', 'ImagingAreaMaxYIndexPos',
                      'MzAcqRangeLower', 'MzAcqRangeUpper')
        """

        cursor.execute(imaging_bounds_query)
        bounds_data = {row[0]: float(row[1]) for row in cursor.fetchall()}

        # Get beam scan sizes from laser info
        laser_query = """
        SELECT BeamScanSizeX, BeamScanSizeY, SpotSize
        FROM MaldiFrameLaserInfo
        LIMIT 1
        """

        cursor.execute(laser_query)
        laser_result = cursor.fetchone()

        # Get coordinate bounds and frame count from frame info
        frame_query = """
        SELECT
            MIN(XIndexPos), MAX(XIndexPos),
            MIN(YIndexPos), MAX(YIndexPos),
            COUNT(*) as frame_count
        FROM MaldiFrameInfo
        """

        cursor.execute(frame_query)
        frame_result = cursor.fetchone()

        try:
            if not frame_result:
                raise ValueError("No data found in MaldiFrameInfo table")

            min_x_raw, max_x_raw, min_y_raw, max_y_raw, frame_count = frame_result

            # Extract imaging area bounds for normalization
            imaging_min_x = bounds_data.get("ImagingAreaMinXIndexPos", min_x_raw or 0)
            imaging_max_x = bounds_data.get("ImagingAreaMaxXIndexPos", max_x_raw or 0)
            imaging_min_y = bounds_data.get("ImagingAreaMinYIndexPos", min_y_raw or 0)
            imaging_max_y = bounds_data.get("ImagingAreaMaxYIndexPos", max_y_raw or 0)

            # Store imaging area offsets for coordinate normalization
            imaging_area_offsets = (int(imaging_min_x), int(imaging_min_y), 0)
            
            # Normalize coordinates to start from 0
            min_x = 0.0  # Normalized coordinates always start from 0
            max_x = float(imaging_max_x - imaging_min_x)
            min_y = 0.0
            max_y = float(imaging_max_y - imaging_min_y)

            # Extract beam sizes for pixel size
            beam_x, beam_y, spot_size = (
                laser_result if laser_result else (None, None, None)
            )

            # Mass range from GlobalMetadata
            min_mass = bounds_data.get("MzAcqRangeLower")
            max_mass = bounds_data.get("MzAcqRangeUpper")

            # --- Error Tracing for Mass Range ---
            missing_mass_keys = []
            if min_mass is None:
                missing_mass_keys.append("MzAcqRangeLower")
            if max_mass is None:
                missing_mass_keys.append("MzAcqRangeUpper")

            if missing_mass_keys:
                error_msg = f"Missing critical mass range bounds in GlobalMetadata: {', '.join(missing_mass_keys)}. Cannot establish mass range."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Calculate dimensions and bounds
            dimensions = self._calculate_dimensions_from_coords(
                min_x, max_x, min_y, max_y
            )
            coordinate_bounds = (
                float(min_x) if min_x is not None else 0.0,
                float(max_x) if max_x is not None else 0.0,
                float(min_y) if min_y is not None else 0.0,
                float(max_y) if max_y is not None else 0.0,
            )

            # Extract pixel size
            pixel_size = None
            if beam_x is not None and beam_y is not None:
                pixel_size = (float(beam_x), float(beam_y))

            # Mass range
            mass_range = (
                float(min_mass) if min_mass is not None else 0.0,
                float(max_mass) if max_mass is not None else 1000.0,
            )

            # Frame count
            n_spectra = int(frame_count) if frame_count is not None else 0

            # Memory estimation
            estimated_memory = self._estimate_memory_from_frames(n_spectra)

            return EssentialMetadata(
                dimensions=dimensions,
                coordinate_bounds=coordinate_bounds,
                mass_range=mass_range,
                pixel_size=pixel_size,
                n_spectra=n_spectra,
                estimated_memory_gb=estimated_memory,
                source_path=str(self.data_path),
                coordinate_offsets=imaging_area_offsets,
            )

        except sqlite3.OperationalError as e:
            logger.error(f"SQL error extracting essential metadata: {e}")
            raise ValueError(
                f"Failed to extract essential metadata from Bruker database: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error extracting essential metadata: {e}")
            raise

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        """Extract comprehensive metadata with additional database queries."""
        essential = self.get_essential()

        return ComprehensiveMetadata(
            essential=essential,
            format_specific=self._extract_bruker_specific(),
            acquisition_params=self._extract_acquisition_params(),
            instrument_info=self._extract_instrument_info(),
            raw_metadata=self._extract_global_metadata(),
        )

    def _calculate_dimensions_from_coords(
        self,
        min_x: Optional[float],
        max_x: Optional[float],
        min_y: Optional[float],
        max_y: Optional[float],
    ) -> Tuple[int, int, int]:
        """Calculate dataset dimensions from coordinate bounds."""
        if any(coord is None for coord in [min_x, max_x, min_y, max_y]):
            return (0, 0, 1)  # Default for problematic data

        # Bruker coordinates are typically in position units
        # Calculate grid dimensions assuming integer grid positions
        x_range = int(max_x - min_x) + 1 if max_x > min_x else 1
        y_range = int(max_y - min_y) + 1 if max_y > min_y else 1

        return (max(1, x_range), max(1, y_range), 1)  # Assume 2D data (z=1)

    def _estimate_memory_from_frames(self, frame_count: int) -> float:
        """Estimate memory usage from frame count."""
        if frame_count <= 0:
            return 0.0

        # Rough estimate for Bruker data:
        # - Average ~2000 peaks per frame
        # - 8 bytes per float64 value
        # - mz + intensity arrays
        avg_peaks_per_frame = 2000
        bytes_per_value = 8
        estimated_bytes = frame_count * avg_peaks_per_frame * 2 * bytes_per_value

        return estimated_bytes / (1024**3)  # Convert to GB

    def _extract_bruker_specific(self) -> Dict[str, Any]:
        """Extract Bruker format-specific metadata."""
        format_specific = {
            "bruker_format": "bruker_tdf" if self._is_tdf_format() else "bruker_tsf",
            "data_format": "bruker_tdf" if self._is_tdf_format() else "bruker_tsf",
            "data_path": str(self.data_path),
            "database_path": str(self.data_path / "analysis.tsf"),
            "is_maldi": self._is_maldi_dataset(),
        }

        # Add file type detection
        if (self.data_path / "analysis.tdf").exists():
            format_specific["binary_file"] = str(self.data_path / "analysis.tdf")
        elif (self.data_path / "analysis.tsf").exists():
            format_specific["binary_file"] = str(self.data_path / "analysis.tsf")

        return format_specific

    def _extract_acquisition_params(self) -> Dict[str, Any]:
        """Extract acquisition parameters from database."""
        params = {}
        cursor = self.conn.cursor()

        # Extract laser parameters if available
        try:
            cursor.execute(
                """
                SELECT DISTINCT LaserPower, LaserFrequency, BeamScanSizeX, BeamScanSizeY, SpotSize
                FROM MaldiFrameLaserInfo
                LIMIT 1
            """
            )
            result = cursor.fetchone()

            if result:
                laser_power, laser_freq, beam_x, beam_y, spot_size = result
                if laser_power is not None:
                    params["laser_power"] = laser_power
                if laser_freq is not None:
                    params["laser_frequency"] = laser_freq
                if beam_x is not None:
                    params["beam_scan_size_x"] = beam_x
                    params[
                        "BeamScanSizeX"
                    ] = beam_x  # Add both formats for compatibility
                if beam_y is not None:
                    params["beam_scan_size_y"] = beam_y
                    params[
                        "BeamScanSizeY"
                    ] = beam_y  # Add both formats for compatibility
                if spot_size is not None:
                    params["laser_spot_size"] = spot_size

        except sqlite3.OperationalError:
            logger.debug("Could not extract laser parameters")

        # Extract timing parameters
        try:
            cursor.execute(
                "SELECT Value FROM GlobalMetadata WHERE Key = 'AcquisitionDateTime'"
            )
            result = cursor.fetchone()
            if result:
                params["acquisition_datetime"] = result[0]
        except sqlite3.OperationalError:
            pass

        return params

    def _extract_instrument_info(self) -> Dict[str, Any]:
        """Extract instrument information from global metadata."""
        instrument = {}
        cursor = self.conn.cursor()

        # Common instrument metadata keys
        instrument_keys = [
            ("InstrumentName", "instrument_name"),
            ("InstrumentSerialNumber", "instrument_serial_number"),
            ("InstrumentModel", "instrument_model"),
            ("SoftwareVersion", "software_version"),
            ("MzCalibrationMode", "mz_calibration_mode"),
        ]

        try:
            for db_key, result_key in instrument_keys:
                cursor.execute(
                    "SELECT Value FROM GlobalMetadata WHERE Key = ?", (db_key,)
                )
                result = cursor.fetchone()
                if result:
                    instrument[result_key] = result[0]
        except sqlite3.OperationalError:
            logger.debug("Could not extract instrument metadata")

        return instrument

    def _extract_global_metadata(self) -> Dict[str, Any]:
        """Extract all global metadata from database."""
        raw_metadata = {}
        cursor = self.conn.cursor()

        try:
            cursor.execute("SELECT Key, Value FROM GlobalMetadata")
            global_metadata = {}
            for key, value in cursor.fetchall():
                global_metadata[key] = value
            raw_metadata["global_metadata"] = global_metadata

            # Extract frame info for tests that expect it
            cursor.execute(
                "SELECT Id, SpotXPos, SpotYPos, BeamScanSizeX, BeamScanSizeY FROM MaldiFrameLaserInfo"
            )
            frame_info = []
            for row in cursor.fetchall():
                frame_info.append(
                    {
                        "id": row[0],
                        "x_pos": row[1],
                        "y_pos": row[2],
                        "beam_x": row[3],
                        "beam_y": row[4],
                    }
                )
            raw_metadata["frame_info"] = frame_info

        except sqlite3.OperationalError:
            logger.debug("GlobalMetadata table not found or accessible")

        return raw_metadata

    def _is_tdf_format(self) -> bool:
        """Check if this is TDF format (vs TSF)."""
        return (self.data_path / "analysis.tdf").exists()

    def _is_maldi_dataset(self) -> bool:
        """Check if this is a MALDI dataset by checking for laser info."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM MaldiFrameLaserInfo")
            result = cursor.fetchone()
            return result and result[0] > 0
        except sqlite3.OperationalError:
            return False
