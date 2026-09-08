# thyra/readers/bruker/rapiflex/rapiflex_reader.py
"""Rapiflex reader for Bruker MALDI-TOF data.

This module provides a pure Python reader for Bruker Rapiflex data format
used by MALDI-TOF instruments (rapifleX, autoflex, microflex, ultraflex).
No SDK is required - the format is read directly from binary files.

File structure expected:
    sample_folder/
        sample.dat              <- Main spectral data (float32 arrays)
        sample.mis              <- Method/alignment XML (optional)
        sample_info.txt         <- Acquisition metadata
        sample_poslog.txt       <- X/Y coordinates
        *.tif                   <- Optical images (handled separately)
        sample.d/               <- MCF containers (not needed)
"""

import logging
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element  # nosec B405 - type hint only

try:
    # Prefer defusedxml for secure parsing of instrument XML files.
    import defusedxml.ElementTree as ET
except ImportError:
    # Fallback to the standard library so the reader still works when
    # defusedxml is unavailable (mirrors imzml_extractor._get_xml_parser).
    import xml.etree.ElementTree as ET  # nosec B405

    logging.getLogger(__name__).warning(
        "defusedxml not available, using xml.etree.ElementTree"
    )

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ....core.base_extractor import MetadataExtractor
from ....core.registry import register_reader
from ....metadata.types import ComprehensiveMetadata, EssentialMetadata
from ..base_bruker_reader import BrukerBaseMSIReader

logger = logging.getLogger(__name__)


class RapiflexMetadataExtractor(MetadataExtractor):
    """Metadata extractor for Rapiflex data."""

    def __init__(self, reader: "RapiflexReader"):
        """Initialize metadata extractor.

        Args:
            reader: RapiflexReader instance
        """
        super().__init__(reader)
        self._reader = reader

    def _extract_essential_impl(self) -> EssentialMetadata:
        """Extract essential metadata for processing."""
        info = self._reader.info_metadata
        header = self._reader._header

        # Get dimensions from header (raster grid size)
        width = header.get("raster_width", 0)
        height = header.get("raster_height", 0)
        dimensions = (width, height, 1)

        # Coordinate bounds are 0-based (normalized)
        coord_bounds = (0.0, float(width - 1), 0.0, float(height - 1))

        # Where on the slide this raster starts, in raster units. The .dat
        # header carries it and it used to be discarded for a hard-coded
        # (0, 0, 0), so no Rapiflex store could be placed against its
        # optical image the way a solariX or timsTOF one can. The converter
        # turns these into stage_offset_um by multiplying by the pixel
        # size, so this is the only thing the reader owes it (issue #237).
        coord_offsets = (
            int(header.get("first_raster_x", 0)),
            int(header.get("first_raster_y", 0)),
            0,
        )

        # The raster step, or nothing. A missing declaration used to become
        # 20 um, which convert.py then recorded as
        # `pixel_size_source: automatic` with `detection_successful: True`
        # -- a fabricated pitch the store called a measurement, while
        # `format_specific.raster_step_um` sat empty. Every other reader
        # returns None here and the CLI refuses with the actionable
        # "use --pixel-size" hint; a refusal beats an invented number
        # (issue #236).
        pixel_x = info.get("raster_x")
        pixel_y = info.get("raster_y")
        if pixel_x is None or pixel_y is None:
            logger.warning(
                "%s declares no raster step (looked for 'Raster' in "
                "sample_info.txt and the .mis), so the pixel size is "
                "unknown. Pass --pixel-size to supply it.",
                self._reader.data_path.name,
            )
            pixel_size = None
        else:
            pixel_size = (float(pixel_x), float(pixel_y))

        # Mass range
        mass_start = info.get("Mass Start", 0.0)
        mass_end = info.get("Mass End", 0.0)

        # Estimate memory (n_spectra * n_datapoints * 4 bytes for float32)
        n_spectra = self._reader.n_spectra
        n_datapoints = self._reader.n_datapoints
        estimated_memory_gb = (n_spectra * n_datapoints * 4) / (1024**3)

        # Total peaks estimate (for profile data, same as total data points)
        total_peaks = n_spectra * n_datapoints

        return EssentialMetadata(
            dimensions=dimensions,
            coordinate_bounds=coord_bounds,
            mass_range=(mass_start, mass_end),
            pixel_size=pixel_size,
            n_spectra=n_spectra,
            total_peaks=total_peaks,
            estimated_memory_gb=estimated_memory_gb,
            source_path=str(self._reader.data_path),
            coordinate_offsets=coord_offsets,
            spectrum_type="profile spectrum",  # FlexImaging produces profile data
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        """Extract comprehensive metadata."""
        essential = self.get_essential()
        info = self._reader.info_metadata
        mis = self._reader.mis_metadata

        format_specific = {
            "format": "Rapiflex",
            "acquisition_mode": info.get("Acquisition Mode", "UNKNOWN"),
            "flexImaging_version": info.get("flexImaging Version", ""),
            "flexControl_version": info.get("flexControl Version", ""),
            "teaching_points": mis.get("teaching_points", []),
            "areas": mis.get("areas", []),
            "raster_step_um": list(
                mis.get("raster") or []
            ),  # [x, y] step in micrometers
            "optical_image": mis.get("ImageFile", ""),
            "original_optical_image": mis.get("OriginalImage", ""),
            "base_geometry": mis.get("BaseGeometry", ""),
        }

        acquisition_params = {
            "shots_per_spot": info.get("Number of Shots", 0),
            "laser_power": info.get("Laser Power", 0),
            "detector_gain": info.get("Detector Gain", 0.0),
            "sample_rate": info.get("Sample Rate", 0.0),
            "method": info.get("Method", ""),
            "start_time": info.get("Start Time", ""),
            "end_time": info.get("End Time", ""),
        }

        instrument_info = {
            "instrument_type": "MALDI-TOF",
            "serial_number": info.get("Instrument Serial Number", ""),
            "manufacturer": "Bruker",
        }

        raw_metadata = {
            "info_metadata": info,
            "mis_metadata": mis,
            "header": self._reader._header,
        }

        return ComprehensiveMetadata(
            essential=essential,
            format_specific=format_specific,
            acquisition_params=acquisition_params,
            instrument_info=instrument_info,
            raw_metadata=raw_metadata,
        )


@register_reader("rapiflex")
class RapiflexReader(BrukerBaseMSIReader):
    """Reader for Bruker Rapiflex MALDI-TOF data.

    This reader handles Rapiflex data format without requiring any SDK.
    The format stores spectral data in a binary .dat file with coordinates
    in a separate text file.

    Features:
        - Pure Python implementation (no SDK required)
        - Lazy loading of spectral data
        - Efficient memory usage via generator-based iteration
        - Support for rectangular raster grids with sparse data
    """

    def __init__(
        self,
        data_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        """Initialize the Rapiflex reader.

        Args:
            data_path: Path to the Rapiflex data folder (containing .dat file)
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(data_path, **kwargs)
        self.progress_callback = progress_callback

        # Find and validate required files
        self._dat_path: Optional[Path] = None
        self._info_path: Optional[Path] = None
        self._poslog_path: Optional[Path] = None
        self._mis_path: Optional[Path] = None
        self._find_data_files()

        # Parse metadata
        self._info_metadata: Dict[str, Any] = {}
        self._mis_metadata: Dict[str, Any] = {}
        self._parse_metadata()

        # Parse coordinates from poslog
        self._positions: List[Dict[str, Any]] = []
        self._parse_positions()

        # Read .dat file header and offset table
        self._header: Dict[str, int] = {}
        self._offsets: NDArray[np.uint32] = np.array([], dtype=np.uint32)
        self._parse_dat_header()

        # Compute valid spectra indices
        # Offsets must be >= data_start (after header + offset table)
        # Some files have small non-zero offsets that point into the offset table itself
        header_size = 48
        offset_table_size = (
            self._header["raster_width"] * self._header["raster_height"] * 4
        )
        data_start = header_size + offset_table_size
        self._valid_indices: NDArray[np.int64] = np.where(self._offsets >= data_start)[
            0
        ]

        # Compute m/z axis
        self._mz_axis: Optional[NDArray[np.float64]] = None

        # Track close state
        self._closed: bool = False

        n_raster = self._header["raster_width"] * self._header["raster_height"]
        logger.info(
            f"Initialized RapiflexReader: {self.n_spectra} valid spectra "
            f"out of {n_raster} raster positions"
        )

    def _find_data_files(self) -> None:
        """Find and validate required data files in the folder."""
        folder = self.data_path

        if not folder.is_dir():
            raise ValueError(f"Rapiflex path must be a directory: {folder}")

        # Find .dat file
        dat_files = list(folder.glob("*.dat"))
        if not dat_files:
            raise ValueError(f"No .dat file found in {folder}")
        if len(dat_files) > 1:
            logger.warning(f"Multiple .dat files found, using first: {dat_files[0]}")
        self._dat_path = dat_files[0]

        # Find _info.txt file
        info_files = list(folder.glob("*_info.txt"))
        if info_files:
            self._info_path = info_files[0]
        else:
            raise ValueError(f"No *_info.txt file found in {folder}")

        # Find _poslog.txt file
        poslog_files = list(folder.glob("*_poslog.txt"))
        if poslog_files:
            self._poslog_path = poslog_files[0]
        else:
            raise ValueError(f"No *_poslog.txt file found in {folder}")

        # Find .mis file (optional)
        mis_files = list(folder.glob("*.mis"))
        if mis_files:
            self._mis_path = mis_files[0]

        logger.debug(
            f"Found files - dat: {self._dat_path}, info: {self._info_path}, "
            f"poslog: {self._poslog_path}, mis: {self._mis_path}"
        )

    def _parse_metadata(self) -> None:
        """Parse metadata from _info.txt and .mis files."""
        # Parse _info.txt
        if self._info_path and self._info_path.exists():
            self._info_metadata = self._parse_info_file(self._info_path)

        # Parse .mis file (optional)
        if self._mis_path and self._mis_path.exists():
            self._mis_metadata = self._parse_mis_file(self._mis_path)

    def _parse_info_file(self, path: Path) -> Dict[str, Any]:
        """Parse the _info.txt metadata file.

        Args:
            path: Path to the _info.txt file

        Returns:
            Dictionary of metadata key-value pairs
        """
        metadata: Dict[str, Any] = {}

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse key: value format
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Try to convert to appropriate type
                    try:
                        if "." in value:
                            metadata[key] = float(value)
                        else:
                            metadata[key] = int(value)
                    except ValueError:
                        metadata[key] = value

        # Parse raster from "20,20" format
        if "Raster" in metadata:
            raster_str = str(metadata["Raster"])
            if "," in raster_str:
                parts = raster_str.split(",")
                metadata["raster_x"] = float(parts[0])
                metadata["raster_y"] = float(parts[1])

        return metadata

    def _parse_mis_file(self, path: Path) -> Dict[str, Any]:
        """Parse the .mis XML file for method and alignment info.

        Args:
            path: Path to the .mis file

        Returns:
            Dictionary of parsed metadata
        """
        metadata: Dict[str, Any] = {}

        try:
            tree = ET.parse(
                path
            )  # nosec B314 - defusedxml preferred; trusted local instrument file
            root = tree.getroot()

            self._extract_basic_elements(root, metadata)
            self._extract_teaching_points(root, metadata)
            self._extract_raster_info(root, metadata)
            self._extract_areas(root, metadata)

        except ET.ParseError as e:
            logger.warning(f"Failed to parse .mis file: {e}")

        return metadata

    def _extract_basic_elements(
        self, root: "Element", metadata: Dict[str, Any]
    ) -> None:
        """Extract basic text elements from .mis XML."""
        for elem_name in ["Method", "ImageFile", "OriginalImage", "BaseGeometry"]:
            elem = root.find(f".//{elem_name}")
            if elem is not None and elem.text:
                metadata[elem_name] = elem.text

    def _extract_teaching_points(
        self, root: "Element", metadata: Dict[str, Any]
    ) -> None:
        """Extract teaching point calibration data from .mis XML."""
        teaching_points = []
        for tp in root.findall(".//TeachPoint"):
            if tp.text and ";" in tp.text:
                img_coords, stage_coords = tp.text.split(";")
                img_x, img_y = map(int, img_coords.split(","))
                stage_x, stage_y = map(int, stage_coords.split(","))
                # Use lists instead of tuples for Zarr serialization
                teaching_points.append(
                    {"image": [img_x, img_y], "stage": [stage_x, stage_y]}
                )
        if teaching_points:
            metadata["teaching_points"] = teaching_points

    def _extract_raster_info(self, root: "Element", metadata: Dict[str, Any]) -> None:
        """Extract raster dimensions from .mis XML."""
        raster_elem = root.find(".//Raster")
        if raster_elem is not None and raster_elem.text:
            parts = raster_elem.text.split(",")
            if len(parts) == 2:
                # Use list instead of tuple for Zarr serialization
                metadata["raster"] = [int(parts[0]), int(parts[1])]

    def _extract_areas(self, root: "Element", metadata: Dict[str, Any]) -> None:
        """Extract Area definitions from .mis XML.

        Areas define the image pixel coordinates for each acquisition region.
        Areas may be rectangular (Type=0, 2 points) or polygon (Type=3, N
        points). In both cases the bounding box of all points is stored.

        Args:
            root: XML root element
            metadata: Dictionary to update with area info
        """
        areas = []
        for area_elem in root.findall(".//Area"):
            area_name = area_elem.get("Name", "")
            points = area_elem.findall("Point")
            if len(points) >= 2:
                try:
                    # Parse ALL point coordinates to get bounding box
                    # Works for both rectangular (2 points) and polygon (N points)
                    all_x = []
                    all_y = []
                    for pt in points:
                        pt_text = pt.text or ""
                        parts = pt_text.split(",")
                        all_x.append(int(parts[0]))
                        all_y.append(int(parts[1]))

                    areas.append(
                        {
                            "name": area_name,
                            "p1": [min(all_x), min(all_y)],
                            "p2": [max(all_x), max(all_y)],
                        }
                    )
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse Area '{area_name}': {e}")
                    continue

        if areas:
            metadata["areas"] = areas
            logger.debug(f"Parsed {len(areas)} area definitions from .mis")

    def _parse_positions(self) -> None:
        """Parse coordinate information from _poslog.txt file."""
        if not self._poslog_path:
            raise ValueError("Position log file not found")

        self._positions = []

        with open(self._poslog_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                pos_id = parts[2]
                if pos_id == "__":
                    continue

                # Parse position ID format: R{region}X{x}Y{y}
                match = re.match(r"R(\d+)X(\d+)Y(\d+)", pos_id)
                if match:
                    self._positions.append(
                        {
                            "region": int(match.group(1)),
                            "raster_x": int(match.group(2)),
                            "raster_y": int(match.group(3)),
                            "phys_x": float(parts[3]),
                            "phys_y": float(parts[4]),
                            "phys_z": float(parts[5]) if len(parts) > 5 else 0.0,
                        }
                    )

        logger.debug(f"Parsed {len(self._positions)} positions from poslog")

    def _parse_dat_header(self) -> None:
        """Parse the .dat file header and offset table."""
        if not self._dat_path:
            raise ValueError("Data file not found")

        with open(self._dat_path, "rb") as f:
            # Read fixed 48-byte header
            header_data = f.read(48)
            if len(header_data) < 48:
                raise ValueError("Invalid .dat file: header too short")

            # Parse header fields
            vals = struct.unpack("<12I", header_data)
            self._header = {
                "header_size": vals[0],
                "unknown1": vals[1],
                "first_raster_x": vals[2],
                "first_raster_y": vals[3],
                "raster_width": vals[4],
                "raster_height": vals[5],
                "n_datapoints": vals[6],
            }

            # Read offset table
            # The offset table has one entry per raster position (width * height)
            # NOT the number of acquired spots (which is what poslog has)
            raster_width = self._header["raster_width"]
            raster_height = self._header["raster_height"]
            n_raster_positions = raster_width * raster_height

            if n_raster_positions > 0:
                self._offsets = np.frombuffer(
                    f.read(n_raster_positions * 4), dtype=np.uint32
                ).copy()
            else:
                raise ValueError("Invalid raster dimensions in header")

        logger.debug(
            f"Parsed .dat header: {self._header['n_datapoints']} datapoints, "
            f"{len(self._offsets)} offset entries"
        )

    @property
    def info_metadata(self) -> Dict[str, Any]:
        """Get parsed _info.txt metadata."""
        return self._info_metadata

    @property
    def mis_metadata(self) -> Dict[str, Any]:
        """Get parsed .mis metadata."""
        return self._mis_metadata

    @property
    def n_spectra(self) -> int:
        """Get number of valid spectra (with data)."""
        return len(self._valid_indices)

    @property
    def n_datapoints(self) -> int:
        """Get number of data points per spectrum."""
        # Prefer header value, fall back to info metadata
        return self._header.get(
            "n_datapoints", self._info_metadata.get("DataPoints", 0)
        )

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Get mass range (start, end) in Da."""
        return (
            self._info_metadata.get("Mass Start", 0.0),
            self._info_metadata.get("Mass End", 0.0),
        )

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create format-specific metadata extractor."""
        return RapiflexMetadataExtractor(self)

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Return the common mass axis for all spectra.

        For FlexImaging data, the m/z axis is uniform (linear spacing)
        from mass_start to mass_end.

        Returns:
            Array of m/z values
        """
        if self._mz_axis is None:
            mass_start, mass_end = self.mass_range
            n_points = self.n_datapoints

            if n_points == 0:
                raise ValueError("Cannot create mass axis: n_datapoints is 0")
            if mass_start >= mass_end:
                raise ValueError(f"Invalid mass range: {mass_start} to {mass_end}")

            self._mz_axis = np.linspace(mass_start, mass_end, n_points)

        return self._mz_axis

    def _get_normalized_coordinates(self, raster_idx: int) -> Tuple[int, int, int]:
        """Get 0-based normalized coordinates for a raster grid index.

        The raster grid is stored row-major, so:
        - raster_idx = (y - first_y) * width + (x - first_x)
        - x = first_x + (raster_idx % width)
        - y = first_y + (raster_idx // width)

        We normalize to 0-based coordinates by subtracting first_x/first_y.

        Args:
            raster_idx: Index into the raster grid (offset table)

        Returns:
            Tuple of (x, y, z) coordinates, 0-based
        """
        raster_width = self._header.get("raster_width", 1)

        # Calculate 0-based coordinates from raster index
        x = raster_idx % raster_width
        y = raster_idx // raster_width

        return (x, y, 0)

    def _read_spectrum(self, idx: int) -> NDArray[np.float32]:
        """Read spectrum intensities at given index.

        Args:
            idx: Index into the positions/offsets list

        Returns:
            Array of intensity values (float32)
        """
        if idx >= len(self._offsets):
            raise IndexError(f"Spectrum index out of range: {idx}")

        offset = self._offsets[idx]
        if offset == 0:
            # No data at this position
            return np.zeros(self.n_datapoints, dtype=np.float32)

        if self._dat_path is None:
            raise RuntimeError("dat_path not set")
        with open(self._dat_path, "rb") as f:
            f.seek(int(offset))
            data = f.read(self.n_datapoints * 4)
            return np.frombuffer(data, dtype=np.float32).copy()

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Iterate through spectra with coordinates.

        Only yields spectra that have data (offset > 0).

        Args:
            batch_size: Ignored, maintained for API compatibility

        Yields:
            Tuple containing:
                - Coordinates (x, y, z) using 0-based indexing
                - m/z values array
                - Intensity values array
        """
        if self._closed:
            raise RuntimeError("Reader has been closed")

        # Get m/z axis once and cache as float64 to avoid repeated conversions
        mz_axis = self.get_common_mass_axis()
        if mz_axis.dtype != np.float64:
            mz_axis = mz_axis.astype(np.float64)

        # Use tqdm for progress bar
        pbar = tqdm(
            self._valid_indices,
            desc="Reading Rapiflex spectra",
            unit=" spectra",
            dynamic_ncols=True,
        )

        try:
            for idx in pbar:
                coords = self._get_normalized_coordinates(idx)
                intensities = self._read_spectrum(idx).astype(np.float64)

                # Apply intensity threshold filtering if configured
                filtered_mzs, filtered_intensities = self._apply_intensity_filter(
                    mz_axis, intensities
                )

                # Only yield if we have data after filtering
                if filtered_mzs.size > 0 and filtered_intensities.size > 0:
                    yield (
                        coords,
                        filtered_mzs,
                        filtered_intensities,
                    )

                if self.progress_callback:
                    self.progress_callback(idx, len(self._valid_indices))

        finally:
            pbar.close()

    def get_peak_counts_per_pixel(self) -> Optional[NDArray[np.int32]]:
        """Get per-pixel peak counts for CSR indptr construction.

        For Rapiflex profile data, all valid pixels have the same number
        of data points. Empty pixels (no spectrum) have 0 peaks.

        Returns:
            Array of size n_pixels where arr[pixel_idx] = peak_count.
            pixel_idx = z * (n_x * n_y) + y * n_x + x
        """
        # Get dimensions from header
        n_x = self._header.get("raster_width", 0)
        n_y = self._header.get("raster_height", 0)
        n_z = 1  # Rapiflex is 2D
        n_pixels = n_x * n_y * n_z

        if n_pixels == 0:
            return None

        # Create output array - all zeros initially
        peak_counts = np.zeros(n_pixels, dtype=np.int32)

        # For profile data, all valid pixels have n_datapoints
        n_datapoints = self.n_datapoints

        # Set peak count for valid indices only
        for idx in self._valid_indices:
            if 0 <= idx < n_pixels:
                peak_counts[idx] = n_datapoints

        logger.info(
            f"Rapiflex peak counts: {len(self._valid_indices)} valid pixels "
            f"x {n_datapoints} datapoints"
        )
        return peak_counts

    def close(self) -> None:
        """Close the reader and release resources."""
        if self._closed:
            return

        # Clear cached data
        self._mz_axis = None
        self._offsets = np.array([], dtype=np.uint32)
        self._positions = []

        self._closed = True
        logger.debug("RapiflexReader closed")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RapiflexReader(path={self.data_path}, "
            f"n_spectra={self.n_spectra}, "
            f"n_datapoints={self.n_datapoints})"
        )
