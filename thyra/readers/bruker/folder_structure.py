# thyra/readers/bruker/folder_structure.py
"""Bruker MSI folder structure abstraction.

This module provides a lightweight, pure Python abstraction for analyzing
Bruker MSI folder structures. It handles format detection and file discovery
without requiring any SDK dependencies.

Supported formats:
- timsTOF: .d folders containing analysis.tdf or analysis.tsf
- Rapiflex: Folders with .dat, _poslog.txt, and _info.txt files
- solariX: .d folders containing peaks.sqlite and ImagingInfo.xml
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from ...errors import ConversionRefused

logger = logging.getLogger(__name__)


class BrukerFormat(Enum):
    """Bruker MSI data formats."""

    TIMSTOF = "timstof"
    RAPIFLEX = "rapiflex"
    SOLARIX = "solarix"
    UNKNOWN = "unknown"


@dataclass
class BrukerFolderInfo:
    """Information about a Bruker MSI folder structure.

    Attributes:
        path: Root path of the Bruker data
        format: Detected Bruker format
        data_path: Path to the main data (e.g., .d folder or data folder)
        optical_images: List of optical image paths (TIFFs)
        teaching_points_file: Path to teaching points file (e.g., .mis)
        metadata_files: Dictionary of metadata file paths
    """

    path: Path
    format: BrukerFormat
    data_path: Path
    optical_images: List[Path] = field(default_factory=list)
    teaching_points_file: Optional[Path] = None
    metadata_files: dict = field(default_factory=dict)


class BrukerFolderStructure:
    """Lightweight analyzer for Bruker MSI folder structures.

    This class provides format detection and file discovery for Bruker MSI
    data without requiring any SDK. It's designed to be used for:
    - Automatic format detection
    - Finding optical images
    - Locating metadata and alignment files

    No SDK is required - all operations are pure Python file system checks.

    Example:
        >>> folder = BrukerFolderStructure(Path("/path/to/data"))
        >>> info = folder.analyze()
        >>> print(f"Format: {info.format.value}")
        >>> print(f"Optical images: {info.optical_images}")
    """

    # File patterns for Rapiflex format
    RAPIFLEX_PATTERNS = {
        "data": "*.dat",
        "poslog": "*_poslog.txt",
        "info": "*_info.txt",
        "mis": "*.mis",
    }

    # File patterns for timsTOF format
    TIMSTOF_PATTERNS = {
        "tdf": "analysis.tdf",
        "tsf": "analysis.tsf",
        "tdf_bin": "analysis.tdf_bin",
        "tsf_bin": "analysis.tsf_bin",
    }

    # File patterns for solariX (FT-ICR / MRMS) format. Detection keys on
    # the processed peak store plus the per-scan index; ``ser`` (the raw
    # transient block) marks the solariX family but is never read.
    SOLARIX_PATTERNS = {
        "peaks": "peaks.sqlite",
        "imaging_info": "ImagingInfo.xml",
        "ser": "ser",
    }

    # Common optical image patterns
    OPTICAL_IMAGE_PATTERNS = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]

    def __init__(self, path: Path):
        """Initialize folder structure analyzer.

        Args:
            path: Path to analyze (can be .d folder or parent folder)
        """
        self.path = Path(path)
        self._info: Optional[BrukerFolderInfo] = None

    def analyze(self) -> BrukerFolderInfo:
        """Analyze the folder structure and return information.

        Returns:
            BrukerFolderInfo with detected format and file paths

        Raises:
            ValueError: If path doesn't exist
        """
        if not self.path.exists():
            raise ConversionRefused(f"Path does not exist: {self.path}")

        # Cache the result
        if self._info is None:
            self._info = self._analyze_structure()

        return self._info

    def _analyze_structure(self) -> BrukerFolderInfo:
        """Perform the actual folder analysis."""
        # First, detect the format
        fmt, data_path = self._detect_format()

        # Find optical images
        optical_images = self._find_optical_images(data_path)

        # Find teaching points file
        teaching_points_file = self._find_teaching_points_file(data_path)

        # Find other metadata files
        metadata_files = self._find_metadata_files(data_path, fmt)

        return BrukerFolderInfo(
            path=self.path,
            format=fmt,
            data_path=data_path,
            optical_images=optical_images,
            teaching_points_file=teaching_points_file,
            metadata_files=metadata_files,
        )

    def _detect_format(self) -> Tuple["BrukerFormat", Path]:
        """Detect the Bruker format and return (format, data_path)."""
        # Check if this is a .d folder (timsTOF or solariX)
        if self.path.suffix.lower() == ".d":
            if self._is_timstof_folder(self.path):
                return BrukerFormat.TIMSTOF, self.path
            if self._is_solarix_folder(self.path):
                return BrukerFormat.SOLARIX, self.path

        # Check if this folder contains Rapiflex data
        if self._is_rapiflex_folder(self.path):
            return BrukerFormat.RAPIFLEX, self.path

        # Check if this is a parent folder containing a .d subfolder
        d_folders = list(self.path.glob("*.d"))
        for d_folder in d_folders:
            if self._is_timstof_folder(d_folder):
                return BrukerFormat.TIMSTOF, d_folder
            if self._is_solarix_folder(d_folder):
                return BrukerFormat.SOLARIX, d_folder

        # Check subfolders for Rapiflex
        for subdir in self.path.iterdir():
            if subdir.is_dir() and self._is_rapiflex_folder(subdir):
                return BrukerFormat.RAPIFLEX, subdir

        return BrukerFormat.UNKNOWN, self.path

    def _is_timstof_folder(self, path: Path) -> bool:
        """Check if path is a timsTOF .d folder."""
        if not path.is_dir():
            return False

        has_tdf = (path / self.TIMSTOF_PATTERNS["tdf"]).exists()
        has_tsf = (path / self.TIMSTOF_PATTERNS["tsf"]).exists()

        return has_tdf or has_tsf

    def _is_solarix_folder(self, path: Path) -> bool:
        """Check if path is a solariX imaging .d folder.

        Requires BOTH the processed peak store and the per-scan index.
        FlexImaging pre-scan directories (``fid`` + ``analysis.baf``) carry
        neither and must fall through to UNKNOWN.
        """
        if not path.is_dir():
            return False

        has_peaks = (path / self.SOLARIX_PATTERNS["peaks"]).exists()
        has_imaging_info = (path / self.SOLARIX_PATTERNS["imaging_info"]).exists()

        return has_peaks and has_imaging_info

    @classmethod
    def is_solarix_without_peaks(cls, path: Path) -> bool:
        """Check if path is a solariX-family .d that lacks processed peaks.

        A ``.d`` holding raw transients (``ser``) and ``ImagingInfo.xml``
        but no ``peaks.sqlite`` is a solariX acquisition Thyra cannot read
        natively; callers use this to name the imzML-export fallback instead
        of reporting a generic detection failure.
        """
        path = Path(path)
        if not path.is_dir():
            return False

        has_ser = (path / cls.SOLARIX_PATTERNS["ser"]).exists()
        has_imaging_info = (path / cls.SOLARIX_PATTERNS["imaging_info"]).exists()
        has_peaks = (path / cls.SOLARIX_PATTERNS["peaks"]).exists()

        return has_ser and has_imaging_info and not has_peaks

    def _is_rapiflex_folder(self, path: Path) -> bool:
        """Check if path is a Rapiflex data folder."""
        if not path.is_dir():
            return False

        has_dat = bool(list(path.glob(self.RAPIFLEX_PATTERNS["data"])))
        has_poslog = bool(list(path.glob(self.RAPIFLEX_PATTERNS["poslog"])))
        has_info = bool(list(path.glob(self.RAPIFLEX_PATTERNS["info"])))

        return has_dat and has_poslog and has_info

    def _find_optical_images(self, data_path: Path) -> List[Path]:
        """Find optical TIFF images in the folder structure.

        Searches both the data folder and its parent for optical images.

        Args:
            data_path: Path to the data folder

        Returns:
            List of paths to TIFF files
        """
        optical_images = []

        # Search paths: data folder, parent folder, and common subdirs
        search_paths = [data_path]
        if data_path != self.path:
            search_paths.append(self.path)
        if data_path.parent != data_path:
            search_paths.append(data_path.parent)

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for pattern in self.OPTICAL_IMAGE_PATTERNS:
                for tiff_path in search_path.glob(pattern):
                    if tiff_path not in optical_images:
                        optical_images.append(tiff_path)
                        logger.debug(f"Found optical image: {tiff_path}")

        return sorted(optical_images)

    def _find_teaching_points_file(self, data_path: Path) -> Optional[Path]:
        """Find the teaching points / alignment file.

        For Rapiflex, this is the .mis file.
        For timsTOF, teaching points may be in other locations (TBD).

        When multiple .mis files exist (e.g., multi-dataset slides where
        each .d has its own .mis), the file whose stem matches the .d
        folder stem is preferred. This ensures each dataset uses its own
        Area definitions for correct optical alignment.

        Args:
            data_path: Path to the data folder

        Returns:
            Path to teaching points file, or None if not found
        """
        # The .d folder stem is the matching key (e.g., "sample_E2506")
        d_stem = data_path.stem

        search_paths = [data_path, self.path, data_path.parent]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            mis_files = list(search_path.glob("*.mis"))
            if mis_files:
                # Prefer .mis file whose stem matches the .d folder stem
                for mis_file in mis_files:
                    if mis_file.stem == d_stem:
                        logger.debug(
                            f"Found matching teaching points file: " f"{mis_file.name}"
                        )
                        return mis_file

                # Fallback to first .mis file found
                logger.debug(f"No .mis matching '{d_stem}', using {mis_files[0].name}")
                return mis_files[0]

        return None

    def _find_metadata_files(self, data_path: Path, fmt: BrukerFormat) -> dict:
        """Find metadata files based on format.

        Args:
            data_path: Path to the data folder
            fmt: Detected Bruker format

        Returns:
            Dictionary of metadata file paths
        """
        metadata = {}

        if fmt == BrukerFormat.RAPIFLEX:
            # Rapiflex metadata files
            for name, pattern in self.RAPIFLEX_PATTERNS.items():
                files = list(data_path.glob(pattern))
                if files:
                    metadata[name] = files[0] if len(files) == 1 else files

        elif fmt == BrukerFormat.TIMSTOF:
            # timsTOF metadata files
            for name, pattern in self.TIMSTOF_PATTERNS.items():
                file_path = data_path / pattern
                if file_path.exists():
                    metadata[name] = file_path

        elif fmt == BrukerFormat.SOLARIX:
            # solariX metadata files inside the .d
            for name, pattern in self.SOLARIX_PATTERNS.items():
                file_path = data_path / pattern
                if file_path.exists():
                    metadata[name] = file_path
            # Acquisition method (XML) inside the *.m method directory
            method_files = sorted(data_path.glob("*.m/apexAcquisition.method"))
            if method_files:
                metadata["method"] = method_files[0]

        return metadata

    @classmethod
    def detect_format(cls, path: Path) -> BrukerFormat:
        """Quick format detection without full analysis.

        Args:
            path: Path to check

        Returns:
            Detected BrukerFormat
        """
        analyzer = cls(path)
        fmt, _ = analyzer._detect_format()
        return fmt

    @classmethod
    def is_bruker_data(cls, path: Path) -> bool:
        """Check if path contains Bruker MSI data.

        Args:
            path: Path to check

        Returns:
            True if Bruker data is detected
        """
        fmt = cls.detect_format(path)
        return fmt != BrukerFormat.UNKNOWN
