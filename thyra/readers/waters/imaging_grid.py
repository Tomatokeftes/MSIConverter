# thyra/readers/waters/imaging_grid.py
"""Imaging grid reconstruction from Waters laser position metadata.

Translates the logic from mzmine's ImagingMetadata.java to reconstruct
a pixel coordinate grid from the laser X/Y positions stored in each scan's
ScanInfo struct. Waters .raw imaging files do not store grid dimensions
explicitly -- they must be derived from the set of unique laser positions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .masslynx_lib import FunctionType, MassLynxLib, ScanInfoData

logger = logging.getLogger(__name__)


@dataclass
class ImagingGrid:
    """Reconstructed imaging grid from Waters laser coordinates.

    Built by scanning all functions/scans and collecting unique laser
    positions. Provides O(1) coordinate lookup during spectrum iteration.
    """

    x_index_map: Dict[float, int]  # position (um) -> 0-based x index
    y_index_map: Dict[float, int]  # position (um) -> 0-based y index
    pixel_count_x: int
    pixel_count_y: int
    pixel_size_x: float  # raster pitch in um; 0.0 when the axis has one position
    pixel_size_y: float  # raster pitch in um; 0.0 when the axis has one position
    lateral_width: float  # max_x - min_x in um, so centre-to-centre, not edge-to-edge
    lateral_height: float  # max_y - min_y in um, so centre-to-centre, not edge-to-edge
    scan_map: Dict[Tuple[int, int], ScanInfoData] = field(repr=False)

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Grid dimensions as (n_x, n_y, n_z). z is always 1 for 2D imaging."""
        return (self.pixel_count_x, self.pixel_count_y, 1)

    def get_coordinates(
        self, scan_info: ScanInfoData
    ) -> Optional[Tuple[int, int, int]]:
        """Map laser position (mm from DLL) to 0-based pixel coordinates.

        The DLL returns positions in mm. We multiply by 1000 to get um,
        matching the keys stored in x_index_map/y_index_map (which were
        also built from mm * 1000).
        """
        if not scan_info.has_position:
            return None

        x_um = _mm_to_um_key(scan_info.laser_x_pos)
        y_um = _mm_to_um_key(scan_info.laser_y_pos)

        x_idx = self.x_index_map.get(x_um)
        y_idx = self.y_index_map.get(y_um)

        if x_idx is not None and y_idx is not None:
            return (x_idx, y_idx, 0)

        logger.warning(
            f"No index found for laser position x={scan_info.laser_x_pos:.4f}mm "
            f"({x_um:.1f}um) -> {x_idx}, y={scan_info.laser_y_pos:.4f}mm "
            f"({y_um:.1f}um) -> {y_idx}"
        )
        return None


def _mm_to_um_key(mm_value: float) -> float:
    """Convert mm to um and round for use as dict key.

    Rounding to 2 decimal places (0.01 um precision) avoids floating-point
    comparison issues when using floats as dictionary keys. The mzmine Java
    code uses Float.compare() which does exact bitwise comparison, but
    Python float arithmetic from c_float -> Python float conversion can
    introduce tiny differences.
    """
    return round(mm_value * 1000.0, 2)


def build_imaging_grid(
    ml: MassLynxLib,
    handle,
    function_types: Dict[int, FunctionType],
) -> ImagingGrid:
    """Build imaging grid by scanning all functions/scans for laser coordinates.

    Translates ImagingMetadata.java constructor logic (lines 56-149).

    Args:
        ml: MassLynxLib instance with open handle.
        handle: The opaque file handle from ml.open_file().
        function_types: Pre-classified function types for all functions.

    Returns:
        ImagingGrid with coordinate maps and pixel metadata.

    Raises:
        ValueError: If no scan carries a laser position, or if every
            positioned scan carries the *same* one -- a single-pixel
            "raster" is a spot acquisition, not an image.
    """
    x_positions: set = set()
    y_positions: set = set()
    scan_map: Dict[Tuple[int, int], ScanInfoData] = {}

    n_functions = ml.get_number_of_functions(handle)
    total_scans = 0

    for func in range(n_functions):
        n_scans = ml.get_number_of_scans_in_function(handle, func)
        for scan in range(n_scans):
            scan_info = ml.get_scan_info(handle, func, scan)
            scan_map[(func, scan)] = scan_info

            if not scan_info.has_position:
                continue

            x_um = _mm_to_um_key(scan_info.laser_x_pos)
            y_um = _mm_to_um_key(scan_info.laser_y_pos)
            x_positions.add(x_um)
            y_positions.add(y_um)
            total_scans += 1

    if not x_positions or not y_positions:
        raise ValueError("No valid laser positions found in Waters imaging data")

    # Build sorted index maps (position_um -> 0-based index)
    sorted_x = sorted(x_positions)
    sorted_y = sorted(y_positions)
    x_index_map = {val: idx for idx, val in enumerate(sorted_x)}
    y_index_map = {val: idx for idx, val in enumerate(sorted_y)}

    pixel_count_x = len(sorted_x)
    pixel_count_y = len(sorted_y)

    # A stage that never moved is not a raster. The check above only catches
    # the total absence of positions; a *constant* position passes it, and
    # everything downstream then agrees the run is a legitimate 1x1 image
    # with a 0.0 um pitch (lateral extent is 0, so both pixel sizes below
    # come out 0.0) and converts the whole acquisition onto one pixel
    # without a word. Measured on the Xevo DESI share, 76 of 105 .raw dirs
    # land here -- every one of them a calibration, a tuning run, a lysis
    # test or a single-spot DDA acquisition, and not one of them an image.
    # Real DESI images on that same share do carry stage coordinates in
    # these fields and grid normally (401x401, 247x140, ...), so this
    # refuses the non-images without costing anything real. See issue #213.
    if pixel_count_x == 1 and pixel_count_y == 1:
        raise ValueError(
            f"Every positioned scan reports the same stage position "
            f"(x={sorted_x[0]:.1f} um, y={sorted_y[0]:.1f} um), so this "
            f"acquisition has a single pixel: {total_scans} positioned scans "
            f"in {len(scan_map)} total. A one-pixel raster with a 0.0 um "
            f"pitch is a spot, calibration or tuning acquisition, not an "
            f"image -- Thyra has no raster to build from it."
        )

    # Laser positions are pixel *centres*, so the span from the first to the
    # last is N - 1 pitches, not N. This used to divide by the count, which
    # mixed a centre-to-centre extent with an edge-to-edge one and reported
    # a pitch low by exactly 1/N -- independently per axis, so a square
    # raster came out anisotropic: measured, a 30 um acquisition gave
    # 29.66 x 28.93 and a 100 um one gave 99.40 x 97.83. Nothing caught it
    # because _determine_pixel_size keeps only x, so the disagreement
    # between the two axes never reached anywhere it would look wrong.
    # See issue #217. (mzmine, which this file translates, computes the
    # extent the other way round -- lateralWidth = count * pixelWidth -- and
    # has the divide-by-count form commented out in ImagingParameters.java.)
    #
    # An axis with a single position has no interval to measure, and keeps
    # 0.0: waters_extractor turns that into pixel_size=None, which becomes
    # the actionable "Pixel size not found in metadata. Use --pixel-size"
    # refusal rather than a fabricated number. Aborted single-line scans
    # (the share holds 138x1 and 48x1) take that path.
    lateral_width = sorted_x[-1] - sorted_x[0] if pixel_count_x > 1 else 0.0
    lateral_height = sorted_y[-1] - sorted_y[0] if pixel_count_y > 1 else 0.0

    pixel_size_x = lateral_width / (pixel_count_x - 1) if pixel_count_x > 1 else 0.0
    pixel_size_y = lateral_height / (pixel_count_y - 1) if pixel_count_y > 1 else 0.0

    logger.info(
        f"Built imaging grid: {pixel_count_x}x{pixel_count_y} pixels, "
        f"pixel size: {pixel_size_x:.1f}x{pixel_size_y:.1f} um, "
        f"lateral: {lateral_width:.1f}x{lateral_height:.1f} um, "
        f"{total_scans} positioned scans in {len(scan_map)} total scans"
    )

    return ImagingGrid(
        x_index_map=x_index_map,
        y_index_map=y_index_map,
        pixel_count_x=pixel_count_x,
        pixel_count_y=pixel_count_y,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        lateral_width=lateral_width,
        lateral_height=lateral_height,
        scan_map=scan_map,
    )
