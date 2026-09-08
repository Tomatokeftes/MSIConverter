# thyra/readers/waters/imaging_grid.py
"""Imaging grid reconstruction from Waters laser position metadata.

Translates the logic from mzmine's ImagingMetadata.java to reconstruct
a pixel coordinate grid from the laser X/Y positions stored in each scan's
ScanInfo struct. Waters .raw imaging files do not store grid dimensions
explicitly -- they must be derived from the set of unique laser positions.

The derivation fits a *lattice* to those positions rather than ranking the
distinct ones. Ranking is exact only for a raster with every row and column
present: one missing interior row shifts every row below it up by one and
inflates the pitch, because the extent still spans the missing interval
while the interval count does not (issue #227). Fitting origin, pitch and
count, then indexing by ``round((pos - origin) / pitch)``, is right in both
cases and is what lets a function be tested against the raster it claims to
belong to (issue #232).
"""

import logging
import statistics
from dataclasses import dataclass, field
from functools import cached_property
from math import isfinite
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .masslynx_lib import NO_POSITION, FunctionType, MassLynxLib, ScanInfoData

logger = logging.getLogger(__name__)

#: How far off its lattice point a stage reading may sit, as a fraction of
#: the pitch, before the raster is not a raster. A quarter pitch is far
#: looser than any real stage -- the four registry acquisitions land on
#: exact multiples of their step -- and still refuses the wobble that used
#: to explode a 20x10 grid into 187x171 (issue #231).
_LATTICE_TOLERANCE = 0.25

#: The fraction of a fitted axis's lattice points that must carry a stage
#: reading. A raster missing a row or two still scores far above this; a
#: pitch mis-estimated from sub-pitch jitter scores a few percent, because
#: the fitted pitch is then the wobble and the lattice has orders of
#: magnitude more points than the acquisition has columns. Held at one half
#: rather than something tighter because a legitimately ragged acquisition
#: -- an aborted raster, a non-rectangular region -- still projects onto
#: nearly every value of each axis, so per-axis occupancy stays high even
#: when the image itself is sparse.
_MIN_LATTICE_OCCUPANCY = 0.5


@dataclass(frozen=True)
class AxisLattice:
    """The regular grid one stage axis was rastered on, in um.

    ``pitch`` is 0.0 and ``count`` is 1 for an axis with a single position:
    an aborted single-line scan has no interval to measure on its short
    axis, and waters_extractor turns a 0.0 pitch into ``pixel_size=None``
    and so into the actionable "use --pixel-size" refusal.
    """

    origin: float
    pitch: float
    count: int

    def index(self, position: float) -> int:
        """The 0-based lattice index a stage reading belongs to."""
        if self.pitch <= 0.0:
            return 0
        return int(round((position - self.origin) / self.pitch))

    def offset(self, position: float) -> float:
        """How far a stage reading sits from its lattice point, in um."""
        if self.pitch <= 0.0:
            return abs(position - self.origin)
        return abs(position - (self.origin + self.index(position) * self.pitch))

    def holds(self, position: float) -> bool:
        """Whether a stage reading lies on this lattice.

        Deliberately unbounded: a raster MassLynx split across functions
        continues past the extent of any one chunk, so the tail of a split
        raster lies on the lattice at an index beyond ``count``. What is
        being asked is whether the position shares the raster's phase, not
        whether it falls inside the part of it seen so far.
        """
        if not isfinite(position):
            return False
        if self.pitch <= 0.0:
            return position == self.origin
        return self.offset(position) <= _LATTICE_TOLERANCE * self.pitch


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
    #: The fitted raster, when this grid was built by :func:`build_imaging_grid`.
    #: ``None`` on a hand-built grid, where every position is taken on trust.
    x_lattice: Optional[AxisLattice] = None
    y_lattice: Optional[AxisLattice] = None

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Grid dimensions as (n_x, n_y, n_z). z is always 1 for 2D imaging."""
        return (self.pixel_count_x, self.pixel_count_y, 1)

    @cached_property
    def scans_by_function(self) -> Dict[int, List[ScanInfoData]]:
        """Positioned scan records grouped by function, in scan order.

        Built once and cached. Function selection asks five different
        questions of one function's scans and used to re-sort the whole
        scan map for each: O(F x N log N) over the file, measured at 87 s
        of selection against a 29 s grid build on a 25 M scan raster
        (issue #235).

        Cached, so ``scan_map`` must not be mutated once coordinates have
        been queried.
        """
        grouped: Dict[int, List[ScanInfoData]] = {}
        for (func, _scan), info in sorted(self.scan_map.items()):
            if info.has_position:
                grouped.setdefault(func, []).append(info)
        return grouped

    def lies_on_raster(self, scan_info: ScanInfoData) -> bool:
        """Whether a scan's stage reading shares the fitted raster's phase.

        True when the grid carries no fitted lattice, so a hand-built grid
        behaves as it always did.
        """
        if self.x_lattice is None or self.y_lattice is None:
            return True
        return self.x_lattice.holds(
            _mm_to_um_key(scan_info.laser_x_pos)
        ) and self.y_lattice.holds(_mm_to_um_key(scan_info.laser_y_pos))

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

        # A miss is not automatically a failure. The grid is fitted to the
        # functions that define the raster, so a function selection has not
        # accepted yet contributes no key to the maps -- including the tail
        # of a raster MassLynx split across functions, whose pixels are the
        # whole reason it gets rescued. A reading that lies on the raster
        # has a pixel even when the extent fitted so far does not reach it;
        # regrid_for_functions() grows the grid once the tail is kept, so
        # an index past the current count never survives into iteration.
        if (
            self.x_lattice is not None
            and self.y_lattice is not None
            and self.lies_on_raster(scan_info)
        ):
            return (self.x_lattice.index(x_um), self.y_lattice.index(y_um), 0)

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


def _fit_axis(positions: Sequence[float], axis: str) -> AxisLattice:
    """Fit origin, pitch and count to one axis's distinct stage readings.

    The pitch is seeded from the *median* neighbour interval, which is
    exact when every column is present and survives a minority of gaps: one
    missing row among ten turns a single interval into a double one and
    leaves the median at the true pitch, where ``extent / (n_distinct - 1)``
    reports it 12.5 % high (issue #227).

    It is then refined to ``extent / (count - 1)`` so the two end positions
    land exactly on lattice points, which keeps a complete raster on the
    integers it was acquired at: the four registry acquisitions still fit
    30.0000, 50.0000, 100.0000 and 500.0000 um exactly.

    Raises:
        ValueError: If the readings do not lie on a regular raster.
    """
    unique = sorted(set(positions))
    if len(unique) == 1:
        return AxisLattice(origin=unique[0], pitch=0.0, count=1)

    intervals = [b - a for a, b in zip(unique, unique[1:])]
    seed = statistics.median(intervals)
    span = unique[-1] - unique[0]
    if seed <= 0.0 or span <= 0.0:
        raise ValueError(
            f"Waters stage readings on the {axis} axis have no measurable "
            f"raster pitch ({len(unique)} distinct positions spanning "
            f"{span:.4f} um)."
        )

    count = int(round(span / seed)) + 1
    pitch = span / (count - 1)
    lattice = AxisLattice(origin=unique[0], pitch=pitch, count=count)

    worst = max(unique, key=lattice.offset)
    worst_offset = lattice.offset(worst)
    if worst_offset > _LATTICE_TOLERANCE * pitch:
        raise ValueError(
            f"Waters stage readings on the {axis} axis do not lie on a "
            f"regular raster: with a fitted pitch of {pitch:.4f} um the "
            f"reading at {worst:.2f} um sits {worst_offset:.4f} um "
            f"({100.0 * worst_offset / pitch:.1f} % of a pitch) off the "
            f"nearest raster line. Thyra will not guess which pixel it "
            f"belongs to."
        )

    occupancy = len(unique) / count
    if occupancy < _MIN_LATTICE_OCCUPANCY:
        raise ValueError(
            f"Waters stage readings on the {axis} axis fit a raster of "
            f"{count} lines at {pitch:.4f} um but only {len(unique)} of "
            f"them carry a reading ({100.0 * occupancy:.1f} %). That is "
            f"sub-pitch jitter being read as the pitch, not a raster with "
            f"gaps: the {len(unique)} distinct readings span "
            f"{span:.2f} um. Thyra will not build a grid from it."
        )

    if len(unique) < count:
        logger.warning(
            "The Waters %s axis is missing %d of its %d raster lines "
            "(pitch %.4f um). The grid keeps every line, so the gaps become "
            "empty pixels rather than shifting the rows past them.",
            axis,
            count - len(unique),
            count,
            pitch,
        )

    return lattice


@dataclass
class _PositionCensus:
    """What the stage readings of a set of functions looked like."""

    x_positions: List[float] = field(default_factory=list)
    y_positions: List[float] = field(default_factory=list)
    positioned: int = 0
    unpositioned: int = 0
    non_finite: int = 0
    partial_sentinel: int = 0


def _census(
    scan_map: Dict[Tuple[int, int], ScanInfoData],
    functions: Optional[Iterable[int]] = None,
) -> _PositionCensus:
    """Collect the stage readings of the given functions (all, when None)."""
    wanted = None if functions is None else set(functions)
    census = _PositionCensus()
    for (func, _scan), info in scan_map.items():
        if wanted is not None and func not in wanted:
            continue
        x_mm, y_mm = info.laser_x_pos, info.laser_y_pos
        if not (isfinite(x_mm) and isfinite(y_mm)):
            census.non_finite += 1
            continue
        if not info.has_position:
            census.unpositioned += 1
            continue
        if x_mm == NO_POSITION or y_mm == NO_POSITION:
            census.partial_sentinel += 1
        census.x_positions.append(_mm_to_um_key(x_mm))
        census.y_positions.append(_mm_to_um_key(y_mm))
        census.positioned += 1
    return census


def _grid_from_scan_map(
    scan_map: Dict[Tuple[int, int], ScanInfoData],
    functions: Optional[Iterable[int]] = None,
) -> ImagingGrid:
    """Fit a grid to the stage readings already collected in ``scan_map``.

    Split out of :func:`build_imaging_grid` so the reader can re-fit the
    raster to the functions it actually converts without touching the DLL
    again. Deriving the pitch from every function -- including the ones
    function selection then excludes -- let one off-raster reference spot
    turn a 10x5 image at 30 um into an 11x6 one at 1097 x 440 um
    (issue #232).
    """
    census = _census(scan_map, functions)

    if census.non_finite:
        logger.warning(
            "%d Waters scan(s) report a non-finite stage position (NaN or "
            "infinity) and are treated as unpositioned. A non-finite "
            "reading used to be admitted as its own column, making the "
            "reported pixel size NaN or infinite.",
            census.non_finite,
        )
    if census.partial_sentinel:
        logger.warning(
            "%d Waters scan(s) report the no-position sentinel (%.1f mm) on "
            "one axis and a real reading on the other. They are kept at "
            "face value, so a genuine stage position at %.1f mm on both "
            "axes would be dropped instead; check the raster origin if the "
            "image looks shifted.",
            census.partial_sentinel,
            NO_POSITION,
            NO_POSITION,
        )

    if not census.x_positions or not census.y_positions:
        raise ValueError("No valid laser positions found in Waters imaging data")

    x_lattice = _fit_axis(census.x_positions, "x")
    y_lattice = _fit_axis(census.y_positions, "y")

    # The no-position sentinel is a coordinate a stage can really visit, and
    # a raster that passes through it loses that pixel with no message. Only
    # worth saying when the raster actually reaches -1.0 mm on both axes,
    # which is why this is checked against the fitted lattice rather than
    # warned about on every file: the registry's DESI runs do use negative
    # stage millimetres (x from -11.4 mm, y from -6.0 mm), so this is not
    # hypothetical for them (issue #231).
    sentinel_um = _mm_to_um_key(NO_POSITION)
    if (
        census.unpositioned
        and x_lattice.holds(sentinel_um)
        and y_lattice.holds(sentinel_um)
        and 0 <= x_lattice.index(sentinel_um) < x_lattice.count
        and 0 <= y_lattice.index(sentinel_um) < y_lattice.count
    ):
        logger.warning(
            "This raster passes through (%.1f, %.1f) mm, which is also the "
            "no-position sentinel, so the %d scan(s) reporting it were "
            "dropped as unpositioned. Pixel (%d, %d) may be empty because "
            "of that rather than because nothing was acquired there.",
            NO_POSITION,
            NO_POSITION,
            census.unpositioned,
            x_lattice.index(sentinel_um),
            y_lattice.index(sentinel_um),
        )

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
    if x_lattice.count == 1 and y_lattice.count == 1:
        raise ValueError(
            f"Every positioned scan reports the same stage position "
            f"(x={x_lattice.origin:.1f} um, y={y_lattice.origin:.1f} um), so "
            f"this acquisition has a single pixel: {census.positioned} "
            f"positioned scans in {len(scan_map)} total. A one-pixel raster "
            f"with a 0.0 um pitch is a spot, calibration or tuning "
            f"acquisition, not an image -- Thyra has no raster to build "
            f"from it."
        )

    # Laser positions are pixel *centres*, so the span from the first to the
    # last is N - 1 pitches, not N. This used to divide by the count, which
    # mixed a centre-to-centre extent with an edge-to-edge one and reported
    # a pitch low by exactly 1/N -- independently per axis, so a square
    # raster came out anisotropic: measured, a 30 um acquisition gave
    # 29.66 x 28.93 and a 100 um one gave 99.40 x 97.83. See issue #217.
    # (mzmine, which this file translates, computes the extent the other way
    # round -- lateralWidth = count * pixelWidth -- and has the
    # divide-by-count form commented out in ImagingParameters.java.)
    x_index_map = {pos: x_lattice.index(pos) for pos in set(census.x_positions)}
    y_index_map = {pos: y_lattice.index(pos) for pos in set(census.y_positions)}

    lateral_width = x_lattice.pitch * (x_lattice.count - 1)
    lateral_height = y_lattice.pitch * (y_lattice.count - 1)

    _warn_on_shared_pixels(scan_map, functions, x_index_map, y_index_map)

    logger.info(
        f"Built imaging grid: {x_lattice.count}x{y_lattice.count} pixels, "
        f"pixel size: {x_lattice.pitch:.1f}x{y_lattice.pitch:.1f} um, "
        f"lateral: {lateral_width:.1f}x{lateral_height:.1f} um, "
        f"{census.positioned} positioned scans in {len(scan_map)} total scans"
    )

    return ImagingGrid(
        x_index_map=x_index_map,
        y_index_map=y_index_map,
        pixel_count_x=x_lattice.count,
        pixel_count_y=y_lattice.count,
        pixel_size_x=x_lattice.pitch,
        pixel_size_y=y_lattice.pitch,
        lateral_width=lateral_width,
        lateral_height=lateral_height,
        scan_map=scan_map,
        x_lattice=x_lattice,
        y_lattice=y_lattice,
    )


def _warn_on_shared_pixels(
    scan_map: Dict[Tuple[int, int], ScanInfoData],
    functions: Optional[Iterable[int]],
    x_index_map: Dict[float, int],
    y_index_map: Dict[float, int],
) -> None:
    """Say so when more than one scan lands on the same pixel.

    Two scans reporting the same stage position are summed into one pixel
    by the converter, which is a defensible choice and was a silent one:
    measured on the registry's 100 um MALDI set, function 2 holds 1275
    positioned scans on 1274 distinct pixels, the stage having stopped
    between two consecutive acquisitions 40 ms apart (issue #233).
    """
    wanted = None if functions is None else set(functions)
    seen: Set[Tuple[int, int]] = set()
    duplicated: Set[Tuple[int, int]] = set()
    extra = 0
    for (func, _scan), info in scan_map.items():
        if wanted is not None and func not in wanted:
            continue
        if not info.has_position:
            continue
        x_idx = x_index_map.get(_mm_to_um_key(info.laser_x_pos))
        y_idx = y_index_map.get(_mm_to_um_key(info.laser_y_pos))
        if x_idx is None or y_idx is None:
            continue
        pixel = (x_idx, y_idx)
        if pixel in seen:
            duplicated.add(pixel)
            extra += 1
        else:
            seen.add(pixel)

    if not duplicated:
        return

    sample = ", ".join(f"({x}, {y})" for x, y in sorted(duplicated)[:5])
    if len(duplicated) > 5:
        sample += ", ..."
    logger.warning(
        "%d Waters scan(s) land on %d pixel(s) that another scan already "
        "covers, so those pixels carry the SUM of every scan on them: %s. "
        "The stage reported the same position twice; the store has "
        "%d pixels for %d positioned scans.",
        extra,
        len(duplicated),
        sample,
        len(seen),
        len(seen) + extra,
    )


def build_imaging_grid(
    ml: MassLynxLib,
    handle,
    function_types: Dict[int, FunctionType],
    include_functions: Optional[Iterable[int]] = None,
) -> ImagingGrid:
    """Build imaging grid by scanning all functions/scans for laser coordinates.

    Translates ImagingMetadata.java constructor logic (lines 56-149).

    Every function's scan records are cached in ``scan_map`` whatever
    ``include_functions`` says, so the raster can be re-fitted later
    (:func:`regrid_for_functions`) without touching the DLL again.

    Args:
        ml: MassLynxLib instance with open handle.
        handle: The opaque file handle from ml.open_file().
        function_types: Pre-classified function types for all functions.
        include_functions: Fit the raster to these functions' stage
            readings only. ``None`` fits it to every function, which is
            what a caller that has not classified them yet wants.

    Returns:
        ImagingGrid with coordinate maps and pixel metadata.

    Raises:
        ValueError: If no scan carries a laser position, if the positions do
            not lie on a regular raster, or if every positioned scan carries
            the *same* one -- a single-pixel "raster" is a spot acquisition,
            not an image.
    """
    scan_map: Dict[Tuple[int, int], ScanInfoData] = {}

    n_functions = ml.get_number_of_functions(handle)
    for func in range(n_functions):
        n_scans = ml.get_number_of_scans_in_function(handle, func)
        for scan in range(n_scans):
            scan_map[(func, scan)] = ml.get_scan_info(handle, func, scan)

    return _grid_from_scan_map(scan_map, include_functions)


def regrid_for_functions(grid: ImagingGrid, functions: Iterable[int]) -> ImagingGrid:
    """Re-fit the raster to the stage readings of ``functions`` alone.

    Uses the scan records already cached on ``grid``, so this costs no
    further DLL calls. Returns ``grid`` unchanged when the restriction
    leaves the geometry identical, which is every acquisition whose
    excluded functions sat on the raster with the rest, and when ``grid``
    carries no fitted lattice -- a hand-built grid is taken on trust, the
    same way :meth:`ImagingGrid.lies_on_raster` takes it.
    """
    if grid.x_lattice is None or grid.y_lattice is None:
        return grid
    refitted = _grid_from_scan_map(grid.scan_map, functions)
    same = (
        refitted.pixel_count_x == grid.pixel_count_x
        and refitted.pixel_count_y == grid.pixel_count_y
        and refitted.pixel_size_x == grid.pixel_size_x
        and refitted.pixel_size_y == grid.pixel_size_y
        and refitted.x_index_map == grid.x_index_map
        and refitted.y_index_map == grid.y_index_map
    )
    if same:
        return grid
    logger.warning(
        "The imaging grid was re-fitted to the converted functions alone: "
        "%dx%d at %.4f x %.4f um, was %dx%d at %.4f x %.4f um. The excluded "
        "functions' stage readings were part of the raster Thyra measured.",
        refitted.pixel_count_x,
        refitted.pixel_count_y,
        refitted.pixel_size_x,
        refitted.pixel_size_y,
        grid.pixel_count_x,
        grid.pixel_count_y,
        grid.pixel_size_x,
        grid.pixel_size_y,
    )
    return refitted
