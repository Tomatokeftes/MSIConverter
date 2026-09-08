"""The raster Thyra fits to Waters stage readings.

Covers the geometry half of the 2026-09-08 sweep: an interior gap in the
raster (#227), stage readings that do not lie on one (#231), a grid built
from functions that are not converted (#232), two scans on one pixel
(#233), and the per-function scan index that function selection reads
(#235). MassLynxLib is never loaded -- these build a ``scan_map`` directly,
which is what ``build_imaging_grid`` hands on once the DLL sweep is done.
"""

import logging
import math

import pytest

from thyra.readers.waters.imaging_grid import (
    _grid_from_scan_map,
    _mm_to_um_key,
    regrid_for_functions,
)
from thyra.readers.waters.masslynx_lib import ScanInfoData


def _scan(x_mm, y_mm, func_level=1):
    return ScanInfoData(
        ms_level=func_level,
        polarity=0,
        drift_scan_count=0,
        is_profile=0,
        precursor_mz=0.0,
        rt=1.0,
        laser_x_pos=x_mm,
        laser_y_pos=y_mm,
    )


def _raster(
    n_x, n_y, pitch_um, func=0, skip_rows=(), jitter_um=0.0, origin_mm=(0.0, 0.0)
):
    """A scan map for one rectangular raster, in MassLynx's millimetres."""
    scan_map = {}
    scan = 0
    x0, y0 = origin_mm
    for row in range(n_y):
        if row in skip_rows:
            continue
        for col in range(n_x):
            wobble_x = jitter_um * math.sin(row * 7.0 + col * 3.0)
            wobble_y = jitter_um * math.cos(row * 5.0 + col * 11.0)
            scan_map[(func, scan)] = _scan(
                x0 + (col * pitch_um + wobble_x) / 1000.0,
                y0 + (row * pitch_um + wobble_y) / 1000.0,
            )
            scan += 1
    return scan_map


class TestCompleteRasterIsUnchanged:
    """The fit must not move a raster that was already right (#217)."""

    @pytest.mark.parametrize("pitch", [30.0, 50.0, 100.0, 500.0])
    def test_a_complete_raster_keeps_its_declared_pitch_exactly(self, pitch):
        grid = _grid_from_scan_map(_raster(8, 6, pitch))

        assert grid.pixel_count_x == 8
        assert grid.pixel_count_y == 6
        assert grid.pixel_size_x == pytest.approx(pitch, abs=1e-9)
        assert grid.pixel_size_y == pytest.approx(pitch, abs=1e-9)

    def test_a_single_line_scan_keeps_a_zero_pitch_on_its_short_axis(self):
        """138x1 and 48x1 runs exist; a 0.0 pitch becomes the --pixel-size refusal."""
        grid = _grid_from_scan_map(_raster(6, 1, 30.0))

        assert (grid.pixel_count_x, grid.pixel_count_y) == (6, 1)
        assert grid.pixel_size_x == pytest.approx(30.0)
        assert grid.pixel_size_y == 0.0


class TestInteriorGap:
    """A missing row must not inflate the pitch or shift the rows past it."""

    def test_the_pitch_comes_from_the_neighbour_interval_not_the_extent(self):
        # Row 5 of 10 is missing: the extent still spans 9 pitches while only
        # 8 intervals are counted, which reported 33.75 um for a 30 um raster.
        grid = _grid_from_scan_map(_raster(20, 10, 30.0, skip_rows=(5,)))

        assert grid.pixel_size_y == pytest.approx(30.0, abs=1e-9)
        assert grid.pixel_size_x == pytest.approx(30.0, abs=1e-9)

    def test_the_grid_keeps_the_missing_row_as_empty_pixels(self):
        grid = _grid_from_scan_map(_raster(20, 10, 30.0, skip_rows=(5,)))

        assert grid.pixel_count_y == 10
        assert grid.lateral_height == pytest.approx(270.0)

    def test_rows_below_the_gap_keep_their_own_index(self):
        """The defect: every row below the gap was stored one row up."""
        grid = _grid_from_scan_map(_raster(20, 10, 30.0, skip_rows=(5,)))

        # Row 6 sits at 180 um. Ranking the 9 distinct values put it at 5.
        assert grid.y_index_map[_mm_to_um_key(0.180)] == 6
        assert grid.y_index_map[_mm_to_um_key(0.270)] == 9
        assert _mm_to_um_key(0.150) not in grid.y_index_map

    def test_a_missing_last_row_simply_shortens_the_raster(self):
        grid = _grid_from_scan_map(_raster(20, 10, 30.0, skip_rows=(9,)))

        assert grid.pixel_count_y == 9
        assert grid.pixel_size_y == pytest.approx(30.0)

    def test_the_gap_is_reported(self, caplog):
        with caplog.at_level(logging.WARNING):
            _grid_from_scan_map(_raster(20, 10, 30.0, skip_rows=(5,)))

        assert "missing 1 of its 10 raster lines" in caplog.text


class TestReadingsThatAreNotARaster:
    """Sub-pitch jitter used to explode the grid instead of being refused."""

    def test_jitter_is_refused_rather_than_made_into_a_grid(self):
        # +-1 % of a 30 um pitch gave a 187x171 store with a 3 x 1.6 um
        # "pitch" and 200 of 31,977 pixels filled, silently.
        with pytest.raises(ValueError, match="do not lie on a regular raster|only"):
            _grid_from_scan_map(_raster(20, 10, 30.0, jitter_um=0.3))

    def test_jitter_far_below_the_key_resolution_still_grids(self):
        """0.003 um is under the 0.01 um key rounding, so it is not jitter."""
        grid = _grid_from_scan_map(_raster(20, 10, 30.0, jitter_um=0.003))

        assert (grid.pixel_count_x, grid.pixel_count_y) == (20, 10)
        assert grid.pixel_size_x == pytest.approx(30.0, abs=1e-6)

    def test_the_refusal_names_the_axis_and_the_pitch_it_fitted(self):
        with pytest.raises(ValueError) as excinfo:
            _grid_from_scan_map(_raster(20, 10, 30.0, jitter_um=0.3))

        assert "x axis" in str(excinfo.value) or "y axis" in str(excinfo.value)


class TestNonFinitePositions:
    """NaN is not equal to itself, so it passed the sentinel test."""

    def test_a_nan_position_is_not_a_position(self):
        assert not _scan(float("nan"), 0.05).has_position
        assert not _scan(0.1, float("nan")).has_position

    def test_an_infinite_position_is_not_a_position(self):
        assert not _scan(float("inf"), 0.05).has_position
        assert not _scan(0.1, float("-inf")).has_position

    def test_a_nan_scan_does_not_reach_the_grid(self):
        scan_map = _raster(5, 4, 30.0)
        scan_map[(0, 999)] = _scan(float("nan"), 0.03)

        grid = _grid_from_scan_map(scan_map)

        assert (grid.pixel_count_x, grid.pixel_count_y) == (5, 4)
        assert math.isfinite(grid.pixel_size_x)
        assert grid.pixel_size_x == pytest.approx(30.0)

    def test_two_nan_scans_do_not_change_the_real_pitch(self):
        scan_map = _raster(5, 4, 30.0)
        scan_map[(0, 998)] = _scan(float("nan"), 0.03)
        scan_map[(0, 999)] = _scan(float("nan"), 0.06)

        grid = _grid_from_scan_map(scan_map)

        assert grid.pixel_count_x == 5
        assert grid.pixel_size_x == pytest.approx(30.0)

    def test_the_dropped_scans_are_reported(self, caplog):
        scan_map = _raster(5, 4, 30.0)
        scan_map[(0, 999)] = _scan(float("inf"), 0.03)

        with caplog.at_level(logging.WARNING):
            _grid_from_scan_map(scan_map)

        assert "non-finite stage position" in caplog.text


class TestSentinelOnTheRaster:
    """-1.0 mm is a coordinate a stage can really visit."""

    def test_a_raster_passing_through_the_sentinel_says_so(self, caplog):
        # 100 um raster from (-1.2, -1.2) mm, so (-1.0, -1.0) is on it.
        scan_map = _raster(5, 5, 100.0, origin_mm=(-1.2, -1.2))
        # The scan that would have landed there reports the sentinel.
        scan_map[(0, 12)] = _scan(-1.0, -1.0)

        with caplog.at_level(logging.WARNING):
            _grid_from_scan_map(scan_map)

        assert "no-position sentinel" in caplog.text

    def test_a_raster_nowhere_near_the_sentinel_stays_quiet(self, caplog):
        scan_map = _raster(5, 5, 100.0)
        scan_map[(0, 99)] = _scan(-1.0, -1.0)

        with caplog.at_level(logging.WARNING):
            _grid_from_scan_map(scan_map)

        assert "no-position sentinel" not in caplog.text


class TestSharedPixels:
    """Two scans on one pixel are summed; that was silent."""

    def test_a_repeated_stage_position_is_reported(self, caplog):
        scan_map = _raster(4, 3, 30.0)
        # The stage stopped: one more scan at the position of scan 0.
        scan_map[(0, 500)] = _scan(0.0, 0.0)

        with caplog.at_level(logging.WARNING):
            _grid_from_scan_map(scan_map)

        assert "carry the SUM of every scan on them" in caplog.text
        assert "(0, 0)" in caplog.text

    def test_a_clean_raster_stays_quiet(self, caplog):
        with caplog.at_level(logging.WARNING):
            _grid_from_scan_map(_raster(4, 3, 30.0))

        assert "SUM of every scan" not in caplog.text


class TestGridFromConvertedFunctionsOnly:
    """The pitch must not come from a function that contributes no pixel."""

    def test_an_off_raster_spot_does_not_stretch_the_grid(self):
        scan_map = _raster(10, 5, 30.0, func=0)
        # A reference spot parked well off the raster, as its own function.
        # Folded in, it used to make an 11x6 image at a 1097 x 440 um
        # "pitch"; the occupancy guard now refuses that outright, and the
        # reader never offers it the chance by fitting the MS functions only.
        scan_map[(1, 0)] = _scan(11.0, 4.4)

        with pytest.raises(ValueError, match="carry a reading"):
            _grid_from_scan_map(scan_map)

        without_spot = _grid_from_scan_map(scan_map, functions=[0])

        assert without_spot.pixel_count_x == 10
        assert without_spot.pixel_count_y == 5
        assert without_spot.pixel_size_x == pytest.approx(30.0)
        assert without_spot.pixel_size_y == pytest.approx(30.0)

    def test_a_half_pitch_offset_function_is_not_interleaved_into_the_raster(self):
        """The second #232 layout: MS2 read half a pitch off the MS1 raster."""
        scan_map = _raster(10, 5, 30.0, func=0)
        scan_map.update(_raster(10, 5, 30.0, func=1, origin_mm=(0.015, 0.0)))

        interleaved = _grid_from_scan_map(scan_map)
        ms1_only = _grid_from_scan_map(scan_map, functions=[0])

        # Folded in, the offset function doubles the columns and halves the
        # pitch; on its own the MS1 raster is what it always was.
        assert interleaved.pixel_count_x == 20
        assert interleaved.pixel_size_x == pytest.approx(15.0)
        assert ms1_only.pixel_count_x == 10
        assert ms1_only.pixel_size_x == pytest.approx(30.0)
        # And the offset readings are not on the MS1 raster, so the reader
        # excludes them rather than converting them as image pixels.
        assert not ms1_only.lies_on_raster(_scan(0.015, 0.0))

    def test_an_off_raster_reading_does_not_lie_on_the_raster(self):
        grid = _grid_from_scan_map(_raster(10, 5, 30.0))

        assert grid.lies_on_raster(_scan(0.060, 0.030))
        assert not grid.lies_on_raster(_scan(0.075, 0.030))  # half a pitch off
        assert not grid.lies_on_raster(_scan(11.0, 4.4))

    def test_the_tail_of_a_split_raster_does_lie_on_it(self):
        """A chunk MassLynx split off continues the raster past its extent."""
        grid = _grid_from_scan_map(_raster(10, 5, 30.0))

        # Row 7 is beyond this chunk's five rows but shares its phase.
        assert grid.lies_on_raster(_scan(0.060, 0.210))

    def test_a_hand_built_grid_is_taken_on_trust(self):
        grid = _grid_from_scan_map(_raster(10, 5, 30.0))
        grid.x_lattice = None

        assert grid.lies_on_raster(_scan(11.0, 4.4))

    def test_regrid_returns_the_same_object_when_nothing_moves(self):
        grid = _grid_from_scan_map(_raster(10, 5, 30.0, func=0))

        assert regrid_for_functions(grid, [0]) is grid

    def test_regrid_reports_a_geometry_it_had_to_change(self, caplog):
        scan_map = _raster(10, 5, 30.0, func=0)
        scan_map.update(_raster(10, 5, 30.0, func=1, origin_mm=(0.015, 0.0)))
        grid = _grid_from_scan_map(scan_map)

        with caplog.at_level(logging.WARNING):
            refitted = regrid_for_functions(grid, [0])

        assert refitted is not grid
        assert refitted.pixel_size_x == pytest.approx(30.0)
        assert "re-fitted to the converted functions" in caplog.text


class TestScansByFunction:
    """One pass instead of a re-sort per query (#235)."""

    def test_scans_are_grouped_by_function_in_scan_order(self):
        scan_map = _raster(3, 2, 30.0, func=0)
        scan_map.update(_raster(3, 2, 30.0, func=1))
        grid = _grid_from_scan_map(scan_map)

        grouped = grid.scans_by_function

        assert set(grouped) == {0, 1}
        assert len(grouped[0]) == 6
        assert grouped[0][0].laser_x_pos == pytest.approx(0.0)
        assert grouped[0][-1].laser_x_pos == pytest.approx(0.060)

    def test_unpositioned_scans_are_left_out(self):
        scan_map = _raster(3, 2, 30.0)
        scan_map[(0, 900)] = _scan(-1.0, -1.0)
        scan_map[(0, 901)] = _scan(float("nan"), float("nan"))
        grid = _grid_from_scan_map(scan_map)

        assert len(grid.scans_by_function[0]) == 6

    def test_a_function_with_no_positioned_scan_is_absent(self):
        scan_map = _raster(3, 2, 30.0, func=0)
        scan_map[(4, 0)] = _scan(-1.0, -1.0)
        grid = _grid_from_scan_map(scan_map)

        assert grid.scans_by_function.get(4, []) == []
