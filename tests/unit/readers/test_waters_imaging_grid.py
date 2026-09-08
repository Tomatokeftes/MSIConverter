"""Tests for Waters imaging grid reconstruction.

These tests cover ImagingGrid, _mm_to_um_key, and build_imaging_grid
without requiring native DLLs -- MassLynxLib is mocked.
"""

from unittest.mock import MagicMock

import pytest

from thyra.readers.waters.imaging_grid import (
    ImagingGrid,
    _mm_to_um_key,
    build_imaging_grid,
)
from thyra.readers.waters.masslynx_lib import FunctionType, ScanInfoData


def _make_scan_info(x_mm, y_mm, has_pos=True):
    """Create a ScanInfoData with given laser position."""
    return ScanInfoData(
        ms_level=1,
        polarity=0,
        drift_scan_count=0,
        is_profile=0,
        precursor_mz=0.0,
        rt=1.0,
        laser_x_pos=x_mm if has_pos else -1.0,
        laser_y_pos=y_mm if has_pos else -1.0,
    )


class TestMmToUmKey:
    """Test the mm-to-um conversion helper."""

    def test_basic_conversion(self):
        """Test that mm values are converted to um with rounding."""
        assert _mm_to_um_key(1.0) == 1000.0
        assert _mm_to_um_key(0.1) == 100.0
        assert _mm_to_um_key(0.001) == 1.0

    def test_rounding(self):
        """Test that values are rounded to 2 decimal places."""
        # 0.1234 mm = 123.4 um -> rounded to 123.4
        assert _mm_to_um_key(0.1234) == 123.4

    def test_zero(self):
        """Test zero value."""
        assert _mm_to_um_key(0.0) == 0.0

    def test_negative(self):
        """Test negative values (sentinel positions)."""
        assert _mm_to_um_key(-1.0) == -1000.0


class TestScanInfoData:
    """Test the ScanInfoData dataclass."""

    def test_has_position_true(self):
        """Test has_position for valid coordinates."""
        info = _make_scan_info(0.5, 0.3)
        assert info.has_position is True

    def test_has_position_false(self):
        """Test has_position for sentinel -1.0 coordinates."""
        info = _make_scan_info(0.0, 0.0, has_pos=False)
        assert info.has_position is False

    def test_has_position_partial_sentinel(self):
        """Test has_position when only one coord is sentinel."""
        # Both must be -1.0 for has_position to be False
        info = ScanInfoData(
            ms_level=1,
            polarity=0,
            drift_scan_count=0,
            is_profile=0,
            precursor_mz=0.0,
            rt=1.0,
            laser_x_pos=-1.0,
            laser_y_pos=0.5,
        )
        assert info.has_position is True


class TestImagingGrid:
    """Test ImagingGrid coordinate mapping."""

    def _make_grid(self):
        """Create a 3x2 grid for testing."""
        x_map = {100.0: 0, 200.0: 1, 300.0: 2}
        y_map = {50.0: 0, 150.0: 1}
        return ImagingGrid(
            x_index_map=x_map,
            y_index_map=y_map,
            pixel_count_x=3,
            pixel_count_y=2,
            pixel_size_x=100.0,
            pixel_size_y=100.0,
            lateral_width=200.0,
            lateral_height=100.0,
            scan_map={},
        )

    def test_dimensions(self):
        """Test that dimensions returns (x, y, 1)."""
        grid = self._make_grid()
        assert grid.dimensions == (3, 2, 1)

    def test_get_coordinates_valid(self):
        """Test coordinate mapping for a known position."""
        grid = self._make_grid()
        # 0.1 mm = 100.0 um, 0.05 mm = 50.0 um
        info = _make_scan_info(0.1, 0.05)
        coords = grid.get_coordinates(info)
        assert coords == (0, 0, 0)

    def test_get_coordinates_middle(self):
        """Test coordinate mapping for a middle position."""
        grid = self._make_grid()
        # 0.2 mm = 200.0 um, 0.15 mm = 150.0 um
        info = _make_scan_info(0.2, 0.15)
        coords = grid.get_coordinates(info)
        assert coords == (1, 1, 0)

    def test_get_coordinates_last(self):
        """Test coordinate mapping for the last position."""
        grid = self._make_grid()
        # 0.3 mm = 300.0 um, 0.15 mm = 150.0 um
        info = _make_scan_info(0.3, 0.15)
        coords = grid.get_coordinates(info)
        assert coords == (2, 1, 0)

    def test_get_coordinates_no_position(self):
        """Test that scans without position return None."""
        grid = self._make_grid()
        info = _make_scan_info(0.0, 0.0, has_pos=False)
        assert grid.get_coordinates(info) is None

    def test_get_coordinates_unknown_position(self):
        """Test that unknown laser positions return None."""
        grid = self._make_grid()
        # 0.999 mm = 999.0 um -- not in the grid
        info = _make_scan_info(0.999, 0.999)
        assert grid.get_coordinates(info) is None


class TestBuildImagingGrid:
    """Test the build_imaging_grid function with mocked MassLynxLib."""

    def _setup_mock_ml(self, positions):
        """Create a mock MassLynxLib that returns given laser positions.

        Args:
            positions: List of (func, scan, x_mm, y_mm) tuples.
                       Use x_mm=-1.0, y_mm=-1.0 for scans without position.
        """
        mock_ml = MagicMock()

        # Group by function to determine function count and scan counts
        func_scans = {}
        for func, scan, x_mm, y_mm in positions:
            func_scans.setdefault(func, []).append((scan, x_mm, y_mm))

        n_functions = max(f for f, _, _, _ in positions) + 1
        mock_ml.get_number_of_functions.return_value = n_functions

        def get_n_scans(handle, func):
            return len(func_scans.get(func, []))

        mock_ml.get_number_of_scans_in_function.side_effect = get_n_scans

        def get_scan_info(handle, func, scan):
            for f, s, x, y in positions:
                if f == func and s == scan:
                    return _make_scan_info(x, y, has_pos=(x != -1.0))
            return _make_scan_info(0, 0, has_pos=False)

        mock_ml.get_scan_info.side_effect = get_scan_info

        return mock_ml

    def test_basic_3x2_grid(self):
        """Test building a 3x2 grid from 6 positioned scans."""
        positions = [
            (0, 0, 0.1, 0.05),
            (0, 1, 0.2, 0.05),
            (0, 2, 0.3, 0.05),
            (0, 3, 0.1, 0.15),
            (0, 4, 0.2, 0.15),
            (0, 5, 0.3, 0.15),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        grid = build_imaging_grid(mock_ml, "handle", func_types)

        assert grid.pixel_count_x == 3
        assert grid.pixel_count_y == 2
        assert grid.lateral_width == 200.0  # 300 - 100 um
        assert grid.lateral_height == 100.0  # 150 - 50 um
        assert len(grid.scan_map) == 6

    def test_skips_scans_without_position(self):
        """Test that scans with sentinel positions are excluded from the grid."""
        positions = [
            (0, 0, 0.1, 0.05),
            (0, 1, -1.0, -1.0),  # no position
            (0, 2, 0.2, 0.05),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        grid = build_imaging_grid(mock_ml, "handle", func_types)

        assert grid.pixel_count_x == 2  # only 0.1 and 0.2
        assert grid.pixel_count_y == 1
        assert len(grid.scan_map) == 3  # all scans in map (including no-pos)

    def test_single_pixel_grid_is_refused(self):
        """A stage that never moved is a spot acquisition, not an image.

        This used to return a 1x1 grid with a 0.0 um pitch and let the
        conversion report success with the whole acquisition on one pixel
        (issue #213). On the Xevo DESI share that is what 76 of 105 .raw
        dirs look like -- calibrations, tuning runs, single-spot DDA.
        """
        positions = [(0, 0, 0.5, 0.5), (0, 1, 0.5, 0.5), (0, 2, 0.5, 0.5)]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        with pytest.raises(ValueError, match="same stage position"):
            build_imaging_grid(mock_ml, "handle", func_types)

    def test_single_row_grid_is_kept(self):
        """One line of a raster is real data, and must survive the refusal.

        The share holds aborted line scans (138x1, 48x1) alongside the
        completed rasters they were retried into. Only a grid that
        collapses on *both* axes is a spot.
        """
        positions = [
            (0, 0, 0.1, 0.05),
            (0, 1, 0.2, 0.05),
            (0, 2, 0.3, 0.05),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        grid = build_imaging_grid(mock_ml, "handle", func_types)

        assert grid.pixel_count_x == 3
        assert grid.pixel_count_y == 1

    def test_no_valid_positions_raises(self):
        """Test error when no scans have valid positions."""
        positions = [
            (0, 0, -1.0, -1.0),
            (0, 1, -1.0, -1.0),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        with pytest.raises(ValueError, match="No valid laser positions"):
            build_imaging_grid(mock_ml, "handle", func_types)

    def test_multiple_functions(self):
        """Test grid built from multiple acquisition functions."""
        positions = [
            (0, 0, 0.1, 0.05),
            (0, 1, 0.2, 0.05),
            (1, 0, 0.1, 0.15),  # same x, different y
            (1, 1, 0.2, 0.15),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS, 1: FunctionType.MS}

        grid = build_imaging_grid(mock_ml, "handle", func_types)

        assert grid.pixel_count_x == 2
        assert grid.pixel_count_y == 2
        assert len(grid.scan_map) == 4

    def test_pixel_size_calculation(self):
        """The pitch is the extent over the *intervals*, not the positions.

        Four positions 100 um apart are a 100 um raster. Dividing the
        300 um centre-to-centre span by the position count reported 75 um
        (issue #217); it is 300 / 3.
        """
        positions = [
            (0, 0, 0.1, 0.1),
            (0, 1, 0.2, 0.1),
            (0, 2, 0.3, 0.1),
            (0, 3, 0.4, 0.1),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        grid = build_imaging_grid(mock_ml, "handle", func_types)

        assert grid.lateral_width == 300.0
        assert grid.pixel_size_x == 100.0  # 300 / (4 - 1)

    def test_a_square_raster_is_not_reported_as_anisotropic(self):
        """The old error was 1/N per axis, so it invented anisotropy.

        A 100 um raster three wide and two tall came out 66.67 x 50.0.
        Only x reaches the store, so the disagreement never showed.
        """
        positions = [
            (0, 0, 0.1, 0.1),
            (0, 1, 0.2, 0.1),
            (0, 2, 0.3, 0.1),
            (0, 3, 0.1, 0.2),
            (0, 4, 0.2, 0.2),
            (0, 5, 0.3, 0.2),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        grid = build_imaging_grid(mock_ml, "handle", func_types)

        assert grid.pixel_size_x == 100.0
        assert grid.pixel_size_y == 100.0

    def test_an_axis_with_one_position_reports_no_pitch(self):
        """One position has no interval, so 0.0 rather than a made-up number.

        The extractor turns that into ``pixel_size=None``, which becomes
        the "use --pixel-size" refusal. Aborted single-line scans take
        this path, so it must not become a division by zero.
        """
        positions = [
            (0, 0, 0.1, 0.05),
            (0, 1, 0.2, 0.05),
        ]
        mock_ml = self._setup_mock_ml(positions)
        func_types = {0: FunctionType.MS}

        grid = build_imaging_grid(mock_ml, "handle", func_types)

        assert grid.pixel_size_x == 100.0
        assert grid.pixel_size_y == 0.0
