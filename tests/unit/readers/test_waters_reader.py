"""Tests for the Waters .raw MSI reader."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thyra.readers.waters.imaging_grid import ImagingGrid
from thyra.readers.waters.masslynx_lib import FunctionType, ScanInfoData
from thyra.readers.waters.waters_reader import WatersReader


def _make_scan_info(x_mm=0.1, y_mm=0.2, ms_level=1, has_pos=True, is_profile=0):
    """Helper to create a ScanInfoData with given laser position."""
    return ScanInfoData(
        ms_level=ms_level,
        polarity=0,
        drift_scan_count=0,
        is_profile=is_profile,
        precursor_mz=0.0,
        rt=1.0,
        laser_x_pos=x_mm if has_pos else -1.0,
        laser_y_pos=y_mm if has_pos else -1.0,
    )


def _make_imaging_grid():
    """Build a minimal 3x2 imaging grid for test fixtures."""
    # Simulate 3 x-positions and 2 y-positions in micrometers
    x_index_map = {100.0: 0, 200.0: 1, 300.0: 2}
    y_index_map = {50.0: 0, 150.0: 1}

    # Scan map: 1 MS function with 6 scans (one per pixel)
    scan_map = {}
    positions = [
        (0.1, 0.05),
        (0.2, 0.05),
        (0.3, 0.05),
        (0.1, 0.15),
        (0.2, 0.15),
        (0.3, 0.15),
    ]
    for scan, (x_mm, y_mm) in enumerate(positions):
        scan_map[(0, scan)] = _make_scan_info(x_mm, y_mm)

    return ImagingGrid(
        x_index_map=x_index_map,
        y_index_map=y_index_map,
        pixel_count_x=3,
        pixel_count_y=2,
        pixel_size_x=100.0,
        pixel_size_y=100.0,
        lateral_width=200.0,
        lateral_height=100.0,
        scan_map=scan_map,
    )


class TestWatersReaderValidation:
    """Test directory validation at construction time."""

    def test_rejects_nonexistent_path(self, tmp_path):
        """Test that a non-existent path raises ValueError."""
        bad_path = tmp_path / "missing.raw"
        with pytest.raises(ValueError, match="must be a directory"):
            WatersReader(bad_path)

    def test_rejects_file_instead_of_dir(self, tmp_path):
        """Test that a regular file with .raw extension is rejected."""
        fake = tmp_path / "test.raw"
        fake.touch()
        with pytest.raises(ValueError, match="must be a directory"):
            WatersReader(fake)

    def test_rejects_empty_raw_dir(self, tmp_path):
        """Test that a .raw dir without _FUNC*.DAT is rejected."""
        raw_dir = tmp_path / "empty.raw"
        raw_dir.mkdir()
        with pytest.raises(ValueError, match="No _FUNC.*DAT files found"):
            WatersReader(raw_dir)

    def test_accepts_valid_raw_dir(self, mock_waters_data):
        """Test that a .raw dir with _FUNC001.DAT passes validation."""
        # Construction should not raise (lazy init means no DLL calls)
        reader = WatersReader(mock_waters_data)
        assert reader.data_path == mock_waters_data


class TestWatersReaderInit:
    """Test lazy initialization and lifecycle."""

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_ensure_initialized_opens_file(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        """Test that _ensure_initialized loads DLL and opens file."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS
        mock_build_grid.return_value = _make_imaging_grid()

        reader = WatersReader(mock_waters_data)
        reader._ensure_initialized()

        mock_ml.open_file.assert_called_once()
        mock_ml.is_imaging_file.assert_called_once()
        mock_ml.set_centroid.assert_called_once()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    def test_rejects_non_imaging_file(self, mock_ml_cls, mock_waters_data):
        """Test error when DLL says file is not imaging."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = False

        reader = WatersReader(mock_waters_data)
        with pytest.raises(ValueError, match="not a Waters imaging file"):
            reader._ensure_initialized()

        # Should have cleaned up the handle
        mock_ml.close_file.assert_called_once()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    def test_rejects_no_ms_functions(self, mock_ml_cls, mock_waters_data):
        """Test error when no MS functions are found."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.LOCKMASS

        reader = WatersReader(mock_waters_data)
        with pytest.raises(ValueError, match="No MS functions found"):
            reader._ensure_initialized()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_second_init_is_noop(self, mock_build_grid, mock_ml_cls, mock_waters_data):
        """Test that calling _ensure_initialized twice doesn't re-open."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS
        mock_build_grid.return_value = _make_imaging_grid()

        reader = WatersReader(mock_waters_data)
        reader._ensure_initialized()
        reader._ensure_initialized()

        # open_file called only once
        assert mock_ml.open_file.call_count == 1


class TestWatersReaderClose:
    """Test close and cleanup behavior."""

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_close_releases_handle(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        """Test that close() calls close_file on the native library."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS
        mock_build_grid.return_value = _make_imaging_grid()

        reader = WatersReader(mock_waters_data)
        reader._ensure_initialized()
        reader.close()

        mock_ml.close_file.assert_called_once()
        assert reader._closed is True

    def test_close_before_init_is_safe(self, mock_waters_data):
        """Test that close() on an uninitialized reader doesn't error."""
        reader = WatersReader(mock_waters_data)
        reader.close()
        assert reader._closed is True

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_double_close_is_safe(self, mock_build_grid, mock_ml_cls, mock_waters_data):
        """Test that closing twice doesn't call close_file twice."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS
        mock_build_grid.return_value = _make_imaging_grid()

        reader = WatersReader(mock_waters_data)
        reader._ensure_initialized()
        reader.close()
        reader.close()

        assert mock_ml.close_file.call_count == 1

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_init_after_close_raises(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        """Test that re-initializing after close raises RuntimeError."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS
        mock_build_grid.return_value = _make_imaging_grid()

        reader = WatersReader(mock_waters_data)
        reader._ensure_initialized()
        reader.close()

        with pytest.raises(RuntimeError, match="closed"):
            reader._ensure_initialized()


class TestWatersReaderIterSpectra:
    """Test spectrum iteration."""

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_iter_spectra_yields_correct_data(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        """Test that iter_spectra yields (coords, mzs, intensities) tuples."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS

        grid = _make_imaging_grid()
        mock_build_grid.return_value = grid

        # Mock 6 scans, each returning a small spectrum
        mock_ml.get_number_of_scans_in_function.return_value = 6
        mzs = np.array([100.0, 200.0, 300.0])
        intensities = np.array([10.0, 20.0, 30.0])
        mock_ml.read_spectrum.return_value = (mzs, intensities)

        reader = WatersReader(mock_waters_data)
        spectra = list(reader.iter_spectra())

        assert len(spectra) == 6
        for coords, s_mzs, s_ints in spectra:
            assert len(coords) == 3
            assert coords[2] == 0  # z is always 0
            np.testing.assert_array_equal(s_mzs, mzs)
            np.testing.assert_array_equal(s_ints, intensities)

        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_iter_spectra_skips_no_position_scans(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        """Test that scans without laser position are skipped."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS

        # Grid with one positioned scan and one unpositioned
        scan_map = {
            (0, 0): _make_scan_info(0.1, 0.05, has_pos=True),
            (0, 1): _make_scan_info(has_pos=False),
        }
        grid = ImagingGrid(
            x_index_map={100.0: 0},
            y_index_map={50.0: 0},
            pixel_count_x=1,
            pixel_count_y=1,
            pixel_size_x=0.0,
            pixel_size_y=0.0,
            lateral_width=0.0,
            lateral_height=0.0,
            scan_map=scan_map,
        )
        mock_build_grid.return_value = grid

        mock_ml.get_number_of_scans_in_function.return_value = 2
        mock_ml.read_spectrum.return_value = (
            np.array([100.0]),
            np.array([50.0]),
        )

        reader = WatersReader(mock_waters_data)
        spectra = list(reader.iter_spectra())

        # Only 1 spectrum should be yielded (the one with a position)
        assert len(spectra) == 1
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_iter_spectra_skips_empty_spectra(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        """Test that spectra with no data points are skipped."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS

        grid = _make_imaging_grid()
        mock_build_grid.return_value = grid

        mock_ml.get_number_of_scans_in_function.return_value = 6

        # First spectrum has data, rest are empty
        empty = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
        full = (np.array([100.0, 200.0]), np.array([10.0, 20.0]))
        mock_ml.read_spectrum.side_effect = [full] + [empty] * 5

        reader = WatersReader(mock_waters_data)
        spectra = list(reader.iter_spectra())

        assert len(spectra) == 1
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_iter_spectra_continues_on_error(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        """Test that read errors for individual spectra are logged and skipped."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS

        grid = _make_imaging_grid()
        mock_build_grid.return_value = grid

        mock_ml.get_number_of_scans_in_function.return_value = 6

        good = (np.array([100.0]), np.array([50.0]))
        mock_ml.read_spectrum.side_effect = [
            RuntimeError("DLL error"),  # scan 0 fails
            good,  # scan 1 succeeds
            good,  # scan 2 succeeds
            good,  # scan 3 succeeds
            good,  # scan 4 succeeds
            good,  # scan 5 succeeds
        ]

        reader = WatersReader(mock_waters_data)
        spectra = list(reader.iter_spectra())

        assert len(spectra) == 5  # 6 scans minus 1 error
        reader.close()


class TestWatersReaderMassAxis:
    """Test common mass axis building."""

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_get_common_mass_axis(self, mock_build_grid, mock_ml_cls, mock_waters_data):
        """Test building common mass axis from multiple spectra."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS

        grid = _make_imaging_grid()
        mock_build_grid.return_value = grid

        mock_ml.get_number_of_scans_in_function.return_value = 6

        # Each scan has slightly different m/z values
        mock_ml.read_spectrum.side_effect = [
            (np.array([100.0, 200.0]), np.array([10.0, 20.0])),
            (np.array([100.0, 300.0]), np.array([15.0, 25.0])),
            (np.array([200.0, 300.0]), np.array([12.0, 22.0])),
            (np.array([100.0, 200.0]), np.array([10.0, 20.0])),
            (np.array([100.0, 300.0]), np.array([15.0, 25.0])),
            (np.array([200.0, 300.0]), np.array([12.0, 22.0])),
        ]

        reader = WatersReader(mock_waters_data)
        mass_axis = reader.get_common_mass_axis()

        expected = np.array([100.0, 200.0, 300.0])
        np.testing.assert_array_equal(mass_axis, expected)
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_mass_axis_is_cached(self, mock_build_grid, mock_ml_cls, mock_waters_data):
        """Test that the mass axis is cached after first call."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS

        grid = _make_imaging_grid()
        mock_build_grid.return_value = grid

        mock_ml.get_number_of_scans_in_function.return_value = 6
        mock_ml.read_spectrum.return_value = (
            np.array([100.0, 200.0]),
            np.array([10.0, 20.0]),
        )

        reader = WatersReader(mock_waters_data)
        axis1 = reader.get_common_mass_axis()
        axis2 = reader.get_common_mass_axis()

        assert axis1 is axis2  # Same object, not recomputed
        reader.close()


class TestWatersReaderProperties:
    """Test reader properties and repr."""

    def test_has_shared_mass_axis_is_false(self, mock_waters_data):
        """Test that Waters data reports no shared mass axis."""
        reader = WatersReader(mock_waters_data)
        assert reader.has_shared_mass_axis is False

    def test_repr_before_init(self, mock_waters_data):
        """Test repr before initialization."""
        reader = WatersReader(mock_waters_data)
        r = repr(reader)
        assert "WatersReader" in r
        assert "centroid=True" in r

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_repr_after_init(self, mock_build_grid, mock_ml_cls, mock_waters_data):
        """Test repr includes grid info after initialization."""
        mock_ml = MagicMock()
        mock_ml_cls.get_instance.return_value = mock_ml
        mock_ml.is_imaging_file.return_value = True
        mock_ml.get_number_of_functions.return_value = 1
        mock_ml.classify_function.return_value = FunctionType.MS
        mock_build_grid.return_value = _make_imaging_grid()

        reader = WatersReader(mock_waters_data)
        reader._ensure_initialized()
        r = repr(reader)
        assert "grid=3x2" in r
        reader.close()


def _two_function_grid(levels, precursors=None):
    """A 3x1 grid where every function records a scan at each of 3 pixels.

    ``levels`` maps function index -> MS level; ``precursors`` maps function
    index -> a precursor m/z per scan (a constant, or a list per scan).
    """
    precursors = precursors or {}
    positions = [(0.1, 0.05), (0.2, 0.05), (0.3, 0.05)]
    scan_map = {}
    for func, level in levels.items():
        per_scan = precursors.get(func, 0.0)
        for scan, (x_mm, y_mm) in enumerate(positions):
            mz = per_scan[scan] if isinstance(per_scan, list) else per_scan
            info = _make_scan_info(x_mm, y_mm, ms_level=level)
            scan_map[(func, scan)] = ScanInfoData(
                **{**info.__dict__, "precursor_mz": mz, "collision_energy": 25.0}
            )
    return ImagingGrid(
        x_index_map={100.0: 0, 200.0: 1, 300.0: 2},
        y_index_map={50.0: 0},
        pixel_count_x=3,
        pixel_count_y=1,
        pixel_size_x=100.0,
        pixel_size_y=100.0,
        lateral_width=200.0,
        lateral_height=0.0,
        scan_map=scan_map,
    )


def _chunked_grid(levels, scans_per_function=3, profile_functions=()):
    """A raster MassLynx split across functions, as the real files are.

    Function ``f`` records ``scans_per_function`` scans on pixels no other
    function visits, so the functions tile the stage instead of competing
    for it. ``levels`` maps function index -> the MS level MassLynx reports,
    which on a real chunked file is 1 for the first chunk, 2 for the middle
    ones and 0 for the last.
    """
    scan_map = {}
    x_index_map = {}
    for column, (func, scan) in enumerate(
        (f, s) for f in sorted(levels) for s in range(scans_per_function)
    ):
        x_mm = (column + 1) / 10.0
        x_index_map[round(x_mm * 1000.0, 2)] = column
        scan_map[(func, scan)] = _make_scan_info(
            x_mm,
            0.05,
            ms_level=levels[func],
            is_profile=1 if func in profile_functions else 0,
        )
    return ImagingGrid(
        x_index_map=x_index_map,
        y_index_map={50.0: 0},
        pixel_count_x=len(x_index_map),
        pixel_count_y=1,
        pixel_size_x=100.0,
        pixel_size_y=100.0,
        lateral_width=100.0 * (len(x_index_map) - 1),
        lateral_height=0.0,
        scan_map=scan_map,
    )


def _open_reader(
    mock_ml_cls,
    mock_build_grid,
    mock_waters_data,
    grid,
    n_funcs,
    types=None,
    scans_per_function=3,
    use_centroid=True,
):
    mock_ml = MagicMock()
    mock_ml_cls.get_instance.return_value = mock_ml
    mock_ml.is_imaging_file.return_value = True
    mock_ml.get_number_of_functions.return_value = n_funcs
    if types is None:
        mock_ml.classify_function.return_value = FunctionType.MS
    else:
        mock_ml.classify_function.side_effect = lambda _h, f: types[f]
    mock_ml.get_number_of_scans_in_function.return_value = scans_per_function
    mock_ml.read_spectrum.return_value = (
        np.array([100.0, 200.0]),
        np.array([1.0, 2.0]),
    )
    mock_build_grid.return_value = grid
    return WatersReader(mock_waters_data, use_centroid=use_centroid), mock_ml


class TestWatersReaderMsLevels:
    """One MS level per store: MS1 wins, MS/MS is recorded, not summed in."""

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_ms1_and_msms_functions_convert_the_ms1_only(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        grid = _two_function_grid({0: 1, 1: 2}, {1: 500.25})
        reader, mock_ml = _open_reader(
            mock_ml_cls, mock_build_grid, mock_waters_data, grid, 2
        )
        spectra = list(reader.iter_spectra())

        # Three pixels, one spectrum each, all read from function 0
        assert len(spectra) == 3
        assert {c for c, _, _ in spectra} == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
        assert {call.args[1] for call in mock_ml.read_spectrum.call_args_list} == {0}

        schedule = reader.get_fragmentation()
        assert schedule.ms_level == 1 and not schedule.is_msms

        excluded = reader._excluded_functions
        assert set(excluded) == {1}
        assert excluded[1]["ms_level"] == 2
        assert excluded[1]["precursor_mz"] == pytest.approx(500.25)
        assert excluded[1]["n_scans"] == 3

        block = reader._create_metadata_extractor()._extract_waters_specific()
        assert block["ms_functions"] == [0]
        assert block["excluded_functions"]["1"]["precursor_mz"] == pytest.approx(500.25)
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_msms_only_file_converts_and_reports_its_precursor(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        grid = _two_function_grid({0: 2}, {0: 760.585})
        reader, _ = _open_reader(
            mock_ml_cls, mock_build_grid, mock_waters_data, grid, 1
        )
        assert len(list(reader.iter_spectra())) == 3
        assert reader._excluded_functions == {}

        schedule = reader.get_fragmentation()
        assert schedule.ms_level == 2 and schedule.constant_across_pixels
        assert len(schedule.windows) == 1
        assert schedule.windows[0].target == pytest.approx(760.585)
        assert schedule.windows[0].collision_energy == pytest.approx(25.0)
        assert not schedule.merges_precursors
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_data_dependent_precursors_are_not_a_constant_schedule(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        grid = _two_function_grid({0: 2}, {0: [500.0, 600.0, 700.0]})
        reader, _ = _open_reader(
            mock_ml_cls, mock_build_grid, mock_waters_data, grid, 1
        )
        schedule = reader.get_fragmentation()
        assert schedule.ms_level == 2
        assert not schedule.constant_across_pixels
        assert schedule.windows == ()
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_two_ms1_functions_are_both_kept(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        grid = _two_function_grid({0: 1, 1: 1})
        reader, mock_ml = _open_reader(
            mock_ml_cls, mock_build_grid, mock_waters_data, grid, 2
        )
        spectra = list(reader.iter_spectra())
        assert len(spectra) == 6
        assert {call.args[1] for call in mock_ml.read_spectrum.call_args_list} == {0, 1}
        assert reader.get_fragmentation().ms_level == 1
        reader.close()


class TestWatersReaderChunkedRaster:
    """One raster split across functions: every chunk is part of the image.

    MassLynx caps a ``_FUNC*.DAT`` at about 1.6 GB and opens a new function
    when a long imaging run reaches it. On the real files the chunks come
    back as MS level 1, then 2, then 0, the last one is what
    ``getLockmassFunction`` names, and none of them carries a precursor --
    so neither the level nor the lockmass index can decide what to convert.
    The laser positions can: chunks tile the stage, parallel functions
    repeat it.
    """

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_every_chunk_of_a_split_raster_is_converted(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        grid = _chunked_grid({0: 1, 1: 2, 2: 0})
        reader, mock_ml = _open_reader(
            mock_ml_cls,
            mock_build_grid,
            mock_waters_data,
            grid,
            3,
            types={0: FunctionType.MS, 1: FunctionType.MS, 2: FunctionType.LOCKMASS},
        )
        spectra = list(reader.iter_spectra())

        # Nine pixels, one spectrum each, read from all three functions
        assert len(spectra) == 9
        assert len({c for c, _, _ in spectra}) == 9
        assert {call.args[1] for call in mock_ml.read_spectrum.call_args_list} == {
            0,
            1,
            2,
        }
        assert reader._ms_functions == [0, 1, 2]
        assert reader._excluded_functions == {}
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_a_level_two_chunk_is_not_a_fragmentation_schedule(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        grid = _chunked_grid({0: 1, 1: 2, 2: 0})
        reader, _ = _open_reader(
            mock_ml_cls,
            mock_build_grid,
            mock_waters_data,
            grid,
            3,
            types={0: FunctionType.MS, 1: FunctionType.MS, 2: FunctionType.LOCKMASS},
        )
        schedule = reader.get_fragmentation()
        assert schedule.ms_level == 1 and not schedule.is_msms
        assert schedule.windows == ()

        block = reader._create_metadata_extractor()._extract_waters_specific()
        # The block keeps MassLynx's own classification next to what was
        # converted, so the rescue is visible in the store.
        assert block["ms_functions"] == [0, 1, 2]
        assert block["function_types"]["2"] == "LOCKMASS"
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_a_lockmass_function_on_the_image_pixels_stays_out(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        # A reference function acquired alongside the image covers no pixel
        # the MS function does not already cover, so it is not rescued.
        grid = _two_function_grid({0: 1, 1: 0})
        reader, mock_ml = _open_reader(
            mock_ml_cls,
            mock_build_grid,
            mock_waters_data,
            grid,
            2,
            types={0: FunctionType.MS, 1: FunctionType.LOCKMASS},
        )
        reader._ensure_initialized()
        assert reader._ms_functions == [0]
        assert len(list(reader.iter_spectra())) == 3
        assert {call.args[1] for call in mock_ml.read_spectrum.call_args_list} == {0}
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_a_chunked_mse_run_keeps_the_ms1_function_of_every_chunk(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        # Two chunks, each an MSe pair: functions 0/1 on one set of pixels,
        # functions 2/3 on the next.
        grid = _chunked_grid({0: 1, 2: 1})
        for func, twin in ((0, 1), (2, 3)):
            for (f, scan), info in list(grid.scan_map.items()):
                if f != func:
                    continue
                grid.scan_map[(twin, scan)] = ScanInfoData(
                    **{**info.__dict__, "ms_level": 2, "precursor_mz": 0.0}
                )
        reader, mock_ml = _open_reader(
            mock_ml_cls, mock_build_grid, mock_waters_data, grid, 4
        )
        reader._ensure_initialized()
        assert reader._ms_functions == [0, 2]
        assert sorted(reader._excluded_functions) == [1, 3]
        assert len(list(reader.iter_spectra())) == 6
        assert reader.get_fragmentation().ms_level == 1
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_a_chunk_masslynx_will_not_centroid_stays_out_of_a_centroid_store(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        # MassLynx's centroider skips the function it names the lockmass
        # function, so that chunk comes back as profile whatever was asked.
        # Converting it anyway would band the top of the image.
        grid = _chunked_grid({0: 1, 1: 2, 2: 0}, profile_functions={2})
        reader, mock_ml = _open_reader(
            mock_ml_cls,
            mock_build_grid,
            mock_waters_data,
            grid,
            3,
            types={0: FunctionType.MS, 1: FunctionType.MS, 2: FunctionType.LOCKMASS},
        )
        assert len(list(reader.iter_spectra())) == 6
        assert reader._ms_functions == [0, 1]
        assert {call.args[1] for call in mock_ml.read_spectrum.call_args_list} == {0, 1}

        left_out = reader._excluded_functions[2]
        assert left_out["n_scans"] == 3
        assert left_out["n_unique_pixels"] == 3
        assert left_out["reason"] == "MassLynx will not centroid this function"
        reader.close()

    @patch("thyra.readers.waters.waters_reader.MassLynxLib")
    @patch("thyra.readers.waters.waters_reader.build_imaging_grid")
    def test_the_profile_read_converts_that_chunk_too(
        self, mock_build_grid, mock_ml_cls, mock_waters_data
    ):
        # Reading the trace asks for no centroiding, so every chunk comes
        # back the same way and the whole image can be converted.
        grid = _chunked_grid({0: 1, 1: 2, 2: 0}, profile_functions={0, 1, 2})
        reader, _ = _open_reader(
            mock_ml_cls,
            mock_build_grid,
            mock_waters_data,
            grid,
            3,
            types={0: FunctionType.MS, 1: FunctionType.MS, 2: FunctionType.LOCKMASS},
            use_centroid=False,
        )
        assert len(list(reader.iter_spectra())) == 9
        assert reader._ms_functions == [0, 1, 2]
        assert reader._excluded_functions == {}
        reader.close()
