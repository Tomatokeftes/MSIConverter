"""BrukerReader's TDF-specific wiring, with the SDK and the database mocked."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thyra.core.mobility import MobilityAxis
from thyra.readers.bruker.timstof.timstof_reader import BrukerReader
from thyra.utils.bruker_exceptions import SDKError


def _only_tdf_exists(self: Path) -> bool:
    """``Path.exists`` that makes a fake ``.d`` look like a TDF acquisition."""
    return not str(self).endswith("analysis.tsf")


def _make_reader(file_type: str, **kwargs) -> tuple:
    exists = _only_tdf_exists if file_type == "tdf" else (lambda self: True)
    with (
        patch("thyra.readers.bruker.timstof.timstof_reader.DLLManager") as dll_manager,
        patch(
            "thyra.readers.bruker.timstof.timstof_reader.SDKFunctions"
        ) as sdk_functions,
        patch.object(Path, "exists", new=exists),
        patch.object(Path, "is_dir", return_value=True),
        patch("sqlite3.connect") as connect,
    ):
        dll_manager.return_value = MagicMock()
        sdk = MagicMock()
        sdk_functions.return_value = sdk
        sdk.open_file.return_value = 42
        sdk.read_spectrum.return_value = (np.array([100.0]), np.array([1.0]))
        connect.return_value = MagicMock()
        reader = BrukerReader(Path("/fake/bruker.d"), **kwargs)
    return reader, sdk, sdk_functions


class TestTdfWiring:
    def test_tdf_read_passes_the_frame_id_and_its_num_scans(self):
        reader, sdk, _ = _make_reader("tdf")
        assert reader.file_type == "tdf"
        reader._num_peaks_cache = {3: 500}
        reader._num_scans_cache = {3: 2382}

        reader._read_frame_spectrum(3)

        sdk.read_spectrum.assert_called_once_with(
            42, 3, buffer_size_hint=500, num_scans=2382
        )

    def test_num_scans_is_fetched_from_the_database_when_not_cached(self):
        reader, sdk, _ = _make_reader("tdf")
        reader._num_scans_cache = {}
        reader.conn = MagicMock()
        reader.conn.execute.return_value.fetchone.return_value = (9400,)

        reader._read_frame_spectrum(11)

        sdk.read_spectrum.assert_called_once_with(
            42, 11, buffer_size_hint=None, num_scans=9400
        )
        assert reader._num_scans_cache[11] == 9400

    def test_tsf_read_passes_no_num_scans(self):
        reader, sdk, _ = _make_reader("tsf")
        assert reader.file_type == "tsf"
        reader._num_peaks_cache = {3: 500}

        reader._read_frame_spectrum(3)

        sdk.read_spectrum.assert_called_once_with(
            42, 3, buffer_size_hint=500, num_scans=None
        )

    def test_tdf_spectrum_mode_reaches_the_sdk(self):
        reader, _, sdk_functions = _make_reader("tdf", tdf_spectrum="scan_sum")
        assert reader.tdf_spectrum == "scan_sum"
        sdk_functions.assert_called_once_with(
            reader.dll_manager, "tdf", tdf_spectrum="scan_sum"
        )

    def test_default_mode_is_the_scan_sum(self):
        # Design decision D1: the instrument's own record is the default,
        # the vendor centroid the opt-in.
        reader, _, _ = _make_reader("tdf")
        assert reader.tdf_spectrum == "scan_sum"

    def test_unknown_mode_is_rejected_before_touching_the_sdk(self):
        with pytest.raises(ValueError, match="tdf_spectrum"):
            _make_reader("tdf", tdf_spectrum="bogus")

    def test_tdf_reports_no_per_pixel_peak_counts(self):
        reader, _, _ = _make_reader("tdf")
        reader._num_peaks_cache = {1: 70000, 2: 80000}
        assert reader.get_peak_counts_per_pixel() is None

    def test_num_peaks_above_65535_are_kept(self):
        reader, _, _ = _make_reader("tdf")
        conn = MagicMock()
        cursor = conn.__enter__.return_value.cursor.return_value
        cursor.fetchall.return_value = [(1, 70000, 477), (2, 0, 477), (3, 96845, 477)]
        with patch("sqlite3.connect", return_value=conn):
            cache = reader._preload_frame_num_peaks()

        assert cache == {1: 70000, 3: 96845}
        assert reader._num_scans_cache == {1: 477, 2: 477, 3: 477}


def _mobility_db(reader, n_scans=2382, first_frame=1, k0=("1.000000", "1.290000")):
    """A ``conn`` answering the three queries the mobility axis needs."""

    def execute(sql, params=()):
        cursor = MagicMock()
        if "MAX(NumScans)" in sql:
            cursor.fetchone.return_value = (n_scans, first_frame)
        elif "OneOverK0AcqRange" in sql:
            cursor.fetchall.return_value = [
                ("OneOverK0AcqRangeLower", k0[0]),
                ("OneOverK0AcqRangeUpper", k0[1]),
            ]
        elif "TimsCalibration" in sql:
            cursor.fetchone.return_value = (1, 2, 1.0, 2381.0, None, 3.5)
            cursor.description = [
                ("Id",),
                ("ModelType",),
                ("C0",),
                ("C1",),
                ("C2",),
                ("C3",),
            ]
        elif "NumScans FROM Frames WHERE" in sql:
            cursor.fetchone.return_value = (n_scans,)
        else:
            raise AssertionError(f"unexpected query: {sql}")
        return cursor

    reader.conn = MagicMock()
    reader.conn.execute.side_effect = execute


def _axis(n_scans: int) -> MobilityAxis:
    return MobilityAxis(
        kind_accession="MS:1002815",
        kind_name="inverse reduced ion mobility",
        unit_accession="MS:1002814",
        unit_name="volt-second per square centimeter",
        values=2.0 - 0.001 * np.arange(n_scans, dtype=np.float64),
        source="bruker_tdf",
    )


class TestMobilityAxis:
    def test_tdf_has_mobility_and_tsf_does_not(self):
        assert _make_reader("tdf")[0].has_ion_mobility is True
        assert _make_reader("tsf")[0].has_ion_mobility is False
        assert _make_reader("tsf")[0].get_mobility_axis() is None
        # No shared (m/z, mobility) feature list: each pixel is its own cloud.
        assert _make_reader("tdf")[0].has_shared_mobility_axis is False

    def test_axis_comes_from_the_sdk_calibration_of_the_longest_ramp(self):
        reader, sdk, _ = _make_reader("tdf")
        _mobility_db(reader, n_scans=2382, first_frame=7)
        sdk.scannum_to_oneoverk0.return_value = np.linspace(1.29, 1.0, 2382)

        axis = reader.get_mobility_axis()

        handle, frame_id, scans = sdk.scannum_to_oneoverk0.call_args[0]
        assert (handle, frame_id) == (42, 7)
        np.testing.assert_array_equal(scans, np.arange(2382, dtype=np.float64))
        assert axis.kind_accession == "MS:1002815"
        assert axis.unit_accession == "MS:1002814"
        assert axis.values.size == 2382 and np.all(np.diff(axis.values) < 0)
        assert axis.acq_range == (1.0, 1.29)
        assert axis.calibration == {
            "coefficients": [1.0, 2381.0, float("nan"), 3.5],
        } or (
            axis.calibration["model_type"] == 2
            and axis.calibration["coefficients"][:2] == [1.0, 2381.0]
            and np.isnan(axis.calibration["coefficients"][2])
            and axis.calibration["coefficients"][3] == 3.5
        )
        assert axis.source == "bruker_tdf"

    def test_axis_is_read_once(self):
        reader, sdk, _ = _make_reader("tdf")
        _mobility_db(reader)
        sdk.scannum_to_oneoverk0.return_value = np.linspace(1.29, 1.0, 2382)
        assert reader.get_mobility_axis() is reader.get_mobility_axis()
        assert sdk.scannum_to_oneoverk0.call_count == 1

    def test_uns_block_uses_the_contract_names_and_no_colons(self):
        reader, sdk, _ = _make_reader("tdf")
        _mobility_db(reader)
        sdk.scannum_to_oneoverk0.return_value = np.linspace(1.29, 1.0, 2382)
        block = reader.get_mobility_axis().to_uns()
        assert block["n_scans"] == 2382
        np.testing.assert_array_equal(block["acq_range"], [1.0, 1.29])
        assert block["calibration"]["model_type"] == 2
        assert block["source"] == "bruker_tdf"
        assert not any(":" in key for key in block)


class TestMobilityIteration:
    def _prime(self, reader, sdk, frames):
        reader._iter_frames = lambda: iter(frames)
        reader._num_scans_cache = {frame_id: 2382 for frame_id, _ in frames}
        reader._num_peaks_cache = {frame_id: 500 for frame_id, _ in frames}
        reader._mobility_axis = _axis(2382)
        sdk.index_to_mz.side_effect = lambda h, f, idx: 100.0 + 0.001 * np.asarray(idx)

    def test_yields_the_raw_cloud_with_the_frame_id_as_is(self):
        reader, sdk, _ = _make_reader("tdf")
        self._prime(reader, sdk, [(3, (0, 0, 0)), (4, (1, 0, 0))])
        sdk.read_tdf_scans.return_value = (
            np.array([10, 20, 10], dtype=np.uint32),
            np.array([5, 6, 7], dtype=np.uint32),
            np.array([0, 0, 2], dtype=np.int32),
        )

        out = list(reader.iter_mobility_spectra())

        assert [coords for coords, *_ in out] == [(0, 0, 0), (1, 0, 0)]
        sdk.read_tdf_scans.assert_any_call(42, 3, 0, 2382, 500)
        sdk.read_tdf_scans.assert_any_call(42, 4, 0, 2382, 500)
        coords, mzs, mobility, intensities = out[0]
        np.testing.assert_allclose(mzs, [100.010, 100.020, 100.010])
        np.testing.assert_allclose(mobility, [2.0, 2.0, 2.0 - 0.002])
        np.testing.assert_array_equal(intensities, [5.0, 6.0, 7.0])
        assert intensities.dtype == np.float64 and mobility.dtype == np.float64
        assert mzs.shape == mobility.shape == intensities.shape

    def test_index_to_mz_sees_only_the_unique_indices(self):
        reader, sdk, _ = _make_reader("tdf")
        self._prime(reader, sdk, [(3, (0, 0, 0))])
        sdk.read_tdf_scans.return_value = (
            np.array([10, 20, 10, 10], dtype=np.uint32),
            np.array([1, 1, 1, 1], dtype=np.uint32),
            np.array([0, 0, 1, 2], dtype=np.int32),
        )
        list(reader.iter_mobility_spectra())
        (handle, frame_id, indices), _ = sdk.index_to_mz.call_args
        np.testing.assert_array_equal(indices, [10.0, 20.0])

    def test_intensity_threshold_masks_all_three_arrays(self):
        reader, sdk, _ = _make_reader("tdf", intensity_threshold=6.0)
        self._prime(reader, sdk, [(3, (0, 0, 0))])
        sdk.read_tdf_scans.return_value = (
            np.array([10, 20, 30], dtype=np.uint32),
            np.array([5, 6, 7], dtype=np.uint32),
            np.array([0, 1, 2], dtype=np.int32),
        )
        _, mzs, mobility, intensities = next(reader.iter_mobility_spectra())
        np.testing.assert_allclose(mzs, [100.020, 100.030])
        np.testing.assert_allclose(mobility, [2.0 - 0.001, 2.0 - 0.002])
        np.testing.assert_array_equal(intensities, [6.0, 7.0])

    def test_empty_frames_are_skipped_and_failures_logged(self, caplog):
        reader, sdk, _ = _make_reader("tdf")
        self._prime(reader, sdk, [(3, (0, 0, 0)), (4, (1, 0, 0)), (5, (2, 0, 0))])
        empty = (
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.int32),
        )
        good = (
            np.array([10], dtype=np.uint32),
            np.array([5], dtype=np.uint32),
            np.array([0], dtype=np.int32),
        )
        sdk.read_tdf_scans.side_effect = [empty, RuntimeError("boom"), good]
        out = list(reader.iter_mobility_spectra())
        assert [coords for coords, *_ in out] == [(2, 0, 0)]
        assert "frame 4" in caplog.text

    def test_scans_beyond_the_axis_are_clipped_with_one_warning(self, caplog):
        reader, sdk, _ = _make_reader("tdf")
        self._prime(reader, sdk, [(3, (0, 0, 0)), (4, (1, 0, 0))])
        reader._mobility_axis = _axis(10)
        sdk.read_tdf_scans.return_value = (
            np.array([10], dtype=np.uint32),
            np.array([5], dtype=np.uint32),
            np.array([12], dtype=np.int32),
        )
        out = list(reader.iter_mobility_spectra())
        assert out[0][2][0] == pytest.approx(2.0 - 0.009)
        assert caplog.text.count("beyond the 10-scan mobility axis") == 1

    def test_tsf_has_nothing_to_iterate(self):
        reader, _, _ = _make_reader("tsf")
        with pytest.raises(NotImplementedError, match="TSF"):
            next(reader.iter_mobility_spectra())

    def test_metadata_only_mode_needs_the_library(self):
        reader, _, _ = _make_reader("tdf")
        reader.sdk = None
        reader.handle = None
        with pytest.raises(SDKError, match="metadata-only"):
            next(reader.iter_mobility_spectra())


class TestSummedIterationSharesTheFrameLoop:
    def test_iter_spectra_goes_through_iter_frames(self):
        reader, sdk, _ = _make_reader("tdf")
        reader._iter_frames = lambda: iter([(3, (0, 0, 0)), (4, (1, 0, 0))])
        reader._num_scans_cache = {3: 2382, 4: 2382}
        reader._num_peaks_cache = {}
        sdk.read_spectrum.side_effect = [
            (np.array([100.0]), np.array([1.0])),
            (np.array([]), np.array([])),
        ]
        out = list(reader.iter_spectra())
        assert [coords for coords, *_ in out] == [(0, 0, 0)]
        sdk.read_spectrum.assert_any_call(42, 3, buffer_size_hint=None, num_scans=2382)

    def test_a_failing_read_skips_only_that_frame(self, caplog):
        reader, sdk, _ = _make_reader("tdf")
        reader._iter_frames = lambda: iter([(3, (0, 0, 0)), (4, (1, 0, 0))])
        reader._num_scans_cache = {3: 2382, 4: 2382}
        reader._num_peaks_cache = {}
        sdk.read_spectrum.side_effect = [
            RuntimeError("boom"),
            (np.array([100.0]), np.array([1.0])),
        ]
        out = list(reader.iter_spectra())
        assert [coords for coords, *_ in out] == [(1, 0, 0)]
        assert "frame 3" in caplog.text
