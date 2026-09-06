"""Splitting a PASEF frame into one spectrum per precursor.

The split is a filter on the scan number: each isolation window owns a
disjoint slice of the mobility ramp, so every ``(index, scan)`` pair the
frame recorded belongs to exactly one window or to none. These tests
drive that filter directly -- the committed synthetic acquisition is
copied and given a ``PasefFrameMsMsInfo`` schedule, and a stand-in for the
vendor library hands the reader a point cloud whose answer is known by
construction, so the whole path runs in CI with no SDK and no new data.

The window scan ranges are the real shape scaled onto the fixture's
240-scan ramp: disjoint, ascending in scan, descending in precursor m/z,
with gaps between them that no precursor claims. That inversion is
deliberate -- the schedule is ordered by precursor m/z, so ``window_index``
0 is the *last* slice of the ramp, and a test that confused the two orders
would pass on a schedule where they agree.
"""

import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from thyra.readers.bruker.timstof.timstof_reader import BrukerReader
from thyra.utils.bruker_exceptions import SDKError

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "synthetic_tims.d"

#: ``(IsolationMz, IsolationWidth, CollisionEnergy, ScanNumBegin, ScanNumEnd)``
PASEF_ROWS = [
    (936.578, 1.0, 49.638, 20, 60),
    (353.320, 1.0, 35.172, 80, 130),
    (313.275, 1.0, 34.307, 150, 200),
]

_PASEF_DDL = """
CREATE TABLE PasefFrameMsMsInfo (
    Frame INTEGER NOT NULL, ScanNumBegin INTEGER NOT NULL,
    ScanNumEnd INTEGER NOT NULL, IsolationMz REAL NOT NULL,
    IsolationWidth REAL NOT NULL, CollisionEnergy REAL NOT NULL,
    Precursor INTEGER)
"""

# (index, intensity, scan) triples every frame is given, ordered by scan
# the way tims_read_scans_v2 returns them. Two pairs share a window and a
# digitizer index (scans 25 and 30), one sits in the gap between two
# windows (scan 70), and one sits past the last window (scan 220).
POINTS = [
    (10, 100, 25),
    (10, 5, 30),
    (40, 70, 55),
    (99, 999, 70),
    (20, 200, 90),
    (30, 300, 125),
    (40, 400, 160),
    (10, 50, 199),
    (77, 777, 220),
]


class FakeSDK:
    """The two vendor calls the demultiplexer makes, over a fixed cloud."""

    def __init__(self, points=POINTS):
        self.points = points
        self.scan_ranges = []

    def read_tdf_scans(self, handle, frame_id, scan_begin, scan_end, hint=None):
        self.scan_ranges.append((frame_id, scan_begin, scan_end))
        indices, intensities, scans = (
            np.array([p[i] for p in self.points], dtype=dtype)
            for i, dtype in enumerate((np.uint32, np.uint32, np.int32))
        )
        return indices, intensities, scans

    def index_to_mz(self, handle, frame_id, indices):
        # Monotonic in the index, which is all the reader relies on.
        return 100.0 + np.asarray(indices, dtype=np.float64) * 0.5

    def close_file(self, handle):
        pass


@pytest.fixture
def tdf(tmp_path) -> Path:
    """A writable copy of the committed synthetic acquisition."""
    target = tmp_path / "copy.d"
    shutil.copytree(FIXTURE, target)
    return target


def _make_pasef(tdf: Path, rows=PASEF_ROWS) -> None:
    with sqlite3.connect(tdf / "analysis.tdf") as conn:
        conn.execute("UPDATE Frames SET MsMsType = 8")
        conn.execute(_PASEF_DDL)
        frames = [row[0] for row in conn.execute("SELECT Id FROM Frames")]
        conn.executemany(
            "INSERT INTO PasefFrameMsMsInfo (Frame, ScanNumBegin, ScanNumEnd, "
            "IsolationMz, IsolationWidth, CollisionEnergy, Precursor) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            [
                (frame, lo, hi, mz, width, ce)
                for frame in frames
                for mz, width, ce, lo, hi in rows
            ],
        )


def _reader(tdf: Path, sdk=None, **kwargs) -> BrukerReader:
    reader = BrukerReader(tdf, metadata_only=True, **kwargs)
    reader.sdk = FakeSDK() if sdk is None else sdk
    reader.handle = 1
    return reader


def _collect(reader):
    return list(reader.iter_precursor_spectra())


class TestTheSplit:
    def test_every_pixel_yields_every_window(self, tdf):
        _make_pasef(tdf)
        reader = _reader(tdf)
        try:
            yielded = _collect(reader)
        finally:
            reader.close()

        # Six frames in the fixture, three windows, all populated.
        assert len(yielded) == 18
        assert {w for _, w, _, _ in yielded} == {0, 1, 2}
        assert len({coords for coords, _, _, _ in yielded}) == 6

    def test_the_frame_is_read_once_over_the_whole_ramp(self, tdf):
        """Not once per window: the filter is on scans already in hand."""
        _make_pasef(tdf)
        sdk = FakeSDK()
        reader = _reader(tdf, sdk=sdk)
        try:
            _collect(reader)
        finally:
            reader.close()

        assert len(sdk.scan_ranges) == 6
        assert {(begin, end) for _, begin, end in sdk.scan_ranges} == {(0, 240)}

    def test_each_window_holds_only_its_own_scans(self, tdf):
        """The assertion the whole feature exists for.

        ``window_index`` indexes the schedule, which is ordered by
        precursor m/z: window 0 is 313.275 on scans 150..199, window 2 is
        936.578 on scans 20..59.
        """
        _make_pasef(tdf)
        reader = _reader(tdf)
        try:
            targets = [w.target for w in reader.get_fragmentation().windows]
            by_window = {
                w: (mzs, intensities)
                for coords, w, mzs, intensities in _collect(reader)
                if coords == (0, 0, 0)
            }
        finally:
            reader.close()

        assert targets == [313.275, 353.320, 936.578]
        # Window 0 is scans 150..199: indices 40 and 10, one point each.
        np.testing.assert_allclose(by_window[0][0], [105.0, 120.0])
        np.testing.assert_allclose(by_window[0][1], [50.0, 400.0])
        # Window 1 is scans 80..129: indices 20 and 30, one point each.
        np.testing.assert_allclose(by_window[1][0], [110.0, 115.0])
        np.testing.assert_allclose(by_window[1][1], [200.0, 300.0])
        # Window 2 is scans 20..59: index 10 twice (100 + 5), and index 40.
        np.testing.assert_allclose(by_window[2][0], [105.0, 120.0])
        np.testing.assert_allclose(by_window[2][1], [105.0, 70.0])

    def test_points_outside_every_window_go_nowhere(self, tdf):
        """A scan no precursor claims is dropped, never shared out."""
        _make_pasef(tdf)
        reader = _reader(tdf)
        try:
            total = sum(
                float(intensities.sum()) for _, _, _, intensities in _collect(reader)
            )
        finally:
            reader.close()

        in_window = sum(i for _, i, s in POINTS if s not in (70, 220))
        assert total == pytest.approx(in_window * 6)
        # The two orphans are the difference, and nothing carries them.
        assert total < sum(i for _, i, _ in POINTS) * 6

    def test_fragment_mz_is_ascending_within_a_window(self, tdf):
        _make_pasef(tdf)
        reader = _reader(tdf)
        try:
            for _, _, mzs, _ in _collect(reader):
                assert np.all(np.diff(mzs) > 0)
        finally:
            reader.close()

    def test_a_window_with_no_points_is_not_yielded(self, tdf):
        """An empty (pixel, precursor) pair is absent, not a row of zeros."""
        rows = PASEF_ROWS + [(500.0, 1.0, 40.0, 205, 215)]
        _make_pasef(tdf, rows)
        reader = _reader(tdf)
        try:
            n_windows = len(reader.get_fragmentation().windows)
            windows = {w for _, w, _, _ in _collect(reader)}
        finally:
            reader.close()

        # 500.0 sorts third of the four precursors, and its scans are empty.
        assert n_windows == 4
        assert windows == {0, 1, 3}

    def test_the_intensity_threshold_applies_to_the_window_sum(self, tdf):
        """Thresholded after the scans are summed, as for the whole ramp."""
        _make_pasef(tdf)
        reader = _reader(tdf, intensity_threshold=110.0)
        try:
            by_window = {
                w: intensities
                for coords, w, _, intensities in _collect(reader)
                if coords == (0, 0, 0)
            }
        finally:
            reader.close()

        # 100 and 5 in window 2 sum to 105, which is below the threshold,
        # and the 70 next to it is below it outright, so 936.578 drops out.
        assert 2 not in by_window
        np.testing.assert_allclose(by_window[0], [400.0])
        np.testing.assert_allclose(by_window[1], [200.0, 300.0])


class TestRefusals:
    def test_a_schedule_without_scan_ranges_cannot_be_split(self, tdf):
        """A single-precursor MALDI file reports no scan range at all."""
        with sqlite3.connect(tdf / "analysis.tdf") as conn:
            conn.execute("UPDATE Frames SET MsMsType = 2")
            conn.executemany(
                "INSERT INTO FrameMsMsInfo (Frame, Parent, TriggerMass, "
                "IsolationWidth, PrecursorCharge, CollisionEnergy) "
                "VALUES (?, NULL, 1046.54, 1.5, NULL, 57.327)",
                [(row[0],) for row in conn.execute("SELECT Id FROM Frames")],
            )
        reader = _reader(tdf)
        try:
            with pytest.raises(NotImplementedError, match="scan range"):
                _collect(reader)
        finally:
            reader.close()

    def test_an_unfragmented_file_has_nothing_to_split(self, tdf):
        reader = _reader(tdf)
        try:
            with pytest.raises(NotImplementedError, match="no isolation windows"):
                _collect(reader)
        finally:
            reader.close()

    def test_without_the_library_it_refuses_rather_than_returns_nothing(self, tdf):
        """Metadata-only mode must not look like an empty acquisition."""
        _make_pasef(tdf)
        reader = BrukerReader(tdf, metadata_only=True)
        try:
            with pytest.raises(SDKError, match="metadata-only"):
                _collect(reader)
        finally:
            reader.close()
