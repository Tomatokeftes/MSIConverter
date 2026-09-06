"""What the TDF reader can tell about fragmentation, from the database alone.

``Frames.MsMsType`` says whether a frame fragmented anything -- 0 is a
survey scan, everything else is MS2 of some flavour -- and which table
carries the precursor depends on the flavour: PASEF frames list one row
per isolation window in ``PasefFrameMsMsInfo``, single-precursor frames
one row per frame in ``FrameMsMsInfo``.

None of this needs the vendor library, so these run everywhere: the
committed synthetic fixture is copied and its tables edited, which is
also the only way to exercise the MS/MS paths in CI. The numbers come
from two real MALDI acquisitions -- a single-precursor image at 1046.54
and a 15-window PASEF one -- so the shapes asserted here are shapes that
occur.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from thyra.readers.bruker.timstof.timstof_reader import BrukerReader

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "synthetic_tims.d"

# Three of the fifteen windows of 220425_MSMS_pos_brain1.d: disjoint
# slices of a 3572-scan ramp, each with its own collision energy.
_PASEF_ROWS = [
    (313.275, 1.0, 34.307, 2933, 3015),
    (353.320, 1.0, 35.172, 2832, 2901),
    (936.578, 1.0, 49.638, 994, 1127),
]

_PASEF_DDL = """
CREATE TABLE PasefFrameMsMsInfo (
    Frame INTEGER NOT NULL, ScanNumBegin INTEGER NOT NULL,
    ScanNumEnd INTEGER NOT NULL, IsolationMz REAL NOT NULL,
    IsolationWidth REAL NOT NULL, CollisionEnergy REAL NOT NULL,
    Precursor INTEGER)
"""


@pytest.fixture
def tdf(tmp_path) -> Path:
    """A writable copy of the committed synthetic acquisition."""
    target = tmp_path / "copy.d"
    shutil.copytree(FIXTURE, target)
    return target


def _frames(tdf: Path):
    return sqlite3.connect(tdf / "analysis.tdf")


def _make_pasef(tdf: Path, frames: list[int]) -> None:
    """Mark the given frames PASEF MS/MS and give them the window schedule."""
    with _frames(tdf) as conn:
        conn.execute("UPDATE Frames SET MsMsType = 8")
        conn.execute(_PASEF_DDL)
        conn.executemany(
            "INSERT INTO PasefFrameMsMsInfo (Frame, ScanNumBegin, ScanNumEnd, "
            "IsolationMz, IsolationWidth, CollisionEnergy, Precursor) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            [
                (frame, lo, hi, mz, width, ce)
                for frame in frames
                for mz, width, ce, lo, hi in _PASEF_ROWS
            ],
        )


def _read(tdf: Path):
    reader = BrukerReader(tdf, metadata_only=True)
    try:
        return reader.get_fragmentation()
    finally:
        reader.close()


def _all_frame_ids(tdf: Path) -> list[int]:
    with _frames(tdf) as conn:
        return [row[0] for row in conn.execute("SELECT Id FROM Frames ORDER BY Id")]


class TestSurveyFrames:
    def test_the_unmodified_fixture_is_ms1(self, tdf):
        """The committed fixture is a survey acquisition; saying so is a claim.

        ``ms_level=1`` here is not the same as the ``None`` a reader
        returns when it cannot tell -- ``MsMsType`` is present and says 0,
        so "this fragmented nothing" is known rather than assumed.
        """
        schedule = _read(tdf)

        assert schedule is not None
        assert schedule.ms_level == 1 and not schedule.is_msms
        assert schedule.windows == ()

    def test_a_database_without_msmstype_says_nothing(self, tdf):
        """A file the reader cannot interrogate must not be called MS1."""
        with _frames(tdf) as conn:
            conn.execute("ALTER TABLE Frames RENAME COLUMN MsMsType TO WasMsMsType")

        assert _read(tdf) is None


class TestSinglePrecursor:
    def test_one_trigger_mass_for_the_whole_image(self, tdf):
        """The shape of a classic MALDI MS/MS image: one precursor, every pixel."""
        with _frames(tdf) as conn:
            conn.execute("UPDATE Frames SET MsMsType = 2")
            conn.executemany(
                "INSERT INTO FrameMsMsInfo (Frame, Parent, TriggerMass, "
                "IsolationWidth, PrecursorCharge, CollisionEnergy) "
                "VALUES (?, NULL, 1046.54, 1.5, NULL, 57.327)",
                [(frame,) for frame in _all_frame_ids(tdf)],
            )

        schedule = _read(tdf)

        assert schedule.ms_level == 2 and schedule.is_msms
        assert len(schedule.windows) == 1
        assert not schedule.merges_precursors
        assert schedule.constant_across_pixels
        window = schedule.windows[0]
        assert window.target == 1046.54
        # 1.5 full width, halved into two offsets the way mzPeak stores it.
        assert window.lower_offset == 0.75 and window.upper_offset == 0.75
        assert window.collision_energy == 57.327
        assert not window.is_mobility_resolved


class TestPasef:
    def test_the_schedule_is_the_distinct_window_set(self, tdf):
        _make_pasef(tdf, _all_frame_ids(tdf))

        schedule = _read(tdf)

        assert schedule.ms_level == 2
        assert len(schedule.windows) == len(_PASEF_ROWS)
        assert [w.target for w in schedule.windows] == [313.275, 353.320, 936.578]
        assert all(w.is_mobility_resolved for w in schedule.windows)

    def test_several_windows_per_pixel_merge_precursors(self, tdf):
        """The fact a consumer needs: the stored spectrum is a chimera.

        Every pixel is fragmented on all three windows and summed into
        one spectrum, so its peaks belong to three different precursors
        with nothing marking which is which.
        """
        _make_pasef(tdf, _all_frame_ids(tdf))

        assert _read(tdf).merges_precursors

    def test_a_schedule_missing_from_some_frames_is_not_constant(self, tdf):
        """Constant-across-pixels is what makes the windows a global axis.

        Asserted because the grouped query cannot see it: the distinct
        window set looks identical either way, and only the per-window
        frame count distinguishes a scheduled method from one where the
        windows vary.
        """
        frames = _all_frame_ids(tdf)
        _make_pasef(tdf, frames[:-1])

        schedule = _read(tdf)

        assert len(schedule.windows) == len(_PASEF_ROWS)
        assert not schedule.constant_across_pixels

    def test_a_mixed_survey_and_fragment_file_is_not_constant(self, tdf):
        """One schedule per pixel is impossible when some frames are MS1."""
        frames = _all_frame_ids(tdf)
        _make_pasef(tdf, frames)
        with _frames(tdf) as conn:
            conn.execute("UPDATE Frames SET MsMsType = 0 WHERE Id = ?", (frames[0],))

        schedule = _read(tdf)

        assert schedule.ms_level == 2
        assert not schedule.constant_across_pixels

    def test_fragment_frames_without_a_precursor_table_record_the_level(self, tdf):
        """``MsMsType`` alone is still worth recording: the axis is fragment m/z."""
        with _frames(tdf) as conn:
            conn.execute("UPDATE Frames SET MsMsType = 8")

        schedule = _read(tdf)

        assert schedule.ms_level == 2 and schedule.is_msms
        assert schedule.windows == ()
        assert schedule.dissociation_accession is None


class TestCaching:
    def test_the_schedule_is_read_once(self, tdf):
        """A property of the method, not of a pixel; re-querying it is waste."""
        _make_pasef(tdf, _all_frame_ids(tdf))
        reader = BrukerReader(tdf, metadata_only=True)
        try:
            first = reader.get_fragmentation()
            assert reader.get_fragmentation() is first
        finally:
            reader.close()
