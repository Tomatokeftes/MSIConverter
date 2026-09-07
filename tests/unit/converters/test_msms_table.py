"""The demultiplexed MS/MS sibling table, built from a stand-in reader.

``build_msms_table`` needs two things from a reader: the precursor
schedule and one spectrum per (pixel, precursor). Both are stubbed here,
so these tests pin the table's contract -- the feature axis, the column
order, the row mirror and the refusals -- without a vendor library, real
data or a conversion.

The refusals are the point of the module as much as the table is: an
acquisition whose precursors cannot be told apart must produce no table
and a reason, never an apportioned guess.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("spatialdata")

from thyra.converters.spatialdata.msms_table import (  # noqa: E402
    build_msms_table,
    demultiplex_refusal,
    msms_table_key,
)
from thyra.core.msms import FragmentationSchedule, IsolationWindow  # noqa: E402

AXIS = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

# Two precursors on disjoint slices of the ramp, listed in the order the
# reader reports them: ascending precursor m/z.
WINDOWS = (
    IsolationWindow(313.275, 0.5, 0.5, 34.3, scan_begin=150, scan_end=200),
    IsolationWindow(936.578, 0.5, 0.5, 49.6, scan_begin=20, scan_end=60),
)
SCHEDULE = FragmentationSchedule(ms_level=2, windows=WINDOWS, source="bruker_tdf")

# (coords, window_index, mzs, intensities) as the reader would yield them.
SPECTRA = [
    ((0, 0, 0), 0, [100.0, 300.0], [10.0, 20.0]),
    ((0, 0, 0), 1, [200.0], [30.0]),
    ((1, 0, 0), 0, [300.0], [40.0]),
    ((1, 0, 0), 1, [200.0, 500.0], [50.0, 60.0]),
]

OBS = pd.DataFrame(
    {"x": [0, 1], "y": [0, 0]},
    index=pd.Index(["0", "1"], name="instance_id"),
)


#: A decreasing 1/K0 ramp, as a TIMS axis really is.
RAMP = np.linspace(1.5, 0.6, 240)


def _reader(schedule=SCHEDULE, spectra=SPECTRA, ramp=None):
    def iter_precursor_spectra(batch_size=None):
        for coords, window, mzs, intensities in spectra:
            yield (
                coords,
                window,
                np.asarray(mzs, dtype=np.float64),
                np.asarray(intensities, dtype=np.float64),
            )

    return SimpleNamespace(
        get_fragmentation=lambda: schedule,
        iter_precursor_spectra=iter_precursor_spectra,
        get_mobility_axis=lambda: (
            None if ramp is None else SimpleNamespace(values=ramp)
        ),
    )


def _build(reader=None, obs=None, axis=AXIS):
    return build_msms_table(
        reader if reader is not None else _reader(),
        OBS if obs is None else obs,
        axis,
        "msi_z0",
        "msi_z0_pixels",
        {"provenance": "unchanged"},
    )


def _dense(table):
    X = table.X
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X)


class TestTheFeatureAxis:
    def test_var_is_the_observed_precursor_fragment_pairs(self):
        var = _build().var

        np.testing.assert_array_equal(
            var["precursor_mz"].to_numpy(), [313.275, 313.275, 936.578, 936.578]
        )
        np.testing.assert_array_equal(
            var["mz"].to_numpy(), [100.0, 300.0, 200.0, 500.0]
        )
        np.testing.assert_array_equal(var["precursor_index"].to_numpy(), [0, 0, 1, 1])
        np.testing.assert_array_equal(var["mz_index"].to_numpy(), [0, 2, 1, 4])
        assert list(var.index) == [
            "p313.275_mz0",
            "p313.275_mz2",
            "p936.578_mz1",
            "p936.578_mz4",
        ]

    def test_feature_labels_name_the_precursor_not_its_rank(self):
        """A rank means nothing outside one store; a label must survive concat.

        Two samples whose schedules differ in length would give the same
        rank to different precursors, and ``anndata.concat`` aligns on
        these labels -- so a positional label merges two unrelated
        precursors into one column without an error.
        """
        short = FragmentationSchedule(
            ms_level=2,
            windows=(
                IsolationWindow(500.0, scan_begin=100, scan_end=140),
                IsolationWindow(936.578, scan_begin=20, scan_end=60),
            ),
        )
        spectra = [((0, 0, 0), 1, [200.0], [5.0])]
        other = _build(_reader(schedule=short, spectra=spectra))

        # 936.578 is rank 1 here and rank 1 in SPECTRA's schedule too, but
        # nothing in the label depends on that.
        assert list(other.var.index) == ["p936.578_mz1"]
        assert "p936.578_mz1" in list(_build().var.index)

    def test_each_precursor_is_one_contiguous_column_block(self):
        """What makes a per-precursor ion image a slice, not a gather."""
        var = _build().var
        precursor = var["precursor_index"].to_numpy()

        assert np.all(np.diff(precursor) >= 0)
        for index in np.unique(precursor):
            block = np.flatnonzero(precursor == index)
            assert np.array_equal(block, np.arange(block[0], block[-1] + 1))
            mz = var["mz"].to_numpy()[block]
            assert np.all(np.diff(mz) > 0)

    def test_no_mobility_column(self):
        """A consumer must never mistake this for a mobility-resolved table."""
        assert "mobility" not in _build().var.columns

    def test_column_dtypes_are_the_frozen_contract(self):
        var = _build().var
        assert var["precursor_mz"].dtype == np.float64
        assert var["mz"].dtype == np.float64
        assert var["precursor_index"].dtype == np.int64
        assert var["mz_index"].dtype == np.int64


class TestTheValues:
    def test_intensities_land_on_their_precursor_and_fragment(self):
        np.testing.assert_array_equal(
            _dense(_build()),
            [
                [10.0, 20.0, 30.0, 0.0],
                [0.0, 40.0, 50.0, 60.0],
            ],
        )

    def test_the_split_conserves_every_pixel_total(self):
        """Nothing dropped, nothing double counted: the point of the split."""
        table = _build()
        expected = {}
        for coords, _window, _mzs, intensities in SPECTRA:
            expected[coords[0]] = expected.get(coords[0], 0.0) + sum(intensities)

        np.testing.assert_allclose(
            _dense(table).sum(axis=1), [expected[0], expected[1]]
        )

    def test_fragments_off_the_mass_axis_are_dropped_not_clipped(self):
        """The summed table's rule: an edge bin must not collect strays."""
        spectra = [((0, 0, 0), 0, [50.0, 300.0, 900.0], [7.0, 20.0, 9.0])]
        table = _build(_reader(spectra=spectra))

        np.testing.assert_array_equal(table.var["mz"].to_numpy(), [300.0])
        np.testing.assert_allclose(_dense(table), [[20.0], [0.0]])

    def test_a_fragment_between_bins_lands_on_the_nearest(self):
        spectra = [((0, 0, 0), 0, [297.0], [5.0])]
        table = _build(_reader(spectra=spectra))

        np.testing.assert_array_equal(table.var["mz_index"].to_numpy(), [2])


class TestTheRowMirror:
    def test_obs_is_the_msi_tables_obs(self):
        table = _build()

        assert list(table.obs.index) == list(OBS.index)
        np.testing.assert_array_equal(table.obs["x"].to_numpy(), OBS["x"].to_numpy())
        assert set(table.obs["region"].astype(str)) == {"msi_z0_pixels"}

    def test_a_pixel_with_no_row_is_skipped(self):
        spectra = SPECTRA + [((9, 9, 0), 0, [100.0], [1.0])]
        table = _build(_reader(spectra=spectra))

        assert table.n_obs == 2
        np.testing.assert_allclose(
            _dense(table).sum(), sum(sum(i) for _c, _w, _m, i in SPECTRA)
        )

    def test_uns_carries_the_provenance_and_the_feature_axis(self):
        uns = _build(_reader(ramp=RAMP)).uns

        assert uns["provenance"] == "unchanged"
        assert json.loads(uns["feature_axis"]["dims"]) == [
            "precursor_mz",
            "precursor_mobility",
            "mz",
        ]
        assert uns["feature_axis"]["summed_table"] == "msi_z0"


class TestTheDescriptorMatchesTheColumns:
    def test_with_a_ramp_the_mobility_dimension_is_named(self):
        table = _build(_reader(ramp=RAMP))
        assert "precursor_mobility" in table.var.columns
        assert json.loads(table.uns["feature_axis"]["dims"]) == [
            "precursor_mz",
            "precursor_mobility",
            "mz",
        ]

    def test_without_a_ramp_it_is_not(self):
        # A descriptor naming a column the table does not carry is a lie
        # a consumer could act on.
        table = _build()
        assert "precursor_mobility" not in table.var.columns
        assert json.loads(table.uns["feature_axis"]["dims"]) == ["precursor_mz", "mz"]


class TestRefusals:
    def test_the_key_is_the_summed_table_plus_a_suffix(self):
        assert msms_table_key("msi_z0") == "msi_z0_msms"

    def test_no_schedule_at_all(self):
        assert "not MS/MS" in demultiplex_refusal(None)

    def test_an_ms1_acquisition(self):
        assert "not MS/MS" in demultiplex_refusal(FragmentationSchedule(ms_level=1))

    def test_a_single_precursor_needs_no_demultiplexing(self):
        """Its summed table already *is* the fragment spectrum."""
        schedule = FragmentationSchedule(ms_level=2, windows=WINDOWS[:1])
        assert "single precursor" in demultiplex_refusal(schedule)
        assert _build(_reader(schedule=schedule)) is None

    def test_a_source_that_records_no_precursor_at_all(self):
        """Not "one precursor": none, which is a different thing to say.

        A diaPASEF acquisition puts its windows in ``DiaFrameMsMsWindows``,
        which the TDF reader does not read, so the schedule it builds is
        MS/MS with an empty window list. Measured on a real one (32 DIA
        windows, 1786 survey and 28566 fragment frames): before the
        conditions were ordered as they are now, this reported "isolates a
        single precursor" about a run that isolates 32.
        """
        schedule = FragmentationSchedule(ms_level=2, windows=())
        assert "no precursor" in demultiplex_refusal(schedule)

    def test_a_schedule_that_varies_per_pixel(self):
        schedule = FragmentationSchedule(
            ms_level=2, windows=WINDOWS, constant_across_pixels=False
        )
        assert "not constant across pixels" in demultiplex_refusal(schedule)
        assert _build(_reader(schedule=schedule)) is None

    def test_survey_and_fragment_frames_are_refused_as_non_constant(self):
        """The mixed-``MsMsType`` case, which has no schedule per pixel.

        The reader sets ``constant_across_pixels`` false as soon as a file
        holds survey frames as well as fragment ones, whatever the
        precursor tables say -- and that fact outranks the window count,
        which for such a file says nothing about the method.
        """
        schedule = FragmentationSchedule(
            ms_level=2, windows=(), constant_across_pixels=False
        )
        assert "not constant across pixels" in demultiplex_refusal(schedule)
        assert _build(_reader(schedule=schedule, spectra=[])) is None

    def test_overlapping_windows(self):
        overlapping = (
            WINDOWS[0],
            IsolationWindow(936.578, scan_begin=180, scan_end=260),
        )
        schedule = FragmentationSchedule(ms_level=2, windows=overlapping)
        assert "overlap" in demultiplex_refusal(schedule)
        assert _build(_reader(schedule=schedule)) is None

    def test_windows_without_a_scan_range(self):
        """Not separable by mobility, so not separable at all."""
        schedule = FragmentationSchedule(
            ms_level=2,
            windows=(IsolationWindow(313.275), IsolationWindow(936.578)),
        )
        assert "no mobility scan range" in demultiplex_refusal(schedule)

    def test_a_separable_schedule_is_not_refused(self):
        assert demultiplex_refusal(SCHEDULE) is None


class TestIsomerPrecursors:
    """Two windows at one m/z, told apart by mobility -- an isomer pair.

    This is what the instrument's mobility dimension is *for* on a
    targeted method, so merging them back together would undo the whole
    point of the acquisition. They must stay two column blocks.
    """

    ISOMERS = (
        IsolationWindow(313.275, scan_begin=150, scan_end=200),
        IsolationWindow(313.275, scan_begin=20, scan_end=60),
    )
    SCHEDULE = FragmentationSchedule(ms_level=2, windows=ISOMERS)
    SPECTRA = [
        ((0, 0, 0), 0, [300.0], [10.0]),
        ((0, 0, 0), 1, [300.0], [20.0]),
    ]

    def _table(self):
        return _build(_reader(schedule=self.SCHEDULE, spectra=self.SPECTRA, ramp=RAMP))

    def test_they_stay_two_precursors(self):
        table = self._table()

        np.testing.assert_array_equal(
            table.var["precursor_mz"].to_numpy(), [313.275, 313.275]
        )
        np.testing.assert_array_equal(table.var["precursor_index"].to_numpy(), [0, 1])
        # Their intensities are never summed together. The lower-mobility
        # window sorts first, which on a decreasing ramp is the later scan
        # range -- window 0 here, carrying 10.
        np.testing.assert_allclose(_dense(table), [[10.0, 20.0], [0.0, 0.0]])

    def test_the_isolation_mobility_is_what_tells_them_apart(self):
        table = self._table()
        mobility = table.var["precursor_mobility"].to_numpy()

        assert mobility[0] < mobility[1]
        # 1/K0 at the middle scan of each window, on a decreasing ramp.
        np.testing.assert_allclose(mobility, [RAMP[174], RAMP[39]])

    def test_their_labels_do_not_collide(self):
        table = self._table()

        assert list(table.var.index) == ["p313.275_mz2", "p313.275_mz2_1"]
        assert table.var.index.is_unique

    def test_a_schedule_of_distinct_precursors_carries_no_repeat(self):
        """The ordinary case is unchanged: one block per m/z."""
        table = _build(_reader(ramp=RAMP))

        assert list(table.var["precursor_mz"].unique()) == [313.275, 936.578]
        assert table.var["precursor_mz"].is_monotonic_increasing
        assert table.var.index.is_unique


class TestAlignmentAcrossScheduleShapes:
    """Two samples whose schedules differ in length must still align.

    The synthetic form of the check run on real data: a copy of an
    acquisition with one isolation window removed from every frame gives
    every remaining precursor a lower rank than it has in the full one, so
    a positional label would concatenate the wrong precursors onto each
    other -- silently, since a rank collides with a rank. The labels are
    named after the precursor's m/z precisely so that cannot happen.
    """

    FULL = (
        IsolationWindow(313.275, 0.5, 0.5, 34.3, scan_begin=150, scan_end=200),
        IsolationWindow(500.0, 0.5, 0.5, 40.0, scan_begin=90, scan_end=140),
        IsolationWindow(936.578, 0.5, 0.5, 49.6, scan_begin=20, scan_end=60),
    )
    FULL_SPECTRA = [
        ((0, 0, 0), 0, [100.0, 300.0], [10.0, 20.0]),
        ((0, 0, 0), 1, [200.0, 400.0], [30.0, 40.0]),
        ((0, 0, 0), 2, [200.0], [50.0]),
        ((1, 0, 0), 1, [400.0], [60.0]),
    ]
    #: The same acquisition with the lowest-m/z window gone: the reader
    #: reports two windows, so what was window 1 is now window 0.
    REDUCED_SPECTRA = [
        (coords, window - 1, mzs, intensities)
        for coords, window, mzs, intensities in FULL_SPECTRA
        if window > 0
    ]

    def _full(self):
        schedule = FragmentationSchedule(ms_level=2, windows=self.FULL)
        return _build(_reader(schedule=schedule, spectra=self.FULL_SPECTRA))

    def _reduced(self):
        schedule = FragmentationSchedule(ms_level=2, windows=self.FULL[1:])
        return _build(_reader(schedule=schedule, spectra=self.REDUCED_SPECTRA))

    def test_the_remaining_precursors_all_shift_one_rank_down(self):
        """The premise: what makes a positional label wrong here."""
        full, reduced = self._full().var, self._reduced().var

        assert full.loc[full["precursor_mz"] == 500.0, "precursor_index"].unique() == [
            1
        ]
        assert reduced.loc[
            reduced["precursor_mz"] == 500.0, "precursor_index"
        ].unique() == [0]
        assert 313.275 not in set(reduced["precursor_mz"])

    def test_every_shared_label_still_names_the_same_precursor_and_bin(self):
        full, reduced = self._full().var, self._reduced().var
        shared = full.index.intersection(reduced.index)

        assert list(shared) == ["p500_mz1", "p500_mz3", "p936.578_mz1"]
        np.testing.assert_array_equal(
            full.loc[shared, "precursor_mz"].to_numpy(),
            reduced.loc[shared, "precursor_mz"].to_numpy(),
        )
        np.testing.assert_array_equal(
            full.loc[shared, "mz_index"].to_numpy(),
            reduced.loc[shared, "mz_index"].to_numpy(),
        )

    def test_concat_merges_no_two_precursors_and_drops_the_rank(self):
        """`merge="unique"` is anndata itself finding the rank disagrees."""
        import anndata

        full, reduced = self._full(), self._reduced()
        joined = anndata.concat(
            {"full": full, "reduced": reduced},
            axis=0,
            join="inner",
            label="sample",
            index_unique="-",
            merge="unique",
        )

        assert list(joined.var.index) == ["p500_mz1", "p500_mz3", "p936.578_mz1"]
        # The intrinsic columns survive because both stores agree on them;
        # precursor_index does not, because the rank means different things.
        assert "precursor_mz" in joined.var.columns
        assert "mz_index" in joined.var.columns
        assert "precursor_index" not in joined.var.columns
        np.testing.assert_array_equal(
            joined.var["precursor_mz"].to_numpy(), [500.0, 500.0, 936.578]
        )

    def test_the_dropped_precursor_is_absent_rather_than_reassigned(self):
        import anndata

        full, reduced = self._full(), self._reduced()
        joined = anndata.concat(
            {"full": full, "reduced": reduced},
            axis=0,
            join="outer",
            label="sample",
            index_unique="-",
        )
        dropped = [label for label in full.var.index if label.startswith("p313.275_")]
        rows = (joined.obs["sample"] == "reduced").to_numpy()
        block = joined[rows, joined.var.index.isin(dropped)]

        assert dropped == ["p313.275_mz0", "p313.275_mz2"]
        assert np.abs(_dense(block)).sum() == 0.0

    def test_a_rank_named_label_would_have_merged_two_precursors(self):
        """Why the label is not `p{rank}_mz{i}`, stated as an assertion."""
        full, reduced = self._full().var, self._reduced().var

        def ranked(var):
            return pd.Series(
                var["precursor_mz"].to_numpy(),
                index=[
                    f"p{p}_mz{m}"
                    for p, m in zip(var["precursor_index"], var["mz_index"])
                ],
            )

        left, right = ranked(full), ranked(reduced)
        shared = left.index.intersection(right.index)

        assert list(shared) == ["p1_mz1"]
        # The same label, two unrelated precursors: 500.0 in one sample and
        # 936.578 in the other would have been summed into one column.
        assert left.loc["p1_mz1"] == 500.0
        assert right.loc["p1_mz1"] == 936.578


class TestTwoUnrelatedSchedules:
    """Two acquisitions with nothing in common must concatenate to nothing shared.

    The negative-mode counterpart of a targeted run isolates a different
    list of precursors from the positive-mode one, so no column of either
    store describes anything in the other. Measured on the two real pairs
    (13 windows over m/z 50-1200 against 15 over 50-1000): **zero** shared
    labels, but 5,262 shared *rank*-named ones, every one of which names
    two different precursors -- and that collision survives the two runs
    having different mass axes, because both axes start at m/z 50 and the
    low bins line up.
    """

    NEGATIVE = (
        IsolationWindow(519.182, scan_begin=20, scan_end=60),
        IsolationWindow(720.469, scan_begin=90, scan_end=140),
    )
    SPECTRA = [
        ((0, 0, 0), 0, [100.0, 300.0], [10.0, 20.0]),
        ((0, 0, 0), 1, [200.0], [30.0]),
    ]

    def _other(self):
        schedule = FragmentationSchedule(ms_level=2, windows=self.NEGATIVE)
        return _build(_reader(schedule=schedule, spectra=self.SPECTRA))

    def test_no_label_is_shared(self):
        one, other = _build().var, self._other().var

        assert set(one["precursor_mz"]).isdisjoint(set(other["precursor_mz"]))
        assert one.index.intersection(other.index).empty

    def test_a_rank_named_label_would_have_shared_several(self):
        """The same columns under a positional name, to show what is avoided."""
        one, other = _build().var, self._other().var

        def ranked(var):
            return pd.Series(
                var["precursor_mz"].to_numpy(),
                index=[
                    f"p{p}_mz{m}"
                    for p, m in zip(var["precursor_index"], var["mz_index"])
                ],
            )

        left, right = ranked(one), ranked(other)
        shared = left.index.intersection(right.index)

        assert shared.size > 0
        assert all(
            left.loc[label] != right.loc[label] for label in shared
        ), "a rank label must name two different precursors here"

    def test_an_outer_concat_keeps_them_apart(self):
        import anndata

        one, other = _build(), self._other()
        joined = anndata.concat(
            {"one": one, "other": other},
            axis=0,
            join="outer",
            label="sample",
            index_unique="-",
        )

        assert joined.n_vars == one.n_vars + other.n_vars
        # Every column belongs to exactly one sample, so nothing was summed.
        rows = (joined.obs["sample"] == "one").to_numpy()
        for mask, table in ((rows, one), (~rows, other)):
            assert _dense(joined[mask]).sum() == pytest.approx(_dense(table).sum())
