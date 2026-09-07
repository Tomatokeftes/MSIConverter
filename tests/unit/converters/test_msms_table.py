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

    def test_a_schedule_that_varies_per_pixel(self):
        schedule = FragmentationSchedule(
            ms_level=2, windows=WINDOWS, constant_across_pixels=False
        )
        assert "not constant across pixels" in demultiplex_refusal(schedule)
        assert _build(_reader(schedule=schedule)) is None

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
