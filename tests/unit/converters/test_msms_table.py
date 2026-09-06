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


def _reader(schedule=SCHEDULE, spectra=SPECTRA):
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
        assert list(var.index) == ["p0_mz0", "p0_mz2", "p1_mz1", "p1_mz4"]

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
        import json

        uns = _build().uns

        assert uns["provenance"] == "unchanged"
        assert json.loads(uns["feature_axis"]["dims"]) == ["precursor_mz", "mz"]
        assert uns["feature_axis"]["summed_table"] == "msi_z0"


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


class TestMergedPrecursors:
    def test_two_windows_on_one_precursor_are_one_column_block(self):
        """Two mobility slices of the same m/z are one precursor, summed.

        Keeping them apart would put the same ``(precursor_mz, mz)`` pair
        in ``var`` twice, which is exactly what the pair contract forbids.
        """
        windows = (
            IsolationWindow(313.275, scan_begin=20, scan_end=60),
            IsolationWindow(313.275, scan_begin=150, scan_end=200),
        )
        schedule = FragmentationSchedule(ms_level=2, windows=windows)
        spectra = [
            ((0, 0, 0), 0, [300.0], [10.0]),
            ((0, 0, 0), 1, [300.0], [20.0]),
        ]
        table = _build(_reader(schedule=schedule, spectra=spectra))

        np.testing.assert_array_equal(table.var["precursor_mz"].to_numpy(), [313.275])
        np.testing.assert_allclose(_dense(table), [[30.0], [0.0]])
