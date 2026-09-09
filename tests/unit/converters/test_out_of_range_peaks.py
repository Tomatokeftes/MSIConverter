# tests/unit/converters/test_out_of_range_peaks.py
"""Peaks outside the target mass range are dropped, not folded into the edges.

``_nearest_neighbor_resample`` clipped every source index into
``[0, len(axis) - 1]`` and then accumulated with ``np.bincount``, so a peak
below the axis was *added to bin 0* and one above it to the last bin.
Narrowing the mass range -- what ``--resample-min-mz`` and
``--resample-max-mz`` exist for -- therefore piled the entire discarded
part of the spectrum onto two bins.

The total was conserved exactly, so no TIC check could see it. On real
``pea.imzML`` resampled to 400-800 m/z, pixel 0's bin 0 held 654,158 counts
against a median interior bin of 118 -- and the stored pixel total was the
*whole* input TIC, 1,090,866, rather than the 364,816 that actually lies in
range.

"In range" is the **declared** ``[min_mz, max_mz]``, which for the uniform
``np.linspace`` axes used throughout this file is exactly the axis span
``[axis[0], axis[-1]]`` -- so every case here reads the same either way. On a
physics axis the two differ by half a bin, and
``tests/unit/converters/test_declared_range_edge_bins.py`` covers that (issue
#239). Both resampling methods follow the one rule, which is what stops them
disagreeing about what the axis covers.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
)


def _stub(axis):
    """The converter's resampling surface, without building a converter.

    Same pattern as ``test_nearest_neighbor_negatives.py``; the reporting
    helper is bound onto it so the count and the one-shot warning are the
    real ones rather than a test double.
    """
    stub = SimpleNamespace(
        _common_mass_axis=np.asarray(axis, float),
        _out_of_range_peaks=0,
        _out_of_range_warned=False,
    )
    stub._count_out_of_range = (
        lambda n_dropped, n_total: BaseSpatialDataConverter._count_out_of_range(
            stub, n_dropped, n_total
        )
    )
    return stub


class _CaptureWarnings:
    """Collect WARNING+ records off the converter's logger.

    ``caplog`` is unreliable across this suite: ``setup_logging`` sets
    ``propagate = False`` on the ``thyra`` logger and other tests leave
    that state behind, so records never reach pytest's root handler.
    """

    LOGGER = "thyra.converters.spatialdata.base_spatialdata_converter"

    def __init__(self):
        self.messages: list[str] = []

    def __enter__(self):
        outer = self

        class _Handler(logging.Handler):
            def emit(self, record):
                outer.messages.append(record.getMessage())

        self._logger = logging.getLogger(self.LOGGER)
        self._handler = _Handler(level=logging.WARNING)
        self._previous = self._logger.level
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._previous)
        return False


def _resample(stub, mzs, intensities):
    return BaseSpatialDataConverter._nearest_neighbor_resample(
        stub, np.asarray(mzs, float), np.asarray(intensities, float)
    )


class TestEdgeBinsHoldOnlyWhatBelongsThere:
    """The audit's proposed test, verbatim: 90/100/110/120 onto 100-110."""

    AXIS = np.linspace(100.0, 110.0, 11)
    MZS = [90.0, 100.0, 110.0, 120.0]
    INTENSITIES = [1000.0, 7.0, 11.0, 5000.0]

    def test_stored_total_is_the_in_range_peaks_only(self):
        _, values = _resample(_stub(self.AXIS), self.MZS, self.INTENSITIES)

        # 1090.0 before: the out-of-range 1000 and 5000 were kept.
        assert values.sum() == pytest.approx(7.0 + 11.0)

    def test_neither_edge_bin_exceeds_its_own_peak(self):
        indices, values = _resample(_stub(self.AXIS), self.MZS, self.INTENSITIES)
        stored = dict(zip(indices.tolist(), values.tolist()))

        assert stored[0] == pytest.approx(7.0)  # was 1007.0
        assert stored[len(self.AXIS) - 1] == pytest.approx(11.0)  # was 5011.0

    def test_a_spectrum_entirely_outside_the_axis_stores_nothing(self):
        indices, values = _resample(_stub(self.AXIS), [50.0, 300.0], [9.0, 9.0])

        assert indices.size == 0
        assert values.size == 0

    def test_the_endpoints_themselves_are_in_range(self):
        """Strict span means inclusive of both ends, not exclusive."""
        indices, values = _resample(_stub(self.AXIS), [100.0, 110.0], [3.0, 4.0])

        assert sorted(indices.tolist()) == [0, len(self.AXIS) - 1]
        assert values.sum() == pytest.approx(7.0)

    def test_a_peak_just_outside_is_out(self):
        """A peak outside the declared range is dropped, however close.

        On this uniform axis the declared range *is* the axis span, so a
        peak a nanodalton below 100.0 is outside both. Half a bin of slack
        would have kept these; the rule reaches the declared bound and
        stops.
        """
        indices, _ = _resample(_stub(self.AXIS), [100.0 - 1e-9, 110.0 + 1e-9], [3, 4])

        assert indices.size == 0


class TestNothingChangesWhenEverythingIsInRange:
    """The regression guard: the ordinary case must be bit-identical."""

    def test_interior_peaks_are_untouched(self):
        axis = np.linspace(250.0, 1200.0, 5_000)
        mzs = [300.0, 500.0, 700.0]
        intensities = [1.0, 2.0, 3.0]

        indices, values = _resample(_stub(axis), mzs, intensities)

        assert indices.size == 3
        assert values.sum() == pytest.approx(6.0)

    def test_no_warning_when_nothing_is_dropped(self):
        axis = np.linspace(250.0, 1200.0, 5_000)
        stub = _stub(axis)

        with _CaptureWarnings() as captured:
            _resample(stub, [300.0, 700.0], [1.0, 2.0])

        assert stub._out_of_range_peaks == 0
        assert not stub._out_of_range_warned
        assert captured.messages == []


class TestTheDropIsReported:
    """Silence is what made this survive; the drop has to say so once."""

    def test_the_count_accumulates(self):
        stub = _stub(np.linspace(100.0, 110.0, 11))

        _resample(stub, [90.0, 105.0, 120.0], [1.0, 1.0, 1.0])
        _resample(stub, [95.0, 105.0], [1.0, 1.0])

        assert stub._out_of_range_peaks == 3

    def test_it_warns_once_and_names_the_axis(self):
        stub = _stub(np.linspace(100.0, 110.0, 11))

        with _CaptureWarnings() as captured:
            _resample(stub, [90.0, 105.0], [1.0, 1.0])
            _resample(stub, [90.0, 105.0], [1.0, 1.0])

        assert len(captured.messages) == 1, "one line per conversion, not per spectrum"
        assert "100.0000" in captured.messages[0]
        assert "110.0000" in captured.messages[0]


class TestTicPreservingAlreadyAgreed:
    """The two methods must not disagree about what the axis covers.

    ``_tic_preserving_resample`` never had this bug -- ``np.interp`` is
    given ``left=0, right=0`` and the rescale target comes from
    ``preserved_tic``, which integrates only over the range the axis
    covers. Pinning it here so a future change cannot make
    nearest-neighbour the odd one out again in the other direction.
    """

    def test_tic_preserving_keeps_only_the_in_range_share(self):
        axis = np.linspace(100.0, 110.0, 11)
        stub = SimpleNamespace(_common_mass_axis=axis, _gap_tolerance_da=None)

        resampled = BaseSpatialDataConverter._tic_preserving_resample(
            stub,
            np.array([90.0, 100.0, 110.0, 120.0]),
            np.array([1000.0, 7.0, 11.0, 5000.0]),
        )

        assert resampled.sum() < 1000.0
        assert resampled.sum() == pytest.approx(
            _resample(_stub(axis), [90.0, 100.0, 110.0, 120.0], [1000, 7, 11, 5000])[
                1
            ].sum(),
            rel=0.5,
        )
