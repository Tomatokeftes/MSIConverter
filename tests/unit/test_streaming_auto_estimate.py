"""``--streaming auto`` sizes a conversion from what the source measured.

Issue #214: the estimate assumed 10,000 peaks per spectrum whatever the
source, which under-counts profile data badly. The mistake falls the unsafe
way -- ``auto`` keeps the conversion in memory exactly when it should have
streamed, and it dies there.

The numbers here are measured on the run in that report, 7,682 pixels of
``180814_EVO_Fresh_image.raw`` read as the profile trace (2026-09-08):

===============================  ==============
``total_peaks``                  653,947,763
``n_spectra``                    7,683
points per spectrum              85,117
the axis the conversion resamples onto  2,590,447 bins
===============================  ==============

Both halves are needed. The source width alone estimates 9.7 GB and stays
under the 10 GB threshold; the run really died holding a 24.5 GiB array,
because interpolating a contiguous trace onto a 30x finer axis fills the
bins in between. The axis width is what puts it over.
"""

from types import SimpleNamespace

import pytest

from thyra.convert import (
    _ASSUMED_VALUES_PER_SPECTRUM,
    _resolved_target_bins,
    _should_use_streaming,
    _values_per_spectrum,
)
from thyra.resampling.constants import SpectrumType, Thresholds

# The Waters profile run from issue #214, as measured.
_EVO_PIXELS = 7682
_EVO_POINTS_PER_SPECTRUM = 85_117
_EVO_TARGET_BINS = 2_590_447


def _meta(dimensions, total_peaks, n_spectra, spectrum_type=None):
    return SimpleNamespace(
        dimensions=dimensions,
        total_peaks=total_peaks,
        n_spectra=n_spectra,
        spectrum_type=spectrum_type,
    )


def _reader(meta):
    return SimpleNamespace(get_essential_metadata=lambda: meta)


def _probe(target_bins, resampling_config=object()):
    """A stand-in for the converter whose axis planner is consulted."""
    return SimpleNamespace(
        _resampling_config=resampling_config,
        _resolve_resampling_plan=lambda: (100.0, 1000.0, "tof", target_bins),
    )


class TestResolvedTargetBins:
    def test_asks_the_converters_own_planner(self):
        assert _resolved_target_bins(_probe(60_000)) == 60_000

    def test_no_probe_means_no_axis(self):
        assert _resolved_target_bins(None) is None

    def test_a_conversion_that_resamples_nothing_has_no_axis(self):
        assert _resolved_target_bins(_probe(60_000, resampling_config=None)) is None

    def test_a_planner_that_cannot_resolve_is_not_fatal(self):
        """Sizing is best-effort; a refusal here must not fail the run."""

        def boom():
            raise ValueError("needs the instrument's width law")

        probe = SimpleNamespace(
            _resampling_config=object(), _resolve_resampling_plan=boom
        )
        assert _resolved_target_bins(probe) is None


class TestValuesPerSpectrum:
    def test_uses_the_measured_source_width(self):
        """``total_peaks`` is measured by every extractor, not guessed."""
        meta = _meta((10, 10, 1), total_peaks=500_000, n_spectra=100)
        assert _values_per_spectrum(meta) == 5_000

    def test_rounds_a_fractional_width_up(self):
        meta = _meta((10, 10, 1), total_peaks=101, n_spectra=100)
        assert _values_per_spectrum(meta) == 2

    def test_a_profile_source_is_sized_at_the_axis_it_is_written_onto(self):
        """A trace has no gaps to keep the resampled row sparse."""
        meta = _meta(
            (10, 10, 1),
            total_peaks=_EVO_POINTS_PER_SPECTRUM * 100,
            n_spectra=100,
            spectrum_type=SpectrumType.PROFILE,
        )
        assert _values_per_spectrum(meta, _EVO_TARGET_BINS) == _EVO_TARGET_BINS

    def test_a_centroid_source_keeps_its_peak_count(self):
        """Peaks stay peaks -- a finer axis does not multiply them.

        Without this, every resampled centroid conversion would be sized at
        its dense width and sent to streaming: a 26k-pixel TDF slide on a
        240k-bin axis would score 93 TB instead of 0.8 GB.
        """
        meta = _meta(
            (10, 10, 1),
            total_peaks=2_000 * 100,
            n_spectra=100,
            spectrum_type=SpectrumType.CENTROID,
        )
        assert _values_per_spectrum(meta, 240_794) == 2_000

    def test_a_coarser_axis_can_only_merge_peaks(self):
        meta = _meta(
            (10, 10, 1),
            total_peaks=50_000 * 100,
            n_spectra=100,
            spectrum_type=SpectrumType.CENTROID,
        )
        assert _values_per_spectrum(meta, 500) == 500

    def test_an_unknown_representation_is_treated_as_centroid(self):
        """Bruker is the source that leaves this unset, and it is sparse.

        No Bruker extractor populates ``spectrum_type``, so this branch is
        the TDF/TSF path rather than an edge case. Its summed spectra are
        per-index counts with gaps, which is the centroid case; sizing them
        densely would send every Bruker conversion to streaming. Measured
        on two real slides (2026-09-08), the routing is unchanged either
        way: a 33,800-pixel slide scores 1.17 GB and a 918,855-pixel one
        28.06 GB, the same side of the threshold as before.
        """
        meta = _meta((10, 10, 1), total_peaks=2_000 * 100, n_spectra=100)
        assert _values_per_spectrum(meta, 240_794) == 2_000

    @pytest.mark.parametrize(
        "total_peaks, n_spectra",
        [(0, 100), (500, 0), (None, 100), (500, None)],
    )
    def test_falls_back_when_the_source_measured_nothing(self, total_peaks, n_spectra):
        """A Bruker preview handle skips the peak count; do not divide by it."""
        meta = _meta((10, 10, 1), total_peaks, n_spectra)
        assert _values_per_spectrum(meta) == _ASSUMED_VALUES_PER_SPECTRUM

    def test_an_unmeasured_source_still_uses_the_axis(self):
        meta = _meta((10, 10, 1), total_peaks=0, n_spectra=0)
        assert _values_per_spectrum(meta, 250_000) == 250_000


class TestAutoNoticesProfileData:
    def _evo_meta(self):
        return _meta(
            (_EVO_PIXELS, 1, 1),
            total_peaks=_EVO_PIXELS * _EVO_POINTS_PER_SPECTRUM,
            n_spectra=_EVO_PIXELS,
            spectrum_type=SpectrumType.PROFILE,
        )

    def test_the_waters_profile_run_upgrades_to_streaming(self):
        """The regression in issue #214, at its measured size."""
        assert (
            _should_use_streaming(
                "auto", _reader(self._evo_meta()), _probe(_EVO_TARGET_BINS)
            )
            is True
        )

    def test_the_old_fixed_guess_would_have_missed_it(self):
        """Pins the gap so a fixed per-spectrum count cannot come back.

        10,000 peaks per spectrum scored this run at 0.57 GB against a
        10 GB threshold -- 130x under the streaming converter's own 74.1 GB.
        """
        old_estimate_gb = (_EVO_PIXELS * 10_000 * 8) / (1024**3)
        assert old_estimate_gb < 1
        assert _values_per_spectrum(self._evo_meta()) == _EVO_POINTS_PER_SPECTRUM

    def test_the_source_width_alone_is_not_enough(self):
        """Why the axis is consulted at all: 9.7 GB is under the threshold.

        This is the measurement that rejected sizing profile data from
        ``total_peaks`` alone.
        """
        source_only_gb = (_EVO_PIXELS * _EVO_POINTS_PER_SPECTRUM * 16) / (1024**3)
        assert source_only_gb < Thresholds.STREAMING_SIZE_GB
        assert _should_use_streaming("auto", _reader(self._evo_meta())) is False

    def test_a_centroid_run_of_the_same_shape_stays_in_memory(self):
        """The upgrade tracks the data, not the pixel count."""
        meta = _meta(
            (_EVO_PIXELS, 1, 1),
            total_peaks=_EVO_PIXELS * 2_000,
            n_spectra=_EVO_PIXELS,
            spectrum_type=SpectrumType.CENTROID,
        )
        assert (
            _should_use_streaming("auto", _reader(meta), _probe(_EVO_TARGET_BINS))
            is False
        )


class TestUnchangedBehaviour:
    def test_explicit_settings_never_touch_the_reader(self):
        def explode():
            raise AssertionError("must not be called")

        reader = SimpleNamespace(get_essential_metadata=explode)
        assert _should_use_streaming(True, reader) is True
        assert _should_use_streaming(False, reader) is False

    def test_a_refusal_still_propagates(self):
        def refuse():
            raise ValueError("imzML spectrum 3 declares a m/z array ending at ...")

        with pytest.raises(ValueError, match="spectrum 3"):
            _should_use_streaming(
                "auto", SimpleNamespace(get_essential_metadata=refuse)
            )

    def test_unusable_dimensions_still_fall_back_quietly(self):
        meta = SimpleNamespace(dimensions=None, total_peaks=10, n_spectra=1)
        assert _should_use_streaming("auto", _reader(meta)) is False
