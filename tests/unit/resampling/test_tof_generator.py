"""The two-term TOF width law: ``FWHM(m) = sqrt(A m + B m^2)``, bins at FWHM / k.

``linear_tof`` and ``reflector_tof`` are its ``B = 0`` and ``A = 0`` limits,
and the generator must reduce to each of them exactly. With the pair fitted
on the SELECT SERIES MRT it must lay ``k`` bins per measured peak width at
every m/z, and keep the 800.5477 / 800.5566 pair in separate bins.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from thyra.converters.spatialdata.base_spatialdata_converter import (
    BaseSpatialDataConverter,
    _normalize_resampling_config,
    _reference_params,
    _tof_plan,
)
from thyra.errors import ConversionRefused
from thyra.resampling.common_axis import CommonAxisBuilder
from thyra.resampling.data_characteristics import DataCharacteristics
from thyra.resampling.decision_tree import ResamplingDecisionTree
from thyra.resampling.instrument_detectors import (
    InstrumentDetectorChain,
    TimsTOFDetector,
    WatersDetector,
    WatersMRTCentroidDetector,
    WatersProfileDetector,
)
from thyra.resampling.mass_axis import (
    DEFAULT_BINS_PER_FWHM,
    MRT_TOF_LAW,
    TIMSTOF_TOF_LAW,
    LinearTOFAxisGenerator,
    ReflectorTOFAxisGenerator,
    TOFAxisGenerator,
    tof_fwhm_mda,
)
from thyra.resampling.types import AxisType, ResamplingMethod

MRT_A, MRT_B = MRT_TOF_LAW


class TestTheLaw:
    def test_mrt_pair_reproduces_the_measured_widths(self):
        """2.97 / 3.79 / 4.54 mDa at m/z 400 / 600 / 800, within 0.1 mDa."""
        got = tof_fwhm_mda([400.0, 600.0, 800.0], MRT_A, MRT_B)
        np.testing.assert_allclose(got, [2.97, 3.79, 4.54], atol=0.1)

    def test_timstof_pair_is_near_constant_ppm(self):
        """1 / sqrt(B) = 34k: the reflector limit dominates over the lipid range."""
        w = tof_fwhm_mda([400.0, 800.0], *TIMSTOF_TOF_LAW)
        assert 800.0 / (w[1] * 1e-3) == pytest.approx(31_900, rel=0.02)
        assert w[1] / w[0] == pytest.approx(1.9, abs=0.05)

    @pytest.mark.parametrize(
        "a, b", [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0), (float("nan"), 1.0)]
    )
    def test_invalid_pairs_are_refused(self, a, b):
        with pytest.raises(ValueError):
            TOFAxisGenerator(a, b)


class TestLimits:
    """B = 0 is the sqrt(m) grid, A = 0 the ln(m) grid the single-term generators lay."""

    def test_b_zero_is_linear_tof(self):
        ours = TOFAxisGenerator(0.02, 0.0).generate_axis(100.0, 1000.0, 5000)
        theirs = LinearTOFAxisGenerator().generate_axis(100.0, 1000.0, 5000)
        np.testing.assert_allclose(ours.mz_values, theirs.mz_values, rtol=1e-12)

    def test_a_zero_is_reflector_tof(self):
        ours = TOFAxisGenerator(0.0, 1e-5).generate_axis(100.0, 1000.0, 5000)
        theirs = ReflectorTOFAxisGenerator().generate_axis(100.0, 1000.0, 5000)
        np.testing.assert_allclose(ours.mz_values, theirs.mz_values, rtol=1e-12)

    def test_general_case_sits_between_the_limits(self):
        """Width ratio across the range: 0.5 < exponent < 1."""
        axis = (
            TOFAxisGenerator(MRT_A, MRT_B).generate_axis(300.0, 1000.0, 20000).mz_values
        )
        widths = np.diff(axis)
        i4, i8 = np.searchsorted(axis, 400.0), np.searchsorted(axis, 800.0)
        ratio = widths[i8] / widths[i4]
        assert np.sqrt(2.0) < ratio < 2.0
        assert ratio == pytest.approx(4.54 / 2.97, rel=0.02)


class TestBinsPerFWHM:
    def setup_method(self):
        self.gen = TOFAxisGenerator(MRT_A, MRT_B)
        self.k = DEFAULT_BINS_PER_FWHM
        self.n = self.gen.bin_count(100.0, 1000.0, self.k)
        self.axis = self.gen.generate_axis(100.0, 1000.0, self.n).mz_values

    @pytest.mark.parametrize("mz", [400.0, 600.0, 800.0])
    def test_k_bins_per_measured_width(self, mz):
        i = int(np.searchsorted(self.axis, mz))
        width = self.axis[i] - self.axis[i - 1]
        assert tof_fwhm_mda(mz, MRT_A, MRT_B) * 1e-3 / width == pytest.approx(
            self.k, rel=0.05
        )

    def test_count_matches_the_closed_form(self):
        """The count is 1000 k times the integral of 1 / FWHM over the range."""
        m = np.linspace(100.0, 1000.0, 200_001)
        numeric = np.trapezoid(1.0 / tof_fwhm_mda(m, MRT_A, MRT_B), m)
        assert self.n == pytest.approx(1e3 * self.k * numeric, rel=1e-4)

    def test_axis_is_ascending_and_spans_the_range(self):
        assert np.all(np.diff(self.axis) > 0)
        assert self.axis[0] > 100.0 and self.axis[-1] < 1000.0
        assert self.axis[0] == pytest.approx(100.0, abs=0.002)
        assert self.axis[-1] == pytest.approx(1000.0, abs=0.002)

    def test_the_doublet_lands_in_separate_bins(self):
        """13C2 PC 34:1 [M+K]+ against PC 34:0 [M+K]+, 8.9 mDa apart at k = 3."""
        a, b = np.searchsorted(self.axis, [800.5477, 800.5566])
        assert b - a >= 5

    def test_width_and_k_are_interchangeable(self):
        k = self.gen.bins_per_fwhm_for(1000.0, 0.002)
        assert self.gen.bin_width_at(1000.0, k) == pytest.approx(0.002)
        assert k == pytest.approx(tof_fwhm_mda(1000.0, MRT_A, MRT_B) / 2.0)

    def test_builder_needs_the_law(self):
        builder = CommonAxisBuilder()
        with pytest.raises(ConversionRefused, match="tof_law"):
            builder.build_physics_axis(100.0, 1000.0, 100, AxisType.TOF)
        axis = builder.build_physics_axis(
            100.0, 1000.0, 100, AxisType.TOF, tof_law=MRT_TOF_LAW
        )
        assert axis.axis_type is AxisType.TOF
        assert axis.num_bins == 100

    def test_builder_refuses_an_axis_type_it_has_no_generator_for(self):
        """``AxisType.UNKNOWN`` has no spacing model, so no axis is built.

        The second of ``build_physics_axis``'s two refusals, and the one
        with no direct coverage before: ``AxisType.UNKNOWN`` reached the
        builder only through ``_normalize_resampling_config``, which is a
        different function and refuses it one frame earlier. An analyser
        nobody could identify is left as ``None`` and auto-detected, not
        labelled ``UNKNOWN``.
        """
        builder = CommonAxisBuilder()
        with pytest.raises(ConversionRefused, match="Unsupported axis type"):
            builder.build_physics_axis(100.0, 1000.0, 100, AxisType.UNKNOWN)


class TestConverterPlan:
    """``_tof_plan``: the caller's pair, else the detected one; k from width or flag."""

    @staticmethod
    def _stub(**kw):
        base = dict(
            _width_at_mz=None,
            _reference_mz=1000.0,
            _tof_a=None,
            _tof_b=None,
            _bins_per_fwhm=None,
            _detected_tof_law=None,
            _detected_reference_width=None,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_default_is_three_bins_per_fwhm_on_the_detected_law(self):
        assert _tof_plan(self._stub(_detected_tof_law=MRT_TOF_LAW)) == (
            MRT_A,
            MRT_B,
            3.0,
        )

    def test_callers_pair_wins(self):
        a, b, k = _tof_plan(
            self._stub(_tof_a=0.1, _tof_b=1e-4, _detected_tof_law=MRT_TOF_LAW)
        )
        assert (a, b) == (0.1, 1e-4)

    def test_explicit_bins_per_fwhm(self):
        assert (
            _tof_plan(self._stub(_detected_tof_law=MRT_TOF_LAW, _bins_per_fwhm=4.0))[2]
            == 4.0
        )

    def test_width_at_reference_derives_k(self):
        a, b, k = _tof_plan(
            self._stub(_detected_tof_law=MRT_TOF_LAW, _width_at_mz=0.002)
        )
        assert k == pytest.approx(tof_fwhm_mda(1000.0, MRT_A, MRT_B) / 2.0)
        width, ref = _reference_params(
            self._stub(_detected_tof_law=MRT_TOF_LAW, _width_at_mz=0.002), "tof"
        )
        assert (width, ref) == (0.002, 1000.0)

    def test_reference_params_report_the_realised_width(self):
        width, ref = _reference_params(self._stub(_detected_tof_law=MRT_TOF_LAW), "tof")
        assert ref == 1000.0
        assert width == pytest.approx(tof_fwhm_mda(1000.0, MRT_A, MRT_B) * 1e-3 / 3.0)

    def test_no_law_anywhere_is_an_error(self):
        with pytest.raises(ValueError, match="--tof-law A B"):
            _tof_plan(self._stub())

    def test_bin_count_goes_through_the_generator(self):
        stub = self._stub(_detected_tof_law=MRT_TOF_LAW)
        bins = BaseSpatialDataConverter._calculate_bins_from_width(
            stub, 100.0, 1000.0, AxisType.TOF
        )
        assert bins == TOFAxisGenerator(MRT_A, MRT_B).bin_count(100.0, 1000.0, 3.0)


class TestConfig:
    def test_tof_axis_and_coefficients_normalise(self):
        cfg = _normalize_resampling_config(
            {"axis_type": "tof", "tof_a": "0.0185", "tof_b": 9.1e-6, "bins_per_fwhm": 4}
        )
        assert cfg.axis_type is AxisType.TOF
        assert cfg.tof_a == 0.0185
        assert cfg.tof_b == 9.1e-6
        assert cfg.bins_per_fwhm == 4.0

    def test_missing_coefficients_stay_none(self):
        cfg = _normalize_resampling_config({"axis_type": "tof"})
        assert cfg.tof_a is None and cfg.tof_b is None and cfg.bins_per_fwhm is None


class TestDetectors:
    MRT_CENTROID = {
        "format_specific": {"format": "Waters MassLynx raw", "is_mrt": True},
        "essential_metadata": {"spectrum_type": "centroid spectrum"},
    }

    def test_mrt_centroid_gets_the_tof_law(self):
        chain = InstrumentDetectorChain()
        characteristics = DataCharacteristics.from_metadata(self.MRT_CENTROID)
        assert isinstance(chain.detect(characteristics), WatersMRTCentroidDetector)
        assert chain.get_axis_type(characteristics) is AxisType.TOF
        assert (
            chain.get_resampling_method(characteristics)
            is ResamplingMethod.NEAREST_NEIGHBOR
        )
        assert chain.get_tof_law(characteristics) == MRT_TOF_LAW
        assert chain.get_reference_width(characteristics) is None

    def test_mrt_profile_never_reaches_it(self):
        """The law describes peak width, not the digitiser grid."""
        characteristics = DataCharacteristics.from_metadata(
            {
                **self.MRT_CENTROID,
                "essential_metadata": {"spectrum_type": "profile spectrum"},
            }
        )
        chain = InstrumentDetectorChain()
        assert isinstance(chain.detect(characteristics), WatersProfileDetector)
        assert chain.get_axis_type(characteristics) is AxisType.LINEAR_TOF
        assert not WatersMRTCentroidDetector().matches(characteristics)

    def test_other_waters_centroids_keep_reflector_tof(self):
        characteristics = DataCharacteristics.from_metadata(
            {
                "format_specific": {"format": "Waters MassLynx raw", "is_mrt": False},
                "essential_metadata": {"spectrum_type": "centroid spectrum"},
            }
        )
        chain = InstrumentDetectorChain()
        assert isinstance(chain.detect(characteristics), WatersDetector)
        assert chain.get_axis_type(characteristics) is AxisType.REFLECTOR_TOF
        assert chain.get_tof_law(characteristics) is None

    def test_timstof_declares_its_pair_but_keeps_reflector_tof(self):
        """Opt-in: --mass-axis-type tof picks the pair up; the default is unchanged."""
        characteristics = DataCharacteristics(is_timstof=True)
        detector = TimsTOFDetector()
        assert detector.get_axis_type() is AxisType.REFLECTOR_TOF
        assert detector.get_tof_law(characteristics) == TIMSTOF_TOF_LAW
        tree = ResamplingDecisionTree()
        metadata = {"GlobalMetadata": {"InstrumentName": "timsTOF fleX"}}
        assert tree.select_axis_type(metadata) is AxisType.REFLECTOR_TOF
        assert tree.select_tof_law(metadata) == TIMSTOF_TOF_LAW
        assert tree.select_tof_law(None) is None

    def test_timstof_pair_is_within_ten_percent_of_the_shipped_axis(self):
        """So keeping reflector_tof at 5 mDa as the default loses little.

        The measured law against the reflector shape, normalised at m/z
        1000: within 7% at m/z 400 and 3% at 600. At m/z 300 the constant
        term shows and the gap reaches 10.2%, which is why the pair is
        opt-in rather than a silent change to every timsTOF store.
        """
        mz = np.array([400.0, 600.0, 1000.0])
        widths = tof_fwhm_mda(mz, *TIMSTOF_TOF_LAW) * 1e-3 / 3.0
        reflector = 0.005 * mz / 1000.0
        ratio = (widths / widths[-1]) / (reflector / reflector[-1])
        assert np.all(np.abs(ratio - 1.0) < 0.10)
        low = tof_fwhm_mda(300.0, *TIMSTOF_TOF_LAW) / tof_fwhm_mda(
            1000.0, *TIMSTOF_TOF_LAW
        )
        assert low / 0.3 == pytest.approx(1.10, abs=0.01)
