"""
Tests for ResamplingDecisionTree with Strategy pattern for instrument detection.
"""

import pytest

from thyra.resampling.constants import SpectrumType
from thyra.resampling.data_characteristics import DataCharacteristics
from thyra.resampling.decision_tree import ResamplingDecisionTree
from thyra.resampling.instrument_detectors import (
    CentroidImzMLDetector,
    DefaultDetector,
    FTICRDetector,
    InstrumentDetector,
    InstrumentDetectorChain,
    OrbitrapDetector,
    PhiToFSIMSDetector,
    RapiflexDetector,
    TimsTOFDetector,
    WatersDetector,
    WatersProfileDetector,
)
from thyra.resampling.types import AxisType, ResamplingMethod


class TestResamplingDecisionTree:
    """Test ResamplingDecisionTree with Strategy pattern."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tree = ResamplingDecisionTree()

    def test_no_metadata_raises_error(self):
        """Test that None metadata raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            self.tree.select_strategy(None)

        assert "metadata" in str(exc_info.value).lower()

    def test_empty_metadata_uses_default(self):
        """Test that empty metadata uses default detector."""
        # With the Strategy pattern, empty metadata falls through to DefaultDetector
        method = self.tree.select_strategy({})
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

    def test_timstof_detection_from_bruker_metadata(self):
        """Test timsTOF detection from Bruker GlobalMetadata."""
        bruker_timstof_metadata = {
            "GlobalMetadata": {"InstrumentName": "timsTOF Maldi 2"}
        }
        method = self.tree.select_strategy(bruker_timstof_metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

        axis_type = self.tree.select_axis_type(bruker_timstof_metadata)
        assert axis_type == AxisType.REFLECTOR_TOF

    @pytest.mark.parametrize(
        "instrument_name",
        [
            "timsTOF Maldi 2",
            "timsTOF Pro 2",
            "timsTOF fleX MALDI-2",
            "timsTOF SCP",
            "timsTOF HT",
            "TIMSTOF",
            "timstof flex",
        ],
    )
    def test_timstof_variants_all_match(self, instrument_name):
        """Every timsTOF variant Bruker ships should pick REFLECTOR_TOF.

        Regression test for the original is_timstof exact-string-match
        bug that caused every variant except the original Maldi 2 to
        fall through to DefaultDetector and return CONSTANT.
        """
        metadata = {"GlobalMetadata": {"InstrumentName": instrument_name}}
        assert self.tree.select_axis_type(metadata) == AxisType.REFLECTOR_TOF
        assert self.tree.select_strategy(metadata) == ResamplingMethod.NEAREST_NEIGHBOR

    def test_non_timstof_instrument_does_not_match(self):
        """Names that just sound similar should not trigger TimsTOFDetector."""
        metadata = {"GlobalMetadata": {"InstrumentName": "AB SCIEX TripleTOF 5600"}}
        # Falls through to the DefaultDetector; that's CONSTANT.
        assert self.tree.select_axis_type(metadata) == AxisType.CONSTANT

    def test_centroid_spectrum_detection(self):
        """Test centroid spectrum detection from essential_metadata."""
        metadata = {"essential_metadata": {"spectrum_type": SpectrumType.CENTROID}}
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

    def test_centroid_spectrum_detection_in_instrument_info(self):
        """Test centroid spectrum detection via instrument info."""
        metadata = {
            "essential_metadata": {"spectrum_type": SpectrumType.CENTROID},
            "instrument_info": {"instrument_type": "Q-TOF"},
        }
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

    def test_profile_spectrum_with_high_density_uses_nearest_neighbor(self):
        """High-density profile data of unknown provenance must not get MALDI-TOF logic.

        Peak density is not a modality. This metadata says only "profile, and
        densely sampled" -- it could be MALDI-TOF, TOF-SIMS, or a profile
        Orbitrap acquisition. ``nearest_neighbor`` bins counts and assumes
        nothing about the axis law, so it is the safe answer for all of them.
        """
        metadata = {
            "essential_metadata": {
                "spectrum_type": SpectrumType.PROFILE,
                "total_peaks": 10000000,  # 10 million peaks
                "n_spectra": 1000,  # 10000 peaks per spectrum
            }
        }
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

        # And it must not pick up the MALDI-shaped axis either.
        assert self.tree.select_axis_type(metadata) == AxisType.CONSTANT

    def test_low_density_profile_uses_nearest_neighbor(self):
        """The same holds below the density threshold -- there is no profile route."""
        metadata = {
            "essential_metadata": {
                "spectrum_type": SpectrumType.PROFILE,
                "total_peaks": 1000,
                "n_spectra": 1000,  # 1 peak per spectrum
            }
        }
        assert self.tree.select_strategy(metadata) == ResamplingMethod.NEAREST_NEIGHBOR

    def test_named_non_bruker_profile_instrument_uses_nearest_neighbor(self):
        """A named vendor that is not Bruker MALDI-TOF gets no TIC-preserving either."""
        metadata = {
            "essential_metadata": {
                "spectrum_type": SpectrumType.PROFILE,
                "total_peaks": 10000000,
                "n_spectra": 1000,
            },
            "instrument_info": {
                "instrument_type": "TOF-SIMS",
                "manufacturer": "IONTOF",
            },
        }
        assert self.tree.select_strategy(metadata) == ResamplingMethod.NEAREST_NEIGHBOR

    def test_rapiflex_format_detection(self):
        """Test Rapiflex format detection."""
        metadata = {"format_specific": {"format": "Rapiflex"}}
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.TIC_PRESERVING

        axis_type = self.tree.select_axis_type(metadata)
        assert axis_type == AxisType.CONSTANT

    def test_bruker_maldi_tof_detection(self):
        """Test Bruker MALDI-TOF detection."""
        metadata = {
            "instrument_info": {
                "instrument_type": "MALDI-TOF",
                "manufacturer": "Bruker",
            }
        }
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.TIC_PRESERVING

    def test_axis_type_selection_no_metadata(self):
        """Test axis type selection with no metadata uses default."""
        axis_type = self.tree.select_axis_type(None)
        assert axis_type == AxisType.CONSTANT


class TestInstrumentDetectorChain:
    """Test InstrumentDetectorChain behavior."""

    def setup_method(self):
        """Setup test fixtures."""
        self.chain = InstrumentDetectorChain()

    def test_default_chain_order(self):
        """Test that default chain has correct order."""
        detector_types = [type(d).__name__ for d in self.chain.detectors]
        assert detector_types == [
            "TimsTOFDetector",
            "RapiflexDetector",
            "FTICRDetector",
            "OrbitrapDetector",
            "PhiToFSIMSDetector",
            "WatersProfileDetector",
            "WatersMRTCentroidDetector",
            "WatersDetector",
            "CentroidImzMLDetector",
            "DefaultDetector",
        ]

    def test_timstof_takes_priority(self):
        """Test that TimsTOF detector takes priority over others."""
        characteristics = DataCharacteristics(is_timstof=True)
        detector = self.chain.detect(characteristics)
        assert isinstance(detector, TimsTOFDetector)

    def test_rapiflex_detected_before_centroid(self):
        """Test Rapiflex detection takes priority over generic centroid."""
        characteristics = DataCharacteristics(
            is_rapiflex_format=True,
            spectrum_type=SpectrumType.CENTROID,
        )
        detector = self.chain.detect(characteristics)
        assert isinstance(detector, RapiflexDetector)

    def test_fallback_to_default(self):
        """Test fallback to DefaultDetector when nothing matches."""
        characteristics = DataCharacteristics()
        detector = self.chain.detect(characteristics)
        assert isinstance(detector, DefaultDetector)

    def test_phi_detected_before_default(self):
        """PHI must not reach DefaultDetector, which would report CONSTANT."""
        characteristics = DataCharacteristics(is_phi_tofsims=True)
        detector = self.chain.detect(characteristics)
        assert isinstance(detector, PhiToFSIMSDetector)


class TestPhiToFSIMSDetector:
    """PHI ToF-SIMS: sparse per-pixel data on a flight-time grid.

    Reaching ``DefaultDetector`` here is not a cosmetic miss. It reports a
    CONSTANT axis, and a caller that maps constant to profile-MALDI
    conventions ends up interpolating a pixel that holds a median of 44
    measured points across m/z 0.5-1850 -- fabricating intensity in every
    bin between them, with the TIC rescale hiding it behind a balanced
    total.
    """

    def setup_method(self):
        self.detector = PhiToFSIMSDetector()

    def test_matches_the_phi_raw_format_stamp(self):
        """PhiMetadataExtractor writes format_specific["format"]."""
        characteristics = DataCharacteristics.from_metadata(
            {"format_specific": {"format": "PHI SmartSoft-TOF raw"}}
        )
        assert characteristics.is_phi_tofsims
        assert self.detector.matches(characteristics)

    def test_does_not_match_other_formats(self):
        for fmt in ("Rapiflex", "imzML", "Bruker TSF", ""):
            characteristics = DataCharacteristics.from_metadata(
                {"format_specific": {"format": fmt}}
            )
            assert not characteristics.is_phi_tofsims, fmt
            assert not self.detector.matches(characteristics), fmt

    def test_missing_format_specific_does_not_match(self):
        assert not DataCharacteristics.from_metadata({}).is_phi_tofsims

    def test_uses_nearest_neighbor(self):
        """Interpolating across the gaps in a sparse pixel invents signal."""
        assert (
            self.detector.get_resampling_method() is ResamplingMethod.NEAREST_NEIGHBOR
        )

    def test_uses_linear_tof_axis(self):
        """PhiMassAxis steps at a constant flight time, so spacing ~ sqrt(m)."""
        assert self.detector.get_axis_type() is AxisType.LINEAR_TOF

    def test_declares_its_source_grid_law(self):
        """An undeclared law cannot clear the TIC-preserving gate."""
        assert self.detector.source_grid_law is AxisType.LINEAR_TOF

    def test_chain_selects_nearest_neighbor_for_phi(self):
        """The whole chain, not just the detector in isolation."""
        chain = InstrumentDetectorChain()
        characteristics = DataCharacteristics.from_metadata(
            {"format_specific": {"format": "PHI SmartSoft-TOF raw"}}
        )
        assert (
            chain.get_resampling_method(characteristics)
            is ResamplingMethod.NEAREST_NEIGHBOR
        )
        assert chain.get_axis_type(characteristics) is AxisType.LINEAR_TOF


class TestWatersDetector:
    """Waters MassLynx .raw: the answer must not hinge on the declared
    representation.

    Before this detector existed, a Waters file's fate was decided by
    ``_detect_spectrum_type``: centroid landed on ``CentroidImzMLDetector``
    (the right pair, reached by accident), profile fell to
    ``DefaultDetector`` -- CONSTANT, which downstream pairs with the 0.1 Da
    default bin width, R = 5,000 at m/z 500 on an MRT built for 100,000+.
    """

    WATERS_STAMP = {"format_specific": {"format": "Waters MassLynx raw"}}

    def setup_method(self):
        self.detector = WatersDetector()

    def test_matches_the_masslynx_format_stamp(self):
        """WatersMetadataExtractor writes format_specific["format"]."""
        characteristics = DataCharacteristics.from_metadata(self.WATERS_STAMP)
        assert characteristics.is_waters_raw
        assert self.detector.matches(characteristics)

    def test_does_not_match_other_formats(self):
        for fmt in ("Rapiflex", "imzML", "PHI SmartSoft-TOF raw", ""):
            characteristics = DataCharacteristics.from_metadata(
                {"format_specific": {"format": fmt}}
            )
            assert not characteristics.is_waters_raw, fmt
            assert not self.detector.matches(characteristics), fmt

    def test_uses_reflector_tof_and_nearest_neighbor(self):
        """Constant relative resolution -- SCiLS's Orthogonal TOF law (p.75)."""
        assert self.detector.get_axis_type() is AxisType.REFLECTOR_TOF
        assert (
            self.detector.get_resampling_method() is ResamplingMethod.NEAREST_NEIGHBOR
        )

    def test_does_not_declare_a_source_grid_law(self):
        """A centroid list has no grid; the trace's law lives on the profile detector.

        Declaring one here would open ``_gate_tic_preserving`` on peak
        lists, which have nothing to interpolate between.
        """
        assert self.detector.source_grid_law is None

    def test_centroid_width_is_two_mda(self):
        """5 mDa gave 1.1 bins per FWHM at m/z 800 on the MRT; 2 mDa gives 2.8."""
        characteristics = DataCharacteristics.from_metadata(self.WATERS_STAMP)
        assert self.detector.get_reference_width(characteristics) == (0.002, 1000.0)

    @pytest.mark.parametrize("spectrum_type", [SpectrumType.CENTROID, None])
    def test_chain_bins_centroids_and_undeclared_by_nearest_neighbour(
        self, spectrum_type
    ):
        """Centroid or undeclared: the vendor peak list, binned on reflector_tof."""
        chain = InstrumentDetectorChain()
        characteristics = DataCharacteristics.from_metadata(
            {
                **self.WATERS_STAMP,
                "essential_metadata": {"spectrum_type": spectrum_type},
            }
        )
        assert isinstance(chain.detect(characteristics), WatersDetector)
        assert chain.get_axis_type(characteristics) is AxisType.REFLECTOR_TOF
        assert (
            chain.get_resampling_method(characteristics)
            is ResamplingMethod.NEAREST_NEIGHBOR
        )
        assert chain.get_reference_width(characteristics) == (0.002, 1000.0)

    def test_chain_hands_the_profile_trace_to_the_profile_detector(self):
        chain = InstrumentDetectorChain()
        characteristics = DataCharacteristics.from_metadata(
            {
                **self.WATERS_STAMP,
                "essential_metadata": {"spectrum_type": SpectrumType.PROFILE},
            }
        )
        assert isinstance(chain.detect(characteristics), WatersProfileDetector)

    def test_beats_the_centroid_detector_in_the_chain(self):
        """A centroid Waters file is identified, not matched by accident."""
        chain = InstrumentDetectorChain()
        characteristics = DataCharacteristics.from_metadata(
            {
                **self.WATERS_STAMP,
                "essential_metadata": {"spectrum_type": SpectrumType.CENTROID},
            }
        )
        assert isinstance(chain.detect(characteristics), WatersDetector)


class TestWatersProfileDetector:
    """The profile trace: samples on the digitiser's sqrt(m/z) grid.

    Measured as ``(m/z)^0.494`` on a SELECT SERIES MRT and ``(m/z)^0.498``
    on a Synapt G2-Si, so the law is declared and ``tic_preserving`` clears
    the gate onto a ``linear_tof`` axis. The bin width is pinned at 1.3 mDa
    at m/z 1000 on an MRT (about 1.14 sample spacings) so every MRT run
    with the same mass range shares one axis, and follows the run's own
    predicted spacing on any other Waters instrument asked for its trace.
    """

    WATERS_PROFILE = {
        "format_specific": {"format": "Waters MassLynx raw"},
        "essential_metadata": {"spectrum_type": SpectrumType.PROFILE},
    }

    def setup_method(self):
        self.detector = WatersProfileDetector()
        self.chain = InstrumentDetectorChain()

    def _characteristics(self, **format_specific):
        return DataCharacteristics.from_metadata(
            {
                **self.WATERS_PROFILE,
                "format_specific": {
                    **self.WATERS_PROFILE["format_specific"],
                    **format_specific,
                },
            }
        )

    def test_matches_the_profile_trace_only(self):
        assert self.detector.matches(self._characteristics())
        for spectrum_type in (SpectrumType.CENTROID, None):
            characteristics = DataCharacteristics.from_metadata(
                {
                    "format_specific": {"format": "Waters MassLynx raw"},
                    "essential_metadata": {"spectrum_type": spectrum_type},
                }
            )
            assert not self.detector.matches(characteristics)

    def test_does_not_match_other_profile_formats(self):
        characteristics = DataCharacteristics.from_metadata(
            {
                "format_specific": {"format": "Rapiflex"},
                "essential_metadata": {"spectrum_type": SpectrumType.PROFILE},
            }
        )
        assert not self.detector.matches(characteristics)

    def test_uses_linear_tof_and_tic_preserving(self):
        assert self.detector.get_axis_type() is AxisType.LINEAR_TOF
        assert self.detector.get_resampling_method() is ResamplingMethod.TIC_PRESERVING

    def test_declares_the_measured_source_grid_law(self):
        assert self.detector.source_grid_law is AxisType.LINEAR_TOF

    def test_chain_clears_the_tic_preserving_gate(self):
        """Source law equals target law, so the interpolating path is allowed."""
        characteristics = self._characteristics()
        assert (
            self.chain.get_resampling_method(characteristics)
            is ResamplingMethod.TIC_PRESERVING
        )
        assert self.chain.get_axis_type(characteristics) is AxisType.LINEAR_TOF

    def test_mrt_width_is_pinned(self):
        characteristics = self._characteristics(
            is_mrt=True, profile_sample_spacing_da_at_1000=1.143e-3
        )
        assert characteristics.is_waters_mrt
        assert self.detector.get_reference_width(characteristics) == (0.0013, 1000.0)
        assert self.chain.get_reference_width(characteristics) == (0.0013, 1000.0)

    def test_other_instruments_follow_their_own_sample_spacing(self):
        """A Synapt G2-Si: 13.8 mDa per sample, times 1.14, up to 0.1 mDa."""
        characteristics = self._characteristics(
            is_mrt=False, profile_sample_spacing_da_at_1000=13.806e-3
        )
        width, reference = self.detector.get_reference_width(characteristics)
        assert reference == 1000.0
        assert width == pytest.approx(0.0158)

    def test_unknown_spacing_falls_back_to_the_mrt_width_with_a_warning(self, caplog):
        import logging

        characteristics = self._characteristics(is_mrt=False)
        with caplog.at_level(logging.WARNING):
            width = self.detector.get_reference_width(characteristics)
        assert width == (0.0013, 1000.0)
        assert "predict its sample spacing" in caplog.text

    def test_decision_tree_exposes_the_width(self):
        tree = ResamplingDecisionTree()
        assert tree.select_reference_width(None) is None
        assert tree.select_reference_width({"essential_metadata": {}}) is None
        assert tree.select_reference_width(
            {
                **self.WATERS_PROFILE,
                "format_specific": {"format": "Waters MassLynx raw", "is_mrt": True},
            }
        ) == (0.0013, 1000.0)

    def test_precedes_the_centroid_waters_detector(self):
        assert isinstance(
            self.chain.detect(self._characteristics()), WatersProfileDetector
        )


class TestAnalyzerFamilyReachability:
    """FTICRDetector and OrbitrapDetector, reached through from_metadata.

    Both used to be unreachable code: they match on
    ``instrument_info["instrument_type"]``, and no extractor in the package
    ever produced ``"FT-ICR"`` or ``"Orbitrap"`` -- the only producer of
    ``instrument_type`` anywhere was the Rapiflex reader's ``"MALDI-TOF"``.
    The imzML extractor now emits these exact strings from the file's own
    cvParams; these tests pin the dict-shaped contract between the two.
    """

    def setup_method(self):
        self.tree = ResamplingDecisionTree()
        self.chain = InstrumentDetectorChain()

    @pytest.mark.parametrize(
        "spectrum_type", [SpectrumType.PROFILE, SpectrumType.CENTROID, None]
    )
    def test_fticr_wins_over_the_spectrum_type_branches(self, spectrum_type):
        """A declared FT-ICR gets the quadratic axis whatever the representation.

        Profile is the branch that used to go wrong: with no instrument_type
        arriving, profile FT-ICR fell to DefaultDetector's CONSTANT at the
        0.1 Da default.
        """
        metadata = {
            "instrument_info": {"instrument_type": "FT-ICR"},
            "essential_metadata": {"spectrum_type": spectrum_type},
        }
        assert self.tree.select_axis_type(metadata) is AxisType.FTICR
        assert self.tree.select_strategy(metadata) is ResamplingMethod.NEAREST_NEIGHBOR

    @pytest.mark.parametrize(
        "spectrum_type", [SpectrumType.PROFILE, SpectrumType.CENTROID, None]
    )
    def test_orbitrap_wins_over_the_spectrum_type_branches(self, spectrum_type):
        metadata = {
            "instrument_info": {"instrument_type": "Orbitrap"},
            "essential_metadata": {"spectrum_type": spectrum_type},
        }
        assert self.tree.select_axis_type(metadata) is AxisType.ORBITRAP
        assert self.tree.select_strategy(metadata) is ResamplingMethod.NEAREST_NEIGHBOR

    def test_timstof_recognised_from_surfaced_instrument_model(self):
        """An imzML-derived timsTOF rides the same substring match as the .d.

        The imzML extractor surfaces the declared model term as
        ``instrument_info["instrument_model"]``; from_metadata folds it into
        ``instrument_name`` so ``is_timstof`` stays the single deciding path
        for both routes.
        """
        metadata = {
            "instrument_info": {"instrument_model": "timsTOF fleX"},
            "essential_metadata": {"spectrum_type": SpectrumType.PROFILE},
        }
        characteristics = DataCharacteristics.from_metadata(metadata)
        assert characteristics.is_timstof
        assert isinstance(self.chain.detect(characteristics), TimsTOFDetector)
        assert self.tree.select_axis_type(metadata) is AxisType.REFLECTOR_TOF

    def test_bruker_global_metadata_stays_authoritative(self):
        """When both sources name an instrument, GlobalMetadata wins."""
        metadata = {
            "GlobalMetadata": {"InstrumentName": "timsTOF fleX MALDI-2"},
            "instrument_info": {"instrument_model": "some other name"},
        }
        characteristics = DataCharacteristics.from_metadata(metadata)
        assert characteristics.instrument_name == "timsTOF fleX MALDI-2"


class _TICPreservingDetector(InstrumentDetector):
    """Detector that asks for TIC_PRESERVING, for exercising the chain's gate.

    ``source_law`` is what it claims about the grid its data arrives on;
    ``axis`` is the target axis it asks for.
    """

    def __init__(self, source_law, axis=AxisType.CONSTANT):
        self._source_law = source_law
        self._axis = axis

    @property
    def name(self) -> str:
        return "Test TIC-preserving"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        return True

    def get_resampling_method(self) -> ResamplingMethod:
        return ResamplingMethod.TIC_PRESERVING

    def get_axis_type(self) -> AxisType:
        return self._axis

    @property
    def source_grid_law(self):
        return self._source_law


class TestTICPreservingGate:
    """The SCiLS "identical axis types" gate on TIC-preserving resampling.

    SCiLS Lab applies TIC-preserving resampling only when the axis types
    being combined are identical, and interpolates otherwise (2026b User
    Guide, p.80). That is also the condition under which Thyra's
    interpolate-then-rescale operator is exact, so the chain enforces it.
    """

    def _method_for(self, detector):
        chain = InstrumentDetectorChain([detector])
        return chain.get_resampling_method(DataCharacteristics())

    def test_matching_laws_keep_tic_preserving(self):
        detector = _TICPreservingDetector(AxisType.CONSTANT, AxisType.CONSTANT)
        assert self._method_for(detector) == ResamplingMethod.TIC_PRESERVING

    @pytest.mark.parametrize(
        "source_law",
        [
            AxisType.LINEAR_TOF,
            AxisType.REFLECTOR_TOF,
            AxisType.ORBITRAP,
            AxisType.FTICR,
        ],
    )
    def test_mismatched_laws_fall_back_to_nearest_neighbor(self, source_law):
        """Off the diagonal the operator distorts peak ratios, so it is refused."""
        detector = _TICPreservingDetector(source_law, AxisType.CONSTANT)
        assert self._method_for(detector) == ResamplingMethod.NEAREST_NEIGHBOR

    def test_unknown_source_law_falls_back_to_nearest_neighbor(self):
        """Not knowing the acquisition is not a licence to assume it matches."""
        detector = _TICPreservingDetector(None, AxisType.CONSTANT)
        assert self._method_for(detector) == ResamplingMethod.NEAREST_NEIGHBOR

    def test_gate_leaves_the_axis_type_alone(self):
        """The gate changes the method only; the axis type is a separate answer."""
        detector = _TICPreservingDetector(None, AxisType.CONSTANT)
        chain = InstrumentDetectorChain([detector])
        assert chain.get_axis_type(DataCharacteristics()) == AxisType.CONSTANT

    def test_gate_does_not_touch_nearest_neighbor_detectors(self):
        """A detector that never asks for TIC_PRESERVING is unaffected."""
        chain = InstrumentDetectorChain([DefaultDetector()])
        assert (
            chain.get_resampling_method(DataCharacteristics())
            == ResamplingMethod.NEAREST_NEIGHBOR
        )

    def test_every_shipped_detector_clears_the_gate(self):
        """No detector in the default chain may ask for an unbacked TIC_PRESERVING."""
        for detector in InstrumentDetectorChain().detectors:
            if detector.get_resampling_method() is not ResamplingMethod.TIC_PRESERVING:
                continue
            assert detector.source_grid_law == detector.get_axis_type(), (
                f"{detector.name} asks for TIC_PRESERVING but its source grid "
                "law does not match the axis it requests"
            )


class TestInstrumentDetectors:
    """Test individual instrument detectors."""

    def test_timstof_detector(self):
        """Test TimsTOF detector matching."""
        detector = TimsTOFDetector()
        assert detector.name == "timsTOF"

        # Should match when is_timstof is True
        chars = DataCharacteristics(is_timstof=True)
        assert detector.matches(chars)

        # Should not match when is_timstof is False
        chars = DataCharacteristics(is_timstof=False)
        assert not detector.matches(chars)

        assert detector.get_resampling_method() == ResamplingMethod.NEAREST_NEIGHBOR
        assert detector.get_axis_type() == AxisType.REFLECTOR_TOF

    def test_rapiflex_detector(self):
        """Test Rapiflex detector matching."""
        detector = RapiflexDetector()
        assert detector.name == "Rapiflex MALDI-TOF"

        # Should match Rapiflex format
        chars = DataCharacteristics(is_rapiflex_format=True)
        assert detector.matches(chars)

        # Should match Bruker MALDI-TOF
        chars = DataCharacteristics(
            instrument_type="MALDI-TOF",
            manufacturer="Bruker",
        )
        assert detector.matches(chars)

        # Must NOT match on peak density alone -- that is not a modality.
        chars = DataCharacteristics(
            spectrum_type=SpectrumType.PROFILE,
            total_peaks=10000000,
            n_spectra=1000,
        )
        assert chars.is_high_density_profile
        assert not detector.matches(chars)

        assert detector.get_resampling_method() == ResamplingMethod.TIC_PRESERVING
        assert detector.get_axis_type() == AxisType.CONSTANT

        # The source grid law it declares is what lets TIC_PRESERVING through
        # the chain's gate: RapiflexReader lays spectra out with np.linspace,
        # so source law == target law == CONSTANT.
        assert detector.source_grid_law == AxisType.CONSTANT
        assert detector.source_grid_law == detector.get_axis_type()

    def test_centroid_imzml_detector(self):
        """Test CentroidImzML detector matching."""
        detector = CentroidImzMLDetector()

        # Should match centroid data
        chars = DataCharacteristics(spectrum_type=SpectrumType.CENTROID)
        assert detector.matches(chars)

        # Should not match profile data
        chars = DataCharacteristics(spectrum_type=SpectrumType.PROFILE)
        assert not detector.matches(chars)

        assert detector.get_resampling_method() == ResamplingMethod.NEAREST_NEIGHBOR
        assert detector.get_axis_type() == AxisType.REFLECTOR_TOF

    def test_fticr_detector(self):
        """Test FTICR detector matching."""
        detector = FTICRDetector()

        chars = DataCharacteristics(instrument_type="FT-ICR")
        assert detector.matches(chars)

        chars = DataCharacteristics(instrument_type="Orbitrap")
        assert not detector.matches(chars)

        assert detector.get_axis_type() == AxisType.FTICR

    def test_orbitrap_detector(self):
        """Test Orbitrap detector matching."""
        detector = OrbitrapDetector()

        chars = DataCharacteristics(instrument_type="Orbitrap")
        assert detector.matches(chars)

        chars = DataCharacteristics(instrument_type="FT-ICR")
        assert not detector.matches(chars)

        assert detector.get_axis_type() == AxisType.ORBITRAP

    def test_default_detector(self):
        """Test DefaultDetector always matches."""
        detector = DefaultDetector()
        assert detector.name == "Unknown (default)"

        # Should always match
        chars = DataCharacteristics()
        assert detector.matches(chars)

        chars = DataCharacteristics(
            instrument_type="Unknown Instrument",
            spectrum_type="unknown",
        )
        assert detector.matches(chars)

        assert detector.get_resampling_method() == ResamplingMethod.NEAREST_NEIGHBOR
        assert detector.get_axis_type() == AxisType.CONSTANT


class TestDataCharacteristics:
    """Test DataCharacteristics dataclass."""

    def test_from_metadata_essential(self):
        """Test creating from essential metadata."""
        metadata = {
            "essential_metadata": {
                "spectrum_type": SpectrumType.CENTROID,
                "total_peaks": 1000000,
                "n_spectra": 500,
            }
        }
        chars = DataCharacteristics.from_metadata(metadata)

        assert chars.spectrum_type == SpectrumType.CENTROID
        assert chars.total_peaks == 1000000
        assert chars.n_spectra == 500
        assert chars.is_centroid_data

    def test_from_metadata_instrument_info(self):
        """Test extracting instrument info from metadata."""
        metadata = {
            "instrument_info": {
                "instrument_type": "MALDI-TOF",
                "manufacturer": "Bruker",
            }
        }
        chars = DataCharacteristics.from_metadata(metadata)

        assert chars.instrument_type == "MALDI-TOF"
        assert chars.manufacturer == "Bruker"

    def test_from_metadata_global_metadata(self):
        """Test extracting from GlobalMetadata."""
        metadata = {"GlobalMetadata": {"InstrumentName": "timsTOF Maldi 2"}}
        chars = DataCharacteristics.from_metadata(metadata)

        assert chars.instrument_name == "timsTOF Maldi 2"
        assert chars.is_timstof

    def test_is_high_density_profile(self):
        """Test high density profile detection."""
        # High density profile (>5000 peaks per spectrum)
        chars = DataCharacteristics(
            spectrum_type=SpectrumType.PROFILE,
            total_peaks=10000000,
            n_spectra=1000,
        )
        assert chars.is_high_density_profile
        assert chars.avg_peaks_per_spectrum == 10000.0

        # Low density profile
        chars = DataCharacteristics(
            spectrum_type=SpectrumType.PROFILE,
            total_peaks=1000,
            n_spectra=1000,
        )
        assert not chars.is_high_density_profile

        # Centroid data should not be high density profile
        chars = DataCharacteristics(
            spectrum_type=SpectrumType.CENTROID,
            total_peaks=10000000,
            n_spectra=1000,
        )
        assert not chars.is_high_density_profile

    def test_needs_resampling(self):
        """Test needs_resampling property."""
        # Continuous data doesn't need resampling
        chars = DataCharacteristics(has_shared_mass_axis=True)
        assert not chars.needs_resampling

        # Processed data needs resampling
        chars = DataCharacteristics(has_shared_mass_axis=False)
        assert chars.needs_resampling

    def test_is_maldi_tof(self):
        """Test MALDI-TOF detection."""
        # Rapiflex format
        chars = DataCharacteristics(is_rapiflex_format=True)
        assert chars.is_maldi_tof

        # Explicit MALDI-TOF type
        chars = DataCharacteristics(instrument_type="MALDI-TOF")
        assert chars.is_maldi_tof

        # Bruker with high-density profile
        chars = DataCharacteristics(
            manufacturer="Bruker",
            spectrum_type=SpectrumType.PROFILE,
            total_peaks=10000000,
            n_spectra=1000,
        )
        assert chars.is_maldi_tof
