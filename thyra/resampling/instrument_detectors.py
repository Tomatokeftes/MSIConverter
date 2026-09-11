"""Instrument detection strategies for resampling decisions.

This module implements the Strategy pattern for instrument detection.
Each detector class encapsulates the logic for identifying a specific
instrument type and returning appropriate resampling parameters.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .data_characteristics import DataCharacteristics
from .mass_axis.tof_generator import DEFAULT_BINS_PER_FWHM, MRT_TOF_LAW, TIMSTOF_TOF_LAW
from .types import AxisType, ResamplingMethod

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BINS_PER_FWHM",
    "MRT_TOF_LAW",
    "TIMSTOF_TOF_LAW",
]

#: ``(width_da, reference_mz)``: a bin width anchored at a reference m/z.
ReferenceWidth = Tuple[float, float]

#: Default bin width for Waters vendor-centroid conversions: 2 mDa at m/z
#: 1000. The previous 5 mDa gave 1.1 bins per measured FWHM at m/z 800 on a
#: SELECT SERIES MRT (FWHM 4.48 mDa); 2 mDa gives 2.8, is the coarsest
#: setting that clears two bins per peak width, and costs 24% more store.
WATERS_CENTROID_WIDTH: ReferenceWidth = (0.002, 1000.0)

#: Default bin width for SELECT SERIES MRT profile conversions: 1.3 mDa at
#: m/z 1000, about 1.14x the digitiser's own sample spacing there (1.143 mDa
#: predicted from Lteff, Veff and the 1.35 GHz ADC clock; 1.16 measured).
#: Pinned rather than derived so that every MRT run with the same mass
#: range lands on the same axis, whatever its calibration constants say.
WATERS_MRT_PROFILE_WIDTH: ReferenceWidth = (0.0013, 1000.0)

#: For a non-MRT Waters profile conversion the width follows the run's own
#: sample spacing, at this ratio, rounded up to the nearest 0.1 mDa. The
#: ratio is the MRT default's (1.3 / 1.143 = 1.14): bins at or just above
#: the sample spacing, so interpolation neither skips samples nor spreads
#: one sample over many bins.
WATERS_PROFILE_BINS_PER_SAMPLE = 1.14


class InstrumentDetector(ABC):
    """Abstract base class for instrument detection strategies.

    Each detector is responsible for:
    1. Detecting if data matches a specific instrument type
    2. Returning the appropriate resampling method
    3. Returning the appropriate mass axis type
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        pass

    @abstractmethod
    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check if data characteristics match this instrument.

        Args:
            characteristics: Extracted data characteristics

        Returns:
            True if this detector matches the data
        """
        pass

    @abstractmethod
    def get_resampling_method(self) -> ResamplingMethod:
        """Get the recommended resampling method for this instrument."""
        pass

    @abstractmethod
    def get_axis_type(self) -> AxisType:
        """Get the recommended mass axis type for this instrument."""
        pass

    @property
    def source_grid_law(self) -> Optional[AxisType]:
        """Spacing law of the *source* m/z grid, when it is known.

        ``tic_preserving`` interpolates onto the target axis and then applies
        one global scaling factor. That composite operator is exact only when
        the source grid follows the same spacing law as the target axis; off
        the diagonal it distorts the ratio between two peaks by exactly
        ``(m_hi / m_lo) ** (p_target - p_source)``, with ``p`` = 0, 0.5, 1,
        1.5, 2 for constant / linear_tof / reflector_tof / orbitrap / fticr.

        SCiLS Lab gates its own TIC-preserving resampling on the same
        condition -- "If all axis types are identical, a TIC preserving
        resampling is applied, otherwise a linear interpolation is performed"
        (SCiLS Lab 2026b User Guide, p.80).

        ``None`` -- the default -- means Thyra does not know the source law.
        That is the honest answer for every detector that matches on
        something other than a vendor format whose grid Thyra itself lays
        out, and :class:`InstrumentDetectorChain` refuses ``TIC_PRESERVING``
        for such a detector.
        """
        return None

    def get_reference_width(
        self, characteristics: DataCharacteristics
    ) -> Optional[ReferenceWidth]:
        """Default bin width for this instrument, as ``(width_da, reference_mz)``.

        ``None`` -- the default -- leaves the choice to the converter's
        per-axis-type defaults (5 mDa at m/z 1000, or 17 mDa at m/z 300 for
        ``linear_tof``). A detector overrides this when it knows what the
        data can support: a peak width it has measured, or a digitiser
        grid it wants the bins to track. The converter honours it only when
        the caller set neither ``--resample-width-at-mz`` nor
        ``--mass-axis-type``, since a width tuned for one axis law is not
        a sensible default for another.
        """
        return None

    def get_tof_law(
        self, characteristics: DataCharacteristics
    ) -> Optional[Tuple[float, float]]:
        """``(A, B)`` of the instrument's measured peak-width law, if known.

        ``FWHM(m) = sqrt(A m + B m^2)``, in mDa; see
        :mod:`thyra.resampling.mass_axis.tof_generator`. Used when the axis
        resolves to :attr:`AxisType.TOF` and the caller gave no pair of
        their own -- which is either this detector asking for that axis
        type itself, or the caller opting in with ``--mass-axis-type tof``
        on an instrument that declares a pair here but defaults to another
        law. ``None`` means no law has been measured for this instrument.
        """
        return None


class CentroidImzMLDetector(InstrumentDetector):
    """Detector for centroid ImzML data.

    Centroid data has discrete peaks and benefits from nearest-neighbor
    resampling with reflector TOF axis spacing (constant relative resolution).
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "ImzML Centroid"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check if data is centroid spectrum type."""
        return characteristics.is_centroid_data

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbor for centroid data."""
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return reflector TOF axis for constant relative resolution."""
        return AxisType.REFLECTOR_TOF


class RapiflexDetector(InstrumentDetector):
    """Detector for Bruker Rapiflex MALDI-TOF data.

    Rapiflex profile data uses TIC-preserving resampling with equidistant
    (constant) axis spacing, matching SCiLS Lab convention.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Rapiflex MALDI-TOF"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check if data came from Bruker flexImaging/Rapiflex.

        Both branches below are written by one producer, the Rapiflex
        metadata extractor, so a match here means the spectra arrive on the
        uniform-in-m/z grid ``RapiflexReader`` builds.

        Peak density is deliberately *not* consulted. It used to be: any
        profile data averaging more than
        ``Thresholds.PROFILE_PEAK_DENSITY`` points per spectrum was handed
        MALDI-TOF treatment regardless of what instrument produced it. That
        made the detector modality-blind -- a dense TOF-SIMS or Orbitrap
        profile acquisition would have been resampled by MALDI-TOF logic
        onto a MALDI-shaped axis, silently. SCiLS Lab does not guess
        modality either; its importer takes ``--project TIMSTOF|TOF|FT`` as
        an argument (2026b User Guide, p.81). Unknown-provenance profile
        data now falls through to :class:`DefaultDetector`, which bins
        counts rather than interpolating and so is safe for any modality.
        """
        # Direct Rapiflex format detection
        if characteristics.is_rapiflex_format:
            return True

        # Bruker MALDI-TOF detection
        return (
            characteristics.instrument_type == "MALDI-TOF"
            and characteristics.manufacturer == "Bruker"
        )

    def get_resampling_method(self) -> ResamplingMethod:
        """Return TIC-preserving for profile MALDI-TOF data."""
        return ResamplingMethod.TIC_PRESERVING

    def get_axis_type(self) -> AxisType:
        """Return constant axis matching SCiLS Lab convention."""
        return AxisType.CONSTANT

    @property
    def source_grid_law(self) -> Optional[AxisType]:
        """Report the constant law of the flexImaging source grid.

        ``RapiflexReader.get_common_mass_axis`` lays every spectrum out with
        ``np.linspace(mass_start, mass_end, n_points)``, so the source
        spacing is constant in m/z -- the same law as the ``constant`` target
        axis :meth:`get_axis_type` asks for. Source law equal to target law
        is what makes ``tic_preserving`` exact here, and it is the reason
        this is the only route on which the chain permits it.
        """
        return AxisType.CONSTANT


class TimsTOFDetector(InstrumentDetector):
    """Detector for Bruker timsTOF data.

    timsTOF produces centroid data with constant relative resolution,
    using nearest-neighbor resampling with reflector TOF axis spacing.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "timsTOF"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check if data is from timsTOF instrument."""
        return characteristics.is_timstof

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbor for timsTOF centroid data."""
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return reflector TOF axis for constant relative resolution.

        The measured law (:data:`TIMSTOF_TOF_LAW`) is within 10% of this
        axis's shape over m/z 400-1000 (the gap reaches 10% at m/z 300),
        so the default stays here and every timsTOF store already built
        keeps its axis; ``--mass-axis-type tof`` opts in to the measured
        pair.
        """
        return AxisType.REFLECTOR_TOF

    def get_tof_law(
        self, characteristics: DataCharacteristics
    ) -> Optional[Tuple[float, float]]:
        """The pair fitted on 180 timsTOF fleX peaks; opt-in, see above."""
        return TIMSTOF_TOF_LAW


class FTICRDetector(InstrumentDetector):
    """Detector for FT-ICR data.

    FT-ICR has quadratic mass axis spacing due to cyclotron frequency physics.
    Uses nearest-neighbor for centroid data.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "FT-ICR"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check if instrument type is FT-ICR."""
        return characteristics.instrument_type == "FT-ICR"

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbor for FT-ICR data."""
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return FTICR axis for quadratic mass scaling."""
        return AxisType.FTICR


class OrbitrapDetector(InstrumentDetector):
    """Detector for Orbitrap data.

    Orbitrap has 1/sqrt(m/z) resolution scaling.
    Uses nearest-neighbor for centroid data.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Orbitrap"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check if instrument type is Orbitrap."""
        return characteristics.instrument_type == "Orbitrap"

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbor for Orbitrap data."""
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return Orbitrap axis for sqrt(m/z) resolution scaling."""
        return AxisType.ORBITRAP


class PhiToFSIMSDetector(InstrumentDetector):
    """Detector for PHI SmartSoft-TOF ToF-SIMS data (nanoTOF instruments).

    Without this detector PHI data reaches :class:`DefaultDetector`, which
    reports ``CONSTANT``. That answer is wrong twice over, and the second
    way is expensive: a downstream caller that maps a constant axis to
    profile-MALDI conventions gets ``tic_preserving`` onto an equidistant
    axis, and interpolating a PHI pixel -- a median of 44 measured points
    spread over m/z 0.5-1850 -- fabricates intensity in every bin between
    them. The TIC rescale then hides the damage behind a total that still
    balances.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "PHI SmartSoft-TOF (ToF-SIMS)"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check whether the reader stamped the PHI raw format."""
        return characteristics.is_phi_tofsims

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbour: PHI pixels are sparse, not profile.

        The instrument records individual ion arrivals, so a pixel holds
        only the channels that happened to fire -- 64 occupied channels out
        of 863,670 on the reference acquisition. That is centroid-like data
        whatever the axis says, and interpolating across the gaps between
        those points invents signal. Nearest-neighbour cannot.
        """
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return the linear-TOF law the detector's own grid follows."""
        return AxisType.LINEAR_TOF

    @property
    def source_grid_law(self) -> Optional[AxisType]:
        """Report the linear-TOF law of the source grid.

        ``PhiMassAxis`` lays channels out at a constant flight-time step
        (``SpecBinSize``, 128 ps on the reference file) and derives m/z as
        ``(slope * t + offset) ** 2``. Constant steps in time put the m/z
        spacing proportional to ``sqrt(m/z)``, which is exactly
        :attr:`AxisType.LINEAR_TOF`. Measured on the reference acquisition:
        9.8e-5 u per channel at m/z 1 against 5.0e-4 at m/z 26, a ratio of
        5.1 where ``sqrt(26)`` is 5.099.

        Declaring it is documentation of the acquisition, not a switch.
        ``_gate_tic_preserving`` governs auto-selection only, and this
        detector's :meth:`get_resampling_method` returns nearest-neighbour,
        so the gate returns that before it ever reads this property. A
        caller who asks for ``tic_preserving`` by name does not reach the
        gate either: an explicit method is taken as given, with a warning
        when it contradicts the detector (D15). An earlier version of this
        docstring claimed the opposite and was wrong on both halves.
        """
        return AxisType.LINEAR_TOF


class WatersProfileDetector(InstrumentDetector):
    """Detector for the profile trace of a Waters MassLynx .raw imaging run.

    The trace is the digitiser's own record: samples at a fixed ADC clock
    rate, so uniformly spaced in flight time, which puts their m/z spacing
    proportional to ``sqrt(m/z)``. Measured as ``(m/z)^0.494 +- 0.005``
    (R2 = 0.9992) on a SELECT SERIES MRT and ``(m/z)^0.498 +- 0.002``
    (R2 = 0.9999) on a Synapt G2-Si -- the same law on both, at 1.1 mDa
    and 13 mDa per sample respectively near m/z 1000. That is exactly
    :attr:`AxisType.LINEAR_TOF`, so a ``linear_tof`` target axis holds the
    bin-to-sample ratio flat across the range, where ``reflector_tof``
    drifts from 1.15x at m/z 400 to 1.78x at m/z 800 on the same run.

    The trace is zero-suppressed: only samples around peaks are stored,
    with an explicit zero at either edge of each cluster. Profile samples
    have width, so binning them by nearest neighbour onto an axis at about
    the sample spacing puts two samples in one bin for ~9% of bins and
    none in others; ``tic_preserving`` interpolates instead, which on a
    fixed grid gives neither, and the explicit zeros mean it never draws a
    line across an unmeasured gap.

    This detector is reached by default only on a SELECT SERIES MRT, whose
    reader defaults to the profile trace; every other Waters instrument
    arrives here only when asked for the trace with ``--waters-spectrum
    profile``, and then gets the same treatment with a bin width taken
    from its own sample spacing.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Waters MassLynx (profile trace)"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """A MassLynx raw run whose reader delivers the profile trace."""
        return characteristics.is_waters_raw and characteristics.is_profile_data

    def get_resampling_method(self) -> ResamplingMethod:
        """Return TIC-preserving interpolation: samples have width."""
        return ResamplingMethod.TIC_PRESERVING

    def get_axis_type(self) -> AxisType:
        """Return linear TOF: the digitiser grid's own ``sqrt(m/z)`` law."""
        return AxisType.LINEAR_TOF

    @property
    def source_grid_law(self) -> Optional[AxisType]:
        """The stored trace is uniform in flight time: ``sqrt(m/z)`` spacing.

        Measured on two instruments (see the class docstring); this is what
        lets ``tic_preserving`` clear ``_gate_tic_preserving``, and it is
        the reason the target axis is ``linear_tof`` and not the centroid
        route's ``reflector_tof``.
        """
        return AxisType.LINEAR_TOF

    def get_reference_width(
        self, characteristics: DataCharacteristics
    ) -> Optional[ReferenceWidth]:
        """1.3 mDa at m/z 1000 on an MRT; the run's own sample spacing elsewhere.

        An MRT is pinned to :data:`WATERS_MRT_PROFILE_WIDTH` so that every
        MRT run with the same mass range shares one axis. Another Waters
        instrument asked for its profile gets
        :data:`WATERS_PROFILE_BINS_PER_SAMPLE` times its predicted sample
        spacing at m/z 1000, rounded up to 0.1 mDa (a Synapt G2-Si: 13.8 mDa
        per sample, so 15.8 mDa bins), and falls back to the MRT width with
        a warning when ``_extern.inf`` lacks the constants to predict it.
        """
        if characteristics.is_waters_mrt:
            return WATERS_MRT_PROFILE_WIDTH
        spacing = characteristics.profile_sample_spacing_da_at_1000
        if spacing is None:
            logger.warning(
                "Waters profile conversion on an instrument that is not an "
                "MRT, and _extern.inf did not yield Lteff, Veff and the ADC "
                "sample frequency to predict its sample spacing; defaulting "
                "to the MRT bin width of %.1f mDa at m/z %.0f. Set "
                "--resample-width-at-mz to the run's sample spacing if the "
                "store comes out oversampled.",
                WATERS_MRT_PROFILE_WIDTH[0] * 1e3,
                WATERS_MRT_PROFILE_WIDTH[1],
            )
            return WATERS_MRT_PROFILE_WIDTH
        width_mda = math.ceil(spacing * 1e3 * WATERS_PROFILE_BINS_PER_SAMPLE * 10) / 10
        width = (width_mda / 1e3, 1000.0)
        logger.info(
            "Waters profile bin width from the run's own digitiser: %.3f mDa "
            "per sample at m/z 1000, so %.1f mDa bins",
            spacing * 1e3,
            width_mda,
        )
        return width


class WatersMRTCentroidDetector(InstrumentDetector):
    """Detector for the vendor-centroid list of a SELECT SERIES MRT run.

    A centroid list's bins should track peak width, and the MRT's measured
    width follows neither single-term TOF law: over m/z 300-1000 the
    log-log exponent is 0.67 +- 0.03, between ``linear_tof``'s 0.5 and
    ``reflector_tof``'s 1.0. The two-term law ``sqrt(A m + B m^2)`` with
    :data:`MRT_TOF_LAW` reproduces the measured 2.97 / 3.79 / 4.54 mDa at
    m/z 400 / 600 / 800, so the axis is laid at that width over
    :data:`DEFAULT_BINS_PER_FWHM` bins per peak.

    The MRT's *profile* trace is a different matter -- its bins follow the
    digitiser grid, not the peak width -- and :class:`WatersProfileDetector`
    handles it ahead of this one.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Waters SELECT SERIES MRT (vendor centroid)"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """An MRT run whose reader delivers something other than the trace."""
        return (
            characteristics.is_waters_raw
            and characteristics.is_waters_mrt
            and not characteristics.is_profile_data
        )

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbour, which bins counts and cannot invent them."""
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return the two-term TOF law: bins that follow the measured width."""
        return AxisType.TOF

    def get_tof_law(
        self, characteristics: DataCharacteristics
    ) -> Optional[Tuple[float, float]]:
        """The pair fitted on 229 MRT peaks."""
        return MRT_TOF_LAW

    # ``get_reference_width`` stays ``None``: the TOF axis is sized in bins
    # per peak width, and ``DEFAULT_BINS_PER_FWHM`` is that default.


class WatersDetector(InstrumentDetector):
    """Detector for the vendor-centroid list of a Waters MassLynx .raw run.

    Without this detector a Waters acquisition's fate hinged on its declared
    spectrum representation: centroid files happened to land on
    :class:`CentroidImzMLDetector` -- the right answer, reached by accident on
    a detector whose name says imzML -- while profile files fell through to
    :class:`DefaultDetector`. That reported ``CONSTANT``, which downstream
    pairs with the 0.1 Da default bin width: R = 5,000 at m/z 500, on an
    instrument built for 100,000+.

    The profile trace is handled by :class:`WatersProfileDetector`, which
    sits ahead of this one in the chain; what reaches here is the vendor
    peak list, or a run whose representation is undeclared.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Waters MassLynx"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Check whether the Waters extractor stamped the MassLynx raw format."""
        return characteristics.is_waters_raw

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbour, which bins counts and cannot invent them."""
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return reflector TOF: bin width proportional to m/z.

        SCiLS Lab 2026b calls the same law Orthogonal TOF (formerly Reflector
        TOF): bin size proportional to m/z, i.e. constant relative resolution
        (2026b User Guide, p.75). For a centroid list the axis should track
        peak width, and measured MRT peak widths go as ``(m/z)^0.67`` --
        between this law and ``linear_tof``, and this one errs toward more
        bins at low m/z, where the width measurement is least certain.
        """
        return AxisType.REFLECTOR_TOF

    def get_reference_width(
        self, characteristics: DataCharacteristics
    ) -> Optional[ReferenceWidth]:
        """2 mDa at m/z 1000; see :data:`WATERS_CENTROID_WIDTH`."""
        return WATERS_CENTROID_WIDTH

    # ``source_grid_law`` stays at the inherited ``None`` on purpose: a
    # centroid list has no grid. The profile trace's law is declared on
    # :class:`WatersProfileDetector`, where it has been measured.


class DefaultDetector(InstrumentDetector):
    """Fallback detector for unknown instruments.

    Uses conservative defaults: nearest-neighbor resampling with
    constant (equidistant) axis spacing.
    """

    @property
    def name(self) -> str:
        """Return detector name."""
        return "Unknown (default)"

    def matches(self, characteristics: DataCharacteristics) -> bool:
        """Always return True as fallback detector."""
        return True

    def get_resampling_method(self) -> ResamplingMethod:
        """Return nearest-neighbor as safe default."""
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self) -> AxisType:
        """Return constant axis as safe default."""
        return AxisType.CONSTANT


class InstrumentDetectorChain:
    """Chain of instrument detectors using Chain of Responsibility pattern.

    Iterates through detectors in priority order until one matches.
    The order is important - more specific detectors should come first.
    """

    def __init__(self, detectors: Optional[List[InstrumentDetector]] = None):
        """Initialize detector chain.

        Args:
            detectors: List of detectors in priority order.
                      If None, uses default detector chain.
        """
        if detectors is None:
            detectors = self._default_detectors()
        self.detectors = detectors

    @staticmethod
    def _default_detectors() -> List[InstrumentDetector]:
        """Create default detector chain in priority order."""
        return [
            # Specific instrument detectors first
            TimsTOFDetector(),
            RapiflexDetector(),
            FTICRDetector(),
            OrbitrapDetector(),
            PhiToFSIMSDetector(),
            # The profile trace first: it needs the representation as well
            # as the format, and the centroid detector matches on the
            # format alone.
            WatersProfileDetector(),
            WatersMRTCentroidDetector(),
            WatersDetector(),
            # Generic spectrum type detector
            CentroidImzMLDetector(),
            # Fallback last
            DefaultDetector(),
        ]

    def detect(self, characteristics: DataCharacteristics) -> InstrumentDetector:
        """Find the first matching detector.

        Args:
            characteristics: Data characteristics to match

        Returns:
            The first detector that matches
        """
        for detector in self.detectors:
            if detector.matches(characteristics):
                logger.info(f"Detected instrument type: {detector.name}")
                return detector

        # Should never reach here due to DefaultDetector
        return DefaultDetector()

    def get_resampling_method(
        self, characteristics: DataCharacteristics
    ) -> ResamplingMethod:
        """Get resampling method for the detected instrument.

        ``TIC_PRESERVING`` additionally has to clear the matching-axis-law
        gate; see :meth:`_gate_tic_preserving`.

        Args:
            characteristics: Data characteristics to match

        Returns:
            Recommended resampling method for the instrument
        """
        detector = self.detect(characteristics)
        method = self._gate_tic_preserving(detector)
        logger.info(f"Selected resampling method: {method.name}")
        return method

    def get_resampling_method_for_axis(
        self, characteristics: DataCharacteristics, axis_type: AxisType
    ) -> ResamplingMethod:
        """Select the method for an axis the caller has already settled.

        :meth:`get_resampling_method` gates against the axis the detector
        would have chosen. That is the right answer only while nothing
        overrides it, and ``--mass-axis-type`` does: the converter resolves
        the axis separately and later, so the gate could clear on the
        detector's axis and the conversion then build a different one.

        Call this once the target axis is known. It is the same gate against
        the axis that will actually be laid.

        Args:
            characteristics: Data characteristics to match.
            axis_type: The axis the conversion will build.

        Returns:
            The detector's method, or ``NEAREST_NEIGHBOR`` in its place.
        """
        detector = self.detect(characteristics)
        return self._gate_tic_preserving(detector, axis_type)

    @staticmethod
    def _gate_tic_preserving(
        detector: InstrumentDetector,
        axis_type: Optional[AxisType] = None,
    ) -> ResamplingMethod:
        """Allow ``TIC_PRESERVING`` only when source and target laws agree.

        SCiLS Lab applies TIC-preserving resampling to profile data only
        when all the mass axes being combined are of the same type, and
        linear interpolation otherwise (2026b User Guide, p.80). That is
        also precisely the condition under which Thyra's operator --
        interpolate, then rescale by one global factor -- is exact. See
        :attr:`InstrumentDetector.source_grid_law` for the error off the
        diagonal.

        A detector that has not declared its source grid law does not clear
        the gate. Auto-selection therefore cannot reach the interpolating
        path on data whose acquisition Thyra has not actually identified.

        Args:
            detector: The detector that matched.
            axis_type: The axis the conversion will build. Defaults to the
                detector's own choice, which is correct only while nothing
                overrides it -- see
                :meth:`get_resampling_method_for_axis`.

        Returns:
            The detector's method, or ``NEAREST_NEIGHBOR`` in its place.
        """
        method = detector.get_resampling_method()
        if method is not ResamplingMethod.TIC_PRESERVING:
            return method

        if axis_type is None:
            axis_type = detector.get_axis_type()
        source_law = detector.source_grid_law
        if source_law is axis_type:
            return method

        logger.info(
            "%s asks for TIC_PRESERVING onto a %s axis, but the source grid "
            "law is %s. TIC-preserving resampling is exact only when the two "
            "match, so NEAREST_NEIGHBOR is used instead.",
            detector.name,
            axis_type.name,
            "unknown" if source_law is None else source_law.name,
        )
        return ResamplingMethod.NEAREST_NEIGHBOR

    def get_axis_type(self, characteristics: DataCharacteristics) -> AxisType:
        """Get axis type for the detected instrument.

        Args:
            characteristics: Data characteristics to match

        Returns:
            Recommended axis type for the instrument
        """
        detector = self.detect(characteristics)
        axis_type = detector.get_axis_type()
        logger.info(f"Selected axis type: {axis_type.name}")
        return axis_type

    def get_reference_width(
        self, characteristics: DataCharacteristics
    ) -> Optional[ReferenceWidth]:
        """Default bin width for the detected instrument, if it declares one.

        Args:
            characteristics: Data characteristics to match

        Returns:
            ``(width_da, reference_mz)``, or ``None`` to leave the choice to
            the converter's per-axis-type defaults.
        """
        detector = self.detect(characteristics)
        width = detector.get_reference_width(characteristics)
        if width is not None:
            logger.info(
                "Selected bin width: %.2f mDa at m/z %.0f (%s)",
                width[0] * 1e3,
                width[1],
                detector.name,
            )
        return width

    def get_tof_law(
        self, characteristics: DataCharacteristics
    ) -> Optional[Tuple[float, float]]:
        """The detected instrument's two-term width law, if it declares one.

        Args:
            characteristics: Data characteristics to match

        Returns:
            ``(A, B)`` in the units of
            :mod:`thyra.resampling.mass_axis.tof_generator`, or ``None``.
        """
        detector = self.detect(characteristics)
        law = detector.get_tof_law(characteristics)
        if law is not None:
            logger.info(
                "Selected TOF width law: A=%.4g mDa^2/Da, B=%.4g (%s)",
                law[0],
                law[1],
                detector.name,
            )
        return law
