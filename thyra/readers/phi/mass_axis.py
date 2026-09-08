# thyra/readers/phi/mass_axis.py
"""Flight-time to m/z conversion for PHI SmartSoft-TOF data.

Flight times are stored in picoseconds at finer granularity than the detector
time channel, so events are binned onto the channel grid implied by
``SpecBinSize`` before being mapped to m/z. The conversion itself is the
standard time-of-flight relation::

    sqrt(m/z) = (Mass/Time) * t_us + MassOffset

Below ``t = -MassOffset / (Mass/Time)`` the root is non-positive and no mass
is defined, so the usable axis is clipped to the mass range the acquisition
declares.
"""

import logging
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ...errors import ConversionRefused

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhiMassAxis:
    """A common m/z axis on the detector's time-channel grid.

    Attributes:
        mz: Strictly increasing m/z values, one per retained channel.
        channel_offset: Index of ``mz[0]`` within the full channel grid.
        bin_ps: Channel width in picoseconds.
        start_ps: Flight time of channel zero of the full grid.
        slope: ``Mass/Time`` used to build the axis.
        offset: ``MassOffset`` used to build the axis.
    """

    mz: NDArray[np.float64]
    channel_offset: int
    bin_ps: float
    start_ps: float
    slope: float
    offset: float

    def __len__(self) -> int:
        return int(self.mz.size)

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Lowest and highest m/z on the axis."""
        return (float(self.mz[0]), float(self.mz[-1]))

    @property
    def tof_us(self) -> NDArray[np.float64]:
        """Flight time in microseconds for each channel of :attr:`mz`.

        This is the quantity the instrument actually measured; ``mz`` is
        derived from it. Keeping it alongside the m/z axis makes the
        calibration reversible -- a different slope and offset can be applied
        to these times without re-reading the raw file.
        """
        channels = self.channel_offset + np.arange(self.mz.size, dtype=np.float64)
        return (self.start_ps + channels * self.bin_ps) * 1e-6

    def mz_to_tof_us(self, mz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Invert the calibration, mapping m/z back to flight time.

        ``sqrt(m/z) = slope * t + offset`` rearranges to
        ``t = (sqrt(m/z) - offset) / slope``.
        """
        return (np.sqrt(np.asarray(mz, dtype=np.float64)) - self.offset) / self.slope

    def to_channel(
        self, tof_ps: NDArray[np.int64]
    ) -> Tuple[NDArray[np.int64], NDArray[np.bool_]]:
        """Bin flight times onto the axis.

        Args:
            tof_ps: Flight times in picoseconds.

        Returns:
            Tuple of channel indices into :attr:`mz`, and a mask marking which
            entries fall inside the axis. Indices where the mask is ``False``
            are meaningless and must not be used.
        """
        raw = np.rint((tof_ps - self.start_ps) / self.bin_ps).astype(np.int64)
        raw -= self.channel_offset
        return raw, (raw >= 0) & (raw < self.mz.size)


def build_mass_axis(
    slope: float,
    offset: float,
    start_flight_time_us: float,
    stop_flight_time_us: float,
    bin_size_ns: float,
    start_mz: float = 0.0,
    stop_mz: float = 0.0,
) -> PhiMassAxis:
    """Build the common m/z axis for a PHI acquisition.

    Args:
        slope: ``Mass/Time`` in sqrt(amu) per microsecond.
        offset: ``MassOffset`` in sqrt(amu).
        start_flight_time_us: First retained flight time.
        stop_flight_time_us: Last retained flight time.
        bin_size_ns: Detector time-channel width in nanoseconds.
        start_mz: Declared low end of the mass range; ignored when zero.
        stop_mz: Declared high end of the mass range; ignored when zero.

    Returns:
        The mass axis.

    Raises:
        ValueError: If the parameters leave no usable channel.
    """
    if bin_size_ns <= 0:
        raise ConversionRefused(f"SpecBinSize must be positive, got {bin_size_ns}")

    bin_ps = bin_size_ns * 1000.0
    start_ps = start_flight_time_us * 1e6
    stop_ps = stop_flight_time_us * 1e6
    n_full = int(math.floor((stop_ps - start_ps) / bin_ps)) + 1
    if n_full <= 0:
        raise ConversionRefused(
            f"Flight-time range {start_flight_time_us}-{stop_flight_time_us} us "
            f"yields no channels at {bin_size_ns} ns per channel"
        )

    t_us = (start_ps + np.arange(n_full, dtype=np.float64) * bin_ps) * 1e-6
    root = slope * t_us + offset
    usable = root > 0.0
    mz_full = np.where(usable, root * root, 0.0)

    if start_mz > 0:
        usable &= mz_full >= start_mz
    if stop_mz > 0:
        usable &= mz_full <= stop_mz

    hits = np.nonzero(usable)[0]
    if hits.size == 0:
        raise ConversionRefused(
            "No time channel maps into the declared mass range "
            f"{start_mz}-{stop_mz} with slope={slope}, offset={offset}"
        )
    lo, hi = int(hits[0]), int(hits[-1])
    # m/z increases monotonically with flight time once the root is positive,
    # so the usable channels form one contiguous run.
    mz = mz_full[lo : hi + 1]

    logger.debug(
        "PHI mass axis: %d channels spanning m/z %.4f-%.4f (channels %d-%d of %d)",
        mz.size,
        mz[0],
        mz[-1],
        lo,
        hi,
        n_full,
    )
    return PhiMassAxis(
        mz=mz,
        channel_offset=lo,
        bin_ps=bin_ps,
        start_ps=start_ps,
        slope=slope,
        offset=offset,
    )
