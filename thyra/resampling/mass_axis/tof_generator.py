"""Two-term time-of-flight width law: bins that follow a measured peak width.

A TOF peak's width in flight time has a part that is constant (detector
and digitiser response, pusher timing) and a part proportional to the
flight time itself (energy spread, turnaround time), added in quadrature.
Converted to m/z that is

    FWHM(m) = sqrt(A * m + B * m^2)        m in Da, FWHM in mDa

with ``A`` in mDa^2/Da and ``B`` dimensionless. The two existing TOF laws
are its limits: ``linear_tof`` is ``B = 0`` (width goes as ``sqrt(m)``),
``reflector_tof`` is ``A = 0`` (width goes as ``m``, constant ppm, with
``1 / sqrt(B)`` the resolving power). Real instruments sit between them,
and where they sit is measurable:

======================  ==========  ===========  ============================
instrument              A           B            fitted from
======================  ==========  ===========  ============================
SELECT SERIES MRT       0.0185      9.1e-6       229 peaks, m/z 300-1000
timsTOF fleX            0.0877      8.74e-4      180 peaks, m/z 300-1000
======================  ==========  ===========  ============================

Refitted here from the same peak lists by least squares on ``FWHM^2 = A m +
B m^2``: the MRT pair reproduces 2.97 / 3.79 / 4.54 mDa at m/z 400 / 600 /
800, where the log-log fit of the same peaks gave an exponent of 0.67 --
between the 0.5 and 1.0 the two single-term laws allow, which is why
neither of them fits an MRT centroid list exactly.

The axis lays bins at ``FWHM(m) / k`` for ``k`` bins per peak width. The
cumulative bin count ``N(m) = 1000 k * integral dm / sqrt(A m + B m^2)`` has
a closed form, ``(2 / sqrt(B)) asinh(sqrt(B m / A))``, so the axis is a
uniform grid in that variable; the two limits reduce to the uniform-in-
``sqrt(m)`` and uniform-in-``ln(m)`` grids the single-term generators lay.

This law describes **peak width**, which is what a centroid list's bins
should track. It says nothing about a digitiser's sample grid, so it must
not be used to bin profile data: the Waters profile trace follows
``linear_tof`` because that is how its samples are spaced, not because of
how wide its peaks are.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...errors import ConversionRefused
from ..types import AxisType, MassAxis
from .base_generator import BaseAxisGenerator

#: ``(A, B)`` fitted on 229 isolated peaks of a SELECT SERIES MRT MALDI run.
MRT_TOF_LAW: Tuple[float, float] = (0.0185, 9.1e-6)

#: ``(A, B)`` fitted on 180 isolated peaks of a timsTOF fleX MALDI run.
TIMSTOF_TOF_LAW: Tuple[float, float] = (0.0877, 8.74e-4)

#: Bins per peak width when the caller gives no width of their own.
DEFAULT_BINS_PER_FWHM = 3.0


def tof_fwhm_mda(mz, a: float, b: float):
    """Peak width in mDa at ``mz`` under the two-term law."""
    mz = np.asarray(mz, dtype=np.float64)
    return np.sqrt(a * mz + b * mz * mz)


def _check_law(a: float, b: float) -> None:
    if not (np.isfinite(a) and np.isfinite(b)) or a < 0 or b < 0 or (a == 0 and b == 0):
        raise ConversionRefused(
            f"A TOF width law needs A >= 0 and B >= 0 with at least one of them "
            f"positive; got A={a!r}, B={b!r}"
        )


class TOFAxisGenerator(BaseAxisGenerator):
    """Mass axis whose bin width tracks ``sqrt(A m + B m^2) / k``.

    Parameters
    ----------
    a, b : float
        The width law's coefficients (mDa^2/Da and dimensionless).
    """

    def __init__(self, a: float, b: float) -> None:
        """Bind the law's coefficients, refusing a pair with no width."""
        _check_law(a, b)
        self.a = float(a)
        self.b = float(b)

    def cumulative(self, mz):
        """``integral dm / sqrt(A m + B m^2)`` from 0, in Da / mDa.

        Multiplied by ``1000 k`` this is the bin count below ``mz``.
        """
        mz = np.asarray(mz, dtype=np.float64)
        if self.b == 0.0:
            return 2.0 * np.sqrt(mz / self.a)
        if self.a == 0.0:
            return np.log(mz) / np.sqrt(self.b)
        return (2.0 / np.sqrt(self.b)) * np.arcsinh(np.sqrt(self.b * mz / self.a))

    def inverse(self, u):
        """The m/z at which :meth:`cumulative` equals ``u``."""
        u = np.asarray(u, dtype=np.float64)
        if self.b == 0.0:
            return self.a * (u / 2.0) ** 2
        if self.a == 0.0:
            return np.exp(u * np.sqrt(self.b))
        return (self.a / self.b) * np.sinh(u * np.sqrt(self.b) / 2.0) ** 2

    def bins_per_fwhm_for(self, reference_mz: float, reference_width: float) -> float:
        """The ``k`` that puts a bin of ``reference_width`` Da at ``reference_mz``."""
        return float(
            tof_fwhm_mda(reference_mz, self.a, self.b) / (reference_width * 1e3)
        )

    def bin_count(self, min_mz: float, max_mz: float, bins_per_fwhm: float) -> int:
        """Bins over ``[min_mz, max_mz]`` at ``bins_per_fwhm`` per peak width."""
        span = float(self.cumulative(max_mz) - self.cumulative(min_mz))
        return max(1, int(1e3 * bins_per_fwhm * span))

    def bin_width_at(self, mz, bins_per_fwhm: float):
        """Bin width in Da at ``mz`` for ``bins_per_fwhm`` per peak width."""
        return tof_fwhm_mda(mz, self.a, self.b) * 1e-3 / bins_per_fwhm

    def generate_axis(
        self,
        min_mz: float,
        max_mz: float,
        target_bins: int,
        reference_mz: float = 1000.0,
        reference_width: float = 0.005,
    ) -> MassAxis:
        """Distribute ``target_bins`` bins uniformly in the law's cumulative.

        ``reference_mz`` and ``reference_width`` are not needed: the bin
        count already fixes ``k``, and the law fixes the shape.
        """
        u = np.linspace(
            self.cumulative(min_mz), self.cumulative(max_mz), target_bins + 1
        )
        edges = self.inverse(u)
        # The inverse of the cumulative is exact in exact arithmetic; pin
        # the ends so the axis spans precisely what was asked for.
        edges[0], edges[-1] = min_mz, max_mz
        centres = (edges[:-1] + edges[1:]) / 2.0
        return MassAxis(
            mz_values=centres,
            min_mz=float(centres[0]),
            max_mz=float(centres[-1]),
            num_bins=len(centres),
            axis_type=AxisType.TOF,
        )

    def get_axis_type(self) -> AxisType:
        """Return the axis type this generator produces."""
        return AxisType.TOF
