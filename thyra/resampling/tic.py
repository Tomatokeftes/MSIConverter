"""Total ion current preservation, shared by both resampling paths.

Thyra interpolates spectra onto a common mass axis in two places: the
converters' hot per-pixel loop
(``BaseSpatialDataConverter._tic_preserving_resample``) and the public
``TICPreservingStrategy``. Both are supposed to honour the same contract --
``ResamplingMethod.TIC_PRESERVING`` promises that "the total ion count is
preserved after rebinning" -- and they drifted apart, with the converter
losing its rescaling step entirely. Keeping the rule in one module is what
stops that happening again.

Why rescaling is needed at all
------------------------------

Linear interpolation *samples* the spectrum at every target point, so the
sum of the interpolated values scales with how densely the target axis is
laid out. Interpolating 4,000 source points onto a 190,000-bin axis returns
roughly 47x the input TIC, and a 150-peak centroid spectrum over 1000x. The
sum only comes out right when the target axis happens to match the source
point density. Rescaling to a TIC computed from the *source* removes that
dependence on axis width.

Why cropping has to reduce the preserved TIC
--------------------------------------------

Interpolation drops everything outside the axis range. Rescaling to the full
input TIC would push that dropped intensity back into the bins that survive,
so a window cropped with ``--resample-min-mz`` / ``--resample-max-mz`` would
claim every ion from the parts that were cut away. The preserved total is
therefore the share of the spectrum that lies inside the axis range.

Why that share is measured by integral, not by counting samples
---------------------------------------------------------------

The obvious implementation -- sum the source intensities whose m/z falls
inside the axis range -- is wrong whenever the target axis is narrower than
the spacing between source points. A 0.6 Da window over a spectrum sampled
every 1 Da can contain no source points at all, and the sum-based rule then
reports zero and blanks a spectrum that interpolation would have represented
perfectly well. Integrating handles that continuously: a narrow window keeps
a proportionally small share rather than falling off a cliff.

The common case, where the axis spans the whole spectrum, is short-circuited
so it returns the input TIC exactly rather than to within rounding.
"""

from typing import Any, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

# ``rescale_to_preserved_tic`` returns the very array it was handed, so the
# element type it was called with is the element type it gives back.
_FloatT = TypeVar("_FloatT", bound=np.floating[Any])


def preserved_tic(
    mzs: npt.NDArray[np.floating[Any]],
    intensities: npt.NDArray[np.floating[Any]],
    axis_min: float,
    axis_max: float,
) -> float:
    """Return the share of a spectrum's TIC that survives a target axis range.

    Args:
        mzs: Source m/z values, ascending. Callers sort before calling;
            the result is meaningless on unsorted input.
        intensities: Source intensities, parallel to ``mzs``.
        axis_min: Lowest m/z on the target axis.
        axis_max: Highest m/z on the target axis.

    Returns:
        The total ion current the resampled spectrum should carry. Equal to
        ``intensities.sum()`` when the axis spans the spectrum, and ``0.0``
        when the two ranges do not overlap.
    """
    if mzs.size == 0:
        return 0.0

    total_tic = float(np.sum(intensities))
    if total_tic <= 0.0:
        return 0.0

    # The axis covers the whole spectrum: nothing is dropped. Short-circuit
    # so the overwhelmingly common case is exact and cheap.
    if axis_min <= mzs[0] and axis_max >= mzs[-1]:
        return total_tic

    lo = max(axis_min, float(mzs[0]))
    hi = min(axis_max, float(mzs[-1]))
    if hi <= lo:
        # Disjoint, or touching at a single point of zero width.
        return 0.0

    total_integral = float(np.trapezoid(intensities, mzs))
    if total_integral <= 0.0:
        # Degenerate sampling -- repeated m/z values, or a spectrum whose
        # points carry intensity but enclose no area. Integration says
        # nothing useful here, so fall back to the samples inside the range.
        inside = (mzs >= lo) & (mzs <= hi)
        return float(np.sum(intensities[inside]))

    # Integrate the source over the overlap, using interpolated values at the
    # two cut points so a window narrower than the source spacing still gets
    # a sensible share rather than zero.
    interior = mzs[(mzs > lo) & (mzs < hi)]
    kept_mz = np.concatenate(([lo], interior, [hi]))
    kept_integral = float(np.trapezoid(np.interp(kept_mz, mzs, intensities), kept_mz))

    fraction = min(max(kept_integral / total_integral, 0.0), 1.0)
    return total_tic * fraction


def rescale_to_preserved_tic(
    resampled: npt.NDArray[_FloatT],
    axis: npt.NDArray[np.floating[Any]],
    mzs: npt.NDArray[np.floating[Any]],
    intensities: npt.NDArray[np.floating[Any]],
    axis_range: Optional[Tuple[float, float]] = None,
) -> npt.NDArray[_FloatT]:
    """Scale an interpolated spectrum onto the TIC it is meant to carry.

    Modifies ``resampled`` in place and returns it.

    Args:
        resampled: Interpolated intensities on ``axis``, as returned by
            ``np.interp`` with out-of-range values set to zero.
        axis: The target mass axis, ascending.
        mzs: Source m/z values, ascending.
        intensities: Source intensities, parallel to ``mzs``.
        axis_range: The m/z range the axis was built to cover, when that is
            wider than the axis points themselves. A physics axis holds bin
            *centres*, so it stops half a bin short of the range it was
            asked for at either end, and measuring the preserved share
            against the end points alone would forfeit a peak sitting
            exactly on a declared bound (issue #239). ``None`` -- what
            ``TICPreservingStrategy`` passes, since it is handed a bare
            axis and has no declared range -- uses the axis's own span, the
            behaviour this function has always had.

    Returns:
        ``resampled``, scaled so that its sum is the preserved TIC.
    """
    if axis_range is None:
        axis_min, axis_max = float(axis[0]), float(axis[-1])
    else:
        axis_min, axis_max = float(axis_range[0]), float(axis_range[1])
    target = preserved_tic(mzs, intensities, axis_min, axis_max)
    if target <= 0.0:
        resampled.fill(0.0)
        return resampled

    current = float(np.sum(resampled))
    if current <= 0.0:
        # Every source peak landed strictly between two axis points. There is
        # nothing to scale and nothing to recover.
        return resampled

    resampled *= target / current
    return resampled
