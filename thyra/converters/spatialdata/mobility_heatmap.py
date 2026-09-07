"""The mass-mobility heatmap: the dataset's mean frame, binned.

A source with an ion mobility dimension is summed over it in the MSI
table. Before anyone pays for a mobility-resolved table they need to see
whether mobility separates anything, and that is what this block is for:
the global mean (m/z, mobility) frame, accumulated from the raw scan
read of every pixel during conversion and stored on the summed table as
``uns["mobility_heatmap"]``::

    mz_edges        float64[m + 1]   bin edges on the common m/z axis
    mobility_edges  float64[k + 1]   bin edges in the axis unit (1/K0)
    counts          float32[m, k]    mean intensity per bin over pixels

``m`` is the common mass axis coarsened to about
:data:`HEATMAP_MZ_BINS` bins; ``k`` is exactly
:data:`HEATMAP_MOBILITY_CHANNELS`. The m/z binning is the converter's
own nearest-bin mapping onto the common axis, coarsened, so that under a
lossless summed spectrum (``--tdf-spectrum scan_sum``) the heatmap's
marginal over mobility is the stored mean spectrum coarsened to the same
bins. Under the vendor centroid it is not, and cannot be: the centroid
keeps 80-90% of the ion current and merges bins, and the heatmap is built
from the raw points, not from the centroid.

The raw pass itself is :func:`scan_mobility`, which maps every pixel's
points onto the mass axis once and hands the result to whichever sinks
were given -- the heatmap, the mobility grid's discovery pass, or both.
Mapping is the expensive part of the pass (the vendor read is a fifth of
it), so a grid conversion that fuses its discovery into the heatmap's
pass pays for it once rather than twice.

Mobility is a feature coordinate. Nothing here knows where a pixel is.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ...core.base_reader import BaseMSIReader
from ...resampling.mobility_grid import (
    MOBILITY_CHANNELS,
    build_mobility_grid,
    linear_channel,
)
from .base_spatialdata_converter import _nn_map_to_bins

logger = logging.getLogger(__name__)

#: Target number of m/z bins. The common axis is coarsened by an integer
#: factor to land near this, so the exact ``m`` depends on the axis.
HEATMAP_MZ_BINS = 4000

#: Number of mobility channels, exactly. This is an alignment anchor, not
#: a rendering choice: the opt-in mobility grid table defaults to the same
#: 256 channels over the same edges, so a box drawn on the heatmap maps
#: onto grid channels by integer index in both directions, with no
#: resampling and no edge off-by-one. It is literally the grid's own
#: constant, and :func:`mobility_bin_edges` is the grid's own generator,
#: so the two cannot drift apart.
HEATMAP_MOBILITY_CHANNELS = MOBILITY_CHANNELS

#: Points buffered before they are folded into the accumulator. One
#: ``bincount`` over a few million entries is far cheaper than an
#: ``add.at`` per frame; the buffer is bounded so memory stays flat.
_FLUSH_POINTS = 4_000_000

Coords = Tuple[int, int, int]


def mz_bin_edges(
    axis: NDArray[np.float64], target_bins: int = HEATMAP_MZ_BINS
) -> Tuple[NDArray[np.float64], int]:
    """Coarsen a mass axis to about ``target_bins`` bins.

    Consecutive axis entries are grouped ``step`` at a time, with
    ``step = ceil(n / target_bins)``, so an axis index ``i`` lands in bin
    ``i // step``. The edges are the axis' own bin edges (midpoints
    between neighbours, extended by half a bin at either end) taken every
    ``step``; the last edge is the axis' upper edge whatever the group
    size, so no entry is cut off.

    Returns:
        ``(edges, step)`` with ``edges`` of length ``m + 1``.
    """
    axis = np.asarray(axis, dtype=np.float64)
    n = int(axis.size)
    if n == 0:
        raise ValueError("The mass axis is empty")
    if n == 1:
        axis_edges = np.array([axis[0] - 0.5, axis[0] + 0.5])
    else:
        axis_edges = np.empty(n + 1, dtype=np.float64)
        axis_edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
        axis_edges[0] = axis[0] - (axis_edges[1] - axis[0])
        axis_edges[-1] = axis[-1] + (axis[-1] - axis_edges[-2])
    step = max(1, -(-n // max(1, int(target_bins))))
    edges = np.concatenate([axis_edges[:-1][::step], axis_edges[-1:]])
    return edges, step


def mobility_bin_edges(
    lower: float, upper: float, channels: int = HEATMAP_MOBILITY_CHANNELS
) -> NDArray[np.float64]:
    """``channels`` equal-width bins over ``[lower, upper]``, ascending.

    The grid table's own linear generator, so the heatmap's edges and a
    default grid's edges are the same array rather than two arrays that
    happen to agree.
    """
    return build_mobility_grid(lower, upper, channels).edges


def map_points_to_axis(
    axis: NDArray[np.float64],
    mzs: NDArray[np.float64],
    mobility: NDArray[np.float64],
    intensities: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], int]:
    """One pixel's points onto the common mass axis, the summed table's way.

    "In range" is the strict axis span, and a point outside it is dropped
    rather than folded onto an edge bin -- the summed table drops it, so
    a marginal over mobility that kept it would exceed the column it
    mirrors. Points in range go to their nearest axis entry through the
    converter's own rule (ties to the right). This is the one place the
    mapping is written: the heatmap, the grid's discovery pass and the
    grid's scatter pass all go through it, so they cannot disagree.

    Returns:
        ``(bins, mobility, intensities, n_dropped)`` -- the axis index of
        every kept point, the two other arrays masked to match, and how
        many points were dropped.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    mobility = np.asarray(mobility, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if mzs.size == 0:
        return np.zeros(0, dtype=np.int64), mobility, intensities, 0
    in_range = (mzs >= axis[0]) & (mzs <= axis[-1])
    n_dropped = 0
    if not in_range.all():
        n_dropped = int(mzs.size - in_range.sum())
        mzs = mzs[in_range]
        mobility = mobility[in_range]
        intensities = intensities[in_range]
        if mzs.size == 0:
            return np.zeros(0, dtype=np.int64), mobility, intensities, n_dropped
    return _nn_map_to_bins(axis, mzs).astype(np.int64), mobility, intensities, n_dropped


class MobilityHeatmap:
    """Accumulator for the mean (m/z, mobility) frame.

    ``add`` takes one pixel's raw point cloud; ``finalize`` divides by
    the number of pixels added and returns the ``uns`` block.
    """

    def __init__(
        self,
        mass_axis: NDArray[np.float64],
        mobility_range: Tuple[float, float],
        mz_bins: int = HEATMAP_MZ_BINS,
        channels: int = HEATMAP_MOBILITY_CHANNELS,
    ) -> None:
        """Size the bins from the mass axis and the mobility range.

        Args:
            mass_axis: The common m/z axis the summed table uses, ascending.
            mobility_range: ``(lower, upper)`` in the axis unit, spanning
                the mobility values the points can take.
            mz_bins: Target number of m/z bins (see :func:`mz_bin_edges`).
            channels: Number of mobility channels.
        """
        self._axis = np.ascontiguousarray(mass_axis, dtype=np.float64)
        self.mz_edges, self._step = mz_bin_edges(self._axis, mz_bins)
        self.mobility_edges = mobility_bin_edges(
            mobility_range[0], mobility_range[1], channels
        )
        self.n_mz = int(self.mz_edges.size - 1)
        self.n_mobility = int(self.mobility_edges.size - 1)
        self._mobility_lower = float(self.mobility_edges[0])
        self._mobility_span = float(self.mobility_edges[-1] - self.mobility_edges[0])
        self._total = np.zeros(self.n_mz * self.n_mobility, dtype=np.float64)
        self._buffer_cells: List[NDArray[np.int64]] = []
        self._buffer_weights: List[NDArray[np.float64]] = []
        self._buffered = 0
        self.n_pixels = 0
        self.n_points = 0
        self.n_out_of_range = 0

    def add(
        self,
        mzs: NDArray[np.float64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """Fold one pixel's ``(m/z, mobility, intensity)`` points in.

        m/z goes to its nearest common-axis entry (the converter's own
        rule, ties to the right) and then to the coarse bin; points
        outside the axis span are dropped, as the summed spectrum drops
        them. Mobility goes to its channel by position in the range;
        values at the upper edge belong to the last channel, and the
        rare point beyond either edge (an axis that overhangs its declared
        range) is clipped into the edge channel rather than lost, so the
        marginal keeps every count.
        """
        bins, mobility, intensities, n_dropped = map_points_to_axis(
            self._axis, mzs, mobility, intensities
        )
        self.add_mapped(None, bins, mobility, intensities, n_dropped)

    def add_mapped(
        self,
        coords: Optional[Coords],
        bins: NDArray[np.int64],
        mobility: NDArray[np.float64],
        intensities: NDArray[np.float64],
        n_dropped: int,
    ) -> None:
        """Fold in one pixel already mapped by :func:`map_points_to_axis`.

        The sink interface :func:`scan_mobility` feeds; the heatmap does
        not care where a pixel is, so ``coords`` is ignored.
        """
        self.n_pixels += 1
        self.n_out_of_range += int(n_dropped)
        if bins.size == 0:
            return
        mz_bin = bins // self._step
        channel = linear_channel(
            mobility,
            self._mobility_lower,
            self._mobility_lower + self._mobility_span,
            self.n_mobility,
        )
        self._buffer_cells.append(mz_bin * self.n_mobility + channel)
        self._buffer_weights.append(np.asarray(intensities, dtype=np.float64))
        self._buffered += int(bins.size)
        self.n_points += int(bins.size)
        if self._buffered >= _FLUSH_POINTS:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer_cells:
            return
        cells = np.concatenate(self._buffer_cells)
        weights = np.concatenate(self._buffer_weights)
        self._total += np.bincount(cells, weights=weights, minlength=self._total.size)
        self._buffer_cells = []
        self._buffer_weights = []
        self._buffered = 0

    def finalize(self) -> Dict[str, Any]:
        """The ``uns["mobility_heatmap"]`` block: mean over the pixels added."""
        self._flush()
        counts = self._total / max(self.n_pixels, 1)
        return {
            "mz_edges": self.mz_edges.astype(np.float64),
            "mobility_edges": self.mobility_edges.astype(np.float64),
            "counts": counts.reshape(self.n_mz, self.n_mobility).astype(np.float32),
        }


def _mobility_range(reader: BaseMSIReader) -> Optional[Tuple[float, float]]:
    """The range to bin mobility over, from the reader's axis.

    The axis values decide when there are any: on a Bruker file the
    per-scan 1/K0 overhangs the declared acquisition range by a few
    scans, and binning over the declared range would pile those scans
    into an edge channel. The declared range is the fallback for an axis
    that has no values.
    """
    axis = reader.get_mobility_axis()
    if axis is None:
        return None
    if axis.values is not None and axis.values.size:
        finite = axis.values[np.isfinite(axis.values)]
        if finite.size:
            lower, upper = float(finite.min()), float(finite.max())
            if upper > lower:
                return lower, upper
    if axis.acq_range is not None:
        lower, upper = float(axis.acq_range[0]), float(axis.acq_range[1])
        if upper > lower:
            return lower, upper
    return None


def prepare_mobility_heatmap(
    reader: BaseMSIReader, mass_axis: NDArray[np.float64]
) -> Optional[MobilityHeatmap]:
    """An empty heatmap sized for ``reader``, or ``None`` with the reason logged.

    ``None`` when the reader has no mobility dimension or its axis gives
    no range to bin over (a per-pixel mobility source without a shared
    axis), or when the common mass axis is empty.
    """
    if not getattr(reader, "has_ion_mobility", False):
        return None
    mobility_range = _mobility_range(reader)
    if mobility_range is None:
        logger.info(
            "No mass-mobility heatmap: the source's mobility axis has no "
            "range to bin over (per-pixel mobility values without a shared axis)"
        )
        return None
    if np.asarray(mass_axis).size == 0:
        logger.warning("No mass-mobility heatmap: the common mass axis is empty")
        return None
    return MobilityHeatmap(np.asarray(mass_axis, dtype=np.float64), mobility_range)


def finish_mobility_heatmap(heatmap: MobilityHeatmap) -> Optional[Dict[str, Any]]:
    """The ``uns`` block of a heatmap the scan has been run over, or ``None``."""
    if heatmap.n_pixels == 0:
        logger.warning(
            "No mass-mobility heatmap: the source yielded no mobility spectra"
        )
        return None
    if heatmap.n_out_of_range:
        logger.info(
            "Mass-mobility heatmap dropped %d of %d points outside the mass axis",
            heatmap.n_out_of_range,
            heatmap.n_points + heatmap.n_out_of_range,
        )
    logger.info(
        "Mass-mobility heatmap: %d x %d bins over %d pixels (%d points)",
        heatmap.n_mz,
        heatmap.n_mobility,
        heatmap.n_pixels,
        heatmap.n_points,
    )
    return heatmap.finalize()


def scan_mobility(
    reader: BaseMSIReader,
    mass_axis: NDArray[np.float64],
    *sinks: Any,
    n_spectra: Optional[int] = None,
    description: str = "Mobility scan",
) -> None:
    """One raw pass over :meth:`BaseMSIReader.iter_mobility_spectra`.

    Every pixel's points are mapped onto the mass axis once, by
    :func:`map_points_to_axis`, and handed to each sink's ``add_mapped``.
    On a Bruker source this is the raw scan read, needed even when the
    summed spectrum is the vendor centroid (one extra library call per
    frame, about a millisecond); the mapping costs more than the read,
    which is why the sinks share a pass rather than each taking one.
    """
    from tqdm import tqdm

    axis = np.asarray(mass_axis, dtype=np.float64)
    with tqdm(total=n_spectra, desc=description, unit="spectrum") as pbar:
        for coords, mzs, mobility, intensities in reader.iter_mobility_spectra():
            mapped = map_points_to_axis(axis, mzs, mobility, intensities)
            for sink in sinks:
                sink.add_mapped(coords, *mapped)
            pbar.update(1)


def build_mobility_heatmap(
    reader: BaseMSIReader,
    mass_axis: NDArray[np.float64],
    n_spectra: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Accumulate the heatmap over every pixel of ``reader``.

    One pass over :meth:`BaseMSIReader.iter_mobility_spectra`. Returns
    ``None``, with the reason logged, when the reader has no mobility
    dimension or its axis gives no range to bin over (a per-pixel mobility
    source without a shared axis).
    """
    heatmap = prepare_mobility_heatmap(reader, mass_axis)
    if heatmap is None:
        return None
    scan_mobility(
        reader,
        mass_axis,
        heatmap,
        n_spectra=n_spectra,
        description="Mobility heatmap",
    )
    return finish_mobility_heatmap(heatmap)
