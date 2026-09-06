"""The common ion mobility grid: channels every pixel's points land on.

A mobility axis is a **continuum**. A Bruker TDF pixel is its own point
cloud -- one ``(m/z, 1/K0, intensity)`` triple per digitizer index per
scan -- and no two pixels are guaranteed to hit the same 1/K0 values, so
there is no feature list to write until the continuum is binned. This
module is that binning: a fixed set of channels over one range, shared by
every pixel of a conversion, so ``(m/z bin, channel)`` becomes a feature
axis.

This is the one place in the mobility work where binning a continuum is
correct. The demultiplexed MS/MS table bins nothing, because a precursor
axis is discrete -- the acquisition schedule names its values.

Two things are fixed rather than tuned:

- **The channel count is an alignment anchor, not a rendering choice.**
  :data:`MOBILITY_CHANNELS` is the same 256 the mass-mobility heatmap
  uses, over the same edges, so a box drawn on the heatmap maps onto grid
  channels by integer index -- no resampling, no edge off-by-one. The
  heatmap's ``HEATMAP_MOBILITY_CHANNELS`` *is* this constant, and
  :func:`linear_channel` is the assignment both go through.
- **Edges come from the axis values, never from the declared acquisition
  range.** The per-scan 1/K0 of a real file overhangs its declared
  ``OneOverK0AcqRange`` by a few scans (1.00003..1.29133 against a
  declared 1.0..1.29 on one measured file); binning over the declared
  range would pile those scans into an edge channel and the 1:1 mapping
  onto the heatmap would be a lie. The caller passes the values' range.

The generator/law split mirrors ``thyra.resampling.mass_axis``: a law
distributes channel edges across a range, and a second law
(``constant_relative``, say) is a drop-in that supplies its own edges and
inherits :meth:`MobilityGrid.assign`.

Mobility is a feature coordinate. Nothing here knows where a pixel is.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Number of mobility channels, exactly, and the default for a grid table.
#: An anchor shared with the mass-mobility heatmap
#: (``HEATMAP_MOBILITY_CHANNELS`` is this constant): change one and the
#: heatmap no longer indexes the grid table.
MOBILITY_CHANNELS = 256

#: The channel widths TIMS resolving power supports, in 1/K0. Imaging
#: separations are 1% to 5% in 1/K0 and the measured FWHM of a strong ion
#: is 0.01 to 0.02, so a channel wider than :data:`MAX_CHANNEL_WIDTH`
#: merges separations the instrument resolved, and one narrower than
#: :data:`MIN_CHANNEL_WIDTH` splits a single peak across channels and pays
#: for it in table size. The width a grid realizes is derived from the
#: channel count, recorded in ``uns["mobility_grid"]`` and reported
#: against this band -- the anchor above is never moved to satisfy it.
MIN_CHANNEL_WIDTH = 0.002
MAX_CHANNEL_WIDTH = 0.02

#: The law a grid follows when the caller names none.
DEFAULT_MOBILITY_GRID_LAW = "linear"


def linear_channel(
    values: Any, lower: float, upper: float, n_channels: int
) -> NDArray[np.int64]:
    """Channel of each value on ``n_channels`` equal-width bins over the range.

    Position in the range times the channel count, floored. A value on the
    upper edge belongs to the last channel, and a value beyond either edge
    -- an axis that overhangs its declared range, a point the reader
    clipped -- is clipped into the edge channel rather than lost, so a
    marginal over channels keeps every count.

    The mass-mobility heatmap assigns its channels through this same
    function, which is what makes the two agree by construction rather
    than by coincidence.
    """
    channel = np.floor((np.asarray(values) - lower) / (upper - lower) * n_channels)
    return np.clip(channel, 0, n_channels - 1).astype(np.int64)


@dataclass(frozen=True)
class MobilityGrid:
    """The channels one conversion bins every pixel's mobility onto.

    ``edges`` has ``n_channels + 1`` entries, ascending; ``centres`` are
    the values written to ``var["mobility"]``. ``law`` names how the edges
    were spaced, and is what a second law would change.

    Not to be confused with :class:`thyra.metadata.schema.models.MobilityGrid`,
    which is the pydantic description of this object inside the versioned
    metadata block; :meth:`to_schema_report` produces its input.
    """

    law: str
    lower: float
    upper: float
    n_channels: int
    edges: NDArray[np.float64]

    @property
    def centres(self) -> NDArray[np.float64]:
        """Channel centres, ascending: what ``var["mobility"]`` carries."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def channel_width(self) -> float:
        """The width one channel spans, in the axis unit.

        Derived from the channel count, not the other way round -- see
        :data:`MIN_CHANNEL_WIDTH` for the band it is reported against.
        """
        return float(self.upper - self.lower) / float(self.n_channels)

    def assign(self, values: Any) -> NDArray[np.int32]:
        """The channel of each mobility value, as ``int32``.

        A linear law has a closed form, which is the expression the
        heatmap uses; any other law is resolved against ``edges`` by
        binary search, so a new law needs no new assignment code.
        """
        if self.law == "linear":
            channel = linear_channel(values, self.lower, self.upper, self.n_channels)
        else:
            channel = np.clip(
                np.searchsorted(self.edges, np.asarray(values), side="right") - 1,
                0,
                self.n_channels - 1,
            )
        return channel.astype(np.int32)

    def to_uns(self) -> Dict[str, Any]:
        """The ``uns["mobility_grid"]`` block: plain-name keys, no colons.

        Its presence is what says a mobility-resolved table was *binned*
        onto a common grid rather than read off a shared feature axis.
        ``edges`` travels with it so a consumer can map a heatmap box onto
        channels without recomputing the range.
        """
        return {
            "law": self.law,
            "lower": float(self.lower),
            "upper": float(self.upper),
            "n_channels": int(self.n_channels),
            "channel_width": self.channel_width,
            "edges": np.asarray(self.edges, dtype=np.float64),
        }

    def to_schema_report(self) -> Dict[str, Any]:
        """The shape ``thyra.metadata.schema.builder`` reads for ``ion_mobility.grid``."""
        return {
            "law": self.law,
            "lower": float(self.lower),
            "upper": float(self.upper),
            "n_channels": int(self.n_channels),
        }


class BaseMobilityGridGenerator(ABC):
    """Distributes channel edges across a mobility range by one law.

    The mobility counterpart of
    :class:`~thyra.resampling.mass_axis.base_generator.BaseAxisGenerator`:
    a generator decides *where* the edges fall, never *how many* there are
    -- that is the anchor in :data:`MOBILITY_CHANNELS` or an explicit
    override.
    """

    @abstractmethod
    def generate(self, lower: float, upper: float, n_channels: int) -> MobilityGrid:
        """``n_channels`` channels spanning ``[lower, upper]``, ascending."""

    @abstractmethod
    def get_law(self) -> str:
        """The name this law is recorded under."""


class LinearMobilityGridGenerator(BaseMobilityGridGenerator):
    """Equal-width channels in the axis unit.

    Right for 1/K0 over an imaging acquisition: the ranges are short
    (0.3 1/K0 is typical) and TIMS resolving power is near enough constant
    across one, so equal width and equal relative width differ by less
    than a channel.
    """

    def generate(self, lower: float, upper: float, n_channels: int) -> MobilityGrid:
        """Channels from ``np.linspace``, which is also the heatmap's rule."""
        return MobilityGrid(
            law=self.get_law(),
            lower=float(lower),
            upper=float(upper),
            n_channels=int(n_channels),
            edges=np.linspace(float(lower), float(upper), int(n_channels) + 1),
        )

    def get_law(self) -> str:
        """The name this law is recorded under."""
        return "linear"


#: The laws a grid can be built under, by name.
MOBILITY_GRID_GENERATORS: Dict[str, BaseMobilityGridGenerator] = {
    "linear": LinearMobilityGridGenerator(),
}


def build_mobility_grid(
    lower: float,
    upper: float,
    n_channels: int = MOBILITY_CHANNELS,
    law: str = DEFAULT_MOBILITY_GRID_LAW,
) -> MobilityGrid:
    """The common mobility grid of one conversion.

    Args:
        lower: Lower edge of the first channel, in the axis unit. Take it
            from the axis *values*, never the declared acquisition range.
        upper: Upper edge of the last channel; must exceed ``lower``.
        n_channels: Channels to divide the range into. The default is the
            heatmap anchor, which is what makes a heatmap box index the
            grid.
        law: How the edges are spaced; ``"linear"`` is the only law today.

    Raises:
        ValueError: If the range has no extent, the channel count is not
            positive, or the law is unknown.
    """
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(
            f"The mobility range [{lower}, {upper}] has no extent to bin over"
        )
    if int(n_channels) < 1:
        raise ValueError(
            f"A mobility grid needs at least one channel, got {n_channels}"
        )
    generator = MOBILITY_GRID_GENERATORS.get(law)
    if generator is None:
        raise ValueError(
            f"Unknown mobility grid law {law!r}; known laws are "
            f"{sorted(MOBILITY_GRID_GENERATORS)}"
        )
    return generator.generate(lower, upper, int(n_channels))


def report_channel_width(grid: MobilityGrid, unit: Optional[str] = None) -> None:
    """Say once whether the realized channel width matches the instrument.

    The channel count is an anchor and is never moved to land inside the
    band; what the band decides is whether the grid is worth its size
    (too fine) or is losing separations (too coarse), and that is a thing
    to say out loud rather than leave in a number nobody reads.
    """
    width = grid.channel_width
    where = f" {unit}" if unit else ""
    if width > MAX_CHANNEL_WIDTH:
        logger.warning(
            "Mobility grid channels are %.4g%s wide, above the %.3g the "
            "instrument resolves: separations TIMS pulled apart may fall in "
            "one channel. Ask for more channels (--mobility-bins) or a "
            "narrower range (--mobility-min/--mobility-max).",
            width,
            where,
            MAX_CHANNEL_WIDTH,
        )
    elif width < MIN_CHANNEL_WIDTH:
        logger.info(
            "Mobility grid channels are %.4g%s wide, finer than the %.3g the "
            "instrument resolves: a single mobility peak spans several "
            "channels and the table is larger than the separation needs. "
            "The count is held at %d so the grid indexes the heatmap.",
            width,
            where,
            MIN_CHANNEL_WIDTH,
            grid.n_channels,
        )
