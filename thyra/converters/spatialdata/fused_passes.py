"""The sibling tables fed from the summed table's own passes (design decision D5).

The converter builds the summed table in two passes over the source:
count, then scatter. The mass-mobility heatmap, the mobility grid and
the demultiplexed MS/MS table are each built the same way, and each used
to take its own passes -- on a whole slide the heatmap's pass alone was
40 to 60 percent of a conversion. When the reader hands its frames over
as records (:mod:`thyra.core.frames`), one raw read per frame per pass
serves everything: the summed spectrum is derived from it, and so are
the mobility point cloud and the precursor spectra the sinks want.

:class:`SiblingPasses` holds those sinks. The converter's pass loops call
:meth:`SiblingPasses.count` and :meth:`SiblingPasses.scatter` once per
frame with the row the frame is getting; everything the sinks see goes
through the same mapping and the same accumulators the standalone passes
use (``map_points_to_axis``, ``GridDiscovery``, ``MsmsAccumulator``), so
the tables come out identical to the ones the standalone passes build.
A record that offers its points indexed by their distinct m/z values is
mapped by ``map_indexed_points_to_axis`` instead, which is that same
mapping evaluated once per distinct value and gathered rather than a
second mapping.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ...core.frames import FrameScans
from .mobility_heatmap import (
    MobilityHeatmap,
    finish_mobility_heatmap,
    map_indexed_points_to_axis,
    map_points_to_axis,
)
from .mobility_table import GridDiscovery, GridScatter, _report_discovery
from .msms_table import MsmsAccumulator

logger = logging.getLogger(__name__)

#: One frame's points on the mass axis: ``(bins, mobility, intensities,
#: n_dropped)``, what both mapping functions return and what every
#: mobility sink's ``add_mapped`` takes.
MappedPoints = Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], int]


class SiblingPasses:
    """The sinks of every sibling table, fed frame by frame by the converter.

    Args:
        axis: The common mass axis every sink maps onto.
        heatmap: The heatmap accumulator, or ``None`` when none is wanted.
        discovery: The grid's pass-1 accumulator, or ``None`` when no grid
            table is planned.
        msms: The MS/MS table's accumulator, or ``None`` when no split is
            planned.
    """

    def __init__(
        self,
        axis: NDArray[np.float64],
        heatmap: Optional[MobilityHeatmap] = None,
        discovery: Optional[GridDiscovery] = None,
        msms: Optional[MsmsAccumulator] = None,
    ) -> None:
        """Hold the sinks; nothing is read until the converter's passes feed them."""
        self.axis = np.asarray(axis, dtype=np.float64)
        self.heatmap = heatmap
        self.heatmap_wanted = heatmap is not None
        self.discovery = discovery
        self.msms = msms
        self.heatmap_block: Optional[Dict[str, Any]] = None
        self.grid_scatter: Optional[GridScatter] = None
        self.grid_scratch: Optional[Path] = None
        self.msms_scratch: Optional[Path] = None
        self._counting_finished = False

    @property
    def wants_mobility(self) -> bool:
        """Whether any sink still needs the frames' point clouds."""
        return self.heatmap is not None or self.discovery is not None

    @property
    def wants_precursors(self) -> bool:
        """Whether the MS/MS accumulator is still in play."""
        return self.msms is not None

    @property
    def empty(self) -> bool:
        """Whether there is nothing left to feed."""
        return not (self.wants_mobility or self.wants_precursors)

    # -- the mapping -----------------------------------------------------

    def _mapped_points(self, frame: FrameScans) -> Optional[MappedPoints]:
        """One frame's points on the mass axis, through the cheapest view it offers.

        A record that hands its points over indexed
        (:meth:`~thyra.core.frames.FrameScans.mobility_points_indexed`,
        which the Bruker TDF record does because it reads digitizer
        indices anyway) is mapped by its distinct m/z values and
        gathered; any other record is mapped point by point. The bins,
        the mask and the drop count are the same either way -- the
        mapping is elementwise in the m/z -- so which view a record
        offers cannot change what a sink sees.

        ``None`` when the frame holds no points.
        """
        indexed = getattr(frame, "mobility_points_indexed", None)
        if indexed is not None:
            points = indexed()
            return (
                None
                if points is None
                else map_indexed_points_to_axis(self.axis, *points)
            )
        flat = frame.mobility_points()
        return None if flat is None else map_points_to_axis(self.axis, *flat)

    # -- pass 1 ----------------------------------------------------------

    def count(self, frame: FrameScans, row: Optional[int]) -> None:
        """Feed one frame to every pass-1 sink.

        ``row`` is the row the frame's pixel gets in the summed table, or
        ``None`` when it gets none (an empty spectrum, a coordinate off
        the grid): the row-bound sinks skip it, exactly as their own pass
        skips a pixel that is not in ``obs``. The heatmap is a mean over
        every pixel with points and takes the frame either way.
        """
        if self.wants_mobility:
            mapped = self._mapped_points(frame)
            if mapped is not None:
                if self.heatmap is not None:
                    self.heatmap.add_mapped(frame.coords, *mapped)
                if self.discovery is not None:
                    self.discovery.add_mapped_row(row, *mapped)
        if self.msms is not None:
            self.msms.count(row, frame.precursor_spectra())

    def finish_counting(
        self, n_rows: int, scratch_for: Callable[[str, Any], Path]
    ) -> None:
        """Close pass 1: the heatmap block, the grid's verdict, the allocations.

        ``n_rows`` is the summed table's row count, known only now, which
        the assemblies size their index dtype from. ``scratch_for(prefix,
        assembly)`` gives a scratch directory per sibling for the memmaps
        pass 2 scatters into, registered with the assembly so the caller
        can release both once the table is written. A sibling that pass 1
        rules out is never scattered.
        """
        if self._counting_finished:
            return
        self._counting_finished = True
        if self.heatmap is not None:
            self.heatmap_block = finish_mobility_heatmap(self.heatmap)
            self.heatmap = None
        # A sibling pass 1 rules out keeps its accumulator, verdict inside,
        # so the builder reaches the same (memoised) verdict and writes no
        # table; only its pass-2 sink is dropped.
        if self.discovery is not None:
            self.discovery.assembly.n_rows = int(n_rows)
            self.discovery.finish()
            if _report_discovery(self.discovery):
                self.grid_scratch = scratch_for("mobility", self.discovery.assembly)
                self.discovery.scratch = self.grid_scratch
                self.discovery.assembly.allocate(self.grid_scratch)
                self.grid_scatter = GridScatter(self.discovery)
        if self.msms is not None:
            self.msms.assembly.n_rows = int(n_rows)
            if self.msms.finish_counting():
                self.msms_scratch = scratch_for("msms", self.msms.assembly)
                self.msms.scratch = self.msms_scratch
                self.msms.allocate(self.msms_scratch)

    def merge_duplicate_rows(self) -> None:
        """Tell every sink a row may be scattered into more than once.

        The summed table's converter knows which grid positions the
        source gave two spectra for; those frames reach these sinks
        twice under the same row as well, so their matrices need the
        same summation the summed table's does (issue #241).
        """
        for sink in (self.discovery, self.msms):
            if sink is not None:
                sink.assembly.merge_duplicates = True

    # -- pass 2 ----------------------------------------------------------

    def scatter(self, frame: FrameScans, row: Optional[int]) -> None:
        """Feed one frame to every pass-2 sink; ``row`` as for :meth:`count`."""
        if row is None:
            return
        if self.grid_scatter is not None:
            mapped = self._mapped_points(frame)
            if mapped is not None:
                self.grid_scatter.add_mapped_row(row, *mapped)
        if self.msms is not None and self.msms.has_rows:
            self.msms.scatter(row, frame.precursor_spectra())

    def finish_scattering(self) -> None:
        """Mark every scattered sibling as built; the builders then skip their passes."""
        if self.discovery is not None and self.grid_scatter is not None:
            self.discovery.scattered = True
        if self.msms is not None and self.msms.has_rows:
            self.msms.scattered = True
        self.grid_scatter = None
