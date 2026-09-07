"""One frame's raw read, from which every table's view of it derives.

A Bruker TDF frame is read as one buffer of ``(digitizer index, scan,
intensity)`` triples, and everything the converter writes about that
pixel is a function of that one buffer: the summed spectrum is the
triples summed per index, the mobility point cloud is the same triples
with the scan turned into 1/K0, and the fragment spectrum of each
precursor is the triples of that precursor's scan range summed per index.
The three iterators of the reader contract each read the buffer again,
so a conversion that wants all three reads the source three times per
pass -- which on a whole slide is most of the conversion (design decision
D5 in ``docs/design-decisions.md``).

A :class:`FrameScans` is that buffer read once, handed to the converter's
pass loop so it can derive the summed spectrum *and* feed the sibling
tables' sinks from it. Every derivation goes through the same helpers
the reader's iterators use, so what a sink sees through a record is what
it would have seen through the iterator.
"""

from typing import List, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

Coords = Tuple[int, int, int]
Spectrum = Tuple[NDArray[np.float64], NDArray[np.float64]]
MobilityPoints = Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
IndexedMobilityPoints = Tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]
PrecursorSpectrum = Tuple[int, NDArray[np.float64], NDArray[np.float64]]


class FrameScans(Protocol):
    """One pixel's raw read, with every view of it derived on demand.

    Each method answers exactly as the corresponding reader iterator
    would have yielded for this frame -- the same arrays, the same
    filtering, the same "nothing" -- so a consumer fed from records is
    fed what its own pass would have read.
    """

    coords: Coords

    def spectrum(self) -> Optional[Spectrum]:
        """``(mzs, intensities)`` as :meth:`BaseMSIReader.iter_spectra` yields, or ``None``.

        ``None`` where the iterator would have skipped the frame (an
        empty spectrum, or one emptied by the intensity threshold).
        """

    def mobility_points(self) -> Optional[MobilityPoints]:
        """``(mzs, mobility, intensities)`` as ``iter_mobility_spectra`` yields, or ``None``."""

    def mobility_points_indexed(self) -> Optional[IndexedMobilityPoints]:
        """The same points with the m/z left factored, or ``None`` for no points.

        ``(unique_mz, inverse, mobility, intensities)`` where
        ``unique_mz[inverse]`` is exactly the ``mzs`` of
        :meth:`mobility_points`, element for element, and the three other
        arrays are that method's own. A digitizer reports the points of
        one frame at far fewer distinct m/z values than it reports
        points, so a consumer that has to map m/z onto an axis can map
        ``unique_mz`` once and gather by ``inverse`` for the same bins.

        ``None`` exactly where :meth:`mobility_points` returns ``None``:
        the frame holds no points. A record whose points do not factor
        this way does not offer the method at all -- consumers reach for
        it with :func:`getattr` and fall back to the flat view -- so
        ``None`` never means "ask the other way".
        """

    def precursor_spectra(self) -> List[PrecursorSpectrum]:
        """``[(window_index, mzs, intensities), ...]`` as ``iter_precursor_spectra`` yields.

        Empty when the acquisition has no separable precursor schedule
        or the frame holds no isolated points.
        """
