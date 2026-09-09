"""Which coordinate value an imzML file means by "the first pixel".

The imzML specification says x and y count from 1, and until issue #244 the
reader took it at its word and subtracted a constant 1. Files written
0-based exist, and on one of those the constant produced ``x = -1`` and
``y = -1`` for the first row and column. A negative index is a legal
*negative* numpy index, so the converter's grid guard dropped those
spectra: a 3x3 file at coordinates 0..2 stored 4 rows and reported
"5 spectra sat outside the declared 2x2x1 grid" -- blaming a grid the file
never declared, and exiting 0.

Why not rebase on the observed minimum, the way z does
-----------------------------------------------------

The z rule -- subtract the smallest z the file holds -- rests on an
argument that is good for z and does not carry over: imzML guarantees
nothing about the base and pyimzml is inconsistent about it. It stops at
z because **z has no physical origin and x and y do**. An
acquisition cropped to a region of the slide legitimately starts at
``x = 5``; rebasing on the observed minimum would slide it to ``x = 0``,
changing the grid width, every ``obs["spatial_x"]``, the TIC image extent
and the pixel footprint of a file that converts correctly today. Nothing
in the file distinguishes that acquisition from a 0-based export whose
first column happens to be empty.

So the rule folds down a 0 and nothing else: ``base = min(observed, 1)``.
A file starting at 0 is 0-based and keeps all its pixels; a file starting
at 1, or at 5, is 1-based and is offset by exactly 1, which is what
happened before. The only files whose stored coordinates move are the ones
that were losing a row and a column.

The declared ``IMS:1000042`` / ``IMS:1000043`` pixel counts are the third
option and are deliberately not used: nothing in Thyra reads them today
except ``mzpeak_extractor``, which carries a comment about a declared
extent disagreeing with the coordinates it ships with. Trusting a declared
extent over the coordinates would be a larger change with a wider blast
radius than the defect it fixes.

Whatever is subtracted is recorded: the imzML extractor reports it as
``EssentialMetadata.coordinate_offsets``, which the converter writes to
``coordinate_systems.global.coordinate_offsets_px``.
"""

import logging
from typing import Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The base the imzML specification declares for x and y. A file whose
#: smallest coordinate is larger than this is a cropped acquisition, not a
#: different convention, so it keeps this base.
SPEC_BASE = 1


def coordinate_bases(
    coordinates: Iterable[Sequence[int]],
) -> Tuple[int, int, int]:
    """The ``(x, y, z)`` values in this file that map onto index 0.

    One pass over the coordinate list, which is what makes it worth
    memoising at the call site: ~60 ms at 900k spectra, against a 63 s
    parse.

    Args:
        coordinates: The parser's coordinate list -- ``(x, y, z)`` triples.

    Returns:
        ``(x_base, y_base, z_base)``. x and y fold a 0 down and otherwise
        keep the specification's base of 1; z is the smallest value present,
        because it has no origin to preserve. ``(1, 1, 0)`` for an empty
        list, so an empty file still normalises the way the spec says.
    """
    min_x: Optional[int] = None
    min_y: Optional[int] = None
    min_z: Optional[int] = None
    for coord in coordinates:
        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
        if min_x is None or x < min_x:
            min_x = x
        if min_y is None or y < min_y:
            min_y = y
        if min_z is None or z < min_z:
            min_z = z

    if min_x is None or min_y is None or min_z is None:
        return SPEC_BASE, SPEC_BASE, 0

    x_base = min(min_x, SPEC_BASE)
    y_base = min(min_y, SPEC_BASE)

    if x_base < SPEC_BASE or y_base < SPEC_BASE:
        logger.info(
            "This imzML numbers its pixels from 0, not from 1 as the "
            "specification says (smallest coordinate x=%d, y=%d). Rebasing "
            "on %d so the first row and column are kept; a file numbered "
            "from 1 is unaffected.",
            min_x,
            min_y,
            min(x_base, y_base),
        )

    return x_base, y_base, int(min_z)
