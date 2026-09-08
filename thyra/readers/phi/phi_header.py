# thyra/readers/phi/phi_header.py
"""ASCII header parsing for PHI SmartSoft-TOF ``.raw`` files.

The file opens with a CRLF-delimited key/value header bracketed by ``SOFH``
and ``EOFH``, zero-padded out to the byte count the header itself declares in
``HeaderSize``. Top-level entries use ``Key: value``; entries below a
``[Section]`` marker use ``Key=value``.

A recalibration performed after acquisition is not written back into this
header. It is appended near the end of the file as an ``AsciiInfoBlock01``
(block id 14), and when present its ``Mass/Time`` and ``MassOffset`` supersede
the values here -- the originals may be stale, which the acquisition header
usually admits via ``Calibrated: no``.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...errors import ConversionRefused

logger = logging.getLogger(__name__)

SOFH_MAGIC = b"SOFH"
EOFH_MARKER = b"EOFH"

# Generous upper bound for the ASCII region; HeaderSize is read from within it.
_MAX_HEADER_PROBE = 1 << 20

_SECTION_RE = re.compile(r"^\[(.+)\]$")
_TOP_LEVEL_RE = re.compile(r"^([A-Za-z][^:]*): ?(.*)$")

#: ``(measured, species, theoretical)`` triplets from a ``Calibration:`` line.
_CALIBRANT_RE = re.compile(r"\(\s*([\d.eE+-]+)\s*,\s*([^,]+?)\s*,\s*([\d.eE+-]+)\s*\)")


@dataclass(frozen=True)
class Calibrant:
    """One entry of a ``Calibration:`` line."""

    measured_mz: float
    species: str
    theoretical_mz: float


@dataclass(frozen=True)
class PhiHeader:
    """Parsed PHI acquisition header.

    Attributes:
        entries: Top-level ``Key: value`` pairs, verbatim.
        sections: ``[Section]`` blocks mapping key to value, verbatim.
        header_size: Byte offset at which the binary block stream begins.
        image_pixels: Raster edge length in pixels (the raster is square).
        mass_slope: ``Mass/Time``, in sqrt(amu) per microsecond.
        mass_offset: ``MassOffset``, in sqrt(amu).
        start_flight_time_us: First flight time retained by the acquisition.
        stop_flight_time_us: Last flight time retained by the acquisition.
        bin_size_ns: Detector time-channel width, from ``SpecBinSize``.
        start_mz: Declared low end of the acquired mass range.
        stop_mz: Declared high end of the acquired mass range.
        polarity: ``"negative"`` or ``"positive"``.
        n_frames: Number of acquisition frames summed into the file.
        msms_active: Whether MS/MS was enabled; selects the data block id.
        raster_size_um: Raster edge length in micrometres, when declared.
        raster_size_calibration: ``Raster Size Calibration`` factor, when
            declared. See :meth:`pixel_size_um` for why this is not applied.
        declared_tiles: ``(x, y)`` tile counts declared for mosaic
            acquisitions. Declared even when mosaic mode is off, so it must
            not be used to size the pixel grid on its own.
        mosaic_active: Whether the header declares mosaic mode active.
        calibrants: Calibrants from the acquisition ``Calibration:`` line.
    """

    entries: Dict[str, str]
    sections: Dict[str, Dict[str, str]]
    header_size: int
    image_pixels: int
    mass_slope: float
    mass_offset: float
    start_flight_time_us: float
    stop_flight_time_us: float
    bin_size_ns: float
    start_mz: float
    stop_mz: float
    polarity: str
    n_frames: int
    msms_active: bool
    raster_size_um: Optional[float] = None
    raster_size_calibration: Optional[float] = None
    declared_tiles: Tuple[int, int] = (1, 1)
    mosaic_active: bool = False
    calibrants: List[Calibrant] = field(default_factory=list)

    @property
    def data_block_id(self) -> int:
        """Block id carrying ion events: 8 under MS/MS, otherwise 1."""
        return 8 if self.msms_active else 1

    @property
    def pixel_size_um(self) -> Optional[float]:
        """Edge length of one pixel in micrometres, or ``None`` if underivable.

        Computed as ``raster_size_um / image_pixels``, which on the reference
        dataset gives the 1.00 um pitch the operator selected at acquisition.

        The header also carries a ``Raster Size Calibration`` factor (1.550 on
        that dataset). It is deliberately *not* applied: it trims the scan
        generator so the raster actually achieves the requested pitch, rather
        than scaling the field of view. Multiplying by it would inflate every
        coordinate by 55%. Pass ``pixel_size_um`` to the reader to override.
        """
        if not self.raster_size_um or self.image_pixels <= 0:
            return None
        return self.raster_size_um / self.image_pixels


def _to_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_int(value: Optional[str], default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def parse_calibrants(line: str) -> List[Calibrant]:
    """Parse the ``(measured, species, theoretical)`` triplets of a line."""
    return [
        Calibrant(float(m), species.strip(), float(t))
        for m, species, t in _CALIBRANT_RE.findall(line or "")
    ]


def _split_header_text(text: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    entries: Dict[str, str] = {}
    sections: Dict[str, Dict[str, str]] = {}
    # Empty means "above the first [Section]"; the section regex requires at
    # least one character, so no real section name can collide with it.
    current = ""
    for line in text.split("\r\n"):
        stripped = line.strip()
        if not stripped:
            continue
        section = _SECTION_RE.match(stripped)
        if section is not None:
            current = section.group(1)
            sections.setdefault(current, {})
            continue
        if not current:
            match = _TOP_LEVEL_RE.match(line)
            if match is not None:
                entries[match.group(1)] = match.group(2).strip()
        elif "=" in line:
            key, _, value = line.partition("=")
            sections[current][key.strip()] = value.strip()
    return entries, sections


def _read_header_text(path: Path) -> str:
    """Read and delimit the ASCII header region of a PHI file."""
    with path.open("rb") as handle:
        probe = handle.read(_MAX_HEADER_PROBE)

    if not probe.startswith(SOFH_MAGIC):
        raise ConversionRefused(
            f"Not a PHI SmartSoft-TOF raw file (missing SOFH magic): {path}"
        )
    end = probe.find(EOFH_MARKER)
    if end == -1:
        raise ConversionRefused(f"Malformed PHI header, no EOFH terminator: {path}")
    return probe[:end].decode("latin-1")


def _parse_polarity(
    entries: Dict[str, str], sections: Dict[str, Dict[str, str]]
) -> str:
    """Read the secondary ion polarity, from either place it is recorded."""
    text = sections.get("Polarity", {}).get(
        "Polarity", entries.get("SecIonPolarity", "")
    )
    return "negative" if "-" in text or "Negative" in text else "positive"


def _parse_raster(
    entries: Dict[str, str], lmig: Dict[str, str]
) -> Tuple[Optional[float], Optional[float]]:
    """Read the raster edge length and its scan-generator calibration factor."""
    raster = _to_float(lmig.get("Raster Size (um)") or entries.get("ScanWidthX"))
    calibration = lmig.get("Raster Size Calibration", "")
    factor = _to_float(calibration.split(",")[0]) if calibration else 0.0
    return (raster or None, factor or None)


def _validate_header(header: PhiHeader, path: Path) -> None:
    """Reject headers whose numbers cannot describe a real acquisition."""
    if header.image_pixels <= 0:
        raise ConversionRefused(
            f"PHI header declares a non-positive ImagePixels: {path}"
        )
    if header.mass_slope <= 0:
        raise ConversionRefused(f"PHI header declares a non-positive Mass/Time: {path}")
    if header.stop_flight_time_us <= header.start_flight_time_us:
        raise ConversionRefused(
            f"PHI header flight-time range is empty: {header.start_flight_time_us} "
            f"to {header.stop_flight_time_us} us in {path}"
        )


def parse_phi_header(path: Path) -> PhiHeader:
    """Parse the ASCII header of a PHI ``.raw`` file.

    Args:
        path: Path to the ``.raw`` file.

    Returns:
        The parsed header.

    Raises:
        ValueError: If the file is not a PHI raw file, the header is
            unterminated, or a field required for conversion is missing.
    """
    path = Path(path)
    entries, sections = _split_header_text(_read_header_text(path))

    acq = sections.get("Acq Base", {})
    lmig = sections.get("PHI LMIG", {})
    mosaic = sections.get("Mosaic Area", {})

    missing = [
        name
        for name in ("HeaderSize", "ImagePixels", "Mass/Time", "MassOffset")
        if name not in entries
    ]
    if missing:
        raise ConversionRefused(
            f"PHI header is missing required field '{missing[0]}': {path}"
        )

    raster_um, raster_calibration = _parse_raster(entries, lmig)

    header = PhiHeader(
        entries=entries,
        sections=sections,
        header_size=_to_int(entries["HeaderSize"]),
        image_pixels=_to_int(entries["ImagePixels"]),
        mass_slope=_to_float(entries["Mass/Time"]),
        mass_offset=_to_float(entries["MassOffset"]),
        start_flight_time_us=_to_float(entries.get("StartFlightTime"), 0.0),
        stop_flight_time_us=_to_float(entries.get("StopFlightTime"), 0.0),
        bin_size_ns=_to_float(entries.get("SpecBinSize"), 0.128),
        start_mz=_to_float(acq.get("Start Mass (amu)"), 0.0),
        stop_mz=_to_float(acq.get("End Mass (amu)"), 0.0),
        polarity=_parse_polarity(entries, sections),
        n_frames=_to_int(entries.get("NoFrames") or acq.get("Frames"), 1),
        msms_active="Inactive" not in acq.get("MSMS Active", "Inactive"),
        raster_size_um=raster_um,
        raster_size_calibration=raster_calibration,
        declared_tiles=(
            _to_int(mosaic.get("Number Of Tiles X"), 1) or 1,
            _to_int(mosaic.get("Number Of Tiles Y"), 1) or 1,
        ),
        mosaic_active="True" in acq.get("Mosaic Active", "False"),
        calibrants=parse_calibrants(entries.get("Calibration", "")),
    )

    _validate_header(header, path)
    logger.debug(
        "Parsed PHI header: %d x %d px, %s polarity, %d frames, "
        "calibration slope=%s offset=%s",
        header.image_pixels,
        header.image_pixels,
        header.polarity,
        header.n_frames,
        header.mass_slope,
        header.mass_offset,
    )
    return header
