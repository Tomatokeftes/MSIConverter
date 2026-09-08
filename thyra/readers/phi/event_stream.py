# thyra/readers/phi/event_stream.py
"""Block-structured event stream of PHI SmartSoft-TOF ``.raw`` files.

Everything past ``HeaderSize`` is a chain of blocks, each introduced by a
16-byte header::

    offset 0   uint16  block id
    offset 2   10 bytes, not interpreted
    offset 12  uint32  record size

Block ids observed in imaging acquisitions:

===  ===========================================================
id   meaning
===  ===========================================================
1    ion events (8 bytes each); ``8`` instead when MS/MS is on
2    end of frame; carries no payload
6    mosaic tile origin; payload is two uint32 tile indices
14   appended ASCII info block, e.g. a post-hoc recalibration
0    end of stream
===  ===========================================================

Any other id is skipped using its record size. The chain tiles the payload
exactly, which is what makes it safe to trust: a parse that ends anywhere
other than end-of-file has gone wrong somewhere.

Each event is a little-endian uint64::

    bit  31      set on a valid event
    bits 0-30    flight time in picoseconds (27 bits are used)
    bits 32-42   X pixel index
    bits 43-53   Y pixel index
    bits 58-63   clear on a valid event

Records that fail the validity test are dropped. The reference dataset
carries 64,493 of them among 18.5 million (0.35%), many holding the
``0xADADADAD`` uninitialised-heap fill; excluding them reproduces the
vendor's own exported ion images exactly.
"""

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ...errors import ConversionRefused
from .phi_header import PhiHeader

logger = logging.getLogger(__name__)

BLOCK_HEADER_SIZE = 16
BLOCK_END = 0
BLOCK_FRAME_END = 2
BLOCK_TILE = 6
BLOCK_ASCII_INFO = 14

EVENT_SIZE = 8
_VALID_HIGH_SHIFT = 58
_BIT31 = np.uint64(1 << 31)
_TOF_MASK = np.uint64(0x7FFFFFFF)
_COORD_MASK = np.uint64(0x7FF)
_X_SHIFT = np.uint64(32)
_Y_SHIFT = np.uint64(43)
_HIGH_SHIFT = np.uint64(_VALID_HIGH_SHIFT)

#: Appended ASCII blocks are small; cap the read rather than trusting the size.
_ASCII_INFO_LIMIT = 1 << 16

_HEADER_STRUCT = struct.Struct("<H")
_SIZE_STRUCT = struct.Struct("<I")
_TILE_STRUCT = struct.Struct("<II")


@dataclass
class DataSpan:
    """One run of ion events, with the mosaic tile origin in effect."""

    offset: int
    size: int
    tile_x: int = 0
    tile_y: int = 0

    @property
    def n_records(self) -> int:
        """Number of 8-byte event records in this span."""
        return self.size // EVENT_SIZE


@dataclass
class BlockIndex:
    """Result of walking the block chain without reading event payloads.

    Attributes:
        data_spans: Byte ranges holding ion events, in file order.
        n_frames: Count of end-of-frame blocks seen.
        tiles: Distinct ``(tile_x, tile_y)`` origins seen, in file order.
        appended: Parsed ``AsciiInfoBlock`` payloads, in file order.
        block_counts: Histogram of block ids, for diagnostics.
        end_offset: Byte offset at which the walk stopped.
        clean: Whether the chain terminated exactly at end-of-file.
        pixels_per_tile: Raster edge length, used to turn the tile indices in
            :attr:`tiles` into pixel offsets.
        data_block_id: Block id the events were read from, for diagnostics.
    """

    data_spans: List[DataSpan] = field(default_factory=list)
    n_frames: int = 0
    tiles: List[Tuple[int, int]] = field(default_factory=list)
    appended: List[Dict[str, str]] = field(default_factory=list)
    block_counts: Dict[int, int] = field(default_factory=dict)
    end_offset: int = 0
    clean: bool = True
    pixels_per_tile: int = 0
    data_block_id: int = 0

    @property
    def n_records(self) -> int:
        """Total event records across every span, valid or not."""
        return sum(span.n_records for span in self.data_spans)

    @property
    def tile_grid(self) -> Tuple[int, int]:
        """Mosaic tile counts implied by the origins actually encountered."""
        if not self.tiles:
            return (1, 1)
        return (
            max(t[0] for t in self.tiles) + 1,
            max(t[1] for t in self.tiles) + 1,
        )

    def appended_calibration(self) -> Optional[Tuple[float, float]]:
        """Return ``(slope, offset)`` from the newest appended block, if any.

        A recalibration run after acquisition is appended rather than written
        back into the acquisition header, so when one is present it, not the
        header, describes the flight times actually stored in the file.
        """
        for info in reversed(self.appended):
            if "Mass/Time" in info and "MassOffset" in info:
                try:
                    return (float(info["Mass/Time"]), float(info["MassOffset"]))
                except ValueError:  # pragma: no cover - malformed vendor block
                    continue
        return None


def _parse_ascii_info(payload: bytes) -> Dict[str, str]:
    """Parse an appended ASCII info block into key/value pairs."""
    info: Dict[str, str] = {}
    for line in payload.decode("latin-1", errors="replace").split("\r\n"):
        key, sep, value = line.partition(":")
        if sep and key and not key.startswith("\x00"):
            info[key.strip()] = value.strip()
    return info


def _read_tile_origin(handle, pos: int, index: BlockIndex) -> Optional[Tuple[int, int]]:
    """Read a block 6 tile origin, recording it as newly seen if it is."""
    handle.seek(pos)
    raw = handle.read(_TILE_STRUCT.size)
    if len(raw) < _TILE_STRUCT.size:
        return None
    tile = _TILE_STRUCT.unpack(raw)
    if tile not in index.tiles:
        index.tiles.append(tile)
    return tile


def _read_ascii_info(handle, pos: int, record_size: int, index: BlockIndex) -> None:
    """Record an appended ASCII info block, if it parses to anything."""
    handle.seek(pos)
    info = _parse_ascii_info(handle.read(min(record_size, _ASCII_INFO_LIMIT)))
    if info:
        index.appended.append(info)


def _finalise_index(index: BlockIndex, path: Path, pos: int, file_size: int) -> None:
    """Record where the walk stopped and complain if it stopped early."""
    index.end_offset = pos
    index.clean = pos == file_size
    if not index.clean:
        logger.warning(
            "PHI block chain for %s ended at byte %d of %d; the trailing %d "
            "bytes were not accounted for",
            path.name,
            pos,
            file_size,
            file_size - pos,
        )
    if not index.data_spans:
        raise ConversionRefused(
            f"No event blocks (id {index.data_block_id}) found in {path}. "
            "Is this a PHI imaging acquisition?"
        )
    logger.info(
        "PHI block scan: %d event blocks, %d records, %d frames, %d tile(s)",
        len(index.data_spans),
        index.n_records,
        index.n_frames,
        max(len(index.tiles), 1),
    )


def scan_blocks(path: Path, header: PhiHeader) -> BlockIndex:
    """Walk the block chain, recording structure without reading event data.

    Payloads are seeked over rather than read, so this is cheap even on very
    large files and can be run before deciding how to process the events.

    Args:
        path: Path to the ``.raw`` file.
        header: The parsed acquisition header.

    Returns:
        The block index.

    Raises:
        ValueError: If the block chain is malformed.
    """
    path = Path(path)
    file_size = path.stat().st_size
    data_id = header.data_block_id
    index = BlockIndex(pixels_per_tile=header.image_pixels, data_block_id=data_id)
    tile = (0, 0)

    with path.open("rb") as handle:
        pos = header.header_size
        while pos + BLOCK_HEADER_SIZE <= file_size:
            handle.seek(pos)
            raw = handle.read(BLOCK_HEADER_SIZE)
            if len(raw) < BLOCK_HEADER_SIZE:
                break
            block_id = _HEADER_STRUCT.unpack_from(raw, 0)[0]
            record_size = _SIZE_STRUCT.unpack_from(raw, 12)[0]
            pos += BLOCK_HEADER_SIZE
            index.block_counts[block_id] = index.block_counts.get(block_id, 0) + 1

            if block_id == BLOCK_END:
                break
            if block_id == data_id:
                usable = min(record_size, file_size - pos)
                usable -= usable % EVENT_SIZE
                if usable > 0:
                    index.data_spans.append(DataSpan(pos, usable, *tile))
                pos += record_size
            elif block_id == BLOCK_TILE:
                read = _read_tile_origin(handle, pos, index)
                if read is None:
                    break
                tile = read
                pos += _TILE_STRUCT.size
            elif block_id == BLOCK_FRAME_END:
                index.n_frames += 1
            else:
                if block_id == BLOCK_ASCII_INFO and record_size:
                    _read_ascii_info(handle, pos, record_size, index)
                pos += record_size

    _finalise_index(index, path, pos, file_size)
    return index


def decode_events(
    words: NDArray[np.uint64],
) -> Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int64]]:
    """Split raw uint64 records into ``(x, y, tof_ps)``, dropping bad records."""
    valid = ((words >> _HIGH_SHIFT) == 0) & ((words & _BIT31) != 0)
    good = words[valid]
    return (
        ((good >> _X_SHIFT) & _COORD_MASK).astype(np.int32),
        ((good >> _Y_SHIFT) & _COORD_MASK).astype(np.int32),
        (good & _TOF_MASK).astype(np.int64),
    )


def iter_event_batches(
    path: Path, index: BlockIndex, max_records: int = 4_000_000
) -> Iterator[Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int64]]]:
    """Stream decoded ``(x, y, tof_ps)`` batches, applying mosaic tile offsets.

    Adjacent event blocks sharing a tile origin are coalesced up to
    ``max_records`` so the caller sees few, large arrays rather than one per
    block (the reference dataset has 32,074 event blocks averaging 575
    records each).
    """
    path = Path(path)
    with path.open("rb") as handle:
        pending: List[NDArray[np.uint64]] = []
        pending_tile: Optional[Tuple[int, int]] = None
        pending_count = 0

        def flush():
            nonlocal pending, pending_tile, pending_count
            if not pending:
                return None
            words = pending[0] if len(pending) == 1 else np.concatenate(pending)
            tile = pending_tile or (0, 0)
            pending, pending_tile, pending_count = [], None, 0
            x, y, tof = decode_events(words)
            if tile != (0, 0):
                # Block 6 carries tile indices, not pixel offsets.
                stride = index.pixels_per_tile
                x = x + np.int32(tile[0] * stride)
                y = y + np.int32(tile[1] * stride)
            return x, y, tof

        for span in index.data_spans:
            tile = (span.tile_x, span.tile_y)
            if pending and (tile != pending_tile or pending_count >= max_records):
                batch = flush()
                if batch is not None and batch[2].size:
                    yield batch
            handle.seek(span.offset)
            buf = handle.read(span.size)
            if len(buf) < EVENT_SIZE:
                continue
            usable = len(buf) - (len(buf) % EVENT_SIZE)
            pending.append(np.frombuffer(buf, dtype="<u8", count=usable // EVENT_SIZE))
            pending_tile = tile
            pending_count += usable // EVENT_SIZE

        batch = flush()
        if batch is not None and batch[2].size:
            yield batch
