# thyra/readers/phi/phi_reader.py
"""PHI SmartSoft-TOF ``.raw`` MSI reader (ToF-SIMS, nanoTOF instruments).

The file records individual ion arrivals rather than per-pixel spectra, so
the reader aggregates events into sparse per-pixel spectra on the detector's
time-channel grid before handing them to the converter.

Unlike the Waters and Bruker readers this needs no vendor SDK -- the format is
parsed directly, so it works identically on every platform.
"""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ...core.base_extractor import MetadataExtractor
from ...core.base_reader import BaseMSIReader
from ...core.registry import register_reader
from ...errors import ConversionRefused
from .event_stream import BlockIndex, iter_event_batches, scan_blocks
from .mass_axis import PhiMassAxis, build_mass_axis
from .phi_header import PhiHeader, parse_phi_header

logger = logging.getLogger(__name__)


@register_reader("phi")
class PhiReader(BaseMSIReader):
    """Reader for PHI SmartSoft-TOF ``.raw`` ToF-SIMS imaging data.

    The reader:

    1. Parses the ASCII acquisition header
    2. Walks the block chain to index event blocks, frames and mosaic tiles
    3. Adopts any appended recalibration in preference to the header's
    4. Builds a common m/z axis on the detector time-channel grid
    5. Aggregates ion events into sparse per-pixel spectra
    """

    def __init__(
        self,
        data_path: Path,
        pixel_size_um: Optional[float] = None,
        use_appended_calibration: bool = True,
        intensity_threshold: Optional[float] = None,
        metadata_only: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialise a PHI reader.

        Args:
            data_path: Path to the ``.raw`` file.
            pixel_size_um: Override the pixel size in micrometres. When
                omitted it is derived as ``Raster Size (um) / ImagePixels``.
                See :attr:`PhiHeader.pixel_size_um` for the caveat about the
                header's ``Raster Size Calibration`` factor.
            use_appended_calibration: Prefer a recalibration appended to the
                file over the acquisition header's coefficients. The appended
                values describe the flight times actually stored, so this
                should only be disabled to reproduce a legacy conversion.
            intensity_threshold: Minimum intensity to retain.
            metadata_only: Answer metadata from the header and the block
                chain alone, without aggregating the event stream. Used by
                ``preview_msi``, which promises no spectra are decoded and
                was until #240 paying a pass over every 8-byte event to
                learn how many pixels carry one -- 0.16 s for a 16 MB file
                and linear in file size, so a multi-gigabyte SmartSoft
                acquisition previewed as slowly as it converted. A reader
                built this way cannot be used for a conversion: the counts
                it reports are the "not counted" sentinel, not zero.
            **kwargs: Passed to :class:`BaseMSIReader`.
        """
        super().__init__(data_path, intensity_threshold=intensity_threshold, **kwargs)

        if not self.data_path.is_file():
            raise ConversionRefused(
                f"PHI .raw must be a file, not a directory: {self.data_path}. "
                "Waters .raw data is a directory and is handled by WatersReader."
            )

        self._pixel_size_override = pixel_size_um
        self._use_appended_calibration = use_appended_calibration
        self._metadata_only = bool(metadata_only)

        self._header: Optional[PhiHeader] = None
        self._index: Optional[BlockIndex] = None
        self._axis: Optional[PhiMassAxis] = None
        self._dimensions: Optional[Tuple[int, int, int]] = None
        self._calibration_source = "header"

        # Aggregated sparse spectra, built on first use.
        self._pixel_ids: Optional[NDArray[np.int64]] = None
        self._group_starts: Optional[NDArray[np.int64]] = None
        self._group_ends: Optional[NDArray[np.int64]] = None
        self._channels: Optional[NDArray[np.int32]] = None
        self._counts: Optional[NDArray[np.int32]] = None
        self._closed = False

    # ------------------------------------------------------------------ setup

    def _ensure_initialized(self) -> None:
        """Parse the header and index the block chain. Idempotent."""
        if self._axis is not None:
            return
        if self._closed:
            raise RuntimeError("Reader has been closed and cannot be reused")

        header = parse_phi_header(self.data_path)
        index = scan_blocks(self.data_path, header)

        slope, offset = header.mass_slope, header.mass_offset
        if self._use_appended_calibration:
            appended = index.appended_calibration()
            if appended is not None:
                slope, offset = appended
                self._calibration_source = "appended"
                logger.info(
                    "Using appended recalibration (Mass/Time=%s, MassOffset=%s) "
                    "in place of the header values (%s, %s)",
                    slope,
                    offset,
                    header.mass_slope,
                    header.mass_offset,
                )

        self._header = header
        self._index = index
        self._axis = build_mass_axis(
            slope=slope,
            offset=offset,
            start_flight_time_us=header.start_flight_time_us,
            stop_flight_time_us=header.stop_flight_time_us,
            bin_size_ns=header.bin_size_ns,
            start_mz=header.start_mz,
            stop_mz=header.stop_mz,
        )

        tiles_x, tiles_y = index.tile_grid
        self._dimensions = (
            header.image_pixels * tiles_x,
            header.image_pixels * tiles_y,
            1,
        )
        logger.info(
            "Initialised PHI reader: %s, %dx%d px, %d m/z channels, %s calibration",
            self.data_path.name,
            self._dimensions[0],
            self._dimensions[1],
            len(self._axis),
            self._calibration_source,
        )

    @property
    def header(self) -> PhiHeader:
        """The parsed acquisition header."""
        self._ensure_initialized()
        assert self._header is not None
        return self._header

    @property
    def block_index(self) -> BlockIndex:
        """The indexed block chain."""
        self._ensure_initialized()
        assert self._index is not None
        return self._index

    @property
    def mass_axis(self) -> PhiMassAxis:
        """The common m/z axis."""
        self._ensure_initialized()
        assert self._axis is not None
        return self._axis

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Grid dimensions as ``(x, y, z)``; ``z`` is always 1."""
        self._ensure_initialized()
        assert self._dimensions is not None
        return self._dimensions

    @property
    def calibration_source(self) -> str:
        """``"appended"`` if a post-hoc recalibration was used, else ``"header"``."""
        self._ensure_initialized()
        return self._calibration_source

    @property
    def pixel_size_um(self) -> Optional[float]:
        """Pixel edge length in micrometres, or ``None`` if underivable."""
        if self._pixel_size_override is not None:
            return self._pixel_size_override
        return self.header.pixel_size_um

    @property
    def pixel_size_source(self) -> str:
        """Where :attr:`pixel_size_um` came from, for provenance."""
        if self._pixel_size_override is not None:
            return "user override"
        return "Raster Size (um) / ImagePixels"

    # ------------------------------------------------------------ aggregation

    def _aggregate(self) -> None:
        """Bin every ion event into sparse per-pixel spectra.

        Events arrive scattered across frames and blocks, so they are keyed by
        ``pixel * n_channels + channel``, sorted once, then run-length encoded.
        Peak memory is about 8 bytes per recorded event during the sort.
        """
        if self._counts is not None:
            return
        self._ensure_initialized()
        assert self._axis is not None and self._index is not None

        n_x, n_y, _ = self.dimensions
        n_channels = len(self._axis)
        keys = np.empty(self._index.n_records, dtype=np.int64)
        filled = 0
        dropped = 0

        for x, y, tof in iter_event_batches(self.data_path, self._index):
            channel, ok = self._axis.to_channel(tof)
            ok &= (x >= 0) & (x < n_x) & (y >= 0) & (y < n_y)
            n_ok = int(ok.sum())
            dropped += ok.size - n_ok
            if n_ok == 0:
                continue
            pixel = y[ok].astype(np.int64) * n_x + x[ok].astype(np.int64)
            batch = pixel * n_channels + channel[ok]
            if filled + n_ok > keys.size:  # pragma: no cover - defensive
                keys = np.resize(keys, filled + n_ok)
            keys[filled : filled + n_ok] = batch
            filled += n_ok

        keys = keys[:filled]
        if keys.size == 0:
            raise ConversionRefused(
                f"No usable ion events in {self.data_path}. All {dropped} records "
                "fell outside the mass axis or the pixel grid."
            )
        keys.sort()

        change = np.empty(keys.size, dtype=bool)
        change[0] = True
        np.not_equal(keys[1:], keys[:-1], out=change[1:])
        starts = np.nonzero(change)[0]
        unique = keys[starts]
        counts = np.diff(np.append(starts, keys.size))
        del keys, change, starts

        pixel_of = unique // n_channels
        self._channels = (unique - pixel_of * n_channels).astype(np.int32)
        self._counts = counts.astype(np.int32)
        del unique, counts

        pchange = np.empty(pixel_of.size, dtype=bool)
        pchange[0] = True
        np.not_equal(pixel_of[1:], pixel_of[:-1], out=pchange[1:])
        gstarts = np.nonzero(pchange)[0]
        self._pixel_ids = pixel_of[gstarts]
        self._group_starts = gstarts
        self._group_ends = np.append(gstarts[1:], pixel_of.size)

        logger.info(
            "Aggregated %d ion events into %d peaks across %d occupied pixels "
            "(%d records dropped)",
            filled,
            self._channels.size,
            self._pixel_ids.size,
            dropped,
        )

    # --------------------------------------------------------- reader interface

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create the PHI metadata extractor."""
        from ...metadata.extractors.phi_extractor import PhiMetadataExtractor

        self._ensure_initialized()
        return PhiMetadataExtractor(self, skip_event_aggregate=self._metadata_only)

    @property
    def has_shared_mass_axis(self) -> bool:
        """Each pixel stores only its occupied channels, so arrays differ."""
        return False

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Return the m/z axis, derived analytically from the calibration."""
        return self.mass_axis.mz

    def get_mass_axis_annotations(self) -> Optional[dict]:
        """Store the measured flight time alongside the derived m/z axis.

        The instrument measures time, not mass. Keeping the flight time makes
        the calibration reversible: if the stored coefficients later turn out
        to be wrong, a corrected m/z axis follows from these times alone,
        without re-reading the raw file.
        """
        return {"tof_us": self.mass_axis.tof_us}

    def get_peak_counts_per_pixel(self) -> Optional[NDArray[np.int32]]:
        """Return occupied channels per pixel, for CSR ``indptr`` construction."""
        self._aggregate()
        assert self._pixel_ids is not None
        assert self._group_starts is not None and self._group_ends is not None

        n_x, n_y, n_z = self.dimensions
        peaks = np.zeros(n_x * n_y * n_z, dtype=np.int32)
        peaks[self._pixel_ids] = (self._group_ends - self._group_starts).astype(
            np.int32
        )
        return peaks

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Iterate occupied pixels in raster order.

        Pixels that recorded no ions are skipped entirely.

        Args:
            batch_size: Ignored; present for interface compatibility.

        Yields:
            ``((x, y, z), mzs, intensities)`` with 0-based coordinates.
        """
        self._aggregate()
        assert self._pixel_ids is not None and self._axis is not None
        assert self._group_starts is not None and self._group_ends is not None
        assert self._channels is not None and self._counts is not None

        n_x = self.dimensions[0]
        mz_axis = self._axis.mz

        for pixel, start, end in zip(
            self._pixel_ids, self._group_starts, self._group_ends
        ):
            mzs = mz_axis[self._channels[start:end]]
            intensities = self._counts[start:end].astype(np.float64)
            mzs, intensities = self._apply_intensity_filter(mzs, intensities)
            if mzs.size == 0:
                continue
            yield (int(pixel % n_x), int(pixel // n_x), 0), mzs, intensities

    def reset(self) -> None:
        """Reset for re-iteration; the aggregate is cached so this is a no-op."""

    def close(self) -> None:
        """Release the aggregated arrays."""
        if self._closed:
            return
        self._pixel_ids = None
        self._group_starts = None
        self._group_ends = None
        self._channels = None
        self._counts = None
        self._closed = True

    def __repr__(self) -> str:
        """Return a string representation of the reader."""
        if self._dimensions is None:
            return f"PhiReader(path={self.data_path}, uninitialised)"
        return (
            f"PhiReader(path={self.data_path}, "
            f"grid={self._dimensions[0]}x{self._dimensions[1]}, "
            f"calibration={self._calibration_source})"
        )
