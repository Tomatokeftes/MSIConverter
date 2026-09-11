# thyra/readers/bruker/solarix/solarix_reader.py
"""Bruker solariX (FT-ICR / MRMS) ``.d`` reader via ``peaks.sqlite``.

ftmsControl writes a processed peak store inside every imaging ``.d``:
``peaks.sqlite`` holds centroided, calibrated per-pixel peak lists together
with stage-raster coordinates and the instrument identity. This reader
consumes that store and nothing else -- the raw transient block (``ser``)
and the proprietary ``.mcf`` containers are never touched, so no vendor
SDK and no FT processing are involved.

Layout expected::

    sample.d/
        peaks.sqlite            <- centroided per-pixel peak lists (read)
        ImagingInfo.xml         <- per-scan index (consistency check only)
        ser                     <- raw transients (ignored)
        *.mcf, *.mcf_idx        <- Bruker containers (ignored)
        *.m/                    <- acquisition method (ignored)
    sample.mis                  <- flexImaging sequence NEXT TO the .d;
                                   the only source of the pixel size

The blob encoding was decode-verified on real acquisitions (2018-2024,
ftmsControl 2.2): ``PeakMzValues`` is little-endian float64 x NumPeaks,
ascending; ``PeakIntensityValues``/``PeakFwhmValues``/``PeakSnrValues`` are
little-endian float32 x NumPeaks. ``XIndexPos``/``YIndexPos`` are absolute
stage-raster indices, so the reader offsets them to 0-based coordinates and
keeps the originals in the metadata.
"""

import logging
import sqlite3
import xml.etree.ElementTree as ET  # nosec B405 - trusted local instrument files
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ....core.base_extractor import MetadataExtractor
from ....core.registry import register_reader
from ....errors import ConversionRefused
from ..base_bruker_reader import BrukerBaseMSIReader
from ..mis_parser import parse_mis_file

logger = logging.getLogger(__name__)


def _read_only_uri(path: Path) -> str:
    """The sqlite URI that opens ``path`` strictly read-only.

    A UNC path -- which is what a mapped network drive resolves to on
    Windows -- starts with ``//server/share``; inside a ``file:`` URI that
    reads as an authority, which sqlite rejects ("invalid uri authority").
    Doubling the leading slashes leaves the authority empty and the path
    intact, so ``file:////server/share/...`` opens where
    ``file://server/share/...`` does not. Drive-letter and POSIX paths
    are unaffected.
    """
    posix = path.as_posix()
    if posix.startswith("//"):
        posix = "//" + posix
    return f"file:{quote(posix)}?mode=ro"


@register_reader("solarix")
class SolarixReader(BrukerBaseMSIReader):
    """Reader for Bruker solariX imaging ``.d`` directories.

    Pure Python: stdlib ``sqlite3`` plus numpy. The database is opened
    read-only through a URI so no journal or WAL files are ever created
    next to it -- solariX data frequently lives on network shares.
    """

    def __init__(
        self,
        data_path: Path,
        pixel_size_um: Optional[float] = None,
        metadata_only: bool = False,
        intensity_threshold: Optional[float] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the solariX reader.

        Args:
            data_path: Path to the solariX imaging ``.d`` directory.
            pixel_size_um: Override the pixel size in micrometres. When
                omitted it is read from the ``<Raster>`` element of the
                sibling ``.mis`` file; without that file the pixel size is
                reported as unknown rather than defaulted.
            metadata_only: If True, skip the eager per-spectrum index load
                and the ImagingInfo.xml consistency check; essential and
                comprehensive metadata then come from the ``Properties``
                table and SQL aggregates alone. Used by ``thyra.preview_msi``
                to keep the per-sample preview cheap. Spectrum iteration
                still works in this mode -- the index loads on demand.
            intensity_threshold: Minimum intensity to retain.
            **kwargs: Passed to :class:`BrukerBaseMSIReader`.

        Raises:
            ConversionRefused: Among the layout and schema checks, from
                the sibling ``.mis`` when that file is a document
                defusedxml refuses. Not caught here: without the ``.mis``
                the pixel size is unknown, and reporting it unknown
                because the file was refused -- while saying nothing about
                the file -- is the failure this reader would otherwise
                present.
        """
        super().__init__(data_path, intensity_threshold=intensity_threshold, **kwargs)

        self._pixel_size_override = pixel_size_um
        self._metadata_only = bool(metadata_only)
        self._closed = False

        self._validate_layout()
        self._conn = self._open_peaks_db()

        self._properties: Dict[str, Any] = self._load_properties()
        self._check_schema()
        self._acquisition_keys: Dict[int, Dict[str, Any]] = (
            self._load_acquisition_keys()
        )

        # Per-spectrum index (Id, X, Y, NumPeaks, RegionNumber). Loaded
        # eagerly for a full open, on demand in metadata-only mode.
        self._spectra_index: Optional[Dict[str, NDArray]] = None
        self._summary: Dict[str, int] = {}
        self._acquisition_stats: Optional[Dict[str, Any]] = None
        self._common_mass_axis: Optional[NDArray[np.float64]] = None

        # Pixel size from the sibling .mis (flexImaging sequence).
        self._mis_path: Optional[Path] = self._resolve_mis_path()
        self._mis_metadata: Dict[str, Any] = (
            parse_mis_file(self._mis_path) if self._mis_path else {}
        )

        if self._metadata_only:
            self._summary = self._load_summary_aggregates()
        else:
            self._load_spectra_index()
            self._check_imaging_info()

        logger.info(
            "Initialized SolarixReader: %s, %d spectra, raster X %d-%d, " "Y %d-%d%s",
            self.data_path.name,
            self._summary["n_spectra"],
            self._summary["x_min"],
            self._summary["x_max"],
            self._summary["y_min"],
            self._summary["y_max"],
            " (metadata only)" if self._metadata_only else "",
        )

    # ------------------------------------------------------------------ setup

    def _validate_layout(self) -> None:
        """Refuse anything that is not a solariX imaging .d directory."""
        if not self.data_path.is_dir():
            raise ConversionRefused(
                f"solariX format requires a .d directory, got: {self.data_path}"
            )
        peaks = self.data_path / "peaks.sqlite"
        if not peaks.exists():
            if (self.data_path / "ser").exists() and (
                self.data_path / "ImagingInfo.xml"
            ).exists():
                raise ConversionRefused(
                    f"solariX .d directory without peaks.sqlite: "
                    f"{self.data_path}. The acquisition holds raw transients "
                    "(ser) but no processed peak store, which is what Thyra "
                    "reads. Export the dataset as imzML from the Bruker "
                    "software (DataAnalysis, SCiLS Lab, or flexImaging) and "
                    "convert the imzML file instead."
                )
            raise ConversionRefused(
                f"Not a solariX imaging .d directory (no peaks.sqlite): "
                f"{self.data_path}"
            )
        self._peaks_path = peaks

    def _open_peaks_db(self) -> sqlite3.Connection:
        """Open peaks.sqlite strictly read-only.

        The URI form with ``mode=ro`` guarantees sqlite never creates
        journal/WAL side files next to the database, which matters when the
        data sits on a read-only network share.
        """
        uri = _read_only_uri(self._peaks_path)
        try:
            return sqlite3.connect(uri, uri=True)
        except sqlite3.Error as exc:
            raise ConversionRefused(
                f"Cannot open {self._peaks_path} read-only: {exc}"
            ) from exc

    def _load_properties(self) -> Dict[str, Any]:
        """Load the Properties key/value table verbatim."""
        try:
            rows = self._conn.execute("SELECT Key, Value FROM Properties").fetchall()
        except sqlite3.Error as exc:
            raise ConversionRefused(
                f"peaks.sqlite has no readable Properties table in "
                f"{self.data_path}: {exc}"
            ) from exc
        return {str(key): value for key, value in rows}

    def _check_schema(self) -> None:
        """Verify the schema this reader was written against.

        All probed acquisitions (2018-2024, ftmsControl 2.2) declare
        ``SchemaType = Imaging`` with ``SchemaVersionMajor = 1``. Anything
        else is unverified territory, so fail loudly with the found values
        rather than decoding blobs on an unknown layout.
        """
        schema_type = self._properties.get("SchemaType")
        major = self._properties.get("SchemaVersionMajor")
        if str(schema_type) != "Imaging" or str(major) != "1":
            raise ConversionRefused(
                f"Unsupported peaks.sqlite schema in {self.data_path}: "
                f"SchemaType={schema_type!r}, SchemaVersionMajor={major!r} "
                "(this reader supports SchemaType='Imaging', "
                "SchemaVersionMajor=1). If this is a newer ftmsControl "
                "schema, export the dataset as imzML instead and report the "
                "version so support can be added."
            )

    def _load_acquisition_keys(self) -> Dict[int, Dict[str, Any]]:
        """Load AcquisitionKeys rows, keyed by Id.

        ``ScanMode``/``AcquisitionMode``/``MsLevel`` are stored as the raw
        integers found in the file -- the full enums are not publicly
        documented, so no labels are invented for them. Only ``Polarity``
        has a verified mapping (0=positive, 1=negative).
        """
        try:
            rows = self._conn.execute(
                "SELECT Id, Polarity, ScanMode, AcquisitionMode, MsLevel "
                "FROM AcquisitionKeys"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ConversionRefused(
                f"peaks.sqlite has no readable AcquisitionKeys table in "
                f"{self.data_path}: {exc}"
            ) from exc
        return {
            int(row[0]): {
                "polarity_raw": row[1],
                "scan_mode_raw": row[2],
                "acquisition_mode_raw": row[3],
                "ms_level_raw": row[4],
            }
            for row in rows
        }

    def _load_summary_aggregates(self) -> Dict[str, int]:
        """Cheap dataset summary straight from SQL aggregates."""
        row = self._conn.execute(
            "SELECT COUNT(*), MIN(XIndexPos), MAX(XIndexPos), "
            "MIN(YIndexPos), MAX(YIndexPos), SUM(NumPeaks) FROM Spectra"
        ).fetchone()
        n_spectra = int(row[0] or 0)
        if n_spectra == 0:
            raise ConversionRefused(
                f"peaks.sqlite holds no spectra in {self.data_path}"
            )
        return {
            "n_spectra": n_spectra,
            "x_min": int(row[1]),
            "x_max": int(row[2]),
            "y_min": int(row[3]),
            "y_max": int(row[4]),
            "total_peaks": int(row[5] or 0),
        }

    def _load_spectra_index(self) -> Dict[str, NDArray]:
        """Load the per-spectrum index (no blobs). Idempotent."""
        if self._spectra_index is not None:
            return self._spectra_index

        rows = self._conn.execute(
            "SELECT Id, XIndexPos, YIndexPos, NumPeaks, "
            "COALESCE(RegionNumber, 0) FROM Spectra ORDER BY Id"
        ).fetchall()
        if not rows:
            raise ConversionRefused(
                f"peaks.sqlite holds no spectra in {self.data_path}"
            )

        arr = np.asarray(rows, dtype=np.int64)
        self._spectra_index = {
            "id": arr[:, 0],
            "x": arr[:, 1],
            "y": arr[:, 2],
            "num_peaks": arr[:, 3],
            "region": arr[:, 4],
        }
        self._summary = {
            "n_spectra": int(arr.shape[0]),
            "x_min": int(arr[:, 1].min()),
            "x_max": int(arr[:, 1].max()),
            "y_min": int(arr[:, 2].min()),
            "y_max": int(arr[:, 2].max()),
            "total_peaks": int(arr[:, 3].sum()),
        }
        return self._spectra_index

    def _check_imaging_info(self) -> None:
        """Cross-check the scan count against ImagingInfo.xml.

        The per-scan index is an independent record of the acquisition, so
        a mismatch means the .d was truncated or partially copied. This is
        a warning rather than an error: the peak store itself is still
        internally consistent.
        """
        info_path = self.data_path / "ImagingInfo.xml"
        try:
            root = ET.parse(info_path).getroot()  # nosec B314
            n_scans = sum(1 for _ in root.iter("scan"))
        except (ET.ParseError, OSError) as exc:
            logger.warning("Could not parse %s: %s", info_path, exc)
            return
        if n_scans != self._summary["n_spectra"]:
            logger.warning(
                "ImagingInfo.xml lists %d scans but peaks.sqlite holds %d "
                "spectra in %s -- the .d may be truncated or partially copied",
                n_scans,
                self._summary["n_spectra"],
                self.data_path,
            )

    def _resolve_mis_path(self) -> Optional[Path]:
        """Locate the flexImaging .mis that belongs to this .d.

        The sequence file sits NEXT TO the .d with the same stem. When the
        stems do not match, a lone .mis in the parent directory is accepted
        as unambiguous; multiple non-matching candidates are refused so a
        wrong region's raster cannot be picked up silently.
        """
        parent = self.data_path.parent
        if not parent.exists():
            return None

        matching = parent / f"{self.data_path.stem}.mis"
        if matching.exists():
            return matching

        candidates = sorted(parent.glob("*.mis"))
        if len(candidates) == 1:
            logger.info(
                "No .mis matching stem '%s'; using the only candidate %s",
                self.data_path.stem,
                candidates[0].name,
            )
            return candidates[0]
        if candidates:
            logger.warning(
                "No .mis matching stem '%s' and %d non-matching candidates "
                "in %s -- refusing to guess, pixel size will be unknown",
                self.data_path.stem,
                len(candidates),
                parent,
            )
        return None

    # ------------------------------------------------------------- properties

    @property
    def properties(self) -> Dict[str, Any]:
        """The peaks.sqlite Properties table, verbatim."""
        return self._properties

    @property
    def acquisition_keys(self) -> Dict[int, Dict[str, Any]]:
        """The AcquisitionKeys rows keyed by Id, raw enum values preserved."""
        return self._acquisition_keys

    @property
    def summary(self) -> Dict[str, int]:
        """Dataset summary: n_spectra, raster extents, total peaks."""
        return self._summary

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Grid dimensions ``(x, y, z)`` of the raster bounding box."""
        s = self._summary
        return (s["x_max"] - s["x_min"] + 1, s["y_max"] - s["y_min"] + 1, 1)

    @property
    def coordinate_offsets(self) -> Tuple[int, int, int]:
        """The absolute stage-raster origin subtracted from coordinates."""
        return (self._summary["x_min"], self._summary["y_min"], 0)

    @property
    def mis_path(self) -> Optional[Path]:
        """Path to the flexImaging .mis used for the pixel size, if any."""
        return self._mis_path

    @property
    def mis_metadata(self) -> Dict[str, Any]:
        """Parsed .mis metadata (raster, areas, teaching points)."""
        return self._mis_metadata

    @property
    def pixel_size_um(self) -> Optional[Tuple[float, float]]:
        """Pixel pitch ``(x_um, y_um)``, or ``None`` when unknown.

        The .d itself does not record the raster pitch anywhere -- the
        acquisition method was swept for raster/pixel tokens and carries
        none -- so the only sources are the sibling .mis or the caller.
        """
        if self._pixel_size_override is not None:
            return (float(self._pixel_size_override), float(self._pixel_size_override))
        raster = self._mis_metadata.get("raster")
        if raster and len(raster) == 2:
            return (float(raster[0]), float(raster[1]))
        return None

    @property
    def pixel_size_source(self) -> str:
        """Where :attr:`pixel_size_um` came from, for provenance."""
        if self._pixel_size_override is not None:
            return "user override"
        if self._mis_metadata.get("raster"):
            return f"flexImaging .mis Raster element ({self._mis_path})"
        return "unknown (no .mis with a Raster element found next to the .d)"

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Acquisition m/z range from the Properties table."""
        try:
            return (
                float(self._properties["MzAcqRangeLower"]),
                float(self._properties["MzAcqRangeUpper"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionRefused(
                f"peaks.sqlite Properties carry no usable MzAcqRangeLower/"
                f"MzAcqRangeUpper in {self.data_path}: {exc}"
            ) from exc

    def get_acquisition_stats(self) -> Dict[str, Any]:
        """Per-spectrum acquisition parameter ranges (one SQL aggregate).

        LaserPower, LaserRepRate and NumSummations are recorded per scan;
        they are constant within a normal acquisition, so min == max is the
        expected case and the pair collapses to a scalar downstream.
        """
        if self._acquisition_stats is None:
            row = self._conn.execute(
                "SELECT MIN(LaserPower), MAX(LaserPower), "
                "MIN(LaserRepRate), MAX(LaserRepRate), "
                "MIN(NumSummations), MAX(NumSummations), "
                "MIN(NumPeaks), MAX(NumPeaks) FROM Spectra"
            ).fetchone()
            self._acquisition_stats = {
                "laser_power": (row[0], row[1]),
                "laser_rep_rate": (row[2], row[3]),
                "num_summations": (row[4], row[5]),
                "num_peaks": (row[6], row[7]),
            }
        return self._acquisition_stats

    # --------------------------------------------------------- reader interface

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create the solariX metadata extractor."""
        from ....metadata.extractors.solarix_extractor import SolarixMetadataExtractor

        return SolarixMetadataExtractor(self)

    @property
    def has_shared_mass_axis(self) -> bool:
        """Centroided per-pixel peak lists: every spectrum differs."""
        return False

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Union of every spectrum's m/z values, sorted and deduplicated.

        Only the no-resampling path consumes this; with resampling enabled
        the FT-ICR axis generator builds the axis from the mass range
        instead.
        """
        if self._common_mass_axis is None:
            chunks: List[NDArray[np.float64]] = []
            cursor = self._conn.execute(
                "SELECT Id, NumPeaks, PeakMzValues FROM Spectra ORDER BY Id"
            )
            for spectrum_id, num_peaks, mz_blob in cursor:
                mzs = self._decode_blob(
                    spectrum_id, "PeakMzValues", mz_blob, int(num_peaks), np.float64
                )
                if mzs.size:
                    chunks.append(mzs)
            if not chunks:
                raise ConversionRefused(
                    f"Cannot build a mass axis: no peaks in {self.data_path}"
                )
            self._common_mass_axis = np.unique(np.concatenate(chunks))
        return self._common_mass_axis

    def _decode_blob(
        self,
        spectrum_id: int,
        column: str,
        blob: Optional[bytes],
        num_peaks: int,
        dtype: type,
    ) -> NDArray:
        """Decode one little-endian peak blob, refusing silent truncation.

        The blob length must equal ``NumPeaks * itemsize`` exactly; a
        mismatch means the database is corrupt or the layout drifted, and
        decoding a truncated read would fabricate a spectrum.
        """
        if num_peaks == 0:
            return np.array([], dtype=dtype)
        itemsize = np.dtype(dtype).itemsize
        expected = num_peaks * itemsize
        actual = len(blob) if blob is not None else 0
        if actual != expected:
            raise ConversionRefused(
                f"Corrupt {column} blob in Spectra row Id={spectrum_id} of "
                f"{self._peaks_path}: NumPeaks={num_peaks} implies "
                f"{expected} bytes, found {actual}"
            )
        return np.frombuffer(blob, dtype=np.dtype(dtype).newbyteorder("<")).astype(
            dtype, copy=True
        )

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Iterate spectra in acquisition order.

        Args:
            batch_size: Ignored; present for interface compatibility.

        Yields:
            ``((x, y, z), mzs, intensities)`` with 0-based coordinates
            (the absolute stage-raster origin is subtracted), float64 m/z
            and float64 intensities.
        """
        if self._closed:
            raise RuntimeError("Reader has been closed")

        x_min, y_min, _ = self.coordinate_offsets
        cursor = self._conn.execute(
            "SELECT Id, XIndexPos, YIndexPos, NumPeaks, PeakMzValues, "
            "PeakIntensityValues FROM Spectra ORDER BY Id"
        )

        pbar = tqdm(
            cursor,
            desc="Reading solariX spectra",
            unit=" spectra",
            total=self._summary["n_spectra"],
            dynamic_ncols=True,
            disable=getattr(self, "_quiet_mode", False),
        )
        try:
            for spectrum_id, x, y, num_peaks, mz_blob, intensity_blob in pbar:
                num_peaks = int(num_peaks)
                if num_peaks == 0:
                    continue
                mzs = self._decode_blob(
                    spectrum_id, "PeakMzValues", mz_blob, num_peaks, np.float64
                )
                intensities = self._decode_blob(
                    spectrum_id,
                    "PeakIntensityValues",
                    intensity_blob,
                    num_peaks,
                    np.float32,
                ).astype(np.float64)

                # Stored ascending in every probed file; downstream mapping
                # assumes sorted m/z, so restore the invariant if a file
                # ever violates it rather than mapping garbage.
                if mzs.size > 1 and np.any(np.diff(mzs) < 0):
                    logger.warning(
                        "Unsorted PeakMzValues in Spectra row Id=%d; sorting",
                        spectrum_id,
                    )
                    order = np.argsort(mzs, kind="stable")
                    mzs = mzs[order]
                    intensities = intensities[order]

                mzs, intensities = self._apply_intensity_filter(mzs, intensities)
                if mzs.size == 0:
                    continue

                yield (int(x - x_min), int(y - y_min), 0), mzs, intensities
        finally:
            pbar.close()

    def get_peak_counts_per_pixel(self) -> Optional[NDArray[np.int32]]:
        """Per-pixel peak counts from the NumPeaks column, no blob decode."""
        index = self._load_spectra_index()
        n_x, n_y, _ = self.dimensions
        x_min, y_min, _ = self.coordinate_offsets

        counts = np.zeros(n_x * n_y, dtype=np.int32)
        pixel_idx = (index["y"] - y_min) * n_x + (index["x"] - x_min)
        counts[pixel_idx] = index["num_peaks"]
        return counts

    def get_region_map(self) -> Optional[dict]:
        """Map 0-based ``(x, y)`` to the RegionNumber recorded per scan."""
        index = self._load_spectra_index()
        x_min, y_min, _ = self.coordinate_offsets
        return {
            (int(x - x_min), int(y - y_min)): int(region)
            for x, y, region in zip(index["x"], index["y"], index["region"])
        }

    def get_region_info(self) -> Optional[list]:
        """Summary of acquisition regions found in the Spectra table.

        A plain SQL aggregate rather than the spectra index, so the
        metadata-only preview path stays cheap.
        """
        rows = self._conn.execute(
            "SELECT COALESCE(RegionNumber, 0) AS region, COUNT(*) "
            "FROM Spectra GROUP BY region ORDER BY region"
        ).fetchall()
        return [
            {"region_number": int(region), "n_spectra": int(count)}
            for region, count in rows
        ]

    def close(self) -> None:
        """Close the database connection and drop caches."""
        if self._closed:
            return
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover - best effort
            pass
        self._spectra_index = None
        self._common_mass_axis = None
        self._closed = True
        logger.debug("SolarixReader closed")

    def __repr__(self) -> str:
        """Return a string representation of the reader."""
        return (
            f"SolarixReader(path={self.data_path}, "
            f"n_spectra={self._summary.get('n_spectra', '?')})"
        )
