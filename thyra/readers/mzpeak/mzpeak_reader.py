# thyra/readers/mzpeak/mzpeak_reader.py
"""Experimental reader for mzPeak archives.

mzPeak is a ZIP container of Parquet members plus a JSON index. Every member
is stored uncompressed, so each Parquet file is a contiguous byte range that
pyarrow can read through a plain file object.

Container facts this module relies on, all measured against the reference
implementation (HUPO-PSI/mzPeak @ 502c3a4) rather than inferred from the
draft prose:

* ``mzpeak_index.json`` is ``{"files": [...], "metadata": {...}}``. Each file
  entry carries ``name``, ``entity_type``, ``data_kind`` and -- optionally --
  ``column_mapping`` (alias ``metadata_mapping``) and ``parameters``. Members
  are resolved through this index, never by hardcoded filename.
* The reference parser lowercases and trims both vocabularies and accepts the
  space spelling as an alias for the underscore one, falling through to an
  ``Other`` variant rather than erroring. This reader matches that tolerance:
  an unrecognised role is skipped, not fatal.
* Signal data is one struct column
  ``point: struct<spectrum_index: uint64, mz: double, intensity: float>``,
  one point per row, sorted by ``spectrum_index``. A chunked layout exists
  (top-level ``chunk`` instead of ``point``) and is out of scope.
* There is no spectrum -> row-group map in the container. The KV key
  ``spectrum_array_index`` sounds like one but describes which column holds
  which CV array type. Spectra are located instead from
  ``spectra_metadata.number_of_data_points`` (exact, O(n_spectra)) with
  row-group statistics used to prune which groups to touch.
* File-level metadata lives in the Parquet key-value footer of the metadata
  member *and*, inconsistently, in the index's ``metadata`` object. The
  footer was populated on every reference archive; the index object was empty
  on the only imaging one. The footer wins, the index fills gaps.
* Positions are ``opt_IMS_1000050_position_x`` / ``opt_IMS_1000051_position_y``
  and exist only for imaging acquisitions. Thyra is MSI-only, so an archive
  without them is refused.

The format is a v0.9 draft. Everything here is written to fail loudly with a
named file and a named cause rather than to guess.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

from ...core.base_reader import BaseMSIReader
from ...core.registry import register_reader
from ...errors import ConversionRefused

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...core.base_extractor import MetadataExtractor

logger = logging.getLogger(__name__)

#: Name of the index member. Fixed by the specification; it is the one
#: filename in the container that may be looked up directly, because it is
#: what tells you the name of everything else.
INDEX_MEMBER = "mzpeak_index.json"

#: ZIP local-file-header magic. Used by format detection so a mislabelled
#: ``.mzpeak`` fails detection instead of failing later inside pyarrow.
ZIP_MAGIC = b"PK\x03\x04"

# Imaging CV terms. Resolved by accession rather than by column or parameter
# name: the reference archives spell IMS:1000046 "pixel size (x)" with
# parentheses and IMS:1000047 "pixel size y" without, so name matching finds
# one axis and misses the other.
IMS_POSITION_X = "IMS:1000050"
IMS_POSITION_Y = "IMS:1000051"

#: Conventional column names for the position terms. Used only as a fallback
#: when the index carries no column mapping, which the schema permits.
DEFAULT_POSITION_X_COLUMN = "opt_IMS_1000050_position_x"
DEFAULT_POSITION_Y_COLUMN = "opt_IMS_1000051_position_y"


def _normalise_token(value: Any) -> str:
    """Fold an index vocabulary token the way the reference parser does.

    ``DataKind::from_str`` and ``EntityType::from_str`` both lowercase and
    trim before matching, and both accept a space where the serialised form
    uses an underscore (``"data arrays"`` for ``"data_arrays"``,
    ``"mass spectrum"`` for ``"spectrum"``). Folding the same way here means
    an archive written by any conforming writer resolves identically.

    Args:
        value: Raw token from the index, or anything else.

    Returns:
        The folded token, or ``""`` when the value is not a string.
    """
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace(" ", "_")


def _lazy_pyarrow() -> Tuple[Any, Any]:
    """Import pyarrow on first use.

    pyarrow reaches every Thyra install already: it is a hard dependency of
    spatialdata, which is a hard dependency of Thyra. It is imported lazily
    anyway so that ``import thyra`` -- which imports every reader package to
    trigger registration -- does not pay for a 50 MB extension module that
    only mzPeak archives need.

    Returns:
        The ``pyarrow`` and ``pyarrow.parquet`` modules.

    Raises:
        ImportError: If pyarrow is missing or too old.
    """
    try:
        import pyarrow  # noqa: WPS433 - deliberate lazy import
        import pyarrow.parquet as pq  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - install-shape dependent
        raise ImportError(
            "Reading mzPeak archives requires pyarrow, which is normally "
            "installed as a dependency of spatialdata. Install it with "
            "`pip install pyarrow>=20`."
        ) from exc

    major = int(str(pyarrow.__version__).split(".")[0])
    if major < 20:
        raise ImportError(
            f"Reading mzPeak archives requires pyarrow >= 20, found "
            f"{pyarrow.__version__}. Earlier versions mis-handle the "
            "large_list and struct columns these archives use."
        )
    return pyarrow, pq


class MzPeakSpatialIndex(NamedTuple):
    """Everything about the archive that is one value per spectrum.

    Small even for a very large archive -- one row per spectrum, not per
    point -- so it is read once and kept, which is what lets the reader cut a
    row group into spectra without consulting the file again.

    Attributes:
        spectrum_indices: ``spectrum_index`` of each positioned spectrum,
            ascending.
        coordinates: ``(n, 2)`` array of 0-based ``(x, y)`` pixel coordinates.
        raw_positions: ``(n, 2)`` array of the positions as written, before
            normalisation. Kept because coordinate bounds are reported in the
            file's own frame.
        point_counts: Number of data points in each spectrum.
        offsets: The ``(x, y)`` minima subtracted to reach 0-based
            coordinates.
    """

    spectrum_indices: NDArray[np.int64]
    coordinates: NDArray[np.int64]
    raw_positions: NDArray[np.int64]
    point_counts: NDArray[np.int64]
    offsets: Tuple[int, int]


class MzPeakArchive:
    """Resolved view over one ``.mzpeak`` container.

    Owns the :class:`zipfile.ZipFile` handle and the parsed index, and hands
    out Parquet members by role. Shared by the reader and the metadata
    extractor so the archive is opened and validated exactly once.
    """

    def __init__(self, path: Path):
        """Open an archive and resolve its members.

        Args:
            path: Path to the ``.mzpeak`` file.

        Raises:
            ValueError: If the file is not a ZIP, carries no index, or the
                index is unreadable.
        """
        self.path = Path(path)
        self._zip: Optional[zipfile.ZipFile] = None
        self._parquet_cache: Dict[str, Any] = {}
        self._file_metadata: Optional[Dict[str, Any]] = None
        self._spatial_index: Optional[MzPeakSpatialIndex] = None
        self._null_count: Optional[int] = None
        self._null_count_cached = False

        if not zipfile.is_zipfile(self.path):
            raise ConversionRefused(
                f"Not an mzPeak archive (not a ZIP container): {self.path}"
            )

        self._zip = zipfile.ZipFile(self.path)
        self._members = set(self._zip.namelist())
        if INDEX_MEMBER not in self._members:
            raise ConversionRefused(
                f"Not an mzPeak archive (no {INDEX_MEMBER} member): {self.path}"
            )

        try:
            index = json.loads(self._zip.read(INDEX_MEMBER))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConversionRefused(
                f"Malformed {INDEX_MEMBER} in {self.path}: {exc}"
            ) from exc
        if not isinstance(index, dict):
            raise ConversionRefused(
                f"Malformed {INDEX_MEMBER} in {self.path}: expected a JSON "
                f"object, found {type(index).__name__}"
            )

        self.index: Dict[str, Any] = index
        self._roles = self._resolve_roles(index)

    def _resolve_roles(self, index: Dict[str, Any]) -> Dict[Tuple[str, str], dict]:
        """Map ``(entity_type, data_kind)`` to the index entry for that member.

        Entries whose member is missing from the ZIP are dropped with a
        warning rather than raising: the index is allowed to describe more
        than a given writer emitted, and only the members this reader
        actually needs are worth failing over.

        Args:
            index: The parsed index document.

        Returns:
            Mapping from folded role pair to the raw index entry.
        """
        roles: Dict[Tuple[str, str], dict] = {}
        entries = index.get("files")
        if not isinstance(entries, list):
            raise ConversionRefused(
                f"Malformed {INDEX_MEMBER} in {self.path}: 'files' must be a "
                f"list, found {type(entries).__name__}"
            )

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or name not in self._members:
                logger.warning(
                    "mzPeak index of %s lists member %r which is not in the "
                    "archive; ignoring it",
                    self.path.name,
                    name,
                )
                continue
            key = (
                _normalise_token(entry.get("entity_type")),
                _normalise_token(entry.get("data_kind")),
            )
            roles.setdefault(key, entry)
        return roles

    def entry(self, entity_type: str, data_kind: str) -> Optional[dict]:
        """Return the index entry for a role, or ``None`` when absent."""
        return self._roles.get(
            (_normalise_token(entity_type), _normalise_token(data_kind))
        )

    def require_entry(self, entity_type: str, data_kind: str) -> dict:
        """Return the index entry for a role, raising when it is absent.

        Raises:
            ValueError: If no member fills the role.
        """
        found = self.entry(entity_type, data_kind)
        if found is None:
            available = sorted(f"{e}/{k}" for e, k in self._roles)
            raise ConversionRefused(
                f"mzPeak archive {self.path} has no "
                f"'{entity_type}/{data_kind}' member. Present roles: "
                f"{', '.join(available) or 'none'}"
            )
        return found

    def parquet(self, entity_type: str, data_kind: str) -> Any:
        """Open the Parquet member filling a role.

        The handle is cached: members are STORED, so pyarrow reads the
        footer once and then seeks within the archive without re-inflating.
        """
        entry = self.require_entry(entity_type, data_kind)
        name = entry["name"]
        if name not in self._parquet_cache:
            _, pq = _lazy_pyarrow()
            assert self._zip is not None
            self._parquet_cache[name] = pq.ParquetFile(self._zip.open(name))
        return self._parquet_cache[name]

    @staticmethod
    def column_for_accession(entry: dict, accession: str) -> Optional[str]:
        """Find the column bound to a CV accession in an index entry.

        ``column_mapping`` carries the binding, but the reference struct
        declares it ``serde(default)`` with the alias ``metadata_mapping``,
        so both spellings occur and both may be absent entirely.

        Args:
            entry: An index file entry.
            accession: The CV accession to look for, e.g. ``"IMS:1000050"``.

        Returns:
            The column path, or ``None`` when the entry binds no such term.
        """
        mapping = entry.get("column_mapping")
        if not isinstance(mapping, list):
            mapping = entry.get("metadata_mapping")
        if not isinstance(mapping, list):
            return None
        for binding in mapping:
            if not isinstance(binding, dict):
                continue
            if binding.get("accession") == accession:
                path = binding.get("path")
                if isinstance(path, str) and path:
                    return path
        return None

    def layout(self) -> str:
        """Return the physical layout of the signal member.

        Returns:
            ``"point"`` or ``"chunk"``.

        Raises:
            ValueError: If the member carries neither top-level column.
        """
        fields = list(self.parquet("spectrum", "data_arrays").schema_arrow.names)
        if "point" in fields:
            return "point"
        if "chunk" in fields:
            return "chunk"
        raise ConversionRefused(
            f"{self.path} has an unrecognised mzPeak data layout: expected a "
            f"top-level 'point' or 'chunk' column, found {fields}."
        )

    def position_columns(self) -> Optional[Tuple[str, str]]:
        """Resolve the scan columns holding the imaging positions.

        Preference is the CV binding in the index, because the column name is
        a convention while the accession is the contract. Falls back to the
        conventional names when the entry carries no mapping, which the
        reference struct permits.

        Returns:
            ``(x_column, y_column)``, or ``None`` for a non-imaging archive.
        """
        entry = self.entry("spectrum", "scans")
        if entry is None:
            return None
        names = set(self.parquet("spectrum", "scans").schema_arrow.names)
        x_column = (
            self.column_for_accession(entry, IMS_POSITION_X)
            or DEFAULT_POSITION_X_COLUMN
        )
        y_column = (
            self.column_for_accession(entry, IMS_POSITION_Y)
            or DEFAULT_POSITION_Y_COLUMN
        )
        if x_column not in names or y_column not in names:
            return None
        return (x_column, y_column)

    def spatial_index(self) -> MzPeakSpatialIndex:
        """Read positions and per-spectrum point counts.

        Scans join to spectra on ``source_index``, not on row order: the
        reference schema numbers scans within a spectrum separately
        (``scan_index``) and nothing promises the two members are ordered
        alike.

        Raises:
            ValueError: If the archive is not an imaging acquisition, or
                carries no positioned spectra.
        """
        if self._spatial_index is not None:
            return self._spatial_index

        columns = self.position_columns()
        if columns is None:
            raise ConversionRefused(
                f"{self.path} is not an imaging mzPeak archive: its scans "
                f"member declares no {IMS_POSITION_X}/{IMS_POSITION_Y} "
                f"position columns. Thyra converts imaging acquisitions only."
            )
        x_column, y_column = columns

        scans = self.parquet("spectrum", "scans").read(
            columns=["source_index", x_column, y_column]
        )
        source = np.asarray(scans.column("source_index").to_numpy(), dtype=np.int64)
        xs = np.asarray(scans.column(x_column).to_numpy(), dtype=np.int64)
        ys = np.asarray(scans.column(y_column).to_numpy(), dtype=np.int64)

        # One spectrum may own several scans; imaging acquisitions write one.
        # Keeping the first per spectrum means a multi-scan file resolves to a
        # single pixel rather than silently overwriting itself.
        _, first = np.unique(source, return_index=True)
        source, xs, ys = source[first], xs[first], ys[first]

        metadata = self.parquet("spectrum", "metadata").read(
            columns=["index", "number_of_data_points"]
        )
        spectrum_index = np.asarray(metadata.column("index").to_numpy(), dtype=np.int64)
        counts = np.asarray(
            metadata.column("number_of_data_points").to_numpy(), dtype=np.int64
        )
        order = np.argsort(spectrum_index, kind="stable")
        spectrum_index, counts = spectrum_index[order], counts[order]

        # A spectrum with no scan row has no pixel, so it is dropped rather
        # than placed at the origin.
        lookup = {int(s): i for i, s in enumerate(source)}
        keep = np.array([int(s) in lookup for s in spectrum_index], dtype=bool)
        if not keep.all():
            logger.warning(
                "%d of %d spectra in %s have no scan row and therefore no "
                "position; they are skipped",
                int((~keep).sum()),
                keep.size,
                self.path.name,
            )
        spectrum_index, counts = spectrum_index[keep], counts[keep]
        if spectrum_index.size == 0:
            raise ConversionRefused(f"{self.path} contains no positioned spectra.")

        rows = np.array([lookup[int(s)] for s in spectrum_index], dtype=np.int64)
        raw = np.stack([xs[rows], ys[rows]], axis=1)

        # Positions are 1-based in the reference archives, but the format
        # promises nothing, so normalise on the observed minimum rather than
        # subtracting a constant.
        offsets = (int(raw[:, 0].min()), int(raw[:, 1].min()))
        coordinates = raw - np.array(offsets, dtype=np.int64)

        self._spatial_index = MzPeakSpatialIndex(
            spectrum_indices=spectrum_index,
            coordinates=coordinates,
            raw_positions=raw,
            point_counts=counts,
            offsets=offsets,
        )
        return self._spatial_index

    def null_count(self) -> Optional[int]:
        """How many rows carry a null m/z, or ``None`` when unknowable.

        Read from Parquet column statistics, so it costs a footer lookup
        rather than a pass over the data.

        mzPeak compresses profile spectra by dropping interior runs of zero
        intensity and marking each gap with a *null pair*: two adjacent rows
        whose m/z **and** intensity are both null. On the reference imaging
        archive that is 13,286 of 36,856 rows, in 512 pairs per spectrum.

        The reference reader regenerates the missing m/z from the
        per-spectrum ``mz_delta_model`` polynomial plus a locally estimated
        median spacing. Those regenerated values are extrapolations rather
        than the instrument's own numbers, and every one of them carries
        zero intensity.

        Thyra writes a sparse matrix, in which a zero-intensity point
        occupies no storage and contributes nothing to any downstream
        analysis. Reconstructing the pairs would therefore only add
        approximate channels to the common mass axis that can never hold a
        value, so the reader drops them and reports the count instead.
        """
        if self._null_count_cached:
            return self._null_count
        self._null_count_cached = True

        metadata = self.parquet("spectrum", "data_arrays").metadata
        total = 0
        seen = False
        for group in range(metadata.num_row_groups):
            row_group = metadata.row_group(group)
            for column in range(row_group.num_columns):
                chunk = row_group.column(column)
                if not chunk.path_in_schema.endswith(".mz"):
                    continue
                statistics = chunk.statistics
                if statistics is None or statistics.null_count is None:
                    self._null_count = None
                    return None
                seen = True
                total += int(statistics.null_count)
        self._null_count = total if seen else None
        return self._null_count

    def file_level_metadata(self) -> Dict[str, Any]:
        """Merge the two places file-level metadata is written.

        The Parquet key-value footer of the metadata member was populated on
        every reference archive. The index's ``metadata`` object was
        populated on two of three -- and empty on the only imaging one, which
        is exactly the file whose pixel size matters. Reading either alone
        loses information, so the footer is taken as authoritative and the
        index fills keys the footer lacks (notably ``version``, which appears
        only in the index).

        Returns:
            Mapping of metadata key to its decoded JSON value.
        """
        if self._file_metadata is not None:
            return self._file_metadata

        merged: Dict[str, Any] = {}

        index_metadata = self.index.get("metadata")
        if isinstance(index_metadata, dict):
            merged.update(index_metadata)

        footer = self.parquet("spectrum", "metadata").metadata.metadata or {}
        for raw_key, raw_value in footer.items():
            key = raw_key.decode("utf-8", "replace")
            if key == "ARROW:schema":
                continue
            try:
                merged[key] = json.loads(raw_value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Scalars such as spectrum_count are written as bare text.
                merged[key] = raw_value.decode("utf-8", "replace")

        self._file_metadata = merged
        return merged

    def close(self) -> None:
        """Release the ZIP handle and any cached Parquet readers."""
        self._parquet_cache.clear()
        if self._zip is not None:
            self._zip.close()
            self._zip = None


@register_reader("mzpeak")
class MzPeakReader(BaseMSIReader):
    """Experimental reader for mzPeak (HUPO-PSI) imaging archives.

    Experimental because the container is a v0.9 draft: column names, the
    index vocabulary and the placement of file-level metadata have all moved
    between prototype revisions, and are expected to move again before v1.0.
    The reader is written against the reference implementation at 502c3a4 and
    validates what it depends on, so a drifted archive produces a named error
    rather than silently wrong pixels.

    Data is processed-imzML-shaped: per-spectrum axes, no shared-axis
    concept anywhere in the format. :attr:`has_shared_mass_axis` is therefore
    always ``False`` and the resampling decision tree treats these files
    exactly as it treats processed imzML.
    """

    def __init__(
        self,
        data_path: Path,
        intensity_threshold: Optional[float] = None,
        **kwargs: object,
    ):
        """Initialise the reader.

        Args:
            data_path: Path to the ``.mzpeak`` archive.
            intensity_threshold: Minimum intensity to keep; see
                :class:`~thyra.core.base_reader.BaseMSIReader`.
            **kwargs: Accepted and ignored, for signature parity with the
                other readers.
        """
        super().__init__(data_path, intensity_threshold=intensity_threshold, **kwargs)
        self._archive: Optional[MzPeakArchive] = None
        self._coordinates: Optional[NDArray[np.int64]] = None
        self._point_counts: Optional[NDArray[np.int64]] = None
        self._spectrum_indices: Optional[NDArray[np.int64]] = None
        self._offsets: Optional[Tuple[int, int]] = None
        self._common_axis: Optional[NDArray[np.float64]] = None
        self._announced = False
        self._dropped_points = 0

    # ------------------------------------------------------------------
    # Archive setup
    # ------------------------------------------------------------------

    @property
    def archive(self) -> MzPeakArchive:
        """The open archive, opened and validated on first access."""
        if self._archive is None:
            self._archive = MzPeakArchive(self.data_path)
            self._validate_layout()
            self._load_spatial_index()
            if not self._announced:
                version = self._archive.file_level_metadata().get("version", "unknown")
                logger.warning(
                    "mzPeak support is EXPERIMENTAL: %s declares container "
                    "version %s, a draft format. Verify converted output "
                    "before relying on it.",
                    self.data_path.name,
                    version,
                )
                self._announced = True
        return self._archive

    def _validate_layout(self) -> None:
        """Refuse layouts this reader cannot honestly read.

        Two cases are refused with a named cause rather than allowed to fail
        somewhere deeper: the chunked encoding, which is a different physical
        layout entirely, and any archive whose scans carry no positions,
        which is a non-imaging acquisition that Thyra has nothing to do with.

        Raises:
            NotImplementedError: If the archive uses the chunked layout.
            ValueError: If the archive carries no spatial positions.
        """
        assert self._archive is not None
        if self._archive.layout() == "chunk":
            raise NotImplementedError(
                f"{self.data_path} uses the mzPeak chunked layout (top-level "
                f"'chunk' column). Thyra reads only the point layout "
                f"(top-level 'point' column). Re-export the archive without "
                f"chunking to convert it."
            )

    def _load_spatial_index(self) -> None:
        """Cache the archive's per-spectrum positions and point counts.

        The archive owns the resolution so the metadata extractor sees the
        same pixels the iteration does; this only keeps the arrays where the
        hot loop can reach them without a dict lookup.
        """
        assert self._archive is not None
        index = self._archive.spatial_index()
        self._spectrum_indices = index.spectrum_indices
        self._coordinates = index.coordinates
        self._point_counts = index.point_counts
        self._offsets = index.offsets

    # ------------------------------------------------------------------
    # BaseMSIReader contract
    # ------------------------------------------------------------------

    def _create_metadata_extractor(self) -> "MetadataExtractor":
        """Create the mzPeak metadata extractor."""
        from ...metadata.extractors.mzpeak_extractor import MzPeakMetadataExtractor

        return MzPeakMetadataExtractor(self.archive, self.data_path)

    @property
    def has_shared_mass_axis(self) -> bool:
        """Always ``False``.

        mzPeak stores one m/z per point with no shared-axis or
        calibration-model mechanism anywhere in the format, spec or reference
        implementation. Claiming otherwise would make the converter read the
        first spectrum's axis and apply it to every pixel.
        """
        return False

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """Build the union of every m/z value in the archive.

        Accumulated row group by row group rather than over the whole column
        at once: the m/z column is the bulk of the archive (~80% on real
        vendor data, where per-frame calibration makes almost every value
        unique), so materialising it entirely would cost more memory than the
        conversion that follows.
        """
        if self._common_axis is not None:
            return self._common_axis

        data = self.archive.parquet("spectrum", "data_arrays")
        axis = np.empty(0, dtype=np.float64)
        for group in range(data.metadata.num_row_groups):
            table = data.read_row_group(group, columns=["point"])
            mzs = self._point_field(table, "mz")
            # Null-pair padding carries no intensity; excluded so the
            # axis holds only channels that can actually take a value.
            mzs = mzs[~np.isnan(mzs)]
            axis = np.union1d(axis, np.unique(mzs))

        if axis.size == 0:
            raise ConversionRefused(
                f"{self.data_path} yielded no m/z values; the archive has no "
                f"usable signal data."
            )
        self._common_axis = axis.astype(np.float64, copy=False)
        return self._common_axis

    @staticmethod
    def _point_field(table: Any, field: str) -> NDArray[Any]:
        """Pull one child out of the ``point`` struct column as numpy."""
        column = table.column("point").combine_chunks()
        # ChunkedArray.combine_chunks() yields a ChunkedArray of one chunk on
        # some pyarrow versions and a StructArray on others; normalise.
        if hasattr(column, "num_chunks"):
            column = column.chunk(0)
        return column.field(field).to_numpy(zero_copy_only=False)

    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """Yield every positioned spectrum in ``spectrum_index`` order.

        Reads one row group at a time and cuts it into spectra in memory, so
        the cost is one pass over the archive regardless of spectrum count.
        Row groups split spectra at their boundaries, so a spectrum's points
        are carried across iterations and emitted only once the next
        ``spectrum_index`` appears -- which is why the emit happens on
        transition rather than per row group.

        Args:
            batch_size: Accepted for interface parity and ignored. Row groups
                already define the read granularity; a second batching layer
                on top would only fragment them.

        Yields:
            ``((x, y, z), mzs, intensities)`` with 0-based coordinates,
            ``z`` always 0, and both arrays float64.
        """
        if batch_size is not None:
            logger.debug(
                "mzPeak reads are row-group sized; ignoring batch_size=%s",
                batch_size,
            )

        data = self.archive.parquet("spectrum", "data_arrays")
        positioned = {
            int(s): i for i, s in enumerate(self._require(self._spectrum_indices))
        }

        pending_index: Optional[int] = None
        pending_mz: List[NDArray[Any]] = []
        pending_intensity: List[NDArray[Any]] = []

        for group in range(data.metadata.num_row_groups):
            table = data.read_row_group(group, columns=["point"])
            if table.num_rows == 0:
                continue
            indices = self._point_field(table, "spectrum_index").astype(
                np.int64, copy=False
            )
            mzs = self._point_field(table, "mz")
            intensities = self._point_field(table, "intensity")

            # Cut at every change of spectrum_index. The column is sorted, so
            # a change is a boundary and nothing needs grouping.
            cuts = np.flatnonzero(np.diff(indices)) + 1
            for start, stop in zip(
                np.concatenate(([0], cuts)), np.concatenate((cuts, [indices.size]))
            ):
                index = int(indices[start])
                if pending_index is not None and index != pending_index:
                    emitted = self._emit(
                        pending_index, pending_mz, pending_intensity, positioned
                    )
                    if emitted is not None:
                        yield emitted
                    pending_mz, pending_intensity = [], []
                pending_index = index
                pending_mz.append(mzs[start:stop])
                pending_intensity.append(intensities[start:stop])

        if pending_index is not None:
            emitted = self._emit(
                pending_index, pending_mz, pending_intensity, positioned
            )
            if emitted is not None:
                yield emitted

        if self._dropped_points:
            logger.info(
                "Dropped %d null-pair padding points from %s; they carry no "
                "intensity and exist only to mark removed zero runs",
                self._dropped_points,
                self.data_path.name,
            )

    def _emit(
        self,
        spectrum_index: int,
        mz_parts: List[NDArray[Any]],
        intensity_parts: List[NDArray[Any]],
        positioned: Dict[int, int],
    ) -> Optional[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]]
    ]:
        """Assemble one spectrum's carried parts into a yieldable tuple.

        Returns ``None`` for a spectrum with no position, with no points left
        after intensity filtering, or with an empty payload -- all three are
        ordinary rather than exceptional, and the converter's sparse grid
        handles the resulting gaps.
        """
        row = positioned.get(spectrum_index)
        if row is None:
            return None

        mzs = np.concatenate(mz_parts).astype(np.float64, copy=False)
        intensities = np.concatenate(intensity_parts).astype(np.float64, copy=False)

        # Drop null-pair padding before anything else sees it. Both columns
        # are null on those rows, so a surviving NaN would propagate into the
        # mass axis and into every intensity statistic downstream. See
        # MzPeakArchive.null_count() for why filling them is the wrong repair.
        valid = ~np.isnan(mzs)
        if not valid.all():
            self._dropped_points += int((~valid).sum())
            mzs = mzs[valid]
            intensities = intensities[valid]

        mzs, intensities = self._apply_intensity_filter(mzs, intensities)
        if mzs.size == 0:
            return None

        coordinates = self._require(self._coordinates)
        return (
            (int(coordinates[row, 0]), int(coordinates[row, 1]), 0),
            mzs,
            intensities,
        )

    @staticmethod
    def _require(value: Optional[NDArray[Any]]) -> NDArray[Any]:
        """Assert that the spatial index has been loaded."""
        if value is None:
            raise RuntimeError("mzPeak spatial index not loaded; access .archive first")
        return value

    @property
    def n_spectra(self) -> int:
        """Number of positioned spectra in the archive."""
        _ = self.archive
        return int(self._require(self._spectrum_indices).size)

    @property
    def mass_range(self) -> Tuple[float, float]:
        """Observed m/z range, from the per-spectrum metadata columns."""
        return self.get_essential_metadata().mass_range

    def get_total_peak_count(self) -> int:
        """Points carrying a value across all positioned spectra.

        Excludes null-pair padding; see :meth:`MzPeakArchive.null_count`.
        """
        return self.get_essential_metadata().total_peaks

    def get_peak_counts_per_pixel(self) -> Optional[NDArray[np.int32]]:
        """Per-pixel point counts, for single-pass streaming conversion.

        Delegated to the extractor, which is the one place that knows whether
        the archive's recorded counts include null-pair padding. Computing
        them here as well would give the converter a second, higher answer.
        """
        return self.get_essential_metadata().peak_counts_per_pixel

    def get_region_map(self) -> Optional[dict]:
        """Always ``None``.

        mzPeak has no region or ROI concept: there is no column, no CV
        binding and no index field carrying acquisition-region identity. It
        is one of the gaps the working group has been asked to close.
        """
        return None

    def close(self) -> None:
        """Close the archive."""
        if self._archive is not None:
            self._archive.close()
            self._archive = None
